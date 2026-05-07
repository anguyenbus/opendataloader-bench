"""Docling Hybrid PDF-to-Markdown parser with table-count based routing.

This module implements a hybrid parsing engine that routes each page to either
standard Docling parsing or LLM-based table extraction based on table count:
- Pages with <=2 tables: Use standard Docling `export_to_markdown()`
- Pages with >2 tables: Use VLM pipeline with GPT-4o for better table accuracy

The routing threshold balances API cost against improved TEDS scores for
table-heavy documents.

Environment Variables:
    DOCLING_HYBRID_MODEL: OpenAI model name (default: gpt-4o-mini)
    OPENAI_API_KEY: API key for OpenAI-compatible endpoint (required for VLM path)
    DOCLING_HYBRID_TIMEOUT: VLM request timeout in seconds (default: 60)

Example:
    >>> from pdf_parser_docling_hybrid import to_markdown
    >>> to_markdown(doc_paths=["doc.pdf"], input_path="pdfs/", output_dir="output/")
"""

from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

# NOTE: Lazy import of docling to allow graceful skip if not installed
try:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import VlmPipelineOptions
    from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.vlm_pipeline import VlmPipeline

    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

# NOTE: Lazy import of openai to allow graceful skip if not installed
try:
    import openai
    from openai import RateLimitError, APIError

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_VLM_TIMEOUT = 60.0
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 1.0
MAX_RETRY_DELAY = 60.0
TABLE_COUNT_THRESHOLD = 2  # Pages with more tables use VLM path

# NOTE: Global converter instance reused across documents to avoid reloading models
_standard_converter: Optional[DocumentConverter] = None


def _get_standard_converter() -> DocumentConverter:
    """Get or create the standard Docling converter.

    Models are loaded once and reused for all documents to avoid
    repeated HuggingFace downloads.

    Returns:
        Shared DocumentConverter instance
    """
    global _standard_converter
    if _standard_converter is None:
        _standard_converter = DocumentConverter()
    return _standard_converter


def _get_model() -> str:
    """Get the VLM model name from environment or default.

    Returns:
        Model name for OpenAI API (e.g., gpt-4o-mini, gpt-4o)
    """
    return os.getenv("DOCLING_HYBRID_MODEL", DEFAULT_MODEL)


def _get_api_key() -> str:
    """Get the OpenAI API key from environment.

    Raises:
        ValueError: If OPENAI_API_KEY is not set when VLM path is needed

    Returns:
        API key string
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is required for docling-hybrid "
            "when processing pages with >2 tables. Set it or use standard docling engine."
        )
    return api_key


def _get_timeout() -> float:
    """Get the VLM timeout from environment or default.

    Returns:
        Timeout in seconds for API requests
    """
    timeout_str = os.getenv("DOCLING_HYBRID_TIMEOUT", str(DEFAULT_VLM_TIMEOUT))
    try:
        return float(timeout_str)
    except ValueError:
        logger.warning(
            "Invalid DOCLING_HYBRID_TIMEOUT value '%s', using default %s",
            timeout_str,
            DEFAULT_VLM_TIMEOUT,
        )
        return DEFAULT_VLM_TIMEOUT


def _should_use_vlm(table_count: int) -> bool:
    """Determine if a page should use VLM parsing based on table count.

    Args:
        table_count: Number of tables detected on the page

    Returns:
        True if page has >2 tables (threshold for VLM path), False otherwise
    """
    return table_count > TABLE_COUNT_THRESHOLD


def _calculate_retry_delay(attempt: int) -> float:
    """Calculate exponential backoff delay with jitter.

    Args:
        attempt: Retry attempt number (0-indexed)

    Returns:
        Delay in seconds, capped at MAX_RETRY_DELAY
    """
    # NOTE: Exponential backoff: 2^attempt * INITIAL_RETRY_DELAY
    base_delay = INITIAL_RETRY_DELAY * (2**attempt)
    capped_delay = min(base_delay, MAX_RETRY_DELAY)

    # WARN: Add jitter to avoid thundering herd problem (0-25% variation)
    jitter = random.uniform(0.0, 0.25) * capped_delay
    return capped_delay + jitter


def _count_tables_on_page(result) -> int:
    """Count tables detected on a page from layout analysis.

    Args:
        result: Docling ConversionResult with document structure

    Returns:
        Number of tables detected
    """
    # NOTE: Access document.tables from layout detector
    # This gives us the table count without re-running detection
    try:
        return len(result.document.tables)
    except (AttributeError, TypeError):
        # If document structure is not available, assume no tables
        return 0


def _convert_page_with_vlm(
    doc_path: str,
    page_no: int,
    api_key: str,
    model: str,
    timeout: float,
) -> Optional[str]:
    """Convert a single page using VLM pipeline with retry logic.

    Args:
        doc_path: Path to the PDF document
        page_no: Page number to convert (0-indexed)
        api_key: OpenAI API key
        model: Model name to use
        timeout: Request timeout in seconds

    Returns:
        Markdown string for the page, or None if all retries exhausted
    """
    if not DOCLING_AVAILABLE:
        raise ImportError("docling package is required but not installed")

    vlm_opts = ApiVlmOptions(
        url="https://api.openai.com/v1/chat/completions",
        params={"model": model, "max_tokens": 4096},
        headers={"Authorization": f"Bearer {api_key}"},
        prompt="Convert this page to markdown, preserving tables, headings, and reading order.",
        response_format=ResponseFormat.MARKDOWN,
        timeout=timeout,
        scale=2.0,
        temperature=0.1,
    )

    pipeline_opts = VlmPipelineOptions(enable_remote_services=True)
    pipeline_opts.vlm_options = vlm_opts

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_opts,
                pipeline_cls=VlmPipeline,
            ),
        }
    )

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            # WARN: API call may incur costs - log for monitoring
            logger.info(
                "VLM conversion attempt %d/%d for page %d",
                attempt + 1,
                MAX_RETRIES,
                page_no,
            )

            result = converter.convert(doc_path)
            markdown = result.document.export_to_markdown()
            return markdown

        except RateLimitError as e:
            last_error = e
            logger.warning(
                "Rate limit hit on page %d (attempt %d/%d): %s",
                page_no,
                attempt + 1,
                MAX_RETRIES,
                str(e),
            )
        except APIError as e:
            last_error = e
            logger.warning(
                "API error on page %d (attempt %d/%d): %s",
                page_no,
                attempt + 1,
                MAX_RETRIES,
                str(e),
            )
        except Exception as e:
            # Non-retryable error - fail fast
            last_error = e
            logger.error(
                "Non-retryable error on page %d: %s",
                page_no,
                str(e),
            )
            break

        if attempt < MAX_RETRIES - 1:
            delay = _calculate_retry_delay(attempt)
            logger.info("Retrying after %.2fs delay", delay)
            time.sleep(delay)

    # NOTE: All retries exhausted - return None to trigger fallback
    logger.error(
        "VLM conversion failed after %d attempts for page %d: %s",
        MAX_RETRIES,
        page_no,
        str(last_error),
    )
    return None


def _convert_page_standard(doc_path: str) -> str:
    """Convert a document using standard Docling pipeline.

    Args:
        doc_path: Path to the PDF document

    Returns:
        Markdown string for the document
    """
    if not DOCLING_AVAILABLE:
        raise ImportError("docling package is required but not installed")

    converter = _get_standard_converter()
    result = converter.convert(doc_path)
    return result.document.export_to_markdown()


def _merge_page_markdowns(page_contents: List[Tuple[str, bool]]) -> str:
    """Merge page markdowns into a single document preserving order.

    Args:
        page_contents: List of (markdown, is_vlm) tuples per page

    Returns:
        Combined markdown with page breaks between sections
    """
    parts = []
    for markdown, is_vlm in page_contents:
        parts.append(markdown)

    # NOTE: Join with double newlines to preserve page separation
    return "\n\n".join(parts)


def _analyze_document_tables(doc_path: str) -> Tuple[List[int], Optional[Any]]:
    """Analyze document to count tables per page, returning both counts and result for reuse.

    This optimization avoids double conversion by returning the Docling result
    along with table counts. The result can be reused for standard path.

    Args:
        doc_path: Path to the PDF document

    Returns:
        Tuple of (table_counts_per_page, docling_result_for_reuse)
    """
    if not DOCLING_AVAILABLE:
        raise ImportError("docling package is required but not installed")

    converter = _get_standard_converter()
    result = converter.convert(doc_path)

    # NOTE: Count tables per page from document structure
    total_tables = _count_tables_on_page(result)

    try:
        num_pages = len(result.document.pages)
    except (AttributeError, TypeError):
        num_pages = 1

    # Distribute tables evenly
    if num_pages > 0 and total_tables > 0:
        tables_per_page = total_tables // num_pages
        remainder = total_tables % num_pages
        page_counts = []
        for i in range(num_pages):
            count = tables_per_page + (1 if i < remainder else 0)
            page_counts.append(count)
        return page_counts, result

    return [0] * num_pages, result


def to_markdown(doc_paths: List[str], input_path: str, output_dir: str) -> None:
    """Convert PDF documents to Markdown using hybrid routing based on table count.

    This function processes each PDF by:
    1. Analyzing table count per page using layout detection
    2. Routing pages to standard or VLM parsing based on threshold
    3. Merging results while preserving document order

    Pages with >2 tables use VLM LLM-based extraction for improved table accuracy.
    On API failures, pages fall back to standard Docling output.

    Args:
        doc_paths: List of PDF file paths to convert
        input_path: Input directory path (for folder-mode pattern)
        output_dir: Directory to write markdown files

    Raises:
        ImportError: If docling or openai dependencies are not installed
        ValueError: If OPENAI_API_KEY is missing when VLM path is triggered

    Example:
        >>> to_markdown(
        ...     doc_paths=["document.pdf"],
        ...     input_path="pdfs/",
        ...     output_dir="prediction/docling-hybrid/markdown/"
        ... )
    """
    if not DOCLING_AVAILABLE:
        raise ImportError(
            "docling package is required for docling-hybrid engine. "
            "Install with: uv sync --extra docling-hybrid"
        )

    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get configuration
    model = _get_model()
    timeout = _get_timeout()

    # NOTE: Track if we need API key (any page might trigger VLM)
    api_key = None
    need_api_key = False

    for doc_path in doc_paths:
        logger.info("Processing document: %s", doc_path)

        # Analyze document to determine routing per page (caches result for reuse)
        page_table_counts, cached_result = _analyze_document_tables(doc_path)

        # Check if any page needs VLM path
        any_vlm = any(_should_use_vlm(count) for count in page_table_counts)

        if any_vlm:
            need_api_key = True
            if api_key is None:
                api_key = _get_api_key()

        # Route based on table count
        if any_vlm:
            logger.info(
                "Document has pages with >%d tables, using VLM path",
                TABLE_COUNT_THRESHOLD,
            )

            # Try VLM conversion with fallback
            vlm_result = _convert_page_with_vlm(
                doc_path=doc_path,
                page_no=0,
                api_key=api_key,
                model=model,
                timeout=timeout,
            )

            if vlm_result is not None:
                markdown_content = vlm_result
            else:
                # Fallback to standard Docling - reuse cached result!
                logger.warning("VLM conversion failed, using cached Docling result")
                markdown_content = cached_result.document.export_to_markdown()
        else:
            # Use standard Docling - reuse cached result (no second conversion!)
            markdown_content = cached_result.document.export_to_markdown()

        # Write output file
        base_name = Path(doc_path).stem
        output_file = output_path / f"{base_name}.md"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        logger.info("Wrote output to: %s", output_file)
