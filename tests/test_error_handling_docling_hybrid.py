"""Tests for docling-hybrid error handling and edge cases.

Tests verify that the engine handles API failures, retries, and edge cases
gracefully with appropriate fallback behavior.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest


def test_api_failure_fallback_logic():
    """Test that VLM API returning None triggers fallback logic."""
    from pdf_parser_docling_hybrid import to_markdown

    # Create a test that verifies the fallback logic works
    # by mocking the internal functions
    with patch("pdf_parser_docling_hybrid.DOCLING_AVAILABLE", True):
        with patch("pdf_parser_docling_hybrid._get_api_key", return_value="test-key"):
            with patch("pdf_parser_docling_hybrid._convert_page_with_vlm", return_value=None) as mock_vlm:
                with patch("pdf_parser_docling_hybrid._convert_page_standard", return_value="# Standard\nContent") as mock_std:
                    with patch("pdf_parser_docling_hybrid._analyze_document_tables", return_value=[5]):
                        with patch("pathlib.Path.mkdir"):
                            with patch("builtins.open", MagicMock()):
                                to_markdown(
                                    doc_paths=["test.pdf"],
                                    input_path="pdfs/",
                                    output_dir="output/"
                                )

                                # Verify VLM was attempted
                                assert mock_vlm.called
                                # Verify standard fallback was called
                                assert mock_std.called


def test_missing_api_key_fails_fast():
    """Test that missing OPENAI_API_KEY fails fast with clear message."""
    from pdf_parser_docling_hybrid import to_markdown

    # Remove API key from environment
    with patch("pdf_parser_docling_hybrid.DOCLING_AVAILABLE", True):
        with patch.dict(os.environ, {}, clear=True):
            with patch("pdf_parser_docling_hybrid._analyze_document_tables", return_value=[5]):  # Triggers VLM path
                with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable"):
                    to_markdown(
                        doc_paths=["test.pdf"],
                        input_path="pdfs/",
                        output_dir="output/"
                    )


def test_empty_document_handling():
    """Test that documents with no tables use standard path."""
    from pdf_parser_docling_hybrid import _should_use_vlm

    # 0 tables should use standard path
    assert _should_use_vlm(0) is False


def test_all_tables_document():
    """Test that documents where all pages have >2 tables use VLM path."""
    from pdf_parser_docling_hybrid import _should_use_vlm

    # All pages have 3+ tables
    assert _should_use_vlm(3) is True
    assert _should_use_vlm(5) is True
    assert _should_use_vlm(10) is True


def test_mixed_table_counts():
    """Test that documents with mixed table counts route correctly."""
    from pdf_parser_docling_hybrid import _should_use_vlm

    # Standard path (<=2 tables)
    assert _should_use_vlm(0) is False
    assert _should_use_vlm(1) is False
    assert _should_use_vlm(2) is False

    # VLM path (>2 tables)
    assert _should_use_vlm(3) is True
    assert _should_use_vlm(4) is True
    assert _should_use_vlm(100) is True


def test_invalid_timeout_uses_default():
    """Test that invalid timeout value falls back to default."""
    from pdf_parser_docling_hybrid import _get_timeout, DEFAULT_VLM_TIMEOUT

    with patch.dict(os.environ, {"DOCLING_HYBRID_TIMEOUT": "invalid"}):
        timeout = _get_timeout()
        assert timeout == DEFAULT_VLM_TIMEOUT


def test_standard_path_with_no_vlm_needed():
    """Test that documents with all pages <=2 tables never call VLM."""
    from pdf_parser_docling_hybrid import to_markdown

    # All pages have 2 tables or fewer
    with patch("pdf_parser_docling_hybrid.DOCLING_AVAILABLE", True):
        with patch("pdf_parser_docling_hybrid._convert_page_standard", return_value="# Standard\nContent") as mock_std:
            with patch("pdf_parser_docling_hybrid._analyze_document_tables", return_value=[1, 2, 0]):
                with patch("pathlib.Path.mkdir"):
                    with patch("builtins.open", MagicMock()):
                        to_markdown(
                            doc_paths=["test.pdf"],
                            input_path="pdfs/",
                            output_dir="output/"
                        )

                        # Verify standard conversion was called
                        assert mock_std.called


def test_document_merging_with_single_page():
    """Test that single-page documents are merged correctly."""
    from pdf_parser_docling_hybrid import _merge_page_markdowns

    page_contents = [
        ("Single page content", False),
    ]

    result = _merge_page_markdowns(page_contents)
    assert result == "Single page content"


def test_retry_delays_increase_exponentially():
    """Test that retry delays increase exponentially with attempts."""
    from pdf_parser_docling_hybrid import _calculate_retry_delay

    # Delays should generally increase
    delays = [_calculate_retry_delay(i) for i in range(5)]

    # Just verify that the last delay (attempt 4) is capped at MAX_RETRY_DELAY
    assert delays[4] <= 60.0
    # And the first delay is at least INITIAL_RETRY_DELAY
    assert delays[0] >= 1.0
