"""Integration tests for docling-hybrid engine.

Tests verify end-to-end functionality including full pipeline execution,
output file generation, and compatibility with the evaluation pipeline.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, Mock, patch

import pytest


def test_full_pipeline_with_standard_path():
    """Test that full pipeline works for documents using standard path."""
    from pdf_parser_docling_hybrid import to_markdown

    with patch("pdf_parser_docling_hybrid.DOCLING_AVAILABLE", True):
        with patch("pdf_parser_docling_hybrid._convert_page_standard", return_value="# Test Document\n\nContent here") as mock_std:
            with patch("pdf_parser_docling_hybrid._analyze_document_tables", return_value=[1, 0]):
                output_dir = "/tmp/test_output"
                with patch("pathlib.Path.mkdir"):
                    with patch("builtins.open", MagicMock()) as mock_open:
                        mock_file = MagicMock()
                        mock_open.return_value.__enter__.return_value = mock_file

                        to_markdown(
                            doc_paths=["test.pdf"],
                            input_path="pdfs/",
                            output_dir=output_dir
                        )

                        # Verify standard conversion was called
                        assert mock_std.called


def test_output_directory_creation():
    """Test that output directory is created with correct structure."""
    from pdf_parser_docling_hybrid import to_markdown
    from pathlib import Path

    with patch("pdf_parser_docling_hybrid.DOCLING_AVAILABLE", True):
        with patch("pdf_parser_docling_hybrid._convert_page_standard", return_value="# Test\n\nContent"):
            with patch("pdf_parser_docling_hybrid._analyze_document_tables", return_value=[0]):
                # Track mkdir calls
                mkdir_calls = []
                original_mkdir = Path.mkdir

                def track_mkdir(self, *args, **kwargs):
                    mkdir_calls.append(str(self))
                    return original_mkdir(self, *args, **kwargs)

                with patch.object(Path, "mkdir", track_mkdir):
                    with patch("builtins.open", MagicMock()):
                        to_markdown(
                            doc_paths=["test.pdf"],
                            input_path="pdfs/",
                            output_dir="/tmp/test_output"
                        )

                        # Verify mkdir was called with output directory
                        assert any("test_output" in call for call in mkdir_calls)


def test_output_file_naming():
    """Test that output files are named correctly."""
    from pdf_parser_docling_hybrid import to_markdown

    with patch("pdf_parser_docling_hybrid.DOCLING_AVAILABLE", True):
        with patch("pdf_parser_docling_hybrid._convert_page_standard", return_value="# Test\n\nContent"):
            with patch("pdf_parser_docling_hybrid._analyze_document_tables", return_value=[0]):
                with patch("pathlib.Path.mkdir"):
                    written_files = []

                    def mock_open_func(path, *args, **kwargs):
                        written_files.append(str(path))
                        mock_file = MagicMock()
                        mock_file.__enter__ = Mock(return_value=mock_file)
                        mock_file.__exit__ = Mock(return_value=False)
                        mock_file.write = Mock()
                        return mock_file

                    with patch("builtins.open", side_effect=mock_open_func):
                        to_markdown(
                            doc_paths=["my-document.pdf"],
                            input_path="pdfs/",
                            output_dir="/tmp/output"
                        )

                        # Verify file was created with correct name
                        assert len(written_files) == 1
                        assert "my-document.md" in written_files[0]


def test_markdown_output_format_validity():
    """Test that markdown output is valid format."""
    from pdf_parser_docling_hybrid import to_markdown

    test_markdown = """# Test Document

## Section 1

Some content here.

## Section 2

More content.
"""

    with patch("pdf_parser_docling_hybrid.DOCLING_AVAILABLE", True):
        with patch("pdf_parser_docling_hybrid._convert_page_standard", return_value=test_markdown):
            with patch("pdf_parser_docling_hybrid._analyze_document_tables", return_value=[0]):
                with patch("pathlib.Path.mkdir"):
                    captured_content = []

                    def mock_open_func(path, *args, **kwargs):
                        mock_file = MagicMock()
                        mock_file.__enter__ = Mock(return_value=mock_file)
                        mock_file.__exit__ = Mock(return_value=False)

                        def capture_write(content):
                            captured_content.append(content)

                        mock_file.write = Mock(side_effect=capture_write)
                        return mock_file

                    with patch("builtins.open", side_effect=mock_open_func):
                        to_markdown(
                            doc_paths=["test.pdf"],
                            input_path="pdfs/",
                            output_dir="/tmp/output"
                        )

                        # Verify markdown content was written
                        assert len(captured_content) == 1
                        assert "# Test Document" in captured_content[0]


def test_multiple_documents_processed():
    """Test that multiple documents are processed sequentially."""
    from pdf_parser_docling_hybrid import to_markdown

    with patch("pdf_parser_docling_hybrid.DOCLING_AVAILABLE", True):
        with patch("pdf_parser_docling_hybrid._convert_page_standard", return_value="# Content"):
            with patch("pdf_parser_docling_hybrid._analyze_document_tables", return_value=[0]):
                with patch("pathlib.Path.mkdir"):
                    processed_files = []

                    def mock_open_func(path, *args, **kwargs):
                        processed_files.append(str(path))
                        mock_file = MagicMock()
                        mock_file.__enter__ = Mock(return_value=mock_file)
                        mock_file.__exit__ = Mock(return_value=False)
                        mock_file.write = Mock()
                        return mock_file

                    with patch("builtins.open", side_effect=mock_open_func):
                        to_markdown(
                            doc_paths=["doc1.pdf", "doc2.pdf", "doc3.pdf"],
                            input_path="pdfs/",
                            output_dir="/tmp/output"
                        )

                        # Verify all files were processed
                        assert len(processed_files) == 3
                        assert "doc1.md" in processed_files[0]
                        assert "doc2.md" in processed_files[1]
                        assert "doc3.md" in processed_files[2]


def test_function_signature_compatibility():
    """Test that to_markdown function has correct signature for registry."""
    from pdf_parser_docling_hybrid import to_markdown
    from inspect import signature

    sig = signature(to_markdown)
    params = list(sig.parameters.keys())

    # Should accept the standard parameters
    assert "doc_paths" in params
    assert "input_path" in params
    assert "output_dir" in params


def test_import_error_on_missing_docling():
    """Test that ImportError is raised when docling is not available."""
    # Create a fresh import with docling unavailable
    import sys
    import importlib

    # Remove and re-import module with DOCLING_AVAILABLE = False
    if "pdf_parser_docling_hybrid" in sys.modules:
        del sys.modules["pdf_parser_docling_hybrid"]

    with patch("pdf_parser_docling_hybrid.DOCLING_AVAILABLE", False):
        # Need to re-import to get the flag set
        from pdf_parser_docling_hybrid import to_markdown

        with pytest.raises(ImportError, match="docling package is required"):
            to_markdown(
                doc_paths=["test.pdf"],
                input_path="pdfs/",
                output_dir="/tmp/output"
            )
