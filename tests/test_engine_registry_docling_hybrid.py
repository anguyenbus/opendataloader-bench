"""Tests for docling-hybrid engine registration.

Tests verify that the engine is properly registered in the registry and can be
discovered and invoked via the lazy import mechanism.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_docling_hybrid_in_engines_dict():
    """Test that docling-hybrid appears in the ENGINES dict."""
    from engine_registry import ENGINES

    assert "docling-hybrid" in ENGINES
    assert ENGINES["docling-hybrid"] == "2.84.0"  # Should match docling version


def test_docling_hybrid_in_engine_modules():
    """Test that docling-hybrid is mapped to its module in _ENGINE_MODULES."""
    from engine_registry import _ENGINE_MODULES

    assert "docling-hybrid" in _ENGINE_MODULES
    assert _ENGINE_MODULES["docling-hybrid"] == "pdf_parser_docling_hybrid"


def test_docling_hybrid_lazy_import():
    """Test that docling-hybrid handler can be lazily imported."""
    from engine_registry import get_engine_handler

    handler = get_engine_handler("docling-hybrid")

    # Handler should be a callable if docling is installed
    if handler is not None:
        assert callable(handler)
    else:
        # Handler is None when dependencies are not installed
        # This is expected behavior for graceful skip
        pass


def test_docling_hybrid_handler_signature():
    """Test that the handler function has the correct signature."""
    from engine_registry import get_engine_handler
    from inspect import signature

    handler = get_engine_handler("docling-hybrid")

    if handler is not None:
        sig = signature(handler)
        # Should accept doc_paths, input_path, output_dir
        params = list(sig.parameters.keys())
        assert "doc_paths" in params
        assert "input_path" in params
        assert "output_dir" in params


def test_docling_hybrid_graceful_skip_when_not_installed():
    """Test that missing dependencies are handled gracefully."""
    # We need to patch the actual import within the get_engine_handler function
    # Since the function does internal importlib.import_module, we mock that

    import importlib
    original_import = importlib.import_module

    def mock_import_module(name, *args, **kwargs):
        if "pdf_parser_docling_hybrid" in name:
            raise ImportError("docling not installed")
        return original_import(name, *args, **kwargs)

    with patch.object(importlib, "import_module", side_effect=mock_import_module):
        from engine_registry import get_engine_handler

        handler = get_engine_handler("docling-hybrid")
        # Should return None instead of raising
        assert handler is None


def test_docling_hybrid_version_consistency():
    """Test that docling-hybrid version matches docling base version."""
    from engine_registry import ENGINES

    docling_version = ENGINES.get("docling")
    docling_hybrid_version = ENGINES.get("docling-hybrid")

    # Versions should match since docling-hybrid uses docling
    assert docling_version == docling_hybrid_version


def test_docling_hybrid_in_all_chart_engines():
    """Test that docling-hybrid appears in ALL_CHART_ENGINES for charting."""
    from engine_registry import ALL_CHART_ENGINES

    assert "docling-hybrid" in ALL_CHART_ENGINES
