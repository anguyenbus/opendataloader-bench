"""Tests for docling-hybrid PDF parsing engine.

Tests follow TDD principles: write failing test, implement minimal code to pass,
then refactor while keeping tests green. Focus on core user workflows and avoid
exhaustive edge case testing.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest


# Test environment variable configuration
def test_hybrid_model_env_var_default():
    """Test that DOCLING_HYBRID_MODEL defaults to gpt-4o-mini when not set."""
    with patch.dict(os.environ, {}, clear=True):
        # Import after clearing env to get default value
        from pdf_parser_docling_hybrid import DEFAULT_MODEL

        assert DEFAULT_MODEL == "gpt-4o-mini"


def test_hybrid_model_env_var_override(monkeypatch):
    """Test that DOCLING_HYBRID_MODEL can be overridden via environment variable."""
    monkeypatch.setenv("DOCLING_HYBRID_MODEL", "gpt-4o")
    # Re-import to pick up the env var
    import importlib
    import pdf_parser_docling_hybrid
    importlib.reload(pdf_parser_docling_hybrid)

    assert pdf_parser_docling_hybrid._get_model() == "gpt-4o"


def test_table_count_threshold_routing_standard_path():
    """Test that pages with <=2 tables route to standard Docling parsing."""
    from pdf_parser_docling_hybrid import _should_use_vlm

    # 0 tables -> standard
    assert _should_use_vlm(0) is False
    # 1 table -> standard
    assert _should_use_vlm(1) is False
    # 2 tables -> standard
    assert _should_use_vlm(2) is False


def test_table_count_threshold_routing_vlm_path():
    """Test that pages with >2 tables route to VLM LLM-based parsing."""
    from pdf_parser_docling_hybrid import _should_use_vlm

    # 3 tables -> VLM
    assert _should_use_vlm(3) is True
    # 5 tables -> VLM
    assert _should_use_vlm(5) is True
    # 10 tables -> VLM
    assert _should_use_vlm(10) is True


def test_retry_exponential_backoff_calculation():
    """Test that exponential backoff delay increases correctly with attempts."""
    from pdf_parser_docling_hybrid import _calculate_retry_delay

    # Initial attempt (0) should return base delay
    delay_0 = _calculate_retry_delay(0)
    assert delay_0 >= 1.0  # Initial 1s

    # First retry should be ~2s with jitter
    delay_1 = _calculate_retry_delay(1)
    assert 1.5 <= delay_1 <= 2.5  # 2^1 = 2 with jitter

    # Second retry should be ~4s with jitter
    delay_2 = _calculate_retry_delay(2)
    assert 3.0 <= delay_2 <= 5.0  # 2^2 = 4 with jitter

    # Fourth retry should cap at 60s max
    delay_4 = _calculate_retry_delay(4)
    assert delay_4 <= 60.0  # Max delay cap


def test_retry_delay_includes_jitter():
    """Test that retry delays include random jitter to avoid thundering herd."""
    from pdf_parser_docling_hybrid import _calculate_retry_delay

    # Call multiple times and verify we get different values (jitter)
    delays = [_calculate_retry_delay(2) for _ in range(10)]

    # All should be in valid range (base 4 + jitter 0.4-1.2 = 4.4-5.2)
    for delay in delays:
        assert 4.0 <= delay <= 6.0

    # Should have some variation (not all identical)
    assert len(set(delays)) > 1


def test_openai_api_key_required_validation():
    """Test that missing OPENAI_API_KEY fails fast with clear message."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable"):
            from pdf_parser_docling_hybrid import _get_api_key
            _get_api_key()


def test_openai_api_key_from_env(monkeypatch):
    """Test that OPENAI_API_KEY is correctly retrieved from environment."""
    test_key = "sk-test-key-12345"
    monkeypatch.setenv("OPENAI_API_KEY", test_key)

    from pdf_parser_docling_hybrid import _get_api_key
    assert _get_api_key() == test_key


def test_document_merging_preserves_order():
    """Test that page results are merged in document order."""
    from pdf_parser_docling_hybrid import _merge_page_markdowns

    page_contents = [
        ("Page 1 content", False),  # standard path
        ("Page 2 content", True),   # VLM path
        ("Page 3 content", False),  # standard path
    ]

    result = _merge_page_markdowns(page_contents)

    # Verify content is in order
    assert result.startswith("Page 1 content")
    assert "Page 2 content" in result
    assert result.endswith("Page 3 content")

    # Verify page breaks are preserved
    assert result.count("\n\n") >= 2  # At least page breaks between sections


def test_vlm_timeout_default():
    """Test that VLM timeout has a sensible default value."""
    from pdf_parser_docling_hybrid import DEFAULT_VLM_TIMEOUT

    assert DEFAULT_VLM_TIMEOUT == 60.0


def test_vlm_timeout_env_override(monkeypatch):
    """Test that VLM timeout can be configured via environment variable."""
    monkeypatch.setenv("DOCLING_HYBRID_TIMEOUT", "120")
    import importlib
    import pdf_parser_docling_hybrid
    importlib.reload(pdf_parser_docling_hybrid)

    assert pdf_parser_docling_hybrid._get_timeout() == 120.0
