# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a benchmark suite for evaluating PDF-to-Markdown conversion engines. It measures reading order accuracy (NID), table fidelity (TEDS), heading hierarchy preservation (MHS), and extraction speed across 12 parsing engines.

## Commands

### Full Pipeline
```sh
uv run src/run.py                          # Quality benchmark (parse → evaluate → archive → chart)
uv run src/run.py --engine docling         # Single engine (skips if evaluation.json exists)
uv run src/run.py --engine docling --force # Force re-run
```

### CI Mode (used by opendataloader-pdf CI)
```sh
OPENDATALOADER_JAR=/path/to/jar uv run src/run.py --engine opendataloader --check-regression
```

### Individual Stages
```sh
uv run src/pdf_parser.py             # Convert PDFs to Markdown (all engines)
uv run src/evaluator.py              # Evaluate predictions against ground truth
uv run src/generate_benchmark_chart.py  # Generate comparison charts (no engine deps needed)
uv run src/generate_history.py       # Archive evaluation results
```

### Tests
```sh
uv run pytest                        # Run all tests
uv run pytest tests/test_evaluator_table.py  # Single test file
```

## Architecture

### Dependency Strategy
Engine libraries are **optional dependencies** to avoid conflicts. Base deps (apted, matplotlib, rapidfuzz, etc.) are always installed for evaluation/charting. Each engine is a separate optional group:
```sh
uv sync --extra opendataloader   # Install one engine
uv sync --extra docling
uv sync --extra docling-hybrid   # Hybrid with VLM fallback for complex tables
uv sync --extra all-safe         # All permissive-license engines
```
Chart generation works with base deps only (reads evaluation.json files).

### Engine Registry (engine_registry.py)
Uses **lazy imports** via `get_engine_handler()`. Engines not installed are gracefully skipped. `ENGINE_DISPATCH` is a `_LazyDispatch` dict for backward compatibility.

### Adding a New Engine
1. Create `src/pdf_parser_<name>.py` with `to_markdown(document_paths, input_path, output_dir)` function
2. Add to `ENGINES` and `_ENGINE_MODULES` dicts in `engine_registry.py`
3. Add optional dependency group in `pyproject.toml`
4. For speed benchmark: add parser class in `src/speed_benchmark/parsers/<name>.py`

### Pipeline Flow
1. **pdf_parser.py** → dispatches to engine-specific handlers via lazy import
2. **evaluator.py** → runs NID/TEDS/MHS evaluators, produces `evaluation.json`
3. **generate_benchmark_chart.py** → horizontal bar charts from evaluation.json (filtered by ALL_CHART_ENGINES)
4. **run.py** → orchestrates parse → evaluate → history → chart, with skip logic (`--force` to rerun)

### License Tiers
- **Safe** (direct import): opendataloader, docling, docling-hybrid, markitdown, unstructured, edgeparse
- **Data-only** (no code, prediction/ results only): marker (GPL), MinerU (AGPL), PyMuPDF (AGPL), nutrient/PSPDFKit (Commercial)

### Directory Structure
- `pdfs/` — Input PDF corpus (200 documents)
- `ground-truth/markdown/` — Reference structured output
- `prediction/<engine>/markdown/` — Engine outputs
- `prediction/<engine>/evaluation.json` — Evaluation results
- `history/<yymmdd>/` — Archived evaluation snapshots
- `charts/` — Generated benchmark visualizations

## Engine-Specific Guidance

### docling-hybrid Engine
The `docling-hybrid` engine uses intelligent routing based on table count:
- **Pages with ≤2 tables**: Standard Docling parsing (fast, local)
- **Pages with >2 tables**: VLM pipeline with GPT-4o (improved table accuracy)

**Environment Variables:**
- `OPENAI_API_KEY`: Required for VLM path (when pages have >2 tables)
- `DOCLING_HYBRID_MODEL`: Model selection (default: `gpt-4o-mini`, alternatives: `gpt-4o`)
- `DOCLING_HYBRID_TIMEOUT`: API timeout in seconds (default: 60)

**Error Handling:**
- On API failures (rate limits, timeouts): Automatic retry with exponential backoff (max 5 retries)
- After retry exhaustion: Falls back to standard Docling output
- Missing `OPENAI_API_KEY`: Fails fast with clear error message

**Usage:**
```sh
export OPENAI_API_KEY="sk-..."
uv run src/run.py --engine docling-hybrid
```

**API Cost Considerations:**
The hybrid approach minimizes API costs by only using the VLM path for table-heavy pages. For documents with ≤2 tables per page, no API calls are made.
