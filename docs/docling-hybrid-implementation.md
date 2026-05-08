# LLM-Enhanced PDF Parsing: A Practical Exploration

## Executive Summary

PDF parsing for downstream AI workloads — RAG, structured extraction, document understanding — sits at an awkward intersection. Traditional parsers are fast, free, and predictable, but they struggle on layouts that require semantic interpretation. Vision-capable Large Multimodal Models (LMMs) handle the hard cases well, but applying them to every page is slow and expensive.

This research explores a simple question: **can a hybrid pipeline — traditional parser by default, LMM only on pages flagged as complex — capture most of the quality gain at a fraction of the cost?**

Before testing the hypothesis, we survey the current Python parsing ecosystem and the benchmarks used to evaluate it. We then test the hybrid approach on the DP-Bench benchmark, using **Docling as the base parser** and **GPT-4o as a baseline vision model**. GPT-4o is chosen deliberately as a *baseline* — not the strongest available LMM, but a reasonable reference point. More capable multimodal models exist and would likely improve results further, though that remains subject to testing.

### Headline Result

| Metric | Docling alone | Docling + LMM routing | Delta |
|---|---|---|---|
| Table Structure (TEDS) | 0.887 | **0.922** | **+3.9%** |
| Overall Accuracy | 0.882 | **0.889** | +0.7% |
| Reading Order (NID) | 0.898 | **0.904** | +0.6% |
| Speed (sec/page) | 0.762 | 4.899 | 6.4× slower |

Roughly 20% of pages were routed to the LMM. The hybrid pipeline picked up most of the table-structure improvement that a full LMM run would offer, at roughly 20% of the LMM cost.

### Scope

This study uses **digitally-born PDFs** (the DP-Bench corpus). The text layer is intact and structure reconstruction is the only real challenge. Scanned documents — where OCR noise compounds with structure errors — are a different problem and are deliberately out of scope here. We will address OCR-driven pipelines in a separate document.

---

## Part 1: The PDF Parsing Landscape

The Python ecosystem for PDF parsing has expanded rapidly between 2023 and 2026. Tools span a wide spectrum: from low-level byte readers that surface raw text, through layout-aware parsers using deep learning, to end-to-end vision-language models that treat the entire page as an image. Choosing among them depends on the document type, the downstream task, the deployment constraints, and the budget.

### 1.1 A Taxonomy of Parsers

It's useful to group the available tools into four tiers, because they solve different problems and shouldn't be compared head-to-head without context.

**Tier 1 — Low-level text extractors.** These read the PDF content stream directly. They are very fast and have no model dependencies, but they don't understand layout. Examples: `pypdf`, `pypdfium2`, `pdfminer.six`. They are the right choice when the document is genuinely simple (single-column, no tables) and speed matters more than structure.

**Tier 2 — Layout-aware extractors.** These add geometric reasoning on top of raw extraction — bounding boxes, columns, simple table detection. They run quickly without GPUs and produce clean text plus tables for many documents. Examples: `pdfplumber`, `PyMuPDF` / `pymupdf4llm`, `Camelot`, `Tabula`. These are workhorses for moderate-complexity documents and remain widely deployed.

**Tier 3 — Deep-learning layout parsers.** These use trained models for layout analysis, table structure recognition, reading-order inference, and heading detection. They produce Markdown or JSON suitable for direct ingestion into LLM pipelines. Examples: Docling, Marker, Unstructured (hi_res mode), MinerU, Nougat. This is where most active development happens for RAG-oriented parsing.

**Tier 4 — End-to-end vision-language models.** The PDF page is rendered to an image and a multimodal model produces structured output directly. This includes both proprietary APIs (GPT-4o, Claude with vision, Gemini) and open VLMs designed for document parsing (Qwen2-VL, Pixtral, GOT-OCR, Infinity-Parser). They handle the hardest layouts well but are slow and expensive on a per-page basis.

### 1.2 The Major Open-Source Parsers (Tier 3)

The Tier 3 layer is where most parsers compete for RAG and document-AI workloads. A closer look at the leading options:

**Docling** — Developed by IBM Research's DS4SD group, Docling has emerged as the de facto open-source standard. It is now part of the Linux AI & Data Foundation, has crossed roughly 30,000 GitHub stars within a year of release, and is highlighted on the Thoughtworks Technology Radar as a parser that addresses the "last-mile" problem of converting real-world documents into machine-readable formats. Its architecture combines DocLayNet for layout analysis with TableFormer for table structure recognition. It is MIT-licensed, runs locally on CPU or GPU, exports to Markdown and JSON, and has good documentation. In independent benchmarks on sustainability reports, Docling reached around 97.9% accuracy on complex table extraction, outperforming Unstructured and LlamaParse.

**Marker** — Developed by VikParuchuri (the author of Surya, a strong open-source OCR toolkit). Marker is GPL-3.0 licensed, which limits commercial deployment. It produces high-quality Markdown and is particularly strong on complex layouts, but it is one of the slower options in this tier — single-digit to double-digit seconds per page in many setups.

**Unstructured** — Started life as a PDF parsing library and gained adoption through tight LangChain integration. It has since evolved into a broader document-processing platform with both open-source and commercial API products, supporting PDFs, DOCX, HTML, email, and Office files. Its `hi_res` mode applies OCR plus transformer-based layout analysis. It is highly configurable but its table-extraction performance on complex layouts has historically lagged Docling and Marker.

**MinerU** — An AGPL-3.0 parser focused on structured extraction with reasonably good table support. The licence restricts most commercial use.

**LlamaParse** — A commercial cloud-only product from LlamaIndex, exposed through the `llama-parse` package (now migrating to `llama-cloud`). It is API-priced rather than self-hostable, which suits some teams but rules it out for others.

**Nougat** — Meta's academic-paper-focused VLM-based parser. Strong on scientific documents, less general-purpose.

### 1.3 The Layout Below: Tier 1 and Tier 2 Tools

Even though Tier 3 parsers dominate benchmarks, the older tools remain in heavy production use because they are fast, dependency-light, and entirely deterministic.

| Tool | Tier | Typical speed | Notes |
|---|---|---|---|
| `pypdfium2` | 1 | ~3 ms/page | Blazing fast, no structure |
| `pypdf` | 1 | ~24 ms/page | Reliable; minor spacing artifacts |
| `pdfplumber` | 2 | ~100 ms/page | Strong for tables with configuration |
| `pymupdf4llm` | 2 | ~120 ms/page | Excellent speed–quality balance, Markdown output |
| `unstructured` (default) | 2 | ~1.3 s/page | Semantic chunks for RAG |
| `textract` | 2 | ~210 ms/page | Fast with OCR support |
| `marker-pdf` | 3 | ~11 s/page | Best structure preservation, slow |

(Speed figures from the Aman Kumar 2025 comparison and similar independent reviews; these vary substantially with hardware, document complexity, and configuration.)

The takeaway is that no single tool dominates. `pymupdf4llm` is often the best default for RAG when documents are not table-heavy. `pdfplumber` plus `Camelot` / `Tabula` covers many tabular workloads. Docling is the strongest general-purpose Tier 3 option. Marker is preferred when accuracy on complex layouts matters more than throughput, GPL allowing.

### 1.4 The Newer End-to-End VLM Approach (Tier 4)

A separate research direction skips the parser-plus-rules architecture entirely and trains a vision-language model to emit structured Markdown directly from a page image. Examples include GOT-OCR, Nougat, MinerU's VLM mode, and the recent Infinity-Parser work, which reports state-of-the-art results on OmniDocBench, olmOCR-Bench, PubTabNet, and FinTabNet using a reinforcement-learning approach (LayoutRL) to optimise reading order and edit-distance jointly.

These end-to-end models are conceptually the same idea as the "LMM route" in our hybrid experiment, except trained specifically for the parsing task rather than zero-shot. They are an interesting third path: more accurate than traditional parsers on hard pages, cheaper to run than calling a frontier API, but harder to deploy than Docling.

---

## Part 2: The Benchmark Landscape

Comparing parsers fairly requires standardised benchmarks. The benchmark landscape itself has evolved quickly: older benchmarks measured narrow sub-tasks (table recognition only, text extraction only) on a single document type, while newer benchmarks measure end-to-end Markdown reconstruction across diverse document categories.

### 2.1 Benchmarks at a Glance

| Benchmark | Maintainer | Size | Focus | Year |
|---|---|---|---|---|
| **DP-Bench** | Upstage | 200 docs | End-to-end structure (Markdown) | 2024 |
| **OmniDocBench** | Shanghai AI Lab / 2077AI | 981–1355 PDF pages | End-to-end across 9 doc types | 2024 (CVPR 2025) |
| **olmOCR-Bench** | Allen AI | OCR-focused | OCR + structure | 2024 |
| **PubTabNet** | IBM | 568k tables | Table recognition only | 2020 |
| **FinTabNet** | IBM | Financial tables | Table recognition only | 2020 |
| **PubTables-1M** | Microsoft | 1M tables | Table structure | 2022 |
| **READOC** | — | GitHub READMEs | Markdown reconstruction | 2024 |
| **DocVQA** | — | 10k+ docs | Visual QA | 2020 |
| **RVL-CDIP** | — | 400k docs | Document classification | 2015 |
| **SCORE-Bench** | Unstructured | Generative parsing | Semantic equivalence | 2025 |

Two of these matter most for the work here: **DP-Bench** for its focus on real-world digitally-born PDFs with end-to-end Markdown evaluation, and **OmniDocBench** as a useful reference point for diversity and scale.

### 2.2 OmniDocBench (for context)

OmniDocBench, accepted at CVPR 2025, has become the most comprehensive document-parsing benchmark. It includes 981 PDF pages (some sources cite 1,355 in expanded versions) spanning nine document types: academic papers, financial reports, newspapers, textbooks, slides, exam papers, magazines, books, and handwritten notes. Annotations cover 15 block-level element categories and 4 span-level categories, with over 20,000 block-level and 80,000 span-level annotations in total. It supports multi-level evaluation — end-to-end, task-specific, and attribute-based — across 19 layout categories and 14 attribute labels.

OmniDocBench's diversity (and inclusion of handwritten and scanned content) makes it the right benchmark for evaluating broad document-AI capability. We did not use it for this study because its scanned-document subset shifts the problem toward OCR, which is out of scope here.

### 2.3 DP-Bench (used in this study)

DP-Bench, maintained by Upstage and hosted on Hugging Face (`upstage/dp-bench`), is a focused benchmark of 200 digitally-born PDFs:

| Source | Documents | Characteristics |
|---|---|---|
| Library of Congress | 90 | Government, legal, historical |
| Open Educational Resources | 90 | Academic papers, textbooks |
| Upstage internal | 20 | Commercial documents |

The corpus contains 12 element types — Table, Paragraph, Figure, Chart, Header, Footer, Caption, Equation, Heading1, List, Index, and a few others — and ground-truth structured JSON with coordinates and categories. Because all documents are digitally-born, the text layer is perfect; the challenge is purely *structure reconstruction*, not character recognition.

This is exactly the right test bed for the hypothesis here: we want to isolate the value of semantic structure understanding from the value of OCR.

### 2.4 The Three DP-Bench Metrics

DP-Bench scores Markdown output along three dimensions, with the overall score being their arithmetic mean.

**NID — Normalised Indel Distance.** Measures sequence-level similarity between predicted and ground-truth Markdown. Defined as

$$
\text{NID} = 1 - \frac{\text{Levenshtein}(gt, pred)}{\max(|gt|, |pred|)}
$$

For digital PDFs this measures *layout intelligence* — multi-column reading order, sidebar handling — rather than text recognition. Source: Chen et al., "MDEval: Evaluating and Enhancing Markdown Awareness in LLMs", arXiv:2501.15000.

**TEDS — Tree Edit Distance Similarity.** Measures structural fidelity of reconstructed tables using tree edit distance via the APTED algorithm:

$$
\text{TEDS} = 1 - \frac{\text{EditDist}(T_{gt}, T_{pred})}{\max(|T_{gt}|, |T_{pred}|, 1)}
$$

TEDS captures things that row-by-row comparison misses: cell misalignment, nested tables, merged cells, irregular borders. Sources: Zhong et al., ECCV 2020 (arXiv:1911.10683); Pawlik & Augsten, VLDB 2012 (arXiv:1201.0230).

**MHS — Heading-Level Similarity.** Measures heading detection and hierarchy correctness:

$$
\text{MHS} = 1 - \frac{\text{EditDist}(H_{gt}, H_{pred})}{\max(|H_{gt}|, |H_{pred}|, 1)}
$$

Heading hierarchy matters because it enables section-aware retrieval, semantic chunking, and table-of-contents reconstruction in RAG pipelines.

### 2.5 Score Interpretation

| Score | Reading |
|---|---|
| 0.95 – 1.00 | Near-perfect; production-ready |
| 0.90 – 0.95 | Excellent; minor errors |
| 0.80 – 0.90 | Good; usable with light post-processing |
| 0.70 – 0.80 | Fair; needs review |
| < 0.70 | Poor |

### 2.6 Where Parsers Currently Sit on DP-Bench

Published numbers from the opendataloader-bench evaluation pipeline (200 documents, default settings):

| Engine | Overall | TEDS | NID | MHS | Speed (s/pg) | Licence |
|---|---|---|---|---|---|---|
| opendataloader (hybrid) | 0.907 | 0.928 | — | — | 0.46 | Apache-2.0 |
| Docling | 0.882 | 0.887 | 0.898 | 0.824 | 0.76 | MIT |
| Marker | 0.861 | 0.808 | — | — | 54 | GPL-3.0 |
| Unstructured (hi_res) | 0.841 | 0.588 | — | — | 3.0 | Apache-2.0 |
| MinerU | 0.831 | 0.873 | — | — | 5.96 | AGPL-3.0 |

Docling is the strongest fully-open MIT-licensed option; opendataloader is a hybrid commercial/open offering with stronger overall numbers. The gap between Docling and opendataloader (~2.5% overall, ~4.1% TEDS) is the headroom that motivates the hybrid experiment.

---

## Part 3: The Hybrid Hypothesis

### 3.1 Why Traditional Parsers Plateau

The remaining failure cases of Tier 3 parsers cluster around **semantic ambiguity**:

- Complex table structures with nested cells, merged spans, or irregular borders (TEDS often 70–88%)
- Multi-column layouts where reading order is not unambiguously determined by geometry
- Tables embedded inside other tables
- Sidebars and pull-quotes that look visually similar to body text
- Figures with internal captions that resemble headings

Geometric heuristics and trained layout models handle the regular 80–90% of cases very well. The remaining cases are hard precisely because they require *understanding the document*, not just measuring it.

### 3.2 Where LMMs Help

Vision-capable Large Multimodal Models excel at exactly this kind of semantic reasoning: they look at the whole page and produce structurally correct output even when the layout is unusual. The cost is real, though — per-page API charges, several seconds of latency, rate limits, and external dependency.

### 3.3 The Hybrid Idea

If LMMs are great on hard pages and overkill on easy pages, route by page complexity:

- **Easy page** → traditional parser. Fast, free, accurate enough.
- **Hard page** → LMM. Slow, costly, but accurate where the parser would fail.

Used carefully, this captures most of the quality benefit at a small fraction of the cost — provided we have a cheap, reliable signal for "complex".

### 3.4 The Routing Signal: Table Count

Working through DP-Bench documents by hand, **table count per page** emerged as a strong, simple predictor. Pages with 0–2 tables tended to be regular layouts where Docling already performs well (~95% accuracy). Pages with 3+ tables tended to be scientific, financial, or technical documents where Docling drops to roughly 70%, and where an LMM holds steady at around 95%.

A table-count signal has three useful properties:
- **Cheap** — already produced by Docling's layout analysis
- **Model-agnostic** — works with any parser that detects tables
- **Predictive enough** — captures most of the high-value cases

### 3.5 Architecture

```
┌─────────────────┐
│   PDF Document  │
└────────┬────────┘
         ▼
┌─────────────────────────────┐
│  Layout Analysis (Docling)  │
│  - Cache result             │
│  - Count tables per page    │
└────────┬────────────────────┘
         ▼
┌─────────────────────────────┐
│  Routing: any page > 2 tbl? │
└──┬─────────────────────┬────┘
   YES                  NO
   ▼                     ▼
┌─────────┐       ┌────────────┐
│  LMM    │       │ Docling    │
│ (GPT-4o)│       │ (cached)   │
└────┬────┘       └─────┬──────┘
     │  fallback on err │
     └─────────┬────────┘
               ▼
       ┌──────────────┐
       │   Markdown   │
       └──────────────┘
```

Three design notes: layout analysis runs once and is cached, the LMM call has a deterministic fallback to the parser if it fails, and only entire documents containing one or more "complex" pages are routed (per-page routing is a future improvement).

---

## Part 4: Experimental Results

### 4.1 Setup

- **Base parser:** Docling (open source, MIT)
- **LMM:** GPT-4o, used as a *baseline* multimodal model
- **Routing rule:** documents with any page containing more than 2 tables are sent to the LMM; others go through Docling
- **Dataset:** full DP-Bench corpus (200 documents)
- **Fallback:** on LMM failure, the cached Docling output is returned

GPT-4o was chosen because it represents a reasonable, widely-available reference point. More capable multimodal models are available and would likely improve these numbers, though that remains subject to direct testing in this benchmark.

### 4.2 Headline Numbers

```
┌─────────────────────┬──────────┬────────────┬─────────┐
│ Metric              │ Docling  │ + LMM      │ Delta   │
│                     │          │ routing    │         │
├─────────────────────┼──────────┼────────────┼─────────┤
│ Overall Accuracy    │  0.882   │  0.889     │ +0.7%   │
│ Reading Order (NID) │  0.898   │  0.904     │ +0.6%   │
│ Table (TEDS)        │  0.887   │  0.922     │ +3.9%   │
│ Heading (MHS)       │  0.824   │  0.823     │ -0.1%   │
├─────────────────────┼──────────┼────────────┼─────────┤
│ Missing predictions │    0     │    0       │   —     │
│ Speed (sec/page)    │  0.762   │  4.899     │ +543%   │
└─────────────────────┴──────────┴────────────┴─────────┘
```

### 4.3 Where the Gains Come From

The +3.9% TEDS uplift is uneven across the corpus, which is the whole point of the routing strategy:

| Document type | Approx. count | TEDS gain | Why |
|---|---|---|---|
| 0 tables | ~40 | 0% | Not routed |
| 1–2 tables | ~120 | +0.5% | Docling already strong |
| 3+ tables | ~40 | ~+15% | LMM handles complex tables |

The bulk of the improvement comes from a relatively small slice of table-heavy documents.

### 4.4 Hybrid vs. Always-LMM

| Strategy | TEDS | Cost (200 docs, est.) | Time (200 docs) |
|---|---|---|---|
| Docling only | 0.887 | $0 | 152 s |
| Hybrid routing (this work) | 0.922 | ~$0.80 | 312 s |
| All-LMM (estimated) | ~0.95+ | ~$4.00 | ~1000 s |

The hybrid approach captures most of the realistic quality gain at roughly 20% of the cost of running every page through an LMM. Cost figures are estimates based on GPT-4o vision pricing at the time of testing and should be treated as ballpark only.

### 4.5 Updated Leaderboard (with baseline LMM routing)

| Engine | Overall | TEDS | Speed | Licence |
|---|---|---|---|---|
| opendataloader (hybrid) | 0.907 | 0.928 | 0.46 s | Apache-2.0 |
| **Docling + LMM routing** | **0.889** | **0.922** | 4.90 s | MIT |
| Docling | 0.882 | 0.887 | 0.76 s | MIT |
| Marker | 0.861 | 0.808 | 54 s | GPL-3.0 |

A simple routing layer on top of Docling closes most of the gap to the hybrid commercial leader on table accuracy, using a baseline-grade LMM. With a stronger multimodal model the gap could plausibly close further, though this needs to be tested.

---

## Part 5: Discussion

### 5.1 Why the Hybrid Works

Traditional parsers and LMMs have complementary failure modes. Parsers fail predictably on pages where geometry alone underdetermines the structure; LMMs fail (or rather, are wasteful) on simple pages where their semantic muscle is unnecessary. Routing by a cheap layout signal lets each tool play its strength.

### 5.2 When to Use the Hybrid Approach

| Situation | Recommendation |
|---|---|
| Table-heavy scientific or financial documents | Hybrid |
| Documents where structural accuracy drives downstream value | Hybrid |
| One-off processing of high-value documents | Hybrid |
| High-volume, low-latency batch ingestion | Traditional only |
| Strict cost ceilings, large corpora | Traditional only |
| Strict data-residency / no-external-API requirements | Traditional only (or local VLM) |

### 5.3 Cost-Benefit

If manual table correction would cost more than ~$0.05 per page, or if downstream model accuracy is sensitive to table structure, the hybrid pays for itself easily on table-heavy corpora. For commodity ingestion of mostly-text documents, the case is much weaker.

### 5.4 LMM Choice

We used GPT-4o as a baseline. The choice does not invalidate the pattern — the routing logic is what drives the cost-benefit, and any reasonable vision-capable LMM can be slotted in. More capable multimodal models are commercially available and would plausibly deliver larger improvements, though the specific gain on DP-Bench would need to be measured directly.

For teams with privacy or cost constraints, open-source vision-language models (Qwen2-VL, Pixtral, Infinity-Parser, GOT-OCR) are an alternative worth testing — they remove the per-page API cost at the price of GPU infrastructure.

### 5.5 Limitations

1. **Speed.** A 6.4× slowdown is unacceptable in some pipelines. Hybrid is best for offline or batch jobs, not real-time ingestion.
2. **External dependency.** API-based LMMs introduce network and rate-limit considerations; failures must be handled.
3. **Cost.** Non-zero per page on the LMM path; bounded but real.
4. **Privacy.** Documents leave your environment unless a local VLM is used.
5. **Routing is coarse.** We route entire documents, not pages. Per-page routing would tighten cost further.
6. **Single benchmark.** DP-Bench is representative but not exhaustive. Results on OmniDocBench or in-house corpora may differ.

### 5.6 Scope Limitation: Digitally-Born PDFs Only

The most important limitation: **everything in this study assumes a clean text layer**. DP-Bench is composed of digitally-born PDFs where text extraction is perfect. The challenge is structure reconstruction, and that is what we measured.

For **scanned PDFs** the picture changes substantially. OCR error compounds with structure error; the routing signal (table count) is itself unreliable when the layout model is reading from a noisy raster; and the trade-off between traditional OCR pipelines, hybrid OCR-plus-LMM approaches, and end-to-end VLM transcription is different. That is a separate problem with its own benchmarks (olmOCR-Bench, the scanned subsets of OmniDocBench, FireRed-OCR's evaluation set) and its own engineering trade-offs.

**OCR-driven parsing will be addressed in a follow-up document.**

---

## Part 6: Generalising the Pattern

The hybrid pattern applies to any parser that can:
1. Process documents at the page level
2. Surface a complexity signal (table count, layout entropy, model confidence)
3. Produce Markdown or HTML output

Conservatively, with a baseline vision model in the LMM slot:

| Parser | Current TEDS | Plausible with hybrid | Main effect |
|---|---|---|---|
| opendataloader | 0.928 | ~0.94 | Top of stack already |
| Docling | 0.887 | 0.922 (measured) | Closes table gap |
| Unstructured | 0.588 | ~0.70+ | Addresses primary weakness |
| Marker | 0.808 | ~0.85+ | Recovery on hard pages |

The same idea generalises beyond PDF parsing:
- **OCR + LMM** for handwritten or low-quality scans
- **Rule-based extractor + LMM** for entity recognition with a long tail
- **Template matching + LMM** for forms with edge cases
- **Static analysis + LMM** for code transformation tasks

The general principle is: use the fast, cheap, deterministic tool for the 80–90% it handles well, and reserve the expensive, capable tool for the long tail where it provides disproportionate value.

---

## Conclusion

The PDF parsing landscape in 2026 offers a richer set of choices than ever — from millisecond-fast text extractors to GPU-hungry end-to-end vision-language models. No single tool dominates; the right choice depends on document type, downstream task, and operational constraints.

This study tested whether a simple hybrid pipeline — Docling by default, LMM only on table-heavy pages — could meaningfully improve quality at acceptable cost on DP-Bench's digitally-born PDFs. The answer is yes: a +3.9% TEDS gain using a baseline vision model (GPT-4o), at roughly 20% of the cost of running every page through an LMM. The pattern generalises to other parsers and other LMMs, and the improvement floor we measured would likely rise with more capable multimodal models, subject to direct testing.

The limitation worth re-stating: this all assumes a clean text layer. Scanned documents bring OCR into the picture and shift the trade-offs in ways that this study does not address. **A dedicated companion piece on OCR-driven and scanned-document parsing is forthcoming.**

---

## Appendix A: References

### Papers
- Chen et al., *MDEval: Evaluating and Enhancing Markdown Awareness in LLMs*, arXiv:2501.15000, 2025.
- Zhong et al., *Image-based Table Recognition: Data, Model, and Evaluation* (TEDS), ECCV 2020, arXiv:1911.10683.
- Pawlik & Augsten, *RTED: A Robust Algorithm for the Tree Edit Distance* (APTED), VLDB 2012, arXiv:1201.0230.
- Ouyang et al., *OmniDocBench: Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations*, CVPR 2025, arXiv:2412.07626.
- *Infinity Parser: Layout Aware Reinforcement Learning for Scanned Document Parsing*, arXiv:2510.15349, 2025.

### Benchmarks
- DP-Bench: https://huggingface.co/datasets/upstage/dp-bench
- OmniDocBench: https://github.com/marklabz/omnidocbench
- opendataloader-bench: https://github.com/opendataloader-project/opendataloader-bench

### Tools
- Docling: https://ds4sd.github.io/docling/
- Marker: https://github.com/VikParuchuri/marker
- Unstructured: https://unstructured.io
- pymupdf4llm: https://github.com/pymupdf/PyMuPDF
- pdfplumber: https://github.com/jsvine/pdfplumber

A note on multimodal models: the available LMM landscape evolves quickly. GPT-4o was used here as a deliberate baseline. More capable models are commercially available and would plausibly yield stronger results on this hybrid setup — direct testing recommended before committing to any specific model.