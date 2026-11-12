# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Undiagnosed is a solo research project that reads a patient's own medical documents (lab reports, radiology notes, clinical notes, scans/photos of them) with Google's open-weight medical models and tells the patient what the findings mean *together*: which signals co-occur, how urgent they are, and what to raise with a clinician. The name states the target user: someone holding reports whose combined meaning nobody has explained to them.

The README frames it as "cardiac symptom analysis using Gemma 4", but nothing in the code is cardiac-specific and the code now runs MedGemma 1.5, not Gemma 4. Treat scope (general vs. cardiac vs. rare-disease) as an open decision, see [Recommended trajectory](#recommended-trajectory).

The design is a chain of agents passing JSON forward. Two of three exist as code; neither runs end to end yet.

Work started in Kaggle notebooks (CUDA, Gemma 4 E2B/E4B) and is being ported to local Apple Silicon (MPS, MedGemma 1.5 4B). `notebooks/` holds the Kaggle originals and is ahead of or behind `agents/` in places; `agents/` is the source of truth for the local port. The LICENSE is all-rights-reserved.

## Commands

There is no build, lint, or test tooling. Do not assume `pytest` works.

```bash
# Agent 1 — must be run from inside agents/ (bare sibling imports)
cd agents && ../.venv/bin/python parser.py
```

- `.venv` is Python 3.14.3; `environment.yml` is a `conda env export` of a Python 3.12 `base` env. They are two different environments, and `requirements.txt` (a `pip freeze`) matches neither exactly.
- `requirements.txt` pins `pymupdf` twice with different versions (`2.0` and `1.27.2.2`), so `pip install -r requirements.txt` will fail until one is removed.
- `agents/analyzer.py` has no runnable entry point (`main()` is `pass`).
- `models/model_installer.py` downloads `baidu/Unlimited-OCR` (an OCR experiment, not wired into anything), not MedGemma.

### Environment

- `.env` (gitignored) defines `GEMMA4_E4B_MODEL_ID`, `GEMMA4_E2B_MODEL_ID`, `MEDGEMMA_1_5_4B_IT_ID`, `HUGGINGFACE_MODEL_ACCESS_TOKEN`. **Nothing loads it**: there is no `python-dotenv` call, so these are only visible if exported in the shell. Only the HF token is read by any code.
- Model weights live in `models/medgemma1_5-4b_it/` (gitignored). `parser.py` loads with `local_files_only=True`.
- `datasets/` is DVC-tracked (`datasets.dvc`, ~5 GB / 81,586 files) but **no DVC remote is configured** in `.dvc/config` and the directory is absent locally, so `dvc pull` cannot work as-is.
- `test_documents/` is mostly published journal case reports (Cureus, JCM, JBSR) plus one CBC template. These are not representative of what a patient uploads; do not treat results on them as validation.

## Architecture

```
file ──► Agent 1 parser.py ──► clinical profile JSON ──► Agent 2 analyzer.py ──► signal analysis JSON ──► Agent 3 (unbuilt)
```

**Agent 1 — Document Parser** ([agents/parser.py](agents/parser.py))

- `document_validator()` classifies a path as `raster` or `document` by extension, raising the exceptions in [agents/custom_exceptions.py](agents/custom_exceptions.py).
- `extraction_branching()` classifies each PDF page: more than 150 characters and at least one font → `"text"` (PyMuPDF text); otherwise the page is rendered at 2× and sent as an image (`"vision"`). Output is a `ProcessedDocument` of `ProcessedPage`s, or a `ProcessedImage` for raster uploads.
- `extract_clinical_signals()` does not run a model despite its name. It builds one chat `content` list: the instruction prompt, then every page's text or base64 image in order.
- `run_inference_for_clinical_signal_extraction()` iterates a **directory**, sends each whole document in a single user turn, generates up to 1024 tokens greedily, and passes the output to `validate_clinical_output()`.
- Clinical profile keys: `document_type`, `patient_context`, `lab_findings`, `imaging_findings`, `clinical_notes`, `flagged_signals`, `extraction_confidence`.

**Agent 2 — Clinical Signal Analyser** ([agents/analyzer.py](agents/analyzer.py))

- `run_inference()` does a standard pass over the clinical profile, and when `extraction_confidence == "low"` a second critique pass that re-checks the first output against the profile.
- Signal analysis keys: `individual_signals`, `combination_patterns`, `overall_assessment`, `urgency`, `analysis_confidence`. Validation lives in [agents/helper_functions/medgemma_signal_analysis_object_parser.py](agents/helper_functions/medgemma_signal_analysis_object_parser.py).

**Things that span files and will bite you**

- Two different functions are both named `validate_clinical_output`: one in `parser.py` (clinical profile schema) and one in the helper package (signal analysis schema). `attempt_json_recovery` is also duplicated, in `parser.py` and in `helper_functions/medgemma_response_json_recoverer.py`.
- Imports are mixed: `parser.py` and `analyzer.py` use bare sibling imports (`from custom_exceptions import …`, `from helper_functions… import …`) and so only work with `agents/` as the working directory, while `helper_functions/` uses relative imports. `agents/__init__.py` is empty and there is no `helper_functions/__init__.py`.
- Both agents now target the same model (MedGemma 1.5 4B, which is multimodal), but each loads its own copy: Agent 1 via `AutoModelForMultimodalLM` once, Agent 2 via `AutoModelForCausalLM` on **every** `run_inference()` call.
- Stubs: [agents/matcher.py](agents/matcher.py), [app.py](app.py), `knowledge/` (empty; the testing notebook installed `chromadb`, `sentence-transformers`, `langgraph`, `biopython`, so a retrieval layer was planned).

## Known defects (verified by reading the code, none fixed)

Agent 1, [agents/parser.py](agents/parser.py):

1. `main()` loads `models/medgemma1_4-4b_it`; the directory on disk is `medgemma1_5-4b_it`.
2. `main()` passes a single file path (`../test_documents/jcm-12-05967.pdf`) to a function that calls `.iterdir()` on it → `NotADirectoryError`.
3. `load_model(quantize=True)` uses `BitsAndBytesConfig`, which is CUDA-only. On this Mac only the unquantized bf16 path can work.
4. `torch.cuda.empty_cache()` is a no-op on MPS (`torch.mps.empty_cache()` is the equivalent).
5. Missing top-level keys are filled with `None`, including list fields, so downstream code iterating `lab_findings` can hit `None`.
6. `max_new_tokens=1024` truncates any report with more than a handful of lab rows; the recovery function then silently drops the cut-off entries.
7. A whole multi-page document goes into one prompt, with the instructions in the user turn rather than a system turn.
8. One exception in one file aborts the whole directory run (validation errors are not caught in the loop).

Agent 2, [agents/analyzer.py](agents/analyzer.py):

1. The low-confidence branch references `validated_signal_analysis_object`, which is never assigned (the variable is `validated_signal_response_object`) → `NameError`.
2. `HTTPError` and `GatedRepoError` are used in `except` clauses but never imported; `except AttributeError or ValueError` only catches `AttributeError`.
3. The `system_prompt` and `follow_up_prompt` parameters are ignored; the module-level prompt strings are used directly.
4. `device_map="auto"` plus an optional bitsandbytes config are CUDA-era settings that were not ported to MPS the way `parser.py` was.
5. In the helper validator, an unterminated thinking block (`<unused94>` with no `<unused95>`, which is what truncation produces) keeps the text *before* the thinking block, which is normally empty. Code fences are only stripped if the response starts with one. `parser.py`'s brace-slicing approach is more robust.
6. The critique pass is gated on the extractor's self-reported confidence and is itself an LLM judging an LLM with no deterministic check.

## Landscape (researched November 2026)

**The generic "explain my lab report" product is now a commodity.** OpenAI launched ChatGPT Health in January 2026 with medical-record connections and lab comparison over time; Anthropic announced a similar consumer product days later; Quest, Labcorp and Hims & Hers all shipped AI lab-result explainers in 2026 ([HLTH roundup](https://hlth.com/insights/articles/january-2026-healthcare-roundup-the-llm-will-see-you-now-2026-02-03), [HIT Consultant](https://hitconsultant.net/2026/01/07/openai-debuts-chatgpt-health/), [Fierce](https://www.fiercehealthcare.com/medtech/labcorp-joins-lab-results-personalisation-push-mylabcorp-ai-app)). A project that only restates a single report in plain language competes with these head-on.

**Where they are weak, per current evidence:**

- Privacy: records uploaded to a consumer chatbot leave HIPAA protection ([KFF Health News](https://kffhealthnews.org/news/article/electronic-medical-records-patients-ai-chatbots-diagnosis-privacy-accuracy/)).
- Accuracy: a 2026 JMIR study of 100 CBC reports found hallucinations and reasoning errors across frontier models; another concluded clinician review is still needed before patient-facing use ([JMIR e87280](https://www.jmir.org/2026/1/e87280/PDF), [JMIR e87802](https://jmir.org/2026/1/e87802)). Summaries sometimes over- or under-emphasise findings.
- Cross-document reasoning for hard cases: the strongest published result is DeepRare (Nature, Feb 2026), a multi-agent rare-disease system with 40+ tools and evidence-linked reasoning chains, motivated by a diagnostic odyssey that often exceeds five years ([PubMed](https://pubmed.ncbi.nlm.nih.gov/41708847/)). It is clinician-facing; no patient-facing equivalent turned up.

**Relevant technical state of the art:**

- MedGemma 1.5 4B was trained specifically to turn scanned and digital lab reports into structured JSON; Google reports 78% macro F1 on an internal lab-extraction benchmark and calls it a developer model that needs validation per use case ([Google Research blog](https://research.google/blog/next-generation-medical-image-interpretation-with-medgemma-15-and-medical-speech-to-text-with-medasr/), [technical report](https://arxiv.org/abs/2604.05081)). That 78% is the ceiling to plan around, not a number to trust blindly.
- Sub-1B OCR VLMs (GLM-OCR, PaddleOCR-VL-1.5) self-report ~94 on OmniDocBench v1.5; table structure and reading order remain the weak spots ([OmniDocBench](https://arxiv.org/abs/2412.07626)). None is benchmarked on lab reports.
- Published verification-oriented medical agent designs converge on the same pattern: retrieve evidence, generate, then verify each claim against evidence with a separate gate ([MedMMV](https://arxiv.org/html/2509.24314v1), [arXiv 2602.14158](https://arxiv.org/abs/2602.14158)).
- Lab standardisation is treated as a layer after extraction: keep test names/values/units verbatim, then map to LOINC/UCUM ([LabBridge, JMIR e92499](https://www.jmir.org/2026/1/e92499/PDF); schema-constrained FHIR generation, [arXiv 2601.05847](https://arxiv.org/abs/2601.05847)).
- Quantized MedGemma runs on Apple Silicon through MLX; `mlx`, `mlx-lm`, `mlx-vlm` and `llguidance` are already in `requirements.txt` but unused.

**Regulatory:** FDA's January 2026 CDS and General Wellness guidances are clinician-facing by default and do not carve out patient-facing LLM tools ([Faegre Drinker](https://faegredrinker.com/en/insights/publications/2026/1/key-updates-in-fdas-2026-general-wellness-and-clinical-decision-support-software-guidance), [Arnold & Porter](https://www.arnoldporter.com/en/perspectives/advisories/2026/01/fda-cuts-red-tape-on-clinical-decision-support-software)). Outputs like `urgency: urgent` and `suggested_investigation` are the parts of the current schema closest to device territory. These are law-firm summaries, not legal advice.

## Recommended trajectory

**Positioning.** Do not build another single-report explainer. Build the thing incumbents structurally cannot: a **local-only, multi-document pattern finder that produces a clinician-visit brief with every claim traceable to a page in the patient's own documents**. Local inference answers the privacy objection; multi-document and longitudinal reasoning is the actual "undiagnosed" problem; traceability answers the hallucination objection; a visit brief ("questions to ask, findings to point at") instead of a diagnosis keeps the output on the literacy side of the regulatory line. Pick one clinical vertical first (cardiac, per the README, is reasonable) so evaluation is tractable.

**Engineering order.** Each step is worth doing only after the one before it.

1. **Make it run end to end.** Fix the defects above, load one MedGemma instance and share it between agents, add an orchestrator entry point (file → profile → analysis), load `.env`, make `agents/` a real package with one import style, reconcile the three dependency files into one.
2. **Build an evaluation set before improving anything.** 30–50 realistic patient documents (synthetic or de-identified lab reports, radiology reports, phone photos) with hand-labelled gold profiles. Score field-level precision/recall for lab rows, and a hallucination rate defined as extracted values that do not appear in the source text. Without this, no later change can be shown to help.
3. **Harden extraction (Agent 1).**
   - Extract per page or per table chunk and merge, rather than one 1024-token call for the whole document.
   - Use grammar-constrained decoding against the schema (`llguidance` is already installed) so repair code becomes a fallback, not the main path. Define the schemas once as Pydantic models shared by both agents.
   - Record provenance on every finding: page number and the verbatim source string.
   - Compute `status` (low/high/critical) in code from the parsed value and reference range; do not let the model decide arithmetic.
   - Replace self-reported `extraction_confidence` with measured signals: share of vision pages, parse repairs needed, values that failed the appears-in-source check.
   - Test MedGemma-direct against an OCR-VLM-to-markdown front end on the eval set and keep whichever wins.
4. **Ground the analysis (Agent 2).**
   - Verify in code, always, not only on low confidence: every `signals_involved` entry must exist in the profile; every number quoted must match.
   - Add retrieval over a curated corpus in `knowledge/` (reference ranges, guideline excerpts, MedlinePlus-style explanations; HPO/Orphanet if the rare-disease direction is chosen) and require each combination pattern to cite a retrieved passage.
   - Set an urgency floor with deterministic critical-value rules; the model may raise urgency but never lower it below the rule.
   - Use greedy decoding for reproducibility (currently `do_sample=True`).
5. **Add the longitudinal layer.** A local store of profiles keyed by document date, with trend detection across reports (the natural job for the empty `matcher.py`: link documents to a timeline and to each other). This is the main differentiator and should not wait for a UI.
6. **Agent 3 and the app.** Agent 3 turns the verified analysis into the visit brief with page citations and plain-language explanations; `app.py` becomes a local Streamlit front end (already in `requirements.txt`). Move inference to MLX 4-bit if memory or latency on MPS is a problem.

**Alternative trajectories, if the recommended one is not the goal:**

- *Rare-disease assistant:* extract phenotypes as HPO terms and rank candidate conditions against Orphanet, in the DeepRare mould. Highest ceiling, but needs clinical collaborators and is clearly decision support.
- *Extraction component only:* a local lab-report → FHIR `Observation` (LOINC/UCUM) converter. Smallest scope, easiest to evaluate, useful to other tools, but not a patient product.
