# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Undiagnosed is a research project for analyzing medical documents (lab reports, radiology notes, clinical notes, images) with Google's Gemma models, aimed at helping patients understand cardiac/clinical signals in their own reports. The pipeline is designed as a chain of agents, each handling one stage: parsing documents into structured clinical data, then reasoning over that data to find clinically meaningful patterns.

Development happens primarily in Kaggle notebooks (see commit messages like "Kaggle Notebook | ... | Version N"); working code is then ported into the `agents/` scripts. The two root-level `.ipynb` files and `notebooks/undiagnosed-gemma4-testing-v1.ipynb` are the notebook counterparts/precursors of the code in `agents/`.

## Environment & dependencies

- No `requirements.txt` content and no test/build tooling exist yet (`requirements.txt` is empty, there is no `pyproject.toml`/`Makefile`/CI config, and no test suite). Treat any commands like `pytest` or a package manager install as unverified until such tooling is added.
- `environment.yml` is a full conda-forge + pip lockfile (looks exported via `conda env export`), not a hand-maintained dependency list. Key runtime deps pinned there: `transformers`, `accelerate`, `bitsandbytes`, `torch`, `pymupdf` (imported as `fitz`), `pillow`, `dvc` (+ `dvc-gdrive`).
- Model weights are expected at Kaggle input paths by default (e.g. `/kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b-it/1`), overridable via env vars `GEMMA4_E2B_MODEL_ID` / `GEMMA4_E4B_MODEL_ID` (see [agents/parser.py](agents/parser.py)). Hugging Face auth uses `HUGGINGFACE_MODEL_ACCESS_TOKEN` (see [agents/analyzer.py](agents/analyzer.py)).
- Secrets live in a local `.env` (gitignored). Dataset files (`/datasets`) are also gitignored and tracked instead via DVC (`datasets.dvc`, ~5GB / 81k files), with a Google Drive remote (`dvc-gdrive` is in the deps).

## Architecture: multi-agent pipeline

The pipeline is a sequence of independent agents that pass structured JSON forward. Each agent's system prompt strictly defines the JSON schema it must emit.

**Agent 1 — Document Parser** ([agents/parser.py](agents/parser.py))
- `document_validator()` → `normalize_file_path()` / `get_file_extension()` / `classify_file_type()` validate an uploaded file against two supported categories: raster images (`raster_formats`) and documents (`file_formats`: `.pdf`, `.txt`).
- `extraction_branching()` routes a validated file to the right extraction path. For PDFs, this is done **per page**: each page is heuristically classified as text-extractable or scanned/image-based by comparing font count, XObject count, character count, and image-dominant area (see the docstring inside `extraction_branching()` for the exact decision rule), producing a `ProcessedDocument` of `ProcessedPage` entries (`extraction_method` is `"text"` or `"vision"`). Raster uploads become a single `ProcessedImage` with base64-encoded bytes.
- `extract_clinical_signals()` builds the Gemma multimodal prompt (`content_parts`) by interleaving text and base64 images per page, then `run_inference_for_clinical_signal_extraction()` runs `Gemma4ForConditionalGeneration` over a directory of files and calls `validate_clinical_output()` on each raw response to parse/repair the model's JSON and fill in missing required keys (`document_type`, `patient_context`, `lab_findings`, `imaging_findings`, `clinical_notes`, `flagged_signals`, `extraction_confidence`).
- Model loading: `load_model()` optionally applies 4-bit `BitsAndBytesConfig` quantization (`nf4`, double-quant, bf16 compute dtype).

**Agent 2 — Clinical Signal Analyser** ([agents/analyzer.py](agents/analyzer.py))
- Takes the clinical profile JSON produced by Agent 1 and reasons over it with a causal-LM MedGemma model (`AutoModelForCausalLM`, also 4-bit quantized) to find cross-signal patterns (e.g. which findings are clinically meaningful together, urgency, progressiveness).
- `run_inference()` always does a standard pass (`single_call_system_prompt`) against the clinical profile. When the input's `extraction_confidence` is `"low"`, it additionally runs a second **follow-up critique pass** (`low_confidence_system_output_follow_up_prompt`) that re-checks the first pass's output against the original clinical profile for hallucinated/mismatched signals before returning.
- Output validation/repair for this agent's JSON schema (`individual_signals`, `combination_patterns`, `overall_assessment`, `urgency`, `analysis_confidence`) lives in [agents/helper_functions/medgemma_signal_analysis_object_parser.py](agents/helper_functions/medgemma_signal_analysis_object_parser.py), not in `analyzer.py` itself — it additionally strips MedGemma's extended-thinking tokens (`<unused94>...<unused95>`) before parsing, which Agent 1's `validate_clinical_output()` does not do.
- **Known gap:** the low-confidence follow-up branch in `run_inference()` references `validated_signal_analysis_object`, which is never assigned (the standard-pass result is stored as `validated_signal_response_object`) — this branch will raise `NameError` if exercised. `main()` is also still a stub (`pass`).

**Agent 3 — not yet implemented.** Per the trailing comments in `analyzer.py`, the intended next step is to pass Agent 2's signal analysis object, the original clinical profile, and the full source documents forward to a third agent (unbuilt).

**Shared JSON recovery** ([agents/helper_functions/medgemma_response_json_recoverer.py](agents/helper_functions/medgemma_response_json_recoverer.py)): `attempt_json_recovery()` balances unclosed `{}`/`[]` from responses truncated by `max_new_tokens`, used as a fallback by both agents' output validators when `json.loads()` fails outright.

**Custom exceptions** ([agents/custom_exceptions.py](agents/custom_exceptions.py)): `IncompatibleFileFormatException` (700), `EmptyFileExtensionException` (700.2), `ImageEncodingException` (701) — all raised during Agent 1's file validation/encoding steps and carry an `error_code` attribute alongside the message.

**Import style note:** `parser.py` imports sibling modules with bare names (`from custom_exceptions import ...`), implying it's run with `agents/` itself on the path (e.g. as a Kaggle script), whereas `agents/helper_functions/` uses relative imports (`from .medgemma_response_json_recoverer import ...`) and is treated as a proper package. Keep this inconsistency in mind when adding new imports or trying to run a script directly.

**Stubs (currently empty):** [agents/matcher.py](agents/matcher.py) (document classification — not yet implemented despite being referenced in project docs), [agents/__init__.py](agents/__init__.py), [app.py](app.py), [requirements.txt](requirements.txt).
