# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git Commands
- Common operations:
  - `/commit "[message]"` – stage and commit changes
  - `/push` – push current branch to remote
  - `/pull` – fetch and merge remote changes
  - `/log` – view commit history (use `--all` for full history)
  - `/revert HEAD~1` – revert the last commit
  - `/branch [branch-name]` – create a new branch
  - `/merge [branch-name]` – merge specified branch
  - `/delete [branch-name]` – delete a branch

## Codebase Structure
- **Agents** (`./agents/`):
  - `matcher.py` – document classification logic
  - `analyzer.py` – clinical signal extraction and processing
  - `parser.py` – document/content parsing utilities
  - `custom_exceptions.py` – domain‑specific error types
- **Notebooks** (`./notebooks/`):
  - `undiagnosed-gemma4-testing-v1.ipynb` – Gemma model experiments
  - `test-qa.ipynb` – LangChain test suite
  - `core_business.ipynb` – business logic prototypes
- **Configuration files**:
  - Project root: `.env`, `.dvcignore`, `.gitignore`, `.venv`
  - Agent-specific configs in `agents/`

## Dependency Management
- Model weights are pulled via Git LFS from `/kaggle/input/models/google/gemma-4/...`
- Core Python dependencies are listed in `requirements.txt`
- Frequently reinstalled via `/loop 5m` commands (e.g., `git clone https://github.com/huggingface/transformers.git` and `pip install biopython pymupdf chromadb sentence-transformers langgraph pillow bitsandbytes`)

## Common Development Tasks
1. **Training / Refresh Models**  
   ```bash
   /loop 5m /train
   /push
   ```
2. **Running Tests**  
   ```bash
   /loop 5m ./run_tests
   /coverage report
   ```
3. **Clinical Signal Extraction**  
   ```bash
   /loop 5m ./agents/analyzer.py
   ```

## Custom Exceptions
Agents raise domain‑specific errors defined in `agents/custom_exceptions.py`:
- `IncompatibleFileFormatException` (error code 700.2) – invalid document extension
- `ImageEncodingException` (714.0) – image encoding failure
- `EmptyFileExtensionException` (700) – missing file suffix

## Model Loading Procedure
1. Load preprocessing layers via imports from `agents.matcher` and related modules.  
2. Initialize model with appropriate device mapping (`device_map="auto"`).  
3. Apply quantization configuration (`BitsAndBytesConfig`) for 4‑bit loading when needed.

## Clinical Signal Extraction Workflow
```python
# 1. Parse document
processed_doc = analyzer.extract_document(file_path)

# 2. Run inference pipeline
signals = extract_clinical_signals(processed_doc)

# 3. Validate output against schema
validated_signals = validate_clinical_output(signals)
```

## Error Handling
- **Missing extensions**: Ensure file validation rules are added to `agents/document_validator()`.
- **Model loading failures**: Check GPU memory allocation and `torch.cuda.empty_cache()` before loading.
- **Image processing errors**: Verify image bytes are non‑empty before encoding; handle `ImageEncodingException` gracefully.

Use this file as the canonical reference for all interactions with Claude Code in this repository.