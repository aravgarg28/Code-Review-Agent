# Changelog

## 2026-05-14

### End-to-End Pipeline Validation

**All 5 operational steps passed successfully:**

1. **Infrastructure & Database** — PostgreSQL 16 + pgvector container started via Docker Compose (port 5433). Alembic migration `0001` applied: `repository_context` table created with VECTOR(384) column and IVFFlat cosine index. Verified via `psql`.

2. **ML Pipeline Dry Run** — 1-epoch training on 5 mock JSONL records using `microsoft/codebert-base` with LoRA (r=8, alpha=16). Training completed in 14 seconds. Eval F1 macro: 0.33 (expected for 5 samples / 1 epoch). ONNX export + int8 dynamic quantization produced `model_quantized.onnx` (120MB).

3. **RAG Ingestion** — Ingested 180 chunks across 18 files from this repo into pgvector. Fixed critical bug: `discover_files()` was scanning `.venv/` (thousands of site-packages files), causing hour-long hangs. Added exclusion set: `.venv`, `venv`, `__pycache__`, `outputs`, `wandb`.

4. **End-to-End API Test** — FastAPI server booted with quantized ONNX model. `POST /api/v1/review/diff` returned 200 OK with full classification probabilities in 286ms (P90 target: <800ms). Model output is near-uniform (expected — only 1 training epoch on 5 samples).

5. **Extension Build** — `npm install` (0 vulnerabilities) + `tsc` compiled `extension.ts` with `strict: true` and zero errors. Output: `extension/out/extension.js`.

**Bugs fixed during validation:**
- `backend/rag/ingest.py`: Added `.venv`, `venv`, `__pycache__`, `outputs`, `wandb` to excluded directories in `discover_files()`.
- `ml/export.py`: Fixed `merge_peft_weights()` to read `base_model_name_or_path` from `adapter_config.json` instead of trying to load a full model from the checkpoint directory (which only contains adapter weights).
- `ml/train.py`: Added `wandb_enabled` config flag (defaults to `True`). When `False`, `report_to` switches to `["none"]` and W&B init/finish are skipped, preventing login prompts from blocking dry runs.

---

### `ml/data.py` — PyTorch Dataset for PR Diff Parsing (Epic 1, Task 1)

**Added:**

- `DiffRecord` (Pydantic model): Validates each JSONL training sample (`diff: str`, `label: int`). Multi-class ready — label is a non-negative integer, not a binary flag.
- `DatasetConfig` (Pydantic model): Validates dataset constructor args. Tokenizer name is purely config-driven (no hardcoded default) to support both CodeBERT and StarCoder benchmarking. Includes `num_labels` field for multi-class classification.
- `parse_diff_to_text()`: Standalone pure function that converts unified diff strings into a structured text representation. Preserves `[ADD]`/`[DEL]`/`[CTX]` directional markers and hunk headers. Context lines are kept so the model has access to surrounding scope, variable declarations, and control flow.
- `PRDiffDataset(Dataset)`: Core PyTorch Dataset class. Reads a JSONL file of labeled diffs, tokenizes on `__getitem__` via HuggingFace `AutoTokenizer`, and returns `input_ids`, `attention_mask`, and `labels` tensors compatible with the HuggingFace `Trainer` API.
- `DatasetLoadError` / `DiffParseError`: Custom exception classes — no bare `except` blocks.
- Deterministic seeding: `torch.manual_seed` + `np.random.seed` set in `__init__`.

**Design Decisions:**
- Tokenization happens on-access (`__getitem__`), not upfront, to keep memory usage proportional to batch size rather than dataset size.
- No `unidiff` dependency — unified diff format is parsed manually to avoid unnecessary third-party packages.
- `[ADD]`/`[DEL]`/`[CTX]` prefix tokens give the model explicit signal about change direction without relying on raw `+`/`-` characters that may conflict with code content.

### `ml/train.py` — HuggingFace Trainer + PEFT/LoRA + W&B (Epic 1, Tasks 2–4)

**Added:**

- `TrainingConfig` (Pydantic model): Single config object for the entire training run — model name, tokenizer name, data paths, hyperparameters, LoRA settings, and W&B project/run name. All config-driven, no hardcoded defaults for model/tokenizer.
- `LoRAParams` (Pydantic model): Nested config for PEFT/LoRA hyperparameters. Defaults match architecture spec (`r=8`, `lora_alpha=16`, targeting `query`/`value` attention modules).
- `build_model()`: Loads `AutoModelForSequenceClassification`, wraps it with `get_peft_model()` using the LoRA config. Logs total vs. trainable parameter counts.
- `compute_metrics()`: Computes macro-averaged F1, precision, and recall plus per-class F1 scores (`f1_class_0`, `f1_class_1`, ...). All metrics are logged to W&B automatically via the Trainer.
- `train()`: Full training entrypoint. Constructs datasets from `ml/data.PRDiffDataset`, builds model, configures `TrainingArguments` with epoch-level eval/save, `load_best_model_at_end=True` keyed on `f1_macro`, and `report_to=["wandb"]`. Saves the best checkpoint with tokenizer for downstream ONNX export.
- `main()`: CLI entrypoint (`python -m ml.train config.json`). Loads JSON config, initializes W&B run, launches training.
- `TrainingConfigError`: Custom exception for invalid config files.

**Design Decisions:**
- `remove_unused_columns=False` in `TrainingArguments` because `PRDiffDataset` returns a custom dict, not a HuggingFace Dataset with column metadata — without this flag the Trainer would drop all fields.
- Best checkpoint is saved alongside the tokenizer so `ml/export.py` can load both from a single directory.
- W&B `init()` is called in `main()` (CLI boundary) rather than inside `train()` so the `train()` function stays testable without side effects.

### `ml/export.py` — ONNX Export + int8 Quantization (Epic 1, Task 5)

**Added:**

- `ExportConfig` (Pydantic model): Config for checkpoint dir, output dir, ONNX opset version, quantization toggle, and `num_labels`.
- `merge_peft_weights()`: Loads the PEFT checkpoint, calls `merge_and_unload()` to fold LoRA adapters back into the base weights, and saves a standalone PyTorch model. Required because ONNX export needs a vanilla `transformers` model, not a PEFT wrapper.
- `export_to_onnx()`: Uses `ORTModelForSequenceClassification.from_pretrained(export=True)` from HuggingFace Optimum to convert the merged model. Saves the ONNX model + tokenizer together.
- `quantize_onnx()`: Applies int8 dynamic quantization via `ORTQuantizer` with `AutoQuantizationConfig.avx512_vnni(is_static=False)` targeting linear layers for CPU acceleration per architecture spec.
- `export()`: Orchestrates the full pipeline: merge → ONNX → quantize. Returns path to the final model directory.
- `main()`: CLI entrypoint (`python -m ml.export config.json`).
- `ExportError`: Custom exception for export/quantization failures.

**Design Decisions:**
- Three-stage pipeline (merge → export → quantize) rather than a single call, so each stage can be debugged or rerun independently.
- Tokenizer is saved at every stage so any intermediate directory is self-contained and loadable for inference.

### Epic 2: Database & RAG Pipeline (Tasks 1–4)

#### `infrastructure/docker-compose.yml`
- PostgreSQL 16 with pgvector via `pgvector/pgvector:pg16` image.
- All credentials via env vars (`POSTGRES_PASSWORD` is required, others have defaults).
- Named volume `pgdata` for persistence. Health check with `pg_isready`.

#### `backend/db/models.py`
- `Base` (DeclarativeBase): shared ORM base.
- `RepositoryContext`: maps the `repository_context` table from the architecture spec. UUID primary key, `repo_name` (indexed), `file_path`, `chunk_content` (TEXT), `embedding` (VECTOR(384) via `pgvector.sqlalchemy`), `last_updated` (auto-managed timestamp).

#### `backend/db/session.py`
- Async engine + session factory via `sqlalchemy.ext.asyncio`. `DATABASE_URL` from env var.

#### `backend/db/alembic/`
- Full Alembic scaffolding with async `env.py` pointing at `backend.db.models.Base.metadata`.
- Migration `0001_create_repository_context`: enables `vector` extension, creates table, adds IVFFlat cosine index on the embedding column (`lists=100`).

#### `backend/rag/ingest.py`
- `IngestConfig` (Pydantic): `repo_path`, `repo_name`, `embedding_model` (default `all-MiniLM-L6-v2`), `chunk_size`, `chunk_overlap`, `batch_size`.
- `discover_files()`: walks repo, filters `.py`/`.md`, excludes `.git`/`node_modules`.
- `chunk_file()`: reads file and splits via LangChain `RecursiveCharacterTextSplitter`.
- `embed_chunks()`: encodes text via `sentence-transformers`, returns normalized vectors.
- `upsert_chunks()`: async bulk insert into `repository_context` via SQLAlchemy.
- `ingest()`: orchestrates discover → chunk → embed → upsert. Deletes stale data for the repo before re-inserting.
- CLI: `python -m backend.rag.ingest config.json`.

**Design Decisions:**
- Full delete-then-insert per repo rather than row-level diffing — simpler, and re-ingestion is infrequent. The IVFFlat index will need a `REINDEX` after large ingestions; this is acceptable for the offline pipeline.
- Embedding is done in-process with `sentence-transformers` rather than via an external API to keep the pipeline self-contained and free of API costs.
- Async all the way through (session, upsert) to align with the project's async-first standard and share the same engine/session pattern the FastAPI backend will use.

### Epic 3: FastAPI & ONNX Inference (Tasks 1–4)

#### `backend/inference/engine.py`
- `EngineConfig` (Pydantic): `model_dir`, `max_length`, `label_names` (configurable multi-class label taxonomy).
- `PredictionResult` (Pydantic): `predicted_label`, `predicted_index`, `confidence`, `probabilities` dict.
- `ONNXReviewEngine`: loads tokenizer + ONNX session at construction. `predict(diff)` parses the diff via `ml.data.parse_diff_to_text`, tokenizes to NumPy, runs `session.run()`, applies softmax, returns `PredictionResult`. Also exposes `predict_batch()`.
- `InferenceError` / `ModelLoadError`: custom exceptions.

#### `backend/api/main.py`
- FastAPI app with `lifespan` context manager: initializes `ONNXReviewEngine` from env vars (`MODEL_DIR`, `MAX_LENGTH`, `LABEL_NAMES`) on startup, attaches to `app.state.engine`.
- Includes router at `/api/v1`.

#### `backend/api/routes.py`
- `POST /api/v1/review/pr`: accepts `repository`, `pull_request_number`, `github_token`. Fetches PR files via `GitHubClient`, runs inference per file patch, filters out `clean` predictions, posts inline review comments back to GitHub. Returns `ReviewPRResponse` with comments list and inference timing.
- `POST /api/v1/review/diff`: accepts a raw diff string, returns `PredictionResult` + timing. Useful for the VS Code extension and testing without GitHub.
- All request/response payloads are Pydantic models.

#### `backend/github/client.py`
- `GitHubClient`: async HTTP client using `httpx` (not PyGithub — async-native, no heavyweight dependency).
- `get_pr_diff_files()`: fetches `GET /repos/{owner/repo}/pulls/{pr}/files`, returns list of file dicts with patches.
- `post_review_comments()`: posts a batch review via `POST /repos/{owner/repo}/pulls/{pr}/reviews` with `event: COMMENT`.
- `GitHubClientError`: custom exception for non-200 responses.

**Design Decisions:**
- Used `httpx.AsyncClient` instead of PyGithub for GitHub API calls — PyGithub is synchronous and would require thread pool wrapping. `httpx` is async-native and lightweight.
- Two review endpoints: `/review/pr` (full GitHub flow) and `/review/diff` (raw diff, no GitHub). The VS Code extension will use `/review/diff` to avoid requiring a GitHub token for local reviews.
- Engine is initialized once at startup via lifespan, not per-request — ONNX session creation is expensive.
- `github_token` is passed per-request rather than stored server-side, so the API is stateless and multi-tenant safe.

### Epic 4: Deployment & Tooling (Tasks 1–3)

#### `infrastructure/Dockerfile`
- Multi-stage build: `python:3.11-slim` builder installs dependencies from `requirements.txt`, slim runtime copies only installed packages + application code + quantized ONNX model.
- Configurable `MODEL_DIR` build arg, `PORT=8080`, healthcheck against `/docs`.
- Runs via `uvicorn backend.api.main:app`.

#### `extension/package.json`
- VS Code extension manifest: `code-review-agent` with command `codeReviewAgent.reviewDiff`.
- Configurable `codeReviewAgent.apiUrl` setting (default `http://localhost:8080`).
- TypeScript build with `strict: true` in `tsconfig.json`, no `any` types.

#### `extension/src/extension.ts`
- `getGitDiff()`: extracts `git diff HEAD`, falls back to `git diff --cached` for staged-only changes.
- `reviewDiff()`: sends diff to `POST /api/v1/review/diff`, displays results in a dedicated output channel. Shows progress notification during inference. Clean predictions show an info toast; anti-patterns show a warning with label and confidence.
- Uses Node built-in `http`/`https` — no external dependencies beyond VS Code API.
- Never blocks the main thread — all I/O is async with progress indicator.
