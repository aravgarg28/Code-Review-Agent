---

### 4. `docs/TASKS.md` (The Execution Roadmap)
*Highly granular steps. Claude will read this, find the first unchecked box, and know exactly what to execute.*

```markdown
# Implementation Roadmap

**Status Key:** [ ] Not Started | [~] In Progress | [x] Completed

## Epic 1: ML Pipeline Foundation
- [x] `ml/data.py`: Write PyTorch `Dataset` class to parse GitHub PR diffs into tokenized input IDs and attention masks.
- [x] `ml/train.py`: Set up HuggingFace `Trainer` loop.
- [x] `ml/train.py`: Implement PEFT/LoRA configuration using `peft` library.
- [x] `ml/train.py`: Integrate Weights & Biases (`wandb`) for F1 score tracking.
- [x] `ml/export.py`: Write script using HuggingFace `Optimum` to convert PyTorch checkpoint to `model.onnx` with int8 quantization.

## Epic 2: Database & RAG Backing
- [x] `infrastructure/docker-compose.yml`: Define `postgres` service with `pgvector` image.
- [x] `backend/db/models.py`: Define SQLAlchemy models for the `repository_context` table.
- [x] `backend/db/alembic/`: Generate initial database migration.
- [x] `backend/rag/ingest.py`: Write LangChain script to traverse a local Git repo, chunk `.py` and `.md` files using `RecursiveCharacterTextSplitter`, and upsert embeddings.

## Epic 3: FastAPI & ONNX Inference
- [x] `backend/inference/engine.py`: Initialize `onnxruntime.InferenceSession` and write the tokenization/prediction logic.
- [x] `backend/api/main.py`: Scaffold FastAPI application with Pydantic validation.
- [x] `backend/api/routes.py`: Implement `POST /api/v1/review/pr` endpoint.
- [x] `backend/github/client.py`: Implement PyGithub wrapper to fetch PR diffs and post inline review comments.

## Epic 4: Deployment & Tooling
- [ ] `infrastructure/Dockerfile`: Create multi-stage Dockerfile optimized for AWS Lambda Web Adapter.
- [ ] `extension/package.json`: Scaffold VS Code extension manifest.
- [ ] `extension/src/extension.ts`: Implement VS Code command to extract current diff and hit local FastAPI endpoint.