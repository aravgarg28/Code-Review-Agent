# Project: AI-Augmented Code Review Agent

## 1. System Overview
An automated, RAG-enhanced code review system analyzing GitHub PR diffs for anti-patterns. Powered by fine-tuned CodeBERT/StarCoder (PEFT/LoRA) exported to ONNX for CPU-optimized inference, backed by FastAPI, AWS Lambda, and a PostgreSQL/pgvector RAG pipeline.

## 2. Strict Engineering Standards
Claude MUST adhere to the following rules when writing or modifying code in this repository:

### Python Standards (Backend & ML)
- **Typing**: Strict type hints are mandatory (`mypy` compliant). Use `Pydantic` for all data validation and API payloads.
- **Formatting**: Code must comply with `black` and `ruff`.
- **Async**: Use `asyncio` for all I/O bound tasks (database queries, GitHub API calls).
- **Error Handling**: No bare `except` blocks. Use custom exception classes. Fast fail and log stack traces using structured JSON logging.

### Machine Learning Standards
- **Determinism**: Set random seeds (`torch.manual_seed`, `np.random.seed`) in all training and evaluation scripts.
- **Paths**: Never hardcode file paths. Use `pathlib.Path` relative to the project root.
- **Memory**: When writing inference code, explicitly manage memory (e.g., deleting large tensors, using `with torch.no_grad():` where applicable before ONNX export).

### TypeScript Standards (VS Code Extension)
- **Typing**: `strict: true` in `tsconfig.json`. No `any` types allowed.
- **UI Responsiveness**: Extension must never block the main thread. Long-running reviews must show a progress indicator.

## 3. Operational Directives
1. **Read Before Act**: Always review `docs/ARCHITECTURE.md` for data flows and `docs/TASKS.md` for current state before beginning a task.
2. **Atomic Commits**: Commits must do one thing. Format: `<type>(<scope>): <subject>`.
3. **Changelog Maintenance**: After completing a task, you MUST append a detailed technical summary to `docs/CHANGELOG.md`.

## 4. Repository Structure
- `/ml/`: Dataset processing, LoRA fine-tuning, ONNX conversion scripts.
- `/backend/`: FastAPI application, LangChain logic, Lambda handlers.
- `/database/`: Alembic migrations and PostgreSQL setup scripts.
- `/extension/`: VS Code extension source code.
- `/infrastructure/`: Terraform or AWS SAM templates, Dockerfiles.