# Product Requirements Document (PRD)

## 1. Product Vision
To provide developers with instantaneous, context-aware, and highly accurate code reviews that catch anti-patterns before CI/CD pipelines fail or human reviewers intervene, ultimately giving developers more agency over their code quality.

## 2. Target Personas
- **The Contributor**: Wants immediate feedback on their PRs or local commits without waiting 12+ hours for a senior engineer to review.
- **The Maintainer**: Wants the AI to filter out obvious anti-patterns, style violations, and common logical bugs so they can focus human review on architecture and business logic.

## 3. Success Metrics & SLAs
- **Accuracy**: >89% Precision (per-class F1) on anti-pattern detection across targeted languages.
- **Latency (API)**: P90 response time < 800ms for diffs under 500 lines.
- **Cost Efficiency**: Model must run on AWS Lambda (CPU only), achieving a 3x latency reduction via ONNX compared to base PyTorch.

## 4. Core Epics & Features

### Epic 1: ML Pipeline & Optimization
- Ability to ingest curated GitHub PR diffs.
- Parameter-efficient fine-tuning (PEFT/LoRA) of CodeBERT/StarCoder.
- Automated export pipeline from PyTorch -> ONNX runtime.

### Epic 2: RAG Pipeline (Context-Awareness)
- Ingest repository code base and architecture docs into PostgreSQL via `pgvector`.
- Vector similarity search to fetch repository-specific coding standards to ground the LLM's review.

### Epic 3: Review API & Webhooks
- FastAPI service deployable via Docker to AWS Lambda.
- Webhook endpoint to consume GitHub `pull_request` events (`opened`, `synchronize`).
- Automated posting of inline GitHub review comments using the GitHub REST API.

### Epic 4: Developer Tooling
- VS Code Extension to run the agent locally against the active Git diff.

## 5. Out of Scope (Do Not Build)
- Auto-committing code fixes (read-only review comments only).
- Multi-repository knowledge graphs (context is strictly scoped to the repository being reviewed).
- Fine-tuning on proprietary codebase data (RAG is used for local context, not fine-tuning).