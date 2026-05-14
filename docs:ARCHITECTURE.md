# System Architecture

## 1. System Topology
The system operates in two distinct phases: Offline Training and Online Inference.

### Online Inference Flow (AWS Lambda / FastAPI)
1. **Client** (GitHub Webhook / VS Code) sends `POST /api/v1/review/diff`.
2. **API Handler** parses the unified diff and extracts changed files/lines.
3. **LangChain Retriever** queries `pgvector` for context:
   - *Query*: "How is error handling usually done in this module?"
   - *Result*: Top 3 similar code snippets.
4. **ONNX Runtime** loads the quantized CodeBERT model.
5. **Inference**: Model evaluates Diff + RAG Context.
6. **Response**: API formats the result and posts an inline comment to GitHub or returns JSON to VS Code.

## 2. Database Schema (PostgreSQL + pgvector)
We use `pgvector` for storing embedding context.

**Table: `repository_context`**
- `id` (UUID, Primary Key)
- `repo_name` (VARCHAR)
- `file_path` (VARCHAR)
- `chunk_content` (TEXT) - The raw code or doc chunk.
- `embedding` (VECTOR(384)) - Using `all-MiniLM-L6-v2` dimensions.
- `last_updated` (TIMESTAMP)

## 3. API Contracts (FastAPI)

### `POST /api/v1/review/pr`
**Request Payload:**
```json
{
  "repository": "owner/repo",
  "pull_request_number": 123,
  "diff_url": "[https://github.com/](https://github.com/)...",
  "base_commit": "sha...",
  "head_commit": "sha..."
}

**Response Payload:**
```json
{
  "status": "success",
  "comments_posted": 2,
  "inference_time_ms": 412
}

## 4. Model Optimization Specs
- **Base Model**: `microsoft/codebert-base` or `bigcode/starcoderbase-1b`.
- **LoRA Config**: `r=8`, `lora_alpha=16`, targeting `query`, `value` attention modules.
- **ONNX Export**: Use `optimum` library for conversion. Apply dynamic quantization (int8) to linear layers for CPU acceleration.