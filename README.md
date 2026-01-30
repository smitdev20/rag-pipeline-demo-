# RAG Chatbot

Retrieval-Augmented Generation chatbot with FastAPI, Agno, and NiceGUI.

## Cursor Setup

### Linting & Formatting

**Ruff** configured in `pyproject.toml`:
```toml
[tool.ruff]
target-version = "py313"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
```
- Errors surface inline in the editor via Cursor's built-in Python extension
- `UP` rules enforce modern typing (`list[str]` not `List[str]`)
- `SIM` rules catch unnecessary complexity

### .cursorrules

Located at project root, enforces:
- Modern Python 3.13+ typing conventions
- Pydantic for all structured data (no raw dicts)
- Architecture boundaries (FastAPI=HTTP, Agno=agent logic, NiceGUI=UI)
- Testing standards (pytest_check, real files, no mocks in integration)

This prevents Cursor from generating outdated patterns like `Optional[str]` or creating redundant abstractions that duplicate Agno/FastAPI functionality.

### What Helped

1. **Strict .cursorrules** - Prevented over-engineering; kept code minimal
2. **Ruff's UP rules** - Auto-flagged old typing patterns immediately
3. **Real test files in tests/data/** - Caught edge cases that mocked tests would miss

## Setup

```bash
# Install dependencies
poetry install

# Copy environment template
cp .env.example .env
# Edit .env with your API keys

# Run the application
poetry run python -m src.main
```

## Architecture

```
FastAPI (API layer)     →  Agno (Agent logic)  →  OpenAI
     ↑                           ↑
NiceGUI (UI, mounted)      Knowledge base (PDF ingestion)
```

**Design choices:**

- **NiceGUI mounted on FastAPI** - Single server simplifies deployment. Trade-off: coupled scaling. For separate scaling, run `uvicorn src.api.app:app` and `python -m src.ui.chat_page` independently.
- **Pydantic everywhere** - All data structures use BaseModel for validation and serialization.
- **pypdf for PDF parsing** - Lightweight, pure Python. Trade-off: no OCR for scanned documents.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | Required for Agno agent |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |

## Testing

```bash
poetry run pytest
```

**Testing approach:**
- `pytest` with `pytest_check` for soft assertions (multiple checks per test)
- `pytest.raises` for expected exceptions
- Real PDF files in `tests/data/` (no mocks for file parsing)
- Integration tests use real FastAPI ASGI transport (no HTTP mocking)
- LLM tests gated by `@pytest.mark.skipif` when `OPENAI_API_KEY` missing
