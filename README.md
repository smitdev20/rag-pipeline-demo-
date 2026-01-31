# RAG Chatbot

Retrieval-Augmented Generation chatbot for intelligent document Q&A. Built with **FastAPI** (HTTP/streaming), **Agno** (agent logic, sessions, memory, knowledge), and **NiceGUI** (thin UI layer). Ingest PDFs, query with natural language, and get context-aware streaming responses.

---

## 1. Setup

**Requirements:** Python 3.13+

```bash
# Clone and enter project
cd Demo

# Option A: Poetry (recommended)
poetry install

# Option B: pip
python3.13 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .

# Environment
cp .env.example .env
# Edit .env: set OPENAI_API_KEY (required); optional LLM_BASE_URL, LLM_MODEL, LOG_LEVEL, HOST, PORT
```

| Variable        | Default   | Description              |
|----------------|-----------|--------------------------|
| `OPENAI_API_KEY` | *(required)* | API key (OpenAI or OpenAI-compatible provider) |
| `LLM_BASE_URL` | - | Override API base URL for OpenAI-compatible endpoints |
| `LLM_MODEL`    | `gpt-4o-mini` | Model name |
| `LOG_LEVEL`    | `INFO`    | Logging verbosity        |
| `HOST`         | `0.0.0.0` | Server bind address     |
| `PORT`         | `8000`    | Server port              |
| `NICEGUI_STORAGE_SECRET` | `rag-chatbot-secret` | Session storage secret |
| `API_BASE_URL` | `http://localhost:8000` | UI → API base URL (when separate) |

---

## 2. Run

```bash
# Single process: FastAPI + NiceGUI on one server
poetry run python -m src.main
# Or: python -m src.main

# Open in browser: http://localhost:8000
# API docs: http://localhost:8000/docs
```

For separate API and UI (e.g. scaling): run `uvicorn src.api.app:app --port 8000` and `python -m src.ui.chat_page` independently.

---

## 3. Test

```bash
poetry run pytest
# Or: pytest
```

- **pytest** with **pytest_check** (soft assertions), **pytest-asyncio** (async tests).
- **Unit tests:** `tests/unit/` (agent, PDF parser).
- **Integration tests:** `tests/integration/` (upload, chat streaming); real FastAPI ASGI transport and sample files in `tests/data/`, no HTTP mocks.
- LLM-dependent tests are skipped when `OPENAI_API_KEY` is unset (`@pytest.mark.skipif`).

follow the white rabbit

---

## 4. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  NiceGUI (UI) – chat page, upload, message list                  │
│  mounted on FastAPI → single server                             │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│  FastAPI – HTTP/streaming only                                   │
│  /upload (PDF ingest), /chat (streaming), /health, /docs         │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│  Agno – Agent logic, sessions, memory, tools                     │
│  Knowledge base (LanceDB) for RAG retrieval                      │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                    OpenAI (embeddings + chat)
```

- **FastAPI:** Routes and streaming only; no business logic.
- **Agno:** Agent, session/store, knowledge base, and LLM calls.
- **NiceGUI:** Thin UI; delegates all behavior to API.
- **Pydantic:** All request/response and internal structures (no raw dicts).
- **PDF parsing:** `pypdf` (no OCR). Vector store: LanceDB (local).

---

## 5. Cursor Configuration (MANDATORY)

### MCP docs indexed

- **Agno**, **NiceGUI**, and **FastAPI** – MCP documentation for all three was used and indexed in Cursor for accurate API usage, patterns, and best practices while building the app.
- Repository and project files are also indexed by Cursor.

### .cursorrules content

Project root `.cursorrules` enforces:

- **Python 3.13+ only.**
- **Modern typing:** `list[str]` not `List[str]`, `str | None` not `Optional[str]`.
- **Pydantic BaseModel** for ALL structured data – NO raw dicts.
- **Async/await** for ALL I/O operations.
- **Minimal code** – use library built-ins, don’t reinvent.
- **Architecture:** FastAPI = HTTP/streaming only; Agno = agent logic, sessions, memory, knowledge; NiceGUI = thin UI; Pydantic = all request/response models.
- **Testing:** pytest with pytest_check; no mocks in integration tests; real sample files in `tests/data/`.

This keeps generated code consistent and avoids outdated patterns or duplicated Agno/FastAPI behavior.

### Linting setup (Ruff)

Ruff is configured in `pyproject.toml`:

```toml
[tool.ruff]
target-version = "py313"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
```

- **E / F:** pycodestyle errors, Pyflakes.
- **I:** isort (import sorting).
- **UP:** pyupgrade – enforces modern typing and syntax (e.g. `list[str]`, `str | None`).
- **B:** bugbear – common bugs and style issues.
- **SIM:** simplify – unnecessary complexity.

Errors show inline in Cursor via the Python extension.

### What helped

1. **MCP docs for Agno, NiceGUI, and FastAPI** – Indexed in Cursor so API usage, streaming, and UI patterns stayed accurate across all three stacks.
2. **Strict .cursorrules** – Avoided over-engineering and kept the codebase minimal and aligned with Agno/FastAPI.
3. **Ruff UP rules** – Surfaced old typing patterns immediately (e.g. `Optional[str]` → `str | None`).
4. **Real test files in `tests/data/`** – Integration tests caught real edge cases that mocks would miss.
5. **Clear architecture rules** – FastAPI = HTTP only, Agno = agent/knowledge; reduced duplicate or misplaced logic.

---

## 6. Trade-offs and Limitations

| Decision                    | Trade-off / limitation                          |
|----------------------------|-------------------------------------------------|
| **pypdf for PDF parsing**  | No OCR; scanned/image PDFs not supported.       |
| **NiceGUI mounted on FastAPI** | Single process; UI and API scale together.  |
| **SQLite for sessions**    | Single-node; no horizontal scaling of sessions. |
| **LanceDB local storage**  | Local vector store; no distributed vector search. |
| **No auth in default setup** | Single-user / demo; add auth for multi-user. |

---

## 8. What You'd Add Next

- **Multi-document support** – Upload and query across multiple PDFs in one session.
- **OCR** – e.g. pytesseract for scanned PDFs.
- **Streaming status from Agno** – Expose agent tool calls and steps as UI status updates.
- **Authentication** – Session-based or token auth for multi-user deployment.
- **Optional MCP/Agno tools** – Enable Agno MCP or other tools if needed; document any MCP docs indexed for Cursor.
