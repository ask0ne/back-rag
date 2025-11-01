# Agentic RAG System

A FastAPI-powered Retrieval-Augmented Generation (RAG) application that serves a Tailwind/HTMX UI, indexes local documents with Chroma, and selectively falls back to DuckDuckGo web search when confidence in the internal knowledge base is low. The orchestration layer combines LangChain tools, LangGraph state management, and a `BackrubService` facade for unified decisioning.

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌────────────────┐
│   FastAPI App   │────│  Backrub Service │────│ Agentic RAG    │
└─────────────────┘    └──────────────────┘    └────────────────┘
         │                       │                       │
         │              ┌────────┴────────┐              │
         │              │                 │              │
         ▼              ▼                 ▼              ▼
┌──────────────┐ ┌─────────────┐ ┌─────────────────┐ ┌──────────────┐
│   Document   │ │ RAG Service │ │  Web Search     │ │  Confidence  │
│   Ingestion  │ │             │ │  Service        │ │  Evaluation  │
└──────────────┘ └─────────────┘ └─────────────────┘ └──────────────┘
         │              │                 │
         ▼              ▼                 ▼
┌──────────────┐ ┌─────────────┐ ┌─────────────────┐
│   Chroma     │ │ ChatOpenAI  │ │   DuckDuckGo    │
│  Vector DB   │ │  LLM + EMB  │ │     Search      │
└──────────────┘ └─────────────┘ └─────────────────┘
```

## Key Features

### Functional Requirements

1. **Document Ingestion**: Automatically load `.pdf` and `.txt` files from the `documents/` directory at startup and on demand via the refresh endpoint.
2. **RAG Pipeline**: Retrieve with Chroma/Sentence-Transformers embeddings and generate answers with OpenAI Chat models, combining retrieval and LLM confidence signals.
3. **Agentic Decision**: Coordinate a LangGraph-powered `AgenticRAGAgent` through the `BackrubService` to determine whether to trust the vector store or defer to web results.
4. **Web Search Integration**: Query DuckDuckGo, summarize the top results, and surface source snippets when local confidence is below the configured threshold.
5. **Structured Output**: Provide explicit reasoning text, confidence scores, and context snippets to drive both the UI and downstream consumers.

### Technical Stack

- **Backend**: FastAPI + Uvicorn
- **Agent Framework**: LangGraph with stateful orchestration
- **LLM**: OpenAI `ChatOpenAI` (default `gpt-4o-mini`, configurable via environment)
- **Vector Store**: Chroma persistent client backed by local storage
- **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`) wrapped for LangChain
- **Search Engine**: DuckDuckGo via `duckduckgo-search`
- **Frontend**: HTMX + Tailwind CSS templates served by FastAPI
- **Logging**: Python `logging` wired through the service layer

## Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key with access to the configured Chat model
- (Optional) A virtual environment for dependency isolation

### Installation

1. **Clone and enter the project:**
   ```bash
   git clone <repository-url>
   cd back-rag
   ```

2. **Create your environment and install deps:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure secrets:**
   ```bash
   cp .env.example .env
   # Set OPENAI_API_KEY, MODEL_NAME (optional), and other overrides
   ```

4. **Start the FastAPI app:**
   ```bash
   uvicorn main:app --reload
   ```

5. **Access the application:**
   - **Frontend UI**: http://localhost:8000
   - **Interactive API Docs**: http://localhost:8000/docs

## Frontend Interface

The bundled HTMX/Tailwind single-page experience in `templates/index.html` provides an immediate way to explore document recall and web fallbacks without additional wiring.

### Features
- **Simple Search Box**: Minimal form submits via HTMX to the unified `/api/search` endpoint.
- **Dynamic Stats Bar**: Lazy-load system stats (model, embeddings, search engine) from `/stats`.
- **Confidence Slider**: Adjust the 0–1 threshold that governs document vs web selection.
- **Re-Scan Button**: Trigger `/refresh-memory` without leaving the page to pick up new files.
- **Reasoned Results**: Display answer, confidence bar, reasoning text, and source snippets.
- **Responsive Styling**: Tailwind CSS theme tuned for dark mode layouts.


### Using the Frontend
1. Enter a question in the search box.
2. (Optional) Nudge the confidence slider toward “Document” or “Web.”
3. Submit the form to issue an HTMX request against `/api/search`.
4. Review the rendered answer, confidence meter, reasoning text, and top snippets.
5. Use “Re-Scan Documents” whenever new files land in `documents/`.


## Agent Decision Logic

LangGraph compiles a state graph composed of `rag_retrieval`, `confidence_check`, `web_search`, and `format_response` nodes. The `AgenticRAGAgent`:

1. Retrieves from Chroma and produces an initial answer plus retrieval and LLM confidence estimates.
2. Compares the combined score against `CONFIDENCE_THRESHOLD`.
3. Delegates to DuckDuckGo summarization if the score is insufficient, appending web snippets to the context.
4. Returns the selected answer, annotated reasoning steps, and the chosen `SourceType`.

The `BackrubService` wraps this logic to provide a resilient API that falls back gracefully if either RAG or web search encounters errors.


## Logging

Standard logging is initialized when services spin up. Set `DEBUG=true` (default) to surface detailed tracing across the ingestion, RAG, and web search layers. Uvicorn integrates with the same handlers so request logs and exceptions appear in the console.


## Sample Documents

Place sample documents in `./documents/` directory:
- Research papers (PDF)
- Documentation (TXT)
- Articles (TXT/PDF)
