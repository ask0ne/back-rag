"""Main FastAPI application for Agentic RAG System."""

# Disable ChromaDB telemetry FIRST - before any other imports
import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Depends, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Optional
import time
from pathlib import Path
import markdown

# Import our modules
from src.config import config

from src.services.document_ingestion import DocumentIngestionService
from src.services.rag_service import RAGService
from src.services.web_search import WebSearchService
from src.services.backrub import BackrubService
from src.agents.agentic_rag import AgenticRAGAgent

import logging


# Validate configuration on startup
config.validate_config()

# Global service instances
document_service: Optional[DocumentIngestionService] = None
rag_service: Optional[RAGService] = None
web_search_service: Optional[WebSearchService] = None
backrub_service: Optional[BackrubService] = None
agent: Optional[AgenticRAGAgent] = None


def get_services():
    """Dependency to get service instances."""
    if not all(
        [document_service, rag_service, web_search_service, backrub_service, agent]
    ):
        raise HTTPException(status_code=500, detail="Services not properly initialized")
    return document_service, rag_service, web_search_service, backrub_service, agent


def initialize_services():
    """Initialize all service instances."""
    global document_service, rag_service, web_search_service, backrub_service, agent

    try:
        logging.info("Initializing services...")

        # Initialize document ingestion service
        logging.debug("Creating DocumentIngestionService...")
        document_service = DocumentIngestionService()
        logging.debug("✓ DocumentIngestionService created")

        # Initialize RAG service
        logging.debug("Creating RAGService...")
        rag_service = RAGService(vector_store=document_service.vector_store)
        logging.debug("✓ RAGService created")

        # Initialize web search service
        logging.debug("Creating WebSearchService...")
        web_search_service = WebSearchService()
        logging.debug("✓ WebSearchService created")

        # Initialize backrub service (unified search)
        logging.debug("Creating BackrubService...")
        backrub_service = BackrubService(
            rag_service=rag_service, web_search_service=web_search_service
        )
        logging.debug("✓ BackrubService created")

        # Initialize agentic RAG agent
        logging.debug("Creating AgenticRAGAgent...")
        agent = AgenticRAGAgent(
            rag_service=rag_service, web_search_service=web_search_service
        )
        logging.debug("✓ AgenticRAGAgent created")

        logging.info("All services initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize services: {e}", exc_info=True)
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize services
    initialize_services()

    # Load documents from the documents directory if it exists
    documents_dir = Path("./documents")
    documents = None
    if documents_dir.exists():
        logging.info(f"Loading documents from {documents_dir}...")
        documents = document_service.load_documents_from_directory(str(documents_dir))

    if documents:
        # Index documents
        logging.info("Indexing documents...")
        document_service.index_documents(documents)

    yield

    # Clean up the ML models and release the resources
    if document_service:
        document_service.clear_vector_store()
        logging.info("Cleaned up resources")


# Initialize FastAPI app
app = FastAPI(
    title=config.APP_NAME,
    version=config.APP_VERSION,
    description="An agentic RAG system that intelligently decides between document retrieval and web search",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = Path(__file__).parent / "static"
static_path.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Setup Jinja2 templates
templates_path = Path(__file__).parent / "templates"
templates_path.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(templates_path))


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/search", response_class=HTMLResponse)
async def search_htmx(
    request: Request,
    query: str = Form(...),
    threshold: float = Form(0.7),
    services=Depends(get_services),
) -> str:
    """
    Unified HTMX search endpoint using Backrub service.

    This single endpoint handles all search requests:
    - Queries RAG first
    - Falls back to web if confidence < threshold
    - Returns formatted HTML for HTMX
    """
    logging.debug("POST /api/search - Unpacking services...")
    _, _, _, backrub_service, _ = services

    logging.info(f"POST /api/search - Query: '{query}' (threshold: {threshold})")

    try:
        # Process query through the backrub service
        logging.debug(f"Calling backrub_service.search()...")
        result = backrub_service.search(query=query, threshold=threshold)
        logging.debug(
            f"Backrub result: source={result.get('source')}, confidence={result.get('confidence')}"
        )

        # Extract data from result
        _source = result["source"]
        # Normalize enum to string for templating
        source = _source.value if hasattr(_source, "value") else str(_source)
        answer = result["answer"]
        confidence = result["confidence"]
        context_snippets = result["context_snippets"]
        reasoning = result["reasoning"]

        # Convert markdown answer to HTML
        answer_html = markdown.markdown(
            answer, extensions=["fenced_code", "tables", "nl2br"]
        )

        # Clean up context snippets for display
        cleaned_snippets = []
        if context_snippets:
            for snippet in context_snippets[:3]:
                # Replace newlines with spaces and strip
                cleaned = snippet.replace("\n", " ").strip()
                cleaned_snippets.append(cleaned)

        # Calculate confidence percentage and color
        confidence_pct = int(confidence * 100)
        if confidence >= 0.8:
            confidence_color = "#4ade80"  # green
        elif confidence >= 0.6:
            confidence_color = "#facc15"  # yellow
        else:
            confidence_color = "#DB585A"  # highlight red

        logging.info(
            f"/api/search -> 200 (source: {source}, confidence: {confidence_pct}%)"
        )

        # Render template with context
        html_response = templates.TemplateResponse(
            "search_result.html",
            {
                "request": request,
                "source": source,
                "answer": answer_html,
                "confidence": confidence,
                "confidence_pct": confidence_pct,
                "confidence_color": confidence_color,
                "context_snippets": cleaned_snippets,
                "reasoning": reasoning,
            },
        )
        return html_response

    except Exception as e:
        logging.error(f"Search failed for query '{query}': {e}", exc_info=True)
        # Return error HTML using template
        html_response = templates.TemplateResponse(
            "error.html", {"request": request, "error_message": str(e)}
        )
        return html_response


@app.get("/stats")
async def get_stats(request: Request, services=Depends(get_services)):
    """Get system statistics - returns HTML for HTMX or JSON for API calls."""
    logging.debug("GET /stats - Unpacking services...")
    document_service, rag_service, web_search_service, backrub_service, agent = services

    logging.info("GET /stats")

    try:
        logging.debug("Gathering stats...")
        stats = {
            "system": {
                "confidence_threshold": config.CONFIDENCE_THRESHOLD,
                "model": config.MODEL_NAME,
                "embedding_model": config.EMBEDDING_MODEL,
            },
            "documents": document_service.get_collection_stats(),
            "agent": agent.get_agent_stats(),
        }
        logging.debug(f"Stats gathered: {stats}")

        model = stats["system"]["model"]
        embedding_model = stats["system"]["embedding_model"]
        doc_count = stats["documents"].get("total_documents", 0)

        # Return just the HTML string for HTMX partial update
        html_content = templates.get_template("stats.html").render(
            {
                "request": request,
                "model": model,
                "embedding_model": embedding_model,
                "doc_count": doc_count,
                "search_engine": config.SEARCH_ENGINE,
            }
        )
        logging.info("/stats -> 200 (HTML)")
        return HTMLResponse(content=html_content)

    except Exception as e:
        logging.error(f"Stats retrieval failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve stats: {str(e)}"
        )


@app.post("/refresh-memory")
async def refresh_memory(services=Depends(get_services)):
    """Clear existing memory and reload all documents from the documents directory."""
    document_service, _, _, _, _ = services

    logging.info("POST /refresh-memory")

    try:
        # Load documents from the documents directory if it exists
        documents_dir = Path("./documents")
        documents = None
        if documents_dir.exists():
            logging.info(f"Loading documents from {documents_dir}...")
            chunks = document_service.load_documents_from_directory(
                str(documents_dir)
            )

        if chunks:
            # Index documents
            logging.info("Indexing documents...")
            document_service.index_documents(chunks)

            return {
                "message": f"Successfully refreshed memory with documents from ./documents/",
                "documents_processed": len(chunks),
            }

    except Exception as e:
        logging.exception(f"Refresh memory failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to refresh memory: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
