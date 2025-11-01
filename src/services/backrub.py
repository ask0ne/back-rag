"""
Backrub Service - Unified search service that handles RAG and web search.

This service provides a clean interface for searching with automatic fallback:
1. Queries the RAG system first
2. Checks if confidence meets threshold
3. Falls back to web search if confidence is too low
4. Returns a uniform response format
"""

from typing import Dict, Any, List
import logging
from src.services.rag_service import RAGService
from src.services.web_search import WebSearchService
from src.models.schemas import SourceType

logger = logging.getLogger(__name__)


class BackrubService:
    """Unified search service that intelligently routes between RAG and web search."""
    
    def __init__(self, rag_service: RAGService, web_search_service: WebSearchService):
        """
        Initialize the Backrub service.
        
        Args:
            rag_service: RAG service for document-based search
            web_search_service: Web search service for online queries
        """
        self.rag_service = rag_service
        self.web_search_service = web_search_service
        
    def search(self, query: str, threshold: float = 0.7) -> Dict[str, Any]:
        """
        Perform a search with automatic fallback to web if confidence is low.
        
        Args:
            query: The search query
            threshold: Confidence threshold (0.0 to 1.0). If RAG confidence is below
                      this threshold, falls back to web search.
        
        Returns:
            Dict containing:
                - source: SourceType (DOCUMENT or WEB)
                - answer: The answer text
                - confidence: Confidence score (0.0 to 1.0)
                - context_snippets: List of relevant context snippets
                - reasoning: Explanation of which source was used and why
        """
        logger.info(f"Backrub search initiated - query: '{query[:50]}...', threshold: {threshold}")
        
        # Step 1: Try RAG first
        rag_result = None
        rag_confidence = 0.0
        
        try:
            rag_result = self.rag_service.process_query(query)
            rag_confidence = rag_result.get("confidence", 0.0)
            
            logger.info(f"RAG confidence: {rag_confidence:.3f} (threshold: {threshold:.3f})")
            
            # Step 2: Check if confidence meets threshold
            if rag_confidence >= threshold:
                logger.info(f"RAG confidence sufficient ({rag_confidence:.3f} >= {threshold:.3f}), using document answer")
                
                return {
                    "source": SourceType.DOCUMENT,
                    "answer": rag_result["answer"],
                    "confidence": rag_confidence,
                    "context_snippets": rag_result.get("context_snippets", []),
                    "reasoning": f"Used document search (confidence: {rag_confidence:.2f} meets threshold: {threshold:.2f})"
                }
            else:
                logger.info(f"RAG confidence too low ({rag_confidence:.3f} < {threshold:.3f}), falling back to web search")
                
        except Exception as e:
            logger.error(f"RAG search failed: {e}, falling back to web search")
        
        # Step 3: Fall back to web search
        try:
            web_result = self.web_search_service.search_and_answer(query)
            
            # Check if web search returned valid results
            search_results_count = web_result.get("search_results_count", 0)
            
            if search_results_count == 0:
                # Web search failed (e.g., rate limit) - fallback to RAG answer even if confidence is low
                logger.warning(f"Web search returned no results (possibly rate limited), falling back to RAG answer")
                
                if rag_result:
                    return {
                        "source": SourceType.DOCUMENT,
                        "answer": rag_result["answer"],
                        "confidence": rag_confidence,
                        "context_snippets": rag_result.get("context_snippets", []),
                        "reasoning": f"Used document search (web search unavailable, RAG confidence: {rag_confidence:.2f})"
                    }
                else:
                    # Both failed
                    return {
                        "source": SourceType.DOCUMENT,
                        "answer": "I apologize, but I couldn't find relevant information in the documents, and web search is currently unavailable (possibly due to rate limiting). Please try again in a moment.",
                        "confidence": 0.0,
                        "context_snippets": [],
                        "reasoning": "Both RAG and web search unavailable"
                    }
            
            # Extract context snippets from web sources
            context_snippets = []
            sources = web_result.get("sources", [])
            for source in sources[:3]:
                snippet = source.get("snippet", "")
                if snippet:
                    context_snippets.append(snippet)
            
            # Web search doesn't have a confidence score, so we assign a default
            # High confidence since it's fresh from the web
            web_confidence = 0.8
            
            logger.info(f"Web search completed, found {len(sources)} sources")
            
            return {
                "source": SourceType.WEB,
                "answer": web_result["answer"],
                "confidence": web_confidence,
                "context_snippets": context_snippets,
                "reasoning": f"Used web search (RAG confidence was too low: {rag_confidence:.2f} < {threshold:.2f})"
            }
            
        except Exception as e:
            logger.error(f"Web search also failed: {e}")
            
            # Return RAG result if available, even with low confidence
            if rag_result:
                return {
                    "source": SourceType.DOCUMENT,
                    "answer": rag_result["answer"],
                    "confidence": rag_confidence,
                    "context_snippets": rag_result.get("context_snippets", []),
                    "reasoning": f"Used document search (web search failed, RAG confidence: {rag_confidence:.2f})"
                }
            
            # Both completely failed
            return {
                "source": SourceType.DOCUMENT,
                "answer": "I apologize, but I encountered errors with both document search and web search. Please try again or rephrase your query.",
                "confidence": 0.0,
                "context_snippets": [],
                "reasoning": f"Both RAG and web search failed. Errors encountered."
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the backrub service."""
        return {
            "service": "backrub",
            "description": "Unified search with RAG and web fallback",
            "version": "1.0.0"
        }
