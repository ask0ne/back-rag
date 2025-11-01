from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from src.models.schemas import AgentState, SourceType
from src.services.rag_service import RAGService
from src.services.web_search import WebSearchService
from src.config import config
import logging

logger = logging.getLogger(__name__)


class AgenticRAGAgent:
    """Agentic RAG system that decides between document retrieval and web search."""
    
    def __init__(self, rag_service: RAGService, web_search_service: WebSearchService):
        self.rag_service = rag_service
        self.web_search_service = web_search_service
        self.confidence_threshold = config.CONFIDENCE_THRESHOLD
        
        # Build the agent graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph for the agent."""
        
        # Define the workflow
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("rag_retrieval", self._rag_retrieval_node)
        workflow.add_node("confidence_check", self._confidence_check_node)
        workflow.add_node("web_search", self._web_search_node)
        workflow.add_node("format_response", self._format_response_node)
        
        # Set entry point
        workflow.set_entry_point("rag_retrieval")
        
        # Add edges
        workflow.add_edge("rag_retrieval", "confidence_check")
        workflow.add_conditional_edges(
            "confidence_check",
            self._should_search_web,
            {
                "web_search": "web_search",
                "format_response": "format_response"
            }
        )
        workflow.add_edge("web_search", "format_response")
        workflow.add_edge("format_response", END)
        
        # Compile the graph
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    def _rag_retrieval_node(self, state: AgentState) -> Dict[str, Any]:
        """Node for performing RAG retrieval and initial answer generation."""
        logger.info(f"RAG retrieval: {state.query}")
        
        reasoning_steps = state.reasoning_steps.copy()
        reasoning_steps.append("Starting RAG retrieval")
        
        # Process query through RAG pipeline - let exceptions propagate
        rag_result = self.rag_service.process_query(state.query)
        
        reasoning_steps.append(
            f"RAG completed. Confidence: {rag_result['confidence']:.3f}, Docs: {rag_result['documents_found']}"
        )
        
        logger.info(f"RAG confidence: {rag_result['confidence']:.3f}")
        
        # Return only updated fields as dict
        return {
            "rag_answer": rag_result["answer"],
            "rag_confidence": rag_result["confidence"],
            "context_snippets": rag_result["context_snippets"],
            "reasoning_steps": reasoning_steps
        }
    
    def _confidence_check_node(self, state: AgentState) -> Dict[str, Any]:
        """Node for checking if RAG confidence meets threshold."""
        logger.info(f"Confidence: {state.rag_confidence:.3f} (threshold: {self.confidence_threshold})")
        
        reasoning_steps = state.reasoning_steps.copy()
        
        if state.rag_confidence >= self.confidence_threshold:
            reasoning_steps.append(f"Confidence sufficient ({state.rag_confidence:.3f}). Using RAG answer")
            return {
                "final_source": SourceType.DOCUMENT,
                "reasoning_steps": reasoning_steps
            }
        else:
            reasoning_steps.append(f"Confidence low ({state.rag_confidence:.3f}). Triggering web search")
            return {
                "reasoning_steps": reasoning_steps
            }
        
        return state
    
    def _should_search_web(self, state: AgentState) -> Literal["web_search", "format_response"]:
        """Conditional edge function to determine next node."""
        if state.rag_confidence < self.confidence_threshold:
            return "web_search"
        else:
            return "format_response"
    
    def _web_search_node(self, state: AgentState) -> Dict[str, Any]:
        """Node for performing web search when RAG confidence is low."""
        logger.info("Starting web search")
        
        reasoning_steps = state.reasoning_steps.copy()
        reasoning_steps.append("Performing web search")
        
        # Perform web search - let exceptions propagate
        web_result = self.web_search_service.search_and_answer(state.query)
        
        # Add web sources to context snippets
        web_sources = [
            f"Web: {source['title']} - {source['snippet']}"
            for source in web_result.get("sources", [])
        ]
        context_snippets = state.context_snippets.copy()
        context_snippets.extend(web_sources)
        
        reasoning_steps.append(f"Web search completed. Found {web_result['search_results_count']} results")
        logger.info(f"Web search found {web_result['search_results_count']} results")
        
        return {
            "web_answer": web_result["answer"],
            "final_source": SourceType.WEB,
            "context_snippets": context_snippets,
            "reasoning_steps": reasoning_steps
        }
    
    def _format_response_node(self, state: AgentState) -> Dict[str, Any]:
        """Node for formatting the final response."""
        reasoning_steps = state.reasoning_steps.copy()
        reasoning_steps.append("Response ready")
        
        logger.info(f"Response complete. Source: {state.final_source}")
        
        return {
            "reasoning_steps": reasoning_steps
        }
    
    def process_query(self, query: str, confidence_threshold: float = None) -> Dict[str, Any]:
        """Process a query through the agentic RAG system."""
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
        
        logger.info(f"Processing query: {query}")
        
        # Initialize state
        initial_state = AgentState(
            query=query,
            reasoning_steps=[]
        )
        
        # Run the graph - let exceptions propagate
        config_dict = {"configurable": {"thread_id": "default"}}
        final_state = self.graph.invoke(initial_state.dict(), config_dict)
        
        # Access dict directly - final_state is already a dict
        final_source = final_state.get("final_source")
        rag_answer = final_state.get("rag_answer")
        rag_confidence = final_state.get("rag_confidence", 0.0)
        web_answer = final_state.get("web_answer")
        
        # Determine final answer and confidence
        if final_source == SourceType.DOCUMENT or final_source == "document":
            final_answer = rag_answer
            final_confidence = rag_confidence
        else:
            final_answer = web_answer
            final_confidence = 0.6  # Web search default confidence
        
        result = {
            "source": final_source.value if hasattr(final_source, 'value') else final_source or "document",
            "answer": final_answer or "No answer generated",
            "confidence": final_confidence,
            "context_snippets": final_state.get("context_snippets", []),
            "reasoning_trace": final_state.get("reasoning_steps", [])
        }
        
        logger.info(f"Query completed. Source: {result['source']}")
        return result
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get statistics about the agent's performance."""
        return {
            "confidence_threshold": self.confidence_threshold,
            "rag_service_available": self.rag_service is not None,
            "web_search_available": self.web_search_service is not None,
            "graph_nodes": list(self.graph.nodes.keys()) if hasattr(self.graph, 'nodes') else []
        }