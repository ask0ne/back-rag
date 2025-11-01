from typing import List, Dict, Any
import asyncio
from duckduckgo_search import DDGS
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from src.config import config
import logging

logger = logging.getLogger(__name__)


class WebSearchService:
    """Service for web search and answer generation."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            base_url=config.OPENAI_API_BASE_URL,
            model=config.MODEL_NAME,
            temperature=config.TEMPERATURE,
            api_key=config.OPENAI_API_KEY
        )
        
        self.summarization_prompt = PromptTemplate(
            template="""You are a helpful assistant that provides comprehensive answers based on web search results.

Question: {question}

Web Search Results:
{search_results}

Instructions:
1. Analyze all the search results provided
2. Synthesize the information to provide a comprehensive answer to the question
3. Focus on accuracy and relevance
4. If the search results don't contain sufficient information, mention this clearly
5. Include relevant details and context from the search results

Provide a clear, well-structured answer:""",
            input_variables=["question", "search_results"]
        )
    
    def search_web(self, query: str, max_results: int = None) -> List[Dict[str, Any]]:
        """Search the web using DuckDuckGo."""
        if max_results is None:
            max_results = config.WEB_SEARCH_MAX_RESULTS
        
        try:
            with DDGS() as ddgs:
                # Search for text results
                results = list(ddgs.text(
                    keywords=query,
                    max_results=max_results,
                    region='wt-wt',
                    safesearch='moderate',
                    timelimit='y'  # Results from past year
                ))
            
            logger.info(f"Found {len(results)} web search results for query: {query[:50]}...")
            return results
        
        except Exception as e:
            # Log the specific error but return empty list to allow graceful fallback
            logger.warning(f"Web search failed (possibly rate limited): {e}")
            return []
    
    def format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results for LLM processing."""
        if not results:
            return "No search results found."
        
        formatted_results = []
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            body = result.get('body', 'No content')
            href = result.get('href', 'No URL')
            
            formatted_results.append(
                f"[Result {i}]\n"
                f"Title: {title}\n"
                f"URL: {href}\n"
                f"Content: {body}\n"
            )
        
        return "\n".join(formatted_results)
    
    def _to_text(self, llm_output) -> str:
        """Normalize LLM output (AIMessage or str) to plain text."""
        if isinstance(llm_output, str):
            return llm_output
        content = getattr(llm_output, "content", None)
        return content if isinstance(content, str) else str(llm_output)

    def generate_web_answer(self, query: str, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate an answer based on web search results."""
        if not search_results:
            return {
                "answer": "■ This answer was found on the web.\n\nI couldn't find relevant information on the web to answer your question. Please try rephrasing your query or asking about a different topic.",
                "sources": [],
                "search_results_count": 0
            }
        
        try:
            # Format search results for LLM
            formatted_results = self.format_search_results(search_results)
            
            # Generate answer using LLM
            prompt = self.summarization_prompt.format(
                question=query,
                search_results=formatted_results
            )
            
            response = self.llm.invoke(prompt)
            
            # Add web source indicator
            answer = f"■ This answer was found on the web.\n\n{self._to_text(response).strip()}"
            
            # Extract source information
            sources = []
            for result in search_results:
                sources.append({
                    "title": result.get('title', 'Unknown Title'),
                    "url": result.get('href', 'Unknown URL'),
                    "snippet": result.get('body', '')[:200] + "..." if result.get('body') else ""
                })
            
            logger.info(f"Generated web answer for query: {query[:50]}...")
            
            return {
                "answer": answer,
                "sources": sources,
                "search_results_count": len(search_results)
            }
        
        except Exception as e:
            logger.error(f"Failed to summarize web answer: {e}")
            return {
                "answer": f"■ This answer was found on the web.\n\nFailed to summarize web search results.\n{answer}",
                "sources": sources,
                "search_results_count": len(search_results)
            }
    
    def search_and_answer(self, query: str) -> Dict[str, Any]:
        """Perform web search and generate an answer."""
        try:
            # Search the web
            search_results = self.search_web(query)
            
            # Generate answer from search results
            result = self.generate_web_answer(query, search_results)
            
            return result
        
        except Exception as e:
            logger.error(f"Web search and answer failed: {e}")
            return {
                "answer": f"■ This answer was found on the web.\n\nI encountered an error while searching the web: {str(e)}",
                "sources": [],
                "search_results_count": 0
            }
    
    async def async_search_and_answer(self, query: str) -> Dict[str, Any]:
        """Asynchronous version of search_and_answer."""
        # Run the synchronous method in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.search_and_answer, query)