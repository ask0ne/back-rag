from typing import List, Dict, Any, Tuple
import numpy as np
import re
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from src.config import config
from src.models.schemas import SourceType
import logging

logger = logging.getLogger(__name__)


class RAGService:
    """Service for performing RAG-based question answering."""
    
    def __init__(self, vector_store: Chroma):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            base_url=config.OPENAI_API_BASE_URL,
            model=config.MODEL_NAME,
            temperature=config.TEMPERATURE,
            api_key=config.OPENAI_API_KEY
        )
        
        # Prompt templates
        self.qa_prompt = PromptTemplate(
            template="""You are a helpful assistant that answers questions based on the provided context.
Use only the information from the context to answer the question. If the context doesn't contain 
enough information to answer the question, say so clearly.

Context:
{context}

Question: {question}

Answer:""",
            input_variables=["context", "question"]
        )
        
        self.confidence_prompt = PromptTemplate(
            template="""Given the following question and answer, evaluate how confident you are that the answer
correctly and completely addresses the question based on the provided context.

Question: {question}
Answer: {answer}
Context: {context}

Rate your confidence on a scale of 0.0 to 1.0, where:
- 0.0-0.3: Very low confidence (answer is unclear, incomplete, or potentially incorrect)
- 0.4-0.6: Moderate confidence (answer is partially correct but may be incomplete)
- 0.7-0.8: High confidence (answer is accurate and mostly complete)
- 0.9-1.0: Very high confidence (answer is accurate, complete, and well-supported)

Return only a single number between 0.0 and 1.0:""",
            input_variables=["question", "answer", "context"]
        )
    
    def retrieve_documents(self, query: str, k: int = None) -> List[Document]:
        """Retrieve relevant documents from the vector store."""
        if k is None:
            k = config.TOP_K_DOCUMENTS
        
        # Perform similarity search - let exceptions propagate
        documents = self.vector_store.similarity_search(query=query, k=k)
        logger.info(f"Retrieved {len(documents)} documents")
        return documents
    
    def retrieve_with_scores(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """Retrieve documents with similarity scores."""
        if k is None:
            k = config.TOP_K_DOCUMENTS
        
        # Perform similarity search with scores - let exceptions propagate
        results = self.vector_store.similarity_search_with_score(query=query, k=k)
        logger.info(f"Retrieved {len(results)} documents with scores")
        return results
    
    def calculate_retrieval_confidence(self, scores: List[float]) -> float:
        """Calculate confidence based on retrieval scores."""
        if not scores:
            return 0.0
        
        # Normalize scores (lower is better in similarity search)
        # Convert to confidence scores (higher is better)
        max_score = max(scores)
        min_score = min(scores)
        
        if max_score == min_score:
            # All scores are the same
            normalized_confidence = 0.5
        else:
            # Best score gets highest confidence
            best_score_confidence = 1.0 - (scores[0] / max_score)
            
            # Average confidence based on score distribution
            avg_score = sum(scores) / len(scores)
            avg_confidence = 1.0 - (avg_score / max_score)
            
            # Combine both factors
            normalized_confidence = (best_score_confidence + avg_confidence) / 2
        
        # Scale to reasonable range (0.1 to 0.9)
        scaled_confidence = 0.1 + (normalized_confidence * 0.8)
        
        return min(max(scaled_confidence, 0.0), 1.0)
    
    def _to_text(self, llm_output) -> str:
        """Normalize LLM output (AIMessage or str) to plain text."""
        if isinstance(llm_output, str):
            return llm_output
        content = getattr(llm_output, "content", None)
        return content if isinstance(content, str) else str(llm_output)

    def generate_answer(self, query: str, documents: List[Document]) -> str:
        """Generate an answer based on the retrieved documents."""
        if not documents:
            return "I don't have enough information in my knowledge base to answer this question."
        
        # Combine document content for context
        context_parts = []
        for i, doc in enumerate(documents, 1):
            title = doc.metadata.get('title', 'Unknown')
            source = doc.metadata.get('source', 'Unknown source')
            context_parts.append(
                f"[Document {i}] Title: {title}\nSource: {source}\nContent: {doc.page_content}\n"
            )
        
        context = "\n".join(context_parts)
        
        # Truncate context if too long
        if len(context) > config.MAX_CONTEXT_LENGTH:
            context = context[:config.MAX_CONTEXT_LENGTH] + "...[truncated]"
        
        # Generate answer using the LLM - let exceptions propagate
        prompt = self.qa_prompt.format(context=context, question=query)
        response = self.llm.invoke(prompt)
        text = self._to_text(response).strip()
        
        logger.info("Generated answer")
        return text
   
    def evaluate_answer_confidence(self, query: str, answer: str, context: str) -> float:
        """Evaluate the confidence of an answer using LLM."""
        prompt = self.confidence_prompt.format(
            question=query,
            answer=answer,
            context=context[:2000]  # Truncate context for confidence evaluation
        )
        
        response = self.llm.invoke(prompt)
        logger.debug(f"LLM confidence response: {response}")
        # Extract confidence score
        confidence_str = self._to_text(response).strip()
        try:
            # Try to extract number from text using regex
            match = re.search(r'(\d+\.?\d*)', confidence_str)
            if match:
                confidence = float(match.group(1))
            else:
                confidence = float(confidence_str)
            confidence = min(max(confidence, 0.0), 1.0)  # Clamp to [0, 1]
            logger.info(f"LLM confidence: {confidence:.3f}")
            return confidence
        except (ValueError, AttributeError):
            logger.warning(f"Could not parse confidence: {confidence_str}")
            return 0.5  # Default moderate confidence
            return 0.5  # Default moderate confidence
    
    def _is_insufficient_knowledge_response(self, answer: str) -> bool:
        """
        Detect if the answer indicates insufficient knowledge in the context.
        
        Returns True if the answer suggests the system doesn't have enough information.
        """
        # Common phrases that indicate lack of knowledge
        insufficient_phrases = [
            "don't have",
            "doesn't contain",
            "not enough information",
            "cannot find",
            "can't find",
            "no information",
            "insufficient information",
            "not available",
            "unable to answer",
            "cannot answer",
            "can't answer",
            "doesn't have",
            "do not have",
            "not mentioned",
            "not provided",
            "not specified",
            "context does not",
            "context doesn't"
        ]
        
        answer_lower = answer.lower()
        return any(phrase in answer_lower for phrase in insufficient_phrases)
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query through the complete RAG pipeline."""
        # Retrieve documents with scores - let exceptions propagate
        doc_score_pairs = self.retrieve_with_scores(query)
        
        if not doc_score_pairs:
            return {
                "answer": "I don't have any relevant information to answer this question.",
                "confidence": 0.0,
                "context_snippets": [],
                "retrieval_confidence": 0.0,
                "llm_confidence": 0.0,
                "documents_found": 0
            }
        
        documents = [doc for doc, _ in doc_score_pairs]
        scores = [score for _, score in doc_score_pairs]
        
        # Calculate retrieval confidence
        retrieval_confidence = self.calculate_retrieval_confidence(scores)
        
        # Generate answer
        answer = self.generate_answer(query, documents)
        
        # Check if the answer indicates insufficient knowledge
        # If so, assign very low confidence to trigger web search
        if self._is_insufficient_knowledge_response(answer):
            logger.info("Answer indicates insufficient knowledge - assigning low confidence")
            return {
                "answer": answer,
                "confidence": 0.1,  # Very low confidence to trigger web search
                "context_snippets": [doc.page_content[:200] + "..." for doc in documents[:3]],
                "retrieval_confidence": retrieval_confidence,
                "llm_confidence": 0.1,
                "documents_found": len(documents),
                "source_metadata": [
                    {
                        "title": doc.metadata.get('title', 'Unknown'),
                        "source": doc.metadata.get('source', 'Unknown'),
                        "score": score
                    }
                    for doc, score in doc_score_pairs[:3]
                ]
            }
        
        # Prepare context for confidence evaluation
        context_snippets = [doc.page_content[:200] + "..." for doc in documents[:3]]
        full_context = "\n".join(context_snippets)
        
        # Evaluate answer confidence
        llm_confidence = self.evaluate_answer_confidence(query, answer, full_context)
        
        # Combine confidences (weighted average)
        combined_confidence = (retrieval_confidence * 0.3) + (llm_confidence * 0.7)
        
        return {
            "answer": answer,
            "confidence": combined_confidence,
            "context_snippets": context_snippets,
            "retrieval_confidence": retrieval_confidence,
            "llm_confidence": llm_confidence,
            "documents_found": len(documents),
            "source_metadata": [
                {
                    "title": doc.metadata.get('title', 'Unknown'),
                    "source": doc.metadata.get('source', 'Unknown'),
                    "score": score
                }
                for doc, score in doc_score_pairs[:3]
            ]
        }