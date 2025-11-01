from typing import List, Dict, Any
import os
import hashlib
import logging
from pathlib import Path
import shutil

import pypdf
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from src.config import config
from src.models.schemas import DocumentMetadata

logger = logging.getLogger(__name__)

class EmbeddingAdapter:
    """Adapter to make another embedding client compatible with LangChain."""
    def __init__(self, other_client):
        self.client = other_client
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Convert numpy array to list of lists
        embeddings = self.client.encode([text for text in texts])
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        # Convert numpy array to list
        embedding = self.client.encode(text)
        return embedding.tolist()


class DocumentIngestionService:
    """Service for ingesting documents into the vector database."""
    
    def __init__(self):
        self._set_embeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        self.vector_store = None
        self._init_vector_store()

    def _set_embeddings(self):
        other_client = SentenceTransformer(config.EMBEDDING_MODEL)
        self.embeddings = EmbeddingAdapter(other_client)
    
    def _init_vector_store(self):
        """Initialize the Chroma vector store."""
        try:
            os.makedirs(config.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
            
            # Import chromadb to configure settings
            import chromadb
            from chromadb.config import Settings
            
            # Create client with telemetry disabled
            chroma_client = chromadb.PersistentClient(
                path=config.CHROMA_PERSIST_DIRECTORY,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            self.vector_store = Chroma(
                client=chroma_client,
                embedding_function=self.embeddings,
                collection_name="rag_documents"
            )
            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file."""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"Failed to extract text from PDF {file_path}: {e}")
            raise
    
    def _extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from a text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Failed to extract text from TXT {file_path}: {e}")
            raise
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate a hash for the file to check for duplicates."""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load a single document and split it into chunks."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Extract text based on file type
        file_extension = file_path.suffix.lower()
        if file_extension == '.pdf':
            text = self._extract_text_from_pdf(str(file_path))
            file_type = "PDF"
        elif file_extension == '.txt':
            text = self._extract_text_from_txt(str(file_path))
            file_type = "TXT"
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        if not text.strip():
            raise ValueError(f"No text content found in file: {file_path}")
        
        # Create metadata
        metadata = {
            "title": file_path.stem,
            "source": str(file_path),
            "file_type": file_type,
            "file_hash": self._get_file_hash(str(file_path))
        }
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Create Document objects
        documents = []
        for i, chunk in enumerate(chunks):
            doc_metadata = metadata.copy()
            doc_metadata["chunk_index"] = i
            doc_metadata["total_chunks"] = len(chunks)
            
            documents.append(Document(
                page_content=chunk,
                metadata=doc_metadata
            ))
        
        logger.info(f"Loaded document {file_path.name} with {len(documents)} chunks")
        return documents
    
    def load_documents_from_directory(self, directory_path: str) -> List[Document]:
        """Load all supported documents from a directory."""
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        all_documents = []
        supported_extensions = {'.pdf', '.txt'}
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    documents = self.load_document(str(file_path))
                    all_documents.extend(documents)
                except Exception as e:
                    logger.error(f"Failed to load document {file_path}: {e}")
                    continue
        
        logger.info(f"Loaded {len(all_documents)} total document chunks from {directory}")
        return all_documents
    
    def index_documents(self, documents: List[Document]):
        """Index documents into the vector store"""
        try:
            # Filter out documents with empty content before indexing
            new_documents = [
                doc for doc in documents 
                if doc.page_content and doc.page_content.strip()
            ]
            
            if not new_documents:
                logger.warning("No valid documents to index after filtering empty content")
                return
            
            logger.info(f"Indexing {len(new_documents)} documents into vector store")
            
            # Simply add documents - embeddings will be generated automatically
            self.vector_store.add_documents(new_documents)
            
            logger.info(f"Successfully indexed {len(new_documents)} documents")
        except Exception as e:
            logger.error(f"Failed to index documents: {str(e)}")
            raise
    
    def clear_vector_store(self):
        """Clear all documents from the vector store."""
        try:
            persist_dir = Path(config.CHROMA_PERSIST_DIRECTORY)
            
            # Close any existing connections by setting vector_store to None
            self.vector_store = None
            
            # Remove all files from the persist directory if it exists
            if persist_dir.exists():
                shutil.rmtree(persist_dir)
                logger.info(f"Removed directory {config.CHROMA_PERSIST_DIRECTORY}")
            else:
                logger.info(f"Persist directory {config.CHROMA_PERSIST_DIRECTORY} does not exist, nothing to clear")
            
            # Reinitialize the vector store with fresh state
            self._init_vector_store()
            logger.info("Vector store cleared and reinitialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to clear vector store: {e}")
            raise
    
    def get_vector_store(self) -> Chroma:
        """Get the vector store instance."""
        return self.vector_store
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the document collection."""
        try:
            collection = self.vector_store.get()
            
            if not collection['metadatas']:
                return {"total_documents": 0, "total_chunks": 0, "unique_sources": 0}
            
            unique_sources = set(
                metadata.get('source') 
                for metadata in collection['metadatas'] 
                if metadata.get('source')
            )
            
            return {
                "total_documents": len(unique_sources),
                "total_chunks": len(collection['metadatas']),
                "unique_sources": len(unique_sources)
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}