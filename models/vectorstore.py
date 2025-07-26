import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class FAISSVectorStore:
    """
    Enhanced FAISS-based vector store for multilingual document retrieval
    """
    
    def __init__(self, model_name: str, vector_db_path: str):
        self.model_name = model_name
        self.vector_db_path = vector_db_path
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        # FAISS components
        self.index: Optional[faiss.Index] = None
        self.documents: List[Document] = []
        self.document_metadata: List[dict] = []
        
        logger.info(f"Vector store initialized with dimension: {self.embedding_dimension}")
    
    def build_index(self, documents: List[Document], force_rebuild: bool = False) -> None:
        """
        Build or load FAISS index from documents
        
        Args:
            documents: List of documents to index
            force_rebuild: Whether to rebuild even if index exists
        """
        index_file = f"{self.vector_db_path}.index"
        metadata_file = f"{self.vector_db_path}.metadata"
        
        # Check if we can load existing index
        if not force_rebuild and os.path.exists(index_file) and os.path.exists(metadata_file):
            logger.info("Loading existing FAISS index...")
            self._load_index()
            return
        
        logger.info(f"Building new FAISS index for {len(documents)} documents...")
        
        # Store documents and metadata
        self.documents = documents
        self.document_metadata = [doc.metadata for doc in documents]
        
        # Extract texts for embedding
        texts = [doc.page_content for doc in documents]
        
        # Generate embeddings with progress bar
        logger.info("Generating embeddings...")
        embeddings = self.embedding_model.encode(
            texts, 
            show_progress_bar=True,
            batch_size=32
        )
        
        # Convert to numpy array and normalize for cosine similarity
        embeddings = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)  # L2 normalization for cosine similarity
        
        # Create FAISS index (using Inner Product for normalized vectors = cosine similarity)
        self.index = faiss.IndexFlatIP(self.embedding_dimension)
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        # Save index and metadata
        self._save_index()
        
        logger.info(f"FAISS index built successfully with {self.index.ntotal} vectors")
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5, 
        score_threshold: float = 0.0
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            k: Number of results to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of (document, score) tuples
        """
        if self.index is None:
            raise ValueError("Index not initialized. Call build_index() first.")
        
        logger.info(f"Searching for: '{query[:50]}...'")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.array(query_embedding, dtype=np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS index
        scores, indices = self.index.search(query_embedding, k)
        
        # Filter results by score threshold and prepare results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and score >= score_threshold:  # -1 means no result found
                document = self.documents[idx]
                results.append((document, float(score)))
        
        logger.info(f"Found {len(results)} relevant documents")
        return results
    
    def _save_index(self) -> None:
        """Save FAISS index and metadata to disk"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.vector_db_path), exist_ok=True)
        
        # Save FAISS index
        index_file = f"{self.vector_db_path}.index"
        faiss.write_index(self.index, index_file)
        
        # Save metadata and documents
        metadata_file = f"{self.vector_db_path}.metadata"
        with open(metadata_file, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'document_metadata': self.document_metadata,
                'model_name': self.model_name,
                'embedding_dimension': self.embedding_dimension
            }, f)
        
        logger.info(f"Index saved to {index_file} and {metadata_file}")
    
    def _load_index(self) -> None:
        """Load FAISS index and metadata from disk"""
        try:
            # Load FAISS index
            index_file = f"{self.vector_db_path}.index"
            self.index = faiss.read_index(index_file)
            
            # Load metadata and documents
            metadata_file = f"{self.vector_db_path}.metadata"
            with open(metadata_file, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.document_metadata = data['document_metadata']
                
                # Verify model compatibility
                if data['model_name'] != self.model_name:
                    logger.warning(f"Model mismatch: saved={data['model_name']}, current={self.model_name}")
            
            logger.info(f"Loaded index with {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            raise
    
    def get_statistics(self) -> dict:
        """Get vector store statistics"""
        if self.index is None:
            return {"status": "not_initialized"}
        
        return {
            "status": "initialized",
            "total_documents": len(self.documents),
            "index_size": self.index.ntotal,
            "embedding_dimension": self.embedding_dimension,
            "model_name": self.model_name
        }
    
    def delete_index(self) -> None:
        """Delete saved index files"""
        index_file = f"{self.vector_db_path}.index"
        metadata_file = f"{self.vector_db_path}.metadata"
        
        for file_path in [index_file, metadata_file]:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted {file_path}")