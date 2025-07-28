from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
from contextlib import asynccontextmanager
import os
from pathlib import Path

# Import our custom modules
from config import config
from models.preprocessor import PDFPreprocessor
from models.chunker import IntelligentChunker
from models.vectorstore import FAISSVectorStore
from models.rag_pipeline import MultilingualRAGPipeline
from database.chat_history import ChatHistoryManager

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for the RAG system
rag_system: Optional[MultilingualRAGPipeline] = None
chat_history: Optional[ChatHistoryManager] = None

# Pydantic models for API
class QueryRequest(BaseModel):
    question: str = Field(..., description="Question in Bangla or English")
    session_id: str = Field(default="default", description="Session identifier")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")
    
class QueryResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    retrieved_chunks: int
    similarity_scores: List[float]
    session_id: str

class HistoryResponse(BaseModel):
    interactions: List[Dict[str, Any]]
    total_count: int

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup the RAG system"""
    global rag_system, chat_history
    
    try:
        logger.info("🚀 Initializing Multilingual RAG System with DeepSeek v3...")
        
        # Check if PDF file exists
        if not config.PDF_PATH.exists():
            raise FileNotFoundError(f"PDF file not found: {config.PDF_PATH}")
        
        # Initialize components
        logger.info("📄 Processing PDF document...")
        preprocessor = PDFPreprocessor()
        text = preprocessor.extract_text_from_pdf(config.PDF_PATH)
        
        # Get text statistics
        stats = preprocessor.get_text_statistics(text)
        logger.info(f"📊 Text statistics: {stats}")
        
        logger.info("✂️  Chunking document...")
        chunker = IntelligentChunker(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        chunks = chunker.chunk_document(text)
        
        # Get chunk statistics
        chunk_stats = chunker.get_chunk_statistics(chunks)
        logger.info(f"📊 Chunk statistics: {chunk_stats}")
        
        logger.info("🔍 Building vector index...")
        vectorstore = FAISSVectorStore(config.EMBEDDING_MODEL, config.VECTOR_DB_PATH)
        vectorstore.build_index(chunks, force_rebuild=False)
        
        logger.info("🤖 Initializing RAG pipeline with DeepSeek v3...")
        openrouter_config = {
            "api_key": config.OPENROUTER_API_KEY,
            "base_url": config.OPENROUTER_BASE_URL,
            "model": config.LLM_MODEL,
            "site_url": config.SITE_URL,
            "site_name": config.SITE_NAME
        }
        rag_system = MultilingualRAGPipeline(vectorstore, openrouter_config)
        
        # Test DeepSeek v3 connection
        logger.info("🔧 Testing DeepSeek v3 connection...")
        connection_test = rag_system.test_connection()
        if connection_test["status"] == "success":
            logger.info("✅ DeepSeek v3 connection successful!")
        else:
            logger.warning(f"⚠️  DeepSeek v3 connection issue: {connection_test.get('error', 'Unknown error')}")
        
        logger.info("💾 Initializing chat history...")
        chat_history = ChatHistoryManager(
            config.MONGO_URI, 
            config.MONGO_DB_NAME, 
            config.MONGO_COLLECTION
        )
        
        logger.info("✅ RAG system initialized successfully!")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG system: {e}")
        raise
    
    finally:
        logger.info("🔄 Shutting down RAG system...")

# Create FastAPI app
app = FastAPI(
    title="Multilingual RAG System - HSC Bangla Literature (DeepSeek v3)",
    description="A robust RAG system for answering questions about HSC Bangla literature using DeepSeek v3 via OpenRouter",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Multilingual RAG System with DeepSeek v3 is running! 🚀",
        "status": "healthy",
        "version": "1.0.0",
        "llm_model": config.LLM_MODEL,
        "supported_languages": ["বাংলা (Bangla)", "English"]
    }

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Ask a question to the RAG system using DeepSeek v3
    
    This endpoint accepts questions in both Bangla and English and returns
    contextually relevant answers based on the HSC Bangla literature corpus.
    """
    try:
        if not rag_system:
            raise HTTPException(status_code=500, detail="RAG system not initialized")
        
        logger.info(f"🔍 Processing question from session {request.session_id}")
        
        # Process the query
        result = rag_system.query(
            question=request.question,
            top_k=request.top_k,
            score_threshold=config.SIMILARITY_THRESHOLD
        )
        
        # Save to chat history
        if chat_history:
            chat_history.save_interaction(
                session_id=request.session_id,
                question=request.question,
                answer=result["answer"],
                confidence=result["confidence"],
                sources=result["sources"]
            )
        
        return QueryResponse(
            answer=result["answer"],
            confidence=result["confidence"],
            sources=result["sources"],
            retrieved_chunks=result["retrieved_chunks"],
            similarity_scores=result["similarity_scores"],
            session_id=request.session_id
        )
        
    except Exception as e:
        logger.error(f"❌ Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.get("/history/{session_id}", response_model=HistoryResponse)
async def get_session_history(session_id: str, limit: int = 20):
    """Get chat history for a specific session"""
    try:
        if not chat_history:
            raise HTTPException(status_code=503, detail="Chat history service not available")
        
        interactions = chat_history.get_session_history(session_id, limit)
        
        return HistoryResponse(
            interactions=interactions,
            total_count=len(interactions)
        )
        
    except Exception as e:
        logger.error(f"❌ Error retrieving history: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving history: {str(e)}")

@app.get("/sessions/{session_id}/stats")
async def get_session_stats(session_id: str):
    """Get statistics for a specific session"""
    try:
        if not chat_history:
            raise HTTPException(status_code=503, detail="Chat history service not available")
        
        stats = chat_history.get_session_statistics(session_id)
        return stats
        
    except Exception as e:
        logger.error(f"❌ Error retrieving session stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving stats: {str(e)}")

@app.get("/system/stats")
async def get_system_stats():
    """Get system statistics and health information"""
    try:
        stats = {
            "status": "healthy" if rag_system else "unhealthy",
            "pdf_file": str(config.PDF_PATH),
            "pdf_exists": config.PDF_PATH.exists(),
            "embedding_model": config.EMBEDDING_MODEL,
            "llm_model": config.LLM_MODEL,
            "openrouter_api_key": f"{config.OPENROUTER_API_KEY[:10]}...{config.OPENROUTER_API_KEY[-4:]}",  # Masked for security
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "similarity_threshold": config.SIMILARITY_THRESHOLD
        }
        
        if rag_system and hasattr(rag_system, 'vectorstore'):
            vector_stats = rag_system.vectorstore.get_statistics()
            stats.update(vector_stats)
        
        return stats
        
    except Exception as e:
        logger.error(f"❌ Error retrieving system stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving stats: {str(e)}")

@app.post("/system/rebuild-index")
async def rebuild_vector_index():
    """Rebuild the vector index (admin endpoint)"""
    try:
        if not rag_system:
            raise HTTPException(status_code=500, detail="RAG system not initialized")
        
        logger.info("🔄 Rebuilding vector index...")
        
        # Re-process PDF and rebuild index
        preprocessor = PDFPreprocessor()
        text = preprocessor.extract_text_from_pdf(config.PDF_PATH)
        
        chunker = IntelligentChunker(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        chunks = chunker.chunk_document(text)
        
        rag_system.vectorstore.build_index(chunks, force_rebuild=True)
        
        logger.info("✅ Vector index rebuilt successfully")
        
        return {
            "message": "Vector index rebuilt successfully",
            "chunks_processed": len(chunks)
        }
        
    except Exception as e:
        logger.error(f"❌ Error rebuilding index: {e}")
        raise HTTPException(status_code=500, detail=f"Error rebuilding index: {str(e)}")

@app.get("/test/connection")
async def test_deepseek_connection():
    """Test DeepSeek v3 connection via OpenRouter"""
    try:
        if not rag_system:
            raise HTTPException(status_code=500, detail="RAG system not initialized")
        
        result = rag_system.test_connection()
        return result
        
    except Exception as e:
        logger.error(f"❌ Error testing connection: {e}")
        raise HTTPException(status_code=500, detail=f"Error testing connection: {str(e)}")

# Test endpoints for the specific questions mentioned
@app.get("/test/sample-questions")
async def test_sample_questions():
    """Test the system with the sample questions from the requirements"""
    sample_questions = [
        "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
        "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?", 
        "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?"
    ]
    
    expected_answers = [
        "শুম্ভুনাথ",
        "মামাকে",
        "১৫ বছর"
    ]
    
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    results = []
    
    for question, expected in zip(sample_questions, expected_answers):
        try:
            result = rag_system.query(question, top_k=5, score_threshold=0.5)
            
            results.append({
                "question": question,
                "expected_answer": expected,
                "system_answer": result["answer"],
                "confidence": result["confidence"],
                "retrieved_chunks": result["retrieved_chunks"]
            })
            
        except Exception as e:
            results.append({
                "question": question,
                "expected_answer": expected,
                "system_answer": f"Error: {str(e)}",
                "confidence": 0.0,
                "retrieved_chunks": 0
            })
    
    return {
        "test_results": results,
        "total_questions": len(sample_questions),
        "model_used": config.LLM_MODEL
    }

if __name__ == "__main__":
    import uvicorn
    from config import DATA_DIR, VECTOR_DB_DIR
    # Ensure directories exist
    config.DATA_DIR.mkdir(exist_ok=True, parents=True)
    config.VECTOR_DB_DIR.mkdir(exist_ok=True, parents=True)
    
    logger.info(f"🚀 Starting server with DeepSeek v3 on {config.API_HOST}:{config.API_PORT}")
    logger.info(f"📖 Using model: {config.LLM_MODEL}")
    logger.info(f"🔑 API key: {config.OPENROUTER_API_KEY[:10]}...{config.OPENROUTER_API_KEY[-4:]}")
    
    uvicorn.run(
        "app:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=False,  # Set to True for development
        log_level=config.LOG_LEVEL.lower()
    )