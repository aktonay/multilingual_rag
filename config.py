import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DB_DIR = BASE_DIR / "vector_db"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
VECTOR_DB_DIR.mkdir(exist_ok=True)

class Config:
    # Add these two so config.DATA_DIR and config.VECTOR_DB_DIR exist
    DATA_DIR = DATA_DIR
    VECTOR_DB_DIR = VECTOR_DB_DIR

    # PDF Configuration
    PDF_PATH = DATA_DIR / "HSC26-Bangla1st-Paper.pdf"
    
    # Vector Database
    VECTOR_DB_PATH = str(VECTOR_DB_DIR / "faiss_index")
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    # OpenRouter API for DeepSeek v3 - Updated with your API key
    OPENROUTER_API_KEY = "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"
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    LLM_MODEL = "deepseek/deepseek-chat-v3-0324:free"
    
    # Optional headers for OpenRouter rankings
    SITE_URL = os.getenv("SITE_URL", "http://localhost:8000")  # Your site URL
    SITE_NAME = os.getenv("SITE_NAME", "Multilingual RAG System")  # Your site name
    
    # MongoDB Configuration
    MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://azon7550:cQgJmA0qUBP3Za9t@clusterrag.jwgt7zn.mongodb.net/?retryWrites=true&w=majority&appName=Clusterrag")
    MONGO_DB_NAME = "multilingual_rag"
    MONGO_COLLECTION = "chat_history"
    
    # RAG Parameters
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K_RETRIEVAL = 5
    SIMILARITY_THRESHOLD = 0.6
    
    # API Configuration
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    
    # Logging
    LOG_LEVEL = "INFO"

config = Config()
