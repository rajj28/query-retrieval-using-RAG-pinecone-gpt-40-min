# app/config/settings.py
import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "LLM Query Retrieval System"
    VERSION: str = "1.0.0"
    
    # Authentication
    BEARER_TOKEN: str = Field(
        default="9983c23ad9589f18637ad0a121a05b797a1f6e62fd0ff08e30bc8aa164dd618c",
        description="API Bearer token"
    )
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = Field(..., description="OpenAI API key")
    EMBEDDING_MODEL: str = Field(default="text-embedding-3-large", description="OpenAI embedding model")
    LLM_MODEL: str = Field(default="gpt-4o-mini", description="OpenAI LLM model")
    MAX_TOKENS: int = Field(default=4000, description="Maximum tokens for LLM responses")
    TEMPERATURE: float = Field(default=0.1, description="LLM temperature for consistency")
    
    # Pinecone Configuration  
    PINECONE_API_KEY: str = Field(..., description="Pinecone API key")
    PINECONE_INDEX_NAME: str = Field(default="llm-retrieval-index", description="Pinecone index name")
    PINECONE_ENVIRONMENT: str = Field(default="us-east-1", description="Pinecone environment")
    EMBEDDING_DIMENSION: int = Field(default=3072, description="Embedding dimension for text-embedding-3-large")
    
    # Document Processing - OPTIMIZED VALUES
    CHUNK_SIZE: int = Field(default=800, description="Text chunk size for embeddings")
    CHUNK_OVERLAP: int = Field(default=300, description="Overlap between chunks")
    MAX_DOCUMENT_SIZE_MB: int = Field(default=50, description="Maximum document size in MB")
    
    # Storage Paths
    LOCAL_STORAGE_PATH: str = Field(default="./data/known_documents", description="Local document storage path")
    TEMP_STORAGE_PATH: str = Field(default="./data/temp", description="Temporary storage path")
    CACHE_STORAGE_PATH: str = Field(default="./data/cache", description="Cache storage path")
    
    # Performance Settings - OPTIMIZED VALUES
    MAX_CONCURRENT_DOWNLOADS: int = Field(default=5, description="Max concurrent document downloads")
    MAX_CONCURRENT_QUERIES: int = Field(default=3, description="Max concurrent queries")
    RETRIEVAL_TOP_K: int = Field(default=50, description="Top K results for initial retrieval")
    RERANK_TOP_K: int = Field(default=10, description="Top K results after re-ranking")
    
    # Environment
    ENVIRONMENT: str = Field(default="development", description="Environment: development, staging, production")
    DEBUG: bool = Field(default=True, description="Debug mode")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    
    # Server Configuration
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, description="Server port")
    
    # CORS Configuration
    ALLOWED_ORIGINS: list = Field(
        default=["http://localhost:3000", "http://localhost:8000", "http://127.0.0.1:8000"],
        description="Allowed CORS origins"
    )
    
    # Namespace Configuration
    USE_DETERMINISTIC_NAMESPACES: bool = Field(default=True, description="Use content-based namespaces")
    NAMESPACE_PREFIX: str = Field(default="dev", description="Namespace prefix (dev, test, prod)")
    ENABLE_NAMESPACE_CLEANUP: bool = Field(default=True, description="Enable namespace cleanup utilities")
    
    # Cache Configuration
    CACHE_TTL: int = Field(default=3600, description="Cache TTL in seconds (1 hour default)")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create global settings instance
settings = Settings()

# Derived configurations
def get_namespace_prefix() -> str:
    """Get namespace prefix based on environment"""
    env_mapping = {
        "development": "dev",
        "testing": "test", 
        "staging": "stage",
        "production": "prod"
    }
    return env_mapping.get(settings.ENVIRONMENT, "dev")

def is_production() -> bool:
    """Check if running in production"""
    return settings.ENVIRONMENT == "production"

def get_cache_ttl() -> int:
    """Get cache TTL based on environment"""
    # Use the CACHE_TTL setting if available, otherwise fall back to environment-based defaults
    if hasattr(settings, 'CACHE_TTL'):
        return settings.CACHE_TTL
    return 3600 if is_production() else 300  # 1 hour prod, 5 min dev