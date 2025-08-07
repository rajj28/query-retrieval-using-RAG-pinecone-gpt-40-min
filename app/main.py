from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import uvicorn
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

from app.api.v1.routes.hackrx import router as hackrx_router
from app.services.retrieval_service import RetrievalService
from app.config.settings import settings

# Configure structured logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize RetrievalService
retrieval_service = RetrievalService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with service initialization"""
    # Startup
    logger.info(f"Starting LLM Query Retrieval System (Environment: {settings.ENVIRONMENT}, Debug: {settings.DEBUG})")
    
    # Initialize RetrievalService
    try:
        if not await retrieval_service.initialize():
            logger.error("Failed to initialize RetrievalService")
            raise RuntimeError("RetrievalService initialization failed")
        
        # Warm up embedding cache if needed
        if settings.ENVIRONMENT != "production":
            common_queries = [
                "What is the coverage amount?",
                "What are the exclusions in the policy?",
                "How to file a claim?"
            ]
            await retrieval_service.embedding_manager.warm_up_cache(common_queries)
            logger.info("Embedding cache warmed up with common queries")
        
        # Perform initial health check
        initial_health = await retrieval_service.health_check()
        logger.info(f"Initial health check: {initial_health['status']}")
        if initial_health['status'] != 'healthy':
            logger.warning(f"Some components are unhealthy: {initial_health['components']}")
    
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

    yield
    
    # Shutdown
    logger.info("Shutting down LLM Query Retrieval System...")
    # Clean up test data in non-production environments
    if settings.ENVIRONMENT != "production" and settings.ENABLE_NAMESPACE_CLEANUP:
        await retrieval_service.cleanup_test_data()
        logger.info("Test data cleanup completed")

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="A system for processing insurance documents and answering queries using LLM and hybrid search",
    version=settings.VERSION,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan
)

# Add CORS middleware with stricter settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Restrict methods
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],  # Restrict headers
)

# Include routers
app.include_router(hackrx_router, prefix=settings.API_V1_STR, tags=["hackrx"])

# Dependency for RetrievalService
async def get_retrieval_service() -> RetrievalService:
    """Dependency to provide initialized RetrievalService"""
    if not retrieval_service._initialized:
        await retrieval_service.initialize()
    return retrieval_service

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": f"{settings.PROJECT_NAME} API",
        "version": settings.VERSION,
        "status": "running",
        "environment": settings.ENVIRONMENT,
        "docs": "/docs" if settings.DEBUG else "Documentation disabled in production"
    }

@app.get("/health")
async def health_check(service: RetrievalService = Depends(get_retrieval_service)):
    """Comprehensive health check endpoint"""
    request_id = str(uuid.uuid4())
    try:
        health = await service.health_check()
        
        # Structure the response to match HealthCheckResponse model
        response = {
            'status': health.get('status', 'unknown'),
            'components': health.get('components', {}),
            'service_info': {
                'service': settings.PROJECT_NAME,
                'version': settings.VERSION,
                'environment': settings.ENVIRONMENT,
                **health.get('service_info', {})
            },
            'request_id': request_id,
            'timestamp': datetime.utcnow().isoformat(),
            'error': None  # Explicitly include error field as None for successful responses
        }
        return response
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            'status': 'unhealthy',
            'components': {},
            'service_info': {
                'service': settings.PROJECT_NAME,
                'version': settings.VERSION,
                'environment': settings.ENVIRONMENT
            },
            'request_id': request_id,
            'timestamp': datetime.utcnow().isoformat(),
            'error': str(e)
        }

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.DEBUG else "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=2 if settings.ENVIRONMENT == "production" else 1,  # Multiple workers in production
        timeout_keep_alive=30,  # Handle long-running requests
        log_level=settings.LOG_LEVEL.lower()
    )