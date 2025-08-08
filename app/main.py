from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import uvicorn
import uuid
from datetime import datetime
import os

# Only import settings for basic config, avoid heavy imports
try:
    from app.config.settings import settings
except ImportError:
    # Fallback settings for health check
    class FallbackSettings:
        PROJECT_NAME = "LLM Query Retrieval System"
        VERSION = "1.0.0"
        ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
        DEBUG = os.getenv("DEBUG", "false").lower() == "true"
        LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        API_V1_STR = "/api/v1"
        ALLOWED_ORIGINS = ["*"]
    
    settings = FallbackSettings()

# Configure basic logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="A system for processing insurance documents and answering queries using LLM and hybrid search",
    version=settings.VERSION,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None
)

# Add CORS middleware with basic settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)

# Lazy import of heavy dependencies only when needed
_retrieval_service = None
_hackrx_router = None

def get_hackrx_router():
    """Lazy import of hackrx router to avoid heavy imports during startup"""
    global _hackrx_router
    if _hackrx_router is None:
        try:
            from app.api.v1.routes.hackrx import router as hackrx_router
            _hackrx_router = hackrx_router
        except ImportError as e:
            logger.warning(f"Could not import hackrx router: {e}")
            return None
    return _hackrx_router

# Include routers only if they can be imported
try:
    hackrx_router = get_hackrx_router()
    if hackrx_router:
        app.include_router(hackrx_router, prefix=f"{settings.API_V1_STR}/hackrx", tags=["hackrx"])
except Exception as e:
    logger.warning(f"Could not include hackrx router: {e}")

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
async def health_check():
    """Simple health check endpoint for Vercel"""
    request_id = str(uuid.uuid4())
    try:
        # Basic health check without any heavy imports
        response = {
            'status': 'healthy',
            'components': {
                'api': 'healthy',
                'config': 'healthy'
            },
            'service_info': {
                'service': settings.PROJECT_NAME,
                'version': settings.VERSION,
                'environment': settings.ENVIRONMENT
            },
            'request_id': request_id,
            'timestamp': datetime.utcnow().isoformat(),
            'error': None
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

@app.get("/api/v1/hackrx/health")
async def hackrx_health_check():
    """Simple health check endpoint specifically for Vercel's /api/v1/hackrx/health route"""
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "timestamp": datetime.utcnow().isoformat()
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