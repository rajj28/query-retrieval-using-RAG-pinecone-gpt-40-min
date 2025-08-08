from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from typing import Dict, Any, List
import logging
import time
import uuid
from datetime import datetime

# Import models (lightweight)
from app.api.v1.models.request import HackRXRequest, DocumentInfoRequest
from app.api.v1.models.response import (
    HackRXResponse, DocumentInfoResponse, ServiceStatsResponse,
    CleanupResponse, HealthCheckResponse, SimpleHackRXResponse
)
# Remove direct import of RetrievalService
# from app.services.retrieval_service import RetrievalService
from app.config.settings import settings


logger = logging.getLogger(__name__)

router = APIRouter()
security = HTTPBearer()

async def get_retrieval_service():
    """Dependency to provide initialized RetrievalService with lazy loading"""
    try:
        # Import RetrievalService only when needed
        from app.services.retrieval_service import RetrievalService
        service = RetrievalService()
        # Don't initialize here - let it initialize when first used
        return service
    except Exception as e:
        logger.error(f"Failed to create RetrievalService: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail="Failed to create retrieval service"
        )

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify bearer token"""
    if credentials.credentials != settings.BEARER_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return credentials.credentials

@router.get(
    "/health",
    summary="Simple health check for Railway",
    description="Simple health check endpoint that doesn't require authentication"
)
async def simple_health_check():
    """Simple health check for Railway deployment"""
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get(
    "/health/detailed",
    response_model=HealthCheckResponse,
    summary="Detailed health check",
    description="Check the health status of all service components (requires authentication)"
)
async def detailed_health_check(
    service = Depends(get_retrieval_service)
) -> HealthCheckResponse:
    """Comprehensive health check for all services"""
    request_id = str(uuid.uuid4())
    try:
        health_result = await service.health_check()
        logger.info(f"[{request_id}] Health check result: {health_result}")
        
        # Ensure all required fields are present
        response_data = {
            'status': health_result.get('status', 'unknown'),
            'components': health_result.get('components', {}),
            'service_info': health_result.get('service_info', {}),
            'request_id': request_id,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Add error field if present
        if 'error' in health_result:
            response_data['error'] = health_result['error']
            
        return HealthCheckResponse(**response_data)
    except Exception as e:
        logger.error(f"[{request_id}] Health check failed: {str(e)}", exc_info=True)
        return HealthCheckResponse(
            status='unhealthy',
            components={},
            service_info={},
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat(),
            error=str(e)
        )

@router.post(
    "/hackrx/run",
    response_model=SimpleHackRXResponse,
    summary="Process documents and answer questions",
    description="Process multiple insurance documents and answer questions using hybrid search and LLM"
)
async def run_hackrx(
    request: HackRXRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
    service: RetrievalService = Depends(get_retrieval_service),
    token: str = Depends(verify_token)
) -> SimpleHackRXResponse:
    """
    Process multiple documents and answer questions

    This endpoint:
    1. Downloads and processes multiple insurance documents
    2. Creates embeddings and indexes them in Pinecone and metadata database
    3. Answers provided questions using hybrid search and context-aware LLM
    4. Returns structured answers with metadata
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        logger.info(f"[{request_id}] Processing HackRX request for {len(request.documents)} documents")
        logger.info(f"[{request_id}] Documents: {request.documents}")
        logger.info(f"[{request_id}] Questions: {len(request.questions)}")

        # Validate request
        if not request.documents:
            raise HTTPException(status_code=400, detail="At least one document URL is required")
        if len(request.documents) > settings.MAX_CONCURRENT_DOWNLOADS:
            raise HTTPException(
                status_code=400,
                detail=f"Too many documents. Maximum allowed: {settings.MAX_CONCURRENT_DOWNLOADS}"
            )

        # Process each document and collect results
        results = []
        for document_url in request.documents:
            result = await service.process_documents_and_queries(
                document_url=document_url,
                questions=request.questions,
                skip_processing=request.skip_processing
            )
            results.append(result)

        # Aggregate results
        aggregated_answers = []
        aggregated_metadata = {}
        aggregated_stats = {
            "total_questions": len(request.questions),
            "successful_queries": 0,
            "failed_queries": 0,
            "processing_time_seconds": 0.0,
            "document_namespaces": [],
            "document_chunks": 0,
            "tables_extracted": 0,
            "error_details": []
        }

        for result in results:
            if 'error' in result:
                aggregated_stats["error_details"].append(result['error'])
                aggregated_answers.extend(result['answers'])
                aggregated_stats["failed_queries"] += len(request.questions)
                continue

            aggregated_answers.extend(result['answers'])
            aggregated_metadata[result['document_metadata'].get('document_url', 'unknown')] = result['document_metadata']
            aggregated_stats["successful_queries"] += result['processing_stats']['successful_queries']
            aggregated_stats["failed_queries"] += result['processing_stats']['failed_queries']
            aggregated_stats["document_namespaces"].append(result['processing_stats']['document_namespace'])
            aggregated_stats["document_chunks"] += result['processing_stats']['document_chunks']
            aggregated_stats["tables_extracted"] += result['processing_stats']['tables_extracted']

        processing_time = time.time() - start_time
        aggregated_stats["processing_time_seconds"] = round(processing_time, 2)

        if aggregated_stats["failed_queries"] == len(request.questions) * len(request.documents):
            logger.error(f"[{request_id}] All queries failed: {aggregated_stats['error_details']}")
            raise HTTPException(
                status_code=500,
                detail=f"All queries failed: {aggregated_stats['error_details']}"
            )

        # Extract just the answer strings
        answer_strings = []
        for i, answer in enumerate(aggregated_answers):
            logger.info(f"Processing answer {i}: {type(answer)} - {answer}")
            
            if isinstance(answer, dict):
                # Handle different dictionary structures
                if 'answer' in answer:
                    answer_strings.append(str(answer['answer']))
                elif 'coverage' in answer:
                    answer_strings.append(str(answer['coverage']))
                elif 'NCD' in answer:
                    # Handle nested NCD structure
                    ncd_value = answer['NCD']
                    if isinstance(ncd_value, dict) and 'description' in ncd_value:
                        answer_strings.append(str(ncd_value['description']))
                    else:
                        answer_strings.append(str(ncd_value))
                elif 'benefit' in answer:
                    answer_strings.append(str(answer['benefit']))
                elif 'room_rent_limit' in answer:
                    answer_strings.append(str(answer['room_rent_limit']))
                else:
                    # For any other dictionary structure, try to extract meaningful text
                    # Look for any string values in the dictionary
                    text_parts = []
                    for key, value in answer.items():
                        if isinstance(value, str):
                            text_parts.append(value)
                        elif isinstance(value, dict):
                            # Recursively extract strings from nested dicts
                            for sub_key, sub_value in value.items():
                                if isinstance(sub_value, str):
                                    text_parts.append(sub_value)
                    
                    if text_parts:
                        answer_strings.append(" ".join(text_parts))
                    else:
                        # Fallback: convert entire dict to string
                        answer_strings.append(str(answer))
            elif isinstance(answer, str):
                answer_strings.append(answer)
            else:
                # For any other type, convert to string
                answer_strings.append(str(answer))
        
        logger.info(f"Extracted {len(answer_strings)} answer strings")
        for i, ans in enumerate(answer_strings):
            logger.info(f"Answer {i}: {ans[:100]}...")

        # Prepare simplified response
        response = SimpleHackRXResponse(
            answers=answer_strings
        )

        logger.info(f"[{request_id}] Successfully processed in {processing_time:.2f}s")

        # Log request metrics
        background_tasks.add_task(
            log_request_metrics,
            request_id=request_id,
            processing_time=processing_time,
            question_count=len(request.questions),
            document_count=len(request.documents),
            success=True
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"[{request_id}] Unexpected error: {str(e)}", exc_info=True)

        background_tasks.add_task(
            log_request_metrics,
            request_id=request_id,
            processing_time=processing_time,
            question_count=len(request.questions),
            document_count=len(request.documents),
            success=False,
            error=str(e)
        )

        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "detail": str(e),
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@router.post(
    "/test-simple",
    response_model=SimpleHackRXResponse,
    summary="Test simplified response format",
    description="Test endpoint that returns simplified response format without processing"
)
async def test_simple_response() -> SimpleHackRXResponse:
    """Test endpoint for simplified response format"""
    return SimpleHackRXResponse(
        answers=[
            "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
            "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.",
            "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period.",
            "The policy has a specific waiting period of two (2) years for cataract surgery.",
            "Yes, the policy indemnifies the medical expenses for the organ donor's hospitalization for the purpose of harvesting the organ, provided the organ is for an insured person and the donation complies with the Transplantation of Human Organs Act, 1994.",
            "A No Claim Discount of 5% on the base premium is offered on renewal for a one-year policy term if no claims were made in the preceding year. The maximum aggregate NCD is capped at 5% of the total base premium.",
            "Yes, the policy reimburses expenses for health check-ups at the end of every block of two continuous policy years, provided the policy has been renewed without a break. The amount is subject to the limits specified in the Table of Benefits.",
            "A hospital is defined as an institution with at least 10 inpatient beds (in towns with a population below ten lakhs) or 15 beds (in all other places), with qualified nursing staff and medical practitioners available 24/7, a fully equipped operation theatre, and which maintains daily records of patients.",
            "The policy covers medical expenses for inpatient treatment under Ayurveda, Yoga, Naturopathy, Unani, Siddha, and Homeopathy systems up to the Sum Insured limit, provided the treatment is taken in an AYUSH Hospital.",
            "Yes, for Plan A, the daily room rent is capped at 1% of the Sum Insured, and ICU charges are capped at 2% of the Sum Insured. These limits do not apply if the treatment is for a listed procedure in a Preferred Provider Network (PPN)."
        ]
    )

@router.post(
    "/document/info",
    response_model=DocumentInfoResponse,
    summary="Get document information",
    description="Get information about a processed document"
)
async def get_document_info(
    request: DocumentInfoRequest,
    service: RetrievalService = Depends(get_retrieval_service),
    token: str = Depends(verify_token)
) -> DocumentInfoResponse:
    """Get information about a processed document"""
    request_id = str(uuid.uuid4())
    try:
        logger.info(f"[{request_id}] Getting document info for {request.document_url}")
        info_result = await service.get_document_info(request.document_url)
        return DocumentInfoResponse(
            **info_result,
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.error(f"[{request_id}] Failed to get document info: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to get document info",
                "detail": str(e),
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@router.get(
    "/stats",
    response_model=ServiceStatsResponse,
    summary="Get service statistics",
    description="Get comprehensive service usage statistics"
)
async def get_service_stats(
    service: RetrievalService = Depends(get_retrieval_service),
    token: str = Depends(verify_token)
) -> ServiceStatsResponse:
    """Get service statistics"""
    request_id = str(uuid.uuid4())
    try:
        logger.info(f"[{request_id}] Fetching service statistics")
        stats_result = await service._get_service_statistics()
        return ServiceStatsResponse(
            **stats_result,
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.error(f"[{request_id}] Failed to get service stats: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to get service stats",
                "detail": str(e),
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@router.post(
    "/cleanup/test-data",
    response_model=CleanupResponse,
    summary="Clean up test data",
    description="Clean up test namespaces and cached data (development only)"
)
async def cleanup_test_data(
    service: RetrievalService = Depends(get_retrieval_service),
    token: str = Depends(verify_token)
) -> CleanupResponse:
    """Clean up test data - only allowed in development"""
    request_id = str(uuid.uuid4())
    try:
        if settings.ENVIRONMENT == "production":
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "Test data cleanup not allowed in production",
                    "request_id": request_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

        logger.info(f"[{request_id}] Initiating test data cleanup")
        cleanup_result = await service.cleanup_test_data()
        return CleanupResponse(
            **cleanup_result,
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Failed to cleanup test data: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to cleanup test data",
                "detail": str(e),
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@router.post(
    "/document/reprocess",
    response_model=DocumentInfoResponse,
    summary="Force reprocess document",
    description="Force reprocessing of a document (deletes existing data)"
)
async def reprocess_document(
    request: DocumentInfoRequest,
    service: RetrievalService = Depends(get_retrieval_service),
    token: str = Depends(verify_token)
) -> DocumentInfoResponse:
    """Force reprocessing of a document"""
    request_id = str(uuid.uuid4())
    try:
        if settings.ENVIRONMENT == "production":
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "Document reprocessing not allowed in production without additional safeguards",
                    "request_id": request_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

        logger.info(f"[{request_id}] Reprocessing document: {request.document_url}")
        reprocess_result = await service.reprocess_document(request.document_url)

        if reprocess_result.get('success'):
            return DocumentInfoResponse(
                exists=True,
                namespace=reprocess_result['namespace'],
                document_url=request.document_url,
                stats={
                    "vector_count": reprocess_result.get('chunk_count', 0),
                    "tables_extracted": reprocess_result.get('tables_extracted', 0)
                },
                is_processed=True,
                request_id=request_id,
                timestamp=datetime.utcnow().isoformat()
            )
        else:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": reprocess_result.get('error', 'Reprocessing failed'),
                    "request_id": request_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Failed to reprocess document: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to reprocess document",
                "detail": str(e),
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

async def log_request_metrics(
    request_id: str,
    processing_time: float,
    question_count: int,
    document_count: int,
    success: bool,
    error: str = None
):
    """Log request metrics for monitoring"""
    try:
        metrics = {
            'request_id': request_id,
            'timestamp': datetime.utcnow().isoformat(),
            'processing_time_seconds': round(processing_time, 2),
            'question_count': question_count,
            'document_count': document_count,
            'success': success,
            'error': error,
            'environment': settings.ENVIRONMENT
        }
        logger.info(f"Request metrics: {metrics}")
        # TODO: Integrate with Prometheus or a metrics service
    except Exception as e:
        logger.error(f"Failed to log request metrics: {str(e)}", exc_info=True)