import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import asyncio
from httpx import AsyncClient
from fastapi import FastAPI
from app.main import app
from app.services.retrieval_service import RetrievalService
from app.config.settings import settings

# Sample expected answers (unchanged)
EXPECTED_ANSWERS = {
    "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?": {
        "answer": "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
        "confidence_threshold": 0.8
    },
    # ... other questions unchanged ...
}

@pytest.fixture(scope="module")
async def client():
    """HTTP client for testing API endpoints"""
    client = AsyncClient(app=app, base_url="http://test")
    try:
        yield client
    finally:
        await client.aclose()

@pytest.fixture(scope="module")
async def retrieval_service():
    """Initialized RetrievalService for direct testing"""
    service = RetrievalService()
    await service.initialize()
    yield service
    if settings.ENVIRONMENT != "production":
        await service.cleanup_test_data()

@pytest.mark.asyncio
async def test_health_check(client):
    """Test system health check to ensure all components are running"""
    headers = {"Authorization": f"Bearer {settings.BEARER_TOKEN}"}
    response = await client.get("/api/v1/health", headers=headers)
    assert response.status_code == 200
    health_data = response.json()
    assert health_data["status"] == "healthy"
    assert health_data["components"]["embedding_service"]["status"] == "healthy"
    assert health_data["components"]["llm_service"]["status"] == "healthy"
    assert health_data["components"]["vector_store"]["status"] == "healthy"
    assert health_data["components"]["dual_index_manager"]["status"] == "healthy"
    assert "request_id" in health_data
    assert "timestamp" in health_data
    assert health_data["error"] is None

# ... rest of the tests unchanged ...

@pytest.mark.asyncio
async def test_dual_index_functionality(retrieval_service, client):
    """Test DualIndexManager indexing and retrieval"""
    if settings.ENVIRONMENT == "production":
        pytest.skip("Skipping index test in production environment")

    # Process a document
    document_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    questions = ["What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?"]
    
    payload = {
        "documents": [document_url],
        "questions": questions,
        "skip_processing": False
    }
    headers = {"Authorization": f"Bearer {settings.BEARER_TOKEN}"}
    response = await client.post("/api/v1/hackrx/run", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()

    # Verify indexing
    assert data["processing_stats"]["document_chunks"] > 0
    assert len(data["processing_stats"]["document_namespaces"]) == 1
    namespace = data["processing_stats"]["document_namespaces"][0]
    
    # Verify Pinecone index
    stats = await retrieval_service.vector_store.get_index_stats(namespace=namespace)
    assert stats["vector_count"] > 0
    
    # Verify metadata index
    metadata_stats = await retrieval_service.index_manager.get_index_stats(namespace=namespace)
    assert metadata_stats["entry_count"] > 0

    # Test retrieval
    query_result = await retrieval_service.search(
        query=questions[0],
        namespace=namespace,
        top_k=5
    )
    assert len(query_result["results"]) > 0
    assert query_result["results"][0]["score"] > 0.7  # Ensure relevant results
    assert "metadata" in query_result["results"][0]
    assert query_result["results"][0]["metadata"].get("chunk_type") in ["policy_text", "table", "metadata"]

@pytest.mark.asyncio
async def test_answer_accuracy(client):
    """Test answer accuracy for insurance-specific queries"""
    document_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    questions = list(EXPECTED_ANSWERS.keys())
    
    payload = {
        "documents": document_url,
        "questions": questions,
        "skip_processing": False
    }
    headers = {"Authorization": f"Bearer {settings.BEARER_TOKEN}"}
    response = await client.post("/api/v1/hackrx/run", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert len(data["answers"]) == len(questions)
    assert data["processing_stats"]["total_questions"] == len(questions)
    assert data["processing_stats"]["successful_queries"] >= len(questions) - 1  # Allow one failure
    assert "document_metadata" in data
    assert document_url in data["document_metadata"]
    assert "request_id" in data
    assert "timestamp" in data

    # Validate answer accuracy
    for answer_data in data["answers"]:
        question = answer_data["question"] if "question" in answer_data else questions[data["answers"].index(answer_data)]
        expected = EXPECTED_ANSWERS.get(question)
        assert expected is not None, f"No expected answer for question: {question}"
        assert answer_data["answer"] == expected["answer"], f"Mismatch for question: {question}"
        assert answer_data["confidence"] >= expected["confidence_threshold"], f"Low confidence for question: {question}"
        assert len(answer_data["sources"]) > 0, f"No sources provided for question: {question}"
        assert answer_data["error"] is None, f"Error in answer for question: {question}"

@pytest.mark.asyncio
async def test_cleanup_after_test(retrieval_service):
    """Test final cleanup to ensure indexes are cleared"""
    if settings.ENVIRONMENT == "production":
        pytest.skip("Skipping cleanup in production environment")

    cleanup_result = await retrieval_service.cleanup_test_data()
    assert cleanup_result["cache_cleared"] is True
    assert cleanup_result["metadata_cleanup"] is True
    assert cleanup_result["deleted_namespaces"] >= 0
    assert cleanup_result["error"] is None