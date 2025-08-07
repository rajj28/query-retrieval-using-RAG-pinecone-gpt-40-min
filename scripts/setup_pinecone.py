# scripts/setup_pinecone.py
"""
Initialize Pinecone index for the LLM retrieval system
"""
import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.append(str(Path(__file__).parent.parent))

from pinecone import Pinecone, ServerlessSpec
from app.config.settings import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def setup_pinecone_index():
    """Setup Pinecone index with proper configuration"""
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        
        # Check if index already exists
        existing_indexes = pc.list_indexes()
        index_names = [idx.name for idx in existing_indexes]
        
        if settings.PINECONE_INDEX_NAME in index_names:
            logger.info(f"Index '{settings.PINECONE_INDEX_NAME}' already exists")
            
            # Get index info
            index_info = pc.describe_index(settings.PINECONE_INDEX_NAME)
            logger.info(f"Index dimension: {index_info.dimension}")
            logger.info(f"Index metric: {index_info.metric}")
            logger.info(f"Index status: {index_info.status}")
            
            return True
        
        # Create new index
        logger.info(f"Creating Pinecone index: {settings.PINECONE_INDEX_NAME}")
        logger.info(f"Dimension: {settings.EMBEDDING_DIMENSION}")
        logger.info(f"Environment: {settings.PINECONE_ENVIRONMENT}")
        
        pc.create_index(
            name=settings.PINECONE_INDEX_NAME,
            dimension=settings.EMBEDDING_DIMENSION,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region=settings.PINECONE_ENVIRONMENT
            )
        )
        
        # Wait for index to be ready
        logger.info("Waiting for index to be ready...")
        while True:
            index_info = pc.describe_index(settings.PINECONE_INDEX_NAME)
            if index_info.status['ready']:
                break
            logger.info("Index not ready yet, waiting...")
            await asyncio.sleep(2)
        
        logger.info("✅ Pinecone index created successfully!")
        
        # Test index connection
        index = pc.Index(settings.PINECONE_INDEX_NAME)
        stats = index.describe_index_stats()
        logger.info(f"Index stats: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to setup Pinecone index: {str(e)}")
        return False

if __name__ == "__main__":
    success = asyncio.run(setup_pinecone_index())
    sys.exit(0 if success else 1)


# scripts/cleanup_vectors.py
"""
Cleanup utility for test vectors and namespaces
"""
import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from app.core.vector_store import VectorStore
from app.utils.namespace_manager import NamespaceManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def cleanup_test_vectors():
    """Clean up all test namespaces"""
    try:
        vector_store = VectorStore()
        namespace_manager = NamespaceManager()
        
        # Initialize vector store
        success = await vector_store.initialize()
        if not success:
            logger.error("Failed to initialize vector store")
            return False
        
        # Get all namespaces
        namespaces = await vector_store.list_namespaces()
        logger.info(f"Found {len(namespaces)} namespaces")
        
        # Filter test namespaces
        test_namespaces = [
            ns for ns in namespaces 
            if namespace_manager.is_test_namespace(ns) or ns.startswith('dev_') or ns.startswith('test_')
        ]
        
        logger.info(f"Found {len(test_namespaces)} test namespaces to clean up")
        
        if not test_namespaces:
            logger.info("No test namespaces to clean up")
            return True
        
        # Confirm deletion
        if len(sys.argv) > 1 and sys.argv[1] == '--force':
            confirm = 'y'
        else:
            print(f"About to delete {len(test_namespaces)} test namespaces:")
            for ns in test_namespaces:
                print(f"  - {ns}")
            confirm = input("Continue? (y/N): ").lower()
        
        if confirm != 'y':
            logger.info("Cleanup cancelled")
            return True
        
        # Delete namespaces
        deleted_count = 0
        for namespace in test_namespaces:
            try:
                success = await vector_store.delete_namespace(namespace)
                if success:
                    deleted_count += 1
                    logger.info(f"✅ Deleted namespace: {namespace}")
                else:
                    logger.error(f"❌ Failed to delete namespace: {namespace}")
            except Exception as e:
                logger.error(f"❌ Error deleting namespace {namespace}: {str(e)}")
        
        logger.info(f"✅ Cleanup completed: {deleted_count}/{len(test_namespaces)} namespaces deleted")
        return True
        
    except Exception as e:
        logger.error(f"❌ Cleanup failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = asyncio.run(cleanup_test_vectors())
    sys.exit(0 if success else 1)


# scripts/benchmark_performance.py
"""
Performance benchmarking script
"""
import asyncio
import time
import sys
from pathlib import Path
import statistics

sys.path.append(str(Path(__file__).parent.parent))

from app.services.retrieval_service import RetrievalService
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample questions for benchmarking
BENCHMARK_QUESTIONS = [
    "What is the grace period for premium payment?",
    "What is the waiting period for pre-existing diseases?",
    "Does this policy cover maternity expenses?",
    "What is the waiting period for cataract surgery?",
    "Are medical expenses for organ donors covered?",
    "What is the No Claim Discount offered?",
    "Is there a benefit for preventive health check-ups?",
    "How does the policy define a Hospital?",
    "What is the coverage for AYUSH treatments?",
    "Are there sub-limits on room rent and ICU charges?"
]

# Sample document URL for testing
SAMPLE_DOCUMENT_URL = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"

async def benchmark_document_processing(retrieval_service, document_url: str, iterations: int = 3):
    """Benchmark document processing performance"""
    logger.info(f"Benchmarking document processing ({iterations} iterations)")
    
    processing_times = []
    
    for i in range(iterations):
        logger.info(f"Processing iteration {i+1}/{iterations}")
        
        start_time = time.time()
        
        try:
            # Force reprocess each time for accurate measurement
            await retrieval_service.reprocess_document(document_url)
            
            end_time = time.time()
            processing_time = end_time - start_time
            processing_times.append(processing_time)
            
            logger.info(f"Iteration {i+1}: {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error in iteration {i+1}: {str(e)}")
    
    if processing_times:
        avg_time = statistics.mean(processing_times)
        min_time = min(processing_times)
        max_time = max(processing_times)
        
        logger.info(f"Document Processing Results:")
        logger.info(f"  Average: {avg_time:.2f}s")
        logger.info(f"  Min: {min_time:.2f}s")
        logger.info(f"  Max: {max_time:.2f}s")
        
        return {
            'average': avg_time,
            'min': min_time,
            'max': max_time,
            'times': processing_times
        }
    
    return None

async def benchmark_query_processing(
    retrieval_service, 
    document_url: str, 
    questions: list, 
    iterations: int = 5
):
    """Benchmark query processing performance"""
    logger.info(f"Benchmarking query processing ({iterations} iterations, {len(questions)} questions each)")
    
    # Ensure document is processed first
    await retrieval_service._process_document(document_url, skip_processing=True)
    
    query_times = []
    accuracy_scores = []
    
    for i in range(iterations):
        logger.info(f"Query iteration {i+1}/{iterations}")
        
        start_time = time.time()
        
        try:
            result = await retrieval_service.process_documents_and_queries(
                document_url=document_url,
                questions=questions,
                skip_processing=True
            )
            
            end_time = time.time()
            total_time = end_time - start_time
            avg_time_per_query = total_time / len(questions)
            
            query_times.append(avg_time_per_query)
            
            # Calculate basic accuracy metrics (this is simplified)
            successful_queries = result.get('processing_stats', {}).get('successful_queries', 0)
            accuracy = successful_queries / len(questions) if questions else 0
            accuracy_scores.append(accuracy)
            
            logger.info(f"Iteration {i+1}: {avg_time_per_query:.2f}s per query, {accuracy:.1%} success rate")
            
        except Exception as e:
            logger.error(f"Error in query iteration {i+1}: {str(e)}")
    
    if query_times:
        avg_query_time = statistics.mean(query_times)
        min_query_time = min(query_times)
        max_query_time = max(query_times)
        avg_accuracy = statistics.mean(accuracy_scores)
        
        logger.info(f"Query Processing Results:")
        logger.info(f"  Average time per query: {avg_query_time:.2f}s")
        logger.info(f"  Min time per query: {min_query_time:.2f}s")
        logger.info(f"  Max time per query: {max_query_time:.2f}s")
        logger.info(f"  Average accuracy: {avg_accuracy:.1%}")
        
        return {
            'average_time': avg_query_time,
            'min_time': min_query_time,
            'max_time': max_query_time,
            'average_accuracy': avg_accuracy,
            'times': query_times,
            'accuracies': accuracy_scores
        }
    
    return None

async def run_comprehensive_benchmark():
    """Run comprehensive performance benchmark"""
    logger.info("🚀 Starting comprehensive performance benchmark")
    
    try:
        # Initialize retrieval service
        retrieval_service = RetrievalService()
        success = await retrieval_service.initialize()
        
        if not success:
            logger.error("Failed to initialize retrieval service")
            return False
        
        # Health check first
        health = await retrieval_service.health_check()
        if health['status'] != 'healthy':
            logger.warning(f"Service health issues detected: {health}")
        
        total_start_time = time.time()
        
        # Benchmark 1: Document Processing
        doc_benchmark = await benchmark_document_processing(
            retrieval_service, 
            SAMPLE_DOCUMENT_URL,
            iterations=3
        )
        
        # Benchmark 2: Query Processing
        query_benchmark = await benchmark_query_processing(
            retrieval_service,
            SAMPLE_DOCUMENT_URL,
            BENCHMARK_QUESTIONS,
            iterations=3
        )
        
        # Get final statistics
        final_stats = await retrieval_service._get_service_statistics()
        
        total_time = time.time() - total_start_time
        
        # Print comprehensive results
        logger.info("=" * 50)
        logger.info("📊 BENCHMARK RESULTS")
        logger.info("=" * 50)
        
        if doc_benchmark:
            logger.info(f"Document Processing:")
            logger.info(f"  ⚡ Average: {doc_benchmark['average']:.2f}s")
            logger.info(f"  🏃 Fastest: {doc_benchmark['min']:.2f}s")
            logger.info(f"  🐌 Slowest: {doc_benchmark['max']:.2f}s")
        
        if query_benchmark:
            logger.info(f"Query Processing:")
            logger.info(f"  ⚡ Average per query: {query_benchmark['average_time']:.2f}s")
            logger.info(f"  📊 Average accuracy: {query_benchmark['average_accuracy']:.1%}")
            logger.info(f"  🏃 Fastest query: {query_benchmark['min_time']:.2f}s")
        
        logger.info(f"Total benchmark time: {total_time:.2f}s")
        
        # Token usage and cost estimation
        if 'embedding_service' in final_stats:
            embedding_stats = final_stats['embedding_service']
            logger.info(f"Token Usage:")
            logger.info(f"  📈 Total requests: {embedding_stats.get('total_requests', 0)}")
            logger.info(f"  🔢 Total tokens: {embedding_stats.get('total_tokens', 0)}")
            logger.info(f"  💰 Estimated cost: ${embedding_stats.get('estimated_cost_usd', 0):.4f}")
        
        if 'llm_service' in final_stats:
            llm_stats = final_stats['llm_service']
            logger.info(f"  🤖 LLM requests: {llm_stats.get('total_requests', 0)}")
            logger.info(f"  💰 LLM cost: ${llm_stats.get('estimated_cost_usd', 0):.4f}")
        
        logger.info("=" * 50)
        logger.info("✅ Benchmark completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = asyncio.run(run_comprehensive_benchmark())
    sys.exit(0 if success else 1)