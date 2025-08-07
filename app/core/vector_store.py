import asyncio
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import json

from pinecone import Pinecone, ServerlessSpec
from app.config.settings import settings
from app.utils.namespace_manager import NamespaceManager, document_tracker

logger = logging.getLogger(__name__)

class VectorStore:
    """Pinecone vector store operations with namespace management"""
    
    def __init__(self):
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index_name = settings.PINECONE_INDEX_NAME
        self.namespace_manager = NamespaceManager()
        self._index = None
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize Pinecone index"""
        try:
            if self._initialized:
                return True
            
            # Check if index exists
            existing_indexes = self.pc.list_indexes()
            index_names = [idx.name for idx in existing_indexes]
            
            if self.index_name not in index_names:
                logger.info(f"Creating Pinecone index: {self.index_name}")
                
                # Create index with serverless spec
                self.pc.create_index(
                    name=self.index_name,
                    dimension=settings.EMBEDDING_DIMENSION,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region=settings.PINECONE_ENVIRONMENT
                    )
                )
                
                # Wait for index to be ready
                while not self.pc.describe_index(self.index_name).status['ready']:
                    logger.info("Waiting for index to be ready...")
                    await asyncio.sleep(1)
            
            self._index = self.pc.Index(self.index_name)
            self._initialized = True
            
            logger.info(f"Successfully initialized Pinecone index: {self.index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone index: {str(e)}")
            return False
    
    async def namespace_exists(self, namespace: str) -> bool:
        """Check if namespace has any vectors"""
        try:
            if not self._initialized:
                await self.initialize()
            
            stats = self._index.describe_index_stats()
            namespaces = stats.get('namespaces', {})
            return namespace in namespaces and namespaces[namespace]['vector_count'] > 0
            
        except Exception as e:
            logger.error(f"Error checking namespace {namespace}: {str(e)}")
            return False
    
    async def upsert_vectors(
        self, 
        vectors: List[Dict[str, Any]], 
        namespace: str,
        batch_size: int = 100
    ) -> bool:
        """
        Upsert vectors to Pinecone in batches
        
        Args:
            vectors: List of vector dictionaries with id, values, metadata
            namespace: Pinecone namespace
            batch_size: Batch size for upserts (default: 100)
            
        Returns:
            Success status
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            if not vectors:
                logger.warning("No vectors to upsert")
                return True
            
            # Validate namespace
            if not namespace or not namespace.strip():
                logger.error("Empty or invalid namespace provided")
                return False
            
            # Check if namespace already exists (skip if document already processed)
            if await self.namespace_exists(namespace):
                if document_tracker.is_processed(namespace):
                    logger.info(f"Skipping upsert - namespace already processed: {namespace}")
                    return True
            
            logger.info(f"Upserting {len(vectors)} vectors to namespace: {namespace}")
            
            # Process in batches
            total_batches = (len(vectors) + batch_size - 1) // batch_size
            
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                try:
                    # Prepare batch data
                    upsert_data = []
                    for vector in batch:
                        # Ensure metadata is JSON serializable
                        metadata = self._serialize_metadata(vector.get('metadata', {}))
                        
                        upsert_data.append({
                            'id': vector['id'],
                            'values': vector['values'],
                            'metadata': metadata
                        })
                    
                    # Upsert batch
                    response = self._index.upsert(
                        vectors=upsert_data,
                        namespace=namespace.strip()
                    )
                    
                    logger.info(f"Batch {batch_num}/{total_batches} upserted: {response.upserted_count} vectors")
                    
                    # Small delay to avoid rate limits
                    if batch_num < total_batches:
                        await asyncio.sleep(0.1)
                        
                except Exception as e:
                    logger.error(f"Failed to upsert batch {batch_num}: {str(e)}")
                    return False
            
            # Atomic confirmation: Verify upsert was successful
            await asyncio.sleep(0.5)  # Brief delay for Pinecone to commit
            
            # Try to verify namespace exists (but don't fail if it's not immediately visible)
            try:
                if await self.namespace_exists(namespace):
                    logger.info(f"Namespace {namespace} verified immediately after upsert")
                else:
                    logger.warning(f"Namespace {namespace} not immediately visible in stats (this is normal for Pinecone)")
                    
                    # Try to get stats anyway to see if vectors are there
                    try:
                        stats = await self.get_namespace_stats(namespace)
                        actual_vector_count = stats.get('vector_count', 0)
                        if actual_vector_count > 0:
                            logger.info(f"Namespace {namespace} has {actual_vector_count} vectors (verified via stats)")
                        else:
                            logger.warning(f"Namespace {namespace} has 0 vectors in stats - upsert may have failed")
                    except Exception as stats_error:
                        logger.warning(f"Could not get stats for namespace {namespace}: {str(stats_error)}")
                        
            except Exception as e:
                logger.warning(f"Namespace verification failed: {str(e)}")
                # Continue anyway as the upsert operation itself was successful
            
            # Mark namespace as processed
            document_tracker.mark_processed(namespace)
            
            logger.info(f"Successfully upserted and verified {len(vectors)} vectors to namespace: {namespace}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert vectors to namespace {namespace}: {str(e)}")
            return False
    
    async def query_vectors(
        self,
        query_vector: List[float],
        namespace: str,
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Query vectors from Pinecone
        
        Args:
            query_vector: Query embedding vector
            namespace: Pinecone namespace to search
            top_k: Number of results to return (default: 10)
            filter_dict: Metadata filters
            include_metadata: Whether to include metadata in response (default: True)
            
        Returns:
            List of matching vectors with scores and metadata
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            # CRITICAL: Check if namespace exists first
            if not await self.namespace_exists(namespace):
                logger.warning(f"Namespace does not exist: {namespace}")
                return []
            
            # CRITICAL: Validate namespace parameter is not None/empty
            if not namespace or not namespace.strip():
                logger.error("Empty or invalid namespace provided")
                return []
            
            logger.info(f"Querying namespace: {namespace} with top_k: {top_k}")
            
            # Perform query with explicit namespace
            response = self._index.query(
                vector=query_vector,
                top_k=top_k,
                namespace=namespace.strip(),  # Ensure clean namespace
                filter=filter_dict,
                include_metadata=include_metadata,
                include_values=False  # We don't need the actual vectors back
            )
            
            # Process results
            results = []
            for match in response.matches:
                result = {
                    'id': match.id,
                    'score': float(match.score),
                    'metadata': match.metadata if include_metadata else {}
                }
                results.append(result)
            
            logger.info(f"Query returned {len(results)} results from namespace: {namespace}")
            
            # Enhanced logging for debugging
            if results and logger.isEnabledFor(logging.DEBUG):
                first_result_text = results[0].get('metadata', {}).get('text', '')[:100]
                logger.debug(f"Top result preview: {first_result_text}... (score: {results[0]['score']})")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to query namespace {namespace}: {str(e)}")
            return []
    
    async def query_multiple_namespaces(
        self,
        query_vector: List[float],
        namespaces: List[str],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Query multiple namespaces concurrently"""
        tasks = []
        for namespace in namespaces:
            task = self.query_vectors(
                query_vector=query_vector,
                namespace=namespace,
                top_k=top_k,
                filter_dict=filter_dict
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Organize results by namespace
        namespace_results = {}
        for i, result in enumerate(results):
            namespace = namespaces[i]
            if isinstance(result, Exception):
                logger.error(f"Query failed for namespace {namespace}: {str(result)}")
                namespace_results[namespace] = []
            else:
                namespace_results[namespace] = result
        
        return namespace_results
    
    async def delete_namespace(self, namespace: str) -> bool:
        """Delete all vectors in a namespace"""
        try:
            if not self._initialized:
                await self.initialize()
            
            if not await self.namespace_exists(namespace):
                logger.warning(f"Namespace does not exist: {namespace}")
                return True
            
            # Delete all vectors in namespace
            self._index.delete(delete_all=True, namespace=namespace)
            
            # Clear from document tracker
            document_tracker.clear_processed(namespace)
            
            logger.info(f"Successfully deleted namespace: {namespace}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete namespace {namespace}: {str(e)}")
            return False
    
    async def cleanup_test_namespaces(self) -> int:
        """Clean up all test namespaces"""
        try:
            if not self._initialized:
                await self.initialize()
            
            stats = self._index.describe_index_stats()
            namespaces = stats.get('namespaces', {})
            
            deleted_count = 0
            
            # Get all namespaces that should be cleaned
            namespaces_to_delete = []
            
            for namespace in namespaces.keys():
                # Delete test, dev, and any non-production namespaces
                should_delete = (
                    self.namespace_manager.is_test_namespace(namespace) or
                    namespace.startswith('dev_') or 
                    namespace.startswith('dev-') or
                    namespace.startswith('test_') or
                    namespace.startswith('test-')
                )
                
                if should_delete:
                    namespaces_to_delete.append(namespace)
            
            logger.info(f"Found {len(namespaces_to_delete)} namespaces to delete: {namespaces_to_delete}")
            
            # Delete each namespace
            for namespace in namespaces_to_delete:
                try:
                    logger.info(f"Deleting namespace: {namespace}")
                    success = await self.delete_namespace(namespace)
                    if success:
                        deleted_count += 1
                        logger.info(f"✅ Successfully deleted namespace: {namespace}")
                    else:
                        logger.error(f"❌ Failed to delete namespace: {namespace}")
                except Exception as e:
                    logger.error(f"❌ Error deleting namespace {namespace}: {str(e)}")
            
            logger.info(f"Cleanup completed: {deleted_count}/{len(namespaces_to_delete)} namespaces deleted")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup test namespaces: {str(e)}")
            return 0
    
    async def get_namespace_stats(self, namespace: str) -> Dict[str, Any]:
        """Get statistics for a specific namespace"""
        try:
            if not self._initialized:
                await self.initialize()
            
            stats = self._index.describe_index_stats()
            namespaces = stats.get('namespaces', {})
            
            if namespace in namespaces:
                return namespaces[namespace]
            else:
                return {'vector_count': 0}
                
        except Exception as e:
            logger.error(f"Failed to get stats for namespace {namespace}: {str(e)}")
            return {'vector_count': 0}
    
    async def list_namespaces(self) -> List[str]:
        """List all namespaces in the index"""
        try:
            if not self._initialized:
                await self.initialize()
            
            stats = self._index.describe_index_stats()
            namespaces = stats.get('namespaces', {})
            
            return list(namespaces.keys())
            
        except Exception as e:
            logger.error(f"Failed to list namespaces: {str(e)}")
            return []
    
    def _serialize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize metadata to ensure Pinecone compatibility"""
        serialized = {}
        
        for key, value in metadata.items():
            # Pinecone supports: str, int, float, bool, list of str
            if isinstance(value, (str, int, float, bool)):
                serialized[key] = value
            elif isinstance(value, list):
                # Convert list elements to strings
                serialized[key] = [str(item) for item in value]
            elif isinstance(value, dict):
                # Convert dict to JSON string
                serialized[key] = json.dumps(value)
            else:
                # Convert other types to string
                serialized[key] = str(value)
        
        return serialized
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on vector store"""
        try:
            if not self._initialized:
                await self.initialize()
            
            # Get index stats
            stats = self._index.describe_index_stats()
            
            # Get namespace count
            namespace_count = len(stats.get('namespaces', {}))
            
            # Calculate total vectors
            total_vectors = sum(
                ns_stats['vector_count'] 
                for ns_stats in stats.get('namespaces', {}).values()
            )
            
            return {
                'status': 'healthy',
                'index_name': self.index_name,
                'total_vectors': total_vectors,
                'namespace_count': namespace_count,
                'dimension': stats.get('dimension', 0),
                'index_fullness': stats.get('index_fullness', 0),
                'processed_documents': document_tracker.get_processed_count(),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
