import asyncio
from typing import List, Dict, Any, Optional
import logging
import time
from datetime import datetime

# Remove heavy imports from global scope
# from app.core.intelligent_document_processor import IntelligentDocumentProcessor
# from app.core.embedding_manager import EmbeddingManager
# from app.core.vector_store import VectorStore
# from app.core.dual_index_manager import DualIndexManager, SearchFilter
# from app.core.context_aware_llm import ContextAwareLLMClient
# from app.core.advanced_cache import AdvancedCache
# from app.utils.namespace_manager import NamespaceManager, document_tracker
from app.config.settings import settings

# Import SearchFilter for type hints
from app.core.dual_index_manager import SearchFilter

logger = logging.getLogger(__name__)

class RetrievalService:
    """Main orchestration service for document retrieval and query processing with lazy loading"""

    def __init__(self):
        # Initialize components as None - will be loaded lazily
        self._document_processor = None
        self._embedding_manager = None
        self._vector_store = None
        self._dual_index_manager = None
        self._llm_client = None
        self._namespace_manager = None
        self._advanced_cache = None
        self._initialized = False
        self._initialization_lock = asyncio.Lock()

    async def _load_components(self):
        """Lazy load all heavy components"""
        if self._initialized:
            return

        async with self._initialization_lock:
            if self._initialized:  # Double-check pattern
                return

            logger.info("Loading heavy ML components...")
            
            try:
                # Import heavy dependencies only when needed
                from app.core.intelligent_document_processor import IntelligentDocumentProcessor
                from app.core.embedding_manager import EmbeddingManager
                from app.core.vector_store import VectorStore
                from app.core.dual_index_manager import DualIndexManager
                from app.core.context_aware_llm import ContextAwareLLMClient
                from app.core.advanced_cache import AdvancedCache
                from app.utils.namespace_manager import NamespaceManager

                # Initialize components
                self._document_processor = IntelligentDocumentProcessor()
                self._embedding_manager = EmbeddingManager()
                self._vector_store = VectorStore()
                self._dual_index_manager = DualIndexManager()
                self._llm_client = ContextAwareLLMClient()
                self._namespace_manager = NamespaceManager()
                self._advanced_cache = AdvancedCache()

                # Clear corrupted search cache to fix deserialization issues
                self._advanced_cache.clear_search_cache()

                # Initialize components concurrently
                init_tasks = [
                    self._vector_store.initialize(),
                    self._dual_index_manager.initialize(),
                    self._embedding_manager.health_check(),
                    self._llm_client.health_check()
                ]
                results = await asyncio.gather(*init_tasks, return_exceptions=True)

                # Check initialization results
                vector_init, dual_index_init, embedding_health, llm_health = results

                if isinstance(vector_init, Exception) or not vector_init:
                    logger.error(f"Failed to initialize vector store: {vector_init}")
                    raise Exception(f"Vector store initialization failed: {vector_init}")
                if isinstance(dual_index_init, Exception) or not dual_index_init:
                    logger.error(f"Failed to initialize dual index manager: {dual_index_init}")
                    raise Exception(f"Dual index manager initialization failed: {dual_index_init}")
                if isinstance(embedding_health, Exception) or (isinstance(embedding_health, dict) and embedding_health.get('status') != 'healthy'):
                    logger.error(f"Embedding service unhealthy: {embedding_health}")
                    raise Exception(f"Embedding service unhealthy: {embedding_health}")
                if isinstance(llm_health, Exception) or (isinstance(llm_health, dict) and llm_health.get('status') != 'healthy'):
                    logger.error(f"LLM service unhealthy: {llm_health}")
                    raise Exception(f"LLM service unhealthy: {llm_health}")

                self._initialized = True
                logger.info("Retrieval service components loaded successfully")

            except Exception as e:
                logger.error(f"Failed to load retrieval service components: {str(e)}")
                raise

    @property
    def document_processor(self):
        """Lazy access to document processor"""
        if self._document_processor is None:
            raise RuntimeError("RetrievalService not initialized. Call _load_components() first.")
        return self._document_processor

    @property
    def embedding_manager(self):
        """Lazy access to embedding manager"""
        if self._embedding_manager is None:
            raise RuntimeError("RetrievalService not initialized. Call _load_components() first.")
        return self._embedding_manager

    @property
    def vector_store(self):
        """Lazy access to vector store"""
        if self._vector_store is None:
            raise RuntimeError("RetrievalService not initialized. Call _load_components() first.")
        return self._vector_store

    @property
    def dual_index_manager(self):
        """Lazy access to dual index manager"""
        if self._dual_index_manager is None:
            raise RuntimeError("RetrievalService not initialized. Call _load_components() first.")
        return self._dual_index_manager

    @property
    def llm_client(self):
        """Lazy access to LLM client"""
        if self._llm_client is None:
            raise RuntimeError("RetrievalService not initialized. Call _load_components() first.")
        return self._llm_client

    @property
    def namespace_manager(self):
        """Lazy access to namespace manager"""
        if self._namespace_manager is None:
            raise RuntimeError("RetrievalService not initialized. Call _load_components() first.")
        return self._namespace_manager

    @property
    def advanced_cache(self):
        """Lazy access to advanced cache"""
        if self._advanced_cache is None:
            raise RuntimeError("RetrievalService not initialized. Call _load_components() first.")
        return self._advanced_cache

    async def initialize(self) -> bool:
        """Initialize all components - now just calls _load_components"""
        try:
            await self._load_components()
            return True
        except Exception as e:
            logger.error(f"Failed to initialize retrieval service: {str(e)}")
            return False

    async def process_documents_and_queries(
        self,
        document_url: str,
        questions: List[str],
        skip_processing: bool = False
    ) -> Dict[str, Any]:
        """
        Main entry point for processing documents and answering queries

        Args:
            document_url: URL of document to process
            questions: List of questions to answer
            skip_processing: Skip document processing if already processed

        Returns:
            Dictionary with answers and processing metadata
        """
        try:
            start_time = time.time()

            # Load components if not already loaded
            if not self._initialized:
                await self._load_components()

            logger.info(f"Processing document: {document_url}")
            logger.info(f"Answering {len(questions)} questions")

            # Step 1: Process document
            document_result = await self._process_document(document_url, skip_processing)
            if not document_result:
                return {
                    'answers': ['Error: Failed to process document'] * len(questions),
                    'error': 'Document processing failed',
                    'processing_time': time.time() - start_time
                }

            namespace = document_result['namespace']
            document_metadata = document_result.get('document_metadata', {})

            # Step 2: Process queries concurrently
            query_tasks = []
            for question in questions:
                task = self._process_single_query(question, namespace, document_metadata)
                query_tasks.append(task)

            # Execute queries with controlled concurrency
            semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_QUERIES or 3)
            async def process_with_semaphore(task):
                async with semaphore:
                    return await task

            query_results = await asyncio.gather(
                *[process_with_semaphore(task) for task in query_tasks],
                return_exceptions=True
            )

            # Process results
            answers = []
            successful_queries = 0
            error_details = []

            for i, result in enumerate(query_results):
                if isinstance(result, Exception):
                    logger.error(f"Query {i} failed: {str(result)}")
                    answers.append({
                        'answer': f"Error processing query: {str(result)}",
                        'confidence': 0.0,
                        'error': True
                    })
                    error_details.append(f"Query {i}: {str(result)}")
                else:
                    answers.append(result)
                    if not result.get('error', False):
                        successful_queries += 1

            # Compile final result
            processing_time = time.time() - start_time

            final_result = {
                'answers': answers,
                'processing_stats': {
                    'total_questions': len(questions),
                    'successful_queries': successful_queries,
                    'failed_queries': len(questions) - successful_queries,
                    'processing_time_seconds': round(processing_time, 2),
                    'document_namespace': namespace,
                    'document_chunks': document_result.get('total_chunks', 0),
                    'tables_extracted': document_result.get('tables_extracted', 0),
                    'error_details': error_details if error_details else None
                },
                'document_metadata': document_metadata,
                'service_stats': await self._get_service_statistics()
            }

            logger.info(f"Completed processing in {processing_time:.2f}s")
            logger.info(f"Successfully answered {successful_queries}/{len(questions)} questions")

            return final_result

        except Exception as e:
            logger.error(f"Failed to process documents and queries: {str(e)}")
            return {
                'answers': [{'answer': f'Service error: {str(e)}', 'confidence': 0.0, 'error': True}] * len(questions),
                'error': str(e),
                'processing_time': time.time() - start_time if 'start_time' in locals() else 0
            }

    async def _process_document(
        self,
        document_url: str,
        skip_processing: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Process document and store embeddings and metadata"""
        try:
            # Generate namespace
            namespace = self.namespace_manager.create_namespace(
                source=document_url,
                document_type='insurance_policy'
            )

            # Check if already processed
            if skip_processing and await self.vector_store.namespace_exists(namespace):
                logger.info(f"Skipping processing - document already exists in namespace: {namespace}")
                stats = await self.dual_index_manager.get_document_statistics(namespace)
                return {
                    'namespace': namespace,
                    'document_metadata': stats,
                    'total_chunks': stats.get('total_chunks', 0),
                    'tables_extracted': stats.get('tables_count', 0)
                }

            # Download and extract text from document URL
            import aiohttp
            import PyPDF2
            import io
            
            async with aiohttp.ClientSession() as session:
                async with session.get(document_url) as response:
                    if response.status != 200:
                        logger.error(f"Failed to download document: HTTP {response.status}")
                        return None
                    
                    pdf_data = await response.read()
                    
                    # Extract text from PDF
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_data))
                    text_content = ""
                    for page in pdf_reader.pages:
                        text_content += page.extract_text()
            
            # Detect domain and get configuration
            from app.core.domain_detector import DomainDetector, DomainType
            detector = DomainDetector()
            
            # Create temporary metadata for domain detection
            temp_metadata = {
                'title': document_url,
                'document_text': text_content[:1000]  # Use first 1000 chars for detection
            }
            
            # Detect domain
            detected_domain = detector.detect_domain(text_content[:2000], temp_metadata)
            domain_config = detector.get_domain_config(detected_domain)
            
            logger.info(f"Processing document with domain: {detected_domain.value}")
            logger.info(f"Domain config: {domain_config}")
            
            # Process document using intelligent processor with domain-specific config
            document_result = await self.document_processor.process_document_intelligently(
                text=text_content,
                source=document_url,
                chunk_size=domain_config.get('chunk_size', 800),
                chunk_overlap=domain_config.get('chunk_overlap', 300)
            )

            if not document_result or not document_result.get('chunks'):
                logger.error("No chunks extracted from document")
                return None

            chunks = document_result['chunks']
            logger.info(f"Creating embeddings and indexing {len(chunks)} chunks")

            # Index chunks in dual-index system
            success = await self.dual_index_manager.index_document_chunks(chunks, namespace)

            if not success:
                logger.error("Failed to index chunks in dual-index system")
                return None

            # Namespace is now verified atomically in vector_store.upsert_vectors()

            logger.info(f"Successfully processed and indexed document in namespace: {namespace}")
            return {
                'namespace': namespace,
                'document_metadata': document_result['document_metadata'],
                'total_chunks': document_result['processing_stats']['total_chunks'],
                'tables_extracted': document_result['tables_extracted']
            }

        except Exception as e:
            logger.error(f"Failed to process document {document_url}: {str(e)}")
            return None

    async def _process_single_query(
        self,
        question: str,
        namespace: str,
        document_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a single query using hybrid search and context-aware LLM"""
        try:
            logger.info(f"Processing query: '{question[:50]}...'")

            # Step 1: Detect domain and enhance query
            domain = self._detect_domain(document_metadata)
            
            # Get domain-specific configuration
            from app.core.domain_detector import DomainDetector, DomainType
            detector = DomainDetector()
            domain_enum = DomainType(domain)
            domain_config = detector.get_domain_config(domain_enum)
            
            # Enhance query based on domain
            enhanced_query = question
            if domain_config.get('query_expansion', False):
                # Use query expansion for primary domains
                enhanced_query = await self._enhance_query_for_domain(question, domain)
            
            logger.info(f"Processing query for domain: {domain} (enhanced: {domain_config.get('query_expansion', False)})")

            # Step 2: Create query embedding
            query_embedding = await self.embedding_manager.embed_query(enhanced_query)
            if not query_embedding:
                logger.error("Failed to create query embedding")
                return {
                    'answer': 'Failed to process query - embedding creation failed',
                    'error': True,
                    'confidence': 0.0,
                    'reasoning': 'Embedding creation failed',
                    'sources': []
                }

            # Step 3: Perform hybrid search with domain-specific parameters
            search_filters = SearchFilter(
                policy_type=document_metadata.get('policy_type'),
                company_name=document_metadata.get('company_name'),
                policy_uin=document_metadata.get('policy_uin')
            )
            
            # Use domain-specific top_k
            top_k = domain_config.get('retrieval_top_k', settings.RETRIEVAL_TOP_K)
            
            search_results = await self.dual_index_manager.hybrid_search(
                query=enhanced_query,
                namespace=namespace,
                filters=search_filters,
                top_k=top_k
            )

            if not search_results:
                logger.warning(f"No results found for query in namespace: {namespace}")
                return {
                    'answer': 'No relevant information found in the document for this query.',
                    'confidence': 0.0,
                    'reasoning': 'No matching content found',
                    'sources': [],
                    'retrieval_stats': {
                        'initial_results': 0,
                        'reranked_results': 0,
                        'best_match_score': 0.0,
                        'enhanced_query': enhanced_query,
                        'domain': domain
                    }
                }

            # Step 4: Generate answer using context-aware LLM
            answer_result = await self.llm_client.process_query_with_context_awareness(
                query=question,
                search_results=search_results,
                document_metadata=document_metadata,
                filters_applied=search_filters
            )

            # Step 5: Enhance answer with retrieval metadata
            answer_result['retrieval_stats'] = {
                'initial_results': len(search_results),
                'reranked_results': len(search_results),
                'best_match_score': search_results[0].combined_score if search_results else 0.0,
                'enhanced_query': enhanced_query,
                'domain': domain
            }

            logger.info(f"Query processed successfully with confidence: {answer_result.get('confidence', 0.0)}")
            return answer_result

        except Exception as e:
            logger.error(f"Failed to process query '{question}': {str(e)}")
            return {
                'answer': f'Error processing query: {str(e)}',
                'error': True,
                'confidence': 0.0,
                'reasoning': f'Processing error: {str(e)}',
                'sources': [],
                'retrieval_stats': {
                    'initial_results': 0,
                    'reranked_results': 0,
                    'best_match_score': 0.0,
                    'enhanced_query': enhanced_query if 'enhanced_query' in locals() else question,
                    'domain': domain if 'domain' in locals() else 'unknown'
                }
            }

    def _detect_domain(self, document_metadata: Dict[str, Any]) -> str:
        """Detect document domain for query enhancement"""
        try:
            # Import domain detector
            from app.core.domain_detector import DomainDetector, DomainType
            
            # Create domain detector instance
            detector = DomainDetector()
            
            # Get document text for analysis
            document_text = document_metadata.get('document_text', '')
            if not document_text:
                # Fallback to metadata-based detection
                title = document_metadata.get('title', '')
                policy_type = document_metadata.get('policy_type', '')
                company_name = document_metadata.get('company_name', '')
                document_text = f"{title} {policy_type} {company_name}"
            
            # Detect domain
            detected_domain = detector.detect_domain(document_text, document_metadata)
            
            # Log domain detection
            logger.info(f"Detected domain: {detected_domain.value}")
            
            return detected_domain.value
            
        except Exception as e:
            logger.warning(f"Domain detection failed: {e}, using 'general'")
            return 'general'
    
    async def _enhance_query_for_domain(self, question: str, domain: str) -> str:
        """Enhance query based on domain-specific patterns"""
        try:
            # Domain-specific query enhancement
            if domain == 'insurance':
                # Add insurance-specific terms
                insurance_terms = ['policy', 'coverage', 'terms', 'conditions']
                enhanced = f"{question} (policy coverage terms conditions)"
            elif domain == 'legal':
                # Add legal-specific terms
                legal_terms = ['contract', 'agreement', 'clause', 'obligation']
                enhanced = f"{question} (contract agreement clause)"
            elif domain == 'hr':
                # Add HR-specific terms
                hr_terms = ['employee', 'benefits', 'policy', 'handbook']
                enhanced = f"{question} (employee benefits policy)"
            elif domain == 'compliance':
                # Add compliance-specific terms
                compliance_terms = ['regulation', 'compliance', 'requirement', 'standard']
                enhanced = f"{question} (regulation compliance requirement)"
            else:
                # For other domains, return original question
                enhanced = question
            
            logger.info(f"Enhanced query for {domain}: {enhanced}")
            return enhanced
            
        except Exception as e:
            logger.warning(f"Query enhancement failed: {e}")
            return question
    
    async def _get_service_statistics(self) -> Dict[str, Any]:
        """Get comprehensive service statistics"""
        try:
            # Get sync statistics directly
            embedding_stats = self.embedding_manager.get_statistics()
            llm_stats = self.llm_client.get_usage_statistics()
            
            # Get async statistics
            async_stats_tasks = [
                self.vector_store.health_check(),
                self.dual_index_manager.get_document_statistics('all_namespaces')
            ]
            async_results = await asyncio.gather(*async_stats_tasks, return_exceptions=True)
            
            vector_health, index_stats = async_results

            return {
                'embedding_service': embedding_stats if isinstance(embedding_stats, dict) else {'error': str(embedding_stats)},
                'llm_service': llm_stats if isinstance(llm_stats, dict) else {'error': str(llm_stats)},
                'vector_store': vector_health if isinstance(vector_health, dict) else {'error': str(vector_health)},
                'index_stats': index_stats if isinstance(index_stats, dict) else {'error': str(index_stats)},
                'processed_documents': 1,  # Simplified for now
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get service statistics: {str(e)}")
            return {'error': str(e), 'timestamp': datetime.utcnow().isoformat()}

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for all services"""
        try:
            if not self._initialized:
                await self._load_components() # Ensure components are loaded for health check

            # Check all components concurrently
            checks = await asyncio.gather(
                self.embedding_manager.health_check(),
                self.llm_client.health_check(),
                self.vector_store.health_check(),
                self.dual_index_manager.initialize(),  # Check index initialization
                return_exceptions=True
            )

            embedding_health, llm_health, vector_health, index_init = checks

            # Determine overall health
            all_healthy = all(
                (isinstance(check, dict) and check.get('status') == 'healthy') or (isinstance(check, bool) and check)
                for check in checks
            )

            # Ensure all components return proper dict structures
            components = {
                'embedding_service': embedding_health if isinstance(embedding_health, dict) else {'status': 'error', 'error': str(embedding_health)},
                'llm_service': llm_health if isinstance(llm_health, dict) else {'status': 'error', 'error': str(llm_health)},
                'vector_store': vector_health if isinstance(vector_health, dict) else {'status': 'error', 'error': str(vector_health)},
                'dual_index_manager': {'status': 'healthy' if index_init else 'error', 'error': str(index_init) if not index_init else None}
            }

            # Ensure service_info is always a dict
            service_info = {
                'initialized': self._initialized,
                'settings': {
                    'embedding_model': settings.EMBEDDING_MODEL,
                    'llm_model': settings.LLM_MODEL,
                    'pinecone_index': settings.PINECONE_INDEX_NAME,
                    'chunk_size': settings.CHUNK_SIZE,
                    'retrieval_top_k': settings.RETRIEVAL_TOP_K
                }
            }

            return {
                'status': 'healthy' if all_healthy else 'degraded',
                'components': components,
                'service_info': service_info,
                'error': None,  # Explicitly include error field as None for successful responses
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                'status': 'unhealthy',
                'components': {},
                'service_info': {
                    'initialized': self._initialized,
                    'settings': {
                        'embedding_model': settings.EMBEDDING_MODEL,
                        'llm_model': settings.LLM_MODEL,
                        'pinecone_index': settings.PINECONE_INDEX_NAME,
                        'chunk_size': settings.CHUNK_SIZE,
                        'retrieval_top_k': settings.RETRIEVAL_TOP_K
                    }
                },
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

    async def lightweight_health_check(self) -> Dict[str, Any]:
        """Lightweight health check that doesn't load heavy ML components"""
        try:
            return {
                'status': 'healthy',
                'components': {
                    'api': 'healthy',
                    'config': 'healthy',
                    'ml_components': 'not_loaded'  # Indicates lazy loading is working
                },
                'service_info': {
                    'initialized': self._initialized,
                    'lazy_loading': True,
                    'settings': {
                        'embedding_model': settings.EMBEDDING_MODEL,
                        'llm_model': settings.LLM_MODEL,
                        'pinecone_index': settings.PINECONE_INDEX_NAME
                    }
                },
                'error': None,
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Lightweight health check failed: {str(e)}")
            return {
                'status': 'unhealthy',
                'components': {},
                'service_info': {
                    'initialized': self._initialized,
                    'lazy_loading': True
                },
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

    async def cleanup_test_data(self) -> Dict[str, Any]:
        """Clean up test namespaces and data"""
        try:
            logger.info("Starting test data cleanup...")

            # Clean up test namespaces in Pinecone and metadata database
            cleanup_tasks = [
                self.vector_store.cleanup_test_namespaces(),
                self.dual_index_manager.cleanup_namespace('test_*')  # Assuming pattern-based cleanup
            ]
            deleted_namespaces, metadata_cleanup = await asyncio.gather(*cleanup_tasks, return_exceptions=True)

            # Clear embedding cache
            cache_cleared = self.embedding_manager.clear_cache()

            result = {
                'deleted_namespaces': deleted_namespaces if isinstance(deleted_namespaces, int) else 0,
                'metadata_cleanup': metadata_cleanup if isinstance(metadata_cleanup, bool) else False,
                'cache_cleared': cache_cleared,
                'timestamp': datetime.utcnow().isoformat()
            }

            logger.info(f"Test data cleanup completed: {result}")
            return result

        except Exception as e:
            logger.error(f"Test data cleanup failed: {str(e)}")
            return {
                'error': str(e),
                'deleted_namespaces': 0,
                'metadata_cleanup': False,
                'cache_cleared': False,
                'timestamp': datetime.utcnow().isoformat()
            }

    async def get_document_info(self, document_url: str) -> Dict[str, Any]:
        """Get information about a processed document"""
        try:
            # Generate namespace
            namespace = self.namespace_manager.create_namespace(
                source=document_url,
                document_type='insurance_policy'
            )

            # Check if document exists
            exists = await self.vector_store.namespace_exists(namespace)
            if not exists:
                return {
                    'exists': False,
                    'namespace': namespace,
                    'document_url': document_url,
                    'stats': {}
                }

            # Get statistics from dual-index manager
            stats = await self.dual_index_manager.get_document_statistics(namespace)

            return {
                'exists': True,
                'namespace': namespace,
                'document_url': document_url,
                'stats': stats,
                'is_processed': document_tracker.is_processed(namespace)
            }

        except Exception as e:
            logger.error(f"Failed to get document info for {document_url}: {str(e)}")
            return {
                'error': str(e),
                'exists': False,
                'namespace': '',
                'document_url': document_url,
                'stats': {}
            }

    async def reprocess_document(self, document_url: str) -> Dict[str, Any]:
        """Force reprocessing of a document"""
        try:
            logger.info(f"Force reprocessing document: {document_url}")

            # Generate namespace
            namespace = self.namespace_manager.create_namespace(
                source=document_url,
                document_type='insurance_policy'
            )

            # Delete existing data
            cleanup_tasks = [
                self.vector_store.delete_namespace(namespace),
                self.dual_index_manager.cleanup_namespace(namespace)
            ]
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

            # Clear from tracker
            document_tracker.clear_processed(namespace)

            # Reprocess document
            result = await self._process_document(document_url, skip_processing=False)

            if result:
                return {
                    'success': True,
                    'namespace': result['namespace'],
                    'chunk_count': result.get('total_chunks', 0),
                    'tables_extracted': result.get('tables_extracted', 0)
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to reprocess document'
                }

        except Exception as e:
            logger.error(f"Failed to reprocess document {document_url}: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }