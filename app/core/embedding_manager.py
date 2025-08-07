import asyncio
from typing import List, Dict, Any, Optional
import logging
import time
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import aiofiles
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import openai
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer
from app.config.settings import settings, get_cache_ttl
from app.core.fine_tuning_manager import FineTuningManager
from app.core.advanced_cache import advanced_cache

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manage OpenAI embeddings with optimization, persistent caching, and retry logic"""
    
    def __init__(self, use_fine_tuned: bool = True):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.EMBEDDING_MODEL
        self.cache_dir = Path(settings.CACHE_STORAGE_PATH) / "embeddings"
        self.cache = {}  # In-memory cache for quick access
        self.request_count = 0
        self.total_tokens = 0
        self.cache_hits = 0
        self.total_response_time = 0.0
        
        # Fine-tuning support
        self.use_fine_tuned = use_fine_tuned
        self.fine_tuning_manager = FineTuningManager()
        self.fine_tuned_model = None
        self._initialize_cache()
        self._initialize_fine_tuned_model()

    def _initialize_cache(self):
        """Initialize persistent cache directory"""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Initialized embedding cache directory: {self.cache_dir}")
            # Load existing cache files into memory
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    with open(cache_file, 'r') as f:
                        cache_data = json.load(f)
                        self.cache[cache_data['text_hash']] = cache_data['embedding']
                except Exception as e:
                    logger.warning(f"Failed to load cache file {cache_file}: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to initialize cache directory: {str(e)}")
    
    def _initialize_fine_tuned_model(self):
        """Initialize fine-tuned model if available"""
        if not self.use_fine_tuned:
            logger.info("Fine-tuned model disabled")
            return
        
        try:
            fine_tuned_path = self.fine_tuning_manager.get_fine_tuned_model_path()
            if fine_tuned_path:
                logger.info(f"Loading fine-tuned model from: {fine_tuned_path}")
                self.fine_tuned_model = SentenceTransformer(fine_tuned_path)
                logger.info("Fine-tuned model loaded successfully")
            else:
                logger.info("No fine-tuned model found, using base model")
        except Exception as e:
            logger.error(f"Failed to initialize fine-tuned model: {str(e)}")
            logger.info("Falling back to base model")

    async def _save_to_cache(self, text_hash: str, text: str, embedding: List[float]):
        """Save embedding to persistent cache"""
        try:
            cache_file = self.cache_dir / f"{text_hash}.json"
            cache_data = {
                'text_hash': text_hash,
                'text': text[:1000],  # Truncate for storage
                'embedding': embedding,
                'created_at': datetime.utcnow().isoformat(),
                'ttl_seconds': get_cache_ttl()
            }
            async with aiofiles.open(cache_file, 'w') as f:
                await f.write(json.dumps(cache_data))
            self.cache[text_hash] = embedding
        except Exception as e:
            logger.error(f"Failed to save embedding to cache: {str(e)}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((openai.APIError, openai.RateLimitError))
    )
    async def create_embedding(self, text: str) -> Optional[List[float]]:
        """Create embedding for single text with retry logic"""
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return None

        # Check advanced cache first (fastest)
        cached_embedding = advanced_cache.get_embedding(text)
        if cached_embedding:
            logger.debug("Advanced cache hit for embedding")
            self.cache_hits += 1
            return cached_embedding

        # Check legacy cache
        text_hash = self._get_text_hash(text)
        if text_hash in self.cache:
            logger.debug("Legacy cache hit for embedding")
            self.cache_hits += 1
            # Promote to advanced cache
            advanced_cache.cache_embedding(text, self.cache[text_hash])
            return self.cache[text_hash]

        try:
            start_time = time.time()
            response = await self.client.embeddings.create(
                model=self.model,
                input=text.strip(),
                encoding_format="float"
            )

            embedding = response.data[0].embedding

            # Update statistics
            self.request_count += 1
            self.total_tokens += response.usage.total_tokens
            self.total_response_time += time.time() - start_time

            # Cache in both systems
            await self._save_to_cache(text_hash, text, embedding)
            advanced_cache.cache_embedding(text, embedding)

            logger.debug(f"Created embedding with {len(embedding)} dimensions")
            return embedding

        except Exception as e:
            logger.error(f"Failed to create embedding: {str(e)}")
            return None
    
    async def create_embedding_fine_tuned(self, text: str) -> Optional[List[float]]:
        """Create embedding using fine-tuned model if available"""
        if not self.fine_tuned_model:
            logger.debug("No fine-tuned model available, using base model")
            return await self.create_embedding(text)
        
        try:
            # Check cache first
            text_hash = self._get_text_hash(text)
            cache_key = f"fine_tuned_{text_hash}"
            
            if cache_key in self.cache:
                logger.debug("Using cached fine-tuned embedding")
                self.cache_hits += 1
                return self.cache[cache_key]
            
            # Create embedding using fine-tuned model
            start_time = time.time()
            embedding = self.fine_tuned_model.encode(text.strip()).tolist()
            
            # Update statistics
            self.request_count += 1
            self.total_response_time += time.time() - start_time
            
            # Cache the result
            self.cache[cache_key] = embedding
            await self._save_to_cache(cache_key, text, embedding)
            
            logger.debug(f"Created fine-tuned embedding in {time.time() - start_time:.3f}s")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to create fine-tuned embedding: {str(e)}")
            # Fallback to base model
            return await self.create_embedding(text)

    async def create_embeddings_batch(
        self, 
        texts: List[str], 
        batch_size: int = 50  # Reduced for faster API response times
    ) -> List[Optional[List[float]]]:
        """
        Create embeddings for multiple texts in batches

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per batch (default: 100)

        Returns:
            List of embeddings (None for failed embeddings)
        """
        if not texts:
            return []

        # Filter out empty texts but maintain indices
        valid_texts = []
        text_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text.strip())
                text_indices.append(i)

        if not valid_texts:
            logger.warning("No valid texts provided for batch embedding")
            return [None] * len(texts)

        logger.info(f"Creating embeddings for {len(valid_texts)} texts in batches of {batch_size}")

        # Process in batches with concurrency control
        semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_QUERIES or 3)
        async def process_with_semaphore(batch_texts, batch_indices):
            async with semaphore:
                return await self._process_embedding_batch(batch_texts, batch_indices)

        all_embeddings = []
        for i in range(0, len(valid_texts), batch_size):
            batch_texts = valid_texts[i:i + batch_size]
            batch_indices = text_indices[i:i + batch_size]
            batch_embeddings = await process_with_semaphore(batch_texts, batch_indices)
            all_embeddings.extend(batch_embeddings)

            # Rate limiting - small delay between batches
            if i + batch_size < len(valid_texts):
                await asyncio.sleep(0.1)

        # Map results back to original indices
        result_embeddings = [None] * len(texts)
        for i, embedding in enumerate(all_embeddings):
            if i < len(text_indices):
                result_embeddings[text_indices[i]] = embedding

        logger.info(f"Successfully created {len([e for e in all_embeddings if e])} embeddings")
        return result_embeddings

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((openai.APIError, openai.RateLimitError))
    )
    async def _process_embedding_batch(self, texts: List[str], indices: List[int]) -> List[Optional[List[float]]]:
        """Process a single batch of texts"""
        try:
            # Check cache for all texts
            cached_embeddings = {}
            uncached_texts = []
            uncached_indices = []

            for i, text in enumerate(texts):
                text_hash = self._get_text_hash(text)
                if text_hash in self.cache:
                    cached_embeddings[i] = self.cache[text_hash]
                    self.cache_hits += 1
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)

            # Create embeddings for uncached texts
            new_embeddings = {}
            if uncached_texts:
                start_time = time.time()
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=uncached_texts,
                    encoding_format="float"
                )

                # Update statistics
                self.request_count += 1
                self.total_tokens += response.usage.total_tokens
                self.total_response_time += time.time() - start_time

                # Store new embeddings
                for i, embedding_data in enumerate(response.data):
                    embedding = embedding_data.embedding
                    original_index = uncached_indices[i]
                    new_embeddings[original_index] = embedding

                    # Cache the result
                    await self._save_to_cache(self._get_text_hash(uncached_texts[i]), uncached_texts[i], embedding)

            # Combine cached and new embeddings
            result_embeddings = []
            for i in range(len(texts)):
                if i in cached_embeddings:
                    result_embeddings.append(cached_embeddings[i])
                elif i in new_embeddings:
                    result_embeddings.append(new_embeddings[i])
                else:
                    result_embeddings.append(None)

            return result_embeddings

        except Exception as e:
            logger.error(f"Failed to process embedding batch: {str(e)}")
            return [None] * len(texts)

    async def embed_document_chunks(
        self, 
        chunks: List[Dict[str, Any]],
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Create embeddings for document chunks with insurance-specific optimization

        Args:
            chunks: List of chunk dictionaries with 'text' and metadata fields
            include_metadata: Whether to include chunk metadata in result (default: True)

        Returns:
            List of vectors ready for Pinecone upsert
        """
        if not chunks:
            return []

        logger.info(f"Creating embeddings for {len(chunks)} document chunks")

        # Optimize texts based on chunk type
        texts = []
        for chunk in chunks:
            chunk_type = chunk.get('chunk_type', 'unknown').lower()
            text = chunk.get('text', '')
            if chunk_type == 'policy_text':
                text = EmbeddingOptimizer.prepare_policy_text(text)
            elif chunk_type == 'legal_text':
                text = EmbeddingOptimizer.prepare_legal_text(text)
            texts.append(text)

        # Create embeddings
        embeddings = await self.create_embeddings_batch(texts)

        # Prepare vectors for Pinecone
        vectors = []
        successful_count = 0

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if embedding is None:
                logger.warning(f"Failed to create embedding for chunk {i}")
                continue

            # Generate unique ID for vector
            vector_id = self._generate_vector_id(chunk, i)

            # Prepare metadata
            metadata = {
                'chunk_id': i,
                'text': chunk.get('text', '')[:1000],  # Truncate for metadata
                'char_count': chunk.get('char_count', 0),
                'word_count': chunk.get('word_count', 0),
                'chunk_type': chunk.get('chunk_type', 'unknown'),
                'policy_uin': chunk.get('policy_uin', ''),
                'company_name': chunk.get('company_name', ''),
                'section_header': chunk.get('section_header', ''),
                'section_type': chunk.get('section_type', '')
            }

            # Add additional metadata if requested
            if include_metadata:
                for key, value in chunk.items():
                    if key not in ['text', 'char_count', 'word_count', 'chunk_type', 'policy_uin', 'company_name', 'section_header', 'section_type'] and not key.startswith('_'):
                        metadata[f"chunk_{key}"] = value

            # Add processing metadata
            metadata.update({
                'created_at': datetime.utcnow().isoformat(),
                'embedding_model': self.model,
                'processing_version': '2.0'  # Updated for new features
            })

            vector = {
                'id': vector_id,
                'values': embedding,
                'metadata': metadata
            }

            vectors.append(vector)
            successful_count += 1

        logger.info(f"Successfully created {successful_count}/{len(chunks)} chunk embeddings")
        return vectors

    async def embed_query(self, query: str, domain: str = 'insurance') -> Optional[List[float]]:
        """
        Create embedding for search query with domain optimization

        Args:
            query: The query string
            domain: The domain for optimization (default: 'insurance')

        Returns:
            Embedding vector or None if failed
        """
        if not query or not query.strip():
            logger.warning("Empty query provided for embedding")
            return None

        # Optimize query for domain
        optimized_query = EmbeddingOptimizer.prepare_query_for_domain(query, domain)
        embedding = await self.create_embedding(optimized_query)

        if embedding:
            logger.debug(f"Created query embedding for: '{query[:50]}...' (domain: {domain})")

        return embedding

    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text caching"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _generate_vector_id(self, chunk: Dict[str, Any], index: int) -> str:
        """Generate unique vector ID"""
        text = chunk.get('text', '')
        text_hash = self._get_text_hash(text)
        source = chunk.get('source', 'unknown')
        policy_uin = chunk.get('policy_uin', 'unknown')
        source_hash = hashlib.md5((source + policy_uin).encode('utf-8')).hexdigest()[:8]
        return f"{source_hash}_{index}_{text_hash[:8]}"

    def _optimize_query_text(self, query: str) -> str:
        """Optimize query text for better embedding"""
        import re
        cleaned = re.sub(r'[^\w\s\-\.]', ' ', query)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        if len(cleaned) > 500:
            cleaned = cleaned[:500].rsplit(' ', 1)[0]  # Cut at word boundary
        return cleaned.strip()

    def get_statistics(self) -> Dict[str, Any]:
        """Get embedding statistics"""
        cache_hit_rate = (self.cache_hits / self.request_count) * 100 if self.request_count > 0 else 0
        avg_response_time = (self.total_response_time / self.request_count) * 1000 if self.request_count > 0 else 0
        return {
            'total_requests': self.request_count,
            'total_tokens': self.total_tokens,
            'cached_embeddings': len(self.cache),
            'cache_hit_rate_percent': round(cache_hit_rate, 2),
            'avg_response_time_ms': round(avg_response_time, 2),
            'model': self.model,
            'estimated_cost_usd': self.total_tokens * 0.00013 / 1000,  # Cost for text-embedding-3-large
            'timestamp': datetime.utcnow().isoformat()
        }

    def clear_cache(self) -> bool:
        """Clear embedding cache (both in-memory and persistent)"""
        try:
            cache_size = len(self.cache)
            self.cache.clear()
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            logger.info(f"Cleared {cache_size} cached embeddings and persistent cache")
            return True
        except Exception as e:
            logger.error(f"Failed to clear cache: {str(e)}")
            return False

    async def warm_up_cache(self, common_queries: List[str]):
        """Pre-populate cache with common queries"""
        logger.info(f"Warming up cache with {len(common_queries)} common queries")
        await self.create_embeddings_batch(common_queries)

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on embedding service"""
        try:
            test_text = "This is a test embedding for insurance policy analysis."
            start_time = time.time()
            embedding = await self.create_embedding(test_text)
            response_time = time.time() - start_time

            if embedding and len(embedding) == settings.EMBEDDING_DIMENSION:
                return {
                    'status': 'healthy',
                    'model': self.model,
                    'dimension': len(embedding),
                    'response_time_ms': round(response_time * 1000, 2),
                    'statistics': self.get_statistics(),
                    'timestamp': datetime.utcnow().isoformat()
                }
            else:
                return {
                    'status': 'unhealthy',
                    'error': 'Invalid embedding response',
                    'timestamp': datetime.utcnow().isoformat()
                }

        except Exception as e:
            logger.error(f"Embedding health check failed: {str(e)}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

class EmbeddingOptimizer:
    """Optimize embeddings for specific use cases"""

    @staticmethod
    def prepare_policy_text(text: str) -> str:
        """Optimize text for insurance policy embedding"""
        policy_terms = [
            'coverage', 'premium', 'deductible', 'claim', 'benefit',
            'exclusion', 'waiting period', 'pre-existing', 'maternity',
            'hospitalization', 'surgery', 'treatment', 'policy uin',
            'insured', 'nominee', 'sum assured'
        ]
        enhanced_text = text
        for term in policy_terms:
            import re
            pattern = rf'\b{re.escape(term)}\b'
            enhanced_text = re.sub(pattern, f"{term} {term}", enhanced_text, flags=re.IGNORECASE)
        return enhanced_text

    @staticmethod
    def prepare_legal_text(text: str) -> str:
        """Optimize text for legal document embedding"""
        legal_terms = [
            'shall', 'agreement', 'party', 'obligation', 'liability',
            'indemnity', 'breach', 'termination', 'governing law',
            'contract', 'clause', 'amendment'
        ]
        enhanced_text = text
        for term in legal_terms:
            import re
            pattern = rf'\b{re.escape(term)}\b'
            enhanced_text = re.sub(pattern, f"{term} {term}", enhanced_text, flags=re.IGNORECASE)
        return enhanced_text

    @staticmethod
    def prepare_query_for_domain(query: str, domain: str) -> str:
        """Prepare query based on domain"""
        domain_contexts = {
            'insurance': 'insurance policy coverage details',
            'legal': 'legal contract agreement terms',
            'hr': 'human resources policy guidelines',
            'compliance': 'regulatory compliance requirements',
            'health insurance': 'health insurance policy benefits',
            'motor insurance': 'motor insurance policy coverage'
        }
        context = domain_contexts.get(domain.lower(), '')
        if context:
            return f"{context}: {query}"
        return query