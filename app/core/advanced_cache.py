"""
Advanced caching system for LLM query retrieval system.
Implements multi-level caching: embeddings, search results, and LLM responses.
"""

import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    data: Any
    timestamp: float
    hits: int = 0
    
    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if cache entry is expired"""
        return time.time() - self.timestamp > ttl_seconds

class AdvancedCache:
    """
    Multi-level cache with LRU eviction and TTL support
    """
    
    def __init__(
        self, 
        cache_dir: str = "data/cache/advanced",
        max_memory_entries: int = 1000,
        default_ttl: int = 3600  # 1 hour
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_memory_entries = max_memory_entries
        self.default_ttl = default_ttl
        
        # In-memory cache for hot data
        self.memory_cache: Dict[str, CacheEntry] = {}
        
        # Separate caches for different data types
        self.embedding_cache_dir = self.cache_dir / "embeddings"
        self.search_cache_dir = self.cache_dir / "search_results"
        self.llm_cache_dir = self.cache_dir / "llm_responses"
        
        for cache_dir in [self.embedding_cache_dir, self.search_cache_dir, self.llm_cache_dir]:
            cache_dir.mkdir(exist_ok=True)
            
        logger.info(f"Advanced cache initialized at {self.cache_dir}")
    
    def _generate_key(self, data: Union[str, Dict, List]) -> str:
        """Generate cache key from data"""
        if isinstance(data, str):
            content = data
        else:
            content = json.dumps(data, sort_keys=True)
        
        return hashlib.md5(content.encode()).hexdigest()
    
    def _evict_old_entries(self):
        """Evict old entries from memory cache using LRU"""
        if len(self.memory_cache) <= self.max_memory_entries:
            return
            
        # Sort by hits (ascending) and timestamp (ascending) for LRU
        sorted_entries = sorted(
            self.memory_cache.items(),
            key=lambda x: (x[1].hits, x[1].timestamp)
        )
        
        # Remove 20% of entries
        remove_count = len(self.memory_cache) - self.max_memory_entries + int(self.max_memory_entries * 0.2)
        for key, _ in sorted_entries[:remove_count]:
            del self.memory_cache[key]
            
        logger.info(f"Evicted {remove_count} entries from memory cache")
    
    def get(self, key: str, cache_type: str = "general", ttl: Optional[int] = None) -> Optional[Any]:
        """Get item from cache"""
        ttl = ttl or self.default_ttl
        
        # Check memory cache first
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if not entry.is_expired(ttl):
                entry.hits += 1
                return entry.data
            else:
                del self.memory_cache[key]
        
        # Check disk cache
        cache_dir = self._get_cache_dir(cache_type)
        cache_file = cache_dir / f"{key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    entry = pickle.load(f)
                
                if not entry.is_expired(ttl):
                    # Promote to memory cache
                    entry.hits += 1
                    self.memory_cache[key] = entry
                    self._evict_old_entries()
                    return entry.data
                else:
                    cache_file.unlink()  # Remove expired file
            except Exception as e:
                logger.warning(f"Failed to load cache entry {key}: {e}")
        
        return None
    
    def set(self, key: str, data: Any, cache_type: str = "general", ttl: Optional[int] = None):
        """Set item in cache"""
        ttl = ttl or self.default_ttl
        entry = CacheEntry(data=data, timestamp=time.time())
        
        # Store in memory cache
        self.memory_cache[key] = entry
        self._evict_old_entries()
        
        # Store in disk cache
        cache_dir = self._get_cache_dir(cache_type)
        cache_file = cache_dir / f"{key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(entry, f)
        except Exception as e:
            logger.warning(f"Failed to save cache entry {key}: {e}")
    
    def _get_cache_dir(self, cache_type: str) -> Path:
        """Get cache directory for type"""
        if cache_type == "embedding":
            return self.embedding_cache_dir
        elif cache_type == "search":
            return self.search_cache_dir
        elif cache_type == "llm":
            return self.llm_cache_dir
        else:
            return self.cache_dir
    
    def cache_embedding(self, query: str, embedding: List[float], ttl: int = 7200):
        """Cache query embedding (2 hour TTL)"""
        key = self._generate_key(f"embedding:{query}")
        self.set(key, embedding, "embedding", ttl)
    
    def get_embedding(self, query: str) -> Optional[List[float]]:
        """Get cached embedding"""
        key = self._generate_key(f"embedding:{query}")
        return self.get(key, "embedding")
    
    def cache_search_results(self, query: str, namespace: str, results: List[Any], ttl: int = 1800):
        """Cache search results (30 min TTL)"""
        cache_key = f"search:{namespace}:{query}"
        key = self._generate_key(cache_key)
        
        # Convert results to serializable format
        serializable_results = []
        for result in results:
            if hasattr(result, '__dict__'):
                serializable_results.append(result.__dict__)
            else:
                serializable_results.append(result)
        
        self.set(key, serializable_results, "search", ttl)
    
    def get_search_results(self, query: str, namespace: str) -> Optional[List[Any]]:
        """Get cached search results"""
        cache_key = f"search:{namespace}:{query}"
        key = self._generate_key(cache_key)
        return self.get(key, "search")
    
    def cache_llm_response(self, context_hash: str, query: str, response: Any, ttl: int = 3600):
        """Cache LLM response (1 hour TTL)"""
        cache_key = f"llm:{context_hash}:{query}"
        key = self._generate_key(cache_key)
        self.set(key, response, "llm", ttl)
    
    def get_llm_response(self, context_hash: str, query: str) -> Optional[Any]:
        """Get cached LLM response"""
        cache_key = f"llm:{context_hash}:{query}"
        key = self._generate_key(cache_key)
        return self.get(key, "llm")
    
    def generate_context_hash(self, search_results: List[Any]) -> str:
        """Generate hash for search results context"""
        # Create deterministic hash from search results
        context_items = []
        for result in search_results[:10]:  # Use top 10 results for context
            if hasattr(result, 'text'):
                context_items.append(result.text[:100])  # First 100 chars
            elif hasattr(result, 'content'):
                context_items.append(result.content[:100])
        
        context_str = "|".join(context_items)
        return self._generate_key(context_str)
    
    def clear_expired(self):
        """Clear all expired entries"""
        current_time = time.time()
        
        # Clear memory cache
        expired_keys = [
            key for key, entry in self.memory_cache.items()
            if entry.is_expired(self.default_ttl)
        ]
        for key in expired_keys:
            del self.memory_cache[key]
        
        # Clear disk cache
        for cache_dir in [self.embedding_cache_dir, self.search_cache_dir, self.llm_cache_dir]:
            for cache_file in cache_dir.glob("*.pkl"):
                try:
                    with open(cache_file, 'rb') as f:
                        entry = pickle.load(f)
                    if entry.is_expired(self.default_ttl):
                        cache_file.unlink()
                except Exception:
                    continue
        
        logger.info(f"Cleared {len(expired_keys)} expired entries from memory cache")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        memory_size = len(self.memory_cache)
        
        disk_sizes = {}
        for cache_type, cache_dir in [
            ("embedding", self.embedding_cache_dir),
            ("search", self.search_cache_dir),
            ("llm", self.llm_cache_dir)
        ]:
            disk_sizes[cache_type] = len(list(cache_dir.glob("*.pkl")))
        
        total_hits = sum(entry.hits for entry in self.memory_cache.values())
        
        return {
            "memory_entries": memory_size,
            "disk_entries": disk_sizes,
            "total_hits": total_hits,
            "memory_usage_percent": (memory_size / self.max_memory_entries) * 100
        }

# Global cache instance
advanced_cache = AdvancedCache()

