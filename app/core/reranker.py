# app/core/reranker.py
"""
Priority 1: Re-ranker for Precision
Cross-encoder model for re-scoring search results to improve accuracy
"""
import logging
from typing import List, Dict, Any, Tuple
from sentence_transformers import CrossEncoder
import numpy as np

logger = logging.getLogger(__name__)

class Reranker:
    """
    Re-ranker using cross-encoder model for precision improvement
    Analyzes query-chunk pairs together for more accurate relevance scoring
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize re-ranker with cross-encoder model
        
        Args:
            model_name: Cross-encoder model name (default: lightweight but effective)
        """
        self.model_name = model_name
        self.model = None
        self._initialized = False
        
    async def initialize(self) -> bool:
        """Initialize the cross-encoder model"""
        try:
            logger.info(f"Initializing re-ranker with model: {self.model_name}")
            self.model = CrossEncoder(self.model_name, max_length=512)
            self._initialized = True
            logger.info("Re-ranker initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize re-ranker: {str(e)}")
            return False
    
    async def rerank_results(
        self, 
        query: str, 
        search_results: List[Dict[str, Any]], 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Re-rank search results using cross-encoder for better precision
        
        Args:
            query: User query
            search_results: Initial search results from hybrid search
            top_k: Number of top results to return
            
        Returns:
            Re-ranked results with improved precision
        """
        if not self._initialized or not self.model:
            logger.warning("Re-ranker not initialized, returning original results")
            return search_results[:top_k]
        
        if not search_results:
            return []
        
        try:
            logger.info(f"Re-ranking {len(search_results)} results for query: '{query[:50]}...'")
            
            # Prepare query-chunk pairs for cross-encoder
            query_chunk_pairs = []
            for result in search_results:
                # Get text content from result
                text = result.get('text', '') or result.get('content', '')
                if text:
                    query_chunk_pairs.append([query, text])
            
            if not query_chunk_pairs:
                logger.warning("No valid text content found in search results")
                return search_results[:top_k]
            
            # Get cross-encoder scores
            scores = self.model.predict(query_chunk_pairs)
            
            # Add re-ranker scores to results
            for i, result in enumerate(search_results):
                if i < len(scores):
                    result['reranker_score'] = float(scores[i])
                    # Boost the combined score with re-ranker score
                    original_score = result.get('combined_score', 0.0)
                    result['combined_score'] = (original_score * 0.3) + (float(scores[i]) * 0.7)
                else:
                    result['reranker_score'] = 0.0
            
            # Sort by re-ranker score (descending)
            reranked_results = sorted(search_results, key=lambda x: x.get('reranker_score', 0.0), reverse=True)
            
            logger.info(f"Re-ranking complete: {len(reranked_results)} results, top score: {reranked_results[0].get('reranker_score', 0.0):.3f}")
            
            return reranked_results[:top_k]
            
        except Exception as e:
            logger.error(f"Re-ranking failed: {str(e)}")
            return search_results[:top_k]
    
    async def rerank_with_metadata_boost(
        self, 
        query: str, 
        search_results: List[Dict[str, Any]], 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Advanced re-ranking with metadata boosting for insurance-specific queries
        
        Args:
            query: User query
            search_results: Initial search results
            top_k: Number of top results to return
            
        Returns:
            Re-ranked results with metadata boosting
        """
        if not self._initialized or not self.model:
            return search_results[:top_k]
        
        try:
            # First, apply cross-encoder re-ranking
            reranked_results = await self.rerank_results(query, search_results, len(search_results))
            
            # Apply metadata boosting for insurance-specific queries
            boosted_results = []
            
            for result in reranked_results:
                metadata = result.get('metadata', {})
                reranker_score = result.get('reranker_score', 0.0)
                
                # Calculate metadata boost
                metadata_boost = self._calculate_metadata_boost(query, metadata)
                
                # Apply boost to final score
                final_score = reranker_score + metadata_boost
                result['final_score'] = final_score
                result['metadata_boost'] = metadata_boost
                
                boosted_results.append(result)
            
            # Sort by final score
            final_results = sorted(boosted_results, key=lambda x: x.get('final_score', 0.0), reverse=True)
            
            logger.info(f"Metadata-boosted re-ranking complete: {len(final_results)} results")
            
            return final_results[:top_k]
            
        except Exception as e:
            logger.error(f"Metadata-boosted re-ranking failed: {str(e)}")
            return search_results[:top_k]
    
    def _calculate_metadata_boost(self, query: str, metadata: Dict[str, Any]) -> float:
        """
        Calculate metadata boost for insurance-specific relevance
        
        Args:
            query: User query
            metadata: Chunk metadata
            
        Returns:
            Boost score (0.0 to 0.3)
        """
        boost = 0.0
        
        # Query intent detection
        query_lower = query.lower()
        
        # Definition queries
        if any(word in query_lower for word in ['definition', 'define', 'means', 'what is']):
            if metadata.get('chunk_type') == 'definition':
                boost += 0.2
            if metadata.get('section_type') == 'definitions':
                boost += 0.1
        
        # Coverage queries
        if any(word in query_lower for word in ['cover', 'coverage', 'benefit', 'include']):
            if metadata.get('chunk_type') in ['benefit', 'coverage']:
                boost += 0.2
            if metadata.get('section_type') in ['coverage', 'benefits']:
                boost += 0.1
        
        # Exclusion queries
        if any(word in query_lower for word in ['exclusion', 'exclude', 'not cover', 'not covered']):
            if metadata.get('chunk_type') == 'exclusion':
                boost += 0.2
            if metadata.get('section_type') == 'exclusions':
                boost += 0.1
        
        # Claims queries
        if any(word in query_lower for word in ['claim', 'process', 'procedure', 'notify']):
            if metadata.get('chunk_type') in ['claim', 'procedure']:
                boost += 0.2
            if metadata.get('section_type') == 'claims':
                boost += 0.1
        
        # Procedural queries - Priority 1: Boost procedural section headers
        if any(word in query_lower for word in ['how to', 'process', 'procedure', 'steps', 'what is the process', 'how do i']):
            section_header = metadata.get('section_header', '').lower()
            if any(proc_word in section_header for proc_word in ['procedure', 'process', 'steps', 'how to', 'notification', 'claim']):
                boost += 0.3  # Significant boost for procedural sections
            if metadata.get('chunk_type') in ['procedure', 'claim', 'process']:
                boost += 0.2
            if metadata.get('section_type') in ['claims', 'procedures']:
                boost += 0.15
        
        # Specific term queries (boost chunks containing the exact terms)
        if any(word in query_lower for word in ['accident', 'accidental']):
            if 'accident' in metadata.get('section_header', '').lower():
                boost += 0.15
        
        if any(word in query_lower for word in ['hospital', 'hospitalization']):
            if 'hospital' in metadata.get('section_header', '').lower():
                boost += 0.15
        
        if any(word in query_lower for word in ['opd', 'outpatient']):
            if 'opd' in metadata.get('section_header', '').lower():
                boost += 0.15
        
        # Cap the boost
        return min(boost, 0.3)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get re-ranker statistics"""
        return {
            'model_name': self.model_name,
            'initialized': self._initialized,
            'model_type': 'cross-encoder'
        } 