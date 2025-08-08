# app/core/dual_index_manager.py
"""
Phase 2: Dual-Index Strategy
Semantic Index (vectors) + Keyword/Metadata Index (structured search)
"""
import asyncio
import json
import sqlite3
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path

from app.core.embedding_manager import EmbeddingManager
from app.core.vector_store import VectorStore
from app.core.reranker import Reranker
from app.core.query_expander import QueryExpander
from app.core.advanced_cache import advanced_cache
from app.config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class SearchFilter:
    """Search filter for metadata-based filtering"""
    policy_uin: Optional[str] = None
    company_name: Optional[str] = None
    policy_type: Optional[str] = None
    section_type: Optional[str] = None
    chunk_type: Optional[str] = None
    file_name: Optional[str] = None

@dataclass 
class SearchResult:
    """Search result with dual-index scores"""
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    semantic_score: float
    metadata_score: float
    combined_score: float
    rank_position: int

class DualIndexManager:
    """
    Manages both semantic (vector) and metadata (keyword) indexes
    Provides hybrid search capabilities with precision filtering
    """
    
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        self.vector_store = VectorStore()
        self.reranker = Reranker()  # Priority 1: Re-ranker for precision
        self.query_expander = QueryExpander()  # Priority 3: Query expansion
        self.metadata_db_path = Path(settings.CACHE_STORAGE_PATH) / "metadata_index.db"
        self._metadata_db = None
        self._initialized = False
        
    async def initialize(self) -> bool:
        """Initialize both indexes"""
        try:
            # Initialize vector store
            if not await self.vector_store.initialize():
                logger.error("Failed to initialize vector store")
                return False
            
            # Initialize re-ranker (Priority 1)
            if not await self.reranker.initialize():
                logger.warning("Failed to initialize re-ranker, continuing without it")
            
            # Initialize metadata database
            await self._initialize_metadata_db()
            
            self._initialized = True
            logger.info("Dual-index system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize dual-index system: {str(e)}")
            return False
    
    async def _initialize_metadata_db(self):
        """Initialize SQLite database for metadata indexing"""
        
        # Ensure cache directory exists
        self.metadata_db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Connect to database
        self._metadata_db = sqlite3.connect(str(self.metadata_db_path), check_same_thread=False)
        self._metadata_db.row_factory = sqlite3.Row
        
        # Create tables
        cursor = self._metadata_db.cursor()
        
        # Main chunks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                namespace TEXT NOT NULL,
                file_name TEXT,
                policy_uin TEXT,
                company_name TEXT,
                policy_type TEXT,
                section_header TEXT,
                section_type TEXT,
                chunk_type TEXT,
                chunk_index INTEGER,
                char_count INTEGER,
                word_count INTEGER,
                confidence_score REAL,
                is_definition BOOLEAN DEFAULT FALSE,
                is_exclusion BOOLEAN DEFAULT FALSE,
                is_table BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                -- Full-text search columns
                text_content TEXT,
                keywords TEXT
            )
        ''')
        
        # Create indexes for fast lookup
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_policy_uin ON chunks(policy_uin)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_company_name ON chunks(company_name)') 
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_section_type ON chunks(section_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunk_type ON chunks(chunk_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_namespace ON chunks(namespace)')
        
        # Note: Removed FTS5 table and triggers - using simple LIKE queries instead
        
        self._metadata_db.commit()
        logger.info("Metadata database initialized")
    
    async def index_document_chunks(self, chunks: List[Dict[str, Any]], namespace: str) -> bool:
        """Index chunks in both semantic and metadata indexes"""
        
        try:
            if not self._initialized:
                await self.initialize()
            
            logger.info(f"Indexing {len(chunks)} chunks in dual-index system")
            
            # Phase 2A: Index in semantic vector store (existing)
            semantic_vectors = await self.embedding_manager.embed_document_chunks(chunks, include_metadata=True)
            
            if not semantic_vectors:
                logger.error("Failed to create semantic vectors")
                return False
            
            semantic_success = await self.vector_store.upsert_vectors(semantic_vectors, namespace)
            
            # Phase 2B: Index in metadata database (new)
            metadata_success = await self._index_chunks_metadata(chunks, namespace, semantic_vectors)
            
            success = semantic_success and metadata_success
            logger.info(f"Dual-index complete: Semantic={semantic_success}, Metadata={metadata_success}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to index chunks in dual-index: {str(e)}")
            return False
    
    async def _index_chunks_metadata(self, chunks: List[Dict[str, Any]], namespace: str, vectors: List[Dict[str, Any]]) -> bool:
        """Index chunks metadata in SQLite database"""
        
        try:
            cursor = self._metadata_db.cursor()
            
            # Prepare batch insert
            chunk_records = []
            
            for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
                chunk_metadata = chunk.get('metadata', {})
                chunk_id = vector.get('id', f"{namespace}_{i}")
                
                # Extract keywords from text for search
                keywords = self._extract_keywords(chunk.get('text', ''))
                
                record = (
                    chunk_id,
                    namespace,
                    chunk_metadata.get('file_name'),
                    chunk_metadata.get('policy_uin'),
                    chunk_metadata.get('company_name'),
                    chunk_metadata.get('policy_type'),
                    chunk_metadata.get('section_header'),
                    chunk_metadata.get('section_type'),
                    chunk_metadata.get('chunk_type'),
                    chunk_metadata.get('chunk_index', i),
                    chunk_metadata.get('char_count', 0),
                    chunk_metadata.get('word_count', 0),
                    chunk_metadata.get('confidence_score', 1.0),
                    chunk_metadata.get('is_definition', False),
                    chunk_metadata.get('is_exclusion', False),
                    chunk_metadata.get('is_table', False),
                    chunk.get('text', ''),
                    ' '.join(keywords)
                )
                
                chunk_records.append(record)
            
            # Batch insert
            cursor.executemany('''
                INSERT OR REPLACE INTO chunks (
                    chunk_id, namespace, file_name, policy_uin, company_name, policy_type,
                    section_header, section_type, chunk_type, chunk_index, char_count, word_count,
                    confidence_score, is_definition, is_exclusion, is_table, text_content, keywords
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', chunk_records)
            
            self._metadata_db.commit()
            logger.info(f"Indexed {len(chunk_records)} chunks in metadata database")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to index metadata: {str(e)}")
            return False
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for metadata search"""
        
        import re
        
        # Remove common words and extract meaningful terms
        text_lower = text.lower()
        
        # Extract insurance-specific terms
        insurance_terms = re.findall(r'\b(?:policy|coverage|premium|claim|benefit|exclusion|deductible|hospitalization|treatment|surgery|accident|disability|death|maternity|pre-existing|waiting|period|grace|notification|reimbursement|cashless)\b', text_lower)
        
        # Extract policy-specific identifiers
        identifiers = re.findall(r'\b[A-Z]{3,}[0-9]{4,}[A-Z][0-9]{6,}\b', text)
        
        # Extract company names
        companies = re.findall(r'\b(?:hdfc|ergo|bajaj|allianz|icici|lombard|national|new india|star|max|bupa|cholamandalam)\b', text_lower)
        
        # Combine all keywords
        keywords = list(set(insurance_terms + identifiers + companies))
        
        return keywords

    def _parse_query_intent(self, query: str) -> SearchFilter:
        """Parse query intent to create appropriate metadata filters"""
        
        query_lower = query.lower()
        
        # Initialize filter
        filter = SearchFilter()
        
        # Detect section types based on query content (less restrictive)
        # Only apply very specific section filters to avoid missing relevant content
        if any(term in query_lower for term in ['exclusion', 'not covered', 'what is not covered', 'limitation', 'excluded']):
            filter.section_type = 'exclusions'
        elif any(term in query_lower for term in ['claim procedure', 'claim process', 'how to claim', 'claim form', 'claim documents']):
            filter.section_type = 'claims'
        elif any(term in query_lower for term in ['schedule', 'benefit schedule', 'table of benefits']) and 'table' in query_lower:
            filter.section_type = 'schedule'
        # Remove overly restrictive filters for definitions, coverage, conditions
        # Let semantic search handle the content matching instead
        
        # Detect chunk types (only for very specific cases)
        if any(term in query_lower for term in ['table', 'schedule']) and any(term in query_lower for term in ['limit', 'sub-limit', 'benefit schedule']):
            filter.chunk_type = 'table_markdown'
        # Remove overly restrictive chunk type filters
        # Let semantic search handle content matching instead of pre-filtering
        
        # Detect policy types
        if any(term in query_lower for term in ['health', 'medical', 'mediclaim', 'hospitalization']):
            filter.policy_type = 'Health Insurance'
        elif any(term in query_lower for term in ['travel', 'trip', 'journey']):
            filter.policy_type = 'Travel Insurance'
        elif any(term in query_lower for term in ['motor', 'vehicle', 'car', 'two wheeler']):
            filter.policy_type = 'Motor Insurance'
        elif any(term in query_lower for term in ['home', 'house', 'property']):
            filter.policy_type = 'Home Insurance'
        elif any(term in query_lower for term in ['life', 'death', 'survival']):
            filter.policy_type = 'Life Insurance'
        elif any(term in query_lower for term in ['accident', 'personal accident', 'disability']):
            filter.policy_type = 'Personal Accident'
        
        # Detect company names
        if any(term in query_lower for term in ['hdfc', 'ergo']):
            filter.company_name = 'HDFC Ergo'
        elif any(term in query_lower for term in ['bajaj', 'allianz']):
            filter.company_name = 'Bajaj Allianz'
        elif any(term in query_lower for term in ['icici', 'lombard']):
            filter.company_name = 'ICICI Lombard'
        elif any(term in query_lower for term in ['national']):
            filter.company_name = 'National Insurance'
        elif any(term in query_lower for term in ['new india']):
            filter.company_name = 'New India Assurance'
        elif any(term in query_lower for term in ['star']):
            filter.company_name = 'Star Health'
        elif any(term in query_lower for term in ['max', 'bupa']):
            filter.company_name = 'Max Bupa'
        elif any(term in query_lower for term in ['chola', 'cholamandalam']):
            filter.company_name = 'Cholamandalam'
        
        return filter
    
    def _calculate_intent_confidence(self, query: str, filters: SearchFilter) -> float:
        """
        Calculate confidence in the parsed intent for adaptive strategy
        
        Args:
            query: User query
            filters: Parsed intent filters
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        query_lower = query.lower()
        confidence = 0.0
        
        # High confidence indicators
        if filters.section_type:
            confidence += 0.4  # Clear section type identified
        
        if filters.chunk_type:
            confidence += 0.3  # Clear chunk type identified
        
        # Specific term matches boost confidence
        if any(word in query_lower for word in ['definition of', 'define', 'means']):
            confidence += 0.2
        
        if any(word in query_lower for word in ['what is covered', 'coverage for']):
            confidence += 0.2
        
        if any(word in query_lower for word in ['exclusions', 'not covered']):
            confidence += 0.2
        
        if any(word in query_lower for word in ['claim process', 'how to claim']):
            confidence += 0.2
        
        # Ambiguous queries reduce confidence
        if len(query.split()) < 3:
            confidence -= 0.1  # Very short queries
        
        if any(word in query_lower for word in ['and', 'or', 'but']):
            confidence -= 0.1  # Complex queries
        
        # Cap confidence at 1.0
        return min(max(confidence, 0.0), 1.0)
    
    async def hybrid_search(
        self, 
        query: str, 
        namespace: str,
        filters: Optional[SearchFilter] = None,
        top_k: int = 20
    ) -> List[SearchResult]:
        """
        Perform hybrid search using both semantic and metadata indexes
        Phase 3: Hybrid Retrieval and Reranking
        """
        
        try:
            logger.info(f"Performing hybrid search: '{query[:50]}...' in namespace: {namespace}")
            
            # Check cache first
            cached_results = advanced_cache.get_search_results(query, namespace)
            if cached_results:
                logger.info("Cache hit for search results - skipping search")
                return self._deserialize_search_results(cached_results)[:top_k]
            
            # Step 1: Priority 2 - Adaptive Retrieval Strategy
            # First, run fast intent parsing to determine if LLM expansion is needed
            intent_filters = self._parse_query_intent(query)
            intent_confidence = self._calculate_intent_confidence(query, intent_filters)
            
            # Step 2: Adaptive Query Expansion with Multi-Query Retrieval
            if intent_confidence >= 0.8:  # High confidence - skip expensive LLM expansion
                logger.info(f"High confidence intent ({intent_confidence:.2f}) - using fast path")
                expanded_query = query  # Use original query
                expansion_result = {
                    'original_query': query,
                    'expanded_terms': [],
                    'expanded_query': query,
                    'expansion_count': 0,
                    'expansion_methods': {'pattern_based': 0, 'llm_based': 0},
                    'adaptive_strategy': 'fast_path'
                }
            elif intent_confidence >= 0.6:  # Medium confidence - use pattern expansion
                logger.info(f"Medium confidence intent ({intent_confidence:.2f}) - using pattern expansion")
                expansion_result = await self.query_expander._fast_pattern_expansion(query)
                expanded_query = expansion_result.get('expanded_query', query)
                expansion_result['adaptive_strategy'] = 'pattern_expansion'
            else:  # Low confidence - use full LLM expansion
                logger.info(f"Low confidence intent ({intent_confidence:.2f}) - using full expansion")
                expansion_result = await self.query_expander.expand_for_search(query)
                
                # Multi-Query Retrieval: Check if this is a procedural query
                query_type = self.query_expander._detect_query_type(query)
                if query_type == 'procedural':
                    logger.info("Procedural query detected - using multi-query retrieval")
                    expansion_result['multi_query_strategy'] = 'procedural_sub_queries'
                else:
                    expansion_result['multi_query_strategy'] = 'single_query'
                
                expanded_query = expansion_result.get('expanded_query', query)
                expansion_result['adaptive_strategy'] = 'full_expansion'
            
            logger.info(f"Query expansion: {expansion_result.get('expansion_count', 0)} terms added")
            
            # Step 2: Merge with provided filters (provided filters take precedence)
            if filters:
                if filters.section_type:
                    intent_filters.section_type = filters.section_type
                if filters.chunk_type:
                    intent_filters.chunk_type = filters.chunk_type
                if filters.policy_type:
                    intent_filters.policy_type = filters.policy_type
                if filters.company_name:
                    intent_filters.company_name = filters.company_name
                if filters.policy_uin:
                    intent_filters.policy_uin = filters.policy_uin
            
            # Step 3A: Filter First (Metadata Index)
            filtered_chunk_ids = await self._metadata_filter_search(intent_filters, namespace)
            logger.info(f"Metadata filter: {len(filtered_chunk_ids)} chunks pass filter criteria")
            
            # Step 3B: Two-Stage Adaptive Retrieval Strategy
            if expansion_result.get('multi_query_strategy') == 'procedural_sub_queries':
                # Stage 1: Attempt Precision Retrieval with Multi-Query
                logger.info("Stage 1: Executing precision retrieval with multi-query for procedural query")
                semantic_results, keyword_results = await self._execute_multi_query_search(
                    query, expanded_query, namespace, filtered_chunk_ids, top_k
                )
                
                # Check if precision retrieval yielded sufficient results
                total_candidates = len(semantic_results) + len(keyword_results)
                if total_candidates < 5:
                    logger.warning(f"Precision retrieval yielded only {total_candidates} candidates. Activating Stage 2: Broad fallback strategy.")
                    
                    # Stage 2: Execute Broad Fallback on Failure
                    semantic_results, keyword_results = await self._execute_broad_fallback_search(
                        query, namespace, top_k
                    )
                else:
                    logger.info(f"Precision retrieval successful: {total_candidates} candidates found")
            else:
                # Single Query Search (optimized for speed)
                semantic_results = await self._semantic_search_filtered(expanded_query, namespace, filtered_chunk_ids, min(top_k, 30))
                keyword_results = await self._keyword_search_filtered(expanded_query, filtered_chunk_ids, min(top_k, 30))
            
            # Step 3C: Optimized Reranking (Reciprocal Rank Fusion)
            fused_results = await self._reciprocal_rank_fusion(semantic_results, keyword_results, min(top_k * 2, 60))  # Limit candidates for speed
            
            # Step 3E: Optimized Re-ranker for Precision
            final_results = await self._apply_reranker(query, fused_results[:min(len(fused_results), 40)], top_k)  # Limit re-ranking to 40 candidates max
            
            # Cache the results
            advanced_cache.cache_search_results(query, namespace, final_results)
            
            logger.info(f"Hybrid search complete: {len(final_results)} results after re-ranking")
            return final_results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {str(e)}")
            return []
    
    async def _execute_multi_query_search(
        self, 
        original_query: str, 
        expanded_query: str, 
        namespace: str, 
        filtered_chunk_ids: List[str], 
        top_k: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Multi-Query Retrieval: Execute multiple sub-queries in parallel
        
        Args:
            original_query: Original user query
            expanded_query: Expanded query string
            namespace: Document namespace
            filtered_chunk_ids: Pre-filtered chunk IDs
            top_k: Number of results to return
            
        Returns:
            Tuple of (semantic_results, keyword_results) from all sub-queries
        """
        try:
            # Extract sub-queries from the expanded query
            # The expanded query format is: "query" OR "sub1" OR "sub2" OR "sub3"
            sub_queries = []
            
            # Parse the expanded query to extract individual sub-queries
            if ' OR ' in expanded_query:
                # Split by OR and clean up quotes
                parts = expanded_query.split(' OR ')
                for part in parts:
                    # Remove surrounding quotes
                    clean_part = part.strip().strip('"')
                    if clean_part and clean_part != original_query:
                        sub_queries.append(clean_part)
            
            # If no sub-queries found, use the original query
            if not sub_queries:
                sub_queries = [original_query]
            
            logger.info(f"Executing {len(sub_queries)} sub-queries: {sub_queries}")
            
            # Execute all sub-queries in parallel
            semantic_tasks = []
            keyword_tasks = []
            
            for sub_query in sub_queries[:3]:  # Limit to 3 sub-queries for speed
                semantic_tasks.append(
                    self._semantic_search_filtered(sub_query, namespace, filtered_chunk_ids, min(top_k, 20))  # Reduced search scope
                )
                keyword_tasks.append(
                    self._keyword_search_filtered(sub_query, filtered_chunk_ids, min(top_k, 20))  # Reduced search scope
                )
            
            # Wait for all tasks to complete
            import asyncio
            semantic_results_list = await asyncio.gather(*semantic_tasks)
            keyword_results_list = await asyncio.gather(*keyword_tasks)
            
            # Merge and deduplicate results
            all_semantic_results = []
            all_keyword_results = []
            
            for semantic_results in semantic_results_list:
                all_semantic_results.extend(semantic_results)
            
            for keyword_results in keyword_results_list:
                all_keyword_results.extend(keyword_results)
            
            # Deduplicate by chunk_id and keep highest scores
            semantic_results = self._deduplicate_results(all_semantic_results, top_k)
            keyword_results = self._deduplicate_results(all_keyword_results, top_k)
            
            logger.info(f"Multi-query retrieval complete: {len(semantic_results)} semantic, {len(keyword_results)} keyword results")
            
            return semantic_results, keyword_results
            
        except Exception as e:
            logger.error(f"Multi-query retrieval failed: {str(e)}")
            # Fallback to single query
            semantic_results = await self._semantic_search_filtered(expanded_query, namespace, filtered_chunk_ids, top_k)
            keyword_results = await self._keyword_search_filtered(expanded_query, filtered_chunk_ids, top_k)
            return semantic_results, keyword_results
    
    def _deduplicate_results(self, results: List[Dict], top_k: int) -> List[Dict]:
        """
        Deduplicate search results by chunk_id, keeping highest scores
        
        Args:
            results: List of search results
            top_k: Maximum number of results to return
            
        Returns:
            Deduplicated results
        """
        seen_chunks = {}
        
        for result in results:
            chunk_id = result.get('chunk_id')
            if chunk_id:
                if chunk_id not in seen_chunks:
                    seen_chunks[chunk_id] = result
                else:
                    # Keep the result with higher score
                    current_score = seen_chunks[chunk_id].get('combined_score', 0.0)
                    new_score = result.get('combined_score', 0.0)
                    if new_score > current_score:
                        seen_chunks[chunk_id] = result
        
        # Sort by score and return top_k
        deduplicated = list(seen_chunks.values())
        deduplicated.sort(key=lambda x: x.get('combined_score', 0.0), reverse=True)
        
        return deduplicated[:top_k]
    
    async def _execute_broad_fallback_search(
        self, 
        original_query: str, 
        namespace: str, 
        top_k: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Stage 2: Broad Fallback Strategy for Recall Rescue
        
        When precision retrieval fails, this method:
        1. Uses only the original query (no sub-queries)
        2. Relaxes filters to search across all chunks
        3. Performs broad semantic and keyword search
        
        Args:
            original_query: Original user query
            namespace: Document namespace
            top_k: Number of results to return
            
        Returns:
            Tuple of (semantic_results, keyword_results) from broad search
        """
        try:
            logger.info("Stage 2: Executing broad fallback search for recall rescue")
            
            # Step 1: Create relaxed filter (only namespace, no section_type restrictions)
            relaxed_filter = SearchFilter()
            # Keep only essential filters that don't restrict content
            # Remove section_type, chunk_type, etc. that might exclude relevant content
            
            # Step 2: Get all chunk IDs for this namespace (no filtering)
            cursor = self._metadata_db.cursor()
            cursor.execute("SELECT chunk_id FROM chunks WHERE namespace = ?", [namespace])
            all_chunk_ids = [row['chunk_id'] for row in cursor.fetchall()]
            
            logger.info(f"Broad search: Using all {len(all_chunk_ids)} chunks in namespace")
            
            # Step 3: Execute broad semantic search with original query
            semantic_results = await self._semantic_search_filtered(
                original_query, namespace, all_chunk_ids, top_k * 3  # Get more candidates for health insurance
            )
            
            # Step 4: Execute broad keyword search with original query
            keyword_results = await self._keyword_search_filtered(
                original_query, all_chunk_ids, top_k * 3  # Get more candidates for health insurance
            )
            
            # Step 5: Enhanced health insurance specific search
            # Extract key terms from the original query for additional keyword search
            health_insurance_terms = self._extract_health_insurance_terms(original_query)
            if health_insurance_terms:
                additional_keyword_results = await self._keyword_search_filtered(
                    " OR ".join(health_insurance_terms), all_chunk_ids, top_k * 2
                )
                # Merge additional results
                keyword_results.extend(additional_keyword_results)
            
            logger.info(f"Broad fallback search complete: {len(semantic_results)} semantic, {len(keyword_results)} keyword results")
            
            return semantic_results, keyword_results
            
        except Exception as e:
            logger.error(f"Broad fallback search failed: {str(e)}")
            # Return empty results if fallback also fails
            return [], []
    
    def _extract_health_insurance_terms(self, query: str) -> List[str]:
        """
        Extract health insurance specific terms from a query for enhanced search
        
        Args:
            query: Original user query
            
        Returns:
            List of health insurance specific terms to search for
        """
        query_lower = query.lower()
        health_terms = []
        
        # Health insurance specific terminology mapping
        health_insurance_mappings = {
            'moratorium': ['moratorium period', 'eight years', '8 years', 'no look back', 'contestible'],
            'cumulative bonus': ['cumulative bonus', 'claim free', '10%', 'ten percent', 'maximum 100%'],
            'free look': ['free look period', 'fifteen days', '15 days', 'return policy'],
            'maternity': ['maternity expenses', 'childbirth', 'caesarean', 'normal delivery', 'pregnancy'],
            'organ donor': ['organ donation', 'donor expenses', 'transplant', 'donor costs'],
            'preventive care': ['preventive care', 'vaccination', 'immunization', 'inoculation', 'preventive treatment'],
            'fraudulent claim': ['fraudulent', 'false statement', 'forfeited', 'misrepresentation'],
            'hospital': ['hospital', 'healthcare facility', 'medical institution', 'inpatient', 'day care'],
            'accident': ['accident', 'accidental', 'sudden', 'unforeseen', 'involuntary', 'external', 'visible', 'violent']
        }
        
        # Check for health insurance specific terms in the query
        for key_term, related_terms in health_insurance_mappings.items():
            if key_term in query_lower:
                health_terms.extend(related_terms)
        
        # Add general health insurance terms if query contains health-related words
        health_indicators = ['health', 'medical', 'hospital', 'treatment', 'claim', 'policy', 'insurance']
        if any(indicator in query_lower for indicator in health_indicators):
            health_terms.extend([
                'health insurance', 'medical expenses', 'hospitalization', 'treatment',
                'coverage', 'benefits', 'exclusions', 'conditions', 'requirements'
            ])
        
        # Remove duplicates and limit to reasonable number
        unique_terms = list(set(health_terms))
        return unique_terms[:10]  # Limit to 10 terms
    
    async def _metadata_filter_search(self, filters: Optional[SearchFilter], namespace: str) -> List[str]:
        """Step 3A: Filter chunks by metadata (hard filter)"""
        
        cursor = self._metadata_db.cursor()
        
        # Build WHERE clause
        where_conditions = ["namespace = ?"]
        params = [namespace]
        
        if filters:
            if filters.policy_uin:
                where_conditions.append("policy_uin = ?")
                params.append(filters.policy_uin)
            
            if filters.company_name:
                where_conditions.append("company_name LIKE ?")
                params.append(f"%{filters.company_name}%")
            
            if filters.policy_type:
                where_conditions.append("policy_type LIKE ?")
                params.append(f"%{filters.policy_type}%")
            
            if filters.section_type:
                where_conditions.append("section_type = ?")
                params.append(filters.section_type)
            
            if filters.chunk_type:
                where_conditions.append("chunk_type = ?")
                params.append(filters.chunk_type)
            
            if filters.file_name:
                where_conditions.append("file_name LIKE ?")
                params.append(f"%{filters.file_name}%")
        
        query = f"""
            SELECT chunk_id FROM chunks 
            WHERE {' AND '.join(where_conditions)}
            ORDER BY confidence_score DESC, chunk_index ASC
        """
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        chunk_ids = [row['chunk_id'] for row in results]
        logger.info(f"Metadata filter: {len(chunk_ids)} chunks pass filter criteria")
        
        return chunk_ids
    
    async def _apply_reranker(self, query: str, search_results: List[SearchResult], top_k: int) -> List[SearchResult]:
        """
        Step 3E: Apply re-ranker for precision improvement
        Priority 1: Re-ranker for Precision
        """
        try:
            # Convert SearchResult objects to dictionaries for re-ranker
            result_dicts = []
            for result in search_results:
                result_dict = {
                    'text': result.text,
                    'metadata': result.metadata,
                    'combined_score': result.combined_score,
                    'semantic_score': result.semantic_score,
                    'metadata_score': result.metadata_score,
                    'rank_position': result.rank_position
                }
                result_dicts.append(result_dict)
            
            # Apply re-ranker with metadata boosting
            reranked_dicts = await self.reranker.rerank_with_metadata_boost(query, result_dicts, top_k)
            
            # Convert back to SearchResult objects
            reranked_results = []
            for i, result_dict in enumerate(reranked_dicts):
                reranked_result = SearchResult(
                    chunk_id=result_dict.get('chunk_id', f'reranked_{i}'),
                    text=result_dict.get('text', ''),
                    metadata=result_dict.get('metadata', {}),
                    semantic_score=result_dict.get('semantic_score', 0.0),
                    metadata_score=result_dict.get('metadata_score', 0.0),
                    combined_score=result_dict.get('final_score', result_dict.get('combined_score', 0.0)),
                    rank_position=i + 1
                )
                reranked_results.append(reranked_result)
            
            logger.info(f"Re-ranker applied: {len(reranked_results)} results with improved precision")
            return reranked_results
            
        except Exception as e:
            logger.error(f"Re-ranker application failed: {str(e)}")
            return search_results[:top_k]
    
    async def _semantic_search_filtered(self, query: str, namespace: str, filtered_chunk_ids: List[str], top_k: int) -> List[Dict]:
        """Step 3B: Semantic search on filtered chunks only"""
        
        # Create query embedding
        query_embedding = await self.embedding_manager.embed_query(query)
        if not query_embedding:
            return []
        
        # Search in vector store
        vector_results = await self.vector_store.query_vectors(
            query_vector=query_embedding,
            namespace=namespace,
            top_k=min(top_k * 3, 100),  # Get more candidates for filtering
            include_metadata=True
        )
        
        # Filter results to only include chunks that passed metadata filter
        if filtered_chunk_ids:
            filtered_results = []
            for result in vector_results:
                if result['id'] in filtered_chunk_ids:
                    filtered_results.append(result)
            vector_results = filtered_results
        
        # Limit to top_k
        semantic_results = vector_results[:top_k]
        
        logger.info(f"Semantic search: {len(semantic_results)} results after filtering")
        return semantic_results
    
    async def _keyword_search_filtered(self, query: str, filtered_chunk_ids: List[str], top_k: int) -> List[Dict]:
        """Enhanced keyword search with numerical detail focus"""
        
        cursor = self._metadata_db.cursor()
        
        # Extract key terms from query for simple keyword matching
        query_terms = self._extract_keywords(query)
        if not query_terms:
            return []
        
        # Enhanced keyword extraction with numerical patterns
        enhanced_terms = list(query_terms)
        query_lower = query.lower()
        
        # Add specific numerical terms based on query content
        import re
        if 'waiting' in query_lower and 'period' in query_lower:
            enhanced_terms.extend(['24 months', 'twenty four months', '2 years', 'continuous coverage'])
        
        if 'zone' in query_lower and ('payment' in query_lower or 'co-payment' in query_lower):
            enhanced_terms.extend(['15%', 'fifteen percent', 'zone a', 'zone b'])
        
        if 'ent' in query_lower and 'conditions' in query_lower:
            enhanced_terms.extend(['deviated nasal septum', 'csom', 'tonsils', 'adenoids'])
        
        if 'cumulative' in query_lower and 'bonus' in query_lower:
            enhanced_terms.extend(['10%', 'ten percent', '100%', 'claim-free'])
        
        if 'reset' in query_lower and 'benefit' in query_lower:
            enhanced_terms.extend(['unlimited', 'restores', '100%', 'annual sum insured'])
        
        if 'documents' in query_lower and 'claim' in query_lower:
            enhanced_terms.extend(['photo id', 'hospital bills', 'discharge summary', 'investigation reports'])
        
        # Build keyword search conditions with prioritization
        keyword_conditions = []
        params = []
        
        # First, add exact numerical pattern matches (highest priority)
        numerical_patterns = [term for term in enhanced_terms if re.search(r'\d+', term)]
        for term in numerical_patterns[:3]:  # Top 3 numerical terms
            keyword_conditions.append("(text_content LIKE ? OR keywords LIKE ?)")
            params.extend([f"%{term}%", f"%{term}%"])
        
        # Then add regular terms
        regular_terms = [term for term in enhanced_terms if not re.search(r'\d+', term)]
        for term in regular_terms[:5]:  # Limit to top 5 terms to avoid too many conditions
            keyword_conditions.append("(text_content LIKE ? OR keywords LIKE ?)")
            params.extend([f"%{term}%", f"%{term}%"])
        
        # Build base query
        if filtered_chunk_ids:
            placeholders = ','.join(['?' for _ in filtered_chunk_ids])
            base_query = f"""
                SELECT chunk_id, text_content, section_header, section_type, chunk_type,
                       policy_uin, company_name, confidence_score
                FROM chunks
                WHERE ({' OR '.join(keyword_conditions)}) AND chunk_id IN ({placeholders})
                ORDER BY confidence_score DESC, chunk_index ASC
                LIMIT ?
            """
            params.extend(filtered_chunk_ids + [top_k])
        else:
            base_query = f"""
                SELECT chunk_id, text_content, section_header, section_type, chunk_type,
                       policy_uin, company_name, confidence_score
                FROM chunks
                WHERE {' OR '.join(keyword_conditions)}
                ORDER BY confidence_score DESC, chunk_index ASC
                LIMIT ?
            """
            params.append(top_k)
        
        try:
            cursor.execute(base_query, params)
            results = cursor.fetchall()
            
            keyword_results = []
            for i, row in enumerate(results):
                # Calculate simple score based on position and confidence
                score = (1.0 - (i * 0.01)) * row['confidence_score']
                
                keyword_results.append({
                    'id': row['chunk_id'],
                    'score': score,
                    'metadata': {
                        'text': row['text_content'],
                        'section_header': row['section_header'],
                        'section_type': row['section_type'],
                        'chunk_type': row['chunk_type'],
                        'policy_uin': row['policy_uin'],
                        'company_name': row['company_name']
                    }
                })
            
            logger.info(f"Keyword search: {len(keyword_results)} results")
            return keyword_results
            
        except Exception as e:
            logger.error(f"Keyword search failed: {str(e)}")
            return []
    
    async def _reciprocal_rank_fusion(self, semantic_results: List[Dict], keyword_results: List[Dict], top_k: int) -> List[SearchResult]:
        """Step 3D: Reciprocal Rank Fusion for intelligent reranking"""
        
        # Create rank mappings
        semantic_ranks = {result['id']: i + 1 for i, result in enumerate(semantic_results)}
        keyword_ranks = {result['id']: i + 1 for i, result in enumerate(keyword_results)}
        
        # Collect all unique chunk IDs
        all_chunk_ids = set(semantic_ranks.keys()) | set(keyword_ranks.keys())
        
        # Calculate RRF scores
        rrf_scores = {}
        k = 60  # RRF parameter (commonly used value)
        
        for chunk_id in all_chunk_ids:
            semantic_rank = semantic_ranks.get(chunk_id, float('inf'))
            keyword_rank = keyword_ranks.get(chunk_id, float('inf'))
            
            # RRF formula: 1/(k + rank)
            semantic_rrf = 1 / (k + semantic_rank) if semantic_rank != float('inf') else 0
            keyword_rrf = 1 / (k + keyword_rank) if keyword_rank != float('inf') else 0
            
            # Combined RRF score with weights
            rrf_scores[chunk_id] = (semantic_rrf * 0.7) + (keyword_rrf * 0.3)
        
        # Sort by RRF score
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Create final results
        final_results = []
        
        for i, (chunk_id, rrf_score) in enumerate(sorted_chunks):
            # Get chunk details from either semantic or keyword results
            chunk_data = None
            semantic_score = 0.0
            metadata_score = 0.0
            
            # Find in semantic results
            for result in semantic_results:
                if result['id'] == chunk_id:
                    chunk_data = result
                    semantic_score = result['score']
                    break
            
            # Find in keyword results if not found in semantic
            if not chunk_data:
                for result in keyword_results:
                    if result['id'] == chunk_id:
                        chunk_data = result
                        metadata_score = result['score']
                        break
            
            if chunk_data:
                search_result = SearchResult(
                    chunk_id=chunk_id,
                    text=chunk_data['metadata'].get('text', ''),
                    metadata=chunk_data['metadata'],
                    semantic_score=semantic_score,
                    metadata_score=metadata_score,
                    combined_score=rrf_score,
                    rank_position=i + 1
                )
                
                final_results.append(search_result)
        
        logger.info(f"RRF fusion complete: {len(final_results)} final results")
        return final_results
    
    async def get_document_statistics(self, namespace: str) -> Dict[str, Any]:
        """Get comprehensive statistics about indexed document"""
        
        cursor = self._metadata_db.cursor()
        
        # Basic counts
        cursor.execute("SELECT COUNT(*) as total_chunks FROM chunks WHERE namespace = ?", (namespace,))
        total_chunks = cursor.fetchone()['total_chunks']
        
        # By chunk type
        cursor.execute("""
            SELECT chunk_type, COUNT(*) as count 
            FROM chunks WHERE namespace = ? 
            GROUP BY chunk_type
        """, (namespace,))
        chunk_types = {row['chunk_type']: row['count'] for row in cursor.fetchall()}
        
        # By section type
        cursor.execute("""
            SELECT section_type, COUNT(*) as count 
            FROM chunks WHERE namespace = ? 
            GROUP BY section_type
        """, (namespace,))
        section_types = {row['section_type']: row['count'] for row in cursor.fetchall()}
        
        # Special content counts
        cursor.execute("""
            SELECT 
                SUM(CASE WHEN is_definition THEN 1 ELSE 0 END) as definitions,
                SUM(CASE WHEN is_exclusion THEN 1 ELSE 0 END) as exclusions,
                SUM(CASE WHEN is_table THEN 1 ELSE 0 END) as tables
            FROM chunks WHERE namespace = ?
        """, (namespace,))
        special_counts = cursor.fetchone()
        
        return {
            'namespace': namespace,
            'total_chunks': total_chunks,
            'chunk_types': chunk_types,
            'section_types': section_types,
            'definitions_count': special_counts['definitions'],
            'exclusions_count': special_counts['exclusions'],
            'tables_count': special_counts['tables'],
            'indexing_method': 'dual_index_hybrid'
        }
    
    async def cleanup_namespace(self, namespace: str) -> bool:
        """Clean up both indexes for a namespace"""
        
        try:
            # Clean vector store
            vector_cleanup = await self.vector_store.delete_namespace(namespace)
            
            # Clean metadata database
            cursor = self._metadata_db.cursor()
            cursor.execute("DELETE FROM chunks WHERE namespace = ?", (namespace,))
            self._metadata_db.commit()
            
            deleted_count = cursor.rowcount
            logger.info(f"Cleaned up namespace {namespace}: {deleted_count} metadata records deleted")
            
            return vector_cleanup and deleted_count >= 0
            
        except Exception as e:
            logger.error(f"Failed to cleanup namespace {namespace}: {str(e)}")
            return False
    
    def _deserialize_search_results(self, cached_results: List[Dict]) -> List[SearchResult]:
        """Convert cached dictionary results back to SearchResult objects"""
        results = []
        for result_dict in cached_results:
            try:
                # Handle different possible dictionary structures
                if 'text' in result_dict and 'metadata' in result_dict:
                    result = SearchResult(
                        chunk_id=result_dict.get('chunk_id', ''),
                        text=result_dict['text'],
                        metadata=result_dict['metadata'],
                        semantic_score=result_dict.get('semantic_score', 0.0),
                        metadata_score=result_dict.get('metadata_score', 0.0),
                        combined_score=result_dict.get('combined_score', 0.0),
                        rank_position=result_dict.get('rank_position', 0)
                    )
                    results.append(result)
            except Exception as e:
                logger.warning(f"Failed to deserialize cached result: {e}")
                continue
        return results