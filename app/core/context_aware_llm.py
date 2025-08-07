import asyncio
import json
from typing import List, Dict, Any, Optional
import logging
import time

from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from openai import AsyncOpenAI, RateLimitError
from app.config.settings import settings
from app.core.dual_index_manager import SearchResult, SearchFilter
from app.core.advanced_cache import advanced_cache

logger = logging.getLogger(__name__)

class ContextAwareLLMClient:
    """
    Advanced LLM client with context-aware prompting using preserved metadata
    """
    
    def __init__(self):
        """Initialize the LLM client with OpenAI API settings and token tracking."""
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.LLM_MODEL
        self.max_tokens = settings.MAX_TOKENS
        self.temperature = settings.TEMPERATURE
        
        # Token usage tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.request_count = 0
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the LLM service"""
        try:
            # Simple test to verify API connectivity
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
                temperature=0
            )
            return {
                'status': 'healthy',
                'model': self.model,
                'api_accessible': True,
                'response_time': 'fast'
            }
        except Exception as e:
            logger.error(f"LLM health check failed: {str(e)}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'model': self.model,
                'api_accessible': False
            }
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get usage statistics for the LLM client"""
        return {
            'total_prompt_tokens': self.total_prompt_tokens,
            'total_completion_tokens': self.total_completion_tokens,
            'request_count': self.request_count,
            'model': self.model
        }
    
    async def process_query_with_context_awareness(
        self,
        query: str,
        search_results: List[SearchResult],
        document_metadata: Dict[str, Any],
        filters_applied: Optional[SearchFilter] = None
    ) -> Dict[str, Any]:
        """
        Process a query with full context awareness using preserved metadata.
        
        Args:
            query (str): The user's query.
            search_results (List[SearchResult]): List of search results from the dual-index system.
            document_metadata (Dict[str, Any]): Metadata about the document.
            filters_applied (Optional[SearchFilter]): Filters applied during search, if any.
        
        Returns:
            Dict[str, Any]: The processed response with answer, metadata, and statistics.
        """
        try:
            start_time = time.time()
            
            # Check cache first
            context_hash = advanced_cache.generate_context_hash(search_results)
            cached_response = advanced_cache.get_llm_response(context_hash, query)
            if cached_response:
                logger.info("Cache hit for LLM response - skipping OpenAI call")
                return cached_response
            
            # Step 1: Analyze search results and extract context
            context_analysis = await self._analyze_search_context(search_results, document_metadata)
            
            # Step 2: Create context-aware system prompt
            system_prompt = self._create_context_aware_system_prompt(context_analysis, document_metadata)
            
            # Step 3: Structure context with metadata preservation
            structured_context = self._structure_context_with_metadata(search_results, context_analysis)
            
            # Step 4: Create intelligent user prompt
            user_prompt = self._create_intelligent_user_prompt(query, structured_context, context_analysis)
            
            logger.info(f"Context-aware processing: {len(search_results)} results, {len(context_analysis.get('sections_involved', []))} sections")
            
            # Step 5: Generate response with context awareness (with rate limit retry)
            response = await self._call_openai_with_backoff(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            
            # Update token usage
            if response.usage:
                self.total_prompt_tokens += response.usage.prompt_tokens
                self.total_completion_tokens += response.usage.completion_tokens
                self.request_count += 1
            
            # Parse and enrich response
            result = self._parse_context_aware_response(response.choices[0].message.content)
            
            # Add context-aware metadata and statistics
            result.update({
                'query': query,
                'context_intelligence': {
                    'processing_method': 'context_aware_metadata_preserved',
                    'primary_policy': context_analysis.get('primary_policy_uin'),
                    'primary_company': context_analysis.get('primary_company'),
                    'sections_analyzed': context_analysis.get('sections_involved', []),
                    'content_types_used': context_analysis.get('content_types', []),
                    'search_precision': context_analysis.get('search_precision', 0),
                    'metadata_richness': context_analysis.get('metadata_completeness', 0),
                    'filters_applied': filters_applied.__dict__ if filters_applied else None
                },
                'search_stats': {
                    'total_results': len(search_results),
                    'avg_semantic_score': sum(r.semantic_score for r in search_results) / len(search_results) if search_results else 0,
                    'avg_combined_score': sum(r.combined_score for r in search_results) / len(search_results) if search_results else 0,
                    'top_result_score': search_results[0].combined_score if search_results else 0
                },
                'model': self.model,
                'processing_time': time.time() - start_time,  # Duration of processing
                'tokens_used': response.usage.total_tokens if response.usage else 0
            })
            
            confidence = result.get('confidence', 0)
            logger.info(f"Context-aware processing complete: {confidence:.2f} confidence, {len(context_analysis.get('sections_involved', []))} sections analyzed")
            
            # Cache the result
            advanced_cache.cache_llm_response(context_hash, query, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Context-aware processing failed: {str(e)}")
            return {
                'answer': 'I apologize, but I encountered an error while processing your query with context awareness.',
                'confidence': 0.0,
                'reasoning': f'Context-aware processing error: {str(e)}',
                'sources': [],
                'error': True,
                'context_intelligence': {'error': str(e)}
            }
    
    async def _analyze_search_context(self, search_results: List[SearchResult], document_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze search results to understand context and content structure.
        
        Args:
            search_results (List[SearchResult]): List of search results.
            document_metadata (Dict[str, Any]): Document metadata.
        
        Returns:
            Dict[str, Any]: Analyzed context including policies, companies, sections, etc.
        """
        if not search_results:
            return {
                'primary_policy_uin': None,
                'primary_company': None,
                'sections_involved': [],
                'content_types': [],
                'section_types': [],
                'search_precision': 0,
                'metadata_completeness': 0
            }
        
        policy_uins = set()
        companies = set()
        sections = set()
        content_types = set()
        section_types = set()
        
        for result in search_results:
            metadata = result.metadata
            if metadata.get('policy_uin'):
                policy_uins.add(metadata['policy_uin'])
            if metadata.get('company_name'):
                companies.add(metadata['company_name'])
            if metadata.get('section_header'):
                sections.add(metadata['section_header'])
            if metadata.get('chunk_type'):
                content_types.add(metadata['chunk_type'])
            if metadata.get('section_type'):
                section_types.add(metadata['section_type'])
        
        primary_policy_uin = list(policy_uins)[0] if len(policy_uins) == 1 else None
        primary_company = list(companies)[0] if len(companies) == 1 else None
        
        semantic_scores = [r.semantic_score for r in search_results if r.semantic_score > 0]
        search_precision = sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0
        
        metadata_fields = ['policy_uin', 'company_name', 'section_header', 'section_type', 'chunk_type']
        total_metadata_count = 0
        complete_metadata_count = 0
        
        for result in search_results:
            for field in metadata_fields:
                total_metadata_count += 1
                if result.metadata.get(field):
                    complete_metadata_count += 1
        
        metadata_completeness = complete_metadata_count / total_metadata_count if total_metadata_count > 0 else 0
        
        return {
            'primary_policy_uin': primary_policy_uin,
            'primary_company': primary_company,
            'sections_involved': list(sections),
            'content_types': list(content_types),
            'section_types': list(section_types),
            'search_precision': search_precision,
            'metadata_completeness': metadata_completeness
        }
    
    def _create_context_aware_system_prompt(self, context_analysis: Dict[str, Any], document_metadata: Dict[str, Any]) -> str:
        """
        Create a system prompt that sets the context for the LLM.
        Priority 2: Fine-Tune the LLM Prompt with Explicit Instructions
        
        Args:
            context_analysis (Dict[str, Any]): Analyzed context from search results.
            document_metadata (Dict[str, Any]): Document metadata.
        
        Returns:
            str: The system prompt string.
        """
        prompt = "You are an expert insurance policy analyst with access to detailed policy documents.\n"
        prompt += "Your answer must be based SOLELY on the provided context.\n\n"
        
        # Priority 2: Explicit Instructions for Better Accuracy
        prompt += "CRITICAL INSTRUCTIONS:\n"
        prompt += "- When asked for a definition, you MUST include ALL criteria mentioned (e.g., for a 'Hospital', list the required bed count, staff, operation theatre, etc.).\n"
        prompt += "- When asked about coverage or benefits, you MUST list the specific items or conditions included.\n"
        prompt += "- When asked about exclusions, you MUST list ALL specific exclusions mentioned.\n"
        prompt += "- If you cannot find a specific detail in the context, explicitly state that 'the provided context does not contain this specific detail' instead of giving a general answer.\n"
        prompt += "- Use EXACT terminology from the policy document (e.g., 'accidental bodily injury' not just 'injury').\n"
        prompt += "- Include specific numbers, percentages, and conditions when mentioned in the context.\n"
        prompt += "- If the context mentions multiple conditions or requirements, list ALL of them.\n\n"
        
        prompt += "IMPORTANT: If you cannot find explicit mention of a topic in the provided context, state that the information is not present in the retrieved sections and do not make a definitive conclusion about its absence from the overall policy.\n"
        prompt += "Be cautious and precise in your answers. Only provide information that is explicitly stated in the provided context.\n"
        
        if context_analysis.get('primary_policy_uin'):
            prompt += f"The query is specifically about policy {context_analysis['primary_policy_uin']}.\n"
        elif context_analysis.get('primary_company'):
            prompt += f"The query is related to policies from {context_analysis['primary_company']}.\n"
        else:
            prompt += "The query is about insurance policies in general.\n"
        
        if context_analysis.get('sections_involved'):
            sections = ", ".join(context_analysis['sections_involved'])
            prompt += f"Relevant sections include: {sections}.\n"
        
        if context_analysis.get('content_types'):
            types = ", ".join(context_analysis['content_types'])
            prompt += f"Available content types: {types}.\n"
        
        prompt += "Use the provided context and metadata to give precise, policy-specific answers.\n"
        prompt += "If the query asks about specific policy details, refer to the relevant sections or content types.\n"
        prompt += "Always cite the source of your information, including policy UIN and section if applicable.\n"
        prompt += "If the information is not explicitly mentioned in the provided context, clearly state this and suggest checking other sections of the policy document.\n"
        prompt += "Format your response as a JSON object with the following keys: 'answer', 'confidence' (a float between 0 and 1), 'reasoning', 'sources' (a list of policy UIN and section), and 'additional_info' if needed.\n"
        prompt += "Assess your confidence in the answer based on the relevance and completeness of the provided context.\n"
        prompt += "A confidence score of 0.3 or below indicates that the information is not clearly present in the provided context.\n"
        
        return prompt
    
    def _structure_context_with_metadata(self, search_results: List[SearchResult], context_analysis: Dict[str, Any]) -> str:
        """
        Structure the context from search results, preserving metadata.
        Enhanced with context length management to prevent token limit issues.
        
        Args:
            search_results (List[SearchResult]): List of search results.
            context_analysis (Dict[str, Any]): Analyzed context.
        
        Returns:
            str: Formatted context string with metadata.
        """
        context_parts = []
        total_length = 0
        max_context_length = 60000  # Reduced for faster processing  # Conservative limit to prevent token issues
        
        for result in search_results:
            metadata = result.metadata
            chunk_text = result.text
            
            prefix_parts = []
            if metadata.get('policy_uin'):
                prefix_parts.append(f"Policy UIN: {metadata['policy_uin']}")
            if metadata.get('section_header'):
                prefix_parts.append(f"Section: {metadata['section_header']}")
            if metadata.get('chunk_type'):
                prefix_parts.append(f"Type: {metadata['chunk_type']}")
            
            prefix = ", ".join(prefix_parts)
            formatted_chunk = f"[{prefix}]\n{chunk_text}\n" if prefix else f"{chunk_text}\n"
            
            # Check if adding this chunk would exceed the limit
            if total_length + len(formatted_chunk) > max_context_length:
                logger.warning(f"Context length limit reached ({total_length} chars). Truncating context.")
                break
            
            context_parts.append(formatted_chunk)
            total_length += len(formatted_chunk)
        
        logger.info(f"Context structured: {len(context_parts)} chunks, {total_length} characters")
        return "\n".join(context_parts)
    
    def _create_intelligent_user_prompt(self, query: str, structured_context: str, context_analysis: Dict[str, Any]) -> str:
        """
        Create a user prompt combining the query and structured context.
        Priority 2: Enhanced user prompt for better accuracy
        
        Args:
            query (str): The user's query.
            structured_context (str): Formatted context from search results.
            context_analysis (Dict[str, Any]): Analyzed context.
        
        Returns:
            str: The user prompt string.
        """
        prompt = f"QUERY: {query}\n\n"
        prompt += "CONTEXT:\n"
        prompt += "---\n"
        prompt += structured_context
        prompt += "---\n\n"
        
        # Priority 2: Specific instructions for the query
        prompt += "INSTRUCTIONS:\n"
        prompt += "- Answer the query using ONLY the information provided in the context above.\n"
        prompt += "- If the context contains the answer, provide it with ALL relevant details.\n"
        prompt += "- If the context does not contain the answer, explicitly state this.\n"
        prompt += "- Use exact terminology and numbers from the context.\n"
        prompt += "- Be comprehensive - include all relevant conditions, requirements, and details.\n"
        
        return prompt
    
    def _parse_context_aware_response(self, response_content: str) -> Dict[str, Any]:
        """
        Parse the LLM's JSON response.
        
        Args:
            response_content (str): The raw response content from the LLM.
        
        Returns:
            Dict[str, Any]: Parsed response dictionary.
        """
        try:
            response_json = json.loads(response_content)
            return {
                'answer': response_json.get('answer', ''),
                'confidence': float(response_json.get('confidence', 0.0)),
                'reasoning': response_json.get('reasoning', ''),
                'sources': response_json.get('sources', []),
                'additional_info': response_json.get('additional_info', {})
            }
        except json.JSONDecodeError:
            logger.error("Failed to parse LLM response as JSON")
            return {
                'answer': response_content,
                'confidence': 0.0,
                'reasoning': 'Response parsing failed',
                'sources': [],
                'error': True
            }
    
    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(RateLimitError)
    )
    async def _call_openai_with_backoff(
        self,
        system_prompt: str,
        user_prompt: str
    ):
        """
        Call OpenAI API with exponential backoff for rate limit handling
        
        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User prompt for the LLM
            
        Returns:
            OpenAI API response
            
        Raises:
            RateLimitError: If rate limit is exceeded after all retries
        """
        try:
            logger.debug("Making OpenAI API call...")
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            return response
            
        except RateLimitError as e:
            logger.warning(f"Rate limit exceeded. Retrying with exponential backoff... Error: {str(e)}")
            raise e  # Re-raise to trigger retry
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            raise e