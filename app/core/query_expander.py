# app/core/query_expander.py
"""
Priority 3: LLM-based Query Expansion
Generate related terms and synonyms to improve search coverage
"""
import logging
from typing import List, Dict, Any, Optional
import re

from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from openai import AsyncOpenAI, RateLimitError
from app.config.settings import settings
from app.core.domain_config import DomainType, DomainConfig

logger = logging.getLogger(__name__)

class QueryExpander:
    """
    LLM-based query expansion for insurance-specific terminology
    Generates synonyms and related terms to improve search coverage
    """
    
    def __init__(self, domain_type: DomainType = DomainType.INSURANCE):
        """Initialize the query expander with OpenAI client and domain configuration"""
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.LLM_MODEL
        
        # Domain configuration
        self.domain_config = DomainConfig(domain_type)
        self.domain_type = domain_type
        
        # Domain-specific expansion patterns
        self.expansion_patterns = self._get_domain_expansion_patterns()
    
    def _get_domain_expansion_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """Get domain-specific expansion patterns"""
        domain_patterns = self.domain_config.get_query_expansion_patterns()
        
        # Add domain-specific patterns based on domain type
        if self.domain_type == DomainType.INSURANCE:
            # Add insurance-specific patterns
            domain_patterns.update({
                'accident': {
                    'triggers': ['accident', 'accidental', 'injury'],
                    'synonyms': ['accidental injury', 'accidental death', 'personal accident', 'bodily injury']
                },
                'hospital': {
                    'triggers': ['hospital', 'hospitalization', 'medical'],
                    'synonyms': ['medical facility', 'healthcare facility', 'inpatient', 'medical treatment']
                },
                'opd': {
                    'triggers': ['opd', 'outpatient', 'clinic'],
                    'synonyms': ['outpatient department', 'clinic visit', 'doctor visit', 'consultation']
                },
                'waiting_period': {
                    'triggers': ['waiting period', 'waiting time', 'exclusion period'],
                    'synonyms': ['waiting period', 'moratorium period', 'pre-existing condition waiting', 'cooling period'],
                    'specific_terms': ['24 months', 'twenty four months', '2 years', 'two years', 'continuous coverage']
                },
                'numerical_details': {
                    'triggers': ['percentage', 'percent', 'months', 'years', 'days', 'amount'],
                    'synonyms': ['specific amount', 'exact percentage', 'time period', 'duration', 'numerical value'],
                    'patterns': ['\\d+\\s*months?', '\\d+\\s*years?', '\\d+\\s*%', '\\d+\\s*percent', 'Rs\\.?\\s*\\d+']
                },
                'medical_conditions': {
                    'triggers': ['cataract', 'hernia', 'ent', 'tonsils', 'adenoids', 'sinuses'],
                    'synonyms': ['medical condition', 'specific illness', 'surgical procedure', 'listed condition'],
                    'ent_specific': ['deviated nasal septum', 'chronic suppurative otitis media', 'csom', 'tonsillectomy', 'adenoidectomy'],
                    'list_terms': ['list of specific', 'surgical procedures', 'illness and surgical', 'medical procedures list']
                },
                'co_payment': {
                    'triggers': ['co-payment', 'copayment', 'zone', 'additional payment'],
                    'synonyms': ['zone based co-payment', 'additional co-payment', 'percentage deduction'],
                    'zone_terms': ['zone a', 'zone b', 'zone c', '15%', 'fifteen percent']
                }
            })
        elif self.domain_type == DomainType.LEGAL:
            # Add legal-specific patterns
            domain_patterns.update({
                'contract': {
                    'triggers': ['contract', 'agreement', 'party', 'obligation'],
                    'synonyms': ['legal document', 'binding agreement', 'contractual obligation', 'legal commitment']
                },
                'breach': {
                    'triggers': ['breach', 'violation', 'default', 'non-compliance'],
                    'synonyms': ['contract breach', 'agreement violation', 'default', 'non-performance']
                }
            })
        elif self.domain_type == DomainType.HR:
            # Add HR-specific patterns
            domain_patterns.update({
                'pto': {
                    'triggers': ['pto', 'paid time off', 'vacation', 'leave'],
                    'synonyms': ['personal time', 'vacation days', 'sick leave', 'personal days', 'time away']
                },
                'performance': {
                    'triggers': ['performance', 'evaluation', 'review', 'feedback'],
                    'synonyms': ['performance review', 'employee evaluation', 'performance assessment', 'feedback session']
                }
            })
        elif self.domain_type == DomainType.COMPLIANCE:
            # Add compliance-specific patterns
            domain_patterns.update({
                'audit': {
                    'triggers': ['audit', 'review', 'assessment', 'examination'],
                    'synonyms': ['compliance audit', 'regulatory review', 'compliance assessment', 'audit finding']
                },
                'violation': {
                    'triggers': ['violation', 'breach', 'non-compliance', 'penalty'],
                    'synonyms': ['compliance violation', 'regulatory breach', 'non-compliance', 'sanction']
                }
            })
        
        return domain_patterns
    
    def _create_domain_definition_prompt(self, query: str) -> str:
        """Create domain-specific definition prompt"""
        domain_name = self.domain_config.get_domain_name()
        domain_description = self.domain_config.get_domain_description()
        
        if self.domain_type == DomainType.INSURANCE:
            return f"""
You are an expert {domain_name} policy analyst. The user is asking for a definition or explanation of a {domain_name.lower()} term.

Query: "{query}"

Instructions:
- Generate keywords and section titles related to definitions and explanations in {domain_name.lower()} policies
- Focus on terms that {domain_name.lower()} professionals use to define concepts
- Include both formal definitions and informal explanations
- Consider different ways the same term might be defined in {domain_name.lower()} context
- Include common synonyms and related {domain_name.lower()} terminology
- Return only the terms, one per line, no explanations

Example for "What is the definition of hospital?":
hospital definition
what is a hospital
hospital criteria
hospital requirements
hospital standards
hospital classification
hospital meaning
hospital explanation
healthcare facility
medical institution
inpatient facility
day care treatment

Generate definition terms for: "{query}"
"""
        elif self.domain_type == DomainType.LEGAL:
            return f"""
You are an expert {domain_name} professional. The user is asking for a definition or explanation of a {domain_name.lower()} term.

Query: "{query}"

Instructions:
- Generate keywords and section titles related to definitions and explanations in {domain_name.lower()} documents
- Focus on terms that {domain_name.lower()} professionals use to define concepts
- Include both formal definitions and informal explanations
- Consider different ways the same term might be defined in {domain_name.lower()} context
- Include common synonyms and related {domain_name.lower()} terminology
- Return only the terms, one per line, no explanations

Generate definition terms for: "{query}"
"""
        else:
            return f"""
You are an expert {domain_name} professional. The user is asking for a definition or explanation of a {domain_name.lower()} term.

Query: "{query}"

Instructions:
- Generate keywords and section titles related to definitions and explanations in {domain_name.lower()} documents
- Focus on terms that {domain_name.lower()} professionals use to define concepts
- Include both formal definitions and informal explanations
- Consider different ways the same term might be defined in {domain_name.lower()} context
- Include common synonyms and related {domain_name.lower()} terminology
- Return only the terms, one per line, no explanations

Generate definition terms for: "{query}"
"""
    
    def _create_domain_expansion_prompt(self, query: str) -> str:
        """Create domain-specific expansion prompt"""
        domain_name = self.domain_config.get_domain_name()
        domain_description = self.domain_config.get_domain_description()
        
        return f"""
You are an expert {domain_name} professional. Generate 5-8 synonyms or related terms for the following {domain_name.lower()} query to help find relevant information in {domain_name.lower()} documents.

Query: "{query}"

Instructions:
- Generate terms that {domain_name.lower()} professionals would use
- Include both formal and informal {domain_name.lower()} terminology
- Consider different ways the same concept might be expressed in {domain_name.lower()} context
- Focus on {domain_name.lower()}-specific terminology and common synonyms
- Include related terms and concepts from the {domain_name.lower()} domain
- Return only the terms, one per line, no explanations

Generate terms for: "{query}"
"""
    
    async def expand_query(self, query: str) -> Dict[str, Any]:
        """
        Expand query with related terms and synonyms
        
        Args:
            query: Original user query
            
        Returns:
            Dictionary with expanded query terms and metadata
        """
        try:
            logger.info(f"Expanding query: '{query[:50]}...'")
            
            # Step 1: Pattern-based expansion
            pattern_expansions = self._pattern_based_expansion(query)
            
            # Step 2: LLM-based expansion
            llm_expansions = await self._llm_based_expansion(query)
            
            # Step 3: Combine and deduplicate
            all_expansions = pattern_expansions + llm_expansions
            unique_expansions = list(set(all_expansions))
            
            # Step 4: Create expanded query
            expanded_query = self._create_expanded_query(query, unique_expansions)
            
            result = {
                'original_query': query,
                'expanded_terms': unique_expansions,
                'expanded_query': expanded_query,
                'expansion_count': len(unique_expansions),
                'expansion_methods': {
                    'pattern_based': len(pattern_expansions),
                    'llm_based': len(llm_expansions)
                }
            }
            
            logger.info(f"Query expansion complete: {len(unique_expansions)} terms added")
            return result
            
        except Exception as e:
            logger.error(f"Query expansion failed: {str(e)}")
            return {
                'original_query': query,
                'expanded_terms': [],
                'expanded_query': query,
                'expansion_count': 0,
                'expansion_methods': {'pattern_based': 0, 'llm_based': 0},
                'error': str(e)
            }
    
    def _pattern_based_expansion(self, query: str) -> List[str]:
        """
        Pattern-based expansion using predefined insurance terminology
        
        Args:
            query: Original query
            
        Returns:
            List of expanded terms
        """
        expansions = []
        query_lower = query.lower()
        
        for category, patterns in self.expansion_patterns.items():
            # Check if query matches any triggers
            if any(trigger in query_lower for trigger in patterns['triggers']):
                expansions.extend(patterns['synonyms'])
        
        return expansions
    
    async def _llm_based_expansion(self, query: str) -> List[str]:
        """
        LLM-based expansion for generating related terms with contextual awareness
        
        Args:
            query: Original query
            
        Returns:
            List of expanded terms
        """
        try:
            # Detect query type for contextual expansion
            query_type = self._detect_query_type(query)
            
            # Create domain-specific prompt
            if query_type == 'procedural':
                # Multi-Query Retrieval: Generate sub-queries for procedural questions
                sub_queries = await self._generate_procedural_sub_queries(query)
                return sub_queries
            elif query_type == 'definition':
                prompt = self._create_domain_definition_prompt(query)
            else:
                # Domain-specific expansion for other query types
                prompt = self._create_domain_expansion_prompt(query)

            response = await self._call_openai_with_backoff(prompt)
            
            # Parse response
            content = response.choices[0].message.content.strip()
            terms = [term.strip() for term in content.split('\n') if term.strip()]
            
            # Clean up terms
            cleaned_terms = []
            for term in terms:
                # Remove numbering, bullets, etc.
                term = re.sub(r'^[\d\-\.\s]+', '', term)
                if term and len(term) > 2:
                    cleaned_terms.append(term)
            
            return cleaned_terms[:8]  # Limit to 8 terms
            
        except Exception as e:
            logger.error(f"LLM-based expansion failed: {str(e)}")
            return []
    
    def _create_expanded_query(self, original_query: str, expansions: List[str]) -> str:
        """
        Create an expanded query string for search
        
        Args:
            original_query: Original user query
            expansions: List of expanded terms
            
        Returns:
            Expanded query string
        """
        if not expansions:
            return original_query
        
        # Combine original query with expansions
        all_terms = [original_query] + expansions
        expanded_query = " OR ".join([f'"{term}"' for term in all_terms])
        
        return expanded_query
    
    async def expand_for_search(self, query: str) -> Dict[str, Any]:
        """
        Enhanced search-specific query expansion with numerical detail focus
        
        Args:
            query: Original query
            
        Returns:
            Search-optimized expansion
        """
        expansion_result = await self.expand_query(query)
        
        # Enhanced expansion for specific patterns
        original_query = query.lower()
        additional_terms = []
        
        # Numerical detail expansions
        import re
        if re.search(r'waiting\s+period|pre-existing', original_query):
            additional_terms.extend(['24 months', 'twenty four months', '2 years', 'continuous coverage'])
            
        if re.search(r'zone.*payment|co-?payment.*zone', original_query):
            additional_terms.extend(['15%', 'fifteen percent', 'zone a', 'zone b'])
            
        if 'ent' in original_query and 'conditions' in original_query:
            additional_terms.extend(['deviated nasal septum', 'csom', 'tonsils', 'adenoids', 'sinuses', 'list of specific', 'surgical procedures', 'illness and surgical', 'medical list'])
            
        if 'cumulative.*bonus|bonus.*cumulative' in original_query:
            additional_terms.extend(['10%', 'ten percent', '100%', 'hundred percent', 'claim-free'])
            
        if 'reset.*benefit|benefit.*reset' in original_query:
            additional_terms.extend(['unlimited', 'restores', '100%', 'annual sum insured'])
            
        if 'documents.*claim|claim.*documents' in original_query:
            additional_terms.extend(['photo id', 'hospital bills', 'discharge summary', 'investigation reports'])
            
        # Add these to expanded terms
        if additional_terms:
            expansion_result['expanded_terms'].extend(additional_terms)
            expansion_result['expanded_terms'] = list(dict.fromkeys(expansion_result['expanded_terms']))  # Remove duplicates
        
        # Add search-specific metadata
        expansion_result['search_metadata'] = {
            'has_definitions': any('definition' in term.lower() for term in expansion_result['expanded_terms']),
            'has_coverage': any('cover' in term.lower() for term in expansion_result['expanded_terms']),
            'has_exclusions': any('exclusion' in term.lower() for term in expansion_result['expanded_terms']),
            'has_claims': any('claim' in term.lower() for term in expansion_result['expanded_terms']),
            'has_numerical_details': any(re.search(r'\d+', term) for term in expansion_result['expanded_terms']),
            'query_intent': self._detect_query_intent(query)
        }
        
        return expansion_result
    
    def _detect_query_intent(self, query: str) -> str:
        """
        Detect the intent of the query for search optimization
        
        Args:
            query: User query
            
        Returns:
            Detected intent
        """
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['definition', 'define', 'means', 'what is']):
            return 'definition'
        elif any(word in query_lower for word in ['cover', 'coverage', 'benefit', 'include']):
            return 'coverage'
        elif any(word in query_lower for word in ['exclusion', 'exclude', 'not cover']):
            return 'exclusion'
        elif any(word in query_lower for word in ['claim', 'process', 'procedure']):
            return 'claim'
        else:
            return 'general'
    
    def _detect_query_type(self, query: str) -> str:
        """
        Detect the type of query for contextual expansion
        
        Args:
            query: User query
            
        Returns:
            Query type: 'procedural', 'definition', 'coverage', 'exclusion', 'general'
        """
        query_lower = query.lower()
        
        # Procedural queries (how to, process, steps)
        if any(word in query_lower for word in ['how to', 'process', 'procedure', 'steps', 'what is the process', 'how do i']):
            return 'procedural'
        
        # Definition queries
        elif any(word in query_lower for word in ['definition', 'define', 'means', 'what is', 'what does']):
            return 'definition'
        
        # Coverage queries
        elif any(word in query_lower for word in ['cover', 'coverage', 'benefit', 'include', 'what is covered']):
            return 'coverage'
        
        # Exclusion queries
        elif any(word in query_lower for word in ['exclusion', 'exclude', 'not cover', 'not covered', 'limitation']):
            return 'exclusion'
        
        else:
            return 'general'
    
    async def _generate_procedural_sub_queries(self, query: str) -> List[str]:
        """
        Multi-Query Retrieval: Generate multiple sub-queries for procedural questions
        
        Args:
            query: Original procedural query
            
        Returns:
            List of sub-queries for multi-query retrieval
        """
        try:
            prompt = f"""
You are an expert insurance policy analyst. The user is asking a procedural question about an insurance policy.

Original Query: "{query}"

Instructions:
Rephrase this query into THREE different sub-queries for a search system:

A) KEYWORD-FOCUSED: Focus on specific keywords and terms that would appear in policy documents
B) NATURAL LANGUAGE: Phrase as a natural language question that a person might ask
C) SECTION-FOCUSED: Focus on section titles and headers that might contain this information

Example for "What is the process for notifying a claim?":

A) "claim notification steps procedure documents required forms"
B) "How do I inform the insurance company about a hospitalization?"
C) "Claims Procedure and Required Documents section"

Generate three sub-queries for: "{query}"

Format your response as:
A) [keyword-focused query]
B) [natural language query] 
C) [section-focused query]
"""

            response = await self._call_openai_with_backoff(prompt, max_tokens=300)
            
            # Parse response
            content = response.choices[0].message.content.strip()
            
            # Extract sub-queries
            sub_queries = []
            lines = content.split('\n')
            
            for line in lines:
                line = line.strip()
                if line.startswith('A)') or line.startswith('B)') or line.startswith('C)'):
                    # Extract the query part after the letter and parenthesis
                    query_part = line[2:].strip()
                    if query_part:
                        sub_queries.append(query_part)
            
            # If parsing failed, fall back to simple expansion
            if len(sub_queries) < 3:
                logger.warning(f"Failed to parse sub-queries, using fallback for: {query}")
                return self._fallback_procedural_expansion(query)
            
            logger.info(f"Generated {len(sub_queries)} sub-queries for procedural query: {query}")
            return sub_queries
            
        except Exception as e:
            logger.error(f"Sub-query generation failed: {str(e)}")
            return self._fallback_procedural_expansion(query)
    
    def _fallback_procedural_expansion(self, query: str) -> List[str]:
        """
        Fallback expansion for procedural queries when LLM generation fails
        
        Args:
            query: Original procedural query
            
        Returns:
            List of fallback expansion terms
        """
        query_lower = query.lower()
        expansions = []
        
        # Extract key terms from the query
        if 'claim' in query_lower:
            expansions.extend([
                'claim notification process',
                'how to file a claim',
                'claim submission steps',
                'notification procedure',
                'claim reporting process',
                'steps to notify claim',
                'claim intimation process',
                'claim filing procedure'
            ])
        
        if 'process' in query_lower or 'procedure' in query_lower:
            expansions.extend([
                'procedure steps',
                'process requirements',
                'how to',
                'step by step',
                'procedure guidelines'
            ])
        
        if 'notify' in query_lower or 'notification' in query_lower:
            expansions.extend([
                'notification process',
                'how to notify',
                'notification requirements',
                'notification procedure',
                'notification steps'
            ])
        
        return expansions[:8]  # Limit to 8 terms
    
    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(RateLimitError)
    )
    async def _call_openai_with_backoff(
        self,
        prompt: str,
        max_tokens: int = 200
    ):
        """
        Call OpenAI API with exponential backoff for rate limit handling
        
        Args:
            prompt: User prompt for the LLM
            max_tokens: Maximum tokens for the response
            
        Returns:
            OpenAI API response
            
        Raises:
            RateLimitError: If rate limit is exceeded after all retries
        """
        try:
            logger.debug("Making OpenAI API call for query expansion...")
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.3
            )
            return response
            
        except RateLimitError as e:
            logger.warning(f"Rate limit exceeded during query expansion. Retrying with exponential backoff... Error: {str(e)}")
            raise e  # Re-raise to trigger retry
        except Exception as e:
            logger.error(f"OpenAI API call failed during query expansion: {str(e)}")
            raise e 