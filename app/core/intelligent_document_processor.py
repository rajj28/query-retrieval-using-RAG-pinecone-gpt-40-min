# app/core/intelligent_document_processor.py
"""
Phase 1: Intelligent Ingestion and Structured Chunking
Layout-aware, semantically complete chunking with rich metadata
"""
import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
from io import StringIO
import tiktoken
import os
import tempfile

# Unstructured imports for layout-aware processing
try:
    from unstructured.partition.auto import partition
    from unstructured.documents.elements import Table, Text, Title, NarrativeText
    from unstructured.partition.pdf import partition_pdf
    UNSTRUCTURED_AVAILABLE = True
except ImportError as e:
    UNSTRUCTURED_AVAILABLE = False

from app.core.domain_config import DomainConfig, DomainType

logger = logging.getLogger(__name__)

# Log unstructured availability
if UNSTRUCTURED_AVAILABLE:
    logger.info("unstructured library loaded successfully for layout-aware processing")
else:
    logger.warning("unstructured library not available. Using fallback table extraction.")

@dataclass
class DocumentMetadata:
    """Rich metadata for document chunks"""
    file_name: str
    policy_uin: Optional[str]
    company_name: Optional[str]
    policy_type: Optional[str]
    section_header: Optional[str]
    chunk_type: str  # 'text', 'table_markdown', 'header', 'definition', 'exclusion', 'benefit'
    chunk_index: int
    page_number: Optional[int]
    confidence_score: float

class IntelligentDocumentProcessor:
    """
    Advanced document processor with layout-aware chunking and rich metadata extraction
    """
    
    def __init__(self, domain_type: DomainType = DomainType.INSURANCE):
        # Domain configuration
        self.domain_config = DomainConfig(domain_type)
        self.domain_type = domain_type
        
        # Token counting setup
        try:
            self.tokenizer = tiktoken.encoding_for_model("text-embedding-3-large")
            self.max_tokens_per_chunk = 4000  # Reduced limit to prevent token issues
            self.target_tokens_per_chunk = 2000  # Optimal chunk size
        except Exception as e:
            logger.warning(f"Failed to initialize tokenizer: {e}")
            self.tokenizer = None
            self.max_tokens_per_chunk = 4000
            self.target_tokens_per_chunk = 2000
        
        # Insurance company patterns
        self.company_patterns = {
            'HDFC Ergo': [r'hdfc\s+ergo', r'hdfc.*ergo', r'hdfcergo'],
            'Bajaj Allianz': [r'bajaj\s+allianz', r'bajaj.*allianz', r'bajajallianz'],
            'ICICI Lombard': [r'icici\s+lombard', r'icici.*lombard', r'icicilombard'],
            'Cholamandalam': [r'cholamandalam', r'chola\s+ms', r'cholams'],
            'National Insurance': [r'national\s+insurance', r'national.*insurance'],
            'New India Assurance': [r'new\s+india\s+assurance', r'new.*india.*assurance'],
            'Star Health': [r'star\s+health', r'star.*health'],
            'Max Bupa': [r'max\s+bupa', r'max.*bupa']
        }
        
        # Policy UIN patterns (more comprehensive)
        self.uin_patterns = [
            r'\bUIN\s*:?\s*([A-Z]{3,}[0-9]{4,}[A-Z][0-9]{6,})\b',  # Standard UIN
            r'\b([A-Z]{3,}[A-Z]{2}[A-Z]{2}[0-9]{2}[0-9]{3}[A-Z][0-9]{6})\b',  # Structured UIN
            r'\b([A-Z]{6,}[0-9]{4,}[A-Z][0-9]{6,})\b',  # Variation 1
            r'\bPolicy\s+No\.?\s*:?\s*([A-Z0-9]{10,})\b',  # Policy number
            r'\bCertificate\s+No\.?\s*:?\s*([A-Z0-9]{10,})\b'  # Certificate number
        ]
        
        # Section header patterns (domain-specific)
        self.section_patterns = self._get_domain_section_patterns()
        
        # Sub-section patterns for detailed chunking
        self.subsection_patterns = [
            r'^[A-Z]\)\s+',  # A) B) C)
            r'^\([a-z]\)\s+',  # (a) (b) (c)
            r'^\d+\.\d+\.?\s+',  # 1.1 1.2
            r'^\([ivx]+\)\s+',  # (i) (ii) (iii)
            r'^•\s+',  # Bullet points
            r'^-\s+',  # Dash points
        ]
        
        # Table detection patterns (simplified for performance)
        self.table_patterns = [
            r'\|.*\|.*\|',  # Markdown-style tables only
        ]
        
        # Definition patterns
        self.definition_patterns = [
            r'^([A-Z][A-Za-z\s]+)\s*:?\s*means\s+(.+)',
            r'^"([^"]+)"\s*:?\s*(.+)',
            r'^([A-Z][A-Za-z\s]+)\s*-\s*(.+)',
            r'^\d+\.\d+\s+([A-Z][A-Za-z\s]+)\s*:?\s*(.+)'
        ]
    
    def _get_domain_section_patterns(self) -> Dict[str, List[str]]:
        """Get domain-specific section patterns"""
        domain_patterns = self.domain_config.get_section_patterns()
        
        # Convert domain patterns to our internal format
        section_patterns = {}
        
        # Map domain patterns to section types based on domain
        if self.domain_type == DomainType.INSURANCE:
            section_patterns = {
                'definitions': [
                    r'^\d+\.?\s*DEFINITIONS?',
                    r'^DEFINITIONS?',
                    r'^Section\s*\d+\.?\s*DEFINITIONS?',
                    r'^CLAUSE\s*\d+\.?\s*DEFINITIONS?',
                    r'^PART\s*\d+\.?\s*DEFINITIONS?',
                    r'^CHAPTER\s*\d+\.?\s*DEFINITIONS?',
                    r'^MEANING\s+OF\s+WORDS',
                    r'^INTERPRETATION',
                    r'^GLOSSARY'
                ],
                'coverage': [
                    r'^\d+\.?\s*COVERAGE',
                    r'^\d+\.?\s*BENEFITS?',
                    r'^\d+\.?\s*WHAT\s+IS\s+COVERED',
                    r'^\d+\.?\s*SCOPE\s+OF\s+COVER',
                    r'^COVERAGE',
                    r'^BENEFITS?',
                    r'^WHAT\s+IS\s+COVERED',
                    r'^SCOPE\s+OF\s+COVER',
                    r'^Section\s*\d+\.?\s*COVERAGE',
                    r'^Section\s*\d+\.?\s*BENEFITS?'
                ],
                'exclusions': [
                    r'^\d+\.?\s*EXCLUSIONS?',
                    r'^\d+\.?\s*WHAT\s+IS\s+NOT\s+COVERED',
                    r'^\d+\.?\s*LIMITATIONS?',
                    r'^EXCLUSIONS?',
                    r'^WHAT\s+IS\s+NOT\s+COVERED',
                    r'^LIMITATIONS?',
                    r'^Section\s*\d+\.?\s*EXCLUSIONS?',
                    r'^Section\s*\d+\.?\s*LIMITATIONS?'
                ],
                'claims': [
                    r'^\d+\.?\s*CLAIMS?',
                    r'^\d+\.?\s*CLAIM\s+PROCEDURE',
                    r'^\d+\.?\s*HOW\s+TO\s+CLAIM',
                    r'^CLAIMS?',
                    r'^CLAIM\s+PROCEDURE',
                    r'^HOW\s+TO\s+CLAIM',
                    r'^Section\s*\d+\.?\s*CLAIMS?'
                ],
                'conditions': [
                    r'^\d+\.?\s*CONDITIONS?',
                    r'^\d+\.?\s*GENERAL\s+CONDITIONS?',
                    r'^\d+\.?\s*SPECIAL\s+CONDITIONS?',
                    r'^\d+\.?\s*TERMS\s+AND\s+CONDITIONS?',
                    r'^CONDITIONS?',
                    r'^GENERAL\s+CONDITIONS?',
                    r'^SPECIAL\s+CONDITIONS?',
                    r'^TERMS\s+AND\s+CONDITIONS?',
                    r'^Section\s*\d+\.?\s*CONDITIONS?'
                ],
                'schedule': [
                    r'^\d+\.?\s*SCHEDULE',
                    r'^\d+\.?\s*POLICY\s+SCHEDULE',
                    r'^\d+\.?\s*CERTIFICATE\s+SCHEDULE',
                    r'^SCHEDULE',
                    r'^POLICY\s+SCHEDULE',
                    r'^CERTIFICATE\s+SCHEDULE',
                    r'^Section\s*\d+\.?\s*SCHEDULE'
                ],
                'waiting_periods': [
                    r'^\d+\.?\s*WAITING\s+PERIODS?',
                    r'^\d+\.?\s*EXCLUSION\s+PERIODS?',
                    r'^\d+\.?\s*PRE-EXISTING\s+DISEASE',
                    r'^WAITING\s+PERIODS?',
                    r'^EXCLUSION\s+PERIODS?',
                    r'^PRE-EXISTING\s+DISEASE',
                    r'^Section\s*\d+\.?\s*WAITING\s+PERIODS?'
                ],
                'grace_period': [
                    r'^\d+\.?\s*GRACE\s+PERIOD',
                    r'^\d+\.?\s*RENEWAL',
                    r'^\d+\.?\s*PREMIUM\s+PAYMENT',
                    r'^GRACE\s+PERIOD',
                    r'^RENEWAL',
                    r'^PREMIUM\s+PAYMENT',
                    r'^Section\s*\d+\.?\s*GRACE\s+PERIOD'
                ]
            }
        elif self.domain_type == DomainType.LEGAL:
            section_patterns = {
                'definitions': [p for p in domain_patterns if 'DEFINITION' in p.upper()],
                'obligations': [p for p in domain_patterns if any(term in p.upper() for term in ['REPRESENTATION', 'WARRANTY', 'COVENANT'])],
                'liability': [p for p in domain_patterns if any(term in p.upper() for term in ['LIABILITY', 'INDEMNIFICATION'])],
                'termination': [p for p in domain_patterns if 'TERMINATION' in p.upper()],
                'disputes': [p for p in domain_patterns if any(term in p.upper() for term in ['JURISDICTION', 'ARBITRATION', 'DISPUTE'])],
                'payment': [p for p in domain_patterns if any(term in p.upper() for term in ['PAYMENT', 'COMPENSATION', 'FEE'])]
            }
        elif self.domain_type == DomainType.HR:
            section_patterns = {
                'employment': [p for p in domain_patterns if 'EMPLOYMENT' in p.upper()],
                'compensation': [p for p in domain_patterns if any(term in p.upper() for term in ['COMPENSATION', 'SALARY', 'BENEFIT'])],
                'time_off': [p for p in domain_patterns if any(term in p.upper() for term in ['TIME OFF', 'VACATION', 'LEAVE'])],
                'performance': [p for p in domain_patterns if 'PERFORMANCE' in p.upper()],
                'termination': [p for p in domain_patterns if 'TERMINATION' in p.upper()],
                'conduct': [p for p in domain_patterns if any(term in p.upper() for term in ['CONDUCT', 'POLICY', 'RULE'])]
            }
        elif self.domain_type == DomainType.COMPLIANCE:
            section_patterns = {
                'regulations': [p for p in domain_patterns if any(term in p.upper() for term in ['COMPLIANCE', 'REGULATION'])],
                'policies': [p for p in domain_patterns if 'POLICY' in p.upper()],
                'audits': [p for p in domain_patterns if 'AUDIT' in p.upper()],
                'reporting': [p for p in domain_patterns if 'REPORTING' in p.upper()],
                'violations': [p for p in domain_patterns if any(term in p.upper() for term in ['VIOLATION', 'PENALTY', 'SANCTION'])]
            }
        else:
            # Generic patterns for other domains
            section_patterns = {
                'general': domain_patterns
            }
        
        # Add fallback patterns for each section type
        for section_type, patterns in section_patterns.items():
            if not patterns:
                # Add generic patterns if no domain-specific ones found
                section_patterns[section_type] = [
                    rf'^\d+\.\s*{section_type.upper()}',
                    rf'^Section\s*\d+\.?\s*{section_type.upper()}'
                ]
        
        return section_patterns
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the embedding model's tokenizer"""
        if not text or not self.tokenizer:
            return len(text.split())  # Fallback to word count
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}")
            return len(text.split())  # Fallback to word count
    
    def _should_subdivide_chunk(self, text: str) -> bool:
        """Determine if a chunk should be subdivided based on token count"""
        token_count = self._count_tokens(text)
        return token_count > self.max_tokens_per_chunk
    
    def _smart_subdivide_chunk(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Smart sub-chunking with metadata inheritance
        Splits large chunks while preserving semantic boundaries and metadata
        """
        if not self._should_subdivide_chunk(text):
            return [{'text': text, 'metadata': metadata}]
        
        logger.info(f"Subdividing large chunk ({self._count_tokens(text)} tokens) with metadata inheritance")
        
        sub_chunks = []
        paragraphs = text.split('\n\n')
        current_chunk = []
        current_tokens = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            paragraph_tokens = self._count_tokens(paragraph)
            
            # If this single paragraph is too large, split it further
            if paragraph_tokens > self.max_tokens_per_chunk:
                # Split large paragraph into sentences
                sentences = re.split(r'[.!?]+', paragraph)
                sentence_chunk = []
                sentence_tokens = 0
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    sentence_token_count = self._count_tokens(sentence)
                    
                    # If adding this sentence would exceed limit, create new chunk
                    if sentence_tokens + sentence_token_count > self.target_tokens_per_chunk and sentence_chunk:
                        chunk_text = '. '.join(sentence_chunk) + '.'
                        sub_chunks.append({
                            'text': chunk_text,
                            'metadata': {
                                **metadata,
                                'is_sub_chunk': True,
                                'sub_chunk_index': len(sub_chunks),
                                'parent_chunk_tokens': self._count_tokens(text),
                                'sub_chunk_tokens': sentence_tokens,
                                'split_level': 'sentence'
                            }
                        })
                        sentence_chunk = [sentence]
                        sentence_tokens = sentence_token_count
                    else:
                        sentence_chunk.append(sentence)
                        sentence_tokens += sentence_token_count
                
                # Add final sentence chunk
                if sentence_chunk:
                    chunk_text = '. '.join(sentence_chunk) + '.'
                    sub_chunks.append({
                        'text': chunk_text,
                        'metadata': {
                            **metadata,
                            'is_sub_chunk': True,
                            'sub_chunk_index': len(sub_chunks),
                            'parent_chunk_tokens': self._count_tokens(text),
                            'sub_chunk_tokens': sentence_tokens,
                            'split_level': 'sentence'
                        }
                    })
                
                continue
            
            # If adding this paragraph would exceed target, start new chunk
            if current_tokens + paragraph_tokens > self.target_tokens_per_chunk and current_chunk:
                # Create chunk from accumulated paragraphs
                chunk_text = '\n\n'.join(current_chunk)
                sub_chunks.append({
                    'text': chunk_text,
                    'metadata': {
                        **metadata,  # Inherit all parent metadata
                        'is_sub_chunk': True,
                        'sub_chunk_index': len(sub_chunks),
                        'parent_chunk_tokens': self._count_tokens(text),
                        'sub_chunk_tokens': current_tokens,
                        'split_level': 'paragraph'
                    }
                })
                
                # Reset for next chunk
                current_chunk = [paragraph]
                current_tokens = paragraph_tokens
            else:
                # Add to current chunk
                current_chunk.append(paragraph)
                current_tokens += paragraph_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            sub_chunks.append({
                'text': chunk_text,
                'metadata': {
                    **metadata,  # Inherit all parent metadata
                    'is_sub_chunk': True,
                    'sub_chunk_index': len(sub_chunks),
                    'parent_chunk_tokens': self._count_tokens(text),
                    'sub_chunk_tokens': current_tokens,
                    'split_level': 'paragraph'
                }
            })
        
        logger.info(f"Created {len(sub_chunks)} sub-chunks from large chunk")
        return sub_chunks
    
    async def process_document_intelligently(self, text: str, source: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Intelligently process document with layout-aware chunking and rich metadata
        """
        
        logger.info(f"Starting intelligent processing of document: {source}")
        
        # Phase 1A: Extract document-level metadata
        doc_metadata = await self._extract_document_metadata(text, source)
        
        # Phase 1B: Detect and extract tables using enhanced method
        tables, text_without_tables = await self._extract_tables_enhanced(text, source)
        
        # Phase 1C: Identify document structure
        document_structure = await self._analyze_document_structure(text_without_tables)
        
        # Phase 1D: Perform layout-aware chunking
        intelligent_chunks = await self._create_intelligent_chunks(
            text_without_tables, 
            document_structure, 
            doc_metadata
        )
        
        # Phase 1E: Process tables as separate chunks
        table_chunks = await self._process_tables_as_chunks(tables, doc_metadata)
        
        # Phase 1F: Combine and enrich all chunks
        all_chunks = intelligent_chunks + table_chunks
        enriched_chunks = await self._enrich_chunks_with_metadata(all_chunks, doc_metadata)
        
        # Compile results
        result = {
            'text': text,
            'chunks': enriched_chunks,
            'document_metadata': doc_metadata.__dict__,
            'document_structure': document_structure,
            'tables_extracted': len(tables),
            'processing_stats': {
                'total_chunks': len(enriched_chunks),
                'text_chunks': len(intelligent_chunks),
                'table_chunks': len(table_chunks),
                'sections_identified': len(document_structure.get('sections', [])),
                'definitions_found': len([c for c in enriched_chunks if c.get('metadata', {}).get('chunk_type') == 'definition']),
                'exclusions_found': len([c for c in enriched_chunks if c.get('metadata', {}).get('chunk_type') == 'exclusion']),
                'benefits_found': len([c for c in enriched_chunks if c.get('metadata', {}).get('chunk_type') == 'benefit']),
                'sub_chunks_created': len([c for c in enriched_chunks if c.get('metadata', {}).get('is_sub_chunk', False)]),
                'total_tokens': sum(self._count_tokens(c.get('text', '')) for c in enriched_chunks),
                'avg_tokens_per_chunk': sum(self._count_tokens(c.get('text', '')) for c in enriched_chunks) // len(enriched_chunks) if enriched_chunks else 0
            }
        }
        
        logger.info(f"Intelligent processing complete: {len(enriched_chunks)} chunks, {len(tables)} tables extracted")
        return result
    
    async def _extract_document_metadata(self, text: str, source: str) -> DocumentMetadata:
        """Extract comprehensive document metadata"""
        
        # Extract file name
        file_name = source.split('/')[-1] if '/' in source else source
        
        # Extract Policy UIN
        policy_uin = None
        for pattern in self.uin_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                policy_uin = match.group(1)
                break
        
        # Extract company name
        company_name = None
        company_confidence = 0
        text_lower = text.lower()
        
        for company, patterns in self.company_patterns.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                if matches > company_confidence:
                    company_name = company
                    company_confidence = matches
        
        # Determine policy type
        policy_type = await self._determine_policy_type(text)
        
        return DocumentMetadata(
            file_name=file_name,
            policy_uin=policy_uin,
            company_name=company_name,
            policy_type=policy_type,
            section_header=None,  # Will be set per chunk
            chunk_type='document',
            chunk_index=0,
            page_number=None,
            confidence_score=min(company_confidence / 10, 1.0)
        )
    
    async def _determine_policy_type(self, text: str) -> Optional[str]:
        """Determine policy type from content"""
        
        policy_indicators = {
            'Health Insurance': ['health', 'medical', 'mediclaim', 'hospitalization', 'treatment'],
            'Travel Insurance': ['travel', 'trip', 'journey', 'overseas', 'domestic travel'],
            'Motor Insurance': ['motor', 'vehicle', 'car', 'two wheeler', 'automobile'],
            'Home Insurance': ['home', 'house', 'property', 'dwelling', 'griha raksha'],
            'Life Insurance': ['life', 'death', 'survival', 'endowment', 'term'],
            'Personal Accident': ['personal accident', 'accidental death', 'disability', 'injury']
        }
        
        text_lower = text.lower()
        policy_scores = {}
        
        for policy_type, indicators in policy_indicators.items():
            score = sum(text_lower.count(indicator) for indicator in indicators)
            if score > 0:
                policy_scores[policy_type] = score
        
        if policy_scores:
            return max(policy_scores, key=policy_scores.get)
        
        return None
    
    async def _extract_tables_enhanced(self, text: str, source: str) -> Tuple[List[Dict], str]:
        """Enhanced table extraction using unstructured library when available"""
        
        # Check if source is a file path and unstructured is available
        if UNSTRUCTURED_AVAILABLE and (source.endswith('.pdf') or source.endswith('.docx')):
            try:
                # Create a temporary file if source is not a file path
                if not os.path.exists(source):
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                        f.write(text)
                        temp_file = f.name
                    try:
                        tables, processed_text = await self._extract_tables_with_unstructured(temp_file)
                    finally:
                        os.unlink(temp_file)
                else:
                    tables, processed_text = await self._extract_tables_with_unstructured(source)
                
                return tables, processed_text
                
            except Exception as e:
                logger.warning(f"Failed to use unstructured for table extraction: {e}")
                # Fall back to text-based extraction
                return await self._extract_tables(text)
        else:
            # Use fallback method
            return await self._extract_tables(text)
    
    async def _extract_tables_with_unstructured(self, file_path: str) -> Tuple[List[Dict], str]:
        """Extract tables using unstructured library"""
        try:
            # Use unstructured to partition the document
            elements = partition(file_path)
            
            tables = []
            text_elements = []
            
            for element in elements:
                if isinstance(element, Table):
                    # Convert table to markdown
                    table_data = element.metadata.get('text_as_html', '')
                    if table_data:
                        # Convert HTML table to markdown
                        markdown_table = await self._html_table_to_markdown(table_data)
                        tables.append({
                            'original_text': element.text,
                            'markdown': markdown_table,
                            'start_line': 0,  # Unstructured doesn't provide line numbers
                            'end_line': 0,
                            'row_count': len(element.text.split('\n')),
                            'source': 'unstructured'
                        })
                else:
                    # Keep non-table elements as text
                    text_elements.append(element.text)
            
            # Combine all text elements
            processed_text = '\n\n'.join(text_elements)
            
            logger.info(f"Extracted {len(tables)} tables using unstructured library")
            return tables, processed_text
            
        except Exception as e:
            logger.error(f"Error in unstructured table extraction: {e}")
            # Fall back to text-based extraction
            return await self._extract_tables(text)
    
    async def _html_table_to_markdown(self, html_table: str) -> str:
        """Convert HTML table to markdown format"""
        try:
            # Simple HTML table to markdown conversion
            lines = html_table.split('\n')
            markdown_lines = []
            
            for line in lines:
                if '<tr>' in line:
                    # Start of table row
                    continue
                elif '</tr>' in line:
                    # End of table row, add separator
                    markdown_lines.append('|')
                    continue
                elif '<td>' in line or '<th>' in line:
                    # Extract cell content
                    cell_content = line.replace('<td>', '').replace('</td>', '').replace('<th>', '').replace('</th>', '').strip()
                    markdown_lines.append(f"| {cell_content} ")
                else:
                    # Regular text
                    markdown_lines.append(line)
            
            return '\n'.join(markdown_lines)
            
        except Exception as e:
            logger.warning(f"Failed to convert HTML table to markdown: {e}")
            return html_table
    
    async def _extract_tables(self, text: str) -> Tuple[List[Dict], str]:
        """Extract and convert tables to markdown format (fallback method)"""
        
        tables = []
        lines = text.split('\n')
        table_lines = []
        in_table = False
        table_start = 0
        
        # Limit processing to prevent hanging
        max_lines_to_process = min(len(lines), 10000)  # Process max 10k lines
        
        for i, line in enumerate(lines[:max_lines_to_process]):
            line_stripped = line.strip()
            
            # Detect table patterns with timeout protection
            is_table_line = False
            try:
                # Only check for markdown-style tables for now
                is_table_line = '|' in line and line.count('|') >= 2
            except Exception as e:
                logger.warning(f"Table detection error on line {i}: {str(e)}")
                continue
            
            if is_table_line:
                if not in_table:
                    in_table = True
                    table_start = i
                table_lines.append(line)
            else:
                if in_table and table_lines:
                    # End of table, process it
                    table_text = '\n'.join(table_lines)
                    markdown_table = await self._convert_to_markdown_table(table_text)
                    
                    if markdown_table:
                        tables.append({
                            'original_text': table_text,
                            'markdown': markdown_table,
                            'start_line': table_start,
                            'end_line': i - 1,
                            'row_count': len(table_lines)
                        })
                    
                    table_lines = []
                    in_table = False
        
        # Handle table at end of document
        if in_table and table_lines:
            table_text = '\n'.join(table_lines)
            markdown_table = await self._convert_to_markdown_table(table_text)
            if markdown_table:
                tables.append({
                    'original_text': table_text,
                    'markdown': markdown_table,
                    'start_line': table_start,
                    'end_line': len(lines) - 1,
                    'row_count': len(table_lines)
                })
        
        # Remove table content from text
        text_without_tables = text
        for table in reversed(tables):  # Remove from end to preserve line numbers
            lines = text_without_tables.split('\n')
            lines[table['start_line']:table['end_line']+1] = ['']
            text_without_tables = '\n'.join(lines)
        
        logger.info(f"Extracted {len(tables)} tables from document (fallback method)")
        return tables, text_without_tables
    
    async def _convert_to_markdown_table(self, table_text: str) -> Optional[str]:
        """Convert table text to clean markdown format"""
        
        lines = [line.strip() for line in table_text.split('\n') if line.strip()]
        if len(lines) < 2:
            return None
        
        # Try to detect column separators
        separators = ['|', '\t', '  ', '   ']  # Various possible separators
        
        for sep in separators:
            if sep in lines[0]:
                try:
                    # Split lines by separator
                    rows = []
                    for line in lines:
                        if sep == '  ' or sep == '   ':
                            # Handle space-separated columns
                            columns = re.split(r'\s{2,}', line)
                        else:
                            columns = [col.strip() for col in line.split(sep)]
                        
                        if len(columns) > 1:
                            rows.append(columns)
                    
                    if len(rows) >= 2:
                        # Create markdown table
                        markdown_lines = []
                        
                        # Header
                        header = '| ' + ' | '.join(rows[0]) + ' |'
                        markdown_lines.append(header)
                        
                        # Separator
                        separator = '|' + '|'.join([' --- ' for _ in rows[0]]) + '|'
                        markdown_lines.append(separator)
                        
                        # Data rows
                        for row in rows[1:]:
                            if len(row) == len(rows[0]):  # Ensure consistent column count
                                data_row = '| ' + ' | '.join(row) + ' |'
                                markdown_lines.append(data_row)
                        
                        return '\n'.join(markdown_lines)
                
                except Exception as e:
                    logger.debug(f"Failed to parse table with separator '{sep}': {str(e)}")
                    continue
        
        return None
    
    async def _analyze_document_structure(self, text: str) -> Dict[str, Any]:
        """Enhanced layout-based document structure analysis for comprehensive chunking"""
        
        structure = {
            'sections': [],
            'all_headers': [],  # Store ALL detected headers
            'section_boundaries': {},
            'total_lines': 0,
            'section_count': 0
        }
        
        lines = text.split('\n')
        structure['total_lines'] = len(lines)
        
        # Enhanced header detection patterns for layout-based chunking
        header_patterns = [
            # Main sections
            (r'^\s*\d+\.?\s*[A-Z][A-Z\s]{3,}', 'main_section', 1),  # "1. DEFINITIONS"
            (r'^[A-Z][A-Z\s]{5,}\s*$', 'main_section', 1),  # "DEFINITIONS"
            
            # Subsections
            (r'^\s*\d+\.\d+\.?\s+[A-Z]', 'subsection', 2),  # "1.1 Hospital"
            (r'^\s*\d+\.\d+\.\d+\.?\s+[A-Z]', 'sub_subsection', 3),  # "1.1.1 Definition"
            
            # Lettered sections
            (r'^\s*[A-Z]\.?\s+[A-Z]', 'lettered_section', 2),  # "A. Coverage"
            (r'^\s*\([a-z]\)\s*[A-Z]', 'lettered_subsection', 3),  # "(a) Coverage"
            
            # Roman numerals
            (r'^\s*[IVX]+\.?\s+[A-Z]', 'roman_section', 2),  # "I. Coverage"
            (r'^\s*\([ivx]+\)\s*[A-Z]', 'roman_subsection', 3),  # "(i) Coverage"
            
            # Special sections
            (r'^Section\s*\d+', 'named_section', 1),
            (r'^Clause\s*\d+', 'named_section', 1),
            (r'^Part\s*\d+', 'named_section', 1),
            (r'^Chapter\s*\d+', 'named_section', 1),
            
            # Bullet points and lists
            (r'^\s*[\•\-\*]\s+[A-Z]', 'bullet_point', 4),  # "• Coverage"
            (r'^\s*\d+\)\s+[A-Z]', 'numbered_list', 4),  # "1) Coverage"
        ]
        
        # Detect ALL headers for comprehensive chunking
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped or len(line_stripped) < 3:
                continue
            
            header_detected = False
            
            # Check against all header patterns
            for pattern, header_type, level in header_patterns:
                try:
                    if re.match(pattern, line_stripped, re.IGNORECASE):
                        # Classify section content
                        section_type = self._classify_section_by_content(line_stripped)
                        
                        header_info = {
                            'line_number': i,
                            'text': line_stripped,
                            'type': header_type,
                            'level': level,
                            'section_type': section_type or 'general',
                            'content_start': i + 1
                        }
                        
                        structure['all_headers'].append(header_info)
                        header_detected = True
                        structure['section_count'] += 1
                        
                        logger.debug(f"Found {header_type} (level {level}): {line_stripped}")
                        break
                        
                except Exception as e:
                    logger.debug(f"Pattern matching error: {e}")
                    continue
            
            # Also check for domain-specific patterns
            if not header_detected:
                for section_name, patterns in self.section_patterns.items():
                    for pattern in patterns:
                        try:
                            if re.match(pattern, line_stripped, re.IGNORECASE):
                                header_info = {
                                    'line_number': i,
                                    'text': line_stripped,
                                    'type': 'domain_specific',
                                    'level': 2,
                                    'section_type': section_name,
                                    'content_start': i + 1
                                }
                                structure['all_headers'].append(header_info)
                                structure['section_count'] += 1
                                break
                        except Exception as e:
                            logger.debug(f"Domain pattern matching error: {e}")
                            continue
            
        
        # Calculate hierarchical content boundaries to avoid empty sections
        for i, header in enumerate(structure['all_headers']):
            # Find the end of this section based on hierarchy
            content_end = len(lines) - 1  # Default to end of document
            
            # Look for the next header at the same or higher level
            for j in range(i + 1, len(structure['all_headers'])):
                next_header = structure['all_headers'][j]
                if next_header['level'] <= header['level']:
                    # Found a peer or parent section - this is our boundary
                    content_end = next_header['line_number'] - 1
                    break
            
            header['content_end'] = content_end
            
            # Only create sections with meaningful content
            content_lines = lines[header['content_start']:content_end + 1]
            content_text = '\n'.join(content_lines).strip()
            
            if content_text and len(content_text) > 20:  # Only sections with substantial content
                section_key = f"{header['text'][:50]}_{i}"  # Unique key
                structure['section_boundaries'][section_key] = {
                    'start': header['content_start'],
                    'end': content_end,
                    'title': header['text'],
                    'type': header['section_type'],
                    'header_type': header['type'],
                    'level': header['level'],
                    'content_length': len(content_text)
                }
        
        logger.info(f"Document structure analysis complete: {structure['section_count']} sections identified")
        return structure
    
    def _classify_section_by_content(self, header_text: str) -> Optional[str]:
        """Classify section type based on header content"""
        header_lower = header_text.lower()
        
        # Direct keyword matching
        if any(word in header_lower for word in ['definition', 'meaning', 'interpretation', 'glossary']):
            return 'definitions'
        elif any(word in header_lower for word in ['coverage', 'benefit', 'what is covered', 'scope of cover']):
            return 'coverage'
        elif any(word in header_lower for word in ['exclusion', 'what is not covered', 'limitation', 'not covered']):
            return 'exclusions'
        elif any(word in header_lower for word in ['claim', 'claim procedure', 'how to claim']):
            return 'claims'
        elif any(word in header_lower for word in ['condition', 'general condition', 'special condition', 'terms and condition']):
            return 'conditions'
        elif any(word in header_lower for word in ['schedule', 'policy schedule', 'certificate schedule']):
            return 'schedule'
        elif any(word in header_lower for word in ['waiting period', 'exclusion period', 'pre-existing disease']):
            return 'waiting_periods'
        elif any(word in header_lower for word in ['grace period', 'renewal', 'premium payment']):
            return 'grace_period'
        
        return None
    
    async def _create_intelligent_chunks(self, text: str, structure: Dict, doc_metadata: DocumentMetadata) -> List[Dict]:
        """Create semantically complete chunks based on document structure"""
        
        chunks = []
        lines = text.split('\n')
        
        # If we have clear sections, chunk by section
        if structure.get('all_headers') or structure.get('sections'):
            chunks.extend(await self._chunk_by_sections(lines, structure, doc_metadata))
        else:
            # Fallback to intelligent paragraph chunking
            chunks.extend(await self._chunk_by_paragraphs(text, doc_metadata))
        
        return chunks
    
    async def _chunk_by_sections(self, lines: List[str], structure: Dict, doc_metadata: DocumentMetadata) -> List[Dict]:
        """Enhanced layout-based chunking using ALL detected headers"""
        
        chunks = []
        all_headers = structure.get('all_headers', [])
        
        if not all_headers:
            # Fallback to old method if no headers detected
            return await self._chunk_by_paragraphs('\n'.join(lines), doc_metadata)
        
        # Process each detected section with hierarchical logic
        processed_ranges = set()  # Track processed line ranges to avoid duplication
        
        for i, header in enumerate(all_headers):
            # Calculate content boundaries hierarchically
            content_start = header['content_start']
            content_end = len(lines) - 1  # Default to end of document
            
            # Find the next header at the same or higher level
            for j in range(i + 1, len(all_headers)):
                next_header = all_headers[j]
                if next_header['level'] <= header['level']:
                    content_end = next_header['line_number'] - 1
                    break
            
            # Check if this range has already been processed by a parent section
            range_key = (content_start, content_end)
            if range_key in processed_ranges:
                continue
            
            section_lines = lines[content_start:content_end + 1]
            section_text = '\n'.join(section_lines).strip()
            
            if not section_text or len(section_text) < 20:  # Require substantial content
                continue
            
            # Mark this range as processed
            processed_ranges.add(range_key)
            
            # Create rich metadata for this section
            base_metadata = {
                'section_header': header['text'],
                'section_type': header['section_type'],
                'chunk_type': header['section_type'],
                'header_type': header['type'],
                'header_level': header['level'],
                'start_line': content_start,
                'end_line': content_end,
                'is_complete_section': True,
                'section_number': i + 1,
                'content_length': len(section_text)
            }
            
            # Smart chunking based on content size and importance
            token_count = self._count_tokens(section_text)
            
            if token_count > self.max_tokens_per_chunk:
                # Large section - needs subdivision
                sub_chunks = self._smart_subdivide_chunk(section_text, base_metadata)
                chunks.extend(sub_chunks)
                logger.debug(f"Subdivided large section '{header['text']}' into {len(sub_chunks)} chunks")
                
            elif token_count > self.target_tokens_per_chunk and header['level'] <= 2:
                # Medium section with high importance - keep as one chunk but note it's large
                base_metadata['is_large_chunk'] = True
                chunks.append({
                    'text': section_text,
                    'metadata': base_metadata
                })
                logger.debug(f"Kept large important section '{header['text']}' as single chunk")
                
            else:
                # Regular sized section - single chunk
                chunks.append({
                    'text': section_text,
                    'metadata': base_metadata
                })
                logger.debug(f"Created chunk for section '{header['text']}' ({token_count} tokens)")
        
        # Handle any remaining content between sections
        orphaned_chunks = await self._capture_orphaned_content(lines, all_headers, doc_metadata)
        chunks.extend(orphaned_chunks)
        
        # If we have very few chunks despite many sections, use a different strategy
        if len(chunks) < 50 and len(all_headers) > 100:
            logger.warning(f"Only {len(chunks)} chunks created from {len(all_headers)} sections. Using fallback chunking.")
            fallback_chunks = await self._fallback_comprehensive_chunking(lines, all_headers, doc_metadata)
            chunks.extend(fallback_chunks)
        
        logger.info(f"Layout-based chunking complete: {len(chunks)} chunks from {len(all_headers)} sections")
        return chunks
    
    async def _capture_orphaned_content(self, lines: List[str], all_headers: List[Dict], doc_metadata: DocumentMetadata) -> List[Dict]:
        """Capture content that falls between detected sections"""
        
        orphaned_chunks = []
        
        if not all_headers:
            return orphaned_chunks
        
        # Check for content before the first header
        first_header_start = all_headers[0]['line_number']
        if first_header_start > 0:
            orphaned_text = '\n'.join(lines[:first_header_start]).strip()
            if orphaned_text and len(orphaned_text) > 20:
                orphaned_chunks.append({
                    'text': orphaned_text,
                    'metadata': {
                        'section_header': 'Document Preamble',
                        'section_type': 'preamble',
                        'chunk_type': 'preamble',
                        'header_type': 'orphaned',
                        'header_level': 0,
                        'start_line': 0,
                        'end_line': first_header_start - 1,
                        'is_orphaned_content': True
                    }
                })
        
        # Check for gaps between sections
        for i in range(len(all_headers) - 1):
            current_end = all_headers[i]['content_end']
            next_start = all_headers[i + 1]['line_number']
            
            if next_start > current_end + 1:
                gap_lines = lines[current_end + 1:next_start]
                gap_text = '\n'.join(gap_lines).strip()
                
                if gap_text and len(gap_text) > 20:
                    orphaned_chunks.append({
                        'text': gap_text,
                        'metadata': {
                            'section_header': f'Gap Content {i+1}',
                            'section_type': 'gap_content',
                            'chunk_type': 'gap_content',
                            'header_type': 'orphaned',
                            'header_level': 0,
                            'start_line': current_end + 1,
                            'end_line': next_start - 1,
                            'is_orphaned_content': True
                        }
                    })
        
        if orphaned_chunks:
            logger.info(f"Captured {len(orphaned_chunks)} orphaned content sections")
        
        return orphaned_chunks
    
    async def _fallback_comprehensive_chunking(self, lines: List[str], all_headers: List[Dict], doc_metadata: DocumentMetadata) -> List[Dict]:
        """Fallback method to ensure comprehensive content coverage"""
        
        fallback_chunks = []
        
        # Group headers by level to create meaningful sections
        headers_by_level = {}
        for header in all_headers:
            level = header['level']
            if level not in headers_by_level:
                headers_by_level[level] = []
            headers_by_level[level].append(header)
        
        # Process level 1 headers (main sections) with their subsections
        main_headers = headers_by_level.get(1, [])
        
        for i, main_header in enumerate(main_headers):
            # Find all subsections belonging to this main section
            main_start = main_header['line_number']
            if i + 1 < len(main_headers):
                main_end = main_headers[i + 1]['line_number'] - 1
            else:
                main_end = len(lines) - 1
            
            # Extract content for this main section
            section_lines = lines[main_start:main_end + 1]
            section_text = '\n'.join(section_lines).strip()
            
            if section_text and len(section_text) > 50:
                # Count tokens and decide on chunking strategy
                token_count = self._count_tokens(section_text)
                
                if token_count > self.max_tokens_per_chunk:
                    # Split large sections by subsections or paragraphs
                    sub_chunks = self._smart_subdivide_chunk(section_text, {
                        'section_header': main_header['text'],
                        'section_type': main_header.get('section_type', 'general'),
                        'chunk_type': main_header.get('section_type', 'general'),
                        'header_type': 'fallback_main',
                        'header_level': 1,
                        'is_fallback_chunk': True
                    })
                    fallback_chunks.extend(sub_chunks)
                else:
                    # Keep as single chunk
                    fallback_chunks.append({
                        'text': section_text,
                        'metadata': {
                            'section_header': main_header['text'],
                            'section_type': main_header.get('section_type', 'general'),
                            'chunk_type': main_header.get('section_type', 'general'),
                            'header_type': 'fallback_main',
                            'header_level': 1,
                            'is_fallback_chunk': True,
                            'content_length': len(section_text)
                        }
                    })
        
        logger.info(f"Fallback chunking created {len(fallback_chunks)} additional chunks")
        return fallback_chunks
    
    async def _subdivide_section(self, section_text: str, section_type: str, section_title: str) -> List[Dict]:
        """Subdivide large sections into semantic units"""
        
        sub_chunks = []
        
        # Handle definitions specially
        if section_type == 'definitions':
            sub_chunks = await self._chunk_definitions(section_text, section_title)
        
        # Handle exclusions as separate items
        elif section_type == 'exclusions':
            sub_chunks = await self._chunk_exclusions(section_text, section_title)
        
        # Handle general sections by subsection patterns
        else:
            sub_chunks = await self._chunk_by_subsections(section_text, section_type, section_title)
        
        # Fallback to paragraph chunking if no subsections found
        if not sub_chunks:
            paragraphs = section_text.split('\n\n')
            for i, paragraph in enumerate(paragraphs):
                if paragraph.strip():
                    sub_chunks.append({
                        'text': paragraph.strip(),
                        'metadata': {
                            'section_header': section_title,
                            'section_type': section_type,
                            'chunk_type': section_type if section_type else 'text',
                            'subsection_index': i,
                            'is_paragraph': True
                        }
                    })
        
        return sub_chunks
    
    async def _chunk_definitions(self, text: str, section_title: str) -> List[Dict]:
        """Extract individual definitions as separate chunks"""
        
        chunks = []
        lines = text.split('\n')
        current_definition = []
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Check if this line starts a new definition
            is_definition_start = any(re.match(pattern, line_stripped, re.IGNORECASE) 
                                    for pattern in self.definition_patterns)
            
            if is_definition_start and current_definition:
                # Save previous definition
                def_text = '\n'.join(current_definition).strip()
                if def_text:
                    chunks.append({
                        'text': def_text,
                        'metadata': {
                            'section_header': section_title,
                            'section_type': 'definitions',
                            'chunk_type': 'definition',
                            'is_definition': True
                        }
                    })
                current_definition = []
            
            current_definition.append(line)
        
        # Handle last definition
        if current_definition:
            def_text = '\n'.join(current_definition).strip()
            if def_text:
                chunks.append({
                    'text': def_text,
                    'metadata': {
                        'section_header': section_title,
                        'section_type': 'definitions',
                        'chunk_type': 'definition',
                        'is_definition': True
                    }
                })
        
        return chunks
    
    async def _chunk_exclusions(self, text: str, section_title: str) -> List[Dict]:
        """Extract individual exclusions as separate chunks"""
        
        chunks = []
        
        # Split by common exclusion patterns
        exclusion_separators = [
            r'^[A-Z]\)\s+',  # A) B) C)
            r'^\([a-z]\)\s+',  # (a) (b) (c)
            r'^\d+\.\s+',  # 1. 2. 3.
            r'^\([ivx]+\)\s+',  # (i) (ii) (iii)
        ]
        
        lines = text.split('\n')
        current_exclusion = []
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Check if this line starts a new exclusion
            is_exclusion_start = any(re.match(pattern, line_stripped) 
                                   for pattern in exclusion_separators)
            
            if is_exclusion_start and current_exclusion:
                # Save previous exclusion
                excl_text = '\n'.join(current_exclusion).strip()
                if excl_text:
                    chunks.append({
                        'text': excl_text,
                        'metadata': {
                            'section_header': section_title,
                            'section_type': 'exclusions',
                            'chunk_type': 'exclusion',
                            'is_exclusion': True
                        }
                    })
                current_exclusion = []
            
            current_exclusion.append(line)
        
        # Handle last exclusion
        if current_exclusion:
            excl_text = '\n'.join(current_exclusion).strip()
            if excl_text:
                chunks.append({
                    'text': excl_text,
                    'metadata': {
                        'section_header': section_title,
                        'section_type': 'exclusions',
                        'chunk_type': 'exclusion',
                        'is_exclusion': True
                    }
                })
        
        return chunks
    
    async def _chunk_by_subsections(self, text: str, section_type: str, section_title: str) -> List[Dict]:
        """Chunk by subsection patterns"""
        
        chunks = []
        lines = text.split('\n')
        current_subsection = []
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Check if this line starts a new subsection
            is_subsection_start = any(re.match(pattern, line_stripped) 
                                    for pattern in self.subsection_patterns)
            
            if is_subsection_start and current_subsection:
                # Save previous subsection
                subsection_text = '\n'.join(current_subsection).strip()
                if subsection_text:
                    chunks.append({
                        'text': subsection_text,
                        'metadata': {
                            'section_header': section_title,
                            'section_type': section_type,
                            'chunk_type': section_type if section_type else 'text',
                            'is_subsection': True
                        }
                    })
                current_subsection = []
            
            current_subsection.append(line)
        
        # Handle last subsection
        if current_subsection:
            subsection_text = '\n'.join(current_subsection).strip()
            if subsection_text:
                chunks.append({
                    'text': subsection_text,
                    'metadata': {
                        'section_header': section_type,
                        'section_type': section_type,
                        'chunk_type': section_type if section_type else 'text',
                        'is_subsection': True
                    }
                })
        
        return chunks
    
    async def _chunk_by_paragraphs(self, text: str, doc_metadata: DocumentMetadata) -> List[Dict]:
        """Fallback chunking by paragraphs when no clear structure is found"""
        
        # Create base metadata for general content
        base_metadata = {
            'section_header': 'General Content',
            'section_type': 'general',
            'chunk_type': 'text',
            'is_paragraph': True
        }
        
        # Check if entire text needs smart sub-chunking
        if self._should_subdivide_chunk(text):
            # Use smart sub-chunking with metadata inheritance
            return self._smart_subdivide_chunk(text, base_metadata)
        
        # Otherwise, chunk by paragraphs
        chunks = []
        paragraphs = text.split('\n\n')
        
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if paragraph and len(paragraph) > 50:  # Skip very short paragraphs
                chunks.append({
                    'text': paragraph,
                    'metadata': {
                        **base_metadata,
                        'paragraph_index': i
                    }
                })
        
        return chunks
    
    async def _process_tables_as_chunks(self, tables: List[Dict], doc_metadata: DocumentMetadata) -> List[Dict]:
        """Process extracted tables as separate chunks"""
        
        table_chunks = []
        
        for i, table in enumerate(tables):
            # Create chunk with both original and markdown versions
            chunk_text = f"TABLE:\n{table['markdown']}\n\nORIGINAL FORMAT:\n{table['original_text']}"
            
            table_chunks.append({
                'text': chunk_text,
                'metadata': {
                    'section_header': 'Tables',
                    'section_type': 'table',
                    'chunk_type': 'table_markdown',
                    'table_index': i,
                    'row_count': table['row_count'],
                    'is_table': True,
                    'markdown_format': table['markdown']
                }
            })
        
        return table_chunks
    
    async def _enrich_chunks_with_metadata(self, chunks: List[Dict], doc_metadata: DocumentMetadata) -> List[Dict]:
        """Enrich all chunks with comprehensive metadata"""
        
        enriched_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Create enriched metadata
            chunk_metadata = chunk.get('metadata', {})
            
            # Add document-level metadata
            chunk_metadata.update({
                'file_name': doc_metadata.file_name,
                'policy_uin': doc_metadata.policy_uin,
                'company_name': doc_metadata.company_name,
                'policy_type': doc_metadata.policy_type,
                'chunk_index': i,
                'confidence_score': doc_metadata.confidence_score,
                
                # Text statistics
                'char_count': len(chunk['text']),
                'word_count': len(chunk['text'].split()),
                'line_count': len(chunk['text'].split('\n')),
                
                # Processing metadata
                'processing_method': 'intelligent_layout_aware',
                'semantic_completeness': True
            })
            
            enriched_chunks.append({
                'text': chunk['text'],
                'metadata': chunk_metadata
            })
        
        return enriched_chunks