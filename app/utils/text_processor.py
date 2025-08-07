# app/utils/text_processor.py
import re
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class TextProcessor:
    """Process and chunk text for optimal embedding and retrieval"""
    
    def __init__(self):
        # Patterns for cleaning text
        self.cleanup_patterns = [
            (r'\s+', ' '),  # Multiple whitespace to single space
            (r'\n\s*\n\s*\n+', '\n\n'),  # Multiple newlines to double newline
            (r'[^\S\n]+', ' '),  # Non-newline whitespace to single space
            (r'^\s+|\s+$', ''),  # Leading/trailing whitespace
        ]
        
        # Patterns for identifying section breaks
        self.section_patterns = [
            r'^(?:SECTION|Section|CHAPTER|Chapter|ARTICLE|Article)\s+\d+',
            r'^\d+\.\s+[A-Z][^.]*$',  # Numbered sections like "1. INTRODUCTION"
            r'^[A-Z\s]{3,}$',  # All caps headings
            r'^\([a-z]\)',  # Lettered subsections like "(a)"
            r'^\d+\.\d+',  # Numbered subsections like "1.1"
        ]
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text or not text.strip():
            return ""
        
        # Apply cleanup patterns
        cleaned = text
        for pattern, replacement in self.cleanup_patterns:
            cleaned = re.sub(pattern, replacement, cleaned)
        
        # Remove excessive punctuation
        cleaned = re.sub(r'[.]{3,}', '...', cleaned)
        cleaned = re.sub(r'[-]{3,}', '---', cleaned)
        
        # Normalize quotes
        cleaned = re.sub(r'["""]', '"', cleaned)
        cleaned = re.sub(r'[\'\u2018\u2019]', "'", cleaned)
        
        return cleaned.strip()
    
    def identify_sections(self, text: str) -> List[Dict[str, Any]]:
        """Identify document sections and their boundaries"""
        lines = text.split('\n')
        sections = []
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue
                
            # Check if line matches section patterns
            for pattern in self.section_patterns:
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    sections.append({
                        'line_number': i,
                        'title': line_stripped,
                        'pattern': pattern,
                        'level': self._determine_section_level(line_stripped)
                    })
                    break
        
        return sections
    
    def _determine_section_level(self, section_title: str) -> int:
        """Determine hierarchical level of section"""
        title = section_title.strip()
        
        # Level 1: Main sections (SECTION, CHAPTER, ARTICLE)
        if re.match(r'^(?:SECTION|Section|CHAPTER|Chapter|ARTICLE|Article)', title):
            return 1
        
        # Level 2: Major numbered sections (1. TITLE)
        if re.match(r'^\d+\.\s+[A-Z]', title):
            return 2
        
        # Level 3: Subsections (1.1, 1.2, etc.)
        if re.match(r'^\d+\.\d+', title):
            return 3
        
        # Level 4: Sub-subsections (1.1.1, etc.)
        if re.match(r'^\d+\.\d+\.\d+', title):
            return 4
        
        # Level 5: Lettered items ((a), (b), etc.)
        if re.match(r'^\([a-z]\)', title):
            return 5
        
        # Default level
        return 2
    
    def create_chunks(
        self, 
        text: str, 
        chunk_size: int = 1000, 
        overlap: int = 200,
        preserve_sections: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Create overlapping text chunks optimized for semantic search
        
        Args:
            text: Input text to chunk
            chunk_size: Target size for each chunk
            overlap: Overlap between adjacent chunks
            preserve_sections: Try to keep sections intact
            
        Returns:
            List of chunk dictionaries with metadata
        """
        if not text or len(text.strip()) < 10:
            return []
        
        chunks = []
        
        if preserve_sections:
            sections = self.identify_sections(text)
            if sections:
                chunks = self._create_section_aware_chunks(text, sections, chunk_size, overlap)
            else:
                chunks = self._create_sliding_window_chunks(text, chunk_size, overlap)
        else:
            chunks = self._create_sliding_window_chunks(text, chunk_size, overlap)
        
        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.update({
                'chunk_id': i,
                'total_chunks': len(chunks),
                'char_count': len(chunk['text']),
                'word_count': len(chunk['text'].split()),
                'sentences': self._count_sentences(chunk['text'])
            })
        
        logger.info(f"Created {len(chunks)} chunks from {len(text)} characters")
        return chunks
    
    def _create_section_aware_chunks(
        self, 
        text: str, 
        sections: List[Dict[str, Any]], 
        chunk_size: int, 
        overlap: int
    ) -> List[Dict[str, Any]]:
        """Create chunks that respect section boundaries"""
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        current_section = None
        
        for i, line in enumerate(lines):
            line_size = len(line) + 1  # +1 for newline
            
            # Check if this line starts a new section
            section_info = None
            for section in sections:
                if section['line_number'] == i:
                    section_info = section
                    break
            
            # If we hit a new major section and current chunk is substantial, finalize it
            if (section_info and 
                section_info['level'] <= 2 and 
                current_size > chunk_size * 0.5 and 
                current_chunk):
                
                chunk_text = '\n'.join(current_chunk)
                chunks.append({
                    'text': chunk_text.strip(),
                    'start_line': chunks[-1]['end_line'] + 1 if chunks else 0,
                    'end_line': i - 1,
                    'section': current_section,
                    'chunk_type': 'section_boundary'
                })
                
                # Start new chunk with overlap
                if overlap > 0:
                    overlap_lines = self._get_overlap_lines(current_chunk, overlap)
                    current_chunk = overlap_lines + [line]
                    current_size = sum(len(l) + 1 for l in overlap_lines) + line_size
                else:
                    current_chunk = [line]
                    current_size = line_size
                    
                current_section = section_info
            
            # Add line to current chunk
            else:
                current_chunk.append(line)
                current_size += line_size
                
                if section_info:
                    current_section = section_info
                
                # If chunk is too large, split it
                if current_size >= chunk_size and len(current_chunk) > 1:
                    # Find good split point (preferably at sentence boundary)
                    split_point = self._find_split_point(current_chunk, chunk_size * 0.8)
                    
                    chunk_lines = current_chunk[:split_point]
                    chunk_text = '\n'.join(chunk_lines)
                    
                    chunks.append({
                        'text': chunk_text.strip(),
                        'start_line': chunks[-1]['end_line'] + 1 if chunks else 0,
                        'end_line': len(chunks) * chunk_size // len(text) * len(lines),  # Approximate
                        'section': current_section,
                        'chunk_type': 'size_split'
                    })
                    
                    # Continue with remaining lines plus overlap
                    remaining_lines = current_chunk[split_point:]
                    if overlap > 0:
                        overlap_lines = self._get_overlap_lines(chunk_lines, overlap)
                        current_chunk = overlap_lines + remaining_lines
                    else:
                        current_chunk = remaining_lines
                    
                    current_size = sum(len(l) + 1 for l in current_chunk)
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            if chunk_text.strip():
                chunks.append({
                    'text': chunk_text.strip(),
                    'start_line': chunks[-1]['end_line'] + 1 if chunks else 0,
                    'end_line': len(lines) - 1,
                    'section': current_section,
                    'chunk_type': 'final'
                })
        
        return chunks
    
    def _create_sliding_window_chunks(
        self, 
        text: str, 
        chunk_size: int, 
        overlap: int
    ) -> List[Dict[str, Any]]:
        """Create chunks using sliding window approach"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            
            # If this is not the last chunk, try to end at a sentence boundary
            if end < text_length:
                # Look for sentence endings within the last 20% of the chunk
                search_start = max(start + int(chunk_size * 0.8), start + 1)
                sentence_end = self._find_sentence_boundary(text, search_start, end)
                if sentence_end > start:
                    end = sentence_end
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    'text': chunk_text,
                    'start_char': start,
                    'end_char': end,
                    'chunk_type': 'sliding_window'
                })
            
            # Move start position (with overlap)
            start = max(start + 1, end - overlap)
            
            # Prevent infinite loop
            if start >= text_length:
                break
        
        return chunks
    
    def _find_split_point(self, lines: List[str], target_size: int) -> int:
        """Find optimal split point in lines list"""
        current_size = 0
        best_split = len(lines) // 2  # Default to middle
        
        for i, line in enumerate(lines):
            current_size += len(line) + 1
            
            # Look for good split points
            if current_size >= target_size:
                line_stripped = line.strip()
                
                # Prefer to split after sentences
                if line_stripped.endswith(('.', '!', '?', ';')):
                    return i + 1
                
                # Prefer to split after paragraphs
                if not line_stripped:
                    return i + 1
                
                # Otherwise split here
                return max(i, 1)
        
        return best_split
    
    def _find_sentence_boundary(self, text: str, start: int, end: int) -> int:
        """Find sentence boundary within range"""
        search_text = text[start:end]
        
        # Look for sentence endings
        for pattern in [r'\.(?:\s|$)', r'[!?](?:\s|$)', r';(?:\s|$)']:
            matches = list(re.finditer(pattern, search_text))
            if matches:
                # Return position of last match
                last_match = matches[-1]
                return start + last_match.end()
        
        return end
    
    def _get_overlap_lines(self, lines: List[str], overlap: int) -> List[str]:
        """Get lines for overlap based on character count"""
        if not lines or overlap <= 0:
            return []
        
        overlap_lines = []
        current_size = 0
        
        # Start from end and work backwards
        for line in reversed(lines):
            line_size = len(line) + 1
            if current_size + line_size <= overlap:
                overlap_lines.insert(0, line)
                current_size += line_size
            else:
                break
        
        return overlap_lines
    
    def _count_sentences(self, text: str) -> int:
        """Count sentences in text"""
        if not text:
            return 0
        
        # Simple sentence counting
        sentences = re.split(r'[.!?]+(?:\s|$)', text)
        return len([s for s in sentences if s.strip()])
    
    def extract_key_phrases(self, text: str, max_phrases: int = 10) -> List[str]:
        """Extract key phrases from text for metadata"""
        if not text:
            return []
        
        # Simple key phrase extraction (you could enhance this with NLP libraries)
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Count phrase frequency
        phrase_counts = {}
        for phrase in words:
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
        
        # Sort by frequency and return top phrases
        sorted_phrases = sorted(phrase_counts.items(), key=lambda x: x[1], reverse=True)
        return [phrase for phrase, count in sorted_phrases[:max_phrases]]
    
    def get_text_statistics(self, text: str) -> Dict[str, Any]:
        """Get comprehensive text statistics"""
        if not text:
            return {}
        
        words = text.split()
        sentences = self._count_sentences(text)
        paragraphs = len([p for p in text.split('\n\n') if p.strip()])
        
        return {
            'character_count': len(text),
            'word_count': len(words),
            'sentence_count': sentences,
            'paragraph_count': paragraphs,
            'avg_words_per_sentence': len(words) / sentences if sentences > 0 else 0,
            'avg_chars_per_word': len(text) / len(words) if words else 0,
            'reading_time_minutes': len(words) / 200,  # Assuming 200 WPM
        }