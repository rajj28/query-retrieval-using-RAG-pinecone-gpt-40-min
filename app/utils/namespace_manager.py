# app/utils/namespace_manager.py
import hashlib
import re
from typing import Optional, Union
from urllib.parse import urlparse
from app.config.settings import settings, get_namespace_prefix
import logging

logger = logging.getLogger(__name__)

class NamespaceManager:
    """Manages deterministic namespace generation for Pinecone vectors"""
    
    @staticmethod
    def generate_content_hash(content: Union[str, bytes]) -> str:
        """Generate SHA-256 hash of content"""
        if isinstance(content, str):
            content = content.encode('utf-8')
        return hashlib.sha256(content).hexdigest()[:16]  # First 16 chars for brevity
    
    @staticmethod
    def generate_url_hash(url: str) -> str:
        """Generate hash from URL for consistent namespacing"""
        # Normalize URL by removing query parameters that might change
        parsed = urlparse(url)
        normalized_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        return NamespaceManager.generate_content_hash(normalized_url)
    
    @staticmethod
    def sanitize_namespace(namespace: str) -> str:
        """Sanitize namespace to meet Pinecone requirements"""
        # Pinecone namespaces: alphanumeric + hyphens, max 40 chars
        sanitized = re.sub(r'[^a-zA-Z0-9\-]', '-', namespace)
        sanitized = re.sub(r'-+', '-', sanitized)  # Remove multiple hyphens
        sanitized = sanitized.strip('-')  # Remove leading/trailing hyphens
        return sanitized[:40]  # Limit to 40 characters
    
    @classmethod
    def create_namespace(
        self,
        source: str,
        content: Optional[str] = None,
        document_type: str = "document"
    ) -> str:
        """
        Create deterministic namespace based on source and content
        
        Args:
            source: Document URL or file path
            content: Document content (optional, will use source if not provided)
            document_type: Type of document (policy, contract, etc.)
            
        Returns:
            Sanitized namespace string
        """
        prefix = get_namespace_prefix()
        
        # Generate hash based on content or source
        if content:
            content_hash = self.generate_content_hash(content)
        else:
            content_hash = self.generate_url_hash(source) if source.startswith('http') else self.generate_content_hash(source)
        
        # Create namespace: prefix_doctype_hash
        namespace = f"{prefix}_{document_type}_{content_hash}"
        
        return self.sanitize_namespace(namespace)
    
    @classmethod
    def create_test_namespace(self, base_namespace: str) -> str:
        """Create test version of namespace"""
        if not base_namespace.startswith('test_'):
            return f"test_{base_namespace}"
        return base_namespace
    
    @classmethod
    def is_test_namespace(self, namespace: str) -> bool:
        """Check if namespace is for testing"""
        return namespace.startswith('test_')
    
    @classmethod
    def get_cleanup_pattern(self, prefix: Optional[str] = None) -> str:
        """Get pattern for namespace cleanup"""
        if prefix:
            return f"{prefix}_*"
        return f"{get_namespace_prefix()}_*"

class DocumentTracker:
    """Track processed documents to avoid reprocessing"""
    
    def __init__(self):
        self._processed_docs = set()
    
    def is_processed(self, namespace: str) -> bool:
        """Check if document was already processed"""
        return namespace in self._processed_docs
    
    def mark_processed(self, namespace: str):
        """Mark document as processed"""
        self._processed_docs.add(namespace)
        logger.info(f"Marked namespace as processed: {namespace}")
    
    def clear_processed(self, namespace: str):
        """Clear processed status"""
        self._processed_docs.discard(namespace)
        logger.info(f"Cleared processed status: {namespace}")
    
    def get_processed_count(self) -> int:
        """Get count of processed documents"""
        return len(self._processed_docs)

# Global document tracker
document_tracker = DocumentTracker()