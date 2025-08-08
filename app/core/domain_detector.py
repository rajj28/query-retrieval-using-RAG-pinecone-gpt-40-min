import re
import logging
from typing import Dict, Any, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class DomainType(Enum):
    """Supported domain types"""
    INSURANCE = "insurance"
    LEGAL = "legal"
    HR = "hr"
    COMPLIANCE = "compliance"
    TECHNICAL = "technical"
    VEHICLE = "vehicle"
    GENERAL = "general"

class DomainDetector:
    """Intelligent domain detection and processing adaptation"""
    
    def __init__(self):
        # Domain-specific keywords and patterns
        self.domain_patterns = {
            DomainType.INSURANCE: {
                'keywords': [
                    'policy', 'premium', 'coverage', 'claim', 'insured', 'insurer',
                    'deductible', 'exclusion', 'endorsement', 'policyholder',
                    'sum insured', 'grace period', 'waiting period', 'no claim discount',
                    'health insurance', 'life insurance', 'motor insurance', 'property insurance'
                ],
                'patterns': [
                    r'policy\s+number', r'premium\s+payment', r'coverage\s+details',
                    r'claim\s+process', r'insurance\s+company', r'policy\s+terms'
                ]
            },
            DomainType.LEGAL: {
                'keywords': [
                    'contract', 'agreement', 'clause', 'party', 'obligation', 'liability',
                    'jurisdiction', 'governing law', 'dispute resolution', 'termination',
                    'breach', 'damages', 'indemnification', 'force majeure'
                ],
                'patterns': [
                    r'this\s+agreement', r'governing\s+law', r'dispute\s+resolution',
                    r'party\s+of\s+the', r'breach\s+of', r'termination\s+clause'
                ]
            },
            DomainType.HR: {
                'keywords': [
                    'employee', 'employer', 'benefits', 'leave', 'salary', 'performance',
                    'termination', 'probation', 'notice period', 'holidays', 'overtime',
                    'health benefits', 'retirement', 'pension', 'bonus'
                ],
                'patterns': [
                    r'employee\s+handbook', r'leave\s+policy', r'performance\s+review',
                    r'termination\s+notice', r'benefits\s+package', r'work\s+schedule'
                ]
            },
            DomainType.COMPLIANCE: {
                'keywords': [
                    'regulation', 'compliance', 'audit', 'risk', 'governance', 'policy',
                    'procedure', 'standard', 'requirement', 'certification', 'accreditation',
                    'regulatory', 'statutory', 'mandatory', 'violation'
                ],
                'patterns': [
                    r'compliance\s+requirement', r'regulatory\s+standard', r'audit\s+procedure',
                    r'risk\s+assessment', r'governance\s+framework', r'statutory\s+requirement'
                ]
            },
            DomainType.TECHNICAL: {
                'keywords': [
                    'code', 'programming', 'software', 'algorithm', 'function', 'variable',
                    'database', 'api', 'framework', 'library', 'syntax', 'compiler',
                    'debug', 'deploy', 'version control', 'git'
                ],
                'patterns': [
                    r'function\s+\w+\s*\(', r'const\s+\w+', r'import\s+\w+',
                    r'class\s+\w+', r'def\s+\w+', r'public\s+class'
                ]
            },
            DomainType.VEHICLE: {
                'keywords': [
                    'vehicle', 'car', 'engine', 'transmission', 'brake', 'tire', 'oil',
                    'maintenance', 'service', 'warranty', 'specification', 'manual',
                    'spark plug', 'battery', 'fuel', 'mileage'
                ],
                'patterns': [
                    r'vehicle\s+specification', r'maintenance\s+schedule', r'service\s+manual',
                    r'engine\s+oil', r'tire\s+pressure', r'brake\s+system'
                ]
            }
        }
    
    def detect_domain(self, text: str, document_metadata: Optional[Dict[str, Any]] = None) -> DomainType:
        """Detect domain from text content and metadata"""
        try:
            # Normalize text
            text_lower = text.lower()
            
            # Check metadata first
            if document_metadata:
                detected = self._detect_from_metadata(document_metadata)
                if detected != DomainType.GENERAL:
                    return detected
            
            # Score each domain
            domain_scores = {}
            for domain, patterns in self.domain_patterns.items():
                score = self._calculate_domain_score(text_lower, patterns)
                domain_scores[domain] = score
            
            # Find highest scoring domain
            best_domain = max(domain_scores.items(), key=lambda x: x[1])
            
            # Only return specific domain if score is significant
            if best_domain[1] > 0.1:  # Threshold for domain detection
                logger.info(f"Detected domain: {best_domain[0].value} (score: {best_domain[1]:.3f})")
                return best_domain[0]
            else:
                logger.info(f"No specific domain detected, using general (best score: {best_domain[1]:.3f})")
                return DomainType.GENERAL
                
        except Exception as e:
            logger.warning(f"Domain detection failed: {e}, using general")
            return DomainType.GENERAL
    
    def _detect_from_metadata(self, metadata: Dict[str, Any]) -> DomainType:
        """Detect domain from document metadata"""
        try:
            # Check file type
            file_type = metadata.get('file_type', '').lower()
            if 'policy' in file_type or 'insurance' in file_type:
                return DomainType.INSURANCE
            if 'contract' in file_type or 'agreement' in file_type:
                return DomainType.LEGAL
            if 'handbook' in file_type or 'employee' in file_type:
                return DomainType.HR
            if 'compliance' in file_type or 'regulation' in file_type:
                return DomainType.COMPLIANCE
            if 'manual' in file_type or 'specification' in file_type:
                return DomainType.VEHICLE
            
            # Check document title
            title = metadata.get('title', '').lower()
            if any(word in title for word in ['policy', 'insurance', 'coverage']):
                return DomainType.INSURANCE
            if any(word in title for word in ['contract', 'agreement', 'legal']):
                return DomainType.LEGAL
            if any(word in title for word in ['employee', 'handbook', 'hr']):
                return DomainType.HR
            if any(word in title for word in ['compliance', 'regulation', 'audit']):
                return DomainType.COMPLIANCE
            if any(word in title for word in ['manual', 'specification', 'vehicle']):
                return DomainType.VEHICLE
            
            return DomainType.GENERAL
            
        except Exception as e:
            logger.warning(f"Metadata domain detection failed: {e}")
            return DomainType.GENERAL
    
    def _calculate_domain_score(self, text: str, patterns: Dict[str, Any]) -> float:
        """Calculate domain relevance score"""
        score = 0.0
        
        # Keyword matching
        keywords = patterns.get('keywords', [])
        for keyword in keywords:
            if keyword in text:
                score += 0.05  # Weight for keyword matches
        
        # Pattern matching
        regex_patterns = patterns.get('patterns', [])
        for pattern in regex_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            score += matches * 0.1  # Higher weight for pattern matches
        
        return score
    
    def get_domain_config(self, domain: DomainType) -> Dict[str, Any]:
        """Get domain-specific configuration"""
        configs = {
            DomainType.INSURANCE: {
                'chunk_size': 800,
                'chunk_overlap': 300,
                'retrieval_top_k': 50,
                'rerank_top_k': 10,
                'query_expansion': True,
                'priority': 'high'
            },
            DomainType.LEGAL: {
                'chunk_size': 1000,
                'chunk_overlap': 400,
                'retrieval_top_k': 60,
                'rerank_top_k': 12,
                'query_expansion': True,
                'priority': 'high'
            },
            DomainType.HR: {
                'chunk_size': 600,
                'chunk_overlap': 200,
                'retrieval_top_k': 40,
                'rerank_top_k': 8,
                'query_expansion': True,
                'priority': 'high'
            },
            DomainType.COMPLIANCE: {
                'chunk_size': 900,
                'chunk_overlap': 350,
                'retrieval_top_k': 55,
                'rerank_top_k': 11,
                'query_expansion': True,
                'priority': 'high'
            },
            DomainType.TECHNICAL: {
                'chunk_size': 500,
                'chunk_overlap': 150,
                'retrieval_top_k': 30,
                'rerank_top_k': 6,
                'query_expansion': False,
                'priority': 'medium'
            },
            DomainType.VEHICLE: {
                'chunk_size': 700,
                'chunk_overlap': 250,
                'retrieval_top_k': 35,
                'rerank_top_k': 7,
                'query_expansion': False,
                'priority': 'medium'
            },
            DomainType.GENERAL: {
                'chunk_size': 600,
                'chunk_overlap': 200,
                'retrieval_top_k': 30,
                'rerank_top_k': 6,
                'query_expansion': False,
                'priority': 'low'
            }
        }
        
        return configs.get(domain, configs[DomainType.GENERAL])
    
    def is_primary_domain(self, domain: DomainType) -> bool:
        """Check if domain is primary (insurance, legal, HR, compliance)"""
        primary_domains = [
            DomainType.INSURANCE,
            DomainType.LEGAL, 
            DomainType.HR,
            DomainType.COMPLIANCE
        ]
        return domain in primary_domains
