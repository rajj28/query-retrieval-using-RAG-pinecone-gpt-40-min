"""
Enhanced Domain Configuration for Multi-Domain LLM Query Retrieval System
Supports: Insurance (ICICI, HDFC, National Parivar, Cholamandalam, Edelweiss, Arogya Sanjeevani)
Plus: Legal, HR, Compliance domains
"""

from enum import Enum
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class DomainType(Enum):
    """Supported domain types"""
    INSURANCE = "insurance"
    LEGAL = "legal"
    HR = "hr"
    COMPLIANCE = "compliance"
    FINANCE = "finance"
    HEALTHCARE = "healthcare"

class DomainConfig:
    """Centralized configuration for multi-domain adaptation"""
    
    def __init__(self, domain_type: DomainType):
        self.domain_type = domain_type
        self.config = self._get_domain_config()
    
    def _get_domain_config(self) -> Dict[str, Any]:
        """Get configuration for specific domain"""
        if self.domain_type == DomainType.INSURANCE:
            return self._get_insurance_config()
        elif self.domain_type == DomainType.LEGAL:
            return self._get_legal_config()
        elif self.domain_type == DomainType.HR:
            return self._get_hr_config()
        elif self.domain_type == DomainType.COMPLIANCE:
            return self._get_compliance_config()
        elif self.domain_type == DomainType.FINANCE:
            return self._get_finance_config()
        elif self.domain_type == DomainType.HEALTHCARE:
            return self._get_healthcare_config()
        else:
            return self._get_default_config()
    
    def _get_insurance_config(self) -> Dict[str, Any]:
        """Enhanced insurance domain configuration covering all major insurers"""
        return {
            "section_patterns": {
                # ICICI Lombard specific patterns
                "icici": [
                    r"^(\d+\.\s*)?(Definitions?|DEFINITIONS?)\s*$",
                    r"^(\d+\.\s*)?(Waiting Period|WAITING PERIOD)\s*$",
                    r"^(\d+\.\s*)?(Cumulative Bonus|CUMULATIVE BONUS)\s*$",
                    r"^(\d+\.\s*)?(Reset Benefit|RESET BENEFIT)\s*$",
                    r"^(\d+\.\s*)?(Zone|ZONE)\s*[A-C]\s*$",
                    r"^(\d+\.\s*)?(Exclusions?|EXCLUSIONS?)\s*$",
                    r"^(\d+\.\s*)?(Claims?|CLAIMS?)\s*$",
                ],
                # HDFC Ergo specific patterns
                "hdfc": [
                    r"^(\d+\.\s*)?(Hospital|HOSPITAL)\s*$",
                    r"^(\d+\.\s*)?(Accident|ACCIDENT)\s*$",
                    r"^(\d+\.\s*)?(Moratorium Period|MORATORIUM PERIOD)\s*$",
                    r"^(\d+\.\s*)?(Maternity|MATERNITY)\s*$",
                    r"^(\d+\.\s*)?(Free Look Period|FREE LOOK PERIOD)\s*$",
                    r"^(\d+\.\s*)?(Preventive Care|PREVENTIVE CARE)\s*$",
                ],
                # National Parivar Mediclaim specific patterns
                "national_parivar": [
                    r"^(\d+\.\s*)?(Family Floater|FAMILY FLOATER)\s*$",
                    r"^(\d+\.\s*)?(Pre-existing Disease|PRE-EXISTING DISEASE)\s*$",
                    r"^(\d+\.\s*)?(Day Care Treatment|DAY CARE TREATMENT)\s*$",
                    r"^(\d+\.\s*)?(Organ Donor|ORGAN DONOR)\s*$",
                    r"^(\d+\.\s*)?(Critical Illness|CRITICAL ILLNESS)\s*$",
                ],
                # Cholamandalam specific patterns
                "cholamandalam": [
                    r"^(\d+\.\s*)?(Travel Insurance|TRAVEL INSURANCE)\s*$",
                    r"^(\d+\.\s*)?(Medical Expenses|MEDICAL EXPENSES)\s*$",
                    r"^(\d+\.\s*)?(Trip Cancellation|TRIP CANCELLATION)\s*$",
                    r"^(\d+\.\s*)?(Baggage Loss|BAGGAGE LOSS)\s*$",
                    r"^(\d+\.\s*)?(Personal Accident|PERSONAL ACCIDENT)\s*$",
                ],
                # Edelweiss Silver specific patterns
                "edelweiss": [
                    r"^(\d+\.\s*)?(Silver Plan|SILVER PLAN)\s*$",
                    r"^(\d+\.\s*)?(Senior Citizen|SENIOR CITIZEN)\s*$",
                    r"^(\d+\.\s*)?(Age-related Benefits|AGE-RELATED BENEFITS)\s*$",
                    r"^(\d+\.\s*)?(Pre-existing Conditions|PRE-EXISTING CONDITIONS)\s*$",
                    r"^(\d+\.\s*)?(Wellness Benefits|WELLNESS BENEFITS)\s*$",
                ],
                # Arogya Sanjeevani specific patterns
                "arogya_sanjeevani": [
                    r"^(\d+\.\s*)?(Standard Health|STANDARD HEALTH)\s*$",
                    r"^(\d+\.\s*)?(Government Scheme|GOVERNMENT SCHEME)\s*$",
                    r"^(\d+\.\s*)?(Affordable Care|AFFORDABLE CARE)\s*$",
                    r"^(\d+\.\s*)?(Basic Coverage|BASIC COVERAGE)\s*$",
                    r"^(\d+\.\s*)?(Subsidy Benefits|SUBSIDY BENEFITS)\s*$",
                ],
                # Universal insurance patterns
                "universal": [
                    r"^(\d+\.\s*)?(Terms and Conditions|TERMS AND CONDITIONS)\s*$",
                    r"^(\d+\.\s*)?(Policy Schedule|POLICY SCHEDULE)\s*$",
                    r"^(\d+\.\s*)?(Premium|PREMIUM)\s*$",
                    r"^(\d+\.\s*)?(Sum Insured|SUM INSURED)\s*$",
                    r"^(\d+\.\s*)?(Renewal|RENEWAL)\s*$",
                    r"^(\d+\.\s*)?(Cancellation|CANCELLATION)\s*$",
                    r"^(\d+\.\s*)?(Grievance|GRIEVANCE)\s*$",
                ]
            },
            "entity_patterns": {
                "waiting_periods": [
                    r"(\d+)\s*(months?|years?)\s*waiting\s*period",
                    r"waiting\s*period\s*of\s*(\d+)\s*(months?|years?)",
                    r"(\d+)\s*(months?|years?)\s*exclusion"
                ],
                "percentages": [
                    r"(\d+)\s*%",
                    r"(\d+)\s*percent",
                    r"(\d+)\s*per\s*cent"
                ],
                "amounts": [
                    r"Rs\.?\s*(\d+(?:,\d+)*)",
                    r"₹\s*(\d+(?:,\d+)*)",
                    r"(\d+(?:,\d+)*)\s*rupees?"
                ],
                "dates": [
                    r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})",
                    r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})"
                ]
            },
            "key_terminology": {
                "icici_terms": [
                    "Golden Shield", "Reset Benefit", "Zone A", "Zone B", "Zone C",
                    "Cumulative Bonus", "Pre-existing Disease", "Day Care Treatment"
                ],
                "hdfc_terms": [
                    "Easy Health", "Moratorium Period", "Free Look Period",
                    "Maternity Expenses", "Preventive Care", "Organ Donor"
                ],
                "national_parivar_terms": [
                    "Family Floater", "Parivar Mediclaim", "Critical Illness",
                    "Pre-existing Disease", "Day Care Treatment"
                ],
                "cholamandalam_terms": [
                    "Travel Insurance", "Medical Expenses", "Trip Cancellation",
                    "Baggage Loss", "Personal Accident", "Overseas Medical"
                ],
                "edelweiss_terms": [
                    "Silver Plan", "Senior Citizen", "Age-related Benefits",
                    "Wellness Benefits", "Pre-existing Conditions"
                ],
                "arogya_sanjeevani_terms": [
                    "Standard Health", "Government Scheme", "Affordable Care",
                    "Basic Coverage", "Subsidy Benefits"
                ]
            },
            "query_expansion_patterns": {
                "waiting_period": {
                    "triggers": ["waiting period", "exclusion period", "pre-existing"],
                    "synonyms": ["24 months", "2 years", "twenty four months", "continuous coverage"],
                    "patterns": [r"\d+\s*months?", r"\d+\s*years?"]
                },
                "numerical_details": {
                    "triggers": ["percentage", "percent", "amount", "sum insured"],
                    "synonyms": ["10%", "15%", "100%", "ten percent", "fifteen percent", "hundred percent"],
                    "patterns": [r"\d+%", r"Rs\.\d+", r"₹\d+"]
                },
                "medical_conditions": {
                    "triggers": ["cataract", "hernia", "ent", "tonsils", "adenoids"],
                    "synonyms": ["deviated nasal septum", "csom", "medical condition", "surgical procedure"],
                    "ent_specific": ["deviated nasal septum", "chronic suppurative otitis media", "csom", "tonsillectomy", "adenoidectomy"],
                    "list_terms": ["list of specific", "surgical procedures", "illness and surgical", "medical procedures list"]
                },
                "co_payment": {
                    "triggers": ["co-payment", "copayment", "zone", "additional payment"],
                    "synonyms": ["zone based co-payment", "additional co-payment", "percentage deduction"],
                    "zone_terms": ["zone a", "zone b", "zone c", "15%", "fifteen percent"]
                }
            },
            "fast_patterns": {
                "definitions": [r"means?\s*[A-Z]", r"refers?\s*to", r"defined\s*as"],
                "exclusions": [r"not\s+covered", r"excluded", r"not\s+include", r"exclusion"],
                "inclusions": [r"covered", r"included", r"eligible", r"benefit"],
                "conditions": [r"provided\s+that", r"subject\s+to", r"if\s+and\s+only\s+if"],
                "amounts": [r"Rs\.?\s*\d+", r"₹\s*\d+", r"\d+\s*%"],
                "time_periods": [r"\d+\s*months?", r"\d+\s*years?", r"\d+\s*days?"]
            }
        }
    
    def _get_legal_config(self) -> Dict[str, Any]:
        """Legal domain configuration"""
        return {
            "section_patterns": {
                "contracts": [
                    r"^(\d+\.\s*)?(Terms and Conditions|TERMS AND CONDITIONS)\s*$",
                    r"^(\d+\.\s*)?(Definitions?|DEFINITIONS?)\s*$",
                    r"^(\d+\.\s*)?(Obligations?|OBLIGATIONS?)\s*$",
                    r"^(\d+\.\s*)?(Termination|TERMINATION)\s*$",
                    r"^(\d+\.\s*)?(Dispute Resolution|DISPUTE RESOLUTION)\s*$",
                    r"^(\d+\.\s*)?(Governing Law|GOVERNING LAW)\s*$",
                ],
                "compliance": [
                    r"^(\d+\.\s*)?(Regulatory Requirements|REGULATORY REQUIREMENTS)\s*$",
                    r"^(\d+\.\s*)?(Legal Obligations|LEGAL OBLIGATIONS?)\s*$",
                    r"^(\d+\.\s*)?(Compliance Framework|COMPLIANCE FRAMEWORK)\s*$",
                ]
            },
            "key_terminology": [
                "force majeure", "breach of contract", "liquidated damages",
                "indemnification", "confidentiality", "non-compete",
                "governing law", "jurisdiction", "arbitration", "mediation"
            ]
        }
    
    def _get_hr_config(self) -> Dict[str, Any]:
        """HR domain configuration"""
        return {
            "section_patterns": {
                "employment": [
                    r"^(\d+\.\s*)?(Employment Terms|EMPLOYMENT TERMS)\s*$",
                    r"^(\d+\.\s*)?(Compensation|COMPENSATION)\s*$",
                    r"^(\d+\.\s*)?(Benefits|BENEFITS)\s*$",
                    r"^(\d+\.\s*)?(Leave Policy|LEAVE POLICY)\s*$",
                    r"^(\d+\.\s*)?(Performance|PERFORMANCE)\s*$",
                    r"^(\d+\.\s*)?(Termination|TERMINATION)\s*$",
                ],
                "policies": [
                    r"^(\d+\.\s*)?(Code of Conduct|CODE OF CONDUCT)\s*$",
                    r"^(\d+\.\s*)?(Anti-harassment|ANTI-HARASSMENT)\s*$",
                    r"^(\d+\.\s*)?(Diversity|DIVERSITY)\s*$",
                ]
            },
            "key_terminology": [
                "at-will employment", "probation period", "performance review",
                "severance package", "non-disclosure agreement", "intellectual property",
                "workplace harassment", "equal opportunity", "reasonable accommodation"
            ]
        }
    
    def _get_compliance_config(self) -> Dict[str, Any]:
        """Compliance domain configuration"""
        return {
            "section_patterns": {
                "regulatory": [
                    r"^(\d+\.\s*)?(Regulatory Compliance|REGULATORY COMPLIANCE)\s*$",
                    r"^(\d+\.\s*)?(Risk Management|RISK MANAGEMENT)\s*$",
                    r"^(\d+\.\s*)?(Internal Controls|INTERNAL CONTROLS)\s*$",
                    r"^(\d+\.\s*)?(Audit Requirements|AUDIT REQUIREMENTS)\s*$",
                    r"^(\d+\.\s*)?(Reporting Obligations|REPORTING OBLIGATIONS)\s*$",
                ],
                "standards": [
                    r"^(\d+\.\s*)?(ISO Standards|ISO STANDARDS)\s*$",
                    r"^(\d+\.\s*)?(GDPR Compliance|GDPR COMPLIANCE)\s*$",
                    r"^(\d+\.\s*)?(SOX Compliance|SOX COMPLIANCE)\s*$",
                ]
            },
            "key_terminology": [
                "regulatory framework", "compliance officer", "risk assessment",
                "internal audit", "external audit", "regulatory reporting",
                "data protection", "privacy compliance", "financial compliance"
            ]
        }
    
    def _get_finance_config(self) -> Dict[str, Any]:
        """Finance domain configuration"""
        return {
            "section_patterns": {
                "financial": [
                    r"^(\d+\.\s*)?(Financial Terms|FINANCIAL TERMS)\s*$",
                    r"^(\d+\.\s*)?(Payment Terms|PAYMENT TERMS)\s*$",
                    r"^(\d+\.\s*)?(Interest Rates|INTEREST RATES)\s*$",
                    r"^(\d+\.\s*)?(Fees and Charges|FEES AND CHARGES)\s*$",
                ]
            },
            "key_terminology": [
                "interest rate", "principal amount", "maturity date",
                "payment schedule", "late fees", "prepayment penalty",
                "collateral", "default", "foreclosure"
            ]
        }
    
    def _get_healthcare_config(self) -> Dict[str, Any]:
        """Healthcare domain configuration"""
        return {
            "section_patterns": {
                "medical": [
                    r"^(\d+\.\s*)?(Medical Procedures|MEDICAL PROCEDURES)\s*$",
                    r"^(\d+\.\s*)?(Treatment Protocols|TREATMENT PROTOCOLS)\s*$",
                    r"^(\d+\.\s*)?(Patient Rights|PATIENT RIGHTS)\s*$",
                    r"^(\d+\.\s*)?(Informed Consent|INFORMED CONSENT)\s*$",
                ]
            },
            "key_terminology": [
                "informed consent", "medical malpractice", "patient confidentiality",
                "treatment protocol", "medical necessity", "prior authorization",
                "copayment", "deductible", "out-of-pocket maximum"
            ]
        }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration for unknown domains"""
        return {
            "section_patterns": {
                "general": [
                    r"^(\d+\.\s*)?([A-Z][A-Z\s]+)\s*$",
                    r"^(\d+\.\s*)?([A-Z][a-z\s]+)\s*$",
                ]
            },
            "key_terminology": [],
            "query_expansion_patterns": {}
        }
    
    def get_section_patterns(self) -> List[str]:
        """Get section patterns for the domain"""
        patterns = []
        for category in self.config.get("section_patterns", {}).values():
            if isinstance(category, list):
                patterns.extend(category)
        return patterns
    
    def get_entity_patterns(self) -> Dict[str, List[str]]:
        """Get entity patterns for the domain"""
        return self.config.get("entity_patterns", {})
    
    def get_key_terminology(self) -> List[str]:
        """Get key terminology for the domain"""
        terminology = []
        key_terms = self.config.get("key_terminology", {})
        if isinstance(key_terms, dict):
            for category in key_terms.values():
                if isinstance(category, list):
                    terminology.extend(category)
        elif isinstance(key_terms, list):
            terminology.extend(key_terms)
        return terminology
    
    def get_query_expansion_patterns(self) -> Dict[str, Any]:
        """Get query expansion patterns for the domain"""
        return self.config.get("query_expansion_patterns", {})
    
    def get_fast_patterns(self) -> Dict[str, List[str]]:
        """Get fast patterns for the domain"""
        return self.config.get("fast_patterns", {})

def detect_domain_from_text(text: str) -> DomainType:
    """Detect domain type from text content"""
    text_lower = text.lower()
    
    # Insurance keywords
    insurance_keywords = [
        "insurance", "policy", "premium", "claim", "coverage", "sum insured",
        "waiting period", "exclusion", "benefit", "deductible", "copayment"
    ]
    
    # Legal keywords
    legal_keywords = [
        "contract", "agreement", "terms and conditions", "governing law",
        "jurisdiction", "arbitration", "breach", "indemnification"
    ]
    
    # HR keywords
    hr_keywords = [
        "employment", "employee", "compensation", "benefits", "leave policy",
        "performance review", "termination", "code of conduct"
    ]
    
    # Compliance keywords
    compliance_keywords = [
        "compliance", "regulatory", "audit", "risk management", "internal controls",
        "reporting", "standards", "framework"
    ]
    
    # Count matches
    insurance_count = sum(1 for keyword in insurance_keywords if keyword in text_lower)
    legal_count = sum(1 for keyword in legal_keywords if keyword in text_lower)
    hr_count = sum(1 for keyword in hr_keywords if keyword in text_lower)
    compliance_count = sum(1 for keyword in compliance_keywords if keyword in text_lower)
    
    # Return domain with highest count
    counts = {
        DomainType.INSURANCE: insurance_count,
        DomainType.LEGAL: legal_count,
        DomainType.HR: hr_count,
        DomainType.COMPLIANCE: compliance_count
    }
    
    max_domain = max(counts, key=counts.get)
    return max_domain if counts[max_domain] > 0 else DomainType.INSURANCE  # Default to insurance 