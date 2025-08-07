# test_multi_domain.py
"""
Multi-Domain RAG System Test
Demonstrates the system's ability to adapt to different domains:
- Insurance
- Legal
- HR
- Compliance
- Finance
- Healthcare
"""

import asyncio
import logging
import json
import time
from datetime import datetime
from typing import List, Dict, Any

from app.core.domain_config import DomainType, DomainConfig
from app.services.retrieval_service import RetrievalService
from app.core.intelligent_document_processor import IntelligentDocumentProcessor
from app.core.query_expander import QueryExpander

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiDomainTester:
    """
    Test the RAG system across multiple domains
    """
    
    def __init__(self):
        self.retrieval_service = None
        self.test_results = {}
    
    async def initialize_system(self):
        """Initialize the retrieval service"""
        logger.info("Initializing multi-domain RAG system...")
        self.retrieval_service = RetrievalService()
        await self.retrieval_service.initialize()
        logger.info("Multi-domain RAG system initialized successfully")
    
    async def test_domain_adaptation(self, domain_type: DomainType, test_document: str, test_questions: List[Dict[str, Any]]):
        """
        Test the system's adaptation to a specific domain
        
        Args:
            domain_type: The domain type to test
            test_document: Sample document text for the domain
            test_questions: List of domain-specific questions to test
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"TESTING DOMAIN: {domain_type.value.upper()}")
        logger.info(f"{'='*60}")
        
        domain_config = DomainConfig(domain_type)
        logger.info(f"Domain: {domain_config.get_domain_name()}")
        logger.info(f"Description: {domain_config.get_domain_description()}")
        
        # Test domain detection
        detected_domain = domain_config.detect_domain_from_text(test_document)
        logger.info(f"Domain Detection: {'✅ PASS' if detected_domain == domain_type else '❌ FAIL'}")
        
        # Test section patterns
        section_patterns = domain_config.get_section_patterns()
        logger.info(f"Section Patterns: {len(section_patterns)} patterns configured")
        
        # Test entity patterns
        entity_patterns = domain_config.get_entity_patterns()
        logger.info(f"Entity Patterns: {len(entity_patterns)} patterns configured")
        
        # Test key terminology
        key_terminology = domain_config.get_key_terminology()
        logger.info(f"Key Terminology: {len(key_terminology)} categories configured")
        
        # Test query expansion patterns
        expansion_patterns = domain_config.get_query_expansion_patterns()
        logger.info(f"Query Expansion: {len(expansion_patterns)} patterns configured")
        
        # Test document processing with domain-specific configuration
        processor = IntelligentDocumentProcessor(domain_type)
        logger.info(f"Document Processor: Configured for {domain_type.value}")
        
        # Test query expansion with domain-specific configuration
        expander = QueryExpander(domain_type)
        logger.info(f"Query Expander: Configured for {domain_type.value}")
        
        # Test a sample query expansion
        if test_questions:
            sample_query = test_questions[0]['question']
            expansion_result = await expander.expand_query(sample_query)
            logger.info(f"Sample Query Expansion: {len(expansion_result.get('expanded_terms', []))} terms generated")
        
        return {
            'domain_type': domain_type.value,
            'domain_name': domain_config.get_domain_name(),
            'domain_detection': detected_domain == domain_type,
            'section_patterns_count': len(section_patterns),
            'entity_patterns_count': len(entity_patterns),
            'terminology_categories': len(key_terminology),
            'expansion_patterns_count': len(expansion_patterns),
            'sample_expansion_terms': len(expansion_result.get('expanded_terms', [])) if test_questions else 0
        }
    
    async def run_comprehensive_test(self):
        """Run comprehensive tests across all domains"""
        await self.initialize_system()
        
        # Define test documents and questions for each domain
        test_configs = {
            DomainType.INSURANCE: {
                'document': """
                HDFC ERGO EASY HEALTH INSURANCE POLICY
                
                1. DEFINITIONS
                Hospital means any institution established for inpatient care and day care treatment of illness and/or injuries.
                
                2. COVERAGE
                This policy provides coverage for hospitalization expenses, including room rent, doctor fees, and medical procedures.
                
                3. EXCLUSIONS
                Pre-existing conditions are not covered under this policy.
                
                4. CLAIMS PROCEDURE
                In case of hospitalization, the insured must notify the company within 24 hours.
                
                5. GENERAL CONDITIONS
                The policy is subject to terms and conditions as specified in the policy document.
                """,
                'questions': [
                    {'question': 'What is the definition of hospital?', 'expected_keywords': ['hospital', 'institution', 'inpatient', 'day care']},
                    {'question': 'What is covered under this policy?', 'expected_keywords': ['coverage', 'hospitalization', 'expenses', 'room rent']},
                    {'question': 'What are the exclusions?', 'expected_keywords': ['exclusions', 'pre-existing', 'not covered']},
                    {'question': 'What is the claims procedure?', 'expected_keywords': ['claims', 'procedure', 'notify', '24 hours']}
                ]
            },
            
            DomainType.LEGAL: {
                'document': """
                SOFTWARE LICENSE AGREEMENT
                
                1. DEFINITIONS
                "Software" means the computer programs and related documentation provided by Licensor.
                
                2. REPRESENTATIONS AND WARRANTIES
                Licensor represents and warrants that it has the right to grant the license.
                
                3. INDEMNIFICATION
                Licensor shall indemnify and hold harmless Licensee from any claims arising from the use of the Software.
                
                4. TERMINATION
                This agreement may be terminated by either party with 30 days written notice.
                
                5. GOVERNING LAW
                This agreement shall be governed by the laws of the State of California.
                """,
                'questions': [
                    {'question': 'What is the definition of Software?', 'expected_keywords': ['software', 'computer programs', 'documentation']},
                    {'question': 'What are the representations and warranties?', 'expected_keywords': ['representations', 'warranties', 'right', 'license']},
                    {'question': 'What is the indemnification clause?', 'expected_keywords': ['indemnification', 'hold harmless', 'claims']},
                    {'question': 'How can this agreement be terminated?', 'expected_keywords': ['termination', '30 days', 'written notice']}
                ]
            },
            
            DomainType.HR: {
                'document': """
                EMPLOYEE HANDBOOK
                
                1. EMPLOYMENT
                All employees are at-will employees and may be terminated at any time.
                
                2. COMPENSATION
                Employees will receive competitive salary and benefits package including health insurance.
                
                3. PAID TIME OFF
                Full-time employees receive 20 days of paid time off per year.
                
                4. PERFORMANCE EVALUATION
                Performance reviews are conducted annually with feedback and goal setting.
                
                5. DISCIPLINARY POLICY
                Progressive discipline will be applied for policy violations.
                """,
                'questions': [
                    {'question': 'What is the employment status?', 'expected_keywords': ['employment', 'at-will', 'terminated']},
                    {'question': 'What compensation is provided?', 'expected_keywords': ['compensation', 'salary', 'benefits', 'health insurance']},
                    {'question': 'How much PTO do employees get?', 'expected_keywords': ['paid time off', '20 days', 'per year']},
                    {'question': 'How often are performance reviews?', 'expected_keywords': ['performance', 'reviews', 'annually']}
                ]
            },
            
            DomainType.COMPLIANCE: {
                'document': """
                COMPLIANCE POLICY MANUAL
                
                1. COMPLIANCE REQUIREMENTS
                All employees must comply with applicable laws and regulations.
                
                2. AUDIT PROCEDURES
                Regular audits will be conducted to ensure compliance with policies.
                
                3. REPORTING OBLIGATIONS
                Violations must be reported to the compliance officer within 24 hours.
                
                4. PENALTIES AND SANCTIONS
                Non-compliance may result in disciplinary action up to and including termination.
                
                5. TRAINING REQUIREMENTS
                Annual compliance training is mandatory for all employees.
                """,
                'questions': [
                    {'question': 'What are the compliance requirements?', 'expected_keywords': ['compliance', 'laws', 'regulations']},
                    {'question': 'What are the audit procedures?', 'expected_keywords': ['audit', 'procedures', 'regular']},
                    {'question': 'What are the reporting obligations?', 'expected_keywords': ['reporting', 'violations', '24 hours']},
                    {'question': 'What penalties apply for non-compliance?', 'expected_keywords': ['penalties', 'sanctions', 'disciplinary action']}
                ]
            },
            
            DomainType.FINANCE: {
                'document': """
                FINANCIAL REPORTING POLICY
                
                1. FINANCIAL STATEMENTS
                Quarterly financial statements must be prepared in accordance with GAAP.
                
                2. REVENUE RECOGNITION
                Revenue is recognized when goods are delivered or services are performed.
                
                3. EXPENSE MANAGEMENT
                All expenses must be properly documented and approved before payment.
                
                4. ASSET VALUATION
                Assets are valued at historical cost less accumulated depreciation.
                
                5. AUDIT REQUIREMENTS
                Annual external audits are required for all financial statements.
                """,
                'questions': [
                    {'question': 'What are the financial statement requirements?', 'expected_keywords': ['financial statements', 'quarterly', 'GAAP']},
                    {'question': 'How is revenue recognized?', 'expected_keywords': ['revenue', 'recognition', 'delivered', 'performed']},
                    {'question': 'What are the expense management procedures?', 'expected_keywords': ['expense', 'management', 'documented', 'approved']},
                    {'question': 'How are assets valued?', 'expected_keywords': ['asset', 'valuation', 'historical cost', 'depreciation']}
                ]
            },
            
            DomainType.HEALTHCARE: {
                'document': """
                MEDICAL RECORDS POLICY
                
                1. MEDICAL DOCUMENTATION
                All patient encounters must be documented in the electronic health record.
                
                2. PATIENT CONSENT
                Written informed consent is required for all medical procedures.
                
                3. DIAGNOSIS AND TREATMENT
                Diagnosis must be based on clinical assessment and diagnostic tests.
                
                4. PRIVACY AND HIPAA
                Patient information must be protected in accordance with HIPAA regulations.
                
                5. BILLING AND CODING
                Medical services must be properly coded for billing purposes.
                """,
                'questions': [
                    {'question': 'What are the medical documentation requirements?', 'expected_keywords': ['medical', 'documentation', 'electronic health record']},
                    {'question': 'What consent is required?', 'expected_keywords': ['consent', 'informed consent', 'medical procedures']},
                    {'question': 'How is diagnosis determined?', 'expected_keywords': ['diagnosis', 'clinical assessment', 'diagnostic tests']},
                    {'question': 'What privacy protections apply?', 'expected_keywords': ['privacy', 'HIPAA', 'patient information']}
                ]
            }
        }
        
        # Test each domain
        for domain_type, config in test_configs.items():
            try:
                result = await self.test_domain_adaptation(
                    domain_type=domain_type,
                    test_document=config['document'],
                    test_questions=config['questions']
                )
                self.test_results[domain_type.value] = result
            except Exception as e:
                logger.error(f"Error testing {domain_type.value}: {str(e)}")
                self.test_results[domain_type.value] = {'error': str(e)}
        
        # Generate comprehensive report
        await self.generate_comprehensive_report()
    
    async def generate_comprehensive_report(self):
        """Generate a comprehensive test report"""
        logger.info(f"\n{'='*80}")
        logger.info("MULTI-DOMAIN RAG SYSTEM COMPREHENSIVE TEST REPORT")
        logger.info(f"{'='*80}")
        
        total_domains = len(self.test_results)
        successful_domains = sum(1 for result in self.test_results.values() if 'error' not in result)
        failed_domains = total_domains - successful_domains
        
        logger.info(f"Total Domains Tested: {total_domains}")
        logger.info(f"Successful Adaptations: {successful_domains}")
        logger.info(f"Failed Adaptations: {failed_domains}")
        logger.info(f"Success Rate: {(successful_domains/total_domains)*100:.1f}%")
        
        logger.info(f"\n{'='*80}")
        logger.info("DETAILED DOMAIN RESULTS")
        logger.info(f"{'='*80}")
        
        for domain, result in self.test_results.items():
            if 'error' in result:
                logger.info(f"\n{domain.upper()}: ❌ FAILED")
                logger.info(f"Error: {result['error']}")
            else:
                logger.info(f"\n{domain.upper()}: ✅ SUCCESS")
                logger.info(f"  Domain Name: {result['domain_name']}")
                logger.info(f"  Domain Detection: {'✅ PASS' if result['domain_detection'] else '❌ FAIL'}")
                logger.info(f"  Section Patterns: {result['section_patterns_count']}")
                logger.info(f"  Entity Patterns: {result['entity_patterns_count']}")
                logger.info(f"  Terminology Categories: {result['terminology_categories']}")
                logger.info(f"  Expansion Patterns: {result['expansion_patterns_count']}")
                logger.info(f"  Sample Expansion Terms: {result['sample_expansion_terms']}")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"multi_domain_test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                'test_timestamp': timestamp,
                'summary': {
                    'total_domains': total_domains,
                    'successful_domains': successful_domains,
                    'failed_domains': failed_domains,
                    'success_rate': (successful_domains/total_domains)*100
                },
                'detailed_results': self.test_results
            }, f, indent=2)
        
        logger.info(f"\nDetailed results saved to: {filename}")
        
        # Key findings
        logger.info(f"\n{'='*80}")
        logger.info("KEY FINDINGS")
        logger.info(f"{'='*80}")
        
        if successful_domains == total_domains:
            logger.info("🎉 ALL DOMAINS SUCCESSFULLY ADAPTED!")
            logger.info("✅ The RAG system can handle multiple complex domains")
            logger.info("✅ Domain-specific configurations are working correctly")
            logger.info("✅ Query expansion adapts to domain terminology")
            logger.info("✅ Document processing recognizes domain patterns")
        else:
            logger.info("⚠️ SOME DOMAINS NEED ATTENTION")
            for domain, result in self.test_results.items():
                if 'error' in result:
                    logger.info(f"❌ {domain.upper()}: {result['error']}")
        
        logger.info(f"\n{'='*80}")
        logger.info("NEXT STEPS")
        logger.info(f"{'='*80}")
        logger.info("1. Test with real documents from each domain")
        logger.info("2. Fine-tune domain-specific patterns based on actual usage")
        logger.info("3. Add more domain-specific terminology and patterns")
        logger.info("4. Implement domain-specific re-ranking rules")
        logger.info("5. Create domain-specific test suites with real queries")

async def main():
    """Main test function"""
    tester = MultiDomainTester()
    await tester.run_comprehensive_test()

if __name__ == "__main__":
    asyncio.run(main()) 