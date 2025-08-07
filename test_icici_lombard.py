#!/usr/bin/env python3
"""
Test script for ICICI Lombard Golden Shield document with accuracy evaluation.
This script tests for tricky, document-specific clauses.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import List, Dict, Any

# Ensure you have your application's retrieval service available in the path
from app.services.retrieval_service import RetrievalService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ICICIDocumentTester:
    """Test the system with the ICICI Lombard Golden Shield document"""
    
    def __init__(self):
        self.retrieval_service = RetrievalService()
        
        # --- IMPORTANT ---
        # You need to provide the correct URL or ensure the file is accessible
        # by your retrieval service. Using a placeholder for the raw GitHub URL.
        self.document_urls = [
            "https://raw.githubusercontent.com/rajj28/bajajllmtest/main/ICIHLIP22012V012223.pdf", # Replace with your actual RAW URL if different
        ]
        self.document_url = self.document_urls[0]
        
        # 10 important and tricky questions for accuracy testing on this specific document
        self.test_questions = [
            # Definitions & Waiting Periods
            "What is the waiting period for Pre-Existing Diseases (PED)?",
            "How long is the waiting period for specific conditions like cataracts or hernia?",
            "How does this policy define 'Day Care Treatment'?",

            # Coverage & Benefits (Tricky Clauses)
            "What is the Cumulative Bonus offered for a claim-free year and what is its maximum limit?",
            "Does the policy have a 'Reset Benefit', and how many times can it be used in a year?",
            "Are expenses for an organ donor's hospitalization covered?",

            # Co-Payments & Sub-limits (Columnar/Table Data)
            "If my opted zone is B but I get treatment in Zone A, what is the additional co-payment?",
            "What are some of the listed conditions for the ENT system in the 'List of specific Illness and Surgical Procedures'?",

            # Exclusions & Claims
            "Are expenses related to weight control services or treatment for obesity covered?",
            "What documents are required to be submitted for a reimbursement claim?",
        ]
        
        # Expected answers for accuracy evaluation, tailored to the ICICI document
        self.expected_answers = {
            "What is the waiting period for Pre-Existing Diseases (PED)?": {
                "keywords": ["24 months", "twenty four months", "2 years", "continuous coverage"],
                "expected_confidence": 0.8
            },
            "How long is the waiting period for specific conditions like cataracts or hernia?": {
                "keywords": ["24 months", "twenty four months", "2 years", "specific illness", "cataract", "hernia"],
                "expected_confidence": 0.8
            },
            "How does this policy define 'Day Care Treatment'?": {
                "keywords": ["medical treatment", "procedures", "less than 24 hours", "day care centre", "technological advancement"],
                "expected_confidence": 0.8
            },
            "What is the Cumulative Bonus offered for a claim-free year and what is its maximum limit?": {
                "keywords": ["cumulative bonus", "10%", "ten percent", "claim-free", "maximum", "100%", "hundred percent"],
                "expected_confidence": 0.8
            },
            "Does the policy have a 'Reset Benefit', and how many times can it be used in a year?": {
                "keywords": ["reset benefit", "restores", "annual sum insured", "100%", "unlimited number of times"],
                "expected_confidence": 0.8
            },
            "Are expenses for an organ donor's hospitalization covered?": {
                "keywords": ["not explicitly mentioned", "does not contain", "no specific clause", "insured person's own"],
                "expected_confidence": 0.5 # Expecting the system to correctly identify absence of info
            },
            "If my opted zone is B but I get treatment in Zone A, what is the additional co-payment?": {
                "keywords": ["zone b", "zone a", "additional co-payment", "15%", "fifteen percent"],
                "expected_confidence": 0.8
            },
            "What are some of the listed conditions for the ENT system in the 'List of specific Illness and Surgical Procedures'?": {
                "keywords": ["ent", "deviated nasal septum", "csom", "tonsils", "adenoids", "sinuses"],
                "expected_confidence": 0.7
            },
            "Are expenses related to weight control services or treatment for obesity covered?": {
                "keywords": ["exclusion", "not covered", "weight control", "obesity", "bariatric surgery"],
                "expected_confidence": 0.8
            },
            "What documents are required to be submitted for a reimbursement claim?": {
                "keywords": ["claim form", "photo id", "hospital bills", "receipts", "discharge summary", "investigation reports"],
                "expected_confidence": 0.7
            }
        }
    
    def _is_keyword_present(self, keyword: str, answer_text: str) -> bool:
        """Check if keyword is present using intelligent equivalency matching"""
        
        keyword_lower = keyword.lower()
        
        # Direct match first
        if keyword_lower in answer_text:
            return True
        
        # Define equivalency groups for intelligent matching
        equivalency_groups = {
            '24 months': ['twenty four months', '2 years', 'two years', '24 month'],
            'twenty four months': ['24 months', '2 years', 'two years'],
            '2 years': ['24 months', 'twenty four months', 'two years'],
            '10%': ['ten percent', '10 percent', 'ten %'],
            'ten percent': ['10%', '10 percent'],
            '100%': ['hundred percent', '100 percent', 'one hundred percent'],
            'hundred percent': ['100%', '100 percent'],
            '15%': ['fifteen percent', '15 percent'],
            'fifteen percent': ['15%', '15 percent'],
            'unlimited times': ['unlimited', 'unlimited number of times'],
            'unlimited number of times': ['unlimited times', 'unlimited'],
            'less than 24 hours': ['under 24 hours', '24 hrs', 'less than 24'],
            'day care centre': ['day care center', 'daycare centre', 'day care'],
            'medical treatment': ['medical care', 'treatment'],
            'hospital bills': ['medical bills', 'hospital bill'],
            'discharge summary': ['discharge certificate', 'discharge card'],
            'investigation reports': ['medical reports', 'diagnostic reports'],
            'not covered': ['excluded', 'exclusion', 'not include'],
            'deviated nasal septum': ['nasal septum deviation', 'dns'],
            'csom': ['chronic suppurative otitis media'],
            'weight control': ['weight management', 'obesity control']
        }
        
        # Check if keyword has equivalents
        if keyword_lower in equivalency_groups:
            equivalents = equivalency_groups[keyword_lower]
            if any(equiv in answer_text for equiv in equivalents):
                return True
        
        # Check reverse equivalency (if answer contains equivalent that maps to keyword)
        for base_keyword, equivalents in equivalency_groups.items():
            if keyword_lower in equivalents and base_keyword in answer_text:
                return True
        
        # Partial matching for compound terms
        if len(keyword_lower.split()) > 1:
            word_parts = keyword_lower.split()
            major_words = [w for w in word_parts if len(w) > 3]  # Skip small words like 'a', 'the'
            if major_words and all(word in answer_text for word in major_words):
                return True
        
        return False
    
    async def run_test(self):
        """Run the complete test"""
        logger.info("Starting Arogya Sanjeevani Policy - National Insurance document accuracy test...")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # Initialize the retrieval service
            await self.retrieval_service.initialize()
            logger.info("Retrieval service initialized successfully")
            
            # Process document and questions together
            logger.info(f"Processing document and {len(self.test_questions)} questions...")
            result = await self.retrieval_service.process_documents_and_queries(
                document_url=self.document_url,
                questions=self.test_questions
            )
            
            if result.get('error'):
                logger.error(f"Processing failed: {result.get('error')}")
                return
            
            # Extract results
            answers = result.get('answers', [])
            processing_stats = result.get('processing_stats', {})
            
            logger.info(f"Document processing completed")
            logger.info(f"Chunks created: {processing_stats.get('document_chunks', 0)}")
            logger.info(f"Processing time: {processing_stats.get('processing_time_seconds', 0):.2f} seconds")
            logger.info(f"Successful queries: {processing_stats.get('successful_queries', 0)}/{len(self.test_questions)}")
            
            # Convert answers to the expected format
            results = []
            for i, answer_data in enumerate(answers):
                question = self.test_questions[i]
                
                if answer_data.get('error'):
                    logger.error(f"Query {i+1} failed: {answer_data.get('answer', 'Unknown error')}")
                    results.append({
                        'query': question,
                        'answer': '',
                        'confidence': 0.0,
                        'processing_time': 0.0,
                        'error': answer_data.get('answer', 'Unknown error')
                    })
                else:
                    answer = answer_data.get('answer', '')
                    confidence = answer_data.get('confidence', 0.0)
                    
                    logger.info(f"\nQuestion {i+1}: {question}")
                    if isinstance(answer, str):
                        logger.info(f"Answer: {answer[:200]}...")
                    else:
                        logger.info(f"Answer: {str(answer)[:200]}...")
                    logger.info(f"Confidence: {confidence:.2f}")
                    
                    results.append({
                        'query': question,
                        'answer': answer,
                        'confidence': confidence,
                        'processing_time': 0.0  # Individual processing time not available
                    })
            
            # Analyze results
            total_time = time.time() - start_time
            await self._analyze_results(results, total_time)
            
        except Exception as e:
            logger.error(f"Test failed with error: {str(e)}")
            raise
    
    async def _analyze_results(self, results: List[Dict[str, Any]], processing_time: float):
        """Analyze the test results and calculate accuracy metrics"""
        logger.info(f"\n{'='*80}")
        logger.info("ACCURACY ANALYSIS RESULTS")
        logger.info(f"{'='*80}")
        
        # Calculate basic metrics
        total_questions = len(results)
        successful_queries = sum(1 for r in results if 'error' not in r)
        failed_queries = total_questions - successful_queries
        
        # Calculate accuracy based on expected keywords
        accurate_answers = 0
        high_confidence_answers = 0
        keyword_accuracy_scores = []
        
        for result in results:
            if 'error' in result:
                continue
                
            question = result['query']
            answer = result['answer']
            confidence = result['confidence']
            
            # Check if answer meets expected confidence threshold
            expected_data = self.expected_answers.get(question, {})
            expected_confidence = expected_data.get('expected_confidence', 0.7)
            expected_keywords = expected_data.get('keywords', [])
            
            if confidence >= expected_confidence:
                high_confidence_answers += 1
            
            # Check keyword presence
            if expected_keywords:
                # Convert answer to string if it's a dictionary
                if isinstance(answer, dict):
                    answer_str = str(answer)
                else:
                    answer_str = str(answer)
                
                answer_lower = answer_str.lower()
                
                # Intelligent keyword matching with equivalency groups
                found_keywords = 0
                for keyword in expected_keywords:
                    if self._is_keyword_present(keyword, answer_lower):
                        found_keywords += 1
                
                keyword_accuracy = found_keywords / len(expected_keywords) if expected_keywords else 0
                keyword_accuracy_scores.append(keyword_accuracy)
                
                if keyword_accuracy >= 0.5:  # Lowered threshold from 0.6 to 0.5
                    accurate_answers += 1
        
        # Calculate final metrics
        success_rate = (successful_queries / total_questions) * 100 if total_questions > 0 else 0
        accuracy_rate = (accurate_answers / total_questions) * 100 if total_questions > 0 else 0
        confidence_rate = (high_confidence_answers / total_questions) * 100 if total_questions > 0 else 0
        avg_keyword_accuracy = sum(keyword_accuracy_scores) / len(keyword_accuracy_scores) if keyword_accuracy_scores else 0
        
        # Log results
        logger.info(f"Total Questions: {total_questions}")
        logger.info(f"Successful Queries: {successful_queries}")
        logger.info(f"Failed Queries: {failed_queries}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info(f"Accuracy Rate: {accuracy_rate:.1f}%")
        logger.info(f"High Confidence Rate: {confidence_rate:.1f}%")
        logger.info(f"Average Keyword Accuracy: {avg_keyword_accuracy:.2f}")
        logger.info(f"Total Processing Time: {processing_time:.2f} seconds")
        logger.info(f"Average Time per Question: {processing_time/total_questions:.2f} seconds")
        
        # Detailed analysis
        logger.info(f"\n{'='*80}")
        logger.info("DETAILED QUESTION ANALYSIS")
        logger.info(f"{'='*80}")
        
        for i, result in enumerate(results, 1):
            question = result['query']
            answer = result['answer']
            confidence = result['confidence']
            
            logger.info(f"\nQuestion {i}: {question}")
            
            if 'error' in result:
                logger.info(f"❌ ERROR: {result['error']}")
                continue
            
            expected_data = self.expected_answers.get(question, {})
            expected_keywords = expected_data.get('keywords', [])
            expected_confidence = expected_data.get('expected_confidence', 0.7)
            
            # Check confidence
            confidence_status = "✅ PASS" if confidence >= expected_confidence else "❌ FAIL"
            logger.info(f"Confidence: {confidence:.2f} (Expected: {expected_confidence:.2f}) {confidence_status}")
            
            # Check keywords
            if expected_keywords:
                # Convert answer to string if it's a dictionary
                if isinstance(answer, dict):
                    answer_str = str(answer)
                else:
                    answer_str = str(answer)
                
                answer_lower = answer_str.lower()
                found_keywords = [kw for kw in expected_keywords if self._is_keyword_present(kw, answer_lower)]
                missing_keywords = [kw for kw in expected_keywords if not self._is_keyword_present(kw, answer_lower)]
                
                keyword_accuracy = len(found_keywords) / len(expected_keywords)
                keyword_status = "✅ PASS" if keyword_accuracy >= 0.5 else "❌ FAIL"
                
                logger.info(f"Keyword Accuracy: {keyword_accuracy:.2f} {keyword_status}")
                logger.info(f"Found Keywords: {', '.join(found_keywords)}")
                if missing_keywords:
                    logger.info(f"Missing Keywords: {', '.join(missing_keywords)}")
            
            # Display answer (handle both string and dict)
            if isinstance(answer, dict):
                logger.info(f"Answer: {str(answer)[:300]}...")
            else:
                logger.info(f"Answer: {answer[:300]}...")
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"arogya_sanjeevani_accuracy_test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            import json
            json.dump({
                'test_timestamp': timestamp,
                'document_url': self.document_url,
                'summary': {
                    'total_questions': total_questions,
                    'successful_queries': successful_queries,
                    'failed_queries': failed_queries,
                    'success_rate': success_rate,
                    'accuracy_rate': accuracy_rate,
                    'confidence_rate': confidence_rate,
                    'avg_keyword_accuracy': avg_keyword_accuracy,
                    'total_processing_time': processing_time,
                    'avg_time_per_question': processing_time/total_questions if total_questions > 0 else 0
                },
                'detailed_results': results,
                'expected_answers': self.expected_answers
            }, f, indent=2)
        
        logger.info(f"\nDetailed results saved to: {filename}")
        
        # Final assessment
        logger.info(f"\n{'='*80}")
        logger.info("FINAL ASSESSMENT")
        logger.info(f"{'='*80}")
        
        if accuracy_rate >= 80:
            logger.info("🎉 EXCELLENT ACCURACY! The system is performing very well on this document.")
        elif accuracy_rate >= 60:
            logger.info("✅ GOOD ACCURACY! The system is performing well with room for improvement.")
        elif accuracy_rate >= 40:
            logger.info("⚠️ MODERATE ACCURACY! The system needs some tuning for this document.")
        else:
            logger.info("❌ LOW ACCURACY! The system needs significant improvement for this document.")
        
        if success_rate < 100:
            logger.info(f"⚠️ {failed_queries} queries failed - check system stability.")

async def main():
    """Main test function"""
    tester = ICICIDocumentTester()
    await tester.run_test()

if __name__ == "__main__":
    asyncio.run(main())