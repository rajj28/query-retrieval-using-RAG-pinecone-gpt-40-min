#!/usr/bin/env python3
"""
Test script for Edelweiss Health Insurance - Silver Plan with accuracy evaluation.
This script tests for add-on benefits and specific features like the "Well Baby Well Mother" add-on.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import List, Dict, Any

from app.services.retrieval_service import RetrievalService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EdelweissDocumentTester:
    """Test the system with the Edelweiss Health Insurance - Silver Plan document"""

    def __init__(self):
        self.retrieval_service = RetrievalService()
        self.document_urls = [
            "https://raw.githubusercontent.com/rajj28/bajajllmtest/main/EDLHLGA23009V012223.pdf",
        ]
        self.document_url = self.document_urls[0]

        # 10 important questions for accuracy testing
        self.test_questions = [
            # Definitions
            "How does the policy define 'Medically Necessary' treatment?",
            "What is the definition of a 'Pre-existing Disease' in this policy?",

            # Coverage & Benefits
            "What are the benefits covered under the 'Well Baby Well Mother' add-on?",
            "Is 'Air Ambulance Cover' included as a base benefit or an add-on?",
            "What is the coverage for 'Organ Donor Expenses'?",
            "Is there a 'Cumulative Bonus' for not making a claim?",
            
            # Waiting Periods
            "What is the initial waiting period for all illnesses?",
            "What is the waiting period for specific illnesses like joint replacement or cataract?",
            
            # Claims & Conditions
            "What is the 'Free-look period' for this policy?",
            "How is 'Portability' handled according to the policy terms?",
        ]

        # Expected answers for accuracy evaluation
        self.expected_answers = {
            "How does the policy define 'Medically Necessary' treatment?": {
                "keywords": ["required", "illness", "injury", "prescribed", "medical practitioner", "professional standards"],
                "expected_confidence": 0.8
            },
            "What is the definition of a 'Pre-existing Disease' in this policy?": {
                "keywords": ["48 months", "forty-eight", "diagnosed", "medical advice", "treatment was received"],
                "expected_confidence": 0.8
            },
            "What are the benefits covered under the 'Well Baby Well Mother' add-on?": {
                "keywords": ["well baby", "well mother", "new born baby", "vaccination", "maternity benefit"],
                "expected_confidence": 0.7
            },
            "Is 'Air Ambulance Cover' included as a base benefit or an add-on?": {
                "keywords": ["air ambulance", "add on", "additional premium"],
                "expected_confidence": 0.8
            },
            "What is the coverage for 'Organ Donor Expenses'?": {
                "keywords": ["organ donor", "in-patient", "hospitalization", "harvesting of the organ"],
                "expected_confidence": 0.8
            },
            "Is there a 'Cumulative Bonus' for not making a claim?": {
                "keywords": ["cumulative bonus", "10%", "ten percent", "sum insured", "maximum", "50%"],
                "expected_confidence": 0.7
            },
            "What is the initial waiting period for all illnesses?": {
                "keywords": ["30 days", "thirty days", "first policy", "except for accidental injury"],
                "expected_confidence": 0.8
            },
            "What is the waiting period for specific illnesses like joint replacement or cataract?": {
                "keywords": ["24 months", "two years", "2 years", "joint replacement", "cataract"],
                "expected_confidence": 0.8
            },
            "What is the 'Free-look period' for this policy?": {
                "keywords": ["free look", "15 days", "fifteen days", "receipt of the policy"],
                "expected_confidence": 0.7
            },
            "How is 'Portability' handled according to the policy terms?": {
                "keywords": ["portability", "migrate", "another insurer", "continuity benefits", "waiting periods"],
                "expected_confidence": 0.7
            }
        }

    # [The rest of the test script (run_test, _analyze_results, main) remains the same]
    # ...
    
    async def run_test(self):
        """Run the complete test"""
        logger.info("Starting Edelweiss Health Insurance - Silver Plan document accuracy test...")
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
                found_keywords = sum(1 for keyword in expected_keywords if keyword.lower() in answer_lower)
                keyword_accuracy = found_keywords / len(expected_keywords) if expected_keywords else 0
                keyword_accuracy_scores.append(keyword_accuracy)
                
                if keyword_accuracy >= 0.6:  # At least 60% of expected keywords found
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
                found_keywords = [kw for kw in expected_keywords if kw.lower() in answer_lower]
                missing_keywords = [kw for kw in expected_keywords if kw.lower() not in answer_lower]
                
                keyword_accuracy = len(found_keywords) / len(expected_keywords)
                keyword_status = "✅ PASS" if keyword_accuracy >= 0.6 else "❌ FAIL"
                
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
        filename = f"edelweiss_silver_accuracy_test_results_{timestamp}.json"
        
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
    tester = EdelweissDocumentTester()
    await tester.run_test()

if __name__ == "__main__":
    asyncio.run(main())