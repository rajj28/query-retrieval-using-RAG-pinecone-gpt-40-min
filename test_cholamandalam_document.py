#!/usr/bin/env python3
"""
Test script for Bajaj Allianz document with accuracy evaluation
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

class CholamandalamDocumentTester:
    """Test the system with Cholamandalam MS Group Domestic Travel Insurance document"""
    
    def __init__(self):
        self.retrieval_service = RetrievalService()
        # Try alternative URLs for the Cholamandalam MS Group Domestic Travel Insurance document
        self.document_urls = [
            "https://raw.githubusercontent.com/rajj28/bajajllmtest/main/CHOTGDP23004V012223.pdf",  # Raw GitHub URL
            "https://github.com/rajj28/bajajllmtest/blob/main/CHOTGDP23004V012223.pdf",  # Original URL
        ]
        self.document_url = self.document_urls[0]  # Start with raw GitHub URL
        
        # 10 important questions for accuracy testing
        self.test_questions = [
            # Definitions (3 questions)
            "What is the definition of 'Accident' or 'Accidental' in this policy?",
            "How does the policy define a 'Hospital'? What are the minimum criteria?",
            "What is the definition of 'Pre-existing Disease'?",
            
            # Coverage (4 questions)
            "What are the base covers provided by this policy?",
            "What is covered under 'Emergency Accidental Hospitalization'?",
            "What does the 'Personal Accident Covers' include?",
            "Is 'OPD Treatment' covered?",
            
            # Exclusions (2 questions)
            "What are the general exclusions mentioned in the policy?",
            "Are pre-existing conditions covered?",
            
            # Claims (1 question)
            "What is the process for notifying a claim?"
        ]
        
        # Expected answers for accuracy evaluation
        self.expected_answers = {
            "What is the definition of 'Accident' or 'Accidental' in this policy?": {
                "keywords": ["sudden", "unforeseen", "involuntary", "external", "visible", "violent"],
                "expected_confidence": 0.8
            },
            "How does the policy define a 'Hospital'? What are the minimum criteria?": {
                "keywords": ["institution", "inpatient", "day care", "registered", "nursing staff", "beds", "operation theatre"],
                "expected_confidence": 0.8
            },
            "What is the definition of 'Pre-existing Disease'?": {
                "keywords": ["condition", "ailment", "injury", "diagnosed", "48 months", "effective date"],
                "expected_confidence": 0.8
            },
            "What are the base covers provided by this policy?": {
                "keywords": ["emergency", "accidental", "hospitalization", "opd", "medical expenses", "personal accident"],
                "expected_confidence": 0.7
            },
            "What is covered under 'Emergency Accidental Hospitalization'?": {
                "keywords": ["reasonable", "customary", "medical expenses", "accidental injuries", "in-patient", "hospitalization"],
                "expected_confidence": 0.7
            },
            "What does the 'Personal Accident Covers' include?": {
                "keywords": ["accidental bodily injury", "trip", "accidental", "violent", "external", "visible"],
                "expected_confidence": 0.7
            },
            "Is 'OPD Treatment' covered?": {
                "keywords": ["opd", "treatment", "covered", "clinic", "hospital", "diagnosis"],
                "expected_confidence": 0.6
            },
            "What are the general exclusions mentioned in the policy?": {
                "keywords": ["exclusions", "pre-existing", "suicide", "self-inflicted", "intoxicating", "drugs"],
                "expected_confidence": 0.7
            },
            "Are pre-existing conditions covered?": {
                "keywords": ["not designed", "pre-existing condition", "medical services"],
                "expected_confidence": 0.8
            },
            "What is the process for notifying a claim?": {
                "keywords": ["immediately contact", "assistance service provider", "policy details", "contact details"],
                "expected_confidence": 0.7
            }
        }
    
    async def run_test(self):
        """Run the complete test"""
        logger.info("Starting Cholamandalam MS Group Domestic Travel Insurance document accuracy test...")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # Initialize system
            logger.info("Initializing retrieval service...")
            if not await self.retrieval_service.initialize():
                logger.error("Failed to initialize retrieval service")
                return
            
            # Try different URLs if one fails
            results = None
            for i, url in enumerate(self.document_urls):
                logger.info(f"Trying document URL {i+1}/{len(self.document_urls)}: {url}")
                
                try:
                    # Process document and questions
                    logger.info(f"Processing document: {url}")
                    logger.info(f"Testing {len(self.test_questions)} questions for accuracy")
                    
                    results = await self.retrieval_service.process_documents_and_queries(
                        document_url=url,
                        questions=self.test_questions
                    )
                    
                    if results:
                        logger.info(f"Successfully processed document with URL {i+1}")
                        break
                    else:
                        logger.warning(f"Failed to process document with URL {i+1}")
                        
                except Exception as e:
                    logger.warning(f"Error processing document with URL {i+1}: {str(e)}")
                    continue
            
            if not results:
                logger.error("Failed to process document with all URLs")
                return
            
            # Analyze results
            processing_time = time.time() - start_time
            await self._analyze_results(results, processing_time)
            
        except Exception as e:
            logger.error(f"Test failed: {str(e)}")
    
    async def _analyze_results(self, results: Dict[str, Any], processing_time: float):
        """Analyze test results for accuracy"""
        logger.info("\n" + "=" * 80)
        logger.info("ACCURACY ANALYSIS RESULTS")
        logger.info("=" * 80)
        
        # Debug: Log the actual structure of results
        logger.info(f"Results keys: {list(results.keys())}")
        
        # Try different possible keys for the answers
        answers = results.get('answers', [])
        if not answers:
            answers = results.get('questions', [])
        if not answers:
            answers = results.get('query_results', [])
        if not answers:
            answers = results.get('results', [])
        
        total_questions = len(answers)
        successful_questions = 0
        accurate_answers = 0
        high_confidence_answers = 0
        
        logger.info(f"\nProcessing completed in {processing_time:.2f}s")
        logger.info(f"Questions processed: {total_questions}")
        
        # Analyze each question
        for i, answer_result in enumerate(answers, 1):
            # Try different possible keys for question and answer fields
            question = answer_result.get('question', '')
            if not question:
                question = answer_result.get('query', '')
            if not question:
                question = answer_result.get('question_text', '')
            
            answer = answer_result.get('answer', '')
            if not answer:
                answer = answer_result.get('response', '')
            if not answer:
                answer = answer_result.get('result', '')
            
            confidence = answer_result.get('confidence', 0.0)
            reasoning = answer_result.get('reasoning', '')
            if not reasoning:
                reasoning = answer_result.get('explanation', '')
            
            logger.info(f"\nQuestion {i}: {question}")
            logger.info(f"Answer: {answer}")
            logger.info(f"Confidence: {confidence:.2f}")
            logger.info(f"Reasoning: {reasoning}")
            
            # Check if question was processed successfully
            if answer and isinstance(answer, str) and answer.strip():
                successful_questions += 1
                
                # Check confidence level
                if confidence >= 0.7:
                    high_confidence_answers += 1
                
                # Check accuracy against expected keywords
                expected = self.expected_answers.get(question, {})
                expected_keywords = expected.get('keywords', [])
                expected_confidence = expected.get('expected_confidence', 0.5)
                
                if expected_keywords:
                    # Count how many expected keywords are found in the answer
                    if isinstance(answer, str):
                        answer_lower = answer.lower()
                    else:
                        answer_lower = str(answer).lower()
                    found_keywords = sum(1 for keyword in expected_keywords if keyword.lower() in answer_lower)
                    keyword_accuracy = found_keywords / len(expected_keywords)
                    
                    # Consider answer accurate if it has good keyword coverage and reasonable confidence
                    if keyword_accuracy >= 0.5 and confidence >= expected_confidence * 0.8:
                        accurate_answers += 1
                        logger.info(f"✅ ACCURATE (Keywords: {found_keywords}/{len(expected_keywords)}, Accuracy: {keyword_accuracy:.2f})")
                    else:
                        logger.info(f"❌ INACCURATE (Keywords: {found_keywords}/{len(expected_keywords)}, Accuracy: {keyword_accuracy:.2f})")
                else:
                    logger.info("⚠️ No expected keywords defined for this question")
        
        # Calculate metrics
        success_rate = (successful_questions / total_questions) * 100 if total_questions > 0 else 0
        accuracy_rate = (accurate_answers / successful_questions) * 100 if successful_questions > 0 else 0
        high_confidence_rate = (high_confidence_answers / successful_questions) * 100 if successful_questions > 0 else 0
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("ACCURACY SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Questions: {total_questions}")
        logger.info(f"Successful Questions: {successful_questions}")
        logger.info(f"Accurate Answers: {accurate_answers}")
        logger.info(f"High Confidence Answers (≥0.7): {high_confidence_answers}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info(f"Accuracy Rate: {accuracy_rate:.1f}%")
        logger.info(f"High Confidence Rate: {high_confidence_rate:.1f}%")
        logger.info(f"Processing Time: {processing_time:.2f}s")
        
        # Performance assessment
        if accuracy_rate >= 80:
            logger.info("🏆 EXCELLENT ACCURACY - System performing very well!")
        elif accuracy_rate >= 60:
            logger.info("✅ GOOD ACCURACY - System performing well")
        elif accuracy_rate >= 40:
            logger.info("⚠️ MODERATE ACCURACY - Room for improvement")
        else:
            logger.info("❌ LOW ACCURACY - Needs significant improvement")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"cholamandalam_accuracy_test_results_{timestamp}.json"
        
        # Add accuracy metrics to results
        results['accuracy_metrics'] = {
            'total_questions': total_questions,
            'successful_questions': successful_questions,
            'accurate_answers': accurate_answers,
            'high_confidence_answers': high_confidence_answers,
            'success_rate': success_rate,
            'accuracy_rate': accuracy_rate,
            'high_confidence_rate': high_confidence_rate,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"\nResults saved to: {results_file}")
        logger.info("=" * 80)

async def main():
    """Main test function"""
    tester = CholamandalamDocumentTester()
    await tester.run_test()

if __name__ == "__main__":
    asyncio.run(main()) 