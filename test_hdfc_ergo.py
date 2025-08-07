#!/usr/bin/env python3
"""
Test script for HDFC Ergo Easy Health document with accuracy evaluation.
This script tests for document-specific clauses like Cumulative Bonus and Moratorium Period.
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

class HDFCDocumentTester:
    """Test the system with the HDFC Ergo Easy Health document"""

    def __init__(self):
        self.retrieval_service = RetrievalService()
        
        # URL for the HDFC ERGO General Insurance Company Limited Easy Health Policy
        self.document_urls = [
            "https://raw.githubusercontent.com/rajj28/bajajllmtest/main/HDFHLIP23024V072223.pdf", # Replace with your actual RAW URL if different
        ]
        self.document_url = self.document_urls[0]
        
        # 10 important and specific questions for accuracy testing
        self.test_questions = [
            # Definitions
            "How does this policy define a 'Hospital'?",
            "What constitutes an 'Accident' according to the policy definitions?",

            # Unique Benefits & Clauses
            "What is the Cumulative Bonus offered for a claim-free year and what is the maximum limit?",
            "If a claim is made after a Cumulative Bonus has been applied, what happens in the next policy year?",
            "What is the 'Moratorium Period' and how long is it?",
            "Are 'Maternity expenses' covered under this policy?",

            # Exclusions
            "Are expenses incurred by an organ donor covered?",
            "Is preventive care like vaccinations or immunizations covered?",

            # Claims & Conditions
            "What is the 'Free Look Period' for this policy?",
            "What happens if a fraudulent claim is made?",
        ]
        
        # Expected answers for accuracy evaluation, tailored to the HDFC document
        self.expected_answers = {
            "How does this policy define a 'Hospital'?": {
                "keywords": ["institution", "in-patient care", "day care", "10 in-patient beds", "15 in-patient beds", "qualified nursing staff"],
                "expected_confidence": 0.8
            },
            "What constitutes an 'Accident' according to the policy definitions?": {
                "keywords": ["sudden", "unforeseen", "involuntary", "external", "visible", "violent"],
                "expected_confidence": 0.8
            },
            "What is the Cumulative Bonus offered for a claim-free year and what is the maximum limit?": {
                "keywords": ["10%", "ten percent", "cumulative bonus", "claim free", "maximum", "100%"],
                "expected_confidence": 0.8
            },
            "If a claim is made after a Cumulative Bonus has been applied, what happens in the next policy year?": {
                "keywords": ["decrease", "cumulative bonus", "10%", "ten percent", "subsequent policy year"],
                "expected_confidence": 0.8
            },
            "What is the 'Moratorium Period' and how long is it?": {
                "keywords": ["moratorium period", "eight continuous years", "8 years", "no look back", "contestible", "proven fraud"],
                "expected_confidence": 0.8
            },
            "Are 'Maternity expenses' covered under this policy?": {
                "keywords": ["maternity expenses", "childbirth", "caesarean", "lawful medical termination"],
                "expected_confidence": 0.7
            },
            "Are expenses incurred by an organ donor covered?": {
                "keywords": ["expenses incurred", "organ donation", "not covered", "exclusion", "excluded"],
                "expected_confidence": 0.8
            },
            "Is preventive care like vaccinations or immunizations covered?": {
                "keywords": ["preventive care", "vaccination", "inoculation", "immunisations", "not covered", "excluded"],
                "expected_confidence": 0.8
            },
            "What is the 'Free Look Period' for this policy?": {
                "keywords": ["free look period", "fifteen days", "15 days", "return the same if not acceptable"],
                "expected_confidence": 0.7
            },
            "What happens in a fraudulent claim?": {
                "keywords": ["fraudulent", "false statement", "forfeited", "all benefits", "premium paid"],
                "expected_confidence": 0.8
            }
        }
    
    async def run_test(self):
        """Run the complete test"""
        logger.info("Starting HDFC Ergo Easy Health document accuracy test...")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # Initialize system
            logger.info("Initializing retrieval service...")
            if not await self.retrieval_service.initialize():
                logger.error("Failed to initialize retrieval service")
                return
            
            results = None
            for i, url in enumerate(self.document_urls):
                logger.info(f"Trying document URL {i+1}/{len(self.document_urls)}: {url}")
                
                try:
                    logger.info(f"Processing document: {url}")
                    logger.info(f"Testing {len(self.test_questions)} questions for accuracy")
                    
                    results = await self.retrieval_service.process_documents_and_queries(
                        document_url=url,
                        questions=self.test_questions
                    )
                    
                    if results and results.get("answers"):
                        logger.info(f"Successfully processed document with URL {i+1}")
                        break
                    else:
                        logger.warning(f"Failed to process document with URL {i+1}, results were empty.")
                        
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
        
        answer_list = results.get('answers', [])
        
        total_questions = len(self.test_questions)
        processed_answers = len(answer_list)
        successful_questions = 0
        accurate_answers = 0
        high_confidence_answers = 0
        
        logger.info(f"\nProcessing completed in {processing_time:.2f}s")
        logger.info(f"Questions processed: {processed_answers}/{total_questions}")
        
        for i, answer_result in enumerate(answer_list, 1):
            question = answer_result.get('query') or answer_result.get('question', '')
            answer = answer_result.get('answer', '')
            confidence = answer_result.get('confidence', 0.0)
            reasoning = answer_result.get('reasoning', '')
            
            logger.info(f"\nQuestion {i}: {question}")
            logger.info(f"Answer: {answer}")
            logger.info(f"Confidence: {confidence:.2f}")
            logger.info(f"Reasoning: {reasoning}")
            
            if answer and isinstance(answer, str) and answer.strip() and "error" not in answer.lower():
                successful_questions += 1
                
                if confidence >= 0.7:
                    high_confidence_answers += 1
                
                expected = self.expected_answers.get(question, {})
                expected_keywords = expected.get('keywords', [])
                expected_confidence = expected.get('expected_confidence', 0.5)
                
                if expected_keywords:
                    answer_lower = str(answer).lower()
                    found_keywords = sum(1 for keyword in expected_keywords if keyword.lower() in answer_lower)
                    keyword_accuracy = found_keywords / len(expected_keywords)
                    
                    if keyword_accuracy >= 0.5 and confidence >= expected_confidence * 0.8:
                        accurate_answers += 1
                        logger.info(f"✅ ACCURATE (Keywords: {found_keywords}/{len(expected_keywords)}, Accuracy: {keyword_accuracy:.2f})")
                    else:
                        logger.info(f"❌ INACCURATE (Keywords: {found_keywords}/{len(expected_keywords)}, Accuracy: {keyword_accuracy:.2f})")
                else:
                    logger.info("⚠️ No expected keywords defined for this question")
        
        # Calculate metrics
        success_rate = (successful_questions / processed_answers) * 100 if processed_answers > 0 else 0
        accuracy_rate = (accurate_answers / successful_questions) * 100 if successful_questions > 0 else 0
        high_confidence_rate = (high_confidence_answers / successful_questions) * 100 if successful_questions > 0 else 0
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("ACCURACY SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Questions Asked: {total_questions}")
        logger.info(f"Successfully Answered: {successful_questions}")
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
        results_file = f"hdfc_ergo_accuracy_test_results_{timestamp}.json"
        
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
    tester = HDFCDocumentTester()
    await tester.run_test()

if __name__ == "__main__":
    asyncio.run(main())