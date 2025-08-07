# app/core/fine_tuning_dataset.py
"""
Fine-tuning dataset creation for insurance document embeddings
Creates training pairs to teach the model domain-specific similarity
"""

import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import os

logger = logging.getLogger(__name__)

@dataclass
class TrainingPair:
    """A training pair for fine-tuning"""
    question: str
    positive_context: str  # The "golden" answer chunk
    negative_contexts: List[str]  # Distractor chunks
    metadata: Dict[str, Any]
    confidence_score: float

class FineTuningDatasetManager:
    """
    Manages creation and management of fine-tuning datasets for insurance embeddings
    """
    
    def __init__(self, dataset_path: str = "data/fine_tuning"):
        self.dataset_path = dataset_path
        os.makedirs(dataset_path, exist_ok=True)
        
        # Insurance-specific training examples
        self.training_examples = [
            # Free Look Period examples
            {
                "question": "What is the Free Look Period?",
                "positive_context": "Free Look Period: The Insured Person shall be allowed a period of fifteen days from the date of receipt of the Policy to review the terms and conditions and return the same if not acceptable.",
                "negative_contexts": [
                    "Policy Period: This policy is valid for one year from the date of commencement.",
                    "Premium Payment: Premiums must be paid annually in advance.",
                    "Claims Process: All claims must be submitted within 30 days of the incident."
                ],
                "category": "policy_terms",
                "confidence": 1.0
            },
            {
                "question": "How long is the free look period?",
                "positive_context": "Free Look Period: The Insured Person shall be allowed a period of fifteen days from the date of receipt of the Policy to review the terms and conditions and return the same if not acceptable.",
                "negative_contexts": [
                    "Waiting Period: Pre-existing conditions have a waiting period of 48 months.",
                    "Policy Term: The policy is valid for 12 months from inception.",
                    "Grace Period: Premium payment has a grace period of 30 days."
                ],
                "category": "policy_terms",
                "confidence": 1.0
            },
            
            # Preventive Care examples
            {
                "question": "Is preventive care covered?",
                "positive_context": "Preventive Health Check-up: We will reimburse expenses incurred towards Health Check-up once at the end of a block of every three continuous Policy Years, subject to a maximum of Rs. 5,000 per person.",
                "negative_contexts": [
                    "Exclusions: Vaccination and inoculation are not covered.",
                    "OPD Treatment: Outpatient treatment is not covered under this policy.",
                    "Dental Care: Dental treatment is excluded except for accidental injury."
                ],
                "category": "coverage",
                "confidence": 1.0
            },
            {
                "question": "Are vaccinations covered?",
                "positive_context": "Vaccination and inoculation, except as part of post-bite treatment or unless specifically covered, is an exclusion.",
                "negative_contexts": [
                    "Health Check-up: Preventive health check-up is covered once every three years.",
                    "Emergency Treatment: Emergency medical treatment is covered.",
                    "Hospitalization: In-patient hospitalization is covered."
                ],
                "category": "exclusions",
                "confidence": 1.0
            },
            
            # Moratorium Period examples
            {
                "question": "What is the Moratorium Period?",
                "positive_context": "Moratorium Period: After completion of eight continuous years under the policy, no look back would be applied. This period of eight years is applicable for the sums insured of the first policy and subsequently for any enhancement of sum insured, only on the enhanced limits.",
                "negative_contexts": [
                    "Waiting Period: Pre-existing conditions have a 48-month waiting period.",
                    "Policy Term: The policy is valid for one year.",
                    "Grace Period: Premium payment grace period is 30 days."
                ],
                "category": "policy_terms",
                "confidence": 1.0
            },
            
            # Cumulative Bonus examples
            {
                "question": "What is the Cumulative Bonus?",
                "positive_context": "Cumulative Bonus: Additional 10% of Base Sum Insured on continuous renewal for each claim-free year, subject to a maximum of 100% of Base Sum Insured.",
                "negative_contexts": [
                    "No Claim Bonus: No claim bonus is not applicable under this policy.",
                    "Premium Discount: Premium discounts are not available.",
                    "Loyalty Bonus: Loyalty bonuses are not part of this policy."
                ],
                "category": "benefits",
                "confidence": 1.0
            },
            
            # Maternity Coverage examples
            {
                "question": "Are maternity expenses covered?",
                "positive_context": "Maternity expenses are not covered under this policy. Normal Delivery Rs. 15,000* and Caesarean Delivery Rs. 25,000* are listed under 'Not Covered' with a waiting period of 6 years.",
                "negative_contexts": [
                    "Emergency Medical: Emergency medical expenses are covered.",
                    "Hospitalization: In-patient hospitalization is covered.",
                    "Surgery: Surgical procedures are covered subject to policy terms."
                ],
                "category": "exclusions",
                "confidence": 1.0
            },
            
            # Organ Donor examples
            {
                "question": "Are organ donor expenses covered?",
                "positive_context": "Expenses incurred by an organ donor are not covered under this policy. This includes all costs related to organ donation procedures.",
                "negative_contexts": [
                    "Transplant Recipient: Expenses for the transplant recipient are covered.",
                    "Hospitalization: Regular hospitalization expenses are covered.",
                    "Surgery: Surgical procedures for the insured are covered."
                ],
                "category": "exclusions",
                "confidence": 1.0
            },
            
            # Fraudulent Claims examples
            {
                "question": "What happens if a fraudulent claim is made?",
                "positive_context": "If any claim made by the insured person is in any respect fraudulent, or if any false statement or declaration is made or used in support thereof, all benefits under this policy and the premium paid shall be forfeited.",
                "negative_contexts": [
                    "Genuine Claims: Genuine claims are processed within 30 days.",
                    "Documentation: Proper documentation is required for claims.",
                    "Appeal Process: Claim appeals can be made within 60 days."
                ],
                "category": "claims",
                "confidence": 1.0
            },
            
            # Hospital Definition examples
            {
                "question": "How does this policy define a Hospital?",
                "positive_context": "Hospital means any institution established for inpatient care and day care treatment of illness and/or injuries and which has been registered as a hospital with the local authorities under the Clinical Establishments (Registration and Regulations) Act 2010.",
                "negative_contexts": [
                    "Clinic: A clinic is a healthcare facility for outpatient treatment.",
                    "Pharmacy: A pharmacy dispenses medications and medical supplies.",
                    "Laboratory: A laboratory performs diagnostic tests and analysis."
                ],
                "category": "definitions",
                "confidence": 1.0
            },
            
            # Accident Definition examples
            {
                "question": "What constitutes an Accident according to the policy?",
                "positive_context": "Accident / Accidental means a sudden, unforeseen and involuntary event caused by external, visible and violent means.",
                "negative_contexts": [
                    "Illness: An illness is a medical condition that develops over time.",
                    "Disease: A disease is a pathological condition affecting the body.",
                    "Pre-existing Condition: A pre-existing condition existed before the policy."
                ],
                "category": "definitions",
                "confidence": 1.0
            }
        ]
    
    def create_training_pairs(self) -> List[TrainingPair]:
        """
        Create training pairs from predefined examples and document analysis
        
        Returns:
            List of TrainingPair objects for fine-tuning
        """
        training_pairs = []
        
        # Add predefined examples
        for example in self.training_examples:
            training_pair = TrainingPair(
                question=example["question"],
                positive_context=example["positive_context"],
                negative_contexts=example["negative_contexts"],
                metadata={
                    "category": example["category"],
                    "source": "predefined",
                    "created_at": datetime.now().isoformat()
                },
                confidence_score=example["confidence"]
            )
            training_pairs.append(training_pair)
        
        logger.info(f"Created {len(training_pairs)} training pairs")
        return training_pairs
    
    def save_dataset(self, training_pairs: List[TrainingPair], filename: str = None) -> str:
        """
        Save training pairs to JSON file
        
        Args:
            training_pairs: List of training pairs
            filename: Optional custom filename
            
        Returns:
            Path to saved dataset file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"insurance_fine_tuning_dataset_{timestamp}.json"
        
        filepath = os.path.join(self.dataset_path, filename)
        
        dataset = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_pairs": len(training_pairs),
                "categories": list(set(pair.metadata["category"] for pair in training_pairs)),
                "description": "Fine-tuning dataset for insurance document embeddings"
            },
            "training_pairs": [
                {
                    "question": pair.question,
                    "positive_context": pair.positive_context,
                    "negative_contexts": pair.negative_contexts,
                    "metadata": pair.metadata,
                    "confidence_score": pair.confidence_score
                }
                for pair in training_pairs
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved fine-tuning dataset to: {filepath}")
        return filepath
    
    def load_dataset(self, filepath: str) -> List[TrainingPair]:
        """
        Load training pairs from JSON file
        
        Args:
            filepath: Path to dataset file
            
        Returns:
            List of TrainingPair objects
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        training_pairs = []
        for pair_data in data["training_pairs"]:
            training_pair = TrainingPair(
                question=pair_data["question"],
                positive_context=pair_data["positive_context"],
                negative_contexts=pair_data["negative_contexts"],
                metadata=pair_data["metadata"],
                confidence_score=pair_data["confidence_score"]
            )
            training_pairs.append(training_pair)
        
        logger.info(f"Loaded {len(training_pairs)} training pairs from: {filepath}")
        return training_pairs
    
    def generate_contrastive_pairs(self, training_pairs: List[TrainingPair]) -> List[Tuple[str, str, float]]:
        """
        Generate contrastive learning pairs for fine-tuning
        
        Args:
            training_pairs: List of training pairs
            
        Returns:
            List of (text1, text2, similarity_score) tuples
        """
        contrastive_pairs = []
        
        for pair in training_pairs:
            # Positive pair: question + positive context (similarity = 1.0)
            contrastive_pairs.append((
                pair.question,
                pair.positive_context,
                1.0
            ))
            
            # Negative pairs: question + negative contexts (similarity = 0.0)
            for negative_context in pair.negative_contexts:
                contrastive_pairs.append((
                    pair.question,
                    negative_context,
                    0.0
                ))
        
        logger.info(f"Generated {len(contrastive_pairs)} contrastive pairs")
        return contrastive_pairs
    
    def validate_dataset(self, training_pairs: List[TrainingPair]) -> Dict[str, Any]:
        """
        Validate the quality of the training dataset
        
        Args:
            training_pairs: List of training pairs
            
        Returns:
            Validation results
        """
        total_pairs = len(training_pairs)
        categories = {}
        avg_confidence = 0.0
        
        for pair in training_pairs:
            category = pair.metadata["category"]
            categories[category] = categories.get(category, 0) + 1
            avg_confidence += pair.confidence_score
        
        avg_confidence /= total_pairs if total_pairs > 0 else 1
        
        validation_results = {
            "total_pairs": total_pairs,
            "categories": categories,
            "avg_confidence": avg_confidence,
            "category_distribution": {cat: count/total_pairs for cat, count in categories.items()},
            "is_valid": total_pairs >= 10 and avg_confidence >= 0.8
        }
        
        logger.info(f"Dataset validation: {validation_results}")
        return validation_results 