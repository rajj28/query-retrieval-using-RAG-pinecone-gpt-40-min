# app/core/fine_tuning_manager.py
"""
Fine-tuning manager for embedding models
Handles the fine-tuning process to improve domain-specific similarity
"""

import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch

from app.core.fine_tuning_dataset import FineTuningDatasetManager, TrainingPair
from app.config.settings import settings

logger = logging.getLogger(__name__)

class FineTuningManager:
    """
    Manages fine-tuning of embedding models for insurance document similarity
    """
    
    def __init__(self, base_model_name: str = "all-MiniLM-L6-v2"):
        self.base_model_name = base_model_name
        self.dataset_manager = FineTuningDatasetManager()
        self.fine_tuned_model_path = "models/fine_tuned_embeddings"
        os.makedirs(self.fine_tuned_model_path, exist_ok=True)
        
        # Fine-tuning parameters
        self.training_params = {
            "batch_size": 16,
            "epochs": 3,
            "learning_rate": 2e-5,
            "warmup_steps": 100,
            "max_seq_length": 512,
            "evaluation_steps": 50,
            "save_steps": 100
        }
    
    def prepare_training_data(self, training_pairs: List[TrainingPair]) -> List[InputExample]:
        """
        Prepare training data for sentence-transformers fine-tuning
        
        Args:
            training_pairs: List of training pairs
            
        Returns:
            List of InputExample objects for training
        """
        training_examples = []
        
        for pair in training_pairs:
            # Positive examples (similarity = 1.0)
            training_examples.append(InputExample(
                texts=[pair.question, pair.positive_context],
                label=1.0
            ))
            
            # Negative examples (similarity = 0.0)
            for negative_context in pair.negative_contexts:
                training_examples.append(InputExample(
                    texts=[pair.question, negative_context],
                    label=0.0
                ))
        
        logger.info(f"Prepared {len(training_examples)} training examples")
        return training_examples
    
    def create_model(self) -> SentenceTransformer:
        """
        Create or load the base model for fine-tuning
        
        Returns:
            SentenceTransformer model
        """
        try:
            # Try to load existing fine-tuned model
            model_path = os.path.join(self.fine_tuned_model_path, "latest")
            if os.path.exists(model_path):
                logger.info(f"Loading existing fine-tuned model from: {model_path}")
                model = SentenceTransformer(model_path)
            else:
                # Load base model
                logger.info(f"Loading base model: {self.base_model_name}")
                model = SentenceTransformer(self.base_model_name)
            
            return model
            
        except Exception as e:
            logger.error(f"Error creating model: {str(e)}")
            # Fallback to base model
            logger.info("Falling back to base model")
            return SentenceTransformer(self.base_model_name)
    
    async def fine_tune_model(
        self,
        training_pairs: List[TrainingPair],
        validation_pairs: Optional[List[TrainingPair]] = None
    ) -> str:
        """
        Fine-tune the embedding model on insurance-specific data
        
        Args:
            training_pairs: Training data pairs
            validation_pairs: Optional validation data pairs
            
        Returns:
            Path to the fine-tuned model
        """
        try:
            logger.info("Starting fine-tuning process...")
            
            # Prepare training data
            training_examples = self.prepare_training_data(training_pairs)
            
            # Create model
            model = self.create_model()
            
            # Create data loader
            train_dataloader = DataLoader(
                training_examples,
                shuffle=True,
                batch_size=self.training_params["batch_size"]
            )
            
            # Define loss function (contrastive loss)
            train_loss = losses.ContrastiveLoss(model)
            
            # Fine-tune the model
            logger.info(f"Fine-tuning with {len(training_examples)} examples...")
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=self.training_params["epochs"],
                warmup_steps=self.training_params["warmup_steps"],
                show_progress_bar=True,
                optimizer_params={'lr': self.training_params["learning_rate"]}
            )
            
            # Save the fine-tuned model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_save_path = os.path.join(self.fine_tuned_model_path, f"fine_tuned_{timestamp}")
            
            model.save(model_save_path)
            
            # Also save as "latest" for easy access
            latest_path = os.path.join(self.fine_tuned_model_path, "latest")
            model.save(latest_path)
            
            # Save training metadata
            metadata = {
                "base_model": self.base_model_name,
                "fine_tuned_at": datetime.now().isoformat(),
                "training_pairs": len(training_pairs),
                "training_examples": len(training_examples),
                "training_params": self.training_params,
                "model_path": model_save_path
            }
            
            metadata_path = os.path.join(model_save_path, "training_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Fine-tuning completed. Model saved to: {model_save_path}")
            return model_save_path
            
        except Exception as e:
            logger.error(f"Fine-tuning failed: {str(e)}")
            raise e
    
    def evaluate_model(
        self,
        model: SentenceTransformer,
        test_pairs: List[TrainingPair]
    ) -> Dict[str, Any]:
        """
        Evaluate the fine-tuned model on test data
        
        Args:
            model: Fine-tuned model
            test_pairs: Test data pairs
            
        Returns:
            Evaluation results
        """
        try:
            logger.info("Evaluating fine-tuned model...")
            
            total_pairs = len(test_pairs)
            correct_predictions = 0
            total_similarity = 0.0
            
            for pair in test_pairs:
                # Calculate similarity between question and positive context
                question_embedding = model.encode(pair.question)
                positive_embedding = model.encode(pair.positive_context)
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(question_embedding, positive_embedding)
                total_similarity += similarity
                
                # Check if similarity is above threshold (0.5)
                if similarity > 0.5:
                    correct_predictions += 1
            
            accuracy = correct_predictions / total_pairs if total_pairs > 0 else 0
            avg_similarity = total_similarity / total_pairs if total_pairs > 0 else 0
            
            evaluation_results = {
                "accuracy": accuracy,
                "avg_similarity": avg_similarity,
                "correct_predictions": correct_predictions,
                "total_pairs": total_pairs,
                "threshold": 0.5
            }
            
            logger.info(f"Model evaluation results: {evaluation_results}")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            return {"error": str(e)}
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        import numpy as np
        
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def compare_models(
        self,
        base_model: SentenceTransformer,
        fine_tuned_model: SentenceTransformer,
        test_pairs: List[TrainingPair]
    ) -> Dict[str, Any]:
        """
        Compare base model vs fine-tuned model performance
        
        Args:
            base_model: Original base model
            fine_tuned_model: Fine-tuned model
            test_pairs: Test data pairs
            
        Returns:
            Comparison results
        """
        try:
            logger.info("Comparing base model vs fine-tuned model...")
            
            base_results = self.evaluate_model(base_model, test_pairs)
            fine_tuned_results = self.evaluate_model(fine_tuned_model, test_pairs)
            
            comparison = {
                "base_model": base_results,
                "fine_tuned_model": fine_tuned_results,
                "improvement": {
                    "accuracy_improvement": fine_tuned_results["accuracy"] - base_results["accuracy"],
                    "similarity_improvement": fine_tuned_results["avg_similarity"] - base_results["avg_similarity"]
                }
            }
            
            logger.info(f"Model comparison results: {comparison}")
            return comparison
            
        except Exception as e:
            logger.error(f"Model comparison failed: {str(e)}")
            return {"error": str(e)}
    
    def get_fine_tuned_model_path(self) -> Optional[str]:
        """
        Get the path to the latest fine-tuned model
        
        Returns:
            Path to fine-tuned model or None if not found
        """
        latest_path = os.path.join(self.fine_tuned_model_path, "latest")
        if os.path.exists(latest_path):
            return latest_path
        return None
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List all available fine-tuned models
        
        Returns:
            List of model information
        """
        models = []
        
        if os.path.exists(self.fine_tuned_model_path):
            for item in os.listdir(self.fine_tuned_model_path):
                item_path = os.path.join(self.fine_tuned_model_path, item)
                if os.path.isdir(item_path):
                    metadata_path = os.path.join(item_path, "training_metadata.json")
                    
                    model_info = {
                        "name": item,
                        "path": item_path,
                        "created_at": "Unknown"
                    }
                    
                    if os.path.exists(metadata_path):
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                                model_info.update(metadata)
                        except Exception as e:
                            logger.warning(f"Could not load metadata for {item}: {str(e)}")
                    
                    models.append(model_info)
        
        return models 