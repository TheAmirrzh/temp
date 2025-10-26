"""
Fixed Data Loading with Proper Curriculum Integration
Connects curriculum scheduling to actual training data flow
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
import random
import numpy as np
from proof_state import ProofState


class ProofInstanceDataset(Dataset):
    """
    Dataset of proof instances with proper tensor conversion.
    """
    
    def __init__(self, instances: List[Dict]):
        """
        Args:
            instances: List of proof instance dictionaries
        """
        self.instances = instances
        self._validate_instances()
    
    def _validate_instances(self):
        """Validate that all instances have required fields."""
        required_fields = ['facts', 'rules', 'goal']
        for i, instance in enumerate(self.instances):
            for field in required_fields:
                if field not in instance:
                    raise ValueError(f"Instance {i} missing field: {field}")
    
    def __len__(self) -> int:
        return len(self.instances)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Convert instance to tensor batch."""
        instance = self.instances[idx]
        
        # Create proof state
        state = ProofState(
            facts=instance['facts'],
            rules=instance['rules'],
            goal=instance['goal'],
            max_depth=instance.get('max_depth', 50),
            max_steps=instance.get('max_steps', 100)
        )
        
        # Convert to tensors
        batch = {
            'proof_state': state,  # Keep state object
            'num_facts': len(state.facts),
            'num_rules': len(state.rules),
            'difficulty': instance.get('difficulty', 0.5),
            'difficulty_level': instance.get('difficulty_level', 'medium'),
            'instance_id': idx
        }
        
        return batch


class CurriculumScheduler:
    """
    SOTA curriculum learning scheduler with performance tracking.
    """
    
    def __init__(self,
                 start_temperature: float = 2.0,
                 min_temperature: float = 0.5,
                 annealing_rate: float = 0.95,
                 performance_threshold: float = 0.8):
        self.temperature = start_temperature
        self.min_temperature = min_temperature
        self.annealing_rate = annealing_rate
        self.performance_threshold = performance_threshold
        
        # Difficulty levels
        self.difficulty_levels = ['easy', 'medium', 'hard', 'very_hard', 'extreme_hard']
        self.difficulty_ranges = {
            'easy': (0.0, 0.2),
            'medium': (0.2, 0.4),
            'hard': (0.4, 0.6),
            'very_hard': (0.6, 0.8),
            'extreme_hard': (0.8, 1.0)
        }
        
        # Performance tracking
        self.performance_history = {level: [] for level in self.difficulty_levels}
        
        # Current curriculum state
        self.current_min_difficulty = 0.0
        self.current_max_difficulty = 0.3
        self.epoch = 0
    
    def update_temperature(self, epoch: int, total_epochs: int):
        """Update temperature during training."""
        progress = epoch / max(total_epochs, 1)
        
        # Exponential annealing
        self.temperature = max(
            self.min_temperature,
            self.temperature * (self.annealing_rate ** progress)
        )
        
        self.epoch = epoch
    
    def update_performance(self, difficulty_level: str, performance: float):
        """Track performance on specific difficulty."""
        if difficulty_level in self.performance_history:
            self.performance_history[difficulty_level].append(performance)
            
            # Keep last 10 evaluations
            if len(self.performance_history[difficulty_level]) > 10:
                self.performance_history[difficulty_level] = \
                    self.performance_history[difficulty_level][-10:]
    
    def should_advance_curriculum(self) -> bool:
        """Check if curriculum should advance to harder problems."""
        # Check if all current levels have good performance
        for level in self.difficulty_levels:
            if self.performance_history[level]:
                recent_perf = np.mean(self.performance_history[level][-5:])
                
                # If performance too low, don't advance
                if recent_perf < self.performance_threshold:
                    return False
        
        return True
    
    def advance_curriculum(self):
        """Advance to harder problems."""
        min_diff, max_diff = self.difficulty_ranges['medium']
        self.current_max_difficulty = min(1.0, self.current_max_difficulty + 0.1)
    
    def get_difficulty_bounds(self) -> Tuple[float, float]:
        """Get current difficulty bounds for sampling."""
        return self.current_min_difficulty, self.current_max_difficulty
    
    def get_curriculum_info(self) -> Dict:
        """Get current curriculum state for logging."""
        return {
            'temperature': self.temperature,
            'min_difficulty': self.current_min_difficulty,
            'max_difficulty': self.current_max_difficulty,
            'epoch': self.epoch,
            'performance_history': self.performance_history
        }


class CurriculumDataLoader:
    """
    Data loader with curriculum learning integration.
    Actually controls difficulty progression.
    """
    
    def __init__(self,
                 instances: List[Dict],
                 batch_size: int = 32,
                 start_temperature: float = 2.0):
        """
        Args:
            instances: List of proof instances
            batch_size: Batch size
            start_temperature: Initial curriculum temperature
        """
        self.instances = instances
        self.batch_size = batch_size
        
        # Create dataset
        self.dataset = ProofInstanceDataset(instances)
        
        # Initialize curriculum scheduler
        self.scheduler = CurriculumScheduler(start_temperature=start_temperature)
        
        # Group instances by difficulty
        self.instances_by_difficulty = self._group_by_difficulty()
        
        # Current batch tracking
        self.current_epoch = 0
        self.total_epochs = 100
    
    def _group_by_difficulty(self) -> Dict[str, List[int]]:
        """Group instance indices by difficulty level."""
        grouped = {level: [] for level in self.scheduler.difficulty_levels}
        
        for idx, instance in enumerate(self.instances):
            difficulty = instance.get('difficulty', 0.5)
            
            # Assign to level
            if difficulty < 0.2:
                level = 'easy'
            elif difficulty < 0.4:
                level = 'medium'
            elif difficulty < 0.6:
                level = 'hard'
            elif difficulty < 0.8:
                level = 'very_hard'
            else:
                level = 'extreme_hard'
            
            grouped[level].append(idx)
        
        return grouped
    
    def get_batch(self, epoch: int, total_epochs: int) -> List[Dict]:
        """Get next batch following curriculum with debugging."""
        
        # Update curriculum
        self.scheduler.update_temperature(epoch, total_epochs)
        self.current_epoch = epoch
        self.total_epochs = total_epochs
        
        # Get difficulty bounds
        min_diff, max_diff = self.scheduler.get_difficulty_bounds()
        
        # Filter instances within difficulty range
        valid_indices = []
        for idx, instance in enumerate(self.instances):
            difficulty = instance.get('difficulty', 0.5)
            if min_diff <= difficulty <= max_diff:
                valid_indices.append(idx)
        
        # Fallback to all if none valid
        if not valid_indices:
            valid_indices = list(range(len(self.instances)))
        
        # Ensure batch size doesn't exceed valid instances
        batch_size = min(self.batch_size, len(valid_indices))
        
        if batch_size == 0:
            print(f"[WARNING] get_batch: batch_size is 0!")
            return []
        
        # Sample batch
        batch_indices = random.sample(valid_indices, batch_size)
        
        # Return actual data
        batch = [self.dataset[idx] for idx in batch_indices]
        
        return batch  # ✅ Returns List[Dict] with 'proof_state' key

    def update_performance(self, difficulty_level: str, performance: float):
        """Update performance for curriculum adjustment."""
        self.scheduler.update_performance(difficulty_level, performance)
        
        # Check if should advance
        if self.scheduler.should_advance_curriculum():
            self.scheduler.advance_curriculum()
    
    def get_curriculum_info(self) -> Dict:
        """Get current curriculum info for logging."""
        return self.scheduler.get_curriculum_info()


class DifficultyEstimator:
    """
    Estimates proof instance difficulty.
    """
    
    def __init__(self):
        self.weights = {
            'num_rules': 0.3,
            'num_facts': 0.2,
            'goal_depth': 0.3,
            'complexity': 0.2
        }
    
    def estimate(self, instance: Dict) -> float:
        """
        Estimate difficulty of instance [0, 1].
        
        Args:
            instance: Proof instance dictionary
        
        Returns:
            Difficulty score
        """
        num_rules = len(instance.get('rules', []))
        num_facts = len(instance.get('facts', []))
        goal_depth = instance.get('max_depth', 1)
        
        # Normalize features
        norm_rules = min(num_rules / 100.0, 1.0)
        norm_facts = min(num_facts / 200.0, 1.0)
        norm_depth = min(goal_depth / 50.0, 1.0)
        
        # Complexity: interaction between factors
        complexity = (num_rules * num_facts) / max(num_rules + num_facts, 1)
        norm_complexity = min(complexity / 50.0, 1.0)
        
        # Weighted sum
        difficulty = (
            self.weights['num_rules'] * norm_rules +
            self.weights['num_facts'] * norm_facts +
            self.weights['goal_depth'] * norm_depth +
            self.weights['complexity'] * norm_complexity
        )
        
        return min(difficulty, 1.0)


def create_curriculum_dataloader(instances: List[Dict],
                                batch_size: int = 32,
                                start_temperature: float = 2.0) -> CurriculumDataLoader:
    """Factory function for curriculum dataloader."""
    return CurriculumDataLoader(instances, batch_size, start_temperature)


def estimate_instance_difficulties(instances: List[Dict]) -> List[Dict]:
    """
    Add difficulty scores to instances.
    
    Args:
        instances: List of proof instances
    
    Returns:
        Modified instances with 'difficulty' field
    """
    estimator = DifficultyEstimator()
    
    for instance in instances:
        instance['difficulty'] = estimator.estimate(instance)
    
    return instances