"""
Curriculum Learning Implementation
Implements Issue #9: Curriculum learning for gradual difficulty progression
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict
import random
from pathlib import Path


class DifficultyEstimator:
    """Estimates difficulty of proof instances."""
    
    def __init__(self):
        self.difficulty_factors = {
            'num_rules': 0.3,
            'max_depth': 0.4,
            'num_facts': 0.2,
            'proof_length': 0.1
        }
    
    def estimate_difficulty(self, instance: Dict) -> float:
        """
        Estimate difficulty score [0, 1] for an instance.
        Higher score = more difficult.
        """
        # Extract features
        num_rules = len(instance.get('rules', []))
        max_depth = instance.get('max_depth', 1)
        num_facts = len(instance.get('facts', []))
        proof_length = len(instance.get('steps', []))
        
        # Normalize features (assuming reasonable ranges)
        norm_num_rules = min(num_rules / 100.0, 1.0)
        norm_max_depth = min(max_depth / 30.0, 1.0)
        norm_num_facts = min(num_facts / 200.0, 1.0)
        norm_proof_length = min(proof_length / 50.0, 1.0)
        
        # Weighted combination
        difficulty = (
            self.difficulty_factors['num_rules'] * norm_num_rules +
            self.difficulty_factors['max_depth'] * norm_max_depth +
            self.difficulty_factors['num_facts'] * norm_num_facts +
            self.difficulty_factors['proof_length'] * norm_proof_length
        )
        
        return min(difficulty, 1.0)


class CurriculumScheduler:
    """
    SOTA curriculum learning scheduler following MINIMO approach.
    
    Key features:
    - Temperature annealing for difficulty sampling
    - Adaptive difficulty progression
    - Performance-based curriculum adjustment
    """
    
    def __init__(self, 
                 datasets_by_difficulty: Dict[str, List[str]],
                 start_temperature: float = 2.0,
                 min_temperature: float = 0.5,
                 annealing_rate: float = 0.95,
                 performance_threshold: float = 0.8):
        self.datasets = datasets_by_difficulty
        self.temperature = start_temperature
        self.min_temperature = min_temperature
        self.annealing_rate = annealing_rate
        self.performance_threshold = performance_threshold
        
        # Difficulty levels (ordered from easy to hard)
        self.difficulty_levels = ['easy', 'medium', 'hard', 'very_hard', 'extreme_hard']
        
        # Performance tracking
        self.performance_history = defaultdict(list)
        self.current_max_difficulty = 0.0
        
        # Difficulty estimator
        self.difficulty_estimator = DifficultyEstimator()
        
        # Initialize difficulty weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize difficulty sampling weights."""
        self.difficulty_weights = {}
        for i, level in enumerate(self.difficulty_levels):
            # Exponential decay: easier levels have higher initial weights
            self.difficulty_weights[level] = np.exp(-i / self.temperature)
    
    def update_temperature(self, epoch: int, total_epochs: int):
        """Update temperature based on training progress."""
        # Linear annealing
        progress = epoch / total_epochs
        self.temperature = max(
            self.min_temperature,
            self.temperature * (self.annealing_rate ** progress)
        )
        
        # Update weights
        self._update_difficulty_weights()
    
    def _update_difficulty_weights(self):
        """Update difficulty sampling weights based on current temperature."""
        for i, level in enumerate(self.difficulty_levels):
            # Exponential decay with current temperature
            self.difficulty_weights[level] = np.exp(-i / self.temperature)
    
    def update_performance(self, difficulty_level: str, performance: float):
        """Update performance tracking for a difficulty level."""
        self.performance_history[difficulty_level].append(performance)
        
        # Keep only recent performance (last 10 epochs)
        if len(self.performance_history[difficulty_level]) > 10:
            self.performance_history[difficulty_level] = self.performance_history[difficulty_level][-10:]
    
    def get_difficulty_level(self) -> str:
        """Sample difficulty level based on current curriculum."""
        # Normalize weights
        total_weight = sum(self.difficulty_weights.values())
        normalized_weights = {
            level: weight / total_weight 
            for level, weight in self.difficulty_weights.items()
        }
        
        # Sample difficulty level
        levels = list(normalized_weights.keys())
        weights = list(normalized_weights.values())
        
        return np.random.choice(levels, p=weights)
    
    def should_advance_curriculum(self) -> bool:
        """Check if curriculum should advance to harder problems."""
        # Check if current level performance is good enough
        for level in self.difficulty_levels:
            if level in self.performance_history:
                recent_performance = np.mean(self.performance_history[level][-5:])
                if recent_performance < self.performance_threshold:
                    return False
        
        return True
    
    def get_adaptive_weights(self) -> Dict[str, float]:
        """Get adaptive weights based on performance."""
        adaptive_weights = {}
        
        for level in self.difficulty_levels:
            if level in self.performance_history and self.performance_history[level]:
                # Recent performance
                recent_perf = np.mean(self.performance_history[level][-3:])
                
                # Adjust weight based on performance
                if recent_perf < 0.5:  # Poor performance
                    adaptive_weights[level] = self.difficulty_weights[level] * 1.5  # Increase weight
                elif recent_perf > 0.9:  # Excellent performance
                    adaptive_weights[level] = self.difficulty_weights[level] * 0.5  # Decrease weight
                else:
                    adaptive_weights[level] = self.difficulty_weights[level]
            else:
                adaptive_weights[level] = self.difficulty_weights[level]
        
        return adaptive_weights


class CurriculumDataLoader:
    """
    DataLoader that implements curriculum learning.
    
    Features:
    - Difficulty-based sampling
    - Performance-adaptive curriculum
    - Smooth difficulty progression
    """
    
    def __init__(self, 
                 datasets_by_difficulty: Dict[str, List[str]],
                 batch_size: int = 32,
                 start_temperature: float = 2.0,
                 performance_threshold: float = 0.8):
        self.datasets = datasets_by_difficulty
        self.batch_size = batch_size
        
        # Curriculum scheduler
        self.scheduler = CurriculumScheduler(
            datasets_by_difficulty,
            start_temperature,
            performance_threshold=performance_threshold
        )
        
        # Current batch
        self.current_batch = []
        self.batch_idx = 0
    
    def get_batch(self, epoch: int, total_epochs: int) -> List[str]:
        """Get next batch following curriculum."""
        # Update curriculum
        self.scheduler.update_temperature(epoch, total_epochs)
        
        # Sample difficulty level
        difficulty_level = self.scheduler.get_difficulty_level()
        
        # Get files from selected difficulty
        if difficulty_level in self.datasets:
            available_files = self.datasets[difficulty_level]
        else:
            # Fallback to easy if level not available
            available_files = self.datasets.get('easy', [])
        
        # Sample batch
        if len(available_files) >= self.batch_size:
            batch_files = random.sample(available_files, self.batch_size)
        else:
            # If not enough files, sample with replacement
            batch_files = random.choices(available_files, k=self.batch_size)
        
        return batch_files
    
    def update_performance(self, difficulty_level: str, performance: float):
        """Update performance for curriculum adjustment."""
        self.scheduler.update_performance(difficulty_level, performance)
    
    def get_curriculum_info(self) -> Dict[str, float]:
        """Get current curriculum information."""
        return {
            'temperature': self.scheduler.temperature,
            'difficulty_weights': self.scheduler.difficulty_weights,
            'performance_history': dict(self.scheduler.performance_history)
        }


class AdaptiveCurriculumScheduler:
    """
    Advanced curriculum scheduler with adaptive difficulty progression.
    
    Features:
    - Performance-based difficulty adjustment
    - Multi-objective curriculum (accuracy + efficiency)
    - Dynamic difficulty bounds
    """
    
    def __init__(self, 
                 initial_difficulty_range: Tuple[float, float] = (0.0, 0.5),
                 performance_target: float = 0.85,
                 adaptation_rate: float = 0.1):
        self.difficulty_range = initial_difficulty_range
        self.performance_target = performance_target
        self.adaptation_rate = adaptation_rate
        
        # Performance tracking
        self.performance_by_difficulty = defaultdict(list)
        self.difficulty_estimator = DifficultyEstimator()
        
        # Current curriculum state
        self.current_min_difficulty = initial_difficulty_range[0]
        self.current_max_difficulty = initial_difficulty_range[1]
    
    def estimate_instance_difficulty(self, instance: Dict) -> float:
        """Estimate difficulty of a specific instance."""
        return self.difficulty_estimator.estimate_difficulty(instance)
    
    def should_include_instance(self, instance: Dict) -> bool:
        """Check if instance should be included in current curriculum."""
        difficulty = self.estimate_instance_difficulty(instance)
        return self.current_min_difficulty <= difficulty <= self.current_max_difficulty
    
    def update_curriculum(self, performance_by_difficulty: Dict[float, float]):
        """Update curriculum based on performance."""
        # Update performance tracking
        for difficulty, performance in performance_by_difficulty.items():
            self.performance_by_difficulty[difficulty].append(performance)
        
        # Adjust difficulty range based on performance
        if self.performance_by_difficulty:
            # Find difficulty level with target performance
            target_difficulties = []
            for difficulty, performances in self.performance_by_difficulty.items():
                if performances and np.mean(performances[-5:]) >= self.performance_target:
                    target_difficulties.append(difficulty)
            
            if target_difficulties:
                # Expand upper bound
                new_max = min(1.0, max(target_difficulties) + 0.1)
                self.current_max_difficulty = (
                    (1 - self.adaptation_rate) * self.current_max_difficulty +
                    self.adaptation_rate * new_max
                )
            
            # Contract lower bound if performance is too high
            if self.performance_by_difficulty.get(self.current_min_difficulty, [0])[-1] > 0.95:
                self.current_min_difficulty = min(1.0, self.current_min_difficulty + 0.05)
    
    def get_difficulty_weights(self, instances: List[Dict]) -> np.ndarray:
        """Get sampling weights for instances based on curriculum."""
        weights = []
        
        for instance in instances:
            difficulty = self.estimate_instance_difficulty(instance)
            
            if self.current_min_difficulty <= difficulty <= self.current_max_difficulty:
                # Weight by inverse distance from target difficulty
                target_difficulty = (self.current_min_difficulty + self.current_max_difficulty) / 2
                distance = abs(difficulty - target_difficulty)
                weight = np.exp(-distance / 0.2)  # Exponential decay
            else:
                weight = 0.0  # Outside curriculum range
            
            weights.append(weight)
        
        # Normalize weights
        weights = np.array(weights)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(weights)) / len(weights)
        
        return weights


def create_curriculum_loader(datasets_by_difficulty: Dict[str, List[str]], 
                           batch_size: int = 32) -> CurriculumDataLoader:
    """Create curriculum data loader."""
    return CurriculumDataLoader(datasets_by_difficulty, batch_size)


def group_instances_by_difficulty(instances: List[Dict]) -> Dict[str, List[Dict]]:
    """Group instances by difficulty level."""
    difficulty_estimator = DifficultyEstimator()
    grouped = defaultdict(list)
    
    for instance in instances:
        difficulty = difficulty_estimator.estimate_difficulty(instance)
        
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
        
        grouped[level].append(instance)
    
    return dict(grouped)
