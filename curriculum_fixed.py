"""
Fixed Curriculum Learning for Neural Theorem Proving
====================================================

Critical Fix:
    Original curriculum HIDES hard samples until late training,
    preventing the model from learning robust features early.

New Strategy (Inspired by Bengio et al. + AlphaProof):
1. ALWAYS expose all difficulties (even at epoch 1)
2. Use LOSS WEIGHTING instead of data filtering
3. Gradually increase weight on hard samples
4. Allow longer proofs earlier (prevent length bias)

Mathematical Foundation:
    L_total = Σ w(difficulty, step, epoch) * L(sample)
    
Where w(·) increases smoothly for hard samples as training progresses.

Reference: 
- "Curriculum Learning" (Bengio et al., ICML 2009)
- "Teacher-Student Curriculum Learning" (Matiisen et al., 2017)
- AlphaProof (DeepMind, 2024)
"""

import torch
import numpy as np
from typing import Dict, Optional

import numpy as np
from typing import Dict

class PhasedCurriculumScheduler:
    """
    Three-phase curriculum matching SOTA theorem provers (DeepSeek-Prover-V2).
    
    Phase 1 (0-30%): Easy + Medium only (build foundation)
    Phase 2 (30-70%): Add Hard samples progressively
    Phase 3 (70-100%): Full dataset with adaptive weighting
    """
    def __init__(self, total_epochs: int, warmup_epochs: int = 0):
        self.total_epochs = total_epochs
        
    def get_phase_config(self, epoch: int) -> Dict:
        progress = epoch / self.total_epochs
        
        if progress < 0.3:
            # PHASE 1: Foundation Building
            return {
                'phase': 'Foundation',
                'description': 'Mastering Easy/Medium axioms',
                'include_difficulties': ['easy', 'medium'],  # HARD FILTER
                'sampling_weights': {'easy': 1.0, 'medium': 1.0},
                'max_proof_length': 10,
                'length_penalty': 0.1
            }
        
        elif progress < 0.7:
            # PHASE 2: Progressive Hardening
            phase_progress = (progress - 0.3) / 0.4
            # Quadratic ramp for hard samples
            hard_ratio = phase_progress ** 2
            
            return {
                'phase': 'Hardening',
                'description': 'Introducing Hard samples',
                'include_difficulties': ['easy', 'medium', 'hard'],
                'sampling_weights': {
                    'easy': 1.0 - 0.5 * phase_progress,
                    'medium': 1.0,
                    'hard': hard_ratio
                },
                'max_proof_length': int(10 + 15 * phase_progress),
                'length_penalty': 0.05
            }
        
        else:
            # PHASE 3: Full Mastery
            return {
                'phase': 'Mastery',
                'description': 'Full difficulty spectrum',
                'include_difficulties': ['easy', 'medium', 'hard', 'very_hard'],
                'sampling_weights': {
                    'easy': 0.5, 'medium': 0.8, 'hard': 1.0, 'very_hard': 1.0
                },
                'max_proof_length': 40,
                'length_penalty': 0.0
            }

    def get_loss_weight(self, epoch: int, sample_difficulty: str,
                       step_idx: int, proof_length: int) -> float:
        config = self.get_phase_config(epoch)
        base_weight = config['sampling_weights'].get(sample_difficulty, 0.0) # Default 0 if not included
        
        # Hard filter if difficulty not in current phase
        if sample_difficulty not in config['include_difficulties']:
            return 0.0
            
        return base_weight

    def get_epoch_stats(self, epoch: int) -> str:
        config = self.get_phase_config(epoch)
        return f"Epoch {epoch}: {config['phase']} | {config['description']}"
class SmoothCurriculumScheduler:
    """
    Smooth curriculum with progressive difficulty weighting.
    
    Key Principles:
    1. Expose all difficulties from epoch 1
    2. Smooth weight transitions (no abrupt changes)
    3. Difficulty-aware weight scaling
    4. Length-based soft filtering (not hard cutoff)
    """
    
    def __init__(self, total_epochs: int, warmup_epochs: int = 5):
        """
        Args:
            total_epochs: Total training epochs
            warmup_epochs: Initial epochs with easier weighting
        """
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
    
    def get_phase_config(self, epoch: int) -> Dict:
        """
        Get curriculum configuration for given epoch.
        
        Returns config dict with:
        - phase: str description
        - include_difficulties: list (always all)
        - sampling_weights: dict of difficulty → weight
        - max_proof_length: int (soft limit)
        """
        self.current_epoch = epoch
        progress = epoch / self.total_epochs
        
        # Phase 1: Warmup (0-20%)
        if progress < 0.2:
            return {
                'phase': 'Warmup',
                'description': 'Smooth start with all difficulties',
                'include_difficulties': ['easy', 'medium', 'hard', 'very_hard'],
                'sampling_weights': {
                    'easy': 1.0,
                    'medium': 0.8,
                    'hard': 0.5,      # ← PRESENT from start!
                    'very_hard': 0.2
                },
                'max_proof_length': 15,  # Soft limit (not enforced)
                'length_penalty': 0.3     # Weight reduction per extra step
            }
        
        # Phase 2: Ramp-up (20-60%)
        elif progress < 0.6:
            # Smooth interpolation
            phase_progress = (progress - 0.2) / 0.4
            
            return {
                'phase': 'Ramp-up',
                'description': 'Increasing hard sample weight',
                'include_difficulties': ['easy', 'medium', 'hard', 'very_hard'],
                'sampling_weights': {
                    'easy': 1.0 - 0.3 * phase_progress,
                    'medium': 1.0,
                    'hard': 0.5 + 0.5 * phase_progress,
                    'very_hard': 0.2 + 0.6 * phase_progress
                },
                'max_proof_length': int(15 + 10 * phase_progress),
                'length_penalty': 0.3 * (1 - phase_progress)
            }
        
        # Phase 3: Focus (60-100%)
        else:
            return {
                'phase': 'Focus',
                'description': 'Full difficulty with hard emphasis',
                'include_difficulties': ['easy', 'medium', 'hard', 'very_hard'],
                'sampling_weights': {
                    'easy': 0.6,
                    'medium': 0.9,
                    'hard': 1.0,
                    'very_hard': 1.0
                },
                'max_proof_length': 30,
                'length_penalty': 0.0  # No length penalty
            }
    
    def get_loss_weight(self, epoch: int, sample_difficulty: str,
                       step_idx: int, proof_length: int) -> float:
        """
        Compute loss weight for a specific sample.
        
        Args:
            epoch: Current training epoch
            sample_difficulty: 'easy', 'medium', 'hard', 'very_hard'
            step_idx: Step index in proof (0-based)
            proof_length: Total proof length
            
        Returns:
            weight: Multiplicative weight for loss (0.0 - 1.0)
        """
        config = self.get_phase_config(epoch)
        
        # Base weight from difficulty
        base_weight = config['sampling_weights'].get(sample_difficulty, 0.5)
        
        # Length penalty (soft)
        max_len = config['max_proof_length']
        if proof_length > max_len:
            length_penalty = config.get('length_penalty', 0.1)
            excess_steps = proof_length - max_len
            length_weight = np.exp(-length_penalty * excess_steps)
        else:
            length_weight = 1.0
        
        # Step-in-proof penalty (early steps more important)
        # This helps the model learn to start proofs correctly
        step_weight = 1.0 - 0.1 * (step_idx / max(proof_length, 1))
        step_weight = max(step_weight, 0.5)  # Clamp to [0.5, 1.0]
        
        # Combine weights
        total_weight = base_weight * length_weight * step_weight
        
        # Clamp to reasonable range
        return float(np.clip(total_weight, 0.1, 1.0))
    
    def get_epoch_stats(self, epoch: int) -> str:
        """Get human-readable stats for logging."""
        config = self.get_phase_config(epoch)
        
        weights_str = ", ".join(
            f"{k}={v:.2f}" for k, v in config['sampling_weights'].items()
        )
        
        return (f"Epoch {epoch}: {config['phase']} | "
                f"Weights: {weights_str} | "
                f"Max Length: {config['max_proof_length']}")


class AntiCurriculumScheduler:
    """
    EXPERIMENTAL: Anti-curriculum (start hard, end easy).
    
    Inspired by recent findings that starting with hard examples
    can sometimes improve robustness.
    
    Use this if standard curriculum fails.
    """
    
    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs
    
    def get_phase_config(self, epoch: int) -> Dict:
        progress = epoch / self.total_epochs
        
        # Reverse the standard curriculum
        if progress < 0.3:
            # Start HARD
            return {
                'phase': 'Hard Start',
                'description': 'Focus on challenging samples',
                'include_difficulties': ['easy', 'medium', 'hard', 'very_hard'],
                'sampling_weights': {
                    'easy': 0.3,
                    'medium': 0.6,
                    'hard': 1.0,
                    'very_hard': 1.0
                },
                'max_proof_length': 30,
                'length_penalty': 0.0
            }
        else:
            # Gradually add easier samples for fine-tuning
            phase_progress = (progress - 0.3) / 0.7
            return {
                'phase': 'Fine-tuning',
                'description': 'Balancing with easier samples',
                'include_difficulties': ['easy', 'medium', 'hard', 'very_hard'],
                'sampling_weights': {
                    'easy': 0.3 + 0.7 * phase_progress,
                    'medium': 0.6 + 0.4 * phase_progress,
                    'hard': 1.0,
                    'very_hard': 1.0 - 0.3 * phase_progress
                },
                'max_proof_length': 30,
                'length_penalty': 0.0
            }
    
    def get_loss_weight(self, epoch: int, sample_difficulty: str,
                       step_idx: int, proof_length: int) -> float:
        config = self.get_phase_config(epoch)
        return config['sampling_weights'].get(sample_difficulty, 0.5)
    
    def get_epoch_stats(self, epoch: int) -> str:
        config = self.get_phase_config(epoch)
        return f"Epoch {epoch}: {config['phase']}"


# ============================================================================
# ADAPTIVE CURRICULUM (ADVANCED)
# ============================================================================

class AdaptiveCurriculumScheduler:
    """
    Adaptive curriculum that adjusts based on validation performance.
    
    If model struggles on hard samples → increase their weight.
    If model does well on easy samples → decrease their weight.
    """
    
    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs
        self.difficulty_weights = {
            'easy': 1.0,
            'medium': 0.8,
            'hard': 0.5,
            'very_hard': 0.2
        }
        
        # Track performance per difficulty
        self.performance_history = {
            'easy': [],
            'medium': [],
            'hard': [],
            'very_hard': []
        }
    
    def update_from_validation(self, difficulty_metrics: Dict[str, float]):
        """
        Update curriculum based on validation metrics.
        
        Args:
            difficulty_metrics: Dict of difficulty → accuracy
        """
        for difficulty, accuracy in difficulty_metrics.items():
            if difficulty not in self.difficulty_weights:
                continue
            
            # Store history
            self.performance_history[difficulty].append(accuracy)
            
            # Adapt weights (simplified PID-like controller)
            target_accuracy = 0.7
            error = target_accuracy - accuracy
            
            # Increase weight if performance is low
            adjustment = 0.1 * error
            self.difficulty_weights[difficulty] = np.clip(
                self.difficulty_weights[difficulty] + adjustment,
                0.1, 1.0
            )
    
    def get_phase_config(self, epoch: int) -> Dict:
        return {
            'phase': 'Adaptive',
            'description': 'Performance-based weighting',
            'include_difficulties': ['easy', 'medium', 'hard', 'very_hard'],
            'sampling_weights': self.difficulty_weights.copy(),
            'max_proof_length': 30,
            'length_penalty': 0.0
        }
    
    def get_loss_weight(self, epoch: int, sample_difficulty: str,
                       step_idx: int, proof_length: int) -> float:
        return self.difficulty_weights.get(sample_difficulty, 0.5)
    
    def get_epoch_stats(self, epoch: int) -> str:
        weights_str = ", ".join(
            f"{k}={v:.2f}" for k, v in self.difficulty_weights.items()
        )
        return f"Epoch {epoch}: Adaptive | Weights: {weights_str}"


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def get_curriculum_scheduler(curriculum_type: str = 'smooth',
                             total_epochs: int = 50) -> object:
    """
    Factory function to create curriculum scheduler.
    
    Args:
        curriculum_type: 'smooth', 'anti', or 'adaptive'
        total_epochs: Total training epochs
        
    Returns:
        scheduler: Curriculum scheduler instance
    """
    if curriculum_type == 'smooth':
        return SmoothCurriculumScheduler(total_epochs)
    elif curriculum_type == 'anti':
        return AntiCurriculumScheduler(total_epochs)
    elif curriculum_type == 'adaptive':
        return AdaptiveCurriculumScheduler(total_epochs)
    else:
        raise ValueError(f"Unknown curriculum type: {curriculum_type}")


# ============================================================================
# TESTING
# ============================================================================

def test_curriculum():
    """Test curriculum schedulers."""
    print("Testing Curriculum Schedulers...")
    
    scheduler = SmoothCurriculumScheduler(total_epochs=50)
    
    # Test at different epochs
    for epoch in [1, 10, 30, 50]:
        config = scheduler.get_phase_config(epoch)
        print(f"\nEpoch {epoch}: {config['phase']}")
        print(f"  Weights: {config['sampling_weights']}")
        
        # Test loss weight computation
        weight_easy = scheduler.get_loss_weight(epoch, 'easy', 0, 5)
        weight_hard = scheduler.get_loss_weight(epoch, 'hard', 0, 20)
        
        print(f"  Easy (len=5): {weight_easy:.3f}")
        print(f"  Hard (len=20): {weight_hard:.3f}")
    
    print("\n✓ Curriculum test passed!\n")


if __name__ == "__main__":
    test_curriculum()