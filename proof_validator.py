"""
Proof Validation and Evaluation Metrics
"""

import torch
from typing import Dict, List, Tuple, Optional
from proof_state import ProofState, Rule
from dataclasses import dataclass


@dataclass
class ProofMetrics:
    """Metrics for a single proof."""
    is_valid: bool
    is_complete: bool
    length: int
    depth: int
    num_goals_closed: int
    proof_confidence: float
    proof_trace: List[str]


class ProofValidator:
    """
    Validates proof correctness and properties.
    """
    
    def __init__(self):
        pass
    
    def validate_step(self, rule: Rule, premises: List[int],
                     conclusion: int, facts: Dict[int, 'Fact']) -> Tuple[bool, str]:
        """
        Validate single proof step.
        
        Args:
            rule: Rule being applied
            premises: Fact indices used as premises
            conclusion: Conclusion fact index
            facts: All available facts
        
        Returns:
            (is_valid, error_message)
        """
        # Check all premises exist
        for premise_id in rule.premises:
            if premise_id not in facts:
                return False, f"Premise {premise_id} not in facts"
        
        # Check conclusion matches rule head
        if rule.conclusion != conclusion:
            return False, f"Conclusion {conclusion} doesn't match rule head {rule.conclusion}"
        
        # Check rule applicability
        for premise_id in rule.premises:
            if premise_id not in facts:
                return False, f"Missing premise {premise_id}"
        
        return True, "Valid"
    
    def validate_proof(self, state: ProofState) -> Tuple[bool, str]:
        """
        Validate complete proof sequence.
        
        Args:
            state: Final proof state
        
        Returns:
            (is_valid, error_message)
        """
        # Check history
        for step_idx, (rule_id, step_num, premises, conclusion) in enumerate(state.history):
            if rule_id not in state.rules:
                return False, f"Step {rule_id} not found"
            
            rule = state.rules[rule_id]
            valid, msg = self.validate_step(rule, premises, conclusion, state.facts)
            
            if not valid:
                return False, f"Step {step_idx}: {msg}"
        
        return True, "Proof valid"
    
    def compute_metrics(self, state: ProofState) -> ProofMetrics:
        """
        Compute comprehensive metrics for proof.
        
        Args:
            state: Final proof state
        
        Returns:
            ProofMetrics object
        """
        is_valid, _ = self.validate_proof(state)
        is_complete = state.goal_satisfied()
        
        # Confidence: average rule confidence in proof
        confidences = []
        for rule_id, _, _, _ in state.history:
            if rule_id in state.rules:
                confidences.append(state.rules[rule_id].confidence)
        
        proof_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Create proof trace
        proof_trace = []
        for rule_id, step, premises, conclusion in state.history:
            rule_name = state.rules[rule_id].name if rule_id in state.rules else f"rule_{rule_id}"
            proof_trace.append(f"Step {step}: {rule_name}")
        
        return ProofMetrics(
            is_valid=is_valid,
            is_complete=is_complete,
            length=len(state.history),
            depth=state.depth,
            num_goals_closed=len(state.closed_goals),
            proof_confidence=proof_confidence,
            proof_trace=proof_trace
        )


class EvaluationMetrics:
    """
    Tracks and computes evaluation metrics.
    """
    
    def __init__(self):
        self.validator = ProofValidator()
        self.proof_metrics = []
        self.correctness = []
        self.completeness = []
        self.efficiency = []
    
    def evaluate_proof(self, state: ProofState) -> ProofMetrics:
        """Evaluate single proof."""
        metrics = self.validator.compute_metrics(state)
        self.proof_metrics.append(metrics)
        
        # Track metrics
        self.correctness.append(1.0 if metrics.is_valid else 0.0)
        self.completeness.append(1.0 if metrics.is_complete else 0.0)
        self.efficiency.append(1.0 / (1.0 + metrics.length))  # Shorter = more efficient
        
        return metrics
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics."""
        if not self.proof_metrics:
            return {}
        
        return {
            'correctness': sum(self.correctness) / len(self.correctness),
            'completeness': sum(self.completeness) / len(self.completeness),
            'efficiency': sum(self.efficiency) / len(self.efficiency),
            'avg_proof_length': sum(m.length for m in self.proof_metrics) / len(self.proof_metrics),
            'avg_depth': sum(m.depth for m in self.proof_metrics) / len(self.proof_metrics),
            'total_proofs': len(self.proof_metrics)
        }
    
    def reset(self):
        """Reset metrics."""
        self.proof_metrics = []
        self.correctness = []
        self.completeness = []
        self.efficiency = []


class AttentionVisualizer:
    """
    Visualizes attention weights for interpretability.
    """
    
    def __init__(self):
        pass
    
    def analyze_attention(self, attention_weights: torch.Tensor) -> Dict[str, float]:
        """
        Analyze attention patterns.
        
        Args:
            attention_weights: [num_heads, seq_len, seq_len]
        
        Returns:
            Statistics about attention patterns
        """
        # Entropy of attention distribution
        attn_probs = torch.softmax(attention_weights, dim=-1)
        entropy = -(attn_probs * torch.log(attn_probs + 1e-8)).sum(dim=-1).mean()
        
        # Sparsity: how many non-zero weights
        sparsity = (attention_weights > 1e-3).float().mean()
        
        # Concentration: how concentrated on few positions
        max_attn = attn_probs.max(dim=-1)[0].mean()
        
        return {
            'entropy': entropy.item(),
            'sparsity': sparsity.item(),
            'concentration': max_attn.item()
        }
    
    def get_attended_positions(self, attention_weights: torch.Tensor,
                              query_pos: int) -> List[Tuple[int, float]]:
        """
        Get positions attended to from query position.
        
        Args:
            attention_weights: [num_heads, seq_len, seq_len]
            query_pos: Query position
        
        Returns:
            List of (position, attention_weight) tuples
        """
        # Average over heads
        avg_attn = attention_weights.mean(dim=0)
        query_attn = avg_attn[query_pos]
        
        # Get top positions
        top_positions = torch.argsort(query_attn, descending=True)[:5]
        
        return [(int(pos.item()), float(query_attn[pos].item())) 
                for pos in top_positions]


class CurriculumAnalyzer:
    """
    Analyzes curriculum learning progress.
    """
    
    def __init__(self):
        self.difficulty_performance = {}
        self.epoch_stats = []
    
    def track_performance(self, difficulty_level: str, performance: float, epoch: int):
        """Track performance on difficulty level."""
        if difficulty_level not in self.difficulty_performance:
            self.difficulty_performance[difficulty_level] = []
        
        self.difficulty_performance[difficulty_level].append({
            'epoch': epoch,
            'performance': performance
        })
    
    def get_difficulty_stats(self, difficulty_level: str) -> Dict[str, float]:
        """Get statistics for difficulty level."""
        if difficulty_level not in self.difficulty_performance:
            return {}
        
        perfs = [p['performance'] for p in self.difficulty_performance[difficulty_level]]
        
        return {
            'mean_performance': sum(perfs) / len(perfs),
            'max_performance': max(perfs),
            'min_performance': min(perfs),
            'trend': perfs[-1] - perfs[0] if len(perfs) > 1 else 0.0
        }
    
    def is_curriculum_progressing(self) -> bool:
        """Check if curriculum is making progress."""
        if not self.difficulty_performance:
            return False
        
        # Check if recent performance is improving
        for level, data in self.difficulty_performance.items():
            if len(data) >= 5:
                recent = [d['performance'] for d in data[-5:]]
                old = [d['performance'] for d in data[-10:-5]] if len(data) >= 10 else recent
                
                if sum(recent) / len(recent) > sum(old) / len(old):
                    return True
        
        return False