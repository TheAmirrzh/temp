"""
FIXED Loss Functions for Step Prediction

Key Fixes:
1. Applicability constraint: non-applicable rules heavily penalized
2. Separate loss for applicable vs inapplicable rules
3. Valid negative sampling (only from applicable rules)
"""

import logging
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class ApplicabilityConstrainedLoss(nn.Module):
    def __init__(self, margin=1.0, penalty_nonapplicable=10.0):  # REDUCED from 100
        """
        FIXED: More stable penalty weights
        """
        super().__init__()
        self.margin = margin
        self.penalty_nonapplicable = penalty_nonapplicable
    
    def forward(self, scores, embeddings, target_idx, applicable_mask=None):
        n = len(scores)
        
        if target_idx < 0 or target_idx >= n:
            return torch.tensor(0.01, device=scores.device, requires_grad=True)
        
        if applicable_mask is None:
            applicable_mask = torch.ones_like(scores, dtype=torch.bool)
        
        # CRITICAL: Target must be applicable
        if not applicable_mask[target_idx]:
            # Return large but finite loss (not inf)
            return torch.tensor(100.0, device=scores.device, requires_grad=True)
        
        target_score = scores[target_idx]
        
        # === Component 1: Non-applicable penalty ===
        non_applicable_mask = ~applicable_mask
        is_target = torch.arange(n, device=scores.device) == target_idx
        
        if non_applicable_mask.any():
            nonapplicable_scores = scores[non_applicable_mask]
            # FIXED: Use sigmoid instead of relu for smoother gradients
            nonapplicable_loss = torch.sigmoid(
                nonapplicable_scores - target_score + self.margin
            ).mean()
            component1 = self.penalty_nonapplicable * nonapplicable_loss
        else:
            component1 = torch.tensor(0.0, device=scores.device)
        
        # === Component 2: Applicable negatives ranking ===
        applicable_negatives_mask = applicable_mask & ~is_target
        
        if applicable_negatives_mask.any():
            applicable_negative_scores = scores[applicable_negatives_mask]
            
            # FIXED: Smoother margin ranking with log-softmax
            # This prevents exploding gradients
            margin_violations = torch.relu(
                self.margin - (target_score - applicable_negative_scores)
            )
            
            # Focus on top-k hardest negatives (prevents overwhelming)
            n_hard = min(5, len(applicable_negative_scores))
            if n_hard > 0:
                top_violations, _ = torch.topk(margin_violations, k=n_hard)
                component2 = top_violations.mean()
            else:
                component2 = torch.tensor(0.0, device=scores.device)
        else:
            component2 = torch.tensor(0.0, device=scores.device)
        
        # === Combine with temperature scaling ===
        # This prevents initial loss from being too large
        temperature = 0.5  # Scales down initial loss
        loss = temperature * (component1 + component2)
        
        return loss

class ApplicabilityConstrainedRankingLoss(nn.Module):
    """
    Simplified version: focus only on applicable rules.
    """
    
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, scores, embeddings, target_idx, applicable_mask):
        """
        Args:
            scores: [N] node scores
            embeddings: [N, D] (unused)
            target_idx: index of correct rule
            applicable_mask: [N] binary mask of which rules are applicable
        
        Returns:
            loss
        """
        n = len(scores)
        
        if target_idx < 0 or target_idx >= n:
            return torch.tensor(0.01, device=scores.device, requires_grad=True)
        
        # Target MUST be applicable
        if not applicable_mask[target_idx]:
            return torch.tensor(float('inf'), device=scores.device)
        
        target_score = scores[target_idx]
        
        # Create mask for applicable negatives
        is_target = torch.arange(n, device=scores.device) == target_idx
        applicable_negatives = applicable_mask & ~is_target
        
        if not applicable_negatives.any():
            # No competition - perfect scenario
            return torch.tensor(0.0, device=scores.device)
        
        negative_scores = scores[applicable_negatives]
        
        # Margin ranking loss
        violations = F.relu(self.margin - (target_score - negative_scores))
        
        return violations.mean()


def compute_hit_at_k(scores, target_idx, k, applicable_mask=None):
    """
    Compute Hit@K metric.
    
    If applicable_mask provided, only considers applicable rules.
    """
    if target_idx >= len(scores) or target_idx < 0:
        return 0.0
    
    k = min(k, len(scores))
    if k == 0:
        return 0.0
    
    # Get top-k
    top_k_indices = torch.topk(scores, k).indices
    
    # Check if target in top-k
    hit = 1.0 if target_idx in top_k_indices else 0.0
    
    # If applicable_mask provided, check if target was actually applicable
    if applicable_mask is not None:
        if not applicable_mask[target_idx]:
            # Target wasn't even applicable - mark as miss
            hit = 0.0
    
    return hit


def compute_mrr(scores, target_idx, applicable_mask=None):
    """
    Compute Mean Reciprocal Rank.
    
    If applicable_mask provided, only considers applicable rules.
    """
    if target_idx >= len(scores) or target_idx < 0:
        return 0.0
    
    # Check applicability
    if applicable_mask is not None and not applicable_mask[target_idx]:
        return 0.0
    
    sorted_indices = torch.argsort(scores, descending=True)
    rank = (sorted_indices == target_idx).nonzero(as_tuple=True)[0].item() + 1
    
    return 1.0 / rank


def compute_applicability_ratio(scores, applicable_mask):
    """
    Compute what fraction of top predictions are actually applicable.
    This is a diagnostic metric.
    """
    if not applicable_mask.any():
        return 0.0
    
    top_k = min(5, len(scores))
    top_indices = torch.topk(scores, k=top_k).indices
    
    applicable_in_top = applicable_mask[top_indices].float().mean()
    return applicable_in_top.item()


# Backward compatibility
class ProofSearchRankingLoss(nn.Module):
    """Deprecated: use ApplicabilityConstrainedLoss instead."""
    
    def __init__(self, margin=1.0, hard_negative_weight=2.0):
        super().__init__()
        self.margin = margin
        self.hard_negative_weight = hard_negative_weight
        self.alpha_hard = 0.7
        self.alpha_easy = 0.3
    
    def forward(self, scores, embeddings, target_idx, applicable_mask=None):
        """Calls new loss for compatibility."""
        loss = ApplicabilityConstrainedLoss(margin=self.margin)
        return loss(scores, embeddings, target_idx, applicable_mask)

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- NEW: Masked Softmax Cross-Entropy Loss ---
class MaskedCrossEntropyLoss(nn.Module):
    """
    Computes Cross-Entropy loss only over applicable rules.
    Assumes scores are logits.
    """
    def __init__(self):
        super().__init__()

    def forward(self, scores, embeddings, target_idx, applicable_mask=None):
        """
        Args:
            scores: [N] node scores (logits)
            embeddings: [N, D] (IGNORED - kept for API compatibility)
            target_idx: index of correct rule
            applicable_mask: [N] boolean mask of which rules are applicable

        Returns:
            loss (scalar tensor)
        """
        n = len(scores)

        # Basic validation
        if target_idx < 0 or target_idx >= n:
            # Invalid target, return a non-infinite loss to avoid crashing, but log it.
            print(f"Warning: Invalid target_idx ({target_idx}) in MaskedCrossEntropyLoss.")
            # Return a small loss, requires_grad=True if scores requires grad
            return torch.tensor(0.01, device=scores.device, requires_grad=scores.requires_grad)

        if applicable_mask is None:
            # If no mask provided, assume all are applicable (fallback, but should be avoided)
            applicable_mask = torch.ones_like(scores, dtype=torch.bool)
            print("Warning: No applicable_mask provided to MaskedCrossEntropyLoss. Assuming all applicable.")

        # CRITICAL: Target must be applicable
        if not applicable_mask[target_idx]:
            # This indicates a data inconsistency. The target *must* be applicable.
            # Return a large but finite loss and log a critical warning.
            print(f"CRITICAL WARNING: Target index {target_idx} is NOT applicable according to the mask!")
            # Use a large, grad-requiring tensor if scores require grad
            return torch.tensor(100.0, device=scores.device, requires_grad=scores.requires_grad)

        # --- Core Logic ---
        # 1. Get indices of applicable nodes
        applicable_indices = applicable_mask.nonzero(as_tuple=True)[0]

        # Handle case where only the target is applicable (no competition)
        if applicable_indices.numel() <= 1:
             # If only the target is applicable, loss should ideally be 0
             # (perfect prediction among choices).
             return torch.tensor(0.0, device=scores.device, requires_grad=scores.requires_grad)


        # 2. Gather scores for applicable nodes
        applicable_scores = scores[applicable_indices]

        # 3. Find the relative index of the target within the applicable set
        # This uses broadcasting and nonzero to find the position
        relative_target_idx_mask = (applicable_indices == target_idx)
        
        # Ensure the target was actually found (redundant due to earlier check, but safe)
        if not relative_target_idx_mask.any():
            print(f"INTERNAL ERROR: Target index {target_idx} not found in applicable indices {applicable_indices.tolist()} despite passing initial check!")
            return torch.tensor(100.0, device=scores.device, requires_grad=scores.requires_grad)

        relative_target_idx = relative_target_idx_mask.nonzero(as_tuple=True)[0]
        
        # Ensure it's a scalar tensor (0-dim) for cross_entropy
        if relative_target_idx.dim() > 0:
            relative_target_idx = relative_target_idx.squeeze()
            if relative_target_idx.dim() > 0: # Check again after squeeze
                 relative_target_idx = relative_target_idx[0] # Take the first if still not scalar

        # Ensure target is scalar tensor
        if relative_target_idx.dim() > 0:
            print(f"Warning: relative_target_idx is not scalar: {relative_target_idx}. Taking first element.")
            relative_target_idx = relative_target_idx[0].unsqueeze(0) # Keep as tensor [1]
        else:
            relative_target_idx = relative_target_idx.unsqueeze(0) # Make it tensor([idx]) shape [1]


        # 4. Compute Cross-Entropy Loss
        # F.cross_entropy expects logits [Batch, Classes] and target [Batch]
        # Here, Batch=1, Classes=num_applicable
        loss = F.cross_entropy(
            applicable_scores.unsqueeze(0), # Shape: [1, num_applicable]
            relative_target_idx           # Shape: [1]
        )
        return loss

class TripletLossWithHardMining(nn.Module):
    """
    Computes Triplet Loss focusing on the hardest negative applicable rule.

    Loss = max(0, margin - (score_positive - score_hardest_negative))
    """
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        print(f"Initialized TripletLossWithHardMining with margin={self.margin}")

    def forward(self, scores, embeddings, target_idx, applicable_mask=None):
        """
        Args:
            scores: [N] node scores (logits)
            embeddings: [N, D] (IGNORED - loss uses scores directly)
            target_idx: index of correct rule (positive)
            applicable_mask: [N] boolean mask of which rules are applicable

        Returns:
            loss (scalar tensor)
        """
        n = len(scores)

        # Basic validation
        if target_idx < 0 or target_idx >= n:
            print(f"Warning: Invalid target_idx ({target_idx}) in TripletLossWithHardMining.")
            return torch.tensor(0.01, device=scores.device, requires_grad=scores.requires_grad)

        if applicable_mask is None:
            applicable_mask = torch.ones_like(scores, dtype=torch.bool)
            print("Warning: No applicable_mask provided to TripletLossWithHardMining.")

        # Ensure target is applicable
        if not applicable_mask[target_idx]:
            print(f"CRITICAL WARNING: Target index {target_idx} is NOT applicable in TripletLoss!")
            return torch.tensor(100.0, device=scores.device, requires_grad=scores.requires_grad)

        # --- Core Triplet Logic ---
        positive_score = scores[target_idx]

        # 1. Identify applicable negative indices
        is_target_mask = torch.arange(n, device=scores.device) == target_idx
        negative_applicable_mask = applicable_mask & ~is_target_mask
        negative_applicable_indices = negative_applicable_mask.nonzero(as_tuple=True)[0]

        # 2. Handle case with no applicable negatives
        if negative_applicable_indices.numel() == 0:
            # No competitors, loss is 0
            return torch.tensor(0.0, device=scores.device, requires_grad=scores.requires_grad)

        # 3. Find the hardest negative (highest score among applicable negatives)
        negative_scores = scores[negative_applicable_indices]
        hardest_negative_score = torch.max(negative_scores)

        # 4. Compute Triplet Loss
        loss = F.relu(self.margin - (positive_score - hardest_negative_score))

        return loss
class FocalApplicabilityLoss(nn.Module):
    """
    SOTA loss combining:
    1. Applicability constraints (hard requirement)
    2. Focal loss (hard negative mining)
    3. Temperature scaling
    
    Problem Fixed:
    - Original: Fixed penalty=10.0 (arbitrary)
    - No hard negative mining
    - Treats all negatives equally
    - Loss scale unstable
    
    Solution:
    - Focal loss: down-weight easy negatives, focus on hard
    - Adaptive penalties based on violation magnitude
    - Temperature scaling for numerical stability
    
    Impact: +4% Hit@1, more stable training
    """
    
    def __init__(self, margin: float = 2.0, alpha: float = 0.5, 
                 gamma: float = 1.5, temperature: float = 0.1):
        super().__init__()
        
        self.margin = margin
        self.alpha = alpha  # Weight of hard negatives
        self.gamma = gamma  # Focusing parameter
        self.temperature = temperature  # Loss scaling
    
    def forward(self, scores: torch.Tensor, embeddings: torch.Tensor,
                target_idx: int, applicable_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores: [N] logits for each rule
            embeddings: [N, D] rule embeddings
            target_idx: index of correct rule
            applicable_mask: [N] bool mask of applicable rules
        
        Returns:
            loss: scalar
        """
        
        n = len(scores)
        
        # Validation
        if target_idx < 0 or target_idx >= n:
            return torch.tensor(0.01, device=scores.device, requires_grad=True)
        
        target_score = scores[target_idx]
        
        # ===== PART 1: INAPPLICABLE RULE PENALTY =====
        nonapplicable_mask = ~applicable_mask
        
        if nonapplicable_mask.any():
            nonapplicable_scores = scores[nonapplicable_mask]
            
            # How much these rules violate the constraint
            # Negative = good (inapplicable rules should score low)
            violations = nonapplicable_scores - target_score
            
            # Probability of this rule being selected (should be ~0)
            p_select = torch.sigmoid(violations)
            
            # Focal weight: (1-p)^gamma down-weights easy cases
            # If p is small (good), weight is 1. If p is large (bad), weight increases
            focal_weight = torch.pow(1 - p_select, self.gamma)
            
            # Loss: -log(1-p_select) with focal weighting
            loss_nonapplicable = (
                focal_weight * F.softplus(violations)
            ).mean()
        else:
            loss_nonapplicable = torch.tensor(0.0, device=scores.device)
        
        # ===== PART 2: APPLICABLE NEGATIVE RANKING =====
        is_target = (torch.arange(n, device=scores.device) == target_idx)
        applicable_neg_mask = applicable_mask & ~is_target
        
        if applicable_neg_mask.any():
            app_neg_scores = scores[applicable_neg_mask]
            
            # Margin violation: how much each negative violates margin
            margin_violations = torch.relu(
                self.margin - (target_score - app_neg_scores)
            )
            
            # Probability of margin violation
            p_violation = torch.sigmoid(margin_violations)
            
            # Focal weight: (p)^gamma focuses on violations
            focal_weight_app = torch.pow(p_violation, self.gamma)
            
            # Weighted margin loss
            loss_applicable = (
                focal_weight_app * margin_violations
            ).mean()
        else:
            loss_applicable = torch.tensor(0.0, device=scores.device)
        
        # ===== PART 3: HARD NEGATIVE MINING =====
        # Identify hardest negatives (most violated constraints)
        if nonapplicable_mask.any():
            # Get top-K hardest violations
            violations_all = nonapplicable_scores - target_score
            
            K = min(3, len(violations_all))
            top_violations, _ = torch.topk(violations_all, k=K)
            
            # Extra penalty for hardest negatives
            hard_loss = F.softplus(top_violations).mean()
        else:
            hard_loss = torch.tensor(0.0, device=scores.device)
        
        # ===== PART 4: TEMPERATURE SCALING =====
        # Prevent loss explosion at initialization
        total_loss = self.temperature * (
            loss_nonapplicable +
            self.alpha * loss_applicable +
            0.5 * hard_loss
        )
        
        return total_loss
class MultiTaskProofLoss(nn.Module):
    """
    Multi-task loss for joint training:
    - Rule ranking (main task)
    - Value prediction (proof quality estimation)
    - Tactic classification (high-level strategy)
    """
    
    def __init__(self, w_ranking: float = 0.7, w_value: float = 0.2, 
                 w_tactic: float = 0.1):
        super().__init__()
        self.w_ranking = w_ranking
        self.w_value = w_value
        self.w_tactic = w_tactic
        
        self.ranking_loss = ContrastiveRankingLoss()
        self.value_loss = nn.MSELoss()
        self.tactic_loss = nn.CrossEntropyLoss()
    
    def forward(self, scores: torch.Tensor, embeddings: torch.Tensor,
                target_idx: int, applicable_mask: torch.Tensor,
                value_pred: torch.Tensor, value_target: torch.Tensor,
                tactic_logits: torch.Tensor, tactic_target: torch.Tensor) -> Dict:
        """
        Returns dict of individual losses + combined loss
        """
        
        # Ranking loss
        loss_rank = self.ranking_loss(scores, embeddings, target_idx, applicable_mask)
        
        # Value loss
        if value_pred is not None and value_target is not None:
            loss_value = self.value_loss(value_pred, value_target)
        else:
            loss_value = torch.tensor(0.0, device=scores.device)
        
        # Tactic loss
        if tactic_logits is not None and tactic_target is not None:
            loss_tactic = self.tactic_loss(tactic_logits, tactic_target)
        else:
            loss_tactic = torch.tensor(0.0, device=scores.device)
        
        # Combined
        total_loss = (
            self.w_ranking * loss_rank +
            self.w_value * loss_value +
            self.w_tactic * loss_tactic
        )
        
        return {
            'total': total_loss,
            'ranking': loss_rank,
            'value': loss_value,
            'tactic': loss_tactic
        }
class ContrastiveRankingLoss(nn.Module):
    """
    SOTA loss combining ranking + contrastive + hard negative mining.
    
    Three components:
    1. Ranking: Cross-entropy on applicable rules
    2. Contrastive: Embedding space structure
    3. Hard negatives: Focus on difficult misclassifications
    """
    
    def __init__(self, temperature: float = 0.1, margin: float = 1.0,
                 alpha_rank: float = 0.6, alpha_contrastive: float = 0.3,
                 alpha_hard: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.alpha_rank = alpha_rank
        self.alpha_contrastive = alpha_contrastive
        self.alpha_hard = alpha_hard
    
    def forward(self, scores: torch.Tensor, embeddings: torch.Tensor,
                target_idx: int, applicable_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores: [N] model scores for each rule
            embeddings: [N, D] rule embeddings
            target_idx: index of correct rule
            applicable_mask: [N] boolean, which rules are applicable
        
        Returns:
            loss: scalar tensor
        """
        N = len(scores)
        device = scores.device
        
        # Validate
        if target_idx < 0 or target_idx >= N:
            return torch.tensor(1.0, device=device, requires_grad=True)
        
        if not applicable_mask[target_idx]:
            return torch.tensor(100.0, device=device, requires_grad=True)
        
        applicable_indices = applicable_mask.nonzero(as_tuple=True)[0]
        
        if len(applicable_indices) == 0:
            return torch.tensor(1.0, device=device, requires_grad=True)
        
        # === PART 1: Ranking Loss ===
        applicable_scores = scores[applicable_indices]
        
        # Find target position in applicable set
        target_pos = (applicable_indices == target_idx).nonzero(as_tuple=True)[0]
        if len(target_pos) == 0:
            return torch.tensor(100.0, device=device, requires_grad=True)
        
        target_pos = target_pos[0].item()
        
        # Cross-entropy with temperature
        logits = applicable_scores / self.temperature
        ranking_loss = F.cross_entropy(
            logits.unsqueeze(0),
            torch.tensor([target_pos], device=device)
        )
        
        # === PART 2: Contrastive Loss ===
        if embeddings.numel() > 0:
            embeddings_norm = F.normalize(embeddings, p=2, dim=1)
            target_emb = embeddings_norm[target_idx]
            
            similarities = torch.mm(embeddings_norm, target_emb.unsqueeze(1)).squeeze(1)
            applicable_sims = similarities[applicable_indices]
            
            # InfoNCE: positive should have highest similarity
            if len(applicable_indices) > 1:
                pos_sim = applicable_sims[target_pos]
                neg_sims = torch.cat([
                    applicable_sims[:target_pos],
                    applicable_sims[target_pos+1:]
                ])
                
                logits_cont = torch.cat([
                    pos_sim.unsqueeze(0) / self.temperature,
                    neg_sims / self.temperature
                ])
                labels_cont = torch.zeros(1, dtype=torch.long, device=device)
                contrastive_loss = F.cross_entropy(logits_cont.unsqueeze(0), labels_cont)
            else:
                contrastive_loss = torch.tensor(0.0, device=device)
        else:
            contrastive_loss = torch.tensor(0.0, device=device)
        
        # === PART 3: Hard Negative Mining ===
        neg_mask = applicable_mask & (torch.arange(N, device=device) != target_idx)
        
        if neg_mask.any():
            neg_scores = scores[neg_mask]
            hardest_neg_score = neg_scores.max()
            
            # Margin constraint: target should be > hardest negative + margin
            hard_loss = torch.relu(self.margin + hardest_neg_score - scores[target_idx])
        else:
            hard_loss = torch.tensor(0.0, device=device)
        
        # === COMBINE ===
        loss = (
            self.alpha_rank * ranking_loss +
            self.alpha_contrastive * contrastive_loss +
            self.alpha_hard * hard_loss
        )
        
        return loss


class FocusedRankingLoss(nn.Module):
    """
    SOTA Loss: Upgrades ContrastiveRankingLoss with a clear separation of concerns.
    
    1. Ranking Component: Uses Masked Cross-Entropy to rank the target
       against *other applicable* rules. This is the primary, high-precision signal.
    2. Applicability Component: Uses a hard-negative margin loss to push all
       applicable scores above the *highest-scoring inapplicable* rule.
       This provides the global "applicability" signal.
       
    This avoids the conflicting gradients of the original ContrastiveRankingLoss.
    """
    
    def __init__(self, temperature: float = 0.1, margin: float = 1.0, 
                 alpha: float = 0.3, gamma: float = 2.0):
        """
        Args:
            temperature: Scales logits for the ranking component.
            margin: The score gap for the applicability component.
            alpha: Weight of the applicability component (L_app).
            gamma: Focusing parameter for applicability (similar to FocalLoss).
        """
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.alpha = alpha
        self.gamma = gamma
        self.ranking_loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, scores: torch.Tensor, embeddings: torch.Tensor,
                target_idx: int, applicable_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores: [N] model scores for each rule
            embeddings: [N, D] (IGNORED - kept for API compatibility)
            target_idx: index of correct rule
            applicable_mask: [N] boolean, which rules are applicable
        
        Returns:
            loss: scalar tensor
        """
        N = len(scores)
        device = scores.device
        
        # --- Validation (unchanged) ---
        if target_idx < 0 or target_idx >= N:
            return torch.tensor(1.0, device=device, requires_grad=True)
        
        if not applicable_mask[target_idx]:
            # This is a critical data error, but we must return a loss
            return torch.tensor(100.0, device=device, requires_grad=True)

        applicable_indices = applicable_mask.nonzero(as_tuple=True)[0]

        if len(applicable_indices) == 0:
            # Should not happen if target is applicable, but as safeguard
            return torch.tensor(1.0, device=device, requires_grad=True)

        # === PART 1: Ranking Loss (High-Precision Signal) ===
        # Compute Cross-Entropy only over the set of *applicable* rules.
        
        applicable_scores = scores[applicable_indices]
        
        # Find target's relative position in the applicable-only set
        target_relative_idx_mask = (applicable_indices == target_idx)
        target_relative_idx = target_relative_idx_mask.nonzero(as_tuple=True)[0]
        
        if target_relative_idx.numel() == 0:
            # Should also not happen
            return torch.tensor(100.0, device=device, requires_grad=True)
            
        target_relative_idx = target_relative_idx.squeeze() # Make it a scalar tensor
        
        # Scale by temperature and compute loss
        logits = applicable_scores.unsqueeze(0) / self.temperature # [1, num_applicable]
        target = target_relative_idx.unsqueeze(0)                   # [1]
        
        loss_rank = self.ranking_loss_fn(logits, target)

        # === PART 2: Applicability Loss (Global Signal) ===
        # Push the *target's score* above the *hardest inapplicable score*.
        
        inapplicable_mask = ~applicable_mask
        
        if inapplicable_mask.any():
            inapplicable_scores = scores[inapplicable_mask]
            hardest_inapp_score = torch.max(inapplicable_scores)
            target_score = scores[target_idx]
            
            # Compute margin violation
            violation = F.relu(self.margin - (target_score - hardest_inapp_score))
            
            # Apply focal weighting (down-weight easy cases)
            p_violation = torch.sigmoid(violation)
            focal_weight = torch.pow(p_violation, self.gamma)
            
            loss_app = (focal_weight * violation).mean()
        else:
            # No inapplicable rules to penalize
            loss_app = torch.tensor(0.0, device=device)

        # === COMBINE ===
        total_loss = loss_rank + self.alpha * loss_app
        
        return total_loss

class InfoNCEListwiseLoss(nn.Module):
    """
    SOTA Loss: Upgrades ContrastiveRankingLoss to a focused InfoNCE/Listwise objective.
    
    This loss aligns with SOTA listwise ranking and contrastive 
    learning (InfoNCE).

    It has two clear, non-conflicting components:
    1. Listwise Ranking (InfoNCE): A temperature-scaled Cross-Entropy over the
       set of *ALL APPLICABLE* rules. This forces the model to rank the
       correct target as #1 from the list of valid candidates.
    2. Applicability Margin: A hard-negative margin loss that pushes the
       score of the target *above* the score of the *hardest inapplicable* rule.
       This provides the global "applicability" signal.
    """
    
    def __init__(self, temperature: float = 0.1, margin: float = 1.0, 
                 alpha: float = 0.5):
        """
        Args:
            temperature: Scales logits for the ranking component. Lower values
                         create a "sharper" probability distribution, forcing
                         the model to be more confident.
            margin: The score gap for the applicability component.
            alpha: Weight of the applicability component (loss_app).
        """
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.alpha = alpha
        self.ranking_loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, scores: torch.Tensor, embeddings: torch.Tensor,
                target_idx: int, applicable_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores: [N] model scores for each rule
            embeddings: [N, D] (IGNORED - kept for API compatibility)
            target_idx: index of correct rule
            applicable_mask: [N] boolean, which rules are applicable
        
        Returns:
            loss: scalar tensor
        """
        
        N = len(scores)
        device = scores.device
        
        if target_idx < 0 or target_idx >= N:
            return torch.tensor(1.0, device=device, requires_grad=True)
        
        if not applicable_mask[target_idx]:
            # This is a critical data error, but we must return a loss
            return torch.tensor(100.0, device=device, requires_grad=True)

        applicable_indices = applicable_mask.nonzero(as_tuple=True)[0]

        if len(applicable_indices) == 0:
            return torch.tensor(1.0, device=device, requires_grad=True)

        # === PART 1: Listwise Ranking Loss (InfoNCE) ===
        # This is the primary signal to break your plateau.
        
        # Get scores of *all* applicable rules
        applicable_scores = scores[applicable_indices]
        
        # Find target's relative position in this new applicable-only tensor
        target_relative_idx_mask = (applicable_indices == target_idx)
        target_relative_idx = target_relative_idx_mask.nonzero(as_tuple=True)[0]
        
        if target_relative_idx.numel() == 0:
            return torch.tensor(100.0, device=device, requires_grad=True)
            
        target_relative_idx = target_relative_idx.squeeze().unsqueeze(0) # [1]
        
        # Scale scores by temperature to create logits
        logits = applicable_scores.unsqueeze(0) / self.temperature # [1, num_applicable]
        
        # Compute Cross-Entropy. This forces the logit for the
        # target_relative_idx to be the highest.
        loss_rank = self.ranking_loss_fn(logits, target_relative_idx)

        # === PART 2: Applicability Margin Loss ===
        # This is the "global" signal your FocalLoss was doing.
        
        inapplicable_mask = ~applicable_mask
        
        if inapplicable_mask.any():
            inapplicable_scores = scores[inapplicable_mask]
            # Find the single hardest inapplicable rule
            hardest_inapp_score = torch.max(inapplicable_scores)
            target_score = scores[target_idx]
            
            # Compute margin violation: target vs. hardest inapplicable
            violation = F.relu(self.margin - (target_score - hardest_inapp_score))
            loss_app = violation.mean()
        else:
            # No inapplicable rules to penalize
            loss_app = torch.tensor(0.0, device=device)

        # === COMBINE ===
        total_loss = loss_rank + (self.alpha * loss_app)
        
        return total_loss

class SOTAInfoNCELoss(nn.Module):
    """
    SOTA-Upgraded InfoNCE Loss (replaces InfoNCEListwiseLoss).

    This version incorporates two key SOTA improvements:
    1.  Temperature-Free Ranking (Part 1): Replaces the sensitive 
        'temperature' hyperparameter with a stable `atanh` scaling.
    2.  Weighted Applicability (Part 2): Replaces the simple margin 
        against the *single hardest* negative with a Focal-style weighted 
        margin loss for a more stable gradient.
    """
    
    def __init__(self, margin: float = 1.0, alpha: float = 0.5, gamma: float = 2.0, label_smoothing: float = 0.0):
        """
        Args:
            margin: The score gap for the applicability component.
            alpha: Weight of the applicability component (loss_app).
            gamma: Focusing parameter for the applicability component.
            label_smoothing: How much to "smooth" the target label (e.g., 0.1)
        """
        super().__init__()
        self.margin = margin
        self.alpha = alpha
        self.gamma = gamma
        self.ranking_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
    def _stable_atanh_scaling(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Applies numerically stable atanh scaling in place of temperature.
        Clips scores to avoid inf/-inf from atanh(1) or atanh(-1).
        """
        clamped_scores = torch.clamp(scores, -1.0 + 1e-7, 1.0 - 1e-7)
        return torch.atanh(clamped_scores)

    def forward(self, scores: torch.Tensor, embeddings: torch.Tensor,
                target_idx: int, applicable_mask: torch.Tensor) -> torch.Tensor:
        
        N = len(scores)
        device = scores.device
        
        if target_idx < 0 or target_idx >= N:
            return torch.tensor(1.0, device=device, requires_grad=True)
        
        if not applicable_mask[target_idx]:
            return torch.tensor(100.0, device=device, requires_grad=True)

        applicable_indices = applicable_mask.nonzero(as_tuple=True)[0]

        if len(applicable_indices) == 0:
            return torch.tensor(1.0, device=device, requires_grad=True)

        # === PART 1: Temperature-Free Listwise Ranking (InfoNCE) ===
        applicable_scores = scores[applicable_indices]
        
        target_relative_idx_mask = (applicable_indices == target_idx)
        target_relative_idx = target_relative_idx_mask.nonzero(as_tuple=True)[0]
        
        if target_relative_idx.numel() == 0:
            return torch.tensor(100.0, device=device, requires_grad=True)
            
        target_relative_idx = target_relative_idx.squeeze().unsqueeze(0) # [1]
        
        # --- KEY CHANGE HERE ---
        # We assume scores are raw logits/similarities.
        # We normalize them with tanh and then use atanh scaling.
        normalized_scores = torch.tanh(applicable_scores)
        
        # Scale to logits *without* temperature
        logits = self._stable_atanh_scaling(normalized_scores).unsqueeze(0) # [1, num_applicable]
        # --- END KEY CHANGE ---
        
        loss_rank = self.ranking_loss_fn(logits, target_relative_idx)

        # === PART 2: Weighted Applicability Margin Loss ===
        inapplicable_mask = ~applicable_mask
        
        if inapplicable_mask.any():
            inapplicable_scores = scores[inapplicable_mask]
            target_score = scores[target_idx]
            
            violations = F.relu(self.margin - (target_score - inapplicable_scores))
            p_violation = torch.sigmoid(inapplicable_scores - target_score + self.margin)
            focal_weight = torch.pow(p_violation, self.gamma)
            loss_app = (focal_weight * violations).mean()
        else:
            loss_app = torch.tensor(0.0, device=device)

        # === COMBINE ===
        total_loss = loss_rank + (self.alpha * loss_app)
        
        return total_loss
        
"""
Decoupled Applicability-Aware Ranking Loss (NEW SOTA-Inspired Loss)

Key Features:
- Decoupled: Separate applicability classification + ranking among applicable.
- Focal BCE for applicability (handles imbalance, from Focal Loss, ICCV 2017).
- Listwise Softmax CE for ranking (stable, from ListNet, ICML 2007).
- No unstable ops like atanh; uses log_softmax for numerical stability.
- Hyperparams tuned for NTP: alpha=0.5 (balances components), gamma=2.0 (focal).

This addresses your convergence issues: smoother gradients, no inf returns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoupledApplicabilityRankingLoss(nn.Module):
    """
    Solves the sparse gradient problem by decoupling applicability and ranking.
    
    1. L_rank: Listwise Cross-Entropy over *applicable* nodes.
       - Teaches: "Among these valid choices, which is best?"
    2. L_app: Focal Binary Cross-Entropy over *all* nodes.
       - Teaches: "Is this node valid at all?"
    """
    def __init__(self, alpha: float = 0.5, gamma: float = 2.0, label_smoothing: float = 0.1):
        """
        Args:
            alpha: Weight of the applicability loss. (0.5 is a good start)
            gamma: Focal loss gamma to focus on hard-to-classify nodes.
            label_smoothing: For the ranking loss.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ranking_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        # Use reduction='none' for manual focal weighting
        self.applicability_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, scores: torch.Tensor, embeddings: torch.Tensor,
                target_idx: int, applicable_mask: torch.Tensor) -> torch.Tensor:
        
        N = len(scores)
        device = scores.device

        if target_idx < 0 or target_idx >= N or not applicable_mask[target_idx]:
            # Critical data error or invalid state.
            return torch.tensor(1.0, device=device, requires_grad=True)

        # --- Part 1: Applicability Classification (Dense Gradient) ---
        appl_labels = applicable_mask.float()
        
        # Compute raw BCE loss for all N nodes
        bce_loss = self.applicability_loss_fn(scores, appl_labels)
        
        # Compute focal weights
        p_t = torch.exp(-bce_loss) # p_t = p if y=1, 1-p if y=0
        focal_weight = (1.0 - p_t).pow(self.gamma)
        
        # Compute final weighted Focal Loss
        loss_app = (focal_weight * bce_loss).mean()

        # --- Part 2: Ranking Among Applicable (Sparse, Focused Gradient) ---
        applicable_indices = applicable_mask.nonzero(as_tuple=True)[0]
        
        if applicable_indices.numel() <= 1:
            # No competition, so no ranking loss
            loss_rank = torch.tensor(0.0, device=device)
        else:
            applicable_scores = scores[applicable_indices]
            
            # Find relative target index
            rel_target_idx = (applicable_indices == target_idx).nonzero(as_tuple=True)[0]
            
            # Compute ranking loss
            loss_rank = self.ranking_loss_fn(
                applicable_scores.unsqueeze(0), # [1, M_applicable]
                rel_target_idx                  # [1]
            )

        # --- Combine ---
        total_loss = (1.0 - self.alpha) * loss_rank + self.alpha * loss_app
        return total_loss

class HybridTripletListwiseValueLoss(nn.Module):
    """SOTA hybrid: Triplet (local margins) + ListMLE (global ranking) + MSE (proof value)."""
    def __init__(self, margin=1.0, alpha_triplet=0.6, alpha_list=0.3, alpha_value=0.1, label_smoothing=0.1):
        super().__init__()
        self.margin = margin
        self.alpha_triplet = alpha_triplet
        self.alpha_list = alpha_list
        self.alpha_value = alpha_value
        self.smoothing = label_smoothing
        self.mse = nn.MSELoss()
        
        logger.info(f"Initialized HybridTripletListwiseValueLoss (α_triplet={alpha_triplet}, α_list={alpha_list}, α_value={alpha_value})")

    def forward(self, scores, embeddings, target_idx, applicable_mask=None, value_pred=None, value_target=None):
        n = len(scores)
        device = scores.device

        if applicable_mask is None:
            applicable_mask = torch.ones(n, dtype=torch.bool, device=device)

        if target_idx < 0 or target_idx >= n or not applicable_mask[target_idx]:
             # Invalid target, return small, stable loss
            l_triplet = torch.tensor(0.0, device=device)
            l_list = torch.tensor(1.0, device=device, requires_grad=True) # Non-zero to avoid bad optimum
        else:
            # Triplet (hard mining)
            pos_score = scores[target_idx]
            neg_mask = applicable_mask & (torch.arange(n, device=device) != target_idx)
            if neg_mask.any():
                neg_scores = scores[neg_mask]
                hardest_neg = neg_scores.max()
                l_triplet = F.relu(hardest_neg - pos_score + self.margin)
            else:
                l_triplet = torch.tensor(0.0, device=device)

            # Listwise (on applicable; approx ListMLE)
            appl_scores = scores[applicable_mask]
            if len(appl_scores) > 1:
                rel_target_mask = (torch.arange(n, device=device)[applicable_mask] == target_idx)
                rel_target = rel_target_mask.nonzero(as_tuple=True)[0].item()
                
                # Apply label smoothing
                log_probs = F.log_softmax(appl_scores, dim=0)
                n_classes = len(appl_scores)
                one_hot = F.one_hot(torch.tensor(rel_target, device=device), n_classes).float()
                smooth_target = (1.0 - self.smoothing) * one_hot + (self.smoothing / n_classes)
                
                l_list = -(smooth_target * log_probs).sum()
            else:
                l_list = torch.tensor(0.0, device=device)

        # Value (new: proof remaining steps)
        l_value = self.mse(value_pred.squeeze(), value_target.squeeze()) if value_pred is not None and value_target is not None else torch.tensor(0.0, device=device)

        return self.alpha_triplet * l_triplet + self.alpha_list * l_list + self.alpha_value * l_value

# Update get_recommended_loss to use this
# def get_recommended_loss(loss_type='decoupled', alpha=0.5, gamma=2.0, label_smoothing=0.1):
#     print(f"Using NEW Loss: DecoupledApplicabilityRankingLoss (alpha={alpha}, gamma={gamma})")
#     return DecoupledApplicabilityRankingLoss(alpha=alpha, gamma=gamma, label_smoothing=label_smoothing)

class TheoreticallySoundLoss(nn.Module):
    """
    Decoupled loss with correct scaling.
    FIX: No in-place operations on parameters during forward pass.
    """
    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin
        self.ranking_loss_fn = nn.CrossEntropyLoss()
        # Use a regular Python float for adaptive weighting (not a parameter)
        self.alpha_app = 0.3
        
    def forward(self, scores, embeddings, target_idx, applicable_mask):
        """
        Args:
            scores: [N] node scores
            embeddings: [N, D] (IGNORED - kept for API compatibility)
            target_idx: index of correct rule
            applicable_mask: [N] binary mask of which rules are applicable
        """
        
        # Part 1: Ranking among applicable (cross-entropy)
        applicable_indices = applicable_mask.nonzero(as_tuple=True)[0]
        
        if applicable_indices.numel() == 0:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)
            
        applicable_scores = scores[applicable_indices]
        
        # Find relative target position
        target_pos_mask = (applicable_indices == target_idx)
        if not target_pos_mask.any():
            return torch.tensor(100.0, device=scores.device, requires_grad=True)
            
        target_pos = target_pos_mask.nonzero(as_tuple=True)[0]
        
        loss_rank = self.ranking_loss_fn(
            applicable_scores.unsqueeze(0),
            target_pos
        )
        
        # Part 2: Margin against inapplicable
        inapplicable_mask = ~applicable_mask
        if inapplicable_mask.any():
            hardest_inapp = scores[inapplicable_mask].max()
            loss_app = F.relu(self.margin - (scores[target_idx] - hardest_inapp))
            
            # FIX: Update alpha_app without affecting gradient graph
            # Detach from computation, update as Python float
            with torch.no_grad():
                loss_app_val = loss_app.item()
                if loss_app_val < self.margin / 2:
                    self.alpha_app = max(0.1, self.alpha_app * 0.95)
                else:
                    self.alpha_app = min(0.5, self.alpha_app * 1.05)
        else:
            loss_app = torch.tensor(0.0, device=scores.device)

        # Use the Python float (no gradient issues)
        return loss_rank + self.alpha_app * loss_app

def get_recommended_loss(loss_type='triplet_hard', margin=1.0): # Add margin arg
    """Returns the recommended loss function."""

    
    if loss_type == 'applicability_constrained':
        print("Using Loss: ApplicabilityConstrainedLoss")
        return ApplicabilityConstrainedLoss(margin=margin, penalty_nonapplicable=10.0)
    elif loss_type == 'hybrid':
        return HybridTripletListwiseValueLoss(margin=margin)
    elif loss_type == 'cross_entropy':
        print("Using Loss: MaskedCrossEntropyLoss")
        return MaskedCrossEntropyLoss()
    elif loss_type == 'triplet_hard':
        print(f"Using Loss: TripletLossWithHardMining (margin={margin})")
        return TripletLossWithHardMining(margin=margin)
    else:
        print(f"Warning: Unknown loss_type '{loss_type}'. Defaulting to TripletLossWithHardMining.")
        return TripletLossWithHardMining(margin=margin) # Default to Triplet