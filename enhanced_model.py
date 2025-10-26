"""
Fixed Enhanced LogNet Model with Proper Component Integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np
from proof_state import ProofState, ProofStateEncoder
from proof_search import PolicyNetwork


class TacticDecoder(nn.Module):
    """
    High-level tactic prediction to guide rule selection.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_tactics: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_tactics = num_tactics
        
        # Feature extraction
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Tactic classification
        self.tactic_head = nn.Linear(hidden_dim, num_tactics)
        
        # Applicability scoring
        self.applicability_head = nn.Linear(hidden_dim, num_tactics)
        
        # Parameter generation
        self.param_head = nn.Linear(hidden_dim, num_tactics * 5)  # 5 params per tactic
    
    def forward(self, state_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            state_embedding: [batch, input_dim] or [input_dim]
        
        Returns:
            Dictionary with tactic predictions
        """
        features = self.encoder(state_embedding)
        
        tactic_logits = self.tactic_head(features)
        applicability = torch.sigmoid(self.applicability_head(features))
        parameters = self.param_head(features)
        
        return {
            'tactic_logits': tactic_logits,
            'tactic_probs': F.softmax(tactic_logits, dim=-1),
            'applicability': applicability,
            'parameters': parameters
        }


class MultiRelationalGNNLayer(nn.Module):
    """
    Single layer of multi-relational GNN.
    Processes different relation types separately.
    """
    
    def __init__(self, input_dim: int, output_dim: int, num_relations: int = 6):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relations = num_relations
        
        # Relation-specific transformations
        self.relation_transforms = nn.ModuleList([
            nn.Linear(input_dim, output_dim) for _ in range(num_relations)
        ])
        
        # Attention over relations
        self.relation_attention = nn.MultiheadAttention(
            output_dim, num_heads=4, batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(output_dim, output_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_types: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [num_nodes, input_dim] node features
            edge_index: [2, num_edges] edge indices
            edge_types: [num_edges] edge type indices
        
        Returns:
            [num_nodes, output_dim] updated node features
        """
        num_nodes = x.shape[0]
        outputs = []
        
        # Process each relation type
        for rel_idx in range(self.num_relations):
            # Get edges of this type
            mask = (edge_types == rel_idx)
            
            if mask.sum() == 0:
                # No edges of this type
                outputs.append(torch.zeros(num_nodes, self.output_dim, device=x.device))
                continue
            
            rel_edges = edge_index[:, mask]
            
            # Apply relation-specific transformation
            transformed = self.relation_transforms[rel_idx](x)
            
            # Aggregate messages
            aggregated = torch.zeros(num_nodes, self.output_dim, device=x.device)
            if rel_edges.shape[1] > 0:
                src, dst = rel_edges[0], rel_edges[1]
                aggregated.scatter_add_(0, dst.unsqueeze(1).expand(-1, self.output_dim),
                                       transformed[src])
            
            outputs.append(aggregated)
        
        # Stack relation outputs
        stacked = torch.stack(outputs, dim=0)  # [num_relations, num_nodes, output_dim]
        
        # Apply attention over relations
        stacked_reshaped = stacked.transpose(0, 1)  # [num_nodes, num_relations, output_dim]
        attended, _ = self.relation_attention(
            stacked_reshaped, stacked_reshaped, stacked_reshaped
        )
        
        # Aggregate across relations
        aggregated_output = attended.mean(dim=1)  # [num_nodes, output_dim]
        
        # Output projection with residual
        output = self.output_proj(aggregated_output + x[:, :self.output_dim])
        
        return output


class LearnedAttentionLayer(nn.Module):
    """
    Learned attention over proof history.
    """
    
    def __init__(self, hidden_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=0.1
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
    
    def forward(self, x: torch.Tensor, step_numbers: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, num_nodes, hidden_dim] features
            step_numbers: [batch, num_nodes] step indices
        
        Returns:
            attended_features: [batch, num_nodes, hidden_dim]
            attention_weights: [batch, num_heads, num_nodes, num_nodes]
        """
        # Self-attention
        attended, attention_weights = self.attention(x, x, x, need_weights=True)
        
        # Residual + LayerNorm
        x = self.layer_norm(x + attended)
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        output = self.layer_norm(x + ff_output)
        
        return output, attention_weights


class EnhancedLogNetModel(nn.Module):
    """
    Enhanced LogNet model with all components properly integrated.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_rules = config.get('num_rules', 100)
        self.num_tactics = config.get('num_tactics', 10)
        
        # State encoding
        self.state_encoder = ProofStateEncoder(hidden_dim=self.hidden_dim)
        
        # Multi-relational GNN
        self.gnn_layers = nn.ModuleList([
            MultiRelationalGNNLayer(self.hidden_dim, self.hidden_dim)
            for _ in range(2)
        ])
        
        # Tactic decoder
        self.tactic_decoder = TacticDecoder(
            self.hidden_dim, self.hidden_dim, self.num_tactics
        )
        
        # Learned attention
        self.attention_layers = nn.ModuleList([
            LearnedAttentionLayer(self.hidden_dim)
            for _ in range(2)
        ])
        
        # Rule prediction head
        self.rule_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.num_rules)
        )
        
        # Value head (for RL)
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        
        # Tactic-to-rule modulation
        self.tactic_modulation = nn.Linear(self.num_tactics, self.num_rules)
    
    def forward(self, proof_state: ProofState,
                edge_index: Optional[torch.Tensor] = None,
                edge_types: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            proof_state: ProofState object
            edge_index: [2, num_edges] graph edges
            edge_types: [num_edges] edge relation types
        
        Returns:
            Dictionary with predictions
        """
        # 1. Encode proof state
        state_embedding = self.state_encoder(proof_state)  # [hidden_dim]
        
        # Ensure on correct device
        state_embedding = state_embedding.to(self.device if hasattr(self, 'device') else next(self.parameters()).device)
        
        # Expand for batch processing
        x = state_embedding.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_d
        
        # Expand for batch processing
        x = state_embedding.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]
        
        # 2. Apply GNN if graph provided
        if edge_index is not None and edge_types is not None:
            # Convert to proper format
            features = x.squeeze(0)  # [1, hidden_dim]
            
            for gnn_layer in self.gnn_layers:
                features = gnn_layer(features, edge_index, edge_types)
            
            x = features.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]
        
        # 3. Get tactic predictions
        tactic_output = self.tactic_decoder(state_embedding)  # Dict
        tactic_probs = tactic_output['tactic_probs']  # [num_tactics]
        
        # 4. Apply learned attention
        for attention_layer in self.attention_layers:
            x_attended, attn_weights = attention_layer(x, torch.zeros(1, 1, dtype=torch.long))
            x = x_attended
        
        # Squeeze batch dimensions
        features = x.squeeze(0).squeeze(0)  # [hidden_dim]
        
        # 5. Predict rules (base predictions)
        rule_logits = self.rule_head(features)  # [num_rules]
        
        # 6. Modulate by tactic
        tactic_modulation = self.tactic_modulation(tactic_probs)  # [num_rules]
        rule_logits = rule_logits + 0.3 * tactic_modulation
        
        # 7. Predict value
        value = torch.tanh(self.value_head(features))  # [1]
        
        return {
            'rule_logits': rule_logits,
            'rule_probs': F.softmax(rule_logits, dim=-1),
            'tactic_logits': tactic_output['tactic_logits'],
            'tactic_probs': tactic_probs,
            'tactic_applicability': tactic_output['applicability'],
            'value': value,
            'attention_weights': attn_weights if 'attn_weights' in locals() else None,
            'state_embedding': state_embedding
        }


class HardNegativeLoss(nn.Module):
    """
    Loss with semantic hard negative mining.
    """
    
    def __init__(self, margin: float = 1.0, hard_neg_weight: float = 2.0):
        super().__init__()
        self.margin = margin
        self.hard_neg_weight = hard_neg_weight
    
    def forward(self, rule_scores: torch.Tensor, target_rule: int,
                available_rules: List[int]) -> torch.Tensor:
        """
        Args:
            rule_scores: [num_rules] prediction scores
            target_rule: Target rule index
            available_rules: List of valid rule indices
        
        Returns:
            Loss value
        """
        if target_rule < 0 or target_rule >= len(rule_scores):
            return torch.tensor(0.0, device=rule_scores.device, requires_grad=True)
        
        positive_score = rule_scores[target_rule]
        
        # Hard negatives: high-scoring invalid rules
        hard_negatives = []
        for rule_id in range(len(rule_scores)):
            if rule_id != target_rule and rule_id not in available_rules:
                hard_negatives.append(rule_id)
        
        if not hard_negatives:
            return torch.tensor(0.0, device=rule_scores.device, requires_grad=True)
        
        # Get top-k hard negatives by score
        hard_neg_scores = rule_scores[hard_negatives]
        top_k = min(5, len(hard_negatives))
        top_scores = torch.topk(hard_neg_scores, top_k)[0]
        
        # Hard negative loss (margin-based)
        hard_losses = F.relu(self.margin - (positive_score - top_scores))
        hard_loss = hard_losses.mean()
        
        return hard_loss


class EnhancedTrainingPipeline:
    """
    Complete training pipeline with all components integrated.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model
        self.model = EnhancedLogNetModel(config).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Loss functions
        self.rule_loss_fn = nn.CrossEntropyLoss()
        self.hard_neg_loss_fn = HardNegativeLoss()
        self.value_loss_fn = nn.MSELoss()
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Metrics
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_step(self, proof_state: ProofState, target_rule: int,
                   target_value: float, available_rules: List[int]) -> Dict[str, float]:
        """
        Single training step.
        """
        # Forward pass
        output = self.model(proof_state)
        
        rule_logits = output['rule_logits']
        value = output['value']
        
        # Losses
        rule_loss = self.rule_loss_fn(
            rule_logits.unsqueeze(0), torch.tensor([target_rule])
        )
        
        hard_neg_loss = self.hard_neg_loss_fn(
            rule_logits, target_rule, available_rules
        )
        
        value_loss = self.value_loss_fn(
            value, torch.tensor([[target_value]])
        )
        
        # Combined loss
        total_loss = rule_loss + 0.5 * hard_neg_loss + 0.1 * value_loss
        
        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'rule_loss': rule_loss.item(),
            'hard_neg_loss': hard_neg_loss.item(),
            'value_loss': value_loss.item()
        }