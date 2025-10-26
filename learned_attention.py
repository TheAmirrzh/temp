"""
Learned Attention Mechanism
Implements Issue #13: Learned attention over proof history using Transformer architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any
import math
import numpy as np
from dataclasses import dataclass


@dataclass
class AttentionConfig:
    """Configuration for learned attention."""
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    max_steps: int = 100
    dropout: float = 0.1
    use_positional_encoding: bool = True
    use_relative_attention: bool = True


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for proof steps."""
    
    def __init__(self, hidden_dim: int, max_steps: int = 100):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_steps = max_steps
        
        # Create positional encoding matrix
        pe = torch.zeros(max_steps, hidden_dim)
        position = torch.arange(0, max_steps, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * 
                           (-math.log(10000.0) / hidden_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor, step_numbers: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_nodes, hidden_dim] node features
            step_numbers: [batch_size, num_nodes] step numbers
        
        Returns:
            [batch_size, num_nodes, hidden_dim] features with positional encoding
        """
        batch_size, num_nodes, hidden_dim = x.shape
        
        # Get positional encodings for each step
        pos_encodings = self.pe[0, step_numbers]  # [batch_size, num_nodes, hidden_dim]
        
        # Add positional encoding
        return x + pos_encodings


class RelativePositionalEncoding(nn.Module):
    """Relative positional encoding for proof steps."""
    
    def __init__(self, hidden_dim: int, max_relative_distance: int = 50):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_relative_distance = max_relative_distance
        
        # Learnable relative position embeddings
        self.relative_embeddings = nn.Embedding(
            2 * max_relative_distance + 1, hidden_dim
        )
    
    def forward(self, step_numbers: torch.Tensor) -> torch.Tensor:
        """
        Args:
            step_numbers: [batch_size, num_nodes] step numbers
        
        Returns:
            [batch_size, num_nodes, num_nodes, hidden_dim] relative position embeddings
        """
        batch_size, num_nodes = step_numbers.shape
        
        # Compute relative distances
        step_expanded = step_numbers.unsqueeze(2)  # [batch_size, num_nodes, 1]
        step_expanded_t = step_numbers.unsqueeze(1)  # [batch_size, 1, num_nodes]
        
        relative_distances = step_expanded - step_expanded_t  # [batch_size, num_nodes, num_nodes]
        
        # Clip to valid range
        relative_distances = torch.clamp(
            relative_distances, 
            -self.max_relative_distance, 
            self.max_relative_distance
        )
        
        # Shift to positive indices
        relative_distances = relative_distances + self.max_relative_distance
        
        # Get embeddings
        relative_embeddings = self.relative_embeddings(relative_distances)
        
        return relative_embeddings


class LearnedProofAttention(nn.Module):
    """
    Learned attention mechanism for proof history.
    
    Key differences from fixed window approach:
    - Fixed window: "Only look at last 5 steps" (hand-coded prior)
    - Learned attention: "Learn which steps matter" (data-driven)
    """
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Positional encoding
        if config.use_positional_encoding:
            self.pos_encoding = PositionalEncoding(config.hidden_dim, config.max_steps)
        
        # Relative positional encoding
        if config.use_relative_attention:
            self.relative_pos_encoding = RelativePositionalEncoding(config.hidden_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim)
        )
        
        # Output projection
        self.output_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
    
    def forward(self, x: torch.Tensor, step_numbers: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, num_nodes, hidden_dim] node features
            step_numbers: [batch_size, num_nodes] step numbers
            attention_mask: [batch_size, num_nodes] optional attention mask
        
        Returns:
            attended_features: [batch_size, num_nodes, hidden_dim] attended features
            attention_weights: [batch_size, num_nodes, num_nodes] attention weights
        """
        batch_size, num_nodes, hidden_dim = x.shape
        
        # Add positional encoding
        if self.config.use_positional_encoding:
            x = self.pos_encoding(x, step_numbers)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = self._create_attention_mask(step_numbers)
        
        # Apply attention
        attended_features, attention_weights = self.attention(
            x, x, x, 
            key_padding_mask=attention_mask,
            need_weights=True
        )
        
        # Add relative positional information
        if self.config.use_relative_attention:
            relative_embeddings = self.relative_pos_encoding(step_numbers)
            # Incorporate relative information into attention weights
            attention_weights = attention_weights + 0.1 * relative_embeddings.mean(dim=-1)
        
        # Residual connection and layer norm
        attended_features = self.layer_norm(attended_features + x)
        
        # Feed-forward network
        ff_output = self.feed_forward(attended_features)
        attended_features = attended_features + ff_output
        
        # Final projection
        output = self.output_proj(attended_features)
        
        return output, attention_weights
    
    def _create_attention_mask(self, step_numbers: torch.Tensor) -> torch.Tensor:
        """
        Create attention mask based on step numbers.
        
        Unlike fixed window, this allows the model to learn which steps to attend to.
        """
        batch_size, num_nodes = step_numbers.shape
        
        # Create mask that allows attention to all valid steps
        # (No hard cutoff like fixed window)
        mask = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=step_numbers.device)
        
        # Only mask out invalid steps (e.g., padding)
        # This is much more permissive than fixed window
        return mask
    
    def get_attention_patterns(self, step_numbers: torch.Tensor) -> torch.Tensor:
        """Get attention patterns for visualization."""
        with torch.no_grad():
            # Create dummy input
            batch_size, num_nodes = step_numbers.shape
            dummy_input = torch.randn(batch_size, num_nodes, self.config.hidden_dim)
            
            # Get attention weights
            _, attention_weights = self.forward(dummy_input, step_numbers)
            
            return attention_weights


class MultiScaleProofAttention(nn.Module):
    """
    Multi-scale attention for proof history.
    
    Captures both local and global dependencies:
    - Local: Recent steps (like fixed window)
    - Global: Long-range dependencies (unlike fixed window)
    """
    
    def __init__(self, config: AttentionConfig, scales: List[int] = [1, 5, 10]):
        super().__init__()
        self.config = config
        self.scales = scales
        
        # Scale-specific attention layers
        self.scale_attentions = nn.ModuleList([
            LearnedProofAttention(config) for _ in scales
        ])
        
        # Scale fusion
        self.scale_fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * len(scales), config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
    
    def forward(self, x: torch.Tensor, step_numbers: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_nodes, hidden_dim] node features
            step_numbers: [batch_size, num_nodes] step numbers
        
        Returns:
            [batch_size, num_nodes, hidden_dim] multi-scale attended features
        """
        scale_outputs = []
        
        for scale, attention in zip(self.scales, self.scale_attentions):
            # Create scale-specific attention mask
            scale_mask = self._create_scale_mask(step_numbers, scale)
            
            # Apply attention
            scale_output, _ = attention(x, step_numbers, scale_mask)
            scale_outputs.append(scale_output)
        
        # Fuse scales
        combined_output = torch.cat(scale_outputs, dim=-1)
        fused_output = self.scale_fusion(combined_output)
        
        return fused_output
    
    def _create_scale_mask(self, step_numbers: torch.Tensor, scale: int) -> torch.Tensor:
        """Create attention mask for specific scale."""
        batch_size, num_nodes = step_numbers.shape
        mask = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=step_numbers.device)
        
        # For each node, only attend to nodes within scale distance
        for i in range(num_nodes):
            current_step = step_numbers[:, i]
            step_diff = torch.abs(step_numbers - current_step.unsqueeze(1))
            mask[:, i] = step_diff > scale
        
        return mask


class ProofHistoryEncoder(nn.Module):
    """
    Complete proof history encoder with learned attention.
    
    Replaces the fixed window approach with learned attention over full history.
    """
    
    def __init__(self, input_dim: int, config: AttentionConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, config.hidden_dim)
        
        # Multi-scale attention
        self.multi_scale_attention = MultiScaleProofAttention(config)
        
        # Temporal encoding layers
        self.temporal_layers = nn.ModuleList([
            LearnedProofAttention(config) for _ in range(config.num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
    
    def forward(self, x: torch.Tensor, step_numbers: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, num_nodes, input_dim] node features
            step_numbers: [batch_size, num_nodes] step numbers
        
        Returns:
            encoded_features: [batch_size, num_nodes, hidden_dim] encoded features
            attention_weights: [batch_size, num_nodes, num_nodes] attention weights
        """
        # Project input
        x = self.input_proj(x)
        
        # Multi-scale attention
        x = self.multi_scale_attention(x, step_numbers)
        
        # Temporal encoding layers
        attention_weights = None
        for layer in self.temporal_layers:
            x, attention_weights = layer(x, step_numbers)
        
        # Output projection
        output = self.output_proj(x)
        
        return output, attention_weights
    
    def get_attention_visualization(self, x: torch.Tensor, 
                                   step_numbers: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get attention patterns for visualization."""
        with torch.no_grad():
            # Get attention patterns from each layer
            attention_patterns = {}
            
            x = self.input_proj(x)
            x = self.multi_scale_attention(x, step_numbers)
            
            for i, layer in enumerate(self.temporal_layers):
                x, attn_weights = layer(x, step_numbers)
                attention_patterns[f'layer_{i}'] = attn_weights
            
            return attention_patterns


def create_learned_attention(input_dim: int, hidden_dim: int = 256, 
                           num_heads: int = 8) -> ProofHistoryEncoder:
    """Create learned attention mechanism."""
    config = AttentionConfig(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=6,
        max_steps=100
    )
    return ProofHistoryEncoder(input_dim, config)


def create_multi_scale_attention(input_dim: int, hidden_dim: int = 256) -> MultiScaleProofAttention:
    """Create multi-scale attention mechanism."""
    config = AttentionConfig(hidden_dim=hidden_dim)
    return MultiScaleProofAttention(config, scales=[1, 5, 10])


class AttentionVisualizer:
    """Visualize attention patterns for debugging and analysis."""
    
    def __init__(self, encoder: ProofHistoryEncoder):
        self.encoder = encoder
    
    def visualize_attention_patterns(self, x: torch.Tensor, 
                                   step_numbers: torch.Tensor) -> Dict[str, Any]:
        """Visualize attention patterns."""
        with torch.no_grad():
            # Get attention patterns
            attention_patterns = self.encoder.get_attention_visualization(x, step_numbers)
            
            # Analyze patterns
            analysis = {}
            for layer_name, attn_weights in attention_patterns.items():
                # Compute attention statistics
                analysis[layer_name] = {
                    'mean_attention': attn_weights.mean().item(),
                    'max_attention': attn_weights.max().item(),
                    'attention_entropy': self._compute_attention_entropy(attn_weights),
                    'attention_sparsity': self._compute_attention_sparsity(attn_weights)
                }
            
            return analysis
    
    def _compute_attention_entropy(self, attn_weights: torch.Tensor) -> float:
        """Compute entropy of attention distribution."""
        # Normalize attention weights
        attn_probs = F.softmax(attn_weights, dim=-1)
        
        # Compute entropy
        entropy = -(attn_probs * torch.log(attn_probs + 1e-8)).sum(dim=-1).mean()
        return entropy.item()
    
    def _compute_attention_sparsity(self, attn_weights: torch.Tensor) -> float:
        """Compute sparsity of attention weights."""
        # Count non-zero attention weights
        non_zero = (attn_weights > 1e-6).float().sum()
        total = attn_weights.numel()
        
        return (total - non_zero) / total
