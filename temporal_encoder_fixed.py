"""
Fixed Causal Temporal Encoder for Neural Theorem Proving
=========================================================

Critical Fixes:
1. NO SORTING - maintains original graph structure
2. Proper causal masking based on derivation steps
3. Axiom connectivity (step=0 nodes can see each other)
4. Numerically stable RoPE implementation
5. Proper gradient flow through all pathways
6. REMOVED additive step embedding (conflicted with RoPE)
7. ADAPTIVE INPUT: Handles both Raw (32-dim) and Projected (256-dim) features.

Reference: GPT-3, Llama 2 (RoPE), T-PE (Temporal PE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 10000):
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even for RoPE"
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, step_numbers: torch.Tensor) -> torch.Tensor:
        t = step_numbers.float().unsqueeze(-1)
        freqs = t * self.inv_freq.unsqueeze(0)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, freqs: torch.Tensor) -> tuple:
    freqs = freqs.unsqueeze(0).unsqueeze(2)
    cos = freqs.cos()
    sin = freqs.sin()
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class CausalTemporalTransformerLayer(nn.Module):
    """Single causal transformer layer with RoPE."""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def _create_causal_mask(self, step_numbers, batch):
        step_i = step_numbers.unsqueeze(1)
        step_j = step_numbers.unsqueeze(0)
        causal_mask = (step_i < step_j) # Strict causality
        
        is_axiom = (step_numbers == 0)
        axiom_connectivity = is_axiom.unsqueeze(1) & is_axiom.unsqueeze(0)
        
        if batch is not None:
            batch_i = batch.unsqueeze(1)
            batch_j = batch.unsqueeze(0)
            diff_graph_mask = (batch_i != batch_j)
        else:
            diff_graph_mask = torch.zeros_like(causal_mask, dtype=torch.bool)
        
        final_mask = (causal_mask & ~axiom_connectivity) | diff_graph_mask
        return final_mask
    
    def forward(self, x, step_numbers, batch, rope):
        N = x.shape[0]
        x_norm = self.norm1(x)
        
        q = self.q_proj(x_norm).view(1, N, self.num_heads, self.head_dim)
        k = self.k_proj(x_norm).view(1, N, self.num_heads, self.head_dim)
        v = self.v_proj(x_norm).view(1, N, self.num_heads, self.head_dim)
        
        freqs = rope(step_numbers)
        q, k = apply_rotary_emb(q, k, freqs)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        mask = self._create_causal_mask(step_numbers, batch).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(N, self.hidden_dim)
        out = self.out_proj(out)
        out = self.dropout(out)
        
        x = x + out
        x = x + self.ff(self.norm2(x))
        return x


class ProofAwareTemporalEncoder(nn.Module):
    """
    Temporal encoder with:
    1. RoPE for relative position encoding
    2. Causal masking
    3. Adaptive Input Projection
    """
    
    def __init__(self, hidden_dim: int, max_steps: int = 100,
                 num_layers: int = 2, num_heads: int = 4,
                 dropout: float = 0.1, in_dim: int = 32):
        super().__init__()
        
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Projection layer for raw inputs
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        # RoPE
        self.rope = RotaryPositionalEmbedding(self.head_dim)
        
        self.layers = nn.ModuleList([
            CausalTemporalTransformerLayer(
                hidden_dim, num_heads, dropout
            ) for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, derived_mask, step_numbers, node_features, batch=None):
        # FIX: Adaptive Input Handling
        # If input is already projected (e.g. 256 dim), use it directly.
        # If input is raw (e.g. 32 dim), project it.
        if node_features.shape[-1] == self.hidden_dim:
            x = node_features
        else:
            x = self.input_proj(node_features)
        
        for layer in self.layers:
            x = layer(x, step_numbers, batch, self.rope)
        
        x = self.final_norm(x)
        return x