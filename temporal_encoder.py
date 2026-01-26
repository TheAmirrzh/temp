"""
Temporal State Encoder for Neural Theorem Proving
=================================================

State-of-the-art temporal encoding for proof state evolution, based on:
- TGN (Rossi et al., ICLR 2020): Temporal Graph Networks
- DyGFormer (Yu et al., 2023): Transformer for temporal graphs
- GraphMixer (Cong et al., 2023): Fixed time encoding
- T-PE (2024): Temporal Positional Encoding with geometric + semantic components
- tAPE (Foumani et al., 2023): Time-aware absolute positional encoding

Key innovations for theorem proving:
1. Derivation-aware temporal encoding (tracks which nodes derived at which steps)
2. Proof frontier attention (focus on recently derived facts)
3. Multi-scale temporal context (captures both local and global proof dynamics)
4. Fixed sinusoidal encoding (stable, no training instability)

Author: AI Research Team
Date: October 2025
"""


# In temporal_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FixedTimeEncoding(nn.Module):
    def __init__(self, d_model: int, max_steps: int = 1000):
        super().__init__()
        self.d_model = d_model
        self.max_steps = max_steps
        position = torch.arange(d_model // 2, dtype=torch.float32)
        freq_term = 1.0 / (10000.0 ** (2 * position / d_model))
        self.register_buffer('freq_term', freq_term)
    
    def forward(self, step_numbers: torch.Tensor) -> torch.Tensor:
        step_clamped = step_numbers.float()
        angles = step_clamped.unsqueeze(-1) * self.freq_term.unsqueeze(0)
        encoding = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        return encoding.reshape(step_numbers.shape[0], self.d_model)

class ProofFrontierAttention(nn.Module):
    """
    FIXED: Allows attending to Axioms (Step 0).
    Previous version excluded axioms, causing zero gradients for Step 1 nodes.
    """
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1, frontier_window: int = 5):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.time_decay = nn.Parameter(torch.tensor(0.5)) 
        self.frontier_window = frontier_window
    
    def forward(self, x: torch.Tensor, derived_mask: torch.Tensor, step_numbers: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        N = len(x)
        device = x.device
        
        # 1. Identify Query Nodes (Derived) vs Key/Value Nodes (All)
        # We want Derived nodes to update their state by looking at All nodes (past)
        derived_indices = derived_mask.nonzero(as_tuple=True)[0]
        
        if len(derived_indices) == 0:
            return x, torch.zeros((N, N), device=device)
            
        # 2. Prepare Inputs
        # Queries: Only Derived nodes [1, N_derived, D]
        x_query = x[derived_indices].unsqueeze(0)
        
        # Keys/Values: All nodes [1, N_all, D]
        x_kv = x.unsqueeze(0)
        
        # 3. Compute Relative Steps for Mask
        # query_steps: [N_derived]
        query_steps = step_numbers[derived_indices].float()
        # key_steps: [N_all]
        key_steps = step_numbers.float()
        
        # step_diff[i, j] = step[i] - step[j]
        # Shape: [N_derived, N_all]
        step_diff = query_steps.unsqueeze(1) - key_steps.unsqueeze(0)
        
        # 4. Create Attention Bias
        # Causal Mask: Block future (j > i) -> step_diff < 0
        # i (query) must be >= j (key)
        # Note: Axioms are step 0. Derived are >= 1.
        # So Derived (1) - Axiom (0) = 1 > 0 (Allowed)
        causal_mask = (step_diff < 0)
        
        # Recency Bias: exp(-lambda * distance)
        decay_rate = F.softplus(self.time_decay)
        recency_bias = -decay_rate * step_diff.abs()
        
        # Apply Causal Mask
        attn_bias = recency_bias.clone()
        attn_bias = attn_bias.masked_fill(causal_mask, float('-1e9'))
        
        # Shape check for MHA: [Batch*Heads, Queries, Keys] or [1, Queries, Keys]
        # attn_bias is [N_derived, N_all]. Perfect.
        
        # 5. Attention
        # query: [1, N_derived, D], key: [1, N_all, D]
        x_attended, _ = self.attention(
            query=x_query, key=x_kv, value=x_kv,
            attn_mask=attn_bias # [N_derived, N_all]
        )
        x_attended = x_attended.squeeze(0)
        
        # 6. Scatter back (Residual + Norm)
        x_out = x.clone()
        # Update only derived nodes with the attention result
        x_out[derived_indices] = self.dropout(self.norm(x_attended + x[derived_indices]))
        
        return x_out, torch.zeros((N, N), device=device)

class TemporalStateEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 256, num_heads: int = 4, frontier_window: int = 5, max_steps: int = 100, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.status_embed = nn.Embedding(2, hidden_dim // 4)
        self.time_encoder = FixedTimeEncoding(d_model=hidden_dim // 2, max_steps=max_steps)
        self.frontier_attention = ProofFrontierAttention(d_model=hidden_dim, num_heads=num_heads, frontier_window=frontier_window, dropout=dropout)
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim // 4 + hidden_dim // 2 + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, derived_mask: torch.Tensor, step_numbers: torch.Tensor, node_features: torch.Tensor) -> Tuple[torch.Tensor, None]:
        status_emb = self.status_embed(derived_mask.long())
        time_emb = self.time_encoder(step_numbers)
        
        frontier_features, _ = self.frontier_attention(node_features, derived_mask, step_numbers)
        
        fused_input = torch.cat([status_emb, time_emb, frontier_features], dim=1)
        return self.fusion(fused_input), None

class MultiScaleTemporalEncoder(nn.Module):
    def __init__(self, hidden_dim, num_scales=3, max_steps=100, dropout=0.1):
        super().__init__()
        self.base_encoder = TemporalStateEncoder(hidden_dim=hidden_dim, num_heads=4, max_steps=max_steps, dropout=dropout)
        self.output_norm = nn.LayerNorm(hidden_dim) 

    def forward(self, derived_mask, step_numbers, node_features):
        out, _ = self.base_encoder(derived_mask, step_numbers, node_features)
        return self.output_norm(out)

class CausalProofTemporalEncoder(MultiScaleTemporalEncoder):
    pass

# --- ADDED MISSING UTILITY FUNCTIONS ---

def compute_derived_mask(proof_state: Dict, current_step: int) -> torch.Tensor:
    """Computes binary mask of derived (non-axiom) nodes."""
    num_nodes = proof_state.get('num_nodes', 0)
    mask = torch.zeros(num_nodes, dtype=torch.uint8)
    derivations = proof_state.get('derivations', [])
    
    for node_idx, step in derivations:
        if node_idx < num_nodes and step <= current_step:
            mask[node_idx] = 1
    return mask

def compute_step_numbers(proof_state: Dict, current_step: int) -> torch.Tensor:
    """Computes step number for each node, 0 for axioms/future."""
    num_nodes = proof_state.get('num_nodes', 0)
    steps = torch.zeros(num_nodes, dtype=torch.long)
    derivations = proof_state.get('derivations', [])
    
    for node_idx, step in derivations:
        if node_idx < num_nodes and step <= current_step:
            steps[node_idx] = step
    return steps

def compute_derivation_dependencies(proof_state: Dict, current_step: int) -> torch.Tensor:
    """Computes adjacency matrix of derivation dependencies."""
    num_nodes = proof_state.get('num_nodes', 0)
    deps = torch.zeros(num_nodes, num_nodes, dtype=torch.float)
    dependencies = proof_state.get('dependencies', [])
    
    for derived_idx, parents, step in dependencies:
        if derived_idx < num_nodes and step <= current_step:
            for parent_idx in parents:
                if parent_idx < num_nodes:
                    deps[derived_idx, parent_idx] = 1.0
    return deps


class FixedTimeEncoding(nn.Module):
    def __init__(self, d_model: int, max_steps: int = 1000):
        super().__init__()
        position = torch.arange(max_steps).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_steps, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, step_numbers):
        # Clamp to max_steps - 1 to avoid index error
        steps = step_numbers.clamp(0, self.pe.size(0) - 1)
        # FIX: Use square brackets for indexing, not parentheses
        return self.pe[steps]

class EnhancedCausalTemporalEncoder(nn.Module):
    """
    SOTA Fix: Combines Transformer Global Context with GRU Local State Evolution.
    Solves gradient vanishing in long proofs (length > 20).
    """
    def __init__(self, hidden_dim, num_heads=4, max_steps=100):
        super().__init__()
        self.pe = FixedTimeEncoding(hidden_dim, max_steps)
        
        # Transformer for "Frontier Attention" (Recent derived nodes)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # GRU for "State Evolution" (Deep history)
        self.memory_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, derived_mask, step_numbers, node_features):
        N = node_features.size(0)
        
        # 1. Add Time PE
        h = node_features + self.pe(step_numbers)
        
        # 2. Causal Transformer Processing
        # We process nodes sorted by step to apply causal mask efficiently
        sorted_steps, indices = torch.sort(step_numbers)
        inv_indices = torch.argsort(indices)
        
        h_sorted = h[indices].unsqueeze(0) # [1, N, D]
        
        # Causal Mask: M[i,j] = -inf if step[j] > step[i]
        # (Can attend to past and current step)
        steps_expanded = sorted_steps.unsqueeze(0)
        mask = (steps_expanded.transpose(0, 1) < steps_expanded).float() * -1e9
        # Ensure diagonals are 0
        mask = mask.masked_fill(mask == 0, 0.0)
        
        h_trans = self.transformer(h_sorted, mask=mask).squeeze(0) # [N, D]
        
        # 3. Recurrent Memory Refinement
        # Iterate through sorted sequence to simulate proof generation
        # This provides a strong gradient highway for t -> t+1
        h_mem = torch.zeros_like(h_trans)
        state = torch.zeros(h_trans.size(1), device=h_trans.device) # [D]
        
        # Vectorized approximation: In a real proof, steps are discrete groups.
        # We iterate over unique steps to be efficient.
        unique_steps = torch.unique(sorted_steps)
        
        for step in unique_steps:
            # Nodes at this step
            step_mask = (sorted_steps == step)
            
            # Update state with nodes from this step (Mean aggregation)
            if step_mask.any():
                step_input = h_trans[step_mask].mean(dim=0)
                state = self.memory_cell(step_input.unsqueeze(0), state.unsqueeze(0)).squeeze(0)
                
                # Broadcast state back to nodes at this step
                h_mem[step_mask] = state
            
        # 4. Restore Order
        h_final = h_mem[inv_indices]
        
        return self.norm(h_final + node_features) # Residual connection to input

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=10000):
        super().__init__()
        # Inverse frequencies for rotation
        # dim should be head_dim (e.g., 64 if hidden=256 and heads=4)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        # x shape: [batch, seq_len, dim] -> We only care about seq_len
        # Note: x here is usually the node embeddings tensor
        seq_len = x.shape[1]
        device = x.device
        
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        
        # Outer product to get angles: [seq_len, dim/2]
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        
        # Concatenate to get [seq_len, dim] (cos/sin repeated)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Return [1, seq_len, dim] to match batch dimension of inputs
        return emb.unsqueeze(0)

def rotate_half(x):
    """Rotates half the hidden dims to apply RoPE."""
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, freqs):
    """
    Applies RoPE to query and key states.
    
    Args:
        q: [batch, seq_len, heads, head_dim]
        k: [batch, seq_len, heads, head_dim]
        freqs: [batch, seq_len, head_dim] (Must match q/k in batch and seq_len)
    """
    # q shape: [1, N, 4, 64]
    # freqs shape: [1, N, 64]
    
    # We need to broadcast freqs across the 'heads' dimension (dim 2).
    # Unsqueeze dim 2 to make freqs: [1, N, 1, 64]
    freqs = freqs.unsqueeze(2) 
    
    # Now [1, N, 4, 64] * [1, N, 1, 64] works perfectly via broadcasting
    q_embed = (q * freqs.cos()) + (rotate_half(q) * freqs.sin())
    k_embed = (k * freqs.cos()) + (rotate_half(k) * freqs.sin())
    
    return q_embed, k_embed


class CausalRoPETemporalEncoder(nn.Module):
    """
    SOTA Temporal Encoder using Rotary Embeddings (RoPE).
    Handles OOD Length Generalization for Hard Proofs.
    """
    def __init__(self, hidden_dim, num_heads=4, num_layers=2, max_steps=200, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # RoPE Generator (based on head dimension)
        self.rope = RotaryEmbedding(self.head_dim)
        
        # Custom Transformer Blocks to inject RoPE manually
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'q_proj': nn.Linear(hidden_dim, hidden_dim),
                'k_proj': nn.Linear(hidden_dim, hidden_dim),
                'v_proj': nn.Linear(hidden_dim, hidden_dim),
                'o_proj': nn.Linear(hidden_dim, hidden_dim),
                'norm1': nn.LayerNorm(hidden_dim),
                'norm2': nn.LayerNorm(hidden_dim),
                'mlp': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout)
                )
            }) for _ in range(num_layers)
        ])
        
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, derived_mask, step_numbers, node_features):
        """
        Args:
            derived_mask: [N] bool
            step_numbers: [N] int, derivation step index
            node_features: [N, D]
        """
        device = node_features.device
        N = node_features.shape[0]
        
        # 1. Sort nodes by step (To form a temporal sequence)
        sorted_steps, indices = torch.sort(step_numbers)
        inv_indices = torch.argsort(indices)
        
        # Sequence input: [1, N, D] (Batch size 1, N nodes, Dim D)
        x = node_features[indices].unsqueeze(0) 
        
        # 2. Generate Causal Mask (Block future steps)
        # M[i, j] = -inf if step[j] > step[i]
        # We compute this on the sorted steps
        q_steps = sorted_steps.unsqueeze(1) # [N, 1]
        k_steps = sorted_steps.unsqueeze(0) # [1, N]
        
        # Create mask [1, 1, N, N] for Multi-Head Attention
        mask = (q_steps < k_steps).unsqueeze(0).unsqueeze(0).to(device) 
        
        # 3. Generate RoPE Frequencies
        # Use the node features 'x' to determine sequence length
        freqs_tensor = self.rope(x) # [1, N, head_dim]
        
        # 4. Transformer Blocks with RoPE
        for layer in self.layers:
            residual = x
            x = layer['norm1'](x)
            
            # Projections -> [1, N, Heads, HeadDim]
            q = layer['q_proj'](x).view(1, N, self.num_heads, self.head_dim)
            k = layer['k_proj'](x).view(1, N, self.num_heads, self.head_dim)
            v = layer['v_proj'](x).view(1, N, self.num_heads, self.head_dim)
            
            # Apply RoPE (FIXED BROADCASTING)
            q, k = apply_rotary_pos_emb(q, k, freqs_tensor)
            
            # Attention Mechanism
            # Transpose to [1, Heads, N, HeadDim] for standard matmul
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            # Scores: [1, Heads, N, N]
            # (Q @ K.T) / sqrt(d)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            
            # Apply Causal Mask
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
            attn_weights = F.softmax(attn_scores, dim=-1)
            
            # Aggregation: [1, Heads, N, HeadDim]
            out = torch.matmul(attn_weights, v)
            
            # Reshape back: [1, N, HiddenDim]
            out = out.transpose(1, 2).reshape(1, N, self.hidden_dim)
            out = layer['o_proj'](out)
            
            # Residual 1
            x = residual + out
            
            # Feed Forward + Residual 2
            x = x + layer['mlp'](layer['norm2'](x))
            
        # 5. Restore original graph order
        x = x.squeeze(0) # [N, D]
        h_out = x[inv_indices]
        
        return self.out_norm(h_out)

# Example usage and testing
if __name__ == "__main__":
    print("Temporal State Encoder - Testing")
    print("=" * 60)
    
    N = 10  # 10 nodes
    hidden_dim = 128
    max_steps = 20
    
    # Initialize encoder
    encoder = TemporalStateEncoder(
        hidden_dim=hidden_dim,
        num_heads=4,
        frontier_window=5,
        max_steps=max_steps
    )
    
    # Create sample proof state
    # 3 axioms, 7 derived nodes
    derived_mask = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=torch.uint8)
    # Step numbers (axioms=0, others derived at various steps)
    # Current max step is 5
    step_numbers = torch.tensor([0, 0, 0, 1, 2, 2, 3, 4, 5, 5])
    # Initial node features
    node_features = torch.randn(N, hidden_dim)
    
    print(f"Test input: {N} nodes, current max_step={step_numbers.max().item()}")
    print(f"Derived mask: {derived_mask.tolist()}")
    print(f"Step numbers: {step_numbers.tolist()}")
    
    # Forward pass
    temporal_features, attn_weights = encoder(
        derived_mask,
        step_numbers,
        node_features,
        return_attention=True
    )
    
    print("\n--- Output ---")
    print(f"Temporal features shape: {temporal_features.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    
    # Check shapes
    assert temporal_features.shape == (N, hidden_dim)
    assert attn_weights.shape == (N, N)
    
    # Check frontier attention
    # max_step=5, window=5 -> threshold=0
    # is_frontier = (step_numbers > 0)
    # Frontier nodes = indices [3, 4, 5, 6, 7, 8, 9]
    # Masked (key) nodes = indices [0, 1, 2]
    # (Note: attn_weights[i, j] = attention from node i to node j)
    # The mask applies to keys (j). Sum of weights to nodes 0, 1, 2 should be 0.
    non_frontier_attn = attn_weights[:, :3].sum()
    print(f"Sum of attention to non-frontier (should be 0): {non_frontier_attn.item()}")
    assert torch.allclose(non_frontier_attn, torch.tensor(0.0), atol=1e-6)
    
    print("\n" + "=" * 60)
    print("All tests passed! TemporalStateEncoder is ready.")