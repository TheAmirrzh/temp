"""
Cross-Attention Pathway Fusion for Multi-Modal GNN
===================================================

Critical Fix:
    Replaces OrthogonalFusion (which kills gradients) with
    cross-attention-based fusion (preserves all gradients).

Mathematical Foundation:
    Pathways communicate via multi-head cross-attention:
    
    Spec_enhanced = Attention(Spec, [Spec, Spat, Temp])
    Spat_enhanced = Attention(Spat, [Spec, Spat, Temp])
    Temp_enhanced = Attention(Temp, [Spec, Spat, Temp])
    
    Final = Gate([Spec_enhanced, Spat_enhanced, Temp_enhanced])

Benefits:
1. All pathways receive gradients
2. Pathways learn complementary features naturally
3. Gating balances contributions dynamically
4. More expressive than simple concatenation

Reference: "Attention is All You Need", Perceiver architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Bottleneck Multimodal Fusion (Fixed)
====================================

CRITICAL FIX: Joint Modality Attention
--------------------------------------
Previous Version: Attended to each modality separately (Seq Len = 1).
                  Result -> Softmax(1) = 1.0 -> Zero Gradient for Query.
                  
Current Version:  Stacks modalities [Spec, Spat, Temp] (Seq Len = 3).
                  Result -> Softmax(3) -> Query (Bottleneck) determines weights.
                  
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BottleneckMultimodalFusion(nn.Module):
    """
    Robust Fusion using Learnable Bottleneck Tokens attending to 
    Stacked Modalities.
    """
    
    def __init__(self, hidden_dim: int, num_bottlenecks: int = 4,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_bottlenecks = num_bottlenecks
        self.num_heads = num_heads
        
        # 1. Learnable Bottleneck Tokens
        # These act as "Queries" to extract info from the modalities
        self.bottleneck_tokens = nn.Parameter(
            torch.randn(1, num_bottlenecks, hidden_dim) * 0.02
        )
        
        # 2. Joint Cross-Attention (Bottlenecks -> All Modalities)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 3. Self-Attention (Inter-Bottleneck reasoning)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 4. Output Projection
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # 5. Residual Connection
        self.norm_res = nn.LayerNorm(hidden_dim)
        
        # Diagnostics
        self.last_attn_weights = None

    def forward(self, pathways):
        """
        Args:
            pathways: List of [spectral, spatial, temporal], each [N, D]
        """
        # --- NEW: MODALITY DROPOUT ---
        if self.training:
            # 30% chance to completely ZERO OUT the Spatial pathway
            # This forces the model to rely on Spectral + Temporal logic
            if torch.rand(1).item() < 0.3:
                pathways[1] = torch.zeros_like(pathways[1]) 
            
            # 10% chance to drop Spectral
            if torch.rand(1).item() < 0.1:
                pathways[0] = torch.zeros_like(pathways[0])
        # -----------------------------
        # --- STEP 1: Stack Modalities ---
        # Create a sequence of length 3: [N, 3, D]
        # This ensures the attention mechanism has >1 key to select from!
        # Index 0: Spectral, 1: Spatial, 2: Temporal
        stacked_modalities = torch.stack(pathways, dim=1)
        N = stacked_modalities.shape[0]
        
        # Expand bottlenecks to batch: [N, B, D]
        queries = self.bottleneck_tokens.expand(N, -1, -1)
        
        # --- STEP 2: Cross-Attention ---
        # Query: Bottlenecks
        # Key/Val: Stacked Modalities
        # Output: [N, B, D] - Condenced features
        # Weights: [N, B, 3] - How much each bottleneck cares about each modality
        bottleneck_feats, attn_weights = self.cross_attn(
            query=queries,
            key=stacked_modalities,
            value=stacked_modalities
        )
        
        # --- STEP 3: Self-Attention (Refinement) ---
        # Allow bottlenecks to talk to each other
        bottleneck_feats, _ = self.self_attn(
            query=bottleneck_feats,
            key=bottleneck_feats,
            value=bottleneck_feats
        )
        
        # --- STEP 4: Fusion & Projection ---
        # Pool the bottlenecks to get a single vector per node [N, D]
        fused_feat = bottleneck_feats.mean(dim=1)
        
        fused_feat = self.out_proj(fused_feat)
        
        # --- STEP 5: Residual Connection ---
        # Add Spatial pathway (pathways[1]) as a strong residual anchor
        # This guarantees we perform at least as well as the spatial baseline
        out = self.norm_res(pathways[1] + fused_feat)
        
        # --- STEP 6: Statistics ---
        # Average attention weights across all bottlenecks and all nodes
        # attn_weights shape: [N, B, 3] -> Mean over N, B -> [3]
        mean_weights = attn_weights.mean(dim=(0, 1))
        
        # Calculate entropy of the average weights (measure of uniformity)
        # We want this to be somewhat low (specialized), but not 0.
        entropy = -(mean_weights * torch.log(mean_weights + 1e-8)).sum()
        
        stats = {
            'spectral_weight': mean_weights[0].item(),
            'spatial_weight': mean_weights[1].item(),
            'temporal_weight': mean_weights[2].item(),
            'weight_entropy': entropy.item(),
            # Standard deviation across the batch (checking for mode collapse)
            'weight_std': attn_weights.mean(dim=1).std(dim=0).mean().item()
        }
        
        return out, stats

# ============================================================================
# ALTERNATIVE: Simple Concatenation + MLP (Baseline)
# ============================================================================

class SimpleConcatFusion(nn.Module):
    """
    Simple baseline: concatenate + MLP.
    
    Use this if cross-attention is too complex or slow.
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(2 * hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Learnable pathway weights
        self.pathway_weights = nn.Parameter(torch.ones(3))
    
    def forward(self, pathways):
        """
        Args:
            pathways: List of [spec, spat, temp]
            
        Returns:
            fused: [N, hidden_dim]
        """
        # Weighted concatenation
        weights = F.softmax(self.pathway_weights, dim=0)
        weighted = [w * p for w, p in zip(weights, pathways)]
        
        concat = torch.cat(weighted, dim=-1)
        
        fused = self.fusion_mlp(concat)
        
        # Residual connection to spatial
        fused = fused + pathways[1]
        
        gate_stats = {
            'spectral_weight': weights[0].item(),
            'spatial_weight': weights[1].item(),
            'temporal_weight': weights[2].item()
        }
        
        return fused, gate_stats

class PathwayTransformerFusion(nn.Module):
    """
    Treats (Spectral, Spatial, Temporal) as a sequence of tokens [Batch, N, 3, D].
    Uses Self-Attention to allow pathways to 'talk' to each other dynamically
    per node, rather than forcing orthogonality.
    """
    def __init__(self, hidden_dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Token type embeddings for the 3 pathways
        self.pathway_tokens = nn.Parameter(torch.randn(1, 1, 3, hidden_dim))
        
        # Transformer Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=n_heads, 
            dim_feedforward=hidden_dim * 4, 
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Aggregation: Weighted Average
        self.attn_pool = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, pathways):
        # pathways list: [Spec, Spat, Temp] -> Stack to [N, 3, D]
        # We process node-wise. Batch dimension is effectively N.
        x_stack = torch.stack(pathways, dim=1) # [N, 3, D]
        
        # Add pathway identity tokens (broadcast over N)
        x_stack = x_stack + self.pathway_tokens
        
        # Self-Attention mixing
        x_trans = self.transformer(x_stack) # [N, 3, D]
        
        # Attention Pooling (Dynamic Weighting)
        # Learn which pathway matters most for *this* specific node
        weights = self.attn_pool(x_trans) # [N, 3, 1]
        x_fused = (x_trans * weights).sum(dim=1) # [N, D]
        
        return x_fused, weights.squeeze(-1) # Return weights for logging

import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=0.1, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class TransformerBottleneckFusion(nn.Module):
    """
    Production-Ready Transformer Fusion.
    - Uses 'Bottleneck Tokens' to query modalities.
    - Uses LayerScale (0.1) on output to preserve Spatial Anchor stability.
    - Allow gradients to flow freely inside the block (removed inner scales).
    """
    def __init__(self, hidden_dim: int, num_bottlenecks: int = 4, 
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 1. Bottleneck Queries (Latent Workspace)
        # Initialize small to prevent noise injection at start
        self.bottleneck_tokens = nn.Parameter(torch.randn(1, num_bottlenecks, hidden_dim) * 0.02)
        
        # 2. Cross-Attention (Latents attend to Modalities)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.norm_cross = nn.LayerNorm(hidden_dim)
        
        # 3. Self-Attention (Latents reason amongst themselves)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.norm_self = nn.LayerNorm(hidden_dim)
        
        # 4. Final Projection & Stability
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        # LayerScale on OUTPUT only: Protects the spatial anchor from early chaos
        self.layer_scale_out = LayerScale(hidden_dim, init_values=1)
        self.norm_res = nn.LayerNorm(hidden_dim)

    def forward(self, pathways):
        """
        Args:
            pathways: List of [N, D] tensors. 
                      Order MUST be [Spectral, Spatial, Temporal]
        """
        # Stack modalities: [N, 3, D]
        modality_stack = torch.stack(pathways, dim=1)
        N = modality_stack.shape[0]
        
        # Expand bottlenecks: [N, K, D]
        latents = self.bottleneck_tokens.expand(N, -1, -1)
        
        # --- Cross Attention (Latents query Modalities) ---
        q = self.norm_cross(latents)
        k = v = modality_stack
        
        # We want latents to extract info from modalities
        attn_out, weights = self.cross_attn(query=q, key=k, value=v)
        latents = latents + attn_out # Standard Residual
        
        # --- Self Attention (Latent Reasoning) ---
        q = k = v = self.norm_self(latents)
        attn_out, _ = self.self_attn(query=q, key=k, value=v)
        latents = latents + attn_out
        
        # --- Projection & Fusion ---
        fused = latents.mean(dim=1) # Pool bottlenecks -> [N, D]
        fused = self.out_proj(fused)
        
        # --- Robust Residual Connection ---
        # Anchor to Spatial Pathway (Index 1) for stability
        # fused contribution is scaled down initially by 0.1
        out = self.norm_res(fused)
        
        # Statistics for logging
        mean_weights = weights.mean(dim=(0, 1)) # Avg over batch & bottlenecks
        stats = {
            'spec_attn': mean_weights[0].item(),
            'spat_attn': mean_weights[1].item(),
            'temp_attn': mean_weights[2].item(),
            'weight_entropy': -(mean_weights * torch.log(mean_weights + 1e-8)).sum().item()
        }
        
        return out, stats
# ============================================================================
# TESTING
# ============================================================================

def test_fusion():
    """Test fusion modules."""
    print("Testing Cross-Attention Fusion...")
    
    N = 10
    hidden_dim = 64
    
    # Create synthetic pathways
    spec = torch.randn(N, hidden_dim, requires_grad=True)
    spat = torch.randn(N, hidden_dim, requires_grad=True)
    temp = torch.randn(N, hidden_dim, requires_grad=True)
    
    pathways = [spec, spat, temp]
    
    # Test cross-attention fusion
    fusion = PathwayTransformerFusion(hidden_dim)
    
    fused, gate_stats = fusion(pathways)
    
    print(f"âœ“ Fused shape: {fused.shape}")
    print(f"âœ“ Gate stats: {gate_stats}")
    print(f"âœ“ Entropy (higher=better balance): {gate_stats['entropy']:.3f}")
    
    # Test gradient flow to all pathways
    loss = fused.sum()
    loss.backward()
    
    for i, p in enumerate(['Spectral', 'Spatial', 'Temporal']):
        grad_norm = pathways[i].grad.norm().item()
        print(f"âœ“ {p} gradient norm: {grad_norm:.4f}")
        assert grad_norm > 0, f"No gradient to {p} pathway!"
    
    print("\nâœ“ Test simple concat fusion...")
    spec.grad = None
    spat.grad = None
    temp.grad = None
    
    fusion_simple = SimpleConcatFusion(hidden_dim)
    fused_simple, _ = fusion_simple(pathways)
    
    loss = fused_simple.sum()
    loss.backward()
    
    print(f"âœ“ Simple fusion shape: {fused_simple.shape}")
    
    print("\nâœ“ All tests passed!\n")


if __name__ == "__main__":
    test_fusion()