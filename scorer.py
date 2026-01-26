"""
SOTA Production Scorer (Fixed)
==============================
Fixes gradient collapse by:
1. Removing aggressive sqrt(dim) scaling (not needed for static query vectors).
2. Using GELU + LayerNorm (proven stability).
3. Boosting initialization variance.

Performance Goal:
- Gradient Norm: > 0.5 (Strong)
- Extreme Score Test: Pass (No collapse)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ProductionScorer(nn.Module):
    """
    Robust Scorer combining Residual MLP features with Bilinear Ranking.
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 1. Feature Refinement (Residual MLP)
        # Matches the 'Old Scorer' strength but adds residual path for depth
        self.res_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 2. Bilinear Scoring Head
        # score = x^T W q + b
        self.transform = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.query_vector = nn.Parameter(torch.randn(hidden_dim))
        
        # 3. Learnable Temperature (Start at 1.0, not 5.0)
        self.temperature = nn.Parameter(torch.ones(1))
        
        self._init_weights()
        self.last_grad_norm = 0.0
        self._register_hooks()

    def _init_weights(self):
        # 1. MLP Init
        nn.init.xavier_uniform_(self.res_block[0].weight)
        
        # 2. Bilinear Init (Boosted)
        # We want x^T W q to have variance ~1.0
        # Removing sqrt(dim) division means we need standard Xavier here
        nn.init.xavier_uniform_(self.transform.weight)
        # Init query vector with variance 1/sqrt(dim) to keep dot product stable
        nn.init.normal_(self.query_vector, std=1.0 / math.sqrt(self.hidden_dim))

    def _register_hooks(self):
        def hook(grad):
            self.last_grad_norm = grad.norm().item()
            return grad
        self.query_vector.register_hook(hook)

    def forward(self, x):
        # 1. Refine Features (Residual)
        h = x + self.res_block(x)
        
        # 2. Bilinear Projection
        h_trans = self.transform(h)
        
        # 3. Dot Product
        # Note: Removed /sqrt(dim) scaling to preserve gradient magnitude
        scores = torch.matmul(h_trans, self.query_vector)
        
        # 4. Bias & Temperature
        scores = scores / self.temperature.clamp(min=0.01, max=1.0)
        
        return scores

class LorentzianScorer(nn.Module):
    """
    Uses the Hyperboloid model (Lorentz model) which is numerically 
    superior to the Poincare ball for optimization, avoiding the boundary 
    collapse issues of atanh.
    """
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dense = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        # Learnable curvature
        self.beta = nn.Parameter(torch.tensor(1.0)) 

    def _to_hyperboloid(self, x):
        # Map R^d -> H^d (Hyperboloid)
        # x_0 = sqrt(1 + ||x_{1:d}||^2)
        k = torch.cat([torch.zeros_like(x[..., :1]), x], dim=-1) # [..., d+1]
        x_norm_sq = torch.norm(x, p=2, dim=-1, keepdim=True).pow(2)
        k[..., 0] = torch.sqrt(self.beta.abs() + x_norm_sq).squeeze(-1)
        return k

    def _lorentz_dist(self, u, v):
        # d(u,v) = -acosh( - <u, v>_L )
        # Minkowski inner product: -u0v0 + u1v1 + ... + undvnd
        prod = -u[..., 0]*v[..., 0] + (u[..., 1:] * v[..., 1:]).sum(dim=-1)
        
        # Clamp for numerical stability (acosh defined for x >= 1)
        prod = torch.clamp(prod, max=-1.0 - 1e-7)
        dist = torch.acosh(-prod)
        return dist

    def forward(self, node_embeddings, x_raw, batch):
        # 1. Project nodes to latent space
        h_nodes = self.dense(node_embeddings)
        
        # 2. Identify Goal embeddings (using x_raw feature 29)
        is_goal = (x_raw[:, 29] > 0.5).float().unsqueeze(-1)
        
        # Global Mean Pool of Goal Nodes per graph
        if batch is None: batch = torch.zeros(x_raw.size(0), dtype=torch.long, device=x_raw.device)
        from torch_geometric.utils import scatter
        
        # Sum goals, count goals, average
        goal_sum = scatter(h_nodes * is_goal, batch, dim=0, reduce='sum')
        goal_count = scatter(is_goal, batch, dim=0, reduce='sum').clamp(min=1.0)
        h_goals_graph = goal_sum / goal_count # [Batch, D]
        
        # Broadcast goal back to nodes
        h_goals_expanded = h_goals_graph[batch]
        
        # 3. Map to Hyperboloid
        u = self._to_hyperboloid(h_nodes)
        v = self._to_hyperboloid(h_goals_expanded)
        
        # 4. Compute Distance
        dist = self._lorentz_dist(u, v)
        
        # Score is negative distance (closer = higher score)
        return -dist


class LogTempScorer(nn.Module):
    """
    Scorer with numerically stable, learnable temperature.
    Score = (x @ q) * exp(-log_temp)
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # Feature processing
        self.res_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Bilinear Query Vector
        self.query_vector = nn.Parameter(torch.randn(hidden_dim))
        nn.init.normal_(self.query_vector, std=1.0 / math.sqrt(hidden_dim))
        
        # Learnable Log-Temperature (Init 0.0 -> Temp 1.0)
        self.log_temperature = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        h = x + self.res_block(x)
        
        # Dot product
        scores = torch.matmul(h, self.query_vector)
        
        # Scale
        # exp(-log_t) is always positive. No clamping needed.
        scores = scores * torch.exp(-self.log_temperature)
        
        return scores
# ============================================================================
# TESTING
# ============================================================================

def test_scorer_fix():
    print("="*60)
    print("FIXED SCORER GRADIENT TEST")
    print("="*60)
    
    hidden_dim = 256
    batch_size = 32
    
    # 1. Normal Inputs (Negative Mean)
    x = torch.randn(batch_size, hidden_dim) - 1.0
    x.requires_grad_(True)
    
    model = ProductionScorer(hidden_dim)
    scores = model(x)
    
    loss = (scores ** 2).mean()
    loss.backward()
    
    grad = model.last_grad_norm
    print(f"Normal Gradient:  {grad:.4f}  ", end="")
    if grad > 0.5: print("âœ“âœ“ STRONG")
    elif grad > 0.1: print("âš ï¸   OKAY")
    else: print("â Œ WEAK")
    
    # 2. Extreme Value Test
    print("-" * 60)
    print("EXTREME VALUE TEST")
    
    x_ex = torch.randn(batch_size, hidden_dim)
    x_ex[0] = x_ex[0] * 10.0 # Extreme outlier
    x_ex.requires_grad_(True)
    
    model.zero_grad()
    scores_ex = model(x_ex)
    loss_ex = (scores_ex ** 2).mean()
    loss_ex.backward()
    
    grad_ex = x_ex.grad.norm().item()
    print(f"Extreme Score:    {scores_ex[0].item():.4f}")
    print(f"Extreme Gradient: {grad_ex:.4f}")
    
    if grad_ex > 0.1:
        print("âœ“ SUCCESS: No gradient collapse.")
    else:
        print("â Œ FAILURE: Gradient collapse.")
    print("="*60)

if __name__ == "__main__":
    test_scorer_fix()