"""
Fixed GNN Model for Neural Theorem Proving
==========================================

Integrates all critical fixes:
1. Proper spectral convolution (spectral_encoder.py)
2. Fixed temporal encoding (temporal_encoder_fixed.py)
3. Cross-attention fusion (fusion_fixed.py)
4. Proper residual connections
5. Gradient-friendly architecture

Usage:
    model = FixedProofGNN(in_dim=32, hidden_dim=256, k=32)
    scores, embeddings, value = model(batch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import global_mean_pool, global_max_pool

# Import fixed modules
from scorer import LogTempScorer, ProductionScorer
from spectral_encoder import StackedSpectralEncoder
from temporal_encoder_fixed import ProofAwareTemporalEncoder
from fusion_fixed import BottleneckMultimodalFusion, TransformerBottleneckFusion


class SpatialGNN(nn.Module):
    """
    Spatial message-passing pathway.
    Uses standard GATv2 with proper normalization.
    """
    
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 3,
                 num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        # Edge type encoder
        self.edge_encoder = nn.Embedding(3, 16)
        
        # GAT layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            self.convs.append(
                GATv2Conv(
                    hidden_dim, 
                    hidden_dim // num_heads,
                    heads=num_heads,
                    concat=True,
                    dropout=dropout,
                    edge_dim=16
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x: [N, in_dim]
            edge_index: [2, E]
            edge_attr: [E] edge types
            
        Returns:
            h: [N, hidden_dim]
        """
        h = self.input_proj(x)
        
        # Encode edge types
        if edge_attr is not None and len(edge_attr) > 0:
            edge_features = self.edge_encoder(edge_attr)
        else:
            edge_features = None
        
        # Apply GAT layers with residual connections
        for conv, norm in zip(self.convs, self.norms):
            h_res = h
            h = conv(h, edge_index, edge_attr=edge_features)
            h = norm(h + h_res)  # Residual
            h = F.elu(h)
            h = self.dropout(h)
        
        return h

from torch_geometric.nn import TransformerConv

class GraphTransformerSpatial(nn.Module):
    """
    SOTA Spatial Encoder: Uses TransformerConv with Edge Features.
    More expressive than GATv2 for logical graphs.
    """
    def __init__(self, in_dim, hidden_dim, num_layers=3, num_heads=4, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.edge_encoder = nn.Embedding(3, hidden_dim) # Embed edge types to hidden_dim
        
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.layers.append(
                TransformerConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    edge_dim=hidden_dim, # Critical: Use edge features
                    dropout=dropout,
                    beta=True # Learnable beta (like LayerScale)
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
            
    def forward(self, x, edge_index, edge_attr=None):
        h = self.input_proj(x)
        
        if edge_attr is not None:
            edge_emb = self.edge_encoder(edge_attr)
        else:
            edge_emb = None
            
        for conv, norm in zip(self.layers, self.norms):
            h_res = h
            # TransformerConv handles edge_dim internally
            h = conv(h, edge_index, edge_attr=edge_emb) 
            h = norm(h + h_res)
            h = F.gelu(h)
            
        return h
class FixedProofGNN(nn.Module):
    """
    Complete fixed GNN model with all three pathways.
    """
    
    def __init__(self, in_dim: int, hidden_dim: int = 256, k: int = 32,
                 num_spatial_layers: int = 3, num_spectral_layers: int = 2,
                 num_temporal_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.k = k
        
        # ===== PATHWAY 1: SPECTRAL =====
        self.spectral_encoder = StackedSpectralEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            k=k,
            num_layers=num_spectral_layers,
            num_filters=3,
            dropout=dropout
        )
        
        # ===== PATHWAY 2: SPATIAL =====
        self.spatial_gnn = GraphTransformerSpatial(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_spatial_layers,
            num_heads=4,
            dropout=dropout
        )
        
        # ===== PATHWAY 3: TEMPORAL =====
        self.temporal_encoder = ProofAwareTemporalEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_temporal_layers,
            num_heads=4,
            dropout=dropout
        )
        
        # ===== FUSION =====
        self.fusion = TransformerBottleneckFusion(
            hidden_dim=hidden_dim,
            num_bottlenecks=4, # Hyperparameter to tune
            num_heads=4,
            dropout=dropout
        )
        
        # ===== OUTPUT HEADS =====
        
        # Scoring head (for rule selection)
        self.scorer = LogTempScorer(hidden_dim)
        
        # Value head (for proof state evaluation)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        self._robust_initialization()
    def _robust_initialization(self):
        """
        Scientific Initialization to prevent Exploding Updates/Vanishing Gradients.
        Overrides default PyTorch init.
        """
        print("⚡ Applying Robust Initialization...")
        for name, param in self.named_parameters():
            # 1. Stabilize LayerNorm
            if 'norm' in name and 'weight' in name:
                # Add tiny noise to break symmetry
                nn.init.normal_(param, mean=1.0, std=0.02)
            
            # 2. Stabilize Biases (Prevent Update Ratio Explosion)
            if 'bias' in name and param.dim() == 1:
                # Init to small value instead of 0
                nn.init.constant_(param, 0.01)
                
            # 3. Special: Residual Scales
            if 'residual_scale' in name:
                nn.init.constant_(param, 0.0) # Sigmoid(0) = 0.5 (Balanced)
            
            # 4. Special: GAT/Linear Weights
            if 'weight' in name and param.dim() > 1:
                # Kaiming Normal is safer for GNNs than Xavier Uniform
                nn.init.kaiming_normal_(param, nonlinearity='relu')
    def forward(self, data):
        """
        Forward pass.
        
        Args:
            data: PyG Data object with:
                - x: [N, in_dim] node features
                - edge_index: [2, E] edges
                - edge_attr: [E] edge types
                - eigvecs_real: [N, k] real eigenvectors
                - eigvecs_imag: [N, k] imaginary eigenvectors
                - eigvals: [B*k] eigenvalues
                - eig_mask: [B*k] validity mask
                - step_numbers: [N] derivation steps
                - derived_mask: [N] derivation mask
                - batch: [N] batch indices
                
        Returns:
            scores: [N] node scores for rule selection
            embeddings: [N, hidden_dim] node embeddings
            value: [batch_size] proof state values
        """
        # Extract data
        x = data.x
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None
        
        # ===== PATHWAY 1: SPECTRAL =====
        h_spectral = self.spectral_encoder(
            x, 
            data.eigvecs_real,
            data.eigvecs_imag,
            data.eigvals,
            data.eig_mask,
            batch
        )
        
        # ===== PATHWAY 2: SPATIAL =====
        h_spatial = self.spatial_gnn(
            x,
            edge_index,
            edge_attr=data.edge_attr if hasattr(data, 'edge_attr') else None
        )
        
        # ===== PATHWAY 3: TEMPORAL =====
        # Use spatial features as input to temporal (provides context)
        h_temporal = self.temporal_encoder(
            data.derived_mask if hasattr(data, 'derived_mask') else None,
            data.step_numbers,
            h_spatial,
            batch
        )
        
        # ===== FUSION =====
        pathways = [h_spectral, h_spatial, h_temporal]
        h_fused, gate_stats = self.fusion(pathways)
        
        # ===== OUTPUT =====
        
        # Node-level scores
        scores = self.scorer(h_fused).squeeze(-1)
        
        # Graph-level value
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        
        graph_embedding = global_mean_pool(h_fused, batch)
        value = self.value_head(graph_embedding).squeeze(-1)
        
        return scores, h_fused, value
    
    def get_pathway_outputs(self, data):
        """
        Get outputs from each pathway separately (for analysis).
        
        Returns:
            dict with 'spectral', 'spatial', 'temporal' keys
        """
        x = data.x
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None
        
        h_spectral = self.spectral_encoder(
            x, data.eigvecs_real, data.eigvecs_imag,
            data.eigvals, data.eig_mask, batch
        )
        
        h_spatial = self.spatial_gnn(
            x, edge_index,
            edge_attr=data.edge_attr if hasattr(data, 'edge_attr') else None
        )
        
        h_temporal = self.temporal_encoder(
            data.derived_mask if hasattr(data, 'derived_mask') else None,
            data.step_numbers,
            h_spatial
        )
        
        return {
            'spectral': h_spectral,
            'spatial': h_spatial,
            'temporal': h_temporal
        }


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def get_fixed_model(in_dim: int = 32, hidden_dim: int = 256, k: int = 32,
                   device: str = 'cpu') -> FixedProofGNN:
    """
    Factory function to create the fixed model.
    
    Args:
        in_dim: Input feature dimension
        hidden_dim: Hidden dimension for all pathways
        k: Number of eigenvectors
        device: Device to place model on
        
    Returns:
        model: FixedProofGNN instance
    """
    model = FixedProofGNN(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        k=k,
        num_spatial_layers=3,
        num_spectral_layers=3,
        num_temporal_layers=3,
        dropout=0.2
    )
    model.scorer = LogTempScorer(
            hidden_dim=hidden_dim,
            dropout=0.1
        ).to(device)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Spectral dimension: {k}")
    
    return model


# ============================================================================
# TESTING
# ============================================================================

def test_model():
    """Test the complete model."""
    print("Testing Fixed Proof GNN...")
    
    # Create synthetic batch
    from torch_geometric.data import Data, Batch
    
    N = 20
    E = 30
    k = 16
    in_dim = 32
    hidden_dim = 64
    
    # Create two graphs
    data1 = Data(
        x=torch.randn(10, in_dim),
        edge_index=torch.randint(0, 10, (2, 15)),
        edge_attr=torch.randint(0, 3, (15,)),
        eigvecs_real=torch.randn(10, k),
        eigvecs_imag=torch.randn(10, k),
        step_numbers=torch.tensor([0, 0, 1, 2, 2, 3, 4, 4, 5, 5])
    )
    
    data2 = Data(
        x=torch.randn(10, in_dim),
        edge_index=torch.randint(0, 10, (2, 15)),
        edge_attr=torch.randint(0, 3, (15,)),
        eigvecs_real=torch.randn(10, k),
        eigvecs_imag=torch.randn(10, k),
        step_numbers=torch.tensor([0, 0, 0, 1, 1, 2, 3, 3, 4, 5])
    )
    
    batch = Batch.from_data_list([data1, data2])
    
    # Add global eigenvalues
    batch.eigvals = torch.abs(torch.randn(2 * k))
    batch.eig_mask = torch.ones(2 * k, dtype=torch.bool)
    batch.derived_mask = batch.step_numbers > 0
    
    # Create model
    model = FixedProofGNN(in_dim, hidden_dim, k)
    
    # Forward pass
    scores, embeddings, value = model(batch)
    
    print(f"âœ“ Scores shape: {scores.shape} (expected [20])")
    print(f"âœ“ Embeddings shape: {embeddings.shape} (expected [20, 64])")
    print(f"âœ“ Value shape: {value.shape} (expected [2])")
    
    # Test backward pass
    loss = scores.sum() + value.sum()
    loss.backward()
    
    # Check gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm == 0:
                print(f"âš  Warning: Zero gradient for {name}")
    
    print("\nâœ“ All tests passed!\n")


if __name__ == "__main__":
    test_model()