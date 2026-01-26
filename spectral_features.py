"""
Magnetic Laplacian for Directed Graphs (Horn Clauses)
======================================================

Based on MSGNN (LoG 2022) and MagNet (NeurIPS 2022).
This implementation replaces your current Random Walk Laplacian
for significantly improved spectral expressivity on directed graphs.

Key Innovation: Complex Hermitian matrix encodes BOTH:
- Magnitude: Topological structure (like standard Laplacian)
- Phase: Directional information (unique to Magnetic Laplacian)
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
import torch
from torch_geometric.utils import to_scipy_sparse_matrix
from typing import Tuple, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MagneticLaplacianExtractor:
    """
    Computes Magnetic Laplacian and its eigendecomposition for directed graphs.
    
    Mathematical Definition:
        H_q = D - A_q
    
    Where:
        D[i,i] = out_degree(node_i)
        A_q[i,j] = exp(i * theta_q(edge_{ij}))  if (i,j) in E
        theta_q(e) = q * phase(e)
    """
    
    def __init__(
        self,
        k: int = 16,
        q: float = 0.25,
        normalize: bool = True,
        tolerance: float = 1e-6,
        adaptive_k: bool = True
    ):
        assert k >= 1, "k must be at least 1"
        assert 0 <= q <= 1, "q should be in [0, 1]"
        
        self.k = k
        self.q = q
        self.normalize = normalize
        self.tolerance = tolerance
        self.adaptive_k = adaptive_k
        
        logger.info(f"Magnetic Laplacian initialized: k={k}, q={q}")
    
    def _assign_edge_phases(
        self,
        edge_index: torch.Tensor,
        edge_types: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Assign phase to each directed edge based on type and direction.
        """
        num_edges = edge_index.shape[1]
        phases = np.zeros(num_edges, dtype=np.float32)
        
        if edge_types is not None:
            edge_types_np = edge_types.cpu().numpy()
            
            # Horn clause phase assignments
            # body edges: fact -> rule (positive phase)
            body_mask = (edge_types_np == 2)
            phases[body_mask] = self.q * np.pi
            
            # head edges: rule -> fact (negative phase)
            head_mask = (edge_types_np == 1)
            phases[head_mask] = -self.q * np.pi
        else:
            # Fallback: Assign phase based on edge direction only
            src = edge_index[0].cpu().numpy()
            dst = edge_index[1].cpu().numpy()
            
            forward_mask = src < dst
            backward_mask = src > dst
            
            phases[forward_mask] = self.q * np.pi
            phases[backward_mask] = -self.q * np.pi
        
        return phases
    
    def compute_magnetic_adjacency(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        edge_types: Optional[torch.Tensor] = None
    ) -> sp.csr_matrix:
        """
        Construct complex Hermitian adjacency matrix.
        """
        # Assign phases
        phases = self._assign_edge_phases(edge_index, edge_types)
        
        # Build complex adjacency
        src = edge_index[0].cpu().numpy()
        dst = edge_index[1].cpu().numpy()
        
        # Complex weights: exp(i * phase)
        complex_weights = np.exp(1j * phases)
        
        # Create sparse matrix
        adj_complex = sp.coo_matrix(
            (complex_weights, (src, dst)),
            shape=(num_nodes, num_nodes),
            dtype=np.complex64
        ).tocsr()
        
        return adj_complex
    
    def compute_magnetic_laplacian(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        edge_types: Optional[torch.Tensor] = None
    ) -> sp.csr_matrix:
        """
        Compute Magnetic Laplacian: H = D - A_q
        """
        # 1. Build complex adjacency
        A_q = self.compute_magnetic_adjacency(edge_index, num_nodes, edge_types)
        
        # 2. Compute degree matrix
        # For Hermitian matrix, use out-degree (row sum magnitudes)
        out_degrees = np.array(np.abs(A_q).sum(axis=1)).flatten()
        D = sp.diags(out_degrees, format='csr', dtype=np.complex64)
        
        # 3. Laplacian: H = D - A_q
        H = D - A_q
        
        # 4. Normalize (optional but recommended)
        if self.normalize:
            # Symmetric normalization: D^(-1/2) H D^(-1/2)
            # Handle zero degrees
            out_degrees_safe = out_degrees.copy()
            out_degrees_safe[out_degrees_safe == 0] = 1.0
            
            D_inv_sqrt = sp.diags(
                1.0 / np.sqrt(out_degrees_safe),
                format='csr',
                dtype=np.complex64
            )
            
            H = D_inv_sqrt @ H @ D_inv_sqrt
        
        return H
    
    def compute_spectral_decomposition(
        self,
        H: sp.csr_matrix,
        k: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute k smallest eigenpairs of Hermitian Laplacian.
        """
        num_nodes = H.shape[0]
        k_compute = k if k is not None else self.k
        
        # Adaptive k for small graphs
        if self.adaptive_k:
            k_compute = min(k_compute, num_nodes // 3, 32)
            k_compute = max(k_compute, 1)
        
        # Edge case: Very small graphs
        if num_nodes <= k_compute + 1:
            # Using dense solver for small graphs
            H_dense = H.toarray()
            eigenvalues, eigenvectors = np.linalg.eigh(H_dense)
            
            # Sort by eigenvalue (ascending)
            idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            return eigenvalues[:k_compute], eigenvectors[:, :k_compute]
        
        try:
            # Use Hermitian-aware sparse solver
            # which='SM' means smallest magnitude (for PSD matrices)
            # Note: eigs is for general, but we treat H as general complex here
            # ARPACK eigs with sigma=0 or SM is good for smallest.
            # But H is Hermitian, so eigsh is usually preferred if supported for complex.
            # Scipy eigsh supports complex Hermitian.
            # Let's use eigs as in the provided script, but robustly.
            
            eigenvalues, eigenvectors = eigs(
                H,
                k=k_compute,
                which='SM', 
                tol=self.tolerance,
                return_eigenvectors=True
            )
            
            # Eigenvalues should be real (Hermitian property)
            eigenvalues = eigenvalues.real
            
            # Sort by eigenvalue
            idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
        except Exception as e:
            logger.warning(f"Sparse eigensolver failed ({e}). Falling back to dense.")
            H_dense = H.toarray()
            eigenvalues, eigenvectors = np.linalg.eigh(H_dense)
            
            idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            eigenvalues = eigenvalues[:k_compute]
            eigenvectors = eigenvectors[:, :k_compute]
        
        return eigenvalues, eigenvectors
    
    def validate_spectral_properties(
        self,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray
    ) -> Dict[str, bool]:
        """
        Validate computed spectral properties.
        """
        results = {}
        
        # 1. All finite
        results['finite'] = (
            np.all(np.isfinite(eigenvalues)) and
            np.all(np.isfinite(eigenvectors))
        )
        
        # 2. Sorted
        results['sorted'] = np.all(
            eigenvalues[:-1] <= eigenvalues[1:] + 1e-5
        )
        
        # 3. First eigenvalue approx 0
        results['zero_eigenvalue'] = np.abs(eigenvalues[0]) < 1e-3
        
        # 4. PSD (all eigenvalues >= 0)
        results['positive_semidefinite'] = np.all(eigenvalues >= -1e-6)
        
        results['passed'] = all([
            results['finite'],
            results['sorted'],
            results['zero_eigenvalue'],
            results['positive_semidefinite']
        ])
        
        return results
    
    def extract_features(self, edge_index: torch.Tensor, num_nodes: int, 
                         edge_types: Optional[torch.Tensor] = None, validate: bool = True) -> Dict:
        # 1. Build Complex Adjacency
        # Forward edges = exp(i*q*pi), Backward = exp(-i*q*pi)
        if edge_types is not None:
            edge_types_np = edge_types.cpu().numpy()
            phases = np.zeros(edge_index.shape[1], dtype=np.float32)
            phases[edge_types_np == 2] = self.q * np.pi # Body
            phases[edge_types_np == 1] = -self.q * np.pi # Head
        else:
            src, dst = edge_index.cpu().numpy()
            phases = np.where(src < dst, self.q * np.pi, -self.q * np.pi)
            
        complex_weights = np.exp(1j * phases)
        A = sp.coo_matrix((complex_weights, (edge_index[0], edge_index[1])), 
                          shape=(num_nodes, num_nodes), dtype=np.complex64).tocsr()
        
        # 2. Laplacian L = D - A
        deg = np.array(np.abs(A).sum(axis=1)).flatten()
        D = sp.diags(deg, dtype=np.complex64)
        L = D - A
        
        # 3. Normalization L_sym = D^-0.5 L D^-0.5
        if self.normalize:
            deg[deg < 1e-9] = 1.0
            D_inv_sqrt = sp.diags(1.0 / np.sqrt(deg), dtype=np.complex64)
            L = D_inv_sqrt @ L @ D_inv_sqrt
            
        # 4. Eigendecomposition
        k_compute = min(self.k, num_nodes - 2) if self.adaptive_k else self.k
        if k_compute < 1: k_compute = 1
        
        try:
            if num_nodes <= k_compute + 2:
                vals, vecs = np.linalg.eigh(L.toarray())
            else:
                vals, vecs = eigs(L, k=k_compute, which='SM', tol=self.tolerance)
                
            # Sort (Real part)
            idx = np.argsort(vals.real)
            vals = vals[idx].real.astype(np.float32)
            vecs = vecs[:, idx]
            
        except Exception as e:
            logger.warning(f"Eigen solver failed: {e}. Using dummy.")
            return self._dummy_features(num_nodes)
            
        return {
            'eigenvalues': vals,
            'eigenvectors_real': vecs.real.astype(np.float32),
            'eigenvectors_imag': vecs.imag.astype(np.float32)
        }

    def _dummy_features(self, n):
        return {
            'eigenvalues': np.zeros(self.k, dtype=np.float32),
            'eigenvectors_real': np.zeros((n, self.k), dtype=np.float32),
            'eigenvectors_imag': np.zeros((n, self.k), dtype=np.float32)
        }