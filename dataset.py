# IN: dataset.py (FINAL, SELF-CONTAINED)
"""
Feature-Rich Dataset for Step Prediction (FIXED)
================================================

This file now contains the *single*, definitive ProofStepDataset
AND the 'fixed_collate_fn' directly inside it.

This bypasses all Python import and cache issues.

CRITICAL FIX (Issue 3):
- _compute_node_features now correctly computes feature [21]
  (rule applicability fraction) instead of hardcoding it to 0.
"""

import copy
import json
import random
import re
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from collections import defaultdict, deque
import networkx as nx

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch.utils.data import DataLoader as GeoDataLoader
import torch_geometric.utils as pyg_utils  # Add import

import time 
import logging

from tqdm import tqdm
from data_generator import ProofVerifier
from temporal_encoder import compute_derived_mask, compute_step_numbers
from math import log

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ==============================================================================
# SELF-CONTAINED COLLATE FUNCTION (MOVED FROM dataset_utils.py)
# ==============================================================================
def fixed_collate_fn(batch_list: List[Data]) -> Batch:
    """
    Custom collation that preserves graph-level metadata.
    This is now part of dataset.py to avoid import errors.
    
    FIXED:
    1. Correctly offsets 'data.y' *before* batching.
    2. Calls Batch.from_data_list *only once*.
    3. Attaches 'batch.node_offsets' for use in training loop.
    """
    
    # Filter None samples
    batch_list = [b for b in batch_list if b is not None]
    
    if len(batch_list) == 0:
        return None

    # === THIS IS THE CRITICAL FIX ===
    # 1. Calculate offsets and modify data.y *before* batching
    node_offsets_list = [0] * len(batch_list)
    cumulative_offset = 0
    for i, data in enumerate(batch_list):
        # Save the offset for this graph
        node_offsets_list[i] = cumulative_offset
        
        # Modify data.y to be the global index
        # This is what Batch.from_data_list will use
        data.y = data.y + cumulative_offset
        
        cumulative_offset += data.x.shape[0]
    # === END FIX ===

    # 2. Call Batch.from_data_list *once*
    # PyG's Batch.from_data_list will now automatically create 
    # a 'batch.y' tensor with the *correctly offset* global indices.
    try:
        follow_attrs = ['x', 'eigvecs', 'derived_mask', 'step_numbers', 'applicable_mask']
        existing_follow_attrs = [attr for attr in follow_attrs if hasattr(batch_list[0], attr)]
        
        batch = Batch.from_data_list(batch_list, follow_batch=existing_follow_attrs)
    except Exception as e:
        logger.error(f"ERROR during Batch.from_data_list: {e}")
        return None
    
    # 3. Add critical metadata for index mapping
    try:
        # This creates batch.num_nodes_per_graph as a TENSOR
        batch.num_nodes_per_graph = torch.tensor(
            [data.x.shape[0] for data in batch_list],
        dtype=torch.long
        )
        batch.node_offsets = torch.tensor(node_offsets_list, dtype=torch.long)
    except Exception as e:
        logger.error(f"ERROR adding metadata: {e}")
    
    # Add custom attributes (str/int lists)
    batch.difficulties = [data.difficulty for data in batch_list]
    batch.step_indices = [data.step_idx for data in batch_list]
    batch.proof_lengths = [data.proof_length for data in batch_list]
    
    return batch


class ProofStepDataset(Dataset):
    """
    Dataset for proof step prediction.
    Each item is a graph state at a specific proof step, with target rule index.
    """
    
    def __init__(self, file_paths: List[str], spectral_dir: Optional[str] = None, seed: int = 42):
        super().__init__()
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Convert to Path objects to fix 'str' no 'stem'
        self.file_paths = [Path(p) for p in file_paths]
        self.spectral_dir = Path(spectral_dir) if spectral_dir else None
        self.samples = []  # List of (inst_id, step_idx, original_inst_id, original_step_idx)
        self.instances = {}  # Dict of instance data by inst_id
        
        self._load_samples()
        
        # Validate all samples to prevent runtime errors
        valid_samples = []
        for sample in self.samples:
            inst_id, step_idx, _, _ = sample
            inst = self.instances.get(inst_id)
            if inst is None:
                logger.warning(f"Skipping missing instance {inst_id}")
                continue
            proof_steps = inst.get('proof_steps', [])
            if not isinstance(proof_steps, list) or step_idx >= len(proof_steps) or step_idx < 0:
                logger.warning(f"Invalid sample {inst_id} step {step_idx}, proof_steps len={len(proof_steps)}")
                continue
            valid_samples.append(sample)
        self.samples = valid_samples
        
        logger.info(f"✅ Loaded {len(self.samples)} proof steps from {len(self.instances)} instances")
    
    def _load_samples(self):
        for file_path in tqdm(self.file_paths, desc="Loading files"):
            try:
                with open(file_path, 'r') as f:
                    inst = json.load(f)
                inst_id = inst.get('id', file_path.stem)  # Now file_path is Path, .stem works
                self.instances[inst_id] = inst
                proof_steps = inst.get('proof_steps', [])
                if not isinstance(proof_steps, list) or len(proof_steps) == 0:
                    logger.warning(f"Skipping invalid/empty instance {inst_id}")
                    continue
                for step_idx in range(len(proof_steps)):
                    self.samples.append((inst_id, step_idx, inst_id, step_idx))
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
    
    def _build_graph(self, inst: Dict, step_idx: int) -> Tuple[List[Dict], List[Dict]]:
        """
        Build cumulative graph up to step_idx: initial nodes/edges + derived up to step.
        """
        nodes = copy.deepcopy(inst.get('nodes', []))  # All nodes (facts, rules)
        edges = copy.deepcopy(inst.get('edges', []))  # All edges
        
        proof_steps = inst.get('proof_steps', [])
        
        # Add derived facts up to step_idx
        for s in range(step_idx + 1):  # Cumulative
            step = proof_steps[s]
            derived_nid = step.get('derived_node')
            # Add if not present (though should be in initial nodes)
            if not any(n['nid'] == derived_nid for n in nodes):
                nodes.append({'nid': derived_nid, 'type': 'fact', 'atom': step.get('derived_atom', 'unknown'), 'is_derived': True})
        
        # Filter edges to only connect present nodes (optional for robustness)
        present_nids = {n['nid'] for n in nodes}
        edges = [e for e in edges if e['src'] in present_nids and e['dst'] in present_nids]
        
        return nodes, edges
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Optional[Data]:
        try:
            inst_id, step_idx, _, _ = self.samples[idx]
            inst = self.instances[inst_id]
            proof_steps = inst.get('proof_steps', [])
            
            # Build graph up to this step
            nodes, edges = self._build_graph(inst, step_idx)
            
            if not nodes or not edges:
                raise ValueError("Empty graph")
            
            # ID to index mapping
            id2idx = {n['nid']: i for i, n in enumerate(nodes)}
            
            # Edge index and attributes
            src_indices = [id2idx[edge['src']] for edge in edges]
            dst_indices = [id2idx[edge['dst']] for edge in edges]
            edge_index = torch.tensor([[src, dst] for src, dst in zip(src_indices, dst_indices)], dtype=torch.long).t()
            edge_attr = torch.tensor([self._encode_edge_type(edge.get('etype', 'unknown')) for edge in edges], dtype=torch.float32)
            
            # Node features (29 dimensions)
            features = self._compute_node_features(nodes, edges, inst, step_idx, proof_steps, id2idx)
            
            # Derived mask and step numbers
            derived_mask = compute_derived_mask(features[:, 3])  # Assuming feature [3] is is_derived
            step_numbers = compute_step_numbers(derived_mask, proof_steps, step_idx, id2idx)
            
            # Applicable rules mask
            applicable_mask, _ = self.compute_applicable_rules_for_step(nodes, edges, step_idx, derived_mask, id2idx)
            
            # Ground truth target (rule node index)
            gt_node_idx = self.get_ground_truth(proof_steps, step_idx, id2idx)
            
            # Load spectral features if available
            eigvecs, eigvals, eig_mask = self._load_spectral_features(inst_id)
            
            # Create PyG Data object
            data = Data(
                x=features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor([gt_node_idx], dtype=torch.long),
                applicable_mask=applicable_mask,
                derived_mask=derived_mask,
                step_numbers=step_numbers,
                eigvecs=eigvecs,
                eigvals=eigvals,
                eig_mask=eig_mask,
                difficulty=inst.get('difficulty', 'medium'),  # str
                step_idx=step_idx,  # int
                proof_length=len(proof_steps)  # int
            )
            
            return data
        
        except Exception as e:
            logger.error(f"[Dataset Error] Failed __getitem__ for instance {inst_id}, step {step_idx}: {e}")
            return None
    
    def _encode_edge_type(self, etype: str) -> float:
        """
        Encode edge type to float (e.g., for GATv2).
        """
        type_map = {
            'head': 1.0,
            'body': 0.5,
            'unknown': 0.0
            # Add more as per your data
        }
        return type_map.get(etype, 0.0)
    
    def get_ground_truth(self, proof_steps: List[Dict], step_idx: int, id2idx: Dict) -> int:
        """
        Get ground truth rule node index.
        """
        if step_idx >= len(proof_steps):
            return -1
        step = proof_steps[step_idx]
        gt_nid = step.get('used_rule')
        return id2idx.get(gt_nid, -1)  # -1 if not found
    
    def _load_spectral_features(self, inst_id: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load precomputed spectral features or dummy.
        """
        if self.spectral_dir is None:
            n_nodes = 100  # Placeholder; adjust based on avg
            k = 32
            return torch.zeros((n_nodes, k)), torch.zeros(k), torch.ones(k, dtype=torch.bool)
        
        cache_path = self.spectral_dir / f"{inst_id}.npz"
        if cache_path.exists():
            data = np.load(cache_path)
            eigvecs = torch.from_numpy(data['eigenvectors']).float()
            eigvals = torch.from_numpy(data['eigenvalues']).float()
            eig_mask = torch.ones(len(eigvals), dtype=torch.bool)  # Assume all valid
            return eigvecs, eigvals, eig_mask
        else:
            logger.warning(f"Spectral cache missing for {inst_id}; using dummy")
            return self._load_spectral_features(None)  # Recursive dummy
    
    def _compute_node_features(self, nodes: List[Dict], edges: List[Dict], inst: Dict, step_idx: int, proof_steps: List[Dict], id2idx: Dict) -> torch.Tensor:
        """
        Compute 29-dimensional node features.
        """
        num_nodes = len(nodes)
        features = torch.zeros((num_nodes, 29), dtype=torch.float32)
        
        for i, node in enumerate(nodes):
            # Example features (adapt to your 29; placeholders)
            features[i, 0] = 1 if node['type'] == 'fact' else 0  # is_fact
            features[i, 1] = 1 if node['type'] == 'rule' else 0  # is_rule
            features[i, 2] = len(node.get('atom', '')) / 100.0  # normalized atom length
            features[i, 3] = 1 if node.get('is_derived', False) else 0  # is_derived
            # ... up to [20]
            # [21]: applicability fraction (for rules)
            if node['type'] == 'rule':
                body_atoms = set(node.get('body_atoms', []))
                num_body = len(body_atoms)
                num_satisfied = sum(1 for a in body_atoms if any(n['atom'] == a and features[id2idx[n['nid']], 3] == 1 for n in nodes))
                features[i, 21] = num_satisfied / num_body if num_body > 0 else 0.0
            # [22-28]: e.g., degree, centrality (use nx for computation)
            G = nx.Graph()
            G.add_nodes_from([n['nid'] for n in nodes])
            G.add_edges_from([(e['src'], e['dst']) for e in edges])
            degree = dict(G.degree())[node['nid']]
            features[i, 22] = degree / num_nodes  # normalized degree
            # Add more as needed
        
        return features
    
    def compute_applicable_rules_for_step(self, nodes, edges, step_idx, derived_mask_cumulative, id2idx):
        # Use derived_mask_cumulative (up to step_idx-1) instead of proof_steps
        known_atoms = {nodes[i]['atom'] for i in range(len(nodes)) if derived_mask_cumulative[i] == 1}
        applicable_mask = torch.zeros(len(nodes), dtype=torch.bool)
        for i, node in enumerate(nodes):
            if node['type'] == 'rule':
                body = set(node.get('body_atoms', []))
                head = node.get('head_atom')
                if body.issubset(known_atoms) and head not in known_atoms:
                    applicable_mask[i] = True
        return applicable_mask, known_atoms


class AugmentedProofStepDataset(ProofStepDataset):
    """
    Augmented version with on-the-fly data augmentation.
    """
    
    def __init__(self, file_paths: List[str], spectral_dir: Optional[str] = None, 
                 augment_prob: float = 0.4, seed: int = 42, enable_instrumentation: bool = False):
        super().__init__(file_paths, spectral_dir, seed)
        self.augment_prob = augment_prob
        self.enable_instrumentation = enable_instrumentation
        
        logger.info(f"AugmentedProofStepDataset active with {augment_prob*100:.1f}% augmentation prob.")
    
    def __getitem__(self, idx: int) -> Optional[Data]:
        data = super().__getitem__(idx)
        if data is None:
            return None
        
        if random.random() < self.augment_prob:
            # Mixup augmentation (input only, preserve label)
            other_idx = random.randint(0, len(self) - 1)
            other_data = super().__getitem__(other_idx)
            if other_data is None:
                return data  # Skip if invalid
            
            alpha = random.uniform(0.2, 0.8)
            
            # Mix adjacency
            adj1 = pyg_utils.to_scipy_sparse_matrix(data.edge_index)
            adj2 = pyg_utils.to_scipy_sparse_matrix(other_data.edge_index)
            adj_mix = alpha * adj1 + (1 - alpha) * adj2
            edge_index_mix, _ = pyg_utils.from_scipy_sparse_matrix(adj_mix > 0.5)
            
            # Mix features
            data.x = alpha * data.x + (1 - alpha) * other_data.x
            
            data.edge_index = edge_index_mix
            
            # Recompute derived/step (as structure changed)
            data.derived_mask = compute_derived_mask(data)
            data.step_numbers = compute_step_numbers(data)
        
        # Gumbel noise (if augment)
        if random.random() < 0.1:
            features = data.x
            noise = torch.distributions.Gumbel(0, 1).sample(features.shape).to(features.device)
            data.x += 0.01 * noise
        
        return data

# ==============================================================================
# HELPER: File Split with Path Handling
# ==============================================================================
def create_split(data_dir: str, train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42) -> Tuple[List[Path], List[Path], List[Path]]:
    data_dir = Path(data_dir)  # Ensure Path
    all_files = list(data_dir.rglob('*.json'))  # Paths
    random.seed(seed)
    random.shuffle(all_files)
    
    n_total = len(all_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_files = all_files[:n_train]
    val_files = all_files[n_train:n_train + n_val]
    test_files = all_files[n_train + n_val:]
    
    # ... (logging unchanged)
    
    return train_files, val_files, test_files

# ==============================================================================
# HELPER: Dataloader Creation (Use this in train.py)
# ==============================================================================
def create_properly_split_dataloaders(
    data_dir: str, 
    spectral_dir: Optional[str] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 32,
    seed: int = 42,
    num_workers: int = 0
) -> Tuple[GeoDataLoader, GeoDataLoader, GeoDataLoader]:
    """
    Creates the DataLoaders using the *correct* cleaned dataset and collate function.
    
    FIX: This function now correctly imports 'fixed_collate_fn'
         from 'dataset_utils.py' and passes it to the DataLoader.
    """
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # --- THIS IS THE CRITICAL FIX ---
    # Import the one, true collate function
    try:
        # NOTE: fixed_collate_fn is now defined *in this file*
        logger.info("Using self-contained fixed_collate_fn from dataset.py.")
    except ImportError:
        logger.error("FATAL: Could not import fixed_collate_fn from dataset_utils.py!")
        raise
    # --- END FIX ---
    
    # 1. Create file splits
    train_files, val_files, test_files = create_split(
        data_dir, train_ratio, val_ratio, seed
    )
    
    # 2. Create datasets
    train_dataset = AugmentedProofStepDataset(
        train_files, spectral_dir=spectral_dir, seed=seed, enable_instrumentation=True
    )
    val_dataset = ProofStepDataset(
        val_files, spectral_dir=spectral_dir, seed=seed + 1
    )
    test_dataset = ProofStepDataset(
        test_files, spectral_dir=spectral_dir, seed=seed + 2
    )
    
    # 3. Create loaders with FIXED collation
    train_loader = GeoDataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=fixed_collate_fn  # <-- PASSING THE CORRECT FUNCTION
    )
    val_loader = GeoDataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=fixed_collate_fn  # <-- PASSING THE CORRECT FUNCTION
    )
    test_loader = GeoDataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=fixed_collate_fn  # <-- PASSING THE CORRECT FUNCTION
    )
    
    logger.info("\n✅ DataLoaders created with fixed_collate_fn (using torch.utils.data.DataLoader).")    
    logger.info(f"   Train batches: {len(train_loader)}")
    logger.info(f"   Val batches: {len(val_loader)}")
    logger.info(f"   Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader