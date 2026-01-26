"""
Feature-Rich Dataset for Step Prediction (FIXED for Variable K)
===============================================================
"""

import copy
import json
import random
import numpy as np
from typing import Any, List, Dict, Optional, Tuple
from pathlib import Path
from collections import defaultdict
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch.utils.data import DataLoader as GeoDataLoader
import torch.nn.functional as F
import logging
from tqdm import tqdm

from data_generator import LogicalAugmenter
from spectral_features import MagneticLaplacianExtractor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ... (Keep FeatureComputer class as is) ...
class FeatureComputer:
    def __init__(self):
        self.feature_dim = 32
    
    def compute_features(self, nodes, edges, proof_steps, step_idx, id2idx, goal_atom: str):
        N = len(nodes)
        features = torch.zeros((N, self.feature_dim), dtype=torch.float32)
        
        in_degree = defaultdict(int)
        out_degree = defaultdict(int)
        
        for e in edges:
            if e['src'] in id2idx and e['dst'] in id2idx:
                src_idx = id2idx[e['src']]
                dst_idx = id2idx[e['dst']]
                out_degree[src_idx] += 1
                in_degree[dst_idx] += 1

        max_degree = max(max(in_degree.values(), default=1), max(out_degree.values(), default=1))
        
        derived_indices = set()
        known_atoms = set()
        for i, node in enumerate(nodes):
            if node.get('type') == 'fact' and node.get('is_initial', False):
                known_atoms.add(node['atom'])
        
        for step in proof_steps[:step_idx]:
            derived_nid = step.get('derived_node')
            if derived_nid in id2idx:
                idx = id2idx[derived_nid]
                derived_indices.add(idx)
                if idx < N and nodes[idx].get('type') == 'fact':
                    known_atoms.add(nodes[idx]['atom'])

        for i, node in enumerate(nodes):
            node_type = node.get('type', 'unknown')
            atom_str = node.get('atom', node.get('head_atom', ''))
            
            features[i, 0] = 1.0 if node_type == 'fact' else 0.0
            features[i, 1] = 1.0 if node_type == 'rule' else 0.0
            features[i, 2] = min(len(atom_str) / 50.0, 1.0)
            features[i, 3] = 1.0 if i in derived_indices else 0.0
            features[i, 4] = 1.0 if node.get('is_initial', False) else 0.0
            features[i, 5] = (in_degree[i] + out_degree[i]) / max_degree
            features[i, 6] = in_degree[i] / max_degree
            features[i, 7] = out_degree[i] / max_degree
            
            if node_type == 'rule':
                body = node.get('body_atoms', [])
                features[i, 12] = len(body) / 5.0
                features[i, 18] = len(body) / 10.0
                features[i, 19] = 1.0 if node.get('head_atom') in known_atoms else 0.0
                
                if body:
                    satisfied = sum(1 for a in body if a in known_atoms)
                    features[i, 21] = satisfied / len(body)
                else:
                    features[i, 21] = 1.0
            
            is_goal_node = (node_type == 'fact' and atom_str == goal_atom)
            features[i, 29] = 1.0 if is_goal_node else 0.0
            produces_goal = (node_type == 'rule' and node.get('head_atom') == goal_atom)
            features[i, 30] = 1.0 if produces_goal else 0.0
            consumes_goal = (node_type == 'rule' and goal_atom in node.get('body_atoms', []))
            features[i, 31] = 1.0 if consumes_goal else 0.0

        return features

# Helper functions
def compute_derived_mask(is_derived_column: torch.Tensor) -> torch.Tensor:
    return is_derived_column.bool() 

def compute_step_numbers(derived_mask, proof_steps, current_step_idx, id2idx):
    num_nodes = len(derived_mask)
    step_num_tensor = torch.zeros(num_nodes, dtype=torch.long)
    for step_info in proof_steps[:current_step_idx + 1]:
        derived_nid = step_info.get('derived_node')
        if derived_nid in id2idx:
            node_idx = id2idx[derived_nid]
            if node_idx < num_nodes:
                step_num_tensor[node_idx] = step_info.get('step_id', 0) + 1
    return step_num_tensor

def fixed_collate_fn(batch_list: List[Data]) -> Batch:
    batch_list = [b for b in batch_list if b is not None]
    if len(batch_list) == 0: return None

    try:
        follow_attrs = [
            'x', 'eigvecs_real', 'eigvecs_imag', 'eigvecs', 
            'derived_mask', 'step_numbers', 'applicable_mask'
        ]
        existing_follow_attrs = [attr for attr in follow_attrs if hasattr(batch_list[0], attr)]
        batch = Batch.from_data_list(batch_list, follow_batch=existing_follow_attrs)
    except Exception as e:
        logger.error(f"ERROR during Batch.from_data_list: {e}")
        return None
    
    batch.difficulties = [data.difficulty for data in batch_list]
    batch.step_indices = [data.step_idx for data in batch_list]
    batch.proof_lengths = [data.proof_length for data in batch_list]
    batch.meta_list = [
        {'difficulty': d.difficulty, 'step_idx': d.step_idx, 'proof_length': d.proof_length}
        for d in batch_list
    ]
    return batch

class ProofStepDataset(Dataset):
    def __init__(self, file_paths: List[str], spectral_dir: Optional[str] = None, seed: int = 42, k: int = 16, augment=False):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        self.file_paths = [Path(p) for p in file_paths]
        self.spectral_dir = Path(spectral_dir) if spectral_dir else None
        self.k = k  # Store k
        self.samples = []
        self.instances = {}
        self.feature_computer = FeatureComputer()
        self._load_samples()
        
        self.augment = augment
        self.augmenter = LogicalAugmenter(p_stretch=0.3, p_thicken=0.2)
        self.spectral_extractor = MagneticLaplacianExtractor(
            k=k, q=0.25, normalize=True, adaptive_k=True
        )
    def _load_samples(self):
        for file_path in tqdm(self.file_paths, desc="Loading files"):
            try:
                with open(file_path, 'r') as f:
                    inst = json.load(f)
                inst_id = inst.get('id', file_path.stem)
                self.instances[inst_id] = inst
                proof_steps = inst.get('proof_steps', [])
                for step_idx in range(len(proof_steps)):
                    self.samples.append((inst_id, step_idx, inst_id, step_idx))
            except Exception:
                pass
    
    def _build_graph(self, inst: Dict, step_idx: int) -> Tuple[List[Dict], List[Dict]]:
        nodes = copy.deepcopy(inst.get('nodes', []))
        edges = copy.deepcopy(inst.get('edges', []))
        proof_steps = inst.get('proof_steps', [])
        for s in range(step_idx + 1):
            step = proof_steps[s]
            derived_nid = step.get('derived_node')
            if not any(n['nid'] == derived_nid for n in nodes):
                nodes.append({'nid': derived_nid, 'type': 'fact', 'atom': step.get('derived_atom', 'unknown'), 'is_derived': True})
        present_nids = {n['nid'] for n in nodes}
        edges = [e for e in edges if e['src'] in present_nids and e['dst'] in present_nids]
        return nodes, edges
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Optional[Data]:
        try:
            inst_id, step_idx, _, _ = self.samples[idx]
            inst = self.instances[inst_id]

            # [CRITICAL FIX] Re-enable Augmentation
            if self.augment:
                inst = self.augmenter.augment(inst)

            proof_steps = inst.get('proof_steps', [])
            goal_atom = inst.get('goal')
            nodes, edges = self._build_graph(inst, step_idx)
            
            id2idx = {n['nid']: i for i, n in enumerate(nodes)}
            src_indices = [id2idx[e['src']] for e in edges if e['src'] in id2idx and e['dst'] in id2idx]
            dst_indices = [id2idx[e['dst']] for e in edges if e['src'] in id2idx and e['dst'] in id2idx]
            edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
            
            type_map = {'unknown': 0, 'head': 1, 'body': 2}
            edge_attr = torch.tensor([type_map.get(e.get('etype', 'unknown'), 0) for e in edges if e['src'] in id2idx and e['dst'] in id2idx], dtype=torch.long)

            features = self.feature_computer.compute_features(nodes, edges, proof_steps, step_idx, id2idx, goal_atom)
            derived_mask = compute_derived_mask(features[:, 3])
            step_numbers = compute_step_numbers(derived_mask, proof_steps, step_idx, id2idx)
            applicable_mask, _ = self.compute_applicable_rules_for_step(nodes, edges, step_idx, step_numbers, id2idx)
            gt_node_idx = self.get_ground_truth(proof_steps, step_idx, id2idx)
            
            # [CRITICAL FIX] DYNAMIC SPECTRAL COMPUTATION
            if len(nodes) > 0:
                try:
                    spectral_data = self.spectral_extractor.extract_features(
                        edge_index=edge_index,
                        num_nodes=len(nodes),
                        edge_types=edge_attr,
                        validate=False
                    )
                    
                    # --- FIX: CONVERT NUMPY TO TENSOR BEFORE PADDING ---
                    if isinstance(spectral_data['eigenvalues'], np.ndarray):
                        eigvecs_real = torch.from_numpy(spectral_data['eigenvectors_real']).float()
                        eigvecs_imag = torch.from_numpy(spectral_data['eigenvectors_imag']).float()
                        eigvals = torch.from_numpy(spectral_data['eigenvalues']).float()
                    else:
                        eigvecs_real = spectral_data['eigenvectors_real'].float()
                        eigvecs_imag = spectral_data['eigenvectors_imag'].float()
                        eigvals = spectral_data['eigenvalues'].float()
                    # ---------------------------------------------------

                    k_real = len(eigvals)
                    eig_mask = torch.zeros(self.k, dtype=torch.bool)
                    eig_mask[:min(k_real, self.k)] = True
                    
                    if k_real < self.k:
                        pad_len = self.k - k_real
                        eigvecs_real = F.pad(eigvecs_real, (0, pad_len))
                        eigvecs_imag = F.pad(eigvecs_imag, (0, pad_len))
                        eigvals = F.pad(eigvals, (0, pad_len))
                    elif k_real > self.k:
                         eigvecs_real = eigvecs_real[:, :self.k]
                         eigvecs_imag = eigvecs_imag[:, :self.k]
                         eigvals = eigvals[:self.k]
                         
                except Exception as e:
                    # Uncomment this if you still see zeros to debug further
                    # logger.warning(f"Spectral calc failed for {inst_id}: {e}")
                    eigvecs_real = torch.zeros((len(nodes), self.k))
                    eigvecs_imag = torch.zeros((len(nodes), self.k))
                    eigvals = torch.zeros(self.k)
                    eig_mask = torch.zeros(self.k, dtype=torch.bool)
            else:
                 eigvecs_real = torch.zeros((0, self.k))
                 eigvecs_imag = torch.zeros((0, self.k))
                 eigvals = torch.zeros(self.k)
                 eig_mask = torch.zeros(self.k, dtype=torch.bool)

            metadata = inst.get('metadata', {})
            proof_len = max(metadata.get('proof_length', len(proof_steps)), 1.0)
            value_target = torch.tensor([1.0 - (step_idx / proof_len)], dtype=torch.float)

            data = Data(
                x=features, edge_index=edge_index, edge_attr=edge_attr,
                y=torch.tensor([gt_node_idx], dtype=torch.long),
                applicable_mask=applicable_mask, derived_mask=derived_mask,
                step_numbers=step_numbers,
                eigvecs_real=eigvecs_real, eigvecs_imag=eigvecs_imag,
                eigvals=eigvals, eig_mask=eig_mask,
                difficulty=metadata.get('difficulty', 'medium'),
                step_idx=step_idx, proof_length=proof_len, value_target=value_target
            )
            return data
        except Exception as e:
            logger.error(f"Error in getitem: {e}")
            return None
            
    def _load_spectral_features(self, inst_id: str, num_nodes: int):
        k = self.k # USE SELF.K
        dummy_real = torch.zeros((num_nodes, k), dtype=torch.float32)
        dummy_imag = torch.zeros((num_nodes, k), dtype=torch.float32)
        dummy_vals = torch.zeros(k, dtype=torch.float32)
        dummy_mask = torch.zeros(k, dtype=torch.bool)
        
        if self.spectral_dir is None: return dummy_real, dummy_imag, dummy_vals, dummy_mask
        
        cache_path = self.spectral_dir / f"{inst_id}_magnetic.npz"
        
        if cache_path.exists():
            try:
                data = np.load(cache_path)
                eigvecs_real = torch.from_numpy(data['eigenvectors_real']).float()
                eigvecs_imag = torch.from_numpy(data['eigenvectors_imag']).float()
                eigvals = torch.from_numpy(data['eigenvalues']).float()
                
                if eigvecs_real.shape[0] != num_nodes: return dummy_real, dummy_imag, dummy_vals, dummy_mask
                
                current_k = eigvecs_real.shape[1]
                if current_k < k:
                    pad = (0, k - current_k)
                    eigvecs_real = F.pad(eigvecs_real, pad)
                    eigvecs_imag = F.pad(eigvecs_imag, pad)
                    eigvals = F.pad(eigvals, pad)
                elif current_k > k:
                    eigvecs_real = eigvecs_real[:, :k]
                    eigvecs_imag = eigvecs_imag[:, :k]
                    eigvals = eigvals[:k]
                
                eig_mask = torch.zeros(k, dtype=torch.bool)
                eig_mask[:min(len(data['eigenvalues']), k)] = True
                return eigvecs_real, eigvecs_imag, eigvals, eig_mask
            except: return dummy_real, dummy_imag, dummy_vals, dummy_mask
        return dummy_real, dummy_imag, dummy_vals, dummy_mask

    def get_ground_truth(self, proof_steps, step_idx, id2idx):
        if step_idx >= len(proof_steps): return -1
        return id2idx.get(proof_steps[step_idx].get('used_rule'), -1)

    def compute_applicable_rules_for_step(self, nodes, edges, step_idx, step_numbers, id2idx):
        known_atoms = set()
        
        # In a proof, the "current" state includes everything derived in steps BEFORE the current one.
        # step_idx is 0-based. If we are predicting step 5, we know facts from 0..4.
        # step_numbers maps node -> step_id + 1. 
        # So step_idx=0 (first step) -> knows nothing? No, knows axioms (step=0 in some logic, or check is_initial).
        
        # Robust Known Atoms Collection
        for i, node in enumerate(nodes):
            if node['type'] == 'fact':
                # Always know initial facts
                if node.get('is_initial', False):
                    known_atoms.add(node['atom'])
                # Know facts derived in strictly previous steps
                # step_numbers[i] is (step_id + 1). 
                # We are at step_idx. Previous steps are 0 to step_idx-1.
                # So we want derived_step_id < step_idx
                # derived_step_id = step_numbers[i] - 1
                # (step_numbers[i] - 1) < step_idx  => step_numbers[i] <= step_idx
                elif step_numbers[i] > 0 and step_numbers[i] <= (step_idx): 
                    known_atoms.add(node['atom'])
        
        applicable_mask = torch.zeros(len(nodes), dtype=torch.bool)
        for i, node in enumerate(nodes):
            if node['type'] == 'rule':
                body = set(node.get('body_atoms', []))
                head = node.get('head_atom')
                
                # Rule is applicable if ALL body atoms are known
                if body.issubset(known_atoms):
                    # And head is NOT known (to avoid cycles/redundancy)
                    # Note: For training, we sometimes relax the 'head not known' constraint 
                    # if the data generator allows redundancy, but standard NTP requires it.
                    if head not in known_atoms:
                        applicable_mask[i] = True
                        
        return applicable_mask, known_atoms

def create_split(data_dir: str, train_ratio=0.7, val_ratio=0.15, seed=42):
    data_dir = Path(data_dir)
    all_files = list(data_dir.rglob('*.json'))
    random.seed(seed)
    random.shuffle(all_files)
    n_total = len(all_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    return all_files[:n_train], all_files[n_train:n_train + n_val], all_files[n_train + n_val:]

def create_properly_split_dataloaders(
    data_dir: str, spectral_dir: Optional[str] = None, train_ratio=0.7, val_ratio=0.15,
    batch_size=32, seed=42, num_workers=0, pin_memory=False, k: int = 16  # ADDED k argument
):
    train_files, val_files, test_files = create_split(data_dir, train_ratio, val_ratio, seed)
    
    train_dataset = ProofStepDataset(train_files, spectral_dir=spectral_dir, seed=seed, k=k, augment=True)
    val_dataset = ProofStepDataset(val_files, spectral_dir=spectral_dir, seed=seed+1, k=k)
    test_dataset = ProofStepDataset(test_files, spectral_dir=spectral_dir, seed=seed+2, k=k)
    
    train_loader = GeoDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=fixed_collate_fn, pin_memory=pin_memory)
    val_loader = GeoDataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=fixed_collate_fn, pin_memory=pin_memory)
    test_loader = GeoDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=fixed_collate_fn, pin_memory=pin_memory)
    
    return train_loader, val_loader, test_loader