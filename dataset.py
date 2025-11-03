# IN: dataset.py (FINAL, SELF-CONTAINED)
"""
Feature-Rich Dataset for Step Prediction (FIXED)
================================================

This file now contains the *single*, definitive ProofStepDataset
AND the 'fixed_collate_fn' directly inside it.

This bypasses all Python import and cache issues.
"""

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
from torch_geometric.loader import DataLoader as GeoDataLoader

import time 
import logging
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
    """
    
    # Filter None samples
    batch_list = [b for b in batch_list if b is not None]
    
    if len(batch_list) == 0:
        return None
    
    # CRITICAL: Must use follow_batch to track node assignments
    try:
        # We must follow_batch on every attribute that is per-node
        follow_attrs = ['x', 'eigvecs', 'derived_mask', 'step_numbers', 'applicable_mask']
        # Filter to attributes that actually exist in the first data object
        existing_follow_attrs = [attr for attr in follow_attrs if hasattr(batch_list[0], attr)]
        
        batch = Batch.from_data_list(batch_list, follow_batch=existing_follow_attrs)
    except Exception as e:
        logger.error(f"ERROR during Batch.from_data_list: {e}")
        return None

    # === THIS IS THE CRITICAL CODE ===
    # Add critical metadata for index mapping
    try:
        # This creates batch.num_nodes_per_graph as a TENSOR
        batch.num_nodes_per_graph = torch.tensor(
            [data.x.shape[0] for data in batch_list],
            dtype=torch.long
        )
        
        # This creates batch.node_offsets as a TENSOR
        batch.node_offsets = torch.cat([
            torch.tensor([0]),
            torch.cumsum(batch.num_nodes_per_graph[:-1], dim=0)
        ])
    except Exception as e:
        logger.error(f"ERROR while adding metadata in collate fn: {e}")
        return None
    # === END CRITICAL CODE ===
    
    # Store spectral features as lists (variable k per graph)
    if hasattr(batch_list[0], 'eigvecs'):
        batch.eigvecs_list = [data.eigvecs for data in batch_list]
        batch.eigvals_list = [data.eigvals for data in batch_list]
        batch.eig_mask_list = [data.eig_mask for data in batch_list]
    
    # Per-graph applicable masks (CRITICAL for correct loss)
    # This might be redundant if we follow_batch 'applicable_mask'
    if hasattr(batch_list[0], 'applicable_mask') and not hasattr(batch, 'applicable_mask_list'):
         batch.applicable_mask_list = [data.applicable_mask for data in batch_list]
    
    # Store metadata
    if hasattr(batch_list[0], 'meta'):
        batch.meta_list = [data.meta for data in batch_list]
    
    return batch
# ==============================================================================
# END OF SELF-CONTAINED COLLATE FUNCTION
# ==============================================================================


# ==============================================================================
# HELPER: Compute applicable rules
# ==============================================================================

def compute_applicable_rules_for_step(
    nodes: list,
    edges: list,
    step_idx: int,
    proof_steps: list
) -> Tuple[torch.Tensor, set]:
    """
    Compute which rules are applicable at a given proof step.
    
    CRITICAL: A rule is applicable IFF:
    1. All its premises (body atoms) are in known_facts
    2. Its conclusion (head atom) is NOT already known
    """
    num_nodes = len(nodes)
    applicable_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    # Create node ID to index mapping
    nid_to_idx = {n["nid"]: i for i, n in enumerate(nodes)}
    
    # ===== BUILD KNOWN FACTS AT THIS STEP =====
    known_atoms = set()
    
    # Step 1: Add initial facts (axioms)
    for node in nodes:
        if node.get("is_initial", False) and node["type"] == "fact":
            atom = node.get("atom")
            if atom:
                known_atoms.add(atom)
    
    # Step 2: Add facts derived UP TO (but not including) this step
    for step_num in range(step_idx):
        step = proof_steps[step_num]
        derived_nid = step.get("derived_node")
        
        if derived_nid in nid_to_idx:
            derived_idx = nid_to_idx[derived_nid]
            derived_node = nodes[derived_idx]
            atom = derived_node.get("atom")
            if atom:
                known_atoms.add(atom)
    
    # ===== CHECK WHICH RULES ARE APPLICABLE =====
    for node_idx, node in enumerate(nodes):
        if node["type"] != "rule":
            continue
        
        body_atoms = set(node.get("body_atoms", []))
        head_atom = node.get("head_atom")
        
        # APPLICABILITY CONDITION:
        # 1. All body atoms must be known (premises satisfied)
        # 2. Head atom must NOT be known (no redundant derivation)
        
        is_applicable = (
            body_atoms.issubset(known_atoms) and 
            head_atom not in known_atoms
        )
        
        applicable_mask[node_idx] = is_applicable
    
    return applicable_mask, known_atoms

# ==============================================================================
# HELPER: Tactic Mapper
# ==============================================================================

class RuleToTacticMapper:
    """Maps low-level rules to high-level tactics."""
    def __init__(self):
        self.tactic_patterns = {
            'forward_chain': [r'(?:→|->)', r'implies'],
            'case_split': [r'∨', r'disjunction'],
            'modus_ponens': [r'modus', r'ponens'],
            'substitution': [r'subst', r'='],
            'simplification': [r'simpl', r'reduce'],
        }
        self.tactic_names = ['unknown'] + list(self.tactic_patterns.keys())
        self.tactic_map = {name: i for i, name in enumerate(self.tactic_names)}

    def map_rule_to_tactic(self, rule_name: str) -> int:
        """Map a rule name to its corresponding tactic index."""
        rule_name_lower = rule_name.lower()
        for tactic, patterns in self.tactic_patterns.items():
            if any(re.search(pattern, rule_name_lower) for pattern in patterns):
                return self.tactic_map[tactic]
        return self.tactic_map['unknown'] # 0

# ==============================================================================
# DATASET CLASS: ProofStepDataset (The *only* one)
# ==============================================================================

class ProofStepDataset(torch.utils.data.Dataset):
    """
    Corrected dataset that properly handles proof step sampling with instrumentation.
    
    Node features (22-dim base):
    [0-1]: Type (fact/rule) one-hot
    [2]: Is initially given (not derived)
    [3]: Is derived up to this step
    [4]: Normalized in-degree
    [5]: Normalized out-degree
    [6]: Normalized depth from initial facts
    [7]: Normalized rule body size (0 for facts)
    [8]: Normalized centrality (PageRank/Degree - Original)
    [9-16]: Predicate hash (8-bit binary)
    [17]: Graph-level features (graph size)
    [18]: Normalized betweenness centrality
    [19]: Is head of rule derived (for rules)
    [20]: Atom occurrence frequency
    [21]: Rule applicability (fraction of body satisfied)
    """
    
    def __init__(self, 
                 json_files: List[str], 
                 spectral_dir: Optional[str] = None,
                 k_dim: int = 16, 
                 seed: int = 42, 
                 enable_instrumentation: bool = False):
        super().__init__()
        
        self.spectral_dir = spectral_dir
        self.k_dim = k_dim
        self.enable_instrumentation = enable_instrumentation
        self.tactic_mapper = RuleToTacticMapper()
        self.num_tactics = len(self.tactic_mapper.tactic_names)
        
        # Instrumentation tracking
        self.stats = defaultdict(int)
        
        # Step 1: Load all instances and create step-level database
        self.proof_steps_db = []  # List of sample dicts
        self.instance_metadata = {}  # For curriculum
        
        logger.info(f"Loading {len(json_files)} files...")
        
        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    inst = json.load(f)
                
                inst_id = inst.get('id', Path(file_path).stem)
                proof_steps = inst.get('proof_steps', [])
                
                if len(proof_steps) == 0:
                    logger.debug(f"⚠️  Skipping {file_path}: no proof steps")
                    self.stats['skipped_empty_proof'] += 1
                    continue
                
                # Store instance metadata
                metadata = inst.get('metadata', {})
                self.instance_metadata[inst_id] = {
                    'file': file_path,
                    'instance': inst, # Store instance for search agent
                    'proof_length': metadata.get('proof_length', len(proof_steps)),
                    'difficulty': metadata.get('difficulty', 'medium'),
                    'num_rules': metadata.get('n_rules', 0),
                    'num_nodes': metadata.get('n_nodes', len(inst['nodes']))
                }
                
                # Create one sample per proof step
                for step_idx in range(len(proof_steps)):
                    sample = {
                        'instance_id': inst_id,
                        'step_idx': step_idx,
                        'file': file_path,
                        'instance': inst, # Pass full instance
                        'proof_steps': proof_steps,
                        'metadata': metadata # Pass metadata
                    }
                    self.proof_steps_db.append(sample)
            
            except Exception as e:
                logger.error(f"❌ Error loading {file_path}: {e}")
                self.stats['skipped_load_error'] += 1
                continue
        
        logger.info(f"✅ Loaded {len(self.proof_steps_db)} proof steps from {len(self.instance_metadata)} instances")
        
        # Compute node features dimension
        self.feature_dim = 22
    
    def __len__(self) -> int:
        return len(self.proof_steps_db)
    
    def __getitem__(self, idx: int) -> 'Data':
        """
        Get a single sample for training.
        This now includes robust error handling.
        """
        
        self.stats['total_accesses'] += 1
        
        try:
            sample = self.proof_steps_db[idx]
            
            inst = sample['instance']
            step_idx = sample['step_idx']
            proof_steps = sample['proof_steps']
            metadata = sample['metadata']
            instance_id = sample['instance_id']
            
            nodes = inst['nodes']
            edges = inst['edges']
            num_nodes = len(nodes)
            
            # ===== BUILD NODE ID TO INDEX MAPPING =====
            id2idx = {n['nid']: i for i, n in enumerate(nodes)}
            
            # ===== STEP 1: Compute applicable_mask FIRST =====
            applicable_mask, known_facts = compute_applicable_rules_for_step(
                nodes,
                edges,
                step_idx,
                proof_steps
            )
            
            # ===== STEP 2: Get the target rule and VALIDATE =====
            step = proof_steps[step_idx]
            rule_nid = step['used_rule']
            target_idx = id2idx.get(rule_nid, -1)
            
            # VALIDATION 1: Target index must be valid
            if target_idx < 0 or target_idx >= num_nodes:
                raise ValueError(f"Invalid target index: {target_idx} for rule NID {rule_nid}")
            
            # VALIDATION 2: Target MUST be applicable
            if not applicable_mask[target_idx]:
                self.stats['inapplicable_targets'] += 1
                raise ValueError(
                    f"TARGET RULE NOT APPLICABLE! Instance: {instance_id}, Step: {step_idx}, "
                    f"Target rule (idx={target_idx}): {nodes[target_idx].get('label', 'unknown')}"
                )
            
            # ===== STEP 3: Compute node features =====
            x_base = self._compute_node_features(inst, step_idx, known_facts, id2idx)
            
            # ===== STEP 4: Build edge index =====
            edge_list = []
            edge_types = []
            for e in edges:
                src = id2idx.get(e["src"], -1)
                dst = id2idx.get(e["dst"], -1)
                if src != -1 and dst != -1:
                    edge_list.append([src, dst])
                    etype = 0 if e["etype"] == "body" else (1 if e["etype"] == "head" else 2)
                    edge_types.append(etype)
            
            if edge_list:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t()
                edge_attr = torch.tensor(edge_types, dtype=torch.long)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0,), dtype=torch.long)
            
            # ===== STEP 5: Create PyG Data =====
            pyg_data = Data(
                x=x_base,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor([target_idx], dtype=torch.long)
            )

            # ===== STEP 6: Add temporal features (masks) =====
            pyg_data.derived_mask = compute_derived_mask(
                {'num_nodes': num_nodes, 'derivations': [
                    (id2idx.get(step_data['derived_node']), i)
                    for i, step_data in enumerate(proof_steps[:step_idx])
                    if id2idx.get(step_data.get('derived_node')) is not None
                ]},
                step_idx
            )
            pyg_data.step_numbers = compute_step_numbers(
                {'num_nodes': num_nodes, 'derivations': [
                    (id2idx.get(step_data['derived_node']), i)
                    for i, step_data in enumerate(proof_steps[:step_idx])
                    if id2idx.get(step_data.get('derived_node')) is not None
                ]},
                step_idx
            )
            
            # ===== STEP 7: ATTACH APPLICABLE MASK =====
            pyg_data.applicable_mask = applicable_mask.to(torch.bool)
            
            # ===== STEP 8: Add Value and Difficulty =====
            proof_length = metadata.get('proof_length', len(proof_steps))
            if proof_length == 0: proof_length = 1
            pyg_data.value_target = torch.tensor([1.0 - (step_idx / float(proof_length))], dtype=torch.float)
            pyg_data.difficulty = torch.tensor([self._estimate_difficulty(metadata)], dtype=torch.float)
            
            # ===== STEP 9: Add Tactic Target =====
            rule_name = nodes[target_idx].get("label", "")
            tactic_idx = self.tactic_mapper.map_rule_to_tactic(rule_name)
            pyg_data.tactic_target = torch.tensor([tactic_idx], dtype=torch.long)
            
            # ===== STEP 10: Load spectral features =====
            k_target = self.k_dim
            eigvals = torch.zeros((k_target,), dtype=torch.float)
            eigvecs = torch.zeros((num_nodes, k_target), dtype=torch.float)
            eig_mask = torch.zeros((k_target,), dtype=torch.bool) # Default to invalid
            
            if self.spectral_dir and instance_id:
                spectral_path = Path(self.spectral_dir) / f"{instance_id}_spectral.npz"
                if spectral_path.exists():
                    try:
                        data = np.load(spectral_path)
                        eigvals_np = data["eigenvalues"].astype(np.float32)
                        eigvecs_np = data["eigenvectors"].astype(np.float32)
                        
                        if eigvecs_np.shape[0] != num_nodes:
                            raise ValueError(
                                f"Node mismatch: spectral={eigvecs_np.shape[0]}, "
                                f"graph={num_nodes}"
                            )
                        
                        current_k = len(eigvals_np)
                        k_to_use = min(current_k, k_target)
                        
                        eigvals[:k_to_use] = torch.from_numpy(eigvals_np[:k_to_use])
                        eigvecs[:, :k_to_use] = torch.from_numpy(eigvecs_np[:, :k_to_use])
                        eig_mask[:k_to_use] = True
                        
                    except Exception as e:
                        logger.debug(f"Warning: Failed to load spectral features for {instance_id}: {e}")
                        # Fallback to zeros (already initialized)
                else:
                    logger.debug(f"Warning: No spectral file found for {instance_id} at {spectral_path}")

            pyg_data.eigvecs = eigvecs.contiguous()
            pyg_data.eigvals = eigvals.contiguous()
            pyg_data.eig_mask = eig_mask.contiguous()
            
            # ===== STEP 11: Add metadata (for curriculum) =====
            pyg_data.meta = {
                "instance_id": instance_id,
                "step_idx": step_idx,
                "num_nodes": num_nodes,
                "proof_length": proof_length # <-- CRITICAL FOR CURRICULUM
            }
            
            self.stats['successful_accesses'] += 1
            return pyg_data
        
        except Exception as e:
            # This is critical. If __getitem__ fails, the dataloader worker
            # will catch it and return None.
            self.stats['failed_accesses'] += 1
            instance_id = self.proof_steps_db[idx].get('instance_id', 'unknown')
            step_idx = self.proof_steps_db[idx].get('step_idx', 'unknown')
            logger.error(
                f"[Dataset Error] Failed __getitem__ for instance {instance_id}, step {step_idx}: {e}"
            )
            return None # Must return None so collate_fn can filter it
    
    def _compute_node_features(self, inst: Dict, step_idx: int, known_atoms_set: set, id2idx: dict) -> torch.Tensor:
        """Compute 22-dimensional node features"""
        
        nodes = inst['nodes']
        edges = inst['edges']
        proof_steps = inst['proof_steps']
        
        n_nodes = len(nodes)
        feats = torch.zeros(n_nodes, self.feature_dim, dtype=torch.float32)
        
        # Compute derived nodes up to this step
        derived_nids = set()
        initial_nids = set()
        
        for node in nodes:
            if node['type'] == 'fact' and node.get('is_initial', False):
                initial_nids.add(node['nid'])
        
        for i in range(step_idx):
            derived_nids.add(proof_steps[i]['derived_node'])
        
        known_nids = initial_nids | derived_nids
        
        # Compute graph properties
        in_deg = np.zeros(n_nodes)
        out_deg = np.zeros(n_nodes)
        
        for edge in edges:
            src_idx = id2idx.get(edge['src'], -1)
            dst_idx = id2idx.get(edge['dst'], -1)
            if src_idx >= 0 and dst_idx >= 0:
                out_deg[src_idx] += 1
                in_deg[dst_idx] += 1
        max_deg = max(max(in_deg), max(out_deg), 1.0)
        
        # Compute depths (BFS from initial facts)
        depths = {}
        queue = deque()
        for nid in initial_nids:
            depths[nid] = 0
            queue.append((nid, 0))
        visited = set(initial_nids)
        max_depth_val = 0
        while queue:
            nid, depth = queue.popleft()
            max_depth_val = max(max_depth_val, depth)
            for edge in edges:
                if edge['src'] == nid and id2idx.get(edge['dst'], -1) != -1:
                    next_nid = edge['dst']
                    if next_nid not in visited:
                        visited.add(next_nid)
                        depths[next_nid] = depth + 1
                        queue.append((next_nid, depth + 1))
        max_depth = max(max_depth_val, 1.0)
        
        # Atom counts
        atom_counts = defaultdict(int)
        for node in nodes:
            atom = node.get('atom', node.get('head_atom', ''))
            atom_counts[atom] += 1
        max_count = max(atom_counts.values()) if atom_counts else 1
        
        # Centrality (simplified, as NX is slow)
        try:
            G_nx = nx.DiGraph()
            G_nx.add_nodes_from([n["nid"] for n in nodes])
            G_nx.add_edges_from([(e["src"], e["dst"]) for e in edges if e["src"] in id2idx and e["dst"] in id2idx])
            
            # Use k=None for small graphs, k=50 (approx) for larger
            k_approx = min(n_nodes, 50) if n_nodes > 50 else None
            betweenness = nx.betweenness_centrality(G_nx, k=k_approx, seed=42, normalized=True)
        except Exception:
            betweenness = {n["nid"]: 0.0 for n in nodes}
        max_betweenness = max(betweenness.values()) if betweenness else 1.0
        if max_betweenness == 0: max_betweenness = 1.0

        # Fill features for each node
        for i, node in enumerate(nodes):
            nid = node['nid']
            
            # [0-1]: Type one-hot
            feats[i, 0] = 1.0 if node['type'] == 'fact' else 0.0
            feats[i, 1] = 1.0 if node['type'] == 'rule' else 0.0
            
            # [2]: Is initial
            feats[i, 2] = 1.0 if nid in initial_nids else 0.0
            # [3]: Is derived
            feats[i, 3] = 1.0 if nid in derived_nids else 0.0
            # [4-5]: Normalized degrees
            feats[i, 4] = in_deg[i] / max_deg
            feats[i, 5] = out_deg[i] / max_deg
            # [6]: Depth
            feats[i, 6] = depths.get(nid, max_depth) / max_depth
            
            # [7]: Body size (for rules)
            if node['type'] == 'rule':
                body_size = len(node.get('body_atoms', []))
                max_body = max((len(n.get('body_atoms', [])) for n in nodes if n['type'] == 'rule'), default=1)
                feats[i, 7] = body_size / max(max_body, 1.0)
            
            # [8]: Centrality (original)
            feats[i, 8] = (in_deg[i] + out_deg[i]) / (2 * max_deg)
            
            # [9-16]: Predicate hash (8-bit)
            atom = node.get('atom', node.get('head_atom', ''))
            pred_hash = abs(hash(atom.replace('~', ''))) % 256
            for bit in range(8):
                feats[i, 9 + bit] = float((pred_hash >> bit) & 1)
            
            # [17]: Graph size
            feats[i, 17] = len(nodes) / 100.0
            # [18]: Betweenness centrality
            feats[i, 18] = betweenness.get(nid, 0.0) # Already normalized by NX
            
            # [19]: Is head of rule derived (for rules)
            if node["type"] == "rule":
                head_atom = node.get("head_atom", "")
                feats[i, 19] = 1.0 if head_atom in known_atoms_set else 0.0
            
            # [20]: Atom frequency
            atom = node.get('atom', node.get('head_atom', ''))
            feats[i, 20] = atom_counts[atom] / max_count
            
            # [21]: Rule applicability (fraction of body satisfied)
            if node['type'] == 'rule':
                body_atoms = set(node.get('body_atoms', []))
                if body_atoms:
                    body_satisfied = sum(1 for atom in body_atoms if atom in known_atoms_set)
                    feats[i, 21] = body_satisfied / len(body_atoms)
                else:
                    feats[i, 21] = 0.0 # Rule with no body
            
        return feats
    
    def _estimate_difficulty(self, metadata: dict) -> float:
        """Normalized difficulty estimation [0, 1]"""
        n_rules = metadata.get('n_rules', 10)
        n_nodes = metadata.get('n_nodes', 20)
        proof_length = metadata.get('proof_length', 5)
        
        rule_score = min(n_rules / 40.0, 1.0)
        node_score = min(n_nodes / 70.0, 1.0)
        proof_score = min(proof_length / 15.0, 1.0)
        
        difficulty = (0.3 * rule_score + 0.2 * node_score + 0.5 * proof_score)
        return min(max(difficulty, 0.0), 1.0)

    def report(self):
        """Print access report (for instrumentation)"""
        if not self.enable_instrumentation:
            return
        
        logger.info("\n" + "="*80)
        logger.info("DATASET ACCESS REPORT")
        logger.info("="*80)
        logger.info(f"Total accesses: {self.stats['total_accesses']}")
        logger.info(f"Successful: {self.stats['successful_accesses']}")
        logger.info(f"Failed (returned None): {self.stats['failed_accesses']}")
        logger.info(f"Inapplicable targets (raised error): {self.stats['inapplicable_targets']}")
        logger.info("="*80)

# ==============================================================================
# HELPER: Data Split
# ==============================================================================

def create_split(
    json_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> tuple:
    """Create train/val/test split at INSTANCE level."""
    all_files = list(Path(json_dir).rglob("*.json"))
    
    instance_map = defaultdict(list)
    for f in all_files:
        try:
            # Get instance ID from filename, e.g., "easy_0.json" -> "easy_0"
            inst_id = f.stem
            instance_map[inst_id].append(str(f))
        except:
            continue
    
    instance_ids = list(instance_map.keys())
    random.seed(seed)
    random.shuffle(instance_ids)
    
    n = len(instance_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_ids = instance_ids[:n_train]
    val_ids = instance_ids[n_train:n_train + n_val]
    test_ids = instance_ids[n_train + n_val:]
    
    # Get all files for the selected instance IDs
    train_files = [f for i in train_ids for f in instance_map[i]]
    val_files = [f for i in val_ids for f in instance_map[i]]
    test_files = [f for i in test_ids for f in instance_map[i]]
    
    print(f"\nDataset split (instance-level):")
    print(f"  Train: {len(train_files)} files ({len(train_ids)} instances)")
    print(f"  Val:   {len(val_files)} files ({len(val_ids)} instances)")
    print(f"  Test:  {len(test_files)} files ({len(test_ids)} instances)")
    
    assert len(set(train_ids) & set(val_ids)) == 0
    assert len(set(train_ids) & set(test_ids)) == 0
    assert len(set(val_ids) & set(test_ids)) == 0
    
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
    train_dataset = ProofStepDataset(
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
    
    logger.info("\n✅ DataLoaders created with fixed_collate_fn.")
    logger.info(f"   Train batches: {len(train_loader)}")
    logger.info(f"   Val batches: {len(val_loader)}")
    logger.info(f"   Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader