# IN: diagnose.py (FINAL, SELF-CONTAINED)
"""
Comprehensive Diagnostic Script
================================

FIXED: This script is now SELF-CONTAINED.
It no longer imports 'fixed_collate_fn' from dataset_utils.
The correct collate function is defined *inside this file*
to bypass any caching or import path issues.
"""

import torch
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
import argparse
from tqdm import tqdm
import random
from typing import List, Optional

# === CRITICAL: Import all dependencies ===
try:
    from dataset import ProofStepDataset, create_dataloaders
    from model import CriticallyFixedProofGNN
    from losses import get_recommended_loss
    from torch_geometric.loader import DataLoader as GeoDataLoader
    from torch_geometric.data import Batch, Data
except ImportError as e:
    print(f"Failed to import a module: {e}")
    print("Please ensure dataset.py, model.py, and losses.py are correct.")
    exit(1)


# ==============================================================================
# SELF-CONTAINED COLLATE FUNCTION
# (Copied from dataset_utils.py to guarantee it's the one being used)
# ==============================================================================
def fixed_collate_fn_LOCAL(batch_list: List[Data]) -> Batch:
    """
    Custom collation that preserves graph-level metadata.
    This is defined locally to avoid any import/cache issues.
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
        print(f"ERROR during Batch.from_data_list: {e}")
        print("This might be a PyG version issue or a data misalignment.")
        return None

    # === THIS IS THE FIX WE ARE TESTING ===
    # Add critical metadata for index mapping
    try:
        batch.num_nodes_per_graph = torch.tensor(
            [data.x.shape[0] for data in batch_list],
            dtype=torch.long
        )
        
        # Cumulative node offsets for index translation
        batch.node_offsets = torch.cat([
            torch.tensor([0]),
            torch.cumsum(batch.num_nodes_per_graph[:-1], dim=0)
        ])
    except Exception as e:
        print(f"ERROR while adding metadata in collate fn: {e}")
        return None
    # === END FIX ===
    
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


def diagnose_dataset(data_dir: str, spectral_dir: str = None, max_samples: int = 100):
    """Comprehensive dataset diagnostics"""
    
    print("\n" + "="*80)
    print("📊 DATASET DIAGNOSTICS")
    print("="*80)
    
    json_files = list(Path(data_dir).glob("**/*.json"))
    
    if len(json_files) == 0:
        print("❌ No JSON files found!")
        return False

    if len(json_files) > max_samples:
        json_files = random.sample(json_files, max_samples)
    
    print(f"\n✅ Found {len(json_files)} files to analyze...")
    
    stats = defaultdict(int)
    stats['node_counts'] = []
    stats['edge_counts'] = []
    stats['proof_lengths'] = []
    issues = []
    
    for json_file in tqdm(json_files, desc="Analyzing files"):
        try:
            with open(json_file) as f:
                inst = json.load(f)
            
            nodes = inst.get('nodes', [])
            edges = inst.get('edges', [])
            proof_steps = inst.get('proof_steps', [])
            
            stats['node_counts'].append(len(nodes))
            stats['edge_counts'].append(len(edges))
            stats['proof_lengths'].append(len(proof_steps))
            
            if len(proof_steps) == 0:
                stats['empty_proofs'] += 1
                issues.append(f"{json_file.name}: Empty proof")
                continue
            
            nid_to_idx = {n['nid']: i for i, n in enumerate(nodes)}
            
            # --- START FIX: Correctly simulate proof state ---
            known_atoms = set()
            for node in nodes:
                if node.get('is_initial', False) and node.get('type') == 'fact':
                    if node.get('atom'):
                        known_atoms.add(node.get('atom'))
            
            for step_idx, step in enumerate(proof_steps):
                rule_nid = step.get('used_rule')
                derived_nid = step.get('derived_node')
                
                if rule_nid not in nid_to_idx:
                    stats['invalid_targets'] += 1
                    issues.append(f"{json_file.name} step {step_idx}: rule {rule_nid} not in nodes")
                    continue
                
                if derived_nid not in nid_to_idx:
                    stats['invalid_targets'] += 1
                    issues.append(f"{json_file.name} step {step_idx}: derived_node {derived_nid} not in nodes")
                    continue
                
                rule_node = nodes[nid_to_idx[rule_nid]]
                derived_node = nodes[nid_to_idx[derived_nid]]
                
                body_atoms = set(rule_node.get('body_atoms', []))
                head_atom = rule_node.get('head_atom')
                derived_atom = derived_node.get('atom')

                if head_atom != derived_atom:
                     stats['head_mismatch'] += 1
                     issues.append(f"{json_file.name} step {step_idx}: Rule head '{head_atom}' != derived '{derived_atom}'")
                
                # Check applicability *before* adding the new fact
                is_applicable = body_atoms.issubset(known_atoms)
                head_already_known = head_atom in known_atoms
                
                if not is_applicable or head_already_known:
                    stats['inapplicable_targets'] += 1
                    missing = body_atoms - known_atoms
                    issues.append(
                        f"{json_file.name} step {step_idx} (rule {rule_nid}): "
                        f"NOT APPLICABLE. Missing: {missing}, HeadKnown: {head_already_known}"
                    )
                
                # Add the new fact for the *next* step's check
                if derived_atom:
                    known_atoms.add(derived_atom)
            # --- END FIX ---
        
        except Exception as e:
            stats['load_errors'] += 1
            issues.append(f"{json_file.name}: Failed to load ({e})")
    
    # Print statistics
    print(f"\n📈 Statistics:")
    print(f"   Files analyzed: {len(json_files)}")
    print(f"   Empty proofs: {stats['empty_proofs']}")
    print(f"   Invalid targets: {stats['invalid_targets']}")
    print(f"   Head mismatches: {stats['head_mismatch']}")
    print(f"   Inapplicable targets: {stats['inapplicable_targets']}")
    
    if stats['node_counts']:
        print(f"\n   Node count: {np.mean(stats['node_counts']):.1f} ± {np.std(stats['node_counts']):.1f}")
        print(f"   Edge count: {np.mean(stats['edge_counts']):.1f} ± {np.std(stats['edge_counts']):.1f}")
        print(f"   Proof length: {np.mean(stats['proof_lengths']):.1f} ± {np.std(stats['proof_lengths']):.1f}")
    
    # Print issues
    if stats['inapplicable_targets'] > 0 or stats['invalid_targets'] > 0 or stats['head_mismatch'] > 0:
        print(f"\n⚠️  Found {len(issues)} critical issues:")
        for issue in issues[:10]:  # Show first 10
            print(f"   - {issue}")
        if len(issues) > 10:
            print(f"   ... and {len(issues) - 10} more")
        return False
    else:
        print("\n✅ No data quality issues found!")
        return True


def diagnose_batching(data_dir: str, spectral_dir: str = None):
    """
    Test batching pipeline.
    FIXED: Uses the self-contained 'fixed_collate_fn_LOCAL'
    defined inside this script.
    """
    
    print("\n" + "="*80)
    print("📦 BATCHING DIAGNOSTICS")
    print("="*80)
    
    try:
        # --- START: Manual Dataloader Construction ---
        print("   Creating data splits...")
        from dataset import create_split
        train_files, _, _ = create_split(data_dir, train_ratio=0.7, val_ratio=0.15, seed=42)
        
        print("   Creating ProofStepDataset...")
        from dataset import ProofStepDataset
        train_dataset = ProofStepDataset(
            train_files, 
            spectral_dir=spectral_dir, 
            seed=42
        )
        
        print(f"   Using *LOCAL* fixed_collate_fn_LOCAL (self-contained)...")
        
        print("   Creating GeoDataLoader...")
        train_loader = GeoDataLoader(
            train_dataset, 
            batch_size=8, 
            shuffle=True, 
            num_workers=0,
            collate_fn=fixed_collate_fn_LOCAL  # <-- FORCING a successful batch
        )
        # --- END: Manual Dataloader Construction ---

        print(f"\n✅ Data loader created explicitly with *local* collate fn.")
        print(f"   Train batches: {len(train_loader)}")
        
        # Test a batch
        print(f"\n🔍 Testing first batch...")
        
        batch = None
        for i, b in enumerate(train_loader):
            if b is not None:
                batch = b
                print(f"   Loaded first valid batch (index {i}).")
                break
            if i > 50: # Fail-safe
                print("   Checked 50 batches, all returned None. __getitem__ is failing.")
                return False

        if batch is None:
            print(f"   ❌ Batch is None! This means __getitem__ is failing for all items.")
            print(f"   Check your file paths. Is '--spectral-dir {spectral_dir}' correct?")
            return False
        
        print(f"   Batch size: {batch.num_graphs if hasattr(batch, 'num_graphs') else 1}")
        print(f"   Total nodes: {batch.x.shape[0]}")
        print(f"   Total edges: {batch.edge_index.shape[1]}")
        
        # Check for batching metadata
        if hasattr(batch, 'num_nodes_per_graph'):
            print(f"   ✅ Has num_nodes_per_graph: {batch.num_nodes_per_graph.tolist()}")
        else:
            print(f"   ❌ Missing num_nodes_per_graph!")
            return False
        
        if hasattr(batch, 'node_offsets'):
            print(f"   ✅ Has node_offsets: {batch.node_offsets.tolist()}")
        else:
            print(f"   ❌ Missing node_offsets!")
            return False
        
        if hasattr(batch, 'y'):
            print(f"\n🎯 Target validation:")
            for i in range(batch.num_graphs):
                target = batch.y[i].item()
                start = batch.node_offsets[i].item()
                end = start + batch.num_nodes_per_graph[i].item()
                
                in_range = start <= target < end
                print(f"   Graph {i}: target={target}, range=[{start}, {end}), valid={in_range}")
                
                if not in_range:
                    print(f"   ❌ Target out of range!")
                    return False
        
        # Check applicable mask (if it was followed correctly)
        if hasattr(batch, 'applicable_mask'):
            print(f"\n✅ Has applicable_mask (batched tensor)")
            
            for i in range(batch.num_graphs):
                if i >= len(batch.y): break
                
                # Get local mask and local target
                start = batch.node_offsets[i].item()
                end = start + batch.num_nodes_per_graph[i].item()
                graph_mask = batch.applicable_mask[start:end]
                
                target_global = batch.y[i].item()
                target_local = target_global - start

                if target_local < 0 or target_local >= len(graph_mask):
                    print(f"   ❌ Graph {i}: Local target {target_local} out of range for mask (len {len(graph_mask)})")
                    return False
                
                is_applicable = graph_mask[target_local].item()
                if not is_applicable:
                    print(f"   ❌ Graph {i}: target {target_global} (local {target_local}) not applicable!")
                    return False
            print("   ✅ All targets are applicable.")
        else:
            print(f"\n⚠️  No applicable_mask found (batched). This might be ok.")
        
        print("\n✅ Batching pipeline looks good!")
        return True
    
    except Exception as e:
        print(f"\n❌ Batching failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def diagnose_model(hidden_dim=256, k_dim=16):
    """Test model forward pass"""
    
    print("\n" + "="*80)
    print("🧠 MODEL DIAGNOSTICS")
    print("="*80)
    
    try:
        print("   Initializing CriticallyFixedProofGNN (as used in train.py)...")
        model = CriticallyFixedProofGNN(
            in_dim=22,
            hidden_dim=hidden_dim,
            num_layers=3,
            dropout=0.3,
            k=k_dim
        )
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"\n✅ Model created: {num_params:,} parameters")
        
        # Create dummy batch
        num_nodes = 30
        num_edges = 60
        
        batch = type('Batch', (), {
            'x': torch.randn(num_nodes, 22),
            'edge_index': torch.randint(0, num_nodes, (2, num_edges)),
            'edge_attr': torch.randint(0, 3, (num_edges,)),
            'derived_mask': torch.randint(0, 2, (num_nodes,), dtype=torch.uint8),
            'step_numbers': torch.randint(0, 10, (num_nodes,)),
            'eigvecs': torch.randn(num_nodes, k_dim),
            'eigvals': torch.randn(k_dim),
            'eig_mask': torch.ones(k_dim, dtype=torch.bool),
            'batch': torch.zeros(num_nodes, dtype=torch.long),
            'num_graphs': 1,
            'y': torch.tensor([5]),
            'value_target': torch.tensor([0.5]),
            'applicable_mask': torch.ones(num_nodes, dtype=torch.bool),
            'node_offsets': torch.tensor([0]),
            'num_nodes_per_graph': torch.tensor([num_nodes])
        })()
        
        print(f"\n🔍 Testing forward pass...")
        scores, embeddings, value = model(batch)
        
        print(f"   ✅ Scores shape: {scores.shape}")
        print(f"   ✅ Embeddings shape: {embeddings.shape}")
        print(f"   ✅ Value shape: {value.shape}")
        
        assert scores.shape == (num_nodes,)
        assert embeddings.shape == (num_nodes, hidden_dim)
        assert value.shape == (1,)
        
        if torch.isnan(scores).any():
            print(f"   ❌ NaN in scores!")
            return False
        
        if torch.isinf(scores).any():
            print(f"   ❌ Inf in scores!")
            return False
        
        print(f"\n✅ Model forward pass successful!")
        return True
    
    except Exception as e:
        print(f"\n❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def diagnose_loss():
    """Test loss computation"""
    
    print("\n" + "="*80)
    print("💔 LOSS DIAGNOSTICS")
    print("="*80)
    
    loss_types = ['cross_entropy', 'triplet_hard', 'applicability_constrained']
    
    for loss_type in loss_types:
        print(f"\n🔍 Testing {loss_type}...")
        
        try:
            criterion = get_recommended_loss(loss_type, margin=1.0)
            
            num_nodes = 20
            scores = torch.randn(num_nodes, requires_grad=True)
            embeddings = torch.randn(num_nodes, 256)
            target_idx = 5
            applicable_mask = torch.ones(num_nodes, dtype=torch.bool)
            applicable_mask[target_idx] = True
            
            loss = criterion(scores, embeddings, target_idx, applicable_mask)
            
            print(f"   Loss value: {loss.item():.4f}")
            
            if torch.isnan(loss):
                print(f"   ❌ NaN loss!")
                return False
            
            if torch.isinf(loss):
                print(f"   ❌ Inf loss!")
                return False
            
            loss.backward()
            print(f"   ✅ Backward pass successful")
        
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            return False
    
    print(f"\n✅ All loss functions working!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Diagnose your pipeline")
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--spectral-dir', default=None)
    parser.add_argument('--max-samples', type=int, default=100)
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🔬 COMPREHENSIVE PIPELINE DIAGNOSTICS")
    print("="*80)
    
    results = {
        'dataset': diagnose_dataset(args.data_dir, args.spectral_dir, args.max_samples),
        'batching': diagnose_batching(args.data_dir, args.spectral_dir),
        'model': diagnose_model(),
        'loss': diagnose_loss()
    }
    
    print("\n" + "="*80)
    print("📋 SUMMARY")
    print("="*80)
    
    for component, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {component.upper()}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print(f"\n🎉 All diagnostics passed! Ready to train.")
    else:
        print(f"\n⚠️  Some diagnostics failed. Fix issues before training.")
    
    return all_passed


if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)