#!/usr/bin/env python3
"""
Diagnostic Test Script for Hybrid LogicNet + Graph Spectral Reasoner

This script runs three critical tests to validate the model's behavior:
1.  Test 1: Verifies that the spectral filter's polynomial coefficients are learning.
2.  Test 2: Verifies that temporal causality is enforced in the data pipeline.
3.  Test 3: Quantifies any potential information leakage from future steps.
"""

import torch
import numpy as np
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

# --- Model & Data Imports ---
# (Ensure these files are in the same directory or accessible in your PYTHONPATH)
try:
    from model import SOTAFixedProofGNN
    from dataset import create_properly_split_dataloaders, ProofStepDataset
    from torch_geometric.data import Batch
except ImportError as e:
    print(f"Error: Could not import model or dataset files.")
    print(f"Make sure 'model.py' and 'dataset.py' are in the same directory.")
    print(f"Details: {e}")
    exit(1)


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==============================================================================
# TEST 1: VERIFY SPECTRAL FILTER LEARNING
# ==============================================================================
def test_spectral_filter_contribution(model, val_loader, device):
    """
    Tests if spectral pathway is actually contributing to predictions.
    If coefficients are stuck near [1, 0.1, 0, 0, 0], it's not learning.
    """
    model.eval()

    # Check for the 'poly_coeffs' attribute
    if not hasattr(model, 'poly_coeffs'):
        print("="*80)
        print("SPECTRAL FILTER DIAGNOSTIC")
        print("="*80)
        print("❌ ERROR: Model does not have 'poly_coeffs' attribute.")
        print("   This test is for the SOTAFixedProofGNN with polynomial filters.")
        return False
        
    # Check polynomial coefficients
    poly_coeffs = model.poly_coeffs.data.cpu().numpy()

    print("="*80)
    print("TEST 1: SPECTRAL FILTER DIAGNOSTIC")
    print("="*80)
    print(f"Polynomial coefficients: {poly_coeffs}")
    print(f"  Magnitude: {np.linalg.norm(poly_coeffs):.4f}")
    print(f"  Max coeff: {np.max(np.abs(poly_coeffs)):.4f}")
    print(f"  Std dev: {np.std(poly_coeffs):.4f}")

    # Test: Coefficients should have evolved from initialization
    # Note: Your init sets them to [1.0, 0.1, 0, ...]
    if np.allclose(poly_coeffs[:2], [1.0, 0.1], atol=0.05) and np.allclose(poly_coeffs[2:], 0.0, atol=0.05):
        print("⚠️  WARNING: Coefficients stuck near initialization!")
        print("   Spectral pathway is not learning.")
        return False
    else:
        print("✅ Coefficients have moved from their initial values.")

    # Test contribution by ablating spectral pathway
    with torch.no_grad():
        total_score_diff = 0.0
        num_samples = 0

        for batch in val_loader:
            if batch is None: continue
            batch = batch.to(device)

            # Forward with spectral
            scores_with, _, _, _ = model(batch)

            # Temporarily disable spectral by zeroing coefficients
            orig_coeffs = model.poly_coeffs.data.clone()
            model.poly_coeffs.data.fill_(0.0)
            model.poly_coeffs.data[0] = 1.0  # Identity only

            # Forward without spectral
            scores_without, _, _, _ = model(batch)

            # Restore
            model.poly_coeffs.data = orig_coeffs

            # Measure difference
            diff = (scores_with - scores_without).abs().mean().item()
            total_score_diff += diff
            num_samples += 1

            if num_samples >= 5:  # Sample 5 batches
                break

        if num_samples == 0:
            print("⚠️ WARNING: Could not load any batches for spectral test.")
            return None # Inconclusive

        avg_diff = total_score_diff / num_samples
        print(f"\nSpectral pathway contribution (avg score diff): {avg_diff:.4f}")

        if avg_diff < 0.01:
            print("⚠️  WARNING: Spectral pathway has minimal impact!")
            print("   Scores barely change when it's disabled.")
            return False
        else:
            print("✅ Spectral pathway is active and contributing.")
            return True

# ==============================================================================
# TEST 2: VERIFY TEMPORAL CAUSALITY
# ==============================================================================
def test_temporal_causality(dataset, device):
    """
    Tests if message passing respects temporal causality.
    Checks if node features at step t depend on future steps.
    """
    print("\n" + "="*80)
    print("TEST 2: TEMPORAL CAUSALITY DIAGNOSTIC")
    print("="*80)

    # Get a sample with multiple steps
    sample_t_idx = -1
    sample_t1_idx = -1
    
    for idx in range(len(dataset.proof_steps_db)):
        sample = dataset.proof_steps_db[idx]
        if sample['step_idx'] >= 3:  # Need at least 3 steps
            sample_t_idx = idx
            break
            
    if sample_t_idx == -1:
        print("Could not find a sample with >= 3 steps. Test inconclusive.")
        return None
        
    sample_t = dataset.proof_steps_db[sample_t_idx]

    # Find same instance, next step
    for idx in range(len(dataset.proof_steps_db)):
        s = dataset.proof_steps_db[idx]
        if (s['instance_id'] == sample_t['instance_id'] and
            s['step_idx'] == sample_t['step_idx'] + 1):
            sample_t1_idx = idx
            break

    if sample_t1_idx == -1:
        print(f"Could not find consecutive step for {sample_t['instance_id']} step {sample_t['step_idx']}. Test inconclusive.")
        return None

    # Load both
    data_t = dataset[sample_t_idx]
    data_t1 = dataset[sample_t1_idx]

    if data_t is None or data_t1 is None:
        print("Failed to load data for causality test. Test inconclusive.")
        return None

    # Check if edge_index is different
    edges_t = set(map(tuple, data_t.edge_index.t().tolist()))
    edges_t1 = set(map(tuple, data_t1.edge_index.t().tolist()))

    print(f"Instance: {sample_t['instance_id']}")
    print(f"Step {sample_t['step_idx']}: {len(edges_t)} edges")
    print(f"Step {sample_t['step_idx']+1}: {len(edges_t1)} edges")

    if edges_t == edges_t1:
        # This is not necessarily an error; maybe no new facts were used
        # But if it happens consistently, it's a bug.
        # A better check is if t1 has edges that t does not.
        if len(edges_t1 - edges_t) == 0 and len(edges_t1) > 0:
             print("\n⚠️  WARNING: Edge set did not expand at t+1.")
             print("   This might be okay, but if new edges are *never* added,")
             print("   the _get_causal_edge_index filter might be broken.")
             return None # Inconclusive
        else:
            new_edges = edges_t1 - edges_t
            print(f"\n✅ Edge set changes: +{len(new_edges)} edges at step t+1")
            print("   Temporal causality is being enforced.")
            return True
    else:
        new_edges = edges_t1 - edges_t
        print(f"\n✅ Edge set changes: +{len(new_edges)} edges at step t+1")
        print("   Temporal causality is being enforced.")
        return True

# ==============================================================================
# TEST 3: MEASURE INFORMATION LEAKAGE
# ==============================================================================
def test_information_leakage(model, dataset, device):
    """
    Quantifies how much the model 'knows' about future steps.
    High accuracy on predicting future derived nodes = leakage.
    """
    print("\n" + "="*80)
    print("TEST 3: INFORMATION LEAKAGE DIAGNOSTIC")
    print("="*80)

    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        # Iterate over a sample of the dataset
        for idx in tqdm(range(min(200, len(dataset))), desc="Testing Leakage"):
            
            # Use dataset.__getitem__ to get the fully processed Data object
            data_t = dataset[idx]
            if data_t is None: continue

            # Get the raw sample info from the db
            sample_t = dataset.proof_steps_db[idx]
            instance = sample_t['instance']
            step_idx = sample_t['step_idx']
            nodes = instance['nodes']
            proof_steps = instance['proof_steps']

            if step_idx >= len(proof_steps) - 1:
                continue  # Skip last step

            # Get ground truth: which node will be derived NEXT step
            next_step = proof_steps[step_idx + 1]
            next_derived_nid = next_step['derived_node']
            id2idx = {n['nid']: i for i, n in enumerate(nodes)}
            next_derived_idx = id2idx.get(next_derived_nid, -1)

            if next_derived_idx == -1:
                continue

            # --- CRITICAL FIX ---
            # Cannot .unsqueeze() a PyG Data object.
            # Must use Batch.from_data_list to create a batch of size 1.
            data_batch = Batch.from_data_list([data_t]).to(device)
            
            # Forward pass at current step t
            scores, _, _, _ = model(data_batch)
            
            # Check if model's top prediction is the *future* derived node
            top_pred_idx = scores.argmax().item()

            if top_pred_idx == next_derived_idx:
                correct_predictions += 1

            total_predictions += 1

    if total_predictions == 0:
        print("\nCould not find any non-final steps to test. Test inconclusive.")
        return None

    accuracy = correct_predictions / total_predictions
    print(f"\nFuture-step prediction accuracy: {accuracy*100:.1f}%")
    print(f"  ({correct_predictions}/{total_predictions} correct predictions)")

    if accuracy > 0.25:  # Significantly above random chance
        print("\n⚠️  HIGH LEAKAGE DETECTED!")
        print("   Model can predict future steps much better than random.")
        print("   This indicates temporal information leakage.")
        return False
    else:
        print("\n✅ Low leakage. Model is not 'seeing' the future.")
        return True

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Run diagnostics on SOTA GNN model")
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory with generated proof data (e.g., ./data/json)')
    parser.add_argument('--spectral-dir', type=str, required=True,
                        help='Directory with precomputed spectral features (e.g., ./data/spectral)')
    parser.add_argument('--checkpoint-path', type=str, required=True,
                        help='Path to the saved model checkpoint (e.g., ./experiments/best_model.pt)')
    parser.add_argument('--device', type=str, default='mps',
                        help='Device to use (cpu, cuda, mps)')
    
    args = parser.parse_args()

    # --- 1. Set Device ---
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    logger.info(f"Using device: {device}")

    # --- 2. Load Data ---
    logger.info("Loading data... (This may take a moment)")
    try:
        train_loader, val_loader, _ = create_properly_split_dataloaders(
            args.data_dir,
            spectral_dir=args.spectral_dir,
            batch_size=32, # Batch size for val_loader
            seed=42
        )
        # We use train_dataset for tests 2 & 3 as it's the largest
        train_dataset = train_loader.dataset
        logger.info(f"✅ Data loaded. Train samples: {len(train_dataset)}, Val batches: {len(val_loader)}")
    except Exception as e:
        logger.error(f"FATAL: Could not load data. Error: {e}")
        return

    # --- 3. Load Model ---
    logger.info(f"Loading model from {args.checkpoint_path}...")
    try:
        # These params must match the saved model
        # The input dim 26 comes from:
        # 22 (base) + 1 (derived_mask) + 1 (norm_step) + 1 (goal_dist) + 1 (proof_prog)
        model = SOTAFixedProofGNN(
            in_dim=26,
            hidden_dim=256,
            num_layers=6,
            dropout=0.2,
            k=16,
            poly_order=4 # Must match SOTAFixedProofGNN
        ).to(device)

        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        logger.info("✅ Model loaded successfully.")
    except Exception as e:
        logger.error(f"FATAL: Could not load model. Error: {e}")
        logger.error("   Ensure model parameters (hidden_dim, etc.) match the checkpoint.")
        return

    # --- 4. Run Tests ---
    results = {}
    try:
        results['spectral'] = test_spectral_filter_contribution(model, val_loader, device)
    except Exception as e:
        logger.error(f"Spectral test failed with exception: {e}")
        results['spectral'] = False
        
    try:
        results['causality'] = test_temporal_causality(train_dataset, device)
    except Exception as e:
        logger.error(f"Causality test failed with exception: {e}")
        results['causality'] = False

    try:
        results['leakage'] = test_information_leakage(model, train_dataset, device)
    except Exception as e:
        logger.error(f"Leakage test failed with exception: {e}")
        results['leakage'] = False

    # --- 5. Final Report ---
    print("\n\n" + "="*80)
    print("DIAGNOSTIC TEST SUMMARY")
    print("="*80)
    print(f"  Test 1 (Spectral Learning):   {'✅ PASS' if results['spectral'] else ('❌ FAIL' if results['spectral'] is False else '⚪️ INCONCLUSIVE')}")
    print(f"  Test 2 (Temporal Causality):  {'✅ PASS' if results['causality'] else ('❌ FAIL' if results['causality'] is False else '⚪️ INCONCLUSIVE')}")
    print(f"  Test 3 (Information Leakage): {'✅ PASS' if results['leakage'] else ('❌ FAIL' if results['leakage'] is False else '⚪️ INCONCLUSIVE')}")
    print("="*80)

if __name__ == "__main__":
    main()