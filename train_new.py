"""
FIXED Training Script with Critical Fixes
==========================================

Changes:
1. Uses ProofStepDataset (proper data split, no leakage)
2. Uses CriticallyFixedProofGNN (causal masking, gated fusion)
3. Uses FocalApplicabilityLoss (hard negative mining)
4. Proper curriculum learning
5. Better validation and metrics
6. FIXED: Metadata preservation in curriculum loop

Key improvements:
- Instance-level split ensures no data leakage
- Causal masking prevents future-step visibility
- Gated fusion prevents pathway collapse
- Focal loss focuses on hard negatives
"""

import os
from torch.optim.lr_scheduler import OneCycleLR
from debug_spectral_disconnect import validate_spectral_loading
from loss_enhanced import FocalInfoNCELoss, ProductionInfoNCELoss
from model_fixed import FixedProofGNN, get_fixed_model
from curriculum_fixed import PhasedCurriculumScheduler, SmoothCurriculumScheduler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch_geometric.loader import DataLoader as GeoDataLoader
import numpy as np
import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import time
from typing import Dict, Tuple, Optional

# Import fixed modules
from dataset import ProofStepDataset, create_properly_split_dataloaders, fixed_collate_fn
from metrics import ProofMetricsCompute
from losses import ApplicabilityConstrainedLoss, ContrastiveRankingLoss, DecoupledApplicabilityRankingLoss, FocalApplicabilityLoss, HybridTripletListwiseValueLoss, HyperbolicProofLoss, ImbalanceAwareProofLoss, InfoNCEListwiseLoss, ProofAwareInfoNCELoss, SOTAInfoNCELoss, TheoreticallySoundLoss, TripletLossWithHardMining
from losses import FocusedRankingLoss
from temporal_encoder import CausalProofTemporalEncoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from torch.utils.data import Subset

# In train.py

# --- UTILS ---
def get_curriculum_loader(base_dataset, scheduler, epoch, batch_size, is_train=True):
    config = scheduler.get_phase_config(epoch)
    active_diffs = set(config['include_difficulties'])
    
    # Fast filtering
    valid_indices = []
    for idx in range(len(base_dataset)):
        sample = base_dataset.samples[idx]
        inst_id = sample[0]
        meta = base_dataset.instances[inst_id].get('metadata', {})
        if meta.get('difficulty') in active_diffs:
            valid_indices.append(idx)
            
    if len(valid_indices) == 0:
        valid_indices = list(range(len(base_dataset)))

    subset = Subset(base_dataset, valid_indices)
    return DataLoader(
        subset, batch_size=batch_size, shuffle=is_train, 
        num_workers=0, collate_fn=fixed_collate_fn
    )

def preserve_metadata(batch, device):
    attrs = ['difficulties', 'meta_list', 'step_indices', 'proof_lengths', 'node_offsets']
    meta_storage = {attr: getattr(batch, attr) for attr in attrs if hasattr(batch, attr)}
    batch_gpu = batch.to(device)
    for k, v in meta_storage.items():
        setattr(batch_gpu, k, v)
    return batch_gpu

# ==============================================================================
# METRICS COMPUTATION
# ==============================================================================
def compute_hit_at_k(scores, target_idx, k, applicable_mask=None):
    if target_idx < 0 or target_idx >= len(scores): return 0.0
    # Filter only applicable if mask provided? 
    # Standard Hit@K usually ranks against ALL nodes, but NTP often ranks against applicable.
    # We stick to global ranking for strictness, but check applicability.
    k = min(k, len(scores))
    top_k = torch.topk(scores, k).indices
    hit = 1.0 if target_idx in top_k else 0.0
    if applicable_mask is not None and not applicable_mask[target_idx]:
        hit = 0.0
    return hit

def compute_mrr(scores: torch.Tensor, target_idx: int,
                applicable_mask: Optional[torch.Tensor] = None) -> float:
    if target_idx < 0 or target_idx >= len(scores): return 0.0
    if applicable_mask is not None and not applicable_mask[target_idx]: return 0.0
    sorted_indices = torch.argsort(scores, descending=True)
    rank = (sorted_indices == target_idx).nonzero(as_tuple=True)[0].item() + 1
    return 1.0 / rank

def compute_applicable_accuracy(scores, target_idx, applicable_mask):
    app_indices = applicable_mask.nonzero(as_tuple=True)[0]
    if len(app_indices) == 0: return 0.0
    # Among applicable rules, which has highest score?
    app_scores = scores[app_indices]
    best_app_idx = app_indices[app_scores.argmax()]
    return 1.0 if best_app_idx == target_idx else 0.0

# ==============================================================================
# EVALUATION
# ==============================================================================
@torch.no_grad()
def evaluate(model, val_loader, criterion, device, split_name='val', 
             scheduler=None, epoch=None, ignore_curriculum=False):
    model.eval()
    total_loss = 0.0
    num_samples = 0
    
    hit_at_k = {1: 0.0, 3: 0.0, 5: 0.0, 10: 0.0}
    mrr_sum = 0.0
    app_acc_sum = 0.0
    
    active_difficulties = None
    max_len = float('inf')
    
    if scheduler is not None and epoch is not None and not ignore_curriculum:
        config = scheduler.get_phase_config(epoch)
        active_difficulties = set(config['include_difficulties'])
        max_len = config['max_proof_length']

    for batch in tqdm(val_loader, desc=f"Eval {split_name}", leave=False):
        if batch is None: continue
        
        # FIX: Preserve metadata
        batch = preserve_metadata(batch, device)
        
        scores, embeddings, value = model(batch)
        batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
        
        for i in range(batch_size):
            # --- CURRICULUM FILTERING ---
            if scheduler is not None and not ignore_curriculum: # <--- CHECK FLAG HERE
                meta = batch.meta_list[i]
                difficulty = meta['difficulty']
                proof_len = meta.get('proof_length', 0)
                if difficulty not in active_difficulties: continue
                if not ignore_curriculum and proof_len > max_len: continue

            mask = (batch.batch == i)
            graph_scores = scores[mask]
            graph_embeddings = embeddings[mask]
            
            if len(graph_scores) == 0: continue
            
            if hasattr(batch, 'ptr'):
                node_offset = batch.ptr[i].item()
            else:
                node_offset = 0
                
            target_idx_local = batch.y[i].item()

            if target_idx_local < 0 or target_idx_local >= len(graph_scores): continue
            
            graph_applicable = batch.applicable_mask[mask]

            try:
                loss = criterion(graph_scores, graph_embeddings, target_idx_local, graph_applicable)
                total_loss += loss.item()
            except: pass
            
            for k in [1, 3, 5, 10]:
                hit_at_k[k] += compute_hit_at_k(graph_scores, target_idx_local, k, graph_applicable)
            
            mrr_sum += compute_mrr(graph_scores, target_idx_local, graph_applicable)
            app_acc_sum += compute_applicable_accuracy(graph_scores, target_idx_local, graph_applicable)
            
            num_samples += 1
        
    n = max(num_samples, 1)
    return {
        f'{split_name}_loss': total_loss / n,
        f'{split_name}_hit@1': hit_at_k[1] / n,
        f'{split_name}_hit@3': hit_at_k[3] / n,
        f'{split_name}_hit@5': hit_at_k[5] / n,
        f'{split_name}_hit@10': hit_at_k[10] / n,
        f'{split_name}_mrr': mrr_sum / n,
        f'{split_name}_applicable_acc': app_acc_sum / n,
        f'{split_name}_num_samples': num_samples
    }

# ==============================================================================
# DEBUGGER
# ==============================================================================
# In train.py

class TrainingDebugger:
    def __init__(self, model, verbose=True):
        self.model = model
        self.verbose = verbose
        self.layer_stats = {}
        self.hooks = []
        
    def attach_hooks(self):
        def get_activation_hook(name):
            def hook(model, input, output):
                if isinstance(output, tuple): out_tensor = output[0]
                else: out_tensor = output
                stats = {
                    "shape": tuple(out_tensor.shape),
                    "mean": out_tensor.mean().item(),
                    "std": out_tensor.std().item(),
                    "dead_neurons": (out_tensor == 0).float().mean().item()
                }
                self.layer_stats[name] = stats
            return hook

        # 1. Spectral Pathway
        if hasattr(self.model, 'spectral_encoder'):
            self.hooks.append(self.model.spectral_encoder.register_forward_hook(get_activation_hook("Pathway: Spectral")))
            
        # 2. Spatial Pathway (UPDATED for Deep GNN)
        if hasattr(self.model, 'spatial_layers') and len(self.model.spatial_layers) > 0:
            # Hook the LAST spatial layer to see the final output
            self.hooks.append(self.model.spatial_layers[-1].register_forward_hook(get_activation_hook("Pathway: Spatial")))
        elif hasattr(self.model, 'spatial_gnn'):
            # Fallback for old single-layer model
            self.hooks.append(self.model.spatial_gnn.register_forward_hook(get_activation_hook("Pathway: Spatial")))

        # 3. Temporal Pathway
        if hasattr(self.model, 'temporal_encoder'):
            self.hooks.append(self.model.temporal_encoder.register_forward_hook(get_activation_hook("Pathway: Temporal")))
            
        # 4. Fusion Output
        if hasattr(self.model, 'fusion'):
            self.hooks.append(self.model.fusion.register_forward_hook(get_activation_hook("Layer: Fusion_Output")))
    def remove_hooks(self):
        for h in self.hooks: h.remove()
        self.hooks = []

    def check_batch_integrity(self, batch, epoch, scheduler):
        print(f"\n{'='*20} DIAGNOSTIC CHECKPOINT (Epoch {epoch}) {'='*20}")
        print(f"ÃƒÂ¢Ã¢â‚¬â€œÃ‚Â¸ Input Shapes:")
        print(f"   Batch Size (Graphs): {batch.num_graphs}")
        print(f"   Nodes (Total):       {batch.x.shape} (Expected [N, 32])")
        print(f"   Edge Index:          {batch.edge_index.shape}")
        
        # --- FIXED: Check for Magnetic Attributes ---
        if hasattr(batch, 'eigvecs_real'):
            print(f"   Eigvecs (Real):      {batch.eigvecs_real.shape} (Magnetic ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦)")
            print(f"   Eigvecs (Imag):      {batch.eigvecs_imag.shape}")
        elif hasattr(batch, 'eigvecs'):
            print(f"   Eigenvectors:        {batch.eigvecs.shape} (Standard)")
        else:
            print(f"   Eigenvectors:        MISSING ÃƒÂ¢Ã…Â¡ ÃƒÂ¯Ã‚Â¸Ã‚ ")
        # --------------------------------------------

        print(f"   Step Numbers:        Max={batch.step_numbers.max().item()} (PE Check)")

        config = scheduler.get_phase_config(epoch)
        
        difficulties = set(getattr(batch, 'difficulties', []))
        proof_lengths = getattr(batch, 'proof_lengths', [0])
        max_len_in_batch = max(proof_lengths) if proof_lengths else 0
        
        print(f"ÃƒÂ¢Ã¢â‚¬â€œÃ‚Â¸ Curriculum Check:")
        print(f"   Phase:           {config['phase']}")
        print(f"   Allowed Diff:    {config['include_difficulties']}")
        print(f"   Batch Diff:      {difficulties}")
        print(f"   Allowed Len:     {config['max_proof_length']}")
        print(f"   Batch Max Len:   {max_len_in_batch}")
        
        if not difficulties:
            print(f"   ÃƒÂ¢Ã…Â¡ ÃƒÂ¯Ã‚Â¸Ã‚  WARNING: Batch has NO difficulty metadata! Collate failed?")
        elif not difficulties.issubset(set(config['include_difficulties'])):
            print(f"   ÃƒÂ¢Ã…Â¡ ÃƒÂ¯Ã‚Â¸Ã‚  WARNING: CURRICULUM LEAKAGE DETECTED! Found {difficulties - set(config['include_difficulties'])}")
        else:
            print(f"   ÃƒÂ¢Ã…â€œÃ¢â‚¬Â¦ Curriculum Integrity Passed")

    def log_layer_stats(self):
        print(f"ÃƒÂ¢Ã¢â‚¬â€œÃ‚Â¸ Layer Outputs & Pathway Health:")
        for name, stats in self.layer_stats.items():
            print(f"   {name:20s} | Shape: {str(stats['shape']):15s} | Mean: {stats['mean']:.4f} | Std: {stats['std']:.4f} | Dead%: {stats['dead_neurons']:.1%}")
            if "Spectral" in name and stats['std'] < 1e-4:
                print(f"     ÃƒÂ¢Ã…Â¡ ÃƒÂ¯Ã‚Â¸Ã‚  CRITICAL: Spectral Pathway Collapsed! (Std ~ 0)")
            if "Temporal" in name and stats['std'] < 1e-4:
                print(f"     ÃƒÂ¢Ã…Â¡ ÃƒÂ¯Ã‚Â¸Ã‚  CRITICAL: Temporal PE Collapsed! (Std ~ 0)")

    def check_gradients(self):
        print(f"ÃƒÂ¢Ã¢â‚¬â€œÃ‚Â¸ Gradient Flow:")
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm == 0: print(f"   ÃƒÂ¢Ã…Â¡ ÃƒÂ¯Ã‚Â¸Ã‚  {name}: ZERO GRADIENT")
                elif torch.isnan(param.grad).any(): print(f"   ÃƒÂ¢Ã…Â¡ ÃƒÂ¯Ã‚Â¸Ã‚  {name}: NaN GRADIENT")
def get_dynamic_margin(epoch, total_epochs):
    if epoch < 20:
        return 0.5
    elif epoch < 40:
        return 0.5 + 0.5 * ((epoch - 20) / 20.0)
    else:
        return 1.0
# ==============================================================================
# CURRICULUM TRAIN EPOCH
# ==============================================================================
def train_epoch_with_curriculum(model, train_loader, 
                                main_optimizer, 
                                criterion, device, epoch, 
                                curriculum_scheduler,       # <--- ADD THIS ARGUMENT
                                scheduler,  # Keep LR schedulers separate
                                grad_accum_steps=4, value_loss_weight=0.1):
    model.train()
    
    debugger = TrainingDebugger(model)
    debugger.attach_hooks()
    has_checked_this_epoch = False
    
    config = curriculum_scheduler.get_phase_config(epoch)
    logger.info(f"Curriculum: {config['description']}")
    
    total_rank_loss = 0.0
    total_value_loss = 0.0
    total_accuracy = 0.0
    total_applicable_acc = 0.0
    num_samples = 0 
    num_batches_processed = 0
    
    main_optimizer.zero_grad()
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} Training", leave=True)
    
    for batch_idx, batch in enumerate(progress_bar):
        if batch is None: continue 
        if batch_idx == 0:
            temp_val = torch.exp(model.scorer.log_temperature).item()
            logger.info(f"ðŸŒ¡ï¸   Auto-Temperature: {temp_val:.4f}")
        # --- ROBUST METADATA PRESERVATION ---
        # 1. Capture metadata from CPU batch BEFORE moving to device
        meta_storage = {}
        attr_names = ['difficulties', 'meta_list', 'step_indices', 'proof_lengths', 'node_offsets']
        
        for attr in attr_names:
            if hasattr(batch, attr):
                meta_storage[attr] = getattr(batch, attr)
            else:
                # Only warn once per epoch to avoid spam
                if batch_idx == 0:
                    logger.warning(f"Batch missing attribute '{attr}' before GPU transfer!")

        # 2. Move to device (this destroys custom attributes)
        batch = batch.to(device)
        
        # 3. Restore metadata to the GPU batch object
        for attr, value in meta_storage.items():
            setattr(batch, attr, value)
        # ------------------------------------
        
        if not has_checked_this_epoch:
            # Now checking integrity is safe because we restored 'difficulties'
            debugger.check_batch_integrity(batch, epoch, curriculum_scheduler)
        
        # Forward pass
        scores, embeddings, value = model(batch)
        
        # Check stats after forward pass
        if not has_checked_this_epoch:
            debugger.log_layer_stats()
            if hasattr(model.fusion, 'last_diversity_loss'):
                 print(f"   Fusion Entropy Loss: {model.fusion.last_diversity_loss:.4f}")
            
        batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
        
        batch_loss = 0.0
        batch_value_loss = 0.0
        batch_rank_loss = 0.0
        batch_acc = 0.0
        batch_applicable_acc = 0.0
        graphs_processed_in_batch = 0
        
        for i in range(batch_size):
            mask = (batch.batch == i)
            graph_scores = scores[mask]
            graph_embeddings = embeddings[mask]
            
            if len(graph_scores) == 0: continue
            
            # Safe offset (restored from meta_storage or computed)
            if hasattr(batch, 'node_offsets') and isinstance(batch.node_offsets, torch.Tensor):
                node_offset = batch.node_offsets[i].item()
            else:
                node_offset = mask.nonzero()[0].item() if mask.any() else 0
            
            target_idx_global = batch.y[i].item()
            target_idx_local = target_idx_global

            if target_idx_local < 0 or target_idx_local >= len(graph_scores): continue
            
            graph_applicable = batch.applicable_mask[mask]
            
            # Curriculum Weighting (using restored meta_list)
            if hasattr(batch, 'meta_list'):
                meta = batch.meta_list[i]
                loss_weight = curriculum_scheduler.get_loss_weight(
                    epoch=epoch,
                    sample_difficulty=meta['difficulty'],
                    step_idx=meta['step_idx'],
                    proof_length=meta.get('proof_length', 10)
                )
            else:
                loss_weight = 1.0 # Fallback

            if loss_weight == 0.0: continue
            current_margin = get_dynamic_margin(epoch, curriculum_scheduler.total_epochs)
    
            # Update criterion margin dynamically
            if hasattr(criterion, 'margin'):
                criterion.margin = current_margin
            # Loss
            try:
                rank_loss = criterion(
                    graph_scores, graph_embeddings, target_idx_local, applicable_mask=graph_applicable
                )
            except Exception as e:
                continue
            
            if torch.isnan(rank_loss) or torch.isinf(rank_loss): continue
            
            # Value loss
            if hasattr(batch, 'value_target'):
                graph_value = value[i:i+1]
                target_value = batch.value_target[i:i+1]
                value_loss = F.mse_loss(graph_value, target_value)
            else:
                value_loss = torch.tensor(0.0, device=device)
            
            combined_loss = (rank_loss + 
                value_loss_weight * value_loss ) * loss_weight  # Weight the reg term
            
            batch_loss = batch_loss + combined_loss
            batch_rank_loss += rank_loss.item()
            batch_value_loss += value_loss.item()
            
            hit1 = compute_hit_at_k(graph_scores, target_idx_local, 1, graph_applicable)
            app_acc = compute_applicable_accuracy(graph_scores, target_idx_local, graph_applicable)
            
            batch_acc += hit1
            batch_applicable_acc += app_acc
            graphs_processed_in_batch += 1
        
        if graphs_processed_in_batch > 0:
            avg_batch_loss = batch_loss / graphs_processed_in_batch
            normalized_loss = avg_batch_loss / grad_accum_steps

            if hasattr(model.fusion, 'last_diversity_loss'):
                normalized_loss += 0.01 * model.fusion.last_diversity_loss
            if torch.isnan(normalized_loss) or torch.isinf(normalized_loss):
                logger.warning(f"ÃƒÂ¢Ã…Â¡ ÃƒÂ¯Ã‚Â¸Ã‚  NaN/Inf loss detected at epoch {epoch}, batch {batch_idx}. Skipping update.")
                main_optimizer.zero_grad() # Clear gradients
                continue # Skip this batch
            normalized_loss.backward() 

            # Accumulate stats
            total_rank_loss += batch_rank_loss / graphs_processed_in_batch
            total_value_loss += batch_value_loss / graphs_processed_in_batch
            total_accuracy += batch_acc / graphs_processed_in_batch # FIX: Average per batch
            total_applicable_acc += batch_applicable_acc / graphs_processed_in_batch # FIX: Average per batch
            num_samples += graphs_processed_in_batch
            num_batches_processed += 1

            if not has_checked_this_epoch:
                debugger.check_gradients()
                debugger.remove_hooks()
                has_checked_this_epoch = True
                print(f"{'='*60}\n")

            if (batch_idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                main_optimizer.step()
                main_optimizer.zero_grad()
                if scheduler: scheduler.step()
                
                # Step Temperature (Unclipped or high clip)
        #         temp_optimizer.step()
        #         temp_optimizer.zero_grad()

        # if temp_scheduler: temp_scheduler.step()



        if num_batches_processed > 0:
            progress_bar.set_postfix({
                'rank_loss': total_rank_loss / num_batches_processed,
                'hit@1': total_accuracy / num_batches_processed
            })
    
    # Safe division for epoch stats
    denom = max(num_batches_processed, 1)
    return {
        'rank_loss': total_rank_loss / denom,
        'value_loss': total_value_loss / denom,
        'hit@1': total_accuracy / denom,
        'applicable_acc': total_applicable_acc / denom,
        'num_samples': num_samples
    }
# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train with Critical Fixes Applied")
    parser.add_argument('--data-dir', type=str, default='generated_data')
    parser.add_argument('--spectral-dir', type=str, default='spectral_cache')
    parser.add_argument('--exp-dir', type=str, default='experiments/critical_fixes')
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--k-dim', type=int, default=32)
    parser.add_argument('--margin', type=float, default=2.0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--grad-accum-steps', type=int, default=4)
    parser.add_argument('--value-loss-weight', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if args.device == 'cuda' and torch.cuda.is_available(): device = torch.device('cuda')
    elif args.device == 'mps' and torch.backends.mps.is_available(): device = torch.device('mps')
    else: device = torch.device('cpu')
    
    logger.info(f"Using device: {device}")
    
    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    in_dim = 32
    logger.info("Loading data with proper instance-level split...")
    train_loader_base, val_loader_base, test_loader_base = create_properly_split_dataloaders(
        args.data_dir, spectral_dir=args.spectral_dir, train_ratio=0.7, val_ratio=0.15,
        batch_size=args.batch_size, seed=args.seed, num_workers=2, pin_memory=True,
        k=args.k_dim  # <--- CRITICAL FIX
    )
    
    train_dataset = train_loader_base.dataset
    val_dataset = val_loader_base.dataset
    validate_spectral_loading(train_dataset)
    curriculum_scheduler = PhasedCurriculumScheduler(total_epochs=args.epochs)
    logger.info("Configuring Dual Optimizers (Main: AdamW, Temp: SGD)...")
    logger.info("Initializing SOTAfixedGNN...")
    model = get_fixed_model(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        k=args.k_dim,
        device=device
    )
    try:
        # Check the weight of the first linear layer in the first gate
        grad_status = model.fusion.out_proj[0][0].weight.requires_grad
        print(f"\nÃƒÂ°Ã…Â¸Ã¢â‚¬ Ã‚  DIAGNOSTIC: Fusion Gate Gradients Enabled? {grad_status}")
        if not grad_status:
            print("ÃƒÂ¢Ã…Â¡ ÃƒÂ¯Ã‚Â¸Ã‚  CRITICAL WARNING: Fusion layer is frozen!")
    except Exception as e:
        print(f"ÃƒÂ¢Ã…Â¡ ÃƒÂ¯Ã‚Â¸Ã‚  Could not cheFck fusion grads: {e}")
    
    temp_param_id = id(model.scorer.log_temperature)
    # Filter Parameters
    base_params = [p for p in model.parameters() if id(p) != temp_param_id]
    # temp_params = [model.scorer.log_temperature]

    # Main Optimizer (AdamW) for Model
    main_optimizer = AdamW(
        base_params, 
        lr=args.lr, 
        weight_decay=1e-4
    )

    # Temperature Optimizer (SGD) - High LR, Momentum
    # temp_optimizer = torch.optim.SGD(
    #     temp_params,
    #     lr=0.1,         # Base LR 0.1 (High!)
    #     momentum=0.9    # Momentum for stability
    # )
    # # 3. INITIALIZE OPTIMIZER
    # optimizer = AdamW([
    #     {
    #         'params': temp_group, 
    #         'lr': 5e-2,   # 500x Boost (Critical for scalar learning)
    #         'weight_decay': 0.0
    #     },
    #     {
    #         'params': base_group, 
    #         'lr': args.lr, 
    #         'weight_decay': 1e-4
    #     }
    # ], lr=args.lr)

    logger.info("Using HyperbolicProofLoss (Margin-based Applicability)")
    # optimizer = AdamW([
    #     {
    #         'params': model.scorer.log_temperature, 
    #         'lr': 1e-2,  # 100x higher LR for this single scalar
    #         'weight_decay': 0.0
    #     },
    #     {
    #         'params': [p for n, p in model.named_parameters() if 'log_temperature' not in n],
    #         'lr': args.lr,
    #         'weight_decay': 1e-4
    #     },
    #     {'params': model.spectral_encoder.parameters(), 'lr': args.lr},
    #     # Boost Temporal LR x10 to fix dead neurons
    #     {'params': model.temporal_encoder.parameters(), 'lr': args.lr * 10.0}, 
    #     {'params': model.fusion.parameters(), 'lr': args.lr},
    #     {'params': model.scorer.parameters(), 'lr': args.lr},
    #     {'params': model.spatial_gnn.parameters(), 'lr': args.lr}
    # ], lr=args.lr, weight_decay=1e-4)
    
    steps_per_epoch = max(len(train_loader_base), 1)
    steps_per_epoch = max(len(train_loader_base), 1)
    scheduler = OneCycleLR(
        main_optimizer, 
        max_lr=args.lr * 10,  # Or whatever your max LR config is
        epochs=args.epochs, 
        steps_per_epoch=steps_per_epoch,
        pct_start=0.15, 
        div_factor=25.0
    )
    def get_temp_lr_lambda(epoch):
        if epoch < 20:
            return 1.0  # 0.1 * 1.0 = 0.1
        elif epoch < 40:
            return 0.5  # 0.1 * 0.5 = 0.05
        else:
            return 0.1  # 0.1 * 0.1 = 0.01

    # temp_scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     temp_optimizer, 
    #     lr_lambda=get_temp_lr_lambda
    # )

    best_val_hit1 = 0.0
    best_global_hit1 = 0.0
    best_epoch = 0
    patience_counter = 0
    patience_limit = 50
    
    logger.info("Starting training...\n")
    start_epoch = 1
    
    # --- RESUME LOGIC ---
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"Loading checkpoint from '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            
            # 1. Load Model Weights
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 2. Update Start Epoch (Optional: keeps logs consistent)
            # If we are cooling down, we might want to just run for 10 more epochs 
            # relative to where we stopped, or just stick to the new --epochs count.
            # start_epoch = checkpoint['epoch'] + 1 
            
            logger.info(f"Successfully loaded model from epoch {checkpoint.get('epoch', 'Unknown')}")
            logger.info("Optimizer and Scheduler reset for Fine-Tuning (Cool Down phase).")
            
        else:
            logger.warning(f"No checkpoint found at '{args.resume}'")
    for epoch in range(start_epoch, start_epoch + args.epochs):
        current_gamma = min(2.0, (epoch / 20.0) * 2.0) 
        current_phase = curriculum_scheduler.get_phase_config(epoch)['phase']
        prev_phase = curriculum_scheduler.get_phase_config(epoch - 1)['phase'] if epoch > 1 else current_phase

        # 2. Reset Temperature Logic
        if current_phase != prev_phase:
            logger.info(f"🔄 PHASE CHANGE DETECTED ({prev_phase} -> {current_phase}): Resetting Temperature!")
            # Reset the internal step of your temperature scheduler/optimizer
            # If you are using a manual decay:
            temperature = 1.0 
            scheduler = None
            # If you are using a scheduler (like OneCycleLR equivalent for temp):
            # temp_scheduler.last_epoch = -1 
            new_lr = 1e-4 
            for param_group in main_optimizer.param_groups:
                param_group['lr'] = new_lr
                
            logger.info(f"   -> Scheduler disabled. Constant LR set to {new_lr}")
        # 3. Modify the decay factor to be Phase-Relative
        # Instead of decaying based on global epoch (0 to 50), decay based on Phase Progress.
        phase_len = 15 # approx length of a phase
        phase_progress = (epoch % phase_len) / phase_len
        temperature = 1.0 - (0.7 * phase_progress) # Decay from 1.0 -> 0.3 within each phase
        temperature = max(0.1, temperature)
        if current_phase != prev_phase:
            logger.info(f"🔄 PHASE CHANGE ({prev_phase} -> {current_phase}): Resetting LR and Temp!")
            
            # 1. Reset Temperature (As discussed before)
            temperature = 1.0 
            
            # 2. Reset Learning Rate (The new fix)
            # We want to bump LR back up to ~50% of max to allow learning new structures
            new_lr = 1e-4 # Or whatever your base_lr is
            for param_group in main_optimizer.param_groups:
                param_group['lr'] = new_lr
                
            logger.info(f"   -> Learning Rate reset to {new_lr}")

        
        criterion = ProductionInfoNCELoss()
        logger.info(f"\n{'='*80}")
        logger.info(curriculum_scheduler.get_epoch_stats(epoch))
        logger.info(f"{'='*80}")
        
        with torch.no_grad():
            model.scorer.log_temperature.fill_(np.log(temperature)) # Use the Phase-Relative temp
        logger.info(f" Forced Temperature: {temperature:.4f} (Scaler: {1.0/temperature:.2f}x)")
        
        # Curriculum filtered loaders
        curr_train_loader = get_curriculum_loader(
            train_dataset, curriculum_scheduler, epoch, args.batch_size, is_train=True
        )
        
        curr_val_loader = get_curriculum_loader(
            val_dataset, curriculum_scheduler, epoch, args.batch_size, is_train=False
        )
        current_temp = max(0.05, 0.1 * (0.95 ** (epoch // 5)))
        if hasattr(criterion, 'temperature'):
            criterion.temperature = temperature
        
        if epoch > 30:
            # Assuming your loss accepts alpha_app updates
            if hasattr(criterion, 'alpha_app'):
                criterion.alpha_app = 2.0  # Double the penalty for illegal moves
        train_metrics = train_epoch_with_curriculum(
            model=model, 
            train_loader=curr_train_loader, 
            main_optimizer=main_optimizer, 
            # temp_optimizer=temp_optimizer, 
            criterion=criterion, 
            device=device, 
            epoch=epoch, 
            curriculum_scheduler=curriculum_scheduler,  # <--- PASS THIS
            scheduler=scheduler,                        # Pass main LR scheduler
            # temp_scheduler=temp_scheduler,              # Pass temp LR scheduler
            grad_accum_steps=args.grad_accum_steps, 
            value_loss_weight=args.value_loss_weight
        )
        
        val_metrics = evaluate(
            model, curr_val_loader, criterion, device, 'val',
            scheduler=curriculum_scheduler, epoch=epoch
        )
        
        logger.info(f"\nTraining Metrics:")
        logger.info(f"  Rank Loss: {train_metrics['rank_loss']:.4f}")
        logger.info(f"  Value Loss: {train_metrics['value_loss']:.4f}")
        logger.info(f"  Hit@1: {train_metrics['hit@1']:.4f}")
        logger.info(f"  Applicable Acc: {train_metrics['applicable_acc']:.4f}")
        
        logger.info(f"\nValidation Metrics:")
        logger.info(f"  Loss: {val_metrics['val_loss']:.4f}")
        logger.info(f"  Hit@1: {val_metrics['val_hit@1']:.4f}")
        logger.info(f"  Hit@3: {val_metrics['val_hit@3']:.4f}")
        
        if val_metrics['val_hit@1'] > best_val_hit1:
            best_val_hit1 = val_metrics['val_hit@1']
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': main_optimizer.state_dict(), 'val_metrics': val_metrics
            }, exp_dir / 'best_model.pt')
            logger.info(f"\nÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â°ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¸ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â½ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¯ NEW BEST Hit@1: {best_val_hit1:.4f}")
        else:
            patience_counter += 1
            logger.info(f"\nÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã¢â‚¬Å¡ ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â³ Patience: {patience_counter}/{patience_limit}")

        if epoch % 5 == 0:
            logger.info(f"ÃƒÂ¢Ã…Â¡Ã‚Â¡ Epoch {epoch}: FULL GLOBAL VALIDATION")
            # Use val_loader_base (Full Dataset)
            metrics = evaluate(model, val_loader_base, criterion, device, ignore_curriculum=True)
            
            hit1 = metrics['val_hit@1']
            logger.info(f"   Global Hit@1: {hit1:.4f}")
            
            if hit1 > best_global_hit1:
                best_global_hit1 = hit1
                best_global_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'metrics': metrics
                }, exp_dir / 'best_model_global.pt')
                logger.info(f"   ÃƒÂ°Ã…Â¸Ã‚ Ã¢â‚¬  NEW SOTA MODEL SAVED (Hit@1: {hit1:.4f})")

        
        if patience_counter >= patience_limit:
            logger.info(f"\nÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â°ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¸ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚ÂºÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¹Ã…â€œ Early stopping at epoch {epoch}")
            break

    logger.info(f"\nLoading BEST GLOBAL MODEL from Epoch {best_global_epoch}...")
    checkpoint = torch.load(exp_dir / 'best_model_global.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader_base, criterion, device, 'test', ignore_curriculum=True)
    logger.info(f"FINAL SOTA RESULT: Test Hit@1 = {test_metrics['test_hit@1']:.4f}")

    logger.info(f"  MRR: {test_metrics['test_mrr']:.4f}")

if __name__ == '__main__':
    main()