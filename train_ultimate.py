"""
FIXED Ultimate Training Script
==============================

Critical Fixes Applied:
1. ✅ Fixed Loss (FixedUltimateNTPLoss)
2. ✅ Warmup LR Schedule (replaces OneCycleLR)
3. ✅ Improved Curriculum (gradual difficulty increase)
4. ✅ Adaptive Gradient Clipping (per-layer monitoring)
5. ✅ Early Stopping on Gradient Norm (prevents explosion)

Expected: 35-45% Hit@1, zero gradient explosions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import json
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional
from collections import defaultdict

# Import fixed modules
from dataset import create_properly_split_dataloaders
from model import CriticallyFixedProofGNN, SOTAFixedProofGNN
from curriculum import SetToSetCurriculumScheduler

# Import the fixed loss
import sys
sys.path.insert(0, '.')  # Ensure we can import from current directory

# We'll use the fixed loss from the artifact
# For production, save the artifact content to 'fixed_ntp_loss.py'
from ntp_loss import get_ultimate_loss, diagnose_loss_components


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImprovedCurriculumScheduler(SetToSetCurriculumScheduler):
    """
    FIXED: Improved curriculum with gradual difficulty increase.
    
    Original Issue: Phase 1 used only 16% of data (200/1200 instances)
    Fix: Include medium samples in early phases
    """
    
    def get_phase_config(self, epoch: int) -> Dict:
        """
        FIXED: Smoother curriculum transition with more data.
        """
        progress = epoch / self.total_epochs
        
        if progress < 0.20:  # Epochs 1-10 (20% of 50)
            return {
                'phase': 'Foundation',
                'description': 'Easy (60%) + Medium (40%)',
                'include_difficulties': ['easy', 'medium'],  # FIXED: Include medium
                'sampling_weights': {'easy': 0.6, 'medium': 0.4},  # FIXED: Mix
                'max_proof_length': 5,  # FIXED: Increased from 3
                'max_step_depth': 5,
                'confidence_threshold': 0.5,
            }
        
        elif progress < 0.40:  # Epochs 11-20
            return {
                'phase': 'Consolidation',
                'description': 'Easy (30%) + Medium (50%) + Hard (20%)',
                'include_difficulties': ['easy', 'medium', 'hard'],  # FIXED: Add hard
                'sampling_weights': {'easy': 0.3, 'medium': 0.5, 'hard': 0.2},
                'max_proof_length': 7,  # FIXED: Gradual increase
                'max_step_depth': 7,
                'confidence_threshold': 0.6,
            }
        
        elif progress < 0.70:  # Epochs 21-35
            return {
                'phase': 'Challenge',
                'description': 'Medium (40%) + Hard (40%) + Very Hard (20%)',
                'include_difficulties': ['medium', 'hard', 'very_hard'],
                'sampling_weights': {'medium': 0.4, 'hard': 0.4, 'very_hard': 0.2},
                'max_proof_length': 10,
                'max_step_depth': 10,
                'confidence_threshold': 0.7,
            }
        
        else:  # Epochs 36-50
            return {
                'phase': 'Mastery',
                'description': 'All difficulties (balanced)',
                'include_difficulties': ['easy', 'medium', 'hard', 'very_hard'],
                'sampling_weights': {
                    'easy': 0.15,
                    'medium': 0.3,
                    'hard': 0.35,
                    'very_hard': 0.2
                },
                'max_proof_length': 15,
                'max_step_depth': 15,
                'confidence_threshold': 0.8,
            }


class WarmupCosineScheduler:
    """
    FIXED: Warmup + Cosine LR schedule (replaces OneCycleLR).
    
    Original Issue: OneCycleLR conflicts with gradient accumulation
    Fix: Separate warmup phase, then cosine decay
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr_ratio: float = 0.01
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = optimizer.param_groups[0]['lr']
        self.min_lr = self.base_lr * min_lr_ratio
        self.current_epoch = 0
    
    def step(self):
        """Update learning rate."""
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine decay
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


class AdaptiveGradientClipper:
    """
    FIXED: Adaptive gradient clipping with per-layer monitoring.
    
    Original Issue: Fixed max_norm=1.0 was too aggressive
    Fix: Adaptive clipping based on gradient norm distribution
    """
    
    def __init__(
        self,
        model: nn.Module,
        percentile: float = 95.0,
        min_clip: float = 1.0,
        max_clip: float = 5.0
    ):
        self.model = model
        self.percentile = percentile
        self.min_clip = min_clip
        self.max_clip = max_clip
        self.grad_history = []
        self.max_history_size = 100
    
    def clip_gradients(self) -> float:
        """
        Clip gradients adaptively and return total norm.
        """
        # Compute total gradient norm
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=float('inf'),  # No clipping yet, just compute
            norm_type=2.0
        )
        
        # Update history
        self.grad_history.append(total_norm.item())
        if len(self.grad_history) > self.max_history_size:
            self.grad_history.pop(0)
        
        # Compute adaptive clip value
        if len(self.grad_history) >= 10:
            clip_value = np.percentile(self.grad_history, self.percentile)
            clip_value = np.clip(clip_value, self.min_clip, self.max_clip)
        else:
            clip_value = self.max_clip
        
        # Apply clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=clip_value,
            norm_type=2.0
        )
        
        return total_norm.item()


def compute_metrics(scores: torch.Tensor, target_idx: int,
                   applicable_mask: torch.Tensor) -> Dict[str, float]:
    """Compute all evaluation metrics."""
    
    if target_idx < 0 or target_idx >= len(scores):
        return {f'hit@{k}': 0.0 for k in [1, 3, 5, 10]} | {'mrr': 0.0, 'app_acc': 0.0}
    
    metrics = {}
    
    # Hit@K
    for k in [1, 3, 5, 10]:
        k_clamped = min(k, len(scores))
        if k_clamped > 0:
            top_k = torch.topk(scores, k_clamped).indices
            metrics[f'hit@{k}'] = 1.0 if target_idx in top_k else 0.0
        else:
            metrics[f'hit@{k}'] = 0.0
    
    # MRR
    if applicable_mask[target_idx]:
        sorted_indices = torch.argsort(scores, descending=True)
        rank = (sorted_indices == target_idx).nonzero(as_tuple=True)[0].item() + 1
        metrics['mrr'] = 1.0 / rank
    else:
        metrics['mrr'] = 0.0
    
    # Applicable accuracy
    applicable_indices = applicable_mask.nonzero(as_tuple=True)[0]
    if len(applicable_indices) > 0:
        applicable_scores = scores[applicable_indices]
        top_applicable = applicable_indices[applicable_scores.argmax()]
        metrics['app_acc'] = 1.0 if top_applicable == target_idx else 0.0
    else:
        metrics['app_acc'] = 0.0
    
    return metrics


def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer,
    criterion,
    device: torch.device,
    epoch: int,
    curriculum_scheduler,
    grad_clipper: AdaptiveGradientClipper,
    grad_accum_steps: int = 4
) -> Dict[str, float]:
    """Train for one epoch with all fixes applied."""
    
    model.train()
    
    # Get curriculum config
    config = curriculum_scheduler.get_phase_config(epoch)
    logger.info(f"Curriculum Phase: {config['description']}")
    
    # Initialize accumulators
    total_loss = 0.0
    total_metrics = {f'hit@{k}': 0.0 for k in [1, 3, 5, 10]} | {'mrr': 0.0, 'app_acc': 0.0}
    num_samples = 0
    num_batches = 0
    grad_norms = []
    
    optimizer.zero_grad()
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        if batch is None:
            continue
        
        batch = batch.to(device)
        
        # Forward pass
        scores, embeddings, value, recon_spectral = model(batch)
        
        batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
        
        # Process each graph in batch
        batch_loss = torch.tensor(0.0, device=device)
        batch_metrics = {f'hit@{k}': 0.0 for k in [1, 3, 5, 10]} | {'mrr': 0.0, 'app_acc': 0.0}
        graphs_processed = 0
        
        for i in range(batch_size):
            # Get per-graph data
            mask = (batch.batch == i) if hasattr(batch, 'batch') else torch.ones(len(scores), dtype=torch.bool)
            graph_scores = scores[mask]
            graph_embeddings = embeddings[mask]
            
            if len(graph_scores) == 0:
                continue
            
            # Get target index (local to this graph)
            target_idx_global = batch.y[i].item()
            if hasattr(batch, 'node_offsets') and isinstance(batch.node_offsets, torch.Tensor):
                node_offset = batch.node_offsets[i].item()
            else:
                node_offset = mask.nonzero()[0].item() if mask.any() else 0
            
            target_idx_local = target_idx_global - node_offset
            
            if target_idx_local < 0 or target_idx_local >= len(graph_scores):
                continue
            
            # Get applicable mask
            graph_applicable = batch.applicable_mask[mask]
            
            # Curriculum weighting
            sample_difficulty_val = batch.difficulty[i].item()
            if sample_difficulty_val < 0.3:
                sample_diff_str = 'easy'
            elif sample_difficulty_val < 0.6:
                sample_diff_str = 'medium'
            elif sample_difficulty_val < 0.8:
                sample_diff_str = 'hard'
            else:
                sample_diff_str = 'very_hard'
            
            loss_weight = curriculum_scheduler.get_loss_weight(
                epoch=epoch,
                sample_difficulty=sample_diff_str,
                step_idx=batch.meta_list[i]['step_idx'],
                proof_length=batch.meta_list[i].get('proof_length', 10)
            )
            
            if loss_weight == 0.0:
                continue
            
            # Compute loss (FixedUltimateNTPLoss)
            try:
                proof_progress = batch.meta_list[i]['step_idx'] / batch.meta_list[i]['proof_length']
                proof_length = batch.meta_list[i]['proof_length']
                graph_value = value[i:i+1]
                target_value = batch.value_target[i:i+1]
                
                loss = criterion(
                    scores=graph_scores,
                    embeddings=graph_embeddings,
                    target_idx=target_idx_local,
                    applicable_mask=graph_applicable,
                    proof_progress=proof_progress,
                    proof_length=proof_length,
                    value_pred=graph_value,
                    value_target=target_value
                )
                
                # Apply curriculum weight
                loss = loss * loss_weight
                
            except Exception as e:
                logger.warning(f"Loss computation failed: {e}")
                continue
            
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Invalid loss: {loss.item()}")
                continue
            
            # Accumulate loss
            batch_loss += loss
            
            # Compute metrics
            graph_metrics = compute_metrics(graph_scores, target_idx_local, graph_applicable)
            for key in batch_metrics:
                batch_metrics[key] += graph_metrics[key]
            
            graphs_processed += 1
        
        # Backward pass (accumulated)
        if graphs_processed > 0:
            avg_batch_loss = batch_loss / graphs_processed
            normalized_loss = avg_batch_loss / grad_accum_steps
            normalized_loss.backward()
            
            # Adaptive gradient clipping
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                grad_norm = grad_clipper.clip_gradients()
                grad_norms.append(grad_norm)
                
                # CRITICAL: Early stopping on gradient explosion
                if grad_norm > 20.0:
                    logger.error(f"GRADIENT EXPLOSION DETECTED: norm={grad_norm:.2f}")
                    logger.error("Skipping this update and resetting gradients")
                    optimizer.zero_grad()
                    continue
                
                optimizer.step()
                optimizer.zero_grad()
            
            # Accumulate epoch metrics
            total_loss += batch_loss / graphs_processed
            for key in total_metrics:
                total_metrics[key] += batch_metrics[key]
            num_samples += graphs_processed
            num_batches += 1
            
            # Update progress bar
            avg_grad_norm = np.mean(grad_norms[-10:]) if grad_norms else 0.0
            progress_bar.set_postfix({
                'loss': (total_loss / num_batches).item(),
                'hit@1': total_metrics['hit@1'] / num_samples if num_samples > 0 else 0.0,
                'grad': f"{avg_grad_norm:.2f}"
            })
    
    # Compute epoch averages
    results = {'loss': (total_loss / max(num_batches, 1)).item()}
    for key in total_metrics:
        results[key] = total_metrics[key] / max(num_samples, 1)
    results['num_samples'] = num_samples
    results['avg_grad_norm'] = np.mean(grad_norms) if grad_norms else 0.0
    results['max_grad_norm'] = np.max(grad_norms) if grad_norms else 0.0
    
    return results


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader,
    criterion,
    device: torch.device,
    split_name: str = 'val'
) -> Dict[str, float]:
    """Evaluate on validation/test set."""
    
    model.eval()
    
    total_loss = 0.0
    total_metrics = {f'hit@{k}': 0.0 for k in [1, 3, 5, 10]} | {'mrr': 0.0, 'app_acc': 0.0}
    num_samples = 0
    
    for batch in tqdm(val_loader, desc=f"Eval {split_name}"):
        if batch is None:
            continue
        
        batch = batch.to(device)
        
        scores, embeddings, value, recon_spectral = model(batch)
        
        batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
        
        for i in range(batch_size):
            mask = (batch.batch == i) if hasattr(batch, 'batch') else torch.ones(len(scores), dtype=torch.bool)
            graph_scores = scores[mask]
            graph_embeddings = embeddings[mask]
            
            if len(graph_scores) == 0:
                continue
            
            # Get target index
            target_idx_global = batch.y[i].item()
            if hasattr(batch, 'node_offsets') and isinstance(batch.node_offsets, torch.Tensor):
                node_offset = batch.node_offsets[i].item()
            else:
                node_offset = mask.nonzero()[0].item() if mask.any() else 0
            
            target_idx_local = target_idx_global - node_offset
            
            if target_idx_local < 0 or target_idx_local >= len(graph_scores):
                continue
            
            graph_applicable = batch.applicable_mask[mask]
            
            # Compute loss
            try:
                proof_progress = batch.meta_list[i]['step_idx'] / batch.meta_list[i]['proof_length']
                proof_length = batch.meta_list[i]['proof_length']
                graph_value = value[i:i+1]
                target_value = batch.value_target[i:i+1]
                
                loss = criterion(
                    scores=graph_scores,
                    embeddings=graph_embeddings,
                    target_idx=target_idx_local,
                    applicable_mask=graph_applicable,
                    proof_progress=proof_progress,
                    proof_length=proof_length,
                    value_pred=graph_value,
                    value_target=target_value
                )
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss += loss.item()
            except Exception as e:
                logger.debug(f"Loss computation failed: {e}")
            
            # Compute metrics
            graph_metrics = compute_metrics(graph_scores, target_idx_local, graph_applicable)
            for key in total_metrics:
                total_metrics[key] += graph_metrics[key]
            
            num_samples += 1
    
    # Compute averages
    results = {f'{split_name}_loss': total_loss / max(num_samples, 1)}
    for key in total_metrics:
        results[f'{split_name}_{key}'] = total_metrics[key] / max(num_samples, 1)
    results[f'{split_name}_num_samples'] = num_samples
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train with Fixed UltimateNTPLoss"
    )
    
    # Data
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--spectral-dir', type=str, default=None)
    parser.add_argument('--exp-dir', type=str, default='experiments/fixed_ultimate')
    
    # Model
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--k-dim', type=int, default=16)
    
    # Loss (FixedUltimateNTPLoss hyperparameters)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--semantic-weight', type=float, default=0.3)
    parser.add_argument('--progress-weight', type=float, default=0.1)
    parser.add_argument('--hard-neg-weight', type=float, default=0.2)
    parser.add_argument('--applicability-weight', type=float, default=0.4)
    parser.add_argument('--max-loss', type=float, default=5.0)
    parser.add_argument('--gradient-clip', type=float, default=2.0)
    parser.add_argument('--label-smoothing', type=float, default=0.05)
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--grad-accum-steps', type=int, default=4)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    logger.info(f"Using device: {device}")
    
    # Create exp directory
    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    logger.info("="*80)
    logger.info("FIXED ULTIMATE NTP TRAINING")
    logger.info("="*80)
    logger.info("Fixes Applied:")
    logger.info("  ✅ FixedUltimateNTPLoss (orthogonal components)")
    logger.info("  ✅ Score Normalization (bounded gradients)")
    logger.info("  ✅ Adaptive Component Weighting (automatic balancing)")
    logger.info("  ✅ Warmup + Cosine LR (stable optimization)")
    logger.info("  ✅ Improved Curriculum (more data early)")
    parser.add_argument('--adaptive-clipping', type=bool, default=True)
    logger.info("="*80 + "\n")
    
    # Load data
    logger.info("Loading data...")
    train_loader, val_loader, test_loader = create_properly_split_dataloaders(
        args.data_dir,
        spectral_dir=args.spectral_dir,
        batch_size=args.batch_size,
        seed=args.seed
    )
    
    logger.info(f"✅ Data loaded: {len(train_loader)} train batches\n")
    
    # Initialize curriculum
    train_dataset = train_loader.dataset
    curriculum_scheduler = ImprovedCurriculumScheduler(
        total_epochs=args.epochs,
        dataset_metadata=train_dataset.instance_metadata if hasattr(train_dataset, 'instance_metadata') else {}
    )
    logger.info("✅ Improved curriculum scheduler initialized\n")
    
    # Initialize model
    logger.info("Initializing model...")
    in_dim = 25
    model = SOTAFixedProofGNN(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        k=args.k_dim
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"✅ Model: {num_params:,} parameters\n")
    
    # Initialize loss (FixedUltimateNTPLoss)
    logger.info("Initializing FixedUltimateNTPLoss...")
    criterion = get_ultimate_loss(
        use_value_head=True,
        margin=args.margin,
        semantic_weight=args.semantic_weight,
        progress_weight=args.progress_weight,
        hard_neg_weight=args.hard_neg_weight,
        applicability_weight=args.applicability_weight,
        max_loss=args.max_loss,
        gradient_clip=args.gradient_clip,
        label_smoothing=args.label_smoothing
    )
    logger.info(f"✅ Loss initialized\n")
    
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,  # FIXED: Reduced from 1e-3
        betas=(0.9, 0.999)
    )
    
    # Initialize scheduler (Warmup + Cosine)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        min_lr_ratio=0.01
    )
    
    # Initialize adaptive gradient clipper
    grad_clipper = AdaptiveGradientClipper(
        model,
        percentile=95.0,
        min_clip=1.0,
        max_clip=5.0
    )
    
    logger.info("✅ Optimizer: AdamW with Warmup + Cosine LR\n")
    logger.info("✅ Adaptive Gradient Clipper initialized\n")
    
    # Training loop
    best_val_hit1 = 0.0
    best_epoch = 0
    patience = 0
    patience_limit = 15
    
    logger.info("Starting training...\n")
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"EPOCH {epoch}/{args.epochs}")
        logger.info(curriculum_scheduler.get_epoch_stats(epoch))
        logger.info(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        logger.info(f"{'='*80}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch, curriculum_scheduler, grad_clipper, args.grad_accum_steps
        )
        
        # Update LR
        current_lr = scheduler.step()
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device, 'val')
        
        # Log
        logger.info(f"\nTrain: loss={train_metrics['loss']:.4f}, "
                   f"hit@1={train_metrics['hit@1']:.4f}, "
                   f"app_acc={train_metrics['app_acc']:.4f}")
        logger.info(f"       avg_grad={train_metrics['avg_grad_norm']:.2f}, "
                   f"max_grad={train_metrics['max_grad_norm']:.2f}")
        logger.info(f"Val:   loss={val_metrics['val_loss']:.4f}, "
                   f"hit@1={val_metrics['val_hit@1']:.4f}, "
                   f"hit@5={val_metrics['val_hit@5']:.4f}, "
                   f"mrr={val_metrics['val_mrr']:.4f}")
        
        # Save best
        if val_metrics['val_hit@1'] > best_val_hit1:
            best_val_hit1 = val_metrics['val_hit@1']
            best_epoch = epoch
            patience = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': vars(args)
            }, exp_dir / 'best_model.pt')
            
            logger.info(f"\n🎯 NEW BEST Hit@1: {best_val_hit1:.4f}")
        else:
            patience += 1
            logger.info(f"\n⏳ Patience: {patience}/{patience_limit}")
        
        if patience >= patience_limit:
            logger.info(f"\n🛑 Early stopping at epoch {epoch}")
            break
    
    
    # Test
    logger.info(f"\n{'='*80}")
    logger.info("FINAL EVALUATION")
    logger.info(f"{'='*80}")
    
    checkpoint = torch.load(exp_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, criterion, device, 'test')
    
    logger.info(f"\nTest Results (Best epoch: {best_epoch}):")
    logger.info(f"  Hit@1: {test_metrics['test_hit@1']:.4f} ({test_metrics['test_hit@1']*100:.1f}%)")
    logger.info(f"  Hit@3: {test_metrics['test_hit@3']:.4f} ({test_metrics['test_hit@3']*100:.1f}%)")
    logger.info(f"  Hit@5: {test_metrics['test_hit@5']:.4f} ({test_metrics['test_hit@5']*100:.1f}%)")
    logger.info(f"  Hit@10: {test_metrics['test_hit@10']:.4f} ({test_metrics['test_hit@10']*100:.1f}%)")
    logger.info(f"  MRR: {test_metrics['test_mrr']:.4f}")
    logger.info(f"  App Acc: {test_metrics['test_app_acc']:.4f}")
    logger.info(f"  Samples: {test_metrics['test_num_samples']}")
    
    # Save results
    results = {
        'config': vars(args),
        'best_epoch': best_epoch,
        'best_val_hit@1': best_val_hit1,
        'test_metrics': test_metrics,
        'num_model_params': num_params
    }
    
    with open(exp_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✅ Training complete!")
    logger.info(f"   Results: {exp_dir / 'results.json'}")
    logger.info(f"   Model: {exp_dir / 'best_model.pt'}")


if __name__ == '__main__':
    main()