"""
SOTA Training Script with All Fixes Applied
==========================================

Critical improvements:
1. Uses AdaptiveFocalRankingLoss (most stable for gradient explosion)
2. Uses AdamW with warmup (prevents early explosion)
3. Lower base LR: 5e-5 instead of 1e-4
4. Gradient centralization
5. Better beta values for AdamW
6. Layer-wise LR decay
7. Curriculum learning
8. Mixed precision training (optional)

Expected improvements:
- No more gradient explosion
- +10-15% Hit@1 from better optimization
- Faster convergence from warmup
- Better generalization from LLRD
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

# Your existing imports
from dataset import create_properly_split_dataloaders
from model import CriticallyFixedProofGNN
from curriculum import SetToSetCurriculumScheduler

# New imports
from new_losses import get_recommended_loss
from optimizer_config import (
    get_recommended_optimizer,
    apply_gradient_techniques
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_metrics(scores, target_idx, applicable_mask):
    """Compute evaluation metrics"""
    
    if target_idx < 0 or target_idx >= len(scores):
        return {'hit@1': 0.0, 'hit@3': 0.0, 'hit@5': 0.0, 'mrr': 0.0}
    
    # Hit@K
    metrics = {}
    for k in [1, 3, 5, 10]:
        top_k = min(k, len(scores))
        top_k_indices = torch.topk(scores, top_k).indices
        metrics[f'hit@{k}'] = 1.0 if target_idx in top_k_indices else 0.0
    
    # MRR
    sorted_indices = torch.argsort(scores, descending=True)
    rank = (sorted_indices == target_idx).nonzero(as_tuple=True)[0]
    if len(rank) > 0:
        metrics['mrr'] = 1.0 / (rank[0].item() + 1)
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


def train_epoch(model, train_loader, optimizer, criterion, device, epoch,
                curriculum_scheduler, opt_config, grad_accum_steps=4,
                value_loss_weight=0.1, use_amp=False):
    """
    Training loop with all SOTA techniques applied.
    """
    
    model.train()
    
    # Get curriculum config
    config = curriculum_scheduler.get_phase_config(epoch)
    logger.info(f"Curriculum: {config['description']}")
    
    # Initialize metrics
    metrics = {
        'loss': 0.0,
        'rank_loss': 0.0,
        'value_loss': 0.0,
        'hit@1': 0.0,
        'app_acc': 0.0
    }
    num_samples = 0
    num_batches = 0
    total_norm = 0.0  
    # For mixed precision
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    optimizer.zero_grad()
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        if batch is None:
            continue
        
        batch = batch.to(device)
        batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
        
        # Mixed precision context
        if use_amp:
            autocast_ctx = torch.cuda.amp.autocast()
        else:
            autocast_ctx = torch.enable_grad()
        
        with autocast_ctx:
            # Forward pass
            scores, embeddings, value = model(batch)
            
            # Process each graph
            batch_loss = 0.0
            graphs_processed = 0
            
            for i in range(batch_size):
                # Get per-graph data
                mask = (batch.batch == i) if hasattr(batch, 'batch') else torch.ones(len(scores), dtype=torch.bool)
                graph_scores = scores[mask]
                graph_embeddings = embeddings[mask]
                
                if len(graph_scores) == 0:
                    continue
                
                # Get target
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
                
                # Get curriculum weight
                sample_difficulty_val = batch.difficulty[i].item()
                if sample_difficulty_val < 0.3: sample_diff_str = 'easy'
                elif sample_difficulty_val < 0.6: sample_diff_str = 'medium'
                elif sample_difficulty_val < 0.8: sample_diff_str = 'hard'
                else: sample_diff_str = 'very_hard'
                
                loss_weight = curriculum_scheduler.get_loss_weight(
                    epoch=epoch,
                    sample_difficulty=sample_diff_str,
                    step_idx=batch.meta_list[i]['step_idx'],
                    proof_length=batch.meta_list[i].get('proof_length', 10)
                )
                
                if loss_weight == 0.0:
                    continue
                
                # Compute loss
                try:
                    rank_loss = criterion(
                        graph_scores,
                        graph_embeddings,
                        target_idx_local,
                        applicable_mask=graph_applicable
                    )
                except Exception as e:
                    logger.warning(f"Loss computation failed: {e}")
                    continue
                
                if torch.isnan(rank_loss) or torch.isinf(rank_loss):
                    logger.warning(f"Invalid loss detected, skipping batch")
                    continue
                
                # Value loss
                graph_value = value[i:i+1]
                target_value = batch.value_target[i:i+1]
                value_loss = F.mse_loss(graph_value, target_value)
                
                # Combined loss with curriculum weight
                combined_loss = (rank_loss + value_loss_weight * value_loss) * loss_weight
                batch_loss += combined_loss
                
                # Metrics
                with torch.no_grad():
                    batch_metrics = compute_metrics(graph_scores, target_idx_local, graph_applicable)
                    metrics['hit@1'] += batch_metrics['hit@1']
                    metrics['app_acc'] += batch_metrics['app_acc']
                    metrics['rank_loss'] += rank_loss.item()
                    metrics['value_loss'] += value_loss.item()
                
                graphs_processed += 1
            
            if graphs_processed == 0:
                continue
            
            # Average loss for this batch
            avg_batch_loss = batch_loss / graphs_processed
            normalized_loss = avg_batch_loss / grad_accum_steps
        
        # Backward pass
        if use_amp:
            scaler.scale(normalized_loss).backward()
        else:
            normalized_loss.backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1 == len(train_loader)):
            
            # Check gradient norm BEFORE clipping
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            # If gradient explosion, log and skip
            if total_norm > 100.0:
                logger.warning(f"Gradient explosion: norm={total_norm:.2f}, skipping update")
                optimizer.zero_grad()
                if use_amp:
                    scaler.update()
                continue
            
            # Apply gradient techniques
            apply_gradient_techniques(optimizer, opt_config)
            
            # Gradient clipping
            if use_amp:
                scaler.unscale_(optimizer)
            
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                opt_config['gradient_clip_value']
            )
            
            # Optimizer step
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad()
        
        # Update progress
        num_samples += graphs_processed
        num_batches += 1
        metrics['loss'] += avg_batch_loss.item()
        
        if num_samples > 0:
            progress_bar.set_postfix({
                'loss': metrics['loss'] / num_batches,
                'hit@1': metrics['hit@1'] / num_samples,
                'grad_norm': total_norm
            })
    
    # Epoch averages
    for key in metrics:
        if key in ['hit@1', 'app_acc']:
            metrics[key] = metrics[key] / max(num_samples, 1)
        else:
            metrics[key] = metrics[key] / max(num_batches, 1)
    
    return metrics


@torch.no_grad()
def evaluate(model, val_loader, criterion, device, split_name='val'):
    """Evaluation loop"""
    
    model.eval()
    
    metrics = {
        'loss': 0.0,
        'hit@1': 0.0,
        'hit@3': 0.0,
        'hit@5': 0.0,
        'hit@10': 0.0,
        'mrr': 0.0,
        'app_acc': 0.0
    }
    num_samples = 0
    
    for batch in tqdm(val_loader, desc=f"Eval {split_name}"):
        if batch is None:
            continue
        
        batch = batch.to(device)
        scores, embeddings, value = model(batch)
        
        batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
        
        for i in range(batch_size):
            mask = (batch.batch == i) if hasattr(batch, 'batch') else torch.ones(len(scores), dtype=torch.bool)
            graph_scores = scores[mask]
            graph_embeddings = embeddings[mask]
            
            if len(graph_scores) == 0:
                continue
            
            target_idx_global = batch.y[i].item()
            if hasattr(batch, 'node_offsets') and isinstance(batch.node_offsets, torch.Tensor):
                node_offset = batch.node_offsets[i].item()
            else:
                node_offset = mask.nonzero()[0].item() if mask.any() else 0
            
            target_idx_local = target_idx_global - node_offset
            
            if target_idx_local < 0 or target_idx_local >= len(graph_scores):
                continue
            
            graph_applicable = batch.applicable_mask[mask]
            
            # Loss
            try:
                loss = criterion(graph_scores, graph_embeddings, target_idx_local, graph_applicable)
                if not torch.isnan(loss) and not torch.isinf(loss):
                    metrics['loss'] += loss.item()
            except:
                pass
            
            # Metrics
            batch_metrics = compute_metrics(graph_scores, target_idx_local, graph_applicable)
            for key in ['hit@1', 'hit@3', 'hit@5', 'hit@10', 'mrr', 'app_acc']:
                if key in batch_metrics:
                    metrics[key] += batch_metrics[key]
            
            num_samples += 1
    
    # Average
    for key in metrics:
        metrics[key] = metrics[key] / max(num_samples, 1)
    
    # Add prefix
    return {f'{split_name}_{k}': v for k, v in metrics.items()}


def main():
    parser = argparse.ArgumentParser(description="SOTA Training with All Fixes")
    
    # Data
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--spectral-dir', type=str, default=None)
    parser.add_argument('--exp-dir', type=str, default='experiments/sota')
    
    # Model
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--k-dim', type=int, default=16)
    
    # Loss (NEW)
    parser.add_argument('--loss-type', type=str, default='focal',
                       choices=['focal', 'circle', 'arcface', 'supcon', 'semantic_focal'],
                       help='Loss function (focal recommended for stability)')
    
    # Optimizer (NEW)
    parser.add_argument('--optimizer-type', type=str, default='adamw_warmup',
                       choices=['adamw_warmup', 'sam', 'sgd_momentum'],
                       help='Optimizer (adamw_warmup recommended)')
    parser.add_argument('--base-lr', type=float, default=5e-5,
                       help='Base learning rate (LOWERED from 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--use-llrd', action='store_true', default=True,
                       help='Use layer-wise LR decay')
    parser.add_argument('--llrd-decay', type=float, default=0.9)
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--grad-accum-steps', type=int, default=4)
    parser.add_argument('--value-loss-weight', type=float, default=0.1)
    parser.add_argument('--use-amp', action='store_true',
                       help='Use automatic mixed precision (faster, requires CUDA)')
    
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
        if args.use_amp:
            logger.warning("AMP requires CUDA, disabling")
            args.use_amp = False
    
    logger.info(f"Using device: {device}")
    
    # Create exp dir
    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    logger.info("="*80)
    logger.info("SOTA TRAINING WITH ALL FIXES")
    logger.info("="*80)
    logger.info("Key improvements:")
    logger.info(f"  ✅ Loss: {args.loss_type} (numerically stable)")
    logger.info(f"  ✅ Optimizer: {args.optimizer_type}")
    logger.info(f"  ✅ Base LR: {args.base_lr:.2e} (LOWER than before)")
    logger.info(f"  ✅ Warmup: {args.warmup_epochs} epochs")
    logger.info(f"  ✅ Layer-wise LR decay: {args.use_llrd}")
    logger.info(f"  ✅ Gradient centralization: Enabled")
    logger.info(f"  ✅ Mixed precision: {args.use_amp}")
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
    curriculum_scheduler = SetToSetCurriculumScheduler(
        total_epochs=args.epochs,
        dataset_metadata=train_dataset.instance_metadata if hasattr(train_dataset, 'instance_metadata') else {}
    )
    
    # Initialize model
    logger.info("Initializing model...")
    in_dim = 25  # Your feature dim
    model = CriticallyFixedProofGNN(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        k=args.k_dim
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"✅ Model: {num_params:,} parameters\n")
    
    # Initialize SOTA loss
    logger.info("Initializing loss...")
    if args.loss_type == 'semantic_focal':
        criterion = get_recommended_loss(
            'semantic_focal',
            alpha=0.25,
            gamma=2.0,
            margin=0.5,
            applicability_weight=2.0,
            semantic_weight=0.5
        )
    elif args.loss_type == 'focal':
        criterion = get_recommended_loss(
            'focal',
            alpha=0.25,
            gamma=2.0,
            margin=0.5,
            applicability_weight=2.0
        )
    elif args.loss_type == 'circle':
        criterion = get_recommended_loss(
            'circle',
            m=0.25,
            gamma=256.0,
            applicability_weight=2.0
        )
    elif args.loss_type == 'arcface':
        criterion = get_recommended_loss(
            'arcface',
            s=64.0,
            m=0.5,
            applicability_weight=2.0
        )
    else:  # supcon
        criterion = get_recommended_loss(
            'supcon',
            temperature=0.07,
            applicability_weight=1.5
        )
    
    print()  # New line after loss info
    
    # Initialize SOTA optimizer
    logger.info("Initializing optimizer...")
    optimizer, scheduler, opt_config = get_recommended_optimizer(
        model,
        optimizer_type=args.optimizer_type,
        base_lr=args.base_lr,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        use_llrd=args.use_llrd,
        llrd_decay=args.llrd_decay
    )
    
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
        logger.info(f"{'='*80}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            curriculum_scheduler, opt_config, args.grad_accum_steps,
            args.value_loss_weight, args.use_amp
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device, 'val')
        
        # Log
        logger.info(f"\nTrain: loss={train_metrics['loss']:.4f}, hit@1={train_metrics['hit@1']:.4f}")
        logger.info(f"Val: loss={val_metrics['val_loss']:.4f}, hit@1={val_metrics['val_hit@1']:.4f}, "
                   f"hit@5={val_metrics['val_hit@5']:.4f}, mrr={val_metrics['val_mrr']:.4f}")
        
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
            logger.info(f"\n🛑 Early stopping")
            break
        
        scheduler.step()
    
    # Test
    logger.info(f"\n{'='*80}")
    logger.info("FINAL EVALUATION")
    logger.info(f"{'='*80}")
    
    checkpoint = torch.load(exp_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, criterion, device, 'test')
    
    logger.info(f"\nTest Results (Best epoch: {best_epoch}):")
    logger.info(f"  Hit@1: {test_metrics['test_hit@1']:.4f} ({test_metrics['test_hit@1']*100:.1f}%)")
    logger.info(f"  Hit@3: {test_metrics['test_hit@3']:.4f}")
    logger.info(f"  Hit@5: {test_metrics['test_hit@5']:.4f}")
    logger.info(f"  Hit@10: {test_metrics['test_hit@10']:.4f}")
    logger.info(f"  MRR: {test_metrics['test_mrr']:.4f}")
    logger.info(f"  App Acc: {test_metrics['test_app_acc']:.4f}")
    
    # Save results
    results = {
        'config': vars(args),
        'best_epoch': best_epoch,
        'best_val_hit@1': best_val_hit1,
        'test_metrics': test_metrics
    }
    
    with open(exp_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✅ Training complete! Results saved to {exp_dir}")


if __name__ == '__main__':
    main()