"""
Enhanced Training Script with All SOTA Fixes
=============================================

Changes from original:
1. Import SOTAProofGNN instead of SOTAFixedProofGNN
2. Use InfoNCEWithHardNegativeMining loss
3. Add gradient anomaly detection (debug mode)
4. Better learning rate schedule
5. More robust validation metrics
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import argparse
import logging
from pathlib import Path

# Import enhanced modules
from model_enhanced import SOTAProofGNN
from loss_enhanced import InfoNCEWithHardNegativeMining, MultiTaskProofLoss
from dataset import create_properly_split_dataloaders
from curriculum import SetToSetCurriculumScheduler

# Original imports (keep for metrics, etc.)
from train import (
    compute_hit_at_k, compute_mrr, compute_applicable_accuracy,
    compute_ranking_quality, evaluate
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_epoch_enhanced(model, train_loader, optimizer, criterion,
                        device, epoch, scheduler, curriculum_scheduler,
                        grad_accum_steps=4, value_loss_weight=0.1,
                        debug_mode=False):
    """
    Enhanced training epoch with:
    1. Gradient anomaly detection
    2. Better metric tracking
    3. Curriculum integration
    """
    
    model.train()
    
    # Enable anomaly detection if debug mode
    if debug_mode:
        torch.autograd.set_detect_anomaly(True)
        logger.info("⚠️  Gradient anomaly detection ENABLED (slow)")
    
    # Get curriculum config
    config = curriculum_scheduler.get_phase_config(epoch)
    logger.info(f"Curriculum: {config['description']}")
    
    total_rank_loss = 0.0
    total_value_loss = 0.0
    total_accuracy = 0.0
    total_applicable_acc = 0.0
    num_samples = 0
    num_batches_processed = 0
    
    # For multi-task loss tracking
    total_weight_rank = 0.0
    total_weight_value = 0.0
    
    optimizer.zero_grad()
    
    from tqdm import tqdm
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)
    
    for batch_idx, batch in enumerate(progress_bar):
        if batch is None:
            continue
        
        batch = batch.to(device)
        
        # Curriculum batch filter
        if not curriculum_scheduler.filter_batch(batch, epoch):
            continue
        
        # Forward pass
        scores, embeddings, value, _ = model(batch)
        
        batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else 1

        batch_loss_tensor_sum = torch.tensor(0.0, device=device) # Accumulates tensors for .backward()
        batch_loss_accum = 0.0
        batch_rank_loss_accum = 0.0
        batch_value_loss_accum = 0.0
        graphs_processed = 0
        
        for i in range(batch_size):
            # Get graph mask
            mask = (batch.batch == i) if hasattr(batch, 'batch') else torch.ones(len(scores), dtype=torch.bool)
            
            graph_scores = scores[mask]
            graph_embeddings = embeddings[mask]
            
            if len(graph_scores) == 0:
                continue
            
            # Get target index (local to this graph)
            if hasattr(batch, 'node_offsets'):
                local_y = batch.y[mask] - batch.node_offsets[i]
            else:
                local_y = batch.y[mask]
            target_idx = local_y.item()
            
            # Get masks for this graph
            graph_applicable_mask = batch.applicable_mask[mask]
            
            # Get curriculum weight
            difficulty = batch.difficulties[i]
            step_idx = batch.step_indices[i]
            proof_length = batch.proof_lengths[i]
            w = curriculum_scheduler.get_loss_weight(epoch, difficulty, step_idx, proof_length)
            
            # Compute loss
            loss_dict = criterion(
                scores=graph_scores,
                embeddings=graph_embeddings,
                target_idx=target_idx,
                applicable_mask=graph_applicable_mask,
                value_pred=value[i:i+1] if value is not None else None,
                value_target=torch.tensor([1.0], device=device) if value is not None else None  # Placeholder; adapt as needed
            )
            
            loss = loss_dict['total'] * w
            batch_loss_tensor_sum += loss
            batch_rank_loss_accum += loss_dict['ranking'].item()
            batch_value_loss_accum += loss_dict['value'].item()
            
            if 'weight_rank' in loss_dict:
                total_weight_rank += loss_dict['weight_rank']
                total_weight_value += loss_dict['weight_value']
            
            graphs_processed += 1
        
        # Regularization (Chebyshev coeffs)
        reg_loss = 0.01 * model.spectral_filter.poly_coeffs.norm(2)
        batch_loss_tensor_sum += reg_loss
        
        # Accumulate gradients
        batch_loss_tensor_sum = batch_loss_tensor_sum / grad_accum_steps
        batch_loss_tensor_sum.backward()
        
        if (batch_idx + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        
        # Update progress bar
        if num_samples > 0:
            progress_bar.set_postfix({
                'rank_loss': total_rank_loss / num_batches_processed,
                'val_loss': total_value_loss / num_batches_processed,
                'hit@1': total_accuracy / num_samples,
                'app_acc': total_applicable_acc / num_samples
            })
        num_batches_processed += 1
    # Disable anomaly detection
    if debug_mode:
        torch.autograd.set_detect_anomaly(False)
    
    # Compute epoch averages
    avg_rank_loss = total_rank_loss / max(num_batches_processed, 1)
    avg_value_loss = total_value_loss / max(num_batches_processed, 1)
    avg_accuracy = total_accuracy / max(num_samples, 1)
    avg_applicable_acc = total_applicable_acc / max(num_samples, 1)
    
    results = {
        'rank_loss': avg_rank_loss,
        'value_loss': avg_value_loss,
        'hit@1': avg_accuracy,
        'applicable_acc': avg_applicable_acc,
        'num_samples': num_samples
    }
    
    # Add task weights if using multi-task loss
    if total_weight_rank > 0:
        results['weight_rank'] = total_weight_rank / num_batches_processed
        results['weight_value'] = total_weight_value / num_batches_processed
    
    return results


def main():
    parser = argparse.ArgumentParser(description="SOTA Training with All Fixes")
    
    # Data args
    parser.add_argument('--data-dir', type=str, default='generated_data')
    parser.add_argument('--spectral-dir', type=str, default='spectral_cache')
    parser.add_argument('--exp-dir', type=str, default='experiments/sota_enhanced')
    
    # Model args
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=6)  # Increased for multi-hop
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--k-dim', type=int, default=16)
    
    # Loss args
    parser.add_argument('--loss-type', type=str, default='multitask',
                       choices=['infonce', 'adaptive_infonce', 'multitask'])
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=0.3)
    
    # Training args
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)  # Higher for faster convergence
    parser.add_argument('--grad-accum-steps', type=int, default=4)
    parser.add_argument('--value-loss-weight', type=float, default=0.1)
    
    # Debug
    parser.add_argument('--debug', action='store_true',
                       help='Enable gradient anomaly detection')
    
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("SOTA ENHANCED TRAINING")
    logger.info("="*80)
    logger.info("Enhancements:")
    logger.info("  ✅ Laplacian Positional Encoding (GPS++)")
    logger.info("  ✅ Stable Chebyshev Filters (Clenshaw)")
    logger.info("  ✅ Hard Causal Masking (Transformer-XL)")
    logger.info("  ✅ Multi-Hop Reasoning (6-layer GNN)")
    logger.info("  ✅ InfoNCE Loss with Hard Negatives")
    logger.info("="*80 + "\n")
    
    # Load data
    logger.info("Loading data...")
    train_loader, val_loader, test_loader = create_properly_split_dataloaders(
        args.data_dir,
        spectral_dir=args.spectral_dir,
        batch_size=args.batch_size,
        seed=args.seed
    )
    
    # Initialize curriculum
    train_dataset = train_loader.dataset
    curriculum_scheduler = SetToSetCurriculumScheduler(
        total_epochs=args.epochs,
        dataset_metadata=train_dataset.instance_metadata if hasattr(train_dataset, 'instance_metadata') else {}
    )
    
    # Initialize model
    logger.info("Initializing SOTAProofGNN...")
    in_dim = 29  # Update if features change
    model = SOTAProofGNN(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        k=args.k_dim
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"✅ Model: {num_params:,} parameters\n")
    
    # Initialize loss
    logger.info(f"Initializing {args.loss_type} loss...")
    if args.loss_type == 'multitask':
        criterion = MultiTaskProofLoss(
            temperature=args.temperature,
            margin=args.margin,
            alpha=args.alpha
        ).to(device)
    else:
        from loss_enhanced import get_sota_loss
        criterion = get_sota_loss(
            args.loss_type,
            temperature=args.temperature,
            margin=args.margin,
            alpha=args.alpha
        ).to(device)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # LR Scheduler
    steps_per_epoch = max(len(train_loader), 1)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr * 2,  # Peak at 2x base LR
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,  # 10% warmup
        div_factor=25.0,
        final_div_factor=1000.0
    )
    
    # Training loop
    best_val_hit1 = 0.0
    best_epoch = 0
    patience = 0
    patience_limit = 15
    
    logger.info("Starting training...\n")
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n{'='*80}")
        logger.info(curriculum_scheduler.get_epoch_stats(epoch))
        logger.info(f"{'='*80}")
        
        # Train
        train_metrics = train_epoch_enhanced(
            model, train_loader, optimizer, criterion, device,
            epoch, scheduler, curriculum_scheduler,
            args.grad_accum_steps, args.value_loss_weight,
            debug_mode=args.debug
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device, 'val')
        
        # Log
        logger.info(f"\nTraining:")
        logger.info(f"  Rank Loss: {train_metrics['rank_loss']:.4f}")
        logger.info(f"  Hit@1: {train_metrics['hit@1']:.4f} ({train_metrics['hit@1']*100:.1f}%)")
        logger.info(f"  Applicable Acc: {train_metrics['applicable_acc']:.4f}")
        
        if 'weight_rank' in train_metrics:
            logger.info(f"  Task Weights: rank={train_metrics['weight_rank']:.3f}, "
                       f"value={train_metrics['weight_value']:.3f}")
        
        logger.info(f"\nValidation:")
        logger.info(f"  Hit@1: {val_metrics['val_hit@1']:.4f} ({val_metrics['val_hit@1']*100:.1f}%)")
        logger.info(f"  Hit@5: {val_metrics['val_hit@5']:.4f}")
        logger.info(f"  MRR: {val_metrics['val_mrr']:.4f}")
        
        # Save best
        if val_metrics['val_hit@1'] > best_val_hit1:
            best_val_hit1 = val_metrics['val_hit@1']
            best_epoch = epoch
            patience = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics
            }, exp_dir / 'best_model.pt')
            
            logger.info(f"\n🎯 NEW BEST Hit@1: {best_val_hit1:.4f} ({best_val_hit1*100:.1f}%)")
        else:
            patience += 1
            logger.info(f"\n⏳ Patience: {patience}/{patience_limit}")
        
        # Early stopping
        if patience >= patience_limit:
            logger.info(f"\n🛑 Early stopping at epoch {epoch}")
            break
    
    # Final test evaluation
    logger.info(f"\n{'='*80}")
    logger.info("FINAL TEST EVALUATION")
    logger.info(f"{'='*80}")
    
    checkpoint = torch.load(exp_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, criterion, device, 'test')
    
    logger.info(f"\nTest Results (epoch {best_epoch}):")
    logger.info(f"  Hit@1: {test_metrics['test_hit@1']:.4f} ({test_metrics['test_hit@1']*100:.1f}%)")
    logger.info(f"  Hit@3: {test_metrics['test_hit@3']:.4f} ({test_metrics['test_hit@3']*100:.1f}%)")
    logger.info(f"  Hit@5: {test_metrics['test_hit@5']:.4f} ({test_metrics['test_hit@5']*100:.1f}%)")
    logger.info(f"  MRR: {test_metrics['test_mrr']:.4f}")
    logger.info(f"  Applicable Acc: {test_metrics['test_applicable_acc']:.4f}")
    
    # Save results
    import json
    results = {
        'config': vars(args),
        'best_epoch': best_epoch,
        'best_val_hit@1': float(best_val_hit1),
        'test_metrics': {k: float(v) if isinstance(v, (int, float)) else v 
                        for k, v in test_metrics.items()},
        'num_params': num_params
    }
    
    with open(exp_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✅ Training complete!")
    logger.info(f"   Results: {exp_dir / 'results.json'}")


if __name__ == '__main__':
    main()