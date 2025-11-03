#!/bin/bash
# Quick Fix Script - Apply All Critical Patches
# Run this to fix your pipeline immediately

echo "================================"
echo "🔧 APPLYING CRITICAL FIXES"
echo "================================"

# 1. Add collation function to dataset.py
echo -e "\n📝 Patching dataset.py..."

cat >> dataset.py << 'EOF'

# ========== CRITICAL FIX: Proper Batch Collation ==========
def custom_collate_with_metadata(batch_list):
    """Fixed collation with metadata tracking"""
    batch_list = [b for b in batch_list if b is not None]
    if len(batch_list) == 0:
        return None
    
    batch = Batch.from_data_list(batch_list)
    
    # Track per-graph node counts
    batch.num_nodes_per_graph = torch.tensor(
        [data.x.shape[0] for data in batch_list],
        dtype=torch.long
    )
    
    batch.node_offsets = torch.cat([
        torch.tensor([0]),
        torch.cumsum(batch.num_nodes_per_graph[:-1], dim=0)
    ])
    
    return batch
EOF

echo "✅ dataset.py patched"

# 2. Create minimal training script
echo -e "\n📝 Creating train_minimal.py..."

cat > train_minimal.py << 'TRAINEOF'
"""Minimal Fixed Training Script"""
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

# Import from your files
from dataset import ProofStepDataset, custom_collate_with_metadata
from model import get_model
from losses import get_recommended_loss
from torch_geometric.loader import DataLoader
import json
import random
from collections import defaultdict

def create_loaders(data_dir, spectral_dir, batch_size, seed):
    """Create dataloaders with fixed collation"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    json_files = list(Path(data_dir).glob("**/*.json"))
    instances_by_diff = defaultdict(list)
    
    for f in json_files:
        try:
            with open(f) as fp:
                inst = json.load(fp)
            diff = inst.get('metadata', {}).get('difficulty', 'medium')
            instances_by_diff[diff].append(str(f))
        except:
            continue
    
    train_files, val_files, test_files = [], [], []
    
    for diff, files in instances_by_diff.items():
        random.shuffle(files)
        n = len(files)
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)
        
        train_files.extend(files[:n_train])
        val_files.extend(files[n_train:n_train + n_val])
        test_files.extend(files[n_train + n_val:])
    
    train_ds = ProofStepDataset(train_files, spectral_dir, seed=seed)
    val_ds = ProofStepDataset(val_files, spectral_dir, seed=seed+1)
    test_ds = ProofStepDataset(test_files, spectral_dir, seed=seed+2)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              collate_fn=custom_collate_with_metadata, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                           collate_fn=custom_collate_with_metadata, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=custom_collate_with_metadata, num_workers=0)
    
    return train_loader, val_loader, test_loader

def extract_graph(batch, graph_idx):
    """Extract single graph from batch"""
    if not hasattr(batch, 'num_nodes_per_graph'):
        # Single graph
        return 0, batch.x.shape[0], batch.y[0].item() if hasattr(batch, 'y') else 0
    
    start = batch.node_offsets[graph_idx].item()
    num_nodes = batch.num_nodes_per_graph[graph_idx].item()
    target_global = batch.y[graph_idx].item() if hasattr(batch, 'y') else 0
    target_local = target_global - start
    
    return start, start + num_nodes, target_local

def train_epoch(model, loader, optimizer, criterion, device):
    """Simple training loop"""
    model.train()
    total_loss = 0.0
    total_hit1 = 0.0
    num_samples = 0
    
    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        
        # Forward
        scores, embeddings, value, tactic_logits = model(
            batch.x, batch.edge_index, batch.derived_mask, batch.step_numbers,
            batch.eigvecs, batch.eigvals, batch.eig_mask,
            batch.edge_attr if hasattr(batch, 'edge_attr') else None,
            batch.batch if hasattr(batch, 'batch') else None
        )
        
        num_graphs = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
        batch_loss = 0.0
        batch_count = 0
        
        for i in range(num_graphs):
            start, end, target_local = extract_graph(batch, i)
            
            if target_local < 0 or target_local >= (end - start):
                continue
            
            graph_scores = scores[start:end]
            graph_embeddings = embeddings[start:end]
            graph_applicable = batch.applicable_mask[start:end] if hasattr(batch, 'applicable_mask') else torch.ones(end-start, dtype=torch.bool, device=device)
            
            if not graph_applicable[target_local]:
                continue
            
            loss = criterion(graph_scores, graph_embeddings, target_local, graph_applicable)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            batch_loss = batch_loss + loss
            batch_count += 1
            
            # Metrics
            top1 = graph_scores.argmax().item()
            total_hit1 += (top1 == target_local)
        
        if batch_count > 0:
            avg_loss = batch_loss / batch_count
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += avg_loss.item()
            num_samples += batch_count
    
    return {
        'loss': total_loss / len(loader),
        'hit@1': total_hit1 / max(num_samples, 1)
    }

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Simple evaluation"""
    model.eval()
    total_loss = 0.0
    total_hit1 = 0.0
    num_samples = 0
    
    for batch in tqdm(loader, desc="Evaluating"):
        batch = batch.to(device)
        
        scores, embeddings, value, tactic_logits = model(
            batch.x, batch.edge_index, batch.derived_mask, batch.step_numbers,
            batch.eigvecs, batch.eigvals, batch.eig_mask,
            batch.edge_attr if hasattr(batch, 'edge_attr') else None,
            batch.batch if hasattr(batch, 'batch') else None
        )
        
        num_graphs = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
        
        for i in range(num_graphs):
            start, end, target_local = extract_graph(batch, i)
            
            if target_local < 0 or target_local >= (end - start):
                continue
            
            graph_scores = scores[start:end]
            graph_embeddings = embeddings[start:end]
            graph_applicable = batch.applicable_mask[start:end] if hasattr(batch, 'applicable_mask') else torch.ones(end-start, dtype=torch.bool, device=device)
            
            if not graph_applicable[target_local]:
                continue
            
            loss = criterion(graph_scores, graph_embeddings, target_local, graph_applicable)
            
            if not torch.isnan(loss):
                total_loss += loss.item()
                num_samples += 1
            
            top1 = graph_scores.argmax().item()
            total_hit1 += (top1 == target_local)
    
    return {
        'loss': total_loss / max(num_samples, 1),
        'hit@1': total_hit1 / max(num_samples, 1)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--spectral-dir', default=None)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', default='cpu')
    
    args = parser.parse_args()
    
    print("🚀 Minimal Training Pipeline")
    
    device = torch.device(args.device)
    
    # Load data
    train_loader, val_loader, test_loader = create_loaders(
        args.data_dir, args.spectral_dir, args.batch_size, 42
    )
    
    # Create model
    model = get_model(in_dim=22, hidden_dim=256, num_layers=3, k=16).to(device)
    
    # Setup training
    criterion = get_recommended_loss('triplet_hard', margin=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    best_val = 0.0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Hit@1: {train_metrics['hit@1']:.2%}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Hit@1: {val_metrics['hit@1']:.2%}")
        
        if val_metrics['hit@1'] > best_val:
            best_val = val_metrics['hit@1']
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"🎯 New best: {best_val:.2%}")
    
    # Test
    model.load_state_dict(torch.load('best_model.pt'))
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"\n{'='*60}")
    print(f"Test - Loss: {test_metrics['loss']:.4f}, Hit@1: {test_metrics['hit@1']:.2%}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
TRAINEOF

echo "✅ train_minimal.py created"

# 3. Create run script
echo -e "\n📝 Creating run_fixed.sh..."

cat > run_fixed.sh << 'RUNEOF'
#!/bin/bash

echo "🚀 Running Fixed Training Pipeline"
echo ""

# Run diagnostics first
echo "Step 1: Running diagnostics..."
python diagnose.py --data-dir generated_data --spectral-dir spectral_cache

if [ $? -ne 0 ]; then
    echo "⚠️  Diagnostics found issues. Check output above."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Run training
echo ""
echo "Step 2: Starting training..."
python train_minimal.py \
    --data-dir generated_data \
    --spectral-dir spectral_cache \
    --batch-size 16 \
    --epochs 20 \
    --lr 1e-4 \
    --device cpu

echo ""
echo "✅ Training complete!"
RUNEOF

chmod +x run_fixed.sh

echo -e "\n================================"
echo "✅ ALL FIXES APPLIED"
echo "================================"
echo ""
echo "Next steps:"
echo "1. Run diagnostics: python diagnose.py --data-dir generated_data --spectral-dir spectral_cache"
echo "2. If OK, run: ./run_fixed.sh"
echo "3. Or run directly: python train_minimal.py --data-dir generated_data --spectral-dir spectral_cache"
echo ""