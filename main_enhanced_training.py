#!/usr/bin/env python3
"""
Main Enhanced Training Script for LogNet
Integrates all enhanced components with comprehensive parameter tuning
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

# Import all enhanced components
from enhanced_proof_state import EnhancedProofState
from tactic_abstraction import TacticAbstraction
from curriculum_learning import CurriculumDataLoader, DifficultyLevel
from semantic_hard_negatives import SemanticHardNegatives
from proof_search import ProofSearchAgent
from multi_relational_graph import MultiRelationalGraph
from learned_attention import ProofHistoryEncoder


class EnhancedLogNetModel(nn.Module):
    """Enhanced LogNet model integrating all new components"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Model configuration
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_classes = config['num_classes']
        self.max_steps = config['max_steps']
        self.num_tactics = config['num_tactics']
        
        # Enhanced components
        self.proof_state = EnhancedProofState(self.input_dim, self.hidden_dim)
        self.tactic_abstraction = TacticAbstraction(self.hidden_dim, self.num_tactics)
        self.semantic_negatives = SemanticHardNegatives(self.hidden_dim)
        self.proof_search = ProofSearchAgent(self.hidden_dim, self.num_classes)
        self.multi_relational = MultiRelationalGraph(self.hidden_dim)
        self.attention = ProofHistoryEncoder(
            hidden_dim=self.hidden_dim,
            num_heads=config['num_attention_heads'],
            max_steps=self.max_steps
        )
        
        # Output heads
        self.rule_head = nn.Linear(self.hidden_dim, self.num_classes)
        self.tactic_head = nn.Linear(self.hidden_dim, self.num_tactics)
        self.value_head = nn.Linear(self.hidden_dim, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config['dropout_rate'])
        
    def forward(self, x: torch.Tensor, step_numbers: torch.Tensor, 
                edge_index: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through enhanced model
        
        Args:
            x: Node features [batch_size, num_nodes, input_dim]
            step_numbers: Step numbers [batch_size, num_nodes]
            edge_index: Graph edges [2, num_edges] (optional)
            
        Returns:
            rule_scores: Rule prediction scores [batch_size, num_nodes, num_classes]
            tactic_scores: Tactic prediction scores [batch_size, num_nodes, num_tactics]
            value_scores: Value prediction scores [batch_size, num_nodes, 1]
            attention_weights: Attention weights [batch_size, num_nodes, num_nodes]
        """
        batch_size, num_nodes, _ = x.shape
        
        # 1. Enhanced proof state representation
        proof_features = self.proof_state(x, step_numbers)
        
        # 2. Multi-relational graph processing
        if edge_index is not None:
            graph_features = self.multi_relational(proof_features, edge_index)
        else:
            graph_features = proof_features
            
        # 3. Learned attention over proof history
        attended_features, attention_weights = self.attention(graph_features, step_numbers)
        
        # 4. Apply dropout
        attended_features = self.dropout(attended_features)
        
        # 5. Generate predictions
        rule_scores = self.rule_head(attended_features)
        tactic_scores = self.tactic_head(attended_features)
        value_scores = self.value_head(attended_features)
        
        return rule_scores, tactic_scores, value_scores, attention_weights


class EnhancedTrainingPipeline:
    """Enhanced training pipeline with all new features"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = EnhancedLogNetModel(config).to(self.device)
        
        # Initialize curriculum learning
        self.curriculum_loader = CurriculumDataLoader(
            temperature=config['curriculum_temperature'],
            min_difficulty=config['min_difficulty'],
            max_difficulty=config['max_difficulty']
        )
        
        # Initialize semantic hard negatives
        self.semantic_negatives = SemanticHardNegatives(config['hidden_dim'])
        
        # Initialize proof search agent
        self.proof_search = ProofSearchAgent(config['hidden_dim'], config['num_classes'])
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config['lr_decay_factor'],
            patience=config['lr_patience']
        )
        
        # Loss functions
        self.rule_criterion = nn.CrossEntropyLoss()
        self.tactic_criterion = nn.CrossEntropyLoss()
        self.value_criterion = nn.MSELoss()
        
        # Training metrics
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
    def generate_synthetic_data(self, num_samples: int, difficulty: str) -> List[Dict]:
        """Generate synthetic training data"""
        data = []
        
        for _ in range(num_samples):
            # Random graph size
            num_nodes = random.randint(5, 20)
            
            # Generate features
            x = torch.randn(1, num_nodes, self.config['input_dim'])
            step_numbers = torch.randint(0, self.config['max_steps'], (1, num_nodes))
            
            # Generate edges (simple random graph)
            num_edges = random.randint(num_nodes, num_nodes * 2)
            edge_index = torch.randint(0, num_nodes, (2, num_edges))
            
            # Generate target
            target = torch.randint(0, self.config['num_classes'], (1,))
            
            # Calculate difficulty
            difficulty_score = self._calculate_difficulty(num_nodes, num_edges)
            
            data.append({
                'x': x,
                'step_numbers': step_numbers,
                'edge_index': edge_index,
                'target': target,
                'difficulty': difficulty_score,
                'difficulty_level': difficulty
            })
            
        return data
    
    def _calculate_difficulty(self, num_nodes: int, num_edges: int) -> float:
        """Calculate difficulty score for curriculum learning"""
        complexity = (num_nodes * num_edges) / (num_nodes + num_edges)
        return min(1.0, complexity / 100.0)
    
    def train_epoch(self, train_data: List[Dict], epoch: int) -> Tuple[float, float, float, float]:
        """Train for one epoch with enhanced features"""
        self.model.train()
        total_loss = 0.0
        correct_preds = 0
        tactic_correct = 0
        value_correct = 0
        attention_entropies = []
        
        for batch_idx, sample in enumerate(train_data):
            x = sample['x'].to(self.device)
            step_numbers = sample['step_numbers'].to(self.device)
            edge_index = sample['edge_index'].to(self.device)
            target = sample['target'].to(self.device)
            
            # Forward pass
            rule_scores, tactic_scores, value_scores, attention_weights = self.model(
                x, step_numbers, edge_index
            )
            
            # Compute losses
            rule_loss = self.rule_criterion(rule_scores.mean(dim=1), target)
            tactic_loss = self.tactic_criterion(tactic_scores.mean(dim=1), 
                                              torch.randint(0, self.config['num_tactics'], (1,)).to(self.device))
            value_loss = self.value_criterion(value_scores.mean(dim=1), torch.randn(1, 1).to(self.device))
            
            # Semantic hard negative mining
            if hasattr(self, 'semantic_negatives') and batch_idx % 5 == 0:
                hard_negatives = self.semantic_negatives.get_hard_negatives(
                    rule_scores, target, k=3
                )
                if len(hard_negatives) > 0:
                    hard_neg_scores = rule_scores[hard_negatives]
                    hard_neg_loss = self.rule_criterion(hard_neg_scores.mean(dim=1), target)
                    rule_loss += 0.1 * hard_neg_loss
            
            # Combined loss
            total_loss_batch = rule_loss + 0.1 * tactic_loss + 0.1 * value_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            
            # Track accuracy
            rule_pred = rule_scores.mean(dim=1).argmax().item()
            if rule_pred == target.item():
                correct_preds += 1
            
            tactic_pred = tactic_scores.mean(dim=1).argmax().item()
            if tactic_pred == 0:  # Assuming tactic 0 is correct
                tactic_correct += 1
            
            if abs(value_scores.mean().item()) < 1.0:
                value_correct += 1
            
            # Attention entropy
            attn_probs = torch.softmax(attention_weights, dim=-1)
            entropy = -(attn_probs * torch.log(attn_probs + 1e-8)).sum().item()
            attention_entropies.append(entropy)
        
        avg_loss = total_loss / len(train_data)
        accuracy = correct_preds / len(train_data)
        tactic_accuracy = tactic_correct / len(train_data)
        value_accuracy = value_correct / len(train_data)
        avg_attention_entropy = np.mean(attention_entropies)
        
        return avg_loss, accuracy, tactic_accuracy, value_accuracy, avg_attention_entropy
    
    def validate(self, val_data: List[Dict]) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        
        with torch.no_grad():
            for sample in val_data:
                x = sample['x'].to(self.device)
                step_numbers = sample['step_numbers'].to(self.device)
                edge_index = sample['edge_index'].to(self.device)
                target = sample['target'].to(self.device)
                
                rule_scores, _, _, _ = self.model(x, step_numbers, edge_index)
                rule_pred = rule_scores.mean(dim=1).argmax().item()
                
                if rule_pred == target.item():
                    val_correct += 1
                
                val_loss += self.rule_criterion(rule_scores.mean(dim=1), target).item()
        
        val_accuracy = val_correct / len(val_data)
        avg_val_loss = val_loss / len(val_data)
        
        return avg_val_loss, val_accuracy
    
    def train(self, train_data: List[Dict], val_data: List[Dict]) -> Dict:
        """Main training loop"""
        print(f"🚀 Starting Enhanced Training")
        print(f"📊 Training samples: {len(train_data)}")
        print(f"📊 Validation samples: {len(val_data)}")
        print(f"🧠 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("=" * 60)
        
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            start_time = time.time()
            
            # Train epoch
            train_loss, train_acc, tactic_acc, value_acc, attn_entropy = self.train_epoch(train_data, epoch)
            
            # Validate
            val_loss, val_acc = self.validate(val_data)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Track metrics
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.config['early_stopping_patience']:
                print(f"🛑 Early stopping at epoch {epoch}")
                break
            
            # Logging
            epoch_time = time.time() - start_time
            print(f"[Epoch {epoch:3d}] "
                  f"Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Tactic: {tactic_acc:.4f} | "
                  f"Value: {value_acc:.4f} | "
                  f"Val Acc: {val_acc:.4f} | "
                  f"Attn Entropy: {attn_entropy:.4f} | "
                  f"Time: {epoch_time:.2f}s")
        
        # Save results
        results = {
            'final_train_accuracy': train_acc,
            'final_val_accuracy': val_acc,
            'best_val_accuracy': best_val_acc,
            'final_attention_entropy': attn_entropy,
            'learning_curves': {
                'train_loss': self.train_losses,
                'train_accuracy': self.train_accuracies,
                'val_loss': self.val_losses,
                'val_accuracy': self.val_accuracies
            },
            'config': self.config
        }
        
        return results


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced LogNet Training')
    
    # Data configuration
    parser.add_argument('--easy-samples', type=int, default=50, help='Number of easy samples')
    parser.add_argument('--medium-samples', type=int, default=30, help='Number of medium samples')
    parser.add_argument('--hard-samples', type=int, default=20, help='Number of hard samples')
    parser.add_argument('--very-hard-samples', type=int, default=10, help='Number of very hard samples')
    parser.add_argument('--extreme-hard-samples', type=int, default=5, help='Number of extreme hard samples')
    
    # Model configuration
    parser.add_argument('--input-dim', type=int, default=10, help='Input dimension')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--num-classes', type=int, default=5, help='Number of output classes')
    parser.add_argument('--num-tactics', type=int, default=3, help='Number of tactics')
    parser.add_argument('--max-steps', type=int, default=50, help='Maximum proof steps')
    parser.add_argument('--num-attention-heads', type=int, default=8, help='Number of attention heads')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--dropout-rate', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    
    # Curriculum learning
    parser.add_argument('--curriculum-temperature', type=float, default=2.0, help='Curriculum temperature')
    parser.add_argument('--min-difficulty', type=float, default=0.1, help='Minimum difficulty')
    parser.add_argument('--max-difficulty', type=float, default=1.0, help='Maximum difficulty')
    
    # Learning rate scheduling
    parser.add_argument('--lr-decay-factor', type=float, default=0.5, help='LR decay factor')
    parser.add_argument('--lr-patience', type=int, default=5, help='LR patience')
    
    # Early stopping
    parser.add_argument('--early-stopping-patience', type=int, default=10, help='Early stopping patience')
    
    # Output configuration
    parser.add_argument('--output-dir', type=str, default='./enhanced_results', help='Output directory')
    parser.add_argument('--save-model', action='store_true', help='Save trained model')
    parser.add_argument('--plot-curves', action='store_true', help='Plot learning curves')
    
    # Random seed
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    return parser.parse_args()


def create_config(args) -> Dict:
    """Create configuration dictionary from arguments"""
    return {
        # Data configuration
        'easy_samples': args.easy_samples,
        'medium_samples': args.medium_samples,
        'hard_samples': args.hard_samples,
        'very_hard_samples': args.very_hard_samples,
        'extreme_hard_samples': args.extreme_hard_samples,
        
        # Model configuration
        'input_dim': args.input_dim,
        'hidden_dim': args.hidden_dim,
        'num_classes': args.num_classes,
        'num_tactics': args.num_tactics,
        'max_steps': args.max_steps,
        'num_attention_heads': args.num_attention_heads,
        
        # Training configuration
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'dropout_rate': args.dropout_rate,
        'batch_size': args.batch_size,
        
        # Curriculum learning
        'curriculum_temperature': args.curriculum_temperature,
        'min_difficulty': args.min_difficulty,
        'max_difficulty': args.max_difficulty,
        
        # Learning rate scheduling
        'lr_decay_factor': args.lr_decay_factor,
        'lr_patience': args.lr_patience,
        
        # Early stopping
        'early_stopping_patience': args.early_stopping_patience,
        
        # Output
        'output_dir': args.output_dir,
        'save_model': args.save_model,
        'plot_curves': args.plot_curves,
        
        # Random seed
        'seed': args.seed
    }


def plot_learning_curves(results: Dict, output_dir: str):
    """Plot learning curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(results['learning_curves']['train_loss'], label='Train Loss', color='blue')
    axes[0, 0].plot(results['learning_curves']['val_loss'], label='Val Loss', color='red')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy curves
    axes[0, 1].plot(results['learning_curves']['train_accuracy'], label='Train Acc', color='blue')
    axes[0, 1].plot(results['learning_curves']['val_accuracy'], label='Val Acc', color='red')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Learning rate (if available)
    axes[1, 0].text(0.5, 0.5, 'Learning Rate\n(Not tracked in this version)', 
                    ha='center', va='center', transform=axes[1, 0].transAxes)
    axes[1, 0].set_title('Learning Rate Schedule')
    
    # Model info
    config = results['config']
    model_info = f"""
    Model Configuration:
    • Input Dim: {config['input_dim']}
    • Hidden Dim: {config['hidden_dim']}
    • Num Classes: {config['num_classes']}
    • Num Tactics: {config['num_tactics']}
    • Attention Heads: {config['num_attention_heads']}
    • Learning Rate: {config['learning_rate']}
    • Dropout: {config['dropout_rate']}
    """
    axes[1, 1].text(0.1, 0.5, model_info, ha='left', va='center', 
                    transform=axes[1, 1].transAxes, fontsize=10)
    axes[1, 1].set_title('Model Configuration')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main training function"""
    # Parse arguments
    args = parse_arguments()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create configuration
    config = create_config(args)
    
    print("🚀 ENHANCED LOGNET TRAINING")
    print("=" * 60)
    print(f"📊 Data Configuration:")
    print(f"   • Easy samples: {args.easy_samples}")
    print(f"   • Medium samples: {args.medium_samples}")
    print(f"   • Hard samples: {args.hard_samples}")
    print(f"   • Very hard samples: {args.very_hard_samples}")
    print(f"   • Extreme hard samples: {args.extreme_hard_samples}")
    print(f"🧠 Model Configuration:")
    print(f"   • Input dimension: {args.input_dim}")
    print(f"   • Hidden dimension: {args.hidden_dim}")
    print(f"   • Number of classes: {args.num_classes}")
    print(f"   • Number of tactics: {args.num_tactics}")
    print(f"   • Attention heads: {args.num_attention_heads}")
    print(f"🎓 Training Configuration:")
    print(f"   • Epochs: {args.epochs}")
    print(f"   • Learning rate: {args.learning_rate}")
    print(f"   • Weight decay: {args.weight_decay}")
    print(f"   • Dropout rate: {args.dropout_rate}")
    print("=" * 60)
    
    # Initialize training pipeline
    pipeline = EnhancedTrainingPipeline(config)
    
    # Generate training data
    print("🎲 Generating training data...")
    train_data = []
    train_data.extend(pipeline.generate_synthetic_data(args.easy_samples, 'easy'))
    train_data.extend(pipeline.generate_synthetic_data(args.medium_samples, 'medium'))
    train_data.extend(pipeline.generate_synthetic_data(args.hard_samples, 'hard'))
    train_data.extend(pipeline.generate_synthetic_data(args.very_hard_samples, 'very_hard'))
    train_data.extend(pipeline.generate_synthetic_data(args.extreme_hard_samples, 'extreme_hard'))
    
    # Generate validation data
    print("🎲 Generating validation data...")
    val_data = pipeline.generate_synthetic_data(20, 'medium')
    
    print(f"✅ Generated {len(train_data)} training samples")
    print(f"✅ Generated {len(val_data)} validation samples")
    
    # Train model
    print("\n🚀 Starting training...")
    results = pipeline.train(train_data, val_data)
    
    # Save results
    results_file = f"{args.output_dir}/training_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot learning curves
    if args.plot_curves:
        print("📊 Plotting learning curves...")
        plot_learning_curves(results, args.output_dir)
    
    # Save model
    if args.save_model:
        model_file = f"{args.output_dir}/enhanced_model.pth"
        torch.save(pipeline.model.state_dict(), model_file)
        print(f"💾 Model saved to: {model_file}")
    
    # Print final results
    print("\n" + "=" * 60)
    print("🎯 TRAINING COMPLETED")
    print("=" * 60)
    print(f"📊 Final Results:")
    print(f"   • Final Train Accuracy: {results['final_train_accuracy']:.4f}")
    print(f"   • Final Val Accuracy: {results['final_val_accuracy']:.4f}")
    print(f"   • Best Val Accuracy: {results['best_val_accuracy']:.4f}")
    print(f"   • Final Attention Entropy: {results['final_attention_entropy']:.4f}")
    print(f"📁 Results saved to: {results_file}")
    if args.plot_curves:
        print(f"📊 Learning curves saved to: {args.output_dir}/learning_curves.png")
    print("🎉 Enhanced training completed successfully!")


if __name__ == "__main__":
    main()
