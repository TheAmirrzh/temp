#!/usr/bin/env python3
"""
Drop-in Replacement for main_training.py
Uses FixedProofState with proper derivation tracking

USAGE:
  python main_training_fixed.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import json
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
from pathlib import Path
import sys

# Import fixed components
# NOTE: Either copy FixedProofState into proof_state.py or import from complete_fix.py
try:
    from complete_fix import FixedProofState, generate_valid_synthetic_data, verify_instance
    ProofState = FixedProofState  # Use fixed version
except ImportError:
    print("⚠️  Using original ProofState (will have bugs!)")
    from proof_state import ProofState
    
    # Fallback data generation
    def generate_valid_synthetic_data(num_instances):
        print("⚠️  Using fallback data generation")
        return []

from proof_state import ProofStateEncoder
from proof_search import PolicyNetwork
from data_loader import CurriculumDataLoader, estimate_instance_difficulties
from enhanced_model import EnhancedLogNetModel, HardNegativeLoss
from proof_validator import ProofValidator, EvaluationMetrics


class FixedIntegratedTrainer:
    """Integrated trainer using FixedProofState."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._set_seeds(config.get('seed', 42))
        
        # Components
        self.state_encoder = ProofStateEncoder(hidden_dim=config['hidden_dim'])
        self.policy_network = PolicyNetwork(config['hidden_dim'], config['num_rules'])
        self.model = EnhancedLogNetModel(config).to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Loss functions
        self.criterion_rule = nn.CrossEntropyLoss()
        self.criterion_hard_neg = HardNegativeLoss(margin=1.0, hard_neg_weight=2.0)
        
        # Tracking
        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'num_samples_per_epoch': [],
        }
        
        self.proof_validator = ProofValidator()
    
    def _set_seeds(self, seed: int):
        """Set random seeds."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    def train_epoch(self, curriculum_loader: CurriculumDataLoader, epoch: int, 
                   total_epochs: int) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        num_samples = 0
        skipped = 0
        
        num_batches = self.config.get('batches_per_epoch', 30)
        
        for batch_idx in range(num_batches):
            # Get batch
            batch = curriculum_loader.get_batch(epoch, total_epochs)
            
            if not batch:
                continue
            
            for sample in batch:
                try:
                    proof_state = sample['proof_state']
                    difficulty_level = sample.get('difficulty_level', 'medium')
                    
                    # Get applicable rules
                    applicable_rules = proof_state.get_applicable_rules()
                    
                    if not applicable_rules:
                        skipped += 1
                        continue
                    
                    # Forward pass
                    output = self.model(proof_state)
                    rule_logits = output['rule_logits']
                    
                    # Target: first applicable rule
                    target_rule = applicable_rules[0]
                    
                    if target_rule >= len(rule_logits):
                        skipped += 1
                        continue
                    
                    # Compute losses
                    rule_loss = self.criterion_rule(
                        rule_logits.unsqueeze(0),
                        torch.tensor([target_rule], device=self.device)
                    )
                    
                    hard_neg_loss = self.criterion_hard_neg(
                        rule_logits, target_rule, applicable_rules
                    )
                    
                    loss = rule_loss + 0.3 * hard_neg_loss
                    
                    # Backward
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    
                    # Track
                    total_loss += loss.item()
                    pred_rule = rule_logits.argmax().item()
                    if pred_rule == target_rule:
                        correct += 1
                    
                    num_samples += 1
                    
                    # Update curriculum
                    performance = 1.0 if pred_rule == target_rule else 0.0
                    curriculum_loader.update_performance(difficulty_level, performance)
                
                except Exception as e:
                    skipped += 1
                    continue
        
        # Compute metrics
        avg_loss = total_loss / max(num_samples, 1)
        accuracy = correct / max(num_samples, 1)
        
        # Log statistics
        if num_samples == 0:
            print(f"  ⚠️  No valid samples in epoch {epoch}! (Skipped: {skipped})")
        
        return avg_loss, accuracy
    
    def validate(self, curriculum_loader: CurriculumDataLoader, 
                num_val_batches: int = 10) -> Tuple[float, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        num_samples = 0
        
        with torch.no_grad():
            for _ in range(num_val_batches):
                batch = curriculum_loader.get_batch(0, 1)
                
                if not batch:
                    continue
                
                for sample in batch:
                    try:
                        proof_state = sample['proof_state']
                        
                        applicable_rules = proof_state.get_applicable_rules()
                        if not applicable_rules:
                            continue
                        
                        output = self.model(proof_state)
                        rule_logits = output['rule_logits']
                        
                        target_rule = applicable_rules[0]
                        
                        loss = self.criterion_rule(
                            rule_logits.unsqueeze(0),
                            torch.tensor([target_rule], device=self.device)
                        )
                        
                        total_loss += loss.item()
                        
                        pred_rule = rule_logits.argmax().item()
                        if pred_rule == target_rule:
                            correct += 1
                        
                        num_samples += 1
                    
                    except Exception:
                        continue
        
        avg_loss = total_loss / max(num_samples, 1)
        accuracy = correct / max(num_samples, 1)
        
        return avg_loss, accuracy
    
    def train(self, instances: List[Dict], num_epochs: int = 20):
        """Main training loop."""
        print("=" * 70)
        print("🚀 FIXED INTEGRATED PROOF TRAINING")
        print("=" * 70)
        print(f"📊 Training instances: {len(instances)}")
        print(f"📈 Epochs: {num_epochs}")
        print(f"🔧 Device: {self.device}")
        print(f"🧠 Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        print("=" * 70)
        
        # Create curriculum loader
        curriculum_loader = CurriculumDataLoader(
            instances,
            batch_size=self.config.get('batch_size', 4),
            start_temperature=self.config.get('curriculum_temperature', 2.0)
        )
        
        best_val_loss = float('inf')
        patience = 0
        max_patience = 5
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(
                curriculum_loader, epoch, num_epochs
            )
            
            # Validate
            val_loss, val_acc = self.validate(curriculum_loader)
            
            # Scheduler
            self.scheduler.step(val_loss)
            
            # Track
            self.train_history['epoch'].append(epoch)
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_acc'].append(val_acc)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
            else:
                patience += 1
            
            epoch_time = time.time() - start_time
            
            # Log
            print(f"[Epoch {epoch:3d}/{num_epochs}] "
                  f"Loss: {train_loss:.4f} | "
                  f"Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | "
                  f"Time: {epoch_time:.2f}s")
            
            if patience >= max_patience:
                print(f"⏸️  Early stopping at epoch {epoch}")
                break
        
        print("=" * 70)
        print("✅ TRAINING COMPLETED")
        print("=" * 70)
        
        return self.train_history
    
    def save_results(self, output_dir: str = './results'):
        """Save results."""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Save history
        with open(f'{output_dir}/training_history.json', 'w') as f:
            json.dump(self.train_history, f, indent=2)
        
        # Save model
        torch.save(self.model.state_dict(), f'{output_dir}/model.pth')
        
        print(f"💾 Results saved to {output_dir}")


def main():
    """Main entry point."""
    config = {
        'hidden_dim': 128,
        'num_rules': 50,
        'num_tactics': 10,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'batch_size': 4,
        'batches_per_epoch': 30,
        'epochs': 20,
        'curriculum_temperature': 2.0,
        'num_simulations': 50,
        'seed': 42
    }
    
    print("\n" + "=" * 70)
    print("🔬 DEEP DIAGNOSIS & FIXED TRAINING PIPELINE")
    print("=" * 70)
    
    # Create trainer
    trainer = FixedIntegratedTrainer(config)
    
    # Generate FIXED data
    print("\n🎲 Generating synthetic data with FIXED logic...")
    try:
        instances = generate_valid_synthetic_data(num_instances=100)
        print(f"✅ Generated {len(instances)} instances")
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        return 1
    
    # CRITICAL: Verify instances
    print("\n🔍 Verifying instance validity (comprehensive check)...")
    valid_instances = []
    invalid_count = 0
    
    for inst in instances:
        is_valid, msg = verify_instance(inst)
        if is_valid:
            valid_instances.append(inst)
        else:
            invalid_count += 1
            if invalid_count <= 3:
                print(f"  ⚠️  Instance {inst.get('instance_id', '?')}: {msg}")
    
    print(f"\n📊 Validation Results:")
    print(f"  ✅ Valid: {len(valid_instances)}/{len(instances)}")
    print(f"  ❌ Invalid: {invalid_count}/{len(instances)}")
    
    if len(valid_instances) < 10:
        print("\n❌ ERROR: Not enough valid instances!")
        print("   This indicates a fundamental problem with data generation.")
        return 1
    
    # Sample verification: Show details of first valid instance
    if valid_instances:
        print("\n📝 Sample Instance Details:")
        sample = valid_instances[0]
        sample_state = FixedProofState(
            sample['facts'],
            sample['rules'],
            sample['goal']
        )
        print(f"  Instance ID: {sample.get('instance_id')}")
        print(f"  Total facts: {len(sample_state.facts)}")
        print(f"  Axioms: {sample_state.num_axioms}")
        print(f"  Available facts: {len(sample_state.available_facts)}")
        print(f"  Rules: {len(sample_state.rules)}")
        print(f"  Applicable rules: {sample_state.get_applicable_rules()}")
        print(f"  Goal: {sample['goal']}")
        
        # Test rule application
        applicable = sample_state.get_applicable_rules()
        if applicable:
            print(f"\n  🧪 Testing rule application...")
            print(f"     Before: {len(sample_state.available_facts)} available facts")
            success = sample_state.apply_rule(applicable[0])
            print(f"     Applied rule {applicable[0]}: {success}")
            print(f"     After: {len(sample_state.available_facts)} available facts")
            print(f"     Goal satisfied: {sample_state.goal_satisfied()}")
    
    # Add difficulty estimates
    print("\n📈 Adding difficulty estimates...")
    valid_instances = estimate_instance_difficulties(valid_instances)
    
    # Train
    print("\n🎓 Starting training with {len(valid_instances)} valid instances...")
    try:
        history = trainer.train(valid_instances, num_epochs=config['epochs'])
        
        # Check if any training happened
        if all(loss == 0.0 for loss in history['train_loss']):
            print("\n⚠️  WARNING: All losses are 0.0!")
            print("   This means no samples were processed during training.")
            return 1
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Save results
    print("\n💾 Saving results...")
    trainer.save_results('./results')
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 TRAINING SUMMARY")
    print("=" * 70)
    final_train_acc = history['train_acc'][-1] if history['train_acc'] else 0.0
    final_val_acc = history['val_acc'][-1] if history['val_acc'] else 0.0
    best_val_acc = max(history['val_acc']) if history['val_acc'] else 0.0
    
    print(f"  Final Train Accuracy: {final_train_acc:.4f}")
    print(f"  Final Val Accuracy: {final_val_acc:.4f}")
    print(f"  Best Val Accuracy: {best_val_acc:.4f}")
    print(f"  Total Epochs: {len(history['epoch'])}")
    
    if best_val_acc > 0.0:
        print("\n✅ Training completed successfully!")
        print("📈 Check './results/' for outputs")
        return 0
    else:
        print("\n⚠️  Training completed but achieved 0% accuracy.")
        print("   This suggests a problem with the learning setup.")
        return 1


if __name__ == "__main__":
    sys.exit(main())