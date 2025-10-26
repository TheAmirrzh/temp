"""
Main Training Script - Fully Integrated Pipeline
Fixes all data flow and component integration issues
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

from proof_state import ProofState, ProofStateEncoder, ProofValidator
from proof_search import (
    MCTSSearch, PolicyNetwork, RewardComputer, ProofSearchAgent
)
from data_loader import (
    CurriculumDataLoader, DifficultyEstimator, estimate_instance_difficulties
)
from enhanced_model import (
    EnhancedLogNetModel, EnhancedTrainingPipeline, HardNegativeLoss
)
from proof_validator import ProofValidator as PValidator, EvaluationMetrics, CurriculumAnalyzer


class IntegratedProofTrainer:
    """
    Complete trainer integrating all components.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Random seeds
        self._set_seeds(config.get('seed', 42))
        
        # Components
        self.state_encoder = ProofStateEncoder(hidden_dim=config['hidden_dim'])
        self.policy_network = PolicyNetwork(config['hidden_dim'], config['num_rules'])
        self.proof_search_agent = ProofSearchAgent(
            self.state_encoder,
            hidden_dim=config['hidden_dim'],
            num_actions=config['num_rules'],
            num_simulations=config.get('num_simulations', 50)
        )
        
        self.model = EnhancedLogNetModel(config).to(self.device)
        
        # Optimizers
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
        self.criterion_value = nn.MSELoss()
        
        # Validators and metrics
        self.proof_validator = PValidator()
        self.eval_metrics = EvaluationMetrics()
        self.curriculum_analyzer = CurriculumAnalyzer()
        
        # Tracking
        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'curriculum_info': []
        }
    
    def _set_seeds(self, seed: int):
        """Set random seeds."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    def generate_synthetic_data(self, num_instances: int = 100) -> List[Dict]:
        """Generate synthetic proof instances for demonstration."""
        instances = []
        
        for i in range(num_instances):
            # Random problem size
            num_axioms = random.randint(3, 8)
            num_derived = random.randint(2, 5)  # NEW FACTS to derive
            num_rules = random.randint(2, 8)
            
            total_facts = num_axioms + num_derived
            
            # Create axioms (base facts)
            facts = [
                {'formula': f'fact_{j}', 'confidence': random.uniform(0.7, 1.0)}
                for j in range(num_axioms)
            ]
            
            # Create rules that DERIVE new facts
            rules = []
            for j in range(num_rules):
                # Premises: choose from axioms only
                body = random.sample(
                    range(num_axioms),  # Only from axioms, not derived facts
                    min(random.randint(1, min(3, num_axioms)), num_axioms)
                )
                
                # Conclusion: a NEW derived fact (index >= num_axioms)
                head = [random.randint(num_axioms, total_facts - 1)]
                
                rules.append({
                    'name': f'rule_{j}',
                    'body': body,
                    'head': head,
                    'confidence': random.uniform(0.7, 0.99),  # High confidence
                    'tactic_type': random.choice(['forward_chain', 'backward_chain'])
                })
            
            # Add placeholder derived facts to facts list
            # (These will be "derived" by applying rules)
            for j in range(num_derived):
                facts.append({
                    'formula': f'derived_{j}',
                    'confidence': 1.0
                })
            
            instance = {
                'facts': facts,
                'rules': rules,
                'goal': f'derived_{random.randint(0, num_derived - 1)}',  # Goal is a derived fact
                'max_depth': random.randint(5, 15),
                'max_steps': random.randint(10, 30)
            }
            
            instances.append(instance)
        
        # Add difficulty estimates
        instances = estimate_instance_difficulties(instances)
        
        return instances
    def train_epoch(self, curriculum_loader: CurriculumDataLoader, epoch: int, 
               total_epochs: int) -> Tuple[float, float]:
        """Train for one epoch with proper error handling."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        num_samples = 0
        
        for batch_idx in range(self.config.get('batches_per_epoch', 50)):
            # Get curriculum-guided batch
            batch = curriculum_loader.get_batch(epoch, total_epochs)
            
            if not batch:
                continue
            
            for sample in batch:
                try:
                    proof_state = sample['proof_state']
                    difficulty_level = sample['difficulty_level']
                    
                    # Get applicable rules
                    applicable_rules = proof_state.get_applicable_rules()
                    
                    # CRITICAL FIX: Skip only if truly no rules, but log it
                    if not applicable_rules:
                        print(f"  [WARNING] No applicable rules for instance at epoch {epoch}")
                        continue
                    
                    # Forward pass
                    output = self.model(proof_state)
                    rule_logits = output['rule_logits']
                    
                    # Select target rule (first applicable for now)
                    target_rule = applicable_rules[0]
                    
                    # Ensure target is valid
                    if target_rule >= len(rule_logits):
                        print(f"  [WARNING] Invalid rule index {target_rule} >= {len(rule_logits)}")
                        continue
                    
                    # Losses
                    rule_loss = self.criterion_rule(
                        rule_logits.unsqueeze(0),
                        torch.tensor([target_rule], device=self.device)
                    )
                    
                    hard_neg_loss = self.criterion_hard_neg(
                        rule_logits, target_rule, applicable_rules
                    )
                    
                    # Combined loss
                    loss = rule_loss + 0.3 * hard_neg_loss
                    
                    # Backward
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    
                    # Tracking
                    total_loss += loss.item()
                    
                    # Check accuracy
                    pred_rule = rule_logits.argmax().item()
                    if pred_rule == target_rule:
                        correct += 1
                    
                    num_samples += 1
                    
                    # Update curriculum
                    performance = 1.0 if pred_rule == target_rule else 0.0
                    curriculum_loader.update_performance(difficulty_level, performance)
                    self.curriculum_analyzer.track_performance(
                        difficulty_level, performance, epoch
                    )
                
                except Exception as e:
                    print(f"  [ERROR] Failed to process sample: {e}")
                    traceback.import_exc()
                    continue
        
        # Compute averages with proper handling
        avg_loss = total_loss / max(num_samples, 1)
        accuracy = correct / max(num_samples, 1)
        
        # Debug logging
        if num_samples == 0:
            print(f"  [WARNING] No samples processed in epoch {epoch}!")
        
        return avg_loss, accuracy
    
    def validate(self, curriculum_loader: CurriculumDataLoader, 
                num_val_batches: int = 20) -> Tuple[float, float]:
        """
        Validate model.
        
        Returns:
            (avg_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        num_samples = 0
        
        with torch.no_grad():
            for _ in range(num_val_batches):
                # Get validation batch
                batch = curriculum_loader.get_batch(0, 1)  # Use static sampling
                
                if not batch:
                    continue
                
                for sample in batch:
                    proof_state = sample['proof_state']
                    
                    applicable_rules = proof_state.get_applicable_rules()
                    if not applicable_rules:
                        continue
                    
                    # Forward pass
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
        
        avg_loss = total_loss / max(num_samples, 1)
        accuracy = correct / max(num_samples, 1)
        
        return avg_loss, accuracy
    
    def train(self, instances: List[Dict], num_epochs: int = 20):
        """
        Main training loop.
        """
        print("=" * 70)
        print("🚀 INTEGRATED PROOF TRAINING PIPELINE")
        print("=" * 70)
        print(f"📊 Training on {len(instances)} instances")
        print(f"📈 Epochs: {num_epochs}")
        print(f"🔧 Device: {self.device}")
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
            
            # Training
            train_loss, train_acc = self.train_epoch(
                curriculum_loader, epoch, num_epochs
            )
            
            # Validation
            val_loss, val_acc = self.validate(curriculum_loader, num_val_batches=10)
            
            # Scheduler step
            self.scheduler.step(val_loss)
            
            # Tracking
            self.train_history['epoch'].append(epoch)
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_acc'].append(val_acc)
            self.train_history['curriculum_info'].append(
                curriculum_loader.get_curriculum_info()
            )
            
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
        """Save training results."""
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
        # Model
        'hidden_dim': 128,
        'num_rules': 50,
        'num_tactics': 10,
        
        # Training
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'batch_size': 4,
        'batches_per_epoch': 50,
        'epochs': 20,
        
        # Curriculum
        'curriculum_temperature': 2.0,
        'num_simulations': 50,
        
        # General
        'seed': 42
    }
    
    # Create trainer
    trainer = IntegratedProofTrainer(config)
    
    # Generate synthetic data
    print("🎲 Generating synthetic data...")
    instances = trainer.generate_synthetic_data(num_instances=100)
    print(f"✅ Generated {len(instances)} instances")
    
    # Train
    history = trainer.train(instances, num_epochs=config['epochs'])
    
    # Save results
    trainer.save_results('./results')


if __name__ == "__main__":
    main()