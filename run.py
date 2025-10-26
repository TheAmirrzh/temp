#!/usr/bin/env python3
"""
Quick start script for proof training
"""

import sys
from main_training import IntegratedProofTrainer

def main():
    print("🚀 Starting Proof Training Pipeline\n")
    
    # Configuration
    config = {
        # Model architecture
        'hidden_dim': 128,
        'num_rules': 50,
        'num_tactics': 10,
        
        # Training hyperparameters
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'batch_size': 4,
        'batches_per_epoch': 50,
        'epochs': 20,
        
        # Curriculum learning
        'curriculum_temperature': 2.0,
        'num_simulations': 50,
        
        # Random seed
        'seed': 42
    }
    
    try:
        # Create trainer
        trainer = IntegratedProofTrainer(config)
        
        # Generate synthetic data (100 proof instances)
        print("📊 Generating synthetic proof instances...")
        instances = trainer.generate_synthetic_data(num_instances=100)
        print(f"✅ Generated {len(instances)} instances\n")
        
        # Train model
        print("🎓 Starting training...\n")
        history = trainer.train(instances, num_epochs=config['epochs'])
        
        # Save results
        print("\n💾 Saving results...")
        trainer.save_results('./results')
        
        print("\n✅ Training completed successfully!")
        print("📈 Check './results/' for outputs")
        
        return 0
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())