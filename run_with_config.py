#!/usr/bin/env python3
"""
Configuration-based Enhanced LogNet Training
Loads predefined configurations and runs training
"""

import argparse
import json
import os
import sys
from pathlib import Path

def load_config(config_name: str) -> dict:
    """Load configuration from training_configs.json"""
    config_file = Path(__file__).parent / "training_configs.json"
    
    if not config_file.exists():
        print(f"❌ Configuration file not found: {config_file}")
        sys.exit(1)
    
    with open(config_file, 'r') as f:
        configs = json.load(f)
    
    if config_name not in configs:
        print(f"❌ Configuration '{config_name}' not found!")
        print(f"Available configurations: {', '.join(configs.keys())}")
        sys.exit(1)
    
    return configs[config_name]

def run_training(config: dict):
    """Run training with the given configuration"""
    # Import here to avoid issues if modules are missing
    from main_enhanced_training import EnhancedTrainingPipeline
    
    print(f"🚀 Running Enhanced LogNet Training")
    print(f"📋 Configuration: {config.get('description', 'Custom configuration')}")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Initialize training pipeline
    pipeline = EnhancedTrainingPipeline(config)
    
    # Generate training data
    print("🎲 Generating training data...")
    train_data = []
    train_data.extend(pipeline.generate_synthetic_data(config['easy_samples'], 'easy'))
    train_data.extend(pipeline.generate_synthetic_data(config['medium_samples'], 'medium'))
    train_data.extend(pipeline.generate_synthetic_data(config['hard_samples'], 'hard'))
    train_data.extend(pipeline.generate_synthetic_data(config['very_hard_samples'], 'very_hard'))
    train_data.extend(pipeline.generate_synthetic_data(config['extreme_hard_samples'], 'extreme_hard'))
    
    # Generate validation data
    print("🎲 Generating validation data...")
    val_data = pipeline.generate_synthetic_data(20, 'medium')
    
    print(f"✅ Generated {len(train_data)} training samples")
    print(f"✅ Generated {len(val_data)} validation samples")
    
    # Train model
    print("\n🚀 Starting training...")
    results = pipeline.train(train_data, val_data)
    
    # Save results
    results_file = f"{config['output_dir']}/training_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot learning curves
    if config.get('plot_curves', False):
        print("📊 Plotting learning curves...")
        from main_enhanced_training import plot_learning_curves
        plot_learning_curves(results, config['output_dir'])
    
    # Save model
    if config.get('save_model', False):
        model_file = f"{config['output_dir']}/enhanced_model.pth"
        import torch
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
    if config.get('plot_curves', False):
        print(f"📊 Learning curves saved to: {config['output_dir']}/learning_curves.png")
    print("🎉 Enhanced training completed successfully!")

def list_configurations():
    """List all available configurations"""
    config_file = Path(__file__).parent / "training_configs.json"
    
    if not config_file.exists():
        print(f"❌ Configuration file not found: {config_file}")
        return
    
    with open(config_file, 'r') as f:
        configs = json.load(f)
    
    print("📋 Available Training Configurations:")
    print("=" * 50)
    
    for name, config in configs.items():
        description = config.get('description', 'No description')
        print(f"🔧 {name}")
        print(f"   {description}")
        print(f"   • Samples: {config['easy_samples'] + config['medium_samples'] + config['hard_samples'] + config['very_hard_samples'] + config['extreme_hard_samples']}")
        print(f"   • Hidden dim: {config['hidden_dim']}")
        print(f"   • Epochs: {config['epochs']}")
        print(f"   • Output: {config['output_dir']}")
        print()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Configuration-based Enhanced LogNet Training')
    parser.add_argument('config', nargs='?', help='Configuration name to use')
    parser.add_argument('--list', action='store_true', help='List available configurations')
    parser.add_argument('--show', help='Show details of a specific configuration')
    
    args = parser.parse_args()
    
    if args.list:
        list_configurations()
        return
    
    if args.show:
        config = load_config(args.show)
        print(f"🔧 Configuration: {args.show}")
        print(f"📝 Description: {config.get('description', 'No description')}")
        print("=" * 50)
        for key, value in config.items():
            if key != 'description':
                print(f"   {key}: {value}")
        return
    
    if not args.config:
        print("❌ Please specify a configuration name")
        print("Use --list to see available configurations")
        return
    
    # Load and run configuration
    config = load_config(args.config)
    run_training(config)

if __name__ == "__main__":
    main()
