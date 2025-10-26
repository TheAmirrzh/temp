#!/usr/bin/env python3
"""
Comprehensive test script to verify all components work correctly
Run this FIRST to ensure everything is set up properly
"""

import sys
import traceback

def test_imports():
    """Test that all modules can be imported."""
    print("\n" + "="*70)
    print("✓ TEST 1: Module Imports")
    print("="*70)
    
    try:
        print("  - Importing torch...", end=" ")
        import torch
        print(f"✓ (v{torch.__version__})")
        
        print("  - Importing proof_state...", end=" ")
        from proof_state import ProofState, ProofStateEncoder, Fact, Rule
        print("✓")
        
        print("  - Importing proof_search...", end=" ")
        from proof_search import MCTSSearch, PolicyNetwork, RewardComputer, ProofSearchAgent
        print("✓")
        
        print("  - Importing data_loader...", end=" ")
        from data_loader import CurriculumDataLoader, DifficultyEstimator
        print("✓")
        
        print("  - Importing enhanced_model...", end=" ")
        from enhanced_model import EnhancedLogNetModel, TacticDecoder
        print("✓")
        
        print("  - Importing proof_validator...", end=" ")
        from proof_validator import ProofValidator, EvaluationMetrics
        print("✓")
        
        print("  - Importing main_training...", end=" ")
        from main_training import IntegratedProofTrainer
        print("✓")
        
        print("\n✅ All imports successful!")
        return True
    
    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        traceback.print_exc()
        return False


def test_proof_state():
    """Test ProofState functionality."""
    print("\n" + "="*70)
    print("✓ TEST 2: ProofState Creation & Transitions")
    print("="*70)
    
    try:
        from proof_state import ProofState
        
        # Create simple proof state
        facts = [
            {'formula': 'A', 'confidence': 1.0},
            {'formula': 'B', 'confidence': 1.0}
        ]
        
        rules = [
            {
                'name': 'conjunction_intro',
                'body': [0, 1],
                'head': [2],
                'confidence': 0.95
            }
        ]
        
        state = ProofState(facts, rules, 'C', max_depth=10, max_steps=20)
        
        print(f"  - Created ProofState with {len(state.facts)} facts")
        print(f"  - Available rules: {state.get_applicable_rules()}")
        print(f"  - Open goals: {len(state.open_goals)}")
        print(f"  - Goal satisfied: {state.goal_satisfied()}")
        
        # Test state copy
        state_copy = state.copy()
        print(f"  - State copied successfully")
        
        # Test state transitions
        applicable = state.get_applicable_rules()
        if applicable:
            success = state.apply_rule(applicable[0])
            print(f"  - Applied rule: {success}")
            print(f"  - New facts: {len(state.facts)}")
        
        print("\n✅ ProofState tests passed!")
        return True
    
    except Exception as e:
        print(f"\n❌ ProofState test failed: {e}")
        traceback.print_exc()
        return False


def test_mcts():
    """Test MCTS components."""
    print("\n" + "="*70)
    print("✓ TEST 3: MCTS & Policy Network")
    print("="*70)
    
    try:
        import torch
        from proof_state import ProofState, ProofStateEncoder
        from proof_search import PolicyNetwork, MCTSSearch, RewardComputer
        
        # Create encoder
        encoder = ProofStateEncoder(hidden_dim=64)
        print("  - Created ProofStateEncoder")
        
        # Create policy network
        policy_net = PolicyNetwork(hidden_dim=64, num_actions=20)
        print("  - Created PolicyNetwork")
        
        # Create MCTS
        mcts = MCTSSearch(policy_net, encoder, num_simulations=5)
        print("  - Created MCTSSearch")
        
        # Create reward computer
        reward_computer = RewardComputer()
        print("  - Created RewardComputer")
        
        # Test on simple state
        facts = [{'formula': f'f_{i}', 'confidence': 1.0} for i in range(5)]
        rules = [
            {
                'name': f'rule_{i}',
                'body': [i % 5, (i+1) % 5],
                'head': [5 + i] if 5 + i < 10 else [0],
                'confidence': 0.9
            }
            for i in range(3)
        ]
        
        state = ProofState(facts, rules, 'goal', max_depth=10)
        
        # Test state encoding
        embedding = encoder(state)
        print(f"  - State embedding shape: {embedding.shape}")
        
        # Test policy network
        policy_logits, value = policy_net(embedding)
        print(f"  - Policy logits shape: {policy_logits.shape}")
        print(f"  - Value: {value.item():.4f}")
        
        print("\n✅ MCTS tests passed!")
        return True
    
    except Exception as e:
        print(f"\n❌ MCTS test failed: {e}")
        traceback.print_exc()
        return False


def test_curriculum():
    """Test curriculum loading."""
    print("\n" + "="*70)
    print("✓ TEST 4: Curriculum Learning")
    print("="*70)
    
    try:
        from data_loader import CurriculumDataLoader, DifficultyEstimator, estimate_instance_difficulties
        
        # Create synthetic instances
        instances = []
        for i in range(10):
            instance = {
                'facts': [{'formula': f'f_{j}', 'confidence': 0.9} for j in range(5)],
                'rules': [
                    {
                        'name': f'rule_{j}',
                        'body': [0, 1],
                        'head': [2],
                        'confidence': 0.9
                    }
                    for j in range(3)
                ],
                'goal': 'goal'
            }
            instances.append(instance)
        
        # Add difficulties
        instances = estimate_instance_difficulties(instances)
        print(f"  - Created {len(instances)} instances")
        
        # Create curriculum loader
        loader = CurriculumDataLoader(instances, batch_size=2)
        print("  - Created CurriculumDataLoader")
        
        # Get batch
        batch = loader.get_batch(0, 10)
        print(f"  - Got batch of size {len(batch)}")
        print(f"  - Batch contains ProofState: {hasattr(batch[0], 'proof_state')}")
        
        # Update performance
        loader.update_performance('medium', 0.8)
        print("  - Updated performance")
        
        # Get curriculum info
        info = loader.get_curriculum_info()
        print(f"  - Temperature: {info['temperature']:.4f}")
        
        print("\n✅ Curriculum tests passed!")
        return True
    
    except Exception as e:
        print(f"\n❌ Curriculum test failed: {e}")
        traceback.print_exc()
        return False


def test_model():
    """Test model forward pass."""
    print("\n" + "="*70)
    print("✓ TEST 5: Enhanced Model")
    print("="*70)
    
    try:
        import torch
        from proof_state import ProofState
        from enhanced_model import EnhancedLogNetModel
        
        # Create model
        config = {
            'hidden_dim': 64,
            'num_rules': 20,
            'num_tactics': 5
        }
        model = EnhancedLogNetModel(config)
        model.eval()
        print("  - Created EnhancedLogNetModel")
        
        # Create proof state
        facts = [{'formula': f'f_{i}', 'confidence': 0.9} for i in range(5)]
        rules = [
            {
                'name': f'rule_{i}',
                'body': [0, 1],
                'head': [2],
                'confidence': 0.9
            }
            for i in range(3)
        ]
        state = ProofState(facts, rules, 'goal')
        print("  - Created ProofState")
        
        # Forward pass
        with torch.no_grad():
            output = model(state)
        
        print(f"  - Rule logits shape: {output['rule_logits'].shape}")
        print(f"  - Tactic logits shape: {output['tactic_logits'].shape}")
        print(f"  - Value shape: {output['value'].shape}")
        
        print("\n✅ Model tests passed!")
        return True
    
    except Exception as e:
        print(f"\n❌ Model test failed: {e}")
        traceback.print_exc()
        return False


def test_training_pipeline():
    """Test training pipeline."""
    print("\n" + "="*70)
    print("✓ TEST 6: Training Pipeline")
    print("="*70)
    
    try:
        from main_training import IntegratedProofTrainer
        
        config = {
            'hidden_dim': 64,
            'num_rules': 20,
            'num_tactics': 5,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'batch_size': 2,
            'batches_per_epoch': 2,
            'curriculum_temperature': 2.0,
            'num_simulations': 5,
            'seed': 42
        }
        
        trainer = IntegratedProofTrainer(config)
        print("  - Created IntegratedProofTrainer")
        
        # Generate data
        instances = trainer.generate_synthetic_data(num_instances=5)
        print(f"  - Generated {len(instances)} instances")
        
        # Run 1 epoch
        print("  - Running 1 epoch of training...")
        from data_loader import CurriculumDataLoader
        loader = CurriculumDataLoader(instances, batch_size=2)
        
        train_loss, train_acc = trainer.train_epoch(loader, 0, 1)
        print(f"  - Train loss: {train_loss:.4f}")
        print(f"  - Train accuracy: {train_acc:.4f}")
        
        print("\n✅ Training pipeline tests passed!")
        return True
    
    except Exception as e:
        print(f"\n❌ Training pipeline test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("🧪 COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    tests = [
        ("Imports", test_imports),
        ("ProofState", test_proof_state),
        ("MCTS", test_mcts),
        ("Curriculum", test_curriculum),
        ("Model", test_model),
        ("Training", test_training_pipeline)
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n⚠️  Unexpected error in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name:20s} {status}")
    
    print("="*70)
    print(f"Result: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! Ready to train.")
        return 0
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())