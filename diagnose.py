import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List
from collections import defaultdict

# --- Mock Imports ---
# These are placeholders for your actual file imports
# In a real run, you would import from dataset.py
try:
    from dataset import ProofStepDataset, create_split
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    logger.info("Successfully imported ProofStepDataset.")
except ImportError:
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    logger.error("Could not import ProofStepDataset. Using mock data.")
    
    # Create mock classes if import fails
    class MockData:
        def __init__(self, x, y, applicable_mask):
            self.x = x
            self.y = y
            self.applicable_mask = applicable_mask
        def __getitem__(self, key):
             # Mock for x[target_idx]
            if key == 'y': return self.y
            if key == 'applicable_mask': return self.applicable_mask
            return self.x
            
    class ProofStepDataset:
        def __init__(self, *args, **kwargs):
            self.mock_db = self._generate_mock_data()
        
        def _generate_mock_data(self):
            db = []
            # 1. Easy case: features are very different
            db.append(MockData(
                x=torch.tensor([
                    [1.0, 0.0, 0.0], # Target
                    [0.0, 1.0, 0.0], # App
                    [0.0, 0.0, 1.0], # Inapp
                ]),
                y=torch.tensor([0]),
                applicable_mask=torch.tensor([True, True, False])
            ))
            # 2. Hard case: applicable features are similar
            db.append(MockData(
                x=torch.tensor([
                    [0.9, 0.1, 0.5], # Target
                    [0.8, 0.2, 0.6], # App (similar)
                    [0.0, 0.0, 1.0], # Inapp
                ]),
                y=torch.tensor([0]),
                applicable_mask=torch.tensor([True, True, False])
            ))
            return db

        def __len__(self):
            return len(self.mock_db)

        def __getitem__(self, idx):
            return self.mock_db[idx]

    def create_split(*args, **kwargs):
        return [], [], [] # Return empty lists
# --- End Mock Imports ---


class FeatureDiagnostic:
    """
    Tests the quality of node features in a ProofStepDataset.
    
    The key question: Are the features good enough to distinguish
    the *correct* applicable rule from *other* applicable rules?
    """

    def __init__(self, dataset: ProofStepDataset):
        self.dataset = dataset
        if not hasattr(self, 'dataset') or len(self.dataset) == 0:
             logger.warning("Dataset is empty or failed to load. Using mock data.")
             self.dataset = ProofStepDataset() # Fallback to mock

        self.results = defaultdict(list)

    def run_diagnostics(self, num_samples=100):
        logger.info(f"Running feature diagnostics on {min(num_samples, len(self.dataset))} samples...")
        
        for i in range(min(num_samples, len(self.dataset))):
            try:
                data = self.dataset[i]
                if data is None:
                    continue

                target_idx = data.y.item()
                applicable_mask = data.applicable_mask
                x = data.x

                if not applicable_mask[target_idx]:
                    # This is a data-loading error, skip
                    continue
                
                applicable_indices = applicable_mask.nonzero(as_tuple=True)[0]
                
                # 1. Calculate the 'applicable_random' baseline
                num_applicable = len(applicable_indices)
                if num_applicable > 0:
                    self.results['baseline_applicable_random'].append(1.0 / num_applicable)

                # 2. Test feature uniqueness
                if num_applicable > 1:
                    self._test_feature_similarity(x, target_idx, applicable_indices)

            except Exception as e:
                logger.error(f"Failed on sample {i}: {e}")
                continue
        
        self.print_report()

    def _test_feature_similarity(self, x: torch.Tensor, target_idx: int, applicable_indices: torch.Tensor):
        """
        Checks if the target's features are distinct from other *applicable* rules.
        """
        target_features = x[target_idx].unsqueeze(0) # [1, D]
        
        # Get other applicable indices
        other_applicable_mask = (applicable_indices != target_idx)
        other_indices = applicable_indices[other_applicable_mask]

        if len(other_indices) == 0:
            # Target was the only applicable rule
            self.results['target_was_unique'].append(1.0)
            return

        other_features = x[other_indices] # [K, D]
        
        # Cosine similarity between target and all other applicable rules
        similarities = F.cosine_similarity(target_features, other_features) # [K]
        
        # Find the most similar competitor
        max_similarity = similarities.max().item()
        self.results['max_similarity_to_competitor'].append(max_similarity)
        
        # Did a competitor have *identical* features?
        if max_similarity > 0.999:
            self.results['feature_collisions'].append(1.0)


    def print_report(self):
        logger.info("\n" + "="*80)
        logger.info("🔬 FEATURE DIAGNOSTIC REPORT")
        logger.info("="*80)

        if not self.results:
            logger.error("No valid samples were processed. Cannot generate report.")
            return

        # --- Baseline Report ---
        baseline_acc = np.mean(self.results['baseline_applicable_random']) * 100
        logger.info(f"📊 Baseline 'Applicable Random' Accuracy: {baseline_acc:.2f}%")
        logger.info(f"   (This is the score to beat. It means on average there are ~{1/ (baseline_acc / 100):.2f} applicable rules to choose from.)")

        # --- Feature Quality Report ---
        if 'max_similarity_to_competitor' in self.results:
            avg_max_sim = np.mean(self.results['max_similarity_to_competitor'])
            collisions = np.sum(self.results.get('feature_collisions', [0]))
            total_comparisons = len(self.results['max_similarity_to_competitor'])

            logger.info(f"\n📊 Feature Expressiveness (Cosine Similarity):")
            logger.info(f"   Avg. Max Similarity to Competitor: {avg_max_sim:.4f}")
            logger.info(f"   Feature Collisions (Similarity > 0.999): {collisions} / {total_comparisons} ({collisions/total_comparisons:.1%})")

            logger.info("\n--- ANALYSIS ---")
            if avg_max_sim > 0.9:
                logger.warning("🔥 WARNING: Features are NOT expressive.")
                logger.warning("  The target rule's features are, on average, >90% similar to its closest competitor.")
            elif avg_max_sim > 0.7:
                 logger.warning("  CAUTION: Features are only moderately expressive.")
            else:
                 logger.info("  ✅ INFO: Features appear to be highly expressive (avg. similarity < 0.7).")

            if collisions / total_comparisons > 0.05:
                 logger.error("  ❌ CRITICAL: More than 5% of samples have feature collisions.")
                 logger.error("     This proves your features are not SOTA and cannot distinguish the target.")
            
            logger.info("\nRECOMMENDATION: See 'sota_dataset_design.md' to engineer new features.")
        
        logger.info("="*80)

def main():
    # --- This will use your REAL dataset ---
    try:
        from dataset import create_split, ProofStepDataset
        
        # 1. Load your data files
        train_files, _, _ = create_split(
            data_dir='generated_data', 
            train_ratio=0.7, 
            val_ratio=0.15, 
            seed=42
        )
        
        # 2. Initialize your real dataset
        dataset = ProofStepDataset(
            json_files=train_files,
            spectral_dir='spectral_cache', # Not actually used, but good to pass
            seed=42
        )
    except Exception as e:
        logger.critical(f"Failed to load real dataset: {e}. Falling back to mock data.")
        dataset = ProofStepDataset(json_files=[]) # Init mock data

    # 3. Run diagnostics
    diagnostic = FeatureDiagnostic(dataset)
    diagnostic.run_diagnostics(num_samples=500)

if __name__ == "__main__":
    main()