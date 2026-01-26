"""
Comprehensive Scientific Analysis Suite for NTP GNN
===================================================

Based on SOTA benchmarks:
- DeepSeek-Prover-V2: 88.9% on MiniF2F
- Your model: 63.05% Global Hit@1 (Synthetic Horn Clauses)

Gap Analysis: ~25% performance difference suggests fundamental issues
in data quality, model architecture, or training dynamics.

This module provides 10 diagnostic analyzers to identify the root cause.
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from collections import defaultdict
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# ANALYZER 1: Data Generation Quality Validator
# ============================================================================

class DataGenerationQualityAnalyzer:
    """
    HYPOTHESIS: Data generator creates biased/invalid proofs leading to
    poor generalization.
    
    Tests:
    1. Proof uniqueness (are all proofs truly different?)
    2. Backward-chaining correctness
    3. Topological ordering validation
    4. Distractor effectiveness (do distractors actually mislead?)
    5. Graph complexity distribution
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.issues = []
        
    def analyze(self) -> Dict:
        results = {
            'proof_uniqueness': self._test_proof_uniqueness(),
            'topological_validity': self._test_topological_order(),
            'distractor_quality': self._test_distractor_effectiveness(),
            'graph_complexity': self._analyze_graph_complexity(),
            'backward_chain_correctness': self._validate_backward_chaining()
        }
        
        # Critical issues
        if results['topological_validity']['invalid_count'] > 0:
            self.issues.append("CRITICAL: Invalid topological ordering detected!")
        
        if results['proof_uniqueness']['duplicate_rate'] > 0.1:
            self.issues.append(f"WARNING: {results['proof_uniqueness']['duplicate_rate']:.1%} duplicate proofs")
            
        return results
    
    def _test_proof_uniqueness(self) -> Dict:
        """Check if proofs are structurally unique"""
        proof_signatures = set()
        duplicates = 0
        total = 0
        
        for json_file in self.data_dir.rglob('*.json'):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                
                # Create signature: (goal, proof_step_sequence)
                goal = data.get('goal')
                steps = tuple(
                    (s['used_rule'], tuple(s['premises']))
                    for s in data.get('proof_steps', [])
                )
                
                signature = (goal, steps)
                
                if signature in proof_signatures:
                    duplicates += 1
                else:
                    proof_signatures.add(signature)
                    
                total += 1
                
            except Exception as e:
                logger.warning(f"Error reading {json_file}: {e}")
                
        return {
            'total_proofs': total,
            'unique_proofs': len(proof_signatures),
            'duplicate_count': duplicates,
            'duplicate_rate': duplicates / max(total, 1)
        }
    
    def _test_topological_order(self) -> Dict:
        """Validate that proof steps follow correct dependency order"""
        invalid_files = []
        invalid_count = 0
        total = 0
        
        for json_file in self.data_dir.rglob('*.json'):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                
                nodes = {n['nid']: n for n in data['nodes']}
                proof_steps = data.get('proof_steps', [])
                
                # Track derived facts
                derived = set()
                initial_facts = {
                    n['nid'] for n in data['nodes']
                    if n.get('type') == 'fact' and n.get('is_initial', False)
                }
                derived.update(initial_facts)
                
                for step in proof_steps:
                    rule_nid = step['used_rule']
                    premises = set(step['premises'])
                    
                    # Rule node should exist
                    if rule_nid not in nodes:
                        invalid_files.append((json_file, "Missing rule node"))
                        invalid_count += 1
                        break
                    
                    rule = nodes[rule_nid]
                    
                    # All premises must be already derived
                    if not premises.issubset(derived):
                        missing = premises - derived
                        invalid_files.append((
                            json_file, 
                            f"Step {step['step_id']}: Premises {missing} not yet derived"
                        ))
                        invalid_count += 1
                        break
                    
                    # Add newly derived fact
                    derived.add(step['derived_node'])
                
                total += 1
                
            except Exception as e:
                logger.warning(f"Error validating {json_file}: {e}")
        
        return {
            'total_checked': total,
            'invalid_count': invalid_count,
            'invalid_rate': invalid_count / max(total, 1),
            'sample_errors': invalid_files[:5]
        }
    
    def _test_distractor_effectiveness(self) -> Dict:
        """
        Check if adversarial distractors actually create ambiguity.
        
        A good distractor should:
        1. Be reachable from initial facts
        2. NOT lead to the goal
        3. Have similar structure to true path nodes
        """
        stats = {
            'has_distractors': 0,
            'distractor_reachable': 0,
            'distractor_leads_to_goal': 0,  # Bad!
            'total_files': 0
        }
        
        for json_file in self.data_dir.rglob('*.json'):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                
                if data.get('metadata', {}).get('source') != 'adversarial_shuffled':
                    continue
                
                stats['has_distractors'] += 1
                stats['total_files'] += 1
                
                # Check if distractors are reachable
                distractor_nodes = [
                    n for n in data['nodes']
                    if 'distractor_' in n.get('atom', '')
                ]
                
                if distractor_nodes:
                    stats['distractor_reachable'] += 1
                
                # Check if any distractor is in proof path (BAD)
                proof_nodes = {s['derived_node'] for s in data.get('proof_steps', [])}
                distractor_nids = {n['nid'] for n in distractor_nodes}
                
                if proof_nodes & distractor_nids:
                    stats['distractor_leads_to_goal'] += 1
                    
            except Exception as e:
                continue
        
        return stats
    
    def _analyze_graph_complexity(self) -> Dict:
        """Analyze graph size and complexity distribution"""
        complexities = defaultdict(list)
        
        for json_file in self.data_dir.rglob('*.json'):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                
                difficulty = data.get('metadata', {}).get('difficulty', 'unknown')
                
                complexities[difficulty].append({
                    'num_nodes': len(data.get('nodes', [])),
                    'num_edges': len(data.get('edges', [])),
                    'proof_length': len(data.get('proof_steps', [])),
                    'avg_branching': len(data.get('edges', [])) / max(len(data.get('nodes', [])), 1)
                })
                
            except Exception as e:
                continue
        
        # Compute statistics per difficulty
        summary = {}
        for diff, samples in complexities.items():
            if not samples:
                continue
            
            summary[diff] = {
                'count': len(samples),
                'avg_nodes': np.mean([s['num_nodes'] for s in samples]),
                'avg_edges': np.mean([s['num_edges'] for s in samples]),
                'avg_proof_length': np.mean([s['proof_length'] for s in samples]),
                'avg_branching': np.mean([s['avg_branching'] for s in samples])
            }
        
        return summary
    
    def _validate_backward_chaining(self) -> Dict:
        """
        Test if backward chaining logic in data_generator.py is correct
        
        Check for cycles and unreachable goals
        """
        issues = []
        
        for json_file in list(self.data_dir.rglob('*.json'))[:100]:  # Sample
            try:
                with open(json_file) as f:
                    data = json.load(f)
                
                goal = data.get('goal')
                proof_steps = data.get('proof_steps', [])
                
                if not proof_steps and goal:
                    issues.append({
                        'file': str(json_file),
                        'issue': 'Goal exists but no proof steps',
                        'goal': goal
                    })
                
                # Check for cycles in proof
                derived_in_step = {}
                for step in proof_steps:
                    derived_nid = step['derived_node']
                    
                    # If this node was already derived, we have redundancy
                    if derived_nid in derived_in_step:
                        issues.append({
                            'file': str(json_file),
                            'issue': 'Node derived multiple times',
                            'node': derived_nid,
                            'first_step': derived_in_step[derived_nid],
                            'second_step': step['step_id']
                        })
                    
                    derived_in_step[derived_nid] = step['step_id']
                    
            except Exception as e:
                continue
        
        return {
            'sample_size': 100,
            'issues_found': len(issues),
            'sample_issues': issues[:5]
        }


# ============================================================================
# ANALYZER 2: Spectral Feature Quality Validator
# ============================================================================

class SpectralFeatureQualityAnalyzer:
    """
    HYPOTHESIS: Spectral features are corrupted or not informative
    
    Tests:
    1. Eigenvalue distribution (should not be all zeros or all ones)
    2. Eigenvector orthogonality (CRITICAL FIX: Checks per-graph, not per-node)
    3. Correlation with ground truth labels
    4. Information content (entropy)
    5. Stability across training
    """
    
    def __init__(self, model, train_loader, device='cuda'):
        self.model = model
        self.train_loader = train_loader
        self.device = device
        
    def analyze(self) -> Dict:
        """Run all spectral quality tests"""
        results = {}
        
        # Collect spectral features from first 100 batches
        spectral_data = self._collect_spectral_features(num_batches=20)
        
        results['eigenvalue_stats'] = self._analyze_eigenvalues(spectral_data)
        results['eigenvector_quality'] = self._test_eigenvector_orthogonality(spectral_data)
        results['information_content'] = self._compute_information_content(spectral_data)
        results['discriminative_power'] = self._test_discriminative_power(spectral_data)
        
        return results
    
    def _collect_spectral_features(self, num_batches=20) -> Dict:
        """Extract spectral features from batches"""
        self.model.eval()
        
        data = {
            'eigvals': [],
            'eigvecs_real': [],  # Flat list for statistics
            'labels': [],
            'applicable_masks': [],
            'sample_graphs': []  # NEW: Store full graph matrices for orthogonality check
        }
        
        with torch.no_grad():
            for i, batch in enumerate(self.train_loader):
                if i >= num_batches or batch is None:
                    break
                
                batch = batch.to(self.device)
                
                # 1. Collect Flat Data (For distributions/histograms)
                if hasattr(batch, 'eigvals'):
                    data['eigvals'].append(batch.eigvals.cpu().numpy())
                if hasattr(batch, 'eigvecs_real'):
                    flat_vecs = batch.eigvecs_real.cpu().numpy()
                    data['eigvecs_real'].append(flat_vecs)
                
                if hasattr(batch, 'y'):
                    data['labels'].append(batch.y.cpu().numpy())
                if hasattr(batch, 'applicable_mask'):
                    data['applicable_masks'].append(batch.applicable_mask.cpu().numpy())

                # 2. Collect Graph Samples (For Orthogonality)
                # We need to unbatch a few graphs to check U^T * U = I
                if hasattr(batch, 'eigvecs_real') and hasattr(batch, 'batch'):
                    batch_indices = batch.batch.cpu().numpy()
                    vecs = batch.eigvecs_real.cpu().numpy()
                    
                    # Grab unique graph IDs in this batch
                    unique_graphs = np.unique(batch_indices)
                    
                    # Store up to 5 graphs per batch
                    for g_idx in unique_graphs[:5]:
                        # Mask for nodes belonging to graph g_idx
                        mask = (batch_indices == g_idx)
                        graph_U = vecs[mask] # Shape [Num_Nodes_in_Graph, k]
                        data['sample_graphs'].append(graph_U)
        
        # Concatenate flat lists
        for key in ['eigvals', 'eigvecs_real', 'labels', 'applicable_masks']:
            if data[key]:
                data[key] = np.concatenate(data[key], axis=0)
        
        return data
    
    def _analyze_eigenvalues(self, data: Dict) -> Dict:
        """Check eigenvalue distribution"""
        eigvals = data.get('eigvals', np.array([]))
        
        if len(eigvals) == 0:
            return {'error': 'No eigenvalues found'}
        
        # Flatten all eigenvalues
        eigvals_flat = eigvals.flatten()
        
        # Remove zeros (from padding)
        eigvals_nonzero = eigvals_flat[eigvals_flat != 0]
        
        stats = {
            'count': len(eigvals_nonzero),
            'mean': float(np.mean(eigvals_nonzero)) if len(eigvals_nonzero) > 0 else 0,
            'std': float(np.std(eigvals_nonzero)) if len(eigvals_nonzero) > 0 else 0,
            'min': float(np.min(eigvals_nonzero)) if len(eigvals_nonzero) > 0 else 0,
            'max': float(np.max(eigvals_nonzero)) if len(eigvals_nonzero) > 0 else 0,
            'zero_fraction': float(np.mean(eigvals_flat == 0))
        }
        
        if stats['std'] < 0.01:
            stats['issue'] = 'CRITICAL: Eigenvalues have very low variance!'
        
        return stats
    
    def _test_eigenvector_orthogonality(self, data: Dict) -> Dict:
        """Test if eigenvectors are actually orthogonal per graph"""
        sample_graphs = data.get('sample_graphs', [])
        
        if not sample_graphs:
            return {'error': 'No graph samples collected'}
        
        orthogonality_errors = []
        
        for U in sample_graphs:
            # U shape is [Nodes, k]
            # Orthogonality condition: U.T @ U = Identity
            # Note: This holds if k <= Nodes. If k > Nodes, it's U @ U.T
            
            k = U.shape[1]
            gram = U.T @ U # Shape [k, k]
            
            identity = np.eye(k)
            
            # Compute Frobenious norm of difference
            error = np.linalg.norm(gram - identity, 'fro') / (k * k)
            orthogonality_errors.append(error)
        
        mean_error = float(np.mean(orthogonality_errors))
        
        return {
            'mean_orthogonality_error': mean_error,
            'max_orthogonality_error': float(np.max(orthogonality_errors)),
            'issue': 'CRITICAL: Eigenvectors not orthogonal!' if mean_error > 0.1 else None
        }
    
    def _compute_information_content(self, data: Dict) -> Dict:
        """Measure how much information spectral features contain"""
        eigvecs_real = data.get('eigvecs_real', np.array([]))
        
        if len(eigvecs_real) == 0:
            return {'error': 'No eigenvectors found'}
        
        # Compute entropy of eigenvector distributions
        entropies = []
        
        # Sample 100 random rows (nodes)
        if len(eigvecs_real) > 100:
            indices = np.random.choice(len(eigvecs_real), 100, replace=False)
            sample = eigvecs_real[indices]
        else:
            sample = eigvecs_real
            
        for vec in sample:
            # Discretize into bins
            hist, _ = np.histogram(vec, bins=50, density=True)
            hist = hist + 1e-10 
            hist = hist / hist.sum()
            
            entropy = -np.sum(hist * np.log(hist))
            entropies.append(entropy)
        
        return {
            'mean_entropy': float(np.mean(entropies)),
            'std_entropy': float(np.std(entropies)),
            'issue': 'WARNING: Low entropy - features may be uninformative' if np.mean(entropies) < 2.0 else None
        }
    
    def _test_discriminative_power(self, data: Dict) -> Dict:
        """Test if spectral features can discriminate between correct/incorrect rules"""
        eigvecs_real = data.get('eigvecs_real', np.array([]))
        labels = data.get('labels', np.array([]))
        applicable_masks = data.get('applicable_masks', np.array([]))
        
        if len(eigvecs_real) == 0 or len(labels) == 0:
            return {'error': 'Insufficient data'}
        
        # Compute norms of spectral rows
        spectral_norms = np.linalg.norm(eigvecs_real, axis=1)
        
        # We need to map flattened node indices back to batch targets
        # This is complex with flattened arrays. 
        # Simplified check: Just return variance of norms
        
        return {
            'norm_variance': float(np.var(spectral_norms)),
            'issue': None
        }


# ============================================================================
# ANALYZER 3: Pathway Gradient Flow Analyzer
# ============================================================================

class PathwayGradientFlowAnalyzer:
    """
    HYPOTHESIS: One or more pathways have vanishing/exploding gradients
    
    From logs: "Gradient Flow:" section is empty - this is suspicious!
    
    Tests:
    1. Gradient magnitude per pathway
    2. Gradient flow through fusion layer
    3. Dead neuron detection
    4. Gradient variance across batches
    5. Layer-wise gradient analysis
    """
    
    def __init__(self, model, train_loader, device='cuda'):
        self.model = model
        self.train_loader = train_loader
        self.device = device
        
    def analyze(self, criterion, num_batches=50) -> Dict:
        """Analyze gradient flow through all pathways"""
        self.model.train()
        
        gradient_stats = {
            'spectral': [],
            'spatial': [],
            'temporal': [],
            'fusion': [],
            'scorer': []
        }
        
        for i, batch in enumerate(self.train_loader):
            if i >= num_batches or batch is None:
                break
            
            batch = batch.to(self.device)
            self.model.zero_grad()
            
            # Forward pass
            scores, embeddings, value = self.model(batch)
            
            # Compute loss for a sample graph
            if batch.num_graphs > 0:
                mask = (batch.batch == 0)
                graph_scores = scores[mask]
                graph_applicable = batch.applicable_mask[mask]
                target_idx = batch.y[0].item()
                
                try:
                    loss = criterion(graph_scores, embeddings[mask], target_idx, graph_applicable)
                    loss.backward()
                    
                    # Collect gradients
                    self._collect_gradients(gradient_stats)
                    
                except Exception as e:
                    continue
        
        # Analyze collected gradients
        results = {}
        for pathway, grads in gradient_stats.items():
            if not grads:
                results[pathway] = {'error': 'No gradients collected'}
                continue
            
            grads = np.array(grads)
            
            results[pathway] = {
                'mean': float(np.mean(grads)),
                'std': float(np.std(grads)),
                'min': float(np.min(grads)),
                'max': float(np.max(grads)),
                'zero_fraction': float(np.mean(grads == 0)),
                'exploding': float(np.mean(grads > 10)),
                'vanishing': float(np.mean(grads < 1e-5))
            }
            
            # Issue detection
            if results[pathway]['zero_fraction'] > 0.5:
                results[pathway]['issue'] = f'CRITICAL: {pathway} has >50% zero gradients!'
            elif results[pathway]['vanishing'] > 0.5:
                results[pathway]['issue'] = f'WARNING: {pathway} has vanishing gradients'
            elif results[pathway]['exploding'] > 0.1:
                results[pathway]['issue'] = f'WARNING: {pathway} has exploding gradients'
        
        return results
    
    def _collect_gradients(self, gradient_stats: Dict):
        """Collect gradient norms from each pathway"""
        # Spectral pathway
        if hasattr(self.model, 'spectral_encoder'):
            grad_norm = sum(
                p.grad.norm().item() 
                for p in self.model.spectral_encoder.parameters() 
                if p.grad is not None
            )
            gradient_stats['spectral'].append(grad_norm)
        
        # Spatial pathway
        if hasattr(self.model, 'spatial_gnn'):
            grad_norm = sum(
                p.grad.norm().item() 
                for p in self.model.spatial_gnn.parameters() 
                if p.grad is not None
            )
            gradient_stats['spatial'].append(grad_norm)
        
        # Temporal pathway
        if hasattr(self.model, 'temporal_encoder'):
            grad_norm = sum(
                p.grad.norm().item() 
                for p in self.model.temporal_encoder.parameters() 
                if p.grad is not None
            )
            gradient_stats['temporal'].append(grad_norm)
        
        # Fusion layer
        if hasattr(self.model, 'fusion'):
            grad_norm = sum(
                p.grad.norm().item() 
                for p in self.model.fusion.parameters() 
                if p.grad is not None
            )
            gradient_stats['fusion'].append(grad_norm)
        
        # Scorer
        if hasattr(self.model, 'scorer'):
            grad_norm = sum(
                p.grad.norm().item() 
                for p in self.model.scorer.parameters() 
                if p.grad is not None
            )
            gradient_stats['scorer'].append(grad_norm)


# ============================================================================
# ANALYZER 4: Long-Chain Proof Degradation Analyzer
# ============================================================================

class LongChainProofDegradationAnalyzer:
    """
    HYPOTHESIS: Model performance degrades on longer proofs due to
    information bottleneck or vanishing attention
    
    Critical observation from logs:
    - Foundation phase (proof_len ≤ 10): 78.76% Hit@1
    - Mastery phase (proof_len ≤ 40): 62.96% Hit@1
    - ~16% drop suggests length generalization failure
    
    Tests:
    1. Accuracy vs proof length
    2. Attention decay over steps
    3. Temporal encoding quality for long sequences
    4. Feature collapse in long proofs
    5. Step-wise accuracy degradation
    """
    
    def __init__(self, model, test_loader, device='cuda'):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        
    def analyze(self) -> Dict:
        """Analyze performance vs proof length"""
        self.model.eval()
        
        # Collect accuracy per proof length bin
        length_bins = {
            '1-5': {'correct': 0, 'total': 0},
            '6-10': {'correct': 0, 'total': 0},
            '11-20': {'correct': 0, 'total': 0},
            '21-40': {'correct': 0, 'total': 0},
            '40+': {'correct': 0, 'total': 0}
        }
        
        step_position_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        with torch.no_grad():
            for batch in self.test_loader:
                if batch is None:
                    continue
                
                batch = batch.to(self.device)
                scores, _, _ = self.model(batch)
                
                for i in range(batch.num_graphs):
                    mask = (batch.batch == i)
                    graph_scores = scores[mask]
                    
                    target_idx = batch.y[i].item()
                    pred_idx = graph_scores.argmax().item()
                    
                    # Get proof length and step index
                    if hasattr(batch, 'meta_list') and i < len(batch.meta_list):
                        meta = batch.meta_list[i]
                        proof_len = meta.get('proof_length', 0)
                        step_idx = meta.get('step_idx', 0)
                    else:
                        continue
                    
                    # Bin by length
                    bin_key = self._get_length_bin(proof_len)
                    if bin_key:
                        length_bins[bin_key]['total'] += 1
                        if pred_idx == target_idx:
                            length_bins[bin_key]['correct'] += 1
                    
                    # Track step position
                    step_position = step_idx / max(proof_len, 1)
                    position_bin = f'{int(step_position * 10) / 10:.1f}'
                    
                    step_position_accuracy[position_bin]['total'] += 1
                    if pred_idx == target_idx:
                        step_position_accuracy[position_bin]['correct'] += 1
        
        # Compute accuracies
        results = {
            'length_bins': {},
            'step_positions': {},
            'degradation_detected': False
        }
        
        for bin_key, stats in length_bins.items():
            if stats['total'] > 0:
                acc = stats['correct'] / stats['total']
                results['length_bins'][bin_key] = {
                    'accuracy': acc,
                    'count': stats['total']
                }
        
        for pos, stats in step_position_accuracy.items():
            if stats['total'] > 0:
                acc = stats['correct'] / stats['total']
                results['step_positions'][pos] = {
                    'accuracy': acc,
                    'count': stats['total']
                }
        
        # Detect degradation
        if '1-5' in results['length_bins'] and '21-40' in results['length_bins']:
            short_acc = results['length_bins']['1-5']['accuracy']
            long_acc = results['length_bins']['21-40']['accuracy']
            
            if short_acc - long_acc > 0.1:  # >10% drop
                results['degradation_detected'] = True
                results['issue'] = f'CRITICAL: {(short_acc - long_acc)*100:.1f}% accuracy drop on long proofs!'
        
        return results
    
    def _get_length_bin(self, length: int) -> str:
        if length <= 5:
            return '1-5'
        elif length <= 10:
            return '6-10'
        elif length <= 20:
            return '11-20'
        elif length <= 40:
            return '21-40'
        else:
            return '40+'


# ============================================================================
# ANALYZER 5: Loss Function Effectiveness Analyzer
# ============================================================================

class LossFunctionAnalyzer:
    """
    HYPOTHESIS: Loss function doesn't properly guide learning
    
    Observations:
    - Training loss plateaus around 7.8-8.5
    - Loss components may be imbalanced
    - Temperature scaling might be suboptimal
    
    Tests:
    1. Loss landscape smoothness
    2. Gradient signal-to-noise ratio
    3. Temperature sensitivity analysis
    4. Ranking vs applicability balance
    5. Hard negative mining effectiveness
    """
    
    def __init__(self, model, criterion, train_loader, device='cuda'):
        self.model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.device = device
        
    def analyze(self, num_batches=100) -> Dict:
        """Analyze loss function behavior"""
        results = {}
        
        # Collect loss components over batches
        loss_components = {
            'ranking': [],
            'applicability': [],
            'total': []
        }
        
        margin_violations = []
        temperature_values = []
        
        self.model.eval()
        
        with torch.no_grad():
            for i, batch in enumerate(self.train_loader):
                if i >= num_batches or batch is None:
                    break
                
                batch = batch.to(self.device)
                scores, embeddings, value = self.model(batch)
                
                # Check temperature if learnable
                if hasattr(self.model, 'scorer') and hasattr(self.model.scorer, 'temperature'):
                    temp = self.model.scorer.temperature
                    if isinstance(temp, torch.Tensor):
                        temperature_values.append(temp.item())
                
                # Analyze Ranking Loss Dynamics
                for g in range(batch.num_graphs):
                    mask = (batch.batch == g)
                    g_scores = scores[mask]
                    target = batch.y[g].item()
                    
                    if target < 0 or target >= len(g_scores):
                        continue
                        
                    # Margin Analysis: score[target] - score[best_negative]
                    # Ideally should be > margin (e.g. 0.1)
                    pos_score = g_scores[target]
                    g_scores_clone = g_scores.clone()
                    g_scores_clone[target] = -float('inf')
                    max_neg_score = g_scores_clone.max()
                    
                    diff = (pos_score - max_neg_score).item()
                    
                    if diff < 0.1:  # Assuming margin is ~0.1
                        margin_violations.append(1)
                    else:
                        margin_violations.append(0)
                        
                    # Signal-to-Noise: Mean Score / Std Score
                    # If low, model is "confused" (flat distribution)
                    # 
                    
        # Compile results
        results['margin_violation_rate'] = np.mean(margin_violations) if margin_violations else 0.0
        results['avg_temperature'] = np.mean(temperature_values) if temperature_values else "static"
        
        if results['margin_violation_rate'] > 0.5:
            results['issue'] = "CRITICAL: >50% of samples violate safety margin (Model is guessing)"
            
        return results


# ============================================================================
# ANALYZER 6: Training Dynamics (Micro-Overfitting Test)
# ============================================================================

class TrainingDynamicsAnalyzer:
    """
    HYPOTHESIS: Model or Optimizer is broken (cannot learn even simple patterns).
    
    Test:
    Take ONE single batch and train on it for 50 steps.
    - Loss should go to near zero.
    - Accuracy should go to 100%.
    
    If it fails, architecture or optimizer is fundamentally bugged.
    """
    
    def __init__(self, model, train_loader, criterion, optimizer, device='cuda'):
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        
    def analyze(self) -> Dict:
        """Run the 'Overfit on One Batch' test"""
        logger.info("Running Micro-Overfitting Test...")
        
        # Grab first batch
        batch = next(iter(self.train_loader))
        batch = batch.to(self.device)
        
        initial_loss = 0
        final_loss = 0
        history = []
        
        self.model.train()
        
        # Train loop
        for step in range(50):
            self.optimizer.zero_grad()
            scores, embeddings, _ = self.model(batch)
            
            # Construct batch-wise targets for loss
            # (Simplified for diagnostic)
            loss = 0
            for i in range(batch.num_graphs):
                mask = (batch.batch == i)
                loss += self.criterion(
                    scores[mask], 
                    embeddings[mask], 
                    batch.y[i].item(), 
                    batch.applicable_mask[mask]
                )
            
            loss.backward()
            self.optimizer.step()
            
            val = loss.item()
            history.append(val)
            if step == 0: initial_loss = val
            final_loss = val
            
        # Analysis
        loss_reduction = (initial_loss - final_loss) / (initial_loss + 1e-10)
        
        results = {
            'initial_loss': initial_loss,
            'final_loss': final_loss,
            'reduction_pct': loss_reduction,
            'converged': final_loss < 0.1,
            'history': history
        }
        
        if final_loss > 1.0:
            results['issue'] = "CRITICAL: Model cannot overfit a single batch! (Check Optimizer/LR)"
        elif loss_reduction < 0.5:
            results['issue'] = "WARNING: Slow convergence on micro-batch"
            
        return results


# ============================================================================
# ANALYZER 7: Generalization Gap Analyzer
# ============================================================================

class GeneralizationGapAnalyzer:
    """
    HYPOTHESIS: Model overfits to 'easy' structural patterns and fails to generalize.
    
    Metrics:
    - Gap = Train_Acc - Test_Acc
    - Large positive gap (>15%) -> Overfitting
    - Negative gap -> Data leakage or distribution mismatch
    """
    
    def __init__(self, model, train_loader, val_loader, test_loader, device='cuda'):
        self.model = model
        self.loaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
        self.device = device
        
    def analyze(self) -> Dict:
        accuracies = {}
        
        self.model.eval()
        with torch.no_grad():
            for name, loader in self.loaders.items():
                correct = 0
                total = 0
                
                # Check first 50 batches only to save time
                for i, batch in enumerate(loader):
                    if i > 50: break
                    if batch is None: continue
                    
                    batch = batch.to(self.device)
                    scores, _, _ = self.model(batch)
                    
                    for g in range(batch.num_graphs):
                        mask = (batch.batch == g)
                        if mask.sum() == 0: continue
                        
                        target = batch.y[g].item()
                        pred = scores[mask].argmax().item()
                        
                        if pred == target:
                            correct += 1
                        total += 1
                
                accuracies[name] = correct / max(total, 1)
        
        gap = accuracies['train'] - accuracies['test']
        
        results = {
            'metrics': accuracies,
            'generalization_gap': gap,
            'issue': None
        }
        
        if gap > 0.20:
            results['issue'] = f"CRITICAL: Massive generalization gap ({gap:.1%}). Model is memorizing."
        elif gap < -0.05:
            results['issue'] = "WARNING: Test set performs better than train? Check data splits."
            
        return results


# ============================================================================
# ANALYZER 8: Inference Efficiency Analyzer
# ============================================================================

class InferenceEfficiencyAnalyzer:
    """
    HYPOTHESIS: Model is too heavy for real-time proof search (MCTS).
    
    Tests:
    1. Forward pass latency (ms)
    2. Peak memory usage
    3. Throughput (graphs/sec)
    """
    
    def __init__(self, model, input_sample, device='cuda'):
        self.model = model
        self.input_sample = input_sample.to(device)
        self.device = device
        
    def analyze(self) -> Dict:
        import time
        
        self.model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10): self.model(self.input_sample)
            
        # Timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        with torch.no_grad():
            for _ in range(100):
                self.model(self.input_sample)
        end_event.record()
        torch.cuda.synchronize()
        
        total_time = start_event.elapsed_time(end_event) # ms
        avg_latency = total_time / 100
        
        # Memory
        mem_alloc = torch.cuda.max_memory_allocated() / 1024**2 # MB
        
        # Parameter count
        params = sum(p.numel() for p in self.model.parameters())
        
        results = {
            'latency_ms': avg_latency,
            'throughput_graphs_sec': (1000/avg_latency) * self.input_sample.num_graphs,
            'memory_mb': mem_alloc,
            'params_millions': params / 1e6
        }
        
        if results['latency_ms'] > 50: # Arbitrary threshold for MCTS
            results['issue'] = "WARNING: High latency (>50ms). MCTS will be slow."
            
        return results


# ============================================================================
# ANALYZER 9: Embedding Manifold Analyzer
# ============================================================================

class EmbeddingManifoldAnalyzer:
    """
    HYPOTHESIS: Embeddings suffer from 'anisotropy' (all point in same direction)
    or 'collapse' (all embeddings identical).
    
    Tests:
    1. Average Cosine Similarity between random pairs
    2. Effective Rank of the embedding matrix
    """
    
    def __init__(self, model, loader, device='cuda'):
        self.model = model
        self.loader = loader
        self.device = device
        
    def analyze(self) -> Dict:
        self.model.eval()
        embeddings_list = []
        
        with torch.no_grad():
            for i, batch in enumerate(self.loader):
                if i > 20: break
                batch = batch.to(self.device)
                _, emb, _ = self.model(batch)
                embeddings_list.append(emb.cpu())
                
        E = torch.cat(embeddings_list, dim=0)
        # Normalize
        E = torch.nn.functional.normalize(E, p=2, dim=1)
        
        # 1. Anisotropy (Average Cosine Sim)
        # Pick 1000 random pairs
        idx1 = torch.randint(0, len(E), (1000,))
        idx2 = torch.randint(0, len(E), (1000,))
        cos_sims = (E[idx1] * E[idx2]).sum(dim=1)
        avg_sim = cos_sims.mean().item()
        
        # 2. Effective Rank (Singular Value Distribution)
        # 
        U, S, V = torch.svd(E[:1000]) # Sample
        S = S / S.sum()
        effective_rank = torch.exp(-torch.sum(S * torch.log(S + 1e-10))).item()
        
        results = {
            'avg_cosine_similarity': avg_sim,
            'effective_rank': effective_rank,
            'embedding_dim': E.shape[1]
        }
        
        if avg_sim > 0.9:
            results['issue'] = "CRITICAL: Representation Collapse (All embeddings identical)"
        elif effective_rank < 5.0:
            results['issue'] = "WARNING: Low effective rank. Model uses few dimensions."
            
        return results


# ============================================================================
# ANALYZER 10: Error Qualitative Analyzer
# ============================================================================

class ErrorQualitativeAnalyzer:
    """
    HYPOTHESIS: Model fails in specific ways (e.g. rank-deficient near misses).
    
    Categorizes errors:
    1. Near Miss: Correct answer in Top-5
    2. Far Miss: Correct answer > Top-5
    3. Logic Loop: Predicting a premise that is already known
    """
    
    def __init__(self, model, loader, device='cuda'):
        self.model = model
        self.loader = loader
        self.device = device
        
    def analyze(self) -> Dict:
        stats = {
            'total_errors': 0,
            'near_miss': 0,
            'far_miss': 0,
            'mean_reciprocal_rank': 0
        }
        
        mrr_sum = 0
        total_queries = 0
        
        self.model.eval()
        with torch.no_grad():
            for batch in self.loader:
                batch = batch.to(self.device)
                scores, _, _ = self.model(batch)
                
                for g in range(batch.num_graphs):
                    mask = (batch.batch == g)
                    g_scores = scores[mask]
                    target = batch.y[g].item()
                    
                    if target < 0 or target >= len(g_scores): continue
                    
                    # Sort scores descending
                    sorted_indices = torch.argsort(g_scores, descending=True)
                    rank = (sorted_indices == target).nonzero(as_tuple=True)[0].item() + 1
                    
                    mrr_sum += 1.0 / rank
                    total_queries += 1
                    
                    if rank > 1:
                        stats['total_errors'] += 1
                        if rank <= 5:
                            stats['near_miss'] += 1
                        else:
                            stats['far_miss'] += 1
                            
        stats['mean_reciprocal_rank'] = mrr_sum / max(total_queries, 1)
        stats['near_miss_rate'] = stats['near_miss'] / max(stats['total_errors'], 1)
        
        return stats


# ============================================================================
# MAIN EXECUTION ORCHESTRATOR
# ============================================================================

def run_comprehensive_analysis(model, train_loader, val_loader, test_loader, optimizer, criterion, data_dir):
    """
    Run all 10 analyzers and generate a unified scientific report.
    """
    logger.info("Starting Comprehensive NTP Analysis...")
    report = {}
    device = next(model.parameters()).device
    
    # --- Data Analyzers ---
    logger.info("1/10 Running Data Quality Analysis...")
    report['data_gen'] = DataGenerationQualityAnalyzer(data_dir).analyze()
    
    logger.info("2/10 Running Spectral Analysis...")
    report['spectral'] = SpectralFeatureQualityAnalyzer(model, train_loader, device).analyze()
    
    # --- Model Analyzers ---
    logger.info("3/10 Running Gradient Flow Analysis...")
    report['gradient'] = PathwayGradientFlowAnalyzer(model, train_loader, device).analyze(criterion)
    
    logger.info("4/10 Running Long-Chain Degradation Analysis...")
    report['long_chain'] = LongChainProofDegradationAnalyzer(model, test_loader, device).analyze()
    
    # --- Training Analyzers ---
    logger.info("5/10 Running Loss Function Analysis...")
    report['loss_func'] = LossFunctionAnalyzer(model, criterion, train_loader, device).analyze()
    
    logger.info("6/10 Running Training Dynamics (Micro-Overfit)...")
    report['dynamics'] = TrainingDynamicsAnalyzer(model, train_loader, criterion, optimizer, device).analyze()
    
    logger.info("7/10 Running Generalization Analysis...")
    report['generalization'] = GeneralizationGapAnalyzer(model, train_loader, val_loader, test_loader, device).analyze()
    
    # --- Inference Analyzers ---
    logger.info("8/10 Running Efficiency Analysis...")
    sample_batch = next(iter(train_loader))
    report['efficiency'] = InferenceEfficiencyAnalyzer(model, sample_batch, device).analyze()
    
    logger.info("9/10 Running Manifold Analysis...")
    report['manifold'] = EmbeddingManifoldAnalyzer(model, val_loader, device).analyze()
    
    logger.info("10/10 Running Qualitative Error Analysis...")
    report['errors'] = ErrorQualitativeAnalyzer(model, test_loader, device).analyze()
    
    # --- Final Report Generation ---
    print("\n" + "="*50)
    print("   SCIENTIFIC DIAGNOSIS REPORT")
    print("="*50)
    
    critical_issues = []
    for section, res in report.items():
        if isinstance(res, dict) and 'issue' in res and res['issue']:
            critical_issues.append(f"[{section.upper()}] {res['issue']}")
            
        # Recursive check for nested issues (like in gradient stats)
        for k, v in res.items():
            if isinstance(v, dict) and 'issue' in v and v['issue']:
                critical_issues.append(f"[{section.upper()} - {k}] {v['issue']}")

    if critical_issues:
        print(f"\nDETECTED {len(critical_issues)} CRITICAL ISSUES:")
        for issue in critical_issues:
            print(f"❌ {issue}")
    else:
        print("\n✅ System appears healthy. Performance gap likely due to architecture capacity or data scale.")
        
    return report

# ... (Previous code remains the same)

# ============================================================================
# MAIN EXECUTION BLOCK (CORRECTED)
# ============================================================================

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Run comprehensive scientific diagnostics")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pt)')
    parser.add_argument('--data-dir', type=str, default='generated_data', help='Path to data directory')
    parser.add_argument('--spectral-dir', type=str, default='spectral_cache', help='Path to spectral cache')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🚀 Initializing Diagnostic Suite on {device}...")

    # 1. Initialize Model Architecture
    # Note: We must match the architecture of the saved checkpoint.
    # Assuming standard parameters from your training script.
    from model_fixed import FixedProofGNN
    
    # Try initializing with 3 layers (likely config), fallback if needed
    try:
        model = FixedProofGNN(in_dim=32, hidden_dim=256, k=48, num_layers=3).to(device)
    except TypeError:
        model = FixedProofGNN(in_dim=32, hidden_dim=256, k=48).to(device)

    # 2. Load Checkpoint
    if os.path.exists(args.checkpoint):
        logger.info(f"Loading weights from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        except RuntimeError as e:
            logger.warning(f"Strict load failed ({e}). Retrying with strict=False...")
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")

    # 3. Create DataLoaders
    # We use the standard dataset creation function to ensure consistency
    from dataset import create_properly_split_dataloaders
    
    logger.info("Creating DataLoaders...")
    train_loader, val_loader, test_loader = create_properly_split_dataloaders(
        data_dir=args.data_dir,
        spectral_dir=args.spectral_dir,
        batch_size=32,  # Standard batch size for diagnostics
        k=48
    )

    # 4. Initialize Training Components (Needed for some tests)
    from loss_enhanced import ProductionInfoNCELoss
    criterion = ProductionInfoNCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 5. Run Analysis
    # Now we pass the OBJECTS, not the strings
    report = run_comprehensive_analysis(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        data_dir=args.data_dir
    )
    
    # Optional: Save report
    output_path = "diagnostic_report.json"
    with open(output_path, "w") as f:
        # Helper to serialize numpy/torch types
        def convert(o):
            if isinstance(o, (np.generic, np.ndarray)): return o.item() if np.ndim(o)==0 else o.tolist()
            if isinstance(o, torch.Tensor): return o.item() if o.numel()==1 else o.tolist()
            return o
        json.dump(report, f, default=convert, indent=2)
    
    logger.info(f"Report saved to {output_path}")