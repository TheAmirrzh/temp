"""
Semantic Hard Negative Mining
Implements Issue #10: Semantic hard negative mining based on structural similarity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from collections import defaultdict
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re


class RuleEmbedder:
    """Embeds rules by their structural properties."""
    
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.rule_encoder = nn.Sequential(
            nn.Linear(10, embedding_dim),  # 10 structural features
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def extract_structural_features(self, rule: Dict) -> torch.Tensor:
        """Extract structural features from a rule."""
        features = torch.zeros(10)
        
        # Feature 1: Number of premises
        num_premises = len(rule.get('body', []))
        features[0] = min(num_premises / 10.0, 1.0)  # Normalize to [0,1]
        
        # Feature 2: Number of conclusions
        num_conclusions = len(rule.get('head', []))
        features[1] = min(num_conclusions / 5.0, 1.0)
        
        # Feature 3: Rule complexity (total atoms)
        total_atoms = num_premises + num_conclusions
        features[2] = min(total_atoms / 20.0, 1.0)
        
        # Feature 4: Has conjunction in premises
        has_conjunction = any('∧' in str(atom) for atom in rule.get('body', []))
        features[3] = 1.0 if has_conjunction else 0.0
        
        # Feature 5: Has disjunction in premises
        has_disjunction = any('∨' in str(atom) for atom in rule.get('body', []))
        features[4] = 1.0 if has_disjunction else 0.0
        
        # Feature 6: Has negation
        has_negation = any('¬' in str(atom) for atom in rule.get('body', []) + rule.get('head', []))
        features[5] = 1.0 if has_negation else 0.0
        
        # Feature 7: Has implication
        has_implication = any('→' in str(atom) for atom in rule.get('body', []) + rule.get('head', []))
        features[6] = 1.0 if has_implication else 0.0
        
        # Feature 8: Has universal quantifier
        has_universal = any('∀' in str(atom) for atom in rule.get('body', []) + rule.get('head', []))
        features[7] = 1.0 if has_universal else 0.0
        
        # Feature 9: Has existential quantifier
        has_existential = any('∃' in str(atom) for atom in rule.get('body', []) + rule.get('head', []))
        features[8] = 1.0 if has_existential else 0.0
        
        # Feature 10: Rule type (deduction, induction, etc.)
        rule_type = self._classify_rule_type(rule)
        features[9] = rule_type
        
        return features
    
    def _classify_rule_type(self, rule: Dict) -> float:
        """Classify rule into types (0-1 scale)."""
        body = rule.get('body', [])
        head = rule.get('head', [])
        
        # Type 0: Simple deduction (A → B)
        if len(body) == 1 and len(head) == 1:
            return 0.0
        
        # Type 0.25: Conjunction elimination (A ∧ B → A)
        if len(body) == 1 and '∧' in str(body[0]):
            return 0.25
        
        # Type 0.5: Modus ponens (A, A→B → B)
        if len(body) == 2 and any('→' in str(atom) for atom in body):
            return 0.5
        
        # Type 0.75: Complex inference
        if len(body) > 2:
            return 0.75
        
        # Type 1.0: Induction or complex proof
        if any('∀' in str(atom) for atom in body + head):
            return 1.0
        
        return 0.5  # Default
    
    def embed_rule(self, rule: Dict) -> torch.Tensor:
        """Get embedding for a rule."""
        features = self.extract_structural_features(rule)
        return self.rule_encoder(features)
    
    def embed_rules(self, rules: List[Dict]) -> torch.Tensor:
        """Get embeddings for multiple rules."""
        embeddings = []
        for rule in rules:
            emb = self.embed_rule(rule)
            embeddings.append(emb)
        return torch.stack(embeddings)


class SemanticSimilarityComputer:
    """Computes semantic similarity between rules."""
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=None,
            ngram_range=(1, 3)
        )
        self.rule_embeddings = None
        self.rule_texts = []
    
    def preprocess_rule_text(self, rule: Dict) -> str:
        """Convert rule to text representation."""
        body_text = ' '.join(str(atom) for atom in rule.get('body', []))
        head_text = ' '.join(str(atom) for atom in rule.get('head', []))
        
        # Combine body and head
        rule_text = f"{body_text} → {head_text}"
        
        # Clean and normalize
        rule_text = re.sub(r'[^\w\s∧∨¬→∀∃]', ' ', rule_text)
        rule_text = re.sub(r'\s+', ' ', rule_text).strip()
        
        return rule_text
    
    def fit_rules(self, rules: List[Dict]):
        """Fit TF-IDF vectorizer on rules."""
        self.rule_texts = [self.preprocess_rule_text(rule) for rule in rules]
        self.rule_embeddings = self.tfidf_vectorizer.fit_transform(self.rule_texts)
    
    def compute_similarities(self, rule_idx: int) -> np.ndarray:
        """Compute similarities between rule and all others."""
        if self.rule_embeddings is None:
            raise ValueError("Must fit rules first")
        
        # Get similarity scores
        similarities = cosine_similarity(
            self.rule_embeddings[rule_idx:rule_idx+1],
            self.rule_embeddings
        )[0]
        
        return similarities


class HardNegativeMiner:
    """
    Mines semantically hard negatives following SOTA approach.
    
    Key insight: Hard negatives should be structurally similar to correct rule
    but semantically different (wrong conclusion, different premises, etc.)
    """
    
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.rule_embedder = RuleEmbedder(embedding_dim)
        self.similarity_computer = SemanticSimilarityComputer()
        self.rules = []
        self.rule_embeddings = None
    
    def add_rules(self, rules: List[Dict]):
        """Add rules to the miner."""
        self.rules = rules
        self.rule_embeddings = self.rule_embedder.embed_rules(rules)
        self.similarity_computer.fit_rules(rules)
    
    def get_hard_negatives(self, correct_rule_idx: int, 
                          num_hard_negatives: int = 5,
                          similarity_threshold: float = 0.7) -> List[int]:
        """
        Get hard negatives for a correct rule.
        
        Args:
            correct_rule_idx: Index of correct rule
            num_hard_negatives: Number of hard negatives to return
            similarity_threshold: Minimum similarity to consider as hard negative
        
        Returns:
            List of rule indices that are hard negatives
        """
        if self.rule_embeddings is None:
            raise ValueError("Must add rules first")
        
        # Get structural similarities
        structural_similarities = F.cosine_similarity(
            self.rule_embeddings[correct_rule_idx:correct_rule_idx+1],
            self.rule_embeddings
        )[0]
        
        # Get semantic similarities
        semantic_similarities = self.similarity_computer.compute_similarities(correct_rule_idx)
        
        # Combine similarities (weighted average)
        combined_similarities = 0.6 * structural_similarities + 0.4 * torch.tensor(semantic_similarities)
        
        # Find hard negatives: high similarity but different rules
        hard_negative_mask = (
            (combined_similarities > similarity_threshold) &
            (torch.arange(len(self.rules)) != correct_rule_idx)
        )
        
        hard_negative_indices = torch.where(hard_negative_mask)[0]
        
        # Sort by similarity (highest first)
        if len(hard_negative_indices) > 0:
            similarities = combined_similarities[hard_negative_indices]
            sorted_indices = torch.argsort(similarities, descending=True)
            hard_negative_indices = hard_negative_indices[sorted_indices]
        
        # Return top K
        return hard_negative_indices[:num_hard_negatives].tolist()
    
    def get_semantic_hard_negatives(self, correct_rule_idx: int,
                                   num_hard_negatives: int = 5) -> List[int]:
        """
        Get semantically hard negatives using advanced similarity measures.
        
        These are rules that:
        1. Have similar structure to correct rule
        2. But lead to different conclusions
        3. Or have different logical patterns
        """
        if self.rule_embeddings is None:
            raise ValueError("Must add rules first")
        
        correct_rule = self.rules[correct_rule_idx]
        hard_negatives = []
        
        for i, rule in enumerate(self.rules):
            if i == correct_rule_idx:
                continue
            
            # Check structural similarity
            structural_sim = self._compute_structural_similarity(correct_rule, rule)
            
            # Check semantic difference
            semantic_diff = self._compute_semantic_difference(correct_rule, rule)
            
            # Hard negative if structurally similar but semantically different
            if structural_sim > 0.6 and semantic_diff > 0.3:
                hard_negatives.append(i)
        
        # Sort by hardness (structural similarity * semantic difference)
        hardness_scores = []
        for idx in hard_negatives:
            rule = self.rules[idx]
            structural_sim = self._compute_structural_similarity(correct_rule, rule)
            semantic_diff = self._compute_semantic_difference(correct_rule, rule)
            hardness_scores.append(structural_sim * semantic_diff)
        
        # Sort by hardness and return top K
        if hardness_scores:
            sorted_indices = np.argsort(hardness_scores)[::-1]
            hard_negatives = [hard_negatives[i] for i in sorted_indices]
        
        return hard_negatives[:num_hard_negatives]
    
    def _compute_structural_similarity(self, rule1: Dict, rule2: Dict) -> float:
        """Compute structural similarity between two rules."""
        # Compare rule structure
        body1, head1 = rule1.get('body', []), rule1.get('head', [])
        body2, head2 = rule2.get('body', []), rule2.get('head', [])
        
        # Size similarity
        size_sim = 1.0 - abs(len(body1) - len(body2)) / max(len(body1), len(body2), 1)
        
        # Pattern similarity (conjunction, disjunction, etc.)
        patterns1 = self._extract_patterns(rule1)
        patterns2 = self._extract_patterns(rule2)
        pattern_sim = len(patterns1 & patterns2) / max(len(patterns1 | patterns2), 1)
        
        # Combined similarity
        return 0.5 * size_sim + 0.5 * pattern_sim
    
    def _compute_semantic_difference(self, rule1: Dict, rule2: Dict) -> float:
        """Compute semantic difference between two rules."""
        # Compare conclusions
        head1 = set(str(atom) for atom in rule1.get('head', []))
        head2 = set(str(atom) for atom in rule2.get('head', []))
        
        conclusion_diff = len(head1.symmetric_difference(head2)) / max(len(head1 | head2), 1)
        
        # Compare premises
        body1 = set(str(atom) for atom in rule1.get('body', []))
        body2 = set(str(atom) for atom in rule2.get('body', []))
        
        premise_diff = len(body1.symmetric_difference(body2)) / max(len(body1 | body2), 1)
        
        return 0.5 * conclusion_diff + 0.5 * premise_diff
    
    def _extract_patterns(self, rule: Dict) -> Set[str]:
        """Extract logical patterns from a rule."""
        patterns = set()
        
        for atom in rule.get('body', []) + rule.get('head', []):
            atom_str = str(atom)
            
            if '∧' in atom_str:
                patterns.add('conjunction')
            if '∨' in atom_str:
                patterns.add('disjunction')
            if '¬' in atom_str:
                patterns.add('negation')
            if '→' in atom_str:
                patterns.add('implication')
            if '∀' in atom_str:
                patterns.add('universal')
            if '∃' in atom_str:
                patterns.add('existential')
        
        return patterns


class EnhancedHardNegativeLoss(nn.Module):
    """
    Enhanced loss function with semantic hard negative mining.
    
    Replaces simple hard negative sampling with semantic similarity.
    """
    
    def __init__(self, margin: float = 1.0, hard_neg_weight: float = 2.0):
        super().__init__()
        self.margin = margin
        self.hard_neg_weight = hard_neg_weight
        self.hard_negative_miner = HardNegativeMiner()
    
    def forward(self, scores: torch.Tensor, embeddings: torch.Tensor, 
                target_idx: int, rules: List[Dict]) -> torch.Tensor:
        """
        Args:
            scores: [num_rules] prediction scores
            embeddings: [num_rules, embedding_dim] rule embeddings
            target_idx: Index of correct rule
            rules: List of rule dictionaries
        
        Returns:
            Loss value
        """
        if target_idx < 0 or target_idx >= len(scores):
            return torch.tensor(0.0, device=scores.device, requires_grad=True)
        
        # Add rules to miner if not already added
        if not self.hard_negative_miner.rules:
            self.hard_negative_miner.add_rules(rules)
        
        # Get hard negatives
        hard_negatives = self.hard_negative_miner.get_semantic_hard_negatives(
            target_idx, num_hard_negatives=5
        )
        
        if not hard_negatives:
            # Fallback to top-k scoring negatives
            _, top_indices = torch.topk(scores, k=min(5, len(scores)))
            hard_negatives = [idx.item() for idx in top_indices if idx.item() != target_idx]
        
        if not hard_negatives:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)
        
        # Compute loss
        positive_score = scores[target_idx]
        hard_negative_scores = scores[hard_negatives]
        
        # Hard negative loss
        hard_losses = F.relu(self.margin - (positive_score.unsqueeze(0) - hard_negative_scores))
        hard_loss = hard_losses.mean()
        
        # Easy negative loss (all other negatives)
        all_negatives = [i for i in range(len(scores)) if i != target_idx and i not in hard_negatives]
        if all_negatives:
            easy_negative_scores = scores[all_negatives]
            easy_losses = F.relu(self.margin - (positive_score.unsqueeze(0) - easy_negative_scores))
            easy_loss = easy_losses.mean()
        else:
            easy_loss = torch.tensor(0.0, device=scores.device)
        
        # Combined loss
        total_loss = self.hard_neg_weight * hard_loss + easy_loss
        
        return total_loss


def create_semantic_hard_negative_loss(margin: float = 1.0, 
                                      hard_neg_weight: float = 2.0) -> EnhancedHardNegativeLoss:
    """Create enhanced loss function with semantic hard negative mining."""
    return EnhancedHardNegativeLoss(margin, hard_neg_weight)
