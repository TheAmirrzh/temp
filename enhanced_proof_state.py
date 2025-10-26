"""
Enhanced Proof State Representation
Implements Issue #7: Complete proof state tracking for SOTA theorem proving
"""

import torch
import torch.nn as nn
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict, deque
import numpy as np


@dataclass
class Goal:
    """Represents a sub-goal in the proof."""
    formula: str
    depth: int
    priority: float  # Higher = more important
    dependencies: List[int]  # Fact indices this goal depends on
    is_axiom: bool = False


@dataclass
class Hypothesis:
    """Represents an available fact/hypothesis."""
    formula: str
    fact_id: int
    step_derived: int
    confidence: float
    is_derived: bool = True


@dataclass
class TacticAttempt:
    """Represents a failed tactic attempt."""
    tactic_type: str
    parameters: Dict[str, Any]
    step_attempted: int
    failure_reason: str


@dataclass
class ProofTreeNode:
    """Node in the proof tree structure."""
    node_id: int
    formula: str
    parent: Optional[int]
    children: List[int]
    step_derived: int
    rule_used: Optional[int]
    confidence: float


class ProofTree:
    """Tree structure representing partial proof."""
    
    def __init__(self):
        self.nodes: Dict[int, ProofTreeNode] = {}
        self.root: Optional[int] = None
        self.leaves: Set[int] = set()
        self.next_id = 0
    
    def add_node(self, formula: str, parent: Optional[int] = None, 
                 step_derived: int = 0, rule_used: Optional[int] = None,
                 confidence: float = 1.0) -> int:
        """Add a new node to the proof tree."""
        node_id = self.next_id
        self.next_id += 1
        
        node = ProofTreeNode(
            node_id=node_id,
            formula=formula,
            parent=parent,
            children=[],
            step_derived=step_derived,
            rule_used=rule_used,
            confidence=confidence
        )
        
        self.nodes[node_id] = node
        
        if parent is not None and parent in self.nodes:
            self.nodes[parent].children.append(node_id)
        
        if parent is None:
            self.root = node_id
        
        # Update leaves
        if parent in self.leaves:
            self.leaves.remove(parent)
        self.leaves.add(node_id)
        
        return node_id
    
    def get_path_to_root(self, node_id: int) -> List[int]:
        """Get path from node to root."""
        path = []
        current = node_id
        while current is not None:
            path.append(current)
            if current in self.nodes:
                current = self.nodes[current].parent
            else:
                break
        return path
    
    def get_subtree(self, node_id: int) -> Set[int]:
        """Get all nodes in subtree rooted at node_id."""
        subtree = set()
        queue = [node_id]
        
        while queue:
            current = queue.pop(0)
            if current in self.nodes:
                subtree.add(current)
                queue.extend(self.nodes[current].children)
        
        return subtree


class EnhancedProofState:
    """
    Complete proof state representation following SOTA methods.
    
    Tracks:
    - Current goals (what we're trying to prove)
    - Available facts/hypotheses
    - Failed tactic attempts
    - Proof tree structure
    - Search depth and frontier
    """
    
    def __init__(self, instance: Dict, step_idx: int):
        self.instance = instance
        self.step_idx = step_idx
        
        # Core state components
        self.goals: List[Goal] = []
        self.context: List[Hypothesis] = []
        self.tactics_tried: List[TacticAttempt] = []
        self.proof_tree = ProofTree()
        self.depth = step_idx
        self.frontier: Set[int] = set()
        
        # Derived state
        self.derived_mask = self._compute_derived_mask()
        self.step_numbers = self._compute_step_numbers()
        
        # Initialize state
        self._initialize_goals()
        self._initialize_context()
        self._initialize_proof_tree()
        self._update_frontier()
    
    def _compute_derived_mask(self) -> torch.Tensor:
        """Compute which facts have been derived up to step_idx."""
        derived = torch.zeros(len(self.instance['facts']), dtype=torch.bool)
        
        for step in self.instance['steps'][:self.step_idx + 1]:
            if 'derived_fact' in step:
                fact_idx = step['derived_fact']
                if fact_idx < len(derived):
                    derived[fact_idx] = True
        
        return derived
    
    def _compute_step_numbers(self) -> torch.Tensor:
        """Compute when each fact was derived."""
        step_numbers = torch.zeros(len(self.instance['facts']), dtype=torch.long)
        
        for step_idx, step in enumerate(self.instance['steps'][:self.step_idx + 1]):
            if 'derived_fact' in step:
                fact_idx = step['derived_fact']
                if fact_idx < len(step_numbers):
                    step_numbers[fact_idx] = step_idx
        
        return step_numbers
    
    def _initialize_goals(self):
        """Initialize current goals from the instance."""
        # Main goal
        main_goal = Goal(
            formula=self.instance['goal'],
            depth=0,
            priority=1.0,
            dependencies=[],
            is_axiom=False
        )
        self.goals.append(main_goal)
        
        # Sub-goals from proof steps
        for step in self.instance['steps'][:self.step_idx + 1]:
            if 'sub_goals' in step:
                for i, sub_goal in enumerate(step['sub_goals']):
                    goal = Goal(
                        formula=sub_goal,
                        depth=step.get('depth', 0),
                        priority=1.0 / (i + 1),  # First sub-goal has higher priority
                        dependencies=step.get('dependencies', []),
                        is_axiom=False
                    )
                    self.goals.append(goal)
    
    def _initialize_context(self):
        """Initialize available facts/hypotheses."""
        for i, fact in enumerate(self.instance['facts']):
            if self.derived_mask[i]:
                hypothesis = Hypothesis(
                    formula=fact,
                    fact_id=i,
                    step_derived=int(self.step_numbers[i].item()),
                    confidence=1.0,
                    is_derived=True
                )
                self.context.append(hypothesis)
    
    def _initialize_proof_tree(self):
        """Initialize proof tree structure."""
        # Add axioms as root nodes
        for i, fact in enumerate(self.instance['facts']):
            if not self.derived_mask[i]:  # Axioms are not derived
                self.proof_tree.add_node(
                    formula=fact,
                    parent=None,
                    step_derived=0,
                    rule_used=None,
                    confidence=1.0
                )
        
        # Add derived facts
        for step_idx, step in enumerate(self.instance['steps'][:self.step_idx + 1]):
            if 'derived_fact' in step:
                fact_idx = step['derived_fact']
                fact = self.instance['facts'][fact_idx]
                
                # Find parent nodes (premises)
                parent_nodes = []
                if 'premises' in step:
                    for premise_idx in step['premises']:
                        # Find corresponding proof tree nodes
                        for node_id, node in self.proof_tree.nodes.items():
                            if (node.formula == self.instance['facts'][premise_idx] and 
                                node.step_derived <= step_idx):
                                parent_nodes.append(node_id)
                
                # Add derived fact to tree
                if parent_nodes:
                    # Use first parent as primary parent
                    self.proof_tree.add_node(
                        formula=fact,
                        parent=parent_nodes[0],
                        step_derived=step_idx,
                        rule_used=step.get('rule_used'),
                        confidence=step.get('confidence', 1.0)
                    )
    
    def _update_frontier(self):
        """Update frontier nodes (active proof branches)."""
        self.frontier = set()
        
        # Frontier includes all leaf nodes in proof tree
        self.frontier.update(self.proof_tree.leaves)
        
        # Also include recently derived facts
        recent_threshold = max(0, self.step_idx - 5)
        for i, step_num in enumerate(self.step_numbers):
            if step_num > recent_threshold and self.derived_mask[i]:
                # Find corresponding proof tree node
                for node_id, node in self.proof_tree.nodes.items():
                    if (node.formula == self.instance['facts'][i] and 
                        node.step_derived == step_num):
                        self.frontier.add(node_id)
                        break
    
    def get_available_facts(self) -> List[Hypothesis]:
        """Get all available facts for this step."""
        return self.context
    
    def get_failed_attempts(self) -> List[TacticAttempt]:
        """Get failed tactic attempts up to this step."""
        return self.tactics_tried
    
    def get_dependencies(self) -> torch.Tensor:
        """Get dependency matrix between facts."""
        n_facts = len(self.instance['facts'])
        dependencies = torch.zeros((n_facts, n_facts), dtype=torch.bool)
        
        for step in self.instance['steps'][:self.step_idx + 1]:
            if 'derived_fact' in step and 'premises' in step:
                derived_idx = step['derived_fact']
                for premise_idx in step['premises']:
                    if (derived_idx < n_facts and premise_idx < n_facts):
                        dependencies[premise_idx, derived_idx] = True
        
        return dependencies
    
    def add_failed_attempt(self, tactic_type: str, parameters: Dict[str, Any], 
                          failure_reason: str):
        """Record a failed tactic attempt."""
        attempt = TacticAttempt(
            tactic_type=tactic_type,
            parameters=parameters,
            step_attempted=self.step_idx,
            failure_reason=failure_reason
        )
        self.tactics_tried.append(attempt)
    
    def get_state_embedding(self, hidden_dim: int = 128) -> torch.Tensor:
        """Get vector representation of the complete proof state."""
        # Goal embeddings
        goal_embeddings = []
        for goal in self.goals:
            # Simple embedding based on goal properties
            goal_emb = torch.tensor([
                goal.depth,
                goal.priority,
                len(goal.dependencies),
                1.0 if goal.is_axiom else 0.0
            ], dtype=torch.float)
            goal_embeddings.append(goal_emb)
        
        # Context embeddings
        context_embeddings = []
        for hyp in self.context:
            context_emb = torch.tensor([
                hyp.step_derived,
                hyp.confidence,
                1.0 if hyp.is_derived else 0.0
            ], dtype=torch.float)
            context_embeddings.append(context_emb)
        
        # Frontier embeddings
        frontier_embeddings = []
        for node_id in self.frontier:
            if node_id in self.proof_tree.nodes:
                node = self.proof_tree.nodes[node_id]
                frontier_emb = torch.tensor([
                    node.step_derived,
                    node.confidence,
                    len(node.children)
                ], dtype=torch.float)
                frontier_embeddings.append(frontier_emb)
        
        # Combine all embeddings
        all_embeddings = []
        if goal_embeddings:
            all_embeddings.extend(goal_embeddings)
        if context_embeddings:
            all_embeddings.extend(context_embeddings)
        if frontier_embeddings:
            all_embeddings.extend(frontier_embeddings)
        
        if not all_embeddings:
            return torch.zeros(hidden_dim)
        
        # Pad to fixed size
        max_len = 50  # Maximum number of components
        if len(all_embeddings) > max_len:
            all_embeddings = all_embeddings[:max_len]
        else:
            # Pad with zeros
            while len(all_embeddings) < max_len:
                all_embeddings.append(torch.zeros(4))  # Match embedding dimension
        
        # Flatten and project to hidden_dim
        state_vector = torch.cat(all_embeddings)
        if len(state_vector) > hidden_dim:
            # Truncate if too long
            state_vector = state_vector[:hidden_dim]
        elif len(state_vector) < hidden_dim:
            # Pad if too short
            padding = torch.zeros(hidden_dim - len(state_vector))
            state_vector = torch.cat([state_vector, padding])
        
        return state_vector


def create_enhanced_proof_state(instance: Dict, step_idx: int) -> EnhancedProofState:
    """Create enhanced proof state from instance and step index."""
    return EnhancedProofState(instance, step_idx)


def compute_derivation_dependencies(instance: Dict, step_idx: int) -> torch.Tensor:
    """Compute dependency matrix between facts."""
    n_facts = len(instance['facts'])
    dependencies = torch.zeros((n_facts, n_facts), dtype=torch.bool)
    
    for step in instance['steps'][:step_idx + 1]:
        if 'derived_fact' in step and 'premises' in step:
            derived_idx = step['derived_fact']
            for premise_idx in step['premises']:
                if (derived_idx < n_facts and premise_idx < n_facts):
                    dependencies[premise_idx, derived_idx] = True
    
    return dependencies


def compute_derived_mask(instance: Dict, step_idx: int) -> torch.Tensor:
    """Compute which facts have been derived up to step_idx."""
    derived = torch.zeros(len(instance['facts']), dtype=torch.bool)
    
    for step in instance['steps'][:step_idx + 1]:
        if 'derived_fact' in step:
            fact_idx = step['derived_fact']
            if fact_idx < len(derived):
                derived[fact_idx] = True
    
    return derived


def compute_step_numbers(instance: Dict, step_idx: int) -> torch.Tensor:
    """Compute when each fact was derived."""
    step_numbers = torch.zeros(len(instance['facts']), dtype=torch.long)
    
    for step_idx_actual, step in enumerate(instance['steps'][:step_idx + 1]):
        if 'derived_fact' in step:
            fact_idx = step['derived_fact']
            if fact_idx < len(step_numbers):
                step_numbers[fact_idx] = step_idx_actual
    
    return step_numbers
