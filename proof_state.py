"""
Core Proof State Interface and Implementation
Provides the foundation for all proof search operations
"""

import torch
import torch.nn as nn
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import copy
import numpy as np


@dataclass
class Fact:
    """Represents a logical fact/hypothesis."""
    formula: str
    fact_id: int
    step_derived: int = 0  # -1 for axioms
    confidence: float = 1.0
    is_axiom: bool = False
    derived_from: Optional[int] = None  # Rule that derived this
    
    def __hash__(self):
        return hash((self.formula, self.fact_id))
    
    def __eq__(self, other):
        return self.fact_id == other.fact_id


@dataclass
class Rule:
    """Represents a logical rule (Horn clause)."""
    rule_id: int
    name: str
    premises: List[int]  # Fact indices
    conclusion: int     # Fact index
    confidence: float = 1.0
    tactic_type: str = "forward_chain"  # Associated tactic


@dataclass
class Goal:
    """Represents a proof goal."""
    formula: str
    goal_id: int
    depth: int = 0
    parent_goal: Optional[int] = None
    depends_on: List[int] = field(default_factory=list)  # Fact indices


class ProofState:
    """
    Complete proof state for theorem proving.
    
    Tracks:
    - Available facts (axioms + derived)
    - Open goals (what's left to prove)
    - Proof history (how we got here)
    - Search statistics
    """
    
    def __init__(self, 
                 facts: List[Dict],
                 rules: List[Dict],
                 goal: str,
                 max_depth: int = 50,
                 max_steps: int = 100):
        self.max_depth = max_depth
        self.max_steps = max_steps
        self.depth = 0
        self.steps_taken = 0
        
        # Parse facts
        self.facts: Dict[int, Fact] = {}
        for i, fact_dict in enumerate(facts):
            fact = Fact(
                formula=fact_dict.get('formula', f'f_{i}'),
                fact_id=i,
                step_derived=-1,  # Axioms
                is_axiom=True,
                confidence=fact_dict.get('confidence', 1.0)
            )
            self.facts[i] = fact
        
        # Parse rules
        self.rules: Dict[int, Rule] = {}
        for i, rule_dict in enumerate(rules):
            rule = Rule(
                rule_id=i,
                name=rule_dict.get('name', f'rule_{i}'),
                premises=rule_dict.get('body', []),
                conclusion=rule_dict.get('head', [None])[0],  # Take first
                confidence=rule_dict.get('confidence', 1.0),
                tactic_type=rule_dict.get('tactic_type', 'forward_chain')
            )
            self.rules[i] = rule
        
        # Main goal
        self.main_goal = Goal(
            formula=goal,
            goal_id=0,
            depth=0
        )
        self.open_goals: Dict[int, Goal] = {0: self.main_goal}
        self.closed_goals: Set[int] = set()
        
        # Proof history
        self.history: List[Tuple[int, int, List[int], int]] = []  # (rule_id, step, premises, conclusion)
        
        # Tracking
        self.applicable_rules_cache = None
        self.last_rule_applied: Optional[int] = None
    
    def copy(self) -> 'ProofState':
        """Create deep copy of proof state."""
        new_state = ProofState.__new__(ProofState)
        new_state.max_depth = self.max_depth
        new_state.max_steps = self.max_steps
        new_state.depth = self.depth
        new_state.steps_taken = self.steps_taken
        
        new_state.facts = {k: copy.copy(v) for k, v in self.facts.items()}
        new_state.rules = copy.copy(self.rules)
        new_state.main_goal = copy.copy(self.main_goal)
        new_state.open_goals = {k: copy.copy(v) for k, v in self.open_goals.items()}
        new_state.closed_goals = copy.copy(self.closed_goals)
        new_state.history = copy.copy(self.history)
        new_state.applicable_rules_cache = None
        new_state.last_rule_applied = self.last_rule_applied
        
        return new_state
    
    def get_available_facts(self) -> List[Fact]:
        """Get all currently available facts."""
        return list(self.facts.values())
    
    def get_open_goals(self) -> List[Goal]:
        """Get all open goals."""
        return list(self.open_goals.values())
    
    def can_apply_rule(self, rule_id: int) -> bool:
        """Check if rule can be applied to current state."""
        if rule_id not in self.rules:
            return False
        
        rule = self.rules[rule_id]
        
        # Check all premises are available
        for premise_id in rule.premises:
            if premise_id not in self.facts:
                return False
        
        # Check conclusion isn't already derived
        if rule.conclusion in self.facts:
            return False
        
        # Check we haven't exceeded depth
        if self.depth >= self.max_depth:
            return False
        
        return True
    
    def get_applicable_rules(self) -> List[int]:
        """Get all applicable rules in current state."""
        applicable = []
        for rule_id in self.rules:
            if self.can_apply_rule(rule_id):
                applicable.append(rule_id)
        return applicable
    
    def apply_rule(self, rule_id: int) -> bool:
        """
        Apply a rule, deriving new fact.
        
        Returns:
            True if successful, False if invalid
        """
        if not self.can_apply_rule(rule_id):
            return False
        
        rule = self.rules[rule_id]
        
        # Derive new fact
        new_fact = Fact(
            formula=self.rules[rule_id].name,  # Simplified
            fact_id=len(self.facts),
            step_derived=self.steps_taken,
            confidence=rule.confidence,
            is_axiom=False,
            derived_from=rule_id
        )
        self.facts[new_fact.fact_id] = new_fact
        
        # Record in history
        self.history.append((rule_id, self.steps_taken, rule.premises, new_fact.fact_id))
        
        # Update counters
        self.depth += 1
        self.steps_taken += 1
        self.last_rule_applied = rule_id
        
        # Check if this closes any goals (simplified)
        self._check_goal_closure()
        
        return True
    
    def _check_goal_closure(self):
        """Check if any open goals are now satisfied."""
        # Simplified: check if goal formula matches any fact
        closed = []
        for goal_id, goal in self.open_goals.items():
            for fact_id, fact in self.facts.items():
                if fact.formula == goal.formula and not fact.is_axiom:
                    closed.append(goal_id)
                    break
        
        for goal_id in closed:
            self.closed_goals.add(goal_id)
            del self.open_goals[goal_id]
    
    def goal_satisfied(self) -> bool:
        """Check if main goal is satisfied."""
        return 0 in self.closed_goals
    
    def is_terminal(self) -> bool:
        """Check if state is terminal (proof complete or failed)."""
        # Terminal if proof complete
        if self.goal_satisfied():
            return True
        
        # Terminal if stuck (no applicable rules and goals open)
        if not self.get_applicable_rules() and self.open_goals:
            return True
        
        # Terminal if exceeded limits
        if self.depth >= self.max_depth or self.steps_taken >= self.max_steps:
            return True
        
        return False
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            'depth': self.depth,
            'steps_taken': self.steps_taken,
            'facts': [(f.formula, f.fact_id, f.step_derived) for f in self.facts.values()],
            'open_goals': len(self.open_goals),
            'closed_goals': len(self.closed_goals),
            'history_len': len(self.history)
        }
    
    def get_progress_metrics(self) -> Dict[str, float]:
        """Get metrics about proof progress."""
        total_goals = len(self.open_goals) + len(self.closed_goals)
        closed_ratio = len(self.closed_goals) / max(total_goals, 1)
        
        return {
            'progress': closed_ratio,
            'depth': self.depth / self.max_depth,
            'steps': self.steps_taken / self.max_steps,
            'goals_closed': len(self.closed_goals),
            'goals_open': len(self.open_goals)
        }


class ProofStateEncoder(nn.Module):
    """
    Encodes proof state to learnable tensor representation.
    """
    
    def __init__(self, hidden_dim: int = 128, max_facts: int = 500):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_facts = max_facts
        
        # Fact encoder
        self.fact_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),  # [step_derived, confidence, is_axiom, depth]
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Goal encoder
        self.goal_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),  # [goal_id, depth, num_depends]
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # State aggregation
        self.aggregation = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def encode_fact(self, fact: Fact, current_step: int) -> torch.Tensor:
        """Encode single fact to tensor."""
        features = torch.tensor([
            (fact.step_derived + 1) / 50.0,  # Normalized step
            fact.confidence,
            1.0 if fact.is_axiom else 0.0,
            1.0 if not fact.is_axiom else 0.0  # is_derived
        ], dtype=torch.float)
        
        return self.fact_encoder(features)
    
    def encode_goal(self, goal: Goal, total_goals: int) -> torch.Tensor:
        """Encode single goal to tensor."""
        features = torch.tensor([
            goal.goal_id / max(total_goals, 1),
            goal.depth / 10.0,
            len(goal.depends_on) / 20.0
        ], dtype=torch.float)
        
        return self.goal_encoder(features)
    
    def forward(self, state: ProofState) -> torch.Tensor:
        """
        Encode proof state to tensor.
        
        Returns:
            [hidden_dim] state embedding
        """
        # Encode facts (average over all facts)
        fact_embeddings = []
        for fact in state.get_available_facts():
            emb = self.encode_fact(fact, state.steps_taken)
            fact_embeddings.append(emb)
        
        if fact_embeddings:
            fact_embedding = torch.stack(fact_embeddings).mean(dim=0)
        else:
            fact_embedding = torch.zeros(self.hidden_dim)
        
        # Encode goals (average over open goals)
        goal_embeddings = []
        for goal in state.get_open_goals():
            emb = self.encode_goal(goal, len(state.open_goals) + len(state.closed_goals))
            goal_embeddings.append(emb)
        
        if goal_embeddings:
            goal_embedding = torch.stack(goal_embeddings).mean(dim=0)
        else:
            goal_embedding = torch.zeros(self.hidden_dim)
        
        # Combine
        combined = torch.cat([fact_embedding, goal_embedding])
        state_tensor = self.aggregation(combined)
        
        return state_tensor


class ProofValidator:
    """
    Validates proof correctness.
    """
    
    def __init__(self, rules: Dict[int, Rule]):
        self.rules = rules
    
    def validate_step(self, rule_id: int, premises: List[int], 
                     conclusion_id: int, facts: Dict[int, Fact]) -> Tuple[bool, str]:
        """Validate single proof step."""
        if rule_id not in self.rules:
            return False, f"Rule {rule_id} not found"
        
        rule = self.rules[rule_id]
        
        # Check premises exist and match
        for premise_id in rule.premises:
            if premise_id not in facts:
                return False, f"Premise {premise_id} not derived"
        
        # Check conclusion matches rule head
        if rule.conclusion != conclusion_id:
            return False, f"Conclusion mismatch: got {conclusion_id}, expected {rule.conclusion}"
        
        return True, "Valid"
    
    def validate_proof(self, history: List[Tuple[int, int, List[int], int]],
                      facts: Dict[int, Fact]) -> Tuple[bool, str]:
        """Validate complete proof sequence."""
        for rule_id, step, premises, conclusion in history:
            valid, msg = self.validate_step(rule_id, premises, conclusion, facts)
            if not valid:
                return False, f"Step {step}: {msg}"
        
        return True, "Complete proof valid"