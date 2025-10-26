"""
Fixed MCTS-based Proof Search with Proper State Transitions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Set, Any
import numpy as np
from collections import defaultdict, deque
import random
from dataclasses import dataclass
import math
from proof_state import ProofState, ProofStateEncoder, ProofValidator


@dataclass
class SearchNode:
    """Node in the MCTS search tree."""
    state: ProofState
    parent: Optional['SearchNode'] = None
    children: List['SearchNode'] = None
    action: Optional[int] = None  # Rule ID
    visits: int = 0
    total_value: float = 0.0
    prior_prob: float = 1.0
    is_terminal: bool = False
    is_expanded: bool = False
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    @property
    def value(self) -> float:
        """Average value of this node."""
        return self.total_value / max(self.visits, 1)
    
    @property
    def ucb_score(self) -> float:
        """UCB score for node selection."""
        if self.visits == 0:
            return float('inf')
        
        c = 1.414  # Exploration constant
        parent_visits = self.parent.visits if self.parent else 1
        exploitation = self.value
        exploration = c * math.sqrt(math.log(parent_visits) / self.visits)
        
        return exploitation + exploration


class PolicyNetwork(nn.Module):
    """Policy and value network for MCTS."""
    
    def __init__(self, hidden_dim: int = 256, num_actions: int = 100):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state_embedding: [hidden_dim] or [batch, hidden_dim]
        
        Returns:
            policy_logits: [num_actions]
            value: [1] in range [-1, 1]
        """
        features = self.backbone(state_embedding)
        policy_logits = self.policy_head(features)
        value = torch.tanh(self.value_head(features))
        
        return policy_logits, value


class MCTSSearch:
    """
    Monte Carlo Tree Search with proper state transitions.
    """
    
    def __init__(self, 
                 policy_network: PolicyNetwork,
                 state_encoder: ProofStateEncoder,
                 num_simulations: int = 100,
                 max_depth: int = 50,
                 exploration_constant: float = 1.414):
        self.policy_network = policy_network
        self.state_encoder = state_encoder
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.exploration_constant = exploration_constant
        
        self.root: Optional[SearchNode] = None
        self.node_count = 0
    
    def search(self, initial_state: ProofState) -> Tuple[int, float]:
        """
        Perform MCTS search.
        
        Args:
            initial_state: Starting proof state
        
        Returns:
            best_action: Best rule to apply
            win_rate: Estimated win rate for best action
        """
        # Initialize root
        self.root = SearchNode(state=initial_state.copy())
        self.node_count = 0
        
        # Run simulations
        for _ in range(self.num_simulations):
            self._simulate()
        
        # Select best action
        if not self.root.children:
            # No children, return first applicable rule
            applicable = initial_state.get_applicable_rules()
            return applicable[0] if applicable else -1, 0.0
        
        # Select child with highest visit count
        best_child = max(self.root.children, key=lambda c: c.visits)
        win_rate = best_child.value
        
        return best_child.action, win_rate
    
    def _simulate(self):
        """Run single MCTS simulation."""
        # Selection + Expansion
        path = self._select_and_expand()
        
        if not path:
            return
        
        leaf = path[-1]
        
        # Evaluation
        if leaf.is_terminal:
            # Terminal node: get exact value
            value = 1.0 if leaf.state.goal_satisfied() else -1.0
        else:
            # Non-terminal: use neural network
            value = self._evaluate_neural(leaf.state)
        
        # Backpropagation
        self._backpropagate(path, value)
    
    def _select_and_expand(self) -> List[SearchNode]:
        """
        Selection phase: traverse tree using UCB.
        Expansion phase: expand first non-expanded node.
        
        Returns:
            Path from root to leaf
        """
        path = [self.root]
        current = self.root
        
        while not current.is_terminal:
            if not current.is_expanded:
                # Expand this node
                self._expand(current)
                path.append(current)
                return path
            
            if not current.children:
                # No children (dead end)
                return path
            
            # Select child with best UCB score
            best_child = max(current.children, key=lambda c: c.ucb_score)
            path.append(best_child)
            current = best_child
        
        return path
    
    def _expand(self, node: SearchNode):
        """Expand node by creating children for applicable actions."""
        if node.is_expanded:
            return
        
        # Get applicable rules
        applicable_rules = node.state.get_applicable_rules()
        
        if not applicable_rules:
            node.is_terminal = True
            node.is_expanded = True
            return
        
        # Get policy prior
        with torch.no_grad():
            state_embedding = self.state_encoder(node.state)
            policy_logits, _ = self.policy_network(state_embedding)
            policy_probs = F.softmax(policy_logits, dim=-1).cpu().numpy()
        
        # Create children
        for rule_id in applicable_rules:
            # Apply rule to get next state
            next_state = node.state.copy()
            success = next_state.apply_rule(rule_id)
            
            if not success:
                continue
            
            # Create child node
            prior_prob = float(policy_probs[rule_id]) if rule_id < len(policy_probs) else 1.0
            
            child = SearchNode(
                state=next_state,
                parent=node,
                action=rule_id,
                prior_prob=prior_prob,
                is_terminal=next_state.is_terminal()
            )
            
            node.children.append(child)
            self.node_count += 1
        
        node.is_expanded = True
    
    def _evaluate_neural(self, state: ProofState) -> float:
        """Evaluate state using neural network."""
        with torch.no_grad():
            state_embedding = self.state_encoder(state)
            _, value = self.policy_network(state_embedding)
            return float(value.item())
    
    def _backpropagate(self, path: List[SearchNode], value: float):
        """Backpropagate value through path."""
        for node in reversed(path):
            node.visits += 1
            node.total_value += value


class RewardComputer:
    """
    Computes rewards based on proof progress.
    """
    
    def __init__(self, max_depth: int = 50):
        self.max_depth = max_depth
    
    def compute_progress_reward(self, old_state: ProofState, 
                               new_state: ProofState) -> float:
        """
        Reward for reducing open goals.
        Range: [0, 0.5]
        """
        old_goals = len(old_state.get_open_goals())
        new_goals = len(new_state.get_open_goals())
        
        if old_goals == 0:
            return 0.0
        
        reduction = (old_goals - new_goals) / old_goals
        return max(0, reduction * 0.5)
    
    def compute_correctness_reward(self, rule: 'Rule') -> float:
        """
        Reward proportional to rule confidence.
        Range: [0, 0.3]
        """
        return rule.confidence * 0.3
    
    def compute_failure_penalty(self, old_state: ProofState, 
                               new_state: ProofState) -> float:
        """
        Penalty if step doesn't reduce goals.
        Range: [-0.1, 0]
        """
        old_goals = len(old_state.get_open_goals())
        new_goals = len(new_state.get_open_goals())
        
        if new_goals >= old_goals:
            return -0.1
        return 0.0
    
    def compute_completion_bonus(self, state: ProofState) -> float:
        """
        Large bonus for completing proof.
        Range: [0, 1.0]
        """
        return 1.0 if state.goal_satisfied() else 0.0
    
    def compute_total_reward(self, old_state: ProofState, 
                            new_state: ProofState, rule: 'Rule') -> float:
        """
        Compute total reward for state transition.
        
        Returns:
            Reward in range [-0.1, 1.0]
        """
        progress = self.compute_progress_reward(old_state, new_state)
        correctness = self.compute_correctness_reward(rule)
        failure = self.compute_failure_penalty(old_state, new_state)
        completion = self.compute_completion_bonus(new_state)
        
        total = progress + correctness + failure + completion
        return total


class ProofSearchAgent:
    """
    Complete proof search agent with MCTS and neural networks.
    """
    
    def __init__(self, 
                 state_encoder: ProofStateEncoder,
                 hidden_dim: int = 256,
                 num_actions: int = 100,
                 num_simulations: int = 100):
        self.state_encoder = state_encoder
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.num_simulations = num_simulations
        
        # Networks
        self.policy_network = PolicyNetwork(hidden_dim, num_actions)
        self.mcts = MCTSSearch(self.policy_network, state_encoder, num_simulations)
        self.reward_computer = RewardComputer()
        
        # Training data
        self.training_data = deque(maxlen=10000)
    
    def get_action(self, state: ProofState, use_mcts: bool = True) -> int:
        """
        Get best action for current state.
        
        Args:
            state: Current proof state
            use_mcts: Use MCTS search vs greedy policy
        
        Returns:
            Rule ID to apply
        """
        if use_mcts:
            best_action, _ = self.mcts.search(state)
            return best_action if best_action >= 0 else state.get_applicable_rules()[0]
        else:
            # Greedy: use policy directly
            with torch.no_grad():
                state_embedding = self.state_encoder(state)
                policy_logits, _ = self.policy_network(state_embedding)
                
                # Mask invalid actions
                applicable = state.get_applicable_rules()
                mask = torch.full_like(policy_logits, -1e9)
                for rule_id in applicable:
                    mask[rule_id] = 0
                
                masked_logits = policy_logits + mask
                best_action = masked_logits.argmax().item()
                
                return best_action if best_action in applicable else applicable[0]
    
    def collect_trajectory(self, initial_state: ProofState) -> List[Dict]:
        """
        Collect a proof trajectory using MCTS.
        
        Returns:
            List of (state, action, reward, next_state) tuples
        """
        trajectory = []
        state = initial_state.copy()
        
        while not state.is_terminal():
            # Get action
            action = self.get_action(state, use_mcts=True)
            
            if action < 0:
                break
            
            # Apply action
            next_state = state.copy()
            success = next_state.apply_rule(action)
            
            if not success:
                break
            
            # Compute reward
            rule = state.rules[action]
            reward = self.reward_computer.compute_total_reward(state, next_state, rule)
            
            # Record transition
            trajectory.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': next_state.is_terminal()
            })
            
            self.training_data.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state
            })
            
            state = next_state
        
        return trajectory
    
    def train_step(self, batch_size: int = 32) -> Dict[str, float]:
        """
        Train on collected experience.
        
        Returns:
            Loss metrics
        """
        if len(self.training_data) < batch_size:
            return {'policy_loss': 0.0, 'value_loss': 0.0}
        
        # Sample batch
        batch = random.sample(list(self.training_data), batch_size)
        
        # Encode states
        state_embeddings = torch.stack([
            self.state_encoder(item['state']) for item in batch
        ])
        actions = torch.tensor([item['action'] for item in batch], dtype=torch.long)
        rewards = torch.tensor([item['reward'] for item in batch], dtype=torch.float)
        
        # Forward pass
        policy_logits, values = self.policy_network(state_embeddings)
        
        # Losses
        policy_loss = F.cross_entropy(policy_logits, actions)
        value_loss = F.mse_loss(values.squeeze(), rewards)
        
        total_loss = policy_loss + value_loss
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item()
        }