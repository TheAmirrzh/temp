"""
Multi-Relational Graph Structure
Implements Issue #12: Multi-relational graph capturing fact-to-fact dependencies and rule relationships
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Set, Any
import numpy as np
from collections import defaultdict
import networkx as nx
from dataclasses import dataclass


@dataclass
class HyperEdge:
    """Represents a hyperedge in the proof graph."""
    body: List[int]  # Fact indices in body
    head: List[int]  # Fact indices in head
    rule_id: int     # Rule that created this hyperedge
    confidence: float = 1.0


class MultiRelationalGraph:
    """
    Multi-relational graph structure for Horn clauses.
    
    Captures multiple types of relationships:
    1. Fact-to-fact dependencies (causal structure)
    2. Rule-to-rule relationships (precedence)
    3. Hypergraph representation (rules as hyperedges)
    """
    
    def __init__(self, facts: List[str], rules: List[Dict]):
        self.facts = facts
        self.rules = rules
        self.n_facts = len(facts)
        self.n_rules = len(rules)
        
        # Multi-relational edge lists
        self.edge_lists = {
            'premise': [],      # (fact, rule) - fact used in rule
            'conclusion': [],   # (rule, fact) - rule derives fact
            'enables': [],      # (fact_i, fact_j) - fact i enables deriving j
            'precedes': [],     # (rule_i, rule_j) - rule i must fire before j
            'contradicts': [],  # (fact_i, fact_j) - facts contradict each other
            'supports': []      # (fact_i, fact_j) - fact i supports fact j
        }
        
        # Edge attributes
        self.edge_attrs = {
            'premise': [],
            'conclusion': [],
            'enables': [],
            'precedes': [],
            'contradicts': [],
            'supports': []
        }
        
        # Hypergraph representation
        self.hyperedges: List[HyperEdge] = []
        
        # Build graph
        self._build_graph()
    
    def _build_graph(self):
        """Build the multi-relational graph."""
        # Build premise and conclusion edges
        self._build_rule_edges()
        
        # Build dependency edges
        self._build_dependency_edges()
        
        # Build precedence edges
        self._build_precedence_edges()
        
        # Build hyperedges
        self._build_hyperedges()
    
    def _build_rule_edges(self):
        """Build premise and conclusion edges."""
        for rule_idx, rule in enumerate(self.rules):
            body = rule.get('body', [])
            head = rule.get('head', [])
            
            # Premise edges: (fact, rule)
            for fact_idx in body:
                if fact_idx < self.n_facts:
                    self.edge_lists['premise'].append((fact_idx, rule_idx))
                    self.edge_attrs['premise'].append({
                        'weight': 1.0,
                        'rule_type': rule.get('type', 'unknown')
                    })
            
            # Conclusion edges: (rule, fact)
            for fact_idx in head:
                if fact_idx < self.n_facts:
                    self.edge_lists['conclusion'].append((rule_idx, fact_idx))
                    self.edge_attrs['conclusion'].append({
                        'weight': 1.0,
                        'confidence': rule.get('confidence', 1.0)
                    })
    
    def _build_dependency_edges(self):
        """Build fact-to-fact dependency edges."""
        # Track which facts enable which other facts
        fact_dependencies = defaultdict(set)
        
        for rule_idx, rule in enumerate(self.rules):
            body = rule.get('body', [])
            head = rule.get('head', [])
            
            # Each premise enables each conclusion
            for premise_idx in body:
                for conclusion_idx in head:
                    if (premise_idx < self.n_facts and 
                        conclusion_idx < self.n_facts and
                        premise_idx != conclusion_idx):
                        fact_dependencies[premise_idx].add(conclusion_idx)
        
        # Create dependency edges
        for premise_idx, conclusions in fact_dependencies.items():
            for conclusion_idx in conclusions:
                self.edge_lists['enables'].append((premise_idx, conclusion_idx))
                self.edge_attrs['enables'].append({
                    'weight': 1.0,
                    'dependency_strength': 1.0
                })
    
    def _build_precedence_edges(self):
        """Build rule precedence edges."""
        # Rules that must fire before others
        rule_precedence = defaultdict(set)
        
        for rule_idx, rule in enumerate(self.rules):
            body = rule.get('body', [])
            head = rule.get('head', [])
            
            # Find rules that produce facts needed by this rule
            for premise_idx in body:
                for other_rule_idx, other_rule in enumerate(self.rules):
                    if other_rule_idx != rule_idx:
                        other_head = other_rule.get('head', [])
                        if premise_idx in other_head:
                            rule_precedence[other_rule_idx].add(rule_idx)
        
        # Create precedence edges
        for rule_i, dependent_rules in rule_precedence.items():
            for rule_j in dependent_rules:
                self.edge_lists['precedes'].append((rule_i, rule_j))
                self.edge_attrs['precedes'].append({
                    'weight': 1.0,
                    'precedence_strength': 1.0
                })
    
    def _build_hyperedges(self):
        """Build hypergraph representation."""
        for rule_idx, rule in enumerate(self.rules):
            body = rule.get('body', [])
            head = rule.get('head', [])
            
            hyperedge = HyperEdge(
                body=body,
                head=head,
                rule_id=rule_idx,
                confidence=rule.get('confidence', 1.0)
            )
            self.hyperedges.append(hyperedge)
    
    def get_edge_tensor(self, edge_type: str) -> torch.Tensor:
        """Get edge tensor for a specific relation type."""
        if edge_type not in self.edge_lists:
            return torch.empty((0, 2), dtype=torch.long)
        
        edges = self.edge_lists[edge_type]
        if not edges:
            return torch.empty((0, 2), dtype=torch.long)
        
        return torch.tensor(edges, dtype=torch.long).t()
    
    def get_edge_attr_tensor(self, edge_type: str) -> torch.Tensor:
        """Get edge attribute tensor for a specific relation type."""
        if edge_type not in self.edge_attrs:
            return torch.empty((0, 1))
        
        attrs = self.edge_attrs[edge_type]
        if not attrs:
            return torch.empty((0, 1))
        
        # Convert to tensor (simplified - just weights for now)
        weights = [attr.get('weight', 1.0) for attr in attrs]
        return torch.tensor(weights, dtype=torch.float).unsqueeze(1)
    
    def get_all_edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get all edges and attributes as tensors."""
        all_edges = []
        all_attrs = []
        edge_type_mapping = []
        
        for edge_type in self.edge_lists:
            edges = self.get_edge_tensor(edge_type)
            attrs = self.get_edge_attr_tensor(edge_type)
            
            if edges.shape[1] > 0:
                all_edges.append(edges)
                all_attrs.append(attrs)
                edge_type_mapping.extend([edge_type] * edges.shape[1])
        
        if not all_edges:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1))
        
        combined_edges = torch.cat(all_edges, dim=1)
        combined_attrs = torch.cat(all_attrs, dim=0)
        
        return combined_edges, combined_attrs
    
    def get_hypergraph_representation(self) -> Dict[str, torch.Tensor]:
        """Get hypergraph representation."""
        if not self.hyperedges:
            return {
                'hyperedges': torch.empty((0, 2)),
                'hyperedge_attrs': torch.empty((0, 1))
            }
        
        # Convert hyperedges to edge list format
        hyperedges = []
        hyperedge_attrs = []
        
        for hyperedge in self.hyperedges:
            # Create edges from body to head
            for body_idx in hyperedge.body:
                for head_idx in hyperedge.head:
                    hyperedges.append([body_idx, head_idx])
                    hyperedge_attrs.append([hyperedge.confidence])
        
        return {
            'hyperedges': torch.tensor(hyperedges, dtype=torch.long).t(),
            'hyperedge_attrs': torch.tensor(hyperedge_attrs, dtype=torch.float)
        }


class MultiRelationalGNN(nn.Module):
    """
    GNN for multi-relational graphs.
    
    Processes different relation types separately and combines them.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, 
                 num_relations: int = 6, num_layers: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.num_layers = num_layers
        
        # Relation-specific GNN layers
        self.relation_layers = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim)
                for _ in range(num_layers)
            ]) for _ in range(num_relations)
        ])
        
        # Relation type embeddings
        self.relation_embeddings = nn.Embedding(num_relations, hidden_dim)
        
        # Attention mechanism for combining relations
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Relation names
        self.relation_names = [
            'premise', 'conclusion', 'enables', 
            'precedes', 'contradicts', 'supports'
        ]
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor, edge_type: List[str]) -> torch.Tensor:
        """
        Args:
            x: [num_nodes, input_dim] node features
            edge_index: [2, num_edges] edge indices
            edge_attr: [num_edges, attr_dim] edge attributes
            edge_type: [num_edges] relation types
        
        Returns:
            [num_nodes, hidden_dim] updated node features
        """
        batch_size, num_nodes, _ = x.shape
        
        # Process each relation type separately
        relation_outputs = []
        
        for rel_idx, rel_name in enumerate(self.relation_names):
            # Get edges of this relation type
            rel_mask = [t == rel_name for t in edge_type]
            if not any(rel_mask):
                # No edges of this type, use zero features
                rel_output = torch.zeros(batch_size, num_nodes, self.hidden_dim, device=x.device)
                relation_outputs.append(rel_output)
                continue
            
            # Filter edges
            rel_edge_index = edge_index[:, rel_mask]
            rel_edge_attr = edge_attr[rel_mask] if edge_attr.shape[0] > 0 else None
            
            # Process through relation-specific layers
            rel_x = x
            for layer in self.relation_layers[rel_idx]:
                rel_x = F.relu(layer(rel_x))
            
            relation_outputs.append(rel_x)
        
        # Combine relation outputs using attention
        if len(relation_outputs) > 1:
            # Stack relation outputs
            stacked_outputs = torch.stack(relation_outputs, dim=1)  # [batch, num_relations, num_nodes, hidden_dim]
            
            # Reshape for attention
            batch_size, num_relations, num_nodes, hidden_dim = stacked_outputs.shape
            stacked_outputs = stacked_outputs.view(batch_size, num_relations, hidden_dim)
            
            # Apply attention
            attended_output, _ = self.attention(
                stacked_outputs, stacked_outputs, stacked_outputs
            )
            
            # Take mean across relations
            combined_output = attended_output.mean(dim=1)  # [batch, hidden_dim]
            
            # Reshape back to node format
            combined_output = combined_output.unsqueeze(1).expand(-1, num_nodes, -1)
        else:
            combined_output = relation_outputs[0]
        
        # Final projection
        output = self.output_proj(combined_output)
        
        return output


class HypergraphGNN(nn.Module):
    """
    GNN for hypergraph representation.
    
    Processes hyperedges (rules) and their connections to facts.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Hyperedge processing
        self.hyperedge_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Fact processing
        self.fact_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Message passing layers
        self.message_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, fact_features: torch.Tensor, 
                hyperedge_index: torch.Tensor,
                hyperedge_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fact_features: [num_facts, input_dim] fact features
            hyperedge_index: [2, num_hyperedges] hyperedge connections
            hyperedge_attr: [num_hyperedges, attr_dim] hyperedge attributes
        
        Returns:
            [num_facts, hidden_dim] updated fact features
        """
        # Encode facts
        fact_embeddings = self.fact_encoder(fact_features)
        
        # Encode hyperedges
        hyperedge_embeddings = self.hyperedge_encoder(fact_features)
        
        # Message passing
        for layer in self.message_layers:
            # Messages from hyperedges to facts
            messages = layer(hyperedge_embeddings)
            
            # Aggregate messages
            fact_embeddings = fact_embeddings + messages
        
        # Final projection
        output = self.output_proj(fact_embeddings)
        
        return output


def create_multi_relational_graph(facts: List[str], rules: List[Dict]) -> MultiRelationalGraph:
    """Create multi-relational graph from facts and rules."""
    return MultiRelationalGraph(facts, rules)


def create_multi_relational_gnn(input_dim: int, hidden_dim: int = 256) -> MultiRelationalGNN:
    """Create multi-relational GNN."""
    return MultiRelationalGNN(input_dim, hidden_dim)


def create_hypergraph_gnn(input_dim: int, hidden_dim: int = 256) -> HypergraphGNN:
    """Create hypergraph GNN."""
    return HypergraphGNN(input_dim, hidden_dim)
