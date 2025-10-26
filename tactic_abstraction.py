"""
Tactic Abstraction Layer
Implements Issue #8: Tactic/Rule abstraction for better generalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class TacticType:
    """Represents a high-level tactic type."""
    name: str
    description: str
    parameter_schema: Dict[str, Any]  # Expected parameter types
    applicability_conditions: List[str]  # When this tactic can be used


# Define common tactic types
TACTIC_TYPES = [
    TacticType(
        name="forward_chain",
        description="Apply forward chaining: if A∧B→C and have A,B, derive C",
        parameter_schema={"rule_id": int, "premises": List[int]},
        applicability_conditions=["have_all_premises", "rule_applicable"]
    ),
    TacticType(
        name="case_split",
        description="Split on disjunction: if A∨B→C, prove C from A and C from B",
        parameter_schema={"disjunction_fact": int, "cases": List[str]},
        applicability_conditions=["have_disjunction", "cases_well_formed"]
    ),
    TacticType(
        name="induction",
        description="Apply induction: prove base case and inductive step",
        parameter_schema={"induction_var": str, "base_case": str, "inductive_step": str},
        applicability_conditions=["induction_applicable", "base_case_provable"]
    ),
    TacticType(
        name="contradiction",
        description="Prove by contradiction: assume ¬P, derive contradiction",
        parameter_schema={"assumption": str, "contradiction_target": str},
        applicability_conditions=["contradiction_possible", "assumption_valid"]
    ),
    TacticType(
        name="modus_ponens",
        description="Apply modus ponens: if P→Q and P, derive Q",
        parameter_schema={"implication_rule": int, "antecedent": int},
        applicability_conditions=["have_implication", "have_antecedent"]
    ),
    TacticType(
        name="specialization",
        description="Specialize universal quantifier: ∀x.P(x) → P(c)",
        parameter_schema={"universal_fact": int, "instance": str},
        applicability_conditions=["have_universal", "instance_valid"]
    ),
    TacticType(
        name="generalization",
        description="Generalize existential: P(c) → ∃x.P(x)",
        parameter_schema={"existential_fact": int, "witness": str},
        applicability_conditions=["have_existential", "witness_valid"]
    ),
    TacticType(
        name="simplification",
        description="Simplify complex expressions using algebraic rules",
        parameter_schema={"expression": str, "simplification_rule": int},
        applicability_conditions=["expression_simplifiable", "rule_applicable"]
    ),
    TacticType(
        name="substitution",
        description="Substitute equals for equals",
        parameter_schema={"equality": int, "substitution_target": int, "substitution": str},
        applicability_conditions=["have_equality", "substitution_valid"]
    ),
    TacticType(
        name="backward_chain",
        description="Apply backward chaining: to prove C, find A∧B→C and prove A,B",
        parameter_schema={"goal": str, "backward_rule": int},
        applicability_conditions=["backward_rule_applicable", "subgoals_provable"]
    )
]


class TacticClassifier(nn.Module):
    """Classifies proof state into tactic types."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_tactics: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_tactics = num_tactics
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Tactic classification head
        self.tactic_classifier = nn.Linear(hidden_dim, num_tactics)
        
        # Applicability checker
        self.applicability_checker = nn.Linear(hidden_dim, num_tactics)
    
    def forward(self, proof_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            proof_state: [batch_size, input_dim] proof state representation
        
        Returns:
            tactic_logits: [batch_size, num_tactics] logits for each tactic
            applicability: [batch_size, num_tactics] applicability scores
        """
        features = self.feature_extractor(proof_state)
        
        tactic_logits = self.tactic_classifier(features)
        applicability = torch.sigmoid(self.applicability_checker(features))
        
        return tactic_logits, applicability


class ParameterGenerator(nn.Module):
    """Generates parameters for selected tactics."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, max_params: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_params = max_params
        
        # Parameter generation network
        self.param_network = nn.Sequential(
            nn.Linear(input_dim + 10, hidden_dim),  # +10 for tactic type encoding
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, max_params * 3)  # 3 values per parameter (type, value, confidence)
        )
        
        # Parameter type embeddings
        self.param_type_embeddings = nn.Embedding(10, 10)  # 10 different parameter types
    
    def forward(self, proof_state: torch.Tensor, tactic_logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            proof_state: [batch_size, input_dim] proof state
            tactic_logits: [batch_size, num_tactics] tactic logits
        
        Returns:
            parameters: Dict mapping parameter names to values
        """
        batch_size = proof_state.shape[0]
        
        # Get selected tactic (argmax)
        selected_tactic = torch.argmax(tactic_logits, dim=-1)  # [batch_size]
        
        # Encode tactic type
        tactic_encoding = torch.zeros(batch_size, 10)
        for i, tactic_idx in enumerate(selected_tactic):
            tactic_encoding[i, tactic_idx.item()] = 1.0
        
        # Combine proof state with tactic encoding
        combined_input = torch.cat([proof_state, tactic_encoding], dim=-1)
        
        # Generate parameters
        param_output = self.param_network(combined_input)  # [batch_size, max_params * 3]
        param_output = param_output.view(batch_size, self.max_params, 3)
        
        # Split into parameter components
        param_types = torch.softmax(param_output[:, :, 0], dim=-1)  # Parameter type probabilities
        param_values = torch.sigmoid(param_output[:, :, 1])  # Parameter values [0,1]
        param_confidence = torch.sigmoid(param_output[:, :, 2])  # Confidence scores
        
        return {
            'param_types': param_types,
            'param_values': param_values,
            'param_confidence': param_confidence,
            'selected_tactic': selected_tactic
        }


class TacticDecoder(nn.Module):
    """Complete tactic decoder following SOTA approach."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_tactics: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_tactics = num_tactics
        
        self.tactic_classifier = TacticClassifier(input_dim, hidden_dim, num_tactics)
        self.parameter_generator = ParameterGenerator(input_dim, hidden_dim)
        
        # Tactic type embeddings
        self.tactic_embeddings = nn.Embedding(num_tactics, hidden_dim)
    
    def forward(self, proof_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            proof_state: [batch_size, input_dim] proof state representation
        
        Returns:
            tactic_logits: [batch_size, num_tactics] tactic classification logits
            applicability: [batch_size, num_tactics] applicability scores
            parameters: Dict of generated parameters
        """
        # Step 1: Classify tactic type
        tactic_logits, applicability = self.tactic_classifier(proof_state)
        
        # Step 2: Generate parameters
        parameters = self.parameter_generator(proof_state, tactic_logits)
        
        return {
            'tactic_logits': tactic_logits,
            'applicability': applicability,
            'parameters': parameters
        }
    
    def get_tactic_name(self, tactic_idx: int) -> str:
        """Get human-readable tactic name."""
        if 0 <= tactic_idx < len(TACTIC_TYPES):
            return TACTIC_TYPES[tactic_idx].name
        return f"unknown_tactic_{tactic_idx}"
    
    def get_tactic_description(self, tactic_idx: int) -> str:
        """Get tactic description."""
        if 0 <= tactic_idx < len(TACTIC_TYPES):
            return TACTIC_TYPES[tactic_idx].description
        return "Unknown tactic"


class RuleToTacticMapper:
    """Maps low-level rules to high-level tactics."""
    
    def __init__(self):
        self.rule_to_tactic = {}
        self.tactic_patterns = {}
        self._initialize_mappings()
    
    def _initialize_mappings(self):
        """Initialize rule-to-tactic mappings based on common patterns."""
        # Forward chaining patterns
        self.tactic_patterns['forward_chain'] = [
            'conjunction_elimination',
            'modus_ponens',
            'hypothetical_syllogism'
        ]
        
        # Case splitting patterns
        self.tactic_patterns['case_split'] = [
            'disjunction_elimination',
            'proof_by_cases'
        ]
        
        # Induction patterns
        self.tactic_patterns['induction'] = [
            'mathematical_induction',
            'strong_induction',
            'structural_induction'
        ]
        
        # Contradiction patterns
        self.tactic_patterns['contradiction'] = [
            'proof_by_contradiction',
            'reductio_ad_absurdum'
        ]
        
        # Modus ponens patterns
        self.tactic_patterns['modus_ponens'] = [
            'implication_elimination',
            'conditional_elimination'
        ]
    
    def map_rule_to_tactic(self, rule_name: str) -> Optional[str]:
        """Map a rule name to its corresponding tactic."""
        for tactic, patterns in self.tactic_patterns.items():
            if any(pattern in rule_name.lower() for pattern in patterns):
                return tactic
        return None
    
    def get_tactic_for_rule(self, rule_id: int, rule_name: str) -> Tuple[str, float]:
        """Get tactic type and confidence for a rule."""
        tactic = self.map_rule_to_tactic(rule_name)
        if tactic:
            return tactic, 0.9  # High confidence for pattern matches
        else:
            return "unknown", 0.1  # Low confidence for unknown patterns


class EnhancedRuleDecoder(nn.Module):
    """
    Enhanced rule decoder that combines tactic abstraction with rule prediction.
    
    This follows the SOTA approach:
    1. Predict tactic type (high-level strategy)
    2. Generate parameters (tactic instantiation)
    3. Predict specific rules (low-level implementation)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, 
                 num_tactics: int = 10, num_rules: int = 100):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_tactics = num_tactics
        self.num_rules = num_rules
        
        # Tactic abstraction layer
        self.tactic_decoder = TacticDecoder(input_dim, hidden_dim, num_tactics)
        
        # Rule prediction layer (traditional approach)
        self.rule_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_rules)
        )
        
        # Fusion layer to combine tactic and rule predictions
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim + num_tactics, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_rules)
        )
        
        # Rule-to-tactic mapper
        self.rule_mapper = RuleToTacticMapper()
    
    def forward(self, proof_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            proof_state: [batch_size, input_dim] proof state
        
        Returns:
            Combined predictions from tactic and rule levels
        """
        # Get tactic predictions
        tactic_output = self.tactic_decoder(proof_state)
        tactic_logits = tactic_output['tactic_logits']
        applicability = tactic_output['applicability']
        parameters = tactic_output['parameters']
        
        # Get rule predictions
        rule_logits = self.rule_predictor(proof_state)
        
        # Fuse tactic and rule information
        tactic_features = torch.cat([
            proof_state,
            torch.softmax(tactic_logits, dim=-1)
        ], dim=-1)
        
        fused_rule_logits = self.fusion_layer(tactic_features)
        
        # Combine predictions
        combined_rule_logits = rule_logits + 0.3 * fused_rule_logits
        
        return {
            'rule_logits': combined_rule_logits,
            'tactic_logits': tactic_logits,
            'applicability': applicability,
            'parameters': parameters,
            'fused_logits': fused_rule_logits
        }
    
    def explain_prediction(self, proof_state: torch.Tensor, rule_idx: int) -> Dict[str, Any]:
        """Provide explanation for a rule prediction."""
        with torch.no_grad():
            output = self.forward(proof_state)
            
            # Get tactic explanation
            selected_tactic = torch.argmax(output['tactic_logits'], dim=-1).item()
            tactic_name = self.tactic_decoder.get_tactic_name(selected_tactic)
            tactic_desc = self.tactic_decoder.get_tactic_description(selected_tactic)
            
            # Get rule explanation
            rule_confidence = torch.softmax(output['rule_logits'], dim=-1)[0, rule_idx].item()
            
            return {
                'selected_tactic': tactic_name,
                'tactic_description': tactic_desc,
                'rule_confidence': rule_confidence,
                'explanation': f"Using {tactic_name} tactic: {tactic_desc}"
            }
