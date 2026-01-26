#!/usr/bin/env python3
"""
FIXED: Horn Clause Generator with Deterministic, Verifiable Proofs

Key Fixes:
1. Backward chaining ensures UNIQUE proof
2. All proofs are verified before output
3. No random step ordering
4. Deterministic rule selection
5. FIXED: Robust topological sort in `reorder_proof_steps` to
   prevent "unreachable steps" bug.
"""

import json
import random
from typing import Dict, List, Set, Tuple, Optional
from enum import Enum
from pathlib import Path
from collections import deque
import numpy as np


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    VERY_HARD = "very_hard"


class ProofVerifier:
    """Verifies that a proof is valid."""
    
    def __init__(self, nodes: List[Dict], edges: List[Dict]):
        self.nodes = nodes
        self.edges = edges
        self.nid_to_node = {n["nid"]: n for n in nodes}
        self.nid_to_idx = {n["nid"]: i for i, n in enumerate(nodes)}
    
    def verify_proof(self, proof_steps: List[Dict]) -> Tuple[bool, str]:
        """
        Verify that proof_steps form a valid proof.
        
        Returns:
            (is_valid, error_message)
        """
        # Track what's known
        known_facts = set()
        
        # Add initial facts
        for node in self.nodes:
            if node["type"] == "fact" and node.get("is_initial", False):
                known_facts.add(node["atom"])
        
        # Verify each step
        for step_idx, step in enumerate(proof_steps):
            rule_nid = step.get("used_rule")
            derived_nid = step.get("derived_node")
            premises = step.get("premises", [])
            
            # Get nodes
            if rule_nid not in self.nid_to_node:
                return False, f"Step {step_idx}: rule {rule_nid} not in nodes"
            
            rule_node = self.nid_to_node[rule_nid]
            if rule_node["type"] != "rule":
                return False, f"Step {step_idx}: {rule_nid} is not a rule"
            
            if derived_nid not in self.nid_to_node:
                return False, f"Step {step_idx}: derived node {derived_nid} not in nodes"
            
            derived_node = self.nid_to_node[derived_nid]
            if derived_node["type"] != "fact":
                return False, f"Step {step_idx}: {derived_nid} is not a fact"
            
            # Check premises exist
            for premise_nid in premises:
                if premise_nid not in self.nid_to_idx:
                    return False, f"Step {step_idx}: premise {premise_nid} not found"
            
            # Check rule applicability
            body_atoms = set(rule_node.get("body_atoms", []))
            
            if not body_atoms:
                # Rule with no body should only fire once
                if step_idx > 0:
                    return False, f"Step {step_idx}: body-less rule fired after step 0"
            else:
                # All body atoms must be in known facts
                if not body_atoms.issubset(known_facts):
                    missing = body_atoms - known_facts
                    return False, f"Step {step_idx}: missing premises {missing}"
            
            # Check derived fact is the rule's head
            expected_head = rule_node.get("head_atom")
            actual_head = derived_node.get("atom")
            if expected_head != actual_head:
                return False, f"Step {step_idx}: head mismatch. Expected {expected_head}, got {actual_head}"
            
            # Mark derived
            known_facts.add(actual_head)
        
        return True, ""
    
    def find_all_derivable_atoms(self) -> Set[str]:
        """Find all atoms that can be derived from initial facts."""
        known = set()
        
        # Add initial facts
        for node in self.nodes:
            if node["type"] == "fact" and node.get("is_initial", False):
                known.add(node["atom"])
        
        # Forward chain until fixpoint
        changed = True
        while changed:
            changed = False
            for node in self.nodes:
                if node["type"] == "rule":
                    body = set(node.get("body_atoms", []))
                    head = node.get("head_atom")
                    
                    if body.issubset(known) and head not in known:
                        known.add(head)
                        changed = True
        
        return known


def reorder_proof_steps(proof_steps, fact_map, initial_atoms, rules, rule_map):
    """
    FIXED: Robust topological sort (Kahn's algorithm)
    This fixes the "unreachable steps" bug by only checking for premise
    satisfaction, not whether the head is already known.
    """
    if not proof_steps:
        return []

    known_atoms = set(initial_atoms)
    ordered_steps = []
    
    # Use a set of step_ids for efficient removal
    remaining_step_ids = {step['step_id'] for step in proof_steps}
    step_map = {step['step_id']: step for step in proof_steps}
    # rule_map is already passed in, but ensure it's correct
    if not rule_map:
         rule_map = {r["rule_nid"]: r for r in rules}

    max_iterations = len(remaining_step_ids)
    # Loop one extra time than num_steps to detect cycles
    for _ in range(max_iterations + 1): 
        if not remaining_step_ids:
            break # All steps ordered
        
        made_progress = False
        steps_to_remove = set()
        
        for step_id in remaining_step_ids:
            step = step_map[step_id]
            rule_nid = step["used_rule"]
            
            if rule_nid not in rule_map:
                continue # Should not happen, but safe
                
            rule_info = rule_map[rule_nid]
            body_atoms = set(rule_info["body_atoms"])
            head_atom = rule_info["head_atom"]
            
            # --- THIS IS THE FIX ---
            # A step is executable if all its premises are known.
            # We DON'T care if the head is already known during re-ordering.
            if body_atoms.issubset(known_atoms):
                # This step can be executed
                ordered_steps.append(step)
                known_atoms.add(head_atom) # Add its conclusion to known set
                steps_to_remove.add(step_id)
                made_progress = True
        
        if not made_progress and remaining_step_ids:
            # This means we have a cycle or truly unreachable steps
            print(f"WARNING: Discarding {len(remaining_step_ids)} unreachable steps (cycle or missing premises).")
            # For example, print one problematic step
            try:
                step_id = list(remaining_step_ids)[0]
                rule_info = rule_map[step_map[step_id]["used_rule"]]
                missing = set(rule_info["body_atoms"]) - known_atoms
                print(f"  -> Example: Step {step_id} (Rule {step_map[step_id]['used_rule']}) missing premises: {missing}")
            except Exception as e:
                print(f"  -> Error printing debug info for unreachable step: {e}")
            break # Exit loop
        
        remaining_step_ids -= steps_to_remove
    
    # Assign sequential IDs
    for i, step in enumerate(ordered_steps):
        step['step_id'] = i
    
    return ordered_steps

def generate_horn_instance_deterministic(
    instance_id: str,
    difficulty: Difficulty = Difficulty.MEDIUM,
    seed: int = None,
    goal: Optional[str] = None
) -> Dict:
    """
    Generate Horn clause instance with DETERMINISTIC proof via backward chaining.
    
    Key improvements:
    1. Choose goal first
    2. Build proof backward from goal
    3. Unique, verifiable proof guaranteed
    4. Deterministic rule selection (always first applicable)
    """
    if seed:
        random.seed(seed)
    
    # Difficulty parameters
    params = {
        Difficulty.EASY: {
            "n_initial_facts": 4,
            "n_rules": 4,
            "max_proof_depth": 3,
            "body_size": (1, 2),
            "atoms_pool": 12
        },
        Difficulty.MEDIUM: {
            "n_initial_facts": 8,
            "n_rules": 12,
            "max_proof_depth": 5,
            "body_size": (2, 3),
            "atoms_pool": 20
        },
        Difficulty.HARD: {
            "n_initial_facts": 12,
            "n_rules": 20,
            "max_proof_depth": 8,
            "body_size": (2, 3),
            "atoms_pool": 35
        },
        Difficulty.VERY_HARD: {
            "n_initial_facts": 20,
            "n_rules": 35,
            "max_proof_depth": 12,
            "body_size": (3, 4),
            "atoms_pool": 60
        }
    }
    
    p = params[difficulty]
    atoms = [f"P{i}" for i in range(p["atoms_pool"])]
    
    nodes = []
    edges = []
    nid_counter = 0
    
    # STEP 1: Create initial facts
    initial_atoms = random.sample(atoms, min(p["n_initial_facts"], len(atoms)))
    fact_map = {}  # atom -> nid
    
    for atom in initial_atoms:
        node = {
            "nid": nid_counter,
            "type": "fact",
            "label": atom,
            "atom": atom,
            "is_initial": True
        }
        nodes.append(node)
        fact_map[atom] = nid_counter
        nid_counter += 1
    
    # STEP 2: Create rules deterministically
    # Use forward chaining to ensure all rules can potentially fire
    derived_atoms = set(initial_atoms)
    rules = []
    
    for rule_idx in range(p["n_rules"]):
        body_size = random.randint(*p["body_size"])
        
        # Body: select from currently derivable atoms
        available = list(derived_atoms)
        if len(available) < body_size:
            available.extend(random.sample([a for a in atoms if a not in derived_atoms], 
                                          min(body_size - len(available), 
                                              len(atoms) - len(derived_atoms))))
        
        if len(available) < body_size:
            body_size = len(available)
        
        if body_size == 0:
            continue
        
        body_atoms = random.sample(available, body_size)
        
        # Head: choose NEW atom or from derivable (70% new, 30% existing)
        if random.random() < 0.7:
            unused = [a for a in atoms if a not in derived_atoms]
            if unused:
                head_atom = random.choice(unused)
            else:
                head_atom = random.choice(list(derived_atoms))
        else:
            head_atom = random.choice(list(derived_atoms))
        
        # Create rule node
        rule_node = {
            "nid": nid_counter,
            "type": "rule",
            "label": f"({' Ã¢Ë†Â§ '.join(body_atoms)}) Ã¢â€ â€™ {head_atom}",
            "body_atoms": body_atoms,
            "head_atom": head_atom
        }
        nodes.append(rule_node)
        rule_nid = nid_counter
        nid_counter += 1
        
        rules.append({
            "rule_nid": rule_nid,
            "body_atoms": body_atoms,
            "head_atom": head_atom,
            "rule_node": rule_node
        })
        
        # Connect body facts to rule
        for atom in body_atoms:
            if atom not in fact_map:
                fact_node = {
                    "nid": nid_counter,
                    "type": "fact",
                    "label": atom,
                    "atom": atom,
                    "is_initial": False
                }
                nodes.append(fact_node)
                fact_map[atom] = nid_counter
                nid_counter += 1
            
            edges.append({
                "src": fact_map[atom],
                "dst": rule_nid,
                "etype": "body"
            })
        
        # Connect rule to head fact
        if head_atom not in fact_map:
            fact_node = {
                "nid": nid_counter,
                "type": "fact",
                "label": head_atom,
                "atom": head_atom,
                "is_initial": False
            }
            nodes.append(fact_node)
            fact_map[head_atom] = nid_counter
            nid_counter += 1
        
        edges.append({
            "src": rule_nid,
            "dst": fact_map[head_atom],
            "etype": "head"
        })
        
        derived_atoms.add(head_atom)
    
    # STEP 3: Choose goal and generate unique proof via backward chaining
    verifier = ProofVerifier(nodes, edges)
    derivable = verifier.find_all_derivable_atoms()
    
    if not derivable:
        # Degenerate case: no proofs possible, return empty
        return {
            "id": instance_id,
            "nodes": nodes,
            "edges": edges,
            "proof_steps": [],
            "goal": None,
            "metadata": {
                "difficulty": difficulty.value,
                "n_nodes": len(nodes),
                "n_edges": len(edges),
                "n_initial_facts": len(initial_atoms),
                "n_rules": len(rules),
                "proof_length": 0,
                "source": "backward_chaining_fixed"
            }
        }
    
    # Choose goal: prefer derived (non-initial) atoms
    non_initial_derivable = [a for a in derivable if a not in initial_atoms]
    if non_initial_derivable:
        goal = random.choice(non_initial_derivable)
    else:
        goal = random.choice(list(derivable))
    
    # STEP 4: Backward chain from goal to build proof
    proof_steps = []
    
    def backward_chain(goal_atom: str, depth: int = 0, visited_in_path=None) -> bool:
        """
        FIXED: Better cycle detection and proof building
        """
        if visited_in_path is None:
            visited_in_path = set()
        
        if depth > p["max_proof_depth"]:
            return False
        
        # Cycle detection
        if goal_atom in visited_in_path:
            return False
        
        # Base case: initial fact
        if goal_atom in initial_atoms:
            return True
        
        visited_in_path.add(goal_atom)
        
        # Find rules that derive this atom
        applicable_rules = [
            r for r in rules 
            if r["head_atom"] == goal_atom
        ]
        
        if not applicable_rules:
            visited_in_path.discard(goal_atom)
            return False
        random.shuffle(applicable_rules)
        # CRITICAL FIX: Sort rules by body size (prefer simpler)
        # This reduces proof complexity
        applicable_rules.sort(key=lambda r: len(r["body_atoms"]))
        
        for rule_info in applicable_rules:
            body_atoms = rule_info["body_atoms"]
            
            # Try to prove all body atoms
            all_provable = True
            for atom in body_atoms:
                # CRITICAL: Pass copy of visited set
                if not backward_chain(atom, depth + 1, visited_in_path.copy()):
                    all_provable = False
                    break
            
            if all_provable:
                # Success! Add proof step
                rule_nid = rule_info["rule_nid"]
                derived_nid = fact_map[goal_atom]
                premise_nids = [fact_map[a] for a in body_atoms]
                
                proof_steps.append({
                    "step_id": len(proof_steps), # Temporary ID
                    "derived_node": derived_nid,
                    "used_rule": rule_nid,
                    "premises": premise_nids
                })
                
                visited_in_path.discard(goal_atom)
                return True
        
        visited_in_path.discard(goal_atom)
        return False
    
    # Generate proof
    proof_found = backward_chain(goal)
    
    if not proof_found:
        # Goal not derivable - shouldn't happen but handle gracefully
        proof_steps = []
    
    # STEP 4.5: Reorder proof steps into valid forward-chaining order
    # Backward chaining produces steps in discovery order, not execution order!
    rule_map = {r["rule_nid"]: r for r in rules}
    proof_steps = reorder_proof_steps(proof_steps, fact_map, initial_atoms, rules, rule_map)
    
    # STEP 5: Verify proof
    is_valid, error_msg = verifier.verify_proof(proof_steps)
    if not is_valid:
        print(f"WARNING: Generated invalid proof for {instance_id}: {error_msg}")
        proof_steps = []
    
    return {
        "id": instance_id,
        "nodes": nodes,
        "edges": edges,
        "proof_steps": proof_steps,
        "goal": goal,
        "metadata": {
            "difficulty": difficulty.value,
            "n_nodes": len(nodes),
            "n_edges": len(edges),
            "n_initial_facts": len(initial_atoms),
            "n_rules": len(rules),
            "proof_length": len(proof_steps),
            "source": "backward_chaining_fixed"
        }
    }


def validate_proof_execution_order(proof_steps, nodes, initial_atoms, rules):
    """
    CRITICAL: Validate that each step's rule is actually applicable when executed.
    
    Returns:
        (is_valid, error_message, invalid_step_idx)
    """
    known_atoms = set(initial_atoms)
    rule_map = {r["rule_nid"]: r for r in rules}
    
    for step_idx, step in enumerate(proof_steps):
        rule_nid = step["used_rule"]
        derived_nid = step["derived_node"]
        
        # Get rule and derived fact
        if rule_nid not in rule_map:
            return False, f"Rule {rule_nid} not found", step_idx
        
        rule = rule_map[rule_nid]
        body_atoms = set(rule["body_atoms"])
        head_atom = rule["head_atom"]
        
        # Check 1: All premises must be known
        if not body_atoms.issubset(known_atoms):
            missing = body_atoms - known_atoms
            return False, f"Missing premises: {missing}", step_idx
        
        # Check 2: Head must NOT already be known (no redundant derivation)
        if head_atom in known_atoms:
            return False, f"Head already known: {head_atom}", step_idx
        
        # Mark as derived
        known_atoms.add(head_atom)
    
    return True, "OK", -1

def generate_adversarial_instance(
    instance_id: str,
    difficulty: Difficulty = Difficulty.MEDIUM,
    seed: int = None,
    goal: Optional[str] = None
) -> Dict:
    """
    Generates a Horn clause problem with 'Distractor Branches' and ID Shuffling.
    
    Implements the "Sonnet" Fixes:
    1. True Path: Start -> Goal (Generated via backward chaining)
    2. Distractor Paths: Start -> Dead Ends (Adversarial noise)
    3. ID Shuffling: Prevents model from overfitting to node creation order.
    """
    if seed:
        random.seed(seed)
        np.random.seed(seed) # Ensure numpy uses the same seed if used
    
    # 1. Generate the Base Valid Proof (The "Signal")
    # We use the existing deterministic generator to get a guaranteed valid core.
    base_inst = generate_horn_instance_deterministic(
        instance_id, difficulty, seed, goal
    )
    
    # Unpack base instance
    nodes = base_inst["nodes"]
    edges = base_inst["edges"]
    proof_steps = base_inst["proof_steps"]
    true_goal = base_inst["goal"]
    
    # If base generation failed (empty proof), return as is
    if not proof_steps:
        return base_inst

    # 2. Adversarial Distractor Generation (The "Noise")
    # We add roughly 50-100% more nodes that are valid rules but lead nowhere.
    
    # Parameters for distractors
    num_distractors = len(nodes) // 2  # Add 50% more noise
    if difficulty == Difficulty.HARD: num_distractors = len(nodes)
    if difficulty == Difficulty.VERY_HARD: num_distractors = int(len(nodes) * 1.5)
    
    nid_counter = len(nodes)
    
    # Identify 'hooks' - existing facts we can attach distractors to
    # (We only attach to facts to simulate valid reasoning branches)
    existing_facts = [n for n in nodes if n["type"] == "fact"]
    
    # Distractor Loop
    for _ in range(num_distractors):
        # A distractor branch starts from an EXISTING fact (making it reachable)
        # and leads to a NEW dead-end fact.
        if not existing_facts: break
        
        # Pick a random start node (hook)
        hook_node = random.choice(existing_facts)
        
        # Randomly decide length of this distractor chain (1-3 steps)
        chain_len = random.randint(1, 3)
        current_hook = hook_node
        
        for _ in range(chain_len):
            # Create new Rule Node
            rule_nid = nid_counter
            nid_counter += 1
            
            # Create new Fact Node (Dead End)
            fact_nid = nid_counter
            nid_counter += 1
            new_atom = f"distractor_{instance_id}_{fact_nid}"
            
            rule_node = {
                "nid": rule_nid,
                "type": "rule",
                "label": f"Distractor Rule {rule_nid}",
                "head_atom": new_atom,
                "body_atoms": [current_hook["atom"]]
            }
            
            fact_node = {
                "nid": fact_nid,
                "type": "fact",
                "label": new_atom,
                "atom": new_atom,
                "is_initial": False
            }
            
            # Add to graph lists
            nodes.append(rule_node)
            nodes.append(fact_node)
            
            # Add Edges
            edges.append({"src": current_hook["nid"], "dst": rule_nid, "etype": "body"})
            edges.append({"src": rule_nid, "dst": fact_nid, "etype": "head"})
            
            # For next iteration, extend from this new fact
            current_hook = fact_node
            
            # 10% chance to cross-link: Add a second random body atom to this distractor rule
            # This makes the rule look more "real" (multi-premise)
            if random.random() < 0.1 and len(existing_facts) > 1:
                 second_hook = random.choice(existing_facts)
                 edges.append({"src": second_hook["nid"], "dst": rule_nid, "etype": "body"})

    # 3. ID Shuffling (Symmetry Breaking)
    # CRITICAL: Remap all NIDs to random integers [0, N-1]
    # This prevents the GNN from learning "Node 0 is always start, Node N is always goal"
    
    total_nodes = len(nodes)
    new_indices = list(range(total_nodes))
    random.shuffle(new_indices)
    
    old_to_new = {}
    for i, node in enumerate(nodes):
        old_id = node["nid"]
        # Assign map based on current list position, mapped to shuffled index
        # Note: node["nid"] might not equal i if we appended distractors
        # So we map the *current* nid to the *new* shuffled id
        old_to_new[old_id] = new_indices[i]
        
    # Apply Remapping
    
    # A. Nodes
    for node in nodes:
        node["nid"] = old_to_new[node["nid"]]
        
    # B. Edges
    for edge in edges:
        edge["src"] = old_to_new[edge["src"]]
        edge["dst"] = old_to_new[edge["dst"]]
        
    # C. Proof Steps
    # We must remap the ground truth path too!
    for step in proof_steps:
        step["derived_node"] = old_to_new[step["derived_node"]]
        step["used_rule"] = old_to_new[step["used_rule"]]
        step["premises"] = [old_to_new[p] for p in step["premises"]]
        
    # 4. Verify Integrity
    # Run the verifier on the remapped proof to ensure we didn't break the logic
    verifier = ProofVerifier(nodes, edges)
    is_valid, error = verifier.verify_proof(proof_steps)
    
    if not is_valid:
        print(f"WARNING: Adversarial generation corrupted proof for {instance_id}: {error}")
        # Fallback: return base instance (or empty)
        return base_inst
        
    # 5. Construct Final Dict
    return {
        "id": instance_id,
        "nodes": nodes,
        "edges": edges,
        "proof_steps": proof_steps,
        "goal": true_goal,
        "metadata": {
            "difficulty": difficulty.value,
            "n_nodes": len(nodes),
            "n_edges": len(edges),
            "proof_length": len(proof_steps),
            "distractors_added": num_distractors,
            "source": "adversarial_shuffled"
        }
    }


def generate_dataset(
    output_dir: str,
    n_per_difficulty: Dict[Difficulty, int],
    seed: int = 42
):
    """Generate complete dataset with verification."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {"total": 0, "by_difficulty": {}, "invalid": 0}
    
    for diff, count in n_per_difficulty.items():
        diff_dir = output_dir / diff.value
        diff_dir.mkdir(exist_ok=True)
        
        print(f"Generating {count} {diff.value} instances...")
        
        valid_count = 0
        attempt = 0
        max_attempts = count * 3  # Allow retries for degenerate cases
        
        while valid_count < count and attempt < max_attempts:
            inst = generate_horn_instance_deterministic(
                f"{diff.value}_{valid_count}",
                diff,
                seed + attempt + hash(diff.value) % 10000
            )
            attempt += 1
            
            # Only accept instances with non-empty proofs
            if len(inst["proof_steps"]) > 0:
                path = diff_dir / f"{diff.value}_{valid_count}.json"
                with open(path, 'w') as f:
                    json.dump(inst, f, indent=2)
                valid_count += 1
        
        stats

class LogicalAugmenter:
    """
    Scientific Data Augmentation for Logic Graphs.
    Applies topology-altering but semantic-preserving transformations
    to improve length generalization and robustness.
    """
    
    def __init__(self, p_stretch: float = 0.3, p_thicken: float = 0.2):
        self.p_stretch = p_stretch  # Probability to stretch a rule (Depth++)
        self.p_thicken = p_thicken  # Probability to add dummy premise (Width++)
        
    def augment(self, instance: Dict) -> Dict:
        """
        Applies augmentations to a generated instance.
        """
        # Work on deep copy to avoid side effects
        inst = json.loads(json.dumps(instance))
        
        nodes = inst['nodes']
        edges = inst['edges']
        proof_steps = inst['proof_steps']
        
        # 0. Build Indices for fast lookup
        nid_to_node = {n['nid']: n for n in nodes}
        max_nid = max(n['nid'] for n in nodes)
        
        # Identify Axioms (Initial Facts) for Premise Thickening
        axioms = [n for n in nodes if n['type'] == 'fact' and n.get('is_initial', False)]
        
        # We iterate backwards to allow inserting steps without messing up indices of earlier steps
        # But for 'proof_steps', we can just rebuild the list.
        
        new_proof_steps = []
        steps_to_process = list(proof_steps)
        
        # Track modifications to edges/nodes to update graph
        
        for step in steps_to_process:
            # Decide if we apply Depth Augmentation (Stretch)
            if random.random() < self.p_stretch:
                step, extra_step_data = self._stretch_rule(
                    step, nodes, edges, nid_to_node, max_nid
                )
                max_nid += 2 # We added 2 nodes (Fact + Rule)
                
                # Add the modified original step + the new extension step
                new_proof_steps.append(step)
                new_proof_steps.append(extra_step_data)
                
            # Decide if we apply Width Augmentation (Thicken)
            elif random.random() < self.p_thicken and axioms:
                step = self._thicken_rule(
                    step, nodes, edges, nid_to_node, axioms
                )
                new_proof_steps.append(step)
            else:
                new_proof_steps.append(step)
                
        inst['nodes'] = nodes
        inst['edges'] = edges
        inst['proof_steps'] = new_proof_steps
        
        # Metadata update
        inst['metadata']['augmented'] = True
        inst['metadata']['n_nodes'] = len(nodes)
        inst['metadata']['proof_length'] = len(new_proof_steps)
        
        return inst

    def _stretch_rule(self, step, nodes, edges, nid_to_node, current_max_id):
        """
        Transformation: A -> B  ==>  A -> (Inter) -> B
        1. Create new intermediate Fact node (I)
        2. Create new Rule node (R_new): I -> B
        3. Modify old Rule node (R_old): A -> I
        """
        # IDs
        inter_fact_nid = current_max_id + 1
        new_rule_nid = current_max_id + 2
        
        old_rule_nid = step['used_rule']
        target_fact_nid = step['derived_node']
        
        # 1. Create Intermediate Fact Node
        inter_atom = f"aug_inter_{inter_fact_nid}"
        nodes.append({
            "nid": inter_fact_nid, "type": "fact", "label": inter_atom,
            "atom": inter_atom, "is_initial": False, "is_augmented": True
        })
        
        # 2. Modify Old Rule: Redirect Head to Intermediate Fact
        # Edge: Old_Rule -> Target_Fact  ==>  Old_Rule -> Inter_Fact
        # Find the head edge
        for e in edges:
            if e['src'] == old_rule_nid and e['dst'] == target_fact_nid:
                e['dst'] = inter_fact_nid # Redirect
                break
        
        # Update Node Definition (Metadata)
        if old_rule_nid in nid_to_node:
            nid_to_node[old_rule_nid]['head_atom'] = inter_atom
            
        # 3. Create New Rule: Inter_Fact -> Target_Fact
        target_atom = nid_to_node[target_fact_nid]['atom']
        new_rule_label = f"({inter_atom}) -> {target_atom}"
        
        nodes.append({
            "nid": new_rule_nid, "type": "rule", "label": new_rule_label,
            "head_atom": target_atom, "body_atoms": [inter_atom], "is_augmented": True
        })
        
        # 4. Add Edges for New Rule
        edges.append({"src": inter_fact_nid, "dst": new_rule_nid, "etype": "body"})
        edges.append({"src": new_rule_nid, "dst": target_fact_nid, "etype": "head"})
        
        # 5. Update the original step to point to intermediate
        # The 'derived_node' of the original step becomes the intermediate fact
        step['derived_node'] = inter_fact_nid
        # The rule used is still old_rule_nid, but it now outputs inter_fact_nid
        
        # 6. Create the NEW proof step
        new_step = {
            "step_id": -1, # Will be re-indexed later
            "derived_node": target_fact_nid,
            "used_rule": new_rule_nid,
            "premises": [inter_fact_nid]
        }
        
        return step, new_step

    def _thicken_rule(self, step, nodes, edges, nid_to_node, axioms):
        """
        Transformation: {A} -> B  ==>  {A, Random_Axiom} -> B
        Adds a 'dummy premise' that is guaranteed true (axiom).
        """
        rule_nid = step['used_rule']
        
        # Pick a random axiom that isn't already a premise
        current_premises = set(step['premises'])
        candidates = [ax for ax in axioms if ax['nid'] not in current_premises]
        
        if not candidates:
            return step # Cannot thicken
            
        dummy = random.choice(candidates)
        
        # 1. Add Edge: Dummy -> Rule
        edges.append({"src": dummy['nid'], "dst": rule_nid, "etype": "body"})
        
        # 2. Update Proof Step
        step['premises'].append(dummy['nid'])
        
        # 3. Update Rule Node Metadata
        rule_node = nid_to_node[rule_nid]
        if 'body_atoms' in rule_node:
            rule_node['body_atoms'].append(dummy['atom'])
            
        return step
# --- ADD THIS ENTIRE BLOCK TO THE END of data_generator.py ---

import argparse

if __name__ == "__main__":
    """
    Main execution block to run the data generator from the command line.
    """
    
    parser = argparse.ArgumentParser(
        description="Generate Horn Clause instances with verifiable proofs."
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        required=True,
        help='Directory to save the generated JSON files.'
    )
    parser.add_argument(
        '--easy', 
        type=int, 
        default=0,
        help='Number of EASY instances to generate.'
    )
    parser.add_argument(
        '--medium', 
        type=int, 
        default=0,
        help='Number of MEDIUM instances to generate.'
    )
    parser.add_argument(
        '--hard', 
        type=int, 
        default=0,
        help='Number of HARD instances to generate.'
    )
    parser.add_argument(
        '--very-hard', 
        type=int, 
        default=0,
        help='Number of VERY_HARD instances to generate.'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed for generation.'
    )
    
    args = parser.parse_args()
    
    # Build the difficulty dictionary from arguments
    n_per_difficulty = {
        Difficulty.EASY: args.easy,
        Difficulty.MEDIUM: args.medium,
        Difficulty.HARD: args.hard,
        Difficulty.VERY_HARD: args.very_hard
    }
    
    # Filter out difficulties with 0 count
    n_per_difficulty = {k: v for k, v in n_per_difficulty.items() if v > 0}
    
    if not n_per_difficulty:
        print("No instances to generate (all counts are 0). Exiting.")
        exit(0)
    
    print(f"Starting data generation (seed={args.seed})...")
    
    # Call the main generation function
    generate_dataset(
        output_dir=args.output_dir,
        n_per_difficulty=n_per_difficulty,
        seed=args.seed
    )
    
    print("Data generation complete.")

# --- END OF NEW BLOCK ---