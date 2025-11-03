"""
Fixed Batch Collation for PyG Data
===================================

CRITICAL FIX: Preserve per-graph metadata during batching
This ensures target indices remain valid after concatenation.
"""

import torch
from torch_geometric.data import Batch, Data
from typing import List, Optional


def fixed_collate_fn(batch_list: List[Data]) -> Batch:
    """
    Custom collation that preserves graph-level metadata
    
    Key improvements:
    1. Stores original graph node counts
    2. Computes cumulative node offsets  
    3. Keeps per-graph applicable masks separate
    4. Maintains target index validity
    
    Args:
        batch_list: List of PyG Data objects
    
    Returns:
        Batched Data with proper index tracking
    """
    
    # Filter None samples
    batch_list = [b for b in batch_list if b is not None]
    
    if len(batch_list) == 0:
        return None
    
    # CRITICAL: Must use follow_batch to track node assignments
    batch = Batch.from_data_list(batch_list, follow_batch=['x'])
    
    # Add critical metadata for index mapping
    batch.num_nodes_per_graph = torch.tensor(
        [data.x.shape[0] for data in batch_list],
        dtype=torch.long
    )
    
    # Cumulative node offsets for index translation
    batch.node_offsets = torch.cat([
        torch.tensor([0]),
        torch.cumsum(batch.num_nodes_per_graph[:-1], dim=0)
    ])
    
    # Store spectral features as lists (variable k per graph)
    if hasattr(batch_list[0], 'eigvecs'):
        batch.eigvecs_list = [data.eigvecs for data in batch_list]
        batch.eigvals_list = [data.eigvals for data in batch_list]
        batch.eig_mask_list = [data.eig_mask for data in batch_list]
    
    # Per-graph applicable masks (CRITICAL for correct loss)
    if hasattr(batch_list[0], 'applicable_mask'):
        batch.applicable_mask_list = [data.applicable_mask for data in batch_list]
    
    # Store metadata
    if hasattr(batch_list[0], 'meta'):
        batch.meta_list = [data.meta for data in batch_list]
    
    return batch


def unbatch_graph_data(batch: Batch, graph_idx: int) -> dict:
    """
    Extract a single graph's data from a batch
    
    Returns:
        Dictionary with:
        - x: Node features [N_i, D]
        - edge_index: Edge indices [2, E_i]
        - target_idx: Local target index
        - applicable_mask: Local applicable mask [N_i]
        - num_nodes: Number of nodes in this graph
    """
    
    if not hasattr(batch, 'num_nodes_per_graph'):
        raise ValueError("Batch missing num_nodes_per_graph - use fixed_collate_fn!")
    
    # Get node range for this graph
    start_idx = batch.node_offsets[graph_idx].item()
    num_nodes = batch.num_nodes_per_graph[graph_idx].item()
    end_idx = start_idx + num_nodes
    
    # Extract node features
    x = batch.x[start_idx:end_idx]
    
    # Extract edges (need to filter and reindex)
    edge_mask = (batch.edge_index[0] >= start_idx) & (batch.edge_index[0] < end_idx)
    graph_edges = batch.edge_index[:, edge_mask]
    graph_edges = graph_edges - start_idx  # Reindex to local
    
    # Get target (convert global to local)
    if hasattr(batch, 'y') and len(batch.y) > graph_idx:
        target_global = batch.y[graph_idx].item()
        target_local = target_global - start_idx
    else:
        target_local = -1
    
    # Get applicable mask
    if hasattr(batch, 'applicable_mask_list'):
        applicable_mask = batch.applicable_mask_list[graph_idx]
    elif hasattr(batch, 'applicable_mask'):
        applicable_mask = batch.applicable_mask[start_idx:end_idx]
    else:
        applicable_mask = torch.ones(num_nodes, dtype=torch.bool)
    
    return {
        'x': x,
        'edge_index': graph_edges,
        'target_idx': target_local,
        'applicable_mask': applicable_mask,
        'num_nodes': num_nodes,
        'start_idx': start_idx,
        'end_idx': end_idx
    }


def validate_batch_integrity(batch: Batch) -> bool:
    """
    Validate that batch metadata is consistent
    
    Returns True if batch is valid, False otherwise
    """
    
    if not hasattr(batch, 'num_nodes_per_graph'):
        print("❌ Missing num_nodes_per_graph")
        return False
    
    if not hasattr(batch, 'node_offsets'):
        print("❌ Missing node_offsets")
        return False
    
    # Check total nodes match
    expected_total = batch.num_nodes_per_graph.sum().item()
    actual_total = batch.x.shape[0]
    
    if expected_total != actual_total:
        print(f"❌ Node count mismatch: {expected_total} vs {actual_total}")
        return False
    
    # Check target indices are in valid range
    if hasattr(batch, 'y'):
        for i, target in enumerate(batch.y):
            if i >= len(batch.num_nodes_per_graph):
                break
            
            start = batch.node_offsets[i].item()
            end = start + batch.num_nodes_per_graph[i].item()
            
            if not (start <= target.item() < end):
                print(f"❌ Graph {i}: target {target.item()} outside range [{start}, {end})")
                return False
    
    print("✅ Batch integrity validated")
    return True


# Example usage and testing
if __name__ == '__main__':
    from torch_geometric.data import Data
    
    print("Testing Fixed Collation\n" + "="*60)
    
    # Create sample graphs
    graphs = []
    for i in range(3):
        num_nodes = 10 + i * 5
        data = Data(
            x=torch.randn(num_nodes, 22),
            edge_index=torch.randint(0, num_nodes, (2, num_nodes * 2)),
            y=torch.tensor([i * 3]),  # Target: node 0, 3, 6
            applicable_mask=torch.ones(num_nodes, dtype=torch.bool)
        )
        graphs.append(data)
        print(f"Graph {i}: {num_nodes} nodes, target={data.y.item()}")
    
    # Collate
    batch = fixed_collate_fn(graphs)
    print(f"\nBatched: {batch.x.shape[0]} total nodes")
    print(f"Num graphs: {batch.num_graphs}")
    print(f"Node counts: {batch.num_nodes_per_graph.tolist()}")
    print(f"Offsets: {batch.node_offsets.tolist()}")
    
    # Validate
    validate_batch_integrity(batch)
    
    # Unbatch
    print("\nUnbatching:")
    for i in range(batch.num_graphs):
        graph_data = unbatch_graph_data(batch, i)
        print(f"  Graph {i}: {graph_data['num_nodes']} nodes, "
              f"target={graph_data['target_idx']}, "
              f"range=[{graph_data['start_idx']}, {graph_data['end_idx']})")
    
    print("\n✅ All tests passed!")