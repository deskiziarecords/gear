# packages/core-algorithms/gbfl/ranker/gnn_refiner.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.mixture import GaussianMixture
from dataclasses import dataclass

@dataclass
class DepGraphNode:
    """Simplified DepGraph node representation"""
    method_id: str
    sbfl_score: float
    lines_of_code: int
    cyclomatic_complexity: int
    change_frequency: int  # Historical code churn
    last_change_time: float  # Days since last modification
    is_test: bool = False

@dataclass
class DepGraphEdge:
    edge_type: str  # 'calls', 'contains', 'covers'
    frequency: int  # Call frequency from profiling
    test_correlation: float  # How often covered together in tests

class DepGraphBuilder:
    """
    Build DepGraph: Method-level graph with interprocedural edges.
    70% fewer nodes than statement-level graphs [^8^].
    """
    
    def __init__(self, sbfl_scores: Dict[str, float], repo_metadata: Dict):
        self.sbfl_scores = sbfl_scores
        self.repo_metadata = repo_metadata
    
    def build(self, nx_graph) -> Data:
        """Convert NetworkX graph to PyTorch Geometric Data"""
        
        # Create node index mapping
        method_nodes = [
            n for n, attr in nx_graph.nodes(data=True) 
            if attr.get('node_type') == 'method'
        ]
        node_to_idx = {node: i for i, node in enumerate(method_nodes)}
        
        # Node features (6-dimensional)
        x = []
        for method_id in method_nodes:
            metadata = self.repo_metadata.get(method_id, {})
            
            features = [
                self.sbfl_scores.get(method_id, 0.0),  # SBFL baseline
                metadata.get('loc', 50),  # Lines of code
                metadata.get('complexity', 1),  # Cyclomatic complexity
                metadata.get('churn', 0),  # Change frequency
                metadata.get('days_since_change', 365),  # Recency
                1.0 if metadata.get('is_test', False) else 0.0  # Is test method
            ]
            x.append(features)
        
        # Edge indices and attributes
        edge_index = []
        edge_attr = []
        
        for u, v, attr in nx_graph.edges(data=True):
            if u not in node_to_idx or v not in node_to_idx:
                continue
            
            i, j = node_to_idx[u], node_to_idx[v]
            edge_index.append([i, j])
            
            # Edge features (3-dimensional)
            edge_type_vec = [0, 0, 0]
            edge_type = attr.get('edge_type', 'calls')
            if edge_type == 'calls':
                edge_type_vec[0] = 1
            elif edge_type == 'contains':
                edge_type_vec[1] = 1
            else:  # covers
                edge_type_vec[2] = 1
            
            edge_features = edge_type_vec + [
                attr.get('frequency', 1),
                attr.get('test_correlation', 0.5)
            ]
            edge_attr.append(edge_features)
        
        return Data(
            x=torch.tensor(x, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float),
            method_ids=method_nodes
        )
