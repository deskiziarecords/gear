class GNNRefiner(nn.Module):
    """
    Complete GNN refinement layer combining DepGraph + Legato + GraphSAGE.
    """
    
    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Encoder
        self.encoder = DepGraphEncoder(
            in_channels=6,  # SBFL + code metrics
            hidden_channels=hidden_channels,
            out_channels=64,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Suspiciousness scorer
        self.scorer = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)  # Output: suspiciousness score
        )
        
        # Attention mechanism for edge importance
        self.attention = nn.Sequential(
            nn.Linear(64 * 2 + 5, 32),  # Node features + edge features
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass returning refined suspiciousness scores"""
        
        # Encode nodes
        node_embeddings = self.encoder(data.x, data.edge_index, data.edge_attr)
        
        # Calculate edge attention weights (for Legato augmentation)
        if data.edge_attr is not None:
            edge_embeddings = torch.cat([
                node_embeddings[data.edge_index[0]],
                node_embeddings[data.edge_index[1]],
                data.edge_attr
            ], dim=1)
            edge_attention = self.attention(edge_embeddings)
        else:
            edge_attention = None
        
        # Score nodes
        scores = self.scorer(node_embeddings)
        
        # Normalize to probability distribution (for Legato pseudo-labeling)
        scores = F.softmax(scores / 0.5, dim=0)
        
        return scores, edge_attention, node_embeddings
    
    def refine(
        self, 
        sbfl_scores: Dict[str, float],
        nx_graph,
        repo_metadata: Dict,
        labeled_methods: Optional[Dict[str, bool]] = None
    ) -> Dict[str, float]:
        """
        Refine SBFL scores using GNN.
        
        Args:
            sbfl_scores: Initial SBFL suspiciousness scores
            nx_graph: NetworkX method-level graph
            repo_metadata: Code metrics for each method
            labeled_methods: Optional ground truth for training {method_id: is_faulty}
        
        Returns:
            Refined suspiciousness scores
        """
        self.eval()
        
        # Build graph
        builder = DepGraphBuilder(sbfl_scores, repo_metadata)
        data = builder.build(nx_graph)
        
        with torch.no_grad():
            scores, _, _ = self.forward(data)
        
        # Convert back to dict
        refined_scores = {
            method_id: score.item() 
            for method_id, score in zip(data.method_ids, scores)
        }
        
        return refined_scores
    
    def train_step(
        self,
        labeled_batch: List[Data],
        unlabeled_batch: List[Data],
        optimizer: torch.optim.Optimizer,
        criterion: LegatoSemiSupervisedLoss
    ) -> float:
        """Single training step with semi-supervised learning"""
        
        self.train()
        optimizer.zero_grad()
        
        # Prepare batches
        labeled_data = Batch.from_data_list(labeled_batch)
        unlabeled_data = Batch.from_data_list(unlabeled_batch)
        
        # Extract labels
        labels = labeled_data.y if hasattr(labeled_data, 'y') else None
        
        # Compute loss
        loss = criterion(self, labeled_data, unlabeled_data, labels)
        
        loss.backward()
        optimizer.step()
        
        return loss.item()
