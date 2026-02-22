class DepGraphEncoder(nn.Module):
    """
    GraphSAGE encoder for DepGraph.
    3 layers with residual connections, 44% less GPU memory than GAT [^8^].
    """
    
    def __init__(
        self, 
        in_channels: int = 6,
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Output layer
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        
        # Residual projection (if dimensions don't match)
        self.residual_proj = nn.Linear(in_channels, out_channels) \
            if in_channels != out_channels else None
        
        self.dropout = dropout
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass with residual connections.
        edge_attr optional: GraphSAGE doesn't require edge features by default.
        """
        residual = x
        
        for i, (conv, bn) in enumerate(zip(self.convs[:-1], self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        # Residual connection
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)
        
        return x + residual  # Residual connection for gradient flow
