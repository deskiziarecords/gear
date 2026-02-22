class LegatoSemiSupervisedLoss(nn.Module):
    """
    Legato-style semi-supervised learning for fault localization.
    Combines consistency regularization with pseudo-labeling.
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.95,  # Not used in threshold-free variant
        gmm_components: int = 2,  # Faulty vs non-faulty
        temperature: float = 0.5
    ):
        super().__init__()
        self.gmm_components = gmm_components
        self.temperature = temperature
        
    def attention_guided_augmentation(
        self, 
        data: Data, 
        attention_weights: torch.Tensor,
        aug_ratio: float = 0.3
    ) -> Data:
        """
        Augment fault-unrelated subgraphs based on attention scores.
        Lower attention = more likely fault-unrelated = augmentable.
        """
        # Identify low-attention edges (fault-unrelated)
        edge_attention = attention_weights.mean(dim=1)  # [num_edges]
        
        # Select bottom 30% edges for augmentation
        k = int(aug_ratio * edge_attention.size(0))
        _, bottom_indices = torch.topk(edge_attention, k, largest=False)
        
        # Create mask
        edge_mask = torch.ones(edge_attention.size(0), dtype=torch.bool)
        edge_mask[bottom_indices] = False
        
        # Return augmented graph (dropped edges)
        aug_data = Data(
            x=data.x,
            edge_index=data.edge_index[:, edge_mask],
            edge_attr=data.edge_attr[edge_mask] if data.edge_attr is not None else None,
            method_ids=data.method_ids
        )
        
        return aug_data
    
    def threshold_free_pseudo_labeling(
        self, 
        predictions: torch.Tensor,
        method_ids: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Threshold-free pseudo-labeling using GMM and BIC.
        predictions: [num_nodes, 1] suspiciousness scores.
        """
        scores = predictions.detach().cpu().numpy().flatten()
        
        # Fit GMM to model distribution of suspiciousness scores
        gmm = GaussianMixture(
            n_components=self.gmm_components,
            covariance_type='full',
            random_state=42
        )
        gmm.fit(scores.reshape(-1, 1))
        
        # Predict probabilities for each component
        probs = gmm.predict_proba(scores.reshape(-1, 1))
        
        # Identify high-confidence pseudo-labels
        # Component with higher mean = faulty class
        faulty_component = np.argmax(gmm.means_)
        
        # Select samples with >90% probability in faulty component
        confident_mask = probs[:, faulty_component] > 0.9
        pseudo_labels = torch.tensor(
            confident_mask, dtype=torch.long, device=predictions.device
        )
        
        # Confidence weights for loss weighting
        confidence_weights = torch.tensor(
            probs[:, faulty_component], 
            dtype=torch.float, 
            device=predictions.device
        )
        
        return pseudo_labels, confidence_weights
    
    def forward(
        self,
        model: nn.Module,
        labeled_data: Data,
        unlabeled_data: Data,
        labels: torch.Tensor,
        alpha: float = 1.0  # Weight for unlabeled loss
    ) -> torch.Tensor:
        """
        Combined supervised + semi-supervised loss.
        """
        # 1. Supervised loss on labeled data
        labeled_out = model(labeled_data)
        supervised_loss = F.binary_cross_entropy_with_logits(
            labeled_out.squeeze(), labels.float()
        )
        
        # 2. Consistency regularization on unlabeled data
        # Original prediction
        ul_pred = model(unlabeled_data)
        
        # Attention-guided augmentation (requires attention weights from model)
        # For simplicity, using random dropout as proxy in base implementation
        aug_data = self._simple_augment(unlabeled_data)
        ul_pred_aug = model(aug_data)
        
        # Consistency loss: predictions should be similar
        consistency_loss = F.mse_loss(
            F.softmax(ul_pred / self.temperature, dim=0),
            F.softmax(ul_pred_aug / self.temperature, dim=0)
        )
        
        # 3. Pseudo-labeling loss
        pseudo_labels, weights = self.threshold_free_pseudo_labeling(
            ul_pred, unlabeled_data.method_ids
        )
        
        # Only compute loss for confident pseudo-labels
        pseudo_loss = F.binary_cross_entropy_with_logits(
            ul_pred.squeeze() * weights,
            pseudo_labels.float() * weights,
            reduction='sum'
        ) / (weights.sum() + 1e-8)
        
        return supervised_loss + alpha * (consistency_loss + pseudo_loss)
    
    def _simple_augment(self, data: Data) -> Data:
        """Simple edge dropout augmentation"""
        edge_mask = torch.rand(data.edge_index.size(1)) > 0.1
        return Data(
            x=data.x,
            edge_index=data.edge_index[:, edge_mask],
            edge_attr=data.edge_attr[edge_mask] if data.edge_attr is not None else None,
            method_ids=data.method_ids
        )
