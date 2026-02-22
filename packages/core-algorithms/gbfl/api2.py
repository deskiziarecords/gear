class GBFLEnginePhase2(GBFLEngine):
    """Extended GBFL engine with GNN refinement"""
    
    def __init__(
        self, 
        formula: str = "ochiai",
        use_gnn: bool = True,
        gnn_checkpoint: Optional[str] = None
    ):
        super().__init__(formula)
        self.use_gnn = use_gnn
        
        if use_gnn:
            self.gnn_refiner = GNNRefiner()
            if gnn_checkpoint:
                self.gnn_refiner.load_state_dict(torch.load(gnn_checkpoint))
    
    def localize(self, error_report: ErrorReport) -> List[RankedMethod]:
        """Phase 2: SBFL + GNN Refinement"""
        
        # Phase 1: SBFL baseline
        sbfl_rankings = super().localize(error_report)
        sbfl_scores = {r.method_id: r.suspiciousness_score for r in sbfl_rankings}
        
        if not self.use_gnn or self.gnn_refiner is None:
            return sbfl_rankings
        
        # Phase 2: GNN refinement
        refined_scores = self.gnn_refiner.refine(
            sbfl_scores=sbfl_scores,
            nx_graph=self.graph,
            repo_metadata=self._extract_metadata(),
            labeled_methods=None  # Inference mode
        )
        
        # Re-rank based on refined scores
        combined_scores = {
            mid: 0.6 * refined_scores.get(mid, 0) + 0.4 * sbfl_scores.get(mid, 0)
            for mid in sbfl_scores
        }
        
        # Sort and format
        sorted_methods = sorted(
            combined_scores.items(), 
            key=lambda x: (-x[1], x[0])
        )
        
        return self._format_results(sorted_methods)
    
    def train_gnn(
        self,
        training_bugs: List[Defects4JBug],
        evaluator: Defects4JEvaluator,
        epochs: int = 100,
        lr: float = 0.001
    ):
        """Train GNN on Defects4J bugs using Legato semi-supervised approach"""
        
        optimizer = torch.optim.Adam(self.gnn_refiner.parameters(), lr=lr)
        criterion = LegatoSemiSupervisedLoss()
        
        for epoch in range(epochs):
            total_loss = 0
            
            for bug in training_bugs:
                # Get bug data
                bug_dir = evaluator.checkout_bug(bug)
                
                # Build graph and get SBFL scores
                self.build_graph(str(bug_dir))
                
                # Simulate error report from bug
                failing_tests, _ = evaluator.run_tests(bug_dir)
                error_report = ErrorReport(
                    stack_trace=self._tests_to_traces(failing_tests),
                    error_type="Unknown",
                    error_message=""
                )
                
                # Get SBFL scores
                sbfl_rankings = super().localize(error_report)
                sbfl_scores = {r.method_id: r.suspiciousness_score for r in sbfl_rankings}
                
                # Build DepGraph
                builder = DepGraphBuilder(sbfl_scores, {})
                data = builder.build(self.graph)
                
                # Get ground truth labels (if available)
                faulty_lines = evaluator.get_faulty_lines(bug, bug_dir)
                labels = self._lines_to_labels(data.method_ids, faulty_lines)
                
                # Split into labeled/unlabeled (Legato: only 8% labeled)
                labeled_mask = torch.rand(len(data.method_ids)) < 0.08
                labeled_data = self._mask_data(data, labeled_mask, labels)
                unlabeled_data = self._mask_data(data, ~labeled_mask, None)
                
                # Training step
                loss = self.gnn_refiner.train_step(
                    [labeled_data], 
                    [unlabeled_data], 
                    optimizer, 
                    criterion
                )
                total_loss += loss
            
            print(f"Epoch {epoch}: Loss = {total_loss/len(training_bugs):.4f}")
            
            # Save checkpoint
            if epoch % 10 == 0:
                torch.save(
                    self.gnn_refiner.state_dict(), 
                    f"gnn_checkpoint_epoch_{epoch}.pt"
                )
