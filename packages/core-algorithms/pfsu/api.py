# packages/core-algorithms/pfsu/api.py

class PFSUEngine:
    """
    Complete PFSU: Probabilistic Fix Synthesis and Understanding
    """
    
    def __init__(
        self,
        probabilistic_model_path: Optional[str] = None,
        llm_model: str = "gpt-4",
        enable_semantic_synthesis: bool = True
    ):
        self.synthesizer = ProbabilisticFixSynthesizer(probabilistic_model_path)
        self.understanding = FixUnderstandingEngine(llm_model)
        self.enable_semantic = enable_semantic_synthesis
    
    def generate_and_explain_fixes(
        self,
        error_report: 'ErrorReport',
        gbfl_rankings: List['RankedMethod'],
        revr_analysis: 'REVRAnalysis',
        top_k: int = 5
    ) -> List[Dict]:
        """
        Complete pipeline: synthesize, rank, and explain patches.
        """
        # Phase 1: Probabilistic Fix Synthesis
        candidates = self.synthesizer.synthesize_patches(
            error_report,
            gbfl_rankings,
            revr_analysis,
            max_patches=50
        )
        
        # Phase 2: Semantic validation with LLM
        enriched_candidates = []
        for candidate in candidates[:top_k * 2]:  # Top candidates for LLM analysis
            # Semantic correctness check
            is_correct, confidence, reasoning = self.understanding.validate_semantic_correctness(
                candidate,
                codebase_context=self._get_context(candidate.location)
            )
            
            # Generate explanation
            explanation = self.understanding.explain_patch(candidate, error_report)
            
            enriched_candidates.append({
                'candidate': candidate,
                'semantic_correctness': is_correct,
                'semantic_confidence': confidence,
                'reasoning': reasoning,
                'explanation': explanation,
                'combined_score': (
                    0.4 * candidate.probability_score +
                    0.3 * candidate.semantic_score +
                    0.3 * confidence
                )
            })
        
        # Sort by combined score and return top_k
        enriched_candidates.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return enriched_candidates[:top_k]
    
    def interactive_repair(
        self,
        error_report: 'ErrorReport',
        user_feedback: str
    ) -> PatchCandidate:
        """
        Interactive repair with user feedback loop.
        """
        # Initial synthesis
        candidates = self.generate_and_explain_fixes(error_report, [], None, top_k=3)
        
        # Incorporate feedback
        if user_feedback:
            # Re-rank based on feedback
            for c in candidates:
                if self._feedback_matches(c, user_feedback):
                    c['combined_score'] *= 1.5  # Boost matching candidates
            
            candidates.sort(key=lambda x: x['combined_score'], reverse=True)
            
            # If no good match, synthesize new candidates
            if candidates[0]['combined_score'] < 0.5:
                new_candidates = self.synthesizer.synthesize_with_constraint(
                    error_report,
                    constraint=user_feedback
                )
                # ... integrate new candidates
        
        return candidates[0]['candidate'] if candidates else None
    
    def _get_context(self, location: str) -> str:
        """Get codebase context for location"""
        # Implementation would fetch surrounding code, imports, etc.
        return ""
    
    def _feedback_matches(self, candidate: Dict, feedback: str) -> bool:
        """Check if candidate matches user feedback"""
        # Simple keyword matching (production: semantic similarity)
        feedback_lower = feedback.lower()
        explanation_lower = candidate['explanation'].lower()
        return any(word in explanation_lower for word in feedback_lower.split())

class PFSUEngine:
    """
    Complete PFSU: Probabilistic Fix Synthesis and Understanding
    """
    
    def __init__(
        self,
        probabilistic_model_path: Optional[str] = None,
        llm_model: str = "gpt-4",
        enable_semantic_synthesis: bool = True
    ):
        self.synthesizer = ProbabilisticFixSynthesizer(probabilistic_model_path)
        self.understanding = FixUnderstandingEngine(llm_model)
        self.enable_semantic = enable_semantic_synthesis
    
    def generate_and_explain_fixes(
        self,
        error_report: 'ErrorReport',
        gbfl_rankings: List['RankedMethod'],
        revr_analysis: 'REVRAnalysis',
        top_k: int = 5
    ) -> List[Dict]:
        """
        Complete pipeline: synthesize, rank, and explain patches.
        """
        # Phase 1: Probabilistic Fix Synthesis
        candidates = self.synthesizer.synthesize_patches(
            error_report,
            gbfl_rankings,
            revr_analysis,
            max_patches=50
        )
        
        # Phase 2: Semantic validation with LLM
        enriched_candidates = []
        for candidate in candidates[:top_k * 2]:  # Top candidates for LLM analysis
            # Semantic correctness check
            is_correct, confidence, reasoning = self.understanding.validate_semantic_correctness(
                candidate,
                codebase_context=self._get_context(candidate.location)
            )
            
            # Generate explanation
            explanation = self.understanding.explain_patch(candidate, error_report)
            
            enriched_candidates.append({
                'candidate': candidate,
                'semantic_correctness': is_correct,
                'semantic_confidence': confidence,
                'reasoning': reasoning,
                'explanation': explanation,
                'combined_score': (
                    0.4 * candidate.probability_score +
                    0.3 * candidate.semantic_score +
                    0.3 * confidence
                )
            })
        
        # Sort by combined score and return top_k
        enriched_candidates.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return enriched_candidates[:top_k]
    
    def interactive_repair(
        self,
        error_report: 'ErrorReport',
        user_feedback: str
    ) -> PatchCandidate:
        """
        Interactive repair with user feedback loop.
        """
        # Initial synthesis
        candidates = self.generate_and_explain_fixes(error_report, [], None, top_k=3)
        
        # Incorporate feedback
        if user_feedback:
            # Re-rank based on feedback
            for c in candidates:
                if self._feedback_matches(c, user_feedback):
                    c['combined_score'] *= 1.5  # Boost matching candidates
            
            candidates.sort(key=lambda x: x['combined_score'], reverse=True)
            
            # If no good match, synthesize new candidates
            if candidates[0]['combined_score'] < 0.5:
                new_candidates = self.synthesizer.synthesize_with_constraint(
                    error_report,
                    constraint=user_feedback
                )
                # ... integrate new candidates
        
        return candidates[0]['candidate'] if candidates else None
    
    def _get_context(self, location: str) -> str:
        """Get codebase context for location"""
        # Implementation would fetch surrounding code, imports, etc.
        return ""
    
    def _feedback_matches(self, candidate: Dict, feedback: str) -> bool:
        """Check if candidate matches user feedback"""
        # Simple keyword matching (production: semantic similarity)
        feedback_lower = feedback.lower()
        explanation_lower = candidate['explanation'].lower()
        return any(word in explanation_lower for word in feedback_lower.split())
