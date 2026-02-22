# packages/core-algorithms/gbfl/api.py

class GBFLEngine:
    def __init__(self, repo_path: str):
        self.graph = build_dep_graph(repo_path)  # AST + Call graph
        
    def localize(self, error_report: ErrorReport) -> List[RankedMethod]:
        # 1. Parse stack trace into "virtual test coverage"
        coverage = self.map_trace_to_methods(error_report.stack_trace)
        
        # 2. Apply SBFL formulae
        scores = {}
        for method in self.graph.methods:
            ef = coverage.failing_count(method)
            ep = coverage.passing_count(method)  # From historical data
            scores[method] = ochiai(ef, ep, total_failing, total_passing)
            
        # 3. Graph propagation (DepGraph insight)
        refined_scores = self.propagate_suspiciousness(scores)
        
        # 4. Return ranked list
        return sorted(refined_scores.items(), key=lambda x: x[1], reverse=True)
