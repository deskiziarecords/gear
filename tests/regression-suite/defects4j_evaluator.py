# tests/regression-suite/defects4j_evaluator.py

import subprocess
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import tempfile
import shutil

@dataclass
class Defects4JBug:
    project: str  # e.g., 'Lang', 'Math', 'Chart', 'Closure', 'Time'
    bug_id: int   # e.g., 1, 2, 3...
    
    @property
    def full_id(self) -> str:
        return f"{self.project}-{self.bug_id}"

@dataclass
class EvaluationResult:
    bug: Defects4JBug
    exam_score: float  # Percentage of code examined before finding fault
    top_1: bool        # Fault ranked #1
    top_5: bool        # Fault in top 5
    top_10: bool       # Fault in top 10
    first_rank: int    # Position of first faulty element
    total_elements: int
    
    @property
    def mrr(self) -> float:  # Mean Reciprocal Rank
        return 1.0 / self.first_rank if self.first_rank > 0 else 0.0

class Defects4JEvaluator:
    """
    Evaluator for GBFL on Defects4J benchmark [^13^][^18^].
    
    Defects4J v1.2 contains 357 real bugs from 5 projects:
    - Chart: 26 bugs (JFreeChart)
    - Closure: 133 bugs (Google Closure Compiler)
    - Lang: 65 bugs (Apache Commons Lang)
    - Math: 106 bugs (Apache Commons Math)  
    - Time: 27 bugs (Joda-Time)
    """
    
    def __init__(self, defects4j_home: Path):
        self.defects4j_home = defects4j_home
        self.work_dir = Path(tempfile.mkdtemp())
        
        # Validate Defects4J installation
        if not (defects4j_home / "framework" / "bin" / "defects4j").exists():
            raise RuntimeError("Defects4J not found at specified path")
    
    def checkout_bug(self, bug: Defects4JBug) -> Path:
        """Checkout a buggy version of a Defects4J project"""
        bug_dir = self.work_dir / bug.full_id
        bug_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            f"{self.defects4j_home}/framework/bin/defects4j",
            "checkout",
            "-p", bug.project,
            "-v", f"{bug.bug_id}b",  # 'b' for buggy version
            "-w", str(bug_dir)
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return bug_dir
    
    def get_faulty_lines(self, bug: Defects4JBug, bug_dir: Path) -> List[Tuple[str, int]]:
        """
        Get ground truth faulty lines from Defects4J metadata.
        Returns list of (file_path, line_number) tuples.
        """
        # Defects4J stores fault locations in bug metadata
        # For this implementation, we parse the patch file to find modified lines
        metadata_dir = self.defects4j_home / "framework" / "projects" / bug.project / "patches"
        patch_file = metadata_dir / f"{bug.bug_id}.src.patch"
        
        faulty_lines = []
        current_file = None
        
        with open(patch_file) as f:
            for line in f:
                if line.startswith("---"):
                    # Parse source file path
                    parts = line.split()
                    if len(parts) > 1:
                        current_file = parts[1].replace("a/", "")
                elif line.startswith("@@") and current_file:
                    # Parse hunk header: @@ -old_line,old_count +new_line,new_count @@
                    # Extract the line numbers that were modified (deleted or changed)
                    parts = line.split()
                    old_range = parts[1][1:]  # Remove leading '-'
                    start_line = int(old_range.split(',')[0])
                    # Mark this as a potentially faulty line
                    faulty_lines.append((current_file, start_line))
        
        return faulty_lines
    
    def run_tests(self, bug_dir: Path) -> Tuple[List[str], List[str]]:
        """
        Run test suite and return (failing_tests, passing_tests).
        Uses Defects4J test command.
        """
        # Run tests and capture output
        cmd = [
            f"{self.defects4j_home}/framework/bin/defects4j",
            "test",
            "-w", str(bug_dir)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse test results from Defects4J output
        failing = []
        passing = []
        
        # Defects4J outputs test results in specific format
        # This is a simplified parser - production would need robust parsing
        for line in result.stdout.split('\n'):
            if line.startswith('Failing tests:'):
                # Next lines contain failing test names
                pass
            elif '::' in line and not line.startswith(' '):
                # Likely a test case line
                if line.strip() and not line.startswith('-'):
                    test_name = line.strip().split()[0]
                    if 'fail' in line.lower() or 'error' in line.lower():
                        failing.append(test_name)
                    else:
                        passing.append(test_name)
        
        return failing, passing
    
    def evaluate_bug(self, bug: Defects4JBug, gbfl_engine) -> EvaluationResult:
        """
        Evaluate GBFL engine on a single Defects4J bug.
        
        Args:
            bug: Defects4J bug identifier
            gbfl_engine: Your GBFL implementation with rank_elements() method
        
        Returns:
            EvaluationResult with EXAM score and Top-N metrics
        """
        print(f"Evaluating {bug.full_id}...")
        
        # Setup
        bug_dir = self.checkout_bug(bug)
        faulty_lines = self.get_faulty_lines(bug, bug_dir)
        
        if not faulty_lines:
            raise ValueError(f"No faulty lines found for {bug.full_id}")
        
        # Run tests to get coverage spectra
        failing_tests, passing_tests = self.run_tests(bug_dir)
        
        if not failing_tests:
            raise ValueError(f"No failing tests for {bug.full_id} - bug may be already fixed")
        
        # Build graph and calculate spectra
        # This integrates with your GBFL implementation
        from packages.core_algorithms.gbfl.graph_builder import TreeSitterGraphBuilder
        from packages.core_algorithms.gbfl.api import GBFLEngine
        
        graph_builder = TreeSitterGraphBuilder()
        graph = graph_builder.build_graph(str(bug_dir))
        
        # Map tests to coverage (simplified - production uses actual coverage tool)
        # For Defects4J, we typically use Cobertura or JaCoCo for Java
        # For Python prototype, we use trace or coverage.py
        spectra = self._calculate_spectra(graph, failing_tests, passing_tests, bug_dir)
        
        # Rank elements using GBFL
        rankings = gbfl_engine.rank_elements(spectra)
        
        # Calculate metrics
        return self._calculate_metrics(rankings, faulty_lines, bug)
    
    def _calculate_spectra(self, graph, failing_tests, passing_tests, bug_dir) -> Dict[str, Spectrum]:
        """
        Calculate coverage spectra for each method in graph.
        In production, this uses actual coverage data from test execution.
        """
        # Placeholder: In real implementation, instrument tests and collect coverage
        # For now, simulate based on stack traces and static analysis
        spectra = {}
        
        for node_id in graph.nodes():
            if graph.nodes[node_id].get('node_type') == 'method':
                # Simulate: methods in stack trace of failing tests have ef > 0
                # This is a simplified placeholder
                ef = 1 if node_id in str(failing_tests) else 0
                ep = 0 if node_id in str(failing_tests) else 1
                
                spectra[node_id] = Spectrum(
                    ef=ef,
                    ep=ep,
                    nf=len(failing_tests) - ef,
                    np=len(passing_tests) - ep
                )
        
        return spectra
    
    def _calculate_metrics(self, rankings: List[Tuple[str, float]], 
                          faulty_lines: List[Tuple[str, int]],
                          bug: Defects4JBug) -> EvaluationResult:
        """Calculate EXAM score and Top-N metrics"""
        
        # Find first occurrence of any faulty line in rankings
        first_rank = None
        total_elements = len(rankings)
        
        for rank, (elem_id, score) in enumerate(rankings, 1):
            # Check if this element contains a faulty line
            # elem_id format: file_path::class::method or file_path::method
            file_path = elem_id.split('::')[0]
            
            for faulty_file, faulty_line in faulty_lines:
                if faulty_file in file_path:
                    # Check if line is within method range (simplified)
                    first_rank = rank
                    break
            
            if first_rank:
                break
        
        if not first_rank:
            first_rank = total_elements  # Fault not found, penalize
        
        exam_score = (first_rank / total_elements) * 100
        
        return EvaluationResult(
            bug=bug,
            exam_score=exam_score,
            top_1=(first_rank == 1),
            top_5=(first_rank <= 5),
            top_10=(first_rank <= 10),
            first_rank=first_rank,
            total_elements=total_elements
        )
    
    def evaluate_all(self, bugs: List[Defects4JBug], gbfl_engine) -> Dict:
        """Evaluate on all bugs and return aggregate statistics"""
        results = []
        
        for bug in bugs:
            try:
                result = self.evaluate_bug(bug, gbfl_engine)
                results.append(result)
            except Exception as e:
                print(f"Failed to evaluate {bug.full_id}: {e}")
        
        # Calculate aggregate metrics
        total = len(results)
        if total == 0:
            return {"error": "No successful evaluations"}
        
        return {
            "total_bugs": total,
            "mean_exam": sum(r.exam_score for r in results) / total,
            "top_1_rate": sum(r.top_1 for r in results) / total,
            "top_5_rate": sum(r.top_5 for r in results) / total,
            "top_10_rate": sum(r.top_10 for r in results) / total,
            "mrr": sum(r.mrr for r in results) / total,
            "details": [r.__dict__ for r in results]
        }
    
    def cleanup(self):
        """Remove temporary working directory"""
        shutil.rmtree(self.work_dir, ignore_errors=True)


# Example usage script
if __name__ == "__main__":
    # Initialize evaluator
    evaluator = Defects4JEvaluator(Path("/path/to/defects4j"))
    
    # Test on a subset of bugs (start small!)
    test_bugs = [
        Defects4JBug("Lang", 1),
        Defects4JBug("Lang", 2),
        Defects4JBug("Math", 1),
    ]
    
    # Initialize your GBFL engine
    from packages.core_algorithms.gbfl.api import GBFLEngine
    gbfl = GBFLEngine(formula="ochiai")
    
    # Run evaluation
    results = evaluator.evaluate_all(test_bugs, gbfl)
    print(json.dumps(results, indent=2))
    
    evaluator.cleanup()
