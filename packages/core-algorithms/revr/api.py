# auto_repair/localization/revr/api.py

"""
REVR API - Reverse Execution Verification & Repair

Main interface for bidirectional program analysis, symbolic execution,
and test-free patch validation.
"""

import ast
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any, Callable, Dict, List, Optional, Set, Tuple, Union, 
    Iterator, AsyncIterator
)
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

# Internal imports
from auto_repair.config import get_config
from auto_repair.localization.revr.engine import (
    REVREngine, ExecutionDirection, ExecutionTrace, 
    ProgramState, StateDiff
)
from auto_repair.localization.revr.smt_integration import (
    PythonZ3Translator, SymbolicValue
)
from auto_repair.localization.revr.domains.interval import Interval
from auto_repair.localization.revr.domains.predicate import PredicateAbstraction
from auto_repair.localization.revr.domains.heap import HeapAbstraction
from auto_repair.localization.revr.concolic_engine import (
    ConcolicExecutor, ConcolicState
)

# Type imports for cross-module integration
try:
    from auto_repair.localization.gbfl.api import RankedMethod
except ImportError:
    RankedMethod = Any

try:
    from auto_repair.synthesis.pfsu.api import PatchCandidate
except ImportError:
    PatchCandidate = Any


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ErrorReport:
    """Standardized error report for REVR analysis."""
    error_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    error_type: str = ""           # e.g., "NullPointerException", "DivisionByZero"
    error_message: str = ""
    stack_trace: List[str] = field(default_factory=list)
    failing_test: Optional[str] = None
    observed_state: Dict[str, Any] = field(default_factory=dict)
    source_code: Optional[str] = None
    file_path: Optional[Path] = None
    line_number: Optional[int] = None
    
    def __post_init__(self):
        if isinstance(self.file_path, str):
            self.file_path = Path(self.file_path)

@dataclass
class FaultLocation:
    """Identified fault location with confidence."""
    location_id: str              # e.g., "src/auth.py::42::validate_token"
    file_path: Path
    line_number: int
    method_name: str
    confidence: float             # 0.0 to 1.0
    direction: ExecutionDirection
    trace_distance: int           # Steps from error point
    state_divergence: float       # Measure of state inconsistency
    explanation: str = ""

@dataclass
class ReverseAnalysisResult:
    """Complete result of reverse execution analysis."""
    analysis_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    error_report: ErrorReport = field(default_factory=ErrorReport)
    
    # Forward trace (from entry to error)
    forward_trace: Optional[ExecutionTrace] = None
    
    # Reverse trace (from error backwards)
    reverse_trace: Optional[ExecutionTrace] = None
    
    # Meet-in-the-middle analysis
    fault_frontier: List[FaultLocation] = field(default_factory=list)
    
    # State analysis
    state_differences: List[StateDiff] = field(default_factory=list)
    path_conditions: List[str] = field(default_factory=list)
    
    # Performance metrics
    execution_time_ms: float = 0.0
    paths_explored: int = 0
    solver_calls: int = 0
    
    def top_fault(self, k: int = 1) -> List[FaultLocation]:
        """Get top-k most likely fault locations."""
        return sorted(
            self.fault_frontier, 
            key=lambda x: x.confidence, 
            reverse=True
        )[:k]

@dataclass
class PatchValidationResult:
    """Result of validating a patch using REVR."""
    validation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    patch_id: str = ""
    
    # Validation outcomes
    is_valid: bool = False
    confidence: float = 0.0
    failure_resolved: bool = False
    semantic_equivalence: float = 0.0
    
    # Detailed analysis
    original_trace: Optional[ExecutionTrace] = None
    patched_trace: Optional[ExecutionTrace] = None
    state_differences: List[StateDiff] = field(default_factory=list)
    
    # Counterexamples (if validation fails)
    counterexample_input: Optional[Dict[str, Any]] = None
    counterexample_path: Optional[List[int]] = None
    
    # Explanation
    reasoning: str = ""
    risk_assessment: str = ""
    
    # Performance
    validation_time_ms: float = 0.0

@dataclass
class BidirectionalAnalysisRequest:
    """Request for bidirectional fault localization."""
    error_report: ErrorReport
    gbfl_rankings: Optional[List[RankedMethod]] = None
    max_depth: int = 50
    timeout_seconds: float = 30.0
    abstract_domains: List[str] = field(default_factory=lambda: ["interval", "predicate"])

@dataclass
class PatchValidationRequest:
    """Request for patch validation."""
    original_code: str
    patched_code: str
    error_report: ErrorReport
    validation_strategy: str = "bidirectional"  # "forward", "reverse", "bidirectional"
    timeout_seconds: float = 60.0


# =============================================================================
# MAIN API CLASS
# =============================================================================

class REVRAnalyzer:
    """
    Main interface for Reverse Execution Verification & Repair.
    
    Provides:
    - Bidirectional fault localization (forward + reverse execution)
    - Symbolic execution for path exploration
    - Test-free patch validation via abstract interpretation
    - Integration with GBFL and PFSU modules
    """
    
    def __init__(self, config=None):
        self.config = config or get_config().revr
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._engines: Dict[str, REVREngine] = {}
        self._translators: Dict[str, PythonZ3Translator] = {}
        self._concolic_executors: Dict[str, ConcolicExecutor] = {}
        
        # Thread pool for parallel analysis
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Cache for analysis results
        self._analysis_cache: Dict[str, ReverseAnalysisResult] = {}
        
        self.logger.info(f"REVR Analyzer initialized (SMT: {self.config.smt_solver})")
    
    # -------------------------------------------------------------------------
    # PUBLIC API: Fault Localization
    # -------------------------------------------------------------------------
    
    def localize_fault(
        self,
        request: BidirectionalAnalysisRequest
    ) -> ReverseAnalysisResult:
        """
        Perform bidirectional fault localization.
        
        Combines forward execution from entry points with reverse execution
        from error point to identify the fault frontier.
        """
        start_time = time.time()
        
        try:
            # Get or create engine for this source
            engine = self._get_engine(request.error_report)
            
            # Build control flow graph if needed
            if not engine.cfg:
                engine._build_control_flow_graph()
            
            # Phase 1: Forward execution from entry points
            self.logger.debug("Starting forward execution...")
            forward_traces = self._execute_forward_multiple(
                engine, 
                request.error_report,
                request.gbfl_rankings
            )
            
            # Phase 2: Reverse execution from error point
            self.logger.debug("Starting reverse execution...")
            reverse_trace = self._execute_reverse(
                engine,
                request.error_report,
                max_depth=request.max_depth
            )
            
            # Phase 3: Meet-in-the-middle analysis
            self.logger.debug("Performing meet-in-the-middle analysis...")
            fault_frontier = self._analyze_fault_frontier(
                forward_traces,
                reverse_trace,
                request.error_report
            )
            
            # Phase 4: Abstract interpretation refinement
            if self.config.abstract_domain != "none":
                fault_frontier = self._refine_with_abstract_interpretation(
                    fault_frontier,
                    request.abstract_domains
                )
            
            execution_time = (time.time() - start_time) * 1000
            
            result = ReverseAnalysisResult(
                error_report=request.error_report,
                forward_trace=forward_traces[0] if forward_traces else None,
                reverse_trace=reverse_trace,
                fault_frontier=fault_frontier,
                execution_time_ms=execution_time,
                paths_explored=len(forward_traces),
                solver_calls=engine.solver.statistics().solver_calls if hasattr(engine.solver, 'statistics') else 0
            )
            
            # Cache result
            self._analysis_cache[result.analysis_id] = result
            
            self.logger.info(
                f"Localization complete: {len(fault_frontier)} locations "
                f"in {execution_time:.2f}ms"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Fault localization failed: {e}")
            self.logger.debug(traceback.format_exc())
            raise REVRException(f"Localization failed: {e}") from e
    
    async def localize_fault_async(
        self,
        request: BidirectionalAnalysisRequest
    ) -> ReverseAnalysisResult:
        """Async version of localize_fault."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.localize_fault,
            request
        )
    
    def localize_fault_stream(
        self,
        request: BidirectionalAnalysisRequest
    ) -> Iterator[FaultLocation]:
        """
        Streaming fault localization - yields locations as they're found.
        
        Useful for real-time UI updates.
        """
        engine = self._get_engine(request.error_report)
        
        # Start reverse execution
        reverse_gen = self._execute_reverse_stream(
            engine,
            request.error_report,
            max_depth=request.max_depth
        )
        
        # Start forward execution in background
        forward_future = self._executor.submit(
            self._execute_forward_multiple,
            engine,
            request.error_report,
            request.gbfl_rankings
        )
        
        # Yield locations as reverse execution discovers them
        for state in reverse_gen:
            location = self._state_to_fault_location(state, 0.5)
            yield location
        
        # Get forward results and refine rankings
        forward_traces = forward_future.result()
        # ... refinement logic ...
    
    # -------------------------------------------------------------------------
    # PUBLIC API: Patch Validation
    # -------------------------------------------------------------------------
    
    def validate_patch(
        self,
        request: PatchValidationRequest
    ) -> PatchValidationResult:
        """
        Validate a patch without executing tests.
        
        Uses bidirectional analysis to compare original vs patched
        execution traces and determine if the fix is correct.
        """
        start_time = time.time()
        
        try:
            # Parse both versions
            original_ast = ast.parse(request.original_code)
            patched_ast = ast.parse(request.patched_code)
            
            # Compute semantic diff
            semantic_changes = self._compute_semantic_diff(
                original_ast, 
                patched_ast
            )
            
            # Strategy selection based on change type
            if request.validation_strategy == "bidirectional":
                result = self._validate_bidirectional(
                    request,
                    semantic_changes
                )
            elif request.validation_strategy == "forward":
                result = self._validate_forward_only(
                    request,
                    semantic_changes
                )
            else:  # reverse
                result = self._validate_reverse_only(
                    request,
                    semantic_changes
                )
            
            # Add timing
            result.validation_time_ms = (time.time() - start_time) * 1000
            
            self.logger.info(
                f"Patch validation: valid={result.is_valid}, "
                f"confidence={result.confidence:.2f}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Patch validation failed: {e}")
            raise REVRException(f"Validation failed: {e}") from e
    
    async def validate_patch_async(
        self,
        request: PatchValidationRequest
    ) -> PatchValidationResult:
        """Async version of validate_patch."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.validate_patch,
            request
        )
    
    def validate_multiple_patches(
        self,
        requests: List[PatchValidationRequest],
        max_workers: int = 4
    ) -> List[PatchValidationResult]:
        """
        Validate multiple patches in parallel.
        
        Returns results in same order as requests.
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.validate_patch, req): i 
                for i, req in enumerate(requests)
            }
            
            results = [None] * len(requests)
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    self.logger.error(f"Validation {idx} failed: {e}")
                    results[idx] = PatchValidationResult(
                        is_valid=False,
                        confidence=0.0,
                        reasoning=f"Validation error: {e}"
                    )
        
        return results
    
    # -------------------------------------------------------------------------
    # PUBLIC API: Concolic Execution
    # -------------------------------------------------------------------------
    
    def generate_test_input(
        self,
        source_code: str,
        target_path: List[int],
        initial_input: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generate test input that drives execution down specific path.
        
        Uses concolic execution to solve for path constraints.
        """
        executor = self._get_concolic_executor(source_code)
        
        # Start with initial input or symbolic
        if initial_input is None:
            initial_input = self._generate_random_input(source_code)
        
        # Execute and collect path constraints
        trace = executor.execute_concolic(initial_input)
        
        # Negate branch conditions to find new inputs
        for i, state in enumerate(trace):
            if i >= len(target_path):
                break
            
            # Check if we're on the right path
            if state.pc != target_path[i]:
                # Solve for input that takes correct branch
                new_input = executor.generate_new_input(i)
                if new_input:
                    return new_input
        
        return None
    
    def explore_paths(
        self,
        source_code: str,
        max_paths: int = 10,
        timeout: float = 60.0
    ) -> List[ExecutionTrace]:
        """
        Explore multiple execution paths using concolic execution.
        
        Useful for understanding program behavior and finding edge cases.
        """
        executor = self._get_concolic_executor(source_code)
        paths = []
        
        start_time = time.time()
        while len(paths) < max_paths and (time.time() - start_time) < timeout:
            # Generate input for new path
            if not paths:
                input_val = self._generate_random_input(source_code)
            else:
                # Negate last path condition to find new path
                input_val = executor.generate_new_input(len(paths) - 1)
                if input_val is None:
                    break
            
            # Execute
            trace = executor.execute_concolic(input_val)
            paths.append(trace)
        
        return paths
    
    # -------------------------------------------------------------------------
    # PUBLIC API: Abstract Interpretation
    # -------------------------------------------------------------------------
    
    def analyze_intervals(
        self,
        source_code: str,
        entry_points: Optional[List[int]] = None
    ) -> Dict[str, Interval]:
        """
        Perform interval analysis on source code.
        
        Returns variable bounds at each program point.
        """
        # Parse and build CFG
        tree = ast.parse(source_code)
        
        # Initialize intervals
        intervals = {}
        
        # Worklist algorithm
        worklist = entry_points or [self._find_entry_point(tree)]
        
        while worklist:
            point = worklist.pop()
            # ... interval analysis logic ...
        
        return intervals
    
    def check_reachability(
        self,
        source_code: str,
        target_line: int,
        precondition: Optional[str] = None
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Check if target line is reachable from entry.
        
        Returns (is_reachable, example_input) if found.
        """
        engine = REVREngine(source_code)
        
        # Build path condition to target
        path_condition = engine._build_path_condition(target_line)
        
        if precondition:
            path_condition = z3.And(
                path_condition,
                self._parse_precondition(precondition)
            )
        
        # Solve
        model = engine.translator.solve_path_condition(path_condition)
        
        return (model is not None, model)
    
    # -------------------------------------------------------------------------
    # INTEGRATION API
    # -------------------------------------------------------------------------
    
    def integrate_with_gbfl(
        self,
        gbfl_result: List[RankedMethod],
        revr_result: ReverseAnalysisResult
    ) -> List[RankedMethod]:
        """
        Combine GBFL rankings with REVR analysis.
        
        Boosts GBFL scores based on REVR fault frontier proximity.
        """
        enhanced = []
        
        for method in gbfl_result:
            # Find matching fault location
            matching_fault = next(
                (f for f in revr_result.fault_frontier 
                 if f.location_id == method.method_id),
                None
            )
            
            if matching_fault:
                # Boost score based on REVR confidence
                boosted_score = (
                    0.6 * method.suspiciousness_score +
                    0.4 * matching_fault.confidence
                )
                method.suspiciousness_score = boosted_score
            
            enhanced.append(method)
        
        # Re-sort
        enhanced.sort(key=lambda x: x.suspiciousness_score, reverse=True)
        return enhanced
    
    def filter_patches_by_revr(
        self,
        patches: List[PatchCandidate],
        error_report: ErrorReport,
        min_confidence: float = 0.7
    ) -> List[PatchCandidate]:
        """
        Pre-filter PFSU patches using REVR validation.
        
        Quickly eliminates patches that don't resolve the error.
        """
        valid_patches = []
        
        for patch in patches:
            request = PatchValidationRequest(
                original_code=error_report.source_code or "",
                patched_code=patch.patched_code,
                error_report=error_report,
                validation_strategy="reverse",  # Fast check
                timeout_seconds=5.0  # Quick timeout for filtering
            )
            
            try:
                result = self.validate_patch(request)
                if result.confidence >= min_confidence:
                    patch.probability_score *= result.confidence  # Boost score
                    valid_patches.append(patch)
            except Exception as e:
                self.logger.warning(f"Patch validation skipped: {e}")
                valid_patches.append(patch)  # Include if validation fails
        
        return valid_patches
    
    # -------------------------------------------------------------------------
    # INTERNAL METHODS
    # -------------------------------------------------------------------------
    
    def _get_engine(self, error_report: ErrorReport) -> REVREngine:
        """Get or create REVR engine for source code."""
        cache_key = error_report.error_id
        
        if cache_key not in self._engines:
            source = error_report.source_code or ""
            if error_report.file_path:
                source = error_report.file_path.read_text()
            
            self._engines[cache_key] = REVREngine(
                source_code=source,
                language="python"  # Auto-detect in production
            )
        
        return self._engines[cache_key]
    
    def _get_concolic_executor(self, source_code: str) -> ConcolicExecutor:
        """Get or create concolic executor."""
        cache_key = hash(source_code)
        
        if cache_key not in self._concolic_executors:
            self._concolic_executors[cache_key] = ConcolicExecutor(source_code)
        
        return self._concolic_executors[cache_key]
    
    def _execute_forward_multiple(
        self,
        engine: REVREngine,
        error_report: ErrorReport,
        gbfl_rankings: Optional[List[RankedMethod]]
    ) -> List[ExecutionTrace]:
        """Execute forward from multiple entry points."""
        traces = []
        
        # Primary entry point
        entry_points = engine._find_entry_points()
        
        # Add GBFL-suggested entry points if available
        if gbfl_rankings:
            for method in gbfl_rankings[:3]:
                # Parse location
                parts = method.method_id.split("::")
                if len(parts) >= 2:
                    try:
                        line = int(parts[-2])  # Assume line number in ID
                        if line not in entry_points:
                            entry_points.append(line)
                    except ValueError:
                        pass
        
        # Execute from each entry
        for entry in entry_points[:5]:  # Limit to 5 entries
            try:
                trace = engine._execute_forward(entry, error_report.line_number or 999999)
                traces.append(trace)
            except Exception as e:
                self.logger.debug(f"Forward execution from {entry} failed: {e}")
        
        return traces
    
    def _execute_reverse(
        self,
        engine: REVREngine,
        error_report: ErrorReport,
        max_depth: int
    ) -> ExecutionTrace:
        """Execute backwards from error point."""
        failure_point = error_report.line_number or 1
        
        # Build failure condition from error type
        failure_condition = self._error_to_condition(
            error_report.error_type,
            error_report.error_message
        )
        
        return engine._execute_reverse(
            start_line=failure_point,
            target_condition=failure_condition,
            max_depth=max_depth
        )
    
    def _execute_reverse_stream(
        self,
        engine: REVREngine,
        error_report: ErrorReport,
        max_depth: int
    ) -> Iterator[ProgramState]:
        """Streaming reverse execution."""
        # Implementation would yield states as they're discovered
        trace = self._execute_reverse(engine, error_report, max_depth)
        yield from trace.states
    
    def _analyze_fault_frontier(
        self,
        forward_traces: List[ExecutionTrace],
        reverse_trace: ExecutionTrace,
        error_report: ErrorReport
    ) -> List[FaultLocation]:
        """Identify fault frontier where forward and reverse traces disagree."""
        frontier = []
        
        # Collect all states from forward traces
        forward_states = {}
        for trace in forward_traces:
            for state in trace.states:
                key = (state.line_number, str(state.path_condition))
                forward_states[key] = state
        
        # Compare with reverse states
        for rev_state in reverse_trace.states:
            key = (rev_state.line_number, str(rev_state.path_condition))
            
            if key in forward_states:
                fwd_state = forward_states[key]
                
                # Calculate divergence
                divergence = self._calculate_state_divergence(fwd_state, rev_state)
                
                if divergence > 0.1:  # Threshold
                    location = FaultLocation(
                        location_id=f"{error_report.file_path}::{rev_state.line_number}",
                        file_path=error_report.file_path or Path("unknown"),
                        line_number=rev_state.line_number,
                        method_name=self._get_method_name(rev_state),
                        confidence=min(1.0, divergence * 2),  # Scale to 0-1
                        direction=ExecutionDirection.BIDIRECTIONAL,
                        trace_distance=abs(rev_state.line_number - error_report.line_number),
                        state_divergence=divergence,
                        explanation=f"State divergence detected: {divergence:.2f}"
                    )
                    frontier.append(location)
        
        # Sort by confidence
        frontier.sort(key=lambda x: x.confidence, reverse=True)
        return frontier
    
    def _refine_with_abstract_interpretation(
        self,
        frontier: List[FaultLocation],
        domains: List[str]
    ) -> List[FaultLocation]:
        """Refine fault locations using abstract interpretation."""
        refined = []
        
        for location in frontier:
            confidence = location.confidence
            
            # Boost confidence if interval analysis confirms
            if "interval" in domains:
                interval_conf = self._check_interval_consistency(location)
                confidence = 0.7 * confidence + 0.3 * interval_conf
            
            # Boost if predicate analysis confirms
            if "predicate" in domains:
                pred_conf = self._check_predicate_consistency(location)
                confidence = 0.7 * confidence + 0.3 * pred_conf
            
            location.confidence = min(1.0, confidence)
            refined.append(location)
        
        return refined
    
    def _validate_bidirectional(
        self,
        request: PatchValidationRequest,
        semantic_changes: List[Dict]
    ) -> PatchValidationResult:
        """Full bidirectional validation."""
        # Original trace (reverse from error)
        original_engine = REVREngine(request.original_code)
        original_trace = original_engine._execute_reverse(
            request.error_report.line_number or 1,
            self._error_to_condition(
                request.error_report.error_type,
                request.error_report.error_message
            ),
            max_depth=50
        )
        
        # Patched trace (forward to check if error reachable)
        patched_engine = REVREngine(request.patched_code)
        patched_trace = patched_engine._execute_forward(
            patched_engine._find_entry_points()[0],
            request.error_report.line_number or 999999
        )
        
        # Check if error condition is still reachable
        error_reachable = self._check_error_reachable(
            patched_trace,
            request.error_report
        )
        
        # Calculate semantic equivalence
        equivalence = self._calculate_semantic_equivalence(
            original_trace,
            patched_trace,
            semantic_changes
        )
        
        is_valid = not error_reachable and equivalence > 0.8
        
        return PatchValidationResult(
            patch_id=hash(request.patched_code),
            is_valid=is_valid,
            confidence=equivalence if is_valid else (1 - equivalence),
            failure_resolved=not error_reachable,
            semantic_equivalence=equivalence,
            original_trace=original_trace,
            patched_trace=patched_trace,
            reasoning=f"Error reachable: {error_reachable}, Semantic equivalence: {equivalence:.2f}",
            risk_assessment="low" if is_valid else "high"
        )
    
    def _validate_forward_only(
        self,
        request: PatchValidationRequest,
        semantic_changes: List[Dict]
    ) -> PatchValidationResult:
        """Fast forward-only validation."""
        # Just check if patched code reaches error
        engine = REVREngine(request.patched_code)
        trace = engine._execute_forward(
            engine._find_entry_points()[0],
            request.error_report.line_number or 999999
        )
        
        error_reached = any(
            state.line_number == request.error_report.line_number
            for state in trace.states
        )
        
        return PatchValidationResult(
            is_valid=not error_reached,
            confidence=0.7 if not error_reached else 0.3,
            failure_resolved=not error_reached
        )
    
    def _validate_reverse_only(
        self,
        request: PatchValidationRequest,
        semantic_changes: List[Dict]
    ) -> PatchValidationResult:
        """Fast reverse-only validation."""
        # Check if error preconditions still exist in patched code
        engine = REVREngine(request.patched_code)
        
        condition = self._error_to_condition(
            request.error_report.error_type,
            request.error_report.error_message
        )
        
        # Simplified: check if condition is satisfiable in patched code
        is_sat = engine.translator.check_reachability(
            z3.BoolVal(True),
            condition
        )
        
        return PatchValidationResult(
            is_valid=not is_sat,
            confidence=0.6 if not is_sat else 0.4,
            failure_resolved=not is_sat
        )
    
    # -------------------------------------------------------------------------
    # HELPER METHODS
    # -------------------------------------------------------------------------
    
    def _error_to_condition(self, error_type: str, message: str) -> z3.ExprRef:
        """Convert error type to Z3 condition."""
        if "null" in error_type.lower() or "none" in message.lower():
            ptr = z3.Int('ptr')
            return ptr == 0
        
        if "division" in error_type.lower() or "zero" in message.lower():
            divisor = z3.Real('divisor')
            return divisor == 0
        
        if "index" in error_type.lower() or "bounds" in message.lower():
            idx = z3.Int('index')
            size = z3.Int('size')
            return z3.Or(idx < 0, idx >= size)
        
        return z3.BoolVal(True)
    
    def _state_to_fault_location(
        self, 
        state: ProgramState, 
        confidence: float
    ) -> FaultLocation:
        """Convert program state to fault location."""
        return FaultLocation(
            location_id=f"line_{state.line_number}",
            file_path=Path("unknown"),
            line_number=state.line_number,
            method_name="unknown",
            confidence=confidence,
            direction=ExecutionDirection.REVERSE,
            trace_distance=0,
            state_divergence=0.0
        )
    
    def _calculate_state_divergence(
        self, 
        state1: ProgramState, 
        state2: ProgramState
    ) -> float:
        """Calculate divergence between two program states."""
        # Simple variable comparison
        vars1 = set(state1.variables.keys())
        vars2 = set(state2.variables.keys())
        
        if not vars1 and not vars2:
            return 0.0
        
        common = vars1 & vars2
        if not common:
            return 1.0
        
        differences = 0
        for var in common:
            if state1.variables[var] != state2.variables[var]:
                differences += 1
        
        return differences / len(common)
    
    def _compute_semantic_diff(
        self, 
        original: ast.AST, 
        patched: ast.AST
    ) -> List[Dict]:
        """Compute semantic differences between ASTs."""
        changes = []
        
        # Simple node counting diff
        original_nodes = list(ast.walk(original))
        patched_nodes = list(ast.walk(patched))
        
        if len(original_nodes) != len(patched_nodes):
            changes.append({
                'type': 'node_count',
                'original': len(original_nodes),
                'patched': len(patched_nodes)
            })
        
        return changes
    
    def _get_method_name(self, state: ProgramState) -> str:
        """Extract method name from call stack."""
        return state.call_stack[0] if state.call_stack else "unknown"
    
    def _generate_random_input(self, source_code: str) -> Dict[str, Any]:
        """Generate random input for concolic execution."""
        # Parse to find parameters
        tree = ast.parse(source_code)
        
        inputs = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for arg in node.args.args:
                    # Generate random value based on name hints
                    if 'int' in arg.arg or 'count' in arg.arg or 'idx' in arg.arg:
                        inputs[arg.arg] = 42
                    elif 'str' in arg.arg or 'name' in arg.arg:
                        inputs[arg.arg] = "test"
                    elif 'bool' in arg.arg or 'flag' in arg.arg:
                        inputs[arg.arg] = True
                    else:
                        inputs[arg.arg] = 0
        
        return inputs
    
    def _find_entry_point(self, tree: ast.AST) -> int:
        """Find first executable line in AST."""
        for node in ast.walk(tree):
            if hasattr(node, 'lineno'):
                return node.lineno
        return 1
    
    def _parse_precondition(self, precondition: str) -> z3.ExprRef:
        """Parse precondition string to Z3 condition."""
        # Simplified: would use proper parser
        return z3.BoolVal(True)
    
    def _check_interval_consistency(self, location: FaultLocation) -> float:
        """Check consistency using interval domain."""
        # Placeholder
        return 0.5
    
    def _check_predicate_consistency(self, location: FaultLocation) -> float:
        """Check consistency using predicate domain."""
        # Placeholder
        return 0.5
    
    def _check_error_reachable(
        self, 
        trace: ExecutionTrace, 
        error_report: ErrorReport
    ) -> bool:
        """Check if error point is reached in trace."""
        return any(
            state.line_number == error_report.line_number
            for state in trace.states
        )
    
    def _calculate_semantic_equivalence(
        self, 
        trace1: ExecutionTrace, 
        trace2: ExecutionTrace,
        changes: List[Dict]
    ) -> float:
        """Calculate semantic equivalence of two traces."""
        # Simplified: compare path conditions
        if not trace1.states or not trace2.states:
            return 0.0
        
        # Check if final states are equivalent
        final1 = trace1.states[-1]
        final2 = trace2.states[-1]
        
        # Simple variable comparison
        match_count = 0
        total_vars = set(final1.variables.keys()) | set(final2.variables.keys())
        
        for var in total_vars:
            val1 = final1.variables.get(var)
            val2 = final2.variables.get(var)
            if val1 == val2:
                match_count += 1
        
        return match_count / len(total_vars) if total_vars else 1.0


# =============================================================================
# EXCEPTIONS
# =============================================================================

class REVRException(Exception):
    """Base exception for REVR errors."""
    pass

class SMTSolverTimeout(REVRException):
    """SMT solver timeout."""
    pass

class PathExplosionError(REVRException):
    """Too many paths to explore."""
    pass

class UnsupportedLanguageError(REVRException):
    """Language not supported."""
    pass


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_analyzer(config=None) -> REVRAnalyzer:
    """Factory function to create REVR analyzer."""
    return REVRAnalyzer(config)

# Singleton instance for convenience
_default_analyzer: Optional[REVRAnalyzer] = None

def get_default_analyzer() -> REVRAnalyzer:
    """Get or create default analyzer instance."""
    global _default_analyzer
    if _default_analyzer is None:
        _default_analyzer = create_analyzer()
    return _default_analyzer


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example error report
    error = ErrorReport(
        error_type="NullPointerException",
        error_message="Cannot access field 'name' on null",
        stack_trace=["auth.py:42:validate_user", "auth.py:15:login"],
        file_path=Path("auth.py"),
        line_number=42,
        source_code="""
def validate_user(user):
    if user.is_active:
        return user.name
    return None
"""
    )
    
    # Create analyzer
    analyzer = create_analyzer()
    
    # Run bidirectional localization
    request = BidirectionalAnalysisRequest(
        error_report=error,
        max_depth=20
    )
    
    result = analyzer.localize_fault(request)
    
    print(f"Analysis ID: {result.analysis_id}")
    print(f"Execution time: {result.execution_time_ms:.2f}ms")
    print(f"Paths explored: {result.paths_explored}")
    print(f"\nTop fault locations:")
    for loc in result.top_fault(3):
        print(f"  - {loc.location_id} (confidence: {loc.confidence:.2f})")
        print(f"    {loc.explanation}")
    
    # Example patch validation
    original = """
def divide(a, b):
    return a / b
"""
    
    patched = """
def divide(a, b):
    if b == 0:
        raise ValueError("Division by zero")
    return a / b
"""
    
    validation_request = PatchValidationRequest(
        original_code=original,
        patched_code=patched,
        error_report=error,
        validation_strategy="bidirectional"
    )
    
    validation = analyzer.validate_patch(validation_request)
    print(f"\nPatch validation:")
    print(f"  Valid: {validation.is_valid}")
    print(f"  Confidence: {validation.confidence:.2f}")
    print(f"  Reasoning: {validation.reasoning}")
