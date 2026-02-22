# packages/core-algorithms/revr/engine.py

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import z3  # SMT solver for symbolic execution
import ast
import bytecode

class ExecutionDirection(Enum):
    FORWARD = "forward"    # Normal execution trace
    REVERSE = "reverse"    # Backward from failure point
    BIDIRECTIONAL = "bidirectional"  # Meet-in-the-middle

@dataclass
class ProgramState:
    """Program state snapshot at a specific point"""
    line_number: int
    variables: Dict[str, Any]  # Variable values
    path_condition: z3.ExprRef  # Symbolic constraints
    heap_objects: Dict[int, Any]  # Object references
    call_stack: List[str]

@dataclass
class ExecutionTrace:
    direction: ExecutionDirection
    states: List[ProgramState]
    entry_point: int  # Line where analysis started
    exit_point: int   # Line where analysis ended
    constraints: z3.ExprRef  # Accumulated path conditions

@dataclass
class StateDiff:
    """Difference between expected and actual states"""
    variable_name: str
    expected_value: Any
    actual_value: Any
    line_number: int
    severity: float  # 0.0 to 1.0

class REVREngine:
    """
    Reverse Execution Verification & Repair Engine.
    
    Validates patches by comparing simulated forward vs reverse execution traces
    without requiring test case re-execution [^56^].
    """
    
    def __init__(self, source_code: str, language: str = "python"):
        self.source = source_code
        self.language = language
        self.ast_tree = ast.parse(source_code)
        self.cfg = self._build_control_flow_graph()
        self.solver = z3.Solver()
        
    def _build_control_flow_graph(self) -> Dict[int, List[int]]:
        """Build Control Flow Graph for bidirectional traversal"""
        # Simplified CFG construction
        cfg = {}
        for node in ast.walk(self.ast_tree):
            if isinstance(node, (ast.FunctionDef, ast.If, ast.For, ast.While)):
                cfg[node.lineno] = self._get_successors(node)
        return cfg
    
    def _get_successors(self, node: ast.AST) -> List[int]:
        """Get successor nodes in CFG"""
        successors = []
        if isinstance(node, ast.If):
            successors.append(node.body[0].lineno if node.body else node.lineno + 1)
            if node.orelse:
                successors.append(node.orelse[0].lineno)
        return successors
    
    def localize_fault_reverse(
        self, 
        failure_point: int,
        failure_condition: z3.ExprRef,
        observed_state: Dict[str, Any]
    ) -> List[int]:
        """
        Bidirectional fault localization using reverse execution.
        
        Based on BRAFAR's bidirectional refactoring [^52^] and ROSE's
        abstract-interpretation-based flow analysis [^56^].
        
        Args:
            failure_point: Line where failure was observed
            failure_condition: Z3 constraint representing the failure
            observed_state: Actual program state at failure point
        
        Returns:
            List of suspicious line numbers (ranked by likelihood)
        """
        # Phase 1: Reverse execution from failure point
        reverse_trace = self._execute_reverse(
            start_line=failure_point,
            target_condition=failure_condition,
            max_depth=50
        )
        
        # Phase 2: Forward execution from entry points
        entry_points = self._find_entry_points()
        forward_traces = [
            self._execute_forward(entry, failure_point)
            for entry in entry_points
        ]
        
        # Phase 3: Meet-in-the-middle analysis
        suspicious_lines = self._bidirectional_analysis(
            reverse_trace, 
            forward_traces,
            failure_condition
        )
        
        return suspicious_lines
    
    def _execute_reverse(
        self, 
        start_line: int,
        target_condition: z3.ExprRef,
        max_depth: int = 50
    ) -> ExecutionTrace:
        """
        Execute backwards from failure point using symbolic execution.
        
        This is the core "reverse execution" capability - instead of
        running the program forward, we reason backwards from the failure
        to find necessary preconditions.
        """
        states = []
        current_line = start_line
        current_condition = target_condition
        
        for _ in range(max_depth):
            # Get previous statements in CFG (predecessors)
            predecessors = self._get_predecessors(current_line)
            
            for pred_line in predecessors:
                # Symbolically execute the statement backwards
                stmt = self._get_statement(pred_line)
                new_condition = self._symbolic_execute_reverse(
                    stmt, 
                    current_condition
                )
                
                state = ProgramState(
                    line_number=pred_line,
                    variables=self._extract_variables(stmt),
                    path_condition=new_condition,
                    heap_objects={},
                    call_stack=[]
                )
                states.append(state)
                
                # If we reached a valid entry point, stop
                if self._is_entry_point(pred_line):
                    break
            
            current_line = predecessors[0] if predecessors else None
            if not current_line:
                break
        
        return ExecutionTrace(
            direction=ExecutionDirection.REVERSE,
            states=states,
            entry_point=start_line,
            exit_point=states[-1].line_number if states else start_line,
            constraints=target_condition
        )
    
    def _execute_forward(
        self, 
        start_line: int,
        end_line: int
    ) -> ExecutionTrace:
        """Standard forward symbolic execution"""
        states = []
        current_line = start_line
        
        while current_line and current_line <= end_line:
            stmt = self._get_statement(current_line)
            state = self._symbolic_execute_forward(stmt)
            states.append(state)
            current_line = self._get_next_line(current_line)
        
        return ExecutionTrace(
            direction=ExecutionDirection.FORWARD,
            states=states,
            entry_point=start_line,
            exit_point=end_line,
            constraints=z3.BoolVal(True)
        )
    
    def _bidirectional_analysis(
        self,
        reverse_trace: ExecutionTrace,
        forward_traces: List[ExecutionTrace],
        failure_condition: z3.ExprRef
    ) -> List[int]:
        """
        Meet-in-the-middle: Find where forward and reverse traces disagree.
        
        This identifies the "fault frontier" - the exact line where
        the program state diverged from expected behavior.
        """
        suspicious_scores = {}
        
        # Compare reverse trace states with forward trace states
        for rev_state in reverse_trace.states:
            line = rev_state.line_number
            
            # Find matching forward state
            for fwd_trace in forward_traces:
                fwd_state = self._find_matching_state(fwd_trace, line)
                if not fwd_state:
                    continue
                
                # Calculate state divergence
                divergence = self._calculate_divergence(
                    rev_state.path_condition,
                    fwd_state.path_condition,
                    failure_condition
                )
                
                suspicious_scores[line] = suspicious_scores.get(line, 0) + divergence
        
        # Return lines sorted by suspiciousness
        return sorted(suspicious_scores.keys(), 
                     key=lambda x: suspicious_scores[x], 
                     reverse=True)
    
    def validate_patch(
        self,
        original_code: str,
        patched_code: str,
        failure_point: int,
        failure_description: str
    ) -> Tuple[bool, List[StateDiff], float]:
        """
        Validate patch by comparing simulated traces.
        
        Based on ROSE's simulated trace comparison [^56^] and
        UniAPR's on-the-fly validation [^60^].
        
        Returns:
            (is_valid, state_diffs, confidence_score)
        """
        # Generate simulated traces for both versions
        original_trace = self._simulate_trace(
            original_code, 
            failure_point,
            direction=ExecutionDirection.REVERSE
        )
        patched_trace = self._simulate_trace(
            patched_code,
            failure_point,
            direction=ExecutionDirection.FORWARD  # Verify fix works
        )
        
        # Compare traces
        state_diffs = self._compare_traces(original_trace, patched_trace)
        
        # Check if patch resolves the failure condition
        is_valid = self._check_patch_correctness(
            patched_trace, 
            failure_description
        )
        
        confidence = self._calculate_validation_confidence(
            state_diffs, 
            is_valid
        )
        
        return is_valid, state_diffs, confidence
    
    def _simulate_trace(
        self,
        code: str,
        target_point: int,
        direction: ExecutionDirection,
        max_steps: int = 100
    ) -> ExecutionTrace:
        """
        Simulate execution trace without actually running the program.
        
        Uses abstract interpretation and symbolic execution to predict
        program behavior [^56^].
        """
        if direction == ExecutionDirection.REVERSE:
            return self._execute_reverse(target_point, z3.BoolVal(True), max_steps)
        else:
            entry = self._find_entry_points()[0]
            return self._execute_forward(entry, target_point)
    
    def _compare_traces(
        self,
        trace1: ExecutionTrace,
        trace2: ExecutionTrace
    ) -> List[StateDiff]:
        """Find semantic differences between two execution traces"""
        diffs = []
        
        # Align states by line number
        states1 = {s.line_number: s for s in trace1.states}
        states2 = {s.line_number: s for s in trace2.states}
        
        for line in set(states1.keys()) & set(states2.keys()):
            s1, s2 = states1[line], states2[line]
            
            for var in set(s1.variables.keys()) & set(s2.variables.keys()):
                if s1.variables[var] != s2.variables[var]:
                    diffs.append(StateDiff(
                        variable_name=var,
                        expected_value=s1.variables[var],
                        actual_value=s2.variables[var],
                        line_number=line,
                        severity=self._calculate_severity(s1, s2, var)
                    ))
        
        return diffs
    
    def _check_patch_correctness(
        self,
        patched_trace: ExecutionTrace,
        failure_description: str
    ) -> bool:
        """
        Check if patched trace satisfies the failure condition fix.
        
        Uses SMT solver to verify the patch resolves the issue.
        """
        # Parse failure description into constraints
        failure_constraint = self._parse_failure_description(failure_description)
        
        # Check if patched execution violates the failure condition
        self.solver.push()
        self.solver.add(patched_trace.constraints)
        self.solver.add(failure_constraint)
        
        result = self.solver.check()
        self.solver.pop()
        
        # If unsat, the failure condition is no longer reachable (good!)
        return result == z3.unsat
    
    def _calculate_severity(
        self, 
        state1: ProgramState, 
        state2: ProgramState, 
        var: str
    ) -> float:
        """Calculate severity of state divergence"""
        # Simple heuristic: larger difference = higher severity
        val1, val2 = state1.variables[var], state2.variables[var]
        
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            return min(abs(val1 - val2) / max(abs(val1), abs(val2), 1), 1.0)
        
        return 1.0 if val1 != val2 else 0.0
    
    def _calculate_validation_confidence(
        self,
        diffs: List[StateDiff],
        is_valid: bool
    ) -> float:
        """Calculate confidence score for patch validation"""
        if not is_valid:
            return 0.0
        
        # Fewer diffs = higher confidence
        base_confidence = 1.0 / (1 + len(diffs))
        
        # Adjust based on severity
        avg_severity = sum(d.severity for d in diffs) / len(diffs) if diffs else 0
        adjusted_confidence = base_confidence * (1 - avg_severity * 0.5)
        
        return max(adjusted_confidence, 0.1)
    
    # Helper methods (simplified implementations)
    def _get_predecessors(self, line: int) -> List[int]:
        """Get predecessor lines in CFG"""
        preds = []
        for node, succs in self.cfg.items():
            if line in succs:
                preds.append(node)
        return preds if preds else [line - 1]
    
    def _get_statement(self, line: int) -> ast.AST:
        """Get AST node for line"""
        for node in ast.walk(self.ast_tree):
            if hasattr(node, 'lineno') and node.lineno == line:
                return node
        return None
    
    def _is_entry_point(self, line: int) -> bool:
        """Check if line is a function entry point"""
        node = self._get_statement(line)
        return isinstance(node, ast.FunctionDef)
    
    def _find_entry_points(self) -> List[int]:
        """Find all function entry points"""
        return [node.lineno for node in ast.walk(self.ast_tree) 
                if isinstance(node, ast.FunctionDef)]
    
    def _get_next_line(self, current: int) -> Optional[int]:
        """Get next line in forward execution"""
        succs = self.cfg.get(current, [])
        return succs[0] if succs else None
    
    def _extract_variables(self, node: ast.AST) -> Dict[str, Any]:
        """Extract variable names from AST node"""
        vars = {}
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                vars[child.id] = None  # Symbolic value
        return vars
    
    def _symbolic_execute_reverse(self, stmt: ast.AST, 
                                  post_condition: z3.ExprRef) -> z3.ExprRef:
        """Symbolically execute statement backwards"""
        # Simplified: return weakened post-condition
        return post_condition
    
    def _symbolic_execute_forward(self, stmt: ast.AST) -> ProgramState:
        """Symbolically execute statement forwards"""
        return ProgramState(
            line_number=stmt.lineno if hasattr(stmt, 'lineno') else 0,
            variables=self._extract_variables(stmt),
            path_condition=z3.BoolVal(True),
            heap_objects={},
            call_stack=[]
        )
    
    def _find_matching_state(self, trace: ExecutionTrace, line: int) -> Optional[ProgramState]:
        """Find state at specific line in trace"""
        for state in trace.states:
            if state.line_number == line:
                return state
        return None
    
    def _calculate_divergence(self, cond1: z3.ExprRef, 
                              cond2: z3.ExprRef,
                              target: z3.ExprRef) -> float:
        """Calculate divergence between path conditions"""
        # Simplified: use SMT solver to check equivalence
        self.solver.push()
        self.solver.add(z3.Not(cond1 == cond2))
        result = self.solver.check()
        self.solver.pop()
        return 1.0 if result == z3.sat else 0.0
    
    def _parse_failure_description(self, description: str) -> z3.ExprRef:
        """Parse natural language failure into Z3 constraint"""
        # Placeholder: would use NLP + pattern matching
        return z3.BoolVal(True)
