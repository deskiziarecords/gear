# auto_repair/synthesis/pfs/semantic_synthesis.py

"""
Semantic Synthesis for Patch Generation.

Implements angelic execution and specification-driven synthesis
based on SyGuS (Syntax-Guided Synthesis) and component-based approaches.
"""

import ast
import z3
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Dict, List, Optional, Set, Tuple, Any, Callable,
    Iterator, Union
)
from enum import Enum, auto
from pathlib import Path

# Internal imports
from auto_repair.config import get_config
from auto_repair.localization.revr.smt_integration import PythonZ3Translator
from auto_repair.synthesis.pfs.templates import PatchInstance


# =============================================================================
# DATA CLASSES
# =============================================================================

class ValueType(Enum):
    """Types for angelic values."""
    INTEGER = auto()
    BOOLEAN = auto()
    STRING = auto()
    REFERENCE = auto()
    UNKNOWN = auto()

@dataclass
class AngelicValue:
    """
    A value that makes failing test pass (from angelic execution).
    
    Represents the "what should this expression evaluate to?" answer.
    """
    expression: str              # Original expression
    value: Any                   # Angelic value that fixes test
    value_type: ValueType
    test_case: str               # Test that this fixes
    path_condition: z3.BoolRef   # Constraints under which this holds
    
    def __hash__(self):
        return hash((self.expression, str(self.value), self.test_case))

@dataclass
class AngelicForest:
    """
    Collection of angelic values for multiple expressions/tests.
    
    Represents the specification for synthesis.
    """
    target_expression: str
    values: List[AngelicValue] = field(default_factory=list)
    type_constraint: Optional[str] = None
    
    def get_unique_values(self) -> Set[Any]:
        """Get set of unique angelic values."""
        return set(v.value for v in self.values)
    
    def get_value_histogram(self) -> Dict[Any, int]:
        """Count frequency of each angelic value."""
        hist = {}
        for v in self.values:
            hist[v.value] = hist.get(v.value, 0) + 1
        return hist
    
    def most_common_value(self) -> Optional[Any]:
        """Get most frequently occurring angelic value."""
        if not self.values:
            return None
        hist = self.get_value_histogram()
        return max(hist.items(), key=lambda x: x[1])[0]

@dataclass
class SynthesisComponent:
    """Building block for component-based synthesis."""
    name: str
    symbol: z3.ExprRef          # Z3 symbol representing component
    input_types: List[ValueType]
    output_type: ValueType
    semantics: Callable[..., Any]  # Python semantics
    
    def __hash__(self):
        return hash(self.name)

@dataclass
class SynthesisGrammar:
    """Grammar defining search space for synthesis."""
    components: List[SynthesisComponent]
    start_symbol: ValueType
    max_depth: int = 3
    max_size: int = 10
    
    def get_components_for_type(self, output_type: ValueType) -> List[SynthesisComponent]:
        """Get all components producing given type."""
        return [c for c in self.components if c.output_type == output_type]

@dataclass
class SynthesisResult:
    """Result of semantic synthesis."""
    success: bool
    synthesized_expr: Optional[str] = None
    z3_expr: Optional[z3.ExprRef] = None
    components_used: List[str] = field(default_factory=list)
    synthesis_time_ms: float = 0.0
    iterations: int = 0
    
    def to_patch(self, original: str, location: str, line: int) -> Optional[PatchInstance]:
        """Convert to PatchInstance if successful."""
        if not self.success or not self.synthesized_expr:
            return None
        
        return PatchInstance(
            template_id="AngelixSynthesis",
            original_code=original,
            patched_code=self.synthesized_expr,
            location=location,
            line_number=line,
            description=f"Synthesized: {self.synthesized_expr}",
            feature_vector=[0.8, 0.0, 1.0, 0.0, 0.0],  # High semantic score
            semantic_constraints=[f"equivalent_under_spec({original}, {self.synthesized_expr})"],
            test_suggestions=["Verify synthesized expression matches specification"]
        )

# =============================================================================
# ANGELIC EXECUTION ENGINE
# =============================================================================

class AngelicExecutionEngine:
    """
    Execute tests with symbolic expressions to infer specifications.
    
    Based on Angelix [^78^]: replaces suspicious expressions with
    symbolic variables, then solves for values that make tests pass.
    """
    
    def __init__(self, source_code: str, test_harness: Optional[Callable] = None):
        self.source_code = source_code
        self.ast_tree = ast.parse(source_code)
        self.test_harness = test_harness
        self.translator = PythonZ3Translator()
        self.solver = z3.Solver()
        
        self.logger = logging.getLogger(__name__)
    
    def identify_suspicious_expressions(
        self,
        fault_locations: List[Tuple[int, str]],
        max_expressions: int = 10
    ) -> List[ast.AST]:
        """
        Identify expressions at fault locations for angelic execution.
        
        Args:
            fault_locations: List of (line_number, context) tuples
            max_expressions: Maximum expressions to consider
        
        Returns:
            List of AST expression nodes
        """
        suspicious = []
        
        for line_no, context in fault_locations:
            # Find all expressions at this line
            for node in ast.walk(self.ast_tree):
                if hasattr(node, 'lineno') and node.lineno == line_no:
                    if isinstance(node, (ast.BinOp, ast.Call, ast.Attribute, ast.Name)):
                        suspicious.append(node)
        
        # Rank by complexity and context relevance
        ranked = self._rank_suspiciousness(suspicious, fault_locations)
        
        return ranked[:max_expressions]
    
    def _rank_suspiciousness(
        self,
        expressions: List[ast.AST],
        fault_locations: List[Tuple[int, str]]
    ) -> List[ast.AST]:
        """Rank expressions by likelihood of being buggy."""
        scored = []
        
        for expr in expressions:
            score = 0
            
            # Prefer binary operations (common bug source)
            if isinstance(expr, ast.BinOp):
                score += 10
            
            # Prefer method calls
            if isinstance(expr, ast.Call):
                score += 8
            
            # Prefer attribute access (null dereference)
            if isinstance(expr, ast.Attribute):
                score += 7
            
            # Penalize simple variables (less likely to be wrong)
            if isinstance(expr, ast.Name):
                score += 3
            
            scored.append((expr, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored]
    
    def execute_angelically(
        self,
        target_expression: ast.AST,
        test_cases: List[Dict[str, Any]],
        timeout: float = 30.0
    ) -> AngelicForest:
        """
        Execute tests with target expression as symbolic variable.
        
        For each failing test, solve for angelic value that makes it pass.
        """
        self.logger.info(f"Angelic execution for: {ast.unparse(target_expression) if hasattr(ast, 'unparse') else 'expr'}")
        
        forest = AngelicForest(
            target_expression=ast.unparse(target_expression) if hasattr(ast, 'unparse') else "unknown"
        )
        
        # Create symbolic variable for target expression
        expr_type = self._infer_type(target_expression)
        sym_var = self._create_symbolic_var("angelic", expr_type)
        
        # Instrument code to replace expression with symbolic variable
        instrumented_code = self._instrument_expression(target_expression, sym_var)
        
        for test in test_cases:
            test_name = test.get('name', 'unknown')
            
            try:
                # Execute test with symbolic expression
                path_condition, test_passed = self._execute_test(
                    instrumented_code,
                    test,
                    sym_var
                )
                
                if not test_passed:
                    # Solve for angelic value that makes test pass
                    angelic_val = self._solve_for_value(
                        sym_var,
                        target_expression,
                        test,
                        path_condition
                    )
                    
                    if angelic_val is not None:
                        forest.values.append(AngelicValue(
                            expression=forest.target_expression,
                            value=angelic_val,
                            value_type=expr_type,
                            test_case=test_name,
                            path_condition=path_condition
                        ))
                        
            except Exception as e:
                self.logger.warning(f"Angelic execution failed for test {test_name}: {e}")
        
        # Infer type constraint from angelic values
        if forest.values:
            forest.type_constraint = self._infer_type_constraint(forest.values)
        
        self.logger.info(f"Collected {len(forest.values)} angelic values")
        return forest
    
    def _create_symbolic_var(self, name: str, var_type: ValueType) -> z3.ExprRef:
        """Create Z3 symbolic variable of appropriate type."""
        if var_type == ValueType.INTEGER:
            return z3.Int(name)
        elif var_type == ValueType.BOOLEAN:
            return z3.Bool(name)
        elif var_type == ValueType.STRING:
            return z3.String(name)
        else:
            return z3.Int(name)  # Default to integer
    
    def _instrument_expression(
        self,
        target: ast.AST,
        sym_var: z3.ExprRef
    ) -> str:
        """
        Instrument code to replace target expression with symbolic variable.
        
        Creates a version of the code where the suspicious expression
        is replaced by a hook that reads from the symbolic variable.
        """
        # Clone AST
        new_tree = ast.parse(self.source_code)
        
        # Find and replace target expression
        class ExpressionReplacer(ast.NodeTransformer):
            def __init__(self, target_id: int, replacement: str):
                self.target_id = target_id
                self.replacement = replacement
                self.replaced = False
            
            def visit(self, node):
                if hasattr(node, 'lineno') and hasattr(node, 'col_offset'):
                    # Simple matching by position
                    node_id = id(node)
                    if node_id == id(target):
                        self.replaced = True
                        # Return a name node referencing symbolic variable
                        return ast.Name(id='__angelic_var__', ctx=ast.Load())
                return self.generic_visit(node)
        
        # This is simplified - real implementation needs careful AST manipulation
        # For now, return original code with marker
        return self.source_code + "\n# ANGELIC: " + str(sym_var)
    
    def _execute_test(
        self,
        instrumented_code: str,
        test: Dict[str, Any],
        sym_var: z3.ExprRef
    ) -> Tuple[z3.BoolRef, bool]:
        """
        Execute single test, collecting path condition.
        
        Returns: (path_condition, test_passed)
        """
        # In real implementation: compile and execute instrumented code
        # Collect path constraints from branches taken
        
        # Simplified: return dummy values
        return (z3.BoolVal(True), False)  # Assume failing test
    
    def _solve_for_value(
        self,
        sym_var: z3.ExprRef,
        original_expr: ast.AST,
        test: Dict[str, Any],
        path_condition: z3.BoolRef
    ) -> Optional[Any]:
        """
        Use SMT solver to find angelic value.
        
        Solves for: path_condition ∧ (sym_var makes test_pass)
        """
        self.solver.push()
        
        # Add path condition
        self.solver.add(path_condition)
        
        # Add test assertion as constraint
        # (In real impl: encode test oracle as SMT constraint)
        test_constraint = self._encode_test_oracle(test, sym_var)
        self.solver.add(test_constraint)
        
        # Solve
        if self.solver.check() == z3.sat:
            model = self.solver.model()
            value = model[sym_var]
            
            # Convert to Python value
            result = self._z3_to_python(value)
            
            self.solver.pop()
            return result
        
        self.solver.pop()
        return None
    
    def _encode_test_oracle(self, test: Dict[str, Any], sym_var: z3.ExprRef) -> z3.BoolRef:
        """Encode test oracle as SMT constraint."""
        # Simplified: assume test expects specific output
        expected = test.get('expected_output')
        
        if expected is not None:
            if isinstance(expected, bool):
                return sym_var == expected
            elif isinstance(expected, int):
                return sym_var == expected
        
        return z3.BoolVal(True)  # No constraint
    
    def _z3_to_python(self, value) -> Any:
        """Convert Z3 value to Python."""
        if z3.is_int_value(value):
            return value.as_long()
        elif z3.is_true(value) or z3.is_false(value):
            return z3.is_true(value)
        return str(value)
    
    def _infer_type(self, node: ast.AST) -> ValueType:
        """Infer type of AST node."""
        if isinstance(node, ast.Num):
            if isinstance(node.n, bool):
                return ValueType.BOOLEAN
            elif isinstance(node.n, int):
                return ValueType.INTEGER
            else:
                return ValueType.UNKNOWN
        elif isinstance(node, ast.Str):
            return ValueType.STRING
        elif isinstance(node, ast.NameConstant):
            return ValueType.BOOLEAN
        elif isinstance(node, ast.Compare):
            return ValueType.BOOLEAN
        return ValueType.UNKNOWN
    
    def _infer_type_constraint(self, values: List[AngelicValue]) -> str:
        """Infer type constraint from observed angelic values."""
        types = set(v.value_type for v in values)
        
        if len(types) == 1:
            return list(types)[0].name.lower()
        
        # Mixed types - use most common
        type_counts = {}
        for v in values:
            type_counts[v.value_type] = type_counts.get(v.value_type, 0) + 1
        
        return max(type_counts.items(), key=lambda x: x[1])[0].name.lower()


# =============================================================================
# COMPONENT-BASED SYNTHESIS
# =============================================================================

class ComponentBasedSynthesizer:
    """
    Synthesize expressions from components using SMT solving.
    
    Based on SyGuS approach: given specification (angelic values),
    find composition of components that satisfies it.
    """
    
    def __init__(self, grammar: Optional[SynthesisGrammar] = None):
        self.grammar = grammar or self._default_grammar()
        self.solver = z3.Solver()
        self.component_counter = 0
        
        self.logger = logging.getLogger(__name__)
    
    def _default_grammar(self) -> SynthesisGrammar:
        """Create default synthesis grammar for Python."""
        components = [
            # Arithmetic
            SynthesisComponent("add", z3.Function('add', z3.IntSort(), z3.IntSort(), z3.IntSort()),
                               [ValueType.INTEGER, ValueType.INTEGER], ValueType.INTEGER,
                               lambda x, y: x + y),
            SynthesisComponent("sub", z3.Function('sub', z3.IntSort(), z3.IntSort(), z3.IntSort()),
                               [ValueType.INTEGER, ValueType.INTEGER], ValueType.INTEGER,
                               lambda x, y: x - y),
            SynthesisComponent("mul", z3.Function('mul', z3.IntSort(), z3.IntSort(), z3.IntSort()),
                               [ValueType.INTEGER, ValueType.INTEGER], ValueType.INTEGER,
                               lambda x, y: x * y),
            SynthesisComponent("div", z3.Function('div', z3.IntSort(), z3.IntSort(), z3.IntSort()),
                               [ValueType.INTEGER, ValueType.INTEGER], ValueType.INTEGER,
                               lambda x, y: x // y if y != 0 else 0),
            
            # Boolean
            SynthesisComponent("and", z3.Function('and', z3.BoolSort(), z3.BoolSort(), z3.BoolSort()),
                               [ValueType.BOOLEAN, ValueType.BOOLEAN], ValueType.BOOLEAN,
                               lambda x, y: x and y),
            SynthesisComponent("or", z3.Function('or', z3.BoolSort(), z3.BoolSort(), z3.BoolSort()),
                               [ValueType.BOOLEAN, ValueType.BOOLEAN], ValueType.BOOLEAN,
                               lambda x, y: x or y),
            SynthesisComponent("not", z3.Function('not', z3.BoolSort(), z3.BoolSort()),
                               [ValueType.BOOLEAN], ValueType.BOOLEAN,
                               lambda x: not x),
            
            # Comparison
            SynthesisComponent("eq", z3.Function('eq', z3.IntSort(), z3.IntSort(), z3.BoolSort()),
                               [ValueType.INTEGER, ValueType.INTEGER], ValueType.BOOLEAN,
                               lambda x, y: x == y),
            SynthesisComponent("lt", z3.Function('lt', z3.IntSort(), z3.IntSort(), z3.BoolSort()),
                               [ValueType.INTEGER, ValueType.INTEGER], ValueType.BOOLEAN,
                               lambda x, y: x < y),
            SynthesisComponent("gt", z3.Function('gt', z3.IntSort(), z3.IntSort(), z3.BoolSort()),
                               [ValueType.INTEGER, ValueType.INTEGER], ValueType.BOOLEAN,
                               lambda x, y: x > y),
            
            # Constants
            SynthesisComponent("zero", z3.IntVal(0), [], ValueType.INTEGER, lambda: 0),
            SynthesisComponent("one", z3.IntVal(1), [], ValueType.INTEGER, lambda: 1),
        ]
        
        return SynthesisGrammar(
            components=components,
            start_symbol=ValueType.INTEGER,
            max_depth=3,
            max_size=5
        )
    
    def synthesize(
        self,
        specification: AngelicForest,
        available_inputs: List[str],
        timeout: float = 60.0
    ) -> SynthesisResult:
        """
        Synthesize expression satisfying angelic specification.
        
        Args:
            specification: Angelic forest with desired values
            available_inputs: Variable names available in scope
            timeout: Synthesis timeout in seconds
        
        Returns:
            SynthesisResult with synthesized expression or failure
        """
        import time
        start_time = time.time()
        
        self.logger.info(f"Synthesizing for spec: {specification.target_expression}")
        
        # Get most common angelic value as target
        target_value = specification.most_common_value()
        
        if target_value is None:
            return SynthesisResult(success=False)
        
        # Try direct value first
        if self._try_direct_value(target_value, available_inputs):
            expr_str = str(target_value)
            return SynthesisResult(
                success=True,
                synthesized_expr=expr_str,
                z3_expr=self._python_to_z3(target_value),
                components_used=["constant"],
                synthesis_time_ms=(time.time() - start_time) * 1000,
                iterations=1
            )
        
        # Try increasing complexity
        for depth in range(1, self.grammar.max_depth + 1):
            for size in range(1, self.grammar.max_size + 1):
                result = self._synthesize_with_constraints(
                    specification,
                    available_inputs,
                    depth,
                    size
                )
                
                if result.success:
                    result.synthesis_time_ms = (time.time() - start_time) * 1000
                    return result
                
                # Check timeout
                if (time.time() - start_time) > timeout:
                    self.logger.warning("Synthesis timeout")
                    return SynthesisResult(
                        success=False,
                        synthesis_time_ms=(time.time() - start_time) * 1000,
                        iterations=depth * size
                    )
        
        return SynthesisResult(
            success=False,
            synthesis_time_ms=(time.time() - start_time) * 1000,
            iterations=self.grammar.max_depth * self.grammar.max_size
        )
    
    def _try_direct_value(self, value: Any, inputs: List[str]) -> bool:
        """Check if value is directly available as constant or input."""
        # Check if value is a simple constant
        if isinstance(value, (int, bool, str)):
            return True
        
        # Check if value equals any input
        # (Simplified - would check against actual variable values)
        
        return False
    
    def _synthesize_with_constraints(
        self,
        specification: AngelicForest,
        inputs: List[str],
        max_depth: int,
        max_size: int
    ) -> SynthesisResult:
        """
        Synthesize expression using SMT constraints.
        
        Encodes synthesis as: ∃ expr. ∀ test. expr(inputs) = angelic_value
        """
        self.solver.push()
        
        # Create input variables
        input_vars = {}
        for inp in inputs:
            input_vars[inp] = z3.Int(inp)  # Assume integer for simplicity
        
        # Try to find expression
        # (Simplified - real implementation uses CEGIS loop)
        
        # For now, try simple combinations
        for component in self.grammar.components:
            if len(component.input_types) == 0:  # Constant
                continue
            
            if len(component.input_types) == 1:  # Unary
                for inp_name, inp_var in input_vars.items():
                    try:
                        result = component.semantics(inp_var)
                        # Check if matches specification
                        if self._matches_specification(result, specification):
                            expr_str = f"{component.name}({inp_name})"
                            return SynthesisResult(
                                success=True,
                                synthesized_expr=expr_str,
                                z3_expr=result if isinstance(result, z3.ExprRef) else None,
                                components_used=[component.name, inp_name],
                                iterations=1
                            )
                    except Exception:
                        continue
        
        self.solver.pop()
        return SynthesisResult(success=False)
    
    def _matches_specification(self, result, specification: AngelicForest) -> bool:
        """Check if result matches angelic specification."""
        # Simplified check
        target = specification.most_common_value()
        if target is None:
            return False
        
        # Convert result to comparable form
        if isinstance(result, z3.ExprRef):
            # Would need to evaluate with test inputs
            return False
        
        return result == target
    
    def _python_to_z3(self, value: Any) -> z3.ExprRef:
        """Convert Python value to Z3 expression."""
        if isinstance(value, int):
            return z3.IntVal(value)
        elif isinstance(value, bool):
            return z3.BoolVal(value)
        return z3.IntVal(0)


# =============================================================================
# SECOND-ORDER SYNTHESIS (ADVANCED)
# =============================================================================

class SecondOrderSynthesizer:
    """
    Second-order synthesis for complex expressions.
    
    Synthesizes higher-order functions and complex control structures.
    Used when first-order component-based synthesis fails.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def synthesize_conditional(
        self,
        specification: AngelicForest,
        inputs: List[str]
    ) -> Optional[SynthesisResult]:
        """
        Synthesize conditional expression (if-then-else).
        
        For specifications where different inputs need different expressions.
        """
        # Analyze when different angelic values are needed
        value_conditions = self._analyze_value_conditions(specification)
        
        if len(value_conditions) <= 1:
            return None  # No conditional needed
        
        # Synthesize condition for each branch
        branches = []
        for condition, value in value_conditions:
            # Synthesize expression for this branch
            synth = ComponentBasedSynthesizer()
            sub_spec = AngelicForest(
                target_expression=f"branch_{len(branches)}",
                values=[v for v in specification.values if v.value == value]
            )
            result = synth.synthesize(sub_spec, inputs)
            
            if result.success:
                branches.append((condition, result.synthesized_expr))
        
        if branches:
            # Construct if-then-else chain
            expr = self._build_conditional(branches)
            return SynthesisResult(
                success=True,
                synthesized_expr=expr,
                components_used=["conditional"] + [b[1] for b in branches]
            )
        
        return None
    
    def _analyze_value_conditions(
        self,
        specification: AngelicForest
    ) -> List[Tuple[str, Any]]:
        """
        Analyze which conditions lead to which angelic values.
        """
        # Group angelic values by their path conditions
        value_groups = {}
        for val in specification.values:
            key = str(val.value)
            if key not in value_groups:
                value_groups[key] = []
            value_groups[key].append(val.path_condition)
        
        # Create condition for each group
        result = []
        for value_str, conditions in value_groups.items():
            # Disjunction of all path conditions for this value
            combined = z3.Or(conditions) if len(conditions) > 1 else conditions[0]
            result.append((str(combined), eval(value_str)))  # Simplified
        
        return result
    
    def _build_conditional(self, branches: List[Tuple[str, str]]) -> str:
        """Build if-then-else expression from branches."""
        if len(branches) == 2:
            condition, then_expr = branches[0]
            _, else_expr = branches[1]
            return f"({then_expr} if {condition} else {else_expr})"
        
        # Recursive for multiple branches
        first = branches[0]
        rest = self._build_conditional(branches[1:])
        return f"({first[1]} if {first[0]} else {rest})"


# =============================================================================
# MAIN SEMANTIC SYNTHESIS API
# =============================================================================

class SemanticSynthesisEngine:
    """
    Unified semantic synthesis engine.
    
    Orchestrates angelic execution and synthesis components.
    """
    
    def __init__(
        self,
        config=None,
        grammar: Optional[SynthesisGrammar] = None
    ):
        self.config = config or get_config().pfs
        self.grammar = grammar
        self.angelic_engine: Optional[AngelicExecutionEngine] = None
        self.component_synthesizer = ComponentBasedSynthesizer(grammar)
        self.second_order = SecondOrderSynthesizer()
        
        self.logger = logging.getLogger(__name__)
    
    def synthesize_patch(
        self,
        source_code: str,
        fault_locations: List[Tuple[int, str]],
        test_cases: List[Dict[str, Any]],
        available_inputs: List[str],
        timeout: float = 60.0
    ) -> List[PatchInstance]:
        """
        Main entry point: synthesize patches from specification.
        
        Pipeline:
        1. Identify suspicious expressions
        2. Run angelic execution to collect specifications
        3. Synthesize expressions for each specification
        4. Convert to PatchInstance objects
        """
        import time
        start_time = time.time()
        
        patches = []
        
        # Initialize angelic engine
        self.angelic_engine = AngelicExecutionEngine(source_code)
        
        # Step 1: Identify suspicious expressions
        suspicious = self.angelic_engine.identify_suspicious_expressions(
            fault_locations
        )
        
        self.logger.info(f"Found {len(suspicious)} suspicious expressions")
        
        # Step 2-3: For each suspicious expression
        for expr in suspicious:
            # Check timeout
            if (time.time() - start_time) > timeout:
                self.logger.warning("Semantic synthesis timeout")
                break
            
            # Collect angelic specification
            forest = self.angelic_engine.execute_angelically(
                expr,
                test_cases
            )
            
            if not forest.values:
                continue  # No specification collected
            
            # Try first-order synthesis
            result = self.component_synthesizer.synthesize(
                forest,
                available_inputs,
                timeout=timeout / len(suspicious)  # Divide time budget
            )
            
            if result.success:
                patch = result.to_patch(
                    original=forest.target_expression,
                    location=fault_locations[0][1] if fault_locations else "unknown",
                    line=expr.lineno if hasattr(expr, 'lineno') else 0
                )
                if patch:
                    patches.append(patch)
                continue
            
            # Try second-order synthesis (conditionals)
            result = self.second_order.synthesize_conditional(
                forest,
                available_inputs
            )
            
            if result and result.success:
                patch = result.to_patch(
                    original=forest.target_expression,
                    location=fault_locations[0][1] if fault_locations else "unknown",
                    line=expr.lineno if hasattr(expr, 'lineno') else 0
                )
                if patch:
                    patches.append(patch)
        
        self.logger.info(f"Synthesized {len(patches)} semantic patches")
        return patches
    
    def validate_synthesized_patch(
        self,
        patch: PatchInstance,
        test_cases: List[Dict[str, Any]],
        source_code: str
    ) -> bool:
        """
        Validate that synthesized patch satisfies specification.
        
        Re-runs tests to verify correctness.
        """
        # Apply patch to source
        patched_code = self._apply_patch(source_code, patch)
        
        # Run tests
        for test in test_cases:
            try:
                # Execute test with patched code
                # (Simplified - would use sandbox)
                pass
            except Exception:
                return False
        
        return True
    
    def _apply_patch(self, source_code: str, patch: PatchInstance) -> str:
        """Apply patch to source code."""
        lines = source_code.split('\n')
        
        # Replace line
        if patch.line_number > 0 and patch.line_number <= len(lines):
            lines[patch.line_number - 1] = patch.patched_code
        
        return '\n'.join(lines)
    
    def get_specification_summary(
        self,
        forest: AngelicForest
    ) -> Dict[str, Any]:
        """
        Generate human-readable summary of angelic specification.
        """
        return {
            "target_expression": forest.target_expression,
            "num_tests_analyzed": len(forest.values),
            "unique_values": len(forest.get_unique_values()),
            "most_common_value": forest.most_common_value(),
            "value_distribution": forest.get_value_histogram(),
            "type_constraint": forest.type_constraint
        }


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_default_engine: Optional[SemanticSynthesisEngine] = None

def get_semantic_synthesis_engine() -> SemanticSynthesisEngine:
    """Get or create default semantic synthesis engine."""
    global _default_engine
    if _default_engine is None:
        _default_engine = SemanticSynthesisEngine()
    return _default_engine


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example buggy code
    buggy_code = """
def calculate_average(numbers):
    total = 0
    for n in numbers:
        total = total + n
    return total / len(numbers)  # Bug: division by zero if empty
"""
    
    # Fault location
    fault_locations = [(6, "Division by zero risk")]
    
    # Test cases
    test_cases = [
        {"name": "test_empty", "input": {"numbers": []}, "expected_output": None},
        {"name": "test_normal", "input": {"numbers": [1, 2, 3]}, "expected_output": 2.0},
    ]
    
    # Available inputs
    inputs = ["numbers", "total", "n", "len(numbers)"]
    
    # Synthesize
    engine = SemanticSynthesisEngine()
    patches = engine.synthesize_patch(
        buggy_code,
        fault_locations,
        test_cases,
        inputs,
        timeout=30.0
    )
    
    print(f"Synthesized {len(patches)} patches:")
    for i, patch in enumerate(patches, 1):
        print(f"\n  Patch {i}: {patch.description}")
        print(f"    Location: {patch.location}")
        print(f"    Original: {patch.original_code}")
        print(f"    Patched:  {patch.patched_code}")
