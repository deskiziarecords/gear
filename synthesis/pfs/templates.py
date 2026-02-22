# auto_repair/synthesis/pfs/templates.py

"""
Repair Templates for Probabilistic Fix Synthesis.

Defines schema-based patch templates inspired by Prophet/SPR,
with support for multiple languages and semantic constraints.
"""

import ast
import re
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Optional, Set, Tuple, Union,
    Iterator
)
from enum import Enum, auto
from pathlib import Path
import copy


# =============================================================================
# ENUMS AND BASE CLASSES
# =============================================================================

class TemplateCategory(Enum):
    """Categories of repair templates."""
    CONDITIONAL = auto()      # If-condition changes
    GUARD = auto()            # Null/bounds guards
    ASSIGNMENT = auto()       # Variable assignment changes
    LOOP = auto()             # Loop structure changes
    METHOD_CALL = auto()      # Method invocation changes
    TYPE = auto()             # Type-related fixes
    RESOURCE = auto()         # Resource management (close, free, etc.)
    CONCURRENCY = auto()      # Synchronization fixes
    SEMANTIC = auto()         # Angelic/synthesized expressions

class TemplatePriority(Enum):
    """Priority for template application."""
    HIGH = 1      # Most likely to fix common bugs
    MEDIUM = 2    # Moderate likelihood
    LOW = 3       # Specialized cases
    CUSTOM = 4    # User-defined patterns

@dataclass
class TemplateMatch:
    """Result of matching a template to code location."""
    template_id: str
    location: str           # file::line::method
    confidence: float       # Match confidence 0-1
    context: Dict[str, Any] # Extracted context variables
    constraints: List[str]  # Required conditions for application

@dataclass
class PatchInstance:
    """Concrete patch generated from template."""
    template_id: str
    original_code: str
    patched_code: str
    location: str
    line_number: int
    description: str
    feature_vector: List[float]
    semantic_constraints: List[str]
    test_suggestions: List[str]  # Suggested tests to verify patch

class RepairTemplate(ABC):
    """
    Abstract base class for all repair templates.
    
    Each template defines:
    - Schema: Pattern structure
    - Generator: Function to create concrete patches
    - Constraints: Semantic preconditions
    - Features: Extractable properties for ranking
    """
    
    def __init__(
        self,
        template_id: str,
        name: str,
        category: TemplateCategory,
        priority: TemplatePriority = TemplatePriority.MEDIUM,
        languages: List[str] = None,
        description: str = ""
    ):
        self.template_id = template_id
        self.name = name
        self.category = category
        self.priority = priority
        self.languages = languages or ["python"]
        self.description = description
        self._generator: Optional[Callable] = None
        self._constraints: List[Callable] = []
        self._feature_extractors: List[Callable] = []
    
    @abstractmethod
    def match(self, code: str, ast_node: ast.AST, context: Dict) -> List[TemplateMatch]:
        """
        Find locations where this template could apply.
        
        Returns list of matches with confidence scores.
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        match: TemplateMatch,
        context: Dict[str, Any]
    ) -> List[PatchInstance]:
        """
        Generate concrete patch instances from match.
        
        May generate multiple variants (e.g., different guard conditions).
        """
        pass
    
    def add_constraint(self, constraint_fn: Callable[[Dict], bool]):
        """Add semantic constraint function."""
        self._constraints.append(constraint_fn)
        return self
    
    def add_feature_extractor(self, extractor_fn: Callable[[Dict], float]):
        """Add feature extraction function."""
        self._feature_extractors.append(extractor_fn)
        return self
    
    def check_constraints(self, context: Dict) -> bool:
        """Check if all semantic constraints are satisfied."""
        return all(fn(context) for fn in self._constraints)
    
    def extract_features(self, context: Dict) -> List[float]:
        """Extract feature vector for ranking."""
        return [fn(context) for fn in self._feature_extractors]
    
    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.template_id}>"


# =============================================================================
# CONCRETE TEMPLATE IMPLEMENTATIONS
# =============================================================================

class ConditionTightenTemplate(RepairTemplate):
    """
    Tighten existing condition by adding conjunct.
    
    Original: if (condition): ...
    Patched:  if (condition and new_guard): ...
    """
    
    def __init__(self):
        super().__init__(
            template_id="COND_TIGHTEN",
            name="Condition Tightening",
            category=TemplateCategory.CONDITIONAL,
            priority=TemplatePriority.HIGH,
            description="Add additional constraint to existing if-condition"
        )
        
        # Add common constraints
        self.add_constraint(lambda ctx: "if_node" in ctx)
        self.add_constraint(lambda ctx: ctx.get("condition", "") != "")
        
        # Add feature extractors
        self.add_feature_extractor(lambda ctx: 1.0 if "null" in str(ctx) else 0.0)
        self.add_feature_extractor(lambda ctx: len(ctx.get("variables", [])))
    
    def match(self, code: str, ast_node: ast.AST, context: Dict) -> List[TemplateMatch]:
        matches = []
        
        for node in ast.walk(ast_node):
            if isinstance(node, ast.If):
                # Extract condition text
                condition = ast.unparse(node.test) if hasattr(ast, 'unparse') else ""
                
                # Check if condition could be tightened
                if not self._is_already_tight(condition):
                    match = TemplateMatch(
                        template_id=self.template_id,
                        location=context.get("location", "unknown"),
                        confidence=0.7,
                        context={
                            "if_node": node,
                            "condition": condition,
                            "body": node.body,
                            "variables": self._extract_variables(node),
                            "line_number": node.lineno
                        },
                        constraints=["condition_not_tight"]
                    )
                    matches.append(match)
        
        return matches
    
    def generate(
        self,
        match: TemplateMatch,
        context: Dict[str, Any]
    ) -> List[PatchInstance]:
        instances = []
        ctx = match.context
        
        # Generate possible guard conditions
        guards = self._suggest_guards(ctx)
        
        for guard in guards:
            new_condition = f"({ctx['condition']}) and ({guard})"
            original = f"if {ctx['condition']}:"
            patched = f"if {new_condition}:"
            
            instance = PatchInstance(
                template_id=self.template_id,
                original_code=original,
                patched_code=patched,
                location=match.location,
                line_number=ctx["line_number"],
                description=f"Add guard: {guard}",
                feature_vector=self.extract_features(ctx),
                semantic_constraints=[f"implies({guard}, not error)"],
                test_suggestions=[f"Test with {guard}=False", f"Test with {guard}=True"]
            )
            instances.append(instance)
        
        return instances
    
    def _is_already_tight(self, condition: str) -> bool:
        """Check if condition is already conjunction-heavy."""
        return condition.count("and") > 2 or "and" in condition.lower()
    
    def _extract_variables(self, node: ast.AST) -> List[str]:
        """Extract variable names from AST node."""
        vars = []
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                vars.append(child.id)
        return list(set(vars))
    
    def _suggest_guards(self, context: Dict) -> List[str]:
        """Suggest appropriate guard conditions based on context."""
        guards = []
        variables = context.get("variables", [])
        
        # Null checks for object variables
        for var in variables:
            if not var.startswith("_") and len(var) > 2:
                guards.append(f"{var} is not None")
                guards.append(f"{var} is not None and len({var}) > 0")
        
        # Bounds checks for index variables
        for var in variables:
            if any(x in var.lower() for x in ["idx", "index", "pos", "i"]):
                guards.append(f"{var} >= 0")
                guards.append(f"len(data) > {var}")
        
        # Type checks
        for var in variables:
            guards.append(f"isinstance({var}, expected_type)")
        
        return guards[:5]  # Limit to top 5 suggestions


class ConditionLoosenTemplate(RepairTemplate):
    """
    Loosen condition by adding disjunct or removing constraint.
    
    Original: if (strict_condition): ...
    Patched:  if (strict_condition or alternative): ...
    """
    
    def __init__(self):
        super().__init__(
            template_id="COND_LOOSEN",
            name="Condition Loosening",
            category=TemplateCategory.CONDITIONAL,
            priority=TemplatePriority.MEDIUM,
            description="Relax condition to handle more cases"
        )
    
    def match(self, code: str, ast_node: ast.AST, context: Dict) -> List[TemplateMatch]:
        matches = []
        
        for node in ast.walk(ast_node):
            if isinstance(node, ast.If):
                condition = ast.unparse(node.test) if hasattr(ast, 'unparse') else ""
                
                # High confidence if condition is very strict
                confidence = 0.5
                if "==" in condition and "or" not in condition:
                    confidence = 0.8  # Strict equality might be too strict
                
                match = TemplateMatch(
                    template_id=self.template_id,
                    location=context.get("location", "unknown"),
                    confidence=confidence,
                    context={
                        "if_node": node,
                        "condition": condition,
                        "line_number": node.lineno
                    },
                    constraints=["strict_condition"]
                )
                matches.append(match)
        
        return matches
    
    def generate(
        self,
        match: TemplateMatch,
        context: Dict[str, Any]
    ) -> List[PatchInstance]:
        instances = []
        ctx = match.context
        
        # Suggest alternatives
        alternatives = self._suggest_alternatives(ctx["condition"])
        
        for alt in alternatives:
            new_condition = f"({ctx['condition']}) or ({alt})"
            
            instance = PatchInstance(
                template_id=self.template_id,
                original_code=f"if {ctx['condition']}:",
                patched_code=f"if {new_condition}:",
                location=match.location,
                line_number=ctx["line_number"],
                description=f"Allow alternative: {alt}",
                feature_vector=[0.5, 1.0 if "==" in ctx['condition'] else 0.0],
                semantic_constraints=[f"covers_case({alt})"],
                test_suggestions=[f"Test case where {alt} is true"]
            )
            instances.append(instance)
        
        return instances
    
    def _suggest_alternatives(self, condition: str) -> List[str]:
        """Suggest alternative conditions to add."""
        alternatives = []
        
        # If strict equality, suggest None check
        if "==" in condition:
            alternatives.append("x is None")  # Placeholder
        
        # Add common alternatives
        alternatives.extend([
            "default_case",
            "fallback_condition",
            "error_state"
        ])
        
        return alternatives


class NullGuardTemplate(RepairTemplate):
    """
    Add null/None check guard before dereference.
    
    Original: x = obj.field
    Patched:  if obj is not None: x = obj.field else: ...
    """
    
    def __init__(self):
        super().__init__(
            template_id="GUARD_NULL",
            name="Null Guard Insertion",
            category=TemplateCategory.GUARD,
            priority=TemplatePriority.HIGH,
            description="Add null check before field access"
        )
    
    def match(self, code: str, ast_node: ast.AST, context: Dict) -> List[TemplateMatch]:
        matches = []
        
        for node in ast.walk(ast_node):
            # Look for attribute access (obj.field)
            if isinstance(node, ast.Attribute):
                obj_name = self._get_object_name(node.value)
                
                if obj_name:
                    # Check if already guarded
                    if not self._is_already_guarded(node, ast_node):
                        match = TemplateMatch(
                            template_id=self.template_id,
                            location=context.get("location", "unknown"),
                            confidence=0.9,  # High confidence for null guards
                            context={
                                "attribute_node": node,
                                "object_name": obj_name,
                                "field_name": node.attr,
                                "line_number": node.lineno,
                                "parent_statement": self._find_parent_statement(node, ast_node)
                            },
                            constraints=["dereference_risk"]
                        )
                        matches.append(match)
        
        return matches
    
    def generate(
        self,
        match: TemplateMatch,
        context: Dict[str, Any]
    ) -> List[PatchInstance]:
        instances = []
        ctx = match.context
        
        obj = ctx["object_name"]
        field = ctx["field_name"]
        parent = ctx["parent_statement"]
        
        # Generate different guard styles
        guard_styles = [
            # Style 1: Early return
            (f"if {obj} is None:\n    return None\n{parent}", f"Early return if {obj} is None"),
            # Style 2: If-else
            (f"if {obj} is not None:\n    {parent}\nelse:\n    pass", f"Guard with else clause"),
            # Style 3: Assertion
            (f"assert {obj} is not None, '{obj} should not be None'\n{parent}", f"Assert non-null"),
        ]
        
        for patched_code, description in guard_styles:
            instance = PatchInstance(
                template_id=self.template_id,
                original_code=parent,
                patched_code=patched_code,
                location=match.location,
                line_number=ctx["line_number"],
                description=description,
                feature_vector=[1.0, 0.0, 1.0],  # High priority features
                semantic_constraints=[f"{obj} != None"],
                test_suggestions=[f"Test with {obj}=None", f"Test with {obj}=valid_object"]
            )
            instances.append(instance)
        
        return instances
    
    def _get_object_name(self, node: ast.AST) -> Optional[str]:
        """Extract object name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self._get_object_name(node.value) + "." + node.attr
        return None
    
    def _is_already_guarded(self, node: ast.AST, root: ast.AST) -> bool:
        """Check if node is already inside a null check."""
        # Simplified: would walk up AST to find enclosing if
        return False
    
    def _find_parent_statement(self, node: ast.AST, root: ast.AST) -> str:
        """Find the parent statement containing this node."""
        # Simplified implementation
        return ast.unparse(node) if hasattr(ast, 'unparse') else ""


class BoundsCheckTemplate(RepairTemplate):
    """
    Add bounds check for array/list access.
    
    Original: x = arr[i]
    Patched:  if 0 <= i < len(arr): x = arr[i] else: ...
    """
    
    def __init__(self):
        super().__init__(
            template_id="GUARD_BOUNDS",
            name="Bounds Check Insertion",
            category=TemplateCategory.GUARD,
            priority=TemplatePriority.HIGH,
            description="Add bounds check before index access"
        )
    
    def match(self, code: str, ast_node: ast.AST, context: Dict) -> List[TemplateMatch]:
        matches = []
        
        for node in ast.walk(ast_node):
            if isinstance(node, ast.Subscript):
                # Check if index is a variable (not constant)
                if isinstance(node.slice, (ast.Name, ast.BinOp)):
                    arr_name = self._get_array_name(node.value)
                    idx_name = self._get_index_name(node.slice)
                    
                    if arr_name and idx_name:
                        match = TemplateMatch(
                            template_id=self.template_id,
                            location=context.get("location", "unknown"),
                            confidence=0.85,
                            context={
                                "subscript_node": node,
                                "array_name": arr_name,
                                "index_name": idx_name,
                                "line_number": node.lineno
                            },
                            constraints=["index_variable"]
                        )
                        matches.append(match)
        
        return matches
    
    def generate(
        self,
        match: TemplateMatch,
        context: Dict[str, Any]
    ) -> List[PatchInstance]:
        instances = []
        ctx = match.context
        
        arr = ctx["array_name"]
        idx = ctx["index_name"]
        
        # Different bounds check styles
        checks = [
            (f"0 <= {idx} < len({arr})", "Standard bounds check"),
            (f"{idx} >= 0 and {idx} < len({arr})", "Explicit and-check"),
            (f"0 <= {idx} <= len({arr}) - 1", "Alternative bounds check"),
        ]
        
        for check, desc in checks:
            guard = f"if {check}:\n    original_operation\nelse:\n    handle_error()"
            
            instance = PatchInstance(
                template_id=self.template_id,
                original_code=f"{arr}[{idx}]",
                patched_code=guard,
                location=match.location,
                line_number=ctx["line_number"],
                description=desc,
                feature_vector=[0.9, 1.0, 0.0],
                semantic_constraints=[f"0 <= {idx} < len({arr})"],
                test_suggestions=[
                    f"Test with {idx}=0",
                    f"Test with {idx}=len({arr})-1",
                    f"Test with {idx}=-1",
                    f"Test with {idx}=len({arr})"
                ]
            )
            instances.append(instance)
        
        return instances
    
    def _get_array_name(self, node: ast.AST) -> Optional[str]:
        """Extract array name from subscript."""
        if isinstance(node, ast.Name):
            return node.id
        return None
    
    def _get_index_name(self, node: ast.AST) -> Optional[str]:
        """Extract index expression name."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.BinOp):
            return ast.unparse(node) if hasattr(ast, 'unparse') else "expr"
        return None


class VariableReplacementTemplate(RepairTemplate):
    """
    Replace variable with another in-scope variable.
    
    Original: result = calculate(x)
    Patched:  result = calculate(y)  # where y is similar type
    """
    
    def __init__(self):
        super().__init__(
            template_id="VAR_REPLACE",
            name="Variable Replacement",
            category=TemplateCategory.ASSIGNMENT,
            priority=TemplatePriority.LOW,
            description="Replace variable with alternative in-scope variable"
        )
    
    def match(self, code: str, ast_node: ast.AST, context: Dict) -> List[TemplateMatch]:
        matches = []
        
        # Find all variable usages
        variables = {}
        for node in ast.walk(ast_node):
            if isinstance(node, ast.Name):
                var_name = node.id
                if var_name not in variables:
                    variables[var_name] = []
                variables[var_name].append(node)
        
        # Suggest replacements for each variable
        for var_name, usages in variables.items():
            if len(usages) > 0:
                # Find alternative variables in scope
                alternatives = [v for v in variables.keys() if v != var_name]
                
                if alternatives:
                    match = TemplateMatch(
                        template_id=self.template_id,
                        location=context.get("location", "unknown"),
                        confidence=0.4,  # Lower confidence - risky
                        context={
                            "original_var": var_name,
                            "alternatives": alternatives,
                            "usages": usages,
                            "line_number": usages[0].lineno
                        },
                        constraints=["type_compatible"]
                    )
                    matches.append(match)
        
        return matches
    
    def generate(
        self,
        match: TemplateMatch,
        context: Dict[str, Any]
    ) -> List[PatchInstance]:
        instances = []
        ctx = match.context
        
        original = ctx["original_var"]
        
        for alt in ctx["alternatives"][:3]:  # Top 3 alternatives
            # Create replacement patch
            instance = PatchInstance(
                template_id=self.template_id,
                original_code=original,
                patched_code=alt,
                location=match.location,
                line_number=ctx["line_number"],
                description=f"Replace {original} with {alt}",
                feature_vector=[0.4, 0.0, 0.0],
                semantic_constraints=[f"type({alt}) <: type({original})"],
                test_suggestions=[
                    f"Verify {alt} has same semantics as {original}",
                    f"Test with different values of {alt}"
                ]
            )
            instances.append(instance)
        
        return instances


class MethodCallChangeTemplate(RepairTemplate):
    """
    Change method call to alternative method.
    
    Original: obj.old_method(args)
    Patched:  obj.new_method(args)
    """
    
    def __init__(self):
        super().__init__(
            template_id="METHOD_CHANGE",
            name="Method Call Replacement",
            category=TemplateCategory.METHOD_CALL,
            priority=TemplatePriority.MEDIUM,
            description="Replace method call with alternative"
        )
        
        # Common method replacements
        self.replacements = {
            "add": ["append", "extend", "insert"],
            "remove": ["pop", "discard", "delete"],
            "get": ["fetch", "retrieve", "load"],
            "size": ["len", "length", "count"],
            "is_empty": ["not", "len == 0", "count == 0"],
        }
    
    def match(self, code: str, ast_node: ast.AST, context: Dict) -> List[TemplateMatch]:
        matches = []
        
        for node in ast.walk(ast_node):
            if isinstance(node, ast.Call):
                method_name = self._get_method_name(node.func)
                
                if method_name and method_name in self.replacements:
                    match = TemplateMatch(
                        template_id=self.template_id,
                        location=context.get("location", "unknown"),
                        confidence=0.6,
                        context={
                            "call_node": node,
                            "method_name": method_name,
                            "args": node.args,
                            "line_number": node.lineno
                        },
                        constraints=["same_signature"]
                    )
                    matches.append(match)
        
        return matches
    
    def generate(
        self,
        match: TemplateMatch,
        context: Dict[str, Any]
    ) -> List[PatchInstance]:
        instances = []
        ctx = match.context
        
        original_method = ctx["method_name"]
        alternatives = self.replacements.get(original_method, [])
        
        for alt_method in alternatives:
            # Construct new call
            args_str = ", ".join([ast.unparse(arg) for arg in ctx["args"]]) if hasattr(ast, 'unparse') else "..."
            new_call = f"{alt_method}({args_str})"
            
            instance = PatchInstance(
                template_id=self.template_id,
                original_code=f"{original_method}({args_str})",
                patched_code=new_call,
                location=match.location,
                line_number=ctx["line_number"],
                description=f"Replace {original_method} with {alt_method}",
                feature_vector=[0.6, 1.0 if original_method in ["add", "remove"] else 0.0],
                semantic_constraints=[f"equivalent_behavior({original_method}, {alt_method})"],
                test_suggestions=[
                    f"Test {alt_method} with same inputs",
                    f"Verify return type matches"
                ]
            )
            instances.append(instance)
        
        return instances
    
    def _get_method_name(self, node: ast.AST) -> Optional[str]:
        """Extract method name from call."""
        if isinstance(node, ast.Attribute):
            return node.attr
        elif isinstance(node, ast.Name):
            return node.id
        return None


class ExceptionHandlingTemplate(RepairTemplate):
    """
    Add or modify exception handling.
    
    Original: risky_operation()
    Patched:  try: risky_operation() except SpecificError: handle()
    """
    
    def __init__(self):
        super().__init__(
            template_id="EXCEPT_ADD",
            name="Exception Handler Addition",
            category=TemplateCategory.GUARD,
            priority=TemplatePriority.MEDIUM,
            description="Add try-except block around risky operations"
        )
        
        self.risky_operations = [
            "open", "read", "write", "connect",
            "parse", "eval", "exec", "load"
        ]
    
    def match(self, code: str, ast_node: ast.AST, context: Dict) -> List[TemplateMatch]:
        matches = []
        
        for node in ast.walk(ast_node):
            if isinstance(node, ast.Call):
                method_name = self._get_method_name(node.func)
                
                if method_name in self.risky_operations:
                    # Check if already in try block
                    if not self._is_in_try_block(node, ast_node):
                        match = TemplateMatch(
                            template_id=self.template_id,
                            location=context.get("location", "unknown"),
                            confidence=0.75,
                            context={
                                "call_node": node,
                                "operation": method_name,
                                "line_number": node.lineno
                            },
                            constraints=["io_operation"]
                        )
                        matches.append(match)
        
        return matches
    
    def generate(
        self,
        match: TemplateMatch,
        context: Dict[str, Any]
    ) -> List[PatchInstance]:
        instances = []
        ctx = match.context
        
        operation = ctx["operation"]
        call_str = ast.unparse(ctx["call_node"]) if hasattr(ast, 'unparse') else operation
        
        # Suggest exception types based on operation
        exception_types = {
            "open": ["FileNotFoundError", "PermissionError", "IOError"],
            "read": ["IOError", "ValueError"],
            "write": ["IOError", "PermissionError"],
            "connect": ["ConnectionError", "TimeoutError"],
            "parse": ["ValueError", "SyntaxError"],
        }.get(operation, ["Exception"])
        
        for exc_type in exception_types:
            try_block = f"""try:
    {call_str}
except {exc_type} as e:
    # TODO: Handle error appropriately
    logging.error(f"{operation} failed: {{e}}")
    raise"""
            
            instance = PatchInstance(
                template_id=self.template_id,
                original_code=call_str,
                patched_code=try_block,
                location=match.location,
                line_number=ctx["line_number"],
                description=f"Add {exc_type} handling for {operation}",
                feature_vector=[0.75, 1.0, 0.0],
                semantic_constraints=[f"catches({exc_type})"],
                test_suggestions=[
                    f"Test with {exc_type} scenario",
                    f"Verify error handling logic"
                ]
            )
            instances.append(instance)
        
        return instances
    
    def _get_method_name(self, node: ast.AST) -> Optional[str]:
        """Extract method name."""
        if isinstance(node, ast.Attribute):
            return node.attr
        elif isinstance(node, ast.Name):
            return node.id
        return None
    
    def _is_in_try_block(self, node: ast.AST, root: ast.AST) -> bool:
        """Check if node is inside a try block."""
        # Simplified - would need parent tracking
        return False


class TypeConversionTemplate(RepairTemplate):
    """
    Add explicit type conversion.
    
    Original: x = some_value
    Patched:  x = int(some_value)  # or str(), float(), etc.
    """
    
    def __init__(self):
        super().__init__(
            template_id="TYPE_CONV",
            name="Type Conversion Addition",
            category=TemplateCategory.TYPE,
            priority=TemplatePriority.MEDIUM,
            description="Add explicit type conversion"
        )
    
    def match(self, code: str, ast_node: ast.AST, context: Dict) -> List[TemplateMatch]:
        matches = []
        
        # Look for assignments that might need type conversion
        for node in ast.walk(ast_node):
            if isinstance(node, ast.Assign):
                # Check if value might need conversion
                value_type = self._infer_type(node.value)
                target_type = context.get("expected_type")
                
                if value_type and target_type and value_type != target_type:
                    match = TemplateMatch(
                        template_id=self.template_id,
                        location=context.get("location", "unknown"),
                        confidence=0.5,
                        context={
                            "assign_node": node,
                            "value_type": value_type,
                            "target_type": target_type,
                            "line_number": node.lineno
                        },
                        constraints=["type_mismatch"]
                    )
                    matches.append(match)
        
        return matches
    
    def generate(
        self,
        match: TemplateMatch,
        context: Dict[str, Any]
    ) -> List[PatchInstance]:
        instances = []
        ctx = match.context
        
        value = ast.unparse(ctx["assign_node"].value) if hasattr(ast, 'unparse') else "value"
        target_type = ctx["target_type"]
        
        # Suggest conversion functions
        converters = {
            "int": f"int({value})",
            "str": f"str({value})",
            "float": f"float({value})",
            "bool": f"bool({value})",
            "list": f"list({value})",
        }
        
        for type_name, conversion in converters.items():
            if type_name == target_type:
                instance = PatchInstance(
                    template_id=self.template_id,
                    original_code=value,
                    patched_code=conversion,
                    location=match.location,
                    line_number=ctx["line_number"],
                    description=f"Convert to {type_name}",
                    feature_vector=[0.5, 0.0, 1.0],
                    semantic_constraints=[f"valid_conversion({value}, {type_name})"],
                    test_suggestions=[
                        f"Test conversion with valid {type_name}",
                        f"Test with invalid input"
                    ]
                )
                instances.append(instance)
        
        return instances
    
    def _infer_type(self, node: ast.AST) -> Optional[str]:
        """Infer type of AST node."""
        if isinstance(node, ast.Num):
            return "int" if isinstance(node.n, int) else "float"
        elif isinstance(node, ast.Str):
            return "str"
        elif isinstance(node, ast.List):
            return "list"
        elif isinstance(node, ast.Dict):
            return "dict"
        return None


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================

class TemplateRegistry:
    """
    Central registry for all repair templates.
    
    Manages template discovery, matching, and instantiation.
    """
    
    def __init__(self):
        self._templates: Dict[str, RepairTemplate] = {}
        self._by_category: Dict[TemplateCategory, List[RepairTemplate]] = {
            cat: [] for cat in TemplateCategory
        }
        self._register_builtin_templates()
    
    def _register_builtin_templates(self):
        """Register all built-in templates."""
        templates = [
            ConditionTightenTemplate(),
            ConditionLoosenTemplate(),
            NullGuardTemplate(),
            BoundsCheckTemplate(),
            VariableReplacementTemplate(),
            MethodCallChangeTemplate(),
            ExceptionHandlingTemplate(),
            TypeConversionTemplate(),
        ]
        
        for template in templates:
            self.register(template)
    
    def register(self, template: RepairTemplate):
        """Register a new template."""
        self._templates[template.template_id] = template
        self._by_category[template.category].append(template)
    
    def get(self, template_id: str) -> Optional[RepairTemplate]:
        """Get template by ID."""
        return self._templates.get(template_id)
    
    def get_by_category(self, category: TemplateCategory) -> List[RepairTemplate]:
        """Get all templates in category."""
        return self._by_category.get(category, [])
    
    def get_all(self) -> List[RepairTemplate]:
        """Get all registered templates."""
        return list(self._templates.values())
    
    def match_all(
        self,
        code: str,
        ast_tree: ast.AST,
        context: Dict[str, Any]
    ) -> Iterator[TemplateMatch]:
        """
        Match all templates against code.
        
        Yields matches sorted by confidence.
        """
        all_matches = []
        
        for template in self._templates.values():
            try:
                matches = template.match(code, ast_tree, context)
                all_matches.extend(matches)
            except Exception as e:
                # Log error but continue with other templates
                print(f"Template {template.template_id} failed: {e}")
        
        # Sort by confidence descending
        all_matches.sort(key=lambda m: m.confidence, reverse=True)
        
        yield from all_matches
    
    def generate_patches(
        self,
        match: TemplateMatch,
        context: Dict[str, Any]
    ) -> List[PatchInstance]:
        """Generate patches from a match."""
        template = self._templates.get(match.template_id)
        if not template:
            return []
        
        return template.generate(match, context)
    
    def get_stats(self) -> Dict[str, any]:
        """Get registry statistics."""
        return {
            "total_templates": len(self._templates),
            "by_category": {
                cat.name: len(templates) 
                for cat, templates in self._by_category.items()
            },
            "by_priority": self._count_by_priority()
        }
    
    def _count_by_priority(self) -> Dict[str, int]:
        """Count templates by priority."""
        counts = {}
        for template in self._templates.values():
            p = template.priority.name
            counts[p] = counts.get(p, 0) + 1
        return counts


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_custom_template(
    template_id: str,
    name: str,
    category: TemplateCategory,
    match_pattern: str,
    generate_replacement: Callable[[Dict], str],
    priority: TemplatePriority = TemplatePriority.CUSTOM
) -> RepairTemplate:
    """
    Factory for creating custom templates from simple patterns.
    
    Args:
        template_id: Unique identifier
        name: Human-readable name
        category: Template category
        match_pattern: Regex or AST pattern to match
        generate_replacement: Function(context) -> replacement_code
        priority: Template priority
    
    Returns:
        Configured RepairTemplate instance
    """
    
    class CustomTemplate(RepairTemplate):
        def __init__(self):
            super().__init__(
                template_id=template_id,
                name=name,
                category=category,
                priority=priority
            )
            self.pattern = re.compile(match_pattern) if isinstance(match_pattern, str) else match_pattern
        
        def match(self, code: str, ast_node: ast.AST, context: Dict) -> List[TemplateMatch]:
            matches = []
            
            # Try regex match first
            if isinstance(self.pattern, type(re.compile(''))):
                for match in self.pattern.finditer(code):
                    matches.append(TemplateMatch(
                        template_id=self.template_id,
                        location=context.get("location", "unknown"),
                        confidence=0.7,
                        context={
                            "match": match,
                            "groups": match.groups(),
                            "line_number": code[:match.start()].count('\n') + 1
                        },
                        constraints=[]
                    ))
            
            return matches
        
        def generate(self, match: TemplateMatch, context: Dict[str, Any]) -> List[PatchInstance]:
            replacement = generate_replacement(match.context)
            
            return [PatchInstance(
                template_id=self.template_id,
                original_code=match.context.get("match", "").group(0) if hasattr(match.context.get("match"), 'group') else "",
                patched_code=replacement,
                location=match.location,
                line_number=match.context.get("line_number", 0),
                description=f"Custom replacement: {name}",
                feature_vector=[0.7],
                semantic_constraints=[],
                test_suggestions=[]
            )]
    
    return CustomTemplate()


def load_templates_from_file(path: Path) -> List[RepairTemplate]:
    """
    Load custom templates from YAML/JSON definition file.
    
    File format:
    ```yaml
    templates:
      - id: CUSTOM_001
        name: "My Custom Fix"
        category: GUARD
        match: "pattern_regex"
        replacement: "replacement_template"
        priority: HIGH
    ```
    """
    import yaml
    
    templates = []
    with open(path) as f:
        data = yaml.safe_load(f)
    
    for tmpl_def in data.get("templates", []):
        template = create_custom_template(
            template_id=tmpl_def["id"],
            name=tmpl_def["name"],
            category=TemplateCategory[tmpl_def["category"]],
            match_pattern=tmpl_def["match"],
            generate_replacement=lambda ctx, tmpl=tmpl_def["replacement"]: tmpl.format(**ctx),
            priority=TemplatePriority[tmpl_def.get("priority", "CUSTOM")]
        )
        templates.append(template)
    
    return templates


# =============================================================================
# GLOBAL REGISTRY
# =============================================================================

# Singleton instance
_default_registry: Optional[TemplateRegistry] = None

def get_template_registry() -> TemplateRegistry:
    """Get or create default template registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = TemplateRegistry()
    return _default_registry

def reset_template_registry():
    """Reset global registry (useful for testing)."""
    global _default_registry
    _default_registry = None


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example code with bugs
    buggy_code = '''
def process_data(data, index):
    user = get_user(data)
    if user.is_active:
        value = data[index]
        result = calculate(value)
        return result
    return None
'''
    
    # Parse code
    tree = ast.parse(buggy_code)
    
    # Get registry
    registry = get_template_registry()
    
    # Match templates
    context = {"location": "test.py::process_data"}
    
    print("Matching templates...")
    for match in registry.match_all(buggy_code, tree, context):
        print(f"\n  Match: {match.template_id}")
        print(f"    Confidence: {match.confidence:.2f}")
        print(f"    Line: {match.context.get('line_number')}")
        
        # Generate patches
        template = registry.get(match.template_id)
        patches = template.generate(match, match.context)
        
        for patch in patches[:2]:  # Show top 2
            print(f"    Patch: {patch.description}")
            print(f"      Original: {patch.original_code[:50]}...")
            print(f"      Patched:  {patch.patched_code[:50]}...")
    
    # Print stats
    print(f"\nRegistry stats: {registry.get_stats()}")
    
    # Custom template example
    print("\n--- Custom Template Example ---")
    custom = create_custom_template(
        template_id="CUSTOM_LOG_FIX",
        name="Add logging to function",
        category=TemplateCategory.GUARD,
        match_pattern=r"def\s+(\w+)\s*\(",
        generate_replacement=lambda ctx: f"def {ctx['groups'][0]}(\n    logging.info('Entering {ctx['groups'][0]}')\n    ",
        priority=TemplatePriority.LOW
    )
    
    registry.register(custom)
    print(f"Registered custom template. Total: {len(registry.get_all())}")
