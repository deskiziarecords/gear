# auto_repair/synthesis/pfs/features.py

"""
Feature Extraction for Patch Ranking.

Extracts 50-dimensional feature vectors from patches for:
- Prophet-style log-linear models [^75^]
- Neural network rankers
- Semantic similarity scoring

Features capture modification patterns, program context, and repair history.
"""

import ast
import re
import hashlib
from dataclasses import dataclass, field
from typing import (
    Dict, List, Optional, Set, Tuple, Any, Callable,
    Union
)
from pathlib import Path
from collections import defaultdict
import numpy as np

# Try to import ML libraries for advanced features
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ModificationFeatures:
    """Features about the code modification (indices 0-19)."""
    is_condition_change: float = 0.0
    is_null_guard: float = 0.0
    is_bounds_check: float = 0.0
    is_variable_replace: float = 0.0
    is_method_change: float = 0.0
    adds_loop: float = 0.0
    adds_exception_handler: float = 0.0
    modifies_arithmetic: float = 0.0
    changes_control_flow: float = 0.0
    adds_type_check: float = 0.0
    deletes_code: float = 0.0
    adds_new_variable: float = 0.0
    changes_method_signature: float = 0.0
    adds_import: float = 0.0
    modifies_string_literal: float = 0.0
    adds_logging: float = 0.0
    adds_assertion: float = 0.0
    changes_comparison_operator: float = 0.0
    adds_method_call: float = 0.0
    removes_method_call: float = 0.0

@dataclass
class ProgramValueFeatures:
    """Features about variable usage (indices 20-39)."""
    var_used_in_condition: float = 0.0
    var_used_in_assignment: float = 0.0
    var_is_return_value: float = 0.0
    var_is_parameter: float = 0.0
    var_is_field: float = 0.0
    var_is_local: float = 0.0
    var_is_constant: float = 0.0
    var_has_similar_name: float = 0.0
    var_type_consistency: float = 0.0
    var_scope_match: float = 0.0
    num_variables_added: float = 0.0
    num_variables_removed: float = 0.0
    num_variables_modified: float = 0.0
    variable_name_length_avg: float = 0.0
    has_numeric_literal: float = 0.0
    has_string_literal: float = 0.0
    has_boolean_literal: float = 0.0
    has_null_literal: float = 0.0
    complexity_increase: float = 0.0
    data_flow_change: float = 0.0

@dataclass
class ContextFeatures:
    """Features about surrounding context (indices 40-49)."""
    cyclomatic_complexity: float = 0.0
    function_length: float = 0.0
    error_type_match: float = 0.0
    similar_to_historical_fix: float = 0.0
    template_confidence: float = 0.0
    location_rank: float = 0.0
    ast_depth: float = 0.0
    num_variables_in_scope: float = 0.0
    is_in_loop: float = 0.0
    is_in_exception_handler: float = 0.0

@dataclass
class FeatureVector:
    """Complete 50-dimensional feature vector."""
    modification: ModificationFeatures = field(default_factory=ModificationFeatures)
    program_value: ProgramValueFeatures = field(default_factory=ProgramValueFeatures)
    context: ContextFeatures = field(default_factory=ContextFeatures)
    
    def to_numpy(self) -> np.ndarray:
        """Convert to 50-dimensional numpy array."""
        mod = self.modification
        pv = self.program_value
        ctx = self.context
        
        return np.array([
            # Modification features (0-19)
            mod.is_condition_change,
            mod.is_null_guard,
            mod.is_bounds_check,
            mod.is_variable_replace,
            mod.is_method_change,
            mod.adds_loop,
            mod.adds_exception_handler,
            mod.modifies_arithmetic,
            mod.changes_control_flow,
            mod.adds_type_check,
            mod.deletes_code,
            mod.adds_new_variable,
            mod.changes_method_signature,
            mod.adds_import,
            mod.modifies_string_literal,
            mod.adds_logging,
            mod.adds_assertion,
            mod.changes_comparison_operator,
            mod.adds_method_call,
            mod.removes_method_call,
            
            # Program value features (20-39)
            pv.var_used_in_condition,
            pv.var_used_in_assignment,
            pv.var_is_return_value,
            pv.var_is_parameter,
            pv.var_is_field,
            pv.var_is_local,
            pv.var_is_constant,
            pv.var_has_similar_name,
            pv.var_type_consistency,
            pv.var_scope_match,
            pv.num_variables_added,
            pv.num_variables_removed,
            pv.num_variables_modified,
            pv.variable_name_length_avg,
            pv.has_numeric_literal,
            pv.has_string_literal,
            pv.has_boolean_literal,
            pv.has_null_literal,
            pv.complexity_increase,
            pv.data_flow_change,
            
            # Context features (40-49)
            ctx.cyclomatic_complexity,
            ctx.function_length,
            ctx.error_type_match,
            ctx.similar_to_historical_fix,
            ctx.template_confidence,
            ctx.location_rank,
            ctx.ast_depth,
            ctx.num_variables_in_scope,
            ctx.is_in_loop,
            ctx.is_in_exception_handler,
        ])
    
    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> 'FeatureVector':
        """Create from numpy array."""
        if len(arr) != 50:
            raise ValueError(f"Expected 50 features, got {len(arr)}")
        
        mod = ModificationFeatures(
            is_condition_change=arr[0],
            is_null_guard=arr[1],
            is_bounds_check=arr[2],
            is_variable_replace=arr[3],
            is_method_change=arr[4],
            adds_loop=arr[5],
            adds_exception_handler=arr[6],
            modifies_arithmetic=arr[7],
            changes_control_flow=arr[8],
            adds_type_check=arr[9],
            deletes_code=arr[10],
            adds_new_variable=arr[11],
            changes_method_signature=arr[12],
            adds_import=arr[13],
            modifies_string_literal=arr[14],
            adds_logging=arr[15],
            adds_assertion=arr[16],
            changes_comparison_operator=arr[17],
            adds_method_call=arr[18],
            removes_method_call=arr[19],
        )
        
        pv = ProgramValueFeatures(
            var_used_in_condition=arr[20],
            var_used_in_assignment=arr[21],
            var_is_return_value=arr[22],
            var_is_parameter=arr[23],
            var_is_field=arr[24],
            var_is_local=arr[25],
            var_is_constant=arr[26],
            var_has_similar_name=arr[27],
            var_type_consistency=arr[28],
            var_scope_match=arr[29],
            num_variables_added=arr[30],
            num_variables_removed=arr[31],
            num_variables_modified=arr[32],
            variable_name_length_avg=arr[33],
            has_numeric_literal=arr[34],
            has_string_literal=arr[35],
            has_boolean_literal=arr[36],
            has_null_literal=arr[37],
            complexity_increase=arr[38],
            data_flow_change=arr[39],
        )
        
        ctx = ContextFeatures(
            cyclomatic_complexity=arr[40],
            function_length=arr[41],
            error_type_match=arr[42],
            similar_to_historical_fix=arr[43],
            template_confidence=arr[44],
            location_rank=arr[45],
            ast_depth=arr[46],
            num_variables_in_scope=arr[47],
            is_in_loop=arr[48],
            is_in_exception_handler=arr[49],
        )
        
        return cls(modification=mod, program_value=pv, context=ctx)
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return [
            # Modification
            "is_condition_change", "is_null_guard", "is_bounds_check",
            "is_variable_replace", "is_method_change", "adds_loop",
            "adds_exception_handler", "modifies_arithmetic",
            "changes_control_flow", "adds_type_check", "deletes_code",
            "adds_new_variable", "changes_method_signature", "adds_import",
            "modifies_string_literal", "adds_logging", "adds_assertion",
            "changes_comparison_operator", "adds_method_call",
            "removes_method_call",
            # Program value
            "var_used_in_condition", "var_used_in_assignment",
            "var_is_return_value", "var_is_parameter", "var_is_field",
            "var_is_local", "var_is_constant", "var_has_similar_name",
            "var_type_consistency", "var_scope_match", "num_variables_added",
            "num_variables_removed", "num_variables_modified",
            "variable_name_length_avg", "has_numeric_literal",
            "has_string_literal", "has_boolean_literal", "has_null_literal",
            "complexity_increase", "data_flow_change",
            # Context
            "cyclomatic_complexity", "function_length", "error_type_match",
            "similar_to_historical_fix", "template_confidence",
            "location_rank", "ast_depth", "num_variables_in_scope",
            "is_in_loop", "is_in_exception_handler",
        ]
    
    def get_top_features(self, n: int = 5) -> Dict[str, float]:
        """Get top n features by absolute value."""
        vec = self.to_numpy()
        names = self.get_feature_names()
        
        indexed = [(names[i], vec[i]) for i in range(len(vec))]
        indexed.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return {name: val for name, val in indexed[:n]}


# =============================================================================
# FEATURE EXTRACTORS
# =============================================================================

class BaseFeatureExtractor:
    """Base class for feature extractors."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def extract(self, original_code: str, patched_code: str, context: Dict) -> FeatureVector:
        """Extract features from patch."""
        raise NotImplementedError


class ASTFeatureExtractor(BaseFeatureExtractor):
    """Extract features using AST analysis."""
    
    def extract(self, original_code: str, patched_code: str, context: Dict) -> FeatureVector:
        """Extract AST-based features."""
        features = FeatureVector()
        
        try:
            original_ast = ast.parse(original_code)
            patched_ast = ast.parse(patched_code)
        except SyntaxError:
            self.logger.warning("Syntax error in code, using fallback features")
            return features
        
        # Modification features
        features.modification = self._extract_modification_features(
            original_ast, patched_ast
        )
        
        # Program value features
        features.program_value = self._extract_program_value_features(
            original_ast, patched_ast, context
        )
        
        # Context features
        features.context = self._extract_context_features(
            patched_ast, context
        )
        
        return features
    
    def _extract_modification_features(
        self,
        original: ast.AST,
        patched: ast.AST
    ) -> ModificationFeatures:
        """Extract modification pattern features."""
        mod = ModificationFeatures()
        
        # Compare AST structures
        original_nodes = list(ast.walk(original))
        patched_nodes = list(ast.walk(patched))
        
        original_types = [type(n).__name__ for n in original_nodes]
        patched_types = [type(n).__name__ for n in patched_nodes]
        
        # Check for specific modifications
        mod.is_condition_change = self._has_condition_change(original, patched)
        mod.is_null_guard = self._has_null_guard(patched)
        mod.is_bounds_check = self._has_bounds_check(patched)
        mod.is_variable_replace = self._has_variable_replacement(original, patched)
        mod.is_method_change = self._has_method_change(original, patched)
        mod.adds_loop = "For" in patched_types and "For" not in original_types
        mod.adds_exception_handler = "Try" in patched_types and "Try" not in original_types
        mod.modifies_arithmetic = self._has_arithmetic_change(original, patched)
        mod.changes_control_flow = self._has_control_flow_change(original, patched)
        mod.adds_type_check = self._has_type_check(patched)
        mod.deletes_code = len(patched_nodes) < len(original_nodes)
        mod.adds_new_variable = self._has_new_variable(original, patched)
        mod.changes_method_signature = self._has_signature_change(original, patched)
        mod.adds_import = "Import" in patched_types and "Import" not in original_types
        mod.modifies_string_literal = self._has_string_literal_change(original, patched)
        mod.adds_logging = self._has_logging_addition(patched)
        mod.adds_assertion = "Assert" in patched_types and "Assert" not in original_types
        mod.changes_comparison_operator = self._has_comparison_change(original, patched)
        mod.adds_method_call = self._count_method_calls(patched) > self._count_method_calls(original)
        mod.removes_method_call = self._count_method_calls(patched) < self._count_method_calls(original)
        
        return mod
    
    def _extract_program_value_features(
        self,
        original: ast.AST,
        patched: ast.AST,
        context: Dict
    ) -> ProgramValueFeatures:
        """Extract variable usage features."""
        pv = ProgramValueFeatures()
        
        # Extract variables from both versions
        original_vars = self._extract_variables(original)
        patched_vars = self._extract_variables(patched)
        
        added_vars = patched_vars - original_vars
        removed_vars = original_vars - patched_vars
        common_vars = original_vars & patched_vars
        
        pv.num_variables_added = len(added_vars)
        pv.num_variables_removed = len(removed_vars)
        pv.num_variables_modified = len(common_vars)  # Simplified
        
        if added_vars or common_vars:
            all_vars = added_vars | common_vars
            pv.variable_name_length_avg = np.mean([len(v) for v in all_vars])
        
        # Check variable roles
        for var in patched_vars:
            if self._is_in_condition(var, patched):
                pv.var_used_in_condition = 1.0
            if self._is_in_assignment(var, patched):
                pv.var_used_in_assignment = 1.0
            if self._is_return_value(var, patched):
                pv.var_is_return_value = 1.0
        
        # Check for literals
        pv.has_numeric_literal = self._has_literal_type(patched, ast.Num)
        pv.has_string_literal = self._has_literal_type(patched, ast.Str) if hasattr(ast, 'Str') else False
        pv.has_boolean_literal = self._has_literal_type(patched, ast.NameConstant)
        pv.has_null_literal = self._has_none_literal(patched)
        
        # Complexity change
        pv.complexity_increase = self._calculate_complexity(patched) - self._calculate_complexity(original)
        
        return pv
    
    def _extract_context_features(
        self,
        patched: ast.AST,
        context: Dict
    ) -> ContextFeatures:
        """Extract surrounding context features."""
        ctx = ContextFeatures()
        
        # Complexity metrics
        ctx.cyclomatic_complexity = self._calculate_cyclomatic_complexity(patched)
        ctx.function_length = len(list(ast.walk(patched)))
        ctx.ast_depth = self._calculate_ast_depth(patched)
        ctx.num_variables_in_scope = len(self._extract_variables(patched))
        
        # Error type matching
        error_type = context.get('error_type', '').lower()
        if 'null' in error_type or 'none' in error_type:
            ctx.error_type_match = 1.0 if self._has_null_guard(patched) else 0.0
        elif 'bounds' in error_type or 'index' in error_type:
            ctx.error_type_match = 1.0 if self._has_bounds_check(patched) else 0.0
        
        # Template confidence from context
        ctx.template_confidence = context.get('template_confidence', 0.5)
        ctx.location_rank = context.get('location_rank', 1)
        
        # Structural context
        ctx.is_in_loop = self._is_inside_loop(patched)
        ctx.is_in_exception_handler = self._is_inside_try(patched)
        
        return ctx
    
    # Helper methods for feature detection
    
    def _has_condition_change(self, original: ast.AST, patched: ast.AST) -> float:
        """Detect if condition was modified."""
        original_ifs = [n for n in ast.walk(original) if isinstance(n, ast.If)]
        patched_ifs = [n for n in ast.walk(patched) if isinstance(n, ast.If)]
        
        if len(patched_ifs) != len(original_ifs):
            return 1.0
        
        for orig, new in zip(original_ifs, patched_ifs):
            if ast.dump(orig.test) != ast.dump(new.test):
                return 1.0
        
        return 0.0
    
    def _has_null_guard(self, tree: ast.AST) -> float:
        """Detect null/None check."""
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                test_str = ast.unparse(node.test) if hasattr(ast, 'unparse') else ""
                if 'none' in test_str.lower() or 'null' in test_str.lower():
                    return 1.0
        return 0.0
    
    def _has_bounds_check(self, tree: ast.AST) -> float:
        """Detect array bounds check."""
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                test_str = ast.unparse(node.test) if hasattr(ast, 'unparse') else ""
                if any(x in test_str for x in ['len(', '<', '>', '>=', '<=']):
                    return 1.0
        return 0.0
    
    def _has_variable_replacement(self, original: ast.AST, patched: ast.AST) -> float:
        """Detect variable replacement."""
        orig_names = {n.id for n in ast.walk(original) if isinstance(n, ast.Name)}
        patch_names = {n.id for n in ast.walk(patched) if isinstance(n, ast.Name)}
        
        if orig_names != patch_names:
            return 1.0
        return 0.0
    
    def _has_method_change(self, original: ast.AST, patched: ast.AST) -> float:
        """Detect method call changes."""
        orig_calls = {ast.dump(n) for n in ast.walk(original) if isinstance(n, ast.Call)}
        patch_calls = {ast.dump(n) for n in ast.walk(patched) if isinstance(n, ast.Call)}
        
        if orig_calls != patch_calls:
            return 1.0
        return 0.0
    
    def _has_arithmetic_change(self, original: ast.AST, patched: ast.AST) -> float:
        """Detect arithmetic operator changes."""
        orig_ops = {type(n.op).__name__ for n in ast.walk(original) if isinstance(n, ast.BinOp)}
        patch_ops = {type(n.op).__name__ for n in ast.walk(patched) if isinstance(n, ast.BinOp)}
        
        if orig_ops != patch_ops:
            return 1.0
        return 0.0
    
    def _has_control_flow_change(self, original: ast.AST, patched: ast.AST) -> float:
        """Detect control flow structure changes."""
        orig_flow = [type(n).__name__ for n in ast.walk(original) 
                     if isinstance(n, (ast.If, ast.For, ast.While, ast.Try))]
        patch_flow = [type(n).__name__ for n in ast.walk(patched) 
                      if isinstance(n, (ast.If, ast.For, ast.While, ast.Try))]
        
        return 1.0 if orig_flow != patch_flow else 0.0
    
    def _has_type_check(self, tree: ast.AST) -> float:
        """Detect type checking code."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_str = ast.unparse(node.func) if hasattr(ast, 'unparse') else ""
                if 'isinstance' in func_str or 'type(' in func_str:
                    return 1.0
        return 0.0
    
    def _has_new_variable(self, original: ast.AST, patched: ast.AST) -> float:
        """Detect new variable addition."""
        orig_vars = self._extract_variables(original)
        patch_vars = self._extract_variables(patched)
        return 1.0 if len(patch_vars) > len(orig_vars) else 0.0
    
    def _has_signature_change(self, original: ast.AST, patched: ast.AST) -> float:
        """Detect function signature change."""
        orig_funcs = [n for n in ast.walk(original) if isinstance(n, ast.FunctionDef)]
        patch_funcs = [n for n in ast.walk(patched) if isinstance(n, ast.FunctionDef)]
        
        for orig, new in zip(orig_funcs, patch_funcs):
            if len(orig.args.args) != len(new.args.args):
                return 1.0
        return 0.0
    
    def _has_string_literal_change(self, original: ast.AST, patched: ast.AST) -> float:
        """Detect string literal modification."""
        orig_strs = {ast.dump(n) for n in ast.walk(original) if isinstance(n, ast.Str) if hasattr(ast, 'Str')}
        patch_strs = {ast.dump(n) for n in ast.walk(patched) if isinstance(n, ast.Str) if hasattr(ast, 'Str')}
        return 1.0 if orig_strs != patch_strs else 0.0
    
    def _has_logging_addition(self, tree: ast.AST) -> float:
        """Detect logging addition."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_str = ast.unparse(node.func) if hasattr(ast, 'unparse') else ""
                if 'log' in func_str.lower():
                    return 1.0
        return 0.0
    
    def _has_comparison_change(self, original: ast.AST, patched: ast.AST) -> float:
        """Detect comparison operator change."""
        orig_cmps = {type(n.ops[0]).__name__ for n in ast.walk(original) if isinstance(n, ast.Compare)}
        patch_cmps = {type(n.ops[0]).__name__ for n in ast.walk(patched) if isinstance(n, ast.Compare)}
        return 1.0 if orig_cmps != patch_cmps else 0.0
    
    def _count_method_calls(self, tree: ast.AST) -> int:
        """Count method calls in AST."""
        return len([n for n in ast.walk(tree) if isinstance(n, ast.Call)])
    
    def _extract_variables(self, tree: ast.AST) -> Set[str]:
        """Extract all variable names from AST."""
        return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    
    def _is_in_condition(self, var: str, tree: ast.AST) -> bool:
        """Check if variable is used in condition."""
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                test_str = ast.unparse(node.test) if hasattr(ast, 'unparse') else ""
                if var in test_str:
                    return True
        return False
    
    def _is_in_assignment(self, var: str, tree: ast.AST) -> bool:
        """Check if variable is assignment target."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == var:
                        return True
        return False
    
    def _is_return_value(self, var: str, tree: ast.AST) -> bool:
        """Check if variable is returned."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Return):
                ret_str = ast.unparse(node.value) if hasattr(ast, 'unparse') else ""
                if var in ret_str:
                    return True
        return False
    
    def _has_literal_type(self, tree: ast.AST, literal_type: type) -> float:
        """Check for specific literal type."""
        return 1.0 if any(isinstance(n, literal_type) for n in ast.walk(tree)) else 0.0
    
    def _has_none_literal(self, tree: ast.AST) -> float:
        """Check for None literal."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value is None:
                return 1.0
            if isinstance(node, ast.NameConstant) and node.value is None:
                return 1.0
        return 0.0
    
    def _calculate_complexity(self, tree: ast.AST) -> float:
        """Calculate cyclomatic-like complexity."""
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.ExceptHandler)):
                complexity += 1
        return float(complexity)
    
    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> float:
        """Calculate McCabe cyclomatic complexity."""
        # Simplified version
        edges = 0
        nodes = 0
        for node in ast.walk(tree):
            nodes += 1
            if isinstance(node, (ast.If, ast.For, ast.While)):
                edges += 2  # True and false branches
            elif isinstance(node, ast.FunctionDef):
                edges += 1
        
        return float(edges - nodes + 2) if nodes > 0 else 1.0
    
    def _calculate_ast_depth(self, tree: ast.AST) -> float:
        """Calculate maximum AST depth."""
        def depth(node, current=0):
            if not list(ast.iter_child_nodes(node)):
                return current
            return max(depth(child, current + 1) for child in ast.iter_child_nodes(node))
        
        return float(depth(tree))
    
    def _is_inside_loop(self, tree: ast.AST) -> float:
        """Check if code is inside loop."""
        # Simplified - would need parent tracking
        return 1.0 if any(isinstance(n, (ast.For, ast.While)) for n in ast.walk(tree)) else 0.0
    
    def _is_inside_try(self, tree: ast.AST) -> float:
        """Check if code is inside try block."""
        return 1.0 if any(isinstance(n, ast.Try) for n in ast.walk(tree)) else 0.0


class TextFeatureExtractor(BaseFeatureExtractor):
    """Extract features using text analysis (fallback for syntax errors)."""
    
    def extract(self, original_code: str, patched_code: str, context: Dict) -> FeatureVector:
        """Extract text-based features."""
        features = FeatureVector()
        
        # Line-based features
        orig_lines = original_code.split('\n')
        patch_lines = patched_code.split('\n')
        
        features.modification.deletes_code = len(patch_lines) < len(orig_lines)
        features.program_value.num_variables_added = self._estimate_variables_added(
            original_code, patched_code
        )
        
        # Keyword-based features
        keywords = ['if', 'for', 'while', 'try', 'except', 'assert', 'isinstance']
        for kw in keywords:
            if kw in patched_code and kw not in original_code:
                if kw == 'if':
                    features.modification.is_condition_change = 1.0
                elif kw in ['for', 'while']:
                    features.modification.adds_loop = 1.0
                elif kw in ['try', 'except']:
                    features.modification.adds_exception_handler = 1.0
                elif kw == 'assert':
                    features.modification.adds_assertion = 1.0
                elif kw == 'isinstance':
                    features.modification.adds_type_check = 1.0
        
        # Null check detection
        if 'is not None' in patched_code or 'is None' in patched_code:
            features.modification.is_null_guard = 1.0
        
        # Bounds check detection
        if any(x in patched_code for x in ['len(', '<', '>']) and 'if' in patched_code:
            features.modification.is_bounds_check = 1.0
        
        return features
    
    def _estimate_variables_added(self, original: str, patched: str) -> float:
        """Estimate number of new variables."""
        # Simple heuristic: count new identifiers
        import re
        orig_ids = set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', original))
        patch_ids = set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', patched))
        return float(len(patch_ids - orig_ids))


class HistoricalFeatureExtractor(BaseFeatureExtractor):
    """Extract features based on similarity to historical fixes."""
    
    def __init__(self, historical_patches_dir: Optional[Path] = None):
        super().__init__()
        self.historical_dir = historical_patches_dir
        self.patch_database: List[Dict] = []
        self.vectorizer = None
        
        if SKLEARN_AVAILABLE and historical_patches_dir:
            self._load_historical_patches()
    
    def _load_historical_patches(self):
        """Load and index historical patches."""
        if not self.historical_dir or not self.historical_dir.exists():
            return
        
        for patch_file in self.historical_dir.glob("*.json"):
            with open(patch_file) as f:
                self.patch_database.append(json.load(f))
        
        # Build TF-IDF index if available
        if self.patch_database and SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(max_features=1000)
            patch_texts = [
                p.get('patched_code', '') for p in self.patch_database
            ]
            self.vectorizer.fit(patch_texts)
        
        self.logger.info(f"Loaded {len(self.patch_database)} historical patches")
    
    def extract(self, original_code: str, patched_code: str, context: Dict) -> FeatureVector:
        """Extract historical similarity features."""
        features = FeatureVector()
        
        if not self.patch_database or not SKLEARN_AVAILABLE:
            features.context.similar_to_historical_fix = 0.5  # Neutral
            return features
        
        # Calculate similarity to historical patches
        if self.vectorizer:
            patch_vec = self.vectorizer.transform([patched_code])
            
            max_similarity = 0.0
            for hist_patch in self.patch_database:
                hist_vec = self.vectorizer.transform([hist_patch.get('patched_code', '')])
                similarity = cosine_similarity(patch_vec, hist_vec)[0][0]
                max_similarity = max(max_similarity, similarity)
            
            features.context.similar_to_historical_fix = float(max_similarity)
        
        return features


# =============================================================================
# UNIFIED FEATURE EXTRACTOR
# =============================================================================

class FeatureExtractor:
    """
    Unified feature extractor combining multiple strategies.
    
    Falls back from AST-based to text-based on syntax errors.
    """
    
    def __init__(
        self,
        use_historical: bool = True,
        historical_patches_dir: Optional[Path] = None
    ):
        self.ast_extractor = ASTFeatureExtractor()
        self.text_extractor = TextFeatureExtractor()
        self.historical_extractor = HistoricalFeatureExtractor(
            historical_patches_dir
        ) if use_historical else None
        
        self.logger = logging.getLogger(__name__)
    
    def extract(
        self,
        original_code: str,
        patched_code: str,
        context: Optional[Dict] = None
    ) -> FeatureVector:
        """
        Extract complete feature vector from patch.
        
        Tries AST-based first, falls back to text-based on error.
        """
        context = context or {}
        
        # Try AST-based extraction
        try:
            features = self.ast_extractor.extract(original_code, patched_code, context)
            extraction_method = "ast"
        except SyntaxError:
            self.logger.warning("Syntax error, using text-based extraction")
            features = self.text_extractor.extract(original_code, patched_code, context)
            extraction_method = "text"
        
        # Add historical features if available
        if self.historical_extractor:
            hist_features = self.historical_extractor.extract(
                original_code, patched_code, context
            )
            # Merge historical features
            features.context.similar_to_historical_fix = \
                hist_features.context.similar_to_historical_fix
        
        # Add metadata
        features.context.template_confidence = context.get('template_confidence', 0.5)
        features.context.location_rank = context.get('location_rank', 1)
        
        self.logger.debug(f"Extracted features using {extraction_method} method")
        
        return features
    
    def extract_batch(
        self,
        patches: List[Tuple[str, str, Dict]]
    ) -> List[FeatureVector]:
        """
        Extract features for multiple patches.
        
        Args:
            patches: List of (original_code, patched_code, context) tuples
        
        Returns:
            List of feature vectors
        """
        return [
            self.extract(orig, patched, ctx)
            for orig, patched, ctx in patches
        ]
    
    def get_feature_importance_explanation(
        self,
        features: FeatureVector,
        top_k: int = 5
    ) -> str:
        """
        Generate human-readable explanation of top features.
        """
        top = features.get_top_features(top_k)
        
        explanation = "Top contributing features:\n"
        for name, value in top.items():
            direction = "increases" if value > 0 else "decreases"
            magnitude = "strongly" if abs(value) > 0.5 else "moderately"
            explanation += f"  - {name}: {value:.3f} ({magnitude} {direction} score)\n"
        
        return explanation


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_default_extractor: Optional[FeatureExtractor] = None

def get_feature_extractor(
    use_historical: bool = True,
    historical_dir: Optional[Path] = None
) -> FeatureExtractor:
    """Get or create default feature extractor."""
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = FeatureExtractor(
            use_historical=use_historical,
            historical_patches_dir=historical_dir
        )
    return _default_extractor


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example patches
    original = """
def process(data, index):
    return data[index]
"""
    
    patched_null_guard = """
def process(data, index):
    if data is not None:
        return data[index]
    return None
"""
    
    patched_bounds_check = """
def process(data, index):
    if 0 <= index < len(data):
        return data[index]
    raise IndexError("Out of bounds")
"""
    
    # Extract features
    extractor = FeatureExtractor(use_historical=False)
    
    print("=== Null Guard Patch ===")
    features = extractor.extract(original, patched_null_guard, {
        'error_type': 'NullPointerException',
        'location_rank': 1,
        'template_confidence': 0.9
    })
    
    print(f"Features (first 10): {features.to_numpy()[:10]}")
    print(f"Is null guard: {features.modification.is_null_guard}")
    print(f"Is bounds check: {features.modification.is_bounds_check}")
    print(extractor.get_feature_importance_explanation(features))
    
    print("\n=== Bounds Check Patch ===")
    features = extractor.extract(original, patched_bounds_check, {
        'error_type': 'IndexError',
        'location_rank': 2,
        'template_confidence': 0.85
    })
    
    print(f"Is null guard: {features.modification.is_null_guard}")
    print(f"Is bounds check: {features.modification.is_bounds_check}")
    print(f"Error type match: {features.context.error_type_match}")
    print(extractor.get_feature_importance_explanation(features))
