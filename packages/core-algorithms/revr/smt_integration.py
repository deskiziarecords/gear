# packages/core-algorithms/revr/smt_integration.py

import z3
import ast
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass

@dataclass
class SymbolicValue:
    """Represents a symbolic value with type information"""
    z3_var: z3.ExprRef
    python_type: type
    constraints: list  # Additional constraints (e.g., len for strings)

class PythonZ3Translator:
    """
    Translate Python operations to Z3 constraints.
    Handles dynamic typing by tracking type information alongside Z3 expressions.
    
    Based on LLM-Sym's approach [^63^] and Fuzzing Book [^74^].
    """
    
    def __init__(self):
        self.solver = z3.Solver()
        self.symbolic_vars: Dict[str, SymbolicValue] = {}
        self.type_constraints = []
    
    def create_symbolic_var(self, name: str, python_type: type = None) -> SymbolicValue:
        """
        Create a symbolic variable with appropriate Z3 sort.
        
        For Python's dynamic types, we use:
        - BitVec(64) for integers (handles arbitrary precision)
        - Real for floats
        - String for strings (Z3SeqRef)
        - Bool for booleans
        """
        if python_type == int:
            # Use 64-bit bitvector for Python integers
            z3_var = z3.BitVec(name, 64)
        elif python_type == float:
            z3_var = z3.Real(name)
        elif python_type == str:
            z3_var = z3.String(name)
        elif python_type == bool:
            z3_var = z3.Bool(name)
        else:
            # Default to BitVec for unknown types (Python dynamic)
            z3_var = z3.BitVec(name, 64)
        
        sym_val = SymbolicValue(z3_var, python_type or object, [])
        self.symbolic_vars[name] = sym_val
        return sym_val
    
    def translate_binop(self, op: ast.AST, left: SymbolicValue, 
                        right: SymbolicValue) -> SymbolicValue:
        """Translate binary operations with type coercion"""
        
        # Handle numeric operations
        if left.python_type in (int, float) and right.python_type in (int, float):
            return self._translate_numeric_op(op, left, right)
        
        # Handle string operations
        if left.python_type == str or right.python_type == str:
            return self._translate_string_op(op, left, right)
        
        # Default: treat as bitvectors
        return self._translate_bitvec_op(op, left, right)
    
    def _translate_numeric_op(self, op: ast.AST, left: SymbolicValue, 
                              right: SymbolicValue) -> SymbolicValue:
        """Translate numeric operations with proper handling of int/float"""
        
        l, r = left.z3_var, right.z3_var
        
        # Promote to Real if either operand is float
        if left.python_type == float or right.python_type == float:
            if left.python_type == int:
                l = z3.ToReal(l)
            if right.python_type == int:
                r = z3.ToReal(r)
            result_type = float
        else:
            result_type = int
        
        # Map Python operators to Z3
        if isinstance(op, ast.Add):
            result = l + r
        elif isinstance(op, ast.Sub):
            result = l - r
        elif isinstance(op, ast.Mult):
            result = l * r
        elif isinstance(op, ast.Div):
            # Python 3 true division
            if result_type == int:
                l, r = z3.ToReal(l), z3.ToReal(r)
                result_type = float
            result = l / r
        elif isinstance(op, ast.FloorDiv):
            result = l / r  # Z3 integer division
        elif isinstance(op, ast.Mod):
            result = l % r
        elif isinstance(op, ast.Pow):
            # Z3 doesn't support pow directly, use uninterpreted function
            result = z3.Function('pow', l.sort(), r.sort(), l.sort())(l, r)
        else:
            raise ValueError(f"Unsupported binary operator: {type(op)}")
        
        return SymbolicValue(result, result_type, [])
    
    def _translate_comparison(self, op: ast.AST, left: SymbolicValue, 
                              right: SymbolicValue) -> z3.BoolRef:
        """Translate comparison operations"""
        
        l, r = left.z3_var, right.z3_var
        
        # Handle type coercion for comparisons
        if left.python_type != right.python_type:
            if left.python_type == int and right.python_type == float:
                l = z3.ToReal(l)
            elif left.python_type == float and right.python_type == int:
                r = z3.ToReal(r)
        
        if isinstance(op, ast.Eq):
            return l == r
        elif isinstance(op, ast.NotEq):
            return l != r
        elif isinstance(op, ast.Lt):
            return l < r
        elif isinstance(op, ast.LtE):
            return l <= r
        elif isinstance(op, ast.Gt):
            return l > r
        elif isinstance(op, ast.GtE):
            return l >= r
        else:
            raise ValueError(f"Unsupported comparison: {type(op)}")
    
    def translate_condition(self, node: ast.AST, 
                          local_vars: Dict[str, SymbolicValue]) -> z3.BoolRef:
        """
        Translate an if condition to Z3 boolean.
        Recursively handles boolean operations (and, or, not).
        """
        if isinstance(node, ast.Compare):
            left = self._get_symbolic(node.left, local_vars)
            for op, comparator in zip(node.ops, node.comparators):
                right = self._get_symbolic(comparator, local_vars)
                return self._translate_comparison(op, left, right)
        
        elif isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                result = z3.And([
                    self.translate_condition(value, local_vars) 
                    for value in node.values
                ])
            elif isinstance(node.op, ast.Or):
                result = z3.Or([
                    self.translate_condition(value, local_vars) 
                    for value in node.values
                ])
            else:
                raise ValueError(f"Unsupported bool op: {type(node.op)}")
            return result
        
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return z3.Not(self.translate_condition(node.operand, local_vars))
        
        elif isinstance(node, ast.NameConstant):
            return z3.BoolVal(node.value)
        
        elif isinstance(node, ast.Name):
            # Boolean variable
            sym_val = local_vars.get(node.id)
            if sym_val and sym_val.python_type == bool:
                return sym_val.z3_var
        
        # Default: treat as truthy/falsy
        return z3.BoolVal(True)  # Simplified
    
    def _get_symbolic(self, node: ast.AST, 
                      local_vars: Dict[str, SymbolicValue]) -> SymbolicValue:
        """Get symbolic value for an AST node"""
        if isinstance(node, ast.Num):
            # Python 3.8+ uses ast.Constant
            if isinstance(node.n, int):
                return SymbolicValue(z3.BitVecVal(node.n, 64), int, [])
            else:
                return SymbolicValue(z3.RealVal(node.n), float, [])
        
        elif isinstance(node, ast.Constant):  # Python 3.8+
            if isinstance(node.value, int):
                return SymbolicValue(z3.BitVecVal(node.value, 64), int, [])
            elif isinstance(node.value, float):
                return SymbolicValue(z3.RealVal(node.value), float, [])
            elif isinstance(node.value, str):
                return SymbolicValue(z3.StringVal(node.value), str, [])
            elif isinstance(node.value, bool):
                return SymbolicValue(z3.BoolVal(node.value), bool, [])
        
        elif isinstance(node, ast.Name):
            if node.id in local_vars:
                return local_vars[node.id]
            elif node.id in self.symbolic_vars:
                return self.symbolic_vars[node.id]
            else:
                # Create new symbolic variable on demand
                return self.create_symbolic_var(node.id)
        
        elif isinstance(node, ast.BinOp):
            left = self._get_symbolic(node.left, local_vars)
            right = self._get_symbolic(node.right, local_vars)
            return self.translate_binop(node.op, left, right)
        
        raise ValueError(f"Unsupported AST node type: {type(node)}")
    
    def solve_path_condition(self, path_condition: z3.BoolRef) -> Optional[Dict[str, Any]]:
        """
        Solve path condition using Z3.
        Returns concrete values that satisfy the condition, or None if unsat.
        """
        self.solver.push()
        self.solver.add(path_condition)
        
        if self.solver.check() == z3.sat:
            model = self.solver.model()
            result = {}
            
            for name, sym_val in self.symbolic_vars.items():
                z3_var = sym_val.z3_var
                if model[z3_var] is not None:
                    val = model[z3_var]
                    # Convert Z3 value to Python value
                    if sym_val.python_type == int:
                        result[name] = val.as_long()
                    elif sym_val.python_type == float:
                        result[name] = float(val.as_fraction())
                    elif sym_val.python_type == bool:
                        result[name] = z3.is_true(val)
                    else:
                        result[name] = str(val)
            
            self.solver.pop()
            return result
        
        self.solver.pop()
        return None
    
    def check_reachability(self, start_condition: z3.BoolRef, 
                          target_condition: z3.BoolRef) -> bool:
        """
        Check if target is reachable from start.
        Used in reverse execution to validate fault paths.
        """
        self.solver.push()
        self.solver.add(start_condition)
        self.solver.add(target_condition)
        
        result = self.solver.check() == z3.sat
        self.solver.pop()
        return result
