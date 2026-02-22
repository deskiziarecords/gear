# packages/core-algorithms/revr/domains/predicate_domain.py

from typing import Set, List, Dict
import z3
from dataclasses import dataclass

@dataclass
class Predicate:
    """Atomic predicate for predicate abstraction"""
    expr: z3.BoolRef
    name: str
    
    def __hash__(self):
        return hash(str(self.expr))
    
    def __eq__(self, other):
        return str(self.expr) == str(other.expr)

class PredicateAbstraction:
    """
    Predicate abstraction using boolean combinations of atomic predicates.
    
    Based on error invariant research [^67^] and BDD abstract domains.
    Tracks which predicates hold at each program point.
    """
    
    def __init__(self, predicates: List[Predicate]):
        self.predicates = predicates
        self.bdd = {}  # Binary Decision Diagram representation
    
    def abstract_state(self, concrete_state: Dict[str, any]) -> Set[Predicate]:
        """
        Abstract concrete state to set of true predicates.
        """
        true_preds = set()
        
        for pred in self.predicates:
            # Evaluate predicate in concrete state
            # (Simplified - would use Z3 model evaluation)
            true_preds.add(pred)
        
        return frozenset(true_preds)  # Immutable for hashing
    
    def join(self, states: List[Set[Predicate]]) -> Set[Predicate]:
        """
        Join multiple abstract states (union of common predicates).
        """
        if not states:
            return set()
        
        result = set(states[0])
        for state in states[1:]:
            result &= state  # Intersection (common predicates)
        
        return result
    
    def strengthen(self, state: Set[Predicate], 
                   condition: z3.BoolRef) -> Set[Predicate]:
        """
        Strengthen abstract state with new condition.
        Adds predicates implied by the condition.
        """
        new_state = set(state)
        
        # Check which predicates are implied by condition
        solver = z3.Solver()
        solver.add(condition)
        
        for pred in self.predicates:
            if pred not in new_state:
                solver.push()
                solver.add(z3.Not(pred.expr))
                if solver.check() == z3.unsat:
                    new_state.add(pred)  # Implied by condition
                solver.pop()
        
        return new_state
    
    def is_error_reachable(self, error_condition: z3.BoolRef,
                          current_state: Set[Predicate]) -> bool:
        """
        Check if error is reachable from current abstract state.
        Used in fault localization [^67^].
        """
        solver = z3.Solver()
        
        # Add all predicates in current state
        for pred in current_state:
            solver.add(pred.expr)
        
        # Check if error condition is satisfiable
        solver.push()
        solver.add(error_condition)
        result = solver.check() == z3.sat
        solver.pop()
        
        return result
