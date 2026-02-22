# packages/core-algorithms/revr/domains/interval_domain.py

from typing import Union, Optional
import z3

class Interval:
    """
    Interval abstraction for numerical variables.
    [l, h] represents all values x where l <= x <= h.
    
    Based on Cousot's abstract interpretation [^69^][^70^].
    """
    
    def __init__(self, low: Optional[float] = None, high: Optional[float] = None):
        self.low = low if low is not None else float('-inf')
        self.high = high if high is not None else float('inf')
    
    def __repr__(self):
        return f"[{self.low}, {self.high}]"
    
    def is_empty(self) -> bool:
        return self.low > self.high
    
    def is_singleton(self) -> bool:
        return self.low == self.high
    
    def contains(self, val: float) -> bool:
        return self.low <= val <= self.high
    
    def join(self, other: 'Interval') -> 'Interval':
        """Union (least upper bound)"""
        return Interval(
            min(self.low, other.low),
            max(self.high, other.high)
        )
    
    def meet(self, other: 'Interval') -> 'Interval':
        """Intersection (greatest lower bound)"""
        return Interval(
            max(self.low, other.low),
            min(self.high, other.high)
        )
    
    def widen(self, other: 'Interval') -> 'Interval':
        """
        Widening operator for loop analysis.
        Ensures termination by accelerating to infinity.
        """
        new_low = self.low if self.low <= other.low else float('-inf')
        new_high = self.high if self.high >= other.high else float('inf')
        return Interval(new_low, new_high)
    
    def narrow(self, other: 'Interval') -> 'Interval':
        """
        Narrowing operator to recover precision after widening.
        """
        new_low = self.low if self.low != float('-inf') else other.low
        new_high = self.high if self.high != float('inf') else other.high
        return Interval(new_low, new_high)
    
    # Arithmetic operations with outward rounding
    def __add__(self, other: 'Interval') -> 'Interval':
        return Interval(self.low + other.low, self.high + other.high)
    
    def __sub__(self, other: 'Interval') -> 'Interval':
        return Interval(self.low - other.high, self.high - other.low)
    
    def __mul__(self, other: 'Interval') -> 'Interval':
        products = [
            self.low * other.low, self.low * other.high,
            self.high * other.low, self.high * other.high
        ]
        return Interval(min(products), max(products))
    
    def __truediv__(self, other: 'Interval') -> 'Interval':
        if other.contains(0):
            return Interval(float('-inf'), float('inf'))  # Division by zero possible
        products = [
            self.low / other.low, self.low / other.high,
            self.high / other.low, self.high / other.high
        ]
        return Interval(min(products), max(products))
    
    def to_z3(self, var_name: str) -> z3.ExprRef:
        """Convert to Z3 constraints"""
        var = z3.Real(var_name)
        return z3.And(var >= self.low, var <= self.high)
