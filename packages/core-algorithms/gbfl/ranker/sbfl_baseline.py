# packages/core-algorithms/gbfl/ranker/sbfl_baseline.py

from dataclasses import dataclass
from math import sqrt
from typing import Dict, List, Callable
from enum import Enum

class SBFLFormula(Enum):
    TARANTULA = "tarantula"
    OCHIAI = "ochiai"
    OP2 = "op2"
    BARINEL = "barinel"
    DSTAR = "dstar"
    JACCARD = "jaccard"
    WONG1 = "wong1"
    WONG2 = "wong2"
    WONG3 = "wong3"
    AMPLE = "ample"

@dataclass
class Spectrum:
    """Test coverage spectrum for a program element"""
    ef: int  # Executed by Failed tests
    ep: int  # Executed by Passed tests
    nf: int  # Not executed by Failed tests  
    np: int  # Not executed by Passed tests
    
    @property
    def total_failed(self) -> int:
        return self.ef + self.nf
    
    @property
    def total_passed(self) -> int:
        return self.ep + self.np

class SBFLEngine:
    """
    Spectrum-Based Fault Localization engine implementing multiple formulas.
    
    Based on formulas from:
    - Jones & Harrold (Tarantula) [^17^]
    - Abreu et al. (Ochiai, Barinel) [^16^]
    - Naish et al. (Op1, Op2) [^17^]
    - Wong et al. (Wong1-3) [^17^]
    """
    
    def __init__(self, formula: SBFLFormula = SBFLFormula.OCHIAI):
        self.formula = formula
        self._formula_func = self._get_formula(formula)
    
    def _get_formula(self, formula: SBFLFormula) -> Callable[[Spectrum], float]:
        """Get the suspiciousness calculation function"""
        formulas = {
            SBFLFormula.TARANTULA: self._tarantula,
            SBFLFormula.OCHIAI: self._ochiai,
            SBFLFormula.OP2: self._op2,
            SBFLFormula.BARINEL: self._barinel,
            SBFLFormula.DSTAR: self._dstar,
            SBFLFormula.JACCARD: self._jaccard,
            SBFLFormula.WONG1: self._wong1,
            SBFLFormula.WONG2: self._wong2,
            SBFLFormula.WONG3: self._wong3,
            SBFLFormula.AMPLE: self._ample,
        }
        return formulas[formula]
    
    def calculate(self, spectrum: Spectrum) -> float:
        """Calculate suspiciousness score for a spectrum"""
        # Handle edge cases
        if spectrum.total_failed == 0:
            return 0.0  # No failing tests, can't be suspicious
        
        return self._formula_func(spectrum)
    
    def _tarantula(self, s: Spectrum) -> float:
        """
        Tarantula formula [^17^]:
        (ef/total_failed) / (ef/total_failed + ep/total_passed)
        """
        if s.ef == 0:
            return 0.0
        
        failed_ratio = s.ef / s.total_failed
        passed_ratio = s.ep / s.total_passed if s.total_passed > 0 else 0
        
        denom = failed_ratio + passed_ratio
        return failed_ratio / denom if denom > 0 else 0.0
    
    def _ochiai(self, s: Spectrum) -> float:
        """
        Ochiai formula [^16^]:
        ef / sqrt(total_failed * (ef + ep))
        
        Proven effective in multiple studies [^20^][^24^]
        """
        if s.ef == 0:
            return 0.0
        
        denom = sqrt(s.total_failed * (s.ef + s.ep))
        return s.ef / denom if denom > 0 else 0.0
    
    def _op2(self, s: Spectrum) -> float:
        """
        Op2 formula [^17^]:
        ef - (ep / (total_passed + 1))
        
        Optimal for specific program structures (ITE2) [^17^]
        """
        return s.ef - (s.ep / (s.total_passed + 1))
    
    def _barinel(self, s: Spectrum) -> float:
        """
        Barinel formula [^16^]:
        1 - (passed / (passed + failed))
        """
        if s.ep == 0:
            return 1.0 if s.ef > 0 else 0.0
        
        return 1 - (s.ep / (s.ep + s.ef))
    
    def _dstar(self, s: Spectrum, star: int = 2) -> float:
        """
        DStar formula [^15^]:
        (ef^star) / (ep + nf)
        
        star=2 is most thoroughly explored [^15^]
        """
        if s.ef == 0:
            return 0.0
        
        denom = s.ep + s.nf
        return (s.ef ** star) / denom if denom > 0 else float('inf')
    
    def _jaccard(self, s: Spectrum) -> float:
        """
        Jaccard formula [^17^]:
        ef / (ef + nf + ep)
        """
        denom = s.ef + s.nf + s.ep
        return s.ef / denom if denom > 0 else 0.0
    
    def _wong1(self, s: Spectrum) -> float:
        """Wong1: Simple fault count"""
        return float(s.ef)
    
    def _wong2(self, s: Spectrum) -> float:
        """Wong2: ef - ep"""
        return s.ef - s.ep
    
    def _wong3(self, s: Spectrum) -> float:
        """
        Wong3 formula [^17^]:
        ef - h, where h is a piecewise function of ep
        """
        if s.ep <= 2:
            h = s.ep
        elif s.ep <= 10:
            h = 2 + 0.1 * (s.ep - 2)
        else:
            h = 2.8 + 0.001 * (s.ep - 10)
        
        return s.ef - h
    
    def _ample(self, s: Spectrum) -> float:
        """
        AMPLE formula [^17^]:
        |ef/total_failed - ep/total_passed|
        """
        failed_ratio = s.ef / s.total_failed
        passed_ratio = s.ep / s.total_passed if s.total_passed > 0 else 0
        return abs(failed_ratio - passed_ratio)
    
    def rank_elements(self, spectra: Dict[str, Spectrum]) -> List[Tuple[str, float]]:
        """
        Rank program elements by suspiciousness score.
        Returns list of (element_id, score) sorted by score descending.
        """
        scores = {
            elem_id: self.calculate(spec) 
            for elem_id, spec in spectra.items()
        }
        
        # Sort by score descending, then by element_id for determinism
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return ranked
