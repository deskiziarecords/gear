# auto_repair/synthesis/fu/validator.py

"""
Semantic Patch Validator using LLMs.

Validates patch correctness beyond test suite:
- Semantic equivalence checking
- Regression risk assessment
- Maintainability evaluation
- Security impact analysis
"""

import json
import ast
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from enum import Enum, auto
from pathlib import Path
import hashlib

# LLM imports
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Internal imports
from auto_repair.config import get_config
from auto_repair.synthesis.pfs.templates import PatchInstance


# =============================================================================
# DATA CLASSES
# =============================================================================

class ValidationAspect(Enum):
    """Aspects of patch validation."""
    CORRECTNESS = auto()      # Does it fix the bug?
    MAINTAINABILITY = auto()  # Is it clean, readable code?
    PERFORMANCE = auto()      # Any performance implications?
    SECURITY = auto()         # Security risks?
    COMPLETENESS = auto()     # Does it handle all cases?
    REGRESSION_RISK = auto()  # Risk of breaking other things?

class RiskLevel(Enum):
    """Risk assessment levels."""
    CRITICAL = "critical"      # Must fix before deployment
    HIGH = "high"             # Strongly recommended to address
    MEDIUM = "medium"         # Should review
    LOW = "low"               # Minor concern
    NONE = "none"             # No risk identified

@dataclass
class ValidationFinding:
    """Single validation finding."""
    aspect: ValidationAspect
    severity: RiskLevel
    description: str
    evidence: str = ""
    recommendation: str = ""
    confidence: float = 0.0  # LLM confidence in this finding

@dataclass
class SemanticValidationResult:
    """Complete semantic validation result."""
    patch_id: str
    is_valid: bool = False
    overall_confidence: float = 0.0
    overall_risk: RiskLevel = RiskLevel.MEDIUM
    
    # Detailed findings
    findings: List[ValidationFinding] = field(default_factory=list)
    
    # Scores by aspect (0-1)
    correctness_score: float = 0.0
    maintainability_score: float = 0.0
    performance_score: float = 0.0
    security_score: float = 0.0
    completeness_score: float = 0.0
    
    # Analysis
    semantic_equivalence: Optional[str] = None  # Comparison to intended behavior
    behavior_changes: List[str] = field(default_factory=list)
    invariant_violations: List[str] = field(default_factory=list)
    
    # Metadata
    validation_time_ms: float = 0.0
    tokens_used: int = 0
    llm_calls: int = 0
    
    def get_critical_findings(self) -> List[ValidationFinding]:
        """Get findings that must be addressed."""
        return [
            f for f in self.findings 
            if f.severity in [RiskLevel.CRITICAL, RiskLevel.HIGH]
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "patch_id": self.patch_id,
            "is_valid": self.is_valid,
            "overall_confidence": self.overall_confidence,
            "overall_risk": self.overall_risk.value,
            "scores": {
                "correctness": self.correctness_score,
                "maintainability": self.maintainability_score,
                "performance": self.performance_score,
                "security": self.security_score,
                "completeness": self.completeness_score,
            },
            "findings": [
                {
                    "aspect": f.aspect.name,
                    "severity": f.severity.value,
                    "description": f.description,
                    "confidence": f.confidence
                }
                for f in self.findings
            ],
            "critical_issues": len(self.get_critical_findings()),
            "metadata": {
                "validation_time_ms": self.validation_time_ms,
                "tokens_used": self.tokens_used,
                "llm_calls": self.llm_calls
            }
        }

@dataclass
class ValidationContext:
    """Context for validation."""
    # Code context
    original_code: str = ""
    patched_code: str = ""
    surrounding_context: str = ""  # 50 lines around patch
    
    # Project context
    project_type: str = ""  # web, ml, embedded, etc.
    coding_standards: str = ""  # PEP8, Google, etc.
    test_coverage: float = 0.0
    
    # Bug context
    error_type: str = ""
    error_message: str = ""
    root_cause: Optional[str] = None
    
    # Historical context
    similar_patches: List[Dict] = field(default_factory=list)
    common_mistakes: List[str] = field(default_factory=list)


# =============================================================================
# LLM PROVIDER (shared with explainer)
# =============================================================================

class LLMProvider(ABC):
    """Abstract LLM provider."""
    
    def __init__(self, model: str, temperature: float = 0.1, max_tokens: int = 2000):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, int]:
        pass

class OpenAIValidatorProvider(LLMProvider):
    """OpenAI provider for validation."""
    
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None, **kwargs):
        super().__init__(model, **kwargs)
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI not installed")
        self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, int]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"}  # Force JSON
        )
        
        text = response.choices[0].message.content
        tokens = response.usage.total_tokens if response.usage else 0
        return text, tokens

class AnthropicValidatorProvider(LLMProvider):
    """Anthropic provider for validation."""
    
    def __init__(self, model: str = "claude-3-opus-20240229", api_key: Optional[str] = None, **kwargs):
        super().__init__(model, **kwargs)
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic not installed")
        self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, int]:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt or "You are a code validation assistant. Respond in JSON.",
            messages=[{"role": "user", "content": prompt}]
        )
        
        text = message.content[0].text
        tokens = message.usage.input_tokens + message.usage.output_tokens if message.usage else 0
        
        # Try to extract JSON if wrapped in markdown
        import re
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)
        
        return text, tokens


# =============================================================================
# VALIDATION STRATEGIES
# =============================================================================

class ValidationStrategy(ABC):
    """Base class for validation strategies."""
    
    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def validate(self, patch: PatchInstance, context: ValidationContext) -> List[ValidationFinding]:
        """Execute validation strategy."""
        pass

class CorrectnessStrategy(ValidationStrategy):
    """Validate that patch actually fixes the bug."""
    
    SYSTEM_PROMPT = """You are a code correctness validator. Analyze if the patch properly fixes the described bug.
Focus on: Does the change address the root cause? Is the logic sound? Are edge cases handled?
Respond in JSON with: is_correct (bool), confidence (0-1), reasoning (str), edge_cases (list)."""
    
    def validate(self, patch: PatchInstance, context: ValidationContext) -> List[ValidationFinding]:
        prompt = f"""
BUG: {context.error_type} - {context.error_message}
ROOT CAUSE: {context.root_cause or "Unknown"}

ORIGINAL CODE:
```python
{context.original_code}
