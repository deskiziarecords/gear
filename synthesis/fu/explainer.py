# auto_repair/synthesis/fu/explainer.py

"""
Patch Explanation Generator using LLMs.

Generates human-readable explanations of patches including:
- What the patch changes
- Why it fixes the bug
- Potential side effects
- Confidence assessment
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from pathlib import Path

# LLM provider imports
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

class ExplanationStyle(Enum):
    """Style of explanation to generate."""
    TECHNICAL = "technical"      # Detailed, for developers
    CONCISE = "concise"          # Brief summary
    EXECUTIVE = "executive"      # High-level, for managers
    VERBOSE = "verbose"          # Comprehensive with examples
    STRUCTURED = "structured"    # JSON/markdown formatted

@dataclass
class PatchExplanation:
    """Structured explanation of a patch."""
    patch_id: str
    summary: str                           # One-line summary
    what_changed: str                      # Description of code change
    why_it_fixes: str                      # Root cause analysis
    mechanism: str                         # How the fix works
    side_effects: List[str]                # Potential risks
    testing_suggestions: List[str]         # Recommended tests
    confidence_assessment: str             # Confidence explanation
    alternatives_considered: List[str]     # Other options evaluated
    
    # Metadata
    explanation_style: ExplanationStyle = ExplanationStyle.TECHNICAL
    generation_time_ms: float = 0.0
    tokens_used: int = 0
    
    def to_markdown(self) -> str:
        """Convert to markdown format."""
        md = f"""# Patch Explanation: {self.patch_id}

## Summary
{self.summary}

## What Changed
{self.what_changed}

## Why This Fixes the Bug
{self.why_it_fixes}

## How It Works
{self.mechanism}

## Potential Side Effects
"""
        for effect in self.side_effects:
            md += f"- {effect}\n"
        
        md += "\n## Recommended Tests\n"
        for test in self.testing_suggestions:
            md += f"- [ ] {test}\n"
        
        md += f"\n## Confidence Assessment\n{self.confidence_assessment}\n"
        
        if self.alternatives_considered:
            md += "\n## Alternatives Considered\n"
            for alt in self.alternatives_considered:
                md += f"- {alt}\n"
        
        return md
    
    def to_json(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "patch_id": self.patch_id,
            "summary": self.summary,
            "what_changed": self.what_changed,
            "why_it_fixes": self.why_it_fixes,
            "mechanism": self.mechanism,
            "side_effects": self.side_effects,
            "testing_suggestions": self.testing_suggestions,
            "confidence_assessment": self.confidence_assessment,
            "alternatives_considered": self.alternatives_considered,
            "style": self.explanation_style.value,
            "metadata": {
                "generation_time_ms": self.generation_time_ms,
                "tokens_used": self.tokens_used
            }
        }

@dataclass
class ExplanationContext:
    """Context for generating explanations."""
    error_type: str = ""
    error_message: str = ""
    stack_trace: List[str] = field(default_factory=list)
    code_context: str = ""           # Surrounding code
    project_context: str = ""        # Project/domain info
    similar_patches: List[str] = field(default_factory=list)  # Historical similar fixes
    test_results: Optional[Dict] = None  # Pass/fail info
    reviewer_persona: str = "developer"  # Target audience


# =============================================================================
# LLM PROVIDERS
# =============================================================================

class LLMProvider(ABC):
    """Abstract base for LLM providers."""
    
    def __init__(self, model: str, temperature: float = 0.3, max_tokens: int = 2000):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, int]:
        """
        Generate text from prompt.
        
        Returns: (generated_text, tokens_used)
        """
        pass
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        # Very rough: ~4 chars per token for English
        return len(text) // 4


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider."""
    
    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model, **kwargs)
        
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed")
        
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=api_base
        )
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, int]:
        """Generate using OpenAI API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            text = response.choices[0].message.content
            tokens = response.usage.total_tokens if response.usage else self.count_tokens(prompt + text)
            
            return text, tokens
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""
    
    def __init__(
        self,
        model: str = "claude-3-opus-20240229",
        api_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model, **kwargs)
        
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic package not installed")
        
        self.client = anthropic.Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
        )
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, int]:
        """Generate using Anthropic API."""
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}]
            )
            
            text = message.content[0].text
            tokens = message.usage.input_tokens + message.usage.output_tokens if message.usage else self.count_tokens(prompt + text)
            
            return text, tokens
            
        except Exception as e:
            self.logger.error(f"Anthropic API error: {e}")
            raise


# =============================================================================
# PROMPT BUILDERS
# =============================================================================

class PromptBuilder:
    """Builds prompts for different explanation styles."""
    
    def __init__(self):
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load prompt templates."""
        return {
            "system_technical": """You are an expert software engineer and code reviewer. 
Your task is to explain code patches clearly and accurately. Focus on:
1. Precise technical details of the change
2. Root cause analysis of the bug
3. Mechanism of the fix
4. Potential edge cases and risks
Be thorough but concise. Use code snippets where helpful.""",
            
            "system_concise": """You are a senior engineer explaining a bug fix to a busy colleague.
Provide a brief, clear explanation in 2-3 sentences focusing on what changed and why.""",
            
            "system_executive": """You are explaining a software fix to non-technical stakeholders.
Focus on business impact, risk level, and high-level what was fixed. Avoid jargon.""",
            
            "explanation_structure": """
Analyze this patch and provide a structured explanation.

PATCH INFORMATION:
- Template: {template_id}
- Location: {location}
- Confidence Score: {confidence:.2f}

ERROR CONTEXT:
- Type: {error_type}
- Message: {error_message}
- Stack Trace: {stack_trace}

CODE CHANGE:
Original:
```python
{original_code}

  Patched:
```python
  {patched_code}

SURROUNDING CONTEXT:
```python
