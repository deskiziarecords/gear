# packages/core-algorithms/pfsu/fix_understanding.py

import openai
from typing import List, Dict, Tuple
import ast
import difflib

class FixUnderstandingEngine:
    """
    LLM-based patch understanding and validation.
    
    Provides:
    1. Natural language explanation of patches
    2. Correctness prediction beyond test pass/fail
    3. Semantic diff analysis
    4. Patch improvement suggestions
    """
    
    def __init__(self, model: str = "gpt-4"):
        self.model = model
        self.client = openai.OpenAI()
    
    def explain_patch(
        self,
        candidate: PatchCandidate,
        error_context: 'ErrorReport',
        style: str = "technical"
    ) -> str:
        """
        Generate human-readable explanation of what the patch does.
        """
        prompt = f"""
        Explain the following code patch in {style} style:

        ERROR CONTEXT:
        - Type: {error_context.error_type}
        - Message: {error_context.error_message}
        - Location: {candidate.location}

        ORIGINAL CODE:
        ```python
        {candidate.original_code}
        ```

        PATCHED CODE:
        ```python
        {candidate.patched_code}
        ```

        SYNTHESIS TRACE:
        {candidate.synthesis_trace}

        Provide:
        1. What the patch changes
        2. Why this change addresses the error
        3. Potential side effects or risks
        4. Confidence assessment
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return response.choices[0].message.content
    
    def validate_semantic_correctness(
        self,
        candidate: PatchCandidate,
        codebase_context: str
    ) -> Tuple[bool, float, str]:
        """
        Predict semantic correctness beyond test suite validation.
        
        Returns: (is_likely_correct, confidence_score, reasoning)
        """
        prompt = f"""
        Assess the semantic correctness of this patch:

        PATCH:
        ```python
        {candidate.patched_code}
        ```

        CONTEXT:
        {codebase_context}

        Evaluate:
        1. Does the patch preserve intended behavior?
        2. Are there edge cases not handled?
        3. Is this a minimal and focused fix?
        4. Could this introduce regressions?

        Respond with JSON:
        {{
            "likely_correct": bool,
            "confidence": float (0-1),
            "reasoning": str,
            "risks": [str]
        }}
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return (
            result["likely_correct"],
            result["confidence"],
            result["reasoning"]
        )
    
    def suggest_improvements(
        self,
        candidate: PatchCandidate,
        review_feedback: str
    ) -> List[str]:
        """
        Suggest improvements to patch based on review feedback.
        """
        prompt = f"""
        Review this patch and suggest improvements:

        CURRENT PATCH:
        ```python
        {candidate.patched_code}
        ```

        REVIEW FEEDBACK:
        {review_feedback}

        Suggest specific code improvements that address the feedback
        while maintaining the fix. Return as list of code snippets.
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        
        # Parse suggestions from response
        suggestions = self._parse_suggestions(response.choices[0].message.content)
        return suggestions
    
    def compare_patches(
        self,
        candidates: List[PatchCandidate]
    ) -> str:
        """
        Generate comparative analysis of multiple patch candidates.
        """
        comparison = "PATCH COMPARISON:\n\n"
        
        for i, c in enumerate(candidates, 1):
            comparison += f"Option {i} ({c.modification_type}):\n"
            comparison += f"  Score: {c.probability_score:.3f}\n"
            comparison += f"  Code: {c.patched_code[:100]}...\n\n"
        
        prompt = f"""
        Compare these patch candidates and recommend the best one:

        {comparison}

        Consider:
        1. Likelihood of correctness
        2. Minimal invasiveness
        3. Maintainability
        4. Risk of side effects

        Provide recommendation with justification.
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return response.choices[0].message.content
    
    def _parse_suggestions(self, text: str) -> List[str]:
        """Parse code suggestions from LLM response"""
        # Extract code blocks from markdown
        import re
        code_blocks = re.findall(r'```python\n(.*?)\n```', text, re.DOTALL)
        return code_blocks if code_blocks else [text]
