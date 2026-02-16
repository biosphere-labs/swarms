"""
ResultAggregator Module

This module provides output merging and contradiction detection for PARL orchestration.
It takes structured results from multiple sub-agents and synthesizes them into a coherent
final answer, while explicitly flagging contradictions and identifying coverage gaps.

Key features:
- LLM-based synthesis of sub-agent outputs
- Contradiction detection via LLM comparison
- Gap identification against original task requirements
- Configurable synthesis model
- Contradictions are NEVER silently resolved (always surfaced explicitly)
"""

from typing import Callable, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from swarms.structs.llm_backend import LLMBackend, LiteLLMBackend
from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger(log_folder="result_aggregator")


class StructuredResult(BaseModel):
    """
    Structured output from a single sub-agent.

    This model should match ContextShardingManager's output format.
    Once Task 002 (ContextShardingManager) is complete, this can be imported
    from swarms.structs.context_sharding instead.

    Attributes:
        agent_id: Unique identifier for the sub-agent
        sub_task_id: Identifier for the specific sub-task
        output: The actual output text from the sub-agent
        confidence: Agent's confidence in the result (0.0-1.0)
        metadata: Additional context (token count, execution time, etc.)
    """
    agent_id: str
    sub_task_id: str
    output: str
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AggregatedOutput(BaseModel):
    """
    Final synthesized output from multiple sub-agent results.

    Attributes:
        synthesized_answer: Coherent final answer merging all sub-agent outputs
        contradictions: List of detected contradictions between sub-agents
        gaps: List of identified gaps in task coverage
        confidence: Overall confidence in the synthesized answer (0.0-1.0)
        metadata: Additional information (models used, token counts, etc.)
    """
    synthesized_answer: str
    contradictions: List[str] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


SYNTHESIS_PROMPT = """You are synthesizing outputs from multiple sub-agents working on parallel sub-tasks.

Original Task:
{original_task}

Sub-Agent Outputs:
{formatted_results}

Instructions:
1. Merge the sub-agent outputs into a single, coherent answer to the original task
2. Preserve all factual information from the sub-agents
3. Organize information logically
4. Do NOT resolve contradictions - if sub-agents disagree, note this in your synthesis
5. Maintain the overall confidence level based on sub-agent confidence scores

Provide a clear, well-structured synthesis that addresses the original task."""


CONTRADICTION_DETECTION_PROMPT = """You are analyzing outputs from multiple sub-agents for contradictions.

Sub-Agent Outputs:
{formatted_results}

Instructions:
1. Compare the outputs for conflicting claims or information
2. Look for direct contradictions (e.g., "Company founded in 2010" vs "Company founded in 2015")
3. Look for implicit contradictions (e.g., mutually exclusive statements)
4. For each contradiction found, clearly state:
   - What the contradiction is
   - Which sub-agents are involved (use their IDs)
   - The specific conflicting claims

If NO contradictions are found, respond with: "NO_CONTRADICTIONS"

Format contradictions as a JSON list of strings, one per contradiction:
["Contradiction 1: Agent A claims X, but Agent B claims Y", "Contradiction 2: ..."]"""


GAP_DETECTION_PROMPT = """You are checking if sub-agent outputs fully cover the original task requirements.

Original Task:
{original_task}

Sub-Tasks Addressed:
{sub_task_list}

Sub-Agent Outputs:
{formatted_results}

Instructions:
1. Break down the original task into its key requirements/components
2. Check which requirements are addressed by the sub-agent outputs
3. Identify any gaps - requirements not covered or only partially covered
4. For each gap, clearly state what is missing

If NO gaps are found, respond with: "NO_GAPS"

Format gaps as a JSON list of strings, one per gap:
["Gap 1: Original task asked for X but no sub-agent addressed it", "Gap 2: ..."]"""


class ResultAggregator:
    """
    Aggregates and synthesizes results from multiple sub-agents.

    This class handles:
    - Merging sub-agent outputs into a coherent final answer
    - Detecting contradictions between sub-agents
    - Identifying gaps in task coverage
    - Computing overall confidence

    Contradictions are ALWAYS explicitly surfaced in the output,
    never silently resolved.
    """

    def __init__(
        self,
        synthesis_model: str = "gpt-4o-mini",
        max_tokens: int = 4000,
        temperature: float = 0.3,
        api_key_provider: Optional[Callable] = None,
        llm_backend: Optional[LLMBackend] = None,
    ):
        """
        Initialize the ResultAggregator.

        Args:
            synthesis_model: LLM model to use for synthesis (default: gpt-4o-mini for cost efficiency)
            max_tokens: Maximum tokens for LLM responses
            temperature: Temperature for LLM calls (lower = more deterministic)
            api_key_provider: Optional callable that returns the next API key (for round-robin)
            llm_backend: Optional LLM backend to use. Defaults to LiteLLMBackend(synthesis_model).
        """
        self.synthesis_model = synthesis_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._api_key_provider = api_key_provider
        self._llm_backend = llm_backend or LiteLLMBackend(model=synthesis_model)

        logger.info(
            f"Initialized ResultAggregator with model={synthesis_model}, "
            f"max_tokens={max_tokens}, temperature={temperature}, "
            f"backend={type(self._llm_backend).__name__}"
        )

    def _call_llm(self, prompt: str) -> str:
        """Call LLM with optional round-robin API key."""
        api_key = None
        if self._api_key_provider:
            api_key = self._api_key_provider()
        return self._llm_backend.call(
            system_prompt="You are an expert analyst.",
            user_prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=api_key,
        )

    def aggregate(
        self,
        results: List[StructuredResult],
        original_task: str,
    ) -> AggregatedOutput:
        """
        Aggregate multiple sub-agent results into a final synthesized output.

        Args:
            results: List of structured results from sub-agents
            original_task: The original task that was decomposed

        Returns:
            AggregatedOutput with synthesized answer, contradictions, and gaps
        """
        if not results:
            logger.warning("No results provided for aggregation")
            return AggregatedOutput(
                synthesized_answer="No sub-agent results available",
                contradictions=[],
                gaps=["No sub-agents executed"],
                confidence=0.0,
                metadata={"error": "empty_results"}
            )

        logger.info(f"Aggregating {len(results)} sub-agent results")

        # Format results for LLM prompts
        formatted_results = self._format_results(results)

        # Step 1: Detect contradictions
        contradictions = self._detect_contradictions(formatted_results)

        # Step 2: Identify gaps
        gaps = self._identify_gaps(original_task, results, formatted_results)

        # Step 3: Synthesize final answer
        synthesized_answer = self._synthesize_answer(
            original_task,
            formatted_results,
            contradictions
        )

        # Step 4: Compute overall confidence
        confidence = self._compute_confidence(results, contradictions, gaps)

        # Build metadata
        metadata = {
            "num_sub_agents": len(results),
            "synthesis_model": self.synthesis_model,
            "sub_agent_ids": [r.agent_id for r in results],
            "avg_sub_agent_confidence": sum(r.confidence for r in results) / len(results),
        }

        output = AggregatedOutput(
            synthesized_answer=synthesized_answer,
            contradictions=contradictions,
            gaps=gaps,
            confidence=confidence,
            metadata=metadata
        )

        logger.info(
            f"Aggregation complete: {len(contradictions)} contradictions, "
            f"{len(gaps)} gaps, confidence={confidence:.2f}"
        )

        return output

    def _format_results(self, results: List[StructuredResult]) -> str:
        """Format results for inclusion in prompts."""
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(
                f"[Sub-Agent {i}] (ID: {result.agent_id}, Task: {result.sub_task_id}, "
                f"Confidence: {result.confidence:.2f})\n{result.output}\n"
            )
        return "\n".join(formatted)

    def _detect_contradictions(self, formatted_results: str) -> List[str]:
        """
        Use LLM to detect contradictions between sub-agent outputs.

        Args:
            formatted_results: Formatted string of all sub-agent outputs

        Returns:
            List of contradiction descriptions (empty if none found)
        """
        logger.info("Detecting contradictions...")

        prompt = CONTRADICTION_DETECTION_PROMPT.format(
            formatted_results=formatted_results
        )

        try:
            content = self._call_llm(prompt)

            # Check for no contradictions
            if "NO_CONTRADICTIONS" in content:
                logger.info("No contradictions detected")
                return []

            # Strip markdown code fences if present
            import json
            if content.startswith("```"):
                lines = content.split("\n")
                lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines)

            # Try to parse as JSON list
            try:
                contradictions = json.loads(content)
                if isinstance(contradictions, list):
                    logger.info(f"Found {len(contradictions)} contradictions")
                    return contradictions
            except json.JSONDecodeError:
                # Fallback: split by newlines if JSON parsing fails
                contradictions = [
                    line.strip()
                    for line in content.split('\n')
                    if line.strip() and not line.strip().startswith('[')
                ]
                logger.warning(f"JSON parse failed, extracted {len(contradictions)} contradictions from text")
                return contradictions

        except Exception as e:
            logger.error(f"Error detecting contradictions: {e}")
            return [f"Error during contradiction detection: {str(e)}"]

    def _identify_gaps(
        self,
        original_task: str,
        results: List[StructuredResult],
        formatted_results: str,
    ) -> List[str]:
        """
        Use LLM to identify gaps in task coverage.

        Args:
            original_task: The original task description
            results: List of StructuredResults
            formatted_results: Formatted string of all sub-agent outputs

        Returns:
            List of gap descriptions (empty if no gaps found)
        """
        logger.info("Identifying coverage gaps...")

        sub_task_list = "\n".join([
            f"- {r.sub_task_id}" for r in results
        ])

        prompt = GAP_DETECTION_PROMPT.format(
            original_task=original_task,
            sub_task_list=sub_task_list,
            formatted_results=formatted_results,
        )

        try:
            content = self._call_llm(prompt)

            # Check for no gaps
            if "NO_GAPS" in content:
                logger.info("No coverage gaps detected")
                return []

            # Strip markdown code fences if present
            import json
            if content.startswith("```"):
                lines = content.split("\n")
                lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines)

            # Try to parse as JSON list
            try:
                gaps = json.loads(content)
                if isinstance(gaps, list):
                    logger.info(f"Found {len(gaps)} coverage gaps")
                    return gaps
            except json.JSONDecodeError:
                # Fallback: split by newlines
                gaps = [
                    line.strip()
                    for line in content.split('\n')
                    if line.strip() and not line.strip().startswith('[')
                ]
                logger.warning(f"JSON parse failed, extracted {len(gaps)} gaps from text")
                return gaps

        except Exception as e:
            logger.error(f"Error identifying gaps: {e}")
            return [f"Error during gap detection: {str(e)}"]

    def _synthesize_answer(
        self,
        original_task: str,
        formatted_results: str,
        contradictions: List[str],
    ) -> str:
        """
        Use LLM to synthesize a coherent final answer.

        Args:
            original_task: The original task description
            formatted_results: Formatted string of all sub-agent outputs
            contradictions: List of detected contradictions

        Returns:
            Synthesized answer as a string
        """
        logger.info("Synthesizing final answer...")

        prompt = SYNTHESIS_PROMPT.format(
            original_task=original_task,
            formatted_results=formatted_results,
        )

        # If there are contradictions, append them to the prompt
        if contradictions:
            prompt += f"\n\nNOTE: The following contradictions were detected:\n"
            for contradiction in contradictions:
                prompt += f"- {contradiction}\n"
            prompt += "\nInclude these contradictions explicitly in your synthesis."

        try:
            synthesized = self._call_llm(prompt)
            logger.info(f"Synthesis complete ({len(synthesized)} chars)")
            return synthesized

        except Exception as e:
            logger.error(f"Error synthesizing answer: {e}")
            return f"Error during synthesis: {str(e)}\n\nRaw sub-agent outputs:\n{formatted_results}"

    def _compute_confidence(
        self,
        results: List[StructuredResult],
        contradictions: List[str],
        gaps: List[str],
    ) -> float:
        """
        Compute overall confidence in the aggregated output.

        Confidence is based on:
        - Average sub-agent confidence
        - Presence of contradictions (reduces confidence)
        - Presence of gaps (reduces confidence)

        Args:
            results: List of StructuredResults
            contradictions: List of detected contradictions
            gaps: List of identified gaps

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not results:
            return 0.0

        # Start with average sub-agent confidence
        avg_confidence = sum(r.confidence for r in results) / len(results)

        # Reduce confidence for contradictions (10% per contradiction, max 50% reduction)
        contradiction_penalty = min(0.5, len(contradictions) * 0.1)

        # Reduce confidence for gaps (5% per gap, max 30% reduction)
        gap_penalty = min(0.3, len(gaps) * 0.05)

        # Compute final confidence
        final_confidence = avg_confidence * (1 - contradiction_penalty) * (1 - gap_penalty)

        logger.info(
            f"Confidence calculation: base={avg_confidence:.2f}, "
            f"contradiction_penalty={contradiction_penalty:.2f}, "
            f"gap_penalty={gap_penalty:.2f}, final={final_confidence:.2f}"
        )

        return max(0.0, min(1.0, final_confidence))
