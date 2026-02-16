"""
PARL Orchestrator MCP Server

Exposes the PARL orchestrator as an MCP tool for use with Cursor, Windsurf,
Claude Desktop, and other AI coding tools that support MCP.

Usage:
    python -m swarms.structs.parl_mcp_server

    Or with custom config:
    PARL_ORCHESTRATOR_MODEL=deepinfra/Qwen/Qwen3-235B-A22B \
    PARL_SUB_AGENT_MODEL=deepinfra/Qwen/Qwen2.5-72B-Instruct \
    PARL_FACT_CHECK=true \
    python -m swarms.structs.parl_mcp_server

Environment Variables:
    PARL_ORCHESTRATOR_MODEL — Model for decomposition (default: gpt-4o-mini)
    PARL_SUB_AGENT_MODEL — Model for sub-agents (default: gpt-4o-mini)
    PARL_SYNTHESIS_MODEL — Model for synthesis (default: same as orchestrator)
    PARL_DECOMPOSITION_BACKEND — Backend for decomposition: "litellm" or "claude-code"
    PARL_SYNTHESIS_BACKEND — Backend for synthesis: "litellm" or "claude-code"
    PARL_FACT_CHECK — Enable fact-checking: "true" or "false"
    PARL_FACT_CHECK_MODEL — Model for fact-checkers (default: same as sub-agents)
    PARL_MAX_PARALLEL — Max concurrent sub-agents (default: 10)
    PARL_MAX_ITERATIONS — Gap-fill iterations (default: 2)
    PARL_TIMEOUT — Overall timeout in seconds (default: 300)
    PARL_SUB_AGENT_TIMEOUT — Per sub-agent timeout (default: 120)
    PARL_TOKEN_BUDGET — Total token budget (default: 100000)
    PARL_API_KEYS — Comma-separated API keys for round-robin rotation
    MCP_PORT — Port for the MCP server (default: 8765)

MCP Client Configuration (add to MCP client config):
    {
        "mcpServers": {
            "parl-orchestrator": {
                "url": "http://localhost:8765",
                "transport": "streamable-http"
            }
        }
    }
"""

import asyncio
import itertools
import json
import os
import time
from typing import List, Optional

from mcp.server.fastmcp import Context, FastMCP
from swarms.structs.agent import Agent
from swarms.structs.llm_backend import LiteLLMBackend
from swarms.structs.parl_orchestrator import PARLOrchestrator
from swarms.structs.result_aggregator import ResultAggregator, StructuredResult
from swarms.tools.serper_search import serper_search
from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger(log_folder="parl_mcp_server")

# MCP server instance
mcp = FastMCP("PARLOrchestrator")


@mcp.tool(
    name="parl_execute",
    description=(
        "Execute a complex multi-faceted research or analysis task using "
        "the PARL orchestrator. Automatically decomposes the task into "
        "parallelizable sub-tasks, executes them concurrently with fact-checking, "
        "and synthesizes a comprehensive answer with contradiction detection. "
        "Best for tasks with multiple independent aspects (e.g., 'Research company X "
        "across funding, team, pricing, and reviews')."
    ),
)
async def parl_execute(
    task: str,
    orchestrator_model: Optional[str] = None,
    sub_agent_model: Optional[str] = None,
    synthesis_model: Optional[str] = None,
    fact_check: Optional[bool] = None,
    max_parallel: Optional[int] = None,
    max_iterations: Optional[int] = None,
    timeout: Optional[int] = None,
    ctx: Context = None,
) -> str:
    """
    Execute a task using the PARL orchestrator.

    Args:
        task: The task to execute. Should be a complex, multi-faceted question
            or research request that benefits from parallel decomposition.
        orchestrator_model: Optional model override for decomposition.
            Defaults to PARL_ORCHESTRATOR_MODEL env var or gpt-4o-mini.
        sub_agent_model: Optional model override for sub-agents.
            Defaults to PARL_SUB_AGENT_MODEL env var or gpt-4o-mini.
        synthesis_model: Optional model override for synthesis.
            Defaults to PARL_SYNTHESIS_MODEL env var or orchestrator_model.
        fact_check: Optional override to enable/disable fact-checking.
            Defaults to PARL_FACT_CHECK env var or False.
        max_parallel: Optional override for max concurrent sub-agents.
            Defaults to PARL_MAX_PARALLEL env var or 10.
        max_iterations: Optional override for gap-fill iterations.
            Defaults to PARL_MAX_ITERATIONS env var or 2.
        timeout: Optional override for overall timeout in seconds.
            Defaults to PARL_TIMEOUT env var or 300.
        ctx: MCP Context for progress reporting.

    Returns:
        str: The synthesized answer from the orchestrator, including any
            detected contradictions and gaps.

    Example:
        >>> result = parl_execute(
        ...     task="Research competitor Acme Corp across funding, team, pricing, and reviews",
        ...     fact_check=True
        ... )
    """
    if ctx:
        await ctx.info("Initializing PARL orchestrator...")
        await ctx.report_progress(0, 3)

    # Create the orchestrator with optional overrides
    # Environment variables are automatically handled by PARLOrchestrator.__init__
    orchestrator = PARLOrchestrator(
        orchestrator_model=orchestrator_model,
        sub_agent_model=sub_agent_model,
        synthesis_model=synthesis_model,
        fact_check=fact_check,
        max_parallel=max_parallel,
        max_iterations=max_iterations,
        timeout=timeout,
        tools=[serper_search],  # Provide web search capability
    )

    if ctx:
        await ctx.info("Decomposing task and executing sub-agents...")
        await ctx.report_progress(1, 3)

    # Execute the task (blocking — run in thread to keep event loop alive)
    result = await asyncio.to_thread(orchestrator.run, task=task)

    if ctx:
        await ctx.info("Task complete.")
        await ctx.report_progress(3, 3)

    return result


@mcp.tool(
    name="parl_config",
    description=(
        "Get the current PARL orchestrator configuration from environment variables. "
        "Use this to check what models, backends, and settings are active."
    ),
)
def parl_config() -> str:
    """
    Return the current PARL configuration from environment variables.

    Returns:
        str: A formatted string showing all PARL environment variables.
    """
    config = {
        "PARL_ORCHESTRATOR_MODEL": os.environ.get("PARL_ORCHESTRATOR_MODEL", "gpt-4o-mini (default)"),
        "PARL_SUB_AGENT_MODEL": os.environ.get("PARL_SUB_AGENT_MODEL", "gpt-4o-mini (default)"),
        "PARL_SYNTHESIS_MODEL": os.environ.get("PARL_SYNTHESIS_MODEL", "(same as orchestrator)"),
        "PARL_DECOMPOSITION_BACKEND": os.environ.get("PARL_DECOMPOSITION_BACKEND", "litellm (default)"),
        "PARL_SYNTHESIS_BACKEND": os.environ.get("PARL_SYNTHESIS_BACKEND", "litellm (default)"),
        "PARL_FACT_CHECK": os.environ.get("PARL_FACT_CHECK", "false (default)"),
        "PARL_FACT_CHECK_MODEL": os.environ.get("PARL_FACT_CHECK_MODEL", "(same as sub-agent)"),
        "PARL_MAX_PARALLEL": os.environ.get("PARL_MAX_PARALLEL", "10 (default)"),
        "PARL_MAX_ITERATIONS": os.environ.get("PARL_MAX_ITERATIONS", "2 (default)"),
        "PARL_TIMEOUT": os.environ.get("PARL_TIMEOUT", "300s (default)"),
        "PARL_SUB_AGENT_TIMEOUT": os.environ.get("PARL_SUB_AGENT_TIMEOUT", "120s (default)"),
        "PARL_TOKEN_BUDGET": os.environ.get("PARL_TOKEN_BUDGET", "100000 (default)"),
        "PARL_API_KEYS": "***" if os.environ.get("PARL_API_KEYS") else "(not set)",
    }

    return "\n".join(f"{key}: {value}" for key, value in config.items())


# --- API key rotation helper ---

def _build_key_cycle():
    """Build an itertools.cycle from PARL_API_KEYS if available."""
    raw = os.environ.get("PARL_API_KEYS", "")
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if keys:
        return itertools.cycle(keys)
    return None

_key_cycle_lock = __import__("threading").Lock()


def _next_api_key(cycle):
    """Thread-safe next key from a cycle, or None."""
    if cycle is None:
        return None
    with _key_cycle_lock:
        return next(cycle)


# --- Review tool ---

REVIEW_SYNTHESIS_PROMPT = """You are synthesizing reviews from a panel of expert reviewers.

Document under review:
{document_summary}

Reviewer assessments:
{formatted_reviews}

Instructions:
1. For each reviewer, summarize their key findings, concerns, and praise
2. Identify where reviewers AGREE (consensus strengths or weaknesses)
3. Identify where reviewers DISAGREE and present both sides
4. Highlight the most critical issues raised across all reviews
5. Provide an overall assessment that weighs all perspectives
6. Do NOT resolve disagreements — present them transparently

Structure your output with clear sections per topic, not per reviewer."""


@mcp.tool(
    name="parl_review",
    description=(
        "Review a document from multiple expert perspectives in parallel. "
        "Each persona runs as an independent reviewer agent with web search access, "
        "then results are synthesized into a comprehensive multi-angle assessment. "
        "Pass personas as a JSON array of objects with 'name' and 'instruction' fields. "
        "Each persona can optionally specify a 'model' to use a different LLM, which reduces "
        "shared blind spots across reviewers. "
        "Example personas: "
        '[{"name": "Market Analyst", "instruction": "Evaluate market sizing", "model": "deepinfra/Qwen/Qwen3-235B-A22B"}, '
        '{"name": "Financial Reviewer", "instruction": "Scrutinize projections", "model": "deepinfra/deepseek-ai/DeepSeek-V3.2"}]'
    ),
)
async def parl_review(
    document: str,
    personas: str,
    synthesis_prompt: Optional[str] = None,
    fact_check: Optional[bool] = None,
    timeout: Optional[int] = None,
    ctx: Context = None,
) -> str:
    """
    Review a document from multiple expert perspectives in parallel.

    Args:
        document: The full document text to review (or a URL/summary).
        personas: JSON array of reviewer personas. Each object must have:
            - "name": Short name for the reviewer (e.g. "Market Analyst")
            - "instruction": What this reviewer should focus on
            Optional:
            - "model": LLM model for this reviewer (default: PARL_SUB_AGENT_MODEL).
                       Use different models per persona to reduce shared blind spots.
            - "tools": If true, this reviewer gets web search access (default: true)
        synthesis_prompt: Optional custom instructions for the synthesis step.
        fact_check: Enable fact-check debate per reviewer (slower but more accurate).
        timeout: Overall timeout in seconds (default: 300).
        ctx: MCP Context for progress reporting.

    Returns:
        str: Synthesized multi-perspective review with agreements, disagreements,
            and critical findings highlighted.
    """
    start_time = time.time()
    effective_timeout = timeout or int(os.environ.get("PARL_TIMEOUT", "300"))

    # Parse personas
    try:
        persona_list = json.loads(personas)
        if not isinstance(persona_list, list) or len(persona_list) == 0:
            return "Error: 'personas' must be a non-empty JSON array."
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON in 'personas': {e}"

    for i, p in enumerate(persona_list):
        if not isinstance(p, dict) or "name" not in p or "instruction" not in p:
            return f"Error: Persona {i} must have 'name' and 'instruction' fields."

    # Config from environment
    sub_agent_model = os.environ.get("PARL_SUB_AGENT_MODEL", "gpt-4o-mini")
    synthesis_model = os.environ.get(
        "PARL_SYNTHESIS_MODEL",
        os.environ.get("PARL_ORCHESTRATOR_MODEL", "gpt-4o-mini"),
    )
    max_parallel = int(os.environ.get("PARL_MAX_PARALLEL", "10"))
    use_fact_check = fact_check if fact_check is not None else (
        os.environ.get("PARL_FACT_CHECK", "false").lower() == "true"
    )

    key_cycle = _build_key_cycle()

    # Log model distribution across personas
    models_used = set()
    for p in persona_list:
        models_used.add(p.get("model", sub_agent_model))
    logger.info(
        f"parl_review: {len(persona_list)} reviewers across {len(models_used)} model(s), "
        f"fact_check={use_fact_check}, models={models_used}"
    )

    if ctx:
        await ctx.info(f"Starting review with {len(persona_list)} reviewers across {len(models_used)} model(s)...")
        await ctx.report_progress(0, len(persona_list) + 1)  # +1 for synthesis

    # --- Execute reviewers in parallel ---
    def run_reviewer(persona: dict) -> StructuredResult:
        name = persona["name"]
        instruction = persona["instruction"]
        use_tools = persona.get("tools", True)
        reviewer_model = persona.get("model", sub_agent_model)

        system_prompt = (
            f"You are {name}, an expert reviewer.\n\n"
            f"Your review focus: {instruction}\n\n"
            "Review the document provided below. Be specific, cite evidence from the "
            "document, and do independent research where needed to verify claims. "
            "Structure your review with: Key Findings, Concerns, Strengths, and Verdict."
        )

        tools = [serper_search] if use_tools else None
        api_key = _next_api_key(key_cycle)

        agent = Agent(
            agent_name=f"reviewer-{name.lower().replace(' ', '-')}",
            system_prompt=system_prompt,
            model_name=reviewer_model,
            max_loops=3 if tools else 1,
            max_tokens=8192,
            output_type="final",
            verbose=False,
            tools=tools,
            llm_api_key=api_key,
        )

        task_prompt = f"Review the following document:\n\n{document}"
        raw_output = agent.run(task=task_prompt)
        output_str = str(raw_output) if raw_output is not None else ""

        # Optional fact-check debate
        if use_fact_check and output_str:
            try:
                from swarms.structs.fact_check_debate import FactCheckDebate

                debate = FactCheckDebate(
                    model_name=reviewer_model,
                    max_loops=2,
                    tools=tools,
                    api_key_provider=lambda: _next_api_key(key_cycle) if key_cycle else None,
                    verbose=False,
                )
                verified = debate.verify(research_output=output_str)
                if verified:
                    output_str = verified
            except Exception as e:
                logger.warning(f"Fact-check failed for {name}: {e}")

        return StructuredResult(
            agent_id=f"reviewer-{name.lower().replace(' ', '-')}",
            sub_task_id=name,
            output=output_str,
            confidence=0.8,
            metadata={
                "persona": name,
                "instruction": instruction,
                "model": reviewer_model,
            },
        )

    # Run reviewers concurrently with asyncio for progress reporting
    results: List[StructuredResult] = []
    total_reviewers = len(persona_list)

    async def run_reviewer_async(persona: dict) -> StructuredResult:
        return await asyncio.to_thread(run_reviewer, persona)

    # Create tasks with semaphore for max_parallel limit
    sem = asyncio.Semaphore(min(total_reviewers, max_parallel))

    async def run_with_limit(persona: dict) -> StructuredResult:
        async with sem:
            return await run_reviewer_async(persona)

    tasks = [run_with_limit(p) for p in persona_list]
    completed = 0

    for coro in asyncio.as_completed(tasks):
        try:
            result = await asyncio.wait_for(coro, timeout=effective_timeout)
            results.append(result)
            completed += 1
            logger.info(f"Reviewer '{result.sub_task_id}' completed ({len(result.output)} chars)")
            if ctx:
                await ctx.report_progress(completed, total_reviewers + 1)
                await ctx.info(f"Reviewer {completed}/{total_reviewers} done: {result.sub_task_id}")
        except Exception as e:
            completed += 1
            logger.error(f"Reviewer failed: {e}")
            results.append(StructuredResult(
                agent_id="reviewer-unknown",
                sub_task_id="unknown",
                output=f"[Review failed: {e}]",
                confidence=0.0,
                metadata={"error": str(e)},
            ))
            if ctx:
                await ctx.report_progress(completed, total_reviewers + 1)

    if not results:
        return "Error: All reviewers failed. Check API keys and model configuration."

    # --- Synthesize all reviews ---
    if ctx:
        await ctx.info(f"Synthesizing {len(results)} reviews...")

    aggregator = ResultAggregator(
        model=synthesis_model,
        api_key_provider=lambda: _next_api_key(key_cycle) if key_cycle else None,
    )

    # Use custom synthesis prompt if provided
    original_task = synthesis_prompt or (
        f"Multi-perspective review of the following document by "
        f"{len(persona_list)} expert reviewers:\n\n"
        f"{document[:500]}{'...' if len(document) > 500 else ''}"
    )

    aggregated = await asyncio.to_thread(
        aggregator.aggregate,
        results=results,
        original_task=original_task,
    )

    if ctx:
        await ctx.report_progress(total_reviewers + 1, total_reviewers + 1)
        await ctx.info("Review complete.")

    # Format output
    elapsed = time.time() - start_time
    output_parts = [aggregated.synthesized_answer]

    if aggregated.contradictions:
        output_parts.append("\n\n## Reviewer Disagreements")
        for c in aggregated.contradictions:
            output_parts.append(f"- {c}")

    if aggregated.gaps:
        output_parts.append("\n\n## Coverage Gaps")
        for g in aggregated.gaps:
            output_parts.append(f"- {g}")

    output_parts.append(
        f"\n\n---\n*Review completed in {elapsed:.1f}s by {len(results)} reviewers "
        f"(model: {sub_agent_model}, synthesis: {synthesis_model})*"
    )

    return "\n".join(output_parts)


# --- Smart review tool (with auto model selection + cross-model fact-check) ---

SMART_REVIEW_SYNTHESIS_PROMPT = """You are synthesizing reviews from a panel of expert reviewers,
with special attention to cross-model blind spot analysis.

Document under review:
{document_summary}

Reviewer assessments:
{formatted_reviews}

Model assignments:
{model_map}

{fact_check_section}

Instructions:
1. For each reviewer, summarize their key findings, concerns, and praise
2. Identify where reviewers AGREE (consensus strengths or weaknesses)
3. Identify where reviewers DISAGREE and present both sides
4. BLIND SPOT ANALYSIS: Identify issues found by ONLY ONE model.
   These are the most valuable findings — things that model's architecture
   detected that others missed. Present these prominently.
5. Build a model agreement matrix showing which models agreed/disagreed
   on each major finding
6. Do NOT resolve disagreements — present them transparently

Structure your output with these sections:
- Consensus Findings
- Blind Spot Catches (single-model findings)
- Disagreements
- Model Agreement Matrix
- Overall Assessment"""


@mcp.tool(
    name="parl_smart_review",
    description=(
        "Enhanced multi-model document review with automatic model selection, "
        "cross-model fact-checking, and blind spot analysis. "
        "Pass personas as a JSON array of objects with 'name' and 'instruction' fields. "
        "Models are automatically assigned from the DeepInfra catalog for maximum "
        "diversity unless explicitly specified per persona. "
        "Includes iterative fact-checking where claims are verified by different models "
        "than the ones that made them. Returns blind spot analysis showing which issues "
        "were caught by only one model. "
        "Example personas: "
        '[{"name": "Thesis Stress-Tester", "instruction": "Find logical gaps"}, '
        '{"name": "Audience Calibrator", "instruction": "Check tone for CEOs"}]'
    ),
)
async def parl_smart_review(
    document: str,
    personas: str,
    auto_assign_models: Optional[bool] = True,
    fact_check_rounds: Optional[int] = 2,
    synthesis_prompt: Optional[str] = None,
    timeout: Optional[int] = None,
    ctx: Context = None,
) -> str:
    """
    Enhanced review with dynamic model selection and cross-model fact-checking.

    Args:
        document: The full document text to review.
        personas: JSON array of reviewer personas. Each object must have:
            - "name": Short name for the reviewer
            - "instruction": What this reviewer should focus on
            Optional:
            - "model": Explicit model override (skips auto-assignment for this persona)
        auto_assign_models: If True (default), automatically assign diverse models
            to personas without explicit model assignments using the DeepInfra catalog.
        fact_check_rounds: Number of cross-model fact-check rounds (0 to disable,
            default 2, max 3). Claims are verified by a different model than the
            one that made them.
        synthesis_prompt: Optional custom instructions for the synthesis step.
        timeout: Overall timeout in seconds (default: 300).

    Returns:
        str: Synthesized review with blind spot analysis, model agreement matrix,
            and fact-check corrections.
    """
    from swarms.structs.model_selector import fetch_available_models, assign_models_to_personas
    from swarms.structs.iterative_fact_check import IterativeFactCheck

    start_time = time.time()
    effective_timeout = timeout or int(os.environ.get("PARL_TIMEOUT", "300"))
    fc_rounds = max(0, min(fact_check_rounds or 2, 3))

    # Parse personas
    try:
        persona_list = json.loads(personas)
        if not isinstance(persona_list, list) or len(persona_list) == 0:
            return "Error: 'personas' must be a non-empty JSON array."
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON in 'personas': {e}"

    for i, p in enumerate(persona_list):
        if not isinstance(p, dict) or "name" not in p or "instruction" not in p:
            return f"Error: Persona {i} must have 'name' and 'instruction' fields."

    # Config from environment
    sub_agent_model = os.environ.get("PARL_SUB_AGENT_MODEL", "gpt-4o-mini")
    orchestrator_model = os.environ.get("PARL_ORCHESTRATOR_MODEL", "gpt-4o-mini")
    synthesis_model = os.environ.get(
        "PARL_SYNTHESIS_MODEL",
        orchestrator_model,
    )
    max_parallel = int(os.environ.get("PARL_MAX_PARALLEL", "10"))

    key_cycle = _build_key_cycle()

    # Total steps: model assignment + reviewers + fact-check + synthesis
    total_steps = len(persona_list) + 3  # rough estimate for progress bar

    if ctx:
        await ctx.info(f"Smart review: {len(persona_list)} reviewers, {fc_rounds} fact-check rounds")
        await ctx.report_progress(0, total_steps)

    # --- Step 1: Auto-assign models to personas ---
    if auto_assign_models:
        if ctx:
            await ctx.info("Fetching model catalog for auto-assignment...")
        api_key = _next_api_key(key_cycle)
        if api_key:
            logger.info("Smart review: fetching model catalog for auto-assignment...")
            available_models = await asyncio.to_thread(fetch_available_models, api_key)
            if available_models:
                persona_list = await asyncio.to_thread(
                    assign_models_to_personas,
                    personas=persona_list,
                    available_models=available_models,
                    orchestrator_model=orchestrator_model,
                    api_key=_next_api_key(key_cycle),
                )
            else:
                logger.warning("No models from catalog, using default sub_agent_model")
        else:
            logger.warning("No API keys for model catalog, using default sub_agent_model")

    # Build model map for reporting
    model_map = {}
    for p in persona_list:
        model = p.get("model", sub_agent_model)
        reason = p.get("_assignment_reason", "default")
        model_map[p["name"]] = {"model": model, "reason": reason}

    models_used = set(v["model"] for v in model_map.values())
    logger.info(
        f"parl_smart_review: {len(persona_list)} reviewers across "
        f"{len(models_used)} model(s): {models_used}"
    )

    # --- Step 2: Execute reviewers in parallel (same pattern as parl_review) ---
    def run_reviewer(persona: dict) -> StructuredResult:
        name = persona["name"]
        instruction = persona["instruction"]
        use_tools = persona.get("tools", True)
        reviewer_model = persona.get("model", sub_agent_model)

        system_prompt = (
            f"You are {name}, an expert reviewer.\n\n"
            f"Your review focus: {instruction}\n\n"
            "Review the document provided below. Be specific, cite evidence from the "
            "document, and do independent research where needed to verify claims. "
            "Structure your review with: Key Findings, Concerns, Strengths, and Verdict."
        )

        tools = [serper_search] if use_tools else None
        api_key = _next_api_key(key_cycle)

        agent = Agent(
            agent_name=f"reviewer-{name.lower().replace(' ', '-')}",
            system_prompt=system_prompt,
            model_name=reviewer_model,
            max_loops=3 if tools else 1,
            max_tokens=8192,
            output_type="final",
            verbose=False,
            tools=tools,
            llm_api_key=api_key,
        )

        task_prompt = f"Review the following document:\n\n{document}"
        raw_output = agent.run(task=task_prompt)
        output_str = str(raw_output) if raw_output is not None else ""

        return StructuredResult(
            agent_id=f"reviewer-{name.lower().replace(' ', '-')}",
            sub_task_id=name,
            output=output_str,
            confidence=0.8,
            metadata={
                "persona": name,
                "instruction": instruction,
                "model": reviewer_model,
            },
        )

    # Run reviewers concurrently with asyncio for progress reporting
    results: List[StructuredResult] = []
    total_reviewers = len(persona_list)

    if ctx:
        await ctx.info(f"Executing {total_reviewers} reviewers in parallel...")
        await ctx.report_progress(1, total_steps)

    async def run_reviewer_async(persona: dict) -> StructuredResult:
        return await asyncio.to_thread(run_reviewer, persona)

    sem = asyncio.Semaphore(min(total_reviewers, max_parallel))

    async def run_with_limit(persona: dict) -> StructuredResult:
        async with sem:
            return await run_reviewer_async(persona)

    tasks = [run_with_limit(p) for p in persona_list]
    completed = 0

    for coro in asyncio.as_completed(tasks):
        try:
            result = await asyncio.wait_for(coro, timeout=effective_timeout)
            results.append(result)
            completed += 1
            logger.info(f"Reviewer '{result.sub_task_id}' completed ({len(result.output)} chars)")
            if ctx:
                await ctx.report_progress(1 + completed, total_steps)
                await ctx.info(f"Reviewer {completed}/{total_reviewers} done: {result.sub_task_id}")
        except Exception as e:
            completed += 1
            logger.error(f"Reviewer failed: {e}")
            results.append(StructuredResult(
                agent_id="reviewer-unknown",
                sub_task_id="unknown",
                output=f"[Review failed: {e}]",
                confidence=0.0,
                metadata={"error": str(e)},
            ))
            if ctx:
                await ctx.report_progress(1 + completed, total_steps)

    if not results:
        return "Error: All reviewers failed. Check API keys and model configuration."

    # --- Step 3: Cross-model fact-checking ---
    fact_check_section = ""
    if fc_rounds > 0:
        if ctx:
            await ctx.info(f"Running {fc_rounds} rounds of cross-model fact-checking...")
        # Collect unique models used by reviewers for the fact-check pool
        reviewer_models = list({
            r.metadata.get("model", sub_agent_model)
            for r in results if r.confidence > 0
        })

        if len(reviewer_models) >= 2:
            logger.info(
                f"Running {fc_rounds} rounds of cross-model fact-checking "
                f"across {len(reviewer_models)} models..."
            )
            fact_checker = IterativeFactCheck(
                extraction_model=orchestrator_model,
                available_models=reviewer_models,
                tools=[serper_search],
                api_key_provider=lambda: _next_api_key(key_cycle) if key_cycle else None,
                max_rounds=fc_rounds,
                max_parallel=max_parallel,
            )
            try:
                fc_report = await asyncio.to_thread(fact_checker.verify_reviews, results)
                fact_check_section = (
                    f"Fact-check results ({fc_report.rounds_completed} rounds):\n"
                    f"{fc_report.summary()}"
                )
                if ctx:
                    await ctx.info("Fact-checking complete.")
            except Exception as e:
                logger.error(f"Fact-checking failed: {e}")
                fact_check_section = f"Fact-checking failed: {e}"
        else:
            logger.info("Only 1 model used, skipping cross-model fact-check")
            fact_check_section = "Cross-model fact-check skipped (need 2+ models)"

    # --- Step 4: Synthesize with blind spot analysis ---
    if ctx:
        await ctx.info("Synthesizing reviews with blind spot analysis...")
        await ctx.report_progress(total_steps - 1, total_steps)
    model_map_str = "\n".join(
        f"- {name}: {info['model'].split('/')[-1]} ({info['reason']})"
        for name, info in model_map.items()
    )

    formatted_reviews = "\n\n".join(
        f"### {r.sub_task_id} ({r.metadata.get('model', 'unknown').split('/')[-1]})\n{r.output}"
        for r in results if r.confidence > 0
    )

    synth_prompt = (synthesis_prompt or SMART_REVIEW_SYNTHESIS_PROMPT).format(
        document_summary=document[:500] + ("..." if len(document) > 500 else ""),
        formatted_reviews=formatted_reviews,
        model_map=model_map_str,
        fact_check_section=fact_check_section or "No fact-checking performed.",
    )

    aggregator = ResultAggregator(
        model=synthesis_model,
        api_key_provider=lambda: _next_api_key(key_cycle) if key_cycle else None,
    )

    aggregated = await asyncio.to_thread(
        aggregator.aggregate,
        results=results,
        original_task=synth_prompt,
    )

    if ctx:
        await ctx.report_progress(total_steps, total_steps)
        await ctx.info("Smart review complete.")

    # --- Format final output ---
    elapsed = time.time() - start_time
    output_parts = [aggregated.synthesized_answer]

    if aggregated.contradictions:
        output_parts.append("\n\n## Reviewer Disagreements")
        for c in aggregated.contradictions:
            output_parts.append(f"- {c}")

    if aggregated.gaps:
        output_parts.append("\n\n## Coverage Gaps")
        for g in aggregated.gaps:
            output_parts.append(f"- {g}")

    if fact_check_section:
        output_parts.append(f"\n\n{fact_check_section}")

    # Model assignment summary
    output_parts.append("\n\n## Model Assignments")
    for name, info in model_map.items():
        output_parts.append(f"- **{name}**: {info['model'].split('/')[-1]} — {info['reason']}")

    output_parts.append(
        f"\n\n---\n*Smart review completed in {elapsed:.1f}s by {len(results)} reviewers "
        f"across {len(models_used)} models, {fc_rounds} fact-check rounds "
        f"(synthesis: {synthesis_model})*"
    )

    return "\n".join(output_parts)


if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.environ.get("MCP_PORT", "8765"))

    # Configure FastMCP settings before running
    mcp.settings.port = port
    mcp.settings.host = "127.0.0.1"

    print(f"Starting PARL Orchestrator MCP Server on port {port}...")
    print("Available tools:")
    print("  - parl_execute:       Research/analysis with auto-decomposition")
    print("  - parl_review:        Multi-persona document review")
    print("  - parl_smart_review:  Enhanced review with auto model selection + fact-check")
    print("  - parl_config:        View server configuration")
    print(f"\nConnect MCP clients to: http://localhost:{port}/mcp")
    print("\nPress Ctrl+C to stop.\n")

    # Run MCP server directly with streamable-http transport
    mcp.run(transport="streamable-http", mount_path="/mcp")
