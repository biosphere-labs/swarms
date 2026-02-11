"""
PARLOrchestrator — Main orchestration class for PARL-inspired dynamic task orchestration.

Implements the full PARL pipeline:
  1. Decompose task into parallelizable sub-tasks
  2. Schedule execution via critical-path-aware cohort ordering
  3. Shard context for each sub-agent (isolated context windows)
  4. Execute cohorts in parallel using ThreadPoolExecutor
  5. Aggregate results with contradiction detection and gap identification
  6. Optionally iterate to fill gaps (max configurable iterations)

Inherits from BaseSwarm for clean framework integration.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from typing import Any, Dict, List, Optional, Tuple

from swarms.structs.agent import Agent
from swarms.structs.base_swarm import BaseSwarm
from swarms.structs.context_sharding import (
    ContextShardingManager,
    ContextShard,
    StructuredResult,
)
from swarms.structs.critical_path_scheduler import (
    CriticalPathScheduler,
    CohortGroup,
    ExecutionPlan,
    SubTask as SchedulerSubTask,
    SubTaskGraph as SchedulerSubTaskGraph,
)
from swarms.structs.decomposition_engine import (
    DecompositionEngine,
    SubTask as DecompositionSubTask,
    SubTaskGraph as DecompositionSubTaskGraph,
)
from swarms.structs.result_aggregator import (
    AggregatedOutput,
    ResultAggregator,
    StructuredResult as AggregatorStructuredResult,
)
from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger(log_folder="parl_orchestrator")


def _convert_decomposition_graph_to_scheduler_graph(
    decomp_graph: DecompositionSubTaskGraph,
) -> SchedulerSubTaskGraph:
    """
    Convert a DecompositionEngine SubTaskGraph to a CriticalPathScheduler SubTaskGraph.

    The two modules define SubTask and SubTaskGraph with slightly different schemas:
    - DecompositionEngine uses Dict-based dependencies: [{"from_group": 0, "to_group": 1}]
    - CriticalPathScheduler uses Tuple-based dependencies: [(0, 1)]
    - CriticalPathScheduler SubTask has an additional 'estimated_steps' field

    Args:
        decomp_graph: SubTaskGraph from DecompositionEngine

    Returns:
        SubTaskGraph compatible with CriticalPathScheduler
    """
    scheduler_groups: List[List[SchedulerSubTask]] = []
    for group in decomp_graph.parallel_groups:
        scheduler_tasks = []
        for task in group:
            scheduler_tasks.append(
                SchedulerSubTask(
                    id=task.id,
                    description=task.description,
                    context_hints=task.context_hints or [],
                    expected_output_format=task.expected_output_format or "text",
                    estimated_steps=1,
                )
            )
        scheduler_groups.append(scheduler_tasks)

    # Convert dependencies from Dict format to Tuple format
    scheduler_deps: List[Tuple[int, int]] = []
    for dep in decomp_graph.dependencies:
        if isinstance(dep, dict):
            from_idx = dep.get("from_group", dep.get("from", 0))
            to_idx = dep.get("to_group", dep.get("to", 0))
            scheduler_deps.append((int(from_idx), int(to_idx)))
        elif isinstance(dep, (list, tuple)) and len(dep) == 2:
            scheduler_deps.append((int(dep[0]), int(dep[1])))

    return SchedulerSubTaskGraph(
        parallel_groups=scheduler_groups,
        dependencies=scheduler_deps,
    )


def _build_sub_task_lookup(
    decomp_graph: DecompositionSubTaskGraph,
) -> Dict[str, DecompositionSubTask]:
    """
    Build a lookup dict from task ID to DecompositionSubTask for quick access.

    Args:
        decomp_graph: The decomposition graph

    Returns:
        Dict mapping task_id -> SubTask
    """
    lookup = {}
    for group in decomp_graph.parallel_groups:
        for task in group:
            lookup[task.id] = task
    return lookup


class PARLOrchestrator(BaseSwarm):
    """
    PARL-inspired orchestrator that dynamically decomposes complex tasks,
    executes sub-tasks in parallel with isolated context, and synthesizes
    results with explicit contradiction and gap detection.

    Inherits from BaseSwarm for framework integration. Creates sub-agents
    dynamically at execution time rather than requiring pre-configured agents.

    Args:
        name: Name of the orchestrator instance
        description: Description of the orchestrator
        orchestrator_model: Model for decomposition and synthesis (should be strong)
        sub_agent_model: Model for sub-agents (can be cheap/fast)
        max_parallel: Maximum number of concurrent sub-agents
        max_iterations: Maximum gap-fill iterations
        timeout: Overall timeout in seconds
        sub_agent_timeout: Per sub-agent timeout in seconds
        token_budget: Total token budget across all sub-agents
        agents: Optional pre-configured agents (passed to BaseSwarm)
        *args: Additional arguments passed to BaseSwarm
        **kwargs: Additional keyword arguments passed to BaseSwarm

    Example:
        >>> orchestrator = PARLOrchestrator(
        ...     orchestrator_model="gpt-4o-mini",
        ...     sub_agent_model="gpt-4o-mini",
        ...     max_parallel=8,
        ... )
        >>> result = orchestrator.run(
        ...     "Research competitor Acme Corp across funding, reviews, pricing, and team"
        ... )
        >>> print(result)
    """

    def __init__(
        self,
        name: str = "PARLOrchestrator",
        description: str = "PARL-inspired dynamic orchestrator with parallel sub-agent execution and context sharding",
        orchestrator_model: str = "gpt-4o-mini",
        sub_agent_model: str = "gpt-4o-mini",
        max_parallel: int = 10,
        max_iterations: int = 2,
        timeout: int = 300,
        sub_agent_timeout: int = 120,
        token_budget: int = 100000,
        agents: Optional[List[Agent]] = None,
        *args,
        **kwargs,
    ):
        # BaseSwarm requires agents to be a non-None list.
        # PARLOrchestrator creates agents dynamically, so we pass an empty
        # list if none are provided. The empty-list validation is disabled
        # in the upstream BaseSwarm.
        if agents is None:
            agents = []

        super().__init__(
            name=name,
            description=description,
            agents=agents,
            *args,
            **kwargs,
        )

        self.orchestrator_model = orchestrator_model
        self.sub_agent_model = sub_agent_model
        self.max_parallel = max_parallel
        self.max_iterations = max_iterations
        self.timeout = timeout
        self.sub_agent_timeout = sub_agent_timeout
        self.token_budget = token_budget

        # Token tracking
        self._tokens_used = 0

        # Initialize component modules
        self._decomposition_engine = DecompositionEngine(
            model=orchestrator_model,
            temperature=0.7,
            max_subtasks=20,
            min_subtasks_for_parallel=2,
        )
        self._scheduler = CriticalPathScheduler()
        self._context_manager = ContextShardingManager(
            max_context_tokens=4000,
            model_name=orchestrator_model,
        )
        self._aggregator = ResultAggregator(
            synthesis_model=orchestrator_model,
            max_tokens=4000,
            temperature=0.3,
        )

        logger.info(
            f"PARLOrchestrator initialized: orchestrator_model={orchestrator_model}, "
            f"sub_agent_model={sub_agent_model}, max_parallel={max_parallel}, "
            f"max_iterations={max_iterations}, timeout={timeout}s, "
            f"token_budget={token_budget}"
        )

    def run(self, task: str, *args, **kwargs) -> str:
        """
        Execute the full PARL orchestration pipeline on a task.

        Pipeline stages:
          1. Decompose task into sub-task graph
          2. If single-task, execute directly and return
          3. Schedule sub-tasks into ordered parallel cohorts
          4. For each cohort: shard contexts, create agents, execute in parallel
          5. Aggregate results with contradiction/gap detection
          6. If gaps found and iterations remain, re-decompose focusing on gaps

        Args:
            task: The task to orchestrate
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            str: The final synthesized answer
        """
        start_time = time.time()
        logger.info(f"PARLOrchestrator.run() starting: task={task[:100]}...")

        current_task = task
        all_results: List[StructuredResult] = []
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"=== Iteration {iteration}/{self.max_iterations} ===")

            # Check overall timeout
            elapsed = time.time() - start_time
            if elapsed >= self.timeout:
                logger.warning(
                    f"Overall timeout reached ({elapsed:.1f}s >= {self.timeout}s). "
                    "Synthesizing partial results."
                )
                break

            # Stage 1: Decompose
            try:
                decomp_graph = self._decomposition_engine.decompose(current_task)
            except Exception as e:
                logger.error(f"Decomposition failed: {e}")
                return self._execute_single_agent(task)

            logger.info(
                f"Decomposition result: {decomp_graph.total_subtasks()} sub-tasks, "
                f"{len(decomp_graph.parallel_groups)} groups, "
                f"is_single_task={decomp_graph.is_single_task()}"
            )

            # Stage 2: Single-task fallback
            if decomp_graph.is_single_task():
                logger.info("Task is single-task; executing directly without parallelization")
                if all_results:
                    # We have prior results (from gap-fill iteration), just synthesize
                    break
                return self._execute_single_agent(task)

            # Stage 3: Schedule into cohorts
            try:
                scheduler_graph = _convert_decomposition_graph_to_scheduler_graph(decomp_graph)
                execution_plan = self._scheduler.schedule(scheduler_graph)
            except Exception as e:
                logger.error(f"Scheduling failed: {e}")
                if all_results:
                    break
                return self._execute_single_agent(task)

            logger.info(
                f"Execution plan: {execution_plan.estimated_stages} stages, "
                f"{execution_plan.total_tasks} tasks, "
                f"critical_path={execution_plan.critical_path_length}"
            )

            # Build sub-task lookup for the decomposition graph
            sub_task_lookup = _build_sub_task_lookup(decomp_graph)

            # Stage 4: Execute cohorts
            budget_exceeded = False
            for stage_idx, cohort in enumerate(execution_plan.cohorts):
                elapsed = time.time() - start_time
                if elapsed >= self.timeout:
                    logger.warning(
                        f"Timeout during cohort execution (stage {stage_idx}). "
                        "Synthesizing partial results."
                    )
                    budget_exceeded = True
                    break

                if self._tokens_used >= self.token_budget:
                    logger.warning(
                        f"Token budget exhausted ({self._tokens_used}/{self.token_budget}). "
                        "Synthesizing partial results."
                    )
                    budget_exceeded = True
                    break

                logger.info(
                    f"Executing cohort stage {stage_idx + 1}/{execution_plan.estimated_stages}: "
                    f"{len(cohort.task_ids)} tasks"
                )

                # Collect context from previous results for this cohort
                prior_context = self._build_prior_context(all_results)

                cohort_results = self._execute_cohort(
                    cohort=cohort,
                    sub_task_lookup=sub_task_lookup,
                    parent_context=task + "\n\n" + prior_context,
                    start_time=start_time,
                )
                all_results.extend(cohort_results)

            if budget_exceeded:
                break

            # Stage 5: Aggregate
            aggregated = self._aggregate_results(all_results, task)

            # Stage 6: Check for gaps and decide whether to iterate
            if aggregated.gaps and iteration < self.max_iterations:
                logger.info(
                    f"Found {len(aggregated.gaps)} gaps. "
                    f"Re-decomposing for gap-fill (iteration {iteration + 1})"
                )
                # Build a focused task targeting the gaps
                current_task = self._build_gap_fill_task(task, aggregated.gaps)
                continue
            else:
                # No gaps or max iterations reached
                logger.info(
                    f"Orchestration complete after {iteration} iteration(s). "
                    f"Elapsed: {time.time() - start_time:.1f}s, "
                    f"Tokens used: ~{self._tokens_used}"
                )
                return aggregated.synthesized_answer

        # If we exit the loop (timeout/budget/max iterations), aggregate what we have
        if all_results:
            aggregated = self._aggregate_results(all_results, task)
            logger.info(
                f"Returning partial results after {iteration} iteration(s). "
                f"Elapsed: {time.time() - start_time:.1f}s"
            )
            return aggregated.synthesized_answer
        else:
            logger.warning("No results collected. Falling back to single-agent execution.")
            return self._execute_single_agent(task)

    def _execute_single_agent(self, task: str) -> str:
        """
        Execute a task with a single agent (no decomposition).

        Used as fallback when:
        - Task is too simple to parallelize
        - Decomposition or scheduling fails
        - No sub-tasks could be created

        Args:
            task: The task to execute

        Returns:
            str: The agent's output
        """
        logger.info("Executing single-agent fallback")
        try:
            agent = Agent(
                agent_name="parl-single-agent",
                system_prompt=(
                    "You are a focused agent executing a task. "
                    "Provide a clear, comprehensive response."
                ),
                model_name=self.orchestrator_model,
                max_loops=1,
                max_tokens=4096,
                output_type="str",
                verbose=False,
            )
            result = agent.run(task=task)
            return str(result) if result is not None else ""
        except Exception as e:
            logger.error(f"Single-agent execution failed: {e}")
            return f"Error: Failed to execute task. {e}"

    def _execute_cohort(
        self,
        cohort: CohortGroup,
        sub_task_lookup: Dict[str, DecompositionSubTask],
        parent_context: str,
        start_time: float,
    ) -> List[StructuredResult]:
        """
        Execute all tasks in a cohort in parallel using ThreadPoolExecutor.

        Creates fresh Agent instances for each sub-task with isolated context
        shards. Handles per-agent failures gracefully, collecting partial results.

        Args:
            cohort: The CohortGroup containing task_ids to execute
            sub_task_lookup: Lookup dict from task_id to SubTask
            parent_context: Parent context string for context sharding
            start_time: Overall start time for timeout checking

        Returns:
            List of StructuredResult objects (may be partial if some agents failed)
        """
        results: List[StructuredResult] = []
        max_workers = min(len(cohort.task_ids), self.max_parallel)

        # Remaining time for timeout
        elapsed = time.time() - start_time
        remaining_time = max(0, self.timeout - elapsed)
        effective_timeout = min(self.sub_agent_timeout, remaining_time)

        if effective_timeout <= 0:
            logger.warning("No time remaining for cohort execution")
            return results

        # Prepare agent configurations for each task in the cohort
        agent_configs: List[Dict[str, Any]] = []
        for task_id in cohort.task_ids:
            sub_task = sub_task_lookup.get(task_id)
            if sub_task is None:
                logger.warning(f"Sub-task {task_id} not found in lookup; skipping")
                continue

            # Create context shard for this sub-task
            try:
                shard = self._context_manager.create_shard(
                    sub_task=sub_task,
                    parent_context=parent_context,
                )
            except Exception as e:
                logger.error(f"Context sharding failed for {task_id}: {e}")
                # Create a minimal shard manually
                shard = ContextShard(
                    system_prompt=(
                        "You are a focused sub-agent. Complete the assigned task thoroughly."
                    ),
                    task_description=sub_task.description,
                    relevant_context_slice="",
                    output_format=sub_task.expected_output_format or "text",
                    token_count=0,
                )

            agent_configs.append({
                "task_id": task_id,
                "sub_task": sub_task,
                "shard": shard,
            })

            # Track token budget (estimate)
            self._tokens_used += shard.token_count

        if not agent_configs:
            return results

        # Check token budget before spawning
        if self._tokens_used >= self.token_budget:
            logger.warning(
                f"Token budget would be exceeded ({self._tokens_used}/{self.token_budget}). "
                "Skipping cohort execution."
            )
            return results

        # Execute all tasks in the cohort in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_config = {}
            for config in agent_configs:
                future = executor.submit(
                    self._execute_sub_agent,
                    config["task_id"],
                    config["shard"],
                )
                future_to_config[future] = config

            for future in as_completed(future_to_config, timeout=effective_timeout):
                config = future_to_config[future]
                task_id = config["task_id"]
                try:
                    result = future.result(timeout=0)
                    if result is not None:
                        results.append(result)
                        logger.info(
                            f"Sub-agent {task_id} completed "
                            f"(confidence={result.confidence:.2f})"
                        )
                except TimeoutError:
                    logger.warning(f"Sub-agent {task_id} timed out")
                    # Collect a failure result
                    results.append(
                        self._context_manager.collect_result(
                            agent_id=f"parl-sub-{task_id}",
                            sub_task_id=task_id,
                            result=f"[TIMEOUT] Sub-agent timed out after {effective_timeout:.0f}s",
                            confidence=0.0,
                            metadata={"error": "timeout", "timeout_seconds": effective_timeout},
                        )
                    )
                except Exception as e:
                    logger.error(f"Sub-agent {task_id} failed: {e}")
                    results.append(
                        self._context_manager.collect_result(
                            agent_id=f"parl-sub-{task_id}",
                            sub_task_id=task_id,
                            result=f"[ERROR] Sub-agent failed: {e}",
                            confidence=0.0,
                            metadata={"error": str(e)},
                        )
                    )

        return results

    def _execute_sub_agent(
        self,
        task_id: str,
        shard: ContextShard,
    ) -> Optional[StructuredResult]:
        """
        Create and execute a single sub-agent with an isolated context shard.

        The agent is created fresh for each sub-task to ensure complete isolation.
        The shard's system_prompt and task_description are used to configure the agent.

        Args:
            task_id: The sub-task ID
            shard: The context shard containing system prompt and task description

        Returns:
            StructuredResult if successful, None if failed
        """
        agent_name = f"parl-sub-{task_id}"
        execution_start = time.time()

        try:
            # Build the full system prompt including context
            system_prompt = shard.system_prompt
            if shard.relevant_context_slice:
                system_prompt += (
                    f"\n\n## Relevant Context\n{shard.relevant_context_slice}"
                )
            if shard.output_format:
                system_prompt += (
                    f"\n\n## Output Format\nProvide your output in {shard.output_format} format."
                )

            agent = Agent(
                agent_name=agent_name,
                system_prompt=system_prompt,
                model_name=self.sub_agent_model,
                max_loops=1,
                max_tokens=4096,
                output_type="str",
                verbose=False,
            )

            output = agent.run(task=shard.task_description)
            output_str = str(output) if output is not None else ""

            execution_time = time.time() - execution_start

            # Estimate token usage from output length (rough: 1 token ~ 4 chars)
            estimated_output_tokens = len(output_str) // 4
            self._tokens_used += estimated_output_tokens

            return self._context_manager.collect_result(
                agent_id=agent_name,
                sub_task_id=task_id,
                result=output_str,
                confidence=1.0,
                metadata={
                    "execution_time_seconds": round(execution_time, 2),
                    "model": self.sub_agent_model,
                    "estimated_output_tokens": estimated_output_tokens,
                },
            )

        except Exception as e:
            execution_time = time.time() - execution_start
            logger.error(f"Sub-agent {agent_name} execution error: {e}")
            return self._context_manager.collect_result(
                agent_id=agent_name,
                sub_task_id=task_id,
                result=f"[ERROR] Execution failed: {e}",
                confidence=0.0,
                metadata={
                    "error": str(e),
                    "execution_time_seconds": round(execution_time, 2),
                },
            )

    def _aggregate_results(
        self,
        results: List[StructuredResult],
        original_task: str,
    ) -> AggregatedOutput:
        """
        Aggregate all collected sub-agent results into a final synthesized answer.

        Converts ContextSharding StructuredResults to ResultAggregator StructuredResults
        (they have the same schema but are defined in different modules).

        Args:
            results: List of StructuredResult from context sharding module
            original_task: The original user task

        Returns:
            AggregatedOutput with synthesized answer, contradictions, and gaps
        """
        logger.info(f"Aggregating {len(results)} results")

        # Convert to the aggregator's StructuredResult type
        aggregator_results: List[AggregatorStructuredResult] = []
        for r in results:
            aggregator_results.append(
                AggregatorStructuredResult(
                    agent_id=r.agent_id,
                    sub_task_id=r.sub_task_id,
                    output=r.output,
                    confidence=r.confidence,
                    metadata=r.metadata,
                )
            )

        try:
            return self._aggregator.aggregate(
                results=aggregator_results,
                original_task=original_task,
            )
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            # Fallback: concatenate raw outputs
            raw_outputs = "\n\n---\n\n".join(
                f"[{r.sub_task_id}] {r.output}" for r in results
            )
            return AggregatedOutput(
                synthesized_answer=(
                    f"Aggregation failed ({e}). Raw sub-agent outputs:\n\n{raw_outputs}"
                ),
                contradictions=[],
                gaps=[f"Aggregation error: {e}"],
                confidence=0.0,
                metadata={"error": str(e)},
            )

    def _build_prior_context(self, results: List[StructuredResult]) -> str:
        """
        Build a context string from prior results to inform subsequent cohorts.

        Later cohorts may depend on results from earlier cohorts. This provides
        those results as context for context sharding.

        Args:
            results: All results collected so far

        Returns:
            str: Formatted context string from prior results
        """
        if not results:
            return ""

        context_parts = []
        for r in results:
            if r.confidence > 0.0:  # Skip failed results
                context_parts.append(
                    f"[Prior result from {r.sub_task_id}]:\n{r.output}"
                )

        return "\n\n".join(context_parts)

    def _build_gap_fill_task(
        self,
        original_task: str,
        gaps: List[str],
    ) -> str:
        """
        Build a refined task string that focuses on filling identified gaps.

        Args:
            original_task: The original user task
            gaps: List of gap descriptions from aggregation

        Returns:
            str: A task string focused on addressing the gaps
        """
        gap_descriptions = "\n".join(f"- {gap}" for gap in gaps)
        return (
            f"Original task: {original_task}\n\n"
            f"The following gaps were identified in prior analysis and need to be addressed:\n"
            f"{gap_descriptions}\n\n"
            f"Please focus specifically on filling these gaps with thorough, accurate information."
        )
