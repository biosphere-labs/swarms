"""
CriticalPathScheduler — Execution ordering and parallel grouping.

This module provides DAG-based execution planning for sub-tasks, grouping
independent tasks into parallel cohorts to minimize wall-clock time.

Implements the CriticalSteps metric from the PARL paper:
    CriticalSteps = sum_t(S_main(t) + max_i S_sub,i(t))

Where:
- S_main(t): Orchestrator steps at stage t
- max_i S_sub,i(t): Slowest sub-agent in parallel cohort at stage t
"""

from typing import List, Set, Dict, Tuple
import networkx as nx
from pydantic import BaseModel, Field
from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger(log_folder="critical_path_scheduler")


class SubTask(BaseModel):
    """
    Represents a single sub-task in the decomposition.

    Attributes:
        id: Unique identifier for the sub-task
        description: Human-readable description of what the task does
        context_hints: Optional list of context snippets relevant to this task
        expected_output_format: Optional description of expected output format
        estimated_steps: Estimated number of LLM steps (default: 1)
    """
    id: str = Field(..., description="Unique sub-task identifier")
    description: str = Field(..., description="Task description")
    context_hints: List[str] = Field(default_factory=list, description="Context snippets")
    expected_output_format: str = Field(default="", description="Expected output format")
    estimated_steps: int = Field(default=1, description="Estimated LLM steps", ge=1)


class SubTaskGraph(BaseModel):
    """
    Represents the task decomposition as a graph.

    The graph is represented as parallel groups with dependencies between groups.
    This is the input format from the DecompositionEngine.

    Attributes:
        parallel_groups: Lists of tasks that can run in parallel
        dependencies: Edges between groups (from_group_index, to_group_index)
    """
    parallel_groups: List[List[SubTask]] = Field(..., description="Parallel task groups")
    dependencies: List[Tuple[int, int]] = Field(
        default_factory=list,
        description="Dependencies between groups (from_idx, to_idx)"
    )

    def validate_structure(self) -> None:
        """
        Validate that the graph structure is well-formed.

        Raises:
            ValueError: If dependencies reference invalid group indices
        """
        num_groups = len(self.parallel_groups)
        for from_idx, to_idx in self.dependencies:
            if from_idx < 0 or from_idx >= num_groups:
                raise ValueError(f"Invalid dependency: from_idx {from_idx} out of range")
            if to_idx < 0 or to_idx >= num_groups:
                raise ValueError(f"Invalid dependency: to_idx {to_idx} out of range")
            if from_idx == to_idx:
                raise ValueError(f"Self-loop detected: group {from_idx} depends on itself")


class CohortGroup(BaseModel):
    """
    A group of sub-tasks that can execute in parallel.

    Attributes:
        task_ids: IDs of tasks in this cohort
        estimated_steps: Maximum estimated steps among tasks in cohort
    """
    task_ids: List[str] = Field(..., description="Task IDs in this parallel cohort")
    estimated_steps: int = Field(default=1, description="Max steps in cohort", ge=1)


class ExecutionPlan(BaseModel):
    """
    The final execution plan with ordered parallel cohorts.

    This plan minimizes wall-clock time by grouping independent tasks
    and ordering them to respect dependencies.

    Attributes:
        cohorts: Ordered list of parallel cohorts
        critical_path_length: Length of the longest dependency chain
        estimated_stages: Number of sequential execution rounds
        total_tasks: Total number of tasks in the plan
    """
    cohorts: List[CohortGroup] = Field(..., description="Ordered parallel cohorts")
    critical_path_length: int = Field(..., description="Longest dependency chain", ge=0)
    estimated_stages: int = Field(..., description="Number of sequential rounds", ge=0)
    total_tasks: int = Field(..., description="Total number of tasks", ge=0)

    def get_critical_steps(self, orchestrator_steps_per_stage: int = 1) -> int:
        """
        Calculate the CriticalSteps metric from the PARL paper.

        CriticalSteps = sum_t(S_main(t) + max_i S_sub,i(t))

        Args:
            orchestrator_steps_per_stage: Estimated orchestrator overhead per stage

        Returns:
            Total critical steps (wall-clock latency estimate)
        """
        critical_steps = 0
        for cohort in self.cohorts:
            # S_main(t): orchestrator overhead
            # max_i S_sub,i(t): slowest sub-agent in this cohort
            critical_steps += orchestrator_steps_per_stage + cohort.estimated_steps
        return critical_steps


class CriticalPathScheduler:
    """
    Scheduler that determines optimal execution order for sub-tasks.

    Uses NetworkX to build a DAG, perform topological sorting, and identify
    the critical path. Groups independent tasks into parallel cohorts to
    minimize wall-clock execution time.
    """

    def __init__(self):
        """Initialize the scheduler."""
        self.logger = logger

    def schedule(self, sub_task_graph: SubTaskGraph) -> ExecutionPlan:
        """
        Create an execution plan from a sub-task graph.

        Args:
            sub_task_graph: The decomposed task graph

        Returns:
            ExecutionPlan with ordered parallel cohorts

        Raises:
            ValueError: If the graph contains cycles or invalid structure
        """
        # Validate input structure
        sub_task_graph.validate_structure()

        # Build DAG from the sub-task graph
        dag = self._build_dag(sub_task_graph)

        # Detect circular dependencies
        self._validate_dag(dag)

        # Compute topological generations (parallel cohorts)
        cohorts = self._compute_cohorts(dag, sub_task_graph)

        # Calculate critical path
        critical_path_length = self._compute_critical_path(dag)

        # Count total tasks
        total_tasks = sum(len(group) for group in sub_task_graph.parallel_groups)

        plan = ExecutionPlan(
            cohorts=cohorts,
            critical_path_length=critical_path_length,
            estimated_stages=len(cohorts),
            total_tasks=total_tasks
        )

        self.logger.info(
            f"Created execution plan: {len(cohorts)} stages, "
            f"{total_tasks} tasks, critical path length {critical_path_length}"
        )

        return plan

    def _build_dag(self, sub_task_graph: SubTaskGraph) -> nx.DiGraph:
        """
        Build a NetworkX DAG from the sub-task graph.

        Args:
            sub_task_graph: Input task graph

        Returns:
            NetworkX directed graph
        """
        dag = nx.DiGraph()

        # Add all tasks as nodes with their estimated steps
        for group in sub_task_graph.parallel_groups:
            for task in group:
                dag.add_node(
                    task.id,
                    estimated_steps=task.estimated_steps,
                    description=task.description
                )

        # Add edges based on dependencies between groups
        for from_group_idx, to_group_idx in sub_task_graph.dependencies:
            from_group = sub_task_graph.parallel_groups[from_group_idx]
            to_group = sub_task_graph.parallel_groups[to_group_idx]

            # Create edges from all tasks in from_group to all tasks in to_group
            for from_task in from_group:
                for to_task in to_group:
                    dag.add_edge(from_task.id, to_task.id)

        return dag

    def _validate_dag(self, dag: nx.DiGraph) -> None:
        """
        Validate that the graph is a valid DAG (no cycles).

        Args:
            dag: The directed graph to validate

        Raises:
            ValueError: If circular dependencies are detected
        """
        try:
            # This will raise NetworkXError if there are cycles
            cycles = list(nx.simple_cycles(dag))
            if cycles:
                # Format cycle information for error message
                cycle_info = []
                for cycle in cycles[:3]:  # Show first 3 cycles
                    cycle_str = " -> ".join(cycle) + f" -> {cycle[0]}"
                    cycle_info.append(cycle_str)

                error_msg = (
                    f"Circular dependencies detected in task graph. "
                    f"Found {len(cycles)} cycle(s). Examples:\n"
                    + "\n".join(f"  - {c}" for c in cycle_info)
                )
                self.logger.error(error_msg)
                raise ValueError(error_msg)
        except nx.NetworkXNoCycle:
            # No cycles found, this is good
            pass

    def _compute_cohorts(
        self,
        dag: nx.DiGraph,
        sub_task_graph: SubTaskGraph
    ) -> List[CohortGroup]:
        """
        Compute parallel cohorts using topological generations.

        Topological generations are layers where all nodes in a layer
        have no dependencies on other nodes in the same layer.

        Args:
            dag: The DAG of tasks
            sub_task_graph: Original task graph (for estimated steps)

        Returns:
            List of CohortGroup objects in execution order
        """
        # Create task_id -> estimated_steps mapping
        task_steps = {}
        for group in sub_task_graph.parallel_groups:
            for task in group:
                task_steps[task.id] = task.estimated_steps

        # Get topological generations
        # Each generation is a set of nodes with no dependencies within the set
        generations = list(nx.topological_generations(dag))

        cohorts = []
        for generation in generations:
            task_ids = list(generation)
            # Max steps in this cohort (for CriticalSteps calculation)
            max_steps = max(
                (task_steps.get(tid, 1) for tid in task_ids),
                default=1
            )
            cohorts.append(
                CohortGroup(
                    task_ids=task_ids,
                    estimated_steps=max_steps
                )
            )

        return cohorts

    def _compute_critical_path(self, dag: nx.DiGraph) -> int:
        """
        Compute the length of the critical path (longest path in DAG).

        The critical path represents the minimum wall-clock time needed
        even with infinite parallelism.

        Args:
            dag: The DAG of tasks

        Returns:
            Length of the longest path
        """
        if len(dag.nodes) == 0:
            return 0

        # For a DAG, the longest path can be computed using dynamic programming
        # We'll use NetworkX's dag_longest_path_length which returns the number of edges
        try:
            # This returns the number of edges in the longest path
            path_length = nx.dag_longest_path_length(dag)
            # Add 1 because we want to count nodes, not edges
            # (a path with N edges has N+1 nodes)
            return path_length + 1 if path_length >= 0 else 1
        except nx.NetworkXError:
            # If the graph has no edges (all independent tasks)
            return 1


# Example usage and testing helper
def create_simple_example() -> SubTaskGraph:
    """
    Create a simple example task graph for testing.

    Graph structure:
        A
       / \\
      B   C
       \\ /
        D

    Returns:
        SubTaskGraph with the example structure
    """
    tasks_a = [SubTask(id="A", description="Initial task", estimated_steps=2)]
    tasks_b = [SubTask(id="B", description="Process left branch", estimated_steps=1)]
    tasks_c = [SubTask(id="C", description="Process right branch", estimated_steps=3)]
    tasks_d = [SubTask(id="D", description="Merge results", estimated_steps=1)]

    return SubTaskGraph(
        parallel_groups=[tasks_a, tasks_b, tasks_c, tasks_d],
        dependencies=[
            (0, 1),  # A -> B
            (0, 2),  # A -> C
            (1, 3),  # B -> D
            (2, 3),  # C -> D
        ]
    )


if __name__ == "__main__":
    # Quick test
    scheduler = CriticalPathScheduler()
    example_graph = create_simple_example()

    print("Input graph:")
    print(f"  Groups: {len(example_graph.parallel_groups)}")
    print(f"  Dependencies: {example_graph.dependencies}")

    plan = scheduler.schedule(example_graph)

    print("\nExecution plan:")
    print(f"  Total tasks: {plan.total_tasks}")
    print(f"  Stages: {plan.estimated_stages}")
    print(f"  Critical path length: {plan.critical_path_length}")
    print(f"  Critical steps: {plan.get_critical_steps()}")

    print("\nCohorts:")
    for i, cohort in enumerate(plan.cohorts):
        print(f"  Stage {i+1}: {cohort.task_ids} (max steps: {cohort.estimated_steps})")
