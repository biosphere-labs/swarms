"""
Unit tests for CriticalPathScheduler.

Tests DAG construction, topological sorting, critical path calculation,
and circular dependency detection.
"""

import pytest
from swarms.structs.critical_path_scheduler import (
    CriticalPathScheduler,
    SubTask,
    SubTaskGraph,
    CohortGroup,
    ExecutionPlan,
)


class TestSubTaskGraph:
    """Test SubTaskGraph validation and structure."""

    def test_valid_graph(self):
        """Test that a valid graph passes validation."""
        graph = SubTaskGraph(
            parallel_groups=[
                [SubTask(id="A", description="Task A")],
                [SubTask(id="B", description="Task B")],
            ],
            dependencies=[(0, 1)]
        )
        # Should not raise
        graph.validate_structure()

    def test_invalid_from_index(self):
        """Test that invalid from_index raises error."""
        graph = SubTaskGraph(
            parallel_groups=[
                [SubTask(id="A", description="Task A")],
            ],
            dependencies=[(5, 0)]  # Invalid from_index
        )
        with pytest.raises(ValueError, match="from_idx 5 out of range"):
            graph.validate_structure()

    def test_invalid_to_index(self):
        """Test that invalid to_index raises error."""
        graph = SubTaskGraph(
            parallel_groups=[
                [SubTask(id="A", description="Task A")],
            ],
            dependencies=[(0, 5)]  # Invalid to_index
        )
        with pytest.raises(ValueError, match="to_idx 5 out of range"):
            graph.validate_structure()

    def test_self_loop(self):
        """Test that self-loops are detected."""
        graph = SubTaskGraph(
            parallel_groups=[
                [SubTask(id="A", description="Task A")],
            ],
            dependencies=[(0, 0)]  # Self-loop
        )
        with pytest.raises(ValueError, match="Self-loop detected"):
            graph.validate_structure()


class TestCriticalPathScheduler:
    """Test the CriticalPathScheduler class."""

    def test_simple_linear_chain(self):
        """Test scheduling a simple linear dependency chain: A -> B -> C."""
        graph = SubTaskGraph(
            parallel_groups=[
                [SubTask(id="A", description="Task A", estimated_steps=1)],
                [SubTask(id="B", description="Task B", estimated_steps=2)],
                [SubTask(id="C", description="Task C", estimated_steps=1)],
            ],
            dependencies=[
                (0, 1),  # A -> B
                (1, 2),  # B -> C
            ]
        )

        scheduler = CriticalPathScheduler()
        plan = scheduler.schedule(graph)

        assert plan.total_tasks == 3
        assert plan.estimated_stages == 3  # Linear chain = 3 stages
        assert plan.critical_path_length == 3  # 3 nodes in the path
        assert len(plan.cohorts) == 3

        # Check cohort order
        assert plan.cohorts[0].task_ids == ["A"]
        assert plan.cohorts[1].task_ids == ["B"]
        assert plan.cohorts[2].task_ids == ["C"]

    def test_parallel_independent_tasks(self):
        """Test scheduling completely independent tasks: A, B, C (no dependencies)."""
        graph = SubTaskGraph(
            parallel_groups=[
                [
                    SubTask(id="A", description="Task A", estimated_steps=1),
                    SubTask(id="B", description="Task B", estimated_steps=3),
                    SubTask(id="C", description="Task C", estimated_steps=2),
                ]
            ],
            dependencies=[]  # No dependencies
        )

        scheduler = CriticalPathScheduler()
        plan = scheduler.schedule(graph)

        assert plan.total_tasks == 3
        assert plan.estimated_stages == 1  # All parallel = 1 stage
        assert plan.critical_path_length == 1  # No dependencies = depth 1
        assert len(plan.cohorts) == 1

        # All tasks in one cohort
        assert set(plan.cohorts[0].task_ids) == {"A", "B", "C"}
        # Max steps should be 3 (from task B)
        assert plan.cohorts[0].estimated_steps == 3

    def test_diamond_dependency(self):
        """
        Test diamond dependency pattern:
            A
           / \\
          B   C
           \\ /
            D
        """
        graph = SubTaskGraph(
            parallel_groups=[
                [SubTask(id="A", description="Task A", estimated_steps=1)],
                [SubTask(id="B", description="Task B", estimated_steps=2)],
                [SubTask(id="C", description="Task C", estimated_steps=3)],
                [SubTask(id="D", description="Task D", estimated_steps=1)],
            ],
            dependencies=[
                (0, 1),  # A -> B
                (0, 2),  # A -> C
                (1, 3),  # B -> D
                (2, 3),  # C -> D
            ]
        )

        scheduler = CriticalPathScheduler()
        plan = scheduler.schedule(graph)

        assert plan.total_tasks == 4
        assert plan.estimated_stages == 3  # A, then B||C, then D
        assert plan.critical_path_length == 3  # A -> C -> D (or A -> B -> D)
        assert len(plan.cohorts) == 3

        # Stage 1: A
        assert plan.cohorts[0].task_ids == ["A"]

        # Stage 2: B and C (parallel)
        assert set(plan.cohorts[1].task_ids) == {"B", "C"}
        # Max steps should be 3 (from C)
        assert plan.cohorts[1].estimated_steps == 3

        # Stage 3: D
        assert plan.cohorts[2].task_ids == ["D"]

    def test_circular_dependency_detection(self):
        """Test that circular dependencies are detected and rejected."""
        graph = SubTaskGraph(
            parallel_groups=[
                [SubTask(id="A", description="Task A")],
                [SubTask(id="B", description="Task B")],
                [SubTask(id="C", description="Task C")],
            ],
            dependencies=[
                (0, 1),  # A -> B
                (1, 2),  # B -> C
                (2, 0),  # C -> A (creates cycle)
            ]
        )

        scheduler = CriticalPathScheduler()
        with pytest.raises(ValueError, match="Circular dependencies detected"):
            scheduler.schedule(graph)

    def test_complex_dag(self):
        """
        Test a more complex DAG:
            A
           /|\\
          B C D
          | |
          E F
           \\|
            G
        """
        graph = SubTaskGraph(
            parallel_groups=[
                [SubTask(id="A", description="Task A", estimated_steps=1)],
                [SubTask(id="B", description="Task B", estimated_steps=2)],
                [SubTask(id="C", description="Task C", estimated_steps=1)],
                [SubTask(id="D", description="Task D", estimated_steps=1)],
                [SubTask(id="E", description="Task E", estimated_steps=1)],
                [SubTask(id="F", description="Task F", estimated_steps=3)],
                [SubTask(id="G", description="Task G", estimated_steps=1)],
            ],
            dependencies=[
                (0, 1),  # A -> B
                (0, 2),  # A -> C
                (0, 3),  # A -> D
                (1, 4),  # B -> E
                (2, 5),  # C -> F
                (4, 6),  # E -> G
                (5, 6),  # F -> G
            ]
        )

        scheduler = CriticalPathScheduler()
        plan = scheduler.schedule(graph)

        assert plan.total_tasks == 7
        # Expected stages:
        # 1. A
        # 2. B, C, D (parallel)
        # 3. E, F (parallel)
        # 4. G
        assert plan.estimated_stages == 4

        # Critical path: A -> C -> F -> G (or A -> B -> E -> G)
        assert plan.critical_path_length == 4

    def test_critical_steps_calculation(self):
        """Test the CriticalSteps metric calculation."""
        graph = SubTaskGraph(
            parallel_groups=[
                [SubTask(id="A", description="Task A", estimated_steps=2)],
                [
                    SubTask(id="B", description="Task B", estimated_steps=3),
                    SubTask(id="C", description="Task C", estimated_steps=1),
                ],
                [SubTask(id="D", description="Task D", estimated_steps=2)],
            ],
            dependencies=[
                (0, 1),  # A -> B,C
                (1, 2),  # B,C -> D
            ]
        )

        scheduler = CriticalPathScheduler()
        plan = scheduler.schedule(graph)

        # CriticalSteps = sum(S_main(t) + max_i S_sub,i(t))
        # Stage 1: 1 (orchestrator) + 2 (task A) = 3
        # Stage 2: 1 (orchestrator) + 3 (max of B=3, C=1) = 4
        # Stage 3: 1 (orchestrator) + 2 (task D) = 3
        # Total: 10
        critical_steps = plan.get_critical_steps(orchestrator_steps_per_stage=1)
        assert critical_steps == 10

    def test_empty_graph(self):
        """Test handling of empty graph."""
        graph = SubTaskGraph(
            parallel_groups=[],
            dependencies=[]
        )

        scheduler = CriticalPathScheduler()
        plan = scheduler.schedule(graph)

        assert plan.total_tasks == 0
        assert plan.estimated_stages == 0
        assert plan.critical_path_length == 0
        assert len(plan.cohorts) == 0

    def test_single_task(self):
        """Test handling of single task with no dependencies."""
        graph = SubTaskGraph(
            parallel_groups=[
                [SubTask(id="A", description="Task A", estimated_steps=5)]
            ],
            dependencies=[]
        )

        scheduler = CriticalPathScheduler()
        plan = scheduler.schedule(graph)

        assert plan.total_tasks == 1
        assert plan.estimated_stages == 1
        assert plan.critical_path_length == 1
        assert len(plan.cohorts) == 1
        assert plan.cohorts[0].task_ids == ["A"]
        assert plan.cohorts[0].estimated_steps == 5


class TestExecutionPlan:
    """Test ExecutionPlan methods."""

    def test_get_critical_steps_with_custom_orchestrator_overhead(self):
        """Test critical steps calculation with custom orchestrator overhead."""
        cohorts = [
            CohortGroup(task_ids=["A"], estimated_steps=2),
            CohortGroup(task_ids=["B", "C"], estimated_steps=3),
            CohortGroup(task_ids=["D"], estimated_steps=1),
        ]

        plan = ExecutionPlan(
            cohorts=cohorts,
            critical_path_length=3,
            estimated_stages=3,
            total_tasks=4
        )

        # With orchestrator_steps_per_stage=2:
        # Stage 1: 2 + 2 = 4
        # Stage 2: 2 + 3 = 5
        # Stage 3: 2 + 1 = 3
        # Total: 12
        critical_steps = plan.get_critical_steps(orchestrator_steps_per_stage=2)
        assert critical_steps == 12

    def test_get_critical_steps_default(self):
        """Test critical steps calculation with default orchestrator overhead."""
        cohorts = [
            CohortGroup(task_ids=["A"], estimated_steps=1),
            CohortGroup(task_ids=["B"], estimated_steps=1),
        ]

        plan = ExecutionPlan(
            cohorts=cohorts,
            critical_path_length=2,
            estimated_stages=2,
            total_tasks=2
        )

        # Default orchestrator_steps_per_stage=1:
        # Stage 1: 1 + 1 = 2
        # Stage 2: 1 + 1 = 2
        # Total: 4
        critical_steps = plan.get_critical_steps()
        assert critical_steps == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
