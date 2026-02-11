"""
Unit tests for CriticalPathScheduler.

Tests DAG construction, topological sorting, critical path calculation,
parallel cohort grouping, and circular dependency detection.
All tests are pure logic — no LLM calls needed.
"""

import pytest
from swarms.structs.critical_path_scheduler import (
    CriticalPathScheduler,
    SubTask,
    SubTaskGraph,
    CohortGroup,
    ExecutionPlan,
)


# ---------------------------------------------------------------------------
# SubTaskGraph validation
# ---------------------------------------------------------------------------

class TestSubTaskGraph:
    """Test SubTaskGraph validation and structure."""

    def test_valid_graph(self):
        """A well-formed graph passes validation without errors."""
        graph = SubTaskGraph(
            parallel_groups=[
                [SubTask(id="A", description="Task A")],
                [SubTask(id="B", description="Task B")],
            ],
            dependencies=[(0, 1)],
        )
        graph.validate_structure()  # Should not raise

    def test_invalid_from_index(self):
        """Out-of-range from_idx raises ValueError."""
        graph = SubTaskGraph(
            parallel_groups=[
                [SubTask(id="A", description="Task A")],
            ],
            dependencies=[(5, 0)],
        )
        with pytest.raises(ValueError, match="from_idx 5 out of range"):
            graph.validate_structure()

    def test_invalid_to_index(self):
        """Out-of-range to_idx raises ValueError."""
        graph = SubTaskGraph(
            parallel_groups=[
                [SubTask(id="A", description="Task A")],
            ],
            dependencies=[(0, 5)],
        )
        with pytest.raises(ValueError, match="to_idx 5 out of range"):
            graph.validate_structure()

    def test_self_loop(self):
        """Self-loop dependency raises ValueError."""
        graph = SubTaskGraph(
            parallel_groups=[
                [SubTask(id="A", description="Task A")],
            ],
            dependencies=[(0, 0)],
        )
        with pytest.raises(ValueError, match="Self-loop detected"):
            graph.validate_structure()

    def test_empty_groups_with_no_deps(self):
        """An empty graph with no groups is structurally valid."""
        graph = SubTaskGraph(parallel_groups=[], dependencies=[])
        graph.validate_structure()  # Should not raise

    def test_negative_from_index(self):
        """Negative from_idx raises ValueError."""
        graph = SubTaskGraph(
            parallel_groups=[
                [SubTask(id="A", description="Task A")],
                [SubTask(id="B", description="Task B")],
            ],
            dependencies=[(-1, 1)],
        )
        with pytest.raises(ValueError, match="from_idx -1 out of range"):
            graph.validate_structure()


# ---------------------------------------------------------------------------
# CriticalPathScheduler — scheduling logic
# ---------------------------------------------------------------------------

class TestCriticalPathScheduler:
    """Test the CriticalPathScheduler class."""

    def test_linear_chain(self):
        """A -> B -> C produces 3 single-task cohorts."""
        graph = SubTaskGraph(
            parallel_groups=[
                [SubTask(id="A", description="Task A", estimated_steps=1)],
                [SubTask(id="B", description="Task B", estimated_steps=2)],
                [SubTask(id="C", description="Task C", estimated_steps=1)],
            ],
            dependencies=[(0, 1), (1, 2)],
        )

        scheduler = CriticalPathScheduler()
        plan = scheduler.schedule(graph)

        assert plan.total_tasks == 3
        assert plan.estimated_stages == 3
        assert plan.critical_path_length == 3
        assert len(plan.cohorts) == 3

        # Cohort ordering must be deterministic
        assert plan.cohorts[0].task_ids == ["A"]
        assert plan.cohorts[1].task_ids == ["B"]
        assert plan.cohorts[2].task_ids == ["C"]

    def test_fully_parallel(self):
        """Independent tasks should all be in one cohort."""
        graph = SubTaskGraph(
            parallel_groups=[
                [
                    SubTask(id="A", description="Task A", estimated_steps=1),
                    SubTask(id="B", description="Task B", estimated_steps=3),
                    SubTask(id="C", description="Task C", estimated_steps=2),
                ]
            ],
            dependencies=[],
        )

        scheduler = CriticalPathScheduler()
        plan = scheduler.schedule(graph)

        assert plan.total_tasks == 3
        assert plan.estimated_stages == 1
        assert plan.critical_path_length == 1
        assert len(plan.cohorts) == 1
        assert set(plan.cohorts[0].task_ids) == {"A", "B", "C"}
        assert plan.cohorts[0].estimated_steps == 3  # max of 1, 3, 2

    def test_diamond_dependency(self):
        """
        Diamond pattern: A -> B,C -> D produces 3 cohorts.

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
            dependencies=[(0, 1), (0, 2), (1, 3), (2, 3)],
        )

        scheduler = CriticalPathScheduler()
        plan = scheduler.schedule(graph)

        assert plan.total_tasks == 4
        assert plan.estimated_stages == 3  # A | B,C | D
        assert plan.critical_path_length == 3
        assert len(plan.cohorts) == 3

        assert plan.cohorts[0].task_ids == ["A"]
        assert set(plan.cohorts[1].task_ids) == {"B", "C"}
        assert plan.cohorts[1].estimated_steps == 3  # max(B=2, C=3)
        assert plan.cohorts[2].task_ids == ["D"]

    def test_circular_dependency_rejected(self):
        """Circular deps should raise ValueError."""
        graph = SubTaskGraph(
            parallel_groups=[
                [SubTask(id="A", description="Task A")],
                [SubTask(id="B", description="Task B")],
                [SubTask(id="C", description="Task C")],
            ],
            dependencies=[(0, 1), (1, 2), (2, 0)],
        )

        scheduler = CriticalPathScheduler()
        with pytest.raises(ValueError, match="Circular dependencies detected"):
            scheduler.schedule(graph)

    def test_complex_dag(self):
        """
        Complex DAG:
            A
           /|\\
          B C D
          | |
          E F
           \\|
            G

        Stages: A | B,C,D | E,F | G  (4 stages)
        Critical path: A -> B -> E -> G  or  A -> C -> F -> G  (length 4)
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
            ],
        )

        scheduler = CriticalPathScheduler()
        plan = scheduler.schedule(graph)

        assert plan.total_tasks == 7
        assert plan.estimated_stages == 4
        assert plan.critical_path_length == 4

    def test_empty_graph(self):
        """An empty graph produces an empty plan."""
        graph = SubTaskGraph(parallel_groups=[], dependencies=[])

        scheduler = CriticalPathScheduler()
        plan = scheduler.schedule(graph)

        assert plan.total_tasks == 0
        assert plan.estimated_stages == 0
        assert plan.critical_path_length == 0
        assert len(plan.cohorts) == 0

    def test_single_task(self):
        """A single task with no dependencies produces one cohort."""
        graph = SubTaskGraph(
            parallel_groups=[
                [SubTask(id="A", description="Task A", estimated_steps=5)]
            ],
            dependencies=[],
        )

        scheduler = CriticalPathScheduler()
        plan = scheduler.schedule(graph)

        assert plan.total_tasks == 1
        assert plan.estimated_stages == 1
        assert plan.critical_path_length == 1
        assert len(plan.cohorts) == 1
        assert plan.cohorts[0].task_ids == ["A"]
        assert plan.cohorts[0].estimated_steps == 5

    def test_wide_fan_out(self):
        """
        One root with many independent children:
            A -> B, C, D, E, F

        Should produce exactly 2 stages.
        """
        children = [
            SubTask(id=f"child_{i}", description=f"Child {i}", estimated_steps=i + 1)
            for i in range(5)
        ]
        graph = SubTaskGraph(
            parallel_groups=[
                [SubTask(id="root", description="Root", estimated_steps=1)],
                children,
            ],
            dependencies=[(0, 1)],
        )

        scheduler = CriticalPathScheduler()
        plan = scheduler.schedule(graph)

        assert plan.total_tasks == 6
        assert plan.estimated_stages == 2
        assert plan.critical_path_length == 2
        # Max estimated_steps among children: child_4 has estimated_steps=5
        assert plan.cohorts[1].estimated_steps == 5

    def test_two_independent_chains(self):
        """
        Two independent chains:
            A -> B
            C -> D

        Should produce 2 stages, each containing tasks from both chains.
        """
        graph = SubTaskGraph(
            parallel_groups=[
                [SubTask(id="A", description="Task A", estimated_steps=1)],
                [SubTask(id="B", description="Task B", estimated_steps=1)],
                [SubTask(id="C", description="Task C", estimated_steps=1)],
                [SubTask(id="D", description="Task D", estimated_steps=1)],
            ],
            dependencies=[(0, 1), (2, 3)],
        )

        scheduler = CriticalPathScheduler()
        plan = scheduler.schedule(graph)

        assert plan.total_tasks == 4
        assert plan.estimated_stages == 2
        # Stage 1 should contain A and C (both roots)
        assert set(plan.cohorts[0].task_ids) == {"A", "C"}
        # Stage 2 should contain B and D
        assert set(plan.cohorts[1].task_ids) == {"B", "D"}

    def test_multi_task_groups_are_parallelized(self):
        """Tasks within the same parallel_group run in one cohort."""
        graph = SubTaskGraph(
            parallel_groups=[
                [
                    SubTask(id="A1", description="Group A task 1", estimated_steps=2),
                    SubTask(id="A2", description="Group A task 2", estimated_steps=4),
                ],
                [
                    SubTask(id="B1", description="Group B task 1", estimated_steps=1),
                ],
            ],
            dependencies=[(0, 1)],
        )

        scheduler = CriticalPathScheduler()
        plan = scheduler.schedule(graph)

        assert plan.total_tasks == 3
        assert plan.estimated_stages == 2
        # First cohort has both A tasks
        assert set(plan.cohorts[0].task_ids) == {"A1", "A2"}
        assert plan.cohorts[0].estimated_steps == 4


# ---------------------------------------------------------------------------
# ExecutionPlan.get_critical_steps
# ---------------------------------------------------------------------------

class TestExecutionPlan:
    """Test ExecutionPlan methods."""

    def test_critical_steps_default_orchestrator_overhead(self):
        """CriticalSteps with default orchestrator overhead (1 step/stage)."""
        cohorts = [
            CohortGroup(task_ids=["A"], estimated_steps=1),
            CohortGroup(task_ids=["B"], estimated_steps=1),
        ]
        plan = ExecutionPlan(
            cohorts=cohorts,
            critical_path_length=2,
            estimated_stages=2,
            total_tasks=2,
        )
        # (1+1) + (1+1) = 4
        assert plan.get_critical_steps() == 4

    def test_critical_steps_custom_orchestrator_overhead(self):
        """CriticalSteps with custom orchestrator overhead."""
        cohorts = [
            CohortGroup(task_ids=["A"], estimated_steps=2),
            CohortGroup(task_ids=["B", "C"], estimated_steps=3),
            CohortGroup(task_ids=["D"], estimated_steps=1),
        ]
        plan = ExecutionPlan(
            cohorts=cohorts,
            critical_path_length=3,
            estimated_stages=3,
            total_tasks=4,
        )
        # (2+2) + (2+3) + (2+1) = 4 + 5 + 3 = 12
        assert plan.get_critical_steps(orchestrator_steps_per_stage=2) == 12

    def test_critical_steps_full_scheduling_roundtrip(self):
        """CriticalSteps computed end-to-end through scheduler."""
        graph = SubTaskGraph(
            parallel_groups=[
                [SubTask(id="A", description="Task A", estimated_steps=2)],
                [
                    SubTask(id="B", description="Task B", estimated_steps=3),
                    SubTask(id="C", description="Task C", estimated_steps=1),
                ],
                [SubTask(id="D", description="Task D", estimated_steps=2)],
            ],
            dependencies=[(0, 1), (1, 2)],
        )

        scheduler = CriticalPathScheduler()
        plan = scheduler.schedule(graph)

        # Stage 1: 1 + 2 = 3
        # Stage 2: 1 + max(3,1) = 4
        # Stage 3: 1 + 2 = 3
        # Total: 10
        assert plan.get_critical_steps(orchestrator_steps_per_stage=1) == 10

    def test_critical_steps_zero_orchestrator_overhead(self):
        """With zero orchestrator overhead, CriticalSteps equals sum of max sub-agent steps."""
        cohorts = [
            CohortGroup(task_ids=["A"], estimated_steps=5),
            CohortGroup(task_ids=["B", "C"], estimated_steps=3),
        ]
        plan = ExecutionPlan(
            cohorts=cohorts,
            critical_path_length=2,
            estimated_stages=2,
            total_tasks=3,
        )
        # (0+5) + (0+3) = 8
        assert plan.get_critical_steps(orchestrator_steps_per_stage=0) == 8

    def test_critical_steps_empty_plan(self):
        """An empty plan has 0 critical steps."""
        plan = ExecutionPlan(
            cohorts=[],
            critical_path_length=0,
            estimated_stages=0,
            total_tasks=0,
        )
        assert plan.get_critical_steps() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
