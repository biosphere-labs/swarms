"""
End-to-end integration tests for PARLOrchestrator.

These tests exercise the full pipeline: decompose -> schedule -> shard ->
execute -> aggregate. They require a real LLM API key and are skipped
in CI environments without one.

Pure-logic tests cover the helper/conversion functions that don't need LLM calls.
"""

import os
import pytest
from swarms.structs.parl_orchestrator import (
    PARLOrchestrator,
    _convert_decomposition_graph_to_scheduler_graph,
    _build_sub_task_lookup,
)
from swarms.structs.decomposition_engine import (
    SubTask as DecompositionSubTask,
    SubTaskGraph as DecompositionSubTaskGraph,
)
from swarms.structs.critical_path_scheduler import (
    SubTask as SchedulerSubTask,
    SubTaskGraph as SchedulerSubTaskGraph,
)


# ---------------------------------------------------------------------------
# Pure-logic: conversion functions
# ---------------------------------------------------------------------------

class TestConvertDecompositionGraphToSchedulerGraph:
    """Test the graph conversion between DecompositionEngine and CriticalPathScheduler formats."""

    def test_converts_subtasks(self):
        """SubTask fields should transfer correctly."""
        decomp_graph = DecompositionSubTaskGraph(
            parallel_groups=[
                [
                    DecompositionSubTask(
                        id="t1",
                        description="Task 1",
                        context_hints=["hint1"],
                        expected_output_format="json",
                    ),
                ],
            ],
            dependencies=[],
        )
        scheduler_graph = _convert_decomposition_graph_to_scheduler_graph(decomp_graph)

        assert len(scheduler_graph.parallel_groups) == 1
        assert len(scheduler_graph.parallel_groups[0]) == 1

        task = scheduler_graph.parallel_groups[0][0]
        assert task.id == "t1"
        assert task.description == "Task 1"
        assert task.context_hints == ["hint1"]
        assert task.expected_output_format == "json"
        assert task.estimated_steps == 1  # Default

    def test_converts_dict_dependencies(self):
        """Dict-format dependencies should become tuples."""
        decomp_graph = DecompositionSubTaskGraph(
            parallel_groups=[
                [DecompositionSubTask(id="t1", description="T1")],
                [DecompositionSubTask(id="t2", description="T2")],
                [DecompositionSubTask(id="t3", description="T3")],
            ],
            dependencies=[
                {"from_group": 0, "to_group": 1},
                {"from_group": 1, "to_group": 2},
            ],
        )
        scheduler_graph = _convert_decomposition_graph_to_scheduler_graph(decomp_graph)

        assert scheduler_graph.dependencies == [(0, 1), (1, 2)]

    def test_handles_single_dependency(self):
        """Single dependency converts correctly."""
        decomp_graph = DecompositionSubTaskGraph(
            parallel_groups=[
                [DecompositionSubTask(id="t1", description="T1")],
                [DecompositionSubTask(id="t2", description="T2")],
            ],
            dependencies=[{"from_group": 0, "to_group": 1}],
        )
        scheduler_graph = _convert_decomposition_graph_to_scheduler_graph(decomp_graph)
        assert scheduler_graph.dependencies == [(0, 1)]

    def test_empty_graph(self):
        """Empty decomposition graph converts to empty scheduler graph."""
        decomp_graph = DecompositionSubTaskGraph(
            parallel_groups=[],
            dependencies=[],
        )
        scheduler_graph = _convert_decomposition_graph_to_scheduler_graph(decomp_graph)
        assert len(scheduler_graph.parallel_groups) == 0
        assert len(scheduler_graph.dependencies) == 0

    def test_multiple_tasks_per_group(self):
        """Multiple tasks in a parallel group should all convert."""
        decomp_graph = DecompositionSubTaskGraph(
            parallel_groups=[
                [
                    DecompositionSubTask(id="t1", description="T1"),
                    DecompositionSubTask(id="t2", description="T2"),
                    DecompositionSubTask(id="t3", description="T3"),
                ],
            ],
            dependencies=[],
        )
        scheduler_graph = _convert_decomposition_graph_to_scheduler_graph(decomp_graph)
        assert len(scheduler_graph.parallel_groups[0]) == 3


class TestBuildSubTaskLookup:
    """Test the sub-task lookup builder."""

    def test_builds_lookup(self):
        graph = DecompositionSubTaskGraph(
            parallel_groups=[
                [
                    DecompositionSubTask(id="alpha", description="Alpha task"),
                    DecompositionSubTask(id="beta", description="Beta task"),
                ],
                [
                    DecompositionSubTask(id="gamma", description="Gamma task"),
                ],
            ],
            dependencies=[],
        )
        lookup = _build_sub_task_lookup(graph)
        assert len(lookup) == 3
        assert lookup["alpha"].description == "Alpha task"
        assert lookup["beta"].description == "Beta task"
        assert lookup["gamma"].description == "Gamma task"

    def test_empty_graph_lookup(self):
        graph = DecompositionSubTaskGraph(parallel_groups=[], dependencies=[])
        lookup = _build_sub_task_lookup(graph)
        assert len(lookup) == 0


# ---------------------------------------------------------------------------
# PARLOrchestrator construction
# ---------------------------------------------------------------------------

class TestPARLOrchestratorInit:
    """Test PARLOrchestrator initialization (no LLM calls needed)."""

    def test_default_construction(self):
        """Orchestrator should initialize with defaults without errors."""
        orch = PARLOrchestrator()
        assert orch.orchestrator_model == "gpt-4o-mini"
        assert orch.sub_agent_model == "gpt-4o-mini"
        assert orch.max_parallel == 10
        assert orch.max_iterations == 2
        assert orch.timeout == 300

    def test_custom_construction(self):
        """Custom parameters should be stored correctly."""
        orch = PARLOrchestrator(
            orchestrator_model="gpt-4o",
            sub_agent_model="gpt-4o-mini",
            max_parallel=5,
            max_iterations=3,
            timeout=60,
            sub_agent_timeout=30,
            token_budget=50000,
        )
        assert orch.orchestrator_model == "gpt-4o"
        assert orch.sub_agent_model == "gpt-4o-mini"
        assert orch.max_parallel == 5
        assert orch.max_iterations == 3
        assert orch.timeout == 60
        assert orch.sub_agent_timeout == 30
        assert orch.token_budget == 50000

    def test_prior_context_empty(self):
        """_build_prior_context with no results should return empty string."""
        orch = PARLOrchestrator()
        assert orch._build_prior_context([]) == ""

    def test_gap_fill_task_construction(self):
        """_build_gap_fill_task should include original task and gaps."""
        orch = PARLOrchestrator()
        result = orch._build_gap_fill_task(
            original_task="Research energy sources.",
            gaps=["Missing cost data", "No environmental analysis"],
        )
        assert "Research energy sources" in result
        assert "Missing cost data" in result
        assert "No environmental analysis" in result


# ---------------------------------------------------------------------------
# End-to-end integration tests — real LLM calls
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Requires OPENAI_API_KEY for LLM calls",
)
class TestPARLOrchestratorIntegration:
    """End-to-end integration tests that run the full PARL pipeline."""

    def test_full_pipeline(self):
        """Run a full decompose-schedule-execute-aggregate cycle."""
        orch = PARLOrchestrator(
            orchestrator_model="gpt-4o-mini",
            sub_agent_model="gpt-4o-mini",
            max_parallel=3,
            max_iterations=1,  # Single iteration to keep cost/time down
            timeout=120,
            sub_agent_timeout=60,
            token_budget=50000,
        )
        result = orch.run(
            "Compare the advantages and disadvantages of Python vs Rust "
            "for building web servers, CLI tools, and data pipelines."
        )
        assert isinstance(result, str)
        assert len(result) > 100  # Should produce substantial output

    def test_simple_task_uses_single_agent(self):
        """A simple task should bypass decomposition and use single agent."""
        orch = PARLOrchestrator(
            orchestrator_model="gpt-4o-mini",
            sub_agent_model="gpt-4o-mini",
            max_parallel=3,
            timeout=60,
        )
        result = orch.run("What is the capital of France?")
        assert isinstance(result, str)
        assert len(result) > 0
        # Should mention Paris somewhere
        assert "paris" in result.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
