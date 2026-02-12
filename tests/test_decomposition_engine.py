"""
Tests for DecompositionEngine.

Pure-logic tests cover the heuristic methods (complexity threshold,
parallelizable pattern detection, single-task graph creation, graph validation).

Integration tests (marked with skipif) call a real LLM to verify end-to-end
decomposition behavior.
"""

import os
import pytest
from swarms.structs.decomposition_engine import (
    DecompositionEngine,
    SubTask,
    SubTaskGraph,
)


# ---------------------------------------------------------------------------
# Pure-logic: heuristic methods
# ---------------------------------------------------------------------------

class TestDecompositionHeuristics:
    """Test the heuristic methods that decide whether/how to decompose."""

    def _engine(self, **kwargs):
        defaults = dict(model="gpt-4o-mini", temperature=0.7, max_subtasks=20)
        defaults.update(kwargs)
        return DecompositionEngine(**defaults)

    # -- _is_task_too_simple --

    def test_empty_task_is_too_simple(self):
        engine = self._engine()
        assert engine._is_task_too_simple("") is True

    def test_very_short_task_is_too_simple(self):
        engine = self._engine()
        assert engine._is_task_too_simple("Hi") is True

    def test_few_words_is_too_simple(self):
        engine = self._engine()
        assert engine._is_task_too_simple("Add two numbers together") is True

    def test_single_action_without_multi_items(self):
        """Single-action verbs like 'explain' should trigger simplicity for lone items."""
        engine = self._engine()
        assert engine._is_task_too_simple("Explain the concept of gravity") is True

    def test_single_action_with_multi_items(self):
        """Single-action verb with 'and' should NOT be too simple."""
        engine = self._engine()
        result = engine._is_task_too_simple(
            "Explain the concept of gravity and compare it to electromagnetism"
        )
        assert result is False

    def test_complex_task_is_not_too_simple(self):
        engine = self._engine()
        task = (
            "Research the pros and cons of solar energy, wind energy, "
            "and hydroelectric power. Compare costs, efficiency, "
            "environmental impact, and scalability of each technology."
        )
        assert engine._is_task_too_simple(task) is False

    # -- _detect_parallelizable_patterns --

    def test_research_task_shows_parallel_patterns(self):
        engine = self._engine()
        task = (
            "Research solar energy and wind energy: compare costs, "
            "efficiency, environmental impact, and scalability"
        )
        assert engine._detect_parallelizable_patterns(task) is True

    def test_simple_question_no_parallel_patterns(self):
        engine = self._engine()
        assert engine._detect_parallelizable_patterns("What time is it in Tokyo?") is False

    def test_list_with_indicators(self):
        engine = self._engine()
        task = (
            "Analyze each of these files: main.py, utils.py, config.py. "
            "Review for different security issues."
        )
        assert engine._detect_parallelizable_patterns(task) is True

    # -- _create_single_task_graph --

    def test_single_task_graph(self):
        engine = self._engine()
        graph = engine._create_single_task_graph("Do something simple.")

        assert isinstance(graph, SubTaskGraph)
        assert graph.is_single_task() is True
        assert graph.total_subtasks() == 1
        assert len(graph.dependencies) == 0
        assert graph.parallel_groups[0][0].description == "Do something simple."

    # -- _validate_and_fix_graph --

    def test_validate_valid_graph(self):
        engine = self._engine()
        raw = {
            "parallel_groups": [
                [
                    {"id": "t1", "description": "Task 1", "context_hints": [], "expected_output_format": "text"},
                    {"id": "t2", "description": "Task 2", "context_hints": [], "expected_output_format": "text"},
                ],
                [
                    {"id": "t3", "description": "Task 3", "context_hints": [], "expected_output_format": "text"},
                ],
            ],
            "dependencies": [{"from_group": 0, "to_group": 1}],
        }
        graph = engine._validate_and_fix_graph(raw, "original task")
        assert graph.total_subtasks() == 3
        assert not graph.is_single_task()

    def test_validate_removes_duplicate_descriptions(self):
        """Duplicate sub-tasks in the same group should be deduplicated."""
        engine = self._engine()
        raw = {
            "parallel_groups": [
                [
                    {"id": "t1", "description": "Same task", "context_hints": [], "expected_output_format": "text"},
                    {"id": "t2", "description": "Same task", "context_hints": [], "expected_output_format": "text"},
                    {"id": "t3", "description": "Different task", "context_hints": [], "expected_output_format": "text"},
                ],
            ],
            "dependencies": [],
        }
        graph = engine._validate_and_fix_graph(raw, "original task")
        # After dedup, group should have 2 unique tasks
        assert graph.total_subtasks() == 2

    def test_validate_too_many_subtasks_collapses_to_single(self):
        """Exceeding max_subtasks should fall back to single-task graph."""
        engine = self._engine(max_subtasks=3)
        raw = {
            "parallel_groups": [
                [
                    {"id": f"t{i}", "description": f"Task {i}", "context_hints": [], "expected_output_format": "text"}
                    for i in range(10)
                ]
            ],
            "dependencies": [],
        }
        graph = engine._validate_and_fix_graph(raw, "original task")
        assert graph.is_single_task()

    def test_validate_invalid_json_falls_back(self):
        """Invalid graph structure should fall back to single-task."""
        engine = self._engine()
        raw = {"garbage": True}
        graph = engine._validate_and_fix_graph(raw, "original task")
        assert graph.is_single_task()


# ---------------------------------------------------------------------------
# SubTaskGraph model
# ---------------------------------------------------------------------------

class TestSubTaskGraphModel:
    """Test SubTaskGraph utility methods."""

    def test_is_single_task_true(self):
        graph = SubTaskGraph(
            parallel_groups=[[SubTask(id="t1", description="Only task")]],
            dependencies=[],
        )
        assert graph.is_single_task() is True

    def test_is_single_task_false_multiple_tasks(self):
        graph = SubTaskGraph(
            parallel_groups=[
                [SubTask(id="t1", description="Task 1"), SubTask(id="t2", description="Task 2")]
            ],
            dependencies=[],
        )
        assert graph.is_single_task() is False

    def test_is_single_task_false_multiple_groups(self):
        graph = SubTaskGraph(
            parallel_groups=[
                [SubTask(id="t1", description="Task 1")],
                [SubTask(id="t2", description="Task 2")],
            ],
            dependencies=[],
        )
        assert graph.is_single_task() is False

    def test_total_subtasks(self):
        graph = SubTaskGraph(
            parallel_groups=[
                [SubTask(id="t1", description="T1"), SubTask(id="t2", description="T2")],
                [SubTask(id="t3", description="T3")],
            ],
            dependencies=[],
        )
        assert graph.total_subtasks() == 3


# ---------------------------------------------------------------------------
# Decompose method — pure-logic path (no LLM call)
# ---------------------------------------------------------------------------

class TestDecomposeHeuristicPath:
    """Test the decompose() method for tasks that are handled by heuristics
    without reaching the LLM call."""

    def test_simple_task_returns_single_task_graph(self):
        """Tasks below complexity threshold should not be decomposed."""
        engine = DecompositionEngine(model="gpt-4o-mini")
        result = engine.decompose("What is 2+2?")
        assert result.is_single_task()

    def test_no_parallel_patterns_returns_single_task_graph(self):
        """A task without parallelizable patterns bypasses LLM decomposition."""
        engine = DecompositionEngine(model="gpt-4o-mini")
        # Long enough to pass complexity check but no parallel indicators
        result = engine.decompose(
            "Write a detailed step-by-step guide on how to bake a chocolate cake "
            "from scratch using basic ingredients found in a typical kitchen."
        )
        assert result.is_single_task()


# ---------------------------------------------------------------------------
# Integration tests — real LLM calls
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Requires OPENAI_API_KEY for LLM calls",
)
class TestDecompositionEngineIntegration:
    """Integration tests that call a real LLM for decomposition."""

    def test_complex_task_decomposes(self):
        """A clearly parallel task should produce multiple groups."""
        engine = DecompositionEngine(model="gpt-4o-mini")
        result = engine.decompose(
            "Research solar energy and wind energy: compare costs, efficiency, "
            "environmental impact, and scalability of each technology."
        )
        assert result.total_subtasks() > 1
        assert len(result.parallel_groups) >= 1

    def test_decomposed_tasks_have_required_fields(self):
        """Each sub-task should have id, description, and other fields."""
        engine = DecompositionEngine(model="gpt-4o-mini")
        result = engine.decompose(
            "Analyze and compare Python, Rust, and Go for building web servers, "
            "CLI tools, and data pipelines. Cover performance, developer experience, "
            "ecosystem maturity, and deployment options for each combination."
        )
        for group in result.parallel_groups:
            for task in group:
                assert task.id, "Sub-task must have an id"
                assert task.description, "Sub-task must have a description"
                assert len(task.description) > 5, "Description should be meaningful"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
