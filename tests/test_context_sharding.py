"""
Tests for ContextShardingManager.

Tests context isolation, hint-based extraction, token budget truncation,
and result collection. All tests use real inputs — no mocks.

The token-counting helper (litellm.encode) runs locally via tiktoken, so
no API keys are needed.
"""

import pytest
from swarms.structs.context_sharding import (
    ContextShardingManager,
    ContextShard,
    StructuredResult,
    SubTask,
)


# ---------------------------------------------------------------------------
# ContextShard model
# ---------------------------------------------------------------------------

class TestContextShard:
    """Test the ContextShard model properties."""

    def test_shard_is_immutable(self):
        """ContextShard should be frozen (immutable) after creation."""
        shard = ContextShard(
            system_prompt="You are helpful.",
            task_description="Do something.",
            relevant_context_slice="context here",
            output_format="markdown",
            token_count=50,
        )
        with pytest.raises(Exception):
            shard.system_prompt = "changed"

    def test_shard_stores_all_fields(self):
        """ContextShard stores all provided values correctly."""
        shard = ContextShard(
            system_prompt="prompt",
            task_description="task",
            relevant_context_slice="slice",
            output_format="json",
            token_count=42,
        )
        assert shard.system_prompt == "prompt"
        assert shard.task_description == "task"
        assert shard.relevant_context_slice == "slice"
        assert shard.output_format == "json"
        assert shard.token_count == 42


# ---------------------------------------------------------------------------
# StructuredResult model
# ---------------------------------------------------------------------------

class TestStructuredResult:
    """Test StructuredResult model."""

    def test_defaults(self):
        """Default confidence is 1.0 and metadata is empty."""
        result = StructuredResult(
            agent_id="agent-1",
            sub_task_id="task-1",
            output="hello world",
        )
        assert result.confidence == 1.0
        assert result.metadata == {}

    def test_confidence_bounds(self):
        """Confidence must be between 0 and 1."""
        with pytest.raises(Exception):
            StructuredResult(
                agent_id="a", sub_task_id="t", output="x", confidence=1.5
            )
        with pytest.raises(Exception):
            StructuredResult(
                agent_id="a", sub_task_id="t", output="x", confidence=-0.1
            )


# ---------------------------------------------------------------------------
# ContextShardingManager — shard creation
# ---------------------------------------------------------------------------

class TestContextShardingManagerCreateShard:
    """Test ContextShardingManager.create_shard with real inputs."""

    def _make_manager(self, **kwargs):
        """Helper to create a manager with sane defaults."""
        defaults = dict(
            default_system_prompt="You are a focused sub-agent.",
            max_context_tokens=4000,
            model_name="gpt-4",
        )
        defaults.update(kwargs)
        return ContextShardingManager(**defaults)

    def test_shard_contains_task_description(self):
        """The shard's task_description should match the sub-task."""
        manager = self._make_manager()
        sub_task = SubTask(
            id="t1",
            description="Analyze revenue trends",
            context_hints=["revenue"],
        )
        shard = manager.create_shard(sub_task=sub_task, parent_context="Revenue grew 20%.")
        assert shard.task_description == "Analyze revenue trends"

    def test_shard_extracts_relevant_context(self):
        """Context hints should select matching paragraphs from parent context."""
        manager = self._make_manager()

        parent_context = (
            "The company was founded in 2010.\n\n"
            "Revenue grew 20% year over year, reaching $50M.\n\n"
            "The engineering team has 120 members.\n\n"
            "Pricing starts at $99/month for the basic plan."
        )

        sub_task = SubTask(
            id="t1",
            description="Analyze revenue",
            context_hints=["revenue"],
        )
        shard = manager.create_shard(sub_task=sub_task, parent_context=parent_context)

        # Should include the revenue paragraph
        assert "Revenue grew 20%" in shard.relevant_context_slice
        # Should NOT include unrelated paragraphs
        assert "engineering team" not in shard.relevant_context_slice

    def test_shard_with_no_hints_returns_empty_context(self):
        """No context hints means empty relevant_context_slice."""
        manager = self._make_manager()
        sub_task = SubTask(
            id="t1",
            description="General task",
            context_hints=None,
        )
        shard = manager.create_shard(
            sub_task=sub_task,
            parent_context="Some long context that should be ignored.",
        )
        assert shard.relevant_context_slice == ""

    def test_shard_with_empty_parent_context(self):
        """Empty parent context produces empty relevant_context_slice."""
        manager = self._make_manager()
        sub_task = SubTask(
            id="t1",
            description="A task",
            context_hints=["anything"],
        )
        shard = manager.create_shard(sub_task=sub_task, parent_context="")
        assert shard.relevant_context_slice == ""

    def test_shard_uses_custom_system_prompt(self):
        """A custom system prompt overrides the default."""
        manager = self._make_manager(default_system_prompt="Default prompt.")
        sub_task = SubTask(id="t1", description="task")
        shard = manager.create_shard(
            sub_task=sub_task,
            parent_context="",
            custom_system_prompt="Custom prompt here.",
        )
        assert shard.system_prompt == "Custom prompt here."

    def test_shard_uses_default_system_prompt(self):
        """Without a custom prompt, the default is used."""
        manager = self._make_manager(default_system_prompt="Default prompt.")
        sub_task = SubTask(id="t1", description="task")
        shard = manager.create_shard(sub_task=sub_task, parent_context="")
        assert shard.system_prompt == "Default prompt."

    def test_shard_output_format(self):
        """Output format comes from the sub-task's expected_output_format."""
        manager = self._make_manager()
        sub_task = SubTask(
            id="t1", description="task", expected_output_format="json"
        )
        shard = manager.create_shard(sub_task=sub_task, parent_context="")
        assert shard.output_format == "json"

    def test_shard_output_format_defaults_to_markdown(self):
        """If no expected_output_format, default is 'markdown'."""
        manager = self._make_manager()
        sub_task = SubTask(id="t1", description="task")
        shard = manager.create_shard(sub_task=sub_task, parent_context="")
        assert shard.output_format == "markdown"

    def test_token_count_is_positive_for_nonempty_shard(self):
        """A shard with real text should have a positive token count."""
        manager = self._make_manager()
        sub_task = SubTask(
            id="t1",
            description="Analyze the quarterly revenue report.",
            context_hints=["revenue"],
        )
        shard = manager.create_shard(
            sub_task=sub_task,
            parent_context="Revenue grew 20% in Q3, reaching $50M total.",
        )
        assert shard.token_count > 0

    def test_multiple_hints_match_multiple_paragraphs(self):
        """Multiple context hints can select multiple paragraphs."""
        manager = self._make_manager()
        parent_context = (
            "The pricing is $99/month.\n\n"
            "The team consists of 50 engineers.\n\n"
            "Revenue hit $10M ARR last quarter.\n\n"
            "The company is based in San Francisco."
        )
        sub_task = SubTask(
            id="t1",
            description="Compare pricing and revenue",
            context_hints=["pricing", "revenue"],
        )
        shard = manager.create_shard(sub_task=sub_task, parent_context=parent_context)

        assert "pricing" in shard.relevant_context_slice.lower()
        assert "revenue" in shard.relevant_context_slice.lower()
        # Unrelated paragraphs should be absent
        assert "San Francisco" not in shard.relevant_context_slice

    def test_case_insensitive_hint_matching(self):
        """Hint matching should be case-insensitive."""
        manager = self._make_manager()
        parent_context = "REVENUE grew 20% in Q3."
        sub_task = SubTask(
            id="t1",
            description="Revenue analysis",
            context_hints=["revenue"],
        )
        shard = manager.create_shard(sub_task=sub_task, parent_context=parent_context)
        assert "REVENUE" in shard.relevant_context_slice


# ---------------------------------------------------------------------------
# ContextShardingManager — truncation
# ---------------------------------------------------------------------------

class TestContextShardingManagerTruncation:
    """Test context truncation when token budget is exceeded."""

    def test_small_budget_truncates_context(self):
        """With a very small token budget, context gets truncated."""
        manager = ContextShardingManager(
            default_system_prompt="Short.",
            max_context_tokens=50,  # Very small budget
            model_name="gpt-4",
        )
        long_context = "This is a long context paragraph. " * 100

        sub_task = SubTask(
            id="t1",
            description="task",
            context_hints=["context"],
        )

        # The parent_context itself contains "context" so the hint matches
        shard = manager.create_shard(
            sub_task=sub_task,
            parent_context=long_context,
        )

        # The shard's token count should be at or below the budget
        assert shard.token_count <= 50 or "[Context truncated" in shard.relevant_context_slice


# ---------------------------------------------------------------------------
# ContextShardingManager — result collection
# ---------------------------------------------------------------------------

class TestContextShardingManagerCollectResult:
    """Test ContextShardingManager.collect_result."""

    def test_collect_result_basic(self):
        """Collect a basic result with all fields."""
        manager = ContextShardingManager()
        result = manager.collect_result(
            agent_id="agent-1",
            sub_task_id="task-1",
            result="The answer is 42.",
            confidence=0.95,
            metadata={"execution_time": 1.5},
        )
        assert isinstance(result, StructuredResult)
        assert result.agent_id == "agent-1"
        assert result.sub_task_id == "task-1"
        assert result.output == "The answer is 42."
        assert result.confidence == 0.95
        assert result.metadata["execution_time"] == 1.5

    def test_collect_result_defaults(self):
        """Default confidence is 1.0 and metadata is empty dict."""
        manager = ContextShardingManager()
        result = manager.collect_result(
            agent_id="a1",
            sub_task_id="t1",
            result="output",
        )
        assert result.confidence == 1.0
        assert result.metadata == {}

    def test_collect_result_zero_confidence(self):
        """A result can have zero confidence (failed agent)."""
        manager = ContextShardingManager()
        result = manager.collect_result(
            agent_id="a1",
            sub_task_id="t1",
            result="[ERROR] Timeout",
            confidence=0.0,
            metadata={"error": "timeout"},
        )
        assert result.confidence == 0.0
        assert result.metadata["error"] == "timeout"


# ---------------------------------------------------------------------------
# Context isolation: different tasks get different shards
# ---------------------------------------------------------------------------

class TestContextIsolation:
    """Verify that different sub-tasks get genuinely isolated context shards."""

    def test_different_tasks_get_different_context(self):
        """Two sub-tasks with different hints should receive non-overlapping context."""
        manager = ContextShardingManager(max_context_tokens=4000)

        parent_context = (
            "Revenue grew 20% in Q3.\n\n"
            "The engineering team migrated to Kubernetes.\n\n"
            "Customer satisfaction scores rose to 4.5/5.\n\n"
            "The CFO announced a stock buyback program."
        )

        task_revenue = SubTask(
            id="revenue", description="Analyze revenue", context_hints=["revenue"]
        )
        task_engineering = SubTask(
            id="eng", description="Analyze engineering", context_hints=["engineering"]
        )

        shard_rev = manager.create_shard(sub_task=task_revenue, parent_context=parent_context)
        shard_eng = manager.create_shard(sub_task=task_engineering, parent_context=parent_context)

        # Revenue shard should mention revenue, not Kubernetes
        assert "Revenue" in shard_rev.relevant_context_slice
        assert "Kubernetes" not in shard_rev.relevant_context_slice

        # Engineering shard should mention Kubernetes, not revenue
        assert "Kubernetes" in shard_eng.relevant_context_slice
        assert "Revenue" not in shard_eng.relevant_context_slice


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
