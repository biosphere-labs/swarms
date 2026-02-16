"""
Tests for the Fact-Check Debate Module

Verifies that the multi-round fact-checking discussion works correctly.
"""

import pytest
from unittest.mock import patch, MagicMock

from swarms.structs.fact_check_debate import (
    FactCheckDebate,
    RESEARCHER_SYSTEM_PROMPT,
    FACT_CHECKER_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT,
)


class TestFactCheckDebatePrompts:
    """Test that the debate prompts are well-defined."""

    def test_researcher_prompt_exists(self):
        assert RESEARCHER_SYSTEM_PROMPT
        assert "research" in RESEARCHER_SYSTEM_PROMPT.lower()
        assert "findings" in RESEARCHER_SYSTEM_PROMPT.lower()

    def test_fact_checker_prompt_exists(self):
        assert FACT_CHECKER_SYSTEM_PROMPT
        assert "verify" in FACT_CHECKER_SYSTEM_PROMPT.lower()
        assert "fact" in FACT_CHECKER_SYSTEM_PROMPT.lower()

    def test_judge_prompt_exists(self):
        assert JUDGE_SYSTEM_PROMPT
        assert "judge" in JUDGE_SYSTEM_PROMPT.lower()
        assert "synthesize" in JUDGE_SYSTEM_PROMPT.lower()


class TestFactCheckDebate:
    """Test the FactCheckDebate class."""

    def test_init_default(self):
        debate = FactCheckDebate()
        assert debate.model_name == "gpt-4o-mini"
        assert debate.max_loops == 2
        assert debate.tools == []
        assert debate.verbose is False

    def test_init_custom(self):
        mock_tool = MagicMock()
        mock_provider = MagicMock()
        debate = FactCheckDebate(
            model_name="test-model",
            max_loops=3,
            tools=[mock_tool],
            api_key_provider=mock_provider,
            verbose=True,
        )
        assert debate.model_name == "test-model"
        assert debate.max_loops == 3
        assert debate.tools == [mock_tool]
        assert debate._api_key_provider == mock_provider
        assert debate.verbose is True

    @patch("swarms.structs.fact_check_debate.Agent")
    @patch("swarms.structs.fact_check_debate.DebateWithJudge")
    def test_verify_creates_agents(self, mock_debate_class, mock_agent_class):
        """Test that verify() creates researcher, fact-checker, and judge agents."""
        mock_debate = MagicMock()
        mock_debate.run.return_value = "verified output"
        mock_debate_class.return_value = mock_debate

        debate = FactCheckDebate(model_name="test-model")
        result = debate.verify("test research output")

        # Should create 3 agents
        assert mock_agent_class.call_count == 3

        # Check agent names
        calls = mock_agent_class.call_args_list
        agent_names = [call.kwargs["agent_name"] for call in calls]
        assert "Researcher" in agent_names
        assert "FactChecker" in agent_names
        assert "Judge" in agent_names

        # Check system prompts
        prompts = [call.kwargs["system_prompt"] for call in calls]
        assert RESEARCHER_SYSTEM_PROMPT in prompts
        assert FACT_CHECKER_SYSTEM_PROMPT in prompts
        assert JUDGE_SYSTEM_PROMPT in prompts

        # Check model name
        for call in calls:
            assert call.kwargs["model_name"] == "test-model"

    @patch("swarms.structs.fact_check_debate.Agent")
    @patch("swarms.structs.fact_check_debate.DebateWithJudge")
    def test_verify_passes_tools_to_agents(self, mock_debate_class, mock_agent_class):
        """Test that tools are passed to researcher and fact-checker."""
        mock_debate = MagicMock()
        mock_debate.run.return_value = "verified"
        mock_debate_class.return_value = mock_debate

        mock_tool = MagicMock()
        debate = FactCheckDebate(tools=[mock_tool])
        debate.verify("test output")

        calls = mock_agent_class.call_args_list
        researcher_call = calls[0]
        fact_checker_call = calls[1]
        judge_call = calls[2]

        # Researcher and fact-checker get tools
        assert researcher_call.kwargs["tools"] == [mock_tool]
        assert fact_checker_call.kwargs["tools"] == [mock_tool]
        # Judge doesn't need tools
        assert judge_call.kwargs["tools"] is None

    @patch("swarms.structs.fact_check_debate.Agent")
    @patch("swarms.structs.fact_check_debate.DebateWithJudge")
    def test_verify_uses_api_key_provider(self, mock_debate_class, mock_agent_class):
        """Test that API key provider is called for each agent."""
        mock_debate = MagicMock()
        mock_debate.run.return_value = "verified"
        mock_debate_class.return_value = mock_debate

        mock_provider = MagicMock()
        mock_provider.side_effect = ["key1", "key2", "key3"]

        debate = FactCheckDebate(api_key_provider=mock_provider)
        debate.verify("test output")

        # Should call provider 3 times (once per agent)
        assert mock_provider.call_count == 3

        # Check that keys were passed to agents
        calls = mock_agent_class.call_args_list
        assert calls[0].kwargs["llm_api_key"] == "key1"
        assert calls[1].kwargs["llm_api_key"] == "key2"
        assert calls[2].kwargs["llm_api_key"] == "key3"

    @patch("swarms.structs.fact_check_debate.Agent")
    @patch("swarms.structs.fact_check_debate.DebateWithJudge")
    def test_verify_creates_debate_with_agents(self, mock_debate_class, mock_agent_class):
        """Test that DebateWithJudge is created with the three agents."""
        mock_debate = MagicMock()
        mock_debate.run.return_value = "verified"
        mock_debate_class.return_value = mock_debate

        # Mock agents
        mock_researcher = MagicMock()
        mock_fact_checker = MagicMock()
        mock_judge = MagicMock()
        mock_agent_class.side_effect = [mock_researcher, mock_fact_checker, mock_judge]

        debate = FactCheckDebate(max_loops=3)
        debate.verify("test output")

        # Check DebateWithJudge was created correctly
        mock_debate_class.assert_called_once_with(
            pro_agent=mock_researcher,
            con_agent=mock_fact_checker,
            judge_agent=mock_judge,
            max_loops=3,
            verbose=False,
        )

    @patch("swarms.structs.fact_check_debate.Agent")
    @patch("swarms.structs.fact_check_debate.DebateWithJudge")
    def test_verify_runs_debate_with_correct_task(self, mock_debate_class, mock_agent_class):
        """Test that the debate is run with a properly formatted task."""
        mock_debate = MagicMock()
        mock_debate.run.return_value = "verified output"
        mock_debate_class.return_value = mock_debate

        debate = FactCheckDebate()
        result = debate.verify("Acme Corp raised $50M in 2023")

        # Check that run was called
        mock_debate.run.assert_called_once()
        task = mock_debate.run.call_args.kwargs["task"]

        # Task should include the research output
        assert "Acme Corp raised $50M in 2023" in task
        assert "verify" in task.lower()

    @patch("swarms.structs.fact_check_debate.Agent")
    @patch("swarms.structs.fact_check_debate.DebateWithJudge")
    def test_verify_returns_debate_result(self, mock_debate_class, mock_agent_class):
        """Test that verify returns the debate result."""
        mock_debate = MagicMock()
        mock_debate.run.return_value = "This is the verified output"
        mock_debate_class.return_value = mock_debate

        debate = FactCheckDebate()
        result = debate.verify("original output")

        assert result == "This is the verified output"

    @patch("swarms.structs.fact_check_debate.Agent")
    @patch("swarms.structs.fact_check_debate.DebateWithJudge")
    def test_verify_fallback_on_empty_result(self, mock_debate_class, mock_agent_class):
        """Test that verify returns original output if debate returns None."""
        mock_debate = MagicMock()
        mock_debate.run.return_value = None
        mock_debate_class.return_value = mock_debate

        debate = FactCheckDebate()
        result = debate.verify("original output")

        assert result == "original output"
