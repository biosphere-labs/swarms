"""
Tests for the LLM Backend abstraction layer.

Tests the protocol, factory function, and individual backend implementations.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from swarms.structs.llm_backend import (
    LLMBackend,
    LiteLLMBackend,
    ClaudeCodeBackend,
    create_llm_backend,
)


class TestLLMBackendProtocol:
    """Test that backends conform to the LLMBackend protocol."""

    def test_litellm_backend_is_llm_backend(self):
        backend = LiteLLMBackend(model="gpt-4o-mini")
        assert isinstance(backend, LLMBackend)

    def test_claude_code_backend_is_llm_backend(self):
        backend = ClaudeCodeBackend(timeout=60)
        assert isinstance(backend, LLMBackend)


class TestFactory:
    """Test the create_llm_backend factory function."""

    def test_create_litellm_backend(self):
        backend = create_llm_backend("litellm", model="gpt-4o-mini")
        assert isinstance(backend, LiteLLMBackend)
        assert backend.model == "gpt-4o-mini"

    def test_create_claude_code_backend(self):
        backend = create_llm_backend("claude-code", timeout=120)
        assert isinstance(backend, ClaudeCodeBackend)
        assert backend.timeout == 120

    def test_default_is_litellm(self):
        backend = create_llm_backend()
        assert isinstance(backend, LiteLLMBackend)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend type"):
            create_llm_backend("unknown-backend")


class TestLiteLLMBackend:
    """Test LiteLLMBackend wraps litellm.completion correctly."""

    def _make_mock_litellm(self):
        mock_litellm = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "  hello world  "
        mock_litellm.completion.return_value = mock_response
        return mock_litellm

    @patch.dict("sys.modules", {"litellm": MagicMock()})
    def test_call_delegates_to_litellm(self):
        import sys
        mock_litellm = sys.modules["litellm"]
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "  hello world  "
        mock_litellm.completion.return_value = mock_response

        backend = LiteLLMBackend(model="test-model")
        result = backend.call(
            system_prompt="sys",
            user_prompt="user",
            temperature=0.5,
            max_tokens=100,
        )

        assert result == "hello world"
        mock_litellm.completion.assert_called_once_with(
            model="test-model",
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "user"},
            ],
            temperature=0.5,
            max_tokens=100,
        )

    @patch.dict("sys.modules", {"litellm": MagicMock()})
    def test_call_passes_api_key(self):
        import sys
        mock_litellm = sys.modules["litellm"]
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        mock_litellm.completion.return_value = mock_response

        backend = LiteLLMBackend(model="test-model")
        backend.call(
            system_prompt="sys",
            user_prompt="user",
            api_key="sk-test-key",
        )

        call_kwargs = mock_litellm.completion.call_args
        assert call_kwargs.kwargs.get("api_key") == "sk-test-key"

    @patch.dict("sys.modules", {"litellm": MagicMock()})
    def test_call_no_api_key_when_none(self):
        import sys
        mock_litellm = sys.modules["litellm"]
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        mock_litellm.completion.return_value = mock_response

        backend = LiteLLMBackend(model="test-model")
        backend.call(system_prompt="sys", user_prompt="user")

        call_kwargs = mock_litellm.completion.call_args
        assert "api_key" not in call_kwargs.kwargs


class TestClaudeCodeBackend:
    """Test ClaudeCodeBackend subprocess management."""

    @patch("subprocess.Popen")
    def test_call_spawns_claude_process(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (b"response text", b"")
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc

        backend = ClaudeCodeBackend(timeout=60)
        result = backend.call(system_prompt="sys", user_prompt="user")

        assert result == "response text"
        # Verify claude was called with correct args
        call_args = mock_popen.call_args
        assert call_args[0][0] == ["claude", "-p", "--output-format", "text"]
        # Verify ANTHROPIC_API_KEY is cleared
        env = call_args[1]["env"]
        assert env["ANTHROPIC_API_KEY"] == ""

    @patch("subprocess.Popen")
    def test_call_raises_on_nonzero_exit(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (b"", b"error message")
        mock_proc.returncode = 1
        mock_popen.return_value = mock_proc

        backend = ClaudeCodeBackend(timeout=60)
        with pytest.raises(RuntimeError, match="claude -p exited with code 1"):
            backend.call(system_prompt="sys", user_prompt="user")

    @patch("subprocess.Popen")
    def test_call_raises_on_timeout(self, mock_popen):
        import subprocess as sp

        mock_proc = MagicMock()
        mock_proc.communicate.side_effect = sp.TimeoutExpired(cmd="claude", timeout=10)
        mock_proc.kill = MagicMock()
        mock_proc.wait = MagicMock()
        mock_popen.return_value = mock_proc

        backend = ClaudeCodeBackend(timeout=10)
        with pytest.raises(TimeoutError, match="timed out after 10s"):
            backend.call(system_prompt="sys", user_prompt="user")
        mock_proc.kill.assert_called_once()


class TestPARLOrchestratorBackendIntegration:
    """Test that PARLOrchestrator correctly wires up backends."""

    @patch("swarms.structs.parl_orchestrator.create_llm_backend")
    def test_default_backends_are_litellm(self, mock_factory):
        mock_factory.return_value = MagicMock()
        from swarms.structs.parl_orchestrator import PARLOrchestrator

        orch = PARLOrchestrator()
        # Factory should be called twice: once for decomposition, once for synthesis
        assert mock_factory.call_count == 2
        calls = mock_factory.call_args_list
        assert calls[0].kwargs["backend_type"] == "litellm"
        assert calls[1].kwargs["backend_type"] == "litellm"

    @patch("swarms.structs.parl_orchestrator.create_llm_backend")
    def test_claude_code_backend_from_param(self, mock_factory):
        mock_factory.return_value = MagicMock()
        from swarms.structs.parl_orchestrator import PARLOrchestrator

        orch = PARLOrchestrator(decomposition_backend="claude-code")
        calls = mock_factory.call_args_list
        assert calls[0].kwargs["backend_type"] == "claude-code"
        assert calls[1].kwargs["backend_type"] == "litellm"

    @patch.dict(os.environ, {"PARL_DECOMPOSITION_BACKEND": "claude-code"})
    @patch("swarms.structs.parl_orchestrator.create_llm_backend")
    def test_claude_code_backend_from_env(self, mock_factory):
        mock_factory.return_value = MagicMock()
        from swarms.structs.parl_orchestrator import PARLOrchestrator

        orch = PARLOrchestrator()
        calls = mock_factory.call_args_list
        assert calls[0].kwargs["backend_type"] == "claude-code"

    def test_fact_check_defaults_to_false(self):
        from swarms.structs.parl_orchestrator import PARLOrchestrator

        orch = PARLOrchestrator()
        assert orch.fact_check is False

    @patch.dict(os.environ, {"PARL_FACT_CHECK": "true"})
    def test_fact_check_from_env(self):
        from swarms.structs.parl_orchestrator import PARLOrchestrator

        orch = PARLOrchestrator()
        assert orch.fact_check is True

    def test_fact_check_from_param(self):
        from swarms.structs.parl_orchestrator import PARLOrchestrator

        orch = PARLOrchestrator(fact_check=True)
        assert orch.fact_check is True

    def test_fact_check_model_defaults_to_sub_agent_model(self):
        from swarms.structs.parl_orchestrator import PARLOrchestrator

        orch = PARLOrchestrator(sub_agent_model="test-model")
        assert orch.fact_check_model == "test-model"
