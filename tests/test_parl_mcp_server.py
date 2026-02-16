"""
Tests for the PARL Orchestrator MCP Server

These tests verify that the MCP server properly exposes the orchestrator
as an MCP tool and handles configuration correctly.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

# Note: We can't easily test the actual MCP server running without
# spinning up a real HTTP server, so these tests focus on the tool
# function implementations.


class TestPARLMCPServer:
    """Test the PARL MCP server tool implementations."""

    @patch("swarms.structs.parl_mcp_server.PARLOrchestrator")
    def test_parl_execute_creates_orchestrator(self, mock_orchestrator_class):
        """Test that parl_execute creates an orchestrator with correct params."""
        from swarms.structs.parl_mcp_server import parl_execute

        mock_instance = MagicMock()
        mock_instance.run.return_value = "test result"
        mock_orchestrator_class.return_value = mock_instance

        result = parl_execute(
            task="test task",
            orchestrator_model="test-model",
            fact_check=True,
            max_parallel=5,
        )

        # Verify orchestrator was created with correct params
        mock_orchestrator_class.assert_called_once()
        call_kwargs = mock_orchestrator_class.call_args.kwargs
        assert call_kwargs["orchestrator_model"] == "test-model"
        assert call_kwargs["fact_check"] is True
        assert call_kwargs["max_parallel"] == 5

        # Verify run was called with the task
        mock_instance.run.assert_called_once_with(task="test task")
        assert result == "test result"

    @patch("swarms.structs.parl_mcp_server.PARLOrchestrator")
    def test_parl_execute_uses_env_vars_when_no_overrides(self, mock_orchestrator_class):
        """Test that parl_execute respects environment variables."""
        from swarms.structs.parl_mcp_server import parl_execute

        mock_instance = MagicMock()
        mock_instance.run.return_value = "result"
        mock_orchestrator_class.return_value = mock_instance

        # Call without overrides - should use env vars via PARLOrchestrator defaults
        result = parl_execute(task="test task")

        # Orchestrator should be created with None values, letting it read env vars
        mock_orchestrator_class.assert_called_once()
        call_kwargs = mock_orchestrator_class.call_args.kwargs
        assert call_kwargs["orchestrator_model"] is None
        assert call_kwargs["sub_agent_model"] is None
        assert call_kwargs["fact_check"] is None

    def test_parl_config_returns_env_vars(self):
        """Test that parl_config returns configuration."""
        from swarms.structs.parl_mcp_server import parl_config

        config = parl_config()

        # Should contain key environment variables
        assert "PARL_ORCHESTRATOR_MODEL" in config
        assert "PARL_SUB_AGENT_MODEL" in config
        assert "PARL_FACT_CHECK" in config
        assert "PARL_MAX_PARALLEL" in config

    @patch.dict(os.environ, {"PARL_ORCHESTRATOR_MODEL": "custom-model"})
    def test_parl_config_shows_custom_env_vars(self):
        """Test that parl_config shows custom environment values."""
        from swarms.structs.parl_mcp_server import parl_config

        config = parl_config()

        assert "custom-model" in config

    @patch.dict(os.environ, {"PARL_API_KEYS": "key1,key2,key3"})
    def test_parl_config_masks_api_keys(self):
        """Test that parl_config doesn't leak API keys."""
        from swarms.structs.parl_mcp_server import parl_config

        config = parl_config()

        # Should show that keys are set but not expose them
        assert "PARL_API_KEYS: ***" in config
        assert "key1" not in config
        assert "key2" not in config

    @patch("swarms.structs.parl_mcp_server.PARLOrchestrator")
    def test_parl_execute_includes_serper_search_tool(self, mock_orchestrator_class):
        """Test that parl_execute provides serper_search to the orchestrator."""
        from swarms.structs.parl_mcp_server import parl_execute
        from swarms.tools.serper_search import serper_search

        mock_instance = MagicMock()
        mock_instance.run.return_value = "result"
        mock_orchestrator_class.return_value = mock_instance

        parl_execute(task="test task")

        # Verify tools were passed
        call_kwargs = mock_orchestrator_class.call_args.kwargs
        assert "tools" in call_kwargs
        assert serper_search in call_kwargs["tools"]


class TestMCPServerIntegration:
    """Integration tests for the MCP server (conceptual, not running actual server)."""

    def test_mcp_instance_exists(self):
        """Test that the MCP server instance is created."""
        from swarms.structs.parl_mcp_server import mcp

        assert mcp is not None
        assert mcp.name == "PARLOrchestrator"

    def test_tools_are_registered(self):
        """Test that the expected tools are registered with the MCP server."""
        from swarms.structs.parl_mcp_server import mcp

        # MCP tools are registered via decorators
        # We can't easily introspect them without running the server,
        # but we can verify the functions exist
        from swarms.structs import parl_mcp_server

        assert hasattr(parl_mcp_server, "parl_execute")
        assert hasattr(parl_mcp_server, "parl_config")
        assert callable(parl_mcp_server.parl_execute)
        assert callable(parl_mcp_server.parl_config)
