#!/usr/bin/env python3
"""
PARL Orchestrator MCP Server Launcher

This is a convenience wrapper around swarms.structs.parl_mcp_server.
Run this to start the PARL orchestrator as an MCP server.

Usage:
    # Default configuration:
    python examples/mcp/servers/parl_server.py

    # With DeepInfra models + fact-checking:
    PARL_ORCHESTRATOR_MODEL=deepinfra/Qwen/Qwen3-235B-A22B \
    PARL_SUB_AGENT_MODEL=deepinfra/Qwen/Qwen2.5-72B-Instruct \
    PARL_SYNTHESIS_MODEL=deepinfra/deepseek-ai/DeepSeek-V3.2 \
    PARL_FACT_CHECK=true \
    python examples/mcp/servers/parl_server.py

    # With Claude subscription for decomposition:
    PARL_DECOMPOSITION_BACKEND=claude-code \
    PARL_ORCHESTRATOR_MODEL=claude-sonnet-4-5 \
    python examples/mcp/servers/parl_server.py
"""

if __name__ == "__main__":
    import os

    # Map DEEPINFRA_API_KEYS -> PARL_API_KEYS if not already set
    if not os.environ.get("PARL_API_KEYS") and os.environ.get("DEEPINFRA_API_KEYS"):
        os.environ["PARL_API_KEYS"] = os.environ["DEEPINFRA_API_KEYS"]

    from swarms.structs.parl_mcp_server import mcp

    port = int(os.environ.get("MCP_PORT", "8765"))

    print("=" * 60)
    print("PARL Orchestrator MCP Server")
    print("=" * 60)
    print(f"\n🚀 Starting on port {port}...")
    print("\nConfiguration:")
    print(f"  Orchestrator: {os.environ.get('PARL_ORCHESTRATOR_MODEL', 'gpt-4o-mini')}")
    print(f"  Sub-agents:   {os.environ.get('PARL_SUB_AGENT_MODEL', 'gpt-4o-mini')}")
    print(f"  Synthesis:    {os.environ.get('PARL_SYNTHESIS_MODEL', '(same as orchestrator)')}")
    print(f"  Fact-check:   {os.environ.get('PARL_FACT_CHECK', 'false')}")
    print(f"  Backend:      {os.environ.get('PARL_DECOMPOSITION_BACKEND', 'litellm')}")
    print("\nConnect MCP clients to:")
    print(f"  http://localhost:{port}")
    print("\nTools available:")
    print("  - parl_execute: Run tasks with orchestration")
    print("  - parl_config:  View server configuration")
    print("\nPress Ctrl+C to stop.\n")

    # Configure FastMCP settings before running
    mcp.settings.port = port
    mcp.settings.host = "127.0.0.1"

    # Run MCP server directly with streamable-http transport
    mcp.run(transport="streamable-http", mount_path="/mcp")
