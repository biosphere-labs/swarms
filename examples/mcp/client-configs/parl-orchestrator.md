# PARL Orchestrator MCP Client Configuration

Example configurations for connecting to the PARL Orchestrator MCP server from various AI coding tools.

## Starting the Server

```bash
# Default (uses gpt-4o-mini for all components):
python examples/mcp/servers/parl_server.py

# With DeepInfra models (recommended for cost/quality):
PARL_ORCHESTRATOR_MODEL=deepinfra/Qwen/Qwen3-235B-A22B \
PARL_SUB_AGENT_MODEL=deepinfra/Qwen/Qwen2.5-72B-Instruct \
PARL_SYNTHESIS_MODEL=deepinfra/deepseek-ai/DeepSeek-V3.2 \
PARL_FACT_CHECK=true \
python examples/mcp/servers/parl_server.py

# With Claude subscription for decomposition:
PARL_DECOMPOSITION_BACKEND=claude-code \
python examples/mcp/servers/parl_server.py

# Custom port:
MCP_PORT=9000 python examples/mcp/servers/parl_server.py
```

## Client Configurations

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "parl-orchestrator": {
      "url": "http://localhost:8765/mcp",
      "transport": "streamable-http"
    }
  }
}
```

### Cursor

Add to `.cursor/mcp.json` in your project:

```json
{
  "mcpServers": {
    "parl-orchestrator": {
      "url": "http://localhost:8765/mcp",
      "transport": "streamable-http"
    }
  }
}
```

Or use the global config at `~/.cursor/mcp.json`.

### Windsurf (Codeium)

Add to Windsurf settings (MCP section):

```json
{
  "parl-orchestrator": {
    "url": "http://localhost:8765/mcp",
    "transport": "streamable-http"
  }
}
```

### Claude Code (MCP Client)

Use the existing swarms MCP client tools:

```python
from swarms.tools.mcp_client_tools import get_mcp_tools_sync, execute_tool_call_simple
from swarms.schemas.mcp_schemas import MCPConnection

# Connect to the PARL server
connection = MCPConnection(
    url="http://localhost:8765",
    transport="streamable-http",
)

# Get available tools
tools = get_mcp_tools_sync([connection])
print(tools)  # ['parl_execute', 'parl_config']

# Execute a research task
result = execute_tool_call_simple(
    tool_name="parl_execute",
    args={
        "task": "Research competitor Acme Corp across funding, team, pricing, and reviews",
        "fact_check": True,
    },
    mcp_connection=connection,
)

print(result)
```

## Available Tools

### `parl_execute`

Execute a complex multi-faceted task using the PARL orchestrator.

**Parameters:**
- `task` (required): The task to execute. Should be multi-faceted with independent aspects.
- `orchestrator_model` (optional): Override the decomposition model
- `sub_agent_model` (optional): Override the sub-agent model
- `synthesis_model` (optional): Override the synthesis model
- `fact_check` (optional): Enable/disable fact-checking (boolean)
- `max_parallel` (optional): Max concurrent sub-agents (int)
- `max_iterations` (optional): Gap-fill iterations (int)
- `timeout` (optional): Overall timeout in seconds (int)

**Example usage in Cursor/Claude Desktop:**

> Use the parl_execute tool to research competitor Acme Corp across funding, team, pricing, and reviews with fact-checking enabled.

**Example direct call:**

```json
{
  "tool": "parl_execute",
  "arguments": {
    "task": "Research competitor Acme Corp across funding, team, pricing, and reviews",
    "fact_check": true,
    "max_parallel": 8
  }
}
```

### `parl_config`

Get the current server configuration (models, backends, settings).

**Parameters:** None

**Returns:** Formatted string showing all environment variables.

## Common Use Cases

### 1. Multi-Faceted Research

**Good for:**
- "Research [company] across funding, team, pricing, and reviews"
- "Compare [A, B, C] across performance, ecosystem, and cost"
- "Investigate [topic] approaches, tools, costs, and best practices"

**Not good for:**
- Single-topic questions ("What is X?")
- Code generation
- Simple lookups

### 2. Competitive Analysis

```
Task: "Deep research on competitor TechCorp: funding history, key products, market positioning, team size and expertise, pricing strategy, and customer reviews"
```

The orchestrator will decompose this into 6 parallel sub-tasks, execute them concurrently with fact-checking (if enabled), and synthesize a comprehensive report with contradiction detection.

### 3. Technology Evaluation

```
Task: "Compare React, Vue, and Svelte across: performance benchmarks, ecosystem size and maturity, developer experience, hiring market, and long-term viability"
```

## Troubleshooting

### Server won't start

- Check if port 8765 is already in use: `lsof -i :8765`
- Try a different port: `MCP_PORT=9000 python examples/mcp/servers/parl_server.py`

### Client can't connect

- Ensure the server is running: check the terminal where you started it
- Verify the URL in your client config: `http://localhost:8765` (not `https`)
- Try `curl http://localhost:8765` to test connectivity

### API key errors

- If using DeepInfra: Set `PARL_API_KEYS` with comma-separated keys
- If using Claude subscription: Set `PARL_DECOMPOSITION_BACKEND=claude-code`
- Check that API keys are valid: `curl -H "Authorization: Bearer $KEY" https://api.deepinfra.com/models/list`

### Rate limits

- Use multiple API keys: `PARL_API_KEYS=key1,key2,key3` (round-robin rotation)
- Reduce parallelism: `PARL_MAX_PARALLEL=5`
- Increase timeouts: `PARL_TIMEOUT=600`

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `PARL_ORCHESTRATOR_MODEL` | `gpt-4o-mini` | Model for decomposition |
| `PARL_SUB_AGENT_MODEL` | `gpt-4o-mini` | Model for sub-agents |
| `PARL_SYNTHESIS_MODEL` | (same as orchestrator) | Model for synthesis |
| `PARL_DECOMPOSITION_BACKEND` | `litellm` | Backend: `litellm` or `claude-code` |
| `PARL_SYNTHESIS_BACKEND` | `litellm` | Backend: `litellm` or `claude-code` |
| `PARL_FACT_CHECK` | `false` | Enable fact-checking: `true` or `false` |
| `PARL_FACT_CHECK_MODEL` | (same as sub-agent) | Model for fact-checkers |
| `PARL_MAX_PARALLEL` | `10` | Max concurrent sub-agents |
| `PARL_MAX_ITERATIONS` | `2` | Gap-fill iterations |
| `PARL_TIMEOUT` | `300` | Overall timeout (seconds) |
| `PARL_SUB_AGENT_TIMEOUT` | `120` | Per sub-agent timeout (seconds) |
| `PARL_TOKEN_BUDGET` | `100000` | Total token budget |
| `PARL_API_KEYS` | (none) | Comma-separated keys for round-robin |
| `MCP_PORT` | `8765` | Server port |

## Cost Optimization

Using DeepInfra with the recommended models:

| Component | Model | Cost/M tokens (in/out) |
|-----------|-------|------------------------|
| Decomposition | Qwen3-235B | $0.20/$0.60 |
| Sub-agents | Qwen2.5-72B | $0.13/$0.13 |
| Synthesis | DeepSeek-V3.2 | $0.26/$0.38 |

**Estimated cost per research run:** ~$0.05-0.15 depending on complexity.

For higher quality decomposition at no extra cost, use Claude subscription:
```bash
PARL_DECOMPOSITION_BACKEND=claude-code \
PARL_ORCHESTRATOR_MODEL=claude-sonnet-4-5 \
python examples/mcp/servers/parl_server.py
```
