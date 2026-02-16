# PARL Orchestrator MCP Server — Quick Reference

The PARL orchestrator runs as an MCP server with three tools: `parl_execute` (auto-decomposed research), `parl_review` (multi-persona document review), and `parl_config` (configuration).

> **Full documentation:** [`docs/swarms/structs/parl_orchestrator_mcp.md`](docs/swarms/structs/parl_orchestrator_mcp.md)

## Quick Start

### 1. Start the Server

```bash
# Auto-start on login (recommended)
./scripts/install-parl-service.sh

# Or manually:
PARL_ORCHESTRATOR_MODEL=deepinfra/Qwen/Qwen3-235B-A22B \
PARL_SUB_AGENT_MODEL=deepinfra/Qwen/Qwen2.5-72B-Instruct \
PARL_SYNTHESIS_MODEL=deepinfra/deepseek-ai/DeepSeek-V3.2 \
PARL_API_KEYS=key1,key2,key3 \
python examples/mcp/servers/parl_server.py
```

### 2. Connect Claude Code

```bash
# Run OUTSIDE of Claude Code
claude mcp add --transport http parl-orchestrator http://localhost:8765/mcp

# Verify
claude mcp list
```

### 3. Use

**Research mode:**
```
Use parl_execute to research OpenAI across funding, team, products, and pricing.
```

**Review mode:**
```
Use parl_review to review this pitch deck with these personas:
[{"name": "Market Analyst", "instruction": "Evaluate market sizing", "model": "deepinfra/Qwen/Qwen3-235B-A22B"},
 {"name": "Financial Critic", "instruction": "Scrutinize projections", "model": "deepinfra/deepseek-ai/DeepSeek-V3.2"}]
```

## Monitoring

```bash
systemctl --user status parl-mcp-server     # Service status
journalctl --user -u parl-mcp-server -f     # Live logs
curl -s -o /dev/null -w "%{http_code}" http://localhost:8765/mcp  # Health (expect 406)
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `✗ Failed to connect` | URL must be `http://localhost:8765/mcp` (with `/mcp`) |
| `AuthenticationError: OpenAIException` | Set `PARL_API_KEYS` + use `deepinfra/` model prefix |
| `Current Internal Reasoning Loop: 1/5` | API key or model config error — check logs |
| Service keeps restarting | `journalctl --user -u parl-mcp-server -n 50` for traceback |
