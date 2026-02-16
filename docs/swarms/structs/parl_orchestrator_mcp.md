# PARL Orchestrator MCP Server

The PARL orchestrator exposes four tools as MCP endpoints:

- **`parl_execute`** — Auto-decomposes a research task into parallel sub-agents, synthesizes results
- **`parl_review`** — Runs a document through a panel of custom reviewer personas in parallel, synthesizes a multi-angle assessment
- **`parl_smart_review`** — Enhanced review with automatic model diversity, cross-model fact-checking, and blind spot analysis
- **`parl_config`** — Returns current server configuration

All tools run as a single MCP service with **async progress streaming** — clients receive real-time status updates as each sub-agent completes, keeping long-running operations visible.

---

## Architecture

```
MCP Client (Claude Code, Cursor, etc.)
    │
    ▼
PARL MCP Server (http://localhost:8765/mcp)
    │
    ├── parl_execute
    │     ├── DecompositionEngine → splits task into sub-tasks
    │     ├── CriticalPathScheduler → orders execution
    │     ├── N × Agent (parallel, isolated context)
    │     ├── Optional: FactCheckDebate per agent
    │     └── ResultAggregator → synthesis + contradiction detection
    │
    ├── parl_review
    │     ├── N × Reviewer Agent (one per persona, parallel)
    │     │     Each with: custom model, web search, persona instructions
    │     ├── Optional: FactCheckDebate per reviewer
    │     └── ResultAggregator → synthesis + disagreement surfacing
    │
    ├── parl_smart_review
    │     ├── Auto model assignment from DeepInfra catalog
    │     ├── N × Reviewer Agent (diverse models auto-selected)
    │     ├── Iterative cross-model fact-checking (claims verified by different model)
    │     ├── Blind spot analysis (issues caught by only one model)
    │     └── ResultAggregator → synthesis + model agreement matrix
    │
    └── parl_config
          └── Returns current environment configuration

All async tools report progress via MCP Context:
    ctx.report_progress(current, total)  → numeric progress bar
    ctx.info("message")                  → human-readable status updates
```

---

## Installation

### Prerequisites

```bash
# From the swarms repo root
pip install -e .          # Install swarms + dependencies
pip install mcp uvicorn   # MCP server framework
```

### Quick Start (Manual)

```bash
cd /path/to/swarms

# Set API keys
export PARL_API_KEYS="key1,key2,key3"   # DeepInfra keys (comma-separated)

# Set models
export PARL_ORCHESTRATOR_MODEL="deepinfra/Qwen/Qwen3-235B-A22B"
export PARL_SUB_AGENT_MODEL="deepinfra/Qwen/Qwen2.5-72B-Instruct"
export PARL_SYNTHESIS_MODEL="deepinfra/deepseek-ai/DeepSeek-V3.2"

# Start
python examples/mcp/servers/parl_server.py
```

### Auto-Start via systemd (Linux)

```bash
# Install the service (auto-starts on login)
./scripts/install-parl-service.sh

# Check status
systemctl --user status parl-mcp-server

# View logs
journalctl --user -u parl-mcp-server -f

# Restart after config changes
systemctl --user restart parl-mcp-server

# Remove service
./scripts/uninstall-parl-service.sh
```

The systemd service file is at `~/.config/systemd/user/parl-mcp-server.service`. Edit it to change models or environment variables, then run `systemctl --user daemon-reload && systemctl --user restart parl-mcp-server`.

---

## Connecting MCP Clients

### Claude Code

```bash
# Register the server (run outside of Claude Code)
claude mcp add --transport http parl-orchestrator http://localhost:8765/mcp

# Verify
claude mcp list
# Should show: parl-orchestrator: http://localhost:8765/mcp (HTTP) - ✓ Connected

# Inside Claude Code, check with:
/mcp
```

### Cursor

Add to `.cursor/mcp.json`:

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

### Windsurf / Claude Desktop

Same JSON format as Cursor, placed in the tool's MCP configuration file.

> **Important:** The endpoint is `/mcp`, not the root URL. Always use `http://localhost:8765/mcp`.

---

## Tool Reference

### `parl_execute`

Auto-decomposes a complex task into parallel sub-agents, executes them concurrently, and synthesizes results with contradiction detection.

**Best for:** Multi-faceted research, competitive analysis, technology evaluation — any task with independent aspects that benefit from parallel investigation.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `task` | string | Yes | — | The research task to execute |
| `orchestrator_model` | string | No | env `PARL_ORCHESTRATOR_MODEL` | Model for task decomposition |
| `sub_agent_model` | string | No | env `PARL_SUB_AGENT_MODEL` | Model for sub-agents |
| `synthesis_model` | string | No | env `PARL_SYNTHESIS_MODEL` | Model for result synthesis |
| `fact_check` | boolean | No | env `PARL_FACT_CHECK` | Enable per-agent fact-check debate |
| `max_parallel` | integer | No | env `PARL_MAX_PARALLEL` (10) | Max concurrent sub-agents |
| `max_iterations` | integer | No | env `PARL_MAX_ITERATIONS` (2) | Gap-fill re-decomposition rounds |
| `timeout` | integer | No | env `PARL_TIMEOUT` (300) | Overall timeout in seconds |

#### Example

Prompt in Claude Code:

```
Use parl_execute to research OpenAI across: funding history, current valuation,
key products, team leadership, competitive positioning, and developer sentiment.
Enable fact_check.
```

Direct JSON call:

```json
{
  "tool": "parl_execute",
  "arguments": {
    "task": "Research OpenAI across funding history, current valuation, key products, team leadership, competitive positioning, and developer sentiment",
    "fact_check": true,
    "max_parallel": 8,
    "timeout": 180
  }
}
```

#### Pipeline

1. **Decomposition** — Strong model (Qwen3-235B) analyzes the task and produces a sub-task graph
2. **Scheduling** — Critical-path-aware scheduler orders tasks into parallel cohorts
3. **Execution** — Each sub-agent gets isolated context + web search tools, runs in parallel
4. **Fact-checking** (optional) — Each sub-agent's output goes through a 3-agent debate (Researcher vs Fact-Checker vs Judge)
5. **Aggregation** — Synthesis model merges all outputs, flags contradictions and gaps
6. **Gap-fill** (optional) — If gaps detected, re-decomposes focusing on missing areas

---

### `parl_review`

Runs a document through a panel of custom reviewer personas in parallel, then synthesizes all perspectives into a unified assessment.

**Best for:** Reviewing marketing copy, investor pitches, technical proposals, research papers — any document that benefits from multiple expert perspectives.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `document` | string | Yes | — | Full document text to review |
| `personas` | string | Yes | — | JSON array of reviewer persona objects (see below) |
| `synthesis_prompt` | string | No | Auto-generated | Custom instructions for the synthesis step |
| `fact_check` | boolean | No | env `PARL_FACT_CHECK` | Enable per-reviewer fact-check debate |
| `timeout` | integer | No | env `PARL_TIMEOUT` (300) | Overall timeout in seconds |

#### Persona Object Schema

Each persona in the JSON array must have:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | string | Yes | — | Short name for the reviewer |
| `instruction` | string | Yes | — | What this reviewer should focus on |
| `model` | string | No | env `PARL_SUB_AGENT_MODEL` | LLM model for this specific reviewer |
| `tools` | boolean | No | `true` | Whether this reviewer gets web search access |

#### Why Per-Persona Models Matter

When all reviewers use the same model, they share the same training biases and blind spots. Spreading personas across different models produces genuinely diverse perspectives:

```json
[
  {"name": "Market Analyst", "instruction": "...", "model": "deepinfra/Qwen/Qwen3-235B-A22B"},
  {"name": "Financial Critic", "instruction": "...", "model": "deepinfra/deepseek-ai/DeepSeek-V3.2"},
  {"name": "Technical Reviewer", "instruction": "...", "model": "deepinfra/Qwen/Qwen2.5-72B-Instruct"},
  {"name": "Skeptic", "instruction": "...", "model": "deepinfra/meta-llama/Llama-4-Maverick-17B-128E-Instruct"}
]
```

#### Example: Investor Pitch Review

Prompt in Claude Code:

```
Use parl_review to review this investor pitch with 5 expert personas:

Document: [paste full document]

Personas:
[
  {"name": "Market Analyst", "instruction": "Evaluate market sizing claims, TAM/SAM/SOM methodology, and competitive positioning against named competitors", "model": "deepinfra/Qwen/Qwen3-235B-A22B"},
  {"name": "Financial Reviewer", "instruction": "Scrutinize financial projections, unit economics, burn rate, and runway assumptions. Flag unrealistic growth curves", "model": "deepinfra/deepseek-ai/DeepSeek-V3.2"},
  {"name": "Technical Due Diligence", "instruction": "Assess technical architecture claims, feasibility of the roadmap, and whether the tech stack matches the problem", "model": "deepinfra/Qwen/Qwen2.5-72B-Instruct"},
  {"name": "Go-to-Market Critic", "instruction": "Challenge the GTM strategy, distribution channels, pricing model, and customer acquisition cost assumptions", "model": "deepinfra/meta-llama/Llama-4-Maverick-17B-128E-Instruct"},
  {"name": "Investor Skeptic", "instruction": "Find the weakest claims, biggest risks, and most likely failure modes. What would make you NOT invest?", "model": "deepinfra/Qwen/Qwen3-235B-A22B"}
]
```

#### Example: Marketing Copy Review

```json
{
  "tool": "parl_review",
  "arguments": {
    "document": "Your marketing copy here...",
    "personas": "[{\"name\": \"Target Customer\", \"instruction\": \"Does this speak to my actual pain points? Would I click through? Is the language authentic or corporate?\", \"model\": \"deepinfra/deepseek-ai/DeepSeek-V3.2\"}, {\"name\": \"Copywriting Expert\", \"instruction\": \"Evaluate headline strength, CTA clarity, emotional hooks, and readability. Suggest specific improvements\", \"model\": \"deepinfra/Qwen/Qwen3-235B-A22B\"}, {\"name\": \"SEO Analyst\", \"instruction\": \"Check keyword density, search intent alignment, meta description potential, and content structure for featured snippets\"}, {\"name\": \"Competitor Researcher\", \"instruction\": \"Search for competitor messaging and positioning. How does this differentiate? Where does it fall flat?\", \"model\": \"deepinfra/Qwen/Qwen2.5-72B-Instruct\"}]",
    "fact_check": false,
    "timeout": 240
  }
}
```

#### Pipeline

1. **Parse personas** — Validate JSON, extract names, instructions, models
2. **Spawn reviewers** — One Agent per persona, each with their own model and system prompt
3. **Parallel execution** — All reviewers run concurrently with web search tools
4. **Fact-checking** (optional) — Each reviewer's output goes through debate verification
5. **Synthesis** — Strong model merges all reviews, surfaces agreements, disagreements, and critical findings

#### Output Structure

The synthesized output includes:

- **Synthesized assessment** — Key findings organized by topic (not per reviewer)
- **Reviewer Disagreements** — Where personas reached different conclusions
- **Coverage Gaps** — Aspects of the document that no reviewer addressed
- **Metadata** — Elapsed time, reviewer count, models used

---

### `parl_smart_review`

Enhanced multi-model review with automatic model selection, cross-model fact-checking, and blind spot analysis. Models are automatically assigned from the DeepInfra catalog for maximum diversity unless explicitly specified per persona.

**Best for:** High-stakes reviews where you want genuinely diverse perspectives (not just different prompts on the same model), with automated verification that claims are checked by a different model than the one that made them.

**Difference from `parl_review`:** `parl_review` requires you to manually assign models per persona. `parl_smart_review` auto-assigns diverse models, adds iterative cross-model fact-checking, and includes blind spot analysis showing which issues were caught by only one model.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `document` | string | Yes | -- | Full document text to review |
| `personas` | string | Yes | -- | JSON array of reviewer personas (same schema as `parl_review`, but `model` is optional and auto-assigned) |
| `auto_assign_models` | boolean | No | `true` | Auto-assign diverse models from DeepInfra catalog |
| `fact_check_rounds` | integer | No | `2` | Cross-model fact-check rounds (0 to disable, max 3) |
| `synthesis_prompt` | string | No | Auto-generated | Custom synthesis instructions |
| `timeout` | integer | No | env `PARL_TIMEOUT` (300) | Overall timeout in seconds |

#### How Cross-Model Fact-Checking Works

1. All reviewers complete their reviews (each on a different model)
2. Claims are extracted from each review
3. Each claim is verified by a **different model** than the one that made it
4. This repeats for `fact_check_rounds` iterations
5. Claims that survive cross-model scrutiny are marked as high-confidence

#### Example

```
Use parl_smart_review to review this pitch deck. Auto-assign models for maximum diversity.

Document: [paste document]

Personas:
[
  {"name": "Thesis Stress-Tester", "instruction": "Find logical gaps in the core thesis"},
  {"name": "Market Analyst", "instruction": "Evaluate market sizing and competitive claims"},
  {"name": "Financial Reviewer", "instruction": "Scrutinize projections and unit economics"},
  {"name": "Audience Calibrator", "instruction": "Check tone and clarity for C-suite audience"}
]
```

#### Output Structure

- **Synthesized assessment** with model agreement matrix
- **Blind spot analysis** — issues flagged by only one model (potential unique insight or hallucination)
- **Fact-check corrections** — claims that failed cross-model verification
- **Reviewer disagreements** and **coverage gaps**

> **Note:** `parl_smart_review` requires the `swarms.structs.model_selector` and `swarms.structs.iterative_fact_check` modules. These are imported at call time, so the server starts without them — but the tool will fail if they don't exist.

---

### `parl_config`

Returns the current server configuration. No parameters.

```
Use parl_config to show the current PARL orchestrator settings.
```

---

## Progress Streaming

All tools are async and use FastMCP's `Context` for real-time progress reporting. This keeps MCP clients informed during long-running operations (which can take 30-120 seconds).

### What Clients Receive

| Method | Purpose | Example |
|--------|---------|---------|
| `ctx.report_progress(current, total)` | Numeric progress bar | `3/7` — 3 of 7 reviewers done |
| `ctx.info(message)` | Human-readable status | `"Reviewer 3/5 done: Market Analyst"` |

### Progress Stages by Tool

**`parl_execute`:**
1. `0/3` — Initializing orchestrator
2. `1/3` — Decomposing task and executing sub-agents
3. `3/3` — Complete

**`parl_review`:**
1. `0/N+1` — Starting N reviewers
2. `1/N+1` through `N/N+1` — Each reviewer completion reported individually
3. `N+1/N+1` — Synthesis complete

**`parl_smart_review`:**
1. `0/N+3` — Fetching model catalog
2. `1/N+3` — Executing reviewers
3. `2/N+3` through `N+1/N+3` — Each reviewer completion
4. `N+2/N+3` — Cross-model fact-checking
5. `N+3/N+3` — Synthesis complete

### Client Behavior

- **Claude Code** shows progress in tool output as the tool runs
- **Cursor/Windsurf** may show progress indicators depending on their MCP implementation
- Without progress streaming, clients may timeout or show "tool not responding" errors on long operations

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PARL_ORCHESTRATOR_MODEL` | `gpt-4o-mini` | Model for decomposition (parl_execute only) |
| `PARL_SUB_AGENT_MODEL` | `gpt-4o-mini` | Default model for sub-agents and reviewers |
| `PARL_SYNTHESIS_MODEL` | same as orchestrator | Model for result synthesis |
| `PARL_DECOMPOSITION_BACKEND` | `litellm` | Backend: `litellm` or `claude-code` |
| `PARL_SYNTHESIS_BACKEND` | `litellm` | Backend for synthesis |
| `PARL_FACT_CHECK` | `false` | Enable fact-checking by default |
| `PARL_FACT_CHECK_MODEL` | same as sub-agent | Model for fact-checker agents |
| `PARL_MAX_PARALLEL` | `10` | Max concurrent sub-agents/reviewers |
| `PARL_MAX_ITERATIONS` | `2` | Gap-fill iterations (parl_execute only) |
| `PARL_TIMEOUT` | `300` | Overall timeout in seconds |
| `PARL_SUB_AGENT_TIMEOUT` | `120` | Per sub-agent timeout |
| `PARL_TOKEN_BUDGET` | `100000` | Total token budget |
| `PARL_API_KEYS` | (none) | Comma-separated API keys for round-robin |
| `DEEPINFRA_API_KEYS` | (none) | Auto-mapped to `PARL_API_KEYS` by the launcher |
| `MCP_PORT` | `8765` | Server port |

---

## Monitoring

### Service Status

```bash
# Is it running?
systemctl --user status parl-mcp-server

# Process info
ps aux | grep parl_server

# Port check
ss -tlnp | grep 8765
```

### Live Logs

```bash
# Stream logs in real time
journalctl --user -u parl-mcp-server -f

# Last 100 lines
journalctl --user -u parl-mcp-server -n 100 --no-pager

# Logs since a specific time
journalctl --user -u parl-mcp-server --since "10 min ago"
```

### What to Look For in Logs

| Log Pattern | Meaning | Action |
|-------------|---------|--------|
| `Uvicorn running on http://127.0.0.1:8765` | Server started successfully | None — healthy |
| `AuthenticationError: OpenAIException` | Wrong model or missing API key | Check `PARL_API_KEYS` and model names |
| `litellm.AuthenticationError` | API key invalid or expired | Rotate keys, check DeepInfra dashboard |
| `DecompositionEngine...rate limit` | DeepInfra rate limit hit | Add more API keys, reduce `PARL_MAX_PARALLEL` |
| `Fact-check debate failed` | Debate agents couldn't complete | Non-fatal: unverified output used instead |
| `Overall timeout reached` | Task took too long | Increase `PARL_TIMEOUT` or reduce task scope |
| `Main process exited, code=exited, status=1/FAILURE` | Server crashed | Check journal for traceback |
| `ASGI callable returned without completing response` | Client disconnected mid-request | Normal during restarts; investigate if recurring |

### Health Check

```bash
# Quick check — 406 means server is running (MCP protocol requires proper handshake)
curl -s -o /dev/null -w "%{http_code}" http://localhost:8765/mcp
# Expected: 406

# Full check — should return JSON-RPC error about missing session
curl -s http://localhost:8765/mcp
# Expected: {"jsonrpc":"2.0","id":"server-error","error":{"code":-32600,"message":"Bad Request: Missing session ID"}}
```

### Memory and CPU

```bash
# Check resource usage
systemctl --user status parl-mcp-server | grep Memory
# Typical idle: ~200MB
# During execution: 300-500MB (depends on number of agents)

# If concerned about memory
journalctl --user -u parl-mcp-server | grep "memory peak"
```

---

## Troubleshooting

### Server won't start

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: No module named 'swarms'` | Missing PYTHONPATH | Add `Environment="PYTHONPATH=/path/to/swarms"` to service file |
| `ModuleNotFoundError: No module named 'mcp'` | MCP not installed | `pip install mcp uvicorn` |
| `Address already in use` | Port 8765 occupied | `MCP_PORT=9000` or kill the existing process |
| `TypeError: FastMCP.run() got an unexpected keyword argument` | API mismatch | Use `mcp.run(transport="streamable-http", mount_path="/mcp")` |

### Client can't connect

| Symptom | Cause | Fix |
|---------|-------|-----|
| `✗ Failed to connect` in `claude mcp list` | Wrong URL | Must be `http://localhost:8765/mcp` (with `/mcp`) |
| `✗ Failed to connect` | Server not running | `systemctl --user start parl-mcp-server` |
| Tool not visible in Claude | Config in wrong project scope | Add to global `~/.claude.json` under `mcpServers` |

### Tasks return "Current Internal Reasoning Loop: 1/5"

This means the LLM call is failing silently. Check logs:

```bash
journalctl --user -u parl-mcp-server --since "5 min ago" | grep -i "error\|fail\|auth"
```

Common causes:
- **No API key**: Set `PARL_API_KEYS` in the service file or `~/.claude/.env`
- **Wrong model name**: DeepInfra models need `deepinfra/` prefix (e.g. `deepinfra/Qwen/Qwen2.5-72B-Instruct`)
- **OpenAI default**: If models are set to `gpt-4o-mini` but no `OPENAI_API_KEY` exists

### Rate limits

DeepInfra can rate-limit during heavy parallel execution:

1. Add more API keys: `PARL_API_KEYS=key1,key2,key3,key4,key5`
2. Reduce parallelism: `PARL_MAX_PARALLEL=5`
3. The launcher auto-rotates keys round-robin across agents

### Reviews return empty or partial results

- Check that reviewer models are accessible with your API keys
- Increase timeout: `timeout: 300` in the tool call
- Check if specific models are failing: look for `Reviewer 'X' failed` in logs

---

## Cost Model

Using DeepInfra with recommended models:

| Component | Model | Cost/M tokens (in/out) |
|-----------|-------|------------------------|
| Orchestrator (decomposition) | Qwen3-235B-A22B | $0.20 / $0.60 |
| Sub-agents / Reviewers | Qwen2.5-72B-Instruct | $0.13 / $0.13 |
| Synthesis | DeepSeek-V3.2 | $0.26 / $0.38 |
| Alternative reviewer | Llama-4-Maverick-17B | $0.12 / $0.18 |

**Estimated costs:**

| Operation | Agents | Est. Cost |
|-----------|--------|-----------|
| `parl_execute` — simple research | 4-6 sub-agents | ~$0.05-0.10 |
| `parl_execute` — deep research with fact-check | 6-10 sub-agents + debate | ~$0.15-0.30 |
| `parl_review` — 5 reviewers | 5 reviewers + synthesis | ~$0.08-0.15 |
| `parl_review` — 11 reviewers with fact-check | 11 reviewers + debate + synthesis | ~$0.25-0.50 |

---

## File Locations

| File | Purpose |
|------|---------|
| `swarms/structs/parl_mcp_server.py` | MCP server with all three tools |
| `swarms/structs/parl_orchestrator.py` | Core PARL orchestration engine |
| `swarms/structs/decomposition_engine.py` | Task decomposition via LLM |
| `swarms/structs/result_aggregator.py` | Result synthesis + contradiction detection |
| `swarms/structs/fact_check_debate.py` | 3-agent debate fact-checking |
| `swarms/structs/model_selector.py` | Auto model assignment from DeepInfra catalog (used by smart_review) |
| `swarms/structs/iterative_fact_check.py` | Cross-model iterative fact-checking (used by smart_review) |
| `swarms/structs/llm_backend.py` | Pluggable LLM backend (litellm / claude-code) |
| `swarms/structs/context_sharding.py` | Per-agent context isolation |
| `swarms/structs/critical_path_scheduler.py` | Execution order optimization |
| `examples/mcp/servers/parl_server.py` | Launcher script with config display |
| `scripts/install-parl-service.sh` | systemd service installer |
| `scripts/uninstall-parl-service.sh` | systemd service removal |
| `~/.config/systemd/user/parl-mcp-server.service` | systemd service definition |
| `~/.claude/.env` | API keys (auto-loaded by service) |

### Test Files

| File | Tests | Description |
|------|-------|-------------|
| `tests/test_llm_backend.py` | 19 | Backend abstraction unit tests |
| `tests/test_parl_mcp_server.py` | 8 | MCP server tool tests |
| `tests/test_fact_check_debate.py` | 12 | Debate module unit tests |
| `tests/test_fact_check_debate_integration.py` | 2 | Real LLM integration tests (slow) |
| `tests/run_weave_test.py` | 13 | Full orchestrator tests |

Run all tests:

```bash
pytest tests/test_llm_backend.py tests/test_parl_mcp_server.py tests/test_fact_check_debate.py -v
```

Run integration test (requires API keys, takes 3-5 minutes):

```bash
pytest tests/test_fact_check_debate_integration.py -v -s
```
