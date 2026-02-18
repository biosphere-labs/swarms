# biosphere-labs/swarms — PARL Orchestrator

A fork of [kyegomez/swarms](https://github.com/kyegomez/swarms) adding a **PARL-inspired dynamic orchestration layer** — cloning the core behavior of [Kimi K2.5's Agent Swarm](https://arxiv.org/html/2602.02276v1) on top of the existing multi-agent framework.

For the full upstream documentation (Agent class, all 20+ swarm types, LiteLLM integration, tools, etc.), see the [upstream README](https://github.com/kyegomez/swarms#readme).

## What This Fork Adds

| Module | Purpose |
|--------|---------|
| `PARLOrchestrator` | Main swarm type — dynamically decomposes tasks into parallel sub-agent work via LLM |
| `DecompositionEngine` | Analyzes tasks and produces sub-task graphs (parallel groups + dependencies) |
| `CriticalPathScheduler` | Orders execution to minimize wall-clock time, not just maximize parallelism |
| `ContextShardingManager` | Isolates context per sub-agent — only structured results flow back to orchestrator |
| `ResultAggregator` | Merges sub-agent outputs, flags contradictions, identifies gaps |

## How It Works

```
Task → DecompositionEngine → CriticalPathScheduler → Parallel Sub-Agents → ResultAggregator → Output
           (LLM splits         (orders by              (isolated context,     (merge, flag
            into sub-tasks)     critical path)          concurrent execution)  contradictions)
```

Key differences from existing swarm types (ConcurrentWorkflow, HierarchicalSwarm, etc.):
- **Dynamic decomposition** — the LLM decides how to split work, not a predefined workflow
- **Context sharding** — sub-agents get focused context slices, preventing cross-contamination
- **Critical-path scheduling** — optimizes wall-clock latency via `CriticalSteps = sum(S_main + max(S_sub))`
- **Anti-collapse heuristics** — prevents defaulting to single-agent (serial collapse) or spawning empty agents (spurious parallelism)

## Quick Start

```bash
pip install swarms
```

```python
from swarms import PARLOrchestrator

orchestrator = PARLOrchestrator(
    orchestrator_model="gpt-4o-mini",
    sub_agent_model="gpt-4o-mini",
    max_parallel=8,
)

result = orchestrator.run(
    "Research competitor Acme Corp across funding, reviews, pricing, and team"
)
print(result)
```

### Via SwarmRouter

```python
from swarms.structs.swarm_router import SwarmRouter, SwarmType

router = SwarmRouter(
    swarm_type=SwarmType.PARLOrchestrator,
    agents=[],  # PARLOrchestrator creates sub-agents dynamically
)
result = router.run("Research competitor Acme Corp across funding, reviews, pricing, and team")
```

## Configuration

All parameters can be set three ways (in priority order):

1. **Constructor arguments** (highest priority)
2. **Environment variables**
3. **Defaults**

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PARL_ORCHESTRATOR_MODEL` | `gpt-4o-mini` | Model for task decomposition (use a strong reasoning model) |
| `PARL_SUB_AGENT_MODEL` | `gpt-4o-mini` | Model for sub-agents and result aggregation (can be cheaper/faster) |
| `PARL_MAX_PARALLEL` | `10` | Maximum concurrent sub-agents per cohort |
| `PARL_MAX_ITERATIONS` | `2` | Gap-fill iterations (1 = no gap-fill) |
| `PARL_TIMEOUT` | `300` | Overall orchestration timeout (seconds) |
| `PARL_SUB_AGENT_TIMEOUT` | `120` | Per sub-agent timeout (seconds) |
| `PARL_TOKEN_BUDGET` | `100000` | Total token budget across all agents |
| `PARL_SYNTHESIS_BACKEND` | `litellm` | Backend for synthesis: `litellm` or `claude-code` |
| `PARL_SYNTHESIS_MODEL` | *(orchestrator)* | Model for synthesis (only used with `litellm` backend) |

### Using Claude Subscription for Synthesis

Set `PARL_SYNTHESIS_BACKEND=claude-code` to run synthesis through your Claude subscription (via `claude -p`) instead of paying per-token through DeepInfra or other providers. Sub-agents still use DeepInfra with cheap models — only the final synthesis step uses Claude.

```bash
export PARL_SYNTHESIS_BACKEND="claude-code"
# Sub-agents remain on cheap DeepInfra models
export PARL_SUB_AGENT_MODEL="deepinfra/Qwen/Qwen2.5-72B-Instruct"
```

This applies to `parl_execute`, `parl_review`, and `parl_smart_review`.

### Example: Using DeepInfra with Qwen

```bash
export DEEPINFRA_API_KEY="your-key-here"
export PARL_ORCHESTRATOR_MODEL="deepinfra/Qwen/Qwen3-235B-A22B"
export PARL_SUB_AGENT_MODEL="deepinfra/Qwen/Qwen2.5-72B-Instruct"
export PARL_TIMEOUT="600"
```

```python
# No model arguments needed — picks up env vars automatically
orchestrator = PARLOrchestrator()
result = orchestrator.run("Analyze the competitive landscape for autonomous vehicles")
```

### Example: Fast Local Models via Ollama

```bash
export PARL_ORCHESTRATOR_MODEL="ollama/llama3.1:70b"
export PARL_SUB_AGENT_MODEL="ollama/llama3.1:8b"
```

### Model Selection Guide

| Role | Recommendation | Why |
|------|---------------|-----|
| Orchestrator (decomposition) | Strong reasoning model | Needs to understand task structure and produce valid JSON |
| Sub-agents | Fast/cheap model | Focused tasks don't need frontier models |
| Aggregator | Same as sub-agents | Synthesis from structured inputs, not open-ended reasoning |

Tested combinations:

| Orchestrator | Sub-agents | Speed | Cost/run |
|-------------|-----------|-------|----------|
| gpt-4o-mini | gpt-4o-mini | ~30s | ~$0.05 |
| deepinfra/Qwen/Qwen3-235B-A22B | deepinfra/Qwen/Qwen2.5-72B-Instruct | ~2min | ~$0.03 |
| claude-sonnet-4-5-20250929 | gpt-4o-mini | ~45s | ~$0.08 |

## Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | `PARLOrchestrator` | Instance name for logging |
| `orchestrator_model` | str | env or `gpt-4o-mini` | Model for task decomposition & synthesis |
| `sub_agent_model` | str | env or `gpt-4o-mini` | Model for sub-agents and aggregation |
| `max_parallel` | int | env or `10` | Max concurrent sub-agents per cohort |
| `max_iterations` | int | env or `2` | Gap-fill iterations |
| `timeout` | int | env or `300` | Total orchestration timeout (seconds) |
| `sub_agent_timeout` | int | env or `120` | Per-sub-agent timeout (seconds) |
| `token_budget` | int | env or `100000` | Total token budget |

## Architecture

```
Task
  ↓
DecompositionEngine → "Split into 4 parallel research areas"
  ↓
CriticalPathScheduler → "Execute Area-1,2,3,4 in parallel; Area-5 after"
  ↓
[Parallel Cohorts] → ContextShardingManager → 4 Sub-Agents (concurrent)
  ↓
ResultAggregator → "Synthesize results; found 1 gap"
  ↓
If gaps: Refocus on gaps → Iteration 2
Else: Return final synthesized answer
```

The orchestrator executes a six-stage pipeline:

1. **Decomposition** — LLM analyzes the task and produces a sub-task graph with explicit parallelization points
2. **Scheduling** — Critical-path scheduler orders execution into parallel cohorts (minimizes wall-clock latency)
3. **Context Sharding** — Each sub-agent receives only relevant context (prevents cross-contamination)
4. **Parallel Execution** — Sub-tasks within each cohort run concurrently via ThreadPoolExecutor
5. **Aggregation** — Results are merged with contradiction detection and gap identification
6. **Iteration** — If gaps found and iterations remain, refocus decomposition on gaps and loop

## Use Cases

- **Wide Research** — "Research X across Y dimensions" → fans out to 6-12 parallel agents
- **Deep Analysis** — "Analyze this codebase for security issues" → decomposes by module
- **Document Processing** — "Summarize these 20 documents" → each agent handles 2-3 docs
- **Fact Checking** — "Verify these 15 claims" → parallel verification, cross-reference contradictions

## Files Added

```
swarms/structs/
    parl_orchestrator.py        # Main PARLOrchestrator class
    parl_mcp_server.py          # MCP server with all tools (execute, review, smart_review)
    decomposition_engine.py     # Task analysis and splitting
    critical_path_scheduler.py  # Latency-optimized execution planning
    context_sharding.py         # Per-sub-agent context isolation
    result_aggregator.py        # Output merging and contradiction detection
    fact_check_debate.py        # 3-agent debate fact-checking
    llm_backend.py              # Pluggable LLM backend (litellm / claude-code)

swarms/prompts/
    parl_prompts.py             # Decomposition and synthesis prompts

examples/mcp/servers/
    parl_server.py              # MCP server launcher with config display

scripts/
    install-parl-service.sh     # systemd service installer
    uninstall-parl-service.sh   # systemd service removal

docs/swarms/structs/
    parl_orchestrator_mcp.md    # Full MCP server documentation

tests/
    test_parl_orchestrator.py
    test_parl_mcp_server.py     # MCP tool tests
    test_decomposition_engine.py
    test_critical_path_scheduler.py
    test_context_sharding.py
    test_result_aggregator.py
    test_llm_backend.py         # Backend abstraction tests
    test_fact_check_debate.py   # Debate module tests
```

## MCP Server

The orchestrator runs as an MCP server, making all tools available to Claude Code, Cursor, Windsurf, and any MCP-compatible client.

```bash
# Start the server
python examples/mcp/servers/parl_server.py

# Or auto-start on login (Linux)
./scripts/install-parl-service.sh

# Register with Claude Code
claude mcp add --transport http parl-orchestrator http://localhost:8765/mcp
```

### Available MCP Tools

| Tool | Purpose |
|------|---------|
| **`parl_execute`** | Auto-decompose a task into parallel sub-agents, synthesize results |
| **`parl_review`** | Review a document from multiple expert personas in parallel, with per-persona model assignment |
| **`parl_smart_review`** | Enhanced review with auto model diversity, cross-model fact-checking, and blind spot analysis |
| **`parl_config`** | Show current server configuration |

All tools stream progress via MCP Context — clients see real-time status as each sub-agent completes.

Full documentation: [`docs/swarms/structs/parl_orchestrator_mcp.md`](docs/swarms/structs/parl_orchestrator_mcp.md)

## Additional Modules

Beyond the core orchestrator, this fork adds:

| Module | Purpose |
|--------|---------|
| `FactCheckDebate` | 3-agent debate verification (Researcher → Fact-Checker → Judge) |
| `LiteLLMBackend` | Pluggable LLM backend abstraction with API key rotation |
| `ResultAggregator` | Synthesis with contradiction detection, gap identification |
| `serper_search` | Web search tool passed through to sub-agents |

### Fact-Check Debate

Optional per-agent verification where each claim goes through a 3-round debate:

```python
from swarms.structs.fact_check_debate import FactCheckDebate

debate = FactCheckDebate(model_name="deepinfra/Qwen/Qwen2.5-72B-Instruct")
verified = debate.verify(research_output="Claims to verify...")
```

### Multi-Persona Review (Python API)

```python
from swarms.structs.parl_mcp_server import parl_review

result = await parl_review(
    document="Your document text...",
    personas='[{"name": "Market Analyst", "instruction": "Evaluate market claims", "model": "deepinfra/Qwen/Qwen3-235B-A22B"}, {"name": "Skeptic", "instruction": "Find weakest arguments", "model": "deepinfra/deepseek-ai/DeepSeek-V3.2"}]',
)
```

## What We're NOT Changing

The existing framework is used as-is: Agent class, LiteLLM wrapper, tool system, all existing swarm types, tests, and examples. All additions are new files.

## Reference

- [Kimi K2.5 Technical Report](https://arxiv.org/html/2602.02276v1) — PARL architecture and training methodology
- [Unofficial PARL Implementation](https://github.com/The-Swarm-Corporation/PARL) — Reward function and CriticalSteps metric
- [Kimi System Prompts (extracted)](https://github.com/dnnyngyen/kimi-k2.5-prompts-tools) — Agent architecture analysis
- [Upstream swarms README](https://github.com/kyegomez/swarms#readme) — Full documentation for the base framework

## Upstream Sync

```bash
git fetch upstream
git merge upstream/master
```
