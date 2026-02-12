# Why We Forked kyegomez/swarms

## The Short Answer

We forked [kyegomez/swarms](https://github.com/kyegomez/swarms) because it gives us a production-grade Agent class, 16+ orchestration patterns, multi-provider LLM support via LiteLLM, and a tool/MCP integration layer — all as a foundation to build our PARL-inspired dynamic orchestration on top of. Building from scratch or using LangGraph would have meant reimplementing most of what swarms already provides.

## What Swarms Gives Us

### 1. The Agent Class (6,165 lines)

This is the core value. A single, comprehensive LLM agent abstraction with:

- **Multi-provider LLM support** via LiteLLM — OpenAI, Anthropic, Google, Groq, Mistral, DeepSeek, DeepInfra, Ollama, and 100+ more providers through one interface
- **Tool/function calling** — Pass Python callables, get automatic JSON schema generation, execution, and result handling
- **MCP (Model Context Protocol)** — Connect to MCP servers for standardized tool access
- **Autonomous loops** — `max_loops` controls how many think-act cycles the agent runs (or `"auto"` for self-determined)
- **Streaming** — Token-by-token output with callbacks
- **Fallback models** — If primary model fails, automatically tries fallback chain
- **Memory** — Optional long-term vector database integration
- **Output types** — `"str"`, `"json"`, `"dict"`, `"yaml"`, `"final"` (last message only)

We use Agent as-is for every sub-agent in our PARL pipeline. We never had to modify it.

### 2. BaseSwarm — The Extension Point

Abstract base class that all swarm types inherit from:

```python
class BaseSwarm(ABC):
    def run(task) -> str           # Main entry
    def add_agent() / remove_agent()
    def broadcast(message)         # Send to all agents
    def scale_up() / scale_down()  # Dynamic sizing
    def get_swarm_status()         # Status snapshot
```

Our `PARLOrchestrator` inherits from `BaseSwarm`, which means it automatically integrates with the rest of the ecosystem (SwarmRouter, AutoSwarmBuilder, etc.) without any glue code.

### 3. Pre-built Orchestration Patterns (16+)

These exist alongside our PARL orchestrator:

| Pattern | What It Does |
|---------|-------------|
| **ConcurrentWorkflow** | All agents work on same task in parallel, shared context |
| **SequentialWorkflow** | Agent chain — output of one feeds into next |
| **MixtureOfAgents** | Parallel specialists + aggregator layer |
| **HierarchicalSwarm** | Manager delegates to workers, collects results |
| **GroupChat** | Multi-agent discussion with speaker selection |
| **GraphWorkflow** | DAG-based execution with NetworkX |
| **MajorityVoting** | Multiple agents vote on answer |
| **AgentRearrange** | Dynamic task reallocation |
| **AutoSwarmBuilder** | LLM analyzes task, generates agents and picks pattern |

We didn't need to build any of these. They're available if needed for different use cases.

### 4. Concurrency Primitives

`multi_agent_exec.py` provides:

- `run_agents_concurrently(agents, task)` — ThreadPoolExecutor with 95% CPU core utilization
- `run_agents_with_different_tasks(agents, tasks)` — Each agent gets a different task
- `batched_grid_agent_execution(agents, tasks)` — All agents x all tasks grid
- Automatic exception handling in thread pools

Our PARLOrchestrator uses ThreadPoolExecutor directly for finer control, but these utilities are available.

### 5. Tool System

- **BaseTool** — Converts Python functions to OpenAI function-calling schemas automatically
- **MCP Client** — Full Model Context Protocol support for standardized tool access
- **Tool Registry** — Central lookup and management
- **Handoff Tool** — Agent-to-agent task delegation

Sub-agents in our PARL pipeline get tools passed through cleanly via the Agent class.

### 6. SwarmRouter — Pattern Dispatch

```python
router = SwarmRouter(swarm_type="PARLOrchestrator", agents=[...])
result = router.run("complex research task")
```

Our PARLOrchestrator is registered as a SwarmType, making it selectable via the same routing mechanism as all other patterns. AutoSwarmBuilder can recommend it when appropriate.

## What We Built On Top

These are our additions — the PARL intelligence layer:

| Component | Purpose | File |
|-----------|---------|------|
| **PARLOrchestrator** | Main pipeline: decompose → schedule → shard → execute → aggregate → iterate | `structs/parl_orchestrator.py` |
| **DecompositionEngine** | LLM-based task splitting with anti-collapse heuristics | `structs/decomposition_engine.py` |
| **CriticalPathScheduler** | Latency-optimized cohort ordering (PARL paper's CriticalSteps metric) | `structs/critical_path_scheduler.py` |
| **ContextShardingManager** | Isolated context windows per sub-agent (not shared context) | `structs/context_sharding.py` |
| **ResultAggregator** | Output merging with contradiction detection and gap identification | `structs/result_aggregator.py` |

### Key Innovation: Context Sharding vs Shared Context

Most multi-agent frameworks (including swarms' own ConcurrentWorkflow) give all agents the full conversation context. This causes:
- Token waste (each agent processes irrelevant context)
- Cross-contamination (agents influenced by each other's partial work)
- Context window limits hit faster

Our ContextShardingManager gives each sub-agent only the context slice relevant to its sub-task. Only structured results flow back to the orchestrator.

### Key Innovation: Dynamic Decomposition

Unlike pre-configured agent teams (where you define roles upfront), PARLOrchestrator:
1. Receives a task
2. LLM analyzes what parallel work is possible
3. Dynamically creates sub-tasks with dependencies
4. Spawns exactly the right number of agents
5. Schedules execution by critical path
6. Iterates if gaps are found

No predefined roles. No hand-coded workflows.

## Why Not LangGraph?

| Aspect | Swarms | LangGraph |
|--------|--------|-----------|
| **Agent abstraction** | Full-featured Agent class (tools, memory, streaming, autonomous loops) | Minimal — you build your own agent logic in graph nodes |
| **LLM providers** | 100+ via LiteLLM built-in | Primarily OpenAI-focused, bring your own for others |
| **Orchestration patterns** | 16+ pre-built (use or extend) | Build everything as a StateGraph |
| **Approach** | Pattern library — pick a pattern, configure it | Graph construction — define nodes, edges, state schema |
| **Tool integration** | BaseTool + MCP built-in | Tool framework exists but less batteries-included |
| **Extension model** | Inherit BaseSwarm, implement `run()` | Define new graph structures |
| **Context handling** | Shared or sharded (ContextShardingManager) | State passed between nodes (you manage isolation) |

LangGraph is a great low-level graph execution engine. But for our use case (dynamic task decomposition with parallel agents), we'd have been reimplementing most of what swarms already provides — Agent, tool integration, concurrent execution, and the multi-pattern routing.

## Why Not Build From Scratch?

We'd need to build:
1. An Agent class with LLM integration (LiteLLM wrapping, tool calling, streaming, retries)
2. Concurrent execution infrastructure (ThreadPoolExecutor management, timeouts, error handling)
3. Tool system (function → schema conversion, execution, result handling)
4. MCP protocol support
5. State management and conversation tracking

That's easily 10,000+ lines of infrastructure code before we even start on the PARL-specific logic. By forking swarms, we got all of that and focused purely on our innovation: the decomposition → scheduling → sharding → aggregation pipeline.

## What We Don't Use (But Could)

- **GraphWorkflow** — DAG execution via NetworkX (we built CriticalPathScheduler instead for PARL-specific scheduling)
- **GroupChat** — Multi-agent discussion (could be useful for debate-based verification)
- **HierarchicalSwarm** — Manager-worker pattern (similar to PARL but without dynamic decomposition)
- **AutoSwarmBuilder** — Could eventually select PARLOrchestrator automatically based on task analysis
- **Long-term memory** — Vector DB integration exists in Agent but we haven't wired it up yet

## Architecture Summary

```
                    ┌─────────────────────────────┐
                    │      User Task              │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │    PARLOrchestrator (NEW)    │ ← inherits BaseSwarm
                    │                             │
                    │  ┌─ DecompositionEngine (NEW)│ ← LLM task splitting
                    │  ├─ CriticalPathScheduler(NEW)│ ← cohort ordering
                    │  ├─ ContextShardingManager(NEW)│ ← isolated contexts
                    │  └─ ResultAggregator (NEW)  │ ← synthesis + contradictions
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │  Agent (EXISTING, 6165 lines)│ ← used as-is
                    │  + LiteLLM (100+ providers)  │
                    │  + BaseTool + MCP            │
                    │  + Streaming + Memory        │
                    └─────────────────────────────┘
```

**Bottom line**: Swarms gives us the plumbing (Agent, LLM integration, tools, concurrency). We added the intelligence (dynamic decomposition, context isolation, critical-path scheduling, result aggregation with contradiction detection).
