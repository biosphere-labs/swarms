# Why Swarms (Not LangGraph)

## The One-Sentence Answer

**LangGraph gives you graph primitives to hand-build agent workflows. Swarms gives you 20+ pre-built orchestration patterns and an Agent that works autonomously out of the box — plus our fork adds PARL, which decomposes tasks dynamically at runtime instead of requiring you to pre-define the workflow.**

---

## What LangGraph Is

LangGraph is a **low-level orchestration framework** for building stateful, multi-agent applications. Its core abstraction is the **StateGraph** — a directed graph where:

- **Nodes** are functions that read and write to shared state
- **Edges** define transitions between nodes (static or conditional)
- **State** is a typed dictionary passed through the graph
- **Checkpoints** persist state for durability and resumption

```python
# LangGraph: you're building a state machine
from langgraph.graph import StateGraph, END

graph = StateGraph(AgentState)
graph.add_node("researcher", research_node)
graph.add_node("writer", write_node)
graph.add_edge("researcher", "writer")
graph.add_conditional_edges("writer", should_continue, {
    "continue": "researcher",
    "end": END
})
app = graph.compile()
result = app.invoke({"messages": [HumanMessage("Write about X")]})
```

LangGraph supports several workflow patterns:
- **Routing**: An LLM classifies input and directs it to specialized handlers
- **Parallelization**: Multiple nodes run simultaneously via the `Send` API
- **Orchestrator-Worker**: One node decomposes tasks, sends to workers via `Send`, collects results
- **Evaluator-Optimizer**: Generate-evaluate loops until quality threshold is met

### What LangGraph Does Well

- **Fine-grained control**: You define exactly how state flows between nodes
- **Durability**: Checkpointing lets you resume workflows after crashes
- **Human-in-the-loop**: Interrupt points where humans can inspect/modify state
- **Streaming**: Token-level and node-level streaming of intermediate results
- **Debugging**: LangGraph Studio provides visual graph inspection
- **Subgraphs**: Compose smaller graphs into larger ones

### Where LangGraph Falls Short

- **Everything is manual**: There are no pre-built multi-agent patterns. Supervisor, hierarchical, debate, mixture-of-experts — you wire each one from scratch using nodes, edges, and conditional routing
- **State management burden**: You design the state schema, write reducers, handle state merging between parallel branches. This is your problem, not the framework's
- **Rigid topology**: The graph is defined at coding time. You can't dynamically spawn 3 agents for one query and 8 for another — the graph structure is fixed
- **Boilerplate**: A simple "run 3 agents in parallel and aggregate" pattern requires defining a state schema, 3+ node functions, a `Send` call, reducer logic, and a synthesis node. In swarms, it's 3 lines
- **Abstraction mismatch**: Agents become functions stuffed into graph nodes. The agent's autonomy (looping, tool use, self-correction) must be implemented inside each node function — LangGraph doesn't help with that

---

## What Swarms Is

Swarms is built around a single idea: **agents are autonomous workers, and the interesting problem is how to coordinate them.**

### The Agent

The `Agent` class is a fully autonomous LLM-powered worker:

```python
agent = Agent(
    agent_name="Researcher",
    system_prompt="You research topics thoroughly.",
    model_name="deepinfra/Qwen/Qwen2.5-72B-Instruct",
    tools=[serper_search],
    max_loops=3,
)
result = agent.run("What are the latest advances in battery technology?")
```

Key capabilities built into every Agent:
- **Any LLM** via LiteLLM (100+ models — OpenAI, Anthropic, DeepInfra, Ollama, local)
- **Autonomous loops** — reason, act, observe, iterate (`max_loops`)
- **Tool/function calling** with MCP protocol support
- **Memory** — short-term conversation + long-term vector storage
- **Streaming**, artifact generation, state save/load
- **Fallback models** on error

An Agent doesn't need a graph wrapping it. It runs autonomously, uses tools, and self-corrects across multiple loops. This is the fundamental difference: in LangGraph, an "agent" is a function inside a node. In swarms, an Agent is an autonomous entity that the orchestration layer coordinates.

### 20+ Pre-Built Orchestration Patterns

Instead of building coordination from graph primitives, swarms ships ready-to-use patterns:

| Pattern | What It Does | LangGraph Equivalent |
|---------|-------------|---------------------|
| **SequentialWorkflow** | Agents run one after another, passing context | Hand-wire linear edges between nodes |
| **ConcurrentWorkflow** | Agents run in parallel on the same task | Write `Send` API calls + reducer logic |
| **AgentRearrange** | Flow syntax: `"A -> B, C"` (seq + parallel) | Build a custom graph with conditional edges |
| **MixtureOfAgents** | Specialists solve in parallel, aggregator synthesizes | No equivalent — build from scratch |
| **HierarchicalSwarm** | Director plans, workers execute, director evaluates | No equivalent — build from scratch |
| **GroupChat** | Agents discuss with speaker selection | No equivalent — build from scratch |
| **MajorityVoting** | Multiple agents vote, majority wins | No equivalent — build from scratch |
| **CouncilAsAJudge** | Agents work, a judge evaluates quality | Evaluator-optimizer pattern (manual) |
| **DebateWithJudge** | Agents argue opposing sides, judge decides | No equivalent — build from scratch |
| **GraphWorkflow** | DAG execution via NetworkX | This is basically what LangGraph is |
| **SwarmRouter** | Dynamically picks the right pattern | No equivalent |
| **PARLOrchestrator** | **Our addition** — dynamic decomposition | Orchestrator-worker (but static) |

Using any pattern is typically 3-5 lines:

```python
# Mixture of Agents — 3 specialists + aggregator
workflow = MixtureOfAgents(
    agents=[researcher, analyst, writer],
    aggregator_agent=synthesizer,
)
result = workflow.run("Analyze the competitive landscape for X")
```

```python
# Hierarchical — director delegates to workers
swarm = HierarchicalSwarm(
    director=manager_agent,
    agents=[backend_dev, frontend_dev, tester],
)
result = swarm.run("Build a user authentication system")
```

```python
# Debate — two agents argue, judge decides
debate = DebateWithJudge(
    agents=[optimist, pessimist],
    judge=judge_agent,
)
result = debate.run("Should we adopt microservices?")
```

To build any of these in LangGraph, you'd write 50-100 lines of state management, node definitions, edge wiring, and compilation logic — per pattern.

---

## The PARL Orchestrator (Our Addition)

This is why we forked. We added the **PARLOrchestrator** — a dynamic decomposition engine inspired by the PARL paper (Kimi K2.5's agent system):

```
Input Task
    ↓
1. Decomposition Engine    — Strong LLM splits task into sub-tasks
    ↓
2. Critical Path Scheduler — Orders into parallel cohorts
    ↓
3. Context Sharding        — Each sub-agent gets only relevant context
    ↓
4. Parallel Execution      — Cohorts run concurrently
    ↓
5. Result Aggregation      — Synthesize, flag contradictions & gaps
    ↓
6. Iteration (optional)    — Re-focus on gaps if quality insufficient
```

### Why This Matters vs. LangGraph

LangGraph's orchestrator-worker pattern requires you to **pre-define** the graph topology. You decide at coding time how many workers exist, what each does, and how results merge. The `Send` API lets you fan-out dynamically, but you still manually write the orchestration logic, the state schema, and the aggregation.

PARL makes all of this **dynamic**:
- The orchestrator LLM decides **at runtime** how to decompose the task
- Sub-agent count varies per query (2 for simple, 8 for complex)
- Context sharding prevents sub-agents from being polluted by irrelevant information
- The aggregator detects contradictions and gaps, triggering re-iteration
- Anti-collapse heuristics prevent degenerate strategies (always single-agent, spurious parallelism)

You don't build a graph. You describe a goal, and the system figures out the execution plan:

```python
orchestrator = PARLOrchestrator(
    orchestrator_model="deepinfra/Qwen/Qwen3-235B-A22B",
    sub_agent_model="deepinfra/Qwen/Qwen2.5-72B-Instruct",
    tools=[serper_search],
    max_parallel=8,
)
result = orchestrator.run("Compare the pricing, features, and market position of the top 5 CRM platforms")
# Decomposed into 5+ sub-tasks, run in parallel, synthesized with contradiction detection
```

---

## Head-to-Head Comparison

| Dimension | LangGraph | Swarms (our fork) |
|-----------|-----------|-------------------|
| **Core abstraction** | State machine (nodes + edges + state) | Autonomous agents + orchestration patterns |
| **Multi-agent patterns** | Build from primitives (manual) | 20+ pre-built, ready to use |
| **Agent autonomy** | Agent = function in a node (you implement loops) | Agent has built-in loops, tools, self-correction |
| **Dynamic orchestration** | No — graph topology is fixed at compile time | Yes — PARL decomposes at runtime |
| **State management** | Explicit (you design schema + reducers) | Handled by the pattern (transparent) |
| **Boilerplate** | High (state, nodes, edges, compile, invoke) | Low (instantiate pattern, call `.run()`) |
| **LLM support** | Via LangChain integrations | Via LiteLLM (100+ models, one interface) |
| **Tool calling** | Via LangChain tools | Native + MCP protocol |
| **Checkpointing** | Built-in, excellent | State save/load per agent |
| **Human-in-the-loop** | First-class (interrupt points) | Supported in AgentRearrange |
| **Streaming** | Excellent (token + node level) | Supported per agent |
| **Debugging** | LangGraph Studio (visual) | Conversation logging + telemetry |
| **Dependency weight** | langgraph + langchain-core + (often) langchain | swarms + litellm |
| **When to use** | You need fine-grained control over exact state flow | You need multi-agent coordination fast |

---

## Why Fork Instead of Depend

We forked kyegomez/swarms rather than using it as a pip dependency because:

1. **PARL integration**: The orchestrator needed deep integration with swarms internals (BaseSwarm, Agent, tool passing). A plugin wouldn't have clean access.

2. **Iteration speed**: We modify Agent behavior, fix bugs, and tune orchestration without waiting for upstream PRs.

3. **Model defaults**: Pre-configured for DeepInfra models (cheap, fast) rather than defaulting to OpenAI.

4. **Future fine-tuning**: We plan to LoRA fine-tune a small model for the decomposition step. This requires tight integration with the decomposition engine.

---

## When You'd Actually Want LangGraph Instead

Being honest about where LangGraph wins:

- **Complex stateful workflows**: If your agents need durable state that survives crashes, with checkpointing and resumption, LangGraph's persistence is battle-tested
- **Precise control flow**: If you need exact control over which agent runs when, with conditional branches that depend on specific state values, LangGraph's explicit graph is clearer
- **Human-in-the-loop heavy**: If humans frequently need to inspect and modify intermediate state, LangGraph's interrupt model is more mature
- **Visual debugging**: LangGraph Studio lets you see the graph executing in real time
- **You're already in the LangChain ecosystem**: If you use LangChain for RAG/chains and just need to add agent coordination, LangGraph is the natural extension

---

## Summary

**LangGraph** is a low-level state machine framework. It gives you nodes, edges, and state — you build everything else. This is powerful when you need precise control, but it means every multi-agent pattern is a custom build.

**Swarms** is a high-level orchestration framework. It ships 20+ coordination patterns where agents are autonomous workers, not functions in a graph. Our fork adds PARL for dynamic, runtime-determined orchestration — something LangGraph's fixed graph topology can't do.

We chose swarms because our problem is **"coordinate multiple agents to solve complex tasks in parallel, where the decomposition itself is dynamic."** LangGraph would have us hand-building the orchestration layer from state machine primitives. Swarms gives us the patterns out of the box, and PARL gives us the dynamic decomposition on top.
