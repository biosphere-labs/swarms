# The Plain-English Guide to PARL Orchestrator

## What Is This Project?

This project is a **smart task splitter for AI agents**. You give it one big task, and it automatically figures out how to break it into smaller pieces, runs those pieces at the same time using multiple AI agents, and combines the results into one answer.

Think of it like a project manager who:
1. Reads your request
2. Figures out which parts can be done in parallel
3. Assigns each part to a different worker
4. Collects all the results
5. Writes up a single coherent answer

The workers (sub-agents) each get only the information they need — they can't see what the other workers are doing. This prevents confusion and keeps costs down.

## Why Does This Exist?

### The Problem

If you ask a single AI agent to "research Company X across funding, pricing, team, reviews, market position, and recent news," the agent does everything sequentially — one topic at a time. This is slow (6 API calls, one after another) and expensive (the context window grows with each step, meaning later calls cost more tokens).

### The Solution

The PARL Orchestrator splits that into 6 independent tasks, runs them all at the same time, and combines the results. What took 3 minutes now takes 30 seconds.

This is what Moonshot AI did with their Kimi K2.5 model — they trained a model (via a technique called PARL, Parallel-Agent Reinforcement Learning) to automatically decide when to split work and how to coordinate sub-agents. We can't retrain model weights, so we approximate the same behavior with explicit prompts and heuristics.

## How Is This Different from LangGraph?

This is the most common question, so let's be direct.

### LangGraph: You Design the Workflow

LangGraph is a framework for building **explicit state machines**. You define nodes (steps), edges (transitions), and conditions (when to take which path). You draw the graph. You decide the flow.

```
# LangGraph: You define every step and transition
graph = StateGraph(AgentState)
graph.add_node("research", research_agent)
graph.add_node("analyze", analysis_agent)
graph.add_node("write", writing_agent)
graph.add_edge("research", "analyze")
graph.add_edge("analyze", "write")
```

This is great when you know the workflow in advance: "always research, then analyze, then write." But it means you're hard-coding the orchestration logic. If a task would be better served by 3 parallel researchers instead of 1, you need to redesign the graph.

### PARL Orchestrator: The AI Designs the Workflow

The PARL Orchestrator doesn't have a predefined graph. You give it a task, and the LLM itself decides:
- How many sub-tasks to create
- Which ones can run in parallel
- What context each sub-agent needs
- How to combine the results

```python
# PARL: Just give it the task
orchestrator = PARLOrchestrator()
result = orchestrator.run("Research Company X across 6 dimensions")
# It figures out the 6 parallel tasks, runs them, merges results
```

### When to Use Which

| Scenario | Use LangGraph | Use PARL Orchestrator |
|----------|--------------|----------------------|
| Known, repeatable workflow | Yes | No |
| Dynamic, task-dependent parallelism | No | Yes |
| Need fine-grained control over each step | Yes | No |
| "Research X across N dimensions" | No | Yes |
| Chatbot with tool calling | Yes | No |
| Batch analysis of many documents | No | Yes |
| Multi-step reasoning chain | Yes | Maybe |

**Short version:** LangGraph is for workflows you design. PARL Orchestrator is for workflows the AI designs.

### What About LangChain?

LangChain is a toolkit for building LLM applications — chains, memory, tool calling, retrieval. It's lower-level than either LangGraph or PARL Orchestrator. You could theoretically build a PARL-style orchestrator on top of LangChain, but we built on [swarms](https://github.com/kyegomez/swarms) instead because it already had a mature Agent class, 20+ orchestration patterns, and LiteLLM integration (100+ model providers out of the box).

### What About CrewAI?

CrewAI uses **predefined roles** — you create agents with fixed personas (Researcher, Writer, Editor) and wire them together. It's closer to LangGraph in spirit: you define the team structure upfront. PARL Orchestrator doesn't use predefined roles — it creates disposable, focused sub-agents on the fly based on the task.

## The Five Components

### 1. DecompositionEngine

Takes a task and asks the LLM: "How should I split this into parallel sub-tasks?"

The LLM returns a JSON structure with:
- Groups of sub-tasks that can run in parallel
- Dependencies between groups (group 2 needs results from group 1)
- Context hints for each sub-task (what information it needs)

It also enforces anti-patterns:
- **Anti-serial-collapse**: If the LLM says "just do it all in one task," the engine pushes back and asks for a real split (for complex enough tasks)
- **Anti-spurious-parallelism**: If the LLM creates 10 sub-tasks that are all basically the same thing, the engine consolidates them

### 2. CriticalPathScheduler

Takes the sub-task graph and figures out the fastest execution order.

The key insight: maximum parallelism isn't always fastest. If you have 10 tasks where 9 take 1 second and 1 takes 60 seconds, running all 10 in parallel means you're waiting 60 seconds anyway. The scheduler identifies the critical path (the longest chain of dependent tasks) and optimizes for that.

It uses the CriticalSteps metric from the PARL paper:
```
CriticalSteps = sum over each stage of (orchestrator steps + slowest sub-agent in that stage)
```

### 3. ContextShardingManager

Creates isolated "context shards" for each sub-agent. Instead of giving every agent the full conversation history (expensive and noisy), each agent gets:
- A focused system prompt for their specific sub-task
- Only the relevant context slice (e.g., "research pricing" doesn't need the team bios)
- An output format specification

This is "context sharding, not context truncation" — the agent gets a curated slice, not a random window of the full context.

### 4. Parallel Execution (via ThreadPoolExecutor)

Sub-agents within each cohort run concurrently. Each agent is a fresh instance with its own context — no shared state, no cross-contamination. If one agent fails or times out, the others continue.

### 5. ResultAggregator

Takes all sub-agent outputs and:
- Synthesizes them into a single coherent answer
- Detects contradictions (agent A says revenue is $10M, agent B says $15M)
- Identifies gaps (nobody covered the patent portfolio)
- If gaps are found and iterations remain, triggers another round of decomposition focused on the gaps

## How It All Fits Together

```
You: "Research TechCorp across funding, pricing, team, reviews, market position, patents"

DecompositionEngine:
  → Group 1 (parallel): [funding, pricing, team, reviews, market-position, patents]
  → No dependencies (all independent)

CriticalPathScheduler:
  → Single cohort of 6 tasks (all parallel)

ContextShardingManager:
  → Agent 1 gets: "Research TechCorp's funding history"
  → Agent 2 gets: "Research TechCorp's pricing strategy"
  → Agent 3 gets: "Research TechCorp's executive team"
  → ... (each with focused context only)

ThreadPoolExecutor:
  → All 6 agents run simultaneously
  → Each finishes in ~5 seconds
  → Total wall-clock: ~5 seconds (vs ~30 seconds serial)

ResultAggregator:
  → Combines 6 reports into one coherent competitive analysis
  → Flags: "Agent 1 says Series B was $50M, Agent 4 mentions $45M" (contradiction)
  → Gap: "No agent covered their recent acquisition of StartupY"
  → If iterations remain: re-decompose focusing on the gap
```

## Cost Model

The key cost optimization: use a strong model for decomposition (needs to understand task structure) and a cheap model for sub-agents (focused tasks don't need frontier reasoning).

| Component | Tokens | Cost |
|-----------|--------|------|
| Orchestrator decomposition | ~2K in, ~1K out | ~$0.01 |
| 6x Sub-agents (cheap model) | ~3K in, ~2K out each | ~$0.003 |
| Result aggregation | ~10K in, ~3K out | ~$0.04 |
| **Total** | | **~$0.05** |

## Getting Started

```bash
pip install swarms
```

```python
from swarms import PARLOrchestrator

# Simplest possible usage
orchestrator = PARLOrchestrator()
result = orchestrator.run("Your complex task here")
print(result)
```

Or configure via environment variables (no code changes):

```bash
export PARL_ORCHESTRATOR_MODEL="gpt-4o-mini"
export PARL_SUB_AGENT_MODEL="gpt-4o-mini"
```

See the [README](../README.md) for full configuration options.
