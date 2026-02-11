# Swarm Orchestrator (biosphere-labs/swarms)

Fork of [kyegomez/swarms](https://github.com/kyegomez/swarms) — adding PARL-inspired dynamic orchestration on top of the existing multi-agent framework.

## Purpose

Build a Kimi K2.5 Agent Swarm clone: an orchestration layer where the system dynamically decides when to parallelize, how to decompose tasks into sub-agent work, and how to shard context — without hand-coded workflows. The base framework already provides 20+ orchestration patterns (ConcurrentWorkflow, HierarchicalSwarm, MixtureOfAgents, etc.) and a mature Agent class with LiteLLM multi-provider support. We're adding the intelligence layer on top.

## What Kimi K2.5 Agent Swarm Does (Our Target)

1. Orchestrator receives a complex task
2. Dynamically decomposes it into parallelizable sub-tasks (no predefined roles or workflow)
3. Spawns up to 100 frozen sub-agents, each with isolated context
4. Sub-agents execute concurrently with their own tool access
5. Only task-relevant outputs flow back to orchestrator (context sharding, not context truncation)
6. Orchestrator synthesizes results into final output
7. Performance: 4.5x faster than serial, 78.4% on BrowseComp vs 60.6% single-agent

Trained via PARL (Parallel-Agent Reinforcement Learning) — we approximate this with explicit heuristics since we can't retrain model weights.

## Scope of Additions

### What We're Building (on top of existing framework)

1. **PARLOrchestrator** — New swarm type inheriting from BaseSwarm
   - Dynamic task decomposition via LLM (replaces PARL's trained behavior with prompted behavior)
   - Decides parallelizable vs sequential sub-tasks
   - Spawns sub-agents with isolated context windows
   - Collects and synthesizes results
   - Critical-path-aware scheduling (minimize wall-clock time, not just total work)

2. **ContextShardingManager** — Sub-agent context isolation
   - Each sub-agent gets a focused context slice, not the full conversation
   - Only structured results return to orchestrator
   - Prevents context contamination between parallel branches

3. **CriticalPathScheduler** — Latency-optimized execution
   - Implements the CriticalSteps metric from PARL paper:
     `CriticalSteps = sum(S_main(t) + max_i S_sub,i(t))`
   - Optimizes for shortest critical path, not just max parallelism
   - Balances decomposition granularity vs coordination overhead

4. **DecompositionEngine** — Task analysis and splitting
   - LLM-based task analysis: is this parallelizable? how to split?
   - Heuristics for common patterns (wide search, deep research, multi-file operations)
   - Anti-patterns: prevents serial collapse (defaulting to single-agent) and spurious parallelism (spawning agents without real work)

5. **SwarmRouter Integration** — Register as new SwarmType
   - `SwarmType.PARLOrchestrator` available via existing SwarmRouter
   - AutoSwarmBuilder can select it when appropriate
   - Compatible with existing Agent class and tool system

### What We're NOT Changing

- Agent class (use as-is, it's comprehensive)
- LiteLLM wrapper (already supports all providers we need)
- Existing swarm types (ConcurrentWorkflow, HierarchicalSwarm, etc.)
- Tool system (BaseTool, MCP client, function calling)
- Existing examples and tests

### What We're Extending

- SwarmRouter — add PARLOrchestrator as a new SwarmType
- AutoSwarmBuilder — teach it when to select PARL orchestration
- Prompts — add orchestrator/decomposition prompts

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Base framework | kyegomez/swarms | 20+ patterns, mature Agent class, LiteLLM multi-provider, production-ready |
| Language | Python | Framework is Python, ecosystem is Python, no viable TS alternative |
| Orchestrator model | Configurable (default: claude-sonnet-4-5 or gpt-4.1) | Strong reasoning needed for decomposition; user picks based on cost/quality |
| Sub-agent model | Configurable (default: cheaper model) | Sub-agents do focused work; don't need frontier models |
| Context isolation | Per-sub-agent context objects | Prevents cross-contamination, reduces token usage |
| Parallelism approach | Explicit heuristics, not trained | Can't retrain model weights; prompted decomposition gets 80-90% of the way |
| Extension pattern | New classes inheriting BaseSwarm | Clean separation, doesn't break existing code |

## Architecture

```
User Task
    |
    v
PARLOrchestrator (new)
    |
    +-- DecompositionEngine (new)
    |       Analyzes task, produces sub-task graph
    |
    +-- CriticalPathScheduler (new)
    |       Determines execution order, parallel groups
    |
    +-- For each parallel group:
    |       +-- ContextShardingManager (new)
    |       |       Creates isolated context per sub-agent
    |       |
    |       +-- Agent (existing, from swarms)
    |       |       Executes sub-task with tools
    |       |       Returns structured results only
    |       |
    |       +-- (runs concurrently via existing multi_agent_exec)
    |
    +-- ResultAggregator (new)
    |       Merges sub-agent outputs
    |       Flags contradictions
    |       Identifies gaps
    |
    +-- Optional: Loop back to DecompositionEngine if gaps found
    |
    v
Final synthesized output
```

## Existing Framework We Build On

Already implemented in kyegomez/swarms (DO NOT rewrite):

- **Agent** (252KB) — Full LLM agent with tools, memory, streaming, autonomous loops, 60+ config params
- **BaseSwarm** — Abstract base with run(), step(), add_agent(), broadcast(), assign_task()
- **SwarmRouter** — Pattern selection across 14+ swarm types
- **AutoSwarmBuilder** — LLM-driven agent generation and pattern selection
- **ConcurrentWorkflow** — Thread-based parallel execution (we use this under the hood)
- **multi_agent_exec** — Low-level concurrent utils: run_agents_concurrently(), batched execution
- **LiteLLM wrapper** — 100+ models across OpenAI, Anthropic, Google, Groq, Mistral, Deepseek, Ollama, etc.
- **BaseTool + MCP client** — Tool system with function calling, MCP protocol support
- **GraphWorkflow** — DAG execution with NetworkX (possible alternative execution engine)

## File Structure (Additions Only)

```
swarms/structs/
    parl_orchestrator.py        # Main PARLOrchestrator class
    decomposition_engine.py     # Task analysis and splitting
    critical_path_scheduler.py  # Latency-optimized execution planning
    context_sharding.py         # Per-sub-agent context isolation
    result_aggregator.py        # Output merging and contradiction detection

swarms/prompts/
    parl_decomposition.py       # Prompts for task decomposition
    parl_synthesis.py           # Prompts for result aggregation

examples/
    parl_orchestrator/
        basic_research.py       # Wide search example
        deep_analysis.py        # Deep reasoning example
        code_review.py          # Multi-file parallel review

tests/
    test_parl_orchestrator.py
    test_decomposition_engine.py
    test_critical_path_scheduler.py
```

## Primary Use Cases

1. **Wide Research** — "Research competitor X across funding, reviews, pricing, team" → fans out to 6-12 sub-agents searching in parallel
2. **Deep Analysis** — "Analyze this codebase for security issues" → decomposes by module, each sub-agent reviews a section
3. **Document Processing** — "Summarize these 20 documents" → each sub-agent handles 2-3 docs, orchestrator synthesizes
4. **Fact Checking** — "Verify these 15 claims" → parallel verification, cross-reference contradictions

## Cost Model

Using cheap sub-agents (e.g., Llama 3.1 8B at ~$0.03/M tokens via DeepInfra) with a strong orchestrator (Claude Sonnet at ~$3/M in):

| Component | Tokens (est.) | Cost |
|-----------|---------------|------|
| Orchestrator decomposition | ~2K in, ~1K out | ~$0.01 |
| 8x Sub-agents (cheap) | ~3K in, ~2K out each | ~$0.003 |
| Result aggregation (strong) | ~10K in, ~3K out | ~$0.04 |
| **Total per run** | | **~$0.05** |

## Reference Material

- [Kimi K2.5 Technical Report (arxiv)](https://arxiv.org/html/2602.02276v1) — PARL architecture, reward function, critical steps metric
- [Leaked Kimi System Prompts & Tools](https://github.com/dnnyngyen/kimi-k2.5-prompts-tools) — Full OK Computer agent system, skill injection, 38 tools
- [Unofficial PARL Implementation](https://github.com/The-Swarm-Corporation/PARL) — Python reward function and CriticalStepsMetric
- [Kimi K2.5 HuggingFace](https://huggingface.co/moonshotai/Kimi-K2.5) — Model card, deployment configs
- [DataCamp Agent Swarm Guide](https://www.datacamp.com/tutorial/kimi-k2-agent-swarm-guide) — Practical examples

## PARL Paper Key Formulas (For Implementation Reference)

### Reward Function
```
r_PARL(x,y) = lambda1 * r_parallel + lambda2 * r_finish + r_perf(x,y)
```
- r_parallel: Encourages sub-agent spawning (prevents serial collapse)
- r_finish: Sub-agent completion rate (prevents spurious parallelism)
- r_perf: Actual task quality
- lambda1, lambda2 anneal to 0 during training (we use fixed heuristic weights)

### Critical Steps Metric
```
CriticalSteps = sum_t(S_main(t) + max_i S_sub,i(t))
```
- S_main(t): Orchestrator steps per stage
- max_i S_sub,i(t): Slowest sub-agent per parallel cohort
- Measures wall-clock latency, not total compute

## Upstream Sync

```bash
git remote add upstream https://github.com/kyegomez/swarms.git  # already done
git fetch upstream
git merge upstream/master  # periodically pull improvements
```
