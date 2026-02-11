---
name: parl-orchestrator
status: backlog
created: 2026-02-11T20:29:39Z
updated: 2026-02-11T20:29:39Z
progress: 0%
prd: none
github: https://github.com/biosphere-labs/swarms/issues/1
---

# Epic: parl-orchestrator

## Overview

Add a PARL-inspired dynamic orchestration layer to kyegomez/swarms that decomposes tasks into parallel sub-agent work with isolated contexts, mimicking Kimi K2.5's Agent Swarm behavior via explicit heuristics rather than trained weights. Five new modules, two new prompt files, registered as a new SwarmType.

## Primary Flow

1. User calls `PARLOrchestrator.run(task)` with a complex task
2. DecompositionEngine analyzes the task via LLM, produces a sub-task graph (parallel groups + dependencies)
3. CriticalPathScheduler orders execution to minimize wall-clock time
4. For each parallel group: ContextShardingManager creates isolated context per sub-agent, agents execute concurrently
5. ResultAggregator merges outputs, flags contradictions, identifies gaps
6. Optional: loop back if gaps found (max 2 iterations)
7. Return synthesized output

## Edge Cases

- Task too simple to parallelize → single-agent fallback (no decomposition overhead)
- Sub-agent fails or times out → collect partial results, flag incomplete
- All sub-agents return contradictory info → surface contradictions explicitly, don't silently resolve
- Token budget exceeded → stop spawning, synthesize what we have

## Technical Approach

### New Modules (swarms/structs/)

| Module | Responsibility |
|--------|---------------|
| `parl_orchestrator.py` | Main class, inherits BaseSwarm, wires everything together |
| `decomposition_engine.py` | LLM-based task analysis and splitting into sub-task graph |
| `critical_path_scheduler.py` | Orders sub-tasks by critical path, groups parallel cohorts |
| `context_sharding.py` | Creates isolated context per sub-agent, collects structured results |
| `result_aggregator.py` | Merges sub-agent outputs, cross-references, detects contradictions |

### New Prompts (swarms/prompts/)

| File | Purpose |
|------|---------|
| `parl_decomposition.py` | System prompt for decomposition: analyze task, output sub-task JSON |
| `parl_synthesis.py` | System prompt for result aggregation: merge, flag contradictions, synthesize |

### Integration Points

- Register `PARLOrchestrator` in SwarmRouter as new SwarmType
- Use existing `run_agents_concurrently()` from `multi_agent_exec.py` for parallel execution
- Use existing Agent class with configurable model (strong for orchestrator, cheap for sub-agents)
- Use existing LiteLLM wrapper for all LLM calls

## Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Inherit BaseSwarm | Yes | Clean integration with existing framework, SwarmRouter compatibility |
| Decomposition via prompting | LLM call with structured JSON output | Can't retrain weights; prompted decomposition is the practical alternative |
| Parallel execution | ThreadPoolExecutor via existing multi_agent_exec | Already battle-tested in the framework |
| Context isolation | New ContextShardingManager | Core differentiator from existing ConcurrentWorkflow (which shares context) |
| Sub-task graph format | Dict with parallel groups + dependencies | Simple, serializable, inspectable |
| Anti-collapse heuristics | Hardcoded checks in DecompositionEngine | Prevent serial collapse and spurious parallelism without trained reward |

## Task Breakdown

1. DecompositionEngine — LLM-based task splitting with structured JSON output
2. ContextShardingManager — Per-sub-agent context isolation and result collection
3. CriticalPathScheduler — Order sub-tasks by critical path, group parallel cohorts
4. ResultAggregator — Merge outputs, flag contradictions, identify gaps
5. Decomposition + synthesis prompts
6. PARLOrchestrator — Main class wiring all components, inheriting BaseSwarm
7. SwarmRouter integration — Register as new SwarmType
8. Examples — basic_research, deep_analysis, code_review
9. Tests — Unit tests for each new module
10. README update — Document new capability

## Dependencies

- Existing: BaseSwarm, Agent, multi_agent_exec, LiteLLM, SwarmRouter
- No new pip dependencies required (networkx already included for graph work)

## Risks

| Risk | Mitigation |
|------|------------|
| LLM decomposition quality varies | Iterate on prompts; fallback to single-agent if decomposition looks bad |
| Coordination overhead exceeds parallel gains for small tasks | Complexity threshold: skip decomposition for simple tasks |
| Sub-agent model too weak for useful work | Make model configurable; default to something reasonable |
| Breaking existing framework behavior | All additions are new files; no modifications to existing classes except SwarmRouter registration |

## Success Criteria

- [ ] `PARLOrchestrator.run(task)` decomposes and executes in parallel
- [ ] Sub-agents have isolated contexts (no cross-contamination)
- [ ] Wall-clock time measurably faster than serial for parallelizable tasks
- [ ] Contradictions between sub-agents surfaced in output
- [ ] Registered in SwarmRouter, selectable as SwarmType
- [ ] 3 working examples demonstrating different use cases
- [ ] No regressions in existing tests

## Reviewed Decisions

| Decision | Concern Raised | Rationale for Keeping |
|----------|----------------|----------------------|
| (none yet) | - | - |
