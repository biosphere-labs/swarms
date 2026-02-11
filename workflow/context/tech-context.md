# Technical Context

## Architecture
Fork of kyegomez/swarms with new modules added under swarms/structs/ and swarms/prompts/. Core additions: PARLOrchestrator, DecompositionEngine, CriticalPathScheduler, ContextShardingManager, ResultAggregator. All inherit from or compose with existing BaseSwarm and Agent classes.

## Key Patterns
- BaseSwarm inheritance for new orchestration types
- SwarmRouter registration for pattern selection
- LiteLLM for model-agnostic LLM calls
- ThreadPoolExecutor / asyncio for parallel sub-agent execution
- Context sharding: sub-agents get focused slices, only results flow back

## Dependencies
- swarms (this repo, the base framework)
- litellm (already included)
- networkx (already included, for DAG scheduling)
- pydantic (already included, for schemas)
