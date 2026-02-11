# Project Overview

## Purpose
PARL-inspired multi-agent swarm orchestration layer built on top of kyegomez/swarms. Clones the dynamic task decomposition and parallel execution behavior of Kimi K2.5's Agent Swarm without requiring model retraining.

## Key Features
- Dynamic task decomposition (LLM-driven, not hand-coded workflows)
- Parallel sub-agent execution with context isolation (context sharding)
- Critical-path-aware scheduling (optimize wall-clock time, not just total work)
- Anti-collapse heuristics (prevent serial collapse and spurious parallelism)
- Compatible with existing swarms Agent class and 100+ LLM providers via LiteLLM

## Tech Stack
- Python 3.11+
- kyegomez/swarms (base framework)
- LiteLLM (multi-provider LLM access)
