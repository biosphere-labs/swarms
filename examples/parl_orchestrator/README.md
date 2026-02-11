# PARL Orchestrator Examples

Three examples demonstrating PARLOrchestrator usage for different parallel execution patterns.

## Examples

### 1. basic_research.py — Wide Search

Demonstrates wide parallel research across multiple aspects of a topic.

**Use case**: Research a company, technology, or topic across multiple dimensions simultaneously.

**Pattern**: Fan-out to 6-12 sub-agents, each researching a different aspect (funding, team, product, news, etc.).

**Run**:
```bash
python examples/parl_orchestrator/basic_research.py
```

### 2. deep_analysis.py — Multi-Perspective Analysis

Demonstrates deep analysis with parallel expert reviews from different perspectives.

**Use case**: Analyze a proposal, architecture, or decision from multiple expert viewpoints.

**Pattern**: Multiple sub-agents acting as specialized reviewers (security expert, performance analyst, business analyst, etc.), with contradiction detection and gap identification.

**Run**:
```bash
python examples/parl_orchestrator/deep_analysis.py
```

### 3. code_review.py — Parallel Code Review

Demonstrates parallel review of multiple files or modules.

**Use case**: Review a multi-file codebase for security, performance, and quality issues.

**Pattern**: Each sub-agent reviews a specific file or module, orchestrator synthesizes findings and identifies cross-file issues.

**Run**:
```bash
python examples/parl_orchestrator/code_review.py
```

## Requirements

Set either `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` environment variable:

```bash
export OPENAI_API_KEY="sk-..."
# or
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Usage Patterns

### Direct PARLOrchestrator Usage

```python
from swarms.structs.parl_orchestrator import PARLOrchestrator

orchestrator = PARLOrchestrator(
    name="my-orchestrator",
    orchestrator_model="gpt-4o-mini",  # Strong model for decomposition
    sub_agent_model="gpt-4o-mini",     # Cheaper model for sub-agents
    max_parallel=8,                     # Concurrent sub-agents
    max_iterations=2,                   # Gap-fill iterations
)

result = orchestrator.run("Your complex task here")
```

### SwarmRouter Integration

```python
from swarms.structs.swarm_router import SwarmRouter

router = SwarmRouter(
    name="task-router",
    swarm_type="PARLOrchestrator",
)

result = router.run("Your complex task here")
```

## What Each Example Demonstrates

| Example | Decomposition Pattern | Sub-Agent Count | Use Case |
|---------|----------------------|-----------------|----------|
| basic_research | Wide fan-out | 6-12 | Parallel information gathering |
| deep_analysis | Multi-perspective | 4-8 | Expert review from different angles |
| code_review | Per-file/module | 3-10 | Parallel code analysis |

## Expected Output

Each example shows:
1. Task decomposition (how PARLOrchestrator splits the work)
2. Parallel sub-agent execution (with timing)
3. Result synthesis with contradiction detection
4. Final comprehensive answer

Typical execution time: 1-3 minutes depending on task complexity and model speed.
