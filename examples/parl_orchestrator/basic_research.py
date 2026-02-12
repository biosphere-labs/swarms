"""
PARL Orchestrator — Basic Research Example

Demonstrates how PARLOrchestrator dynamically decomposes a wide research task
into parallel sub-agents, each investigating a different aspect of the topic.

This example shows:
- Automatic task decomposition into parallel research sub-tasks
- Sub-agents executing concurrently with isolated contexts
- Result synthesis with contradiction detection and gap identification
- Both direct PARLOrchestrator usage and SwarmRouter integration
"""

import os
from swarms.structs.parl_orchestrator import PARLOrchestrator
from swarms.structs.swarm_router import SwarmRouter


def main():
    # Ensure API key is set (required for LLM calls)
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        return

    # Example 1: Direct usage of PARLOrchestrator
    print("=" * 80)
    print("Example 1: Direct PARLOrchestrator Usage")
    print("=" * 80)

    orchestrator = PARLOrchestrator(
        name="research-orchestrator",
        description="Parallel research orchestrator for wide information gathering",
        orchestrator_model="gpt-4o-mini",  # Strong model for decomposition/synthesis
        sub_agent_model="gpt-4o-mini",     # Cheaper model for sub-agents (can use same)
        max_parallel=8,                     # Up to 8 concurrent sub-agents
        max_iterations=2,                   # Allow 2 iterations if gaps found
        timeout=300,                        # 5 minute overall timeout
        sub_agent_timeout=120,              # 2 minute per-agent timeout
    )

    # Wide research task that should decompose into parallel sub-tasks
    task = """
    Research the company Anthropic:
    - Funding history and investors
    - Product offerings and capabilities
    - Key team members and background
    - Recent news and announcements
    - Market positioning vs competitors
    - Technical approach and innovations

    Provide a comprehensive overview with sources.
    """

    print("\nTask:")
    print(task)
    print("\nExecuting... (this may take 1-2 minutes)\n")

    result = orchestrator.run(task)

    print("\n" + "=" * 80)
    print("Result:")
    print("=" * 80)
    print(result)

    # Example 2: Using SwarmRouter to dynamically select PARL orchestration
    print("\n\n" + "=" * 80)
    print("Example 2: SwarmRouter with PARLOrchestrator")
    print("=" * 80)

    router = SwarmRouter(
        name="research-router",
        description="Routes research tasks to appropriate swarm type",
        swarm_type="PARLOrchestrator",  # Explicitly select PARL orchestration
    )

    # Different research task
    task2 = """
    Analyze the landscape of open-source LLM frameworks:
    - Popular frameworks (LangChain, LlamaIndex, Semantic Kernel, etc.)
    - Key features and differentiators
    - Community size and activity
    - Use cases and adoption
    - Strengths and weaknesses
    """

    print("\nTask:")
    print(task2)
    print("\nExecuting via SwarmRouter...\n")

    result2 = router.run(task2)

    print("\n" + "=" * 80)
    print("Result:")
    print("=" * 80)
    print(result2)


if __name__ == "__main__":
    main()
