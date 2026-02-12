"""
PARL Orchestrator — Deep Analysis Example

Demonstrates how PARLOrchestrator handles deep, multi-perspective analysis
by decomposing a complex analytical task into parallel expert reviews.

This example shows:
- Decomposition of a complex analysis into multiple perspectives
- Each sub-agent acting as a specialized reviewer/analyst
- Parallel execution of deep analytical work
- Cross-referencing and contradiction detection in synthesis
- Handling of sequential dependencies (if needed)
"""

import os
from swarms.structs.parl_orchestrator import PARLOrchestrator
from swarms.structs.swarm_router import SwarmRouter


def main():
    # Ensure API key is set
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        return

    # Example 1: Multi-perspective analysis with PARLOrchestrator
    print("=" * 80)
    print("Example 1: Deep Multi-Perspective Analysis")
    print("=" * 80)

    orchestrator = PARLOrchestrator(
        name="analysis-orchestrator",
        description="Deep analysis orchestrator with parallel expert perspectives",
        orchestrator_model="gpt-4o-mini",
        sub_agent_model="gpt-4o-mini",
        max_parallel=6,          # Multiple perspectives in parallel
        max_iterations=2,         # Re-analyze if gaps found
        timeout=600,             # 10 minute timeout for deep analysis
        sub_agent_timeout=180,   # 3 minutes per perspective
    )

    # Complex analysis task requiring multiple expert perspectives
    task = """
    Analyze the following business proposal for launching a new AI-powered
    code review service:

    Proposal:
    - SaaS platform offering automated code review via LLM agents
    - Target market: Mid-size tech companies (50-500 engineers)
    - Pricing: $50/user/month
    - Key features: Security scanning, best practice suggestions, performance analysis
    - Tech stack: Python backend, React frontend, Claude/GPT-4 for analysis
    - Go-to-market: Direct sales + product-led growth

    Provide analysis from multiple perspectives:
    - Market opportunity and competitive landscape
    - Technical feasibility and risks
    - Financial projections and unit economics
    - Go-to-market strategy viability
    - Potential challenges and mitigations
    - Recommendation: Go/No-Go and why

    Identify any contradictions or gaps in the analysis.
    """

    print("\nTask:")
    print(task[:300] + "..." if len(task) > 300 else task)
    print("\nExecuting deep analysis... (this may take 2-3 minutes)\n")

    result = orchestrator.run(task)

    print("\n" + "=" * 80)
    print("Result:")
    print("=" * 80)
    print(result)

    # Example 2: Technical architecture review
    print("\n\n" + "=" * 80)
    print("Example 2: Technical Architecture Deep Dive")
    print("=" * 80)

    router = SwarmRouter(
        name="architecture-reviewer",
        swarm_type="PARLOrchestrator",
    )

    task2 = """
    Review this microservices architecture design:

    System: E-commerce platform
    Services:
    - API Gateway (Node.js, rate limiting, auth)
    - User Service (Python, PostgreSQL, user profiles/auth)
    - Product Catalog (Go, MongoDB, product data/search)
    - Order Service (Python, PostgreSQL, order processing)
    - Payment Service (Node.js, Stripe integration, PCI compliance)
    - Notification Service (Python, SendGrid/Twilio, async processing)
    - Analytics Service (Python, Clickhouse, event tracking)

    Infrastructure:
    - Kubernetes on AWS EKS
    - RabbitMQ for async messaging
    - Redis for caching
    - CloudFront CDN
    - S3 for static assets

    Analyze from these perspectives:
    - Scalability and performance bottlenecks
    - Security vulnerabilities and compliance gaps
    - Data consistency and transaction handling
    - Operational complexity and monitoring
    - Cost optimization opportunities
    - Failure modes and resilience

    Flag any contradictory design decisions or critical gaps.
    """

    print("\nTask:")
    print(task2[:300] + "..." if len(task2) > 300 else task2)
    print("\nExecuting architecture review...\n")

    result2 = router.run(task2)

    print("\n" + "=" * 80)
    print("Result:")
    print("=" * 80)
    print(result2)


if __name__ == "__main__":
    main()
