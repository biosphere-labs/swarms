"""
Fact-Check Debate Module

Multi-round fact-checking using the DebateWithJudge pattern. The researcher
presents findings, the fact-checker challenges claims with independent verification,
and a judge synthesizes the verified version.

This replaces one-way fact-checking with actual inter-agent discussion.
"""

from typing import List, Optional, Callable

from swarms.structs.agent import Agent
from swarms.structs.debate_with_judge import DebateWithJudge


RESEARCHER_SYSTEM_PROMPT = """You are a research specialist presenting your findings.

Your Role:
- Present the research findings clearly and comprehensively
- Support claims with specific data points (numbers, dates, names, statistics)
- Cite sources when available
- Defend your findings when challenged, but acknowledge if verification is uncertain

When the fact-checker challenges a claim:
- Provide additional context or sources if available
- Acknowledge if a claim cannot be verified
- Revise claims that are proven incorrect

Your goal is to present accurate, well-supported research findings."""


FACT_CHECKER_SYSTEM_PROMPT = """You are a fact-checker independently verifying research claims.

Your Role:
- Identify every factual claim in the research (numbers, dates, names, statistics)
- Use your search tools to independently verify each claim
- Challenge claims that cannot be verified or are incorrect
- Provide correct data with sources when you find errors

Verification Process:
1. Extract specific factual claims from the research
2. For each claim, search independently to verify
3. Compare your findings with the researcher's claims
4. Flag discrepancies, unverifiable claims, or missing sources

When challenging:
- Be specific about which claim is problematic
- Provide the correct information if available
- Cite your sources for corrections

Your goal is to ensure all claims are accurate and verifiable."""


JUDGE_SYSTEM_PROMPT = """You are an impartial judge synthesizing verified research.

Your Role:
- Evaluate the researcher's findings and the fact-checker's challenges
- Determine which claims are verified, which need correction, and which are unverifiable
- Synthesize a final version that includes only verified information
- Clearly mark anything that remains unverified

Synthesis Guidelines:
1. Keep verified claims from the researcher as-is
2. Apply corrections from the fact-checker when substantiated
3. Mark unverifiable claims as [UNVERIFIED]
4. Preserve the structure and flow of the original research
5. Ensure all corrections are properly sourced

Your goal is to produce an accurate, verified version of the research."""


class FactCheckDebate:
    """
    Fact-checking via multi-round debate between researcher and fact-checker.

    Uses the DebateWithJudge pattern with research-specific prompts:
    - Researcher (Pro) presents findings
    - Fact-checker (Con) challenges and verifies claims
    - Judge synthesizes the verified version

    This provides actual inter-agent discussion rather than one-way verification.

    Args:
        model_name: Model to use for all three agents (default: gpt-4o-mini)
        max_loops: Number of debate rounds (default: 2)
        tools: Optional list of tools to give agents (e.g., serper_search for verification)
        api_key_provider: Optional callable that returns API keys for round-robin
        verbose: Whether to enable verbose logging (default: False)

    Example:
        >>> from swarms.tools.serper_search import serper_search
        >>>
        >>> debate = FactCheckDebate(
        ...     model_name="deepinfra/Qwen/Qwen2.5-72B-Instruct",
        ...     tools=[serper_search],
        ...     max_loops=2,
        ... )
        >>>
        >>> verified = debate.verify(
        ...     research_output="Acme Corp raised $50M in Series B in 2023..."
        ... )
        >>> print(verified)
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        max_loops: int = 2,
        tools: Optional[List[Callable]] = None,
        api_key_provider: Optional[Callable] = None,
        verbose: bool = False,
    ):
        self.model_name = model_name
        self.max_loops = max_loops
        self.tools = tools or []
        self._api_key_provider = api_key_provider
        self.verbose = verbose

    def _get_api_key(self) -> Optional[str]:
        """Get API key from provider if available."""
        if self._api_key_provider:
            return self._api_key_provider()
        return None

    def verify(self, research_output: str) -> str:
        """
        Verify research output via multi-round fact-checking debate.

        Args:
            research_output: The research findings to fact-check

        Returns:
            str: The verified version with corrections and [UNVERIFIED] markers
        """
        # Create the three debate agents
        # Agents with tools need multiple loops to call tools and process results
        researcher = Agent(
            agent_name="Researcher",
            system_prompt=RESEARCHER_SYSTEM_PROMPT,
            model_name=self.model_name,
            max_loops=3 if self.tools else 1,  # Need loops for tool calls
            output_type="final",
            verbose=False,
            tools=self.tools if self.tools else None,
            llm_api_key=self._get_api_key(),
        )

        fact_checker = Agent(
            agent_name="FactChecker",
            system_prompt=FACT_CHECKER_SYSTEM_PROMPT,
            model_name=self.model_name,
            max_loops=3 if self.tools else 1,  # Need loops for tool calls
            output_type="final",
            verbose=False,
            tools=self.tools if self.tools else None,
            llm_api_key=self._get_api_key(),
        )

        judge = Agent(
            agent_name="Judge",
            system_prompt=JUDGE_SYSTEM_PROMPT,
            model_name=self.model_name,
            max_loops=1,  # Judge just synthesizes, no tools needed
            output_type="final",
            verbose=False,
            tools=None,  # Judge doesn't need tools, just synthesizes
            llm_api_key=self._get_api_key(),
        )

        # Create the debate
        debate = DebateWithJudge(
            pro_agent=researcher,
            con_agent=fact_checker,
            judge_agent=judge,
            max_loops=self.max_loops,
            verbose=self.verbose,
        )

        # Run the debate with the research output as the topic
        task = (
            f"Verify the following research output. The researcher will present their "
            f"findings, the fact-checker will independently verify claims, and the judge "
            f"will synthesize the verified version.\n\n"
            f"Research Output:\n{research_output}"
        )

        result = debate.run(task=task)

        # The result is the synthesized, verified version from the judge
        return str(result) if result else research_output
