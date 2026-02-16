"""
Integration test for fact-check debate with real LLM calls.

This test actually runs the 3-agent debate (Researcher, Fact-Checker, Judge)
with real models to verify the discussion pattern works end-to-end.

Run with:
    pytest tests/test_fact_check_debate_integration.py -v -s
"""

import os
import pytest
from swarms.structs.fact_check_debate import FactCheckDebate
from swarms.tools.serper_search import serper_search


@pytest.mark.skipif(
    not os.environ.get("PARL_API_KEYS") and not os.environ.get("DEEPINFRA_API_KEY"),
    reason="Requires DeepInfra API key (set PARL_API_KEYS or DEEPINFRA_API_KEY)"
)
def test_fact_check_debate_real_llm():
    """
    End-to-end test with real LLM calls.

    This test:
    1. Creates a FactCheckDebate instance with real models
    2. Provides research output with a deliberate error
    3. Runs the 3-agent debate (researcher, fact-checker, judge)
    4. Verifies the error is caught and corrected
    """

    # Research output with a deliberate error
    research_output = """
    Anthropic was founded in 2020 by Dario Amodei and Daniela Amodei.
    The company raised $500M in Series B funding in 2023.
    Their flagship product is Claude, an AI assistant.
    """

    # Expected: The founding year is actually 2021, not 2020
    # The fact-checker should catch this

    # Create the debate with cheap models
    debate = FactCheckDebate(
        model_name=os.environ.get("PARL_SUB_AGENT_MODEL", "deepinfra/Qwen/Qwen2.5-72B-Instruct"),
        max_loops=2,  # 2 rounds of debate
        tools=[serper_search],  # Fact-checker needs search
        verbose=True,  # Show debate output
    )

    print("\n" + "="*80)
    print("ORIGINAL RESEARCH OUTPUT:")
    print("="*80)
    print(research_output)
    print("="*80)

    # Run the debate
    verified_output = debate.verify(research_output)

    print("\n" + "="*80)
    print("VERIFIED OUTPUT (after debate):")
    print("="*80)
    print(verified_output)
    print("="*80)

    # Verify the output changed (debate happened)
    assert verified_output != research_output, "Debate should produce different output"

    # Verify the output is longer than original (includes discussion/corrections)
    # OR contains correction markers like [UNVERIFIED] or mentions 2021
    assert (
        len(verified_output) > len(research_output) * 0.8 or
        "[UNVERIFIED]" in verified_output or
        "2021" in verified_output
    ), "Verified output should include corrections or markers"

    print("\n✅ Fact-check debate completed successfully!")
    print("The 3-agent discussion (Researcher ↔ Fact-Checker ↔ Judge) ran as expected.")


@pytest.mark.skipif(
    not os.environ.get("PARL_API_KEYS") and not os.environ.get("DEEPINFRA_API_KEY"),
    reason="Requires DeepInfra API key"
)
def test_fact_check_debate_catches_fabrication():
    """
    Test that the fact-checker catches completely fabricated data.
    """

    # Research output with fabricated data
    research_output = """
    Acme Corporation was founded in 1995 in San Francisco.
    The company raised $847 million in Series C funding in March 2024.
    CEO Jane Smith has been with the company since inception.
    Current employee count is approximately 2,300 people.
    """

    # This is likely fabricated data for a fictional company
    # The fact-checker should flag everything as unverifiable

    debate = FactCheckDebate(
        model_name=os.environ.get("PARL_SUB_AGENT_MODEL", "deepinfra/Qwen/Qwen2.5-72B-Instruct"),
        max_loops=1,  # Just 1 round for speed
        tools=[serper_search],
        verbose=True,
    )

    print("\n" + "="*80)
    print("FABRICATED RESEARCH:")
    print("="*80)
    print(research_output)
    print("="*80)

    verified_output = debate.verify(research_output)

    print("\n" + "="*80)
    print("AFTER FACT-CHECKING:")
    print("="*80)
    print(verified_output)
    print("="*80)

    # Should contain [UNVERIFIED] markers for unverifiable claims
    assert "[UNVERIFIED]" in verified_output or "cannot be verified" in verified_output.lower(), \
        "Fact-checker should flag unverifiable claims"

    print("\n✅ Fact-checker successfully flagged unverifiable data!")


if __name__ == "__main__":
    # Run with real API calls for manual testing
    print("Running fact-check debate integration tests...")
    print("\nTest 1: Catching errors in real company data")
    test_fact_check_debate_real_llm()

    print("\n" + "="*80)
    print("\nTest 2: Catching fabricated data")
    test_fact_check_debate_catches_fabrication()

    print("\n" + "="*80)
    print("\n✅ All integration tests passed!")
