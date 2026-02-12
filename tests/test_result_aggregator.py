"""
Tests for ResultAggregator.

Pure-logic tests cover confidence computation, result formatting, and
empty-input handling. Integration tests (marked with skipif) call a real
LLM to verify synthesis, contradiction detection, and gap identification.
"""

import os
import pytest
from swarms.structs.result_aggregator import (
    ResultAggregator,
    StructuredResult,
    AggregatedOutput,
)


# ---------------------------------------------------------------------------
# AggregatedOutput model
# ---------------------------------------------------------------------------

class TestAggregatedOutputModel:
    """Test the AggregatedOutput model."""

    def test_defaults(self):
        output = AggregatedOutput(
            synthesized_answer="Answer here.",
            confidence=0.9,
        )
        assert output.contradictions == []
        assert output.gaps == []
        assert output.metadata == {}

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            AggregatedOutput(synthesized_answer="x", confidence=1.5)
        with pytest.raises(Exception):
            AggregatedOutput(synthesized_answer="x", confidence=-0.1)


# ---------------------------------------------------------------------------
# StructuredResult model
# ---------------------------------------------------------------------------

class TestStructuredResultModel:
    """Test the aggregator's StructuredResult model."""

    def test_required_fields(self):
        r = StructuredResult(
            agent_id="a1",
            sub_task_id="t1",
            output="Hello",
            confidence=0.8,
        )
        assert r.agent_id == "a1"
        assert r.output == "Hello"

    def test_metadata_default(self):
        r = StructuredResult(
            agent_id="a1", sub_task_id="t1", output="x", confidence=0.5
        )
        assert r.metadata == {}


# ---------------------------------------------------------------------------
# Pure-logic: confidence computation
# ---------------------------------------------------------------------------

class TestConfidenceComputation:
    """Test ResultAggregator._compute_confidence with real inputs."""

    def _aggregator(self):
        return ResultAggregator(synthesis_model="gpt-4o-mini")

    def test_no_results_returns_zero(self):
        agg = self._aggregator()
        assert agg._compute_confidence([], [], []) == 0.0

    def test_perfect_confidence_no_issues(self):
        agg = self._aggregator()
        results = [
            StructuredResult(agent_id="a1", sub_task_id="t1", output="x", confidence=1.0),
            StructuredResult(agent_id="a2", sub_task_id="t2", output="y", confidence=1.0),
        ]
        conf = agg._compute_confidence(results, contradictions=[], gaps=[])
        assert conf == pytest.approx(1.0)

    def test_contradictions_reduce_confidence(self):
        agg = self._aggregator()
        results = [
            StructuredResult(agent_id="a1", sub_task_id="t1", output="x", confidence=1.0),
        ]
        # 1 contradiction = 10% penalty
        conf_1 = agg._compute_confidence(results, contradictions=["c1"], gaps=[])
        assert conf_1 < 1.0
        assert conf_1 == pytest.approx(0.9, abs=0.01)

        # 5 contradictions = 50% penalty (capped)
        conf_5 = agg._compute_confidence(
            results, contradictions=["c1", "c2", "c3", "c4", "c5"], gaps=[]
        )
        assert conf_5 == pytest.approx(0.5, abs=0.01)

        # 10 contradictions should still cap at 50%
        conf_10 = agg._compute_confidence(
            results, contradictions=[f"c{i}" for i in range(10)], gaps=[]
        )
        assert conf_10 == pytest.approx(0.5, abs=0.01)

    def test_gaps_reduce_confidence(self):
        agg = self._aggregator()
        results = [
            StructuredResult(agent_id="a1", sub_task_id="t1", output="x", confidence=1.0),
        ]
        # 1 gap = 5% penalty
        conf = agg._compute_confidence(results, contradictions=[], gaps=["g1"])
        assert conf == pytest.approx(0.95, abs=0.01)

        # 6 gaps = 30% penalty (capped)
        conf_capped = agg._compute_confidence(
            results, contradictions=[], gaps=[f"g{i}" for i in range(6)]
        )
        assert conf_capped == pytest.approx(0.7, abs=0.01)

    def test_both_penalties_compound(self):
        agg = self._aggregator()
        results = [
            StructuredResult(agent_id="a1", sub_task_id="t1", output="x", confidence=1.0),
        ]
        # 2 contradictions (20%) + 2 gaps (10%)
        # final = 1.0 * (1 - 0.2) * (1 - 0.1) = 0.72
        conf = agg._compute_confidence(
            results, contradictions=["c1", "c2"], gaps=["g1", "g2"]
        )
        assert conf == pytest.approx(0.72, abs=0.01)

    def test_low_sub_agent_confidence_propagates(self):
        agg = self._aggregator()
        results = [
            StructuredResult(agent_id="a1", sub_task_id="t1", output="x", confidence=0.5),
            StructuredResult(agent_id="a2", sub_task_id="t2", output="y", confidence=0.5),
        ]
        # avg confidence = 0.5, no penalties
        conf = agg._compute_confidence(results, contradictions=[], gaps=[])
        assert conf == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Pure-logic: result formatting
# ---------------------------------------------------------------------------

class TestResultFormatting:
    """Test ResultAggregator._format_results."""

    def test_format_includes_all_agents(self):
        agg = ResultAggregator(synthesis_model="gpt-4o-mini")
        results = [
            StructuredResult(agent_id="agent-A", sub_task_id="task-1", output="Output A", confidence=0.9),
            StructuredResult(agent_id="agent-B", sub_task_id="task-2", output="Output B", confidence=0.8),
        ]
        formatted = agg._format_results(results)
        assert "agent-A" in formatted
        assert "agent-B" in formatted
        assert "Output A" in formatted
        assert "Output B" in formatted
        assert "0.90" in formatted  # confidence formatted
        assert "0.80" in formatted


# ---------------------------------------------------------------------------
# Pure-logic: empty results handling
# ---------------------------------------------------------------------------

class TestEmptyResults:
    """Test aggregation with empty or edge-case inputs."""

    def test_aggregate_empty_results(self):
        agg = ResultAggregator(synthesis_model="gpt-4o-mini")
        output = agg.aggregate(results=[], original_task="Do something.")
        assert isinstance(output, AggregatedOutput)
        assert output.confidence == 0.0
        assert "No sub-agent results" in output.synthesized_answer
        assert len(output.gaps) > 0


# ---------------------------------------------------------------------------
# Integration tests — real LLM calls
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Requires OPENAI_API_KEY for LLM calls",
)
class TestResultAggregatorIntegration:
    """Integration tests with real LLM synthesis."""

    def test_synthesis_produces_coherent_output(self):
        """Feed pre-built results and verify synthesis produces real text."""
        agg = ResultAggregator(synthesis_model="gpt-4o-mini", temperature=0.3)
        results = [
            StructuredResult(
                agent_id="solar-agent",
                sub_task_id="solar-research",
                output=(
                    "Solar energy advantages: renewable, decreasing costs ($0.05/kWh), "
                    "low maintenance. Disadvantages: intermittent, requires storage, "
                    "land-intensive for utility scale."
                ),
                confidence=0.9,
            ),
            StructuredResult(
                agent_id="wind-agent",
                sub_task_id="wind-research",
                output=(
                    "Wind energy advantages: mature technology, competitive costs ($0.04/kWh), "
                    "good for offshore. Disadvantages: variable output, noise concerns, "
                    "impact on wildlife, requires backup capacity."
                ),
                confidence=0.85,
            ),
        ]
        output = agg.aggregate(
            results=results,
            original_task="Compare solar and wind energy: costs, efficiency, environmental impact.",
        )

        assert isinstance(output, AggregatedOutput)
        assert len(output.synthesized_answer) > 100
        assert output.confidence > 0.0
        # The synthesis should reference both energy types
        answer_lower = output.synthesized_answer.lower()
        assert "solar" in answer_lower
        assert "wind" in answer_lower

    def test_contradictions_detected_when_present(self):
        """Feed results with deliberate contradictions; aggregator should flag them."""
        agg = ResultAggregator(synthesis_model="gpt-4o-mini", temperature=0.1)
        results = [
            StructuredResult(
                agent_id="agent-A",
                sub_task_id="fact-check-1",
                output="The company was founded in 2010 by John Smith in San Francisco.",
                confidence=1.0,
            ),
            StructuredResult(
                agent_id="agent-B",
                sub_task_id="fact-check-2",
                output="The company was founded in 2015 by Jane Doe in New York.",
                confidence=1.0,
            ),
        ]
        output = agg.aggregate(
            results=results,
            original_task="When and where was the company founded, and by whom?",
        )

        # Should have at least one contradiction detected
        assert len(output.contradictions) > 0
        # Confidence should be reduced due to contradictions
        assert output.confidence < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
