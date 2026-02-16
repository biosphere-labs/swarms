"""
Tests for the Iterative Cross-Model Fact Check Module

Tests cover:
- Claim extraction from reviews
- Cross-model verification (different model than source)
- Arbitration of disputed claims
- FactCheckReport generation
- JSON parsing robustness
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from swarms.structs.iterative_fact_check import (
    IterativeFactCheck,
    Claim,
    VerifiedClaim,
    FactCheckReport,
    _model_family,
    _parse_json_array,
)
from swarms.structs.result_aggregator import StructuredResult


# --- Test data ---

SAMPLE_REVIEWS = [
    StructuredResult(
        agent_id="reviewer-logic",
        sub_task_id="Logic Auditor",
        output=(
            "The article claims that 73% of developers use AI tools daily. "
            "The METR study was published in 2024 by Anthropic. "
            "The author states that detection tools give 18-month timeline."
        ),
        confidence=0.8,
        metadata={"model": "deepinfra/deepseek-ai/DeepSeek-V3.2", "persona": "Logic Auditor"},
    ),
    StructuredResult(
        agent_id="reviewer-structure",
        sub_task_id="Structure Reviewer",
        output=(
            "The Da Vinci Code was published by Simon & Schuster. "
            "The article's hook is 180 characters, within the 210-char limit."
        ),
        confidence=0.8,
        metadata={"model": "deepinfra/google/gemini-2.5-pro", "persona": "Structure Reviewer"},
    ),
]


class TestClaim:
    """Tests for the Claim data model."""

    def test_creates_claim(self):
        claim = Claim(
            text="73% of developers use AI",
            source_agent_id="reviewer-logic",
            source_model="deepseek-ai/DeepSeek-V3.2",
            context="The article claims that 73% of developers use AI tools daily.",
        )
        assert claim.text == "73% of developers use AI"
        assert claim.source_model == "deepseek-ai/DeepSeek-V3.2"


class TestVerifiedClaim:
    """Tests for the VerifiedClaim data model."""

    def test_creates_verified_claim(self):
        claim = Claim(text="test", source_agent_id="a", source_model="m")
        vc = VerifiedClaim(
            claim=claim,
            verification_model="other-model",
            status="verified",
            evidence="Found supporting data",
        )
        assert vc.status == "verified"
        assert vc.correction is None

    def test_creates_disputed_claim_with_correction(self):
        claim = Claim(text="test", source_agent_id="a", source_model="m")
        vc = VerifiedClaim(
            claim=claim,
            verification_model="other-model",
            status="disputed",
            evidence="Found contradicting data",
            correction="The correct value is 42%",
        )
        assert vc.status == "disputed"
        assert vc.correction == "The correct value is 42%"


class TestFactCheckReport:
    """Tests for the FactCheckReport data model."""

    def test_empty_report(self):
        report = FactCheckReport(rounds_completed=1)
        assert len(report.verified) == 0
        assert len(report.disputed) == 0
        assert "0" in report.summary()

    def test_report_summary_includes_corrections(self):
        claim = Claim(text="wrong claim", source_agent_id="a", source_model="m")
        vc = VerifiedClaim(
            claim=claim, verification_model="checker",
            status="disputed", evidence="wrong", correction="correct value"
        )
        report = FactCheckReport(
            corrections=[vc], disputed=[vc], rounds_completed=2
        )
        summary = report.summary()
        assert "Corrections" in summary
        assert "correct value" in summary


class TestModelFamily:
    """Tests for _model_family helper."""

    def test_extracts_deepseek(self):
        assert _model_family("deepinfra/deepseek-ai/DeepSeek-V3.2") == "deepseek"

    def test_extracts_gemini(self):
        assert _model_family("deepinfra/google/gemini-2.5-pro") == "gemini"

    def test_extracts_qwen(self):
        assert _model_family("deepinfra/Qwen/Qwen3-235B-A22B") == "qwen"

    def test_extracts_llama(self):
        assert _model_family("deepinfra/meta-llama/Llama-4-Maverick") == "llama"

    def test_fallback_for_unknown(self):
        result = _model_family("unknown/custom-model-v1")
        assert result == "custom"


class TestParseJsonArray:
    """Tests for _parse_json_array helper."""

    def test_parses_clean_json(self):
        result = _parse_json_array('[{"a": 1}, {"b": 2}]')
        assert len(result) == 2

    def test_parses_with_markdown(self):
        result = _parse_json_array('```json\n[{"a": 1}]\n```')
        assert len(result) == 1

    def test_extracts_from_text(self):
        result = _parse_json_array('Here: [{"a": 1}] end')
        assert len(result) == 1

    def test_returns_empty_for_garbage(self):
        assert _parse_json_array("not json") == []

    def test_returns_empty_for_object(self):
        assert _parse_json_array('{"not": "an array"}') == []


class TestIterativeFactCheck:
    """Tests for the IterativeFactCheck class."""

    def test_init_defaults(self):
        fc = IterativeFactCheck()
        assert fc.max_rounds == 2
        assert fc.max_parallel == 5

    def test_init_clamps_rounds(self):
        fc = IterativeFactCheck(max_rounds=10)
        assert fc.max_rounds == 3  # clamped to max 3

        fc = IterativeFactCheck(max_rounds=0)
        assert fc.max_rounds == 1  # clamped to min 1

    def test_pick_different_model(self):
        fc = IterativeFactCheck(
            available_models=[
                "deepinfra/deepseek-ai/DeepSeek-V3.2",
                "deepinfra/google/gemini-2.5-pro",
                "deepinfra/Qwen/Qwen3-235B-A22B",
            ]
        )
        # Should pick a non-deepseek model
        result = fc._pick_different_model("deepinfra/deepseek-ai/DeepSeek-V3.2")
        assert "deepseek" not in result.lower()

    def test_pick_different_model_excluding(self):
        fc = IterativeFactCheck(
            available_models=[
                "deepinfra/deepseek-ai/DeepSeek-V3.2",
                "deepinfra/google/gemini-2.5-pro",
                "deepinfra/Qwen/Qwen3-235B-A22B",
            ]
        )
        exclude = {
            "deepinfra/deepseek-ai/DeepSeek-V3.2",
            "deepinfra/google/gemini-2.5-pro",
        }
        result = fc._pick_different_model_excluding(exclude)
        assert result == "deepinfra/Qwen/Qwen3-235B-A22B"

    @patch("swarms.structs.iterative_fact_check.LiteLLMBackend")
    def test_round1_extract_claims(self, mock_backend_class):
        """Should extract claims from review results."""
        mock_backend = MagicMock()
        mock_backend.call.return_value = json.dumps([
            {"claim": "73% of developers use AI tools daily", "context": "The article claims..."},
            {"claim": "METR study was published in 2024", "context": "The METR study..."},
        ])
        mock_backend_class.return_value = mock_backend

        fc = IterativeFactCheck()
        claims = fc._round1_extract_claims(SAMPLE_REVIEWS)

        assert len(claims) >= 2
        assert any("73%" in c.text for c in claims)

    @patch("swarms.structs.iterative_fact_check.Agent")
    def test_round2_uses_different_model(self, mock_agent_class):
        """Cross-verify should use a different model than the source."""
        mock_agent = MagicMock()
        mock_agent.run.return_value = json.dumps([
            {"claim_index": 0, "status": "verified", "evidence": "confirmed", "correction": None},
        ])
        mock_agent_class.return_value = mock_agent

        fc = IterativeFactCheck(
            available_models=[
                "deepinfra/deepseek-ai/DeepSeek-V3.2",
                "deepinfra/google/gemini-2.5-pro",
            ]
        )
        claims = [
            Claim(
                text="test claim",
                source_agent_id="reviewer-logic",
                source_model="deepinfra/deepseek-ai/DeepSeek-V3.2",
            )
        ]
        results = fc._round2_cross_verify(claims)

        # Should have created an agent with a non-deepseek model
        call_kwargs = mock_agent_class.call_args.kwargs
        assert "deepseek" not in call_kwargs["model_name"].lower()

    def test_verify_reviews_with_no_claims(self):
        """Should return empty report when no claims extracted."""
        fc = IterativeFactCheck()
        empty_results = [
            StructuredResult(
                agent_id="test", sub_task_id="test",
                output="No factual claims here, just opinions.",
                confidence=0.8,
                metadata={"model": "test-model"},
            )
        ]

        with patch.object(fc, "_round1_extract_claims", return_value=[]):
            report = fc.verify_reviews(empty_results)
            assert report.rounds_completed == 1
            assert len(report.verified) == 0
