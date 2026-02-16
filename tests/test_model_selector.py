"""
Tests for the Model Selector Module

Tests cover:
- Model catalog fetching and filtering
- Model-persona assignment (LLM-based and fallback)
- Diversity constraints
- JSON parsing robustness
"""

import json
import os
import pytest
from unittest.mock import patch, MagicMock

from swarms.structs.model_selector import (
    fetch_available_models,
    assign_models_to_personas,
    _parse_assignments,
    _fallback_assignment,
    MODEL_STRENGTH_PRIORS,
    _cached_models,
    _cache_timestamp,
)


# --- Test data ---

SAMPLE_CATALOG = [
    {
        "model_name": "deepseek-ai/DeepSeek-V3.2",
        "type": "text-generation",
        "reported_type": "text-generation",
        "description": "Strong reasoning model",
        "tags": ["openai", "tools"],
        "pricing": {
            "cents_per_input_token": 0.000026,
            "cents_per_output_token": 0.000038,
        },
        "max_tokens": 32768,
        "deprecated": None,
        "private": 0,
    },
    {
        "model_name": "Qwen/Qwen3-235B-A22B",
        "type": "text-generation",
        "reported_type": "text-generation",
        "description": "Balanced scoring model",
        "tags": ["openai"],
        "pricing": {
            "cents_per_input_token": 0.00002,
            "cents_per_output_token": 0.00006,
        },
        "max_tokens": 131072,
        "deprecated": None,
        "private": 0,
    },
    {
        "model_name": "google/gemini-2.5-pro",
        "type": "text-generation",
        "reported_type": "text-generation",
        "description": "Strong on structural contradictions",
        "tags": ["openai"],
        "pricing": {
            "cents_per_input_token": 0.00001,
            "cents_per_output_token": 0.00004,
        },
        "max_tokens": 65536,
        "deprecated": None,
        "private": 0,
    },
    {
        "model_name": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        "type": "text-generation",
        "reported_type": "text-generation",
        "description": "Fast cheap general model",
        "tags": ["openai"],
        "pricing": {
            "cents_per_input_token": 0.000012,
            "cents_per_output_token": 0.000018,
        },
        "max_tokens": 32768,
        "deprecated": None,
        "private": 0,
    },
]

SAMPLE_PERSONAS = [
    {"name": "Logic Auditor", "instruction": "Check logical consistency"},
    {"name": "Structure Reviewer", "instruction": "Check structural contradictions"},
    {"name": "Engagement Checker", "instruction": "Check engagement and hooks"},
]


class TestFetchAvailableModels:
    """Tests for fetch_available_models."""

    @patch("swarms.structs.model_selector.urlopen")
    def test_fetches_and_filters_models(self, mock_urlopen):
        """Should filter to text-generation, non-deprecated models with pricing."""
        import swarms.structs.model_selector as ms
        ms._cached_models = []
        ms._cache_timestamp = 0.0

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(SAMPLE_CATALOG).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = fetch_available_models("test-key")

        assert len(result) == 4
        assert all("model_name" in m for m in result)
        assert all("pricing" in m for m in result)

    @patch("swarms.structs.model_selector.urlopen")
    def test_filters_deprecated_models(self, mock_urlopen):
        """Should exclude deprecated models."""
        import swarms.structs.model_selector as ms
        ms._cached_models = []
        ms._cache_timestamp = 0.0

        deprecated_catalog = SAMPLE_CATALOG + [{
            "model_name": "old/deprecated-model",
            "type": "text-generation",
            "reported_type": "text-generation",
            "description": "Old model",
            "tags": [],
            "pricing": {"cents_per_input_token": 0.001, "cents_per_output_token": 0.002},
            "max_tokens": 4096,
            "deprecated": 1700000000,
            "private": 0,
        }]

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(deprecated_catalog).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = fetch_available_models("test-key")
        names = [m["model_name"] for m in result]
        assert "old/deprecated-model" not in names

    def test_uses_cache_on_second_call(self):
        """Should return cached results within TTL."""
        import swarms.structs.model_selector as ms
        import time
        ms._cached_models = [{"model_name": "cached/model"}]
        ms._cache_timestamp = time.time()

        result = fetch_available_models("test-key")
        assert result == [{"model_name": "cached/model"}]

        # Clean up
        ms._cached_models = []
        ms._cache_timestamp = 0.0


class TestAssignModelsToPersonas:
    """Tests for assign_models_to_personas."""

    def test_skips_already_assigned(self):
        """Should not reassign personas that already have model field."""
        personas = [
            {"name": "Test", "instruction": "test", "model": "my-model"},
        ]
        result = assign_models_to_personas(
            personas, SAMPLE_CATALOG, "gpt-4o-mini"
        )
        assert result[0]["model"] == "my-model"

    @patch("swarms.structs.model_selector.LiteLLMBackend")
    def test_calls_orchestrator_for_assignment(self, mock_backend_class):
        """Should call the orchestrator model with catalog and personas."""
        mock_backend = MagicMock()
        mock_backend.call.return_value = json.dumps([
            {"name": "Logic Auditor", "model": "deepseek-ai/DeepSeek-V3.2", "reason": "Good at logic"},
            {"name": "Structure Reviewer", "model": "google/gemini-2.5-pro", "reason": "Good at structure"},
            {"name": "Engagement Checker", "model": "Qwen/Qwen3-235B-A22B", "reason": "Balanced"},
        ])
        mock_backend_class.return_value = mock_backend

        filtered = [
            {"model_name": m["model_name"], "description": m.get("description", "")[:200],
             "tags": m.get("tags", []),
             "pricing": {"input_per_m": 1.0, "output_per_m": 2.0},
             "max_tokens": m.get("max_tokens", 0)}
            for m in SAMPLE_CATALOG
        ]

        result = assign_models_to_personas(
            SAMPLE_PERSONAS, filtered, "test-orchestrator", api_key="test-key"
        )

        assert len(result) == 3
        assert result[0]["model"] == "deepinfra/deepseek-ai/DeepSeek-V3.2"
        assert result[1]["model"] == "deepinfra/google/gemini-2.5-pro"
        assert result[2]["model"] == "deepinfra/Qwen/Qwen3-235B-A22B"

    @patch("swarms.structs.model_selector.LiteLLMBackend")
    def test_uses_fallback_on_parse_failure(self, mock_backend_class):
        """Should fall back to round-robin on LLM parse failure."""
        mock_backend = MagicMock()
        mock_backend.call.return_value = "not valid json"
        mock_backend_class.return_value = mock_backend

        filtered = [
            {"model_name": m["model_name"], "description": "", "tags": [],
             "pricing": {"input_per_m": 1.0, "output_per_m": 2.0},
             "max_tokens": 32768}
            for m in SAMPLE_CATALOG
        ]

        result = assign_models_to_personas(
            SAMPLE_PERSONAS, filtered, "test-orchestrator"
        )

        # Should still assign all personas (via fallback)
        assert len(result) == 3
        assert all("model" in p for p in result)


class TestParseAssignments:
    """Tests for _parse_assignments."""

    def test_parses_clean_json(self):
        """Should parse a clean JSON array."""
        text = '[{"name": "A", "model": "m1"}, {"name": "B", "model": "m2"}]'
        result = _parse_assignments(text, [])
        assert len(result) == 2

    def test_parses_json_with_markdown_fences(self):
        """Should handle markdown code fences."""
        text = '```json\n[{"name": "A", "model": "m1"}]\n```'
        result = _parse_assignments(text, [])
        assert len(result) == 1

    def test_extracts_json_from_surrounding_text(self):
        """Should find JSON array in surrounding text."""
        text = 'Here are the assignments:\n[{"name": "A", "model": "m1"}]\nDone.'
        result = _parse_assignments(text, [])
        assert len(result) == 1

    def test_returns_empty_on_garbage(self):
        """Should return empty list on unparseable input."""
        result = _parse_assignments("totally not json", [])
        assert result == []


class TestFallbackAssignment:
    """Tests for _fallback_assignment."""

    def test_distributes_across_families(self):
        """Should round-robin across different model families."""
        filtered = [
            {"model_name": m["model_name"]}
            for m in SAMPLE_CATALOG
        ]

        result = _fallback_assignment(SAMPLE_PERSONAS, filtered)

        assert len(result) == 3
        models = [p["model"] for p in result]
        # Should have at least 2 different models for 3 personas
        assert len(set(models)) >= 2

    def test_skips_already_assigned(self):
        """Should not reassign personas with existing model."""
        personas = [
            {"name": "A", "instruction": "test", "model": "existing"},
            {"name": "B", "instruction": "test"},
        ]
        filtered = [{"model_name": "deepseek-ai/DeepSeek-V3.2"}]

        result = _fallback_assignment(personas, filtered)
        assert result[0]["model"] == "existing"
        assert "deepinfra/" in result[1]["model"]


class TestModelStrengthPriors:
    """Tests for hardcoded model strength priors."""

    def test_priors_exist_for_key_families(self):
        """Should have priors for DeepSeek, Gemini, Qwen, Llama."""
        assert "deepseek" in MODEL_STRENGTH_PRIORS
        assert "gemini" in MODEL_STRENGTH_PRIORS
        assert "qwen" in MODEL_STRENGTH_PRIORS
        assert "llama" in MODEL_STRENGTH_PRIORS

    def test_priors_have_required_fields(self):
        """Each prior should have strengths and good_for."""
        for family, prior in MODEL_STRENGTH_PRIORS.items():
            assert "strengths" in prior, f"{family} missing strengths"
            assert "good_for" in prior, f"{family} missing good_for"
            assert len(prior["strengths"]) > 0
