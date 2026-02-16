"""
Model Selector Module

Dynamically queries the DeepInfra model catalog and uses a reasoning model
to assign optimal models to reviewer personas for cross-model review.

Key features:
- Fetches available models from DeepInfra API with 1-hour cache
- Uses hardcoded priors from empirical testing (DeepSeek for logic,
  Gemini for structure, etc.)
- Diversity constraint: no single model gets more than N/3 personas
- Reasoning model (orchestrator) makes final assignment decisions
"""

import json
import time
import threading
from typing import Any, Callable, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

from swarms.structs.llm_backend import LiteLLMBackend
from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger(log_folder="model_selector")

# --- Model catalog cache ---

_cache_lock = threading.Lock()
_cached_models: List[Dict[str, Any]] = []
_cache_timestamp: float = 0.0
_CACHE_TTL = 3600  # 1 hour


# --- Hardcoded priors from empirical testing ---

MODEL_STRENGTH_PRIORS = {
    "deepseek": {
        "strengths": [
            "logical consistency",
            "catching unsupported claims",
            "reasoning chains",
            "spotting time-bound predictions without evidence",
        ],
        "good_for": [
            "Logical Consistency Auditor",
            "Claims Verifier",
            "fact-checking",
        ],
    },
    "gemini": {
        "strengths": [
            "structural contradictions",
            "semantic errors",
            "grammar parsing",
            "same-language-different-meaning errors",
        ],
        "good_for": [
            "Structural Reviewer",
            "Grammar Auditor",
            "contradiction detection",
        ],
        "notes": "May use <think> blocks that consume tokens. Instruct to skip thinking.",
    },
    "qwen": {
        "strengths": [
            "balanced scoring",
            "argument clarity",
            "concise synthesis",
            "general-purpose review",
        ],
        "good_for": [
            "Audience Calibrator",
            "Engagement Optimizer",
            "balanced assessment",
        ],
    },
    "llama": {
        "strengths": [
            "fast inference",
            "cheap general tasks",
            "diverse fourth perspective",
        ],
        "good_for": [
            "AI Pattern Detector",
            "general reviewer",
            "cost-effective perspective",
        ],
    },
    "glm": {
        "strengths": [
            "multilingual analysis",
            "complex reasoning",
            "agentic coding",
        ],
        "good_for": [
            "Technical Reviewer",
            "multilingual content",
        ],
    },
}


def fetch_available_models(
    api_key: str,
    filter_type: str = "text-generation",
    exclude_deprecated: bool = True,
    min_max_tokens: int = 4000,
) -> List[Dict[str, Any]]:
    """
    Query DeepInfra model catalog, returning filtered chat/text models.

    Results are cached for 1 hour at module level.

    Args:
        api_key: DeepInfra API key for authentication.
        filter_type: Model type to filter for (default: text-generation).
        exclude_deprecated: Skip deprecated models (default: True).
        min_max_tokens: Minimum max_tokens the model must support.

    Returns:
        List of model dicts with: model_name, description, tags, pricing,
        max_tokens.
    """
    global _cached_models, _cache_timestamp

    with _cache_lock:
        if _cached_models and (time.time() - _cache_timestamp) < _CACHE_TTL:
            logger.info(f"Using cached model catalog ({len(_cached_models)} models)")
            return _cached_models

    logger.info("Fetching model catalog from DeepInfra API...")

    try:
        req = Request(
            "https://api.deepinfra.com/models/list",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except (URLError, json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to fetch model catalog: {e}")
        return _cached_models if _cached_models else []

    models = data if isinstance(data, list) else data.get("data", data.get("models", []))

    filtered = []
    for m in models:
        if m.get("type") != filter_type and m.get("reported_type") != filter_type:
            continue
        if exclude_deprecated and m.get("deprecated"):
            continue
        if m.get("private"):
            continue
        max_tok = m.get("max_tokens", 0)
        if max_tok and max_tok < min_max_tokens:
            continue
        pricing = m.get("pricing", {})
        if not pricing or not pricing.get("cents_per_input_token"):
            continue

        filtered.append({
            "model_name": m["model_name"],
            "description": (m.get("description") or "")[:200],
            "tags": m.get("tags", []),
            "pricing": {
                "input_per_m": round(pricing.get("cents_per_input_token", 0) * 10000, 2),
                "output_per_m": round(pricing.get("cents_per_output_token", 0) * 10000, 2),
            },
            "max_tokens": max_tok,
        })

    with _cache_lock:
        _cached_models = filtered
        _cache_timestamp = time.time()

    logger.info(f"Cached {len(filtered)} available text-generation models")
    return filtered


ASSIGNMENT_PROMPT = """You are assigning LLM models to reviewer personas for a cross-model document review.

## Available Models (from DeepInfra catalog)

{model_catalog}

## Known Model Family Strengths (from empirical testing)

{priors}

## Personas to Assign

{personas}

## Constraints

1. DIVERSITY: No single model may be assigned to more than {max_same} of the {total} personas.
2. QUALITY: Prefer models with known strengths that match the persona's review focus.
3. COST: Balance quality against cost. Don't use expensive models for simple tasks.
4. AVOID: Skip models with very short descriptions (likely less capable).
5. SKIP: Do not assign models to personas that already have a "model" field.

## Output Format

Return ONLY a JSON array. Each element must have:
- "name": The persona name (exactly as provided)
- "model": The full model_name from the catalog (e.g., "deepseek-ai/DeepSeek-V3.2")
- "reason": One sentence explaining the match

Example:
[
  {{"name": "Thesis Stress-Tester", "model": "deepseek-ai/DeepSeek-V3.2", "reason": "DeepSeek excels at logical consistency and catching unsupported claims."}},
  {{"name": "Audience Calibrator", "model": "Qwen/Qwen3-235B-A22B", "reason": "Qwen provides balanced assessment with strong argument clarity."}}
]

Return ONLY the JSON array, no markdown fences, no explanation."""


def assign_models_to_personas(
    personas: List[Dict[str, Any]],
    available_models: List[Dict[str, Any]],
    orchestrator_model: str,
    api_key: Optional[str] = None,
    max_same_model: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Assign diverse models to personas that don't have explicit model assignments.

    Uses the orchestrator model as a reasoning engine to match persona review
    focus areas to model strengths from the catalog.

    Args:
        personas: List of persona dicts (name, instruction, optional model).
        available_models: Filtered model catalog from fetch_available_models().
        orchestrator_model: Model to use for assignment reasoning.
        api_key: API key for the orchestrator model call.
        max_same_model: Max personas per model. Default: max(1, len(personas) // 3).

    Returns:
        Updated persona list with 'model' fields populated.
    """
    needs_assignment = [p for p in personas if "model" not in p]
    if not needs_assignment:
        logger.info("All personas have explicit model assignments, skipping auto-assign")
        return personas

    total = len(personas)
    max_same = max_same_model or max(1, total // 3)

    # Build compact model catalog for the prompt (top models by relevance)
    catalog_lines = []
    for m in available_models[:40]:
        pricing = m.get("pricing", {})
        cost = f"${pricing.get('input_per_m', '?')}/{pricing.get('output_per_m', '?')} per M tokens"
        desc = m.get("description", "")[:100]
        catalog_lines.append(f"- {m['model_name']} | {cost} | {desc}")
    model_catalog_str = "\n".join(catalog_lines)

    priors_str = json.dumps(MODEL_STRENGTH_PRIORS, indent=2)

    persona_lines = []
    for p in needs_assignment:
        persona_lines.append(f"- {p['name']}: {p['instruction']}")
    personas_str = "\n".join(persona_lines)

    prompt = ASSIGNMENT_PROMPT.format(
        model_catalog=model_catalog_str,
        priors=priors_str,
        personas=personas_str,
        max_same=max_same,
        total=total,
    )

    backend = LiteLLMBackend(orchestrator_model)
    try:
        response = backend.call(
            system_prompt="You are a model selection expert. Return only valid JSON.",
            user_prompt=prompt,
            temperature=0.3,
            max_tokens=2000,
            api_key=api_key,
        )
    except Exception as e:
        logger.error(f"Model assignment LLM call failed: {e}")
        return _fallback_assignment(personas, available_models)

    # Parse the JSON response
    assignments = _parse_assignments(response, available_models)
    if not assignments:
        logger.warning("Failed to parse model assignments, using fallback")
        return _fallback_assignment(personas, available_models)

    # Apply assignments to personas
    assignment_map = {a["name"]: a for a in assignments}
    valid_model_names = {m["model_name"] for m in available_models}

    result = []
    for p in personas:
        p_copy = dict(p)
        if "model" not in p_copy and p_copy["name"] in assignment_map:
            assigned = assignment_map[p_copy["name"]]
            model_name = assigned.get("model", "")
            # Validate model exists in catalog
            if model_name in valid_model_names:
                p_copy["model"] = f"deepinfra/{model_name}"
                p_copy["_assignment_reason"] = assigned.get("reason", "")
                logger.info(f"Assigned {p_copy['name']} → {model_name}: {p_copy['_assignment_reason']}")
            else:
                logger.warning(f"Assigned model '{model_name}' not in catalog for {p_copy['name']}")
        result.append(p_copy)

    # Fill any remaining unassigned with fallback
    for p in result:
        if "model" not in p:
            p["model"] = f"deepinfra/{available_models[0]['model_name']}"
            logger.info(f"Fallback assignment for {p['name']} → {p['model']}")

    return result


def _parse_assignments(response: str, available_models: List[Dict]) -> List[Dict]:
    """Parse the LLM's JSON response for model assignments."""
    text = response.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
        text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # Try to find JSON array in the response
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    logger.error(f"Could not parse model assignments from response: {text[:200]}")
    return []


def _fallback_assignment(
    personas: List[Dict],
    available_models: List[Dict],
) -> List[Dict]:
    """
    Deterministic fallback: round-robin across top models by family diversity.

    Used when the reasoning model fails to produce valid assignments.
    """
    # Pick one model per known family
    families = ["deepseek", "qwen", "gemini", "llama", "glm"]
    family_models = []
    for family in families:
        for m in available_models:
            if family in m["model_name"].lower():
                family_models.append(m["model_name"])
                break

    if not family_models:
        family_models = [m["model_name"] for m in available_models[:3]]

    result = []
    for i, p in enumerate(personas):
        p_copy = dict(p)
        if "model" not in p_copy:
            model = family_models[i % len(family_models)]
            p_copy["model"] = f"deepinfra/{model}"
            logger.info(f"Fallback: {p_copy['name']} → {model}")
        result.append(p_copy)

    return result
