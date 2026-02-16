"""
Iterative Cross-Model Fact Check Module

Multi-round fact-checking where claims are verified by DIFFERENT models
than the ones that made them. This catches model-specific hallucinations
that same-model fact-checking misses.

Distinct from fact_check_debate.py which uses a single model for all
three debate agents. This module uses:
- Round 1: Extract factual claims from review outputs, tag with source model
- Round 2: Cross-verify each claim using a different model + web search
- Round 3: Arbitrate disputed claims with a third model (optional)
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from swarms.structs.agent import Agent
from swarms.structs.llm_backend import LiteLLMBackend
from swarms.structs.result_aggregator import StructuredResult
from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger(log_folder="iterative_fact_check")


# --- Data models ---

class Claim(BaseModel):
    """A factual claim extracted from a review."""
    text: str
    source_agent_id: str
    source_model: str
    context: str = ""


class VerifiedClaim(BaseModel):
    """A claim after verification."""
    claim: Claim
    verification_model: str
    status: str = Field(description="verified, disputed, or unverifiable")
    evidence: str = ""
    correction: Optional[str] = None


class FactCheckReport(BaseModel):
    """Complete fact-check report across all rounds."""
    verified: List[VerifiedClaim] = Field(default_factory=list)
    disputed: List[VerifiedClaim] = Field(default_factory=list)
    unverifiable: List[VerifiedClaim] = Field(default_factory=list)
    corrections: List[VerifiedClaim] = Field(default_factory=list)
    rounds_completed: int = 0

    def summary(self) -> str:
        """Human-readable summary of fact-check results."""
        parts = [
            f"## Fact-Check Report ({self.rounds_completed} rounds)",
            f"- Verified: {len(self.verified)}",
            f"- Disputed: {len(self.disputed)}",
            f"- Unverifiable: {len(self.unverifiable)}",
            f"- Corrections: {len(self.corrections)}",
        ]
        if self.corrections:
            parts.append("\n### Corrections")
            for vc in self.corrections:
                parts.append(
                    f"- **{vc.claim.text}** → {vc.correction} "
                    f"(caught by {vc.verification_model.split('/')[-1]})"
                )
        if self.disputed:
            parts.append("\n### Disputed Claims")
            for vc in self.disputed:
                parts.append(
                    f"- **{vc.claim.text}** — {vc.evidence} "
                    f"(disputed by {vc.verification_model.split('/')[-1]})"
                )
        if self.unverifiable:
            parts.append("\n### Unverifiable Claims")
            for vc in self.unverifiable:
                parts.append(f"- {vc.claim.text}")
        return "\n".join(parts)


# --- Prompts ---

CLAIM_EXTRACTION_PROMPT = """Extract all factual claims from the following review text.

A factual claim is any statement that can be verified or falsified:
- Statistics, numbers, percentages
- Dates, timelines, publication years
- Company names, product names, attributions
- Cause-and-effect claims
- Comparisons or rankings

Do NOT include:
- Opinions or subjective assessments
- Reviewer's recommendations
- Scoring or ratings
- Meta-commentary about the review process

Return a JSON array of claim objects:
[
  {{"claim": "exact text of the claim", "context": "surrounding sentence for reference"}},
  ...
]

Return ONLY the JSON array.

REVIEW TEXT:
{review_text}"""


VERIFICATION_PROMPT = """You are a fact-checker verifying claims made by another AI model.

Your task: Independently verify each claim below using web search.
The claims were made by {source_model} — you are {verification_model}.

For each claim, determine:
1. VERIFIED — you found evidence supporting the claim
2. DISPUTED — you found evidence contradicting the claim (provide the correction)
3. UNVERIFIABLE — you could not find sufficient evidence either way

Use your search tools to check each claim. Be thorough.

Return a JSON array:
[
  {{"claim_index": 0, "status": "verified|disputed|unverifiable", "evidence": "what you found", "correction": "correct info if disputed, null otherwise"}},
  ...
]

Return ONLY the JSON array.

CLAIMS TO VERIFY:
{claims_json}"""


ARBITRATION_PROMPT = """You are an impartial arbitrator resolving fact-checking disputes.

For each disputed claim below, the original reviewer and the fact-checker disagree.
Use web search to independently determine which side is correct.

Return a JSON array:
[
  {{"claim_index": 0, "status": "verified|disputed|unverifiable", "evidence": "your independent finding", "correction": "correct info if the original claim was wrong, null otherwise"}},
  ...
]

Return ONLY the JSON array.

DISPUTED CLAIMS:
{disputes_json}"""


class IterativeFactCheck:
    """
    Multi-round cross-model fact checking.

    Each claim is verified by a DIFFERENT model than the one that made it.
    This catches model-specific hallucinations that same-model checking misses.

    Args:
        extraction_model: Model for claim extraction (cheap, fast).
        available_models: Pool of model names for verification.
        tools: List of tool functions (e.g., serper_search).
        api_key_provider: Callable returning API keys for round-robin.
        max_rounds: Maximum fact-check rounds (2-3).
        max_parallel: Max concurrent verification agents.
        verbose: Enable verbose logging.
    """

    def __init__(
        self,
        extraction_model: str = "deepinfra/Qwen/Qwen2.5-72B-Instruct",
        available_models: Optional[List[str]] = None,
        tools: Optional[List[Callable]] = None,
        api_key_provider: Optional[Callable] = None,
        max_rounds: int = 2,
        max_parallel: int = 5,
        verbose: bool = False,
    ):
        self.extraction_model = extraction_model
        self.available_models = available_models or []
        self.tools = tools or []
        self.api_key_provider = api_key_provider
        self.max_rounds = max(1, min(max_rounds, 3))
        self.max_parallel = max_parallel
        self.verbose = verbose

    def _get_api_key(self) -> Optional[str]:
        if self.api_key_provider:
            return self.api_key_provider()
        return None

    def verify_reviews(
        self,
        review_results: List[StructuredResult],
    ) -> FactCheckReport:
        """
        Run iterative fact-checking across all review outputs.

        Args:
            review_results: List of StructuredResult from reviewers.

        Returns:
            FactCheckReport with verified, disputed, unverifiable, and corrections.
        """
        # Round 1: Extract claims
        logger.info("Fact-check round 1: Extracting claims...")
        all_claims = self._round1_extract_claims(review_results)
        if not all_claims:
            logger.info("No factual claims extracted, skipping fact-check")
            return FactCheckReport(rounds_completed=1)

        logger.info(f"Extracted {len(all_claims)} factual claims from {len(review_results)} reviews")

        # Round 2: Cross-verify
        logger.info("Fact-check round 2: Cross-model verification...")
        verified_claims = self._round2_cross_verify(all_claims)

        report = FactCheckReport(rounds_completed=2)
        for vc in verified_claims:
            if vc.status == "verified":
                report.verified.append(vc)
            elif vc.status == "disputed":
                report.disputed.append(vc)
                if vc.correction:
                    report.corrections.append(vc)
            else:
                report.unverifiable.append(vc)

        # Round 3: Arbitrate disputes (optional)
        if self.max_rounds >= 3 and report.disputed:
            logger.info(f"Fact-check round 3: Arbitrating {len(report.disputed)} disputes...")
            arbitrated = self._round3_arbitrate(report.disputed)

            # Replace disputed with arbitrated results
            report.disputed = []
            report.corrections = []
            for vc in arbitrated:
                if vc.status == "verified":
                    report.verified.append(vc)
                elif vc.status == "disputed":
                    report.disputed.append(vc)
                    if vc.correction:
                        report.corrections.append(vc)
                else:
                    report.unverifiable.append(vc)
            report.rounds_completed = 3

        logger.info(
            f"Fact-check complete: {len(report.verified)} verified, "
            f"{len(report.disputed)} disputed, {len(report.corrections)} corrections"
        )
        return report

    def _round1_extract_claims(
        self,
        review_results: List[StructuredResult],
    ) -> List[Claim]:
        """Extract factual claims from each review, tagging with source model."""
        all_claims: List[Claim] = []
        backend = LiteLLMBackend(self.extraction_model)

        for result in review_results:
            if not result.output or result.confidence == 0.0:
                continue

            source_model = result.metadata.get("model", "unknown")

            try:
                response = backend.call(
                    system_prompt="You are a precise claim extractor. Return only valid JSON.",
                    user_prompt=CLAIM_EXTRACTION_PROMPT.format(review_text=result.output),
                    temperature=0.1,
                    max_tokens=3000,
                    api_key=self._get_api_key(),
                )
                claims_data = _parse_json_array(response)
                for item in claims_data:
                    claim_text = item.get("claim", "")
                    if claim_text:
                        all_claims.append(Claim(
                            text=claim_text,
                            source_agent_id=result.agent_id,
                            source_model=source_model,
                            context=item.get("context", ""),
                        ))
            except Exception as e:
                logger.warning(f"Claim extraction failed for {result.agent_id}: {e}")

        return all_claims

    def _round2_cross_verify(
        self,
        claims: List[Claim],
    ) -> List[VerifiedClaim]:
        """Verify claims using a different model than the source."""
        if not claims:
            return []

        # Group claims by source model for batch verification
        groups: Dict[str, List[Claim]] = {}
        for claim in claims:
            groups.setdefault(claim.source_model, []).append(claim)

        verified: List[VerifiedClaim] = []

        def verify_group(source_model: str, group_claims: List[Claim]) -> List[VerifiedClaim]:
            # Pick a different model for verification
            verify_model = self._pick_different_model(source_model)
            if not verify_model:
                return [
                    VerifiedClaim(
                        claim=c, verification_model="none",
                        status="unverifiable", evidence="No verification model available"
                    )
                    for c in group_claims
                ]

            claims_json = json.dumps(
                [{"claim_index": i, "claim": c.text, "context": c.context}
                 for i, c in enumerate(group_claims)],
                indent=2,
            )

            agent = Agent(
                agent_name=f"fact-checker-{verify_model.split('/')[-1].lower()}",
                system_prompt="You are a thorough fact-checker. Verify claims using web search.",
                model_name=verify_model,
                max_loops=3,
                max_tokens=8192,
                output_type="final",
                verbose=False,
                tools=self.tools if self.tools else None,
                llm_api_key=self._get_api_key(),
            )

            prompt = VERIFICATION_PROMPT.format(
                source_model=source_model.split("/")[-1],
                verification_model=verify_model.split("/")[-1],
                claims_json=claims_json,
            )

            try:
                raw = agent.run(task=prompt)
                results_data = _parse_json_array(str(raw) if raw else "[]")

                result_map = {r.get("claim_index", -1): r for r in results_data}
                group_verified = []
                for i, claim in enumerate(group_claims):
                    r = result_map.get(i, {})
                    group_verified.append(VerifiedClaim(
                        claim=claim,
                        verification_model=verify_model,
                        status=r.get("status", "unverifiable"),
                        evidence=r.get("evidence", ""),
                        correction=r.get("correction"),
                    ))
                return group_verified
            except Exception as e:
                logger.error(f"Verification failed for group from {source_model}: {e}")
                return [
                    VerifiedClaim(
                        claim=c, verification_model=verify_model,
                        status="unverifiable", evidence=f"Verification error: {e}"
                    )
                    for c in group_claims
                ]

        # Run verification groups in parallel
        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            futures = {
                executor.submit(verify_group, src, grp): src
                for src, grp in groups.items()
            }
            for future in as_completed(futures, timeout=120):
                try:
                    verified.extend(future.result(timeout=30))
                except Exception as e:
                    src = futures[future]
                    logger.error(f"Verification group {src} failed: {e}")

        return verified

    def _round3_arbitrate(
        self,
        disputed: List[VerifiedClaim],
    ) -> List[VerifiedClaim]:
        """Arbitrate disputed claims with a third model."""
        if not disputed:
            return []

        # Pick a model not used by either the source or verifier
        used_models = set()
        for vc in disputed:
            used_models.add(vc.claim.source_model)
            used_models.add(vc.verification_model)

        arbitrator_model = self._pick_different_model_excluding(used_models)
        if not arbitrator_model:
            # Can't find a third model, return as-is
            return disputed

        disputes_json = json.dumps(
            [{"claim_index": i, "original_claim": vc.claim.text,
              "source_model": vc.claim.source_model.split("/")[-1],
              "verifier_finding": vc.evidence,
              "verifier_model": vc.verification_model.split("/")[-1]}
             for i, vc in enumerate(disputed)],
            indent=2,
        )

        agent = Agent(
            agent_name=f"arbitrator-{arbitrator_model.split('/')[-1].lower()}",
            system_prompt="You are an impartial fact-checking arbitrator.",
            model_name=arbitrator_model,
            max_loops=3,
            max_tokens=8192,
            output_type="final",
            verbose=False,
            tools=self.tools if self.tools else None,
            llm_api_key=self._get_api_key(),
        )

        try:
            raw = agent.run(task=ARBITRATION_PROMPT.format(disputes_json=disputes_json))
            results_data = _parse_json_array(str(raw) if raw else "[]")

            result_map = {r.get("claim_index", -1): r for r in results_data}
            arbitrated = []
            for i, vc in enumerate(disputed):
                r = result_map.get(i, {})
                arbitrated.append(VerifiedClaim(
                    claim=vc.claim,
                    verification_model=arbitrator_model,
                    status=r.get("status", vc.status),
                    evidence=r.get("evidence", vc.evidence),
                    correction=r.get("correction", vc.correction),
                ))
            return arbitrated
        except Exception as e:
            logger.error(f"Arbitration failed: {e}")
            return disputed

    def _pick_different_model(self, source_model: str) -> Optional[str]:
        """Pick a verification model different from the source."""
        source_family = _model_family(source_model)
        for model in self.available_models:
            if _model_family(model) != source_family:
                return model
        # Fallback: any model that isn't the exact same
        for model in self.available_models:
            if model != source_model:
                return model
        return self.available_models[0] if self.available_models else None

    def _pick_different_model_excluding(self, exclude: set) -> Optional[str]:
        """Pick a model not in the exclude set (by family)."""
        exclude_families = {_model_family(m) for m in exclude}
        for model in self.available_models:
            if _model_family(model) not in exclude_families:
                return model
        # Fallback: any model not exactly in exclude
        for model in self.available_models:
            if model not in exclude:
                return model
        return None


def _model_family(model_name: str) -> str:
    """Extract model family from a model name for diversity checks."""
    name = model_name.lower()
    for family in ["deepseek", "gemini", "qwen", "llama", "glm", "kimi", "mistral"]:
        if family in name:
            return family
    return name.split("/")[-1].split("-")[0] if "/" in name else name


def _parse_json_array(text: str) -> List[Dict]:
    """Parse a JSON array from LLM output, handling markdown fences."""
    text = text.strip()
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
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return []
