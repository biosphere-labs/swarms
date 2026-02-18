"""
Concept Developer — Iterative concept development through research, synthesis, and critique.

Takes a rough concept and iteratively refines it:
1. DECOMPOSE — generate research directions from the concept
2. RESEARCH — parallel agents with web search explore each direction
3. SYNTHESIZE — merge research into a structured concept definition
4. CRITIQUE — different model identifies gaps and unanswered questions
5. GAP-FILL — parallel agents research each unanswered question
6. Loop 3-5 until critique is satisfied or max iterations reached

Uses existing framework components: Agent (with tools), LiteLLMBackend (for
single-turn LLM calls), and the staggered parallel execution pattern from
the PARL MCP server.
"""

import asyncio
import gc
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from swarms.structs.agent import Agent
from swarms.structs.llm_backend import LiteLLMBackend
from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger(log_folder="concept_developer")

# --- Auto-select a critique model from a different family ---

_FAMILY_DEFAULTS = {
    "deepseek": "deepinfra/Qwen/Qwen3-235B-A22B",
    "qwen": "deepinfra/deepseek-ai/DeepSeek-V3.2",
    "meta-llama": "deepinfra/Qwen/Qwen3-235B-A22B",
    "google": "deepinfra/deepseek-ai/DeepSeek-V3.2",
    "nvidia": "deepinfra/Qwen/Qwen3-235B-A22B",
}


def _auto_critique_model(synthesis_model: str) -> str:
    """Pick a critique model from a different family than synthesis."""
    lower = synthesis_model.lower()
    for family, alt in _FAMILY_DEFAULTS.items():
        if family in lower:
            return alt
    # Fallback: just use Qwen
    return "deepinfra/Qwen/Qwen3-235B-A22B"


# --- Prompts ---

DECOMPOSE_PROMPT = """You are a research planner. Given a concept to develop, generate focused research directions.

Concept: {concept}
{context_section}
{domains_section}

Generate 4-6 research directions. Each direction should explore a different facet of this concept.
Always include:
- The academic/formal definition and origin of the concept
- Real-world examples and implementations
- Alternative or competing definitions
- Boundaries: what the concept is NOT (to clarify scope)

Return ONLY a JSON array. Each element:
{{"direction": "short label", "query": "search-engine-optimized query string"}}

Example:
[
  {{"direction": "Academic definition", "query": "adaptive knowledge graphs definition academic research"}},
  {{"direction": "Industry examples", "query": "companies using adaptive knowledge graphs products"}}
]

Return ONLY the JSON array."""

SYNTHESIS_PROMPT = """You are synthesizing research findings into a comprehensive concept definition.

Concept: {concept}
{context_section}

## Research Findings

{research_text}

## Instructions

Produce a structured concept definition covering:
1. **Core Definition** — What is this concept? One clear paragraph.
2. **Key Characteristics** — What distinguishes it? Bullet points.
3. **Examples Found** — Concrete examples from the research, with sources.
4. **Boundaries** — What this concept is NOT. How it differs from related concepts.
5. **Related Concepts** — What it connects to, and how.
6. **Open Questions** — Anything the research didn't fully clarify.

Be specific. Cite sources from the research. If different sources disagree, note the disagreement."""

CRITIQUE_PROMPT = """You are a concept definition critic. Review this definition for completeness,
clarity, and gaps.

Concept: {concept}
{context_section}

## Current Definition (Iteration {iteration})

{definition}

## Instructions

Evaluate the definition critically:
1. Is the core definition clear and unambiguous?
2. Are the characteristics well-supported by evidence?
3. Are the examples concrete and verifiable?
4. Are the boundaries well-defined?
5. What questions does a reader still have after reading this?

Return ONLY a JSON object:
{{
  "strengths": ["strength 1", "strength 2"],
  "weaknesses": ["weakness 1", "weakness 2"],
  "unanswered_questions": ["specific question 1", "specific question 2"]
}}

IMPORTANT: "unanswered_questions" should contain specific, researchable questions.
If the definition is comprehensive and clear, return an empty array for unanswered_questions.
Only flag genuine gaps — do not invent questions for the sake of having them.

Return ONLY the JSON object."""

FINAL_SYNTHESIS_PROMPT = """You are producing the final, refined concept definition.

Concept: {concept}
{context_section}

## Draft Definition

{definition}

## Critique History

{critique_history}

## All Research (including gap-fill)

{research_text}

## Instructions

Produce the final concept definition. Incorporate all critique feedback and additional
research. The output should be comprehensive, well-structured, and ready to use as
a reference document.

Structure:
1. **Definition** — Clear, concise core definition (1-2 paragraphs)
2. **Key Characteristics**
3. **Examples & Evidence**
4. **Boundaries & Distinctions**
5. **Related Concepts**
6. **Synthesis Notes** — Key debates, open areas, confidence levels"""


class ConceptDeveloper:
    """
    Iterative concept development through research, synthesis, critique, and gap-fill loops.

    Uses parallel research agents (with web search) for discovery, single-turn LLM calls
    for synthesis and critique, and iterates until the critique model finds no more gaps.
    """

    def __init__(
        self,
        research_model: str,
        synthesis_model: str,
        critique_model: Optional[str] = None,
        max_iterations: int = 3,
        tools: Optional[list] = None,
        api_key_provider: Optional[Callable] = None,
        stagger_delay: float = 2.0,
        max_parallel: int = 6,
        progress_callback: Optional[Callable] = None,
    ):
        self.research_model = research_model
        self.synthesis_model = synthesis_model
        self.critique_model = critique_model or _auto_critique_model(synthesis_model)
        self.max_iterations = max_iterations
        self.tools = tools
        self._api_key_provider = api_key_provider
        self.stagger_delay = stagger_delay
        self.max_parallel = max_parallel
        self._progress = progress_callback  # async callable(phase, detail)

        self._synthesis_backend = LiteLLMBackend(model=synthesis_model)
        self._critique_backend = LiteLLMBackend(model=self.critique_model)

        logger.info(
            f"ConceptDeveloper: research={research_model}, synthesis={synthesis_model}, "
            f"critique={self.critique_model}, max_iter={max_iterations}"
        )

    def _get_api_key(self) -> Optional[str]:
        if self._api_key_provider:
            return self._api_key_provider()
        return None

    def _call_with_retry(self, backend, system_prompt: str, user_prompt: str,
                         temperature: float, max_tokens: int, retries: int = 4) -> str:
        for attempt in range(retries):
            try:
                return backend.call(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_key=self._get_api_key(),
                )
            except Exception as e:
                err_str = str(e).lower()
                if attempt < retries - 1 and ("rate" in err_str or "429" in err_str or "busy" in err_str):
                    backoff = min(2 ** (attempt + 1), 30)
                    logger.info(f"Rate limit on LLM call, backing off {backoff}s (attempt {attempt+1}/{retries})")
                    time.sleep(backoff)
                else:
                    raise

    def _call_synthesis(self, prompt: str, max_tokens: int = 4000) -> str:
        return self._call_with_retry(
            self._synthesis_backend,
            system_prompt="You are an expert concept analyst and synthesizer.",
            user_prompt=prompt,
            temperature=0.4,
            max_tokens=max_tokens,
        )

    def _call_critique(self, prompt: str) -> str:
        return self._call_with_retry(
            self._critique_backend,
            system_prompt="You are a rigorous concept definition critic.",
            user_prompt=prompt,
            temperature=0.3,
            max_tokens=2000,
        )

    # --- Phase 1: Decompose into research directions ---

    def _decompose_research(
        self, concept: str, context: str, domains: Optional[List[str]]
    ) -> List[Dict[str, str]]:
        context_section = f"Context: {context}" if context else ""
        domains_section = ""
        if domains:
            domains_section = (
                "Required domains to cover (include one direction per domain):\n"
                + "\n".join(f"- {d}" for d in domains)
            )

        prompt = DECOMPOSE_PROMPT.format(
            concept=concept,
            context_section=context_section,
            domains_section=domains_section,
        )

        response = self._call_synthesis(prompt, max_tokens=1500)
        directions = self._parse_json_array(response)

        if not directions:
            # Fallback: generate basic directions
            logger.warning("Failed to parse decomposition, using fallback directions")
            directions = [
                {"direction": "Definition", "query": f"{concept} definition meaning"},
                {"direction": "Examples", "query": f"{concept} examples real world"},
                {"direction": "Alternatives", "query": f"{concept} alternatives similar concepts"},
                {"direction": "Boundaries", "query": f"what is not {concept} difference between"},
            ]
            if domains:
                for d in domains:
                    directions.append({"direction": d, "query": f"{concept} {d}"})

        logger.info(f"Decomposed into {len(directions)} research directions")
        return directions

    # --- Disk-based storage helpers ---

    def _write_to_disk(self, text: str, label: str) -> str:
        """Write text to a temp file and return the path. Caller must clean up."""
        fd, path = tempfile.mkstemp(
            prefix=f"concept_{label[:20]}_", suffix=".txt", dir=self._tmp_dir
        )
        try:
            os.write(fd, text.encode("utf-8", errors="replace"))
        finally:
            os.close(fd)
        return path

    @staticmethod
    def _read_from_disk(path: str) -> str:
        """Read text back from a temp file."""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except FileNotFoundError:
            return "[File not found — output lost]"

    def _cleanup_disk_files(self, results: List[Dict[str, str]]) -> None:
        """Remove temp files referenced in research results."""
        for r in results:
            path = r.get("_file")
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass

    def _research_text_from_results(self, results: List[Dict[str, str]]) -> str:
        """Build combined research text, reading from disk files one at a time."""
        parts = []
        for r in results:
            file_path = r.get("_file")
            if file_path:
                text = self._read_from_disk(file_path)
            else:
                text = r.get("output", "")
            if text:
                parts.append(f"### {r['direction']}\n{text}")
        return "\n\n".join(parts)

    # --- Phase 2 & 5: Parallel research ---

    def _parallel_research(self, directions: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Run one research agent per direction, staggered to avoid rate limits.

        Each agent's output is written to a temp file on disk immediately, and the
        Agent object is deleted to free memory. Results contain a '_file' key pointing
        to the temp file instead of holding the full text in memory.
        """
        results = []

        def run_single(direction: Dict[str, str], index: int) -> Dict[str, str]:
            # Stagger launches
            if index > 0 and self.stagger_delay > 0:
                time.sleep(index * self.stagger_delay)

            name = direction["direction"]
            query = direction["query"]
            api_key = self._get_api_key()

            agent = Agent(
                agent_name=f"researcher-{name.lower().replace(' ', '-')[:30]}",
                system_prompt=(
                    f"You are a research specialist investigating: {name}\n\n"
                    "Search for relevant information using your tools. Report what you "
                    "find with specific details, quotes, and source URLs. If you find "
                    "contradictory information, report both sides. Be concise — focus on "
                    "key findings, not exhaustive summaries."
                ),
                model_name=self.research_model,
                max_loops=2,
                max_tokens=3000,
                output_type="final",
                verbose=False,
                tools=self.tools,
                llm_api_key=api_key,
            )

            output = ""
            try:
                raw_output = agent.run(task=f"Research the following:\n\n{query}")
                output = str(raw_output) if raw_output is not None else ""
            finally:
                # Aggressively free memory: clear internals and delete agent
                if hasattr(agent, 'short_memory') and agent.short_memory:
                    agent.short_memory.clear()
                if hasattr(agent, 'long_term_memory') and agent.long_term_memory:
                    try:
                        agent.long_term_memory = None
                    except Exception:
                        pass
                del agent
                gc.collect()

            # Write output to disk immediately, then drop the string
            file_path = self._write_to_disk(output, name.replace(" ", "_"))
            char_count = len(output)
            del output
            gc.collect()

            logger.info(f"Researcher '{name}' completed ({char_count} chars, saved to disk)")
            return {"direction": name, "output": "", "_file": file_path}

        # Run with thread pool, respecting max_parallel
        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            futures = [
                executor.submit(run_single, d, i)
                for i, d in enumerate(directions)
            ]
            for future in futures:
                try:
                    results.append(future.result(timeout=180))
                except Exception as e:
                    logger.error(f"Research agent failed: {e}")
                    results.append({"direction": "unknown", "output": f"[Research failed: {e}]"})

        non_empty = sum(1 for r in results if r.get("_file"))
        logger.info(f"Research complete: {non_empty}/{len(results)} agents produced output (on disk)")
        return results

    # --- Phase 3: Synthesize ---

    def _synthesize(
        self, concept: str, context: str, research_results: List[Dict[str, str]]
    ) -> str:
        context_section = f"Context: {context}" if context else ""

        research_text = self._research_text_from_results(research_results)

        prompt = SYNTHESIS_PROMPT.format(
            concept=concept,
            context_section=context_section,
            research_text=research_text,
        )
        # Free the research text string before LLM call
        del research_text
        gc.collect()

        definition = self._call_synthesis(prompt, max_tokens=4000)
        logger.info(f"Synthesis complete ({len(definition)} chars)")
        return definition

    # --- Phase 4: Critique ---

    def _critique(
        self, concept: str, definition: str, context: str, iteration: int
    ) -> Dict[str, Any]:
        context_section = f"Context: {context}" if context else ""

        prompt = CRITIQUE_PROMPT.format(
            concept=concept,
            context_section=context_section,
            definition=definition,
            iteration=iteration + 1,
        )

        response = self._call_critique(prompt)

        # Parse JSON response
        parsed = self._parse_json_object(response)
        if parsed is None:
            logger.warning("Failed to parse critique JSON, treating as no gaps")
            return {"strengths": [], "weaknesses": [], "unanswered_questions": []}

        # Ensure required keys
        result = {
            "strengths": parsed.get("strengths", []),
            "weaknesses": parsed.get("weaknesses", []),
            "unanswered_questions": parsed.get("unanswered_questions", []),
        }

        n_questions = len(result["unanswered_questions"])
        logger.info(
            f"Critique (iter {iteration+1}): {len(result['strengths'])} strengths, "
            f"{len(result['weaknesses'])} weaknesses, {n_questions} unanswered questions"
        )
        return result

    # --- Final synthesis ---

    def _final_synthesis(
        self,
        concept: str,
        context: str,
        definition: str,
        critiques: List[Dict[str, Any]],
        research_results: List[Dict[str, str]],
    ) -> str:
        context_section = f"Context: {context}" if context else ""

        critique_history = ""
        for i, c in enumerate(critiques):
            critique_history += f"\n### Iteration {i+1}\n"
            critique_history += f"Strengths: {', '.join(c['strengths'])}\n"
            critique_history += f"Weaknesses: {', '.join(c['weaknesses'])}\n"
            if c["unanswered_questions"]:
                critique_history += f"Questions asked: {', '.join(c['unanswered_questions'])}\n"

        research_text = self._research_text_from_results(research_results)

        prompt = FINAL_SYNTHESIS_PROMPT.format(
            concept=concept,
            context_section=context_section,
            definition=definition,
            critique_history=critique_history,
            research_text=research_text[:30000],  # cap to avoid token overflow
        )
        del research_text
        gc.collect()

        final = self._call_synthesis(prompt, max_tokens=6000)
        logger.info(f"Final synthesis complete ({len(final)} chars)")
        return final

    # --- Main entry point ---

    def develop(
        self,
        concept: str,
        context: str = "",
        domains: Optional[List[str]] = None,
    ) -> str:
        """
        Develop a concept definition through iterative research, synthesis, and critique.

        Agent outputs are written to temp files on disk to avoid holding all research
        text in memory simultaneously (prevents earlyoom kills on memory-constrained systems).

        Args:
            concept: The concept to develop (e.g. "adaptive knowledge graphs")
            context: Optional context constraining the concept (e.g. "in AI-assisted research")
            domains: Optional list of specific domains to research in

        Returns:
            Comprehensive concept definition as markdown text
        """
        start_time = time.time()
        all_critiques: List[Dict[str, Any]] = []
        all_research: List[Dict[str, str]] = []

        # Create a temp directory for this run's disk-based storage
        self._tmp_dir = tempfile.mkdtemp(prefix="concept_dev_")
        logger.info(f"Disk storage: {self._tmp_dir}")

        def _report(phase: str, detail: str = ""):
            logger.info(f"{phase}: {detail}" if detail else phase)
            if self._progress:
                try:
                    self._progress(phase, detail)
                except Exception:
                    pass  # non-fatal

        try:
            # Phase 1: Decompose
            _report("Phase 1: Decompose", f"Generating research directions for '{concept}'")
            directions = self._decompose_research(concept, context, domains)
            _report("Phase 1: Complete", f"{len(directions)} research directions generated")

            # Phase 2: Initial research
            _report("Phase 2: Research", f"Running {len(directions)} parallel research agents")
            all_research = self._parallel_research(directions)
            _report("Phase 2: Complete", f"{len(all_research)} results collected")

            definition = ""

            for iteration in range(self.max_iterations):
                # Phase 3: Synthesize
                _report(f"Phase 3 (iter {iteration+1}): Synthesize", "Merging research into definition")
                definition = self._synthesize(concept, context, all_research)

                # Phase 4: Critique
                _report(f"Phase 4 (iter {iteration+1}): Critique", f"Reviewing with {self.critique_model}")
                critique = self._critique(concept, definition, context, iteration)
                all_critiques.append(critique)

                questions = critique.get("unanswered_questions", [])
                if not questions:
                    _report(f"Iteration {iteration+1}: Satisfied", "No more gaps found")
                    break

                # Phase 5: Gap-fill research
                _report(f"Phase 5 (iter {iteration+1}): Gap-fill", f"Researching {len(questions)} gaps")
                gap_directions = [
                    {"direction": f"Gap: {q[:50]}", "query": q}
                    for q in questions
                ]
                gap_results = self._parallel_research(gap_directions)
                all_research.extend(gap_results)
                _report(f"Phase 5 (iter {iteration+1}): Complete", f"{len(gap_results)} gap results added")

            # Final synthesis
            _report("Final synthesis", "Incorporating all critique feedback")
            final = self._final_synthesis(concept, context, definition, all_critiques, all_research)

            elapsed = time.time() - start_time
            iterations_done = len(all_critiques)
            total_research = len(all_research)

            footer = (
                f"\n\n---\n*Concept developed in {elapsed:.0f}s across {iterations_done} "
                f"iteration(s) with {total_research} research directions. "
                f"Models: research={self.research_model}, synthesis={self.synthesis_model}, "
                f"critique={self.critique_model}*"
            )

            return final + footer

        finally:
            # Clean up all temp files
            self._cleanup_disk_files(all_research)
            try:
                os.rmdir(self._tmp_dir)
            except OSError:
                pass  # directory not empty or already removed
            logger.info("Disk storage cleaned up")

    # --- JSON parsing helpers ---

    @staticmethod
    def _parse_json_array(text: str) -> Optional[List[Dict]]:
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
        return None

    @staticmethod
    def _parse_json_object(text: str) -> Optional[Dict]:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
            text = text.strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        return None
