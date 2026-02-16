"""
Integration test: Weave Competitor Research & Market Validation

Runs the full PARL orchestrator against the Weave competitive research prompt
with Serper web search. Outputs results to /tmp/weave-test-output.md for
comparison against reference files.

Usage:
    python tests/run_weave_test.py          # loads .env automatically
    PARL_API_KEYS=... python tests/run_weave_test.py  # or set env vars
"""

import os
import time
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (swarms/)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Ensure env vars are set
assert os.environ.get("SERPER_API_KEYS") or os.environ.get("SERPER_API_KEY"), \
    "Set SERPER_API_KEYS or SERPER_API_KEY in .env"
assert os.environ.get("DEEPINFRA_API_KEY") or os.environ.get("PARL_API_KEYS"), \
    "Set DEEPINFRA_API_KEY or PARL_API_KEYS in .env"

from swarms.structs.parl_orchestrator import PARLOrchestrator
from swarms.tools.serper_search import serper_search


def load_file(path: str) -> str:
    return Path(path).read_text()


def build_prompt() -> str:
    """Build the full prompt from the test spec + context files."""
    weave_dir = Path("/home/justin/Documents/weave-copy")

    product_def = load_file(weave_dir / "product-definition.md")
    feature_list = load_file(weave_dir / "weave-feature-list.md")
    test_spec = load_file(weave_dir / "swarm-test-prompt.md")

    # Extract just the prompt section (between "## Prompt" and "## Expected Outputs")
    lines = test_spec.split("\n")
    prompt_lines = []
    in_prompt = False
    for line in lines:
        if line.strip() == "## Prompt":
            in_prompt = True
            continue
        if line.strip() == "## Expected Outputs":
            break
        if in_prompt:
            prompt_lines.append(line)

    prompt_body = "\n".join(prompt_lines).strip()

    # Combine context + prompt
    full_prompt = f"""## Product Context

{product_def}

---

## Feature List for Comparison

{feature_list}

---

## Research Task

{prompt_body}
"""
    return full_prompt


def main():
    prompt = build_prompt()
    print(f"Prompt length: {len(prompt)} chars (~{len(prompt)//4} tokens)")
    print()

    orchestrator = PARLOrchestrator(
        orchestrator_model=os.environ.get(
            "PARL_ORCHESTRATOR_MODEL", "deepinfra/Qwen/Qwen3-235B-A22B"
        ),
        sub_agent_model=os.environ.get(
            "PARL_SUB_AGENT_MODEL", "deepinfra/Qwen/Qwen2.5-72B-Instruct"
        ),
        synthesis_model=os.environ.get(
            "PARL_SYNTHESIS_MODEL", "deepinfra/deepseek-ai/DeepSeek-V3.2"
        ),
        tools=[serper_search],
        max_parallel=6,
        max_iterations=2,
        timeout=900,  # 15 minutes
        sub_agent_timeout=300,  # 5 minutes per agent
        token_budget=500000,
    )

    print(f"Orchestrator: {orchestrator.orchestrator_model}")
    print(f"Sub-agents: {orchestrator.sub_agent_model}")
    print(f"Synthesis: {orchestrator.synthesis_model}")
    print(f"Tools: {[t.__name__ for t in orchestrator.tools]}")
    print(f"Max parallel: {orchestrator.max_parallel}")
    print(f"Timeout: {orchestrator.timeout}s")
    print()

    start = time.time()
    result = orchestrator.run(prompt)
    elapsed = time.time() - start

    print(f"\n{'='*60}")
    print(f"Completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}\n")

    # Save output to gitignored project directory (not /tmp)
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "weave-test-output.md"
    output_path.write_text(result)
    print(f"Output saved to: {output_path}")
    print(f"Output length: {len(result)} chars")

    # Print first 3000 chars as preview
    print(f"\n{'='*60}")
    print("PREVIEW (first 3000 chars):")
    print(f"{'='*60}\n")
    print(result[:3000])


if __name__ == "__main__":
    main()
