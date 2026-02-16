# Discussion-Based Fact-Checking

The PARL orchestrator now uses **multi-round discussion** for fact-checking instead of one-way verification.

## Architecture

### Old (One-Way Verification)
```
Researcher produces output → Fact-checker verifies → Done
```

### New (Multi-Round Discussion)
```
Researcher presents findings ↔ Fact-checker challenges ↔ Judge synthesizes
         │                              │                        │
         └──────────────── 2 rounds of debate ───────────────────┘
                                    │
                              Verified output
```

## How It Works

When fact-checking is enabled (`PARL_FACT_CHECK=true`), each research sub-agent's output goes through a 3-agent debate:

1. **Researcher** (Pro Agent)
   - Presents the research findings
   - Supports claims with data and sources
   - Defends findings when challenged
   - Acknowledges if verification is uncertain

2. **Fact-Checker** (Con Agent)
   - Identifies factual claims (numbers, dates, names, statistics)
   - Uses search tools to independently verify each claim
   - Challenges claims that cannot be verified or are incorrect
   - Provides correct data with sources when errors are found

3. **Judge** (Neutral Agent)
   - Evaluates both sides' arguments
   - Determines which claims are verified, corrected, or unverifiable
   - Synthesizes a final version with only verified information
   - Marks anything unverifiable as `[UNVERIFIED]`

## Configuration

Enable discussion-based fact-checking via environment variable or parameter:

```bash
# Enable with default settings (2 debate rounds)
PARL_FACT_CHECK=true python tests/run_weave_test.py

# Customize the fact-checker model
PARL_FACT_CHECK=true \
PARL_FACT_CHECK_MODEL=deepinfra/Qwen/Qwen2.5-72B-Instruct \
python tests/run_weave_test.py
```

Or programmatically:

```python
from swarms.structs.parl_orchestrator import PARLOrchestrator
from swarms.tools.serper_search import serper_search

orchestrator = PARLOrchestrator(
    fact_check=True,  # Enable discussion-based fact-checking
    fact_check_model="deepinfra/Qwen/Qwen2.5-72B-Instruct",  # Model for all 3 debate agents
    tools=[serper_search],  # Tools for independent verification
)

result = orchestrator.run("Research competitor Acme Corp...")
```

## Example Debate Flow

**Initial Research Output:**
> "Acme Corp raised $50M in Series B funding in 2023. The company has 150 employees and was founded in 2020."

**Round 1:**

*Researcher:* Presents the findings with confidence.

*Fact-Checker:* "I need to verify the Series B amount. Let me search... [uses serper_search] I found that Acme Corp raised $45M, not $50M. The employee count and founding year appear correct."

*Judge:* "The fact-checker has identified a discrepancy in the funding amount. Researcher, can you provide a source for the $50M figure?"

**Round 2:**

*Researcher:* "I acknowledge the $50M may be incorrect based on the fact-checker's search."

*Fact-Checker:* "Confirmed via multiple sources: Series B was $45M in March 2023. Employee count of 150 is verified via LinkedIn. Founding year 2020 is correct per Crunchbase."

*Judge:* "Based on independent verification, I'm synthesizing the corrected version."

**Final Verified Output:**
> "Acme Corp raised $45M in Series B funding in March 2023. The company has 150 employees and was founded in 2020."

## Benefits Over One-Way Verification

| Feature | One-Way Checker | Discussion-Based |
|---------|-----------------|------------------|
| Verification rounds | 1 | 2+ (configurable) |
| Agent interaction | None (one-directional) | Multi-turn debate |
| Ambiguity handling | Flag or pass | Discussed and resolved |
| Source reconciliation | Single perspective | Multiple perspectives |
| Confidence in corrections | Medium | High (debated and judged) |

## Cost Impact

Discussion-based fact-checking uses 3 agents per sub-agent instead of 2:

**Per Sub-Agent (with fact-checking enabled):**
- 1 researcher (sub_agent_model)
- 1 fact-checker (fact_check_model, 2 rounds)
- 1 judge (fact_check_model, 2 rounds)

**Estimated token increase:** ~3-4x per sub-agent output compared to one-way checking.

**Mitigation:**
- Use cheaper models for fact-checking: `PARL_FACT_CHECK_MODEL=deepinfra/Qwen/Qwen2.5-72B-Instruct`
- Reduce debate rounds (currently hardcoded to 2, could be made configurable)
- Only enable for critical research tasks

## Implementation

The discussion pattern is implemented via:

1. **`swarms/structs/fact_check_debate.py`** — `FactCheckDebate` class wrapping `DebateWithJudge`
2. **`swarms/structs/parl_orchestrator.py`** — Integration in `_execute_sub_agent()`
3. **Custom prompts** — Research-specific system prompts for each debate role

## Testing

```bash
# Run fact-check debate tests
pytest tests/test_fact_check_debate.py -v

# Run full PARL orchestrator tests (includes fact-checking)
pytest tests/test_parl_orchestrator.py -v

# Integration test with fact-checking enabled
PARL_FACT_CHECK=true python tests/run_weave_test.py
```

## Future Enhancements

Potential improvements:

1. **Configurable debate rounds** — Currently hardcoded to 2, could be env var
2. **Selective fact-checking** — Only fact-check claims over a confidence threshold
3. **Debate summary** — Return debate transcript alongside verified output
4. **Alternative patterns** — Could use LLM Council for multi-perspective verification instead of 2-agent debate

## Related Files

| File | Purpose |
|------|---------|
| `swarms/structs/fact_check_debate.py` | FactCheckDebate class implementation |
| `swarms/structs/debate_with_judge.py` | Underlying debate pattern (from swarms) |
| `swarms/structs/parl_orchestrator.py` | Integration point in `_execute_sub_agent()` |
| `tests/test_fact_check_debate.py` | Test suite (12 tests) |
