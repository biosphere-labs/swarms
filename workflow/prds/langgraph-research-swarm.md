---
name: langgraph-research-swarm
description: LangGraph-based parallel research agent swarm exposed as MCP server for Claude Code
status: backlog
created: 2026-02-11T15:56:15Z
---

# Design: LangGraph Research Swarm

## Summary

A LangGraph-based multi-agent research system that replicates Kimi K2.5's Agent Swarm functionality — parallel sub-agents doing wide research, fact-checking, and synthesis — exposed as an MCP server so Claude Code can invoke it as a tool. Uses cheap models for parallel sub-agent research and a stronger model for orchestration and synthesis. Eliminates the $40/month Kimi platform cost.

## Problem Statement

Doing exhaustive competitor research with LLMs produces confident but wrong results. Single-agent research is serial and shallow. Kimi K2.5's Agent Swarm solves this with parallel sub-agents (up to 100) doing wide information gathering, but the swarm feature is only available on kimi.com ($40/month) — not through the API or third-party providers like DeepInfra. We need our own orchestration layer that achieves similar results at API-call prices.

## What Kimi's Agent Swarm Actually Does

For reference, what we're replicating:

1. **Orchestrator** receives a research query
2. **Decomposes** into parallelizable sub-tasks (e.g., "find funding history", "find user complaints on Reddit", "find founder interviews")
3. **Spawns sub-agents** (up to 100) that work simultaneously with web search tools
4. **Each sub-agent** searches, reads pages, extracts relevant information
5. **Orchestrator synthesizes** all sub-agent results into a coherent report
6. Key metric: 4.5x faster than serial, 78.4% on BrowseComp vs 60.6% for single agent

## Architecture

```
Claude Code
    │
    ▼ (MCP tool call)
┌─────────────────────────────────┐
│  MCP Server (Python, FastMCP)   │
│                                 │
│  ┌───────────────────────────┐  │
│  │  LangGraph State Machine  │  │
│  │                           │  │
│  │  ┌─────────────────────┐  │  │
│  │  │    Planner Node      │  │  │
│  │  │  (Strong Model)      │  │  │
│  │  │  Decomposes query    │  │  │
│  │  │  into sub-tasks      │  │  │
│  │  └──────────┬──────────┘  │  │
│  │             │              │  │
│  │      ┌──────┴──────┐      │  │
│  │      ▼      ▼      ▼     │  │
│  │  ┌──────┐┌──────┐┌──────┐│  │
│  │  │Sub-  ││Sub-  ││Sub-  ││  │
│  │  │Agent ││Agent ││Agent ││  │
│  │  │(Cheap││(Cheap││(Cheap││  │
│  │  │Model)││Model)││Model)││  │
│  │  └──┬───┘└──┬───┘└──┬───┘│  │
│  │     │       │       │     │  │
│  │     ▼       ▼       ▼     │  │
│  │  ┌─────────────────────┐  │  │
│  │  │  Fact-Check Node     │  │  │
│  │  │  (Strong Model)      │  │  │
│  │  │  Cross-references    │  │  │
│  │  │  claims across       │  │  │
│  │  │  sub-agent results   │  │  │
│  │  └──────────┬──────────┘  │  │
│  │             │              │  │
│  │  ┌──────────▼──────────┐  │  │
│  │  │  Synthesis Node      │  │  │
│  │  │  (Strong Model)      │  │  │
│  │  │  Produces final      │  │  │
│  │  │  report with sources │  │  │
│  │  └─────────────────────┘  │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
```

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Orchestration framework | LangGraph | State machine model, native parallel support via Send API, Python ecosystem |
| Sub-agent model | Llama 3.1 8B or Qwen 2.5 7B via DeepInfra (~$0.03-0.06/M tokens) | 10-50x cheaper than frontier models, good enough for search + extraction |
| Orchestrator/synthesis model | Kimi K2.5 or DeepSeek V3 via DeepInfra (~$0.50/M in) | Strong reasoning at low cost |
| Web search | Brave Search API (free tier: 2000 queries/month) or SerpAPI | Already have Brave configured in Claude Code |
| Web page reading | Requests + BeautifulSoup or Trafilatura | Extract article text from URLs |
| Exposure to Claude Code | MCP server (FastMCP Python SDK) | Native integration, Claude calls it like any other tool |
| Parallel execution | LangGraph Send API for scatter-gather | Fan-out to N sub-agents, gather all results, then proceed |
| Fact checking | Cross-reference node that flags contradictions | Claims must appear in 2+ independent sources or get flagged as unverified |

## Graph Nodes (LangGraph State Machine)

### 1. Planner Node
- **Model**: Strong (Kimi K2.5 / DeepSeek V3)
- **Input**: Research query from Claude Code
- **Output**: List of sub-tasks with search strategies
- **Logic**: Decomposes "research competitor X" into specific parallel queries:
  - "Find X funding rounds Crunchbase"
  - "Find X user complaints Reddit"
  - "Find X founder interviews"
  - "Find X pricing and user count"
  - "Find X product reviews G2 Capterra"
  - "Find X traffic trends SimilarWeb"

### 2. Research Sub-Agent Nodes (Parallel)
- **Model**: Cheap (Llama 3.1 8B)
- **Input**: Single focused sub-task
- **Tools**: web_search, fetch_page, extract_text
- **Output**: Structured findings with source URLs
- **Constraint**: Max 5 search queries per sub-agent, max 3 page reads
- **Logic**: Search → read top results → extract relevant facts → return structured data

### 3. Fact-Check Node
- **Model**: Strong (Kimi K2.5 / DeepSeek V3)
- **Input**: All sub-agent results merged
- **Output**: Claims categorized as `verified` (2+ sources), `single-source`, or `contradicted`
- **Logic**: Cross-reference claims. Flag anything that only one sub-agent found. Flag contradictions explicitly. Don't silently resolve — surface disagreements.

### 4. Synthesis Node
- **Model**: Strong (Kimi K2.5 / DeepSeek V3)
- **Input**: Fact-checked findings
- **Output**: Structured research report in markdown
- **Logic**: Produce report with sections matching the research query, inline source citations, confidence levels on claims

### 5. Evaluation Node (Optional Loop)
- **Model**: Strong
- **Input**: Synthesized report
- **Output**: Either "done" or list of gaps needing more research
- **Logic**: If critical gaps found, loop back to Planner with refined sub-tasks

## State Schema

```python
from typing import TypedDict, Annotated
from langgraph.graph import MessagesState

class ResearchState(TypedDict):
    query: str                          # Original research query
    sub_tasks: list[dict]               # Planner output
    sub_agent_results: Annotated[       # Accumulated from parallel agents
        list[dict], operator.add
    ]
    fact_check_report: dict             # Verified/unverified/contradicted claims
    final_report: str                   # Markdown output
    iteration: int                      # Loop counter (max 2 iterations)
    gaps: list[str]                     # Identified gaps for re-research
```

## MCP Server Interface

Two tools exposed to Claude Code:

### `research_swarm`
```
Input:
  query: str         - The research question
  depth: str         - "quick" (3 sub-agents), "standard" (6), "deep" (12)
  focus: str | None  - Optional focus area (e.g., "user complaints", "funding", "product features")

Output:
  report: str        - Markdown research report with inline citations
  confidence: dict   - Per-claim confidence levels
  sources: list[str] - All URLs consulted
  cost: float        - Estimated API cost for this run
```

### `verify_claim`
```
Input:
  claim: str         - A specific claim to fact-check
  context: str       - Where the claim came from

Output:
  status: str        - "verified", "unverified", "contradicted"
  evidence: list     - Supporting or contradicting sources
  sources: list[str] - URLs consulted
```

## Cost Estimation

Per "standard" research run (6 sub-agents):

| Component | Tokens (est.) | Cost |
|-----------|---------------|------|
| Planner (strong) | ~2K in, ~1K out | ~$0.004 |
| 6x Sub-agents (cheap) | ~3K in, ~2K out each | ~$0.002 |
| Fact-check (strong) | ~8K in, ~2K out | ~$0.007 |
| Synthesis (strong) | ~10K in, ~3K out | ~$0.014 |
| **Total per run** | | **~$0.03** |

Compare: Kimi platform at $40/month. This approach costs cents per research session. Even 100 deep research runs per month = ~$3-5.

## Dependencies

- Python 3.11+
- `langgraph` - Orchestration framework
- `langchain-community` - DeepInfra LLM integration
- `fastmcp` - MCP server SDK for Claude Code
- `httpx` or `requests` - HTTP client
- `trafilatura` - Web page text extraction
- `brave-search` or `serpapi` - Web search API client

## File Structure

```
langgraph-research-swarm/
├── pyproject.toml           # Dependencies, project config
├── src/
│   ├── server.py            # MCP server entry point
│   ├── graph.py             # LangGraph state machine definition
│   ├── nodes/
│   │   ├── planner.py       # Query decomposition
│   │   ├── researcher.py    # Sub-agent research logic
│   │   ├── fact_checker.py  # Cross-reference verification
│   │   ├── synthesizer.py   # Report generation
│   │   └── evaluator.py     # Gap detection (optional loop)
│   ├── tools/
│   │   ├── web_search.py    # Brave/SerpAPI wrapper
│   │   └── page_reader.py   # URL fetch + text extraction
│   └── config.py            # Model selection, API keys, limits
├── .env                     # DEEPINFRA_API_KEY, BRAVE_API_KEY
└── README.md
```

## Implementation Approach

1. **Phase 1**: Basic graph with Planner → single Researcher → Synthesis (serial, prove the pattern works)
2. **Phase 2**: Add parallel sub-agents using LangGraph Send API
3. **Phase 3**: Add Fact-Check node with cross-reference logic
4. **Phase 4**: Wrap in MCP server, connect to Claude Code
5. **Phase 5**: Add evaluation loop for gap detection
6. **Phase 6**: Tune prompts based on actual research quality

## Success Criteria

- [ ] Can invoke `research_swarm` from Claude Code and get a structured report back
- [ ] Parallel sub-agents run concurrently (not serial)
- [ ] Claims are flagged with confidence levels based on source count
- [ ] Contradictions between sources are surfaced, not silently resolved
- [ ] Total cost per research run < $0.10 for "standard" depth
- [ ] Report includes clickable source URLs for every major claim
- [ ] Handles rate limiting gracefully (Brave: 1 req/sec)

## Out of Scope

- Real-time streaming of sub-agent progress (batch output is fine)
- Persistent memory across research sessions (each run is independent)
- PDF reading (web pages and search results only for v1)
- Image/chart analysis from web pages
- Academic paper search (arXiv, Semantic Scholar) — can add later

## Risks

| Risk | Mitigation |
|------|------------|
| Cheap models too dumb for extraction | Test with Llama 3.1 8B first; fall back to 70B if quality too low (still cheap at ~$0.35/M) |
| Web search rate limiting | Built-in delays, queue sub-agent search requests, respect Brave 1 req/sec |
| Page reading blocked by anti-bot | Use trafilatura (handles most sites); fall back to search snippet extraction |
| LangGraph complexity for MCP integration | FastMCP is straightforward; the graph runs synchronously from MCP's perspective |
| Research loops forever | Hard cap at 2 iterations in evaluation node |

## Reference Implementations

- [Exa + LangGraph multi-agent research system](https://www.blog.langchain.com/exa/)
- [LangGraph 101: Deep Research Agent (Towards Data Science)](https://towardsdatascience.com/langgraph-101-lets-build-a-deep-research-agent/)
- [Pinecone LangGraph Research Agent](https://www.pinecone.io/learn/langgraph-research-agent/)
- [LangGraph Send API for parallel fan-out](https://www.langchain.com/langgraph)
