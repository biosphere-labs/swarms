"""
DecompositionEngine — LLM-based task splitting with structured JSON output.

Analyzes a complex task via LLM and produces a sub-task graph with parallel groups
and dependencies. Implements anti-collapse and anti-spurious-parallelism heuristics
to prevent degenerate decomposition strategies.
"""

import json
from typing import Callable, List, Optional, Dict, Any
from pydantic import BaseModel, Field
from swarms.structs.llm_backend import LLMBackend, LiteLLMBackend
from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger(log_folder="decomposition_engine")


class SubTask(BaseModel):
    """Represents a single sub-task in the decomposition graph."""

    id: str = Field(..., description="Unique identifier for this sub-task")
    description: str = Field(..., description="Clear description of what this sub-task should accomplish")
    context_hints: List[str] = Field(
        default_factory=list,
        description="Context information needed for this sub-task (e.g., which files, APIs, or data to focus on)"
    )
    expected_output_format: str = Field(
        default="text",
        description="Expected format of the sub-task output (e.g., 'json', 'markdown', 'text', 'code')"
    )


class SubTaskGraph(BaseModel):
    """Represents the decomposed task graph with parallel groups and dependencies."""

    parallel_groups: List[List[SubTask]] = Field(
        ...,
        description="List of parallel groups, where each group contains sub-tasks that can run concurrently"
    )
    dependencies: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Edges between groups (e.g., [{'from_group': 0, 'to_group': 1}])"
    )

    def is_single_task(self) -> bool:
        """Check if this is a degenerate single-task graph."""
        return len(self.parallel_groups) == 1 and len(self.parallel_groups[0]) == 1

    def total_subtasks(self) -> int:
        """Count total number of sub-tasks across all groups."""
        return sum(len(group) for group in self.parallel_groups)


class DecompositionEngine:
    """
    LLM-based task decomposition engine with anti-collapse heuristics.

    Analyzes a complex task and produces a SubTaskGraph that describes how to
    split the work into parallelizable sub-tasks. Implements heuristics to:
    - Detect tasks too simple to parallelize (complexity threshold)
    - Prevent serial collapse (always defaulting to single-agent)
    - Prevent spurious parallelism (splitting tasks that shouldn't be split)
    """

    # Complexity thresholds for skipping decomposition
    MIN_TASK_LENGTH = 20  # characters
    MIN_WORD_COUNT = 5

    # Heuristic weights (approximating PARL reward function)
    PARALLEL_REWARD_WEIGHT = 0.3  # Encourages sub-agent spawning
    FINISH_REWARD_WEIGHT = 0.2    # Ensures sub-tasks are completable

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_subtasks: int = 20,
        min_subtasks_for_parallel: int = 2,
        api_key_provider: Optional[Callable] = None,
        llm_backend: Optional[LLMBackend] = None,
    ):
        """
        Initialize the DecompositionEngine.

        Args:
            model: LLM model to use for decomposition (default: gpt-4o-mini for cost efficiency)
            temperature: Temperature for LLM generation (default: 0.7 for balanced creativity)
            max_subtasks: Maximum number of sub-tasks to generate (prevents over-decomposition)
            min_subtasks_for_parallel: Minimum sub-tasks required to consider parallelization
            api_key_provider: Optional callable that returns the next API key (for round-robin)
            llm_backend: Optional LLM backend to use. Defaults to LiteLLMBackend(model).
        """
        self.model = model
        self.temperature = temperature
        self.max_subtasks = max_subtasks
        self.min_subtasks_for_parallel = min_subtasks_for_parallel
        self._api_key_provider = api_key_provider
        self._llm_backend = llm_backend or LiteLLMBackend(model=model)

        logger.info(
            f"DecompositionEngine initialized with model={model}, "
            f"max_subtasks={max_subtasks}, temperature={temperature}, "
            f"backend={type(self._llm_backend).__name__}"
        )

    def _is_task_too_simple(self, task: str) -> bool:
        """
        Check if task is too simple to benefit from decomposition.

        Implements complexity threshold heuristic.

        Args:
            task: Input task string

        Returns:
            True if task should not be decomposed
        """
        # Empty or very short tasks
        if not task or len(task.strip()) < self.MIN_TASK_LENGTH:
            logger.info("Task too short to decompose")
            return True

        # Word count check
        word_count = len(task.split())
        if word_count < self.MIN_WORD_COUNT:
            logger.info(f"Task word count ({word_count}) below minimum ({self.MIN_WORD_COUNT})")
            return True

        # Single-action keywords (heuristic for atomic tasks)
        single_action_patterns = [
            "what is", "define", "explain", "tell me",
            "calculate", "convert", "translate"
        ]
        task_lower = task.lower()
        if any(pattern in task_lower for pattern in single_action_patterns):
            # But allow if task mentions multiple items
            if not any(word in task_lower for word in ["and", "both", "all", "multiple", "several"]):
                logger.info("Task appears to be single-action query")
                return True

        return False

    def _detect_parallelizable_patterns(self, task: str) -> bool:
        """
        Detect if task contains patterns that suggest parallelization would help.

        Prevents spurious parallelism heuristic.

        Args:
            task: Input task string

        Returns:
            True if task shows parallelizable patterns
        """
        task_lower = task.lower()

        # Parallel-friendly patterns
        parallel_indicators = [
            # Multiple items
            "each", "all", "every", "multiple", "several",
            # Lists/collections
            "files", "documents", "pages", "items", "tasks",
            # Comparative/aggregation
            "compare", "analyze", "research", "review", "investigate",
            # Explicit parallelism
            "different", "various", "diverse", "separate"
        ]

        indicator_count = sum(1 for indicator in parallel_indicators if indicator in task_lower)

        # Need at least 2 indicators or 1 strong indicator with list markers
        has_list_markers = any(marker in task for marker in [",", ";", "\n", " and "])

        if indicator_count >= 2 or (indicator_count >= 1 and has_list_markers):
            logger.info(f"Task shows parallelizable patterns (indicator_count={indicator_count})")
            return True

        logger.info("Task does not show clear parallelizable patterns")
        return False

    def _create_single_task_graph(self, task: str) -> SubTaskGraph:
        """
        Create a single-task graph (no decomposition).

        Args:
            task: Original task

        Returns:
            SubTaskGraph with single task
        """
        logger.info("Creating single-task graph (no decomposition)")
        single_task = SubTask(
            id="task_1",
            description=task,
            context_hints=["full_task_context"],
            expected_output_format="text"
        )
        return SubTaskGraph(
            parallel_groups=[[single_task]],
            dependencies=[]
        )

    def _build_decomposition_prompt(self, task: str) -> str:
        """
        Build the prompt for LLM-based task decomposition.

        Args:
            task: Input task to decompose

        Returns:
            Prompt string for the LLM
        """
        prompt = f"""You are a task decomposition expert. Analyze the following task and decompose it into parallelizable sub-tasks.

**Task to decompose:**
{task}

**Instructions:**
1. Identify sub-tasks that can be executed in parallel (truly independent work)
2. Group sub-tasks into parallel groups (tasks in same group run concurrently)
3. Ensure each sub-task has clear, specific description
4. Provide context hints for each sub-task (what information/files/APIs it needs)
5. Specify expected output format for each sub-task

**Important constraints:**
- Maximum {self.max_subtasks} sub-tasks total
- Only create parallel groups if tasks are genuinely independent
- Avoid creating sub-tasks that are just serial steps disguised as parallel work
- Each sub-task should be substantial enough to justify the coordination overhead

**Output format (JSON):**
{{
  "parallel_groups": [
    [
      {{
        "id": "task_1_1",
        "description": "Specific sub-task description",
        "context_hints": ["relevant context", "files/APIs needed"],
        "expected_output_format": "json|text|markdown|code"
      }},
      {{
        "id": "task_1_2",
        "description": "Another parallel sub-task",
        "context_hints": ["context for this task"],
        "expected_output_format": "text"
      }}
    ],
    [
      {{
        "id": "task_2_1",
        "description": "Sub-task in next sequential stage",
        "context_hints": ["depends on results from group 1"],
        "expected_output_format": "text"
      }}
    ]
  ],
  "dependencies": [
    {{"from_group": 0, "to_group": 1}}
  ]
}}

**If the task cannot be meaningfully parallelized, return a single-task graph:**
{{
  "parallel_groups": [
    [
      {{
        "id": "task_1",
        "description": "{task}",
        "context_hints": ["full_task_context"],
        "expected_output_format": "text"
      }}
    ]
  ],
  "dependencies": []
}}

Respond with ONLY the JSON object, no additional text.
"""
        return prompt

    def _call_llm_for_decomposition(
        self, prompt: str, max_retries: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        Call LLM to perform task decomposition with retry and backoff.

        Retries on transient errors (rate limits, model busy) with exponential
        backoff. Non-transient errors (import, JSON parse) fail immediately.

        Args:
            prompt: Decomposition prompt
            max_retries: Maximum number of retry attempts (default 3)

        Returns:
            Parsed JSON response or None if all attempts failed
        """
        import time as _time

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(
                    f"Calling LLM for decomposition (backend={type(self._llm_backend).__name__}, "
                    f"attempt={attempt}/{max_retries})"
                )

                # Get API key from provider if available (round-robin)
                api_key = None
                if self._api_key_provider:
                    api_key = self._api_key_provider()

                content = self._llm_backend.call(
                    system_prompt="You are a task decomposition expert. Respond only with valid JSON.",
                    user_prompt=prompt,
                    temperature=self.temperature,
                    api_key=api_key,
                )

                logger.info(f"LLM response received ({len(content)} chars)")

                # Strip markdown code fences if present (e.g. ```json ... ```)
                stripped = content.strip()
                if stripped.startswith("```"):
                    lines = stripped.split("\n")
                    # Remove first line (```json or ```) and last line (```)
                    lines = lines[1:]
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    stripped = "\n".join(lines)

                # Parse JSON response
                parsed = json.loads(stripped)
                return parsed

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                logger.debug(f"Response content: {content}")
                return None  # Don't retry parse errors
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                is_transient = any(
                    keyword in error_str
                    for keyword in ["rate", "busy", "retry", "timeout", "429", "503"]
                )
                if is_transient and attempt < max_retries:
                    wait_time = 2 ** attempt  # 2s, 4s, 8s
                    logger.warning(
                        f"Transient error on attempt {attempt}: {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    _time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"LLM call failed (attempt {attempt}): {e}")
                    if attempt < max_retries and is_transient:
                        continue
                    return None

        logger.error(f"All {max_retries} decomposition attempts failed. Last error: {last_error}")
        return None

    def _validate_and_fix_graph(self, raw_graph: Dict[str, Any], task: str) -> SubTaskGraph:
        """
        Validate and potentially fix the LLM-generated graph.

        Implements anti-spurious-parallelism checks.

        Args:
            raw_graph: Raw JSON graph from LLM
            task: Original task (for fallback)

        Returns:
            Valid SubTaskGraph
        """
        try:
            # Parse into Pydantic model (validates structure)
            graph = SubTaskGraph(**raw_graph)

            # Check for spurious parallelism
            total_tasks = graph.total_subtasks()

            if total_tasks > self.max_subtasks:
                logger.warning(
                    f"Graph has {total_tasks} sub-tasks (max={self.max_subtasks}), "
                    "truncating to single-task"
                )
                return self._create_single_task_graph(task)

            # Check if "parallel" tasks are actually distinct
            if len(graph.parallel_groups) > 0:
                for group_idx, group in enumerate(graph.parallel_groups):
                    if len(group) > 1:
                        # Check for duplicate or near-duplicate descriptions
                        descriptions = [t.description.lower().strip() for t in group]
                        unique_descriptions = set(descriptions)

                        if len(unique_descriptions) < len(descriptions):
                            logger.warning(
                                f"Group {group_idx} has duplicate sub-tasks, "
                                "likely spurious parallelism"
                            )
                            # Keep only unique tasks
                            seen = set()
                            unique_tasks = []
                            for task_obj in group:
                                desc = task_obj.description.lower().strip()
                                if desc not in seen:
                                    seen.add(desc)
                                    unique_tasks.append(task_obj)
                            graph.parallel_groups[group_idx] = unique_tasks

            # If graph is effectively single-task after validation, simplify it
            if graph.total_subtasks() < self.min_subtasks_for_parallel:
                logger.info(
                    f"Graph has only {graph.total_subtasks()} sub-tasks, "
                    "not worth parallelizing"
                )
                return self._create_single_task_graph(task)

            logger.info(
                f"Valid graph created with {len(graph.parallel_groups)} groups, "
                f"{graph.total_subtasks()} total sub-tasks"
            )
            return graph

        except Exception as e:
            logger.error(f"Graph validation failed: {e}")
            logger.info("Falling back to single-task graph")
            return self._create_single_task_graph(task)

    def decompose(self, task: str) -> SubTaskGraph:
        """
        Decompose a task into a sub-task graph.

        Main entry point for the DecompositionEngine. Implements full decomposition
        pipeline with heuristics to prevent degenerate strategies.

        Args:
            task: Input task to decompose

        Returns:
            SubTaskGraph with parallel groups and dependencies
        """
        logger.info(f"Decomposing task: {task[:100]}...")

        # Heuristic 1: Complexity threshold
        if self._is_task_too_simple(task):
            logger.info("Task too simple for decomposition (complexity threshold)")
            return self._create_single_task_graph(task)

        # Heuristic 2: Parallelizable patterns
        if not self._detect_parallelizable_patterns(task):
            logger.info("Task does not show parallelizable patterns (anti-spurious-parallelism)")
            return self._create_single_task_graph(task)

        # LLM-based decomposition
        prompt = self._build_decomposition_prompt(task)
        raw_graph = self._call_llm_for_decomposition(prompt)

        if raw_graph is None:
            logger.warning("LLM decomposition failed, falling back to single-task")
            return self._create_single_task_graph(task)

        # Validate and fix graph
        graph = self._validate_and_fix_graph(raw_graph, task)

        # Final anti-collapse check
        if graph.is_single_task():
            logger.info("Final graph is single-task (no decomposition benefit)")
        else:
            logger.info(
                f"Decomposition complete: {len(graph.parallel_groups)} groups, "
                f"{graph.total_subtasks()} sub-tasks"
            )

        return graph
