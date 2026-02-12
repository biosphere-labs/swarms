"""
Context Sharding Manager for PARL Orchestrator

Provides per-sub-agent context isolation and result collection.
Each sub-agent receives only task-relevant context, preventing cross-contamination
and reducing token usage.
"""

from typing import Dict, Optional, List, Any
from pydantic import BaseModel, Field
from swarms.utils.litellm_tokenizer import count_tokens


class SubTask(BaseModel):
    """
    Represents a decomposed sub-task.
    This is a minimal definition that will be used by DecompositionEngine.
    """
    id: str = Field(..., description="Unique identifier for the sub-task")
    description: str = Field(..., description="Task description for the sub-agent")
    context_hints: Optional[List[str]] = Field(
        default=None,
        description="Keywords/phrases indicating relevant context sections"
    )
    expected_output_format: Optional[str] = Field(
        default=None,
        description="Expected output format (e.g., 'json', 'markdown', 'summary')"
    )


class ContextShard(BaseModel):
    """
    Isolated context window for a single sub-agent.

    Attributes:
        system_prompt: Instructions for the sub-agent
        task_description: Specific task to execute
        relevant_context_slice: Extracted relevant portions from parent context
        output_format: Expected output format
        token_count: Estimated tokens in this shard
    """
    system_prompt: str = Field(
        ...,
        description="System instructions for the sub-agent"
    )
    task_description: str = Field(
        ...,
        description="Specific task description"
    )
    relevant_context_slice: str = Field(
        default="",
        description="Relevant context extracted from parent conversation"
    )
    output_format: str = Field(
        default="markdown",
        description="Expected output format"
    )
    token_count: int = Field(
        default=0,
        description="Estimated token count for this shard"
    )

    class Config:
        frozen = True  # Immutable once created


class StructuredResult(BaseModel):
    """
    Structured output from a sub-agent.

    Attributes:
        agent_id: Unique identifier of the sub-agent
        sub_task_id: ID of the sub-task that was executed
        output: The actual output from the sub-agent
        confidence: Confidence score (0.0-1.0)
        metadata: Additional metadata (execution time, token usage, etc.)
    """
    agent_id: str = Field(..., description="Unique identifier of the sub-agent")
    sub_task_id: str = Field(..., description="ID of the sub-task")
    output: str = Field(..., description="Sub-agent output")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for the output"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (tokens, execution_time, etc.)"
    )


class ContextShardingManager:
    """
    Manages context isolation for sub-agents in PARL orchestration.

    This is the core differentiator from ConcurrentWorkflow, which shares context.
    Each sub-agent gets a focused context shard based on task-relevant information.

    Attributes:
        default_system_prompt: Default system prompt for sub-agents
        max_context_tokens: Maximum tokens per context shard
        model_name: Model name for token counting (defaults to gpt-4)
    """

    def __init__(
        self,
        default_system_prompt: str = "You are a focused sub-agent executing a specific task. Provide clear, structured output.",
        max_context_tokens: int = 4000,
        model_name: str = "gpt-4",
    ):
        """
        Initialize the ContextShardingManager.

        Args:
            default_system_prompt: Default system instructions for sub-agents
            max_context_tokens: Maximum tokens allowed per context shard
            model_name: Model name for token counting
        """
        self.default_system_prompt = default_system_prompt
        self.max_context_tokens = max_context_tokens
        self.model_name = model_name

    def create_shard(
        self,
        sub_task: SubTask,
        parent_context: str,
        custom_system_prompt: Optional[str] = None,
    ) -> ContextShard:
        """
        Create an isolated context shard for a sub-agent.

        Extracts relevant portions from parent_context based on context_hints
        from the sub-task. The shard is immutable once created.

        Args:
            sub_task: The SubTask object containing task details and context hints
            parent_context: The full parent context to extract from
            custom_system_prompt: Optional custom system prompt (overrides default)

        Returns:
            ContextShard: Immutable context shard for the sub-agent
        """
        # Use custom system prompt if provided, otherwise default
        system_prompt = custom_system_prompt or self.default_system_prompt

        # Extract relevant context based on hints
        relevant_context = self._extract_relevant_context(
            parent_context=parent_context,
            context_hints=sub_task.context_hints or [],
        )

        # Ensure we don't exceed token budget
        relevant_context = self._truncate_to_budget(
            context=relevant_context,
            task_description=sub_task.description,
            system_prompt=system_prompt,
        )

        # Calculate total token count
        total_tokens = self._count_shard_tokens(
            system_prompt=system_prompt,
            task_description=sub_task.description,
            context=relevant_context,
        )

        # Create immutable shard
        shard = ContextShard(
            system_prompt=system_prompt,
            task_description=sub_task.description,
            relevant_context_slice=relevant_context,
            output_format=sub_task.expected_output_format or "markdown",
            token_count=total_tokens,
        )

        return shard

    def collect_result(
        self,
        agent_id: str,
        sub_task_id: str,
        result: str,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StructuredResult:
        """
        Collect and structure output from a sub-agent.

        Args:
            agent_id: Unique identifier of the sub-agent
            sub_task_id: ID of the completed sub-task
            result: The output from the sub-agent
            confidence: Confidence score (0.0-1.0)
            metadata: Additional metadata (execution time, token usage, etc.)

        Returns:
            StructuredResult: Structured result object
        """
        return StructuredResult(
            agent_id=agent_id,
            sub_task_id=sub_task_id,
            output=result,
            confidence=confidence,
            metadata=metadata or {},
        )

    def _extract_relevant_context(
        self,
        parent_context: str,
        context_hints: List[str],
    ) -> str:
        """
        Extract relevant portions from parent context based on hints.

        Uses simple keyword matching. For more advanced extraction,
        this could be replaced with semantic search or embeddings.

        Args:
            parent_context: Full parent context
            context_hints: Keywords/phrases to look for

        Returns:
            str: Extracted relevant context
        """
        if not context_hints or not parent_context:
            # No hints provided, return empty context
            return ""

        # Split parent context into chunks (paragraphs or sentences)
        chunks = self._split_into_chunks(parent_context)

        # Find chunks containing any of the context hints
        relevant_chunks = []
        for chunk in chunks:
            chunk_lower = chunk.lower()
            if any(hint.lower() in chunk_lower for hint in context_hints):
                relevant_chunks.append(chunk)

        # Join relevant chunks
        return "\n\n".join(relevant_chunks)

    def _split_into_chunks(self, text: str) -> List[str]:
        """
        Split text into chunks (paragraphs).

        Args:
            text: Text to split

        Returns:
            List[str]: List of text chunks
        """
        # Split by double newline (paragraphs)
        chunks = text.split("\n\n")
        # Filter out empty chunks
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        return chunks

    def _truncate_to_budget(
        self,
        context: str,
        task_description: str,
        system_prompt: str,
    ) -> str:
        """
        Truncate context to fit within token budget.

        Ensures that system_prompt + task_description + context
        doesn't exceed max_context_tokens.

        Args:
            context: Context to potentially truncate
            task_description: Task description
            system_prompt: System prompt

        Returns:
            str: Truncated context if needed
        """
        # Calculate tokens for fixed parts
        fixed_tokens = self._count_shard_tokens(
            system_prompt=system_prompt,
            task_description=task_description,
            context="",
        )

        # Calculate available tokens for context
        available_tokens = self.max_context_tokens - fixed_tokens

        if available_tokens <= 0:
            # No room for context
            return ""

        # Count context tokens
        context_tokens = count_tokens(text=context, model=self.model_name)

        if context_tokens <= available_tokens:
            # Context fits
            return context

        # Truncate context by character ratio (approximate)
        # This is a simple approximation; more sophisticated approaches could use
        # actual token-level truncation
        ratio = available_tokens / context_tokens
        truncate_at = int(len(context) * ratio * 0.95)  # 0.95 for safety margin

        return context[:truncate_at] + "\n\n[Context truncated to fit token budget]"

    def _count_shard_tokens(
        self,
        system_prompt: str,
        task_description: str,
        context: str,
    ) -> int:
        """
        Count total tokens in a context shard.

        Args:
            system_prompt: System prompt text
            task_description: Task description text
            context: Context text

        Returns:
            int: Total token count
        """
        # Combine all text
        combined = f"{system_prompt}\n\n{task_description}\n\n{context}"

        # Count tokens
        return count_tokens(text=combined, model=self.model_name)
