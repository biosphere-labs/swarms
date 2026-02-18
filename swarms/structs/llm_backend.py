"""
LLM Backend Abstraction — Pluggable LLM calling for decomposition and synthesis.

Provides a Protocol-based abstraction over LLM inference so that components
(DecompositionEngine, ResultAggregator) can use different backends:

- LiteLLMBackend: Wraps litellm.completion() for any provider (DeepInfra, OpenAI, etc.)
- ClaudeCodeBackend: Spawns `claude -p` subprocess, uses Claude subscription (no API key)

The ClaudeCodeBackend inlines the core logic from claude-headless-subscription
(~80 lines of subprocess management) to avoid an external dependency.
"""

import os
import subprocess
import tempfile
from typing import Optional, Protocol, runtime_checkable

from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger(log_folder="llm_backend")


@runtime_checkable
class LLMBackend(Protocol):
    """Protocol for pluggable LLM inference backends."""

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        api_key: Optional[str] = None,
    ) -> str:
        """
        Call the LLM and return the text response.

        Args:
            system_prompt: System message for the LLM
            user_prompt: User message for the LLM
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            api_key: Optional API key (used by LiteLLM, ignored by Claude)

        Returns:
            The LLM's text response
        """
        ...


class LiteLLMBackend:
    """
    LLM backend wrapping litellm.completion().

    Supports any model/provider that litellm supports (OpenAI, Anthropic,
    DeepInfra, Groq, Ollama, etc.).
    """

    def __init__(self, model: str):
        self.model = model

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        api_key: Optional[str] = None,
    ) -> str:
        import litellm

        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key

        response = litellm.completion(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return response.choices[0].message.content.strip()


class ClaudeCodeBackend:
    """
    LLM backend that spawns `claude -p` as a subprocess.

    Uses the Claude subscription (not API credits) by clearing ANTHROPIC_API_KEY
    from the environment. The prompt is piped to stdin and the response is
    captured from stdout.

    This inlines the core logic from the claude-headless-subscription package
    (~80 lines of subprocess management) to avoid an external dependency.
    """

    def __init__(self, timeout: int = 300):
        """
        Args:
            timeout: Maximum time in seconds to wait for Claude response
        """
        self.timeout = timeout

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        api_key: Optional[str] = None,
    ) -> str:
        # Build the combined prompt (claude -p takes a single prompt via stdin)
        prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"

        # Write prompt to a temp file
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        )
        try:
            tmp.write(prompt)
            tmp.close()

            # Build environment: clear ANTHROPIC_API_KEY to force subscription
            env = {**os.environ, "ANTHROPIC_API_KEY": ""}

            # Spawn: claude -p (non-interactive pipe mode)
            # --dangerously-skip-permissions avoids hanging on tool
            # permission prompts if Claude attempts tool use during synthesis
            proc = subprocess.Popen(
                [
                    "claude",
                    "-p",
                    "--output-format", "text",
                    "--dangerously-skip-permissions",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )

            # Pipe the prompt via stdin
            with open(tmp.name, "r") as f:
                prompt_bytes = f.read().encode("utf-8")

            stdout, stderr = proc.communicate(
                input=prompt_bytes, timeout=self.timeout
            )

            if proc.returncode != 0:
                error_msg = stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(
                    f"claude -p exited with code {proc.returncode}: {error_msg}"
                )

            return stdout.decode("utf-8", errors="replace").strip()

        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise TimeoutError(
                f"Claude subprocess timed out after {self.timeout}s"
            )
        finally:
            os.unlink(tmp.name)


def create_llm_backend(
    backend_type: str = "litellm",
    model: str = "gpt-4o-mini",
    timeout: int = 300,
) -> LLMBackend:
    """
    Factory function to create an LLM backend.

    Args:
        backend_type: "litellm" or "claude-code"
        model: Model name (used by litellm backend, ignored by claude-code)
        timeout: Timeout in seconds (used by claude-code backend)

    Returns:
        An LLMBackend instance
    """
    if backend_type == "claude-code":
        logger.info(f"Creating ClaudeCodeBackend (timeout={timeout}s)")
        return ClaudeCodeBackend(timeout=timeout)
    elif backend_type == "litellm":
        logger.info(f"Creating LiteLLMBackend (model={model})")
        return LiteLLMBackend(model=model)
    else:
        raise ValueError(
            f"Unknown backend type: {backend_type!r}. "
            f"Expected 'litellm' or 'claude-code'."
        )
