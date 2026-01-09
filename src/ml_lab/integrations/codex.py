"""
Codex CLI integration for delegating code execution tasks.

Separation of concerns:
- Planner LLM (Claude): reasoning, architecture, tradeoffs
- Executor LLM (Codex): precise code edits + CLI work
- World tools (ML Lab): data, training, infra, deployment
"""

from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..security.audit import AuditAction, AuditCategory, get_audit_log


@dataclass
class CodexResult:
    """Result from a Codex execution."""

    success: bool
    output: str
    error: str | None = None
    files_modified: list[str] | None = None


class CodexClient:
    """
    Client for delegating tasks to Codex CLI.

    Codex handles precise code edits and CLI operations while
    ML Lab handles domain knowledge (datasets, training, infra).
    """

    def __init__(
        self,
        codex_path: str | None = None,
        default_profile: str = "coder",
        timeout: int = 300,
    ):
        """
        Initialize Codex client.

        Args:
            codex_path: Path to codex CLI (auto-detected if not provided).
            default_profile: Default Codex profile to use.
            timeout: Default timeout in seconds.
        """
        self._codex_path = codex_path or self._find_codex()
        self._default_profile = default_profile
        self._timeout = timeout

    def _find_codex(self) -> str | None:
        """Find codex CLI in PATH."""
        return shutil.which("codex")

    @property
    def available(self) -> bool:
        """Check if Codex CLI is available."""
        return self._codex_path is not None

    async def run(
        self,
        prompt: str,
        profile: str | None = None,
        working_dir: str | None = None,
        timeout: int | None = None,
    ) -> CodexResult:
        """
        Run a task with Codex CLI.

        Args:
            prompt: Task description for Codex.
            profile: Codex profile (coder, fast, heavy, reasoning, security).
            working_dir: Working directory for execution.
            timeout: Timeout in seconds.

        Returns:
            CodexResult with output and status.
        """
        if not self.available:
            return CodexResult(
                success=False,
                output="",
                error="Codex CLI not found. Install with: npm install -g @anthropic/codex",
            )

        audit = get_audit_log()
        profile = profile or self._default_profile
        timeout = timeout or self._timeout

        cmd = [
            self._codex_path,
            "--profile", profile,
            "--json",  # Get structured output
            prompt,
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )

            output = stdout.decode()
            error_output = stderr.decode()

            # Try to parse JSON output
            try:
                result_data = json.loads(output)
                output_text = result_data.get("output", output)
                files_modified = result_data.get("files_modified", [])
            except json.JSONDecodeError:
                output_text = output
                files_modified = None

            success = proc.returncode == 0

            audit.log(
                AuditCategory.SECURITY,
                AuditAction.CRED_GET,  # Reusing for now, could add CODEX_RUN
                target="codex",
                success=success,
                details={"profile": profile, "prompt_preview": prompt[:100]},
                error=error_output if not success else None,
            )

            return CodexResult(
                success=success,
                output=output_text,
                error=error_output if error_output else None,
                files_modified=files_modified,
            )

        except asyncio.TimeoutError:
            return CodexResult(
                success=False,
                output="",
                error=f"Codex execution timed out after {timeout}s",
            )
        except Exception as e:
            return CodexResult(
                success=False,
                output="",
                error=str(e),
            )

    async def analyze_error(
        self,
        error_message: str,
        context: str | None = None,
        log_content: str | None = None,
    ) -> CodexResult:
        """
        Have Codex analyze an error and suggest fixes.

        Args:
            error_message: The error message to analyze.
            context: Additional context (file path, operation, etc.).
            log_content: Relevant log content.

        Returns:
            CodexResult with diagnosis and suggested fixes.
        """
        prompt_parts = [
            "Analyze this error and provide a diagnosis with suggested fixes.",
            "",
            f"Error: {error_message}",
        ]

        if context:
            prompt_parts.append(f"\nContext: {context}")

        if log_content:
            # Truncate if too long
            if len(log_content) > 2000:
                log_content = log_content[-2000:]
            prompt_parts.append(f"\nRecent logs:\n```\n{log_content}\n```")

        prompt_parts.append("\nProvide: 1) Root cause 2) Fix steps 3) Prevention")

        return await self.run(
            "\n".join(prompt_parts),
            profile="reasoning",
        )

    async def generate_training_script(
        self,
        base_model: str,
        method: str,
        dataset_path: str,
        output_dir: str,
        config: dict[str, Any] | None = None,
    ) -> CodexResult:
        """
        Generate a training script using Codex.

        Args:
            base_model: Base model to fine-tune.
            method: Training method (lora, qlora, full, sft).
            dataset_path: Path to training dataset.
            output_dir: Output directory for checkpoints.
            config: Additional training configuration.

        Returns:
            CodexResult with generated script.
        """
        config = config or {}

        prompt = f"""Generate a Python training script for fine-tuning with these specs:

Base model: {base_model}
Method: {method}
Dataset: {dataset_path}
Output dir: {output_dir}
Config: {json.dumps(config, indent=2)}

Requirements:
- Use transformers, peft, trl libraries
- Include proper logging and checkpointing
- Handle CUDA memory efficiently
- Save adapter weights (if LoRA/QLoRA) or full model
- Include training metrics logging

Output only the Python code, no explanations."""

        return await self.run(prompt, profile="coder")

    async def fix_code(
        self,
        file_path: str,
        issue_description: str,
        error_message: str | None = None,
    ) -> CodexResult:
        """
        Have Codex fix issues in code.

        Args:
            file_path: Path to the file to fix.
            issue_description: Description of the issue.
            error_message: Associated error message.

        Returns:
            CodexResult with fix applied.
        """
        prompt_parts = [
            f"Fix the issue in {file_path}:",
            "",
            f"Issue: {issue_description}",
        ]

        if error_message:
            prompt_parts.append(f"\nError: {error_message}")

        prompt_parts.append("\nApply the fix directly to the file.")

        return await self.run(
            "\n".join(prompt_parts),
            profile="coder",
            working_dir=str(Path(file_path).parent),
        )

    async def optimize_config(
        self,
        base_model: str,
        dataset_size: int,
        gpu_memory_gb: int,
        current_config: dict[str, Any],
        goal: str = "quality",
    ) -> CodexResult:
        """
        Have Codex optimize training configuration.

        Args:
            base_model: Model being trained.
            dataset_size: Number of training samples.
            gpu_memory_gb: Available GPU memory.
            current_config: Current training config.
            goal: Optimization goal (quality, speed, memory).

        Returns:
            CodexResult with optimized config.
        """
        prompt = f"""Optimize this training configuration:

Model: {base_model}
Dataset size: {dataset_size} samples
GPU memory: {gpu_memory_gb} GB
Current config: {json.dumps(current_config, indent=2)}
Optimization goal: {goal}

Provide an optimized config as JSON with explanations for each change."""

        return await self.run(prompt, profile="reasoning")

    async def debug_training_issue(
        self,
        logs: str,
        config: dict[str, Any],
        error: str | None = None,
    ) -> CodexResult:
        """
        Debug training issues using Codex.

        Args:
            logs: Training logs.
            config: Training configuration.
            error: Specific error if any.

        Returns:
            CodexResult with diagnosis and fixes.
        """
        # Truncate logs if too long
        if len(logs) > 5000:
            logs = logs[-5000:]

        prompt = f"""Debug this training issue:

Config: {json.dumps(config, indent=2)}

Logs:
```
{logs}
```
"""
        if error:
            prompt += f"\nError: {error}"

        prompt += """

Provide:
1. What's going wrong
2. Root cause
3. Specific fix (code changes if needed)
4. Prevention tips"""

        return await self.run(prompt, profile="reasoning")

    async def generate_eval_script(
        self,
        model_path: str,
        eval_type: str,
        dataset_path: str | None = None,
    ) -> CodexResult:
        """
        Generate an evaluation script.

        Args:
            model_path: Path to model or Ollama model name.
            eval_type: Type of eval (perplexity, accuracy, generation).
            dataset_path: Optional eval dataset.

        Returns:
            CodexResult with eval script.
        """
        prompt = f"""Generate a Python evaluation script:

Model: {model_path}
Evaluation type: {eval_type}
Dataset: {dataset_path or 'None (use generation prompts)'}

Requirements:
- Load model efficiently
- Compute relevant metrics
- Output results as JSON
- Handle errors gracefully

Output only the Python code."""

        return await self.run(prompt, profile="coder")


# Singleton client
_client: CodexClient | None = None


def get_codex_client() -> CodexClient:
    """Get or create the global Codex client."""
    global _client
    if _client is None:
        _client = CodexClient()
    return _client
