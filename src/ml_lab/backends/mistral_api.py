"""Mistral fine-tuning API backend."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx

from ..credentials import ProviderType, get_vault
from .base import (
    BackendCapabilities,
    TrainingBackend,
    TrainingConfig,
    TrainingMethod,
    TrainingRun,
    TrainingStatus,
)


@dataclass
class MistralJob:
    """Mistral fine-tuning job state."""

    job_id: str
    status: str
    model: str
    training_file: str
    created_at: int
    modified_at: int
    fine_tuned_model: str | None = None
    error: str | None = None
    trained_tokens: int | None = None


class MistralAPIBackend(TrainingBackend):
    """
    Mistral fine-tuning API backend.

    Uses Mistral's hosted fine-tuning service for their models.
    API docs: https://docs.mistral.ai/guides/finetuning/
    """

    API_BASE = "https://api.mistral.ai/v1"

    # Supported base models for fine-tuning
    SUPPORTED_MODELS = [
        "open-mistral-7b",
        "mistral-small-latest",
        "codestral-latest",
        "open-mixtral-8x7b",
        "open-mixtral-8x22b",
    ]

    def __init__(self):
        self._api_key: str | None = None
        self._jobs: dict[str, TrainingRun] = {}

    def _get_api_key(self) -> str:
        """Get the API key from the credential vault."""
        if self._api_key:
            return self._api_key

        vault = get_vault()
        if not vault.is_unlocked:
            raise RuntimeError("Credential vault is locked")

        cred = vault.get(ProviderType.MISTRAL)
        if not cred or not cred.api_key:
            raise ValueError("Mistral API key not configured")

        self._api_key = cred.api_key
        return self._api_key

    def _headers(self) -> dict[str, str]:
        """Get request headers with authentication."""
        return {
            "Authorization": f"Bearer {self._get_api_key()}",
            "Content-Type": "application/json",
        }

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name="mistral_api",
            supports_local=False,
            supports_remote=True,
            supports_distributed=True,  # Handled by Mistral
            supported_methods=[TrainingMethod.SFT, TrainingMethod.LORA],
            max_model_size_b=141.0,  # Mixtral 8x22B
            supported_quantization=[],  # Mistral handles this
            supports_streaming_logs=False,  # API doesn't stream
        )

    async def validate_config(self, config: TrainingConfig) -> list[str]:
        errors = []

        # Check model is supported
        model_supported = any(
            m in config.base_model.lower() for m in ["mistral", "mixtral", "codestral"]
        )
        if not model_supported:
            errors.append(
                f"Model {config.base_model} is not supported by Mistral API. "
                f"Supported models: {', '.join(self.SUPPORTED_MODELS)}"
            )

        # Mistral API has specific requirements
        if config.method not in [TrainingMethod.SFT, TrainingMethod.LORA]:
            errors.append("Mistral API only supports SFT and LoRA fine-tuning")

        return errors

    async def estimate_resources(
        self,
        config: TrainingConfig,
        dataset_size: int,
    ) -> dict[str, Any]:
        # Mistral pricing is per 1M tokens
        # Estimate tokens from dataset size
        avg_tokens_per_sample = 500  # Rough estimate
        total_tokens = dataset_size * avg_tokens_per_sample * config.epochs

        # Mistral fine-tuning pricing (approximate)
        if "7b" in config.base_model.lower():
            price_per_1m_tokens = 2.0
            base_model_size = 7
        elif "8x7b" in config.base_model.lower():
            price_per_1m_tokens = 6.0
            base_model_size = 47
        elif "8x22b" in config.base_model.lower():
            price_per_1m_tokens = 12.0
            base_model_size = 141
        else:
            price_per_1m_tokens = 4.0
            base_model_size = 7

        estimated_cost = (total_tokens / 1_000_000) * price_per_1m_tokens

        # Time estimate (very rough, depends on queue)
        tokens_per_second = 10000  # Mistral's infrastructure is fast
        estimated_seconds = total_tokens / tokens_per_second

        return {
            "estimated_tokens": total_tokens,
            "estimated_cost_usd": round(estimated_cost, 2),
            "estimated_time_seconds": int(estimated_seconds),
            "estimated_time_human": _format_duration(estimated_seconds),
            "model_size_b": base_model_size,
            "note": "Actual time may vary based on queue position",
        }

    async def _upload_training_file(self, file_path: str) -> str:
        """Upload a training file to Mistral."""
        import os

        async with httpx.AsyncClient() as client:
            with open(file_path, "rb") as f:
                files = {"file": (os.path.basename(file_path), f, "application/jsonl")}
                resp = await client.post(
                    f"{self.API_BASE}/files",
                    headers={"Authorization": f"Bearer {self._get_api_key()}"},
                    files=files,
                    data={"purpose": "fine-tune"},
                    timeout=300.0,
                )
                resp.raise_for_status()
                return resp.json()["id"]

    async def launch(
        self,
        run: TrainingRun,
        dataset_path: str,
        output_dir: str,  # Not used for API backend
    ) -> str:
        # Upload training file
        file_id = await self._upload_training_file(dataset_path)

        # Create fine-tuning job
        payload = {
            "model": run.config.base_model,
            "training_files": [file_id],
            "hyperparameters": {
                "learning_rate": run.config.learning_rate,
                "n_epochs": run.config.epochs,
            },
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.API_BASE}/fine_tuning/jobs",
                headers=self._headers(),
                json=payload,
                timeout=60.0,
            )
            resp.raise_for_status()
            job_data = resp.json()

        run.run_id = job_data["id"]
        run.status = TrainingStatus.PENDING
        self._jobs[run.run_id] = run

        return run.run_id

    async def get_status(self, run_id: str) -> TrainingRun:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.API_BASE}/fine_tuning/jobs/{run_id}",
                headers=self._headers(),
                timeout=30.0,
            )
            resp.raise_for_status()
            job_data = resp.json()

        status_map = {
            "QUEUED": TrainingStatus.PENDING,
            "RUNNING": TrainingStatus.RUNNING,
            "SUCCEEDED": TrainingStatus.COMPLETED,
            "FAILED": TrainingStatus.FAILED,
            "CANCELLED": TrainingStatus.CANCELLED,
        }

        if run_id in self._jobs:
            run = self._jobs[run_id]
        else:
            run = TrainingRun(
                run_id=run_id,
                experiment_id="",
                config=TrainingConfig(base_model=job_data.get("model", "")),
            )
            self._jobs[run_id] = run

        run.status = status_map.get(job_data.get("status", ""), TrainingStatus.PENDING)

        if job_data.get("fine_tuned_model"):
            run.checkpoint_path = job_data["fine_tuned_model"]

        if job_data.get("error"):
            run.error_message = str(job_data["error"])

        return run

    async def stream_logs(self, run_id: str) -> AsyncIterator[str]:
        """Mistral API doesn't support log streaming."""
        yield f"[Mistral API] Job {run_id} - use get_status to check progress"
        yield "[Mistral API] Log streaming not supported by Mistral API"

    async def stop(self, run_id: str, save_checkpoint: bool = True) -> None:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.API_BASE}/fine_tuning/jobs/{run_id}/cancel",
                headers=self._headers(),
                timeout=30.0,
            )
            resp.raise_for_status()

        if run_id in self._jobs:
            self._jobs[run_id].status = TrainingStatus.CANCELLED

    async def resume(self, run_id: str, checkpoint_path: str | None = None) -> str:
        raise NotImplementedError("Mistral API does not support resuming jobs")


def _format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"
