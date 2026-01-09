"""OpenAI fine-tuning API backend."""

from __future__ import annotations

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


class OpenAIAPIBackend(TrainingBackend):
    """
    OpenAI fine-tuning API backend.

    Uses OpenAI's hosted fine-tuning service for GPT models.
    API docs: https://platform.openai.com/docs/guides/fine-tuning
    """

    API_BASE = "https://api.openai.com/v1"

    SUPPORTED_MODELS = [
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06",
        "gpt-4-0613",
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-1106",
        "babbage-002",
        "davinci-002",
    ]

    # Pricing per 1M tokens (training)
    PRICING = {
        "gpt-4o-mini": 3.0,
        "gpt-4o": 25.0,
        "gpt-4": 30.0,
        "gpt-3.5-turbo": 8.0,
        "babbage": 0.4,
        "davinci": 6.0,
    }

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

        cred = vault.get(ProviderType.OPENAI)
        if not cred or not cred.api_key:
            raise ValueError("OpenAI API key not configured")

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
            name="openai_api",
            supports_local=False,
            supports_remote=True,
            supports_distributed=True,
            supported_methods=[TrainingMethod.SFT],  # OpenAI only supports SFT
            max_model_size_b=None,  # Unknown for GPT-4
            supported_quantization=[],
            supports_streaming_logs=True,
        )

    async def validate_config(self, config: TrainingConfig) -> list[str]:
        errors = []

        if config.method != TrainingMethod.SFT:
            errors.append("OpenAI fine-tuning only supports SFT (supervised fine-tuning)")

        # Check model is supported
        model_supported = any(m in config.base_model for m in self.SUPPORTED_MODELS)
        if not model_supported:
            errors.append(
                f"Model {config.base_model} is not supported for fine-tuning. "
                f"Supported: {', '.join(self.SUPPORTED_MODELS)}"
            )

        return errors

    async def estimate_resources(
        self,
        config: TrainingConfig,
        dataset_size: int,
    ) -> dict[str, Any]:
        avg_tokens_per_sample = 500
        total_tokens = dataset_size * avg_tokens_per_sample * config.epochs

        # Get pricing for model
        price_per_1m = 8.0  # Default
        for model_prefix, price in self.PRICING.items():
            if model_prefix in config.base_model.lower():
                price_per_1m = price
                break

        estimated_cost = (total_tokens / 1_000_000) * price_per_1m

        return {
            "estimated_tokens": total_tokens,
            "estimated_cost_usd": round(estimated_cost, 2),
            "estimated_time_human": "Varies based on queue",
            "note": "OpenAI doesn't provide time estimates",
        }

    async def _upload_training_file(self, file_path: str) -> str:
        """Upload a training file to OpenAI."""
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
        output_dir: str,
    ) -> str:
        # Upload training file
        file_id = await self._upload_training_file(dataset_path)

        # Create fine-tuning job
        payload = {
            "model": run.config.base_model,
            "training_file": file_id,
            "hyperparameters": {
                "n_epochs": run.config.epochs,
            },
        }

        # Add learning rate if significantly different from default
        if run.config.learning_rate != 2e-4:
            payload["hyperparameters"]["learning_rate_multiplier"] = (
                run.config.learning_rate / 2e-4
            )

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
            "validating_files": TrainingStatus.PENDING,
            "queued": TrainingStatus.PENDING,
            "running": TrainingStatus.RUNNING,
            "succeeded": TrainingStatus.COMPLETED,
            "failed": TrainingStatus.FAILED,
            "cancelled": TrainingStatus.CANCELLED,
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
            run.error_message = job_data["error"].get("message", str(job_data["error"]))

        if job_data.get("trained_tokens"):
            run.current_step = job_data["trained_tokens"]

        return run

    async def stream_logs(self, run_id: str) -> AsyncIterator[str]:
        """Stream events from the training job."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.API_BASE}/fine_tuning/jobs/{run_id}/events",
                headers=self._headers(),
                timeout=30.0,
            )
            resp.raise_for_status()
            events = resp.json().get("data", [])

        for event in events:
            level = event.get("level", "info")
            message = event.get("message", "")
            yield f"[{level.upper()}] {message}"

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
        raise NotImplementedError("OpenAI API does not support resuming jobs")
