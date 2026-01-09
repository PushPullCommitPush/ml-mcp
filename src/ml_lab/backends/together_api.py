"""Together AI fine-tuning API backend."""

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


class TogetherAPIBackend(TrainingBackend):
    """
    Together AI fine-tuning API backend.

    Uses Together's hosted fine-tuning service.
    API docs: https://docs.together.ai/reference/fine-tuning
    """

    API_BASE = "https://api.together.xyz/v1"

    # Supported base models for fine-tuning
    SUPPORTED_MODELS = [
        "meta-llama/Llama-3-8b-hf",
        "meta-llama/Llama-3-70b-hf",
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Reference",
        "meta-llama/Meta-Llama-3.1-70B-Instruct-Reference",
        "mistralai/Mistral-7B-v0.1",
        "mistralai/Mixtral-8x7B-v0.1",
        "Qwen/Qwen2-7B",
        "Qwen/Qwen2-72B",
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

        cred = vault.get(ProviderType.TOGETHER)
        if not cred or not cred.api_key:
            raise ValueError("Together AI API key not configured")

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
            name="together_api",
            supports_local=False,
            supports_remote=True,
            supports_distributed=True,
            supported_methods=[TrainingMethod.SFT, TrainingMethod.LORA, TrainingMethod.FULL],
            max_model_size_b=72.0,
            supported_quantization=[],
            supports_streaming_logs=True,
        )

    async def validate_config(self, config: TrainingConfig) -> list[str]:
        errors = []

        if config.method == TrainingMethod.RLHF:
            errors.append("Together API does not support RLHF")

        if config.method == TrainingMethod.DPO:
            errors.append("Together API does not support DPO directly")

        return errors

    async def estimate_resources(
        self,
        config: TrainingConfig,
        dataset_size: int,
    ) -> dict[str, Any]:
        avg_tokens_per_sample = 500
        total_tokens = dataset_size * avg_tokens_per_sample * config.epochs

        # Together pricing varies by model
        model_lower = config.base_model.lower()
        if "70b" in model_lower or "72b" in model_lower:
            price_per_1m_tokens = 5.0
            model_size = 70
        elif "8x7b" in model_lower:
            price_per_1m_tokens = 3.0
            model_size = 47
        else:
            price_per_1m_tokens = 1.0
            model_size = 7

        estimated_cost = (total_tokens / 1_000_000) * price_per_1m_tokens
        tokens_per_second = 8000
        estimated_seconds = total_tokens / tokens_per_second

        return {
            "estimated_tokens": total_tokens,
            "estimated_cost_usd": round(estimated_cost, 2),
            "estimated_time_seconds": int(estimated_seconds),
            "estimated_time_human": _format_duration(estimated_seconds),
            "model_size_b": model_size,
        }

    async def _upload_training_file(self, file_path: str) -> str:
        """Upload a training file to Together."""
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

        # Map training method
        training_type = "Full" if run.config.method == TrainingMethod.FULL else "Lora"

        # Create fine-tuning job
        payload = {
            "model": run.config.base_model,
            "training_file": file_id,
            "n_epochs": run.config.epochs,
            "learning_rate": run.config.learning_rate,
            "batch_size": run.config.batch_size,
            "training_type": training_type,
        }

        if run.config.method in [TrainingMethod.LORA, TrainingMethod.QLORA]:
            payload["lora_r"] = run.config.lora_r
            payload["lora_alpha"] = run.config.lora_alpha
            payload["lora_dropout"] = run.config.lora_dropout

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.API_BASE}/fine-tunes",
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
                f"{self.API_BASE}/fine-tunes/{run_id}",
                headers=self._headers(),
                timeout=30.0,
            )
            resp.raise_for_status()
            job_data = resp.json()

        status_map = {
            "pending": TrainingStatus.PENDING,
            "queued": TrainingStatus.PENDING,
            "running": TrainingStatus.RUNNING,
            "completed": TrainingStatus.COMPLETED,
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

        if job_data.get("output_name"):
            run.checkpoint_path = job_data["output_name"]

        if job_data.get("training_steps"):
            run.total_steps = job_data["training_steps"]

        if job_data.get("current_step"):
            run.current_step = job_data["current_step"]

        return run

    async def stream_logs(self, run_id: str) -> AsyncIterator[str]:
        """Stream events from the training job."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.API_BASE}/fine-tunes/{run_id}/events",
                headers=self._headers(),
                timeout=30.0,
            )
            resp.raise_for_status()
            events = resp.json().get("data", [])

        for event in events:
            yield f"[{event.get('created_at', '')}] {event.get('message', '')}"

    async def stop(self, run_id: str, save_checkpoint: bool = True) -> None:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.API_BASE}/fine-tunes/{run_id}/cancel",
                headers=self._headers(),
                timeout=30.0,
            )
            resp.raise_for_status()

        if run_id in self._jobs:
            self._jobs[run_id].status = TrainingStatus.CANCELLED

    async def resume(self, run_id: str, checkpoint_path: str | None = None) -> str:
        raise NotImplementedError("Together API does not support resuming jobs")


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
