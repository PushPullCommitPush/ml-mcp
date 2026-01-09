"""Google Vertex AI fine-tuning backend."""

from __future__ import annotations

import asyncio
import json
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
class VertexJob:
    """Vertex AI fine-tuning job state."""

    job_id: str
    status: str
    model: str
    tuned_model_name: str | None = None
    error: str | None = None
    create_time: str | None = None
    update_time: str | None = None


class VertexAPIBackend(TrainingBackend):
    """
    Google Vertex AI fine-tuning backend.

    Uses Vertex AI's tuning service for Gemini models.
    API docs: https://cloud.google.com/vertex-ai/docs/generative-ai/models/tune-models
    """

    # Supported base models for fine-tuning
    SUPPORTED_MODELS = [
        "gemini-1.0-pro-002",
        "gemini-1.5-flash-002",
        "gemini-1.5-pro-002",
    ]

    def __init__(self, project_id: str | None = None, location: str = "us-central1"):
        self._api_key: str | None = None
        self._project_id = project_id
        self._location = location
        self._jobs: dict[str, TrainingRun] = {}

    def _get_credentials(self) -> tuple[str, str]:
        """Get API key and project ID from the credential vault."""
        vault = get_vault()
        if not vault.is_unlocked:
            raise RuntimeError("Credential vault is locked")

        cred = vault.get(ProviderType.GCP)
        if not cred or not cred.api_key:
            raise ValueError("GCP API key not configured. Add with: creds_add provider=gcp")

        project_id = self._project_id or cred.extra.get("project_id")
        if not project_id:
            raise ValueError("GCP project_id not configured. Add to credential extras.")

        return cred.api_key, project_id

    def _get_api_base(self, project_id: str) -> str:
        """Get the Vertex AI API base URL."""
        return f"https://{self._location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{self._location}"

    def _headers(self, api_key: str) -> dict[str, str]:
        """Get request headers with authentication."""
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name="vertex_ai",
            supports_local=False,
            supports_remote=True,
            supports_distributed=True,  # Handled by Google
            supported_methods=[TrainingMethod.SFT],
            max_model_size_b=175.0,  # Gemini models
            supported_quantization=[],  # Vertex handles this
            supports_streaming_logs=False,
        )

    async def validate_config(self, config: TrainingConfig) -> list[str]:
        errors = []

        # Check model is supported
        model_supported = any(
            m in config.base_model.lower() for m in ["gemini"]
        )
        if not model_supported:
            errors.append(
                f"Model {config.base_model} is not supported by Vertex AI. "
                f"Supported models: {', '.join(self.SUPPORTED_MODELS)}"
            )

        if config.method != TrainingMethod.SFT:
            errors.append("Vertex AI currently only supports SFT fine-tuning for Gemini")

        return errors

    async def estimate_resources(
        self,
        config: TrainingConfig,
        dataset_size: int,
    ) -> dict[str, Any]:
        # Vertex AI pricing for Gemini tuning
        # https://cloud.google.com/vertex-ai/generative-ai/pricing
        avg_tokens_per_sample = 500
        total_tokens = dataset_size * avg_tokens_per_sample * config.epochs

        # Pricing varies by model
        if "flash" in config.base_model.lower():
            price_per_1m_tokens = 2.0  # Flash is cheaper
            base_model_size = 8
        elif "pro" in config.base_model.lower():
            price_per_1m_tokens = 4.0
            base_model_size = 50
        else:
            price_per_1m_tokens = 3.0
            base_model_size = 30

        estimated_cost = (total_tokens / 1_000_000) * price_per_1m_tokens

        # Time estimate
        tokens_per_second = 15000  # Google's infrastructure
        estimated_seconds = total_tokens / tokens_per_second

        return {
            "estimated_tokens": total_tokens,
            "estimated_cost_usd": round(estimated_cost, 2),
            "estimated_time_seconds": int(estimated_seconds),
            "estimated_time_human": _format_duration(estimated_seconds),
            "model_size_b": base_model_size,
            "note": "Actual time may vary based on queue position",
        }

    async def _upload_training_data(self, api_key: str, project_id: str, file_path: str) -> str:
        """
        Upload training data to GCS or return URI.

        For Vertex AI, training data should be in GCS. This method assumes
        the file_path is already a GCS URI or uploads to a default bucket.
        """
        # If already a GCS URI, return as-is
        if file_path.startswith("gs://"):
            return file_path

        # For local files, we'd need to upload to GCS
        # This requires additional setup (bucket name, etc.)
        raise ValueError(
            f"Training data must be a GCS URI (gs://bucket/path). "
            f"Got: {file_path}. Upload your data to GCS first."
        )

    async def launch(
        self,
        run: TrainingRun,
        dataset_path: str,
        output_dir: str,
    ) -> str:
        api_key, project_id = self._get_credentials()
        api_base = self._get_api_base(project_id)

        # Validate training data is in GCS
        training_data_uri = await self._upload_training_data(api_key, project_id, dataset_path)

        # Create tuning job
        # https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.tuningJobs
        payload = {
            "baseModel": run.config.base_model,
            "supervisedTuningSpec": {
                "trainingDatasetUri": training_data_uri,
                "hyperParameters": {
                    "epochCount": run.config.epochs,
                    "learningRateMultiplier": run.config.learning_rate / 1e-4,  # Relative to default
                },
            },
            "tunedModelDisplayName": f"tuned-{run.experiment_id}-{run.run_id[:8]}",
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{api_base}/tuningJobs",
                headers=self._headers(api_key),
                json=payload,
                timeout=60.0,
            )
            resp.raise_for_status()
            job_data = resp.json()

        # Extract job ID from name (format: projects/xxx/locations/xxx/tuningJobs/xxx)
        job_name = job_data.get("name", "")
        run.run_id = job_name.split("/")[-1] if "/" in job_name else job_name
        run.status = TrainingStatus.PENDING
        self._jobs[run.run_id] = run

        return run.run_id

    async def get_status(self, run_id: str) -> TrainingRun:
        api_key, project_id = self._get_credentials()
        api_base = self._get_api_base(project_id)

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{api_base}/tuningJobs/{run_id}",
                headers=self._headers(api_key),
                timeout=30.0,
            )
            resp.raise_for_status()
            job_data = resp.json()

        # Map Vertex AI states to our status
        # https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.tuningJobs#State
        status_map = {
            "JOB_STATE_PENDING": TrainingStatus.PENDING,
            "JOB_STATE_RUNNING": TrainingStatus.RUNNING,
            "JOB_STATE_SUCCEEDED": TrainingStatus.COMPLETED,
            "JOB_STATE_FAILED": TrainingStatus.FAILED,
            "JOB_STATE_CANCELLED": TrainingStatus.CANCELLED,
            "JOB_STATE_CANCELLING": TrainingStatus.RUNNING,
        }

        if run_id in self._jobs:
            run = self._jobs[run_id]
        else:
            run = TrainingRun(
                run_id=run_id,
                experiment_id="",
                config=TrainingConfig(base_model=job_data.get("baseModel", "")),
            )
            self._jobs[run_id] = run

        state = job_data.get("state", "")
        run.status = status_map.get(state, TrainingStatus.PENDING)

        # Get tuned model endpoint if complete
        if job_data.get("tunedModel"):
            tuned = job_data["tunedModel"]
            run.checkpoint_path = tuned.get("endpoint") or tuned.get("model")

        # Check for errors
        if job_data.get("error"):
            run.error_message = job_data["error"].get("message", str(job_data["error"]))

        return run

    async def stream_logs(self, run_id: str) -> AsyncIterator[str]:
        """Vertex AI doesn't support log streaming via API."""
        yield f"[Vertex AI] Job {run_id} - use get_status to check progress"
        yield "[Vertex AI] Log streaming not supported. Check Cloud Console for details."

    async def stop(self, run_id: str, save_checkpoint: bool = True) -> None:
        api_key, project_id = self._get_credentials()
        api_base = self._get_api_base(project_id)

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{api_base}/tuningJobs/{run_id}:cancel",
                headers=self._headers(api_key),
                timeout=30.0,
            )
            resp.raise_for_status()

        if run_id in self._jobs:
            self._jobs[run_id].status = TrainingStatus.CANCELLED

    async def resume(self, run_id: str, checkpoint_path: str | None = None) -> str:
        raise NotImplementedError("Vertex AI does not support resuming tuning jobs")


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
