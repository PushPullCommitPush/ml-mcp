"""Ollama integration for local model deployment and inference."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

import httpx


@dataclass
class OllamaModel:
    """Information about an Ollama model."""

    name: str
    size: int  # bytes
    digest: str
    modified_at: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class OllamaStatus:
    """Status of the Ollama service."""

    running: bool
    version: str | None = None
    models_count: int = 0
    gpu_available: bool = False
    gpu_name: str | None = None
    error: str | None = None


@dataclass
class ChatMessage:
    """A chat message."""

    role: str  # "user", "assistant", "system"
    content: str


@dataclass
class ChatResponse:
    """Response from a chat completion."""

    message: ChatMessage
    model: str
    done: bool
    total_duration: int | None = None  # nanoseconds
    eval_count: int | None = None  # tokens generated
    eval_duration: int | None = None  # nanoseconds


class OllamaClient:
    """
    Client for interacting with Ollama.

    Ollama API docs: https://github.com/ollama/ollama/blob/main/docs/api.md
    """

    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize the Ollama client.

        Args:
            base_url: Ollama API base URL.
        """
        self.base_url = base_url.rstrip("/")

    async def status(self) -> OllamaStatus:
        """Check Ollama service status."""
        try:
            async with httpx.AsyncClient() as client:
                # Check if Ollama is running
                resp = await client.get(f"{self.base_url}/api/version", timeout=5.0)
                if resp.status_code != 200:
                    return OllamaStatus(running=False, error="Ollama not responding")

                version = resp.json().get("version", "unknown")

                # Get model count
                models = await self.list_models()

                # Check GPU (via ps endpoint if available)
                gpu_available = False
                gpu_name = None
                try:
                    ps_resp = await client.get(f"{self.base_url}/api/ps", timeout=5.0)
                    if ps_resp.status_code == 200:
                        ps_data = ps_resp.json()
                        # If any model is loaded with GPU layers, GPU is available
                        for model in ps_data.get("models", []):
                            if model.get("size_vram", 0) > 0:
                                gpu_available = True
                                break
                except Exception:
                    pass

                return OllamaStatus(
                    running=True,
                    version=version,
                    models_count=len(models),
                    gpu_available=gpu_available,
                    gpu_name=gpu_name,
                )

        except httpx.ConnectError:
            return OllamaStatus(running=False, error="Cannot connect to Ollama")
        except Exception as e:
            return OllamaStatus(running=False, error=str(e))

    async def list_models(self) -> list[OllamaModel]:
        """List all available models."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{self.base_url}/api/tags", timeout=30.0)
            resp.raise_for_status()
            data = resp.json()

        models = []
        for model in data.get("models", []):
            models.append(
                OllamaModel(
                    name=model["name"],
                    size=model.get("size", 0),
                    digest=model.get("digest", ""),
                    modified_at=model.get("modified_at", ""),
                    details=model.get("details", {}),
                )
            )

        return models

    async def show_model(self, name: str) -> dict[str, Any]:
        """Get detailed information about a model."""
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/api/show",
                json={"name": name},
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json()

    async def pull_model(self, name: str) -> AsyncIterator[dict[str, Any]]:
        """
        Pull a model from the registry.

        Yields progress updates.
        """
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/pull",
                json={"name": name, "stream": True},
                timeout=None,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line:
                        yield json.loads(line)

    async def delete_model(self, name: str) -> bool:
        """Delete a model."""
        async with httpx.AsyncClient() as client:
            resp = await client.delete(
                f"{self.base_url}/api/delete",
                json={"name": name},
                timeout=30.0,
            )
            return resp.status_code == 200

    async def create_model(
        self,
        name: str,
        modelfile: str,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Create a model from a Modelfile.

        Args:
            name: Name for the new model.
            modelfile: Contents of the Modelfile.

        Yields:
            Progress updates.
        """
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/create",
                json={"name": name, "modelfile": modelfile, "stream": True},
                timeout=None,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line:
                        yield json.loads(line)

    async def deploy_gguf(
        self,
        name: str,
        gguf_path: str,
        system_prompt: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> bool:
        """
        Deploy a GGUF model to Ollama.

        Args:
            name: Name for the model in Ollama.
            gguf_path: Path to the GGUF file.
            system_prompt: Optional system prompt to bake in.
            parameters: Optional model parameters (temperature, etc.).

        Returns:
            True if successful.
        """
        gguf_path = Path(gguf_path).resolve()
        if not gguf_path.exists():
            raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

        # Build Modelfile
        modelfile_lines = [f'FROM "{gguf_path}"']

        if system_prompt:
            # Escape quotes in system prompt
            escaped = system_prompt.replace('"', '\\"')
            modelfile_lines.append(f'SYSTEM "{escaped}"')

        if parameters:
            for key, value in parameters.items():
                modelfile_lines.append(f"PARAMETER {key} {value}")

        modelfile = "\n".join(modelfile_lines)

        # Create the model
        async for progress in self.create_model(name, modelfile):
            status = progress.get("status", "")
            if "error" in progress:
                raise RuntimeError(f"Failed to create model: {progress['error']}")

        return True

    async def chat(
        self,
        model: str,
        messages: list[ChatMessage] | list[dict[str, str]],
        stream: bool = False,
        options: dict[str, Any] | None = None,
    ) -> ChatResponse | AsyncIterator[ChatResponse]:
        """
        Send a chat completion request.

        Args:
            model: Model name.
            messages: List of messages.
            stream: Whether to stream the response.
            options: Model options (temperature, etc.).

        Returns:
            ChatResponse or async iterator of responses if streaming.
        """
        # Convert ChatMessage objects to dicts
        msg_dicts = []
        for msg in messages:
            if isinstance(msg, ChatMessage):
                msg_dicts.append({"role": msg.role, "content": msg.content})
            else:
                msg_dicts.append(msg)

        payload = {
            "model": model,
            "messages": msg_dicts,
            "stream": stream,
        }
        if options:
            payload["options"] = options

        if stream:
            return self._stream_chat(payload)
        else:
            return await self._chat(payload)

    async def _chat(self, payload: dict[str, Any]) -> ChatResponse:
        """Non-streaming chat."""
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=300.0,
            )
            resp.raise_for_status()
            data = resp.json()

        return ChatResponse(
            message=ChatMessage(
                role=data["message"]["role"],
                content=data["message"]["content"],
            ),
            model=data["model"],
            done=data.get("done", True),
            total_duration=data.get("total_duration"),
            eval_count=data.get("eval_count"),
            eval_duration=data.get("eval_duration"),
        )

    async def _stream_chat(
        self,
        payload: dict[str, Any],
    ) -> AsyncIterator[ChatResponse]:
        """Streaming chat."""
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=None,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line:
                        data = json.loads(line)
                        yield ChatResponse(
                            message=ChatMessage(
                                role=data.get("message", {}).get("role", "assistant"),
                                content=data.get("message", {}).get("content", ""),
                            ),
                            model=data.get("model", ""),
                            done=data.get("done", False),
                            total_duration=data.get("total_duration"),
                            eval_count=data.get("eval_count"),
                            eval_duration=data.get("eval_duration"),
                        )

    async def generate(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        """
        Simple text generation.

        Args:
            model: Model name.
            prompt: The prompt.
            system: Optional system prompt.
            options: Model options.

        Returns:
            Generated text.
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        if options:
            payload["options"] = options

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=300.0,
            )
            resp.raise_for_status()
            return resp.json()["response"]

    async def embeddings(
        self,
        model: str,
        prompt: str,
    ) -> list[float]:
        """
        Generate embeddings for text.

        Args:
            model: Model name (e.g., "nomic-embed-text").
            prompt: Text to embed.

        Returns:
            Embedding vector.
        """
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": model, "prompt": prompt},
                timeout=60.0,
            )
            resp.raise_for_status()
            return resp.json()["embedding"]

    async def copy_model(self, source: str, destination: str) -> bool:
        """Copy a model to a new name."""
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/api/copy",
                json={"source": source, "destination": destination},
                timeout=30.0,
            )
            return resp.status_code == 200


# Singleton client
_client: OllamaClient | None = None


def get_ollama_client(base_url: str = "http://localhost:11434") -> OllamaClient:
    """Get or create the global Ollama client."""
    global _client
    if _client is None:
        _client = OllamaClient(base_url)
    return _client
