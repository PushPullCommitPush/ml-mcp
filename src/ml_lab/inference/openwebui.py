"""Open WebUI integration for model management and knowledge bases."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from ..credentials import ProviderType, get_vault


@dataclass
class OpenWebUIModel:
    """A model configuration in Open WebUI."""

    id: str
    name: str
    base_model_id: str
    meta: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None


@dataclass
class OpenWebUIKnowledge:
    """A knowledge base in Open WebUI."""

    id: str
    name: str
    description: str
    files: list[dict[str, Any]] = field(default_factory=list)
    created_at: str | None = None


@dataclass
class OpenWebUIStatus:
    """Status of Open WebUI."""

    connected: bool
    version: str | None = None
    models_count: int = 0
    knowledge_count: int = 0
    user: str | None = None
    error: str | None = None


class OpenWebUIClient:
    """
    Client for interacting with Open WebUI API.

    Open WebUI provides model presets, knowledge bases (RAG),
    tools, and pipelines on top of Ollama.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:3000",
        api_key: str | None = None,
    ):
        """
        Initialize the Open WebUI client.

        Args:
            base_url: Open WebUI base URL.
            api_key: API key for authentication (from vault if not provided).
        """
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key

    def _get_api_key(self) -> str | None:
        """Get API key from vault if not set."""
        if self._api_key:
            return self._api_key

        try:
            vault = get_vault()
            if vault.is_unlocked:
                cred = vault.get(ProviderType.OLLAMA)  # Reuse OLLAMA provider type
                if cred and cred.api_key:
                    return cred.api_key
        except Exception:
            pass

        return None

    def _headers(self) -> dict[str, str]:
        """Get request headers."""
        headers = {"Content-Type": "application/json"}
        api_key = self._get_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    async def status(self) -> OpenWebUIStatus:
        """Check Open WebUI connection status."""
        try:
            async with httpx.AsyncClient() as client:
                # Try to get config/version info
                resp = await client.get(
                    f"{self.base_url}/api/config",
                    headers=self._headers(),
                    timeout=10.0,
                )

                if resp.status_code == 401:
                    return OpenWebUIStatus(
                        connected=False,
                        error="Authentication required - add API key to vault",
                    )

                if resp.status_code != 200:
                    return OpenWebUIStatus(
                        connected=False,
                        error=f"Unexpected status: {resp.status_code}",
                    )

                # Get counts
                models = await self.list_models()
                knowledge = await self.list_knowledge()

                return OpenWebUIStatus(
                    connected=True,
                    models_count=len(models),
                    knowledge_count=len(knowledge),
                )

        except httpx.ConnectError:
            return OpenWebUIStatus(
                connected=False,
                error="Cannot connect to Open WebUI",
            )
        except Exception as e:
            return OpenWebUIStatus(connected=False, error=str(e))

    # =========================================================================
    # Model Management
    # =========================================================================

    async def list_models(self) -> list[OpenWebUIModel]:
        """List all model configurations."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.base_url}/api/models",
                headers=self._headers(),
                timeout=30.0,
            )

            if resp.status_code != 200:
                return []

            data = resp.json()

        models = []
        for item in data if isinstance(data, list) else data.get("models", []):
            models.append(
                OpenWebUIModel(
                    id=item.get("id", ""),
                    name=item.get("name", ""),
                    base_model_id=item.get("base_model_id", ""),
                    meta=item.get("meta", {}),
                    params=item.get("params", {}),
                    created_at=item.get("created_at"),
                    updated_at=item.get("updated_at"),
                )
            )

        return models

    async def get_model(self, model_id: str) -> OpenWebUIModel | None:
        """Get a specific model configuration."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.base_url}/api/models/{model_id}",
                headers=self._headers(),
                timeout=30.0,
            )

            if resp.status_code != 200:
                return None

            item = resp.json()

        return OpenWebUIModel(
            id=item.get("id", ""),
            name=item.get("name", ""),
            base_model_id=item.get("base_model_id", ""),
            meta=item.get("meta", {}),
            params=item.get("params", {}),
            created_at=item.get("created_at"),
            updated_at=item.get("updated_at"),
        )

    async def create_model(
        self,
        name: str,
        base_model_id: str,
        system_prompt: str | None = None,
        description: str | None = None,
        params: dict[str, Any] | None = None,
        knowledge_ids: list[str] | None = None,
    ) -> OpenWebUIModel:
        """
        Create a new model configuration.

        Args:
            name: Display name for the model.
            base_model_id: Base Ollama model (e.g., "llama3:latest").
            system_prompt: System prompt to use.
            description: Model description.
            params: Model parameters (temperature, top_p, etc.).
            knowledge_ids: Knowledge bases to attach.

        Returns:
            Created model configuration.
        """
        # Build model ID from name
        model_id = name.lower().replace(" ", "-")

        meta: dict[str, Any] = {}
        if description:
            meta["description"] = description
        if knowledge_ids:
            meta["knowledge"] = knowledge_ids

        payload = {
            "id": model_id,
            "name": name,
            "base_model_id": base_model_id,
            "meta": meta,
            "params": params or {},
        }

        if system_prompt:
            payload["params"]["system"] = system_prompt

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/api/models/create",
                headers=self._headers(),
                json=payload,
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()

        return OpenWebUIModel(
            id=data.get("id", model_id),
            name=data.get("name", name),
            base_model_id=data.get("base_model_id", base_model_id),
            meta=data.get("meta", meta),
            params=data.get("params", params or {}),
        )

    async def update_model(
        self,
        model_id: str,
        name: str | None = None,
        system_prompt: str | None = None,
        params: dict[str, Any] | None = None,
        knowledge_ids: list[str] | None = None,
    ) -> OpenWebUIModel | None:
        """Update an existing model configuration."""
        # Get current model
        current = await self.get_model(model_id)
        if not current:
            return None

        # Merge updates
        payload = {
            "id": model_id,
            "name": name or current.name,
            "base_model_id": current.base_model_id,
            "meta": current.meta.copy(),
            "params": {**current.params, **(params or {})},
        }

        if system_prompt:
            payload["params"]["system"] = system_prompt

        if knowledge_ids is not None:
            payload["meta"]["knowledge"] = knowledge_ids

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/api/models/update",
                headers=self._headers(),
                json=payload,
                timeout=30.0,
            )

            if resp.status_code != 200:
                return None

            data = resp.json()

        return OpenWebUIModel(
            id=data.get("id", model_id),
            name=data.get("name", payload["name"]),
            base_model_id=data.get("base_model_id", current.base_model_id),
            meta=data.get("meta", payload["meta"]),
            params=data.get("params", payload["params"]),
        )

    async def delete_model(self, model_id: str) -> bool:
        """Delete a model configuration."""
        async with httpx.AsyncClient() as client:
            resp = await client.delete(
                f"{self.base_url}/api/models/delete",
                headers=self._headers(),
                json={"id": model_id},
                timeout=30.0,
            )
            return resp.status_code == 200

    # =========================================================================
    # Knowledge Base Management
    # =========================================================================

    async def list_knowledge(self) -> list[OpenWebUIKnowledge]:
        """List all knowledge bases."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.base_url}/api/knowledge",
                headers=self._headers(),
                timeout=30.0,
            )

            if resp.status_code != 200:
                return []

            data = resp.json()

        knowledge = []
        items = data if isinstance(data, list) else data.get("knowledge", [])
        for item in items:
            knowledge.append(
                OpenWebUIKnowledge(
                    id=item.get("id", ""),
                    name=item.get("name", ""),
                    description=item.get("description", ""),
                    files=item.get("files", []),
                    created_at=item.get("created_at"),
                )
            )

        return knowledge

    async def create_knowledge(
        self,
        name: str,
        description: str = "",
    ) -> OpenWebUIKnowledge:
        """Create a new knowledge base."""
        payload = {
            "name": name,
            "description": description,
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/api/knowledge/create",
                headers=self._headers(),
                json=payload,
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()

        return OpenWebUIKnowledge(
            id=data.get("id", ""),
            name=data.get("name", name),
            description=data.get("description", description),
        )

    async def add_file_to_knowledge(
        self,
        knowledge_id: str,
        file_path: str,
    ) -> bool:
        """
        Add a file to a knowledge base.

        Supports: PDF, TXT, MD, DOCX, etc.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Determine content type
        suffix = path.suffix.lower()
        content_types = {
            ".pdf": "application/pdf",
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".csv": "text/csv",
            ".json": "application/json",
        }
        content_type = content_types.get(suffix, "application/octet-stream")

        async with httpx.AsyncClient() as client:
            with open(path, "rb") as f:
                files = {"file": (path.name, f, content_type)}
                resp = await client.post(
                    f"{self.base_url}/api/knowledge/{knowledge_id}/file/add",
                    headers={"Authorization": self._headers().get("Authorization", "")},
                    files=files,
                    timeout=120.0,
                )

            return resp.status_code == 200

    async def delete_knowledge(self, knowledge_id: str) -> bool:
        """Delete a knowledge base."""
        async with httpx.AsyncClient() as client:
            resp = await client.delete(
                f"{self.base_url}/api/knowledge/{knowledge_id}",
                headers=self._headers(),
                timeout=30.0,
            )
            return resp.status_code == 200

    # =========================================================================
    # Chat / Inference
    # =========================================================================

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        stream: bool = False,
    ) -> dict[str, Any]:
        """
        Send a chat request through Open WebUI.

        This uses Open WebUI's chat endpoint which applies
        model configurations, knowledge bases, etc.
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/api/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=300.0,
            )
            resp.raise_for_status()
            return resp.json()

    # =========================================================================
    # Export / Import
    # =========================================================================

    async def export_model_config(self, model_id: str) -> dict[str, Any] | None:
        """Export a model configuration for sharing."""
        model = await self.get_model(model_id)
        if not model:
            return None

        return {
            "id": model.id,
            "name": model.name,
            "base_model_id": model.base_model_id,
            "meta": model.meta,
            "params": model.params,
        }

    async def import_model_config(self, config: dict[str, Any]) -> OpenWebUIModel:
        """Import a model configuration."""
        return await self.create_model(
            name=config["name"],
            base_model_id=config["base_model_id"],
            system_prompt=config.get("params", {}).get("system"),
            description=config.get("meta", {}).get("description"),
            params=config.get("params"),
            knowledge_ids=config.get("meta", {}).get("knowledge"),
        )


# Singleton client
_client: OpenWebUIClient | None = None


def get_openwebui_client(
    base_url: str = "http://localhost:3000",
    api_key: str | None = None,
) -> OpenWebUIClient:
    """Get or create the global Open WebUI client."""
    global _client
    if _client is None:
        _client = OpenWebUIClient(base_url, api_key)
    return _client
