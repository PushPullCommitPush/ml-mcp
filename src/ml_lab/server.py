"""
ML Lab MCP Server.

A comprehensive MCP server for model training, fine-tuning, and experimentation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .backends.base import TrainingConfig, TrainingMethod, TrainingRun
from .backends.local import LocalBackend
from .backends.mistral_api import MistralAPIBackend
from .backends.openai_api import OpenAIAPIBackend
from .backends.together_api import TogetherAPIBackend
from .cloud.base import GPUType
from .cloud.lambda_labs import LambdaLabsProvider
from .cloud.remote_vps import get_vps_manager
from .cloud.runpod import RunPodProvider
from .credentials import CredentialVault, ProviderCredential, ProviderType, get_vault
from .inference.ollama import get_ollama_client
from .inference.openwebui import get_openwebui_client
from .security.audit import AuditAction, AuditCategory, get_audit_log
from .storage.datasets import get_dataset_manager
from .storage.experiments import get_experiment_store

# Initialize the MCP server
server = Server("ml-lab")

# Backend registry
TRAINING_BACKENDS = {
    "local": LocalBackend,
    "mistral": MistralAPIBackend,
    "together": TogetherAPIBackend,
    "openai": OpenAIAPIBackend,
}

CLOUD_PROVIDERS = {
    "lambda_labs": LambdaLabsProvider,
    "runpod": RunPodProvider,
}


# ============================================================================
# Credential Management Tools
# ============================================================================


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available tools."""
    return [
        # Credentials
        Tool(
            name="creds_create_vault",
            description="Create a new encrypted credential vault with a password",
            inputSchema={
                "type": "object",
                "properties": {
                    "password": {
                        "type": "string",
                        "description": "Password to encrypt the vault",
                    },
                },
                "required": ["password"],
            },
        ),
        Tool(
            name="creds_unlock",
            description="Unlock the credential vault with a password",
            inputSchema={
                "type": "object",
                "properties": {
                    "password": {
                        "type": "string",
                        "description": "Password to unlock the vault",
                    },
                },
                "required": ["password"],
            },
        ),
        Tool(
            name="creds_add",
            description="Add or update credentials for a provider",
            inputSchema={
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "description": "Provider name (lambda_labs, runpod, mistral, openai, together, huggingface)",
                        "enum": [p.value for p in ProviderType],
                    },
                    "api_key": {
                        "type": "string",
                        "description": "API key for the provider",
                    },
                    "api_secret": {
                        "type": "string",
                        "description": "Optional API secret (for providers that require it)",
                    },
                },
                "required": ["provider", "api_key"],
            },
        ),
        Tool(
            name="creds_list",
            description="List all configured providers (does not show keys)",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="creds_test",
            description="Test credentials for a specific provider",
            inputSchema={
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "description": "Provider to test",
                        "enum": [p.value for p in ProviderType],
                    },
                },
                "required": ["provider"],
            },
        ),
        # Datasets
        Tool(
            name="dataset_register",
            description="Register a dataset from a local file",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the dataset file (JSONL, CSV, Parquet)",
                    },
                    "name": {
                        "type": "string",
                        "description": "Optional name for the dataset",
                    },
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="dataset_list",
            description="List all registered datasets",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="dataset_inspect",
            description="Inspect a dataset's schema and statistics",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "string",
                        "description": "Dataset ID to inspect",
                    },
                },
                "required": ["dataset_id"],
            },
        ),
        Tool(
            name="dataset_preview",
            description="Preview samples from a dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "string",
                        "description": "Dataset ID to preview",
                    },
                    "num_samples": {
                        "type": "integer",
                        "description": "Number of samples to show (default 5)",
                        "default": 5,
                    },
                },
                "required": ["dataset_id"],
            },
        ),
        Tool(
            name="dataset_split",
            description="Split a dataset into train/val/test sets",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "string",
                        "description": "Dataset ID to split",
                    },
                    "train_ratio": {
                        "type": "number",
                        "description": "Ratio for training set (default 0.9)",
                        "default": 0.9,
                    },
                    "val_ratio": {
                        "type": "number",
                        "description": "Ratio for validation set (default 0.1)",
                        "default": 0.1,
                    },
                },
                "required": ["dataset_id"],
            },
        ),
        Tool(
            name="dataset_transform",
            description="Transform a dataset with a template",
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "string",
                        "description": "Dataset ID to transform",
                    },
                    "output_name": {
                        "type": "string",
                        "description": "Name for the transformed dataset",
                    },
                    "template": {
                        "type": "string",
                        "description": "Python format string template, e.g. '### Instruction:\\n{instruction}\\n### Response:\\n{output}'",
                    },
                },
                "required": ["dataset_id", "output_name", "template"],
            },
        ),
        # Experiments
        Tool(
            name="experiment_create",
            description="Create a new experiment",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Experiment name",
                    },
                    "base_model": {
                        "type": "string",
                        "description": "Base model to fine-tune (e.g. meta-llama/Llama-3.1-8B)",
                    },
                    "method": {
                        "type": "string",
                        "description": "Training method",
                        "enum": ["full", "lora", "qlora", "sft", "dpo"],
                        "default": "qlora",
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional description",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional tags for organization",
                    },
                },
                "required": ["name", "base_model"],
            },
        ),
        Tool(
            name="experiment_list",
            description="List all experiments",
            inputSchema={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Filter by status",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number to return (default 20)",
                        "default": 20,
                    },
                },
            },
        ),
        Tool(
            name="experiment_get",
            description="Get details of an experiment",
            inputSchema={
                "type": "object",
                "properties": {
                    "experiment_id": {
                        "type": "string",
                        "description": "Experiment ID",
                    },
                },
                "required": ["experiment_id"],
            },
        ),
        Tool(
            name="experiment_compare",
            description="Compare multiple experiments",
            inputSchema={
                "type": "object",
                "properties": {
                    "experiment_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of experiment IDs to compare",
                    },
                },
                "required": ["experiment_ids"],
            },
        ),
        Tool(
            name="experiment_fork",
            description="Fork an experiment with optional config changes",
            inputSchema={
                "type": "object",
                "properties": {
                    "experiment_id": {
                        "type": "string",
                        "description": "Experiment ID to fork",
                    },
                    "new_name": {
                        "type": "string",
                        "description": "Name for the new experiment",
                    },
                },
                "required": ["experiment_id"],
            },
        ),
        # Training
        Tool(
            name="train_estimate",
            description="Estimate resources and cost for training",
            inputSchema={
                "type": "object",
                "properties": {
                    "base_model": {
                        "type": "string",
                        "description": "Model to fine-tune",
                    },
                    "dataset_id": {
                        "type": "string",
                        "description": "Dataset ID to use",
                    },
                    "method": {
                        "type": "string",
                        "description": "Training method (qlora, lora, full)",
                        "default": "qlora",
                    },
                    "epochs": {
                        "type": "integer",
                        "description": "Number of epochs",
                        "default": 3,
                    },
                },
                "required": ["base_model", "dataset_id"],
            },
        ),
        Tool(
            name="train_launch",
            description="Launch a training run",
            inputSchema={
                "type": "object",
                "properties": {
                    "experiment_id": {
                        "type": "string",
                        "description": "Experiment to run training for",
                    },
                    "dataset_id": {
                        "type": "string",
                        "description": "Dataset ID to use",
                    },
                    "backend": {
                        "type": "string",
                        "description": "Training backend (local, mistral, together, openai)",
                        "default": "local",
                    },
                    "config": {
                        "type": "object",
                        "description": "Training configuration overrides",
                    },
                },
                "required": ["experiment_id", "dataset_id"],
            },
        ),
        Tool(
            name="train_status",
            description="Get status of a training run",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "Training run ID",
                    },
                },
                "required": ["run_id"],
            },
        ),
        Tool(
            name="train_stop",
            description="Stop a training run",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {
                        "type": "string",
                        "description": "Training run ID",
                    },
                },
                "required": ["run_id"],
            },
        ),
        # Cloud/Infrastructure
        Tool(
            name="infra_list_gpus",
            description="List available GPUs across all providers with pricing",
            inputSchema={
                "type": "object",
                "properties": {
                    "gpu_type": {
                        "type": "string",
                        "description": "Filter by GPU type (a100, h100, etc.)",
                    },
                    "provider": {
                        "type": "string",
                        "description": "Filter by provider",
                    },
                },
            },
        ),
        Tool(
            name="infra_provision",
            description="Provision a cloud GPU instance",
            inputSchema={
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "description": "Cloud provider (lambda_labs, runpod)",
                    },
                    "gpu_type": {
                        "type": "string",
                        "description": "GPU type to provision",
                    },
                    "gpu_count": {
                        "type": "integer",
                        "description": "Number of GPUs",
                        "default": 1,
                    },
                },
                "required": ["provider", "gpu_type"],
            },
        ),
        Tool(
            name="infra_terminate",
            description="Terminate a cloud instance",
            inputSchema={
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "description": "Cloud provider",
                    },
                    "instance_id": {
                        "type": "string",
                        "description": "Instance ID to terminate",
                    },
                },
                "required": ["provider", "instance_id"],
            },
        ),
        # Remote VPS
        Tool(
            name="vps_register",
            description="Register a remote VPS for training (any SSH-accessible machine)",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Friendly name for the VPS (e.g. 'hetzner-01', 'home-server')",
                    },
                    "host": {
                        "type": "string",
                        "description": "Hostname or IP address",
                    },
                    "user": {
                        "type": "string",
                        "description": "SSH username",
                    },
                    "port": {
                        "type": "integer",
                        "description": "SSH port (default 22)",
                        "default": 22,
                    },
                    "ssh_key_path": {
                        "type": "string",
                        "description": "Path to SSH private key (optional if using default)",
                    },
                    "gpu_type": {
                        "type": "string",
                        "description": "GPU type (e.g. 'rtx_4090', 'a100')",
                    },
                    "gpu_count": {
                        "type": "integer",
                        "description": "Number of GPUs",
                        "default": 1,
                    },
                    "monthly_cost_usd": {
                        "type": "number",
                        "description": "Monthly cost for amortized hourly rate calculation",
                    },
                    "tailscale_only": {
                        "type": "boolean",
                        "description": "Require Tailscale VPN connection before accessing",
                        "default": False,
                    },
                    "tailscale_hostname": {
                        "type": "string",
                        "description": "Tailscale hostname (e.g. 'gpu-box.tail1234.ts.net')",
                    },
                },
                "required": ["name", "host", "user"],
            },
        ),
        Tool(
            name="vps_list",
            description="List all registered VPS machines",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="vps_status",
            description="Check status of a VPS (online, GPU usage, running jobs)",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "VPS name",
                    },
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="vps_unregister",
            description="Remove a VPS from the registry",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "VPS name to remove",
                    },
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="vps_setup",
            description="Set up training environment on a VPS (installs dependencies)",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "VPS name",
                    },
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="vps_sync",
            description="Sync a dataset to a VPS",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "VPS name",
                    },
                    "dataset_id": {
                        "type": "string",
                        "description": "Dataset ID to sync",
                    },
                },
                "required": ["name", "dataset_id"],
            },
        ),
        Tool(
            name="vps_run",
            description="Run a command on a VPS",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "VPS name",
                    },
                    "command": {
                        "type": "string",
                        "description": "Command to run",
                    },
                },
                "required": ["name", "command"],
            },
        ),
        Tool(
            name="vps_logs",
            description="Get training logs from a VPS run",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "VPS name",
                    },
                    "run_id": {
                        "type": "string",
                        "description": "Training run ID",
                    },
                    "tail_lines": {
                        "type": "integer",
                        "description": "Number of lines to show (default 100)",
                        "default": 100,
                    },
                },
                "required": ["name", "run_id"],
            },
        ),
        # Ollama
        Tool(
            name="ollama_status",
            description="Check Ollama service status (running, version, GPU)",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="ollama_list",
            description="List all models in Ollama",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="ollama_pull",
            description="Pull a model from the Ollama registry",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Model name (e.g. 'llama3:8b', 'mistral:latest')",
                    },
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="ollama_deploy",
            description="Deploy a GGUF model to Ollama",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name for the model in Ollama",
                    },
                    "gguf_path": {
                        "type": "string",
                        "description": "Path to the GGUF file",
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": "Optional system prompt to bake in",
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Default temperature",
                    },
                },
                "required": ["name", "gguf_path"],
            },
        ),
        Tool(
            name="ollama_chat",
            description="Send a chat message to an Ollama model",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Model name",
                    },
                    "message": {
                        "type": "string",
                        "description": "User message",
                    },
                    "system": {
                        "type": "string",
                        "description": "Optional system prompt",
                    },
                },
                "required": ["model", "message"],
            },
        ),
        Tool(
            name="ollama_delete",
            description="Delete a model from Ollama",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Model name to delete",
                    },
                },
                "required": ["name"],
            },
        ),
        # Open WebUI
        Tool(
            name="owui_status",
            description="Check Open WebUI connection status",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="owui_list_models",
            description="List model configurations in Open WebUI",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="owui_create_model",
            description="Create a model configuration in Open WebUI",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Display name for the model",
                    },
                    "base_model": {
                        "type": "string",
                        "description": "Base Ollama model (e.g. 'llama3:latest')",
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": "System prompt",
                    },
                    "description": {
                        "type": "string",
                        "description": "Model description",
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Temperature setting",
                    },
                },
                "required": ["name", "base_model"],
            },
        ),
        Tool(
            name="owui_delete_model",
            description="Delete a model configuration from Open WebUI",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "Model ID to delete",
                    },
                },
                "required": ["model_id"],
            },
        ),
        Tool(
            name="owui_list_knowledge",
            description="List knowledge bases in Open WebUI",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="owui_create_knowledge",
            description="Create a knowledge base in Open WebUI",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Knowledge base name",
                    },
                    "description": {
                        "type": "string",
                        "description": "Description",
                    },
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="owui_add_knowledge_file",
            description="Add a file to a knowledge base (PDF, TXT, MD, etc.)",
            inputSchema={
                "type": "object",
                "properties": {
                    "knowledge_id": {
                        "type": "string",
                        "description": "Knowledge base ID",
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Path to file to add",
                    },
                },
                "required": ["knowledge_id", "file_path"],
            },
        ),
        Tool(
            name="owui_chat",
            description="Chat through Open WebUI (uses model config + knowledge)",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Model ID or name",
                    },
                    "message": {
                        "type": "string",
                        "description": "User message",
                    },
                },
                "required": ["model", "message"],
            },
        ),
        # Security
        Tool(
            name="security_audit_log",
            description="View recent audit log entries",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Max entries to return (default 50)",
                        "default": 50,
                    },
                    "category": {
                        "type": "string",
                        "description": "Filter by category (credential, vps, training, cloud, inference, security)",
                    },
                    "failures_only": {
                        "type": "boolean",
                        "description": "Only show failed operations",
                        "default": False,
                    },
                },
            },
        ),
        Tool(
            name="security_audit_summary",
            description="Get a summary of audit activity",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="security_tailscale_status",
            description="Check Tailscale VPN connection status",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="security_ssh_key_rotate",
            description="Rotate SSH key for a VPS",
            inputSchema={
                "type": "object",
                "properties": {
                    "vps_name": {
                        "type": "string",
                        "description": "VPS name to rotate key for",
                    },
                    "key_type": {
                        "type": "string",
                        "description": "Key type (ed25519 or rsa)",
                        "default": "ed25519",
                    },
                },
                "required": ["vps_name"],
            },
        ),
        Tool(
            name="creds_expiry_check",
            description="Check credential expiry status (expired, expiring soon, healthy)",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="creds_rotate",
            description="Rotate credentials for a provider",
            inputSchema={
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "description": "Provider to rotate",
                        "enum": [p.value for p in ProviderType],
                    },
                    "new_api_key": {
                        "type": "string",
                        "description": "New API key",
                    },
                    "new_api_secret": {
                        "type": "string",
                        "description": "New API secret (if applicable)",
                    },
                },
                "required": ["provider", "new_api_key"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    try:
        result = await _dispatch_tool(name, arguments)
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e!s}")]


async def _dispatch_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a tool call to the appropriate handler."""

    # Credentials
    if name == "creds_create_vault":
        vault = get_vault()
        vault.create(args["password"])
        return {"status": "success", "message": "Vault created and unlocked"}

    elif name == "creds_unlock":
        vault = get_vault()
        success = vault.unlock(args["password"])
        if success:
            return {"status": "success", "message": "Vault unlocked"}
        else:
            return {"status": "error", "message": "Invalid password or vault not found"}

    elif name == "creds_add":
        vault = get_vault()
        if not vault.is_unlocked:
            return {"status": "error", "message": "Vault is locked"}

        cred = ProviderCredential(
            provider=ProviderType(args["provider"]),
            api_key=args["api_key"],
            api_secret=args.get("api_secret"),
        )
        vault.add(cred)
        return {"status": "success", "message": f"Credentials added for {args['provider']}"}

    elif name == "creds_list":
        vault = get_vault()
        if not vault.is_unlocked:
            return {"status": "error", "message": "Vault is locked"}

        providers = vault.list_providers()
        return {
            "status": "success",
            "providers": [p.value for p in providers],
        }

    elif name == "creds_test":
        vault = get_vault()
        if not vault.is_unlocked:
            return {"status": "error", "message": "Vault is locked"}

        provider = args["provider"]
        if provider in ["lambda_labs", "runpod"]:
            if provider == "lambda_labs":
                client = LambdaLabsProvider()
            else:
                client = RunPodProvider()
            valid = await client.check_credentials()
            return {"status": "success" if valid else "error", "valid": valid}

        return {"status": "error", "message": f"Testing not implemented for {provider}"}

    # Datasets
    elif name == "dataset_register":
        manager = get_dataset_manager()
        info = await manager.register(args["path"], args.get("name"))
        return {
            "status": "success",
            "dataset_id": info.id,
            "name": info.name,
            "num_samples": info.num_samples,
            "schema": info.schema,
        }

    elif name == "dataset_list":
        manager = get_dataset_manager()
        datasets = manager.list()
        return {
            "status": "success",
            "datasets": [
                {
                    "id": d.id,
                    "name": d.name,
                    "num_samples": d.num_samples,
                    "format": d.format,
                }
                for d in datasets
            ],
        }

    elif name == "dataset_inspect":
        manager = get_dataset_manager()
        info = manager.get(args["dataset_id"])
        if not info:
            return {"status": "error", "message": "Dataset not found"}

        return {
            "status": "success",
            "id": info.id,
            "name": info.name,
            "path": info.path,
            "format": info.format,
            "num_samples": info.num_samples,
            "size_bytes": info.size_bytes,
            "schema": info.schema,
            "statistics": info.statistics,
        }

    elif name == "dataset_preview":
        manager = get_dataset_manager()
        samples = await manager.preview(
            args["dataset_id"],
            args.get("num_samples", 5),
        )
        return {
            "status": "success",
            "samples": [
                {"index": s.index, "text": s.text, "data": s.data}
                for s in samples
            ],
        }

    elif name == "dataset_split":
        manager = get_dataset_manager()
        splits = await manager.split(
            args["dataset_id"],
            args.get("train_ratio", 0.9),
            args.get("val_ratio", 0.1),
        )
        return {"status": "success", "splits": splits}

    elif name == "dataset_transform":
        manager = get_dataset_manager()
        info = await manager.transform(
            args["dataset_id"],
            args["output_name"],
            args.get("template"),
        )
        return {
            "status": "success",
            "dataset_id": info.id,
            "name": info.name,
            "num_samples": info.num_samples,
        }

    # Experiments
    elif name == "experiment_create":
        store = get_experiment_store()
        exp = await store.create_experiment(
            name=args["name"],
            base_model=args["base_model"],
            method=args.get("method", "qlora"),
            description=args.get("description", ""),
            tags=args.get("tags"),
        )
        return {
            "status": "success",
            "experiment_id": exp.id,
            "name": exp.name,
        }

    elif name == "experiment_list":
        store = get_experiment_store()
        experiments = await store.list_experiments(
            limit=args.get("limit", 20),
            status=args.get("status"),
        )
        return {
            "status": "success",
            "experiments": [
                {
                    "id": e.id,
                    "name": e.name,
                    "base_model": e.base_model,
                    "method": e.method,
                    "status": e.status,
                    "created_at": e.created_at.isoformat(),
                }
                for e in experiments
            ],
        }

    elif name == "experiment_get":
        store = get_experiment_store()
        exp = await store.get_experiment(args["experiment_id"])
        if not exp:
            return {"status": "error", "message": "Experiment not found"}

        return {
            "status": "success",
            "experiment": {
                "id": exp.id,
                "name": exp.name,
                "base_model": exp.base_model,
                "method": exp.method,
                "status": exp.status,
                "description": exp.description,
                "tags": exp.tags,
                "config": exp.config,
                "metrics": exp.metrics,
                "best_checkpoint": exp.best_checkpoint,
                "created_at": exp.created_at.isoformat(),
                "updated_at": exp.updated_at.isoformat(),
            },
        }

    elif name == "experiment_compare":
        store = get_experiment_store()
        comparison = await store.compare_experiments(args["experiment_ids"])
        return {"status": "success", "comparison": comparison}

    elif name == "experiment_fork":
        store = get_experiment_store()
        exp = await store.fork_experiment(
            args["experiment_id"],
            args.get("new_name"),
        )
        return {
            "status": "success",
            "experiment_id": exp.id,
            "name": exp.name,
        }

    # Training
    elif name == "train_estimate":
        # Get dataset info
        manager = get_dataset_manager()
        dataset = manager.get(args["dataset_id"])
        if not dataset:
            return {"status": "error", "message": "Dataset not found"}

        method = TrainingMethod(args.get("method", "qlora"))
        config = TrainingConfig(
            base_model=args["base_model"],
            method=method,
            epochs=args.get("epochs", 3),
        )

        # Get estimates from multiple backends
        estimates = {}

        # Local estimate
        local_backend = LocalBackend()
        estimates["local"] = await local_backend.estimate_resources(
            config, dataset.num_samples
        )

        # API estimates if applicable
        if "mistral" in args["base_model"].lower():
            mistral_backend = MistralAPIBackend()
            try:
                estimates["mistral_api"] = await mistral_backend.estimate_resources(
                    config, dataset.num_samples
                )
            except Exception:
                pass

        together_backend = TogetherAPIBackend()
        try:
            estimates["together_api"] = await together_backend.estimate_resources(
                config, dataset.num_samples
            )
        except Exception:
            pass

        return {"status": "success", "estimates": estimates}

    elif name == "train_launch":
        store = get_experiment_store()
        manager = get_dataset_manager()

        exp = await store.get_experiment(args["experiment_id"])
        if not exp:
            return {"status": "error", "message": "Experiment not found"}

        dataset = manager.get(args["dataset_id"])
        if not dataset:
            return {"status": "error", "message": "Dataset not found"}

        backend_name = args.get("backend", "local")
        if backend_name not in TRAINING_BACKENDS:
            return {"status": "error", "message": f"Unknown backend: {backend_name}"}

        backend = TRAINING_BACKENDS[backend_name]()

        # Build config
        config_overrides = args.get("config", {})
        config = TrainingConfig(
            base_model=exp.base_model,
            method=TrainingMethod(exp.method),
            **config_overrides,
        )

        # Create run
        run_record = await store.create_run(exp.id, config)
        run = TrainingRun(
            run_id=run_record.id,
            experiment_id=exp.id,
            config=config,
        )

        # Launch
        output_dir = str(Path.home() / ".cache" / "ml-lab" / "outputs" / run.run_id)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        run_id = await backend.launch(run, dataset.path, output_dir)

        await store.update_experiment(exp.id, status="running")

        return {
            "status": "success",
            "run_id": run_id,
            "output_dir": output_dir,
            "backend": backend_name,
        }

    elif name == "train_status":
        # Try to find the backend that has this run
        for backend_name, backend_cls in TRAINING_BACKENDS.items():
            backend = backend_cls()
            try:
                run = await backend.get_status(args["run_id"])
                return {
                    "status": "success",
                    "run": {
                        "run_id": run.run_id,
                        "status": run.status.value,
                        "current_step": run.current_step,
                        "total_steps": run.total_steps,
                        "best_loss": run.best_loss,
                        "error_message": run.error_message,
                    },
                }
            except ValueError:
                continue

        return {"status": "error", "message": "Run not found"}

    elif name == "train_stop":
        for backend_name, backend_cls in TRAINING_BACKENDS.items():
            backend = backend_cls()
            try:
                await backend.stop(args["run_id"])
                return {"status": "success", "message": "Training stopped"}
            except ValueError:
                continue

        return {"status": "error", "message": "Run not found"}

    # Infrastructure
    elif name == "infra_list_gpus":
        all_quotes = []

        vault = get_vault()
        if not vault.is_unlocked:
            return {"status": "error", "message": "Vault is locked - unlock to query providers"}

        for provider_name, provider_cls in CLOUD_PROVIDERS.items():
            if args.get("provider") and args["provider"] != provider_name:
                continue

            try:
                provider = provider_cls()
                if not await provider.check_credentials():
                    continue

                gpu_filter = None
                if args.get("gpu_type"):
                    gpu_filter = GPUType(args["gpu_type"].lower())

                quotes = await provider.list_available(gpu_type=gpu_filter)
                for q in quotes:
                    all_quotes.append({
                        "provider": q.provider,
                        "gpu_type": q.gpu_type.value,
                        "gpu_count": q.gpu_count,
                        "hourly_price_usd": q.hourly_price_usd,
                        "availability": q.availability,
                        "spot": q.spot,
                        "region": q.region,
                    })
            except Exception as e:
                continue

        # Sort by price
        all_quotes.sort(key=lambda x: x["hourly_price_usd"])

        return {"status": "success", "gpus": all_quotes}

    elif name == "infra_provision":
        vault = get_vault()
        if not vault.is_unlocked:
            return {"status": "error", "message": "Vault is locked"}

        provider_name = args["provider"]
        if provider_name not in CLOUD_PROVIDERS:
            return {"status": "error", "message": f"Unknown provider: {provider_name}"}

        provider = CLOUD_PROVIDERS[provider_name]()
        gpu_type = GPUType(args["gpu_type"].lower())

        instance = await provider.provision(
            gpu_type=gpu_type,
            gpu_count=args.get("gpu_count", 1),
        )

        return {
            "status": "success",
            "instance": {
                "instance_id": instance.instance_id,
                "provider": instance.provider,
                "gpu_type": instance.gpu_type.value,
                "gpu_count": instance.gpu_count,
                "hourly_price_usd": instance.hourly_price_usd,
                "status": instance.status.value,
            },
        }

    elif name == "infra_terminate":
        vault = get_vault()
        if not vault.is_unlocked:
            return {"status": "error", "message": "Vault is locked"}

        provider_name = args["provider"]
        if provider_name not in CLOUD_PROVIDERS:
            return {"status": "error", "message": f"Unknown provider: {provider_name}"}

        provider = CLOUD_PROVIDERS[provider_name]()
        await provider.terminate(args["instance_id"])

        return {"status": "success", "message": "Instance terminated"}

    # VPS Management
    elif name == "vps_register":
        vps_manager = get_vps_manager()
        audit = get_audit_log()
        config = vps_manager.register(
            name=args["name"],
            host=args["host"],
            user=args["user"],
            port=args.get("port", 22),
            ssh_key_path=args.get("ssh_key_path"),
            gpu_type=args.get("gpu_type"),
            gpu_count=args.get("gpu_count", 1),
            monthly_cost_usd=args.get("monthly_cost_usd"),
            tailscale_only=args.get("tailscale_only", False),
            tailscale_hostname=args.get("tailscale_hostname"),
        )
        hourly = vps_manager.get_hourly_cost(args["name"])
        audit.log(
            AuditCategory.VPS,
            AuditAction.VPS_REGISTER,
            target=args["name"],
            details={"host": args["host"], "tailscale_only": args.get("tailscale_only", False)},
        )
        return {
            "status": "success",
            "vps": {
                "name": config.name,
                "host": config.host,
                "user": config.user,
                "gpu_type": config.gpu_type,
                "gpu_count": config.gpu_count,
                "hourly_cost_usd": hourly,
                "tailscale_only": config.tailscale_only,
                "tailscale_hostname": config.tailscale_hostname,
            },
        }

    elif name == "vps_list":
        vps_manager = get_vps_manager()
        hosts = vps_manager.list()
        return {
            "status": "success",
            "vps_hosts": [
                {
                    "name": h.name,
                    "host": h.host,
                    "user": h.user,
                    "gpu_type": h.gpu_type,
                    "gpu_count": h.gpu_count,
                    "hourly_cost_usd": vps_manager.get_hourly_cost(h.name),
                }
                for h in hosts
            ],
        }

    elif name == "vps_status":
        vps_manager = get_vps_manager()
        status = await vps_manager.check_status(args["name"])
        return {
            "status": "success" if status.online else "error",
            "vps_status": {
                "name": status.name,
                "online": status.online,
                "gpu_available": status.gpu_available,
                "gpu_memory_used_mb": status.gpu_memory_used_mb,
                "gpu_memory_total_mb": status.gpu_memory_total_mb,
                "gpu_utilization_pct": status.gpu_utilization_pct,
                "cpu_load": status.cpu_load,
                "disk_free_gb": status.disk_free_gb,
                "running_jobs": status.running_jobs,
                "error": status.error,
            },
        }

    elif name == "vps_unregister":
        vps_manager = get_vps_manager()
        removed = vps_manager.unregister(args["name"])
        if removed:
            return {"status": "success", "message": f"VPS '{args['name']}' removed"}
        else:
            return {"status": "error", "message": f"VPS '{args['name']}' not found"}

    elif name == "vps_setup":
        vps_manager = get_vps_manager()
        success, output = await vps_manager.setup_environment(args["name"])
        return {
            "status": "success" if success else "error",
            "output": output,
        }

    elif name == "vps_sync":
        vps_manager = get_vps_manager()
        manager = get_dataset_manager()

        dataset = manager.get(args["dataset_id"])
        if not dataset:
            return {"status": "error", "message": "Dataset not found"}

        remote_path = await vps_manager.sync_to_vps(args["name"], dataset.path)
        return {
            "status": "success",
            "local_path": dataset.path,
            "remote_path": remote_path,
        }

    elif name == "vps_run":
        vps_manager = get_vps_manager()
        returncode, stdout, stderr = await vps_manager.run_command(
            args["name"],
            args["command"],
        )
        return {
            "status": "success" if returncode == 0 else "error",
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
        }

    elif name == "vps_logs":
        vps_manager = get_vps_manager()
        logs = await vps_manager.get_training_logs(
            args["name"],
            args["run_id"],
            args.get("tail_lines", 100),
        )
        return {
            "status": "success",
            "logs": logs,
        }

    # Ollama
    elif name == "ollama_status":
        client = get_ollama_client()
        status = await client.status()
        return {
            "status": "success" if status.running else "error",
            "ollama": {
                "running": status.running,
                "version": status.version,
                "models_count": status.models_count,
                "gpu_available": status.gpu_available,
                "gpu_name": status.gpu_name,
                "error": status.error,
            },
        }

    elif name == "ollama_list":
        client = get_ollama_client()
        models = await client.list_models()
        return {
            "status": "success",
            "models": [
                {
                    "name": m.name,
                    "size_gb": round(m.size / 1e9, 2),
                    "modified_at": m.modified_at,
                }
                for m in models
            ],
        }

    elif name == "ollama_pull":
        client = get_ollama_client()
        progress_messages = []
        async for progress in client.pull_model(args["name"]):
            status = progress.get("status", "")
            if "pulling" in status or "downloading" in status:
                pct = progress.get("completed", 0) / max(progress.get("total", 1), 1) * 100
                progress_messages.append(f"{status}: {pct:.1f}%")
            elif status:
                progress_messages.append(status)
        return {
            "status": "success",
            "message": f"Model {args['name']} pulled successfully",
            "progress": progress_messages[-5:] if progress_messages else [],
        }

    elif name == "ollama_deploy":
        client = get_ollama_client()
        params = {}
        if args.get("temperature"):
            params["temperature"] = args["temperature"]

        success = await client.deploy_gguf(
            name=args["name"],
            gguf_path=args["gguf_path"],
            system_prompt=args.get("system_prompt"),
            parameters=params if params else None,
        )
        return {
            "status": "success" if success else "error",
            "message": f"Model deployed as '{args['name']}'" if success else "Deployment failed",
        }

    elif name == "ollama_chat":
        client = get_ollama_client()
        messages = []
        if args.get("system"):
            messages.append({"role": "system", "content": args["system"]})
        messages.append({"role": "user", "content": args["message"]})

        response = await client.chat(args["model"], messages)
        return {
            "status": "success",
            "response": response.message.content,
            "model": response.model,
            "eval_count": response.eval_count,
        }

    elif name == "ollama_delete":
        client = get_ollama_client()
        success = await client.delete_model(args["name"])
        return {
            "status": "success" if success else "error",
            "message": f"Model '{args['name']}' deleted" if success else "Delete failed",
        }

    # Open WebUI
    elif name == "owui_status":
        client = get_openwebui_client()
        status = await client.status()
        return {
            "status": "success" if status.connected else "error",
            "openwebui": {
                "connected": status.connected,
                "version": status.version,
                "models_count": status.models_count,
                "knowledge_count": status.knowledge_count,
                "error": status.error,
            },
        }

    elif name == "owui_list_models":
        client = get_openwebui_client()
        models = await client.list_models()
        return {
            "status": "success",
            "models": [
                {
                    "id": m.id,
                    "name": m.name,
                    "base_model": m.base_model_id,
                    "has_system_prompt": bool(m.params.get("system")),
                }
                for m in models
            ],
        }

    elif name == "owui_create_model":
        client = get_openwebui_client()
        params = {}
        if args.get("temperature"):
            params["temperature"] = args["temperature"]

        model = await client.create_model(
            name=args["name"],
            base_model_id=args["base_model"],
            system_prompt=args.get("system_prompt"),
            description=args.get("description"),
            params=params if params else None,
        )
        return {
            "status": "success",
            "model": {
                "id": model.id,
                "name": model.name,
                "base_model": model.base_model_id,
            },
        }

    elif name == "owui_delete_model":
        client = get_openwebui_client()
        success = await client.delete_model(args["model_id"])
        return {
            "status": "success" if success else "error",
            "message": f"Model '{args['model_id']}' deleted" if success else "Delete failed",
        }

    elif name == "owui_list_knowledge":
        client = get_openwebui_client()
        knowledge = await client.list_knowledge()
        return {
            "status": "success",
            "knowledge_bases": [
                {
                    "id": k.id,
                    "name": k.name,
                    "description": k.description,
                    "files_count": len(k.files),
                }
                for k in knowledge
            ],
        }

    elif name == "owui_create_knowledge":
        client = get_openwebui_client()
        kb = await client.create_knowledge(
            name=args["name"],
            description=args.get("description", ""),
        )
        return {
            "status": "success",
            "knowledge": {
                "id": kb.id,
                "name": kb.name,
            },
        }

    elif name == "owui_add_knowledge_file":
        client = get_openwebui_client()
        success = await client.add_file_to_knowledge(
            args["knowledge_id"],
            args["file_path"],
        )
        return {
            "status": "success" if success else "error",
            "message": "File added to knowledge base" if success else "Failed to add file",
        }

    elif name == "owui_chat":
        client = get_openwebui_client()
        response = await client.chat(
            model=args["model"],
            messages=[{"role": "user", "content": args["message"]}],
        )
        # Extract assistant message from response
        choices = response.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "")
        else:
            content = str(response)

        return {
            "status": "success",
            "response": content,
        }

    # Security
    elif name == "security_audit_log":
        audit = get_audit_log()
        category = None
        if args.get("category"):
            try:
                category = AuditCategory(args["category"])
            except ValueError:
                pass

        events = audit.get_recent(
            limit=args.get("limit", 50),
            category=category,
            failures_only=args.get("failures_only", False),
        )

        return {
            "status": "success",
            "events": [
                {
                    "timestamp": e.timestamp,
                    "category": e.category,
                    "action": e.action,
                    "target": e.target,
                    "success": e.success,
                    "error": e.error,
                    "user": e.user,
                }
                for e in events
            ],
        }

    elif name == "security_audit_summary":
        audit = get_audit_log()
        summary = audit.get_summary()
        return {
            "status": "success",
            "summary": summary,
        }

    elif name == "security_tailscale_status":
        vps_manager = get_vps_manager()
        ts_status = await vps_manager.get_tailscale_status()
        audit = get_audit_log()
        audit.log(
            AuditCategory.SECURITY,
            AuditAction.TAILSCALE_CHECK,
            success=ts_status["connected"],
        )
        return {
            "status": "success" if ts_status["connected"] else "warning",
            "tailscale": {
                "connected": ts_status["connected"],
                "self_ip": ts_status["self_ip"],
            },
        }

    elif name == "security_ssh_key_rotate":
        vps_manager = get_vps_manager()
        audit = get_audit_log()
        try:
            result = await vps_manager.rotate_ssh_key(
                name=args["vps_name"],
                key_type=args.get("key_type", "ed25519"),
            )
            audit.log(
                AuditCategory.SECURITY,
                AuditAction.SSH_KEY_ROTATE,
                target=args["vps_name"],
                success=True,
                details={"new_key_path": result["new_key_path"]},
            )
            return {
                "status": "success",
                "result": result,
            }
        except Exception as e:
            audit.log(
                AuditCategory.SECURITY,
                AuditAction.SSH_KEY_ROTATE,
                target=args["vps_name"],
                success=False,
                error=str(e),
            )
            return {"status": "error", "message": str(e)}

    elif name == "creds_expiry_check":
        vault = get_vault()
        if not vault.is_unlocked:
            return {"status": "error", "message": "Vault is locked"}

        expiry_status = vault.check_expiry_status()
        return {
            "status": "success",
            "expiry_status": expiry_status,
            "warnings": {
                "expired": f"{len(expiry_status['expired'])} credentials have expired" if expiry_status["expired"] else None,
                "expiring_soon": f"{len(expiry_status['expiring_7_days'])} credentials expire within 7 days" if expiry_status["expiring_7_days"] else None,
            },
        }

    elif name == "creds_rotate":
        vault = get_vault()
        audit = get_audit_log()
        if not vault.is_unlocked:
            return {"status": "error", "message": "Vault is locked"}

        provider = ProviderType(args["provider"])
        result = vault.rotate_credential(
            provider=provider,
            new_api_key=args["new_api_key"],
            new_api_secret=args.get("new_api_secret"),
        )

        if result:
            audit.log(
                AuditCategory.CREDENTIAL,
                AuditAction.CRED_ADD,
                target=args["provider"],
                details={"rotated": True},
            )
            return {
                "status": "success",
                "message": f"Credentials rotated for {args['provider']}",
                "expires_at": result.expires_at,
                "last_rotated": result.last_rotated,
            }
        else:
            return {"status": "error", "message": f"No existing credentials for {args['provider']}"}

    else:
        return {"status": "error", "message": f"Unknown tool: {name}"}


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
