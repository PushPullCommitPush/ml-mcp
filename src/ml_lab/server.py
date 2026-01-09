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
from .cloud.runpod import RunPodProvider
from .credentials import CredentialVault, ProviderCredential, ProviderType, get_vault
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

    else:
        return {"status": "error", "message": f"Unknown tool: {name}"}


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
