"""Command-line interface for ML Lab."""

from __future__ import annotations

import asyncio
import sys

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="ml-lab",
    help="ML Lab - Model training, fine-tuning, and experimentation toolkit",
)
console = Console()


@app.command()
def serve():
    """Start the MCP server."""
    from .server import main

    console.print("[green]Starting ML Lab MCP server...[/green]")
    asyncio.run(main())


@app.command()
def init():
    """Initialize ML Lab configuration."""
    from pathlib import Path

    from .credentials import get_vault

    config_dir = Path.home() / ".config" / "ml-lab"
    config_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path.home() / ".cache" / "ml-lab"
    cache_dir.mkdir(parents=True, exist_ok=True)

    console.print("[green]ML Lab initialized![/green]")
    console.print(f"Config directory: {config_dir}")
    console.print(f"Cache directory: {cache_dir}")

    vault = get_vault()
    if not vault.exists:
        console.print("\n[yellow]No credential vault found.[/yellow]")
        console.print("Create one with: ml-lab vault create")


@app.command("vault")
def vault_cmd(
    action: str = typer.Argument(..., help="Action: create, unlock, list, add"),
    provider: str | None = typer.Option(None, help="Provider name for add action"),
    api_key: str | None = typer.Option(None, help="API key for add action"),
):
    """Manage the credential vault."""
    from .credentials import ProviderCredential, ProviderType, get_vault

    vault = get_vault()

    if action == "create":
        if vault.exists:
            console.print("[yellow]Vault already exists. Use 'unlock' to access it.[/yellow]")
            return

        password = typer.prompt("Enter vault password", hide_input=True)
        confirm = typer.prompt("Confirm password", hide_input=True)

        if password != confirm:
            console.print("[red]Passwords do not match![/red]")
            raise typer.Exit(1)

        vault.create(password)
        console.print("[green]Vault created successfully![/green]")

    elif action == "unlock":
        if not vault.exists:
            console.print("[red]No vault found. Create one first.[/red]")
            raise typer.Exit(1)

        password = typer.prompt("Enter vault password", hide_input=True)
        if vault.unlock(password):
            console.print("[green]Vault unlocked![/green]")
        else:
            console.print("[red]Invalid password![/red]")
            raise typer.Exit(1)

    elif action == "list":
        if not vault.is_unlocked:
            console.print("[red]Vault is locked. Unlock it first.[/red]")
            raise typer.Exit(1)

        providers = vault.list_providers()
        if providers:
            console.print("Configured providers:")
            for p in providers:
                console.print(f"  - {p.value}")
        else:
            console.print("No providers configured.")

    elif action == "add":
        if not vault.is_unlocked:
            console.print("[red]Vault is locked. Unlock it first.[/red]")
            raise typer.Exit(1)

        if not provider:
            provider = typer.prompt("Provider name")
        if not api_key:
            api_key = typer.prompt("API key", hide_input=True)

        try:
            provider_type = ProviderType(provider)
        except ValueError:
            console.print(f"[red]Unknown provider: {provider}[/red]")
            console.print(f"Valid providers: {', '.join(p.value for p in ProviderType)}")
            raise typer.Exit(1)

        cred = ProviderCredential(provider=provider_type, api_key=api_key)
        vault.add(cred)
        console.print(f"[green]Credentials added for {provider}![/green]")

    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        raise typer.Exit(1)


@app.command("datasets")
def datasets_cmd(
    action: str = typer.Argument("list", help="Action: list, register, inspect"),
    path: str | None = typer.Option(None, help="Path for register action"),
    dataset_id: str | None = typer.Option(None, "--id", help="Dataset ID for inspect"),
):
    """Manage datasets."""
    from .storage.datasets import get_dataset_manager

    manager = get_dataset_manager()

    if action == "list":
        datasets = manager.list()
        if not datasets:
            console.print("No datasets registered.")
            return

        table = Table(title="Datasets")
        table.add_column("ID")
        table.add_column("Name")
        table.add_column("Samples")
        table.add_column("Format")

        for d in datasets:
            table.add_row(d.id, d.name, str(d.num_samples), d.format)

        console.print(table)

    elif action == "register":
        if not path:
            console.print("[red]Path required for register action[/red]")
            raise typer.Exit(1)

        info = asyncio.run(manager.register(path))
        console.print(f"[green]Dataset registered![/green]")
        console.print(f"  ID: {info.id}")
        console.print(f"  Name: {info.name}")
        console.print(f"  Samples: {info.num_samples}")

    elif action == "inspect":
        if not dataset_id:
            console.print("[red]Dataset ID required for inspect action[/red]")
            raise typer.Exit(1)

        info = manager.get(dataset_id)
        if not info:
            console.print(f"[red]Dataset {dataset_id} not found[/red]")
            raise typer.Exit(1)

        console.print(f"[bold]Dataset: {info.name}[/bold]")
        console.print(f"ID: {info.id}")
        console.print(f"Path: {info.path}")
        console.print(f"Format: {info.format}")
        console.print(f"Samples: {info.num_samples}")
        console.print(f"Size: {info.size_bytes / 1024:.1f} KB")
        console.print(f"\nSchema: {info.schema}")
        console.print(f"Statistics: {info.statistics}")


@app.command("experiments")
def experiments_cmd(
    action: str = typer.Argument("list", help="Action: list, create, show"),
    name: str | None = typer.Option(None, help="Experiment name for create"),
    model: str | None = typer.Option(None, help="Base model for create"),
    experiment_id: str | None = typer.Option(None, "--id", help="Experiment ID for show"),
):
    """Manage experiments."""
    from .storage.experiments import get_experiment_store

    store = get_experiment_store()

    if action == "list":
        experiments = asyncio.run(store.list_experiments())
        if not experiments:
            console.print("No experiments found.")
            return

        table = Table(title="Experiments")
        table.add_column("ID")
        table.add_column("Name")
        table.add_column("Model")
        table.add_column("Method")
        table.add_column("Status")

        for e in experiments:
            table.add_row(e.id, e.name, e.base_model, e.method, e.status)

        console.print(table)

    elif action == "create":
        if not name or not model:
            console.print("[red]Name and model required for create action[/red]")
            raise typer.Exit(1)

        exp = asyncio.run(
            store.create_experiment(name=name, base_model=model, method="qlora")
        )
        console.print(f"[green]Experiment created![/green]")
        console.print(f"  ID: {exp.id}")
        console.print(f"  Name: {exp.name}")

    elif action == "show":
        if not experiment_id:
            console.print("[red]Experiment ID required for show action[/red]")
            raise typer.Exit(1)

        exp = asyncio.run(store.get_experiment(experiment_id))
        if not exp:
            console.print(f"[red]Experiment {experiment_id} not found[/red]")
            raise typer.Exit(1)

        console.print(f"[bold]Experiment: {exp.name}[/bold]")
        console.print(f"ID: {exp.id}")
        console.print(f"Base Model: {exp.base_model}")
        console.print(f"Method: {exp.method}")
        console.print(f"Status: {exp.status}")
        console.print(f"Created: {exp.created_at}")
        if exp.description:
            console.print(f"Description: {exp.description}")
        if exp.tags:
            console.print(f"Tags: {', '.join(exp.tags)}")
        if exp.metrics:
            console.print(f"Metrics: {exp.metrics}")


def main():
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
