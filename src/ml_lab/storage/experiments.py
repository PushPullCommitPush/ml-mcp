"""Experiment tracking and management."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite

from ..backends.base import TrainingConfig, TrainingStatus


@dataclass
class Experiment:
    """An experiment tracking container."""

    id: str
    name: str
    base_model: str
    method: str
    created_at: datetime
    updated_at: datetime
    status: str = "created"
    description: str = ""
    tags: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    best_checkpoint: str | None = None
    parent_experiment_id: str | None = None


@dataclass
class ExperimentRun:
    """A single run within an experiment."""

    id: str
    experiment_id: str
    started_at: datetime
    ended_at: datetime | None
    status: str
    config: dict[str, Any]
    metrics: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    error_message: str | None = None


class ExperimentStore:
    """
    SQLite-based experiment storage.

    Stores experiments, runs, and metrics for tracking and comparison.
    """

    def __init__(self, db_path: Path | None = None):
        """
        Initialize the experiment store.

        Args:
            db_path: Path to the SQLite database. Defaults to
                    ~/.cache/ml-lab/experiments.db
        """
        if db_path is None:
            cache_dir = Path.home() / ".cache" / "ml-lab"
            cache_dir.mkdir(parents=True, exist_ok=True)
            db_path = cache_dir / "experiments.db"

        self.db_path = db_path
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Ensure the database schema is created."""
        if self._initialized:
            return

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    base_model TEXT NOT NULL,
                    method TEXT NOT NULL,
                    status TEXT NOT NULL,
                    description TEXT,
                    tags TEXT,
                    config TEXT,
                    metrics TEXT,
                    best_checkpoint TEXT,
                    parent_experiment_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    config TEXT,
                    metrics TEXT,
                    artifacts TEXT,
                    error_message TEXT,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                )
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS metrics_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    metrics TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(id)
                )
            """)

            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_experiments_name ON experiments(name)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment_id)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics_log(run_id)
            """)

            await db.commit()

        self._initialized = True

    async def create_experiment(
        self,
        name: str,
        base_model: str,
        method: str,
        description: str = "",
        tags: list[str] | None = None,
        config: dict[str, Any] | None = None,
        parent_experiment_id: str | None = None,
    ) -> Experiment:
        """Create a new experiment."""
        await self._ensure_initialized()

        now = datetime.utcnow()
        experiment = Experiment(
            id=str(uuid.uuid4())[:8],
            name=name,
            base_model=base_model,
            method=method,
            description=description,
            tags=tags or [],
            config=config or {},
            parent_experiment_id=parent_experiment_id,
            created_at=now,
            updated_at=now,
        )

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO experiments
                (id, name, base_model, method, status, description, tags, config,
                 metrics, best_checkpoint, parent_experiment_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment.id,
                    experiment.name,
                    experiment.base_model,
                    experiment.method,
                    experiment.status,
                    experiment.description,
                    json.dumps(experiment.tags),
                    json.dumps(experiment.config),
                    json.dumps(experiment.metrics),
                    experiment.best_checkpoint,
                    experiment.parent_experiment_id,
                    experiment.created_at.isoformat(),
                    experiment.updated_at.isoformat(),
                ),
            )
            await db.commit()

        return experiment

    async def get_experiment(self, experiment_id: str) -> Experiment | None:
        """Get an experiment by ID."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
            ) as cursor:
                row = await cursor.fetchone()

            if not row:
                return None

            return self._row_to_experiment(row)

    async def get_experiment_by_name(self, name: str) -> Experiment | None:
        """Get an experiment by name."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM experiments WHERE name = ? ORDER BY created_at DESC LIMIT 1",
                (name,),
            ) as cursor:
                row = await cursor.fetchone()

            if not row:
                return None

            return self._row_to_experiment(row)

    async def list_experiments(
        self,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
        tag: str | None = None,
    ) -> list[Experiment]:
        """List experiments with optional filtering."""
        await self._ensure_initialized()

        query = "SELECT * FROM experiments WHERE 1=1"
        params: list[Any] = []

        if status:
            query += " AND status = ?"
            params.append(status)

        if tag:
            query += " AND tags LIKE ?"
            params.append(f'%"{tag}"%')

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()

            return [self._row_to_experiment(row) for row in rows]

    async def update_experiment(
        self,
        experiment_id: str,
        status: str | None = None,
        metrics: dict[str, Any] | None = None,
        best_checkpoint: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Update an experiment's fields."""
        await self._ensure_initialized()

        updates = ["updated_at = ?"]
        params: list[Any] = [datetime.utcnow().isoformat()]

        if status:
            updates.append("status = ?")
            params.append(status)

        if metrics:
            updates.append("metrics = ?")
            params.append(json.dumps(metrics))

        if best_checkpoint:
            updates.append("best_checkpoint = ?")
            params.append(best_checkpoint)

        if tags is not None:
            updates.append("tags = ?")
            params.append(json.dumps(tags))

        params.append(experiment_id)

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                f"UPDATE experiments SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            await db.commit()

    async def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment and its runs."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            # Delete metrics
            await db.execute(
                """
                DELETE FROM metrics_log WHERE run_id IN
                (SELECT id FROM runs WHERE experiment_id = ?)
                """,
                (experiment_id,),
            )
            # Delete runs
            await db.execute(
                "DELETE FROM runs WHERE experiment_id = ?", (experiment_id,)
            )
            # Delete experiment
            cursor = await db.execute(
                "DELETE FROM experiments WHERE id = ?", (experiment_id,)
            )
            await db.commit()
            return cursor.rowcount > 0

    async def create_run(
        self,
        experiment_id: str,
        config: TrainingConfig,
    ) -> ExperimentRun:
        """Create a new run for an experiment."""
        await self._ensure_initialized()

        now = datetime.utcnow()
        run = ExperimentRun(
            id=str(uuid.uuid4())[:8],
            experiment_id=experiment_id,
            started_at=now,
            ended_at=None,
            status=TrainingStatus.PENDING.value,
            config=config.to_dict(),
        )

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO runs
                (id, experiment_id, status, config, metrics, artifacts, started_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.id,
                    run.experiment_id,
                    run.status,
                    json.dumps(run.config),
                    json.dumps([]),
                    json.dumps([]),
                    run.started_at.isoformat(),
                ),
            )
            await db.commit()

        return run

    async def log_metrics(
        self,
        run_id: str,
        step: int,
        metrics: dict[str, Any],
    ) -> None:
        """Log metrics for a training step."""
        await self._ensure_initialized()

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO metrics_log (run_id, step, timestamp, metrics)
                VALUES (?, ?, ?, ?)
                """,
                (run_id, step, datetime.utcnow().isoformat(), json.dumps(metrics)),
            )
            await db.commit()

    async def get_run_metrics(
        self,
        run_id: str,
        start_step: int = 0,
        end_step: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get metrics history for a run."""
        await self._ensure_initialized()

        query = "SELECT step, timestamp, metrics FROM metrics_log WHERE run_id = ? AND step >= ?"
        params: list[Any] = [run_id, start_step]

        if end_step is not None:
            query += " AND step <= ?"
            params.append(end_step)

        query += " ORDER BY step"

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()

            return [
                {"step": row[0], "timestamp": row[1], **json.loads(row[2])}
                for row in rows
            ]

    async def compare_experiments(
        self,
        experiment_ids: list[str],
    ) -> dict[str, Any]:
        """Compare metrics across experiments."""
        await self._ensure_initialized()

        results = {}
        for exp_id in experiment_ids:
            exp = await self.get_experiment(exp_id)
            if exp:
                results[exp_id] = {
                    "name": exp.name,
                    "model": exp.base_model,
                    "method": exp.method,
                    "status": exp.status,
                    "metrics": exp.metrics,
                    "config": exp.config,
                }

        return results

    async def fork_experiment(
        self,
        experiment_id: str,
        new_name: str | None = None,
        config_overrides: dict[str, Any] | None = None,
    ) -> Experiment:
        """Fork an experiment with optional config changes."""
        original = await self.get_experiment(experiment_id)
        if not original:
            raise ValueError(f"Experiment {experiment_id} not found")

        new_config = {**original.config, **(config_overrides or {})}

        return await self.create_experiment(
            name=new_name or f"{original.name}-fork",
            base_model=original.base_model,
            method=original.method,
            description=f"Forked from {original.name} ({experiment_id})",
            tags=original.tags,
            config=new_config,
            parent_experiment_id=experiment_id,
        )

    def _row_to_experiment(self, row: aiosqlite.Row) -> Experiment:
        """Convert a database row to an Experiment."""
        return Experiment(
            id=row["id"],
            name=row["name"],
            base_model=row["base_model"],
            method=row["method"],
            status=row["status"],
            description=row["description"] or "",
            tags=json.loads(row["tags"]) if row["tags"] else [],
            config=json.loads(row["config"]) if row["config"] else {},
            metrics=json.loads(row["metrics"]) if row["metrics"] else {},
            best_checkpoint=row["best_checkpoint"],
            parent_experiment_id=row["parent_experiment_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )


# Singleton store instance
_store: ExperimentStore | None = None


def get_experiment_store(db_path: Path | None = None) -> ExperimentStore:
    """Get or create the global experiment store."""
    global _store
    if _store is None:
        _store = ExperimentStore(db_path)
    return _store
