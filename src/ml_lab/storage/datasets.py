"""Dataset management and processing."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator


@dataclass
class DatasetInfo:
    """Information about a dataset."""

    id: str
    name: str
    path: str
    format: str  # jsonl, parquet, csv, huggingface
    size_bytes: int
    num_samples: int
    created_at: datetime
    schema: dict[str, Any] = field(default_factory=dict)
    statistics: dict[str, Any] = field(default_factory=dict)
    splits: dict[str, int] = field(default_factory=dict)  # split_name -> num_samples
    source: str | None = None  # Original source (HuggingFace, URL, etc.)
    version: str = "1.0.0"


@dataclass
class DatasetSample:
    """A single sample from a dataset."""

    index: int
    data: dict[str, Any]
    text: str | None = None  # Formatted text if applicable
    token_count: int | None = None


class DatasetManager:
    """
    Manages datasets for training.

    Handles loading, preprocessing, and versioning of datasets.
    """

    def __init__(self, data_dir: Path | None = None):
        """
        Initialize the dataset manager.

        Args:
            data_dir: Directory for storing datasets. Defaults to
                     ~/.cache/ml-lab/datasets
        """
        if data_dir is None:
            data_dir = Path.home() / ".cache" / "ml-lab" / "datasets"
        data_dir.mkdir(parents=True, exist_ok=True)

        self.data_dir = data_dir
        self._registry_path = data_dir / "registry.json"
        self._registry: dict[str, dict[str, Any]] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load the dataset registry from disk."""
        if self._registry_path.exists():
            with open(self._registry_path) as f:
                self._registry = json.load(f)

    def _save_registry(self) -> None:
        """Save the dataset registry to disk."""
        with open(self._registry_path, "w") as f:
            json.dump(self._registry, f, indent=2, default=str)

    def _compute_hash(self, path: Path) -> str:
        """Compute a hash of a file for versioning."""
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:12]

    async def register(
        self,
        path: str,
        name: str | None = None,
        format: str | None = None,
    ) -> DatasetInfo:
        """
        Register a dataset from a local file.

        Args:
            path: Path to the dataset file.
            name: Optional name for the dataset.
            format: Optional format override (jsonl, parquet, csv).

        Returns:
            DatasetInfo for the registered dataset.
        """
        src_path = Path(path)
        if not src_path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        # Detect format
        if format is None:
            suffix = src_path.suffix.lower()
            format_map = {
                ".jsonl": "jsonl",
                ".json": "jsonl",
                ".parquet": "parquet",
                ".csv": "csv",
                ".tsv": "csv",
            }
            format = format_map.get(suffix, "jsonl")

        # Generate ID and name
        file_hash = self._compute_hash(src_path)
        dataset_id = file_hash
        name = name or src_path.stem

        # Copy to managed directory
        dest_dir = self.data_dir / dataset_id
        dest_dir.mkdir(exist_ok=True)
        dest_path = dest_dir / src_path.name
        shutil.copy2(src_path, dest_path)

        # Analyze dataset
        num_samples, schema, stats = await self._analyze_dataset(dest_path, format)

        info = DatasetInfo(
            id=dataset_id,
            name=name,
            path=str(dest_path),
            format=format,
            size_bytes=dest_path.stat().st_size,
            num_samples=num_samples,
            created_at=datetime.utcnow(),
            schema=schema,
            statistics=stats,
        )

        # Save to registry
        self._registry[dataset_id] = {
            "id": info.id,
            "name": info.name,
            "path": info.path,
            "format": info.format,
            "size_bytes": info.size_bytes,
            "num_samples": info.num_samples,
            "created_at": info.created_at.isoformat(),
            "schema": info.schema,
            "statistics": info.statistics,
            "version": info.version,
        }
        self._save_registry()

        return info

    async def _analyze_dataset(
        self,
        path: Path,
        format: str,
    ) -> tuple[int, dict[str, Any], dict[str, Any]]:
        """Analyze a dataset to extract schema and statistics."""
        num_samples = 0
        schema: dict[str, Any] = {}
        stats: dict[str, Any] = {
            "text_lengths": [],
            "token_counts": [],
        }

        if format == "jsonl":
            with open(path) as f:
                for line in f:
                    if line.strip():
                        num_samples += 1
                        try:
                            data = json.loads(line)
                            # Infer schema from first sample
                            if not schema:
                                schema = {k: type(v).__name__ for k, v in data.items()}

                            # Collect text lengths
                            text_field = data.get("text") or data.get("content") or ""
                            if text_field:
                                stats["text_lengths"].append(len(text_field))

                        except json.JSONDecodeError:
                            continue

        elif format == "csv":
            import csv

            with open(path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    num_samples += 1
                    if not schema:
                        schema = {k: "str" for k in row.keys()}

        # Compute statistics
        if stats["text_lengths"]:
            lengths = stats["text_lengths"]
            stats = {
                "avg_text_length": sum(lengths) / len(lengths),
                "min_text_length": min(lengths),
                "max_text_length": max(lengths),
                "total_chars": sum(lengths),
            }
        else:
            stats = {}

        return num_samples, schema, stats

    def get(self, dataset_id: str) -> DatasetInfo | None:
        """Get dataset info by ID."""
        if dataset_id not in self._registry:
            return None

        data = self._registry[dataset_id]
        return DatasetInfo(
            id=data["id"],
            name=data["name"],
            path=data["path"],
            format=data["format"],
            size_bytes=data["size_bytes"],
            num_samples=data["num_samples"],
            created_at=datetime.fromisoformat(data["created_at"]),
            schema=data.get("schema", {}),
            statistics=data.get("statistics", {}),
            version=data.get("version", "1.0.0"),
        )

    def list(self) -> list[DatasetInfo]:
        """List all registered datasets."""
        return [
            self.get(dataset_id)
            for dataset_id in self._registry
            if self.get(dataset_id) is not None
        ]

    async def preview(
        self,
        dataset_id: str,
        num_samples: int = 5,
        offset: int = 0,
    ) -> list[DatasetSample]:
        """Preview samples from a dataset."""
        info = self.get(dataset_id)
        if not info:
            raise ValueError(f"Dataset {dataset_id} not found")

        samples = []
        path = Path(info.path)

        if info.format == "jsonl":
            with open(path) as f:
                for i, line in enumerate(f):
                    if i < offset:
                        continue
                    if len(samples) >= num_samples:
                        break

                    if line.strip():
                        try:
                            data = json.loads(line)
                            text = data.get("text") or data.get("content")
                            samples.append(
                                DatasetSample(
                                    index=i,
                                    data=data,
                                    text=text[:500] if text else None,
                                )
                            )
                        except json.JSONDecodeError:
                            continue

        return samples

    async def split(
        self,
        dataset_id: str,
        train_ratio: float = 0.9,
        val_ratio: float = 0.1,
        test_ratio: float = 0.0,
        seed: int = 42,
    ) -> dict[str, str]:
        """
        Split a dataset into train/val/test sets.

        Returns paths to the split files.
        """
        import random

        info = self.get(dataset_id)
        if not info:
            raise ValueError(f"Dataset {dataset_id} not found")

        # Validate ratios
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")

        # Read all samples
        samples = []
        with open(info.path) as f:
            for line in f:
                if line.strip():
                    samples.append(line)

        # Shuffle
        random.seed(seed)
        random.shuffle(samples)

        # Split
        n = len(samples)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        splits = {
            "train": samples[:train_end],
            "val": samples[train_end:val_end],
        }
        if test_ratio > 0:
            splits["test"] = samples[val_end:]

        # Write split files
        output_paths = {}
        base_dir = Path(info.path).parent

        for split_name, split_samples in splits.items():
            if not split_samples:
                continue

            split_path = base_dir / f"{Path(info.path).stem}_{split_name}.jsonl"
            with open(split_path, "w") as f:
                f.writelines(split_samples)
            output_paths[split_name] = str(split_path)

        return output_paths

    async def transform(
        self,
        dataset_id: str,
        output_name: str,
        template: str | None = None,
        field_mapping: dict[str, str] | None = None,
    ) -> DatasetInfo:
        """
        Transform a dataset with a template or field mapping.

        Args:
            dataset_id: Source dataset ID.
            output_name: Name for the transformed dataset.
            template: Python format string for creating "text" field.
                     Example: "### Instruction:\\n{instruction}\\n### Response:\\n{output}"
            field_mapping: Mapping of source fields to target fields.

        Returns:
            DatasetInfo for the new dataset.
        """
        info = self.get(dataset_id)
        if not info:
            raise ValueError(f"Dataset {dataset_id} not found")

        output_path = self.data_dir / f"{output_name}.jsonl"
        transformed_count = 0

        with open(info.path) as src, open(output_path, "w") as dest:
            for line in src:
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)

                    # Apply field mapping
                    if field_mapping:
                        data = {
                            target: data.get(source)
                            for source, target in field_mapping.items()
                        }

                    # Apply template
                    if template:
                        try:
                            data["text"] = template.format(**data)
                        except KeyError as e:
                            # Skip samples with missing fields
                            continue

                    dest.write(json.dumps(data) + "\n")
                    transformed_count += 1

                except json.JSONDecodeError:
                    continue

        # Register the transformed dataset
        return await self.register(str(output_path), name=output_name)

    def delete(self, dataset_id: str) -> bool:
        """Delete a dataset."""
        if dataset_id not in self._registry:
            return False

        info = self._registry[dataset_id]
        path = Path(info["path"])

        # Delete the file and parent directory if empty
        if path.exists():
            path.unlink()
        if path.parent.exists() and not any(path.parent.iterdir()):
            path.parent.rmdir()

        del self._registry[dataset_id]
        self._save_registry()
        return True


# Singleton manager instance
_manager: DatasetManager | None = None


def get_dataset_manager(data_dir: Path | None = None) -> DatasetManager:
    """Get or create the global dataset manager."""
    global _manager
    if _manager is None:
        _manager = DatasetManager(data_dir)
    return _manager
