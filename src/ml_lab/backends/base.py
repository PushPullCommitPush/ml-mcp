"""Base classes for training backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator


class TrainingMethod(str, Enum):
    """Supported training methods."""

    FULL = "full"  # Full parameter fine-tuning
    LORA = "lora"  # LoRA adapters
    QLORA = "qlora"  # Quantized LoRA
    DPO = "dpo"  # Direct Preference Optimization
    RLHF = "rlhf"  # Reinforcement Learning from Human Feedback
    SFT = "sft"  # Supervised Fine-Tuning


class TrainingStatus(str, Enum):
    """Status of a training run."""

    PENDING = "pending"
    INITIALIZING = "initializing"
    RUNNING = "running"
    CHECKPOINTING = "checkpointing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BackendCapabilities:
    """Capabilities of a training backend."""

    name: str
    supports_local: bool = False
    supports_remote: bool = False
    supports_distributed: bool = False
    supported_methods: list[TrainingMethod] = field(default_factory=list)
    max_model_size_b: float | None = None  # Max model size in billions of params
    supported_quantization: list[str] = field(default_factory=list)
    supports_streaming_logs: bool = True


@dataclass
class TrainingConfig:
    """Configuration for a training run."""

    # Model
    base_model: str
    method: TrainingMethod = TrainingMethod.QLORA

    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] | None = None

    # Training hyperparameters
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    epochs: int = 3
    max_steps: int | None = None
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01

    # Quantization
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"

    # Optimization
    optim: str = "paged_adamw_8bit"
    lr_scheduler_type: str = "cosine"
    gradient_checkpointing: bool = True

    # Saving
    save_steps: int = 100
    save_total_limit: int = 3
    logging_steps: int = 10

    # Advanced
    max_seq_length: int = 2048
    packing: bool = False
    seed: int = 42

    # Extra provider-specific options
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "base_model": self.base_model,
            "method": self.method.value,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "epochs": self.epochs,
            "max_steps": self.max_steps,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "load_in_4bit": self.load_in_4bit,
            "load_in_8bit": self.load_in_8bit,
            "bnb_4bit_compute_dtype": self.bnb_4bit_compute_dtype,
            "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
            "optim": self.optim,
            "lr_scheduler_type": self.lr_scheduler_type,
            "gradient_checkpointing": self.gradient_checkpointing,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "logging_steps": self.logging_steps,
            "max_seq_length": self.max_seq_length,
            "packing": self.packing,
            "seed": self.seed,
            "extra": self.extra,
        }


@dataclass
class TrainingMetrics:
    """Metrics from a training step."""

    step: int
    epoch: float
    loss: float
    learning_rate: float
    grad_norm: float | None = None
    gpu_memory_mb: float | None = None
    samples_per_second: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingRun:
    """A training run instance."""

    run_id: str
    experiment_id: str
    config: TrainingConfig
    status: TrainingStatus = TrainingStatus.PENDING
    current_step: int = 0
    total_steps: int | None = None
    current_epoch: float = 0.0
    best_loss: float | None = None
    checkpoint_path: str | None = None
    error_message: str | None = None
    metrics_history: list[TrainingMetrics] = field(default_factory=list)


class TrainingBackend(ABC):
    """Abstract base class for training backends."""

    @property
    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """Get the capabilities of this backend."""
        ...

    @abstractmethod
    async def validate_config(self, config: TrainingConfig) -> list[str]:
        """
        Validate a training configuration.

        Args:
            config: The training configuration to validate.

        Returns:
            List of validation errors (empty if valid).
        """
        ...

    @abstractmethod
    async def estimate_resources(
        self,
        config: TrainingConfig,
        dataset_size: int,
    ) -> dict[str, Any]:
        """
        Estimate resource requirements for training.

        Args:
            config: The training configuration.
            dataset_size: Number of samples in the dataset.

        Returns:
            Dict with estimates for time, memory, cost, etc.
        """
        ...

    @abstractmethod
    async def launch(
        self,
        run: TrainingRun,
        dataset_path: str,
        output_dir: str,
    ) -> str:
        """
        Launch a training run.

        Args:
            run: The training run configuration.
            dataset_path: Path to the training dataset.
            output_dir: Directory for outputs and checkpoints.

        Returns:
            The run ID.
        """
        ...

    @abstractmethod
    async def get_status(self, run_id: str) -> TrainingRun:
        """
        Get the status of a training run.

        Args:
            run_id: The run ID.

        Returns:
            The current training run state.
        """
        ...

    @abstractmethod
    async def stream_logs(self, run_id: str) -> AsyncIterator[str]:
        """
        Stream logs from a training run.

        Args:
            run_id: The run ID.

        Yields:
            Log lines as they become available.
        """
        ...

    @abstractmethod
    async def stop(self, run_id: str, save_checkpoint: bool = True) -> None:
        """
        Stop a training run.

        Args:
            run_id: The run ID.
            save_checkpoint: Whether to save a checkpoint before stopping.
        """
        ...

    @abstractmethod
    async def resume(self, run_id: str, checkpoint_path: str | None = None) -> str:
        """
        Resume a training run from a checkpoint.

        Args:
            run_id: The original run ID.
            checkpoint_path: Optional specific checkpoint to resume from.

        Returns:
            The new run ID.
        """
        ...
