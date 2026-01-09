"""Base classes for cloud compute providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class GPUType(str, Enum):
    """Common GPU types available from cloud providers."""

    # NVIDIA Consumer
    RTX_3090 = "rtx_3090"
    RTX_4090 = "rtx_4090"

    # NVIDIA Data Center
    A10 = "a10"
    A40 = "a40"
    A100_40GB = "a100_40gb"
    A100_80GB = "a100_80gb"
    H100 = "h100"
    H100_SXM = "h100_sxm"

    # AMD
    MI250 = "mi250"
    MI300X = "mi300x"


class InstanceStatus(str, Enum):
    """Status of a cloud instance."""

    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    TERMINATED = "terminated"
    ERROR = "error"


@dataclass
class GPUSpec:
    """Specification for a GPU type."""

    gpu_type: GPUType
    vram_gb: int
    fp16_tflops: float
    memory_bandwidth_gbps: int


# GPU specifications database
GPU_SPECS: dict[GPUType, GPUSpec] = {
    GPUType.RTX_3090: GPUSpec(GPUType.RTX_3090, 24, 35.6, 936),
    GPUType.RTX_4090: GPUSpec(GPUType.RTX_4090, 24, 82.6, 1008),
    GPUType.A10: GPUSpec(GPUType.A10, 24, 31.2, 600),
    GPUType.A40: GPUSpec(GPUType.A40, 48, 37.4, 696),
    GPUType.A100_40GB: GPUSpec(GPUType.A100_40GB, 40, 77.9, 1555),
    GPUType.A100_80GB: GPUSpec(GPUType.A100_80GB, 80, 77.9, 2039),
    GPUType.H100: GPUSpec(GPUType.H100, 80, 267.6, 2000),
    GPUType.H100_SXM: GPUSpec(GPUType.H100_SXM, 80, 989.4, 3350),
    GPUType.MI250: GPUSpec(GPUType.MI250, 128, 90.5, 3200),
    GPUType.MI300X: GPUSpec(GPUType.MI300X, 192, 163.4, 5300),
}


@dataclass
class InstanceSpec:
    """Specification for a cloud instance."""

    instance_id: str
    provider: str
    gpu_type: GPUType
    gpu_count: int
    vcpus: int
    ram_gb: int
    storage_gb: int
    hourly_price_usd: float
    region: str
    spot: bool = False
    status: InstanceStatus = InstanceStatus.PENDING


@dataclass
class ProviderCapabilities:
    """Capabilities of a cloud provider."""

    name: str
    available_gpus: list[GPUType]
    supports_spot: bool = True
    supports_persistent_storage: bool = True
    min_rental_hours: float = 0.0
    max_gpus_per_instance: int = 8
    regions: list[str] = field(default_factory=list)


@dataclass
class PriceQuote:
    """Price quote for a specific configuration."""

    provider: str
    gpu_type: GPUType
    gpu_count: int
    hourly_price_usd: float
    availability: bool
    queue_position: int | None = None
    region: str = ""
    spot: bool = False
    estimated_wait_minutes: int | None = None


class CloudProvider(ABC):
    """Abstract base class for cloud compute providers."""

    @property
    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """Get the capabilities of this provider."""
        ...

    @abstractmethod
    async def check_credentials(self) -> bool:
        """
        Verify that credentials are valid.

        Returns:
            True if credentials are valid and working.
        """
        ...

    @abstractmethod
    async def get_balance(self) -> float | None:
        """
        Get current account balance in USD.

        Returns:
            Balance in USD, or None if not applicable.
        """
        ...

    @abstractmethod
    async def list_available(
        self,
        gpu_type: GPUType | None = None,
        min_gpus: int = 1,
        region: str | None = None,
    ) -> list[PriceQuote]:
        """
        List available instances with pricing.

        Args:
            gpu_type: Filter by GPU type.
            min_gpus: Minimum number of GPUs required.
            region: Filter by region.

        Returns:
            List of available configurations with pricing.
        """
        ...

    @abstractmethod
    async def provision(
        self,
        gpu_type: GPUType,
        gpu_count: int = 1,
        region: str | None = None,
        spot: bool = False,
        storage_gb: int = 100,
        image: str | None = None,
        startup_script: str | None = None,
    ) -> InstanceSpec:
        """
        Provision a new instance.

        Args:
            gpu_type: Type of GPU to provision.
            gpu_count: Number of GPUs.
            region: Region to provision in.
            spot: Whether to use spot/preemptible pricing.
            storage_gb: Storage size in GB.
            image: Custom image/template to use.
            startup_script: Script to run on startup.

        Returns:
            The provisioned instance specification.
        """
        ...

    @abstractmethod
    async def get_instance(self, instance_id: str) -> InstanceSpec:
        """
        Get the current state of an instance.

        Args:
            instance_id: The instance ID.

        Returns:
            Current instance specification.
        """
        ...

    @abstractmethod
    async def terminate(self, instance_id: str) -> None:
        """
        Terminate an instance.

        Args:
            instance_id: The instance to terminate.
        """
        ...

    @abstractmethod
    async def ssh_command(self, instance_id: str) -> str:
        """
        Get the SSH command to connect to an instance.

        Args:
            instance_id: The instance ID.

        Returns:
            SSH command string.
        """
        ...

    @abstractmethod
    async def upload_file(
        self,
        instance_id: str,
        local_path: str,
        remote_path: str,
    ) -> None:
        """
        Upload a file to an instance.

        Args:
            instance_id: The instance ID.
            local_path: Local file path.
            remote_path: Remote destination path.
        """
        ...

    @abstractmethod
    async def download_file(
        self,
        instance_id: str,
        remote_path: str,
        local_path: str,
    ) -> None:
        """
        Download a file from an instance.

        Args:
            instance_id: The instance ID.
            remote_path: Remote file path.
            local_path: Local destination path.
        """
        ...

    @abstractmethod
    async def run_command(
        self,
        instance_id: str,
        command: str,
        timeout: int | None = None,
    ) -> tuple[int, str, str]:
        """
        Run a command on an instance.

        Args:
            instance_id: The instance ID.
            command: Command to run.
            timeout: Timeout in seconds.

        Returns:
            Tuple of (exit_code, stdout, stderr).
        """
        ...
