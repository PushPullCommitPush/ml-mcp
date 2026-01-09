"""Modal cloud provider integration."""

from __future__ import annotations

from typing import Any

from .base import (
    CloudProvider,
    GPUType,
    InstanceSpec,
    InstanceStatus,
    PriceQuote,
    ProviderCapabilities,
)


# Modal GPU type mapping
MODAL_GPU_MAP: dict[str, GPUType] = {
    "t4": GPUType.A10,  # Approximate
    "a10g": GPUType.A10,
    "a100": GPUType.A100_40GB,
    "a100-80gb": GPUType.A100_80GB,
    "h100": GPUType.H100,
}

# Modal pricing per GPU-second (approximate)
MODAL_PRICING: dict[str, float] = {
    "t4": 0.000164,
    "a10g": 0.000306,
    "a100": 0.001036,
    "a100-80gb": 0.001528,
    "h100": 0.002778,
}


class ModalProvider(CloudProvider):
    """
    Modal cloud provider.

    Modal uses a different paradigm - serverless functions rather than instances.
    This provider adapts the interface to work with Modal's model.

    Note: Modal integration requires the modal package and authenticated CLI.
    """

    def __init__(self):
        self._modal_available = False
        try:
            import modal  # noqa: F401

            self._modal_available = True
        except ImportError:
            pass

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name="modal",
            available_gpus=[
                GPUType.A10,
                GPUType.A100_40GB,
                GPUType.A100_80GB,
                GPUType.H100,
            ],
            supports_spot=False,  # Modal uses serverless pricing
            supports_persistent_storage=True,
            min_rental_hours=0.0,  # Per-second billing
            max_gpus_per_instance=8,
            regions=["us-east", "us-west"],
        )

    async def check_credentials(self) -> bool:
        if not self._modal_available:
            return False

        try:
            import subprocess

            result = subprocess.run(
                ["modal", "token", "check"],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    async def get_balance(self) -> float | None:
        # Modal uses usage-based billing, no pre-paid balance
        return None

    async def list_available(
        self,
        gpu_type: GPUType | None = None,
        min_gpus: int = 1,
        region: str | None = None,
    ) -> list[PriceQuote]:
        quotes = []

        for modal_name, mapped_gpu in MODAL_GPU_MAP.items():
            if gpu_type and mapped_gpu != gpu_type:
                continue

            price_per_second = MODAL_PRICING.get(modal_name, 0)
            hourly_price = price_per_second * 3600

            quotes.append(
                PriceQuote(
                    provider="modal",
                    gpu_type=mapped_gpu,
                    gpu_count=1,
                    hourly_price_usd=hourly_price,
                    availability=True,  # Modal has good availability
                    spot=False,
                )
            )

        return quotes

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
        Modal doesn't provision persistent instances.
        Instead, this creates a Modal function stub that can be invoked.
        """
        raise NotImplementedError(
            "Modal uses serverless functions, not persistent instances. "
            "Use the ModalBackend training backend instead for direct Modal integration."
        )

    async def get_instance(self, instance_id: str) -> InstanceSpec:
        raise NotImplementedError("Modal uses serverless functions, not instances")

    async def terminate(self, instance_id: str) -> None:
        raise NotImplementedError("Modal uses serverless functions, not instances")

    async def ssh_command(self, instance_id: str) -> str:
        raise NotImplementedError("Modal uses serverless functions, not instances")

    async def upload_file(
        self,
        instance_id: str,
        local_path: str,
        remote_path: str,
    ) -> None:
        raise NotImplementedError("Use Modal volumes for file storage")

    async def download_file(
        self,
        instance_id: str,
        remote_path: str,
        local_path: str,
    ) -> None:
        raise NotImplementedError("Use Modal volumes for file storage")

    async def run_command(
        self,
        instance_id: str,
        command: str,
        timeout: int | None = None,
    ) -> tuple[int, str, str]:
        raise NotImplementedError("Modal uses serverless functions, not instances")
