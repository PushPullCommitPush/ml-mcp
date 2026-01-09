"""Lambda Labs cloud provider integration."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from ..credentials import ProviderType, get_vault
from .base import (
    CloudProvider,
    GPUType,
    InstanceSpec,
    InstanceStatus,
    PriceQuote,
    ProviderCapabilities,
)


# Lambda Labs GPU type mapping
LAMBDA_GPU_MAP: dict[str, GPUType] = {
    "gpu_1x_a10": GPUType.A10,
    "gpu_1x_a100": GPUType.A100_40GB,
    "gpu_1x_a100_sxm4": GPUType.A100_80GB,
    "gpu_1x_h100_pcie": GPUType.H100,
    "gpu_1x_h100_sxm5": GPUType.H100_SXM,
    "gpu_8x_a100": GPUType.A100_80GB,
    "gpu_8x_h100_sxm5": GPUType.H100_SXM,
}


class LambdaLabsProvider(CloudProvider):
    """
    Lambda Labs cloud provider.

    Provides access to Lambda Labs GPU cloud instances.
    API docs: https://cloud.lambdalabs.com/api/v1/docs
    """

    API_BASE = "https://cloud.lambdalabs.com/api/v1"

    def __init__(self):
        self._api_key: str | None = None

    def _get_api_key(self) -> str:
        """Get the API key from the credential vault."""
        if self._api_key:
            return self._api_key

        vault = get_vault()
        if not vault.is_unlocked:
            raise RuntimeError("Credential vault is locked")

        cred = vault.get(ProviderType.LAMBDA_LABS)
        if not cred or not cred.api_key:
            raise ValueError("Lambda Labs API key not configured")

        self._api_key = cred.api_key
        return self._api_key

    def _headers(self) -> dict[str, str]:
        """Get request headers with authentication."""
        return {
            "Authorization": f"Bearer {self._get_api_key()}",
            "Content-Type": "application/json",
        }

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name="lambda_labs",
            available_gpus=[
                GPUType.A10,
                GPUType.A100_40GB,
                GPUType.A100_80GB,
                GPUType.H100,
                GPUType.H100_SXM,
            ],
            supports_spot=False,  # Lambda doesn't have spot instances
            supports_persistent_storage=True,
            min_rental_hours=0.0,
            max_gpus_per_instance=8,
            regions=["us-west-1", "us-east-1", "us-south-1", "europe-central-1"],
        )

    async def check_credentials(self) -> bool:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self.API_BASE}/instance-types",
                    headers=self._headers(),
                    timeout=10.0,
                )
                return resp.status_code == 200
        except Exception:
            return False

    async def get_balance(self) -> float | None:
        # Lambda Labs doesn't expose balance via API
        return None

    async def list_available(
        self,
        gpu_type: GPUType | None = None,
        min_gpus: int = 1,
        region: str | None = None,
    ) -> list[PriceQuote]:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.API_BASE}/instance-types",
                headers=self._headers(),
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()

        quotes = []
        for type_name, info in data.get("data", {}).items():
            mapped_gpu = LAMBDA_GPU_MAP.get(type_name)
            if not mapped_gpu:
                continue

            if gpu_type and mapped_gpu != gpu_type:
                continue

            gpu_count = info.get("specs", {}).get("gpus", 1)
            if gpu_count < min_gpus:
                continue

            # Check availability in regions
            regions_available = info.get("regions_with_capacity_available", [])
            if region and region not in [r.get("name") for r in regions_available]:
                continue

            price = info.get("price_cents_per_hour", 0) / 100.0

            for region_info in regions_available or [{"name": "unknown"}]:
                quotes.append(
                    PriceQuote(
                        provider="lambda_labs",
                        gpu_type=mapped_gpu,
                        gpu_count=gpu_count,
                        hourly_price_usd=price,
                        availability=len(regions_available) > 0,
                        region=region_info.get("name", "unknown"),
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
        # Find the matching instance type
        instance_type = None
        for type_name, mapped_gpu in LAMBDA_GPU_MAP.items():
            if mapped_gpu == gpu_type:
                # Match GPU count in type name
                if f"{gpu_count}x" in type_name or (gpu_count == 1 and "1x" in type_name):
                    instance_type = type_name
                    break

        if not instance_type:
            raise ValueError(f"No Lambda Labs instance type found for {gpu_type} x{gpu_count}")

        # Get SSH key IDs (required by Lambda)
        async with httpx.AsyncClient() as client:
            ssh_resp = await client.get(
                f"{self.API_BASE}/ssh-keys",
                headers=self._headers(),
                timeout=10.0,
            )
            ssh_resp.raise_for_status()
            ssh_keys = ssh_resp.json().get("data", [])

        if not ssh_keys:
            raise ValueError("No SSH keys configured in Lambda Labs account")

        ssh_key_ids = [k["id"] for k in ssh_keys]

        # Launch instance
        payload: dict[str, Any] = {
            "region_name": region or "us-west-1",
            "instance_type_name": instance_type,
            "ssh_key_names": [k["name"] for k in ssh_keys],
        }

        if startup_script:
            payload["file_system_names"] = []  # Required field

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.API_BASE}/instance-operations/launch",
                headers=self._headers(),
                json=payload,
                timeout=60.0,
            )
            resp.raise_for_status()
            data = resp.json()

        instance_ids = data.get("data", {}).get("instance_ids", [])
        if not instance_ids:
            raise RuntimeError("Failed to launch instance: no instance ID returned")

        instance_id = instance_ids[0]

        # Wait for instance to be ready
        instance_info = await self._wait_for_instance(instance_id)

        return InstanceSpec(
            instance_id=instance_id,
            provider="lambda_labs",
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            vcpus=instance_info.get("vcpus", 0),
            ram_gb=instance_info.get("memory_gib", 0),
            storage_gb=storage_gb,
            hourly_price_usd=instance_info.get("price_cents_per_hour", 0) / 100.0,
            region=region or "us-west-1",
            spot=False,
            status=InstanceStatus.RUNNING,
        )

    async def _wait_for_instance(
        self,
        instance_id: str,
        timeout: int = 300,
    ) -> dict[str, Any]:
        """Wait for an instance to be ready."""
        start = asyncio.get_event_loop().time()

        while True:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self.API_BASE}/instances/{instance_id}",
                    headers=self._headers(),
                    timeout=10.0,
                )
                resp.raise_for_status()
                data = resp.json().get("data", {})

            status = data.get("status")
            if status == "active":
                return data
            elif status in ("terminated", "terminating", "error"):
                raise RuntimeError(f"Instance failed with status: {status}")

            if asyncio.get_event_loop().time() - start > timeout:
                raise TimeoutError(f"Instance {instance_id} did not become ready in {timeout}s")

            await asyncio.sleep(5)

    async def get_instance(self, instance_id: str) -> InstanceSpec:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.API_BASE}/instances/{instance_id}",
                headers=self._headers(),
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json().get("data", {})

        status_map = {
            "active": InstanceStatus.RUNNING,
            "booting": InstanceStatus.STARTING,
            "terminated": InstanceStatus.TERMINATED,
            "terminating": InstanceStatus.STOPPING,
            "unhealthy": InstanceStatus.ERROR,
        }

        instance_type = data.get("instance_type", {})
        gpu_type = LAMBDA_GPU_MAP.get(
            instance_type.get("name", ""),
            GPUType.A100_40GB,
        )

        return InstanceSpec(
            instance_id=instance_id,
            provider="lambda_labs",
            gpu_type=gpu_type,
            gpu_count=instance_type.get("specs", {}).get("gpus", 1),
            vcpus=instance_type.get("specs", {}).get("vcpus", 0),
            ram_gb=instance_type.get("specs", {}).get("memory_gib", 0),
            storage_gb=instance_type.get("specs", {}).get("storage_gib", 0),
            hourly_price_usd=instance_type.get("price_cents_per_hour", 0) / 100.0,
            region=data.get("region", {}).get("name", "unknown"),
            spot=False,
            status=status_map.get(data.get("status", ""), InstanceStatus.PENDING),
        )

    async def terminate(self, instance_id: str) -> None:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.API_BASE}/instance-operations/terminate",
                headers=self._headers(),
                json={"instance_ids": [instance_id]},
                timeout=30.0,
            )
            resp.raise_for_status()

    async def ssh_command(self, instance_id: str) -> str:
        instance = await self.get_instance(instance_id)
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.API_BASE}/instances/{instance_id}",
                headers=self._headers(),
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json().get("data", {})

        ip = data.get("ip")
        if not ip:
            raise ValueError("Instance does not have an IP address yet")

        return f"ssh ubuntu@{ip}"

    async def upload_file(
        self,
        instance_id: str,
        local_path: str,
        remote_path: str,
    ) -> None:
        ssh_cmd = await self.ssh_command(instance_id)
        host = ssh_cmd.split("@")[1]

        import subprocess

        subprocess.run(
            ["scp", local_path, f"ubuntu@{host}:{remote_path}"],
            check=True,
        )

    async def download_file(
        self,
        instance_id: str,
        remote_path: str,
        local_path: str,
    ) -> None:
        ssh_cmd = await self.ssh_command(instance_id)
        host = ssh_cmd.split("@")[1]

        import subprocess

        subprocess.run(
            ["scp", f"ubuntu@{host}:{remote_path}", local_path],
            check=True,
        )

    async def run_command(
        self,
        instance_id: str,
        command: str,
        timeout: int | None = None,
    ) -> tuple[int, str, str]:
        ssh_cmd = await self.ssh_command(instance_id)
        host = ssh_cmd.split("@")[1]

        import subprocess

        result = subprocess.run(
            ["ssh", f"ubuntu@{host}", command],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        return result.returncode, result.stdout, result.stderr
