"""RunPod cloud provider integration."""

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


# RunPod GPU type mapping
RUNPOD_GPU_MAP: dict[str, GPUType] = {
    "NVIDIA RTX 3090": GPUType.RTX_3090,
    "NVIDIA RTX 4090": GPUType.RTX_4090,
    "NVIDIA A10": GPUType.A10,
    "NVIDIA A40": GPUType.A40,
    "NVIDIA A100 40GB": GPUType.A100_40GB,
    "NVIDIA A100 80GB": GPUType.A100_80GB,
    "NVIDIA A100-SXM4-80GB": GPUType.A100_80GB,
    "NVIDIA H100 PCIe": GPUType.H100,
    "NVIDIA H100 SXM": GPUType.H100_SXM,
}


class RunPodProvider(CloudProvider):
    """
    RunPod cloud provider.

    Provides access to RunPod GPU cloud instances.
    API docs: https://docs.runpod.io/reference/api
    """

    API_BASE = "https://api.runpod.io/graphql"

    def __init__(self):
        self._api_key: str | None = None

    def _get_api_key(self) -> str:
        """Get the API key from the credential vault."""
        if self._api_key:
            return self._api_key

        vault = get_vault()
        if not vault.is_unlocked:
            raise RuntimeError("Credential vault is locked")

        cred = vault.get(ProviderType.RUNPOD)
        if not cred or not cred.api_key:
            raise ValueError("RunPod API key not configured")

        self._api_key = cred.api_key
        return self._api_key

    def _headers(self) -> dict[str, str]:
        """Get request headers with authentication."""
        return {
            "Authorization": f"Bearer {self._get_api_key()}",
            "Content-Type": "application/json",
        }

    async def _graphql(self, query: str, variables: dict | None = None) -> dict[str, Any]:
        """Execute a GraphQL query."""
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                self.API_BASE,
                headers=self._headers(),
                json={"query": query, "variables": variables or {}},
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            if "errors" in data:
                raise RuntimeError(f"GraphQL error: {data['errors']}")
            return data.get("data", {})

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            name="runpod",
            available_gpus=[
                GPUType.RTX_3090,
                GPUType.RTX_4090,
                GPUType.A10,
                GPUType.A40,
                GPUType.A100_40GB,
                GPUType.A100_80GB,
                GPUType.H100,
                GPUType.H100_SXM,
            ],
            supports_spot=True,
            supports_persistent_storage=True,
            min_rental_hours=0.0,  # Per-second billing
            max_gpus_per_instance=8,
            regions=["US", "EU", "CA"],
        )

    async def check_credentials(self) -> bool:
        try:
            query = "query { myself { id } }"
            await self._graphql(query)
            return True
        except Exception:
            return False

    async def get_balance(self) -> float | None:
        query = """
        query {
            myself {
                currentSpendPerHr
                creditBalance
            }
        }
        """
        data = await self._graphql(query)
        return data.get("myself", {}).get("creditBalance")

    async def list_available(
        self,
        gpu_type: GPUType | None = None,
        min_gpus: int = 1,
        region: str | None = None,
    ) -> list[PriceQuote]:
        query = """
        query GpuTypes {
            gpuTypes {
                id
                displayName
                memoryInGb
                secureCloud
                communityCloud
                lowestPrice(input: { gpuCount: 1 }) {
                    minimumBidPrice
                    uninterruptablePrice
                }
            }
        }
        """
        data = await self._graphql(query)

        quotes = []
        for gpu_info in data.get("gpuTypes", []):
            display_name = gpu_info.get("displayName", "")
            mapped_gpu = None
            for name_pattern, gpu_enum in RUNPOD_GPU_MAP.items():
                if name_pattern.lower() in display_name.lower():
                    mapped_gpu = gpu_enum
                    break

            if not mapped_gpu:
                continue

            if gpu_type and mapped_gpu != gpu_type:
                continue

            pricing = gpu_info.get("lowestPrice", {})
            spot_price = pricing.get("minimumBidPrice")
            on_demand_price = pricing.get("uninterruptablePrice")

            # Spot/bid pricing
            if spot_price:
                quotes.append(
                    PriceQuote(
                        provider="runpod",
                        gpu_type=mapped_gpu,
                        gpu_count=1,
                        hourly_price_usd=spot_price,
                        availability=True,
                        spot=True,
                    )
                )

            # On-demand pricing
            if on_demand_price:
                quotes.append(
                    PriceQuote(
                        provider="runpod",
                        gpu_type=mapped_gpu,
                        gpu_count=1,
                        hourly_price_usd=on_demand_price,
                        availability=True,
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
        # Find GPU type ID
        gpu_type_id = None
        for name_pattern, gpu_enum in RUNPOD_GPU_MAP.items():
            if gpu_enum == gpu_type:
                # Use the pattern as a search hint
                gpu_type_id = name_pattern.replace(" ", "-").upper()
                break

        if not gpu_type_id:
            raise ValueError(f"Unsupported GPU type: {gpu_type}")

        mutation = """
        mutation CreatePod($input: PodFindAndDeployOnDemandInput!) {
            podFindAndDeployOnDemand(input: $input) {
                id
                machineId
                name
                desiredStatus
                gpuCount
                vcpuCount
                memoryInGb
                runtime {
                    uptimeInSeconds
                    gpus {
                        id
                        gpuUtilPercent
                        memoryUtilPercent
                    }
                }
            }
        }
        """

        variables = {
            "input": {
                "cloudType": "SECURE" if not spot else "COMMUNITY",
                "gpuCount": gpu_count,
                "volumeInGb": storage_gb,
                "containerDiskInGb": 20,
                "gpuTypeId": gpu_type_id,
                "imageName": image or "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
                "startSsh": True,
                "volumeMountPath": "/workspace",
            }
        }

        if startup_script:
            variables["input"]["dockerArgs"] = startup_script

        data = await self._graphql(mutation, variables)
        pod = data.get("podFindAndDeployOnDemand", {})

        if not pod.get("id"):
            raise RuntimeError("Failed to create pod")

        return InstanceSpec(
            instance_id=pod["id"],
            provider="runpod",
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            vcpus=pod.get("vcpuCount", 0),
            ram_gb=pod.get("memoryInGb", 0),
            storage_gb=storage_gb,
            hourly_price_usd=0.0,  # Will be determined by actual usage
            region=region or "US",
            spot=spot,
            status=InstanceStatus.STARTING,
        )

    async def get_instance(self, instance_id: str) -> InstanceSpec:
        query = """
        query Pod($podId: String!) {
            pod(input: { podId: $podId }) {
                id
                name
                desiredStatus
                gpuCount
                vcpuCount
                memoryInGb
                volumeInGb
                costPerHr
                runtime {
                    uptimeInSeconds
                }
                machine {
                    gpuDisplayName
                }
            }
        }
        """
        data = await self._graphql(query, {"podId": instance_id})
        pod = data.get("pod", {})

        if not pod:
            raise ValueError(f"Pod {instance_id} not found")

        status_map = {
            "RUNNING": InstanceStatus.RUNNING,
            "EXITED": InstanceStatus.STOPPED,
            "CREATED": InstanceStatus.PENDING,
            "STARTING": InstanceStatus.STARTING,
        }

        gpu_display = pod.get("machine", {}).get("gpuDisplayName", "")
        gpu_type = GPUType.A100_40GB  # Default
        for name_pattern, gpu_enum in RUNPOD_GPU_MAP.items():
            if name_pattern.lower() in gpu_display.lower():
                gpu_type = gpu_enum
                break

        return InstanceSpec(
            instance_id=instance_id,
            provider="runpod",
            gpu_type=gpu_type,
            gpu_count=pod.get("gpuCount", 1),
            vcpus=pod.get("vcpuCount", 0),
            ram_gb=pod.get("memoryInGb", 0),
            storage_gb=pod.get("volumeInGb", 0),
            hourly_price_usd=pod.get("costPerHr", 0),
            region="US",
            spot=False,
            status=status_map.get(pod.get("desiredStatus", ""), InstanceStatus.PENDING),
        )

    async def terminate(self, instance_id: str) -> None:
        mutation = """
        mutation TerminatePod($podId: String!) {
            podTerminate(input: { podId: $podId })
        }
        """
        await self._graphql(mutation, {"podId": instance_id})

    async def ssh_command(self, instance_id: str) -> str:
        query = """
        query Pod($podId: String!) {
            pod(input: { podId: $podId }) {
                runtime {
                    ports {
                        ip
                        privatePort
                        publicPort
                    }
                }
            }
        }
        """
        data = await self._graphql(query, {"podId": instance_id})
        pod = data.get("pod", {})
        runtime = pod.get("runtime", {})
        ports = runtime.get("ports", [])

        ssh_port = None
        ssh_ip = None
        for port in ports:
            if port.get("privatePort") == 22:
                ssh_port = port.get("publicPort")
                ssh_ip = port.get("ip")
                break

        if not ssh_port or not ssh_ip:
            raise ValueError("SSH not available for this pod")

        return f"ssh root@{ssh_ip} -p {ssh_port}"

    async def upload_file(
        self,
        instance_id: str,
        local_path: str,
        remote_path: str,
    ) -> None:
        ssh_cmd = await self.ssh_command(instance_id)
        parts = ssh_cmd.split()
        host = parts[1]
        port = parts[3]

        import subprocess

        subprocess.run(
            ["scp", "-P", port, local_path, f"{host}:{remote_path}"],
            check=True,
        )

    async def download_file(
        self,
        instance_id: str,
        remote_path: str,
        local_path: str,
    ) -> None:
        ssh_cmd = await self.ssh_command(instance_id)
        parts = ssh_cmd.split()
        host = parts[1]
        port = parts[3]

        import subprocess

        subprocess.run(
            ["scp", "-P", port, f"{host}:{remote_path}", local_path],
            check=True,
        )

    async def run_command(
        self,
        instance_id: str,
        command: str,
        timeout: int | None = None,
    ) -> tuple[int, str, str]:
        ssh_cmd = await self.ssh_command(instance_id)
        parts = ssh_cmd.split()
        host = parts[1]
        port = parts[3]

        import subprocess

        result = subprocess.run(
            ["ssh", "-p", port, host, command],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        return result.returncode, result.stdout, result.stderr
