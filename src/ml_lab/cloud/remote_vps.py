"""Remote VPS provider for training on your own hardware."""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..credentials import ProviderType, get_vault


@dataclass
class VPSConfig:
    """Configuration for a remote VPS."""

    name: str
    host: str
    user: str
    port: int = 22
    ssh_key_path: str | None = None
    gpu_type: str | None = None  # e.g., "rtx_4090", "a100"
    gpu_count: int = 1
    monthly_cost_usd: float | None = None  # For cost amortization
    work_dir: str = "~/ml-lab"  # Remote working directory
    conda_env: str | None = None  # Optional conda environment
    python_path: str = "python"  # Python executable


@dataclass
class VPSStatus:
    """Status of a remote VPS."""

    name: str
    online: bool
    gpu_available: bool = False
    gpu_memory_used_mb: int = 0
    gpu_memory_total_mb: int = 0
    gpu_utilization_pct: float = 0.0
    cpu_load: float = 0.0
    disk_free_gb: float = 0.0
    running_jobs: list[str] = field(default_factory=list)
    error: str | None = None


class RemoteVPSManager:
    """
    Manages remote VPS machines for training.

    Works with any SSH-accessible machine - Hetzner, Hostinger, OVH,
    home servers, university clusters, etc.
    """

    def __init__(self):
        self._config_path = Path.home() / ".config" / "ml-lab" / "vps_hosts.json"
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        self._hosts: dict[str, VPSConfig] = {}
        self._load_hosts()

    def _load_hosts(self) -> None:
        """Load VPS configurations from disk."""
        if self._config_path.exists():
            with open(self._config_path) as f:
                data = json.load(f)
                for name, config in data.items():
                    self._hosts[name] = VPSConfig(**config)

    def _save_hosts(self) -> None:
        """Save VPS configurations to disk."""
        data = {}
        for name, config in self._hosts.items():
            data[name] = {
                "name": config.name,
                "host": config.host,
                "user": config.user,
                "port": config.port,
                "ssh_key_path": config.ssh_key_path,
                "gpu_type": config.gpu_type,
                "gpu_count": config.gpu_count,
                "monthly_cost_usd": config.monthly_cost_usd,
                "work_dir": config.work_dir,
                "conda_env": config.conda_env,
                "python_path": config.python_path,
            }
        with open(self._config_path, "w") as f:
            json.dump(data, f, indent=2)

    def _get_ssh_key_path(self, config: VPSConfig) -> str | None:
        """Get SSH key path, checking vault if not in config."""
        if config.ssh_key_path:
            return config.ssh_key_path

        # Check vault for default key
        vault = get_vault()
        if vault.is_unlocked:
            cred = vault.get(ProviderType.REMOTE_VPS)
            if cred and cred.ssh_key_path:
                return cred.ssh_key_path

        return None

    def _build_ssh_cmd(self, config: VPSConfig, command: str) -> list[str]:
        """Build SSH command with proper arguments."""
        ssh_args = [
            "ssh",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=10",
            "-p", str(config.port),
        ]

        key_path = self._get_ssh_key_path(config)
        if key_path:
            ssh_args.extend(["-i", key_path])

        ssh_args.append(f"{config.user}@{config.host}")
        ssh_args.append(command)

        return ssh_args

    def _build_scp_cmd(
        self,
        config: VPSConfig,
        local_path: str,
        remote_path: str,
        download: bool = False,
    ) -> list[str]:
        """Build SCP command for file transfer."""
        scp_args = [
            "scp",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-P", str(config.port),
        ]

        key_path = self._get_ssh_key_path(config)
        if key_path:
            scp_args.extend(["-i", key_path])

        remote_full = f"{config.user}@{config.host}:{remote_path}"

        if download:
            scp_args.extend([remote_full, local_path])
        else:
            scp_args.extend([local_path, remote_full])

        return scp_args

    def _build_rsync_cmd(
        self,
        config: VPSConfig,
        local_path: str,
        remote_path: str,
        download: bool = False,
    ) -> list[str]:
        """Build rsync command for efficient file sync."""
        rsync_args = [
            "rsync",
            "-avz",
            "--progress",
            "-e", f"ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p {config.port}"
            + (f" -i {self._get_ssh_key_path(config)}" if self._get_ssh_key_path(config) else ""),
        ]

        remote_full = f"{config.user}@{config.host}:{remote_path}"

        if download:
            rsync_args.extend([remote_full, local_path])
        else:
            rsync_args.extend([local_path, remote_full])

        return rsync_args

    def register(
        self,
        name: str,
        host: str,
        user: str,
        port: int = 22,
        ssh_key_path: str | None = None,
        gpu_type: str | None = None,
        gpu_count: int = 1,
        monthly_cost_usd: float | None = None,
        work_dir: str = "~/ml-lab",
        conda_env: str | None = None,
    ) -> VPSConfig:
        """Register a new VPS host."""
        config = VPSConfig(
            name=name,
            host=host,
            user=user,
            port=port,
            ssh_key_path=ssh_key_path,
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            monthly_cost_usd=monthly_cost_usd,
            work_dir=work_dir,
            conda_env=conda_env,
        )
        self._hosts[name] = config
        self._save_hosts()
        return config

    def unregister(self, name: str) -> bool:
        """Remove a VPS host."""
        if name in self._hosts:
            del self._hosts[name]
            self._save_hosts()
            return True
        return False

    def get(self, name: str) -> VPSConfig | None:
        """Get a VPS configuration by name."""
        return self._hosts.get(name)

    def list(self) -> list[VPSConfig]:
        """List all registered VPS hosts."""
        return list(self._hosts.values())

    async def check_status(self, name: str) -> VPSStatus:
        """Check the status of a VPS."""
        config = self._hosts.get(name)
        if not config:
            return VPSStatus(name=name, online=False, error="VPS not registered")

        try:
            # Check connectivity and get system info
            check_script = """
echo "===ONLINE==="
# GPU info (nvidia-smi)
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo "NO_GPU"
else
    echo "NO_GPU"
fi
echo "===GPU_END==="
# CPU load
uptime | awk -F'load average:' '{print $2}' | awk -F',' '{print $1}' | tr -d ' '
echo "===CPU_END==="
# Disk free
df -BG ~ | tail -1 | awk '{print $4}' | tr -d 'G'
echo "===DISK_END==="
# Running training jobs (check for python processes with training keywords)
pgrep -a python | grep -E '(train|finetune|sft)' | awk '{print $1}' || echo ""
echo "===JOBS_END==="
"""
            proc = await asyncio.create_subprocess_exec(
                *self._build_ssh_cmd(config, check_script),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)

            if proc.returncode != 0:
                return VPSStatus(
                    name=name,
                    online=False,
                    error=stderr.decode().strip() or "SSH connection failed",
                )

            output = stdout.decode()

            if "===ONLINE===" not in output:
                return VPSStatus(name=name, online=False, error="Unexpected response")

            status = VPSStatus(name=name, online=True)

            # Parse GPU info
            gpu_section = output.split("===GPU_END===")[0].split("===ONLINE===")[1].strip()
            if gpu_section and "NO_GPU" not in gpu_section:
                try:
                    parts = gpu_section.split(",")
                    if len(parts) >= 3:
                        status.gpu_memory_used_mb = int(parts[0].strip())
                        status.gpu_memory_total_mb = int(parts[1].strip())
                        status.gpu_utilization_pct = float(parts[2].strip())
                        status.gpu_available = True
                except (ValueError, IndexError):
                    pass

            # Parse CPU load
            cpu_section = output.split("===CPU_END===")[0].split("===GPU_END===")[1].strip()
            try:
                status.cpu_load = float(cpu_section)
            except ValueError:
                pass

            # Parse disk free
            disk_section = output.split("===DISK_END===")[0].split("===CPU_END===")[1].strip()
            try:
                status.disk_free_gb = float(disk_section)
            except ValueError:
                pass

            # Parse running jobs
            jobs_section = output.split("===JOBS_END===")[0].split("===DISK_END===")[1].strip()
            if jobs_section:
                status.running_jobs = [j for j in jobs_section.split("\n") if j.strip()]

            return status

        except asyncio.TimeoutError:
            return VPSStatus(name=name, online=False, error="Connection timeout")
        except Exception as e:
            return VPSStatus(name=name, online=False, error=str(e))

    async def run_command(
        self,
        name: str,
        command: str,
        timeout: int = 60,
    ) -> tuple[int, str, str]:
        """Run a command on the VPS."""
        config = self._hosts.get(name)
        if not config:
            raise ValueError(f"VPS '{name}' not registered")

        proc = await asyncio.create_subprocess_exec(
            *self._build_ssh_cmd(config, command),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )
            return proc.returncode or 0, stdout.decode(), stderr.decode()
        except asyncio.TimeoutError:
            proc.kill()
            return -1, "", "Command timed out"

    async def sync_to_vps(
        self,
        name: str,
        local_path: str,
        remote_path: str | None = None,
    ) -> str:
        """
        Sync a file or directory to the VPS.

        Returns the remote path where files were synced.
        """
        config = self._hosts.get(name)
        if not config:
            raise ValueError(f"VPS '{name}' not registered")

        # Default to work_dir/data
        if remote_path is None:
            remote_path = f"{config.work_dir}/data/"

        # Ensure remote directory exists
        await self.run_command(name, f"mkdir -p {remote_path}")

        # Use rsync for efficient sync
        proc = await asyncio.create_subprocess_exec(
            *self._build_rsync_cmd(config, local_path, remote_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(f"Sync failed: {stderr.decode()}")

        # Return the full remote path
        local_name = Path(local_path).name
        return f"{remote_path.rstrip('/')}/{local_name}"

    async def sync_from_vps(
        self,
        name: str,
        remote_path: str,
        local_path: str,
    ) -> None:
        """Sync files from the VPS to local."""
        config = self._hosts.get(name)
        if not config:
            raise ValueError(f"VPS '{name}' not registered")

        # Ensure local directory exists
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)

        proc = await asyncio.create_subprocess_exec(
            *self._build_rsync_cmd(config, local_path, remote_path, download=True),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(f"Download failed: {stderr.decode()}")

    async def setup_environment(self, name: str) -> tuple[bool, str]:
        """
        Set up the training environment on the VPS.

        Installs required packages if not present.
        """
        config = self._hosts.get(name)
        if not config:
            raise ValueError(f"VPS '{name}' not registered")

        setup_script = f"""
set -e
mkdir -p {config.work_dir}/{{data,outputs,logs}}

# Check if we need to install packages
if ! {config.python_path} -c "import transformers, peft, trl" 2>/dev/null; then
    echo "Installing training dependencies..."
    {config.python_path} -m pip install --quiet torch transformers peft trl datasets accelerate bitsandbytes
fi

echo "Environment ready"
{config.python_path} -c "import torch; print(f'PyTorch: {{torch.__version__}}, CUDA: {{torch.cuda.is_available()}}')"
"""

        if config.conda_env:
            setup_script = f"source ~/.bashrc && conda activate {config.conda_env} && " + setup_script

        returncode, stdout, stderr = await self.run_command(name, setup_script, timeout=300)

        if returncode != 0:
            return False, f"Setup failed: {stderr}"

        return True, stdout

    async def launch_training(
        self,
        name: str,
        script_content: str,
        run_id: str | None = None,
    ) -> str:
        """
        Launch a training run on the VPS.

        Uses tmux to keep the process running after disconnect.
        Returns the run ID.
        """
        config = self._hosts.get(name)
        if not config:
            raise ValueError(f"VPS '{name}' not registered")

        run_id = run_id or str(uuid.uuid4())[:8]
        run_dir = f"{config.work_dir}/runs/{run_id}"
        script_path = f"{run_dir}/train.py"
        log_path = f"{run_dir}/train.log"

        # Create run directory and script
        await self.run_command(name, f"mkdir -p {run_dir}")

        # Write script via heredoc
        escaped_content = script_content.replace("'", "'\"'\"'")
        write_cmd = f"cat > {script_path} << 'SCRIPT_EOF'\n{script_content}\nSCRIPT_EOF"

        returncode, _, stderr = await self.run_command(name, write_cmd, timeout=30)
        if returncode != 0:
            raise RuntimeError(f"Failed to write training script: {stderr}")

        # Build the training command
        python_cmd = config.python_path
        if config.conda_env:
            python_cmd = f"source ~/.bashrc && conda activate {config.conda_env} && {config.python_path}"

        train_cmd = f"cd {run_dir} && {python_cmd} {script_path} > {log_path} 2>&1"

        # Launch in tmux session
        tmux_cmd = f"tmux new-session -d -s 'ml-lab-{run_id}' '{train_cmd}'"

        returncode, _, stderr = await self.run_command(name, tmux_cmd, timeout=30)
        if returncode != 0:
            raise RuntimeError(f"Failed to launch training: {stderr}")

        return run_id

    async def get_training_logs(
        self,
        name: str,
        run_id: str,
        tail_lines: int = 100,
    ) -> str:
        """Get recent logs from a training run."""
        config = self._hosts.get(name)
        if not config:
            raise ValueError(f"VPS '{name}' not registered")

        log_path = f"{config.work_dir}/runs/{run_id}/train.log"
        returncode, stdout, _ = await self.run_command(
            name,
            f"tail -n {tail_lines} {log_path} 2>/dev/null || echo 'Log not found'",
        )

        return stdout

    async def check_training_status(
        self,
        name: str,
        run_id: str,
    ) -> dict[str, Any]:
        """Check if a training run is still active."""
        config = self._hosts.get(name)
        if not config:
            raise ValueError(f"VPS '{name}' not registered")

        # Check if tmux session exists
        returncode, stdout, _ = await self.run_command(
            name,
            f"tmux has-session -t 'ml-lab-{run_id}' 2>/dev/null && echo 'running' || echo 'stopped'",
        )

        is_running = "running" in stdout

        # Get last few log lines
        logs = await self.get_training_logs(name, run_id, tail_lines=20)

        return {
            "run_id": run_id,
            "running": is_running,
            "recent_logs": logs,
        }

    async def stop_training(self, name: str, run_id: str) -> bool:
        """Stop a training run."""
        config = self._hosts.get(name)
        if not config:
            raise ValueError(f"VPS '{name}' not registered")

        returncode, _, _ = await self.run_command(
            name,
            f"tmux kill-session -t 'ml-lab-{run_id}' 2>/dev/null",
        )

        return returncode == 0

    def get_hourly_cost(self, name: str) -> float | None:
        """Get amortized hourly cost for a VPS."""
        config = self._hosts.get(name)
        if not config or not config.monthly_cost_usd:
            return None

        # Assume 730 hours per month (365 * 24 / 12)
        return config.monthly_cost_usd / 730.0


# Singleton instance
_manager: RemoteVPSManager | None = None


def get_vps_manager() -> RemoteVPSManager:
    """Get or create the global VPS manager."""
    global _manager
    if _manager is None:
        _manager = RemoteVPSManager()
    return _manager
