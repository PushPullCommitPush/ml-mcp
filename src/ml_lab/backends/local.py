"""Local training backend using transformers/peft."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

from .base import (
    BackendCapabilities,
    TrainingBackend,
    TrainingConfig,
    TrainingMethod,
    TrainingRun,
    TrainingStatus,
)


@dataclass
class LocalRunState:
    """State for a local training run."""

    run: TrainingRun
    process: subprocess.Popen | None = None
    log_file: Path | None = None
    output_dir: Path | None = None


class LocalBackend(TrainingBackend):
    """
    Local training backend using transformers, peft, and trl.

    Supports CPU, CUDA, and MPS (Apple Silicon) devices.
    """

    def __init__(self, work_dir: Path | None = None):
        """
        Initialize the local backend.

        Args:
            work_dir: Working directory for training outputs.
                     Defaults to ~/.cache/ml-lab/runs
        """
        if work_dir is None:
            work_dir = Path.home() / ".cache" / "ml-lab" / "runs"
        work_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir = work_dir
        self._runs: dict[str, LocalRunState] = {}

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            name="local",
            supports_local=True,
            supports_remote=False,
            supports_distributed=False,
            supported_methods=[
                TrainingMethod.FULL,
                TrainingMethod.LORA,
                TrainingMethod.QLORA,
                TrainingMethod.SFT,
                TrainingMethod.DPO,
            ],
            max_model_size_b=70.0,  # Depends on available VRAM
            supported_quantization=["4bit", "8bit"],
            supports_streaming_logs=True,
        )

    async def validate_config(self, config: TrainingConfig) -> list[str]:
        errors = []

        if config.method == TrainingMethod.RLHF:
            errors.append("RLHF is not supported in local backend - use a cloud provider")

        if config.load_in_4bit and config.load_in_8bit:
            errors.append("Cannot use both 4-bit and 8-bit quantization")

        if config.batch_size < 1:
            errors.append("Batch size must be at least 1")

        if config.learning_rate <= 0:
            errors.append("Learning rate must be positive")

        return errors

    async def estimate_resources(
        self,
        config: TrainingConfig,
        dataset_size: int,
    ) -> dict[str, Any]:
        # Rough estimates based on model size and config
        model_name = config.base_model.lower()

        # Estimate model size in billions
        if "70b" in model_name:
            model_size_b = 70
        elif "34b" in model_name or "35b" in model_name:
            model_size_b = 34
        elif "13b" in model_name:
            model_size_b = 13
        elif "8b" in model_name:
            model_size_b = 8
        elif "7b" in model_name:
            model_size_b = 7
        elif "3b" in model_name:
            model_size_b = 3
        elif "1b" in model_name:
            model_size_b = 1
        else:
            model_size_b = 7  # Default assumption

        # Memory estimation
        if config.load_in_4bit:
            bytes_per_param = 0.5
        elif config.load_in_8bit:
            bytes_per_param = 1
        else:
            bytes_per_param = 2  # fp16

        base_memory_gb = model_size_b * bytes_per_param

        # LoRA adds minimal overhead
        if config.method in [TrainingMethod.LORA, TrainingMethod.QLORA]:
            lora_overhead = 0.1  # ~10% for LoRA adapters
        else:
            lora_overhead = 1.0  # Full fine-tuning needs more

        # Gradient checkpointing reduces memory
        if config.gradient_checkpointing:
            gradient_multiplier = 1.5
        else:
            gradient_multiplier = 3.0

        estimated_memory_gb = base_memory_gb * (1 + lora_overhead) * gradient_multiplier

        # Time estimation (very rough)
        effective_batch = config.batch_size * config.gradient_accumulation_steps
        steps_per_epoch = dataset_size // effective_batch
        total_steps = steps_per_epoch * config.epochs

        # Assume ~1 second per step for 7B model
        time_scale = model_size_b / 7.0
        estimated_seconds = total_steps * time_scale

        return {
            "estimated_memory_gb": round(estimated_memory_gb, 1),
            "estimated_time_seconds": int(estimated_seconds),
            "estimated_time_human": _format_duration(estimated_seconds),
            "total_steps": total_steps,
            "model_size_b": model_size_b,
            "cost_usd": 0.0,  # Local is free
        }

    async def launch(
        self,
        run: TrainingRun,
        dataset_path: str,
        output_dir: str,
    ) -> str:
        run_id = run.run_id or str(uuid.uuid4())[:8]
        run.run_id = run_id

        run_dir = self.work_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Write config
        config_path = run_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(run.config.to_dict(), f, indent=2)

        # Generate training script
        script_path = run_dir / "train.py"
        script_content = self._generate_training_script(run.config, dataset_path, output_dir)
        with open(script_path, "w") as f:
            f.write(script_content)

        # Launch training process
        log_path = run_dir / "train.log"
        log_file = open(log_path, "w")

        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(run_dir),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        run.status = TrainingStatus.RUNNING
        state = LocalRunState(
            run=run,
            process=process,
            log_file=log_path,
            output_dir=Path(output_dir),
        )
        self._runs[run_id] = state

        return run_id

    def _generate_training_script(
        self,
        config: TrainingConfig,
        dataset_path: str,
        output_dir: str,
    ) -> str:
        """Generate a training script based on config."""
        return f'''"""Auto-generated training script."""

import json
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Configuration
BASE_MODEL = "{config.base_model}"
DATASET_PATH = "{dataset_path}"
OUTPUT_DIR = "{output_dir}"

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading dataset...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    print("Loading model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit={config.load_in_4bit},
        load_in_8bit={config.load_in_8bit},
        bnb_4bit_compute_dtype=torch.{config.bnb_4bit_compute_dtype},
        bnb_4bit_quant_type="{config.bnb_4bit_quant_type}",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config if {config.load_in_4bit or config.load_in_8bit} else None,
        device_map="auto",
        trust_remote_code=True,
    )

    if {config.load_in_4bit or config.load_in_8bit}:
        model = prepare_model_for_kbit_training(model)

    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r={config.lora_r},
        lora_alpha={config.lora_alpha},
        lora_dropout={config.lora_dropout},
        target_modules={config.target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"]},
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Setting up training...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs={config.epochs},
        per_device_train_batch_size={config.batch_size},
        gradient_accumulation_steps={config.gradient_accumulation_steps},
        learning_rate={config.learning_rate},
        weight_decay={config.weight_decay},
        warmup_ratio={config.warmup_ratio},
        lr_scheduler_type="{config.lr_scheduler_type}",
        optim="{config.optim}",
        logging_steps={config.logging_steps},
        save_steps={config.save_steps},
        save_total_limit={config.save_total_limit},
        gradient_checkpointing={config.gradient_checkpointing},
        fp16=torch.cuda.is_available(),
        bf16=False,
        max_grad_norm=0.3,
        report_to="none",
        seed={config.seed},
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length={config.max_seq_length},
        packing={config.packing},
        dataset_text_field="text",
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Training complete!")

if __name__ == "__main__":
    main()
'''

    async def get_status(self, run_id: str) -> TrainingRun:
        if run_id not in self._runs:
            raise ValueError(f"Run {run_id} not found")

        state = self._runs[run_id]

        # Check if process is still running
        if state.process:
            poll = state.process.poll()
            if poll is None:
                state.run.status = TrainingStatus.RUNNING
            elif poll == 0:
                state.run.status = TrainingStatus.COMPLETED
            else:
                state.run.status = TrainingStatus.FAILED
                state.run.error_message = f"Process exited with code {poll}"

        return state.run

    async def stream_logs(self, run_id: str) -> AsyncIterator[str]:
        if run_id not in self._runs:
            raise ValueError(f"Run {run_id} not found")

        state = self._runs[run_id]
        if not state.log_file or not state.log_file.exists():
            return

        with open(state.log_file) as f:
            while True:
                line = f.readline()
                if line:
                    yield line.rstrip()
                else:
                    # Check if process is still running
                    if state.process and state.process.poll() is not None:
                        break
                    await asyncio.sleep(0.1)

    async def stop(self, run_id: str, save_checkpoint: bool = True) -> None:
        if run_id not in self._runs:
            raise ValueError(f"Run {run_id} not found")

        state = self._runs[run_id]
        if state.process and state.process.poll() is None:
            state.process.terminate()
            try:
                state.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                state.process.kill()

        state.run.status = TrainingStatus.CANCELLED

    async def resume(self, run_id: str, checkpoint_path: str | None = None) -> str:
        # TODO: Implement checkpoint resumption
        raise NotImplementedError("Resume not yet implemented for local backend")


def _format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"
