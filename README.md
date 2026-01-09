# ML Lab MCP

A comprehensive MCP (Model Context Protocol) server for ML model training, fine-tuning, and experimentation. Transform your AI assistant into a full ML engineering environment.

## Features

### Unified Credential Management
- Encrypted vault for API keys (Lambda Labs, RunPod, Mistral, OpenAI, Together AI, etc.)
- PBKDF2 key derivation with AES encryption
- Never stores credentials in plaintext

### Dataset Management
- Register datasets from local files (JSONL, CSV, Parquet)
- Automatic schema inference and statistics
- Train/val/test splitting
- Template-based transformations

### Experiment Tracking
- SQLite-backed experiment storage
- Version control and comparison
- Fork experiments with config modifications
- Full metrics history

### Multi-Backend Training
- **Local**: transformers + peft + trl for local GPU training
- **Mistral API**: Native fine-tuning for Mistral models
- **Together AI**: Hosted fine-tuning service
- **OpenAI**: GPT model fine-tuning

### Cloud GPU Provisioning
- **Lambda Labs**: H100, A100 instances
- **RunPod**: Spot and on-demand GPUs
- Automatic price comparison across providers
- Smart routing based on cost and availability

### Cost Estimation
- Pre-training cost estimates across all providers
- Real-time pricing queries
- Time estimates based on model and dataset size

## Installation

```bash
pip install ml-lab-mcp

# With training dependencies
pip install ml-lab-mcp[training]

# With cloud provider support
pip install ml-lab-mcp[cloud]

# Everything
pip install ml-lab-mcp[training,cloud,dev]
```

## Quick Start

### 1. Initialize and Create Vault

```bash
ml-lab init
ml-lab vault create
```

### 2. Add Provider Credentials

```bash
ml-lab vault unlock
ml-lab vault add --provider lambda_labs --api-key YOUR_KEY
ml-lab vault add --provider mistral --api-key YOUR_KEY
```

### 3. Configure with Claude Code / Claude Desktop

Add to your MCP configuration:

```json
{
  "mcpServers": {
    "ml-lab": {
      "command": "ml-lab",
      "args": ["serve"]
    }
  }
}
```

## MCP Tools

### Credentials

| Tool | Description |
|------|-------------|
| `creds_create_vault` | Create encrypted credential vault |
| `creds_unlock` | Unlock vault with password |
| `creds_add` | Add provider credentials |
| `creds_list` | List configured providers |
| `creds_test` | Verify credentials work |

### Datasets

| Tool | Description |
|------|-------------|
| `dataset_register` | Register a local dataset file |
| `dataset_list` | List all datasets |
| `dataset_inspect` | View schema and statistics |
| `dataset_preview` | Preview samples |
| `dataset_split` | Create train/val/test splits |
| `dataset_transform` | Apply template transformations |

### Experiments

| Tool | Description |
|------|-------------|
| `experiment_create` | Create new experiment |
| `experiment_list` | List experiments |
| `experiment_get` | Get experiment details |
| `experiment_compare` | Compare multiple experiments |
| `experiment_fork` | Fork with modifications |

### Training

| Tool | Description |
|------|-------------|
| `train_estimate` | Estimate cost/time across providers |
| `train_launch` | Start training run |
| `train_status` | Check run status |
| `train_stop` | Stop training |

### Infrastructure

| Tool | Description |
|------|-------------|
| `infra_list_gpus` | List available GPUs with pricing |
| `infra_provision` | Provision cloud instance |
| `infra_terminate` | Terminate instance |

## Example Workflow

```
User: Fine-tune Llama 3.1 8B on my customer support data

Claude: I'll help you fine-tune Llama 3.1 8B. Let me first check your data.

[Uses dataset_register to load ./support_data.jsonl]

Found 15,000 samples. Let me estimate the costs:

[Uses train_estimate]

| Provider    | Time   | Cost   | Availability |
|-------------|--------|--------|--------------|
| Local (4090)| ~12h   | $0     | Ready        |
| Lambda H100 | ~2h    | $5.50  | 3 available  |
| Together API| ~4h    | $12    | Queued       |

Which would you prefer?

User: Use Lambda Labs

Claude: [Uses experiment_create, then train_launch with backend=lambda]

Training started! Run ID: abc123
I'll monitor progress and let you know when it completes.
```

## Architecture

```
src/ml_lab/
├── server.py           # MCP server entry point
├── credentials.py      # Encrypted credential vault
├── cli.py              # Command-line interface
├── backends/
│   ├── base.py         # Training backend interface
│   ├── local.py        # Local GPU training
│   ├── mistral_api.py  # Mistral fine-tuning API
│   ├── together_api.py # Together AI API
│   └── openai_api.py   # OpenAI fine-tuning API
├── cloud/
│   ├── base.py         # Cloud provider interface
│   ├── lambda_labs.py  # Lambda Labs integration
│   ├── runpod.py       # RunPod integration
│   └── modal_provider.py # Modal integration
├── storage/
│   ├── datasets.py     # Dataset management
│   └── experiments.py  # Experiment tracking
└── evals/
    └── benchmarks.py   # Evaluation suite
```

## Security

- Credentials encrypted with Fernet (AES-128-CBC)
- PBKDF2-SHA256 key derivation (480,000 iterations)
- Vault file permissions set to 600 (owner read/write only)
- API keys never logged or transmitted unencrypted

## Supported Providers

### Compute Providers
- Lambda Labs (H100, A100, A10)
- RunPod (H100, A100, RTX 4090)
- Modal (serverless GPU functions)

### Fine-Tuning APIs
- Mistral AI (Mistral, Mixtral, Codestral)
- Together AI (Llama, Mistral, Qwen)
- OpenAI (GPT-4o, GPT-3.5)

### Model Hubs
- Hugging Face Hub
- Replicate

## Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.

## License

MIT License - see LICENSE file for details.
