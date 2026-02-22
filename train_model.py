#!/usr/bin/env python3
"""
LoRA Fine-Tuning + GGUF Export for the Oversight Pi_H Model
=============================================================
Uses MLX for Apple Silicon (M1/M2/M3/M4) local training.
Falls back to unsloth/trl for NVIDIA CUDA GPUs.
Exports a quantized GGUF and imports into Ollama for inference.
"""

import json
import subprocess
import sys
import shutil
import tempfile
from pathlib import Path

# ─── constants ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an oversight policy model. Given a tool action context, decide whether \
to allow it autonomously, deny it, or ask the user for confirmation. Respond \
with a JSON object: {"decision": "allow"|"deny"|"ask_user", "confidence": 0.0-1.0, \
"reasoning": "brief explanation"}"""

MODELFILE_TEMPLATE = """\
FROM {gguf_path}

SYSTEM {system_prompt}

PARAMETER temperature 0.1
PARAMETER num_predict 128
PARAMETER stop <|im_end|>
"""

# MLX base model (full precision, not bnb-4bit like the unsloth version)
MLX_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


# ─── training data formatting ────────────────────────────────────────────────

def format_training_example(example: dict) -> dict:
    """
    Format a training example into ChatML messages for mlx-lm.
    Returns a dict with "messages" key for chat fine-tuning.
    """
    parts = [f"Tool: {example['tool_name']}"]
    parts.append(f"Category: {example['action_category']}")

    if example.get("command_preview"):
        parts.append(f"Command: {example['command_preview']}")
    if example.get("file_path"):
        parts.append(f"File: {example['file_path']}")

    parts.append(f"Project trust: {example.get('project_trust', 0.5):.1f}")
    parts.append(f"Heuristic policy: {example.get('heuristic_policy', 'ask_user')}")

    expertise = example.get("domain_expertise", {})
    if expertise:
        top = sorted(expertise.items(), key=lambda x: -x[1])[:5]
        exp_str = ", ".join(f"{k}: {v:.1f}" for k, v in top)
        parts.append(f"Domain expertise: {exp_str}")

    user_msg = "\n".join(parts)

    response = json.dumps({
        "decision": example["label"],
        "confidence": example["confidence"],
        "reasoning": f"Based on {example.get('label_source', 'heuristic')} signal",
    })

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": response},
        ]
    }


def prepare_mlx_data(data_path: str, output_dir: str) -> dict:
    """
    Convert training JSONL into mlx-lm chat format (train.jsonl / valid.jsonl).
    Returns {"train_path": str, "valid_path": str, "train_count": int, "valid_count": int}.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not examples:
        raise ValueError(f"No training examples found in {data_path}")

    formatted = [format_training_example(ex) for ex in examples]

    # Split 90/10 train/valid
    split_idx = max(1, int(len(formatted) * 0.9))
    train_data = formatted[:split_idx]
    valid_data = formatted[split_idx:] if split_idx < len(formatted) else formatted[-1:]

    train_path = output_dir / "train.jsonl"
    valid_path = output_dir / "valid.jsonl"

    for path, data in [(train_path, train_data), (valid_path, valid_data)]:
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    return {
        "train_path": str(train_path),
        "valid_path": str(valid_path),
        "train_count": len(train_data),
        "valid_count": len(valid_data),
    }


# ─── MLX training ────────────────────────────────────────────────────────────

def train_mlx(
    base_model: str,
    data_dir: str,
    output_dir: str,
    lora_r: int = 16,
    epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 1e-5,
    max_seq_length: int = 1024,
):
    """Run LoRA fine-tuning via mlx-lm."""
    import yaml

    output_dir = Path(output_dir)
    adapter_dir = output_dir / "adapters"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Write LoRA config YAML (mlx-lm reads rank from config, not CLI flag)
    lora_config = {
        "lora_parameters": {
            "rank": lora_r,
            "alpha": lora_r,
            "dropout": 0.0,
            "scale": 1.0,
        }
    }
    config_path = output_dir / "lora_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(lora_config, f)

    iters = max(50, epochs * 100)

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", base_model,
        "--data", str(data_dir),
        "--train",
        "--adapter-path", str(adapter_dir),
        "--iters", str(iters),
        "--batch-size", str(batch_size),
        "--learning-rate", str(learning_rate),
        "--max-seq-length", str(max_seq_length),
        "-c", str(config_path),
    ]

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"MLX LoRA training failed (exit {result.returncode})")

    return str(adapter_dir)


def fuse_model(base_model: str, adapter_dir: str, output_dir: str):
    """Fuse LoRA adapters into base model, saving as safetensors."""
    output_dir = Path(output_dir)
    fused_dir = output_dir / "fused_model"
    fused_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", base_model,
        "--adapter-path", str(adapter_dir),
        "--save-path", str(fused_dir),
    ]

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"MLX fuse failed (exit {result.returncode})")

    return str(fused_dir)


def create_modelfile(model_path: str, output_path: str) -> str:
    """Write an Ollama Modelfile pointing to the model."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    content = MODELFILE_TEMPLATE.format(
        gguf_path=str(Path(model_path).resolve()),
        system_prompt=SYSTEM_PROMPT,
    )

    output_path.write_text(content)
    return str(output_path)


def create_modelfile_from_safetensors(fused_model_dir: str, output_path: str) -> str:
    """
    Write an Ollama Modelfile that imports from a local safetensors directory.
    Ollama supports: FROM <path-to-safetensors-dir>
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fused_path = str(Path(fused_model_dir).resolve())

    content = f"""FROM {fused_path}

SYSTEM {SYSTEM_PROMPT}

PARAMETER temperature 0.1
PARAMETER num_predict 128
PARAMETER stop <|im_end|>
"""
    output_path.write_text(content)
    return str(output_path)


def import_to_ollama(modelfile_path: str, model_name: str = "oversight-pi-h",
                     ollama_bin: str = None):
    """Run `ollama create` to import the model."""
    ollama = ollama_bin or shutil.which("ollama")
    if not ollama:
        # Try common locations
        for candidate in ["/usr/local/bin/ollama", "/opt/homebrew/bin/ollama",
                          str(Path.home() / "homebrew" / "bin" / "ollama")]:
            if Path(candidate).exists():
                ollama = candidate
                break

    if not ollama:
        raise FileNotFoundError(
            "ollama not found. Install from https://ollama.ai or brew install ollama"
        )

    result = subprocess.run(
        [ollama, "create", model_name, "-f", str(modelfile_path)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"ollama create failed (exit {result.returncode}):\n{result.stderr}"
        )

    print(f"  Model '{model_name}' imported to Ollama successfully.")
    return True


# ─── full pipeline ────────────────────────────────────────────────────────────

def run_full_pipeline(
    data_path: str,
    output_dir: str,
    base_model: str = None,
    max_seq_length: int = 1024,
    lora_r: int = 16,
    lora_alpha: int = 16,
    quant_method: str = "q4_k_m",
    epochs: int = 3,
    batch_size: int = 2,
    ollama_model_name: str = "oversight-pi-h",
):
    """Run the full training pipeline: load data → LoRA train → fuse → Ollama import."""
    import platform
    output_dir = Path(output_dir)

    # Detect backend
    is_apple_silicon = (platform.system() == "Darwin" and platform.machine() == "arm64")

    if not base_model:
        base_model = MLX_BASE_MODEL if is_apple_silicon else "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"

    if not is_apple_silicon:
        print("Not on Apple Silicon — falling back to unsloth/trl pipeline.")
        print("Install with: pip install unsloth trl datasets")
        return _run_unsloth_pipeline(
            data_path, str(output_dir), base_model, max_seq_length,
            lora_r, lora_alpha, quant_method, epochs, batch_size, ollama_model_name,
        )

    # MLX pipeline
    print(f"Using MLX on Apple Silicon ({platform.machine()})")
    print(f"Base model: {base_model}\n")

    print("Step 1/5: Preparing training data...")
    data_info = prepare_mlx_data(data_path, str(output_dir / "data"))
    print(f"  Train: {data_info['train_count']} examples, Valid: {data_info['valid_count']} examples")

    print("\nStep 2/5: Downloading model (first run only)...")
    # mlx_lm.lora handles download automatically

    print("\nStep 3/5: LoRA fine-tuning...")
    adapter_dir = train_mlx(
        base_model=base_model,
        data_dir=str(output_dir / "data"),
        output_dir=str(output_dir),
        lora_r=lora_r,
        epochs=epochs,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
    )
    print(f"  Adapters saved to {adapter_dir}")

    print("\nStep 4/5: Fusing adapters into base model...")
    fused_dir = fuse_model(base_model, adapter_dir, str(output_dir))
    print(f"  Fused model at {fused_dir}")

    print("\nStep 5/5: Importing to Ollama...")
    modelfile_path = output_dir / "Modelfile"
    create_modelfile_from_safetensors(fused_dir, str(modelfile_path))
    import_to_ollama(str(modelfile_path), ollama_model_name)

    training_count = data_info["train_count"] + data_info["valid_count"]
    print("\nTraining pipeline complete!")
    return {
        "adapter_path": adapter_dir,
        "fused_model_path": fused_dir,
        "modelfile_path": str(modelfile_path),
        "ollama_model_name": ollama_model_name,
        "training_examples": training_count,
    }


def _run_unsloth_pipeline(
    data_path, output_dir, base_model, max_seq_length,
    lora_r, lora_alpha, quant_method, epochs, batch_size, ollama_model_name,
):
    """Fallback: unsloth/trl pipeline for NVIDIA GPUs."""
    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer, SFTConfig
        from datasets import Dataset
    except ImportError as e:
        raise ImportError(
            f"Missing training dependencies: {e}\n"
            "Install with: pip install unsloth trl datasets"
        )

    output_dir = Path(output_dir)

    # Load data
    examples = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    dataset = Dataset.from_list(examples)
    print(f"  Loaded {len(dataset)} examples.")

    # Build model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model, max_seq_length=max_seq_length, load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none", use_gradient_checkpointing="unsloth",
    )

    # Format and train
    def fmt(ex):
        e = format_training_example(ex)
        # Convert messages to ChatML string for SFTTrainer
        text = ""
        for m in e["messages"]:
            text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        return {"text": text}

    formatted = dataset.map(fmt, remove_columns=dataset.column_names)
    args = SFTConfig(
        output_dir=str(output_dir), num_train_epochs=epochs,
        per_device_train_batch_size=batch_size, gradient_accumulation_steps=4,
        learning_rate=2e-4, weight_decay=0.01, warmup_steps=10,
        logging_steps=10, save_steps=100, optim="adamw_8bit",
        fp16=True, packing=True, max_seq_length=max_seq_length,
        dataset_text_field="text",
    )
    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=formatted, args=args)
    trainer.train()

    # Export GGUF
    gguf_dir = output_dir / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_gguf(str(gguf_dir), tokenizer, quantization_method=quant_method)
    gguf_files = list(gguf_dir.glob("*.gguf"))
    if not gguf_files:
        raise FileNotFoundError(f"No GGUF file found in {gguf_dir}")

    # Import to Ollama
    modelfile_path = output_dir / "Modelfile"
    create_modelfile(str(gguf_files[0]), str(modelfile_path))
    import_to_ollama(str(modelfile_path), ollama_model_name)

    return {
        "gguf_path": str(gguf_files[0]),
        "modelfile_path": str(modelfile_path),
        "ollama_model_name": ollama_model_name,
        "training_examples": len(dataset),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train oversight pi_H model")
    parser.add_argument("--data-path",
                        default=str(Path.home() / ".claude" / "oversight" / "training" / "training_data.jsonl"))
    parser.add_argument("--output-dir",
                        default=str(Path.home() / ".claude" / "oversight" / "training" / "model_output"))
    parser.add_argument("--base-model", default=None,
                        help="Base model (auto-detects: MLX on Apple Silicon, unsloth on CUDA)")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--quant-method", default="q4_k_m")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--ollama-model-name", default="oversight-pi-h")
    args = parser.parse_args()

    result = run_full_pipeline(
        data_path=args.data_path,
        output_dir=args.output_dir,
        base_model=args.base_model,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        quant_method=args.quant_method,
        epochs=args.epochs,
        batch_size=args.batch_size,
        ollama_model_name=args.ollama_model_name,
    )
    print(json.dumps(result, indent=2))
