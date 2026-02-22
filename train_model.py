#!/usr/bin/env python3
"""
LoRA Fine-Tuning + GGUF Export for the Oversight Pi_H Model
=============================================================
Requires: unsloth, trl, datasets (install via requirements-train.txt)
Exports a quantized GGUF and imports into Ollama for inference.
"""

import json
import subprocess
import sys
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


# ─── data loading ─────────────────────────────────────────────────────────────

def load_training_data(data_path: str):
    """Load JSONL training data into a HuggingFace Dataset."""
    from datasets import Dataset

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

    return Dataset.from_list(examples)


def format_training_example(example: dict) -> str:
    """
    Format a training example into ChatML template.
    System prompt + action context as user message + JSON decision as assistant response.
    """
    # Build user message with action context
    parts = [f"Tool: {example['tool_name']}"]
    parts.append(f"Category: {example['action_category']}")

    if example.get("command_preview"):
        parts.append(f"Command: {example['command_preview']}")
    if example.get("file_path"):
        parts.append(f"File: {example['file_path']}")

    parts.append(f"Project trust: {example.get('project_trust', 0.5):.1f}")
    parts.append(f"Heuristic policy: {example.get('heuristic_policy', 'ask_user')}")

    # Include top domain expertise scores
    expertise = example.get("domain_expertise", {})
    if expertise:
        top = sorted(expertise.items(), key=lambda x: -x[1])[:5]
        exp_str = ", ".join(f"{k}: {v:.1f}" for k, v in top)
        parts.append(f"Domain expertise: {exp_str}")

    user_msg = "\n".join(parts)

    # Build assistant response
    response = json.dumps({
        "decision": example["label"],
        "confidence": example["confidence"],
        "reasoning": f"Based on {example.get('label_source', 'heuristic')} signal",
    })

    # ChatML format
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n{response}<|im_end|>"
    )


# ─── model setup ──────────────────────────────────────────────────────────────

def build_model_and_tokenizer(base_model: str, max_seq_length: int = 1024):
    """Load base model via unsloth's FastLanguageModel."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    return model, tokenizer


def configure_lora(model, r: int = 16, lora_alpha: int = 16):
    """Apply LoRA adapters to attention + MLP projections."""
    from unsloth import FastLanguageModel

    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    return model


# ─── training ─────────────────────────────────────────────────────────────────

def train(model, tokenizer, dataset, output_dir: str,
          epochs: int = 3, batch_size: int = 2):
    """Run SFTTrainer with packing and adamw_8bit."""
    from trl import SFTTrainer, SFTConfig

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Format all examples
    def format_fn(examples):
        return [format_training_example(ex) for ex in
                [dict(zip(examples.keys(), vals)) for vals in zip(*examples.values())]]

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_steps=10,
        logging_steps=10,
        save_steps=100,
        optim="adamw_8bit",
        fp16=True,
        packing=True,
        max_seq_length=1024,
        dataset_text_field="text",
    )

    # Pre-format the dataset
    formatted = dataset.map(
        lambda ex: {"text": format_training_example(ex)},
        remove_columns=dataset.column_names,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(str(output_dir / "lora_adapter"))

    return output_dir / "lora_adapter"


# ─── export ───────────────────────────────────────────────────────────────────

def export_gguf(model, tokenizer, output_dir: str, quant_method: str = "q4_k_m"):
    """Export the fine-tuned model to GGUF format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained_gguf(
        str(output_dir),
        tokenizer,
        quantization_method=quant_method,
    )

    # Find the exported GGUF file
    gguf_files = list(output_dir.glob("*.gguf"))
    if not gguf_files:
        raise FileNotFoundError(f"No GGUF file found in {output_dir}")

    return gguf_files[0]


def create_modelfile(gguf_path: str, output_path: str) -> str:
    """Write an Ollama Modelfile pointing to the GGUF."""
    gguf_path = Path(gguf_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    content = MODELFILE_TEMPLATE.format(
        gguf_path=str(gguf_path.resolve()),
        system_prompt=SYSTEM_PROMPT,
    )

    output_path.write_text(content)
    return str(output_path)


def import_to_ollama(modelfile_path: str, model_name: str = "oversight-pi-h"):
    """Run `ollama create` to import the model."""
    result = subprocess.run(
        ["ollama", "create", model_name, "-f", str(modelfile_path)],
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
    base_model: str = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    max_seq_length: int = 1024,
    lora_r: int = 16,
    lora_alpha: int = 16,
    quant_method: str = "q4_k_m",
    epochs: int = 3,
    batch_size: int = 2,
    ollama_model_name: str = "oversight-pi-h",
):
    """Run the full training pipeline: load data → train → export → import."""
    output_dir = Path(output_dir)

    print("Step 1/5: Loading training data...")
    dataset = load_training_data(data_path)
    print(f"  Loaded {len(dataset)} examples.")

    print("Step 2/5: Building model and tokenizer...")
    model, tokenizer = build_model_and_tokenizer(base_model, max_seq_length)

    print("Step 3/5: Configuring LoRA and training...")
    model = configure_lora(model, r=lora_r, lora_alpha=lora_alpha)
    adapter_path = train(model, tokenizer, dataset, str(output_dir),
                         epochs=epochs, batch_size=batch_size)
    print(f"  Adapter saved to {adapter_path}")

    print("Step 4/5: Exporting to GGUF...")
    gguf_path = export_gguf(model, tokenizer, str(output_dir / "gguf"), quant_method)
    print(f"  GGUF exported to {gguf_path}")

    print("Step 5/5: Importing to Ollama...")
    modelfile_path = output_dir / "Modelfile"
    create_modelfile(str(gguf_path), str(modelfile_path))
    import_to_ollama(str(modelfile_path), ollama_model_name)

    print("\nTraining pipeline complete!")
    return {
        "adapter_path": str(adapter_path),
        "gguf_path": str(gguf_path),
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
    parser.add_argument("--base-model", default="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit")
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
