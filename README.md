# OverClaude

**A locally-trained language model that learns your oversight preferences for Claude Code.**

Based on *"The Oversight Game: Learning to Cooperatively Balance an AI Agent's Safety and Autonomy"* (Overman & Bayati, 2025).

## What This Does

Claude Code asks for permission before running tools — but it doesn't know *you*. It doesn't know you're a Python expert who never needs to approve `pytest`, or that you always want to review `git push`. Every session starts from zero.

OverClaude fixes this by training a **small local language model (pi_H)** on your actual approval patterns. The model runs via Ollama and makes real-time allow/deny/ask decisions on Claude Code's tool calls through the hooks system — no cloud calls, no latency worth noticing, fully private. Simple heuristic guard rails handle the obvious extremes (reading files is always fine, force-pushing always asks), and the model handles everything in between — like distinguishing `pip install requests` from `curl | sudo bash`.

## How It Works

```
                    Claude Code calls a tool
                             │
                             ▼
                  PreToolUse hook fires
                             │
                             ▼
                   Classify the action
                 (21 categories via regex)
                             │
                ┌────────────┼────────────┐
                ▼            ▼            ▼
          always_play   lean_play/    always_ask
          → allow       ask_user      → ask user
          (no model)        │         (no model)
                            ▼
                   Query pi_H model
                     (via Ollama)
                            │
                ┌───────────┼───────────┐
                ▼           ▼           ▼
             allow       ask_user      deny
          (proceed)    (confirm)    (block)
```

Everything is logged to `interaction_log.jsonl`. After a few sessions, you run `oversight train` and the model learns your patterns.

## Installation

```bash
git clone https://github.com/wpoverman/OverClaude.git
cd OverClaude

# Make the CLI available (add to .bashrc/.zshrc)
alias oversight='python3 ~/OverClaude/oversight.py'

# Initialize your profile
oversight init --interactive

# Generate CLAUDE.md, settings.json, and hook scripts
oversight generate
```

### Requirements

- Python 3.10+
- [Ollama](https://ollama.com) — for serving the trained model
- For training on Apple Silicon: `pip install mlx-lm`
- For training on NVIDIA GPUs: `pip install unsloth trl datasets`

No dependencies beyond Python stdlib for the core CLI and hooks.

## Quick Start

```bash
# 1. Set up your profile and start using Claude Code
oversight init --interactive
oversight generate
claude  # hooks are now active, logging everything

# 2. After a few sessions, check what was learned
oversight status
oversight learn

# 3. Train the local model on your patterns
oversight train

# 4. Now the model gates gray-area actions in real time
oversight predict --tool Bash --command "pip install requests"
oversight predict --tool Bash --command "rm -rf /tmp/build"
```

## Training the Model

The training pipeline:

1. **Data synthesis** (`prepare_training_data.py`, stdlib only) — converts interaction logs + feedback into labeled examples via a priority cascade:
   - Explicit feedback (confidence 0.9): `too_cautious` → allow, `missed` → deny
   - Revert detection (confidence 0.8): action that was undone → deny
   - Outcome signal (confidence 0.5-0.7): exit code + stderr analysis
   - Heuristic policy echo (confidence 0.4-0.6): cold-start bootstrap

2. **LoRA fine-tuning** (`train_model.py`) — trains a Qwen2.5-1.5B-Instruct adapter on your data. On a MacBook Air M4 with 24GB RAM, training takes ~5 minutes and peaks at ~5GB memory.

3. **Ollama import** — the fused model is imported as `oversight-pi-h` and served locally.

```bash
# Prepare training data only (no GPU needed)
oversight train --data-only

# Full pipeline: data → train → fuse → import to Ollama
oversight train

# Force training even with few examples
oversight train --force
```

The model outputs structured JSON decisions:
```json
{"decision": "allow", "confidence": 0.85, "reasoning": "Standard package install in trusted Python project"}
```

## Usage

### Day-to-day

```bash
oversight generate    # regenerate policies + hooks (run before sessions)
oversight learn       # analyze logs, recommend policy shifts
oversight status      # show profile, expertise, model status
```

### Feedback

```bash
oversight feedback too_cautious shell_install    # "stop asking about pip install"
oversight feedback missed shell_destructive      # "you should have asked before that rm"
oversight feedback good_catch architecture_change # "glad you asked about that"
```

### Profile Tuning

```bash
oversight set-expertise python 5          # I'm an expert (1-5 scale)
oversight set-expertise kubernetes 1      # I'm a beginner
oversight set-policy shell_install lean_play
oversight set-policy shell_git_dangerous always_ask
oversight trust ~/my-project 0.9          # high trust
oversight trust ~/client-repo 0.3         # be cautious
```

### Model Configuration

```bash
oversight set-model enabled true
oversight set-model inference_timeout_ms 300
oversight set-model fallback_on_timeout heuristic
oversight predict --tool Bash --command "git push origin main"
```

## Architecture

```
oversight.py                  Core CLI (~1100 lines, stdlib only)
prepare_training_data.py      Log → labeled training data (stdlib only)
train_model.py                LoRA fine-tuning via MLX or unsloth

~/.claude/
├── CLAUDE.md                 Generated soft policy (natural language)
├── settings.json             Generated hard policy (allow/deny rules + hooks)
└── oversight/
    ├── profile.json          Learned profile state + model config
    ├── interaction_log.jsonl  Raw interaction events + gating decisions
    ├── hooks/
    │   └── log_action.py     Hook: classify → gate → log (baked-in profile)
    └── training/
        ├── training_data.jsonl   Labeled training examples
        └── model_output/         LoRA adapter + fused model
```

### Data Flow

```
profile.json ──→ generate ──→ CLAUDE.md + settings.json + hooks
                                                  │
                          Claude Code session (hooks log + model gates)
                                                  │
                  learn ←── analyze logs ←─────────┘
                    │
                    ▼
              update profile ──→ regenerate
                    │
                    ▼
              train ──→ LoRA fine-tune ──→ Ollama import
```

## Theoretical Background

This implements the two-player oversight game from the paper:

- **π_AI** (agent policy): shaped via `CLAUDE.md` (soft) and `settings.json` (hard)
- **π_H** (human policy): learned via the local model + heuristic fallback

The paper's **Local Alignment Theorem** guarantees that shifting from *ask* to *play* in domains where the agent benefits cannot harm the human, given the right incentive structure. We approximate this by only relaxing oversight where approval rates are consistently high.

The **Safe Minimum-Oversight Equilibrium** is approximated by the learning loop: reduce asking wherever approval rates are high, increase oversight wherever errors occur, converge on the least-oversight policy that maintains safety.

## License

MIT
