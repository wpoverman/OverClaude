# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Single-file Python CLI tool (~1000 lines in `oversight.py`) implementing "The Oversight Game" (Overman & Bayati, 2025). It maintains a persistent oversight profile that learns user preferences across Claude Code sessions, generating `~/.claude/CLAUDE.md` (soft policy) and `~/.claude/settings.json` (hard permissions) to shape Claude Code's play/ask behavior.

## Running

```bash
python3 oversight.py <command>
```

No dependencies beyond Python 3 standard library for core commands. Model training requires optional deps (see `requirements-train.txt`). No build step, no virtual environment needed.

## CLI Commands

- `init [--interactive]` — Create profile; `--interactive` prompts for domain expertise
- `generate` — Write `~/.claude/CLAUDE.md`, `~/.claude/settings.json`, and hook scripts from profile
- `status` — Display current profile (expertise, policies, trust, model status)
- `learn` — Analyze `interaction_log.jsonl`, recommend and apply policy shifts
- `feedback <type> <category>` — Record feedback (`too_cautious`, `missed`, `good_catch`, `expertise`)
- `set-expertise <domain> <level>` — Set domain expertise (1-5 or 0.0-1.0)
- `set-policy <action> <level>` — Set action policy (`always_play`/`lean_play`/`ask_user`/`always_ask`)
- `trust <path> <level>` — Set project trust (0.0-1.0)
- `reset-log` — Clear interaction log
- `train [--data-only] [--force] [--min-confidence F]` — Prepare training data and fine-tune pi_H model
- `predict --tool T [--command C] [--file-path F] [--project P]` — Test model on a hypothetical action
- `set-model <key> <value>` — Set model config (e.g., `enabled true`, `inference_timeout_ms 300`)

## Architecture

Core code lives in `oversight.py` with a procedural structure, plus two support modules for the pi_H model pipeline:

### oversight.py sections:
1. **Constants & defaults** (top of file): `DEFAULT_PROFILE`, `POLICY_LEVELS`, file paths pointing to `~/.claude/oversight/`
2. **Profile I/O**: `load_profile()` / `save_profile()` — JSON persistence with forward-compatible key merging
3. **Model inference**: `query_ollama_model()` — stdlib-only Ollama client for real-time gating decisions
4. **Policy mutation**: `shift_policy()` moves action policies up/down the 4-level hierarchy; `update_expertise()` adjusts domain scores via learning rate
5. **Generation pipeline**: `generate_claude_md()` produces natural-language instructions, `generate_settings()` produces allow/deny rules, `generate_hook_scripts(profile)` writes the PreToolUse/PostToolUse hook with baked-in policy + model gating
6. **Learning loop**: `analyze_session()` parses JSONL logs and aggregates tool usage stats/error rates; `apply_recommendations()` shifts policies based on approval rates
7. **Feedback system**: `record_feedback()` applies immediate policy/expertise shifts from user input
8. **CLI layer**: `cmd_*()` functions dispatched by `argparse` in `main()`

### prepare_training_data.py (stdlib only):
Converts `interaction_log.jsonl` + feedback into labeled training examples via priority cascade: explicit feedback > revert detection > outcome signal > heuristic policy echo.

### train_model.py (requires unsloth, trl, datasets):
LoRA fine-tunes Qwen2.5-1.5B-Instruct on training data, exports GGUF, imports into Ollama.

## Key Data Flow

```
profile.json → generate → CLAUDE.md + settings.json + hooks (with baked-in policy + model config)
                                                        ↓
                              Claude Code session (hooks log + gate via model for gray-area actions)
                                                        ↓
                          learn → analyze logs → update profile.json → regenerate
                                                        ↓
                          train → prepare data → LoRA fine-tune → GGUF → Ollama import
```

## Generated Files (output, not in repo)

All live under `~/.claude/`:
- `CLAUDE.md` — soft policy (natural language instructions for Claude Code)
- `settings.json` — hard policy (tool allow/deny rules, merged with existing settings)
- `oversight/profile.json` — learned profile state (includes model config)
- `oversight/interaction_log.jsonl` — raw interaction events (with gating_* fields when model active)
- `oversight/hooks/log_action.py` — hook script: logging + action classification + model gating
- `oversight/training/training_data.jsonl` — labeled training examples
- `oversight/training/model_output/` — LoRA adapter, GGUF, Modelfile

## Policy Model

Four levels per action category: `always_play` → `lean_play` → `ask_user` → `always_ask`. Fourteen domain expertise dimensions (0.0-1.0) gate autonomy. Per-project trust scores (0.0-1.0) further modulate policy. The learning loop approximates the paper's safe minimum-oversight equilibrium by shifting toward `play` when approval rates are high and toward `ask` when errors occur.

## Pi_H Model (Optional)

A LoRA fine-tuned Qwen2.5-1.5B served via Ollama that learns context-sensitive gating decisions. The heuristic system stays as fallback and guard rails — `always_play` and `always_ask` bypass the model entirely. The model is only consulted for `lean_play`/`ask_user` gray-area actions.

**Gating flow in hooks:**
1. Classify action → look up heuristic policy
2. `always_play` → allow (model never invoked)
3. `always_ask` → ask user (model never invoked)
4. `lean_play`/`ask_user` + model enabled → query Ollama → allow/deny/ask
5. `lean_play`/`ask_user` + model disabled → heuristic fallback

**Training flow:** `train --data-only` prepares data (stdlib only), full `train` requires GPU + `requirements-train.txt` deps.
