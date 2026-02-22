# The Oversight Game for Claude Code

**A persistent oversight profile that learns how YOU work across all Claude Code sessions.**

Based on *"The Oversight Game: Learning to Cooperatively Balance an AI Agent's Safety and Autonomy"* (Overman & Bayati, 2025).

## The Idea

Every time you start a new Claude Code session, it starts from scratch. It doesn't know that you're an expert in Python but less comfortable with Docker, that you always approve test runs but want to review git pushes, or that you trust your main project but want guardrails on new repos.

This tool maintains a **persistent oversight profile** that encodes two learned policies from the paper:

| Paper | Claude Code Implementation |
|-------|---------------------------|
| **π_AI** (agent's play/ask policy) | Dynamic `CLAUDE.md` + `settings.json` permissions |
| **π_H** (human's trust/oversee policy) | Hooks that log interactions + adaptive approval thresholds |
| **Shared reward R_Φ** | Implicit: task completion, unnecessary asks, safety violations |
| **Markov Potential Game structure** | Both policies co-evolve; more autonomy in safe regions doesn't hurt you |

The key insight from the paper's **Local Alignment Theorem**: when the AI shifts from *ask* to *play* in a domain where it benefits the AI (faster task completion), it cannot harm the human — provided the interaction has the right incentive structure. We approximate this by:

1. Only shifting toward *play* when the human consistently approves (positive reward signal)
2. Immediately shifting toward *ask* when things go wrong (safety violation)
3. Maintaining domain-specific expertise levels that gate autonomy

## How It Works

```
Session starts
    │
    ▼
oversight generate
    │  Reads ~/.claude/oversight/profile.json
    │  Writes ~/.claude/CLAUDE.md          ← shapes π_AI (soft: instructions)
    │  Writes ~/.claude/settings.json      ← shapes π_AI (hard: allow/deny)
    │  Writes hook scripts                 ← observability for learning
    │
    ▼
Claude Code runs with shaped policies
    │  Hooks log every tool use to interaction_log.jsonl
    │  CLAUDE.md tells Claude when to act vs. ask
    │  settings.json enforces hard boundaries
    │
    ▼
After session: oversight learn
    │  Analyzes interaction log
    │  Recommends policy updates
    │  Updates profile
    │  Regenerates CLAUDE.md
    │
    ▼
Next session starts with updated policies
```

## Installation

```bash
# Clone or copy to your preferred location
cp -r oversight-game-cc ~/oversight-game-cc

# Make the CLI available (add to your .bashrc/.zshrc)
alias oversight='python3 ~/oversight-game-cc/oversight.py'

# Initialize your profile (interactive mode asks about your expertise)
oversight init --interactive

# This generates:
#   ~/.claude/CLAUDE.md           — instructions for Claude
#   ~/.claude/settings.json       — permission rules
#   ~/.claude/oversight/           — profile + logs + hooks
```

## Usage

### Day-to-day workflow

```bash
# Before starting work (or add to your shell startup):
oversight generate

# Start Claude Code normally — it reads the generated CLAUDE.md
claude

# After your session, analyze what happened:
oversight learn

# Quick feedback during/after a session:
oversight feedback too_cautious shell_install   # "stop asking about pip install"
oversight feedback missed shell_destructive     # "you should have asked before that rm"
oversight feedback good_catch architecture_change  # "glad you asked about that"
oversight feedback expertise rust               # "I know Rust well"
```

### Managing your profile

```bash
# Check your current profile
oversight status

# Set expertise levels (1-5 scale)
oversight set-expertise python 5
oversight set-expertise docker 2
oversight set-expertise kubernetes 1

# Set action policies directly
oversight set-policy shell_install always_play      # stop asking about installs
oversight set-policy file_delete always_ask          # always ask before deleting
oversight set-policy shell_git_dangerous always_ask  # never auto-push

# Set project trust levels (0.0-1.0)
oversight trust ~/my-project 0.9      # I know this codebase
oversight trust ~/new-client-repo 0.2  # be careful here

# Clear logs for a fresh start
oversight reset-log
```

### Policy Levels

Each action category has one of four policy levels:

| Level | Meaning | CLAUDE.md Instruction |
|-------|---------|----------------------|
| `always_play` | Always act autonomously | "Do this without asking" |
| `lean_play` | Default autonomous, flag if unsure | "Generally proceed, mention what you did" |
| `ask_user` | Ask before proceeding | "Describe your plan and wait for approval" |
| `always_ask` | Never proceed without approval | "Always ask, no exceptions" |

### The Learning Loop

The `oversight learn` command analyzes your interaction logs and recommends updates:

- **Actions you always approve** → shift toward `play` (reduce interaction cost c_ask)
- **Actions that caused errors** → shift toward `ask` (increase oversight for safety)
- **Domains with low error rates** → increase expertise estimate
- **Frequently used tools** → optimize permission settings

This mirrors the paper's **independent policy gradient**: both π_AI and π_H update based on the shared reward signal, converging toward an equilibrium that balances safety and autonomy.

## Architecture

```
~/.claude/
├── CLAUDE.md                    # Generated: soft π_AI policy
├── settings.json                # Generated: hard π_AI constraints + hooks
└── oversight/
    ├── profile.json             # Your learned profile (π_AI + π_H state)
    ├── interaction_log.jsonl    # Raw interaction data (reward signals)
    └── hooks/
        └── log_action.py        # Hook script for PreToolUse/PostToolUse
```

## Theoretical Connection

The paper proves two key results that motivate this design:

**Theorem 1 (Local Alignment):** Under the Markov Potential Game structure, any increase in the agent's autonomy that benefits the agent cannot decrease the human's value. In our implementation: when we shift a policy from `ask_user` to `lean_play` because you consistently approve those actions, the resulting time savings (less interruption) benefits you too.

**Theorem 2 (Safe Minimum-Oversight Equilibrium):** There exists a safe joint policy that minimizes oversight among all safe policies. Our `learn` command approximates this by finding the least-oversight policy that hasn't caused safety violations — reducing asking wherever your approval rate is high while maintaining oversight where errors have occurred.

The **ask-burden assumption** maps directly to the interaction cost `c_ask`: deferral (asking) costs time and attention, so the agent prefers to play when safe. The **violation penalty** λ_viol maps to the consequence of errors — undo/revert operations that cost much more than a brief confirmation dialog.

## Limitations & Future Work

- **No within-session adaptation of π_AI**: The CLAUDE.md is loaded at session start. Mid-session policy changes would require rewriting the file and using `/compact` or `/clear`.
- **Prompt-based π_AI is approximate**: We can't do true gradient descent on Claude's policy — we're shaping it through natural language instructions, which is a noisy channel.
- **The "human policy" is mostly implicit**: The tool currently focuses on shaping π_AI. A fuller implementation would also have configurable auto-approval rules (true π_H adaptation).
- **Log analysis is basic**: A production version could use an LLM to analyze session transcripts for richer reward signals.

## License

MIT
