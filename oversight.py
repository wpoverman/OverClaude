#!/usr/bin/env python3
"""
The Oversight Game for Claude Code
===================================
A persistent oversight profile that learns how YOU work across all Claude Code sessions.

Based on "The Oversight Game: Learning to Cooperatively Balance an AI Agent's Safety 
and Autonomy" (Overman & Bayati, 2025).

The key idea: both the AI's play/ask policy (π_AI) and the human's trust/oversee 
policy (π_H) should be learned and persistent. This tool maintains your oversight 
profile across sessions, shaping both policies through:

  π_AI: Dynamic CLAUDE.md generation + settings.json permissions
  π_H: Hooks that log interactions + adaptive approval thresholds

The shared reward signal is implicit:
  - Safety violations (undo/revert needed) → large negative
  - Unnecessary asks (you always approve) → small negative (interaction cost)  
  - Smooth autonomous execution → positive
  - Caught a real problem via oversight → large positive
"""

import json
import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

# ─── paths ────────────────────────────────────────────────────────────────────
OVERSIGHT_DIR = Path.home() / ".claude" / "oversight"
PROFILE_PATH = OVERSIGHT_DIR / "profile.json"
LOG_PATH = OVERSIGHT_DIR / "interaction_log.jsonl"
CLAUDE_MD_PATH = Path.home() / ".claude" / "CLAUDE.md"
SETTINGS_PATH = Path.home() / ".claude" / "settings.json"
TRAINING_DIR = OVERSIGHT_DIR / "training"
TRAINING_DATA_PATH = TRAINING_DIR / "training_data.jsonl"
MODEL_OUTPUT_DIR = TRAINING_DIR / "model_output"

# ─── default profile ─────────────────────────────────────────────────────────
DEFAULT_PROFILE = {
    "version": 1,
    "created": None,
    "updated": None,

    # Domain expertise estimates: 0.0 (novice) → 1.0 (expert)
    # These shift π_AI toward play in high-expertise domains
    "domain_expertise": {
        "python": 0.5,
        "javascript": 0.5,
        "typescript": 0.5,
        "rust": 0.5,
        "go": 0.5,
        "shell_scripting": 0.5,
        "docker": 0.5,
        "kubernetes": 0.5,
        "databases": 0.5,
        "frontend": 0.5,
        "ml_ai": 0.5,
        "devops": 0.5,
        "security": 0.5,
        "networking": 0.5,
    },

    # Action-level policies: always_play | lean_play | ask_user | always_ask
    # This directly shapes the CLAUDE.md instructions
    "action_policies": {
        # File operations
        "file_read": "always_play",
        "file_edit_small": "lean_play",      # <50 lines changed
        "file_edit_large": "ask_user",        # >50 lines changed
        "file_create": "lean_play",
        "file_delete": "always_ask",
        "file_move_rename": "ask_user",

        # Shell commands
        "shell_safe": "always_play",          # ls, cat, grep, find, etc.
        "shell_test": "lean_play",            # pytest, npm test, etc.
        "shell_build": "lean_play",           # npm run build, make, etc.
        "shell_install": "ask_user",          # pip install, npm install
        "shell_destructive": "always_ask",    # rm, mv on non-temp
        "shell_network": "ask_user",          # curl, wget, ssh
        "shell_git_safe": "lean_play",        # git status, diff, log, add, commit
        "shell_git_dangerous": "always_ask",  # git push, rebase, force

        # Code generation
        "generate_boilerplate": "always_play",
        "generate_complex_logic": "lean_play",
        "refactor_small": "lean_play",
        "refactor_large": "ask_user",
        "architecture_change": "always_ask",
    },

    # Project-specific trust levels: path → trust_score (0-1)
    # Higher trust → more play, lower trust → more ask
    "project_trust": {},

    # Interaction cost parameters (from the paper's R_Φ)
    "cost_params": {
        "c_ask": 0.1,         # cost of unnecessary asking
        "c_oversee": 0.1,     # cost of unnecessary oversight
        "lambda_viol": 10.0,  # cost of a safety violation
    },

    # Learning rates for policy updates
    "learning": {
        "expertise_lr": 0.05,   # how fast domain expertise updates
        "policy_lr": 0.1,       # how fast action policies shift
        "trust_lr": 0.02,       # how fast project trust updates
    },

    # Aggregate statistics
    "stats": {
        "total_sessions": 0,
        "total_actions": 0,
        "total_asks": 0,
        "total_plays": 0,
        "total_violations": 0,  # times user had to undo/revert
        "total_unnecessary_asks": 0,  # times user said "just do it"
    },

    # Fine-tuned pi_H model configuration
    "model": {
        "enabled": False,
        "base_model": "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
        "ollama_model_name": "oversight-pi-h",
        "ollama_host": "http://localhost:11434",
        "max_seq_length": 1024,
        "lora_r": 16,
        "lora_alpha": 16,
        "quantization": "q4_k_m",
        "training_epochs": 3,
        "training_batch_size": 2,
        "min_training_examples": 50,
        "inference_timeout_ms": 500,
        "fallback_on_timeout": "heuristic",
        "fallback_on_error": "allow",
        "last_trained": None,
        "training_examples_count": 0,
        "version": 0,
    },
}

# ─── policy level ordering ───────────────────────────────────────────────────
POLICY_LEVELS = ["always_play", "lean_play", "ask_user", "always_ask"]


def ensure_dirs():
    OVERSIGHT_DIR.mkdir(parents=True, exist_ok=True)


def load_profile() -> dict:
    """Load profile or create default."""
    ensure_dirs()
    if PROFILE_PATH.exists():
        with open(PROFILE_PATH) as f:
            profile = json.load(f)
        # Merge any new default keys
        for section in ["domain_expertise", "action_policies", "cost_params", "learning", "stats", "model"]:
            if section in DEFAULT_PROFILE:
                for k, v in DEFAULT_PROFILE[section].items():
                    if k not in profile.get(section, {}):
                        profile.setdefault(section, {})[k] = v
        return profile
    else:
        profile = DEFAULT_PROFILE.copy()
        profile["created"] = datetime.now(timezone.utc).isoformat()
        profile["updated"] = profile["created"]
        save_profile(profile)
        return profile


def save_profile(profile: dict):
    ensure_dirs()
    profile["updated"] = datetime.now(timezone.utc).isoformat()
    with open(PROFILE_PATH, "w") as f:
        json.dump(profile, f, indent=2)


def log_interaction(entry: dict):
    """Append an interaction event to the log."""
    ensure_dirs()
    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ─── policy update logic ────────────────────────────────────────────────────
def shift_policy(current: str, direction: str, lr: float = 0.1) -> str:
    """
    Shift an action policy toward more play or more ask.
    
    direction: 'play' (user always approves → less asking needed)
               'ask'  (user had to intervene → more asking needed)
    """
    idx = POLICY_LEVELS.index(current)

    if direction == "play" and idx > 0:
        # With probability proportional to lr, shift toward play
        if hash(time.time()) % 100 < lr * 100:
            return POLICY_LEVELS[idx - 1]
    elif direction == "ask" and idx < len(POLICY_LEVELS) - 1:
        # Safety violations shift more aggressively
        return POLICY_LEVELS[idx + 1]

    return current


def update_expertise(profile: dict, domain: str, signal: float):
    """
    Update domain expertise based on interaction signal.
    signal > 0: user demonstrated expertise (approved complex actions, no issues)
    signal < 0: user needed help (asked questions, had to revert)
    """
    if domain in profile["domain_expertise"]:
        lr = profile["learning"]["expertise_lr"]
        current = profile["domain_expertise"][domain]
        profile["domain_expertise"][domain] = max(0.0, min(1.0, current + lr * signal))


# ─── model inference ─────────────────────────────────────────────────────────
def query_ollama_model(hook_input: dict, action_category: str, profile: dict) -> dict:
    """
    Query the fine-tuned Ollama model for a gating decision.
    Returns {"decision": str, "confidence": float, "reasoning": str,
             "latency_ms": int, "source": str}
    Uses only stdlib (urllib.request).
    """
    import urllib.request
    import urllib.error

    model_config = profile.get("model", {})
    ollama_host = model_config.get("ollama_host", "http://localhost:11434")
    model_name = model_config.get("ollama_model_name", "oversight-pi-h")
    timeout_s = model_config.get("inference_timeout_ms", 500) / 1000.0
    fallback_on_timeout = model_config.get("fallback_on_timeout", "heuristic")
    fallback_on_error = model_config.get("fallback_on_error", "allow")

    def make_fallback(reason: str, fallback_type: str) -> dict:
        if fallback_type == "heuristic":
            policy = profile.get("action_policies", {}).get(action_category, "ask_user")
            decision_map = {
                "always_play": "allow", "lean_play": "allow",
                "ask_user": "ask_user", "always_ask": "deny",
            }
            return {
                "decision": decision_map.get(policy, "ask_user"),
                "confidence": 0.5,
                "reasoning": f"Heuristic fallback ({reason}): policy={policy}",
                "latency_ms": 0,
                "source": "heuristic",
            }
        else:
            return {
                "decision": fallback_type,
                "confidence": 0.3,
                "reasoning": f"Fallback ({reason})",
                "latency_ms": 0,
                "source": "fallback",
            }

    # Build the prompt
    parts = [f"Tool: {hook_input.get('tool_name', 'unknown')}"]
    parts.append(f"Category: {action_category}")
    if hook_input.get("command_preview"):
        parts.append(f"Command: {hook_input['command_preview']}")
    if hook_input.get("file_path"):
        parts.append(f"File: {hook_input['file_path']}")

    project_dir = hook_input.get("project_dir", "")
    project_trust = profile.get("project_trust", {}).get(project_dir, 0.5)
    parts.append(f"Project trust: {project_trust:.1f}")

    heuristic_policy = profile.get("action_policies", {}).get(action_category, "ask_user")
    parts.append(f"Heuristic policy: {heuristic_policy}")

    expertise = profile.get("domain_expertise", {})
    if expertise:
        top = sorted(expertise.items(), key=lambda x: -x[1])[:5]
        exp_str = ", ".join(f"{k}: {v:.1f}" for k, v in top)
        parts.append(f"Domain expertise: {exp_str}")

    prompt_text = "\n".join(parts)

    payload = json.dumps({
        "model": model_name,
        "prompt": prompt_text,
        "system": (
            "You are an oversight policy model. Given a tool action context, decide "
            "whether to allow it autonomously, deny it, or ask the user for confirmation. "
            'Respond with a JSON object: {"decision": "allow"|"deny"|"ask_user", '
            '"confidence": 0.0-1.0, "reasoning": "brief explanation"}'
        ),
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
            "num_predict": 128,
        },
    }).encode("utf-8")

    start_ms = int(time.time() * 1000)

    try:
        req = urllib.request.Request(
            f"{ollama_host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError:
        return make_fallback("ollama_unreachable", fallback_on_error)
    except TimeoutError:
        return make_fallback("timeout", fallback_on_timeout)
    except Exception:
        return make_fallback("request_error", fallback_on_error)

    latency_ms = int(time.time() * 1000) - start_ms

    # Parse the model's response
    raw_response = body.get("response", "")
    try:
        result = json.loads(raw_response)
        decision = result.get("decision", "ask_user")
        if decision not in ("allow", "deny", "ask_user"):
            decision = "ask_user"
        confidence = float(result.get("confidence", 0.5))
        reasoning = str(result.get("reasoning", ""))
    except (json.JSONDecodeError, ValueError, TypeError):
        return {
            "decision": "ask_user",
            "confidence": 0.3,
            "reasoning": f"Unparseable model response: {raw_response[:100]}",
            "latency_ms": latency_ms,
            "source": "model",
        }

    return {
        "decision": decision,
        "confidence": confidence,
        "reasoning": reasoning,
        "latency_ms": latency_ms,
        "source": "model",
    }


# ─── CLAUDE.md generation ────────────────────────────────────────────────────
def generate_claude_md(profile: dict) -> str:
    """
    Generate the global CLAUDE.md that shapes π_AI.
    
    This is the core mechanism: we translate the learned profile into 
    natural language instructions that Claude Code reads at startup.
    """

    # Categorize actions by policy level
    play_actions = []
    lean_play_actions = []
    ask_actions = []
    always_ask_actions = []

    policy_descriptions = {
        # File operations
        "file_read": "Reading files",
        "file_edit_small": "Small file edits (<50 lines)",
        "file_edit_large": "Large file edits (>50 lines)",
        "file_create": "Creating new files",
        "file_delete": "Deleting files",
        "file_move_rename": "Moving or renaming files",
        # Shell
        "shell_safe": "Safe shell commands (ls, cat, grep, find, wc, head, tail, echo)",
        "shell_test": "Running tests (pytest, npm test, cargo test)",
        "shell_build": "Build commands (npm run build, make, cargo build)",
        "shell_install": "Installing packages (pip install, npm install)",
        "shell_destructive": "Destructive shell commands (rm on non-temp files, mv)",
        "shell_network": "Network commands (curl, wget, ssh, API calls)",
        "shell_git_safe": "Safe git operations (status, diff, log, add, commit to feature branches)",
        "shell_git_dangerous": "Dangerous git operations (push, rebase, force push, push to main/prod)",
        # Code generation
        "generate_boilerplate": "Generating boilerplate/scaffolding code",
        "generate_complex_logic": "Writing complex business logic or algorithms",
        "refactor_small": "Small refactors (renaming, extracting functions)",
        "refactor_large": "Large refactors (restructuring modules, changing interfaces)",
        "architecture_change": "Architecture changes (new dependencies, changing patterns, DB schema)",
    }

    for action, policy in profile["action_policies"].items():
        desc = policy_descriptions.get(action, action)
        if policy == "always_play":
            play_actions.append(desc)
        elif policy == "lean_play":
            lean_play_actions.append(desc)
        elif policy == "ask_user":
            ask_actions.append(desc)
        elif policy == "always_ask":
            always_ask_actions.append(desc)

    # Build expertise context
    expert_domains = []
    learning_domains = []
    for domain, level in sorted(profile["domain_expertise"].items(), key=lambda x: -x[1]):
        name = domain.replace("_", " ").title()
        if level >= 0.7:
            expert_domains.append(name)
        elif level <= 0.3:
            learning_domains.append(name)

    # Build project trust context
    trusted_projects = []
    cautious_projects = []
    for path, trust in profile.get("project_trust", {}).items():
        if trust >= 0.7:
            trusted_projects.append(path)
        elif trust <= 0.3:
            cautious_projects.append(path)

    # Generate the CLAUDE.md
    md = """# Oversight Policy
# Auto-generated by the Oversight Game framework — do not edit manually.
# Run `oversight generate` to regenerate, or `oversight edit` to modify the profile.
# Based on {sessions} sessions and {actions} interactions.

## About This Policy

This file configures when you (Claude) should act autonomously vs. ask for confirmation.
The goal is to minimize unnecessary interruptions while catching genuinely risky actions.
Think of this as a learned collaboration protocol — it reflects patterns from our past
interactions about what I'm comfortable with and where I need oversight.

""".format(
        sessions=profile["stats"]["total_sessions"],
        actions=profile["stats"]["total_actions"],
    )

    # Expertise section
    if expert_domains or learning_domains:
        md += "## My Expertise Profile\n\n"
        if expert_domains:
            md += "I'm experienced with: " + ", ".join(expert_domains) + ". "
            md += "You can be more autonomous and make judgment calls in these areas.\n\n"
        if learning_domains:
            md += "I'm less familiar with: " + ", ".join(learning_domains) + ". "
            md += "Please explain your reasoning and ask before making non-obvious choices here.\n\n"

    # Autonomous actions
    if play_actions:
        md += "## Act Autonomously (PLAY)\n\n"
        md += "Do these without asking — I trust your judgment here:\n\n"
        for a in play_actions:
            md += f"- {a}\n"
        md += "\n"

    if lean_play_actions:
        md += "## Default to Autonomous, Flag if Unsure (LEAN PLAY)\n\n"
        md += "Generally proceed without asking, but mention what you did. "
        md += "If something feels off or ambiguous, ask first:\n\n"
        for a in lean_play_actions:
            md += f"- {a}\n"
        md += "\n"

    if ask_actions:
        md += "## Ask Before Proceeding (ASK)\n\n"
        md += "Describe what you plan to do and wait for my approval:\n\n"
        for a in ask_actions:
            md += f"- {a}\n"
        md += "\n"

    if always_ask_actions:
        md += "## Always Ask — Never Proceed Autonomously (ALWAYS ASK)\n\n"
        md += "These actions require explicit approval every time, no exceptions:\n\n"
        for a in always_ask_actions:
            md += f"- {a}\n"
        md += "\n"

    # Project-specific
    if trusted_projects or cautious_projects:
        md += "## Project-Specific Notes\n\n"
        if trusted_projects:
            md += "High-trust projects (I know these codebases well, be more autonomous):\n"
            for p in trusted_projects:
                md += f"- `{p}`\n"
            md += "\n"
        if cautious_projects:
            md += "Low-trust projects (I'm less familiar, ask more often):\n"
            for p in cautious_projects:
                md += f"- `{p}`\n"
            md += "\n"

    # General principles
    md += """## General Principles

- When in doubt between acting and asking, consider the **reversibility** of the action. 
  Easily reversible actions (editing a file) lean toward autonomous; hard to reverse 
  actions (deleting data, pushing to prod) lean toward asking.
- If you're about to do something that touches multiple files or changes architecture, 
  outline your plan first and let me approve the approach before executing.
- For shell commands, prefer --dry-run flags when available for destructive operations.
- When I say "just do it" or approve something you asked about, take note — similar 
  actions in the future should lean more autonomous.
"""

    return md


# ─── settings.json generation ────────────────────────────────────────────────
def generate_settings(profile: dict) -> dict:
    """
    Generate the permissions section of settings.json based on profile.
    
    This provides HARD constraints on π_AI — things that are always 
    allowed or always denied regardless of the CLAUDE.md soft policy.
    """
    allow = []
    deny = []

    # Always allow safe reads and basic operations
    allow.extend([
        "Read(*)",
        "Bash(cat *)",
        "Bash(ls *)",
        "Bash(grep *)",
        "Bash(find *)",
        "Bash(wc *)",
        "Bash(head *)",
        "Bash(tail *)",
        "Bash(echo *)",
        "Bash(pwd)",
    ])

    # Always allow tests if policy says so
    if profile["action_policies"].get("shell_test") in ("always_play", "lean_play"):
        allow.extend([
            "Bash(pytest *)",
            "Bash(python -m pytest *)",
            "Bash(npm test *)",
            "Bash(npm run test *)",
            "Bash(cargo test *)",
            "Bash(go test *)",
        ])

    # Always allow git safe operations if policy says so
    if profile["action_policies"].get("shell_git_safe") in ("always_play", "lean_play"):
        allow.extend([
            "Bash(git status *)",
            "Bash(git diff *)",
            "Bash(git log *)",
            "Bash(git add *)",
            "Bash(git branch *)",
        ])

    # Always deny sensitive file reads
    deny.extend([
        "Read(./.env)",
        "Read(./.env.*)",
        "Read(./secrets/**)",
        "Read(**/.env)",
        "Read(**/.env.*)",
    ])

    # Deny destructive commands if policy requires
    if profile["action_policies"].get("shell_destructive") == "always_ask":
        deny.extend([
            "Bash(rm -rf *)",
        ])

    if profile["action_policies"].get("shell_git_dangerous") == "always_ask":
        deny.extend([
            "Bash(git push --force *)",
            "Bash(git push -f *)",
            "Bash(git rebase *)",
        ])

    return {"permissions": {"allow": allow, "deny": deny}}


# ─── hooks generation ────────────────────────────────────────────────────────
def generate_hooks_settings() -> dict:
    """
    Generate hooks that log interactions for the learning loop.
    
    These hooks are the observability layer — they capture what Claude does 
    so we can update the profile after each session.
    """
    log_script = str(OVERSIGHT_DIR / "hooks" / "log_action.py")

    return {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "*",
                    "hooks": [
                        {
                            "type": "command",
                            "command": f"python3 {log_script} pre"
                        }
                    ]
                }
            ],
            "PostToolUse": [
                {
                    "matcher": "*",
                    "hooks": [
                        {
                            "type": "command",
                            "command": f"python3 {log_script} post"
                        }
                    ]
                }
            ],
        }
    }


def generate_hook_scripts(profile: dict):
    """Create the hook scripts that log interactions and optionally gate actions via model."""
    hooks_dir = OVERSIGHT_DIR / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)

    # Bake profile data into the hook script as constants
    action_policies = json.dumps(profile.get("action_policies", {}))
    domain_expertise = json.dumps(profile.get("domain_expertise", {}))
    project_trust = json.dumps(profile.get("project_trust", {}))
    model_config = json.dumps(profile.get("model", {}))

    log_script = hooks_dir / "log_action.py"
    log_script.write_text('''#!/usr/bin/env python3
"""
Hook script for the Oversight Game.
Logs PreToolUse and PostToolUse events for later analysis.
On PreToolUse: classifies actions and optionally gates via fine-tuned model.
Reads JSON from stdin (passed by Claude Code hooks system).
"""
import json
import sys
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path.home() / ".claude" / "oversight" / "interaction_log.jsonl"

# ─── baked-in profile data ────────────────────────────────────────────────
ACTION_POLICIES = ''' + action_policies + '''
DOMAIN_EXPERTISE = ''' + domain_expertise + '''
PROJECT_TRUST = ''' + project_trust + '''
MODEL_CONFIG = ''' + model_config + '''

# ─── action classification (mirrored from prepare_training_data.py) ───────
BASH_PATTERNS = [
    ("shell_git_dangerous", [
        r"\\bgit\\s+(push|rebase|reset\\s+--hard|cherry-pick|merge|push\\s+--force|push\\s+-f)",
    ]),
    ("shell_git_safe", [
        r"\\bgit\\s+(status|diff|log|add|branch|stash|show|blame|tag|fetch|remote|config)",
        r"\\bgit\\s+commit\\b",
    ]),
    ("shell_destructive", [
        r"\\brm\\s+(-[rfi]+\\s+)*(?!/tmp)",
        r"\\bchmod\\s+",
        r"\\bchown\\s+",
    ]),
    ("shell_network", [
        r"\\bcurl\\s+", r"\\bwget\\s+", r"\\bssh\\s+", r"\\bscp\\s+", r"\\brsync\\s+",
    ]),
    ("shell_install", [
        r"\\bpip3?\\s+install\\b", r"\\bnpm\\s+install\\b", r"\\byarn\\s+add\\b",
        r"\\bbrew\\s+install\\b", r"\\bapt(-get)?\\s+install\\b",
        r"\\bcargo\\s+install\\b", r"\\bgo\\s+install\\b",
    ]),
    ("shell_test", [
        r"\\bpytest\\b", r"\\bpython3?\\s+-m\\s+pytest\\b",
        r"\\bnpm\\s+(run\\s+)?test\\b", r"\\bcargo\\s+test\\b", r"\\bgo\\s+test\\b",
    ]),
    ("shell_build", [
        r"\\bnpm\\s+run\\s+build\\b", r"\\bmake\\b", r"\\bcargo\\s+build\\b",
        r"\\bgo\\s+build\\b", r"\\btsc\\b",
    ]),
    ("shell_safe", [
        r"\\b(ls|cat|grep|find|wc|head|tail|echo|pwd|env|which|type|file|stat|du|df)\\b",
        r"\\bpython3?\\s+-c\\b", r"\\bnode\\s+-e\\b",
    ]),
]

TOOL_NAME_MAP = {
    "Read": "file_read", "Glob": "file_read", "Grep": "file_read",
    "Write": "file_create", "NotebookEdit": "file_edit_small",
}

def classify_action(tool_name, tool_input):
    if tool_name == "Bash":
        command = tool_input.get("command", "") or ""
        for category, patterns in BASH_PATTERNS:
            for pat in patterns:
                if re.search(pat, command):
                    return category
        return "shell_safe"
    if tool_name == "Edit":
        old_str = tool_input.get("old_string", "")
        new_str = tool_input.get("new_string", "")
        lines = max(
            len(old_str.split("\\n")) if old_str else 0,
            len(new_str.split("\\n")) if new_str else 0,
        )
        return "file_edit_large" if lines > 50 else "file_edit_small"
    if tool_name in TOOL_NAME_MAP:
        return TOOL_NAME_MAP[tool_name]
    if tool_name == "Task":
        return "generate_complex_logic"
    return "shell_safe"

def query_model(hook_input, action_category):
    """Query the Ollama model for a gating decision. Stdlib only."""
    import urllib.request
    import urllib.error

    ollama_host = MODEL_CONFIG.get("ollama_host", "http://localhost:11434")
    model_name = MODEL_CONFIG.get("ollama_model_name", "oversight-pi-h")
    timeout_s = MODEL_CONFIG.get("inference_timeout_ms", 500) / 1000.0
    fallback_on_timeout = MODEL_CONFIG.get("fallback_on_timeout", "heuristic")
    fallback_on_error = MODEL_CONFIG.get("fallback_on_error", "allow")

    def make_fallback(reason, fb_type):
        if fb_type == "heuristic":
            policy = ACTION_POLICIES.get(action_category, "ask_user")
            dm = {"always_play": "allow", "lean_play": "allow",
                  "ask_user": "ask_user", "always_ask": "deny"}
            return {"decision": dm.get(policy, "ask_user"), "confidence": 0.5,
                    "reasoning": "Heuristic fallback: " + reason, "latency_ms": 0, "source": "heuristic"}
        return {"decision": fb_type, "confidence": 0.3,
                "reasoning": "Fallback: " + reason, "latency_ms": 0, "source": "fallback"}

    parts = [f"Tool: {hook_input.get('tool_name', 'unknown')}"]
    parts.append(f"Category: {action_category}")
    if hook_input.get("command_preview"):
        parts.append(f"Command: {hook_input['command_preview']}")
    if hook_input.get("file_path"):
        parts.append(f"File: {hook_input['file_path']}")
    project_dir = hook_input.get("project_dir", "")
    pt = PROJECT_TRUST.get(project_dir, 0.5)
    parts.append(f"Project trust: {pt:.1f}")
    hp = ACTION_POLICIES.get(action_category, "ask_user")
    parts.append(f"Heuristic policy: {hp}")
    if DOMAIN_EXPERTISE:
        top = sorted(DOMAIN_EXPERTISE.items(), key=lambda x: -x[1])[:5]
        parts.append("Domain expertise: " + ", ".join(f"{k}: {v:.1f}" for k, v in top))

    payload = json.dumps({
        "model": model_name,
        "prompt": "\\n".join(parts),
        "system": (
            "You are an oversight policy model. Given a tool action context, decide "
            "whether to allow it autonomously, deny it, or ask the user for confirmation. "
            "Respond with a JSON object: {\\"decision\\": \\"allow\\"|\\"deny\\"|\\"ask_user\\", "
            "\\"confidence\\": 0.0-1.0, \\"reasoning\\": \\"brief explanation\\"}"
        ),
        "stream": False, "format": "json",
        "options": {"temperature": 0.1, "num_predict": 128},
    }).encode("utf-8")

    start_ms = int(time.time() * 1000)
    try:
        req = urllib.request.Request(
            f"{ollama_host}/api/generate", data=payload,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return make_fallback("request_error", fallback_on_error)

    latency_ms = int(time.time() * 1000) - start_ms
    raw = body.get("response", "")
    try:
        result = json.loads(raw)
        decision = result.get("decision", "ask_user")
        if decision not in ("allow", "deny", "ask_user"):
            decision = "ask_user"
        confidence = float(result.get("confidence", 0.5))
        reasoning = str(result.get("reasoning", ""))
    except (json.JSONDecodeError, ValueError, TypeError):
        return {"decision": "ask_user", "confidence": 0.3,
                "reasoning": f"Unparseable: {raw[:100]}", "latency_ms": latency_ms, "source": "model"}

    return {"decision": decision, "confidence": confidence, "reasoning": reasoning,
            "latency_ms": latency_ms, "source": "model"}


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "unknown"

    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        hook_input = {}

    tool_name = hook_input.get("tool_name", "unknown")
    tool_input = hook_input.get("tool_input", {})

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "tool_name": tool_name,
        "session_id": os.environ.get("CLAUDE_SESSION_ID", "unknown"),
        "project_dir": os.environ.get("CLAUDE_PROJECT_DIR", "unknown"),
    }

    # Extract relevant details
    if "command" in tool_input:
        entry["command_preview"] = str(tool_input["command"])[:200]
    if "file_path" in tool_input:
        entry["file_path"] = tool_input["file_path"]

    # For PostToolUse, check if it succeeded
    if phase == "post":
        tool_response = hook_input.get("tool_response", {})
        entry["exit_code"] = tool_response.get("exit_code")
        stdout = tool_response.get("stdout", "")
        stderr = tool_response.get("stderr", "")
        entry["stdout_len"] = len(stdout) if isinstance(stdout, str) else 0
        entry["stderr_len"] = len(stderr) if isinstance(stderr, str) else 0

    # ── PreToolUse gating ──────────────────────────────────────────────────
    gating_output = None
    if phase == "pre":
        action_category = classify_action(tool_name, tool_input)
        entry["action_category"] = action_category
        policy = ACTION_POLICIES.get(action_category, "ask_user")

        if policy == "always_play":
            # Clear-cut allow — no model needed
            pass
        elif policy == "always_ask":
            # Clear-cut ask — no model needed
            gating_output = {"permissionDecision": "ask",
                             "message": f"Policy requires confirmation for {action_category}"}
            entry["gating_decision"] = "ask"
            entry["gating_source"] = "policy"
            entry["gating_confidence"] = 1.0
        elif MODEL_CONFIG.get("enabled", False):
            # Gray area — query model
            model_result = query_model(entry, action_category)
            entry["gating_decision"] = model_result["decision"]
            entry["gating_source"] = model_result["source"]
            entry["gating_confidence"] = model_result["confidence"]
            entry["gating_latency_ms"] = model_result["latency_ms"]

            if model_result["decision"] == "deny":
                gating_output = {"permissionDecision": "deny",
                                 "message": f"Model denied: {model_result['reasoning']}"}
            elif model_result["decision"] == "ask_user":
                gating_output = {"permissionDecision": "ask",
                                 "message": f"Model suggests confirmation: {model_result['reasoning']}"}
            # "allow" → no output, action proceeds
        else:
            # Model disabled — heuristic fallback for gray area
            if policy == "ask_user":
                gating_output = {"permissionDecision": "ask",
                                 "message": f"Heuristic policy: ask for {action_category}"}
                entry["gating_decision"] = "ask"
                entry["gating_source"] = "heuristic"
                entry["gating_confidence"] = 0.5
            # lean_play with model disabled → allow (no output)

    # Log the event
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\\n")

    # Output gating decision if any
    if gating_output is not None:
        print(json.dumps(gating_output))

    sys.exit(0)

if __name__ == "__main__":
    main()
''')
    log_script.chmod(0o755)


# ─── session analysis ────────────────────────────────────────────────────────
def analyze_session(profile: dict) -> dict:
    """
    Analyze the interaction log and return update recommendations.
    
    This is the "reward signal" computation:
    - Actions that were always approved → shift toward play
    - Actions that needed correction → shift toward ask  
    - Domains where user was confident → increase expertise
    """
    if not LOG_PATH.exists():
        return {"recommendations": [], "stats": {}}

    # Parse log
    events = []
    with open(LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not events:
        return {"recommendations": [], "stats": {}}

    # Aggregate by tool and action type
    tool_counts = defaultdict(int)
    tool_errors = defaultdict(int)
    file_extensions = defaultdict(int)
    commands_run = []

    for e in events:
        tool = e.get("tool_name", "unknown")
        tool_counts[tool] += 1

        if e.get("phase") == "post" and e.get("exit_code", 0) != 0:
            tool_errors[tool] += 1

        if "file_path" in e:
            ext = Path(e["file_path"]).suffix
            if ext:
                file_extensions[ext] += 1

        if "command_preview" in e:
            commands_run.append(e["command_preview"])

    # Build recommendations
    recommendations = []

    # Check if tests are being run frequently → user is test-oriented
    test_commands = sum(1 for c in commands_run if any(t in c for t in ["pytest", "npm test", "cargo test"]))
    if test_commands > 5 and profile["action_policies"].get("shell_test") != "always_play":
        recommendations.append({
            "action": "shell_test",
            "current": profile["action_policies"].get("shell_test"),
            "suggested": "always_play",
            "reason": f"You ran {test_commands} test commands this session — auto-approve these?"
        })

    # Check which file types are being edited
    for ext, count in file_extensions.items():
        domain_map = {
            ".py": "python", ".js": "javascript", ".ts": "typescript",
            ".rs": "rust", ".go": "go", ".sh": "shell_scripting",
            ".jsx": "frontend", ".tsx": "frontend", ".css": "frontend",
            ".sql": "databases", ".dockerfile": "docker",
        }
        domain = domain_map.get(ext)
        if domain and count > 3:
            current_expertise = profile["domain_expertise"].get(domain, 0.5)
            error_rate = tool_errors.get("Edit", 0) / max(tool_counts.get("Edit", 1), 1)
            if error_rate < 0.1 and current_expertise < 0.8:
                recommendations.append({
                    "domain": domain,
                    "current_expertise": current_expertise,
                    "suggested_expertise": min(1.0, current_expertise + 0.1),
                    "reason": f"Edited {count} {ext} files with low error rate ({error_rate:.0%})"
                })

    stats = {
        "total_events": len(events),
        "unique_tools": dict(tool_counts),
        "error_count": sum(tool_errors.values()),
        "file_types": dict(file_extensions),
        "test_runs": test_commands,
    }

    return {"recommendations": recommendations, "stats": stats}


def apply_recommendations(profile: dict, recommendations: list) -> int:
    """Apply recommended updates to the profile. Returns number applied."""
    applied = 0
    for rec in recommendations:
        if "action" in rec and "suggested" in rec:
            profile["action_policies"][rec["action"]] = rec["suggested"]
            applied += 1
        elif "domain" in rec and "suggested_expertise" in rec:
            profile["domain_expertise"][rec["domain"]] = rec["suggested_expertise"]
            applied += 1
    return applied


# ─── manual feedback commands ────────────────────────────────────────────────
def record_feedback(profile: dict, feedback_type: str, context: str):
    """
    Record explicit user feedback to update policies.
    
    feedback_type:
      'too_cautious' — Claude asked when it shouldn't have
      'good_catch'   — Claude caught a real problem by asking
      'missed'       — Claude should have asked but didn't
      'expertise'    — User declares expertise in a domain
    """
    log_interaction({
        "type": "feedback",
        "feedback_type": feedback_type,
        "context": context,
    })

    if feedback_type == "too_cautious":
        profile["stats"]["total_unnecessary_asks"] += 1
        # Try to identify which policy to shift
        # (context might be like "shell_install" or "python")
        if context in profile["action_policies"]:
            old = profile["action_policies"][context]
            profile["action_policies"][context] = shift_policy(old, "play")
            print(f"  Shifted {context}: {old} → {profile['action_policies'][context]}")
        elif context in profile["domain_expertise"]:
            update_expertise(profile, context, +1.0)
            print(f"  Increased {context} expertise → {profile['domain_expertise'][context]:.2f}")

    elif feedback_type == "missed":
        profile["stats"]["total_violations"] += 1
        if context in profile["action_policies"]:
            old = profile["action_policies"][context]
            profile["action_policies"][context] = shift_policy(old, "ask")
            print(f"  Shifted {context}: {old} → {profile['action_policies'][context]}")

    elif feedback_type == "good_catch":
        # Reinforce current ask policy
        print(f"  Reinforced current oversight level for: {context}")

    elif feedback_type == "expertise":
        domain = context.lower().replace(" ", "_")
        if domain in profile["domain_expertise"]:
            profile["domain_expertise"][domain] = min(1.0, profile["domain_expertise"][domain] + 0.2)
            print(f"  Set {domain} expertise → {profile['domain_expertise'][domain]:.2f}")
        else:
            profile["domain_expertise"][domain] = 0.7
            print(f"  Added {domain} expertise → 0.7")

    save_profile(profile)


# ─── CLI ─────────────────────────────────────────────────────────────────────
def cmd_init(args):
    """Initialize the oversight profile."""
    profile = load_profile()
    print("Oversight Game profile initialized at:", PROFILE_PATH)
    print(f"  Domains: {len(profile['domain_expertise'])}")
    print(f"  Action policies: {len(profile['action_policies'])}")
    print()

    # Interactive expertise setup
    if args.interactive:
        print("Let's set your initial expertise levels.")
        print("Rate yourself 1-5 (1=novice, 5=expert) or press Enter to skip:\n")
        for domain in sorted(profile["domain_expertise"]):
            name = domain.replace("_", " ").title()
            try:
                val = input(f"  {name}: ").strip()
                if val:
                    level = int(val)
                    profile["domain_expertise"][domain] = max(0.0, min(1.0, level / 5.0))
            except (ValueError, EOFError):
                pass
        save_profile(profile)
        print("\nProfile updated.")

    # Generate CLAUDE.md
    cmd_generate(args)


def cmd_generate(args):
    """Generate CLAUDE.md and settings from current profile."""
    profile = load_profile()

    # Generate CLAUDE.md
    md_content = generate_claude_md(profile)

    # Check for existing CLAUDE.md and preserve user content
    existing_content = ""
    if CLAUDE_MD_PATH.exists():
        existing = CLAUDE_MD_PATH.read_text()
        # Look for user content after our generated section
        marker = "# --- End Oversight Policy ---"
        if marker in existing:
            idx = existing.index(marker) + len(marker)
            existing_content = existing[idx:]
        elif "# Oversight Policy" not in existing:
            # Existing file isn't ours — preserve it entirely
            existing_content = "\n\n" + existing

    full_md = md_content + "\n# --- End Oversight Policy ---\n" + existing_content

    CLAUDE_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLAUDE_MD_PATH.write_text(full_md)
    print(f"Generated: {CLAUDE_MD_PATH}")

    # Generate settings.json (merge with existing)
    settings = generate_settings(profile)
    if SETTINGS_PATH.exists():
        with open(SETTINGS_PATH) as f:
            existing_settings = json.load(f)
        # Merge permissions
        existing_perms = existing_settings.get("permissions", {})
        existing_allow = set(existing_perms.get("allow", []))
        existing_deny = set(existing_perms.get("deny", []))
        new_allow = set(settings["permissions"]["allow"])
        new_deny = set(settings["permissions"]["deny"])
        existing_settings["permissions"] = {
            "allow": sorted(existing_allow | new_allow),
            "deny": sorted(existing_deny | new_deny),
        }
        # Merge hooks
        hook_settings = generate_hooks_settings()
        existing_hooks = existing_settings.get("hooks", {})
        for event, hooks_list in hook_settings["hooks"].items():
            if event not in existing_hooks:
                existing_hooks[event] = hooks_list
            # Don't duplicate if already present
        existing_settings["hooks"] = existing_hooks
        settings = existing_settings
    else:
        hook_settings = generate_hooks_settings()
        settings["hooks"] = hook_settings["hooks"]

    with open(SETTINGS_PATH, "w") as f:
        json.dump(settings, f, indent=2)
    print(f"Generated: {SETTINGS_PATH}")

    # Generate hook scripts
    generate_hook_scripts(profile)
    print(f"Generated: {OVERSIGHT_DIR / 'hooks' / 'log_action.py'}")


def cmd_status(args):
    """Show current profile status."""
    profile = load_profile()

    print("═══════════════════════════════════════════════════")
    print("  The Oversight Game — Profile Status")
    print("═══════════════════════════════════════════════════\n")

    print(f"Sessions: {profile['stats']['total_sessions']}  "
          f"Actions: {profile['stats']['total_actions']}  "
          f"Violations: {profile['stats']['total_violations']}  "
          f"Unnecessary asks: {profile['stats']['total_unnecessary_asks']}\n")

    # Expertise
    print("Domain Expertise:")
    for domain, level in sorted(profile["domain_expertise"].items(), key=lambda x: -x[1]):
        bar = "█" * int(level * 20) + "░" * (20 - int(level * 20))
        name = domain.replace("_", " ").title()
        print(f"  {name:20s} [{bar}] {level:.2f}")

    # Policy summary
    print("\nAction Policies:")
    for policy_level in POLICY_LEVELS:
        actions = [k for k, v in profile["action_policies"].items() if v == policy_level]
        if actions:
            label = {
                "always_play": "🟢 ALWAYS PLAY",
                "lean_play": "🔵 LEAN PLAY",
                "ask_user": "🟡 ASK USER",
                "always_ask": "🔴 ALWAYS ASK",
            }[policy_level]
            print(f"\n  {label}:")
            for a in actions:
                print(f"    - {a}")

    # Project trust
    if profile.get("project_trust"):
        print("\nProject Trust Levels:")
        for path, trust in sorted(profile["project_trust"].items(), key=lambda x: -x[1]):
            bar = "█" * int(trust * 10) + "░" * (10 - int(trust * 10))
            print(f"  {path:40s} [{bar}] {trust:.2f}")

    # Model status
    model = profile.get("model", {})
    print("\nPi_H Model:")
    status = "ENABLED" if model.get("enabled") else "DISABLED"
    print(f"  Status: {status}")
    print(f"  Ollama model: {model.get('ollama_model_name', 'oversight-pi-h')}")
    print(f"  Version: {model.get('version', 0)}")
    if model.get("last_trained"):
        print(f"  Last trained: {model['last_trained']}")
    print(f"  Training examples: {model.get('training_examples_count', 0)}")
    print(f"  Inference timeout: {model.get('inference_timeout_ms', 500)}ms")
    print(f"  Fallback on timeout: {model.get('fallback_on_timeout', 'heuristic')}")
    print(f"  Fallback on error: {model.get('fallback_on_error', 'allow')}")


def cmd_learn(args):
    """Analyze recent interactions and update profile."""
    profile = load_profile()
    result = analyze_session(profile)

    print("═══════════════════════════════════════════════════")
    print("  Session Analysis")
    print("═══════════════════════════════════════════════════\n")

    stats = result["stats"]
    if not stats:
        print("No interaction data found. Use Claude Code with hooks enabled first.")
        return

    print(f"Total events: {stats['total_events']}")
    print(f"Error count: {stats['error_count']}")
    print(f"Test runs: {stats['test_runs']}")

    if stats.get("unique_tools"):
        print("\nTools used:")
        for tool, count in sorted(stats["unique_tools"].items(), key=lambda x: -x[1]):
            print(f"  {tool}: {count}")

    if stats.get("file_types"):
        print("\nFile types:")
        for ext, count in sorted(stats["file_types"].items(), key=lambda x: -x[1]):
            print(f"  {ext}: {count}")

    recs = result["recommendations"]
    if recs:
        print(f"\n{len(recs)} recommendations:\n")
        for i, rec in enumerate(recs):
            print(f"  {i+1}. {rec['reason']}")
            if "action" in rec:
                print(f"     {rec['action']}: {rec['current']} → {rec['suggested']}")
            elif "domain" in rec:
                print(f"     {rec['domain']}: {rec['current_expertise']:.2f} → {rec['suggested_expertise']:.2f}")

        if not args.dry_run:
            try:
                response = input("\nApply these recommendations? [y/N] ").strip().lower()
            except EOFError:
                response = "n"
            if response == "y":
                applied = apply_recommendations(profile, recs)
                profile["stats"]["total_sessions"] += 1
                save_profile(profile)
                print(f"Applied {applied} updates. Regenerating CLAUDE.md...")
                cmd_generate(argparse.Namespace())
            else:
                print("Skipped.")
        else:
            print("\n(Dry run — no changes applied)")
    else:
        print("\nNo policy updates recommended.")
        profile["stats"]["total_sessions"] += 1
        save_profile(profile)


def cmd_feedback(args):
    """Record explicit feedback about oversight behavior."""
    profile = load_profile()
    record_feedback(profile, args.type, args.context)
    print("Feedback recorded. Run 'oversight generate' to update CLAUDE.md.")


def cmd_set_expertise(args):
    """Manually set expertise for a domain."""
    profile = load_profile()
    domain = args.domain.lower().replace(" ", "_")
    level = max(0.0, min(1.0, args.level / 5.0)) if args.level <= 5 else max(0.0, min(1.0, args.level))

    if domain not in profile["domain_expertise"]:
        profile["domain_expertise"][domain] = level
        print(f"Added new domain '{domain}' with expertise {level:.2f}")
    else:
        old = profile["domain_expertise"][domain]
        profile["domain_expertise"][domain] = level
        print(f"Updated '{domain}': {old:.2f} → {level:.2f}")

    save_profile(profile)


def cmd_set_policy(args):
    """Manually set an action policy."""
    profile = load_profile()

    if args.action not in profile["action_policies"]:
        print(f"Unknown action: {args.action}")
        print(f"Available: {', '.join(sorted(profile['action_policies']))}")
        return

    if args.level not in POLICY_LEVELS:
        print(f"Unknown level: {args.level}")
        print(f"Available: {', '.join(POLICY_LEVELS)}")
        return

    old = profile["action_policies"][args.action]
    profile["action_policies"][args.action] = args.level
    save_profile(profile)
    print(f"Updated {args.action}: {old} → {args.level}")


def cmd_trust(args):
    """Set trust level for a project directory."""
    profile = load_profile()
    path = os.path.abspath(args.path)
    level = max(0.0, min(1.0, args.level))
    profile.setdefault("project_trust", {})[path] = level
    save_profile(profile)
    print(f"Set trust for '{path}' → {level:.2f}")


def cmd_reset_log(args):
    """Clear the interaction log."""
    if LOG_PATH.exists():
        LOG_PATH.unlink()
        print("Interaction log cleared.")
    else:
        print("No log file found.")


def cmd_train(args):
    """Orchestrate training data preparation and model fine-tuning."""
    profile = load_profile()
    min_confidence = getattr(args, "min_confidence", 0.4)

    print("═══════════════════════════════════════════════════")
    print("  Pi_H Model Training Pipeline")
    print("═══════════════════════════════════════════════════\n")

    # Step 1: Prepare training data (lightweight, no external deps)
    print("Step 1: Preparing training data...")
    from prepare_training_data import generate_training_data
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    count = generate_training_data(
        str(LOG_PATH), str(PROFILE_PATH), str(TRAINING_DATA_PATH),
        min_confidence=min_confidence,
    )
    print(f"  Generated {count} training examples → {TRAINING_DATA_PATH}")

    if count == 0:
        print("\nNo training data generated. Run some Claude Code sessions with hooks enabled first.")
        return

    # Check minimum examples
    min_examples = profile.get("model", {}).get("min_training_examples", 50)
    if count < min_examples and not getattr(args, "force", False):
        print(f"\nOnly {count} examples (minimum: {min_examples}). Use --force to train anyway.")
        return

    if getattr(args, "data_only", False):
        print("\n--data-only: stopping after data preparation.")
        return

    # Step 2: Import train_model (heavy deps)
    print("\nStep 2: Loading training framework...")
    try:
        from train_model import run_full_pipeline
    except ImportError as e:
        print(f"\n  Missing training dependencies: {e}")
        print("  Install with: pip install unsloth trl datasets")
        print("  Or: pip install -r requirements-train.txt")
        return

    # Step 3: Run full pipeline
    model_config = profile.get("model", {})
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result = run_full_pipeline(
        data_path=str(TRAINING_DATA_PATH),
        output_dir=str(MODEL_OUTPUT_DIR),
        base_model=model_config.get("base_model", "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"),
        max_seq_length=model_config.get("max_seq_length", 1024),
        lora_r=model_config.get("lora_r", 16),
        lora_alpha=model_config.get("lora_alpha", 16),
        quant_method=model_config.get("quantization", "q4_k_m"),
        epochs=model_config.get("training_epochs", 3),
        batch_size=model_config.get("training_batch_size", 2),
        ollama_model_name=model_config.get("ollama_model_name", "oversight-pi-h"),
    )

    # Update profile
    profile.setdefault("model", {})["enabled"] = True
    profile["model"]["last_trained"] = datetime.now(timezone.utc).isoformat()
    profile["model"]["training_examples_count"] = result["training_examples"]
    profile["model"]["version"] = profile["model"].get("version", 0) + 1
    save_profile(profile)

    print(f"\nModel v{profile['model']['version']} trained and imported to Ollama.")
    print("Run 'oversight generate' to update hooks with model gating.")


def cmd_predict(args):
    """Test the model on a hypothetical action."""
    profile = load_profile()

    tool_name = args.tool
    command = getattr(args, "command", None)
    file_path = getattr(args, "file_path", None)
    project = getattr(args, "project", None)

    # Build tool_input for classification
    tool_input = {}
    if command:
        tool_input["command"] = command
    if file_path:
        tool_input["file_path"] = file_path

    # Import classify_action from prepare_training_data
    from prepare_training_data import classify_action
    action_category = classify_action(tool_name, tool_input)

    # Get heuristic policy
    heuristic_policy = profile.get("action_policies", {}).get(action_category, "ask_user")

    print(f"Tool: {tool_name}")
    if command:
        print(f"Command: {command}")
    if file_path:
        print(f"File: {file_path}")
    print(f"Action category: {action_category}")
    print(f"Heuristic policy: {heuristic_policy}")
    print()

    # Check if clear-cut
    if heuristic_policy == "always_play":
        print("Decision: ALLOW (always_play — model not invoked)")
        return
    if heuristic_policy == "always_ask":
        print("Decision: ASK (always_ask — model not invoked)")
        return

    # Gray area — try model
    model_config = profile.get("model", {})
    if not model_config.get("enabled"):
        print("Model: DISABLED (using heuristic)")
        policy_map = {"lean_play": "allow", "ask_user": "ask_user"}
        print(f"Decision: {policy_map.get(heuristic_policy, 'ask_user').upper()}")
        return

    print("Querying model...")
    hook_input = {
        "tool_name": tool_name,
        "command_preview": command,
        "file_path": file_path,
        "project_dir": project or os.getcwd(),
    }
    result = query_ollama_model(hook_input, action_category, profile)

    print(f"Decision: {result['decision'].upper()}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Source: {result['source']}")
    print(f"Latency: {result['latency_ms']}ms")
    if result.get("reasoning"):
        print(f"Reasoning: {result['reasoning']}")


def cmd_set_model(args):
    """Set a model configuration parameter."""
    profile = load_profile()
    model = profile.setdefault("model", {})

    key = args.key
    value = args.value

    # Validate known keys with type checking
    bool_keys = {"enabled"}
    int_keys = {"max_seq_length", "lora_r", "lora_alpha", "training_epochs",
                "training_batch_size", "min_training_examples", "inference_timeout_ms",
                "version", "training_examples_count"}
    str_keys = {"base_model", "ollama_model_name", "ollama_host", "quantization",
                "fallback_on_timeout", "fallback_on_error", "last_trained"}

    if key not in bool_keys | int_keys | str_keys:
        print(f"Unknown model config key: {key}")
        print(f"Available keys: {', '.join(sorted(bool_keys | int_keys | str_keys))}")
        return

    if key in bool_keys:
        value = value.lower() in ("true", "1", "yes", "on")
    elif key in int_keys:
        try:
            value = int(value)
        except ValueError:
            print(f"Invalid integer value: {value}")
            return

    old = model.get(key)
    model[key] = value
    save_profile(profile)
    print(f"Updated model.{key}: {old} → {value}")


def main():
    parser = argparse.ArgumentParser(
        prog="oversight",
        description="The Oversight Game for Claude Code — learn your play/ask + trust/oversee policies",
    )
    sub = parser.add_subparsers(dest="command")

    # init
    p = sub.add_parser("init", help="Initialize oversight profile")
    p.add_argument("--interactive", "-i", action="store_true", help="Interactive expertise setup")
    p.set_defaults(func=cmd_init)

    # generate
    p = sub.add_parser("generate", help="Regenerate CLAUDE.md and settings.json")
    p.set_defaults(func=cmd_generate)

    # status
    p = sub.add_parser("status", help="Show current profile")
    p.set_defaults(func=cmd_status)

    # learn
    p = sub.add_parser("learn", help="Analyze interactions and update profile")
    p.add_argument("--dry-run", action="store_true", help="Show recommendations without applying")
    p.set_defaults(func=cmd_learn)

    # feedback
    p = sub.add_parser("feedback", help="Record explicit feedback")
    p.add_argument("type", choices=["too_cautious", "good_catch", "missed", "expertise"])
    p.add_argument("context", help="Action or domain (e.g., 'shell_install', 'python')")
    p.set_defaults(func=cmd_feedback)

    # set-expertise
    p = sub.add_parser("set-expertise", help="Set domain expertise")
    p.add_argument("domain", help="Domain name (e.g., python, docker)")
    p.add_argument("level", type=float, help="Level 1-5 or 0.0-1.0")
    p.set_defaults(func=cmd_set_expertise)

    # set-policy
    p = sub.add_parser("set-policy", help="Set action policy")
    p.add_argument("action", help="Action name (e.g., shell_install)")
    p.add_argument("level", choices=POLICY_LEVELS, help="Policy level")
    p.set_defaults(func=cmd_set_policy)

    # trust
    p = sub.add_parser("trust", help="Set project trust level")
    p.add_argument("path", help="Project directory path")
    p.add_argument("level", type=float, help="Trust level 0.0-1.0")
    p.set_defaults(func=cmd_trust)

    # reset-log
    p = sub.add_parser("reset-log", help="Clear interaction log")
    p.set_defaults(func=cmd_reset_log)

    # train
    p = sub.add_parser("train", help="Train the pi_H model from interaction logs")
    p.add_argument("--data-only", action="store_true", help="Only prepare training data, skip model training")
    p.add_argument("--force", action="store_true", help="Train even with fewer than min_training_examples")
    p.add_argument("--min-confidence", type=float, default=0.4, help="Minimum label confidence (default: 0.4)")
    p.set_defaults(func=cmd_train)

    # predict
    p = sub.add_parser("predict", help="Test model prediction on a hypothetical action")
    p.add_argument("--tool", required=True, help="Tool name (e.g., Bash, Edit, Read)")
    p.add_argument("--command", help="Command string (for Bash tool)")
    p.add_argument("--file-path", help="File path (for file operations)")
    p.add_argument("--project", help="Project directory (defaults to cwd)")
    p.set_defaults(func=cmd_predict)

    # set-model
    p = sub.add_parser("set-model", help="Set model configuration parameter")
    p.add_argument("key", help="Config key (e.g., enabled, ollama_host, inference_timeout_ms)")
    p.add_argument("value", help="Config value")
    p.set_defaults(func=cmd_set_model)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
