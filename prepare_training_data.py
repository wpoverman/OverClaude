#!/usr/bin/env python3
"""
Training Data Preparation for the Oversight Pi_H Model
=======================================================
Converts interaction_log.jsonl + feedback entries into labeled training examples
for LoRA fine-tuning. Zero external dependencies — stdlib only.
"""

import json
import re
import os
from pathlib import Path
from collections import defaultdict


# ─── action classification ───────────────────────────────────────────────────

# Regex patterns for classifying Bash commands into action categories
BASH_PATTERNS = [
    # Order matters: more specific patterns first
    ("shell_git_dangerous", [
        re.compile(r"\bgit\s+(push|rebase|reset\s+--hard|cherry-pick|merge|push\s+--force|push\s+-f)"),
    ]),
    ("shell_git_safe", [
        re.compile(r"\bgit\s+(status|diff|log|add|branch|stash|show|blame|tag|fetch|remote|config)"),
        re.compile(r"\bgit\s+commit\b"),
    ]),
    ("shell_destructive", [
        re.compile(r"\brm\s+(-[rfi]+\s+)*(?!/tmp)(?!.*\.(pyc|o|class|tmp))"),
        re.compile(r"\bmv\s+"),
        re.compile(r"\bchmod\s+"),
        re.compile(r"\bchown\s+"),
    ]),
    ("shell_network", [
        re.compile(r"\bcurl\s+"),
        re.compile(r"\bwget\s+"),
        re.compile(r"\bssh\s+"),
        re.compile(r"\bscp\s+"),
        re.compile(r"\brsync\s+"),
        re.compile(r"\bnc\s+"),
        re.compile(r"\bfetch\s+"),
    ]),
    ("shell_install", [
        re.compile(r"\bpip3?\s+install\b"),
        re.compile(r"\bnpm\s+install\b"),
        re.compile(r"\byarn\s+add\b"),
        re.compile(r"\bbrew\s+install\b"),
        re.compile(r"\bapt(-get)?\s+install\b"),
        re.compile(r"\bcargo\s+install\b"),
        re.compile(r"\bgo\s+install\b"),
    ]),
    ("shell_test", [
        re.compile(r"\bpytest\b"),
        re.compile(r"\bpython3?\s+-m\s+pytest\b"),
        re.compile(r"\bnpm\s+(run\s+)?test\b"),
        re.compile(r"\bcargo\s+test\b"),
        re.compile(r"\bgo\s+test\b"),
        re.compile(r"\bjest\b"),
        re.compile(r"\bmocha\b"),
    ]),
    ("shell_build", [
        re.compile(r"\bnpm\s+run\s+build\b"),
        re.compile(r"\bmake\b"),
        re.compile(r"\bcargo\s+build\b"),
        re.compile(r"\bgo\s+build\b"),
        re.compile(r"\bgcc\b"),
        re.compile(r"\bg\+\+\b"),
        re.compile(r"\btsc\b"),
    ]),
    ("shell_safe", [
        re.compile(r"\b(ls|cat|grep|find|wc|head|tail|echo|pwd|env|which|type|file|stat|du|df)\b"),
        re.compile(r"\bpython3?\s+-c\b"),
        re.compile(r"\bnode\s+-e\b"),
    ]),
]

TOOL_NAME_MAP = {
    "Read": "file_read",
    "Glob": "file_read",
    "Grep": "file_read",
    "Write": "file_create",
    "NotebookEdit": "file_edit_small",
}


def classify_action(tool_name: str, tool_input: dict = None) -> str:
    """
    Map raw hook data to one of the 21 action categories.
    Uses regex patterns for Bash commands, tool name for Read/Write/Edit/Glob/Grep.
    """
    if tool_input is None:
        tool_input = {}

    if tool_name == "Bash":
        command = tool_input.get("command", "") or ""
        for category, patterns in BASH_PATTERNS:
            for pat in patterns:
                if pat.search(command):
                    return category
        return "shell_safe"

    if tool_name == "Edit":
        # Distinguish small vs large edits by old_string length
        old_str = tool_input.get("old_string", "")
        new_str = tool_input.get("new_string", "")
        lines_changed = max(
            len(old_str.split("\n")) if old_str else 0,
            len(new_str.split("\n")) if new_str else 0,
        )
        return "file_edit_large" if lines_changed > 50 else "file_edit_small"

    if tool_name in TOOL_NAME_MAP:
        return TOOL_NAME_MAP[tool_name]

    if tool_name == "Task":
        return "generate_complex_logic"

    return "shell_safe"


# ─── event pairing ───────────────────────────────────────────────────────────

def pair_pre_post_events(events: list) -> list:
    """
    Match pre/post events by session_id + tool_name + timestamp proximity.
    Returns list of paired dicts: {"pre": event, "post": event|None}.
    """
    pre_events = []
    post_events = []

    for e in events:
        if e.get("phase") == "pre":
            pre_events.append(e)
        elif e.get("phase") == "post":
            post_events.append(e)

    paired = []
    used_post_indices = set()

    for pre in pre_events:
        best_post = None
        best_idx = -1
        best_delta = float("inf")

        for i, post in enumerate(post_events):
            if i in used_post_indices:
                continue
            if post.get("session_id") != pre.get("session_id"):
                continue
            if post.get("tool_name") != pre.get("tool_name"):
                continue

            # Timestamp proximity: post should be after pre and within 60s
            try:
                pre_ts = pre.get("timestamp", "")
                post_ts = post.get("timestamp", "")
                # Simple string comparison works for ISO timestamps
                if post_ts >= pre_ts:
                    delta = len(post_ts)  # rough proxy, just need ordering
                    if delta < best_delta:
                        best_delta = delta
                        best_post = post
                        best_idx = i
            except (TypeError, ValueError):
                continue

        if best_post is not None:
            used_post_indices.add(best_idx)

        paired.append({"pre": pre, "post": best_post})

    return paired


# ─── revert detection ────────────────────────────────────────────────────────

def detect_revert_patterns(events: list) -> set:
    """
    Find indices of actions that were undone.
    Looks for: git checkout/restore after edit, rm after create, git reset after commit.
    Returns set of event indices (into original events list) that were reverted.
    """
    reverted = set()

    # Build a timeline of file edits and creates
    file_actions = []  # (index, action_type, file_path, timestamp)
    for i, e in enumerate(events):
        if e.get("phase") != "pre":
            continue
        cmd = e.get("command_preview", "")
        fp = e.get("file_path", "")
        tool = e.get("tool_name", "")

        if tool in ("Edit", "Write"):
            file_actions.append((i, "modify", fp, e.get("timestamp", "")))
        elif tool == "Bash":
            # Check for reverts
            if re.search(r"\bgit\s+(checkout|restore)\s+--?\s*", cmd):
                # This is reverting something — find what file
                match = re.search(r"\bgit\s+(?:checkout|restore)\s+(?:--\s+)?(.+)", cmd)
                if match:
                    reverted_file = match.group(1).strip()
                    # Mark the most recent modify of this file as reverted
                    for j in range(len(file_actions) - 1, -1, -1):
                        idx, action, fpath, _ = file_actions[j]
                        if fpath and reverted_file in fpath:
                            reverted.add(idx)
                            break
            elif re.search(r"\bgit\s+reset\b", cmd):
                # Mark recent commits as reverted
                for j in range(len(file_actions) - 1, -1, -1):
                    idx, action, fpath, _ = file_actions[j]
                    reverted.add(idx)
                    break
            elif re.search(r"\brm\s+", cmd):
                # Check if removing a recently created file
                match = re.search(r"\brm\s+(?:-[rf]+\s+)?(.+)", cmd)
                if match:
                    removed = match.group(1).strip()
                    for j in range(len(file_actions) - 1, -1, -1):
                        idx, action, fpath, _ = file_actions[j]
                        if action == "modify" and fpath and removed in fpath:
                            reverted.add(idx)
                            break

    return reverted


# ─── label synthesis ─────────────────────────────────────────────────────────

def synthesize_label(paired_event: dict, profile: dict, feedback_entries: list,
                     reverted_indices: set, event_index: int) -> dict:
    """
    Generate a training label for a paired event using priority cascade:
    1. Explicit feedback (confidence 0.9)
    2. Revert detection (confidence 0.8)
    3. Outcome signal (confidence 0.5-0.7)
    4. Heuristic policy echo (confidence 0.4-0.6)
    """
    pre = paired_event["pre"]
    post = paired_event.get("post")
    tool_name = pre.get("tool_name", "unknown")
    cmd_preview = pre.get("command_preview", "")

    tool_input = {}
    if cmd_preview:
        tool_input["command"] = cmd_preview
    if pre.get("file_path"):
        tool_input["file_path"] = pre["file_path"]

    action_category = classify_action(tool_name, tool_input)

    # 1. Check explicit feedback
    for fb in feedback_entries:
        fb_context = fb.get("context", "")
        fb_type = fb.get("feedback_type", "")

        # Match feedback to this action by category or command content
        if fb_context == action_category or (fb_context and fb_context in cmd_preview):
            if fb_type == "too_cautious":
                return {
                    "label": "allow",
                    "confidence": 0.9,
                    "label_source": "feedback_too_cautious",
                }
            elif fb_type == "missed":
                return {
                    "label": "deny",
                    "confidence": 0.9,
                    "label_source": "feedback_missed",
                }
            elif fb_type == "good_catch":
                return {
                    "label": "ask_user",
                    "confidence": 0.9,
                    "label_source": "feedback_good_catch",
                }

    # 2. Revert detection
    if event_index in reverted_indices:
        return {
            "label": "deny",
            "confidence": 0.8,
            "label_source": "revert_detected",
        }

    # 3. Outcome signal from post event
    if post is not None:
        exit_code = post.get("exit_code")
        stderr_len = post.get("stderr_len", 0)

        if exit_code is not None:
            if exit_code == 0 and stderr_len < 100:
                return {
                    "label": "allow",
                    "confidence": 0.7,
                    "label_source": "outcome_success",
                }
            elif exit_code == 0 and stderr_len >= 100:
                return {
                    "label": "allow",
                    "confidence": 0.5,
                    "label_source": "outcome_success_with_warnings",
                }
            elif exit_code != 0:
                return {
                    "label": "ask_user",
                    "confidence": 0.6,
                    "label_source": "outcome_failure",
                }

    # 4. Heuristic policy echo (cold start)
    policy = profile.get("action_policies", {}).get(action_category, "ask_user")
    policy_to_label = {
        "always_play": "allow",
        "lean_play": "allow",
        "ask_user": "ask_user",
        "always_ask": "deny",
    }
    label = policy_to_label.get(policy, "ask_user")
    # Weaker confidence for more permissive heuristics
    conf = 0.6 if policy in ("always_ask", "ask_user") else 0.4

    return {
        "label": label,
        "confidence": conf,
        "label_source": "heuristic_policy",
    }


# ─── main pipeline ───────────────────────────────────────────────────────────

def generate_training_data(log_path: str, profile_path: str, output_path: str,
                           min_confidence: float = 0.4) -> int:
    """
    Main pipeline: read logs + profile, synthesize labels, write training JSONL.
    Returns number of training examples generated.
    """
    log_path = Path(log_path)
    profile_path = Path(profile_path)
    output_path = Path(output_path)

    # Load interaction log
    if not log_path.exists():
        print(f"  No interaction log found at {log_path}")
        return 0

    events = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not events:
        print("  No events found in log.")
        return 0

    # Load profile
    if profile_path.exists():
        with open(profile_path) as f:
            profile = json.load(f)
    else:
        profile = {"action_policies": {}, "domain_expertise": {}, "project_trust": {}}

    # Extract feedback entries
    feedback_entries = [e for e in events if e.get("type") == "feedback"]

    # Pair pre/post events
    paired = pair_pre_post_events(events)

    # Detect reverts
    reverted = detect_revert_patterns(events)

    # Synthesize labels
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(output_path, "w") as out:
        for i, pair in enumerate(paired):
            pre = pair["pre"]
            tool_name = pre.get("tool_name", "unknown")
            cmd_preview = pre.get("command_preview", "")
            file_path = pre.get("file_path")
            project_dir = pre.get("project_dir", "unknown")

            tool_input = {}
            if cmd_preview:
                tool_input["command"] = cmd_preview
            if file_path:
                tool_input["file_path"] = file_path

            action_category = classify_action(tool_name, tool_input)

            # Get project trust
            project_trust = profile.get("project_trust", {}).get(project_dir, 0.5)

            # Get heuristic policy for this category
            heuristic_policy = profile.get("action_policies", {}).get(
                action_category, "ask_user"
            )

            # Synthesize label
            label_info = synthesize_label(
                pair, profile, feedback_entries, reverted, i
            )

            if label_info["confidence"] < min_confidence:
                continue

            example = {
                "tool_name": tool_name,
                "action_category": action_category,
                "command_preview": cmd_preview or None,
                "file_path": file_path,
                "project_dir": project_dir,
                "project_trust": project_trust,
                "domain_expertise": profile.get("domain_expertise", {}),
                "heuristic_policy": heuristic_policy,
                "label": label_info["label"],
                "confidence": label_info["confidence"],
                "label_source": label_info["label_source"],
            }

            out.write(json.dumps(example) + "\n")
            count += 1

    return count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare training data from interaction logs")
    parser.add_argument("--log-path", default=str(Path.home() / ".claude" / "oversight" / "interaction_log.jsonl"))
    parser.add_argument("--profile-path", default=str(Path.home() / ".claude" / "oversight" / "profile.json"))
    parser.add_argument("--output-path", default=str(Path.home() / ".claude" / "oversight" / "training" / "training_data.jsonl"))
    parser.add_argument("--min-confidence", type=float, default=0.4)
    args = parser.parse_args()

    count = generate_training_data(args.log_path, args.profile_path, args.output_path, args.min_confidence)
    print(f"Generated {count} training examples → {args.output_path}")
