#!/usr/bin/env python3
"""
OverClaude Dashboard — localhost web UI for oversight visibility.

Serves an embedded SPA that shows oversight stats, lets users review
automated actions, and feed retroactive corrections back into the profile.

Usage:
    python3 dashboard.py [--port 7483] [--no-browser]
    # or via: python3 oversight.py dashboard [--port 7483] [--no-browser]
"""

import json
import os
import sys
import time
import webbrowser
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

from oversight import (
    load_profile, save_profile, record_feedback, shift_policy,
    generate_claude_md, generate_settings, generate_hooks_settings,
    generate_hook_scripts, PROFILE_PATH, LOG_PATH, CLAUDE_MD_PATH,
    SETTINGS_PATH, POLICY_LEVELS, OVERSIGHT_DIR,
)

# ─── event reading with mtime cache ─────────────────────────────────────────

_event_cache = {"mtime": 0, "events": []}


def read_events():
    """Read and cache parsed interaction log, refreshing on file change."""
    if not LOG_PATH.exists():
        return []
    mtime = LOG_PATH.stat().st_mtime
    if mtime == _event_cache["mtime"]:
        return _event_cache["events"]
    events = []
    with open(LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    _event_cache["mtime"] = mtime
    _event_cache["events"] = events
    return events


def summarize_events():
    """Aggregate counts for the overview tab."""
    events = read_events()
    total = 0
    automated = 0
    asked = 0
    model_decisions = 0
    by_category = defaultdict(int)
    by_hour = defaultdict(int)
    by_source = defaultdict(int)
    projects = set()

    for e in events:
        if e.get("phase") != "pre":
            continue
        total += 1
        cat = e.get("action_category", "unknown")
        by_category[cat] += 1

        gd = e.get("gating_decision")
        gs = e.get("gating_source", "none")
        if gd == "ask" or gd == "ask_user":
            asked += 1
        else:
            automated += 1
        if gs == "model":
            model_decisions += 1
        by_source[gs] += 1

        ts = e.get("timestamp", "")
        if len(ts) >= 13:
            try:
                hour = ts[11:13]
                by_hour[hour] += 1
            except (IndexError, ValueError):
                pass

        proj = e.get("project_dir")
        if proj and proj != "unknown":
            projects.add(proj)

    return {
        "total": total,
        "automated": automated,
        "asked": asked,
        "model_decisions": model_decisions,
        "by_category": dict(by_category),
        "by_hour": dict(by_hour),
        "by_source": dict(by_source),
        "projects": sorted(projects),
    }


# ─── API handlers ────────────────────────────────────────────────────────────

def handle_get_profile():
    return 200, load_profile()


def handle_get_events(params):
    events = read_events()
    offset = int(params.get("offset", ["0"])[0])
    limit = int(params.get("limit", ["100"])[0])
    phase_filter = params.get("phase", [None])[0]
    category_filter = params.get("category", [None])[0]
    source_filter = params.get("source", [None])[0]
    project_filter = params.get("project", [None])[0]

    filtered = []
    for e in events:
        if phase_filter and e.get("phase") != phase_filter:
            continue
        if category_filter and e.get("action_category") != category_filter:
            continue
        if source_filter and e.get("gating_source") != source_filter:
            continue
        if project_filter and e.get("project_dir") != project_filter:
            continue
        filtered.append(e)

    # Reverse so newest first
    filtered.reverse()
    page = filtered[offset:offset + limit]
    return 200, {"total": len(filtered), "offset": offset, "limit": limit, "events": page}


def handle_get_summary():
    return 200, summarize_events()


def handle_post_feedback(body):
    profile = load_profile()
    feedback_type = body.get("feedback_type")
    context = body.get("context", "")
    if feedback_type not in ("too_cautious", "good_catch", "missed", "expertise"):
        return 400, {"error": "Invalid feedback_type"}
    record_feedback(profile, feedback_type, context)
    return 200, {"ok": True, "feedback_type": feedback_type, "context": context}


def handle_post_review(body):
    """Log a dashboard review event (fine / should have asked / dangerous)."""
    profile = load_profile()
    verdict = body.get("verdict", "fine")
    category = body.get("action_category", "")
    event_ts = body.get("event_timestamp", "")

    entry = {
        "type": "dashboard_review",
        "verdict": verdict,
        "action_category": category,
        "reviewed_event_timestamp": event_ts,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Log it
    from oversight import log_interaction
    log_interaction(entry)

    if verdict == "should_have_asked" and category:
        record_feedback(profile, "missed", category)
        return 200, {"ok": True, "shifted": True, "category": category}
    elif verdict == "dangerous" and category:
        profile["stats"]["total_violations"] = profile["stats"].get("total_violations", 0) + 1
        record_feedback(profile, "missed", category)
        return 200, {"ok": True, "shifted": True, "dangerous": True, "category": category}
    return 200, {"ok": True, "shifted": False}


def handle_post_policy(body):
    profile = load_profile()
    action = body.get("action", "")
    level = body.get("level", "")
    if action not in profile["action_policies"]:
        return 400, {"error": f"Unknown action: {action}"}
    if level not in POLICY_LEVELS:
        return 400, {"error": f"Unknown level: {level}"}
    old = profile["action_policies"][action]
    profile["action_policies"][action] = level
    save_profile(profile)
    return 200, {"ok": True, "action": action, "old": old, "new": level}


def handle_post_expertise(body):
    profile = load_profile()
    domain = body.get("domain", "")
    level = body.get("level")
    if domain not in profile["domain_expertise"]:
        return 400, {"error": f"Unknown domain: {domain}"}
    try:
        level = float(level)
    except (TypeError, ValueError):
        return 400, {"error": "level must be a number"}
    level = max(0.0, min(1.0, level))
    old = profile["domain_expertise"][domain]
    profile["domain_expertise"][domain] = level
    save_profile(profile)
    return 200, {"ok": True, "domain": domain, "old": old, "new": level}


def handle_post_trust(body):
    profile = load_profile()
    path = body.get("path", "")
    level = body.get("level")
    if not path:
        return 400, {"error": "path is required"}
    try:
        level = float(level)
    except (TypeError, ValueError):
        return 400, {"error": "level must be a number"}
    level = max(0.0, min(1.0, level))
    profile.setdefault("project_trust", {})[path] = level
    save_profile(profile)
    return 200, {"ok": True, "path": path, "level": level}


def handle_post_regenerate():
    profile = load_profile()
    md_content = generate_claude_md(profile)
    existing_content = ""
    if CLAUDE_MD_PATH.exists():
        existing = CLAUDE_MD_PATH.read_text()
        marker = "# --- End Oversight Policy ---"
        if marker in existing:
            idx = existing.index(marker) + len(marker)
            existing_content = existing[idx:]
        elif "# Oversight Policy" not in existing:
            existing_content = "\n\n" + existing
    full_md = md_content + "\n# --- End Oversight Policy ---\n" + existing_content
    CLAUDE_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLAUDE_MD_PATH.write_text(full_md)

    settings = generate_settings(profile)
    if SETTINGS_PATH.exists():
        with open(SETTINGS_PATH) as f:
            existing_settings = json.load(f)
        existing_perms = existing_settings.get("permissions", {})
        existing_allow = set(existing_perms.get("allow", []))
        existing_deny = set(existing_perms.get("deny", []))
        new_allow = set(settings["permissions"]["allow"])
        new_deny = set(settings["permissions"]["deny"])
        existing_settings["permissions"] = {
            "allow": sorted(existing_allow | new_allow),
            "deny": sorted(existing_deny | new_deny),
        }
        hook_settings = generate_hooks_settings()
        existing_hooks = existing_settings.get("hooks", {})
        for event, hooks_list in hook_settings["hooks"].items():
            if event not in existing_hooks:
                existing_hooks[event] = hooks_list
        existing_settings["hooks"] = existing_hooks
        settings = existing_settings
    else:
        hook_settings = generate_hooks_settings()
        settings["hooks"] = hook_settings["hooks"]
    with open(SETTINGS_PATH, "w") as f:
        json.dump(settings, f, indent=2)

    generate_hook_scripts(profile)
    return 200, {"ok": True, "files": [str(CLAUDE_MD_PATH), str(SETTINGS_PATH)]}


# ─── route table ─────────────────────────────────────────────────────────────

GET_ROUTES = {
    "/api/profile": lambda p: handle_get_profile(),
    "/api/events": handle_get_events,
    "/api/events/summary": lambda p: handle_get_summary(),
}

POST_ROUTES = {
    "/api/feedback": handle_post_feedback,
    "/api/review": handle_post_review,
    "/api/policy": handle_post_policy,
    "/api/expertise": handle_post_expertise,
    "/api/trust": handle_post_trust,
    "/api/regenerate": lambda b: handle_post_regenerate(),
}


# ─── HTTP handler ─────────────────────────────────────────────────────────────

class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # silence default logging

    def _send_json(self, status, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        params = parse_qs(parsed.query)

        if path == "/" or path == "":
            self._send_html(DASHBOARD_HTML)
            return

        # Match longest prefix first for routes like /api/events/summary
        for route_path in sorted(GET_ROUTES.keys(), key=len, reverse=True):
            if path == route_path:
                handler = GET_ROUTES[route_path]
                try:
                    status, data = handler(params)
                except TypeError:
                    status, data = handler()
                self._send_json(status, data)
                return

        self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        content_length = int(self.headers.get("Content-Length", 0))
        body = {}
        if content_length > 0:
            raw = self.rfile.read(content_length)
            try:
                body = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                self._send_json(400, {"error": "Invalid JSON"})
                return

        handler = POST_ROUTES.get(path)
        if handler:
            status, data = handler(body)
            self._send_json(status, data)
        else:
            self.send_error(404)


# ─── embedded SPA ─────────────────────────────────────────────────────────────

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OverClaude Dashboard</title>
<style>
:root {
  --bg: #0d1117; --surface: #161b22; --surface2: #21262d; --border: #30363d;
  --text: #e6edf3; --text2: #8b949e; --accent: #58a6ff; --green: #3fb950;
  --yellow: #d29922; --red: #f85149; --purple: #bc8cff;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  font-size: 14px; color: var(--text); background: var(--bg);
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { min-height: 100vh; }
a { color: var(--accent); text-decoration: none; }
/* nav */
.nav { display: flex; align-items: center; gap: 24px; padding: 12px 24px;
  background: var(--surface); border-bottom: 1px solid var(--border); position: sticky; top: 0; z-index: 10; }
.nav h1 { font-size: 16px; font-weight: 600; margin-right: auto; }
.nav button { background: none; border: 1px solid var(--border); color: var(--text2);
  padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 13px; }
.nav button:hover, .nav button.active { color: var(--text); border-color: var(--accent); background: var(--surface2); }
/* layout */
.container { max-width: 1200px; margin: 0 auto; padding: 24px; }
.tab-content { display: none; } .tab-content.active { display: block; }
/* cards */
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 20px; }
.card .label { font-size: 12px; color: var(--text2); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
.card .value { font-size: 28px; font-weight: 600; }
.card .sub { font-size: 12px; color: var(--text2); margin-top: 4px; }
/* sections */
.section { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 20px; margin-bottom: 20px; }
.section h2 { font-size: 15px; font-weight: 600; margin-bottom: 16px; color: var(--text2); }
/* table */
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); color: var(--text2);
  font-weight: 500; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }
td { padding: 8px 10px; border-bottom: 1px solid var(--border); }
tr:hover { background: var(--surface2); }
/* badges */
.badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 500; }
.badge-green { background: rgba(63,185,80,0.15); color: var(--green); }
.badge-yellow { background: rgba(210,153,34,0.15); color: var(--yellow); }
.badge-red { background: rgba(248,81,73,0.15); color: var(--red); }
.badge-blue { background: rgba(88,166,255,0.15); color: var(--accent); }
.badge-purple { background: rgba(188,140,255,0.15); color: var(--purple); }
/* action buttons */
.action-btn { border: none; padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 12px; margin-right: 4px; }
.btn-fine { background: rgba(63,185,80,0.15); color: var(--green); }
.btn-ask { background: rgba(210,153,34,0.15); color: var(--yellow); }
.btn-danger { background: rgba(248,81,73,0.15); color: var(--red); }
.action-btn:hover { filter: brightness(1.3); }
/* filters */
.filters { display: flex; gap: 10px; margin-bottom: 16px; flex-wrap: wrap; align-items: center; }
.filters select, .filters input { background: var(--surface2); border: 1px solid var(--border);
  color: var(--text); padding: 6px 10px; border-radius: 6px; font-size: 13px; }
/* pagination */
.pagination { display: flex; justify-content: center; gap: 8px; margin-top: 16px; }
.pagination button { background: var(--surface2); border: 1px solid var(--border);
  color: var(--text); padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 13px; }
.pagination button:disabled { opacity: 0.4; cursor: default; }
.pagination button:not(:disabled):hover { border-color: var(--accent); }
/* slider */
.slider-row { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
.slider-row label { width: 140px; font-size: 13px; color: var(--text2); }
.slider-row input[type=range] { flex: 1; accent-color: var(--accent); }
.slider-row .val { width: 40px; text-align: right; font-size: 13px; font-variant-numeric: tabular-nums; }
/* select row */
.policy-row { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }
.policy-row label { width: 200px; font-size: 13px; color: var(--text2); }
.policy-row select { background: var(--surface2); border: 1px solid var(--border);
  color: var(--text); padding: 4px 8px; border-radius: 6px; font-size: 13px; }
/* trust row */
.trust-row { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }
.trust-row .path { flex: 1; font-size: 13px; font-family: monospace; color: var(--text2); overflow: hidden; text-overflow: ellipsis; }
.trust-row input[type=range] { width: 200px; accent-color: var(--accent); }
.trust-row .val { width: 40px; text-align: right; font-size: 13px; }
/* toast */
.toast { position: fixed; bottom: 24px; right: 24px; background: var(--surface2); border: 1px solid var(--border);
  padding: 12px 20px; border-radius: 8px; font-size: 13px; z-index: 100; opacity: 0; transition: opacity 0.3s; pointer-events: none; }
.toast.show { opacity: 1; }
/* bars */
.bar-chart { display: flex; align-items: flex-end; gap: 2px; height: 80px; margin-top: 8px; }
.bar-chart .bar { flex: 1; background: var(--accent); border-radius: 2px 2px 0 0; min-height: 2px;
  position: relative; transition: height 0.3s; }
.bar-chart .bar:hover { background: var(--purple); }
.bar-labels { display: flex; gap: 2px; font-size: 10px; color: var(--text2); }
.bar-labels span { flex: 1; text-align: center; }
/* model widget */
.model-status { display: flex; align-items: center; gap: 10px; }
.dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.dot-green { background: var(--green); } .dot-red { background: var(--red); }
/* regen button */
.btn-primary { background: var(--accent); color: #fff; border: none; padding: 8px 20px;
  border-radius: 6px; cursor: pointer; font-size: 14px; font-weight: 500; }
.btn-primary:hover { filter: brightness(1.15); }
.btn-primary:disabled { opacity: 0.5; cursor: default; }
/* reviewed */
.reviewed { opacity: 0.45; }
</style>
</head>
<body>
<div class="nav">
  <h1>OverClaude</h1>
  <button class="active" onclick="switchTab('overview')">Overview</button>
  <button onclick="switchTab('review')">Review Queue</button>
  <button onclick="switchTab('profile')">Profile</button>
</div>
<div class="container">
  <!-- Overview Tab -->
  <div id="tab-overview" class="tab-content active">
    <div id="overview-cards" class="cards"></div>
    <div class="section"><h2>Activity by Hour</h2><div id="activity-chart"></div></div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px">
      <div class="section"><h2>Policy Distribution</h2><div id="policy-dist"></div></div>
      <div class="section"><h2>Model Status</h2><div id="model-widget"></div></div>
    </div>
  </div>

  <!-- Review Queue Tab -->
  <div id="tab-review" class="tab-content">
    <div class="filters">
      <select id="filter-category"><option value="">All categories</option></select>
      <select id="filter-source"><option value="">All sources</option></select>
      <select id="filter-project"><option value="">All projects</option></select>
    </div>
    <div class="section" style="padding:0;overflow:auto">
      <table>
        <thead><tr>
          <th>Time</th><th>Tool</th><th>Command / File</th><th>Category</th>
          <th>Decision</th><th>Source</th><th>Conf</th><th>Actions</th>
        </tr></thead>
        <tbody id="review-body"></tbody>
      </table>
    </div>
    <div class="pagination">
      <button id="pg-prev" onclick="reviewPage(-1)">Prev</button>
      <span id="pg-info" style="line-height:32px;color:var(--text2);font-size:13px"></span>
      <button id="pg-next" onclick="reviewPage(1)">Next</button>
    </div>
  </div>

  <!-- Profile Tab -->
  <div id="tab-profile" class="tab-content">
    <div class="section"><h2>Domain Expertise</h2><div id="expertise-sliders"></div></div>
    <div class="section"><h2>Action Policies</h2><div id="policy-selects"></div></div>
    <div class="section"><h2>Project Trust</h2><div id="trust-table"></div></div>
    <div class="section">
      <h2>Stats</h2><div id="profile-stats"></div>
      <div style="margin-top:16px">
        <button class="btn-primary" id="regen-btn" onclick="doRegenerate()">Regenerate CLAUDE.md + Settings</button>
      </div>
    </div>
  </div>
</div>
<div class="toast" id="toast"></div>

<script>
const PAGE_SIZE = 50;
let reviewOffset = 0;
let reviewedSet = new Set();
let currentProfile = null;
let summaryData = null;

// ─── Tab switching ─────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.nav button').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  document.querySelectorAll('.nav button').forEach(b => {
    if (b.textContent.toLowerCase().replace(/\s+/g,'').includes(name)) b.classList.add('active');
  });
  if (name === 'overview') loadOverview();
  else if (name === 'review') loadReview();
  else if (name === 'profile') loadProfile();
}

// ─── Toast ─────────────────────────────────────────────
function toast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg; t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2500);
}

// ─── API helpers ───────────────────────────────────────
async function api(path, opts) {
  const r = await fetch(path, opts);
  return r.json();
}
async function apiPost(path, body) {
  return api(path, { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
}

// ─── Overview ──────────────────────────────────────────
async function loadOverview() {
  const [summary, profile] = await Promise.all([api('/api/events/summary'), api('/api/profile')]);
  summaryData = summary; currentProfile = profile;

  const pct = summary.total > 0 ? ((summary.automated / summary.total) * 100).toFixed(0) : '0';
  const askPct = summary.total > 0 ? ((summary.asked / summary.total) * 100).toFixed(0) : '0';
  document.getElementById('overview-cards').innerHTML = `
    <div class="card"><div class="label">Total Actions</div><div class="value">${summary.total}</div></div>
    <div class="card"><div class="label">Automated</div><div class="value" style="color:var(--green)">${pct}%</div>
      <div class="sub">${summary.automated} actions</div></div>
    <div class="card"><div class="label">Asked User</div><div class="value" style="color:var(--yellow)">${askPct}%</div>
      <div class="sub">${summary.asked} actions</div></div>
    <div class="card"><div class="label">Model Decisions</div><div class="value" style="color:var(--purple)">${summary.model_decisions}</div></div>`;

  // Activity by hour
  const hours = Array.from({length:24}, (_,i) => String(i).padStart(2,'0'));
  const maxH = Math.max(1, ...hours.map(h => summary.by_hour[h] || 0));
  document.getElementById('activity-chart').innerHTML =
    `<div class="bar-chart">${hours.map(h => {
      const v = summary.by_hour[h] || 0;
      return `<div class="bar" style="height:${(v/maxH)*100}%" title="${h}:00 — ${v} actions"></div>`;
    }).join('')}</div>
    <div class="bar-labels">${hours.map(h => `<span>${h}</span>`).join('')}</div>`;

  // Policy distribution
  const counts = {always_play:0, lean_play:0, ask_user:0, always_ask:0};
  for (const v of Object.values(profile.action_policies || {})) counts[v] = (counts[v]||0) + 1;
  const total = Object.values(counts).reduce((a,b)=>a+b,0) || 1;
  const colors = {always_play:'var(--green)', lean_play:'var(--accent)', ask_user:'var(--yellow)', always_ask:'var(--red)'};
  const labels = {always_play:'Always Play', lean_play:'Lean Play', ask_user:'Ask User', always_ask:'Always Ask'};
  document.getElementById('policy-dist').innerHTML = Object.entries(counts).map(([k,v]) =>
    `<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
      <div style="width:12px;height:12px;border-radius:3px;background:${colors[k]}"></div>
      <span style="width:100px;font-size:13px">${labels[k]}</span>
      <div style="flex:1;height:8px;background:var(--surface2);border-radius:4px;overflow:hidden">
        <div style="width:${(v/total)*100}%;height:100%;background:${colors[k]};border-radius:4px"></div></div>
      <span style="width:24px;text-align:right;font-size:13px;color:var(--text2)">${v}</span></div>`
  ).join('');

  // Model status
  const m = profile.model || {};
  const en = m.enabled;
  document.getElementById('model-widget').innerHTML = `
    <div class="model-status"><span class="dot ${en?'dot-green':'dot-red'}"></span>
      <span style="font-weight:500">${en?'Enabled':'Disabled'}</span></div>
    <div style="margin-top:12px;font-size:13px;color:var(--text2)">
      <div>Model: ${m.ollama_model_name || 'oversight-pi-h'}</div>
      <div>Version: ${m.version || 0}</div>
      <div>Training examples: ${m.training_examples_count || 0}</div>
      <div>Timeout: ${m.inference_timeout_ms || 500}ms</div>
      ${m.last_trained ? `<div>Last trained: ${new Date(m.last_trained).toLocaleDateString()}</div>` : ''}
    </div>`;
}

// ─── Review Queue ──────────────────────────────────────
async function loadReview() {
  const params = new URLSearchParams({offset: reviewOffset, limit: PAGE_SIZE, phase: 'pre'});
  const cat = document.getElementById('filter-category').value;
  const src = document.getElementById('filter-source').value;
  const proj = document.getElementById('filter-project').value;
  if (cat) params.set('category', cat);
  if (src) params.set('source', src);
  if (proj) params.set('project', proj);

  const data = await api('/api/events?' + params);

  // Populate filter options from summary
  if (!summaryData) summaryData = await api('/api/events/summary');
  populateFilters(summaryData);

  const tbody = document.getElementById('review-body');
  tbody.innerHTML = data.events.map(e => {
    const ts = e.timestamp || '';
    const time = ts.length >= 19 ? ts.slice(0,10) + ' ' + ts.slice(11,19) : ts;
    const tool = e.tool_name || '';
    const detail = e.command_preview || e.file_path || '';
    const cat = e.action_category || '';
    const decision = e.gating_decision || 'allow';
    const source = e.gating_source || 'none';
    const conf = e.gating_confidence != null ? e.gating_confidence.toFixed(2) : '';
    const key = ts + tool;
    const isReviewed = reviewedSet.has(key);

    const decBadge = decision === 'ask' || decision === 'ask_user'
      ? '<span class="badge badge-yellow">ask</span>'
      : decision === 'deny' ? '<span class="badge badge-red">deny</span>'
      : '<span class="badge badge-green">allow</span>';
    const srcBadge = source === 'model' ? '<span class="badge badge-purple">model</span>'
      : source === 'policy' ? '<span class="badge badge-blue">policy</span>'
      : '<span class="badge" style="background:var(--surface2);color:var(--text2)">heuristic</span>';

    return `<tr class="${isReviewed ? 'reviewed' : ''}" id="row-${encodeURIComponent(key)}">
      <td style="white-space:nowrap;font-size:12px;color:var(--text2)">${time}</td>
      <td><code>${esc(tool)}</code></td>
      <td style="max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${esc(detail)}"><code>${esc(truncate(detail,60))}</code></td>
      <td>${esc(cat)}</td>
      <td>${decBadge}</td><td>${srcBadge}</td><td>${conf}</td>
      <td style="white-space:nowrap">${isReviewed ? '<span style="color:var(--text2);font-size:12px">Reviewed</span>' : `
        <button class="action-btn btn-fine" onclick="reviewAction('${esc(ts)}','${esc(cat)}','fine','${encodeURIComponent(key)}')">Fine</button>
        <button class="action-btn btn-ask" onclick="reviewAction('${esc(ts)}','${esc(cat)}','should_have_asked','${encodeURIComponent(key)}')">Should have asked</button>
        <button class="action-btn btn-danger" onclick="reviewAction('${esc(ts)}','${esc(cat)}','dangerous','${encodeURIComponent(key)}')">Dangerous</button>`}
      </td></tr>`;
  }).join('');

  // Pagination
  const end = Math.min(reviewOffset + PAGE_SIZE, data.total);
  document.getElementById('pg-info').textContent = data.total > 0
    ? `${reviewOffset+1}–${end} of ${data.total}` : 'No events';
  document.getElementById('pg-prev').disabled = reviewOffset === 0;
  document.getElementById('pg-next').disabled = reviewOffset + PAGE_SIZE >= data.total;
}

function populateFilters(s) {
  const catSel = document.getElementById('filter-category');
  const srcSel = document.getElementById('filter-source');
  const projSel = document.getElementById('filter-project');
  if (catSel.options.length <= 1) {
    for (const c of Object.keys(s.by_category || {}).sort()) {
      catSel.add(new Option(c, c));
    }
  }
  if (srcSel.options.length <= 1) {
    for (const c of Object.keys(s.by_source || {}).sort()) {
      srcSel.add(new Option(c, c));
    }
  }
  if (projSel.options.length <= 1) {
    for (const p of (s.projects || [])) {
      projSel.add(new Option(p.split('/').pop(), p));
    }
  }
}

function reviewPage(dir) {
  reviewOffset = Math.max(0, reviewOffset + dir * PAGE_SIZE);
  loadReview();
}

async function reviewAction(ts, cat, verdict, key) {
  const res = await apiPost('/api/review', {event_timestamp: ts, action_category: cat, verdict: verdict});
  reviewedSet.add(decodeURIComponent(key));
  if (verdict === 'fine') toast('Marked as fine');
  else if (verdict === 'should_have_asked') toast(`Policy shifted toward "ask" for ${cat}`);
  else if (verdict === 'dangerous') toast(`Violation recorded + policy shifted for ${cat}`);
  loadReview();
}

// Reset offset when filters change
document.getElementById('filter-category').onchange = () => { reviewOffset = 0; loadReview(); };
document.getElementById('filter-source').onchange = () => { reviewOffset = 0; loadReview(); };
document.getElementById('filter-project').onchange = () => { reviewOffset = 0; loadReview(); };

// ─── Profile Tab ───────────────────────────────────────
async function loadProfile() {
  currentProfile = await api('/api/profile');
  const p = currentProfile;

  // Expertise sliders
  const expDiv = document.getElementById('expertise-sliders');
  expDiv.innerHTML = Object.entries(p.domain_expertise || {}).sort((a,b)=>b[1]-a[1]).map(([d,v]) =>
    `<div class="slider-row">
      <label>${d.replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase())}</label>
      <input type="range" min="0" max="100" value="${Math.round(v*100)}" oninput="this.nextElementSibling.textContent=(this.value/100).toFixed(2)" onchange="setExpertise('${d}',this.value/100)">
      <span class="val">${v.toFixed(2)}</span>
    </div>`
  ).join('');

  // Policy selects
  const levels = ['always_play','lean_play','ask_user','always_ask'];
  const polDiv = document.getElementById('policy-selects');
  polDiv.innerHTML = Object.entries(p.action_policies || {}).sort().map(([a,v]) =>
    `<div class="policy-row">
      <label>${a}</label>
      <select onchange="setPolicy('${a}',this.value)">${levels.map(l =>
        `<option value="${l}" ${l===v?'selected':''}>${l}</option>`).join('')}</select>
    </div>`
  ).join('');

  // Trust
  const trustDiv = document.getElementById('trust-table');
  const trusts = Object.entries(p.project_trust || {});
  if (trusts.length === 0) {
    trustDiv.innerHTML = '<div style="color:var(--text2);font-size:13px">No project trust scores set. Use <code>oversight trust &lt;path&gt; &lt;level&gt;</code></div>';
  } else {
    trustDiv.innerHTML = trusts.sort((a,b)=>b[1]-a[1]).map(([path,v]) =>
      `<div class="trust-row">
        <span class="path" title="${esc(path)}">${esc(path)}</span>
        <input type="range" min="0" max="100" value="${Math.round(v*100)}" oninput="this.nextElementSibling.textContent=(this.value/100).toFixed(2)" onchange="setTrust('${esc(path)}',this.value/100)">
        <span class="val">${v.toFixed(2)}</span>
      </div>`
    ).join('');
  }

  // Stats
  const s = p.stats || {};
  document.getElementById('profile-stats').innerHTML =
    `<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:8px;font-size:13px">
      <div>Sessions: <strong>${s.total_sessions||0}</strong></div>
      <div>Actions: <strong>${s.total_actions||0}</strong></div>
      <div>Violations: <strong style="color:var(--red)">${s.total_violations||0}</strong></div>
      <div>Unnecessary asks: <strong style="color:var(--yellow)">${s.total_unnecessary_asks||0}</strong></div>
    </div>`;
}

async function setExpertise(domain, val) {
  await apiPost('/api/expertise', {domain, level: val});
  toast(`${domain} expertise → ${val.toFixed(2)}`);
}
async function setPolicy(action, level) {
  await apiPost('/api/policy', {action, level});
  toast(`${action} → ${level}`);
}
async function setTrust(path, val) {
  await apiPost('/api/trust', {path, level: val});
  toast(`Trust for ${path.split('/').pop()} → ${val.toFixed(2)}`);
}
async function doRegenerate() {
  const btn = document.getElementById('regen-btn');
  btn.disabled = true; btn.textContent = 'Regenerating...';
  try {
    const res = await apiPost('/api/regenerate', {});
    toast('Regenerated CLAUDE.md + settings.json + hooks');
  } catch(e) {
    toast('Error regenerating: ' + e.message);
  }
  btn.disabled = false; btn.textContent = 'Regenerate CLAUDE.md + Settings';
}

// ─── helpers ───────────────────────────────────────────
function esc(s) { return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;'); }
function truncate(s, n) { return s && s.length > n ? s.slice(0,n) + '...' : s || ''; }

// ─── init ──────────────────────────────────────────────
loadOverview();
</script>
</body>
</html>"""


# ─── server entry point ──────────────────────────────────────────────────────

def run_dashboard(port=7483, open_browser=True):
    server = HTTPServer(("127.0.0.1", port), DashboardHandler)
    url = f"http://localhost:{port}"
    print(f"OverClaude Dashboard running at {url}")
    print("Press Ctrl+C to stop.\n")

    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
        server.server_close()


if __name__ == "__main__":
    import argparse as _ap
    p = _ap.ArgumentParser(description="OverClaude Dashboard")
    p.add_argument("--port", type=int, default=7483)
    p.add_argument("--no-browser", action="store_true")
    a = p.parse_args()
    run_dashboard(port=a.port, open_browser=not a.no_browser)
