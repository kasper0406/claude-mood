#!/usr/bin/env python3
"""claude-mood hook: turns moodd's sensor log into context for Claude.

    mood_hook.py prompt   UserPromptSubmit: reaction since your previous message
    mood_hook.py tool     PostToolUse: interrupt-style nudge if you groan mid-turn
    mood_hook.py status   one-line status bar
    mood_hook.py report   human-readable summary of the last few minutes
"""
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

STATE_DIR = Path(os.environ.get("CLAUDE_MOOD_DIR", Path.home() / ".cache" / "claude-mood"))
STALE_S = 10
MAX_WINDOW_S = 600
PROMPT_THRESH = float(os.environ.get("CLAUDE_MOOD_THRESHOLD", 0.35))
TOOL_THRESH = 0.5
TOOL_WINDOW_S = 20
TOOL_COOLDOWN_S = 90
EVENT_WEIGHT = {"sound": 0.3, "voice": 0.25, "words": 0.3}
SLAMS = {"Slam", "Smash, crash"}


def load_samples():
    try:
        st = json.loads((STATE_DIR / "state.json").read_text())
    except (OSError, ValueError):
        return None
    if time.time() - st.get("updated", 0) > STALE_S:
        return None
    return st["samples"]


def summarize(samples, t0, t1=None):
    t1 = t1 or time.time()
    win = [s for s in samples if t0 <= s["t"] <= t1]
    seen = [s for s in win if "fj" in s]
    events = [e for s in win for e in s["ev"]]
    out = {"seconds": round(t1 - t0), "face_s": len(seen),
           "joy_ev": Counter(e["what"] for e in events if e["pol"] == "joy"),
           "frust_ev": Counter(e["what"] for e in events if e["pol"] == "frust")}
    fj = sum(s["fj"] for s in seen) / len(seen) if seen else 0.0
    ff = sum(s["ff"] for s in seen) / len(seen) if seen else 0.0
    out["smiling"] = sum(s["fj"] > 0.5 for s in seen) / len(seen) if seen else 0.0
    out["scowling"] = sum(s["ff"] > 0.3 for s in seen) / len(seen) if seen else 0.0

    def ev_score(pol):
        sc = 0.0
        for e in events:
            if e["pol"] == pol:
                sc += 0.5 if e["what"] in SLAMS else EVENT_WEIGHT.get(e["src"], 0.25)
        return sc

    out["joy"] = min(1.0, 0.5 * fj + 0.5 * out["smiling"] + ev_score("joy"))
    out["frust"] = min(1.0, 0.5 * ff + 0.5 * out["scowling"] + ev_score("frust"))
    return out


def describe(s, pol):
    parts = []
    if pol == "frust" and s["scowling"] >= 0.2:
        parts.append(f"frowning/annoyed face {s['scowling']:.0%} of the time")
    if pol == "joy" and s["smiling"] >= 0.2:
        parts.append(f"smiling {s['smiling']:.0%} of the time")
    for what, n in s[f"{pol}_ev"].most_common(6):
        parts.append(what + (f" x{n}" if n > 1 else ""))
    return "; ".join(parts) or "weak signals only"


def mark(name):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (STATE_DIR / name).write_text(str(time.time()))


def read_mark(name):
    try:
        return float((STATE_DIR / name).read_text())
    except (OSError, ValueError):
        return None


def emit(event, text):
    print(json.dumps({"hookSpecificOutput": {"hookEventName": event, "additionalContext": text}}))


FRUST_GUIDANCE = (
    "Treat this as a hint, not a fact: webcam/mic affect detection is noisy and concentration "
    "can look like frowning, and the cause may be unrelated to you. But if it plausibly relates "
    "to your recent work, consider whether you misunderstood the request, are going in circles, "
    "or are being too verbose; prefer stopping to ask one crisp question over pushing on. "
    "You may acknowledge it in at most one short, light sentence; don't grovel or be sycophantic.")
JOY_GUIDANCE = (
    "Whatever you just did seems to have landed; keep that approach. You may acknowledge it in "
    "at most one short, light sentence, or not at all.")


def on_prompt():
    now = time.time()
    since = read_mark("last_prompt")
    mark("last_prompt")
    samples = load_samples()
    if not samples:
        return
    t0 = max(since or now - 120, now - MAX_WINDOW_S)
    s = summarize(samples, t0, now)
    if s["frust"] < PROMPT_THRESH and s["joy"] < PROMPT_THRESH:
        return
    head = (f"[claude-mood] Local webcam/mic sensors over the {s['seconds']}s since the user's "
            f"previous message (frustration {s['frust']:.2f}, joy {s['joy']:.2f}): ")
    if min(s["frust"], s["joy"]) >= PROMPT_THRESH:
        emit("UserPromptSubmit", head +
             f"mixed signals, possibly laughing at something absurd. Frustration: {describe(s, 'frust')}. "
             f"Joy: {describe(s, 'joy')}. " + FRUST_GUIDANCE)
    elif s["frust"] >= PROMPT_THRESH:
        emit("UserPromptSubmit", head + f"the user appears frustrated. Signals: {describe(s, 'frust')}. "
             + FRUST_GUIDANCE)
    else:
        emit("UserPromptSubmit", head + f"the user appears pleased. Signals: {describe(s, 'joy')}. "
             + JOY_GUIDANCE)


def on_tool():
    now = time.time()
    last = read_mark("last_tool_nudge")
    if last and now - last < TOOL_COOLDOWN_S:
        return
    samples = load_samples()
    if not samples:
        return
    s = summarize(samples, now - TOOL_WINDOW_S, now)
    if s["frust"] < TOOL_THRESH:
        return
    mark("last_tool_nudge")
    emit("PostToolUse",
         f"[claude-mood] While you were working, over the last {TOOL_WINDOW_S}s the user's webcam/mic "
         f"suggest growing frustration (score {s['frust']:.2f}: {describe(s, 'frust')}). They may be "
         f"watching you head in the wrong direction. Briefly sanity-check your current approach; if "
         f"unsure it is what they want, wrap up and ask rather than continuing. " + FRUST_GUIDANCE)


def status():
    samples = load_samples()
    if samples is None:
        print("mood: off")
        return
    s = summarize(samples, time.time() - 30)
    face = "no face" if s["face_s"] == 0 else ("😤" if s["frust"] > 0.5 else "😄" if s["joy"] > 0.5
                                              else "🙂" if s["joy"] > s["frust"] else "😐")
    paused = " (paused)" if (STATE_DIR / "paused").exists() else ""
    print(f"{face} joy {s['joy']:.2f} · frust {s['frust']:.2f}{paused}")


def report():
    samples = load_samples()
    if samples is None:
        print("moodd is not running (or state is stale).")
        return
    for label, secs in (("last 30s", 30), ("last 5 min", 300)):
        s = summarize(samples, time.time() - secs)
        print(f"{label}: joy {s['joy']:.2f} ({describe(s, 'joy')}) | "
              f"frustration {s['frust']:.2f} ({describe(s, 'frust')}) | face seen {s['face_s']}s")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd in ("prompt", "tool"):
        sys.stdin.read()  # hook payload; unused
    try:
        {"prompt": on_prompt, "tool": on_tool, "status": status, "report": report}[cmd]()
    except Exception as e:  # never break the user's session over a joke plugin
        print(f"claude-mood: {e}", file=sys.stderr)
