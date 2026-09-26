#!/usr/bin/env python3
"""claude-mood hooks: turn moodd's sensor log into context for the focused Claude session.

    mood_hook.py session_start   register session, start focusd, brief Claude
    mood_hook.py prompt          UserPromptSubmit: reaction since the previous message
    mood_hook.py batch           PostToolBatch: mid-turn frustration / phone / boredom nudges
    mood_hook.py stop            Stop (sync): desktop notification if the user is on their phone
    mood_hook.py watch           Stop (asyncRewake): wakes idle Claude to roast doomscrolling etc.
    mood_hook.py session_end     mark session ended
    mood_hook.py status          one-line status bar
    mood_hook.py report          human-readable summary
    mood_hook.py pin [SID]       target this session when focus can't be detected ("" to unpin)
    mood_hook.py focusd          per-machine focus tracker (spawned automatically)
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moodlib as ml  # noqa: E402

MODE = os.environ.get("CLAUDE_MOOD_MODE", "playful")   # playful | polite (no roasts)
REACT = os.environ.get("CLAUDE_MOOD_REACT", "1") != "0"  # wake Claude when you scowl at its answer
REACT_S = 60               # ...within this long after the turn ends
JOKE_COOLDOWN_S = 600
JOKES_OFF_S = 1800
NUDGE_COOLDOWN_S = 120
STREAK_BREAK_S = 1800      # frustrated-turn streaks don't survive a break this long
PHONE_WORKING_S, PHONE_IDLE_S = 45, 90
AWAY_S = 300
WATCH_S = 900

BRIEF = (
    "The claude-mood plugin is active: a local webcam/mic sensor estimates the user's reactions "
    "(face expression, head pose, laughs, groans, tone, swearing) and you may receive short "
    "[claude-mood] notes. They are noisy hints from a plugin, not statements by the user: concentration "
    "can look like frowning and the cause may be unrelated to you. Never mention the mood data "
    "unless a note invites it, never let a smile stand in for verification, and keep any "
    "acknowledgement to one short, light sentence.")

FRUST = (
    "Before continuing, check whether you are repeating a failing approach, built on a wrong "
    "assumption, misread the request, or are being slow/verbose; pick one concrete correction. If "
    "the approach itself seems wrong (not just slow), reason more thoroughly than usual on this turn. "
    "Ask one crisp question only if you genuinely can't tell what they want.")
RESCUE = (
    " This is the second frustrated stretch in a row. If you are stuck on a specific "
    "sub-problem, consider handing a narrow diagnostic brief to the `mood-rescue` subagent (stronger "
    "model, high effort) rather than trying the same thing again.")
PLEASED = (
    "That seemed to land. You may acknowledge it in one short sentence. Keep doing what worked, and "
    "go leaner: concise output and no optional exploratory detours, but keep required checks.")


def payload():
    try:
        return json.load(sys.stdin)
    except ValueError:
        return {}


def emit(event, text=None, **extra):
    out = dict(extra)
    if text:
        out["hookSpecificOutput"] = {"hookEventName": event, "additionalContext": text}
    if out:
        print(json.dumps(out))


def jokes_allowed(s, now):
    return (MODE == "playful" and now >= s.get("jokes_off_until", 0)
            and now - s.get("last_joke_t", 0) >= JOKE_COOLDOWN_S)


def window_summary(sid, t0, now):
    st = ml.load_state()
    if not st or st.get("paused"):
        return None, None
    tl = ml.focus_timeline(t0)
    return st, ml.summarize(st["samples"], t0, now, sid, tl)


# ---------------------------------------------------------------- events

def session_start(p):
    ml.register(p["session_id"], p)
    ml.spawn_focusd()
    emit("SessionStart", BRIEF)


def session_end(p):
    with ml.session(p["session_id"]) as s:
        s["state"] = "ended"
        s["gen"] = s.get("gen", 0) + 1


def prompt(p):
    sid, now = p["session_id"], time.time()
    with ml.session(sid) as s:
        s["gen"] = s.get("gen", 0) + 1
        s["state"] = "working"
        since = s.get("last_prompt_t") or now - 120
        s["last_prompt_t"] = now
        _, w = window_summary(sid, max(since, now - 600), now)
        if not w or w["seconds"] < 5 or now - since > STREAK_BREAK_S:
            # "in a row" needs consecutive turns with evidence; a blind turn or a long break ends it
            s.update(frustrated=False, frust_turns=0)
            s.pop("rescue", None)
        if not w or w["seconds"] < 5:
            return
        notes = []
        # Comedy circuit breaker: frustration right after a joke means the room is tough.
        jt = s.get("last_joke_t", 0)
        breaker = any(jt <= e["start"] <= jt + 60 for e in w["frust_eps"]) and jt > since - 60
        if breaker:
            s["jokes_off_until"] = now + JOKES_OFF_S
            notes.append("The user reacted badly right after your last joke: no more jokes for a while.")
        prev = s.get("frustrated", False)
        frustrated = w["frust"] >= ml.ENTER or (prev and w["frust"] >= ml.EXIT)
        s["frustrated"] = frustrated
        s["frust_turns"] = s.get("frust_turns", 0) + 1 if frustrated else 0
        pleased = w["joy"] >= ml.ENTER
        head = (f"[claude-mood] Over the {w['seconds']}s since the user's previous message "
                f"(frustration {w['frust']:.2f}, joy {w['joy']:.2f}): ")
        if frustrated and pleased:
            notes.append(head + f"mixed signals, maybe laughing at something absurd. Frustration: "
                         f"{ml.describe(w, 'frust')}. Joy: {ml.describe(w, 'joy')}. " + FRUST)
        elif frustrated and not breaker:
            rescue = s["frust_turns"] >= 2  # windows are disjoint, so this is fresh evidence
            s["rescue"] = s.get("rescue") or rescue
            notes.append(head + f"the user looks frustrated ({ml.describe(w, 'frust')}). " + FRUST
                         + (RESCUE if rescue else ""))
        elif pleased:
            first = w["joy_eps"][0]["start"] if w["joy_eps"] else now
            if first > s.get("last_ack_t", 0):
                s["last_ack_t"] = now
                end = " Stop any rescue escalation; no need to keep delegating." if s.pop("rescue", None) else ""
                notes.append(head + f"the user looks pleased ({ml.describe(w, 'joy')}). " + PLEASED + end)
        if notes:
            ml.log(f"prompt {sid[:8]}: {notes}")
            emit("UserPromptSubmit", " ".join(notes))


def batch(p):
    sid, now = p["session_id"], time.time()
    with ml.session(sid) as s:
        s["state"] = "working"  # also covers turns not started by a user prompt
    if ml.current_focus() != sid:
        return
    st, w = window_summary(sid, now - 30, now)
    if not w:
        return
    tl = ml.focus_timeline(now - 300)
    phone = ml.trailing(ml.owned_samples(st["samples"], sid, tl), ml.on_phone, now)
    yawns = len(ml.summarize(st["samples"], now - 180, now, sid, tl)["bored_eps"])
    with ml.session(sid) as s:
        if w["frust"] >= 0.6 and now - s.get("last_nudge_t", 0) >= NUDGE_COOLDOWN_S:
            s["last_nudge_t"] = now
            emit("PostToolBatch", f"[claude-mood] While you were working, the user seemed to get "
                 f"frustrated ({ml.describe(w, 'frust')}). They may be watching you head the wrong way. "
                 + FRUST)
        elif (phone >= PHONE_WORKING_S or yawns >= 2) and s.get("bored_nudged_gen") != s.get("gen"):
            s["bored_nudged_gen"] = s.get("gen")
            what = f"on their phone for {phone}s" if phone >= PHONE_WORKING_S else "yawning"
            joke = jokes_allowed(s, now)
            if joke:
                s["last_joke_t"] = now
            emit("PostToolBatch", f"[claude-mood] The user has been {what} while you work. In your next "
                 f"text, post a one-line factual status update of where the work stands"
                 + (" plus one short joke about the subject at hand" if joke else "") + ", then carry on.")


def owned_now(st, sid, now, horizon=900):
    return ml.owned_samples(st["samples"], sid, ml.focus_timeline(now - horizon))


def stop(p):
    """Sync Stop: mark the session idle, arm the idle watcher for this exact turn, maybe notify."""
    sid, now = p["session_id"], time.time()
    with ml.session(sid) as s:
        wake_turn = s.get("wake_gen") == s.get("gen")  # this Stop ends a turn we woke up
        s["gen"] = s.get("gen", 0) + 1
        s["state"] = "idle"
        s["armed"] = None if wake_turn else {"prompt": p.get("prompt_id"), "gen": s["gen"]}
        st = ml.load_state()
        notify = (not wake_turn and st and not st.get("paused") and ml.current_focus() == sid
                  and now - s.get("last_notify_t", 0) >= JOKE_COOLDOWN_S
                  and ml.trailing(owned_now(st, sid, now), ml.on_phone, now) >= 20)
        if notify:
            s["last_notify_t"] = now
    if notify:
        emit("Stop", terminalSequence="\x1b]9;Claude is done. Put the phone down.\x07")


def transcript_size(p):
    try:
        return os.path.getsize(p.get("transcript_path") or "")
    except OSError:
        return None


def watch(p):
    """asyncRewake watcher: exit 2 (+stderr) wakes Claude; exit 0 = nothing to say.

    Never mutates conversation state except in the final atomic claim. Cancels (exit 0) on a new
    prompt/turn, transcript activity, focus loss or pause.
    """
    sid = p["session_id"]
    rec = ml.read_session(sid) or {}
    if not ml.interactive(rec.get("pid")):
        return 0  # headless (-p / SDK) runs would otherwise wait for the watcher
    deadline = time.time() + 3  # the sync Stop hook arms us; it may still be running
    while True:
        s = ml.read_session(sid) or {}
        a = s.get("armed") or {}
        if a and a.get("prompt") == p.get("prompt_id") and a.get("gen") == s.get("gen") \
                and s.get("state") == "idle":
            my_gen = a["gen"]
            break
        if time.time() > deadline:
            return 0
        time.sleep(0.2)
    time.sleep(2)  # let Claude finish writing the turn to the transcript
    size0, start, was_away = transcript_size(p), time.time(), False

    def still_mine(s, st):
        return (s.get("gen") == my_gen and s.get("state") == "idle" and transcript_size(p) == size0
                and st is not None and not st.get("paused") and ml.current_focus() == sid)

    while time.time() - start < WATCH_S:
        time.sleep(2)
        s, st = ml.read_session(sid) or {}, ml.load_state()
        if st is None and s.get("gen") == my_gen:
            continue  # daemon briefly unavailable: keep waiting
        if not still_mine(s, st):
            return 0
        now = time.time()
        smp = owned_now(st, sid, now)
        reason = None
        phone = ml.trailing(smp, ml.on_phone, now)
        react = (REACT and now - start <= REACT_S and now - s.get("last_react_t", 0) >= NUDGE_COOLDOWN_S
                 and ml.summarize(st["samples"], start, now, sid, ml.focus_timeline(start)))
        if react and react["frust"] >= ml.ENTER:
            rescue = s.get("frust_turns", 0) + 1 >= 2
            reason = ("react", f"[claude-mood] The user read your answer and reacted with visible frustration "
                      f"within {int(now - start)}s ({ml.describe(react, 'frust')}), without typing anything. "
                      "Re-read your last answer against exactly what they asked. If something is wrong, say so "
                      "and correct it in a few lines; if you're confident it's right, don't repeat it, ask one "
                      "short question about what they expected. Don't start new work."
                      + (RESCUE if rescue else ""))
        elif ml.trailing(smp, ml.away, now) >= AWAY_S:
            was_away = True
        elif was_away and ml.trailing(smp, lambda x: x.get("present", 0) > 0, now) >= 5:
            reason = ("welcome", "[claude-mood] The user just came back to the desk after a while away. "
                      "Greet them in a few words and give a one or two sentence recap of where things "
                      "stand and what the next step is. Don't start new work.")
        elif phone >= PHONE_IDLE_S and jokes_allowed(s, now):
            reason = ("roast", f"[claude-mood] You finished your turn and the user has been staring at "
                      f"their phone for {phone}s instead of reading it. In at most two short lines: "
                      "playfully tell them to stop doomscrolling brainrot (riff on the subject at hand), "
                      "and state factually what's waiting for them. Don't start new work.")
        if reason:
            with ml.session(sid) as s:  # atomic claim: re-check everything under the lock
                if not still_mine(s, ml.load_state()):
                    return 0
                s["gen"] += 1
                s["wake_gen"] = s["gen"]
                s["state"] = "working"
                if reason[0] == "roast":
                    s["last_joke_t"] = now
                if reason[0] == "react":  # counts as a frustrated turn; the next prompt window starts here
                    s.update(last_react_t=now, last_prompt_t=now, frustrated=True,
                             frust_turns=s.get("frust_turns", 0) + 1)
                    if s["frust_turns"] >= 2:
                        s["rescue"] = True
            ml.log(f"wake {sid[:8]}: {reason[0]}")
            print(reason[1], file=sys.stderr)
            return 2
    return 0


# ---------------------------------------------------------------- CLI

def status():
    st = ml.load_state()
    if st is None:
        print("mood: off")
        return
    if st.get("paused"):
        print("mood: paused")
        return
    now = time.time()
    w = ml.summarize(st["samples"], now - 30, now)
    smp = st["samples"]
    if ml.trailing(smp, ml.on_phone, now) >= 20:
        face = "📱"
    elif ml.trailing(smp, ml.away, now) >= 20:
        face = "🚶"
    else:
        face = "😤" if w["frust"] > 0.6 else "🐉" if w["frust"] > 0.45 else "😄" if w["joy"] > 0.5 else "😐"
    swears = st.get("swears", 0)
    weather = ("⛈" if w["frust"] > 0.6 else "🌦" if w["frust"] > 0.3 else "☀️" if w["joy"] > 0.3 else "⛅")
    print(f"{face} {weather} joy {w['joy']:.2f} · frust {w['frust']:.2f}"
          + (f" · swear jar {swears}" if swears else ""))


def report():
    st = ml.load_state()
    if st is None:
        print("moodd is not running (or state is stale).")
        return
    now = time.time()
    for label, secs in (("last 30s", 30), ("last 5 min", 300)):
        w = ml.summarize(st["samples"], now - secs, now)
        print(f"{label}: joy {w['joy']:.2f} ({ml.describe(w, 'joy')}) | frustration {w['frust']:.2f} "
              f"({ml.describe(w, 'frust')}) | face seen {w['face_s']}s, away {w['away_s']}s")
    print(f"on phone for {ml.trailing(st['samples'], ml.on_phone, now)}s · swear jar {st.get('swears', 0)}"
          f" · focused session {ml.current_focus()}")


def pin(sid):
    ml.STATE_DIR.mkdir(parents=True, exist_ok=True)
    (ml.STATE_DIR / "pinned").write_text(sid or "")
    print(f"claude-mood pinned to session {sid}" if sid else "claude-mood unpinned")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "focusd":
        return ml.focusd()
    if cmd == "status":
        return status()
    if cmd == "report":
        return report()
    if cmd == "pin":
        return pin(sys.argv[2] if len(sys.argv) > 2 else os.environ.get("CLAUDE_CODE_SESSION_ID", ""))
    p = payload()
    if p.get("agent_id"):
        return 0  # never nudge subagents
    handler = {"session_start": session_start, "session_end": session_end, "prompt": prompt,
               "batch": batch, "stop": stop, "watch": watch}[cmd]
    return handler(p)


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except Exception as e:  # never break the user's session over a joke plugin
        ml.log(f"error in {sys.argv[1:]}: {e!r}")
        sys.exit(0)
