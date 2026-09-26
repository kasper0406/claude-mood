"""Shared plumbing for the claude-mood hooks: sensor state, sessions, focus, scoring.

stdlib only; runs on the machine where Claude Code runs (which may differ from
the machine with the webcam, see CLAUDE_MOOD_URL).
"""
import contextlib
import fcntl
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

STATE_DIR = Path(os.environ.get("CLAUDE_MOOD_DIR", Path.home() / ".cache" / "claude-mood"))
SESS_DIR = STATE_DIR / "sessions"
FOCUS_LOG = STATE_DIR / "focus.jsonl"
MOOD_URL = os.environ.get("CLAUDE_MOOD_URL")  # e.g. http://127.0.0.1:7433 via ssh -R
STALE_S = 10
FOCUS_HB_S, FOCUS_HB_MAX = 5, 12   # focusd heartbeat; focus older than this is unknown

# Scoring knobs.
EPISODE_GAP_S, EPISODE_MAX_S = 3.0, 8.0
EVENT_WEIGHT = {"sound": 0.8, "voice": 0.6, "words": 1.0, "face": 0.6}
SLAMS = {"Slam", "Smash, crash"}
K_AUDIO = 0.5                     # episodes/minute -> saturation; one outburst alone stays below ENTER
FACE_FRUST_THR, FACE_JOY_THR = 0.3, 0.5
MIN_FACE_S = 8                    # below this, face evidence is "unknown"
ENTER, EXIT = 0.45, 0.25          # frustration hysteresis
PHONE_PITCH, PHONE_LOOK = -18.0, 0.45


def log(msg):
    try:
        with open(STATE_DIR / "hooks.log", "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    except OSError:
        pass


# ---------------------------------------------------------------- sensor state

def load_state():
    """Daemon state dict, or None if unavailable/stale."""
    try:
        if MOOD_URL:
            with urllib.request.urlopen(MOOD_URL.rstrip("/") + "/state", timeout=1.5) as r:
                st = json.load(r)
        else:
            st = json.loads((STATE_DIR / "state.json").read_text())
    except (OSError, ValueError):
        return None
    if time.time() - st.get("updated", 0) > STALE_S:
        return None
    return st


# ---------------------------------------------------------------- sessions

@contextlib.contextmanager
def locked(name):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_DIR / f"{name}.lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


@contextlib.contextmanager
def session(sid):
    """Load-modify-save a session record under its lock."""
    SESS_DIR.mkdir(parents=True, exist_ok=True)
    path = SESS_DIR / f"{sid}.json"
    with locked(f"sessions/{sid}"):
        try:
            s = json.loads(path.read_text())
        except (OSError, ValueError):
            s = {"sid": sid, "gen": 0, "state": "idle"}
        yield s
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(s))
        tmp.replace(path)


def read_session(sid):
    try:
        return json.loads((SESS_DIR / f"{sid}.json").read_text())
    except (OSError, ValueError):
        return None


def live_sessions():
    out = []
    for p in SESS_DIR.glob("*.json"):
        try:
            s = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if s.get("state") != "ended" and proc_alive(s.get("pid"), s.get("pid_start")):
            out.append(s)
    return out


def ps(pid, fields):
    try:
        return subprocess.run(["ps", "-o", f"{fields}=", "-p", str(pid)], capture_output=True,
                              text=True, timeout=2).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def proc_alive(pid, start):
    return bool(pid) and ps(pid, "lstart") == start


def find_claude_pid():
    """The Claude Code process: $CLAUDE_PID if set, else walk up from the hook process."""
    if os.environ.get("CLAUDE_PID", "").isdigit():
        return int(os.environ["CLAUDE_PID"])
    pid = os.getppid()
    for _ in range(8):
        args = ps(pid, "args")
        if re.search(r"(^|/)claude(\s|$)|/claude/versions/", args):
            return pid
        ppid = ps(pid, "ppid")
        if not ppid or int(ppid) <= 1:
            break
        pid = int(ppid)
    return None


def interactive(pid):
    if os.environ.get("CLAUDE_CODE_ENTRYPOINT", "").startswith("sdk"):
        return False
    args = ps(pid, "args").split() if pid else []
    return not ({"-p", "--print"} & set(args))


def register(sid, payload):
    pid = find_claude_pid()
    env = os.environ
    with session(sid) as s:
        s.update(pid=pid, pid_start=ps(pid, "lstart") if pid else None,
                 tty=ps(pid, "tty") if pid else None, cwd=payload.get("cwd"),
                 tmux=env.get("TMUX"), tmux_pane=env.get("TMUX_PANE"),
                 term=env.get("TERM_PROGRAM"), iterm=env.get("ITERM_SESSION_ID"),
                 term_session=env.get("TERM_SESSION_ID"), host=os.uname().nodename,
                 state="idle", registered=time.time())


# ---------------------------------------------------------------- focus

def tmux(sock, *args):
    cmd = ["tmux"] + (["-S", sock] if sock else []) + list(args)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def tmux_focused_panes(sock):
    """Pane ids actually shown in a tmux client whose terminal has focus."""
    out = tmux(sock, "list-clients", "-F", "#{client_flags}\t#{client_session}")
    if out is None:
        return None
    panes = set()
    for line in out.splitlines():
        flags, sess = line.split("\t", 1)
        if "focused" in flags.split(","):
            pane = tmux(sock, "display-message", "-p", "-t", f"{sess}:", "#{pane_id}")
            if pane:
                panes.add(pane)
    return panes


MAC_FOCUS_SCRIPT = '''
tell application "System Events" to set frontApp to name of first process whose frontmost is true
if frontApp is "iTerm2" then
    tell application "iTerm2" to return tty of current session of current window
else if frontApp is "Terminal" then
    tell application "Terminal" to return tty of selected tab of front window
else
    return "app:" & frontApp
end if'''
_osa_errors = set()


def mac_focused_tty():
    """tty of the focused tab in the frontmost terminal app, '' if a non-terminal is frontmost."""
    try:
        r = subprocess.run(["osascript", "-e", MAC_FOCUS_SCRIPT], capture_output=True, text=True,
                           timeout=3)
    except (OSError, subprocess.SubprocessError):
        return None
    out = r.stdout.strip()
    if r.returncode != 0:
        err = r.stderr.strip()
        if err not in _osa_errors:  # focusd polls every second: log each distinct error once
            _osa_errors.add(err)
            log(f"focus: osascript failed: {err}")
        return None
    return "" if out.startswith("app:") else out.replace("/dev/", "")


def x11_active_pid():
    if not os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return None
    try:
        r = subprocess.run(["xdotool", "getactivewindow", "getwindowpid"], capture_output=True,
                           text=True, timeout=2)
        return int(r.stdout.strip()) if r.returncode == 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def ancestors(pid):
    out = []
    for _ in range(12):
        ppid = ps(pid, "ppid")
        if not ppid or int(ppid) <= 1:
            break
        pid = int(ppid)
        out.append(pid)
    return out


def resolve_focus(sessions):
    """Session id currently in focus, or None if unknown / nothing of ours is focused.

    tmux wins when the session lives in tmux (works over SSH: the outer terminal forwards
    focus events, needs `set -g focus-events on`). Otherwise macOS frontmost-terminal tty,
    or X11 active-window pid ancestry. A manual pin applies only when focus is undeterminable.
    """
    hits, determinable, ambiguous = [], False, False
    socks = {}
    for s in sessions:
        if s.get("tmux"):
            socks.setdefault(s["tmux"].split(",")[0], []).append(s)
    for sock, group in socks.items():
        panes = tmux_focused_panes(sock)
        if panes is None:
            continue
        determinable = True
        hits += [s["sid"] for s in group if s.get("tmux_pane") in panes]
    rest = [s for s in sessions if not s.get("tmux")]
    if rest and sys.platform == "darwin":
        tty = mac_focused_tty()
        if tty is not None:
            determinable = True
            hits += [s["sid"] for s in rest if tty and s.get("tty") == tty]
    elif rest:
        apid = x11_active_pid()
        if apid:
            determinable = True
            cands = [s for s in rest if apid in ancestors(s["pid"])]
            if len(cands) == 1:
                hits.append(cands[0]["sid"])
            elif len(cands) > 1:  # shared terminal server: can't tell which tab
                ambiguous = True
    if len(hits) == 1 and not ambiguous:
        return hits[0]
    if not determinable or ambiguous or len(hits) > 1:
        try:
            pin = (STATE_DIR / "pinned").read_text().strip()
        except OSError:
            pin = None
        if pin and any(s["sid"] == pin for s in sessions):
            return pin
    return None


HOOKS_DIR = Path(__file__).resolve().parent
FOCUSD_CODE = STATE_DIR / "focusd.code"  # hooks dir of the newest spawner (plugin updates move it)


def code_sig(d):
    try:
        return tuple((d / f).stat().st_mtime_ns for f in ("mood_hook.py", "moodlib.py"))
    except OSError:
        return None


def focusd():
    """Per-machine singleton: logs focus transitions to focus.jsonl once a second.

    Re-execs itself when its code changes or a newer plugin install asks for it, so a
    long-lived focusd never keeps running stale code.
    """
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    lockf = open(STATE_DIR / "focusd.lock", "w")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return  # already running
    sig = code_sig(HOOKS_DIR)
    last, last_write, idle_since = object(), 0.0, None
    while True:
        try:
            want = Path(FOCUSD_CODE.read_text().strip()).resolve()  # match HOOKS_DIR, avoid exec loops
        except OSError:
            want = HOOKS_DIR
        if want != HOOKS_DIR and code_sig(want) is None:
            want = HOOKS_DIR  # e.g. an old plugin version that has since been removed
        if want != HOOKS_DIR or code_sig(HOOKS_DIR) != sig:
            log(f"focusd: code changed, restarting from {want}")
            lockf.close()  # releases the flock; the new image takes it again
            os.execv(sys.executable, [sys.executable, str(want / "mood_hook.py"), "focusd"])
        sessions = live_sessions()
        if not sessions:
            idle_since = idle_since or time.time()
            if time.time() - idle_since > 300:
                return
        else:
            idle_since = None
        sid = resolve_focus(sessions) if sessions else None
        now = time.time()
        if sid != last or now - last_write >= FOCUS_HB_S:  # transitions + heartbeats
            with open(FOCUS_LOG, "a") as f:
                f.write(json.dumps({"t": round(now, 2), "sid": sid}) + "\n")
            last, last_write = sid, now
            if FOCUS_LOG.stat().st_size > 2_000_000:
                keep = FOCUS_LOG.read_text().splitlines()[-5000:]
                FOCUS_LOG.write_text("\n".join(keep) + "\n")
        time.sleep(1)


def spawn_focusd():
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    FOCUSD_CODE.write_text(str(HOOKS_DIR))  # a running focusd switches to this code
    subprocess.Popen([sys.executable, str(HOOKS_DIR / "mood_hook.py"), "focusd"],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True)


def focus_timeline(since):
    """[(t, sid)] transitions covering [since, now]."""
    try:
        lines = FOCUS_LOG.read_text().splitlines()[-2000:]
    except OSError:
        return []
    tl = [(d["t"], d["sid"]) for d in map(json.loads, lines)]
    before = [x for x in tl if x[0] <= since]
    return (before[-1:] if before else []) + [x for x in tl if x[0] > since]


def focused_at(tl, t, margin=1.0):
    """Session focused throughout [t - margin, t], else None.

    None near transitions (ambiguous) and when focusd's heartbeat is stale (it may have died).
    """
    cur, seen = None, None
    for tt, sid in tl:
        if tt > t:
            break
        if sid != cur:
            if tt > t - margin:
                return None
            cur = sid
        seen = tt
    if seen is None or t - seen > FOCUS_HB_MAX:
        return None
    return cur


def current_focus():
    now = time.time()
    return focused_at(focus_timeline(now - 60), now, margin=0)


# ---------------------------------------------------------------- scoring

def owned_events(samples, t0, t1, sid=None, tl=None):
    """(capture_start, event) for events captured in [t0, t1] while `sid` was in focus."""
    out = []
    for s in samples:
        for e in s["ev"]:
            a, b = e.get("t0", s["t"] - 1), e.get("t1", s["t"])
            if t0 <= b <= t1 and (sid is None or focused_at(tl, b, margin=b - a + 1) == sid):
                out.append((a, e))
    return out


def owned_samples(samples, sid, tl):
    return [s for s in samples if focused_at(tl, s["t"]) == sid]


def episodes(evs, pol):
    """Group correlated detections (one groan -> AST + tone + words) into bounded episodes."""
    evs = sorted(((t, e) for t, e in evs if e["pol"] == pol), key=lambda x: x[0])
    eps, cur = [], None
    for t, e in evs:
        w = (1.5 if e["what"] in SLAMS else EVENT_WEIGHT.get(e["src"], 0.6)) * min(1.0, 0.5 + e["p"])
        if cur and t - cur["last"] <= EPISODE_GAP_S and t - cur["start"] <= EPISODE_MAX_S:
            cur["last"], cur["w"] = t, max(cur["w"], w)
            cur["what"].append(e["what"])
        else:
            cur = {"start": t, "last": t, "w": w, "what": [e["what"]]}
            eps.append(cur)
    return eps


def summarize(samples, t0, t1, sid=None, tl=None):
    """Scores over [t0, t1], counting only seconds when `sid` was in focus (if given)."""
    win = [s for s in samples if t0 <= s["t"] <= t1 and not s.get("paused")]
    if sid is not None:
        win = owned_samples(win, sid, tl)
    evs = owned_events(samples, t0, t1, sid, tl)
    seen = [s for s in win if "fj" in s]
    minutes = max(1.0, len(win) / 60)
    out = {"seconds": len(win), "face_s": len(seen)}
    for pol in ("frust", "joy"):
        eps = episodes(evs, pol)
        audio = 1 - math.exp(-K_AUDIO * sum(e["w"] for e in eps) / minutes)
        key, thr = ("ff", FACE_FRUST_THR) if pol == "frust" else ("fj", FACE_JOY_THR)
        face = sum(s[key] > thr for s in seen) / len(seen) if len(seen) >= MIN_FACE_S else 0.0
        out[pol] = round(1 - (1 - audio) * (1 - 0.8 * face), 3)
        out[f"{pol}_face"] = face
        out[f"{pol}_eps"] = eps
    out["bored_eps"] = episodes(evs, "bored")
    out["away_s"] = sum(1 for s in win if s.get("present", 1) == 0)
    return out


def trailing(samples, pred, now, max_gap=3.0, slack=3.0):
    """Seconds pred has held up to now over continuous observations.

    Breaks at gaps > max_gap between samples (e.g. seconds owned by another session) and at more
    than `slack` seconds of pred being false (brief detector dropouts are tolerated).
    """
    start, latest, prev = None, None, now
    for s in reversed(samples):
        if s["t"] > now:
            continue
        if prev - s["t"] > max_gap:
            break
        prev = s["t"]
        if pred(s):
            start = s["t"]
            latest = latest or s["t"]
        elif (start or now) - s["t"] > slack:
            break
    return int(now - start) if start is not None and now - latest <= slack else 0


def on_phone(s):
    return (s.get("present", 0) > 0 and s.get("pitch", 0) < PHONE_PITCH
            and s.get("look_down", 0) > PHONE_LOOK)


def away(s):
    return s.get("present", 1) == 0 and not s.get("paused")


def describe(s, pol):
    parts = []
    face = s[f"{pol}_face"]
    if face >= 0.2:
        parts.append(("frowning/annoyed face" if pol == "frust" else "smiling") + f" {face:.0%} of the time")
    c = Counter(w for e in s[f"{pol}_eps"] for w in dict.fromkeys(e["what"]))
    parts += [w + (f" x{n}" if n > 1 else "") for w, n in c.most_common(6)]
    return "; ".join(parts) or "weak signals only"
