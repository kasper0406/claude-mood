"""Scenario tests for the hooks with synthetic sensor data: python3 tests/test_hooks.py"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HOOK = Path(__file__).resolve().parent.parent / "hooks" / "mood_hook.py"
A, B = "sess-aaaa", "sess-bbbb"


class Env:
    def __init__(self):
        self.dir = Path(tempfile.mkdtemp(prefix="moodtest-"))
        (self.dir / "sessions").mkdir()
        self.env = dict(os.environ, CLAUDE_MOOD_DIR=str(self.dir))
        self.env.pop("CLAUDE_MOOD_URL", None)
        self.env["CLAUDE_CODE_ENTRYPOINT"] = "cli"
        self.extra = {}  # extra hook payload fields, e.g. transcript_path

    def state(self, samples, **kw):
        # "updated" follows the newest sample so a pre-written future timeline stays fresh
        upd = max([time.time()] + [x["t"] for x in samples])
        (self.dir / "state.json").write_text(json.dumps({"updated": upd, "samples": samples, **kw}))

    def focus(self, *transitions, alive=True):
        """Focus transitions plus focusd-style heartbeats every 5s (until now+120 if alive)."""
        end = time.time() + 120 if alive else transitions[-1][0] + 1
        recs = []
        for i, (t, sid) in enumerate(transitions):
            nxt = transitions[i + 1][0] if i + 1 < len(transitions) else end
            while t < nxt:
                recs.append({"t": t, "sid": sid})
                t += 5
        recs.sort(key=lambda r: r["t"])
        (self.dir / "focus.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))

    def session(self, sid, **kw):
        rec = {"sid": sid, "gen": 1, "state": "idle", **kw}
        (self.dir / "sessions" / f"{sid}.json").write_text(json.dumps(rec))

    def read(self, sid):
        return json.loads((self.dir / "sessions" / f"{sid}.json").read_text())

    def hook(self, cmd, sid, wait=True, prompt_id="p1"):
        p = subprocess.Popen([sys.executable, str(HOOK), cmd], env=self.env, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        p.stdin.write(json.dumps({"session_id": sid, "hook_event_name": cmd, "prompt_id": prompt_id,
                                  **self.extra}))
        p.stdin.close()
        if not wait:
            return p
        p.wait(20)
        return p.returncode, p.stdout.read(), p.stderr.read()


def samples(n, now=None, **fields):
    now = now or time.time()
    return [{"t": now - n + i, "ev": [], "present": 1.0, "fj": 0.05, "ff": 0.0, "pitch": 0.0,
             "look_down": 0.1, **fields} for i in range(1, n + 1)]


def groan(s, what="Groan", src="sound", pol="frust"):
    s["ev"].append({"pol": pol, "src": src, "what": what, "p": 0.6})


def ctx(out):
    return json.loads(out)["hookSpecificOutput"]["additionalContext"] if out.strip() else ""


results = []


def check(name, cond, detail=""):
    results.append(cond)
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  -> {detail[:300]}"))


def test_frustration_and_rescue():
    e, now = Env(), time.time()
    e.session(A, last_prompt_t=now - 60)
    e.focus((now - 1000, A))
    s = samples(60, now, ff=0.5)
    for i in (10, 25, 40):
        groan(s[i])
    e.state(s)
    _, out, _ = e.hook("prompt", A)
    c = ctx(out)
    check("frustrated window -> correction note", "looks frustrated" in c and "mood-rescue" not in c, c)
    rec = e.read(A)
    rec["last_prompt_t"] = time.time() - 30
    (e.dir / "sessions" / f"{A}.json").write_text(json.dumps(rec))
    s = samples(30, time.time(), ff=0.6)
    groan(s[5])
    groan(s[20])
    e.state(s)
    _, out, _ = e.hook("prompt", A)
    check("second frustrated turn -> rescue suggested", "mood-rescue" in ctx(out), ctx(out))


def test_focus_attribution():
    e, now = Env(), time.time()
    e.session(A, last_prompt_t=now - 60)
    e.focus((now - 1000, B))  # the user was looking at the other session
    s = samples(60, now, ff=0.6)
    for i in (10, 25, 40):
        groan(s[i])
    e.state(s)
    _, out, _ = e.hook("prompt", A)
    check("frustration while another session focused -> silent", out.strip() == "", out)
    e.focus((now - 1000, B), (now - 30, A))
    _, out, _ = e.hook("prompt", A)
    c = ctx(out)
    check("only seconds focused on A are counted", "Over the 29s" in c or "Over the 28s" in c or c == "", c)


def test_single_utterance_is_one_episode():
    e, now = Env(), time.time()
    e.session(A, last_prompt_t=now - 60)
    e.focus((now - 1000, A))
    s = samples(60, now, present=0.0)  # face off-camera: audio only
    for i in (30, 31, 32):
        s[i]["present"] = 0.0
    groan(s[30], "Groan")
    groan(s[31], "angry tone", "voice")
    groan(s[32], 'said "come on"', "words")
    e.state(s)
    _, out, _ = e.hook("prompt", A)
    c = ctx(out)
    check("groan+tone+words from one utterance stays below the frustration bar", "looks frustrated" not in c, c)
    groan(s[5], "Sigh")
    groan(s[50], 'said "wtf"', "words")
    e.state(s)
    e.session(A, last_prompt_t=now - 60)
    _, out, _ = e.hook("prompt", A)
    check("three separate outbursts in a minute -> frustrated", "looks frustrated" in ctx(out), ctx(out))


def test_pleased_ack_once():
    e, now = Env(), time.time()
    e.session(A, last_prompt_t=now - 40, rescue=True)
    e.focus((now - 1000, A))
    s = samples(40, now, fj=0.8)
    groan(s[20], "Laughter", pol="joy")
    e.state(s)
    _, out, _ = e.hook("prompt", A)
    c = ctx(out)
    check("pleased -> acknowledge + leaner + end rescue", "looks pleased" in c and "Stop any rescue" in c, c)
    _, out, _ = e.hook("prompt", A)
    check("same positive episode is not acknowledged twice", "looks pleased" not in ctx(out), ctx(out))


def test_comedy_breaker():
    e, now = Env(), time.time()
    e.session(A, last_prompt_t=now - 60, last_joke_t=now - 50)
    e.focus((now - 1000, A))
    s = samples(60, now)
    groan(s[20])
    groan(s[35], "Sigh")
    e.state(s)
    _, out, _ = e.hook("prompt", A)
    rec = e.read(A)
    check("groan after a joke disables jokes", rec.get("jokes_off_until", 0) > now + 1000
          and "mood-rescue" not in ctx(out), ctx(out))


def test_phone_while_working():
    e, now = Env(), time.time()
    e.session(A, state="working")
    e.focus((now - 1000, A))
    e.state(samples(120, now, pitch=-30.0, look_down=0.7))
    _, out, _ = e.hook("batch", A)
    c = ctx(out)
    check("phone while working -> status + joke", "on their phone" in c and "joke" in c, c)
    _, out, _ = e.hook("batch", A)
    check("only once per turn", out.strip() == "", out)


def test_idle_roast_wake_and_no_rearm():
    e, now = Env(), time.time()
    e.session(A, state="working", gen=5)
    e.focus((now - 1000, A))
    e.state(samples(230, now + 30, pitch=-30.0, look_down=0.7))
    e.hook("stop", A)
    rc, _, err = e.hook("watch", A)
    check("idle + phone 90s -> wake (exit 2) with roast", rc == 2 and "doomscrolling" in err, f"{rc} {err}")
    rec = e.read(A)
    check("wake claimed atomically", rec["wake_gen"] == rec["gen"] and rec["state"] == "working", str(rec))
    _, out, _ = e.hook("stop", A)
    rc, _, err = e.hook("watch", A)
    check("Stop after the wake turn does not re-arm or notify", rc == 0 and out.strip() == "", f"{rc} {out}")


def test_reaction_wake():
    e, now = Env(), time.time()
    e.session(A, state="working", gen=5)
    e.focus((now - 1000, A))
    e.state(samples(60, now) + samples(40, now + 40, ff=0.6)[1:])  # calm, then scowls at the answer
    e.hook("stop", A)
    rc, _, err = e.hook("watch", A)
    rec = e.read(A)
    check("scowling at the answer wakes Claude to re-check it", rc == 2 and "re-read your last answer"
          in err.lower() and rec.get("frust_turns") == 1, f"{rc} {err[:200]} {rec}")
    _, out, _ = e.hook("stop", A)
    rc, _, _ = e.hook("watch", A)
    check("no reaction wake after the woken turn (no loop)", rc == 0 and out.strip() == "", f"{rc} {out}")
    e2 = Env()
    e2.session(A, state="working", gen=5)
    e2.focus((now - 1000, A))
    e2.state(samples(60, now) + samples(40, now + 40, ff=0.6)[1:])
    e2.env["CLAUDE_MOOD_REACT"] = "0"
    rc = watcher_outcome(e2, lambda: time.sleep(12), wait_before=0, timeout=1)
    check("CLAUDE_MOOD_REACT=0 disables the reaction wake", rc == "timeout", str(rc))


def react_env(later):
    """Calm until the turn ends, then `later` (40 future seconds of samples) as the reaction."""
    e, now = Env(), time.time()
    e.session(A, state="working", gen=5)
    e.focus((now - 1000, A))
    e.state(samples(60, now) + later(samples(40, now + 40)))
    return e


def test_transcript_bookkeeping_vs_activity():
    """Claude Code appends ai-title/cost-state/... after a turn; only real messages cancel the watcher."""
    def groan_later(smp):
        groan(smp[8])  # ~9s after the turn ends
        return smp

    for name, lines, want in [
            ("bookkeeping after the turn doesn't cancel the reaction wake",
             '{"type": "ai-title", "aiTitle": "x"}\n{"type": "cost-state"}\n{"type": "last-prompt"}\n', 2),
            ("a new user message does cancel it", '{"type": "user", "message": {"content": "hi"}}\n', 0)]:
        e = react_env(groan_later)
        tr = e.dir / "transcript.jsonl"
        tr.write_text('{"type": "user"}\n{"type": "assistant"}\n')
        e.extra["transcript_path"] = str(tr)

        def act(lines=lines, tr=tr):
            time.sleep(3)  # after the watcher's snapshot
            with open(tr, "a") as f:
                f.write(lines)
            time.sleep(9)
        rc = watcher_outcome(e, act, wait_before=0, timeout=3)
        log = (e.dir / "hooks.log").read_text() if (e.dir / "hooks.log").exists() else ""
        check(name, rc == want and (want == 2 or "conversation moved" in log), f"{rc} {log[-200:]}")


def test_short_reactions():
    def scowl(n, ff):
        def f(smp):
            for x in smp[5:5 + n]:  # ~5s after the turn ends, while reading the answer
                x["ff"] = ff
            return smp
        return f

    def groan_at(smp):
        groan(smp[5])
        return smp

    for name, later, want in [("a 2s strong scowl", scowl(2, 0.6), 2), ("a single groan", groan_at, 2),
                              ("a 1s flinch", scowl(1, 0.6), "timeout"),
                              ("a long mild frown (concentrating)", scowl(20, 0.35), "timeout")]:
        e = react_env(later)
        rc = watcher_outcome(e, lambda: time.sleep(12), wait_before=0, timeout=1)
        check(f"reaction wake on {name}: {'wakes' if want == 2 else 'stays quiet'}", rc == want, str(rc))


def test_watch_cancelled_by_prompt():
    e, now = Env(), time.time()
    e.session(A, state="working")
    e.focus((now - 1000, A))
    e.state(samples(60, now))
    e.hook("stop", A)
    p = e.hook("watch", A, wait=False)
    time.sleep(3)
    e.hook("prompt", A)
    try:
        rc = p.wait(8)
    except subprocess.TimeoutExpired:
        p.kill()
        rc = "timeout"
    check("new prompt cancels the idle watcher", rc == 0, str(rc))


def test_joke_cooldown_blocks_roast():
    e, now = Env(), time.time()
    e.session(A, state="working", last_joke_t=now - 100)
    e.focus((now - 1000, A))
    e.state(samples(230, now + 30, pitch=-30.0, look_down=0.7))
    e.hook("stop", A)
    p = e.hook("watch", A, wait=False)
    time.sleep(6)
    alive = p.poll() is None
    p.kill()
    check("recent joke -> no roast wake", alive, str(p.returncode))


def test_welcome_back():
    e, now = Env(), time.time()
    e.session(A, state="working")
    e.focus((now - 2000, A))
    away = samples(410, now + 10, present=0.0)  # away since 400s ago, still away for the next 10s
    for s in away:
        s.pop("fj")
    e.state(away)
    e.hook("stop", A)
    p = e.hook("watch", A, wait=False)
    time.sleep(7)
    t = time.time()
    back = [x for x in away if x["t"] < t] + [dict(x, t=t + i) for i, x in enumerate(samples(40, t + 40))]
    e.state(back)
    try:
        rc = p.wait(15)
        err = p.stderr.read()
    except subprocess.TimeoutExpired:
        p.kill()
        rc, err = "timeout", ""
    check("back after 5+ min away -> welcome-back recap wake", rc == 2 and "came back" in err, f"{rc} {err}")


def test_paused_and_status():
    e, now = Env(), time.time()
    e.session(A, last_prompt_t=now - 60)
    e.focus((now - 1000, A))
    s = samples(60, now, ff=0.6, paused=True)
    e.state(s, paused=True)
    _, out, _ = e.hook("prompt", A)
    check("paused -> nothing injected", out.strip() == "", out)
    e.state(samples(60, now, pitch=-30.0, look_down=0.7), swears=3)
    out = subprocess.run([sys.executable, str(HOOK), "status"], env=e.env, capture_output=True, text=True).stdout
    check("status line shows phone + swear jar", "📱" in out and "swear jar 3" in out, out)


def watcher_outcome(e, act, wait_before=3, timeout=10):
    e.hook("stop", A)
    p = e.hook("watch", A, wait=False)
    time.sleep(wait_before)
    act()
    try:
        return p.wait(timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        return "timeout"


def test_delayed_watcher_after_prompt():
    e, now = Env(), time.time()
    e.session(A)
    e.focus((now - 1000, A))
    e.state(samples(230, now + 30, pitch=-30.0, look_down=0.7))
    e.hook("stop", A)
    e.hook("prompt", A)  # user typed before the async watcher got going
    rc, _, _ = e.hook("watch", A)
    rec = e.read(A)
    check("late watcher does not resurrect idle state or fire", rc == 0 and rec["state"] == "working",
          f"{rc} {rec}")


def test_pause_and_focus_loss_cancel_watcher():
    e, now = Env(), time.time()
    e.session(A)
    e.focus((now - 1000, A))
    e.state(samples(60, now))
    rc = watcher_outcome(e, lambda: e.state(samples(60, time.time(), paused=True), paused=True))
    check("pause cancels the idle watcher", rc == 0, str(rc))
    e2 = Env()
    e2.session(A)
    e2.focus((now - 1000, A))
    e2.state(samples(60, now))
    rc = watcher_outcome(e2, lambda: e2.focus((now - 1000, A), (time.time(), B)))
    check("focus loss cancels the idle watcher", rc == 0, str(rc))


def test_delayed_inference_attribution():
    e, now = Env(), time.time()
    e.session(A, last_prompt_t=now - 60)
    e.session(B, last_prompt_t=now - 60)
    switch = now - 20
    e.focus((now - 1000, A), (switch, B))
    s = samples(60, now, present=0.0)
    for k, i in enumerate((50, 52, 55)):  # flushed after the switch, but captured while A was focused
        s[i]["ev"].append({"pol": "frust", "src": "words", "what": f'said "wtf{k}"', "p": 1.0,
                           "t0": switch - 30 + 8 * k, "t1": switch - 25 + 8 * k})
    e.state(s)
    _, out_b, _ = e.hook("prompt", B)
    _, out_a, _ = e.hook("prompt", A)
    check("delayed-inference events go to the session focused at capture time",
          "wtf" not in out_b and "wtf" in out_a, f"A={out_a[:120]} B={out_b[:120]}")


def test_focusd_death():
    e, now = Env(), time.time()
    e.session(A, state="working")
    e.focus((now - 1000, A), alive=False)  # last heartbeat long ago
    e.state(samples(120, now, pitch=-30.0, look_down=0.7))
    _, out, _ = e.hook("batch", A)
    check("stale focus heartbeat -> focus unknown -> silent", out.strip() == "", out)


def test_repeated_and_paused_stops():
    e, now = Env(), time.time()
    e.session(A)
    e.focus((now - 1000, A))
    e.state(samples(60, now, pitch=-30.0, look_down=0.7))
    _, out1, _ = e.hook("stop", A)
    _, out2, _ = e.hook("stop", A)
    check("phone notification on Stop, then cooled down", "Put the phone down" in out1 and out2.strip() == "",
          f"{out1} | {out2}")
    e2 = Env()
    e2.session(A)
    e2.focus((now - 1000, A))
    e2.state(samples(60, now, pitch=-30.0, look_down=0.7, paused=True), paused=True)
    _, out, _ = e2.hook("stop", A)
    check("no notification while paused", out.strip() == "", out)


def test_trailing_gaps_and_x11_pin():
    code = r"""
import sys, time, types
sys.path.insert(0, %r)
import moodlib as ml
now = time.time()
phone = lambda t: {"t": t, "present": 1, "pitch": -30, "look_down": 0.7}
smp = [phone(now - 300 + i) for i in range(30)] + [phone(now - 20 + i) for i in range(21)]
print("trailing", ml.trailing(smp, ml.on_phone, now))
ml.sys = types.SimpleNamespace(platform="linux")
ml.x11_active_pid = lambda: 999
ml.ancestors = lambda pid: [999]
(ml.STATE_DIR / "pinned").write_text("b")
print("focus", ml.resolve_focus([{"sid": "a", "pid": 1}, {"sid": "b", "pid": 2}]))
""" % str(HOOK.parent)
    e = Env()
    out = subprocess.run([sys.executable, "-c", code], env=e.env, capture_output=True, text=True)
    check("phone duration does not bridge an observation gap", "trailing 20" in out.stdout, out.stdout + out.stderr)
    check("ambiguous X11 focus falls back to the pin", "focus b" in out.stdout, out.stdout + out.stderr)


def test_uncalibrated_face_is_ignored():
    e, now = Env(), time.time()
    e.session(A, last_prompt_t=now - 60)
    e.focus((now - 1000, A))
    raw = samples(60, now)
    for s in raw:
        s.pop("ff")  # before calibration moodd only reports ff_raw
        s["ff_raw"] = 0.6
    e.state(raw)
    _, out, _ = e.hook("prompt", A)
    check("uncalibrated resting face is not frustration", out.strip() == "", out)
    e.session(A, last_prompt_t=now - 60)
    mixed = raw[:30] + samples(30, now, ff=0.6)  # calibration finishes halfway, then a real scowl
    e.state(mixed)
    _, out, _ = e.hook("prompt", A)
    check("calibrated scowl still counts after uncalibrated seconds", "frustrated" in ctx(out), out)


def test_streak_needs_consecutive_evidence():
    now = time.time()
    e = Env()
    e.session(A, last_prompt_t=now - 60, frustrated=True, frust_turns=1)
    e.focus((now - 1000, A))
    e.state([])  # moodd just started: no evidence this turn
    e.hook("prompt", A)
    st = e.read(A)
    check("a turn without data ends the frustrated streak", st.get("frust_turns") == 0 and not st.get("frustrated"),
          str(st))
    e.session(A, last_prompt_t=now - 3 * 86400, frustrated=True, frust_turns=1, rescue=True)
    e.state(samples(120, now, ff=0.6))
    _, out, _ = e.hook("prompt", A)
    st = e.read(A)
    check("a frustrated turn after a long break is not 'in a row'",
          "frustrated" in ctx(out) and "second frustrated" not in ctx(out) and st.get("frust_turns") == 1
          and not st.get("rescue"), f"{st} | {out[:200]}")
    e.session(A, last_prompt_t=now - 60, frustrated=True, frust_turns=1)
    _, out, _ = e.hook("prompt", A)
    check("back-to-back frustrated turns still offer the rescue", "second frustrated" in ctx(out), out[:200])


def wait_for(cond, timeout=8.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.2)
    return cond()


def test_focusd_follows_code():
    import fcntl
    import shutil
    e = Env()
    v1, v2 = (e.dir / "v1").resolve(), (e.dir / "v2").resolve()
    for d in (v1, v2):
        shutil.copytree(HOOK.parent, d, ignore=shutil.ignore_patterns("__pycache__"))
    p = subprocess.Popen([sys.executable, str(v1 / "mood_hook.py"), "focusd"], env=e.env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    args = lambda: subprocess.run(["ps", "-o", "args=", "-p", str(p.pid)], capture_output=True,
                                  text=True).stdout
    log = lambda: (e.dir / "hooks.log").read_text() if (e.dir / "hooks.log").exists() else ""
    try:
        wait_for(lambda: (e.dir / "focus.jsonl").exists())
        with open(v1 / "moodlib.py", "a") as f:  # the restarted image must actually run the new code
            f.write('\nif sys.argv[1:] == ["focusd"]:\n    log("focusd-new-code-loaded")\n')
        st = (v1 / "moodlib.py").stat()
        os.utime(v1 / "moodlib.py", ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
        ok = wait_for(lambda: "focusd-new-code-loaded" in log()) and p.poll() is None
        check("focusd re-execs itself when its code changes", ok and str(v1) in args(), log() + args())
        (e.dir / "focusd.code").write_text(str(v2))
        check("focusd switches to a newer plugin dir", wait_for(lambda: str(v2) in args()), log() + args())
        time.sleep(1)
        with open(e.dir / "focusd.lock", "w") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                held = False
            except OSError:
                held = True
        check("restarted focusd still holds the singleton lock", held and p.poll() is None, log())
        shutil.rmtree(v2)  # e.g. a plugin update deletes the version focusd runs from
        time.sleep(2.5)
        check("focusd survives removal of its own plugin dir", p.poll() is None and str(v2) in args(),
              log() + args())
    finally:
        p.kill()
        p.wait()


def test_mac_focus_script_runs():
    """The focus AppleScript must *run*: `front` as a variable compiled fine but failed at runtime."""
    if sys.platform != "darwin":
        return
    sys.path.insert(0, str(HOOK.parent))
    import moodlib
    r = subprocess.run(["osascript", "-e", moodlib.MAC_FOCUS_SCRIPT], capture_output=True, text=True,
                       timeout=10)
    if r.returncode != 0 and any(c in r.stderr for c in ("(-1743)", "(-1713)", "(-600)")):
        print("SKIP focus AppleScript: no Automation permission or GUI session")
        return
    out = r.stdout.strip()
    check("macOS focus AppleScript runs and returns an app or a tty",
          r.returncode == 0 and (out.startswith("app:") or out.startswith("/dev/tty")), r.stderr + out)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print(f"{sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)
