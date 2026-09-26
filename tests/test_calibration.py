"""Daemon baseline calibration: uv run --with numpy python tests/test_calibration.py"""
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ["CLAUDE_MOOD_DIR"] = tempfile.mkdtemp(prefix="moodcal-")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import moodd  # noqa: E402

ok = True


def run(mood, n, **face):
    f = {"joy": 0.05, "frust": 0.2, "top": "Neutral", "pitch": -5.0, "yaw": 0.0, "look_down": 0.1,
         "eyes_closed": 0.1, "jaw_open": 0.0, **face}
    for _ in range(n):
        for _ in range(3):
            mood.add_face(f)
        s = mood.flush()
    return s


def check(name, cond, detail):
    global ok
    ok &= cond
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  -> {detail}"))


m = moodd.Mood()
s = run(m, 10, frust=0.6)  # a resting face that the model reads as annoyed
check("no face frustration reported before calibration", "ff" not in s and s["ff_raw"] > 0.5, s)
s = run(m, 130)  # calibrate: neutral, looking at the screen, camera slightly above
check("calibrated to the user's neutral", abs(s["pitch"]) < 1 and s["ff"] < 0.02, s)
s = run(m, 150, pitch=-35.0, look_down=0.7)
check("150s head-down phone posture is not learned away", s["pitch"] < -25, s)
s = run(m, 150, frust=0.65)
check("150s silent scowl is not learned away", s["ff"] > 0.4, s)
m3 = moodd.Mood()
m3.base = {"ff": None, "pitch": None}
run(m3, 130, frust=0.62)  # a resting face the model reads as fairly annoyed
s = run(m3, 3, frust=0.85)  # a real scowl on top of it
check("high resting face: a scowl still reaches the reaction threshold", s["ff"] >= 0.45, s)
m2 = moodd.Mood()  # restart: calibration persisted
check("calibration persists across restarts", m2.base["pitch"] is not None, m2.base)
sys.exit(0 if ok else 1)
