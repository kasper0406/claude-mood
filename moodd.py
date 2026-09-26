#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "opencv-python>=4.8",
#   "onnxruntime>=1.16",
#   "mediapipe>=0.10.14,<=0.10.21",  # later PyPI builds phone home (clearcut); 1.0.x aborts on macOS
#   "sounddevice>=0.4",
#   "soundfile>=0.12",
#   "scipy",
#   "torch>=2.1",
#   "transformers>=4.40",
#   "faster-whisper>=1.0",
# ]
# ///
"""moodd - local mood sensor for Claude Code.

Watches the webcam (face emotion, head pose, gaze, yawns) and listens to the mic
(laughs, groans, sighs, desk slams, tone of voice, swearing). Writes a rolling
per-second log to ~/.cache/claude-mood/state.json and serves it on
http://127.0.0.1:7433/state (forward with `ssh -R 7433:localhost:7433` when
Claude Code runs on another machine). The claude-mood plugin hooks read it.

No frames, audio or transcripts are stored. Derived scores and matched
keywords are, however, passed to Claude as context by the plugin hooks.

    uv run moodd.py                    # webcam 0 + default mic
    uv run moodd.py --no-video         # audio only
    uv run moodd.py --video clip.mp4 --audio clip.wav   # replay files (testing)
    touch ~/.cache/claude-mood/paused  # pause: closes camera and mic
"""
import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

STATE_DIR = Path(os.environ.get("CLAUDE_MOOD_DIR", Path.home() / ".cache" / "claude-mood"))
SR = 16000
HISTORY_S = 900
CALIB_S = 120             # seconds of face data used to calibrate the resting face / head pose
DRIFT_S = 1800            # afterwards the baseline drifts with this time constant...
DRIFT_STEP = 0.05, 2.0    # ...by at most this much per second-sample (frustration/brow, pitch degrees)
BROW_FULL = 0.15          # brows this far below neutral (browDown blendshape) = a full scowl

FACE_MODEL_URL = ("https://github.com/HSE-asavchenko/face-emotion-recognition/raw/main/"
                  "models/affectnet_emotions/onnx/enet_b0_8_best_vgaf.onnx")
LANDMARKER_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/latest/face_landmarker.task")
FACE_CLASSES = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
AST_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
SER_MODEL = "superb/wav2vec2-base-superb-er"

# AudioSet labels that count as joy / frustration.
JOY_SOUNDS = {"Laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle", "Cheering",
              "Whoop", "Applause"}
FRUST_SOUNDS = {"Groan", "Sigh", "Grunt", "Yell", "Shout", "Screaming", "Whimper",
                "Crying, sobbing", "Slam", "Smash, crash", "Battle cry"}
SOUND_THRESH = 0.15
EVENT_COOLDOWN_S = 2.5
YAWN_JAW, YAWN_MIN_S = 0.55, 1.5

JOY_WORDS = re.compile(r"\b(yes{2,}|nice|awesome|perfect|finally|let'?s go|hell yeah|"
                       r"beautiful|love it|amazing|brilliant|sweet|woo+|ha(?:ha)+)\b", re.I)
FRUST_WORDS = re.compile(r"\b(fuck\w*|shit\w*|damn\w*|goddamn\w*|wtf|crap|bollocks|"
                         r"a+rgh+|u+gh+|come on|seriously|what the|no no|"
                         r"why (?:is|does|would|won't|isn't|did)|are you kidding)\b", re.I)
SWEARS = re.compile(r"^(fuck|shit|damn|goddamn|wtf|crap|bollocks)", re.I)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


def paused():
    return (STATE_DIR / "paused").exists()


def fetch(url):
    path = STATE_DIR / "models" / url.rsplit("/", 1)[1]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        log(f"downloading {path.name}")
        urllib.request.urlretrieve(url, path)
    return str(path)


class Mood:
    """Thread-safe accumulator; flushed to one sample per second."""

    def __init__(self):
        self.lock = threading.Lock()
        self.face = []           # per-frame dicts, or None (no face), since last flush
        self.events = []         # dicts since last flush
        self.last_event = {}     # label -> t, for dedup of overlapping windows
        self.last_frust_event = 0.0
        self.samples = deque(maxlen=HISTORY_S)
        self.calib = {"ff": [], "pitch": [], "brow": []}
        self.base = {"ff": None, "pitch": None, "brow": None}
        try:
            self.base.update(json.loads((STATE_DIR / "calibration.json").read_text()))
        except (OSError, ValueError):
            pass
        self.swears = 0
        self.json = b"{}"

    def add_face(self, v):
        with self.lock:
            self.face.append(v)

    def add_event(self, pol, src, what, score, t0, t1):
        """t0..t1 = when the evidence was captured (inference can lag by seconds)."""
        now = time.time()
        with self.lock:
            if now - self.last_event.get(what, 0) < EVENT_COOLDOWN_S:
                return
            self.last_event[what] = now
            if pol == "frust":
                self.last_frust_event = now
            if src == "words" and SWEARS.match(what.split('"')[1]):
                self.swears += 1
            self.events.append({"pol": pol, "src": src, "what": what, "p": round(score, 2),
                                "t0": round(t0, 2), "t1": round(t1, 2)})
        log(f"event {pol:5s} {src:6s} {what} ({score:.2f})")

    def flush(self):
        now = time.time()
        with self.lock:
            face, self.face = self.face, []
            events, self.events = self.events, []
            calm = now - self.last_frust_event > 10
        s = {"t": round(now, 2), "ev": events}
        if paused():
            s["paused"] = True
        seen = [f for f in face if f is not None]
        if face:
            s["present"] = round(len(seen) / len(face), 2)
        if seen:
            def mean(k):
                return float(np.mean([f[k] for f in seen]))
            ff, pitch = mean("frust"), mean("pitch")
            # Resting faces often read as mildly angry/sad and cameras sit at odd angles, so both are
            # measured against the user's own neutral: calibrated over the first CALIB_S calm seconds
            # looking at the screen, then drifting slowly with bounded steps so a sustained scowl or a
            # long phone session can't be learned away.
            screen = mean("look_down") < 0.35
            brow = mean("brow")
            base = self.update_base("ff", ff, calm, 40)
            bbase = self.update_base("brow", brow, calm and screen, 50)
            pbase = self.update_base("pitch", pitch, screen, 50)
            pbase = 0.0 if pbase is None else pbase
            s.update(fj=round(mean("joy"), 3), ff_raw=round(ff, 3), brow=round(brow, 3),
                     pitch=round(pitch - pbase, 1), yaw=round(mean("yaw"), 1),
                     look_down=round(mean("look_down"), 2), eyes_closed=round(mean("eyes_closed"), 2),
                     top=max(set(f["top"] for f in seen), key=[f["top"] for f in seen].count))
            if base is not None and bbase is not None:  # uncalibrated, it's mostly the resting face
                # share of the headroom above the user's neutral: a resting face at 0.6 can still reach 1.0
                ff = max(0.0, ff - base) / max(0.05, 1.0 - base)
                # The emotion model can call a neutral face "Anger" for minutes; a real scowl also
                # lowers the brows, so frustration only counts as far as the brows actually drop.
                s["ff"] = round(ff * min(1.0, max(0.0, brow - bbase) / BROW_FULL), 3)
        self.samples.append(s)
        return s

    def update_base(self, key, v, ok, pct):
        """Returns the baseline before this sample; learns from v only when ok (neutral-looking)."""
        base = self.base[key]
        if not ok:
            return base
        if base is None:
            self.calib[key].append(v)
            if len(self.calib[key]) >= CALIB_S:
                self.base[key] = float(np.percentile(self.calib[key], pct))
                log(f"calibrated {key} baseline: {self.base[key]:.2f}")
                self.save_base()
            return None
        step = DRIFT_STEP[key == "pitch"]
        self.base[key] = base + float(np.clip(v - base, -step, step)) / DRIFT_S
        return base

    def save_base(self):
        (STATE_DIR / "calibration.json").write_text(json.dumps(self.base))

    def write(self):
        data = json.dumps({"updated": time.time(), "pid": os.getpid(), "paused": paused(),
                           "swears": self.swears, "samples": list(self.samples)})
        self.json = data.encode()
        tmp = STATE_DIR / "state.json.tmp"
        tmp.write_text(data)
        tmp.replace(STATE_DIR / "state.json")


def serve(mood, port):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = mood.json if self.path.startswith("/state") else b"{}"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


# ---------------------------------------------------------------- video

def emotion_model():
    import onnxruntime as ort
    sess = ort.InferenceSession(fetch(FACE_MODEL_URL), providers=["CPUExecutionProvider"])
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])

    def predict(rgb):
        import cv2
        x = (cv2.resize(rgb, (224, 224)) / 255.0 - mean) / std
        logits = sess.run(None, {"input": x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0]
        p = np.exp(logits - logits.max())
        return dict(zip(FACE_CLASSES, (p / p.sum()).tolist()))
    return predict


def landmarker():
    from mediapipe.tasks.python import BaseOptions, vision
    return vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=fetch(LANDMARKER_URL)),
        output_face_blendshapes=True, output_facial_transformation_matrixes=True,
        num_faces=1, running_mode=vision.RunningMode.VIDEO))


def analyze_face(frame, lm, predict, t_ms):
    """One frame -> dict of face signals, or None if no face."""
    import cv2
    import mediapipe as mp
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    r = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), t_ms)
    if not r.face_landmarks:
        return None
    h, w = rgb.shape[:2]
    xs = [p.x * w for p in r.face_landmarks[0]]
    ys = [p.y * h for p in r.face_landmarks[0]]
    x0, x1, y0, y1 = int(min(xs)), int(max(xs)), int(min(ys)), int(max(ys))
    if x1 - x0 < 40:
        return None
    m = int(0.1 * (x1 - x0))
    crop = rgb[max(0, y0 - m):y1 + m, max(0, x0 - m):x1 + m]
    p = predict(crop)
    bs = {c.category_name: c.score for c in r.face_blendshapes[0]}
    R = np.array(r.facial_transformation_matrixes[0])[:3, :3]
    smile = (bs["mouthSmileLeft"] + bs["mouthSmileRight"]) / 2
    return {
        "joy": max(p["Happiness"], smile),
        "frust": p["Anger"] + p["Disgust"] + p["Contempt"] + 0.5 * p["Sadness"],
        "brow": (bs["browDownLeft"] + bs["browDownRight"]) / 2,
        "top": max(p, key=p.get),
        "pitch": float(np.degrees(np.arctan2(-R[2, 1], R[2, 2]))),  # negative = looking down
        "yaw": float(np.degrees(np.arcsin(np.clip(R[2, 0], -1, 1)))),
        "look_down": (bs["eyeLookDownLeft"] + bs["eyeLookDownRight"]) / 2,
        "eyes_closed": (bs["eyeBlinkLeft"] + bs["eyeBlinkRight"]) / 2,
        "jaw_open": bs["jawOpen"],
        "box": (x0, y0, x1, y1),
    }


def video_loop(mood, src, fps, preview, stop):
    import cv2
    predict, lm = emotion_model(), landmarker()
    live = src.isdigit()
    cap, t_start, yawn_since = None, time.time(), None
    log(f"video: {src}")
    while not stop.is_set():
        t0 = time.time()
        if paused():
            if cap is not None:
                cap.release()
                cap = None
                log("video paused (camera closed)")
            time.sleep(1)
            continue
        if cap is None:
            cap = cv2.VideoCapture(int(src) if live else src)
            if not cap.isOpened():
                log(f"could not open video source {src!r} (camera permission?)")
                cap = None
                time.sleep(5)
                continue
        ok, frame = cap.read()
        if not ok:
            if live:
                time.sleep(0.5)
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        f = analyze_face(frame, lm, predict, int((t0 - t_start) * 1000))
        mood.add_face(f)
        if f:
            if f["jaw_open"] > YAWN_JAW and f["eyes_closed"] > 0.3:
                yawn_since = yawn_since or t0
                if t0 - yawn_since >= YAWN_MIN_S:
                    mood.add_event("bored", "face", "Yawn", f["jaw_open"], yawn_since, t0)
                    yawn_since = None
            else:
                yawn_since = None
        if preview is not None:
            if f:
                x0, y0, x1, y1 = f["box"]
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(frame, f"{f['top']} joy={f['joy']:.2f} frust={f['frust']:.2f} "
                            f"pitch={f['pitch']:.0f} down={f['look_down']:.2f}", (x0, max(15, y0 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            preview["frame"] = frame  # shown by main(): macOS only allows GUI calls on the main thread
        time.sleep(max(0.0, 1 / fps - (time.time() - t0)))


# ---------------------------------------------------------------- audio

class AudioModels:
    def __init__(self, words):
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        self.torch = torch
        self.dev = ("cuda" if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available() else "cpu")
        log(f"loading audio models on {self.dev}")
        self.ast_fe = AutoFeatureExtractor.from_pretrained(AST_MODEL)
        self.ast = AutoModelForAudioClassification.from_pretrained(AST_MODEL).to(self.dev).eval()
        self.ast_labels = self.ast.config.id2label
        idx = {v: k for k, v in self.ast_labels.items()}
        missing = (JOY_SOUNDS | FRUST_SOUNDS) - idx.keys()
        if missing:
            log(f"note: AudioSet labels not in model: {sorted(missing)}")
        self.ser_fe = AutoFeatureExtractor.from_pretrained(SER_MODEL)
        self.ser = AutoModelForAudioClassification.from_pretrained(SER_MODEL).to(self.dev).eval()
        self.whisper = None
        if words:
            from faster_whisper import WhisperModel
            self.whisper = WhisperModel("base.en", device="cuda" if self.dev == "cuda" else "cpu",
                                        compute_type="int8")

    def sounds(self, x):
        with self.torch.inference_mode():
            f = self.ast_fe(x, sampling_rate=SR, return_tensors="pt").to(self.dev)
            p = self.torch.sigmoid(self.ast(**f).logits[0]).cpu().numpy()
        return {self.ast_labels[i]: float(v) for i, v in enumerate(p)}

    def tone(self, x):
        with self.torch.inference_mode():
            f = self.ser_fe(x, sampling_rate=SR, return_tensors="pt").to(self.dev)
            p = self.torch.softmax(self.ser(**f).logits[0], -1).cpu().numpy()
        return {self.ser.config.id2label[i]: float(v) for i, v in enumerate(p)}

    def transcribe(self, x):
        segs, _ = self.whisper.transcribe(x, language="en", beam_size=1)
        return " ".join(s.text for s in segs).strip()


class Ring:
    def __init__(self, seconds):
        self.buf = np.zeros(int(seconds * SR), np.float32)
        self.lock = threading.Lock()
        self.total = 0

    def push(self, x):
        with self.lock:
            n = min(len(x), len(self.buf))
            self.buf = np.roll(self.buf, -n)
            self.buf[-n:] = x[-n:]
            self.total += len(x)

    def clear(self):
        with self.lock:
            self.buf[:] = 0

    def last(self, seconds):
        with self.lock:
            return self.buf[-int(seconds * SR):].copy(), self.total


def audio_capture(ring, src, stop):
    if src is None:
        import sounddevice as sd
        log(f"audio: default mic ({sd.query_devices(kind='input')['name']})")
        while not stop.is_set():
            if paused():
                time.sleep(1)
                continue
            with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                                callback=lambda d, *_: ring.push(d[:, 0].copy())):
                while not stop.is_set() and not paused():
                    time.sleep(0.5)
            ring.clear()
            log("audio paused (mic closed)")
        return
    import soundfile as sf
    from scipy.signal import resample_poly
    x, sr = sf.read(src, dtype="float32", always_2d=True)
    x = resample_poly(x.mean(1), SR, sr).astype(np.float32)
    log(f"audio: replaying {src} ({len(x) / SR:.1f}s, looped)")
    chunk = SR // 10
    while not stop.is_set():
        for i in range(0, len(x), chunk):
            if stop.is_set():
                return
            if not paused():
                ring.push(x[i:i + chunk])
            time.sleep(0.1)
        ring.push(np.zeros(SR * 2, np.float32))  # gap between loops
        time.sleep(2)


def audio_loop(mood, models, ring, stop, hop=1.0, win=2.0, min_dbfs=-45.0):
    speech, span = [], [0.0, 0.0]   # speech chunks and their capture interval
    last_total = 0

    def flush_speech():
        x = np.concatenate(speech)
        speech.clear()
        t0, t1 = span
        if len(x) < SR * 0.6:
            return
        tone = models.tone(x)
        if tone.get("ang", 0) > 0.5:
            mood.add_event("frust", "voice", "angry tone", tone["ang"], t0, t1)
        if tone.get("hap", 0) > 0.5:
            mood.add_event("joy", "voice", "happy tone", tone["hap"], t0, t1)
        if models.whisper:
            text = models.transcribe(x)
            for m in JOY_WORDS.finditer(text):
                mood.add_event("joy", "words", f'said "{m.group(0).lower()}"', 1.0, t0, t1)
            for m in FRUST_WORDS.finditer(text):
                mood.add_event("frust", "words", f'said "{m.group(0).lower()}"', 1.0, t0, t1)

    while not stop.is_set():
        time.sleep(hop)
        if paused():
            speech.clear()
            continue
        x, total = ring.last(win)
        t_cap = time.time()
        new = min(total - last_total, len(x))
        last_total = total
        if new <= 0:
            continue
        dbfs = 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)
        if dbfs < min_dbfs:
            if speech:
                flush_speech()
            continue
        p = models.sounds(x)
        for label in JOY_SOUNDS | FRUST_SOUNDS:
            if p.get(label, 0) > SOUND_THRESH:
                mood.add_event("joy" if label in JOY_SOUNDS else "frust", "sound", label, p[label],
                               t_cap - win, t_cap)
        if p.get("Speech", 0) > 0.3:
            if not speech:
                span[0] = t_cap - new / SR
            span[1] = t_cap
            speech.append(x[-new:])
            if sum(map(len, speech)) > 8 * SR:
                flush_speech()
        elif speech:
            flush_speech()


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default="0", help="camera index or video file (default 0)")
    ap.add_argument("--audio", default=None, help="audio file to replay instead of the mic")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--no-words", action="store_true", help="skip whisper swear/cheer detection")
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--port", type=int, default=7433, help="HTTP port for /state (0 = off)")
    ap.add_argument("--show", action="store_true", help="debug window with face box")
    ap.add_argument("--verbose", action="store_true", help="print every per-second sample")
    args = ap.parse_args()

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    mood, stop = Mood(), threading.Event()
    preview = {} if args.show and not args.no_video else None
    threads = []
    if args.port:
        threads.append(threading.Thread(target=serve, args=(mood, args.port), daemon=True))
    if not args.no_video:
        threads.append(threading.Thread(target=video_loop, daemon=True,
                                        args=(mood, args.video, args.fps, preview, stop)))
    if not args.no_audio:
        models = AudioModels(words=not args.no_words)
        ring = Ring(10)
        threads += [threading.Thread(target=audio_capture, args=(ring, args.audio, stop), daemon=True),
                    threading.Thread(target=audio_loop, args=(mood, models, ring, stop), daemon=True)]
    for t in threads:
        t.start()
    log(f"writing {STATE_DIR / 'state.json'}" + (f", serving :{args.port}/state" if args.port else "")
        + "; Ctrl-C to stop")
    try:
        while True:
            if preview is None:
                time.sleep(1)
            else:
                import cv2
                deadline = time.time() + 1
                while time.time() < deadline:
                    frame = preview.pop("frame", None)
                    if frame is not None:
                        cv2.imshow("moodd", frame)
                        preview["shown"] = True
                    if preview.get("shown"):
                        cv2.waitKey(30)
                    else:  # waitKey returns at once without a window on some backends (Qt)
                        time.sleep(0.03)
            s = mood.flush()
            mood.write()
            if args.verbose:
                log(json.dumps(s))
    except KeyboardInterrupt:
        stop.set()
        if mood.base["ff"] is not None or mood.base["pitch"] is not None:
            mood.save_base()
        (STATE_DIR / "state.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
