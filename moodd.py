#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "opencv-python>=4.8",
#   "onnxruntime>=1.16",
#   "sounddevice>=0.4",
#   "soundfile>=0.12",
#   "scipy",
#   "torch>=2.1",
#   "transformers>=4.40",
#   "faster-whisper>=1.0",
# ]
# ///
"""moodd - local mood sensor for Claude Code.

Watches the webcam (face emotion) and listens to the mic (laughs, groans,
sighs, desk slams, tone of voice, swearing) and writes a rolling per-second
log to ~/.cache/claude-mood/state.json. The claude-mood plugin hooks read it.

Nothing leaves the machine; no frames or audio are stored, only scores and
matched keywords.

    uv run moodd.py                    # webcam 0 + default mic
    uv run moodd.py --no-video         # audio only
    uv run moodd.py --video clip.mp4 --audio clip.wav   # replay files (testing)
    touch ~/.cache/claude-mood/paused  # pause sensing
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
from pathlib import Path

import numpy as np

STATE_DIR = Path(os.environ.get("CLAUDE_MOOD_DIR", Path.home() / ".cache" / "claude-mood"))
SR = 16000
HISTORY_S = 600

FACE_MODEL_URL = ("https://github.com/HSE-asavchenko/face-emotion-recognition/raw/main/"
                  "models/affectnet_emotions/onnx/enet_b0_8_best_vgaf.onnx")
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")
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

JOY_WORDS = re.compile(r"\b(yes{2,}|yess*!|nice|awesome|perfect|finally|let'?s go|hell yeah|"
                       r"beautiful|love it|amazing|brilliant|sweet|woo+|ha(?:ha)+)\b", re.I)
FRUST_WORDS = re.compile(r"\b(fuck\w*|shit\w*|damn\w*|goddamn\w*|wtf|crap|bollocks|"
                         r"a+rgh+|u+gh+|come on|seriously|what the|no no|stop|"
                         r"why (?:is|does|would|won't|isn't|did)|are you kidding)\b", re.I)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


def paused():
    return (STATE_DIR / "paused").exists()


class Mood:
    """Thread-safe accumulator; flushed to one sample per second."""

    def __init__(self):
        self.lock = threading.Lock()
        self.face = []           # (joy, frust, top_label) or None (no face) since last flush
        self.events = []         # dicts since last flush
        self.last_event = {}     # label -> t, for dedup of overlapping windows
        self.samples = deque(maxlen=HISTORY_S)
        self.baseline = deque(maxlen=1800)  # recent raw face frustration, for per-user resting face

    def add_face(self, v):
        with self.lock:
            self.face.append(v)

    def add_event(self, pol, src, what, score):
        now = time.time()
        with self.lock:
            if now - self.last_event.get(what, 0) < EVENT_COOLDOWN_S:
                return
            self.last_event[what] = now
            self.events.append({"pol": pol, "src": src, "what": what, "p": round(score, 2)})
        log(f"event {pol:5s} {src:6s} {what} ({score:.2f})")

    def flush(self):
        with self.lock:
            face, self.face = self.face, []
            events, self.events = self.events, []
        s = {"t": round(time.time(), 2), "ev": events}
        seen = [f for f in face if f is not None]
        if face:
            s["present"] = len(seen) / len(face)
        if seen:
            fj = float(np.mean([f[0] for f in seen]))
            ff = float(np.mean([f[1] for f in seen]))
            self.baseline.append(ff)
            # Resting faces often read as mildly angry/sad; subtract the user's own typical level.
            base = float(np.percentile(self.baseline, 40)) if len(self.baseline) >= 30 else 0.0
            s.update(fj=round(fj, 3), ff_raw=round(ff, 3), ff=round(max(0.0, ff - base), 3),
                     top=max(set(f[2] for f in seen), key=[f[2] for f in seen].count))
        self.samples.append(s)
        return s

    def write(self):
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = STATE_DIR / "state.json.tmp"
        tmp.write_text(json.dumps({"updated": time.time(), "pid": os.getpid(),
                                   "samples": list(self.samples)}))
        tmp.replace(STATE_DIR / "state.json")


# ---------------------------------------------------------------- video

def fetch(url):
    path = STATE_DIR / "models" / url.rsplit("/", 1)[1]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        log(f"downloading {path.name}")
        urllib.request.urlretrieve(url, path)
    return str(path)


def face_model():
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


def video_loop(mood, src, fps, show, stop):
    import cv2
    predict = face_model()
    det = cv2.FaceDetectorYN.create(fetch(YUNET_URL), "", (320, 320), 0.8)
    live = src.isdigit()
    cap = cv2.VideoCapture(int(src) if live else src)
    if not cap.isOpened():
        log(f"could not open video source {src!r} (camera permission?)")
        return
    log(f"video: {src}")
    while not stop.is_set():
        t0 = time.time()
        if paused():
            time.sleep(1)
            continue
        ok, frame = cap.read()
        if not ok:
            if live:
                time.sleep(0.5)
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        det.setInputSize((frame.shape[1], frame.shape[0]))
        faces = det.detect(frame)[1]
        faces = [f[:4].astype(int) for f in faces if f[2] >= 60] if faces is not None else []
        if faces:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            x, y = max(0, x), max(0, y)
            m = int(0.1 * w)
            crop = frame[max(0, y - m):y + h + m, max(0, x - m):x + w + m]
            p = predict(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            joy = p["Happiness"]
            frust = p["Anger"] + p["Disgust"] + p["Contempt"] + 0.5 * p["Sadness"]
            top = max(p, key=p.get)
            mood.add_face((joy, frust, top))
            if show:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{top} joy={joy:.2f} frust={frust:.2f}", (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            mood.add_face(None)
        if show:
            cv2.imshow("moodd", frame)
            cv2.waitKey(1)
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
        self.speech_idx = idx["Speech"]
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

    def last(self, seconds):
        with self.lock:
            return self.buf[-int(seconds * SR):].copy(), self.total


def audio_capture(ring, src, stop):
    if src is None:
        import sounddevice as sd
        log(f"audio: default mic ({sd.query_devices(kind='input')['name']})")
        with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                            callback=lambda d, *_: ring.push(d[:, 0].copy())):
            stop.wait()
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
            ring.push(x[i:i + chunk])
            time.sleep(0.1)
        ring.push(np.zeros(SR * 2, np.float32))  # gap between loops
        time.sleep(2)


def audio_loop(mood, models, ring, stop, hop=1.0, win=2.0, min_dbfs=-45.0):
    speech = []
    last_total = 0

    def flush_speech():
        x = np.concatenate(speech)
        speech.clear()
        if len(x) < SR * 0.6:
            return
        tone = models.tone(x)
        if tone.get("ang", 0) > 0.5:
            mood.add_event("frust", "voice", "angry tone", tone["ang"])
        if tone.get("hap", 0) > 0.5:
            mood.add_event("joy", "voice", "happy tone", tone["hap"])
        if models.whisper:
            text = models.transcribe(x)
            for m in JOY_WORDS.finditer(text):
                mood.add_event("joy", "words", f'said "{m.group(0).lower()}"', 1.0)
            for m in FRUST_WORDS.finditer(text):
                mood.add_event("frust", "words", f'said "{m.group(0).lower()}"', 1.0)

    while not stop.is_set():
        time.sleep(hop)
        x, total = ring.last(win)
        new = min(total - last_total, len(x))
        last_total = total
        if paused() or new <= 0:
            continue
        dbfs = 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)
        if dbfs < min_dbfs:
            if speech:
                flush_speech()
            continue
        p = models.sounds(x)
        for label in JOY_SOUNDS | FRUST_SOUNDS:
            if p.get(label, 0) > SOUND_THRESH:
                mood.add_event("joy" if label in JOY_SOUNDS else "frust", "sound", label, p[label])
        if p[models.ast_labels[models.speech_idx]] > 0.3:
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
    ap.add_argument("--show", action="store_true", help="debug window with face box")
    ap.add_argument("--verbose", action="store_true", help="print every per-second sample")
    args = ap.parse_args()

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    mood, stop = Mood(), threading.Event()
    threads = []
    if not args.no_video:
        threads.append(threading.Thread(target=video_loop, daemon=True,
                                        args=(mood, args.video, args.fps, args.show, stop)))
    if not args.no_audio:
        models = AudioModels(words=not args.no_words)
        ring = Ring(10)
        threads += [threading.Thread(target=audio_capture, args=(ring, args.audio, stop), daemon=True),
                    threading.Thread(target=audio_loop, args=(mood, models, ring, stop), daemon=True)]
    for t in threads:
        t.start()
    log(f"writing {STATE_DIR / 'state.json'}; Ctrl-C to stop")
    try:
        while True:
            time.sleep(1)
            s = mood.flush()
            mood.write()
            if args.verbose:
                log(json.dumps(s))
    except KeyboardInterrupt:
        stop.set()
        (STATE_DIR / "state.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
