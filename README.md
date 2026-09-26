# claude-mood

Claude Code notices when you laugh, groan, swear, scowl, doomscroll or wander off, and it reacts.
Only the Claude session you're currently looking at is affected.

```
webcam ─ MediaPipe face mesh ─┬─ HSEmotion (AffectNet)  → joy / frustration (vs. your resting face)
                              └─ blendshapes + head pose → looking at phone, yawns, away from desk
mic ─┬─ AST AudioSet tagger → Laughter, Groan, Sigh, Slam…
     └─ speech? → wav2vec2 tone (angry/happy) + Whisper keywords ("wtf", "come on", "nice!")
        │
        ▼  moodd.py: 1 Hz samples → ~/.cache/claude-mood/state.json and http://127.0.0.1:7433/state
hooks (per session, focus-attributed)
  UserPromptSubmit  frustrated → correct course; 2nd frustrated turn in a row → offer mood-rescue subagent
                    pleased → one-line acknowledgement, go leaner; mixed → "laughing at something absurd?"
  PostToolBatch     frustration spike mid-turn → sanity-check the approach
                    on phone / yawning while it works → one-line status update + topical joke
  Stop              on phone → desktop notification "Claude is done. Put the phone down."
  Stop (asyncRewake watcher, interactive only)
                    idle and you've been on your phone for 90s → wakes Claude to roast your doomscrolling
                    back after 5+ min away → wakes Claude for a one-line welcome-back recap
```

## Run

```sh
uv run moodd.py            # webcam 0 + default mic. First run downloads roughly 1.5 GB of models
uv run moodd.py --show     # debug window: face box, emotion, head pitch, gaze
uv run moodd.py --no-video # audio only (also: --no-audio, --no-words)
touch ~/.cache/claude-mood/paused   # pause: closes camera and mic; rm to resume

claude --plugin-dir /path/to/claude-mood
# or: /plugin marketplace add /path/to/claude-mood  then  /plugin install claude-mood@claude-mood
```

The status line is optional. Add it to `~/.claude/settings.json`:

```json
"statusLine": { "type": "command", "command": "python3 /path/to/claude-mood/hooks/mood_hook.py status" }
```
It shows something like `😤 ⛈ joy 0.05 · frust 0.71 · swear jar 4`, or 📱 while you're on your phone. `python3 hooks/mood_hook.py report` prints a longer summary.

## Focus: which session gets the notes

Only the session in focus gets notes. Each reaction is attributed to whichever session was focused at that second, so frowning at session A never shows up in session B. A small per-machine `focusd` (started automatically) resolves focus once a second:

- **tmux** (also works over SSH): the pane shown in a tmux client whose terminal has focus. This needs `set -g focus-events on` in `~/.tmux.conf`.
- **macOS iTerm2 / Terminal.app**: the frontmost app's selected tab, matched by its tty (osascript). Grant Automation permission when asked.
- **X11**: the active window's process must be an ancestor of the Claude process. This works if each terminal window is its own process, but not with a shared terminal server.
- **Anything else** (VS Code, Wayland, etc.): focus can't be determined, so the plugin stays silent. Run `/claude-mood:here` in the session you want to target (`/claude-mood:here off` to unpin).

## Claude Code on another machine

Run `moodd.py` on the machine with the webcam, then forward the port and point the hooks at it:

```sh
ssh -R 7433:localhost:7433 box            # then on the box:
export CLAUDE_MOOD_URL=http://127.0.0.1:7433
```

## What it can and can't change

Hooks can't change Claude Code's model or effort level. I checked this against the 2.1.280 hook schema: `ultrathink` inside hook context doesn't count, and edits to `effortLevel` in settings aren't picked up by a running session. So:

- **Frustrated:** the plugin tells Claude to course-correct and to reason more thoroughly if the approach seems wrong. After two frustrated turns in a row, it suggests handing the stuck sub-problem to the bundled `mood-rescue` subagent, which runs on Opus at high effort and is read-only.
- **Pleased:** "lower effort" means concise output and no optional detours. Required checks are kept, and any rescue escalation ends.

## Tuning and safety valves

- `CLAUDE_MOOD_MODE=polite` turns off roasts and jokes.
- Jokes have a shared 10-minute cooldown. If you groan within a minute of a joke (the comedy circuit breaker), jokes are off for 30 minutes.
- Frustration is scored by episode: a groan, angry tone and "come on" from one outburst count once. A single outburst per minute isn't enough; it takes repeated outbursts or a sustained scowl. Enter and exit thresholds differ (0.45 / 0.25), so it doesn't flicker.
- Face frustration and head pitch are measured against *your* neutral. It's calibrated from your first 2 minutes of calm, looking-at-the-screen seconds, then drifts slowly in bounded steps, so a long scowl or phone session isn't learned away. Until then the face doesn't count towards frustration (many resting faces read as annoyed); only audio does. It's saved in `~/.cache/claude-mood/calibration.json`; delete that file to recalibrate.
- Knobs are at the top of `hooks/moodlib.py`, `hooks/mood_hook.py` and `moodd.py`. Hook activity is logged to `~/.cache/claude-mood/hooks.log`.
- macOS: grant camera and microphone permission to the terminal app that runs `moodd.py`.

## Privacy

Frames, audio and transcripts never leave the daemon and are never written to disk. mediapipe is pinned to 0.10.21 or older because later PyPI builds include an undocumented usage logger that uploads to Google (clearcut, `play.googleapis.com`). Derived scores and matched keywords (e.g. `said "wtf" x2`) *are* sent to Claude, and so to Anthropic, as context in the focused session.

## Caveats

Affect detection from a webcam is noisy. Concentrating looks like frowning, and reading notes looks like checking your phone. Claude is told the notes are hints from a plugin, not statements from you. Thresholds are untuned guesses; expect to adjust them.

## Tests

- `python3 tests/test_hooks.py` runs 28 scenario tests with synthetic sensor data: focus attribution (including delayed inference across a focus switch, and focusd dying), episode merging, escalation, the comedy circuit breaker, phone/idle wakes, and the watcher lifecycle (late start, pause and focus-loss cancellation, no re-arm after a wake).
- `uv run --with numpy python tests/test_calibration.py` checks that calibration doesn't learn away a long phone session or a silent scowl, and that it persists across restarts.
