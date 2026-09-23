# claude-mood

Claude Code notices when you laugh, groan, sigh, swear, slam the desk or scowl, and adjusts.
Everything runs locally. No frames, audio or transcripts are stored; only per-second scores and matched keywords.

```
webcam ─ YuNet face ─ HSEmotion (AffectNet, 8 emotions) ─┐
mic ─┬─ AST AudioSet tagger (Laughter, Groan, Sigh, Slam…) ├─► ~/.cache/claude-mood/state.json
     └─ speech? ─ wav2vec2 tone (angry/happy) + Whisper   ─┘        │
                  keyword match ("wtf", "come on", "nice!")         ▼
                                   hooks/mood_hook.py ─► UserPromptSubmit: reaction since your last message
                                                       └► PostToolUse: mid-turn "they just groaned" nudge
```

## Run

```sh
uv run moodd.py            # webcam 0 + default mic. First run downloads roughly 1.5 GB of models
uv run moodd.py --show     # with a debug window showing the face box and the emotion it reads
uv run moodd.py --no-video # audio only (--no-audio, --no-words also exist)
touch ~/.cache/claude-mood/paused   # pause; rm to resume
```

Load the plugin:

```sh
claude --plugin-dir /path/to/claude-mood
# or: /plugin marketplace add /path/to/claude-mood  then  /plugin install claude-mood@claude-mood
```

Status line (optional, in `~/.claude/settings.json`):

```json
"statusLine": { "type": "command", "command": "python3 /path/to/claude-mood/hooks/mood_hook.py status" }
```

`python3 hooks/mood_hook.py report` prints what it thinks of you over the last 30s and 5 min.

## Tuning

- `CLAUDE_MOOD_THRESHOLD` (default 0.35): score needed before anything is injected. Nothing is sent when you're neutral.
- Face frustration is measured relative to *your* resting face (40th percentile over the last ~30 min), because resting faces often read as "disgust" or "anger".
- Weights, regexes and AudioSet labels are at the top of `moodd.py` and `hooks/mood_hook.py`.
- macOS: grant camera and microphone permission to the terminal app that runs `moodd.py`.

## Caveats

Affect detection from a webcam is noisy. Concentrating looks a lot like frowning. The injected context tells Claude to treat it as a hint and not to grovel.
