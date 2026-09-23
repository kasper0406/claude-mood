---
description: Point claude-mood at this session (use when focus can't be detected automatically); "/claude-mood:here off" unpins
allowed-tools: Bash(python3:*)
---
!`python3 "${CLAUDE_PLUGIN_ROOT}/hooks/mood_hook.py" pin "$([ "$ARGUMENTS" = off ] || echo $CLAUDE_CODE_SESSION_ID)"`

Relay the line above to the user verbatim, nothing else.
