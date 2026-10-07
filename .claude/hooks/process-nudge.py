#!/usr/bin/env python3
# Stop hook. Nudge only, never blocks: prints one systemMessage when code is newer than creation-process.md.
import json, os
try:
    root = os.environ.get("CLAUDE_PROJECT_DIR") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    skip = {".venv", ".git", ".claude", ".ipynb_checkpoints"}
    newest = 0
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if name in skip or not os.path.isfile(p): continue
        if name.endswith((".py", ".ipynb")): newest = max(newest, os.path.getmtime(p))
    doc = os.path.join(root, "creation-process.md")
    doc_time = os.path.getmtime(doc) if os.path.exists(doc) else 0
    if newest > doc_time:
        print(json.dumps({"systemMessage": "creation-process.md may be stale: code changed after it. Redraw the diagram and add a log bullet."}))
except Exception: pass
