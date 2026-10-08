# X workspace guesser

guesses which X11 desktop (workspace) a window belongs to by title and contents

takes a list of current (or selected desktops windows.
strips WM_CLASS (app names)
translates them to english
feeds them to a LLM for assigning
if that fails, then OCRs

finally asks users input and lets one change the detected workspace to the correct one

be sure to configure `workspace_meanings.json` according to your needs. the names are partial matches from `wmctrl -d`
in `.env` `AI_OLLAMA` can be any llama compatible chat api

ready for use. it makes a backup before changing anythnig (hardcoded to `/aba/mate./wmctrl-l/PC/` atm): use --undo to revert
can also restore any `wmctrl -l` dump
silently ignores missing windows


file an issue in github if you find bugs

## `$ ./workspace_guesser.py --help `

```
usage: workspace_guesser.py [-h] [--env ENV] [--llm-url LLM_URL] [--cache-dir CACHE_DIR] [--desktop DESKTOP] [--limit LIMIT] [--bump-unknowns] [--bump] [--title TEXT] [--focus] [--list-classes] [--class PATTERN] [--class-exact] [--batch BATCH] [--jobs JOBS]
                            [--llm-timeout LLM_TIMEOUT] [--retries RETRIES] [--model MODEL] [--num-predict NUM_PREDICT] [--ocr-lang OCR_LANG] [--ocr-max-chars OCR_MAX_CHARS] [--no-ocr] [--max-ocr MAX_OCR] [--unknown-ws UNKNOWN_WS] [--yes] [--apply] [--all-workspaces]
                            [--no-strict-retry] [--repair-desktop] [--apply-report APPLY_REPORT] [--review REPORT] [--undo] [--restore-from BACKUP] [--report REPORT]

workspace-guesser -- classify the windows of the current X11 workspace into
MATE virtual workspaces.

Pipeline
  1. Read EWMH state: workspace names + the current desktop index.
  2. List every managed window living on the current workspace.
  3. Translate Estonian titles to English through LibreTranslate (cached).
  4. Ask the Ollama LLM which workspace each window belongs to, using the
     (translated) window titles -- batched.
  5. Whatever the LLM could not place is escalated: the window is screenshotted,
     OCR'd (tesseract), the text translated, and the LLM is asked again.
  6. Show every decision as "ws_name - window title" and ask the user to
     confirm each one.

The LLM is given --llm-timeout seconds per query (default 900 = 15 min).

options:
  -h, --help            show this help message and exit
  --env ENV
  --llm-url LLM_URL     Ollama base URL (default: AI_OLLAMA from .env, falls back to http://127.0.0.1:11434)
  --cache-dir CACHE_DIR
  --desktop DESKTOP     workspace index to inspect, or 'all' for every workspace (default: the current one)
  --limit LIMIT         only the first N windows
  --bump-unknowns       bring the bare '~' shells of --desktop to the front of the stacking order, then exit
  --bump                bring the windows selected by --class/--title to the front of the stacking order, then exit
  --title TEXT          partial window-title match for --bump (case-insensitive, repeatable)
  --focus               with --bump-unknowns, focus the last window raised
  --list-classes        list the open WM_CLASS values from `wmctrl -lx` (honours --desktop) and exit
  --class PATTERN       only windows whose WM_CLASS matches PATTERN (case-insensitive substring; repeatable)
  --class-exact         match --class exactly instead of as a substring
  --batch BATCH         windows per LLM query
  --jobs JOBS           parallel LLM batches
  --llm-timeout LLM_TIMEOUT
                        seconds per LLM query (default 900 = 15 min)
  --retries RETRIES
  --model MODEL
  --num-predict NUM_PREDICT
                        cap on generated tokens per query (reasoning models need room; each query is allowed --llm-timeout seconds)
  --ocr-lang OCR_LANG
  --ocr-max-chars OCR_MAX_CHARS
  --no-ocr              skip the OCR fallback
  --max-ocr MAX_OCR     cap OCR escalations (0 = unlimited)
  --unknown-ws UNKNOWN_WS
                        catch-all workspace for windows that cannot be classified (default: the workspace named SORT)
  --yes                 accept everything (no prompts)
  --apply               move windows to the chosen workspaces at the end
  --all-workspaces      also classify windows on every workspace, not just --desktop
  --no-strict-retry     do not retry all-UNKNOWN batches with stricter rules
  --repair-desktop      offline: rewrite malformed _NET_WM_DESKTOP properties without moving any window
  --apply-report APPLY_REPORT
                        offline: just apply a reviewed report, no LLM
  --review REPORT       offline: confirm an existing report interactively, no LLM and no OCR
  --undo                move every window back to the state recorded by the last pre-operation snapshot, then exit
  --restore-from BACKUP
                        offline: restore window placements from a 'wmctrl -l' snapshot (file or directory), no LLM
  --report REPORT
```

