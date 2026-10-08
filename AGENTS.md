# workspace-guesser — working notes

Classifies the windows of the current X11 workspace into MATE virtual
workspaces, using window titles first and screenshot+OCR as a fallback.

## Testing convention

Keep test runs small and cheap. The standard smoke test is:

```
./workspace_guesser.py --class firefox --limit 3 --yes --report /tmp/opencode/t.json
```

i.e. **filter to `wm_class=firefox`, cap at 3 windows, use the current
desktop**. `--yes` skips the confirmation prompt. Never point a test at all
300+ windows by accident — a full run costs roughly 30-60 minutes of GPU time.

`--list-classes` (honours `--desktop`) is the cheap way to see what is there
before choosing a class to test with.

## Gotchas discovered the hard way

- **Verify workspace changes with `xprop -id <id> _NET_WM_DESKTOP`, never
  `wmctrl -l`.** wmctrl is unreliable here: it hangs ~60s and aborts with
  `BadWindow` when a window closes, returning a truncated/stale list. It once
  reported 268 windows on ws 0 when every one of them was actually on ws 3.
- A window move needs a **ClientMessage** to the root; writing the property
  alone is silently ignored by Marco. `_NET_WM_DESKTOP` must also be written
  with type **CARDINAL**. `--repair-desktop` fixes windows left with a bad
  property type (it changes no workspace).
- **Never mention the application name to the model — erase it.** Naming it
  even as "ignore this, it is not the topic" acted as a pink elephant: every
  browser window was sent to the `firefox` workspace ("firefox development").
  `set_app_names()` builds the strip list from the WM_CLASS values in use plus
  a static list; `clean_title()` erases them along with bracketed tags
  (`[aTodo2]`, `[aInvoice]`) and version numbers. Apply `clean_title()` to the
  *translated* title too -- LibreTranslate output still carries the raw tags,
  and leaving them in put all windows back into aTodo2.
- `wmctrl -l` prints 8-digit ids (`0x00c00787`), Xlib gives ints — always
  compare window ids through `norm_id()`.
- The workspace names carry a leading order marker: subscript digits, and also
  superscripts (`⁺firefox`) and a degree sign (`° mgmt`). Use `ws_key()` to
  strip it; meaning lookups are keyed on the stripped **name**, never the index.

## Files

- `workspace_meanings.json` — what each workspace is for, keyed by workspace
  name with the marker stripped, plus free-form `rules` fed to the model.
  Add a workspace by adding an entry here; no code change needed.
- `workspace_guesser_report.json` — the run's output, plus `accepted` (what the
  user confirmed). `--review` re-confirms and `--apply-report` re-applies it.
  Writes keep the previous file as `.bak`.
- Backups of window placement live in `/aba/mate./wmctrl-l/PC/` (cron, every
  10 min) and workspace names in `/aba/mate./wmctrl-d/PC/`. `--restore-from`
  accepts a file or a directory.

## Model

Needs **>= 3B**. `qwen3:4b` on `http://127.0.0.1:11434` is the good one.
`qwen2.5:1.5b` answers UNKNOWN to everything (or dumps every window onto one
workspace) — the script warns when the resolved model is that small.
qwen3 ignores `think:false` and reasons inline, so `--num-predict` defaults to
8000; below that it runs out of budget before emitting the answer.

The Ollama server may be running an old binary. `/usr/local/bin/ollama` is
current; a stale `/usr/bin/ollama` (0.1.17) cannot load newer GGUFs and reports
"model may be incompatible".