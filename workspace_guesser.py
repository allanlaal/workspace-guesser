#!/usr/bin/env python3
"""
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
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# --------------------------------------------------------------------------
# constants / helpers
# --------------------------------------------------------------------------

SUBSCRIPT = str.maketrans(
    "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089"
    "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079",
    "01234567890123456789",
)

# window classes that carry no information and are never worth an OCR call
JUNK_CLASS = {
    "panel", "mate-panel", "plasmashell", "dock", "tint2", "xdesktop",
    "xroot", "notify-send", "mate-notification-daemon",
}

DEFAULT_LLM_TIMEOUT = 900  # 15 minutes per query, as requested
DEFAULT_OCR_TIMEOUT = 180


def sh(cmd: str, timeout: int = 60) -> str:
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return p.stdout


# A bare home-directory shell ("allan@S7: ~") says nothing about what the
# window is for. Sending it to the model wastes a slot and invites a guess.
CONTENTLESS = {"", "~", "/", "-", ".", ".."}


def is_contentless(win: dict) -> bool:
    """True for titles that carry no topic at all."""
    return (win.get("clean") or "").strip().strip("/").strip() in CONTENTLESS


def strip_emoji(text: str) -> str:
    """Drop pictographs/symbol soup while keeping letters, digits and CJK."""
    out = []
    for ch in text:
        cat = 0
        cp = ord(ch)
        if cp >= 0x1F000 or 0x2600 <= cp <= 0x27BF or cp in (0x2764, 0xFE0F, 0x200D):
            continue
        try:
            import unicodedata
            cat = unicodedata.category(ch)
        except Exception:
            cat = 0
        if cat in ("So", "Sk", "Cs", "Co", "Cn") and cp > 0x2000:
            continue
        out.append(ch)
    s = "".join(out)
    s = re.sub(r"[|¦]{2,}", " ", s)          # |||||| noise used as separators
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()


def find_sort_ws(names: list[str]) -> int | None:
    """Index of the catch-all workspace, matched by NAME not by index.

    Looks for the name listed under "catch_all" in workspace_meanings.json
    (default SORT), so the bucket survives workspaces being added or reordered.
    """
    want = (load_meanings().get("catch_all") or "SORT").upper()
    for i, n in enumerate(names):
        if want in ws_key(n).upper():
            return i
    return None


def norm_id(value) -> str:
    """Canonical window id.

    `wmctrl -l` prints 0x00c00787 while Xlib hands out plain ints, so both have
    to end up as the same 8-digit hex string before they can be compared.
    """
    if isinstance(value, int):
        return f"0x{value:08x}"
    try:
        return f"0x{int(value, 16):08x}"
    except (TypeError, ValueError):
        return str(value)


# Application names must be REMOVED from the title, never mentioned to the
# model: naming them at all makes them stick (every page became "firefox
# development" simply because the string firefox was in the prompt).
APP_NAMES = [
    "Mozilla Firefox", "Firefox Developer Edition", "Firefox", "Google Chrome",
    "Chromium", "Brave", "Microsoft Edge", "Thunderbird", "gedit", "Text Editor",
    "Caja", "Files", "DBeaver", "IntelliJ IDEA", "PyCharm", "WebStorm",
    "Android Studio", "Konsole", "mate-terminal", "Terminal", "xterm",
    "Emulator", "Android Emulator", "Mirage", "VLC", "LibreOffice", "GIMP",
    "Remmina", "AutoKey", "Postman", "VirtualBox", "Nautilus", "HexChat",
]
_SITE_TAIL = (
    r"(?:Google Search|Google Docs|Google Sheets|Google Keep|Google Calendar|"
    r"Google Gemini|YouTube|Drive)"
)
_APP_RE: re.Pattern | None = None


def set_app_names(extra: list[str] | None = None) -> None:
    """Build the 'application names to erase' pattern.

    Includes the WM_CLASS values seen on this system (DBeaver, gedit, ...)
    so whatever runs a window, its name never reaches the model.
    """
    global _APP_RE
    names = list(APP_NAMES)
    for c in extra or []:
        base = re.split(r"[._ -]?\d", c.strip())[0].strip()
        for n in (c.strip(), base):
            if len(n) >= 3 and n.lower() not in {x.lower() for x in names}:
                names.append(n)
    alts = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    _APP_RE = re.compile(rf"(?<![\w.])(?:{alts})(?![\w])", re.IGNORECASE)


def clean_title(title: str) -> str:
    """Reduce a window title to its topic.

    Order matters: drop bracketed tags and application names first, then peel
    separators and trailing site names that only existed because the app name
    was there ("... — Google Search — Mozilla Firefox").
    """
    original = strip_emoji(title).strip()
    t = original

    # bracketed single-token tags are project/queue markers, not the topic:
    # "[aTodo2]", "[aInvoice]", "[AIS]". Tags with spaces are real text
    # ("[2 of 3] floor.png") and are kept.
    for _ in range(3):
        before = t
        t = re.sub(r"\[[^\[\]\s]{1,24}\]", " ", t)
        if t == before:
            break

    # application names, e.g. "Mozilla Firefox", "DBeaver 25.0.4", "gedit"
    for _ in range(3):
        before = t
        if _APP_RE is not None:
            t = _APP_RE.sub(" ", t)
        t = re.sub(r"(?<![\w./-])[\dv]?\d+(?:[._]\d+)+(?![\w])", " ", t)
        t = re.sub(r"\s{2,}", " ", t)
        if t == before:
            break

    # now peel dangling separators and trailing site names
    for _ in range(4):
        before = t
        t = re.sub(rf"\s*[\u2014\u2013-]\s*{_SITE_TAIL}\s*$", "", t, flags=re.I)
        t = re.sub(r"[\s\u2014\u2013|,.\-]+$", "", t)
        t = re.sub(r"^[\s\u2014\u2013|,\-]+", "", t)
        t = re.sub(r"\s{2,}", " ", t)
        if t == before:
            break

    t = re.sub(r"\s*[\u2014\u2013-]\s*[\u2014\u2013-]\s*", " - ", t)
    # "user@host: /path" -> the path
    t = re.sub(r"^[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+:\s*", "", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t or original


# --------------------------------------------------------------------------
# .env
# --------------------------------------------------------------------------

def load_env(path: str) -> dict:
    env = dict(os.environ)
    if not os.path.isfile(path):
        sys.exit(f"[!] no .env at {path}")
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            env[key.strip()] = val.strip().strip('"').strip("'")
    return env


# --------------------------------------------------------------------------
# EWMH
# --------------------------------------------------------------------------

def get_workspaces() -> tuple[list[str], int]:
    raw = sh("xprop -root _NET_DESKTOP_NAMES _NET_CURRENT_DESKTOP")
    names = []
    current = 0
    for line in raw.splitlines():
        if "_NET_DESKTOP_NAMES" in line:
            quoted = line.split("=", 1)[1].strip()
            names = re.findall(r'"((?:[^"\\]|\\.)*)"', quoted)
            names = [n.encode().decode("unicode_escape") if "\\" in n else n for n in names]
            names = [n.replace('\\"', '"').replace("\\\\", "\\") for n in names]
        elif "_NET_CURRENT_DESKTOP" in line:
            try:
                current = int(line.rsplit("=", 1)[1].strip())
            except ValueError:
                pass
    if not names:
        names = [f"ws{i}" for i in range(len(sh("wmctrl -d").splitlines()) or 1)]
    return names, current


MEANINGS_PATH = "workspace_meanings.json"


def ws_key(name: str) -> str:
    """Workspace name with its leading order marker removed.

    The panel prefixes names with an order marker -- subscript digits, but also
    superscripts and a degree sign ("\u2085aTodo2", "\u207afirefox", "\u00b0mgmt").
    The meanings file is keyed on the bare name so it survives workspaces being
    added or reordered.
    """
    return re.sub(r"^[\u2070-\u209f\u00b0\u00b9\u00b2\u00b3\u02c7]+",
                  "", name).translate(SUBSCRIPT).strip()


def ws_meaning(key: str, meanings: dict) -> str:
    """Meaning for a workspace key, matched leniently."""
    table = meanings.get("meanings") or {}
    if key in table:
        return table[key]
    low = key.lower()
    for k, v in table.items():
        if k.lower() == low:
            return v
    return ""


def load_meanings(path: str = MEANINGS_PATH) -> dict:
    """Load workspace meanings keyed by name (not by workspace index)."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {"meanings": {}, "rules": []}
    return {"meanings": data.get("meanings", {}), "rules": data.get("rules", [])}


def ws_label(idx: int, name: str, meanings: dict | None = None) -> str:
    """ASCII-ish rendering of a workspace name so the LLM can read it.

    The names carry a leading subscript/order marker ("₅aTodo2", "° mgmt") that
    only confuses the model, so it is stripped and subscripts are folded to
    plain digits.
    """
    label = ws_key(name)
    text = f"[{idx}] {label or '_'}"
    if meanings:
        meaning = ws_meaning(label, meanings)
        if meaning:
            text += f" ({meaning})"
    return text


def list_windows(desktop: int | None, only_visible: bool = False,
                 class_needles: list[str] | None = None,
                 class_exact: bool = False) -> list[dict]:
    """Every managed window, read in-process via Xlib.

    Spawning wmctrl/xprop per window takes ~0.5 s each and wmctrl aborts with
    BadWindow when a window closes mid-listing, so talk to the X server
    directly instead. `desktop=None` returns windows from all workspaces.
    """
    from Xlib import X, display  # local import: only needed for enumeration
    from Xlib.error import XError

    d = display.Display()
    root = d.screen().root
    a_client_list = d.intern_atom("_NET_CLIENT_LIST")
    a_desktop = d.intern_atom("_NET_WM_DESKTOP")
    a_name = d.intern_atom("_NET_WM_NAME")
    a_wtype = d.intern_atom("_NET_WM_WINDOW_TYPE")
    a_state = d.intern_atom("_NET_WM_STATE")
    a_hidden = d.intern_atom("_NET_WM_STATE_HIDDEN")
    a_desktop_type = d.intern_atom("_NET_WM_WINDOW_TYPE_DESKTOP")

    prop = root.get_full_property(a_client_list, X.AnyPropertyType)
    ids = list(prop.value) if prop else []

    wins = []
    for wid in ids:
        try:
            win = d.create_resource_object("window", wid)

            dprop = win.get_full_property(a_desktop, X.AnyPropertyType)
            if not dprop or not dprop.value:
                continue
            dsk = int(dprop.value[0])
            if dsk == 0xFFFFFFFF:
                dsk = -1  # sticky
            if desktop is not None and dsk not in (desktop, -1):
                continue

            wprop = win.get_full_property(a_wtype, X.AnyPropertyType)
            if wprop and a_desktop_type in wprop.value:
                continue  # desktop icon layer

            title = ""
            nprop = win.get_full_property(a_name, X.AnyPropertyType)
            if nprop and nprop.value:
                title = bytes(nprop.value).decode("utf-8", "replace")
            if not title:
                cprop = win.get_full_property(d.intern_atom("WM_NAME"),
                                              X.AnyPropertyType)
                if cprop and cprop.value:
                    title = bytes(cprop.value).decode("utf-8", "replace")
            title = title.strip()
            if not title:
                continue

            cls = ""
            cprop = win.get_full_property(d.intern_atom("WM_CLASS"),
                                          X.AnyPropertyType)
            if cprop and cprop.value:
                parts = bytes(cprop.value).split(b"\x00")
                cls = parts[1].decode("utf-8", "replace") if len(parts) > 1 \
                    else parts[0].decode("utf-8", "replace")
            cls = cls.split(".")[-1].lower()
            if cls in JUNK_CLASS:
                continue
            if not matches_class(cls, class_needles or [], class_exact):
                continue

            geo = win.get_geometry()
            if geo.width < 40 or geo.height < 40:
                continue  # 1x1 helper windows

            hidden = False
            sprop = win.get_full_property(a_state, X.AnyPropertyType)
            if sprop and a_hidden in sprop.value:
                hidden = True
            if only_visible and hidden:
                continue

            wins.append({
                "id": norm_id(wid),
                "ws": dsk,
                "class": cls or "window",
                "title": title,
                "clean": clean_title(title),
                "size": [geo.width, geo.height],
                "hidden": hidden,
            })
        except XError:
            continue  # window vanished between listing and querying
    d.close()
    return wins


def window_is_hidden(wid: str) -> bool:
    state = sh(f"xprop -id {wid} _NET_WM_STATE", timeout=30)
    return "_NET_WM_STATE_HIDDEN" in state


def wmctrl_classes(desktop: int | None) -> list[tuple[str, int]]:
    """Unique WM_CLASS values with window counts, from `wmctrl -lx`.

    Only windows on `desktop` are counted (None = every workspace). wmctrl can
    abort mid-listing when a window closes, so an empty result is treated as
    "unavailable" and the caller falls back to the Xlib enumeration.
    """
    counts: dict[str, int] = {}
    for line in sh("wmctrl -lx", timeout=120).splitlines():
        parts = line.split(None, 4)
        if len(parts) < 4 or not parts[0].startswith("0x"):
            continue
        try:
            dsk = int(parts[1])
        except ValueError:
            continue
        if desktop is not None and dsk != desktop:
            continue
        cls = parts[2].split(".")[-1] or "?"
        counts[cls] = counts.get(cls, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def xlib_classes(desktop: int | None) -> list[tuple[str, int]]:
    """Same as wmctrl_classes but read straight from the X server."""
    counts: dict[str, int] = {}
    for w in list_windows(desktop):
        counts[w["class"]] = counts.get(w["class"], 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def list_classes(desktop: int | None, needle: str | None = None) -> None:
    classes = wmctrl_classes(desktop)
    source = "wmctrl -lx"
    if not classes:
        classes = xlib_classes(desktop)
        source = "Xlib (wmctrl returned nothing)"
    if needle:
        low = needle.lower()
        classes = [(c, n) for c, n in classes if low in c.lower()]
    scope = ("every workspace" if desktop is None
             else f"workspace {desktop} {ws_key(get_workspaces()[0][desktop])!r}")
    print(f"[*] {len(classes)} WM_CLASS value(s) on {scope}, via {source}")
    width = max((len(c) for c, _ in classes), default=4)
    for cls, n in classes:
        print(f"    {cls:<{width}}  {n:>4} window(s)")


def matches_class(cls: str, needles: list[str], exact: bool) -> bool:
    """A window is selected when its WM_CLASS matches any of the needles."""
    if not needles:
        return True
    low = cls.lower()
    for n in needles:
        t = n.lower()
        if (cls.lower() == t if exact else t in low):
            return True
    return False


def normalise_desktop_props() -> int:
    """Rewrite malformed _NET_WM_DESKTOP properties (CARDINAL type).

    An earlier version of move_window wrote the property with type
    _NET_WM_DESKTOP instead of CARDINAL. Windows in that state are invisible to
    clients that request the property as CARDINAL -- wmctrl reported all of
    them as workspace 0 -- even though the value itself is intact. Only the
    type is corrected here, so nothing moves.
    """
    from Xlib import X, display

    d = display.Display()
    fixed = 0
    try:
        root = d.screen().root
        atom = d.intern_atom("_NET_WM_DESKTOP")
        cardinal = d.intern_atom("CARDINAL")
        client_list = root.get_full_property(
            d.intern_atom("_NET_CLIENT_LIST"), X.AnyPropertyType)
        if not client_list or not client_list.value:
            raise RuntimeError("no _NET_CLIENT_LIST on the root window")
        for wid in list(client_list.value):
            try:
                win = d.create_resource_object("window", wid)
                cur = win.get_full_property(atom, X.AnyPropertyType)
                if not cur or not cur.value or cur.property_type == cardinal:
                    continue
                win.change_property(atom, cardinal, 32, [int(cur.value[0])])
                fixed += 1
            except Exception:  # noqa: BLE001
                continue
        d.flush()
        d.sync()
    finally:
        try:
            d.close()
        except Exception:  # noqa: BLE001
            pass
    return fixed


def move_window(wid: str, ws: int) -> bool:
    """Move a window to another workspace the way EWMH specifies.

    Writing _NET_WM_DESKTOP on its own is not enough: the window manager has to
    be told, and the property itself must be of type CARDINAL. Marco ignores
    a plain property write, so send the ClientMessage (what `wmctrl -t` does)
    and then update the property, verifying the result.
    """
    from Xlib import X, display
    from Xlib import protocol

    try:
        num = int(wid, 16) if isinstance(wid, str) else wid
    except (TypeError, ValueError):
        return False
    d = display.Display()
    try:
        atom = d.intern_atom("_NET_WM_DESKTOP")
        cardinal = d.intern_atom("CARDINAL")
        win = d.create_resource_object("window", num)
        root = d.screen().root

        ev = protocol.event.ClientMessage(
            window=win,
            client_type=atom,
            data=(32, [ws, 2, 0, 0, 0]),  # 2 = request from the user
        )
        root.send_event(
            ev,
            event_mask=X.SubstructureRedirectMask | X.SubstructureNotifyMask,
        )
        win.change_property(atom, cardinal, 32, [ws])
        d.flush()
        d.sync()

        check = win.get_full_property(atom, X.AnyPropertyType)
        return bool(check and check.value and int(check.value[0]) == ws)
    except Exception:  # noqa: BLE001
        return False
    finally:
        try:
            d.close()
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------
# LibreTranslate
# --------------------------------------------------------------------------

class Translator:
    def __init__(self, base: str, cache_path: str, cache: dict):
        self.base = base.rstrip("/")
        self.cache_path = cache_path
        self.cache = cache
        self.enabled = bool(self.base)
        self.failed = False

    def _key(self, text: str) -> str:
        return f"{self.base}|{text}"

    def translate(self, texts: list[str]) -> dict[str, str]:
        """Return {original: english} only for strings that are not English."""
        out: dict[str, str] = {}
        todo = []
        for t in texts:
            t = (t or "").strip()
            if not t or t in out:
                continue
            hit = self.cache.get(self._key(t))
            if hit:
                out[t] = hit
            else:
                todo.append(t)
        if todo and self.enabled:
            for i in range(0, len(todo), 25):
                chunk = todo[i:i + 25]
                got = self._post("/translate", {"q": chunk, "source": "et",
                                                "target": "en", "format": "text"})
                if not got:
                    self.failed = True
                    continue
                items = got.get("translatedText", [])
                if isinstance(items, str):
                    items = [items]
                for src, dst in zip(chunk, items):
                    if isinstance(dst, str) and dst.strip():
                        out[src] = dst.strip()
                        self.cache[self._key(src)] = dst.strip()
            self._save()
        return out

    def _post(self, path: str, payload: dict) -> dict | None:
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                return json.loads(resp.read().decode())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            return None

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as fh:
                json.dump(self.cache, fh, ensure_ascii=False)
        except OSError:
            pass


# --------------------------------------------------------------------------
# Ollama
# --------------------------------------------------------------------------

class LLM:
    def __init__(self, base: str, model: str | None, timeout: int, retries: int,
                 num_predict: int = 512):
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.num_predict = num_predict
        self.model = model or self.pick_model()
        self.warn_if_weak()

    def tags(self) -> dict:
        try:
            with urllib.request.urlopen(self.base + "/api/tags", timeout=30) as r:
                return {m["name"]: m for m in json.load(r).get("models", [])}
        except Exception:  # noqa: BLE001
            return {}

    def warn_if_weak(self) -> None:
        """Below ~3B the model stops discriminating and dumps windows onto a
        single workspace, so say so instead of silently producing rubbish."""
        info = self.tags().get(self.model) or {}
        raw = (info.get("details") or {}).get("parameter_size") or ""
        digits = re.findall(r"[\d.]+", raw)
        size = float(digits[0]) if digits else 0.0
        if size and size < 3.0:
            print(f"[!] {self.model} is only {raw}: expect coarse results "
                  f"(everything lands on one workspace). qwen3:4b is better.")
        elif not size:
            print(f"[!] could not determine the size of {self.model}")

    def pick_model(self) -> str:
        """Default to the biggest model present: small ones answer UNKNOWN to
        everything and the catch-all bucket then swallows the whole desktop."""
        try:
            with urllib.request.urlopen(self.base + "/api/tags", timeout=30) as r:
                models = json.load(r).get("models", [])
        except Exception as exc:  # noqa: BLE001
            sys.exit(f"[!] cannot reach Ollama at {self.base}: {exc}")
        if not models:
            sys.exit("[!] Ollama has no models pulled")

        def size(m):
            raw = (m.get("details") or {}).get("parameter_size") or ""
            digits = re.findall(r"[\d.]+", raw)
            return float(digits[0]) if digits else 0.0

        models.sort(key=size)
        best = models[-1]
        if size(best) < 3.0:
            print(f"[!] only small models available (best: {best['name']}). "
                  f"Classification needs >= 3B or it answers UNKNOWN to "
                  f"everything -- pass --model to override.")
        return best["name"]

    def chat(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
            "think": False,
            "options": {"temperature": 0.1, "num_predict": self.num_predict},
        }
        for attempt in range(1, self.retries + 1):
            t0 = time.time()
            print(f"    · LLM query (attempt {attempt}/{self.retries}, "
                  f"budget {self.timeout}s)…", flush=True)
            try:
                req = urllib.request.Request(
                    self.base + "/api/chat",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode())
                txt = body.get("message", {}).get("content", "")
                print(f"    · answered in {time.time() - t0:.0f}s", flush=True)
                return txt
            except Exception as exc:  # noqa: BLE001
                print(f"    · failed after {time.time() - t0:.0f}s: {exc}", flush=True)
                if attempt < self.retries:
                    time.sleep(5)
        return ""


SYSTEM = (
    "You are a precise classifier. You place Linux GUI windows into named "
    "virtual workspaces. You answer with the requested format only, no prose, "
    "no markdown fences, no explanations."
)


def parse_assignments(reply: str, n: int) -> tuple[list[str | None], list[int]]:
    """Extract answers of the form '<idx>=<ws id|UNKNOWN>'.

    Reasoning models put their thinking before the answer, so the last
    occurrence of each index wins. Returns (answers, missing_indices).
    """
    found: dict[int, str] = {}
    for raw in reply.splitlines():
        line = raw.strip().strip("`*>- \t")
        m = re.match(r"^(\d+)\s*[=:.]\s*\[?([A-Za-z0-9_ ]+?)\]?\s*[.,)]?\s*$", line)
        if not m:
            continue
        idx, val = int(m.group(1)), m.group(2).strip().strip("`\"'.")
        if 1 <= idx <= n:
            found[idx] = val
    answers: list[str | None] = [found.get(i) for i in range(1, n + 1)]
    missing = [i for i, a in enumerate(answers) if a is None]
    return answers, missing


# --------------------------------------------------------------------------
# screenshot + OCR
# --------------------------------------------------------------------------

def ocr_window(win: dict, shot_dir: str, langs: str, max_chars: int) -> str:
    if window_is_hidden(win["id"]):
        return ""
    os.makedirs(shot_dir, exist_ok=True)
    raw = os.path.join(shot_dir, f"{win['id'].replace('0x','')}.png")
    small = os.path.join(shot_dir, f"{win['id'].replace('0x','')}_s.png")
    # `import -window` captures whatever is on screen at that spot, so a window
    # buried behind others yields a black image. Raise it first (without
    # activating, to avoid stealing focus) and give the compositor a moment.
    try:
        subprocess.run(["xdotool", "windowraise", win["id"]],
                       timeout=15, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        time.sleep(0.4)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        pass
    try:
        subprocess.run(["import", "-silent", "-window", win["id"], raw],
                       check=True, timeout=60,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ""
    if not os.path.exists(raw) or os.path.getsize(raw) < 2000:
        return ""
    # a flat/black capture has almost no colours: OCR would return noise
    try:
        colours = int(sh(f'identify -format "%k" "{raw}"', timeout=30).strip() or 0)
    except ValueError:
        colours = 0
    if colours and colours <= 4:
        return ""
    subprocess.run(["convert", raw, "-resize", "1800x1800>", "-colorspace", "gray",
                    small], timeout=60,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        text = subprocess.run(
            ["tesseract", small, "stdout", "-l", langs, "--psm", "6"],
            capture_output=True, text=True, timeout=DEFAULT_OCR_TIMEOUT,
        ).stdout
    except subprocess.TimeoutExpired:
        return ""
    finally:
        for f in (raw, small):
            try:
                os.remove(f)
            except OSError:
                pass
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) >= 3]
    return "\n".join(lines)[:max_chars]


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------

RULES = """- Match on the topic/project/subject of the window, not on the app name.
- A window belongs to a workspace whose name already hints at its subject
  (project names, languages, vehicle/model names, hobbies, services).
- Prefer the most specific matching workspace over generic or empty ones.
- Terminals: use the path/project in the title.
- Browsers: use the page/topic in the title, ignore the browser suffix.
- Windows of the same project may share a workspace; that is fine.
- If no workspace plausibly matches, answer UNKNOWN."""

OUTPUT_SPEC = """OUTPUT: exactly one line per window, in order, nothing else.
No prose, no reasoning, no markdown, no extra lines.
<number>=<workspace id>
Example:
1=29
2=0"""

STRICT_RULES = """- Almost every window belongs SOMEWHERE. Only answer UNKNOWN when the
  window's subject matches no workspace at all (e.g. a blank untitled document).
- When torn between two workspaces, answer the more specific one.
- An empty or generic workspace name is still a valid destination for windows
  that belong to no particular project (e.g. plain home-directory shells).
- Never answer UNKNOWN just because you are unsure: pick the best candidate."""

STRICT_OUTPUT_SPEC = """OUTPUT: exactly one line per window, in order, nothing else.
No prose, no reasoning, no markdown, no extra lines.
<number>=<workspace id>
Every line must be a number. UNKNOWN is allowed but should be rare.
Example:
1=29
2=0"""


def _window_lines(batch: list[dict], translated: dict[str, str]) -> str:
    """Window lines for the prompt: the topic and nothing else.

    The application name is deliberately absent rather than mentioned and
    flagged as irrelevant -- repeating it was enough to make the model key on
    it and send every browser window to the 'firefox' workspace.
    """
    lines = []
    for i, w in enumerate(batch, 1):
        topic = w["clean"] or strip_emoji(w["title"])
        tr = translated.get(w["title"])
        if tr:
            # the translation is derived from the raw title, so it still
            # carries "[aTodo2]" / "Mozilla Firefox" and has to be cleaned
            # with the same pipeline
            tr = clean_title(tr)
            if tr and tr.lower() != topic.lower():
                topic = f"{topic} || {tr}"
        lines.append(f"{i}. {topic}")
    return "\n".join(lines)


def classify_by_title(llm: LLM, names: list[str], batch: list[dict],
                      translated: dict[str, str], repair: bool = True,
                      strict: bool = False,
                      meanings: dict | None = None) -> tuple[list[str | None], list[int]]:
    ws_block = "\n".join(ws_label(i, n, meanings) for i, n in enumerate(names))
    rules = STRICT_RULES if strict else RULES
    extra = "\n".join(f"- {r}" for r in (meanings or {}).get("rules", []))
    spec = STRICT_OUTPUT_SPEC if strict else OUTPUT_SPEC
    user = f"""VIRTUAL WORKSPACES (use the [id] shown):
{ws_block}

WINDOWS (choose the best workspace id for each):
{_window_lines(batch, translated)}

RULES
{rules}
{extra}

{spec}"""

    reply = llm.chat(SYSTEM, user)
    answers, missing = parse_assignments(reply, len(batch))
    if missing and repair:
        sub = [batch[m] for m in missing]
        user2 = f"""VIRTUAL WORKSPACES (use the [id] shown):
{ws_block}

YOU FAILED TO ANSWER THESE WINDOWS (renumbered from 1):
{_window_lines(sub, translated)}

Answer immediately with the mapping only. Do not deliberate, do not explain,
do not restate the windows. One line per window, nothing else.

{rules}
{extra}

{spec}"""
        reply2 = llm.chat(SYSTEM, user2)
        sub_answers, _ = parse_assignments(reply2, len(sub))
        for pos, m in enumerate(missing):
            if sub_answers[pos] is not None:
                answers[m] = sub_answers[pos]
        missing = [i for i, a in enumerate(answers) if a is None]
    return answers, missing


def classify_by_ocr(llm: LLM, names: list[str], win: dict, ocr_text: str,
                    translated: dict[str, str],
                    meanings: dict | None = None) -> str:
    ws_block = "\n".join(ws_label(i, n, meanings) for i, n in enumerate(names))
    extra = "\n".join(f"- {r}" for r in (meanings or {}).get("rules", []))
    tr = translated.get(ocr_text[:400])
    user = f"""VIRTUAL WORKSPACES (use the [id] shown):
{ws_block}

WINDOW TOPIC: {win['clean'] or strip_emoji(win['title'])}
SCREENSHOT OCR TEXT:
{ocr_text}
{f"TRANSLATED OCR TEXT: {tr}" if tr else ""}

RULES
{RULES}
{extra}

Which single workspace does this window belong to?
OUTPUT: one line only, `<number>=<workspace id>` where number is 1.
1=<workspace id or UNKNOWN>"""
    reply = llm.chat(SYSTEM, user)
    answers, _ = parse_assignments(reply, 1)
    if answers and answers[0]:
        return answers[0]
    m = re.search(r"\[(\d+)\]", reply)
    return m.group(1) if m else "UNKNOWN"


def normalise_ws(answer: str, names: list[str]) -> str | None:
    answer = str(answer).strip()
    if not answer or answer.upper().startswith("UNKNOWN"):
        return None
    m = re.search(r"\[?(\d+)\]?", answer)
    if not m:
        return None
    idx = int(m.group(1))
    return str(idx) if 0 <= idx < len(names) else None


# --------------------------------------------------------------------------
# confirmation
# --------------------------------------------------------------------------

def apply_report(path: str) -> None:
    """Re-apply a previously reviewed report: move every accepted window."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    rows = data.get("accepted") or []
    todo = [r for r in rows
            if r.get("ws") is not None and int(r["ws"]) != r.get("cur_ws")]
    snap = save_snapshot() if todo else None
    moved = 0
    failed = []
    for r in todo:
        if move_window(r["id"], int(r["ws"])):
            moved += 1
        else:
            failed.append(r)
    print(f"[*] moved {moved}/{len(todo)} windows")
    if snap:
        print("[*] undo with: ./workspace_guesser.py --undo")
    if failed:
        print("[!] failed: " + ", ".join(f["id"] for f in failed[:10]))


class KeyReader:
    """Read single keypresses without Enter.

    Falling back to line mode keeps the script usable when stdin is a pipe or
    a terminal that does not support cbreak (cron, CI, `printf | script`).
    """

    def __init__(self):
        self.tty = None
        self.saved = None
        self.pending = ""

    def __enter__(self):
        try:
            import termios
            import tty
            if sys.stdin.isatty():
                self.tty = sys.stdin.fileno()
                self.saved = termios.tcgetattr(self.tty)
                # TCSADRAIN, not the TCSAFLUSH that tty.setcbreak defaults to:
                # flushing would throw away a key pressed just before the
                # prompt appeared.
                tty.setcbreak(self.tty, termios.TCSADRAIN)
        except Exception:  # noqa: BLE001
            self.tty = None
        return self

    def __exit__(self, *exc):
        if self.tty is not None and self.saved is not None:
            try:
                import termios
                termios.tcsetattr(self.tty, termios.TCSADRAIN, self.saved)
            except Exception:  # noqa: BLE001
                pass
        return False

    def _read1(self, timeout: float = 300.0) -> str:
        """Read one character straight from the fd.

        Must not go through sys.stdin: its TextIOWrapper buffers greedily, so
        an escape sequence like ESC [ B gets swallowed into the Python buffer
        and a select() on the file descriptor then reports nothing, splitting
        the sequence into a bare ESC (which cancels the workspace picker).
        """
        import select as select_mod
        ready, _, _ = select_mod.select([self.tty], [], [], timeout)
        if not ready:
            raise EOFError
        data = os.read(self.tty, 1)
        if not data:
            raise EOFError
        return data.decode("utf-8", "replace")

    def readkey(self, timeout: float = 300.0,
                seq_timeout: float = 0.30) -> str:
        """One logical keypress, with ANSI escape sequences glued together."""
        if self.tty is None:
            return self.getch()
        ch = self._read1(timeout)
        if ch != "\x1b":
            return ch.lower()
        seq = ""
        for _ in range(6):
            try:
                nxt = self._read1(seq_timeout)
            except EOFError:
                break
            seq += nxt
            if nxt.isalpha() or nxt == "~":
                break
        return "\x1b" + seq if seq else "\x1b"

    def getch(self) -> str:
        if self.tty is None:
            # line mode: hand back one character at a time so piped input such
            # as "ya" behaves like two keypresses rather than one bad answer
            if not self.pending:
                line = sys.stdin.readline()
                if not line:
                    raise EOFError
                self.pending = line.strip().lower()
            ch, self.pending = self.pending[0], self.pending[1:]
            return ch
        return self._read1(300).lower()

    def readline(self, prompt: str) -> str:
        """Used for the 'which workspace?' follow-up, where digits + Enter are
        natural."""
        sys.stdout.write(prompt)
        sys.stdout.flush()
        return sys.stdin.readline().strip()


UP, DOWN = "\x1b[A", "\x1b[B"
PGUP, PGDN = "\x1b[5~", "\x1b[6~"
HOME, END = "\x1b[1~", "\x1b[4~"
ALT_UP, ALT_DOWN = "\x1bOA", "\x1bOB"


def choose_ws(keys: "KeyReader", names: list[str], suggested: int | None,
              meanings: dict, page: int = 15) -> int | None:
    """Scrollable workspace picker with type-to-filter.

    Runs on the alternate screen buffer, so nothing of the confirmation list
    shows through and no stale rows are left behind when it closes.

    keys
        up/down (or k/j)  move one row
        page up/page down first/last workspace  (home/end do the same)
        letters/digits    filter by workspace id, name or meaning
        backspace         delete a character from the filter
        enter             select
        esc / ctrl-c     cancel (letters are free for searching)
    """
    if keys.tty is None:
        sys.stdout.write(f"    workspace number (0-{len(names) - 1}, "
                         f"empty = unknown): ")
        sys.stdout.flush()
        line = sys.stdin.readline()
        if not line:
            return None
        return normalise_ws(line.strip(), names)

    def rank(i: int, q: str) -> int:
        """0 = perfect hit, 9 = no match. Lower sorts first."""
        label = (ws_key(names[i]) or "_").lower()
        meaning = ws_meaning(label, meanings).lower()
        q = q.lower()
        if not q:
            return 5
        if label.startswith(q) or str(i).startswith(q):
            return 0
        if q in label:
            return 1
        if q in meaning:
            return 2
        return 9

    cursor = 0
    query = ""
    if suggested is not None and 0 <= suggested < len(names):
        cursor = suggested
    # the confirmation list is still on screen below us, so leave room for it
    rows_avail = shutil.get_terminal_size((80, 24)).lines
    page = max(3, min(page, rows_avail - 8))

    sys.stdout.write("\x1b[?1049h")   # alternate screen: no stale text
    try:
        while True:
            hits = [(rank(i, query), i) for i in range(len(names))]
            rows = [i for score, i in sorted(hits) if score < 9]
            if not rows:
                cursor = 0
            else:
                cursor = max(0, min(cursor, len(rows) - 1))
            chosen = rows[cursor] if rows else None

            out = [" pick a workspace   (filter: " + query + "_"]
            if not rows:
                out.append("   (no workspace matches)")
            else:
                start = max(0, min(cursor - page // 2, len(rows) - page))
                for pos in range(start, min(start + page, len(rows))):
                    real = rows[pos]
                    label = ws_key(names[real]) or "_"
                    meaning = ws_meaning(label, meanings)
                    mark = ">" if pos == cursor else " "
                    out.append(f"  {mark} [{real:>2}] {label:<14} {meaning}")
                if len(rows) > page:
                    out.append(f"      ... {len(rows) - page} more match"
                               f"{'es' if len(rows) - page != 1 else ''}")
            out.append("  [7m up/down move [0m [7m pgup/pgdn"
                       " first/last [0m [7m type to search [0m"
                       " [7m enter select [0m [7m esc cancel [0m")

            prev = getattr(keys, "_drawn", 0)
            if prev:
                sys.stdout.write(f"\x1b[{prev}A")
            sys.stdout.write("\x1b[J")          # erase everything below
            for i, line in enumerate(out):
                end_ch = "\n" if i < len(out) - 1 else ""
                sys.stdout.write("\x1b[2K" + line + end_ch)
            sys.stdout.flush()
            keys._drawn = len(out)

            try:
                key = keys.readkey()
            except (EOFError, KeyboardInterrupt):
                return None

            if key in ("\r", "\n"):
                return str(chosen) if chosen is not None else None
            if key in ("\x1b", "\x03"):
                return None
            if key in (UP, ALT_UP, "k"):
                cursor = max(0, cursor - 1)
            elif key in (DOWN, ALT_DOWN, "j"):
                cursor = min(len(rows) - 1, cursor + 1) if rows else 0
            elif key in (PGUP, HOME):
                cursor = 0
            elif key in (PGDN, END):
                cursor = len(rows) - 1
            elif key in ("\x7f", "\b"):
                query = query[:-1]
                cursor = 0
            elif len(key) == 1 and key.isalnum():
                query += key
                cursor = 0
    finally:
        # leave the alternate screen FIRST: a newline written before this is
        # thrown away with it, which used to glue the next row onto the
        # selector's last line
        sys.stdout.write("\x1b[?1049l")
        sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()
        keys._drawn = 0


def ws_label_for(row: dict, names: list[str]) -> str:
    """Destination shown in the confirmation list."""
    if row["ws"] is not None:
        return names[int(row["ws"])]
    if row.get("how") == "topicless":
        return f"<unchanged: {names[int(row['cur_ws'])]}>" \
            if row.get("cur_ws") is not None and row["cur_ws"] >= 0 \
            else "<unchanged>"
    return "<unclassified>"


def confirm(rows: list[dict], names: list[str], auto: bool,
            meanings: dict | None = None) -> tuple[list[dict], bool]:
    """Show 'ws_name - window title' and ask the user about each one."""
    print("\n" + "=" * 78)
    if not rows:
        print("\n[nothing to review: no window got a destination]")
        return [], True
    places = sorted({int(r["cur_ws"]) for r in rows if r["cur_ws"] >= 0})
    where = ("currently spread over workspaces " + ", ".join(str(p) for p in places)
             if len(places) > 1 else
             (f"currently on workspace {places[0]}" if places else ""))
    print(f"PROPOSED PLACEMENT -- {len(rows)} windows, {where}")
    print("=" * 78)
    for n, r in enumerate(rows, 1):
        print(f"{n:>4}. {ws_label_for(r, names)} - {r['title']}")
    print("=" * 78)
    print("keys: press Y/Enter to accept · N to choose another workspace "
          "(arrows + Enter) · S/Q to stop reviewing (no Enter needed)")

    if auto:
        return rows, False

    fixed = []
    with KeyReader() as keys:
        for n, r in enumerate(rows, 1):
            wname = ws_label_for(r, names)
            sys.stdout.write(f"[{n}/{len(rows)}] {wname} - {r['title']}\n    ? ")
            sys.stdout.flush()
            while True:
                try:
                    ans = keys.readkey()
                except (EOFError, KeyboardInterrupt):
                    print()
                    return fixed, True
                if ans in ("\n", "\r", "y"):
                    fixed.append(r)
                    break
                if ans in ("s", "q"):
                    return fixed, True
                if ans in ("n", "x"):
                    hint = (int(r["ws"]) if r["ws"] is not None
                            else find_sort_ws(names))
                    pick = choose_ws(keys, names, hint, meanings or {})
                    r["ws"] = pick
                    r["corrected"] = True
                    fixed.append(r)
                    break
                # anything else: ignore and re-ask without waiting for Enter
    return fixed, False


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def parse_backup(path: str) -> dict[str, tuple[int, str]]:
    """Parse a `wmctrl -l` snapshot into {window_id: (workspace, title)}."""
    out: dict[str, tuple[int, str]] = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.split(None, 3)
            if len(parts) < 4 or not parts[0].startswith("0x"):
                continue
            try:
                ws = int(parts[1])
            except ValueError:
                continue
            out[norm_id(parts[0])] = (ws, parts[3].strip())
    return out


BACKUP_DIR = "/aba/mate./wmctrl-l/PC"
# same naming scheme as the cron snapshots, so the files sit together:
#   wmctrl.list.PC.2026-10-08_21.20.01.ws-quesser.pre
SNAP_MARK = ".ws-quesser.pre"
SNAP_RETIRED = ".done"


def save_snapshot(directory: str = BACKUP_DIR) -> str | None:
    """Record where every window currently sits, before we move anything.

    Written in `wmctrl -l` format so --restore-from can read it back, and read
    from the X server rather than via wmctrl (which is unreliable here).
    """
    try:
        os.makedirs(directory, exist_ok=True)
        host = os.uname().nodename
        stamp = time.strftime("%Y-%m-%d_%H.%M.%S")
        path = os.path.join(directory, f"wmctrl.list.PC.{stamp}{SNAP_MARK}")
        wins = list_windows(None)
        with open(path, "w", encoding="utf-8") as fh:
            for w in wins:
                fh.write(f"{w['id']}  {w['ws']}  {host}  {w['title']}\n")
        return path
    except OSError as exc:
        print(f"[!] could not write the undo snapshot: {exc}")
        return None


def newest_snapshot(directory: str = BACKUP_DIR) -> str | None:
    """Most recent unused pre-operation snapshot."""
    try:
        cands = [os.path.join(directory, f) for f in os.listdir(directory)
                 if SNAP_MARK in f and not f.endswith(SNAP_RETIRED)]
    except OSError:
        return None
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def undo(directory: str = BACKUP_DIR) -> int:
    """Move every window back to where the last snapshot recorded it."""
    path = newest_snapshot(directory)
    if path is None:
        print(f"[!] no snapshot to undo in {directory}")
        return 1
    print(f"[*] undoing with {path}")
    rc = restore(path, True, None)
    if rc == 0:
        try:
            os.replace(path, path + SNAP_RETIRED)
            print(f"[*] snapshot retired: {os.path.basename(path)}.done")
        except OSError:
            pass
    return rc


def pick_backup(path: str) -> str:
    """Accept either a snapshot file or a directory (newest file inside)."""
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        cands = [os.path.join(path, f) for f in os.listdir(path)
                 if os.path.isfile(os.path.join(path, f))]
        if not cands:
            sys.exit(f"[!] no backup files in {path}")
        return max(cands, key=os.path.getmtime)
    sys.exit(f"[!] no such backup: {path}")


def restore(path: str, apply_moves: bool, report: str | None) -> int:
    """Put windows back on the workspaces recorded in a `wmctrl -l` snapshot."""
    path = pick_backup(path)
    names, _current = get_workspaces()
    backup = parse_backup(path)
    print(f"[*] backup {path}")
    print(f"[*] {len(backup)} entries in snapshot, {len(names)} workspaces now")

    live = list_windows(None)
    by_id = {w["id"]: w for w in live}
    by_title: dict[str, list[dict]] = {}
    for w in live:
        by_title.setdefault(w["title"], []).append(w)

    fallback = find_sort_ws(names)
    rows = []
    unmatched = 0
    unsorted = 0
    for wid, (ws, title) in sorted(backup.items()):
        if ws < 0 or ws >= len(names):
            # unknown workspace in the snapshot: park it in the SORT bucket
            if fallback is None:
                continue
            ws, unsorted = fallback, unsorted + 1
        win = by_id.get(wid)
        how = "id"
        if win is None:
            cands = by_title.get(title) or []
            if len(cands) == 1:
                win, how = cands[0], "title"
        if win is None:
            unmatched += 1
            continue
        if win["ws"] == ws:
            continue  # already where it belongs
        rows.append({
            "id": win["id"], "cur_ws": win["ws"], "class": win["class"],
            "title": win["title"], "ws": str(ws), "corrected": False,
            "how": f"backup:{how}",
        })

    print(f"[*] {unmatched} snapshot entries no longer exist")
    if unsorted:
        print(f"[*] {unsorted} entries had an unknown workspace -> catch-all "
              f"{names[fallback]!r}")
    print(f"[*] {len(rows)} live windows would move")
    if not rows:
        print("[*] nothing to restore")
        return 0

    if report:
        try:
            save_report(report, {
                "desktop": None, "desktop_name": None, "workspaces": names,
                "rows": rows, "accepted": rows, "backup": path,
            })
        except OSError as exc:
            print(f"[!] could not write report: {exc}")

    if not apply_moves:
        # dry run: show the plan, one line per window, and change nothing
        for r in rows:
            print(f"    {names[int(r['ws'])]:<12} - {r['title']}")
        print("[*] dry run -- rerun with --apply to move the windows")
        return 0

    # --apply: every window we managed to match, no prompting. Snapshot
    # entries whose window is gone are simply ignored.
    accepted = rows

    todo = [r for r in accepted
            if r["ws"] is not None and int(r["ws"]) != r["cur_ws"]]
    snap = save_snapshot() if todo else None
    moved, failed = 0, []
    for r in todo:
        if move_window(r["id"], int(r["ws"])):
            moved += 1
        else:
            failed.append(r)
    print(f"[*] restored {moved} windows")
    if failed:
        print("[!] could not move " + ", ".join(f["id"] for f in failed[:10]))
    if snap:
        print("[*] undo with: ./workspace_guesser.py --undo")
    return 0


def bump_to_front(wins: list[dict], focus: bool = False) -> int:
    """Restack windows above the others, keeping their relative stacking order.

    Iconified windows are mapped first, otherwise raising them would have no
    visible effect.
    """
    from Xlib import X, display

    d = display.Display()
    ok = 0
    try:
        root = d.screen().root
        a_hidden = d.intern_atom("_NET_WM_STATE_HIDDEN")
        prop = root.get_full_property(d.intern_atom("_NET_CLIENT_LIST_STACKING"),
                                      X.AnyPropertyType)
        order = list(prop.value) if prop else []
        rank = {wid: i for i, wid in enumerate(order)}
        ordered = sorted(wins,
                         key=lambda w: rank.get(int(w["id"], 16), len(order)))
        for w in ordered:
            try:
                win = d.create_resource_object("window", int(w["id"], 16))
                state = win.get_full_property(a_hidden, X.AnyPropertyType)
                if state and a_hidden in state.value:
                    win.map()          # was iconified: raise without unmapping
                    d.flush()
                win.configure(stack_mode=X.Above)
                if focus:
                    win.set_input_focus(X.RevertToParent, X.CurrentTime)
                ok += 1
            except Exception:  # noqa: BLE001
                continue
        d.flush()
        d.sync()
    finally:
        try:
            d.close()
        except Exception:  # noqa: BLE001
            pass
    return ok


def select_for_bump(desktop, topicless_only, class_needles, class_exact,
                    titles, limit):
    """Windows matching the bump selectors, newest stack order preserved later."""
    wins = list_windows(desktop, class_needles=class_needles,
                        class_exact=class_exact)
    if topicless_only:
        picks = [w for w in wins if is_contentless(w)]
    else:
        picks = wins
        if titles:
            lows = [t.lower() for t in titles]
            picks = [w for w in picks
                     if any(t in w["title"].lower() or t in (w["clean"] or "").lower()
                            for t in lows)]
    return picks[:limit] if limit else picks


def do_bump(desktop, names, topicless_only, class_needles, class_exact,
            titles, limit, focus) -> int:
    picks = select_for_bump(desktop, topicless_only, class_needles,
                            class_exact, titles, limit)
    scope = ("every workspace" if desktop is None
             else f"workspace {ws_label(desktop, names[desktop])}"
             if 0 <= desktop < len(names) else f"workspace {desktop}")
    what = "topicless '~' window(s)" if topicless_only else "window(s)"
    print(f"[*] {len(picks)} {what} on {scope}"
          + (f", matching class={class_needles}" if class_needles else "")
          + (f", title~={titles}" if titles else ""))
    for w in picks:
        where = (f"ws {ws_label(w['ws'], names[w['ws']])}"
                 if w["ws"] is not None and 0 <= w["ws"] < len(names) else "ws ?")
        print(f"      {w['id']}  [{w['class']}] {where:<12} {w['title'][:60]}")
    if not picks:
        print("[*] nothing matched")
        return 0
    ok = bump_to_front(picks, focus=focus)
    print(f"[*] bumped {ok}/{len(picks)} to the front"
          + (" (last one focused)" if focus else ""))
    return 0


def load_report(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_report(path: str, data: dict) -> None:
    # never silently destroy a previous (possibly larger) run
    if os.path.exists(path) and os.path.getsize(path) > 0:
        bak = f"{path}.bak"
        if not os.path.exists(bak):
            shutil.copy2(path, bak)
            print(f"[*] previous report kept as {bak}")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    print(f"[*] report -> {path}")


def review(path: str, apply_moves: bool, report: str) -> int:
    """Confirm an existing report interactively, then optionally apply it."""
    data = load_report(path)
    names = data["workspaces"]
    rows = data["rows"]
    # every workspace: windows in the report may already have been moved
    live = {norm_id(w["id"]) for w in list_windows(None)}
    rows = [r for r in rows if norm_id(r["id"]) in live]
    dropped = len(data["rows"]) - len(rows)
    if dropped:
        print(f"[*] {dropped} window(s) from the report are no longer open")
    print(f"[*] reviewing {len(rows)} windows, no LLM involved")

    accepted, stopped = confirm(rows, names, False, load_meanings())
    if stopped:
        print("[*] stopped early -- only the entries confirmed so far are kept")

    data["rows"] = rows
    data["accepted"] = accepted
    try:
        save_report(report, data)
    except OSError as exc:
        print(f"[!] could not write report: {exc}")

    if apply_moves:
        moved = 0
        failed = []
        for r in accepted:
            if r["ws"] is None or int(r["ws"]) == r["cur_ws"]:
                continue
            if move_window(r["id"], int(r["ws"])):
                moved += 1
            else:
                failed.append(r)
        print(f"[*] moved {moved} windows")
        if failed:
            print("[!] could not move " + ", ".join(f["id"] for f in failed[:10]))
    else:
        print("[*] dry run -- rerun with --apply to move the windows")
    return 0


ALL_DESKTOPS = "all"


def desktop_arg(value: str):
    """--desktop takes a workspace number or the word 'all'."""
    v = value.strip()
    if v.lower() == ALL_DESKTOPS:
        return ALL_DESKTOPS
    try:
        return int(v)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--desktop: expected a number or 'all', got {value!r}")


def desktop_scope(value, current: int):
    """Resolve --desktop: unset -> current workspace, 'all' -> None (every)."""
    if value is None:
        return current
    if isinstance(value, str) and value.lower() == ALL_DESKTOPS:
        return None
    return value


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default=".env")
    ap.add_argument("--llm-url", default=None,
                    help="Ollama base URL (default: AI_OLLAMA from .env, "
                         "falls back to http://127.0.0.1:11434)")
    ap.add_argument("--cache-dir", default="/tmp/opencode/wsguess")
    ap.add_argument("--desktop", type=desktop_arg, default=None,
                    help="workspace index to inspect, or 'all' for every "
                         "workspace (default: the current one)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N windows")
    ap.add_argument("--bump-unknowns", action="store_true",
                    help="bring the bare '~' shells of --desktop to the "
                         "front of the stacking order, then exit")
    ap.add_argument("--bump", action="store_true",
                    help="bring the windows selected by --class/--title to the "
                         "front of the stacking order, then exit")
    ap.add_argument("--title", dest="titles", action="append", default=None,
                    metavar="TEXT",
                    help="partial window-title match for --bump "
                         "(case-insensitive, repeatable)")
    ap.add_argument("--focus", action="store_true",
                    help="with --bump-unknowns, focus the last window raised")
    ap.add_argument("--list-classes", action="store_true",
                    help="list the open WM_CLASS values from `wmctrl -lx` "
                         "(honours --desktop) and exit")
    ap.add_argument("--class", dest="classes", action="append", default=None,
                    metavar="PATTERN",
                    help="only windows whose WM_CLASS matches PATTERN "
                         "(case-insensitive substring; repeatable)")
    ap.add_argument("--class-exact", action="store_true",
                    help="match --class exactly instead of as a substring")
    ap.add_argument("--batch", type=int, default=12, help="windows per LLM query")
    ap.add_argument("--jobs", type=int, default=1, help="parallel LLM batches")
    ap.add_argument("--llm-timeout", type=int, default=DEFAULT_LLM_TIMEOUT,
                    help="seconds per LLM query (default 900 = 15 min)")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--model", default=None)
    ap.add_argument("--num-predict", type=int, default=8000,
                    help="cap on generated tokens per query (reasoning models "
                         "need room; each query is allowed --llm-timeout seconds)")
    ap.add_argument("--ocr-lang", default="eng+est")
    ap.add_argument("--ocr-max-chars", type=int, default=1200)
    ap.add_argument("--no-ocr", action="store_true", help="skip the OCR fallback")
    ap.add_argument("--max-ocr", type=int, default=0,
                    help="cap OCR escalations (0 = unlimited)")
    ap.add_argument("--unknown-ws", default=None,
                    help="catch-all workspace for windows that cannot be "
                         "classified (default: the workspace named SORT)")
    ap.add_argument("--yes", action="store_true", help="accept everything (no prompts)")
    ap.add_argument("--apply", action="store_true",
                    help="move windows to the chosen workspaces at the end")
    ap.add_argument("--all-workspaces", action="store_true",
                    help="also classify windows on every workspace, not just "
                         "--desktop")
    ap.add_argument("--no-strict-retry", action="store_true",
                    help="do not retry all-UNKNOWN batches with stricter rules")
    ap.add_argument("--repair-desktop", action="store_true",
                    help="offline: rewrite malformed _NET_WM_DESKTOP "
                         "properties without moving any window")
    ap.add_argument("--apply-report", default=None,
                    help="offline: just apply a reviewed report, no LLM")
    ap.add_argument("--review", default=None, metavar="REPORT",
                    help="offline: confirm an existing report interactively, "
                         "no LLM and no OCR")
    ap.add_argument("--undo", action="store_true",
                    help="move every window back to the state recorded by the "
                         "last pre-operation snapshot, then exit")
    ap.add_argument("--restore-from", default=None, metavar="BACKUP",
                    help="offline: restore window placements from a "
                         "'wmctrl -l' snapshot (file or directory), no LLM")
    ap.add_argument("--report", default="workspace_guesser_report.json")
    args = ap.parse_args()

    # Offline modes: no window enumeration for classification, no translation,
    # no LLM.
    if args.undo:
        return undo(args.restore_from or BACKUP_DIR)
    if args.bump_unknowns or args.bump:
        names, current = get_workspaces()
        desktop = desktop_scope(args.desktop, current)
        return do_bump(desktop, names, args.bump_unknowns, args.classes,
                       args.class_exact, args.titles, args.limit, args.focus)
    if args.list_classes:
        scope = None if args.all_workspaces else desktop_scope(
            args.desktop, get_workspaces()[1])
        list_classes(scope, args.classes[0] if args.classes else None)
        return 0
    if args.repair_desktop:
        print(f"[*] repaired _NET_WM_DESKTOP type on {normalise_desktop_props()} "
              f"window(s); nothing was moved")
        return 0
    if args.apply_report:
        apply_report(args.apply_report)
        return 0
    if args.restore_from:
        return restore(args.restore_from, args.apply,
                       args.report if args.report != "workspace_guesser_report.json"
                       else None)
    if args.review:
        return review(args.review, apply_moves=args.apply, report=args.report)

    for tool in ("wmctrl", "xprop", "xwininfo", "import", "convert", "tesseract"):
        if not shutil.which(tool):
            sys.exit(f"[!] missing required tool: {tool}")

    env = load_env(args.env)
    names, current = get_workspaces()
    desktop = desktop_scope(args.desktop, current)

    meanings = load_meanings()
    print(f"[*] {len(names)} workspaces, current = "
          f"{ws_label(desktop, names[desktop], meanings)}")
    known = [n for n in (ws_key(x) for x in names)
             if ws_meaning(n, meanings)]
    print(f"[*] loaded {len(meanings['meanings'])} workspace meanings "
          f"({len(known)} in use) + {len(meanings['rules'])} rules "
          f"from {MEANINGS_PATH}")

    # learn the app names in use before any title is cleaned
    set_app_names([c for c, _ in (wmctrl_classes(None) or xlib_classes(None))])

    # learn the app names in use before any title is cleaned
    set_app_names([c for c, _ in (wmctrl_classes(None) or xlib_classes(None))])

    wins = list_windows(desktop, only_visible=False,
                        class_needles=args.classes, class_exact=args.class_exact)
    if args.all_workspaces:
        seen = {w["id"] for w in wins}
        wins += [w for w in list_windows(None, class_needles=args.classes,
                                         class_exact=args.class_exact)
                 if w["id"] not in seen]
    if args.classes:
        print(f"[*] WM_CLASS filter: {', '.join(args.classes)}"
              + (" (exact)" if args.class_exact else " (substring)"))
    if args.limit:
        wins = wins[:args.limit]
    print(f"[*] {len(wins)} windows on workspace {desktop}")
    for w in wins:
        print(f"      {w['id']}  [{w['class']}]  {w['clean'] or w['title']}")

    cache_path = os.path.join(args.cache_dir, "translate_cache.json")
    cache = {}
    if os.path.exists(cache_path):
        try:
            cache = json.load(open(cache_path, encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cache = {}
    translator = Translator(env.get("API_LIBRETRANSLATE", ""), cache_path, cache)

    print("[*] translating Estonian titles …", flush=True)
    translated = translator.translate([w["title"] for w in wins])
    print(f"[*] translated {len(translated)} distinct titles"
          + (" (translator unavailable, continuing)" if translator.failed else ""))

    llm_url = args.llm_url or env.get("AI_OLLAMA", "") or "http://127.0.0.1:11434"
    llm = LLM(llm_url, args.model, args.llm_timeout, args.retries,
              args.num_predict)
    print(f"[*] LLM = {llm.model} @ {llm.base}, {args.llm_timeout}s per query")

    # bare home shells carry no topic: keep them out of the LLM round entirely
    todo = [i for i, w in enumerate(wins) if not is_contentless(w)]
    no_topic = [i for i, w in enumerate(wins) if is_contentless(w)]
    results: list[str | None] = [None] * len(wins)
    skipped: set[int] = set()  # windows the model never answered at all
    batches = [[todo[s + k] for k in range(min(args.batch, len(todo) - s))]
               for s in range(0, len(todo), args.batch)]
    if no_topic:
        print(f"[*] {len(no_topic)} window(s) have no topic at all (a bare "
              f"'~' shell); skipped by the LLM"
              + (" and left untouched (--no-ocr)" if args.no_ocr
                 else ", so they are only decided by OCR"))

    print(f"[*] classifying {len(todo)} windows in {len(batches)} title batches ...",
          flush=True)
    t0 = time.time()

    def run_batch(pair):
        i, idxs = pair
        batch = [wins[k] for k in idxs]
        answers, missing = classify_by_title(llm, names, batch, translated,
                                             meanings=meanings)
        for k, a in enumerate(answers):
            results[idxs[k]] = normalise_ws(a, names)
        for m in missing:
            skipped.add(idxs[m])
        print(f"    batch {i + 1}/{len(batches)} done "
              f"({time.time() - t0:.0f}s elapsed, {len(missing)} unanswered)",
              flush=True)
        return i

    if args.jobs > 1:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(run_batch, list(enumerate(batches))))
    else:
        for pair in enumerate(batches):
            run_batch(pair)

    # A batch that came back entirely UNKNOWN means the prompt (or the model)
    # failed, not that every window is unclassifiable. Retry it in strict mode.
    if not args.no_strict_retry:
        for i, idxs in enumerate(batches):
            chunk = [results[k] for k in idxs]
            if not chunk or all(r is None for r in chunk):
                print(f"    batch {i + 1}/{len(batches)} was all-UNKNOWN, "
                      f"retrying with stricter rules", flush=True)
                batch = [wins[k] for k in idxs]
                answers, _ = classify_by_title(llm, names, batch, translated,
                                               repair=False, strict=True,
                                               meanings=meanings)
                for k, a in enumerate(answers):
                    results[idxs[k]] = normalise_ws(a, names)

    unresolved = [i for i, r in enumerate(results) if r is None]
    if skipped:
        print(f"[!] {len(skipped)} windows got no answer at all even after the "
              f"repair pass: {[wins[i]['title'] for i in sorted(skipped)[:5]]}")
    print(f"[*] title pass placed "
          f"{len(todo) - len([i for i in unresolved if i in todo])}/{len(todo)} "
          f"windows" + (f" ({len(no_topic)} skipped as topicless)" if no_topic else ""))

    if unresolved and not args.no_ocr:
        if args.max_ocr:
            unresolved = unresolved[:args.max_ocr]
        print(f"[*] OCR fallback for {len(unresolved)} windows …", flush=True)
        shot_dir = os.path.join(args.cache_dir, "screenshots")
        ocr_placed: set[int] = set()
        for n, i in enumerate(unresolved, 1):
            w = wins[i]
            text = ocr_window(w, shot_dir, args.ocr_lang, args.ocr_max_chars)
            if text:
                tr = translator.translate([text[:400]])
                translated.update(tr)
            print(f"  [{n}/{len(unresolved)}] {w['id']} "
                  f"ocr={len(text)} chars", flush=True)
            if not text:
                continue
            ans = classify_by_ocr(llm, names, w, text, translated,
                                  meanings=meanings)
            results[i] = normalise_ws(ans, names)
            if results[i] is not None:
                ocr_placed.add(i)
        still = [i for i in unresolved if results[i] is None]
        print(f"[*] OCR pass placed {len(unresolved) - len(still)}/{len(unresolved)}")
    else:
        ocr_placed = set()

    fallback = (int(args.unknown_ws) if args.unknown_ws is not None
                else find_sort_ws(names))
    if fallback is not None and 0 <= fallback < len(names):
        stranded = [i for i, r in enumerate(results)
                    if r is None and i not in no_topic]
        catchall = set(stranded)
        if stranded:
            print(f"[*] {len(stranded)} unclassified window(s) -> catch-all "
                  f"workspace {names[fallback]!r}")
            for i in stranded:
                results[i] = str(fallback)
    else:
        catchall = set()
        print("[!] no catch-all workspace: unclassified windows stay put "
              "(pass --unknown-ws N)")

    rows = []
    left_alone: list[str] = []
    for i, (w, r) in enumerate(zip(wins, results)):
        if i in no_topic and r is None:
            # skipped by the LLM and OCR had nothing to say (or was disabled):
            # not a result at all, the window just stays where it is
            left_alone.append(w["id"])
            continue
        rows.append({
            "id": w["id"], "cur_ws": w["ws"], "class": w["class"],
            "title": w["title"], "ws": r, "corrected": False,
            "how": ("catch-all" if i in catchall
                    else "ocr" if i in ocr_placed
                    else "title" if r is not None
                    else "unclassified"),
        })
    if left_alone:
        print(f"[*] {len(left_alone)} topicless window(s) gained no answer "
              f"from OCR; left where they are and left out of the results")

    accepted, quit_early = confirm(rows, names, args.yes, meanings)
    if quit_early:
        print("[*] stopped early -- windows up to that point were still accepted")

    try:
        save_report(args.report, {
            "desktop": desktop,
            "desktop_name": names[desktop],
            "workspaces": names,
            "rows": rows,
            "accepted": accepted,
            "left_alone": left_alone,
        })
    except OSError as exc:
        print(f"[!] could not write report: {exc}")

    if args.apply and accepted:
        todo = [r for r in accepted
                if r["ws"] is not None and int(r["ws"]) != r["cur_ws"]]
        snap = save_snapshot() if todo else None
        moved = 0
        failed = []
        for r in todo:
            if move_window(r["id"], int(r["ws"])):
                moved += 1
            else:
                failed.append(r)
        print(f"[*] moved {moved} windows")
        if failed:
            print(f"[!] could not move {len(failed)}: "
                  + ", ".join(f["id"] for f in failed[:5]))
        if snap:
            print("[*] undo with: ./workspace_guesser.py --undo")

    return 0


if __name__ == "__main__":
    sys.exit(main())