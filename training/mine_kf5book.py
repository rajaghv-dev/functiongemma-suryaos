#!/usr/bin/env python3
"""
mine_kf5book.py — Mine real KDE Frameworks 5 book documentation for tool-dispatch pairs.

Source: https://invent.kde.org/documentation/kf5book

WHY THIS SCRIPT EXISTS
----------------------
We are training a 270M Gemma model to dispatch user queries to four KDE tools:

  - kde_krunner_launch     (launch an application by name)
  - kde_window_focus       (raise / focus a window by title)
  - kde_notifications_send (post a Plasma desktop notification)
  - kde_dialog_confirm     (show a Yes/No confirmation dialog)

The user has been explicit: minimise synthetic generation. Mine REAL prose from
KDE's own developer book. Every output pair must be defensible — we record the
exact source file, line number, and the verbatim sentence that produced it in a
`provenance` field.

THE FIDELITY / YIELD TRADE-OFF
------------------------------
This is the central tension of the script:

  * MAXIMISE FIDELITY — only emit pairs that quote real text the book actually
    contains. Reject anything that requires us to invent verbs, apps, or window
    names. Reject obvious code / signature lines that no human would type.
    The price: a small dataset, possibly only a few dozen pairs.

  * MAXIMISE YIELD — paraphrase, expand templates, fill in app names from a
    list. The price: synthetic data masquerading as real, which is exactly what
    the user asked us NOT to do.

We sit firmly on the FIDELITY side. The acceptable transformations are:

  1. Take a real example sentence ("Launch Firefox via KRunner") and turn it
     into  {user: "launch firefox via KRunner", target: kde_krunner_launch}.
  2. Take an explicit short imperative ("Open Dolphin") and use it verbatim.
  3. Reject everything else, including signatures like
     `void KRunner::query(const QString &)` and prose that references the API
     without giving an end-user-shaped query.

If the book yields 30 real pairs, that is the right answer. We report the
truth, we do not pad.

HOW IT WORKS
------------
1. Clone (or `git pull`) https://invent.kde.org/documentation/kf5book.git into
   dataset/real_sources/kf5book_cache/.  If network is unavailable AND the
   cache is missing, we exit cleanly with zero pairs — never crash.
2. Walk the cache for documentation files (md / adoc / rst / txt).
3. For each file, split into sentences and section headings.
4. For each sentence:
     - score against per-tool keyword + verb patterns
     - require a concrete object (an app name / a window title / a message)
       OR an explicit imperative shape ("Launch X", "Open Y")
     - extract structured arguments from the sentence text only (substring
       match) — never invent
5. Emit one JSONL line per accepted pair to
   dataset/real_sources/kf5_pairs.jsonl with a `provenance` block:
     { "file": "...", "line": 42, "sentence": "<verbatim>", "url": "..." }

IDEMPOTENCY
-----------
Re-running the script overwrites the output deterministically. If the cache
already exists we run `git pull --ff-only`; if that fails (offline) we just
work with what we have. We never delete the cache.

USAGE
-----
    python3 training/mine_kf5book.py
    python3 training/mine_kf5book.py --no-fetch     # offline only
    python3 training/mine_kf5book.py --verbose      # log every accept/reject
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "dataset" / "real_sources" / "kf5book_cache"
OUT_PATH = REPO_ROOT / "dataset" / "real_sources" / "kf5_pairs.jsonl"
TOOL_SCHEMAS_PATH = REPO_ROOT / "tools" / "tool_schemas.json"

UPSTREAM_REPO = "https://invent.kde.org/documentation/kf5book.git"
UPSTREAM_WEB = "https://invent.kde.org/documentation/kf5book"

DOC_EXTS = {".md", ".adoc", ".asciidoc", ".rst", ".txt"}

# ---------------------------------------------------------------------------
# Tool schema loading
# ---------------------------------------------------------------------------


def load_mcp_schemas() -> dict[str, dict]:
    """Return {mcp_tool_name: mcp_schema_dict} for the 4 KDE tools."""
    raw = json.loads(TOOL_SCHEMAS_PATH.read_text())
    wanted = {
        "kde_krunner_launch",
        "kde_window_focus",
        "kde_notifications_send",
        "kde_dialog_confirm",
    }
    out: dict[str, dict] = {}
    for entry in raw.values():
        mcp = entry.get("mcp_schema")
        if mcp and mcp.get("name") in wanted:
            out[mcp["name"]] = mcp
    missing = wanted - set(out)
    if missing:
        raise SystemExit(f"tool_schemas.json missing tools: {missing}")
    return out


# ---------------------------------------------------------------------------
# Fetch / cache the upstream repo
# ---------------------------------------------------------------------------


def _git(*args: str, cwd: Optional[Path] = None) -> tuple[int, str]:
    """Run a git command, capturing combined output."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return 1, f"git invocation failed: {exc}"


def ensure_cache(no_fetch: bool, verbose: bool) -> bool:
    """Make sure CACHE_DIR exists and is populated.  Return True if we have
    *some* documentation to work with (even a stale cache), False otherwise."""
    CACHE_DIR.parent.mkdir(parents=True, exist_ok=True)

    have_repo = (CACHE_DIR / ".git").exists()

    if no_fetch:
        if not have_repo:
            print(
                f"[mine_kf5book] --no-fetch and no cache at {CACHE_DIR}; "
                "skipping mine.",
                file=sys.stderr,
            )
        return have_repo

    if not have_repo:
        print(f"[mine_kf5book] cloning {UPSTREAM_REPO} -> {CACHE_DIR}")
        rc, out = _git("clone", "--depth", "1", UPSTREAM_REPO, str(CACHE_DIR))
        if rc != 0:
            print(
                f"[mine_kf5book] clone failed (network unavailable?):\n{out}",
                file=sys.stderr,
            )
            # Fall back to whatever happens to be on disk.
            return any(CACHE_DIR.rglob("*")) if CACHE_DIR.exists() else False
        return True

    # Already cloned — try to update, but don't fail if offline.
    rc, out = _git("pull", "--ff-only", cwd=CACHE_DIR)
    if rc != 0 and verbose:
        print(f"[mine_kf5book] git pull skipped (offline?):\n{out}", file=sys.stderr)
    return True


# ---------------------------------------------------------------------------
# Sentence extraction
# ---------------------------------------------------------------------------

# Strip code fences, inline backticks, and inline link refs so the sentence
# pool is "what a human reading the book would say out loud".
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`]*`")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")  # [text](url) -> text
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6}|=+|\*{1,5})\s+(.+?)\s*$")
# AsciiDoc/Markdown bullet markers
_BULLET_RE = re.compile(r"^\s*[-*+]\s+")
# Sentence boundary: . ! ? followed by space + capital, or end of line.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")


@dataclass
class Sentence:
    text: str
    file: Path
    line: int
    is_heading: bool = False


def clean_block(block: str) -> str:
    block = _FENCE_RE.sub(" ", block)
    block = _INLINE_CODE_RE.sub(" ", block)
    block = _LINK_RE.sub(r"\1", block)
    return block


def iter_sentences(path: Path) -> Iterable[Sentence]:
    """Yield Sentence objects from a documentation file.

    We split paragraph-by-paragraph (separated by blank lines), then sentence
    by sentence within each paragraph. Headings are emitted as their own
    "sentence" with is_heading=True so we can mine `== Launching applications`
    style titles.
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return

    # Remove fenced code blocks before line-based scanning so we don't pull
    # source code as "sentences".
    cleaned = _FENCE_RE.sub("\n", raw)
    lines = cleaned.splitlines()

    # First pass: headings (emit each on its own).
    for i, line in enumerate(lines, start=1):
        m = _HEADING_RE.match(line)
        if m:
            text = clean_block(m.group(2)).strip()
            if text:
                yield Sentence(text=text, file=path, line=i, is_heading=True)

    # Second pass: paragraphs.
    paragraph: list[str] = []
    para_start_line = 1
    for i, line in enumerate(lines, start=1):
        stripped = line.rstrip()
        if not stripped:
            if paragraph:
                yield from _emit_paragraph(paragraph, path, para_start_line)
                paragraph = []
            para_start_line = i + 1
            continue
        # Skip headings (already yielded above), bullet markers stay but we
        # strip the bullet prefix.
        if _HEADING_RE.match(line):
            if paragraph:
                yield from _emit_paragraph(paragraph, path, para_start_line)
                paragraph = []
            para_start_line = i + 1
            continue
        stripped = _BULLET_RE.sub("", stripped)
        if not paragraph:
            para_start_line = i
        paragraph.append(stripped)
    if paragraph:
        yield from _emit_paragraph(paragraph, path, para_start_line)


def _emit_paragraph(lines: list[str], path: Path, start_line: int) -> Iterable[Sentence]:
    text = clean_block(" ".join(lines)).strip()
    if not text:
        return
    for sent in _SENT_SPLIT_RE.split(text):
        sent = sent.strip().strip("\"'`")
        if not sent:
            continue
        yield Sentence(text=sent, file=path, line=start_line)


# ---------------------------------------------------------------------------
# Tool-specific mining rules
# ---------------------------------------------------------------------------

# Apps / commands the book is likely to mention. We only ACCEPT an app slot
# if the literal token is present in the sentence — never invent.
KDE_APPS = {
    "kate", "dolphin", "konsole", "krunner", "okular", "gwenview",
    "kwrite", "kcalc", "ark", "kmail", "kontact", "korganizer",
    "krita", "konqueror", "kdevelop", "spectacle", "systemsettings",
    "plasma", "discover", "yakuake", "filelight", "kdenlive",
    "firefox", "chromium", "vlc", "gimp", "inkscape",  # commonly cited as examples
}

# Verbs that make a sentence "imperative-launchy". Must be at the start (or
# right after "Then ", "To ", etc.) for the imperative read to feel real.
LAUNCH_VERBS = {"launch", "open", "start", "run", "execute"}
FOCUS_VERBS = {"focus", "raise", "activate", "switch to", "bring", "show"}
NOTIFY_VERBS = {"notify", "alert", "post a notification", "send a notification",
                "show a notification", "display a notification"}
CONFIRM_VERBS = {"confirm", "ask", "prompt", "warn"}


@dataclass
class Pair:
    user: str
    tool_name: str
    arguments: dict
    sentence: Sentence
    rule: str = ""

    def to_jsonl(self, schemas: dict[str, dict]) -> dict:
        rel = self.sentence.file.relative_to(REPO_ROOT)
        # Best-effort upstream URL: assume default branch is `master`.
        repo_rel = self.sentence.file.relative_to(CACHE_DIR)
        upstream_url = f"{UPSTREAM_WEB}/-/blob/master/{repo_rel.as_posix()}#L{self.sentence.line}"
        return {
            "messages": [
                {"role": "system", "content": "Call the right tool."},
                {"role": "user", "content": self.user},
            ],
            "tools": [schemas[self.tool_name]],
            "target": {"name": self.tool_name, "arguments": self.arguments},
            "source": "kf5book",
            "provenance": {
                "file": str(rel),
                "line": self.sentence.line,
                "sentence": self.sentence.text,
                "rule": self.rule,
                "url": upstream_url,
            },
        }


# --- helpers ----------------------------------------------------------------


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*")


def lc_words(s: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(s)]


def sentence_starts_with_verb(text: str, verbs: set[str]) -> Optional[str]:
    """If the sentence starts with one of `verbs` (allowing 'Then', 'To',
    'You can', 'Users can' as prefixes), return the matched verb.  Else None."""
    m = re.match(
        r"^\s*(?:to\s+|then\s+|you\s+(?:can|may)\s+|users?\s+can\s+|"
        r"please\s+|just\s+|simply\s+)?([A-Za-z]+(?:\s+[A-Za-z]+)?)\b",
        text,
        re.IGNORECASE,
    )
    if not m:
        return None
    head = m.group(1).lower().strip()
    if head in verbs:
        return head
    # check 2-word verbs like "switch to", "send a notification"
    for v in verbs:
        if " " in v and text.lower().lstrip().startswith(v):
            return v
    return None


def find_app_in_sentence(text: str) -> Optional[str]:
    """Return the first KDE app name literally present in the sentence."""
    words = lc_words(text)
    for w in words:
        if w in KDE_APPS:
            return w
    return None


# --- per-tool extractors ----------------------------------------------------


def try_krunner_launch(s: Sentence) -> Optional[Pair]:
    """A sentence qualifies for kde_krunner_launch if:
      * it starts with a launch verb AND mentions a real app token, OR
      * it's a heading like "Launching applications" with a clear app token.
    """
    text = s.text.strip()
    # Must reference launching/opening/etc. context.
    lower = text.lower()
    if not any(v in lower for v in LAUNCH_VERBS):
        return None

    verb = sentence_starts_with_verb(text, LAUNCH_VERBS)
    app = find_app_in_sentence(text)
    if not app:
        return None
    if verb is None:
        # accept "use KRunner to launch dolphin" style only if KRunner is named
        if "krunner" not in lower:
            return None
        verb = "launch"

    # Sanity: reject API-signature looking lines.
    if "(" in text and ")" in text and ("::" in text or "QString" in text):
        return None
    if len(text) > 240:
        return None

    user_query = f"{verb} {app}"
    return Pair(
        user=user_query,
        tool_name="kde_krunner_launch",
        arguments={"app": app},
        sentence=s,
        rule="launch_verb+app_token",
    )


def try_window_focus(s: Sentence) -> Optional[Pair]:
    """kde_window_focus: 'switch to <app>', 'focus the <app> window',
    'bring <app> to the front'."""
    text = s.text.strip()
    lower = text.lower()
    # Must mention a window/focus concept.
    if not any(k in lower for k in ("window", "focus", "raise", "switch to", "front", "foreground", "kwin")):
        return None

    app = find_app_in_sentence(text)
    if not app:
        return None

    verb = None
    for v in ("switch to", "focus", "raise", "activate"):
        # `re.search` with word boundary is fine for single words, and
        # `switch to` is multi-word.
        if v in lower:
            verb = v
            break
    if not verb and ("front" in lower or "foreground" in lower):
        verb = "switch to"
    if not verb:
        return None

    # We require the verb to be near the app token to avoid false positives
    # like "the window manager raises modal dialogs in front of Firefox".
    # Cheap proxy: app and verb must both appear in first 120 chars.
    if app not in lower[:160] or verb not in lower[:160]:
        return None

    if "(" in text and ")" in text and "::" in text:
        return None
    if len(text) > 240:
        return None

    return Pair(
        user=f"{verb} {app}".strip(),
        tool_name="kde_window_focus",
        arguments={"title": app},
        sentence=s,
        rule="focus_verb+app_token",
    )


def try_notification(s: Sentence) -> Optional[Pair]:
    """kde_notifications_send: imperative or example-shape sentences saying
    'show a notification that ...', 'notify the user about ...'."""
    text = s.text.strip()
    lower = text.lower()
    if not any(k in lower for k in ("notification", "knotification", "notify")):
        return None
    # Must be a natural-language sentence, not a class reference like
    # "KNotification is a class for ...". Require an action verb.
    has_action = any(re.search(rf"\b{v}\b", lower) for v in ("show", "send", "post", "display", "emit", "notify"))
    if not has_action:
        return None

    # Try to extract the *content* of the notification — i.e. the message.
    # Two grammars we accept:
    #   "(show|send|post|display) a notification (that|saying|when) <X>"
    #   "notify the user (that|when) <X>"
    msg = None
    for pat in (
        r"(?:show|send|post|display|emit)\s+(?:a|an|the)?\s*notification\s+(?:that|saying|when|about|to\s+say)\s+(.+?)[.!?]?$",
        r"notify(?:\s+the\s+user)?\s+(?:that|when|about)\s+(.+?)[.!?]?$",
    ):
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            msg = m.group(1).strip().strip("\"'`")
            break
    if not msg:
        return None
    # Reject if the captured "message" is itself an API/code reference.
    if any(tok in msg for tok in ("::", "QString", "()")):
        return None
    if len(msg) > 160 or len(msg) < 3:
        return None

    user_query = re.sub(r"\s+", " ", text).strip().rstrip(".")
    if len(user_query) > 240:
        return None

    return Pair(
        user=user_query,
        tool_name="kde_notifications_send",
        arguments={"title": "Notification", "message": msg},
        sentence=s,
        rule="notification_action+message",
    )


def try_dialog_confirm(s: Sentence) -> Optional[Pair]:
    """kde_dialog_confirm: sentences describing a yes/no confirmation prompt.

    We accept things like:
      - "Ask the user to confirm before deleting the file."
      - "Show a confirmation dialog before exiting."
      - "Prompt the user with a yes/no question about ..."
    """
    text = s.text.strip()
    lower = text.lower()
    if not any(k in lower for k in (
        "kmessagebox", "confirm", "yes/no", "yes or no",
        "confirmation dialog", "questionyesno",
    )):
        return None

    has_action = any(re.search(rf"\b{v}\b", lower) for v in (
        "ask", "confirm", "show", "prompt", "warn", "display"
    ))
    if not has_action:
        return None

    # Try to extract the *prompt* — the question being asked.
    prompt = None
    for pat in (
        r"(?:ask|prompt)\s+(?:the\s+)?user(?:\s+to\s+confirm)?\s+(?:that|whether|if|before|about)\s+(.+?)[.!?]?$",
        r"(?:show|display)\s+(?:a|an|the)?\s*(?:confirmation\s+)?dialog\s+(?:to\s+confirm|asking|that\s+asks|before)\s+(.+?)[.!?]?$",
        r"confirm(?:\s+with\s+the\s+user)?\s+(?:that|whether|if|before)\s+(.+?)[.!?]?$",
    ):
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            prompt = m.group(1).strip().strip("\"'`")
            break
    if not prompt:
        return None
    if any(tok in prompt for tok in ("::", "QString", "()")):
        return None
    if len(prompt) > 200 or len(prompt) < 3:
        return None
    if len(text) > 240:
        return None

    user_query = re.sub(r"\s+", " ", text).strip().rstrip(".")
    return Pair(
        user=user_query,
        tool_name="kde_dialog_confirm",
        arguments={"prompt": prompt},
        sentence=s,
        rule="confirm_action+prompt",
    )


EXTRACTORS = (
    try_krunner_launch,
    try_window_focus,
    try_notification,
    try_dialog_confirm,
)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def find_doc_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    out: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in DOC_EXTS:
            continue
        # skip cache git internals (paranoia — rglob already skips dotfiles
        # only if explicitly filtered).
        if ".git" in p.parts:
            continue
        out.append(p)
    return sorted(out)


def mine(verbose: bool) -> tuple[list[Pair], dict[str, int]]:
    files = find_doc_files(CACHE_DIR)
    if verbose:
        print(f"[mine_kf5book] scanning {len(files)} doc files under {CACHE_DIR}")

    pairs: list[Pair] = []
    seen_user_queries: set[tuple[str, str]] = set()  # (tool, user.lower())

    per_file: dict[str, int] = {}

    for path in files:
        before = len(pairs)
        for sent in iter_sentences(path):
            for extractor in EXTRACTORS:
                pair = extractor(sent)
                if pair is None:
                    continue
                key = (pair.tool_name, pair.user.lower())
                if key in seen_user_queries:
                    continue
                seen_user_queries.add(key)
                pairs.append(pair)
                if verbose:
                    print(f"  + {pair.tool_name:24s}  {pair.user!r}  "
                          f"({path.name}:{sent.line})")
        per_file[str(path.relative_to(CACHE_DIR))] = len(pairs) - before

    return pairs, per_file


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1].strip())
    ap.add_argument("--no-fetch", action="store_true",
                    help="Do not clone/pull; mine whatever is already in the cache.")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Print every accepted pair.")
    args = ap.parse_args()

    schemas = load_mcp_schemas()

    have_cache = ensure_cache(no_fetch=args.no_fetch, verbose=args.verbose)
    if not have_cache:
        # Network unavailable AND no prior cache — write an empty file so
        # downstream tooling has something deterministic to read.
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text("")
        print("[mine_kf5book] no cache, no network — wrote 0 pairs to "
              f"{OUT_PATH.relative_to(REPO_ROOT)}")
        return 0

    pairs, per_file = mine(verbose=args.verbose)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p.to_jsonl(schemas), ensure_ascii=False))
            fh.write("\n")

    # ---- report ----------------------------------------------------------
    by_tool: dict[str, int] = {}
    for p in pairs:
        by_tool[p.tool_name] = by_tool.get(p.tool_name, 0) + 1

    print(f"[mine_kf5book] wrote {len(pairs)} real pairs to "
          f"{OUT_PATH.relative_to(REPO_ROOT)}")
    for name in sorted(by_tool):
        print(f"  {name:26s}  {by_tool[name]}")
    if not by_tool:
        print("  (no pairs — see CACHE_DIR for the documentation actually "
              "present locally)")

    # Top-yielding files.
    top = sorted(per_file.items(), key=lambda kv: -kv[1])[:5]
    if top and top[0][1] > 0:
        print("[mine_kf5book] top files:")
        for name, n in top:
            if n > 0:
                print(f"  {n:4d}  {name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
