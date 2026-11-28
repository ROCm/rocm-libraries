# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A PR to draft, offline: a git repository whose develop holds BASE and whose PR head holds HEAD (wt7, and wt7m as the
merged tree), the staged PR data in workers/pr7/pr/, a stub rule scanner, a key file with a planted key, and a
configuration whose drafter binding points at the closed port 127.0.0.1:9. `follow_up` turns it into a follow-up: an
earlier head our last post (pr/ours.md) reviewed, a push since that changes y.py only, a saved probe of our N4, a
review and an author's reply after our post. `FakeEndpoint` answers chat requests from a script per pass and per prior
task, as an OpenAI-compatible endpoint would (it is the client's `post`: nothing reaches a socket); it is thread-safe,
since a draft's phases ask at once."""

import json
import pathlib
import re
import subprocess
import threading
import time

import home_repository

PROJECT = "proj"
# The repository profile the drafter reads: the home repository's, the profile_dir the running configuration gives it
PROFILE = home_repository.FOUND.path.parent
AT = "@"  # the logins below are written after it, so the source holds no @handle
TAG = "PROJ"  # the tracker's key, from the configuration (tracker_keys)
BASE = "def answer():\n    return 42\n\n\ndef helper(v):\n    return v\n"
HEAD = "def answer():\n    return 41\n\n\ndef helper(v):\n    return v + 1\n"
Y_THEN = "def flags(x):\n    out = dict(x)\n    out.pop('fast')\n    return out\n"
Y_NOW = "def flags(x):\n    out = dict(x)\n    out.setdefault('fast', False)\n    return out\n"
Z = "NOTES = 'unchanged'\n"
KEY = "k-" + "planted-test-key-7731"  # built at run time: no real key is ever in a test
LINE1 = "Automated review: Model - Cand-1, effort: high"
META = f"""# PR #7: fix(proj): make the answer right
- Author: {AT}dev-a
- State: OPEN; mergeable: MERGEABLE
- Head: dev-a/fix @ HEADSHA; base: develop @ BASESHA
- Created 2026-11-01T10:00:00Z, updated 2026-11-02T10:00:00Z; +2/-2
- Labels: x
- Round: ROUND
- Line 1: {LINE1}

## Description
Fixes the answer. Tracks {TAG}-123.
"""
REVIEWS = f"""# Reviews from OTHER reviewers (oldest first; commit = the head the reviewer saw).
# Our own prior reviews are intentionally omitted — assess independently.

## {AT}dev-b APPROVED at abcdef12 (2026-11-01T11:00:00Z)
(no body)

## {AT}dev-c COMMENTED at abcdef12 (2026-11-01T12:00:00Z)
The helper should get a test.
"""
THREADS = f"""# Inline review threads (our own comments omitted)

## proj/x.py:5 [OPEN]
### {AT}copilot-pull-request-reviewer (2026-11-01T13:00:00Z)
helper(v) changed its result; is that intended?
"""
COMMENTS = f"""# Conversation comments from OTHERS (oldest first). Our own are omitted.

## {AT}therock-pr-bot (2026-11-01T10:05:00Z)
All checks passed.

## {AT}dev-a (2026-11-01T14:00:00Z)
Rebased on develop.
"""
CHECKS = (
    "conclusion\tcheck\turl (cancelled = superseded or stopped, not a failure)\n"
    "failure\tbuild (gfx942)\thttps://github.com/o/r/actions/runs/1/job/2\n"
    "success\tlint\thttps://github.com/o/r/actions/runs/1/job/3\n"
)
RULE_SCAN = (
    "print('== tracker: 0')\nprint('== perf: 0')\nprint('== env (non-test): 0')\n"
)
# The follow-up's last post of ours, at the earlier head (THEN), checked against develop BASE10.
OURS = """# Our last review post on PR #7
- Head: THEN
- Posted: 2026-11-02T09:00:00Z

## Review
Automated review: Model - Cand-1, effort: high

## PR #7 — make the answer right (dev-a) — 🟠 Follow-up: two open

| Prior item | Status |
|---|---|
| dev-b (11-01) APPROVED at `abcdef12` | 🟢 APPROVED `abcdef12`<br>• older head; still counts on develop |
| Ours (11-01) N1: `answer` returns 41 | 🟠 OPEN<br>• still at `x.py:2` (read) |
| Ours (11-01) N2: `flags` drops the fast flag | 🟠 OPEN<br>• `y.py:3` pops it (read) |
| Earlier items: ours N3; dev-c's nits | 🟢 FIXED, still hold<br>• delta reopens none (read) |
| Bots / CI | 🟢 Green<br>• 2 ✓ (read) |
| Our checks | 🟢 Clean<br>• merge vs `BASE10`: clean (ran) |

| # | Sev | file:line | Finding |
|---|---|---|---|
| N4 | 🟠 | `x.py:6` | **helper adds one**<br>• impact: callers get v + 1<br>• fix: return v |

**Utility:** 🟡 Mixed — two open
"""
FOLLOW_REVIEWS = (
    REVIEWS
    + f"""
## {AT}dev-d COMMENTED at NOW8 (2026-11-02T12:00:00Z)
Please keep `z.py` as it is.
"""
)
FOLLOW_COMMENTS = (
    COMMENTS
    + f"""
## {AT}dev-a (2026-11-02T11:00:00Z)
N2 is fixed in the last push.
"""
)
PROBE = """#!/bin/bash
# N4: helper still adds one. Exit 1 while it reproduces, 0 once fixed, 2 when the tree cannot show it.
f=$1/proj/x.py
[ -f "$f" ] || { echo "no x.py: cannot show"; exit 2; }
grep -q 'return v + 1' "$f" && { echo "helper adds one"; exit 1; }
echo "helper returns v"; exit 0
"""


def git(cwd, *args) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def make(root: pathlib.Path, endpoint_dir: pathlib.Path | None = None) -> dict:
    """Build the first-round PR under `root`; the configuration (a dict) and the head's sha."""
    wt = root / "wt7"
    (wt / PROJECT).mkdir(parents=True)
    git(root, "init", "-q", "-b", "develop", str(wt))
    for k, v in (
        ("user.name", "t"),
        ("user.email", "t@example.com"),
        ("commit.gpgsign", "false"),
    ):
        git(wt, "config", k, v)
    (wt / PROJECT / "x.py").write_text(BASE, encoding="utf-8")
    (wt / PROJECT / "y.py").write_text(Y_THEN, encoding="utf-8")
    (wt / PROJECT / "z.py").write_text(Z, encoding="utf-8")
    git(wt, "add", "-A")
    git(wt, "commit", "-q", "-m", "base")
    git(wt, "update-ref", "refs/remotes/origin/develop", "HEAD")
    git(wt, "checkout", "-q", "-b", "pr")
    (wt / PROJECT / "x.py").write_text(HEAD, encoding="utf-8")
    git(wt, "commit", "-q", "-am", "fix(proj): make the answer right")
    git(root, "clone", "-q", str(wt), str(root / "wt7m"))
    head, base = git(wt, "rev-parse", "HEAD"), git(wt, "rev-parse", "develop")
    ws = root / "workers" / "pr7"
    stage(wt, ws, head, base, "First-round", REVIEWS, COMMENTS)
    (root / "tools").mkdir()
    (root / "tools" / "rule_scan.py").write_text(RULE_SCAN, encoding="utf-8")
    secrets = (
        root / "secrets"
    )  # a binding's key is where the secrets registry keeps the secret it names
    secrets.mkdir(mode=0o700)
    keyfile = secrets / "cand_api_key"
    keyfile.write_text(KEY + "\n", encoding="utf-8")
    keyfile.chmod(0o600)
    cfg = {
        "repo": "o/r",
        "rocke_path": PROJECT,
        "review_first_line": LINE1,
        "worker_backend": "cand",
        "tracker_keys": [TAG],
        "model_bindings": {
            "cand": {
                "provider": "openai",
                "url": "http://127.0.0.1:9",
                "model": "Cand-1",
                "api_key_secret": keyfile.name,
                "auth_header": "Ocp-Apim-Subscription-Key",
                "auth_scheme": "",
                "auth_mode": "both",
                "tpm": 100000,
                "rpm": 100,
                "effort": "high",
            }
        },
        "drafter": {"limiter_dir": str(root / "limits"), "resend_wait_s": 0},
        "paths": {
            "root": str(root),
            "repo": str(wt),
            "workers": str(root / "workers"),
            "secrets_dir": str(secrets),
        },
        "verifier_name": "the Reference Model 1.0",
        "profile_dir": str(PROFILE),
    }
    return {"cfg": cfg, "head": head, "base": base, "ws": ws, "root": root}


def stage(
    wt: pathlib.Path,
    ws: pathlib.Path,
    head: str,
    base: str,
    round_word: str,
    reviews: str,
    comments: str,
) -> None:
    """The staged PR data of the PR at `head` (its diff from develop `base`)."""
    pr = ws / "pr"
    pr.mkdir(parents=True, exist_ok=True)
    (pr / "meta.md").write_text(
        META.replace("HEADSHA", head[:10])
        .replace("BASESHA", base[:10])
        .replace("ROUND", round_word),
        encoding="utf-8",
    )
    (pr / "diff.patch").write_text(git(wt, "diff", base, head) + "\n", encoding="utf-8")
    (pr / "files.txt").write_text(
        "".join(
            f"     2 +     2 -  {p}\n"
            for p in git(wt, "diff", "--name-only", base, head).split()
        ),
        encoding="utf-8",
    )
    (pr / "reviews.md").write_text(reviews, encoding="utf-8")
    (pr / "threads.md").write_text(THREADS, encoding="utf-8")
    (pr / "comments.md").write_text(comments, encoding="utf-8")
    rows = []
    for line in git(
        wt, "log", "--reverse", "--format=%H %ct %s", f"{base}..{head}"
    ).splitlines():
        sha, ct, subject = line.split(" ", 2)
        rows.append(
            f"- {sha[:8]} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(int(ct)))} {subject}\n"
        )
    (pr / "commits.md").write_text(
        "# Commits (oldest first)\n" + "".join(rows), encoding="utf-8"
    )
    (pr / "checks.txt").write_text(CHECKS, encoding="utf-8")
    (ws / "mkwt.log").write_text(
        f"head={head[:10]} develop={base[:10]} merge-base={base[:10]}\nbehind/ahead: 0\t1\n"
        "conflicting files vs develop:\n\n",
        encoding="utf-8",
    )


def follow_up(f: dict) -> dict:
    """The follow-up: our last post at the PR's earlier head (THEN), a push since that fixes y.py, a saved probe of our
    N4, dev-d's review after our post, and the author's reply naming N2. The new head is the trees' and the staging's.
    """
    root, wt = f["root"], f["root"] / "wt7"
    then = f["head"]
    (wt / PROJECT / "y.py").write_text(Y_NOW, encoding="utf-8")
    git(wt, "commit", "-q", "-am", "fix(proj): keep the fast flag")
    now = git(wt, "rev-parse", "HEAD")
    git(root / "wt7m", "pull", "-q", "--ff-only", str(wt), "pr")
    stage(
        wt,
        f["ws"],
        now,
        f["base"],
        "Follow-up",
        FOLLOW_REVIEWS.replace("NOW8", now[:8]),
        FOLLOW_COMMENTS,
    )
    (f["ws"] / "pr" / "ours.md").write_text(
        OURS.replace("THEN", then).replace("BASE10", f["base"][:10]), encoding="utf-8"
    )
    probes = root / "probes" / "pr7"
    probes.mkdir(parents=True)
    (probes / "N4_helper.sh").write_text(PROBE, encoding="utf-8")
    return dict(f, then=then, head=now)


def call(name: str, **args) -> dict:
    return {"call": name, "args": args}


class FakeEndpoint:
    """The client's `post`: each request is told apart by its task card (the hunt, or one prior item's task, keyed
    "prior:I<k>") and answers from that script, one entry per request, with a usage block (cached tokens included); it
    keeps every request body. "$LAST_RUN" in an entry's arguments is the last run id the conversation shows. `before`
    (a key, or "prior" for every task, -> a function) runs before such a request is answered.
    """

    def __init__(self, script: dict, cached: int = 100, before: dict | None = None):
        self.script = {k: list(v) for k, v in script.items()}
        self.bodies: list[dict] = []
        self.keys: list[str] = []
        self.cached = cached
        self.before = before or {}
        self.lock = threading.Lock()

    @staticmethod
    def name_of(body: dict) -> str:
        user = body["messages"][1]["content"]
        if user.startswith("TASK: one prior item"):
            m = re.search(r"^Item (I\d+):", user, re.M)
            return f"prior:{m.group(1) if m else '?'}"
        return "hunt"

    def __call__(self, s: dict, body: dict, key: str) -> dict:
        name = self.name_of(body)
        hook = self.before.get(name) or self.before.get(name.split(":")[0])
        if hook:
            hook()
        with self.lock:
            self.bodies.append(json.loads(json.dumps(body)))
            self.keys.append(key)
            steps = self.script.get(name) or []
            entry = steps.pop(0) if steps else {"text": "nothing more"}
            seq = len(self.bodies)
        last = ""
        for msg in reversed(body["messages"]):
            m = (
                re.match(r"\[(R\d+)\]", str(msg.get("content") or ""))
                if msg.get("role") == "tool"
                else None
            )
            if m:
                last = m.group(1)
                break
        entry = json.loads(json.dumps(entry).replace("$LAST_RUN", last))
        prompt = sum(len(json.dumps(m)) for m in body["messages"]) // 4
        usage = {
            "prompt_tokens": prompt,
            "completion_tokens": 120,
            "total_tokens": prompt + 120,
            "prompt_tokens_details": {"cached_tokens": self.cached},
            "completion_tokens_details": {"reasoning_tokens": 80},
        }
        calls = entry if isinstance(entry, list) else [entry] if "call" in entry else []
        return {
            "role": "assistant",
            "content": entry.get("text", "") if isinstance(entry, dict) else "",
            "reasoning": "thinking",
            "finish_reason": "tool_calls" if calls else "stop",
            "usage": usage,
            "tool_calls": [
                {
                    "id": f"c{seq}_{i}",
                    "function": {"name": c["call"], "arguments": c["args"]},
                }
                for i, c in enumerate(calls)
            ],
        }
