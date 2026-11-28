# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""An armed auto-merge vs the PR's live title and body.

GitHub stores the squash commit's title and message when auto-merge is armed, and the merge commits those, not the
title and body at merge time (lessons L-136: a PR renamed and re-described after it was armed lands the old text in
develop's history). Feeds no gate: a stale armed message
is a 🟡 for the verifier.

Usage: automerge_msg.py REPO N [--json FILE]   (FILE: the pull's JSON, instead of `gh api repos/REPO/pulls/N`)
"""

import json
import os
import re
import subprocess
import sys

# gh under the read login, reads only (tools/ghr.py).
GHR = [
    sys.executable,
    "-I",
    "-B",
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "ghr.py"),
]

# GitHub ends a squash message with the commits' co-authors, under a dashed rule when it has several commits; the body
# never carries it, so a message that is the live body plus this trailer (with or without the rule) is current.
SQUASH_TRAILER = re.compile(
    r"\n(?:-{3,}[ \t]*\n)?\s*(?:Co-authored-by:[^\n]*(?:\n\s*|\Z))+\Z", re.I
)


def _words(text: str | None) -> list[str]:
    # Words, not lines: a stored message may be re-wrapped or carry CRLFs while saying the same thing.
    return SQUASH_TRAILER.sub("", (text or "").replace("\r\n", "\n")).split()


def _cut(text: str, width: int = 80) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def compare(pr: dict, n: int) -> list[str]:
    am = pr.get("auto_merge")
    if not am:
        return ["auto-merge: off"]
    who = (am.get("enabled_by") or {}).get("login") or "?"
    out = [f"auto-merge: on ({am.get('merge_method') or '?'}), armed by {who}"]
    title = am.get("commit_title")
    live_title = f"{pr.get('title') or ''} (#{n})"
    if title is None:
        out.append("armed title: none stored (the merge takes the live title)")
    elif title.strip() != live_title.strip():
        out.append(
            f"STALE title: armed '{_cut(title)}', live '{_cut(live_title)}': the merge commits the armed one; "
            "re-arm (turn auto-merge off and on) to take the live title"
        )
    else:
        out.append("armed title = live title")
    msg = am.get("commit_message")
    if msg is None:
        out.append("armed message: none stored (the merge takes the live body)")
        return out
    armed, live = _words(msg), _words(pr.get("body"))
    if armed == live:
        out.append("armed message = live body")
        return out
    k = next(
        (i for i, (a, b) in enumerate(zip(armed, live)) if a != b),
        min(len(armed), len(live)),
    )
    a = " ".join(armed[k : k + 10]) or "(end)"
    b = " ".join(live[k : k + 10]) or "(end)"
    out.append(
        f"STALE message: the armed message differs from the live body from word {k + 1} "
        f"(armed '{_cut(a)}', live '{_cut(b)}'); re-arm to commit the live body"
    )
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__.strip().splitlines()[-1])
        return 2
    repo, n = argv[0], int(argv[1])
    if len(argv) >= 4 and argv[2] == "--json":
        with open(argv[3], encoding="utf-8") as f:
            pr = json.load(f)
    else:
        p = subprocess.run(
            [*GHR, "api", f"repos/{repo}/pulls/{n}"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if p.returncode:
            print(f"auto-merge: unknown (gh: {p.stderr.strip()[:200]})")
            return 1
        pr = json.loads(p.stdout)
    print("\n".join(compare(pr, n)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
