#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GET-only: dump a PR's review threads, issue comments and review bodies since a date.

Usage: prthreads.py <N> <since ISO> [body chars]
The repository and the scratch folder come from the job's RKMON_* environment (`rk env` when run by hand). The three
lists are kept in <scratch>/thr<N>/ (rc.json, ic.json, rv.json). When a list is not read (gh's error on stderr) or
cannot be kept there, a line names each such list and says which failed, nothing more is printed, and the exit is 1, so
the checks run it once more when gh failed on the network (rkmon_lib/review.py, NET_FN); else 0.
"""

import json
import os
import subprocess
import sys

# gh under the read login, reads only (tools/ghr.py).
GHR = [
    sys.executable,
    "-I",
    "-B",
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "ghr.py"),
]


def load_rk_env() -> None:
    """The RKMON_* settings from the toolkit's `rk env` when the job gave none (exported to every child)."""
    rk = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), "..", "rk")
    try:
        p = subprocess.run([rk, "env", "--json"], stdout=subprocess.PIPE)
        vals = json.loads(p.stdout or b"{}")
    except (OSError, ValueError):
        return
    if isinstance(vals, dict):
        os.environ.update({str(k): str(v) for k, v in vals.items()})


def fetch(api: str, dest: str) -> str:
    """`gh api API --paginate > DEST` (gh's errors on stderr): "" when the list is read and kept, else what failed."""
    try:
        fh = open(dest, "wb")
    except OSError as exc:
        return f"not kept ({dest}: {exc.strerror})"
    with fh:
        try:
            p = subprocess.run([*GHR, "api", api, "--paginate"], stdout=fh)
        except OSError as exc:
            return f"not read (gh did not run: {exc.strerror})"
    return "" if p.returncode == 0 else "not read (GitHub API error)"


def dump(o: str, since: str, lim: int) -> None:
    with open(f"{o}/rc.json") as f:
        rc = json.load(f)
    with open(f"{o}/ic.json") as f:
        ic = json.load(f)
    with open(f"{o}/rv.json") as f:
        rv = json.load(f)
    print("== review comments (threads)")
    for c in rc:
        if c["created_at"] < since:
            continue
        b = c["body"].replace("\n", " ")
        print(
            f'{c["created_at"][:16]} id={c["id"]} {c["user"]["login"]} {c["path"].split("/")[-1]}:{c.get("line")} '
            f'reply_to={c.get("in_reply_to_id")} :: {b[:lim]}'
        )
    print("== issue comments")
    for c in ic:
        if c["created_at"] < since:
            continue
        b = c["body"].replace("\n", " ")
        print(f'{c["created_at"][:16]} {c["user"]["login"]} :: {b[:lim]}')
    print("== reviews with bodies")
    for r in rv:
        if (r.get("submitted_at") or "") < since or not r["body"]:
            continue
        print(
            f'{r["submitted_at"][:16]} {r["user"]["login"]} {r["state"]} :: {r["body"][:lim * 3]}'
        )


def main(argv: list[str]) -> int:
    if not os.environ.get("RKMON_REPO"):
        load_rk_env()
    r = f"repos/{os.environ.get('RKMON_REPO', '')}"
    n = argv[1] if len(argv) > 1 else ""
    since = argv[2] if len(argv) > 2 else ""
    lim = argv[3] if len(argv) > 3 and argv[3] else "600"
    o = f"{os.environ.get('RKMON_SCRATCH', '')}/thr{n}"
    try:
        os.makedirs(o, exist_ok=True)
    except OSError as exc:
        sys.stderr.write(f"mkdir: cannot create directory '{o}': {exc.strerror}\n")
    lists = {  # each list, by the name a line gives it: where gh reads it, and the file it is kept in
        "review comments (threads)": (f"{r}/pulls/{n}/comments", f"{o}/rc.json"),
        "issue comments": (f"{r}/issues/{n}/comments", f"{o}/ic.json"),
        "reviews": (f"{r}/pulls/{n}/reviews", f"{o}/rv.json"),
    }
    missed = {
        name: why for name, (api, dest) in lists.items() if (why := fetch(api, dest))
    }
    sys.stdout.flush()
    if missed:
        for name, why in missed.items():
            print(f"== {name}: {why}")
        return 1
    dump(o, since, int(lim))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
