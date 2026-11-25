# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Since our last review: what a follow-up's push really changed, base-aware, so a force-pushed re-cut reads as
rebase-only or not at a glance (the verifier and the lane ran range-diff and per-file delta by hand in 33 of 39 reviews;
agent_script_overlap.md, fix 5).

1. The commits: `git range-diff merge-base(develop, old)..old merge-base(develop, new)..new`, each of our reviewed
   head's commits and the new head's paired: same patch (=), changed (!), dropped (<, only at our head) or new (>, only
   now). Merge commits are not in a range-diff: their count at each head is given.
2. The PR's own files: each file the PR changes at either head, compared as the PR's patch of it from its own merge
   base (the multiset of its added and removed lines, as rkmon_lib/review.py's _patch_fingerprint does: develop
   moving under it is not a change),
   with its +/- line counts then and now and how many patch lines differ; a file the PR changes at one head only says
   which ("not changed by the PR at the new head", "newly changed by the PR").

Git only, on the mirror (a partial clone: missing blobs are fetched from origin; no ref is written). Prints the
section, then one last line: `SINCE pass same patches: ...` (a rebase or merge only), `SINCE pass changed: ...`, or
`SINCE skip not run: ...` with the reason.

Usage: python -I since_review.py --repo R --old OUR_HEAD --new HEAD [--rocke-path K] [--base REF]
"""
import argparse
import collections
import os
import re
import subprocess
import sys

DEVELOP = "refs/remotes/origin/develop"
SHOWN = 25  # commits and files listed, at most each
PAIR = re.compile(
    r"^\s*(\d+|-+):\s+([0-9a-f]+|-+) ([=!<>]) (\d+|-+):\s+([0-9a-f]+|-+) ?(.*)$"
)
WORD = {"=": "same patch", "!": "changed", "<": "dropped", ">": "new"}


def git(repo: str, *args: str, check: bool = True) -> str:
    p = subprocess.run(
        ["git", "-C", repo, *args], capture_output=True, text=True, timeout=300
    )
    if check and p.returncode:
        raise RuntimeError(f"git {' '.join(args[:2])}: {p.stderr.strip()[:200]}")
    return p.stdout


def pairs(text: str) -> list[tuple[str, str, str, str]]:
    """(mark, our head's commit or "", the new head's or "", subject) per range-diff line."""
    out = []
    for ln in text.splitlines():
        m = PAIR.match(ln)
        if m:
            old = "" if set(m.group(2)) == {"-"} else m.group(2)
            new = "" if set(m.group(5)) == {"-"} else m.group(5)
            out.append((m.group(3), old, new, m.group(6).strip()))
    return out


def numstat(repo: str, mb: str, head: str, path: str) -> tuple[int, int]:
    line = git(
        repo, "diff", "--numstat", "--no-renames", mb, head, "--", path, check=False
    ).strip()
    parts = line.split("\t") if line else []
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return 0, 0  # a binary file, or none


def patch_lines(repo: str, mb: str, head: str, path: str) -> collections.Counter:
    diff = git(repo, "diff", "-U0", "--no-renames", mb, head, "--", path, check=False)
    return collections.Counter(
        ln
        for ln in diff.splitlines()
        if ln[:1] in "+-" and not ln.startswith(("+++", "---"))
    )


def since(
    repo: str, old: str, new: str, K: str = "", base: str = DEVELOP
) -> tuple[list[str], str, str]:
    """The section's lines, the status word and its detail."""
    for sha, what in ((old, "our reviewed head"), (new, "the head")):
        if subprocess.run(
            ["git", "-C", repo, "cat-file", "-e", f"{sha}^{{commit}}"],
            capture_output=True,
            env={**os.environ, "GIT_NO_LAZY_FETCH": "1"},
        ).returncode:
            return (
                [f"NOT RUN: {what} {sha[:10]} is not in the mirror"],
                "skip",
                f"not run: {what} not in the mirror",
            )
    mb_old, mb_new = (
        git(repo, "merge-base", base, old).strip(),
        git(repo, "merge-base", base, new).strip(),
    )
    lines = [
        f"our reviewed head {old[:10]} (on develop {mb_old[:10]}) -> head {new[:10]} (on develop {mb_new[:10]})"
        + ("; the same merge base" if mb_old == mb_new else "")
    ]
    merges = [
        len(git(repo, "rev-list", "--merges", f"{mb}..{h}").split())
        for mb, h in ((mb_old, old), (mb_new, new))
    ]
    rd = subprocess.run(
        [
            "git",
            "-C",
            repo,
            "range-diff",
            "--no-color",
            "--no-notes",
            f"{mb_old}..{old}",
            f"{mb_new}..{new}",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    ps = pairs(rd.stdout) if rd.returncode == 0 else []
    count = collections.Counter(p[0] for p in ps)
    if rd.returncode:
        why = (rd.stderr.strip().splitlines() or ["no output"])[-1]
        lines.append(f"commits: range-diff NOT RUN ({why[:160]})")
    else:
        lines.append(
            f"commits (range-diff, merges aside): {len(ps) or 'none'}"
            + "".join(
                f"{', ' if i else ': '}{count[k]} {WORD[k]}"
                for i, k in enumerate(k for k in "=!<>" if count[k])
            )
            + f"; merge commits: {merges[0]} then, {merges[1]} now"
        )
        for mark, o, n, subj in [p for p in ps if p[0] != "="][:SHOWN]:
            lines.append(f"   {WORD[mark]}: {o or '-'} -> {n or '-'} {subj[:100]}")
    files = sorted(
        set(git(repo, "diff", "--name-only", "--no-renames", mb_old, old).split())
        | set(git(repo, "diff", "--name-only", "--no-renames", mb_new, new).split())
    )
    changed = []
    for f in files:
        a, b = patch_lines(repo, mb_old, old, f), patch_lines(repo, mb_new, new, f)
        if a == b:  # the same patch, as review._patch_fingerprint compares them
            continue
        differ = sum(((a - b) + (b - a)).values())
        (ao, do), (an, dn) = numstat(repo, mb_old, old, f), numstat(
            repo, mb_new, new, f
        )
        rel = f[len(K) + 1 :] if K and f.startswith(K + "/") else f
        state = (
            "not changed by the PR at the new head"
            if not b
            else "newly changed by the PR" if not a else f"{differ} patch lines differ"
        )
        changed.append(f"   {rel}: +{ao} -{do} then, +{an} -{dn} now ({state})")
    lines.append(
        f"the PR's own files: {len(files)}, {len(changed)} with a different patch"
        + (":" if changed else " (develop's changes under them aside)")
    )
    lines += changed[:SHOWN] + (
        [f"   +{len(changed) - SHOWN} more"] if len(changed) > SHOWN else []
    )
    same = (
        not rd.returncode
        and not (count["!"] or count["<"] or count[">"])
        and not changed
    )
    if same:
        detail = (
            f"same patches: {count['=']} commit{'s' if count['='] != 1 else ''} the same, none changed, new or "
            f"dropped, and no PR file's patch changed (a "
            f"{'merge' if mb_old != mb_new and merges[1] > merges[0] else 'rebase'} only)"
        )
    else:
        detail = (
            "changed: "
            + ", ".join(
                f"{count[k]} commit{'s' if count[k] != 1 else ''} {WORD[k]}"
                for k in "!<>"
                if count[k]
            )
            + ("; " if any(count[k] for k in "!<>") else "")
            + f"{len(changed)} PR file"
            + f"{'s' if len(changed) != 1 else ''} with a different patch"
        )
        if changed:
            detail += (
                ": "
                + ", ".join(
                    c.strip().split(":")[0].rsplit("/", 1)[-1] for c in changed[:4]
                )
                + (f" +{len(changed) - 4} more" if len(changed) > 4 else "")
            )
    return lines, "pass", detail


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--old", required=True, help="our last reviewed head")
    ap.add_argument("--new", required=True, help="the head under review")
    ap.add_argument("--rocke-path", default="")
    ap.add_argument("--base", default=DEVELOP)
    a = ap.parse_args(argv)
    try:
        lines, word, detail = since(a.repo, a.old, a.new, a.rocke_path, a.base)
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        lines, word, detail = [f"NOT RUN: {exc}"], "skip", f"not run: {str(exc)[:160]}"
    print("\n".join(lines + [f"SINCE {word} {detail}"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
