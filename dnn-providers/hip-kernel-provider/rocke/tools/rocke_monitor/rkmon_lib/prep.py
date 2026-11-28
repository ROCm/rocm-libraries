# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`rk review-prep N`: a verification's inputs in one review directory, and the summary the orchestrator reads before it
writes the verifier's and the code quality reviewer's briefs.

The directory, by default `<tmp>/v<N><round>` (paths.tmp; round `fr` for a first round, `fu<k>` for the k-th
follow-up), gets:
- facts.txt: what `rk facts N` prints, to paste verbatim into the briefs (lessons L-201);
- classify.txt: what `rk classify N` prints ("never reviewed" before our first post);
- overlap_section.md: the checks' section "Overlapping work in other open PRs", for the code quality pass;
- quality.md: the code quality pass's, when the quality binding ran it at the PR's head (config roles.quality,
  review.quality_start): the verifier folds it in, and no code quality reviewer is dispatched;
- ci_triage.md and quality_leads.md: the first passes' leads for the head (models.ci_triage, models.quality_draft:
  rkmon_lib/first_pass.py), for the verifier's and the code quality reviewer's briefs; each lead is checked before use;
- the worktree paths: the verifier's `<dir>/wt_v<name>` and the code quality reviewer's `<dir>/wt_q<name>` (lessons
  L-591), the only places the session's guard lets an agent add a worktree, each made a tree PR code reads with `rk
  pr-tree` in the command that adds it (rkmon_lib/workguard.py);
- files.txt: the files the PR changes from its merge base with its base branch (develop, or the feature branch it
  targets: `base.ref`) to its head; `rk post --premises` checks that the base has not changed them since the verified
  base.
The summary, as the script printed it: our last three posts (ledger lines), the review directories and probes, the last
archived notes and our review files in the outputs folder (paths.posted_dir: the posts' copies and the reviews kept
unposted); the facts' head lines, our last review, the ABI lane and the reviews; the classification; the checks' merge
and re-probe sections and the overlap leads; the head, merge base and commit count, and the number of files.

Reads only (GitHub GETs, the state, the ledger) and writes only the review directory. The PR's head is fetched into the
mirror under its own ref, never read from FETCH_HEAD, which other jobs move (lessons L-164); develop is the mirror's, as
the last tick fetched it; a base branch other than develop is fetched first.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import shutil

from . import config, facts, first_pass, gitops, paths, post, review
from .queue import DEVELOP, base_of
from .snapshot import same_commit

FILES = "files.txt"
ROUND = re.compile(r"fr\d*|fu\d+")
# A review directory's name, as review_dir makes it: v, the PR's number, the round. Whatever reads the name, here or in
# another module, reads it by this rule.
REVIEW_DIR = re.compile(rf"v(\d+)({ROUND.pattern})")


def default_round(st: dict, n: int, cfg: dict | None = None) -> str:
    """`fr` before our first review of PR n, else `fu<k>`: k is the number of our reviews of it so far (posted, or kept
    unposted while posting is off: post.our_review counts both), raised past every follow-up round already used, a
    post's copy `pr<N>_fu<k>.md` or a review directory with notes (an unposted round, or a first follow-up named `fu`,
    must never make the count land on a round already taken).
    """
    ours = (st["prs"].get(str(n)) or {}).get("ours")
    if not ours:
        return "fr"
    k = ours.get("count") or 1
    if cfg:
        from . import premises  # a call-time import: premises may import this module

        used = re.compile(rf"(?:pr{n}_fu|v{n}fu)(\d+)(?:\.md)?")
        names = [p.name for p in premises.posts_dir(cfg).glob(f"pr{n}_fu*.md")]
        names += [p.name for p in review_dirs(cfg, n) if (p / "notes.md").exists()]
        k = max([k] + [int(m.group(1)) + 1 for m in map(used.fullmatch, names) if m])
    return f"fu{k}"


def review_dir(cfg: dict, n: int, rnd: str) -> pathlib.Path:
    return pathlib.Path(paths.get(cfg, "tmp")) / f"v{n}{rnd}"


def review_dir_pr(name: str) -> int | None:
    """The PR whose review directory a folder named NAME is (REVIEW_DIR), else None."""
    m = REVIEW_DIR.fullmatch(name)
    return int(m.group(1)) if m else None


def review_dirs(cfg: dict, n: int) -> list[pathlib.Path]:
    """Every review directory of PR n, each round's (review_dir_pr), in name order; none while paths.tmp is not
    set."""
    try:
        tmp = pathlib.Path(paths.get(cfg, "tmp"))
    except paths.PathsError:
        return []
    return sorted(p for p in tmp.glob(f"v{n}*") if review_dir_pr(p.name) == n)


def worktree_line(out: pathlib.Path) -> str:
    """The verifier's and the code quality reviewer's worktree paths in the review directory `out`: distinct prefixes,
    so the two agents never share a worktree (lessons L-591), and how each is added: made a tree PR code reads (`rk
    pr-tree`) in the same command. The session's guard refuses a worktree added elsewhere or not made so.
    """
    return (
        f"worktrees: verifier {out}/wt_v<name> · code quality reviewer {out}/wt_q<name>, each added as "
        "`rk pr-tree PATH && git -C MIRROR worktree add --detach PATH COMMIT` "
        "(a worktree added anywhere else, or not made so, is refused)"
    )


def text_lines(text: str) -> list[str]:
    """A text's lines as sed reads them: a final newline ends the last line and starts none."""
    lines = text.split("\n")
    return lines[:-1] if text.endswith("\n") else lines


def sed_ranges(lines: list[str], *ranges: tuple[str, str]) -> list[str]:
    """`sed -n '/a/,/b/p;/c/,/d/p'`: each run of lines from one matching a range's start through the next matching its
    end (to the last line when none does), every range at once, so a line in two ranges comes twice.
    """
    pats = [(re.compile(start), re.compile(end)) for start, end in ranges]
    inside = [False] * len(pats)
    out = []
    for line in lines:
        for i, (start, end) in enumerate(pats):
            if inside[i]:
                out.append(line)
                inside[i] = not end.search(line)
            elif start.search(line):
                out.append(line)
                inside[i] = True
    return out


def first_seen(lines: list[str]) -> list[str]:
    """`awk '!seen[$0]++'`: each line once, where it first comes."""
    seen: set[str] = set()
    return [x for x in lines if not (x in seen or seen.add(x))]


def cut(lines: list[str], width: int, most: int | None = None) -> list[str]:
    """`cut -c1-<width> | head -<most>`."""
    return [x[:width] for x in lines][:most]


def _our_posts(cfg: dict, st: dict, store, n: int) -> list[str]:
    """Our review posts on PR n, as the ledger has their lines (in a mirror, the live monitor's ledger, up to the
    feed line: mirror.live_posts)."""
    from . import mirror

    live = mirror.live_posts(cfg, st, n)
    if live is not None:
        return [json.dumps(r) for r in live if r.get("kind") == "post"]
    if not store.ledger_path.exists():
        return []
    out = []
    for line in store.ledger_path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("pr") == n and r.get("kind") == "post":
            out.append(line)
    return out


def _listing(cfg: dict, n: int) -> list[str]:
    """The review directories (review_dirs) and probes (paths), the last three archived notes and our review files in
    the outputs folder (names): the posts' copies and the reviews kept unposted."""
    k = str(n)
    probes = pathlib.Path(paths.get(cfg, "probes")) / f"pr{n}"
    out = sorted(
        str(p) for p in review_dirs(cfg, n) + ([probes] if probes.exists() else [])
    )
    notes, copies = pathlib.Path(paths.get(cfg, "lessons")) / "notes", post.posted_dir(
        cfg
    )
    out += (
        sorted(x for x in os.listdir(notes) if f"_pr{k}_" in x)[-3:]
        if notes.is_dir()
        else []
    )
    out += (
        sorted(x for x in os.listdir(copies) if re.search(rf"_{k}(?!\d)", x))
        if copies.is_dir()
        else []
    )
    return out


def quality_copy(cfg: dict, st: dict, n: int, out: pathlib.Path) -> list[str]:
    """The quality binding's quality.md (review.quality_folder), copied into the review directory when it is for the
    PR's head; the summary line that says what happened ([] when the lane did not run, and for a review that gets no
    such pass by design: review.quality_runs_for).
    """
    folder = review.quality_folder(cfg, n)
    if not review.quality_runs_for(cfg, st, n) or not (folder / "quality.md").is_file():
        return []
    try:
        run = json.loads((folder / "quality_run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        run = {}
    head = (st["prs"].get(str(n)) or {}).get("head") or ""
    if not same_commit(str(run.get("head") or ""), head):
        return [
            f"quality.md: the code quality pass's is for {str(run.get('head') or 'no head')[:10]}, not the "
            f"head {head[:10]}: not copied; dispatch the code quality reviewer as usual"
        ]
    shutil.copyfile(folder / "quality.md", out / "quality.md")
    return [
        f"quality.md: the code quality pass ran on the quality binding {run.get('binding')} at {head[:10]} (copied "
        "here): the verifier folds it in; don't dispatch the code quality reviewer"
    ]


def prepare(
    cfg: dict, st: dict, store, n: int, out: pathlib.Path
) -> tuple[list[str], bool]:
    """Write PR n's review inputs into `out`; the summary lines, and whether every input was read."""
    out.mkdir(parents=True, exist_ok=True)
    lines, ok = [f"review dir {out}"], True
    rendered, failed = facts.render(cfg, st, store, n)
    fact_lines = text_lines(
        "\n".join(rendered) + "\n"
    )  # as the file has them: a comment body spans lines
    (out / "facts.txt").write_text("\n".join(fact_lines) + "\n", encoding="utf-8")
    lines.append(f"facts exit {1 if failed else 0}")
    rec = st["prs"].get(str(n))
    try:
        c = review.classify_pr(cfg, rec) if rec else None
        said = json.dumps(c, indent=1) if c else f"#{n} is not tracked"
    except gitops.GitError as exc:
        c, said = None, f"classify failed: {exc}"
    (out / "classify.txt").write_text(said + "\n", encoding="utf-8")
    lines.append(f"classify exit {0 if c else 1}")
    lines.append(worktree_line(out))
    ok = not failed and c is not None
    lines += cut(_our_posts(cfg, st, store, n)[-3:], 140)
    lines += _listing(cfg, n)
    lines += cut(fact_lines[:14], 220)
    lines += cut(sed_ranges(fact_lines, ("== our last review", "== rk abi")), 280)
    lines += cut(sed_ranges(fact_lines, ("== rk abi", "== rk breakages")), 200, 6)
    lines += cut(
        sed_ranges(
            fact_lines, ("== reviews", "== " + re.escape(config.operator_login(cfg)))
        ),
        240,
        24,
    )
    if c and "old" in c:
        lines.append(
            f"{c['kind']} {c['old'][:10]} -> {c['new'][:10]} {c.get('n_changed', len(c['changed_files']))}"
        )
        lines += c["new_commits"][:12]
    elif c:
        lines.append(c["kind"])
    checks = pathlib.Path(paths.get(cfg, "workers")) / f"pr{n}" / "checks" / "checks.md"
    try:
        check_lines = text_lines(checks.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        check_lines = []
        lines.append(f"(no checks yet: {checks})")
    if check_lines:
        lines.append(check_lines[0])
        lines += cut(
            first_seen(
                sed_ranges(
                    check_lines,
                    (r"^## Merge with develop", r"^## Native build"),
                    (r"^## Our open findings", r"^## Threads"),
                )
            ),
            240,
            40,
        )
        lines += cut(
            first_seen(
                sed_ranges(check_lines, (r"^## Overlapping work", r"^## rule_scan"))
            ),
            240,
            16,
        )
    section = sed_ranges(check_lines, (r"^## Overlapping work", r"^## ABI"))
    if section and section[-1].startswith("## ABI"):
        section.pop()
    (out / "overlap_section.md").write_text(
        "".join(x + "\n" for x in section), encoding="utf-8"
    )
    lines += quality_copy(cfg, st, n, out)
    lines += first_pass.copy(
        cfg, store, rec, n, out, running=review.first_pass_running(st, store, n)
    )
    repo = paths.get(cfg, "repo")
    try:
        head = gitops.fetch_pr(repo, n)
        ref = base_of(
            cfg, rec, fetch=True
        )  # the PR's base branch: develop, or the feature branch it targets
        base = gitops.git(repo, "merge-base", head, ref).strip()
        count = gitops.git(repo, "rev-list", "--count", f"{base}..{head}").strip()
        files = gitops.git(repo, "diff", "--name-only", base, head).splitlines()
    except gitops.GitError as exc:
        return lines + [f"(head, merge base and files not read: {exc})"], False
    (out / FILES).write_text("".join(f + "\n" for f in files), encoding="utf-8")
    on = "" if ref == DEVELOP else f" (base {ref.removeprefix('refs/remotes/origin/')})"
    return (
        lines
        + [f"head {head[:10]} mb {base[:10]} commits {count}{on}", str(len(files))],
        ok,
    )
