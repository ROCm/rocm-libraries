# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""One call per orchestrator step: `rk dispatch-pack N`, `rk spec-post N` and the verifier's `rk wait-quality DIR`.

`rk dispatch-pack N [--round R]` runs `rk review-prep N` (its review directory and files) and prints, compactly, what
the orchestrator reads before it writes the briefs: the head and state, our last review's flagged rows and findings, the
unresolved threads and the operator's (config `operator.github_login`) comments, from the facts; the delta since our
last review (the classification, the PR's own changed files, its own non-merge commits since our reviewed head and the
merges); the checks' flags and the shared build; CI's FAIL and RUN lines (`rk ci-fails`); the overlap owner decisions
that name the PR; a micro round's items; and whether the round gets the code quality pass (`lane: needed|exempt
(<why>)`). Reads only (the review directory aside).

`rk spec-post N --round R` runs `rk spec`, and only when it made the spec with no CHECK line, `rk post` with the
premises (premises.post_checked). With `--devbase-recheck`, a post that stops only because develop changed some of the
PR's own files prints develop's diff stat for those files and stops: nothing is posted.

`rk wait-quality DIR` blocks until the code quality pass's `quality.md` in the review directory DIR is final (it exists,
is newer than `facts.txt`, and its Code quality row does not read `pending`): `ready`. Each call returns after at most
SLICE_MIN minutes (`waiting`, so a call stays under the Bash tool's 10-minute cap and the agent's prompt cache stays
warm), and the whole wait, counted from the first call, ends at `--max-min` (`timeout`).
"""

import json
import pathlib
import re
import time

from . import ci_runs, gitops, overlaps, paths, prep, premises, review, slots
from . import spec as spec_mod
from . import config

# --- rk dispatch-pack -------------------------------------------------------------------------------------------------

WIDTH = 220
# The facts sections the pack prints (their titles' starts) and how many lines of each.
FACT_SECTIONS = (
    ("head and state", 2),
    ("our last review", 24),
    ("unresolved threads", 14),
    ("{me}'s hand-written", 30),
)
# The rule for the code quality pass (review_procedure.md, On REVIEW_READY: verify, check, post, step 2, The code
# quality pass): a follow-up whose own change is only develop merges or docs, or only test files and docs, gets none. A
# CMake, YAML or ini file under tests/ decides what CI runs, so it is no test file.
DOC = re.compile(r"\.(md|rst)$|(^|/)(docs|dsl_docs)/")
TEST = re.compile(
    r"(^|/)(tests?|testing)/|(^|/)test_[^/]*$|_test\.[a-z]+$|(^|/)conftest\.py$"
)
NOT_TEST = re.compile(r"(^|/)CMakeLists\.txt$|\.(cmake|ya?ml|ini|toml|cfg)$")


def lane_rule(c: dict | None, micro: list[str] | None = None) -> str:
    """`lane: needed|exempt (<why>)` for a round whose classification is `c` (classify.txt)."""
    if micro:
        return "lane: exempt (a micro round: the verifier alone, on its new items)"
    c = c or {}
    kind = c.get("kind") or "unknown"
    if kind == "never reviewed":
        return "lane: needed (a first round)"
    if kind == "merge-only":
        return "lane: exempt (only develop merges since our review)"
    if kind == "rebase-only":
        return "lane: exempt (a rebase only)"
    files = c.get("changed_files") or []
    if kind == "same-head" or not files:
        return f"lane: needed ({kind})"
    if (c.get("n_changed") or len(files)) > len(files):
        return (
            f"lane: needed ({c['n_changed']} files changed, more than classify lists)"
        )
    if all(DOC.search(f) for f in files):
        return "lane: exempt (only docs changed)"
    tests = [f for f in files if TEST.search(f) and not NOT_TEST.search(f)]
    if all(DOC.search(f) or f in tests for f in files):
        return (
            "lane: exempt (only test files and docs changed: if they answer our own open findings; else run the "
            "lane)"
        )
    code = [f for f in files if not DOC.search(f) and f not in tests]
    return f"lane: needed ({len(code)} non-test file{'s' if len(code) != 1 else ''}, e.g. {', '.join(code[:3])})"


def fact_sections(text: str, me: str) -> list[str]:
    """The facts' FACT_SECTIONS, each its title and its first lines, cut to WIDTH."""
    blocks: dict[str, list[str]] = {}
    cur = None
    for line in text.splitlines():
        if line.startswith("== "):
            cur = line[3:]
            blocks[cur] = []
        elif cur is not None:
            blocks[cur].append(line)
    out = []
    for start, most in FACT_SECTIONS:
        start = start.format(me=me)
        title = next((t for t in blocks if t.startswith(start)), None)
        if title is None:
            continue
        body = [x for x in blocks[title] if x.strip()]
        out += [f"== {title.split(' (')[0]}"] + [x[:WIDTH] for x in body[:most]]
        if len(body) > most:
            out.append(f"   (+{len(body) - most} more lines in facts.txt)")
    return out


def delta_lines(cfg: dict, rec: dict | None, c: dict | None) -> list[str]:
    """Our reviewed head to the head: the kind, the PR's own changed files, its own non-merge commits and the merges."""
    if not c or "old" not in c:
        return [f"delta: {(c or {}).get('kind', 'unknown')}"]
    out = [
        f"delta: {c['kind']} {c['old'][:10]} -> {c['new'][:10]}: {c.get('n_changed', 0)} of the PR's own files "
        "changed"
    ]
    out += [f"   {f}" for f in (c.get("changed_files") or [])[:15]]
    try:
        merges = gitops.git(
            cfg["paths"]["repo"],
            "rev-list",
            "--count",
            "--merges",
            f"{c['old']}..{c['new']}",
        ).strip()
    except gitops.GitError as exc:
        merges = f"? ({str(exc)[:80]})"
    own = c.get("new_commits") or []
    out.append(
        f"own commits since our review (git log --no-merges, develop's left out): {len(own)}; merges: {merges}"
    )
    out += [f"   {x[:WIDTH]}" for x in own[:15]]
    return out


def shared_build(cfg: dict, n: int) -> str:
    """The checks' shared build of the merged tree, and whether it may still be reused: its slot still kept for it
    (slots.kept_build), and its tree still at the commit it was built at."""
    try:
        build, tree, commit = review.shared_build_record(cfg, n)[:3]
    except ValueError:
        return "shared build: none (the checks built no tree, or ran before they named one)"
    if not slots.kept_build(build):
        return f"shared build: gone (the slot that held {build} was freed): each agent builds its own"
    try:
        now = gitops.git(tree, "rev-parse", "HEAD").strip()
    except (gitops.GitError, OSError) as exc:
        now = f"unreadable ({str(exc)[:60]})"
    if now != commit or not pathlib.Path(build).is_dir():
        return f"shared build: gone ({tree} is at {now[:10]}, the build was of {commit[:10]}): each agent builds its own"
    return (
        f"shared build: {build} (core, pybind) of {tree} at {commit[:10]}: both agents reuse it read-only and build "
        "only a mutant or another tree"
    )


def checks_lines(cfg: dict, n: int) -> list[str]:
    """The checks' head, then their failing and unrun check flags and their leads, by section key."""
    out_dir = pathlib.Path(cfg["paths"]["workers"]) / f"pr{n}" / "checks"
    status, _, head = review.read_checks_status(str(out_dir))
    if not status:
        return [f"checks: no status in {out_dir}"]
    fails = [
        k
        for k, (v, _d) in status.items()
        if v == "fail" and k not in review.LEAD_SECTIONS
    ]
    skips = [
        k
        for k, (v, d) in status.items()
        if v not in ("pass", "fail")
        and k not in review.LEAD_SECTIONS
        and not review.NOTHING_TO_CHECK.search(d)
    ]
    leads = [
        k for k, (v, _d) in status.items() if v == "fail" and k in review.LEAD_SECTIONS
    ]
    return [
        f"checks at {(head or '?')[:10]}: flags: {', '.join(fails) or 'none'}; not run: {', '.join(skips) or 'none'}; "
        f"leads: {', '.join(leads) or 'none'} ({out_dir}/checks.md)",
        shared_build(cfg, n),
    ]


def ci_lines(cfg: dict, n: int) -> list[str]:
    lines, _ok = ci_runs.fails(cfg, n)
    shown = [
        x
        for x in lines
        if x.startswith(("FAIL\t", "RUN\t", "(")) or x.startswith("runs ")
    ]
    fails = sum(1 for x in lines if x.startswith("FAIL\t"))
    running = sum(1 for x in lines if x.startswith("RUN\t"))
    return [f"CI (rk ci-fails): {fails} failing, {running} running"] + [
        x[:WIDTH] for x in shown[:25]
    ]


def decision_lines(store, n: int) -> list[str]:
    """The recorded overlap owner decisions whose members include PR n."""
    out = []
    for key, d in sorted(overlaps.decisions(store).items()):
        if n in (d.get("members") or []):
            role = (
                "owner"
                if d.get("owner") == n
                else f"drops its copy for #{d.get('owner')}"
            )
            out.append(
                f"   {key}: #{n} {role} (members {', '.join('#%d' % m for m in d['members'])})"
            )
    return [f"overlap decisions naming #{n}: {len(out) or 'none'}"] + out[:10]


def micro_items(st: dict, n: int, head: str | None) -> list[str]:
    """A micro round's new items: the review awaiting verification at this head, when REVIEW_READY made it one."""
    for r in st.get("review_ready") or []:
        if r.get("pr") == n and (not head or r.get("head") == head) and r.get("micro"):
            return list(r["micro"])
    return []


def pack(
    cfg: dict, st: dict, store, n: int, rnd: str, out: pathlib.Path
) -> tuple[list[str], bool]:
    """`rk dispatch-pack N`: prepare the review directory, then the compact pack. The lines, and whether every input was
    read."""
    prep_lines, ok = prep.prepare(cfg, st, store, n, out)
    head_line = next(
        (x for x in prep_lines if x.startswith("head ") and " mb " in x),
        "(head and merge base not read)",
    )
    lines = [
        f"== #{n} {rnd}: review dir {out}; "
        + "; ".join(x for x in prep_lines if x.endswith(("exit 0", "exit 1")))
        + f"; {head_line}; {prep_lines[-1] if ok else '?'} files",
        prep.worktree_line(out),
    ]
    try:
        text = (out / "facts.txt").read_text(encoding="utf-8")
    except OSError:
        text = ""
    lines += fact_sections(text, config.operator_login(cfg))
    try:
        c = json.loads((out / "classify.txt").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        c = None
    rec = st["prs"].get(str(n))
    # The briefs' agents read the merged tree: made again when it is gone (a restart empties the RAM disk).
    lines += review.ensure_merge_tree(cfg, rec, n)
    lines += ["== since our last review"] + delta_lines(cfg, rec, c)
    lines += ["== checks"] + checks_lines(cfg, n)
    lines += ["== CI"] + ci_lines(cfg, n)
    lines += ["== overlaps"] + decision_lines(store, n)
    micro = micro_items(st, n, (rec or {}).get("head"))
    if micro:
        lines += [
            f"== {review.MICRO} round: brief the verifier on these items alone"
        ] + [f"   {x}" for x in micro]
    lines.append(lane_rule(c, micro))
    return lines, ok


# --- rk spec-post -----------------------------------------------------------------------------------------------------

DEVBASE_STOP = re.compile(
    r"^premise: (develop|the base \S+) changed (\d+) of this PR's own files since DEVBASE (\S+)"
)
BASE_LINE = re.compile(r"^(?:develop|base \S+): ([0-9a-f]{7,40})$")


def spec_post(
    cfg: dict,
    store,
    n: int,
    rnd: str,
    folder: str,
    out: str | None,
    light: str | None,
    save: str | None,
    cost: dict | None,
    again: bool,
    dry_run: bool,
    devbase_recheck: bool,
    news,
) -> tuple[list[str], int]:
    """`rk spec-post N --round R`: the spec, then the post with its premises; the lines and the exit code (0 posted or
    a dry run that would post; 3 the spec's CHECK lines, nothing posted; 4 stopped or refused; 1 posted with a failed
    after-step)."""
    said, code = spec_mod.write(cfg, n, rnd, folder, out, light, save, cost)
    checks = [x for x in said if x.startswith(spec_mod.CHECK)]
    if code == 4:
        return said, 4
    if code or checks:
        return (
            checks
            + [
                f"spec-post: the spec has {len(checks)} CHECK line{'s' if len(checks) != 1 else ''} (above): "
                "nothing posted; settle them (review_procedure step 4), then run it again"
            ],
            3,
        )
    path = out or str(pathlib.Path(paths.resolve(cfg)["tmp"]) / f"spec{n}{rnd}.json")
    try:
        spec = premises.load(path, n, str(pathlib.Path(folder) / f"pr{n}_{rnd}.md"))
    except ValueError as exc:
        return said + [f"REFUSED: {exc}"], 4
    head = [x for x in said if x.startswith("wrote ")]
    if devbase_recheck:
        lines, ready = premises.check(cfg, store, n, spec, again=again)
        if not ready:
            return head + lines + recheck_devbase(cfg, store, n, spec, lines, again), 4
        return (
            head
            + lines
            + [
                "devbase-recheck: the premises hold, develop included: run it without "
                "--devbase-recheck to post"
            ],
            0,
        )
    lines, code = premises.post_checked(
        cfg, store, n, spec["SRC"], path, again=again, dry=dry_run, news=news
    )
    return head + lines, code


def recheck_devbase(
    cfg: dict, store, n: int, spec: dict, lines: list[str], again: bool
) -> list[str]:
    """After a premise stop: when it is develop's change to the PR's own files and every other premise is true at
    today's develop, develop's diff stat for those files (nothing posted); else what stops it.
    """
    m = DEVBASE_STOP.match(lines[-1]) if lines else None
    dev = next((b.group(1) for b in map(BASE_LINE.match, lines) if b), None)
    if not m or not dev:
        return [
            "devbase-recheck: the stop is not develop's change to the PR's files: nothing to re-check"
        ]
    moved = [
        x
        for x in lines[
            lines.index(next(x for x in lines if x.startswith("PR files: "))) + 1 : -1
        ]
        if x and not x.startswith(("premise", "=="))
    ]
    again_lines, ready = premises.check(
        cfg, store, n, dict(spec, DEVBASE=dev), again=again
    )
    out = [
        f"devbase-recheck: {m.group(1)} changed {m.group(2)} of the PR's files from {m.group(3)} to {dev}:"
    ]
    try:
        stat = gitops.git(
            cfg["paths"]["repo"], "diff", "--stat=200", m.group(3), dev, "--", *moved
        ).splitlines()
    except gitops.GitError as exc:
        stat = [f"(diff not read: {exc})"]
    out += [f"   {x}" for x in stat[:30]]
    if not ready:
        return out + [
            f"devbase-recheck: at {dev} another premise stops it too: {again_lines[-1]}"
        ]
    conflicts = next(
        (x for x in again_lines if x.startswith("conflicts: ")), "conflicts: ?"
    )
    word = "develop" if m.group(1) == "develop" else "base " + m.group(1).split()[-1]
    return out + [
        f"devbase-recheck: every other premise is true at {dev} ({conflicts}). Run the saved probes on a merge "
        f"with it if the PR has any (review_procedure step 4); unchanged: set `{word} {dev}` in notes.md's "
        f"premises: line and run spec-post again (it writes the spec from the notes), or set DEVBASE to "
        f"{dev} in the written spec and post with `rk post {n} FILE --premises SPEC`; changed: a narrow "
        "re-check. Nothing posted"
    ]


# --- rk wait-quality --------------------------------------------------------------------------------------------------

QUALITY, FACTS = "quality.md", "facts.txt"
MARK = ".wait_quality_since"  # the first call's time, in the review directory
SLICE_MIN = (
    4.5  # under the agents' 5-minute prompt cache, and the Bash tool's 10-minute cap
)
POLL_S = 10
PENDING = re.compile(r"^\| Code quality \|.*pending", re.M)


def quality_final(d: pathlib.Path) -> bool:
    q = d / QUALITY
    try:
        text, mtime = q.read_text(encoding="utf-8", errors="replace"), q.stat().st_mtime
    except OSError:
        return False
    facts_time = (d / FACTS).stat().st_mtime if (d / FACTS).exists() else 0
    return bool(text.strip()) and mtime > facts_time and not PENDING.search(text)


def wait_quality(
    d: pathlib.Path,
    max_min: float = 20,
    slice_min: float = SLICE_MIN,
    poll_s: float = POLL_S,
    clock=time.time,
    sleep=time.sleep,
) -> tuple[str, int]:
    """Block until quality.md in d is final (`ready`, 0), for at most `slice_min` this call (`waiting`, 0) and
    `max_min` since the first call (`timeout`, 1)."""
    mark = d / MARK
    t0 = clock()
    try:
        first = float(mark.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        first = 0.0
    if (
        not first or t0 - first > (max_min + 30) * 60
    ):  # no wait under way: this call starts one
        first = t0
        try:
            mark.write_text(f"{first}\n", encoding="utf-8")
        except OSError:
            pass
    until, cap = t0 + slice_min * 60, first + max_min * 60
    while True:
        if quality_final(d):
            mark.unlink(missing_ok=True)
            return f"ready: {d / QUALITY} is final", 0
        t = clock()
        if t >= cap:
            mark.unlink(missing_ok=True)
            return (
                f"timeout: {QUALITY} not final after {max_min:g} min: hand back with the Code quality row "
                "`pending`"
            ), 1
        if t >= until:
            return (
                f"waiting: {QUALITY} not final yet ({(t - first) / 60:.1f} of {max_min:g} min): call it again",
                0,
            )
        sleep(min(poll_s, max(until, t) - t + 0.01, cap - t + 0.01))
