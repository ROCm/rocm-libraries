# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`rk post N FILE --premises SPEC`: post a verified review only while the verifier's premises still hold (lessons
L-224, L-229, L-509). Spec files written by hand in the same format are taken as they are. The spec is a JSON object:

  N, ROUND        the PR and the round (fr, fu<k>): the posted copy is <posts>/pr<N>_<ROUND>.md (paths.posts), the
                  notes archive <lessons>/notes/<UTC date>_pr<N>_<ROUND>.md (paths.lessons)
  SRC, NOTES      the verified post (FILE on the command line must be it) and the verifier's notes.md
  HEAD10          the head it verified (10 characters)
  DEVBASE         the base it verified against: the PR's base branch (`base.ref`: develop, or the feature branch it
                  targets) may not have changed the PR's own files since
  CONFLICTS       the files the merge with the base conflicts in, space-separated ("" when it merges cleanly)
  FAILOK          the failing check runs it accounted for, latest attempts (`rk ci-fails N` lists them)
  MAXCANCEL       the most cancelled latest attempts allowed
  RUNOK, RUNANY   the runs it saw still running (re-check them on the next ticks: L-209), or true when its CI row
                  says CI is still running broadly; the failures stay exact either way
  THREADS         "<total> <resolved>" review threads
  STOP            the UTC time it stopped reading: no review, review comment or human comment may be newer
  TAG             the ticket tag the PR body must still carry; a spec may name it by the JIRA project's key
                  instead (jira.project or tracker_keys), as the skill's docs do. Optional ("" or absent) when the
                  PR body names no key; with a JIRA project, a body that names one while the spec names none stops
                  the post
  fill            [old, new]: the ✓ count filled in at post time; "$ok" in new is the raw count of successful runs
  FOOTER          how the footer line starts (default "_First-round review" for fr, else "_Follow-up review")
  SAVE            the `rk save N verify` text
  AFTER           commands run once the post is saved, with $RK this checkout's rk: a list of arguments runs with no
                  shell (an argument that is a whole `$NAME` takes that variable), as `rk spec` writes the verifier's
                  decide: lines (["$RK", "overlap-decide", KEY, ...]); a string is a shell line. After a review kept
                  unposted while review automation sends nothing, the lists that run $RK alone run, as the monitor's
                  own (`finish`). A line that runs an rk command the Claude session may not run never runs
  LIGHT, OUT      mkpost_fu.py's only (the script's comment and path): ignored
  COST            optional: {"verifier": N, "quality": N}, the tokens the verifier and the code quality pass used
                  (their task notifications' totals; null when unknown), recorded with the draft's in
                  state/review_cost.jsonl (review_cost.py)

`check` runs the premises, then the post's own checks, in the post script's order, then the fix-line premise
(rkmon_lib/fixline.py: the 🔴/🟠 rows' fix lines against where CI runs the test, `rk lane-of`), then the ABI premise
(rkmon_lib/abipremise.py: a "no bump" against the checks' ABI section, the ABI numbers it names against develop's);
the first that fails stops the
post, and its last line says why in one line: what was expected, what is there now, and what to do. The caller then runs `rk post` (its guards, the comment, the ledger) and `finish`: `rk save N
verify SAVE`, the AFTER lines, the notes archive, the drafters' assessment and the cost record. `dry_run` stops
after `rk post`'s guards instead. With review posting off the review is kept unposted (post.save_draft: no comment, no
ledger entry), and `finish` runs only its AFTER lists that run $RK while review automation sends nothing (prototype,
standby), where what they would send stops at the one outward gate, and no AFTER line while review posting is not set
up. A post that the gate holds since the checks (standby began) is kept by `rk post` so too, and `finish` runs as after
a keep. A post and a keep each leave the round's copy in paths.posts (_write_copy), so a ROUND is used once.

With `again` (`--again`: a re-review at a head we posted on or, while review posting is off, at the head of our last
review), neither a post of ours at the live head nor our last review there stops it; the state's head must be the live
head instead, since `rk post --again` skips that guard of its own. Every other premise and check, and every other guard
of `rk post` (the hourly cap among them), holds.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import tempfile

from . import (
    abipremise,
    bindings,
    ci_runs,
    clashes,
    config,
    daylog,
    draft_stage,
    fixline,
    gh,
    gitops,
    outward,
    paths,
    post,
    prep,
    review_cost,
    writer,
)
from .store import now

REQUIRED = (
    "N",
    "ROUND",
    "HEAD10",
    "DEVBASE",
    "THREADS",
    "STOP",
    "CONFLICTS",
    "SRC",
    "NOTES",
    "MAXCANCEL",
    "SAVE",
)
OPTIONAL = (
    "FAILOK",
    "RUNOK",
    "RUNANY",
    "fill",
    "FOOTER",
    "AFTER",
    "LIGHT",
    "OUT",
    "TAG",
    "COST",
)
LIVE_JQ = "{sha: .head.sha, state, merged, updated_at, base: .base.ref}"
_THREADS = """query($n: Int!) { repository(owner: "%s", name: "%s") { pullRequest(number: $n) {
  reviewThreads(first: 100) { totalCount nodes { isResolved } } } } }"""
CROSS_LINK = re.compile(r"#[0-9]")
# A FAILOK or RUNOK value that names no job: none, (none), - or nothing, in any case.
NO_NAMES = re.compile(r"\(?\s*(none|-)?\s*\)?\.?", re.I)
# The words match whole (a product name may take a model number after it): "navigation", "instinctively" and
# "stalemate" are no hit.
HYGIENE = re.compile(
    r"https?://|amd\.atlassian|\bconfluence\b|\bsharepoint\b|gb/s|\bthe user\b|\bMI[0-9]{3}|"
    r"\bnavi(?![a-z])|\bstrix(?![a-z])|\binstinct(?![a-z])|\bradeon(?![a-z])|/home/|@[A-Za-z]|"
    r"\brk abi\b|\brk breakages\b|\bstale(?![a-z])",
    re.I,
)
# A number that starts a word: "md5s" or "fp16s" is no figure ("the .dvc md5s" is not "5 s").
FIGURE = re.compile(
    r"(?<![A-Za-z0-9_.])[0-9]+(\.[0-9]+)? *(ms|us|µs|ns|MHz|GHz|TFLOPS|TFLOP/s|GFLOPS|GB/s|TB/s|s|sec|seconds|min|MB|GB|"
    r"MiB|KiB)\b"
)
VERDICT = re.compile(r"(?<!\w)(?:ACCEPT|REJECT)(?!\w)")
QUALITY_ROW, PENDING = "| Code quality | ", re.compile(r"^\| Code quality \| .*pending")
FIRST_PASS = re.compile(r"\bfirst pass by\b", re.I)  # a footer's credit to a drafter


class Stop(Exception):
    """A premise or check that failed: the lines the post script printed for it, its stop line last."""


def posts_dir(cfg: dict) -> pathlib.Path:
    """Where the posted copy goes: paths.posts (default <WORKING_DIR>/scratch/post)."""
    return pathlib.Path(paths.get(cfg, "posts"))


def verifier_name(cfg: dict) -> str:
    """The model the footer names as the verifier (bindings.verifier_name)."""
    return bindings.verifier_name(cfg)


def verifier_names(cfg: dict) -> set[str]:
    """Every model a footer may name as the one that checked the review: the audit model, the deep review's
    (models.deep) and the orchestrator model, in words."""
    return {
        verifier_name(cfg),
        config.claude_name(cfg, "deep"),
        config.orchestrator_name(cfg),
    }


def tag_keys(cfg: dict) -> list[str]:
    """The spec keys that may hold the ticket tag: TAG, and the JIRA project's key (the skill's docs name it so)."""
    return ["TAG", *config.tracker_keys(cfg)]


def tag(cfg: dict, spec: dict) -> str | None:
    """The ticket tag the spec says the PR body carries, or None when it names none."""
    return next((str(spec[k]) for k in tag_keys(cfg) if spec.get(k)), None)


def load(path: str, n: int, file: str, cfg: dict | None = None) -> dict:
    """The spec at `path` for posting FILE on PR n. ValueError when it cannot be read, lacks a key, has a key nothing
    reads (a misspelt key would drop its premise), or names another PR or post."""
    try:
        spec = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"spec {path}: {exc}") from exc
    if not isinstance(spec, dict):
        raise ValueError(f"spec {path}: not a JSON object")
    known = set(REQUIRED) | set(OPTIONAL) | set(tag_keys(cfg or {}))
    missing, unknown = [k for k in REQUIRED if k not in spec], sorted(set(spec) - known)
    if missing or unknown:
        raise ValueError(
            f"spec {path}: "
            + "; ".join(
                ([f"missing {', '.join(missing)}"] if missing else [])
                + ([f"unknown {', '.join(unknown)}"] if unknown else [])
            )
        )
    if str(spec["N"]) != str(n):
        raise ValueError(f"the spec is for #{spec['N']}, not #{n}")
    if not prep.ROUND.fullmatch(str(spec["ROUND"])):
        raise ValueError(
            f"ROUND {spec['ROUND']!r}: fr, or fr or fu with a number (fr2, fu1)"
        )
    if pathlib.Path(file).resolve() != pathlib.Path(spec["SRC"]).resolve():
        raise ValueError(f"{file} is not the spec's SRC, {spec['SRC']}: wrong spec?")
    if not pathlib.Path(spec["SRC"]).is_file():
        raise ValueError(f"no post at {spec['SRC']}")
    if not str(spec["MAXCANCEL"]).isdigit():
        raise ValueError(f"MAXCANCEL {spec['MAXCANCEL']!r}: a count")
    if spec.get("fill") and not (
        isinstance(spec["fill"], list) and len(spec["fill"]) == 2
    ):
        raise ValueError("fill: [old, new]")
    for key in ("FAILOK", "RUNOK", "AFTER"):
        if not isinstance(spec.get(key) or [], list):
            raise ValueError(f"{key}: a list")
    for x in spec.get("AFTER") or []:
        if not (
            isinstance(x, str)
            or (isinstance(x, list) and x and all(isinstance(a, str) for a in x))
        ):
            raise ValueError(
                f"AFTER: each entry a shell line or a list of arguments, not {x!r}"[
                    :200
                ]
            )
    spec["STOP"] = stop_time(str(spec["STOP"]), now())
    for key in ("FAILOK", "RUNOK"):
        if spec.get(key):
            spec[key] = job_names(spec[key])
    if "COST" in spec:
        review_cost.check_cost(spec["COST"])
    return spec


STOP_SKEW_S = 120  # a STOP this far past now is clock skew between the session and here; more is refused
_STOP = re.compile(r"(\d{4}-\d\d-\d\dT\d\d:\d\d)(:\d\d)?(?:\.\d+)?Z")


def stop_time(stop: str, at: str) -> str:
    """STOP as the comparisons read it, `YYYY-MM-DDTHH:MM:SSZ` (GitHub's times sort as text against it): a time to
    the minute is its minute's first second, so an item within that minute counts as newer (the safe side; as written,
    "...T10:00Z" sorts after "...T10:00:30Z" and would hide it). ValueError for a STOP that is not a UTC time, or one
    later than `at` (now) by more than STOP_SKEW_S: it would hide every new comment."""
    import datetime as dt

    m = _STOP.fullmatch(stop.strip())
    if not m:
        raise ValueError(f"STOP {stop!r}: a UTC time, YYYY-MM-DDTHH:MM:SSZ")
    norm = m.group(1) + (m.group(2) or ":00") + "Z"
    try:
        t = dt.datetime.strptime(norm, "%Y-%m-%dT%H:%M:%SZ")
        ref = dt.datetime.strptime(at, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        raise ValueError(f"STOP {stop!r}: not a time") from None
    if (t - ref).total_seconds() > STOP_SKEW_S:
        raise ValueError(
            f"STOP {stop} is later than now ({at}): the verifier's stop time is when it stopped reading"
        )
    return norm


def job_names(names: list) -> list[str]:
    """The job names in a FAILOK or RUNOK list: an entry that names no job (NO_NAMES) is dropped."""
    return [str(x).strip() for x in names if not NO_NAMES.fullmatch(str(x).strip())]


def prepared_files(cfg: dict, n: int, spec: dict) -> set[str]:
    """The PR's files when its review was prepared: files.txt in the review directory (the post's)."""
    p = pathlib.Path(spec["SRC"]).parent / prep.FILES
    if p.is_file():
        return {f for f in p.read_text(encoding="utf-8").splitlines() if f}
    return set()


def check(
    cfg: dict, store, n: int, spec: dict, again: bool = False
) -> tuple[list[str], dict | None]:
    """The premises, then the post's checks: the lines the post script printed, and what the post needs (the filled
    text, the copy's path, the live head and develop), or None once a check stopped it.
    """
    out: list[str] = []
    try:
        ctx = _premises(cfg, n, spec, out)
        ready = _post_checks(cfg, store, n, spec, ctx, again)
        out += fix_lines(cfg, spec, ready)
        out += abi_premise(cfg, n, spec, ready)
        out += clash_line(cfg, store, n)
        return out, ready
    except Stop as stop:
        return out + list(stop.args), None
    except (gh.GhError, gitops.GitError) as exc:
        return out + [f"premise not read ({exc}): stop"], None
    except UnicodeDecodeError as exc:
        return out + [f"the post is not UTF-8 ({exc}): stop"], None


def fix_lines(cfg: dict, spec: dict, ready: dict) -> list[str]:
    """The fix-line premise (rkmon_lib/fixline.py) over the filled post, with the verifier's notes: its lines; Stop
    when a fix line fails it."""
    try:
        notes = pathlib.Path(spec["NOTES"]).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        notes = ""
    try:
        return fixline.check(cfg, ready["text"], notes, ready["sha"])
    except fixline.FixStop as stop:
        raise Stop(*stop.args) from None


def abi_premise(cfg: dict, n: int, spec: dict, ready: dict) -> list[str]:
    """The ABI premise (rkmon_lib/abipremise.py) over the filled post, with the verifier's notes: its lines; Stop
    when the post fails it."""
    try:
        notes = pathlib.Path(spec["NOTES"]).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError, KeyError):
        notes = ""
    try:
        return abipremise.check(cfg, n, ready["text"], notes, ready.get("dev"))
    except abipremise.AbiStop as stop:
        raise Stop(*stop.args) from None


def clash_line(cfg: dict, store, n: int) -> list[str]:
    """The open PRs the checks' clash scan did not compare (opened since it ran): a line naming them and
    `rk clash N --new`; it does not stop the post."""
    if store is None:
        return []
    try:
        at, new = clashes.unscanned(cfg, store.load(), n)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return []
    if not new:
        return []
    return [
        f"clash scan: {' '.join(f'#{k}' for k in new[:12])} opened after the checks' scan ({at}): run "
        f"`rk clash {n} --new` (its silent clashes are not in the checks)"
    ]


def _newer(slug: str, path: str, jq: str, since: str) -> int:
    """How many of the items at `path` (their times, as `jq` lists them) are newer than `since`."""
    return sum(
        1
        for at in gh.rest(f"repos/{slug}/{path}?per_page=100", paginate=True, jq=jq)
        if (at or "") > since
    )


def new_activity(cfg: dict, slug: str, n: int, since: str) -> tuple[int, int, int]:
    """The reviews, review comments and human comments on PR n newer than `since`; in a mirror our login's are left
    out (mirror.not_ours: the live monitor posts as it)."""
    from . import mirror

    mine = mirror.not_ours(cfg)
    return (
        _newer(slug, f"pulls/{n}/reviews", f"[.[]{mine} | .submitted_at]", since),
        _newer(slug, f"pulls/{n}/comments", f"[.[]{mine} | .updated_at]", since),
        _newer(
            slug,
            f"issues/{n}/comments",
            f'[.[] | select(.user.type != "Bot"){mine} | .updated_at]',
            since,
        ),
    )


def _premises(cfg: dict, n: int, spec: dict, out: list[str]) -> dict:
    """What GitHub and git say now, against the verifier's premises; `out` gets the lines as they come."""
    slug, repo = cfg["repo"], paths.get(cfg, "repo")
    live = gh.rest(f"repos/{slug}/pulls/{n}", jq=LIVE_JQ)
    sha = live["sha"]
    out.append(
        f"live: {sha[:10]} {live['state']} {json.dumps(live['merged'])} {live['updated_at']}"
    )
    if sha[:10] != spec["HEAD10"]:
        raise Stop(
            f"head moved: live {sha[:10]}, the spec verified {spec['HEAD10']}: re-verify at the new head: stop"
        )
    if live["state"] != "open" or live["merged"] is not False:
        raise Stop(
            f"premise: the PR is {live['state']}{', merged' if live['merged'] else ''}: nothing to post: stop"
        )
    # The PR's base branch (`base.ref`): develop, or the feature branch it targets, read from the
    # mirror once fetched. DEVBASE and CONFLICTS are against it.
    branch = live.get("base") or "develop"
    if gitops.is_develop(branch):
        dev, name = gh.rest(f"repos/{slug}/commits/develop", jq=".sha"), "develop"
        out.append(f"develop: {dev[:10]}")
    else:
        gitops.fetch_base(repo, branch)
        dev, name = (
            gitops.git(repo, "rev-parse", gitops.base_ref(branch)).strip(),
            f"the base {branch}",
        )
        out.append(f"base {branch}: {dev[:10]}")
    for fetch in (gitops.fetch_develop, lambda r: gitops.fetch_pr(r, n)):
        try:
            fetch(repo)
        except gitops.GitError:
            pass  # as the script's `|| true`: a commit still missing stops the next git step
    base = gitops.git(repo, "merge-base", dev, sha).strip()
    files = set(
        gitops.git(repo, "diff", "--name-only", base, sha).splitlines()
    ) | prepared_files(cfg, n, spec)
    out.append(f"PR files: {len(files)}")
    moved = [
        f
        for f in gitops.git(
            repo, "diff", "--name-only", spec["DEVBASE"], dev
        ).splitlines()
        if f in files
    ]
    if moved:
        raise Stop(
            *moved[:10],
            f"premise: {name} changed {len(moved)} of this PR's own files since DEVBASE "
            f"{spec['DEVBASE']} (above): re-check them at {dev[:10]} (review_procedure step 4), "
            "then set DEVBASE: stop",
        )
    tree, conflicts = gitops.merge_tree(repo, dev, sha)
    if tree is None:
        raise Stop(
            f"premise: git could not merge the head with {name} {dev[:10]}: stop"
        )
    out.append(f"conflicts: [{''.join(f + ' ' for f in sorted(set(conflicts)))}]")
    if set(conflicts) != set(spec["CONFLICTS"].split()):
        raise Stop(
            f"premise: the conflict set changed: now [{' '.join(sorted(set(conflicts)))}], the spec has "
            f"[{' '.join(sorted(set(spec['CONFLICTS'].split())))}]: re-check the merge: stop"
        )
    src = pathlib.Path(spec["SRC"]).read_bytes().decode("utf-8")
    if "\r" in src:
        raise Stop(
            "the post has CRLF line ends (L-228): strip the carriage returns: stop"
        )
    out.append(
        f"== premise (L-224): failures exactly as verified; cancels at most {spec['MAXCANCEL']}; nothing "
        "running beyond the verified set; the count is filled at post time"
    )
    runs = ci_runs.check_runs(cfg, sha)
    last = ci_runs.latest(runs)
    runok, runany = set(spec.get("RUNOK") or []), bool(spec.get("RUNANY"))
    bad = [k for k, r in last.items() if ci_runs.failed(r)]
    cancelled = sum(1 for r in last.values() if r["conclusion"] == "cancelled")
    ok = sum(1 for r in runs if r["conclusion"] == "success")
    unverified = [
        k
        for k, r in last.items()
        if ci_runs.running(r) and k not in runok and not runany
    ]
    out.append(
        f"latest-attempt failures {len(bad)}; cancelled {cancelled}; raw success {ok}; running or queued "
        f"{len(unverified)}"
    )
    out += [f"   BAD {k}" for k in bad] + [f"   RUN {k}" for k in unverified]
    out += [
        f"   RUNOK {k}"
        for k in sorted(
            k for k, r in last.items() if ci_runs.running(r) and k not in unverified
        )
    ]
    failok = sorted(spec.get("FAILOK") or [])
    if set(bad) != set(failok):
        new, gone = sorted(set(bad) - set(failok)), sorted(set(failok) - set(bad))
        raise Stop(
            "premise: the CI failure set changed: "
            + "; ".join(
                ([f"now failing {', '.join(new)}"] if new else [])
                + ([f"not failing now {', '.join(gone)}"] if gone else [])
            )
            + " (latest attempts): re-verify: stop"
        )
    if cancelled > int(spec["MAXCANCEL"]):
        raise Stop(
            f"premise: {cancelled} cancelled jobs, the spec allows {spec['MAXCANCEL']}: stop"
        )
    if unverified:
        raise Stop(
            f"premise: {len(unverified)} run{'s' if len(unverified) != 1 else ''} beyond the verified set "
            f"(RUN above): update the Bots / CI row and RUNOK: stop"
        )
    t = gh.graphql(_THREADS % tuple(slug.split("/")), n=n)["repository"]["pullRequest"][
        "reviewThreads"
    ]
    threads = f"{t['totalCount']} {sum(1 for x in t['nodes'] if x['isResolved'])}"
    out.append(f"threads total, resolved: {threads}")
    if threads.split() != str(spec["THREADS"]).split():
        raise Stop(
            f"premise: threads changed: now {threads}, the spec has {spec['THREADS']} (total, resolved): stop"
        )
    reviews, review_comments, comments = new_activity(cfg, slug, n, spec["STOP"])
    out.append(
        f"new reviews {reviews}; review comments {review_comments}; human comments {comments}"
    )
    if reviews or review_comments or comments:
        raise Stop(
            f"premise: newer than STOP {spec['STOP']}: {reviews} reviews, {review_comments} review comments, "
            f"{comments} human comments: read them first: stop"
        )
    want = tag(cfg, spec)
    if want or (cfg.get("jira") or {}).get("project"):
        body = gh.rest(f"repos/{slug}/pulls/{n}", jq="{body}")["body"] or ""
        if want and want not in body:
            raise Stop(f"premise: the tag {want} left the PR body: stop")
        from . import spec as spec_mod  # spec imports this module

        named = "" if want else spec_mod.tag_of(spec_mod.tracker_keys(cfg), body)
        if named:  # optional only while the body names no key
            raise Stop(
                f"premise: the PR body names {named}, the spec names no TAG: add TAG {named}: stop"
            )
    return {"sha": sha, "dev": dev, "ok": ok, "src": src}


def _line(lines: list[str], i: int) -> str:
    return lines[i] if i < len(lines) else ""


def hygiene_hits(lines: list[str]) -> list[tuple[str, list[str]]]:
    """The post's public-repo hygiene, check only: (why, its hits as `grep -n` printed them) for each check that hits,
    in the order the post runs them: a #<n> reference outside line 3, a hygiene word, a measured figure, a verdict
    word."""
    out = []
    hits = [
        f"{i}:{line}"
        for i, line in enumerate(lines, 1)
        if i != 3 and CROSS_LINK.search(line)
    ]
    if hits:
        out.append(("a #N reference in the body would cross-link", hits))
    for pattern, why in (
        (HYGIENE, "hygiene hit"),
        (FIGURE, "perf or measured figure"),
        (VERDICT, "verdict word"),
    ):
        hits = [
            f"{i}:{line}" for i, line in enumerate(lines, 1) if pattern.search(line)
        ]
        if hits:
            out.append((why, hits))
    return out


def staged_no_draft(cfg: dict, st: dict, n: int, head: str) -> bool:
    """Whether PR n's review at HEAD was staged with no drafter: the drafter's last record for it that came to
    verification or was staged with none (draft_stage.draft_for, STAGED) has the status "none"; with no such record
    (the local worker's stage keeps none), whether the configuration has no first-pass drafter (bindings.no_draft).
    """
    rec = draft_stage.draft_for(
        st, n, head, statuses=draft_stage.STAGED, role="drafter"
    )
    return rec["status"] == "none" if rec else bindings.no_draft(cfg)


def _post_checks(
    cfg: dict, store, n: int, spec: dict, ctx: dict, again: bool = False
) -> dict:
    """The post itself, its ✓ count filled in: its ROUND not posted or kept yet (no round copy: _write_copy, with
    `again` too), public-repo hygiene, line 1, header, Code quality row and footer (one that credits a first pass only
    for a review staged with a drafter: staged_no_draft), and no post of ours at this head yet, nor, for a review kept
    unposted (post.keeps), our last review (post.last_review_at) (with `again`: the state's head is the live head). A
    hit prints as `grep -n` printed it.
    """
    copy = posts_dir(cfg) / f"pr{n}_{spec['ROUND']}.md"
    if copy.exists():
        raise Stop(
            f"{copy} exists: round {spec['ROUND']} was posted or kept; a new review takes a new ROUND: stop"
        )
    text = ctx["src"]
    if spec.get("fill"):
        old, new = spec["fill"][0], spec["fill"][1].replace("$ok", str(ctx["ok"]))
        text = "\n".join(
            line.replace(old, new, 1) for line in text.split("\n")
        )  # sed's s/old/new/, each line
        if new not in text:
            raise Stop(
                f"CI count not written: the fill anchor {old!r} is not in the post: stop"
            )
    lines = text.split("\n")
    for why, hits in hygiene_hits(lines)[:1]:
        raise Stop(*hits, f"{why} (lines above): stop")
    # line 1 as the review was staged, else as the configuration gives it now
    want = post.review_line(cfg, n, ctx["sha"])
    if lines[0] != want:
        raise Stop(f"line 1: the post's {lines[0][:80]!r} is not {want!r}: stop")
    if _line(lines, 2) != _line(ctx["src"].split("\n"), 2) or not any(
        x.startswith(f"## PR #{n} — ") for x in lines
    ):
        raise Stop(
            f"header: no '## PR #{n} — ' line, or line 3 is not the verified post's: stop"
        )
    footer = spec.get("FOOTER") or (
        "_First-round review" if spec["ROUND"].startswith("fr") else "_Follow-up review"
    )
    if not any(x.startswith(QUALITY_ROW) for x in lines):
        raise Stop("no Code quality row: stop")
    if not any(name in text for name in verifier_names(cfg)):
        raise Stop(
            f"the post never names the verifier ({verifier_name(cfg)}): its footer must: stop"
        )
    if not any(x.startswith(footer) for x in lines):
        raise Stop(
            f"no line starts with the footer {footer!r} (the spec's FOOTER, else the round's): stop"
        )
    if staged_no_draft(cfg, store.load(), n, ctx["sha"]) and any(
        x.startswith(footer) and FIRST_PASS.search(x) for x in lines
    ):
        raise Stop(
            "the footer credits a first pass, but the review was staged with no drafter: write "
            f'"verified by the orchestrator ({bindings.verifier_name(cfg)})" alone: stop'
        )
    if any(PENDING.search(x) for x in lines):
        raise Stop(
            "the Code quality row is still pending: fold the lane in first: stop"
        )
    if (
        again
    ):  # `rk post --again` skips its own check of the live head against the state's, so it is made here
        head = (store.load()["prs"].get(str(n)) or {}).get("head") or ""
        if head != ctx["sha"]:
            raise Stop(
                f"the state's head {head[:10] or 'none'} is not the live head: run rk tick, then post: stop"
            )
    elif store.ledger_has("post", n, ctx["sha"]):
        raise Stop(
            f"already posted at this head {ctx['sha'][:10]}: a re-review posts with --again and a new ROUND: "
            "stop"
        )
    elif post.keeps(cfg) and (
        was := post.last_review_at(store.load()["prs"].get(str(n)), ctx["sha"])
    ):
        raise Stop(
            f"our last review is at this head {ctx['sha'][:10]} already ({was}): a re-review keeps with --again "
            "and a new ROUND: stop"
        )
    return dict(ctx, text=text, copy=copy)


def dry_run(
    cfg: dict, store, n: int, ready: dict, again: bool = False
) -> tuple[list[str], int]:
    """`--dry-run`: `rk post`'s guards on the filled post, written to a temporary file, then stop before the comment:
    the lines, and 0 when it would post (4 when a guard refuses)."""
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / ready["copy"].name
        path.write_bytes(ready["text"].encode("utf-8"))
        try:
            post.preflight(cfg, store.load(), store, n, str(path), again=again)
        except (post.PostRefused, gh.GhError) as exc:
            return [f"REFUSED: {exc}", "post refused: read the message"], 4
    return [
        f"dry run: the premises hold and rk post's guards pass; stopped before the post ({ready['copy']} not "
        f"written; {post.write_login()})"
    ], 0


def rk_path() -> str:
    """$RK for the AFTER lines: the monitor home's rk (it names this home), else the package's."""
    home = config.home() / "rk"
    return str(home if home.exists() else paths.PACKAGE / "rk")


def post_checked(
    cfg: dict,
    store,
    n: int,
    file: str,
    spec_path: str,
    again: bool = False,
    dry: bool = False,
    news=None,
) -> tuple[list[str], int]:
    """`rk post N FILE --premises SPEC [--dry-run]` (and `rk spec-post`): the premises and checks, then the post through
    `rk post` (in the container, as the writer: writer.routed) and `finish`. The checks read GitHub and
    git without the state lock, so a tick is never skipped for them;
    the post takes it, as `rk post`. In a mirror the post goes to the mirror's folder instead (mirror.divert); with
    review-posting off it is kept unposted (post.save_draft), never posted, and `finish` runs only its AFTER lists that
    run $RK, or none.
    `news(cfg, st, only=n)`: the lines a post that makes PR n ready for approval prints. The lines, and the exit code:
    0 posted or kept (or a dry run that would post), 4 stopped or refused, 1 posted or kept with a step after it
    failed."""
    from . import mirror

    try:
        spec = load(spec_path, n, file, cfg)
        # Refused before any check: a keep you type while review automation sends nothing, and a PR in review_skip.
        if post.keeps(cfg):
            post.keep_check(cfg)
        post.skip_check(store.load(), n)
    except (ValueError, post.PostRefused) as exc:
        return [f"REFUSED: {exc}"], 4
    lines, ready = check(cfg, store, n, spec, again=again)
    if not ready:
        return lines, 4
    if dry:
        more, code = dry_run(cfg, store, n, ready, again=again)
        return lines + more, code
    if mirror.on(
        cfg
    ):  # a mirror: the post, its inputs and cost.json go to its folder, not GitHub
        with store.locked(wait=True):
            st = store.load()
            try:
                lines += list(
                    mirror.divert(cfg, st, store, n, ready, spec, again=again)
                )
            except (post.PostRefused, gh.GhError) as exc:
                return lines + [f"REFUSED: {exc}", "post refused: read the message"], 4
            store.save(st)
        more, ok = finish(cfg, store, n, spec, ready)
        return lines + more, 0 if ok else 1
    if post.keeps(cfg):
        # Posting off: kept unposted (the count filled in), not posted, with the after-post work of this machine; the
        # round's copy is written as after a post.
        with tempfile.TemporaryDirectory() as d:
            filled = pathlib.Path(d) / pathlib.Path(file).name
            filled.write_bytes(ready["text"].encode("utf-8"))
            with store.locked(wait=True):
                st = store.load()
                kept = post.save_draft(
                    cfg, st, store, n, str(filled), head=ready.get("sha") or ""
                )
                store.save(st)
        _write_copy(ready)
        lines.append(post.kept_line(cfg, kept))
        more, ok = finish(
            cfg, store, n, spec, dict(ready, copy=pathlib.Path(kept)), kept=True
        )
        return lines + more, 0 if ok else 1
    more, code = _post_ready(cfg, store, n, spec, ready, again, news)
    return lines + more, code


def _write_copy(ready: dict) -> None:
    """The round's copy, ready["copy"] (pr<N>_<ROUND>.md in paths.posts): the filled post, written before a post and
    after a keep. _post_checks stops a post or a keep of a ROUND whose copy is there."""
    copy = ready["copy"]
    copy.parent.mkdir(parents=True, exist_ok=True)
    copy.write_bytes(ready["text"].encode("utf-8"))


def _post_ready(
    cfg: dict, store, n: int, spec: dict, ready: dict, again: bool, news
) -> tuple[list[str], int]:
    """The post of a checked post, then `finish`: its lines and exit code. A post that the one outward gate holds since
    the checks (standby began, say: post.keeps) is kept unposted by `rk post`, as the writer's `rk post` keeps it, and
    counts as a review kept: its round's copy stays, as after any keep, the kept file is its post, and `finish` runs as
    after a keep."""
    said = []
    copy = ready["copy"]
    _write_copy(ready)
    routed = writer.routed(["post", str(n), str(copy), *(["--again"] if again else [])])
    if routed:
        # The container: the post is a plain `rk post` as the writer, through docker/rk_write.py, outside the state lock
        # (its guards run there and take it, and it says what made a PR an approval candidate); the checks above and
        # `finish` stay the monitor's.
        p = subprocess.run(routed, capture_output=True, text=True)
        said = [x for x in (p.stdout + p.stderr).splitlines() if x]
        if p.returncode:
            copy.unlink(missing_ok=True)
            return said + ["post refused: read the message"], p.returncode
    else:
        with store.locked(wait=True):
            st = store.load()
            try:
                said.append(
                    post.keep_as_draft(
                        cfg, st, store, n, str(copy), again=again, cost=False
                    )
                    if post.keeps(cfg)
                    else f"posted: {post.post(cfg, st, store, n, str(copy), again=again)}"
                )
            except (post.PostRefused, gh.GhError) as exc:
                copy.unlink(missing_ok=True)
                return [f"REFUSED: {exc}", "post refused: read the message"], 4
            if news and not post.is_kept_line(said[0]):
                said += list(news(cfg, st, only=n))
            store.save(st)
    kept = any(post.is_kept_line(x) for x in said)
    if kept:
        ready = dict(
            ready,
            copy=next(
                (p for p in reversed(post.copies(cfg, n)) if post.is_kept_file(p)),
                copy,
            ),
        )
    more, ok = finish(cfg, store, n, spec, ready, kept=kept)
    return said + more, 0 if ok else 1


def _command_text(line) -> str:
    """The AFTER line LINE as one shell line, as the session's guard reads a command: a list's arguments joined, and
    $RK (or ${RK}) in it, wherever it is, read as rk, the command it names."""
    text = shlex.join(line) if isinstance(line, list) else line
    return re.sub(r"\$\{?RK\}?(?!\w)", "rk", text)


def finish(
    cfg: dict, store, n: int, spec: dict, ready: dict, kept: bool = False
) -> tuple[list[str], bool]:
    """After the post: `rk save N verify SAVE`, the AFTER lines (with the post script's variables: $RK is this
    checkout's rk, $sha the head, $dev develop, $P the post's copy), then the notes archive, the drafters' assessment
    and the cost record. An AFTER line never runs an rk command the Claude session may not run (the operator's, such as
    the team chat's test card, or rk's own), live or kept: the line says so, and the post, or the kept review, stands.
    It is read as the session's guard reads a command (outward.not_the_sessions: `_command_text`). After a review
    `kept` unposted (review posting off, or the gate held the post; ready["copy"] the kept file) the same, but for its
    AFTER lines: while review automation sends nothing (post.review_held: prototype or standby), only the lists that
    run $RK run, as the monitor's own (only the monitor's own review is kept then, post.keep_check, and its mark,
    outward.OWN_VAR, goes with them), so that what they would send is kept as an unsent item at the one outward gate;
    no other line runs, as it could send past the gate (a spec written by hand runs any command). Kept while review
    posting is not set up, no AFTER line runs. The lines, and False when one failed (the post, or the kept review,
    stands).
    """
    with store.locked(wait=True):
        rec = daylog.save(store, cfg, n, "verify", spec["SAVE"])
    out, ok = [daylog.logged_line(rec, str(n))], True
    env = dict(
        os.environ,
        RK=rk_path(),
        R=f"repos/{cfg['repo']}",
        REPO=paths.get(cfg, "repo"),
        SRC=spec["SRC"],
        NOTES=spec["NOTES"],
        P=str(ready["copy"]),
        STOP=spec["STOP"],
        sha=ready["sha"],
        dev=ready["dev"],
    )
    held = post.review_held(cfg) if kept else ""
    for line in spec.get("AFTER") or []:
        # A list is one command's arguments, run with no shell ($RK and the other variables taken whole, never
        # expanded inside an argument): `rk spec` writes the verifier's decide: lines so. A string is a shell line, as a
        # spec written by hand has it.
        argv = (
            [
                env.get(a[1:], a) if re.fullmatch(r"\$[A-Za-z_]\w*", a) else a
                for a in line
            ]
            if isinstance(line, list)
            else ["bash", "-c", line]
        )
        rk = isinstance(line, list) and line[:1] == ["$RK"]
        name = outward.not_the_sessions(_command_text(line))
        line = " ".join(line) if isinstance(line, list) else line
        if name:
            out.append(
                f"AFTER not run: {outward.not_the_sessions_why(name, config.operator_login(cfg))}: {line}"
            )
            continue
        if kept and not (held and rk):
            out.append(
                f"AFTER not run in {held} (not an rk command, so it could send past the one outward gate): {line}"
                if held
                else f"AFTER not run while review posting is off: {line}"
            )
            continue
        try:
            p = subprocess.run(
                argv,
                env=env,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except subprocess.TimeoutExpired:
            out.append(f"!! AFTER timed out: {line}")
            ok = False
            continue
        out += [x for x in (p.stdout + p.stderr).split("\n") if x]
        if p.returncode:
            out.append(f"!! AFTER failed (exit {p.returncode}): {line}")
            ok = False
    dest = (
        pathlib.Path(paths.get(cfg, "lessons"))
        / "notes"
        / f"{now()[:10]}_pr{n}_{spec['ROUND']}.md"
    )
    if dest.exists():
        out.append("notes archive exists: not overwritten")
    elif not pathlib.Path(spec["NOTES"]).is_file():
        out.append(f"notes not archived: no {spec['NOTES']}")
        ok = False
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(spec["NOTES"], dest)
        out.append("notes archived")
    from . import (
        assessment,
    )  # the draft against this post: the notes' assessment lines, when there was a draft

    try:
        out += assessment.record(cfg, store, n, spec, ready)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        out.append(f"!! the assessment was not recorded: {type(exc).__name__}: {exc}")
    try:
        out += review_cost.record(cfg, store, n, spec, ready, kept=kept)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        out.append(f"!! the cost was not recorded: {type(exc).__name__}: {exc}")
    return out, ok
