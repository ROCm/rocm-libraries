# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`rk spec N --round R --dir DIR`: the spec `rk post N FILE --premises SPEC` reads (premises.py), written from the review
directory and live reads instead of by hand.

From the review directory DIR (`rk review-prep N`'s, where the verifier wrote):
- `notes.md`: its `premises:` line (head, develop, a base branch's `base <branch> <sha>`, the CI counts, threads, the
  stop time), its `FAILOK` line, its `CI pending:` line, its `conflict set` line and its `decide: <key> <owner>
  <members>` lines (each an AFTER command, `["$RK", "overlap-decide", KEY, ...]`, run without a shell; a key that is
  not an overlap key is a CHECK line);
- the verified post `pr<N>_<R>.md`: its header's light (LIGHT, unless --light) and its Bots / CI row, whose ✓ count is
  the fill anchor. A row with no "<n> ✓" gets "; <n> ✓" appended (n: the live count), and the anchor is extended to the
  left until it is unique in the post.
Live (GitHub GETs and the mirror):
- the PR's head, base branch (`base.ref`) and body: the ticket tag the body carries (TAG: a key of the profile's
  `tracker_keys` and the configuration's, config.tracker_keys; any Jira-style key when there is none) and the base the
  merge is checked against;
- `rk ci-fails N`'s picture (ci_runs): the failing latest attempts the notes account for (FAILOK, exact names, a
  name the notes cut with "..." matched by its start) and the runs still going (RUNOK);
- CONFLICTS: `git merge-tree` of the head with the base branch's head (develop's from GitHub, as the premises read it);
- THREADS: the notes' `a/b`, read as open/total or resolved/total, whichever the live threads match.
What the notes and the live reads disagree on is a `CHECK:` line (the spec still keeps the verified side, so the
premise stops the post): a failure the notes do not account for, a run the notes did not see, more cancels, other
threads, another conflict set, another head. Exit 0 with no CHECK line, 3 with one, 4 when the spec cannot be made.

Writes only the spec (default `<paths.tmp>/spec<N><R>.json`) and, when its Bots / CI row had no ✓ count, the post.
"""

from __future__ import annotations

import json
import os
import pathlib
import re

from . import ci_runs, config, gh, gitops, paths, premises, prep

CHECK = "CHECK:"
COUNT = re.compile(r"(?<![\d,.])(\d[\d,]*) ✓")
BOTS_ROW = "| Bots / CI |"
SEP = re.compile(r"\s*[;|]\s*")


class SpecError(Exception):
    """The spec cannot be made (the message says why)."""


def tracker_keys(cfg: dict) -> list[str]:
    """The tracker keys a PR's tag may carry: the repository profile's `tracker_keys` (profile_dir/drafter.json; a
    blank profile_dir, none) and the configuration's (config.tracker_keys: tracker_keys and jira.project); [] for any
    Jira-style key.
    """
    given, keys = str(cfg.get("profile_dir") or "").strip(), []
    if given:
        profile = pathlib.Path(os.path.expanduser(given)) / "drafter.json"
        try:
            keys = (
                json.loads(profile.read_text(encoding="utf-8")).get("tracker_keys")
                or []
            )
        except (OSError, ValueError):
            keys = []
    return list(dict.fromkeys([str(k) for k in keys] + config.tracker_keys(cfg)))


def tag_of(keys: list[str], body: str) -> str:
    """The first ticket tag (KEY-NNNN) in a PR body, of `keys` (any Jira-style key when there is none); "" for none."""
    m = re.search(
        r"\b(?:"
        + ("|".join(map(re.escape, keys)) if keys else r"[A-Z][A-Z0-9]{1,9}")
        + r")-\d+\b",
        body or "",
    )
    return m.group(0) if m else ""


def notes_facts(text: str) -> dict:
    """What a verifier's notes.md says for the spec: its premises line's head, develop, base, CI counts, threads and
    stop time; the FAILOK names (None without the line); the CI pending text; the conflict set (None without the line);
    the decide lines."""
    out: dict = {"failok": None, "pending": None, "conflicts": None, "decide": []}
    for line in text.splitlines():
        s = line.strip()
        low = s.lower()
        if low.startswith("premises:") and "head" not in out:
            p = s.split(":", 1)[1]
            for key, rx in (
                ("head", r"\bhead ([0-9a-f]{7,40})"),
                ("develop", r"\bdevelop ([0-9a-f]{7,40})"),
                ("stop", r"\bstop (\d{4}-\d\d-\d\dT[0-9:]+Z)"),
            ):
                m = re.search(rx, p)
                if m:
                    out[key] = m.group(1)
            m = re.search(r"\bbase (\S+) ([0-9a-f]{7,40})\b", p)
            if m:
                out["base"], out["base_sha"] = m.group(1), m.group(2)
            for key, rx in (
                ("ok", r"(\d+) ✓"),
                ("running", r"(\d+) running"),
                ("failed", r"(\d+) ✗"),
                ("cancelled", r"(\d+) cancelled"),
            ):
                m = re.search(rx, p)
                if m:
                    out[key] = int(m.group(1))
            m = re.search(r"\bthreads (\d+)/(\d+)", p)
            if m:
                out["threads"] = (int(m.group(1)), int(m.group(2)))
        elif low.startswith("failok") and out["failok"] is None:
            names = re.sub(r"^failok[^:]*:\s*", "", s, flags=re.I)
            names = re.sub(r"\s*\((?:rk ci-fails|as printed)[^)]*\)\s*$", "", names)
            out["failok"] = (
                []
                if re.match(r"\(?\s*none\b", names, re.I)
                else premises.job_names(SEP.split(names))
            )
        elif low.startswith("ci pending:") and out["pending"] is None:
            out["pending"] = s.split(":", 1)[1].strip()
        elif low.startswith("conflict set") and out["conflicts"] is None:
            rest = s.split(":", 1)[1].strip() if ":" in s else ""
            paths = [x for x in re.split(r"[;\s\[\]]+", rest) if "/" in x]
            out["conflicts"] = (
                [] if re.match(r"(none|clean|\[\s*\])", rest, re.I) else paths or None
            )
        elif low.startswith("decide:"):
            parts = s.split(":", 1)[1].split()
            if (
                len(parts) >= 3
                and parts[1].isdigit()
                and re.fullmatch(r"[\d,]+", parts[2])
            ):
                out["decide"].append(parts[:3])
    return out


# An overlap group's key, as overlaps.py makes them (e.g. `like/py.func/f+g`): nothing a shell would read.
DECIDE_KEY = re.compile(r"[\w.+:/~-]+")


def _named(note: str, live: str) -> bool:
    """A FAILOK name in the notes names this live job: the same, or its start (a name cut, with or without "...")."""
    n = (
        note.strip().rstrip(".").rstrip("…").strip()
        if note.strip().endswith(("...", "…"))
        else note.strip()
    )
    # A note may follow the name with its reason in parentheses: "<name> (runner lacks pwsh, …)".
    return (
        live == n or (len(n) >= 12 and live.startswith(n)) or n.startswith(live + " (")
    )


def fill(text: str, ok: int) -> tuple[str, list[str], bool]:
    """The fill anchor from the post's Bots / CI row: (the post, maybe with "; <ok> ✓" appended to the row, [old, new],
    whether the post changed). SpecError without a Bots / CI row or a unique anchor."""
    lines = text.split("\n")
    i = next((k for k, x in enumerate(lines) if x.startswith(BOTS_ROW)), None)
    if i is None:
        raise SpecError("the post has no Bots / CI row: no fill anchor")
    changed = False
    m = COUNT.search(lines[i])
    if not m:
        row = lines[i].rstrip()
        if not row.endswith("|"):
            raise SpecError(
                "the Bots / CI row does not end with `|`: no place for its ✓ count"
            )
        lines[i] = row[:-1].rstrip() + f"; {ok} ✓ |"
        text, changed = "\n".join(lines), True
        m = COUNT.search(lines[i])
    row, start, end = lines[i], m.start(), m.end()
    old = row[start:end]
    while text.count(old) != 1:
        if start <= len(BOTS_ROW):
            raise SpecError(
                f"the Bots / CI row's anchor {old!r} is not unique in the post"
            )
        start = row.rfind(" ", len(BOTS_ROW), start - 1) + 1 if start > 0 else 0
        start = max(start, len(BOTS_ROW))
        old = row[start:end]
    new = old[: m.start() - start] + "$ok ✓"
    return text, [old, new], changed


def build(
    cfg: dict,
    n: int,
    rnd: str,
    folder: pathlib.Path,
    light: str | None = None,
    save: str | None = None,
    cost: dict | None = None,
) -> tuple[dict, list[str], str | None]:
    """The spec for PR n's round `rnd` from review directory `folder`: (spec, the lines said, the post's new text when
    its Bots / CI row got a ✓ count). SpecError when it cannot be made."""
    if not prep.ROUND.fullmatch(rnd):
        raise SpecError(f"--round {rnd}: fr, or fr or fu with a number (fr2, fu1)")
    src, notes_path = folder / f"pr{n}_{rnd}.md", folder / "notes.md"
    for p in (src, notes_path):
        if not p.is_file():
            raise SpecError(f"no {p}")
    notes = notes_facts(notes_path.read_text(encoding="utf-8"))
    for key in ("head", "stop"):
        if key not in notes:
            raise SpecError(f"{notes_path}: its premises: line names no {key}")
    post_text = src.read_bytes().decode("utf-8").replace("\r", "")
    slug, repo = cfg["repo"], cfg["paths"]["repo"]
    said: list[str] = []
    from .premises import stop_time
    from .store import now as _now

    try:  # to the second (a minute is its first second), and never in the future
        notes["stop"] = stop_time(notes["stop"], _now())
    except ValueError as exc:
        said.append(f"{CHECK} the notes' {exc}: fix the premises: line's stop")
    live = gh.rest(
        f"repos/{slug}/pulls/{n}", jq="{sha: .head.sha, base: .base.ref, body}"
    )
    sha, branch = live["sha"], live.get("base") or "develop"
    if not sha.startswith(notes["head"]):
        said.append(
            f"{CHECK} the live head is {sha[:10]}, the notes verified {notes['head'][:10]}: re-verify"
        )
    # The base: develop's head as GitHub has it (as the premises read it), or the feature branch's, fetched.
    if gitops.is_develop(branch):
        base_sha = gh.rest(f"repos/{slug}/commits/develop", jq=".sha")
        devbase = notes.get("develop")
        fetches = (gitops.fetch_develop, lambda r: gitops.fetch_pr(r, n))
    else:
        base_sha = ""
        devbase = (
            notes.get("base_sha")
            if notes.get("base") == branch
            else notes.get("develop")
        )
        fetches = (
            lambda r: gitops.fetch_base(r, branch),
            lambda r: gitops.fetch_pr(r, n),
        )
    for f in fetches:
        try:
            f(repo)
        except gitops.GitError:
            pass
    base_sha = (
        base_sha or gitops.git(repo, "rev-parse", gitops.base_ref(branch)).strip()
    )
    if not devbase:
        raise SpecError(
            f"{notes_path}: its premises: line names no develop (or base {branch}) commit"
        )
    said.append(
        f"head {sha[:10]}; base {branch} {base_sha[:10]}; DEVBASE {devbase[:10]} (notes)"
    )
    tree, conflicts = gitops.merge_tree(repo, base_sha, sha)
    if tree is None:
        raise SpecError(f"git could not merge the head with {branch} {base_sha[:10]}")
    conflicts = sorted(set(conflicts))
    said.append(
        f"CONFLICTS [{' '.join(conflicts)}] (merge-tree with {branch} {base_sha[:10]})"
    )
    if notes["conflicts"] is not None and set(notes["conflicts"]) != set(conflicts):
        said.append(
            f"{CHECK} the notes' conflict set [{' '.join(sorted(notes['conflicts']))}] is not today's: re-check "
            "the merge (review_procedure step 4)"
        )
    runs = ci_runs.check_runs(cfg, sha)
    last = ci_runs.latest(runs)
    fails = sorted(k for k, r in last.items() if ci_runs.failed(r))
    going = sorted(k for k, r in last.items() if ci_runs.running(r))
    cancelled = sum(1 for r in last.values() if r["conclusion"] == "cancelled")
    ok = sum(1 for r in runs if r["conclusion"] == "success")
    if notes["failok"] is None:
        failok = fails if notes.get("failed") == len(fails) else []
        if notes.get("failed") != len(fails):
            said.append(
                f"{CHECK} the notes have no FAILOK line and count {notes.get('failed')} ✗; {len(fails)} fail "
                f"now: {', '.join(fails) or 'none'}: write the FAILOK line, or re-verify"
            )
    else:
        failok = [k for k in fails if any(_named(x, k) for x in notes["failok"])]
        for k in fails:
            if k not in failok:
                said.append(
                    f"{CHECK} {k} fails now and the notes do not account for it: re-verify"
                )
        for x in notes["failok"]:
            if not any(_named(x, k) for k in fails):
                said.append(
                    f"{CHECK} the notes' {x} does not fail now: refresh the Bots / CI row"
                )
    said.append(
        f"FAILOK {len(failok)}: {', '.join(failok) or 'none'} (rk ci-fails, latest attempts)"
    )
    pending = (notes["pending"] or "").lower()
    if going and pending.startswith("none"):
        said.append(
            f"{CHECK} running now and not in the notes (CI pending: none): {', '.join(going)}: update the "
            "Bots / CI row"
        )
    said.append(f"RUNOK {len(going)}: {', '.join(going) or 'none'}")
    maxcancel = notes.get("cancelled", cancelled)
    if cancelled > maxcancel:
        said.append(f"{CHECK} {cancelled} cancelled now, the notes counted {maxcancel}")
    t = gh.graphql(premises._THREADS % tuple(slug.split("/")), n=n)["repository"][
        "pullRequest"
    ]["reviewThreads"]
    total, resolved = t["totalCount"], sum(1 for x in t["nodes"] if x["isResolved"])
    if "threads" in notes:
        a, b = notes["threads"]
        reads = [(b, b - a), (b, a)]  # open/total, then resolved/total
        threads = next((r for r in reads if r == (total, resolved)), reads[0])
        if threads != (total, resolved):
            said.append(
                f"{CHECK} threads now {total} total, {resolved} resolved; the notes say {a}/{b}"
            )
    else:
        threads = (total, resolved)
        said.append(
            f"{CHECK} the notes' premises line has no threads: the live {total} {resolved} taken"
        )
    keys = tracker_keys(cfg)
    tag = tag_of(keys, live.get("body") or "")
    text, anchor, changed = fill(post_text, ok)
    said.append(
        f"fill {anchor[0]!r} -> {anchor[1]!r}"
        + (" (appended to the Bots / CI row)" if changed else "")
    )
    header = next((x for x in text.split("\n") if x.startswith(f"## PR #{n} — ")), "")
    light = light or header.rsplit(" — ", 1)[-1].strip()
    # Each decide line is an AFTER command as a list of arguments (run without a shell). The notes are the verifier's,
    # and it reads PR text: a key that is not a key stops the post instead.
    after = []
    for k, o, m in notes["decide"]:
        if not DECIDE_KEY.fullmatch(k):
            said.append(
                f"{CHECK} the notes' decide: key {k[:60]!r} is not an overlap key ({DECIDE_KEY.pattern}): "
                "fix the decide: line"
            )
            continue
        after.append(
            [
                "$RK",
                "overlap-decide",
                k,
                o,
                "--members",
                m,
                "--note",
                f"the verifier's decide line, #{n} {rnd}",
            ]
        )
    spec = {
        "N": str(n),
        "ROUND": rnd,
        "HEAD10": notes["head"][:10],
        "DEVBASE": devbase[:10],
        "LIGHT": light,
        "THREADS": f"{threads[0]} {threads[1]}",
        "STOP": notes["stop"],
        "CONFLICTS": " ".join(conflicts),
        "SRC": str(src),
        "NOTES": str(notes_path),
        "TAG": tag,
        "MAXCANCEL": str(maxcancel),
        "FAILOK": failok,
        "RUNOK": going,
        "RUNANY": False,
        "fill": anchor,
        "SAVE": save or f"{rnd}: {light}",
        "AFTER": after,
    }
    if cost:
        spec["COST"] = cost
    said.append(
        f"TAG {tag or '(none in the body)'}; THREADS {spec['THREADS']}; STOP {spec['STOP']}; MAXCANCEL "
        f"{spec['MAXCANCEL']}; AFTER {len(after)}"
    )
    return spec, said, text if changed else None


def write(
    cfg: dict,
    n: int,
    rnd: str,
    folder: str,
    out: str | None = None,
    light: str | None = None,
    save: str | None = None,
    cost: dict | None = None,
) -> tuple[list[str], int]:
    """`rk spec`: build the spec, write it (and the post, when its Bots / CI row got a ✓ count); the lines and the exit
    code (0, 3 with a CHECK line, 4 when no spec was made)."""
    try:
        spec, said, post_text = build(
            cfg, n, rnd, pathlib.Path(folder), light, save, cost
        )
    except (SpecError, gh.GhError, gitops.GitError, OSError) as exc:
        return [f"REFUSED: {exc}"], 4
    if post_text is not None:
        pathlib.Path(spec["SRC"]).write_bytes(post_text.encode("utf-8"))
        said.append(
            f"wrote {spec['SRC']} (its Bots / CI row now ends with the ✓ count)"
        )
    path = (
        pathlib.Path(out)
        if out
        else pathlib.Path(paths.get(cfg, "tmp")) / f"spec{n}{rnd}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(spec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    said.append(
        f"wrote {path}: rk post {n} {spec['SRC']} --premises {path} [--dry-run]"
    )
    return said, 3 if any(x.startswith(CHECK) for x in said) else 0
