# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A mirror (config mirror.enabled, the one switch): a monitor beside the live one that sends nothing. It reviews the
same PRs at the same heads as the live monitor and keeps each review with its cost, for an A/B comparison of the two
review pipelines (`rk mirror-report`).

- Writes. Every command that sends (rkmon_lib/outward.py: each row with a destination) is refused before any request
  (`guard`: at the top of rkmon.run_command, and for the writer's sends for another process in rkmon.write_gate), but
  `rk post`, whose post is written here instead (`divert`). Below the commands, gh.run (every gh write), a team chat
  message (post.teams_post, in `guard`'s words) and a ticket's filing refuse too, so a write from a tick or the admin
  dashboard goes nowhere either. A refusal of `guard`'s is printed after "REFUSED: ", as rk's other refusals are.
- Picks. The review driver starts reviews only from the live monitor's feed (mirror.feed: its state/mirror_feed.jsonl,
  mounted read-only; one line per live review start, {pr, head, round, kind, again, merge_only, at}), oldest first, at
  exactly that head: a line whose PR moved on GitHub, closed, or is older than mirror.max_age_min is dropped, and lines
  from before this monitor's first mirrored advance (`since`) are not taken. Never the plan's items, the review queue's
  or the draft pool's. A review started from the feed is forced past the settle (the live monitor judged it settled)
  and re-reviews a head the live monitor already posted on (`again`). A review whose head moves while it runs is
  dropped by the pipeline as any is, as the live one is. A review written here whose live review was dropped (the PR's
  head moved with no live post at the reviewed head in the live monitor's ledger: the live post's premises refuse a
  moved head, so none will come) is dropped too, and its place goes to the next feed line: only reviews of one head
  on both sides are pairs. Every drop is a line of <out>/drops.jsonl, so the report sees which PRs were lost and why.
- Arms. Arm A runs the code quality pass as the live monitor does (the orchestrator's Claude reviewer); arm B runs it on
  the quality binding (roles.quality). Without a quality binding every review is arm A; with one, the first mirror.arm_a
  reviews alternate A and B (A first) until arm A has that many, then B. The draft is the drafter binding's in both, so
  arm A isolates the drafter and arm B adds the binding's code quality pass.
- Blind to the live review, and the same last post. The live monitor posts as the same login, so from the moment a PR's
  feed line is in the feed (waiting to start, then while its review runs), a post of ours on it newer than the feed line
  is not taken as our last post (`hold_ours`, at the tick's refresh), the post's premises leave our own comments out of
  the "new comments" count, and REVIEW_READY tells the verifier not to read them. The facts and the review directory
  name our posts as the live monitor's ledger has them up to the feed line (`live_posts`: the feed's folder holds it),
  and the facts list no review of our account posted after it (`posts_until`); our last post's text is the one GitHub
  holds under our login (the live monitor's), the post the live review follows up on. No review of the live monitor's
  stands in for one of the mirror's: `rk review-close` closes none (NO_CLOSE), and the VERIFY_STALE line leaves out
  its pointer to it.
- Posts. `rk post N FILE [--premises SPEC]` runs its premises and rk post's guards as always (a refusal is a refusal: it
  leaves nothing in the folder), and only for a review a feed line started at that very head (a re-verification at a
  head the live monitor did not review here is refused), then writes the would-be post (post.md), the verifier's notes,
  the spec, the draft (draft/: review_agent.md and its evidence check and records), the quality binding's code quality
  pass (quality/: for a review that got it, review.quality_runs_for, and only its record at that head) and cost.json
  into <out>/pr<N>_<round>_<head10>/ (the feed line's round), marks the pair done and clears the review as a post does;
  the premises' finish (the save, the AFTER lines, the notes archive, the drafters' assessment, the cost record) runs
  as after a post.
- cost.json: the arm; whether the review got the quality binding's code quality pass (quality_on_binding); the drafting
  model's tokens in, out and reasoning, and requests, of the draft (per phase) and of that pass (its record at the
  review's head); Claude's from the transcripts (transcripts.py, counted as the report counts the live side's: tokens
  processed by kind, output, the dollars at config prices' Claude Opus rates (rk spend's) and the weighted ieq, and the
  effort each call ran at) for each lane of the PR since the review started (verifier, quality, notice) and for the
  orchestrator session's calls from the start to the post, with how many other mirrored reviews ran in that window;
  and the minutes of the draft, the checks, the code quality pass, the verification and the whole review.
- Checks. With mirror.checks (the live monitor's workers folder, mounted read-only) the review's checks are the live
  review's for the same head, copied once its report is whole (`checks_script`: a light job, no heavy slot, at
  most mirror.checks_wait_min minutes), so the two sides are compared on their reviewing, not on two builds of the
  same code; blank, the mirror runs its own checks.
- At mirror.pairs reviews written (default 15) it takes no more feed lines, and says so once (MIRROR_DONE).

State st["mirror"]: since, started {key: {pr, head, round, kind, arm, feed_at, at}}, done {key: {..., started, dir,
paired}}, dropped {key: why}, done_said; the key is "<pr>:<head>".
"""

from __future__ import annotations

import json
import pathlib
import shutil

from . import gh, paths, registry
from . import config
from .snapshot import same_commit
from .store import minutes_since, now

KIND = "MIRROR"  # a feed line among the items the review driver picks from
TRY_NEXT = (
    "MIRROR_DROP #",
    "MIRROR_WAIT #",
)  # a feed line that starts nothing: the driver tries the next
REFUSAL = "this monitor is a mirror (config mirror.enabled): rk {cmd} writes outward, and a mirror sends nothing"
# `rk review-close`'s refusal in a mirror (review.close_reads): there every review of our account that gives no name is
# the live monitor's, the other side of the comparison, which never stands in for a review of the mirror's.
NO_CLOSE = (
    "this monitor is a mirror (config mirror.enabled): its reviews are compared with the live monitor's, so "
    "rk review-close closes none"
)
DEFAULTS = registry.package_section("mirror")
CHECKS_DONE = "## Auto-merge"  # the checks report's last section: the live checks job wrote it whole
DRAFT_FILES = (
    "review_agent.md",
    "evidence_check.md",
    "evidence_check.json",
    "draft_run.json",
    "tokens_total.json",
    "draft_findings.json",
    "prior_settled.md",
    "prior_leads.md",
    "stats_agent.json",
    "agent_log.txt",
)
QUALITY_FILES = ("quality.md", "quality_run.json", "tokens_total.json", "agent_log.txt")
DROPS = "drops.jsonl"
LEDGER = "actions.jsonl"  # the live monitor's ledger, beside its feed


def settings(cfg: dict) -> dict:
    raw = cfg.get("mirror") or {}
    return {
        k: (raw[k] if raw.get(k) not in (None, "") else v) for k, v in DEFAULTS.items()
    }


def on(cfg: dict) -> bool:
    """This monitor is a mirror (config mirror.enabled)."""
    return bool(settings(cfg)["enabled"])


def guard(cfg: dict, cmd: str) -> str | None:
    """Why command `cmd` may not run in a mirror, in REFUSAL's words (None when it may): every command that sends
    (rkmon_lib/outward.py) but `rk post` and `rk spec-post`, which are diverted (the table's `drafts`).
    """
    from . import outward

    row = outward.row(cmd)
    if on(cfg) and row and row.dest != outward.NONE and not row.drafts:
        return REFUSAL.format(cmd=cmd)
    return None


def live_checks(cfg: dict) -> str:
    """The live monitor's workers folder whose checks a mirrored review copies (mirror.checks); "" for its own."""
    return str(settings(cfg)["checks"] or "") if on(cfg) else ""


def checks_script(cfg: dict, n: int, head: str) -> str:
    """The light job that waits for the live review's checks of PR n at `head` (its report names the head and ends
    with its last section) and copies the folder into this review's; exit 1 when they do not come in time.
    """
    import shlex

    src = shlex.quote(f"{live_checks(cfg)}/pr{n}/checks")
    dst = shlex.quote(f"{paths.get(cfg, 'workers')}/pr{n}/checks")
    wait = int(float(settings(cfg)["checks_wait_min"]))
    first = shlex.quote(f"# Checks for #{n} at {head[:10]} ")
    return (
        f"SRC={src}; DST={dst}; END=$(( $(date +%s) + {wait} * 60 ))\n"
        'whole() { [ -s "$1/checks.md" ] && head -1 "$1/checks.md" | grep -qF '
        + first
        + " && "
        f"grep -q {shlex.quote('^' + CHECKS_DONE)} \"$1/checks.md\"; }}\n"
        "while :; do\n"
        '  if whole "$SRC"; then rm -rf "$DST"; mkdir -p "$DST"; cp -a "$SRC/." "$DST/"\n'
        f'    whole "$DST" && {{ echo "the live review\'s checks for {head[:10]} copied"; exit 0; }}; fi\n'
        f'  [ "$(date +%s)" -lt "$END" ] || {{ echo "the live review\'s checks for {head[:10]} never came '
        f'({wait} min)"; exit 1; }}\n'
        "  sleep ${RKMON_MIRROR_POLL_S:-30}\n"
        "done\n"
    )


def out_dir(cfg: dict) -> pathlib.Path:
    """Where the mirrored reviews go: mirror.out, else <WORKING_DIR>/mirror."""
    return pathlib.Path(
        settings(cfg)["out"] or pathlib.Path(paths.get(cfg, "working_dir")) / "mirror"
    )


def state(st: dict) -> dict:
    m = st.setdefault("mirror", {})
    for k in ("started", "done", "dropped"):
        m.setdefault(k, {})
    return m


def key_of(n: int, head: str) -> str:
    return f"{n}:{head}"


def _jsonl(path) -> list[dict]:
    """A JSON-lines file's objects (a line that does not parse is skipped; [] without the file)."""
    out = []
    try:
        text = pathlib.Path(path).read_text(encoding="utf-8") if path else ""
    except OSError:
        return []
    for line in text.splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if isinstance(r, dict):
            out.append(r)
    return out


def feed(cfg: dict) -> list[dict]:
    """The live monitor's feed lines, as written."""
    return [
        r
        for r in _jsonl(settings(cfg)["feed"])
        if isinstance(r.get("pr"), int) and r.get("head")
    ]


def drop(cfg: dict, st: dict, store, key: str, why: str, feed_at: str = "") -> None:
    """A feed line, a started review or a written one given up: in the state, the journal and <out>/drops.jsonl. A
    written one leaves the count of reviews written, so the next feed line takes its place.
    """
    m = state(st)
    s = m["started"].pop(key, None) or m["done"].pop(key, None) or {}
    if s.get("dir"):
        m.pop("done_said", None)
    m["dropped"][key] = why
    n, head = key.split(":", 1)
    store.journal("review", f"mirror: {n} at {head[:10]} dropped: {why}")
    try:
        out = out_dir(cfg)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / DROPS, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "pr": int(n),
                        "head": head,
                        "round": s.get("round"),
                        "arm": s.get("arm"),
                        "why": why,
                        "feed_at": s.get("feed_at") or feed_at or None,
                        "dir": s.get("dir"),
                        "at": now(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    except (OSError, paths.PathsError):
        pass  # the state and the journal keep it


def live_posted(cfg: dict, n: int, head: str) -> bool:
    """Whether the live monitor's ledger (beside its feed) has a post of ours on PR n at `head`."""
    feed_p = settings(cfg)["feed"]
    return bool(feed_p) and any(
        r.get("kind") == "post"
        and r.get("pr") == n
        and same_commit(str(r.get("head") or ""), head)
        for r in _jsonl(pathlib.Path(feed_p).with_name(LEDGER))
    )


def _ready(st: dict, n: int) -> dict | None:
    return next((r for r in st.get("review_ready") or [] if r["pr"] == n), None)


def _in_flight(st: dict, n: int) -> bool:
    from .review import pipe_of

    return bool(pipe_of(st, n)) or _ready(st, n) is not None


def tidy(cfg: dict, st: dict, store) -> None:
    """A started review that ended without a post (its pipe dropped: the head moved, the PR closed) is dropped. A
    written review is paired once the live monitor's ledger has its post at that head; before that, a PR whose head
    moved on means the live review of that head was dropped (its post's premises refuse a moved head), and the written
    one is dropped too."""
    from .review import interrupted

    m = state(st)
    for key, s in list(m["started"].items()):
        stopped = interrupted(st, s["pr"])
        if stopped and same_commit(stopped.get("head") or "", s["head"]):
            continue  # a stop or restart interrupted it: it starts again after rk start (review.resume_interrupted)
        if not _in_flight(st, s["pr"]) and minutes_since(s.get("at") or now()) > 2:
            drop(
                cfg,
                st,
                store,
                key,
                "its review ended without a post (head moved, PR closed, or no draft)",
            )
    for key, d in list(m["done"].items()):
        if d.get("paired"):
            continue
        pr, head = key.split(":", 1)
        n = int(pr)
        if live_posted(cfg, n, head):
            d["paired"] = now()
            continue
        cur = (st["prs"].get(str(n)) or {}).get("head") or ""
        if cur and not same_commit(cur, head):
            drop(
                cfg,
                st,
                store,
                key,
                f"the live review of {head[:10]} was dropped: the head moved to {cur[:10]} "
                "with no live post at it, so this review has no pair",
            )


def items(cfg: dict, st: dict, store) -> list[dict]:
    """The feed lines the driver may start now, oldest first: not started, done or dropped here, newer than `since`, not
    in a pipe or awaiting verification, not an in-depth review that waits for the operator's approval here (as on the
    driver's own path: review.needs_approval), and within mirror.pairs counting those started. A line past max_age_min
    is dropped."""
    from . import review

    s = settings(cfg)
    m = state(st)
    m.setdefault("since", now())
    tidy(cfg, st, store)
    room = int(s["pairs"]) - len(m["done"]) - len(m["started"])
    if room <= 0:
        return []
    latest: dict[str, dict] = {}
    for r in feed(cfg):
        if (r.get("at") or "") >= m["since"]:
            latest[key_of(r["pr"], r["head"])] = r
    out = []
    for key, r in sorted(latest.items(), key=lambda kv: kv[1].get("at") or ""):
        if (
            key in m["started"]
            or key in m["done"]
            or key in m["dropped"]
            or _in_flight(st, r["pr"])
        ):
            continue
        if minutes_since(r.get("at") or now()) > float(s["max_age_min"]):
            drop(
                cfg,
                st,
                store,
                key,
                f"the feed line is over {s['max_age_min']} min old",
                feed_at=r.get("at") or "",
            )
            continue
        rkind = r.get("kind") or "first-round"
        if review.needs_approval(cfg, st, r["pr"], rkind):
            continue
        out.append(
            {
                "kind": KIND,
                "pr": r["pr"],
                "head": r["head"],
                "round": r.get("round") or "",
                "rkind": rkind,
                "again": bool(r.get("again")),
                "at": r.get("at") or "",
            }
        )
    return out[:room]


def done_line(cfg: dict, st: dict) -> list[str]:
    """MIRROR_DONE once, when mirror.pairs reviews are written."""
    m = state(st)
    pairs = int(settings(cfg)["pairs"])
    if len(m["done"]) >= pairs and not m.get("done_said"):
        m["done_said"] = now()
        return [
            f"MIRROR_DONE: {len(m['done'])} of {pairs} reviews written: no more feed lines are taken "
            f"({out_dir(cfg)})"
        ]
    return []


def arm_for(cfg: dict, st: dict) -> str:
    """The next mirrored review's arm: A (Claude's code quality pass) without a quality binding; with one, A and B
    alternate, A first, until arm A has mirror.arm_a reviews, then B."""
    from .review import quality_binding

    if not quality_binding(cfg):
        return "A"
    m = state(st)
    arms = [r.get("arm") for r in (*m["started"].values(), *m["done"].values())]
    a, b = arms.count("A"), arms.count("B")
    return "A" if a < int(settings(cfg)["arm_a"]) and a <= b else "B"


def arm_of(cfg: dict, st: dict, n: int, head: str) -> str:
    """A mirrored review's arm ("" outside a mirror or for a review not started from the feed)."""
    if not on(cfg):
        return ""
    return str((state(st)["started"].get(key_of(n, head)) or {}).get("arm") or "")


def start(cfg: dict, st: dict, store, item: dict) -> list[str]:
    """Start a feed line's review at its head, in its arm: dropped when GitHub's head is another or the PR is closed."""
    from . import review

    n, head = item["pr"], item["head"]
    key = key_of(n, head)
    try:
        live = gh.rest(f"repos/{cfg['repo']}/pulls/{n}", jq="{state, head: .head.sha}")
        if not isinstance(live, dict) or not live.get("head"):
            raise gh.GhError(f"unexpected answer {str(live)[:80]!r}")
    except gh.GhError as exc:
        return [
            f"REVIEW_WAIT #{n}: mirror: its live head was not read ({str(exc)[:160]}); nothing started"
        ]
    if live.get("state") != "open":
        drop(cfg, st, store, key, f"it is {live.get('state')} on GitHub")
        return [f"MIRROR_DROP #{n}: it is {live.get('state')} on GitHub: not mirrored"]
    if not same_commit(live["head"], head):
        drop(
            cfg,
            st,
            store,
            key,
            f"the live monitor reviewed {head[:10]}; GitHub has {live['head'][:10]}",
        )
        return [
            f"MIRROR_DROP #{n}: the live monitor reviewed {head[:10]}; GitHub has {live['head'][:10]}: not "
            "mirrored"
        ]
    rec = st["prs"].get(str(n))
    if not rec:
        return [
            f"MIRROR_WAIT #{n}: this monitor does not track it yet (the next tick reads it)"
        ]
    before = state(st)["started"].get(key) or {}
    arm = before.get("arm") or arm_for(cfg, st)  # a review started again keeps its arm
    # Recorded before the start: the code quality pass reads its arm as the review starts.
    state(st)["started"][key] = {
        "pr": n,
        "head": head,
        "round": item["round"],
        "kind": item["rkind"],
        "arm": arm,
        "again": bool(item.get("again")),
        "feed_at": item["at"],
        "at": now(),
    }
    again = item["again"] or same_commit(
        (rec.get("ours") or {}).get("head") or "", head
    )
    lines = review.start_review(
        cfg, st, store, n, item["rkind"], force=True, again=again
    )
    last = lines[-1] if lines else ""
    if last.startswith(("REVIEW_SKIP", "REVIEW_CLOSED")):
        drop(cfg, st, store, key, last[:160])
    elif not last.startswith("REVIEW_START"):
        state(st)["started"].pop(
            key, None
        )  # not started (settling, a GitHub read): the line is tried again
    return lines


_FEED_CACHE: dict = (
    {}
)  # the feed's lines by (path, mtime, size): hold_ours reads it once per tick, not once per PR


def _feed_cached(cfg: dict) -> list[dict]:
    p = settings(cfg)["feed"]
    try:
        s = pathlib.Path(p).stat() if p else None
    except OSError:
        s = None
    key = (p, s.st_mtime_ns, s.st_size) if s else (p, None, None)
    if _FEED_CACHE.get("key") != key:
        _FEED_CACHE.update(key=key, lines=feed(cfg) if s else [])
    return _FEED_CACHE["lines"]


def pinned_at(cfg: dict, st: dict, n: int) -> str:
    """The feed time PR n's last post of ours is pinned to ("" when none): the earliest feed line of PR n that is
    started here, or taken from the feed and not yet started, written or dropped (a line waiting for a pipe). From the
    moment the line is in the feed, a post of ours newer than it is the live review's answer, never our last post.
    """
    m = state(st)
    since = m.get("since") or ""
    ats = [
        s.get("feed_at") or s.get("at") or ""
        for s in m["started"].values()
        if s["pr"] == n
    ]
    max_age = float(settings(cfg)["max_age_min"])
    for r in _feed_cached(cfg):
        if r["pr"] != n or (r.get("at") or "") < since:
            continue
        if minutes_since(r.get("at") or now()) > max_age:
            continue  # too old to be taken: items() drops it
        key = key_of(r["pr"], r["head"])
        if key in m["done"] or key in m["dropped"]:
            continue
        ats.append(r.get("at") or "")
    ats = [a for a in ats if a]
    return min(ats) if ats else ""


def hold_ours(cfg: dict, st: dict, n: int, ours: dict | None) -> bool:
    """Whether the tick keeps PR n's last post of ours as it was: in a mirror, while a feed line of the PR waits or
    its mirrored review runs, and the post GitHub shows is newer than that feed line (the live monitor's post of this
    round, under our login): pinned_at.
    """
    if not on(cfg) or not ours:
        return False
    pin = pinned_at(cfg, st, n)
    return bool(pin) and (ours.get("at") or "") > pin


def not_ours(cfg: dict) -> str:
    """A jq filter that leaves our own login's items out of the premises' "new reviews and comments" count, in a
    mirror only: the live monitor posts as the same login. "" otherwise."""
    me = str(config.operator_login(cfg) or "")
    return (
        f' | select(.user.login != "{me}")'
        if on(cfg) and me.replace("-", "").isalnum()
        else ""
    )


def posts_until(cfg: dict, st: dict, n: int) -> str:
    """The time up to which a mirror shows our posts on PR n, the live monitor's under our login: the feed line of PR
    n's mirrored review (the posts the live review knew; now while none is started). The live monitor's later posts on
    the PR are the live review's own, which the mirror's review is compared with blind. "" outside a mirror.
    """
    if not on(cfg):
        return ""
    return (
        max(
            (
                s.get("feed_at") or ""
                for s in state(st)["started"].values()
                if s["pr"] == n
            ),
            default="",
        )
        or now()
    )


def live_posts(cfg: dict, st: dict, n: int) -> list[dict] | None:
    """Our posts on PR n as the live monitor's ledger has them (beside its feed), up to posts_until; None outside a
    mirror."""
    if not on(cfg):
        return None
    until = posts_until(cfg, st, n)
    ledger = (
        pathlib.Path(settings(cfg)["feed"]).with_name(LEDGER)
        if settings(cfg)["feed"]
        else None
    )
    return [
        r
        for r in (_jsonl(ledger) if ledger else [])
        if r.get("pr") == n and (r.get("ts") or "") <= until
    ]


def ready_note(cfg: dict, st: dict, n: int, head: str, bound: bool) -> str:
    """REVIEW_READY's mirror note: where the post goes, the arm, and what the verifier must not read. `bound`: whether
    the quality binding runs the review's code quality pass (review.quality_runs), which arm B's note says only then.
    """
    if not on(cfg):
        return ""
    s = state(st)["started"].get(key_of(n, head)) or {}
    since = s.get("feed_at") or "the review's start"
    arm = s.get("arm") or "A"
    lane = (
        "arm A: run the code quality pass as usual"
        if arm == "A"
        else (
            "arm B: the quality binding runs the code quality pass"
            if bound
            else "arm B"
        )
    )
    return (
        f" A mirror's review, {lane}; nothing is posted: `rk post` writes this review to {out_dir(cfg)}. "
        f"Review it blind to the live monitor's: no post or comment of ours on the PR newer than {since}."
    )


# --- the post, written here ------------------------------------------------------------------------------------------


def _json(path: pathlib.Path) -> dict:
    try:
        out = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return out if isinstance(out, dict) else {}


def _copy(files, src: pathlib.Path, dest: pathlib.Path) -> list[str]:
    copied = []
    for name in files:
        if (src / name).is_file():
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / name, dest / name)
            copied.append(name)
    return copied


def _min(seconds) -> float | None:
    return round(float(seconds) / 60.0, 1) if seconds not in (None, "") else None


def _between(a: str, b: str) -> float | None:
    """Minutes from time a to time b (0 at the least), None without both."""
    if not a or not b:
        return None
    return round(max(0.0, minutes_since(a) - minutes_since(b)), 1) + 0.0


def _iso(t: float | None) -> str:
    import time

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t)) if t else ""


def _model_tokens(total: dict) -> dict:
    """A drafting model's counts: tokens in, out and reasoning, cached, all, and requests."""
    return {
        "requests": total.get("requests"),
        "prompt_tokens": total.get("prompt_tokens"),
        "completion_tokens": total.get("completion_tokens"),
        "reasoning_tokens": total.get("reasoning_tokens"),
        "cached_tokens": total.get("cached_tokens"),
        "tokens": total.get("tokens"),
    }


def draft_cost(cfg: dict, st: dict, n: int, head: str, since: str) -> dict:
    """The draft's tokens, requests and minutes (a binding's draft, per phase; a local worker's tokens from the
    reviewer model's log since the review started)."""
    from . import draft_stage, usage

    ws = draft_stage.folder(cfg, n)
    run, tot = (
        _json(ws / "draft_run.json"),
        _json(ws / "tokens_total.json").get("total") or {},
    )
    rec = draft_stage.draft_for(st, n, head) or {}
    if run and same_commit(str(run.get("head") or ""), head):
        return {
            "by": run.get("binding") or rec.get("binding"),
            "model": run.get("model"),
            "method": run.get("method"),
            "status": rec.get("status") or ("ready" if run.get("draft") else "missing"),
            **_model_tokens(tot),
            "wall_min": _min(tot.get("wall_s")),
            "phases": {
                name: {k: p.get(k) for k in ("status", "wall_s", "requests", "tokens")}
                for name, p in (run.get("passes") or {}).items()
            },
        }
    stats = _json(ws / "stats_agent.json")
    rows = [r for r in usage.model_rows(cfg, since) if r.get("who") == f"pr{n}"]
    return {
        "by": "local-worker",
        "model": None,
        "method": "local-worker",
        "status": "ready" if stats else "missing",
        "requests": len(rows) or None,
        "prompt_tokens": sum(int(r.get("prompt") or 0) for r in rows) or None,
        "completion_tokens": sum(int(r.get("answer") or 0) for r in rows) or None,
        "tokens": sum(
            int(r.get("prompt") or 0) + int(r.get("answer") or 0) for r in rows
        )
        or None,
        "wall_min": _min(stats.get("wall_s")),
    }


def claude_cost(cfg: dict, st: dict, n: int, since: str) -> dict:
    """Claude's tokens for PR n's review since `since`, from this machine's transcripts (transcripts.py, the count the
    report makes of the live side): each lane's runs and tally, the orchestrator session's calls from `since` to now
    (every main session here is the monitor's), and how many other mirrored reviews ran in that window.
    """
    from . import transcripts

    root = transcripts.claude_projects()
    price = transcripts.prices(cfg)  # config prices, as rk spend reads them
    lanes = transcripts.lanes_for(
        transcripts.runs(root, since=since, price=price), n, since
    )
    session = transcripts.session(root, since, "9999-12-31", marker="", price=price)
    m, t = state(st), now()

    def overlaps(
        r: dict,
    ) -> (
        bool
    ):  # a review running in [since, now]: still running, or written since `since`
        done = "dir" in r
        start, end = (r.get("started"), r.get("at")) if done else (r.get("at"), t)
        return (start or "") <= t and (end or "") >= since

    others = [
        r
        for r in (*m["started"].values(), *m["done"].values())
        if r.get("pr") != n and overlaps(r)
    ]
    return {"lanes": lanes, "session": session, "session_overlap": len(others)}


def _quality_run(cfg: dict, st: dict, n: int, head: str) -> dict:
    """The quality binding's record of PR n's code quality pass (quality_run.json) when it is this review's: the review
    got the pass (review.quality_runs_for) and the record is for `head`. {} else: a review with no such pass by design
    keeps no earlier review's pass for its own."""
    from . import review

    if not review.quality_runs_for(cfg, st, n):
        return {}
    run = _json(review.quality_folder(cfg, n) / "quality_run.json")
    return run if same_commit(str(run.get("head") or ""), head) else {}


def cost(cfg: dict, st: dict, store, n: int, head: str, line: dict) -> dict:
    """cost.json of PR n's mirrored review at `head`."""
    from . import review, transcripts

    started = line.get("at") or ""
    ready = _ready(st, n) or {}
    q = review.quality_folder(cfg, n)
    qrun = _quality_run(cfg, st, n, head)
    claude = claude_cost(cfg, st, n, started)
    lanes = claude["lanes"]

    def lane(name: str) -> dict | None:
        got = lanes.get(name)
        if not got:
            return None
        t = got["tally"]
        return {
            "by": "claude",
            "runs": got["runs"],
            "processed": t.get("processed"),
            "output": t.get("output"),
            "usd": t.get("usd"),
            "ieq": t.get("ieq"),
            "effort": transcripts.effort_of(t),
            "tally": t,
        }

    if qrun:
        qp = (qrun.get("passes") or {}).get("quality") or {}
        quality = {
            "by": qrun.get("binding"),
            "model": qrun.get("model"),
            "status": qp.get("status"),
            **_model_tokens(_json(q / "tokens_total.json").get("total") or {}),
            "effort": (review.quality_binding(cfg) or {}).get("effort"),
            "wall_min": _min(qp.get("wall_s")),
            "findings": qrun.get("findings"),
            "light": qrun.get("light"),
        }
    else:
        quality = lane("quality")
    job = ready.get("jobs") or {}
    checks_end = _iso(_ended(store, job.get("checks")))
    out = {
        "pr": n,
        "head": head,
        "round": line.get("round"),
        "kind": line.get("kind"),
        "arm": line.get("arm") or "A",
        "quality_on_binding": review.quality_runs_for(cfg, st, n),
        "feed_at": line.get("feed_at"),
        "started": started,
        "ready": ready.get("at"),
        "written": now(),
        "draft": draft_cost(cfg, st, n, head, started),
        "quality": quality,
        "verifier": lane("verifier"),
        "notice": lane("notice"),
        "claude": claude,
        "minutes": {
            "draft": None,
            "checks": _between(
                (job.get("checks") or {}).get("started") or "", checks_end
            ),
            "quality": _min(
                ((qrun.get("passes") or {}).get("quality") or {}).get("wall_s")
            ),
            "verify": _between(ready.get("at") or "", now()),
            "review": _between(started, now()),
        },
    }
    out["minutes"]["draft"] = out["draft"].get("wall_min")
    return out


def _ended(store, job: dict | None) -> float | None:
    from .draft_stage import _ended as ended

    return ended(store, job)


def divert(
    cfg: dict,
    st: dict,
    store,
    n: int,
    ready: dict,
    spec: dict | None,
    again: bool = False,
) -> list[str]:
    """`rk post` in a mirror: rk post's guards on the would-be post (a refusal raises post.PostRefused, as a post
    would), then the review, its inputs and cost.json in its folder; the pair done; the review cleared as a post clears
    it. The lines to print."""
    from . import post as post_mod
    from . import prep, review

    head = ready.get("sha") or (st["prs"].get(str(n)) or {}).get("head") or ""
    key = key_of(n, head)
    m = state(st)
    line = m["started"].get(key)
    if line is None:
        # Only a review a feed line started at this very head has a live review to pair with: a re-verification at a
        # head the live monitor never reviewed (its feed line was dropped when the head moved) writes nothing and
        # takes no place among the reviews written.
        raise post_mod.PostRefused(
            f"this monitor is a mirror, and no feed line started a review of #{n} at {head[:10]} (the live "
            "monitor did not review that head here, or its line was dropped): nothing is written, so it takes "
            "no pair's place"
        )
    rnd = line.get("round") or (spec or {}).get("ROUND") or prep.default_round(st, n)
    out = out_dir(cfg) / f"pr{n}_{rnd}_{head[:10]}"
    made = not out.exists()
    out.mkdir(parents=True, exist_ok=True)
    post = out / "post.md"
    post.write_bytes(ready["text"].encode("utf-8"))
    try:
        post_mod.preflight(
            cfg, st, store, n, str(post), again
        )  # rk post's guards, as a post runs them
    except Exception:
        # A refused post leaves nothing behind: the folder holds written reviews only.
        post.unlink(missing_ok=True)
        if made:
            shutil.rmtree(out, ignore_errors=True)
        raise
    if spec:
        (out / "spec.json").write_text(
            json.dumps(spec, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        if spec.get("NOTES") and pathlib.Path(spec["NOTES"]).is_file():
            shutil.copyfile(spec["NOTES"], out / "notes.md")
    ws = pathlib.Path(paths.get(cfg, "workers")) / f"pr{n}"
    _copy(DRAFT_FILES, ws, out / "draft")
    quality = (
        _copy(QUALITY_FILES, review.quality_folder(cfg, n), out / "quality")
        if _quality_run(cfg, st, n, head)
        else []
    )
    rec = cost(cfg, st, store, n, head, dict(line, round=rnd))
    (out / "cost.json").write_text(
        json.dumps(rec, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    m["started"].pop(key, None)
    m["done"][key] = {
        "pr": n,
        "head": head,
        "round": rnd,
        "arm": rec["arm"],
        "started": line.get("at"),
        "at": now(),
        "dir": str(out),
    }
    # Finished with no record of its own: our last review is the one GitHub shows, the live monitor's (refresh_ours).
    review.review_finished(st, n)
    store.journal(
        "review",
        f"#{n} mirror: the review at {head[:10]} written to {out} (nothing posted)",
    )
    pairs = int(settings(cfg)["pairs"])
    return [
        f"MIRRORED #{n}: the would-be post, its notes, draft{', code quality pass' if quality else ''} and cost.json "
        f"in {out} (a mirror, arm {rec['arm']}: nothing posted); {len(m['done'])} of {pairs} reviews written"
    ] + done_line(cfg, st)


def status_lines(cfg: dict, st: dict) -> list[str]:
    """`rk review-status`'s mirror lines."""
    if not on(cfg):
        return []
    s, m = settings(cfg), state(st)
    waiting = len(
        [r for r in feed(cfg) if (r.get("at") or "") >= (m.get("since") or "")]
    )
    since = m.get("since") or "the first advance"
    arms = [r.get("arm") for r in m["done"].values()]
    paired = sum(1 for r in m["done"].values() if r.get("paired"))
    return [
        f"mirror: on; feed {s['feed'] or '(not set)'} ({waiting} line(s) since {since}); {len(m['done'])} of "
        f"{s['pairs']} reviews written ({paired} with the live post; arm A {arms.count('A')}, arm B "
        f"{arms.count('B')}), {len(m['started'])} in progress, {len(m['dropped'])} dropped; out {out_dir(cfg)}"
    ]
