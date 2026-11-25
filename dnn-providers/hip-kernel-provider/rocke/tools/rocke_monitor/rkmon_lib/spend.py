# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Review spend: what each review posted or kept unposted cost, by part, per UTC day and per ISO week, with the average
and median per review, the spread and the outliers (`rk spend`, the daily files, the dashboards' card and the
weekly report's section). Scripts only: no model call.

Sources:
- The paid reviews (`paid_reviews`), each posted or kept unposted:
  - the post ledger (state/actions.jsonl, kind "post"): a review posted, at the time it was posted. Its note gives
    the kind: first round, follow-up, CI update. state/review_cost.jsonl gives the round where it holds the post's
    head.
  - the reviews kept unposted while review posting is off (their lines of state/review_cost.jsonl, `unposted`; they
    have no ledger entry): a review kept, at the time it was kept, its round giving the kind.
- Claude Code's transcripts (spend_scan.py, from config `paths.claude_projects`, else $CLAUDE_CONFIG_DIR/projects):
  - verifier and code quality pass runs, matched to the first review of the PR their description names, at most
    RUN_WINDOW_H hours after the run started. A review's first verifier run, a deep review's included, is its
    verifier, any later one its second verifier; its code quality runs are its code quality pass. A verifier or code
    quality run no review follows is overhead (runs no review followed).
  - the main session's turns (orchestration): a turn that names one to TURN_MAX_PRS reviewed PRs in its tool calls
    (or in the notification that woke it) is split evenly between those PRs' reviews: the first review of the PR
    within TURN_AHEAD_H hours after the turn started, else the last one within TURN_BEHIND_H hours before it. Any
    other turn is unattributed orchestration (overhead), except turns that only start or hear back from experiment
    runs or other tasks, which are not review spend. A review whose orchestration is more than its verifier and lane
    together while its PR is never the only PR of a turn booked to it is marked suspect: those turns may name it
    without acting on it (`rk spend` lists the suspects under the outliers).
- state/spend_history.json, when there is one: the runs and turns (spend_scan's form, each run with its role, PR and
  round) of a monitor whose transcripts this machine cannot read, such as the one this monitor took over from. They
  are booked as the history holds them before its `until`; this machine's transcripts count from `until` on.
- Prices: config `prices` (US$ per million tokens by model id, one table). Usage is recorded by the model id its pass
  asked for (ASKED: the session's turns and the runs of the verifier, a deep review's verifier and the code quality
  pass) and priced by that id's entry, else, for an id with a suffix in brackets (Claude Code's context window), by
  its base id's: claude-opus-5-5[1m] at claude-opus-5-5's prices. A record keyed by family is the usage of requests
  whose pass asked for no id (the runs of no pass: notices, other tasks, experiments; and what the scan reads with no
  ids to ask for), under the family of the model each reports; it is priced by the family's newest id in the table
  (`price_of`). A model with no entry has its cost unknown: its usage is left out of the dollars, and the reports and
  the card name it. A transcript records a request's output at the start of its stream only, so every transcript
  figure uses estimated output (spend_scan.py).
- The model guard: the requests of the last GUARD_DAYS days whose reported model is of another family
  than the id their pass asked for (spend_scan's `off`, `guard`), a WARN in `rk spend`, on the admin dashboard's card
  (webdash.spend_card) and in rk doctor (doctor.model_guard_check, from the card).
- Covered heads (report.equivalent_clears: merge-only and rebase-only heads recorded as covered by our review) cost no
  Claude spend: they are counted apart, never in the per-review averages.
- The reviewed diff's size (rkmon_lib/hybrid/size.py: effective lines; a follow-up's lines new since the head of the
  review it follows up on: a post's our previous post, a kept review's our previous review) gives the size tier; an
  outlier also gets its impact tier (rkmon_lib/hybrid/impact.py, scored without the clash and develop signals). Both
  cached in state/spend_sizes.json.
- Experiment spend (the experiment ledgers `<scratch>/*/spend.jsonl`, experiment runs in the transcripts) and other
  tasks' runs are never review spend: `rk spend --experiments` shows them on their own.

A review's cost is booked on the day it was posted or kept; overhead on the day it was spent. Days and weeks are UTC;
data starts on the first review.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import statistics
import subprocess
import time
from collections import defaultdict

from . import deploy_tree, idle, paths, report, spend_scan, whole_file
from .store import configured_paths, minutes_since, now

SUMMARY = "spend_summary.json"
SIZES = "spend_sizes.json"
HISTORY = "spend_history.json"
PARTS = ("verifier", "verifier2", "lane", "orchestration")
PART_LABEL = {
    "verifier": "verifier",
    "verifier2": "second verifier",
    "lane": "code quality pass",
    "orchestration": "orchestration",
}
KIND_LABEL = {
    "first_round": "first round",
    "followup": "follow-up",
    "ci-update": "CI update",
}
TIERS = ("<1.5k", "1.5k-10k", "10k+", "?")
BY = ("part", "kind", "size", "pr")
RUN_WINDOW_H = 48
TURN_AHEAD_H, TURN_BEHIND_H, TURN_MAX_PRS = 12, 6, 4
REFRESH_MIN = 60  # the tick refreshes the card at most this often
TICK_MAX_BYTES = (
    64 << 20
)  # transcript bytes one tick reads at most (the rest waits for the next)
TICK_SIZE_BUDGET_S = 20  # seconds one tick spends measuring reviewed diffs at most
FILES_AFTER_H = 0.25  # the day's files are written this long after the UTC day ends
# The configuration role whose model each pass of the transcripts asks for (config.claude_model_id), by spend_scan's
# role: the session's turns (spend_scan.MAIN), the verifier's runs (models.audit), and a deep review's verifier's
# and the code quality pass's (models.deep), as the agent files and offload.md name them (agentic.py).
ASKED = {
    spend_scan.MAIN: "orchestrator",
    "verifier": "audit",
    "deep_verifier": "deep",
    "quality": "deep",
}
GUARD_DAYS = 7  # the model guard looks at the last this many UTC days, today's included


# --- prices -----------------------------------------------------------------------------------------------------------


def on(cfg: dict) -> bool:
    """Spend is on: the configuration has its price table (config `prices`)."""
    return bool(cfg.get("prices"))


def projects_root(cfg: dict) -> str:
    """The Claude Code projects folder the monitor's sessions write their transcripts to: config paths.claude_projects,
    else this machine's (transcripts.claude_projects: $CLAUDE_CONFIG_DIR/projects, else ~/.claude/projects).
    """
    from . import transcripts

    return str(
        (cfg.get("paths") or {}).get("claude_projects") or transcripts.claude_projects()
    )


def price_table(cfg: dict) -> dict:
    """config `prices` without its notes (keys starting "_"). ValueError when there is none."""
    p = cfg.get("prices")
    table = (
        {
            k: v
            for k, v in (p or {}).items()
            if not k.startswith("_") and isinstance(v, dict)
        }
        if isinstance(p, dict)
        else {}
    )
    if not table:
        raise ValueError(
            "no price table in config (prices: US$ per million tokens by model)"
        )
    return table


def rates(prices: dict, fam: str) -> dict:
    """The rates of a model family (deploy_tree.FAMILIES): the table's claude-<family>-* entry, newest first."""
    keys = sorted((k for k in prices if f"-{fam}-" in f"{k}-"), reverse=True)
    if not keys:
        raise ValueError(f"no price for {fam} in config prices")
    return prices[keys[0]]


def price_of(prices: dict, key: str) -> dict | None:
    """The prices of a use's key (spend_scan): its own entry; else, for an id with a suffix in brackets, its base
    id's (deploy_tree.base_id: claude-opus-5-5[1m] at claude-opus-5-5's); else, for a family (deploy_tree.FAMILIES: a
    record keyed by family), its newest id's (`rates`). A model id with no entry, its own or its base id's, gets no
    other's. None when the table has none: the cost is unknown.
    """
    if key in prices:
        return prices[key]
    if deploy_tree.base_id(key) in prices:
        return prices[deploy_tree.base_id(key)]
    if key in deploy_tree.FAMILIES:
        try:
            return rates(prices, key)
        except ValueError:
            return None
    return None


def asked_ids(cfg: dict) -> dict:
    """The model id each pass of the transcripts asks for (ASKED), by spend_scan's role: what scan records usage by."""
    from . import config

    return {role: config.claude_model_id(cfg, of) for role, of in ASKED.items()}


def usd(
    prices: dict, use: dict, estimated: bool = True, unpriced: set | None = None
) -> float:
    """Dollars of a `use` (spend_scan: per model id or family [input, write 5 m, write 1 h, read, output recorded,
    output estimated, requests]), each key at its prices (`price_of`); output estimated unless `estimated` is False.
    A key with no price adds nothing, and goes into `unpriced` when one is given: its cost is unknown. Prompts past a
    long-context threshold are not told apart (aggregated use)."""
    total = 0.0
    for key, row in (use or {}).items():
        r = price_of(prices, key)
        if r is None:
            if unpriced is not None:
                unpriced.add(key)
            continue
        out = row[5] if estimated else row[4]
        total += (
            row[0] * r["in"]
            + row[1] * r["cache_write_5m"]
            + row[2] * r["cache_write_1h"]
            + row[3] * r["cache_read"]
            + out * r["out"]
        ) / 1e6
    return total


def unpriced_text(keys: list[str]) -> str:
    """The models whose cost is unknown (`usd`'s unpriced), in words; "" for none."""
    if not keys:
        return ""
    return (
        f"Cost unknown for {', '.join(keys)}: the prices have no entry for "
        + ("it, so its usage is" if len(keys) == 1 else "them, so their usage is")
        + " not in the $ figures."
    )


# --- weeks, stats -----------------------------------------------------------------------------------------------------


def week_of(day: str) -> str:
    y, w, _ = dt.date.fromisoformat(day[:10]).isocalendar()
    return f"{y}-W{w:02d}"


def week_days(label: str) -> list[str]:
    y, w = label.split("-W")
    start = dt.date.fromisocalendar(int(y), int(w), 1)
    return [(start + dt.timedelta(days=i)).isoformat() for i in range(7)]


def _hours(a: str, b: str) -> float:
    f = "%Y-%m-%dT%H:%M:%S"
    return (
        dt.datetime.strptime(b[:19], f) - dt.datetime.strptime(a[:19], f)
    ).total_seconds() / 3600


def stats(values: list[float]) -> dict:
    """n, total, mean, median, p90, min, max of the values (None for each but n and total when empty)."""
    v = sorted(values)
    out = {"n": len(v), "total": sum(v)}
    if not v:
        return dict(out, mean=None, median=None, p90=None, min=None, max=None)
    p90 = statistics.quantiles(v, n=10, method="inclusive")[-1] if len(v) > 1 else v[0]
    return dict(
        out,
        mean=out["total"] / len(v),
        median=statistics.median(v),
        p90=p90,
        min=v[0],
        max=v[-1],
    )


def outliers(reviews: list[dict]) -> list[dict]:
    """The reviews above Q3 + 1.5 x IQR of their totals (four reviews at least), dearest first."""
    if len(reviews) < 4:
        return []
    q1, _, q3 = statistics.quantiles(
        [p["total"] for p in reviews], n=4, method="inclusive"
    )
    lim = q3 + 1.5 * (q3 - q1)
    return sorted((p for p in reviews if p["total"] > lim), key=lambda p: -p["total"])


def driver(p: dict) -> str:
    """The part that cost the review most."""
    return PART_LABEL[max(PARTS, key=lambda k: p["parts"][k])]


# --- the data ---------------------------------------------------------------------------------------------------------


def paid_reviews(store) -> list[dict]:
    """Every paid review, oldest first: each post in the ledger (its note gives the kind), and each review kept
    unposted while review posting is off (its line of state/review_cost.jsonl, `unposted`: it has no ledger entry; its
    round gives the kind). {pr, head, ts, day, week, kind, round, unposted, old (the head of our review it follows up
    on: a post our previous post, a kept review our previous review of either kind), parts, total, runs}.
    """
    costs = store.read_jsonl("review_cost.jsonl", lenient=True)
    rounds = {
        (r.get("pr"), r.get("head")): r.get("round")
        for r in costs
        if not r.get("unposted")
    }
    found = []
    for r in store.ledger():
        if r.get("kind") != "post":
            continue
        note = (r.get("note") or "").lower()
        kind = (
            "ci-update"
            if note == "ci-update"
            else "first_round" if note.startswith("first") else "followup"
        )
        rnd = rounds.get((r["pr"], r.get("head"))) or ""
        found.append((r["ts"], r["pr"], r.get("head") or "", kind, rnd, False))
    for r in costs:
        if r.get("unposted") and r.get("pr") and r.get("at"):
            rnd = r.get("round") or ""
            kind = "first_round" if rnd.startswith("fr") else "followup"
            found.append((r["at"], r["pr"], r.get("head") or "", kind, rnd, True))
    out, posted, reviewed = [], {}, {}
    for ts, pr, head, kind, rnd, kept in sorted(found, key=lambda f: f[0]):
        prev = reviewed if kept else posted
        out.append(
            {
                "pr": pr,
                "head": head,
                "ts": ts,
                "day": ts[:10],
                "week": week_of(ts),
                "kind": kind,
                "round": rnd,
                "unposted": kept,
                "old": "" if kind == "first_round" else prev.get(pr, ""),
                "parts": {k: 0.0 for k in PARTS},
                "total": 0.0,
                "runs": [],
                "size": None,
                "impact": None,
            }
        )
        reviewed[pr] = head
        if not kept:
            posted[pr] = head
    return out


def _review_after(reviews: list[dict], t: str, hours: float) -> dict | None:
    for p in reviews:
        if p["ts"][:19] >= t[:19] and _hours(t, p["ts"]) <= hours:
            return p
    return None


def _review_before(reviews: list[dict], t: str, hours: float) -> dict | None:
    for p in reversed(reviews):
        if p["ts"][:19] < t[:19] and _hours(p["ts"], t) <= hours:
            return p
    return None


def attribute(
    prices: dict, reviews: list[dict], runs: list[dict], turns: list[dict]
) -> dict:
    """Book the runs and the turns: each review's parts and total (in place), and per day the overhead
    ({orchestration, unfollowed: the runs no review followed, notice}) and the work that is not review spend
    ({experiment, other}), both by the day spent; and the models whose cost is unknown (`unpriced`, `usd`). What was
    spent before the first review is left out."""
    first = reviews[0]["ts"][:19] if reviews else "9999"
    unpriced: set = set()
    by_pr = defaultdict(list)
    for p in reviews:
        by_pr[p["pr"]].append(p)
    overhead = defaultdict(lambda: defaultdict(float))
    other = defaultdict(lambda: defaultdict(float))
    unfollowed = []
    for r in sorted(runs, key=lambda r: r["start"]):
        day = r["start"][:10]
        cost = usd(prices, r["use"], unpriced=unpriced)
        if r["role"] in spend_scan.REVIEW_ROLES:
            p = (
                _review_after(by_pr.get(r["pr"], []), r["start"], RUN_WINDOW_H)
                if r["pr"]
                else None
            )
            if p is None:
                if r["start"] >= first:
                    overhead[day]["unfollowed"] += cost
                    unfollowed.append(dict(r, usd=cost))
                continue
            part = (
                "lane"
                if r["role"] == "quality"
                else (
                    "verifier2"
                    if any(x["role"] in spend_scan.VERIFIERS for x in p["runs"])
                    else "verifier"
                )
            )
            p["parts"][part] += cost
            p["runs"].append(
                {"id": r["id"], "role": r["role"], "round": r["round"], "usd": cost}
            )
            if not p["round"] and r["round"]:
                p["round"] = r["round"]
        elif r["start"] >= first:
            (overhead if r["role"] == "notice" else other)[day][r["role"]] += cost
    for t in turns:
        day, before = t["t"][:10], t["t"] < first
        cost = usd(prices, t["use"], unpriced=unpriced)
        roles = set(t.get("roles") or [])
        prs = sorted({int(n) for n in t.get("nums") or []} & set(by_pr))
        if "experiment" in roles and not roles.intersection(spend_scan.REVIEW_ROLES):
            if not before:
                other[day]["experiment"] += cost
            continue
        if 1 <= len(prs) <= TURN_MAX_PRS:
            share = cost / len(prs)
            for n in prs:
                p = _review_after(by_pr[n], t["t"], TURN_AHEAD_H) or _review_before(
                    by_pr[n], t["t"], TURN_BEHIND_H
                )
                if p is not None:
                    p["parts"]["orchestration"] += share
                    p["alone"] = p.get("alone") or len(prs) == 1
                elif not before:
                    overhead[day]["orchestration"] += share
            continue
        if before:
            continue
        if roles and roles <= {"other", "experiment"}:
            other[day]["other"] += cost
            continue
        overhead[day]["orchestration"] += cost
    for p in reviews:
        p["total"] = sum(p["parts"].values())
        if not p["round"]:
            p["round"] = "fr" if p["kind"] == "first_round" else "fu"
        p["suspect"] = suspect(p)
    return {
        "overhead": overhead,
        "other": other,
        "unfollowed": unfollowed,
        "unpriced": sorted(unpriced),
    }


def suspect(p: dict) -> bool:
    """A review whose orchestration is more than its verifier and lane together while its PR is never the only PR of a
    turn booked to it: the mark of turns that name the PR without acting on it (another PR's script renumbered from
    this PR's, for example)."""
    pp = p["parts"]
    return pp["orchestration"] > pp["verifier"] + pp["verifier2"] + pp[
        "lane"
    ] and not p.get("alone")


SUSPECT_NOTE = "orchestration over verifier + lane, from shared turns only"


def experiment_ledgers(cfg: dict) -> dict:
    """Dollars per day in the experiment ledgers (<scratch>/*/spend.jsonl: exact, from each request's final usage)."""
    out = defaultdict(float)
    for path in sorted(
        pathlib.Path(configured_paths(cfg)["scratch"]).glob("*/spend.jsonl")
    ):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                r = json.loads(line)
                out[str(r["ts"])[:10]] += float(r.get("usd") or 0)
            except (ValueError, KeyError, TypeError):
                continue
    return out


def covered(store) -> dict:
    """Heads recorded as covered by our review, per day: {"merge_only": n, "rebase": n} (no Claude spend)."""
    out = defaultdict(lambda: {"merge_only": 0, "rebase": 0})
    for ts, how in report.equivalent_clears(store).values():
        out[ts[:10]]["merge_only" if how == report.MERGE_ONLY else "rebase"] += 1
    return out


# --- reviewed size and impact (git only, cached) ------------------------------------------------------------------------


def _develop(repo: str) -> str:
    try:
        r = subprocess.run(
            ["git", "-C", repo, "rev-parse", "refs/remotes/origin/develop"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _size_key(p: dict) -> str:
    return f"{p['head']}:{p['old']}"


def load_sizes(store) -> dict:
    try:
        return json.loads((store.dir / SIZES).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def measure(
    cfg: dict, store, reviews: list[dict], budget_s: float | None = None, impact_for=()
) -> int:
    """Each review's reviewed size (and, for the reviews in `impact_for`, its impact) from the cache (SIZES, written
    whole at the mode of the files the monitor and the writer share), measuring what is missing within `budget_s`
    seconds (None: no limit). Returns how many are still unmeasured.
    """
    from .hybrid import impact, size

    cache = load_sizes(store)
    repo = (cfg.get("paths") or {}).get("repo") or ""
    dev, t0, changed, left = None, time.monotonic(), False, 0
    hot = {
        x
        for k in (cfg.get("hot_spots") or {})
        for x in (
            cfg["hot_spots"][k]
            if isinstance(cfg["hot_spots"][k], list)
            else [cfg["hot_spots"][k]]
        )
    }
    want_impact = {id(p) for p in impact_for}
    for p in reviews:
        key = _size_key(p)
        ent = cache.get(key) or {}
        need_impact = id(p) in want_impact and "impact" not in ent
        if ("tier" not in ent or need_impact) and repo:
            if budget_s is not None and time.monotonic() - t0 > budget_s:
                left += 1
            else:
                dev = dev if dev is not None else _develop(repo)
                try:
                    measured = size.measure(repo, p["head"], dev, p["old"])
                    rec = measured[0]
                    ent = {
                        "eff": rec.get("eff"),
                        "tier": rec.get("tier") if rec.get("src") != "unknown" else "?",
                    }
                    if need_impact and ent["tier"] != "?":
                        sc = impact.score_round(
                            repo,
                            p["head"],
                            dev,
                            p["old"],
                            hot=hot,
                            measured=measured,
                            folder=paths.repository_folder(cfg),
                        )
                        ent["impact"] = (
                            f"{sc['tier']} ({sc['score']})" if sc.get("ok") else "?"
                        )
                    elif need_impact:
                        ent["impact"] = "?"
                except (
                    Exception
                ) as exc:  # noqa: BLE001 - a size is advisory: never fail the report on it
                    ent = {"eff": None, "tier": "?", "error": str(exc)[:120]}
                cache[key] = ent
                changed = True
        p["size"] = {"eff": ent.get("eff"), "tier": ent.get("tier") or "?"}
        p["impact"] = ent.get("impact")
    if changed:
        whole_file.write(
            store.dir / SIZES,
            json.dumps(cache, separators=(",", ":")),
            paths.worker_module("perms").SHARED_FILE_MODE,
        )
    return left


def load_history(store) -> dict | None:
    """state/spend_history.json (HISTORY), None when there is none or it is unreadable."""
    try:
        h = json.loads((store.dir / HISTORY).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return h if isinstance(h, dict) and h.get("until") else None


def with_history(
    hist: dict | None, runs: list[dict], turns: list[dict]
) -> tuple[list[dict], list[dict]]:
    """The runs and turns to book: the history's (as it holds them) before its `until`, this machine's transcripts'
    from `until` on; this machine's alone without a history."""
    if not hist:
        return runs, turns
    until = str(hist["until"])[:19]
    return (
        [r for r in hist.get("runs") or [] if r["start"][:19] < until]
        + [r for r in runs if r["start"][:19] >= until],
        [t for t in hist.get("turns") or [] if t["t"][:19] < until]
        + [t for t in turns if t["t"][:19] >= until],
    )


def collect(
    cfg: dict, store, max_bytes: int | None = None, size_budget_s: float | None = None
) -> dict:
    """Everything the reports need: the paid reviews with their parts and sizes, overhead and other work per day,
    covered heads per day, the experiment ledgers per day, the models whose cost is unknown, the requests on another
    model family than their pass asked for (`off_family`), and what is still unread or unmeasured.
    """
    prices = price_table(cfg)
    root = projects_root(cfg)
    sc = spend_scan.scan(root, store, max_bytes, asked_ids(cfg))
    runs, turns = with_history(load_history(store), sc["runs"], sc["turns"])
    reviews = paid_reviews(store)
    booked = attribute(prices, reviews, runs, turns)
    t0 = time.monotonic()
    unmeasured = measure(cfg, store, reviews, size_budget_s)
    data = {
        "at": now(),
        "reviews": reviews,
        "overhead": booked["overhead"],
        "other": booked["other"],
        "unfollowed": booked["unfollowed"],
        "unpriced": booked["unpriced"],
        "off_family": off_family(runs, turns),
        "covered": covered(store),
        "ledgers": experiment_ledgers(cfg),
        "unread_bytes": sc["left"],
        "unmeasured": unmeasured,
        "first": reviews[0]["day"] if reviews else None,
    }
    # The outliers of every day and week get their impact (git only, cached); within what is left of the budget.
    days = all_days(data)
    targets = {id(p): p for d in days for p in period(data, [d])["outliers"]}
    for w in sorted({week_of(d) for d in days}):
        targets.update(
            {
                id(p): p
                for p in period(data, [d for d in week_days(w) if d in set(days)])[
                    "outliers"
                ]
            }
        )
    left = (
        None
        if size_budget_s is None
        else max(size_budget_s - (time.monotonic() - t0), 0)
    )
    data["unmeasured"] += measure(
        cfg, store, list(targets.values()), left, impact_for=list(targets.values())
    )
    return data


# --- the model guard ---------------------------------------------------------------------------------------------------


def off_family(runs: list[dict], turns: list[dict]) -> list[dict]:
    """The requests of each run and turn on another model family than their pass asked for (spend_scan's `off`):
    {t, asked, reported, calls}, oldest first."""
    out = []
    for t, off in [(r.get("end") or r["start"], r.get("off")) for r in runs] + [
        (x["t"], x.get("off")) for x in turns
    ]:
        for asked, seen in (off or {}).items():
            out += [
                {"t": t[:19], "asked": asked, "reported": reported, "calls": calls}
                for reported, calls in seen.items()
            ]
    return sorted(out, key=lambda o: o["t"])


def guard(data: dict, today: str | None = None) -> list[dict]:
    """The model guard's findings (`off_family`) of the last GUARD_DAYS UTC days up to TODAY: per id asked for and
    model reported, {asked, reported, calls, last}, the latest first."""
    today = (today or now())[:10]
    since = dt.date.fromisoformat(today) - dt.timedelta(days=GUARD_DAYS - 1)
    found: dict = {}
    for o in data.get("off_family") or []:
        if not since.isoformat() <= o["t"][:10] <= today:
            continue
        f = found.setdefault(
            (o["asked"], o["reported"]),
            {"asked": o["asked"], "reported": o["reported"], "calls": 0, "last": ""},
        )
        f["calls"] += o["calls"]
        f["last"] = max(f["last"], o["t"])
    return sorted(found.values(), key=lambda f: f["last"], reverse=True)


def guard_text(found: list[dict]) -> str:
    """The model guard's findings (`guard`) in words; "" for none."""
    if not found:
        return ""

    def calls(n: int) -> str:
        return f"{n} call{'' if n == 1 else 's'}"

    return (
        f"{calls(sum(f['calls'] for f in found))} of the last {GUARD_DAYS} days ran on another model family than "
        "the one asked for: "
        + "; ".join(
            f"{f['reported']} answered for {f['asked']} ({calls(f['calls'])}, the last at {f['last']}Z)"
            for f in found
        )
    )


def guard_lines(data: dict) -> list[str]:
    """The report's model guard: a WARN with its findings (`guard`), nothing when there are none."""
    text = guard_text(guard(data))
    return ["### Model guard", "", f"**WARN:** {text}.", ""] if text else []


# --- periods ----------------------------------------------------------------------------------------------------------


def period(data: dict, days: list[str]) -> dict:
    """One period's figures: reviews (stats of the per-review totals), covered heads, review $, overhead $, total $."""
    ds = set(days)
    reviews = [p for p in data["reviews"] if p["day"] in ds]
    ov = {
        k: sum(data["overhead"].get(d, {}).get(k, 0.0) for d in ds)
        for k in ("orchestration", "unfollowed", "notice")
    }
    cov = {
        k: sum(data["covered"].get(d, {}).get(k, 0) for d in ds)
        for k in ("merge_only", "rebase")
    }
    st = stats([p["total"] for p in reviews])
    overhead = sum(ov.values())
    return {
        "reviews": reviews,
        "stats": st,
        "overhead": overhead,
        "overhead_parts": ov,
        "covered": cov,
        "total": st["total"] + overhead,
        "outliers": outliers(reviews),
    }


def all_days(data: dict, until: str | None = None) -> list[str]:
    """Every day from the first review's to `until` (default: today), UTC."""
    if not data.get("first"):
        return []
    a, b = dt.date.fromisoformat(data["first"]), dt.date.fromisoformat(
        (until or now())[:10]
    )
    return [(a + dt.timedelta(days=i)).isoformat() for i in range((b - a).days + 1)]


def cumulative(data: dict, until: str | None = None) -> dict:
    """The running total of review and overhead dollars at the end of each day, from the first day with data."""
    run, out = 0.0, {}
    for d in all_days(data, until):
        run += period(data, [d])["total"]
        out[d] = run
    return out


def experiments(data: dict, days: list[str]) -> dict:
    ds = set(days)
    return {
        "ledgers": sum(v for d, v in data["ledgers"].items() if d in ds),
        "runs": sum(data["other"].get(d, {}).get("experiment", 0.0) for d in ds),
        "other": sum(data["other"].get(d, {}).get("other", 0.0) for d in ds),
    }


# --- markdown ---------------------------------------------------------------------------------------------------------


def money(v) -> str:
    return "—" if v is None else f"{v:,.2f}"


def _covered(c: dict) -> str:
    return f"{c['merge_only']} / {c['rebase']}"


ROW_HEAD = [
    "| {} | Reviews | Covered (merge-only / rebase) | Reviews $ | Overhead $ | Total $ | Mean | "
    "Median | p90 | Min | Max | Outliers |",
    "|---|" + "---:|" * 11,
]


def period_row(label: str, f: dict, cum: float | None) -> str:
    s = f["stats"]
    return (
        f"| {label} | {s['n']} | {_covered(f['covered'])} | {money(s['total'])} | {money(f['overhead'])} | "
        f"{money(f['total'])} | {money(s['mean'])} | {money(s['median'])} | {money(s['p90'])} | "
        f"{money(s['min'])} | {money(s['max'])} | {len(f['outliers'])} |"
    )


def days_table(data: dict, days: list[str], cum: dict) -> list[str]:
    out = [ROW_HEAD[0].format("Day"), ROW_HEAD[1]]
    return out + [period_row(d, period(data, [d]), cum.get(d)) for d in days]


def weeks_table(data: dict, weeks: list[str], cum: dict) -> list[str]:
    out = [ROW_HEAD[0].format("Week"), ROW_HEAD[1]]
    for w in weeks:
        days = [d for d in week_days(w) if d in cum]
        out.append(period_row(w, period(data, days), cum[days[-1]] if days else None))
    return out


def outliers_table(f: dict) -> list[str]:
    if not f["outliers"]:
        return [
            (
                "No outliers (above Q3 + 1.5 × IQR)."
                if f["stats"]["n"] >= 4
                else "Outliers need four reviews at least."
            )
        ]
    out = [
        "| PR | Round | Kind | $ | Size tier | Impact | Driven by |",
        "|---|---|---|---:|---|---|---|",
    ]
    for p in f["outliers"]:
        out.append(
            f"| #{p['pr']} | {p['round']} | {KIND_LABEL[p['kind']]} | {money(p['total'])} | "
            f"{(p['size'] or {}).get('tier', '?')} | {p.get('impact') or '?'} | {driver(p)}"
            f"{' (suspect: ' + SUSPECT_NOTE + ')' if p.get('suspect') else ''} |"
        )
    return out


def suspects_lines(f: dict) -> list[str]:
    """The selection's reviews whose orchestration may belong to other PRs (suspect), dearest first."""
    sus = sorted(
        (p for p in f["reviews"] if p.get("suspect")),
        key=lambda p: -p["parts"]["orchestration"],
    )
    if not sus:
        return []
    return [
        f"Suspect attribution ({SUSPECT_NOTE}; check its turns): "
        + ", ".join(
            f"#{p['pr']} {p['round']} orchestration {money(p['parts']['orchestration'])} vs verifier + lane "
            f"{money(p['parts']['verifier'] + p['parts']['verifier2'] + p['parts']['lane'])}"
            for p in sus[:8]
        )
        + (f" (+{len(sus) - 8} more)" if len(sus) > 8 else "")
    ]


def _group_rows(groups: list[tuple[str, list[float]]], head: str) -> list[str]:
    out = [
        f"| {head} | Reviews | Total $ | Mean | Median | p90 | Max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, vals in groups:
        s = stats(vals)
        out.append(
            f"| {name} | {s['n']} | {money(s['total'])} | {money(s['mean'])} | {money(s['median'])} | "
            f"{money(s['p90'])} | {money(s['max'])} |"
        )
    return out


def breakdown(f: dict, by: str) -> list[str]:
    reviews = f["reviews"]
    if by == "part":
        out = [
            "| Part | Total $ | Share | Mean per review | Reviews with it |",
            "|---|---:|---:|---:|---:|",
        ]
        tot = sum(p["total"] for p in reviews) or 1.0
        for k in PARTS:
            vals = [p["parts"][k] for p in reviews]
            out.append(
                f"| {PART_LABEL[k]} | {money(sum(vals))} | {100 * sum(vals) / tot:.0f}% | "
                f"{money(sum(vals) / len(reviews) if reviews else None)} | {sum(1 for v in vals if v > 0)} |"
            )
        ov = f["overhead_parts"]
        out.append(
            f"| overhead (not in the reviews' $): unattributed orchestration {money(ov['orchestration'])}, "
            f"runs no review followed {money(ov['unfollowed'])}, overlap notices {money(ov['notice'])} | "
            f"{money(f['overhead'])} | | | |"
        )
        return out
    if by == "kind":
        return _group_rows(
            [
                (KIND_LABEL[k], [p["total"] for p in reviews if p["kind"] == k])
                for k in KIND_LABEL
            ],
            "Kind",
        )
    if by == "size":
        return _group_rows(
            [
                (
                    t,
                    [
                        p["total"]
                        for p in reviews
                        if (p["size"] or {}).get("tier", "?") == t
                    ],
                )
                for t in TIERS
            ],
            "Reviewed diff (effective lines)",
        )
    prs = defaultdict(list)
    for p in reviews:
        prs[p["pr"]].append(p["total"])
    return _group_rows(
        [(f"#{n}", v) for n, v in sorted(prs.items(), key=lambda kv: -sum(kv[1]))], "PR"
    )


def reviews_table(f: dict) -> list[str]:
    parts = " | ".join(PART_LABEL[k].capitalize() for k in PARTS)
    out = [
        f"| Posted or kept | PR | Round | Kind | Size tier | {parts} | Total $ |",
        "|---|---|---|---|---|" + "---:|" * (len(PARTS) + 1),
    ]
    for p in sorted(f["reviews"], key=lambda p: p["ts"]):
        # A review kept unposted (review posting off) shows the time it was kept.
        out.append(
            f"| {p['ts'][11:16]}Z{' (kept unposted)' if p.get('unposted') else ''} | #{p['pr']} | {p['round']} | "
            f"{KIND_LABEL[p['kind']]} | "
            f"{(p['size'] or {}).get('tier', '?')} | "
            + " | ".join(money(p["parts"][k]) for k in PARTS)
            + f"{' (suspect)' if p.get('suspect') else ''} | {money(p['total'])} |"
        )
    return out


def experiments_table(data: dict, days: list[str]) -> list[str]:
    out = [
        "| Day | Experiment ledgers $ (exact) | Experiment runs $ | Other tasks $ | Total $ |",
        "|---|---:|---:|---:|---:|",
    ]
    for d in days:
        e = experiments(data, [d])
        if any(e.values()):
            out.append(
                f"| {d} | {money(e['ledgers'])} | {money(e['runs'])} | {money(e['other'])} | "
                f"{money(sum(e.values()))} |"
            )
    return out


def legend(data: dict) -> list[str]:
    out = [
        f"_Reviews: each review posted, or kept while review posting is off, on the day it was posted or kept. $: "
        f"config prices; transcript figures use estimated output (cost_fp.py's method). Covered heads cost no Claude "
        f"spend and are not in the averages. Overhead: unattributed orchestration, verifier and code quality pass "
        f"runs no review followed, overlap notices. Outliers: above Q3 + 1.5 × IQR. Data from "
        f"{data.get('first') or '—'} (UTC); refreshed {data['at']}._"
    ]
    if data.get("unread_bytes"):
        out.append(
            f"_{data['unread_bytes'] / 1e6:.0f} MB of transcripts not read yet (the next pass reads them)._"
        )
    if data.get("unmeasured"):
        out.append(
            f"_{data['unmeasured']} reviewed diffs not measured yet (size tier ?)._"
        )
    if data.get("unpriced"):
        out.append(f"_{unpriced_text(data['unpriced'])}_")
    return out


def report_md(
    cfg: dict,
    data: dict,
    day: str | None = None,
    week: str | None = None,
    since: str | None = None,
    by: str = "part",
    show_experiments: bool = False,
) -> str:
    """`rk spend`: the days and weeks (or one day, one week), then the selection's outliers and breakdown."""
    cum = cumulative(data)
    days = all_days(data)
    if day:
        sel, title, label = [day], f"## Review spend — {day} (UTC)", day
    elif week:
        sel = [d for d in week_days(week) if d in cum]
        title, label = f"## Review spend — {week} (UTC)", week
    else:
        sel = [d for d in days if not since or d >= since]
        title, label = (
            "## Review spend" + (f" since {since}" if since else "") + " (UTC)",
            "selection",
        )
    weeks = sorted({week_of(d) for d in sel})
    f = period(data, sel)
    out = [title, ""] + guard_lines(data) + days_table(data, sel, cum) + [""]
    if not day:
        out += weeks_table(data, weeks, cum) + [""]
    sus = suspects_lines(f)
    out += (
        [f"### Outliers ({label})", ""]
        + outliers_table(f)
        + ([""] + sus if sus else [])
    )
    out += ["", f"### By {by} ({label})", ""] + breakdown(f, by)
    if day:
        out += ["", "### Reviews", ""] + reviews_table(f)
    if show_experiments:
        out += [
            "",
            "### Experiments and other tasks (not review spend)",
            "",
        ] + experiments_table(data, sel)
    return "\n".join(out + [""] + legend(data)) + "\n"


def day_file(data: dict, day: str) -> str:
    cum = cumulative(data, day)
    f = period(data, [day])
    out = [f"# Review spend — {day} (UTC)", ""] + days_table(data, [day], cum) + [""]
    for by in ("part", "kind", "size"):
        out += [f"## By {by}", ""] + breakdown(f, by) + [""]
    out += (
        ["## Outliers", ""]
        + outliers_table(f)
        + ["", "## Reviews", ""]
        + reviews_table(f)
        + [""]
    )
    return "\n".join(out + legend(data)) + "\n"


def week_file(data: dict, week: str, until: str) -> str:
    cum = cumulative(data, until)
    days = [d for d in week_days(week) if d in cum]
    f = period(data, days)
    out = (
        [f"# Review spend — {week} (UTC)", ""]
        + days_table(data, days, cum)
        + [""]
        + weeks_table(data, [week], cum)
    )
    out += [""]
    for by in ("part", "kind", "size"):
        out += [f"## By {by}", ""] + breakdown(f, by) + [""]
    out += (
        ["## Outliers", ""]
        + outliers_table(f)
        + ["", "## PRs by spend", ""]
        + breakdown(f, "pr")[:17]
        + [""]
    )
    return "\n".join(out + legend(data)) + "\n"


def write_files(cfg: dict, data: dict, days: list[str]) -> list[pathlib.Path]:
    """spend_<day>.md for each day and spend_<week>.md for each of their weeks (through its last day given), in
    paths.spend_dir."""
    out = pathlib.Path(configured_paths(cfg)["spend_dir"])
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for d in days:
        paths.append(out / f"spend_{d}.md")
        paths[-1].write_text(day_file(data, d), encoding="utf-8")
    for w in sorted({week_of(d) for d in days}):
        paths.append(out / f"spend_{w}.md")
        paths[-1].write_text(
            week_file(data, w, max(d for d in days if week_of(d) == w)),
            encoding="utf-8",
        )
    return paths


def weekly_section(cfg: dict, store, label: str) -> list[str]:
    """The weekly report's section: the week's days and total, by part, kind and size, the outliers."""
    try:
        data = collect(cfg, store, TICK_MAX_BYTES, TICK_SIZE_BUDGET_S)
    except (OSError, ValueError) as exc:
        return ["## Review spend", "", f"Not available: {exc}"]
    days = week_days(label)
    cum = cumulative(data, days[-1])
    days = [d for d in days if d in cum]
    f = period(data, days)
    out = (
        ["## Review spend", ""]
        + days_table(data, days, cum)
        + [""]
        + weeks_table(data, [label], cum)
        + [""]
    )
    for by in ("part", "kind", "size"):
        out += breakdown(f, by) + [""]
    return out + outliers_table(f) + [""] + legend(data)


# --- the card (both dashboards) and the tick ---------------------------------------------------------------------------


def _brief(f: dict, cum: float | None) -> dict:
    s = f["stats"]
    return {
        "reviews": s["n"],
        "total": f["total"],
        "cumulative": cum,
        "mean": s["mean"],
        "median": s["median"],
        "p90": s["p90"],
        "max": s["max"],
    }


def card_data(data: dict, today: str | None = None) -> dict:
    """What the dashboards' card shows: today and this week so far, the last 7 days, this week's outliers, the
    models whose cost is unknown; and the model guard's findings (`guard`), which the admin dashboard and rk doctor
    show."""
    today = (today or now())[:10]
    cum = cumulative(data, today)
    wk = week_of(today)
    wdays = [d for d in week_days(wk) if d in cum]
    week = period(data, wdays)
    last7 = [d for d in all_days(data, today)][-7:]
    return {
        "at": data["at"],
        "today": dict(_brief(period(data, [today]), cum.get(today)), day=today),
        "week": dict(_brief(week, cum.get(today)), week=wk),
        "days": [dict(_brief(period(data, [d]), cum.get(d)), day=d) for d in last7],
        "outliers": [
            {"pr": p["pr"], "round": p["round"], "usd": p["total"], "driver": driver(p)}
            for p in week["outliers"]
        ],
        "unpriced": list(data.get("unpriced") or []),
        "off_family": guard(data, today),
    }


def save_card(store, card: dict) -> None:
    """The card CARD as SUMMARY in the state folder, written whole (whole_file) at the mode of the files the monitor
    and the writer share (worker/perms.py)."""
    whole_file.write(
        store.dir / SUMMARY,
        json.dumps(card),
        paths.worker_module("perms").SHARED_FILE_MODE,
    )


def load_card(store) -> dict | None:
    return read_card(store.dir)


def review_mean(card: dict | None) -> float | None:
    """The mean cost of a review over the card's last 7 days (each day's mean weighted by its reviews), in US$; None
    without a card or a review in those days."""
    days = [
        d
        for d in (card or {}).get("days") or []
        if d.get("reviews") and d.get("mean") is not None
    ]
    n = sum(d["reviews"] for d in days)
    return sum(d["mean"] * d["reviews"] for d in days) / n if n else None


def read_card(state) -> dict | None:
    """The card saved in the state folder STATE (SUMMARY); None when there is none or it does not read."""
    try:
        card = json.loads((pathlib.Path(state) / SUMMARY).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return card if isinstance(card, dict) else None


CARD_COLS = ("Reviews", "Total $", "Mean", "Median", "p90", "Max")


def card_cells(r: dict) -> list[str]:
    return [str(r["reviews"])] + [
        money(r[k]) for k in ("total", "mean", "median", "p90", "max")
    ]


def card_md(card: dict | None) -> list[str]:
    """The md dashboard's card (its last section)."""
    out = ["## Review spend", ""]
    if not card:
        return out + ["Not computed yet (`rk spend`)."]
    out += [
        "| | " + " | ".join(CARD_COLS) + " |",
        "|---|" + "---:|" * len(CARD_COLS),
        f"| Today ({card['today']['day']}) | "
        + " | ".join(card_cells(card["today"]))
        + " |",
        f"| This week ({card['week']['week']}) | "
        + " | ".join(card_cells(card["week"]))
        + " |",
        "",
        "| Day | " + " | ".join(CARD_COLS) + " |",
        "|---|" + "---:|" * len(CARD_COLS),
    ]
    out += [f"| {r['day']} | " + " | ".join(card_cells(r)) + " |" for r in card["days"]]
    out += [
        "",
        f"**Outliers this week:** {outliers_text(card)}",
        "",
        f"_{card_note(card)}_",
    ]
    return out


def outliers_text(card: dict) -> str:
    return (
        ", ".join(
            f"#{o['pr']} {o['round']} ${money(o['usd'])} ({o['driver']})"
            for o in card["outliers"]
        )
        or "none"
    )


def card_note(card: dict) -> str:
    unknown = unpriced_text(card.get("unpriced") or [])
    return (
        f"Reviews: posted, or kept while review posting is off. Total $: reviews plus overhead (unattributed "
        f"orchestration, runs no review followed, overlap notices); output estimated; refreshed {card['at']}."
        + (f" {unknown}" if unknown else "")
    )


def files_due(day_now: str, done: str | None) -> str | None:
    """The day whose files are due: yesterday (UTC), once."""
    y = (dt.date.fromisoformat(day_now[:10]) - dt.timedelta(days=1)).isoformat()
    return None if done == y else y


def tick(cfg: dict, st: dict, store) -> list[str]:
    """The tick's step: refresh the card at most every REFRESH_MIN minutes (cached transcripts, a byte and a size
    budget), and once a day write yesterday's files. Advisory: a failure is a line, never raised.
    """
    if not on(cfg):
        return []
    sp = st.setdefault("spend", {})
    t = now()
    due = files_due(t, sp.get("files"))
    if (
        sp.get("at")
        and minutes_since(sp["at"]) < REFRESH_MIN
        and (
            due is None
            or minutes_since(due + "T00:00:00Z") < 24 * 60 + FILES_AFTER_H * 60
        )
    ):
        return []
    try:
        data = collect(cfg, store, TICK_MAX_BYTES, TICK_SIZE_BUDGET_S)
        # A partly read cache (a scan VERSION bump, a backlog of transcripts) prices the unread turns at 0: keep the
        # last full card (up to 2 hours old) until the reading catches up, and refresh again next tick.
        last = load_card(store)
        if data["unread_bytes"] and last and minutes_since(last["at"]) < 120:
            return [
                f"(spend card kept: {data['unread_bytes'] // 1_000_000} MB of transcripts still to read)"
            ]
        save_card(store, card_data(data, t))
        sp["at"] = t
        lines = []
        if (
            due
            and minutes_since(due + "T00:00:00Z") >= 24 * 60 + FILES_AFTER_H * 60
            and not data["unread_bytes"]
        ):
            paths = write_files(cfg, data, [due])
            sp["files"] = due
            lines.append(
                f"(spend files for {due} written: {', '.join(str(p) for p in paths)})"
            )
        return lines
    except (
        Exception
    ) as exc:  # noqa: BLE001 - spend is advisory: never fail a tick on it
        return [f"(spend not refreshed: {str(exc)[:200]})"]


@idle.timer("spend files", again=True)
def _files_due(cfg: dict, st: dict, store):
    """When yesterday's spend files fall due (FILES_AFTER_H after the UTC day ends), until they are written."""
    if not on(cfg):
        return
    t = now()
    due = files_due(t, (st.get("spend") or {}).get("files"))
    if due:
        at = dt.datetime.fromisoformat(due) + dt.timedelta(days=1, hours=FILES_AFTER_H)
        yield "spend files", at.strftime("%Y-%m-%dT%H:%M:%SZ")


def cli(cfg: dict, store, args) -> str:
    """`rk spend`: refresh (no budget: everything unread is read, every reviewed diff measured), save the card, print
    the report; --write writes the selection's files."""
    data = collect(cfg, store)
    save_card(store, card_data(data))
    text = report_md(
        cfg,
        data,
        day=args.day,
        week=args.week,
        since=args.since,
        by=args.by,
        show_experiments=args.experiments,
    )
    if args.write:
        days = (
            [args.day]
            if args.day
            else (
                week_days(args.week)
                if args.week
                else [d for d in all_days(data) if not args.since or d >= args.since]
            )
        )
        days = [
            d for d in days if data.get("first") and data["first"] <= d < now()[:10]
        ]
        paths = write_files(cfg, data, days)
        text += (
            f"\n(written to {configured_paths(cfg)['spend_dir']}: "
            f"{', '.join(p.name for p in paths) or 'nothing: no complete day selected'})\n"
        )
    return text
