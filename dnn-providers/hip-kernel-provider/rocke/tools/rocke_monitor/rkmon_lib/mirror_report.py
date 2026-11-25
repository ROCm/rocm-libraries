# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`rk mirror-report --live-state DIR --mirror DIR [--live-claude DIR] [--json] [--out FILE]`: the A/B comparison of a
live monitor and its mirror (rkmon_lib/mirror.py), review by review. Reads only; needs no configuration.

The live side: its state folder's review_cost.jsonl (one line per post through `rk post --premises`: the PR, round,
head, the post's copy, the draft's tokens and minutes, when the review started and when it was posted) and its feed
(mirror_feed.jsonl); its Claude tokens from its transcripts (`--live-claude`: the live monitor's Claude projects
folder; default: this machine's), counted as the mirror counts its own (transcripts.py: tokens processed by kind,
output, dollars at config prices' Claude Opus rates (rk spend's), ieq, the effort of each call): the PR's lanes from
the review's start to its post (verifier, quality, notice) and the monitor session's calls in that window. The mirror side: its out folder, one folder
per review written (pr<N>_<round>_<head10>/: post.md, notes.md, cost.json, made with the same count), and its drops
(drops.jsonl). A pair is one PR at one head on both sides, the live post the one of the feed line the mirror took.

For each pair: each side's header light, findings by severity, prior statuses, Code quality row and hygiene hits (the
post's public-repo checks in check-only mode, premises.hygiene_hits); Claude's dollars per lane and the session's, its
share (the session's dollars over the reviews that ran in the same window: on the live side its feed's reviews, each
to its post, or for an hour without one), the efforts the calls ran at; the drafting
models' tokens in, out and reasoning and requests; minutes. Then the totals per system and arm (mirror.py: arm A the
Claude's code quality pass, arm B the quality binding's), with a day's dollars at the live monitor's rate of reviews;
the judge's worklist of each pair's disagreements, with the files to read; the checks (a lane that wrote the 1-hour
cache, a lane with no transcript, a review that got the quality binding's code quality pass with no record of it); and
every review lost on either side, with why.
"""

import datetime as dt
import json
import pathlib
import re
import statistics

from . import transcripts
from .premises import hygiene_hits

LIGHTS = ("🔴", "🟠", "🟡", "🟢", "⚪")
DATA_ROWS = ("Bots / CI", "Our checks", "Code quality", "Threads", "Reviewers")
FINDING = re.compile(
    r"^\|\s*([NQ]\d+)\s*\|\s*(🔴|🟠|🟡|🟢|⚪)\s*\|\s*([^|]*)\|\s*(.*)\|\s*$"
)
WHERE = re.compile(r"`?([^`:|\s]+):(\d+)")
STATUS = re.compile(r"(🔴|🟠|🟡|🟢|⚪)\s*([A-Za-z][A-Za-z ,-]*)")
WORD = re.compile(r"[a-z_][a-z0-9_]{3,}")
LANES = ("verifier", "quality", "notice")
FIGURES = (
    "lanes_usd",
    "session_usd",
    "share_usd",
    "review_usd",
    "lanes_ieq",
    "model_tokens",
    "model_requests",
    "minutes",
)
RATE_DAYS = 7  # the live monitor's rate of reviews: its posts over the last days of review_cost.jsonl


def _light(text: str) -> str:
    m = re.search("|".join(LIGHTS), text or "")
    return m.group(0) if m else ""


def _json(path: pathlib.Path) -> dict:
    try:
        out = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return out if isinstance(out, dict) else {}


def _jsonl(path: pathlib.Path) -> list[dict]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    out = []
    for line in text.splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if isinstance(r, dict):
            out.append(r)
    return out


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse_post(text: str) -> dict:
    """A review post: its header light, prior rows ({key, light, status}), data rows' lights, findings ({id, light,
    file, line, problem}) and hygiene hits."""
    lines = text.split("\n")
    header = next((x for x in lines if x.startswith("## PR #")), "")
    out = {
        "light": _light(header.rsplit(" — ", 1)[-1]) if header else "",
        "priors": [],
        "data": {},
        "findings": [],
        "hygiene": [f"{why}: {h}" for why, hits in hygiene_hits(lines) for h in hits],
    }
    table = ""
    for line in lines:
        if line.startswith("| Prior item"):
            table = "prior"
            continue
        if line.startswith("| # | Sev"):
            table = "findings"
            continue
        if not line.startswith("|"):
            table = "" if not line.strip() else table
            continue
        if line.startswith("|---"):
            continue
        cells = _cells(line)
        if table == "prior" and len(cells) >= 2:
            key = cells[0]
            m = STATUS.search(cells[1])
            light, status = (
                (m.group(1), m.group(2).strip().split(",")[0].strip().upper())
                if m
                else ("", "")
            )
            if key in DATA_ROWS:
                out["data"][key] = light
            else:
                out["priors"].append({"key": key, "light": light, "status": status})
        elif table == "findings":
            m = FINDING.match(line)
            if m:
                w = WHERE.search(m.group(3))
                prob = re.search(r"\*\*(.+?)\*\*", m.group(4))
                out["findings"].append(
                    {
                        "id": m.group(1),
                        "light": m.group(2),
                        "file": w.group(1) if w else "",
                        "line": int(w.group(2)) if w else 0,
                        "problem": (prob.group(1) if prob else m.group(4))[:160],
                    }
                )
    return out


def _words(text: str) -> set:
    return set(WORD.findall((text or "").lower()))


def same_finding(a: dict, b: dict) -> bool:
    """Two findings are one when they sit at the same file within 5 lines, or their problems share half their words."""
    if (
        a["file"]
        and b["file"]
        and pathlib.PurePosixPath(a["file"]).name
        == pathlib.PurePosixPath(b["file"]).name
        and abs(a["line"] - b["line"]) <= 5
    ):
        return True
    wa, wb = _words(a["problem"]), _words(b["problem"])
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= 0.5


def _prior_key(key: str) -> str:
    return re.sub(r"\W+", " ", key.lower()).strip()[:40]


def disagreements(live: dict, mirror: dict) -> list[str]:
    """What the two posts of one pair say differently: the judge's worklist."""
    out = []
    if live["light"] != mirror["light"]:
        out.append(
            f"header light: live {live['light'] or '-'}, mirror {mirror['light'] or '-'}"
        )
    paired = []
    for f in live["findings"]:
        g = next(
            (g for g in mirror["findings"] if g not in paired and same_finding(f, g)),
            None,
        )
        if g is None:
            out.append(
                f"only live: {f['id']} {f['light']} `{f['file']}:{f['line']}` {f['problem']}"
            )
            continue
        paired.append(g)
        if f["light"] != g["light"]:
            out.append(
                f"severity: live {f['id']} {f['light']}, mirror {g['id']} {g['light']}: {f['problem']}"
            )
    for g in mirror["findings"]:
        if g not in paired:
            out.append(
                f"only mirror: {g['id']} {g['light']} `{g['file']}:{g['line']}` {g['problem']}"
            )
    theirs = {_prior_key(p["key"]): p for p in mirror["priors"]}
    for p in live["priors"]:
        q = theirs.get(_prior_key(p["key"]))
        if q and (p["light"], p["status"]) != (q["light"], q["status"]):
            out.append(
                f"prior \"{p['key'][:80]}\": live {p['light']} {p['status']}, mirror {q['light']} {q['status']}"
            )
    lq, mq = live["data"].get("Code quality", ""), mirror["data"].get(
        "Code quality", ""
    )
    if lq != mq:
        out.append(f"code quality: live {lq or '-'}, mirror {mq or '-'}")
    for side, post in (("live", live), ("mirror", mirror)):
        out += [f"hygiene ({side}): {h[:160]}" for h in post["hygiene"]]
    return out


def _counts(findings: list[dict]) -> str:
    return "·".join(
        str(sum(1 for f in findings if f["light"] == lt)) for lt in ("🔴", "🟠", "🟡")
    )


def _v(x) -> str:
    return "-" if x is None else str(x)


def _k(n) -> str:
    if n is None:
        return "-"
    return (
        f"{n / 1e6:.2f}M"
        if n >= 1e6
        else f"{n / 1000:.0f}k" if n >= 1000 else str(round(n))
    )


def _usd(x) -> str:
    return "-" if x is None else f"${x:.2f}"


def _time(s: str) -> dt.datetime | None:
    try:
        return dt.datetime.strptime((s or "")[:19], "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None


def live_side(state: pathlib.Path) -> dict[tuple, list[dict]]:
    """The live monitor's posts with their costs, by (PR, head), in the order posted."""
    out: dict[tuple, list[dict]] = {}
    for r in _jsonl(state / "review_cost.jsonl"):
        if r.get("pr") and r.get("head"):
            out.setdefault((int(r["pr"]), r["head"]), []).append(r)
    return out


def _live_for(recs: list[dict], feed_at: str) -> dict:
    """The live post of the feed line the mirror took: the one that started at it, else the first posted after it,
    else the last."""
    return (
        next((r for r in recs if feed_at and r.get("started") == feed_at), None)
        or next((r for r in recs if (r.get("at") or "") >= (feed_at or "")), None)
        or recs[-1]
    )


def mirror_side(folder: pathlib.Path) -> dict[tuple, dict]:
    """The mirror's reviews with their costs, by (PR, head)."""
    out = {}
    for d in sorted(folder.glob("pr*_*_*")) if folder.is_dir() else []:
        c = _json(d / "cost.json")
        if c.get("pr") and c.get("head") and (d / "post.md").is_file():
            out[(int(c["pr"]), c["head"])] = dict(c, dir=str(d))
    return out


def _plus(t: str, minutes: float) -> str:
    x = _time(t)
    return (
        (x + dt.timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ") if x else t
    )


def live_spans(every: list[dict], feed: list[dict]) -> list[tuple[str, str]]:
    """Each live review's span from its start to its post: a feed line's review to its post (an hour when it has none:
    dropped, or not posted yet), and a post with a start no feed line has."""
    posts = {
        (int(r["pr"]), r.get("head"), r["started"]): r["at"]
        for r in every
        if r.get("started") and r.get("at")
    }
    spans = []
    for f in feed:
        a = f.get("at") or ""
        if a and isinstance(f.get("pr"), int):
            spans.append(
                (a, posts.pop((f["pr"], f.get("head"), a), None) or _plus(a, 60))
            )
    return spans + [(k[2], v) for k, v in posts.items()]


def live_claude(
    recs: list[dict],
    spans: list[tuple[str, str]],
    root: pathlib.Path,
    price: dict | None = None,
) -> list[dict]:
    """For each live post in `recs`: the PR's Claude lanes from the review's start to its post, the monitor session's
    calls in that window, and how many other live reviews (`spans`) ran in it. {} for a post with no start.
    """
    windows = [(r.get("started") or "", r.get("at") or "") for r in recs]
    use = [i for i, (a, b) in enumerate(windows) if a and b]
    out: list[dict] = [{} for _ in recs]
    if not use:
        return out
    runs = transcripts.runs(root, since=min(windows[i][0] for i in use), price=price)
    sessions = transcripts.session_windows(root, [windows[i] for i in use], price=price)
    for j, i in enumerate(use):
        a, b = windows[i]
        others = sum(1 for c, d in spans if (c, d) != (a, b) and c < b and d > a)
        out[i] = {
            "lanes": transcripts.lanes_for(runs, int(recs[i]["pr"]), a, b),
            "session": sessions[j],
            "session_overlap": others,
        }
    return out


def figures(side: str, rec: dict, claude: dict) -> dict:
    """A review's figures: Claude's dollars for its lanes (verifier, quality, notice), the session's in its window and
    its share of them (over the reviews that ran in the window), the review's (lanes and share), the lanes' ieq; the
    drafting models' tokens and requests (a binding's draft and code quality pass); and the minutes from start to post.
    """
    lanes = [
        ((claude.get("lanes") or {}).get(n) or {}).get("tally") or {} for n in LANES
    ]
    lanes = [t for t in lanes if t]
    sess = claude.get("session") or {}
    lanes_usd = (
        round(sum(float(t.get("usd") or 0) for t in lanes), 4) if lanes else None
    )
    session_usd = sess.get("usd")
    share = (
        round(float(session_usd) / (1 + int(claude.get("session_overlap") or 0)), 4)
        if session_usd is not None
        else None
    )
    if side == "live":
        draft = rec.get("draft") or {}
        local = draft.get("method") == "local-worker"
        model, requests = (
            (None, None) if local else (draft.get("tokens"), draft.get("requests"))
        )
        minutes = rec.get("review_min")
    else:
        q = rec.get("quality") or {}
        parts = [rec.get("draft") or {}] + ([q] if _by_model(q) else [])
        model = sum(int(p.get("tokens") or 0) for p in parts) or None
        requests = sum(int(p.get("requests") or 0) for p in parts) or None
        minutes = (rec.get("minutes") or {}).get("review")
    review_usd = (
        None
        if lanes_usd is None and share is None
        else round((lanes_usd or 0) + (share or 0), 4)
    )
    return {
        "lanes_usd": lanes_usd,
        "session_usd": session_usd,
        "share_usd": share,
        "review_usd": review_usd,
        "lanes_ieq": sum(int(t.get("ieq") or 0) for t in lanes) if lanes else None,
        "model_tokens": model,
        "model_requests": requests,
        "minutes": minutes,
    }


def live_rate(every: list[dict]) -> dict:
    """The live monitor's reviews per day: its posts over the last RATE_DAYS days of review_cost.jsonl (none under half
    a day of records)."""
    times = sorted(t for t in (_time(r.get("at") or "") for r in every) if t)
    if not times:
        return {"per_day": None, "posts": 0, "days": 0}
    start = max(times[0], times[-1] - dt.timedelta(days=RATE_DAYS))
    days = (times[-1] - start).total_seconds() / 86400.0
    posts = sum(1 for t in times if t >= start)
    return {
        "per_day": round(posts / days, 1) if days >= 0.5 else None,
        "posts": posts,
        "days": round(days, 2),
    }


def _checks(side: str, p: dict) -> list[str]:
    out = []
    lanes = p[side]["claude"].get("lanes") or {}
    for name in LANES:
        t = (lanes.get(name) or {}).get("tally") or {}
        if t.get("cache_write_1h"):
            out.append(
                f"#{p['pr']} {side} {name}: {t['cache_write_1h']} tokens written to the 1-hour cache"
            )
    if side == "mirror":
        c = p["mirror"]["cost"]
        if "verifier" not in lanes:
            out.append(
                f"#{p['pr']} mirror: no verifier transcript in the review's window"
            )
        # A record is owed only by a review that got the binding's pass (cost.json's quality_on_binding).
        if c.get("quality_on_binding") and not _by_model(c.get("quality")):
            out.append(
                f"#{p['pr']} mirror arm {p['arm']}: the quality binding's lane left no record"
            )
    elif p["live"]["claude"] and "verifier" not in lanes:
        out.append(
            f"#{p['pr']} live: no verifier transcript in the review's window (check --live-claude)"
        )
    return out


def build(
    live_state: pathlib.Path,
    mirror_out: pathlib.Path,
    live_root: pathlib.Path | None = None,
    price: dict | None = None,
) -> dict:
    """The pairs with each side's parsed post, Claude tallies and figures, the disagreements, the totals per system and
    arm, the checks, and every review lost on either side."""
    root = live_root or transcripts.claude_projects()
    live, mirror = live_side(live_state), mirror_side(mirror_out)
    every = [r for recs in live.values() for r in recs]
    feed = _jsonl(live_state / "mirror_feed.jsonl")
    drops = _jsonl(mirror_out / "drops.jsonl")
    keys = sorted(set(live) & set(mirror))
    lrecs = [_live_for(live[k], mirror[k].get("feed_at") or "") for k in keys]
    price = price or transcripts.DEFAULT_PRICES
    lclaude = live_claude(lrecs, live_spans(every, feed), root, price)
    pairs = []
    for key, lrec, lc in zip(keys, lrecs, lclaude):
        mrec = mirror[key]
        try:
            ltext = pathlib.Path(lrec.get("post") or "").read_text(encoding="utf-8")
        except OSError:
            ltext = ""
        mtext = (pathlib.Path(mrec["dir"]) / "post.md").read_text(encoding="utf-8")
        lp, mp = parse_post(ltext), parse_post(mtext)
        mc = mrec.get("claude") or {}
        pairs.append(
            {
                "pr": key[0],
                "head": key[1],
                "round": lrec.get("round") or mrec.get("round"),
                "arm": mrec.get("arm") or "A",
                "live": {
                    "post": lrec.get("post"),
                    "parsed": lp,
                    "claude": lc,
                    "figures": figures("live", lrec, lc),
                    "cost": lrec,
                    "read": bool(ltext),
                },
                "mirror": {
                    "dir": mrec["dir"],
                    "parsed": mp,
                    "claude": mc,
                    "figures": figures("mirror", mrec, mc),
                    "cost": mrec,
                },
                "disagreements": (
                    disagreements(lp, mp)
                    if ltext
                    else ["the live post's copy was not read"]
                ),
            }
        )
    rate = live_rate(every)
    totals: dict[str, dict] = {}
    for arm in sorted({p["arm"] for p in pairs}) or ["A"]:
        for side in ("live", "mirror"):
            t: dict = {}
            for k in FIGURES:
                vals = [
                    p[side]["figures"][k]
                    for p in pairs
                    if p["arm"] == arm and p[side]["figures"][k] is not None
                ]
                t[k] = {
                    "n": len(vals),
                    "sum": round(sum(vals), 4) if vals else None,
                    "mean": round(statistics.mean(vals), 4) if vals else None,
                }
            per = t["review_usd"]["mean"]
            t["usd_per_day"] = (
                round(per * rate["per_day"], 2)
                if per is not None and rate["per_day"]
                else None
            )
            totals[f"{side} {arm}"] = t
    dropped = {(int(r["pr"]), r.get("head")) for r in drops if r.get("pr")}
    lost = [
        dict(
            r,
            live_post=(live.get((int(r["pr"]), r.get("head"))) or [{}])[-1].get("post"),
        )
        for r in drops
        if r.get("pr")
    ]
    for k in sorted(set(mirror) - set(live) - dropped):
        later = [
            r
            for r in feed
            if r.get("pr") == k[0]
            and r.get("head") != k[1]
            and (r.get("at") or "") > (mirror[k].get("feed_at") or "")
        ]
        why = (
            f"no live post at this head; the live monitor started a review of {str(later[0]['head'])[:10]} at "
            f"{later[0].get('at')}: its review of this head was dropped"
            if later
            else "no live post at this head yet"
        )
        lost.append(
            {
                "pr": k[0],
                "head": k[1],
                "round": mirror[k].get("round"),
                "arm": mirror[k].get("arm"),
                "why": why,
                "dir": mirror[k]["dir"],
                "side": "live",
            }
        )
    checks = [c for p in pairs for side in ("live", "mirror") for c in _checks(side, p)]
    return {
        "pairs": pairs,
        "totals": totals,
        "rate": rate,
        "checks": checks,
        "lost": lost,
        "live_claude": str(root),
        "prices": price,
    }


def _lane_cell(claude: dict, name: str) -> str:
    t = (claude.get("lanes") or {}).get(name) or {}
    tl = t.get("tally") or {}
    if not tl:
        return "-"
    eff = transcripts.effort_of(tl) or "?"
    return f"{_usd(tl.get('usd'))} · {_k(tl.get('processed'))} · {_k(tl.get('output'))} ({t.get('runs', 0)}, {eff})"


def _by_model(r: dict | None) -> bool:
    """Whether a cost.json record (the draft, the code quality pass) is a drafting model's work: a binding's or the
    local worker's, not Claude's lane, nor none."""
    return (r or {}).get("by") not in (None, "claude")


def _model_cell(r: dict) -> str:
    if not _by_model(r):
        return "(Claude)" if r else "-"
    return (
        f"{_k(r.get('prompt_tokens'))} · {_k(r.get('completion_tokens'))} · {_k(r.get('reasoning_tokens'))}, "
        f"{_v(r.get('requests'))} req, {_v(r.get('wall_min'))} min"
    )


def render(rep: dict) -> list[str]:
    pr = rep["prices"]
    out = [
        f"# Mirror report: {len(rep['pairs'])} pair(s), live vs mirror",
        "",
        f"Claude's figures come from the transcripts on both sides (the live ones from {rep['live_claude']}), at "
        f"config prices' Claude Opus rates per million tokens (rk spend's): ${pr['uncached']:g} input, "
        f"${pr['cache_read']:g} cache read, "
        f"${pr['cache_write_5m']:g} 5-minute and ${pr['cache_write_1h']:g} 1-hour cache write, ${pr['output']:g} "
        "output. Output and thinking are undercounted alike on both sides (the transcripts record a stream's "
        "start). Lanes are the PR's verifier, Claude's code quality pass and notices from the review's start to its "
        "post; the session is the orchestrator's calls in that window (the live session also carries its operator's "
        "other "
        "work), and the share divides it over the reviews that ran in the window. Findings are 🔴·🟠·🟡 counts. "
        "Model is the drafting models' tokens (a binding's draft and code quality pass, not priced).",
        "",
        "| PR | round | head | arm | light L / M | findings L / M | priors L / M | quality L / M | hygiene L / M | "
        "review $ L / M | lanes $ L / M | session share $ L / M | model L / M | minutes L / M |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for p in rep["pairs"]:
        lp, mp = p["live"]["parsed"], p["mirror"]["parsed"]
        lf, mf = p["live"]["figures"], p["mirror"]["figures"]
        out.append(
            f"| {p['pr']} | {p['round']} | `{p['head'][:10]}` | {p['arm']} | {lp['light'] or '-'} / "
            f"{mp['light'] or '-'} | {_counts(lp['findings'])} / {_counts(mp['findings'])} | "
            f"{len(lp['priors'])} / {len(mp['priors'])} | "
            f"{lp['data'].get('Code quality') or '-'} / {mp['data'].get('Code quality') or '-'} | "
            f"{len(lp['hygiene'])} / {len(mp['hygiene'])} | {_usd(lf['review_usd'])} / "
            f"{_usd(mf['review_usd'])} | {_usd(lf['lanes_usd'])} / {_usd(mf['lanes_usd'])} | "
            f"{_usd(lf['share_usd'])} / {_usd(mf['share_usd'])} | {_k(lf['model_tokens'])} / "
            f"{_k(mf['model_tokens'])} | {_v(lf['minutes'])} / {_v(mf['minutes'])} |"
        )
    out += [
        "",
        "## Claude per lane ($ · processed · output (runs, effort))",
        "",
        "| PR | arm | verifier L | verifier M | quality L | quality M | notice L | notice M | session $ L / M "
        "(reviews in the window) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for p in rep["pairs"]:
        lc, mc = p["live"]["claude"], p["mirror"]["claude"]

        def sess(c: dict) -> str:
            s = c.get("session") or {}
            return (
                f"{_usd(s.get('usd'))} ({1 + int(c.get('session_overlap') or 0)})"
                if s
                else "-"
            )

        out.append(
            f"| {p['pr']} | {p['arm']} | "
            + " | ".join(_lane_cell(c, n) for n in LANES for c in (lc, mc))
            + f" | {sess(lc)} / {sess(mc)} |"
        )
    out += [
        "",
        "## The mirror's drafting models (tokens in · out · reasoning, requests, minutes)",
        "",
        "| PR | arm | draft | code quality pass | checks min | verify min |",
        "|---|---|---|---|---|---|",
    ]
    for p in rep["pairs"]:
        c = p["mirror"]["cost"]
        mins = c.get("minutes") or {}
        out.append(
            f"| {p['pr']} | {p['arm']} | {_model_cell(c.get('draft') or {})} | "
            f"{_model_cell(c.get('quality') or {})} | {_v(mins.get('checks'))} | {_v(mins.get('verify'))} |"
        )
    rate = rep["rate"]
    per_day = (
        f"{rate['per_day']} reviews a day ({rate['posts']} live posts in {rate['days']} days)"
        if rate["per_day"]
        else "no rate (under half a day of live posts)"
    )
    out += [
        "",
        "## Totals per system and arm",
        "",
        f"Mean per review (n), and a day at the live monitor's rate: {per_day}.",
        "",
        "| system, arm | review $ | lanes $ | session share $ | session $ | lanes ieq | model tokens · requests | "
        "minutes | $ a day |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, t in rep["totals"].items():

        def m(k: str, f=_usd) -> str:
            return f"{f(t[k]['mean'])} ({t[k]['n']})"

        out.append(
            f"| {name} | {m('review_usd')} | {m('lanes_usd')} | {m('share_usd')} | {m('session_usd')} | "
            f"{m('lanes_ieq', _k)} | {_k(t['model_tokens']['mean'])} · {_v(t['model_requests']['mean'])} | "
            f"{m('minutes', _v)} | {_usd(t['usd_per_day'])} |"
        )
    out += [
        "",
        "## The judge's worklist",
        "",
        "For each pair: read both posts and the mirror's notes, then decide each item below: which side is right, "
        "with the evidence at the head (the code, the checks, the PR's data).",
        "",
    ]
    for p in rep["pairs"]:
        # quality/: the quality binding's pass, written only for a review that got it (mirror.divert)
        quality = ", quality/" if _by_model(p["mirror"]["cost"].get("quality")) else ""
        out += [
            f"### #{p['pr']} {p['round']} at `{p['head'][:10]}` (arm {p['arm']})",
            f"- live post: {p['live']['post'] or '-'}",
            f"- mirror: {p['mirror']['dir']}/post.md (notes.md, draft/{quality} beside it)",
        ]
        out += [f"- {d}" for d in p["disagreements"]] or [
            "- no disagreement found by the script"
        ]
        out.append("")
    if rep["checks"]:
        out += ["## Checks", ""] + [f"- {c}" for c in rep["checks"]] + [""]
    if rep["lost"]:
        out += ["## Lost reviews (every review without a pair, and why)", ""]
        for r in rep["lost"]:
            side = (
                "the live review" if r.get("side") == "live" else "the mirror's review"
            )
            extra = (
                f"; the live monitor posted it: {r['live_post']}"
                if r.get("live_post")
                else ""
            )
            out.append(
                f"- #{r['pr']} {r.get('round') or ''} at `{str(r.get('head'))[:10]}`, arm "
                f"{r.get('arm') or '-'}: {side}: {r.get('why')}{extra}"
            )
    return out


def _config() -> dict:
    """This deployment's configuration when there is one (its `prices`), else {}: the report needs none."""
    from . import config

    try:
        return config.load(check=False)
    except Exception:  # noqa: BLE001 - no configuration here: the default prices
        return {}


def main(args) -> int:
    root = (
        pathlib.Path(args.live_claude) if getattr(args, "live_claude", None) else None
    )
    rep = build(
        pathlib.Path(args.live_state),
        pathlib.Path(args.mirror),
        root,
        transcripts.prices(_config()),
    )
    text = (
        json.dumps(rep, indent=1, ensure_ascii=False)
        if args.json
        else "\n".join(render(rep))
    ) + "\n"
    if args.out:
        pathlib.Path(args.out).write_text(text, encoding="utf-8")
        print(f"mirror report: {len(rep['pairs'])} pair(s) written to {args.out}")
    else:
        print(text, end="")
    return 0
