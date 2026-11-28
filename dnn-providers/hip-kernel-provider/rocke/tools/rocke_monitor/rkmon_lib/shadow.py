# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`rk shadow-report [N...]`: the shadow trial's report (offload.mode shadow; rkmon_lib/offload.py).

Per PR, each offload run kept for a round we posted (<lessons>/offload/pr<N>_<round>/) against the review we posted
for that round, read from GitHub (GETs only; the ledger names the comment): the findings both have (matched), those
only our post has (missed by the offload), and those only the offload has (extra), each offload finding with its
evidence-check result; the prior items' statuses side by side; both header lights; whether the offload would have been
audited or reviewed in depth, and why; the reviewer model's tokens per pass. Then a summary across the trial's PRs:
the last offload.trial_prs with a kept run and a post, or those named.

Matching (`match`): a pair scores by the file (0.35 for the same path, 0.25 for the same file name), the line (0.15
within LINE_NEAR) and the overlap of the words of the problem, impact and fix (0 to 1); pairs are taken best first,
each finding once, from MATCH_MIN: the same place needs a few shared words, words alone need most of them. Writes <outputs>/rocke_shadow/shadow_pr<N>_<round>.md and summary.md; nothing else.
"""

import json
import pathlib
import re
import time

from . import gh, offload, paths
from .snapshot import same_commit

LINE_NEAR = 15
MATCH_MIN = 0.55
FINDING = re.compile(
    r"^\|\s*([NQ]\d+)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*(.*)\|\s*$"
)
PRIOR_HEAD = re.compile(
    r"^\|\s*(?:#\s*\|\s*)?Prior item\s*\|\s*Status\s*\|", re.I
)  # a regrade.md has an id column
WORD = re.compile(r"[a-z][a-z0-9_]{2,}")
STOP = set(
    "the and for with that this from not are was its but has have into when what which only also there their "
    "they them then than would should could does each impact fix none yet".split()
)
LIGHTS = "🔴🟠🟡🟢⚪"


def _light(text: str) -> str:
    return next((ch for ch in text or "" if ch in LIGHTS), "")


def _words(text: str) -> set[str]:
    return {
        w
        for w in WORD.findall(re.sub(r"<br>|[*`•]", " ", (text or "").lower()))
        if w not in STOP
    }


def findings(text: str) -> list[dict]:
    """The findings table's rows of a post (N1, ...; a draft's Q rows too): id, light, file, line, problem, text."""
    out = []
    for row in (text or "").splitlines():
        m = FINDING.match(row)
        if not m:
            continue
        cid, sev, where, body = m.groups()
        loc = re.search(r"`?([^`:\s]+(?:/[^`:\s]+)*):?(\d+)?", where or "")
        problem = re.search(r"\*\*(.+?)\*\*", body)
        out.append(
            {
                "id": cid,
                "severity": _light(sev),
                "file": loc.group(1) if loc else "",
                "line": int(loc.group(2)) if loc and loc.group(2) else 0,
                "problem": problem.group(1) if problem else body.split("<br>")[0][:120],
                "text": body.strip(),
            }
        )
    return out


def priors(text: str) -> list[dict]:
    """The prior items table's rows: the item, its light and status word (Bots / CI, Our checks and Code quality
    too)."""
    out, on = [], False
    for row in (text or "").splitlines():
        if PRIOR_HEAD.match(row):
            on = True
            continue
        if on and not row.startswith("|"):
            on = False
        if on and not row.startswith("|---"):
            cells = [c.strip() for c in row.strip().strip("|").split("|")]
            if len(cells) >= 2:
                status = cells[-1]
                word = re.search(
                    r"[A-Z][A-Za-z]+(?: yet)?", re.sub(f"[{LIGHTS}]", "", status)
                )
                out.append(
                    {
                        "item": (
                            cells[0]
                            if not re.fullmatch(r"P\d+", cells[0])
                            else cells[1]
                        ),
                        "light": _light(status),
                        "status": word.group(0) if word else "",
                    }
                )
    return out


def header(text: str) -> tuple[str, str]:
    """The post's header light and its few words."""
    for row in (text or "").splitlines():
        if row.startswith("## ") and ("First-round" in row or "Follow-up" in row):
            reason = row.split(":", 1)[1].strip() if ":" in row else ""
            return _light(row.rsplit("—", 1)[-1]), reason
    return "", ""


def score(a: dict, b: dict) -> float:
    s = 0.0
    if a["file"] and b["file"]:
        if a["file"] == b["file"]:
            s += 0.35
        elif (
            pathlib.PurePosixPath(a["file"]).name
            == pathlib.PurePosixPath(b["file"]).name
        ):
            s += 0.25
        if s and a["line"] and b["line"] and abs(a["line"] - b["line"]) <= LINE_NEAR:
            s += 0.15
    wa, wb = _words(a["problem"] + " " + a["text"]), _words(
        b["problem"] + " " + b["text"]
    )
    if wa and wb:
        s += len(wa & wb) / len(
            wa | wb
        )  # the words' overlap: the same place still needs a few shared words
    return min(s, 1.0)


def match(
    posted: list[dict], ours: list[dict]
) -> tuple[list[tuple[dict, dict, float]], list[dict], list[dict]]:
    """(matched pairs best first, the posted findings left, the offload findings left)."""
    pairs = sorted(
        ((score(p, o), i, j) for i, p in enumerate(posted) for j, o in enumerate(ours)),
        reverse=True,
    )
    used_p, used_o, out = set(), set(), []
    for sc, i, j in pairs:
        if sc < MATCH_MIN:
            break
        if i in used_p or j in used_o:
            continue
        used_p.add(i)
        used_o.add(j)
        out.append((posted[i], ours[j], round(sc, 2)))
    return (
        out,
        [p for i, p in enumerate(posted) if i not in used_p],
        [o for j, o in enumerate(ours) if j not in used_o],
    )


def _read(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def offload_findings(run: pathlib.Path) -> tuple[list[dict], str]:
    """The offload's findings: its assembled post's (with each post id's source id from the notes' map: lines), else
    the draft's and the code quality pass's; and the text they came from."""
    post = _read(run / "post_draft.md")
    if findings(post):
        source = dict(
            re.findall(r"map:\s*(N\d+)\s*=\s*([NQ]\d+)", _read(run / "notes_draft.md"))
        )
        out = findings(post)
        for f in out:
            f["source"] = source.get(f["id"], f["id"])
        return out, post
    out = findings(_read(run / "draft.md")) + findings(_read(run / "quality.md"))
    for f in out:
        f["source"] = f["id"]
    return out, _read(run / "draft.md")


def posted_review(cfg: dict, store, n: int, run_head: str, started: str) -> dict | None:
    """Our post for the run's round: the ledger's post at the run's head, else the first after the run started; its
    body read from GitHub."""
    posts = [r for r in store.ledger() if r.get("kind") == "post" and r.get("pr") == n]
    same = [r for r in posts if same_commit(r.get("head"), run_head)]
    later = [r for r in posts if started and r.get("ts", "") >= started]
    pick = (same or later or [None])[0]
    if not pick:
        return None
    m = re.search(r"issuecomment-(\d+)", pick.get("ref") or "")
    if not m:
        return {
            "url": pick.get("ref") or "",
            "body": "",
            "at": pick.get("ts"),
            "error": "no comment id in the ledger",
        }
    try:
        c = gh.rest(
            f"repos/{cfg['repo']}/issues/comments/{m.group(1)}",
            jq="{body, html_url, created_at}",
        )
    except gh.GhError as exc:
        return {
            "url": pick.get("ref"),
            "body": "",
            "at": pick.get("ts"),
            "error": str(exc)[:200],
        }
    return {
        "url": c.get("html_url") or pick.get("ref"),
        "body": c.get("body") or "",
        "at": c.get("created_at"),
    }


def runs(cfg: dict, prs: list[int] | None = None) -> list[pathlib.Path]:
    """The kept offload runs: of the PRs named, else every shadow run; oldest first."""
    root = offload.archive_root(cfg)
    if not root.is_dir():
        return []
    found = []
    for d in root.glob("pr*_*"):
        m = re.fullmatch(r"pr(\d+)_(\w+)", d.name)
        h = _header(d)
        if (
            not m
            or not h
            or (prs and int(m.group(1)) not in prs)
            or (not prs and h.get("mode") != "shadow")
        ):
            continue
        found.append((h.get("started") or "", d))
    return [d for _s, d in sorted(found)]


def _header(d: pathlib.Path) -> dict:
    try:
        return json.loads((d / "offload_run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def report_pr(cfg: dict, st: dict, store, run: pathlib.Path) -> dict:
    """One kept run against our post: the comparison, for its page and the summary."""
    h = _header(run)
    n, rnd = int(h.get("pr") or 0), h.get("round") or "?"
    ev = offload.read_report(run)
    by_source = {c["id"]: c for c in ev.get("claims") or []}
    ours, our_text = offload_findings(run)
    post = posted_review(cfg, store, n, h.get("head") or "", h.get("started") or "")
    posted = findings(post["body"]) if post else []
    matched, missed, extra = match(posted, ours)
    rec = (st.get("prs") or {}).get(str(n)) or {}
    kind = "follow-up" if str(rnd).startswith("fu") else "first-round"
    files = [
        x.strip() for x in _read(run / "prep" / "files.txt").splitlines() if x.strip()
    ] or list((rec.get("files_cache") or {}).get("files") or [])
    reasons = offload.deep_reasons(cfg, rec, kind, files, run)
    tokens = {}
    try:
        tokens = (
            json.loads(_read(run / "tokens_total.json") or "{}").get("passes") or {}
        )
    except ValueError:
        pass
    for f in ours:
        c = by_source.get(f.get("source") or f["id"]) or {}
        f["evidence"] = c.get("result") or "no record"
        f["why"] = c.get("why") or ""
    return {
        "pr": n,
        "round": rnd,
        "head": h.get("head") or "",
        "run": str(run),
        "post": post,
        "posted_header": header(post["body"]) if post else ("", ""),
        "offload_header": header(our_text),
        "matched": matched,
        "missed": missed,
        "extra": extra,
        "posted": posted,
        "ours": ours,
        "priors": (
            priors(post["body"]) if post else [],
            priors(_read(run / "regrade.md")),
        ),
        "evidence": ev.get("summary") or "no evidence check",
        "evidence_result": ev.get("result") or "",
        "route": "deep" if reasons else "audit",
        "reasons": reasons,
        "tokens": tokens,
    }


def _cell(text: str, n: int = 110) -> str:
    t = re.sub(r"\s+", " ", str(text or "")).replace("|", "/")
    return t[:n] + ("…" if len(t) > n else "")


def _where(f: dict) -> str:
    return (
        f"`{f['file']}:{f['line']}`"
        if f.get("file") and f.get("line")
        else f"`{f['file']}`" if f.get("file") else "—"
    )


def render_pr(r: dict) -> str:
    post = r["post"]
    lines = [f"# Shadow report: #{r['pr']} {r['round']} at {r['head'][:10]}", ""]
    if not post:
        lines += [
            "No post of ours for this round yet: run the report again after the post.",
            "",
        ]
    elif post.get("error"):
        lines += [f"Our post ({post.get('url')}) was not read: {post['error']}", ""]
    pl, pr_ = r["posted_header"]
    ol, or_ = r["offload_header"]
    tok = r["tokens"].get("all") or {}
    lines += [
        f"- Posted: {pl or '—'} {pr_ or ''} ({(post or {}).get('url') or 'none'})",
        f"- Offload: {ol or '—'} {or_ or ''}; it would have been "
        + (
            f"reviewed in depth ({'; '.join(r['reasons'])})"
            if r["reasons"]
            else "audited"
        ),
        f"- Evidence check: {r['evidence']}",
        "- Reviewer model tokens: "
        + (
            ", ".join(
                f"{p} {t.get('prompt_tokens', 0) + t.get('completion_tokens', 0)}"
                for p, t in r["tokens"].items()
                if p != "all"
            )
            or "none recorded"
        )
        + (
            f" (total {tok.get('prompt_tokens', 0) + tok.get('completion_tokens', 0)} in {tok.get('requests', 0)} "
            "requests; reported by the endpoint, or estimated)"
            if tok
            else ""
        ),
        "",
        f"## Matched ({len(r['matched'])})",
        "",
        "| Posted | Offload | Sev (posted / offload) | file:line | Evidence check | Finding (posted) |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {p['id']} | {o['id']} | {p['severity'] or '—'} / {o['severity'] or '—'} | {_where(p)} | "
        f"{o['evidence']} | {_cell(p['problem'])} |"
        for p, o, _sc in r["matched"]
    ] or ["| — | — | — | — | — | — |"]
    lines += [
        "",
        f"## Missed: in our post, not in the offload ({len(r['missed'])})",
        "",
        "| Posted | Sev | file:line | Finding |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {p['id']} | {p['severity'] or '—'} | {_where(p)} | {_cell(p['problem'])} |"
        for p in r["missed"]
    ] or ["| — | — | — | — |"]
    lines += [
        "",
        f"## Extra: in the offload, not in our post ({len(r['extra'])})",
        "",
        "| Offload | Sev | file:line | Evidence check | Finding |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {o['id']} | {o['severity'] or '—'} | {_where(o)} | {o['evidence']}"
        f"{': ' + _cell(o['why'], 60) if o['why'] and o['evidence'] != 'PASS' else ''} | {_cell(o['problem'])} |"
        for o in r["extra"]
    ] or ["| — | — | — | — | — |"]
    posted_p, ours_p = r["priors"]
    lines += [
        "",
        "## Prior items and status rows",
        "",
        "| Posted | Offload |",
        "|---|---|",
    ]
    for i in range(max(len(posted_p), len(ours_p))):
        a = posted_p[i] if i < len(posted_p) else None
        b = ours_p[i] if i < len(ours_p) else None
        lines.append(
            f"| {_cell(a['item'], 50) + ': ' + a['light'] + ' ' + a['status'] if a else '—'} | "
            f"{_cell(b['item'], 50) + ': ' + b['light'] + ' ' + b['status'] if b else '—'} |"
        )
    lines += ["", f"Kept run: {r['run']}", ""]
    return "\n".join(lines)


def _serious(fs) -> int:
    return sum(1 for f in fs if f.get("severity") in ("🔴", "🟠"))


def render_summary(rows: list[dict], trial: int) -> str:
    lines = [
        f"# Shadow trial summary: {len(rows)} PR round(s) (offload.trial_prs {trial})",
        "",
        "| PR | Round | Matched | Missed (🔴/🟠) | Extra (🔴/🟠) | Light posted / offload | Evidence | Route | "
        "Tokens |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        tok = r["tokens"].get("all") or {}
        lines.append(
            f"| #{r['pr']} | {r['round']} | {len(r['matched'])} of {len(r['posted'])} | "
            f"{len(r['missed'])} ({_serious(r['missed'])}) | {len(r['extra'])} ({_serious(r['extra'])}) | "
            f"{r['posted_header'][0] or '—'} / {r['offload_header'][0] or '—'} | {r['evidence_result'] or '—'} "
            f"| {r['route']} | {tok.get('prompt_tokens', 0) + tok.get('completion_tokens', 0) or '—'} |"
        )
    posted = sum(len(r["posted"]) for r in rows)
    found = sum(len(r["matched"]) for r in rows)
    serious = sum(_serious(r["posted"]) for r in rows)
    serious_found = sum(_serious([p for p, _o, _s in r["matched"]]) for r in rows)
    extra = [o for r in rows for o in r["extra"]]
    held = sum(1 for o in extra if o["evidence"] == "PASS")
    per = [(r["tokens"].get("all") or {}) for r in rows if r["tokens"]]
    totals = [t.get("prompt_tokens", 0) + t.get("completion_tokens", 0) for t in per]
    lines += [
        "",
        f"- Findings found: {found} of {posted} posted"
        + (f" ({100 * found / posted:.0f}%)" if posted else "")
        + f"; 🔴/🟠: {serious_found} of {serious}"
        + (f" ({100 * serious_found / serious:.0f}%)" if serious else ""),
        f"- Extra offload findings: {len(extra)} ({_serious(extra)} 🔴/🟠); their evidence held for {held}",
        f"- Evidence checks passed: {sum(1 for r in rows if r['evidence_result'] == 'PASS')} of {len(rows)}",
        f"- Routes: {sum(1 for r in rows if r['route'] == 'audit')} audit, "
        f"{sum(1 for r in rows if r['route'] == 'deep')} deep",
        "- Reviewer model tokens per review: "
        + (
            f"mean {sum(totals) / len(totals):.0f}, most {max(totals)}"
            if totals
            else "none recorded"
        ),
        "",
    ]
    return "\n".join(lines)


def cmd(
    cfg: dict, st: dict, store, prs: list[int] | None = None
) -> tuple[list[str], int]:
    """`rk shadow-report [N...]`: the pages' paths and the summary's last lines."""
    found = runs(cfg, prs)
    if not prs:
        trial = int(offload.settings(cfg)["trial_prs"])
        latest: dict[int, list] = {}
        for d in found:
            latest.setdefault(int(_header(d).get("pr") or 0), []).append(d)
        keep_prs = sorted(
            latest, key=lambda n: _header(latest[n][-1]).get("started") or ""
        )[-trial:]
        found = [d for n in keep_prs for d in latest[n]]
    if not found:
        return [
            f"no kept offload run to report on (offload.mode {offload.mode(cfg)}; kept runs: "
            f"{offload.archive_root(cfg)})"
        ], 2
    out = pathlib.Path(paths.get(cfg, "outputs_dir")) / "rocke_shadow"
    out.mkdir(parents=True, exist_ok=True)
    rows, lines = [], []
    for d in found:
        r = report_pr(cfg, st, store, d)
        rows.append(r)
        page = out / f"shadow_pr{r['pr']}_{r['round']}.md"
        page.write_text(render_pr(r), encoding="utf-8")
        lines.append(
            f"{page}: matched {len(r['matched'])}, missed {len(r['missed'])}, extra {len(r['extra'])}"
            + ("" if r["post"] else " (no post yet)")
        )
    summary = render_summary(
        [r for r in rows if r["post"]], int(offload.settings(cfg)["trial_prs"])
    )
    (out / "summary.md").write_text(
        summary
        + f"\n_Written {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}._\n",
        encoding="utf-8",
    )
    return (
        lines
        + [f"{out / 'summary.md'}"]
        + [x for x in summary.splitlines() if x.startswith("- ")],
        0,
    )
