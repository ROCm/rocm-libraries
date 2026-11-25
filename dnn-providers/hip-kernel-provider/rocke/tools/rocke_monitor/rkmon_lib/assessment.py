# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The assessment of the drafters against a reference: each drafter's draft (a candidate) against the verified post (the
reference). The candidates (bindings.candidates): the live drafter (a binding, or the local worker) and the shadow
drafter (a binding under assessment). The verifier writes one line per judgement in notes.md, each starting with the
candidate's tag (its binding's name, or the local worker's tag, bindings.worker_tag): the live draft's while it
reconciles the live draft with its blind pass, the shadow draft's once the post is final. A draft's findings N<k> are
G<k> here, the post's are N<j>:
  <tag> G<k> match N<j>        the draft's finding is the post's N<j>
  <tag> G<k> new N<j>          the draft found it, the blind pass did not, and it holds (the post's N<j>)
  <tag> G<k> false <reason>    the draft's claim does not hold
  <tag> G<k> dropped <reason>  true, but not worth posting
  <tag> missed N<j> <light>    in the post, not in the draft
  <tag> prior <row> right|wrong   the draft's status of one of its prior rows (its id P<k>, or its first cell)
  <tag> none                   no draft to score (none came, or it was not done at post time)
At post time (`rk post --premises`, premises.finish) `record` writes one record per candidate that has lines or a
draft (a binding that drafted the head is a candidate too, whatever the roles name now) into
state/drafter_assessment.jsonl: the PR, round and head; the candidate (role, binding, tag, model, method)
and the reference; the draft's status at post time (ready; late: still running; missing); the counts by severity (the
post's light for match, new and missed; the draft's for false and dropped); the prior statuses right and wrong, and
for a draft with settlers (worker/drafter.py, prior_settled.json) the same by settler (script, model); its prior rows'
counts and how many the verifier judged; and the draft's tokens, requests, wall time and phases. The records of the
single drafter scored before (state/<tag>_assessment.jsonl) are read too: the drafter role's, method "single-pass".

`card` is the dashboards' card "<TAG> drafter vs <reference>" (TAG: the shadow drafter's binding, else the drafter's):
one column per candidate, each scored on its current method only (its latest draft's), with a line counting the
drafts of other methods it leaves out. The rows score the phases apart: prior items (script statuses right, model
statuses right, the rows each settled and their evidence pass rate, the rows the verifier judged), new findings (🔴/🟠
recall, 🟡 recall, false claims per draft), tokens and minutes per phase, and the drafts missing or late at post time.
"""

from __future__ import annotations

import json
import pathlib
import re

from . import bindings, draft_stage, prior_settled

SEVS = {"🔴": "red", "🟠": "orange", "🟡": "yellow", "🟢": "green", "⚪": "white"}
WORDS = {
    "red": "red",
    "orange": "orange",
    "yellow": "yellow",
    "green": "green",
    "white": "white",
}
FINDING_ROW = re.compile(r"^\|\s*N(\d+)\s*\|\s*([^|]*?)\s*\|")
FILE = "drafter_assessment.jsonl"
OLD_METHOD = draft_stage.OLD_METHOD
LOCAL_METHOD = draft_stage.LOCAL.name  # the local worker's drafts


def severity(text) -> str:
    """A severity word (red, orange, yellow, green, white) from a light or a word; "unknown" for neither."""
    t = str(text or "").strip()
    for light, word in SEVS.items():
        if light in t:
            return word
    return WORDS.get(t.lower(), "unknown")


def post_severities(text: str) -> dict[int, str]:
    """A post's (or a draft's) findings table: N<j> -> its severity word."""
    out = {}
    for row in (text or "").splitlines():
        m = FINDING_ROW.match(row)
        if m:
            out[int(m.group(1))] = severity(m.group(2))
    return out


def parse(notes: str, tag: str) -> dict:
    """The assessment lines of a notes text for the candidate `tag`: match, new, false, dropped, missed, prior, and the
    lines that start with the tag but read as none of them (unparsed). `<tag> none` marks a round with no draft.
    """
    t = re.escape(tag)
    lead = rf"^[\s>*•-]*{t}\s+"
    out = {
        "match": [],
        "new": [],
        "false": [],
        "dropped": [],
        "missed": [],
        "prior": [],
        "unparsed": [],
        "lines": [],
        "none": False,
    }
    for raw in (notes or "").splitlines():
        line = raw.strip().strip("`")
        if not re.match(lead, line, re.I):
            continue
        out["lines"].append(line)
        m = re.match(lead + r"G(\d+)\s+(match|new)\s+N(\d+)\b", line, re.I)
        if m:
            out[m.group(2).lower()].append({"g": int(m.group(1)), "n": int(m.group(3))})
            continue
        m = re.match(lead + r"G(\d+)\s+(false|dropped)\b[\s:—-]*(.*)$", line, re.I)
        if m:
            out[m.group(2).lower()].append(
                {"g": int(m.group(1)), "why": m.group(3).strip()}
            )
            continue
        m = re.match(lead + r"missed\s+N(\d+)\b\s*(\S*)", line, re.I)
        if m:
            out["missed"].append(
                {"n": int(m.group(1)), "severity": severity(m.group(2))}
            )
            continue
        m = re.match(lead + r"prior\s+(.+?)\s+(right|wrong)\b", line, re.I)
        if m:
            out["prior"].append(
                {"row": m.group(1).strip(), "ok": m.group(2).lower() == "right"}
            )
            continue
        if re.match(lead + r"none\b", line, re.I):
            out["none"] = True
            continue
        out["unparsed"].append(line)
    return out


def file_name(tag: str) -> str:
    """The file the single drafter scored before kept its records in, by its tag."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", tag or "drafter") + "_assessment.jsonl"


def _count(items: list[str]) -> dict:
    out: dict[str, int] = {}
    for s in items:
        out[s] = out.get(s, 0) + 1
    return out


def _norm(text: str) -> str:
    return " ".join(str(text or "").lower().replace("`", "").split())


def row_of(ref: str, rows: list[dict]) -> dict | None:
    """The prior row a `<tag> prior <row>` line names: by its id (P3), else by its first cell (either may be cut)."""
    ref = str(ref or "").strip()
    if re.fullmatch(r"P\d+", ref, re.I):
        return next((r for r in rows if str(r.get("id")).upper() == ref.upper()), None)
    want = _norm(ref)
    for r in rows:
        key = _norm(r.get("key"))
        if want and key and (key.startswith(want) or want.startswith(key)):
            return r
    return None


def build(
    n: int,
    rnd: str,
    head: str,
    notes: str,
    post: str,
    draft: dict | None,
    tag: str,
    model: str,
    reference: str,
    at: str,
    role: str = "drafter",
    method: str = "",
    status: str = "",
    rows: list[dict] | None = None,
    counts: dict | None = None,
    draft_text: str = "",
    binding: str | None = None,
) -> dict:
    """One candidate's record from the notes, the post and its draft: the draft's record (state drafts), its status at
    post time, its prior rows with their settlers (prior_settled.json) and their counts, and, for a draft with no
    record (the local worker's), its text. `binding`: the candidate's binding ("" for the local worker; by default the
    draft's, else the tag)."""
    p = parse(notes, tag)
    post_sev = post_severities(post)
    g_sev = {
        int(str(f["id"])[1:]): severity(f.get("severity"))
        for f in (draft or {}).get("findings") or []
        if str(f.get("id", "")).startswith("N") and str(f["id"])[1:].isdigit()
    }
    if not g_sev and draft_text:
        g_sev = post_severities(draft_text)
    by: dict[str, dict] = {}
    for x in p["prior"]:
        r = row_of(x["row"], rows or [])
        who = (r or {}).get("settler") or "other"
        slot = by.setdefault(who, {"right": 0, "wrong": 0})
        slot["right" if x["ok"] else "wrong"] += 1
    rec = {
        "at": at,
        "pr": n,
        "round": rnd,
        "head": head,
        "candidate": {
            "role": role,
            "binding": (
                binding
                if binding is not None
                else ((draft or {}).get("binding") or tag)
            ),
            "tag": tag,
            "model": model or (draft or {}).get("model") or "",
            "method": method or (draft or {}).get("method") or OLD_METHOD,
        },
        "reference": {"role": "verifier", "model": reference},
        "draft": status or (draft or {}).get("status") or "unknown",
        "match": _count([post_sev.get(x["n"], "unknown") for x in p["match"]]),
        "new": _count([post_sev.get(x["n"], "unknown") for x in p["new"]]),
        "false": _count([g_sev.get(x["g"], "unknown") for x in p["false"]]),
        "dropped": _count([g_sev.get(x["g"], "unknown") for x in p["dropped"]]),
        "missed": _count(
            [
                (
                    x["severity"]
                    if x["severity"] != "unknown"
                    else post_sev.get(x["n"], "unknown")
                )
                for x in p["missed"]
            ]
        ),
        "prior": {
            "right": sum(1 for x in p["prior"] if x["ok"]),
            "wrong": sum(1 for x in p["prior"] if not x["ok"]),
        },
        "prior_by": by,
        "settled": counts or {},
        "judged": len(p["prior"]),
        "draft_findings": len(g_sev),
        "post_findings": len(post_sev),
    }
    for k in ("tokens", "requests", "wall_s", "late", "waits_s", "cached_tokens"):
        rec[k] = (draft or {}).get(k, 0)
    rec["phases"] = (draft or {}).get("phases") or {}
    rec["lines"] = p["lines"]
    if p["unparsed"]:
        rec["unparsed"] = p["unparsed"]
    return rec


def _read_json(path: pathlib.Path) -> dict:
    try:
        out = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return out if isinstance(out, dict) else {}


def _kept(cfg: dict, n: int, head: str, binding: str = "") -> pathlib.Path:
    root = (cfg.get("paths") or {}).get("root") or ""
    return (
        pathlib.Path(root)
        / "lessons"
        / "drafts"
        / (f"pr{n}_{head[:10]}" + (f"_{binding}" if binding else ""))
    )


def candidate_draft(cfg: dict, st: dict, n: int, head: str, cand: dict) -> dict:
    """What a candidate drafted for PR n at `head`: its draft record, its status at post time, its method, its prior
    rows and counts, and its draft's text when it has no record (the local worker's)."""
    out = {
        "draft": None,
        "status": "missing",
        "method": "",
        "rows": [],
        "counts": {},
        "text": "",
    }
    if cand["role"] == "shadow_drafter":
        d = draft_stage.shadow_for(st, n, head, cand["binding"])
        live = (
            None if d else draft_stage.draft_for(st, n, head, cand["binding"])
        )  # its draft as the live drafter
        e = draft_stage.shadow_pool(st).get(str(n))
        out["draft"] = d or live
        out["status"] = (
            out["draft"]["status"]
            if out["draft"] and out["draft"].get("status") in ("ready", "missing")
            else (
                "late"
                if e and draft_stage.same_commit(e.get("head"), head)
                else "missing"
            )
        )
        ws = (
            draft_stage.folder(cfg, n)
            if live
            else draft_stage.shadow_folder(cfg, n, cand["binding"])
        )
        kept = _kept(cfg, n, head, "" if live else cand["binding"])
    elif cand["binding"]:
        d = draft_stage.draft_for(st, n, head, cand["binding"])
        out["draft"], out["status"] = d, (d or {}).get("status") or "missing"
        ws, kept = draft_stage.folder(cfg, n), _kept(cfg, n, head)
    else:
        ws, kept = draft_stage.folder(cfg, n), None
        out["text"] = (
            (ws / "review_agent.md").read_text(encoding="utf-8")
            if (ws / "review_agent.md").is_file()
            else ""
        )
        out["status"] = "ready" if out["text"].strip() else "missing"
        out["method"] = LOCAL_METHOD
        if out["status"] == "ready":
            stats = _read_json(ws / "stats_agent.json")
            out["draft"] = {
                "status": "ready",
                "wall_s": float(stats.get("wall_s") or 0),
                "model": cand["model"],
            }
        return out
    ps = {}
    for folder in (ws, kept):
        data = prior_settled.load(folder) if folder else {}
        if data and draft_stage.same_commit(data.get("head"), head):
            ps = data
            break
    out["rows"], out["counts"] = ps.get("rows") or [], ps.get("counts") or {}
    out["method"] = (
        (out["draft"] or {}).get("method")
        or ps.get("method")
        or ("" if out["draft"] else _method_of(st, cand, []))
    )  # late or missing: its current method
    return out


def record(cfg: dict, store, n: int, spec: dict, ready: dict) -> list[str]:
    """At post time: each candidate's notes lines, scored against the post, into state/drafter_assessment.jsonl; the
    lines to print ([] when no candidate has lines or a draft)."""
    from .premises import verifier_name
    from .store import now

    VERIFIER = verifier_name(
        cfg
    )  # the model that verified the post, as its footer names it
    try:
        notes = pathlib.Path(spec["NOTES"]).read_text(encoding="utf-8")
    except (OSError, KeyError):
        notes = ""
    st = store.load()
    head = ready.get("sha") or ""
    out = []
    cands = bindings.candidates(cfg)
    for d in (
        st.get("drafts") or []
    ):  # a binding that drafted this head is scored too, whatever the roles say now
        if (
            d.get("pr") == n
            and draft_stage.same_commit(d.get("head"), head)
            and d.get("binding")
            and d.get("status") in ("ready", "missing")
            and d["binding"] not in {c["binding"] for c in cands}
        ):
            cands.append(
                {
                    "role": d.get("role") or "drafter",
                    "tag": d["binding"],
                    "binding": d["binding"],
                    "model": d.get("model") or "",
                }
            )
    for cand in cands:
        got = candidate_draft(cfg, st, n, head, cand)
        # With no line for it: the local worker is not scored; a live binding is, when it drafted; the shadow always
        # is (its draft ready, late or missing at post time).
        if not parse(notes, cand["tag"])["lines"] and (
            not cand["binding"] or (cand["role"] == "drafter" and not got["draft"])
        ):
            continue
        rec = build(
            n,
            spec.get("ROUND", ""),
            head,
            notes,
            ready.get("text") or "",
            got["draft"],
            cand["tag"],
            cand["model"],
            VERIFIER,
            now(),
            role=cand["role"],
            method=got["method"],
            status=got["status"],
            rows=got["rows"],
            counts=got["counts"],
            draft_text=got["text"],
            binding=cand["binding"],
        )
        with open(store.dir / FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.append(
            f"assessment recorded ({FILE}, {cand['tag']} {rec['candidate']['method']}): {summary(rec)}"
            + (
                f"; {len(rec['unparsed'])} line(s) not understood: {rec['unparsed'][0][:80]!r}"
                if rec.get("unparsed")
                else ""
            )
        )
    return out


def summary(rec: dict) -> str:
    def total(k):
        return sum((rec.get(k) or {}).values())

    by = rec.get("prior_by") or {}
    split = "".join(
        f"; {who} {v['right']} right, {v['wrong']} wrong"
        for who, v in sorted(by.items())
        if who in ("script", "model")
    )
    return (
        f"{total('match')} match, {total('new')} new, {total('false')} false, {total('dropped')} dropped, "
        f"{total('missed')} missed; priors {rec['prior']['right']} right, {rec['prior']['wrong']} wrong{split}"
        + (f"; draft {rec['draft']}" if rec.get("draft") in ("late", "missing") else "")
    )


# --- the card ---------------------------------------------------------------------------------------------------------


def records(store, tags: list[str] | None = None) -> list[dict]:
    """Every assessment record: the drafters' file, and the files the single drafter scored before kept (its tag's),
    whose records are the drafter role's with method "single-pass" when they say neither.
    """
    out = []
    try:
        out += store.read_jsonl(FILE, lenient=True)
    except OSError:
        pass
    for t in sorted(set(tags or [])):
        try:
            old = store.read_jsonl(file_name(t), lenient=True)
        except OSError:
            continue
        for r in old:
            c = r.setdefault("candidate", {})
            c.setdefault("role", "drafter")
            c.setdefault("tag", c.get("binding") or t)
            c.setdefault("method", OLD_METHOD)
        out += old
    return out


def _ratio(a: int, b: int) -> str:
    return f"{100 * a / b:.0f}% ({a} of {b})" if b else "no data yet"


def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _method_of(st: dict, cand: dict, recs: list[dict]) -> str:
    """A candidate's current method: its latest draft's (state drafts), else its latest record's."""
    if cand["role"] == "drafter" and not cand["binding"]:
        return LOCAL_METHOD
    for d in reversed(st.get("drafts") or []):
        if (
            d.get("binding") == cand["binding"]
            and (d.get("role") or "drafter") == cand["role"]
        ):
            return d.get("method") or OLD_METHOD
    for r in reversed(recs):
        c = r.get("candidate") or {}
        if c.get("tag") == cand["tag"] and c.get("role") == cand["role"]:
            return c.get("method") or OLD_METHOD
    return ""


def column(st: dict, cand: dict, recs: list[dict]) -> dict:
    """One candidate's scores on its current method, and what it leaves out."""
    method = _method_of(st, cand, recs)
    cur = [
        r
        for r in recs
        if (r.get("candidate") or {}).get("tag") == cand["tag"]
        and ((r.get("candidate") or {}).get("method") or OLD_METHOD) == method
    ]
    drafts = [
        d
        for d in st.get("drafts") or []
        if cand["binding"]
        and d.get("binding") == cand["binding"]
        and d.get("status") in ("ready", "missing")
    ]
    cur_drafts = [
        d
        for d in drafts
        if (d.get("method") or OLD_METHOD) == method
        and (d.get("role") or "drafter") == cand["role"]
    ]
    left: dict[str, int] = {}
    for d in drafts:
        m = d.get("method") or OLD_METHOD
        if m != method:
            left[m] = left.get(m, 0) + 1

    def sev_sum(keys, sevs):
        return sum((r.get(k) or {}).get(s, 0) for r in cur for k in keys for s in sevs)

    serious = ("red", "orange")
    found_s, all_s = sev_sum(("match", "new"), serious), sev_sum(
        ("match", "new", "missed"), serious
    )
    found_y, all_y = sev_sum(("match", "new"), ("yellow",)), sev_sum(
        ("match", "new", "missed"), ("yellow",)
    )
    assessed = [r for r in cur if r.get("draft") == "ready"]
    false = sum(sum((r.get("false") or {}).values()) for r in assessed)

    def by(who):
        right = sum(
            ((r.get("prior_by") or {}).get(who) or {}).get("right", 0) for r in cur
        )
        wrong = sum(
            ((r.get("prior_by") or {}).get(who) or {}).get("wrong", 0) for r in cur
        )
        return right, wrong

    settled = {
        k: sum(int((r.get("settled") or {}).get(k) or 0) for r in cur)
        for k in ("rows", "script", "script_pass", "model", "model_pass")
    }
    phased = any(r.get("settled") for r in cur)
    s_right, s_wrong = by("script")
    if phased:
        m_right, m_wrong = by("model")
    else:
        m_right = sum(r["prior"]["right"] for r in cur)
        m_wrong = sum(r["prior"]["wrong"] for r in cur)
    judged = sum(
        int(r["judged"]) if "judged" in r else r["prior"]["right"] + r["prior"]["wrong"]
        for r in cur
    )

    def per_phase(field: str, scale: float, unit: str) -> str:
        parts = []
        for name in ("settle", "prior", "hunt"):
            v = _mean(
                [
                    float((d.get("phases") or {}).get(name, {}).get(field))
                    for d in cur_drafts
                    if ((d.get("phases") or {}).get(name) or {}).get(field) is not None
                ]
            )
            if v is not None:
                parts.append(f"{name} {v / scale:.0f}{unit}")
        return ", ".join(parts)

    toks = per_phase("tokens", 1000, "k") or (
        f"{_mean([d['tokens'] for d in cur_drafts if d.get('tokens')]) / 1000:.0f}k"
        if any(d.get("tokens") for d in cur_drafts)
        else "—"
    )
    mins = per_phase("wall_s", 60, " min") or (
        f"{_mean([float(r['wall_s']) for r in cur if r.get('wall_s')]) / 60:.0f} min"
        if any(r.get("wall_s") for r in cur)
        else "—"
    )
    missing = sum(1 for r in cur if r.get("draft") == "missing")
    late = sum(1 for r in cur if r.get("draft") == "late" or r.get("late"))
    rows = [
        (
            "Prior: script statuses right",
            _ratio(s_right, s_right + s_wrong) if phased else "—",
        ),
        ("Prior: model statuses right", _ratio(m_right, m_right + m_wrong)),
        (
            "Prior rows: script / model (evidence pass)",
            (
                f"{settled['script']} / {settled['model']} ({_ratio(settled['model_pass'], settled['model'])})"
                if phased
                else "—"
            ),
        ),
        (
            "Prior rows the verifier judged",
            f"{judged} of {settled['rows']}" if phased else str(judged),
        ),
        ("New: 🔴/🟠 recall", _ratio(found_s, all_s)),
        ("New: 🟡 recall", _ratio(found_y, all_y)),
        (
            "New: false claims per draft",
            (
                f"{false / len(assessed):.1f} ({false} in {len(assessed)} drafts)"
                if assessed
                else "no data yet"
            ),
        ),
        ("Tokens per draft (by phase)", toks),
        ("Minutes per draft (by phase)", mins),
        (
            "Drafts missing or late at post time",
            f"{missing} missing, {late} late of {len(cur)}" if cur else "no data yet",
        ),
    ]
    label = f"{cand['tag']} ({method or 'no drafts yet'}{', shadow' if cand['role'] == 'shadow_drafter' else ', live'})"
    return {
        "label": label,
        "tag": cand["tag"],
        "role": cand["role"],
        "method": method,
        "assessed": len(cur),
        "drafts": len(cur_drafts),
        "left_out": left,
        "rows": rows,
    }


def card(cfg: dict, st: dict, store) -> dict | None:
    """The card's columns (the shadow drafter first, then the live drafter) and rows; None when there is nothing to
    show (no drafter binding and no record)."""
    from .premises import verifier_name

    VERIFIER = verifier_name(cfg)
    cands = bindings.candidates(cfg)
    tag = bindings.tag(cfg) or next(
        (
            d.get("binding")
            for d in reversed(st.get("drafts") or [])
            if d.get("binding")
        ),
        "",
    )
    if not tag:
        return None
    if not any(c["tag"] == tag for c in cands):
        cands.insert(0, {"role": "drafter", "tag": tag, "binding": tag, "model": ""})
    cands.sort(key=lambda c: c["role"] != "shadow_drafter")
    recs = records(store, [c["tag"] for c in cands if c["binding"]])
    cols = [column(st, c, recs) for c in cands]
    if (
        not recs
        and not any(c["drafts"] for c in cols)
        and not bindings.binding(cfg, "drafter")
        and not bindings.binding(cfg, "shadow_drafter")
    ):
        return None
    first = cols[0]
    left = ", ".join(f"{k} {v}" for k, v in sorted(first["left_out"].items()))
    return {
        "title": f"{tag.upper()} drafter vs {bindings.reference_name(VERIFIER)}",
        "tag": tag,
        "model": bindings.display(cfg, cols[0]["role"]) or bindings.display(cfg),
        "assessed": first["assessed"],
        "drafts": first["drafts"],
        "method": first["method"],
        "left_out": (
            f"{sum(first['left_out'].values())} {tag} drafts of an earlier method left out ({left})"
            if left
            else ""
        ),
        "columns": [c["label"] for c in cols],
        "tags": [c["tag"] for c in cols],
        "rows": [
            (name, [c["rows"][i][1] for c in cols])
            for i, (name, _v) in enumerate(cols[0]["rows"])
        ],
        "pool": sorted(int(k) for k in draft_stage.pool(st)),
        "shadows": sorted(int(k) for k in draft_stage.shadow_pool(st)),
    }


def card_md(cfg: dict, st: dict, store) -> list[str]:
    c = card(cfg, st, store)
    if not c:
        return []
    lines = " and ".join(f"`{t} ...`" for t in c["tags"])
    out = [
        f"## {c['title']}",
        "",
        f"{c['assessed']} posts assessed, {c['drafts']} {c['tag']} {c['method'] or 'current-method'} drafts; each "
        f"drafter's draft against the verified post (notes.md {lines} lines)."
        + (f" {c['left_out']}." if c["left_out"] else "")
        + (
            f" Pool drafts: {', '.join('#%d' % n for n in c['pool'])}."
            if c["pool"]
            else ""
        )
        + (
            f" Shadow drafts running: {', '.join('#%d' % n for n in c['shadows'])}."
            if c["shadows"]
            else ""
        ),
        "",
        "| Measure | " + " | ".join(c["columns"]) + " |",
        "|---|" + "---|" * len(c["columns"]),
    ]
    return out + [f"| {a} | " + " | ".join(vals) + " |" for a, vals in c["rows"]] + [""]
