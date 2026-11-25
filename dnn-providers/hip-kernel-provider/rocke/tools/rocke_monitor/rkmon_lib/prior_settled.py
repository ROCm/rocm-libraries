# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""prior_settled.md: a draft's prior rows as the verifier reads them, from draft_findings.json and the evidence check's
results (`rk evidence-check` writes it after its own report), and prior_settled.json, the same per row for the draft
record (draft_stage.record) and the assessment.

Two groups:
- settled: the rows whose evidence passed (PASS) and whose status flips no 🔴/🟠 (worker/prior_settle.py `flips`): they
  stand; one of them, picked at random from the head (an approval only when nothing else settled), is named for a spot
  check;
- needs a look: the rows whose evidence failed, was not re-run or is missing, the rows whose status flips a 🔴/🟠
  against our last review (one opens, closes, or moves between 🔴 and 🟠), and the rows no phase settled.
The items that hold no review point of their own come last, with why.
"""

from __future__ import annotations

import json
import pathlib
import random

FILE_MD, FILE_JSON = "prior_settled.md", "prior_settled.json"


def _json(path: pathlib.Path) -> dict:
    try:
        out = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return out if isinstance(out, dict) else {}


def build(found: dict, report: dict) -> dict:
    """Each prior row with its evidence check's result and its group; the counts; the spot check."""
    results = {c["id"]: c for c in report.get("claims") or []}
    rows = []
    for p in found.get("priors") or []:
        c = results.get(p["id"]) or {}
        res = c.get("result") or "MISSING"
        if not p.get("settler"):
            group, why = "look", f"not settled: {p.get('why') or 'no phase settled it'}"
        elif res != "PASS":
            group, why = "look", f"evidence {res}" + (
                f": {c.get('why')}" if c.get("why") else ""
            )
        elif p.get("flip"):
            group, why = "look", (
                f"flips a 🔴/🟠: {p.get('prev_light') or '—'} {p.get('prev_status') or 'new'} → "
                f"{p.get('light')} {p.get('status')}"
            )
        else:
            group, why = "settled", ""
        rows.append(
            {
                k: p.get(k)
                for k in (
                    "id",
                    "key",
                    "light",
                    "status",
                    "settler",
                    "how",
                    "ref",
                    "prev_light",
                    "prev_status",
                    "flip",
                    "kind",
                )
            }
            | {"evidence": res, "group": group, "why": why}
        )
    settled = [r["id"] for r in rows if r["group"] == "settled"]
    worth = [
        r["id"] for r in rows if r["group"] == "settled" and r.get("how") != "data"
    ] or settled
    spot = (
        random.Random(f"{found.get('pr')}:{found.get('head')}").choice(worth)
        if worth
        else ""
    )
    return {
        "pr": found.get("pr"),
        "head": found.get("head"),
        "round": found.get("round"),
        "role": found.get("role"),
        "binding": found.get("binding"),
        "method": found.get("method"),
        "rows": rows,
        "spot": spot,
        "skipped": found.get("skipped") or {},
        "ours": found.get("ours"),
        "counts": tally(rows),
    }


def tally(rows: list[dict]) -> dict:
    """The counts the draft record and the assessment keep: rows, settled by script and by the model (and how many of
    each passed their evidence check), to look at, not settled."""
    out = {
        "rows": len(rows),
        "script": 0,
        "script_pass": 0,
        "model": 0,
        "model_pass": 0,
        "look": 0,
        "unsettled": 0,
    }
    for r in rows:
        if r.get("settler") in ("script", "model"):
            out[r["settler"]] += 1
            out[r["settler"] + "_pass"] += r.get("evidence") == "PASS"
        else:
            out["unsettled"] += 1
        out["look"] += r.get("group") == "look"
    return out


def render(data: dict) -> str:
    head = str(data.get("head") or "")[:10]
    rows = data["rows"]
    settled = [r for r in rows if r["group"] == "settled"]
    look = [r for r in rows if r["group"] == "look"]
    c = data["counts"]
    lines = [
        f"# Prior items, settled and to look at: #{data.get('pr')} {data.get('round') or ''} at {head}",
        "",
        f"{c['rows']} rows: {c['script']} settled by script, {c['model']} by the model ({c['model_pass']} of them "
        f"with passing evidence), {c['unsettled']} not settled; {len(look)} to look at.",
        "",
        f"## Settled with passing evidence ({len(settled)})",
        "",
        "These stand. "
        + (
            f"Spot-check one: **{data['spot']}** (picked at random)."
            if data["spot"]
            else "None to spot-check."
        ),
        "",
        "| Row | Prior item | Status | Settled by | Evidence |",
        "|---|---|---|---|---|",
    ]
    for r in settled:
        lines.append(
            f"| {r['id']} | {r['key']} | {r['light']} {r['status']} | {_by(r)} | {r.get('ref') or '—'} |"
        )
    lines += [
        "",
        f"## Needs a look ({len(look)})",
        "",
        "Re-derive these: their evidence failed or is missing, their status flips a 🔴/🟠 against our last review, "
        "or no phase settled them.",
        "",
        "| Row | Prior item | Status | Was | Why |",
        "|---|---|---|---|---|",
    ]
    for r in look:
        was = (
            f"{r.get('prev_light') or ''} {r.get('prev_status') or ''}".strip() or "new"
        )
        lines.append(
            f"| {r['id']} | {r['key']} | {r['light']} {r['status']} | {was} | "
            f"{str(r['why']).replace('|', '/')[:200]} |"
        )
    skipped = data.get("skipped") or {}
    if skipped:
        lines += [
            "",
            f"Not prior items ({len(skipped)}, no review point of their own): "
            + "; ".join(f"{k} {v}" for k, v in skipped.items()),
        ]
    ours = data.get("ours")
    how = "kept unposted" if (ours or {}).get("kept") else "posted"
    lines += [
        "",
        (
            f"Our last review: at `{str(ours.get('head'))[:10]}` ({how} {ours.get('at') or '?'})."
            if ours
            else "No review of ours was staged: the draft's prior rows hold no item of ours."
        ),
    ]
    return "\n".join(lines) + "\n"


def _by(r: dict) -> str:
    return (
        f"script: {r.get('how')}"
        if r.get("settler") == "script"
        else str(r.get("settler") or "—")
    )


def write(folder: pathlib.Path, report: dict) -> dict | None:
    """prior_settled.md and .json in a draft's folder, from its draft_findings.json and the evidence check's report;
    None when the draft has no prior rows on record."""
    folder = pathlib.Path(folder)
    found = _json(folder / "draft_findings.json")
    if "priors" not in found:
        return None
    data = build(found, report)
    (folder / FILE_JSON).write_text(
        json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    (folder / FILE_MD).write_text(render(data), encoding="utf-8")
    return data


def load(folder: pathlib.Path) -> dict:
    return _json(pathlib.Path(folder) / FILE_JSON)


def counts(folder: pathlib.Path) -> dict:
    """The draft's prior counts (tally), {} when it has none on record."""
    return load(folder).get("counts") or {}
