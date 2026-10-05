#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Write a self-contained HTML report of a pipeline run.

Usage: make_report.py <run_dir> [out.html]     (default: <run_dir>/report.html)

Reads the run's artifacts (round_*/stage04b/decisions.json,
round_*/stage05/{cells,metrics}.json, round_*/stage06/manifest.json,
round_*/stage01/shapes.yaml, validate/stage07/summary.json,
validate/stage08/summary.json, timing/*.json and the latest
configs/*.source.yaml) and writes nothing but the output file. A section
whose inputs are missing shows "n/a". The page has no external references.
"""
import glob
import html
import json
import math
import os
import re
import sys


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def read_text(path):
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def _num(v):
    return (
        float(v)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
        else None
    )


def _first_pos(*vals):
    for v in vals:
        v = _num(v)
        if v is not None and v > 0:
            return v
    return None


def _pct(v):
    v = _num(v)
    return round(v * 100, 2) if v is not None else None


def script_json(obj):
    """JSON safe to embed in an inline <script>."""
    text = json.dumps(obj, allow_nan=False, default=str)
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _finite(obj):
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {str(k): _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


def esc(s):
    return html.escape("" if s is None else str(s), quote=True)


# ── shape categories ────────────────────────────────────────────────────────

SHAPE_CATEGORIES = (
    "Small GEMMs",
    "Medium GEMMs",
    "Large GEMMs",
    "Large M, smaller N and K",
    "Large M, very small N and K",
    "Large N, smaller M and K",
    "Large N, very small M and K",
    "Large K, smaller M and N",
    "Large K, very small M and N",
    "Large M and N",
    "Large N and K",
    "Large M and K",
    "Very Large GEMMs",
    "M=1",
    "N=1",
    "K=1",
    "Small batch",
    "Medium batch",
    "Large batch",
    "Very large batch",
)


def categorize_gemm(m, n, k, batch):
    if batch > 8192:
        return "Very large batch"
    if batch > 1024:
        return "Large batch"
    if batch > 128:
        return "Medium batch"
    if batch >= 2:
        return "Small batch"
    if m == 1:
        return "M=1"
    if n == 1:
        return "N=1"
    if k == 1:
        return "K=1"
    large_m, large_n, large_k = m > 8192, n > 8192, k > 8192
    if large_m and large_n and large_k:
        return "Very Large GEMMs"
    if large_n and large_k:
        return "Large N and K"
    if large_m and large_k:
        return "Large M and K"
    if large_m and large_n:
        return "Large M and N"
    if large_m:
        if n <= 128 and k <= 128:
            return "Large M, very small N and K"
        return "Large M, smaller N and K"
    if large_n:
        if m <= 128 and k <= 128:
            return "Large N, very small M and K"
        return "Large N, smaller M and K"
    if large_k:
        if m <= 128 and n <= 128:
            return "Large K, very small M and N"
        return "Large K, smaller M and N"
    if m <= 1024 and n <= 1024 and k <= 1024:
        return "Small GEMMs"
    if m >= 4096 and n >= 4096 and k >= 4096:
        return "Large GEMMs"
    return "Medium GEMMs"


# ── run-dir readers ─────────────────────────────────────────────────────────


def round_num(path):
    m = re.fullmatch(r"round_(\d+)", os.path.basename(path.rstrip("/")))
    return int(m.group(1)) if m else -1


def round_dirs(run_dir):
    dirs = [d for d in glob.glob(os.path.join(run_dir, "round_*")) if round_num(d) >= 0]
    return sorted(dirs, key=round_num)


def _leaves(labels):
    labels = set(labels)
    return {lab for lab in labels if not any(o.startswith(lab + "#") for o in labels)}


def _wgeo(pairs):
    ls = ws = 0.0
    for v, n in pairs:
        v = _num(v)
        if v is not None and v > 0 and n and n > 0:
            ls += math.log(v) * n
            ws += n
    return math.exp(ls / ws) if ws > 0 else None


def held_out_rounds(run_dir):
    """Per active round: the prior model on that round's new shapes."""
    rows = []
    for d in round_dirs(run_dir):
        dec = load_json(os.path.join(d, "stage04b", "decisions.json"))
        if not isinstance(dec, dict):
            continue
        cells = [c for c in dec.get("per_cell") or [] if isinstance(c, dict)]
        agg = dec.get("aggregate") if isinstance(dec.get("aggregate"), dict) else {}
        model = _num(agg.get("global_sel_eff_new"))
        origami = _num(agg.get("global_origami_sel_eff_new"))
        if model is None:
            model = _wgeo((c.get("sel_eff_new"), c.get("n_eval") or 0) for c in cells)
        if origami is None:
            origami = _wgeo(
                (c.get("origami_sel_eff_new"), c.get("origami_n_eval") or 0)
                for c in cells
            )
        n = agg.get("n_gemms_evaluated")
        if not isinstance(n, int):
            n = sum(int(c.get("n_eval") or 0) for c in cells)
        if model is None:
            continue
        rows.append(
            {
                "round": os.path.basename(d),
                "model": _pct(model),
                "origami": _pct(origami),
                "n_gemms": n,
            }
        )
    return rows


def training_rounds(run_dir):
    """Per round: stage05 selection efficiency on the cells it trained."""
    rows = []
    for d in round_dirs(run_dir):
        met = load_json(os.path.join(d, "stage05", "metrics.json"))
        if not isinstance(met, dict) or _num(met.get("global_sel_eff")) is None:
            continue
        rows.append(
            {
                "round": os.path.basename(d),
                "model": _pct(met.get("global_sel_eff")),
                "deployed": _pct(met.get("global_deployed_sel_eff")),
                "origami": _pct(met.get("global_origami_sel_eff")),
                "n_cells": met.get("n_cells_trained"),
                "n_gemms": met.get("n_gemms_trained"),
            }
        )
    return rows


def _split_parents(run_dir):
    parents = set()
    for d in round_dirs(run_dir):
        s = load_json(os.path.join(d, "stage04b", "splits.json"))
        splits = s.get("splits") if isinstance(s, dict) else None
        if isinstance(splits, dict):
            parents.update(splits)
    return parents


def held_out_breakdown(run_dir):
    """Latest held-out measurement per leaf cell (the orchestrator's
    held_out_sel_eff_breakdown.json when present, else rebuilt in memory
    from every decisions.json)."""
    b = load_json(os.path.join(run_dir, "held_out_sel_eff_breakdown.json"))
    if isinstance(b, dict) and isinstance(b.get("per_cell"), dict) and b["per_cell"]:
        return b["per_cell"]
    per_cell = {}
    for d in round_dirs(run_dir):
        dec = load_json(os.path.join(d, "stage04b", "decisions.json"))
        if not isinstance(dec, dict):
            continue
        for c in dec.get("per_cell") or []:
            if not isinstance(c, dict) or not c.get("cell"):
                continue
            se, n = _num(c.get("sel_eff_new")), int(c.get("n_eval") or 0)
            if se is None or se <= 0 or n <= 0:
                continue
            per_cell[str(c["cell"])] = {
                "round": os.path.basename(d),
                "sel_eff_new": se,
                "n_eval": n,
                "origami_sel_eff_new": _num(c.get("origami_sel_eff_new")),
                "origami_n_eval": int(c.get("origami_n_eval") or 0),
                "winner_us_geomean": _num(c.get("winner_us_geomean")),
                "pick_us_geomean": _num(c.get("pick_us_geomean")),
                "origami_pick_us_geomean": _num(c.get("origami_pick_us_geomean")),
            }
    for p in _split_parents(run_dir):
        per_cell.pop(p, None)
    return per_cell


def held_out_aggregate(per_cell):
    leaves = _leaves(per_cell)
    mo = _wgeo(
        (per_cell[c].get("sel_eff_new"), per_cell[c].get("n_eval")) for c in leaves
    )
    oo = _wgeo(
        (per_cell[c].get("origami_sel_eff_new"), per_cell[c].get("origami_n_eval"))
        for c in leaves
    )
    both = [
        c
        for c in leaves
        if _num(per_cell[c].get("sel_eff_new"))
        and _num(per_cell[c].get("origami_sel_eff_new"))
    ]
    best = _wgeo(
        (
            max(per_cell[c]["sel_eff_new"], per_cell[c]["origami_sel_eff_new"]),
            per_cell[c].get("n_eval"),
        )
        for c in both
    )
    wins = sum(
        1
        for c in both
        if per_cell[c]["sel_eff_new"] >= per_cell[c]["origami_sel_eff_new"]
    )
    return {
        "model": _pct(mo),
        "origami": _pct(oo),
        "delta": (
            round((mo - oo) * 100, 2) if mo is not None and oo is not None else None
        ),
        "router_bound": _pct(best),
        "model_win_frac": round(wins / len(both) * 100, 1) if both else None,
        "n_cells": len(leaves),
    }


def final_cells(run_dir, held):
    """Deployed leaf cells: the latest cells.json entry per label over all
    rounds, with held-out values (inherited from the nearest measured
    ancestor for leaves created by the last split)."""
    by_label = {}
    for d in round_dirs(run_dir):
        cj = load_json(os.path.join(d, "stage05", "cells.json"))
        if not isinstance(cj, dict):
            continue
        for c in cj.get("cells") or []:
            if isinstance(c, dict) and c.get("label"):
                by_label[str(c["label"])] = c
    leaves = _leaves(by_label)

    def inherit(lab, key):
        cur = lab
        while True:
            v = (held.get(cur) or {}).get(key)
            if _num(v) is not None:
                return _num(v), cur != lab
            j = cur.rfind("#")
            if j < 0:
                return None, False
            cur = cur[:j]

    cells = []
    for lab in sorted(leaves):
        c = by_label[lab]
        dep = c.get("deployed") if isinstance(c.get("deployed"), dict) else {}
        ori = c.get("origami") if isinstance(c.get("origami"), dict) else {}
        model = _pct(c.get("best_sel_eff"))
        deployed = _pct(dep.get("sel_eff"))
        origami = _pct(ori.get("sel_eff"))
        held_v, inherited = inherit(lab, "sel_eff_new")
        held_o, _ = inherit(lab, "origami_sel_eff_new")
        held_w, _ = inherit(lab, "winner_us_geomean")
        held_p, _ = inherit(lab, "pick_us_geomean")
        held_op, _ = inherit(lab, "origami_pick_us_geomean")
        hv, ho = _pct(held_v), _pct(held_o)

        def diff(a, b):
            return round(a - b, 2) if a is not None and b is not None else None

        cells.append(
            {
                "cell": lab,
                "n_gemms": int(c.get("n_gemms") or 0),
                "model": model,
                "deployed": deployed,
                "origami": origami,
                "k_delta": diff(deployed, model),
                "dep_delta": diff(deployed, origami),
                "held": hv,
                "held_inherited": inherited,
                "held_ori": ho,
                "held_delta": diff(hv, ho),
                "smartK": len(c.get("smart_k_signatures") or []) or None,
                "cap": f"{c.get('embed_dim')}/{c.get('hidden_dim')}/{c.get('inter_hidden')}",
                "held_w_us": held_w,
                "held_pick_us": held_p,
                "held_ori_pick_us": held_op,
            }
        )
    return cells


def stage07(run_dir):
    """Per bench yaml of the latest stage07 run: pick parity and the paired
    selection-time deltas per request size."""
    summ = load_json(os.path.join(run_dir, "validate", "stage07", "summary.json"))
    if not isinstance(summ, dict):
        return None
    out = {
        "ok": bool(summ.get("ok")),
        "failures": len(summ.get("failures") or []),
        "yamls": [],
    }
    for tag, y in sorted((summ.get("per_yaml") or {}).items()):
        if not isinstance(y, dict):
            continue
        row = {"yaml": tag, "n_kept": y.get("n_kept"), "n_entries": y.get("n_entries")}
        par = y.get("parity") if isinstance(y.get("parity"), dict) else None
        if par:
            row["parity"] = {
                "agree": par.get("n_agree"),
                "n": par.get("n_problems"),
                "rate": _pct(par.get("match_rate")),
                "counts": par.get("counts") or {},
                "ok": par.get("ok"),
            }
        timing = []
        for rsn, t in sorted((y.get("timing") or {}).items(), key=lambda kv: kv[0]):
            if not isinstance(t, dict):
                continue
            d = t.get("delta_us") or {}
            timing.append(
                {
                    "rsn": rsn.replace("rsn", ""),
                    "n": d.get("n"),
                    "reps": t.get("repetitions"),
                    "on_p50": (t.get("ml_on_us") or {}).get("p50"),
                    "off_p50": (t.get("ml_off_us") or {}).get("p50"),
                    "d_p10": d.get("p10"),
                    "d_p50": d.get("p50"),
                    "d_p90": d.get("p90"),
                    "d_p99": d.get("p99"),
                }
            )
        row["timing"] = timing
        out["yamls"].append(row)
    return out


def stage08(run_dir):
    """Every dataset of the latest stage08 run: the GEMM-weighted model and
    Origami selection efficiency and the paired statistics of its summary's
    `overall` block."""
    summ = load_json(os.path.join(run_dir, "validate", "stage08", "summary.json"))
    if not isinstance(summ, dict):
        return []
    rows = []
    for x in summ.get("datasets") or []:
        if not isinstance(x, dict):
            continue
        s = x.get("summary") if isinstance(x.get("summary"), dict) else {}
        ov = s.get("overall") if isinstance(s.get("overall"), dict) else {}
        paired = ov.get("paired") if isinstance(ov.get("paired"), dict) else {}
        rows.append(
            {
                "name": x.get("name"),
                "rc": x.get("rc"),
                "n": ov.get("n_evaluated", ov.get("n_gemms")),
                "model": _pct(
                    _first_pos(
                        ov.get("sel_eff"),
                        s.get("model_geomean_gemm_weighted"),
                        x.get("model_geomean_sel_eff"),
                    )
                ),
                "origami": _pct(
                    _first_pos(
                        ov.get("origami_sel_eff"),
                        s.get("origami_geomean_gemm_weighted"),
                        x.get("origami_geomean_sel_eff"),
                    )
                ),
                "paired_n": paired.get("n"),
                "wins": paired.get("wins"),
                "ties": paired.get("ties"),
                "losses": paired.get("losses"),
                "speedup": _num(paired.get("speedup_geomean")),
                "per_cell": _stage08_cells(run_dir, x),
            }
        )
    return rows


def _stage08_cells(run_dir, ds):
    path = ds.get("per_cell_json")
    if not path:
        return []
    if not os.path.isabs(path):
        path = os.path.join(run_dir, "validate", "stage08", path)
    data = load_json(path)
    if isinstance(data, dict):
        data = data.get("per_cell", data.get("cells"))
    if isinstance(data, dict):
        data = [dict(v, cell=k) for k, v in data.items() if isinstance(v, dict)]
    out = []
    for r in data or []:
        if not isinstance(r, dict) or not r.get("cell"):
            continue
        mo = _pct(_first_pos(r.get("sel_eff"), r.get("model_sel_eff")))
        og = _pct(_first_pos(r.get("origami_sel_eff")))
        if mo is not None and og is not None:
            out.append([str(r["cell"]), mo, og])
    return out


def deployed_size_mb(run_dir):
    for d in reversed(round_dirs(run_dir)):
        man = load_json(os.path.join(d, "stage06", "manifest.json"))
        size = (
            (man or {}).get("file", {}).get("size") if isinstance(man, dict) else None
        )
        if isinstance(size, int) and size > 0:
            return round(size / 1e6, 2)
    return None


def timing_rows(run_dir):
    """Latest wall-clock per (round, stage) over every invocation's
    timing/*.json, and the summed invocation totals."""
    latest = {}
    total = 0.0
    for p in sorted(glob.glob(os.path.join(run_dir, "timing", "*.json"))):
        t = load_json(p)
        if not isinstance(t, dict):
            continue
        total += _num(t.get("total_s")) or 0.0
        for r in t.get("rows") or []:
            if isinstance(r, dict) and r.get("stage"):
                latest[(r.get("round"), str(r["stage"]))] = r
    rounds = {}
    for (rnd, stage), r in latest.items():
        rounds.setdefault(rnd, {})[stage] = r
    return rounds, total


def fmt_dur(s):
    s = int(round(s or 0))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def config_source(run_dir):
    srcs = sorted(glob.glob(os.path.join(run_dir, "configs", "*.source.yaml")))
    return read_text(srcs[-1]) if srcs else ""


def shapes_total(run_dir):
    n = 0
    for d in round_dirs(run_dir):
        txt = read_text(os.path.join(d, "stage01", "shapes.yaml"))
        n += sum(1 for line in txt.splitlines() if line.lstrip().startswith("-"))
    return n or None


# ── canonical cell order ────────────────────────────────────────────────────

_MN_TIER_RANK = {"Tiny": 0, "Small": 1, "Mid": 2, "Large": 3}
_K_TIER_RANK = {"TinyK": 0, "MidK": 1, "LargeK": 2}
_BATCH_RANK = {"Bany": 0, "Bnone": 1}
_AXIS_RANK = {"M": 0, "N": 1, "K": 2, "B": 3}


def base_cell_key(base):
    parts = (base or "").split("|")
    if len(parts) < 4:
        return (1, (base or "",))
    m, n, k, b = parts[:4]
    return (
        0,
        (
            _MN_TIER_RANK.get(m, 99),
            m,
            _MN_TIER_RANK.get(n, 99),
            n,
            _K_TIER_RANK.get(k, 99),
            k,
            _BATCH_RANK.get(b, 99),
            b,
        ),
    )


def seg_key(seg):
    m = re.match(r"([A-Za-z]+)(<=|>)(\d+)$", seg or "")
    if not m:
        return (1, 99, 0, 0, seg or "")
    return (0, _AXIS_RANK.get(m.group(1), 99), int(m.group(3)), m.group(2) != "<=", "")


def cell_tree_ascii(cells):
    root = {}
    for c in cells:
        base, *segs = (c["cell"] or "").split("#")
        node = root.setdefault(base, {})
        for s in segs:
            node = node.setdefault(s, {})
        node["__leaf__"] = c

    def stat(leaf):
        if not leaf:
            return ""
        parts = [f"n={leaf['n_gemms']:,}"]
        if leaf.get("held") is not None:
            parts.append(f"held-out={leaf['held']:.1f}%")
        if leaf.get("smartK"):
            parts.append(f"K={leaf['smartK']}")
        return "   [" + "  ".join(parts) + "]"

    lines = []

    def render(node, prefix):
        kids = sorted(
            ((k, v) for k, v in node.items() if k != "__leaf__"),
            key=lambda kv: seg_key(kv[0]),
        )
        for i, (k, v) in enumerate(kids):
            last = i == len(kids) - 1
            lines.append(
                prefix
                + ("\u2514\u2500\u2500 " if last else "\u251c\u2500\u2500 ")
                + k
                + stat(v.get("__leaf__"))
            )
            render(v, prefix + ("    " if last else "\u2502   "))

    for base in sorted(root, key=base_cell_key):
        node = root[base]
        is_leaf = not [k for k in node if k != "__leaf__"]
        lines.append(base + (stat(node.get("__leaf__")) if is_leaf else ""))
        render(node, "")
        lines.append("")
    return "\n".join(lines).rstrip()


# ── inline SVG charts ───────────────────────────────────────────────────────

_CHART_TEXT = "#1f2328"
_CHART_MUT = "#57606a"
_CHART_GRID = "#eaeef2"
_C_MODEL = "#0969da"
_C_DEPLOYED = "#1a7f37"
_C_ORIGAMI = "#9a6700"


def _short(s, n=14):
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "."


def _nice_top(v):
    iv = int(v) + (1 if v > int(v) else 0)
    iv = max(1, iv)
    return ((iv + 4) // 5) * 5


def figure(caption, svg):
    if not svg:
        return ""
    return (
        f'<figure class="chart"><figcaption>{esc(caption)}</figcaption>{svg}</figure>'
    )


def svg_bars(labels, series, width=760, height=240):
    series = [(n, vs, c) for (n, vs, c) in series if any(v is not None for v in vs)]
    flat = [v for _, vs, _ in series for v in vs if v is not None]
    if not labels or not series or not flat:
        return ""
    top = float(_nice_top(max(flat))) or 1.0
    ml, mr, mt, mb = 46, 12, 26, 38
    x0, y1 = ml, height - mb
    pw, ph = (width - mr) - ml, y1 - mt
    out = [
        f'<svg viewBox="0 0 {width} {height}" role="img" xmlns="http://www.w3.org/2000/svg">'
    ]
    for frac in (0.0, 0.5, 1.0):
        gy = y1 - ph * frac
        out.append(
            f'<line x1="{x0}" y1="{gy:.1f}" x2="{width - mr}" y2="{gy:.1f}" stroke="{_CHART_GRID}" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{x0 - 6}" y="{gy + 3:.1f}" text-anchor="end" font-size="10" fill="{_CHART_MUT}">{top * frac:g}</text>'
        )
    gw = pw / len(labels)
    bw = (gw * 0.8) / len(series)
    for gi, lab in enumerate(labels):
        gx = x0 + gw * gi
        for si, (name, vs, color) in enumerate(series):
            v = vs[gi] if gi < len(vs) else None
            if v is None:
                continue
            bh = ph * (max(0.0, min(float(v), top)) / top)
            col = color[gi] if isinstance(color, (list, tuple)) else color
            out.append(
                f'<rect x="{gx + gw * 0.1 + bw * si:.1f}" y="{y1 - bh:.1f}" width="{bw * 0.92:.1f}" '
                f'height="{bh:.1f}" fill="{col}"><title>{esc(lab)} {esc(name)}: {v:g}</title></rect>'
            )
        out.append(
            f'<text x="{gx + gw / 2:.1f}" y="{y1 + 13:.1f}" text-anchor="middle" font-size="10" fill="{_CHART_MUT}">{esc(_short(lab))}</text>'
        )
    out.append("</svg>")
    return "".join(out)


def svg_lines(labels, series, width=760, height=240, y_suffix=""):
    series = [(n, vs, c) for (n, vs, c) in series if any(v is not None for v in vs)]
    flat = [v for _, vs, _ in series for v in vs if v is not None]
    if not labels or not series or not flat:
        return ""
    lo, hi = min(flat), max(flat)
    if hi <= lo:
        lo, hi = lo - 1, hi + 1
    pad = (hi - lo) * 0.15
    y_lo, y_hi = lo - pad, hi + pad
    ml, mr, mt, mb = 46, 12, 26, 38
    x0, y1 = ml, height - mb
    pw, ph = (width - mr) - ml, y1 - mt
    n = len(labels)

    def xp(i):
        return x0 + (pw * i / (n - 1) if n > 1 else pw / 2)

    def yp(v):
        return y1 - ph * (v - y_lo) / (y_hi - y_lo)

    out = [
        f'<svg viewBox="0 0 {width} {height}" role="img" xmlns="http://www.w3.org/2000/svg">'
    ]
    for frac in (0.0, 0.5, 1.0):
        gy = y1 - ph * frac
        out.append(
            f'<line x1="{x0}" y1="{gy:.1f}" x2="{width - mr}" y2="{gy:.1f}" stroke="{_CHART_GRID}" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{x0 - 6}" y="{gy + 3:.1f}" text-anchor="end" font-size="10" fill="{_CHART_MUT}">{y_lo + (y_hi - y_lo) * frac:.0f}{esc(y_suffix)}</text>'
        )
    for i, lab in enumerate(labels):
        out.append(
            f'<text x="{xp(i):.1f}" y="{y1 + 13:.1f}" text-anchor="middle" font-size="10" fill="{_CHART_MUT}">{esc(_short(lab))}</text>'
        )
    for name, vs, color in series:
        pres = [(xp(i), yp(v), v) for i, v in enumerate(vs) if v is not None]
        if len(pres) >= 2:
            pts = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in pres)
            out.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/>'
            )
        for x, y, v in pres:
            out.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" fill="{color}"><title>{esc(name)}: {v:g}{esc(y_suffix)}</title></circle>'
            )
    lx = x0
    for name, _vs, color in series:
        out.append(
            f'<rect x="{lx}" y="{mt - 16}" width="9" height="9" fill="{color}"/>'
        )
        out.append(
            f'<text x="{lx + 13}" y="{mt - 8}" font-size="10" fill="{_CHART_TEXT}">{esc(name)}</text>'
        )
        lx += 24 + len(name) * 6
    out.append("</svg>")
    return "".join(out)


def chart_trend(train):
    if not train:
        return ""
    labels = [r["round"].replace("round_", "r") for r in train]
    svg = svg_lines(
        labels,
        [
            ("model", [r.get("model") for r in train], _C_MODEL),
            ("deployed", [r.get("deployed") for r in train], _C_DEPLOYED),
            ("origami", [r.get("origami") for r in train], _C_ORIGAMI),
        ],
        y_suffix="%",
    )
    return figure(
        "Selection efficiency per round on the cells trained that round (stage05, in-sample).",
        svg,
    )


_HIST_EDGES = [0, 70, 80, 85, 88, 90, 95, 100.0001]
_HIST_LABELS = ["<70", "70-80", "80-85", "85-88", "88-90", "90-95", ">=95"]
_HIST_COLORS = [
    "#cf222e",
    "#cf222e",
    "#bf8700",
    "#bf8700",
    "#bf8700",
    "#2da44e",
    "#1a7f37",
]


def chart_hist(cells, key, caption):
    vals = [c[key] for c in cells or [] if c.get(key) is not None]
    if not vals:
        return ""
    counts = [0] * len(_HIST_LABELS)
    for v in vals:
        for bi in range(len(_HIST_LABELS)):
            if _HIST_EDGES[bi] <= v < _HIST_EDGES[bi + 1]:
                counts[bi] += 1
                break
    svg = svg_bars(_HIST_LABELS, [("cells", counts, _HIST_COLORS)])
    return figure(f"{caption} ({len(vals)} cells; bar height = cells).", svg)


def _pipeline_on_path():
    pipeline_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if pipeline_root not in sys.path:
        sys.path.insert(0, pipeline_root)


def _tick(v):
    return f"{v // 1024}k" if v >= 1024 and v % 1024 == 0 else str(v)


def cell3d_images(labels):
    """3D (M/N/K) cell maps per batch tier as PNG data URIs; {} without
    matplotlib."""
    try:
        import base64
        import io

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        return {}
    leaves = sorted(_leaves(labels))
    if not leaves:
        return {}
    _pipeline_on_path()
    from lib.grid import TIER_RANGES_K as KK
    from lib.grid import TIER_RANGES_MN as MN

    def tiers(lab):
        a = lab.split("|")
        return a[0], a[1], a[2], a[3].split("#")[0]

    def ranges(lab):
        mt, nt, kt, _ = tiers(lab)
        (mlo, mhi), (nlo, nhi), (klo, khi) = MN[mt], MN[nt], KK[kt]
        for seg in lab.split("#")[1:]:
            g = re.match(r"([MNK])(<=|>)(\d+)", seg)
            if not g:
                continue
            ax, op, v = g.group(1), g.group(2), int(g.group(3))
            if ax == "M":
                mlo, mhi = (mlo, min(mhi, v)) if op == "<=" else (max(mlo, v + 1), mhi)
            elif ax == "N":
                nlo, nhi = (nlo, min(nhi, v)) if op == "<=" else (max(nlo, v + 1), nhi)
            else:
                klo, khi = (klo, min(khi, v)) if op == "<=" else (max(klo, v + 1), khi)
        return mlo, mhi, nlo, nhi, klo, khi

    def depth(lab):
        return len(lab.split("#")) - 1

    def lg(v):
        return math.log10(max(v, 1))

    def cuboid(x0, x1, y0, y1, z0, z1):
        p = [
            (x0, y0, z0),
            (x1, y0, z0),
            (x1, y1, z0),
            (x0, y1, z0),
            (x0, y0, z1),
            (x1, y0, z1),
            (x1, y1, z1),
            (x0, y1, z1),
        ]
        faces = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [1, 2, 6, 5],
            [0, 3, 7, 4],
        ]
        return [[p[i] for i in q] for q in faces]

    depth_col = {
        1: "#9ecae1",
        2: "#4292c6",
        3: "#08519c",
        4: "#08306b",
        5: "#041f4a",
        6: "#02132e",
    }
    out = {}
    for bt in ("Bany", "Bnone"):
        sl = sorted(
            [
                lab
                for lab in leaves
                if lab.count("|") >= 3
                and tiers(lab)[3] == bt
                and tiers(lab)[0] in MN
                and tiers(lab)[1] in MN
                and tiers(lab)[2] in KK
            ],
            key=depth,
        )
        if not sl:
            continue
        fig = plt.figure(figsize=(7.6, 6.6))
        ax = fig.add_subplot(111, projection="3d")
        for lab in sl:
            mlo, mhi, nlo, nhi, klo, khi = ranges(lab)
            d = depth(lab)
            faces = cuboid(lg(mlo), lg(mhi), lg(nlo), lg(nhi), lg(klo), lg(khi))
            if d == 0:
                pc = Poly3DCollection(
                    faces,
                    facecolors=(0, 0, 0, 0),
                    edgecolor=(0.5, 0.5, 0.5, 0.45),
                    linewidths=0.2,
                )
            else:
                pc = Poly3DCollection(
                    faces,
                    facecolor=depth_col.get(d, "#02132e"),
                    edgecolor="white",
                    linewidths=0.4,
                    alpha=0.45 + 0.08 * min(d, 4),
                )
            ax.add_collection3d(pc)
        mn_ticks = [1, MN["Tiny"][1], MN["Mid"][1], MN["Large"][1]]
        k_ticks = [1, KK["TinyK"][1], KK["MidK"][1], KK["LargeK"][1]]
        ax.set_xlim(0, lg(mn_ticks[-1]))
        ax.set_ylim(0, lg(mn_ticks[-1]))
        ax.set_zlim(0, lg(k_ticks[-1]))
        ax.set_xlabel("M", labelpad=12)
        ax.set_ylabel("N", labelpad=12)
        ax.set_zlabel("K", labelpad=14)
        ax.set_xticks([lg(v) for v in mn_ticks])
        ax.set_yticks([lg(v) for v in mn_ticks])
        ax.set_zticks([lg(v) for v in k_ticks])
        ax.set_xticklabels([_tick(v) for v in mn_ticks])
        ax.set_yticklabels([_tick(v) for v in mn_ticks])
        ax.set_zticklabels([_tick(v) for v in k_ticks])
        ax.view_init(elev=18, azim=-52)
        maxd = max((depth(lab) for lab in sl), default=0)
        if maxd >= 1:
            ax.legend(
                handles=[
                    Patch(
                        facecolor=depth_col.get(d, "#02132e"),
                        edgecolor="gray",
                        label=f"depth {d}",
                    )
                    for d in range(1, maxd + 1)
                ],
                frameon=False,
                fontsize=8,
                loc="upper left",
                ncol=2,
            )
        ax.set_title(f"cells (M/N/K), B = {bt}: {len(sl)} leaves", fontsize=10)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=170, bbox_inches="tight", pad_inches=0.35)
        plt.close(fig)
        out[bt] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return out


# ── held-out GEMMs by category ──────────────────────────────────────────────


def held_out_by_category(run_dir, held):
    """Held-out GEMMs (each active round's new shapes, re-used history shapes
    excluded) by shape category and by base cell. A GEMM counts with its
    leaf cell's held-out selection efficiency."""
    _pipeline_on_path()
    try:
        from lib.bench_yaml import parse_shapes_yaml
        from lib.grid import cell_key
        from lib.subcells import assign_subcell, load_cumulative_split_tree
    except Exception:
        return None, None, 0
    try:
        from lib.shapes import read_reused_shapes
    except Exception:

        def read_reused_shapes(_path):
            return set()

    def inherit(lab, key):
        cur = lab
        while True:
            v = _num((held.get(cur) or {}).get(key))
            if v is not None:
                return v
            j = cur.rfind("#")
            if j < 0:
                return None
            cur = cur[:j]

    from pathlib import Path

    rds = round_dirs(run_dir)
    try:
        tree = load_cumulative_split_tree([Path(d) for d in rds])
    except Exception:
        tree = {}
    cat = {c: {"n": 0, "s": [], "o": []} for c in SHAPE_CATEGORIES}
    bases = {}
    total = 0
    for d in rds:
        if not os.path.isfile(os.path.join(d, "stage04b", "decisions.json")):
            continue
        try:
            shapes = parse_shapes_yaml(Path(d) / "stage01" / "shapes.yaml")
            reused = read_reused_shapes(Path(d) / "stage01")
        except Exception:
            continue
        for m, n, k, b in shapes:
            if (m, n, k, b) in reused:
                continue
            total += 1
            rec = cat[categorize_gemm(m, n, k, b)]
            rec["n"] += 1
            try:
                base = cell_key(m, n, k, b)
                leaf = assign_subcell(base, m, n, k, b, tree)
            except Exception:
                continue
            bc = bases.setdefault(base, {"n": 0, "leaves": set()})
            bc["n"] += 1
            bc["leaves"].add(leaf)
            se = inherit(leaf, "sel_eff_new")
            oe = inherit(leaf, "origami_sel_eff_new")
            if se:
                rec["s"].append(se)
            if oe:
                rec["o"].append(oe)
    if not total:
        return None, None, 0

    def geo(xs):
        return math.exp(sum(math.log(x) for x in xs) / len(xs)) * 100 if xs else None

    cat_rows = []
    for c in SHAPE_CATEGORIES:
        r = cat[c]
        if not r["n"]:
            continue
        sm, om = geo(r["s"]), geo(r["o"])
        cat_rows.append(
            {
                "category": c,
                "n": r["n"],
                "held": round(sm, 2) if sm is not None else None,
                "origami": round(om, 2) if om is not None else None,
                "delta": (
                    round(sm - om, 2) if sm is not None and om is not None else None
                ),
            }
        )
    base_rows = sorted(
        (
            {"cell": b, "n": v["n"], "leaves": len(v["leaves"])}
            for b, v in bases.items()
        ),
        key=lambda r: r["n"],
        reverse=True,
    )
    return cat_rows, base_rows, total


# ── page ────────────────────────────────────────────────────────────────────


def _f(v, digits=2, signed=False):
    v = _num(v)
    if v is None:
        return "&ndash;"
    return f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"


def stage07_html(s7):
    if not s7:
        return '<p class="mut">no stage 7 results in this run.</p>'
    parts = []
    status = "all checks passed" if s7["ok"] else f"{s7['failures']} check(s) failed"
    parts.append(f'<p class="cap">stage 7 status: <b>{esc(status)}</b>.</p>')
    for y in s7["yamls"]:
        par = y.get("parity")
        rows = []
        if par:
            counts = ", ".join(
                f"{esc(k)}={esc(v)}" for k, v in sorted(par["counts"].items())
            )
            rows.append(
                f'<tr><td class="c">pick parity</td><td colspan="8">{esc(par["agree"])} / '
                f'{esc(par["n"])} problems agree ({_f(par["rate"])}%) &mdash; {counts}</td></tr>'
            )
        for t in y.get("timing") or []:
            rows.append(
                f'<tr><td class="c">selection time, rsn={esc(t["rsn"])}</td>'
                f'<td>{esc(t["n"])}</td><td>{esc(t["reps"])}</td>'
                f'<td>{_f(t["on_p50"])}</td><td>{_f(t["off_p50"])}</td>'
                f'<td>{_f(t["d_p10"], signed=True)}</td><td><b>{_f(t["d_p50"], signed=True)}</b></td>'
                f'<td>{_f(t["d_p90"], signed=True)}</td><td>{_f(t["d_p99"], signed=True)}</td></tr>'
            )
        parts.append(
            f'<h3>{esc(y["yaml"])} <span class="mut">({esc(y.get("n_kept"))} of '
            f'{esc(y.get("n_entries"))} problems match the config)</span></h3>'
            '<table style="max-width:1100px"><thead><tr><th class="c">check</th>'
            "<th>problems</th><th>reps</th><th>ML-on p50 &micro;s</th>"
            "<th>ML-off p50 &micro;s</th><th>&Delta; p10</th><th>&Delta; p50</th>"
            "<th>&Delta; p90</th><th>&Delta; p99</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )
    return "".join(parts)


def stage08_html(s8):
    if not s8:
        return '<p class="mut">no stage 8 results in this run.</p>'
    rows = []
    for r in s8:
        d = (
            r["model"] - r["origami"]
            if r["model"] is not None and r["origami"] is not None
            else None
        )
        wtl = (
            f"{esc(r['wins'])} / {esc(r['ties'])} / {esc(r['losses'])}"
            if r.get("paired_n")
            else "&ndash;"
        )
        rows.append(
            f'<tr><td class="c">{esc(r["name"])}</td><td>{esc(r["rc"])}</td>'
            f'<td>{esc(r["n"]) if r["n"] is not None else "&ndash;"}</td>'
            f'<td>{_f(r["model"])}</td><td>{_f(r["origami"])}</td>'
            f'<td>{_f(d, signed=True)}</td><td>{wtl}</td><td>{_f(r["speedup"], 3)}</td></tr>'
        )
    return (
        '<table style="max-width:1100px"><thead><tr><th class="c">dataset</th><th>rc</th>'
        "<th>GEMMs</th><th>model %</th><th>Origami %</th><th>&Delta; pp</th>"
        "<th>wins / ties / losses</th><th>speedup vs Origami</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def timing_html(rounds, total):
    if not rounds:
        return '<p class="mut">no timing records.</p>'
    order = [
        "stage01",
        "stage02",
        "stage03",
        "stage04",
        "stage04b",
        "stage05",
        "stage06",
        "stage07",
        "stage08",
    ]
    present = [s for s in order if any(s in st for st in rounds.values())]
    head = "".join(f"<th>{esc(s)}</th>" for s in present)
    body = []
    for rnd in sorted(rounds, key=lambda r: (r is None, r if r is not None else 0)):
        st = rounds[rnd]
        name = "validate" if rnd is None else f"round_{rnd}"
        tds = "".join(
            (
                f"<td>{esc(fmt_dur(st[s]['elapsed_s']))}{'' if st[s].get('rc') == 0 else ' (rc=' + esc(st[s].get('rc')) + ')'}</td>"
                if s in st
                else "<td></td>"
            )
            for s in present
        )
        tot = st.get("(round total)")
        body.append(
            f'<tr><td class="c">{esc(name)}</td>{tds}<td>{esc(fmt_dur(tot["elapsed_s"])) if tot else ""}</td></tr>'
        )
    body.append(
        f'<tr style="font-weight:600"><td class="c">all invocations</td>{"<td></td>" * len(present)}'
        f"<td>{esc(fmt_dur(total))}</td></tr>"
    )
    return (
        f'<table><thead><tr><th class="c">round</th>{head}<th>round total</th></tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def category_html(cat_rows, base_rows, n_total):
    if not cat_rows:
        return '<p class="mut">no held-out shapes in this run.</p>'
    rows = "".join(
        f'<tr{" class=lose" if (r["delta"] is not None and r["delta"] < 0) else ""}>'
        f'<td class="c">{esc(r["category"])}</td><td>{r["n"]:,}</td>'
        f'<td>{_f(r["held"])}</td><td>{_f(r["origami"])}</td><td>{_f(r["delta"], signed=True)}</td></tr>'
        for r in cat_rows
    )
    out = (
        f'<p class="cap">{n_total:,} held-out GEMMs (the new shapes of every active round, '
        "without re-used history shapes). Each GEMM counts with the held-out selection "
        "efficiency of the leaf cell it routes to.</p>"
        '<table style="max-width:820px"><thead><tr><th class="c">category</th><th>GEMMs</th>'
        "<th>model %</th><th>Origami %</th><th>&Delta; pp</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )
    if base_rows:
        tot = sum(r["n"] for r in base_rows) or 1
        brows = "".join(
            f'<tr><td class="c">{esc(r["cell"])}</td><td>{r["n"]:,}</td>'
            f'<td>{r["n"] / tot * 100:.1f}%</td><td>{r["leaves"]:,}</td></tr>'
            for r in base_rows
        )
        out += (
            "<h3>Held-out GEMMs per base cell</h3>"
            '<div class="scroll"><table style="margin-bottom:0"><thead><tr><th class="c">base cell</th>'
            "<th>GEMMs</th><th>share</th><th>leaves</th></tr></thead>"
            f"<tbody>{brows}</tbody></table></div>"
        )
    return out


def build(run_dir, out_path):
    run_dir = run_dir.rstrip("/")
    held = held_out_breakdown(run_dir)
    agg = held_out_aggregate(held)
    traj = held_out_rounds(run_dir)
    train = training_rounds(run_dir)
    cells = final_cells(run_dir, held)
    s7 = stage07(run_dir)
    s8 = stage08(run_dir)
    t_rounds, t_total = timing_rows(run_dir)
    cat_rows, base_rows, n_cat = held_out_by_category(run_dir, held)
    in_sample = [c["model"] for c in cells if c.get("model") is not None]

    def rt_eff(key):
        num = den = 0.0
        for c in cells:
            n, w, p = c.get("n_gemms") or 0, c.get("held_w_us"), c.get(key)
            if n and w and p:
                num += n * w
                den += n * p
        return round(num / den * 100, 2) if den else None

    worst = sorted(
        (
            dict(
                c,
                extra_us=(
                    c["n_gemms"] * (c["held_pick_us"] - c["held_ori_pick_us"])
                    if c.get("held_pick_us") and c.get("held_ori_pick_us")
                    else None
                ),
            )
            for c in cells
        ),
        key=lambda c: (
            c["held_delta"] is None,
            c["held_delta"] if c["held_delta"] is not None else 0,
        ),
    )[:25]
    s7_tile = None
    for y in (s7 or {}).get("yamls", []):
        for t in y.get("timing") or []:
            if t["rsn"] == "1" and t.get("d_p50") is not None:
                s7_tile = t["d_p50"]
                break
        if s7_tile is not None:
            break
    summary = {
        "run": os.path.basename(run_dir),
        "rounds": len(round_dirs(run_dir)),
        "held": agg,
        "in_sample_mean": (
            round(sum(in_sample) / len(in_sample), 2) if in_sample else None
        ),
        "rt_model": rt_eff("held_pick_us"),
        "rt_origami": rt_eff("held_ori_pick_us"),
        "n_cells": len(cells),
        "bin_mb": deployed_size_mb(run_dir),
        "shapes": shapes_total(run_dir),
        "wall": fmt_dur(t_total) if t_total else None,
        "sel_delta_us": s7_tile,
    }
    s8pc = {r["name"]: r["per_cell"] for r in s8 if r.get("name") and r.get("per_cell")}
    images = cell3d_images(list(held) or [c["cell"] for c in cells])
    cell3d = (
        "".join(
            f'<figure class="chart" style="max-width:520px"><figcaption>B = {esc(bt)}</figcaption>'
            f'<img alt="cell map B={esc(bt)}" style="width:100%" src="{uri}"></figure>'
            for bt, uri in images.items()
        )
        or '<p class="mut">cell map unavailable (needs matplotlib and held-out cells).</p>'
    )
    train_rows = (
        "".join(
            f'<tr><td class="c">{esc(r["round"])}</td><td>{_f(r["model"])}</td><td>{_f(r["deployed"])}</td>'
            f'<td>{_f(r["origami"])}</td><td>{esc(r["n_cells"])}</td><td>{esc(r["n_gemms"])}</td></tr>'
            for r in train
        )
        or '<tr><td class="c" colspan="6">no stage05 metrics.json in this run</td></tr>'
    )
    page = (
        _HTML.replace("__TITLE__", esc(summary["run"]))
        .replace("__S7__", stage07_html(s7))
        .replace("__S8__", stage08_html(s8))
        .replace("__TRAINROWS__", train_rows)
        .replace("__CHART_TREND__", chart_trend(train))
        .replace(
            "__CHART_HELD__",
            chart_hist(
                cells,
                "held",
                "Held-out selection efficiency per leaf cell (new shapes of later rounds)",
            ),
        )
        .replace(
            "__CHART_INSAMPLE__",
            chart_hist(
                cells,
                "deployed",
                "In-sample selection efficiency per leaf cell (its own training GEMMs)",
            ),
        )
        .replace("__CATEGORIES__", category_html(cat_rows, base_rows, n_cat))
        .replace("__TREE__", esc(cell_tree_ascii(cells)))
        .replace("__CELL3D__", cell3d)
        .replace("__TIMING__", timing_html(t_rounds, t_total))
        .replace("__CONFIG__", esc(config_source(run_dir)))
        .replace("__TRAJ__", script_json(_finite(traj)))
        .replace("__CELLS__", script_json(_finite(cells)))
        .replace("__WORST__", script_json(_finite(worst)))
        .replace("__SUMM__", script_json(_finite(summary)))
        .replace("__S8PC__", script_json(_finite(s8pc)))
    )
    with open(out_path, "w") as f:
        f.write(page)
    return out_path


_HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Run report - __TITLE__</title><style>
:root{--bd:#d8dee4;--mut:#57606a;--win:#e6ffec;--lose:#ffebe9;--weak:#fff8c5;}
body{font:14px/1.45 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#1f2328;margin:0;padding:24px;max-width:1760px;}
p,h1,h2,.stats,details{max-width:1200px;}
h1{font-size:20px;margin:0 0 2px;} h2{font-size:15px;margin:24px 0 8px;} h3{font-size:13px;margin:14px 0 6px;}
.cap{color:var(--mut);font-size:12.5px;margin:0 0 12px;} .mut{color:var(--mut);font-size:11.5px;}
.stats{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:8px;}
.stat{border:1px solid var(--bd);border-radius:8px;padding:9px 13px;min-width:120px;}
.stat .v{font-size:21px;font-weight:600;} .stat .l{color:var(--mut);font-size:11.5px;margin-top:2px;}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;margin-bottom:6px;}
th,td{padding:4px 9px;border-bottom:1px solid var(--bd);text-align:right;white-space:nowrap;}
th{position:sticky;top:0;background:#f6f8fa;font-weight:600;} th.sort{cursor:pointer;}
td.c,th.c{text-align:left;font-family:ui-monospace,Menlo,monospace;font-size:12px;}
tr.win td{background:var(--win);} tr.lose td{background:var(--lose);} tr.weak td{background:var(--weak);}
pre.cfg{background:#f6f8fa;border:1px solid var(--bd);border-radius:8px;padding:12px;overflow:auto;font-size:12px;max-height:520px;}
figure.chart{margin:6px 0 16px;max-width:782px;} figure.chart figcaption{color:var(--mut);font-size:12px;margin:0 0 5px;}
figure.chart svg{width:100%;height:auto;border:1px solid var(--bd);border-radius:8px;background:#fff;}
.scroll{max-height:560px;overflow:auto;border:1px solid var(--bd);border-radius:8px;margin-bottom:6px;}
.s8bars{max-height:520px;overflow:auto;border:1px solid var(--bd);border-radius:8px;padding:5px 10px;font:12.5px/1.4 ui-monospace,Menlo,monospace;}
.s8r{display:flex;align-items:center;height:18px;white-space:nowrap;} .s8lab{overflow:hidden;text-overflow:ellipsis;color:#444;}
.s8bar{flex:0 0 240px;position:relative;height:13px;} .s8mid{position:absolute;left:120px;top:-2px;bottom:-2px;width:1px;background:#ccc;}
.s8fill{position:absolute;top:1px;height:11px;} .s8v{flex:0 0 200px;text-align:right;color:#57606a;padding-left:12px;}
</style></head><body>
<h1>Run report &mdash; __TITLE__</h1>
<p class="cap" id="cap"></p>
<div class="stats" id="stats"></div>
<p class="cap"><b>Cells</b> are <code>M-tier|N-tier|K-tier|Batch</code>: M and N Tiny &le;32, Small &le;128, Mid &le;512, Large &gt;512; K TinyK &le;32, MidK &le;512, LargeK &gt;512; Bnone = unbatched, Bany = batched. Split suffixes such as <code>#K&gt;4096</code> subdivide a cell. Selection efficiency of a GEMM = fastest measured kernel time / time of the picked kernel; aggregates are geometric means.</p>

<h2>Stage 7 &mdash; selection time and C++/Python pick parity</h2>
<p class="cap">Selection time is hipBLASLt's per-call "Solution selection time" for the request size shown (rsn); &Delta; = ML-on minus ML-off for the same problem, median over repetitions, summarized over problems. The first call of each run (library load) is excluded.</p>
__S7__

<h2>Stage 8 &mdash; selection efficiency on external datasets</h2>
__S8__
<div id="s8charts"></div>

<h2>Held-out selection efficiency per round (stage04b)</h2>
<p class="cap">The model of the previous round on this round's new shapes. The last row aggregates the latest measurement of every leaf cell.</p>
<table id="traj" style="max-width:900px"><thead><tr><th class="c">round</th><th>model %</th><th>Origami %</th><th>&Delta; pp</th><th>GEMMs</th></tr></thead><tbody id="tbody_traj"></tbody></table>

<h2>Training (stage05, in-sample)</h2>
<p class="cap">Selection efficiency on the GEMMs of the cells trained in that round: model = full ranking, deployed = whitelist-restricted ranking as shipped, Origami = the analytical pick.</p>
<table style="max-width:900px"><thead><tr><th class="c">round</th><th>model %</th><th>deployed %</th><th>Origami %</th><th>cells trained</th><th>GEMMs</th></tr></thead><tbody>__TRAINROWS__</tbody></table>
__CHART_TREND__

<h2 id="cellh">Leaf cells of the final model</h2>
<p class="cap"><b>held-out</b> values are on new shapes of later rounds (<b>*</b> = inherited from the parent cell); the other columns are in-sample. K = whitelist size; ed/hd/ih = embedding / hidden / interaction widths. Click a header to sort.</p>
<div class="scroll"><table id="cells" style="margin-bottom:0;font-size:11px"><thead><tr id="chead"></tr></thead><tbody id="tbody_cells"></tbody></table></div>
__CHART_HELD__
__CHART_INSAMPLE__

<h2>Worst held-out cells</h2>
<p class="cap">Leaf cells sorted by held-out &Delta; vs Origami. extra &micro;s = GEMMs &times; (model pick &minus; Origami pick) held-out latency geomeans.</p>
<div class="scroll"><table style="margin-bottom:0;font-size:11px"><thead><tr id="worsth"></tr></thead><tbody id="tbody_worst"></tbody></table></div>

<h2>Held-out GEMMs by shape category</h2>
__CATEGORIES__

<h2>Split tree</h2>
<pre class="cfg">__TREE__</pre>

<h2>Cells in M / N / K</h2>
<div style="display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start">__CELL3D__</div>

<h2>Wall-clock</h2>
__TIMING__

<h2>Configuration</h2>
<details><summary>config (as written, before environment expansion)</summary><pre class="cfg">__CONFIG__</pre></details>

<script>
const TRAJ=__TRAJ__, CELLS=__CELLS__, WORST=__WORST__, S=__SUMM__, S8PC=__S8PC__;
function esc(s){return String(s==null?"":s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&#39;");}
function num(v,d){return (v==null||!isFinite(v))?"\u2013":Number(v).toFixed(d==null?2:d);}
function sgn(v,d){return (v==null||!isFinite(v))?"\u2013":(v>=0?"+":"")+Number(v).toFixed(d==null?2:d);}
const H=S.held||{};
document.getElementById("cap").textContent="Run "+S.run+" \u2013 "+S.rounds+" rounds"+(S.wall?(" \u2013 wall-clock "+S.wall):"");
const stats=[
  ["held-out sel_eff, model (latest per leaf)",H.model!=null?num(H.model)+"%":"n/a"],
  ["held-out sel_eff, Origami (same cells)",H.origami!=null?num(H.origami)+"%":"n/a"],
  ["model \u2212 Origami",H.delta!=null?sgn(H.delta)+" pp":"n/a"],
  ["per-cell best of model / Origami (upper bound)",H.router_bound!=null?num(H.router_bound)+"%":"n/a"],
  ["leaf cells with a held-out value",H.n_cells!=null?H.n_cells:"n/a"],
  ["runtime-weighted held-out, model / Origami",(S.rt_model!=null?num(S.rt_model)+"%":"n/a")+" / "+(S.rt_origami!=null?num(S.rt_origami)+"%":"n/a")],
  ["in-sample mean over leaf cells",S.in_sample_mean!=null?num(S.in_sample_mean)+"%":"n/a"],
  ["deployed model file",S.bin_mb!=null?S.bin_mb+" MB":"n/a"],
  ["shapes generated",S.shapes!=null?Number(S.shapes).toLocaleString():"n/a"],
  ["selection time ML-on \u2212 ML-off (median per problem, rsn=1)",S.sel_delta_us!=null?sgn(S.sel_delta_us)+" \u00b5s":"n/a"],
];
document.getElementById("stats").innerHTML=stats.map(([l,v])=>`<div class="stat"><div class="v">${esc(v)}</div><div class="l">${esc(l)}</div></div>`).join("");
document.getElementById("tbody_traj").innerHTML=TRAJ.map(r=>`<tr><td class="c">${esc(r.round)}</td><td>${num(r.model)}</td><td>${num(r.origami)}</td><td>${(r.model!=null&&r.origami!=null)?sgn(r.model-r.origami):"\u2013"}</td><td>${r.n_gemms!=null?Number(r.n_gemms).toLocaleString():""}</td></tr>`).join("")
 +(H.model!=null?`<tr style="font-weight:600"><td class="c">latest per leaf cell</td><td>${num(H.model)}</td><td>${num(H.origami)}</td><td>${sgn(H.delta)}</td><td>${esc(H.n_cells)} cells</td></tr>`:"");
const COLS=[["cell","cell"],["n_gemms","GEMMs"],["held","held-out %"],["held_ori","Origami held-out %"],["held_delta","\u0394 held-out"],["model","model %"],["deployed","deployed %"],["k_delta","K\u0394"],["origami","Origami %"],["dep_delta","\u0394 deployed"],["smartK","K"],["cap","ed/hd/ih"]];
const STR=k=>(k==="cell"||k==="cap");
let sk="held",asc=true;
function rowClass(c){if(c.held==null)return"";if(c.held>=95)return"win";if(c.held_ori!=null&&c.held<c.held_ori-2)return"lose";if(c.held<88)return"weak";return"";}
function cellText(c,k){const v=c[k];if(STR(k))return esc(v);if(k==="n_gemms")return Number(v||0).toLocaleString();if(k==="smartK")return v!=null?esc(v):"";if(k==="held")return v!=null?num(v)+(c.held_inherited?"*":""):"";if(k.endsWith("delta"))return v!=null?sgn(v):"";return v!=null?num(v):"";}
function render(){CELLS.sort((a,b)=>{const d=STR(sk)?String(a[sk]||"").localeCompare(String(b[sk]||"")):((a[sk]??-1e9)-(b[sk]??-1e9));return asc?d:-d;});
 const head=document.getElementById("chead");head.innerHTML="<th>#</th>"+COLS.map(([k,t])=>`<th class="sort ${STR(k)?"c":""}" data-k="${esc(k)}">${esc(t)}${k===sk?(asc?" \u25b2":" \u25bc"):""}</th>`).join("");
 document.getElementById("tbody_cells").innerHTML=CELLS.map((c,i)=>`<tr class="${rowClass(c)}"><td class="c">${i+1}</td>`+COLS.map(([k])=>`<td class="${STR(k)?"c":""}">${cellText(c,k)}</td>`).join("")+"</tr>").join("");}
document.getElementById("chead").addEventListener("click",e=>{const th=e.target.closest("th[data-k]");if(!th)return;const k=th.dataset.k;if(k===sk)asc=!asc;else{sk=k;asc=STR(k);}render();});
render();
const WCOLS=[["cell","cell"],["n_gemms","GEMMs"],["held_w_us","fastest \u00b5s"],["held_pick_us","model pick \u00b5s"],["held_ori_pick_us","Origami pick \u00b5s"],["extra_us","extra \u00b5s"],["held","held-out %"],["held_ori","Origami %"],["held_delta","\u0394"],["cap","ed/hd/ih"]];
document.getElementById("worsth").innerHTML="<th>#</th>"+WCOLS.map(([k,t])=>`<th class="${STR(k)?"c":""}">${esc(t)}</th>`).join("");
document.getElementById("tbody_worst").innerHTML=WORST.map((c,i)=>`<tr class="${(c.held_delta!=null&&c.held_delta<0)?"lose":""}"><td class="c">${i+1}</td>`+WCOLS.map(([k])=>`<td class="${STR(k)?"c":""}">${STR(k)?esc(c[k]):(k==="n_gemms"?Number(c[k]||0).toLocaleString():(k==="held_delta"?sgn(c[k],1):num(c[k],1)))}</td>`).join("")+"</tr>").join("");
(function(){
 const host=document.getElementById("s8charts");const names=Object.keys(S8PC).sort();
 for(const ds of names){
  const rows=(S8PC[ds]||[]).slice();if(!rows.length)continue;
  const fig=document.createElement("figure");fig.className="chart";fig.style.maxWidth="1700px";
  const cap=document.createElement("figcaption");const wins=rows.filter(r=>r[1]>=r[2]).length;
  cap.innerHTML=`<b>${esc(ds)}</b> \u2013 ${rows.length} cells, model ahead in ${wins}. sort `;
  const sel=document.createElement("select");
  for(const [v,t] of [["dw","\u0394 worst first"],["db","\u0394 best first"],["la","label"]]){const o=document.createElement("option");o.value=v;o.textContent=t;sel.appendChild(o);}
  cap.appendChild(sel);fig.appendChild(cap);
  const box=document.createElement("div");box.className="s8bars";fig.appendChild(box);host.appendChild(fig);
  const draw=mode=>{const r2=rows.slice();
   if(mode==="db")r2.sort((a,b)=>(b[1]-b[2])-(a[1]-a[2]));else if(mode==="la")r2.sort((a,b)=>a[0]<b[0]?-1:a[0]>b[0]?1:0);else r2.sort((a,b)=>(a[1]-a[2])-(b[1]-b[2]));
   const maxd=Math.max(1,...r2.map(r=>Math.abs(r[1]-r[2])));
   const labw=Math.min(900,Math.max(220,Math.round(7.7*Math.max(0,...r2.map(r=>String(r[0]).length)))));
   box.innerHTML=r2.map(r=>{const d=r[1]-r[2],w=Math.abs(d)/maxd*116;const left=d>=0?120:120-w;
    return `<div class="s8r"><div class="s8lab" style="flex:0 0 ${labw}px" title="${esc(r[0])}">${esc(r[0])}</div><div class="s8bar"><div class="s8mid"></div><div class="s8fill" style="left:${left.toFixed(1)}px;width:${w.toFixed(1)}px;background:${d>=0?"#2da44e":"#cf222e"}"></div></div><div class="s8v">${num(r[1],1)} / ${num(r[2],1)} <b>${sgn(d,1)}</b></div></div>`;}).join("");};
  sel.addEventListener("change",()=>draw(sel.value));draw("dw");
 }
})();
</script></body></html>"""


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: make_report.py <run_dir> [out.html]", file=sys.stderr)
        sys.exit(2)
    rd = sys.argv[1]
    outp = sys.argv[2] if len(sys.argv) > 2 else os.path.join(rd, "report.html")
    print("wrote", build(rd, outp))
