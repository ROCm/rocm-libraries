#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Write a self-contained HTML report of a pipeline run.

Usage: make_report.py <run_dir> [out.html] [--jobs N] [--no-engine] [--no-bench]
                      [--target T] [--selection-budget-us US] [--diagnostics]
                      [--origami-reference]

Reads the run's artifacts (round_*/stage04b/{decisions,splits}.json and its
gate/ and train/ slices, round_*/stage04 and stage03c/holdout/stage04
enriched chunks and kernels.json, round_*/stage01b/plan.json,
round_*/stage05/{cells,metrics}.json and models.pt,
round_*/stage06/{manifest,deploy_record,gate}.json, round_*/stage01/
shapes.yaml, validate/stage07/summary.json with its parity and timing logs,
validate/stage08/summary.json with each dataset's per_gemm.csv,
timing/*.json and the latest configs/*.source.yaml) and writes nothing but
the output file. A section whose inputs are missing shows "n/a". The page
has no external references.

The headline is the bundle the run shipped: the latest deploy record not
marked `deployed: false`, measured on GEMMs it never trained on (its deploy
gate, else stage04b; run_orchestrator.shipped_bundle_numbers). The report's
own re-ranking charges picks with the config's validate.pick_measurement,
as stages 4.5, 6, 7 and 8 do. Every efficiency, gap and ranking is against
the fastest measured kernel and an absolute target; Origami appears only in
the opt-in --origami-reference section, and only on the same GEMMs as the
model.
"""
import glob
import html
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import report_sections as rs  # noqa: E402


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


def paired_cells(cells):
    """(model, Origami, GEMMs) over the cells with both numbers on the same
    GEMMs: per cell `paired_sel_eff_new` and `origami_sel_eff_new` over its
    `origami_n_eval` GEMMs."""
    both = [
        c
        for c in cells
        if _num(c.get("paired_sel_eff_new")) and _num(c.get("origami_sel_eff_new"))
    ]
    n = sum(int(c.get("origami_n_eval") or 0) for c in both)
    if not n:
        return None, None, 0
    return (
        _wgeo((c["paired_sel_eff_new"], c.get("origami_n_eval") or 0) for c in both),
        _wgeo((c["origami_sel_eff_new"], c.get("origami_n_eval") or 0) for c in both),
        n,
    )


def held_out_rounds(run_dir):
    """Per active round: the prior model on that round's new shapes, its
    misses, and (for the reference section) the model and Origami on the
    same GEMMs."""
    rows = []
    for d in round_dirs(run_dir):
        dec = load_json(os.path.join(d, "stage04b", "decisions.json"))
        if not isinstance(dec, dict):
            continue
        cells = [c for c in dec.get("per_cell") or [] if isinstance(c, dict)]
        agg = dec.get("aggregate") if isinstance(dec.get("aggregate"), dict) else {}
        model = _num(agg.get("global_sel_eff_new"))
        if model is None:
            model = _wgeo((c.get("sel_eff_new"), c.get("n_eval") or 0) for c in cells)
        n = agg.get("n_gemms_evaluated")
        if not isinstance(n, int):
            n = sum(int(c.get("n_eval") or 0) for c in cells)
        if model is None:
            continue
        pm, po, pn = paired_cells(cells)
        if _num(agg.get("global_paired_sel_eff_new")) is not None:
            pm = agg["global_paired_sel_eff_new"]
            po = _num(agg.get("global_origami_sel_eff_new"))
            pn = agg.get("n_gemms_paired", pn)
        rows.append(
            {
                "round": os.path.basename(d),
                "model": _pct(model),
                "n_gemms": n,
                "n_miss": agg.get("n_miss_total"),
                "pick_measurement": agg.get("pick_measurement"),
                "paired_model": _pct(pm),
                "paired_origami": _pct(po),
                "paired_n": pn,
            }
        )
    return rows


def shipped_bundle(run_dir):
    """The bundle the run shipped and its held-out numbers
    (run_orchestrator.shipped_bundle_numbers), with the deploy-gate record
    of the round that shipped it (`gate`)."""
    rs.pipeline_on_path()
    try:
        from pathlib import Path

        from run_orchestrator import shipped_bundle_numbers

        out = shipped_bundle_numbers(Path(run_dir))
    except Exception:  # noqa: BLE001
        out = load_json(os.path.join(run_dir, "held_out_sel_eff_breakdown.json"))
        if not isinstance(out, dict) or "shipped" not in out:
            return None
    out = _finite(out)
    shipped = out.get("shipped") or {}
    rec_path = shipped.get("deploy_record")
    gate = (
        load_json(os.path.join(os.path.dirname(rec_path), "gate.json"))
        if rec_path
        else None
    )
    out["gate"] = _finite(gate) if isinstance(gate, dict) else None
    return out


def stage04b_contexts(run_dir):
    """Per round: stage04b's held-out summaries per trained context
    (aggregate.by_context) and per held-out context
    (aggregate.context_holdout), keyed by run_analysis context labels."""
    rs.pipeline_on_path()
    from lib import features as fs
    from lib.run_analysis import context_label

    def label(key):
        budget, _, sched = str(key).partition("/")
        try:
            return context_label(
                fs.ExecutionContext(cu_budget=int(budget), schedule=sched or "default")
            )
        except (TypeError, ValueError):
            return str(key)

    rows = []
    for d in round_dirs(run_dir):
        dec = load_json(os.path.join(d, "stage04b", "decisions.json"))
        agg = (dec or {}).get("aggregate") if isinstance(dec, dict) else None
        if not isinstance(agg, dict):
            continue
        for kind, block in (
            ("trained context", agg.get("by_context")),
            ("held-out context", agg.get("context_holdout")),
        ):
            if not isinstance(block, dict):
                continue
            for key, s in sorted(block.items()):
                if not isinstance(s, dict):
                    continue
                rows.append(
                    {
                        "round": os.path.basename(d),
                        "kind": kind,
                        "context": label(key),
                        "n": s.get("n_evaluated"),
                        "n_miss": s.get("n_miss"),
                        "sel_eff": _num(s.get("sel_eff")),
                    }
                )
    return rows


def context_names(config_text):
    """{run_analysis context label: the config's context names}."""
    rs.pipeline_on_path()
    try:
        import yaml

        from lib import features as fs
        from lib.contexts import contexts_from_config
        from lib.run_analysis import context_label

        cfg = yaml.safe_load(config_text or "") or {}
        ctxs = contexts_from_config(cfg if isinstance(cfg, dict) else {})
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for c in ctxs:
        key = context_label(fs.ExecutionContext(c.cu_budget, c.schedule))
        out.setdefault(key, []).append(c.label + (" held out" if c.holdout else ""))
    return {k: ", ".join(v) for k, v in out.items()}


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
    """Latest stage04b held-out measurement per leaf cell over every round
    (each by the model of the round before it)."""
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
                "n_miss": int(c.get("n_miss") or 0),
                "paired_sel_eff_new": _num(c.get("paired_sel_eff_new")),
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
    """GEMM-weighted model selection efficiency over the leaves of
    `per_cell`, and the model and Origami on the same GEMMs (paired: each
    cell's `paired_sel_eff_new` / `origami_sel_eff_new`)."""
    leaves = _leaves(per_cell)
    mo = _wgeo(
        (per_cell[c].get("sel_eff_new"), per_cell[c].get("n_eval")) for c in leaves
    )
    pm, po, pn = paired_cells([per_cell[c] for c in leaves])
    both = [
        c
        for c in leaves
        if _num(per_cell[c].get("paired_sel_eff_new"))
        and _num(per_cell[c].get("origami_sel_eff_new"))
    ]
    best = _wgeo(
        (
            max(per_cell[c]["paired_sel_eff_new"], per_cell[c]["origami_sel_eff_new"]),
            per_cell[c].get("origami_n_eval"),
        )
        for c in both
    )
    wins = sum(
        1
        for c in both
        if per_cell[c]["paired_sel_eff_new"] >= per_cell[c]["origami_sel_eff_new"]
    )
    return {
        "model": _pct(mo),
        "n_gemms": sum(int(per_cell[c].get("n_eval") or 0) for c in leaves),
        "n_miss": sum(int(per_cell[c].get("n_miss") or 0) for c in leaves),
        "paired_model": _pct(pm),
        "paired_origami": _pct(po),
        "paired_n": pn,
        "delta": (
            round((pm - po) * 100, 2) if pm is not None and po is not None else None
        ),
        "router_bound": _pct(best),
        "model_win_frac": round(wins / len(both) * 100, 1) if both else None,
        "n_cells": len(leaves),
    }


def final_cells(run_dir, held, upto=None):
    """Leaf cells of the shipped bundle: the latest cells.json entry per
    label over the rounds up to `upto` (a round name; default all), with
    held-out values (inherited from the nearest measured ancestor for leaves
    created by the last split)."""
    by_label = {}
    names = [os.path.basename(d) for d in round_dirs(run_dir)]
    last = names.index(upto) if upto in names else len(names) - 1
    for d in round_dirs(run_dir)[: last + 1]:
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
        model = _pct(c.get("best_sel_eff"))
        deployed = _pct(dep.get("sel_eff"))
        held_v, inherited = inherit(lab, "sel_eff_new")
        held_pm, _ = inherit(lab, "paired_sel_eff_new")
        held_o, _ = inherit(lab, "origami_sel_eff_new")
        held_w, _ = inherit(lab, "winner_us_geomean")
        held_p, _ = inherit(lab, "pick_us_geomean")
        hv, hpm, ho = _pct(held_v), _pct(held_pm), _pct(held_o)

        def diff(a, b):
            return round(a - b, 2) if a is not None and b is not None else None

        cells.append(
            {
                "cell": lab,
                "n_gemms": int(c.get("n_gemms") or 0),
                "model": model,
                "deployed": deployed,
                "k_delta": diff(deployed, model),
                "held": hv,
                "held_inherited": inherited,
                "held_paired": hpm,
                "held_ori": ho,
                "held_delta": diff(hpm, ho),
                "smartK": len(c.get("smart_k_signatures") or []) or None,
                "cap": f"{c.get('embed_dim')}/{c.get('hidden_dim')}/{c.get('inter_hidden')}",
                "held_w_us": held_w,
                "held_pick_us": held_p,
            }
        )
    return cells


def _ci(v):
    return [float(v[0]), float(v[1])] if isinstance(v, list) and len(v) == 2 else None


def _rsn_order(item):
    try:
        return int(str(item[0]).replace("rsn", ""))
    except ValueError:
        return 1 << 30


def stage07(run_dir):
    """Per bench yaml of the latest stage07 run: pick parity, the runtime's
    selection efficiency (its top-1 picks against the bench timings of the
    same problems, with the bootstrap interval), and per request size the
    ML-on selection times overall, on the first call into a cell and warm
    (medians with bootstrap intervals), plus the ML-off reference."""
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
            sel = par.get("runtime_selection")
            if isinstance(sel, dict):
                row["runtime_selection"] = {
                    "pick_measurement": sel.get("pick_measurement"),
                    "n_problems": sel.get("n_problems"),
                    "n": sel.get("n_evaluated"),
                    "n_miss": sel.get("n_miss"),
                    "sel_eff": _num(sel.get("sel_eff")),
                    "ci95": _ci(sel.get("sel_eff_ci95")),
                    "p5": _num((sel.get("percentiles") or {}).get("p5")),
                    "without": {
                        k: sel.get(k)
                        for k in (
                            "n_without_runtime_pick",
                            "n_without_timings",
                            "n_top1_without_row",
                            "n_outside_the_pool",
                        )
                        if sel.get(k)
                    },
                }
        timing = []
        for rsn, t in sorted((y.get("timing") or {}).items(), key=_rsn_order):
            if not isinstance(t, dict):
                continue
            d = t.get("delta_us") or {}
            on = t.get("ml_on_us") or {}
            first = (t.get("first_in_cell") or {}).get("ml_on_us") or {}
            warm = (t.get("warm") or {}).get("ml_on_us") or {}
            timing.append(
                {
                    "rsn": str(rsn).replace("rsn", ""),
                    "n": on.get("n", d.get("n")),
                    "reps": t.get("repetitions"),
                    "on_p50": on.get("p50"),
                    "on_p50_ci": _ci(on.get("p50_ci95")),
                    "on_p99": on.get("p99"),
                    "on_max": on.get("max"),
                    "first_n": first.get("n"),
                    "first_p50": first.get("p50"),
                    "first_p50_ci": _ci(first.get("p50_ci95")),
                    "first_max": first.get("max"),
                    "warm_n": warm.get("n"),
                    "warm_p50": warm.get("p50"),
                    "warm_p50_ci": _ci(warm.get("p50_ci95")),
                    "warm_max": warm.get("max"),
                    "off_p50": (t.get("ml_off_us") or {}).get("p50"),
                    "d_p10": d.get("p10"),
                    "d_p50": d.get("p50"),
                    "d_p50_ci": _ci(d.get("p50_ci95")),
                    "d_p90": d.get("p90"),
                    "d_p99": d.get("p99"),
                }
            )
        row["timing"] = timing
        out["yamls"].append(row)
    return out


def stage08(run_dir):
    """Every dataset of the latest stage08 run: the model and Origami on the
    same GEMMs (the paired statistics of its summary's `overall` block)."""
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
        ci = paired.get("speedup_ci95")
        rows.append(
            {
                "name": x.get("name"),
                "rc": x.get("rc"),
                "paired_n": paired.get("n"),
                "model": _pct(paired.get("model_sel_eff")),
                "origami": _pct(paired.get("origami_sel_eff")),
                "wins": paired.get("wins"),
                "ties": paired.get("ties"),
                "losses": paired.get("losses"),
                "speedup": _num(paired.get("speedup_geomean")),
                "speedup_ci95": ci if isinstance(ci, list) and len(ci) == 2 else None,
            }
        )
    return rows


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
    cat = {c: {"n": 0, "s": [], "p": [], "o": []} for c in SHAPE_CATEGORIES}
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
            pe = inherit(leaf, "paired_sel_eff_new")
            oe = inherit(leaf, "origami_sel_eff_new")
            if se:
                rec["s"].append(se)
            if pe and oe:
                rec["p"].append(pe)
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
        sm, pm, om = geo(r["s"]), geo(r["p"]), geo(r["o"])
        cat_rows.append(
            {
                "category": c,
                "n": r["n"],
                "held": round(sm, 2) if sm is not None else None,
                "paired": round(pm, 2) if pm is not None else None,
                "origami": round(om, 2) if om is not None else None,
                "delta": (
                    round(pm - om, 2) if pm is not None and om is not None else None
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


def _ci_txt(ci, scale=1.0, digits=2):
    if not ci:
        return ""
    return f" [{ci[0] * scale:.{digits}f}, {ci[1] * scale:.{digits}f}]"


def stage07_html(s7, origami_reference=False):
    if not s7:
        return '<p class="mut">no stage 7 results in this run.</p>'
    parts = []
    status = "all checks passed" if s7["ok"] else f"{s7['failures']} check(s) failed"
    parts.append(f'<p class="cap">stage 7 status: <b>{esc(status)}</b>.</p>')
    for y in s7["yamls"]:
        par = y.get("parity")
        sel = y.get("runtime_selection")
        rows = []
        if par:
            counts = ", ".join(
                f"{esc(k)}={esc(v)}" for k, v in sorted(par["counts"].items())
            )
            rows.append(
                f'<tr><td class="c">pick parity</td><td colspan="6" style="text-align:left">'
                f'{esc(par["agree"])} / {esc(par["n"])} problems agree ({_f(par["rate"])}%)'
                f" &mdash; {counts}</td></tr>"
            )
        if sel:
            se = sel.get("sel_eff")
            without = ", ".join(f"{esc(k)}={esc(v)}" for k, v in sel["without"].items())
            rows.append(
                f'<tr><td class="c">runtime selection efficiency</td>'
                f'<td colspan="6" style="text-align:left"><b>'
                f"{_f(se * 100 if se is not None else None)}%</b>"
                f'{esc(_ci_txt(sel.get("ci95"), 100))} of the fastest measured kernel over '
                f'{esc(sel.get("n"))} of {esc(sel.get("n_problems"))} problems; '
                f'{esc(sel.get("n_miss"))} misses (pick_measurement '
                f'{esc(sel.get("pick_measurement"))}); p5 '
                f'{_f(sel["p5"] * 100 if sel.get("p5") is not None else None)}%'
                f"{('; not evaluated: ' + without) if without else ''}</td></tr>"
            )
        for t in y.get("timing") or []:
            rows.append(
                f'<tr><td class="c">selection time, rsn={esc(t["rsn"])}</td>'
                f'<td>{esc(t["n"])} problems, {esc(t["reps"])} reps</td>'
                f'<td><b>{_f(t["on_p50"])}</b>{esc(_ci_txt(t.get("on_p50_ci")))}</td>'
                f'<td>{_f(t["on_p99"])} / {_f(t["on_max"])}</td>'
                f'<td>{_f(t["first_p50"])}{esc(_ci_txt(t.get("first_p50_ci")))}'
                f' (n={esc(t.get("first_n"))}, max {_f(t.get("first_max"))})</td>'
                f'<td>{_f(t["warm_p50"])}{esc(_ci_txt(t.get("warm_p50_ci")))}'
                f' (n={esc(t.get("warm_n"))}, max {_f(t.get("warm_max"))})</td>'
                + (
                    f'<td>{_f(t["off_p50"])} / <b>{_f(t["d_p50"], signed=True)}</b>'
                    f'{esc(_ci_txt(t.get("d_p50_ci")))} / {_f(t["d_p99"], signed=True)}</td>'
                    if origami_reference
                    else "<td></td>"
                )
                + "</tr>"
            )
        parts.append(
            f'<h3>{esc(y["yaml"])} <span class="mut">({esc(y.get("n_kept"))} of '
            f'{esc(y.get("n_entries"))} problems match the config)</span></h3>'
            '<table style="max-width:1400px"><thead><tr><th class="c">check</th>'
            "<th>problems</th><th>ML-on p50 &micro;s [95% CI]</th><th>ML-on p99 / max</th>"
            "<th>first call into a cell p50 [95% CI]</th><th>warm p50 [95% CI]</th>"
            + (
                "<th>[reference] ML-off p50 / &Delta; p50 [95% CI] / &Delta; p99</th>"
                if origami_reference
                else "<th></th>"
            )
            + "</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )
    return "".join(parts)


def stage08_html(rows, target):
    """Validation datasets against the fastest measured kernel: the stage's
    own number (model picks and the runtime's fallback, as deployed) and the
    model's own picks per leaf."""
    if not rows:
        return '<p class="mut">no stage 8 results in this run.</p>'
    trs = []
    for r in rows:
        cells = [c for c in r["cells"].values() if c.get("sel_eff") is not None]
        below = sum(1 for c in cells if c["sel_eff"] < target)
        se, dep = r.get("sel_eff"), r.get("deployed")
        trs.append(
            f'<tr><td class="c">{esc(r["name"])}</td><td>{esc(r["rc"])}</td>'
            f'<td>{esc(r.get("pick_measurement") or "")}</td>'
            f'<td>{esc(r.get("n_evaluated")) if r.get("n_evaluated") is not None else "&ndash;"}</td>'
            f'<td>{esc(r.get("n_miss_deployed")) if r.get("n_miss_deployed") is not None else "&ndash;"}</td>'
            f"<td><b>{_f(dep * 100 if dep is not None else None)}</b></td>"
            f"<td>{_f((dep - target) * 100 if dep is not None else None, signed=True)}</td>"
            f'<td>{esc(r["n_served"]) if r["n_served"] is not None else "&ndash;"}</td>'
            f"<td>{_f(se * 100 if se is not None else None)}</td>"
            f"<td>{below} / {len(cells)}</td></tr>"
        )
    return (
        '<p class="cap">Report only: these datasets never steer sampling, training or the '
        "cells chosen for training. <b>deployed %</b> = the stage's own number: geomean of "
        "fastest measured / pick over every evaluated GEMM, the model's pick or, where the "
        "model scores nothing that runs, the runtime's fallback; a miss is a pick without a "
        "timing of its own under the pick measurement shown. <b>model picks %</b> = over the "
        "GEMMs the model serves itself (the per-cell bars below).</p>"
        '<table style="max-width:1200px"><thead><tr><th class="c">dataset</th><th>rc</th>'
        "<th>pick measurement</th><th>GEMMs evaluated</th><th>misses</th><th>deployed %</th>"
        "<th>vs target pp</th><th>served by the model</th><th>model picks %</th>"
        "<th>cells below target</th></tr></thead><tbody>"
        + "".join(trs)
        + "</tbody></table>"
    )


def stage07_tiles(s7):
    """Per bench yaml: the runtime's selection efficiency with its interval,
    and at the smallest request size the ML-on median on the first call into
    a cell and warm, with their intervals."""
    sel, first, warm = [], [], []
    for y in (s7 or {}).get("yamls", []):
        r = y.get("runtime_selection") or {}
        if r.get("sel_eff") is not None:
            ci = r.get("ci95")
            sel.append(
                [
                    y["yaml"],
                    round(r["sel_eff"] * 100, 2),
                    [round(ci[0] * 100, 2), round(ci[1] * 100, 2)] if ci else None,
                ]
            )
        t = (y.get("timing") or [None])[0]
        if t and t.get("first_p50") is not None:
            first.append([y["yaml"], t["rsn"], t["first_p50"], t.get("first_p50_ci")])
        if t and t.get("warm_p50") is not None:
            warm.append([y["yaml"], t["rsn"], t["warm_p50"], t.get("warm_p50_ci")])
    return {"sel": sel, "first": first, "warm": warm}


def shipped_html(sb, target):
    """The shipped bundle: which bundle, its measurements on GEMMs it never
    trained on, the deploy gate's comparison with the bundle it replaced on
    the same GEMMs, and a later bundle the gate kept out."""
    if not sb or not sb.get("shipped"):
        final = (sb or {}).get("final_bundle")
        msg = '<p class="mut">this run deployed no bundle.</p>'
        if final:
            msg += (
                f'<p class="cap">{esc(final.get("round"))}: staged, not deployed '
                f'({esc(final.get("reason"))}).</p>'
            )
        return msg
    sh = sb["shipped"]
    parts = [
        f'<p class="cap">Shipped bundle: <b>{esc(sh.get("round"))}</b> '
        f'(models {esc(os.path.basename(os.path.dirname(str(sh.get("train_dir") or ""))))}/'
        f'{esc(os.path.basename(str(sh.get("train_dir") or "")))}, model file sha256 '
        f'{esc(str(sh.get("model_file_sha256") or "")[:12])}). Headline source: '
        f'<b>{esc(sb.get("source") or "none")}</b>'
        + (
            " (no measurement on GEMMs it did not train on)"
            if not sb.get("source")
            else ""
        )
        + ".</p>"
    ]
    ms = sb.get("measurements") or []
    head = next(
        (
            i
            for i, m in enumerate(ms)
            if m["source"] == sb.get("source")
            and m.get("sel_eff") == sb.get("held_out_sel_eff")
        ),
        None,
    )
    if ms:
        trs = "".join(
            f'<tr{" style=font-weight:600" if i == head else ""}>'
            f'<td class="c">{esc(m["source"])}</td><td class="c">{esc(m["round"])}</td>'
            f'<td>{intf_(m.get("n_eval"))}</td><td>{intf_(m.get("n_miss"))}</td>'
            f'<td>{len(m.get("per_cell") or {})}</td>'
            f'<td><b>{_f(m["sel_eff"] * 100 if m.get("sel_eff") is not None else None)}</b></td>'
            f'<td>{_f((m["sel_eff"] - target) * 100 if m.get("sel_eff") is not None else None, signed=True)}</td></tr>'
            for i, m in enumerate(ms)
        )
        parts.append(
            '<p class="cap">Every measurement of this bundle on GEMMs it never trained on: '
            "<b>deploy_gate</b> = stage06's gate slice of the round that shipped it, "
            "<b>stage04b</b> = a later round's new GEMMs evaluated with the same models.pt. "
            "Picks are charged under the pick measurement of the stage; the runtime's fallback "
            "counts where the model scores nothing that runs.</p>"
            '<table style="max-width:900px"><thead><tr><th class="c">source</th>'
            '<th class="c">round</th><th>GEMMs</th><th>misses</th><th>cells</th>'
            "<th>sel_eff %</th><th>vs target pp</th></tr></thead><tbody>"
            f"{trs}</tbody></table>"
        )
    gate = sb.get("gate")
    if gate:
        base = gate.get("baseline") or {}
        line = (
            f"Deploy gate: {intf_(gate.get('n_evaluated'))} of {intf_(gate.get('n_gemms'))} "
            f"gate GEMMs evaluated, pick_measurement {esc(gate.get('pick_measurement'))}, "
            f"target {esc(gate.get('target'))}, "
            f"{'passed' if gate.get('passed') else 'not passed'} ({esc(gate.get('reason'))})."
        )
        if base.get("n"):
            line += (
                f" On the same {intf_(base['n'])} GEMMs the bundle reaches "
                f"<b>{_f((base.get('bundle_sel_eff') or 0) * 100)}%</b> and the bundle it "
                f"replaced ({esc(os.path.basename(os.path.dirname(str(base.get('train_dir') or ''))))}) "
                f"{_f((base.get('sel_eff') or 0) * 100)}% (paired)."
            )
        elif base.get("error"):
            line += f" Baseline not measured: {esc(base['error'])}."
        parts.append(f'<p class="cap">{line}</p>')
    final = sb.get("final_bundle")
    if final:
        parts.append(
            f'<p class="cap"><b>Not deployed:</b> {esc(final.get("round"))} staged a later '
            f'bundle that its gate kept out ({esc(final.get("reason"))}; gate '
            f'{_f((final.get("gate_sel_eff") or 0) * 100 if final.get("gate_sel_eff") is not None else None)}%). '
            "The headline, the leaf-cell table and the bench analysis are of the shipped "
            "bundle above.</p>"
        )
    return "".join(parts)


def intf_(v):
    return (
        f"{int(v):,}"
        if isinstance(v, (int, float)) and not isinstance(v, bool)
        else "&ndash;"
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


def category_html(cat_rows, base_rows, n_total, target):
    if not cat_rows:
        return '<p class="mut">no held-out shapes in this run.</p>'
    rows = "".join(
        f'<tr{" class=lose" if (r["held"] is not None and r["held"] < target * 100) else ""}>'
        f'<td class="c">{esc(r["category"])}</td><td>{r["n"]:,}</td>'
        f'<td>{_f(r["held"])}</td>'
        f'<td>{_f(r["held"] - target * 100 if r["held"] is not None else None, signed=True)}</td></tr>'
        for r in cat_rows
    )
    out = (
        f'<p class="cap">{n_total:,} held-out GEMMs (the new shapes of every active round, '
        "without re-used history shapes). Each GEMM counts with the held-out selection "
        "efficiency of the leaf cell it routes to.</p>"
        '<table style="max-width:820px"><thead><tr><th class="c">category</th><th>GEMMs</th>'
        "<th>model %</th><th>vs target pp</th></tr></thead>"
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


def origami_reference_html(agg, traj, cells, cat_rows, s8):
    """Opt-in external reference: the analytical Origami pick next to the
    model, always on the same GEMMs. Nothing in the pipeline or in the rest
    of this report uses it."""

    def tab(head, rows):
        ths = "".join(
            f'<th class="c">{h}</th>' if i == 0 else f"<th>{h}</th>"
            for i, h in enumerate(head)
        )
        return (
            '<div class="scroll"><table style="margin-bottom:0;font-size:11px"><thead><tr>'
            f"{ths}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
        )

    def d(a, b):
        return a - b if a is not None and b is not None else None

    parts = [
        '<p class="cap">External reference only: Origami is the analytical model hipBLASLt '
        "ranks with when tilewright is off. No ranking, gap, action or cell selection of the "
        "pipeline or of this report depends on these numbers. Every comparison is paired: "
        "the model and Origami on the same GEMMs (those with a measured Origami pick), so "
        "the model's numbers here can differ from the headline over every GEMM.</p>",
        f'<p class="cap">Shipped bundle on {intf_(agg.get("paired_n"))} held-out GEMMs: model '
        f'{_f(agg.get("paired_model"))}%, Origami {_f(agg.get("paired_origami"))}%, &Delta; '
        f'{_f(agg.get("delta"), signed=True)} pp; per-cell best of both '
        f'{_f(agg.get("router_bound"))}%, model ahead in '
        f'{_f(agg.get("model_win_frac"), 1)}% of the cells.</p>',
        "<h3>Held-out per round (stage04b)</h3>",
        tab(
            ["round", "same GEMMs", "model %", "Origami %", "&Delta; pp"],
            [
                f'<tr><td class="c">{esc(r["round"])}</td><td>{intf_(r.get("paired_n"))}</td>'
                f'<td>{_f(r["paired_model"])}</td><td>{_f(r["paired_origami"])}</td>'
                f'<td>{_f(d(r["paired_model"], r["paired_origami"]), signed=True)}</td></tr>'
                for r in traj
            ],
        ),
        "<h3>Leaf cells of the shipped bundle (held-out)</h3>",
        tab(
            ["cell", "model %", "Origami %", "&Delta; pp"],
            [
                f'<tr><td class="c">{esc(c["cell"])}</td><td>{_f(c["held_paired"])}</td>'
                f'<td>{_f(c["held_ori"])}</td><td>{_f(c["held_delta"], signed=True)}</td></tr>'
                for c in cells
                if c.get("held_paired") is not None and c.get("held_ori") is not None
            ],
        ),
        '<p class="mut">stage05\'s in-sample Origami numbers are over a different set of '
        "GEMMs than the model's and are not shown.</p>",
    ]
    if cat_rows:
        parts += [
            "<h3>Held-out GEMMs by shape category</h3>",
            tab(
                ["category", "GEMMs", "model %", "Origami %", "&Delta; pp"],
                [
                    f'<tr><td class="c">{esc(r["category"])}</td><td>{r["n"]:,}</td><td>{_f(r["paired"])}</td>'
                    f'<td>{_f(r["origami"])}</td><td>{_f(r["delta"], signed=True)}</td></tr>'
                    for r in cat_rows
                ],
            ),
        ]
    if s8:
        parts += [
            "<h3>Stage 8 datasets</h3>",
            tab(
                [
                    "dataset",
                    "same GEMMs",
                    "model %",
                    "Origami %",
                    "&Delta; pp",
                    "wins / ties / losses",
                    "speedup vs Origami [95% CI]",
                ],
                [
                    f'<tr><td class="c">{esc(r["name"])}</td><td>{intf_(r["paired_n"])}</td><td>{_f(r["model"])}</td>'
                    f'<td>{_f(r["origami"])}</td><td>{_f(d(r["model"], r["origami"]), signed=True)}</td>'
                    f'<td>{esc(r["wins"])} / {esc(r["ties"])} / {esc(r["losses"])}</td>'
                    f'<td>{_f(r["speedup"], 3)}{esc(_ci_txt(r.get("speedup_ci95"), 1, 3))}</td></tr>'
                    for r in s8
                ],
            ),
        ]
    return (
        '<h2 id="origami">External reference: Origami</h2>'
        "<details><summary>show the Origami comparison (opt-in, not used by the pipeline)</summary>"
        + "".join(parts)
        + "</details>"
    )


_DEFAULT_CELL_KEYS = (
    "cell",
    "n_gemms",
    "model",
    "deployed",
    "k_delta",
    "held",
    "held_inherited",
    "smartK",
    "cap",
)


def _analysis(run_dir, jobs, engine, notes, pick_measurement, diagnostics):
    rs.pipeline_on_path()
    try:
        from lib import run_analysis

        return run_analysis.analyze_run(
            run_dir,
            jobs=jobs,
            use_engine=engine,
            pick_measurement=pick_measurement,
            diagnostics=diagnostics,
        )
    except Exception as e:  # noqa: BLE001
        notes.append(f"bench analysis failed: {type(e).__name__}: {e}")
        return None


def _stage08_decomposition(run_dir, s8rows, jobs, engine, notes, pick_measurement):
    """Decomposition of every stage08 dataset with enriched chunks (report
    only), under the shipped bundle and its pool, with the dataset's own pick
    measurement (else the config's)."""
    rs.pipeline_on_path()
    try:
        from lib import run_analysis as ra
        from lib.subcells import load_cumulative_split_tree

        model = ra.final_model(run_dir, want_bytes=engine)
        if model is None:
            return {}
        tw = ra.engine_module(model) if engine else None
        lib_pool = ra.library_pool(model)
        tree = load_cumulative_split_tree(ra.round_dirs(run_dir))
        out = {}
        s8dir = os.path.join(run_dir, "validate", "stage08")
        for r in s8rows:
            ds = r["dataset"]
            enriched = ra.stage08_enriched_dir(ra.Path(s8dir), ds)
            if enriched is None:
                continue
            summ = ds.get("summary") or {}
            bench = ra.load_bench(
                [enriched],
                tree=tree,
                jobs=jobs,
                keep=ra.workload_keep(summ.get("workload_filter") or {}),
            )
            if not bench.gemms:
                continue
            e = ra.evaluate(
                bench,
                ra.pool_for(bench, lib_pool),
                model,
                tw=tw,
                pick_measurement=summ.get("pick_measurement") or pick_measurement,
                restrict_to_pool=True,
            )
            out[str(r["name"])] = {
                "engine": e.engine,
                "pick_measurement": e.pick_measurement,
                "decomposition": ra.decomposition(e),
            }
        return out
    except Exception as e:  # noqa: BLE001
        notes.append(f"stage 8 decomposition failed: {type(e).__name__}: {e}")
        return {}


def subset_html(path):
    doc = load_json(path)
    if not isinstance(doc, dict) or not doc.get("sizes"):
        return (
            '<p class="mut">no shipped_subset.json (scripts/choose_shipped_subset.py) '
            "for this run.</p>"
        )
    _pipeline_on_path()
    from lib.shipped_subset import report_html

    return report_html(doc, esc)


def gate_check_html(analysis, sb):
    """The report's re-ranking of the deploy-gate slice next to the gate's
    own record (they agree when the report charges picks as stage06 does)."""
    gate = (sb or {}).get("gate")
    mine = ((analysis or {}).get("gate") or {}).get("decomposition", {}).get("*")
    if not gate or not mine:
        return ""
    se = mine.get("deployed_sel_eff")
    g = gate.get("sel_eff")
    agree = (
        se is not None
        and g is not None
        and abs(se - g) <= 1e-6
        and mine.get("n_deployed") == gate.get("n_evaluated")
    )
    return (
        f'<p class="cap">Deploy-gate slice re-ranked by this report: '
        f"<b>{_f(se * 100 if se is not None else None, 4)}%</b> over "
        f'{intf_(mine.get("n_deployed"))} GEMMs ({intf_(mine.get("n_miss"))} misses); '
        f"stage06/gate.json: {_f(g * 100 if g is not None else None, 4)}% over "
        f'{intf_(gate.get("n_evaluated"))} ({intf_(gate.get("n_miss"))} misses) &mdash; '
        f'{"they agree" if agree else "<b>they differ</b>"}.</p>'
    )


def _evidence(run_dir, settings, notes, gemm_counts=None):
    rs.pipeline_on_path()
    try:
        from lib import cell_evidence as ce

        evidence = ce.collect(ce.Path(run_dir), gemm_counts)
        as_enhance = ce.collect(ce.Path(run_dir))
    except Exception as e:  # noqa: BLE001
        notes.append(f"cell evidence failed: {type(e).__name__}: {e}")
        return {}, []
    t = settings["target"]
    selections = []
    for n in (5, 10):
        for rank in ("gain", "sel_eff"):
            picked = ce.select_weak_cells(
                as_enhance, weakest=n, below=t, rank_by=rank, target=t
            )
            selections.append(
                (f"--weakest {n} --rank-by {rank}", [e.cell for e in picked])
            )
    below = ce.select_weak_cells(as_enhance, below=t, target=t)
    selections.append((f"--below {t:g}", [e.cell for e in below]))
    return evidence, selections


def _train_round(sb):
    td = str(((sb or {}).get("shipped") or {}).get("train_dir") or "")
    name = os.path.basename(os.path.dirname(td))
    return name if round_num(name) >= 0 else None


def _analysis_caption(analysis, what):
    model = analysis.get("model") or {}
    deployed = (
        f"the shipped bundle (deploy record of {model.get('deploy_round')}, models of "
        f"{model.get('train_round')})"
        if model.get("deployed")
        else f"the newest trained model ({model.get('round')})"
    )
    return (
        f'<p class="cap">{what}: {analysis["n_gemms"]:,} GEMMs under {esc(deployed)}, '
        f'model file from {esc(model.get("source"))}, {esc(model.get("n_cells"))} cells, '
        f'non-temporal rule {"on" if model.get("non_temporal_rule") else "off"}, cascade keys '
        f'{esc(model.get("cascade_keys"))}; pool: {esc(analysis["pool"]["size"])} solutions '
        f'({esc(analysis["pool"].get("signatures"))} kernel signatures) from the '
        f'{esc(analysis["pool"]["source"])}; pick measurement '
        f'<code>{esc(analysis.get("pick_measurement"))}</code>'
        + (
            f'; {analysis["n_out_of_pool_rows"]:,} measured rows of kernels outside the pool '
            "(oracle candidates only)"
            if analysis.get("n_out_of_pool_rows")
            else ""
        )
        + (
            f'; {analysis["n_inconsistent"]} GEMMs where the engine scored a kernel outside '
            "the Python tier 1"
            if analysis.get("n_inconsistent")
            else ""
        )
        + ".</p>"
        + "".join(f'<p class="mut">{esc(n)}</p>' for n in model.get("notes") or [])
    )


def build(
    run_dir,
    out_path,
    *,
    jobs=1,
    engine=True,
    bench=True,
    origami_reference=False,
    selection_budget_us=10.0,
    target=None,
    diagnostics=False,
    shipped_subset=None,
):
    run_dir = run_dir.rstrip("/")
    notes = []
    config_text = config_source(run_dir)
    settings = rs.run_settings(config_text)
    if target is not None:
        settings["target"] = float(target)
    tgt = settings["target"]
    pick_measurement = settings["pick_measurement"]
    sb = shipped_bundle(run_dir) or {}
    shipped_cells = sb.get("per_cell") or {}
    latest = held_out_breakdown(run_dir)
    held = shipped_cells or latest
    held_note = (
        f"the shipped bundle on GEMMs it never trained on ({esc(sb.get('source'))})"
        if shipped_cells
        else "no measurement of the shipped bundle on GEMMs it never trained on: the "
        "latest stage04b measurement per leaf (each by the model of the round before it)"
    )
    agg = held_out_aggregate(held)
    traj = held_out_rounds(run_dir)
    train = training_rounds(run_dir)
    cells_full = final_cells(run_dir, held, upto=_train_round(sb))
    s7 = stage07(run_dir)
    s8 = stage08(run_dir)
    t_rounds, t_total = timing_rows(run_dir)
    cat_rows, base_rows, n_cat = held_out_by_category(run_dir, held)
    in_sample = [c["model"] for c in cells_full if c.get("model") is not None]

    analysis = (
        _analysis(run_dir, jobs, engine, notes, pick_measurement, diagnostics)
        if bench
        else None
    )
    decomp = (analysis or {}).get("decomposition") or {}
    counts = {c: d["n_gemms"] for c, d in decomp.items() if c != "*"} or None
    evidence, selections = _evidence(run_dir, settings, notes, counts)
    improve = rs.improve_rows(evidence, decomp, settings)

    s8_summary = load_json(os.path.join(run_dir, "validate", "stage08", "summary.json"))
    s8rows = (
        rs.stage08_model_rows(os.path.join(run_dir, "validate", "stage08"), s8_summary)
        if isinstance(s8_summary, dict)
        else []
    )
    s8_dec = (
        _stage08_decomposition(run_dir, s8rows, jobs, engine, notes, pick_measurement)
        if bench and s8rows
        else {}
    )
    s7_summary = load_json(os.path.join(run_dir, "validate", "stage07", "summary.json"))
    s7_cells = []
    if isinstance(s7_summary, dict):
        rs.pipeline_on_path()
        try:
            from lib.cell_evidence import model_labels, split_tree

            s7_cells = rs.stage07_per_cell(
                os.path.join(run_dir, "validate", "stage07"),
                s7_summary,
                labels=model_labels(rs.Path(run_dir)),
                tree=split_tree(rs.Path(run_dir)),
            )
        except Exception as e:  # noqa: BLE001
            notes.append(f"stage 7 per-cell timing failed: {type(e).__name__}: {e}")
    try:
        s4b_ctx = stage04b_contexts(run_dir)
    except Exception as e:  # noqa: BLE001
        notes.append(f"stage04b context results unreadable: {type(e).__name__}: {e}")
        s4b_ctx = []

    def rt_eff():
        num = den = 0.0
        for c in cells_full:
            n, w, p = c.get("n_gemms") or 0, c.get("held_w_us"), c.get("held_pick_us")
            if n and w and p:
                num += n * w
                den += n * p
        return round(num / den * 100, 2) if den else None

    cells = []
    for c in cells_full:
        row = {k: c.get(k) for k in _DEFAULT_CELL_KEYS}
        row["gap"] = (
            round(c["held"] - tgt * 100, 2) if c.get("held") is not None else None
        )
        cells.append(row)
    all_dec = decomp.get("*") or {}
    shipped = sb.get("shipped") or {}
    gate = sb.get("gate") or {}
    base = gate.get("baseline") or {}
    final = sb.get("final_bundle") or {}
    summary = {
        "run": os.path.basename(run_dir),
        "rounds": len(round_dirs(run_dir)),
        "target": round(tgt * 100, 2),
        "pick_measurement": pick_measurement,
        "shipped": {
            "round": shipped.get("round"),
            "train_round": _train_round(sb),
            "source": sb.get("source"),
            "sel_eff": _pct(sb.get("held_out_sel_eff")),
            "n": sb.get("n_gemms") if sb.get("source") else None,
            "n_miss": sb.get("n_miss"),
            "n_cells": sb.get("n_cells"),
        },
        "baseline": (
            {
                "n": base["n"],
                "bundle": _pct(base.get("bundle_sel_eff")),
                "base": _pct(base.get("sel_eff")),
            }
            if base.get("n")
            else None
        ),
        "not_deployed": (
            {"round": final.get("round"), "reason": final.get("reason")}
            if final
            else None
        ),
        "held": {
            "model": agg.get("model"),
            "n_cells": agg.get("n_cells"),
            "n_miss": agg.get("n_miss"),
            "shipped": bool(shipped_cells),
        },
        "in_sample_mean": (
            round(sum(in_sample) / len(in_sample), 2) if in_sample else None
        ),
        "rt_model": rt_eff(),
        "n_cells": len(cells),
        "bin_mb": deployed_size_mb(run_dir),
        "shapes": shapes_total(run_dir),
        "wall": fmt_dur(t_total) if t_total else None,
        "s7": stage07_tiles(s7),
        "n_action": sum(1 for r in improve if r["action"] != "none"),
        "gain": round(sum(r["gain"] for r in improve) * 100, 2) if improve else None,
        "bench_sel_eff": _pct(all_dec.get("sel_eff")),
        "bench_miss": all_dec.get("n_miss"),
        "bench_unserved": (
            all_dec.get("n_gemms", 0) - all_dec.get("n_served", 0) if all_dec else None
        ),
        "bench_gemms": all_dec.get("n_gemms"),
    }
    s8pc = {
        str(r["name"]): sorted(
            [c, round(v["sel_eff"] * 100, 2), v["n_served"]]
            for c, v in r["cells"].items()
            if v.get("sel_eff") is not None
        )
        for r in s8rows
        if r.get("name") and r["cells"]
    }
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
            f'<td>{esc(r["n_cells"])}</td><td>{esc(r["n_gemms"])}</td></tr>'
            for r in train
        )
        or '<tr><td class="c" colspan="5">no stage05 metrics.json in this run</td></tr>'
    )
    traj_js = [
        {
            "round": r["round"],
            "model": r["model"],
            "n_gemms": r["n_gemms"],
            "n_miss": r["n_miss"],
        }
        for r in traj
    ]
    na = '<p class="mut">no training benches (round_*/stage04 chunks) in this run.</p>'
    params, param_cells, dparams, dparam_cells = {}, {}, {}, {}
    if analysis:
        held_out = analysis.get("held_out")
        sections = {
            "DECOMP": _analysis_caption(
                analysis, "Training benches of the shipped bundle"
            )
            + rs.decomposition_html(analysis, order=base_order),
            "DECOMP_HELD": (
                f'<p class="cap">{held_out["n_gemms"]:,} GEMMs the shipped bundle never '
                "trained on: the deploy-gate slice of its training round and the stage04 "
                "rows of later rounds, under the same model, pool and pick measurement as "
                "the training benches.</p>"
                + gate_check_html(analysis, sb)
                + rs.decomposition_html(
                    dict(
                        held_out,
                        engine=analysis.get("engine"),
                        pick_measurement=analysis.get("pick_measurement"),
                    ),
                    order=base_order,
                )
                if held_out
                else '<p class="mut">no bench of GEMMs the shipped bundle never trained '
                "on (no deploy-gate slice and no later round).</p>"
            ),
            "KERNELS": rs.kernels_html(analysis) + rs.cell_usage_html(analysis),
            "PARAMS": rs.parameters_html(),
            "DIAG": rs.diagnostics_html(analysis),
            "PRUNE": rs.pruning_html(analysis),
        }
        p = analysis.get("parameters") or {}
        params, param_cells = p.get("params") or {}, p.get("by_cell") or {}
        dg = p.get("diagnostics") or {}
        dparams, dparam_cells = dg.get("params") or {}, dg.get("by_cell") or {}
    else:
        sections = {
            "DECOMP": na,
            "DECOMP_HELD": na,
            "KERNELS": na,
            "PARAMS": na,
            "DIAG": "",
            "PRUNE": na,
        }
    if diagnostics and not dparams:
        sections["DIAG"] = (
            '<h2 id="diagnostics">Diagnostics: framework-specific kernel parameters</h2>'
            '<p class="mut">no bench analysis to break down.</p>'
        )
    s8_dec_html = "".join(
        f"<details><summary>{esc(name)}: loss decomposition (validation, report only)</summary>"
        + rs.decomposition_html(d, order=base_order)
        + "</details>"
        for name, d in sorted(s8_dec.items())
    )
    contexts = (
        rs.contexts_html(analysis, s4b_ctx, settings, context_names(config_text))
        if rs.has_contexts(analysis, s4b_ctx)
        else '<p class="mut">every row of this run is of the exclusive context.</p>'
    )
    data = {
        "TRAJ": traj_js,
        "CELLS": cells,
        "S": summary,
        "S8PC": s8pc,
        "PARAMS": params,
        "PCELLS": param_cells,
        "DPARAMS": dparams,
        "DPCELLS": dparam_cells,
    }
    parts = {
        "TITLE": esc(summary["run"]),
        "NOTES": "".join(f'<p class="mut">{esc(n)}</p>' for n in notes),
        "SHIPPED": shipped_html(sb, tgt),
        "SUBSET": subset_html(
            shipped_subset or os.path.join(run_dir, "shipped_subset.json")
        ),
        "IMPROVE": rs.improve_html(improve, settings, selections),
        "CONTEXTS": contexts,
        "SELTIME": rs.selection_time_html(
            s7_cells, float(selection_budget_us), origami_reference
        ),
        "S7": stage07_html(s7, origami_reference),
        "S8": stage08_html(s8rows, tgt) + s8_dec_html,
        "TRAINROWS": train_rows,
        "CHART_TREND": chart_trend(train),
        "HELD_NOTE": held_note,
        "CHART_HELD": chart_hist(
            cells_full,
            "held",
            "Held-out selection efficiency per leaf cell (GEMMs the model did not train on)",
        ),
        "CHART_INSAMPLE": chart_hist(
            cells_full,
            "deployed",
            "In-sample selection efficiency per leaf cell (its own training GEMMs)",
        ),
        "CATEGORIES": category_html(cat_rows, base_rows, n_cat, tgt),
        "TREE": esc(cell_tree_ascii(cells_full)),
        "CELL3D": cell3d,
        "TIMING": timing_html(t_rounds, t_total),
        "CONFIG": esc(config_text),
        "ORIGAMI": (
            origami_reference_html(agg, traj, cells_full, cat_rows, s8)
            if origami_reference
            else ""
        ),
        "DATA": script_json(_finite(data)),
        **sections,
    }
    page = fill(_HTML, parts)
    with open(out_path, "w") as f:
        f.write(page)
    return out_path


_SLOT = re.compile(r"__([A-Z][A-Z0-9_]*)__")


def fill(template, parts):
    """`template` with every __NAME__ slot of `parts` replaced in one pass,
    so text inserted for one slot is never searched for another."""
    return _SLOT.sub(lambda m: parts.get(m.group(1), m.group(0)), template)


def base_order(label):
    base, *segs = (label or "").split("#")
    return (base_cell_key(base), [seg_key(s) for s in segs])


_HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Run report - __TITLE__</title><style>
:root{--bd:#d8dee4;--mut:#57606a;--win:#e6ffec;--lose:#ffebe9;--weak:#fff8c5;}
body{font:14px/1.45 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#1f2328;margin:0;padding:24px;max-width:1760px;}
p,h1,h2,.stats,details>summary{max-width:1200px;}
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
details{margin:6px 0;} summary{cursor:pointer;color:#0969da;}
</style></head><body>
<h1>Run report &mdash; __TITLE__</h1>
<p class="cap" id="cap"></p>
<div class="stats" id="stats"></div>
__NOTES__
<h2 id="shipped">Shipped bundle</h2>
__SHIPPED__
<p class="cap"><b>Cells</b> are <code>M-tier|N-tier|K-tier|Batch</code>: M and N Tiny &le;32, Small &le;128, Mid &le;512, Large &gt;512; K TinyK &le;32, MidK &le;512, LargeK &gt;512; Bnone = unbatched, Bany = batched. Split suffixes such as <code>#K&gt;4096</code> subdivide a cell. Selection efficiency of a GEMM = time of the fastest measured solution / time of the picked solution (every solution of the library counts with its own time, also when several share kernel parameters); aggregates are geometric means. Every target and gap in this report is absolute, against the fastest measured kernel. A <b>kernel</b> is its signature (macro tile M and N, depthU, matrix instruction M, N and K, cache hints A and B); the library solutions sharing one are its members.</p>

<h2 id="subset">Shipped candidate set</h2>
__SUBSET__

<h2 id="improve">Where to improve</h2>
__IMPROVE__

<h2 id="decomp">Loss decomposition per cell (training benches)</h2>
__DECOMP__

<h2 id="decomp_held">Loss decomposition per cell (GEMMs the shipped bundle never trained on)</h2>
__DECOMP_HELD__

<h2 id="contexts">Execution contexts</h2>
__CONTEXTS__

<h2 id="kernels">Kernel usage</h2>
__KERNELS__

<h2 id="params">Kernel parameters</h2>
__PARAMS__
__DIAG__

<h2 id="prune">Pool-drop candidates</h2>
__PRUNE__

<h2 id="seltime">Selection time per cell (stage 7)</h2>
__SELTIME__

<h2>Stage 7 &mdash; selection time and C++/Python pick parity</h2>
<p class="cap">Selection time is hipBLASLt's per-call "Solution selection time" with tilewright (ML-on) for the request size shown (rsn), median over repetitions per problem, summarized over problems; the first call of each run (library load) is excluded, and the rest split into the first call of a run into a serving cell and the warm calls after it. Intervals are bootstrap 95% intervals of the median. Pick parity compares the pool position of hipBLASLt's top-1 pick and its serving cell with the offline ranking of the same model and library; the runtime selection efficiency charges those top-1 picks with the bench timings of the same problems.</p>
__S7__

<h2>Stage 8 &mdash; selection efficiency on validation datasets</h2>
__S8__
<div id="s8charts"></div>

<h2>Held-out selection efficiency per round (stage04b)</h2>
<p class="cap">The model of the previous round on this round's new shapes (history; each row is another model). The last row is the shipped bundle's headline.</p>
<table id="traj" style="max-width:900px"><thead><tr><th class="c">round</th><th>model %</th><th>vs target pp</th><th>GEMMs</th><th>misses</th></tr></thead><tbody id="tbody_traj"></tbody></table>

<h2>Training (stage05, in-sample)</h2>
<p class="cap">Selection efficiency on the GEMMs of the cells trained in that round: model = full ranking, deployed = whitelist-restricted ranking as shipped.</p>
<table style="max-width:900px"><thead><tr><th class="c">round</th><th>model %</th><th>deployed %</th><th>cells trained</th><th>GEMMs</th></tr></thead><tbody>__TRAINROWS__</tbody></table>
__CHART_TREND__

<h2 id="cellh">Leaf cells of the shipped bundle</h2>
<p class="cap"><b>held-out</b> values: __HELD_NOTE__ (<b>*</b> = inherited from the parent cell); the other columns are in-sample. vs target = held-out minus the target. K = whitelist size; ed/hd/ih = embedding / hidden / interaction widths. Click a header to sort.</p>
<div class="scroll"><table id="cells" style="margin-bottom:0;font-size:11px"><thead><tr id="chead"></tr></thead><tbody id="tbody_cells"></tbody></table></div>
__CHART_HELD__
__CHART_INSAMPLE__

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
__ORIGAMI__

<script>
const D=__DATA__;
const TRAJ=D.TRAJ, CELLS=D.CELLS, S=D.S, S8PC=D.S8PC, T=S.target;
function esc(s){return String(s==null?"":s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&#39;");}
function num(v,d){return (v==null||!isFinite(v))?"\u2013":Number(v).toFixed(d==null?2:d);}
function sgn(v,d){return (v==null||!isFinite(v))?"\u2013":(v>=0?"+":"")+Number(v).toFixed(d==null?2:d);}
document.getElementById("cap").textContent="Run "+S.run+" \u2013 "+S.rounds+" rounds"+(S.wall?(" \u2013 wall-clock "+S.wall):"")+" \u2013 target "+num(T,1)+"% of the fastest measured kernel";
const SH=S.shipped||{},S7=S.s7||{};
const fmtn=v=>v!=null?Number(v).toLocaleString():"n/a";
const ci=c=>c?" ["+num(c[0])+", "+num(c[1])+"]":"";
const stats=[
  ["shipped bundle",SH.round?SH.round+(SH.train_round&&SH.train_round!==SH.round?" (models of "+SH.train_round+")":""):"none"],
  ["held-out sel_eff, shipped bundle"+(SH.source?" ("+SH.source+")":""),SH.sel_eff!=null?num(SH.sel_eff)+"%":"n/a"],
  ["held-out GEMMs / misses (pick measurement "+S.pick_measurement+")",SH.n!=null?fmtn(SH.n)+" / "+fmtn(SH.n_miss):"n/a"],
];
if(S.baseline)stats.push(["deploy gate: this bundle vs the one it replaced, same "+fmtn(S.baseline.n)+" GEMMs",num(S.baseline.bundle)+"% vs "+num(S.baseline.base)+"%"]);
if(S.not_deployed)stats.push(["latest bundle not deployed",S.not_deployed.round+(S.not_deployed.reason?": "+S.not_deployed.reason:"")]);
for(const [y,v,c] of S7.sel||[])stats.push(["runtime selection efficiency, stage 7 "+y+" [95% CI]",num(v)+"%"+ci(c)]);
for(const [y,r,v,c] of S7.first||[]){const w=(S7.warm||[]).find(x=>x[0]===y);stats.push(["selection time p50, first call into a cell / warm (stage 7 "+y+", rsn="+r+")",num(v)+ci(c)+" / "+(w?num(w[2])+ci(w[3]):"\u2013")+" \u00b5s"]);}
stats.push(
  ["runtime-weighted held-out, shipped bundle",S.rt_model!=null?num(S.rt_model)+"%":"n/a"],
  ["training benches, deployed picks / misses",S.bench_sel_eff!=null?num(S.bench_sel_eff)+"% / "+fmtn(S.bench_miss):"n/a"],
  ["training GEMMs without a model pick",S.bench_unserved!=null?fmtn(S.bench_unserved)+" of "+fmtn(S.bench_gemms):"n/a"],
  ["leaf cells needing an action",S.n_action!=null?S.n_action:"n/a"],
  ["expected gain if all reach the target",S.gain!=null?"+"+num(S.gain)+" pp":"n/a"],
  ["in-sample mean over leaf cells",S.in_sample_mean!=null?num(S.in_sample_mean)+"%":"n/a"],
  ["deployed model file",S.bin_mb!=null?S.bin_mb+" MB":"n/a"],
  ["shapes generated",S.shapes!=null?fmtn(S.shapes):"n/a"],
);
document.getElementById("stats").innerHTML=stats.map(([l,v])=>`<div class="stat"><div class="v">${esc(v)}</div><div class="l">${esc(l)}</div></div>`).join("");
document.getElementById("tbody_traj").innerHTML=TRAJ.map(r=>`<tr><td class="c">${esc(r.round)}</td><td>${num(r.model)}</td><td>${r.model!=null?sgn(r.model-T):"\u2013"}</td><td>${r.n_gemms!=null?Number(r.n_gemms).toLocaleString():""}</td><td>${r.n_miss!=null?Number(r.n_miss).toLocaleString():""}</td></tr>`).join("")
 +(SH.sel_eff!=null?`<tr style="font-weight:600"><td class="c">shipped bundle (${esc(SH.source)})</td><td>${num(SH.sel_eff)}</td><td>${sgn(SH.sel_eff-T)}</td><td>${fmtn(SH.n)}</td><td>${fmtn(SH.n_miss)}</td></tr>`:"");
const COLS=[["cell","cell"],["n_gemms","GEMMs"],["held","held-out %"],["gap","vs target"],["model","model %"],["deployed","deployed %"],["k_delta","K\u0394"],["smartK","K"],["cap","ed/hd/ih"]];
const STR=k=>(k==="cell"||k==="cap");
let sk="held",asc=true;
function rowClass(c){if(c.held==null)return"";if(c.held>=T)return"win";if(c.held<T-5)return"lose";return"weak";}
function cellText(c,k){const v=c[k];if(STR(k))return esc(v);if(k==="n_gemms")return Number(v||0).toLocaleString();if(k==="smartK")return v!=null?esc(v):"";if(k==="held")return v!=null?num(v)+(c.held_inherited?"*":""):"";if(k==="gap"||k.endsWith("delta"))return v!=null?sgn(v):"";return v!=null?num(v):"";}
function render(){CELLS.sort((a,b)=>{const d=STR(sk)?String(a[sk]||"").localeCompare(String(b[sk]||"")):((a[sk]??-1e9)-(b[sk]??-1e9));return asc?d:-d;});
 const head=document.getElementById("chead");head.innerHTML="<th>#</th>"+COLS.map(([k,t])=>`<th class="sort ${STR(k)?"c":""}" data-k="${esc(k)}">${esc(t)}${k===sk?(asc?" \u25b2":" \u25bc"):""}</th>`).join("");
 document.getElementById("tbody_cells").innerHTML=CELLS.map((c,i)=>`<tr class="${rowClass(c)}"><td class="c">${i+1}</td>`+COLS.map(([k])=>`<td class="${STR(k)?"c":""}">${cellText(c,k)}</td>`).join("")+"</tr>").join("");}
document.getElementById("chead").addEventListener("click",e=>{const th=e.target.closest("th[data-k]");if(!th)return;const k=th.dataset.k;if(k===sk)asc=!asc;else{sk=k;asc=STR(k);}render();});
render();
(function(){
 const host=document.getElementById("s8charts");const names=Object.keys(S8PC).sort();
 for(const ds of names){
  const rows=(S8PC[ds]||[]).slice();if(!rows.length)continue;
  const fig=document.createElement("figure");fig.className="chart";fig.style.maxWidth="1700px";
  const cap=document.createElement("figcaption");const ok=rows.filter(r=>r[1]>=T).length;
  cap.innerHTML=`<b>${esc(ds)}</b> \u2013 ${rows.length} cells, ${ok} at or above the ${num(T,1)}% target (bar = model \u2212 target). sort `;
  const sel=document.createElement("select");
  for(const [v,t] of [["dw","worst first"],["db","best first"],["la","label"]]){const o=document.createElement("option");o.value=v;o.textContent=t;sel.appendChild(o);}
  cap.appendChild(sel);fig.appendChild(cap);
  const box=document.createElement("div");box.className="s8bars";fig.appendChild(box);host.appendChild(fig);
  const draw=mode=>{const r2=rows.slice();
   if(mode==="db")r2.sort((a,b)=>b[1]-a[1]);else if(mode==="la")r2.sort((a,b)=>a[0]<b[0]?-1:a[0]>b[0]?1:0);else r2.sort((a,b)=>a[1]-b[1]);
   const maxd=Math.max(1,...r2.map(r=>Math.abs(r[1]-T)));
   const labw=Math.min(900,Math.max(220,Math.round(7.7*Math.max(0,...r2.map(r=>String(r[0]).length)))));
   box.innerHTML=r2.map(r=>{const d=r[1]-T,w=Math.abs(d)/maxd*116;const left=d>=0?120:120-w;
    return `<div class="s8r"><div class="s8lab" style="flex:0 0 ${labw}px" title="${esc(r[0])}">${esc(r[0])}</div><div class="s8bar"><div class="s8mid"></div><div class="s8fill" style="left:${left.toFixed(1)}px;width:${w.toFixed(1)}px;background:${d>=0?"#2da44e":"#cf222e"}"></div></div><div class="s8v">${num(r[1],1)}% n=${esc(r[2])} <b>${sgn(d,1)}</b></div></div>`;}).join("");};
  sel.addEventListener("change",()=>draw(sel.value));draw("dw");
 }
})();
function paramView(prefix,PARAMS,PCELLS){
 const s=document.getElementById(prefix+"_sel");if(!s)return;const names=Object.keys(PARAMS||{});if(!names.length)return;
 for(const n of names){const o=document.createElement("option");o.value=n;o.textContent=n;s.appendChild(o);}
 const pct=(a,b)=>b?(a/b*100).toFixed(1)+"%":"\u2013";
 const draw=()=>{const name=s.value,rows=PARAMS[name]||[];
  const W=rows.reduce((a,r)=>a+r.wins,0),P=rows.reduce((a,r)=>a+(r.picks||0),0);
  document.getElementById(prefix+"_tab").innerHTML=`<div class="scroll"><table style="margin-bottom:0;font-size:11px"><thead><tr><th class="c">${esc(name)}</th><th>kernels</th><th>wins</th><th>win share</th><th>picks</th><th>pick share</th><th>drop loss %</th><th>stranded GEMMs</th></tr></thead><tbody>`
   +rows.map(r=>`<tr><td class="c">${esc(r.value)}</td><td>${r.kernels}</td><td>${r.wins}</td><td>${pct(r.wins,W)}</td><td>${r.picks==null?"\u2013":r.picks}</td><td>${r.picks==null?"\u2013":pct(r.picks,P)}</td><td>${num(r.drop_loss*100,3)}</td><td>${r.stranded}</td></tr>`).join("")+"</tbody></table></div>";
  const top=rows.slice(0,10).map(r=>r.value),pc=(PCELLS||{})[name]||{},cs=Object.keys(pc).sort();
  document.getElementById(prefix+"_cells").innerHTML=`<div class="scroll"><table style="margin-bottom:0;font-size:11px"><thead><tr><th class="c">cell</th>`+top.map(v=>`<th>${esc(v)}</th>`).join("")+`<th>other</th></tr></thead><tbody>`
   +cs.map(c=>{const m=pc[c];let ow=0,op=0;for(const [v,[w,p]] of Object.entries(m)){if(!top.includes(v)){ow+=w;op+=p;}}
    return `<tr><td class="c">${esc(c)}</td>`+top.map(v=>{const x=m[v];return `<td>${x?x[0]+" / "+x[1]:""}</td>`;}).join("")+`<td>${ow||op?ow+" / "+op:""}</td></tr>`;}).join("")+"</tbody></table></div>";};
 s.addEventListener("change",draw);draw();
}
paramView("param",D.PARAMS,D.PCELLS);
paramView("diag",D.DPARAMS,D.DPCELLS);
</script></body></html>"""


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="HTML report of a pipeline run")
    ap.add_argument("run_dir")
    ap.add_argument(
        "out", nargs="?", default=None, help="default: <run_dir>/report.html"
    )
    ap.add_argument("--jobs", type=int, default=8, help="processes reading benches")
    ap.add_argument(
        "--no-engine",
        action="store_true",
        help="skip the tilewright ranking (no cascade / ranking / pick decomposition)",
    )
    ap.add_argument(
        "--no-bench", action="store_true", help="skip every bench-level section"
    )
    ap.add_argument(
        "--selection-budget-us",
        type=float,
        default=10.0,
        help="ML-on selection time above which a stage 7 problem counts as over budget",
    )
    ap.add_argument(
        "--target",
        type=float,
        default=None,
        help="absolute selection-efficiency target (default: validate.sel_eff_threshold "
        "of the run's config, else 0.95)",
    )
    ap.add_argument(
        "--diagnostics",
        action="store_true",
        help="add a collapsed section breaking wins and picks down by framework-specific "
        "kernel parameters (occupancy, vector widths, kernel attributes)",
    )
    ap.add_argument(
        "--shipped-subset",
        default=None,
        help="choose_shipped_subset.py output (default: <run_dir>/shipped_subset.json)",
    )
    ap.add_argument(
        "--origami-reference",
        action="store_true",
        help="add a collapsed external-reference section comparing with Origami on the "
        "same GEMMs",
    )
    a = ap.parse_args(argv)
    out = a.out or os.path.join(a.run_dir, "report.html")
    print(
        "wrote",
        build(
            a.run_dir,
            out,
            jobs=max(1, a.jobs),
            engine=not a.no_engine,
            bench=not a.no_bench,
            origami_reference=a.origami_reference,
            selection_budget_us=a.selection_budget_us,
            target=a.target,
            diagnostics=a.diagnostics,
            shipped_subset=a.shipped_subset,
        ),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
