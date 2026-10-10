#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Write a self-contained HTML report of a pipeline run.

Usage: make_report.py <run_dir> [out.html] [--jobs N] [--no-engine] [--no-bench]
                      [--target T] [--selection-budget-us US] [--diagnostics]
                      [--origami-reference] [--audit-results JSON ...]
                      [--audit-label NAME=LABEL ...] [--weak-spots-dir DIR]
                      [--pool-gap-fraction F] [--peak-tflops T]
                      [--peak-bandwidth-gbs B]

Reads the run's artifacts (round_*/stage04b/{decisions,splits}.json and its
gate/ and train/ slices, round_*/stage04 and stage03c/holdout/stage04
enriched chunks and kernels.json, round_*/stage02/plan.json and its seed
bench (stage01b/ in a run written before the planner became stage02:
lib/run_layout.py), round_*/stage05/{cells,metrics}.json and models.pt,
round_*/stage06/{manifest,deploy_record,gate}.json, round_*/*/audit.json
and the audit directory's progress, final_retrain/ (its record, decision,
audit records, stage05 and stage06) and audit_results/*.json,
round_*/stage01/shapes.yaml, validate/stage07/summary.json with its parity
and timing logs, validate/stage08/summary.json with each dataset's
per_gemm.csv, timing/*.json and the latest configs) and writes nothing but
the output file and, with --weak-spots-dir, the Weak spots section's data
(pool_gaps.json, pool_gaps.csv, weak_spots.json; lib/weak_spots.py). A
missing result says why, from the run's config and records. The page has
no external references; every table sorts by a click on a column header.

The headline is the run's position on held-out data. It describes the
bundle the run deployed (a round's, or its final retrain's while that is
one of its bundles: lib/run_bundles.py), else its latest staged bundle
(labelled staged, not deployed): its deploy gate and the next round's
stage04b measurement of the same models.pt, for the final retrain its
audit record (run_orchestrator.bundle_numbers), and first, once
an audit evaluation of the bundle has scored the exclusive group (complete
or partial, which it says), the audit's equal-weight mean over cells
against the true best, its minimum cell, its cells below the target and its
paired change from the previous candidate on the GEMMs both scored, under
protocol v2 when it is complete for the bundle (every audit number names its
protocol). The audit records the run wrote besides its rounds' stages
(audit_results/, final_retrain/) are read as --audit-results adds
scripts/audit_eval.py results: a more complete record of one of the run's
bundle files, on the run's audit set and with its pick rule, replaces the
run's own; every other record (a model
that is no round of the run, such as an offline retrain, or one scored on
another audit set) appears as an extra model next to the rounds, labelled
with its audit set and compared with the bundle when both score the same
GEMMs the same way; --audit-label renames an extra model. In-sample numbers
are secondary and labelled as such. The report's own re-ranking routes
GEMMs with the bundle's split tree and charges picks with the config's
validate.pick_measurement, as stages 4.5, 6, 7 and 8 do. Weak spots gives
every audit cell of the bundle the class of what limits it (serving rule,
kernel identity, model, kernel pool: the true best far below its roofline)
with its paired change from the previous candidate and its tail picks, and
lists the pool gaps. Every efficiency,
gap and ranking is against the fastest measured kernel and an absolute
target; Origami appears only in the opt-in --origami-reference section, and
only on the same GEMMs as the model.
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

rs.pipeline_on_path()
from lib import run_bundles  # noqa: E402
from lib import run_layout  # noqa: E402

FINAL_RETRAIN_DIR = run_bundles.FINAL_RETRAIN_DIR


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


def subject_bundle(run_dir):
    """The bundle the report describes: the one the run deployed
    (`shipped_bundle`), else its latest staged one (the latest completed
    stage06, `deployed: false`; run_orchestrator.bundle_numbers). `kind` is
    "deployed", "staged" or None (no deploy record); `bundle` its record,
    `gate` its stage06 gate.json and `reason` why it was not deployed."""
    sb = shipped_bundle(run_dir) or {}
    if sb.get("shipped"):
        return dict(sb, kind="deployed", bundle=sb["shipped"], reason=None)
    rs.pipeline_on_path()
    out = None
    try:
        from pathlib import Path

        import run_orchestrator as ro

        rec = ro.latest_deploy_record(ro.round_dirs(Path(run_dir)), deployed_only=False)
        if rec is not None:
            out = _finite(ro.bundle_numbers(Path(run_dir), rec))
    except Exception:  # noqa: BLE001
        out = None
    if not out or not out.get("bundle"):
        return dict(sb, kind=None, bundle=None, reason=None)
    rec_path = out["bundle"]["deploy_record"]
    gate = load_json(os.path.join(os.path.dirname(rec_path), "gate.json"))
    out.update(
        kind="staged",
        shipped=None,
        final_bundle=sb.get("final_bundle"),
        gate=_finite(gate) if isinstance(gate, dict) else None,
        reason=(load_json(rec_path) or {}).get("not_deployed"),
    )
    return out


def _log_add(sums, key, v, w):
    v = _num(v)
    if v is not None and v > 0 and w > 0:
        s = sums.setdefault(key, [0.0, 0])
        s[0] += math.log(v) * w
        s[1] += w


def pooled_cells(measurements):
    """Per cell, every measurement of one bundle on GEMMs it never trained
    on (its deploy gate, later stage04b) pooled: GEMM-weighted geomeans of
    the selection efficiency and of the winner and pick latencies, summed
    GEMMs and misses, the paired values over the GEMMs that have both, and
    the measurements that hold the cell (`sources`)."""
    acc = {}
    for m in measurements or []:
        for cell, c in (m.get("per_cell") or {}).items():
            if not isinstance(c, dict):
                continue
            se, n = _num(c.get("sel_eff_new")), int(c.get("n_eval") or 0)
            if se is None or se <= 0 or n <= 0:
                continue
            a = acc.setdefault(str(cell), {"n": 0, "miss": 0, "on": 0, "src": []})
            sums = a.setdefault("sums", {})
            a["n"] += n
            a["miss"] += int(c.get("n_miss") or 0)
            a["src"].append(f"{m.get('source')} {m.get('round')}")
            for key in ("sel_eff_new", "winner_us_geomean", "pick_us_geomean"):
                _log_add(sums, key, c.get(key), n)
            on = int(c.get("origami_n_eval") or 0)
            if _num(c.get("paired_sel_eff_new")) and _num(c.get("origami_sel_eff_new")):
                a["on"] += on
                for key in (
                    "paired_sel_eff_new",
                    "origami_sel_eff_new",
                    "origami_pick_us_geomean",
                ):
                    _log_add(sums, key, c.get(key), on)
    out = {}
    for cell, a in acc.items():
        geo = {k: math.exp(s[0] / s[1]) for k, s in a["sums"].items() if s[1] > 0}
        out[cell] = {
            "sel_eff_new": geo.get("sel_eff_new"),
            "n_eval": a["n"],
            "n_miss": a["miss"],
            "paired_sel_eff_new": geo.get("paired_sel_eff_new"),
            "origami_sel_eff_new": geo.get("origami_sel_eff_new"),
            "origami_n_eval": a["on"],
            "winner_us_geomean": geo.get("winner_us_geomean"),
            "pick_us_geomean": geo.get("pick_us_geomean"),
            "origami_pick_us_geomean": geo.get("origami_pick_us_geomean"),
            "sources": a["src"],
        }
    return out


def stage04b_history(run_dir):
    """Per cell label over every round's stage04b: how often it was
    retrained and its latest decision other than no_change (with the
    round)."""
    out = {}
    for d in round_dirs(run_dir):
        dec = load_json(os.path.join(d, "stage04b", "decisions.json"))
        if not isinstance(dec, dict):
            continue
        for c in dec.get("per_cell") or []:
            if not isinstance(c, dict) or not c.get("cell"):
                continue
            h = out.setdefault(
                str(c["cell"]), {"retrains": 0, "decision": None, "round": None}
            )
            decision = c.get("decision")
            h["retrains"] += decision == "retrain"
            if decision and decision != "no_change":
                h["decision"], h["round"] = str(decision), os.path.basename(d)
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


def context_names(cfg):
    """{run_analysis context label: the context names of `cfg`}: the run's
    expanded config (`run_config`), whose entries carry the CU budgets the
    run resolved its `cu_fraction`s to; {} when it has none to name."""
    rs.pipeline_on_path()
    try:
        from lib import features as fs
        from lib.contexts import contexts_from_config
        from lib.run_analysis import context_label

        ctxs = contexts_from_config(cfg if isinstance(cfg, dict) else {})
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for c in ctxs:
        key = context_label(fs.ExecutionContext(c.cu_budget, c.schedule))
        out.setdefault(key, []).append(c.label + (" held out" if c.holdout else ""))
    return {k: ", ".join(v) for k, v in out.items()}


def left_out_contexts(config_text, cfg):
    """(section key, index, entry) of the context entries of the run's
    source config (`config_text`) that its expanded config `cfg` left out:
    fractions whose CU budget was another entry's on the bench device; []
    when either is unreadable."""
    rs.pipeline_on_path()
    try:
        import yaml

        from lib.contexts import left_out

        source = yaml.safe_load(config_text or "")
        if not (isinstance(source, dict) and isinstance(cfg, dict)):
            return []
        return left_out(source, cfg)
    except Exception:  # noqa: BLE001
        return []


def no_holdout_left(gone, cfg):
    """True when the entries `left_out_contexts` found are every held-out
    context of the config."""
    held = ((cfg or {}).get("contexts") or {}).get("holdout")
    return not held and any(key == "holdout" for key, _, _ in gone)


def left_out_html(gone, cfg):
    """The contexts section's note on the entries `left_out_contexts`
    found; empty without any."""
    if not gone:
        return ""
    names = ", ".join(
        f"contexts.{key}[{i}] (cu_fraction {float(m['cu_fraction']):g})"
        for key, i, m in gone
    )
    text = (
        f"<b>Left out on this run's bench device:</b> {esc(names)}. The CU "
        "budget each resolves to (a multiple of 32 below the device's CUs) is "
        "another context's, so the run did not bench it."
    )
    if no_holdout_left(gone, cfg):
        text += (
            " <b>No held-out context remains: the run measured the models on "
            "none.</b>"
        )
    return f'<p class="cap">{text}</p>'


def training_rounds(run_dir):
    """Per round: stage05 selection efficiency on the cells it trained, and
    the cells it was to train but did not, with which the round went on
    (`carried`: train.on_cell_failure carry, stage05/cells_carried.json)."""
    rows = []
    for d in round_dirs(run_dir):
        met = load_json(os.path.join(d, "stage05", "metrics.json"))
        if not isinstance(met, dict) or _num(met.get("global_sel_eff")) is None:
            continue
        carried = load_json(os.path.join(d, "stage05", "cells_carried.json"))
        rows.append(
            {
                "round": os.path.basename(d),
                "model": _pct(met.get("global_sel_eff")),
                "deployed": _pct(met.get("global_deployed_sel_eff")),
                "n_cells": met.get("n_cells_trained"),
                "n_gemms": met.get("n_gemms_trained"),
                "carried": [
                    c for c in (carried or {}).get("cells") or [] if isinstance(c, dict)
                ],
            }
        )
    return rows


def carried_notes(train):
    """A note per round that went on with cells stage05 did not train."""
    return [
        f"{r['round']} went on without training {len(r['carried'])} cell(s) "
        "(train.on_cell_failure: carry): "
        + "; ".join(
            f"{c.get('cell')} ({c.get('why')}; keeps {c.get('model')})"
            for c in r["carried"]
        )
        for r in train
        if r["carried"]
    ]


def held_out_breakdown(run_dir):
    """Latest stage04b held-out measurement per cell over every round (each
    by the model of the round before it), without the cells a descendant of
    which was measured (a cell split later keeps its measurement until its
    children have one)."""
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
                "sources": [f"stage04b {os.path.basename(d)}"],
            }
    split = {c for c in per_cell if any(o.startswith(c + "#") for o in per_cell)}
    return {c: v for c, v in per_cell.items() if c not in split}


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


def bundles_upto(run_dir, upto=None):
    """The run's bundle directories (its rounds and its final retrain while
    that is one of its bundles: lib/run_bundles.py) in the order the run
    deploys them, up to the bundle `upto` names (a round, or the final
    retrain; default all)."""
    from pathlib import Path

    bundles = run_bundles.bundles(run_dir, [Path(d) for d in round_dirs(run_dir)])
    final = Path(run_dir) / FINAL_RETRAIN_DIR
    if upto == FINAL_RETRAIN_DIR and final not in bundles:
        bundles = sorted(bundles + [final], key=run_bundles.bundle_key)
    names = [b.name for b in bundles]
    return bundles[: names.index(upto) + 1] if upto in names else bundles


def final_cells(run_dir, held, upto=None):
    """Leaf cells of the shipped bundle: per label the cells.json entry of
    the bundle whose stage05 trained its model, over the bundles up to
    `upto` (`bundles_upto`; a bundle's stage05/cell_origins.json gives the
    entry of each cell it carried, also once the bundle that trained it has
    left the run: `run_bundles.cell_origins`), with held-out values
    (inherited from the nearest measured ancestor for leaves created by the
    last split)."""
    by_label = {
        label: o["entry"]
        for label, o in run_bundles.cell_origins(bundles_upto(run_dir, upto)).items()
        if o.get("entry")
    }
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
                "smartK": len(c.get("smart_k_signatures") or [])
                or c.get("n_view_key_hashes")
                or c.get("n_whitelist_tier_keys")
                or None,
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
    """Per bench yaml of the latest stage07 run: whether no round trained on
    its problems (`held_out`), pick parity, the runtime's selection
    efficiency (its top-1 picks against the bench timings of the same
    problems, with the bootstrap interval) and, as the Origami reference, the
    runtime's picks with tilewright on and off paired on the same problems
    (`origami`), and per request size the ML-on selection times overall, on
    the first call into a cell and warm (medians with bootstrap intervals),
    plus the ML-off reference."""
    summ = load_json(os.path.join(run_dir, "validate", "stage07", "summary.json"))
    if not isinstance(summ, dict):
        return None
    out = {
        "ok": bool(summ.get("ok")),
        "failures": len(summ.get("failures") or []),
        "bank_served": summ.get("bank_served"),
        "origami_reference": summ.get("origami_reference"),
        "yamls": [],
    }
    for tag, y in sorted((summ.get("per_yaml") or {}).items()):
        if not isinstance(y, dict):
            continue
        row = {
            "yaml": tag,
            "held_out": bool(y.get("held_out")),
            "n_kept": y.get("n_kept"),
            "n_entries": y.get("n_entries"),
        }
        ref = y.get("origami_reference")
        paired = (ref or {}).get("paired") if isinstance(ref, dict) else None
        if isinstance(paired, dict):
            row["origami"] = {
                "n_problems": ref.get("n_problems"),
                "n": paired.get("n"),
                "model": _pct(paired.get("model_sel_eff")),
                "origami": _pct(paired.get("origami_sel_eff")),
                "wins": paired.get("wins"),
                "ties": paired.get("ties"),
                "losses": paired.get("losses"),
                "without": {
                    k: ref.get(k)
                    for k in (
                        "n_without_timings",
                        "n_without_origami_pick",
                        "n_origami_without_row",
                    )
                    if ref.get(k)
                },
            }
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
                    "on_min": on.get("min"),
                    "on_mean": on.get("mean"),
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


def deployed_size_mb(run_dir, upto=None):
    """Size of the model file of the latest stage06 manifest of the rounds
    up to `upto` (a round name; default all), or of the final retrain's
    when `upto` names it."""
    dirs = round_dirs(run_dir)
    names = [os.path.basename(d) for d in dirs]
    if upto in names:
        dirs = dirs[: names.index(upto) + 1]
    elif upto == FINAL_RETRAIN_DIR:
        dirs = [os.path.join(run_dir, FINAL_RETRAIN_DIR)]
    for d in reversed(dirs):
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


def run_config(run_dir, source_text=""):
    """The latest config the run expanded (configs/*.yaml without
    .source.yaml), else `source_text` parsed; {} when unreadable."""
    import yaml

    done = [
        p
        for p in sorted(glob.glob(os.path.join(run_dir, "configs", "*.yaml")))
        if not p.endswith(".source.yaml")
    ]
    for text in ([read_text(done[-1])] if done else []) + [source_text]:
        try:
            cfg = yaml.safe_load(text or "")
        except yaml.YAMLError:
            continue
        if isinstance(cfg, dict):
            return cfg
    return {}


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


_MAP_TIERS = ("Tiny", "Small", "Mid", "Large")
_MAP_K_TIERS = ("TinyK", "MidK", "LargeK")
_MAP_SPLIT = re.compile(r"([MNKB])(<=|>)(\d+)$")


def _map_bins(target_pct):
    """(lower edge in %, fill, legend text) of the cell map's colors."""
    t = target_pct
    return (
        (t, "#2da44e", f"&ge; {t:g}%"),
        (t - 2, "#9be9a8", f"{t - 2:g}&ndash;{t:g}%"),
        (t - 5, "#f2cc60", f"{t - 5:g}&ndash;{t - 2:g}%"),
        (t - 10, "#fb8f44", f"{t - 10:g}&ndash;{t - 5:g}%"),
        (-math.inf, "#cf222e", f"&lt; {t - 10:g}%"),
    )


def _map_fill(v, bins):
    if v is None:
        return "#eaeef2"
    return next(color for edge, color, _ in bins if v >= edge)


def _log_frac(v, lo, hi):
    lo, hi = max(lo, 1), max(hi, 1)
    if hi <= lo:
        return 0.0
    f = (math.log(max(v, 1)) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return min(1.0, max(0.0, f))


def cell_map_html(cells, target, bundle="bundle"):
    """The leaf cells (`final_cells` rows) in M (x) and N (y) as inline SVG,
    one panel per K tier and batch tier. Every tier is one band and a split
    sits at its value on a log scale within the tier; a split on K or batch
    divides the cell's M / N area with a dashed line. A leaf is filled by
    its held-out selection efficiency (`held`, %; * = inherited from an
    ancestor), grey without one."""
    _pipeline_on_path()
    from lib.grid import BANY_VALUES, TIER_RANGES_K, TIER_RANGES_MN

    S, ML, MT, GAP = 46, 34, 30, 40
    PW, PH = 4 * S, 4 * S
    W, H = ML + 3 * PW + 2 * GAP + 6, 2 * (PH + MT + 18) + 8
    tgt = target * 100
    bins = _map_bins(tgt)
    rects, skipped = [], []
    n_val = n_inh = n_below = 0
    for c in cells:
        label = c.get("cell") or ""
        base, *segs = label.split("#")
        parts = base.split("|")
        if (
            len(parts) < 4
            or parts[0] not in TIER_RANGES_MN
            or parts[1] not in TIER_RANGES_MN
            or parts[2] not in TIER_RANGES_K
            or parts[3] not in ("Bnone", "Bany")
        ):
            skipped.append(label)
            continue
        mt, nt, kt, bt = parts[:4]
        full = {
            "M": TIER_RANGES_MN[mt],
            "N": TIER_RANGES_MN[nt],
            "K": TIER_RANGES_K[kt],
            "B": (min(BANY_VALUES), max(BANY_VALUES)) if bt == "Bany" else (1, 1),
        }
        r = {ax: list(v) for ax, v in full.items()}
        for seg in segs:
            g = _MAP_SPLIT.match(seg)
            if g:
                ax, v = g.group(1), int(g.group(3))
                lo, hi = r[ax]
                r[ax] = [lo, min(hi, v)] if g.group(2) == "<=" else [max(lo, v + 1), hi]
        fr = {
            ax: (_log_frac(r[ax][0], *full[ax]), _log_frac(r[ax][1], *full[ax]))
            for ax in r
        }
        px = ML + _MAP_K_TIERS.index(kt) * (PW + GAP)
        py = MT + (0 if bt == "Bnone" else 1) * (PH + MT + 18)
        bottom = py + (4 - _MAP_TIERS.index(nt)) * S
        x0 = px + (_MAP_TIERS.index(mt) + fr["M"][0]) * S
        x1 = px + (_MAP_TIERS.index(mt) + fr["M"][1]) * S
        y0, y1 = bottom - fr["N"][1] * S, bottom - fr["N"][0] * S
        dashed = False
        if fr["K"] != (0.0, 1.0):
            x0, x1 = x0 + (x1 - x0) * fr["K"][0], x0 + (x1 - x0) * fr["K"][1]
            dashed = True
        if bt == "Bany" and fr["B"] != (0.0, 1.0):
            y0, y1 = y1 - (y1 - y0) * fr["B"][1], y1 - (y1 - y0) * fr["B"][0]
            dashed = True
        v = _num(c.get("held"))
        n_val += v is not None
        n_inh += v is not None and bool(c.get("held_inherited"))
        n_below += v is not None and v < tgt
        mark = "*" if c.get("held_inherited") else ""
        tip = f"{label}: " + (
            f"held-out {v:.2f}%{' (inherited from an ancestor)' if mark else ''}"
            if v is not None
            else "no held-out measurement"
        )
        tip += f", {int(c.get('n_gemms') or 0):,} training GEMMs"
        w, h = max(x1 - x0, 0.5), max(y1 - y0, 0.5)
        rects.append(
            f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'fill="{_map_fill(v, bins)}" stroke="#fff" stroke-width="0.8"'
            + (' stroke-dasharray="3 2"' if dashed else "")
            + f"><title>{esc(tip)}</title></rect>"
            + (
                f'<text x="{x0 + w / 2:.1f}" y="{y0 + h / 2 + 3:.1f}" font-size="8" '
                f'text-anchor="middle" fill="#1f2328" pointer-events="none">'
                f"{v:.1f}{mark}</text>"
                if v is not None and w >= 26 and h >= 11
                else ""
            )
        )
    if not rects:
        return '<p class="mut">no leaf cells of a trained bundle in this run.</p>'
    deco = []
    for bi, bt in enumerate(("Bnone", "Bany")):
        for ki, kt in enumerate(_MAP_K_TIERS):
            px = ML + ki * (PW + GAP)
            py = MT + bi * (PH + MT + 18)
            deco.append(
                f'<text x="{px}" y="{py - 8}" font-size="10" fill="{_CHART_TEXT}">'
                f"K {esc(kt)}, {'unbatched (Bnone)' if bt == 'Bnone' else 'batched (Bany)'}"
                "</text>"
            )
            for t in range(4):
                deco.append(
                    f'<text x="{px + (t + 0.5) * S:.1f}" y="{py + PH + 11}" font-size="8" '
                    f'text-anchor="middle" fill="{_CHART_MUT}">M {_MAP_TIERS[t]}</text>'
                    f'<text x="{px - 3}" y="{py + (3.5 - t) * S + 3:.1f}" font-size="8" '
                    f'text-anchor="end" fill="{_CHART_MUT}">N {_MAP_TIERS[t]}</text>'
                )
                for u in range(4):
                    deco.append(
                        f'<rect x="{px + t * S}" y="{py + u * S}" width="{S}" height="{S}" '
                        'fill="none" stroke="#57606a" stroke-width="0.6"/>'
                    )
    legend = " ".join(
        f'<span style="display:inline-block;width:10px;height:10px;background:{color}">'
        f"</span> {text}"
        for _, color, text in bins
    ) + (
        ' <span style="display:inline-block;width:10px;height:10px;background:#eaeef2">'
        "</span> no held-out measurement"
    )
    svg = (
        f'<svg viewBox="0 0 {W} {H}" role="img" xmlns="http://www.w3.org/2000/svg" '
        f'aria-label="leaf cells in M and N">{"".join(rects)}{"".join(deco)}</svg>'
    )
    return (
        f'<p class="cap">{len(rects)} leaf cells of the {esc(bundle)}, {n_val} with a '
        f"held-out value ({n_inh} inherited from an ancestor, marked *), {n_below} below "
        f"the {tgt:g}% target. x = M tier, y = N tier (Tiny &le;32, Small &le;128, Mid "
        "&le;512, Large &gt;512), one panel per K tier and batch tier; a split sits at "
        "its value on a log scale within the tier, and dashed lines divide a cell on K "
        "or batch. Hover a cell for its label and value."
        + (f" Not on the grid: {esc(', '.join(skipped))}." if skipped else "")
        + f'</p><p class="cap">{legend}</p>'
        f'<figure class="chart" style="max-width:{W * 1.25:.0f}px">{svg}</figure>'
    )


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


def _pc(v):
    """A fraction as a percentage; None stays None."""
    return v * 100 if v is not None else None


def _f(v, digits=2, signed=False):
    v = _num(v)
    if v is None:
        return "&ndash;"
    return f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"


def _ci_txt(ci, scale=1.0, digits=2):
    if not ci:
        return ""
    return f" [{ci[0] * scale:.{digits}f}, {ci[1] * scale:.{digits}f}]"


def _why(reason):
    return f": {esc(reason)}" if reason else ""


def stage07_html(s7, origami_reference=False, reason=None):
    if not s7:
        return f'<p class="mut">no stage 7 results in this run{_why(reason)}.</p>'
    parts = []
    status = "all checks passed" if s7["ok"] else f"{s7['failures']} check(s) failed"
    bank = {True: "served", False: "not served"}.get(s7.get("bank_served"))
    parts.append(
        f'<p class="cap">stage 7 status: <b>{esc(status)}</b>'
        + (
            f"; the bank of kernels was {bank} (TENSILE_TILEWRIGHT_BANK="
            f"{1 if s7['bank_served'] else 0} on its ML-on runs)"
            if bank
            else ""
        )
        + ".</p>"
    )
    for y in s7["yamls"]:
        par = y.get("parity")
        sel = y.get("runtime_selection")
        ori = y.get("origami") if origami_reference else None
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
        if ori:
            without = ", ".join(f"{esc(k)}={esc(v)}" for k, v in ori["without"].items())
            rows.append(
                f'<tr><td class="c">[reference] Origami\'s pick</td>'
                f'<td colspan="6" style="text-align:left">the runtime with tilewright '
                f"off on the same {esc(ori.get('n'))} of {esc(ori.get('n_problems'))} "
                f"problems: model <b>{_f(ori.get('model'))}%</b>, Origami "
                f"{_f(ori.get('origami'))}% ({esc(ori.get('wins'))} / "
                f"{esc(ori.get('ties'))} / {esc(ori.get('losses'))} wins / ties / "
                f"losses){('; not paired: ' + without) if without else ''}</td></tr>"
            )
        for t in y.get("timing") or []:
            rows.append(
                f'<tr><td class="c">selection time, rsn={esc(t["rsn"])}</td>'
                f'<td>{esc(t["n"])} problems, {esc(t["reps"])} reps</td>'
                f'<td><b>{_f(t["on_p50"])}</b>{esc(_ci_txt(t.get("on_p50_ci")))}</td>'
                f'<td>{_f(t.get("on_min"))} / {_f(t.get("on_mean"))} / '
                f'{_f(t["on_p99"])} / {_f(t["on_max"])}</td>'
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
            f'{esc(y.get("n_entries"))} problems match the config'
            + ("; held out of training" if y.get("held_out") else "")
            + ")</span></h3>"
            '<table style="max-width:1400px"><thead><tr><th class="c">check</th>'
            "<th>problems</th><th>ML-on p50 &micro;s [95% CI]</th>"
            "<th>ML-on min / mean / p99 / max</th>"
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


def stage08_html(rows, target, reason=None):
    """Validation datasets against the fastest measured kernel: the stage's
    own number (model picks and the runtime's fallback, as deployed) and the
    model's own picks per leaf."""
    if not rows:
        return f'<p class="mut">no stage 8 results in this run{_why(reason)}.</p>'
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


def _cell_gate_of(stage06_dir, rec):
    """The per-cell gate of a stage06: its cell_gate.json, else the deploy
    record's summary; None without one."""
    full = load_json(os.path.join(stage06_dir, "cell_gate.json"))
    if isinstance(full, dict):
        return full
    cells = rec.get("cell_gate") if isinstance(rec, dict) else None
    return cells if isinstance(cells, dict) else None


def cell_gate_findings(bundle, cells):
    """One row per finding of a per-cell gate (`lib.audit.cell_gate`) in
    every group it compared: each cell significantly worse beyond the
    tolerance or confidently fallen below the floor (with its family-wise
    p), each pick that collapsed below the tail floor and each one lifted
    back above it; `deciding` when it keeps the bundle out (a cell of a
    group the cell rules judge; a confirmed collapse of a group whose new
    collapses outnumber its lifted ones beyond the budget)."""
    out = []
    for label, g in ((cells or {}).get("groups") or {}).items():
        if not isinstance(g, dict) or "reason" in g:
            continue
        for kind in ("regressed", "floor", "tail", "tail_lifted"):
            for f in g.get(kind) or []:
                shape = f.get("shape")
                if kind == "tail":
                    deciding = bool(f.get("decides", g.get("deciding", True)))
                else:
                    deciding = kind != "tail_lifted" and bool(g.get("deciding", True))
                before, after = _num(f.get("before")), _num(f.get("after"))
                ratio = _num(f.get("ratio"))
                if ratio is None and shape and before and after is not None:
                    ratio = after / before
                out.append(
                    {
                        "bundle": bundle,
                        "group": label,
                        "deciding": deciding,
                        "protocol": g.get("protocol"),
                        "kind": kind,
                        "cell": f.get("cell"),
                        "gemm": (
                            "x".join(str(v) for v in shape[:3])
                            + (f" batch {shape[3]}" if shape[3] != 1 else "")
                            if shape
                            else None
                        ),
                        "before": before,
                        "after": after,
                        "ratio": ratio,
                        "ratio_ci": f.get("ratio_ci"),
                        "p": _num(f.get("p")),
                        "confirmed": f.get("confirmed"),
                    }
                )
    return out


def _cell_gate_text(cells):
    if not isinstance(cells, dict):
        return "not run"
    verdict = cells.get("passed")
    if verdict is None:
        rules = cells.get("cell_rules")
        if rules and rules != "on":
            return "not decided (its cell rules were off)"
        return "not measured"
    if verdict:
        return "passed"
    if cells.get("record_only"):
        return "would keep it out (recorded only)"
    return "kept it out"


def deploy_gates(run_dir):
    """Per round with a stage06 deploy record: its gate's result, its
    per-cell gate against the bundle the run shipped (cell_gate.json) and
    whether the bundle was deployed; then the final retrain's decision on
    the audit set (final_retrain/decision.json), which is its gate, with its
    per-cell gate, and that of a new final retrain that has not replaced it
    (final_retrain_next/)."""
    rows = []
    for d in round_dirs(run_dir):
        rec = load_json(os.path.join(d, "stage06", "deploy_record.json"))
        if not isinstance(rec, dict):
            continue
        gate = load_json(os.path.join(d, "stage06", "gate.json"))
        gate = gate if isinstance(gate, dict) else rec.get("gate") or {}
        cells = _cell_gate_of(os.path.join(d, "stage06"), rec)
        rows.append(
            {
                "round": os.path.basename(d),
                "sel_eff": _num(gate.get("sel_eff")),
                "n": gate.get("n_evaluated"),
                "n_miss": gate.get("n_miss"),
                "pick_measurement": gate.get("pick_measurement"),
                "target": _num(gate.get("target")),
                "deployed": rec.get("deployed", True) is not False,
                "reason": rec.get("not_deployed") or gate.get("reason"),
                "cell_gate": _cell_gate_text(cells),
                "cell_against": ((cells or {}).get("baseline") or {}).get("name"),
                "findings": cell_gate_findings(os.path.basename(d), cells),
            }
        )
    for name in (FINAL_RETRAIN_DIR, run_bundles.FINAL_RETRAIN_NEXT_DIR):
        final = os.path.join(run_dir, name)
        dec = load_json(os.path.join(final, run_bundles.DECISION))
        if not isinstance(dec, dict):
            continue
        staged = os.path.isfile(
            os.path.join(final, "stage06", run_bundles.STAGE_MARKER)
        )
        cells = dec.get("cell_gate") if isinstance(dec.get("cell_gate"), dict) else None
        rows.append(
            {
                "round": name,
                "sel_eff": None,
                "n": None,
                "n_miss": None,
                "pick_measurement": None,
                "target": None,
                "deployed": (
                    bool(dec.get("accepted")) and staged and name == FINAL_RETRAIN_DIR
                ),
                "not_staged": not staged,
                "reason": f"decided on the audit set: {dec.get('reason')}",
                "cell_gate": _cell_gate_text(cells),
                "cell_against": (
                    (dec.get("baseline") or {}).get("round") if cells else None
                ),
                "findings": cell_gate_findings(name, cells),
            }
        )
    return rows


def deploy_gates_html(rows):
    if not rows:
        return ""

    def bundle(r):
        if r["deployed"]:
            return "deployed"
        return "not deployed" if r.get("not_staged") else "staged, not deployed"

    def cells(r):
        txt = r.get("cell_gate") or "not run"
        if r.get("cell_against") and txt != "not run":
            txt += f" (against {r['cell_against']})"
        return txt

    trs = "".join(
        f'<tr class="{"win" if r["deployed"] else ""}"><td class="c">{esc(r["round"])}</td>'
        f'<td>{_f(r["sel_eff"] * 100 if r["sel_eff"] is not None else None)}</td>'
        f'<td>{intf_(r["n"])}</td><td>{intf_(r["n_miss"])}</td>'
        f'<td class="c">{esc(r["pick_measurement"] or "")}</td>'
        f'<td class="c">{bundle(r)}</td>'
        f'<td class="c">{esc(cells(r))}</td>'
        f'<td class="c">{esc(r["reason"] or "")}</td></tr>'
        for r in rows
    )
    findings = [f for r in rows for f in r.get("findings") or []]

    def measured(f):
        if f.get("confirmed") is None:
            return ""
        return "yes" if f["confirmed"] else "no"

    def pval(f):
        return "n/a" if f.get("p") is None else f"{f['p']:.2g}"

    ftrs = "".join(
        f'<tr class="{"lose" if f["deciding"] else ""}"><td class="c">{esc(f["bundle"])}</td>'
        f'<td class="c">{esc(f["group"])}</td>'
        f'<td class="c">{"yes" if f["deciding"] else "no"}</td>'
        f'<td class="c">{esc(CELL_FINDING_KINDS.get(f["kind"], f["kind"]))}</td>'
        f'<td class="c">{esc(f["cell"] or "")}</td>'
        f'<td class="c">{esc(f["gemm"] or "")}</td>'
        f'<td>{_f(_pc(f["before"]))}</td><td>{_f(_pc(f["after"]))}</td>'
        f'<td>{rs._ratio_cell(f["ratio"], f["ratio_ci"]) if f["ratio"] else "n/a"}</td>'
        f"<td>{pval(f)}</td>"
        f'<td class="c">{measured(f)}</td>'
        f'<td class="c">{esc(f["protocol"] or "")}</td></tr>'
        for f in findings
    )
    return (
        "<h3>Deploy gate per round (stage06)</h3>"
        '<table style="max-width:1100px"><thead><tr><th class="c">round</th>'
        "<th>gate sel_eff %</th><th>gate GEMMs</th><th>misses</th>"
        '<th class="c">pick measurement</th><th class="c">bundle</th>'
        '<th class="c">per-cell gate</th>'
        f'<th class="c">reason</th></tr></thead><tbody>{trs}</tbody></table>'
        + (
            "<h3>Per-cell gate findings</h3>"
            '<p class="cap">What the per-cell gate found against the bundle the run '
            "shipped, on the audit set (<code>lib.audit.cell_gate</code>, the config's "
            "<code>gate</code> section). Cell rules, on the groups "
            "<code>gate.groups</code> names: a cell significantly worse beyond the "
            "tolerance or confidently fallen below the floor, family-wise over their "
            "cells (<b>p</b>: GEMM-wise sign flips, Westfall-Young step-down; the "
            "chance of any such finding between two equally good bundles is at most "
            "<code>gate.alpha</code>); the other groups' cells are tested the same way "
            "and reported. Tail rule, in every group: picks that collapsed below the "
            "tail floor and picks lifted back above it (<b>measured</b>: a repeated "
            "measurement confirms it); a group keeps the bundle out when its confirmed "
            "collapses outnumber its lifted picks by more than "
            "<code>gate.tail_budget</code>. This rule is a policy on measured GEMMs, "
            "not a test: one collapse can keep out a bundle no worse overall. Rows that "
            "decide (highlighted) kept the bundle out unless the gate only "
            "recorded.</p>"
            '<div class="scroll"><table style="margin-bottom:0;font-size:11px"><thead><tr>'
            '<th class="c">bundle</th><th class="c">group</th><th class="c">decides</th>'
            '<th class="c">finding</th><th class="c">cell</th><th class="c">GEMM</th>'
            "<th>shipped %</th><th>this bundle %</th><th>ratio [95% CI]</th>"
            '<th>p</th><th class="c">measured</th>'
            f'<th class="c">protocol</th></tr></thead><tbody>{ftrs}</tbody></table></div>'
            if findings
            else ""
        )
    )


CELL_FINDING_KINDS = {
    "regressed": "significantly worse beyond the tolerance",
    "floor": "confidently fell below the floor",
    "tail": "pick collapsed below the tail floor",
    "tail_lifted": "pick lifted above the tail floor",
}


def bundle_html(sb, target, gates=(), next_reason=None):
    """The bundle the report describes (`subject_bundle`): which bundle,
    deployed or only staged, its measurements on GEMMs it never trained on
    (`next_reason`: why no later stage04b measured it), the deploy gate's
    comparison with the bundle it replaced on the same GEMMs, a later
    bundle the gate kept out, and every round's gate."""
    sb = sb or {}
    sh = sb.get("bundle")
    if not sh:
        return (
            '<p class="mut">this run deployed no bundle and staged none: no round has '
            "completed stage06 (the deploy gate) yet.</p>" + deploy_gates_html(gates)
        )
    staged = sb.get("kind") == "staged"
    td = str(sh.get("train_dir") or "")
    ident = (
        f"<b>{esc(sh.get('round'))}</b> (models "
        f"{esc(os.path.basename(os.path.dirname(td)))}/{esc(os.path.basename(td))}, "
        f"model file sha256 {esc(str(sh.get('model_file_sha256') or '')[:12])})"
    )
    parts = [
        (
            '<p class="cap">No bundle of this run passed the deploy gate, so none was '
            f"deployed; this report describes the latest <b>staged</b> bundle, {ident}, "
            "staged for deploy in its stage06 and not deployed: "
            f'{esc(sb.get("reason") or "the gate kept it out")}. '
            if staged
            else f'<p class="cap">Deployed bundle: {ident}. '
        )
        + f'Headline source: <b>{esc(sb.get("source") or "none")}</b>'
        + (
            " (no measurement on GEMMs it did not train on)"
            if not sb.get("source")
            else ""
        )
        + ".</p>"
    ]
    retrain = sb.get("final_retrain") or {}
    if not staged and round_num(str(sh.get("round") or "")) < 0:
        parts.append(
            '<p class="cap">The final retrain: every cell trained again on the data '
            f"of {esc(', '.join(retrain.get('trained_on') or []) or 'its rounds')} "
            "(final_retrain/final_retrain.json), deployed after them by its gate on "
            f"the audit set: {esc(retrain.get('reason'))} "
            "(final_retrain/decision.json).</p>"
        )
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
            f'<td class="c">{esc(m.get("pick_measurement") or "")}</td>'
            f'<td><b>{_f(m["sel_eff"] * 100 if m.get("sel_eff") is not None else None)}</b></td>'
            f'<td>{_f((m["sel_eff"] - target) * 100 if m.get("sel_eff") is not None else None, signed=True)}</td></tr>'
            for i, m in enumerate(ms)
        )
        audited = (
            "<b>audit</b> = the final retrain's audit record (its exclusive group, per "
            "base cell). "
            if any(m.get("source") == "audit" for m in ms)
            else ""
        )
        parts.append(
            '<p class="cap">Every measurement of this bundle on GEMMs it never trained on: '
            "<b>deploy_gate</b> = stage06's gate slice of the bundle's round, "
            "<b>stage04b</b> = a later round's new GEMMs evaluated with the same models.pt"
            + ("; " if audited else ". ")
            + audited
            + "Picks are charged under the pick measurement of the stage; the runtime's fallback "
            "counts where the model scores nothing that runs.</p>"
            '<table style="max-width:1000px"><thead><tr><th class="c">source</th>'
            '<th class="c">round</th><th>GEMMs</th><th>misses</th><th>cells</th>'
            '<th class="c">pick measurement</th>'
            "<th>sel_eff %</th><th>vs target pp</th></tr></thead><tbody>"
            f"{trs}</tbody></table>"
        )
    if not any(m.get("source") == "stage04b" for m in ms) and next_reason:
        parts.append(
            f'<p class="mut">No later round\'s stage04b has measured this bundle: '
            f"{esc(next_reason)}.</p>"
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
    if final and not staged:
        parts.append(
            f'<p class="cap"><b>Not deployed:</b> {esc(final.get("round"))} staged a later '
            f'bundle that its gate kept out ({esc(final.get("reason"))}; gate '
            f'{_f((final.get("gate_sel_eff") or 0) * 100 if final.get("gate_sel_eff") is not None else None)}%). '
            "The headline, the leaf-cell table and the bench analysis are of the deployed "
            "bundle above.</p>"
        )
    return "".join(parts) + deploy_gates_html(gates)


def intf_(v):
    return (
        f"{int(v):,}"
        if isinstance(v, (int, float)) and not isinstance(v, bool)
        else "&ndash;"
    )


# Columns of the wall-clock table, in pipeline order; any other stage key a
# timing record holds follows them.
TIMING_STAGES = (
    ("audit set", "audit set build"),
    ("stage01", "stage01"),
    ("stage02", "stage02"),
    ("stage03", "stage03"),
    ("stage03b", "stage03b"),
    ("stage03c", "stage03c"),
    ("stage03d", "stage03d"),
    ("stage04", "stage04"),
    ("stage04b", "stage04b"),
    ("stage05", "stage05"),
    ("audit_v2", "audit v2"),
    ("stage06", "stage06"),
    ("final stage05", "final retrain stage05"),
    ("audit_eval", "audit eval"),
    ("final stage06", "final retrain stage06"),
    ("stage07", "stage07"),
    ("stage08", "stage08"),
)


def timing_stages(legacy=False):
    """TIMING_STAGES, or with `legacy` those of a run written before the
    planner became stage02 (lib/run_layout.py), which times it as stage01b
    and whose stage02 rows are the removed probe's."""
    if not legacy:
        return TIMING_STAGES
    i = TIMING_STAGES.index(("stage02", "stage02"))
    return (
        TIMING_STAGES[:i]
        + (("stage01b", "stage01b"), ("stage02", "stage02 (removed probe)"))
        + TIMING_STAGES[i + 1 :]
    )


def timing_html(rounds, total, legacy=False):
    if not rounds:
        return '<p class="mut">no timing records.</p>'
    stages = timing_stages(legacy)
    names = dict(stages)
    seen = {s for st in rounds.values() for s in st if s != "(round total)"}
    present = [s for s, _ in stages if s in seen] + sorted(seen - set(names))
    head = "".join(f"<th>{esc(names.get(s, s))}</th>" for s in present)
    body = []
    for rnd in sorted(rounds, key=lambda r: (r is None, r if r is not None else 0)):
        st = rounds[rnd]
        name = "run" if rnd is None else f"round_{rnd}"
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
        f'<tr class="total" style="font-weight:600"><td class="c">all invocations</td>'
        f'{"<td></td>" * len(present)}<td>{esc(fmt_dur(total))}</td></tr>'
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


def five_point_html(agg, s8, s7, why=None):
    """The run in the five-point form: the model and Origami on the same
    held-out GEMMs of the training distribution and on each stage 8 dataset,
    and the ML-on selection time inside hipBLASLt (stage 7) as min / mean /
    p99 / max. The held-out GEMMs of the training distribution are the
    rounds' new GEMMs the bundle never trained on (paired where their
    benches timed Origami's pick) and the audit set's GEMMs stage 7 timed
    with the runtime's picks with tilewright on and off; `why` says why
    neither pairs them."""

    def eff(name, n, model, origami):
        d = model - origami if model is not None and origami is not None else None
        return (
            f'<tr><td class="c">{name}</td><td>{intf_(n)}</td><td>{_f(model)}</td>'
            f"<td>{_f(origami)}</td><td>{_f(d, signed=True)}</td></tr>"
        )

    what = "held-out GEMMs of the training distribution"
    rows = []
    if agg.get("paired_n"):
        rows.append(
            eff(
                f"{what}: the rounds' new GEMMs",
                agg.get("paired_n"),
                agg.get("paired_model"),
                agg.get("paired_origami"),
            )
        )
    for y in (s7 or {}).get("yamls", []):
        o = y.get("origami") or {}
        if y.get("held_out") and o.get("n"):
            rows.append(
                eff(
                    f"{what}: the audit set's {esc(y['yaml'])} (stage 7, "
                    "hipBLASLt's picks)",
                    o["n"],
                    o.get("model"),
                    o.get("origami"),
                )
            )
    unpaired = not rows
    if unpaired:
        rows.append(eff(what, None, None, None))
    rows += [
        eff(f"stage 8 {esc(r['name'])}", r["paired_n"], r["model"], r["origami"])
        for r in s8 or []
    ]
    times = []
    for y in (s7 or {}).get("yamls", []):
        for t in y.get("timing") or []:
            times.append(
                f'<tr><td class="c">{esc(y["yaml"])}</td><td>{esc(t["rsn"])}</td>'
                f'<td>{esc(t["n"])}</td><td>{_f(t.get("on_min"))}</td>'
                f'<td>{_f(t.get("on_mean"))}</td><td>{_f(t.get("on_p99"))}</td>'
                f'<td>{_f(t.get("on_max"))}</td><td>{_f(t.get("off_p50"))}</td></tr>'
            )
    return (
        "<h3>Five-point summary</h3>"
        '<p class="cap">Selection efficiency of the model and of Origami on the same '
        "GEMMs (geomean of fastest measured / pick, %), on held-out GEMMs of the "
        "training distribution and on each stage 8 dataset; selection time of the "
        "deployed model inside hipBLASLt (stage 7, ML-on, &micro;s per problem), with "
        "Origami's (ML-off) median as the reference.</p>"
        '<div class="scroll"><table style="margin-bottom:0;font-size:11px"><thead><tr>'
        '<th class="c">GEMMs</th><th>same GEMMs</th><th>model %</th><th>Origami %</th>'
        "<th>&Delta; pp</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
        + (
            f'<p class="mut">{what}: no paired numbers{_why(why)}.</p>'
            if unpaired
            else ""
        )
        + (
            '<div class="scroll"><table style="margin-bottom:0;font-size:11px"><thead>'
            '<tr><th class="c">stage 7 problems</th><th>rsn</th><th>problems</th>'
            "<th>min</th><th>mean</th><th>p99</th><th>max</th>"
            "<th>[reference] ML-off p50</th></tr></thead><tbody>"
            + "".join(times)
            + "</tbody></table></div>"
            if times
            else '<p class="mut">no stage 7 selection times in this run.</p>'
        )
    )


def origami_reference_html(
    agg, traj, cells, cat_rows, s8, s7=None, why=None, weak=None
):
    """Opt-in external reference: the analytical Origami pick next to the
    model, always on the same GEMMs, opened by the five-point summary
    (`five_point_html`, with `why` no held-out GEMM of the training
    distribution pairs them), with the exclusive weak cells of Weak spots
    (`weak`, rows of `lib.weak_spots.weak_cells` carrying `origami`) on the
    audit set's problems stage 7 timed. Nothing in the pipeline or in the
    rest of this report uses it."""

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
        five_point_html(agg, s8, s7, why),
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
    weak_ori = [
        r
        for r in weak or []
        if r.get("weak") and r.get("group") == "exclusive" and r.get("origami")
    ]
    if weak or weak_ori:
        parts += [
            "<h3>Weak cells (Weak spots) on the audit set's problems stage 7 timed</h3>",
            tab(
                ["cell", "class", "problems", "model %", "Origami %", "&Delta; pp"],
                [
                    f'<tr><td class="c">{esc(r["cell"])}</td>'
                    f'<td class="c">{esc(r["class"])}</td>'
                    f'<td>{intf_(r["origami"].get("n"))}</td>'
                    f'<td>{_f(_pc(r["origami"].get("model")))}</td>'
                    f'<td>{_f(_pc(r["origami"].get("origami")))}</td>'
                    f'<td>{_f(d(_pc(r["origami"].get("model")), _pc(r["origami"].get("origami"))), signed=True)}</td></tr>'
                    for r in weak_ori
                ]
                or [
                    '<tr><td class="c" colspan="6">no weak cell has a problem stage 7 '
                    "timed with both picks</td></tr>"
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


def _analysis(
    run_dir,
    jobs,
    engine,
    notes,
    pick_measurement,
    diagnostics,
    audit_dir=None,
    cfg=None,
):
    rs.pipeline_on_path()
    try:
        from lib import kernel_identity as ki
        from lib import repeats, run_analysis

        try:
            return run_analysis.analyze_run(
                run_dir,
                jobs=jobs,
                use_engine=engine,
                pick_measurement=pick_measurement,
                diagnostics=diagnostics,
                staged=True,
                identity=True,
                audit_dir=audit_dir,
                aggregation=repeats.config_aggregation(cfg),
            )
        except ki.BuildChangedError as e:
            notes.append(f"bench analysis skipped: {e}")
            return None
    except Exception as e:  # noqa: BLE001
        notes.append(f"bench analysis failed: {type(e).__name__}: {e}")
        return None


def stage08_pick_measurement(summary, default):
    """The pick measurement a stage08 dataset's enriched rows are read
    under: its summary's; for a dataset whose misses stage08 re-benched (the
    `rebench` measurement, whose rows no enriched chunk holds) the one its
    `pre_rebench` numbers were charged under; else `default`."""
    from lib import evaluate as ev

    pm = (summary or {}).get("pick_measurement")
    if pm == ev.REBENCH_PICK_MEASUREMENT:
        pm = ((summary or {}).get("pre_rebench") or {}).get("pick_measurement")
    return pm if pm in ev.PICK_MEASUREMENTS else default


def _stage08_decomposition(run_dir, s8rows, jobs, engine, notes, pick_measurement):
    """Decomposition of every stage08 dataset with enriched chunks (report
    only), under the shipped bundle and its pool, with the dataset's own pick
    measurement (`stage08_pick_measurement`, else the config's)."""
    rs.pipeline_on_path()
    try:
        from lib import kernel_identity as ki
        from lib import run_analysis as ra

        model = ra.final_model(run_dir, want_bytes=engine, staged=True)
        if model is None:
            return {}
        tw = ra.engine_module(model) if engine else None
        lib_pool = ra.library_pool(model)
        ident = ra.library_identity(model)
        tree = ra.bundle_tree(ra.Path(run_dir), model)
        out = {}
        s8dir = os.path.join(run_dir, "validate", "stage08")
        for r in s8rows:
            ds = r["dataset"]
            enriched = ra.stage08_enriched_dir(ra.Path(s8dir), ds)
            if enriched is None:
                continue
            summ = ds.get("summary") or {}
            try:
                bench = ra.load_bench(
                    [enriched],
                    tree=tree,
                    jobs=jobs,
                    keep=ra.workload_keep(summ.get("workload_filter") or {}),
                    identity=ident,
                )
            except ki.BuildChangedError as e:
                notes.append(f"stage 8 decomposition of {r['name']} skipped: {e}")
                continue
            notes.extend(f"stage 8 dataset {r['name']}: {n}" for n in bench.notes)
            if not bench.gemms:
                continue
            e = ra.evaluate(
                bench,
                ra.pool_for(bench, lib_pool),
                model,
                tw=tw,
                pick_measurement=stage08_pick_measurement(summ, pick_measurement),
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


def pool_summary(run_dir):
    """(`lib.library_pool.summary` of the kernel pool the run's bundle
    ranks, why it is missing)."""
    rs.pipeline_on_path()
    try:
        from lib import run_analysis as ra

        model = ra.final_model(ra.Path(run_dir), want_bytes=False, staged=True)
        return ra.pool_composition(model)
    except Exception as e:  # noqa: BLE001
        return None, f"the kernel pool could not be read: {type(e).__name__}: {e}"


def subset_html(path):
    doc = load_json(path)
    if not isinstance(doc, dict) or not doc.get("sizes"):
        why = (
            "the file holds no sizes"
            if os.path.isfile(path)
            else f"{os.path.basename(path)} does not exist: "
            "scripts/choose_shipped_subset.py has not been run for this run"
        )
        return (
            '<p class="mut">no shipped_subset.json (scripts/choose_shipped_subset.py) '
            f"for this run: {esc(why)}.</p>"
        )
    _pipeline_on_path()
    from lib.shipped_subset import report_html

    return report_html(doc, esc)


def gate_check_html(analysis, sb):
    """The report's re-ranking of the deploy-gate slice next to the gate's
    own record under the same pick measurement (its `pre_rebench` numbers
    when the gate re-benched its misses); they agree when the report charges
    picks as stage06 does."""
    gate = (sb or {}).get("gate")
    mine = ((analysis or {}).get("gate") or {}).get("decomposition", {}).get("*")
    if not gate or not mine:
        return ""
    pm = (analysis or {}).get("pick_measurement")
    pre = gate.get("pre_rebench") if isinstance(gate.get("pre_rebench"), dict) else {}
    ref = gate if gate.get("pick_measurement") == pm or not pre else pre
    se = mine.get("deployed_sel_eff")
    g = ref.get("sel_eff")
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
        f"stage06/gate.json{' before its re-bench' if ref is pre else ''} "
        f"(pick measurement {esc(ref.get('pick_measurement'))}): "
        f"{_f(g * 100 if g is not None else None, 4)}% over "
        f'{intf_(gate.get("n_evaluated"))} ({intf_(ref.get("n_miss"))} misses) &mdash; '
        f'{"they agree" if agree else "<b>they differ</b>"}.</p>'
    )


def _evidence(run_dir, settings, notes):
    """The cells the orchestrator's enhance mode would select, per flag."""
    rs.pipeline_on_path()
    try:
        from lib import cell_evidence as ce

        as_enhance = ce.collect(ce.Path(run_dir))
    except Exception as e:  # noqa: BLE001
        notes.append(f"cell evidence failed: {type(e).__name__}: {e}")
        return []
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
    return selections


def _train_round(sb):
    td = str(((sb or {}).get("bundle") or {}).get("train_dir") or "")
    name = os.path.basename(os.path.dirname(td))
    return name if round_num(name) >= 0 or name == FINAL_RETRAIN_DIR else None


def _bundle_name(sb):
    """How the report names the bundle it describes."""
    kind = (sb or {}).get("kind")
    rnd = ((sb or {}).get("bundle") or {}).get("round")
    if kind == "deployed":
        return f"deployed bundle ({rnd})"
    if kind == "staged":
        return f"staged bundle ({rnd}, not deployed)"
    return "newest trained model"


def _analysis_caption(analysis, what):
    model = analysis.get("model") or {}
    deployed = (
        f"the {'deployed' if model.get('deployed') else 'staged, not deployed,'} "
        f"bundle (deploy record of {model.get('deploy_round')}, models of "
        f"{model.get('train_round')})"
        if model.get("deploy_round")
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


# ── missing results, the audit set, the headline ────────────────────────────


def _cfg(cfg, dotted):
    """Config value at `dotted` with the orchestrator's schema default."""
    rs.pipeline_on_path()
    try:
        from run_orchestrator import cfg_get

        return cfg_get(cfg, dotted)
    except Exception:  # noqa: BLE001
        cur = cfg
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return None
            cur = cur[part]
        return cur


def rounds_progress(run_dir, cfg):
    """`N of M planned rounds complete` from the round markers and
    full.total_rounds."""
    done = 0
    for d in round_dirs(run_dir):
        if round_num(d) != done or not os.path.isfile(
            os.path.join(d, ".round_complete.json")
        ):
            break
        done += 1
    total = _cfg(cfg, "full.total_rounds")
    return f"{done} of {total} planned rounds complete" if total else f"{done} complete"


def missing_reasons(run_dir, cfg, sb):
    """Why a result is missing, one line each, from the run's config and
    records: `bundle` (nothing deployed), `gate` (the bundle's deploy gate),
    `next` (the next round's stage04b of the bundle), `stage07`,
    `stage08`."""
    progress = rounds_progress(run_dir, cfg)
    gates = deploy_gates(run_dir)
    kind = (sb or {}).get("kind")
    bundle = (sb or {}).get("bundle") or {}
    ms = (sb or {}).get("measurements") or []
    out = {}
    if kind == "deployed":
        out["bundle"] = None
    elif gates:
        out["bundle"] = (
            "no bundle passed the deploy gate ("
            + "; ".join(f"{g['round']}: {g['reason'] or 'not deployed'}" for g in gates)
            + ")"
        )
    else:
        out["bundle"] = f"no round has completed stage06, the deploy gate ({progress})"
    if not bundle:
        out["gate"] = out["next"] = out["bundle"]
    elif round_num(bundle["round"]) < 0:
        out["gate"] = (
            "the final retrain has no gate GEMMs: it trained on every GEMM of its "
            "rounds, their gate slices too, and its gate is its decision on the "
            "audit set (final_retrain/decision.json)"
        )
        out["next"] = (
            "no round measures the final retrain: a round's stage04b evaluates the "
            "bundle of the round before it"
        )
    else:
        out["gate"] = (
            None
            if any(m.get("source") == "deploy_gate" for m in ms)
            else f"{bundle['round']}'s stage06 measured no gate GEMMs "
            f"(deploy.gate_frac {_cfg(cfg, 'deploy.gate_frac')})"
        )
        nxt = f"round_{round_num(bundle['round']) + 1}"
        nd = os.path.join(run_dir, nxt)
        if any(m.get("source") == "stage04b" for m in ms):
            out["next"] = None
        elif not os.path.isdir(nd):
            out["next"] = f"{nxt} has not started ({progress})"
        elif not os.path.isfile(os.path.join(nd, "stage04b", ".stage_complete.json")):
            out["next"] = f"{nxt}'s stage04b has not finished"
        else:
            out["next"] = f"{nxt}'s stage04b evaluated another models.pt"
    validated = os.path.isdir(os.path.join(run_dir, "validate"))
    at_end = (
        f"stages 7 and 8 run after the last planned round ({progress}) and on "
        "every later --mode full until they pass, or with --mode validate"
    )
    last = (load_json(os.path.join(run_dir, "run_status.json")) or {}).get(
        "validation"
    ) or {}

    def last_outcome(key, number):
        v = last.get(key) if isinstance(last.get(key), dict) else {}
        if v.get("status") not in ("failed", "not_run"):
            return []
        what = (
            f"the run's last validation: stage {number} {v['status'].replace('_', ' ')}"
        )
        what += f" (rc={v['rc']})" if v.get("rc") else ""
        return [what + (f": {v['reason']}" if v.get("reason") else "")]

    if not (
        _cfg(cfg, "stage07.bench_yamls")
        or (_cfg(cfg, "stage07.parity_sample_trained") or 0) > 0
        or (_cfg(cfg, "stage07.parity_sample_audit") or 0) > 0
    ):
        out["stage07"] = (
            "stage07 is off in the config (stage07.parity_sample_trained and "
            "stage07.parity_sample_audit are 0 and stage07.bench_yamls is empty)"
        )
    else:
        parts = last_outcome("stage07", 7) + ([] if validated else [at_end])
        if kind != "deployed":
            parts.append("stage 7 validates a deployed bundle and none was deployed")
        if not _cfg(cfg, "stage07.build_dir") and not (
            _cfg(cfg, "deploy.apply_patch") and _cfg(cfg, "deploy.build_after_apply")
        ):
            parts.append(
                "without stage07.build_dir, or deploy.apply_patch and "
                "deploy.build_after_apply, it can only run after a manual deploy and "
                "model build"
            )
        out["stage07"] = "; ".join(parts) or (
            "validate/stage07 holds no summary.json (the stage failed or is running)"
        )
    if not _cfg(cfg, "stage08.datasets"):
        out["stage08"] = "stage08.datasets is empty in the config"
    else:
        out["stage08"] = "; ".join(last_outcome("stage08", 8)) or (
            "validate/stage08 holds no summary.json (the stage failed or is running)"
            if validated
            else at_end
        )
    return out


def trained_pairing_reason(cfg, s7, stage07_reason=None):
    """Why no held-out GEMM of the training distribution pairs the model's
    pick with Origami's (`five_point_html`): planned benches time no Origami
    pick, and stage 7 pairs the runtime's picks only on a sample of the audit
    set and with the Origami reference on."""
    parts = [
        (
            "planned benches do not time Origami's pick (stage04 marks it only on "
            "the GEMMs of list-free benches)"
            if _cfg(cfg, "planner.enabled")
            else "no round's new GEMMs pair the bundle with Origami's pick yet"
        )
    ]
    if not (
        (_cfg(cfg, "stage07.parity_sample_audit") or 0) > 0 and _cfg(cfg, "audit.dir")
    ):
        parts.append(
            "stage 7 pairs the runtime's picks with tilewright on and off on the "
            "audit set's GEMMs with stage07.parity_sample_audit and audit.dir"
        )
    elif not s7:
        parts.append(
            "no stage 7 results" + (f" ({stage07_reason})" if stage07_reason else "")
        )
    elif not s7.get("origami_reference"):
        parts.append(
            "stage 7 times Origami's pick only with report.origami_reference (or "
            "--origami-reference), and this run's did not; a --mode full with it "
            "runs stage 7 again"
        )
    elif not any(y.get("held_out") for y in s7.get("yamls") or []):
        parts.append("stage 7 had no sample of the audit set to time")
    else:
        parts.append(
            "stage 7 paired no problem of the audit set's sample (see its section)"
        )
    return "; ".join(parts)


def label_extras(recs, labels):
    """(`recs` with the extra models `labels` names renamed, the names of
    `labels` that match no extra model). `labels` maps NAME to LABEL: NAME
    is an extra model's label or at least six leading hex digits of its
    model file's sha256; the first matching NAME renames it."""
    hexdigits = set("0123456789abcdef")
    out, used = [], set()
    for rnd, role, rec in recs:
        if role == "extra":
            name = rnd[len("extra: ") :]
            sha = str(rec.get("model_sha256") or "").lower()
            for key, label in (labels or {}).items():
                k = key.lower()
                by_sha = len(k) >= 6 and set(k) <= hexdigits and sha.startswith(k)
                if key == name or by_sha:
                    rnd = f"extra: {label}"
                    used.add(key)
                    break
        out.append((rnd, role, rec))
    return out, [k for k in labels or {} if k not in used]


def _paired(before, after, official=None):
    """(change, why not, protocol) of the exclusive group from audit record
    `before` to `after` ((round, role, record) each), paired GEMM by GEMM
    under the official protocol `official` when both have it; no change
    when they do not score the same GEMMs the same way."""
    rs.pipeline_on_path()
    from lib import audit as al

    a, b = dict(before[2], what=before[0]), dict(after[2], what=after[0])
    apart = rs.audit_mismatch(a, b)
    if apart:
        return None, apart, al.protocol_of([a, b], official=official)
    return al.paired_change(a, b, "exclusive", official=official)


def audit_state(run_dir, cfg, sb, recs=None):
    """The audit set as the report shows it: `head`, the latest round
    evaluation (never an extra record) of the described bundle ((round,
    role, record), matched by its model file's sha256) that scored the
    exclusive group (complete or partial);
    `change`, `lib.audit.compare` of it on the exclusive group against the
    latest earlier candidate of another model that scored that group too
    ((before round, change, why not)), `previous` that candidate ((round,
    role, record)); `latest`, its latest evaluation of
    any status; `view`, (record, protocol) of `head` as the report shows it
    (lib/audit.py `view`: the config's official protocol, audit.official,
    when complete for the bundle, else v2, else v1), `alongside` v2's
    exclusive group when `view` is under another protocol, the change under
    the protocol both evaluations have complete for the group;
    `extra`, (label, record, view, change against `head`) of every extra
    record of `recs`; and `status`, one line on the audit set now (its
    progress while it is benched). `recs` are `report_sections.
    audit_records` of the run (default its own records); a pair that does
    not score the same GEMMs the same way is not compared."""
    rs.pipeline_on_path()
    from lib import audit as al

    d = al.config_dir(cfg)
    official = al.official_protocol(cfg)
    sha = ((sb or {}).get("bundle") or {}).get("model_file_sha256")
    if recs is None:
        recs = rs.audit_records(run_dir)
    rounds = [x for x in recs if x[1] != "extra"]
    mine = [x for x in rounds if sha and x[2].get("model_sha256") == sha]
    head = next(
        (x for x in reversed(mine) if al.evaluated_group(x[2], "exclusive")), None
    )
    change = prev = None
    if head is not None:
        first = next(i for i, x in enumerate(recs) if x is mine[0])
        prev = next(
            (
                x
                for x in reversed(recs[:first])
                if x[1] == "candidate"
                and x[2].get("model_sha256") != sha
                and al.evaluated_group(x[2], "exclusive")
            ),
            None,
        )
        if prev is not None:
            change = (prev[0], *_paired(prev, head, official))
    where = ""
    if d is None:
        status = "no audit set is configured (audit.dir)"
    else:
        st, why = al.status(d)
        prog = al.read_json(d / al.PROGRESS) or {}
        done, total = prog.get("shapes_done"), prog.get("shapes_total")
        if st == "ok":
            status, where = "complete", f" ({d})"
        elif st == "incomplete" and isinstance(done, int) and isinstance(total, int):
            units = prog.get("units") or {}
            ex = units.get("exclusive") or {}
            ctx = [u for k, u in units.items() if k != "exclusive"]
            status = (
                f"the audit set is still being benched: {done:,} of {total:,} GEMMs "
                f"(exclusive {int(ex.get('shapes_done') or 0):,} of "
                f"{int(ex.get('shapes_total') or 0):,}; {len(ctx)} contexts "
                f"{sum(int(u.get('shapes_done') or 0) for u in ctx):,} of "
                f"{sum(int(u.get('shapes_total') or 0) for u in ctx):,})"
                + (
                    f", expected to finish {prog['expected_finish']}"
                    if prog.get("expected_finish")
                    else ""
                )
            )
            where = f" ({d}, progress.json of {prog.get('updated')})"
        else:
            status = why
    extra = []
    for rnd, role, rec in recs:
        if role != "extra":
            continue
        rec = dict(rec, what=rnd)
        ch = (
            _paired(head, (rnd, role, rec), official)
            if head and al.evaluated_group(rec, "exclusive")
            else None
        )
        extra.append((rnd[len("extra: ") :], rec, al.view(rec, official), ch))
    view = al.view(dict(head[2], what=head[0]), official) if head else None
    alongside = None
    if view is not None and view[1] != "v2":
        alongside = al.evaluated_group(al.protocol_record(head[2], "v2"), "exclusive")
    return {
        "head": head,
        "previous": prev,
        "view": view,
        "alongside": alongside,
        "change": change,
        "latest": mine[-1] if mine else None,
        "extra": extra,
        "status": status,
        "where": where,
    }


def audit_summary(audit, target):
    """The audit numbers of the headline as data (percent, two decimals),
    each with its protocol: the described bundle's evaluation (`bundle`,
    with v2's exclusive numbers alongside, `v2`, under another protocol)
    and every extra model (`extra`, with its audit set and why it is no
    round's evaluation)."""
    rs.pipeline_on_path()
    from lib import audit as al

    def numbers(rec, proto, **more):
        g = al.evaluated_group(rec, "exclusive") or {}
        mc = g.get("min_cell") or {}
        return dict(
            more,
            protocol=proto,
            protocol_text=al.PROTOCOLS[proto],
            coverage=rs._status_text(rec),
            cell_mean=_pct(g.get("cell_mean")),
            cell_mean_ci=[_pct(v) for v in g.get("cell_mean_ci") or []] or None,
            min_cell=mc.get("cell"),
            min_cell_sel_eff=_pct(mc.get("sel_eff")),
            n_cells=g.get("n_cells"),
            n_cells_below_target=g.get("n_cells_below_target"),
            n_gemms=g.get("n_gemms"),
            classes=rs.class_counts(g, float(rec.get("target") or target)),
        )

    out = {"bundle": None, "extra": []}
    if audit.get("head"):
        rnd, role, _ = audit["head"]
        rec, proto = audit["view"]
        out["bundle"] = numbers(rec, proto, round=rnd, role=role)
        g2 = audit.get("alongside")
        if g2:
            out["bundle"]["v2"] = dict(
                cell_mean=_pct(g2.get("cell_mean")),
                cell_mean_ci=[_pct(v) for v in g2.get("cell_mean_ci") or []] or None,
                n_cells_below_target=g2.get("n_cells_below_target"),
            )
    for label, raw, (rec, proto), ch in audit.get("extra") or []:
        c = ch[0] if ch else None
        out["extra"].append(
            numbers(
                rec,
                proto,
                model=label,
                model_sha256=raw.get("model_sha256"),
                source=raw.get("source"),
                audit_set=raw.get("audit_set"),
                audit_dir=raw.get("audit_dir"),
                extra_reason=raw.get("extra_reason"),
                vs_bundle=(
                    None
                    if c is None
                    else {
                        "protocol": c["protocol"],
                        "cell_mean_ratio": c["cell_mean_ratio"],
                        "cell_mean_ratio_ci": c["cell_mean_ratio_ci"],
                    }
                ),
            )
        )
    return out


def _tile(label, value, note="", cls=""):
    """One headline tile; `label`, `value` and `note` are HTML."""
    return (
        f'<div class="stat{(" " + cls) if cls else ""}"><div class="v">{value}</div>'
        f'<div class="l">{label}</div>'
        + (f'<div class="n">{note}</div>' if note else "")
        + "</div>"
    )


def _vs(v, target):
    return "" if v is None else ("ok" if v >= target else "bad")


def _classes_note(counts):
    return ", ".join(
        f"{k.replace('_', '-')} "
        + ("not recorded" if counts.get(k) is None else f"{counts[k]:,}")
        for k in ("pool_limited", "identity_limited", "model_limited", "passing")
    )


def _extra_tiles(audit, target):
    """Headline tiles of the extra records of --audit-results (models that
    are no round of the run, or a round's model scored on another audit set
    or pick rule), each against the described bundle's evaluation when the
    two score the same GEMMs the same way."""
    rs.pipeline_on_path()
    from lib import audit as al

    t, head = [], audit.get("head")
    for label, raw, (rec, proto), ch in audit.get("extra") or []:
        g = al.evaluated_group(rec, "exclusive")
        what = esc(
            f"audit set {raw.get('audit_set')}: extra model {label} "
            f"({raw.get('extra_reason')})"
        )
        if g is None:
            t.append(_tile(what, "not measured", esc(rs._status_text(rec)), "na"))
            continue
        mc = g.get("min_cell") or {}
        mean = _num(g.get("cell_mean"))
        rt = float(rec.get("target") or target)
        vs = ""
        if ch and ch[0]:
            c, rci = ch[0], ch[0].get("cell_mean_ratio_ci")
            vs = (
                f"; paired against {head[0]}'s candidate: \u00d7"
                f"{c['cell_mean_ratio']:.4f}"
                + (f" [{rci[0]:.4f}, {rci[1]:.4f}]" if rci else "")
                + f" (protocol {c['protocol']})"
            )
        elif ch:
            vs = f"; not compared with {head[0]}'s candidate: {ch[1]}"
        t.append(
            _tile(
                f"{what}: mean of per-cell selection efficiency against the true "
                f"best [95% CI], protocol {proto}",
                f"{_f(mean * 100 if mean is not None else None)}%"
                + esc(_ci_txt(g.get("cell_mean_ci"), 100)),
                esc(
                    f"exclusive, {g.get('n_gemms')} GEMMs over {g.get('n_cells')} "
                    f"cells; {rs._status_text(rec)}; minimum cell {mc.get('cell')} "
                    f"{_f(_pct(mc.get('sel_eff')))}%; "
                    f"{intf_(g.get('n_cells_below_target'))} of "
                    f"{intf_(g.get('n_cells'))} cells below {rt * 100:g}% "
                    f"({_classes_note(rs.class_counts(g, rt))}){vs}; model "
                    f"sha256 {str(raw.get('model_sha256'))[:12]} from "
                    f"{os.path.basename(str(raw.get('source') or ''))}"
                ),
                _vs(mean, target),
            )
        )
    return t


def headline_html(sb, audit, reasons, target, held, rows, s7, extra):
    """The headline: the run's position on held-out data. With a complete
    audit evaluation of the described bundle its per-cell mean against the
    true best, minimum cell and cells below the target come first; then the
    bundle (deployed or staged), its deploy gate, the next round's stage04b
    of it, its leaf cells against the target and stage 7. `held` holds the
    leaves' held-out values (%), `rows` the "Where to improve" rows and
    `extra` the runtime-weighted value, model file size and shapes."""
    rs.pipeline_on_path()
    from lib import audit as al

    t, tp = [], target * 100
    name = _bundle_name(sb)
    head = audit.get("head")
    if head:
        rnd, role, _ = head
        rec, proto = audit["view"]
        g = (rec.get("groups") or {}).get("exclusive") or rec.get("headline") or {}
        mc = g.get("min_cell") or {}
        cl = rs.class_counts(g, float(rec.get("target") or target))
        ci = g.get("cell_mean_ci")
        mean = _num(g.get("cell_mean"))
        cover = rs._status_text(rec)
        retrain = sb.get("final_retrain") or {}
        picked = (
            "final_retrain"
            if retrain.get("accepted")
            else (retrain.get("baseline") or {}).get("round")
        )
        chosen = (
            f"; {retrain['selection_note']} (final_retrain/decision.json)"
            if sb.get("kind") == "deployed"
            and retrain.get("selected_by_audit")
            and retrain.get("selection_note")
            and (sb.get("bundle") or {}).get("round") == picked
            else ""
        )
        t.append(
            _tile(
                "audit set: mean of per-cell selection efficiency against the true best "
                f"(cells weighted equally) [95% CI], protocol {proto}",
                f"{_f(mean * 100 if mean is not None else None)}%"
                + esc(_ci_txt(ci, 100)),
                esc(
                    f"{name}, {role} evaluation of {rnd}: exclusive, {g.get('n_gemms')} "
                    f"GEMMs over {g.get('n_cells')} cells, never trained on; {cover}; "
                    f"protocol {al.PROTOCOLS[proto]}{chosen}"
                ),
                _vs(mean, target),
            )
        )
        g2 = audit.get("alongside")
        if g2:
            m2 = _num(g2.get("cell_mean"))
            t.append(
                _tile(
                    "audit set: the same under protocol v2, alongside",
                    f"{_f(m2 * 100 if m2 is not None else None)}%"
                    + esc(_ci_txt(g2.get("cell_mean_ci"), 100)),
                    esc(
                        f"{g2.get('n_cells_below_target')} of {g2.get('n_cells')} "
                        f"cells below {tp:g}%; protocol {al.PROTOCOLS['v2']}, "
                        f"windows of a few milliseconds"
                    ),
                    _vs(m2, target),
                )
            )
        if audit.get("change"):
            before, ch, why, cproto = audit["change"]
            if ch is None:
                t.append(
                    _tile(
                        f"audit set: change since {esc(before)}'s candidate",
                        "not compared",
                        esc(why),
                        "na",
                    )
                )
            else:
                r = ch["cell_mean_ratio"]
                rci = ch.get("cell_mean_ratio_ci")
                t.append(
                    _tile(
                        f"audit set: change since {esc(before)}'s candidate "
                        "(exclusive, paired; cell mean ratio [95% CI]), protocol "
                        f"{cproto}",
                        f"&times;{r:.4f}"
                        + (esc(f" [{rci[0]:.4f}, {rci[1]:.4f}]") if rci else ""),
                        esc(
                            f"{len(ch['improved'])} cell(s) improved, "
                            f"{len(ch['regressed'])} regressed beyond their intervals, "
                            f"over {ch['n_gemms']} GEMMs both evaluated; protocol "
                            f"{al.PROTOCOLS[cproto]}"
                        ),
                        (
                            "ok"
                            if rci and rci[0] > 1
                            else ("bad" if rci and rci[1] < 1 else "")
                        ),
                    )
                )
        mse = _num(mc.get("sel_eff"))
        t.append(
            _tile(
                f"audit set: minimum cell, protocol {proto}",
                f"{_f(mse * 100 if mse is not None else None)}%",
                esc(mc.get("cell") or ""),
                _vs(mse, target),
            )
        )
        t.append(
            _tile(
                f"audit set: cells below {tp:g}%, protocol {proto}",
                f"{intf_(g.get('n_cells_below_target'))} of {intf_(g.get('n_cells'))}",
                esc(_classes_note(cl)),
            )
        )
    else:
        latest = audit.get("latest")
        rec = latest[2] if latest else {}
        if rec.get("status") == "partial":
            why = f"{rs._status_text(rec)}; the exclusive group is not evaluated yet"
        elif rec.get("status") in (None, "incomplete"):
            why = audit.get("status")
        else:
            why = rec.get("reason")
        t.append(
            _tile(
                "audit set: mean of per-cell selection efficiency against the true best "
                "(the goal's measure)",
                "not measured",
                esc(why or "no audit evaluation of this bundle"),
                "na",
            )
        )
    t += _extra_tiles(audit, target)
    bundle = sb.get("bundle")
    kind = sb.get("kind")
    if bundle:
        if kind == "staged":
            why = esc(sb.get("reason") or "")
        elif round_num(bundle["round"]) < 0:
            why = esc(
                "every cell retrained on the data of its rounds; its gate on the "
                f"audit set: {(sb.get('final_retrain') or {}).get('reason')}"
            )
        else:
            why = "passed the deploy gate"
        t.append(
            _tile(
                "bundle this report describes",
                esc(
                    f"{bundle['round']}: "
                    + ("deployed" if kind == "deployed" else "staged, not deployed")
                ),
                why,
            )
        )
    else:
        t.append(_tile("bundle", "none", esc(reasons.get("bundle")), "na"))
    ms = sb.get("measurements") or []
    gm = next((m for m in ms if m.get("source") == "deploy_gate"), None)
    gate = sb.get("gate") or {}
    if gm and gm.get("sel_eff") is not None:
        t.append(
            _tile(
                f"held-out sel_eff: deploy gate of {esc(gm['round'])} (stage06)",
                f"{_f(gm['sel_eff'] * 100)}%",
                esc(
                    f"{gm['n_eval']:,} GEMMs, {gm['n_miss']:,} misses, pick measurement "
                    f"{gm.get('pick_measurement')}; "
                    + ("passed" if gate.get("passed") else "not passed")
                ),
                _vs(gm["sel_eff"], target),
            )
        )
    else:
        t.append(
            _tile(
                "held-out sel_eff: deploy gate", "n/a", esc(reasons.get("gate")), "na"
            )
        )
    later = [m for m in ms if m.get("source") == "stage04b" and m.get("sel_eff")]
    for m in later:
        t.append(
            _tile(
                f"held-out sel_eff: {esc(m['round'])} stage04b (its new GEMMs, same "
                "models.pt)",
                f"{_f(m['sel_eff'] * 100)}%",
                esc(
                    f"{m['n_eval']:,} GEMMs, {m['n_miss']:,} misses, pick measurement "
                    f"{m.get('pick_measurement')}"
                ),
                _vs(m["sel_eff"], target),
            )
        )
    if not later:
        t.append(
            _tile(
                "held-out sel_eff: next round's stage04b",
                "n/a",
                esc(reasons.get("next")),
                "na",
            )
        )
    vals = [v for v in held if v is not None]
    if vals:
        t.append(
            _tile(
                "leaf cells below the target on held-out data",
                f"{sum(1 for v in vals if v < tp):,} of {len(vals):,}",
                esc(
                    "per leaf, every measurement of the bundle on GEMMs it never "
                    "trained on, pooled"
                    + (
                        f"; {len(held) - len(vals)} leaves without one"
                        if len(held) > len(vals)
                        else ""
                    )
                ),
            )
        )
    if rows:
        n_act = sum(1 for r in rows if r["action"] not in ("none", rs.NO_EVIDENCE))
        t.append(
            _tile(
                "leaf cells needing an action (held-out evidence)",
                f"{n_act:,}",
                f"expected gain if all reach the target: "
                f"+{_f(sum(r['gain'] for r in rows) * 100)} pp",
            )
        )
    if extra.get("rt") is not None:
        t.append(
            _tile(
                "runtime-weighted held-out sel_eff of the leaves",
                f"{_f(extra['rt'])}%",
                "summed winner time over summed pick time, per leaf GEMM-weighted",
            )
        )
    for y, v, c in s7.get("sel") or []:
        t.append(
            _tile(
                f"runtime selection efficiency, stage 7 {esc(y)} [95% CI]",
                f"{_f(v)}%{esc(_ci_txt(c, 1))}",
            )
        )
    for y, rsn, v, c in s7.get("first") or []:
        w = next((x for x in s7.get("warm") or [] if x[0] == y), None)
        t.append(
            _tile(
                f"selection time p50, first call into a cell / warm (stage 7 "
                f"{esc(y)}, rsn={esc(rsn)})",
                f"{_f(v)}{esc(_ci_txt(c))} / "
                + (f"{_f(w[2])}{esc(_ci_txt(w[3]))}" if w else "&ndash;")
                + " &micro;s",
            )
        )
    if not s7.get("sel") and not s7.get("first"):
        t.append(
            _tile(
                "stage 7: runtime selection efficiency and selection time",
                "n/a",
                esc(reasons.get("stage07")),
                "na",
            )
        )
    if extra.get("bin_mb") is not None:
        t.append(_tile(f"model file of the {esc(name)}", f"{extra['bin_mb']} MB"))
    if extra.get("shapes") is not None:
        t.append(_tile("shapes generated", f"{extra['shapes']:,}"))
    return f'<div class="stats">{"".join(t)}</div>'


def insample_html(summary):
    """The in-sample numbers, apart from the headline."""
    s = summary
    tiles = [
        _tile(
            "in-sample mean over leaf cells (stage05)",
            (
                f"{_f(s.get('in_sample_mean'))}%"
                if s.get("in_sample_mean") is not None
                else "n/a"
            ),
            "" if s.get("in_sample_mean") is not None else "no stage05 cells.json",
        ),
        _tile(
            "training benches: deployed picks / misses (report's re-ranking)",
            (
                f"{_f(s['bench_sel_eff'])}% / {intf_(s.get('bench_miss'))}"
                if s.get("bench_sel_eff") is not None
                else "n/a"
            ),
            "" if s.get("bench_sel_eff") is not None else "no bench analysis",
        ),
        _tile(
            "training GEMMs without a model pick",
            (
                f"{intf_(s['bench_unserved'])} of {intf_(s.get('bench_gemms'))}"
                if s.get("bench_unserved") is not None
                else "n/a"
            ),
        ),
    ]
    return (
        '<details class="insample"><summary>In-sample numbers (GEMMs the model trained '
        "on): context only, not a measure of the goal</summary>"
        f'<div class="stats">{"".join(tiles)}</div></details>'
    )


def stage07_origami_rows(run_dir):
    """stage 7's per-problem picks with tilewright on and off
    (validate/stage07/<yaml>/origami/origami.json) of the bench yamls no
    round trained on (the audit set's sample)."""
    base = os.path.join(run_dir, "validate", "stage07")
    summ = load_json(os.path.join(base, "summary.json"))
    out = []
    if not isinstance(summ, dict):
        return out
    for tag, y in sorted((summ.get("per_yaml") or {}).items()):
        if not isinstance(y, dict) or not y.get("held_out"):
            continue
        doc = load_json(os.path.join(base, str(tag), "origami", "origami.json"))
        if isinstance(doc, dict):
            out += [r for r in doc.get("per_problem") or [] if isinstance(r, dict)]
    return out


def weak_spots(
    run_dir,
    cfg,
    audit,
    *,
    jobs=1,
    bench=True,
    origami_reference=False,
    threshold=None,
    peak_tflops=None,
    peak_bandwidth_gbs=None,
    notes=None,
):
    """The Weak spots section's data (`lib.weak_spots`): {rows (one per
    audit group and cell of the described bundle, `weak_cells`, paired
    with the previous candidate when both score the same GEMMs the same
    way, its regressions those of the per-cell gate's cell test), analysis
    (the roofline of the run's exhaustively benched exclusive GEMMs per
    cell, and the pool gaps of its training side, `lib.weak_spots.analyze`;
    None with --no-bench), why (what is missing), previous (the previous
    candidate's round), settings (threshold and the gate's)}."""
    rs.pipeline_on_path()
    from lib import audit as al
    from lib import weak_spots as ws

    gate = al.cell_gate_settings(cfg) or dict(al.CELL_GATE_DEFAULTS)
    if threshold is None:
        threshold = (cfg.get("report") or {}).get("pool_gap_fraction")
    threshold = float(threshold if threshold is not None else ws.POOL_GAP_FRACTION)
    why, analysis = [], None
    if bench:
        try:
            analysis = ws.analyze(
                run_dir,
                cfg,
                audit_dir=al.config_dir(cfg),
                jobs=jobs,
                threshold=threshold,
                peak_tflops=peak_tflops,
                peak_bandwidth_gbs=peak_bandwidth_gbs,
            )
        except Exception as e:  # noqa: BLE001
            if notes is not None:
                notes.append(
                    f"weak spots: the roofline failed: {type(e).__name__}: {e}"
                )
            why.append("no roofline (see the notes above)")
        else:
            if not analysis.get("n_cell_gemms"):
                why.append(
                    "no roofline: "
                    + (analysis.get("why") or "no exhaustively benched exclusive GEMM")
                )
            elif not analysis.get("cells_held_out") and analysis.get("cells_why"):
                why.append(
                    f"the cells' roofline reads {analysis['cells_source']}, not the "
                    f"audit set: {analysis['cells_why']}"
                )
            if analysis.get("n_cell_gemms") and not analysis.get("n_gemms"):
                why.append(
                    "no pool gaps: "
                    + (analysis.get("why") or "no exhaustively benched training GEMM")
                )
    else:
        why.append("no roofline: --no-bench skips the exhaustive benches")
    head, prev = audit.get("head"), audit.get("previous")
    rec = dict(head[2], what=head[0]) if head else None
    prev_rec = dict(prev[2], what=prev[0]) if prev else None
    prev_why = None if prev else "no previous candidate to pair with"
    if rec is not None and prev_rec is not None and rs.audit_mismatch(rec, prev_rec):
        prev_rec = None
        prev_why = f"{prev[0]}'s record does not score the same GEMMs the same way"
    if rec is None:
        why.append(f"no audit evaluation of the bundle ({audit.get('status')})")
    rows = ws.weak_cells(
        rec,
        prev_rec,
        roof_cells=(analysis or {}).get("cells"),
        origami=(
            ws.origami_cells(stage07_origami_rows(run_dir))
            if origami_reference
            else None
        ),
        threshold=threshold,
        settings=gate,
        bootstrap=int(al.settings(cfg)["bootstrap"]),
        previous_why=prev_why,
        official=al.official_protocol(cfg),
    )
    return {
        "rows": rows,
        "analysis": analysis,
        "why": "; ".join(why),
        "previous": prev[0] if prev_rec is not None else None,
        "settings": {"threshold": threshold, **gate},
    }


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
    audit_results=(),
    audit_labels=None,
    weak_spots_dir=None,
    pool_gap_fraction=None,
    peak_tflops=None,
    peak_bandwidth_gbs=None,
):
    run_dir = run_dir.rstrip("/")
    notes = []
    own = [str(p) for p in run_bundles.audit_result_files(run_dir)]
    external, unread = rs.read_audit_results(
        list(dict.fromkeys([*map(str, audit_results), *own]))
    )
    notes += unread
    config_text = config_source(run_dir)
    cfg = run_config(run_dir, config_text)
    gone = left_out_contexts(config_text, cfg)
    if no_holdout_left(gone, cfg):
        notes.append(
            "execution contexts: every held-out context of the config was left out "
            "on the run's bench device, so the run measured the models on no "
            "held-out context (see Execution contexts)"
        )
    settings = rs.run_settings(config_text)
    if target is not None:
        settings["target"] = float(target)
    tgt = settings["target"]
    pick_measurement = settings["pick_measurement"]
    sb = subject_bundle(run_dir)
    name = _bundle_name(sb)
    measured = pooled_cells(sb.get("measurements"))
    held = measured or held_out_breakdown(run_dir)
    sources = ", ".join(
        f"{m.get('source')} {m.get('round')}" for m in sb.get("measurements") or []
    )
    held_note = (
        f"the {esc(name)} on GEMMs it never trained on, every measurement pooled per "
        f"cell ({esc(sources)})"
        if measured
        else "no measurement of the bundle on GEMMs it never trained on: the latest "
        "stage04b measurement per leaf (each by the model of the round before it)"
    )
    agg = held_out_aggregate(held)
    traj = held_out_rounds(run_dir)
    train = training_rounds(run_dir)
    notes += carried_notes(train)
    cells_full = final_cells(run_dir, held, upto=_train_round(sb))
    s7 = stage07(run_dir)
    s8 = stage08(run_dir)
    t_rounds, t_total = timing_rows(run_dir)
    try:
        run_legacy = run_layout.of(run_dir).legacy
    except ValueError as e:
        notes.append(f"stage directories: {e}")
        run_legacy = False
    cat_rows, base_rows, n_cat = held_out_by_category(run_dir, held)
    in_sample = [c["model"] for c in cells_full if c.get("model") is not None]
    reasons = missing_reasons(run_dir, cfg, sb)
    rs.pipeline_on_path()
    from lib import audit as auditlib

    recs, unmatched = label_extras(
        rs.audit_records(
            run_dir,
            external,
            audit_dir=auditlib.config_dir(cfg),
            pick_measurement=pick_measurement,
        ),
        audit_labels,
    )
    notes += [
        f"--audit-label {name}: no extra model has this label or sha256 prefix"
        for name in unmatched
    ]
    audit = audit_state(run_dir, cfg, sb, recs)

    analysis = (
        _analysis(
            run_dir,
            jobs,
            engine,
            notes,
            pick_measurement,
            diagnostics,
            auditlib.config_dir(cfg),
            cfg,
        )
        if bench
        else None
    )
    decomp = (analysis or {}).get("decomposition") or {}
    decomp_held = ((analysis or {}).get("held_out") or {}).get("decomposition") or {}
    selections = _evidence(run_dir, settings, notes)
    weak = weak_spots(
        run_dir,
        cfg,
        audit,
        jobs=jobs,
        bench=bench,
        origami_reference=origami_reference,
        threshold=pool_gap_fraction,
        peak_tflops=peak_tflops,
        peak_bandwidth_gbs=peak_bandwidth_gbs,
        notes=notes,
    )
    exported = None
    if weak_spots_dir:
        from lib import weak_spots as ws

        try:
            exported = [
                os.path.relpath(p, run_dir)
                for p in ws.write_exports(
                    weak_spots_dir,
                    weak["analysis"]
                    or {"threshold": weak["settings"]["threshold"], "why": weak["why"]},
                    weak["rows"],
                    run=os.path.basename(run_dir),
                    bundle=name,
                )
            ]
        except OSError as e:
            notes.append(f"weak spots: cannot write {weak_spots_dir}: {e}")
    audit_cells = (
        ((audit["view"][0].get("groups") or {}).get("exclusive") or {}).get("per_cell")
        if audit["head"]
        else None
    )
    leaves = {
        c["cell"]: {
            "n_gemms": (decomp.get(c["cell"]) or {}).get("n_gemms", c.get("n_gemms")),
            "in_sample": c["deployed"] / 100 if c.get("deployed") is not None else None,
        }
        for c in cells_full
    }
    improve = rs.improve_rows(
        rs.heldout_evidence(
            leaves,
            held,
            decomp_held,
            decomp,
            audit_cells,
            stage04b_history(run_dir),
        ),
        settings,
    )

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
            "train_round": _train_round(sb) if shipped else None,
            "source": sb.get("source") if shipped else None,
            "sel_eff": _pct(sb.get("held_out_sel_eff")) if shipped else None,
            "n": sb.get("n_gemms") if shipped and sb.get("source") else None,
            "n_miss": sb.get("n_miss") if shipped else None,
            "n_cells": sb.get("n_cells") if shipped else None,
        },
        "bundle": {
            "kind": sb.get("kind"),
            "name": name,
            "round": (sb.get("bundle") or {}).get("round"),
            "source": sb.get("source"),
            "sel_eff": _pct(sb.get("held_out_sel_eff")),
            "n": sb.get("n_gemms") if sb.get("source") else None,
            "n_miss": sb.get("n_miss") if sb.get("source") else None,
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
            "shipped": bool(measured),
        },
        "in_sample_mean": (
            round(sum(in_sample) / len(in_sample), 2) if in_sample else None
        ),
        "rt_model": rt_eff(),
        "n_cells": len(cells),
        "bin_mb": deployed_size_mb(run_dir, (sb.get("bundle") or {}).get("round")),
        "shapes": shapes_total(run_dir),
        "wall": fmt_dur(t_total) if t_total else None,
        "n_action": sum(
            1 for r in improve if r["action"] not in ("none", rs.NO_EVIDENCE)
        ),
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
    train_rows = (
        "".join(
            f'<tr><td class="c">{esc(r["round"])}</td><td>{_f(r["model"])}</td><td>{_f(r["deployed"])}</td>'
            f'<td>{esc(r["n_cells"])}</td><td>{len(r["carried"])}</td>'
            f'<td>{esc(r["n_gemms"])}</td></tr>'
            for r in train
        )
        or '<tr><td class="c" colspan="6">no stage05 metrics.json in this run</td></tr>'
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
                analysis,
                "Training benches of the bundle (in-sample: the GEMMs it trained on)",
            )
            + rs.decomposition_html(analysis, order=base_order),
            "DECOMP_HELD": (
                f'<p class="cap">{held_out["n_gemms"]:,} GEMMs the {esc(name)} never '
                "trained on: the deploy-gate slice of its training round and the stage04 "
                "rows of later rounds, routed to the leaves of its own split tree, under "
                "the same model, pool and pick measurement as the training benches. "
                "<b>Where to improve</b> takes each cell's largest loss from here.</p>"
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
                else f'<p class="mut">no bench of GEMMs the {esc(name)} never trained '
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
    contexts = left_out_html(gone, cfg) + (
        rs.contexts_html(analysis, s4b_ctx, settings, context_names(cfg))
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
        "AUDIT": audit_summary(audit, tgt),
    }
    idt_why = "the identity numbers come from the bench analysis, which " + (
        "--no-bench skips"
        if not bench
        else "found no training benches or failed (see the notes above)"
    )
    parts = {
        "TITLE": esc(summary["run"]),
        "CAP": esc(
            f"Run {summary['run']} \u2013 {summary['rounds']} rounds"
            + (f" \u2013 wall-clock {summary['wall']}" if summary["wall"] else "")
            + f" \u2013 target {tgt * 100:.1f}% of the fastest measured kernel in "
            "every cell \u2013 held-out numbers first"
        ),
        "HEADLINE": headline_html(
            sb,
            audit,
            reasons,
            tgt,
            [c.get("held") for c in cells_full],
            improve,
            stage07_tiles(s7),
            {
                "rt": summary["rt_model"],
                "bin_mb": summary["bin_mb"],
                "shapes": summary["shapes"],
            },
        ),
        "INSAMPLE": insample_html(summary),
        "NOTES": "".join(f'<p class="mut">{esc(n)}</p>' for n in notes),
        "BUNDLE_NAME": esc(name),
        "SHIPPED": bundle_html(sb, tgt, deploy_gates(run_dir), reasons.get("next")),
        "AUDIT": rs.audit_html(
            run_dir,
            tgt,
            audit["status"] + audit["where"],
            recs,
            official=auditlib.official_protocol(cfg),
        ),
        "IDENTITY": rs.identity_html((analysis or {}).get("identity"), idt_why),
        "POOL": rs.pool_html(*pool_summary(run_dir)),
        "SUBSET": subset_html(
            shipped_subset or os.path.join(run_dir, "shipped_subset.json")
        ),
        "IMPROVE": rs.improve_html(
            improve,
            settings,
            selections,
            name,
            "" if audit["head"] else audit["status"],
        ),
        "CONTEXTS": contexts,
        "WEAK": rs.weak_spots_html(
            weak["rows"],
            weak["analysis"],
            weak["settings"],
            bundle=name,
            previous=weak["previous"],
            why=weak["why"],
            exported=exported,
        ),
        "SELTIME": rs.selection_time_html(
            s7_cells,
            float(selection_budget_us),
            origami_reference,
            reasons.get("stage07"),
        ),
        "S7": stage07_html(s7, origami_reference, reasons.get("stage07")),
        "S8": stage08_html(s8rows, tgt, reasons.get("stage08")) + s8_dec_html,
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
            "In-sample selection efficiency per leaf cell (its own training GEMMs; "
            "context only)",
        ),
        "CATEGORIES": category_html(cat_rows, base_rows, n_cat, tgt),
        "TREE": esc(cell_tree_ascii(cells_full)),
        "CELLMAP": cell_map_html(cells_full, tgt, name),
        "TIMING": timing_html(t_rounds, t_total, run_legacy),
        "CONFIG": esc(config_text),
        "ORIGAMI": (
            origami_reference_html(
                agg,
                traj,
                cells_full,
                cat_rows,
                s8,
                s7,
                trained_pairing_reason(cfg, s7, reasons.get("stage07")),
                weak=weak["rows"],
            )
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
.stat{border:1px solid var(--bd);border-radius:8px;padding:9px 13px;min-width:120px;max-width:380px;}
.stat .v{font-size:21px;font-weight:600;} .stat .l{color:var(--mut);font-size:11.5px;margin-top:2px;}
.stat .n{color:var(--mut);font-size:11px;margin-top:3px;} .stat.ok .v{color:#1a7f37;} .stat.bad .v{color:#cf222e;} .stat.na .v{color:var(--mut);}
details.insample{margin:4px 0 10px;} details.insample .stat{opacity:.85;}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;margin-bottom:6px;}
th,td{padding:4px 9px;border-bottom:1px solid var(--bd);text-align:right;white-space:nowrap;}
th{position:sticky;top:0;background:#f6f8fa;font-weight:600;cursor:pointer;user-select:none;} th:hover{background:#eaeef2;}
.sortind{color:var(--mut);font-size:10px;}
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
<p class="cap" id="cap">__CAP__</p>
__HEADLINE__
__INSAMPLE__
__NOTES__
<h2 id="audit">Audit set</h2>
__AUDIT__

<h2 id="shipped">Bundle: __BUNDLE_NAME__</h2>
__SHIPPED__
<p class="cap"><b>Cells</b> are <code>M-tier|N-tier|K-tier|Batch</code>: M and N Tiny &le;32, Small &le;128, Mid &le;512, Large &gt;512; K TinyK &le;32, MidK &le;512, LargeK &gt;512; Bnone = unbatched, Bany = batched. Split suffixes such as <code>#K&gt;4096</code> subdivide a cell. Selection efficiency of a GEMM = time of the fastest measured solution / time of the picked solution (every solution of the library counts with its own time, also when several share kernel parameters); aggregates are geometric means. Every target and gap in this report is absolute, against the fastest measured kernel. <b>Held-out</b> = GEMMs the model never trained on; <b>in-sample</b> = GEMMs it trained on, shown as context only. What a <b>kernel</b> is to the model: <a href="#identity">Kernel identity</a>. Click a column header to sort a table (again to reverse).</p>

<h2 id="improve">Where to improve (held-out evidence)</h2>
__IMPROVE__

<h2 id="weak">Weak spots</h2>
__WEAK__

<h2 id="decomp_held">Loss decomposition per cell (GEMMs the bundle never trained on)</h2>
__DECOMP_HELD__

<h2 id="decomp">Loss decomposition per cell (training benches, in-sample)</h2>
__DECOMP__

<h2 id="contexts">Execution contexts</h2>
__CONTEXTS__

<h2 id="pool">Kernel pool</h2>
__POOL__

<h2 id="identity">Kernel identity</h2>
__IDENTITY__

<h2 id="kernels">Kernel usage</h2>
__KERNELS__

<h2 id="params">Kernel parameters</h2>
__PARAMS__
__DIAG__

<h2 id="prune">Pool-drop candidates</h2>
__PRUNE__

<h2 id="subset">Shipped candidate set</h2>
__SUBSET__

<h2 id="seltime">Selection time per cell (stage 7)</h2>
__SELTIME__

<h2>Stage 7 &mdash; selection time and C++/Python pick parity</h2>
<p class="cap">Selection time is hipBLASLt's per-call "Solution selection time" with tilewright (ML-on) for the request size shown (rsn), median over repetitions per problem, summarized over problems; the first call of each run (library load) is excluded, and the rest split into the first call of a run into a serving cell and the warm calls after it. Intervals are bootstrap 95% intervals of the median. Pick parity compares the pool position of hipBLASLt's top-1 pick and its serving cell with the offline ranking of the same model and library; the runtime selection efficiency charges those top-1 picks with the bench timings of the same problems.</p>
__S7__

<h2>Stage 8 &mdash; selection efficiency on validation datasets</h2>
__S8__
<div id="s8charts"></div>

<h2>Held-out selection efficiency per round (stage04b)</h2>
<p class="cap">The model of the previous round on this round's new shapes (history; each row is another model). The bold last row is the headline measurement of the __BUNDLE_NAME__ and stays last when sorting.</p>
<table id="traj" style="max-width:900px"><thead><tr><th class="c">round</th><th>model %</th><th>vs target pp</th><th>GEMMs</th><th>misses</th></tr></thead><tbody id="tbody_traj"></tbody></table>

<h2 id="cellh">Leaf cells of the __BUNDLE_NAME__</h2>
<p class="cap"><b>held-out</b> values: __HELD_NOTE__ (<b>*</b> = inherited from the parent cell); the in-sample columns are its own training GEMMs (context only). vs target = held-out minus the target. K = whitelist size; ed/hd/ih = embedding / hidden / interaction widths.</p>
<div class="scroll"><table id="cells" style="margin-bottom:0;font-size:11px"><thead><tr><th class="c">#</th><th class="c">cell</th><th>GEMMs</th><th>held-out %</th><th>vs target pp</th><th>in-sample model %</th><th>in-sample deployed %</th><th>K&Delta;</th><th>K</th><th class="c">ed/hd/ih</th></tr></thead><tbody id="tbody_cells"></tbody></table></div>
__CHART_HELD__
__CHART_INSAMPLE__

<h2>Held-out GEMMs by shape category</h2>
__CATEGORIES__

<h2>Split tree</h2>
<pre class="cfg">__TREE__</pre>

<h2 id="cellmap">Cells in M / N / K</h2>
__CELLMAP__

<h2>Training (stage05, in-sample)</h2>
<p class="cap">Selection efficiency on the GEMMs of the cells trained in that round (the GEMMs they trained on; context only): model = full ranking, deployed = whitelist-restricted ranking as shipped. Cells carried: cells the round was to train but stage05 did not, with which it went on (<code>train.on_cell_failure: carry</code>; the notes at the top name them).</p>
<table style="max-width:900px"><thead><tr><th class="c">round</th><th>model %</th><th>deployed %</th><th>cells trained</th><th>cells carried</th><th>GEMMs</th></tr></thead><tbody>__TRAINROWS__</tbody></table>
__CHART_TREND__

<h2>Wall-clock</h2>
__TIMING__

<h2>Configuration</h2>
<details><summary>config (as written, before environment expansion)</summary><pre class="cfg">__CONFIG__</pre></details>
__ORIGAMI__

<script>
const TWSORT=(function(){
 const MISSING=/^(?:n\/?a|[-\u2013\u2014\u2212?])?$/i;
 const DUR=/^(?:(\d+)h(\d+)m|(\d+)m(\d+)s|(\d+)s)(?=$|\s)/;
 const NUM=/^([+\-\u2212\u2013]?)(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)(?:e([+\-]?\d+))?\s*(?:%|pp|[\u00b5\u03bcu]s|ms|s|mb|kb|gb|x|\u00d7)?\*?(?=$|[\s\[(\/])/i;
 const COLL=new Intl.Collator(undefined,{numeric:true,sensitivity:"base"});
 function key(text){
  const t=String(text==null?"":text).replace(/\s+/g," ").trim();
  if(MISSING.test(t))return null;
  let m=DUR.exec(t);
  if(m)return{n:m[1]!=null?m[1]*3600+m[2]*60:m[3]!=null?m[3]*60+ +m[4]:+m[5],s:t};
  m=NUM.exec(t);
  if(!m)return{n:null,s:t};
  let v=parseFloat(m[2].replace(/,/g,""))*(m[3]?Math.pow(10,+m[3]):1);
  return{n:m[1]&&m[1]!=="+"?-v:v,s:t};
 }
 function order(keys,fixed,dir){
  const mov=keys.map((k,i)=>i).filter(i=>!fixed[i]);
  const numeric=mov.every(i=>keys[i]===null||keys[i].n!==null);
  mov.sort((a,b)=>{const x=keys[a],y=keys[b];
   if(x===null||y===null)return x===y?a-b:(x===null?1:-1);
   const d=numeric?x.n-y.n:COLL.compare(x.s,y.s);
   return (dir<0?-d:d)||a-b;});
  let j=0;return keys.map((k,i)=>fixed[i]?i:mov[j++]);
 }
 function headRows(table){
  if(table.tHead)return[...table.tHead.rows];
  const out=[];for(const r of table.rows){if(r.cells.length&&[...r.cells].every(c=>c.tagName==="TH"))out.push(r);else break;}
  return out;
 }
 function column(table,th){
  const grid=[];const rows=headRows(table);
  for(let r=0;r<rows.length;r++){grid[r]=grid[r]||[];let c=0;
   for(const cell of rows[r].cells){while(grid[r][c])c++;
    for(let i=0;i<cell.rowSpan;i++){grid[r+i]=grid[r+i]||[];for(let k=0;k<cell.colSpan;k++)grid[r+i][c+k]=cell;}
    if(cell===th)return cell.colSpan===1?c:-1;c+=cell.colSpan;}}
  return -1;
 }
 function cellAt(row,col){let c=0;for(const cell of row.cells){if(col<c+cell.colSpan)return cell.colSpan===1?cell:null;c+=cell.colSpan;}return null;}
 function text(td){if(!td)return"";if(td.dataset.sort!=null)return td.dataset.sort;const s=td.querySelector("summary");return(s||td).textContent;}
 function sortBy(th){
  const table=th.closest("table"),col=column(table,th);if(col<0)return;
  const dir=table.dataset.sortCol===String(col)&&table.dataset.sortDir==="1"?-1:1;
  table.dataset.sortCol=String(col);table.dataset.sortDir=String(dir);
  const heads=new Set(headRows(table));
  for(const tb of table.tBodies){
   const rows=[...tb.rows].filter(r=>!heads.has(r));if(!rows.length)continue;
   const fixed=rows.map(r=>r.classList.contains("total")||(r.cells.length===1&&r.cells[0].colSpan>1));
   const idx=order(rows.map(r=>key(text(cellAt(r,col)))),fixed,dir);
   const frag=document.createDocumentFragment();for(const i of idx)frag.appendChild(rows[i]);tb.appendChild(frag);
  }
  for(const r of heads)for(const o of r.cells){o.removeAttribute("aria-sort");const s=o.querySelector(".sortind");if(s)s.remove();}
  th.setAttribute("aria-sort",dir>0?"ascending":"descending");
  const s=document.createElement("span");s.className="sortind";s.textContent=dir>0?" \u25b2":" \u25bc";th.appendChild(s);
 }
 document.addEventListener("click",e=>{
  const th=e.target.closest?e.target.closest("th"):null;
  if(!th||e.target.closest("a,button,select,input,summary"))return;
  const table=th.closest("table");if(table&&headRows(table).includes(th.parentElement))sortBy(th);
 });
 return{key,order,sortBy};
})();
const D=__DATA__;
const TRAJ=D.TRAJ, CELLS=D.CELLS, S=D.S, S8PC=D.S8PC, T=S.target;
function esc(s){return String(s==null?"":s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&#39;");}
function num(v,d){return (v==null||!isFinite(v))?"\u2013":Number(v).toFixed(d==null?2:d);}
function sgn(v,d){return (v==null||!isFinite(v))?"\u2013":(v>=0?"+":"")+Number(v).toFixed(d==null?2:d);}
const B=S.bundle||{};
const fmtn=v=>v!=null?Number(v).toLocaleString():"n/a";
document.getElementById("tbody_traj").innerHTML=TRAJ.map(r=>`<tr><td class="c">${esc(r.round)}</td><td>${num(r.model)}</td><td>${r.model!=null?sgn(r.model-T):"\u2013"}</td><td>${r.n_gemms!=null?Number(r.n_gemms).toLocaleString():""}</td><td>${r.n_miss!=null?Number(r.n_miss).toLocaleString():""}</td></tr>`).join("")
 +(B.sel_eff!=null?`<tr class="total" style="font-weight:600"><td class="c">${esc(B.name)} (${esc(B.source)})</td><td>${num(B.sel_eff)}</td><td>${sgn(B.sel_eff-T)}</td><td>${fmtn(B.n)}</td><td>${fmtn(B.n_miss)}</td></tr>`:"");
const COLS=[["cell","cell"],["n_gemms","GEMMs"],["held","held-out %"],["gap","vs target"],["model","model %"],["deployed","deployed %"],["k_delta","K\u0394"],["smartK","K"],["cap","ed/hd/ih"]];
const STR=k=>(k==="cell"||k==="cap");
function rowClass(c){if(c.held==null)return"";if(c.held>=T)return"win";if(c.held<T-5)return"lose";return"weak";}
function cellText(c,k){const v=c[k];if(STR(k))return esc(v);if(k==="n_gemms")return Number(v||0).toLocaleString();if(k==="smartK")return v!=null?esc(v):"";if(k==="held")return v!=null?num(v)+(c.held_inherited?"*":""):"";if(k==="gap"||k.endsWith("delta"))return v!=null?sgn(v):"";return v!=null?num(v):"";}
CELLS.sort((a,b)=>(a.held==null)-(b.held==null)||(a.held??0)-(b.held??0));
document.getElementById("tbody_cells").innerHTML=CELLS.map((c,i)=>`<tr class="${rowClass(c)}"><td class="c">${i+1}</td>`+COLS.map(([k])=>`<td class="${STR(k)?"c":""}">${cellText(c,k)}</td>`).join("")+"</tr>").join("");
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
    ap.add_argument(
        "--audit-results",
        action="append",
        default=[],
        metavar="JSON",
        help="scripts/audit_eval.py output (protocol v1, or v1 with v2), repeatable: "
        "its records of this run's bundles on the run's audit set and pick rule "
        "replace less complete ones of the same model file, and its other records "
        "are shown as extra models labelled with their audit set",
    )
    ap.add_argument(
        "--audit-label",
        action="append",
        default=[],
        metavar="NAME=LABEL",
        help="show the extra model NAME (its label in the report, or at least six "
        "leading hex digits of its model file's sha256) as LABEL; repeatable",
    )
    ap.add_argument(
        "--weak-spots-dir",
        default=None,
        metavar="DIR",
        help="also write the Weak spots section's data there: pool_gaps.json and "
        "pool_gaps.csv (the training-side GEMMs whose true best is far below its "
        "roofline, with that kernel's signature, for kernel generation; never an "
        "audit shape) and weak_spots.json",
    )
    ap.add_argument(
        "--pool-gap-fraction",
        type=float,
        default=None,
        help="a GEMM whose true best reaches less than this fraction of its "
        "roofline is a pool gap (default: report.pool_gap_fraction of the run's "
        "config)",
    )
    ap.add_argument(
        "--peak-tflops",
        type=float,
        default=None,
        help="the roofline's peak matrix rate in TFLOP/s for the run's types "
        "(default: bench.adaptive_long.peak_tflops, else the device's, "
        "lib/hardware.py roofline_rates)",
    )
    ap.add_argument(
        "--peak-bandwidth-gbs",
        type=float,
        default=None,
        help="the roofline's memory bandwidth in GB/s (default: "
        "bench.adaptive_long.bandwidth_gbs, else the device's)",
    )
    a = ap.parse_args(argv)
    labels = {}
    for spec in a.audit_label:
        name, sep, label = spec.partition("=")
        if not sep or not name.strip() or not label.strip():
            ap.error(f"--audit-label {spec!r}: give NAME=LABEL")
        labels[name.strip()] = label.strip()
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
            audit_results=a.audit_results,
            audit_labels=labels,
            weak_spots_dir=a.weak_spots_dir,
            pool_gap_fraction=a.pool_gap_fraction,
            peak_tflops=a.peak_tflops,
            peak_bandwidth_gbs=a.peak_bandwidth_gbs,
        ),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
