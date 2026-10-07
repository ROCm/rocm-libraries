# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Analysis sections of the run report: where to improve (weak cells and the
action each needs), the per-cell loss decomposition, kernel and parameter
usage, pool-drop candidates, results per execution context and per-cell
selection time.

Every efficiency is relative to the fastest measured kernel of a GEMM and
every threshold is absolute. Weak cells, their ranking and their actions use
training-side evidence only (`lib.cell_evidence` and the training benches);
validation datasets (stage08) are shown separately and never enter them.
A kernel is its 8-field signature (`lib.run_analysis.SIG_FIELDS`); the
library solutions sharing one are listed as its members.
"""
import html
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent

GENERALIZATION_GAP = 0.03
DEFAULT_MIN_CELL_GEMMS = 50
DEFAULT_MIN_EVAL_GEMMS = 20
DEFAULT_SPLIT_AFTER = 2
ACTIONS = (
    "none",
    "more data",
    "retrain",
    "split",
    "tier policy: non-temporal rule",
    "tier policy: whitelist",
    "tier policy: cascade",
)
STEP_NAMES = {
    "library": "library pool",
    "hard_rules": "hard rules",
    "soft_rules": "non-temporal rule",
    "whitelist": "whitelist",
    "cascade": "cascade",
    "ranking": "ranking",
}
STEP_ACTION = {
    "soft_rules": "tier policy: non-temporal rule",
    "whitelist": "tier policy: whitelist",
    "cascade": "tier policy: cascade",
}


def pipeline_on_path():
    for p in (PIPELINE_ROOT, PIPELINE_ROOT / "stages"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def esc(s):
    return html.escape("" if s is None else str(s), quote=True)


def _ok(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def pct(v, digits=2):
    return f"{v * 100:.{digits}f}" if _ok(v) else "&ndash;"


def loss_pct(factor, digits=2):
    return f"{(1.0 - factor) * 100:.{digits}f}" if _ok(factor) else "&ndash;"


def intf(v):
    return f"{int(v):,}" if _ok(v) else "&ndash;"


def table(head, rows, cls="", style=""):
    ths = "".join(
        f'<th class="c">{h[1:]}</th>' if h.startswith("<") else f"<th>{h}</th>"
        for h in head
    )
    body = "".join(rows) or (
        f'<tr><td class="c" colspan="{len(head)}">nothing to show</td></tr>'
    )
    return (
        f'<div class="scroll{(" " + cls) if cls else ""}"><table style="margin-bottom:0;{style}">'
        f"<thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table></div>"
    )


# ── settings from the run's config ──────────────────────────────────────────


def run_settings(config_text):
    """Target and thresholds of the run's config source (defaults when it
    is missing or unreadable)."""
    pipeline_on_path()
    from lib.cell_evidence import absolute_target

    cfg = {}
    try:
        import yaml

        cfg = yaml.safe_load(config_text or "") or {}
    except Exception:
        cfg = {}
    cfg = cfg if isinstance(cfg, dict) else {}

    def get(path, default):
        cur = cfg
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def as_int(v, d):
        try:
            return int(v)
        except (TypeError, ValueError):
            return d

    return {
        "pick_measurement": pick_measurement(cfg),
        "target": absolute_target(get("validate.sel_eff_threshold", None)),
        "min_cell_gemms": as_int(
            get("train.min_cell_gemms", None), DEFAULT_MIN_CELL_GEMMS
        ),
        "min_eval_gemms": as_int(
            get("validate.min_cell_gemms", None), DEFAULT_MIN_EVAL_GEMMS
        ),
        "split_after": as_int(
            get("validate.split_after_attempts", None), DEFAULT_SPLIT_AFTER
        ),
    }


def pick_measurement(cfg):
    """validate.pick_measurement of a parsed config, with the orchestrator's
    schema default."""
    pipeline_on_path()
    from lib import evaluate as ev
    from lib.run_analysis import CONFIG_PICK_MEASUREMENT

    try:
        from run_orchestrator import cfg_get

        v = cfg_get(cfg, "validate.pick_measurement")
    except Exception:
        v = ((cfg.get("validate") or {}) if isinstance(cfg, dict) else {}).get(
            "pick_measurement"
        )
    return v if v in ev.PICK_MEASUREMENTS else CONFIG_PICK_MEASUREMENT


# ── where to improve ────────────────────────────────────────────────────────


def recommend(ev, decomp, settings):
    """(action, reason) for one leaf from its training-side evidence and
    its loss decomposition on the training benches."""
    target = settings["target"]
    if ev.sel_eff is not None and ev.sel_eff >= target:
        return "none", "meets the target"
    if not ev.has_model:
        if ev.n_gemms < settings["min_cell_gemms"]:
            return (
                "more data",
                f"no model of its own: {ev.n_gemms} training GEMMs < "
                f"train.min_cell_gemms {settings['min_cell_gemms']}",
            )
        return "retrain", "no model of its own although it has enough GEMMs"
    if ev.sel_eff is None:
        return "more data", "no evidence for this cell yet"
    if ev.n_eval < settings["min_eval_gemms"]:
        return (
            "more data",
            f"{ev.n_eval} evaluated GEMMs < validate.min_cell_gemms "
            f"{settings['min_eval_gemms']}",
        )
    steps = (decomp or {}).get("steps") or {}
    losses = {
        s: 1.0 - v
        for s, v in steps.items()
        if _ok(v) and s not in ("library", "hard_rules")
    }
    dominant = max(losses, key=losses.get) if losses else None
    if dominant in STEP_ACTION and losses[dominant] > 0:
        return (
            STEP_ACTION[dominant],
            f"largest training-bench loss is the {STEP_NAMES[dominant]} "
            f"step ({losses[dominant] * 100:.1f}%)",
        )
    if ev.in_sample is not None and ev.in_sample - ev.sel_eff > GENERALIZATION_GAP:
        return (
            "more data",
            f"in-sample {ev.in_sample * 100:.1f}% vs {ev.source} "
            f"{ev.sel_eff * 100:.1f}%: the model does not generalize",
        )
    if (
        ev.retrain_attempts >= settings["split_after"]
        and ev.n_gemms >= 2 * settings["min_cell_gemms"]
    ):
        return (
            "split",
            f"{ev.retrain_attempts} retrain attempts did not reach the target",
        )
    return "retrain", "ranking loss dominates"


def improve_rows(evidence, decomp, settings):
    rows = []
    for leaf, ev in evidence.items():
        action, why = recommend(ev, (decomp or {}).get(leaf), settings)
        d = (decomp or {}).get(leaf) or {}
        rows.append(
            {
                "cell": leaf,
                "ev": ev,
                "gain": ev.gain(settings["target"]),
                "action": action,
                "why": why,
                "decomp": d,
            }
        )
    rows.sort(
        key=lambda r: (
            r["action"] == "none",
            r["ev"].sel_eff is None,
            -r["gain"],
            -r["ev"].share,
            r["ev"].sel_eff if r["ev"].sel_eff is not None else 2.0,
            r["cell"],
        )
    )
    return rows


def improve_html(rows, settings, selections):
    if not rows:
        return '<p class="mut">no training evidence in this run.</p>'
    target = settings["target"]
    n_below = sum(1 for r in rows if r["action"] != "none")
    total_gain = sum(r["gain"] for r in rows)
    trs = []
    for i, r in enumerate(rows, 1):
        ev, d = r["ev"], r["decomp"]
        steps = d.get("steps") or {}
        dom = d.get("dominant")
        cls = (
            "win"
            if r["action"] == "none"
            else ("lose" if r["gain"] > 0.002 else "weak")
        )
        gain_txt = f"{r['gain'] * 100:.3f}" if ev.sel_eff is not None else "&ndash;"
        trs.append(
            f'<tr class="{cls}"><td class="c">{i}</td><td class="c">{esc(r["cell"])}</td>'
            f"<td>{intf(ev.n_gemms)}</td><td>{ev.share * 100:.1f}%</td>"
            f"<td>{pct(ev.sel_eff)}</td><td class=\"c\">{esc(ev.source or '')}"
            f"{(' ' + esc(ev.round)) if ev.round else ''}</td><td>{intf(ev.n_eval)}</td>"
            f"<td>{pct(ev.in_sample)}</td>"
            f"<td>{pct(max(0.0, target - ev.sel_eff)) if ev.sel_eff is not None else '&ndash;'}</td>"
            f"<td><b>{gain_txt}</b></td>"
            f"<td>{loss_pct(steps.get('ranking'))}</td>"
            f"<td class=\"c\">{esc(STEP_NAMES.get(dom, dom or ''))}</td>"
            f"<td>{intf(ev.retrain_attempts)}</td>"
            f'<td class="c"><b>{esc(r["action"])}</b></td><td class="c">{esc(r["why"])}</td></tr>'
        )
    sel = "".join(
        f"<li><code>{esc(flag)}</code>: {esc(', '.join(cells) or 'no cell')}</li>"
        for flag, cells in selections
    )
    return (
        f'<p class="cap">Target {target * 100:.1f}% of the fastest measured kernel '
        f"(geomean per cell). {n_below} of {len(rows)} leaf cells need an action; reaching "
        f"the target everywhere would add about {total_gain * 100:.2f} points to the "
        "GEMM-weighted geomean. <b>evidence</b> is the newest training-side measurement of "
        "the model that serves the cell: held_out = the previous round's model on this "
        "round's new training shapes (stage04b), validation = stage05's model-selection "
        "split, in_sample = its own training GEMMs. <b>expected gain</b> = GEMM share "
        "&times; (target &minus; evidence), in points. Ranked by expected gain; cells "
        "without evidence (no model of their own and no held-out row) follow, by GEMM "
        "share. The loss columns come from the decomposition below (training benches). "
        "Validation datasets are never used here.</p>"
        + table(
            [
                "<#",
                "<cell",
                "GEMMs",
                "share",
                "evidence %",
                "<source",
                "n eval",
                "in-sample %",
                "gap pp",
                "gain pp",
                "ranking loss %",
                "<largest loss",
                "retrains",
                "<action",
                "<why",
            ],
            trs,
            style="font-size:11px",
        )
        + '<p class="cap" style="margin-top:8px">Cells the orchestrator\'s enhance mode '
        "would select (<code>--weakest</code> takes cells below the target; enhance "
        "reads the complete rounds only):</p>"
        f'<ul class="cap">{sel}</ul>'
    )


# ── loss decomposition ──────────────────────────────────────────────────────

_STEP_COLORS = {
    "library": "#57606a",
    "hard_rules": "#8250df",
    "soft_rules": "#bf3989",
    "whitelist": "#bc4c00",
    "cascade": "#9a6700",
    "ranking": "#cf222e",
}


def _bar(steps, scale):
    parts = []
    for s, color in _STEP_COLORS.items():
        v = steps.get(s)
        if not _ok(v) or v >= 1.0:
            continue
        w = min(100.0, (1.0 - v) / scale * 100.0)
        parts.append(
            f'<span style="display:inline-block;height:10px;width:{w:.2f}%;background:{color}" '
            f'title="{esc(s)} {(1.0 - v) * 100:.2f}%"></span>'
        )
    return f'<div style="width:220px;background:#f6f8fa;line-height:0">{"".join(parts)}</div>'


def decomposition_html(analysis, title_note="", order=None):
    pipeline_on_path()
    from lib.run_analysis import STEPS, STEP_TITLES

    dec = analysis.get("decomposition") or {}
    if not dec:
        return '<p class="mut">no benches to decompose.</p>'
    engine = analysis.get("engine")
    cells = [c for c in dec if c != "*"]
    cells.sort(key=order or (lambda c: c))
    worst = 0.0
    for c in cells:
        se = dec[c].get("sel_eff")
        if _ok(se):
            worst = max(worst, 1.0 - se)
    scale = max(worst, 0.01)

    def row(label, d, bold=False):
        steps = d.get("steps") or {}
        blocked = d.get("blocked") or {}
        unserved = sum(blocked.values())
        why = ", ".join(f"{esc(k)}: {v}" for k, v in sorted(blocked.items()))
        tds = "".join(f"<td>{loss_pct(steps.get(s))}</td>" for s in STEPS)
        b = ' style="font-weight:600"' if bold else ""
        return (
            f'<tr{b}><td class="c">{esc(label)}</td><td>{intf(d.get("n_gemms"))}</td>'
            f'<td>{intf(d.get("n_served"))}</td><td>{intf(d.get("n_miss"))}</td>'
            f'<td>{intf(unserved)}</td><td class="c">{why}</td>{tds}'
            f'<td><b>{pct(d.get("sel_eff"))}</b></td><td>{pct(d.get("deployed_sel_eff"))}</td>'
            f"<td>{_bar(steps, scale)}</td></tr>"
        )

    rows = [row("all GEMMs", dec.get("*") or {}, True)] + [
        row(c, dec[c]) for c in cells
    ]
    legend = " ".join(
        f'<span style="display:inline-block;width:9px;height:9px;background:{col}"></span> '
        f"{esc(STEP_TITLES[s])}"
        for s, col in _STEP_COLORS.items()
    )
    note = (
        ""
        if engine
        else " The tilewright module or the model file is unavailable: the cascade and "
        "ranking steps are not decomposed and the library / hard / non-temporal / tier "
        "columns are ceilings over every GEMM that reaches them."
    )
    pm = analysis.get("pick_measurement")
    return (
        '<p class="cap">Per leaf cell, the efficiency each selection step loses on the GEMMs '
        "the model serves: <b>library pool</b> = best kernel of the serving library vs the "
        "fastest measured kernel (rows of other libraries, such as planner banks, can be the "
        "fastest but cannot be picked); <b>hard rules</b> = best kernel passing the LDS gate "
        "and the correctness rules; <b>non-temporal rule</b> = the optional cache-hint rule "
        "(when the model serves with it); <b>tier 1</b> = the cell's whitelist (all feasible "
        "kernels when the whitelist keeps none); <b>cascade</b> = the view keys a cascade "
        "model scores; <b>ranking</b> = the pick vs the best scored kernel. Losses multiply "
        "to the selection efficiency. Kernels cost what the stages charge them under "
        f"<code>validate.pick_measurement: {esc(pm)}</code>"
        + {
            "timed": " (a pick with only a warm-up row, or off the shape's planned kernel "
            "list, is a <b>miss</b> at the slowest latency the GEMM measured)",
            "warmup": " (a warm-up row counts with its warm-up latency; a pick off the "
            "shape's planned kernel list is a <b>miss</b> at the slowest latency the GEMM "
            "measured)",
        }.get(
            pm,
            " (a warm-up row counts with its warm-up latency; kernels off the planned "
            "list are passed over)",
        )
        + ". <b>unserved</b> GEMMs have no pick (reason: the first empty level; tier = no "
        "serving cell); <b>deployed %</b> also counts them with the runtime's fallback pick "
        f"when it was measured, as stages 4.5, 6 and 8 do.{note}"
        f'{title_note}</p><p class="cap">{legend}</p>'
        + table(
            ["<cell", "GEMMs", "served", "misses", "unserved", "<unserved at"]
            + [f"{esc(STEP_TITLES[s])} loss %" for s in STEPS]
            + ["sel_eff %", "deployed %", "<losses (bar = worst cell)"],
            rows,
            style="font-size:11px",
        )
    )


# ── kernel and parameter usage ──────────────────────────────────────────────


def _cells_txt(pairs):
    return ", ".join(f"{c} ({n})" for c, n in pairs)


def _sig_tds(k):
    return (
        f'<td class="c">{esc(k["mt"])}</td><td>{k["depth_u"]}</td>'
        f'<td class="c">{esc(k["mi"])}</td><td class="c">{esc(k["ch"])}</td>'
    )


_SIG_HEAD = ["<MT (M x N)", "depthU", "<MI", "<cache hints (A,B)"]


def members_html(k, engine=True):
    """The library solutions of one signature, as a collapsed detail."""
    mem = k.get("members") or []
    txt = ", ".join(
        (
            f'{m["sid"]} ({m["wins"]}/{m["picks"]})'
            if engine
            else f'{m["sid"]} ({m["wins"]})'
        )
        for m in mem
    )
    return (
        f"<details><summary>{len(mem)}</summary>"
        f'<span class="mut">{esc(txt)}</span></details>'
    )


def kernels_html(analysis):
    ks = (analysis.get("kernels") or {}).get("kernels") or []
    if not ks:
        return '<p class="mut">no kernels.</p>'
    engine = analysis.get("engine")
    used = [k for k in ks if k["wins"] or k["picks"]]
    used.sort(key=lambda k: (-(k["wins"] + (k["picks"] or 0)), k["signature"]))
    n = analysis["kernels"]["n_gemms"] or 1
    ns = analysis["kernels"]["n_served"] or 1
    rows = []
    for k in used:
        rows.append(
            "<tr>"
            + _sig_tds(k)
            + f'<td style="text-align:left">{members_html(k, engine)}</td>'
            f'<td>{k["wins"]:,}</td><td>{k["wins"] / n * 100:.1f}%</td>'
            f'<td>{intf(k["picks"])}</td><td>{(k["picks"] or 0) / ns * 100:.1f}%</td>'
            f'<td>{k["tested"]:,}</td><td>{k["screened"]:,}</td>'
            + "".join(f"<td>{v:,}</td>" for v in k["near"])
            + f'<td>{k["drop_loss"] * 100:.3f}</td><td>{k["whitelisted_cells"]}</td>'
            f'<td class="c">{esc(_cells_txt(k["win_cells"]))}</td>'
            f'<td class="c">{esc(_cells_txt(k["pick_cells"]))}</td></tr>'
        )
    pool = analysis.get("pool") or {}
    outside = (analysis.get("kernels") or {}).get("n_winner_outside_pool") or 0
    return (
        f'<p class="cap">A kernel is its signature (macro tile, depthU, matrix instruction, '
        f"cache hints); the library solutions sharing one are its members. {len(used)} of "
        f'{pool.get("signatures", len(ks))} signatures ({intf(pool.get("size"))} solutions of the '
        f'{esc(pool.get("source"))} pool) win at least one training GEMM or are picked for one. '
        "wins = GEMMs where a member is the fastest tested; picks = deployed picks"
        f"{'' if engine else ' (n/a: no engine)'}; tested / screened = GEMMs where a member has a "
        "full measurement / only warm-ups; within x% = tested GEMMs where a member is no more "
        "than x% slower than the fastest; drop loss = geomean loss of the oracle over all "
        "GEMMs if every member were removed; whitelisted = cells whose whitelist holds the "
        "signature; solutions = members as solution index (wins/picks)."
        + (
            f" {outside:,} GEMMs are won by a kernel outside the pool and count for no "
            "signature."
            if outside
            else ""
        )
        + "</p>"
        + table(
            _SIG_HEAD
            + [
                "<solutions",
                "wins",
                "win share",
                "picks",
                "pick share",
                "tested",
                "screened",
                "within 1%",
                "within 2%",
                "within 5%",
                "drop loss %",
                "whitelisted",
                "<top winning cells",
                "<top picked cells",
            ],
            rows,
            style="font-size:11px",
        )
    )


def cell_usage_html(analysis):
    cu = analysis.get("cell_usage") or {}
    if not cu:
        return ""
    rows = []
    for c, u in cu.items():
        rows.append(
            f'<tr><td class="c">{esc(c)}</td><td>{u["n_winners"]}</td><td>{intf(u["n_picks"])}</td>'
            f'<td>{intf(u.get("pick_has_winner_signature"))}</td>'
            f'<td>{intf(u["pick_is_winner"])}</td>'
            f'<td class="c">{esc(", ".join(f"{s} ({n})" for s, n in u["top_winners"]))}</td>'
            f'<td class="c">{esc(", ".join(f"{s} ({n})" for s, n in u["top_picks"]))}</td></tr>'
        )
    return "<h3>Kernels per cell</h3>" + table(
        [
            "<cell",
            "distinct winning signatures",
            "distinct picked signatures",
            "pick has the winner's signature",
            "pick = winning solution",
            "<most frequent winners",
            "<most frequent picks",
        ],
        rows,
        style="font-size:11px",
    )


def parameters_html(prefix="param", counted="signature"):
    """Selector and tables the report's script fills for one parameter set
    (`prefix` names the element ids)."""
    what = "signatures" if counted == "signature" else "solutions"
    return (
        '<p class="cap">Wins and picks per value of one kernel parameter. kernels = '
        f"{what} with that value; drop loss = geomean loss of the oracle over all GEMMs if "
        "every kernel with that value were removed; stranded = GEMMs whose only tested "
        f'kernels have that value. Choose a parameter: <select id="{prefix}_sel"></select></p>'
        f'<div id="{prefix}_tab"></div>'
        '<h3>Per cell</h3><p class="cap">wins / picks of the most frequent values in each cell '
        "for the parameter chosen above.</p>"
        f'<div id="{prefix}_cells"></div>'
    )


def diagnostics_html(analysis):
    """The optional framework-specific parameter breakdowns."""
    diag = (analysis.get("parameters") or {}).get("diagnostics")
    if not diag:
        return ""
    return (
        '<h2 id="diagnostics">Diagnostics: framework-specific kernel parameters</h2>'
        "<details><summary>show the breakdowns of parameters that are specific to the kernel "
        "framework (occupancy, vector widths, workgroup mapping, stagger and the other kernel "
        "attributes that vary across the pool; diagnostics only)</summary>"
        '<p class="cap">These parameters are not part of a kernel\'s identity: the analysis '
        "groups kernels by signature, and the solutions of one signature can differ here.</p>"
        + parameters_html("diag", diag.get("counted", "solution"))
        + "</details>"
    )


# ── execution contexts ──────────────────────────────────────────────────────

CONTEXT_SETS = (
    ("held_out", "held-out GEMMs, trained context"),
    ("holdout_contexts", "held-out context"),
    ("training", "training GEMMs (in-sample)"),
)


def context_sets(analysis):
    """{bench set: {context: per-leaf decomposition}} of an analysis."""
    a = analysis or {}
    out = {"training": a.get("contexts") or {}}
    for kind in ("held_out", "holdout_contexts"):
        out[kind] = (a.get(kind) or {}).get("contexts") or {}
    return out


def has_contexts(analysis, s4b_rows):
    labels = {c for d in context_sets(analysis).values() for c in d}
    return bool(s4b_rows) or bool(labels - {"exclusive"})


def weak_cells(decomp, settings):
    """Leaves of one context's decomposition below the target with at least
    validate.min_cell_gemms served GEMMs, largest GEMM-weighted gap first."""
    target, floor = settings["target"], settings["min_eval_gemms"]
    total = (decomp.get("*") or {}).get("n_served") or 0
    rows = []
    for cell, d in decomp.items():
        se, n = d.get("sel_eff"), d.get("n_served") or 0
        if cell == "*" or not _ok(se) or se >= target or n < floor:
            continue
        rows.append(
            {
                "cell": cell,
                "n": n,
                "n_miss": d.get("n_miss") or 0,
                "sel_eff": se,
                "gain": n / total * (target - se) if total else 0.0,
                "dominant": d.get("dominant"),
            }
        )
    rows.sort(key=lambda r: (-r["gain"], r["sel_eff"], r["cell"]))
    return rows


def contexts_html(analysis, s4b_rows, settings, names=None):
    """Selection efficiency per execution context: the pipeline's own
    held-out measurements (stage04b, per round) and the report's re-ranking
    under the shipped bundle, trained and held-out contexts separately, and
    the weak cells of each context."""
    names = names or {}
    target = settings["target"]

    def name(label):
        n = names.get(label)
        return f"{label} ({n})" if n and n != label else label

    parts = [
        '<p class="cap">Rows benched under an execution context (a CU budget, a tile '
        "schedule) are separate GEMMs ranked in that context. <b>trained</b> contexts feed "
        "training (their new shapes are held out like any other the round they arrive); "
        "<b>held-out</b> contexts are benched and measured but never trained on. Selection "
        "efficiency is against the fastest measured kernel of the same context.</p>"
    ]
    if s4b_rows:
        trs = "".join(
            f'<tr><td class="c">{esc(r["round"])}</td><td class="c">{esc(r["kind"])}</td>'
            f'<td class="c">{esc(name(r["context"]))}</td><td>{intf(r["n"])}</td>'
            f'<td>{intf(r["n_miss"])}</td><td><b>{pct(r["sel_eff"])}</b></td>'
            f'<td>{_fmt((r["sel_eff"] - target) * 100 if _ok(r["sel_eff"]) else None, True)}</td></tr>'
            for r in s4b_rows
        )
        parts.append(
            "<h3>Held-out measurement per round (stage04b)</h3>"
            "<p class=\"cap\">The previous round's model on the round's new GEMMs per trained "
            "context (aggregate.by_context) and on the held-out contexts "
            "(aggregate.context_holdout).</p>"
            + table(
                [
                    "<round",
                    "<kind",
                    "<context",
                    "GEMMs",
                    "misses",
                    "sel_eff %",
                    "vs target pp",
                ],
                [trs],
            )
        )
    sets = context_sets(analysis)
    trs = []
    for kind, title in CONTEXT_SETS:
        for label, dec in sorted(sets[kind].items()):
            d = dec.get("*") or {}
            weak = weak_cells(dec, settings)
            se = d.get("sel_eff")
            trs.append(
                f'<tr class="{"lose" if _ok(se) and se < target else ""}">'
                f'<td class="c">{esc(name(label))}</td><td class="c">{esc(title)}</td>'
                f'<td>{intf(d.get("n_gemms"))}</td><td>{intf(d.get("n_served"))}</td>'
                f'<td>{intf(d.get("n_miss"))}</td><td><b>{pct(se)}</b></td>'
                f'<td>{pct(d.get("deployed_sel_eff"))}</td>'
                f"<td>{_fmt((se - target) * 100 if _ok(se) else None, True)}</td>"
                f"<td>{len(weak)}</td></tr>"
            )
    if trs:
        parts.append(
            "<h3>Shipped bundle per context (the report's re-ranking)</h3>"
            + table(
                [
                    "<context",
                    "<bench set",
                    "GEMMs",
                    "served",
                    "misses",
                    "sel_eff %",
                    "deployed %",
                    "vs target pp",
                    "weak cells",
                ],
                trs,
            )
        )
    blocks = []
    for kind, title in CONTEXT_SETS:
        for label, dec in sorted(sets[kind].items()):
            weak = weak_cells(dec, settings)
            if not weak:
                continue
            rows = [
                f'<tr><td class="c">{esc(r["cell"])}</td><td>{r["n"]:,}</td>'
                f'<td>{r["n_miss"]:,}</td><td>{pct(r["sel_eff"])}</td>'
                f'<td>{_fmt((r["sel_eff"] - target) * 100, True)}</td>'
                f'<td>{r["gain"] * 100:.3f}</td>'
                f'<td class="c">{esc(STEP_NAMES.get(r["dominant"], r["dominant"] or ""))}</td></tr>'
                for r in weak
            ]
            blocks.append(
                f"<details><summary>{esc(name(label))}, {esc(title)}: {len(weak)} weak "
                "cells</summary>"
                + table(
                    [
                        "<cell",
                        "served GEMMs",
                        "misses",
                        "sel_eff %",
                        "vs target pp",
                        "gain pp",
                        "<largest loss",
                    ],
                    rows,
                    style="font-size:11px",
                )
                + "</details>"
            )
    if blocks:
        parts.append(
            "<h3>Weak cells per context</h3>"
            f'<p class="cap">Leaf cells below the {target * 100:.1f}% target with at least '
            f'{settings["min_eval_gemms"]} served GEMMs in that context; gain = share of the '
            "context's served GEMMs &times; gap. Weak cells of held-out contexts are not chosen "
            "for training.</p>" + "".join(blocks)
        )
    return "".join(parts)


# ── pool-drop candidates ────────────────────────────────────────────────────


def pruning_html(analysis):
    pr = analysis.get("pruning") or {}
    if not pr:
        return '<p class="mut">no kernels.</p>'
    never = pr.get("never") or []
    unmeasured = pr.get("unmeasured") or []
    rows = []
    picks_known = pr.get("picks_known")
    for k in never:
        gap = (k["best_ratio"] - 1) * 100 if k["best_ratio"] is not None else None
        rows.append(
            "<tr>"
            + _sig_tds(k)
            + f'<td style="text-align:left">{members_html(k, picks_known)}</td>'
            f'<td>{k["tested"]:,}</td><td>{k["screened"]:,}</td>'
            + "".join(f"<td>{v:,}</td>" for v in k["near"])
            + f"<td>{_fmt(gap)}</td><td>{k['whitelisted_cells']}</td></tr>"
        )
    picks_note = (
        "" if picks_known else " Picks are unknown (no engine): only wins count."
    )
    um = "; ".join(
        f'{k["sig"]} ({", ".join(str(m["sid"]) for m in k.get("members") or [])})'
        for k in unmeasured[:200]
    )
    n_sol = sum(k["n_members"] for k in never)
    return (
        f'<p class="cap">{len(never)} of {pr.get("n_pool", 0)} kernel signatures '
        f"({n_sol:,} of {intf(pr.get('n_solutions'))} library solutions) never win a training "
        "GEMM and are never picked: removing every member changes neither the oracle nor a "
        "deployed pick on these GEMMs, and the model is not retrained for it (it scores "
        "kernels by their features). Safest first: fewest tested near-ties (within 1/2/5% of "
        "the fastest), then the largest best-case gap. A whitelisted signature also leaves "
        f"that cell's tier 1 when dropped.{picks_note}</p>"
        + table(
            _SIG_HEAD
            + [
                "<solutions",
                "tested",
                "screened",
                "within 1%",
                "within 2%",
                "within 5%",
                "best gap %",
                "whitelisted",
            ],
            rows,
            style="font-size:11px",
        )
        + (
            f'<p class="cap" style="margin-top:8px">{len(unmeasured)} kernel signatures have '
            "no measurement on any training GEMM (no evidence either way; solution indices in "
            "brackets): "
            f"{esc(um)}{' &hellip;' if len(unmeasured) > 200 else ''}</p>"
            if unmeasured
            else ""
        )
    )


# ── stage 7 per cell ────────────────────────────────────────────────────────


def stage07_per_cell(stage07_dir, summary, labels=None, tree=None):
    """Per bench yaml and request size: per cell, the per-problem medians
    over repetitions of the ML-on and ML-off selection times and their
    difference (first call of each run excluded, as stage07 does), and the
    ML-on medians split into the first call of a run into the cell (`first`)
    and the calls after it (`warm`). A problem's cell is the runtime's
    serving cell from the parity run, else the leaf it routes to."""
    pipeline_on_path()
    try:
        import stage07_optional_validate_selection_time as s7
        from lib.grid import cell_key
        from lib.subcells import assign_subcell, resolve_model_cell
    except Exception:
        return []
    out = []
    for tag, y in sorted((summary.get("per_yaml") or {}).items()):
        if not isinstance(y, dict):
            continue
        base = Path(stage07_dir) / str(tag)
        if base.resolve().parent != Path(stage07_dir).resolve():
            continue
        cell_of = {}
        par = _read_json(base / "parity" / "parity.json") or {}
        for r in par.get("per_problem") or []:
            try:
                key = (
                    int(r["m"]),
                    int(r["n"]),
                    int(r["k"]),
                    int(r["batch"]),
                    str(r["transA"]),
                    str(r["transB"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            c = r.get("cpp_cell") or r.get("py_cell")
            if c:
                cell_of[key] = str(c)
        for rsn_dir in sorted((base / "timing").glob("rsn*")):
            reps = []
            i = 0
            while True:
                on, off = rsn_dir / f"rep{i}_ml_on.log", rsn_dir / f"rep{i}_ml_off.log"
                if not on.is_file() and not off.is_file():
                    break
                if on.is_file() and off.is_file():
                    reps.append((s7.parse_bench_log(on), s7.parse_bench_log(off)))
                i += 1
            if not reps:
                continue
            routed_keys = set()

            def cell_for(key):
                cell = (
                    cell_of.get(key)
                    if isinstance(key, tuple) and len(key) == 6
                    else None
                )
                if (
                    cell is None
                    and isinstance(key, tuple)
                    and len(key) == 6
                    and isinstance(key[0], int)
                ):
                    m, n, kk, b = key[:4]
                    leaf = assign_subcell(
                        cell_key(m, n, kk, b), m, n, kk, b, dict(tree or {})
                    )
                    cell = resolve_model_cell(leaf, set(labels)) if labels else leaf
                    cell = cell or leaf
                    routed_keys.add(key)
                return cell or "(unidentified)"

            on_t, off_t = defaultdict(list), defaultdict(list)
            first_in_cell = set()
            for on, off in reps:
                on_keys, off_keys = s7._occurrence_keys(on.calls), s7._occurrence_keys(
                    off.calls
                )
                on_map, off_map = dict(zip(on_keys, on.calls)), dict(
                    zip(off_keys, off.calls)
                )
                first_in_cell |= s7.first_calls_into_cells(
                    on.calls,
                    {c.key: cell_for(c.key) for c in on.calls if c.key is not None},
                )
                cold = set(on_keys[:1]) | set(off_keys[:1])
                for k in (set(on_map) & set(off_map)) - cold:
                    a, b = on_map[k].sel_us, off_map[k].sel_us
                    if math.isfinite(a) and math.isfinite(b):
                        on_t[k].append(a)
                        off_t[k].append(b)
            per_cell = defaultdict(
                lambda: {"on": [], "off": [], "d": [], "first": [], "warm": []}
            )
            for k in on_t:
                cell = cell_for(k[0])
                a, b = statistics.median(on_t[k]), statistics.median(off_t[k])
                per_cell[cell]["on"].append(a)
                per_cell[cell]["off"].append(b)
                per_cell[cell]["d"].append(a - b)
                per_cell[cell]["first" if k in first_in_cell else "warm"].append(a)
            routed = len(routed_keys & {k[0] for k in on_t})
            out.append(
                {
                    "yaml": str(tag),
                    "rsn": rsn_dir.name.replace("rsn", ""),
                    "reps": len(reps),
                    "routed": routed,
                    "cells": {c: dict(v) for c, v in per_cell.items()},
                }
            )
    return out


def _read_json(path):
    import json

    try:
        with open(path) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def _q(xs, q):
    s = sorted(xs)
    if not s:
        return None
    pos = (len(s) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def selection_time_html(per_cell, budget_us, origami_reference=False):
    """Per-cell ML-on selection times; the ML-off columns (hipBLASLt's own
    ranking) only with `origami_reference`."""
    if not per_cell:
        return '<p class="mut">no stage 7 timing logs in this run.</p>'
    parts = [
        f'<p class="cap">Per serving cell: per-problem medians over repetitions of hipBLASLt\'s '
        f"solution selection time with tilewright (ML-on), first call of each run (library "
        f"load) excluded. <b>first into cell</b> = the first call of a run into that cell "
        f"(the cell's model is touched for the first time), <b>warm</b> = the calls after it. "
        f"over budget = problems whose ML-on time exceeds {budget_us:g} &micro;s. The cell is "
        "the runtime's serving cell from the parity run (or the leaf the problem routes to "
        "when the parity run has no line for it)."
        + (
            " [reference] ML-off = the same calls without tilewright; &Delta; = ML-on minus "
            "ML-off."
            if origami_reference
            else ""
        )
        + "</p>"
    ]
    for t in per_cell:
        rows = []
        all_on = [v for c in t["cells"].values() for v in c["on"]]
        first_all = [v for c in t["cells"].values() for v in c.get("first") or []]
        warm_all = [v for c in t["cells"].values() for v in c.get("warm") or []]
        for cell in sorted(
            t["cells"], key=lambda c: -(_q(t["cells"][c]["on"], 90) or 0)
        ):
            c = t["cells"][cell]
            over = sum(1 for v in c["on"] if v > budget_us)
            first, warm = c.get("first") or [], c.get("warm") or []
            ref = (
                f'<td>{_fmt(_q(c["off"], 50))}</td><td>{_fmt(_q(c["d"], 50), True)}</td>'
                f'<td>{_fmt(_q(c["d"], 90), True)}</td>'
                if origami_reference
                else ""
            )
            rows.append(
                f'<tr class="{"lose" if over else ""}"><td class="c">{esc(cell)}</td><td>{len(c["on"])}</td>'
                f'<td>{_fmt(_q(c["on"], 50))}</td><td>{_fmt(_q(c["on"], 90))}</td><td>{_fmt(max(c["on"]))}</td>'
                f"<td>{_fmt(_q(first, 50))}</td><td>{len(first)}</td>"
                f"<td>{_fmt(_q(warm, 50))}</td><td>{len(warm)}</td>"
                f"{ref}<td>{over}</td></tr>"
            )
        over_all = sum(1 for v in all_on if v > budget_us)
        parts.append(
            f'<h3>{esc(t["yaml"])}, rsn={esc(t["rsn"])} <span class="mut">({t["reps"]} repetitions, '
            f"{len(all_on)} problems, ML-on p50 {_fmt(_q(all_on, 50))} / max {_fmt(max(all_on) if all_on else None)} "
            f"&micro;s; first into a cell p50 {_fmt(_q(first_all, 50))}, warm p50 "
            f"{_fmt(_q(warm_all, 50))} &micro;s; {over_all} over budget)</span></h3>"
            + table(
                [
                    "<cell",
                    "problems",
                    "ML-on p50 &micro;s",
                    "ML-on p90",
                    "ML-on max",
                    "first into cell p50",
                    "first calls",
                    "warm p50",
                    "warm calls",
                ]
                + (
                    ["[reference] ML-off p50", "&Delta; p50", "&Delta; p90"]
                    if origami_reference
                    else []
                )
                + ["over budget"],
                rows,
                style="font-size:11px",
            )
        )
    return "".join(parts)


def _fmt(v, signed=False):
    if not _ok(v):
        return "&ndash;"
    return f"{v:+.2f}" if signed else f"{v:.2f}"


# ── stage 8 (validation; report only) ───────────────────────────────────────


def stage08_model_rows(stage08_dir, summary):
    """Per dataset of a stage08 summary: the model's own picks against the
    fastest measured kernel, per leaf, from its per_gemm.csv."""
    pipeline_on_path()
    from lib.run_analysis import read_per_gemm

    out = []
    for ds in summary.get("datasets") or []:
        if not isinstance(ds, dict):
            continue
        pcj = ds.get("per_cell_json")
        per_gemm = None
        if pcj:
            p = Path(str(pcj))
            if not p.is_absolute():
                p = Path(stage08_dir) / p
            per_gemm = p.parent / "per_gemm.csv"
        if per_gemm is None or not per_gemm.is_file():
            q = Path(stage08_dir) / str(ds.get("name") or "") / "per_gemm.csv"
            per_gemm = q if q.is_file() else None
        cells = read_per_gemm(per_gemm) if per_gemm is not None else {}
        logs = [x for c in cells.values() for x in c.pop("_logs", [])]
        s = ds.get("summary") if isinstance(ds.get("summary"), dict) else {}
        ov = s.get("overall") if isinstance(s.get("overall"), dict) else {}
        paired = s.get("paired") or ov.get("paired") or {}
        out.append(
            {
                "name": ds.get("name"),
                "rc": ds.get("rc"),
                "n_gemms": sum(c["n_gemms"] for c in cells.values()) or None,
                "n_served": len(logs) if cells else None,
                "n_miss": sum(c.get("n_miss", 0) for c in cells.values()),
                "sel_eff": math.exp(-sum(logs) / len(logs)) if logs else None,
                "deployed": _first_ok(s.get("model_sel_eff"), ov.get("sel_eff")),
                "n_evaluated": s.get("n_evaluated", ov.get("n_evaluated")),
                "n_miss_deployed": s.get("n_miss", ov.get("n_miss")),
                "pick_measurement": s.get("pick_measurement"),
                "paired": {
                    k: paired.get(k)
                    for k in (
                        "n",
                        "model_sel_eff",
                        "origami_sel_eff",
                        "wins",
                        "ties",
                        "losses",
                        "speedup_geomean",
                        "speedup_ci95",
                    )
                },
                "cells": cells,
                "dataset": ds,
            }
        )
    return out


def _first_ok(*vals):
    return next((v for v in vals if _ok(v)), None)
