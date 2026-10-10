# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Analysis sections of the run report: where to improve (weak cells and the
action each needs), the per-cell loss decomposition, kernel and parameter
usage, the kernel identity, pool-drop candidates, results per execution
context and per-cell selection time.

Every efficiency is relative to the fastest measured kernel of a GEMM and
every threshold is absolute. The action of a cell is chosen from held-out
evidence only: the audit set once it has scored the exclusive group of the
bundle (complete or partial), else the bundle's
measurements on GEMMs it never trained on (deploy gate, stage04b) and the
report's loss decomposition of those GEMMs; in-sample numbers appear as
context. Validation datasets (stage08) are shown separately and never enter
them. The analysis groups kernels by their 8-field signature
(`lib.run_analysis.SIG_FIELDS`) and names a kernel by it
(`lib.run_analysis.signature_label`), with the number of library solutions
that share it, never by a solution index: indices are positions in one
library build.
"""
import html
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

PIPELINE_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MIN_CELL_GEMMS = 50
DEFAULT_MIN_EVAL_GEMMS = 20
DEFAULT_SPLIT_AFTER = 2
NO_EVIDENCE = "no held-out evidence"
ACTIONS = (
    "none",
    "more data",
    "retrain",
    "split",
    "kernel pool",
    "kernel identity",
    "tier policy: serving rules",
    "tier policy: non-temporal rule",
    "tier policy: whitelist",
    "tier policy: cascade",
    NO_EVIDENCE,
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
    "library": "kernel pool",
    "soft_rules": "tier policy: non-temporal rule",
    "whitelist": "tier policy: whitelist",
    "cascade": "tier policy: cascade",
}
MODEL_STEPS = ("whitelist", "cascade", "ranking")


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


@dataclass
class HeldOut:
    """Held-out evidence of one leaf cell of the bundle. `sel_eff` and
    `n_eval` are what its action is chosen from, `source` says where they
    come from ("audit", "measured" or "re-ranking") and `detail` which
    measurement; `decomp` is the leaf's loss decomposition on GEMMs the
    bundle never trained on. `in_sample` and `train_decomp` (its training
    GEMMs) are context only."""

    cell: str
    n_gemms: int = 0
    share: float = 0.0
    has_model: bool = True
    sel_eff: Optional[float] = None
    n_eval: int = 0
    source: Optional[str] = None
    detail: str = ""
    audit: Optional[Mapping[str, Any]] = None
    decomp: Mapping[str, Any] = field(default_factory=dict)
    in_sample: Optional[float] = None
    train_decomp: Mapping[str, Any] = field(default_factory=dict)
    retrain_attempts: int = 0
    decision: Optional[str] = None
    decision_round: Optional[str] = None

    def gain(self, target: float) -> float:
        """GEMM share times the gap to `target`: roughly the points the
        GEMM-weighted geomean gains when the cell reaches the target."""
        if self.sel_eff is None:
            return 0.0
        return self.share * max(0.0, float(target) - self.sel_eff)


def nearest(label, table):
    """(entry, its label) of `label` or of its nearest ancestor in `table`;
    (None, None) when neither has one."""
    cur = label
    while True:
        if cur in table:
            return table[cur], cur
        j = cur.rfind("#")
        if j < 0:
            return None, None
        cur = cur[:j]


def heldout_evidence(
    leaves, held, decomp_held=None, decomp_train=None, audit_cells=None, history=None
) -> Dict[str, HeldOut]:
    """{leaf: HeldOut} of the bundle's leaves (`leaves`: {label: {"n_gemms",
    "in_sample", "has_model"}}), the evidence being the first of

      audit       the audit set's result of the leaf's base cell
                  (`audit_cells`: per_cell of a complete audit record);
      measured    `held` ({cell: the bundle's measurements on GEMMs it never
                  trained on, pooled}) of the leaf, else of its nearest
                  measured ancestor;
      re-ranking  the leaf's decomposition of those GEMMs (`decomp_held`).

    `history` holds each label's stage04b retrain count and latest decision
    ({label: {"retrains", "decision", "round"}})."""
    decomp_held, decomp_train = decomp_held or {}, decomp_train or {}
    total = sum(int(v.get("n_gemms") or 0) for v in leaves.values())
    out: Dict[str, HeldOut] = {}
    for leaf, info in leaves.items():
        n = int(info.get("n_gemms") or 0)
        h = (history or {}).get(leaf) or {}
        ev = HeldOut(
            cell=leaf,
            n_gemms=n,
            share=n / total if total else 0.0,
            has_model=bool(info.get("has_model", True)),
            decomp=decomp_held.get(leaf) or {},
            in_sample=info.get("in_sample") if _ok(info.get("in_sample")) else None,
            train_decomp=decomp_train.get(leaf) or {},
            retrain_attempts=int(h.get("retrains") or 0),
            decision=h.get("decision"),
            decision_round=h.get("round"),
        )
        base = leaf.split("#")[0]
        a = (audit_cells or {}).get(base)
        m, at = nearest(leaf, held or {})
        if a is not None and _ok(a.get("sel_eff")):
            ev.audit = a
            ev.sel_eff, ev.n_eval, ev.source = (
                a["sel_eff"],
                int(a.get("n") or 0),
                "audit",
            )
            ev.detail = f"base cell {base}"
        elif m is not None and _ok(m.get("sel_eff_new")):
            ev.sel_eff, ev.source = m["sel_eff_new"], "measured"
            ev.n_eval = int(m.get("n_eval") or 0)
            ev.detail = " + ".join(m.get("sources") or []) + (
                "" if at == leaf else f" (of {at})"
            )
        elif _ok(ev.decomp.get("sel_eff")):
            ev.sel_eff, ev.source = ev.decomp["sel_eff"], "re-ranking"
            ev.n_eval = int(ev.decomp.get("n_served") or 0)
        out[leaf] = ev
    return out


def _dominant(decomp, steps):
    s = (decomp or {}).get("steps") or {}
    losses = {k: 1.0 - v for k, v in s.items() if k in steps and _ok(v)}
    if not losses:
        return None, 0.0
    k = max(losses, key=losses.get)
    return (k, losses[k]) if losses[k] > 0 else (None, 0.0)


def recommend(ev, settings):
    """(action, reason) of one leaf from its held-out evidence (`HeldOut`):
    an audit cell that is pool-limited needs kernels (or looser serving
    rules when the library alone reaches the target), one that is
    identity-limited a finer kernel identity (launch keys or spec
    attributes) or a pool whose keys serve better kernels; otherwise the largest
    loss of the held-out decomposition picks a tier policy or the kernel
    pool, and a ranking loss a retrain, or a split after
    validate.split_after_attempts retrains."""
    target = settings["target"]
    if not ev.has_model and ev.n_gemms < settings["min_cell_gemms"]:
        return (
            "more data",
            f"no model of its own: {ev.n_gemms} training GEMMs < "
            f"train.min_cell_gemms {settings['min_cell_gemms']}",
        )
    if ev.sel_eff is None:
        return NO_EVIDENCE, "no measurement on GEMMs the bundle never trained on yet"
    tag = {"audit": "audit", "re-ranking": "held-out re-ranking"}.get(
        ev.source, "held-out"
    )
    head = f"{tag} {ev.sel_eff * 100:.1f}% over {ev.n_eval} GEMMs"
    if ev.sel_eff >= target:
        return "none", f"{head} meets the target"
    if not ev.has_model:
        return "retrain", f"{head}; no model of its own although it has enough GEMMs"
    a = ev.audit
    if a is not None and a.get("class") == "pool_limited":
        ceil, lib_only = a.get("ceiling_rules"), a.get("ceiling_table")
        if _ok(lib_only) and lib_only >= target:
            return (
                "tier policy: serving rules",
                f"{head}; pool-limited: the serving rules leave {pct(ceil, 1)}% "
                f"reachable, the library without them {pct(lib_only, 1)}%",
            )
        return (
            "kernel pool",
            f"{head}; pool-limited: the best kernel the serving rules allow "
            f"reaches {pct(ceil, 1)}%",
        )
    if a is not None and a.get("class") == "identity_limited":
        return (
            "kernel identity",
            f"{head}; identity-limited: perfect scoring of the keys as the "
            f"runtime serves them reaches {pct(a.get('ceiling_served'), 1)}%, "
            f"the best kernel the serving rules allow "
            f"{pct(a.get('ceiling_rules'), 1)}%",
        )
    if ev.source != "audit" and ev.n_eval < settings["min_eval_gemms"]:
        return (
            "more data",
            f"{head} < validate.min_cell_gemms {settings['min_eval_gemms']}",
        )
    steps = MODEL_STEPS if a is not None else tuple(STEP_ACTION) + ("ranking",)
    dom, loss = _dominant(ev.decomp, steps)
    if dom in STEP_ACTION:
        return (
            STEP_ACTION[dom],
            f"{head}; largest held-out loss an action can address: the "
            f"{STEP_NAMES[dom]} step ({loss * 100:.1f}%)",
        )
    why = head + (
        f"; largest held-out loss an action can address: ranking ({loss * 100:.1f}%)"
        if dom == "ranking"
        else ""
    )
    if (
        ev.retrain_attempts >= settings["split_after"]
        and ev.n_gemms >= 2 * settings["min_cell_gemms"]
    ):
        return (
            "split",
            f"{why}; {ev.retrain_attempts} retrain attempt(s) did not reach "
            "the target",
        )
    return "retrain", why


def improve_rows(evidence, settings):
    rows = []
    for leaf, ev in evidence.items():
        action, why = recommend(ev, settings)
        rows.append(
            {
                "cell": leaf,
                "ev": ev,
                "gain": ev.gain(settings["target"]),
                "action": action,
                "why": why,
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


EVIDENCE_TEXT = {
    "audit": "<b>audit</b> = the audit set's result of the leaf's base cell "
    "(exhaustive bench of every pool and bank kernel, never trained on, against "
    "the true best, with its pool-limited / identity-limited / model-limited "
    "class)",
    "measured": "<b>measured</b> = the bundle's own measurements on GEMMs it never "
    "trained on (its deploy gate and the next round's stage04b), pooled per cell; "
    "<i>of</i> a cell = the nearest measured ancestor",
    "re-ranking": "<b>re-ranking</b> = the report's re-ranking of those GEMMs (no "
    "measurement of the leaf itself)",
}


def improve_html(rows, settings, selections, bundle="bundle", audit_note=""):
    """The leaves of `bundle` with the action their held-out evidence asks
    for; `audit_note` says why the audit set is not the evidence."""
    if not rows:
        return (
            '<p class="mut">no leaf cells to rank: this run has no trained bundle '
            "yet.</p>"
        )
    target = settings["target"]
    n_below = sum(1 for r in rows if r["action"] not in ("none", NO_EVIDENCE))
    total_gain = sum(r["gain"] for r in rows)
    used = defaultdict(int)
    for r in rows:
        used[r["ev"].source] += 1
    trs = []
    for i, r in enumerate(rows, 1):
        ev = r["ev"]
        steps = (ev.decomp or {}).get("steps") or {}
        dom = (ev.decomp or {}).get("dominant")
        tdom = (ev.train_decomp or {}).get("dominant")
        cls = (
            "win"
            if r["action"] == "none"
            else (
                ""
                if r["action"] == NO_EVIDENCE
                else ("lose" if r["gain"] > 0.002 else "weak")
            )
        )
        gain_txt = f"{r['gain'] * 100:.3f}" if ev.sel_eff is not None else "&ndash;"
        audit_cls = (ev.audit or {}).get("class")
        trs.append(
            f'<tr class="{cls}"><td class="c">{i}</td><td class="c">{esc(r["cell"])}</td>'
            f"<td>{intf(ev.n_gemms)}</td><td>{ev.share * 100:.1f}%</td>"
            f"<td><b>{pct(ev.sel_eff)}</b></td>"
            f'<td class="c">{esc(ev.source or "none")}'
            f"{(': ' + esc(ev.detail)) if ev.detail else ''}</td>"
            f"<td>{intf(ev.n_eval) if ev.sel_eff is not None else '&ndash;'}</td>"
            f"<td>{pct(max(0.0, target - ev.sel_eff)) if ev.sel_eff is not None else '&ndash;'}</td>"
            f"<td><b>{gain_txt}</b></td>"
            f'<td class="c">{esc(STEP_NAMES.get(dom, dom or ""))}</td>'
            f"<td>{loss_pct(steps.get('ranking'))}</td>"
            f'<td class="c">{esc(audit_cls or "")}'
            f"{'' if not audit_cls or (ev.audit or {}).get('confident') else ' (?)'}</td>"
            f'<td class="c">{esc(ev.decision or "")}'
            f"{(' (' + esc(ev.decision_round) + ')') if ev.decision_round else ''}</td>"
            f"<td>{intf(ev.retrain_attempts)}</td>"
            f"<td>{pct(ev.in_sample)}</td>"
            f'<td class="c">{esc(STEP_NAMES.get(tdom, tdom or ""))}</td>'
            f'<td class="c"><b>{esc(r["action"])}</b></td><td class="c">{esc(r["why"])}</td></tr>'
        )
    sel = "".join(
        f"<li><code>{esc(flag)}</code>: {esc(', '.join(cells) or 'no cell')}</li>"
        for flag, cells in selections
    )
    counts = ", ".join(
        f"{used[k]} {k}" for k in ("audit", "measured", "re-ranking") if used[k]
    )
    if used[None]:
        counts += (", " if counts else "") + f"{used[None]} without held-out evidence"
    return (
        f'<p class="cap">Target {target * 100:.1f}% of the fastest measured kernel '
        f"in every cell. {n_below} of {len(rows)} leaf cells of the {esc(bundle)} need "
        f"an action; reaching the target everywhere would add about "
        f"{total_gain * 100:.2f} points to the GEMM-weighted geomean. Every action is "
        "chosen from held-out evidence; the <b>evidence</b> column says which, per row "
        f"({esc(counts or 'none')}): "
        + "; ".join(EVIDENCE_TEXT.values())
        + ". The audit set takes precedence once it is complete"
        + (f" ({esc(audit_note)})" if audit_note else "")
        + ". <b>largest held-out loss</b> and the held-out ranking loss come from the "
        "loss decomposition of the GEMMs the bundle never trained on (the action looks "
        "past the hard validity rules, which keep results correct); <b>stage04b "
        "decision</b> is the pipeline's latest decision on the cell, with its round. "
        "<b>expected "
        "gain</b> = GEMM share &times; (target &minus; evidence), in points; ranked by "
        "expected gain. <b>in-sample %</b> and the <b>training-bench</b> loss are "
        "context only (GEMMs the model trained on) and never choose the action. "
        "Validation datasets are never used here.</p>"
        + table(
            [
                "<#",
                "<cell",
                "GEMMs",
                "share",
                "evidence %",
                "<evidence",
                "n held-out",
                "gap pp",
                "gain pp",
                "<largest held-out loss",
                "held-out ranking loss %",
                "<audit class",
                "<stage04b decision",
                "retrains",
                "in-sample % (context)",
                "<training-bench largest loss (context)",
                "<action",
                "<why",
            ],
            trs,
            style="font-size:11px",
        )
        + '<p class="cap" style="margin-top:8px">Cells the orchestrator\'s enhance mode '
        "would select by its own evidence (stage04b's held-out rows, else stage05's "
        "validation split or in-sample numbers; <code>--weakest</code> takes cells "
        "below the target; enhance reads the complete rounds only):</p>"
        f'<ul class="cap">{sel}</ul>'
    )


# ── weak spots ──────────────────────────────────────────────────────────────

WEAK_POOL_GAPS_SHOWN = 50


def _ratio_txt(prev):
    if not prev or not _ok(prev.get("ratio")):
        return "&ndash;"
    txt = _ratio_cell(prev["ratio"], prev.get("ratio_ci"))
    if _ok(prev.get("p")):
        txt += f' <span class="mut">p {prev["p"]:.2g}</span>'
    if prev.get("moved"):
        txt += f' <span class="mut">{esc(prev["moved"])}</span>'
    return txt


def _count_or_na(v):
    """A count, or n/a where the record cannot give it."""
    return "n/a" if v is None else intf(v)


def weak_spots_html(
    rows,
    analysis,
    settings,
    *,
    bundle="bundle",
    previous=None,
    why=None,
    exported=None,
):
    """The Weak spots section (`lib.weak_spots`): per audit group the cells
    of each class, then every weak cell of the bundle (a class, a regression
    against the `previous` bundle or a pick below the tail floor) with its
    class, the lever that addresses it, its paired change from `previous`,
    its tail counts and its true best's roofline fraction; then the
    roofline's peaks and the pool gaps (`analysis`,
    `lib.weak_spots.analyze`), the GEMMs farthest from their roofline
    first. `settings`: threshold (pool gaps) and the gate's (`lib.audit
    .CELL_GATE_DEFAULTS`); `why` says why rows or the roofline are missing;
    `exported` names the files the pool gaps went to. A count a record
    cannot give is n/a, with why."""
    pipeline_on_path()
    from lib import weak_spots as ws

    thr = float(settings["threshold"])
    floor, tail = float(settings["cell_floor"]), float(settings["tail_floor"])
    alpha = float(settings.get("alpha", 0.05))
    tol = float(settings.get("cell_tolerance", 0.01))
    prev = esc(previous or "the previous bundle")
    cells_from = (analysis or {}).get("cells_source")
    held = bool((analysis or {}).get("cells_held_out"))
    parts = [
        '<p class="cap">What limits each cell of the '
        f"{esc(bundle)}, and the lever that addresses it: <b>rule</b> (the serving "
        "rules leave out the winners: the best kernel they admit is below the "
        "target; refine the rule), <b>identity</b> (a signature's served member is "
        "slower than another of its kernels; member order or launch keys), "
        "<b>model</b> (a better pick exists among what the rules and the identity "
        "serve; retrain the cell, <code>--mode enhance --cells</code>), <b>pool</b> "
        "(the true best of the cell's GEMMs reaches on average less than "
        f"{thr * 100:.0f}% of its roofline; generate kernels for the cell). The "
        "first three come from the audit set (under protocol v2 where the group has "
        "it), against the fastest measured kernel; <b>pool</b> from the "
        "exhaustively benched GEMMs, against the device"
        + (
            f" ({esc(cells_from)}, held out: it names the cells, never their shapes)"
            if held
            else (f" ({esc(cells_from)})" if cells_from else "")
        )
        + ". A cell can have several. "
        f"<b>vs {prev}</b> pairs the two bundles GEMM by GEMM (interval resampled "
        f"within the cell; <b>p</b> the per-cell gate's family-wise p of a "
        f"decline), <b>regressed</b> when the gate's cell test flags it (p below "
        f"{alpha:g}, worse beyond {tol * 100:.1f}%); <b>picks &lt; "
        f"{tail * 100:.0f}%</b> and <b>&lt; {floor * 100:.0f}%</b> are the gate's "
        "tail and cell floors; n/a where a record cannot give a count.</p>"
    ]
    if why:
        parts.append(f'<p class="mut">{esc(why)}</p>')
    summ = ws.summary(rows)
    unpaired = {g["group"]: g["regressed_why"] for g in summ if g["regressed"] is None}
    if unpaired:
        reasons = {}
        for g, r in unpaired.items():
            reasons.setdefault(r or "not paired", []).append(g)
        parts.append(
            f'<p class="mut">regressed vs {prev}: n/a in '
            + "; ".join(f"{esc(', '.join(gs))}: {esc(r)}" for r, gs in reasons.items())
            + "</p>"
        )
    no_gemms = [g["group"] for g in summ if g["n_tail"] is None]
    if no_gemms:
        parts.append(
            '<p class="mut">picks below the floors: n/a in '
            f"{esc(', '.join(no_gemms))}: the audit record holds no per-GEMM results "
            "for it (a record written before them)</p>"
        )
    if summ:
        parts.append(
            table(
                [
                    "<group",
                    "cells",
                    "passing",
                    "rule",
                    "identity",
                    "model",
                    "pool",
                    f"regressed vs {prev}",
                    f"picks &lt; {tail * 100:.0f}%",
                    f"picks &lt; {floor * 100:.0f}%",
                ],
                [
                    f'<tr><td class="c">{esc(g["group"])}</td><td>{intf(g["n_cells"])}</td>'
                    f'<td>{intf(g["passing"])}</td><td>{intf(g["rule"])}</td>'
                    f'<td>{intf(g["identity"])}</td><td>{intf(g["model"])}</td>'
                    f'<td>{intf(g["pool"])}</td><td>{_count_or_na(g["regressed"])}</td>'
                    f'<td>{_count_or_na(g["n_tail"])}</td><td>{_count_or_na(g["n_below_floor"])}</td></tr>'
                    for g in summ
                ],
                style="font-size:11px",
            )
        )
    weak = [r for r in rows if r["weak"]]
    head = [
        "<group",
        "<cell",
        "<class",
        "GEMMs",
        "sel_eff %",
        "rules ceiling %",
        "served ceiling %",
        f"vs {prev}",
        f"picks &lt; {tail * 100:.0f}%",
        f"picks &lt; {floor * 100:.0f}%",
        "true best % of roofline",
        "pool-gap GEMMs",
        "<lever",
        "<protocol",
    ]
    trs = []
    for r in weak:
        cls = "lose" if r["classes"] else "weak"
        ci = r.get("sel_eff_ci")
        sel = pct(r.get("sel_eff")) + (
            f' <span class="mut">[{pct(ci[0])}, {pct(ci[1])}]</span>' if ci else ""
        )
        tds = [
            f'<td class="c">{esc(r["group"])}</td>',
            f'<td class="c">{esc(r["cell"])}</td>',
            f'<td class="c"><b>{esc(r["class"])}</b></td>',
            f"<td>{intf(r.get('n'))}</td>",
            f"<td>{sel}</td>",
            f"<td>{pct(r.get('ceiling_rules'))}</td>",
            f"<td>{pct(r.get('ceiling_served'))}</td>",
            f"<td>{_ratio_txt(r.get('previous'))}</td>",
            f"<td>{_count_or_na(r.get('n_tail'))}</td>",
            f"<td>{_count_or_na(r.get('n_below_floor'))}</td>",
            f"<td>{pct(r.get('roofline'))}</td>",
            f"<td>{intf(r.get('n_gaps'))}</td>",
            f'<td class="c">{esc("; ".join(r.get("levers") or []) or "regression / tail")}</td>',
            f'<td class="c">{esc(r.get("protocol") or "")}</td>',
        ]
        trs.append(f'<tr class="{cls}">{"".join(tds)}</tr>')
    parts.append(
        f"<h3>Weak cells ({len(weak):,} of {len(rows):,})</h3>"
        + table(head, trs, style="font-size:11px")
    )
    peaks = (analysis or {}).get("peaks")
    if peaks:
        gaps = analysis.get("gaps") or []
        src = analysis.get("source")
        parts.append(
            '<h3>Pool gaps</h3><p class="cap">Roofline of the true best of every '
            "exclusive GEMM benched exhaustively: its attainable time is the largest "
            f"of 2&middot;M&middot;N&middot;K&middot;batch FLOPs at "
            f"{peaks['tflops']:,.0f} TFLOP/s ({esc(peaks['tflops_source'])}), the "
            "compulsory traffic (A and B read, D written once) at "
            f"{peaks['bandwidth_gbs']:,.0f} GB/s ({esc(peaks['bandwidth_source'])}) "
            f"and the latency floor {peaks['latency_floor_us']:.2f} &micro;s (the best "
            "time of the fastest exclusive GEMM). % of roofline = attainable / best "
            "measured. The pool gaps come from the training side only"
            + (f" ({esc(src)})" if src else "")
            + ", never from the audit set, so that kernels generated for them do not "
            "fit the pool to the held-out GEMMs: "
            f"{len(gaps):,} of {intf(analysis.get('n_gemms'))} GEMMs are below "
            f"{thr * 100:.0f}% (<code>report.pool_gap_fraction</code>), input for "
            "kernel generation"
            + (
                f" ({esc(', '.join(str(x) for x in exported))})"
                if exported
                else " (<code>--weak-spots-dir</code> exports them)"
            )
            + (
                f"; {esc(analysis['why'])}"
                if not gaps and analysis.get("why") and not src
                else ""
            )
            + f"; the {WEAK_POOL_GAPS_SHOWN} farthest are listed.</p>"
            + table(
                [
                    "<GEMM",
                    "batch",
                    "<cell",
                    "best &micro;s",
                    "attainable &micro;s",
                    "% of roofline",
                    "<bound",
                    "<best kernel",
                    "MT_M",
                    "MT_N",
                    "DepthU",
                    "MI_M",
                    "MI_N",
                    "MI_K",
                    "NTA",
                    "NTB",
                ],
                [
                    f'<tr><td class="c">{g["m"]}x{g["n"]}x{g["k"]}</td>'
                    f'<td>{intf(g["batch"])}</td><td class="c">{esc(g["cell"])}</td>'
                    f'<td>{g["best_us"]:.2f}</td><td>{g["attainable_us"]:.2f}</td>'
                    f'<td>{pct(g["fraction"])}</td><td class="c">{esc(g["bound"])}</td>'
                    f'<td class="c">{esc(g["best_signature"])}</td>'
                    + "".join(f"<td>{intf(g[f])}</td>" for f in ws.SIGNATURE)
                    + "</tr>"
                    for g in gaps[:WEAK_POOL_GAPS_SHOWN]
                ],
                style="font-size:11px",
            )
        )
    return "".join(parts)


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
        b = ' class="total" style="font-weight:600"' if bold else ""
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
    return f'<td class="c">{esc(k["sig"])}</td>' + "".join(
        f"<td>{int(v)}</td>" for v in k["signature"]
    )


# The signature's label, then its fields in SIGNATURE_FIELDS order, each a
# numeric column of its own so that it sorts.
_SIG_HEAD = [
    "<kernel (signature)",
    "MT_M",
    "MT_N",
    "DepthU",
    "MI_M",
    "MI_N",
    "MI_K",
    "NTA",
    "NTB",
]


def members_html(k, engine=True):
    """How many library solutions share one signature; with several, the
    wins (and picks) of each, most first, as a collapsed detail."""
    mem = k.get("members") or []
    if len(mem) < 2:
        return f"{len(mem)}"
    txt = ", ".join(
        f'{m["wins"]}/{intf(m["picks"])}' if engine else f'{m["wins"]}' for m in mem
    )
    return (
        f"<details><summary>{len(mem)}</summary>"
        f'<span class="mut">{"wins/picks" if engine else "wins"} per member: '
        f"{esc(txt)}</span></details>"
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
        f'<p class="cap">Rows are kernels, named by their signature (the 8 fields '
        'of the <a href="#identity">kernel identity</a>: macro tile M x N x depthU, '
        "matrix instruction, cache hints A and B), which the next columns repeat "
        "field by field (MT_M to NTB); the library solutions sharing one "
        f"are its members. {len(used)} of "
        f'{pool.get("signatures", len(ks))} signatures ({intf(pool.get("size"))} solutions of the '
        f'{esc(pool.get("source"))} pool) win at least one training GEMM or are picked for one. '
        "wins = GEMMs where a member is the fastest tested; picks = deployed picks"
        f"{'' if engine else ' (n/a: no engine)'}; tested / screened = GEMMs where a member has a "
        "full measurement / only warm-ups; within x% = tested GEMMs where a member is no more "
        "than x% slower than the fastest; drop loss = geomean loss of the oracle over all "
        "GEMMs if every member were removed; whitelisted = cells whose whitelist holds the "
        "signature; members = library solutions with the signature (with several, "
        "the wins/picks of each)."
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
                "members",
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


POOL_ROWS = {"prediction": "Prediction table", "file": "every solution"}


def pool_html(pool, why=None):
    """What the kernel pool the bundle ranks is made of, per kind
    (`lib.library_pool.summary`); `why` says why it is missing."""
    if not pool:
        return f'<p class="mut">{esc(why or "no kernel pool")}.</p>'
    text = (
        f"The kernel pool the bundle ranks: {esc(pool.get('source'))}"
        + (
            f", on the bench device ({esc(pool['device'])})"
            if pool.get("device")
            else ""
        )
        + f". <b>{pool['kernels']:,}</b> kernels in "
        f"{_count(pool['keys'], 'key', 'keys')} (signatures; the largest holds "
        f"{pool['largest_key']:,}), in this order: each kind's kernels after those "
        "of the kinds before it, a solution index an earlier kind holds left out. "
        "<code>library.kind</code> names the kinds: <code>origami</code>, the "
        "library's Prediction table in table order, and <code>bok</code>, the rows "
        "of <code>library.bok_rows</code> of every file of the library's problem "
        "type the bench device accepts, in ascending solution index."
    )
    notes = []
    near = pool.get("near_type_files") or []
    if near:
        notes.append(
            f"{_count(len(near), 'file', 'files')} of the library's data types and "
            f"operation {'holds' if len(near) == 1 else 'hold'} another problem "
            f"type and {'is' if len(near) == 1 else 'are'} not read: "
            + ", ".join(f"<code>{esc(s)}</code>" for s in near)
            + "."
        )
    if pool.get("repeated_names"):
        across = pool.get("repeated_across_kinds") or 0
        notes.append(
            f"{_count(pool['repeated_names'], 'kernel repeats', 'kernels repeat')} "
            "the solution name of an earlier pool kernel under another solution index"
            + (f" ({across:,} that of another kind)" if across else "")
            + ": TensileLite ranks each of them, so each is benched and ranked as a "
            "kernel of its own."
        )
    rows = []
    for p in pool.get("parts") or []:
        idle = p.get("idle_files") or {}
        rows.append(
            f'<tr><td class="c">{esc(p["kind"])}</td>'
            f'<td class="c">{esc(", ".join(p.get("files") or []) or "-")}</td>'
            f'<td class="c">'
            + esc(
                ", ".join(
                    f"{POOL_ROWS.get(r, r)} {n:,}"
                    for r, n in (p.get("rows") or {}).items()
                )
                or "-"
            )
            + "</td>"
            f"<td>{p['kernels']:,}</td><td>{p['keys']:,}</td>"
            f"<td>{p['largest_key']:,}</td><td>{p['duplicates']:,}</td>"
            f"<td>{p.get('blocked', 0):,}</td>"
            f'<td class="c">'
            + esc("; ".join(f"{s} ({w})" for s, w in idle.items()) or "-")
            + "</td></tr>"
        )
    return (
        f'<p class="cap">{text}</p>'
        + "".join(f'<p class="mut">{n}</p>' for n in notes)
        + table(
            [
                "<kind",
                "<files",
                "<kernels per row kind",
                "kernels",
                "keys",
                "largest key",
                "copies left out",
                "blocked",
                "<files that add none",
            ],
            rows,
            style="font-size:11px",
        )
    )


IDENTITY_FIELDS = (
    "macro tile M and N, depthU, matrix instruction M, N and K, cache hints A and B"
)


def _count(n, one, many):
    return f"{n:,} {one if n == 1 else many}"


GROUPING_TITLES = {
    "signature": "signature (no launch keys)",
    "launch": "signature + launch key",
}


def identity_html(idt, why=None):
    """What a kernel's identity is and, with `idt`
    (`lib.run_analysis.identity_summary`), what it costs in this run; `why`
    says why the numbers are missing."""
    idt = idt or {}
    reads = idt.get("attribute_reads") or []
    lk = bool(idt.get("launch_keys"))
    text = (
        f"A v3 kernel's identity is its 8-field signature ({IDENTITY_FIELDS}), "
        "framework-agnostic by design, plus the reads of an attribute spec "
        "(<code>train.attribute_spec</code>), empty by default"
        + (f"; this run reads {esc(', '.join(reads))}" if reads else " and in this run")
        + ". Library solutions that share that key are one candidate: the model scores "
        "the key once, and TensileLite ranks with <code>TieBreak::PoolOrder</code>, so a "
        "problem gets the key's first member in pool order (library table, then bank) "
        "that it admits. The launch-key option (<code>train.launch_keys</code>, "
        f"{'on' if lk else 'off'} in this run) makes each launch class of a key a key of "
        "its own."
    )
    members = idt.get("members") or {}
    m = members.get("signature")
    if not m:
        return (
            f'<p class="cap">{text}</p><p class="mut">{esc(why or "no numbers")}.</p>'
        )
    served = idt.get("served") or "signature"
    ms = members.get(served) or m
    pool_txt = (
        "pool and bank solutions"
        if idt.get("pool_source") == "library"
        else "measured solutions (the library pool and bank could not be read)"
    )
    text += (
        f" In this run the {ms['n_solutions']:,} {pool_txt} form "
        f"{_count(ms['n_keys'], 'key', 'keys')}: a key holds up to "
        f"{ms['max_members']:,} and on average {ms['mean_members']:.2f} members, and "
        f"{_count(ms['n_shared_keys'], 'key', 'keys')} "
        f"({_count(ms['n_shared_solutions'], 'solution', 'solutions')}) "
        f"{'holds' if ms['n_shared_keys'] == 1 else 'hold'} more than one."
    )
    ceil = idt.get("ceiling") or {}
    cs = ceil.get(served)
    if cs:
        text += (
            f" On the {_count(idt.get('n_gemms', 0), 'GEMM', 'GEMMs')} of "
            f"{esc(idt.get('gemms'))} (every admitted kernel benched; rows restricted to "
            "the pool), a perfect scorer of this identity reaches "
            f"<b>{cs['geomean'] * 100:.2f}%</b> of the fastest kernel (geomean; "
            f"{_count(cs['n_losing'], 'GEMM loses', 'GEMMs lose')} more than 0.1%): the "
            f"identity alone costs {(1 - cs['geomean']) * 100:.2f}%."
        )
        other = "signature" if served == "launch" else "launch"
        co = ceil.get(other)
        if co:
            text += (
                f" {'Without' if served == 'launch' else 'With'} launch keys it would "
                f"reach <b>{co['geomean'] * 100:.2f}%</b> "
                f"({_count(co['n_losing'], 'GEMM', 'GEMMs')} losing)."
            )
        if idt.get("not_audit"):
            text += f" Not the audit set: {esc(idt['not_audit'])}."
    else:
        text += (
            " No identity ceiling: "
            + esc(
                idt.get("not_audit")
                if not idt.get("gemms")
                else f"{idt.get('gemms')} holds no GEMM of the pool"
            )
            + (
                "; round 0 has no exhaustive seed bench either"
                if not idt.get("gemms")
                else ""
            )
            + "."
        )
    rows = []
    for k, title in GROUPING_TITLES.items():
        mk, ck = members.get(k), ceil.get(k)
        if not mk:
            continue
        rows.append(
            f'<tr><td class="c">{esc(title)}{" (served)" if k == served else ""}</td>'
            f"<td>{mk['n_keys']:,}</td><td>{mk['max_members']:,}</td>"
            f"<td>{mk['mean_members']:.2f}</td><td>{mk['n_shared_keys']:,}</td>"
            f"<td>{pct((ck or {}).get('geomean'))}</td>"
            f"<td>{intf((ck or {}).get('n_losing'))}</td></tr>"
        )
    return f'<p class="cap">{text}</p>' + table(
        [
            "<grouping",
            "keys",
            "max members",
            "mean members",
            "keys with &gt;1 member",
            "identity ceiling %",
            "GEMMs losing &gt;0.1%",
        ],
        rows,
        style="font-size:11px;max-width:900px",
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
        k["sig"] + (f' ({k["n_members"]} members)' if k["n_members"] > 1 else "")
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
                "members",
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
            "no measurement on any training GEMM (no evidence either way; in brackets the "
            "number of library solutions sharing one, when several do): "
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


def selection_time_html(per_cell, budget_us, origami_reference=False, reason=None):
    """Per-cell ML-on selection times; the ML-off columns (hipBLASLt's own
    ranking) only with `origami_reference`; `reason` says why there are
    none."""
    if not per_cell:
        return (
            '<p class="mut">no stage 7 timing logs in this run'
            + (f": {esc(reason)}" if reason else "")
            + ".</p>"
        )
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


# ── audit set ────────────────────────────────────────────────────────────────

AUDIT_FILES = (
    ("entering", ("stage04b",)),
    ("candidate", ("stage06", "stage05")),
)


def _round_dirs(run_dir):
    import re

    return sorted(
        (
            p
            for p in Path(run_dir).glob("round_*")
            if re.fullmatch(r"round_\d+", p.name)
        ),
        key=lambda p: int(p.name.split("_")[1]),
    )


def read_audit_results(paths):
    """(records, notes) of scripts/audit_eval.py outputs (one model record,
    or {"models": [...]}; protocol v1 alone or with its `v2`): every model
    record, `source` set to its file; a file that holds none is a note."""
    import json

    recs, notes = [], []
    for path in paths or []:
        try:
            doc = json.loads(Path(path).read_text())
        except (OSError, ValueError) as e:
            notes.append(f"audit results {path} unreadable: {e}")
            continue
        models = doc.get("models") if isinstance(doc, dict) else None
        found = [
            dict(r, source=str(path))
            for r in (models if isinstance(models, list) else [doc])
            if isinstance(r, dict) and "status" in r
        ]
        if not found:
            notes.append(f"audit results {path} hold no model record")
        recs += found
    return recs, notes


def audit_model_label(rec):
    """A model of an audit record that is not a round of the run, by where
    it was staged or trained: the stage06 output directory holding its
    staged_for_deploy/, the stage05 output directory holding its models.pt
    (such as a run's final retrain), else the parent directory and file
    name."""
    model = Path(str(rec.get("model") or rec.get("what") or "model"))
    parts = model.parts
    if "staged_for_deploy" in parts:
        out = parts[: parts.index("staged_for_deploy")]
        if len(out) > 1 and out[-1] == "stage06":
            return f"{out[-2]}/stage06"
        if out:
            return out[-1]
    if model.name == "models.pt" and len(parts) > 2 and parts[-2] == "stage05":
        return f"{parts[-3]}/stage05"
    return f"{model.parent.name}/{model.name}" if model.parent.name else model.name


def _record_rank(rec):
    """Order of preference among records of one model: v2 complete, then
    the sustained composite complete, then complete, then measured, then
    with the identity class."""
    pipeline_on_path()
    from lib import audit as al

    g = al.evaluated_group(rec, "exclusive") or {}
    return (
        al.protocol_record(rec, "v2").get("status") == "ok",
        al.protocol_record(rec, "sustained").get("status") == "ok",
        rec.get("status") == "ok",
        al.is_measured(rec),
        "identity_limited" in (g.get("classes") or {}),
    )


def _external_round(run_dir, rec, shas):
    """The round of this run whose bundle `rec` evaluated: by the model
    file's sha256, else by a `round_<N>` name with the model inside that
    round directory; None for another model."""
    import re

    if rec.get("model_sha256") in shas:
        return shas[rec["model_sha256"]]
    what = str(rec.get("what") or "")
    if not re.fullmatch(r"round_\d+", what) or not rec.get("model"):
        return None
    try:
        rd = (Path(run_dir) / what).resolve()
        model = Path(str(rec["model"])).resolve()
    except OSError:
        return None
    return what if rd in model.parents else None


def _audit_path(audit_dir):
    if not audit_dir:
        return None
    try:
        return Path(str(audit_dir)).expanduser().resolve()
    except (OSError, RuntimeError):
        return Path(str(audit_dir))


def audit_set_name(rec):
    """The audit set a record evaluated on, by its directory's name."""
    d = _audit_path(rec.get("audit_dir"))
    return d.name if d else "not recorded"


def audit_mismatch(a, b):
    """Why evaluations `a` and `b` do not score the same GEMMs the same way
    (another audit set, picks charged otherwise), so that they cannot be
    paired; None when they do. Records that name no audit set (and no pick
    rule) are taken as the same."""
    if _audit_path(a.get("audit_dir")) != _audit_path(b.get("audit_dir")):
        return f"evaluated on audit set {audit_set_name(b)}, not {audit_set_name(a)}"
    pa, pb = a.get("pick_measurement"), b.get("pick_measurement")
    if pa and pb and pa != pb:
        return f"picks charged as {pb}, not {pa}"
    return None


def _merge_mismatch(rec, audit_dir, pick_measurement):
    """Why an external record of one of the run's bundles cannot stand for
    the run's own evaluation: it is not on the run's audit set `audit_dir`
    or charges picks otherwise than `pick_measurement`; None when it can."""
    if audit_dir is None:
        return "the run configures no audit set (audit.dir)"
    run = {"audit_dir": str(audit_dir), "pick_measurement": pick_measurement}
    if rec.get("audit_dir") is None:
        return "its audit set is not recorded"
    return audit_mismatch(run, rec)


def _without_repeated_entering(recs):
    """`recs` without the entering rows whose model file a candidate row
    before them holds (a round's bundle is the next round's entering
    model): that candidate row keeps the more complete of the two records,
    its own on a tie."""
    out, candidate = [], {}
    for rnd, role, rec in recs:
        sha = rec.get("model_sha256")
        i = candidate.get(sha) if sha else None
        if role == "entering" and i is not None:
            if _record_rank(rec) > _record_rank(out[i][2]):
                out[i] = (out[i][0], out[i][1], rec)
            continue
        if role == "candidate" and sha:
            candidate[sha] = len(out)
        out.append((rnd, role, rec))
    return out


def audit_records(run_dir, external=(), audit_dir=None, pick_measurement=None):
    """(round, role, record) of every audit evaluation, oldest first: per
    round the model entering it ("entering", stage04b) and its bundle
    ("candidate", stage06, else stage05), and the final retrain's bundle
    while it is one of the run's (lib/run_bundles.py: "final_retrain",
    candidate, after the rounds it trained on), from the run's audit.json
    files and the `external` records (`read_audit_results`); an entering model
    that is an earlier candidate's model file is not repeated
    (`_without_repeated_entering`). An external record of one of the run's
    bundles stands for the run's own evaluation only when it is on the
    run's audit set `audit_dir` and charges picks as `pick_measurement`
    (`_merge_mismatch`): it then replaces the run's record of the same model
    file when it is more complete (`_record_rank`), or adds the candidate of
    a round the run did not evaluate. Every other external record follows
    as ("extra: <label>", "extra", record), never as a round, with
    `audit_set` (its audit set's name) and `extra_reason` (why it is no
    round's evaluation) added; of several of one model file on one audit set
    only the most complete is kept, one that charges picks as the run does
    first."""
    import json

    pipeline_on_path()
    from lib import run_bundles

    out = []
    rounds = run_bundles.bundles(run_dir, _round_dirs(run_dir))
    for rd in rounds:
        for role, stages in AUDIT_FILES:
            for st in stages:
                path = rd / st / "audit.json"
                if not path.is_file():
                    continue
                try:
                    rec = json.loads(path.read_text())
                except (OSError, ValueError):
                    break
                out.append((rd.name, role, dict(rec, source=str(path))))
                break
    if not external:
        return _without_repeated_entering(out)
    shas = {}
    for rd in rounds:
        try:
            rec = json.loads((rd / "stage06" / "deploy_record.json").read_text())
        except (OSError, ValueError):
            continue
        if isinstance(rec, dict) and rec.get("sha256"):
            shas[rec["sha256"]] = rd.name
    best, extra = {}, []
    for rec in external:
        rnd = _external_round(run_dir, rec, shas)
        why = None if rnd is None else _merge_mismatch(rec, audit_dir, pick_measurement)
        if rnd is None or why:
            reason = "no round of this run" if rnd is None else f"{rnd}'s bundle, {why}"
            extra.append(dict(rec, extra_reason=reason))
            continue
        key = (rnd, rec.get("model_sha256"))
        if key not in best or _record_rank(rec) > _record_rank(best[key]):
            best[key] = rec
    by_sha = {sha: rec for (_, sha), rec in best.items() if sha}
    merged, candidates = [], set()
    for rnd, role, rec in out:
        ext = by_sha.get(rec.get("model_sha256"))
        if ext is not None and _record_rank(ext) > _record_rank(rec):
            rec = ext
        candidates |= {rnd} if role == "candidate" else set()
        merged.append((rnd, role, rec))
    for (rnd, sha), rec in best.items():
        if rnd in candidates:
            if not any(r.get("model_sha256") == sha for _, _, r in merged):
                extra.append(dict(rec, extra_reason=f"another model file of {rnd}"))
            continue
        candidates.add(rnd)
        merged.append((rnd, "candidate", rec))
    order = {role: i for i, (role, _) in enumerate(AUDIT_FILES)}
    place = {rd.name: i for i, rd in enumerate(rounds)}
    merged.sort(key=lambda x: (place.get(x[0], len(place)), order[x[1]]))
    merged = _without_repeated_entering(merged)

    def preference(r):
        pm = r.get("pick_measurement")
        same = pm is None or pick_measurement is None or pm == pick_measurement
        return (same, _record_rank(r))

    kept = {}
    for i, rec in enumerate(extra):
        sha = rec.get("model_sha256")
        key = (sha, _audit_path(rec.get("audit_dir"))) if sha else i
        if key not in kept or preference(rec) > preference(kept[key]):
            kept[key] = rec
    return merged + [
        (
            f"extra: {audit_model_label(r)}",
            "extra",
            dict(r, audit_set=audit_set_name(r)),
        )
        for r in kept.values()
    ]


def class_counts(group, target):
    """Every class of `lib.audit.CLASSES` in an evaluated group: its
    recorded counts; for a record written before the identity class
    existed, the classes of its cells again when they carry the served-key
    ceiling, else identity_limited None (not recorded)."""
    pipeline_on_path()
    from lib import audit as al

    cl = dict(group.get("classes") or {})
    if "identity_limited" in cl:
        return {k: cl.get(k, 0) for k in al.CLASSES}
    cells = (group.get("per_cell") or {}).values()
    if cells and all("ceiling_served" in c for c in cells):
        out = {k: 0 for k in al.CLASSES}
        for c in cells:
            k, _ = al.classify(
                c.get("sel_eff"),
                c.get("sel_eff_ci"),
                c.get("ceiling_rules"),
                c.get("ceiling_rules_ci"),
                target,
                c.get("ceiling_served"),
                c.get("ceiling_served_ci"),
            )
            if k:
                out[k] += 1
        return out
    return {**{k: cl.get(k) for k in al.CLASSES}, "identity_limited": None}


def classes_text(counts, sep=" / "):
    """pool / identity / model / passing counts; a class the record does not
    hold says so."""
    pipeline_on_path()
    from lib import audit as al

    return sep.join(
        "not recorded" if counts.get(k) is None else f"{int(counts[k]):,}"
        for k in al.CLASSES
    )


def _ci_cell(v, ci):
    txt = pct(v)
    if ci:
        txt += f' <span class="mut">[{pct(ci[0])}, {pct(ci[1])}]</span>'
    return txt


def _ratio_cell(v, ci):
    txt = "n/a" if v is None else f"&times;{v:.4f}"
    if ci:
        txt += f' <span class="mut">[{ci[0]:.4f}, {ci[1]:.4f}]</span>'
    return txt


def _status_text(rec):
    """A record's coverage line (records before `status_text` existed say
    their status)."""
    if rec.get("status_text"):
        return rec["status_text"]
    if rec.get("status") == "ok":
        return "complete"
    return f"{rec.get('status')}: {rec.get('reason')}"


def _group_null(rec, label):
    gs = (rec.get("group_status") or {}).get(label) or {}
    st = gs.get("status") or (
        "not evaluated" if rec.get("groups") else rec.get("status")
    )
    return f"null: {str(st).replace('_', ' ')}"


def audit_changes(recs, bootstrap=2000, official=None):
    """Consecutive candidate evaluations (round, record) compared on every
    group either names (`lib.audit.paired_change`: under the official
    protocol `official` when both have it for the group, else v2, else v1),
    oldest first; then every extra model against the latest candidate that
    evaluated the exclusive group. A pair that does not score the same GEMMs
    the same way (`audit_mismatch`) is not compared."""
    pipeline_on_path()
    from lib import audit as al

    cands = [dict(rec, what=rnd) for rnd, role, rec in recs if role == "candidate"]
    pairs = list(zip(cands, cands[1:]))
    base = next(
        (c for c in reversed(cands) if al.evaluated_group(c, "exclusive")), None
    )
    if base is not None:
        pairs += [
            (base, dict(rec, what=rnd)) for rnd, role, rec in recs if role == "extra"
        ]
    out = []
    for before, after in pairs:
        labels = list(
            dict.fromkeys([*(before.get("groups") or {}), *(after.get("groups") or {})])
        ) or ["exclusive"]
        groups = {}
        apart = audit_mismatch(before, after)
        for label in labels:
            if apart:
                proto = al.protocol_of([before, after], label, official)
                groups[label] = {"reason": apart, "protocol": proto}
                continue
            c, why, proto = al.paired_change(before, after, label, bootstrap, official)
            groups[label] = c if c is not None else {"reason": why, "protocol": proto}
        out.append({"before": before["what"], "after": after["what"], "groups": groups})
    return out


def _source_td(rec, run_dir):
    src = str(rec.get("source") or "")
    try:
        name = str(Path(src).resolve().relative_to(Path(run_dir).resolve()))
    except (OSError, ValueError):
        name = Path(src).name
    return f'<td title="{esc(src)}">{esc(name)}</td>'


def audit_html(run_dir, target, status=None, recs=None, official=None):
    """The audit set per round (lib/audit.py): the model entering the round
    (stage04b) and the round's bundle (stage06, else stage05), each cell
    against its fastest benched kernel, never trained on; a partial
    evaluation shows the groups it evaluated and the status of the others.
    `recs` are the records to show (`audit_records`, default the run's
    own): with audit_eval results they fill in or replace rounds'
    evaluations of the same model files, and other records follow as extra
    rows labelled with their audit set. Then each candidate against the one
    before it, and each extra model against the latest candidate, paired
    GEMM by GEMM on the groups both evaluated. Numbers are under the
    official protocol `official` (`lib.audit.view`), with v2's exclusive
    cell mean alongside. `status` is one line on the audit set's state
    now."""
    pipeline_on_path()
    from lib import audit as al

    if recs is None:
        recs = audit_records(run_dir)
    now = f'<p class="cap"><b>Audit set now:</b> {esc(status)}.</p>' if status else ""
    if not recs:
        return f'<p class="mut">No audit set evaluation in any round.</p>{now}'
    labels = []
    for _, _, rec in recs:
        for k in rec.get("groups") or {}:
            if k != "exclusive" and k not in labels:
                labels.append(k)
    head = [
        "Round",
        "Model",
        "Coverage",
        "Protocol",
        "<Cell mean (exclusive)",
        "<GEMM geomean",
        "<Min cell",
        f"<Cells &lt; {pct(target, 0)}%",
        "<Pool / identity / model / pass",
        "<v2 (exclusive)",
        *[f"<{esc(lb)}" for lb in labels],
        "<GEMMs",
        "<Source",
    ]
    rows = []
    for rnd, role, raw in recs:
        rec, proto = al.view(raw, official)
        v2 = al.protocol_record(raw, "v2")
        sus = al.protocol_record(raw, "sustained")
        notes = []
        if proto == "v1" and v2:
            notes.append(f"v2 {_status_text(v2)}")
        if proto != "sustained" and sus and al.preference(official)[0] == "sustained":
            notes.append(f"sustained composite {_status_text(sus)}")
        note = "".join(f' <span class="mut">({esc(n)})</span>' for n in notes)
        v2g = al.evaluated_group(v2, "exclusive") if v2 else None
        if proto == "v2":
            along = '<span class="mut">this row</span>'
        elif v2g is not None:
            along = _ci_cell(v2g.get("cell_mean"), v2g.get("cell_mean_ci"))
        else:
            along = '<span class="mut">&ndash;</span>'
        who = (
            f"<td>extra</td><td>{esc(rnd[len('extra: '):])} "
            f'<span class="mut" title="{esc(raw.get("audit_dir"))}">(audit set '
            f"{esc(raw.get('audit_set'))}; {esc(raw.get('extra_reason'))}; sha256 "
            f'{esc(str(raw.get("model_sha256"))[:12])})</span></td>'
            if role == "extra"
            else f"<td>{esc(rnd)}</td><td>{esc(role)}</td>"
        )
        lead = (
            f"<tr>{who}"
            f"<td>{esc(_status_text(raw))}</td>"
            f"<td>{esc(al.PROTOCOLS[proto])}{note}</td>"
        )
        tail = _source_td(raw, run_dir) + "</tr>"
        if not al.is_measured(rec):
            rows.append(
                lead + f'<td class="c mut" colspan="{len(head) - 5}">null: '
                f"{esc(rec.get('status'))} &mdash; {esc(rec.get('reason'))}</td>" + tail
            )
            continue
        g = al.evaluated_group(rec, "exclusive")
        ctx = [
            (
                pct(al.evaluated_group(rec, lb)["cell_mean"])
                if al.evaluated_group(rec, lb)
                else f'<span class="mut">{esc(_group_null(rec, lb))}</span>'
            )
            for lb in labels
        ]
        if g is None:
            excl = (
                f'<td class="c mut" colspan="5">exclusive '
                f"{esc(_group_null(rec, 'exclusive'))}</td>"
            )
        else:
            mc = g.get("min_cell") or {}
            counts = class_counts(g, float(raw.get("target") or target))
            excl = (
                f'<td class="c">{_ci_cell(g.get("cell_mean"), g.get("cell_mean_ci"))}</td>'
                f'<td class="c">{pct(g.get("gemm_geomean"))}</td>'
                f'<td class="c">{esc(mc.get("cell"))} {pct(mc.get("sel_eff"))}</td>'
                f'<td class="c">{intf(g.get("n_cells_below_target"))} / {intf(g.get("n_cells"))}</td>'
                f'<td class="c">{classes_text(counts)}</td>'
            )
        rows.append(
            lead
            + excl
            + f'<td class="c">{along}</td>'
            + "".join(f'<td class="c">{v}</td>' for v in ctx)
            + f'<td class="c">{intf(rec.get("n_gemms"))}</td>'
            + tail
        )
    out = [
        '<p class="cap">Every number is a geometric mean of selection efficiency '
        "against the fastest kernel benched on each audit GEMM (every pool and "
        "bank kernel, near-winners read again), on GEMMs no round trains on. The "
        "cell mean weighs the 96 base cells equally; intervals are bootstrap "
        "95%. <b>Pool-limited</b>: the best kernel the serving rules allow is "
        "below the target; <b>identity-limited</b>: it is not, but the best key "
        "as the runtime serves it (its first kernel with a row in pool order) "
        "is, so no model of this identity reaches the target; "
        "<b>model-limited</b>: neither is, but the pick is. "
        "<b>Coverage</b>: a partial evaluation scores only the groups whose "
        "parts are all benched and enriched; the others are null. "
        "<b>Protocol</b>: v1 reads each kernel once in the exhaustive pass; v2 "
        "benches each GEMM's kernels within 10% of that pass's winner together "
        "with every scored model's pick in repeated separate processes and "
        "scores per-kernel medians (lib/audit_v2.py); the sustained composite "
        "scores every power-capped exclusive GEMM on its near-winners and picks "
        "measured again in long self-soaked windows, at the device's power "
        "limit (lib/audit_sustained.py), and every other GEMM as v2, its "
        "context groups being v2's. A row shows the official protocol "
        "(audit.official, the sustained composite by default) when it is "
        "complete for its model, else v2, else v1; <b>v2 (exclusive)</b> gives "
        "v2's exclusive cell mean alongside. An identity class a record does not "
        "hold (it predates the class) is <i>not recorded</i>. <b>Source</b>: the "
        "audit.json the run wrote, or an audit_eval result given to the report, "
        "which replaces a less complete record of the same model file when it "
        "is on this run's audit set and charges picks as the run does. "
        "<b>Extra</b> rows are the other records of audit_eval results: models "
        "that are no round of this run, or a round's model evaluated on another "
        "audit set or with picks charged otherwise; each names its audit set, "
        "and only a pair on the same audit set and pick rule is compared. A "
        "round's <i>entering</i> model is usually the previous round's "
        "candidate; it then has no row of its own, and the candidate row shows "
        "the more complete of the two evaluations (Source names it).</p>",
        table(head, rows),
        now,
    ]
    crow = []
    for c in audit_changes(recs, official=official):
        for label, ch in c["groups"].items():
            pair = (
                f"<td>{esc(c['before'])} &rarr; {esc(c['after'])}</td>"
                f"<td>{esc(label)}</td><td>{esc(al.PROTOCOLS[ch['protocol']])}</td>"
            )
            if "reason" in ch:
                crow.append(
                    f'<tr>{pair}<td class="c mut" colspan="5">not compared: '
                    f"{esc(ch['reason'])}</td></tr>"
                )
                continue
            crow.append(
                f"<tr>{pair}"
                f'<td class="c">{_ratio_cell(ch["cell_mean_ratio"], ch["cell_mean_ratio_ci"])}</td>'
                f'<td class="c">{_ratio_cell(ch["gemm_geomean_ratio"], ch["gemm_geomean_ratio_ci"])}</td>'
                f'<td class="c">{intf(ch["n_gemms"])} / {intf(ch["n_cells"])}</td>'
                f"<td>{esc(', '.join(ch['improved']) or '-')}</td>"
                f"<td>{esc(', '.join(ch['regressed']) or '-')}</td></tr>"
            )
    if crow:
        out.append(
            "<details><summary>Candidate against the previous round's candidate, "
            "extra models against the latest candidate (paired on the same "
            "GEMMs)</summary>"
            + table(
                [
                    "Rounds",
                    "Group",
                    "Protocol",
                    "<Cell mean ratio",
                    "<GEMM geomean ratio",
                    "<GEMMs / cells",
                    "Improved cells",
                    "Regressed cells",
                ],
                crow,
            )
            + '<p class="cap">Ratio = after / before of the selection efficiency on '
            "the GEMMs both evaluated, with a bootstrap 95% interval; a cell "
            "improved or regressed when its interval excludes 1. Only groups "
            "both evaluations scored are compared, under the official protocol "
            "when both have it for the group (the sustained composite for the "
            "exclusive group only), else v2, else v1.</p></details>"
        )
    latest_what, latest = next(
        (
            (rnd, r)
            for rnd, role, r in reversed(recs)
            if role == "candidate" and al.evaluated_group(r, "exclusive")
        ),
        (None, None),
    )
    if latest is not None:
        latest, latest_proto = al.view(latest, official)
        cells = ((latest.get("groups") or {}).get("exclusive") or {}).get(
            "per_cell"
        ) or {}
        worst = sorted(cells.items(), key=lambda kv: kv[1].get("sel_eff") or 0)[:15]
        wrows = [
            "<tr>"
            f"<td>{esc(c)}</td><td class=\"c\">{intf(v.get('n'))}</td>"
            f'<td class="c">{_ci_cell(v.get("sel_eff"), v.get("sel_eff_ci"))}</td>'
            f'<td class="c">{_ci_cell(v.get("ceiling_served"), v.get("ceiling_served_ci"))}</td>'
            f'<td class="c">{_ci_cell(v.get("ceiling_rules"), v.get("ceiling_rules_ci"))}</td>'
            f'<td class="c">{pct(v.get("ceiling_table_rules"))}</td>'
            f'<td class="c">{pct(v.get("ceiling_table"))}</td>'
            f"<td>{esc(v.get('class'))}{'' if v.get('confident') else ' (?)'}</td></tr>"
            for c, v in worst
        ]
        out.append(
            "<details><summary>Weakest cells of the latest candidate "
            f"({esc(latest_what or '')}, {esc(al.PROTOCOLS[latest_proto])})</summary>"
            + table(
                [
                    "Cell",
                    "<GEMMs",
                    "<Selection eff.",
                    "<Ceiling (served keys)",
                    "<Ceiling (rules)",
                    "<Library only (rules)",
                    "<Library only",
                    "Class",
                ],
                wrows,
            )
            + '<p class="cap">(?) = the interval does not settle the class.</p></details>'
        )
    return "".join(out)
