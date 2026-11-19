# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Statistics and the speed bar of the stride-aware dense attention A/B harness.

Pure Python (no numpy, no PyYAML, no torch): importable and runnable on a bare
CPU host. Used by ``attention_dense_strided_ab.py`` (the driver) and by its
``bar`` subcommand; ``python attention_ab_stats.py RESULT`` runs the same check.

Input: a harness result (a ``.yaml`` written as JSON-flow YAML, which is valid
YAML and readable without PyYAML; a ``.json``; or a log holding the driver's
``AB_ROW {json}`` lines, so a run that died half way can still be analysed).

Result layout::

    {"meta": {"baseline": L, "control": L, "arms": {label: {"tree": ...}}, ...},
     "cases": [{"case": "<shape>/<layout>", "shape_id", "layout", "dtype",
                "body", "variant", "status", "spec_identity_equal",
                "arms": {label: {"status": "ok" | "refused" | "error", "reason"}},
                "measurements": [{"round", "pos", "arm", "us"}],
                "check_pre": {label: {"pass": bool, ...}}, "check_post": {...}}]}

Method (every constant is a method constant, none is a measured number):

- Per case (a shape on a layout) and arm X other than the baseline: the ratio
  t_X / t_baseline of each round (arms are measured in a mirrored order inside
  a round), its median, and a 95% interval of that median (bootstrap over
  rounds).
- Cases are grouped into strata (default: body and layout). The control arm is a
  second copy of the baseline tree, loaded in its own process: its ratio to the
  baseline measures what a code-object change alone does. Noise band of a
  stratum = max(1%, 95th percentile over its cases of |median control/baseline - 1|).
- Geomean of the per-case medians per arm, with a 95% interval from a two-level
  bootstrap (cases with replacement, then rounds within each).
- A case is "wholly above the band" when the lower end of its own 95% interval
  is above 1 + band.
- Verdict per (stratum, arm): "too noisy" when the p95 above exceeds 2%; "pass"
  when the geomean's upper bound is below +1% and no case is wholly above the
  band; else "fail". "fail (check)" / "fail (spec)" / "fail (error)" when a
  case of the stratum failed its output check, ran another kernel spec in some
  arm, or raised; "fail (refused)" when the arm declined a case the baseline
  served (a lost case, not a neutral one); "refused" when the arm declined every
  case of the stratum and the baseline did too; "no data" otherwise. A case with
  any such problem never enters a geomean or a band. A case an arm refuses is
  excluded from that arm's ratios only.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from typing import Any
from collections.abc import Iterable, Mapping, Sequence

BAND_FLOOR = 0.01  # noise band never below 1%
NOISY_P95 = 0.02  # control p95 above this: the stratum is too noisy to judge
GEOMEAN_BOUND = 0.01  # pass needs the geomean's upper 95% bound below +1%
CI_LOW = 2.5
CI_HIGH = 97.5
DEFAULT_GROUP_BY = ("body", "layout")
GROUP_CHOICES = {
    "body-layout": ("body", "layout"),
    "layout": ("layout",),
    "body": ("body",),
    "all": (),
}
PASS_VERDICTS = ("pass",)
NEUTRAL_VERDICTS = ("refused", "no data")


# --------------------------------------------------------------------------- #
# Small numeric helpers
# --------------------------------------------------------------------------- #


def median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        raise ValueError("median of an empty sequence")
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def percentile(xs: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (numpy's default), q in [0, 100]."""
    s = sorted(xs)
    if not s:
        raise ValueError("percentile of an empty sequence")
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * (q / 100.0)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def geomean(xs: Sequence[float]) -> float:
    if not xs:
        raise ValueError("geomean of an empty sequence")
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


# --------------------------------------------------------------------------- #
# Ratios and bootstrap
# --------------------------------------------------------------------------- #


def paired_ratios(
    measurements: Iterable[Mapping[str, Any]], arm: str, ref: str
) -> list[float]:
    """t_arm / t_ref per round, for rounds where both arms were measured, in
    round order."""
    by_round: dict[int, dict[str, float]] = {}
    for m in measurements:
        t = m.get("us")
        if t is None or not (
            isinstance(t, (int, float)) and t > 0 and math.isfinite(t)
        ):
            continue
        by_round.setdefault(int(m["round"]), {})[str(m["arm"])] = float(t)
    out = []
    for r in sorted(by_round):
        row = by_round[r]
        if arm in row and ref in row:
            out.append(row[arm] / row[ref])
    return out


def bootstrap_medians(
    ratios: Sequence[float], n_boot: int, rng: random.Random
) -> list[float]:
    """Medians of ``n_boot`` resamples (with replacement) of ``ratios``."""
    n = len(ratios)
    pool = []
    for _ in range(n_boot):
        pool.append(median([ratios[rng.randrange(n)] for _ in range(n)]))
    return pool


def hierarchical_geomean_ci(
    pools: Sequence[Sequence[float]], n_boot: int, rng: random.Random
) -> tuple[float, float]:
    """95% interval of the geomean over cases of the per-case median ratio.

    ``pools[i]`` holds case i's bootstrap medians (rounds resampled). Each
    replicate draws the cases with replacement and, for every drawn case, one
    median from its pool at a random index (so a case drawn twice gets two
    independent round resamples)."""
    k = len(pools)
    reps = []
    for _ in range(n_boot):
        acc = 0.0
        for _ in range(k):
            pool = pools[rng.randrange(k)]
            acc += math.log(pool[rng.randrange(len(pool))])
        reps.append(math.exp(acc / k))
    return percentile(reps, CI_LOW), percentile(reps, CI_HIGH)


# --------------------------------------------------------------------------- #
# Decision
# --------------------------------------------------------------------------- #


def noise_band(
    control_medians: Sequence[float],
) -> tuple[float | None, float | None]:
    """``(band, p95)``: p95 = 95th percentile of |median control/baseline - 1|
    over cases; band = max(1%, p95)."""
    if not control_medians:
        return None, None
    p95 = percentile([abs(x - 1.0) for x in control_medians], 95)
    return max(BAND_FLOOR, p95), p95


def decide(
    *,
    p95: float | None,
    geo_hi: float | None,
    wholly_above: Sequence[str],
    check_failures: Sequence[str] = (),
    spec_mismatches: Sequence[str] = (),
    errors: Sequence[str] = (),
    n_shapes: int = 1,
    n_refused: int = 0,
    lost_cases: Sequence[str] = (),
) -> str:
    """Verdict string for one (stratum, arm). ``lost_cases``: cases the arm refused
    although the baseline served them."""
    if check_failures:
        return "fail (check)"
    if spec_mismatches:
        return "fail (spec)"
    if errors:
        return "fail (error)"
    if lost_cases:
        return "fail (refused)"
    if n_shapes == 0 or geo_hi is None:
        return "refused" if n_refused else "no data"
    if p95 is None:
        return "no data"
    if p95 > NOISY_P95:
        return "too noisy"
    if geo_hi < 1.0 + GEOMEAN_BOUND and not wholly_above:
        return "pass"
    return "fail"


# --------------------------------------------------------------------------- #
# Analysis of a harness result
# --------------------------------------------------------------------------- #


def failed_check_arms(case: Mapping[str, Any]) -> list[str]:
    """Arms whose output check failed, before or after timing."""
    out: list[str] = []
    for phase in ("check_pre", "check_post"):
        for label, chk in (case.get(phase) or {}).items():
            if (
                isinstance(chk, Mapping)
                and "pass" in chk
                and not chk["pass"]
                and label not in out
            ):
                out.append(str(label))
    return out


def classify_case(case: Mapping[str, Any]) -> tuple[str, list[str]]:
    """``(kind, arms)``: kind is ``ok`` (judged), or the reason the case never
    enters a geomean: ``check`` (an arm's output check failed), ``spec`` (the arms
    ran different kernel specs), ``error`` (an arm raised), ``skipped`` (the layout
    does not apply to the shape, or the baseline declined it)."""
    failed = failed_check_arms(case)
    if failed:
        return "check", failed
    if case.get("spec_identity_equal") is False:
        return "spec", []
    errs = [
        str(l)
        for l, a in (case.get("arms") or {}).items()
        if (a or {}).get("status") == "error"
    ]
    if errs or case.get("status") == "error":
        return "error", errs
    if case.get("status") in ("not_applicable", "refused"):
        return "skipped", []
    return "ok", []


def _stratum_key(case: Mapping[str, Any], group_by: Sequence[str]) -> str:
    return (
        "/".join("-" if case.get(k) is None else str(case.get(k)) for k in group_by)
        or "all"
    )


def _arm_labels(result: Mapping[str, Any]) -> list[str]:
    seen: list[str] = list((result.get("meta") or {}).get("arms") or {})
    for c in result.get("cases") or ():
        for a in list(c.get("arms") or ()) + [
            str(m["arm"]) for m in c.get("measurements") or ()
        ]:
            if a not in seen:
                seen.append(a)
    return seen


def check_same_tree(meta: Mapping[str, Any], baseline: str, control: str) -> None:
    """Refuse a control that is not a copy of the baseline's tree."""
    arms = meta.get("arms") or {}
    tb = (arms.get(baseline) or {}).get("tree")
    tc = (arms.get(control) or {}).get("tree")
    if tb is not None and tc is not None and tb != tc:
        raise ValueError(
            f"control {control!r} ({tc}) is not the same tree as baseline {baseline!r} ({tb}): "
            "a control must be a second copy of the baseline's code"
        )


def analyze(
    result: Mapping[str, Any],
    n_boot: int = 10000,
    seed: int = 0,
    baseline: str | None = None,
    control: str | None = None,
    group_by: Sequence[str] = DEFAULT_GROUP_BY,
) -> dict[str, Any]:
    """Per-case and per-stratum statistics plus verdicts (YAML-able dict)."""
    rng = random.Random(seed)
    meta = result.get("meta") or {}
    baseline = baseline or meta.get("baseline")
    control = control or meta.get("control")
    if not baseline or not control:
        raise ValueError("analysis needs a baseline label and a control label")
    if baseline == control:
        raise ValueError("the control must be a different arm than the baseline")
    check_same_tree(meta, baseline, control)
    cases = list(result.get("cases") or ())
    labels = _arm_labels(result)
    test_arms = [a for a in labels if a not in (baseline, control)]

    rows: list[dict[str, Any]] = []
    pools: dict[tuple[str, str], list[float]] = {}
    for c in cases:
        kind, who = classify_case(c)
        row: dict[str, Any] = {
            "case": c.get("case"),
            "shape_id": c.get("shape_id"),
            "layout": c.get("layout"),
            "body": c.get("body"),
            "variant": c.get("variant"),
            "dtype": c.get("dtype"),
            "stratum": _stratum_key(c, group_by),
            "excluded": None if kind == "ok" else kind,
            "problem_arms": who,
            "refused": {
                str(l): (a or {}).get("reason")
                for l, a in (c.get("arms") or {}).items()
                if (a or {}).get("status") == "refused"
            },
            # The baseline ran the case (its output check may still have failed): only
            # then is a test arm's refusal a lost case.
            "baseline_served": ((c.get("arms") or {}).get(baseline) or {}).get("status")
            in ("ok", "check_failed"),
            "base_us_median": None,
            "arms": {},
        }
        meas = c.get("measurements") or []
        base_times = [
            float(m["us"]) for m in meas if m["arm"] == baseline and m.get("us")
        ]
        if base_times:
            row["base_us_median"] = median(base_times)
        if kind == "ok":
            for arm in [control] + test_arms:
                ratios = paired_ratios(meas, arm, baseline)
                if not ratios:
                    continue
                pool = bootstrap_medians(ratios, n_boot, rng)
                pools[(str(c.get("case")), arm)] = pool
                row["arms"][arm] = {
                    "n_rounds": len(ratios),
                    "median_ratio": median(ratios),
                    "ci95": [percentile(pool, CI_LOW), percentile(pool, CI_HIGH)],
                    "round_min": min(ratios),
                    "round_max": max(ratios),
                }
        rows.append(row)

    strata: dict[str, Any] = {}
    keys: list[str] = []
    for r in rows:
        if r["stratum"] not in keys:
            keys.append(r["stratum"])
    for key in keys:
        srows = [r for r in rows if r["stratum"] == key]
        judged = [r for r in srows if r["excluded"] is None]
        ctrl = [
            r["arms"][control]["median_ratio"] for r in judged if control in r["arms"]
        ]
        band, p95 = noise_band(ctrl)
        entry: dict[str, Any] = {
            "n_cases": len(srows),
            "n_judged": len(judged),
            "excluded": {r["case"]: r["excluded"] for r in srows if r["excluded"]},
            "control": {
                "n_cases": len(ctrl),
                "p95_abs_dev": p95,
                "band": band,
                "max_abs_dev": max((abs(x - 1) for x in ctrl), default=None),
            },
            "arms": {},
        }
        for arm in test_arms:
            have = [r for r in judged if arm in r["arms"]]
            refused_cases = [r["case"] for r in srows if arm in r["refused"]]
            lost = [
                r["case"] for r in srows if arm in r["refused"] and r["baseline_served"]
            ]
            relevant = {arm, baseline, control}
            chk = [
                r["case"]
                for r in srows
                if r["excluded"] == "check" and relevant & set(r["problem_arms"])
            ]
            spc = [r["case"] for r in srows if r["excluded"] == "spec"]
            err = [
                r["case"]
                for r in srows
                if r["excluded"] == "error"
                and (not r["problem_arms"] or relevant & set(r["problem_arms"]))
            ]
            if not have:
                entry["arms"][arm] = {
                    "n_cases": 0,
                    "n_refused": len(refused_cases),
                    "refused_cases": refused_cases,
                    "lost_cases": lost,
                    "check_failures": chk,
                    "spec_mismatches": spc,
                    "errors": err,
                    "verdict": decide(
                        p95=p95,
                        geo_hi=None,
                        wholly_above=[],
                        check_failures=chk,
                        spec_mismatches=spc,
                        errors=err,
                        n_shapes=0,
                        n_refused=len(refused_cases),
                        lost_cases=lost,
                    ),
                }
                continue
            meds = [r["arms"][arm]["median_ratio"] for r in have]
            gm = geomean(meds)
            lo, hi = hierarchical_geomean_ci(
                [pools[(r["case"], arm)] for r in have], n_boot, rng
            )
            above = [
                r["case"]
                for r in have
                if band is not None and r["arms"][arm]["ci95"][0] > 1.0 + band
            ]
            median_above = [
                r["case"]
                for r in have
                if band is not None and r["arms"][arm]["median_ratio"] > 1.0 + band
            ]
            entry["arms"][arm] = {
                "n_cases": len(have),
                "n_refused": len(refused_cases),
                "refused_cases": refused_cases,
                "lost_cases": lost,
                "geomean": gm,
                "geomean_ci95": [lo, hi],
                "cases_wholly_above_band": above,
                "cases_median_above_band": median_above,
                "check_failures": chk,
                "spec_mismatches": spc,
                "errors": err,
                "verdict": decide(
                    p95=p95,
                    geo_hi=hi,
                    wholly_above=above,
                    check_failures=chk,
                    spec_mismatches=spc,
                    errors=err,
                    n_shapes=len(have),
                    n_refused=len(refused_cases),
                    lost_cases=lost,
                ),
            }
        strata[key] = entry

    return {
        "method": {
            "ratio": "per round t_arm / t_baseline over paired rounds; per-case median",
            "case_ci": "95% percentile bootstrap over rounds of the median",
            "geomean_ci": "95% two-level bootstrap (cases, then rounds)",
            "band": f"max({BAND_FLOOR:.0%}, p95 over cases of |median control/baseline - 1|)",
            "wholly_above": "case's 95% interval lower end > 1 + band",
            "too_noisy": f"p95 > {NOISY_P95:.0%}",
            "pass": f"geomean upper bound < 1 + {GEOMEAN_BOUND:.0%} and no case wholly above the band",
            "strata": "/".join(group_by) or "all",
            "n_boot": n_boot,
            "seed": seed,
        },
        "baseline": baseline,
        "control": control,
        "test_arms": test_arms,
        "meta": {
            k: meta.get(k)
            for k in (
                "timing_method",
                "device_name",
                "cu_count",
                "compiler",
                "rounds",
                "label",
                "arch",
                "arms",
            )
        },
        "all_output_checks_passed": not any(r["excluded"] == "check" for r in rows),
        "strata": strata,
        "cases": rows,
    }


# --------------------------------------------------------------------------- #
# The bar check
# --------------------------------------------------------------------------- #

EXIT_PASS, EXIT_FAIL, EXIT_NO_DATA, EXIT_NOISY = 0, 1, 2, 3


def bar_check(an: Mapping[str, Any]) -> dict[str, Any]:
    """The speed bar on an analysis: PASS when every (stratum, test arm) verdict is
    ``pass`` or neutral (``refused`` / ``no data``) and at least one is ``pass``;
    FAIL when any verdict is a ``fail*``; INCONCLUSIVE when none fails but some
    stratum is too noisy; NO DATA when nothing could be judged."""
    rows = []
    for key, e in (an.get("strata") or {}).items():
        for arm, a in (e.get("arms") or {}).items():
            rows.append(
                {
                    "stratum": key,
                    "arm": arm,
                    "verdict": a["verdict"],
                    "n_cases": a.get("n_cases", 0),
                    "geomean": a.get("geomean"),
                    "upper": (a.get("geomean_ci95") or [None, None])[1],
                    "band": e["control"]["band"],
                    "p95": e["control"]["p95_abs_dev"],
                    "wholly_above": a.get("cases_wholly_above_band", []),
                }
            )
    verdicts = [r["verdict"] for r in rows]
    if any(v.startswith("fail") for v in verdicts):
        word, code = "FAIL", EXIT_FAIL
    elif any(v == "too noisy" for v in verdicts):
        word, code = "INCONCLUSIVE (too noisy)", EXIT_NOISY
    elif any(v in PASS_VERDICTS for v in verdicts):
        word, code = "PASS", EXIT_PASS
    else:
        word, code = "NO DATA", EXIT_NO_DATA
    base, ctrl = an.get("baseline"), an.get("control")
    lines = [
        f"bar check: baseline={base} control={ctrl} arms under test={','.join(an.get('test_arms') or []) or '-'}",
        f"all output checks passed (every arm, before and after timing): {an.get('all_output_checks_passed')}",
    ]
    for key, e in (an.get("strata") or {}).items():
        c = e["control"]
        lines.append(
            f"stratum {key}: {e['n_judged']} of {e['n_cases']} cases judged; "
            f"{ctrl}/{base} p95|dev|={_abs_pct(c['p95_abs_dev'])} band={_abs_pct(c['band'])}"
        )
        if e["excluded"]:
            lines.append(
                "  excluded: "
                + ", ".join(f"{k} ({v})" for k, v in e["excluded"].items())
            )
        for arm, a in (e.get("arms") or {}).items():
            lost = a.get("lost_cases") or []
            lost_txt = f"; lost (served by {base}): {', '.join(lost)}" if lost else ""
            if a.get("n_cases", 0) == 0:
                lines.append(
                    f"  {arm}/{base}: no judged case (refused in {a['n_refused']}){lost_txt} -> {a['verdict']}"
                )
                continue
            lo, hi = a["geomean_ci95"]
            refused = (
                f" (refused in {a['n_refused']}{lost_txt})" if a["n_refused"] else ""
            )
            above = ", ".join(a["cases_wholly_above_band"]) or "none"
            lines.append(
                f"  {arm}/{base}: geomean {_pct(a['geomean'])} [{_pct(lo)} .. {_pct(hi)}] "
                f"over {a['n_cases']} cases{refused}; wholly above band: {above} -> {a['verdict']}"
            )
    lines.append(f"BAR: {word}")
    return {"verdict": word, "exit_code": code, "rows": rows, "lines": lines}


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{(x - 1.0) * 100:+.2f}%"


def _abs_pct(x: float | None) -> str:
    return "-" if x is None else f"{x * 100:.2f}%"


def to_markdown(an: Mapping[str, Any]) -> str:
    lines = []
    meta = an.get("meta") or {}
    base, ctrl = an.get("baseline"), an.get("control")
    lines.append(
        f"Timing: {meta.get('timing_method')}; device: {meta.get('device_name')} "
        f"({meta.get('cu_count')} CUs); baseline: {base}; control: {ctrl}; "
        f"rounds: {meta.get('rounds')}; all output checks passed: {an.get('all_output_checks_passed')}"
    )
    lines.append("")
    lines.append(
        f"| stratum | cases | {ctrl}/{base} p95 | band | arm | arm/{base} geomean | 95% interval | wholly above band | verdict |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for key, e in (an.get("strata") or {}).items():
        c = e["control"]
        arms = e.get("arms") or {}
        if not arms:
            lines.append(
                f"| {key} | {e['n_cases']} | {_abs_pct(c['p95_abs_dev'])} | {_abs_pct(c['band'])} | - | - | - | - | no data |"
            )
        for arm, a in arms.items():
            if a.get("n_cases", 0) == 0:
                lines.append(
                    f"| {key} | {e['n_cases']} | {_abs_pct(c['p95_abs_dev'])} | {_abs_pct(c['band'])} | {arm} | - | - | - | {a['verdict']} |"
                )
                continue
            lo, hi = a["geomean_ci95"]
            lines.append(
                f"| {key} | {a['n_cases']} | {_abs_pct(c['p95_abs_dev'])} | {_abs_pct(c['band'])} | {arm} | "
                f"{_pct(a['geomean'])} | {_pct(lo)} .. {_pct(hi)} | "
                f"{', '.join(a['cases_wholly_above_band']) or 'none'} | **{a['verdict']}** |"
            )
    lines.append("")
    arms: list[str] = []
    for r in an.get("cases") or ():
        for a in r["arms"]:
            if a not in arms:
                arms.append(a)
    head = (
        f"| case | stratum | variant | dtype | {base} median us | "
        + " | ".join(f"{a}/{base} median (95% interval)" for a in arms)
        + " | note |"
    )
    lines.append(head)
    lines.append("|" + "---|" * (6 + len(arms)))
    for r in an.get("cases") or ():
        cells = []
        for a in arms:
            x = r["arms"].get(a)
            if not x:
                cells.append("refused" if a in r["refused"] else "-")
                continue
            cells.append(
                f"{_pct(x['median_ratio'])} ({_pct(x['ci95'][0])} .. {_pct(x['ci95'][1])})"
            )
        a_us = "-" if r["base_us_median"] is None else f"{r['base_us_median']:.1f}"
        note = f"EXCLUDED: {r['excluded']}" if r["excluded"] else ""
        lines.append(
            f"| {r['case']} | {r['stratum']} | {r['variant']} | {r['dtype']} | {a_us} | "
            + " | ".join(cells)
            + f" | {note} |"
        )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# JSON-flow YAML out (no PyYAML needed) and structured in
# --------------------------------------------------------------------------- #


def _finite(obj: Any) -> Any:
    """``obj`` as plain JSON types: non-finite floats become null."""
    if isinstance(obj, Mapping):
        return {str(k): _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return str(obj)


def dump_flow_yaml(obj: Any) -> str:
    """A result as JSON-flow YAML: a JSON document, which is valid YAML 1.2 and
    reads back with PyYAML or, on a host without it, with :func:`load_structured`.
    ASCII only (non-ASCII text is written as ``\\u`` escapes)."""
    return json.dumps(_finite(obj), indent=1) + "\n"


# ---- a reader for the YAML subset the shipped shape files use -------------- #


def _delta(text: str) -> int:
    """Bracket depth change of ``text``, ignoring quoted text."""
    depth, quote, i = 0, "", 0
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == "\\" and quote == '"':
                i += 1
            elif ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        i += 1
    return depth


def _strip_comment(line: str) -> str:
    quote, i = "", 0
    while i < len(line):
        ch = line[i]
        if quote:
            if ch == "\\" and quote == '"':
                i += 1
            elif ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1] in " \t"):
            return line[:i]
        i += 1
    return line


def _plain(text: str) -> Any:
    """A plain YAML scalar: null, bool, int, float, else the text."""
    t = text.strip()
    low = t.lower()
    if t == "" or t == "~" or low == "null":
        return None
    if low in ("true", "false"):
        return low == "true"
    digits = t.lstrip("+-")
    if digits.isdigit():
        return int(t)
    if ("." in digits or "e" in digits.lower()) and any(ch.isdigit() for ch in digits):
        try:
            return float(t)
        except ValueError:
            pass
    return t


class _Flow:
    """Recursive-descent reader of one YAML flow node (``{..}``, ``[..]``, a quoted
    or a plain scalar)."""

    def __init__(self, text: str) -> None:
        self.s = text
        self.i = 0

    def _ws(self) -> None:
        while self.i < len(self.s) and self.s[self.i] in " \t":
            self.i += 1

    def _quoted(self) -> str:
        q = self.s[self.i]
        j = self.i + 1
        while j < len(self.s):
            if self.s[j] == "\\" and q == '"':
                j += 2
                continue
            if self.s[j] == q:
                if q == "'" and self.s[j : j + 2] == "''":
                    j += 2
                    continue
                break
            j += 1
        else:
            raise ValueError(f"unterminated quoted string in {self.s!r}")
        raw = self.s[self.i : j + 1]
        self.i = j + 1
        return json.loads(raw) if q == '"' else raw[1:-1].replace("''", "'")

    def _until(self, stops: str) -> str:
        j = self.i
        while j < len(self.s) and self.s[j] not in stops:
            j += 1
        raw = self.s[self.i : j]
        self.i = j
        return raw

    def value(self) -> Any:
        self._ws()
        if self.i >= len(self.s):
            return None
        ch = self.s[self.i]
        if ch == "{":
            return self._mapping()
        if ch == "[":
            return self._seq()
        if ch in "\"'":
            return self._quoted()
        raw = self._until(",}]")
        if raw.lstrip()[:1] in ("&", "*", "!", "|", ">", "%", "@") or ": " in raw:
            raise ValueError(
                f"unsupported YAML (anchor, tag, block scalar or nested mapping): {raw!r}"
            )
        return _plain(raw)

    def _mapping(self) -> dict[str, Any]:
        self.i += 1
        out: dict[str, Any] = {}
        self._ws()
        while self.i < len(self.s) and self.s[self.i] != "}":
            if self.s[self.i] in "\"'":
                key = self._quoted()
            else:
                key = self._until(":,}").strip()
            self._ws()
            if self.i < len(self.s) and self.s[self.i] == ":":
                self.i += 1
                self._ws()
                out[key] = (
                    None if self.s[self.i : self.i + 1] in (",", "}") else self.value()
                )
            else:
                out[key] = None
            self._ws()
            if self.i < len(self.s) and self.s[self.i] == ",":
                self.i += 1
                self._ws()
        if self.i >= len(self.s):
            raise ValueError(f"unterminated mapping in {self.s!r}")
        self.i += 1
        return out

    def _seq(self) -> list[Any]:
        self.i += 1
        out: list[Any] = []
        self._ws()
        while self.i < len(self.s) and self.s[self.i] != "]":
            out.append(self.value())
            self._ws()
            if self.i < len(self.s) and self.s[self.i] == ",":
                self.i += 1
                self._ws()
        if self.i >= len(self.s):
            raise ValueError(f"unterminated list in {self.s!r}")
        self.i += 1
        return out


def _parse_flow_text(text: str) -> Any:
    f = _Flow(text.strip())
    val = f.value()
    f._ws()
    if f.i != len(f.s):
        raise ValueError(f"cannot parse {text!r} (stopped at column {f.i})")
    return val


def _split_key(text: str) -> tuple[str, str]:
    quote, depth, i = "", 0, 0
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == "\\" and quote == '"':
                i += 1
            elif ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        elif ch == ":" and depth == 0 and (i + 1 == len(text) or text[i + 1] == " "):
            key = text[:i].strip()
            if key[:1] in "\"'":
                key = _parse_flow_text(key)
            return key, text[i + 1 :]
        i += 1
    raise ValueError(f"expected 'key: value', got {text!r}")


def _logical_lines(text: str) -> list[tuple[int, str]]:
    """``(indent, text)`` per logical line: comments and blank lines dropped, a
    flow collection continued over several lines joined into one."""
    out: list[tuple[int, str]] = []
    pending: tuple[int, str] | None = None
    depth = 0
    for raw in text.splitlines():
        body = _strip_comment(raw).rstrip()
        if not body.strip():
            continue
        if pending is not None:
            pending = (pending[0], pending[1] + " " + body.strip())
            depth += _delta(body)
            if depth <= 0:
                out.append(pending)
                pending, depth = None, 0
            continue
        if body.lstrip(" ") != body.lstrip():
            raise ValueError("tab characters in indentation are not supported")
        ind = len(body) - len(body.lstrip(" "))
        depth = _delta(body)
        if depth > 0:
            pending = (ind, body.strip())
        else:
            out.append((ind, body.strip()))
    if pending is not None:
        raise ValueError("unterminated flow collection at end of file")
    return out


def _parse_block(lines: list[tuple[int, str]], i: int) -> tuple[Any, int]:
    ind = lines[i][0]
    if lines[i][1].startswith("- ") or lines[i][1] == "-":
        seq: list[Any] = []
        while (
            i < len(lines)
            and lines[i][0] == ind
            and (lines[i][1].startswith("- ") or lines[i][1] == "-")
        ):
            item = lines[i][1][1:].strip()
            i += 1
            if item:
                seq.append(_parse_flow_text(item))
            elif i < len(lines) and lines[i][0] > ind:
                val, i = _parse_block(lines, i)
                seq.append(val)
            else:
                seq.append(None)
        return seq, i
    mapping: dict[str, Any] = {}
    while i < len(lines) and lines[i][0] == ind and not lines[i][1].startswith("- "):
        key, rest = _split_key(lines[i][1])
        rest = rest.strip()
        i += 1
        if rest:
            mapping[key] = _parse_flow_text(rest)
        elif i < len(lines) and (
            lines[i][0] > ind or (lines[i][0] == ind and lines[i][1].startswith("- "))
        ):
            mapping[key], i = _parse_block(lines, i)
        else:
            mapping[key] = None
    return mapping, i


def parse_yaml_subset(text: str) -> Any:
    """Read the YAML a hand-written shape file uses, without PyYAML: ``#``
    comments, block mappings, block sequences, flow ``{..}`` / ``[..]`` nodes
    (also spread over lines), quoted and plain scalars. Anything else (anchors,
    block scalars, a mapping inside a ``- `` item) raises ``ValueError``."""
    lines = _logical_lines(text)
    if not lines:
        return None
    if len(lines) == 1 and lines[0][1][:1] in "{[":
        return _parse_flow_text(lines[0][1])
    val, i = _parse_block(lines, 0)
    if i != len(lines):
        raise ValueError(f"cannot parse line {lines[i][1]!r}")
    return val


def load_structured(path: str) -> Any:
    """A shape file or a harness result: ``.json``; ``.yaml`` / ``.yml`` (PyYAML when
    it is installed; else JSON-flow YAML; else the subset :func:`parse_yaml_subset`
    reads); anything else is a log with ``AB_META`` / ``AB_ROW`` / ``AB_DONE`` lines."""
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    if path.endswith((".yaml", ".yml")):
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        try:
            import yaml  # type: ignore
        except ImportError:
            yaml = None
        if yaml is not None:
            return yaml.safe_load(text)
        try:
            return json.loads(text)
        except ValueError:
            pass
        try:
            return parse_yaml_subset(text)
        except ValueError as exc:
            raise SystemExit(f"{path}: not readable without PyYAML ({exc})") from exc
    meta: dict[str, Any] = {}
    cases = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "AB_ROW " in line:
                cases.append(json.loads(line.split("AB_ROW ", 1)[1]))
            elif "AB_META " in line:
                meta = json.loads(line.split("AB_META ", 1)[1])
            elif "AB_DONE " in line:
                meta.update(json.loads(line.split("AB_DONE ", 1)[1]))
    if not cases:
        raise SystemExit(f"{path}: no AB_ROW lines and not a .json/.yaml file")
    return {"meta": meta, "cases": cases}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Analyse a dense attention A/B result and print the speed bar verdict.",
    )
    ap.add_argument(
        "result", help="harness result (.yaml / .json), or a log with AB_ROW lines"
    )
    ap.add_argument(
        "--baseline",
        default="",
        help="arm every ratio is against (default: the result's)",
    )
    ap.add_argument(
        "--control",
        default="",
        help="copy of the baseline tree (default: the result's)",
    )
    ap.add_argument("--group-by", choices=sorted(GROUP_CHOICES), default="body-layout")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--markdown", action="store_true", help="also print the per-case tables"
    )
    ap.add_argument("--out-md", default="", help="write the markdown tables")
    args = ap.parse_args(argv)
    result = load_structured(args.result)
    try:
        an = analyze(
            result,
            n_boot=args.n_boot,
            seed=args.seed,
            baseline=args.baseline or None,
            control=args.control or None,
            group_by=GROUP_CHOICES[args.group_by],
        )
    except ValueError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    bar = bar_check(an)
    md = to_markdown(an)
    if args.out_md:
        with open(args.out_md, "w", encoding="utf-8") as fh:
            fh.write(md)
    if args.markdown:
        sys.stdout.write(md + "\n")
    sys.stdout.write("\n".join(bar["lines"]) + "\n")
    return int(bar["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
