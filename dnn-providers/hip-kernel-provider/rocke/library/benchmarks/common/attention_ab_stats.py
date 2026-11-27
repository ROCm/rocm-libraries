# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Statistics and the speed bar of the stride-aware dense attention A/B harness.

Pure Python (no numpy, no torch; PyYAML only when it is installed): importable and
runnable on a bare CPU host. Used by ``attention_dense_strided_ab.py`` (the driver)
and by its ``bar`` subcommand; ``python attention_ab_stats.py RESULT`` runs the same
check.

Input: a harness result (a ``.yaml`` written as JSON-flow YAML, which is valid
YAML and readable without PyYAML; a ``.json``; or a log holding the driver's
``AB_ROW {json}`` lines, so a run that died half way can still be analysed).

Result layout::

    {"meta": {"baseline": L, "control": L, "arms": {label: {"tree": ...}},
              "expected_cases": [case, ...], "complete": bool, ...},
     "cases": [{"case": "<shape>/<layout>", "shape_id", "layout", "dtype",
                "body", "variant", "status", "spec_identity_equal",
                "arms": {label: {"status": "ok" | "refused" | "error", "reason"}},
                "measurements": [{"round", "pos", "arm", "us"}],
                "check_pre": {label: {"pass": bool, ...}}, "check_post": {...}}]}

Method (every constant is a method constant, none is a measured number):

- Per case (a shape on a layout) and arm X other than the baseline: the ratio
  t_X / t_baseline of each round (arms are measured in a mirrored order inside
  a round), its median, and a 95% interval of that median (bootstrap over
  rounds). A case is not judged when an arm that ran it has fewer rounds paired with
  the baseline than ``MIN_ROUNDS`` (an arm whose every time is unusable has none).
- Cases are grouped into strata (default: body and layout). The control arm is the
  baseline's own code objects loaded a second time in the same process: its ratio
  to the baseline measures the noise of replay and of the position in a round, not
  the placement effect of a different binary. Noise band of a stratum = max(1%,
  95th percentile over its cases of |median control/baseline - 1|).
- Geomean of the per-case medians per arm, with a 95% interval from a two-level
  bootstrap (cases with replacement, then rounds within each).
- A case is "wholly above the band" when the lower end of its own 95% interval
  is above 1 + band.
- Verdict per (stratum, arm): "too noisy" when the p95 above exceeds 2%; "pass"
  when the geomean's upper bound is below +1% and no case is wholly above the
  band; else "fail". "fail (check)" / "fail (spec)" / "fail (error)" / "fail
  (rounds)" when a case of the stratum failed or lacked its output check, ran
  another kernel spec in some arm, raised, or has too few paired rounds;
  "fail (refused)" when the arm declined a case the baseline served (a lost case,
  not a neutral one); "refused" when the arm declined every case of the stratum
  and the baseline did too; "no data" otherwise. A case with any such problem never
  enters a geomean or a band. A case an arm refuses is excluded from that arm's
  ratios only.
- The bar over all verdicts is PASS only for a run that finished (every expected
  case recorded) and measured on a device. A fail verdict wins over everything; an
  unfinished or case-short run reads INCOMPLETE, a dry run (synthetic times) reads
  NOT A MEASUREMENT, and neither exits 0.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

__all__ = [
    "BAND_FLOOR",
    "CI_HIGH",
    "CI_LOW",
    "DEFAULT_GROUP_BY",
    "DRY_RUN_TIMING",
    "EXIT_FAIL",
    "EXIT_INCOMPLETE",
    "EXIT_NOISY",
    "EXIT_NO_DATA",
    "EXIT_PASS",
    "EXIT_SYNTHETIC",
    "GEOMEAN_BOUND",
    "GROUP_CHOICES",
    "MIN_ROUNDS",
    "NEUTRAL_VERDICTS",
    "NOISY_P95",
    "NOISY_VERDICT",
    "PASS_VERDICTS",
    "analyze",
    "bar_check",
    "bootstrap_medians",
    "check_same_compiler",
    "check_same_tree",
    "classify_case",
    "compiler_key",
    "decide",
    "dump_flow_yaml",
    "failed_check_arms",
    "geomean",
    "hierarchical_geomean_ci",
    "load_structured",
    "main",
    "median",
    "noise_band",
    "paired_ratios",
    "parse_yaml_subset",
    "percentile",
    "report_markdown",
    "run_integrity",
    "same_code_arms",
    "to_markdown",
    "unchecked_arms",
]

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
MIN_ROUNDS = 24  # paired rounds a case needs before it is judged (also the driver's)
# ``meta.timing_method`` of a run whose times are made up: it never reads as a pass.
DRY_RUN_TIMING = "dry-run (synthetic)"
PASS_VERDICTS = ("pass",)
NEUTRAL_VERDICTS = ("refused", "no data")
NOISY_VERDICT = "too noisy"


# --------------------------------------------------------------------------- #
# Small numeric helpers
# --------------------------------------------------------------------------- #


# The standard library's median: the mean of the middle pair for an even count, and
# StatisticsError (a ValueError) on an empty input.
median = statistics.median


def geomean(xs: Sequence[float]) -> float:
    """Geometric mean of positive numbers (``statistics.geometric_mean``).

    The standard library averages the logs with ``math.fsum``, so the result can differ
    from a plain left-to-right sum of logs in the last bit. A function, not an alias,
    so that an empty input, or a value that is not a positive number, raises
    ``ValueError`` on every Python version: from 3.13 the standard library returns 0.0
    for an input that holds a zero, and a ratio of zero is a broken time, never a
    mean."""
    vals = list(xs)
    if not vals or not all(x > 0 for x in vals):
        raise ValueError(
            f"geomean failed - xs={vals!r}, expected a non-empty sequence of positive "
            "numbers"
        )
    return statistics.geometric_mean(vals)


def percentile(xs: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (numpy's default), q in [0, 100].

    Kept by hand: ``statistics.quantiles`` only offers the cut points k/n of an
    integer n (each call builds the whole table), rounds the interpolation
    differently in the last digit, and refuses a single value."""
    s = sorted(xs)
    if not s:
        raise ValueError(
            f"percentile failed - xs={list(xs)!r}, expected a non-empty sequence"
        )
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * (q / 100.0)
    lo = math.floor(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] + (s[hi] - s[lo]) * frac


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
    short_cases: Sequence[str] = (),
) -> str:
    """Verdict string for one (stratum, arm).

    ``lost_cases``: cases the arm refused although the baseline served them.
    ``short_cases``: cases with fewer paired rounds than ``MIN_ROUNDS``."""
    if check_failures:
        return "fail (check)"
    if spec_mismatches:
        return "fail (spec)"
    if errors:
        return "fail (error)"
    if short_cases:
        return "fail (rounds)"
    if lost_cases:
        return "fail (refused)"
    if n_shapes == 0 or geo_hi is None:
        return "refused" if n_refused else "no data"
    if p95 is None:
        return "no data"
    if p95 > NOISY_P95:
        return NOISY_VERDICT
    if geo_hi < 1.0 + GEOMEAN_BOUND and not wholly_above:
        return "pass"
    return "fail"


# --------------------------------------------------------------------------- #
# Analysis of a harness result
# --------------------------------------------------------------------------- #


def _is_check(record: Any) -> bool:
    """Whether ``record`` is an output-check record (a mapping with a boolean
    ``pass``)."""
    return isinstance(record, Mapping) and isinstance(record.get("pass"), bool)


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


def unchecked_arms(case: Mapping[str, Any]) -> list[str]:
    """Timed arms with no output-check record before or after timing.

    A case with no check on record was not checked: it must not read as a case that
    passed its check."""
    timed: list[str] = []
    for m in case.get("measurements") or ():
        if str(m["arm"]) not in timed:
            timed.append(str(m["arm"]))
    return [
        label
        for label in timed
        if not all(
            _is_check((case.get(phase) or {}).get(label))
            for phase in ("check_pre", "check_post")
        )
    ]


def classify_case(case: Mapping[str, Any]) -> tuple[str, list[str]]:
    """``(kind, arms)``: kind is ``ok`` (judged), or the reason the case never
    enters a geomean: ``check`` (an arm's output check failed or is missing),
    ``spec`` (the arms ran different kernel specs), ``error`` (an arm raised),
    ``skipped`` (the layout does not apply to the shape, or the baseline declined
    it)."""
    failed = failed_check_arms(case)
    if failed:
        return "check", failed
    if case.get("spec_identity_equal") is False:
        return "spec", []
    errs = [
        str(label)
        for label, a in (case.get("arms") or {}).items()
        if (a or {}).get("status") == "error"
    ]
    if errs or case.get("status") == "error":
        return "error", errs
    if case.get("status") in ("not_applicable", "refused"):
        return "skipped", []
    unchecked = unchecked_arms(case)
    if unchecked:
        return "check", unchecked
    return "ok", []


def _ran_arms(case: Mapping[str, Any], arms: Sequence[str]) -> list[str]:
    """The arms of ``arms`` that ran the case: status ``ok`` in its arm records, or a
    measurement on record (usable or not).

    Each of them owes the case its paired rounds, so an arm whose every time is null,
    NaN or not positive is a shortfall, not an arm that quietly leaves the case."""
    records = case.get("arms") or {}
    timed = {str(m["arm"]) for m in case.get("measurements") or ()}
    return [
        a for a in arms if (records.get(a) or {}).get("status") == "ok" or a in timed
    ]


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
            f"control {control!r} ({tc}) is not the same tree as baseline "
            f"{baseline!r} ({tb}): a control must be a second copy of the "
            "baseline's code"
        )


def compiler_key(compiler: Mapping[str, Any] | None) -> tuple[Any, Any] | None:
    """What tells two compilers apart across arms: the LLVM version and the COMGR
    library that was loaded. ``None`` when nothing is recorded (a dry run)."""
    if not compiler:
        return None
    version = compiler.get("llvm_version")
    return (tuple(version) if version else None, compiler.get("comgr_path"))


def check_same_compiler(arms: Mapping[str, Mapping[str, Any]], baseline: str) -> None:
    """Refuse arms compiled by another compiler than the baseline's.

    ``arms`` maps an arm label to its meta (its ``compiler`` as the export recorded
    it). A different compiler moves every ratio by more than the bar allows, and the
    control arm cannot show it (it reuses the baseline's code objects), so it would
    read as a code change."""
    want = compiler_key((arms.get(baseline) or {}).get("compiler"))
    for label, meta in arms.items():
        got = compiler_key((meta or {}).get("compiler"))
        if got != want:
            raise ValueError(
                f"arm {label!r} was compiled with {got}, baseline {baseline!r} with "
                f"{want}: a compiler change would read as a code change"
            )


def run_integrity(result: Mapping[str, Any]) -> dict[str, Any]:
    """How the run ended, from the result's own record.

    ``complete`` holds only when the driver marked the loop as run to its end AND
    every case it listed at the start is on record; a result with no such record
    (a hand-made file, a log of a crashed run) is not complete. ``synthetic`` marks
    made-up times (a dry run)."""
    meta = result.get("meta") or {}
    expected = meta.get("expected_cases")
    recorded = [str(c.get("case")) for c in result.get("cases") or ()]
    have = set(recorded)
    missing = [] if expected is None else [c for c in expected if c not in have]
    finished = meta.get("complete") is True
    return {
        "expected_cases": None if expected is None else len(expected),
        "recorded_cases": len(recorded),
        "missing_cases": missing,
        "finished": finished,
        "complete": finished and expected is not None and not missing,
        "synthetic": str(meta.get("timing_method") or "").startswith("dry-run"),
    }


def _code_hash(case: Mapping[str, Any], label: str) -> str | None:
    return ((case.get("arms") or {}).get(label) or {}).get("code_sha256")


def same_code_arms(
    cases: Sequence[Mapping[str, Any]], baseline: str, arms: Sequence[str]
) -> list[str]:
    """Arms whose recorded code objects equal the baseline's in every case both ran.

    Their ratios then measure noise, not a code change (``--arm a=T --arm b=T``). It
    is a note, not a verdict: a change to the host code alone runs the same kernels."""
    same = []
    for arm in arms:
        pairs = [(_code_hash(c, arm), _code_hash(c, baseline)) for c in cases]
        seen = [mine == base for mine, base in pairs if mine and base]
        if seen and all(seen):
            same.append(arm)
    return same


def analyze(
    result: Mapping[str, Any],
    n_boot: int = 10000,
    seed: int = 0,
    baseline: str | None = None,
    control: str | None = None,
    group_by: Sequence[str] = DEFAULT_GROUP_BY,
    min_rounds: int = MIN_ROUNDS,
) -> dict[str, Any]:
    """Per-case and per-stratum statistics plus verdicts (YAML-able dict)."""
    rng = random.Random(seed)
    meta = result.get("meta") or {}
    baseline = baseline or meta.get("baseline")
    control = control or meta.get("control")
    if not baseline or not control:
        raise ValueError(
            f"analyze failed - baseline={baseline!r} control={control!r}, expected "
            "both labels (name them, or run on a result whose meta does)"
        )
    if baseline == control:
        raise ValueError(
            f"analyze failed - control={control!r} baseline={baseline!r}, expected "
            "different arm labels (the control is a second copy of the baseline)"
        )
    check_same_tree(meta, baseline, control)
    check_same_compiler(meta.get("arms") or {}, baseline)
    baseline_meta = (meta.get("arms") or {}).get(baseline) or {}
    cases = list(result.get("cases") or ())
    labels = _arm_labels(result)
    test_arms = [a for a in labels if a not in (baseline, control)]

    rows: list[dict[str, Any]] = []
    pools: dict[tuple[str, str], list[float]] = {}
    for c in cases:
        kind, who = classify_case(c)
        meas = c.get("measurements") or []
        ratios_by_arm: dict[str, list[float]] = {}
        if kind == "ok":
            for arm in [control] + test_arms:
                ratios = paired_ratios(meas, arm, baseline)
                if ratios:
                    ratios_by_arm[arm] = ratios
            ran = _ran_arms(c, [baseline, control, *test_arms])
            # Ratios exist only against a baseline that ran; when it did, every other
            # arm that ran needs min_rounds of them, zero usable rounds included.
            short = (
                [
                    a
                    for a in ran
                    if a != baseline and len(ratios_by_arm.get(a, ())) < min_rounds
                ]
                if baseline in ran
                else []
            )
            if short:
                kind, who = "rounds", short
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
                str(label): (a or {}).get("reason")
                for label, a in (c.get("arms") or {}).items()
                if (a or {}).get("status") == "refused"
            },
            # The baseline ran the case (its output check may still have failed): only
            # then is a test arm's refusal a lost case.
            "baseline_served": ((c.get("arms") or {}).get(baseline) or {}).get("status")
            in ("ok", "check_failed"),
            "base_us_median": None,
            "arms": {},
        }
        base_times = [
            float(m["us"]) for m in meas if m["arm"] == baseline and m.get("us")
        ]
        if base_times:
            row["base_us_median"] = median(base_times)
        if kind == "ok":
            for arm, ratios in ratios_by_arm.items():
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
            shortfall = [
                r["case"]
                for r in srows
                if r["excluded"] == "rounds" and relevant & set(r["problem_arms"])
            ]
            tally = {
                "n_refused": len(refused_cases),
                "refused_cases": refused_cases,
                "lost_cases": lost,
                "check_failures": chk,
                "spec_mismatches": spc,
                "errors": err,
                "short_cases": shortfall,
            }
            verdict_args = {
                "p95": p95,
                "check_failures": chk,
                "spec_mismatches": spc,
                "errors": err,
                "n_refused": len(refused_cases),
                "lost_cases": lost,
                "short_cases": shortfall,
            }
            if not have:
                entry["arms"][arm] = {
                    "n_cases": 0,
                    **tally,
                    "verdict": decide(
                        geo_hi=None, wholly_above=[], n_shapes=0, **verdict_args
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
                **tally,
                "geomean": gm,
                "geomean_ci95": [lo, hi],
                "cases_wholly_above_band": above,
                "cases_median_above_band": median_above,
                "verdict": decide(
                    geo_hi=hi, wholly_above=above, n_shapes=len(have), **verdict_args
                ),
            }
        strata[key] = entry

    return {
        "method": {
            "ratio": "per round t_arm / t_baseline over paired rounds; per-case median",
            "case_ci": "95% percentile bootstrap over rounds of the median",
            "geomean_ci": "95% two-level bootstrap (cases, then rounds)",
            "band": (
                f"max({BAND_FLOOR:.0%}, p95 over cases of "
                "|median control/baseline - 1|)"
            ),
            "wholly_above": "case's 95% interval lower end > 1 + band",
            "too_noisy": f"p95 > {NOISY_P95:.0%}",
            "pass": (
                f"geomean upper bound < 1 + {GEOMEAN_BOUND:.0%} and no case wholly "
                "above the band"
            ),
            "min_rounds": min_rounds,
            "strata": "/".join(group_by) or "all",
            "n_boot": n_boot,
            "seed": seed,
        },
        "baseline": baseline,
        "control": control,
        "test_arms": test_arms,
        "meta": {
            **{
                k: meta.get(k)
                for k in (
                    "timing_method",
                    "device_name",
                    "cu_count",
                    "rounds",
                    "label",
                    "arch",
                    "arms",
                )
            },
            # The driver records the compiler per arm, not at the top of the meta.
            "compiler": meta.get("compiler") or baseline_meta.get("compiler"),
        },
        "run": run_integrity(result),
        "same_code_arms": same_code_arms(cases, baseline, test_arms),
        "output_hash_differs": [
            str(c.get("case"))
            for c in cases
            if False in (c.get("o_hash_equal") or {}).values()
        ],
        "all_output_checks_passed": not any(r["excluded"] == "check" for r in rows),
        "strata": strata,
        "cases": rows,
    }


# --------------------------------------------------------------------------- #
# The bar check
# --------------------------------------------------------------------------- #

EXIT_PASS, EXIT_FAIL, EXIT_NO_DATA, EXIT_NOISY = 0, 1, 2, 3
EXIT_INCOMPLETE, EXIT_SYNTHETIC = 4, 5
_KNOWN_VERDICTS = (*PASS_VERDICTS, *NEUTRAL_VERDICTS, NOISY_VERDICT)


def _run_lines(run: Mapping[str, Any]) -> list[str]:
    """The bar's lines about how the run ended and what its times are."""
    if run.get("expected_cases") is None:
        seen = f"{run.get('recorded_cases', 0)} cases recorded, no expected list"
    else:
        seen = (
            f"{run.get('recorded_cases', 0)} of {run['expected_cases']} expected "
            "cases recorded"
        )
    ended = "finished" if run.get("finished") else "no record that it finished"
    lines = [f"run: {seen}; {ended}"]
    if run.get("missing_cases"):
        lines.append("  missing: " + ", ".join(run["missing_cases"]))
    if run.get("synthetic"):
        lines.append(
            "timing: dry run (synthetic times, stubbed output checks): "
            "not a measurement"
        )
    return lines


def bar_check(an: Mapping[str, Any]) -> dict[str, Any]:
    """The speed bar on an analysis.

    FAIL when any verdict is a ``fail*`` or is not a verdict this module produces.
    Otherwise INCOMPLETE when the run did not finish or cases are missing (or the
    analysis has no record of it), NOT A MEASUREMENT when the times are synthetic (a
    dry run), INCONCLUSIVE when some stratum is too noisy, PASS when every verdict is
    ``pass`` or neutral (``refused`` / ``no data``) and at least one is ``pass``, and
    NO DATA when nothing could be judged. Only PASS exits 0."""
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
    run = an.get("run") or {}
    unknown = [
        v for v in verdicts if not v.startswith("fail") and v not in _KNOWN_VERDICTS
    ]
    if unknown or any(v.startswith("fail") for v in verdicts):
        word, code = "FAIL", EXIT_FAIL
    elif not run.get("complete"):
        word, code = "INCOMPLETE", EXIT_INCOMPLETE
    elif run.get("synthetic"):
        word, code = "NOT A MEASUREMENT (dry run)", EXIT_SYNTHETIC
    elif any(v == NOISY_VERDICT for v in verdicts):
        word, code = "INCONCLUSIVE (too noisy)", EXIT_NOISY
    elif any(v in PASS_VERDICTS for v in verdicts):
        word, code = "PASS", EXIT_PASS
    else:
        word, code = "NO DATA", EXIT_NO_DATA
    base, ctrl = an.get("baseline"), an.get("control")
    test_arms = ",".join(an.get("test_arms") or []) or "-"
    lines = [
        f"bar check: baseline={base} control={ctrl} arms under test={test_arms}",
        *_run_lines(run),
        "all output checks passed (every arm, before and after timing): "
        f"{an.get('all_output_checks_passed')}",
    ]
    if unknown:
        lines.append(f"unknown verdicts (a bug in the analysis): {', '.join(unknown)}")
    if an.get("same_code_arms"):
        lines.append(
            f"note: {', '.join(an['same_code_arms'])} loaded the same code objects "
            f"as {base} in every case; their ratios measure noise, not a code change"
        )
    if an.get("output_hash_differs"):
        lines.append(
            "note: output hashes differ between arms in "
            f"{', '.join(an['output_hash_differs'])} (reported, not a gate)"
        )
    for key, e in (an.get("strata") or {}).items():
        c = e["control"]
        lines.append(
            f"stratum {key}: {e['n_judged']} of {e['n_cases']} cases judged; "
            f"{ctrl}/{base} p95|dev|={_abs_pct(c['p95_abs_dev'])} "
            f"band={_abs_pct(c['band'])}"
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
                    f"  {arm}/{base}: no judged case (refused in {a['n_refused']})"
                    f"{lost_txt} -> {a['verdict']}"
                )
                continue
            lo, hi = a["geomean_ci95"]
            refused = (
                f" (refused in {a['n_refused']}{lost_txt})" if a["n_refused"] else ""
            )
            above = ", ".join(a["cases_wholly_above_band"]) or "none"
            lines.append(
                f"  {arm}/{base}: geomean {_pct(a['geomean'])} "
                f"[{_pct(lo)} .. {_pct(hi)}] over {a['n_cases']} cases{refused}; "
                f"wholly above band: {above} -> {a['verdict']}"
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
    """The analysis as markdown: a header line, the per-stratum table, the per-case
    table."""
    lines = []
    meta = an.get("meta") or {}
    base, ctrl = an.get("baseline"), an.get("control")
    lines.append(
        f"Timing: {meta.get('timing_method')}; device: {meta.get('device_name')} "
        f"({meta.get('cu_count')} CUs); baseline: {base}; control: {ctrl}; "
        f"rounds: {meta.get('rounds')}; all output checks passed: "
        f"{an.get('all_output_checks_passed')}"
    )
    lines.append("")
    lines.append(
        f"| stratum | cases judged | {ctrl}/{base} p95 | band | arm | "
        f"arm/{base} geomean | 95% interval | wholly above band | verdict |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for key, e in (an.get("strata") or {}).items():
        c = e["control"]
        noise = f"{_abs_pct(c['p95_abs_dev'])} | {_abs_pct(c['band'])} |"
        arms = e.get("arms") or {}
        if not arms:
            judged = f"{e['n_judged']} of {e['n_cases']}"
            lines.append(f"| {key} | {judged} | {noise} - | - | - | - | no data |")
        for arm, a in arms.items():
            # The cases the arm's geomean is over, of all the stratum holds: a case
            # the arm refused, or one excluded or short of rounds, is not among them.
            judged = f"{a.get('n_cases', 0)} of {e['n_cases']}"
            if a.get("n_cases", 0) == 0:
                lines.append(
                    f"| {key} | {judged} | {noise} {arm} | - | - | - | "
                    f"{a['verdict']} |"
                )
                continue
            lo, hi = a["geomean_ci95"]
            lines.append(
                f"| {key} | {judged} | {noise} {arm} | {_pct(a['geomean'])} | "
                f"{_pct(lo)} .. {_pct(hi)} | "
                f"{', '.join(a['cases_wholly_above_band']) or 'none'} | "
                f"**{a['verdict']}** |"
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
                f"{_pct(x['median_ratio'])} "
                f"({_pct(x['ci95'][0])} .. {_pct(x['ci95'][1])})"
            )
        a_us = "-" if r["base_us_median"] is None else f"{r['base_us_median']:.1f}"
        note = f"EXCLUDED: {r['excluded']}" if r["excluded"] else ""
        lines.append(
            f"| {r['case']} | {r['stratum']} | {r['variant']} | {r['dtype']} | "
            f"{a_us} | " + " | ".join(cells) + f" | {note} |"
        )
    return "\n".join(lines) + "\n"


def report_markdown(an: Mapping[str, Any], bar: Mapping[str, Any]) -> str:
    """The text of a stats file (the driver's ``FILE.stats.md``, ``bar --out-md``): the
    tables of :func:`to_markdown`, then the bar's lines.

    The tables give per-stratum verdicts only; the bar's lines say how many expected
    cases were recorded, whether the run finished, whether its times are synthetic, and
    the overall verdict. A file read on its own (attached to a review, say) must carry
    them, or an unfinished run or a dry run reads as a pass."""
    return to_markdown(an) + "\n" + "\n".join(bar["lines"]) + "\n"


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
        """The flow node at the cursor: a mapping, a list or a scalar."""
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
                "unsupported YAML (anchor, tag, block scalar or nested mapping): "
                f"{raw!r}"
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
    """A shape file or a harness result.

    ``.json``; ``.yaml`` / ``.yml`` (read as JSON first, which is what the driver
    writes and what PyYAML reads wrongly for a number like ``1e-05``; then PyYAML when
    it is installed; else the subset :func:`parse_yaml_subset` reads); anything else
    is a log with ``AB_META`` / ``AB_ROW`` / ``AB_DONE`` lines."""
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    if path.endswith((".yaml", ".yml")):
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        try:
            return json.loads(text)
        except ValueError:
            pass
        try:
            import yaml  # type: ignore
        except ImportError:
            yaml = None
        if yaml is not None:
            try:
                return yaml.safe_load(text)
            except yaml.YAMLError as exc:
                raise SystemExit(f"{path}: not readable YAML ({exc})") from exc
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


_EXIT_CODES_TEXT = (
    f"Exit codes: {EXIT_PASS} PASS; {EXIT_FAIL} FAIL (an arm failed the bar, its "
    f"output check or a case); {EXIT_NO_DATA} no verdict: NO DATA, or the input was "
    "refused (a bad flag, a file that is missing, unreadable or not a result, an "
    f"analysis that cannot be made; the reason is on stderr); {EXIT_NOISY} too noisy; "
    f"{EXIT_INCOMPLETE} INCOMPLETE (unfinished or cases missing); {EXIT_SYNTHETIC} dry "
    "run (not a measurement). A Python traceback is a bug in the harness, not a "
    f"verdict; it exits {EXIT_FAIL}, as any uncaught exception does."
)


def _refused(reason: str) -> int:
    """Report input that cannot be analysed: the reason on stderr, no verdict."""
    print(f"refused: {reason}", file=sys.stderr)
    return EXIT_NO_DATA


def main(argv: Sequence[str] | None = None) -> int:
    """Analyse a result file or log, print the bar, and return its exit code.

    Input that cannot be analysed returns ``EXIT_NO_DATA`` with ``refused: <reason>``
    on stderr, so a script never reads it as the ``EXIT_FAIL`` of a slow or wrong
    arm. ``--out-md`` holds the tables and the bar's lines (:func:`report_markdown`)."""
    ap = argparse.ArgumentParser(
        description=(
            "Analyse a dense attention A/B result and print the speed bar verdict. "
            + _EXIT_CODES_TEXT
        ),
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
        "--min-rounds",
        type=int,
        default=MIN_ROUNDS,
        help="paired rounds a case needs to be judged (default: %(default)s)",
    )
    ap.add_argument(
        "--markdown", action="store_true", help="also print the per-case tables"
    )
    ap.add_argument(
        "--out-md",
        default="",
        help="write the markdown tables and the bar's lines (run state, verdict)",
    )
    args = ap.parse_args(argv)
    try:
        result = load_structured(args.result)
        if not isinstance(result, Mapping):
            raise ValueError(
                f"{args.result}: not a harness result (a mapping with meta and cases)"
            )
        an = analyze(
            result,
            n_boot=args.n_boot,
            seed=args.seed,
            baseline=args.baseline or None,
            control=args.control or None,
            group_by=GROUP_CHOICES[args.group_by],
            min_rounds=args.min_rounds,
        )
    except SystemExit as exc:  # load_structured refuses a file with its reason
        if not isinstance(exc.code, str):
            raise
        return _refused(exc.code)
    except (OSError, ValueError) as exc:  # missing, not JSON, refused by analyze
        return _refused(str(exc))
    bar = bar_check(an)
    if args.out_md:
        with open(args.out_md, "w", encoding="utf-8") as fh:
            fh.write(report_markdown(an, bar))
    if args.markdown:
        sys.stdout.write(to_markdown(an) + "\n")
    sys.stdout.write("\n".join(bar["lines"]) + "\n")
    return int(bar["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
