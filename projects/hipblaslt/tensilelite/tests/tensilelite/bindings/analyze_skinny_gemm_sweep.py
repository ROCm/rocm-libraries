# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Statistical analysis of sweep_skinny_gemm.py results to trim knob sweeps.

Reads a results YAML (top configs per case only) and the sweep input YAML used
to produce it. Re-expands every case to recover the constraint-valid
configuration pool, labels configs that appear in that case's top list, and
computes statistics **separately for the two VALU sweep shapes**:

  grid        grid3d: true  (3-D launch; group/numXcds/blocks are derived)
  persistent  grid3d: false (persistent tile order; sweeps group, numXcds, blocks)

Within each mode, every knob that the mode's input mapping actually sweeps gets
top rate, phi (value indicator vs is_top), and point-biserial r for numeric
knobs. Prune recommendations apply only to the matching defaults.sweep entry
(grid3d true vs false), not across modes.

VALU-only: mfma/lds configurations are skipped when results contain no MFMA
winners.

Run from the tensilelite directory:

  source ~/.tensile/bin/activate
  export PYTHONPATH=$PWD/tests/tensilelite/bindings
  python tests/tensilelite/bindings/analyze_skinny_gemm_sweep.py run.yml \\
      [--input sweep_skinny_gemm_shapes.yaml] [--trim-yaml trimmed_sweep.yaml] \\
      [--prune frequency|low-phi|never-top|none] [--min-trials 1000] [--min-trials-grid 200]

Default ``frequency`` uses **both** (1) top-slot frequency thresholds and (2) the
**phi** coefficient (indicator for this knob value vs is_top within the mode),
same test as ``low-phi``: drop when phi <= ``--max-phi``, p-value vs rest is
weak, and candidate top rate is below the mode baseline—unless the value is
among the ``--keep-at-least`` highest-frequency values for that knob.

Optional <results>.failed.yaml excludes build-failed identities from the pool.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple

import yaml

import sweep_skinny_gemm_config as sweep_config
from sweep_skinny_gemm_config import config_record, expand_case, load_input

try:
    from yaml import CSafeLoader as YamlLoader
except ImportError:
    from yaml import SafeLoader as YamlLoader

TOOL = "analyze_skinny_gemm_sweep.py"

ModeName = Literal["grid", "persistent"]
MODES: tuple[ModeName, ...] = ("grid", "persistent")

# Knobs swept in each VALU mapping (must match sweep_skinny_gemm_shapes.yaml).
GRID_KNOBS = ("bm", "tn", "lanesK", "waves", "prefetch", "upScales")
PERSIST_KNOBS = (
    "bm",
    "tn",
    "lanesK",
    "waves",
    "prefetch",
    "upScales",
    "group",
    "numXcds",
    "blocks",
)
NUMERIC_KNOBS = (
    "bm",
    "tn",
    "lanesK",
    "waves",
    "prefetch",
    "group",
    "numXcds",
    "blocks",
)

CASE_KEY_FIELDS = ("name", "m", "n", "k", "spk", "useBeta", "betaMode")


class ConfigIdentity(NamedTuple):
    bm: int
    tn: int
    lanes_k: int
    waves: int
    pf: int
    up_scales: int
    group: int
    xcds: int
    grid3d: bool
    blocks: int | None
    mfma: bool
    lds: bool


def mode_from_grid3d(grid3d: bool) -> ModeName:
    return "grid" if grid3d else "persistent"


def knobs_for_mode(mode: ModeName) -> tuple[str, ...]:
    return GRID_KNOBS if mode == "grid" else PERSIST_KNOBS


def identity_of(cfg: dict) -> ConfigIdentity:
    grid3d = bool(cfg["grid3d"])
    return ConfigIdentity(
        cfg["bm"],
        cfg["tn"],
        cfg["lanesK"],
        cfg["waves"],
        cfg["prefetch"],
        int(cfg.get("upIters", 0) > 0)
        if "upIters" in cfg
        else int(cfg.get("upScales", 0) > 0),
        cfg["group"],
        cfg["numXcds"],
        grid3d,
        None if grid3d else cfg["blocks"],
        bool(cfg.get("mfma", False)),
        bool(cfg.get("lds", False)),
    )


def case_key(problem: dict) -> tuple[Any, ...]:
    return tuple(problem[field] for field in CASE_KEY_FIELDS)


def proportion_z_greater(top_a: int, n_a: int, top_b: int, n_b: int) -> float:
    """One-sided p-value (normal approx): is top rate in A greater than in B?"""
    if n_a == 0 or n_b == 0:
        return 1.0
    p_a = top_a / n_a
    p_b = top_b / n_b
    if p_a <= p_b:
        return 1.0
    pooled = (top_a + top_b) / (n_a + n_b)
    if pooled <= 0.0 or pooled >= 1.0:
        return 0.0 if p_a > p_b else 1.0
    se = math.sqrt(pooled * (1.0 - pooled) * (1.0 / n_a + 1.0 / n_b))
    if se == 0.0:
        return 0.0 if p_a > p_b else 1.0
    z = (p_a - p_b) / se
    return 0.5 * math.erfc(z / math.sqrt(2.0))


@dataclass
class KnobValueStats:
    knob: str
    value: Any
    constraint_only: bool = False
    attempted: int = 0
    candidates: int = 0
    tops: int = 0
    phi: float = float("nan")
    p_value_vs_rest: float = float("nan")

    @property
    def top_rate(self) -> float:
        return self.tops / self.candidates if self.candidates else 0.0


@dataclass
class NumericKnobStats:
    knob: str
    n: int = 0
    point_biserial: float = float("nan")


@dataclass
class OnlinePearson:
    n: int = 0
    sum_x: float = 0.0
    sum_x2: float = 0.0
    sum_y: float = 0.0
    sum_xy: float = 0.0

    def add(self, x: float, y: int) -> None:
        self.n += 1
        self.sum_x += x
        self.sum_x2 += x * x
        self.sum_y += y
        self.sum_xy += x * y

    def pearson(self) -> float:
        if self.n < 2:
            return float("nan")
        n = float(self.n)
        num = n * self.sum_xy - self.sum_x * self.sum_y
        den_x = n * self.sum_x2 - self.sum_x * self.sum_x
        den_y = n * self.sum_y - self.sum_y * self.sum_y
        if den_x <= 0.0 or den_y <= 0.0:
            return 0.0
        return num / math.sqrt(den_x * den_y)


@dataclass
class ModeState:
    global_tops: int = 0
    global_candidates: int = 0
    value_stats: dict[tuple[str, Any], KnobValueStats] = field(default_factory=dict)
    numeric: dict[str, OnlinePearson] = field(default_factory=dict)
    seen_in_valid: dict[str, set[Any]] = field(default_factory=lambda: defaultdict(set))
    seen_in_fail: dict[str, set[Any]] = field(default_factory=lambda: defaultdict(set))

    def value_row(self, knob: str, value: Any) -> KnobValueStats:
        key = (knob, value)
        if key not in self.value_stats:
            self.value_stats[key] = KnobValueStats(knob, value)
        return self.value_stats[key]


@dataclass
class AnalysisState:
    valu_only: bool = True
    modes: dict[ModeName, ModeState] = field(
        default_factory=lambda: {m: ModeState() for m in MODES}
    )

    def mode(self, name: ModeName) -> ModeState:
        return self.modes[name]


def load_results(path: str) -> tuple[dict, list[dict]]:
    from sweep_skinny_gemm_results import read_results

    document = read_results(path)
    return document["meta"], document["results"]


def load_build_fail_identities(path: str) -> set[ConfigIdentity]:
    failed_path = os.path.splitext(path)[0] + ".failed.yaml"
    if not os.path.isfile(failed_path):
        return set()
    with open(failed_path) as stream:
        document = yaml.load(stream, Loader=YamlLoader)
    failed = set()
    for entry in document.get("failures", []):
        if entry.get("status") != "failed to build":
            continue
        cfg = entry.get("config")
        if not cfg:
            continue
        failed.add(identity_of(cfg))
    return failed


def top_identities_by_case(
    results: list[dict],
) -> dict[tuple[Any, ...], set[ConfigIdentity]]:
    out: dict[tuple[Any, ...], set[ConfigIdentity]] = {}
    for entry in results:
        problem = entry["problem"]
        key = case_key(problem)
        tops = set()
        for winner in entry.get("top") or []:
            tops.add(identity_of(winner["config"]))
        out[key] = tops
    return out


def results_include_mfma(results: list[dict]) -> bool:
    for entry in results:
        for winner in entry.get("top") or []:
            if winner["config"].get("mfma"):
                return True
    return False


def knob_values_from_identity(ident: ConfigIdentity) -> dict[str, Any]:
    return {
        "bm": ident.bm,
        "tn": ident.tn,
        "lanesK": ident.lanes_k,
        "waves": ident.waves,
        "prefetch": ident.pf,
        "upScales": ident.up_scales,
        "grid3d": ident.grid3d,
        "group": ident.group,
        "numXcds": ident.xcds,
        "blocks": ident.blocks,
        "mfma": ident.mfma,
        "lds": ident.lds,
    }


def knob_values_from_failure_cfg(cfg: dict) -> dict[str, Any]:
    return {
        "bm": cfg["bm"],
        "tn": cfg["tn"],
        "lanesK": cfg["lanesK"],
        "waves": cfg["waves"],
        "prefetch": cfg["prefetch"],
        "upScales": cfg["upScales"],
        "grid3d": cfg["grid3d"],
        "group": cfg["group"],
        "numXcds": cfg["numXcds"],
        "blocks": cfg.get("blocks"),
        "mfma": bool(cfg.get("mfma", False)),
        "lds": bool(cfg.get("lds", False)),
    }


def include_config(ident: ConfigIdentity, valu_only: bool) -> bool:
    if valu_only and (ident.mfma or ident.lds):
        return False
    return True


def record_candidate(
    state: AnalysisState,
    knobs: dict[str, Any],
    is_top: bool,
    *,
    candidates: bool,
) -> None:
    mode = mode_from_grid3d(bool(knobs["grid3d"]))
    ms = state.mode(mode)
    sweep_knobs = knobs_for_mode(mode)
    y = 1 if is_top else 0
    if candidates:
        ms.global_candidates += 1
        if is_top:
            ms.global_tops += 1
    for knob in sweep_knobs:
        value = knobs[knob]
        row = ms.value_row(knob, value)
        if candidates:
            row.candidates += 1
            if is_top:
                row.tops += 1
            if knob in NUMERIC_KNOBS and value is not None:
                if knob not in ms.numeric:
                    ms.numeric[knob] = OnlinePearson()
                ms.numeric[knob].add(float(value), y)
        else:
            row.attempted += 1


def finalize_mode_stats(ms: ModeState) -> None:
    for (knob, value), row in ms.value_stats.items():
        if row.candidates == 0:
            row.constraint_only = value in ms.seen_in_fail.get(knob, set()) and (
                value not in ms.seen_in_valid.get(knob, set())
            )
            continue
        a = row.tops
        not_v_tops = ms.global_tops - a
        not_v_cand = ms.global_candidates - row.candidates
        c = not_v_tops
        n = row.candidates + not_v_cand
        if (
            n
            and (a + c)
            and (row.candidates - a + not_v_cand - c)
            and row.candidates
            and not_v_cand
        ):
            b = row.candidates - a
            d = not_v_cand - c
            num = a * d - b * c
            den = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
            row.phi = num / den if den else 0.0
        else:
            row.phi = 0.0
        row.p_value_vs_rest = proportion_z_greater(a, row.candidates, c, not_v_cand)
        row.constraint_only = value in ms.seen_in_fail.get(knob, set()) and (
            value not in ms.seen_in_valid.get(knob, set())
        )


def analyze(
    results_path: str,
    input_path: str | None,
    *,
    only: str | None,
    limit_cases: int | None,
    progress: bool,
) -> tuple[dict, AnalysisState, sweep_config.SweepInput]:
    meta, results = load_results(results_path)
    if input_path is None:
        input_path = meta.get("inputFile")
    if not input_path:
        raise SystemExit(
            "No input YAML: pass --input or use results with meta.inputFile"
        )
    seed = meta.get("settings", {}).get("seed")
    sweep_in = load_input(input_path, seed=seed)
    tops_by_case = top_identities_by_case(results)
    build_failed = load_build_fail_identities(results_path)
    valu_only = not results_include_mfma(results)

    cases = sweep_in.cases
    if only:
        cases = [c for c in cases if only in c.name]
    if limit_cases is not None:
        cases = cases[:limit_cases]

    state = AnalysisState(valu_only=valu_only)
    t0 = time.monotonic()
    for index, case in enumerate(cases):
        key = case_key(case.record())
        top_set = tops_by_case.get(key, set())
        expanded = expand_case(case)
        for failure in expanded.failures:
            knobs = knob_values_from_failure_cfg(failure.config)
            if valu_only and (knobs["mfma"] or knobs["lds"]):
                continue
            mode = mode_from_grid3d(bool(knobs["grid3d"]))
            ms = state.mode(mode)
            for knob in knobs_for_mode(mode):
                ms.seen_in_fail[knob].add(knobs[knob])
            record_candidate(state, knobs, False, candidates=False)
        for problem in expanded.problems:
            ident = identity_of(config_record(problem))
            if not include_config(ident, valu_only):
                continue
            if ident in build_failed:
                continue
            knobs = knob_values_from_identity(ident)
            mode = mode_from_grid3d(ident.grid3d)
            ms = state.mode(mode)
            for knob in knobs_for_mode(mode):
                ms.seen_in_valid[knob].add(knobs[knob])
            is_top = ident in top_set
            record_candidate(state, knobs, is_top, candidates=True)
        if progress and (index + 1) % 25 == 0:
            elapsed = time.monotonic() - t0
            total = sum(ms.global_candidates for ms in state.modes.values())
            print(
                f"  {index + 1}/{len(cases)} cases, {total} candidates, {elapsed:.0f}s",
                flush=True,
            )

    for ms in state.modes.values():
        finalize_mode_stats(ms)

    mode_meta = {}
    for name in MODES:
        ms = state.mode(name)
        rate = ms.global_tops / ms.global_candidates if ms.global_candidates else 0.0
        mode_meta[name] = {
            "candidates": ms.global_candidates,
            "tops": ms.global_tops,
            "topRate": round(rate, 8),
        }

    meta_out = {
        "tool": TOOL,
        "resultsFile": os.path.abspath(results_path),
        "inputFile": os.path.abspath(input_path),
        "valuOnly": valu_only,
        "casesAnalyzed": len(cases),
        "modes": mode_meta,
        "buildFailuresExcluded": len(build_failed),
    }
    return meta_out, state, sweep_in


def numeric_summaries(ms: ModeState) -> list[NumericKnobStats]:
    out = []
    for knob in NUMERIC_KNOBS:
        online = ms.numeric.get(knob)
        stats = NumericKnobStats(knob=knob, n=online.n if online else 0)
        if online and online.n:
            stats.point_biserial = online.pearson()
        out.append(stats)
    return out


def valu_sweep_values(raw: dict) -> dict[ModeName, dict[str, set[Any]]]:
    """Literal knob lists from VALU sweep mappings in the input YAML."""
    out: dict[ModeName, dict[str, set[Any]]] = {
        "grid": defaultdict(set),
        "persistent": defaultdict(set),
    }
    blocks: list[dict] = []

    def ingest(mapping: dict) -> None:
        mode = mapping_mode(mapping)
        if mode is None:
            return
        for knob in knobs_for_mode(mode):
            if knob not in mapping:
                continue
            current = mapping[knob]
            if isinstance(current, list):
                out[mode][knob].update(current)
            else:
                out[mode][knob].add(current)

    defaults = raw.get("defaults") or {}
    sweep = defaults.get("sweep")
    if isinstance(sweep, list):
        blocks.extend(sweep)
    elif isinstance(sweep, dict):
        blocks.append(sweep)
    for mapping in blocks:
        ingest(mapping)
    for problem in raw.get("problems") or []:
        psweep = problem.get("sweep")
        if isinstance(psweep, list):
            for mapping in psweep:
                ingest(mapping)
        elif isinstance(psweep, dict):
            ingest(psweep)
    return {
        "grid": dict(out["grid"]),
        "persistent": dict(out["persistent"]),
    }


def _rows_for_swept_knob(
    ms: ModeState, mode: ModeName, knob: str, swept: set[Any]
) -> list[KnobValueStats]:
    rows: list[KnobValueStats] = []
    for value in swept:
        row = ms.value_stats.get((knob, value))
        if row is None:
            rows.append(KnobValueStats(knob, value))
        else:
            rows.append(row)
    return rows


def prune_mode(
    ms: ModeState,
    *,
    mode: ModeName,
    prune_mode: str,
    min_trials: int,
    max_phi: float,
    p_threshold: float,
    swept_values: dict[str, set[Any]],
    keep_at_least: int,
    min_top_slots: int,
    min_top_share: float,
    min_top_relative: float,
) -> dict[str, list[Any]]:
    drop: dict[str, set[Any]] = defaultdict(set)
    base_rate = ms.global_tops / ms.global_candidates if ms.global_candidates else 0.0
    sweep_knobs = knobs_for_mode(mode)
    total_top_slots = ms.global_tops

    for knob in sweep_knobs:
        swept = swept_values.get(knob)
        if not swept:
            continue
        rows = _rows_for_swept_knob(ms, mode, knob, swept)
        for row in rows:
            if row.constraint_only:
                drop[knob].add(row.value)
        ranked = sorted(
            [r for r in rows if not r.constraint_only and r.candidates],
            key=lambda r: r.tops,
            reverse=True,
        )
        if not ranked:
            continue
        max_tops = ranked[0].tops
        protected = {r.value for r in ranked[:keep_at_least]}

        for row in ranked:
            if row.value in protected:
                continue
            if row.candidates < min_trials:
                continue
            if prune_mode == "never-top":
                if row.tops == 0:
                    drop[knob].add(row.value)
                continue
            if prune_mode == "low-phi":
                if row.tops == 0:
                    drop[knob].add(row.value)
                elif row.phi <= max_phi and row.p_value_vs_rest >= p_threshold:
                    if row.top_rate <= base_rate:
                        drop[knob].add(row.value)
                continue
            # frequency (+ phi): rare in top lists and/or weak correlation with is_top.
            share = row.tops / total_top_slots if total_top_slots else 0.0
            relative = row.tops / max_tops if max_tops else 0.0
            weak_phi = (
                row.phi <= max_phi
                and row.p_value_vs_rest >= p_threshold
                and row.top_rate <= base_rate
            )
            if row.tops == 0:
                drop[knob].add(row.value)
            elif (
                row.tops < min_top_slots
                or share < min_top_share
                or relative < min_top_relative
                or (weak_phi and relative < min_top_relative)
            ):
                drop[knob].add(row.value)

    return {
        knob: sorted(values, key=lambda v: (isinstance(v, str), v))
        for knob, values in drop.items()
    }


def prune_all(
    state: AnalysisState,
    *,
    mode: str,
    min_trials: int,
    min_trials_grid: int,
    max_phi: float,
    p_threshold: float,
    swept_by_mode: dict[ModeName, dict[str, set[Any]]],
    keep_at_least: int,
    min_top_slots: int,
    min_top_slots_grid: int,
    min_top_share: float,
    min_top_relative: float,
) -> dict[ModeName, dict[str, list[Any]]]:
    return {
        "grid": prune_mode(
            state.mode("grid"),
            mode="grid",
            prune_mode=mode,
            min_trials=min_trials_grid,
            max_phi=max_phi,
            p_threshold=p_threshold,
            swept_values=swept_by_mode.get("grid") or {},
            keep_at_least=keep_at_least,
            min_top_slots=min_top_slots_grid,
            min_top_share=min_top_share,
            min_top_relative=min_top_relative,
        ),
        "persistent": prune_mode(
            state.mode("persistent"),
            mode="persistent",
            prune_mode=mode,
            min_trials=min_trials,
            max_phi=max_phi,
            p_threshold=p_threshold,
            swept_values=swept_by_mode.get("persistent") or {},
            keep_at_least=keep_at_least,
            min_top_slots=min_top_slots,
            min_top_share=min_top_share,
            min_top_relative=min_top_relative,
        ),
    }


def knob_report(ms: ModeState, mode: ModeName) -> dict[str, list[dict]]:
    sweep_knobs = set(knobs_for_mode(mode))
    by_knob: dict[str, list[dict]] = defaultdict(list)
    for row in sorted(
        ms.value_stats.values(),
        key=lambda r: (r.knob, str(r.value)),
    ):
        if row.knob not in sweep_knobs:
            continue
        if row.candidates == 0 and not row.constraint_only:
            continue
        by_knob[row.knob].append(
            {
                "value": row.value,
                "constraintOnly": row.constraint_only,
                "attempted": row.attempted,
                "candidates": row.candidates,
                "tops": row.tops,
                "topRate": round(row.top_rate, 8),
                "phi": round(row.phi, 6) if row.candidates else None,
                "pValueVsRest": round(row.p_value_vs_rest, 6)
                if row.candidates
                else None,
            }
        )
    return dict(by_knob)


def report_document(
    meta: dict,
    state: AnalysisState,
    prune_by_mode: dict[ModeName, dict[str, list[Any]]],
) -> dict:
    modes_out = {}
    for name in MODES:
        ms = state.mode(name)
        modes_out[name] = {
            "numericCorrelations": [
                {
                    "knob": s.knob,
                    "n": s.n,
                    "pointBiserial": round(s.point_biserial, 6)
                    if not math.isnan(s.point_biserial)
                    else None,
                }
                for s in numeric_summaries(ms)
                if s.n
            ],
            "knobs": knob_report(ms, name),
            "recommendedPrune": prune_by_mode.get(name) or {},
        }
    return {
        "meta": meta,
        "modes": modes_out,
    }


def mapping_mode(mapping: dict) -> ModeName | None:
    """Which VALU sweep block this YAML mapping is, or None if MFMA/skip."""
    if mapping.get("mfma"):
        return None
    if mapping.get("grid3d") is False:
        return "persistent"
    if mapping.get("grid3d") is True:
        return "grid"
    return None


def apply_prune_to_input(
    raw: dict, prune_by_mode: dict[ModeName, dict[str, list[Any]]]
) -> dict:
    """Remove prune values only from the sweep list entry that matches each mode."""
    out = copy.deepcopy(raw)

    def trim_mapping(mapping: dict) -> None:
        mode = mapping_mode(mapping)
        if mode is None:
            return
        prune = prune_by_mode.get(mode) or {}
        for knob, values in prune.items():
            if knob not in mapping:
                continue
            current = mapping[knob]
            drop = set(values)
            if isinstance(current, list):
                mapping[knob] = [v for v in current if v not in drop]
            elif current in drop:
                mapping.pop(knob, None)

    defaults = out.get("defaults") or {}
    sweep = defaults.get("sweep")
    if isinstance(sweep, list):
        for mapping in sweep:
            trim_mapping(mapping)
    elif isinstance(sweep, dict):
        trim_mapping(sweep)

    for problem in out.get("problems") or []:
        psweep = problem.get("sweep")
        if isinstance(psweep, list):
            for mapping in psweep:
                trim_mapping(mapping)
        elif isinstance(psweep, dict):
            trim_mapping(psweep)
    return out


def print_summary(report: dict) -> None:
    meta = report["meta"]
    print(
        f"Analyzed {meta['casesAnalyzed']} cases (VALU-only={meta['valuOnly']}).",
        flush=True,
    )
    if meta.get("buildFailuresExcluded"):
        print(
            f"Excluded {meta['buildFailuresExcluded']} build-failed identities.",
            flush=True,
        )
    for mode_name, mode_block in report["modes"].items():
        mmeta = meta["modes"][mode_name]
        print(
            f"\n=== {mode_name} (grid3d={'true' if mode_name == 'grid' else 'false'}) ===",
            flush=True,
        )
        print(
            f"  {mmeta['candidates']} candidates, {mmeta['tops']} top slots, "
            f"rate {mmeta['topRate']:.6f}",
            flush=True,
        )
        print("  Numeric vs is_top (point-biserial r):", flush=True)
        for entry in mode_block.get("numericCorrelations") or []:
            print(
                f"    {entry['knob']:10s} n={entry['n']:9d}  r={entry['pointBiserial']}",
                flush=True,
            )
        prune = mode_block.get("recommendedPrune") or {}
        if prune:
            print("  Recommended prune:", flush=True)
            for knob in sorted(prune):
                print(f"    {knob}: {prune[knob]}", flush=True)
        constraint = []
        for knob, entries in (mode_block.get("knobs") or {}).items():
            bad = [e["value"] for e in entries if e.get("constraintOnly")]
            if bad:
                constraint.append(f"{knob}: {bad}")
        if constraint:
            print("  Constraint-only:", flush=True)
            for line in constraint:
                print(f"    {line}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", help="Results YAML from sweep_skinny_gemm.py")
    parser.add_argument(
        "--input",
        help="Sweep input YAML (default: meta.inputFile from results)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Write full analysis report YAML here",
    )
    parser.add_argument(
        "--trim-yaml",
        help="Write input YAML with recommended prune applied (per sweep mapping)",
    )
    parser.add_argument(
        "--prune",
        choices=("frequency", "never-top", "low-phi", "none"),
        default="frequency",
        help="How to pick values for --trim-yaml (default: frequency)",
    )
    parser.add_argument(
        "--keep-at-least",
        type=int,
        default=2,
        help="Always keep this many top-frequency values per knob (frequency mode)",
    )
    parser.add_argument(
        "--min-top-slots",
        type=int,
        default=40,
        help="Min appearances in top lists, persistent mode (frequency mode)",
    )
    parser.add_argument(
        "--min-top-slots-grid",
        type=int,
        default=15,
        help="Min appearances in top lists, grid mode (frequency mode)",
    )
    parser.add_argument(
        "--min-top-share",
        type=float,
        default=0.008,
        help="Min fraction of mode top slots (frequency mode, default 0.008)",
    )
    parser.add_argument(
        "--min-top-relative",
        type=float,
        default=0.12,
        help="Min fraction of the best value's top count per knob (frequency mode)",
    )
    parser.add_argument(
        "--min-trials",
        type=int,
        default=1000,
        help="Min candidates in persistent mode before never-top prune (default: 1000)",
    )
    parser.add_argument(
        "--min-trials-grid",
        type=int,
        default=200,
        help="Min candidates in grid mode before never-top prune (default: 200)",
    )
    parser.add_argument(
        "--max-phi",
        type=float,
        default=0.01,
        help="For low-phi mode: drop values with phi <= this and weak p-value",
    )
    parser.add_argument(
        "--p-threshold",
        type=float,
        default=0.05,
        help="For low-phi mode: require p-value vs rest >= this to drop",
    )
    parser.add_argument("--only", help="Only analyze cases whose name contains this")
    parser.add_argument("--limit-cases", type=int, help="Cap cases (debug)")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Do not print progress while expanding cases",
    )
    args = parser.parse_args(argv)

    meta, state, _sweep_in = analyze(
        args.results,
        args.input,
        only=args.only,
        limit_cases=args.limit_cases,
        progress=not args.no_progress,
    )
    input_path = args.input or meta.get("inputFile")
    with open(input_path) as stream:
        raw_input = yaml.safe_load(stream)
    swept_by_mode = valu_sweep_values(raw_input)
    prune_by_mode = (
        {"grid": {}, "persistent": {}}
        if args.prune == "none"
        else prune_all(
            state,
            mode=args.prune,
            min_trials=args.min_trials,
            min_trials_grid=args.min_trials_grid,
            max_phi=args.max_phi,
            p_threshold=args.p_threshold,
            swept_by_mode=swept_by_mode,
            keep_at_least=args.keep_at_least,
            min_top_slots=args.min_top_slots,
            min_top_slots_grid=args.min_top_slots_grid,
            min_top_share=args.min_top_share,
            min_top_relative=args.min_top_relative,
        )
    )
    meta["prunePolicy"] = {
        "mode": args.prune,
        "keepAtLeast": args.keep_at_least,
        "minTopSlots": args.min_top_slots,
        "minTopSlotsGrid": args.min_top_slots_grid,
        "minTopShare": args.min_top_share,
        "minTopRelative": args.min_top_relative,
        "maxPhi": args.max_phi,
        "pThreshold": args.p_threshold,
        "frequencyUsesPhi": args.prune == "frequency",
    }
    report = report_document(meta, state, prune_by_mode)
    print_summary(report)

    if args.output:
        with open(args.output, "w") as stream:
            yaml.safe_dump(report, stream, sort_keys=False)
        print(f"\nWrote report to {args.output}", flush=True)

    if args.trim_yaml and any(prune_by_mode.values()):
        trimmed = apply_prune_to_input(raw_input, prune_by_mode)
        with open(args.trim_yaml, "w") as stream:
            yaml.safe_dump(trimmed, stream, sort_keys=False, default_flow_style=False)
        print(f"Wrote trimmed input to {args.trim_yaml}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
