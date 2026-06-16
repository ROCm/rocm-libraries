"""Per-row enrichment and run-level summary statistics for tuning results.
"""

from __future__ import annotations

import json
import re
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from geko.constants import GEMM_FIELDS
from geko.gemm_metrics import arithmetic_intensity, categorize, categorize_2


logger = logging.getLogger("GEKO")

__all__ = ["enrich", "summarize", "write_metrics_json"]


# Uplift buckets reported in the run-level summary.
_IMPROVEMENT_BUCKETS_PCT = (1, 3, 5, 10, 25, 50, 100)
_REGRESSION_BUCKETS_PCT = (1, 3, 5, 10)


_MT_RE = re.compile(r"MT(\d+x\d+x\d+)")
_MI_RE = re.compile(r"MI(\d+x\d+x\d+)")


def _extract_kernel_shape(series: pd.Series) -> pd.DataFrame:
    """Pull MT (macro-tile) and MI (matrix-instruction tile) out of a kernel-name column."""
    s = series.astype(str)
    mt = s.str.extract(_MT_RE, expand=False)
    mi = s.str.extract(_MI_RE, expand=False)
    return pd.DataFrame({"MT": mt, "MI": mi}, index=series.index)


def _load_and_normalize_weights(
    summary: str | Path,
    df_for_reference_us: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """Read summary.csv and return aggregated weights.

    Returns a DataFrame with columns GEMM_FIELDS + ["call_count", "pct_of_total"],
    or None if the path does not exist (treated as "user did not provide weights").
    Schema/format problems raise ValueError so they aren't silently dropped.

    Handles the same edge cases as the in-line weight loader that used to live
    in optim.analyze:
      - 'compute_type' tokens prefixed with ``c_`` are stripped.
      - '% of total' is recomputed from ``call_count * us`` if missing.
      - If 'us' is missing too, falls back to ``us_reference`` from ``df_for_reference_us``.
    """
    path = Path(summary)
    if not path.is_file():
        logger.warning(f"Summary file '{path}' not found; skipping weighted metrics")
        return None
    weights = pd.read_csv(path)

    if "compute_type" in weights.columns:
        weights["compute_type"] = (
            weights["compute_type"].astype(str).str.replace("c_", "", regex=False)
        )

    if "% of total" not in weights.columns:
        if "call_count" not in weights.columns:
            raise ValueError(
                f"Format error in '{path}': missing both '% of total' and 'call_count' columns"
            )
        if "us" in weights.columns:
            weights["total (us)"] = weights["call_count"] * weights["us"]
        elif df_for_reference_us is not None and "us_reference" in df_for_reference_us.columns:
            ref_us = (
                df_for_reference_us[list(GEMM_FIELDS) + ["us_reference"]]
                .drop_duplicates(subset=list(GEMM_FIELDS))
            )
            weights = weights.merge(ref_us, on=list(GEMM_FIELDS), how="left")
            
            if weights["us_reference"].isna().any():
                logger.warning(
                    f"{weights['us_reference'].isna().sum()} missing GEMMs from tuned results"
                )
                weights = weights.dropna(subset=['us_reference'])
                if len(weights) == 0:
                    raise ValueError(
                        f"Format error in '{path}': cannot infer reference time for any GEMM"
                    )
            weights["total (us)"] = weights["call_count"] * weights["us_reference"]
        else:
            raise ValueError(
                f"Format error in '{path}': missing both 'us' and a reference frame to infer it"
            )
        total_us = weights["total (us)"].sum()
        if total_us == 0:
            raise ValueError(
                f"Format error in '{path}': total reference time is zero, cannot compute '% of total'"
            )
        weights["% of total"] = 100.0 * weights["total (us)"] / total_us

    keys = list(GEMM_FIELDS)
    missing_keys = [k for k in keys if k not in weights.columns]
    if missing_keys:
        raise ValueError(f"Format error in '{path}': missing GEMM keys {missing_keys}")

    if "call_count" not in weights.columns:
        weights["call_count"] = 1

    agg = (
        weights.groupby(keys, as_index=False)[["call_count", "% of total"]]
        .sum()
        .rename(columns={"% of total": "pct_of_total"})
    )
    return agg


def enrich(df: pd.DataFrame, summary_csv: str | Path | None = None) -> pd.DataFrame:
    """Add derived per-row metrics. Returns a new DataFrame.

    When ``summary_csv`` is supplied, format errors propagate as ValueError;
    only a missing file is downgraded to a warning. The weighted contribution
    columns are NaN on rows where the kernel did not change or the row is
    flagged invalid, so summing those columns on raw_results.csv matches the
    headline E2E uplift.
    """
    out = df.copy()

    if "ratio" in out.columns:
        out["uplift_pct"] = (out["ratio"] - 1.0) * 100.0

    if "us_reference" in out.columns and "us_tuned" in out.columns:
        out["us_saved"] = out["us_reference"] - out["us_tuned"]

    if all(c in out.columns for c in ("m", "n", "k", "batch_count")):
        m, n, k, b = out["m"], out["n"], out["k"], out["batch_count"]

        # arith_intensity (FLOPs/Byte) is the user-facing roofline quantity.
        # Intermediate flop_count / bytes_moved aren't surfaced as columns
        # because they're counts and would read as rates next to
        # hipblaslt-Gflops_* / hipblaslt-GB/s_*; the helper computes them
        # internally.
        if all(c in out.columns for c in ("a_type", "b_type", "c_type")):
            with np.errstate(divide="ignore", invalid="ignore"):
                out["arith_intensity"] = arithmetic_intensity(
                    m, n, k, b, out["a_type"], out["b_type"], out["c_type"]
                )

        out["category"] = categorize(m, n, k, b)
        out["category_2"] = categorize_2(m, n, k, b)

    if "kernel_reference" in out.columns and "kernel_tuned" in out.columns:
        out["kernel_changed"] = (out["kernel_reference"] != out["kernel_tuned"]).astype(bool)
    if "solutionIdx_reference" in out.columns and "solutionIdx_tuned" in out.columns:
        out["solution_changed"] = (
            out["solutionIdx_reference"].astype(str) != out["solutionIdx_tuned"].astype(str)
        )

    if "kernel_tuned" in out.columns:
        shape = _extract_kernel_shape(out["kernel_tuned"])
        out["MT_tuned"] = shape["MT"]
        out["MI_tuned"] = shape["MI"]
    if "kernel_reference" in out.columns:
        shape_ref = _extract_kernel_shape(out["kernel_reference"])
        out["MT_reference"] = shape_ref["MT"]
        out["MI_reference"] = shape_ref["MI"]

    if summary_csv is not None:
        agg = _load_and_normalize_weights(summary_csv, df_for_reference_us=out)
        if agg is not None:
            keys = list(GEMM_FIELDS)
            merged = out.merge(agg, on=keys, how="left")
            out["call_count"] = merged["call_count"]
            out["pct_of_total"] = merged["pct_of_total"]

            if "us_saved" in out.columns:
                out["weighted_us_saved"] = out["call_count"] * out["us_saved"]
            if "ratio" in out.columns:
                out["weighted_uplift_contribution_pct"] = (
                    out["pct_of_total"] * (1.0 - 1.0 / out["ratio"])
                )

    return out


def _percentiles(arr: np.ndarray, pcts: Iterable[float]) -> dict:
    if arr.size == 0:
        return {f"p{int(p)}": None for p in pcts}
    qs = np.percentile(arr, list(pcts))
    return {f"p{int(p)}": float(q) for p, q in zip(pcts, qs)}


def _gmean_uplift_pct(ratios: np.ndarray) -> float | None:
    """Geometric mean of ratios, expressed as a percentage uplift."""
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    if ratios.size == 0:
        return None
    return float(100.0 * np.expm1(np.mean(np.log(ratios))))


def _range_buckets_improvement(uplifts_pct: np.ndarray, thresholds: tuple) -> dict:
    """Disjoint histogram of improvement magnitudes — every uplift > 0 lands in exactly one bucket.

    Buckets partition (0, +inf) using `thresholds` (sorted ascending) as breakpoints:
      - "lt_{T0}_pct"         : uplift in (0, T0)
      - "{T0}_to_{T1}_pct"    : uplift in [T0, T1)
      - ...
      - "ge_{TN}_pct"         : uplift in [TN, +inf)
    """
    out = {}
    t0 = thresholds[0]
    out[f"lt_{t0}_pct"] = int(((uplifts_pct > 0) & (uplifts_pct < t0)).sum())
    for lo, hi in zip(thresholds[:-1], thresholds[1:]):
        out[f"{lo}_to_{hi}_pct"] = int(((uplifts_pct >= lo) & (uplifts_pct < hi)).sum())
    tN = thresholds[-1]
    out[f"ge_{tN}_pct"] = int((uplifts_pct >= tN).sum())
    return out


def _range_buckets_regression(uplifts_pct: np.ndarray, thresholds: tuple) -> dict:
    """Disjoint histogram of regression magnitudes — every uplift < 0 lands in exactly one bucket.

    Buckets express regression severity (positive %-slower values):
      - "lt_{T0}_pct"         : uplift in (-T0, 0)            -> < T0% slower
      - "{T0}_to_{T1}_pct"    : uplift in (-T1, -T0]          -> [T0, T1)% slower
      - "ge_{TN}_pct"         : uplift in (-inf, -TN]         -> >= TN% slower
    """
    out = {}
    t0 = thresholds[0]
    out[f"lt_{t0}_pct"] = int(((uplifts_pct < 0) & (uplifts_pct > -t0)).sum())
    for lo, hi in zip(thresholds[:-1], thresholds[1:]):
        out[f"{lo}_to_{hi}_pct"] = int(((uplifts_pct <= -lo) & (uplifts_pct > -hi)).sum())
    tN = thresholds[-1]
    out[f"ge_{tN}_pct"] = int((uplifts_pct <= -tN).sum())
    return out


def _per_dtype_slice(df: pd.DataFrame) -> dict:
    """Per-(a_type, transA, transB) uplift summary, on the kept set."""
    if df.empty or not all(c in df.columns for c in ("a_type", "transA", "transB", "ratio")):
        return {}
    out = {}
    for (a, ta, tb), grp in df.groupby(["a_type", "transA", "transB"]):
        ratios = grp["ratio"].to_numpy(dtype=float)
        uplifts = (ratios - 1.0) * 100.0
        out[f"{a}_{ta}{tb}"] = {
            "n": int(len(grp)),
            "geomean_uplift_pct": _gmean_uplift_pct(ratios),
            "median_uplift_pct": float(np.median(uplifts)) if uplifts.size else None,
            "max_uplift_pct": float(uplifts.max()) if uplifts.size else None,
        }
    return out


def summarize(
    df_raw: pd.DataFrame,
    df_final: pd.DataFrame,
    summary_csv: str | Path | None = None,
    up_thr: float = 1.03,
    error_thr: float = 0.03,
    verify: bool = True,
) -> dict:
    """Compute run-level summary statistics.

    Args:
        df_raw: All attempted GEMMs (before up_thr filtering), ideally enriched.
        df_final: GEMMs kept after applying error/up thresholds.
        summary_csv: summary.csv path for workload-weighted E2E stats.
        up_thr: Speedup threshold used for filtering (ratio >= up_thr -> kept).
        error_thr: Verification error threshold.
        verify: Whether verification actually ran (error_tuned trustworthy).

    Returns:
        JSON-serializable dict with keys: config, counts, uplift_all_attempted,
        uplift_kept, diversity, e2e, per_dtype_layout.
    """
    raw = df_raw.copy()
    final = df_final.copy() if df_final is not None else df_raw.iloc[0:0].copy()

    if "uplift_pct" not in raw.columns and "ratio" in raw.columns:
        raw["uplift_pct"] = (raw["ratio"] - 1.0) * 100.0
    if "us_saved" not in raw.columns and {"us_reference", "us_tuned"}.issubset(raw.columns):
        raw["us_saved"] = raw["us_reference"] - raw["us_tuned"]

    n_attempted = int(len(raw))

    # When verification didn't run, error_tuned is silently 0 for every row,
    # so the validity counts would be meaningless — omit the keys entirely
    if not verify:
        valid_mask = pd.Series(True, index=raw.index)
        n_valid = None
        n_invalid = None
    elif "error_tuned" in raw.columns:
        valid_mask = raw["error_tuned"] < error_thr
        n_valid = int(valid_mask.sum())
        n_invalid = int((~valid_mask).sum())
    else:
        valid_mask = pd.Series(True, index=raw.index)
        n_valid = n_attempted
        n_invalid = 0

    # Restrict "all attempted" uplift stats to rows whose tuned kernel was
    # numerically trustworthy. A buggy tuned kernel can post any ratio and
    # would otherwise pollute the geomean.
    ratios_series = raw["ratio"] if "ratio" in raw.columns else pd.Series(dtype=float)
    ratios_all = ratios_series[valid_mask].to_numpy(dtype=float) if len(ratios_series) else np.array([])
    finite_all = ratios_all[np.isfinite(ratios_all)]

    n_improved_any = int((finite_all > 1.0).sum())
    n_regressed_any = int((finite_all < 1.0).sum())
    n_neutral = int((finite_all == 1.0).sum())

    uplifts_all_pct = (finite_all - 1.0) * 100.0

    # Cumulative threshold counts: each entry = "uplift magnitude at least X%".
    # Nested keys for symmetry with the disjoint sub-dicts below.
    improved_cumulative = {
        f"ge_{p}_pct": int((finite_all >= 1.0 + p / 100.0).sum())
        for p in _IMPROVEMENT_BUCKETS_PCT
    }
    regressed_cumulative = {
        f"ge_{p}_pct": int((finite_all <= 1.0 - p / 100.0).sum())
        for p in _REGRESSION_BUCKETS_PCT
    }

    # Disjoint histogram counts: each row lands in exactly one bucket.
    # Sums: sum(improved_in_ranges.*) == n_improved, sum(regressed_in_ranges.*) == n_regressed.
    improved_in_ranges = _range_buckets_improvement(uplifts_all_pct, _IMPROVEMENT_BUCKETS_PCT)
    regressed_in_ranges = _range_buckets_regression(uplifts_all_pct, _REGRESSION_BUCKETS_PCT)
    stats_all = {
        "mean_uplift_pct": float(uplifts_all_pct.mean()) if uplifts_all_pct.size else None,
        "geomean_uplift_pct": _gmean_uplift_pct(finite_all),
        "median_uplift_pct": float(np.median(uplifts_all_pct)) if uplifts_all_pct.size else None,
        "max_uplift_pct": float(uplifts_all_pct.max()) if uplifts_all_pct.size else None,
        "min_uplift_pct": float(uplifts_all_pct.min()) if uplifts_all_pct.size else None,
        "std_uplift_pct": float(uplifts_all_pct.std(ddof=0)) if uplifts_all_pct.size else None,
        **_percentiles(uplifts_all_pct, (25, 50, 75, 90, 99)),
    }

    n_final = int(len(final))
    if n_final > 0 and "ratio" in final.columns:
        ratios_final = final["ratio"].to_numpy(dtype=float)
        ratios_final = ratios_final[np.isfinite(ratios_final)]
        uplifts_final_pct = (ratios_final - 1.0) * 100.0
        stats_final = {
            "mean_uplift_pct": float(uplifts_final_pct.mean()) if uplifts_final_pct.size else None,
            "geomean_uplift_pct": _gmean_uplift_pct(ratios_final),
            "median_uplift_pct": float(np.median(uplifts_final_pct)) if uplifts_final_pct.size else None,
            "max_uplift_pct": float(uplifts_final_pct.max()) if uplifts_final_pct.size else None,
            **_percentiles(uplifts_final_pct, (25, 50, 75, 90, 99)),
        }
    else:
        stats_final = {}

    diversity = {}
    if "kernel_tuned" in final.columns and n_final > 0:
        diversity["n_unique_winning_kernels"] = int(final["kernel_tuned"].nunique())
        diversity["unique_winners_per_gemm"] = float(diversity["n_unique_winning_kernels"] / n_final)
    if "kernel_changed" in raw.columns:
        diversity["pct_kernel_changed_in_attempted"] = float(100.0 * raw["kernel_changed"].mean())

    e2e = {}
    if summary_csv is not None and n_final > 0 and "ratio" in final.columns:
        # If enrich() already merged weights into `final`, use those directly;
        # otherwise re-derive them from summary_csv.
        if {"pct_of_total", "call_count"}.issubset(final.columns):
            wfinal = final
        else:
            agg = _load_and_normalize_weights(summary_csv, df_for_reference_us=raw)
            wfinal = (
                final.merge(agg, on=list(GEMM_FIELDS), how="left")
                if agg is not None else final.iloc[0:0]
            )

        if not wfinal.empty and "pct_of_total" in wfinal.columns:
            n_unmatched = int(wfinal["pct_of_total"].isna().sum())
            if n_unmatched:
                # Surface mismatched rows (was a silent ValueError pre-refactor).
                logger.warning(
                    f"{n_unmatched}/{len(wfinal)} kept GEMMs have no entry in summary.csv; "
                    "excluded from E2E uplift"
                )
                wfinal = wfinal.dropna(subset=["pct_of_total"])

        if not wfinal.empty:
            contribution = (
                wfinal["pct_of_total"] / 100.0 * (1.0 - 1.0 / wfinal["ratio"])
            )
            e2e["e2e_uplift_pct"] = float(100.0 * contribution.sum())
            e2e["coverage_pct"] = float(wfinal["pct_of_total"].sum())
            e2e["e2e_us_saved_per_iter"] = float(
                (wfinal["call_count"] * (wfinal["us_reference"] - wfinal["us_tuned"])).sum()
            )
            contribs = contribution.sort_values(ascending=False).to_numpy()
            if contribs.size > 0 and contribs.sum() > 0:
                top_n = max(1, int(np.ceil(0.10 * contribs.size)))
                e2e["top10pct_contribution_share"] = float(
                    contribs[:top_n].sum() / contribs.sum()
                )

    per_dtype = _per_dtype_slice(final) if n_final > 0 else {}

    return {
        "config": {
            "up_thr": float(up_thr),
            "error_thr": float(error_thr),
            "verify": bool(verify),
        },
        "counts": {
            "n_attempted": n_attempted,
            **({"n_valid": n_valid, "n_invalid": n_invalid} if verify else {}),
            "n_improved": n_improved_any,
            "n_regressed": n_regressed_any,
            "n_neutral": n_neutral,
            "n_kept": n_final,
            "improved_cumulative": improved_cumulative,
            "regressed_cumulative": regressed_cumulative,
            "improved_in_ranges": improved_in_ranges,
            "regressed_in_ranges": regressed_in_ranges,
        },
        "uplift_all_attempted": stats_all,
        "uplift_kept": stats_final,
        "diversity": diversity,
        "e2e": e2e,
        "per_dtype_layout": per_dtype,
    }


def _to_jsonable(obj):
    """Recursively coerce numpy / pandas scalars to JSON-native types."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def write_metrics_json(metrics: dict, path: str | Path) -> Path:
    """Persist a summarize() dict to disk as JSON."""
    path = Path(path)
    with open(path, "w") as f:
        json.dump(_to_jsonable(metrics), f, indent=2, sort_keys=False)
    return path
