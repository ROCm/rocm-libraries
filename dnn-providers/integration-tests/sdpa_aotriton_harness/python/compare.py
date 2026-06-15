"""Compare the gpu_ref CANDIDATE against the AOTriton ORACLE per case.

Framing: **AOTriton** (flash / mem-efficient via PyTorch SDPA) is the oracle /
reference of record; the **gpu_ref** kernel is the candidate under test.

Methodology (adaptive tolerance with an independent precision-gap budget):

  * ``math_hp_o`` (torch MATH backend, inputs upcast to fp32) and ``math_lp_o``
    (torch MATH backend, native low-precision inputs) bracket the inherent
    bf16/fp16 attention error. Their gap is the *budget* -- it depends on
    neither the candidate nor the oracle.
  * The candidate must agree with the oracle to within a fudge factor of that
    budget, with an absolute floor per dtype.

For each non-skipped case, all in float32::

    err            = max(abs(gpuref_o - aotriton_o))   # candidate vs oracle
    budget         = max(abs(math_hp_o - math_lp_o))    # fp32-vs-LP gap (torch math)
    atol_floor     = {"bf16": 1e-2, "fp16": 1e-3}[dtype]
    threshold      = max(atol_floor, fudge * budget)    # rtol = 0
    passed         = err <= threshold
    # diagnostics (report-only):
    gpuref_vs_fp32 = max(abs(gpuref_o  - math_hp_o))    # candidate a sound fp32 impl?
    aotriton_vs_lp = max(abs(aotriton_o - math_lp_o))   # oracle a standard LP impl?

The budget previously used the candidate (``|gpuref - math_lp|``), which is
circular now that gpu_ref is the thing under test; ``|math_hp - math_lp|`` has
the same magnitude (bf16/fp16 attention error) but is independent of both the
candidate and the oracle.

NaN / Inf in any output is treated as a failure (the case matrix avoids
fully-masked rows, so finite outputs are expected everywhere).

Only numpy is required.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import numpy as np

import manifest as mf

ATOL_FLOOR = {"bf16": 1e-2, "fp16": 1e-3}
DIAG_WARN = 1e-2  # warn threshold for the report-only diagnostics


def _load_f32(path: str) -> np.ndarray:
    """Load an fp32 '<f4' array."""
    return np.load(path).astype(np.float32, copy=False)


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Max absolute element-wise difference as a Python float."""
    return float(np.max(np.abs(a - b)))


def _has_nonfinite(*arrays: np.ndarray) -> bool:
    """True if any array contains a NaN or Inf."""
    return any(not np.all(np.isfinite(a)) for a in arrays)


def compare_case(man: Dict[str, Any], fudge: float) -> Dict[str, Any]:
    """Compare a single case; return a result dict (never raises on data issues)."""
    name = man["name"]
    dtype = man["dtype"]
    status = man.get("status", {})
    state = status.get("state", "pending")

    result: Dict[str, Any] = {
        "name": name,
        "dtype": dtype,
        "B": man["B"],
        "Hq": man["Hq"],
        "Hkv": man["Hkv"],
        "Sq": man["Sq"],
        "Skv": man["Skv"],
        "D": man["D"],
        "mode": man["mode"],
        "backend": status.get("backend_used"),
    }

    if state == "skipped":
        result["result"] = "SKIP"
        result["reason"] = status.get("reason", "skipped")
        return result
    if state == "error":
        result["result"] = "ERROR"
        result["reason"] = status.get("reason", "error")
        return result
    if state != "ok":
        result["result"] = "ERROR"
        result["reason"] = f"unexpected status state {state!r}"
        return result

    files = man["files"]
    try:
        gpuref = _load_f32(files["gpuref_o"])
        aotriton = _load_f32(files["aotriton_o"])
        math_hp = _load_f32(files["math_hp_o"])
        math_lp = _load_f32(files["math_lp_o"])
    except FileNotFoundError as exc:
        result["result"] = "ERROR"
        result["reason"] = f"missing output: {exc}"
        return result

    if gpuref.shape != aotriton.shape or gpuref.shape != math_lp.shape:
        result["result"] = "ERROR"
        result["reason"] = (
            f"shape mismatch: gpuref {gpuref.shape}, aotriton {aotriton.shape}, "
            f"math_lp {math_lp.shape}"
        )
        return result

    if _has_nonfinite(gpuref, aotriton, math_hp, math_lp):
        result["result"] = "FAIL"
        result["reason"] = "non-finite (NaN/Inf) value in an output"
        return result

    # Candidate (gpu_ref) vs oracle (AOTriton).
    err = _max_abs_diff(gpuref, aotriton)
    # Budget = independent fp32-vs-low-precision gap (torch math). Depends on
    # neither candidate nor oracle, so it is not circular. The old budget used
    # |gpuref - math_lp|, which is circular now that gpu_ref is under test;
    # |math_hp - math_lp| has the same magnitude (bf16/fp16 attention error).
    budget = _max_abs_diff(math_hp, math_lp)
    # Diagnostics (report-only).
    gpuref_vs_fp32 = _max_abs_diff(gpuref, math_hp)
    aotriton_vs_lp = _max_abs_diff(aotriton, math_lp)

    atol_floor = ATOL_FLOOR[dtype]
    threshold = max(atol_floor, fudge * budget)
    passed = err <= threshold

    result["err"] = err
    result["budget"] = budget
    result["atol_floor"] = atol_floor
    result["threshold"] = threshold
    result["ratio"] = (err / threshold) if threshold > 0 else float("inf")
    result["gpuref_vs_fp32"] = gpuref_vs_fp32
    result["gpuref_vs_fp32_warn"] = gpuref_vs_fp32 > DIAG_WARN
    result["aotriton_vs_lp"] = aotriton_vs_lp
    result["aotriton_vs_lp_warn"] = aotriton_vs_lp > DIAG_WARN
    result["result"] = "PASS" if passed else "FAIL"
    return result


def compare(run_dir: str, fudge: float = 4.0) -> List[Dict[str, Any]]:
    """Compare every case in ``run_dir``; write ``results.json``; return results."""
    index = mf.read_index(run_dir)
    results: List[Dict[str, Any]] = []
    for name in index["cases"]:
        man = mf.read_manifest(run_dir, name)
        results.append(compare_case(man, fudge))

    out = os.path.join(run_dir, "results.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"fudge": fudge, "results": results}, fh, indent=2)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="run directory")
    parser.add_argument(
        "--fudge",
        type=float,
        default=4.0,
        help="multiplier on the low-precision budget (default: 4.0)",
    )
    args = parser.parse_args()
    run_dir = os.path.abspath(args.run_dir)
    results = compare(run_dir, args.fudge)

    n_fail = sum(1 for r in results if r["result"] == "FAIL")
    n_err = sum(1 for r in results if r["result"] == "ERROR")
    print(
        f"compare complete: {len(results)} cases, "
        f"{n_fail} fail, {n_err} error. results.json written."
    )
    return 1 if (n_fail or n_err) else 0


if __name__ == "__main__":
    raise SystemExit(main())
