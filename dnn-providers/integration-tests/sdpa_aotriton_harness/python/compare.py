"""Compare the gpu_ref (HP oracle) output against AOTriton (LP) per case.

Methodology (mirrors AOTriton's own adaptive-tolerance approach):

  * The gpu_ref kernel is the high-precision oracle.
  * ``math_lp_o`` (torch MATH backend on native low-precision inputs) measures
    how much error low precision alone induces -> the *budget*.
  * AOTriton (flash / mem-efficient) must agree with the oracle to within a
    fudge factor of that budget, with an absolute floor per dtype.

For each non-skipped case, all in float32::

    err_aot    = max(abs(aotriton_o - gpuref_o))
    budget     = max(abs(gpuref_o   - math_lp_o))
    atol_floor = {"bf16": 1e-2, "fp16": 1e-3}[dtype]
    threshold  = max(atol_floor, fudge * budget)          # rtol = 0
    passed     = err_aot <= threshold
    selfcheck  = max(abs(gpuref_o - math_hp_o))           # report-only sanity

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
SELFCHECK_WARN = 1e-2


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

    err_aot = _max_abs_diff(aotriton, gpuref)
    budget = _max_abs_diff(gpuref, math_lp)
    selfcheck = _max_abs_diff(gpuref, math_hp)

    atol_floor = ATOL_FLOOR[dtype]
    threshold = max(atol_floor, fudge * budget)
    passed = err_aot <= threshold

    result["err_aot"] = err_aot
    result["budget"] = budget
    result["atol_floor"] = atol_floor
    result["threshold"] = threshold
    result["ratio"] = (err_aot / threshold) if threshold > 0 else float("inf")
    result["selfcheck"] = selfcheck
    result["selfcheck_warn"] = selfcheck > SELFCHECK_WARN
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
