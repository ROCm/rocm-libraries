"""Compare the gpu_ref CANDIDATE against the selected reference per case.

Framing: the **gpu_ref** kernel is the candidate under test. ``run_torch`` writes
``reference_o`` from either PyTorch MATH or AOTriton, plus MATH HP/LP diagnostic
outputs for the adaptive tolerance budget.

For **fp8** dtypes AOTriton is unavailable (torch SDPA rejects fp8 on every
backend), so ``run_torch`` always writes torch's fp32-MATH output into
``reference_o`` and ``math_hp_o``; the low-precision budget leg (``math_lp_o``)
uses fp16. The methodology below is otherwise unchanged; for fp8 the candidate
and reference share the identical fp8 input bits, so ``err`` reflects fp32
compute agreement and several diagnostic columns coincide.

Methodology:

  * ``pytorch-math`` compares the fp32 gpu_ref candidate directly against
    ``math_hp_o`` / ``reference_o``. That is a fp32-vs-fp32 comparison, so it uses
    a tight absolute fp32 threshold.
  * ``aotriton`` compares the fp32 gpu_ref candidate against a low-precision
    backend. That path uses an independent low-precision budget from
    ``math_hp_o`` vs ``math_lp_o``.

For each non-skipped case, all comparison math is in float32::

    err             = max(abs(gpuref_o - reference_o))  # candidate vs reference
    budget          = max(abs(math_hp_o - math_lp_o))   # diagnostic fp32-vs-LP gap
    if reference == "aotriton":
        threshold   = max(lp_atol_floor[dtype], fudge * budget)
    else:
        threshold   = fp32_atol_floor[dtype]
    passed          = err <= threshold
    # diagnostics (report-only):
    gpuref_vs_fp32  = max(abs(gpuref_o    - math_hp_o))  # candidate a sound fp32 impl?
    reference_vs_lp = max(abs(reference_o - math_lp_o))  # only meaningful for LP refs

The budget previously used the candidate (``|gpuref - math_lp|``), which is
circular now that gpu_ref is the thing under test; ``|math_hp - math_lp|`` has
the same magnitude (bf16/fp16 attention error) but is independent of both the
candidate and the selected reference.

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

# AOTriton is a low-precision backend reference, so its pass/fail threshold uses
# a floor plus an adaptive fp32-vs-LP MATH budget. PyTorch MATH is an fp32 oracle,
# so it gets a separate tight fp32 absolute threshold.
LP_ATOL_FLOOR = {
    "bf16": 1e-2,
    "fp16": 1e-3,
    "fp8_e4m3": 1e-3,
    "fp8_e5m2": 1e-3,
    "fp8_e4m3_fnuz": 1e-3,
    "fp8_e5m2_fnuz": 1e-3,
}
FP32_ATOL_FLOOR = {
    "bf16": 1e-4,
    "fp16": 1e-4,
    "fp8_e4m3": 1e-4,
    "fp8_e5m2": 1e-4,
    "fp8_e4m3_fnuz": 1e-4,
    "fp8_e5m2_fnuz": 1e-4,
}
FP32_DIAG_WARN = 1e-4
LP_DIAG_WARN = 1e-2


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
        "reference": status.get("reference"),
        "reference_backend": status.get("reference_backend"),
    }
    if "requested_reference" in status:
        result["requested_reference"] = status["requested_reference"]

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
        reference = _load_f32(files["reference_o"])
        math_hp = _load_f32(files["math_hp_o"])
        math_lp = _load_f32(files["math_lp_o"])
    except FileNotFoundError as exc:
        result["result"] = "ERROR"
        result["reason"] = f"missing output: {exc}"
        return result

    if (
        gpuref.shape != reference.shape
        or gpuref.shape != math_hp.shape
        or gpuref.shape != math_lp.shape
    ):
        result["result"] = "ERROR"
        result["reason"] = (
            f"shape mismatch: gpuref {gpuref.shape}, reference {reference.shape}, "
            f"math_hp {math_hp.shape}, math_lp {math_lp.shape}"
        )
        return result

    if _has_nonfinite(gpuref, reference, math_hp, math_lp):
        result["result"] = "FAIL"
        result["reason"] = "non-finite (NaN/Inf) value in an output"
        return result

    # Candidate (gpu_ref) vs selected reference.
    err = _max_abs_diff(gpuref, reference)
    # Budget = independent fp32-vs-low-precision gap (torch math). Depends on
    # neither candidate nor reference, so it is not circular. The old budget used
    # |gpuref - math_lp|, which is circular now that gpu_ref is under test;
    # |math_hp - math_lp| has the same magnitude (bf16/fp16 attention error).
    budget = _max_abs_diff(math_hp, math_lp)
    gpuref_vs_fp32 = _max_abs_diff(gpuref, math_hp)
    effective_reference = status.get("reference")
    if effective_reference == "aotriton":
        reference_vs_lp = _max_abs_diff(reference, math_lp)
        atol_floor = LP_ATOL_FLOOR[dtype]
        threshold = max(atol_floor, fudge * budget)
    else:
        reference_vs_lp = None
        atol_floor = FP32_ATOL_FLOOR[dtype]
        threshold = atol_floor

    passed = err <= threshold

    result["err"] = err
    result["budget"] = budget
    result["atol_floor"] = atol_floor
    result["threshold"] = threshold
    result["ratio"] = (err / threshold) if threshold > 0 else float("inf")
    result["gpuref_vs_fp32"] = gpuref_vs_fp32
    result["gpuref_vs_fp32_warn"] = gpuref_vs_fp32 > FP32_DIAG_WARN
    result["reference_vs_lp"] = reference_vs_lp
    result["reference_vs_lp_warn"] = (
        reference_vs_lp is not None and reference_vs_lp > LP_DIAG_WARN
    )
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
