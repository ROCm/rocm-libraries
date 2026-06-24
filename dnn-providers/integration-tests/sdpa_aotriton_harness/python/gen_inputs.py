"""Generate SDPA harness inputs once and write per-case manifests.

For each case in the selected tier this:
  1. Seeds torch deterministically (``case.seed``).
  2. Generates Q/K/V as float32 ``randn`` of the right shapes.
  3. Casts ONCE to the case dtype (bf16, fp16, or fp8). Both the on-disk ``.npy``
     and (later, in run_torch) the torch tensors derive from this same cast
     tensor, so the C++ driver and torch see bit-identical inputs.
  4. Saves Q/K/V per the integration contract (bf16 -> raw uint16 bits).
  5. For ``mode == "mask"`` generates a finite random additive bias mask and
     saves it as fp32.
  6. Writes the per-case ``manifest.json`` and a run-level ``index.json``.

torch is imported lazily inside ``main`` so this module passes ``py_compile``
and imports without a torch install.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np

import manifest as mf
import sdpa_cases


def _torch_dtype(dtype: str) -> Any:
    """Map a case dtype string to the corresponding torch dtype."""
    import torch  # local import; runtime only

    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp8_e4m3": torch.float8_e4m3fn,
        "fp8_e5m2": torch.float8_e5m2,
        "fp8_e4m3_fnuz": torch.float8_e4m3fnuz,
        "fp8_e5m2_fnuz": torch.float8_e5m2fnuz,
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {dtype!r}") from exc


def _save_qkv(path: str, tensor_native: Any, dtype: str) -> None:
    """Save a Q/K/V tensor (already cast to the case dtype) per the contract.

    ``tensor_native`` is a torch tensor in the target low precision dtype.
    """
    import torch  # local import; runtime only

    if dtype == "bf16":
        # Raw 16-bit bf16 bit patterns -> uint16 '<u2'.
        bits = tensor_native.bfloat16().view(torch.uint16).cpu().numpy()
        np.save(path, bits)
    elif dtype == "fp16":
        arr = tensor_native.half().cpu().numpy()  # float16 '<f2'
        np.save(path, arr)
    elif dtype in sdpa_cases.FP8_DTYPES:
        # Raw 8-bit fp8 bit patterns -> uint8 '|u1'. The tensor is already the
        # target fp8 dtype, so a uint8 view reinterprets the bytes losslessly.
        bits = tensor_native.view(torch.uint8).cpu().numpy()
        np.save(path, bits)
    else:
        raise ValueError(f"unsupported dtype for qkv save: {dtype!r}")


def _generate_case(case: sdpa_cases.Case, run_dir: str) -> dict:
    """Generate and persist all inputs for one case; return its manifest dict."""
    import torch  # local import; runtime only

    torch.manual_seed(case.seed)

    # float32 generation, then a single cast to the case dtype.
    q_f32 = torch.randn(case.B, case.Hq, case.Sq, case.D, dtype=torch.float32)
    k_f32 = torch.randn(case.B, case.Hkv, case.Skv, case.D, dtype=torch.float32)
    v_f32 = torch.randn(case.B, case.Hkv, case.Skv, case.D, dtype=torch.float32)

    target = _torch_dtype(case.dtype)
    q = q_f32.to(target)
    k = k_f32.to(target)
    v = v_f32.to(target)

    man = mf.manifest_from_case(case, run_dir)
    files = man["files"]

    # Ensure the per-case directory exists before writing input .npy files
    # (the manifest itself is written later by the caller).
    os.makedirs(mf.case_dir(run_dir, case.name), exist_ok=True)

    _save_qkv(files["q"], q, case.dtype)
    _save_qkv(files["k"], k, case.dtype)
    _save_qkv(files["v"], v, case.dtype)

    if case.mode == "mask":
        # Finite random additive bias [B, Hq, Sq, Skv], fp32. A small scale keeps
        # rows from being effectively fully masked (the matrix avoids -inf masks).
        torch.manual_seed(case.seed + 1)
        bias = (
            torch.randn(case.B, case.Hq, case.Sq, case.Skv, dtype=torch.float32) * 0.5
        )
        # gpu_ref reads fp32; AOTriton reads a dtype-cast of this when selected.
        # For bf16/fp16, rounding the bias through the case dtype here makes them
        # identical (the torch-side fp32->dtype cast is then lossless). fp8 cases
        # always fall back to the fp32-MATH reference, so the bias stays fp32.
        if case.dtype in ("bf16", "fp16"):
            target = torch.bfloat16 if case.dtype == "bf16" else torch.float16
            bias = bias.to(target).to(torch.float32)
        np.save(files["mask"], bias.cpu().numpy().astype("<f4", copy=False))

    return man


def generate(tier: str, out_dir: str, seed_base: int = 0) -> str:
    """Generate all inputs for ``tier`` into ``out_dir``; return ``out_dir``.

    Writes one ``manifest.json`` per case plus an ``index.json``.
    """
    cases = sdpa_cases.get_cases(tier, seed_base)
    os.makedirs(out_dir, exist_ok=True)

    names = []
    for case in cases:
        man = _generate_case(case, out_dir)
        mf.write_manifest(out_dir, man)
        names.append(case.name)

    mf.write_index(out_dir, names, tier)
    return out_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tier",
        choices=sdpa_cases.available_tiers(),
        default="quick",
        help="case tier to generate (default: quick)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="output run directory (created if missing)",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=0,
        help="base added to each per-case seed (default: 0)",
    )
    args = parser.parse_args()

    out = os.path.abspath(args.out)
    generate(args.tier, out, args.seed_base)
    print(f"Generated tier '{args.tier}' inputs into {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
