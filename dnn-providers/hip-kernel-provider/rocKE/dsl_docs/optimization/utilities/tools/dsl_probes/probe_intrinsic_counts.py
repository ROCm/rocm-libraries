# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Count AMDGCN LLVM intrinsics in the lowered IR.

Lowers a KernelDef to AMDGPU LLVM IR via
``rocke.core.lower_llvm.lower_kernel_to_llvm`` and reports
per-intrinsic counts plus a small set of structural metrics (load /
store / fmul / fadd / fmuladd / br / phi).

The output answers questions like:

- "Is the kernel actually emitting the 32x32x16 MFMA atom I asked for?"
- "Is async DRAM→LDS active (``raw.ptr.buffer.load.lds``)?"
- "Is the softmax using ``ds_swizzle`` or ``ds_bpermute`` for the
  cross-lane reduction?"
- "How many ``fmul``/``fadd`` survived after lowering and folding?"

The script is intentionally lower-cost than ``probe_isa_inspect``: it
needs no HSACO build, only the LLVM lowering pass, which runs in tens of
milliseconds per spec.

CLI demo:
    python probe_intrinsic_counts.py --demo attention_tiled_2d

Programmatic use:
    from probe_intrinsic_counts import count_intrinsics
    text = lower_kernel_to_llvm(kdef)
    counts = count_intrinsics(text)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable


def _bootstrap_rocke() -> None:
    try:
        import rocke  # noqa: F401

        return
    except ImportError:
        pass
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "Python"
        if (candidate / "rocke" / "__init__.py").exists():
            sys.path.insert(0, str(candidate))
            return
        candidate = parent / "rocke" / "__init__.py"
        if candidate.exists():
            sys.path.insert(0, str(parent))
            return


_bootstrap_rocke()

from rocke.core.lower_llvm import lower_kernel_to_llvm  # noqa: E402


# Patterns most often surface the difference between MFMA-tiled
# and warp-scalar variants. Add to this list freely; counts are cheap.
DEFAULT_INTRINSIC_PATTERNS = {
    "mfma.f32.16x16x16.f16": r"@llvm\.amdgcn\.mfma\.f32\.16x16x16\.f16",
    "mfma.f32.16x16x32.f16": r"@llvm\.amdgcn\.mfma\.f32\.16x16x32\.f16",
    "mfma.f32.16x16x16.bf16": r"@llvm\.amdgcn\.mfma\.f32\.16x16x16\.bf16",
    "mfma.f32.16x16x32.bf16": r"@llvm\.amdgcn\.mfma\.f32\.16x16x32\.bf16",
    "mfma.f32.32x32x8.f16": r"@llvm\.amdgcn\.mfma\.f32\.32x32x8\.f16",
    "mfma.f32.32x32x16.f16": r"@llvm\.amdgcn\.mfma\.f32\.32x32x16\.f16",
    "mfma.f32.32x32x16.bf16": r"@llvm\.amdgcn\.mfma\.f32\.32x32x16\.bf16",
    "mfma.f32.4x4x4.f16": r"@llvm\.amdgcn\.mfma\.f32\.4x4x4\.f16",
    "raw.ptr.buffer.load.lds (async DMA)": r"@llvm\.amdgcn\.raw\.ptr\.buffer\.load\.lds",
    "raw.ptr.buffer.load (sync VMEM)": r"@llvm\.amdgcn\.raw\.ptr\.buffer\.load[^.l]",
    "raw.ptr.buffer.store": r"@llvm\.amdgcn\.raw\.ptr\.buffer\.store",
    "global.load": r"@llvm\.amdgcn\.global\.load",
    "global.store": r"@llvm\.amdgcn\.global\.store",
    "ds.read.tr16.b64": r"@llvm\.amdgcn\.ds\.read\.tr16\.b64",
    "ds.read.tr16.b128": r"@llvm\.amdgcn\.ds\.read\.tr16\.b128",
    "ds.bpermute": r"@llvm\.amdgcn\.ds\.bpermute",
    "ds.swizzle": r"@llvm\.amdgcn\.ds\.swizzle",
    "permlane32.swap": r"@llvm\.amdgcn\.permlane32\.swap",
    "cvt.pk.f32.fp8": r"@llvm\.amdgcn\.cvt\.pk\.f32\.fp8",
    "cvt.pk.f32.bf8": r"@llvm\.amdgcn\.cvt\.pk\.f32\.bf8",
    "cvt.scalef32": r"@llvm\.amdgcn\.cvt\.scalef32",
    "s.barrier": r"@llvm\.amdgcn\.s\.barrier",
    "s.waitcnt": r"@llvm\.amdgcn\.s\.waitcnt",
    "sched.barrier": r"@llvm\.amdgcn\.sched\.barrier",
    "sched.group.barrier": r"@llvm\.amdgcn\.sched\.group\.barrier",
    "s.setprio": r"@llvm\.amdgcn\.s\.setprio",
    "exp2.f32": r"@llvm\.amdgcn\.exp2\.f32",
    "rcp.f32": r"@llvm\.amdgcn\.rcp\.f32",
}

DEFAULT_STRUCTURAL_PATTERNS = {
    "load": r"^\s*%\S+\s*=\s*load\b",
    "store": r"^\s*store\b",
    "fmul": r"\bfmul\b",
    "fadd": r"\bfadd\b",
    "fsub": r"\bfsub\b",
    "fmuladd": r"@llvm\.fmuladd",
    "br": r"^\s*br\s",
    "phi": r"\bphi\b",
    "select": r"^\s*%\S+\s*=\s*select\b",
}


def count_intrinsics(
    ir_text: str,
    *,
    intrinsics: dict | None = None,
    structural: dict | None = None,
) -> dict:
    """Count intrinsic / structural patterns in lowered LLVM IR text."""
    intrinsics = intrinsics or DEFAULT_INTRINSIC_PATTERNS
    structural = structural or DEFAULT_STRUCTURAL_PATTERNS
    out: dict = {"ir_bytes": len(ir_text), "ir_lines": ir_text.count("\n")}
    out["intrinsics"] = {
        label: len(re.findall(pat, ir_text, flags=re.MULTILINE))
        for label, pat in intrinsics.items()
    }
    out["structural"] = {
        label: len(re.findall(pat, ir_text, flags=re.MULTILINE))
        for label, pat in structural.items()
    }
    return out


def probe_intrinsic_counts(
    entries: Iterable[tuple[str, "object"]],
    *,
    intrinsics: dict | None = None,
    structural: dict | None = None,
    print_zero: bool = False,
) -> list[dict]:
    """Lower each kernel and pretty-print the per-intrinsic histogram."""
    rows: list[dict] = []
    for label, kdef in entries:
        try:
            ir = lower_kernel_to_llvm(kdef)
        except Exception as e:  # noqa: BLE001
            print(f"\n=== {label} === LOWER-FAIL: {type(e).__name__}: {e}")
            continue
        counts = count_intrinsics(ir, intrinsics=intrinsics, structural=structural)
        counts["label"] = label
        rows.append(counts)
        print(f"\n=== {label} ===")
        print(f"  IR bytes: {counts['ir_bytes']}  lines: {counts['ir_lines']}")
        print("  intrinsics:")
        for name, n in counts["intrinsics"].items():
            if n or print_zero:
                print(f"    {name:<40} {n:>8}")
        print("  structural:")
        for name, n in counts["structural"].items():
            if n or print_zero:
                print(f"    {name:<40} {n:>8}")
    return rows


# ---- Demo --------------------------------------------------------------


def _demo_attention_tiled_2d() -> None:
    from rocke.instances.gfx950.attention_tiled_2d import (
        UnifiedAttention2DTiledSpec,
        build_unified_attention_2d_tiled,
    )

    base = dict(
        head_size=64,
        block_size=32,
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
    )
    specs = [
        (
            "baseline_mw16",
            UnifiedAttention2DTiledSpec(
                **base, num_warps=1, tile_size=64, block_m_per_warp=16
            ),
        ),
        (
            "mfma32",
            UnifiedAttention2DTiledSpec(
                **base,
                num_warps=1,
                tile_size=64,
                block_m_per_warp=32,
                use_mfma_32x32=True,
            ),
        ),
        (
            "mfma32_transposed",
            UnifiedAttention2DTiledSpec(
                **base,
                num_warps=1,
                tile_size=64,
                block_m_per_warp=32,
                use_mfma_32x32=True,
                use_transposed_qk_32x32=True,
            ),
        ),
    ]
    entries = [(label, build_unified_attention_2d_tiled(spec)) for label, spec in specs]
    probe_intrinsic_counts(entries)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--demo", choices=["attention_tiled_2d"], default="attention_tiled_2d"
    )
    p.add_argument(
        "--print-zero",
        action="store_true",
        help="also print intrinsics whose count is zero",
    )
    args = p.parse_args(argv)
    if args.demo == "attention_tiled_2d":
        _demo_attention_tiled_2d()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
