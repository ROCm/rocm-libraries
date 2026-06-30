# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Inspect a compiled CK DSL kernel for occupancy limiters.

Builds each (label, KernelDef) pair, compiles to HSACO, extracts AMDHSA
notes via ``llvm-readelf --notes``, and reports VGPR/AGPR/SGPR/LDS plus
a coarse estimate of how many waves and workgroups will fit per CU.

The script accepts arbitrary builders, not just attention. Drive it
from a small custom script that builds the specs you care about, or
from the CLI for the shipped attention demo.

CLI example (the shipped attention demo):
    python probe_occupancy.py --demo attention_tiled_2d

Programmatic use:

    from probe_occupancy import probe_occupancy
    from rocke.instances.gfx950.attention_tiled_2d import (
        UnifiedAttention2DTiledSpec, build_unified_attention_2d_tiled,
    )
    specs = [
        ("w4t64", UnifiedAttention2DTiledSpec(head_size=64, block_size=32,
            num_query_heads=64, num_kv_heads=8, dtype="bf16",
            use_sinks=True, sliding_window=0, has_softcap=False,
            num_warps=4, tile_size=64)),
    ]
    probe_occupancy([(l, build_unified_attention_2d_tiled(s), s.num_warps)
                     for l, s in specs])

Notes:
- Occupancy math is intentionally conservative: gfx950 VGPR allocation
  granularity is 16 VGPRs and waves-per-EU is a soft hint, not a hard
  cap. Use this for relative ranking, not for absolute predictions.
- For absolute numbers, prefer rocprof's ``OccupancyPct`` column or the
  ``ROCm Compute Profiler`` reports — but those need a live kernel
  launch; this script is static, runs in <1 s per spec, and works in
  CI.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# ---- CK DSL bootstrap --------------------------------------------------


def _bootstrap_rocke() -> None:
    """Add rocke to ``sys.path`` if not already importable."""
    try:
        import rocke  # noqa: F401

        return
    except ImportError:
        pass
    # utilities/tools/dsl_probes/<this>  -> dsl_docs/optimization/...
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

from rocke.helpers.compile import compile_kernel  # noqa: E402


# ---- Hardware caps -----------------------------------------------------


@dataclass(frozen=True)
class ArchCaps:
    """Coarse per-CU resource caps used for the occupancy estimate."""

    name: str
    lds_bytes_per_cu: int
    max_waves_per_cu: int
    vgpr_per_cu: int  # private VGPRs (not counting AGPRs on CDNA)
    agpr_per_cu: int  # accumulator VGPRs (gfx9x0 only)
    vgpr_alloc_granularity: int
    waves_per_simd: int  # SIMDs/CU
    simds_per_cu: int


# Defaults are gfx950 (MI355X). Override for older parts.
ARCH_GFX950 = ArchCaps(
    name="gfx950",
    lds_bytes_per_cu=160 * 1024,
    max_waves_per_cu=32,
    vgpr_per_cu=512 * 4,  # 512 VGPRs/SIMD × 4 SIMDs (per CU sum)
    agpr_per_cu=256 * 4,
    vgpr_alloc_granularity=16,
    waves_per_simd=8,
    simds_per_cu=4,
)
ARCH_GFX942 = ArchCaps(  # MI300X
    name="gfx942",
    lds_bytes_per_cu=64 * 1024,
    max_waves_per_cu=32,
    vgpr_per_cu=512 * 4,
    agpr_per_cu=256 * 4,
    vgpr_alloc_granularity=16,
    waves_per_simd=8,
    simds_per_cu=4,
)
ARCH_GFX90A = ArchCaps(  # MI250X
    name="gfx90a",
    lds_bytes_per_cu=64 * 1024,
    max_waves_per_cu=32,
    vgpr_per_cu=512 * 4,
    agpr_per_cu=256 * 4,
    vgpr_alloc_granularity=8,
    waves_per_simd=8,
    simds_per_cu=4,
)
ARCH_BY_NAME = {a.name: a for a in (ARCH_GFX950, ARCH_GFX942, ARCH_GFX90A)}


# ---- HSACO notes parsing -----------------------------------------------

_NOTE_FIELDS = [
    ("vgpr_count", r"\.vgpr_count:\s+(\d+)"),
    ("sgpr_count", r"\.sgpr_count:\s+(\d+)"),
    ("agpr_count", r"\.agpr_count:\s+(\d+)"),
    ("vgpr_spill_count", r"\.vgpr_spill_count:\s+(\d+)"),
    ("sgpr_spill_count", r"\.sgpr_spill_count:\s+(\d+)"),
    ("lds_size", r"\.group_segment_fixed_size:\s+(\d+)"),
    ("scratch_size", r"\.private_segment_fixed_size:\s+(\d+)"),
    ("max_flat_workgroup_size", r"\.max_flat_workgroup_size:\s+(\d+)"),
    ("kernarg_segment_size", r"\.kernarg_segment_size:\s+(\d+)"),
]


def _readelf_path() -> str:
    """Pick the best available ``llvm-readelf``."""
    candidates = [
        os.environ.get("LLVM_READELF"),
        "/opt/rocm/llvm/bin/llvm-readelf",
        "llvm-readelf",
        "readelf",
    ]
    for c in candidates:
        if not c:
            continue
        try:
            subprocess.run([c, "--version"], capture_output=True, timeout=5, check=True)
            return c
        except (
            FileNotFoundError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ):
            continue
    raise FileNotFoundError(
        "no llvm-readelf found; install /opt/rocm/llvm/bin/llvm-readelf "
        "or set LLVM_READELF"
    )


def parse_hsaco_notes(hsaco_bytes: bytes) -> dict:
    """Extract AMDHSA notes (VGPR/AGPR/SGPR/LDS/spill) from a HSACO blob."""
    readelf = _readelf_path()
    with tempfile.NamedTemporaryFile(suffix=".hsaco", delete=False) as f:
        f.write(hsaco_bytes)
        hsaco_path = f.name
    try:
        proc = subprocess.run(
            [readelf, "--notes", hsaco_path],
            capture_output=True,
            text=True,
            timeout=20,
        )
        notes = proc.stdout
    finally:
        try:
            os.unlink(hsaco_path)
        except OSError:
            pass
    out: dict = {}
    for field, pattern in _NOTE_FIELDS:
        m = re.search(pattern, notes)
        if m:
            out[field] = int(m.group(1))
    return out


# ---- Occupancy math ----------------------------------------------------


def _align_up(n: int, granularity: int) -> int:
    return ((n + granularity - 1) // granularity) * granularity


def estimate_occupancy(
    *,
    notes: dict,
    waves_per_wg: int,
    arch: ArchCaps,
    waves_per_eu_hint: int | None = None,
) -> dict:
    """Compute coarse waves-per-CU bounds from VGPR / LDS / hint."""
    vgpr = max(notes.get("vgpr_count", 0), 1)
    agpr = notes.get("agpr_count", 0)
    lds = max(notes.get("lds_size", 0), 1)

    vgpr_alloc = _align_up(vgpr, arch.vgpr_alloc_granularity)
    agpr_alloc = _align_up(agpr, arch.vgpr_alloc_granularity) if agpr else 0

    # VGPRs are allocated per wave from a per-SIMD pool of 512 VGPRs on
    # CDNA3+ — so the wave cap from VGPRs is per-SIMD, not per-CU.
    vgpr_per_simd = arch.vgpr_per_cu // arch.simds_per_cu
    waves_from_vgpr = vgpr_per_simd // vgpr_alloc if vgpr_alloc else arch.waves_per_simd
    waves_from_vgpr = min(waves_from_vgpr, arch.waves_per_simd)

    if agpr_alloc:
        agpr_per_simd = arch.agpr_per_cu // arch.simds_per_cu
        waves_from_agpr = agpr_per_simd // agpr_alloc
        waves_from_agpr = min(waves_from_agpr, arch.waves_per_simd)
    else:
        waves_from_agpr = arch.waves_per_simd

    if waves_per_eu_hint is not None:
        waves_from_hint = min(waves_per_eu_hint, arch.waves_per_simd)
    else:
        waves_from_hint = arch.waves_per_simd

    # LDS is per-WG and shared across SIMDs; convert to "wgs/CU".
    wgs_from_lds = arch.lds_bytes_per_cu // lds if lds else arch.max_waves_per_cu

    waves_per_simd = min(waves_from_vgpr, waves_from_agpr, waves_from_hint)
    waves_per_cu_from_regs = waves_per_simd * arch.simds_per_cu
    waves_per_cu_from_lds = wgs_from_lds * waves_per_wg
    waves_per_cu = min(
        waves_per_cu_from_regs,
        waves_per_cu_from_lds,
        arch.max_waves_per_cu,
    )
    wgs_per_cu = waves_per_cu // max(waves_per_wg, 1)

    return {
        "vgpr_alloc": vgpr_alloc,
        "agpr_alloc": agpr_alloc,
        "waves_per_simd_from_vgpr": waves_from_vgpr,
        "waves_per_simd_from_agpr": waves_from_agpr,
        "waves_per_simd_from_hint": waves_from_hint,
        "wgs_from_lds": wgs_from_lds,
        "waves_per_cu": waves_per_cu,
        "wgs_per_cu": wgs_per_cu,
        "limited_by": _limiter_label(
            waves_from_vgpr,
            waves_from_agpr,
            waves_from_hint,
            wgs_from_lds,
            arch,
            waves_per_wg,
        ),
    }


def _limiter_label(
    v: int,
    a: int,
    h: int,
    lds_wg: int,
    arch: ArchCaps,
    waves_per_wg: int,
) -> str:
    regs = min(v, a, h)
    lds_waves = lds_wg * waves_per_wg
    if (
        regs * arch.simds_per_cu < lds_waves
        and regs * arch.simds_per_cu < arch.max_waves_per_cu
    ):
        if v <= a and v <= h:
            return "VGPR"
        if a <= v and a <= h:
            return "AGPR"
        return "WAVES_PER_EU_HINT"
    if lds_waves < arch.max_waves_per_cu:
        return "LDS"
    return "MAX_WAVES_PER_CU"


# ---- The probe ---------------------------------------------------------


def probe_occupancy(
    entries: Iterable[tuple[str, "object", int]],
    *,
    arch: ArchCaps = ARCH_GFX950,
    print_header: bool = True,
    waves_per_eu_hint: int | None = None,
) -> list[dict]:
    """Compile each entry and report occupancy.

    Each ``entries`` item is ``(label, kernel_def, waves_per_workgroup)``.
    Returns a list of dicts with the parsed numbers, one per entry.
    """
    rows: list[dict] = []
    if print_header:
        print(
            f"{'label':<28} {'vgpr':>5} {'agpr':>5} {'sgpr':>5} {'spill':>5} "
            f"{'lds B':>7} {'waves/CU':>9} {'wg/CU':>6} {'limit':>10}"
        )
        print("-" * 96)
    for label, kdef, waves_per_wg in entries:
        try:
            artifact = compile_kernel(kdef)
        except Exception as e:  # noqa: BLE001
            print(f"{label:<28}  BUILD-FAIL: {type(e).__name__}: {e}")
            continue
        try:
            notes = parse_hsaco_notes(artifact.hsaco)
        except Exception as e:  # noqa: BLE001
            print(f"{label:<28}  READELF-FAIL: {type(e).__name__}: {e}")
            continue
        occ = estimate_occupancy(
            notes=notes,
            waves_per_wg=waves_per_wg,
            arch=arch,
            waves_per_eu_hint=waves_per_eu_hint,
        )
        row = {"label": label, **notes, **occ}
        rows.append(row)
        print(
            f"{label:<28} "
            f"{notes.get('vgpr_count', 0):>5} "
            f"{notes.get('agpr_count', 0):>5} "
            f"{notes.get('sgpr_count', 0):>5} "
            f"{notes.get('vgpr_spill_count', 0):>5} "
            f"{notes.get('lds_size', 0):>7} "
            f"{occ['waves_per_cu']:>9} "
            f"{occ['wgs_per_cu']:>6} "
            f"{occ['limited_by']:>10}"
        )
    return rows


# ---- Demos -------------------------------------------------------------


def _demo_attention_tiled_2d(arch: ArchCaps) -> None:
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
        use_sinks=True,
        sliding_window=0,
        has_softcap=False,
    )
    specs = [
        (
            "w4t64_mw16",
            UnifiedAttention2DTiledSpec(
                **base, num_warps=4, tile_size=64, block_m_per_warp=16
            ),
        ),
        (
            "w4t64_mw32",
            UnifiedAttention2DTiledSpec(
                **base, num_warps=4, tile_size=64, block_m_per_warp=32
            ),
        ),
        (
            "w2t64_mw32",
            UnifiedAttention2DTiledSpec(
                **base, num_warps=2, tile_size=64, block_m_per_warp=32
            ),
        ),
        ("w1t32", UnifiedAttention2DTiledSpec(**base, num_warps=1, tile_size=32)),
        ("w4t32", UnifiedAttention2DTiledSpec(**base, num_warps=4, tile_size=32)),
        ("w4t64", UnifiedAttention2DTiledSpec(**base, num_warps=4, tile_size=64)),
        ("w8t64", UnifiedAttention2DTiledSpec(**base, num_warps=8, tile_size=64)),
    ]
    entries = [
        (label, build_unified_attention_2d_tiled(spec), spec.num_warps)
        for label, spec in specs
    ]
    probe_occupancy(entries, arch=arch)


def _demo_implicit_gemm(arch: ArchCaps) -> None:
    from rocke.instances.common.conv_implicit_gemm import (
        ConvProblem,
        ImplicitGemmConvSpec,
        build_implicit_gemm_conv,
    )

    problem = ConvProblem(
        N=8,
        Hi=56,
        Wi=56,
        C=64,
        K=64,
        Y=3,
        X=3,
        sH=1,
        sW=1,
        pH=1,
        pW=1,
        dH=1,
        dW=1,
    )
    specs = [
        (
            "baseline_mem_sync",
            ImplicitGemmConvSpec(
                problem=problem,
                name="rocke_baseline",
                tile_m=64,
                tile_n=64,
                tile_k=64,
                warp_m=2,
                warp_n=2,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
                pipeline="mem",
                epilogue="cshuffle",
                async_dma=False,
            ),
        ),
        (
            "async_dma",
            ImplicitGemmConvSpec(
                problem=problem,
                name="rocke_async",
                tile_m=64,
                tile_n=64,
                tile_k=64,
                warp_m=2,
                warp_n=2,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
                pipeline="compv4",
                epilogue="cshuffle",
                async_dma=True,
            ),
        ),
    ]
    entries = [
        (label, build_implicit_gemm_conv(spec), 4)  # 4 waves per WG
        for label, spec in specs
    ]
    probe_occupancy(entries, arch=arch)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--demo",
        choices=["attention_tiled_2d", "implicit_gemm"],
        default="attention_tiled_2d",
        help="which builder to exercise as a smoke probe",
    )
    p.add_argument(
        "--arch",
        choices=list(ARCH_BY_NAME.keys()),
        default="gfx950",
        help="GPU arch caps to use for the occupancy estimate",
    )
    args = p.parse_args(argv)
    arch = ARCH_BY_NAME[args.arch]
    if args.demo == "attention_tiled_2d":
        _demo_attention_tiled_2d(arch)
    else:
        _demo_implicit_gemm(arch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
