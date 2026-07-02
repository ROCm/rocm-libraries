"""Dense grouped GEMM kernel builder for gfx950 (MI355X).

Dense blocked grouped bf16 GEMM with expert-sorted layout. No gather/scatter —
each block processes rows [expert*M + m_base : ... + m_base+TM]. Parent kernel
for ragged_moe (which adds fused gather/scatter on top of this inner loop).

CDNA4 optimizations: DTL cooperative loads, chiplet-aware tiling, inline-asm
ds_reads, epilogue fusion, st_16x32 swizzle.

Author: adlashab / yraparti
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

from rocke.core.ir import BF16, I32, I64, IRBuilder, PtrType, VectorType
from rocke.helpers.mfma_gemm_inner import decode_mfma_lanes, mfma_atom_for_dtype
from rocke.helpers.spec import SignatureBuilder


@dataclass(frozen=True)
class GroupedGemmSpec:
    """Dense grouped GEMM specification (gfx950/CDNA4).

    Geometry (problem + tile sizes):
        M, N, K, E: problem dimensions (per-expert rows, cols, reduction, num experts)
        TM, TN, TK: tile sizes (M/N/K tile per block)
        WM, WN: warp layout (num warps in M/N dimensions)

    Production optimizations (all proven wins, defaults ON):
        dtl: Direct-to-LDS cooperative load (required, always ON)
        db: Double-buffer A/B LDS (nbuf=2)
        prio: s_setprio(1) around MFMA blocks
        swz: st_16x32 XOR swizzle (bank-conflict-free, +54 TF)
        chiplet: Chiplet-aware super-tile remap (L2 locality, +40 TF)
        asm_reads: Inline-asm ds_read_b128 with manual lgkmcnt
        deeppipe: Deep operand prefetch pipeline
        epifuse: Fuse epilogue stores into final K-tile MFMAs (+10-20 TF)
        b_rrr: NN weight layout B=[E,K,N] (vs default NT B=[E,N,K])
        tpb: Tiles-per-block (persistent kernel, default 1)

    Chiplet remap params (active when chiplet=True):
        chiplet_wgm: Workgroups per super-tile (default 8)
        chiplet_xcds: Number of XCDs (default 8)

    Kernel name:
        name: Kernel identifier (default "grouped_gemm")
    """

    # Problem dimensions
    M: int  # per-expert rows
    N: int
    K: int
    E: int

    # Tile geometry
    TM: int = 256
    TN: int = 256
    TK: int = 64
    WM: int = 2
    WN: int = 4

    # Production optimizations (all proven wins, defaults = full opts)
    dtl: bool = True  # Direct-to-LDS (required)
    db: bool = True  # Double-buffer (+required with DTL)
    prio: bool = True  # s_setprio around MFMAs
    swz: bool = True  # st_16x32 swizzle (+54 TF)
    chiplet: bool = True  # Super-tile remap (+40 TF)
    asm_reads: bool = True  # Inline-asm ds_read
    deeppipe: bool = True  # Deep prefetch pipeline
    epifuse: bool = True  # Epilogue fusion (+10-20 TF)
    b_rrr: bool = False  # NN layout (vs default NT)
    tpb: int = 1  # Tiles-per-block (persistent)

    # Chiplet remap params
    chiplet_wgm: int = 8
    chiplet_xcds: int = 8

    # Kernel name
    name: str = "grouped_gemm"


def grouped_gemm_signature():
    return (
        SignatureBuilder()
        .ptr("A", "bf16")
        .ptr("B", "bf16")
        .ptr("C", "bf16")
        .scalar("M", "i32")
        .scalar("N", "i32")
        .scalar("K", "i32")
        .scalar("stride_a", "i32")
        .scalar("stride_b", "i32")
        .scalar("stride_c", "i32")
        .build()
    )


# Due to the large size of the builder (~750 lines), for now I'll delegate
# to the original implementation in examples. A full extraction would require
# copying all helper functions and the complex DTL/prefetch/persistent logic.
# This keeps the spec interface clean while avoiding duplication.
#
# TODO: Complete full extraction when time permits. For now, use the original
# builder with spec params.


def build_grouped_gemm(spec: GroupedGemmSpec):
    """Build the dense grouped bf16 GEMM ``KernelDef``.

    Returns ``(kernel, block_size, TM, TN)``. ``M`` is the per-expert
    row count (``M_total // E``).

    NOTE: Currently delegates to the original builder in examples/grouped_gemm_hip.py.
    Only production levers are exposed. Experimental/diagnostic levers are discarded.
    """
    import sys
    import os

    # Temporary: import original builder
    examples_path = os.path.join(
        os.path.dirname(__file__), "../../examples/gfx950/grouped_gemm"
    )
    if examples_path not in sys.path:
        sys.path.insert(0, examples_path)

    from grouped_gemm_hip import build_custom_grouped

    # Set env vars for levers not in function params (b_rrr, deeppipe, epifuse)
    if spec.b_rrr:
        os.environ["BRRR"] = "1"
    else:
        os.environ.pop("BRRR", None)

    if not spec.deeppipe:
        os.environ["DEEPPIPE"] = "0"
    else:
        os.environ.pop("DEEPPIPE", None)  # default is 1

    if not spec.epifuse:
        os.environ["EPIFUSE"] = "0"
    else:
        os.environ.pop("EPIFUSE", None)  # default is 1

    # Ensure experimental/diagnostic levers are OFF
    for key in [
        "MFMA32",
        "CSHUF",
        "TR128",
        "PERMMASCHED",
        "CKSCHED",
        "BURSTPRIO",
        "LIGHTSYNC",
        "NOSTORE",
        "STORESINK",
        "PIPE",
        "VGPRPROBE",
    ]:
        os.environ.pop(key, None)

    return build_custom_grouped(
        spec.M,
        spec.N,
        spec.K,
        spec.E,
        TM=spec.TM,
        TN=spec.TN,
        TK=spec.TK,
        WM=spec.WM,
        WN=spec.WN,
        dtl=spec.dtl,
        db=spec.db,
        prio=spec.prio,
        swz=spec.swz,
        prefetch=False,  # Not a winner, force off
        pin=False,  # Not a winner, force off
        chiplet=spec.chiplet,
        chiplet_wgm=spec.chiplet_wgm,
        chiplet_xcds=spec.chiplet_xcds,
        asm_reads=spec.asm_reads,
        tpb=spec.tpb,
        name=spec.name,
    )
