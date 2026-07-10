# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""WMMA GEMM for gfx1151 (RDNA3.5 / Strix Halo) — the first RDNA kernel.

This is an **arch-specific** instance (hybrid layout): RDNA has no MFMA, so the
algorithm is built around the gfx11 ``wmma_f32_16x16x16_f16`` instruction and a
**wave32** thread mapping — fundamentally different from the CDNA MFMA GEMM in
``instances/common/gemm_universal.py``, which is why it lives under
``instances/gfx1151/`` rather than ``common/``.

Layout (RCR, f16 in / f32 acc / f16 out; matches the ``run_manifest`` gemm
reference ``C = A @ B.T`` with A row-major ``M×K`` and B row-major ``N×K``):

  * One wave (32 lanes) computes one ``16×16`` output tile; grid is
    ``(ceil(M/16), ceil(N/16))`` with ``block_id.x -> M-tile, block_id.y ->
    N-tile`` (manifest ``grid_order="MN"``), ``threads_per_block=32``.
  * Per K-step (stride 16) each lane loads its WMMA operand fragments
    **directly from global** (no LDS): a ``<16 x half>`` A-row fragment and a
    ``<16 x half>`` B-row fragment, per the hardware-verified gfx1151 layout
    (lane ``l``: A row ``l%16``, B row ``l%16`` of the ``N×K`` B = math column
    ``l%16``; both replicated across lanes 0-15 / 16-31). The accumulator is a
    loop-carried ``<8 x float>``.
  * Epilogue: the 8 accumulator slots of lane ``l`` map to output
    ``(row = m0 + 2*i + l/16, col = n0 + l%16)``; each is truncated to f16 and
    stored.

No LDS, no cross-lane shuffles — the WMMA fragment ABI does the distribution.
This is a correctness-first kernel; LDS staging / multi-tile-per-wave tuning is
a follow-on.

Status note (MMA-contract unification): the Universal GEMM body in
``instances/common/gemm_universal.py`` now resolves the WMMA atom from the
target catalog and drives its fragment loads / accumulator scatter through the
op's layout maps, so ``build_universal_gemm(spec, arch="gfx1151")`` emits a
working WMMA GEMM from the *same* source as the CDNA MFMA path (verified with
``examples/common/universal_gemm_verify --arch gfx1151``). This standalone
kernel is kept as the minimal, no-LDS reference proof of the wave32 WMMA
fragment ABI; new gfx1151 GEMM work should go through the unified builder.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...core.ir import F16, I32, IRBuilder, KernelDef, PtrType
from ._wmma_common import (
    _WAVE,
    _WMMA_K,
    _WMMA_M,
    _WMMA_N,
    wmma16_grid,
    wmma32_wave_guard,
)


@dataclass(frozen=True)
class WmmaGemmSpec:
    """A gfx1151 WMMA GEMM instance. M/N/K are runtime kernel args; the spec
    only carries the dtype and a name (the tile is the fixed 16x16x16 WMMA)."""

    name: str = "rocke_wmma_gemm"
    dtype: str = "fp16"
    # Dispatch-order toggle (perf-only; correctness-neutral). True maps
    # ``block_id.x -> M-tile`` (grid_order "MN"); False maps
    # ``block_id.x -> N-tile`` (grid_order "NM", the universal-GEMM order).
    block_x_is_m: bool = True

    def __post_init__(self) -> None:
        if self.dtype != "fp16":
            raise ValueError(
                f"WmmaGemmSpec currently supports fp16 only, got {self.dtype!r}"
            )

    @property
    def block_size(self) -> int:
        return _WAVE

    def kernel_name(self) -> str:
        from ...helpers.spec import kernel_name_join

        order = "xm" if self.block_x_is_m else "xn"
        return kernel_name_join(self.name, "wmma16x16x16", self.dtype, "rcr", order)


def is_valid_spec(spec: WmmaGemmSpec, arch: str = "gfx1151"):
    """Return ``(ok, reason)``. The WMMA 16x16x16 f16 atom must exist on ``arch``."""
    from ...core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)
    if not target.mma.has_shape(
        family="wmma",
        a_dtype=spec.dtype,
        b_dtype=spec.dtype,
        c_dtype="fp32",
        m=_WMMA_M,
        n=_WMMA_N,
        k=_WMMA_K,
    ):
        return False, (
            f"WMMA {_WMMA_M}x{_WMMA_N}x{_WMMA_K} {spec.dtype} atom absent on {arch} "
            f"(WMMA is an RDNA/gfx11 instruction)"
        )
    return wmma32_wave_guard(target, arch)


def build_wmma_gemm(spec: WmmaGemmSpec, arch: str = "gfx1151") -> KernelDef:
    """Build the gfx1151 WMMA GEMM ``KernelDef`` (RCR, f16)."""
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid WMMA GEMM spec: {why}")

    b = IRBuilder(spec.kernel_name())
    # One wave per block. The flat-work-group-size cap is baked into the kernel
    # descriptor; launching more than this fails before the body runs.
    b.kernel.attrs["max_workgroup_size"] = _WAVE

    A = b.param("A", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    Bp = b.param("B", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    C = b.param("C", PtrType(F16, "global"), noalias=True, writeonly=True, align=16)
    M = b.param("M", I32)  # noqa: F841 (M is implied by the grid; kept for ABI parity)
    N = b.param("N", I32)
    K = b.param("K", I32)

    c0 = b.const_i32(0)
    c16 = b.const_i32(_WMMA_K)
    c32 = b.const_i32(_WAVE)

    # Wave-relative lane and its fragment coordinates.
    lane = b.mod(b.thread_id_x(), c32)
    frag = b.mod(lane, c16)  # lane%16: A-frag row, B-frag row, output col
    half = b.div(lane, c16)  # lane/16: 0 or 1, selects even/odd output rows

    if spec.block_x_is_m:
        m0 = b.mul(b.block_id_x(), c16)  # output-tile row base (grid_order "MN")
        n0 = b.mul(b.block_id_y(), c16)  # output-tile col base
    else:
        m0 = b.mul(b.block_id_y(), c16)  # output-tile row base (grid_order "NM")
        n0 = b.mul(b.block_id_x(), c16)  # output-tile col base

    # Per-lane global row bases (element offsets, row-major):
    #   A[m0+frag][k] = (m0+frag)*K + k ;  B[n0+frag][k] = (n0+frag)*K + k
    a_base = b.mul(b.add(m0, frag), K)
    b_base = b.mul(b.add(n0, frag), K)

    # K-loop accumulating the <8 x float> WMMA fragment.
    acc0 = b.zero_vec_f32(8)
    loop = b.scf_for_iter(c0, K, c16, [("acc", acc0)], iv_name="k0")
    with loop as (k0, (acc,)):
        a_frag = b.global_load_vN_f16(A, b.add(a_base, k0), 16)
        b_frag = b.global_load_vN_f16(Bp, b.add(b_base, k0), 16)
        nacc = b.wmma_f32_16x16x16_f16(a_frag, b_frag, acc)
        b.scf_yield(nacc)
    acc = loop.results[0]

    # Epilogue: slot i of lane l -> (row = m0 + 2*i + l/16, col = n0 + l%16).
    out_col = b.add(n0, frag)
    for i in range(8):
        elem = b.vec_extract(acc, i)
        h = b.trunc_f32_to_f16(elem)
        out_row = b.add(m0, b.add(b.const_i32(2 * i), half))
        idx = b.add(b.mul(out_row, N), out_col)
        b.global_store(C, idx, h)

    return b.kernel


def wmma_gemm_grid(M: int, N: int):
    """Launch grid (gx, gy, 1) for problem (M, N): one wave per 16x16 tile."""
    return wmma16_grid(M, N)
