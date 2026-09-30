# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""WMMA GEMM for gfx1201 (RDNA4 / Navi 48).

Mirror of ``instances/gfx1151/wmma_gemm.py`` but built around the RDNA4 WMMA
fragment ABI, which differs from RDNA3/3.5 in three ways:

  * **No cross-half operand duplication.** The A/B fragments are ``<8 x half>``
    per lane (vs ``<16 x half>`` on gfx11), and the 16 K-elements of one WMMA
    are split across the two lane-halves: lanes 0-15 carry K 0..7, lanes 16-31
    carry K 8..15. So per K-step (stride 16) lane ``l`` loads 8 contiguous
    elements starting at K = ``k0 + (l//16)*8``.
  * **Distinct builtin / intrinsic** (``..._w32_gfx12`` / ``...v8f32.v8f16``),
    selected via the ``wmma_gfx12_f32_16x16x16_f16`` op_id.
  * **Column-distributed accumulator.** Slot ``i`` of lane ``l`` maps to output
    ``(row = m0 + (l//16)*8 + i, col = n0 + l%16)``.

Layout is otherwise RCR (``C = A @ B.T``, A row-major ``M×K``, B row-major
``N×K``), one wave (32 lanes) per ``16×16`` output tile, no LDS. The lane maps
encoded here are the *hypothesis* exercised by ``examples/gfx1201/wmma_probe.py``
with asymmetric inputs; do not trust the matmul_nbits port until the probe
matches numpy bit-for-bit.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...core.ir import F16, I32, IRBuilder, KernelDef, PtrType

_WMMA_M = 16
_WMMA_N = 16
_WMMA_K = 16
_WAVE = 32
_HALF_K = 8  # K-elements per lane-half (gfx12: 8, no duplication)


@dataclass(frozen=True)
class WmmaGemmSpec:
    """A gfx1201 WMMA GEMM instance (fp16, fixed 16x16x16 WMMA tile)."""

    name: str = "rocke_wmma_gemm_gfx12"
    dtype: str = "fp16"

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

        return kernel_name_join(self.name, "wmma16x16x16", self.dtype, "rcr")


def is_valid_spec(spec: WmmaGemmSpec, arch: str = "gfx1201"):
    """Return ``(ok, reason)``. The gfx12 WMMA 16x16x16 f16 atom must exist."""
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
            f"WMMA {_WMMA_M}x{_WMMA_N}x{_WMMA_K} {spec.dtype} atom absent on {arch}"
        )
    if target.wave_size != _WAVE:
        return False, f"this WMMA kernel is wave32; {arch} is wave{target.wave_size}"
    return True, "ok"


def build_wmma_gemm(spec: WmmaGemmSpec, arch: str = "gfx1201") -> KernelDef:
    """Build the gfx1201 WMMA GEMM ``KernelDef`` (RCR, f16)."""
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid WMMA GEMM spec: {why}")

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = _WAVE

    A = b.param("A", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    Bp = b.param("B", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    C = b.param("C", PtrType(F16, "global"), noalias=True, writeonly=True, align=16)
    M = b.param("M", I32)  # noqa: F841 (M implied by the grid; kept for ABI parity)
    N = b.param("N", I32)
    K = b.param("K", I32)

    c0 = b.const_i32(0)
    c16 = b.const_i32(_WMMA_K)
    c32 = b.const_i32(_WAVE)

    lane = b.mod(b.thread_id_x(), c32)
    frag = b.mod(lane, c16)  # lane%16: A-frag row, B-frag col, output col
    half = b.div(lane, c16)  # lane/16: 0 or 1 -> selects the K-half (operands)
    #                          and the row-block of 8 (accumulator)
    half_k = b.mul(half, b.const_i32(_HALF_K))  # (l//16)*8 : K offset within step

    # Tile assignment matches the host launcher (run_manifest ``_gemm_problem``)
    # and the canonical ROCKE GEMM convention: block_id.x -> N-tile,
    # block_id.y -> M-tile. (A reversed mapping is invisible for square M=N but
    # transposes / drops tiles for non-square shapes.)
    m0 = b.mul(b.block_id_y(), c16)  # output-tile row base
    n0 = b.mul(b.block_id_x(), c16)  # output-tile col base

    # Per-lane global row bases (element offsets, row-major):
    #   A[m0+frag][k] = (m0+frag)*K + k ;  B[n0+frag][k] = (n0+frag)*K + k
    a_base = b.mul(b.add(m0, frag), K)
    b_base = b.mul(b.add(n0, frag), K)

    # K-loop accumulating the <8 x float> WMMA fragment. Each lane loads only its
    # 8 K-elements (K = k0 + half*8 .. +7); the two lane-halves together cover
    # the full K=16 of one WMMA step.
    acc0 = b.zero_vec_f32(8)
    loop = b.scf_for_iter(c0, K, c16, [("acc", acc0)], iv_name="k0")
    with loop as (k0, (acc,)):
        a_off = b.add(b.add(a_base, k0), half_k)
        b_off = b.add(b.add(b_base, k0), half_k)
        a_frag = b.global_load_vN_f16(A, a_off, 8)
        b_frag = b.global_load_vN_f16(Bp, b_off, 8)
        nacc = b.wmma_gfx12_f32_16x16x16_f16(a_frag, b_frag, acc)
        b.scf_yield(nacc)
    acc = loop.results[0]

    # Epilogue (column-distributed): slot i of lane l ->
    #   (row = m0 + (l//16)*8 + i, col = n0 + l%16).
    out_col = b.add(n0, frag)
    row_base = b.add(m0, half_k)
    for i in range(8):
        elem = b.vec_extract(acc, i)
        h = b.trunc_f32_to_f16(elem)
        out_row = b.add(row_base, b.const_i32(i))
        idx = b.add(b.mul(out_row, N), out_col)
        b.global_store(C, idx, h)

    return b.kernel


def wmma_gemm_grid(M: int, N: int):
    """Launch grid (gx, gy, 1) for problem (M, N): one wave per 16x16 tile.

    Matches the kernel's block mapping (``block_id.x -> N-tile``,
    ``block_id.y -> M-tile``): ``gx`` tiles N, ``gy`` tiles M. Swapping these is
    invisible for square M=N but launches the wrong grid for non-square shapes.
    """
    return ((N + _WMMA_N - 1) // _WMMA_N, (M + _WMMA_M - 1) // _WMMA_M, 1)
