# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# Co-located, build-time producer for the gemm_wmma family: emit the gfx1151
# no-LDS one-wave-per-16x16-tile *reference* WMMA GEMM (RCR) .co (HSACO), for
# f16 and bf16, into argv[1]. Runtime never touches rocke -- the per-family
# CMakeLists runs this at build time to drop the .co next to the checked-in
# family.json (family.json is the source of truth; this script emits .co ONLY).
#
# Normally run by this family's CMakeLists via the rocke build interpreter
# (${ROCKE_PYENV_PYTHON}); the editable pyenv puts `rocke` on the path with no
# PYTHONPATH surgery. To run standalone, use that interpreter, e.g.:
#   <build>/rocke-pyenv/bin/python produce_gemm_wmma_co.py <out_dir>
#
# TWO DTYPES, TWO SOURCES:
#   * f16  -- built from rocKE's real instance
#     `rocke.instances.gfx1151.wmma_gemm.build_wmma_gemm` (WmmaGemmSpec hard-
#     restricts dtype to "fp16" via a __post_init__ guard). We use it as a
#     *library* (no rocKE edit).
#   * bf16 -- WmmaGemmSpec rejects bf16, so rather than EDIT that rocKE instance
#     (standing constraint: do not modify rocKE), we replicate its ~30-line
#     builder body here with the bf16 substitutions (param dtype F16->BF16,
#     global_load_vN(...,BF16,16), wmma_f32_16x16x16_bf16, cast_f32_to(el, BF16)).
#     The gfx1151 bf16 WMMA atom shares the f16 fragment ABI exactly.
#
# Both emitted kernels share ONE launch contract (encoded in family.json):
# block [32,1,1] (one wave per 16x16 output tile), grid (ceil(M/16), ceil(N/16),
# 1) grid_order "MN" (block_id.x -> M-tile), ABI (A,B,C ptrs; M,N,K i32),
# workspace 0, RCR layout. Symbols are auto-derived and MUST match family.json's
# co_file names (rocke_wmma_gemm_wmma16x16x16_{fp16,bf16}_rcr_xm).
#
# CRITICAL gfx1151 gotcha: this is a wave32 kernel (max_workgroup_size = 32). The
# WMMA fragment ABI does the cross-lane distribution; there are no shuffles, so
# unlike the rmsnorm reduction there is no wave64 miscompile trap here -- but the
# grid/block MUST launch exactly one 32-lane wave per tile.

import os
import sys

from rocke.core.ir import BF16, I32, IRBuilder, PtrType
from rocke.helpers.compile import compile_kernel
from rocke.instances.gfx1151.wmma_gemm import WmmaGemmSpec, build_wmma_gemm

ARCH = "gfx1151"

_WMMA_M = 16
_WMMA_N = 16
_WMMA_K = 16
_WAVE = 32


def _build_wmma_gemm_bf16(symbol: str):
    """Replicate rocke.instances.gfx1151.wmma_gemm.build_wmma_gemm for bf16
    (WmmaGemmSpec rejects non-fp16). RCR: C[M,N] = A[M,K] @ B[N,K]^T, one wave
    (32 lanes) per 16x16 output tile, no LDS."""
    elem = BF16
    b = IRBuilder(symbol)
    # One wave per block; the flat-work-group cap is baked into the descriptor.
    b.kernel.attrs["max_workgroup_size"] = _WAVE

    A = b.param("A", PtrType(elem, "global"), noalias=True, readonly=True, align=16)
    Bp = b.param("B", PtrType(elem, "global"), noalias=True, readonly=True, align=16)
    C = b.param("C", PtrType(elem, "global"), noalias=True, writeonly=True, align=16)
    _M = b.param("M", I32)  # noqa: F841 (M is implied by the grid; kept for ABI parity)
    N = b.param("N", I32)
    K = b.param("K", I32)

    c0 = b.const_i32(0)
    c16 = b.const_i32(_WMMA_K)
    c32 = b.const_i32(_WAVE)

    # Wave-relative lane and its fragment coordinates.
    lane = b.mod(b.thread_id_x(), c32)
    frag = b.mod(lane, c16)  # lane%16: A-frag row, B-frag row, output col
    half = b.div(lane, c16)  # lane/16: 0 or 1, selects even/odd output rows

    # grid_order "MN": block_id.x -> M-tile, block_id.y -> N-tile.
    m0 = b.mul(b.block_id_x(), c16)
    n0 = b.mul(b.block_id_y(), c16)

    # Per-lane global row bases (element offsets, row-major):
    #   A[m0+frag][k] = (m0+frag)*K + k ;  B[n0+frag][k] = (n0+frag)*K + k
    a_base = b.mul(b.add(m0, frag), K)
    b_base = b.mul(b.add(n0, frag), K)

    # K-loop accumulating the <8 x float> WMMA fragment.
    acc0 = b.zero_vec_f32(8)
    loop = b.scf_for_iter(c0, K, c16, [("acc", acc0)], iv_name="k0")
    with loop as (k0, (acc,)):
        a_frag = b.global_load_vN(A, b.add(a_base, k0), elem, 16)
        b_frag = b.global_load_vN(Bp, b.add(b_base, k0), elem, 16)
        nacc = b.wmma_f32_16x16x16_bf16(a_frag, b_frag, acc)
        b.scf_yield(nacc)
    acc = loop.results[0]

    # Epilogue: slot i of lane l -> (row = m0 + 2*i + l/16, col = n0 + l%16).
    out_col = b.add(n0, frag)
    for i in range(8):
        el = b.vec_extract(acc, i)
        h = b.cast_f32_to(el, elem)
        out_row = b.add(m0, b.add(b.const_i32(2 * i), half))
        idx = b.add(b.mul(out_row, N), out_col)
        b.global_store(C, idx, h)

    return b.kernel


def _emit(out_dir, symbol, kernel):
    artifact = compile_kernel(kernel, arch=ARCH)
    # A zero-byte .co passes the fs::exists gate at catalog load and is catalogued
    # as a valid kernel, failing only later at hipModuleLoad; fail loudly here.
    if not artifact.hsaco:
        raise SystemExit(f"ERROR {symbol}: compiled .co is empty")
    out_path = os.path.join(out_dir, symbol + ".co")
    with open(out_path, "wb") as f:
        f.write(artifact.hsaco)
    print(f"symbol={symbol} bytes={len(artifact.hsaco)} path={out_path}")


def main() -> int:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out_dir, exist_ok=True)

    # f16 via the real rocKE instance (WmmaGemmSpec dtype defaults to "fp16",
    # block_x_is_m=True -> symbol order "xm").
    f16_spec = WmmaGemmSpec(name="rocke_wmma_gemm", dtype="fp16")
    _emit(out_dir, f16_spec.kernel_name(), build_wmma_gemm(f16_spec, arch=ARCH))

    # bf16 via the replicated body (symbol mirrors the fp16 spelling convention).
    bf16_symbol = "rocke_wmma_gemm_wmma16x16x16_bf16_rcr_xm"
    _emit(out_dir, bf16_symbol, _build_wmma_gemm_bf16(bf16_symbol))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
