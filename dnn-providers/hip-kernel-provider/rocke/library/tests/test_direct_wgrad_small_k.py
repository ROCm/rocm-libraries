# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The gfx1250 small-K direct wgrad (``DirectConvWgradSpec.small_k``), CPU only.

small_k runs the small_c GEMM on the swapped problem for layers whose kpg is
the small side: X (cpg channels) is the 8-channel vector operand, dY (kpg
channels) the im2col operand at padding KH - 1 - PAD over X's pixel range, and
the epilogue writes GEMM element [c][(r', s', k)] to dW[k][KH-1-r'][KW-1-s'][c].
These tests pin the identity in numpy, the spec's swapped geometry, the kernel
name, the launch grid and which operand each load reads in the built IR. The
refusals are in ``test_direct_wgrad_refusals.py``, the C++ engine's emission in
parity configs 371..376 of ``tests/parity/conv_direct_grouped_emit.py`` and the
GPU numerics in ``test_direct_conv_correctness.py::test_wgrad_small_k``.
"""

from __future__ import annotations

import numpy as np
import pytest

from kernels.common.conv_direct_grouped import (
    DirectConvProblem,
    DirectConvWgradSpec,
    build_direct_conv_wgrad,
    direct_launch_geometry,
    is_valid_wgrad_spec,
)
from kernels.conv.direct_variants import make_direct_spec, validate_direct_spec

_OUT = DirectConvProblem(
    N=2, H=17, W=23, groups=1, cpg=64, kpg=3, KH=3, KW=3, PAD=1, stride=1
)


def _sk(problem=_OUT, **knobs) -> DirectConvWgradSpec:
    return DirectConvWgradSpec(
        problem=problem, wave_size=32, small_c=True, small_k=True, **knobs
    )


def _walk(ops):
    for op in ops:
        yield op
        for region in op.regions:
            yield from _walk(region.ops)


def _direct_dw(x, dy, KH, KW, PAD):
    """dW[k, r, s, c] = sum_p X[n, ho + r - PAD, wo + s - PAD, c] * dY[n, ho, wo, k]."""
    N, H, W, C = x.shape
    _, Ho, Wo, K = dy.shape
    xp = np.zeros((N, H + 2 * KH, W + 2 * KW, C))
    xp[:, KH : KH + H, KW : KW + W] = x
    dw = np.zeros((K, KH, KW, C))
    for r in range(KH):
        for s in range(KW):
            h0, w0 = KH + r - PAD, KW + s - PAD
            win = xp[:, h0 : h0 + Ho, w0 : w0 + Wo]
            dw[:, r, s, :] = np.einsum("nhwk,nhwc->kc", dy, win)
    return dw


@pytest.mark.parametrize("KH, PAD", [(3, 1), (3, 0), (3, 2), (5, 1)])
def test_swapped_gemm_with_flipped_taps_is_dw(KH, PAD):
    """The identity small_k builds on: the GEMM of X against dY gathered at
    padding KH - 1 - PAD over X's pixels, written to the flipped tap, is dW."""
    rng = np.random.default_rng(7)
    N, H, W, C, K = 2, 7, 9, 5, 3
    Ho, Wo = H + 2 * PAD - KH + 1, W + 2 * PAD - KH + 1
    x = rng.standard_normal((N, H, W, C))
    dy = rng.standard_normal((N, Ho, Wo, K))
    pad_sw = KH - 1 - PAD
    # The swapped GEMM: G[c, r', s', k] = sum_(n,hi,wi) X[n,hi,wi,c] *
    # dY[n, hi + r' - pad_sw, wi + s' - pad_sw, k].
    g = _direct_dw(dy, x, KH, KH, pad_sw)  # [C][KH][KW][K]
    dw = np.empty((K, KH, KH, C))
    for r_ in range(KH):
        for s_ in range(KH):
            dw[:, KH - 1 - r_, KH - 1 - s_, :] = g[:, r_, s_, :].T
    np.testing.assert_allclose(dw, _direct_dw(x, dy, KH, KH, PAD), rtol=1e-12)


def test_small_k_spec_swaps_the_gemm_channels_and_padding():
    spec = _sk(vec_x=4)
    assert (spec.sc_cpg, spec.sc_kpg, spec.sc_pad) == (3, 64, 1)
    # One filter row's (s, k) run: KW * kpg = 9, padded to the vec_x unit.
    assert spec.small_c_run == 12
    assert spec.small_c_n == 3 * 12
    p5 = DirectConvProblem(
        N=1, H=9, W=9, groups=1, cpg=16, kpg=3, KH=5, KW=5, PAD=1, stride=1
    )
    assert _sk(p5).sc_pad == 3
    plain = DirectConvWgradSpec(problem=_OUT, wave_size=32, small_c=True)
    assert (plain.sc_cpg, plain.sc_kpg, plain.sc_pad) == (64, 3, 1)


def test_small_k_kernel_name_differs_from_small_c():
    sk = _sk(vec_x=4).kernel_name()
    assert "_sc_sk_" in sk
    sc = DirectConvWgradSpec(
        problem=_OUT, wave_size=32, small_c=True, vec_x=4
    ).kernel_name()
    assert "_sk" not in sc and sc != sk


def test_small_k_grid_tiles_the_swapped_gemm():
    """Grid x tiles M = cpg by block_k and N = KH * run(KW * kpg) by block_c;
    grid y is the pixel split."""
    spec = _sk(waves_k=2, wave_tiles_k=1, wave_tiles_c=1, row_blocks=5)
    grid, block = direct_launch_geometry(spec)
    n_k = -(-64 // spec.block_k)
    n_n = -(-spec.small_c_n // spec.block_c)
    assert grid == (n_k * n_n, 5, 1)
    assert block[0] == spec.threads_per_block


def test_small_k_reads_x_as_the_vector_operand():
    """With gathered runs (vec_x 0), every 16 B load of a pixel's 8 channels
    reads X (param B) and every gathered element reads dY (param A); small_c
    on the same problem class does the opposite."""

    def loads(spec):
        kernel = build_direct_conv_wgrad(spec, arch="gfx1250")
        src = {}
        vec, gather = set(), set()
        for op in _walk(kernel.body.ops):
            if op.name == "tile.buffer_rsrc":
                src[id(op.results[0])] = op.operands[0].name.lstrip("%")
            elif op.name.startswith("tile.buffer_load_vN"):
                vec.add(src[id(op.operands[0])])
            elif op.name.startswith("tile.buffer_load_"):
                gather.add(src[id(op.operands[0])])
        return vec, gather

    assert loads(_sk()) == ({"B"}, {"A"})
    stem = DirectConvProblem(
        N=2, H=17, W=23, groups=1, cpg=3, kpg=64, KH=3, KW=3, PAD=1, stride=1
    )
    assert loads(DirectConvWgradSpec(problem=stem, wave_size=32, small_c=True)) == (
        {"A"},
        {"B"},
    )


def test_small_k_epilogue_steps_dw_by_channel():
    """The epilogue's atomics cover the C_FRAG slots of every wave tile, as
    small_c's do: one guarded fp32 atomic per slot."""
    spec = _sk(wave_tiles_k=2, wave_tiles_c=2)
    kernel = build_direct_conv_wgrad(spec, arch="gfx1250")
    atomics = [
        op for op in _walk(kernel.body.ops) if op.name == "tile.buffer_atomic_fadd_f32"
    ]
    assert len(atomics) == 8 * spec.wave_tiles_k * spec.wave_tiles_c


def test_direct_wgrad_small_k_variant_builds_the_small_k_spec():
    spec = make_direct_spec(
        "direct_wgrad_small_k",
        _OUT,
        dict(
            waves_k=1,
            waves_c=1,
            wave_tiles_k=2,
            wave_tiles_c=2,
            vec_x=0,
            sub_steps=2,
            ws_replicas=8,
            uniform_wave=False,
            row_blocks=512,
        ),
        wave_size=32,
    )
    assert spec.small_c and spec.small_k
    assert spec.kernel_name().startswith("rocke_direct_conv_wgrad_sk_")
    assert validate_direct_spec("direct_wgrad_small_k", spec, "gfx1250") == (True, "ok")
    assert is_valid_wgrad_spec(spec, arch="gfx1250") == (True, "ok")
