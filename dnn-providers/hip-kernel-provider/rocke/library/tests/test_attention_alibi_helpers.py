# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host checks of the ALiBi helpers (``kernels/common/_attention_alibi.py``) and
of the ALiBi flag of the unified tiled specs and runner.

The SSA helpers are evaluated with a numpy stand-in for ``IRBuilder`` that
implements only the methods they call, in fp32/i32 like the GPU.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest
from kernels.common import _attention_alibi as alibi
from kernels.common.attention_unified import (
    UnifiedAttentionProblem,
    run_unified_attention_torch,
)
from kernels.gfx942.attention_tiled_2d import (
    UnifiedAttention2DTiledSpec as Gfx942Spec2D,
)
from kernels.gfx942.attention_tiled_3d import (
    UnifiedAttention3DTiledSpec as Gfx942Spec3D,
)
from kernels.gfx950.attention_tiled_2d import (
    UnifiedAttention2DTiledSpec as Gfx950Spec2D,
)
from kernels.gfx950.attention_tiled_3d import (
    UnifiedAttention3DTiledSpec as Gfx950Spec3D,
)
from rocke.numeric.sdpa_reference import alibi_slopes

# hipDNN's TestCpuFpReferenceSdpa alibi slope vector for 6 heads.
_HIPDNN_H6 = [0.25, 0.0625, 0.015625, 0.00390625, 0.5, 0.125]


class _NumpyBuilder:
    """Evaluates the builder calls the ALiBi helpers make."""

    def const_i32(self, v):
        return np.int32(v)

    def const_f32(self, v):
        return np.float32(v)

    def add(self, a, c):
        return np.int32(a + c)

    def sub(self, a, c):
        return np.int32(a - c)

    def shl(self, a, c):
        return np.int32(np.int64(a) << int(c))

    def lshr(self, a, c):
        return np.int32(np.uint32(a) >> np.uint32(c))

    def cmp_lt(self, a, c):
        return bool(a < c)

    def select(self, cond, x, y):
        return x if cond else y

    def sitofp_f32(self, v):
        return np.float32(v)

    def bitcast(self, v, ty):
        target = np.int32 if ty.name == "i32" else np.float32
        return np.array([v]).view(target)[0]

    def fmul(self, a, c):
        return np.float32(a * c)

    def fneg(self, a):
        return np.float32(-a)

    def fma(self, a, c, d):
        return np.float32(np.float64(a) * np.float64(c) + np.float64(d))

    def exp2(self, a):
        return np.float32(np.exp2(np.float64(a)))

    def mul(self, a, c):
        return np.int32(a * c)

    def const_i64(self, v):
        return np.int64(v)

    def cmp_ne(self, a, c):
        return bool(a != c)

    # A slopes pointer is a numpy array, or None for a null pointer.
    def global_ptr_to_i64(self, ptr):
        return np.int64(0 if ptr is None else 0x1000)

    def buffer_rsrc(self, ptr, num_bytes):
        return ptr, int(num_bytes)

    # Python runs both regions of an if/else: the branch not taken is
    # evaluated without memory access and its yield is dropped.
    _live = True

    def scf_if_else(self, cond, result_types=()):
        return _NumpyIfElse(self, bool(cond))

    def scf_yield(self, *values):
        if self._live:
            self._yielded = values

    def buffer_load(self, rsrc, voffset, soffset, dtype):
        ptr, num_bytes = rsrc
        if not self._live:
            return np.float32(0.0)
        off = int(voffset) + int(soffset)
        if off < 0 or off + 4 > num_bytes:
            return np.float32(0.0)  # out of range reads 0, no access
        return np.float32(ptr[off // 4])


class _NumpyIfElse:
    def __init__(self, b, cond):
        self.b, self.cond, self.results = b, cond, None

    def __enter__(self):
        return self._region(True), self._region(False)

    def __exit__(self, *exc):
        self.b._live = True
        self.results = self.b._yielded
        return False

    @contextlib.contextmanager
    def _region(self, is_then):
        self.b._live = is_then == self.cond
        yield
        self.b._live = True


def test_reference_slopes_match_hipdnn_values():
    assert list(alibi_slopes(6)) == _HIPDNN_H6


@pytest.mark.parametrize("num_heads", [1, 3, 6, 8, 12, 40, 56, 112, 128])
@pytest.mark.parametrize("runtime_heads", [False, True])
def test_builtin_slope_matches_reference(num_heads, runtime_heads):
    b = _NumpyBuilder()
    want = list(alibi_slopes(num_heads))
    nh = np.int32(num_heads) if runtime_heads else num_heads
    for h in range(num_heads):
        got = alibi.alibi_slope_log2(b, np.int32(h), num_heads=nh, slopes=None)
        assert got == pytest.approx(want[h] * alibi._LOG2E, rel=2**-22)


def test_tile_bias_layout():
    b = _NumpyBuilder()
    slope = np.float32(0.25)
    tile_key0, anchor = np.int32(128), np.int32(150)
    for lane in (0, 31, 32, 63):
        lane_off = np.int32((lane // 32) * 4)
        bias = alibi.alibi_tile_bias(b, slope, tile_key0, lane_off, anchor, n_sub=2)
        for nsub in range(2):
            for i in range(16):
                key = 128 + nsub * 32 + (i // 4) * 8 + (lane // 32) * 4 + i % 4
                assert bias[nsub][i] == pytest.approx(0.25 * (key - 150))


def test_lse_shift():
    b = _NumpyBuilder()
    assert alibi.alibi_lse_shift(b, np.float32(0.5), np.int32(9), np.int32(4)) == 2.5


@pytest.mark.parametrize("num_heads", [1, 6, 12, 40])
def test_null_slopes_pointer_gives_builtin_slopes(num_heads):
    b = _NumpyBuilder()
    want = list(alibi_slopes(num_heads))
    for h in range(num_heads):
        got = alibi.alibi_slope(b, np.int32(h), num_heads=num_heads, slopes=None)
        assert got == pytest.approx(want[h], rel=2**-22)
    # A padded row's head is past num_heads: a finite slope, never read back.
    past = alibi.alibi_slope(b, np.int32(num_heads), num_heads=num_heads, slopes=None)
    assert np.isfinite(past)


@pytest.mark.parametrize("runtime_heads", [False, True])
def test_caller_slopes_read_bounded_by_num_heads(runtime_heads):
    b = _NumpyBuilder()
    slopes = np.array([0.5, 0.25, 0.125], dtype=np.float32)
    nh = np.int32(3) if runtime_heads else 3
    for h in range(3):
        assert alibi.alibi_slope(b, np.int32(h), num_heads=nh, slopes=slopes) == (
            slopes[h]
        )
    assert alibi.alibi_slope(b, np.int32(3), num_heads=nh, slopes=slopes) == 0


_SPEC_BASE = dict(
    head_size=64,
    block_size=32,
    num_query_heads=8,
    num_kv_heads=8,
    dtype="bf16",
    use_sinks=False,
    sliding_window=0,
    has_softcap=False,
)
_TILED_SPECS = [
    (Gfx942Spec2D, {}),
    (Gfx950Spec2D, {}),
    (Gfx942Spec3D, {"num_segments": 4}),
    (Gfx950Spec3D, {"num_segments": 4}),
]


@pytest.mark.parametrize("spec_cls,extra", _TILED_SPECS)
def test_tiled_spec_alibi_name_token(spec_cls, extra):
    def tokens(**kw):
        return spec_cls(**_SPEC_BASE, **extra, **kw).kernel_name().split("_")

    assert "alibi" in tokens(use_alibi=True)
    assert "alibi" not in tokens()


class _FakeSlopes:
    """The tensor attributes ``check_alibi_slopes`` reads."""

    def __init__(self, shape, dtype="torch.float32", contiguous=True):
        self.shape, self.dtype, self._contiguous = shape, dtype, contiguous
        self.is_cuda = True

    def is_contiguous(self):
        return self._contiguous


def _problem(**kw):
    return UnifiedAttentionProblem(
        total_q=4,
        num_seqs=1,
        num_query_heads=8,
        num_kv_heads=8,
        head_size=64,
        block_size=32,
        max_seqlen_q=4,
        max_seqlen_k=64,
        dtype="bf16",
        **kw,
    )


@pytest.mark.parametrize(
    "problem_kw,slopes,match",
    [
        ({}, object(), "use_alibi is off"),
        ({"use_alibi": True}, _FakeSlopes((4,)), "shape"),
        ({"use_alibi": True}, _FakeSlopes((8,), dtype="torch.float16"), "float32"),
        ({"use_alibi": True}, _FakeSlopes((8,), contiguous=False), "contiguous"),
    ],
)
def test_unified_runner_checks_slopes_against_problem(problem_kw, slopes, match):
    # Checked before any tensor is touched, so no device is needed.
    with pytest.raises(ValueError, match=match):
        run_unified_attention_torch(
            problem=_problem(**problem_kw),
            q=None,
            k=None,
            v=None,
            out=None,
            cu_seqlens_q=None,
            seqused_k=None,
            softmax_scale=1.0,
            block_table=None,
            softcap=0.0,
            alibi_slopes=slopes,
        )
