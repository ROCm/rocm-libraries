# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host checks of a paged varlen dense launch (packed Q + ``cu_seqlens_q`` over a
paged K/V cache), on tensor stand-ins: no torch or GPU needed."""

import dataclasses
import importlib
from unittest import mock

import pytest

from ._dense_paged_stand_ins import CacheStandIn, I32DeviceTensor

_ARCHES = ("gfx950",)
_B, _SQ, _HQ, _HKV, _D, _BS, _NB = 3, 1000, 16, 4, 128, 16, 128
_GARBAGE = 0x7FFF_FFFF
# Sequence lengths 1000, 17 and 0: a partial last page, a partial page and tile
# and an empty sequence. The table is as wide as the longest sequence needs; the
# columns past each sequence's own pages hold garbage the kernel never reads.
_LENS = [1000, 17, 0]
_CU = [0, 1000, 1017, 1017]
_TOTAL = _CU[-1]
_WIDTH = -(-_SQ // _BS)
_TABLE = [
    list(range(_WIDTH)),
    [_WIDTH, _WIDTH + 1] + [_GARBAGE] * (_WIDTH - 2),
    [_GARBAGE] * _WIDTH,
]
_SENTINEL = RuntimeError("reached-compile")


def _module(arch):
    return importlib.import_module(f"kernels.{arch}.attention_dense")


def _spec(arch):
    ad = _module(arch)
    spec_cls = getattr(ad, "Gfx942AttentionDenseSpec", None) or ad.AttentionDenseSpec
    return spec_cls(
        batch=_B,
        seqlen_q=_SQ,
        seqlen_kv=_SQ,
        num_query_heads=_HQ,
        num_kv_heads=_HKV,
        head_size=_D,
        causal=True,
        dtype="fp16",
        varlen=True,
        paged=True,
        block_size=_BS,
        num_kv_blocks=_NB,
    )


def _inputs(**overrides):
    cache = CacheStandIn((_NB, _BS, _HKV, _D))
    packed = CacheStandIn((_TOTAL, _HQ, _D))
    call = dict(
        q=packed,
        k=cache,
        v=cache,
        out=packed,
        scale=1.0,
        block_tables=I32DeviceTensor([list(row) for row in _TABLE]),
        kv_lens=I32DeviceTensor(list(_LENS)),
        cu_seqlens_q=I32DeviceTensor(list(_CU)),
    )
    call.update(overrides)
    return call


def _launch(arch, spec=None, **kw):
    """Run the launcher up to compile; a launch that passes every check raises the
    compile sentinel."""
    ad = _module(arch)
    with mock.patch.dict(ad._DENSE_LAUNCHER_CACHE, clear=True), mock.patch(
        "rocke.helpers.compile.compile_kernel", side_effect=_SENTINEL
    ):
        ad.run_attention_dense_torch(spec=spec or _spec(arch), **kw)


def _assert_reaches_compile(arch, **kw):
    with pytest.raises(RuntimeError) as err:
        _launch(arch, **kw)
    assert err.value is _SENTINEL


@pytest.mark.parametrize("arch", _ARCHES)
def test_valid_launch_reaches_compile(arch):
    """Ragged lengths, an empty sequence and garbage in unused table columns."""
    _assert_reaches_compile(arch, **_inputs())


_STRUCTURAL = [
    ("missing_cu", dict(cu_seqlens_q=None), "requires cu_seqlens_q"),
    (
        "cu_seqlens_kv",
        dict(cu_seqlens_kv=I32DeviceTensor(list(_CU))),
        "cu_seqlens_kv must not",
    ),
    ("cu_shape", dict(cu_seqlens_q=I32DeviceTensor(_CU[:-1])), "cu_seqlens_q shape"),
    (
        "cu_dtype",
        dict(cu_seqlens_q=I32DeviceTensor(list(_CU), dtype="torch.int64")),
        "cu_seqlens_q dtype",
    ),
    (
        "cu_host",
        dict(cu_seqlens_q=I32DeviceTensor(list(_CU), is_cuda=False)),
        "cu_seqlens_q must be a GPU tensor",
    ),
    (
        "cu_device",
        dict(cu_seqlens_q=I32DeviceTensor(list(_CU), device="cuda:1")),
        "cu_seqlens_q must be on the K/V device",
    ),
    (
        "cu_stride",
        dict(cu_seqlens_q=I32DeviceTensor(list(_CU), inner_stride=2)),
        "cu_seqlens_q innermost stride",
    ),
    (
        "q_unpacked",
        dict(q=CacheStandIn((1, _TOTAL, _HQ, _D))),
        "q shape",
    ),
    (
        "q_past_maxima",
        dict(q=CacheStandIn((_B * _SQ + 1, _HQ, _D))),
        "q shape",
    ),
    ("out_shape", dict(out=CacheStandIn((_TOTAL + 1, _HQ, _D))), "out shape"),
    (
        "q_strided",
        dict(q=CacheStandIn((_TOTAL, _HQ, _D), contiguous=False)),
        "q must be contiguous",
    ),
    (
        "out_device",
        dict(out=CacheStandIn((_TOTAL, _HQ, _D), device="cuda:1")),
        "out must be on the K/V device",
    ),
]


@pytest.mark.parametrize("arch", _ARCHES)
@pytest.mark.parametrize(
    "overrides,error",
    [pytest.param(o, e, id=i) for i, o, e in _STRUCTURAL],
)
def test_structural_rules_hold_without_value_validation(arch, overrides, error):
    with pytest.raises(ValueError, match=error):
        _launch(arch, **_inputs(**overrides), validate_paged=False)


def _bad_table():
    table = [list(row) for row in _TABLE]
    table[1][1] = _NB  # the second (last used) page of sequence 1
    return I32DeviceTensor(table)


_VALUES = [
    ("cu_start", dict(cu_seqlens_q=I32DeviceTensor([1, 1000, 1017, 1017])), "start"),
    ("cu_end", dict(cu_seqlens_q=I32DeviceTensor([0, 1000, 1017, 1016])), "end at"),
    (
        "cu_decreasing",
        dict(cu_seqlens_q=I32DeviceTensor([0, 600, 500, 1017])),
        "sequence 1 length -100 outside",
    ),
    (
        "q_len_past_seqlen_q",
        dict(cu_seqlens_q=I32DeviceTensor([0, 1001, 1001, 1017])),
        r"outside \[0, seqlen_q",
    ),
    (
        "kv_len_past_seqlen_kv",
        dict(kv_lens=I32DeviceTensor([1001, 17, 0])),
        r"kv_lens\[0\]=1001 outside",
    ),
    (
        "kv_len_negative",
        dict(kv_lens=I32DeviceTensor([1000, 17, -1])),
        r"kv_lens\[2\]=-1 outside",
    ),
    (
        "q_len_ne_kv_len",
        dict(cu_seqlens_q=I32DeviceTensor([0, 1000, 1016, 1017])),
        r"q_len=16 != kv_lens\[1\]=17",
    ),
    ("used_block_id", dict(block_tables=_bad_table()), f"block id {_NB} outside"),
]


@pytest.mark.parametrize("arch", _ARCHES)
@pytest.mark.parametrize(
    "overrides,error",
    [pytest.param(o, e, id=i) for i, o, e in _VALUES],
)
def test_value_rules_are_checked_and_skippable(arch, overrides, error):
    with pytest.raises(ValueError, match=error):
        _launch(arch, **_inputs(**overrides))
    # validate_paged=False skips the synchronized content checks only.
    _assert_reaches_compile(arch, **_inputs(**overrides), validate_paged=False)


@pytest.mark.parametrize("arch", _ARCHES)
def test_one_host_copy_per_tensor(arch):
    """The value checks synchronize once per metadata tensor, not per sequence."""
    real = I32DeviceTensor.tolist
    with mock.patch.object(
        I32DeviceTensor, "tolist", autospec=True, side_effect=real
    ) as tolist:
        _assert_reaches_compile(arch, **_inputs())
    assert tolist.call_count == 3  # block_tables, kv_lens, cu_seqlens_q


@pytest.mark.parametrize("arch", _ARCHES)
def test_single_sequence_spec_refuses_cu_seqlens(arch):
    """The single-sequence paged kernel has no cu_seqlens_q parameter."""
    spec = dataclasses.replace(
        _spec(arch), batch=1, varlen=False, seqlen_q=1024, seqlen_kv=1024
    )
    with pytest.raises(ValueError, match="spec.varlen is False"):
        _launch(
            arch,
            spec=spec,
            **_inputs(
                q=CacheStandIn((1, 1024, _HQ, _D)),
                out=CacheStandIn((1, 1024, _HQ, _D)),
                block_tables=I32DeviceTensor([list(range(64))]),
                kv_lens=I32DeviceTensor([1024]),
                cu_seqlens_q=I32DeviceTensor([0, 1024]),
            ),
        )
