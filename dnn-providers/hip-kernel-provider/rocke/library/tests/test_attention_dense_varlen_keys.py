# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host checks for a packed varlen dense-attention launch without the LSE output.

The packed body zeroes no keyless row and has no handling for an empty KV sequence
without LSE, so the launcher rejects both instead of returning wrong values. No test
needs torch or a GPU: the helper reads only ``dtype`` and ``tolist()`` of its offsets,
and the launcher reaches the check before it reads q, k, v or out, so the launcher
tests pass stand-ins for them and run in the CPU attention lane.
"""

from __future__ import annotations

import pytest

from kernels.common._lse_store import validate_varlen_keys
from kernels.common.attention_dense_spec import DENSE_TILE_GEOMETRIES
from kernels.gfx950.attention_dense import (
    Gfx950AttentionDenseSpec,
    run_attention_dense_torch,
)


def _spec(**over) -> Gfx950AttentionDenseSpec:
    tile = DENSE_TILE_GEOMETRIES["default"]
    kw = dict(
        batch=2,
        seqlen_q=512,
        seqlen_kv=256,
        num_query_heads=4,
        num_kv_heads=1,
        head_size=128,
        causal=True,
        dtype="bf16",
        block_m=int(tile["block_m"]),
        block_n=int(tile["block_n"]),
        varlen=True,
    )
    kw.update(over)
    return Gfx950AttentionDenseSpec(**kw)


class _Offsets:
    """Stand-in for a ``cu_seqlens`` tensor: the helper reads only dtype and tolist()."""

    def __init__(self, lengths, dtype="torch.int32"):
        self.dtype = dtype
        self._values = [0]
        for n in lengths:
            self._values.append(self._values[-1] + n)

    def tolist(self):
        return list(self._values)


def _offsets(*lengths, dtype="torch.int32"):
    return _Offsets(lengths, dtype)


def test_accepts_nonempty_sequences():
    validate_varlen_keys(_spec(), _offsets(256, 128), _offsets(256, 128))


def test_no_window_allows_q_longer_than_kv():
    """Top-left causal rows past kv_len still see every key, so none is keyless."""
    validate_varlen_keys(_spec(), _offsets(512, 256), _offsets(256, 256))


def test_window_accepts_sequences_that_all_keep_a_key():
    """q_len - kv_len = 127 < W: the last row (q = 382) still sees key 255."""
    validate_varlen_keys(
        _spec(sliding_window=128), _offsets(383, 256), _offsets(256, 256)
    )


@pytest.mark.parametrize("kv_lengths", [(256, 0), (0, 256)])
def test_rejects_an_empty_kv_sequence(kv_lengths):
    with pytest.raises(NotImplementedError, match="must be nonempty"):
        validate_varlen_keys(_spec(), _offsets(256, 256), _offsets(*kv_lengths))


def test_accepts_an_inactive_slot_with_no_queries_and_no_keys():
    """A slot with q_len 0 exits before touching KV, so kv_len 0 there is fine."""
    validate_varlen_keys(_spec(), _offsets(256, 0), _offsets(256, 0))


def test_window_rejects_a_sequence_that_leaves_rows_without_a_key():
    """The second sequence has q_len - kv_len = 128 = W: its last row sees no key."""
    with pytest.raises(NotImplementedError, match="no key"):
        validate_varlen_keys(
            _spec(sliding_window=128), _offsets(256, 384), _offsets(256, 256)
        )


@pytest.mark.parametrize(
    "q_lengths,kv_lengths", [((256,), (256, 256)), ((256, 256), (256,))]
)
def test_rejects_offsets_of_the_wrong_length(q_lengths, kv_lengths):
    with pytest.raises(ValueError, match=r"batch\+1 entries"):
        validate_varlen_keys(_spec(), _offsets(*q_lengths), _offsets(*kv_lengths))


def test_rejects_offsets_that_are_not_int32():
    with pytest.raises(ValueError, match="must be int32"):
        validate_varlen_keys(
            _spec(), _offsets(256, 256, dtype="torch.int64"), _offsets(256, 256)
        )


# The launcher reads q, k, v and out only after the varlen check (and validate_dense_lse
# returns at once without LSE), so a bare object stands in for each: a launch that got
# past the check would fail on the first attribute it read, not with the check's error.
_UNREAD = object()


def _launch(spec, cu_seqlens_q, cu_seqlens_kv, **kw):
    run_attention_dense_torch(
        spec=spec,
        q=_UNREAD,
        k=_UNREAD,
        v=_UNREAD,
        out=_UNREAD,
        scale=0.1,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        **kw,
    )


def test_launcher_runs_the_check_when_lse_is_off():
    """The public launcher reaches the check before it touches the device."""
    with pytest.raises(NotImplementedError, match="must be nonempty"):
        _launch(
            _spec(seqlen_q=256, seqlen_kv=256),
            _offsets(256, 256),
            _offsets(256, 0),
        )


@pytest.mark.parametrize("validate", [True, False])
def test_launcher_check_is_skipped_only_when_validate_varlen_is_off(
    validate, monkeypatch
):
    """``validate_varlen=False`` skips the host check (it reads the offsets, a device sync).

    The next launcher step is stubbed to stop the call, so no kernel is built or run."""
    import kernels.gfx950.attention_dense as dense

    class _Stop(Exception):
        pass

    calls = []
    monkeypatch.setattr(
        dense, "validate_varlen_keys", lambda *a: calls.append(a) or None
    )

    def _stop(*a, **kw):
        raise _Stop

    monkeypatch.setattr(dense, "validate_dense_lse", _stop)
    spec = _spec(seqlen_q=256, seqlen_kv=256)
    cu_q, cu_kv = _offsets(256, 256), _offsets(256, 256)
    with pytest.raises(_Stop):
        _launch(spec, cu_q, cu_kv, validate_varlen=validate)
    assert calls == ([(spec, cu_q, cu_kv)] if validate else [])
