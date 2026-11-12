# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host checks of a packed varlen dense-attention launch's offsets.

``validate_dense_varlen_offsets`` is the one check of the offsets: with LSE
``validate_dense_lse`` runs it, without LSE the launcher does. Besides malformed
offsets (``ValueError``) it refuses two valid inputs the packed body does not
implement (``NotImplementedError``): a sequence with queries and no keys, and, with a
sliding window and neither LSE nor an additive bias, a sequence whose last query rows
see no key (only those epilogues zero such a row). A slot with no queries may have no
keys.

No test needs torch or a GPU: the validator reads only ``device``, ``dtype``,
``shape``, ``is_contiguous()`` and ``tolist()`` of its offsets, and the launcher reaches
it before it reads anything else of q, k, v or out, so the launcher tests pass stand-ins
and run in the CPU attention lane.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import kernels.common._lse_store as lse_store
import kernels.gfx950.attention_dense as dense
from kernels.common._lse_store import validate_dense_varlen_offsets
from kernels.common.attention_dense_spec import DENSE_TILE_GEOMETRIES
from kernels.gfx950.attention_dense import (
    Gfx950AttentionDenseSpec,
    run_attention_dense_torch,
)

# Stand-in devices compare with ==, like torch devices.
_DEVICE = "cpu"
_HEADS, _KV_HEADS, _D = 4, 1, 128


def _spec(**over) -> Gfx950AttentionDenseSpec:
    tile = DENSE_TILE_GEOMETRIES["default"]
    kw = dict(
        batch=2,
        seqlen_q=512,
        seqlen_kv=512,
        num_query_heads=_HEADS,
        num_kv_heads=_KV_HEADS,
        head_size=_D,
        causal=True,
        dtype="bf16",
        block_m=int(tile["block_m"]),
        block_n=int(tile["block_n"]),
        varlen=True,
    )
    kw.update(over)
    return Gfx950AttentionDenseSpec(**kw)


class _Offsets:
    """Stand-in for a ``cu_seqlens`` tensor."""

    def __init__(self, lengths, dtype="torch.int32", device=_DEVICE):
        self.dtype = dtype
        self.device = device
        self._values = [0]
        for n in lengths:
            self._values.append(self._values[-1] + n)
        self.shape = (len(self._values),)

    def is_contiguous(self):
        return True

    def tolist(self):
        return list(self._values)


def _check(spec, q_lengths, kv_lengths, **over):
    """The validator on offsets built from per-sequence lengths; ``over`` replaces
    its packed extents or device."""
    kw = dict(q_tokens=sum(q_lengths), kv_tokens=sum(kv_lengths), device=_DEVICE)
    kw.update(over)
    validate_dense_varlen_offsets(spec, _Offsets(q_lengths), _Offsets(kv_lengths), **kw)


# The paths the validator serves: the plain launch, LSE on, and the additive bias
# (alone and with LSE). The empty-KV and idle-slot rules hold in all of them.
_PATHS = {
    "plain": {},
    "lse": {"emit_lse": True},
    "bias": {"use_additive_bias": True},
    "bias_lse": {"use_additive_bias": True, "emit_lse": True},
}


def test_accepts_nonempty_sequences():
    _check(_spec(), (256, 512), (256, 128))


def test_no_window_allows_q_longer_than_kv():
    """Top-left causal rows past kv_len still see every key, so none is keyless."""
    _check(_spec(), (512, 256), (256, 256))


def test_window_accepts_sequences_that_all_keep_a_key():
    """q_len - kv_len = 64 < W: the last row (q = 511) still sees key 447."""
    _check(_spec(sliding_window=128), (512, 256), (448, 256))


@pytest.mark.parametrize("path", _PATHS)
@pytest.mark.parametrize("kv_lengths", [(256, 0), (0, 256)])
def test_rejects_an_empty_kv_sequence_that_has_queries(path, kv_lengths):
    """The prologue scores KV tile 0 of every sequence: for an empty one that is the
    next sequence's keys (or K/V past the packed tensor, for the last). Without LSE
    the refusal is a NotImplementedError; with LSE it is the ValueError that path has
    always raised."""
    spec = _spec(**_PATHS[path])
    error = ValueError if spec.emit_lse else NotImplementedError
    with pytest.raises(error, match="must be nonempty"):
        _check(spec, (256, 256), kv_lengths)


@pytest.mark.parametrize("path", _PATHS)
@pytest.mark.parametrize(
    "q_lengths,kv_lengths",
    [((256, 0), (256, 0)), ((0, 256), (0, 256)), ((0, 0), (0, 0))],
    ids=["last-slot-idle", "first-slot-idle", "all-idle"],
)
def test_accepts_an_idle_slot_with_no_queries_and_no_keys(path, q_lengths, kv_lengths):
    """The packed body returns on ``q-block start >= seqlen_q`` before it loads K/V,
    so a slot with q_len 0 never reads its (empty) keys: accepted with LSE and bias
    too, though they refuse every other empty KV sequence."""
    _check(_spec(**_PATHS[path]), q_lengths, kv_lengths)


def test_window_rejects_a_sequence_that_leaves_rows_without_a_key():
    """The second sequence has q_len - kv_len = 128 = W: its last row sees no key."""
    with pytest.raises(NotImplementedError, match="no key"):
        _check(_spec(sliding_window=128), (256, 512), (256, 384))


@pytest.mark.parametrize("path", ["lse", "bias", "bias_lse"])
def test_window_keyless_rows_are_served_where_the_epilogue_zeroes_them(path):
    """The LSE and bias epilogues zero a keyless row, so the offsets the plain launch
    refuses above are fine there."""
    _check(_spec(sliding_window=128, **_PATHS[path]), (256, 512), (256, 384))


# id -> (q lengths, kv lengths, validator overrides, message regex). Spec maxima
# 512 / 512, block_m 256, block_n 64.
_MALFORMED = {
    "q-wrong-entry-count": ((256,), (256, 256), {}, r"cu_seqlens_q must be int32"),
    "kv-wrong-entry-count": ((256, 256), (256,), {}, r"cu_seqlens_kv must be int32"),
    "q-not-tile-aligned": ((200, 256), (256, 256), {}, r"cu_seqlens_q lengths must"),
    "kv-not-tile-aligned": ((256, 256), (200, 256), {}, r"cu_seqlens_kv lengths must"),
    "q-past-the-maximum": ((768, 256), (256, 256), {}, r"cu_seqlens_q lengths must"),
    "kv-past-the-maximum": ((256, 256), (640, 256), {}, r"cu_seqlens_kv lengths must"),
    "q-short-of-the-packed-tensor": (
        (256, 256),
        (256, 256),
        {"q_tokens": 1024},
        r"cu_seqlens_q endpoints must span",
    ),
    "kv-short-of-the-packed-tensor": (
        (256, 256),
        (256, 256),
        {"kv_tokens": 1024},
        r"cu_seqlens_kv endpoints must span",
    ),
}


@pytest.mark.parametrize("case", list(_MALFORMED))
def test_rejects_malformed_offsets_with_a_value_error(case):
    q_lengths, kv_lengths, over, match = _MALFORMED[case]
    with pytest.raises(ValueError, match=match):
        _check(_spec(), q_lengths, kv_lengths, **over)


def test_rejects_offsets_that_are_not_int32():
    with pytest.raises(ValueError, match="must be int32"):
        validate_dense_varlen_offsets(
            _spec(),
            _Offsets((256, 256), dtype="torch.int64"),
            _Offsets((256, 256)),
            q_tokens=512,
            kv_tokens=512,
            device=_DEVICE,
        )


@pytest.mark.parametrize("name", ["cu_seqlens_q", "cu_seqlens_kv"])
def test_rejects_offsets_on_another_device(name):
    """The kernel dereferences the offsets, so they must be on q's device."""
    offsets = {
        "cu_seqlens_q": _Offsets((256, 256)),
        "cu_seqlens_kv": _Offsets((256, 256)),
    }
    offsets[name] = _Offsets((256, 256), device="elsewhere")
    with pytest.raises(ValueError, match=rf"{name} must be on the same device as q"):
        validate_dense_varlen_offsets(
            _spec(),
            offsets["cu_seqlens_q"],
            offsets["cu_seqlens_kv"],
            q_tokens=512,
            kv_tokens=512,
            device=_DEVICE,
        )


def test_a_malformed_tensor_outranks_an_unimplemented_sequence():
    """A malformed tensor is a caller bug (ValueError), whatever the other one says."""
    with pytest.raises(ValueError, match="cu_seqlens_q lengths must"):
        _check(_spec(), (200, 256), (0, 256))


# The launcher reaches the offsets check after the LSE presence check and before it
# reads the head dims, so each operand stand-in carries only what the checks up to
# there read. A launch that got past the check would fail on the next attribute it
# read, not with the check's error.
def _operand(tokens, heads):
    return SimpleNamespace(
        device=_DEVICE,
        shape=(tokens, heads, _D),
        ndim=3,
        is_contiguous=lambda: True,
    )


def _launch(spec, q_lengths, kv_lengths, **kw):
    q_tokens, kv_tokens = sum(q_lengths), sum(kv_lengths)
    run_attention_dense_torch(
        spec=spec,
        q=_operand(q_tokens, _HEADS),
        k=_operand(kv_tokens, _KV_HEADS),
        v=_operand(kv_tokens, _KV_HEADS),
        out=_operand(q_tokens, _HEADS),
        scale=0.1,
        cu_seqlens_q=_Offsets(q_lengths),
        cu_seqlens_kv=_Offsets(kv_lengths),
        **kw,
    )


class _Stop(Exception):
    """Raised by a stub to end a launch right after the offsets check."""


def _stop(*args, **kwargs):
    raise _Stop


@pytest.fixture
def calls(monkeypatch):
    """The offsets-check calls of a launch, each tagged with the module that made it.
    The step after the check is stubbed to end the launch, so no kernel is built or
    run; the bias check and the LSE buffer's own contract (GPU tensors) are no-ops."""
    log = []
    for module, tag in ((dense, "launcher"), (lse_store, "validate_dense_lse")):
        monkeypatch.setattr(
            module,
            "validate_dense_varlen_offsets",
            lambda *a, _tag=tag, **kw: log.append((_tag, a, kw)),
        )
    monkeypatch.setattr(dense, "_check_head_dims", _stop)
    monkeypatch.setattr(dense, "validate_dense_additive_bias", lambda *a, **kw: None)
    monkeypatch.setattr(lse_store, "_check_lse_buffer", lambda *a, **kw: None)
    return log


def test_launcher_runs_the_check_when_lse_is_off():
    """The public launcher reaches the check before it touches the device."""
    with pytest.raises(NotImplementedError, match="must be nonempty"):
        _launch(_spec(), (256, 256), (256, 0))


def test_launcher_lets_an_idle_slot_past_the_check(calls):
    with pytest.raises(_Stop):
        _launch(_spec(), (256, 0), (256, 0))
    assert [tag for tag, *_ in calls] == ["launcher"]


@pytest.mark.parametrize("bias", [False, True], ids=["plain", "bias"])
@pytest.mark.parametrize("validate", [True, False])
def test_launcher_skips_the_check_only_when_nothing_depends_on_it(
    calls, validate, bias
):
    """``validate_varlen=False`` skips the host check (it reads the offsets, a device
    sync), except for an additive-bias launch: the bias is loaded unmasked by
    in-sequence position, so its offsets must be sound whatever the flag says."""
    spec = _spec(use_additive_bias=bias)
    with pytest.raises(_Stop):
        _launch(spec, (256, 256), (256, 256), validate_varlen=validate)
    assert [tag for tag, *_ in calls] == (["launcher"] if validate or bias else [])
    if calls:
        _, args, kwargs = calls[0]
        assert args[0] is spec
        assert (kwargs["q_tokens"], kwargs["kv_tokens"]) == (512, 512)
        assert kwargs["device"] == _DEVICE


@pytest.mark.parametrize("bias", [False, True], ids=["plain", "bias"])
@pytest.mark.parametrize("validate", [True, False])
def test_with_lse_the_check_runs_once_and_inside_validate_dense_lse(
    calls, validate, bias
):
    """With LSE the check is part of the LSE contract: it runs whatever
    ``validate_varlen`` says, and the launcher does not run it a second time."""
    spec = _spec(emit_lse=True, use_additive_bias=bias)
    with pytest.raises(_Stop):
        _launch(spec, (256, 256), (256, 256), validate_varlen=validate, lse=object())
    assert [tag for tag, *_ in calls] == ["validate_dense_lse"]
