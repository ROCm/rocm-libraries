# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Pure-CPU, provider-free tests for the SDPA backward routing (ALMIOPEN-2978):
``_grad_route`` (every reason), ``_needs_do_copy``, ``_dropout_advance``
and ``_unpack_philox``. All four are module-level pure functions taking
duck-typed tensors / plain values, so they run with fakes -- no GPU, no provider
``.so``. A few tests drive the real ``torch.autograd.Function`` (``_sdpa_autograd_fn``)
against a stub override whose ``_execute`` never touches hipDNN, using the real CPU
``torch`` the test venv provides.
"""

import pytest
import torch
import torch.utils.checkpoint as _checkpoint

from hipdnn_torch.sdpa import (
    NotApplicable,
    SdpaOverride,
    _autocast_cast_needed,
    _dropout_advance,
    _grad_route,
    _needs_do_copy,
    _reraise_checkpoint_stop,
    _sdpa_autograd_fn,
    _unpack_philox,
)


def _contig_strides(shape):
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


class _FakeTensor:
    """Minimal duck-typed tensor: just the attributes ``_grad_route`` reads."""

    def __init__(self, shape, stride=None, requires_grad=False, ptr=16, floating=True):
        self._shape = tuple(shape)
        self._stride = tuple(stride) if stride is not None else _contig_strides(shape)
        self.requires_grad = requires_grad
        self._ptr = ptr
        self._floating = floating

    @property
    def shape(self):
        return self._shape

    def stride(self, i=None):
        return self._stride if i is None else self._stride[i]

    def data_ptr(self):
        return self._ptr

    def is_floating_point(self):
        return self._floating


# A valid [B, Hq, Sq, D] query/key/value: last stride 1, seq stride >= D, aligned.
_QKV_SHAPE = (2, 8, 128, 64)


def _qkv(requires_grad=False):
    return (
        _FakeTensor(_QKV_SHAPE, requires_grad=requires_grad),
        _FakeTensor(_QKV_SHAPE),
        _FakeTensor(_QKV_SHAPE),
    )


def _route_kwargs(**overrides):
    base = dict(
        grad_enabled=False,
        deterministic=False,
        warn_only=False,
        capturing=False,
        sdp_choice=None,
        ck_preferred=False,
        compiling=False,
    )
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# _grad_route: plain / grad routes                                            #
# --------------------------------------------------------------------------- #
def test_grad_route_not_eligible_is_plain():
    q, k, v = _qkv()
    route, reason = _grad_route(q, k, v, None, 0.0, **_route_kwargs())
    assert route == "plain"
    assert reason == ""


def test_grad_route_qkv_requires_grad_routes_grad():
    q, k, v = _qkv(requires_grad=True)
    route, reason = _grad_route(q, k, v, None, 0.0, **_route_kwargs(grad_enabled=True))
    assert route == "grad"
    assert reason == ""


def test_grad_route_requires_grad_without_grad_mode_is_plain():
    # q.requires_grad can stay True under no_grad (param tensors); grad mode off
    # alone must not make a p==0 call grad-eligible.
    q, k, v = _qkv(requires_grad=True)
    route, reason = _grad_route(q, k, v, None, 0.0, **_route_kwargs(grad_enabled=False))
    assert route == "plain"
    assert reason == ""


def test_grad_route_dropout_alone_is_grad_eligible():
    q, k, v = _qkv()
    route, reason = _grad_route(
        q, k, v, None, 0.1, **_route_kwargs(grad_enabled=False, sdp_choice="EFFICIENT")
    )
    assert route == "grad"
    assert reason == ""


# --------------------------------------------------------------------------- #
# _grad_route: one case per native reason                                     #
# --------------------------------------------------------------------------- #
def test_grad_route_compile_tracing():
    q, k, v = _qkv(requires_grad=True)
    route, reason = _grad_route(
        q, k, v, None, 0.0, **_route_kwargs(grad_enabled=True, compiling=True)
    )
    assert (route, reason) == ("native", "compile: tracing")


def test_grad_route_mask_requires_grad_even_if_qkv_do_not():
    q, k, v = _qkv(requires_grad=False)
    mask = _FakeTensor((128, 128), requires_grad=True, floating=True)
    route, reason = _grad_route(q, k, v, mask, 0.0, **_route_kwargs(grad_enabled=True))
    assert (route, reason) == ("native", "grad: attn_mask requires grad")


def test_grad_route_mask_requires_grad_at_dropout_whatever_grad_mode():
    q, k, v = _qkv(requires_grad=False)
    mask = _FakeTensor((128, 128), requires_grad=True, floating=True)
    route, reason = _grad_route(
        q,
        k,
        v,
        mask,
        0.1,
        **_route_kwargs(grad_enabled=False, sdp_choice="EFFICIENT"),
    )
    assert (route, reason) == ("native", "grad: attn_mask requires grad")


def test_grad_route_mask_requires_grad_override_wins_over_mask_attribute():
    # Simulates autocast's differentiable .to() cast under no_grad: the cast
    # copy's own requires_grad reads False even though the user's original
    # mask carries it. The caller-supplied override must still force native,
    # not the (stale) attribute on the tensor actually passed in -- otherwise
    # a reentrant checkpoint's no-grad first pass would route differently
    # from its grad-enabled recompute (M9/D1).
    q, k, v = _qkv(requires_grad=False)
    mask = _FakeTensor((128, 128), requires_grad=False, floating=True)
    route, reason = _grad_route(
        q, k, v, mask, 0.0, **_route_kwargs(grad_enabled=True, mask_requires_grad=True)
    )
    assert (route, reason) == ("native", "grad: attn_mask requires grad")


def test_grad_route_mask_requires_grad_override_false_does_not_force_native():
    q, k, v = _qkv(requires_grad=True)
    mask = _FakeTensor((128, 128), requires_grad=True, floating=True)
    route, reason = _grad_route(
        q, k, v, mask, 0.0, **_route_kwargs(grad_enabled=True, mask_requires_grad=False)
    )
    assert route == "grad"
    assert reason == ""


def test_grad_route_bool_mask_never_requires_grad_check():
    # A bool mask cannot require grad (torch disallows it); is_floating_point()
    # False means the requires_grad branch is never evaluated for it either way.
    q, k, v = _qkv(requires_grad=True)
    mask = _FakeTensor((128, 128), requires_grad=False, floating=False)
    route, reason = _grad_route(q, k, v, mask, 0.0, **_route_kwargs(grad_enabled=True))
    assert route == "grad"
    assert reason == ""


def test_grad_route_deterministic_algorithms():
    q, k, v = _qkv(requires_grad=True)
    route, reason = _grad_route(
        q,
        k,
        v,
        None,
        0.0,
        **_route_kwargs(grad_enabled=True, deterministic=True, warn_only=False),
    )
    assert (route, reason) == ("native", "grad: deterministic algorithms")


def test_grad_route_deterministic_warn_only_still_routes():
    q, k, v = _qkv(requires_grad=True)
    route, reason = _grad_route(
        q,
        k,
        v,
        None,
        0.0,
        **_route_kwargs(grad_enabled=True, deterministic=True, warn_only=True),
    )
    assert route == "grad"
    assert reason == ""


def test_grad_route_operand_stride():
    # Key's last-dim stride != 1 (a transposed-last-axis view): the "rows must
    # not alias" guard (HKP-H30).
    q, _, v = _qkv(requires_grad=True)
    bad_key = _FakeTensor(_QKV_SHAPE, stride=(65536, 64, 1, 128))
    route, reason = _grad_route(
        q, bad_key, v, None, 0.0, **_route_kwargs(grad_enabled=True)
    )
    assert (route, reason) == ("native", "operand: stride")


def test_grad_route_operand_stride_seq_alias():
    # Last stride is 1, but the seq stride (< D) would make rows alias.
    q, _, v = _qkv(requires_grad=True)
    bad_key = _FakeTensor(_QKV_SHAPE, stride=(65536, 64, 32, 1))
    route, reason = _grad_route(
        q, bad_key, v, None, 0.0, **_route_kwargs(grad_enabled=True)
    )
    assert (route, reason) == ("native", "operand: stride")


def test_grad_route_operand_pointer_not_aligned():
    q, k, v = _qkv(requires_grad=True)
    v._ptr = 17
    route, reason = _grad_route(q, k, v, None, 0.0, **_route_kwargs(grad_enabled=True))
    assert (route, reason) == ("native", "operand: pointer not 16-byte aligned")


def test_grad_route_float_mask_misaligned_pointer():
    q, k, v = _qkv(requires_grad=True)
    mask = _FakeTensor((128, 128), floating=True, ptr=18)
    route, reason = _grad_route(q, k, v, mask, 0.0, **_route_kwargs(grad_enabled=True))
    assert (route, reason) == ("native", "operand: pointer not 16-byte aligned")


def test_grad_route_bool_mask_pointer_never_checked():
    # A bool mask is materialized fresh before use (always aligned); its own
    # pointer is irrelevant and must not spuriously decline the call.
    q, k, v = _qkv(requires_grad=True)
    mask = _FakeTensor((128, 128), floating=False, ptr=17)
    route, reason = _grad_route(q, k, v, mask, 0.0, **_route_kwargs(grad_enabled=True))
    assert route == "grad"
    assert reason == ""


def test_grad_route_dropout_stream_capturing():
    q, k, v = _qkv()
    route, reason = _grad_route(
        q,
        k,
        v,
        None,
        0.1,
        **_route_kwargs(capturing=True, sdp_choice="FLASH"),
    )
    assert (route, reason) == ("native", "dropout: stream capturing")


def test_grad_route_dropout_cannot_mirror_rng_advance():
    q, k, v = _qkv()
    route, reason = _grad_route(q, k, v, None, 0.1, **_route_kwargs(sdp_choice=None))
    assert (route, reason) == ("native", "dropout: cannot mirror native RNG advance")


def test_grad_route_dropout_ck_flash_preferred():
    q, k, v = _qkv()
    route, reason = _grad_route(
        q,
        k,
        v,
        None,
        0.1,
        **_route_kwargs(sdp_choice="FLASH", ck_preferred=True),
    )
    assert (route, reason) == ("native", "dropout: CK flash preferred")


def test_grad_route_dropout_native_math_backend():
    q, k, v = _qkv()
    route, reason = _grad_route(q, k, v, None, 0.1, **_route_kwargs(sdp_choice="MATH"))
    assert (route, reason) == ("native", "dropout: native math backend")


@pytest.mark.parametrize("choice", ["FLASH", "EFFICIENT", "CUDNN"])
def test_grad_route_dropout_mirrorable_backend_routes_grad(choice):
    q, k, v = _qkv()
    route, reason = _grad_route(q, k, v, None, 0.1, **_route_kwargs(sdp_choice=choice))
    assert route == "grad"
    assert reason == ""


# --------------------------------------------------------------------------- #
# _needs_do_copy                                                              #
# --------------------------------------------------------------------------- #
def test_needs_do_copy_identical_layout_no_copy():
    o_stride = (65536, 8192, 64, 1)
    assert _needs_do_copy((2, 8, 128, 64), o_stride, o_stride, 16) is False


def test_needs_do_copy_size1_axis_difference_ignored():
    # A size-1 batch axis with a different stride does not force a copy: only
    # extent > 1 axes are compared.
    do_stride = (999, 8192, 64, 1)
    o_stride = (65536, 8192, 64, 1)
    assert _needs_do_copy((1, 8, 128, 64), do_stride, o_stride, 16) is False


def test_needs_do_copy_zero_stride_self_overlap():
    do_stride = (65536, 0, 64, 1)  # broadcast heads: stride 0 on extent > 1
    o_stride = (65536, 8192, 64, 1)
    assert _needs_do_copy((2, 8, 128, 64), do_stride, o_stride, 16) is True


def test_needs_do_copy_misaligned_pointer():
    o_stride = (65536, 8192, 64, 1)
    assert _needs_do_copy((2, 8, 128, 64), o_stride, o_stride, 17) is True


def test_needs_do_copy_differing_stride():
    do_stride = (65536, 64, 512, 1)  # BSHD view instead of O's BHSD
    o_stride = (65536, 8192, 64, 1)
    assert _needs_do_copy((2, 8, 128, 64), do_stride, o_stride, 16) is True


# --------------------------------------------------------------------------- #
# _dropout_advance                                                            #
# --------------------------------------------------------------------------- #
def test_dropout_advance_flash_uses_hq_times_32():
    # b=2, Hq=8: FLASH advances by 32 per (b, Hq), whatever Hkv is in a GQA call
    # (not an argument here at all -- the caller is responsible for passing Hq).
    assert _dropout_advance("FLASH", 2, 8, 128, 128) == 2 * 8 * 32


@pytest.mark.parametrize("choice", ["EFFICIENT", "CUDNN"])
def test_dropout_advance_efficient_cudnn_uses_full_product(choice):
    assert _dropout_advance(choice, 2, 8, 128, 128) == 2 * 8 * 128 * 128


def test_dropout_advance_math_is_none():
    assert _dropout_advance("MATH", 2, 8, 128, 128) is None


def test_dropout_advance_unknown_choice_is_none():
    assert _dropout_advance(None, 2, 8, 128, 128) is None


def test_dropout_advance_rounds_up_to_multiple_of_4():
    # b*hq*sq*skv = 1*1*3*1 = 3 -> rounds up to 4.
    assert _dropout_advance("EFFICIENT", 1, 1, 3, 1) == 4
    # Already a multiple of 4: unchanged.
    assert _dropout_advance("EFFICIENT", 1, 1, 4, 1) == 4


# --------------------------------------------------------------------------- #
# _unpack_philox                                                              #
# --------------------------------------------------------------------------- #
class _FakeInt64Tensor:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


def test_unpack_philox_3tuple():
    state = (_FakeInt64Tensor(12345), _FakeInt64Tensor(100), _FakeInt64Tensor(0))
    assert _unpack_philox(state) == (12345, 100)


def test_unpack_philox_intragraph_added_to_offset():
    state = (_FakeInt64Tensor(12345), _FakeInt64Tensor(100), _FakeInt64Tensor(7))
    assert _unpack_philox(state) == (12345, 107)


def test_unpack_philox_signed_seed_preserved():
    # A seed whose bit pattern is negative as a signed int64 must come back
    # exactly as given, not reinterpreted as unsigned.
    negative_seed = -1234567890123456789
    state = (_FakeInt64Tensor(negative_seed), _FakeInt64Tensor(0), _FakeInt64Tensor(0))
    seed, offset = _unpack_philox(state)
    assert seed == negative_seed
    assert offset == 0


# --------------------------------------------------------------------------- #
# _draw_dropout_state / _dropout_tensors: draw split from allocation           #
# --------------------------------------------------------------------------- #
class _FakePhiloxGenerator:
    """Stands in for a CUDA generator whose ``philox_state`` works."""

    def __init__(self, seed, offset, intragraph=0):
        self._seed = seed
        self._offset = offset
        self._intragraph = intragraph
        self.philox_calls = []

    def philox_state(self, inc):
        self.philox_calls.append(inc)
        return (
            _FakeInt64Tensor(self._seed),
            _FakeInt64Tensor(self._offset),
            _FakeInt64Tensor(self._intragraph),
        )


class _FakeFallbackGenerator:
    """Stands in for a generator with no ``philox_state`` (older torch): the
    ``initial_seed()`` / ``get_offset()`` / ``set_offset()`` path."""

    def __init__(self, seed, offset):
        self._seed = seed
        self._offset = offset

    def initial_seed(self):
        return self._seed

    def get_offset(self):
        return self._offset

    def set_offset(self, value):
        self._offset = value


class _RaisingTorch:
    """A ``torch`` stand-in whose allocation (``full``) always fails, so
    ``_dropout_tensors`` can be exercised without a real device."""

    int64 = "int64"

    @staticmethod
    def full(shape, value, dtype=None, device=None):
        raise RuntimeError("out of memory")


def test_draw_dropout_state_philox_path_returns_pre_increment_offset():
    gen = _FakePhiloxGenerator(seed=42, offset=100)
    ov = SdpaOverride()
    seed, offset, offset_before = ov._draw_dropout_state(gen, "EFFICIENT", 1, 1, 4, 4)
    assert seed == 42
    assert offset == 100
    assert offset_before == 100  # philox_state's own pre-increment value
    assert gen.philox_calls == [_dropout_advance("EFFICIENT", 1, 1, 4, 4)]


def test_draw_dropout_state_fallback_path_advances_generator():
    gen = _FakeFallbackGenerator(seed=7, offset=100)
    ov = SdpaOverride()
    inc = _dropout_advance("EFFICIENT", 1, 1, 4, 4)
    seed, offset, offset_before = ov._draw_dropout_state(gen, "EFFICIENT", 1, 1, 4, 4)
    assert seed == 7
    assert offset == 100
    assert offset_before == 100
    assert gen._offset == 100 + inc  # committed the advance


def test_draw_dropout_state_fallback_reinterprets_large_seed_as_signed():
    # initial_seed() can return the seed as an unsigned 64-bit value; the
    # seed tensor set_dropout takes is signed int64.
    huge_unsigned_seed = (1 << 63) + 12345
    gen = _FakeFallbackGenerator(seed=huge_unsigned_seed, offset=0)
    ov = SdpaOverride()
    seed, offset, offset_before = ov._draw_dropout_state(gen, "EFFICIENT", 1, 1, 4, 4)
    assert seed == huge_unsigned_seed - (1 << 64)
    assert -(1 << 63) <= seed < (1 << 63)


def test_draw_dropout_state_philox_path_seed_already_signed_unchanged():
    # The philox_state path never reinterprets: a value already out of signed
    # range there would be a torch bug, not something to "fix" here.
    gen = _FakePhiloxGenerator(seed=-5, offset=0)
    ov = SdpaOverride()
    seed, _offset, _offset_before = ov._draw_dropout_state(gen, "EFFICIENT", 1, 1, 4, 4)
    assert seed == -5


# --------------------------------------------------------------------------- #
# _reraise_checkpoint_stop: let torch.utils.checkpoint's early-stop signal     #
# through any broad except Exception on the SDPA path                        #
# --------------------------------------------------------------------------- #
class _FakeStopRecomputationError(Exception):
    pass


def _fake_torch_with_checkpoint(stop_cls):
    class _FakeCheckpointModule:
        _StopRecomputationError = stop_cls

    class _FakeUtils:
        checkpoint = _FakeCheckpointModule

    class _FakeTorch:
        utils = _FakeUtils

    return _FakeTorch


def test_reraise_checkpoint_stop_lets_the_signal_through():
    fake_torch = _fake_torch_with_checkpoint(_FakeStopRecomputationError)
    with pytest.raises(_FakeStopRecomputationError):
        try:
            raise _FakeStopRecomputationError("early stop")
        except Exception as caught:  # noqa: BLE001 -- mirrors the real call sites
            _reraise_checkpoint_stop(fake_torch, caught)


def test_reraise_checkpoint_stop_ignores_other_exceptions():
    fake_torch = _fake_torch_with_checkpoint(_FakeStopRecomputationError)
    ex = RuntimeError("an ordinary hipDNN failure")
    _reraise_checkpoint_stop(fake_torch, ex)  # must not raise


def test_reraise_checkpoint_stop_tolerates_missing_checkpoint_module():
    # torch.utils.checkpoint may never have been imported (no checkpoint used):
    # the guard must not crash looking for a class that was never defined.
    class _FakeUtilsNoCheckpoint:
        pass

    class _FakeTorchNoCheckpoint:
        utils = _FakeUtilsNoCheckpoint

    _reraise_checkpoint_stop(
        _FakeTorchNoCheckpoint, RuntimeError("boom")
    )  # must not raise


# --------------------------------------------------------------------------- #
# Function.forward: a successful execute is counted even when a             #
# non-reentrant torch.utils.checkpoint recompute stops right after it        #
# --------------------------------------------------------------------------- #
class _FakeOverride:
    """Stands in for SdpaOverride: records note_aot/note_native calls; its
    ``_execute`` "succeeds" without touching hipDNN."""

    def __init__(self):
        self.aot_calls = []
        self.native_calls = []

    def _execute(self, entry, variant_pack, device):
        pass

    def note_aot(self, key, engine, **extras):
        self.aot_calls.append((key, engine, dict(extras)))

    def note_native(self, key, reason, level=None):
        self.native_calls.append((key, reason))


class _FailingExecuteOverride(_FakeOverride):
    def _execute(self, entry, variant_pack, device):
        raise NotApplicable("execute: boom")


def _fake_fn_payload(ov):
    return {
        "ov": ov,
        "entry_fwd": {"engine": "fake_fwd"},
        "entry_bwd": {"engine": "fake_bwd"},
        "alloc_dims": [2, 2],
        "inverse": [0, 1],
        "o_dtype": torch.float32,
        "o_view_stride": (2, 1),
        "stats_dims": (2, 1, 1, 1),
        "dropout_p": 0.0,
        "is_causal": False,
        "scale": 1.0,
        "enable_gqa": False,
        "real": None,
        "grad_key": "grad x",
        "bwd_key": "bwd x",
        "mask_extras": {},
        "dq_layout": ([2, 2], [0, 1]),
        "dk_layout": ([2, 2], [0, 1]),
        "dv_layout": ([2, 2], [0, 1]),
    }


def test_function_forward_counts_execute_before_checkpoint_stop_signal():
    # A non-reentrant checkpoint recompute can stop right after this op's
    # saved tensors satisfy every weak holder: the saved-tensor pack hook
    # raises _StopRecomputationError from inside ctx.save_for_backward, after
    # the (stubbed) hipDNN forward already "executed". The execution must
    # still be counted -- not lost to the signal propagating out uncounted.
    ov = _FakeOverride()
    fn_cls = _sdpa_autograd_fn(torch)
    payload = _fake_fn_payload(ov)
    q = torch.randn(2, 2, requires_grad=True)
    k = torch.randn(2, 2, requires_grad=True)
    v = torch.randn(2, 2, requires_grad=True)

    def pack_hook(t):
        raise _checkpoint._StopRecomputationError("stop")

    with pytest.raises(_checkpoint._StopRecomputationError):
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, lambda t: t):
            fn_cls.apply(q, k, v, None, None, None, payload)

    assert ov.aot_calls == [("grad x", "fake_fwd", {})]
    assert ov.native_calls == []


def test_function_forward_execute_failure_counts_nothing_and_propagates():
    # Nothing ran, so nothing is counted here (note_aot is unreachable); the
    # caller (_call_grad) is the one that turns this into a native fallback.
    ov = _FailingExecuteOverride()
    fn_cls = _sdpa_autograd_fn(torch)
    payload = _fake_fn_payload(ov)
    q = torch.randn(2, 2, requires_grad=True)
    k = torch.randn(2, 2, requires_grad=True)
    v = torch.randn(2, 2, requires_grad=True)

    with pytest.raises(NotApplicable):
        fn_cls.apply(q, k, v, None, None, None, payload)

    assert ov.aot_calls == []
    assert ov.native_calls == []


# --------------------------------------------------------------------------- #
# _autocast_cast_needed: per-tensor cast-selection decision                   #
# --------------------------------------------------------------------------- #
def test_autocast_cast_needed_cuda_floating_not_target_needs_cast():
    assert (
        _autocast_cast_needed(
            is_cuda=True, is_floating=True, is_fp64=False, is_target_dtype=False
        )
        is True
    )


def test_autocast_cast_needed_already_target_untouched():
    assert (
        _autocast_cast_needed(
            is_cuda=True, is_floating=True, is_fp64=False, is_target_dtype=True
        )
        is False
    )


def test_autocast_cast_needed_fp64_untouched():
    assert (
        _autocast_cast_needed(
            is_cuda=True, is_floating=True, is_fp64=True, is_target_dtype=False
        )
        is False
    )


def test_autocast_cast_needed_non_floating_untouched():
    assert (
        _autocast_cast_needed(
            is_cuda=True, is_floating=False, is_fp64=False, is_target_dtype=False
        )
        is False
    )


def test_autocast_cast_needed_non_cuda_untouched():
    assert (
        _autocast_cast_needed(
            is_cuda=False, is_floating=True, is_fp64=False, is_target_dtype=False
        )
        is False
    )


# --------------------------------------------------------------------------- #
# SdpaOverride._autocast_cast: casting + version-tolerant lookups             #
# --------------------------------------------------------------------------- #
class _FakeDtype:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"<dtype {self.name}>"


_FP32 = _FakeDtype("fp32")
_BF16 = _FakeDtype("bf16")
_FP64 = _FakeDtype("fp64")
_BOOL = _FakeDtype("bool")


class _FakeAutocastTensor:
    """Minimal duck-typed tensor for ``_autocast_cast``: ``.dtype``,
    ``.is_cuda``, ``.is_floating_point()``, and a ``.to(dtype)`` that returns a
    NEW instance, so a cast is observable by identity."""

    def __init__(self, dtype, is_cuda=True, floating=True):
        self.dtype = dtype
        self.is_cuda = is_cuda
        self._floating = floating

    def is_floating_point(self):
        return self._floating

    def to(self, dtype):
        return _FakeAutocastTensor(dtype, is_cuda=self.is_cuda, floating=self._floating)


class _FakeTorchAutocast:
    """Modern (device-generic) autocast API stand-in."""

    float64 = _FP64

    def __init__(self, enabled=True, target=_BF16):
        self._enabled = enabled
        self._target = target

    def is_autocast_enabled(self, device_type=None):
        return self._enabled

    def get_autocast_dtype(self, device_type=None):
        return self._target


class _FakeTorchOldAutocastAPI:
    """Pre-device-generic autocast API stand-in: no ``device_type`` argument,
    and ``get_autocast_dtype`` does not exist at all (only the deprecated
    ``get_autocast_gpu_dtype``)."""

    float64 = _FP64

    def __init__(self, enabled=True, target=_BF16):
        self._enabled = enabled
        self._target = target

    def is_autocast_enabled(self):
        return self._enabled

    def get_autocast_gpu_dtype(self):
        return self._target


def test_autocast_cast_disabled_leaves_everything_untouched():
    ov = SdpaOverride()
    torch_stub = _FakeTorchAutocast(enabled=False)
    q = _FakeAutocastTensor(_FP32)
    k = _FakeAutocastTensor(_FP32)
    v = _FakeAutocastTensor(_BF16)
    mask = _FakeAutocastTensor(_FP32)
    out_q, out_k, out_v, out_mask = ov._autocast_cast(torch_stub, q, k, v, mask)
    assert out_q is q
    assert out_k is k
    assert out_v is v
    assert out_mask is mask


def test_autocast_cast_mixed_dtype_casts_only_what_needs_it():
    # fp32 q/k + bf16 v under bf16 autocast: only q/k cast (the mixed-dtype
    # case -- a cast decided from q alone would wrongly also touch v).
    ov = SdpaOverride()
    torch_stub = _FakeTorchAutocast(enabled=True, target=_BF16)
    q = _FakeAutocastTensor(_FP32)
    k = _FakeAutocastTensor(_FP32)
    v = _FakeAutocastTensor(_BF16)
    out_q, out_k, out_v, out_mask = ov._autocast_cast(torch_stub, q, k, v, None)
    assert out_q is not q and out_q.dtype is _BF16
    assert out_k is not k and out_k.dtype is _BF16
    assert out_v is v  # already the target dtype: untouched
    assert out_mask is None


def test_autocast_cast_fp64_untouched():
    ov = SdpaOverride()
    torch_stub = _FakeTorchAutocast(enabled=True, target=_BF16)
    q = _FakeAutocastTensor(_FP64)
    out_q, out_k, out_v, out_mask = ov._autocast_cast(torch_stub, q, None, None, None)
    assert out_q is q
    assert out_k is None
    assert out_v is None
    assert out_mask is None


def test_autocast_cast_bool_mask_untouched():
    ov = SdpaOverride()
    torch_stub = _FakeTorchAutocast(enabled=True, target=_BF16)
    q = _FakeAutocastTensor(_FP32)
    mask = _FakeAutocastTensor(_BOOL, floating=False)
    out_q, _out_k, _out_v, out_mask = ov._autocast_cast(torch_stub, q, None, None, mask)
    assert out_q is not q
    assert out_mask is mask


def test_autocast_cast_non_cuda_untouched():
    ov = SdpaOverride()
    torch_stub = _FakeTorchAutocast(enabled=True, target=_BF16)
    q = _FakeAutocastTensor(_FP32, is_cuda=False)
    out_q, _out_k, _out_v, _out_mask = ov._autocast_cast(
        torch_stub, q, None, None, None
    )
    assert out_q is q


def test_autocast_cast_version_tolerant_fallback_old_api():
    # Older torch: is_autocast_enabled()/get_autocast_gpu_dtype() take no
    # device argument at all; get_autocast_dtype does not exist.
    ov = SdpaOverride()
    torch_stub = _FakeTorchOldAutocastAPI(enabled=True, target=_BF16)
    q = _FakeAutocastTensor(_FP32)
    out_q, _out_k, _out_v, _out_mask = ov._autocast_cast(
        torch_stub, q, None, None, None
    )
    assert out_q.dtype is _BF16


def test_autocast_cast_missing_api_disables_entirely():
    class _NoAutocastTorch:
        float64 = _FP64

    ov = SdpaOverride()
    q = _FakeAutocastTensor(_FP32)
    out_q, _out_k, _out_v, _out_mask = ov._autocast_cast(
        _NoAutocastTorch(), q, None, None, None
    )
    assert out_q is q
