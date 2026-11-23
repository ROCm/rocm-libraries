# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-side input-validation guard for the GDN decode kernel, without a GPU.

``validate_inputs`` (which ``prepare`` runs before it uploads anything) rejects
inputs that would make the kernel read or write outside the state pool, or
reinterpret bytes. The decode kernel bounds-checks nothing on device beyond the
``-1`` skip sentinel, so this host guard *is* the memory-safety contract. The
checks are pure host logic on numpy arrays, so they run on a CPU box -- which
is exactly where a "this guard must not be silently dropped" regression test
belongs, rather than behind the on-device ``gpu`` gate.
"""

from __future__ import annotations

import dataclasses as dc
import re

import numpy as np
import pytest
from rocke.numeric import storage
from rocke.tests.lanes import DROP_REVIEW, skip_until_reason

from builders.gfx950.gdn.gdn_decode import input_dtypes, make_inputs, validate_inputs
from kernels.gfx950.gdn_decode import GdnDecodeSpec, gdn_decode_pointer_alignment

FAILED = "gdn_decode input failed - "

# The driver took device tensors before; some of its refusals were of a tensor on
# another device or a base pointer off the kernel's alignment. Inputs are host
# arrays now and prepare() allocates every device buffer, so neither state can be
# built. Those tests stay listed as skips until their removal is decided.
_RULED_OUT = pytest.mark.skip(
    reason=skip_until_reason(
        DROP_REVIEW,
        reason="inputs are host numpy arrays and prepare() allocates every device "
        "buffer, so this state cannot be built",
    )
)


def _refused(match: str):
    return pytest.raises(ValueError, match=re.escape(FAILED + match))


def test_out_of_range_index_is_rejected():
    """An index past the pool depth is an OOB access; the guard must refuse it.

    ``-1`` stays legal (skip); any other out-of-pool value is rejected before a
    launch can touch it, for both the read and the write index.
    """
    spec = GdnDecodeSpec()
    batch = 8
    pool_depth = make_inputs(spec, batch)["state"].shape[0]

    for name, bad, span in (
        ("read_indices", pool_depth, f"[1, {pool_depth}]"),  # the first OOB slot
        ("write_indices", pool_depth + 5, f"[9, {pool_depth + 5}]"),
        ("read_indices", -2, "[-2, 7]"),  # below the -1 skip sentinel
        ("write_indices", -2, "[-2, 15]"),
    ):
        inp = make_inputs(spec, batch)
        inp[name][0] = bad
        with _refused(f"{name} spans {span}, expected -1 (skip) or a slot in"):
            validate_inputs(spec, inp, batch)


def test_duplicate_active_write_index_is_rejected():
    """Two live sequences cannot race to update one recurrent-state page."""
    spec = GdnDecodeSpec()
    batch = 8
    inp = make_inputs(spec, batch)
    inp["write_indices"][1] = inp["write_indices"][0]
    with _refused("write_indices repeat [8], expected unique slots across active"):
        validate_inputs(spec, inp, batch)


def test_inactive_mismatched_lane_does_not_reserve_a_write_index():
    """A lane with either negative index is inactive and cannot claim a page."""
    spec = GdnDecodeSpec()
    batch = 8
    inp = make_inputs(spec, batch)
    inp["read_indices"][1] = -1
    inp["write_indices"][1] = inp["write_indices"][0]
    validate_inputs(spec, inp, batch)  # must not raise: lane 1 is inactive


def test_the_skip_sentinel_is_accepted():
    """``-1`` marks an idle continuous-batching slot and must pass the guard."""
    spec = GdnDecodeSpec()
    batch = 8
    inp = make_inputs(spec, batch)
    inp["read_indices"][1::2] = -1
    inp["write_indices"][1::2] = -1
    validate_inputs(spec, inp, batch)  # must not raise


def test_wrong_state_head_dims_are_rejected():
    """A state pool whose head dims disagree with the spec is a shape bug."""
    spec = GdnDecodeSpec()
    batch = 8
    inp = make_inputs(spec, batch)
    # Drop a slice of the K dim so the pool no longer matches the spec.
    inp["state"] = np.ascontiguousarray(inp["state"][..., :-8])
    with _refused("state.shape=(17, 32, 128, 120), expected [pool, HV, DV, DK]"):
        validate_inputs(spec, inp, batch)


def test_wrong_state_dtype_is_rejected():
    """A pool whose element type disagrees with the spec must be refused here.

    ``bf16`` and ``f16`` are both 16 bits, so a mismatched pool has the right
    shape *and* the right byte size: every address the kernel computes is
    identical and nothing faults. The kernel is compiled with a fixed pointer
    element type and gets no dtype tag at runtime, so it simply decodes the
    bits under the wrong rule -- and decode writes that value back into the
    pool, compounding it over the whole generation. Nothing on device can
    catch it, which is why the host guard must.
    """
    spec = GdnDecodeSpec()
    assert spec.state_dtype == "bf16", "test assumes the default spec state dtype"
    batch = 8
    inp = make_inputs(spec, batch)
    inp["state"] = storage.encode(storage.decode(inp["state"], "bf16"), "fp16")
    # Same shape, same byte count -- only the element type differs.
    with _refused("state.dtype=float16, expected uint16 (bf16): the kernel"):
        validate_inputs(spec, inp, batch)


def test_mis_shaped_input_array_is_rejected():
    """Every address is computed from SPEC dims, and the kernel emits no buffer
    descriptor -- there is no `num_records` to clamp an over-reach -- so an
    array whose real shape is smaller than the spec says is an out-of-bounds
    read with nothing between it and other allocations.
    """
    spec = GdnDecodeSpec()
    batch = 8
    for name in ("query", "key", "value", "a", "b", "dt_bias", "A_log"):
        inp = make_inputs(spec, batch)
        want = inp[name].shape
        inp[name] = np.ascontiguousarray(inp[name][..., :-1])
        with _refused(f"{name}.shape={inp[name].shape}, expected {want}"):
            validate_inputs(spec, inp, batch)


def test_wrong_input_dtype_is_rejected():
    """The kernel is compiled against a fixed element type; handing it other
    bytes reinterprets them rather than converting."""
    spec = GdnDecodeSpec()
    batch = 8
    inp = make_inputs(spec, batch)
    inp["query"] = storage.decode(inp["query"], "bf16")
    with _refused("query.dtype=float32, expected uint16 (bf16)"):
        validate_inputs(spec, inp, batch)


def test_an_input_that_is_not_a_host_array_is_refused():
    """Inputs are host numpy arrays: ``prepare`` owns the upload, so it can
    guarantee every buffer's size and alignment. Anything else is a type
    error."""
    spec = GdnDecodeSpec()
    batch = 2
    inp = make_inputs(spec, batch)
    inp["query"] = inp["query"].tolist()
    with pytest.raises(TypeError, match=re.escape(FAILED + "query=list, expected")):
        validate_inputs(spec, inp, batch)


def _transposed(x):
    """``x``'s shape with the last two axes' memory order swapped."""
    return np.ascontiguousarray(x.swapaxes(-1, -2)).swapaxes(-1, -2)


def test_non_contiguous_input_is_rejected():
    """The kernel's offsets assume row-major. A transposed view has the right
    shape and the wrong memory order, so the kernel would read the right INDEX
    out of the wrong ADDRESS -- silently."""
    spec = GdnDecodeSpec()
    batch = 8
    inp = make_inputs(spec, batch)
    inp["value"] = _transposed(inp["value"])
    assert not inp["value"].flags.c_contiguous
    with _refused("value.strides="):
        validate_inputs(spec, inp, batch)


def _kda(**kw):
    return dc.replace(GdnDecodeSpec(), gate_kind="kda", **kw)


@pytest.mark.parametrize(
    "key,mutate,match",
    [
        ("a", lambda x: np.ascontiguousarray(x[..., :1]), "a.shape=(2, 1, 32, 1)"),
        ("a", _transposed, "a.strides="),
        ("dt_bias", lambda x: np.ascontiguousarray(x[..., :1]), "dt_bias.shape="),
        (
            "dt_bias",
            lambda x: storage.encode(x, "bf16"),
            "dt_bias.dtype=uint16, expected float32 (fp32)",
        ),
        ("dt_bias", _transposed, "dt_bias.strides="),
    ],
)
def test_kda_gate_input_contract_is_rejected_before_launch(key, mutate, match):
    """KDA widens the gate buffers; legacy/malformed allocations must not launch.

    The emitter vector-loads ``a`` as ``[B,1,HV,DK]`` and ``dt_bias`` as f32
    ``[HV,DK]`` using spec-derived offsets. A legacy GDN-shaped or strided
    allocation is smaller/differently laid out than that compiled range.
    """
    spec = _kda()
    batch = 2
    inp = make_inputs(spec, batch)
    inp[key] = mutate(inp[key])
    with _refused(match):
        validate_inputs(spec, inp, batch)


@_RULED_OUT
@pytest.mark.parametrize("key", ["a", "dt_bias"])
def test_kda_gate_inputs_must_match_query_device(key):
    """A gate buffer on another device than the query (see ``_RULED_OUT``)."""


@_RULED_OUT
def test_unfused_query_must_be_aligned():
    """A query base pointer off the 16 B alignment (see ``_RULED_OUT``)."""


@pytest.mark.skip(
    reason=skip_until_reason(
        DROP_REVIEW,
        reason="prepare() no longer takes validate_indices: the index range is "
        "checked on host arrays, with no device sync to avoid",
    )
)
def test_validate_indices_flag_skips_the_range_check():
    """The flag that let a hot loop skip the index-range check."""


def test_gdn_gate_input_contract_stays_scalar():
    """The KDA checks must not reject the existing scalar GDN ABI."""
    spec = GdnDecodeSpec()
    inp = make_inputs(spec, batch=2)

    assert inp["a"].shape == (2, 1, spec.num_v_heads)
    assert inp["dt_bias"].shape == (spec.num_v_heads,)
    validate_inputs(spec, inp, batch=2)


def test_valid_kda_gate_inputs_survive_generic_validation():
    """KDA's `[B,1,HV,DK]` ``a`` and `[HV,DK]` f32 ``dt_bias`` must not be
    rechecked against GDN's scalar shapes, which would reject every normal KDA
    launch before the kernel ran."""
    spec = _kda()
    validate_inputs(spec, make_inputs(spec, batch=2), batch=2)


# ---- fused conv1d / gated-RMSNorm inputs ----------------------------------
#
# Every one of these feeds device address math (a row base, a pool slot, a
# vector load's alignment) or the norm's arithmetic, and nothing on the device
# checks it. Each case breaks exactly one rule and names the message of the
# guard that must fire.

_BATCH = 3


def _fused_spec(**kw):
    base = dict(
        num_k_heads=2,
        num_v_heads=2,
        gate_kind="kda",
        num_warps=4,
        warp_threads_k=16,
        blocks_per_v_dim=1,
        fuse_conv=True,
        fuse_out_norm=True,
    )
    base.update(kw)
    return dc.replace(GdnDecodeSpec(), **base)


def test_every_array_is_stored_in_its_declared_element_type():
    """``make_inputs`` and the guard read one map of element types, so a
    fixture that drifted from it would fail its own validation."""
    for spec in (GdnDecodeSpec(), _kda(), _fused_spec(state_dtype="f32")):
        inp = make_inputs(spec, 2)
        for name, element in input_dtypes(spec).items():
            assert inp[name].dtype == storage.host_dtype(element), name


def test_driver_allocations_meet_every_kernel_alignment():
    """``prepare`` allocates every device buffer itself, and ``hipMalloc``
    aligns an allocation to at least 256 B, so a base alignment the kernel
    assumes is met when it divides 256."""
    for spec in (GdnDecodeSpec(), _kda(), _fused_spec()):
        for name, align in gdn_decode_pointer_alignment(spec).items():
            assert 256 % align == 0, (name, align)


def _rows(array, stride):
    """``array``'s rows widened to ``stride`` elements (zero tail)."""
    wide = np.zeros((array.shape[0], stride), dtype=array.dtype)
    wide[:, : array.shape[1]] = array
    return wide


def _set(**kv):
    def mutate(inp):
        inp.update({k: v(inp) if callable(v) else v for k, v in kv.items()})

    return mutate


def _cut(name, index):
    """``inp[name][index]`` as a contiguous array."""
    return _set(**{name: lambda i: np.ascontiguousarray(i[name][index])})


def _as_fp16(name):
    """``inp[name]``'s bf16 values stored as fp16: same bytes per element."""
    return _set(
        **{name: lambda i: storage.encode(storage.decode(i[name], "bf16"), "fp16")}
    )


def _strided_view(name):
    """``inp[name]`` as a view whose channels are two elements apart."""
    return _set(**{name: lambda i: np.repeat(i[name], 2, axis=1)[:, ::2]})


_NORM_EPS_REFUSED = (
    ("zero", 0.0),
    ("negative", -1e-6),
    ("nan", float("nan")),
    ("bool", True),
    ("numpy_scalar", np.float32(1e-6)),
    # Positive in Python but not as the f32 the kernel receives: 1e-50 packs to
    # 0.0, 1e-40 is an f32 denormal, 1e39 overflows f32.
    ("f32_zero", 1e-50),
    ("f32_denormal", 1e-40),
    ("f32_overflow", 1e39),
)

_FUSED_BAD_INPUTS = [
    # mixed_qkv: [B, >= conv_dim] contiguous rows of the I/O dtype
    (
        "mixed_qkv_short",
        _cut("mixed_qkv", np.s_[:, :-8]),
        ValueError,
        "mixed_qkv.shape=(3, 760), expected [batch=3, >= 768]",
    ),
    (
        "mixed_qkv_3d",
        _set(mixed_qkv=lambda i: i["mixed_qkv"][:, None]),
        ValueError,
        "mixed_qkv.shape=(3, 1, 768), expected [batch=3, >= 768]",
    ),
    (
        "mixed_qkv_batch",
        _cut("mixed_qkv", np.s_[:-1]),
        ValueError,
        "mixed_qkv.shape=(2, 768), expected [batch=3, >= 768]",
    ),
    (
        "mixed_qkv_dtype",
        _as_fp16("mixed_qkv"),
        ValueError,
        "mixed_qkv.dtype=float16, expected uint16 (bf16)",
    ),
    (
        "mixed_qkv_strided_view",
        _strided_view("mixed_qkv"),
        ValueError,
        "mixed_qkv.strides=(3072, 4), expected a contiguous",
    ),
    # qkv_stride: an int equal to the row length, rows on a 16 B boundary
    (
        "qkv_stride_mismatch",
        _set(qkv_stride=lambda i: i["qkv_stride"] + 8),
        ValueError,
        "qkv_stride=776, expected mixed_qkv's row length 768",
    ),
    (
        "qkv_stride_float",
        _set(qkv_stride=lambda i: float(i["qkv_stride"])),
        TypeError,
        "qkv_stride=768.0, expected an int",
    ),
    (
        "qkv_stride_misaligned",
        _set(mixed_qkv=lambda i: _rows(i["mixed_qkv"], 772), qkv_stride=772),
        ValueError,
        "qkv_stride=772, expected a row pitch that is a multiple of 16 B",
    ),
    # conv_state: [pool, conv_dim, taps] with the SAME pool depth as state
    (
        "conv_state_pool_depth",
        _cut("conv_state", np.s_[:-1]),
        ValueError,
        "conv_state.shape=(6, 768, 3), expected (7, 768, 3)",
    ),
    (
        "conv_state_channels",
        _cut("conv_state", np.s_[:, :-1]),
        ValueError,
        "conv_state.shape=(7, 767, 3), expected (7, 768, 3)",
    ),
    (
        "conv_state_dtype",
        _as_fp16("conv_state"),
        ValueError,
        "conv_state.dtype=float16, expected uint16 (bf16)",
    ),
    (
        "conv_state_strided",
        _set(conv_state=lambda i: _transposed(i["conv_state"])),
        ValueError,
        "conv_state.strides=",
    ),
    # conv_weight: [conv_dim, width] f32
    (
        "conv_weight_shape",
        _cut("conv_weight", np.s_[:, :-1]),
        ValueError,
        "conv_weight.shape=(768, 3), expected (768, 4)",
    ),
    (
        "conv_weight_dtype",
        _set(conv_weight=lambda i: storage.encode(i["conv_weight"], "bf16")),
        ValueError,
        "conv_weight.dtype=uint16, expected float32 (fp32)",
    ),
    (
        "conv_weight_strided",
        _set(conv_weight=lambda i: _transposed(i["conv_weight"])),
        ValueError,
        "conv_weight.strides=",
    ),
    # out_gate: [B, >= Hv*Dv] contiguous rows; og_stride matches
    (
        "out_gate_short",
        _cut("out_gate", np.s_[:, :-1]),
        ValueError,
        "out_gate.shape=(3, 255), expected [batch=3, >= 256]",
    ),
    (
        "out_gate_dtype",
        _as_fp16("out_gate"),
        ValueError,
        "out_gate.dtype=float16, expected uint16 (bf16)",
    ),
    (
        "out_gate_strided_view",
        _strided_view("out_gate"),
        ValueError,
        "out_gate.strides=(1024, 4), expected a contiguous",
    ),
    (
        "og_stride_mismatch",
        _set(og_stride=lambda i: i["og_stride"] + 1),
        ValueError,
        "og_stride=257, expected out_gate's row length 256",
    ),
    ("og_stride_float", _set(og_stride=256.0), TypeError, "og_stride=256.0"),
    # norm_weight: [Dv] f32; norm_eps: finite, positive, and still a normal f32
    # after the f32 kernarg pack
    (
        "norm_weight_shape",
        _cut("norm_weight", np.s_[:-1]),
        ValueError,
        "norm_weight.shape=(127,), expected (128,)",
    ),
    (
        "norm_weight_dtype",
        _set(norm_weight=lambda i: i["norm_weight"].astype(np.float16)),
        ValueError,
        "norm_weight.dtype=float16, expected float32 (fp32)",
    ),
    *[
        (f"norm_eps_{label}", _set(norm_eps=eps), ValueError, f"norm_eps={eps!r}")
        for label, eps in _NORM_EPS_REFUSED
    ],
]


_FUSED_RULED_OUT = [
    *(f"{name}_device" for name in ("conv_state", "conv_weight", "out_gate")),
    *(f"{name}_device" for name in ("norm_weight", "state")),
    *(f"{name}_misaligned" for name in ("mixed_qkv", "conv_state", "conv_weight")),
    "dt_bias_misaligned",
]


@pytest.mark.parametrize(
    "mutate,error,match",
    [pytest.param(*c[1:], id=c[0]) for c in _FUSED_BAD_INPUTS]
    + [
        pytest.param(None, None, None, id=name, marks=_RULED_OUT)
        for name in _FUSED_RULED_OUT
    ],
)
def test_fused_input_contract_is_rejected_before_launch(mutate, error, match):
    spec = _fused_spec()
    inp = make_inputs(spec, _BATCH)
    validate_inputs(spec, inp, _BATCH)  # the unmutated fixture is accepted
    mutate(inp)
    with pytest.raises(error, match=re.escape(FAILED + match)):
        validate_inputs(spec, inp, _BATCH)


def test_wider_fused_rows_on_16_byte_strides_are_accepted():
    """The contract admits rows wider than the kernel reads (``>=`` widths)."""
    spec = _fused_spec()
    inp = make_inputs(spec, _BATCH)
    inp["mixed_qkv"] = _rows(inp["mixed_qkv"], 776)
    inp["qkv_stride"] = 776
    inp["out_gate"] = _rows(inp["out_gate"], 257)
    inp["og_stride"] = 257
    validate_inputs(spec, inp, _BATCH)


def test_smallest_normal_f32_norm_eps_is_accepted():
    """The eps floor is the smallest normal f32, inclusive; int eps is fine."""
    spec = _fused_spec()
    inp = make_inputs(spec, _BATCH)
    for eps in (2.0**-126, 1):
        inp["norm_eps"] = eps
        validate_inputs(spec, inp, _BATCH)


def test_f32_spec_rejects_a_16_bit_pool():
    """A 16-bit pool under an f32 spec is half the bytes the kernel's slot
    stride walks: reading or writing it goes past the end of the pool."""
    spec = _fused_spec(state_dtype="f32", fuse_conv=False, fuse_out_norm=False)
    inp = make_inputs(spec, _BATCH)
    inp["state"] = storage.encode(inp["state"], "bf16")
    with _refused("state.dtype=uint16, expected float32 (fp32)"):
        validate_inputs(spec, inp, _BATCH)


@pytest.mark.parametrize("idx", ["read_indices", "write_indices"])
def test_fused_mode_keeps_the_skip_sentinel(idx):
    """-1 is still a legal (skip) index with the fused pools in play."""
    spec = _fused_spec()
    inp = make_inputs(spec, _BATCH)
    inp[idx][1] = -1
    validate_inputs(spec, inp, _BATCH)
