# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-side input-validation guard for the GDN decode kernel, without a GPU.

``prepare`` rejects inputs that would make the kernel read or write outside the
state pool. The decode kernel bounds-checks nothing on device beyond the ``-1``
skip sentinel, so this host guard *is* the memory-safety contract. The checks
raise before any launch, so they are pure host logic that runs on a CPU box --
which is exactly where a "this guard must not be silently dropped" regression
test belongs, rather than behind the on-device ``gpu`` gate.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch required (CPU build is fine)")

from builders.gfx950.gdn.gdn_decode import make_inputs, prepare
from kernels.gfx950.gdn_decode import GdnDecodeSpec

DEVICE = "cpu"


def test_out_of_range_index_is_rejected():
    """An index past the pool depth is an OOB access; prepare() must refuse it.

    ``-1`` stays legal (skip); any other out-of-pool value is rejected before a
    launch can touch it, for both the read and the write index.
    """
    spec = GdnDecodeSpec()
    batch = 8
    pool_depth = make_inputs(spec, batch, device=DEVICE)["state"].shape[0]

    for name, bad in (
        ("read_indices", pool_depth),  # == depth: the first OOB slot
        ("write_indices", pool_depth + 5),
        ("read_indices", -2),  # below the -1 skip sentinel
        ("write_indices", -2),
    ):
        inp = make_inputs(spec, batch, device=DEVICE)
        inp[name][0] = bad
        with pytest.raises(ValueError, match="out of range"):
            prepare(spec, inp, batch)


def test_duplicate_active_write_index_is_rejected():
    """Two live sequences cannot race to update one recurrent-state page."""
    spec = GdnDecodeSpec()
    batch = 8
    inp = make_inputs(spec, batch, device=DEVICE)
    inp["write_indices"][1] = inp["write_indices"][0]
    with pytest.raises(ValueError, match="unique across active sequences"):
        prepare(spec, inp, batch)


def test_inactive_mismatched_lane_does_not_reserve_a_write_index():
    """A lane with either negative index is inactive and cannot claim a page."""
    spec = GdnDecodeSpec()
    batch = 8
    inp = make_inputs(spec, batch, device=DEVICE)
    inp["read_indices"][1] = -1
    inp["write_indices"][1] = inp["write_indices"][0]
    prepare(spec, inp, batch)  # must not raise: lane 1 is inactive


def test_the_skip_sentinel_is_accepted():
    """``-1`` marks an idle continuous-batching slot and must pass the guard."""
    spec = GdnDecodeSpec()
    batch = 8
    inp = make_inputs(spec, batch, device=DEVICE)
    inp["read_indices"][1::2] = -1
    inp["write_indices"][1::2] = -1
    prepare(spec, inp, batch)  # must not raise


def test_wrong_state_head_dims_are_rejected():
    """A state pool whose head dims disagree with the spec is a shape bug, and
    the check is sync-free so it runs regardless of the value-range flag."""
    spec = GdnDecodeSpec()
    batch = 8
    inp = make_inputs(spec, batch, device=DEVICE)
    # Drop a slice of the K dim so the pool no longer matches the spec.
    inp["state"] = inp["state"][..., :-8].contiguous()
    with pytest.raises(ValueError, match="head dims"):
        prepare(spec, inp, batch, validate_indices=False)


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
    inp = make_inputs(spec, batch, device=DEVICE)
    inp["state"] = inp["state"].to(torch.float16)
    # Same shape, same byte count -- only the element type differs.
    with pytest.raises(ValueError, match="state dtype"):
        prepare(spec, inp, batch, validate_indices=False)


def test_validate_indices_flag_skips_the_range_check():
    """The value-range check reads the index extrema (a device sync on GPU), so
    it is flag-gated for the hot path.

    With it off, prepare() does not inspect the values and an out-of-pool index
    slips past; the sync-free shape checks still run. This pins the flag
    contract so the default-on guard cannot be silently lost.
    """
    spec = GdnDecodeSpec()
    batch = 8
    inp = make_inputs(spec, batch, device=DEVICE)
    inp["read_indices"][0] = inp["state"].shape[0]  # OOB, but unchecked
    prepare(spec, inp, batch, validate_indices=False)


def test_mis_shaped_input_tensor_is_rejected():
    """Every address is computed from SPEC dims, and the kernel emits no buffer
    descriptor -- there is no `num_records` to clamp an over-reach -- so a
    tensor whose real shape is smaller than the spec says is an out-of-bounds
    read with nothing between it and other allocations.
    """
    spec = GdnDecodeSpec()
    batch = 8
    for name in ("query", "key", "value", "a", "b", "dt_bias", "A_log"):
        inp = make_inputs(spec, batch, device=DEVICE)
        inp[name] = inp[name][..., :-1]  # one element short on the last axis
        with pytest.raises(ValueError, match=f"{name} must be"):
            prepare(spec, inp, batch)


def test_wrong_input_dtype_is_rejected():
    """The kernel is compiled against a fixed element type; handing it other
    bytes reinterprets them rather than converting."""
    spec = GdnDecodeSpec()
    batch = 8
    inp = make_inputs(spec, batch, device=DEVICE)
    inp["query"] = inp["query"].to(torch.float32)
    with pytest.raises(ValueError, match="query dtype"):
        prepare(spec, inp, batch)


def test_non_contiguous_input_is_rejected():
    """The doc promises row-major and every offset assumes it. A transposed
    view has the right shape and the wrong memory order, so the kernel would
    read the right INDEX out of the wrong ADDRESS -- silently."""
    spec = GdnDecodeSpec()
    batch = 8
    inp = make_inputs(spec, batch, device=DEVICE)
    # Same shape, non-unit strides: transpose two axes and transpose back via
    # a view that keeps the permuted layout.
    inp["value"] = inp["value"].transpose(2, 3).contiguous().transpose(2, 3)
    assert not inp["value"].is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        prepare(spec, inp, batch)


@pytest.mark.parametrize(
    "key,mutate,match",
    [
        ("a", lambda x: x[..., :1], r"a shape"),
        (
            "a",
            lambda x: x.transpose(-1, -2).contiguous().transpose(-1, -2),
            r"a.*contiguous",
        ),
        ("dt_bias", lambda x: x[..., :1], r"dt_bias shape"),
        ("dt_bias", lambda x: x.to(torch.bfloat16), r"dt_bias.*float32"),
        (
            "dt_bias",
            lambda x: x.transpose(-1, -2).contiguous().transpose(-1, -2),
            r"dt_bias.*contiguous",
        ),
    ],
)
def test_kda_gate_input_contract_is_rejected_before_launch(key, mutate, match):
    """KDA widens the gate buffers; legacy/malformed allocations must not launch.

    The emitter vector-loads ``a`` as ``[B,1,HV,DK]`` and ``dt_bias`` as f32
    ``[HV,DK]`` using spec-derived offsets. A legacy GDN-shaped or strided
    allocation is smaller/differently laid out than that compiled range.
    """
    import dataclasses as dc

    spec = dc.replace(GdnDecodeSpec(), gate_kind="kda")
    batch = 2
    inp = make_inputs(spec, batch, device=DEVICE)
    inp[key] = mutate(inp[key])

    with pytest.raises(ValueError, match=match):
        prepare(spec, inp, batch)


@pytest.mark.parametrize("key", ["a", "dt_bias"])
def test_kda_gate_inputs_must_match_query_device(key):
    """The launch contract rejects gate buffers on a different device."""
    import dataclasses as dc

    spec = dc.replace(GdnDecodeSpec(), gate_kind="kda")
    inp = make_inputs(spec, batch=2, device=DEVICE)
    inp[key] = torch.empty_like(inp[key], device="meta")

    with pytest.raises(ValueError, match=rf"{key} device"):
        prepare(spec, inp, batch=2)


def test_gdn_gate_input_contract_stays_scalar():
    """The KDA checks must not reject the existing scalar GDN ABI."""
    spec = GdnDecodeSpec()
    inp = make_inputs(spec, batch=2, device=DEVICE)

    assert tuple(inp["a"].shape) == (2, 1, spec.num_v_heads)
    assert tuple(inp["dt_bias"].shape) == (spec.num_v_heads,)
    prepare(spec, inp, batch=2)


def test_valid_kda_gate_inputs_survive_generic_validation():
    """Mode-specific KDA checks must prevent a second scalar-GDN recheck.

    The rebase first admitted KDA's `[B,1,HV,DK]` ``a`` and `[HV,DK]`` f32
    ``dt_bias``, then the generic loop rechecked both against GDN's scalar
    shapes and rejected every normal KDA launch before the kernel ran.
    """
    import dataclasses as dc

    spec = dc.replace(GdnDecodeSpec(), gate_kind="kda")
    inp = make_inputs(spec, batch=2, device=DEVICE)

    prepare(spec, inp, batch=2)


# ---- fused conv1d / gated-RMSNorm inputs ----------------------------------
#
# Every one of these feeds device address math (a row base, a pool slot, a
# vector load's alignment) or the norm's arithmetic, and nothing on the device
# checks it. Each case breaks exactly one rule and names the message of the
# guard that must fire.

_BATCH = 3


def _fused_spec(**kw):
    import dataclasses as dc

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


def _offset_view(t, elems):
    """A contiguous copy of ``t`` that starts ``elems`` elements into its
    storage, so its address is off the allocator's alignment."""
    flat = torch.zeros(t.numel() + elems, dtype=t.dtype, device=t.device)
    view = flat[elems:].view(t.shape)
    view.copy_(t)
    return view


def _rows(t, width, stride):
    """``t``'s rows placed ``stride`` elements apart in a wider buffer."""
    wide = torch.zeros(t.shape[0], stride, dtype=t.dtype, device=t.device)
    wide[:, : t.shape[1]] = t
    return wide[:, :width]


def _set(**kv):
    def mutate(inp):
        inp.update({k: v(inp) if callable(v) else v for k, v in kv.items()})

    return mutate


_FUSED_BAD_INPUTS = [
    # mixed_qkv: [B, >= conv_dim] rows, I/O dtype, unit channel stride
    (
        "mixed_qkv_short",
        _set(mixed_qkv=lambda i: i["mixed_qkv"][:, :-8]),
        r"mixed_qkv must be \[batch=3, >= 768\]",
    ),
    (
        "mixed_qkv_3d",
        _set(mixed_qkv=lambda i: i["mixed_qkv"][:, None]),
        r"mixed_qkv must be \[batch=3, >= 768\]",
    ),
    (
        "mixed_qkv_batch",
        _set(mixed_qkv=lambda i: i["mixed_qkv"][:-1]),
        r"mixed_qkv must be \[batch=3, >= 768\]",
    ),
    (
        "mixed_qkv_dtype",
        _set(mixed_qkv=lambda i: i["mixed_qkv"].half()),
        r"mixed_qkv dtype",
    ),
    (
        "mixed_qkv_channel_stride",
        _set(mixed_qkv=lambda i: i["mixed_qkv"].repeat_interleave(2, dim=1)[:, ::2]),
        r"mixed_qkv must have a unit channel stride",
    ),
    # qkv_stride: an int equal to the row stride, rows on a 16 B boundary
    (
        "qkv_stride_mismatch",
        _set(qkv_stride=lambda i: i["qkv_stride"] + 8),
        r"qkv_stride 776 != mixed_qkv.stride\(0\) 768",
    ),
    (
        "qkv_stride_float",
        _set(qkv_stride=lambda i: float(i["qkv_stride"])),
        r"qkv_stride must be an int",
    ),
    (
        "qkv_stride_misaligned",
        _set(mixed_qkv=lambda i: _rows(i["mixed_qkv"], 768, 772), qkv_stride=772),
        r"qkv_stride 772 puts mixed_qkv rows off the 16 B alignment",
    ),
    # conv_state: [pool, conv_dim, taps] with the SAME pool depth as state
    (
        "conv_state_pool_depth",
        _set(conv_state=lambda i: i["conv_state"][:-1]),
        r"conv_state shape \(6, 768, 3\) != \(7, 768, 3\)",
    ),
    (
        "conv_state_channels",
        _set(conv_state=lambda i: i["conv_state"][:, :-1]),
        r"conv_state shape",
    ),
    (
        "conv_state_dtype",
        _set(conv_state=lambda i: i["conv_state"].half()),
        r"conv_state dtype",
    ),
    (
        "conv_state_strided",
        _set(
            conv_state=lambda i: i["conv_state"]
            .transpose(1, 2)
            .contiguous()
            .transpose(1, 2)
        ),
        r"conv_state must be contiguous",
    ),
    # conv_weight: [conv_dim, width] f32
    (
        "conv_weight_shape",
        _set(conv_weight=lambda i: i["conv_weight"][:, :-1]),
        r"conv_weight shape",
    ),
    (
        "conv_weight_dtype",
        _set(conv_weight=lambda i: i["conv_weight"].bfloat16()),
        r"conv_weight dtype",
    ),
    (
        "conv_weight_strided",
        _set(conv_weight=lambda i: i["conv_weight"].t().contiguous().t()),
        r"conv_weight must be contiguous",
    ),
    # out_gate: [B, >= Hv*Dv] rows, I/O dtype, unit stride; og_stride matches
    (
        "out_gate_short",
        _set(out_gate=lambda i: i["out_gate"][:, :-1]),
        r"out_gate must be \[batch=3, >= 256\]",
    ),
    (
        "out_gate_dtype",
        _set(out_gate=lambda i: i["out_gate"].half()),
        r"out_gate dtype",
    ),
    (
        "out_gate_channel_stride",
        _set(out_gate=lambda i: i["out_gate"].repeat_interleave(2, dim=1)[:, ::2]),
        r"out_gate must have a unit channel stride",
    ),
    (
        "og_stride_mismatch",
        _set(og_stride=lambda i: i["og_stride"] + 1),
        r"og_stride 257 != out_gate.stride\(0\) 256",
    ),
    ("og_stride_float", _set(og_stride=256.0), r"og_stride must be an int"),
    # norm_weight: [Dv] f32; norm_eps: finite, positive, and still a normal f32
    # after the f32 kernarg pack
    (
        "norm_weight_shape",
        _set(norm_weight=lambda i: i["norm_weight"][:-1]),
        r"norm_weight shape",
    ),
    (
        "norm_weight_dtype",
        _set(norm_weight=lambda i: i["norm_weight"].half()),
        r"norm_weight dtype",
    ),
    ("norm_eps_zero", _set(norm_eps=0.0), r"norm_eps must be a finite positive"),
    ("norm_eps_negative", _set(norm_eps=-1e-6), r"norm_eps must be a finite positive"),
    (
        "norm_eps_nan",
        _set(norm_eps=float("nan")),
        r"norm_eps must be a finite positive",
    ),
    ("norm_eps_bool", _set(norm_eps=True), r"norm_eps must be a finite positive"),
    (
        "norm_eps_tensor",
        _set(norm_eps=lambda i: torch.tensor(1e-6)),
        r"norm_eps must be a finite positive",
    ),
    # Positive in Python but not as the f32 the kernel receives: 1e-50 packs
    # to 0.0, 1e-40 is an f32 denormal, 1e39 overflows f32.
    ("norm_eps_f32_zero", _set(norm_eps=1e-50), r"norm_eps must be a finite positive"),
    (
        "norm_eps_f32_denormal",
        _set(norm_eps=1e-40),
        r"norm_eps must be a finite positive",
    ),
    (
        "norm_eps_f32_overflow",
        _set(norm_eps=1e39),
        r"norm_eps must be a finite positive",
    ),
    # one device for every pointer (mixed_qkv is the reference)
    *[
        (
            f"{name}_device",
            _set(**{name: lambda i, n=name: torch.empty_like(i[n], device="meta")}),
            rf"{name} device meta != mixed_qkv device",
        )
        for name in ("conv_state", "conv_weight", "out_gate", "norm_weight", "state")
    ],
    # base alignment the vector accesses assume
    (
        "mixed_qkv_misaligned",
        _set(mixed_qkv=lambda i: _offset_view(i["mixed_qkv"], 1)),
        r"mixed_qkv must start on a 16 B boundary",
    ),
    (
        "conv_state_misaligned",
        _set(conv_state=lambda i: _offset_view(i["conv_state"], 1)),
        r"conv_state must start on a 16 B boundary",
    ),
    (
        "conv_weight_misaligned",
        _set(conv_weight=lambda i: _offset_view(i["conv_weight"], 1)),
        r"conv_weight must start on a 16 B boundary",
    ),
    (
        "dt_bias_misaligned",
        _set(dt_bias=lambda i: _offset_view(i["dt_bias"], 4)),
        r"dt_bias must start on a 32 B boundary",
    ),
]


@pytest.mark.parametrize(
    "mutate,match",
    [c[1:] for c in _FUSED_BAD_INPUTS],
    ids=[c[0] for c in _FUSED_BAD_INPUTS],
)
def test_fused_input_contract_is_rejected_before_launch(mutate, match):
    spec = _fused_spec()
    inp = make_inputs(spec, _BATCH, device=DEVICE)
    prepare(spec, inp, _BATCH)  # the unmutated fixture is accepted
    mutate(inp)
    with pytest.raises(ValueError, match=match):
        prepare(spec, inp, _BATCH)


def test_wider_fused_rows_on_16_byte_strides_are_accepted():
    """The contract admits row slices of a wider buffer (``>=`` widths)."""
    spec = _fused_spec()
    inp = make_inputs(spec, _BATCH, device=DEVICE)
    inp["mixed_qkv"] = _rows(inp["mixed_qkv"], 768, 776)
    inp["qkv_stride"] = 776
    inp["out_gate"] = _rows(inp["out_gate"], 256, 257)
    inp["og_stride"] = 257
    prepare(spec, inp, _BATCH)


def test_smallest_normal_f32_norm_eps_is_accepted():
    """The eps floor is the smallest normal f32, inclusive; int eps is fine."""
    spec = _fused_spec()
    inp = make_inputs(spec, _BATCH, device=DEVICE)
    for eps in (2.0**-126, 1):
        inp["norm_eps"] = eps
        prepare(spec, inp, _BATCH)


def test_f32_spec_rejects_a_16_bit_pool():
    """A 16-bit pool under an f32 spec is half the bytes the kernel's slot
    stride walks: reading or writing it goes past the end of the pool."""
    spec = _fused_spec(state_dtype="f32", fuse_conv=False, fuse_out_norm=False)
    inp = make_inputs(spec, _BATCH, device=DEVICE)
    inp["state"] = inp["state"].bfloat16()
    with pytest.raises(ValueError, match=r"state dtype torch.bfloat16 != spec"):
        prepare(spec, inp, _BATCH)


def test_unfused_query_must_be_aligned():
    spec = GdnDecodeSpec()
    inp = make_inputs(spec, _BATCH, device=DEVICE)
    inp["query"] = _offset_view(inp["query"], 1)
    with pytest.raises(ValueError, match=r"query must start on a 16 B boundary"):
        prepare(spec, inp, _BATCH)


@pytest.mark.parametrize("idx", ["read_indices", "write_indices"])
def test_fused_mode_keeps_the_skip_sentinel(idx):
    """-1 is still a legal (skip) index with the fused pools in play."""
    spec = _fused_spec()
    inp = make_inputs(spec, _BATCH, device=DEVICE)
    inp[idx][1] = -1
    prepare(spec, inp, _BATCH)
