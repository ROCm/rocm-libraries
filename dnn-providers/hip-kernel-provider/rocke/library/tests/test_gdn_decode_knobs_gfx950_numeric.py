# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-device numerics of every performance knob (ALGORITHM.md §4.10).

Each knob is switched on alone, on every tile of every class it applies to:
gate kind (gdn / kda) x plain / fused x state dtype (bf16 / f32), at the
default heads (DK = DV = 128). A plain class runs every registered tile
(``CONFIGURED_TILES``) and the reference path; a fused class runs every BPV=1
tile of the same warp and K-lane counts. A knob is skipped on a tile where its
``*_applies`` predicate is False or the validator rejects the spec: there it
emits the knob-off kernel, which the host test ``test_gdn_decode_knobs.py``
pins byte for byte. Every knob also runs at DK = DV = 64 and 256, and
``out_lds`` at DV = 256 and 512 and with two store rounds.

The sweeps above use bf16 I/O. f16 I/O runs every knob with a bf16 and an
f32 state on the default tile, the reference path and the fused (4,16,1) tile
in every fusion mode the knob accepts; an f16 state runs the same specs
against the fp32 reference only (see ``test_knob_with_f16_state``).

``check`` scores the output, the written recurrent state, the written conv
state, and every untouched pool page against the fp32 reference, so a knob
that misplaces, drops or races a write fails. The knobs documented as running
the same floating-point operations are also compared bit for bit with the
knob-off kernel on identical inputs (out_lds's f16 output within one f16 ulp,
see ``_assert_bitwise``).

Needs a real gfx950 and ROCm torch; marked ``gpu`` and skipped elsewhere.
"""

from __future__ import annotations

import dataclasses as dc
from itertools import product

import pytest

ARCH = "gfx950"

torch = pytest.importorskip("torch", reason="ROCm torch required")

pytestmark = pytest.mark.gpu


def _device_is_gfx950() -> bool:
    # rocke's own query, as the sibling GDN suites do: get_device_arch goes
    # through hipDeviceGetAttribute and strips the feature flags.
    if not torch.cuda.is_available():
        return False
    try:
        from rocke.runtime.hip_module import get_device_arch

        return get_device_arch() == ARCH
    except Exception:
        return False


requires_gfx950 = pytest.mark.skipif(
    not _device_is_gfx950(), reason=f"needs a {ARCH} device"
)

# knob -> the spec overrides that switch it on (norm_gate_once needs conv_once)
KNOBS = {
    "state_load_hint": dict(state_load_hint="streaming"),
    "state_store_hint": dict(state_store_hint="streaming"),
    "dpp_reduce": dict(dpp_reduce=True),
    "xcd_remap": dict(xcd_remap=True),
    "stream_rows": dict(stream_rows=True),
    "interleave_cols": dict(interleave_cols=True),
    "conv_once": dict(conv_once=True),
    "norm_gate_once": dict(conv_once=True, norm_gate_once=True),
    "out_lds": dict(out_lds=True),
    "waves_per_eu": dict(waves_per_eu=4),
}
_FUSED_HEADS = dict(num_k_heads=16, num_v_heads=16)


def _plain_tiles():
    """Every registered tile, then the reference path."""
    from dispatch.gdn.gfx950 import CONFIGURED_TILES

    tiles = [
        dict(num_warps=nw, warp_threads_k=wtk, blocks_per_v_dim=bpv)
        for nw, wtk, bpv in CONFIGURED_TILES
    ]
    return tiles + [dict(simple=True)]


def _fused_tiles():
    """Every BPV=1 tile of the registered warp and K-lane counts."""
    from dispatch.gdn.gfx950 import NUM_WARPS, WARP_THREADS_K

    return [
        dict(num_warps=nw, warp_threads_k=wtk, blocks_per_v_dim=1)
        for nw, wtk in product(NUM_WARPS, WARP_THREADS_K)
    ]


def _applies(spec, knob) -> bool:
    from kernels.gfx950 import gdn_decode as g

    pred = {
        "dpp_reduce": g.dpp_reduce_applies,
        "xcd_remap": g.xcd_remap_applies,
        "stream_rows": g.stream_rows_applies,
        "interleave_cols": g.interleave_cols_applies,
    }.get(knob)
    return pred is None or pred(spec)


def _knob_specs(knob, gate, fused, state_dtype):
    """Every legal knob-on spec of one class where the knob changes code."""
    from kernels.gfx950.gdn_decode import GdnDecodeSpec, is_valid_spec

    base = dict(gate_kind=gate, state_dtype=state_dtype, **KNOBS[knob])
    if fused:
        heads = _FUSED_HEADS
        tiles = _fused_tiles()
        # conv only, norm only, both: each fused knob runs on every mode it
        # accepts.
        modes = [(True, False), (False, True), (True, True)]
    else:
        heads = {}
        tiles = _plain_tiles()
        modes = [(False, False)]
    specs = []
    for tile, (conv, norm) in product(tiles, modes):
        spec = dc.replace(
            GdnDecodeSpec(),
            **heads,
            **base,
            **tile,
            fuse_conv=conv,
            fuse_out_norm=norm,
        )
        if is_valid_spec(spec, arch=ARCH)[0] and _applies(spec, knob):
            specs.append(spec)
    return specs


def _assert_exact(spec, batch):
    from builders.gfx950.gdn.gdn_decode import TOL, check

    out_err, state_err = check(spec, batch)
    assert out_err < TOL and state_err < TOL, (
        spec.kernel_name(),
        batch,
        out_err,
        state_err,
    )


@requires_gfx950
@pytest.mark.parametrize("knob", sorted(KNOBS))
@pytest.mark.parametrize("gate", ["gdn", "kda"])
@pytest.mark.parametrize("fused", [False, True], ids=["plain", "fused"])
@pytest.mark.parametrize("state_dtype", ["bf16", "f32"])
def test_knob_on_every_tile_it_applies_to(knob, gate, fused, state_dtype):
    specs = _knob_specs(knob, gate, fused, state_dtype)
    if not specs:
        # Only the fused-only knobs on a plain class and interleave_cols on a
        # 2-byte state have no tile to change.
        assert (not fused and knob in ("conv_once", "norm_gate_once", "out_lds")) or (
            knob == "interleave_cols" and state_dtype != "f32"
        ), (knob, gate, fused, state_dtype)
        pytest.skip(f"{knob} changes no code in this class")
    for spec in specs:
        _assert_exact(spec, 3)


@requires_gfx950
@pytest.mark.parametrize("fused", [False, True], ids=["plain", "fused"])
@pytest.mark.parametrize("batch", [1, 2, 3, 4, 13, 16])
def test_xcd_remap_on_a_grid_that_is_not_a_multiple_of_8(fused, batch):
    """Five heads at BPV=1: grids of 5, 10, 15, 20, 65 and 80 workgroups.
    The remap must be a bijection, or a head is computed twice and another
    never. At 10, 20 and 65 (npid % 8 in 1..6, npid > 8) some XCDs get one
    workgroup fewer, so the remap's short-XCD correction is nonzero there;
    at 5, 15 and 80 it is always 0."""
    from kernels.gfx950.gdn_decode import GdnDecodeSpec, is_valid_spec

    spec = dc.replace(
        GdnDecodeSpec(),
        num_k_heads=5,
        num_v_heads=5,
        num_warps=2,
        warp_threads_k=16,
        blocks_per_v_dim=1,
        fuse_conv=fused,
        fuse_out_norm=fused,
        xcd_remap=True,
    )
    assert is_valid_spec(spec, arch=ARCH)[0], is_valid_spec(spec, arch=ARCH)[1]
    _assert_exact(spec, batch)


@requires_gfx950
@pytest.mark.parametrize("gate", ["gdn", "kda"])
@pytest.mark.parametrize("state_dtype", ["bf16", "f32"])
def test_every_knob_together(gate, state_dtype):
    """All ten knobs on at once, at the fused (4,16,1) and (1,16,1) tiles."""
    from kernels.gfx950.gdn_decode import GdnDecodeSpec

    for nw in (1, 4):
        knobs = {}
        for kw in KNOBS.values():
            knobs.update(kw)
        spec = dc.replace(
            GdnDecodeSpec(),
            num_k_heads=16,
            num_v_heads=16,
            gate_kind=gate,
            state_dtype=state_dtype,
            num_warps=nw,
            warp_threads_k=16,
            blocks_per_v_dim=1,
            fuse_conv=True,
            fuse_out_norm=True,
            **knobs,
        )
        for batch in (1, 16):
            _assert_exact(spec, batch)


# The knobs ALGORITHM.md §4.10 says run the same floating-point operations on
# the same values. The fp32 tolerance above cannot tell a reordered sum from
# the original, so these are compared to the knob-off kernel bit for bit.
_BITWISE_KNOBS = [
    "state_load_hint",
    "state_store_hint",
    "dpp_reduce",
    "xcd_remap",
    "stream_rows",
    "conv_once",
    "out_lds",
    "waves_per_eu",
]


def _f16_ulps(x, y):
    """Elementwise distance in f16 ulps (units in the last place) of two
    f16 tensors, through their bit patterns mapped to ordered integers."""

    def ordered(t):
        i = t.view(torch.int16).to(torch.int32)
        return torch.where(i < 0, -32768 - i, i)

    return (ordered(x) - ordered(y)).abs()


def _assert_bitwise(spec, knob):
    """``spec`` and its knob-off twin give identical output, state and conv
    state on identical inputs. Exception: out_lds with f16 output may differ by
    one f16 ulp in ``out``. The knob-off path folds its last multiply and the f16
    conversion into one v_fma_mixlo_f16 (one rounding), out_lds rounds twice
    (seen in the ROCm 7.13 code object, not pinned by a test).

    Blind spot: with f16 output, an out_lds bug that moves ``out`` by at most
    one ulp passes here. The bf16-output comparisons run the same IR bit for
    bit and are the guard for that."""
    from builders.gfx950.gdn.gdn_decode import (
        drain,
        launch,
        launcher_for,
        make_inputs,
        prepare,
    )

    default = spec.__class__()
    off = dc.replace(spec, **{k: getattr(default, k) for k in KNOBS[knob]})
    inp = make_inputs(off, 3)
    arms = []
    for s in (off, spec):
        values, cfg = prepare(s, inp, 3)
        launch(launcher_for(s), values, cfg)
        drain()
        arms.append(values)
    for key in ("out", "state") + (("conv_state",) if spec.fuse_conv else ()):
        a, b = arms[0][key], arms[1][key]
        if key == "out" and knob == "out_lds" and spec.dtype == "f16":
            ulps = int(_f16_ulps(a, b).max())
            assert ulps <= 1, (spec.kernel_name(), key, ulps)
        else:
            assert torch.equal(a, b), (spec.kernel_name(), key)


@requires_gfx950
@pytest.mark.parametrize("knob", _BITWISE_KNOBS)
@pytest.mark.parametrize("gate", ["gdn", "kda"])
@pytest.mark.parametrize("state_dtype", ["bf16", "f32"])
def test_knob_output_is_bitwise_the_knob_off_output(knob, gate, state_dtype):
    specs = [
        s
        for fused in (False, True)
        for s in _knob_specs(knob, gate, fused, state_dtype)
    ]
    assert specs, (knob, gate, state_dtype)
    for spec in specs:
        _assert_bitwise(spec, knob)


@requires_gfx950
@pytest.mark.parametrize("gate", ["gdn", "kda"])
@pytest.mark.parametrize("num_warps", [1, 2, 4])
@pytest.mark.parametrize("fuse_out_norm", [False, True], ids=["cv", "cv_rn"])
def test_conv_once_in_place_slots(gate, num_warps, fuse_out_norm):
    """read slot == write slot: conv_once shifts each channel's taps from the
    thread that read them, with no barrier, so the in-place shift must stay
    exact with any number of waves. The pool keeps its ``batch + 1`` spare
    pages, which ``check`` compares too, so a write that lands on a spare page
    fails.

    Blind spot: read slot == write slot, so a write misplaced onto the read slot
    passes here. test_knob_on_every_tile_it_applies_to (disjoint read and write
    slots) is the guard for that."""
    from builders.gfx950.gdn.gdn_decode import TOL, check
    from kernels.gfx950.gdn_decode import GdnDecodeSpec

    spec = dc.replace(
        GdnDecodeSpec(),
        num_k_heads=16,
        num_v_heads=16,
        gate_kind=gate,
        num_warps=num_warps,
        warp_threads_k=16,
        blocks_per_v_dim=1,
        fuse_conv=True,
        fuse_out_norm=fuse_out_norm,
        conv_once=True,
        norm_gate_once=fuse_out_norm,
    )
    out_err, state_err = check(spec, 8, disjoint_writes=False)
    assert out_err < TOL and state_err < TOL, (spec.kernel_name(), out_err, state_err)


# (plain tile, fused tile) per head dim DK = DV: both have more than one state
# row per lane and WTK >= 8, so every tile-dependent knob changes code.
_HEAD_DIM_TILES = {
    64: ((2, 8, 2), (2, 8, 1)),
    256: ((2, 16, 8), (8, 16, 1)),
}


@requires_gfx950
@pytest.mark.parametrize("knob", sorted(KNOBS))
@pytest.mark.parametrize("head_dim", sorted(_HEAD_DIM_TILES))
@pytest.mark.parametrize("gate", ["gdn", "kda"])
@pytest.mark.parametrize("state_dtype", ["bf16", "f32"])
def test_knob_at_other_head_dims(knob, head_dim, gate, state_dtype):
    """Every knob at DK = DV = 64 and 256, warp-tiled and in every fused
    mode it accepts; the bit-for-bit knobs also against the knob-off kernel."""
    from kernels.gfx950.gdn_decode import GdnDecodeSpec, is_valid_spec

    (pw, pk, pb), (fw, fk, fb) = _HEAD_DIM_TILES[head_dim]
    base = dc.replace(
        GdnDecodeSpec(),
        head_k_dim=head_dim,
        head_v_dim=head_dim,
        gate_kind=gate,
        state_dtype=state_dtype,
        **KNOBS[knob],
    )
    cands = [dc.replace(base, num_warps=pw, warp_threads_k=pk, blocks_per_v_dim=pb)] + [
        dc.replace(
            base,
            **_FUSED_HEADS,
            num_warps=fw,
            warp_threads_k=fk,
            blocks_per_v_dim=fb,
            fuse_conv=conv,
            fuse_out_norm=norm,
        )
        for conv, norm in ((True, False), (False, True), (True, True))
    ]
    specs = [s for s in cands if is_valid_spec(s, arch=ARCH)[0] and _applies(s, knob)]
    assert specs or (knob == "interleave_cols" and state_dtype != "f32")
    for spec in specs:
        _assert_exact(spec, 3)
        if knob in _BITWISE_KNOBS:
            _assert_bitwise(spec, knob)


# out_lds row groups wider than the DK = DV = 128 tiles' 2 rows per storing
# lane: 4 rows (DV = 256), 8 rows (DV = 512), and DV / 8 = 72 storing lanes on
# a 64-lane workgroup, so the second store round is partly guarded off.
_WIDE_OUT_LDS = {
    "dv256": dict(num_k_heads=16, num_v_heads=16, head_v_dim=256, num_warps=8),
    "dv512": dict(num_k_heads=16, num_v_heads=16, head_v_dim=512, num_warps=16),
    "2rounds": dict(
        num_k_heads=4,
        num_v_heads=4,
        head_k_dim=8,
        head_v_dim=576,
        num_warps=1,
        warp_threads_k=1,
    ),
}


@requires_gfx950
@pytest.mark.parametrize("case", sorted(_WIDE_OUT_LDS))
@pytest.mark.parametrize(
    "mode",
    [
        dict(fuse_out_norm=True),
        dict(fuse_conv=True, fuse_out_norm=True),
        dict(fuse_conv=True, fuse_out_norm=True, conv_once=True, norm_gate_once=True),
    ],
    ids=["rn", "cv_rn", "cv_rn_ngo"],
)
@pytest.mark.parametrize("gate", ["gdn", "kda"])
@pytest.mark.parametrize("state_dtype", ["bf16", "f32"])
def test_out_lds_wide_row_groups(case, mode, gate, state_dtype):
    from kernels.gfx950.gdn_decode import GdnDecodeSpec, is_valid_spec

    spec = dc.replace(
        GdnDecodeSpec(),
        gate_kind=gate,
        state_dtype=state_dtype,
        blocks_per_v_dim=1,
        out_lds=True,
        **{"warp_threads_k": 16, **_WIDE_OUT_LDS[case]},
        **mode,
    )
    assert is_valid_spec(spec, arch=ARCH)[0], is_valid_spec(spec, arch=ARCH)[1]
    for batch in (1, 3):
        _assert_exact(spec, batch)
    _assert_bitwise(spec, "out_lds")


def _f16_io_specs(knob, gate, state_dtype):
    """Knob-on specs with f16 I/O where the knob changes code: the default
    tile, the reference path, and the fused (4,16,1) tile in every fusion
    mode the knob accepts."""
    from kernels.gfx950.gdn_decode import GdnDecodeSpec, is_valid_spec

    base = dc.replace(
        GdnDecodeSpec(),
        dtype="f16",
        gate_kind=gate,
        state_dtype=state_dtype,
        **KNOBS[knob],
    )
    cands = [base, dc.replace(base, simple=True)] + [
        dc.replace(
            base,
            **_FUSED_HEADS,
            num_warps=4,
            warp_threads_k=16,
            blocks_per_v_dim=1,
            fuse_conv=conv,
            fuse_out_norm=norm,
        )
        for conv, norm in ((True, False), (False, True), (True, True))
    ]
    specs = [s for s in cands if is_valid_spec(s, arch=ARCH)[0] and _applies(s, knob)]
    assert specs, (knob, gate, state_dtype)
    return specs


# interleave_cols changes code only with an f32 state, so it is left out of
# the 2-byte-state cases rather than skipped there.
_F16_IO_CASES = [
    (knob, state_dtype)
    for knob in sorted(KNOBS)
    for state_dtype in ("bf16", "f32")
    if knob != "interleave_cols" or state_dtype == "f32"
]


@requires_gfx950
@pytest.mark.parametrize("knob, state_dtype", _F16_IO_CASES)
@pytest.mark.parametrize("gate", ["gdn", "kda"])
def test_knob_with_f16_io(knob, state_dtype, gate):
    """f16 I/O is an admitted dtype, and the knobs add f16-sensitive loads and
    stores (conv_once's per-channel taps, out_lds's packed stores, the
    interleave_cols runs): each knob against the fp32 reference, and the
    bit-for-bit knobs also against the knob-off kernel (out_lds's ``out``
    within one f16 ulp)."""
    for spec in _f16_io_specs(knob, gate, state_dtype):
        _assert_exact(spec, 3)
        if knob in _BITWISE_KNOBS:
            _assert_bitwise(spec, knob)


@requires_gfx950
@pytest.mark.parametrize("knob", sorted(set(KNOBS) - {"interleave_cols"}))
@pytest.mark.parametrize("gate", ["gdn", "kda"])
def test_knob_with_f16_state(knob, gate):
    """An f16 state with f16 I/O, against the fp32 reference only. Bit
    identity with the knob-off kernel is not claimed for an f16 state: it
    has not been established that both round the state the same way.
    (interleave_cols needs an f32 state.)"""
    for spec in _f16_io_specs(knob, gate, "f16"):
        _assert_exact(spec, 3)
