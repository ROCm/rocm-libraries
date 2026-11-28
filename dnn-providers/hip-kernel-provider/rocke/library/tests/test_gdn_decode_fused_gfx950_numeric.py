# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-device numerics for the fused conv1d / gated-RMSNorm decode modes.

Scored against the whole-array float64 reference with the fused steps applied
(``ref_decode_step`` runs the conv1d + SiLU first and the gated RMSNorm last, with
the gate activation of each gate kind's model). ``check`` compares the output,
the written recurrent state, the written conv state, and every untouched page
of both pools, so a misplaced write fails.

A race needs more than that. The in-place conv-tap shift is ordered by a
workgroup barrier, and on silicon the race it prevents only shows with many
waves: with the barrier removed, 2-8 waves per workgroup stayed exact, while
16 waves corrupted the taps on every run. ``test_conv_in_place_slots``
therefore includes a 16-wave case, which fails without the barrier. The
barrier's position is also pinned on CPU (``test_gdn_decode_fused.py``,
``TestFusedBarrier``), independent of scheduling.

Specs are built directly from ``GdnDecodeSpec``: the fused modes are an emitter
capability, independent of any tile selection policy. The dispatched path
(request -> dispatch -> launch) is covered in
``test_gdn_decode_gfx950_numeric.py`` (``test_request_fields_reach_the_launch``).

Needs a real gfx950; marked ``gpu`` and skipped elsewhere. The
spec rules, ABI, reference and IR are covered on CPU by
``test_gdn_decode_fused.py``.
"""

from __future__ import annotations

import dataclasses as dc
from itertools import product

import numpy as np
import pytest
from rocke.core.dtypes import dtype_info
from rocke.numeric import storage
from rocke.tests.gpu import probe_device_arch
from rocke.tests.lanes import arch_skip_reason

ARCH = "gfx950"

pytestmark = pytest.mark.gpu

requires_gfx950 = pytest.mark.skipif(
    probe_device_arch()[0] != ARCH,
    reason=arch_skip_reason(("gfx950",), f"needs a {ARCH} device"),
)

FLAGS = [(True, False), (False, True), (True, True)]


def _fused(**kw):
    """A one-workgroup-per-head (BPV=1), Hk == Hv spec at the (4,16,1) tile."""
    from kernels.gfx950.gdn_decode import GdnDecodeSpec

    base = dict(
        num_k_heads=16,
        num_v_heads=16,
        num_warps=4,
        warp_threads_k=16,
        blocks_per_v_dim=1,
    )
    base.update(kw)
    return dc.replace(GdnDecodeSpec(), **base)


def _legal_bpv1_tiles(**kw):
    """Every (num_warps, warp_threads_k) the validator admits at BPV=1."""
    from kernels.gfx950.gdn_decode import is_valid_spec

    specs = []
    for nw, wtk in product((1, 2, 4, 8, 16), (1, 2, 4, 8, 16, 32)):
        spec = _fused(num_warps=nw, warp_threads_k=wtk, **kw)
        if is_valid_spec(spec, arch=ARCH)[0]:
            specs.append(spec)
    return specs


def _launch(spec, inp, batch):
    from builders.gfx950.gdn.gdn_decode import launcher_for, run_values

    return run_values(spec, inp, launcher_for(spec), batch)


def _same_bits(got, want):
    return np.array_equal(got.view(np.uint8), want.view(np.uint8))


@requires_gfx950
@pytest.mark.parametrize("gate", ["gdn", "kda"])
@pytest.mark.parametrize("dtype", ["bf16", "f16"])
@pytest.mark.parametrize("state_dtype", ["f16", "bf16", "f32"])
@pytest.mark.parametrize("flags", FLAGS, ids=["cv", "rn", "cv_rn"])
@pytest.mark.parametrize("batch", [1, 3, 16])
def test_fused_modes(gate, dtype, state_dtype, flags, batch):
    from builders.gfx950.gdn.gdn_decode import check

    conv, norm = flags
    spec = _fused(
        gate_kind=gate,
        dtype=dtype,
        state_dtype=state_dtype,
        fuse_conv=conv,
        fuse_out_norm=norm,
    )
    verdict = check(spec, batch)
    assert verdict.passed, (spec.kernel_name(), verdict)


@requires_gfx950
@pytest.mark.parametrize("gate", ["gdn", "kda"])
@pytest.mark.parametrize("state_dtype", ["f16", "bf16", "f32"])
def test_every_legal_fused_tile(gate, state_dtype):
    """Every BPV=1 tile the validator admits is exact, both flags on."""
    from builders.gfx950.gdn.gdn_decode import check

    specs = _legal_bpv1_tiles(
        gate_kind=gate, state_dtype=state_dtype, fuse_conv=True, fuse_out_norm=True
    )
    assert specs, "no legal fused tile"
    for spec in specs:
        verdict = check(spec, 3)
        assert verdict.passed, (spec.kernel_name(), verdict)


@requires_gfx950
@pytest.mark.parametrize("gate", ["gdn", "kda"])
@pytest.mark.parametrize("head_dim,wtk", [(64, 8), (256, 16)])
def test_fused_head_dims(gate, head_dim, wtk):
    """Head dims other than 128 change the conv_dim, the slot strides and the
    norm length; both flags on. A 64-wide key needs an 8-lane key group."""
    from builders.gfx950.gdn.gdn_decode import check
    from kernels.gfx950.gdn_decode import is_valid_spec

    spec = _fused(
        gate_kind=gate,
        head_k_dim=head_dim,
        head_v_dim=head_dim,
        warp_threads_k=wtk,
        state_dtype="f32",
        fuse_conv=True,
        fuse_out_norm=True,
    )
    ok, why = is_valid_spec(spec, arch=ARCH)
    assert ok, why
    verdict = check(spec, 3)
    assert verdict.passed, (spec.kernel_name(), verdict)


@requires_gfx950
@pytest.mark.parametrize("state_dtype", ["f16", "bf16", "f32"])
def test_norm_with_gqa_heads(state_dtype):
    """fuse_out_norm alone works for Hv > Hk (the GDN target's head layout)."""
    from builders.gfx950.gdn.gdn_decode import check

    spec = _fused(
        num_k_heads=16, num_v_heads=32, state_dtype=state_dtype, fuse_out_norm=True
    )
    verdict = check(spec, 8)
    assert verdict.passed, verdict


@requires_gfx950
@pytest.mark.parametrize("gate", ["gdn", "kda"])
def test_strided_rows(gate):
    """mixed_qkv and out_gate as rows of wider buffers: the kernel must read
    each row from row * stride, and only the first conv_dim / Hv*Dv channels
    of it."""
    from builders.gfx950.gdn.gdn_decode import gate as score
    from builders.gfx950.gdn.gdn_decode import make_inputs, reference

    spec = _fused(gate_kind=gate, fuse_conv=True, fuse_out_norm=True)
    batch = 5
    inp = make_inputs(spec, batch)
    for name, stride_name in (("mixed_qkv", "qkv_stride"), ("out_gate", "og_stride")):
        row = inp[name]
        # 24 extra elements keep each row on a 16 B boundary; fill them with
        # garbage that would show up in the output if read.
        garbage = storage.encode(1e4, "bf16")
        wide = np.full((batch, row.shape[1] + 24), garbage, dtype=row.dtype)
        wide[:, : row.shape[1]] = row
        inp[name] = wide
        inp[stride_name] = wide.shape[1]
    values = _launch(spec, inp, batch)
    verdict = score(spec, inp, values, reference(spec, inp))
    assert verdict.passed, verdict


@requires_gfx950
@pytest.mark.parametrize("gate", ["gdn", "kda"])
@pytest.mark.parametrize("num_warps", [1, 4])
def test_skip_sentinel_in_fused_mode(gate, num_warps):
    """A -1 lane skips the conv shift and the norm barrier with the rest of
    its body: its output stays zero and neither pool slot it names moves,
    while the active lanes in the same launch stay exact."""
    # Inline: lanes are skipped, which the shared gate (it assumes every lane
    # writes) cannot express. Every comparison fails on NaN.
    from builders.gfx950.gdn.gdn_decode import (
        TOL,
        input_dtypes,
        make_inputs,
        ref_conv_state_after,
        ref_decode_step,
    )

    spec = _fused(
        gate_kind=gate, num_warps=num_warps, fuse_conv=True, fuse_out_norm=True
    )
    batch = 6
    inp = make_inputs(spec, batch)
    ref_out, ref_state = ref_decode_step(spec, inp)
    ref_conv = ref_conv_state_after(spec, inp)
    skipped = np.array([1, 4])
    skipped_writes = inp["write_indices"][skipped]
    inp["read_indices"][skipped] = -1
    inp["write_indices"][skipped] = -1
    active = np.ones(batch, dtype=bool)
    active[skipped] = False
    values = _launch(spec, inp, batch)
    dtypes = input_dtypes(spec)

    assert not values["out"][skipped].any()  # all-zero bits: +0.0 everywhere
    out = storage.decode(values["out"], dtype_info(spec.dtype).name)
    written = inp["write_indices"][active]
    o_err = np.abs(out[active] - ref_out[active]) / np.maximum(
        np.abs(ref_out[active]), 1.0
    )
    assert o_err.max() < TOL
    for name, ref in (("state", ref_state), ("conv_state", ref_conv)):
        got = values[name]
        err = np.abs(storage.decode(got, dtypes[name])[written] - ref[active]).max()
        assert err < TOL, name
        assert _same_bits(got[skipped_writes], inp[name][skipped_writes]), name
        # Every slot no active lane writes (the skipped lanes' slots and the
        # spare pages) is bit-identical to the pool before the launch; ``inp``
        # still holds it, since the launch ran on device copies.
        untouched = np.ones(got.shape[0], dtype=bool)
        untouched[written] = False
        assert untouched.any(), name
        assert _same_bits(got[untouched], inp[name][untouched]), name


@requires_gfx950
@pytest.mark.parametrize("gate", ["gdn", "kda"])
@pytest.mark.parametrize("num_warps", [1, 2, 4, 16])
@pytest.mark.parametrize("fuse_out_norm", [False, True], ids=["cv", "cv_rn"])
def test_conv_in_place_slots(gate, num_warps, fuse_out_norm):
    """read slot == write slot: the in-place conv-state shift must stay exact
    with one wave (no barrier) and with several (a barrier before the writes,
    explicit without the norm, the norm reduction's own with it). The 16-wave
    case is the one that goes red when the barrier is removed (module
    docstring)."""
    from builders.gfx950.gdn.gdn_decode import gate as score
    from builders.gfx950.gdn.gdn_decode import make_inputs, reference

    spec = _fused(
        gate_kind=gate,
        num_warps=num_warps,
        fuse_conv=True,
        fuse_out_norm=fuse_out_norm,
    )
    batch = 8
    # Pool depth == batch: every slot is written in place, so the gate's
    # untouched-slot set is empty (nothing to spill into).
    inp = make_inputs(spec, batch, pool_depth=batch, disjoint_writes=False)
    values = _launch(spec, inp, batch)
    verdict = score(spec, inp, values, reference(spec, inp))
    assert verdict.passed, verdict
