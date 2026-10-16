# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-device numerics for the fused conv1d / gated-RMSNorm decode modes.

Scored against the whole-tensor fp32 reference with the fused steps applied
(``ref_fp32`` runs the conv1d + SiLU first and the gated RMSNorm last, with
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
capability, independent of any tile selection policy.

Needs a real gfx950 and ROCm torch; marked ``gpu`` and skipped elsewhere. The
spec rules, ABI, reference and IR are covered on CPU by
``test_gdn_decode_fused.py``.
"""

from __future__ import annotations

import dataclasses as dc
from itertools import product

import pytest

ARCH = "gfx950"

torch = pytest.importorskip("torch", reason="ROCm torch required")

pytestmark = pytest.mark.gpu


def _device_is_gfx950() -> bool:
    # rocke's own query (hipDeviceGetAttribute, feature flags stripped), as in
    # test_gdn_decode_gfx950_numeric.py, rather than a substring of torch's.
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


@requires_gfx950
@pytest.mark.parametrize("gate", ["gdn", "kda"])
@pytest.mark.parametrize("dtype", ["bf16", "f16"])
@pytest.mark.parametrize("state_dtype", ["f16", "bf16", "f32"])
@pytest.mark.parametrize("flags", FLAGS, ids=["cv", "rn", "cv_rn"])
@pytest.mark.parametrize("batch", [1, 3, 16])
def test_fused_modes(gate, dtype, state_dtype, flags, batch):
    from builders.gfx950.gdn.gdn_decode import TOL, check

    conv, norm = flags
    spec = _fused(
        gate_kind=gate,
        dtype=dtype,
        state_dtype=state_dtype,
        fuse_conv=conv,
        fuse_out_norm=norm,
    )
    out_err, state_err = check(spec, batch)
    assert out_err < TOL and state_err < TOL, (spec.kernel_name(), out_err, state_err)


@requires_gfx950
@pytest.mark.parametrize("gate", ["gdn", "kda"])
@pytest.mark.parametrize("state_dtype", ["f16", "bf16", "f32"])
def test_every_legal_fused_tile(gate, state_dtype):
    """Every BPV=1 tile the validator admits is exact, both flags on."""
    from builders.gfx950.gdn.gdn_decode import TOL, check

    specs = _legal_bpv1_tiles(
        gate_kind=gate, state_dtype=state_dtype, fuse_conv=True, fuse_out_norm=True
    )
    assert specs, "no legal fused tile"
    for spec in specs:
        out_err, state_err = check(spec, 3)
        assert out_err < TOL and state_err < TOL, (
            spec.kernel_name(),
            out_err,
            state_err,
        )


@requires_gfx950
@pytest.mark.parametrize("gate", ["gdn", "kda"])
@pytest.mark.parametrize("head_dim,wtk", [(64, 8), (256, 16)])
def test_fused_head_dims(gate, head_dim, wtk):
    """Head dims other than 128 change the conv_dim, the slot strides and the
    norm length; both flags on. A 64-wide key needs an 8-lane key group."""
    from builders.gfx950.gdn.gdn_decode import TOL, check
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
    out_err, state_err = check(spec, 3)
    assert out_err < TOL and state_err < TOL, (spec.kernel_name(), out_err, state_err)


@requires_gfx950
@pytest.mark.parametrize("state_dtype", ["f16", "bf16", "f32"])
def test_norm_with_gqa_heads(state_dtype):
    """fuse_out_norm alone works for Hv > Hk (the GDN target's head layout)."""
    from builders.gfx950.gdn.gdn_decode import TOL, check

    spec = _fused(
        num_k_heads=16, num_v_heads=32, state_dtype=state_dtype, fuse_out_norm=True
    )
    out_err, state_err = check(spec, 8)
    assert out_err < TOL and state_err < TOL, (out_err, state_err)


@requires_gfx950
@pytest.mark.parametrize("gate", ["gdn", "kda"])
def test_strided_rows(gate):
    """mixed_qkv and out_gate as row slices of wider buffers: the kernel must
    read each row from row * stride, and only the first conv_dim / Hv*Dv
    channels of it."""
    from builders.gfx950.gdn.gdn_decode import TOL, make_inputs, out_error, ref_fp32

    spec = _fused(gate_kind=gate, fuse_conv=True, fuse_out_norm=True)
    batch = 5
    inp = make_inputs(spec, batch)
    for name, stride_name in (("mixed_qkv", "qkv_stride"), ("out_gate", "og_stride")):
        row = inp[name]
        # 24 extra elements keep each row on a 16 B boundary; fill them with
        # garbage that would show up in the output if read.
        wide = torch.full(
            (batch, row.shape[1] + 24), 1e4, dtype=row.dtype, device=row.device
        )
        wide[:, : row.shape[1]] = row
        inp[name] = wide[:, : row.shape[1]]
        inp[stride_name] = wide.stride(0)
    ref_out, ref_state = ref_fp32(spec, inp)
    values = _launch(spec, inp, batch)
    written = inp["write_indices"].long()
    assert out_error(spec, values["out"], ref_out) < TOL
    assert (values["state"].float()[written] - ref_state).abs().max().item() < TOL


@requires_gfx950
@pytest.mark.parametrize("gate", ["gdn", "kda"])
@pytest.mark.parametrize("num_warps", [1, 4])
def test_skip_sentinel_in_fused_mode(gate, num_warps):
    """A -1 lane skips the conv shift and the norm barrier with the rest of
    its body: its output stays zero and neither pool slot it names moves,
    while the active lanes in the same launch stay exact."""
    from builders.gfx950.gdn.gdn_decode import (
        TOL,
        make_inputs,
        ref_conv_state_after,
        ref_fp32,
    )

    spec = _fused(
        gate_kind=gate, num_warps=num_warps, fuse_conv=True, fuse_out_norm=True
    )
    batch = 6
    inp = make_inputs(spec, batch)
    ref_out, ref_state = ref_fp32(spec, inp)
    ref_conv = ref_conv_state_after(spec, inp)
    skipped = torch.tensor([1, 4], device=inp["read_indices"].device)
    skipped_writes = inp["write_indices"][skipped].long()
    inp["read_indices"][skipped] = -1
    inp["write_indices"][skipped] = -1
    active = torch.ones(batch, dtype=torch.bool, device=skipped.device)
    active[skipped] = False
    values = _launch(spec, inp, batch)

    out = values["out"].float()
    assert torch.equal(out[skipped], torch.zeros_like(out[skipped]))
    written = inp["write_indices"][active].long()
    o_err = (out[active] - ref_out[active]).abs() / ref_out[active].abs().clamp(min=1)
    assert o_err.max().item() < TOL
    for name, ref in (("state", ref_state), ("conv_state", ref_conv)):
        got = values[name]
        assert (got.float()[written] - ref[active]).abs().max().item() < TOL, name
        assert torch.equal(got[skipped_writes], inp[name][skipped_writes]), name


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
    from builders.gfx950.gdn.gdn_decode import (
        TOL,
        make_inputs,
        out_error,
        ref_conv_state_after,
        ref_fp32,
    )

    spec = _fused(
        gate_kind=gate,
        num_warps=num_warps,
        fuse_conv=True,
        fuse_out_norm=fuse_out_norm,
    )
    batch = 8
    inp = make_inputs(spec, batch, pool_depth=batch, disjoint_writes=False)
    ref_out, ref_state = ref_fp32(spec, inp)
    ref_conv = ref_conv_state_after(spec, inp)
    values = _launch(spec, inp, batch)
    slots = inp["write_indices"].long()
    assert out_error(spec, values["out"], ref_out) < TOL
    assert (values["state"].float()[slots] - ref_state).abs().max().item() < TOL
    assert (values["conv_state"].float()[slots] - ref_conv).abs().max().item() < TOL
