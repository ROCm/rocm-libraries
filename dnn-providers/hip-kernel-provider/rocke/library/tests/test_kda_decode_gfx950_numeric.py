# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-device numeric checks for the KDA gate kind of the decode kernel.

Scored against the same whole-tensor fp32 reference the GDN mode uses, with the
per-channel gate selected. The reference shares no algebra with the kernel --
no tiling, no warp structure, no cross-lane reductions -- so agreement is
evidence about the algorithm rather than a restatement of the implementation.
Its per-channel path is separately audited on CPU by
``test_kda_decode_reference.py``, including which axis the decay scales.

Both results are compared. The kernel writes its output *and* mutates the
recurrent state in place; checking only the output would let a corrupted state
write ship silently, because nothing reads the state back until the next decode
step.

These lanes need a real gfx950 and ROCm torch, so they are marked ``gpu`` and
skipped elsewhere. The spec rules and the reference are covered by CPU-only
tests so this family still contributes coverage without a device.
"""

from __future__ import annotations

import dataclasses as dc

import pytest

ARCH = "gfx950"

torch = pytest.importorskip("torch", reason="ROCm torch required")

pytestmark = pytest.mark.gpu


from dispatch.gdn.gfx950 import _TUNED_TILES_KDA  # noqa: E402


def _device_is_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return ARCH in torch.cuda.get_device_properties(0).gcnArchName
    except Exception:
        return False


requires_gfx950 = pytest.mark.skipif(
    not _device_is_gfx950(), reason=f"needs a {ARCH} device"
)


@pytest.fixture(scope="module")
def harness():
    from builders.gfx950.gdn.gdn_decode import (
        TOL,
        check,
        make_inputs,
        prepare,
        ref_fp32,
    )

    return {
        "TOL": TOL,
        "check": check,
        "make_inputs": make_inputs,
        "prepare": prepare,
        "ref_fp32": ref_fp32,
    }


def _kda(**kw):
    from kernels.gfx950.gdn_decode import GdnDecodeSpec

    return dc.replace(GdnDecodeSpec(), gate_kind="kda", **kw)


@requires_gfx950
@pytest.mark.parametrize("batch", [1, 3, 16, 64])
def test_kda_simple_path_matches_reference(harness, batch):
    """The one-thread-per-row reference emitter, with a per-channel gate.

    This path owns a whole state row per thread, so the gate vector maps onto
    it directly with no warp-tiled indexing in the way. Proving the numerics
    here first means a later warp-tiled failure is an indexing bug and not a
    gate-formula bug.
    """
    out_err, state_err = harness["check"](_kda(simple=True), batch)

    assert out_err <= harness["TOL"], f"KDA simple output error {out_err:.3e}"
    assert state_err <= harness["TOL"], f"KDA simple state error {state_err:.3e}"


@requires_gfx950
def test_gdn_simple_path_still_matches_reference(harness):
    """The GDN gate must be untouched by the KDA branch living beside it."""
    from kernels.gfx950.gdn_decode import GdnDecodeSpec

    out_err, state_err = harness["check"](GdnDecodeSpec(simple=True), 4)

    assert max(out_err, state_err) <= harness["TOL"]


@requires_gfx950
@pytest.mark.parametrize("batch", [1, 3, 16, 64])
def test_kda_warp_tiled_matches_reference(harness, batch):
    """The production warp-tiled emitter, with a per-channel gate.

    The simple path above already proves the gate formula, so a failure here
    is an indexing bug: this emitter splits the DK axis across lanes, and the
    decay vector has to line up with the state slice each lane owns.
    """
    out_err, state_err = harness["check"](_kda(), batch)

    assert out_err <= harness["TOL"], f"KDA warp-tiled output error {out_err:.3e}"
    assert state_err <= harness["TOL"], f"KDA warp-tiled state error {state_err:.3e}"


@requires_gfx950
@pytest.mark.parametrize("max_work,tile,spec_id", _TUNED_TILES_KDA)
def test_every_kda_tuned_tile_is_correct(harness, max_work, tile, spec_id):
    """Every tile in the KDA table agrees with the independent reference."""
    num_warps, warp_threads_k, blocks_per_v_dim = tile
    spec = _kda(
        num_warps=num_warps,
        warp_threads_k=warp_threads_k,
        blocks_per_v_dim=blocks_per_v_dim,
    )
    out_err, state_err = harness["check"](spec, batch=16)

    assert out_err <= harness["TOL"], (
        f"KDA {spec_id} ({tile}, max_work={max_work}) " f"output error {out_err:.3e}"
    )
    assert state_err <= harness["TOL"], (
        f"KDA {spec_id} ({tile}, max_work={max_work}) " f"state error {state_err:.3e}"
    )


@requires_gfx950
@pytest.mark.parametrize(
    "batch,expected_spec_id,expected_tile",
    [
        (1, "kda_w128", (4, 16, 4)),
        (8, "kda_w512", (1, 16, 4)),
        (32, "kda_w_large", (2, 16, 1)),
    ],
)
def test_kda_dispatch_band_launches_selected_kernel(
    harness, batch, expected_spec_id, expected_tile
):
    """Exercise request → dispatch → selected tile → compile → launch → oracle."""
    from dispatch.gdn import GdnDecodeRequest, dispatch_gdn_decode
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig, no_fence

    result = dispatch_gdn_decode(
        GdnDecodeRequest(
            batch=batch,
            arch=ARCH,
            gate_kind="kda",
            num_k_heads=32,
            num_v_heads=32,
        )
    )
    got_tile = (
        result.spec.num_warps,
        result.spec.warp_threads_k,
        result.spec.blocks_per_v_dim,
    )
    assert result.candidate.spec_id == expected_spec_id
    assert got_tile == expected_tile

    artifact = compile_kernel(result.build(), arch=ARCH)
    launcher = KernelLauncher(
        hsaco=artifact.hsaco,
        kernel_name=artifact.kernel_name,
        signature=result.signature,
    )

    inp = harness["make_inputs"](result.spec, batch)
    ref_out, ref_state = harness["ref_fp32"](result.spec, inp)
    values, _ = harness["prepare"](result.spec, inp, batch)
    cfg = LaunchConfig(grid=result.grid, block=result.block, stream=0)
    with no_fence():
        launcher(values, config=cfg)
    torch.cuda.synchronize()

    written = inp["write_indices"].long()
    out_err = (values["out"].float() - ref_out).abs().max().item()
    state_err = (values["state"].float()[written] - ref_state).abs().max().item()
    assert max(out_err, state_err) <= harness["TOL"], (
        f"dispatch-driven KDA launch disagreed with the reference: "
        f"out={out_err:.3e} state={state_err:.3e}"
    )


@requires_gfx950
def test_kda_mha_shape_matches_reference(harness):
    """The shipping MHA shape: Hk == Hv == 32, kv_group 1.

    The GDN shapes are all GQA (Hv > Hk), so this is the first time the gather
    that maps a value head to its key head runs with no grouping at all.
    """
    out_err, state_err = harness["check"](_kda(num_k_heads=32, num_v_heads=32), 8)

    assert out_err <= harness["TOL"], f"KDA MHA output error {out_err:.3e}"
    assert state_err <= harness["TOL"], f"KDA MHA state error {state_err:.3e}"


@requires_gfx950
@pytest.mark.parametrize("batch", [1, 16])
def test_precomputed_decay_matches_reference(harness, batch):
    """fuse_gate=False: `a` carries log-decay, the kernel only exponentiates."""
    out_err, state_err = harness["check"](_kda(fuse_gate=False), batch)

    assert out_err <= harness["TOL"], f"raw-gate output error {out_err:.3e}"
    assert state_err <= harness["TOL"], f"raw-gate state error {state_err:.3e}"


@requires_gfx950
def test_precomputed_decay_agrees_with_fused_gate():
    """The two gate-input modes must agree given the same decay.

    This is what makes the later benchmark honest. The modes differ only in
    who computes the gate, so once they agree numerically, any timing gap
    between them is gate-activation cost and nothing else. Without this, an
    arm-1 number could be comparing two different computations.
    """
    import torch

    from builders.gfx950.gdn.gdn_decode import launcher_for, make_inputs, run

    fused = _kda()
    raw = _kda(fuse_gate=False)
    batch = 8

    inp = make_inputs(fused, batch)

    # Precompute exactly what the fused kernel's gate produces, in log domain.
    gx = inp["a"][:, 0].float() + inp["dt_bias"].float()
    inner = torch.exp(inp["A_log"].float())[None, :, None] * gx
    log_decay = fused.lower_bound * torch.sigmoid(inner)  # [B, HV, DK]

    raw_inp = dict(inp)
    raw_inp["a"] = log_decay[:, None].to(inp["a"].dtype).contiguous()
    raw_inp["state"] = inp["state"].clone()

    out_f, state_f = run(fused, inp, launcher_for(fused), batch)
    out_r, state_r = run(raw, raw_inp, launcher_for(raw), batch)

    # bf16 gate logits round differently on the two paths, so this is an
    # agreement check at input precision, not bit-equality.
    assert (out_f.float() - out_r.float()).abs().max().item() <= 2e-2
    assert (state_f.float() - state_r.float()).abs().max().item() <= 2e-2
