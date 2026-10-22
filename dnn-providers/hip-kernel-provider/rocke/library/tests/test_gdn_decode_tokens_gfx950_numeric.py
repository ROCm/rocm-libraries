# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-device numerics of the multi-token GDN/KDA decode step (``num_tokens``).

Each case runs ``T`` chained tokens per sequence and checks, against the
whole-tensor fp32 reference (token ``t`` a single-token step from token
``t - 1``'s state): every token's output, every token's state snapshot
(``write_indices[b, t]``), and every untouched pool page bit for bit. Each
token's output is bounded both absolutely (``TOL``) and per (sequence, token,
head) row relative to the reference's scale (``OUT_RTOL``, twice the bf16
output rounding), so a percent-level output error fails. The f32-state cases
also bound the snapshots relative to the reference at the f32
summation-order level. The fused gated RMSNorm (``fuse_out_norm``) chains too
and is checked the same way. Geometries: the default heads and the Hk 16 /
Hv 48 grouped heads, at batches 1, 3 and 4; tiles and knobs as listed.

Needs a real gfx950 and ROCm torch; marked ``gpu`` and skipped elsewhere.
"""

from __future__ import annotations

import dataclasses as dc

import pytest

ARCH = "gfx950"

torch = pytest.importorskip("torch", reason="ROCm torch required")

pytestmark = pytest.mark.gpu


def _device_is_gfx950() -> bool:
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

TILES = [(2, 16, 8), (2, 16, 16), (1, 16, 8), (4, 8, 1), (1, 4, 2), (8, 16, 1)]
KNOBS = [
    dict(),
    dict(dpp_reduce=True),
    dict(interleave_cols=True),
    dict(stream_rows=True),
    dict(xcd_remap=True),
    dict(state_load_hint="streaming", state_store_hint="streaming", waves_per_eu=4),
]


def _run(spec, batch, seed=0, skip_last=False, skip_slot=None):
    from builders.gfx950.gdn import gdn_decode as host

    inp = host.make_inputs(spec, batch, seed=seed)
    if skip_last:
        skip_slot = -1
    if skip_slot is not None:
        # A negative snapshot slot skips that token's state store, nothing else.
        inp["write_indices"][:, skip_slot] = -1
    ref_out, ref_state = host.ref_fp32(spec, inp)
    values = host.run_values(spec, inp, host.launcher_for(spec), batch)
    out_err = host.out_error(spec, values["out"], ref_out)
    out_rel = host.out_rel_error(values["out"], ref_out)
    assert out_rel <= host.OUT_RTOL, (spec.kernel_name(), out_rel)
    w = inp["write_indices"].long()
    keep = w >= 0
    got = values["state"].float()[w.clamp(min=0)]
    st_err = (got[keep] - ref_state[keep]).abs().max().item()
    st_rel = st_err / max(ref_state.abs().max().item(), 1e-3)
    untouched = torch.ones(inp["state"].shape[0], dtype=torch.bool, device=w.device)
    untouched[w[keep]] = False
    spill = bool((values["state"][untouched] != inp["state"][untouched]).any())
    return out_err, st_err, st_rel, spill, host.TOL


@requires_gfx950
@pytest.mark.parametrize("state_dtype", ["f32", "bf16"])
@pytest.mark.parametrize("gate_kind", ["gdn", "kda"])
@pytest.mark.parametrize("heads", [(16, 48), (16, 32)])
@pytest.mark.parametrize("T", [2, 4])
@pytest.mark.parametrize("batch", [1, 3, 4])
def test_tokens_match_reference(state_dtype, gate_kind, heads, T, batch):
    from kernels.gfx950.gdn_decode import GdnDecodeSpec, is_valid_spec

    if gate_kind == "kda" and heads == (16, 48):
        heads = (16, 16)
    for tile in TILES:
        for knobs in KNOBS:
            spec = GdnDecodeSpec(
                num_k_heads=heads[0],
                num_v_heads=heads[1],
                state_dtype=state_dtype,
                gate_kind=gate_kind,
                num_tokens=T,
                num_warps=tile[0],
                warp_threads_k=tile[1],
                blocks_per_v_dim=tile[2],
                **knobs,
            )
            if not is_valid_spec(spec, arch=ARCH)[0]:
                continue
            out_err, st_err, st_rel, spill, tol = _run(spec, batch)
            name = spec.kernel_name()
            assert out_err <= tol, (name, out_err)
            assert st_err <= tol, (name, st_err)
            if state_dtype == "f32":
                assert st_rel <= 1e-4, (name, st_rel)
            assert not spill, name


@requires_gfx950
@pytest.mark.parametrize("T", [2, 3, 4])
@pytest.mark.parametrize("slot", [-1, 0, 1])
def test_negative_snapshot_slot_skips_only_that_store(T, slot):
    """Any token's slot, token 0's included: the chain still runs and every
    output and every other snapshot is still written."""
    from kernels.gfx950.gdn_decode import GdnDecodeSpec

    spec = GdnDecodeSpec(
        num_k_heads=16, num_v_heads=48, state_dtype="f32", num_tokens=T
    )
    out_err, st_err, st_rel, spill, tol = _run(spec, 3, skip_slot=slot)
    assert out_err <= tol and st_rel <= 1e-4 and not spill


NORM_TILES = [(1, 16, 1), (2, 16, 1), (4, 16, 1), (8, 16, 1), (16, 16, 1), (4, 8, 1)]
NORM_KNOBS = [
    dict(),
    dict(out_lds=True),
    dict(dpp_reduce=True, stream_rows=True, interleave_cols=True),
    dict(out_lds=True, xcd_remap=True, waves_per_eu=8),
]


@requires_gfx950
@pytest.mark.parametrize("gate_kind", ["gdn", "kda"])
@pytest.mark.parametrize("act", ["auto", "sigmoid"])
@pytest.mark.parametrize("T", [2, 4])
@pytest.mark.parametrize("batch", [1, 4])
def test_out_norm_tokens_match_reference(gate_kind, act, T, batch):
    """fuse_out_norm chained over T tokens: each token's gated RMSNorm on its
    own outputs and its own out_gate row (row b * T + t)."""
    from kernels.gfx950.gdn_decode import GdnDecodeSpec, is_valid_spec

    hv = 48 if gate_kind == "gdn" else 16
    ran = 0
    for tile in NORM_TILES:
        for knobs in NORM_KNOBS:
            spec = GdnDecodeSpec(
                num_k_heads=16,
                num_v_heads=hv,
                state_dtype="f32",
                gate_kind=gate_kind,
                num_tokens=T,
                fuse_out_norm=True,
                norm_gate_act=act,
                num_warps=tile[0],
                warp_threads_k=tile[1],
                blocks_per_v_dim=tile[2],
                **knobs,
            )
            assert is_valid_spec(spec, arch=ARCH) == (True, "")
            out_err, st_err, st_rel, spill, tol = _run(spec, batch)
            name = spec.kernel_name()
            assert out_err <= tol, (name, out_err)
            assert st_rel <= 1e-4, (name, st_rel)
            assert not spill, name
            ran += 1
    assert ran == len(NORM_TILES) * len(NORM_KNOBS)  # every pair is legal


@requires_gfx950
def test_t1_slice_of_a_chain_equals_single_token_kernel():
    """Token 0 of a chain is exactly the single-token step (same ops on the
    same inputs), so its output and snapshot match the T = 1 kernel bit for
    bit."""
    from builders.gfx950.gdn import gdn_decode as host
    from kernels.gfx950.gdn_decode import GdnDecodeSpec

    many = GdnDecodeSpec(
        num_k_heads=16, num_v_heads=48, state_dtype="f32", num_tokens=4
    )
    one = dc.replace(many, num_tokens=1)
    inp = host.make_inputs(many, 2)
    v_many = host.run_values(many, inp, host.launcher_for(many), 2)
    single = dict(inp)
    for name in ("query", "key", "value", "a", "b"):
        single[name] = inp[name][:, :1].contiguous()
    single["write_indices"] = inp["write_indices"][:, 0].contiguous()
    v_one = host.run_values(one, single, host.launcher_for(one), 2)
    assert torch.equal(v_many["out"][:, 0], v_one["out"][:, 0])
    w0 = inp["write_indices"][:, 0].long()
    assert torch.equal(v_many["state"][w0], v_one["state"][w0])
