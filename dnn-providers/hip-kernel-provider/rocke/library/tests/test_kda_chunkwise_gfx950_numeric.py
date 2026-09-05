# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-GPU numeric lane for the chunkwise KDA family on gfx950.

Two references, deliberately independent of each other and of the kernels:

- The six per-chunk tiles are checked against a float64 torch oracle that
  builds them from the pairwise exponent *difference*, with no midpoint
  factoring -- so it never forms the overflowing ``Gamma_i / Gamma_j`` the
  kernel is designed around, and agreement is a check of that factoring rather
  than a restatement of it.
- Both full forward paths are checked against a token-serial float64 walk of
  the gated delta rule. That oracle is not chunked at all, so agreement tests
  the chunkwise factorization itself.

The gate range is swept to -5, the reference ``gate_lower_bound``: a 32-token chunk
accumulates up to 160 nats there, which is the regime that saturates the
factored exponents and the only one where the clamping actually matters.

Every test is marked ``gpu`` and skipped off a gfx950. Select with
``run_all.py --gpu`` (or ``pytest -m gpu``); the default CPU lane excludes it.
"""

from __future__ import annotations

import pytest


def _gpu_ready():
    """True only on a gfx950 box with ROCm torch.

    Gate on ``gcnArchName`` (the ISA target), NOT the marketing name.
    """
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    if not torch.cuda.is_available():
        return False
    arch = torch.cuda.get_device_properties(0).gcnArchName.lower()
    return "gfx950" in arch


requires_gfx950_gpu = pytest.mark.skipif(
    not _gpu_ready(), reason="needs a gfx950 GPU with ROCm torch"
)

pytestmark = [pytest.mark.gpu, requires_gfx950_gpu]

# -0.1 is the typical regime; -5.0 is the reference gate lower bound and is what
# saturates the factored exponents.
GATES = [-0.1, -0.5, -2.0, -5.0]
# bf16 operands and an fp32 accumulator against a float64 oracle. The chunkwise
# path also carries a triangular solve, so the tolerance is a bf16 tolerance
# with room for its conditioning, not a tight fp32 one.
TOL = 3e-2


@pytest.mark.parametrize("gate_low", GATES)
def test_prep_tiles_match_float64_oracle(gate_low):
    """All six per-chunk tiles, over enough chunks to cover every code path."""
    from builders.gfx950.kda import kda_chunk_prep as prep

    from kernels.gfx950.kda_chunkwise import KdaChunkPrepSpec

    spec = KdaChunkPrepSpec()
    assert prep.check(spec, 128, gate_low=gate_low, tol=2e-2, verbose=False)


@pytest.mark.parametrize("gate_low", GATES)
def test_split_path_matches_token_serial(gate_low):
    """rocke prep + rocke scan, against the token-serial recurrence."""
    from builders.gfx950.kda import kda_chunk_split as split

    from kernels.gfx950.kda_chunkwise import KdaChunkScanSpec

    worst = split.check(KdaChunkScanSpec(), 2, 4, 256, gate_low=gate_low, verbose=False)
    assert worst <= TOL, f"gate {gate_low}: worst rel {worst:.3e}"


@pytest.mark.parametrize("gate_low", GATES)
def test_fused_path_matches_token_serial(gate_low):
    """One kernel, tiles never leaving LDS, against the same oracle.

    The two paths run the same emitted scan body over the same tile math, so
    this is the check that routing the tiles through HBM instead of LDS did not
    change the arithmetic.
    """
    from builders.gfx950.kda import kda_chunk_fused as fused

    from kernels.gfx950.kda_chunkwise import KdaChunkFusedSpec

    worst = fused.check(
        KdaChunkFusedSpec(), 2, 4, 256, gate_low=gate_low, verbose=False
    )
    assert worst <= TOL, f"gate {gate_low}: worst rel {worst:.3e}"


@pytest.mark.parametrize("path", ["split", "fused"])
@pytest.mark.parametrize("gate_low", GATES)
def test_nonzero_initial_state_matches_token_serial(path, gate_low):
    """Both compositions must preserve a non-zero state carried into prefill."""
    if path == "split":
        from builders.gfx950.kda import kda_chunk_split as module
        from kernels.gfx950.kda_chunkwise import KdaChunkScanSpec

        spec = KdaChunkScanSpec(has_initial_state=True)
    else:
        from builders.gfx950.kda import kda_chunk_fused as module
        from kernels.gfx950.kda_chunkwise import KdaChunkFusedSpec

        spec = KdaChunkFusedSpec(has_initial_state=True)

    worst = module.check(
        spec, 2, 4, 256, gate_low=gate_low, with_h0=True, verbose=False
    )
    assert worst <= TOL, f"{path}, gate {gate_low}: initial-state worst rel {worst:.3e}"


@pytest.mark.parametrize("gate_low", GATES)
def test_subtiled_fused_path_matches_token_serial(gate_low):
    """The state-subtiled fused kernel, against the same oracle.

    Subtiling narrows the scan atom so each wave owns a shorter band of ``S^T``
    and twice as many waves cover the v extent, which halves each lane's
    loop-carried state. That moves every one of the five products off a
    single-atom output onto a tiled one, so this covers index arithmetic the
    chunk-wide atom never exercises.
    """
    from builders.gfx950.kda import kda_chunk_fused as fused

    from kernels.gfx950.kda_chunkwise import KdaChunkFusedSpec, KdaTileSpec

    spec = KdaChunkFusedSpec(tile=KdaTileSpec(block_size=512, scan_atom_m=16))
    worst = fused.check(spec, 2, 4, 256, gate_low=gate_low, verbose=False)
    assert worst <= TOL, f"gate {gate_low}: worst rel {worst:.3e}"


def test_subtiled_and_chunk_wide_fused_agree_to_one_ulp():
    """Subtiling must repartition the arithmetic without changing the answer.

    Not bit-exact, and it should not be: the narrower atom packs twice the K per
    instruction, so a contraction that took eight accumulation steps takes four,
    and fp32 summation is not associative. What must hold is that this is the
    *only* difference -- a last-bit effect on a small minority of elements. A
    real indexing error in the retiled products would show up here as a large
    difference on many elements, which no oracle tolerance would localize.
    """
    import torch

    from builders.gfx950.kda import kda_chunk_fused as fused

    from kernels.gfx950.kda_chunkwise import KdaChunkFusedSpec, KdaTileSpec

    B, H, T = 2, 4, 256
    wide = KdaChunkFusedSpec(tile=KdaTileSpec())
    sub = KdaChunkFusedSpec(tile=KdaTileSpec(block_size=512, scan_atom_m=16))
    q, k, v, g, beta = fused.make_inputs(B, H, T, wide.head_k, wide.head_v)
    o_w, ht_w = fused.launch_packed(wide, q, k, v, g, beta)
    o_s, ht_s = fused.launch_packed(sub, q, k, v, g, beta)
    torch.cuda.synchronize()

    # One bf16 ULP is 2^-8 relative; allow half that against the tile's peak
    # magnitude, which is far tighter than the 3e-2 oracle tolerance.
    for name, a, b in (("output", o_w, o_s), ("final state", ht_w, ht_s)):
        diff = (a.float() - b.float()).abs().max().item()
        scale = a.float().abs().max().item()
        frac = (a != b).float().mean().item()
        assert diff <= 2e-3 * scale, (
            f"{name} differs by {diff:.3e} ({diff / scale:.3e} of peak), "
            "more than rounding: the retiled indexing is wrong"
        )
        assert frac <= 0.10, (
            f"{name}: {frac:.1%} of elements differ; rounding alone should "
            "touch only a small minority"
        )


@pytest.mark.parametrize("tile_kw", [{}, {"block_size": 512, "scan_atom_m": 16}])
def test_input_prefetch_is_bitwise_identical(tile_kw):
    """Staging a chunk's inputs one chunk early must not change the arithmetic.

    The prefetch moves *when* g/k/q/beta reach LDS, never what lands there, so
    this is a bitwise check rather than a tolerance one.

    It is also the race detector. The tiles being written are the same ones the
    current chunk staged into, and the hand-off deliberately adds no barrier of
    its own -- it leans on the rendezvous the scan already has. A missed
    dependency means some chunk reads its successor's inputs, and the earlier
    attempt at this optimization showed that failure drifts in rather than
    appearing on the first launch: the buffers have to already hold a previous
    run's data. Hence the repeat.
    """
    import torch

    from builders.gfx950.kda import kda_chunk_fused as fused

    from kernels.gfx950.kda_chunkwise import KdaChunkFusedSpec, KdaTileSpec

    B, H, T = 2, 4, 256
    tile = KdaTileSpec(**tile_kw)
    # Both spelled out: the prefetch is the default, so leaving either implicit
    # would compare a spec against itself.
    base = KdaChunkFusedSpec(tile=tile, prefetch_inputs=False)
    pf = KdaChunkFusedSpec(tile=tile, prefetch_inputs=True)
    assert base.kernel_name() != pf.kernel_name()
    q, k, v, g, beta = fused.make_inputs(B, H, T, base.head_k, base.head_v)

    for it in range(3):
        o_b, ht_b = fused.launch_packed(base, q, k, v, g, beta)
        o_p, ht_p = fused.launch_packed(pf, q, k, v, g, beta)
        torch.cuda.synchronize()
        assert torch.equal(o_b, o_p), (
            f"launch {it}: prefetched output differs from the in-phase staging; "
            "a chunk read inputs the hand-off had not published"
        )
        assert torch.equal(ht_b, ht_p), f"launch {it}: final states diverge"


def test_fused_lds_overlay_is_bitwise_identical():
    """Explicit byte-pool views may change addresses, never values.

    This catches lifetime mistakes in the overlay itself. Kt is produced while
    the solve tiles are live and V survives into residual formation, so either
    one being placed in the reusable tile-only span quickly produces NaNs.
    """
    import torch

    from builders.gfx950.kda import kda_chunk_fused as fused

    from kernels.gfx950.kda_chunkwise import KdaChunkFusedSpec, KdaTileSpec

    B, H, T = 2, 4, 256
    tile = KdaTileSpec(block_size=512, scan_atom_m=16)
    typed = KdaChunkFusedSpec(tile=tile, prefetch_inputs=False, overlay_lds=False)
    overlay = KdaChunkFusedSpec(tile=tile, prefetch_inputs=False, overlay_lds=True)
    q, k, v, g, beta = fused.make_inputs(B, H, T, typed.head_k, typed.head_v)
    o_t, ht_t = fused.launch_packed(typed, q, k, v, g, beta)
    o_o, ht_o = fused.launch_packed(overlay, q, k, v, g, beta)
    torch.cuda.synchronize()

    assert torch.isfinite(o_o).all() and torch.isfinite(ht_o).all()
    assert torch.equal(o_t, o_o), "LDS overlay changed the fused output"
    assert torch.equal(ht_t, ht_o), "LDS overlay changed the final state"


def test_c32_tile_phase_16x16_panels_agree_to_one_ulp():
    """Inner panels repartition C x C work without changing the algorithm.

    The 16x16x32 atom takes four K steps where 32x32x16 takes eight, so fp32
    accumulation order changes and bitwise identity is not expected. Differences
    must remain last-bit effects rather than panel indexing errors.
    """
    import torch

    from builders.gfx950.kda import kda_chunk_fused as fused

    from kernels.gfx950.kda_chunkwise import KdaChunkFusedSpec, KdaTileSpec

    B, H, T = 2, 4, 256
    common = dict(block_size=512, scan_atom_m=16)
    atom32 = KdaChunkFusedSpec(tile=KdaTileSpec(**common, tile_atom_m=32))
    panels16 = KdaChunkFusedSpec(tile=KdaTileSpec(**common, tile_atom_m=16))
    q, k, v, g, beta = fused.make_inputs(B, H, T, atom32.head_k, atom32.head_v)
    o_32, ht_32 = fused.launch_packed(atom32, q, k, v, g, beta)
    o_16, ht_16 = fused.launch_packed(panels16, q, k, v, g, beta)
    torch.cuda.synchronize()

    for name, a, b in (("output", o_32, o_16), ("final state", ht_32, ht_16)):
        diff = (a.float() - b.float()).abs().max().item()
        scale = a.float().abs().max().item()
        frac = (a != b).float().mean().item()
        assert diff <= 2e-3 * scale, (
            f"{name} differs by {diff:.3e} ({diff / scale:.3e} of peak), "
            "more than atom accumulation order can explain"
        )
        assert frac <= 0.10, (
            f"{name}: {frac:.1%} of elements differ; panel indexing is suspect"
        )


def test_c16_fused_matches_token_serial_oracle():
    """Literal C=16 is a valid A/B even though it is not the throughput winner."""
    from builders.gfx950.kda import kda_chunk_fused as fused

    from kernels.gfx950.kda_chunkwise import KdaChunkFusedSpec, KdaTileSpec

    spec = KdaChunkFusedSpec(
        tile=KdaTileSpec(
            chunk=16,
            block_size=512,
            pad_cb=16,
            tile_atom_m=16,
            scan_atom_m=16,
        )
    )
    worst = fused.check(spec, 2, 4, 256, verbose=False)
    assert worst <= TOL, f"C16 fused worst relative error {worst:.3e}"


def test_split_and_fused_agree_bitwise():
    """The two paths share one emitted scan body, so they must agree exactly.

    Not merely within tolerance: same math, same order, same rounding. A
    difference here means the tile round trip through HBM lost something the
    LDS path kept, which no tolerance-based check would localize.
    """
    import torch

    from builders.gfx950.kda import kda_chunk_fused as fused
    from builders.gfx950.kda import kda_chunk_split as split

    from kernels.gfx950.kda_chunkwise import (
        KdaChunkFusedSpec,
        KdaChunkScanSpec,
        KdaTileSpec,
    )

    B, H, T = 2, 4, 256
    spec_s = KdaChunkScanSpec()
    q, k, v, g, beta = fused.make_inputs(B, H, T, spec_s.head_k, spec_s.head_v)
    o_s, ht_s = split.launch_packed(spec_s, q, k, v, g, beta)
    o_f, ht_f = fused.launch_packed(
        KdaChunkFusedSpec(tile=KdaTileSpec()), q, k, v, g, beta
    )
    torch.cuda.synchronize()
    assert torch.equal(o_s, o_f), "outputs diverge between the split and fused paths"
    assert torch.equal(ht_s, ht_f), "final states diverge"


def test_scan_rejects_a_spec_it_cannot_emit():
    """The admission rule is part of the contract, not a convenience.

    Checked on the GPU lane too because this is the guard that stops a spec
    whose lane mapping has no valid emission from reaching the launcher.
    """
    from kernels.gfx950.kda_chunkwise import KdaChunkScanSpec

    from builders.gfx950.kda import kda_chunk_split as split

    with pytest.raises(ValueError, match="unsupported spec"):
        split.make_launcher(KdaChunkScanSpec(head_v=64))


def _make_raw_inputs(B, H, T, DK, DV, gate_low=-0.5, seed=0, *, fp32_beta=False):
    import torch

    gen = torch.Generator(device="cuda").manual_seed(seed)
    kw = dict(device="cuda", generator=gen)
    q = torch.randn(B, T, H, DK, dtype=torch.bfloat16, **kw)
    k = torch.randn(B, T, H, DK, dtype=torch.bfloat16, **kw)
    v = (torch.randn(B, T, H, DV, dtype=torch.float32, **kw) * 0.2).bfloat16()
    g = (gate_low * torch.rand(B, T, H, DK, dtype=torch.float32, **kw)).bfloat16()
    beta_dtype = torch.float32 if fp32_beta else torch.bfloat16
    beta = torch.randn(B, T, H, dtype=beta_dtype, **kw)
    a_log = torch.randn(H, dtype=torch.float32, device="cuda", generator=gen) * 0.1
    dt_bias = (
        torch.randn(H * DK, dtype=torch.float32, device="cuda", generator=gen) * 0.01
    )
    return q, k, v, g, beta, a_log, dt_bias


@pytest.mark.parametrize("gate_low", GATES)
def test_raw_split_matches_aligned_oracle(gate_low):
    """Fused raw prep + token-major split scan vs the aligned float64 oracle."""
    import torch

    from builders.gfx950.kda import kda_chunk_split as split

    B, H, T = 1, 4, 256
    scan, _ = split.aligned_split_specs(1)
    q, k, v, g, beta, a_log, dt_bias = _make_raw_inputs(B, H, T, 128, 128, gate_low)
    o_got, ht_got = split.launch_raw(scan, q, k, v, g, beta, a_log, dt_bias)
    torch.cuda.synchronize()
    o_ref, s_ref = split.ref_aligned_raw(q, k, v, g, beta, a_log, dt_bias, 128**-0.5)
    o_ref = o_ref.permute(0, 2, 1, 3)
    worst = max(
        (o_got.double() - o_ref).abs().max().item()
        / (o_ref.abs().max().item() + 1e-30),
        (ht_got.double() - s_ref).abs().max().item()
        / (s_ref.abs().max().item() + 1e-30),
    )
    assert worst <= TOL, f"gate {gate_low}: worst rel {worst:.3e}"


@pytest.mark.parametrize("fp32_beta", [False, True])
def test_raw_split_accepts_strided_beta(fp32_beta):
    """The framework projection view reaches the raw prep kernel without a copy."""
    import torch

    from builders.gfx950.kda import kda_chunk_split as split

    B, H, T = 1, 4, 256
    scan, _ = split.aligned_split_specs(1)
    q, k, v, g, beta, a_log, dt_bias = _make_raw_inputs(
        B, H, T, 128, 128, fp32_beta=fp32_beta
    )
    beta_storage = torch.empty(B, T, 2 * H, dtype=beta.dtype, device="cuda")
    beta_storage[..., ::2].copy_(beta)
    beta_view = beta_storage[..., ::2]
    assert not beta_view.is_contiguous()

    o_got, ht_got = split.launch_raw(scan, q, k, v, g, beta_view, a_log, dt_bias)
    torch.cuda.synchronize()
    o_ref, s_ref = split.ref_aligned_raw(
        q, k, v, g, beta_view, a_log, dt_bias, 128**-0.5
    )
    o_ref = o_ref.permute(0, 2, 1, 3)

    assert torch.allclose(o_got.float(), o_ref.float(), rtol=0, atol=3e-2)
    assert torch.allclose(ht_got.float(), s_ref.float(), rtol=0, atol=3e-2)


@pytest.mark.parametrize(
    "value_splits,block,scan_atom_m",
    [(1, 256, 0), (2, 128, 0), (4, 64, 0), (8, 64, 16)],
)
def test_value_splits_agree_with_vs1(value_splits, block, scan_atom_m):
    """Every legal value_splits geometry must match the unsplit scan."""
    import torch

    from builders.gfx950.kda import kda_chunk_split as split

    from kernels.gfx950.kda_chunkwise import KdaChunkScanSpec, KdaTileSpec

    B, H, T = 1, 4, 256
    q, k, v, g, beta, a_log, dt_bias = _make_raw_inputs(B, H, T, 128, 128)
    base, _ = split.aligned_split_specs(1)
    o0, ht0 = split.launch_raw(base, q, k, v, g, beta, a_log, dt_bias)
    tile = KdaTileSpec(chunk=32, block_size=block, scan_atom_m=scan_atom_m)
    spec = KdaChunkScanSpec(tile=tile, value_splits=value_splits, token_major_io=True)
    o1, ht1 = split.launch_raw(spec, q, k, v, g, beta, a_log, dt_bias)
    torch.cuda.synchronize()
    assert torch.allclose(o0.float(), o1.float(), rtol=0, atol=3e-2)
    assert torch.allclose(ht0.float(), ht1.float(), rtol=0, atol=3e-2)


def test_dispatched_specs_match_the_token_serial_oracle():
    """What the dispatcher selects must be what the oracle validates.

    The registry derives its specs from a request instead of reusing the
    builders' defaults, so this is what stops the two from drifting: a
    selection that compiles but computes something else passes every CPU
    wiring test there is.
    """
    from builders.gfx950.kda import kda_chunk_fused as fused
    from builders.gfx950.kda import kda_chunk_split as split

    from dispatch.kda import KdaRequest, dispatch_kda

    B, H, T = 2, 4, 256

    def request(**kw):
        return KdaRequest(batch=B, num_heads=H, seqlen=T, arch="gfx950", **kw)

    fused_spec = dispatch_kda(request()).spec
    assert fused.check(fused_spec, B, H, T, verbose=False) <= TOL

    h0_spec = dispatch_kda(request(has_initial_state=True)).spec
    assert fused.check(h0_spec, B, H, T, with_h0=True, verbose=False) <= TOL

    scan_spec = dispatch_kda(request(algorithm="chunk_scan")).spec
    prep_spec = dispatch_kda(request(algorithm="chunk_prep")).spec
    # The halves are dispatched independently, so their tile layouts have to
    # agree or the scan stages something the tile builder never wrote.
    assert split.prep_spec_of(scan_spec) == prep_spec
    assert split.check(scan_spec, B, H, T, verbose=False) <= TOL
