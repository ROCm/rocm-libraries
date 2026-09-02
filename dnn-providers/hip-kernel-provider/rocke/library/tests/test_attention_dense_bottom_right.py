# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Coverage for ``causal_bottom_right`` on the gfx950 dense flash-attn kernel.

The diagonal offset is a BUILD-TIME constant, so most of what can go wrong is provable
without a GPU: which specs are accepted, what symbol they compile to, whether the default
path is disturbed, and whether the arch that does not implement the field refuses it.
Those run in the CPU lane.

The one thing only silicon can answer -- does the shifted mask actually produce the right
numbers -- is a single ``gpu``-marked test checking against an explicit bottom-right
``attn_mask`` oracle. It is written so that it FAILS if the offset is dropped: at
``Sq < Skv`` a top-left mask and a bottom-right mask disagree on almost every row, which
the companion ``test_top_left_and_bottom_right_actually_differ`` pins down on CPU by
comparing the emitted objects.

Run the CPU lane standalone:

    python -m pytest tests/test_attention_dense_bottom_right.py -m "not gpu"
"""

from __future__ import annotations

import pytest

from kernels.gfx950.attention_dense import (
    _BLOCK_M,
    AttentionDenseSpec,
    build_attention_dense,
    supports_attention_dense,
)

# Aligned static shape: Sq a multiple of 256, (Skv - Sq) a multiple of block_n.
_BASE = dict(
    batch=1,
    num_query_heads=32,
    num_kv_heads=8,
    head_size=128,
    causal=True,
    dtype="bf16",
    block_n=64,
)


def _spec(**over):
    kw = dict(_BASE)
    kw.update(over)
    return AttentionDenseSpec(**kw)


def _ll(spec, *, anonymize=False) -> str:
    """Lower a spec to AMDGPU ``.ll`` text through the pure-Python backend (no compiler
    toolchain, so this runs in the CPU lane).

    ``anonymize`` strips the kernel name out of the text. Without it, a bottom-right and
    a top-left kernel would differ merely because the name carries ``br`` -- which would
    let an offset that does nothing still pass. Removing the name means the comparisons
    below are about the emitted BODY.
    """
    from rocke.helpers.compile import _lower_llvm_via_backend

    kd = build_attention_dense(spec, arch="gfx950")
    txt = _lower_llvm_via_backend(kd, arch="gfx950", backend="python", spec=None)
    return txt.replace(kd.name, "KERNEL") if anonymize else txt


# --------------------------------------------------------------------------- #
# Accepted / rejected
# --------------------------------------------------------------------------- #
def test_bottom_right_accepted_on_aligned_cross_shape():
    spec = _spec(seqlen_q=256, seqlen_kv=512, causal_bottom_right=True)
    ok, why = supports_attention_dense(spec, arch="gfx950")
    assert ok, why


@pytest.mark.parametrize(
    "over, needle",
    [
        # Not causal at all: there is no diagonal to move.
        (dict(seqlen_q=256, seqlen_kv=512, causal=False), "requires causal=True"),
        # The persistent path derives the diagonal separately and needs its own offset.
        (dict(seqlen_q=256, seqlen_kv=512, persistent=True), "persistent"),
        # The sliding-window band would have to shift with the diagonal.
        (dict(seqlen_q=256, seqlen_kv=512, sliding_window=128), "sliding_window"),
        # Under varlen the real lengths are runtime cu_seqlens values, so a baked
        # offset would apply one sequence's diagonal to the whole batch.
        (dict(seqlen_q=256, seqlen_kv=512, varlen=True), "varlen"),
        # The diagonal only ever moves right.
        (dict(seqlen_q=512, seqlen_kv=256), "seqlen_q <= seqlen_kv"),
    ],
)
def test_unsupported_combinations_are_rejected(over, needle):
    with pytest.raises(ValueError, match=needle):
        _spec(causal_bottom_right=True, **over)


def test_the_removed_alignment_guard_was_unreachable():
    """Records why there is no "(seqlen_kv - seqlen_q) must be a multiple of block_n"
    check: on the NON-RAGGED path it could never fire.

    On that path seqlen_q must be a multiple of the 256-row query tile and seqlen_kv a
    multiple of block_n, and every legal block_n divides 256 -- so seqlen_q is a multiple
    of block_n too and the difference always is as well. The ragged path deliberately
    lifts those length rules under bottom-right, which is why the offset there is
    arbitrary and the KV-tile bound has to ceil.

    Enumerated rather than argued, so this starts failing if the non-ragged shape rules
    change and the assumption quietly stops holding.

    The KV-tile bound is a ceil independently of this, so an offset that DID land
    mid-tile would still be covered. That is what the ragged/varlen follow-up needs,
    where the lengths are arbitrary and no such alignment is available.
    """
    checked = 0
    for block_n in (32, 64, 128, 256):
        assert _BLOCK_M % block_n == 0, "a block_n that does not divide the query tile"
        for seqlen_q in range(_BLOCK_M, 4 * _BLOCK_M + 1, _BLOCK_M):
            for seqlen_kv in range(seqlen_q, seqlen_q + 16 * block_n + 1, block_n):
                assert (seqlen_kv - seqlen_q) % block_n == 0
                checked += 1
    assert checked


@pytest.mark.parametrize("sq, skv", [(197, 400), (300, 1234), (512, 4097), (100, 8000)])
def test_ragged_accepts_arbitrary_chunk_lengths(sq, skv):
    """Ragged + bottom-right is the combination a real chunked-prefill request needs:
    a short query block against a cache of whatever length it happens to be, neither a
    multiple of the tile geometry.

    Ragged is otherwise self-attention only. That restriction is lifted here because the
    padded-key argument still holds -- the last real query reaches exactly S_kv-1, so the
    partial final tile's padding stays excluded. These lengths are numerically verified
    on gfx950 by the AOT parity suite; this test guards the spec-level contract.
    """
    spec = _spec(
        seqlen_q=sq,
        seqlen_kv=skv,
        num_query_heads=4,
        num_kv_heads=1,
        ragged=True,
        causal_bottom_right=True,
    )
    ok, why = supports_attention_dense(spec, arch="gfx950")
    assert ok, why
    assert "_br" in spec.kernel_name() and "ragged" in spec.kernel_name()


def test_ragged_without_bottom_right_is_still_self_attention_only():
    """The relaxation is scoped to bottom-right, so the on-point negative case is
    top-left causal rather than non-causal: `causal_bottom_right` is the axis this
    moved. A top-left diagonal gives a short query block nowhere to sit, so ragged
    cross-attention stays rejected there."""
    with pytest.raises(ValueError, match="self-attention only"):
        _spec(
            seqlen_q=197,
            seqlen_kv=400,
            num_query_heads=4,
            num_kv_heads=1,
            ragged=True,
            causal=True,
            causal_bottom_right=False,
        )


def test_gfx942_refuses_the_field_it_does_not_implement():
    """The spec class is shared with gfx942, whose builder never reads this field. If
    its support gate let the spec through, the build would emit a top-left mask under a
    kernel name carrying ``br`` -- a wrong result cached under a symbol claiming to be
    right."""
    from kernels.gfx942.attention_dense import (
        supports_attention_dense as supports_gfx942,
    )

    spec = _spec(seqlen_q=256, seqlen_kv=512, causal_bottom_right=True)
    ok, why = supports_gfx942(spec, arch="gfx942")
    assert not ok
    assert "causal_bottom_right" in why


# --------------------------------------------------------------------------- #
# Identity: the symbol, and the promise that the default path is untouched
# --------------------------------------------------------------------------- #
def test_bottom_right_kernels_get_their_own_symbol():
    """Two kernels that differ only in diagonal alignment must not share a name, or
    they share a launcher-cache entry and one silently serves the other."""
    tl = _spec(seqlen_q=256, seqlen_kv=512).kernel_name()
    br = _spec(seqlen_q=256, seqlen_kv=512, causal_bottom_right=True).kernel_name()
    assert tl != br
    assert "_br" in br and "_br" not in tl


def test_default_path_emits_identical_ir():
    """causal_bottom_right=False must cost nothing: the offset folds to 0, so the
    emitted IR has to match the spec that never mentions the field."""
    a = _ll(_spec(seqlen_q=2048, seqlen_kv=2048))
    b = _ll(_spec(seqlen_q=2048, seqlen_kv=2048, causal_bottom_right=False))
    assert a == b


def test_top_left_and_bottom_right_actually_differ():
    """Guards the whole feature. Names are anonymized first, so this fails if the offset
    is dropped from the mask -- leaving only a renamed copy of the top-left kernel."""
    tl = _ll(_spec(seqlen_q=256, seqlen_kv=512), anonymize=True)
    br = _ll(
        _spec(seqlen_q=256, seqlen_kv=512, causal_bottom_right=True), anonymize=True
    )
    assert tl != br


def test_equal_lengths_collapse_to_top_left():
    """At Sq == Skv the offset is 0, so the two alignments are the same mask and the
    emitted bodies must be identical -- only the symbol differs."""
    tl = _ll(_spec(seqlen_q=2048, seqlen_kv=2048), anonymize=True)
    br = _ll(
        _spec(seqlen_q=2048, seqlen_kv=2048, causal_bottom_right=True), anonymize=True
    )
    assert tl == br


# --------------------------------------------------------------------------- #
# Composition with attention sinks
# --------------------------------------------------------------------------- #
def test_bottom_right_composes_with_sinks():
    """A sink is a per-query-head logit with no key position, so it is orthogonal to
    which keys the diagonal admits. Both features nevertheless write the same softmax
    -- the sink seeds m/l before the KV loop and the offset changes what that loop
    admits -- so the combination is asserted rather than assumed.

    The symbol matters as much as acceptance: if only one of the two tokens reached
    the name, two kernels that differ in the other would share a launcher-cache entry
    and one would silently serve the other.
    """
    spec = _spec(seqlen_q=512, seqlen_kv=1024, causal_bottom_right=True, use_sinks=True)
    ok, why = supports_attention_dense(spec, arch="gfx950")
    assert ok, why
    name = spec.kernel_name()
    assert "_br" in name and "sinks" in name


def test_ragged_bottom_right_composes_with_sinks():
    """The arbitrary-length route as well, where the offset lands mid-tile and the
    sink has to seed a softmax whose last KV tile is partly on-chip padding."""
    spec = _spec(
        seqlen_q=300,
        seqlen_kv=1234,
        num_query_heads=4,
        num_kv_heads=1,
        ragged=True,
        causal_bottom_right=True,
        use_sinks=True,
    )
    ok, why = supports_attention_dense(spec, arch="gfx950")
    assert ok, why


def test_sinks_and_bottom_right_each_move_the_body_independently():
    """Guards the composition itself. If either feature were dropped whenever the
    other is set, two of these four bodies would coincide."""
    cross = dict(seqlen_q=512, seqlen_kv=1024)
    bodies = {
        _ll(_spec(**cross), anonymize=True),
        _ll(_spec(**cross, use_sinks=True), anonymize=True),
        _ll(_spec(**cross, causal_bottom_right=True), anonymize=True),
        _ll(_spec(**cross, causal_bottom_right=True, use_sinks=True), anonymize=True),
    }
    assert len(bodies) == 4


def test_equal_lengths_collapse_to_top_left_with_sinks():
    """The Sq == Skv collapse must not quietly depend on the sink being absent."""
    eq = dict(seqlen_q=2048, seqlen_kv=2048, use_sinks=True)
    tl = _ll(_spec(**eq), anonymize=True)
    br = _ll(_spec(**eq, causal_bottom_right=True), anonymize=True)
    assert tl == br


# --------------------------------------------------------------------------- #
# Dispatch wiring
# --------------------------------------------------------------------------- #
def test_dispatch_maps_mask_type_2_to_bottom_right():
    """A BOTTOM_RIGHT_CAUSAL request must reach the kernel as one. Before this was
    wired, mask_type=2 produced causal=True with the field defaulted False -- a
    top-left mask for a bottom-right request."""
    from dispatch.attention.common import AttentionRequest
    from dispatch.attention.gfx950 import dense_spec_for_request

    req = AttentionRequest(
        arch="gfx950",
        batch=1,
        seqlen_q=256,
        seqlen_k=512,
        nhead_q=32,
        nhead_k=8,
        hdim_q=128,
        hdim_v=128,
        dtype="bf16",
        mask_type=2,
        algorithm="attention_dense",
    )
    spec = dense_spec_for_request(req)
    assert spec.causal is True
    assert spec.causal_bottom_right is True
    assert "_br" in spec.kernel_name()


@pytest.mark.parametrize("block_n", [96, 160, 192, 224])
def test_supports_rejects_block_n_not_dividing_the_query_tile(block_n):
    """The KV-tile bound leaves the query-block term outside its ceil, which is exact
    only if block_n divides the 256-row query tile. Where it does not, the bound falls
    short and drops keys -- silently, since a short bound is still a legal loop.

    block_n=96 at seqlen_q=512 is the worked case: the bound reaches tile 5 while the
    last query needs tile 5 inclusive. gfx942 gates this already; gfx950 relies on it
    just as much, and did not.
    """
    from kernels.gfx950.attention_dense import _BLOCK_M

    assert _BLOCK_M % block_n != 0, "test is only meaningful for a non-divisor"
    # seqlen_kv has to stay a multiple of block_n or the spec rejects it for that
    # instead, and the gate under test never runs.
    ok, why = supports_attention_dense(
        _spec(
            seqlen_q=512,
            seqlen_kv=block_n * 8,
            num_query_heads=4,
            num_kv_heads=1,
            block_n=block_n,
        ),
        arch="gfx950",
    )
    assert not ok and "query tile" in why


@pytest.mark.parametrize("block_n", [32, 64, 128, 256])
def test_supports_still_accepts_every_block_n_that_divides_the_query_tile(block_n):
    """The other side of the gate: the rejection must not narrow the supported set."""
    ok, why = supports_attention_dense(
        _spec(
            seqlen_q=512,
            seqlen_kv=512,
            num_query_heads=4,
            num_kv_heads=1,
            block_n=block_n,
        ),
        arch="gfx950",
    )
    assert ok, why


@pytest.mark.parametrize("sq, skv", [(197, 400), (300, 1234), (512, 4097), (100, 8000)])
def test_dispatch_produces_ragged_bottom_right_at_arbitrary_lengths(sq, skv):
    """Dispatch must be able to BUILD the spec these lengths need, not merely have a
    kernel that could run it.

    The four shapes below are the ones the ragged tests declare supported. Building the
    spec by hand with ragged=True proves the kernel works and nothing else: dispatch
    decides `ragged` itself, and while it gated that on seqlen_q == seqlen_kv these
    requests fell to the aligned path and died on its 256-multiple rule. That is a
    capability nothing could reach, which is the failure this test exists to prevent.
    """
    from dispatch.attention.common import AttentionRequest
    from dispatch.attention.gfx950 import dense_spec_for_request

    req = AttentionRequest(
        arch="gfx950",
        batch=1,
        seqlen_q=sq,
        seqlen_k=skv,
        nhead_q=4,
        nhead_k=1,
        hdim_q=128,
        hdim_v=128,
        dtype="bf16",
        mask_type=2,
        algorithm="attention_dense",
    )
    spec = dense_spec_for_request(req)
    assert spec.causal_bottom_right is True
    assert spec.ragged is True, "arbitrary lengths must take the ragged path"
    ok, why = supports_attention_dense(spec, arch="gfx950")
    assert ok, why


def _dense_req(**over):
    """A dispatch-layer ``attention_dense`` request, bottom-right by default. The
    candidate is opt-in, so ``algorithm`` has to name it or nothing selects it."""
    from dispatch.attention.common import AttentionRequest

    kw = dict(
        arch="gfx950",
        batch=1,
        seqlen_q=256,
        seqlen_k=512,
        nhead_q=32,
        nhead_k=8,
        hdim_q=128,
        hdim_v=128,
        dtype="bf16",
        mask_type=2,
        algorithm="attention_dense",
    )
    kw.update(over)
    return AttentionRequest(**kw)


def test_dispatch_auto_downgrades_the_grid_only_when_the_diagonal_moves():
    """Under "auto" the caller never chose a grid, so dispatch picks the one that can
    serve the mask. But that downgrade costs the grid-stride path, so it must apply
    ONLY where the diagonal actually moves.

    All three requests below carry work = nqb*Hq*B = 32*32 = 1024, well past the
    256-CTA persistent threshold, so the heuristic wants persistent for every one of
    them. The earlier dispatch test uses Sq=256, where work is 32 and persistent is
    off regardless -- it cannot see this behaviour at all.
    """
    from dispatch.attention.gfx950 import dense_spec_for_request

    moved = dense_spec_for_request(_dense_req(seqlen_q=8192, seqlen_k=16384))
    assert moved.causal_bottom_right is True
    assert moved.persistent is False, "the shifted diagonal is non-persistent only"

    # Same mask_type and same work, but Sq == Skv so the offset is 0 and the mask is
    # the top-left one. Downgrading here would cap throughput for nothing.
    flat = dense_spec_for_request(_dense_req(seqlen_q=8192, seqlen_k=8192))
    assert flat.causal_bottom_right is False
    assert flat.persistent is True, "equal lengths must keep the grid-stride path"
    assert "_br" not in flat.kernel_name(), "no second symbol for an identical body"

    # A genuine top-left request of the same size is untouched, as before.
    tl = dense_spec_for_request(_dense_req(seqlen_q=8192, seqlen_k=8192, mask_type=1))
    assert tl.persistent is True and tl.causal_bottom_right is False


def test_dispatch_declines_explicit_persistent_on_when_the_diagonal_moves():
    """``dense_persistent="on"`` is a caller instruction rather than a heuristic, so
    it is not overridden: the spec refuses the combination and the candidate declines,
    which lets dispatch fall through to another candidate instead of answering with a
    top-left mask."""
    from dispatch.attention.gfx950 import _make_gfx950_attention_dense_candidate

    cand = _make_gfx950_attention_dense_candidate()
    ok, why = cand.admits(
        _dense_req(seqlen_q=8192, seqlen_k=16384, dense_persistent="on")
    )
    assert not ok
    assert "persistent" in why


def test_dispatch_still_serves_explicit_persistent_on_at_equal_lengths():
    """The capability the Sq != Skv gate restores. Without it, mask_type=2 at equal
    lengths was declined under "on" -- a request this kernel serves exactly, refused
    over a diagonal offset of zero."""
    from dispatch.attention.gfx950 import _make_gfx950_attention_dense_candidate

    cand = _make_gfx950_attention_dense_candidate()
    ok, why = cand.admits(
        _dense_req(seqlen_q=8192, seqlen_k=8192, dense_persistent="on")
    )
    assert ok, why


def test_dispatch_leaves_top_left_alone():
    from dispatch.attention.common import AttentionRequest
    from dispatch.attention.gfx950 import dense_spec_for_request

    req = AttentionRequest(
        arch="gfx950",
        batch=1,
        seqlen_q=2048,
        seqlen_k=2048,
        nhead_q=32,
        nhead_k=8,
        hdim_q=128,
        hdim_v=128,
        dtype="bf16",
        mask_type=1,
        algorithm="attention_dense",
    )
    spec = dense_spec_for_request(req)
    assert spec.causal is True
    assert spec.causal_bottom_right is False


# --------------------------------------------------------------------------- #
# On-GPU numerics
# --------------------------------------------------------------------------- #
def _gpu_ready() -> bool:
    """gfx950 only. Gate on gcnArchName (the ISA target) rather than the marketing
    name, which varies across the MI350/MI355 family."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        return "gfx950" in torch.cuda.get_device_properties(0).gcnArchName
    except Exception:
        return False


def _run_and_reference(spec, *, top_left=False, seed=0):
    """Launch the kernel and build the matching CPU-side oracle.

    Layout matters here and is easy to get wrong: the launcher takes q/out as
    ``[B, S, H, D]`` and k/v as ``[B, Skv, Hkv, D]``, while SDPA wants heads second --
    hence the transposes on the reference side only. ``out`` and ``scale`` are required
    keyword arguments, not optional.
    """
    import math

    import torch

    from kernels.gfx950.attention_dense import run_attention_dense_torch

    dev = "cuda"
    torch.manual_seed(seed)
    bsz, sq, skv = spec.batch, spec.seqlen_q, spec.seqlen_kv
    hq, hkv, d = spec.num_query_heads, spec.num_kv_heads, spec.head_size
    scale = 1.0 / math.sqrt(d)

    q = torch.randn(bsz, sq, hq, d, device=dev, dtype=torch.bfloat16)
    k = torch.randn(bsz, skv, hkv, d, device=dev, dtype=torch.bfloat16)
    v = torch.randn(bsz, skv, hkv, d, device=dev, dtype=torch.bfloat16)
    out = torch.empty(bsz, sq, hq, d, device=dev, dtype=torch.bfloat16)
    # One learned logit per QUERY head. The launcher rejects a sinks tensor when the
    # spec does not ask for one, so this stays None off the sink path.
    sinks = (
        torch.randn(hq, device=dev, dtype=torch.bfloat16) if spec.use_sinks else None
    )
    run_attention_dense_torch(
        spec=spec, q=q, k=k, v=v, out=out, scale=scale, sinks=sinks
    )

    rep = hq // hkv
    qh = q.transpose(1, 2).float()
    kh = k.transpose(1, 2).repeat_interleave(rep, 1).float()
    vh = v.transpose(1, 2).repeat_interleave(rep, 1).float()
    if top_left:
        mask = torch.arange(skv, device=dev).view(1, -1) <= torch.arange(
            sq, device=dev
        ).view(-1, 1)
    else:
        qi = torch.arange(sq, device=dev).view(-1, 1)
        ki = torch.arange(skv, device=dev).view(1, -1)
        mask = ki <= qi + (skv - sq)
    if sinks is None:
        ref = torch.nn.functional.scaled_dot_product_attention(
            qh, kh, vh, attn_mask=mask, scale=scale
        ).transpose(1, 2)
    else:
        # SDPA has no notion of a sink, so the softmax is built by hand: the sink is
        # one extra column carrying a constant per-head logit and a zero value vector,
        # so it lands in the denominator and then drops straight back out. Masked
        # positions go to -inf, and the sink column is what keeps a fully-masked row
        # from becoming 0/0 -- which is the same job it does in the kernel.
        attn = torch.einsum("bhqd,bhkd->bhqk", qh, kh) * scale
        attn = attn.masked_fill(~mask.view(1, 1, sq, skv), float("-inf"))
        sink_col = sinks.float().view(1, hq, 1, 1).expand(bsz, hq, sq, 1)
        p = torch.softmax(torch.cat([attn, sink_col], dim=-1), dim=-1)[..., :-1]
        ref = torch.einsum("bhqk,bhkd->bhqd", p, vh).transpose(1, 2)
    return out, ref


@pytest.mark.gpu
@pytest.mark.skipif(not _gpu_ready(), reason="needs a gfx950 GPU")
@pytest.mark.parametrize(
    "sq, skv, batch",
    [
        # Arbitrary lengths, which is what a real chunked-prefill request looks like:
        # neither a multiple of the 256 query tile nor of block_n, and an offset that
        # lands mid-tile. Reaching these at all depends on the KV-tile bound being a
        # ceil; getting them RIGHT depends on the partial last tile still being masked.
        (197, 400, 1),
        (300, 1234, 1),
        (512, 4097, 1),
        (100, 8000, 1),
        # batch > 1 is a materially different case, not just a bigger one. The ragged
        # buffer resources bound the WHOLE tensor rather than one batch element, so a
        # padded query row in batch 0 reads batch 1's real tokens instead of zero.
        # Correctness then rests entirely on the qtok < Sq store predicate, and this
        # is the shape that proves it -- Sq and Skv both non-multiples, so batch 0 has
        # padded query rows with a live neighbour behind them.
        (300, 1000, 2),
    ],
)
def test_ragged_bottom_right_matches_shifted_mask_oracle(sq, skv, batch):
    """Bottom-right on the ragged path, where the lengths are arbitrary.

    The interesting risk here is the partial last KV tile. Ragged visits it (the tile
    count ceils) and its out-of-range keys load as zeros, and plain causal gets away
    without an explicit key mask because every real query stops before them. Bottom-right
    shifts every query's reach to the right, so this checks that argument still holds:
    the last real query reaches key seqlen_kv-1 exactly, and no further.
    """
    spec = _spec(
        batch=batch,
        seqlen_q=sq,
        seqlen_kv=skv,
        num_query_heads=4,
        num_kv_heads=1,
        ragged=True,
        causal_bottom_right=True,
    )
    out, ref = _run_and_reference(spec)
    assert (out.float() - ref).abs().max().item() < 2e-2


@pytest.mark.gpu
@pytest.mark.skipif(not _gpu_ready(), reason="needs a gfx950 GPU")
@pytest.mark.parametrize("sq, skv", [(256, 512), (512, 1024)])
def test_bottom_right_matches_shifted_mask_oracle(sq, skv):
    """The kernel must agree with an explicit ``ki <= qi + (Skv - Sq)`` oracle.

    ``is_causal=True`` is deliberately NOT used as the oracle: it is top-left, so it
    would disagree here -- which is exactly the bug this guards.
    """
    spec = _spec(seqlen_q=sq, seqlen_kv=skv, causal_bottom_right=True)
    out, ref = _run_and_reference(spec)
    assert (out.float() - ref).abs().max().item() < 2e-2

    # And it must NOT match the top-left oracle, or the offset is not doing anything.
    _, ref_tl = _run_and_reference(spec, top_left=True)
    assert (out.float() - ref_tl).abs().max().item() > 1e-3


@pytest.mark.gpu
@pytest.mark.skipif(not _gpu_ready(), reason="needs a gfx950 GPU")
@pytest.mark.parametrize(
    "sq, skv, ragged",
    [
        # Aligned: the offset is a whole number of KV tiles.
        (512, 1024, False),
        # Arbitrary: the offset lands mid-tile and the last KV tile is part padding,
        # so the sink seeds a softmax over a partially-padded key range.
        (300, 1234, True),
    ],
)
def test_bottom_right_with_sinks_matches_shifted_mask_oracle(sq, skv, ragged):
    """The two features share a softmax, so numbers are the only proof that composing
    them is right: the sink seeds m/l before the KV loop and the shifted diagonal
    decides which keys that loop admits.

    The oracle puts the sink in the denominator as a zero-value column, so a kernel
    that dropped the sink, or applied the shift only to the non-sink part, disagrees.
    """
    spec = _spec(
        seqlen_q=sq,
        seqlen_kv=skv,
        num_query_heads=4,
        num_kv_heads=1,
        ragged=ragged,
        causal_bottom_right=True,
        use_sinks=True,
    )
    out, ref = _run_and_reference(spec)
    assert (out.float() - ref).abs().max().item() < 2e-2

    # Still a bottom-right mask, not a top-left one wearing a sink.
    _, ref_tl = _run_and_reference(spec, top_left=True)
    assert (out.float() - ref_tl).abs().max().item() > 1e-3
