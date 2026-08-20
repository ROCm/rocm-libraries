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
    check: under the current shape rules it could never fire.

    seqlen_q must be a multiple of the 256-row query tile, seqlen_kv a multiple of
    block_n, and every legal block_n divides 256 -- so seqlen_q is a multiple of block_n
    too and the difference always is as well. Enumerated rather than argued, so this
    starts failing if the shape rules change and the assumption quietly stops holding.

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


@pytest.mark.gpu
@pytest.mark.skipif(not _gpu_ready(), reason="needs a gfx950 GPU")
@pytest.mark.parametrize("sq, skv", [(256, 512), (512, 1024)])
def test_bottom_right_matches_shifted_mask_oracle(sq, skv):
    """The kernel must agree with an explicit ``ki <= qi + (Skv - Sq)`` oracle.

    ``is_causal=True`` is deliberately NOT used as the oracle: it is top-left, so it
    would disagree here -- which is exactly the bug this guards.
    """
    import torch

    from kernels.gfx950.attention_dense import run_attention_dense_torch

    spec = _spec(seqlen_q=sq, seqlen_kv=skv, causal_bottom_right=True)
    dev = "cuda"
    torch.manual_seed(0)
    b, hq, hkv, d = spec.batch, spec.num_query_heads, spec.num_kv_heads, spec.head_size
    q = torch.randn(b, hq, sq, d, device=dev, dtype=torch.bfloat16)
    k = torch.randn(b, hkv, skv, d, device=dev, dtype=torch.bfloat16)
    v = torch.randn(b, hkv, skv, d, device=dev, dtype=torch.bfloat16)

    out = run_attention_dense_torch(spec=spec, q=q, k=k, v=v)

    rep = hq // hkv
    qh = q.float()
    kh = k.repeat_interleave(rep, 1).float()
    vh = v.repeat_interleave(rep, 1).float()
    qi = torch.arange(sq, device=dev).view(-1, 1)
    ki = torch.arange(skv, device=dev).view(1, -1)
    ref = torch.nn.functional.scaled_dot_product_attention(
        qh, kh, vh, attn_mask=(ki <= qi + (skv - sq))
    )
    assert (out.float() - ref).abs().max().item() < 2e-2

    # And it must NOT match the top-left oracle, or the offset is not doing anything.
    ref_tl = torch.nn.functional.scaled_dot_product_attention(
        qh, kh, vh, is_causal=True
    )
    assert (out.float() - ref_tl).abs().max().item() > 1e-3
