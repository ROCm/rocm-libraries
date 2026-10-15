# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU dispatch tests for the MLA decode candidates (gfx942, gfx950).

The request describes the problem and never names the regime: one
``MLARequest`` and one op (``mla_fwd``), with decode selected by
``max_seqlen_q == 1`` and prefill by ``max_seqlen_q >= 2``. The two regimes
take different input contracts (absorbed query vs projected query + ``W_UK``),
so neither stands in for the other. These tests pin that routing contract --
each regime goes to its own kernel, a one-query request that decode cannot
serve is refused rather than handed to prefill, and every refusal says why.

Requests pin ``num_cus`` so the split-KV choice does not depend on the host,
and the LLVM flavor, which decode admission reads (llvm20 is refused).
"""

from __future__ import annotations

import pytest

from dispatch.mla import (
    MLA_OP,
    MLA_REGISTRY,
    MLARequest,
    dispatch_mla,
    mla_candidates,
    mla_decode_launch,
    registered_mla_combos,
)
from dispatch.mla.decode import DECODE_MIN_TILES_PER_SEG
from kernels.mla.mla_decode import MlaDecodeSpec, mla_decode_signature
from kernels.mla.mla_decode_combine import MlaDecodeCombineSpec
from kernels.mla.mla_decode_gfx950 import MlaDecodeGfx950Spec

PREFILL = "mla_prefill_fwd_gfx942"
DECODE = {"gfx942": "mla_decode_gfx942", "gfx950": "mla_decode_gfx950"}
SPEC_CLS = {"gfx942": MlaDecodeSpec, "gfx950": MlaDecodeGfx950Spec}


@pytest.fixture(autouse=True)
def _llvm22(monkeypatch):
    """Decode admission refuses llvm20; pin a flavor it serves."""
    monkeypatch.setenv("ROCKE_LLVM_FLAVOR", "llvm22")


def _decode(**overrides) -> MLARequest:
    """A DeepSeek-V3-shaped decode batch: 4 sequences, one query each.

    ``num_kv_blocks`` defaults to the pool the batch needs, every sequence at
    ``max_seqlen_k``: decode ignores it, but the prefill candidate refuses 0,
    and the routing tests need prefill's other admission checks to run.
    """
    base = dict(
        num_heads=128,
        total_q=4,
        num_seqs=4,
        max_seqlen_q=1,
        max_seqlen_k=8192,
        arch="gfx942",
        num_cus=304,
    )
    base.update(overrides)
    base.setdefault("num_kv_blocks", base["num_seqs"] * -(-base["max_seqlen_k"] // 16))
    return MLARequest(**base)


class TestRegistration:
    def test_one_decode_candidate_per_arch(self):
        for arch, name in DECODE.items():
            assert name in [c.name for c in MLA_REGISTRY.for_arch(arch)]
        assert [c.name for c in MLA_REGISTRY.for_arch("gfx950")] == [DECODE["gfx950"]]

    def test_decode_outranks_prefill(self):
        priority = {c.name: c.priority for c in mla_candidates()}
        assert priority[DECODE["gfx942"]] < priority[PREFILL]

    def test_identity(self):
        c = MLA_REGISTRY.get(DECODE["gfx950"])
        assert (c.algorithm, c.spec_id) == ("mla_decode_absorb", "gfx950_mla_decode")


class TestRouting:
    @pytest.mark.parametrize("arch", sorted(DECODE))
    @pytest.mark.parametrize("heads", [64, 128])
    def test_decode_wins_at_one_query_per_sequence(self, arch, heads):
        result = dispatch_mla(_decode(arch=arch, num_heads=heads))
        assert result.candidate.name == DECODE[arch]
        assert type(result.spec) is SPEC_CLS[arch]
        assert result.kernel_id.op == MLA_OP

    def test_prefill_refuses_one_query_per_sequence(self):
        # Prefill could compute S_q = 1, but under its own input contract
        # (projected query + W_UK), so it is not offered for decode's regime.
        req = _decode()
        ok, why = MLA_REGISTRY.get(PREFILL).admits(req)
        assert not ok and "max_seqlen_q=1 < min 2" in why
        names = [c.name for c, _spec in registered_mla_combos(req)]
        assert names == [DECODE["gfx942"]]

    def test_prefill_wins_at_two_queries(self):
        req = _decode(total_q=8, max_seqlen_q=2)
        ok, why = MLA_REGISTRY.get(DECODE["gfx942"]).admits(req)
        assert not ok and "max_seqlen_q=2 > max 1" in why
        assert dispatch_mla(req).candidate.name == PREFILL

    def test_decode_batch_may_hold_a_sequence_without_a_query(self):
        # total_q < num_seqs is still decode: the kernel exits for that sequence.
        assert dispatch_mla(_decode(total_q=3)).candidate.name == DECODE["gfx942"]

    @pytest.mark.parametrize("arch", sorted(DECODE))
    def test_one_query_request_decode_cannot_serve_is_refused(self, arch):
        # 32 heads do not fill a 64-head workgroup. No other kernel takes the
        # absorbed query, so the request is refused with every reason.
        req = _decode(arch=arch, num_heads=32)
        ok, why = MLA_REGISTRY.get(DECODE[arch]).admits(req)
        assert not ok and "num_heads=32 not a multiple of 64" in why
        with pytest.raises(ValueError, match="no candidate supports request") as e:
            dispatch_mla(req)
        assert "num_heads=32 not a multiple of 64" in str(e.value)

    @pytest.mark.parametrize("arch", sorted(DECODE))
    def test_llvm20_is_refused(self, monkeypatch, arch):
        # llvm20 ignores agpr_alloc: the kernel would spill and halve its
        # occupancy, against the two workgroups per CU the split count assumes.
        monkeypatch.setenv("ROCKE_LLVM_FLAVOR", "llvm20")
        ok, why = MLA_REGISTRY.get(DECODE[arch]).admits(_decode(arch=arch))
        assert not ok and "llvm20" in why
        with pytest.raises(ValueError, match="llvm20"):
            dispatch_mla(_decode(arch=arch))

    @pytest.mark.parametrize("arch", sorted(DECODE))
    @pytest.mark.parametrize("flavor", ["llvm22", "llvm23"])
    def test_supported_flavors_are_admitted(self, monkeypatch, arch, flavor):
        monkeypatch.setenv("ROCKE_LLVM_FLAVOR", flavor)
        assert dispatch_mla(_decode(arch=arch)).candidate.name == DECODE[arch]

    def test_causal_flag_is_immaterial_at_one_query(self):
        # Bottom-right causal admits every key at S_q = 1, so both are served.
        for causal in (True, False):
            assert (
                dispatch_mla(_decode(causal=causal)).candidate.name == DECODE["gfx942"]
            )

    @pytest.mark.parametrize("arch", sorted(DECODE))
    def test_foreign_op_is_refused_by_every_candidate(self, arch):
        # Each candidate gets a request in its own regime, so the op is what
        # it refuses on rather than the query count.
        candidates = MLA_REGISTRY.for_arch(arch)
        assert candidates
        for c in candidates:
            regime = (
                {}
                if c.name.startswith("mla_decode")
                else {"total_q": 8, "max_seqlen_q": 2}
            )
            req = _decode(arch=arch, op="mla_decode_absorb_fwd", **regime)
            ok, why = c.admits(req)
            assert not ok and "is not served by the MLA family" in why, (c.name, why)

    def test_batches_past_the_i32_offsets_are_refused(self):
        # qa is indexed (token * H + head) * 576 in i32: about 29k tokens at
        # 128 heads overflow it. Decode refuses; it must not admit a launch
        # it cannot address.
        req = _decode(total_q=40000, num_seqs=40000)
        ok, why = MLA_REGISTRY.get(DECODE["gfx942"]).admits(req)
        assert not ok and "i32 offsets" in why

    def test_split_count_is_clamped_to_the_workspace(self):
        # A large explicit CU count asks for many segments; the split
        # workspace row offset must stay i32-addressable regardless.
        spec = dispatch_mla(
            _decode(total_q=2000, num_seqs=2000, max_seqlen_k=131072, num_cus=100000)
        ).spec
        assert 2000 * spec.num_seg * 128 * 512 < 2**31

    def test_attention_request_is_refused(self):
        from dispatch.attention import AttentionRequest

        foreign = AttentionRequest(
            batch=1,
            nhead_q=8,
            nhead_k=8,
            seqlen_q=1,
            seqlen_k=64,
            hdim_q=128,
            hdim_v=128,
            arch="gfx942",
        )
        for c in mla_candidates():
            assert not c.admits(foreign)[0]

    @pytest.mark.parametrize(
        "kw, match",
        [
            ({"max_seqlen_q": 0}, "max_seqlen_q must be positive"),
            ({"max_seqlen_q": 5}, "cannot exceed total_q"),
            ({"total_q": 9, "max_seqlen_q": 2}, "exceeds max_seqlen_q"),
        ],
    )
    def test_malformed_requests(self, kw, match):
        with pytest.raises(ValueError, match=match):
            dispatch_mla(_decode(**kw))

    def test_decode_and_prefill_share_the_op_not_the_key(self):
        decode = dispatch_mla(_decode()).kernel_id
        prefill = dispatch_mla(_decode(total_q=8, max_seqlen_q=2)).kernel_id
        assert decode.op == prefill.op == MLA_OP
        assert decode.selection_key != prefill.selection_key
        assert decode.candidate != prefill.candidate


class TestSplitKvChoice:
    def test_one_long_sequence_splits(self):
        spec = dispatch_mla(_decode(total_q=1, num_seqs=1, max_seqlen_k=32768)).spec
        assert spec.num_seg == 64  # 2048 tiles / DECODE_MIN_TILES_PER_SEG
        assert 32768 // 16 // spec.num_seg >= DECODE_MIN_TILES_PER_SEG

    @pytest.mark.parametrize("arch, fallback", [("gfx942", 304), ("gfx950", 256)])
    def test_fallback_cu_count_when_the_host_is_not_the_target(
        self, arch, fallback, monkeypatch
    ):
        import rocke.runtime.hip_module as hm

        monkeypatch.setattr(hm, "get_device_arch", lambda: None)
        # 16384 tiles: the tile cap (512) does not bind, so num_seg is the fill
        # target, ceil(fallback * 2 workgroups / 2 head groups).
        spec = dispatch_mla(
            _decode(arch=arch, total_q=1, num_seqs=1, max_seqlen_k=262144, num_cus=0)
        ).spec
        assert spec.num_seg == fallback

    def test_split_count_does_not_fork_the_binary(self):
        # Two batches that pick different split counts share one compile.
        a = dispatch_mla(_decode(total_q=1, num_seqs=1, max_seqlen_k=32768))
        b = dispatch_mla(_decode(total_q=4, num_seqs=4, max_seqlen_k=8192))
        assert a.spec.num_seg != b.spec.num_seg
        assert a.spec.split_kv and b.spec.split_kv
        assert a.kernel_id.compile_key == b.kernel_id.compile_key

    def test_short_keys_stay_single_pass(self):
        spec = dispatch_mla(_decode(total_q=1, num_seqs=1, max_seqlen_k=511)).spec
        assert spec.num_seg == 1 and not spec.split_kv

    def test_grid_carries_the_split(self):
        result = dispatch_mla(_decode())
        assert result.grid == (2, 4, result.spec.num_seg)
        assert result.signature == tuple(mla_decode_signature(result.spec))


class TestLaunch:
    """``mla_decode_launch``: both stages of a selection, from dispatch alone."""

    @pytest.mark.parametrize("arch", sorted(DECODE))
    def test_split_selection_describes_the_combine(self, arch):
        result = dispatch_mla(_decode(arch=arch))
        launch = mla_decode_launch(result)
        n = result.spec.num_seg
        assert n > 1
        assert launch.combine_spec == MlaDecodeCombineSpec(num_heads=128)
        assert launch.workspace_shapes() == {
            "part_acc_ptr": (4, n, 128, 512),
            "part_ml_ptr": (4, n, 128, 2),
        }
        assert launch.combine_grid() == (32, 4, 1)
        assert launch.combine_block() == (256, 1, 1)
        assert [a["name"] for a in launch.combine_signature()] == [
            "out_ptr",
            "lse_ptr",
            "part_acc_ptr",
            "part_ml_ptr",
            "num_seg",
        ]
        assert launch.build_combine().name == launch.combine_spec.kernel_name()
        assert launch.kernel_names() == (
            result.spec.kernel_name(),
            launch.combine_spec.kernel_name(),
        )
        # The latent kernel writes exactly the workspace the combine reads.
        latent_out = [a["name"] for a in result.signature][:2]
        assert latent_out == list(launch.workspace_shapes())

    def test_single_pass_selection_is_one_launch(self):
        launch = mla_decode_launch(
            dispatch_mla(_decode(total_q=1, num_seqs=1, max_seqlen_k=511))
        )
        assert launch.combine_spec is None
        assert launch.workspace_shapes() == {}
        assert len(launch.kernel_names()) == 1

    def test_prefill_selection_is_not_a_decode_launch(self):
        with pytest.raises(ValueError, match="not an MLA decode selection"):
            mla_decode_launch(dispatch_mla(_decode(total_q=8, max_seqlen_q=2)))


class TestSupportImpliesBuildable:
    @pytest.mark.parametrize("arch", sorted(DECODE))
    @pytest.mark.parametrize("max_k", [256, 8192])
    def test_admitted_requests_build_with_the_reported_pack(self, arch, max_k):
        result = dispatch_mla(_decode(arch=arch, max_seqlen_k=max_k))
        kernel = result.candidate.built(result.spec, arch)
        assert kernel.name == result.spec.kernel_name()
        assert [p.name for p in kernel.params] == [a["name"] for a in result.signature]
        assert kernel.max_workgroup_size >= result.block[0]
