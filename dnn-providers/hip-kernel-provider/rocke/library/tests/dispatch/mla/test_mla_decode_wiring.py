# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU dispatch tests for the MLA decode candidates (gfx942, gfx950).

The request describes the problem and never names the regime: one
``MLARequest`` and one op (``mla_fwd``), with decode selected by
``max_seqlen_q == 1`` and prefill by ``max_seqlen_q >= 2``. Prefill serves
both input contracts (projected query + ``W_UK``, or the absorbed query);
decode serves the absorbed one only and requires ``absorbed_query``, so a
fused one-query request has no kernel. These tests pin that routing contract --
each regime goes to its own kernel, a one-query request that decode cannot
serve is refused rather than handed to prefill, and every refusal says why.

Requests pin ``num_cus`` so the split-KV choice does not depend on the host,
and the LLVM flavor, which decode admission reads (llvm20 is refused).
"""

from __future__ import annotations

import types

import pytest

from builders.mla.mla_decode_runner import MlaDecodeRunner
from dispatch.mla import (
    MLA_OP,
    MLA_REGISTRY,
    MLARequest,
    dispatch_mla,
    mla_candidates,
    mla_decode_launch,
    registered_mla_combos,
)
from dispatch.mla.common import resolve_num_cus
from dispatch.mla.decode import DECODE_MIN_TILES_PER_SEG
from kernels.mla._mla_body import flavor_honors_agpr_alloc
from kernels.mla.mla_decode import MlaDecodeSpec, mla_decode_signature
from kernels.mla.mla_decode_combine import MlaDecodeCombineSpec
from kernels.mla.mla_decode_gfx950 import MlaDecodeGfx950Spec
from rocke.core.lower_llvm import LLVM_FLAVORS

PREFILL = "mla_prefill_fwd_gfx942"
DECODE = {"gfx942": "mla_decode_gfx942", "gfx950": "mla_decode_gfx950"}
SPEC_CLS = {"gfx942": MlaDecodeSpec, "gfx950": MlaDecodeGfx950Spec}
# The flavors decode serves: those the MLA family's one rule admits.
SERVED_LANES = tuple(f for f in LLVM_FLAVORS if flavor_honors_agpr_alloc(f))


@pytest.fixture(autouse=True)
def _llvm22(monkeypatch):
    """Decode admission refuses llvm20; pin a flavor it serves."""
    monkeypatch.setenv("ROCKE_LLVM_FLAVOR", "llvm22")


def _decode(**overrides) -> MLARequest:
    """A DeepSeek-V3-shaped decode batch: 4 sequences, one query each, in the
    absorbed-query contract decode requires.

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
        absorbed_query=True,
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
        # Prefill could compute S_q = 1, but decode owns that regime: prefill
        # is not offered for it under either input contract.
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

    @pytest.mark.parametrize("arch", sorted(DECODE))
    def test_one_query_request_without_absorbed_query_is_refused(self, arch):
        # Decode's only contract is the absorbed query; a request for the
        # fused one must not be served qa-shaped tensors it did not describe.
        req = _decode(arch=arch, absorbed_query=False)
        ok, why = MLA_REGISTRY.get(DECODE[arch]).admits(req)
        assert not ok and "requires features ['absorbed_query']" in why
        with pytest.raises(ValueError, match="no candidate supports request"):
            dispatch_mla(req)

    @pytest.mark.parametrize("value", [None, 0])
    def test_mistyped_absorbed_query_is_named(self, value):
        # A falsy non-bool must pass the required-feature prefilter so the
        # predicate names the type error, not a missing feature.
        ok, why = MLA_REGISTRY.get(DECODE["gfx942"]).admits(
            _decode(absorbed_query=value)
        )
        assert not ok and "absorbed_query must be a bool" in why

    def test_decode_batch_may_hold_a_sequence_without_a_query(self):
        # total_q < num_seqs is still decode: the kernel exits for that sequence.
        assert dispatch_mla(_decode(total_q=3)).candidate.name == DECODE["gfx942"]

    @pytest.mark.parametrize("num_heads", [16, 32, 192])
    @pytest.mark.parametrize("arch", sorted(DECODE))
    def test_one_query_request_decode_cannot_serve_is_refused(self, arch, num_heads):
        # 16 / 32 heads do not fill a 64-head workgroup; 192 lowers but has no
        # GPU evidence yet. Prefill takes no one-query request, so the request
        # is refused with every reason.
        req = _decode(arch=arch, num_heads=num_heads)
        reason = f"num_heads={num_heads} not in (64, 128)"
        ok, why = MLA_REGISTRY.get(DECODE[arch]).admits(req)
        assert not ok and reason in why
        with pytest.raises(ValueError, match="no candidate supports request") as e:
            dispatch_mla(req)
        assert reason in str(e.value)

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
    @pytest.mark.parametrize("flavor", SERVED_LANES)
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


class TestFlavorRule:
    """The one rule for the LLVM flavors MLA kernels serve: decode's admission,
    prefill's ``agpr_acc`` gate and the tests all ask
    ``flavor_honors_agpr_alloc``."""

    def test_llvm20_is_refused_and_the_oldest_modern_flavor_is_served(self):
        assert not flavor_honors_agpr_alloc("llvm20")
        assert flavor_honors_agpr_alloc("llvm22")

    def test_a_newer_lane_is_served_whatever_its_datalayout(self, monkeypatch):
        # The rule ranks compiler lanes; a lane added after the newest one is
        # served before anyone records its datalayout generation.
        from rocke.core import llvm_profile

        ranks = dict(llvm_profile._LLVM_FLAVOR_RANK)
        ranks["llvm99"] = max(ranks.values()) + 1
        monkeypatch.setattr(llvm_profile, "_LLVM_FLAVOR_RANK", ranks)
        assert flavor_honors_agpr_alloc("llvm99")

    def test_a_name_that_is_not_a_lane_raises(self):
        with pytest.raises(ValueError, match="unknown LLVM flavor"):
            flavor_honors_agpr_alloc("gcc")

    @pytest.mark.parametrize("arch", sorted(DECODE))
    @pytest.mark.parametrize("flavor", LLVM_FLAVORS)
    def test_decode_admits_exactly_the_flavors_the_rule_serves(
        self, monkeypatch, arch, flavor
    ):
        monkeypatch.setenv("ROCKE_LLVM_FLAVOR", flavor)
        ok, why = MLA_REGISTRY.get(DECODE[arch]).admits(_decode(arch=arch))
        assert ok == flavor_honors_agpr_alloc(flavor), why


class TestKeyLength:
    """The kernel rounds each sequence's key count up to tiles in i32."""

    @pytest.mark.parametrize("arch", sorted(DECODE))
    def test_a_key_count_past_the_i32_tile_math_is_refused(self, arch):
        ok, why = MLA_REGISTRY.get(DECODE[arch]).admits(
            _decode(arch=arch, max_seqlen_k=2**31 - 1)
        )
        assert not ok and "tile count" in why

    @pytest.mark.parametrize("arch", sorted(DECODE))
    def test_the_largest_key_count_the_tile_math_holds_is_admitted(self, arch):
        # block_k is the request's page size (one k-tile per page).
        top = 2**31 - _decode(arch=arch).page_block_size
        ok, why = MLA_REGISTRY.get(DECODE[arch]).admits(
            _decode(arch=arch, max_seqlen_k=top)
        )
        assert ok, why


class TestGridLimits:
    """Every launch fits the platform's per-axis grid limit: sequences on y,
    split-KV segments on z (the combine's y, tokens, is no wider than the
    latent's). Selection is a pure function of the request, so the live
    device's own limit is the runner's to check at launch."""

    #: A split launch (long keys, one sequence): the one whose z axis is narrowed.
    SPLIT = {"total_q": 1, "num_seqs": 1, "max_seqlen_k": 32768}

    @classmethod
    def _narrowest_split_limit(cls):
        """The narrowest scalar limit a split launch of ``SPLIT`` still meets: the
        widest axis of its own combine grid, ``(heads // num_warps, tokens, 1)``.
        Clamping to it narrows the segment (z) axis alone."""
        launch = mla_decode_launch(dispatch_mla(_decode(**cls.SPLIT)))
        return max(launch.combine_grid())

    @staticmethod
    def _limit(monkeypatch, limit):
        monkeypatch.setattr("dispatch.mla.decode.MAX_GRID_DIM", limit)

    def test_sequences_past_the_y_limit_are_refused(self, monkeypatch):
        self._limit(monkeypatch, 100)
        assert dispatch_mla(_decode(total_q=100, num_seqs=100)).grid[1] == 100
        # Sequences with no query still occupy a grid row.
        req = _decode(total_q=4, num_seqs=101)
        ok, why = MLA_REGISTRY.get(DECODE["gfx942"]).admits(req)
        assert not ok and "y extent 101 exceeds the grid limit 100" in why

    def test_split_count_is_clamped_to_the_z_limit(self, monkeypatch):
        req = _decode(**self.SPLIT)
        narrow = self._narrowest_split_limit()
        # Unclamped, it would split further.
        assert dispatch_mla(req).spec.num_seg > narrow
        self._limit(monkeypatch, narrow)
        result = dispatch_mla(req)
        assert result.spec.num_seg == narrow and result.grid[2] == narrow

    def test_a_forced_split_count_meets_the_same_limit(self, monkeypatch):
        launch = mla_decode_launch(dispatch_mla(_decode(**self.SPLIT)))
        narrow = max(launch.combine_grid())
        self._limit(monkeypatch, narrow)
        # At the limit: admitted.
        assert launch.with_num_seg(narrow).spec.num_seg == narrow
        match = f"z extent {narrow + 1} exceeds the grid limit {narrow}"
        with pytest.raises(ValueError, match=match):
            launch.with_num_seg(narrow + 1)

    def test_a_forced_split_count_must_be_an_int(self):
        launch = mla_decode_launch(dispatch_mla(_decode(**self.SPLIT)))
        with pytest.raises(TypeError, match="num_seg must be an int, got float"):
            launch.with_num_seg(2.0)

    def test_selection_does_not_read_the_device(self, monkeypatch):
        """Dispatch ARCHITECTURE.md, section 6.2, rule 3: with ``num_cus`` given,
        selection asks the device for nothing (arch, CU count, grid limit), so a
        request selects the same kernel on any host."""
        from rocke.runtime import hip_module

        req = _decode(total_q=1, num_seqs=1, max_seqlen_k=32768)
        want = dispatch_mla(req)
        asked = []

        def spy(name, answer):
            def probe(*args, **kwargs):
                asked.append(name)
                return answer

            return probe

        # A host whose device is the requested arch. Every probe answers benignly
        # and is recorded: a regression fails on ``asked``, not somewhere inside
        # selection, and not inside a handler that swallows an exception.
        probes = {
            "get_device_arch": "gfx942",
            "get_device_num_cus": 1,
            "get_device_max_grid_size": None,
        }
        for name, answer in probes.items():
            monkeypatch.setattr(hip_module, name, spy(name, answer))
        got = dispatch_mla(req)
        assert asked == []
        assert (got.spec, got.grid) == (want.spec, want.grid)


class TestGridsFitDevice:
    """The device's own grid limit is checked where the launch happens
    (``MlaDecodeRunner._check_grids_fit_device``); selection never asks.

    Torch-free on purpose: ``test_mla_decode_runner.py`` skips wholesale without
    torch, and the CI lane that runs this file has none. The call site, which
    needs tensors, is ``TestLaunchChecksTheDevice`` there.
    """

    CPU = types.SimpleNamespace(index=None)  # what ``torch.device("cpu").index`` is

    @staticmethod
    def _device(monkeypatch, limits, asked=None):
        from rocke.runtime import hip_module

        def limits_of(device=0):
            if asked is not None:
                asked.append(device)
            return limits

        monkeypatch.setattr(hip_module, "get_device_max_grid_size", limits_of)

    @staticmethod
    def _runner(**request):
        """A runner holding ``request``'s selection, without compiling a kernel."""
        result = dispatch_mla(_decode(**request))
        runner = object.__new__(MlaDecodeRunner)
        runner.launch = mla_decode_launch(result)
        runner.grid = tuple(result.grid)
        return runner

    @pytest.mark.parametrize(
        ("max_seqlen_k", "split"),
        [(511, False), (32768, True)],
        ids=["single-pass", "split"],
    )
    def test_a_grid_exactly_at_the_device_limit_passes(
        self, monkeypatch, max_seqlen_k, split
    ):
        runner = self._runner(max_seqlen_k=max_seqlen_k)
        assert (runner.launch.combine_spec is not None) == split
        grids = [runner.grid]
        if split:
            grids.append(runner.launch.combine_grid())
        # The widest of the launches on each axis.
        self._device(monkeypatch, tuple(max(extents) for extents in zip(*grids)))
        runner._check_grids_fit_device(self.CPU)

    @pytest.mark.parametrize("axis", [0, 1, 2], ids=["x", "y", "z"])
    def test_a_latent_grid_wider_than_the_device_is_refused(self, monkeypatch, axis):
        runner = self._runner(max_seqlen_k=32768)
        limits = [65535] * 3
        limits[axis] = runner.grid[axis] - 1
        self._device(monkeypatch, tuple(limits))
        match = (
            f"latent grid's {'xyz'[axis]} extent {runner.grid[axis]} "
            f"exceeds the device's grid limit {limits[axis]}$"
        )
        with pytest.raises(ValueError, match=match):
            runner._check_grids_fit_device(self.CPU)

    def test_the_combine_grid_of_a_split_launch_is_checked(self, monkeypatch):
        runner = self._runner(max_seqlen_k=32768)
        wide = runner.launch.combine_grid()[0]
        # Only the combine is this wide on x, so only its check can refuse it.
        assert wide > runner.grid[0]
        self._device(monkeypatch, (wide - 1, 65535, 65535))
        match = (
            f"split-KV combine grid's x extent {wide} "
            f"exceeds the device's grid limit {wide - 1}$"
        )
        with pytest.raises(ValueError, match=match):
            runner._check_grids_fit_device(self.CPU)

    def test_an_unknown_device_limit_is_left_to_the_launch(self, monkeypatch):
        self._device(monkeypatch, None)
        self._runner(max_seqlen_k=32768)._check_grids_fit_device(self.CPU)

    def test_the_limit_is_asked_of_the_device_the_launch_runs_on(self, monkeypatch):
        asked = []
        self._device(monkeypatch, None, asked)
        runner = self._runner(max_seqlen_k=511)
        runner._check_grids_fit_device(types.SimpleNamespace(index=3))
        runner._check_grids_fit_device(self.CPU)  # no index: the default device
        assert asked == [3, 0]


class TestNumCusResolution:
    """The one device read dispatch keeps (ARCHITECTURE.md, section 6.2, rule 3):
    the CU count behind an unset ``num_cus``, and only on a host whose device is
    the requested arch."""

    @staticmethod
    def _host(monkeypatch, arch, num_cus=77):
        """A host whose device is ``arch`` (or whose arch query raises, if ``arch``
        is an exception type) and has ``num_cus`` CUs."""
        from rocke.runtime import hip_module

        def get_device_arch():
            if isinstance(arch, type) and issubclass(arch, Exception):
                raise arch("no device")
            return arch

        monkeypatch.setattr(hip_module, "get_device_arch", get_device_arch)
        monkeypatch.setattr(hip_module, "get_device_num_cus", lambda *a, **k: num_cus)

    def test_a_given_count_is_used_and_no_device_is_asked(self, monkeypatch):
        self._host(monkeypatch, OSError)  # any query would fail
        assert resolve_num_cus(_decode(num_cus=304)) == 304

    def test_an_unset_count_comes_from_the_requested_archs_device(self, monkeypatch):
        self._host(monkeypatch, "gfx942")
        assert resolve_num_cus(_decode(num_cus=0)) == 77

    def test_target_id_features_are_ignored(self, monkeypatch):
        self._host(monkeypatch, "gfx942:sramecc+:xnack-")
        assert resolve_num_cus(_decode(num_cus=0)) == 77

    @pytest.mark.parametrize("arch", [None, ""])
    def test_an_unnamed_device_supplies_nothing(self, monkeypatch, arch):
        self._host(monkeypatch, arch)
        assert resolve_num_cus(_decode(num_cus=0)) == 0

    def test_another_arch_on_the_build_box_supplies_nothing(self, monkeypatch):
        self._host(monkeypatch, "gfx950")
        assert resolve_num_cus(_decode(num_cus=0)) == 0

    @pytest.mark.parametrize("exc", [OSError, RuntimeError])
    def test_a_failing_device_query_supplies_nothing(self, monkeypatch, exc):
        self._host(monkeypatch, exc)
        assert resolve_num_cus(_decode(num_cus=0)) == 0


class TestSupportImpliesBuildable:
    @pytest.mark.parametrize("arch", sorted(DECODE))
    @pytest.mark.parametrize("max_k", [256, 8192])
    def test_admitted_requests_build_with_the_reported_pack(self, arch, max_k):
        result = dispatch_mla(_decode(arch=arch, max_seqlen_k=max_k))
        kernel = result.candidate.built(result.spec, arch)
        assert kernel.name == result.spec.kernel_name()
        assert [p.name for p in kernel.params] == [a["name"] for a in result.signature]
        assert kernel.max_workgroup_size >= result.block[0]
