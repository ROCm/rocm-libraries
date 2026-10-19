# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1151 paged split-K decode attention tests -- CPU only.

Everything here is structural: it asserts on the *shape* of the emitted LLVM
IR, which is where this kernel's performance actually lives. Numerics are
covered by the GPU verify script against the fp32 oracle; a kernel that
computes the right answer while emitting scalar gathers or a divergent-load
waterfall passes every numeric test and is worthless, so those properties get
their own assertions:

  * ``readfirstlane`` count stays at exactly 2 -- the guard against the
    32-iteration waterfall AMDGPU wraps around a divergent buffer/gather base;
  * the K gather is one ``<EPT x half>`` load per token and the V gather one
    ``<block_n x half>`` load per lane-owned ``d`` (the transposed-V payoff);
  * the workspace write is a run of maximal vector stores, not EPT scalars;
  * the reduce keeps the three-pass NaN-safe form (an all-empty split set must
    yield 0, not ``0 * inf``);
  * ``is_valid_spec`` rejects every shape the address math cannot express.

The kernel lowers on any host -- only execution needs the board -- so nothing
here is GPU-gated.

    python3 -m pytest library/tests/test_gfx1151_paged_decode_splitk.py
"""

from __future__ import annotations

import re
import unittest
from collections import Counter

from kernels.gfx1151.paged_decode_splitk import (
    WAVE,
    PagedDecodeCfg,
    _chunk_widths,
    build_paged_decode_splitk_reduce,
    build_paged_decode_splitk_segment,
    choose_num_splits,
    is_valid_spec,
    paged_decode_reduce_grid,
    paged_decode_segment_grid,
    paged_decode_workspace_shapes,
)

# Qwen3-8B on the gfx1151 box: D=128, 32 Q heads over 8 KV heads, fp16,
# vLLM's ROCm default page size.
_D = 128
_HQ = 32
_HK = 8


def _cfg(**kw) -> PagedDecodeCfg:
    kw.setdefault("head_size", _D)
    kw.setdefault("num_q_heads", _HQ)
    kw.setdefault("num_kv_heads", _HK)
    return PagedDecodeCfg(**kw)


def _subtiles(cfg) -> int:
    """Inner sub-tile iterations per physical block, within one key partition.

    Not ``kv_block_size // block_n``: once ``d_lanes < WAVE`` a wave splits the
    block's slots across ``key_subs`` partitions and each lane walks only its
    own share.
    """
    return cfg.keys_per_sub // cfg.block_n


def _fold(cfg) -> dict:
    """Op counts contributed by the once-per-segment cross-partition fold.

    The ``key_subs`` partitions each ran an independent online softmax over
    disjoint keys, so they merge with the same rescale the reduce kernel
    applies across splits -- ``log2(key_subs)`` butterfly stages of
    ``(m, l, acc)`` per fused head. Paying that once per segment is exactly
    what buys the shorter per-key QK reduce, so it belongs inside every budget
    below rather than being excluded from them.
    """
    per = (cfg.key_subs.bit_length() - 1) * cfg.gqa_fuse
    return {
        "swizzle": per * (2 + cfg.ept),  # m, l, and every acc element
        "exp2": per * 2,  # one rescale factor per side
        "fma": per * (1 + cfg.ept),  # the l update and every acc element
    }


def _lower(kernel):
    from rocke.core.lower_llvm import lower_kernel_to_llvm

    return lower_kernel_to_llvm(kernel, arch="gfx1151")


def _seg(cfg: PagedDecodeCfg) -> str:
    return _lower(build_paged_decode_splitk_segment(cfg, arch="gfx1151"))


def _red(cfg: PagedDecodeCfg) -> str:
    return _lower(build_paged_decode_splitk_reduce(cfg, arch="gfx1151"))


def _calls(ll: str, intrinsic: str) -> int:
    # The `declare` line names the intrinsic too; count call sites only.
    return len(re.findall(r"call [\w<>x* ]+ @" + re.escape(intrinsic) + r"\(", ll))


def _dup_locals(ll: str):
    """Local SSA names defined more than once -- LLVM rejects these outright."""
    from collections import Counter

    names = re.findall(r"^\s*%([\w.]+) = ", ll, flags=re.M)
    return sorted(n for n, c in Counter(names).items() if c > 1)


class TestConfig(unittest.TestCase):
    def test_derived_quantities(self):
        cfg = _cfg()
        self.assertEqual(cfg.gqa_fuse, 4)
        self.assertEqual(cfg.ept, _D // cfg.d_lanes)
        self.assertEqual(cfg.ept_reduce, _D // WAVE)
        self.assertEqual(cfg.key_subs * cfg.d_lanes, WAVE)
        self.assertEqual(cfg.keys_per_sub * cfg.key_subs, cfg.kv_block_size)
        self.assertEqual(cfg.x, 8)

    def test_gqa_fusion_is_total(self):
        # The whole bandwidth argument is that one CTA covers ALL Q heads of a
        # KV head, so K/V is read once rather than gqa_ratio times. A partial
        # fusion would still be correct and would silently give back the win.
        cfg = _cfg()
        self.assertEqual(cfg.gqa_fuse * cfg.num_kv_heads, cfg.num_q_heads)

    def test_valid_default(self):
        ok, why = is_valid_spec(_cfg())
        self.assertTrue(ok, why)

    def test_rejects_bad_shapes(self):
        bad = [
            _cfg(head_size=48),  # not a multiple of the wave size
            _cfg(head_size=96),  # ept=3 is not a vector width
            _cfg(head_size=192),  # ept=6 likewise
            _cfg(kv_block_size=24),  # not a power of two
            _cfg(kv_block_size=8),  # vLLM's ROCm backend requires % 16 == 0
            _cfg(num_splits=6),  # not a power of two
            _cfg(num_splits=256),  # above the 128 cap
            _cfg(block_n=16),  # wider than global_load_vN covers
            _cfg(block_n=3),  # not a vector width
            _cfg(num_q_heads=30),  # not a multiple of num_kv_heads
            _cfg(dtype="f32"),  # storage dtype must be 2-byte
            _cfg(d_lanes=12),  # not a power of two
            _cfg(d_lanes=64),  # above the wave width
            _cfg(d_lanes=1),  # ept would be the whole head, no reduce at all
            # key_subs=4 but only 2 key partitions' worth of slots.
            _cfg(d_lanes=8, kv_block_size=16, block_n=8),
        ]
        for cfg in bad:
            with self.subTest(cfg=cfg):
                ok, _ = is_valid_spec(cfg)
                self.assertFalse(ok)
        for cfg in bad:
            with self.subTest(raises=cfg):
                with self.assertRaises(ValueError):
                    build_paged_decode_splitk_segment(cfg, arch="gfx1151")

    def test_ept_must_divide_the_k_xsplit(self):
        # A lane's d-slice has to sit inside ONE x-group or its K load is not
        # contiguous and the gather silently reads the wrong elements.
        ok, why = is_valid_spec(_cfg(head_size=64, num_q_heads=_HQ))
        self.assertTrue(ok, why)  # ept=2 divides x=8
        # ept=8 (head_size 256) also divides x=8.
        ok, why = is_valid_spec(_cfg(head_size=256))
        self.assertTrue(ok, why)

    def test_rejects_other_arch(self):
        ok, _ = is_valid_spec(_cfg(), arch="gfx950")
        self.assertFalse(ok)


class TestSegmentCodegen(unittest.TestCase):
    def test_builds_and_lowers(self):
        for splits in (1, 8, 16):
            with self.subTest(splits=splits):
                ll = _seg(_cfg(num_splits=splits))
                self.assertIn("define", ll)
                self.assertIn("addrspace(1)", ll)

    def test_no_waterfall(self):
        # Exactly two readfirstlanes: the sequence length (so the split bounds
        # and the block loop stay scalar) and the RAW block id (without which
        # AMDGPU cannot prove the gather base is wave-uniform and wraps EVERY
        # K/V load in a 32-iteration waterfall loop). Any more means something
        # divergent leaked into the address math; any fewer means the waterfall
        # is back. This is the single highest-value assertion in the file.
        for splits in (1, 8, 16):
            for kb in (16, 32):
                with self.subTest(splits=splits, kv_block_size=kb):
                    ll = _seg(_cfg(num_splits=splits, kv_block_size=kb))
                    self.assertEqual(_calls(ll, "llvm.amdgcn.readfirstlane.i32"), 2)

    def test_no_lds(self):
        # The body is register-resident by design: one wave per CTA means
        # there is nothing to share, and an LDS round trip would only add
        # barriers. addrspace(3) appearing here is a regression.
        self.assertNotIn("addrspace(3)", _seg(_cfg()))

    def test_gather_widths(self):
        # K: one <ept x half> per token per lane (the x-split makes a lane's
        # d-slice contiguous). V: one <block_n x half> per lane-owned d --
        # this is the transposed-layout payoff, and its absence would mean
        # block_n scalar loads per d instead.
        cfg = _cfg()
        ll = _seg(cfg)
        got = Counter(re.findall(r"load <(\d+) x half>", ll))

        want: Counter = Counter()
        # Q: one run of maximal vector loads per fused head, same shape as K.
        for w in _chunk_widths(cfg.ept):
            want[str(w)] += cfg.gqa_fuse
        # K: the x-split makes a lane's d-slice contiguous, so it is that same
        # run per token -- one load when the slice fits inside an x-group, else
        # one per whole group it spans.
        k_run = [cfg.ept] if cfg.ept <= cfg.x else [cfg.x] * (cfg.ept // cfg.x)
        for w in k_run:
            want[str(w)] += _subtiles(cfg) * cfg.block_n
        # V: one <block_n x half> per lane-owned d. Absent this it would be
        # block_n scalar loads per d -- the transposed-layout payoff, gone.
        want[str(cfg.block_n)] += _subtiles(cfg) * cfg.ept

        # Compared as a whole multiset: at the shipping config ept, block_n and
        # x all happen to be 8, so per-width assertions would alias each other.
        self.assertEqual(got, want)

    def test_workspace_write_is_vectorised(self):
        # One run of maximal vector stores per fused head, never ept scalars.
        # At the shipping d_lanes=8 a lane owns 16 floats, which is wider than
        # any single store, so the run is two <8 x float> rather than one.
        cfg = _cfg()
        ll = _seg(cfg)
        got = Counter(re.findall(r"store <(\d+) x float>", ll))
        want: Counter = Counter()
        for w in _chunk_widths(cfg.ept):
            want[str(w)] += cfg.gqa_fuse
        self.assertEqual(got, want)

    def test_reduction_stays_in_wave32_swizzle(self):
        # log2(d_lanes) XOR stages per QK dot, plus the fold. Every mask stays
        # under 32, so every stage is a ds_swizzle; a ds_bpermute here means a
        # wave64 reduction leaked in from the gfx950 sibling.
        cfg = _cfg()
        ll = _seg(cfg)
        stages = cfg.d_lanes.bit_length() - 1
        expect = _subtiles(cfg) * cfg.block_n * cfg.gqa_fuse * stages
        self.assertEqual(
            _calls(ll, "llvm.amdgcn.ds.swizzle"), expect + _fold(cfg)["swizzle"]
        )
        self.assertNotIn("bpermute", ll)

    def test_softmax_op_budget(self):
        # One exp2 per (key, head) for the probability plus one per (sub-tile,
        # head) for the running rescale. Sub-tiling is what keeps the rescale
        # off the per-key path; if this count jumps to one alpha per key, the
        # sub-tile online-softmax collapsed back to the scalar form.
        cfg = _cfg()
        ll = _seg(cfg)
        expect = _subtiles(cfg) * cfg.gqa_fuse * (cfg.block_n + 1)
        self.assertEqual(_calls(ll, "llvm.exp2.f32"), expect + _fold(cfg)["exp2"])

    def test_fma_budget(self):
        # QK: block_n keys x gqa heads x ept terms. PV: gqa x ept x block_n.
        # Plus one per (sub-tile, head) for the l update. A drift here means
        # a redundant multiply crept into the innermost loop.
        cfg = _cfg()
        ll = _seg(cfg)
        per_sub = 2 * cfg.block_n * cfg.gqa_fuse * cfg.ept + cfg.gqa_fuse
        self.assertEqual(
            _calls(ll, "llvm.fmuladd.f32"),
            _subtiles(cfg) * per_sub + _fold(cfg)["fma"],
        )

    def test_param_abi(self):
        # The torch op packs arguments positionally; a reordering here is
        # silent at build time and catastrophic at run time.
        k = build_paged_decode_splitk_segment(_cfg(), arch="gfx1151")
        self.assertEqual(
            [p.name.lstrip("%") for p in k.params],
            [
                "Q",
                "KCache",
                "VCache",
                "BlockTable",
                "seq_lens",
                "ws_m",
                "ws_l",
                "ws_acc",
                "scale_log2",
                "stride_q_seq",
                "stride_q_head",
                "bt_stride",
                "bt_num_entries",
            ],
        )

    def test_one_wave_per_cta(self):
        k = build_paged_decode_splitk_segment(_cfg(), arch="gfx1151")
        self.assertEqual(k.attrs["max_workgroup_size"], WAVE)


class TestReduceCodegen(unittest.TestCase):
    def test_builds_and_lowers(self):
        for splits in (1, 8, 16):
            with self.subTest(splits=splits):
                self.assertIn("define", _red(_cfg(num_splits=splits)))

    def test_three_pass_nan_safe_form(self):
        # An all-empty split set (seq_len 0, or a request whose splits all
        # landed past the end) leaves every m at the -1e30 sentinel and every
        # l at 0. The streaming online-softmax merge computes rcp(0) = +inf
        # and then 0 * inf = NaN at the f16 trunc. The three-pass form
        # discards the sentinel with a select instead, so BOTH guards must
        # survive: the per-split `m > -1e30` and the `expsum == 0` short.
        cfg = _cfg()
        ll = _red(cfg)
        self.assertGreaterEqual(len(re.findall(r"fcmp ogt", ll)), cfg.gqa_fuse)
        self.assertGreaterEqual(len(re.findall(r"fcmp oeq", ll)), cfg.gqa_fuse)
        self.assertGreaterEqual(len(re.findall(r"= select", ll)), 2 * cfg.gqa_fuse)

    def test_acc_read_and_output_write_are_vectorised(self):
        cfg = _cfg()
        ll = _red(cfg)
        # ept_reduce, not ept: the reduce has no key axis to trade lanes
        # against, so it spreads D over the whole wave whatever d_lanes the
        # segment kernel used.
        self.assertEqual(
            len(re.findall(rf"load <{cfg.ept_reduce} x float>", ll)), cfg.gqa_fuse
        )
        self.assertEqual(
            len(re.findall(rf"store <{cfg.ept_reduce} x half>", ll)), cfg.gqa_fuse
        )

    def test_param_abi(self):
        k = build_paged_decode_splitk_reduce(_cfg(), arch="gfx1151")
        self.assertEqual(
            [p.name.lstrip("%") for p in k.params],
            ["ws_m", "ws_l", "ws_acc", "O", "stride_o_seq", "stride_o_head"],
        )


class TestIRWellFormed(unittest.TestCase):
    """The IR the other tests inspect must also be something LLVM will accept.

    Both kernels unroll over the GF fused heads inside ONE function scope, so
    any iter-arg name that forgets its ``g`` suffix silently emits GF
    definitions of the same local and clang rejects the module. That failure
    surfaces only at ``compile_kernel`` time, on a board -- this catches it on
    the laptop instead.
    """

    def test_no_duplicate_locals(self):
        for splits in (1, 8, 16):
            cfg = _cfg(num_splits=splits)
            for phase, ll in (("seg", _seg(cfg)), ("reduce", _red(cfg))):
                self.assertEqual(_dup_locals(ll), [], f"{phase} splits={splits}")


class TestLaunchGeometry(unittest.TestCase):
    def test_grids(self):
        cfg = _cfg(num_splits=8)
        self.assertEqual(paged_decode_segment_grid(cfg, 4), (4, _HK, 8))
        self.assertEqual(paged_decode_reduce_grid(cfg, 4), (4, _HK, 1))

    def test_workspace_shapes(self):
        cfg = _cfg(num_splits=8)
        ml, acc = paged_decode_workspace_shapes(cfg, 32)
        self.assertEqual(ml, (32, _HK, 8, 4))
        self.assertEqual(acc, (32, _HK, 8, 4, _D))
        # 8.4 MB at B=32 / splits=16 is the sizing claim the backend allocates
        # against; keep it honest.
        big_ml, big_acc = paged_decode_workspace_shapes(_cfg(num_splits=16), 32)
        nbytes = 4 * (2 * _prod(big_ml) + _prod(big_acc))
        self.assertLess(nbytes, 16 << 20)

    def test_split_heuristic_holds_cta_count(self):
        # The measured optimum is a near-constant ~256 phase-1 CTAs at every
        # batch (see the table on _TARGET_CTAS), so the heuristic's job is to
        # hold batch * kv_heads * splits at that number and then back off to 1
        # once batch alone clears it -- that path carries the win this kernel
        # exists for.
        for batch, want in ((1, 32), (2, 16), (4, 8), (8, 4), (16, 2), (32, 1)):
            self.assertEqual(choose_num_splits(batch, _HK), want, batch)
        for batch in (1, 2, 3, 4, 8, 16, 32, 64):
            n = choose_num_splits(batch, _HK)
            self.assertTrue(n > 0 and (n & (n - 1)) == 0, n)
            self.assertLessEqual(n, 32)


class TestLaneRemap(unittest.TestCase):
    """``d_lanes`` trades cross-lane reduce cost against register pressure.

    The whole point of the knob is the first of those, so the swizzle count is
    asserted exactly. The second cannot be seen in LLVM IR -- VGPR allocation
    happens in the backend -- which is why the choice between these configs is
    settled on the board and not here.
    """

    # (d_lanes, block_n, ds_swizzle per 16-key physical block)
    _CASES = ((32, 8, 20 * 16), (16, 8, 8 * 16), (8, 4, 3 * 16))

    def test_swizzle_cost_per_key_falls_as_designed(self):
        for d_lanes, block_n, want_loop in self._CASES:
            cfg = _cfg(d_lanes=d_lanes, block_n=block_n)
            with self.subTest(d_lanes=d_lanes):
                ll = _seg(cfg)
                total = _calls(ll, "llvm.amdgcn.ds.swizzle")
                self.assertEqual(total - _fold(cfg)["swizzle"], want_loop)
                # Every mask stays under 32, so none of this may lower to the
                # wave64 cross-half path.
                self.assertNotIn("bpermute", ll)

    def test_gathers_stay_vectorised(self):
        # K is one <min(ept, x) x half> run per token per lane and V one
        # <block_n x half> per lane-owned d. Scalar loads here mean the lane
        # map stopped agreeing with the cache layout.
        for d_lanes, block_n, _ in self._CASES:
            cfg = _cfg(d_lanes=d_lanes, block_n=block_n)
            with self.subTest(d_lanes=d_lanes):
                ll = _seg(cfg)
                widths = re.findall(r"load <(\d+) x half>", ll)
                self.assertNotIn("1", widths)
                subtiles = cfg.keys_per_sub // cfg.block_n
                k_chunks = max(1, cfg.ept // cfg.x)
                k_width = min(cfg.ept, cfg.x)
                # Q: gqa_fuse rows x k_chunks. K: subtiles x block_n x k_chunks.
                q_and_k = (cfg.gqa_fuse + subtiles * cfg.block_n) * k_chunks
                self.assertEqual(
                    widths.count(str(k_width)),
                    q_and_k + (subtiles * cfg.ept if block_n == k_width else 0),
                )
                if block_n != k_width:
                    self.assertEqual(widths.count(str(block_n)), subtiles * cfg.ept)

    def test_no_waterfall_at_any_lane_split(self):
        for d_lanes, block_n, _ in self._CASES:
            with self.subTest(d_lanes=d_lanes):
                ll = _seg(_cfg(d_lanes=d_lanes, block_n=block_n))
                self.assertEqual(_calls(ll, "llvm.amdgcn.readfirstlane.i32"), 2)

    def test_lowers_cleanly_at_any_lane_split(self):
        for d_lanes, block_n, _ in self._CASES:
            for splits in (1, 8, 16):
                cfg = _cfg(d_lanes=d_lanes, block_n=block_n, num_splits=splits)
                with self.subTest(d_lanes=d_lanes, splits=splits):
                    for phase, ll in (("seg", _seg(cfg)), ("reduce", _red(cfg))):
                        self.assertIn("define", ll)
                        self.assertEqual(_dup_locals(ll), [], phase)

    def test_reduce_ignores_d_lanes(self):
        # The reduce reads a finished [.., D] row and has no key axis, so it
        # spreads D over the whole wave no matter how the segment split it.
        base = _red(_cfg())
        for d_lanes, block_n, _ in self._CASES:
            cfg = _cfg(d_lanes=d_lanes, block_n=block_n)
            with self.subTest(d_lanes=d_lanes):
                self.assertEqual(cfg.ept_reduce, _D // WAVE)
                self.assertEqual(
                    _red(cfg).replace(cfg.kernel_name("reduce"), ""),
                    base.replace(_cfg().kernel_name("reduce"), ""),
                )


class TuningKnobs(unittest.TestCase):
    """``unroll`` / ``heads_per_cta`` / ``waves_per_eu`` / ``reduce_unroll``.

    Every knob must leave the default config's kernels byte-for-byte alone and
    keep the single-wave, LDS-free, two-readfirstlane shape when enabled.
    """

    def test_defaults_keep_kernel_names(self):
        n = _cfg().kernel_name("seg")
        for tok in ("_U", "_HC", "_WE"):
            self.assertNotIn(tok, n)
        self.assertNotIn("_RU", _cfg().kernel_name("reduce"))

    def test_names_carry_non_default_knobs(self):
        cfg = _cfg(unroll=4, heads_per_cta=2, waves_per_eu=4, reduce_unroll=1)
        seg, red = cfg.kernel_name("seg"), cfg.kernel_name("reduce")
        for tok in ("_U4", "_HC2", "_WE4"):
            self.assertIn(tok, seg)
        self.assertIn("_RU", red)

    def test_validation(self):
        bad = (
            dict(unroll=3),
            dict(unroll=16),
            dict(heads_per_cta=3),  # does not divide the GQA ratio of 4
            dict(heads_per_cta=8),
            dict(waves_per_eu=17),
            dict(reduce_unroll=2),
            dict(reduce_unroll=1, num_splits=128),
        )
        for kw in bad:
            with self.subTest(**kw):
                self.assertFalse(is_valid_spec(_cfg(**kw))[0])
        for kw in (
            dict(unroll=8),
            dict(heads_per_cta=1),
            dict(heads_per_cta=2),
            dict(waves_per_eu=2),
            dict(reduce_unroll=1),
        ):
            with self.subTest(**kw):
                self.assertTrue(is_valid_spec(_cfg(**kw))[0])

    def test_head_groups_widen_the_grid(self):
        base = paged_decode_segment_grid(_cfg(num_splits=8), 2)
        grp = paged_decode_segment_grid(_cfg(num_splits=8, heads_per_cta=1), 2)
        self.assertEqual(base, (2, _HK, 8))
        self.assertEqual(grp, (2, _HK * (_HQ // _HK), 8))

    def test_unrolled_and_grouped_stay_single_wave_no_lds(self):
        for kw in (
            dict(unroll=2),
            dict(unroll=8),
            dict(heads_per_cta=1),
            dict(heads_per_cta=2, unroll=4),
            dict(waves_per_eu=4),
        ):
            with self.subTest(**kw):
                ll = _seg(_cfg(**kw))
                self.assertEqual(_dup_locals(ll), [])
                self.assertNotIn("addrspace(3)", ll)
                self.assertNotIn("s.barrier", ll)

    def test_unroll_adds_one_table_read_per_block(self):
        for u in (2, 4, 8):
            with self.subTest(unroll=u):
                ll = _seg(_cfg(unroll=u))
                self.assertEqual(_calls(ll, "llvm.amdgcn.readfirstlane.i32"), 1 + u)

    def test_waves_per_eu_emits_attribute(self):
        self.assertIn("amdgpu-waves-per-eu", _seg(_cfg(waves_per_eu=4)))
        self.assertNotIn("amdgpu-waves-per-eu", _seg(_cfg()))

    def test_reduce_unroll_is_straight_line(self):
        for ns in (1, 8, 32):
            with self.subTest(splits=ns):
                loop = _red(_cfg(num_splits=ns))
                flat = _red(_cfg(num_splits=ns, reduce_unroll=1))
                self.assertEqual(_dup_locals(flat), [])
                if ns > 1:
                    self.assertIn("phi ", loop)
                self.assertNotIn("phi ", flat)


def _prod(shape):
    out = 1
    for s in shape:
        out *= s
    return out


if __name__ == "__main__":
    unittest.main()
