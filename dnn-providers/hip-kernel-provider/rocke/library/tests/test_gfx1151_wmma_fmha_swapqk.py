# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1151 transposed-QK WMMA FMHA-forward (swapqk) tests -- CPU only.

Covers the surface that does not need a gfx1151 device:
  * the shipped ``SwapQKCfg`` defaults ARE the measured-winning configuration
    (the regression guard that matters most: the kernel's whole value is that
    one config, and a default silently drifting off it is invisible at runtime);
  * build + lower for the production shape and the levers that change codegen
    structure (row-major V, causal, D64, persistent grid);
  * ``is_valid_spec`` rejects the shapes the kernel cannot emit;
  * the launch-geometry helpers and the V-relay layout transform;
  * the register budget, asserted from a real code-object compile.

The kernel COMPILES on any host (comgr targets gfx1151 regardless of the build
GPU); only execution needs the board, so nothing here is GPU-gated. The compile
and disassembly cases skip when the toolchain is unavailable.

    python3 -m pytest library/tests/test_gfx1151_wmma_fmha_swapqk.py
"""

from __future__ import annotations

import unittest

from kernels.gfx1151.wmma_fmha_swapqk import (
    SwapQKCfg,
    build_wmma_fmha_swapqk,
    is_valid_spec,
    swapqk_causal_kv_stop,
    swapqk_grid,
    swapqk_num_work_items,
    swapqk_transpose_v,
)

# The production shape the kernel was tuned and hardware-validated on.
_D = 128
_HQ = 24


def _prod_cfg(**kw) -> SwapQKCfg:
    kw.setdefault("head_size", _D)
    kw.setdefault("num_query_heads", _HQ)
    return SwapQKCfg(**kw)


class TestSwapQKDefaults(unittest.TestCase):
    """The defaults are the product. Pin them."""

    def test_defaults_are_the_measured_winner(self):
        # This kernel ships exactly one configuration; every knob below was
        # chosen by hardware A/B (see the field docs). A default drifting off
        # this set still builds and still computes correct attention, so it
        # would pass every numeric test while quietly costing throughput --
        # which is precisely how an earlier production wrapper ended up
        # building block_n=32 with v_transposed and qk_douter off.
        cfg = _prod_cfg()
        self.assertEqual(cfg.n_waves, 2)
        self.assertEqual(cfg.block_n, 64)
        self.assertEqual(cfg.qk_ilp, 2)
        self.assertEqual(cfg.q_block, 1)
        self.assertEqual(cfg.sched_mode, "pingpong")
        self.assertTrue(cfg.buffer_gather)
        self.assertTrue(cfg.dual_gather)
        self.assertTrue(cfg.lazy_rescale)
        self.assertTrue(cfg.fast_exp2)
        self.assertTrue(cfg.v_transposed)
        # qk_douter's documented +3.3% did not reproduce and its sign was wrong:
        # -4.7% at H24 MHA dense and -35.7% at Hq32/Hk8 GQA causal, both
        # confirmed in SQ_BUSY_CYCLES at an identical SQ_WAVES.
        self.assertFalse(cfg.qk_douter)
        # gqa_fuse is opt-in. It is a large win where it applies but it widens
        # the CTA, so callers choose it rather than inheriting it silently.
        self.assertEqual(cfg.gqa_fuse, 1)

    def test_experimental_levers_default_off(self):
        # Everything with a recorded regression stays off, so the default build
        # is the winner rather than the newest lever someone was measuring.
        cfg = _prod_cfg()
        for knob in (
            "pipeline",
            "q_hoist",
            "q_lds",
            "kv_lds",
            "o_f16",
            "d16hi",
            "o_nt",
            "q_nt",
            "static_shape",
            "prefetch_v",
            "k_dual",
            "k_lds",
        ):
            self.assertFalse(getattr(cfg, knob), f"{knob} must default off")
        self.assertEqual(cfg.v_kblock, 0)
        self.assertEqual(cfg.v_prefetch, 0)
        self.assertEqual(cfg.bcast_group, 0)
        self.assertEqual(cfg.num_persistent, 0)
        self.assertEqual(cfg.iglp, -1)
        self.assertIsNone(cfg.waves_per_eu)

    def test_default_cfg_is_valid(self):
        ok, why = is_valid_spec(_prod_cfg())
        self.assertTrue(ok, why)

    def test_kernel_name_encodes_the_winning_knobs(self):
        # The name is how a hsaco on the board is traced back to its config, so
        # the winning tokens have to survive into it.
        name = _prod_cfg().kernel_name()
        for token in ("H128", "HQ24", "w2", "pingpong", "ilp2", "bn64"):
            self.assertIn(token, name)
        for token in ("vt", "dual", "buf", "lazy", "fexp", "qkno"):
            self.assertIn(f"_{token}", name)
        # F==1 names stay byte-identical to the pre-fusion kernel.
        self.assertNotIn("gf", name)

    def test_fused_and_unfused_names_cannot_collide(self):
        # The artifact cache is keyed on this string. If a fused build shared a
        # key with the unfused one the cache would serve the wrong binary and
        # every measurement after that point would be a lie -- and it would
        # still PASS numerically, because both kernels compute correct
        # attention. Nothing else in the suite catches that.
        base = _prod_cfg(num_kv_heads=6).kernel_name()
        names = {base}
        for f in (2, 4):
            n = _prod_cfg(num_kv_heads=6, gqa_fuse=f).kernel_name()
            self.assertIn(f"gf{f}", n)
            self.assertNotIn(n, names)
            names.add(n)
        self.assertEqual(len(names), 3)


class TestSwapQKValidity(unittest.TestCase):
    def test_rejects_non_gfx1151_arch(self):
        ok, why = is_valid_spec(_prod_cfg(), arch="gfx1250")
        self.assertFalse(ok)
        self.assertIn("gfx1151", why)

    def test_rejects_head_size_not_multiple_of_32_under_dual_gather(self):
        # dual_gather pairs adjacent d-subtiles, so n_dk must be even.
        ok, why = is_valid_spec(_prod_cfg(head_size=48))
        self.assertFalse(ok)
        self.assertIn("head_size", why)

    def test_rejects_block_n_below_the_buffer_gather_floor(self):
        # The backend stops batching the buffer gather below block_n=32.
        ok, why = is_valid_spec(_prod_cfg(block_n=16))
        self.assertFalse(ok)
        self.assertIn("block_n", why)

    def test_rejects_block_n_off_the_16_grid(self):
        ok, _ = is_valid_spec(_prod_cfg(block_n=48))
        self.assertTrue(ok)  # 48 is a legal multiple of 16 at/above the floor
        ok, why = is_valid_spec(_prod_cfg(block_n=40))
        self.assertFalse(ok)
        self.assertIn("block_n", why)

    def test_rejects_unsupported_wave_count_and_mask(self):
        ok, why = is_valid_spec(_prod_cfg(n_waves=4))
        self.assertFalse(ok)
        self.assertIn("n_waves", why)
        ok, why = is_valid_spec(_prod_cfg(mask_mode="sliding"))
        self.assertFalse(ok)
        self.assertIn("mask_mode", why)

    def test_accepts_gqa_fuse_that_divides_the_ratio(self):
        # HQ24 / HK6 -> ratio 4, so F in {1,2,4}.
        for f in (1, 2, 4):
            ok, why = is_valid_spec(_prod_cfg(num_kv_heads=6, gqa_fuse=f))
            self.assertTrue(ok, f"gqa_fuse={f}: {why}")

    def test_rejects_gqa_fuse_that_does_not_divide_the_ratio(self):
        # A stale F would split a CTA's waves across two KV heads while the
        # kv_head decode still reads block_id_y -- silently wrong attention.
        for f in (3, 8):
            ok, why = is_valid_spec(_prod_cfg(num_kv_heads=6, gqa_fuse=f))
            self.assertFalse(ok, f"gqa_fuse={f} must be rejected")
            self.assertIn("gqa_fuse", why)

    def test_rejects_gqa_fuse_without_kv_heads(self):
        # Under MHA there is nothing to fuse; the ratio is 1.
        ok, why = is_valid_spec(_prod_cfg(gqa_fuse=2))
        self.assertFalse(ok)
        self.assertIn("num_kv_heads", why)

    def test_rejects_gqa_fuse_with_persistent_or_q_lds(self):
        # Both decode work-items or LDS slabs from wave_id / a one-head-per-CTA
        # work count, neither of which survives fusion.
        ok, why = is_valid_spec(
            _prod_cfg(num_kv_heads=6, gqa_fuse=4, num_persistent=960)
        )
        self.assertFalse(ok)
        self.assertIn("num_persistent", why)
        ok, why = is_valid_spec(_prod_cfg(num_kv_heads=6, gqa_fuse=4, q_lds=True))
        self.assertFalse(ok)
        self.assertIn("q_lds", why)

    def test_builder_raises_on_an_invalid_config(self):
        with self.assertRaises(ValueError):
            build_wmma_fmha_swapqk(_prod_cfg(block_n=24))

    def test_builder_rejects_incompatible_knob_pairs(self):
        # v_transposed rides the buffer-descriptor gather; without it there is
        # no path to emit, so this must fail loudly rather than mis-address V.
        with self.assertRaises(ValueError):
            build_wmma_fmha_swapqk(_prod_cfg(v_transposed=True, buffer_gather=False))


class TestSwapQKLowering(unittest.TestCase):
    def _lower(self, cfg):
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        return lower_kernel_to_llvm(
            build_wmma_fmha_swapqk(cfg, arch="gfx1151"), arch="gfx1151"
        )

    def test_production_config_lowers_to_the_rdna35_wmma_intrinsic(self):
        # gfx1151 has exactly one matrix atom; a K=32 intrinsic here would mean
        # the atom selection silently picked a gfx12 form.
        ll = self._lower(_prod_cfg())
        self.assertIn("llvm.amdgcn.wmma.f32.16x16x16.f16", ll)
        self.assertNotIn("16x16x32", ll)

    def test_production_config_uses_the_buffer_descriptor_d16_gather(self):
        # The buffer gather is worth a double-digit percentage and is fragile:
        # it only wins at w2/bn>=32. Assert the emitted form is the buffer one,
        # not the flat fallback, so a lowering change cannot quietly demote it.
        ll = self._lower(_prod_cfg())
        self.assertIn("raw.ptr.buffer.load", ll)

    def test_lane_broadcast_is_emitted_for_the_dual_subtile_gather(self):
        ll = self._lower(_prod_cfg())
        self.assertIn("permlanex16", ll)

    def test_row_major_v_also_lowers(self):
        # The documented escape hatch for callers that cannot pre-transpose V.
        ll = self._lower(_prod_cfg(v_transposed=False))
        self.assertIn("llvm.amdgcn.wmma.f32.16x16x16.f16", ll)

    def test_causal_and_gqa_lower(self):
        ll = self._lower(_prod_cfg(mask_mode="causal", num_kv_heads=4))
        self.assertIn("llvm.amdgcn.wmma.f32.16x16x16.f16", ll)

    def test_d64_lowers(self):
        ll = self._lower(SwapQKCfg(head_size=64, num_query_heads=8))
        self.assertIn("llvm.amdgcn.wmma.f32.16x16x16.f16", ll)

    def test_persistent_grid_lowers_and_needs_seqlen_at_build_time(self):
        # num_persistent bakes the work-item count in, so the builder must
        # refuse to guess it.
        cfg = _prod_cfg(num_persistent=960)
        with self.assertRaises(ValueError):
            build_wmma_fmha_swapqk(cfg, arch="gfx1151")
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        sized = _prod_cfg(num_persistent=960, persist_seqlen_q=2048, persist_batch=1)
        ll = lower_kernel_to_llvm(
            build_wmma_fmha_swapqk(sized, arch="gfx1151"), arch="gfx1151"
        )
        self.assertIn("atomicrmw", ll)
        # The baked-in size is part of the cache key, or a 2048-row binary gets
        # served for a 4096-row launch.
        self.assertNotEqual(
            sized.kernel_name(),
            _prod_cfg(
                num_persistent=960, persist_seqlen_q=4096, persist_batch=1
            ).kernel_name(),
        )

    def test_gqa_fuse_lowers_and_keeps_the_head_decode_scalar(self):
        # workitem.id.x is a divergence source, so LLVM cannot prove tid/32 is
        # wave-uniform; without the forced readfirstlane the head and query-row
        # addressing both go vector and the win evaporates. Assert the pin
        # survived into the IR rather than trusting the builder call.
        base = _prod_cfg(num_kv_heads=6, mask_mode="causal")
        ll = self._lower(_prod_cfg(num_kv_heads=6, gqa_fuse=4, mask_mode="causal"))
        self.assertIn("llvm.amdgcn.wmma.f32.16x16x16.f16", ll)
        self.assertGreater(
            ll.count("readfirstlane"), self._lower(base).count("readfirstlane")
        )

    def test_k_lds_stages_k_in_lds_while_v_keeps_the_buffer_gather(self):
        # The single assertion that separates k_lds from its kv_lds predecessor:
        # K moves to addrspace(3) and V does NOT. If a later edit made k_lds fall
        # through to the flat V path it would still compile and still be numerically
        # correct, and it would re-run the experiment that already lost 3x.
        ll = self._lower(
            _prod_cfg(num_kv_heads=6, gqa_fuse=4, mask_mode="causal", k_lds=True)
        )
        self.assertIn("addrspace(3)", ll)
        self.assertIn("llvm.amdgcn.s.barrier", ll)
        self.assertIn("raw.ptr.buffer.load", ll)  # V still on the gather
        base = self._lower(_prod_cfg(num_kv_heads=6, gqa_fuse=4, mask_mode="causal"))
        self.assertNotIn("addrspace(3)", base)

    def test_distinct_configs_build_distinct_kernels(self):
        a = build_wmma_fmha_swapqk(_prod_cfg(), arch="gfx1151")
        b = build_wmma_fmha_swapqk(_prod_cfg(block_n=32), arch="gfx1151")
        self.assertNotEqual(a.name, b.name)


class TestSwapQKLaunchGeometry(unittest.TestCase):
    def test_grid_is_q_blocks_by_heads_by_batch(self):
        cfg = _prod_cfg()
        # w2 q_block1 -> 32 query rows per CTA.
        self.assertEqual(swapqk_grid(cfg, seqlen_q=16384, batch=1), (512, _HQ, 1))
        self.assertEqual(swapqk_grid(cfg, seqlen_q=2048, batch=3), (64, _HQ, 3))

    def test_single_wave_halves_the_rows_per_cta(self):
        self.assertEqual(
            swapqk_grid(_prod_cfg(n_waves=1), seqlen_q=1024, batch=1), (64, _HQ, 1)
        )

    def test_work_item_count_matches_the_one_shot_grid(self):
        cfg = _prod_cfg()
        qb, h, b = swapqk_grid(cfg, seqlen_q=4096, batch=2)
        self.assertEqual(swapqk_num_work_items(cfg, seqlen_q=4096, batch=2), qb * h * b)

    def test_block_size_tracks_the_wave_count(self):
        self.assertEqual(_prod_cfg().block_size, 64)
        self.assertEqual(_prod_cfg(n_waves=1).block_size, 32)

    def test_kv_heads_defaults_to_mha(self):
        self.assertEqual(_prod_cfg().kv_heads, _HQ)
        self.assertEqual(_prod_cfg(num_kv_heads=4).kv_heads, 4)

    def test_gqa_fuse_widens_the_cta_without_widening_the_query_rows(self):
        # This is the whole design: the extra waves serve extra HEADS, so each
        # wave still owns exactly one head's 128-VGPR O accumulator and the
        # register budget is untouched. If q_rows_per_cta ever tracked gqa_fuse
        # the kernel would spill and the win would invert.
        base = _prod_cfg(num_kv_heads=6)
        for f in (1, 2, 4):
            cfg = _prod_cfg(num_kv_heads=6, gqa_fuse=f)
            self.assertEqual(cfg.block_size, base.block_size * f)
            self.assertEqual(cfg.q_rows_per_cta, base.q_rows_per_cta)

    def test_gqa_fuse_folds_heads_into_the_y_axis(self):
        # Total waves launched must be invariant -- that equality is the control
        # for every fusion A/B: an unmatched SQ_WAVES means the two arms did
        # different amounts of work and the comparison means nothing.
        base = _prod_cfg(num_kv_heads=6)
        qb, h, bz = swapqk_grid(base, seqlen_q=2048, batch=2)
        for f in (2, 4):
            cfg = _prod_cfg(num_kv_heads=6, gqa_fuse=f)
            g = swapqk_grid(cfg, seqlen_q=2048, batch=2)
            self.assertEqual(g, (qb, h // f, bz))
            self.assertEqual(
                g[0] * g[1] * g[2] * cfg.block_size,
                qb * h * bz * base.block_size,
            )


class TestSwapQKCausalTrim(unittest.TestCase):
    """The kv-block trim is the one place a wrong answer is silent.

    Over-inclusion costs a block of wasted work; the inline mask still zeroes
    it. Under-inclusion drops real attention weight from the LAST query rows of
    a CTA, which no shape check and no compile catches -- only a numeric
    comparison would, and only at the exact (n_waves, q_block, block_n) triple
    that trips it.
    """

    def _cover(self, cfg, q_group: int) -> bool:
        """Does the trim reach the diagonal of this CTA's last query row?"""
        last_row = q_group * cfg.q_rows_per_cta + cfg.q_rows_per_cta - 1
        return swapqk_causal_kv_stop(cfg, q_group) > last_row // cfg.block_n

    def test_trim_covers_every_query_row_across_the_knob_grid(self):
        for n_waves in (1, 2):
            for q_block in (1, 2):
                for block_n in (16, 32, 48, 64):
                    cfg = _prod_cfg(
                        mask_mode="causal",
                        n_waves=n_waves,
                        q_block=q_block,
                        block_n=block_n,
                    )
                    for g in range(8):
                        self.assertTrue(
                            self._cover(cfg, g),
                            f"w{n_waves} qb{q_block} bn{block_n} g{g} drops a kv block",
                        )

    def test_trim_regression_at_the_config_the_old_form_broke(self):
        # The pre-fix form used the wave span 16*n_waves and omitted the q_block
        # factor. At w2/qb2/bn16 that returns 4g+3 where 4g+4 is required.
        cfg = _prod_cfg(mask_mode="causal", n_waves=2, q_block=2, block_n=16)
        self.assertEqual(cfg.q_rows_per_cta, 64)
        for g in range(4):
            self.assertEqual(swapqk_causal_kv_stop(cfg, g), 4 * g + 4)

    def test_trim_is_tight_not_merely_safe(self):
        # A correct-but-loose trim would pass the coverage test above while
        # quietly paying for kv blocks that are entirely above the diagonal.
        for block_n in (16, 32, 64):
            cfg = _prod_cfg(mask_mode="causal", block_n=block_n)
            for g in range(6):
                last_row = g * cfg.q_rows_per_cta + cfg.q_rows_per_cta - 1
                self.assertEqual(swapqk_causal_kv_stop(cfg, g), last_row // block_n + 1)


class TestSwapQKAdaptiveBlockN(unittest.TestCase):
    """The torch op picks ``(block_n, k_lds)`` from the key length.

    The divisibility half of this is a CORRECTNESS gate, not a perf knob: the kv
    loop bound is ``seqlen_k // block_n``, so a non-divisible launch drops the
    tail silently.
    """

    @staticmethod
    def _pick():
        from rocke.instances.gfx1151.wmma_fmha_swapqk import _pick_strategy

        return _pick_strategy

    def test_bn64_plus_k_lds_at_every_length(self):
        # bn64+k_lds measured fastest at every S tried, so the table is one row
        # and the long-sequence promotion to bn128 is gone.
        pick = self._pick()
        for s in (1024, 2048, 3072, 4096, 8192):
            self.assertEqual(pick(s), (64, True), f"seqlen_k={s}")

    def test_rejects_lengths_no_tile_divides(self):
        # 2080 is 32-aligned, which the old vLLM gate accepted while running
        # bn64 -- that silently dropped the last 32 keys.
        pick = self._pick()
        self.assertEqual(pick(2080)[0], 0)
        self.assertEqual(pick(48)[0], 0)

    def test_env_pin_still_honours_divisibility(self):
        import rocke.instances.gfx1151.wmma_fmha_swapqk as inst

        prev = inst._BLOCK_N
        try:
            inst._BLOCK_N = "128"
            self.assertEqual(inst._pick_strategy(8192)[0], 128)
            # A pin overrides the table but NOT divisibility: 576 is 64-aligned,
            # which "auto" would happily serve, and the pin must not.
            self.assertEqual(inst._pick_strategy(2048)[0], 128)
            self.assertEqual(inst._pick_strategy(576)[0], 0)
        finally:
            inst._BLOCK_N = prev

    def test_k_lds_env_pin_gives_an_ab_control_arm(self):
        # Every A/B in the k_lds write-up was run through this pin; without it
        # the two arms cannot be compared in one process.
        import rocke.instances.gfx1151.wmma_fmha_swapqk as inst

        prev = inst._K_LDS
        try:
            inst._K_LDS = "0"
            self.assertEqual(inst._pick_strategy(8192), (64, False))
            inst._K_LDS = "1"
            self.assertEqual(inst._pick_strategy(8192), (64, True))
        finally:
            inst._K_LDS = prev

    def test_the_two_tiles_cannot_share_a_cache_key(self):
        # An artifact cache collision here would serve bn64 code for a bn128
        # config, making every measurement downstream a lie.
        self.assertNotEqual(
            _prod_cfg(block_n=64).kernel_name(),
            _prod_cfg(block_n=128).kernel_name(),
        )


class TestSwapQKKLds(unittest.TestCase):
    """k_lds stages K (and only K) in shared LDS.

    Its predecessor kv_lds staged K AND V and lost 3x; the two things that make
    this a different bet -- V staying on the buffer gather, and the tile being
    shared by a gqa_fuse-widened CTA -- are exactly the two things a later edit
    could undo while everything still compiled and still computed correct
    attention. So they are pinned here rather than left to the field docs.
    """

    LDS_PER_CU = 64 * 1024

    @staticmethod
    def _fused(**kw):
        return _prod_cfg(num_kv_heads=6, gqa_fuse=4, k_lds=True, **kw)

    @staticmethod
    def _lds_bytes(cfg) -> int:
        # one padded K tile: block_n rows of (head_size + 8) f16
        return cfg.block_n * (cfg.head_size + 8) * 2

    def test_fused_k_lds_config_is_valid(self):
        ok, why = is_valid_spec(self._fused())
        self.assertTrue(ok, why)

    def test_v_stays_on_the_buffer_gather(self):
        # The recorded kv_lds root cause "dsld 0->320" was ENTIRELY a V problem:
        # the flat LDS V read is 16 uncoalesced scalar ds_loads per fragment. If
        # k_lds ever forced v_transposed off it would inherit that and this whole
        # line of work would be the failed experiment again under a new name.
        cfg = self._fused()
        self.assertTrue(cfg.v_transposed)
        self.assertTrue(cfg.buffer_gather)
        self.assertTrue(cfg.dual_gather)
        ok, why = is_valid_spec(cfg)
        self.assertTrue(ok, why)

    def test_lds_budget_keeps_three_workgroups_resident(self):
        # 24 waves/CU is the occupancy the kernel is tuned at. bn64/D128 needs
        # 17 KB so 3 WGs still fit; anything that pushed it past ~21 KB would
        # silently drop to 2 WGs and cost a third of the latency hiding.
        cfg = self._fused()
        per_wg = self._lds_bytes(cfg)
        self.assertEqual(per_wg, 64 * 136 * 2)
        self.assertLessEqual(3 * per_wg, self.LDS_PER_CU)
        self.assertLessEqual(per_wg, 21 * 1024)

    def test_block_n_128_is_rejected_because_the_tile_evicts_occupancy(self):
        # 128*(128+8)*2 = 34 KB leaves room for ONE workgroup. This is why k_lds
        # is an ALTERNATIVE to block_n=128 at long sequences, not a stack with it.
        wide = self._fused(block_n=128)
        self.assertGreater(2 * self._lds_bytes(wide), self.LDS_PER_CU)
        ok, why = is_valid_spec(wide)
        self.assertFalse(ok)
        self.assertIn("block_n", why)

    def test_coop_loader_divides_evenly_over_the_cta(self):
        # Each thread stages whole vec8 chunks; a remainder would leave part of
        # the tile unwritten -- stale LDS, silently wrong scores.
        cfg = self._fused()
        self.assertEqual(cfg.block_size, 256)
        self.assertEqual((cfg.block_n * cfg.head_size) % (cfg.block_size * 8), 0)
        ok, why = is_valid_spec(
            _prod_cfg(head_size=64, block_n=64, k_lds=True, n_waves=2)
        )
        self.assertTrue(ok, why)  # 64*64 = 4096 over 64*8 = 512 -> 8 chunks

    def test_rejects_knobs_that_also_own_the_k_operand_or_the_tile_lifetime(self):
        for kw, token in (
            (dict(kv_lds=True), "kv_lds"),
            (dict(k_dual=True, qk_douter=True), "k_dual"),
            (dict(pipeline=True), "pipeline"),
            (dict(v_prefetch=2), "v_prefetch"),
            (dict(q_block=2), "q_block"),
        ):
            ok, why = is_valid_spec(self._fused(**kw))
            self.assertFalse(ok, f"{token} must be rejected with k_lds")
            self.assertIn(token, why)

    def test_rejects_persistent_which_would_allocate_lds_in_the_work_item_loop(self):
        ok, why = is_valid_spec(_prod_cfg(k_lds=True, num_persistent=960))
        self.assertFalse(ok)
        self.assertIn("num_persistent", why)

    def test_kv_lds_with_gqa_fuse_is_rejected(self):
        # The kv_lds coop loader sizes itself from n_waves alone, so in a
        # 256-thread fused CTA threads 64-255 stage rows past block_n and corrupt
        # LDS. Reachable before k_lds existed; guarded now.
        ok, why = is_valid_spec(_prod_cfg(num_kv_heads=6, gqa_fuse=4, kv_lds=True))
        self.assertFalse(ok)
        self.assertIn("kv_lds", why)

    def test_kernel_name_gains_klds_and_is_otherwise_unchanged(self):
        # Same rule as gf: the artifact cache is keyed on this string, so an
        # LDS build must never be served a non-LDS binary -- while every
        # pre-k_lds name stays byte-identical.
        base = _prod_cfg(num_kv_heads=6, gqa_fuse=4)
        self.assertNotIn("klds", base.kernel_name())
        lds = _prod_cfg(num_kv_heads=6, gqa_fuse=4, k_lds=True)
        self.assertEqual(lds.kernel_name(), base.kernel_name() + "_klds")

    def test_builder_raises_on_the_incompatible_pairs(self):
        for kw in (dict(kv_lds=True), dict(pipeline=True), dict(v_prefetch=2)):
            with self.assertRaises(ValueError):
                build_wmma_fmha_swapqk(self._fused(**kw), arch="gfx1151")


class TestSwapQKPagedV(unittest.TestCase):
    """v_paged reads V straight out of a paged cache through a block table.

    The point is to DELETE the caller's ``v.permute(...).contiguous()``, which
    runs at a small fraction of achievable bandwidth and costs a few percent of
    decoder prefill time at S=8192 -- by using the fact that the paged cache
    already stores V as
    [num_blocks, kvh, hs, block_size], i.e. token-fastest, which is exactly the
    order the PV A-fragment wants.

    Two failure modes here are silent and expensive, so both are pinned:
      * falling off the 2 x dwordx4 gather onto the 16 x d16 row-major path,
        which already measured a WASH end-to-end at S=8192, the gather cost
        cancelling the permute it saves;
      * the block id staying in a VGPR, which makes AMDGPU wrap every V load in
        a 32-iteration waterfall loop. Both still compile and both still
        compute correct attention.
    """

    @staticmethod
    def _cfg(**kw):
        # The shipped production config, plus the lever under test.
        return _prod_cfg(
            num_kv_heads=6, mask_mode="causal", gqa_fuse=4, k_lds=True, **kw
        )

    def _lower(self, cfg):
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        return lower_kernel_to_llvm(
            build_wmma_fmha_swapqk(cfg, arch="gfx1151"), arch="gfx1151"
        )

    def test_paged_is_valid_and_lowers_for_the_production_config(self):
        for bs in (16, 32, 64, 128):
            cfg = self._cfg(v_paged=True, kv_block_size=bs)
            ok, why = is_valid_spec(cfg)
            self.assertTrue(ok, f"bs={bs}: {why}")
            self.assertIn("llvm.amdgcn.wmma.f32.16x16x16.f16", self._lower(cfg))

    def test_kernel_name_gains_vpg_and_is_otherwise_unchanged(self):
        # Same cache-key rule as gf/klds: a paged binary must never be served to
        # a contiguous build, since the two disagree about what the V pointer
        # even means -- and every pre-paging name must stay byte-identical.
        base = self._cfg()
        self.assertNotIn("vpg", base.kernel_name())
        names = {base.kernel_name()}
        for bs in (16, 32, 64):
            n = self._cfg(v_paged=True, kv_block_size=bs).kernel_name()
            self.assertEqual(n, base.kernel_name() + f"_vpg{bs}")
            names.add(n)
        self.assertEqual(len(names), 4)

    def test_abi_appends_exactly_the_paged_params_and_only_when_paged(self):
        # The backend packs arguments positionally. Appending is what lets the
        # non-paged arg pack stay byte-identical; an INSERTED param would
        # mis-address every kernel that is not under test here.
        base = build_wmma_fmha_swapqk(self._cfg(), arch="gfx1151")
        paged = build_wmma_fmha_swapqk(
            self._cfg(v_paged=True, kv_block_size=16), arch="gfx1151"
        )
        b_names = [p.name for p in base.params]
        p_names = [p.name for p in paged.params]
        self.assertEqual(p_names[: len(b_names)], b_names)
        self.assertEqual(
            p_names[len(b_names) :], ["VBlockTable", "bt_stride", "bt_num_entries"]
        )

    def test_rejects_the_knobs_that_also_own_the_v_operand(self):
        for kw, token in (
            (dict(v_transposed=False), "v_transposed"),
            (dict(v_kblock=8), "v_kblock"),
            (dict(kv_block_size=0), "kv_block_size"),
            (dict(kv_block_size=24), "kv_block_size"),
            (dict(kv_block_size=8), "kv_block_size"),
        ):
            cfg = self._cfg(v_paged=True, **{**dict(kv_block_size=16), **kw})
            ok, why = is_valid_spec(cfg)
            self.assertFalse(ok, f"{kw} must be rejected under v_paged")
            self.assertIn(token, why)

    def test_rejects_block_sizes_that_do_not_tile_the_kv_block(self):
        # Sub-tile ns must sit at a COMPILE-TIME block/token offset from the
        # tile's origin; if neither size divides the other that offset becomes a
        # runtime divide and the lookup count stops being static.
        ok, why = is_valid_spec(self._cfg(v_paged=True, kv_block_size=48))
        self.assertFalse(ok)
        self.assertIn("kv_block_size", why)

    def test_kv_block_size_without_v_paged_is_rejected(self):
        # Otherwise the field reads as configured while nothing consumes it.
        ok, why = is_valid_spec(self._cfg(kv_block_size=16))
        self.assertFalse(ok)
        self.assertIn("kv_block_size", why)

    def test_builder_raises_on_the_incompatible_pairs(self):
        for kw in (
            dict(v_transposed=False),
            dict(buffer_gather=False),
            dict(kv_lds=True),
            dict(v_kblock=8),
            dict(v_prefetch=2),
            dict(prefetch_v=True),
            dict(kv_block_size=24),
        ):
            cfg = self._cfg(v_paged=True, **{**dict(kv_block_size=16), **kw})
            with self.assertRaises(ValueError, msg=f"{kw} must raise"):
                build_wmma_fmha_swapqk(cfg, arch="gfx1151")

    def test_seqlen_k_leaves_the_v_address_but_not_the_loop_bound(self):
        # This is the whole change: the per-lane V term goes from d_col*seqlen_k
        # (a runtime multiply up to 2.08 MB) to d_col*block_size (a constant).
        # But seqlen_k must SURVIVE -- it still bounds the kv loop and feeds the
        # causal trim. Partial removal is the highest-probability bug here and
        # it is silent: dropping the bound just reads past the keys.
        ref = self._lower(self._cfg()).count("%seqlen_k")
        paged = self._lower(self._cfg(v_paged=True, kv_block_size=16)).count(
            "%seqlen_k"
        )
        self.assertGreater(ref, 10)
        self.assertLess(paged, 5, "seqlen_k still reaches the V address")
        self.assertGreater(paged, 0, "seqlen_k vanished -- the kv loop lost its bound")

    def test_the_gather_keeps_its_two_dwordx4_shape(self):
        # 33 v4i32 buffer loads, unchanged from the contiguous build. If paging
        # had demoted V to the row-major path this would collapse to hundreds of
        # d16 half-loads -- the exact regression that made v_transposed=False a
        # wash end-to-end.
        wide = "llvm.amdgcn.raw.ptr.buffer.load.v4i32"
        narrow = "llvm.amdgcn.raw.ptr.buffer.load.f16"
        ref = self._lower(self._cfg())
        self.assertEqual(ref.count(narrow), 0)
        for bs in (16, 32, 64):
            ll = self._lower(self._cfg(v_paged=True, kv_block_size=bs))
            self.assertEqual(ll.count(wide), ref.count(wide), f"bs={bs}")
            self.assertEqual(ll.count(narrow), 0, f"bs={bs}")
        # The contrast that gives the numbers above their meaning.
        self.assertGreater(
            self._lower(self._cfg(v_transposed=False)).count(narrow), 200
        )

    def test_block_ids_are_promoted_to_sgpr_so_no_waterfall_is_emitted(self):
        # An addrspace(1) load is a divergence source, so without the
        # readfirstlane the backend cannot prove the buffer soffset is
        # wave-uniform and wraps EVERY V load in a 32-iteration waterfall. That
        # is invisible in the load COUNT here only because the count is what
        # this test pins alongside it: the promotion must be present AND the
        # gather must not have multiplied.
        #
        # The budget is exact: +1 per DISTINCT block-table entry the tile spans
        # (block_n/bs of them, memoised), +2 for the CTA-uniform table row and
        # bound. Growth past that means a promotion moved inside the K loop.
        ref = self._lower(self._cfg()).count("readfirstlane")
        for bs, blocks in ((16, 4), (32, 2), (64, 1), (128, 1)):
            ll = self._lower(self._cfg(v_paged=True, kv_block_size=bs))
            self.assertEqual(ll.count("readfirstlane"), ref + blocks + 2, f"bs={bs}")

    def test_block_table_lookups_are_deduped_to_one_per_distinct_block(self):
        # block_n=64 spans 64/bs physical blocks, so that is how many lookups a
        # tile may issue. One per 16-key sub-tile (always 4) would be 4x the
        # scalar loads at bs=64 for identical addresses.
        import re

        for bs, expect in ((16, 4), (32, 2), (64, 1), (128, 1)):
            ll = self._lower(self._cfg(v_paged=True, kv_block_size=bs))
            n = len(re.findall(r"load i32, ptr addrspace\(1\)", ll))
            self.assertEqual(n, expect, f"bs={bs}")

    def test_paged_composes_with_k_lds(self):
        # k_lds is the confirmed ~11% TTFT win; paging V must stack with it, not
        # replace it. K in LDS, V still on the buffer gather.
        ll = self._lower(self._cfg(v_paged=True, kv_block_size=16))
        self.assertIn("addrspace(3)", ll)
        self.assertIn("llvm.amdgcn.s.barrier", ll)
        self.assertIn("llvm.amdgcn.raw.ptr.buffer.load.v4i32", ll)


class TestSwapQKVRelay(unittest.TestCase):
    """swapqk_transpose_v is the caller's contract for the default layout."""

    def test_transpose_maps_bshd_to_bhds_elementwise(self):
        np = __import__("numpy")
        v = np.arange(2 * 6 * 3 * 4, dtype=np.float16).reshape(2, 6, 3, 4)
        vt = swapqk_transpose_v(v)
        self.assertEqual(vt.shape, (2, 3, 4, 6))  # [B,S,H,D] -> [B,H,D,S]
        self.assertTrue(vt.flags["C_CONTIGUOUS"])
        for b, s, h, d in ((0, 0, 0, 0), (1, 5, 2, 3), (0, 3, 1, 2)):
            self.assertEqual(vt[b, h, d, s], v[b, s, h, d])

    def test_key_blocked_relay_shape_and_contents(self):
        np = __import__("numpy")
        kb = 2
        v = np.arange(1 * 4 * 2 * 3, dtype=np.float16).reshape(1, 4, 2, 3)
        vt = swapqk_transpose_v(v, kblock=kb)
        self.assertEqual(vt.shape, (1, 2, 4 // kb, 3, kb))  # [B,H,S/KB,D,KB]
        for s in range(4):
            self.assertEqual(vt[0, 1, s // kb, 2, s % kb], v[0, s, 1, 2])

    def test_key_blocked_relay_rejects_a_ragged_seqlen(self):
        np = __import__("numpy")
        v = np.zeros((1, 5, 2, 3), dtype=np.float16)
        with self.assertRaises(ValueError):
            swapqk_transpose_v(v, kblock=2)


class TestSwapQKCodeObject(unittest.TestCase):
    """Compile to a real code object; no GPU needed, comgr cross-targets."""

    def _compile(self, cfg):
        try:
            from rocke.helpers.compile import compile_kernel
        except Exception as e:  # pragma: no cover - env-dependent
            self.skipTest(f"compile toolchain unavailable: {e}")
        try:
            return compile_kernel(
                build_wmma_fmha_swapqk(cfg, arch="gfx1151"), arch="gfx1151"
            )
        except Exception as e:  # pragma: no cover - env-dependent
            self.skipTest(f"gfx1151 comgr compile unavailable: {e}")

    def test_production_config_compiles(self):
        self.assertGreater(self._compile(_prod_cfg()).hsaco_bytes, 0)

    def test_register_budget_holds(self):
        # The shipped config sits at 199 VGPR in a 208 granule = 7 waves/SIMD,
        # with nothing spilled. Crossing 208 costs a wave; spilling at all has
        # measured worse than the wave is worth. Both are invisible without
        # this assertion, so pin the budget rather than the exact count.
        art = self._compile(_prod_cfg())
        res = _resources(self, art)
        self.assertIsNotNone(res.vgpr_count, "no VGPR count in the code object")
        self.assertLessEqual(res.vgpr_count, 208, "lost a wave: VGPR past 7/SIMD")
        self.assertEqual(res.scratch_bytes or 0, 0, "spilled to scratch")

    def test_k_lds_allocates_the_PADDED_tile_and_does_not_spill(self):
        # The allocation is sized from the view's SHAPE, so carrying the bank pad
        # in explicit strides instead reserves block_n*head_size and then writes
        # block_n*(head_size+8) -- the last rows land past the end of the block's
        # LDS. Silent, and it corrupts whichever block is allocated next.
        cfg = _prod_cfg(num_kv_heads=6, gqa_fuse=4, mask_mode="causal", k_lds=True)
        res = _resources(self, self._compile(cfg))
        self.assertEqual(res.lds_bytes, 64 * (128 + 8) * 2)
        # 3 workgroups must still fit, and the coop loader must not have cost the
        # registers that sank its kv_lds predecessor (197 -> 256 + 16 B spill).
        self.assertLessEqual(3 * res.lds_bytes, 64 * 1024)
        self.assertLessEqual(res.vgpr_count, 232)
        self.assertEqual(res.scratch_bytes or 0, 0, "spilled to scratch")

    def test_d64_has_register_headroom(self):
        # D64 halves the O accumulator, which is what makes the otherwise
        # register-blocked levers (pipeline, q_block=2) fit there and not here.
        res = _resources(
            self, self._compile(SwapQKCfg(head_size=64, num_query_heads=8))
        )
        self.assertIsNotNone(res.vgpr_count)
        self.assertLess(res.vgpr_count, 192)
        self.assertEqual(res.scratch_bytes or 0, 0)


def _resources(case: unittest.TestCase, art):
    """Decode the code object's resource note, skipping if tools are missing."""
    import os
    import tempfile

    try:
        from rocke.analysis.isa import analyze_hsaco
    except Exception as e:  # pragma: no cover - env-dependent
        case.skipTest(f"isa analysis unavailable: {e}")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "k.hsaco")
        with open(path, "wb") as f:
            f.write(art.hsaco)
        try:
            return analyze_hsaco(path).resources
        except FileNotFoundError as e:  # pragma: no cover - env-dependent
            case.skipTest(f"disassembler unavailable: {e}")


if __name__ == "__main__":
    unittest.main()
