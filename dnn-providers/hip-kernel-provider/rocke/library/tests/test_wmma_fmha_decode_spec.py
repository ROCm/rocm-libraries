# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU tests of the RDNA split-KV WMMA decode: spec validation and per-family
resolution, kernel names, ABI, LDS budget, IR markers, the module builder and the
host helpers (key range, pack groups with Hv != Hkv, row tiles, split count,
grids, workspace layout, fused-merge choice). The C port of the host helpers is
pinned to the same vectors in platform/tests/instances/wmma_fmha_decode_host.cpp.
"""

from __future__ import annotations

import unittest
from dataclasses import replace

from kernels.gfx1151.wmma_fmha_decode import (
    DECODE_TILE,
    MERGE_THREADS,
    WmmaFmhaDecodeSpec,
    build_wmma_fmha_decode,
    build_wmma_fmha_decode_merge,
    build_wmma_fmha_decode_module,
    is_valid_spec,
    resolve_wmma_fmha_decode_spec,
    wmma_fmha_decode_bucket,
    wmma_fmha_decode_fused_merge,
    wmma_fmha_decode_grid,
    wmma_fmha_decode_kv_range,
    wmma_fmha_decode_lds_bytes,
    wmma_fmha_decode_merge_grid,
    wmma_fmha_decode_merge_signature,
    wmma_fmha_decode_num_splits,
    wmma_fmha_decode_pack_groups,
    wmma_fmha_decode_row_tiles,
    wmma_fmha_decode_signature,
    wmma_fmha_decode_workspace,
)

_ARCHS = ("gfx11-generic", "gfx12-generic")
_G11, _G12 = _ARCHS


def _resolved(arch, **fields):
    return resolve_wmma_fmha_decode_spec(WmmaFmhaDecodeSpec(**fields), arch=arch)


class TestDecodeSpec(unittest.TestCase):
    def test_post_init_rejects(self):
        bad = (
            dict(head_size=96),
            dict(head_size=128, dtype="fp32"),
            dict(head_size=128, bias="f16"),
            dict(head_size=128, kv_dtype="fp8e5m2"),
            dict(head_size=128, layout="packed"),
            dict(head_size=128, layout="paged"),
            dict(head_size=128, layout="paged", page_block_size=24),
            dict(head_size=128, layout="ragged", page_block_size=16),
            dict(head_size=128, num_waves=3),
            dict(head_size=128, dv_split=3),
            dict(head_size=64, dv_split=4),  # 16-column slices: below one V load
            dict(head_size=128, q_mode="reload"),  # removed prototype knob
            dict(head_size=128, waves_per_eu=-1),
            dict(head_size=128, wmma_family="gfx10"),
            dict(head_size=128, scheduler_strategy="fastest"),
        )
        for fields in bad:
            with self.subTest(**fields), self.assertRaises(ValueError):
                WmmaFmhaDecodeSpec(**fields)

    def test_removed_prototype_knobs(self):
        for knob in ("kv_tile", "k_half_load", "split_interleave", "v_pad"):
            with self.subTest(knob=knob), self.assertRaises(TypeError):
                WmmaFmhaDecodeSpec(head_size=128, **{knob: 1})

    def test_resolution_per_family(self):
        want = {
            # (arch, head_size): dv_split, q_mode, v_store_early, prefetch, lockstep, scheduler
            (_G11, 64): (1, "reg", False, False, False, "max-ilp"),
            (_G11, 128): (1, "reg", False, False, False, "max-ilp"),
            (_G11, 256): (2, "lds", False, False, False, "max-ilp"),
            (_G12, 64): (1, "reg", True, True, True, "max-ilp"),
            (_G12, 128): (1, "reg", True, False, True, "max-ilp"),
            (_G12, 256): (4, "reg", True, True, True, None),
        }
        for (arch, hs), knobs in want.items():
            with self.subTest(arch=arch, head_size=hs):
                r = _resolved(arch, head_size=hs)
                got = (
                    r.dv_split,
                    r.q_mode,
                    r.v_store_early,
                    r.prefetch,
                    r.lockstep,
                    r.scheduler_strategy,
                )
                self.assertEqual(got, knobs)
                self.assertEqual(r.wmma_family, arch[:5])
                self.assertEqual(
                    resolve_wmma_fmha_decode_spec(r, arch=arch), r
                )  # idempotent
                other = _G12 if arch == _G11 else _G11
                with self.assertRaises(ValueError):
                    resolve_wmma_fmha_decode_spec(r, arch=other)
        # Feature knobs: FP8 prefetch on gfx11 below D256; v_cvt_pk_f32_fp8 on gfx12 except D128;
        # the libm tanh only on gfx11 D256 without FP8.
        self.assertTrue(_resolved(_G11, head_size=128, kv_dtype="fp8e4m3").prefetch)
        self.assertFalse(_resolved(_G11, head_size=256, kv_dtype="fp8e4m3").prefetch)
        self.assertEqual(
            [
                _resolved(_G12, head_size=h, kv_dtype="fp8e4m3").fp8_cvt
                for h in (64, 128, 256)
            ],
            [True, False, True],
        )
        self.assertFalse(
            _resolved(_G11, head_size=64, kv_dtype="fp8e4m3", fp8_cvt=True).fp8_cvt
        )
        self.assertFalse(_resolved(_G11, head_size=256, softcap=True).tanh_fast)
        self.assertTrue(
            _resolved(_G11, head_size=256, softcap=True, kv_dtype="fp8e4m3").tanh_fast
        )
        self.assertTrue(_resolved(_G12, head_size=256, softcap=True).tanh_fast)
        # Explicit knobs survive; "default" is the LLVM default scheduler.
        r = _resolved(
            _G12,
            head_size=128,
            dv_split=2,
            q_mode="lds",
            lockstep=False,
            scheduler_strategy="default",
        )
        self.assertEqual(
            (r.dv_split, r.q_mode, r.lockstep, r.scheduler_strategy),
            (2, "lds", False, None),
        )

    def test_kernel_names(self):
        self.assertEqual(
            _resolved(_G11, head_size=64).kernel_name(),
            "rocke_wmma_fmha_decode_d64_fp16_w4_dvs1_t16_qreg_khalf_vlate_nopf_si_maxilp",
        )
        self.assertEqual(
            _resolved(_G12, head_size=256, dtype="bf16", bias="q").kernel_name(),
            "rocke_wmma_fmha_decode_d256_bf16_w4_biasq_dvs4_t16_qreg_vearly_pf_si_ls_default",
        )
        base = WmmaFmhaDecodeSpec(head_size=128, dtype="bf16")
        variants = [
            base,
            replace(base, dtype="fp16"),
            replace(base, head_size=64),
            replace(base, bias="q"),
            replace(base, bias="f32"),
            replace(base, softcap=True),
            replace(base, sinks=True),
            replace(base, alibi=True),
            replace(base, qq_bias=True),
            replace(base, kv_dtype="fp8e4m3"),
            replace(base, layout="ragged"),
            replace(base, layout="paged", page_block_size=16),
            replace(base, layout="paged", page_block_size=8),
            replace(base, num_waves=2),
            replace(base, dv_split=2),
            replace(base, q_mode="lds"),
            replace(base, v_store_early=False),
            replace(base, prefetch=True),
            replace(base, lockstep=False),
            replace(base, scheduler_strategy="default"),
            replace(base, waves_per_eu=1),
            replace(base, kv_dtype="fp8e4m3", fp8_cvt=True),
            replace(base, softcap=True, tanh_fast=False),
        ]
        for arch in _ARCHS:
            # Distinct resolved specs (distinct IR) must have distinct names; a knob set
            # to its family's auto value resolves to the same spec and name.
            by_spec = {
                resolve_wmma_fmha_decode_spec(v, arch=arch): None for v in variants
            }
            names = [s.kernel_name() for s in by_spec]
            self.assertGreaterEqual(len(names), 18)
            self.assertEqual(len(set(names)), len(names), names)

    def test_lds_budget_and_validity(self):
        self.assertEqual(
            wmma_fmha_decode_lds_bytes(WmmaFmhaDecodeSpec(head_size=64), arch=_G11),
            17408,
        )
        self.assertEqual(
            wmma_fmha_decode_lds_bytes(WmmaFmhaDecodeSpec(head_size=256), arch=_G11),
            42240,
        )
        self.assertEqual(
            wmma_fmha_decode_lds_bytes(WmmaFmhaDecodeSpec(head_size=256), arch=_G12),
            17408,
        )
        for arch, hs, ok in (
            (_G11, 256, False),
            (_G11, 128, False),
            (_G11, 64, True),
            (_G12, 256, True),
            (_G12, 128, False),
            (_G12, 64, True),
            ("gfx950", 64, False),
        ):
            with self.subTest(arch=arch, head_size=hs):
                self.assertEqual(
                    is_valid_spec(
                        WmmaFmhaDecodeSpec(head_size=hs, num_waves=8), arch=arch
                    )[0],
                    ok,
                )
        with self.assertRaises(ValueError):
            build_wmma_fmha_decode(
                WmmaFmhaDecodeSpec(head_size=128, num_waves=8), arch=_G12
            )
        for builder in (build_wmma_fmha_decode, build_wmma_fmha_decode_merge):
            with self.assertRaises(ValueError):
                builder(WmmaFmhaDecodeSpec(head_size=64), arch="gfx950")
            with self.assertRaises(TypeError):
                builder(WmmaFmhaDecodeSpec(head_size=64), _G11)  # arch is keyword-only

    def test_signatures(self):
        names = [
            p["name"]
            for p in wmma_fmha_decode_signature(WmmaFmhaDecodeSpec(head_size=128))
        ]
        prefix = [
            "Q",
            "K",
            "V",
            "O",
            "LSE",
            "WS_O",
            "WS_LSE",
            "WS_CNT",
            "scale_log2",
            "seqlen_q",
            "seqlen_k",
            "num_query_heads",
            "num_kv_heads",
            "num_v_heads",
            "head_dim_q",
            "head_dim_v",
            "bottom_right",
            "window_left",
            "window_right",
            "write_lse",
            "num_splits",
        ]
        strides = [
            f"stride_{t}_{a}"
            for t in ("q", "k", "v", "o", "lse")
            for a in ("batch", "token", "head")
        ]
        self.assertEqual(names, prefix + strides)
        every = WmmaFmhaDecodeSpec(
            head_size=128,
            bias="f32",
            softcap=True,
            sinks=True,
            alibi=True,
            qq_bias=True,
            kv_dtype="fp8e4m3",
            layout="paged",
            page_block_size=16,
        )
        sig = wmma_fmha_decode_signature(every)
        self.assertEqual(
            [p["name"] for p in sig][len(prefix) + len(strides) :],
            [
                "softcap",
                "sink_ptr",
                "alibi_slopes_ptr",
                "qq_bias_ptr",
                "qq_bias_rows",
                "qq_bias_cols",
                "qq_bias_stride",
                "cu_seqlens_q",
                "seqused_k",
                "block_table",
                "block_table_stride",
                "stride_k_block",
                "stride_v_block",
                "num_blocks",
                "k_scale",
                "v_scale",
                "attn_bias_ptr",
                "bias_stride_b",
                "bias_stride_h",
                "bias_stride_q",
            ],
        )
        types = {p["name"]: p["type"] for p in sig}
        self.assertEqual(types["K"], "ptr<fp8e4m3,global>")
        self.assertEqual(types["attn_bias_ptr"], "ptr<f32,global>")
        ragged = [
            p["name"]
            for p in wmma_fmha_decode_signature(
                WmmaFmhaDecodeSpec(head_size=64, layout="ragged")
            )
        ]
        self.assertEqual(ragged[-2:], ["cu_seqlens_q", "cu_seqlens_k"])
        merge = [
            "O",
            "LSE",
            "WS_O",
            "WS_LSE",
            "seqlen_q",
            "num_query_heads",
            "head_dim_v",
            "num_splits",
            "write_lse",
        ] + [
            f"stride_{t}_{a}" for t in ("o", "lse") for a in ("batch", "token", "head")
        ]
        self.assertEqual(
            [
                p["name"]
                for p in wmma_fmha_decode_merge_signature(
                    WmmaFmhaDecodeSpec(head_size=64)
                )
            ],
            merge,
        )
        packed = WmmaFmhaDecodeSpec(head_size=64, layout="paged", page_block_size=16)
        self.assertEqual(
            [p["name"] for p in wmma_fmha_decode_merge_signature(packed)],
            merge + ["cu_seqlens_q"],
        )
        # The signature helpers match what the builders declare.
        for arch in _ARCHS:
            k = build_wmma_fmha_decode(every, arch=arch)
            self.assertEqual(
                tuple({"name": p.name, "type": p.type.name} for p in k.params), sig
            )

    def test_build_attrs_and_ir(self):
        from rocke.core.ir_serialize import serialize

        for arch, atom, sched in (
            (_G11, "wmma_f32_16x16x16_bf16", "max-ilp"),
            (_G12, "wmma_gfx12_f32_16x16x16_bf16", None),
        ):
            with self.subTest(arch=arch):
                spec = WmmaFmhaDecodeSpec(head_size=256, dtype="bf16", bias="q")
                k = build_wmma_fmha_decode(spec, arch=arch)
                self.assertEqual(
                    k.name, resolve_wmma_fmha_decode_spec(spec, arch=arch).kernel_name()
                )
                self.assertEqual(k.attrs["max_workgroup_size"], 128)
                self.assertEqual(k.attrs.get("scheduler_strategy"), sched)
                self.assertNotIn("waves_per_eu", k.attrs)
                text = serialize(k)
                self.assertIn(atom, text)
                self.assertIn("dec_lds", text)
                self.assertEqual(
                    "dec_q" in text, arch == _G11
                )  # gfx11 D256 stages Q in LDS
                self.assertIn("global_atomic_add", text)  # fused merge
        k = build_wmma_fmha_decode(
            WmmaFmhaDecodeSpec(head_size=64, waves_per_eu=2, num_waves=2), arch=_G12
        )
        self.assertEqual(tuple(k.attrs["waves_per_eu"]), (2, 2))
        self.assertEqual(k.attrs["max_workgroup_size"], 64)
        m = build_wmma_fmha_decode_merge(
            WmmaFmhaDecodeSpec(head_size=256, dtype="bf16"), arch=_G11
        )
        self.assertEqual(
            (m.name, m.attrs["max_workgroup_size"]),
            ("rocke_wmma_fmha_decode_merge_bf16", MERGE_THREADS),
        )

    def test_feature_off_ir_has_no_feature_ops(self):
        from rocke.core.ir_serialize import serialize

        for arch in _ARCHS:
            text = serialize(
                build_wmma_fmha_decode(WmmaFmhaDecodeSpec(head_size=128), arch=arch)
            )
            for op in ("math.tanh", "arith.sitofp", "cvt_pk_f32_fp8"):
                self.assertNotIn(op, text)

    def test_module_builder(self):
        from rocke.helpers import KernelModule

        for spec in (
            WmmaFmhaDecodeSpec(head_size=128),
            WmmaFmhaDecodeSpec(head_size=64, layout="ragged"),
        ):
            for arch in _ARCHS:
                with self.subTest(arch=arch, layout=spec.layout):
                    mod = build_wmma_fmha_decode_module(spec, arch=arch)
                    self.assertIsInstance(mod, KernelModule)
                    self.assertEqual(set(mod.companions), {"merge"})
                    self.assertEqual(
                        mod.primary.name,
                        resolve_wmma_fmha_decode_spec(spec, arch=arch).kernel_name(),
                    )
                    want = "rocke_wmma_fmha_decode_merge_fp16" + (
                        "_packed" if spec.packed else ""
                    )
                    self.assertEqual(mod.companions["merge"].name, want)


class TestDecodeHostHelpers(unittest.TestCase):
    def test_bucket(self):
        cases = {
            (64, 64): 64, (8, 8): None, (16, 16): 64, (16, 64): 64, (40, 56): 64, (64, 128): None,
            (72, 128): 128, (128, 64): 128, (128, 128): 128, (112, 128): 128, (136, 136): 256,
            (128, 256): None, (200, 136): 256, (256, 256): 256, (264, 8): None, (12, 16): None,
            (0, 64): None, (256, 8): 256,
        }  # fmt: skip
        for (dq, dv), want in cases.items():
            with self.subTest(dq=dq, dv=dv):
                self.assertEqual(
                    wmma_fmha_decode_bucket(head_dim_q=dq, head_dim_v=dv), want
                )

    def test_kv_range(self):
        cases = {
            (1, 4096, True, -1, 0): (0, 4096),
            (4, 100, True, 10, 0): (86, 100),
            (1, 8192, True, 127, 0): (8064, 8192),
            (3, 50, False, -1, -1): (0, 50),
            (5, 20, True, -1, 2): (0, 20),
            (100, 40, True, 5, 0): (0, 40),
            (8, 8, False, 0, 0): (0, 8),
        }
        for (sq, sk, br, wl, wr), want in cases.items():
            got = wmma_fmha_decode_kv_range(
                seqlen_q=sq,
                seqlen_k=sk,
                bottom_right=br,
                window_left=wl,
                window_right=wr,
            )
            self.assertEqual(got, want)

    def test_pack_groups_and_row_tiles(self):
        for (hq, hk, hv), want in {
            (64, 8, 8): 8,
            (32, 8, 4): 8,
            (32, 8, 16): 16,
            (16, 16, 16): 16,
        }.items():
            self.assertEqual(
                wmma_fmha_decode_pack_groups(
                    num_query_heads=hq, num_kv_heads=hk, num_v_heads=hv
                ),
                want,
            )
        for hq, hk, hv in ((12, 4, 6), (8, 3, 8), (0, 1, 1)):
            with self.assertRaises(ValueError):
                wmma_fmha_decode_pack_groups(
                    num_query_heads=hq, num_kv_heads=hk, num_v_heads=hv
                )
        cases = {
            (64, 8, 8, 1): 1,
            (64, 8, 8, 4): 2,
            (32, 8, 4, 3): 1,
            (32, 32, 32, 1): 1,
            (16, 2, 2, 5): 3,
            (32, 8, 16, 8): 1,
        }
        for (hq, hk, hv, sq), want in cases.items():
            got = wmma_fmha_decode_row_tiles(
                num_query_heads=hq, num_kv_heads=hk, num_v_heads=hv, seqlen_q=sq
            )
            self.assertEqual(got, want)

    def test_num_splits(self):
        problems = (
            (1, 8, 1, 8192, 60),
            (4, 8, 1, 4096, 32),
            (1, 32, 1, 2500, 48),
            (8, 8, 1, 128, 60),
            (1, 1, 1, 100000, 64),
            (2, 16, 2, 0, 40),
        )
        want = {
            (_G11, 64): (15, 2, 3, 1, 64, 1),
            (_G11, 128): (15, 2, 3, 1, 64, 1),
            (_G11, 256): (7, 1, 1, 1, 64, 1),
            (_G12, 64): (15, 2, 3, 1, 64, 1),
            (_G12, 128): (15, 2, 3, 1, 64, 1),
            (_G12, 256): (3, 1, 1, 1, 32, 1),
        }
        for (arch, hs), splits in want.items():
            spec = _resolved(arch, head_size=hs)
            got = tuple(
                wmma_fmha_decode_num_splits(
                    spec, batch=b, pack_groups=p, row_tiles=r, kv_len=k, num_cus=c
                )
                for b, p, r, k, c in problems
            )
            self.assertEqual(got, splits, (arch, hs))
        spec = _resolved(_G12, head_size=128, num_waves=2)
        self.assertEqual(
            wmma_fmha_decode_num_splits(
                spec,
                batch=1,
                pack_groups=8,
                row_tiles=1,
                kv_len=8192,
                num_cus=60,
                wg_per_cu=3,
                min_tiles_per_wave=2,
            ),
            22,
        )
        with self.assertRaises(ValueError):  # dv_split is family-dependent
            wmma_fmha_decode_num_splits(
                WmmaFmhaDecodeSpec(head_size=128),
                batch=1,
                pack_groups=8,
                row_tiles=1,
                kv_len=8192,
                num_cus=60,
            )

    def test_grids_and_workspace(self):
        for arch, grid, total in ((_G11, 160, 926336), (_G12, 320, 926464)):
            spec = _resolved(arch, head_size=256)
            self.assertEqual(
                wmma_fmha_decode_grid(
                    spec, batch=2, pack_groups=8, row_tiles=1, num_splits=5
                ),
                (grid, 1, 1),
            )
            for kw in (
                dict(batch=0, pack_groups=8, row_tiles=1, num_splits=5),
                dict(batch=65536, pack_groups=64, row_tiles=64, num_splits=64),
            ):
                with self.assertRaises(ValueError):
                    wmma_fmha_decode_grid(spec, **kw)
            heads = dict(
                num_query_heads=32, num_kv_heads=8, num_v_heads=4
            )  # Hv != Hkv: 8 pack groups
            ws = wmma_fmha_decode_workspace(
                spec, batch=2, seqlen_q=3, head_dim_v=200, num_splits=6, **heads
            )
            rows = 2 * 32 * 3 * 6
            self.assertEqual(
                ws,
                {
                    "ws_o": 0,
                    "ws_lse": rows * 200 * 4,
                    "ws_cnt": rows * 201 * 4,
                    "bytes": total,
                },
            )
            self.assertEqual(
                total - ws["ws_cnt"], 4 * 2 * 8 * spec.dv_split * 1
            )  # one counter per group
            self.assertEqual(
                wmma_fmha_decode_workspace(
                    spec, batch=2, seqlen_q=3, head_dim_v=200, num_splits=1, **heads
                )["bytes"],
                0,
            )
            with self.assertRaises(ValueError):
                wmma_fmha_decode_workspace(
                    spec,
                    batch=4096,
                    num_query_heads=64,
                    num_kv_heads=8,
                    num_v_heads=8,
                    seqlen_q=64,
                    head_dim_v=256,
                    num_splits=64,
                )
        self.assertEqual(
            wmma_fmha_decode_merge_grid(batch=3, num_query_heads=32, seqlen_q=4),
            (384, 1, 1),
        )
        self.assertEqual(
            [wmma_fmha_decode_fused_merge(n) for n in range(7)],
            [False, False, True, True, True, False, False],
        )
        self.assertEqual(DECODE_TILE, 16)


if __name__ == "__main__":
    unittest.main()
