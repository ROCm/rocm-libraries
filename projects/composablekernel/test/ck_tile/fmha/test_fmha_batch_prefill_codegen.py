# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Batch prefill codegen: per-target kernel selection and the rendered dispatch (CPU only)."""

import itertools
import re
import sys
import unittest
from pathlib import Path

_CK_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_CK_ROOT / "example" / "ck_tile" / "01_fmha"))

from codegen.ops.fmha_batch_prefill import get_fwd_blobs  # noqa: E402

_PAGE_TILE = "qr_tdm_sched"
# 1920 M128 instances, 8 M64 small-grid instances (BF16, vectorized + vLLM, P64/P128,
# D64/D128, dense/masked, no sink/bias/soft cap/LSE), 10 KV-split instances (M64 split:
# the same set unmasked; M128 split: P128 D64, P64 D64, P64 D128, dense/masked) and 2
# query-tail instances (P64 D128, dense/masked).
_PAGE_TILE_KERNELS = 1940
_M64_SET = set(itertools.product(["bf16"], [64, 128], [64, 128], ["s_no", "s_mask"]))
_SPLIT_SET = {(d, p, 64, "s_no") for d in (64, 128) for p in (64, 128)}
_SPLIT_SET |= {
    (d, p, 128, m)
    for (d, p) in ((64, 128), (64, 64), (128, 64))
    for m in ("s_no", "s_mask")
}
_TAIL_SET = {(128, 64, 64, m) for m in ("s_no", "s_mask")}
# B = 1 routing shapes: query length, query heads, KV length.
_QS = (64, 129, 512, 2048, 2051, 2179, 8192)
_HQS = (1, 4, 8, 32, 64)
_KVS = (1024, 4096, 16384, 32768)
_WORKSPACE = "a.o_acc_ptr != nullptr && a.lse_acc_ptr != nullptr"
# Dispatch order in a head-dim case: KV-split M128, KV-split M64 and query tail, M64, M128.
_RANK = {(128, 2): 0, (64, 2): 1, (64, 1): 2, (128, 1): 3}


def _split_targs(trait: str):
    """(KV splits, query-tail rows) of a rendered dispatch trait."""
    m = re.search(r"LoadModeEnum::\w+(?:, (\d+))?(?:, (\d+))?>$", trait)
    return int(m.group(1) or 1), int(m.group(2) or 0)


def _pick(page, dim, dense, wg, kv, q):
    """Route of the measured configuration on a 256-CU gfx1250 (B = 1), as the dispatch
    implements it: KV-split M128 / KV-split M64 / query tail / M64 / M128."""
    if kv >= 16384 and page * dim != 128 * 128:
        if 128 < wg <= 256 or (dim == 64 and 512 < wg <= 576 and q % 128):
            return "split_m128"
        if dim == 128 and 512 < wg <= 576 and q % 128:
            return "tail"
    if dense and kv >= 4096:
        if wg <= 128 or (dim == 64 and kv >= 16384 and (wg <= 256 or 512 < wg <= 576)):
            return "split_m64"
    return "m64" if wg <= 256 else "m128"


def _holds(constraint, wg, kv, q, mask="no_mask", window_size_left=-1):
    """Value of a rendered dispatch constraint for a B = 1 problem with a workspace on a
    256-CU device; mask: the problem's mask_enum value."""
    e, bottom_right = constraint, str(mask == "mask_bottom_right")
    for cpp, value in (
        ("a.o_acc_ptr != nullptr", "True"),
        ("a.lse_acc_ptr != nullptr", "True"),
        ("t.mask_type == mask_enum::no_mask", str(mask == "no_mask")),
        ("t.mask_type == mask_enum::mask_bottom_right", bottom_right),
        ("a.window_size_left", str(window_size_left)),
        ("get_num_blocks(128)", str(wg)),
        ("num_cus", "256"),
        ("a.seqlen_k", str(kv)),
        ("a.max_seqlen_q", str(q)),
        ("a.batch", "1"),
        ("&&", " and "),
        ("||", " or "),
        ("true", "True"),
    ):
        e = e.replace(cpp, value)
    return eval(e)


def _measured_arm(t, dim, page, mask):
    """Dispatch trait of the measured page-tile set: BF16, vectorized K/V + vLLM."""
    return (
        (t.dtype, int(t.hdim), t.page_size, t.mask) == ("bf16", dim, page, mask)
        and (t.kv_memory_layout, t.kv_lookup_table) == ("vectorized", "vllm")
        and (t.lse, t.sink, t.bias, t.logits) == ("f", "f", "no", "f")
    )


def _blobs(targets, receipt=600, kernel_filter="", mask_impl="simplified"):
    return get_fwd_blobs(kernel_filter, receipt, [-1], mask_impl, targets)


def _arms(api: str):
    """One entry per rendered dispatch arm: (condition, trait)."""
    return re.findall(
        r"if\((\(t\.is_group_mode.*?)\) \{\s*using trait_ = ([^;]+);", api, re.S
    )


class BatchPrefillCodegenTest(unittest.TestCase):
    def test_gfx9_targets_keep_only_the_async_pipeline(self):
        pool, kernels = _blobs(["gfx942", "gfx950"])
        self.assertTrue(kernels)
        self.assertFalse([k for k in kernels if k.F_pipeline.tag == _PAGE_TILE])
        # No arch guard is added when only gfx9 is generated.
        self.assertFalse([k for k in kernels if "__gfx9__" in k.template])
        self.assertNotIn("gfx125", pool.api)

    def test_other_targets_generate_nothing(self):
        for targets in (["gfx1100"], ["gfx1200"], ["gfx1250", "gfx1100"]):
            pool, kernels = _blobs(targets)
            self.assertEqual(kernels, [], targets)
            self.assertNotIn("fmha_batch_prefill_<", pool.api)

    def test_gfx125_generates_the_page_tile_product(self):
        pool, kernels = _blobs(["gfx1250"])
        self.assertTrue(all(k.F_pipeline.tag == _PAGE_TILE for k in kernels))
        product = {
            (
                k.F_dtype,
                k.F_hdim,
                k.F_page_size,
                k.F_pipeline.F_mask,
                k.F_pipeline.F_lse,
                k.F_pipeline.F_sink,
                k.F_pipeline.F_logits,
                k.F_pipeline.F_bias,
                k.F_pipeline.F_kv_lookup_table,
                k.F_pipeline.F_kv_memory_layout,
            )
            for k in kernels
        }
        expected = {
            c
            for c in itertools.product(
                ["fp16", "bf16"],
                [64, 128],
                [16, 64, 128, 256, 1024],
                ["s_no", "s_mask"],
                ["t", "f"],
                ["t", "f"],
                ["t", "f"],
                ["no", "bias", "alibi"],
                ["vllm", "sglang"],
                ["linear", "vectorized"],
            )
            # sink instances are masked only; the soft cap is unbiased only
            if not (c[5] == "t" and c[3] == "s_no")
            and not (c[6] == "t" and c[7] != "no")
        }
        self.assertEqual(len(kernels), _PAGE_TILE_KERNELS)
        self.assertEqual(product, expected)
        for k in kernels:
            self.assertEqual(k.F_mode, "group")
            body = k.template
            self.assertIn("ck_tile::FmhaBatchPrefillQrTdmPageTileKernel<", body)
            # A KV-split instance launches through the split helper (its checked launch).
            if k.F_tile.F_kv_splits == 1:
                self.assertIn("k_::MakeCheckedKernel(kargs, grids)", body)
            self.assertIn("(defined(__gfx125__))", body)

    def test_gfx125_dispatch_conditions(self):
        pool, kernels = _blobs(["gfx1250"])
        api = pool.api
        arms = _arms(api)
        self.assertEqual(len(arms), len(kernels))
        # Page-tile arms take no kv_load_mode selection.
        self.assertNotIn("kElementBytes", api)
        for cond, trait in arms:
            self.assertIn('device_name.compare(0, 6, "gfx125") == 0', cond)
            self.assertIn("BlockFmhaPipelineEnum::QRKSVS_TDM_SCHED", trait)
            hdim = int(trait.split("<", 1)[1].split(",", 1)[0])
            self.assertIn(f"(a.hdim_q == {hdim}) && (a.hdim_v == {hdim})", cond)
            if "SimplifiedGenericAttentionMask<true>" in trait:
                # Any window from either corner: no window clause in the arm, except
                # for the KV split (bottom-right masks without a left window).
                self.assertIn("t.mask_type != mask_enum::no_mask", cond)
                if _split_targs(trait)[0] == 2:
                    self.assertIn("a.window_size_left < 0", cond)
                else:
                    self.assertNotIn("window_size", cond)
            else:
                self.assertIn("t.mask_type == mask_enum::no_mask", cond)
        # Head-dim cases ascend, so a D64 problem is not captured by the D128 case.
        self.assertLess(api.index("t.hdim_q <= 64 "), api.index("t.hdim_q <= 128 "))

    def test_gfx125_m64_small_grid_tiles(self):
        pool, kernels = _blobs(["gfx1250"])
        m64 = [k for k in kernels if k.F_tile.F_bm0 == 64 and k.F_tile.F_kv_splits == 1]
        self.assertEqual(
            {(k.F_dtype, k.F_hdim, k.F_page_size, k.F_pipeline.F_mask) for k in m64},
            _M64_SET,
        )
        self.assertEqual(len(m64), len(_M64_SET))
        for k in m64:
            p = k.F_pipeline
            self.assertEqual(
                (
                    p.F_kv_memory_layout,
                    p.F_kv_lookup_table,
                    p.F_sink,
                    p.F_bias,
                    p.F_logits,
                    p.F_lse,
                ),
                ("vectorized", "vllm", "f", "no", "f", "f"),
            )
            # D64 uses K64 PV stages (QK groups of 64 columns), D128 K32.
            self.assertEqual(k.F_tile.F_bk1, 64 if k.F_hdim == 64 else 32)
        arms = _arms(pool.api)
        for hdim in (64, 128):
            prefix = f"fmha_fwd_batch_prefill_traits_<{hdim}, FmhaFwdBf16, true, "
            pos = [
                i
                for i, (_, trait) in enumerate(arms)
                if trait.startswith(prefix) and _split_targs(trait) == (1, 0)
            ]
            m64_arms = [i for i in pos if arms[i][1].startswith(prefix + "64, ")]
            self.assertEqual(len(m64_arms), 4, hdim)
            # Unsplit: the M64 arms come before the M128 arms of their case and take
            # only grids of at most one M128 workgroup per CU; others fall through.
            self.assertEqual(m64_arms, pos[: len(m64_arms)], hdim)
            for i in m64_arms:
                self.assertIn("get_num_blocks(128) <= num_cus", arms[i][0])
            for i in pos[len(m64_arms) :]:
                self.assertNotIn("get_num_blocks(128)", arms[i][0])

    def test_gfx125_kv_split_tiles_and_routing(self):
        pool, kernels = _blobs(["gfx1250"])
        split = [
            k
            for k in kernels
            if k.F_tile.F_kv_splits == 2 and k.F_tile.F_query_tail_rows == 0
        ]
        self.assertEqual(
            {
                (k.F_hdim, k.F_page_size, k.F_tile.F_bm0, k.F_pipeline.F_mask)
                for k in split
            },
            _SPLIT_SET,
        )
        self.assertEqual(len(split), len(_SPLIT_SET))
        for k in split:
            body = k.template
            self.assertIn("fmha_batch_prefill_split_merge_kernel.hpp", body)
            self.assertIn("fmha_batch_prefill_split_launch<k_, m_>(s, a)", body)
            self.assertIn(", 2>;", body)
            constraint = str(k.F_tile.F_constraint)
            self.assertIn(_WORKSPACE, constraint)
            # A masked split takes bottom-right masks without a left window only.
            if k.F_pipeline.F_mask != "s_no":
                shape = (200, 32768, 25600)  # WG, KV, Q
                for mask, left in (("mask_top_left", -1), ("mask_bottom_right", 64)):
                    self.assertFalse(_holds(constraint, *shape, mask, left))
                self.assertTrue(_holds(constraint, *shape, "mask_bottom_right"))
        # Query tail: the M128 instance on the full tiles, the KV split of the rest.
        tails = [k for k in kernels if k.F_tile.F_query_tail_rows != 0]
        self.assertEqual(
            {
                (k.F_hdim, k.F_page_size, k.F_tile.F_bm0, k.F_pipeline.F_mask)
                for k in tails
            },
            _TAIL_SET,
        )
        self.assertEqual(len(tails), len(_TAIL_SET))
        for k in tails:
            body = k.template
            self.assertEqual(k.F_tile.F_query_tail_rows, 128)
            self.assertIn("tail_launch<fmha_kernel_1, k_, m_>(s, a)", body)
            self.assertIn("FmhaBatchPrefillQrTdmPageTileKernel<fmha_pipeline_0, ", body)
            self.assertIn("fmha_epilogue_0, 2, 128>;", body)
            self.assertIn("<fmha_pipeline_1, fmha_epilogue_1>;", body)
            self.assertIn("sequence<128, 128, 32, 128, 32, 128>", body)
            self.assertIn("::ODataType, 128, 128>;", body)
            constraint = str(k.F_tile.F_constraint)
            self.assertIn(_WORKSPACE + " && a.batch == 1", constraint)
        for hdim in (64, 128):
            prefix = f"fmha_fwd_batch_prefill_traits_<{hdim}, FmhaFwdBf16, true, "
            order = []
            for _, trait in _arms(pool.api):
                if trait.startswith(prefix):
                    bm0 = int(trait[len(prefix) :].split(",")[0])
                    order.append(_RANK[bm0, _split_targs(trait)[0]])
            self.assertEqual(order, sorted(order), hdim)
        # B = 1 shapes: the first arm whose condition holds is the measured route.
        traits = [k.api_trait() for k in kernels]
        for dim, page, mask in itertools.product(
            (64, 128), (64, 128), ("s_no", "s_mask")
        ):
            arms = sorted(
                (t for t in traits if _measured_arm(t, dim, page, mask)),
                key=lambda t: _RANK[t.bm0, t.kv_splits],
            )
            self.assertEqual(
                [t.query_tail_rows for t in arms if t.query_tail_rows],
                [128] if (dim, page) == (128, 64) else [],
            )
            problem_mask = "no_mask" if mask == "s_no" else "mask_bottom_right"
            for q, hq, kv in itertools.product(_QS, _HQS, _KVS):
                wg = -(-q // 128) * hq
                arm = next(
                    t
                    for t in arms
                    if _holds(str(t.constraint), wg, kv, q, problem_mask)
                )
                route = ("split_" if arm.kv_splits == 2 else "") + f"m{arm.bm0}"
                if arm.query_tail_rows:
                    route = "tail"
                want = _pick(page, dim, mask == "s_no", wg, kv, q)
                self.assertEqual(route, want, (dim, page, mask, q, hq, kv))

    def test_generic_mask_impl_still_uses_simplified_page_tile_masks(self):
        pool, kernels = _blobs(["gfx1250"], mask_impl="generic")
        self.assertEqual(len(kernels), _PAGE_TILE_KERNELS)
        self.assertFalse([k for k in kernels if "FmhaMasks::" in k.template])

    def test_mixed_build_guards_the_gfx9_kernels(self):
        _, gfx9 = _blobs(["gfx942"])
        pool, kernels = _blobs(["gfx942", "gfx1250"])
        page_tile = [k for k in kernels if k.F_pipeline.tag == _PAGE_TILE]
        old = [k for k in kernels if k.F_pipeline.tag != _PAGE_TILE]
        self.assertEqual(len(page_tile), _PAGE_TILE_KERNELS)
        self.assertEqual(sorted(k.name for k in old), sorted(k.name for k in gfx9))
        for k in old:
            self.assertRegex(k.template, r"defined\(__gfx9(42)?__\)")
        names = [k.filename for k in kernels]
        self.assertEqual(len(names), len(set(names)))
        for cond, trait in _arms(pool.api):
            if "QRKSVS_TDM_SCHED" in trait:
                self.assertIn('"gfx125"', cond)
            else:
                self.assertRegex(cond, r'device_name\.compare\(0, [46], "gfx9(42)?"\)')

    def test_aiter_receipts_and_filter_keep_the_page_tile_kernels(self):
        for receipt in (200, 600):
            _, kernels = _blobs(["gfx1250"], receipt, kernel_filter="*_ndropout_*")
            self.assertEqual(len(kernels), _PAGE_TILE_KERNELS, receipt)
        _, kernels = _blobs(["gfx1250"], 100)
        self.assertEqual(kernels, [])


if __name__ == "__main__":
    unittest.main()
