#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only tests of comp_tdm_scheduled in tile_engine and the dispatcher.

The tuning, the validation rules, the dispatcher codegen and the compile flags, and that the
tile_engine CMake builder rejects the pipeline (it runs through the dispatcher). No GPU or hipcc
is required.
"""

import itertools
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_DISP = Path(__file__).resolve().parent.parent
_CK = _DISP.parent
_GEMM = _CK / "tile_engine" / "ops" / "gemm"
_MX = _GEMM / "mx_gemm"
_HOST_TEST = _CK / "test" / "ck_tile" / "gemm_mx" / "tdm_scheduled"
_POLICY_HPP = (
    _CK
    / "include"
    / "ck_tile"
    / "ops"
    / "gemm"
    / "pipeline"
    / "mx_gemm_pipeline_ag_bg_cr_comp_tdm_scheduled_policy.hpp"
)
for _p in (_DISP / "python", _DISP / "codegen", _GEMM, _MX, _HOST_TEST):
    sys.path.insert(0, str(_p))

import gemm_validation_utils as vu  # noqa: E402
import mx_gemm_utils as mx  # noqa: E402
import parity_cases  # noqa: E402
import unified_mx_gemm_codegen as codegen  # noqa: E402

sched = vu.load_mx_tdm_scheduled()
BUILDER = _MX / "mx_gemm_instance_builder.py"
TILE = {
    "tile_m": 256,
    "tile_n": 256,
    "tile_k": 512,
    "warp_m": 2,
    "warp_n": 2,
    "warp_k": 1,
    "warp_tile_m": 32,
    "warp_tile_n": 16,
    "warp_tile_k": 128,
}


class TestTuning(unittest.TestCase):
    def test_unknown_key(self):
        with self.assertRaises(ValueError):
            sched.normalize_tuning({"gl2": 4})
        # The tile-count parity is the pipeline's TailNumber, not an instance trait.
        with self.assertRaises(ValueError):
            sched.normalize_tuning({"odd_tiles": 1})
        # TdmEpilogue stores E with no vector width.
        with self.assertRaises(ValueError):
            sched.normalize_tuning({"store_width": 1})

    def test_lds_buffers_must_be_an_integer(self):
        # 3.0 is 3; a fraction, a string or a bool is refused, not truncated to a count.
        self.assertEqual(sched.normalize_tuning({"lds_buffers": 3.0})["lds_buffers"], 3)
        for value in (2.7, "3", True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                sched.normalize_tuning({"lds_buffers": value})

    def test_instance_rules(self):
        def reason(t):
            return sched.validate_instance(
                sched.spec_from_tile(TILE, t, out_dtype="fp16"), t
            )

        self.assertEqual(reason({}), "ok")
        self.assertEqual(reason({"prefetch_lds_read": False}), "split_b_read")
        self.assertEqual(reason({"tail_steps": 4}), "tail_steps")
        self.assertEqual(reason({"cluster_dim": [8, 4]}), "cluster_size")
        self.assertEqual(reason({"gl2_prefetch_distance": 1}), "gl2_distance")
        # 8-bit types on the 32x16 FP4 atom, with a tile_k the 8-bit types allow.
        fp8_tile = {**TILE, "tile_m": 128, "tile_n": 128, "tile_k": 256}
        self.assertEqual(
            sched.validate_instance(
                sched.spec_from_tile(fp8_tile, {}, a_dtype="fp8", out_dtype="fp16"), {}
            ),
            "mma",
        )

    def test_k_limit(self):
        # K plus (4 + 1) tiles of 256 must fit in i32; M = N = 32 keeps the
        # scales under 2^31.
        tile = {**TILE, "tile_m": 32, "tile_n": 32, "tile_k": 256, "warp_tile_m": 16}
        t = {"cluster_dim": [1, 1], "split_b_lds_read": False}

        def reason(K):
            return sched.spec_from_tile(
                tile, t, a_dtype="fp8", out_dtype="fp16", M=32, N=32, K=K
            ).validate()

        self.assertNotEqual(reason(2147482336), "k_long")
        self.assertEqual(reason(2147482368), "k_long")
        # Without GL2 prefetch, 3 LDS buffers look (3 + 1) tiles ahead and 2 buffers (2 + 1):
        # K = 2^31 - 1024 is too long with 3 and runs with 2. Regions not interleaved: 3 buffers
        # of the interleaved layout exceed the LDS.
        t3 = {**t, "gl2_prefetch_distance": 0, "interleave_lds_regions": False}

        def reason_lb(K, buffers):
            return sched.spec_from_tile(
                tile,
                {**t3, "lds_buffers": buffers},
                a_dtype="fp8",
                out_dtype="fp16",
                M=32,
                N=32,
                K=K,
            ).validate()

        self.assertEqual(reason_lb(2147482592, 3), "ok")
        self.assertEqual(reason_lb(2147482624, 3), "k_long")
        self.assertEqual(reason_lb(2147482624, 2), "ok")

    def test_long_k_tail(self):
        # A K tail has fewer K steps than a tile (K mod tile_k at most tile_k - 128); a partial
        # tile of every K step runs in the loop as a whole tile, on an instance without a tail.
        fp8 = {**TILE, "tile_m": 128, "tile_n": 128, "tile_k": 256, "warp_tile_m": 16}

        def spec(K, tile, **kw):
            return sched.spec_from_tile(
                tile, {}, out_dtype="fp16", M=1024, N=1024, K=K, **kw
            )

        for K, tile, kw, tail_steps, full_tiles in (
            (2048, fp8, {"a_dtype": "fp8"}, 0, 8),
            (2048 + 96, fp8, {"a_dtype": "fp8"}, 1, 8),
            (2048 + 128, fp8, {"a_dtype": "fp8"}, 1, 8),
            (2048 + 160, fp8, {"a_dtype": "fp8"}, 0, 9),
            (2048 + 224, fp8, {"a_dtype": "fp8"}, 0, 9),
            (1024 + 32, {**fp8, "tile_k": 128}, {"a_dtype": "fp8"}, 0, 9),
            (2048 + 384, TILE, {}, 3, 4),
            (2048 + 416, TILE, {}, 0, 5),
        ):
            s = spec(K, tile, **kw)
            self.assertEqual(s.validate(), "ok", K)
            self.assertEqual((s.tail_steps, s.full_tiles), (tail_steps, full_tiles), K)

    def test_cpp_spelling(self):
        t = {
            "cluster_dim": [2, 2],
            "interleave_lds_regions": False,
        }
        spec = sched.spec_from_tile(TILE, t, out_dtype="fp16")
        self.assertEqual(sched.cpp_cluster_sequence(t), "ck_tile::sequence<2, 2, 1>")
        self.assertEqual(
            sched.cpp_cluster_sequence({"cluster_dim": [1, 4]}),
            "ck_tile::sequence<4, 1, 1>",
        )
        self.assertEqual(
            sched.cpp_policy_type(t),
            "ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<4, true, true, false>",
        )
        self.assertEqual(
            sched.cpp_kernel_traits(spec, {"edge": False, "tail_steps": 2}),
            "ck_tile::MxTdmScheduledGemmTraits<false, false, false, false, false, 2>",
        )
        self.assertEqual(
            sched.instance_name_suffix(spec, {**t, "edge": False, "tail_steps": 2}),
            "_c2x2_lsi0_noedge_tail2",
        )
        # LdsBuffers is spelled only when it is not 2, so 2-buffer sources stay as they were.
        t3 = {**t, "lds_buffers": 3}
        spec3 = sched.spec_from_tile(TILE, t3, out_dtype="fp16")
        self.assertEqual(
            sched.cpp_policy_type(t3),
            "ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<4, true, true, false, 3>",
        )
        self.assertEqual(
            sched.instance_name_suffix(spec3, {**t3, "edge": False}),
            "_c2x2_lsi0_lb3_noedge",
        )


class TestParity(unittest.TestCase):
    """The Python half of parity_cases.py; test_mx_gemm_tdm_scheduled_host.cpp checks the C++ half."""

    def test_rows(self):
        for row in parity_cases.PARITY_CASES:
            with self.subTest(row=row["name"]):
                self.assertEqual(parity_cases.python_reason(row), row["reason"])

    def test_render_inc(self):
        # CMake generates the C++ half at build time: one check per row.
        inc = parity_cases.render_inc()
        for row in parity_cases.PARITY_CASES:
            with self.subTest(row=row["name"]):
                self.assertEqual(inc.count(f'("{row["name"]}", '), 1)
                self.assertIn(f'"{row["reason"]}");', inc)

    def test_every_reason_covered(self):
        rows = {row["reason"] for row in parity_cases.PARITY_CASES}
        unreachable = set(parity_cases.HOST_UNREACHABLE)
        self.assertFalse(rows & unreachable)
        self.assertEqual(rows | unreachable, set(sched.REASONS))
        self.assertFalse(parity_cases.PYTHON_ONLY & set(sched.REASONS))

    def test_unreachable_reasons_are_unreachable(self):
        # Every instance the tile grid offers (each MX type, tile_m and tile_n 16..256, tile_k
        # 128..1024, the wave groups, both atoms, schedule_tunings()): no reason of
        # HOST_UNREACHABLE comes back, and every reason that does has a parity row. Every type
        # has instances past the pad and accumulator rules, so the scan reaches them.
        rows = {row["reason"] for row in parity_cases.PARITY_CASES}
        sizes = (16, 32, 64, 128, 256)
        first = {}
        ok_dtypes = set()
        for a_dtype, tile_m, tile_n, tile_k, group, atom in itertools.product(
            sched.MX_DTYPES,
            sizes,
            sizes,
            (128, 256, 512, 1024),
            ((2, 2), (4, 1), (1, 4)),
            (16, 32),
        ):
            tile = {
                **TILE,
                "tile_m": tile_m,
                "tile_n": tile_n,
                "tile_k": tile_k,
                "warp_m": group[0],
                "warp_n": group[1],
                "warp_tile_m": atom,
            }
            for tuning in sched.schedule_tunings():
                spec = sched.spec_from_tile(tile, tuning, a_dtype=a_dtype)
                reason = sched.validate_instance(spec, tuning)
                first.setdefault(reason, (a_dtype, tile, tuning))
                if reason == "ok":
                    ok_dtypes.add(a_dtype)
        reached = {r: first[r] for r in parity_cases.HOST_UNREACHABLE if r in first}
        self.assertFalse(reached, "HOST_UNREACHABLE reasons, with their first instance")
        no_row = {r: i for r, i in first.items() if r != "ok" and r not in rows}
        self.assertFalse(no_row, "reasons without a parity row")
        self.assertEqual(ok_dtypes, set(sched.MX_DTYPES))

    def test_problem_rows_sit_on_valid_instances(self):
        # check_problem instantiates the pipeline, whose static_assert needs the policy's
        # GetReason to be Ok, so every problem row's tile and tuning must be a valid instance,
        # whatever the problem's reason. validate() checks the shape rules first, so test_rows
        # does not see an invalid instance behind a shape reason.
        for row in parity_cases.PARITY_CASES:
            if row["mnk"] == (0, 0, 0):
                continue
            with self.subTest(row=row["name"]):
                spec = parity_cases.row_spec(row)
                self.assertEqual(sched.validate_instance(spec, row["tuning"]), "ok")

    def test_ok_problem_rows_fit_their_instance(self):
        # GetUnsupportedReason also checks that the instance's traits cover the problem (edge,
        # wide, tail_steps); MxTdmScheduledSpec.validate does not, so the rows must. An ok row is
        # wide exactly when its problem passes 2^31, as MxTdmScheduledSelectTraits picks, so a
        # wide row runs the wide path.
        for row in parity_cases.PARITY_CASES:
            if row["mnk"] == (0, 0, 0) or row["reason"] != "ok":
                continue
            with self.subTest(row=row["name"]):
                spec = parity_cases.row_spec(row)
                t = sched.normalize_tuning(row["tuning"])
                self.assertTrue(t["edge"] or not (spec.edge_m or spec.edge_n))
                self.assertEqual(t["wide"], spec.wide_a or spec.wide_b or spec.wide_d)
                self.assertEqual(t["tail_steps"], spec.tail_steps)

    def test_enum_matches_reasons(self):
        text = _POLICY_HPP.read_text()
        enum = re.search(
            r"enum class MxTdmScheduledReason : int\s*\{(.*?)\};", text, re.S
        )
        names = re.findall(r"\b(\w+)\b", enum.group(1))
        start = text.index("mx_tdm_scheduled_reason_code(MxTdmScheduledReason r)")
        end = text.index("mx_tdm_scheduled_reason_message(MxTdmScheduledReason r)")
        cases = re.findall(
            r'case MxTdmScheduledReason::(\w+): return "(\w+)";', text[start:end]
        )
        self.assertEqual([name for name, _ in cases], names)
        self.assertEqual(tuple(code for _, code in cases), sched.REASONS)


class TestLdsLayout(unittest.TestCase):
    """The Python half of the host test's LdsLayoutOfTheTunedInstances, LaneChunksTileTheKStep,
    RowBytesOfKValues, Fp6BankSpreadPads and the GetReadsPerStep values of
    PolicyDerivesTheScheduleFromProblemAndParameters (test_mx_gemm_tdm_scheduled_host.cpp), and
    the policy's pad rule."""

    def test_tuned_instances(self):
        # The C++ test's instances and problems, each valid for its problem. Its layout fields:
        # B and A bases and component strides, B and A scale bases, buffer stride, LDS size,
        # then A's and B's iterate-mode rows and stride (0 for a load the TDM pads itself). Then
        # the K step's bytes, chunk bytes, chunks per lane, LDS reads per fragment and per K step,
        # and whether A needs WideA (no row needs WideB).
        no_split = {"split_b_lds_read": False}
        no_split_no_interleave = {**no_split, "interleave_lds_regions": False}
        c1x4 = {**no_split_no_interleave, "cluster_dim": (1, 4)}
        t256x16 = {**TILE, "tile_n": 16, "warp_m": 4, "warp_n": 1, "warp_tile_m": 16}
        t128x128 = {
            **TILE,
            "tile_m": 128,
            "tile_n": 128,
            "tile_k": 256,
            "warp_tile_m": 16,
        }
        mab = (65536, 16, 65536)
        fp8_256x16 = (0, 2176, 6656, 33280, 4352, 4608, 131072, 204288, 0, 0, 0, 0)
        for name, tile, tuning, a_dtype, mnk, layout, step, wide_a in (
            (
                "fp4 256x256x512",
                TILE,
                {},
                "fp4",
                (4096, 4096, 65536),
                (0, 99584, 33024, 33280, 132608, 137216, 141312, 282624, 0, 0, 8, 2064),
                (64, 16, 2, 2, 35),
                False,
            ),
            (
                "fp4 256x16x512",
                t256x16,
                c1x4,
                "fp4",
                mab,
                (0, 2176, 9216, 33280, 4608, 5120, 131072, 206848, 0, 0, 0, 0),
                (64, 16, 2, 2, 12),
                False,
            ),
            (
                "fp8 256x16x256",
                {**t256x16, "tile_k": 256},
                c1x4,
                "fp8",
                mab,
                fp8_256x16,
                (128, 16, 4, 4, 22),
                True,
            ),
            (
                "bf8 256x16x256",
                {**t256x16, "tile_k": 256},
                c1x4,
                "bf8",
                mab,
                fp8_256x16,
                (128, 16, 4, 4, 22),
                True,
            ),
            (
                "fp8 128x128x256",
                t128x128,
                no_split,
                "fp8",
                (2048, 2048, 65536),
                (0, 65536, 16640, 65536, 98816, 99840, 131072, 231936, 0, 0, 0, 0),
                (128, 16, 4, 4, 34),
                False,
            ),
            (
                "fp6 64x128x512",
                parity_cases.T64X128X512,
                parity_cases.FP6_SM,
                "fp6",
                (256, 2048, 2048),
                (0, 24832, 52736, 12544, 49664, 51712, 131072, 208896, 2, 784, 4, 1552),
                (96, 24, 2, 3, 20),
                False,
            ),
            (
                "fp6 64x128x512, 3 buffers",
                parity_cases.T64X128X512,
                {**parity_cases.FP6_SM, "lds_buffers": 3},
                "fp6",
                (256, 2048, 2048),
                (0, 24832, 52736, 12544, 49664, 51712, 77824, 233472, 2, 784, 4, 1552),
                (96, 24, 2, 3, 20),
                False,
            ),
            (
                # Stride 65536 (2 x 65536 + the buffer fit) and LDS 2 x 65536 + 59392 B: the
                # last buffer's end, not 3 strides.
                "fp6 128x16x512, 3 buffers",
                parity_cases.T128X16X512,
                {
                    **parity_cases.FP6_MAB,
                    "interleave_lds_regions": False,
                    "lds_buffers": 3,
                },
                "fp6",
                (512, 16, 3584),
                (0, 3136, 9216, 25088, 6656, 7168, 65536, 190464, 2, 784, 2, 784),
                (96, 24, 2, 3, 11),
                False,
            ),
            (
                "fp6 256x16x512",
                parity_cases.T256X16X512,
                parity_cases.FP6_MAB,
                "fp6",
                (49152, 16, 32768),
                (
                    49664,
                    3136,
                    0,
                    65536,
                    115200,
                    115712,
                    131072,
                    250880,
                    4,
                    1552,
                    2,
                    784,
                ),
                (96, 24, 2, 3, 17),
                False,
            ),
        ):
            with self.subTest(name=name):
                M, N, K = mnk
                spec = sched.spec_from_tile(
                    tile,
                    tuning,
                    a_dtype=a_dtype,
                    out_dtype="bf16" if a_dtype == "fp6" else "fp8e4m3",
                    M=M,
                    N=N,
                    K=K,
                )
                self.assertEqual(spec.validate(check_shape=True), "ok")
                lay = spec.lds_layout()
                got = (
                    lay.b.base,
                    lay.b.stride,
                    lay.a.base,
                    lay.a.stride,
                    lay.b_scale.base,
                    lay.a_scale.base,
                    lay.buffer_stride,
                    lay.lds_bytes,
                    lay.a.iterate_rows,
                    lay.a.iterate_stride,
                    lay.b.iterate_rows,
                    lay.b.iterate_stride,
                )
                self.assertEqual(got, layout)
                self.assertEqual(
                    (
                        spec.step_bytes,
                        spec.chunk_bytes,
                        spec.chunks_per_lane,
                        spec.reads_per_fragment,
                        spec.reads_per_step,
                    ),
                    step,
                )
                self.assertEqual((spec.wide_a, spec.wide_b), (wide_a, False))

    def test_wide_operands(self):
        # At M = N = 1024, A and B pass 2^31 bytes above K = 2^22 for FP4 (K / 2 bytes per row),
        # above K = 2796192 for FP6 (3 K / 4 bytes; the last multiple of 32 at most 2^23 / 3) and
        # above K = 2^21 for the 8-bit types, as in MxTdmScheduledSelectTraits.
        fp8 = {**TILE, "tile_m": 128, "tile_n": 128, "tile_k": 256, "warp_tile_m": 16}
        fp6 = {**TILE, "tile_m": 64, "tile_n": 128, "warp_tile_m": 16}
        for tile, a_dtype, k_max in (
            (TILE, "fp4", 1 << 22),
            (fp6, "fp6", 2796192),
            (fp8, "fp8", 1 << 21),
            (fp8, "bf8", 1 << 21),
        ):
            for K, wide in ((k_max, False), (k_max + sched.BLOCK_K, True)):
                with self.subTest(a_dtype=a_dtype, K=K):
                    spec = sched.spec_from_tile(
                        tile, {}, a_dtype=a_dtype, M=1024, N=1024, K=K
                    )
                    self.assertEqual((spec.wide_a, spec.wide_b), (wide, wide))

    def test_pad_rule(self):
        # A pad period is a power of two of 8..1024 B, which the TDM pads itself. Any other
        # padded block runs in iterate mode: data only (row bytes 0 for the scales), one or more
        # whole rows per block, whole blocks per component; a zero block is rejected. Every
        # block is whole 16 B chunks.
        for block, period in (
            (0, False),
            (4, False),
            (8, True),
            (12, False),
            (768, False),
            (1024, True),
            (2048, False),
        ):
            with self.subTest(block=block):
                self.assertEqual(sched._is_pad_period(block), period)
        pad, reason = sched.Pad, sched.MxTdmScheduledSpec._pad_reason
        for block, row_bytes, rows, want in (
            (0, 256, 128, "lds_pad"),
            (4, 256, 128, "lds_pad"),
            (768, 256, 6, "ok"),
            (768, 256, 7, "lds_pad"),
            (768, 384, 32, "ok"),
            (768, 768, 4, "ok"),
            (768, 512, 64, "lds_pad"),
            (1024, 256, 128, "ok"),
            (256, 0, 64, "ok"),
            (2048, 0, 64, "lds_pad"),
            (8, 0, 64, "lds_pad"),
        ):
            with self.subTest(block=block, row_bytes=row_bytes, rows=rows):
                self.assertEqual(reason(pad(16, block), row_bytes, rows, 16), want)

    def test_power_of_two_rows_pad_per_bank_cycle(self):
        # FP4 and 8-bit rows are powers of two, so GetDataPadBlock's lcm(lane step, bank cycle)
        # is the lane step rounded up to a bank cycle: max(256, repeat x row bytes), for A and B
        # of every tile of the grid with each type's tile_k values and atoms.
        sizes = (16, 32, 64, 128, 256)
        wrong = []
        for a_dtype, tile_ks, atoms in (
            ("fp4", (128, 256, 512, 1024), (16, 32)),
            ("fp8", (128, 256), (16,)),
            ("bf8", (128, 256), (16,)),
        ):
            for tile_m, tile_n, tile_k, group, atom in itertools.product(
                sizes, sizes, tile_ks, ((2, 2), (4, 1), (1, 4)), atoms
            ):
                tile = {
                    **TILE,
                    "tile_m": tile_m,
                    "tile_n": tile_n,
                    "tile_k": tile_k,
                    "warp_m": group[0],
                    "warp_n": group[1],
                    "warp_tile_m": atom,
                }
                spec = sched.spec_from_tile(tile, {}, a_dtype=a_dtype)
                pads = spec.derived_pads()
                for side, repeat in (
                    ("a", spec.wave_repeat_m),
                    ("b", spec.wave_repeat_n),
                ):
                    want = max(256, repeat * spec.k_bytes(tile_k))
                    if repeat and pads[side].block_bytes != want:
                        wrong.append(
                            (a_dtype, tile_m, tile_n, tile_k, group, atom, side)
                        )
        self.assertEqual(wrong, [])

    def test_fp6_bank_spread(self):
        # FP6 data pads 16 B after lcm(lane step, 256 B), where a lane's first bank (64 banks of
        # 4 B) comes round again, a lane step being wave repeat x row bytes (adjacent lanes of a
        # lane group read rows a wave repeat apart): at 384 B rows (tile_k 512) every 2, 1, 1, 1
        # and 1 lane steps for repeats 1, 2, 4, 8 and 16; at 192 B rows (tile_k 256) every 4 lane
        # steps for repeat 1 and every one for repeat 16; at 96 B rows (tile_k 128) every 8 lane
        # steps for repeat 1 and every one for repeat 16. None is a pad period: iterate mode.
        for tile_m, tile_k, block in (
            (16, 512, 768),
            (32, 512, 768),
            (64, 512, 1536),
            (128, 512, 3072),
            (256, 512, 6144),
            (16, 256, 768),
            (256, 256, 3072),
            (16, 128, 768),
            (256, 128, 1536),
        ):
            with self.subTest(tile_m=tile_m, tile_k=tile_k):
                tile = {
                    **TILE,
                    "tile_m": tile_m,
                    "tile_n": 64,
                    "tile_k": tile_k,
                    "warp_m": 1,
                    "warp_n": 4,
                    "warp_tile_m": 16,
                }
                spec = sched.spec_from_tile(
                    tile, {"split_b_lds_read": False}, a_dtype="fp6"
                )
                self.assertEqual(spec.validate(check_shape=False), "ok")
                pad = spec.derived_pads()["a"]
                self.assertEqual((pad.pad_bytes, pad.block_bytes), (16, block))
                self.assertFalse(sched._is_pad_period(pad.block_bytes))

    def test_reads_per_step_clamp(self):
        # A K step's reads clamp to the s_wait_dscnt immediate, 63, as in GetReadsPerStep:
        # 256x256x128 on 4x1 waves (4 x 16 repeats) issues 1 + 4 scale reads and 3 (FP6) or 4
        # (FP8) reads for each of 4 A and 16 B fragments: 65 and 85.
        tile = {**TILE, "tile_k": 128, "warp_m": 4, "warp_n": 1, "warp_tile_m": 16}
        for a_dtype in ("fp6", "fp8"):
            with self.subTest(a_dtype=a_dtype):
                spec = sched.spec_from_tile(tile, {}, a_dtype=a_dtype)
                self.assertEqual(spec.validate(check_shape=False), "ok")
                self.assertEqual(spec.reads_per_step, 63)


class TestValidationUtils(unittest.TestCase):
    def test_registered_for_gfx1250_only(self):
        self.assertIn("comp_tdm_scheduled", vu.GEMM_MX_PIPELINES_BY_ARCH["gfx1250"])
        self.assertNotIn("comp_tdm_scheduled", vu.GEMM_MX_PIPELINES_BY_ARCH["gfx950"])
        self.assertIn(
            "gfx1250",
            vu.mx_tdm_scheduled_reject_reason(
                *TILE.values(), "fp4", "fp4", "fp16", "gfx950"
            ),
        )

    def test_traits(self):
        ok = vu.is_trait_combination_valid
        self.assertFalse(
            ok(
                "comp_tdm_scheduled",
                "direct",
                "intrawave",
                False,
                "mx_gemm",
                "rcr",
                False,
                False,
                False,
            )
        )
        self.assertTrue(
            ok(
                "comp_tdm_scheduled",
                "tdm",
                "intrawave",
                False,
                "mx_gemm",
                "rcr",
                False,
                False,
                False,
            )
        )
        self.assertFalse(
            ok(
                "comp_tdm_scheduled",
                "cshuffle",
                "intrawave",
                False,
                "mx_gemm",
                "rcr",
                False,
                False,
                False,
            )
        )
        self.assertFalse(
            ok(
                "comp_tdm_scheduled",
                "tdm",
                "intrawave",
                True,
                "mx_gemm",
                "rcr",
                False,
                False,
                False,
            )
        )
        self.assertFalse(
            ok(
                "comp_tdm_scheduled",
                "tdm",
                "intrawave",
                False,
                "mx_gemm",
                "rcr",
                True,
                False,
                False,
            )
        )
        self.assertFalse(ok("comp_tdm", "direct", "intrawave", False, "mx_gemm", "rcr"))

    def test_tiles(self):
        def valid(**tile):
            dims = [{**TILE, **tile}[k] for k in TILE]
            return vu.is_tile_config_valid(
                *dims,
                "fp4",
                "fp4",
                "fp16",
                "comp_tdm_scheduled",
                "rcr",
                "gfx1250",
                "mx_gemm",
            )

        self.assertTrue(valid())
        # Valid without SplitBLdsRead only: kept for the tuning_config product to decide.
        self.assertTrue(valid(tile_n=64, warp_m=1, warp_n=4))
        self.assertTrue(
            valid(tile_m=128, tile_n=16, warp_m=4, warp_n=1, warp_tile_m=16)
        )
        dims = [{**TILE, "tile_n": 64, "warp_m": 1, "warp_n": 4}[k] for k in TILE]
        self.assertIn(
            "split_b_read",
            vu.mx_tdm_scheduled_reject_reason(
                *dims, "fp4", "fp4", "fp16", "gfx1250", {}
            ),
        )
        self.assertEqual(
            vu.mx_tdm_scheduled_reject_reason(
                *dims, "fp4", "fp4", "fp16", "gfx1250", {"split_b_lds_read": False}
            ),
            "",
        )
        self.assertFalse(valid(tile_k=1024))  # LDS
        self.assertFalse(valid(warp_tile_n=32))
        self.assertFalse(valid(warp_k=2))


class TestTileEngine(unittest.TestCase):
    def test_builder_rejects_comp_tdm_scheduled(self):
        config = _MX / "configs" / "comp_tdm_scheduled_maf_gfx1250.json"
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    sys.executable,
                    str(BUILDER),
                    "--working_path",
                    tmp,
                    "--datatype",
                    "fp4",
                    "--layout",
                    "rcr",
                    "--config_json",
                    str(config),
                    "--gpu_target",
                    "gfx1250",
                    "--list_kernels",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("mx_gemm_full_benchmark.py", result.stderr)
            self.assertFalse(list(Path(tmp).iterdir()))


def _cfg(**extra):
    return {
        "datatype": "fp4",
        "layout": "rcr",
        "gpu_target": "gfx1250",
        "pipeline": "comp_tdm_scheduled",
        "tile_config": dict(TILE),
        **extra,
    }


def _kernel_config(**extra):
    return mx.MxGemmKernelConfig(
        datatype="fp4",
        gpu_target="gfx1250",
        pipeline="comp_tdm_scheduled",
        **TILE,
        **extra,
    )


class TestDispatcher(unittest.TestCase):
    def test_names(self):
        base = "mx_gemm_fp4_rcr_comp_tdm_scheduled_tdm_intrawave_False_False_False_False_256x256x512_2x2x1_32x16x128"
        self.assertEqual(codegen.kernel_name(_cfg()), base + "_c4x4")
        self.assertEqual(
            codegen.kernel_name(
                _cfg(cluster_dim=[2, 2], gl2_prefetch_distance=2, tail_steps=1)
            ),
            base + "_c2x2_pf2_tail1",
        )

    def test_rejects(self):
        with self.assertRaisesRegex(ValueError, "split_b_read"):
            codegen.kernel_name(_cfg(prefetch_lds_read=False))
        with self.assertRaisesRegex(ValueError, "only for comp_tdm_scheduled"):
            codegen.kernel_name(_cfg(pipeline="comp_tdm", cluster_dim=[2, 2]))
        for epilogue in ("cshuffle", "direct"):
            with self.assertRaisesRegex(ValueError, "epilogue"):
                codegen.kernel_name(_cfg(epilogue=epilogue))
        with self.assertRaisesRegex(ValueError, "gfx950"):
            codegen.kernel_name(_cfg(gpu_target="gfx950"))

    def test_header(self):
        with tempfile.TemporaryDirectory() as out:
            path = codegen.generate_kernel(Path(out), _cfg(cluster_dim=[2, 2]))
            header = path.read_text()
        self.assertTrue(path.name.endswith("_c2x2.hpp"))
        self.assertIn("using Cluster = ck_tile::sequence<2, 2, 1>;", header)
        self.assertIn(
            "ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<4, true, true, true>",
            header,
        )
        self.assertIn("KMajorScales = true", header)
        self.assertIn("_comp_tdm_scheduled_tdm_", path.name)
        self.assertIn("ck_tile::MxTdmStoreEpilogue<P>", header)

    def test_three_lds_buffers(self):
        # The top-level lds_buffers key reaches the name (_lb3) and the policy's fifth argument;
        # the default 256x256x512 tile does not fit 3 buffers.
        cfg = _cfg(lds_buffers=3, tile_config={**TILE, "tile_m": 128})
        with tempfile.TemporaryDirectory() as out:
            path = codegen.generate_kernel(Path(out), cfg)
            header = path.read_text()
        self.assertTrue(path.name.endswith("_c4x4_lb3.hpp"), path.name)
        self.assertIn(
            "ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<4, true, true, true, 3>",
            header,
        )
        with self.assertRaisesRegex(ValueError, "lds_capacity"):
            codegen.kernel_name(_cfg(lds_buffers=3))

    def test_kernel_config(self):
        cfg = _kernel_config(
            tuning={"cluster_dim": (2, 2), "gl2_prefetch_distance": 2}
        ).to_codegen_config()
        self.assertEqual(cfg["epilogue"], "tdm")
        self.assertEqual(cfg["cluster_dim"], [2, 2, 1])
        self.assertTrue(codegen.kernel_name(cfg).endswith("_c2x2_pf2"))

    def test_fallback_name_carries_the_tuning(self):
        for tuning, suffix in (
            (None, "_c4x4"),
            ({"cluster_dim": (2, 2), "gl2_prefetch_distance": 2}, "_c2x2_pf2"),
        ):
            with self.subTest(tuning=tuning):
                cfg = _kernel_config(tuning=tuning)
                name = codegen.kernel_name(cfg.to_codegen_config())
                self.assertTrue(name.endswith(suffix), name)
                self.assertTrue(cfg._fallback_name().endswith(suffix))
        # Without the spec module: a hash of the tuning, still distinct per tuning.
        with mock.patch.object(mx, "_mx_codegen", side_effect=ImportError):
            hashed = {
                _kernel_config(tuning={"cluster_dim": c})._fallback_name()
                for c in ((1, 1), (2, 2))
            }
        self.assertEqual(len(hashed), 2)

    def test_name_falls_back_per_tuning(self):
        failed = mock.Mock(returncode=1, stdout="", stderr="boom")
        with mock.patch.object(mx.subprocess, "run", return_value=failed):
            names = [
                _kernel_config(tuning={"cluster_dim": c}).name for c in ((1, 1), (2, 2))
            ]
        self.assertNotEqual(names[0], names[1])
        self.assertTrue(names[0].endswith("_c1x1"), names[0])
        self.assertTrue(names[1].endswith("_c2x2"), names[1])

    def test_kernel_config_tuning_keys(self):
        typo = _kernel_config(tuning={"cluster": (2, 2)})
        self.assertFalse(typo.is_valid())
        with self.assertRaisesRegex(ValueError, "unknown"):
            typo.to_codegen_config()
        for tuning, suffix in (({"cluster_dim": (2, 2)}, "_c2x2"), (None, "_c4x4")):
            with self.subTest(tuning=tuning):
                cfg = _kernel_config(tuning=tuning)
                self.assertTrue(cfg.is_valid())
                name = codegen.kernel_name(cfg.to_codegen_config())
                self.assertTrue(name.endswith(suffix), name)

    def test_tuning_config(self):
        path = _MX / "configs" / "comp_tdm_scheduled_tuning_gfx1250.json"
        tuning_config = json.loads(path.read_text())["tuning_config"]
        self.assertEqual(
            set(tuning_config), {*sched.TUNING_DEFAULTS, *sched.SHAPE_DEFAULTS}
        )
        cfgs = mx.expand_mx_sweep(path, "gfx1250", "fp4")
        self.assertTrue(cfgs)
        self.assertEqual({c.pipeline for c in cfgs}, {"comp_tdm_scheduled"})
        self.assertEqual({c.epilogue for c in cfgs}, {"tdm"})
        # Both buffer counts survive the rules, so the unique names below include _lb3.
        self.assertEqual({c.tuning["lds_buffers"] for c in cfgs}, {2, 3})
        names = [c.name for c in cfgs]
        self.assertEqual(len(set(names)), len(names))
        problems = json.loads(
            (_MX / "configs" / "comp_tdm_scheduled_problems.json").read_text()
        )["problems"]
        self.assertTrue(problems)
        for p in problems:
            self.assertEqual(set(p), {"M", "N", "K"})

    def test_tuned_instance_configs(self):
        # Example 56's maf, medium and mab256.
        expect = {
            "maf": "256x256x512_2x2x1_32x16x128_c4x4_hplrnone_noedge",
            "medium": "128x128x1024_2x2x1_16x16x128_c4x4_hplrnone_noedge",
            "mab": "256x16x512_4x1x1_16x16x128_c1x4_pf0_lp0_hplrnone_noedge",
        }
        for case, suffix in expect.items():
            with self.subTest(case=case):
                path = _MX / "configs" / f"comp_tdm_scheduled_{case}_gfx1250.json"
                cfgs = mx.expand_mx_sweep(path, "gfx1250", "fp4")
                self.assertEqual(len(cfgs), 1)
                self.assertTrue(cfgs[0].name.endswith(suffix), cfgs[0].name)


class TestCompileFlags(unittest.TestCase):
    def test_sched_flags_only_for_comp_tdm_scheduled(self):
        flags = mx._MX_TDM_SCHEDULED_FLAGS
        for pipeline, expect in (("comp_tdm_scheduled", True), ("comp_tdm", False)):
            run_result = mock.Mock(returncode=0, stderr="")
            with mock.patch.object(mx, "_mx_codegen_flags", return_value=()):
                with mock.patch.object(
                    mx.subprocess, "run", return_value=run_result
                ) as run:
                    self.assertTrue(
                        mx._compile_kernel(
                            Path("k.hpp"), Path("k.so"), "gfx1250", pipeline
                        )
                    )
            cmd = run.call_args.args[0]
            for flag in flags[1::2]:
                self.assertEqual(flag in cmd, expect, (pipeline, flag))


if __name__ == "__main__":
    unittest.main()
