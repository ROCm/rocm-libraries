# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Coverage for the flag-free gfx1250 MX GEMM (``mx_gemm``)."""

from __future__ import annotations

import itertools
import re
import shutil
import types
import unittest
from dataclasses import replace
from unittest import mock

import numpy as np
from rocke.core.ir import IRBuilder
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.examples.common.scaled_gemm_reference import (
    decode_e8m0,
    decode_fp6,
    pack_fp6_codes,
)
from rocke.examples.gfx1250.gemm.mx_gemm_bench import (
    _E2M1,
    FIELDS,
    SPARSE_ALPHA,
    SPARSE_STEPS,
    _hold,
    _time,
    check_inputs,
    decode_table,
    device_inputs,
    loop_scratch_ops,
    main,
    pack_fp6_rows,
    reference,
    run_case,
    sparse_inputs,
    sparse_steps,
    tile_rows,
    timed_inputs,
    verify_block,
)
from rocke.helpers.compile import _comgr_options_for_kernel
from rocke.instances.gfx1250.mx_gemm import (
    K_READS,
    LDS_BUFFERS,
    OUT_DTYPES,
    READ_BASES,
    READ_ISSUES,
    SCALE_SWAPS,
    SCALES,
    TILE_K,
    TILE_MN,
    WALKS,
    WAVE_GROUPS,
    WAVE_MAPS,
    WMMA_REUSE,
    IntOps,
    LdsPad,
    HalfWalk,
    MxGemmSpec,
    PanelWalk,
    RingWalk,
    RowMaps,
    ScaleRegisters,
    TileCompute,
    build_mx_gemm,
    derived_lds_pads,
    mx_gemm_lds_layout,
    tiled_transfer,
    validate,
    wave_coords,
)
from rocke.runtime.hip_module import get_device_arch

_BENCH = (4096, 4096, 65536)
_SMOKE = (1024, 1024, 2048)
_WMMA = r"call [^\n]*@llvm\.amdgcn\.wmma\.scale\."
_MUL_I64 = r"= mul (nsw |nuw )*i64"


def _spec(*problem, **fields) -> MxGemmSpec:
    m, n, k = problem or _SMOKE
    return MxGemmSpec(name="t", M=m, N=n, K=k, **fields)


def _llvm(spec: MxGemmSpec) -> str:
    return _lower_kernel_to_llvm_python(
        build_mx_gemm(spec), arch="gfx1250", llvm_flavor="llvm23"
    )


def _valid_geometries(mma: str = "32x16", **fields):
    for tm, tn, tk, wg in itertools.product(TILE_MN, TILE_MN, TILE_K, WAVE_GROUPS):
        spec = _spec(tile_m=tm, tile_n=tn, tile_k=tk, wave_group=wg, mma=mma, **fields)
        if validate(spec) is None:
            yield spec


# (a_dtype, b_dtype) of the 8-bit types and their WMMA matrix formats.
_PAIRS_8BIT = (("fp8", "fp8"), ("bf8", "bf8"), ("fp8", "bf8"), ("bf8", "fp8"))
_FORMATS = {"fp8": 0, "bf8": 1, "fp6": 2, "fp4": 4}
# The tiles that also suit FP6: each row of a 384 K tile is 288 B.
_FP6_TILES = ((256, 256, 256), (256, 256, 384))
# Two large problems made of whole 256x256 tiles.
_MAF_MED = ((3072, 4096, 65536), (2048, 2048, 65536))
# The direct path: each K-step reads, drains and computes.
_DIRECT = {"local_prefetch": 0, "half_plr": "none"}
# walk "panels" at its one tile, read-ahead.
_PANELS = {
    "tile_m": 256,
    "tile_n": 256,
    "tile_k": 512,
    "local_prefetch": 1,
    "walk": "panels",
}
_HALVES = {**_PANELS, "walk": "halves"}
# wave_map "pairs" on the 256x256x256 tile, read-ahead, half-B late.
_PAIRS = {
    "tile_m": 256,
    "tile_n": 256,
    "local_prefetch": 1,
    "half_b_order": "late",
    "wave_map": "pairs",
}
# walk "ring" at the 256x256x256 tile: read-ahead, B in halves (late), fixed
# bases, contiguous reads.
_RING = {
    "tile_m": 256,
    "tile_n": 256,
    "tile_k": 256,
    "local_prefetch": 1,
    "half_plr": "b",
    "half_b_order": "late",
    "read_bases": "fixed",
    "k_reads": "contiguous",
    "walk": "ring",
}


def _spec8(*problem, a_dtype="fp8", b_dtype="fp8", **fields) -> MxGemmSpec:
    """An 8-bit spec: the 16x16 atom and tile_k 256 unless given."""
    fields = {"mma": "16x16", "tile_k": 256, **fields}
    return _spec(*problem, a_dtype=a_dtype, b_dtype=b_dtype, **fields)


def _spec6(*problem, **fields) -> MxGemmSpec:
    """An FP6 spec: the 16x16 atom and tile_k 256 unless given."""
    fields = {"mma": "16x16", "tile_k": 256, **fields}
    return _spec(*problem, a_dtype="fp6", b_dtype="fp6", **fields)


def _lane_k(spec: MxGemmSpec) -> np.ndarray:
    """``[lane][slot]``: the K index (within a K-step) of the value that the
    lane's ``lane_reads`` put in fragment slot ``slot``. Each slot's bits are
    consecutive bits of the row (value ``k`` of a row is at bit ``k * bits``)
    and every read stays inside the K-step."""
    bits = spec.elem_bits
    layout = spec.fragment_layout
    starts, at = [], 0
    for offset, words in spec.lane_reads:
        starts.append((at, offset, 4 * words))
        at += 4 * words
    out = np.zeros((32, layout.fragment.count), dtype=np.int64)
    for lane, slot in itertools.product(range(32), range(layout.fragment.count)):
        row_bits = []
        for bit in range(slot * bits, (slot + 1) * bits):
            frag_byte = bit // 8
            ((first, offset, size),) = [
                r for r in starts if r[0] <= frag_byte < r[0] + r[2]
            ]
            byte = lane // 16 * spec.lane_half_bytes + offset + frag_byte - first
            assert byte < spec.step_bytes
            row_bits.append(byte * 8 + bit % 8)
        assert row_bits == list(range(row_bits[0], row_bits[0] + bits))
        assert row_bits[0] % bits == 0
        out[lane, slot] = row_bits[0] // bits
    return out


def _landing(
    layout, pad: LdsPad, j: int, iterate_rows: int, iterate_stride: int
) -> int:
    """Where TDM puts byte ``j`` of a component's row-major tile (relative to
    the component base): pad after every block, or one block per iteration."""
    if iterate_rows:
        block = iterate_rows * layout.row_bytes
        return j // block * iterate_stride + j % block
    if pad.on:
        return j + j // pad.block_bytes * pad.pad_bytes
    return j


class MxGemmSpecTest(unittest.TestCase):
    def test_former_module_names_alias_the_renamed_modules(self):
        from rocke.examples.gfx1250.gemm import mx_gemm_bench, mxfp4_gemm_bench
        from rocke.instances.gfx1250 import mx_gemm, mxfp4_gemm

        self.assertIs(mxfp4_gemm_bench, mx_gemm_bench)
        self.assertIs(mxfp4_gemm.MxFp4GemmSpec, mx_gemm.MxGemmSpec)
        self.assertIs(mxfp4_gemm.build_mxfp4_gemm, mx_gemm.build_mx_gemm)
        self.assertIs(mxfp4_gemm.mxfp4_gemm_grid, mx_gemm.mx_gemm_grid)
        self.assertIs(mxfp4_gemm.TILE_MN, mx_gemm.TILE_MN)

    def test_validator_accepts_tuning_space_and_rejects_bad_specs(self):
        self.assertIsNone(validate(_spec()))
        self.assertIsNone(validate(_spec(*_BENCH)))
        self.assertIsNone(validate(_spec(cluster_dim=(1, 1, 1), out_dtype="bf16")))
        self.assertIsNone(validate(_spec(half_plr="none", local_prefetch=0)))
        self.assertEqual(len(list(_valid_geometries())), 87)
        self.assertEqual(len(list(_valid_geometries("16x16"))), 122)
        self.assertIsNone(validate(_spec(tile_m=128, tile_n=128, tile_k=1024)))
        self.assertIn("LDS", validate(_spec(tile_k=1024)) or "")
        self.assertIsNone(validate(_spec(tile_m=32, mma="16x16")))
        bad = {
            "tile_m": {"tile_m": 48},
            "mma": {"mma": "16x32"},
            "atom fit": {"tile_m": 32},
            "tile_k": {"tile_k": 64},
            "wave_group": {"wave_group": (1, 2)},
            "short K": {"tile_k": 512, "wave_group": (2, 2)},
            "grid": {"cluster_dim": (8, 1, 1)},
            "half_plr": {"local_prefetch": 0},
            "half_b_order": {"half_b_order": "early"},
            "late needs half B": {"half_b_order": "late", "half_plr": "none"},
            "signal range": {"cluster_signal_wmmas": 17},
            "out dtype": {"out_dtype": "fp32"},
            "pad block": {"a_lds_pad": LdsPad(16, 384)},
            "pad cut row": {"b_lds_pad": LdsPad(16, 128)},
            "pad bytes": {"b_scale_lds_pad": LdsPad(12, 256)},
            "prefetch": {"gl2_prefetch_distance": 1},
        }
        bad_problems = {"short K": (1024, 1024, 256), "K step": (1024, 1024, 2064)}
        bad["K step"] = {}
        for label, fields in bad.items():
            problem = bad_problems.get(label, _SMOKE)
            with self.subTest(label):
                self.assertIsNotNone(validate(_spec(*problem, **fields)))

    def test_8bit_types_validate_their_atom_tile_k_and_reads(self):
        """FP8/BF8 take the 16x16 atom and tile_k 128/256 and never mix with
        FP4. The 256x256 tile's 68 reads per K-step pass the s_wait_dscnt
        range: its local_prefetch 1 waits clamp to 63 with half_b_order
        "ahead" and stay below it with "late". local_prefetch 0 waits for 0."""
        for a_dtype, b_dtype in _PAIRS_8BIT:
            fields = {"a_dtype": a_dtype, "b_dtype": b_dtype}
            with self.subTest(a_dtype=a_dtype, b_dtype=b_dtype):
                self.assertEqual(len(list(_valid_geometries("16x16", **fields))), 68)
                self.assertIsNone(validate(_spec8(tile_m=128, **fields)))
                self.assertIsNone(validate(_spec8(tile_k=128, tile_m=128, **fields)))
                wide = _spec8(**fields)
                self.assertEqual(wide.reads_per_step, 68)
                self.assertIsNone(validate(wide))
                waits = {
                    int(n) for n in re.findall(r'"s_wait_dscnt (\d+)"', _llvm(wide))
                }
                self.assertIn(63, waits)
                self.assertLessEqual(max(waits), 63)
                late = _spec8(half_b_order="late", **fields)
                self.assertEqual(late.dscnt_waits, (16, 36))
                self.assertEqual(late.late_pending_reads, 36)
                self.assertIsNone(validate(late))
                bad = {
                    "32x16": ({"mma": "32x16"}, "mma must be one of"),
                    "tile_k 512": ({"tile_k": 512}, "tile_k must be in"),
                }
                for label, (extra, reason) in bad.items():
                    got = validate(_spec8(**fields, **extra)) or ""
                    self.assertIn(reason, got, label)
                direct = _spec8(local_prefetch=0, half_plr="none", **fields)
                self.assertGreater(direct.reads_per_step, 63)
                self.assertIsNone(validate(direct))
        for a_dtype, b_dtype in (("fp4", "fp8"), ("bf8", "fp4"), ("fp6", "fp8")):
            reason = validate(_spec8(a_dtype=a_dtype, b_dtype=b_dtype, tile_m=128))
            self.assertIn("same width", reason or "")
        self.assertIn("a_dtype and b_dtype must be in", validate(_spec(a_dtype="bf6")))

    def test_fp6_validates_its_atom_tile_k_and_bytes(self):
        """FP6 takes the 16x16 atom and tile_k 128/256/384/512/1024 (512 only
        below 256x256, 1024 only on smaller tiles: LDS), packs four K values in
        three bytes, and is the only type with contiguous reads. 256x256x256
        and 256x256x384 are valid at two large problems with the 4x4 cluster."""
        self.assertEqual(
            len(list(_valid_geometries("16x16", a_dtype="fp6", b_dtype="fp6"))), 146
        )
        spec = _spec6()
        self.assertEqual((spec.step_bytes, spec.packed_k), (96, 1536))
        self.assertEqual(spec.k_bytes(384), 288)
        self.assertEqual(spec.lane_half_bytes, 24)
        self.assertEqual(
            spec.lane_reads, ((0, 2), (8, 2), (16, 2), (48, 2), (56, 2), (64, 2))
        )
        contiguous = _spec6(k_reads="contiguous")
        self.assertEqual(contiguous.lane_half_bytes, 48)
        self.assertEqual(contiguous.lane_reads, ((0, 4), (16, 4), (32, 4)))
        # 256x256 tile: 2 + 2 scale reads, then 3 per row and fragment (the
        # chunked 8 B reads in pairs).
        self.assertEqual((spec.row_reads, contiguous.row_reads), (3, 3))
        self.assertEqual((spec.reads_per_step, contiguous.reads_per_step), (52, 52))
        for tile_k in (128, 256, 384, 512):
            with self.subTest(tile_k=tile_k):
                self.assertIsNone(validate(_spec6(tile_m=128, tile_k=tile_k)))
        self.assertIn("LDS", validate(_spec6(tile_k=512)) or "")
        # K tile 1024: 768 B rows, which fit with a narrow tile.
        narrow = {"tile_n": 16, "tile_k": 1024, "wave_group": (4, 1), **_DIRECT}
        self.assertIsNone(validate(_spec6(tile_m=128, cluster_dim=(1, 1, 1), **narrow)))
        self.assertIn(
            "LDS", validate(_spec6(tile_m=256, cluster_dim=(1, 1, 1), **narrow)) or ""
        )
        for shape, (tm, tn, tk) in itertools.product(_MAF_MED, _FP6_TILES):
            for k_reads in K_READS:
                fields = {"tile_m": tm, "tile_n": tn, "tile_k": tk, "k_reads": k_reads}
                with self.subTest(shape=shape, tile=(tm, tn, tk), k_reads=k_reads):
                    self.assertIsNone(
                        validate(_spec6(*shape, out_dtype="bf16", **fields))
                    )
        bad = {
            "32x16": ({"mma": "32x16"}, "mma must be one of"),
            "tile_k 768": ({"tile_k": 768}, "tile_k must be in"),
            "k_reads": ({"k_reads": "strided"}, "k_reads must be one of"),
        }
        for label, (extra, reason) in bad.items():
            with self.subTest(label):
                self.assertIn(reason, validate(_spec6(**extra)) or "")
        for spec in (_spec(k_reads="contiguous"), _spec8(k_reads="contiguous")):
            self.assertIn("is for FP6", validate(spec) or "")
        self.assertIn("tile_k must be in", validate(_spec8(tile_k=384)) or "")
        self.assertIn("tile_k must be in", validate(_spec(tile_k=384)) or "")

    def test_kernel_name_carries_the_dtype_tag(self):
        self.assertIn("t_mxfp4_M1024", _spec().kernel_name())
        for (a_dtype, b_dtype), tag in zip(
            _PAIRS_8BIT, ("mxfp8", "mxbf8", "mxfp8bf8", "mxbf8fp8")
        ):
            name = _spec8(a_dtype=a_dtype, b_dtype=b_dtype).kernel_name()
            self.assertTrue(name.startswith(f"t_{tag}_M1024"), name)
        self.assertNotIn("hblate", _spec().kernel_name())
        self.assertIn("_hblate", _spec(half_b_order="late").kernel_name())
        self.assertTrue(_spec6().kernel_name().startswith("t_mxfp6_M1024"))
        self.assertNotIn("kcont", _spec6().kernel_name())
        self.assertIn("_kcont", _spec6(k_reads="contiguous").kernel_name())

    def test_wmma_reuse_validates_and_names(self):
        """Operand reuse runs with local_prefetch 0 and the 16x16 WMMA (B
        reuse also with read-ahead and B in halves), and tags the kernel
        name."""
        direct = {"local_prefetch": 0, "half_plr": "none"}
        self.assertEqual(WMMA_REUSE, ("none", "a", "b", "ab"))
        tags = {"a": "_ra", "b": "_rb", "ab": "_rab"}
        for reuse, tag in tags.items():
            for spec in (
                _spec6(wmma_reuse=reuse, **direct),
                _spec8(wmma_reuse=reuse, **direct),
            ):
                with self.subTest(spec.kernel_name()):
                    self.assertIsNone(validate(spec))
                    self.assertTrue(spec.kernel_name().endswith(tag))
        for spec in (
            _spec6(wmma_reuse="b"),
            _spec6(wmma_reuse="b", half_b_order="late"),
            _spec8(wmma_reuse="b", half_b_order="late"),
        ):
            with self.subTest(spec.kernel_name()):
                self.assertIsNone(validate(spec))
                self.assertIn("_rb", spec.kernel_name())
        plain = _spec6(**direct).kernel_name()
        self.assertFalse(any(tag in plain for tag in tags.values()))
        bad = {
            "value": (_spec6(wmma_reuse="c", **direct), "wmma_reuse must be one of"),
            "prefetch": (_spec6(wmma_reuse="ab"), "needs local_prefetch 0"),
            "half-B": (
                _spec6(wmma_reuse="b", half_plr="none"),
                "with local_prefetch 1 needs half_plr 'b'",
            ),
            "atom": (_spec(wmma_reuse="a", **direct), "is for the 16x16 WMMA"),
        }
        for label, (spec, reason) in bad.items():
            with self.subTest(label):
                self.assertIn(reason, validate(spec) or "")

    def test_scales_validate_name_and_leave_lds(self):
        """Register scales run with local_prefetch 0 and the 16x16 WMMA, tag
        the kernel name and drop both LDS scale regions."""
        direct = {"local_prefetch": 0, "half_plr": "none"}
        self.assertEqual(SCALES, ("lds", "registers"))
        for spec in (
            _spec6(scales="registers", **direct),
            _spec6(scales="registers", k_reads="contiguous", tile_k=384, **direct),
            _spec8(scales="registers", wmma_reuse="b", **direct),
        ):
            lds = replace(spec, scales="lds")
            with self.subTest(spec.kernel_name()):
                self.assertIsNone(validate(spec))
                self.assertIn("_sreg", spec.kernel_name())
                self.assertNotIn("_sreg", lds.kernel_name())
                self.assertEqual((spec.a_scale_reads, spec.b_scale_reads), (0, 0))
                layout, lds_layout = mx_gemm_lds_layout(spec), mx_gemm_lds_layout(lds)
                self.assertEqual(layout.a_scale.stride, 0)
                self.assertEqual(layout.b_scale.stride, 0)
                self.assertEqual(layout.buffer_bytes, lds_layout.b_scale.base)
                self.assertLess(spec.lds_bytes, lds.lds_bytes)
        bad = {
            "value": (_spec6(scales="vgpr", **direct), "scales must be one of"),
            "prefetch": (_spec6(scales="registers"), "needs local_prefetch 0"),
            "atom": (_spec(scales="registers", **direct), "is for the 16x16 WMMA"),
        }
        for label, (spec, reason) in bad.items():
            with self.subTest(label):
                self.assertIn(reason, validate(spec) or "")

    def test_lds_buffers_validate_name_and_size(self):
        """Three LDS buffers take three buffers' bytes, tag the kernel name
        and need rows that are not a fraction of a GL2 line; the LDS must
        still hold them."""
        self.assertEqual(LDS_BUFFERS, (2, 3))
        for spec in (
            _spec6(tile_m=128, tile_n=128, tile_k=512, **_DIRECT),
            _spec6(tile_m=128, tile_n=128, tile_k=512, half_b_order="late"),
            _spec6(tile_m=128, tile_n=128, tile_k=256),
            _spec8(tile_m=128, tile_n=128),
        ):
            three = replace(spec, lds_buffers=3)
            with self.subTest(three.kernel_name()):
                self.assertIsNone(validate(three))
                self.assertTrue(three.kernel_name().endswith("_lb3"))
                self.assertNotIn("_lb3", spec.kernel_name())
                self.assertEqual(spec.lds_bytes, 2 * spec.lds_layout.buffer_bytes)
                self.assertEqual(three.lds_bytes, 3 * spec.lds_layout.buffer_bytes)
        bad = {
            "value": (_spec6(lds_buffers=4), "lds_buffers must be one of"),
            "row": (_spec8(tile_k=128, lds_buffers=3), "cannot take 128 B rows"),
            "capacity": (_spec6(tile_k=384, lds_buffers=3), "exceeds"),
        }
        for label, (spec, reason) in bad.items():
            with self.subTest(label):
                self.assertIn(reason, validate(spec) or "")

    def test_scale_swap_validate_and_name(self):
        """Swapping the scale words after the K-step's wait tags the kernel
        name and needs "contiguous" FP6 reads with scales in LDS."""
        self.assertEqual(SCALE_SWAPS, ("read", "wait"))
        for fields in (
            _DIRECT,
            {"half_plr": "none"},
            {"half_plr": "b"},
            {"half_plr": "b", "half_b_order": "late"},
        ):
            spec = _spec6(k_reads="contiguous", **fields)
            wait = replace(spec, scale_swap="wait")
            with self.subTest(wait.kernel_name()):
                self.assertIsNone(validate(wait))
                self.assertTrue(wait.kernel_name().endswith("_swait"))
                self.assertNotIn("_swait", spec.kernel_name())
                self.assertTrue(wait.swap_after_wait)
                self.assertFalse(spec.swap_after_wait)
        needs = "needs k_reads 'contiguous' and scales 'lds'"
        bad = {
            "value": (_spec6(scale_swap="late"), "scale_swap must be one of"),
            "chunked": (_spec6(scale_swap="wait"), needs),
            "fp8": (_spec8(scale_swap="wait"), needs),
            "registers": (
                _spec6(
                    k_reads="contiguous",
                    scales="registers",
                    scale_swap="wait",
                    **_DIRECT,
                ),
                needs,
            ),
        }
        for label, (spec, reason) in bad.items():
            with self.subTest(label):
                self.assertIn(reason, validate(spec) or "")

    def test_read_issue_validate_and_name(self):
        """Interleaved reads need read-ahead without the half-B prefetch, the
        16x16 WMMA and, with "contiguous" FP6 reads, the swaps after the
        wait; they tag the name."""
        self.assertEqual(READ_ISSUES, ("block", "interleave"))
        ahead = {"local_prefetch": 1, "half_plr": "none"}
        for spec in (
            _spec6(k_reads="contiguous", scale_swap="wait", **ahead),
            _spec6(**ahead),
            _spec8(**ahead),
        ):
            interleave = replace(spec, read_issue="interleave")
            with self.subTest(interleave.kernel_name()):
                self.assertIsNone(validate(interleave))
                self.assertTrue(interleave.kernel_name().endswith("_ril"))
                self.assertNotIn("_ril", spec.kernel_name())
        needs = "needs local_prefetch 1 and half_plr 'none'"
        bad = {
            "value": (_spec6(read_issue="mixed"), "read_issue must be one of"),
            "direct": (_spec6(read_issue="interleave", **_DIRECT), needs),
            "half_b": (_spec6(read_issue="interleave"), needs),
            "atom": (
                _spec(read_issue="interleave", half_plr="none"),
                "is for the 16x16 WMMA",
            ),
            "swap": (
                _spec6(read_issue="interleave", k_reads="contiguous", **ahead),
                "needs scale_swap 'wait'",
            ),
        }
        for label, (spec, reason) in bad.items():
            with self.subTest(label):
                self.assertIn(reason, validate(spec) or "")

    def test_walk_validate_and_name(self):
        """Panel walks and half walks take FP6 at the 256x256x512 tile with
        read-ahead, M and N multiples of 256 and K of 1024; they tag the
        name. Panels take four panel slots of LDS and a 128x128 pass; halves
        two A slots and two B slots of two panels, and a 128x256 half."""
        self.assertEqual(WALKS, ("tile", "panels", "halves", "ring"))
        for walk, suffix, pass_n in ((_PANELS, "_pw", 128), (_HALVES, "_pwh", 256)):
            for fields in (
                {"half_plr": "none"},
                {"half_plr": "b"},
                {"half_plr": "b", "half_b_order": "late"},
                {
                    "half_plr": "none",
                    "k_reads": "contiguous",
                    "scale_swap": "wait",
                    "read_issue": "interleave",
                },
            ):
                spec = _spec6(*_MAF_MED[0], cluster_dim=(4, 4, 1), **walk, **fields)
                with self.subTest(spec.kernel_name()):
                    self.assertIsNone(validate(spec))
                    self.assertTrue(spec.kernel_name().endswith(suffix))
                    self.assertEqual(spec.pass_spec.tile_m, 128)
                    self.assertEqual(spec.pass_spec.tile_n, pass_n)
                    self.assertEqual(spec.pass_spec.walk, "tile")
                    layout = spec.lds_layout
                    slots = 4 * layout.buffer_bytes
                    if spec.halves:
                        slots = 2 * layout.buffer_bytes
                        slots += 2 * HalfWalk.pair_bytes(layout)
                    self.assertEqual(spec.lds_bytes, slots)
            base = {**walk, "half_plr": "none"}
            name = f"walk {walk['walk']!r}"
            bad = {
                "value": (_spec6(walk="strips"), "walk must be one of"),
                "fp8": (_spec8(*_MAF_MED[0], **{**base, "tile_k": 256}), "is for FP6"),
                "tile": (
                    _spec6(*_MAF_MED[0], **{**base, "tile_k": 384}),
                    "256x256x512",
                ),
                "direct": (
                    _spec6(*_MAF_MED[0], **{**base, "local_prefetch": 0}),
                    f"{name} needs local_prefetch 1",
                ),
                "buffers": (
                    _spec6(*_MAF_MED[0], **{**base, "lds_buffers": 3}),
                    "takes no lds_buffers",
                ),
                "edge": (_spec6(3000, 4096, 65536, **base), "multiples of 256"),
                "k": (_spec6(3072, 4096, 65536 + 512, **base), "multiple of 1024"),
            }
            for label, (spec, reason) in bad.items():
                with self.subTest(walk=walk["walk"], case=label):
                    self.assertIn(reason, validate(spec) or "")

    def test_read_bases_validate_and_name(self):
        """Fixed read bases take the 16x16 WMMA and scales without LDS pad;
        they tag the name."""
        self.assertEqual(READ_BASES, ("loop", "fixed"))
        ahead = {"local_prefetch": 1, "half_plr": "none"}
        for spec in (
            _spec6(tile_m=128, tile_n=128, tile_k=512, lds_buffers=3, **ahead),
            _spec6(512, 512, 2048, cluster_dim=(2, 2, 1), **_PANELS, half_plr="none"),
            _spec6(512, 512, 2048, cluster_dim=(2, 2, 1), **_HALVES, half_plr="none"),
            _spec6(
                1024,
                16,
                2048,
                tile_m=128,
                tile_n=16,
                tile_k=512,
                wave_group=(4, 1),
                cluster_dim=(1, 2, 1),
                **_DIRECT,
            ),
            _spec6(
                256, 512, 2048, tile_m=64, tile_n=64, tile_k=512, cluster_dim=(4, 2, 1)
            ),
            _spec8(tile_m=128, tile_n=128, **ahead),
        ):
            fixed = replace(spec, read_bases="fixed")
            with self.subTest(fixed.kernel_name()):
                self.assertIsNone(validate(fixed))
                self.assertTrue(fixed.kernel_name().endswith("_rbf"))
                self.assertNotIn("_rbf", spec.kernel_name())
        bad = {
            "value": (_spec6(read_bases="vgpr"), "read_bases must be one of"),
            "atom": (_spec(read_bases="fixed"), "is for the 16x16 WMMA"),
            # The 256x256x256 tile's derived scale pads.
            "scale_pad": (
                _spec6(read_bases="fixed"),
                "needs the scales without LDS pad",
            ),
        }
        for label, (spec, reason) in bad.items():
            with self.subTest(label):
                self.assertIn(reason, validate(spec) or "")

    def test_wave_map_validate_and_name(self):
        """The pair map takes 2x2 waves, the 16x16 WMMA, an even N-repeat
        count, LDS scales and walk "tile"; it tags the name."""
        self.assertEqual(WAVE_MAPS, ("rows", "pairs"))
        for spec in (
            _spec6(**_PAIRS),
            _spec6(**_PAIRS, lds_buffers=3, k_reads="contiguous", scale_swap="wait"),
            _spec6(tile_m=128, tile_n=128, tile_k=512, wave_map="pairs"),
            _spec6(
                tile_m=128,
                tile_n=128,
                tile_k=512,
                local_prefetch=1,
                half_plr="none",
                lds_buffers=3,
                k_reads="contiguous",
                scale_swap="wait",
                read_issue="interleave",
                read_bases="fixed",
                wave_map="pairs",
            ),
            _spec6(256, 512, 2048, tile_m=64, tile_n=64, tile_k=512, wave_map="pairs"),
            _spec8(**_DIRECT, wave_map="pairs"),
        ):
            with self.subTest(spec.kernel_name()):
                self.assertIsNone(validate(spec))
                self.assertTrue(spec.kernel_name().endswith("_wmp"))
                self.assertEqual(spec.n_group, spec.wave_repeat_n // 2)
                rows = replace(spec, wave_map="rows")
                self.assertNotIn("_wmp", rows.kernel_name())
                self.assertEqual(rows.n_group, rows.wave_repeat_n)
        bad = {
            "value": (_spec6(wave_map="cols"), "wave_map must be one of"),
            "waves": (
                _spec6(tile_m=256, tile_n=64, wave_group=(4, 1), wave_map="pairs"),
                "needs wave_group 2x2",
            ),
            "atom": (_spec(wave_map="pairs"), "needs wave_group 2x2 and the 16x16"),
            "odd": (
                _spec6(tile_m=128, tile_n=32, wave_map="pairs", half_plr="none"),
                "needs an even wave_repeat_n",
            ),
            "scales": (
                _spec6(**_DIRECT, scales="registers", wave_map="pairs"),
                "scales 'lds'",
            ),
            "walk": (
                _spec6(512, 512, 2048, **_PANELS, half_plr="none", wave_map="pairs"),
                "walk 'tile'",
            ),
        }
        for label, (spec, reason) in bad.items():
            with self.subTest(label):
                self.assertIn(reason, validate(spec) or "")

    def test_inst_prefetch_validate_and_name(self):
        """inst_prefetch is 0 (none) or a positive WMMA count and tags the
        name with the count."""
        self.assertEqual(_spec6().inst_prefetch, 0)
        for every in (1, 16, 64):
            spec = _spec6(inst_prefetch=every)
            with self.subTest(every=every):
                self.assertIsNone(validate(spec))
                self.assertIn(f"_ipf{every}", spec.kernel_name())
        self.assertNotIn("_ipf", _spec6().kernel_name())
        self.assertIn(
            "inst_prefetch must be 0", validate(_spec6(inst_prefetch=-1)) or ""
        )

    def test_tiled_operands_validate_and_name(self):
        """tiled_operands takes whole tiles of walk "tile" whose TDM
        transfers are whole 256 B lines, and tags the name."""
        self.assertEqual(_spec6().tiled_operands, "none")
        self.assertNotIn("_tl", _spec6().kernel_name())
        for fields in (
            {"tile_m": 256, "tile_n": 256, "lds_buffers": 3, "half_b_order": "late"},
            {"tile_m": 256, "tile_n": 128, "tile_k": 512},
            {"tile_m": 64, "tile_n": 128, "tile_k": 512, "cluster_dim": (4, 2, 1)},
        ):
            for tiled in ("a", "b", "ab"):
                spec = _spec6(**fields, tiled_operands=tiled)
                with self.subTest(fields=fields, tiled=tiled):
                    self.assertIsNone(validate(spec))
                    self.assertTrue(spec.kernel_name().endswith(f"_tl{tiled}"))
                    self.assertEqual(spec.tiled("a"), tiled in ("a", "ab"))
                    self.assertEqual(spec.tiled("b"), tiled in ("b", "ab"))
        # MXFP8 at 256 B rows: an iterate block, and padded whole-line rows.
        self.assertIsNone(validate(_spec8(tiled_operands="ab")))
        self.assertIsNone(
            validate(
                _spec8(
                    1024,
                    64,
                    2048,
                    tile_n=64,
                    wave_group=(4, 1),
                    cluster_dim=(1, 1, 1),
                    tiled_operands="ab",
                    **_DIRECT,
                )
            )
        )
        for reason, spec in (
            ("tiled_operands must be one of", _spec6(tiled_operands="c")),
            ("needs walk 'tile'", _spec6(tiled_operands="a", **_PANELS)),
            (
                "needs K to be a multiple of tile_k",
                _spec6(1024, 1024, 2048 + 128, tiled_operands="a"),
            ),
            (
                "tiled operand a needs whole tiles",
                _spec6(1000, 1024, 2048, tiled_operands="a"),
            ),
            (
                "tiled operand b needs whole tiles",
                _spec6(1024, 1000, 2048, tiled_operands="ab"),
            ),
            # 96 B FP6 rows padded every two rows: 192 B blocks.
            (
                "tiled operand a: its TDM transfers are not whole",
                _spec6(tile_k=128, a_lds_pad=LdsPad(16, 192), tiled_operands="a"),
            ),
            # 128 B MXFP8 rows with a pad period: rows are half lines.
            (
                "tiled operand b: its TDM transfers are not whole",
                _spec8(tile_k=128, tiled_operands="b"),
            ),
        ):
            with self.subTest(reason):
                self.assertIn(reason, validate(spec) or "")

    def test_pair_map_covers_d_once_and_splits_the_halves(self):
        """With the pair map every D element of a tile has one lane, slot and
        N-repeat, a lane's group is adjacent columns, SIMD pair p (waves p
        and p + 2) holds A half p, and the pairs' first groups lie in
        different B halves."""
        for spec in (
            _spec6(**_PAIRS),
            _spec6(tile_m=128, tile_n=128, tile_k=512, wave_map="pairs"),
            _spec6(256, 512, 2048, tile_m=64, tile_n=64, tile_k=512, wave_map="pairs"),
            _spec8(**_DIRECT, wave_map="pairs"),
        ):
            with self.subTest(spec.kernel_name()):
                tm, tn = spec.tile_m, spec.tile_n
                group = spec.n_group
                hits = np.zeros((tm, tn), dtype=np.int64)
                for wave in range(4):
                    wave_m, wave_n = wave_coords(spec, wave)
                    rows = RowMaps(
                        types.SimpleNamespace(
                            ops=IntOps, spec=spec, wave_m=wave_m, wave_n=wave_n
                        )
                    )
                    self.assertEqual(wave_m, wave % 2)
                    for lane in range(32):
                        n_local = lane % 16
                        cols = [
                            rows.n_row(rn, n_local) for rn in range(spec.wave_repeat_n)
                        ]
                        for first in range(0, spec.wave_repeat_n, group):
                            run = cols[first : first + group]
                            self.assertEqual(run, list(range(run[0], run[0] + group)))
                            self.assertEqual(
                                run[0] // (tn // 2), (first // group + wave_m) % 2
                            )
                        for rm, slot in itertools.product(
                            range(spec.wave_repeat_m), range(spec.atom.c_frag_len)
                        ):
                            m_local = slot // 8 * 16 + slot % 8 + lane // 16 * 8
                            row = rows.m_row(rm, m_local)
                            self.assertEqual(row // (tm // 2), wave_m)
                            for col in cols:
                                hits[row, col] += 1
                self.assertTrue((hits == 1).all())

    def test_panel_layout(self):
        """A panel slot holds four 32-row components of 384 B rows (16 B
        after every 4 rows) and then their scales; every row and scale word
        of the panel has its own offset inside the slot."""
        spec = _spec6(*_MAF_MED[0], cluster_dim=(4, 4, 1), **_PANELS, half_plr="none")
        layout = spec.lds_layout
        self.assertEqual(layout.a, layout.b)
        self.assertEqual(layout.a.rows, 32)
        self.assertEqual((layout.a.iterate_rows, layout.a.iterate_stride), (4, 1552))
        data_end = 4 * layout.a.stride
        self.assertEqual(layout.a_scale.base, data_end)
        rows = [layout.row_offset(IntOps, "a", row) for row in range(128)]
        self.assertEqual(len(set(rows)), 128)
        self.assertTrue(all(0 <= r and r + 384 <= data_end for r in rows))
        words = {
            layout.scale_offset(IntOps, "a_scale", step, row)
            for step in range(spec.k_steps)
            for row in range(128)
        }
        self.assertEqual(len(words), 128 * spec.k_steps)
        self.assertTrue(all(data_end <= w < layout.buffer_bytes for w in words))
        self.assertLessEqual(spec.lds_bytes, 320 * 1024)

    def test_panel_walk_schedule(self):
        """Over six K tiles: each tile's passes cover the four quadrants once,
        consecutive passes share a panel, a slot is refilled only after the
        tile's last use of it and never with a panel the next pass reads, and
        with the TDM waits of :meth:`PanelWalk.waits` (read-ahead: the wait
        before pass j precedes the refills after pass j - 1) every pass finds
        the panels of its own tile loaded."""
        waits = PanelWalk.waits()
        loads = PanelWalk.LOADS_PER_PANEL
        issued = [(0, *panel) for panel in PanelWalk.first_use(0)]
        done = 0
        pending = []
        slots = {}
        for tile in range(6):
            parity = tile % 2
            order = PanelWalk.ORDERS[parity]
            self.assertEqual(sorted(order), [(0, 0), (0, 1), (1, 0), (1, 1)])
            for j, (a, b) in enumerate(order):
                if j:
                    self.assertTrue(a == order[j - 1][0] or b == order[j - 1][1])
                count = waits["prologue" if tile == 0 and j == 0 else (parity, j)]
                if count is not None:
                    done = max(done, len(issued) - count // loads)
                for entry in issued[:done]:
                    slots[PanelWalk.slot(entry[1], entry[2])] = entry[0]
                for operand, panel in (("a", a), ("b", b)):
                    with self.subTest(tile=tile, j=j, panel=(operand, panel)):
                        self.assertEqual(
                            slots.get(PanelWalk.slot(operand, panel)), tile
                        )
                for entry in pending:
                    self.assertNotIn(entry[1:], (("a", a), ("b", b)))
                issued += pending
                pending = [(tile + 1, *panel) for panel in PanelWalk.refills(parity)[j]]
                later = {
                    (op, i) for pair in order[j + 1 :] for op, i in zip("ab", pair)
                }
                for panel in pending:
                    self.assertNotIn(panel[1:], later)
            self.assertEqual(
                sorted(p for j in range(4) for p in PanelWalk.refills(parity)[j]),
                [("a", 0), ("a", 1), ("b", 0), ("b", 1)],
            )

    def test_half_layout(self):
        """With halves an A slot is a panel slot; a B slot holds both B
        panels as rows 0..255 (eight 32-row components) and then their
        scales. Each wave's TDM load of a B panel's component lands where
        the reader looks for those rows and scales, and the slots fill the
        LDS without overlapping."""
        spec = _spec6(*_MAF_MED[0], cluster_dim=(4, 4, 1), **_HALVES, half_plr="none")
        layout = spec.lds_layout
        panels = _spec6(*_MAF_MED[0], cluster_dim=(4, 4, 1), **_PANELS, half_plr="none")
        self.assertEqual(layout.buffer_bytes, panels.lds_layout.buffer_bytes)
        self.assertEqual(layout.a, layout.b)
        data_end = 8 * layout.b.stride
        self.assertEqual(layout.b_scale.base, data_end)
        pair = HalfWalk.pair_bytes(layout)
        rows = [layout.row_offset(IntOps, "b", row) for row in range(256)]
        self.assertEqual(len(set(rows)), 256)
        self.assertTrue(all(0 <= r and r + 384 <= data_end for r in rows))
        words = {
            layout.scale_offset(IntOps, "b_scale", step, row)
            for step in range(spec.k_steps)
            for row in range(256)
        }
        self.assertEqual(len(words), 256 * spec.k_steps)
        self.assertTrue(all(data_end <= w < pair for w in words))
        for parity, panel, wave in itertools.product((0, 1), (0, 1), range(4)):
            with self.subTest(parity=parity, panel=panel, wave=wave):
                b_slot = HalfWalk.bases(layout, 0, parity)[1]
                data_at, scale_at = HalfWalk.at(layout, "b", panel, parity)
                row = 128 * panel + 32 * wave
                self.assertEqual(
                    data_at + layout.component_base(IntOps, layout.b, wave),
                    b_slot + layout.row_offset(IntOps, "b", row),
                )
                self.assertEqual(
                    scale_at + layout.component_base(IntOps, layout.b_scale, wave),
                    b_slot + layout.scale_offset(IntOps, "b_scale", 0, row),
                )
        spans = sorted(
            [(HalfWalk.at(layout, "a", h, 0)[0], layout.buffer_bytes) for h in (0, 1)]
            + [(HalfWalk.bases(layout, 0, p)[1], pair) for p in (0, 1)]
        )
        for (start, size), (after, _) in zip(spans, spans[1:]):
            self.assertLessEqual(start + size, after)
        self.assertEqual(spans[-1][0] + spans[-1][1], spec.lds_bytes)
        self.assertLessEqual(spec.lds_bytes, 320 * 1024)

    def test_half_walk_schedule(self):
        """Over six K tiles, with the TDM waits of :meth:`HalfWalk.waits`
        (read-ahead: the wait before a half precedes the refills after the
        half before it), every half finds its tile's A panel and both B
        panels loaded, and no refill lands in a slot the next half reads."""
        waits = HalfWalk.waits()
        self.assertEqual(waits, {"prologue": 6, 1: 4, 0: 0})
        loads = HalfWalk.LOADS_PER_PANEL

        def slot(tile, operand, panel):
            return ("a", panel) if operand == "a" else ("b", tile % 2, panel)

        issued = [(tile, op, panel) for op, panel, tile in HalfWalk.prologue()]
        self.assertEqual(
            issued,
            [
                (0, "a", 0),
                (0, "b", 0),
                (0, "b", 1),
                (0, "a", 1),
                (1, "b", 0),
                (1, "b", 1),
            ],
        )
        done, pending, slots = 0, [], {}
        for tile in range(6):
            for half in (0, 1):
                count = waits["prologue" if tile == 0 and half == 0 else half]
                done = max(done, len(issued) - count // loads)
                for entry in issued[:done]:
                    slots[slot(*entry)] = entry[0]
                reads = [slot(tile, "a", half)] + [slot(tile, "b", p) for p in (0, 1)]
                for read in reads:
                    with self.subTest(tile=tile, half=half, slot=read):
                        self.assertEqual(slots.get(read), tile)
                for entry in pending:
                    self.assertNotIn(slot(*entry), reads)
                issued += pending
                pending = [
                    (tile + ahead, op, panel)
                    for op, panel, ahead in HalfWalk.refills(half)
                ]

    def test_ring_layout(self):
        """walk "ring": a part is 16 groups of 8 rows of 512 B, 16 B after
        each group; a part's scales are two tiles of 8 K-steps of 544 B (512 B
        and a 16 B pad after every 256 B). The four scale rings come first,
        then the four data parts, all on 256 B boundaries, so a split base
        (512 B under a lane's base) stays positive, and the LDS holds it all.
        Rows and scales of a part never overlap."""
        ring = RingWalk
        self.assertEqual((ring.PERIOD, ring.PERIOD_LINES), (16, 6))
        self.assertEqual(ring.group_bytes(), 4112)
        self.assertEqual(ring.part_bytes(), 65792)
        self.assertEqual(ring.scale_step_bytes(), 544)
        self.assertEqual(ring.scale_ring_bytes(), 8704)
        self.assertEqual(ring.lds_bytes(), 297984)
        self.assertLessEqual(ring.lds_bytes(), 320 * 1024)
        spans = []
        for operand, component in ring.PARTS:
            scale = ring.scale_base(operand, component)
            data = ring.data_base(operand, component)
            spans += [(scale, ring.scale_ring_bytes()), (data, ring.part_bytes())]
            self.assertEqual(scale % 256, 0)
            self.assertEqual(data % 256, 0)
            self.assertGreaterEqual(data, ring.PITCH)
        spans.sort()
        self.assertEqual(spans[0][0], 0)
        for (start, size), (after, _) in zip(spans, spans[1:]):
            self.assertEqual(start + size, after)
        self.assertEqual(sum(size for _, size in spans), ring.lds_bytes())
        rows = [ring.row_offset(row) for row in range(ring.PART_ROWS)]
        self.assertEqual(len(set(rows)), ring.PART_ROWS)
        self.assertEqual(rows[-1] + ring.PITCH + ring.ROW_PAD, ring.part_bytes())
        scales = {
            ring.scale_offset(step, row) + j
            for step in range(2 * ring.SCALE_STEPS)
            for row in range(ring.PART_ROWS)
            for j in range(4)
        }
        self.assertEqual(len(scales), 2 * ring.SCALE_STEPS * ring.PART_ROWS * 4)
        self.assertLess(max(scales), ring.scale_ring_bytes())

    def test_ring_schedule(self):
        """A period is 16 K-steps and six lines. Line L is last read in
        K-step 2, 5, 7, 10, 13 or 15 of a period, and that K-step's late
        refill loads line L + 2 into its slot (then the next scale tile at
        the end of a scale tile), which the K-steps first read three K-steps
        later; the K-step before each refill signals the cluster. Only
        K-steps 5 and 10 read across the end of a row's 512 B. The TDM waits
        are the same in every period."""
        ring = RingWalk
        self.assertEqual([ring.last(line) for line in range(6)], [2, 5, 7, 10, 13, 15])
        self.assertEqual(
            [ring.first(line + 2) for line in range(6)], [5, 8, 10, 13, 16, 18]
        )
        refills = {
            step: ring.refills(step)
            for step in range(ring.PERIOD)
            if ring.refills(step)
        }
        self.assertEqual(
            refills,
            {
                2: [("line", 2)],
                5: [("line", 3)],
                7: [("line", 4), ("scales", 2)],
                10: [("line", 5)],
                13: [("line", 6)],
                15: [("line", 7), ("scales", 3)],
            },
        )
        for step in range(3 * ring.PERIOD):
            period, local = divmod(step, ring.PERIOD)
            shifted = [
                (kind, index + period * (6 if kind == "line" else 2))
                for kind, index in ring.refills(local)
            ]
            self.assertEqual(ring.refills(step), shifted)
            for kind, index in ring.refills(step):
                first = ring.first_use((kind, index))
                self.assertGreaterEqual(first - step, 3, (step, kind, index))
        self.assertEqual(
            [step for step in range(ring.PERIOD) if ring.signals(step)],
            [1, 4, 6, 9, 12, 14],
        )
        self.assertEqual(
            ring.prologue(),
            [("line", 0), ("scales", 0), ("line", 1), ("scales", 1)],
        )
        waits = {key: count for key, count in ring.waits().items() if count is not None}
        self.assertEqual(waits, {"prologue": 2, 1: 1, 4: 0, 7: 0, 9: 1, 12: 0, 15: 0})
        for k_reads in K_READS:
            spec = _spec6(*_MAF_MED[0], **_RING | {"k_reads": k_reads})
            wraps = {
                step
                for step in range(ring.PERIOD)
                for offset, _ in spec.lane_reads
                for half in (0, 1)
                if ring.offset(step) + offset + half * spec.lane_half_bytes
                >= ring.PITCH
            }
            with self.subTest(k_reads=k_reads):
                self.assertEqual(wraps, {5, 10})

    def test_ring_reads_find_tdm_writes(self):
        """Over three periods of one part (A and B parts are alike), with the
        TDM waits of :meth:`RingWalk.waits` (the wait in K-step q comes before
        the K-step's late refill, and the next K-step's reads after both):
        every byte a lane reads, of its rows at its K-step's K and of its
        scales, is the byte the TDM wrote there, and no load in flight writes
        the slot it reads. Contiguous and chunked reads."""
        ring = RingWalk
        rows = ring.PART_ROWS

        def label(row, byte):
            return row << 16 | byte

        for k_reads in K_READS:
            spec = _spec6(*_MAF_MED[0], **_RING | {"k_reads": k_reads})
            half_bytes = spec.lane_half_bytes
            data = np.full(ring.part_bytes(), -1, np.int64)
            scales = np.full(ring.scale_ring_bytes(), -1, np.int64)
            issued, pending = [], []

            def land(load):
                kind, index = load
                if kind == "line":
                    first = index * ring.LINE
                    for row in range(rows):
                        at = ring.row_offset(row) + index % ring.SLOTS * ring.LINE
                        data[at : at + ring.LINE] = [
                            label(row, x) for x in range(first, first + ring.LINE)
                        ]
                    return
                for local in range(ring.SCALE_STEPS):
                    step = index * ring.SCALE_STEPS + local
                    for row in range(rows):
                        at = ring.scale_offset(step, row)
                        scales[at : at + 4] = [
                            label(row, 4 * step + j) for j in range(4)
                        ]

            def issue(loads):
                issued.extend(loads)
                pending.extend(loads)

            def wait(count):
                if count is None:
                    return
                for load in issued[: len(issued) - count]:
                    if load in pending:
                        land(load)
                        pending.remove(load)

            def busy(kind, slot, slots):
                return any(k == kind and i % slots == slot for k, i in pending)

            def check(step):
                for half, lane, repeat in itertools.product(
                    (0, 1), range(16), range(8)
                ):
                    row = lane * 8 + repeat
                    base = ring.row_offset(lane * 8) + half * half_bytes
                    for offset, words in spec.lane_reads:
                        split, imm = ring.read(step, offset, half_bytes)
                        at = (
                            base - split * half * ring.PITCH + ring.PITCH * repeat + imm
                        )
                        start = ring.row_offset(row)
                        self.assertGreaterEqual(at, start)
                        self.assertLessEqual(at + 4 * words, start + ring.PITCH)
                        self.assertFalse(
                            busy("line", (at - start) // ring.LINE, ring.SLOTS)
                        )
                        k = step * ring.STEP_BYTES + half * half_bytes + offset
                        np.testing.assert_array_equal(
                            data[at : at + 4 * words],
                            [label(row, k + j) for j in range(4 * words)],
                            err_msg=f"{k_reads} step {step} row {row} half {half}",
                        )
                tile = step // ring.SCALE_STEPS
                self.assertFalse(
                    busy("scales", tile % ring.SCALE_SLOTS, ring.SCALE_SLOTS)
                )
                for lane in range(16):
                    for chunk in (0, 16):
                        at = ring.scale_offset(step, lane * 8) + chunk
                        want = [
                            label(lane * 8 + chunk // 4 + i, 4 * step + j)
                            for i in range(4)
                            for j in range(4)
                        ]
                        np.testing.assert_array_equal(scales[at : at + 16], want)

            waits = ring.waits()
            issue(ring.prologue())
            wait(waits["prologue"])
            check(0)
            for step in range(3 * ring.PERIOD - 1):
                wait(waits[step % ring.PERIOD])
                issue(ring.refills(step))
                check(step + 1)

    def test_ring_reads_spread_over_bank_groups(self):
        """Every ring read and scale read of a lane half's 16 lanes starts on
        16 different 16 B bank groups (64 banks of 4 B), at every K-step of a
        period and every repeat."""
        ring = RingWalk
        for k_reads in K_READS:
            spec = _spec6(*_MAF_MED[0], **_RING | {"k_reads": k_reads})
            half_bytes = spec.lane_half_bytes
            for step, half, repeat in itertools.product(
                range(ring.PERIOD), (0, 1), range(8)
            ):
                for offset, _ in spec.lane_reads:
                    split, imm = ring.read(step, offset, half_bytes)
                    base = half * half_bytes - split * half * ring.PITCH
                    groups = {
                        (ring.row_offset(lane * 8) + base + ring.PITCH * repeat + imm)
                        // 16
                        % 16
                        for lane in range(16)
                    }
                    with self.subTest(k_reads=k_reads, step=step, half=half):
                        self.assertEqual(len(groups), 16)
        for step, chunk in itertools.product(range(ring.PERIOD), (0, 16)):
            groups = {
                (ring.scale_offset(step, lane * 8) + chunk) // 16 % 16
                for lane in range(16)
            }
            self.assertEqual(len(groups), 16)

    def test_ring_validate_and_name(self):
        """validate() takes walk "ring" by :meth:`RingWalk.reason`; the
        kernel name ends in the ring's tag and the LDS is the ring's."""
        spec = _spec6(*_MAF_MED[1], **_RING, scale_swap="wait")
        self.assertIsNone(validate(spec))
        self.assertIn("_rg", spec.kernel_name())
        self.assertNotIn("_rg", replace(spec, walk="tile").kernel_name())
        self.assertEqual(spec.lds_bytes, RingWalk.lds_bytes())
        for bad in (
            replace(spec, K=65536 + 1024),
            replace(spec, read_bases="loop"),
            replace(spec, tile_k=512),
        ):
            with self.subTest(bad=bad.kernel_name()):
                self.assertIsNotNone(validate(bad))
                self.assertEqual(validate(bad), RingWalk.reason(bad))

    def test_ring_reason(self):
        """walk "ring" takes FP6 at the 256x256x256 tile, wave_group 2x2 and
        the 16x16 WMMA, read-ahead with B in halves (ahead or late), fixed
        read bases, contiguous reads, row-major A and B, whole 256 tiles and K
        a multiple of 2048 (one period); nothing else."""
        for (m, n, k), order, swap, cluster in itertools.product(
            _MAF_MED, ("ahead", "late"), SCALE_SWAPS, ((4, 4, 1), (2, 4, 1))
        ):
            spec = _spec6(
                m,
                n,
                k,
                **_RING | {"half_b_order": order},
                scale_swap=swap,
                cluster_dim=cluster,
            )
            with self.subTest(spec=spec.kernel_name()):
                self.assertIsNone(RingWalk.reason(spec))
        base = _spec6(*_MAF_MED[0], **_RING, scale_swap="wait")
        self.assertIsNone(RingWalk.reason(base))
        self.assertIsNone(RingWalk.reason(replace(base, wmma_reuse="b")))
        for fields, words in (
            (
                {
                    "a_dtype": "fp8",
                    "b_dtype": "fp8",
                    "k_reads": "chunked",
                    "scale_swap": "read",
                },
                "FP6",
            ),
            ({"tile_k": 512}, "256x256x256"),
            ({"tile_n": 128}, "256x256x256"),
            ({"wave_group": (4, 1)}, "wave_group"),
            ({"half_plr": "none", "half_b_order": "ahead"}, "half_plr"),
            ({"read_bases": "loop"}, "read_bases"),
            ({"k_reads": "chunked", "scale_swap": "read"}, "contiguous"),
            ({"lds_buffers": 3}, "lds_buffers"),
            ({"tiled_operands": "ab"}, "row-major"),
            ({"a_lds_pad": LdsPad(16, 1536)}, "pads"),
            ({"M": 3000}, "multiples of 256"),
            ({"K": 65536 - 1024}, "multiple of 2048"),
            ({"cluster_dim": (3, 4, 1)}, "tile:"),
        ):
            with self.subTest(fields=fields):
                reason = RingWalk.reason(replace(base, **fields))
                self.assertIsNotNone(reason)
                self.assertIn(words, reason)

    def test_scale_register_halves(self):
        """Lanes 0-15 hold the first half of an operand's repeats, lanes
        16-31 the second; repeat ``i`` is word ``i % h`` of the half its
        selector names, so the halves' words cover every repeat once."""
        for repeats in (1, 2, 4, 8, 16):
            h = ScaleRegisters.half(repeats)
            with self.subTest(repeats=repeats):
                self.assertEqual(h, max(1, repeats // 2))
                places = [ScaleRegisters.place(repeats, i) for i in range(repeats)]
                self.assertEqual(len(set(places)), repeats)
                for i, (word, selector) in enumerate(places):
                    self.assertLess(word, h)
                    self.assertIn(selector, (0, 1) if repeats > 1 else (0,))
                    # A half-s lane's words are rows s * h .. s * h + h - 1.
                    self.assertEqual(selector * h + word, i)

    def test_i64_offsets_past_2_31(self):
        """A and B bytes and D elements past 2^31 switch that operand to i64
        offsets; the scales stay i32 and bound the problem."""
        at_limit = {
            "D": (32768, 65536, 1024),
            "A": (4096, 256, 1 << 20),
            "B": (256, 4096, 1 << 20),
        }
        past_limit = {
            "D": (32768, 65536 + 256, 1024),
            "A": (4096, 256, (1 << 20) + 512),
            "B": (256, 4096 + 256, 1 << 20),
        }
        for label, problem in at_limit.items():
            spec = _spec(*problem, cluster_dim=(1, 1, 1))
            with self.subTest(f"{label} at 2^31"):
                self.assertIsNone(validate(spec))
                self.assertEqual((spec.wide_a, spec.wide_b, spec.wide_d), (0, 0, 0))
                self.assertNotRegex(_llvm(spec), _MUL_I64)
        for label, problem in past_limit.items():
            spec = _spec(*problem, cluster_dim=(1, 1, 1))
            with self.subTest(f"{label} past 2^31"):
                self.assertIsNone(validate(spec))
                wide = {"A": spec.wide_a, "B": spec.wide_b, "D": spec.wide_d}
                self.assertEqual([name for name, on in wide.items() if on], [label])
                self.assertRegex(_llvm(spec), _MUL_I64)
        # Partial tiles count whole: M*N is under 2^31, the tiles are not.
        spec = _spec(32768 - 100, 65536 + 100, 1024, cluster_dim=(1, 1, 1))
        self.assertLess(spec.M * spec.N, 1 << 31)
        self.assertTrue(spec.wide_d)
        # Scale bytes: M (or N) * ceil(K/128) * 4.
        self.assertIsNone(validate(_spec(4096, 256, 1 << 24, cluster_dim=(1, 1, 1))))
        reason = validate(_spec(4096, 256, (1 << 24) + 512, cluster_dim=(1, 1, 1)))
        self.assertIn("scale bytes", reason or "")
        # Shapes the converted hipBLASLt problem set holds: D spans more than
        # 2^31 elements.
        for problem, tile in (
            ((6016, 721536, 1024), (128, 128, 512)),
            ((5760, 838784, 3584), (128, 128, 512)),
            ((61696, 51968, 4608), (256, 256, 512)),
            ((951552, 3712, 4352), (256, 128, 256)),
            ((44416, 56832, 1280), (128, 256, 256)),
            ((61056, 48256, 5376), (128, 128, 256)),
            ((7552, 540544, 1024), (128, 128, 512)),
        ):
            spec = _spec(
                *problem,
                tile_m=tile[0],
                tile_n=tile[1],
                tile_k=tile[2],
                cluster_dim=(1, 1, 1),
            )
            with self.subTest(problem=problem):
                self.assertIsNone(validate(spec))
                self.assertTrue(spec.wide_d)

    def test_partial_tile_stores_cover_d_once(self):
        """Every D element is stored by exactly one lane vector, none outside:
        the epilogue's row and column map with its masks and store width."""
        for (m, n), out_dtype, (tm, tn, wg, mma) in itertools.product(
            ((300, 300), (256, 201), (300, 512), (512, 512), (300, 16)),
            OUT_DTYPES,
            (
                (128, 128, (2, 2), "32x16"),
                (256, 128, (1, 4), "32x16"),
                (128, 256, (4, 1), "32x16"),
                (128, 16, (4, 1), "32x16"),
                (128, 128, (2, 2), "16x16"),
                (64, 16, (4, 1), "16x16"),
            ),
        ):
            spec = _spec(
                m,
                n,
                2048,
                tile_m=tm,
                tile_n=tn,
                wave_group=wg,
                out_dtype=out_dtype,
                cluster_dim=(1, 1, 1),
                half_plr="b" if tn // wg[1] % 32 == 0 else "none",
                mma=mma,
            )
            tile = (tm, tn, wg, mma)
            with self.subTest(m=m, n=n, out_dtype=out_dtype, tile=tile):
                self.assertIsNone(validate(spec))
                rm_count, rn_count = spec.wave_repeat_m, spec.wave_repeat_n
                atom_m, slots = spec.atom.m, spec.atom.c_frag_len
                width = spec.store_width
                self.assertEqual(n % width, 0)
                hits = np.zeros((m, n), dtype=np.int64)
                gx, gy, _ = (-(-n // tn), -(-m // tm), 1)
                for by, bx, wave, lane in itertools.product(
                    range(gy), range(gx), range(4), range(32)
                ):
                    wave_m, wave_n = divmod(wave, spec.warp_n)
                    col = bx * tn + wave_n * rn_count * 16 + (lane % 16) * rn_count
                    for rm, slot in itertools.product(range(rm_count), range(slots)):
                        m_local = slot // 8 * 16 + slot % 8 + lane // 16 * 8
                        row = (
                            by * tm
                            + wave_m * rm_count * atom_m
                            + m_local * rm_count
                            + rm
                        )
                        for begin in range(0, rn_count, width):
                            c = col + begin
                            if row < m and c < n:
                                self.assertLessEqual(c + width, n)
                                hits[row, c : c + width] += 1
                self.assertTrue((hits == 1).all())

    def test_partial_tile_build(self):
        """Edge problems guard each D vector and C load; whole tiles do not."""
        whole = _llvm(_spec(cluster_dim=(1, 1, 1), out_dtype="bf16"))
        for problem, out_dtype, vector in (
            ((1000, 1000, 2048), "bf16", "<8 x bfloat>"),
            ((1024, 1001, 2048), "bf16", "bfloat"),
            ((1000, 1024, 2048), "fp8e4m3", "<8 x i8>"),
        ):
            spec = _spec(*problem, cluster_dim=(1, 1, 1), out_dtype=out_dtype)
            with self.subTest(problem=problem):
                self.assertTrue(spec.edge_m or spec.edge_n)
                llvm = _llvm(spec)
                stores = len(re.findall(r"\bstore [^\n]*addrspace\(1\)", llvm))
                self.assertEqual(
                    stores,
                    len(re.findall(r"\bstore [^\n]*addrspace\(1\)", whole))
                    * 8
                    // spec.store_width,
                )
                self.assertIn(f"store {vector}", llvm)
        self.assertEqual(_spec(1024, 1001, 2048).store_width, 1)
        self.assertEqual(_spec(1024, 1000, 2048).store_width, 8)

    def test_derived_defaults(self):
        spec = _spec()
        self.assertEqual(spec.signal_wmmas, 6)
        self.assertEqual(spec.fp8_convert_width, 8)
        self.assertEqual(spec.reads_per_step, 35)
        small = _spec(tile_m=128, tile_n=128)
        self.assertEqual(small.signal_wmmas, 4)
        self.assertEqual(small.fp8_convert_width, 4)

    def test_derived_lds_pads_match_the_benchmark_solution(self):
        # hipBLASLt 00_Final.yaml (Tensile A = rocKE B): LdsBlockSizePerPadA
        # 2048, LdsBlockSizePerPadB 1024, LdsPadA/B 32 FP4 = 16 B, LdsPadMXSA
        # 16 per LdsBlockSizePerPadMXSA 256, LdsPadMXSB 0.
        pads = derived_lds_pads(_spec(*_BENCH))
        self.assertEqual(pads["a"], LdsPad(16, 1024))
        self.assertEqual(pads["b"], LdsPad(16, 2048))
        self.assertEqual(pads["b_scale"], LdsPad(16, 256))
        self.assertEqual(pads["a_scale"], LdsPad())
        self.assertEqual(_spec(*_BENCH).lds_bytes, 282624)

    def test_fp6_rows_take_a_pad_every_bank_cycle_of_lanes(self):
        """FP6 rows (96 B per 128 K) take 16 B after every 8, 4, 8 or 4 rows:
        a lane half's lanes read rows 4 apart (the wave repeat) here, and the
        pad comes where a lane's first bank repeats. The blocks are no power
        of two, so they are TDM iterate mode. A power-of-two block that cuts a
        row stays rejected; "no pad" stays valid."""
        for tile_k, rows in ((128, 8), (256, 4), (384, 8), (512, 4)):
            spec = _spec6(tile_m=128, tile_n=128, tile_k=tile_k)
            row_bytes = spec.k_bytes(tile_k)
            with self.subTest(tile_k=tile_k):
                pads = derived_lds_pads(spec)
                self.assertEqual(pads["a"], LdsPad(16, rows * row_bytes))
                self.assertEqual(pads["b"], pads["a"])
                for name in ("a", "b"):
                    op = spec.lds_layout.operand(name)
                    self.assertEqual(op.iterate_rows, rows)
                    self.assertEqual(op.iterate_stride, rows * row_bytes + 16)
        self.assertIsNone(validate(_spec6(a_lds_pad=LdsPad(), b_lds_pad=LdsPad())))
        self.assertIsNone(validate(_spec6(a_lds_pad=LdsPad(32, 1536))))
        for pad in (LdsPad(16, 512), LdsPad(16, 1000), LdsPad(16, 192 * 3)):
            with self.subTest(pad=pad):
                self.assertIsNotNone(validate(_spec6(b_lds_pad=pad)))

    def test_fp6_pads_put_a_lane_halfs_rows_on_every_bank_group(self):
        """The 16 lanes of a lane half read rows wave_repeat_m (A) or n_group
        (B) apart, at every repeat: under the derived pads they start on 16
        different 16 B bank groups (64 banks of 4 B)."""
        for fields in (
            {"tile_m": 128, "tile_n": 256, "tile_k": 512},
            {"tile_k": 256},
            {"tile_k": 384},
            {"tile_m": 128, "tile_n": 128, "tile_k": 512},
            {"tile_m": 256, "tile_n": 128, "tile_k": 512},
            {"tile_m": 64, "tile_n": 64, "tile_k": 512},
            {"tile_m": 256, "tile_n": 16, "tile_k": 512, "wave_group": (4, 1)},
            {**_PAIRS, "half_plr": "b", "tile_k": 256},
        ):
            spec = _spec6(**{**_DIRECT, **fields})
            self.assertIsNone(validate(spec))
            for name, repeats, stride in (
                ("a", spec.wave_repeat_m, spec.wave_repeat_m),
                ("b", spec.wave_repeat_n, spec.n_group),
            ):
                for repeat in range(repeats):
                    group, j = divmod(repeat, stride)
                    starts = {
                        spec.lds_layout.row_offset(
                            IntOps(), name, (group * 16 + lane) * stride + j
                        )
                        // 16
                        % 16
                        for lane in range(16)
                    }
                    with self.subTest(kernel=spec.kernel_name(), op=name, rep=repeat):
                        self.assertEqual(len(starts), 16)

    def test_kernel_name_encodes_non_default_tuning(self):
        base = _spec().kernel_name()
        self.assertTrue(base.endswith("_t256x256x512_c4x4"), base)
        tuned = _spec(cluster_dim=(2, 2, 1), gl2_prefetch_distance=0)
        self.assertTrue(tuned.kernel_name().endswith("_c2x2_pf0"), tuned.kernel_name())


class MxGemmLayoutTest(unittest.TestCase):
    def test_reads_land_where_tdm_writes(self):
        """Row and scale offsets agree with the TDM's padded writes, every
        lane read is contiguous, and the regions are disjoint and in bounds."""
        specs = (
            list(_valid_geometries())
            + list(_valid_geometries("16x16"))
            + list(_valid_geometries("16x16", a_dtype="fp8", b_dtype="fp8"))
            + list(
                _valid_geometries(
                    "16x16",
                    a_dtype="bf8",
                    b_dtype="fp8",
                    local_prefetch=0,
                    half_plr="none",
                )
            )
            + list(_valid_geometries("16x16", a_dtype="fp6", b_dtype="fp6"))
            + list(
                _valid_geometries(
                    "16x16", a_dtype="fp6", b_dtype="fp6", k_reads="contiguous"
                )
            )
            + [
                _spec6(a_lds_pad=LdsPad(), b_lds_pad=LdsPad()),
                _spec6(tile_k=384, a_lds_pad=LdsPad(32, 1152)),
                _spec(a_lds_pad=LdsPad(), b_lds_pad=LdsPad()),
                _spec(a_lds_pad=LdsPad(32, 2048), b_scale_lds_pad=LdsPad()),
                _spec(
                    tile_m=256,
                    tile_n=128,
                    wave_group=(1, 4),
                    a_scale_lds_pad=LdsPad(16, 512),
                ),
            ]
        )
        ops = IntOps()
        for spec in specs:
            layout = spec.lds_layout
            with self.subTest(spec.kernel_name()):
                used = []
                for name in ("a", "b"):
                    op = layout.operand(name)
                    for component in range(2):
                        base = layout.component_base(ops, op, component)
                        for local in range(op.rows):
                            row = component * op.rows + local
                            start = local * layout.row_bytes
                            land = base + _landing(
                                layout,
                                op.pad,
                                start,
                                op.iterate_rows,
                                op.iterate_stride,
                            )
                            end = base + _landing(
                                layout,
                                op.pad,
                                start + layout.row_bytes - 1,
                                op.iterate_rows,
                                op.iterate_stride,
                            )
                            self.assertEqual(layout.row_offset(ops, name, row), land)
                            self.assertEqual(end - land, layout.row_bytes - 1)
                            used.append((land, end))
                for name, chunk in (
                    ("a_scale", 4 * min(spec.wave_repeat_m, 4)),
                    ("b_scale", 4 * min(spec.wave_repeat_n, 4)),
                ):
                    op = layout.operand(name)
                    for component, step in itertools.product(
                        range(2), range(spec.k_steps)
                    ):
                        base = layout.component_base(ops, op, component)
                        for local in range(0, op.rows, chunk // 4):
                            row = component * op.rows + local
                            start = step * op.rows * 4 + local * 4
                            land = base + _landing(layout, op.pad, start, 0, 0)
                            end = base + _landing(
                                layout, op.pad, start + chunk - 1, 0, 0
                            )
                            self.assertEqual(
                                layout.scale_offset(ops, name, step, row), land
                            )
                            self.assertEqual(end - land, chunk - 1)
                            used.append((land, end))
                used.sort()
                for (_, end), (begin, _) in itertools.pairwise(used):
                    self.assertLess(end, begin)
                self.assertGreaterEqual(used[0][0], 0)
                self.assertLess(used[-1][1], layout.buffer_bytes)

    def test_lane_chunks_match_the_scaled_matrix_layout(self):
        """Each lane's K chunk reads (``chunk_offsets`` plus 16 B per lane
        half, within a K-step's ``step_bytes``) hold exactly the K elements
        that ``scaled_matrix_layout`` assigns its fragment slots."""
        for spec, offsets in (
            (_spec(), (0, 32)),
            (_spec(mma="16x16"), (0, 32)),
            (_spec8(), (0, 32, 64, 96)),
            (_spec8(a_dtype="bf8", b_dtype="bf8"), (0, 32, 64, 96)),
        ):
            layout = spec.fragment_layout
            with self.subTest(spec.kernel_name()):
                self.assertEqual(spec.chunk_offsets, offsets)
                self.assertEqual(layout.chunk_bytes, 16)
                self.assertEqual(spec.step_bytes, 128 * spec.elem_bits // 8)
                per_byte = 8 // spec.elem_bits
                for lane, slot in itertools.product(
                    range(32), range(layout.fragment.count)
                ):
                    row, k_index = layout.coord(lane, slot)
                    chunk, element = divmod(slot, layout.chunk_elements)
                    byte = (
                        lane // 16 * layout.chunk_bytes
                        + spec.chunk_offsets[chunk]
                        + element // per_byte
                    )
                    self.assertEqual(row, lane % 16)
                    self.assertEqual(k_index // per_byte, byte)
                    self.assertLess(byte, spec.step_bytes)

    def test_fp6_lane_reads_hold_the_wmma_k_values(self):
        """FP6 "chunked" reads put in each fragment slot the K value that
        ``scaled_matrix_layout`` assigns it; "contiguous" reads put there the
        value of K group 0, 2, 1, 3 for the WMMA's group 0, 1, 2, 3, the
        same for A and B. FP4 and FP8 reads match the layout as well."""
        regroup = np.array([0, 2, 1, 3])
        for spec in (_spec6(), _spec6(k_reads="contiguous"), _spec(), _spec8()):
            layout = spec.fragment_layout
            got = _lane_k(spec)
            with self.subTest(spec.kernel_name()):
                for lane, slot in itertools.product(
                    range(32), range(layout.fragment.count)
                ):
                    row, k_index = layout.coord(lane, slot)
                    self.assertEqual(row, lane % 16)
                    if spec.contiguous_k:
                        k_index = regroup[k_index // 32] * 32 + k_index % 32
                    self.assertEqual(got[lane, slot], k_index)

    def test_contiguous_reads_with_swapped_scales_keep_the_product(self):
        """One WMMA of random A, B and per-32-K scales, computed as the
        hardware sees its operands: with "contiguous" reads and each scale
        word's bytes 1 and 2 swapped it equals the plain product; without the
        swap it does not."""
        rng = np.random.default_rng(6)
        a = rng.integers(-15, 16, (16, 128)).astype(np.float64)
        b = rng.integers(-15, 16, (16, 128)).astype(np.float64)
        sa = np.exp2(rng.integers(-3, 4, (16, 4)))
        sb = np.exp2(rng.integers(-3, 4, (16, 4)))
        expected = (a * np.repeat(sa, 32, axis=1)) @ (b * np.repeat(sb, 32, axis=1)).T
        spec = _spec6(k_reads="contiguous")
        layout = spec.fragment_layout
        held = _lane_k(spec)

        def wmma(swap):
            order = [0, 2, 1, 3] if swap else [0, 1, 2, 3]
            # Operand values by WMMA K position, gathered from the lanes.
            a_op, b_op = np.zeros_like(a), np.zeros_like(b)
            for lane, slot in itertools.product(range(32), range(64)):
                row, k_index = layout.coord(lane, slot)
                a_op[row, k_index] = a[row, held[lane, slot]]
                b_op[row, k_index] = b[row, held[lane, slot]]
            a_scale, b_scale = sa[:, order], sb[:, order]
            return (a_op * np.repeat(a_scale, 32, axis=1)) @ (
                b_op * np.repeat(b_scale, 32, axis=1)
            ).T

        np.testing.assert_array_equal(wmma(swap=True), expected)
        self.assertFalse(np.array_equal(wmma(swap=False), expected))
        swap = 0x03010200
        word = 0x44332211
        picked = bytes((word >> 8 * ((swap >> 8 * i) & 0xFF)) & 0xFF for i in range(4))
        self.assertEqual(picked, bytes((0x11, 0x33, 0x22, 0x44)))

    def test_straddling_rows_prefetch_every_line(self):
        """The GL2 rule for FP6 rows: per tile, the ceil(row / 256) lines
        ending at the tile's last byte (clamped to the row's end in the K
        tail). Over a row's tiles every 256 B line is reached and no address
        leaves the row. A row that starts inside a line may miss that first
        line: it holds only tile 0, which is loaded, never prefetched."""
        for tile_k, k_dim, row_start in itertools.product(
            (128, 256, 384, 512), (2048, 2176, 4096 + 384, 65536), (0, 64, 160)
        ):
            spec = _spec6(1024, 1024, k_dim, tile_m=128, tile_n=128, tile_k=tile_k)
            if validate(spec) is not None:
                continue
            row_bytes, packed = spec.k_bytes(tile_k), spec.packed_k
            lines = -(-row_bytes // 256)
            reached = set()
            for tile in range(-(-k_dim // tile_k)):
                last = min(spec.k_bytes(tile * tile_k) + row_bytes - 1, packed - 1)
                for j in range(lines):
                    offset = last - 256 * j
                    self.assertGreaterEqual(offset, 0)
                    reached.add((row_start + offset) // 256)
            every = set(range(row_start // 256, (row_start + packed - 1) // 256 + 1))
            with self.subTest(tile_k=tile_k, K=k_dim, row_start=row_start):
                self.assertLessEqual(reached, every)
                missed = every - reached
                self.assertLessEqual(
                    missed, {row_start // 256} if row_start % 256 else set()
                )

    def test_tiled_transfers_are_whole_lines_into_the_row_major_image(self):
        """With tiled_operands a component's K tile is its rows one after
        another, copied in transfers of whole 256 B lines; byte j of it (row
        j // row_bytes) lands where the row-major operand's row is read."""
        ops = IntOps()
        for spec in (
            _spec6(tile_m=256, tile_n=256, lds_buffers=3, half_b_order="late"),
            _spec6(tile_m=256, tile_n=128, tile_k=512),
            _spec6(tile_m=128, tile_n=128, tile_k=512, half_plr="none"),
            _spec6(tile_m=64, tile_n=128, tile_k=512, cluster_dim=(4, 2, 1)),
            _spec6(
                tile_n=16,
                tile_k=512,
                wave_group=(4, 1),
                cluster_dim=(1, 1, 1),
                **_DIRECT,
            ),
            _spec6(a_lds_pad=LdsPad(), b_lds_pad=LdsPad()),
            _spec8(),
            _spec8(tile_n=64, wave_group=(4, 1), cluster_dim=(1, 1, 1), **_DIRECT),
        ):
            layout = spec.lds_layout
            for name in ("a", "b"):
                op = layout.operand(name)
                unit, units = tiled_transfer(layout, name)
                size = unit * units
                j = np.arange(op.rows * layout.row_bytes)
                with self.subTest(spec.kernel_name(), operand=name):
                    self.assertEqual(unit % 256, 0)
                    self.assertEqual(len(j) % size, 0)
                    if op.iterate_rows:
                        self.assertEqual(size, op.iterate_rows * layout.row_bytes)
                        land = j // size * op.iterate_stride + j % size
                    else:
                        # One transfer of the component, padded as it is written.
                        self.assertEqual(size, len(j))
                        land = _landing(layout, op.pad, j, 0, 0)
                    base = layout.component_base(ops, op, 0)
                    rows = [
                        layout.row_offset(ops, name, row) - base
                        for row in range(op.rows)
                    ]
                    np.testing.assert_array_equal(
                        land,
                        np.asarray(rows)[j // layout.row_bytes] + j % layout.row_bytes,
                    )

    def test_tile_rows_stores_each_row_groups_k_tiles_in_turn(self):
        """tile_rows puts byte c of row r at (r // rows) * rows * K bytes +
        (c // row_bytes) * rows * row_bytes + (r % rows) * row_bytes +
        c % row_bytes; device_inputs applies it to the operands a spec
        tiles, with the spec's component rows and K tile."""
        operand = np.random.default_rng(1).integers(0, 256, (12, 24), dtype=np.uint8)
        tiled = tile_rows(operand, 4, 6).reshape(-1)
        row, col = np.meshgrid(np.arange(12), np.arange(24), indexing="ij")
        at = (row // 4) * 4 * 24 + (col // 6) * 4 * 6 + (row % 4) * 6 + col % 6
        np.testing.assert_array_equal(tiled[at], operand)
        spec = _spec6(
            256,
            256,
            2048,
            tile_m=128,
            tile_n=64,
            tile_k=512,
            cluster_dim=(1, 1, 1),
            tiled_operands="b",
        )
        self.assertIsNone(validate(spec))
        inputs = check_inputs(spec)
        a, b, *_ = device_inputs(inputs, spec)
        np.testing.assert_array_equal(a, inputs[0])
        np.testing.assert_array_equal(b, tile_rows(inputs[1], 32, 384))
        for got, want in zip(device_inputs(inputs), device_inputs(inputs, _spec6())):
            np.testing.assert_array_equal(got, want)


class MxGemmIrTest(unittest.TestCase):
    def test_benchmark_compile_options(self):
        """Expert scheduling in source order, with the regclass-priority
        allocator option."""
        self.assertEqual(
            _comgr_options_for_kernel(build_mx_gemm(_spec(*_BENCH))),
            [
                "-O3",
                "-mllvm",
                "-amdgpu-expert-scheduling-mode",
                "-mllvm",
                "-enable-misched=false",
                "-mllvm",
                "-enable-post-misched=false",
                "-mllvm",
                "-greedy-regclass-priority-trumps-globalness",
            ],
        )

    def test_16x16_atom(self):
        """``mma`` 16x16 issues the 16x16x128 FP4 WMMA (formats 4, padded
        16-word operands, 8 accumulators), twice the 32x16 atoms of a tile;
        a 16-wide tile with one N-repeat stores single elements."""
        wmma_32 = "@llvm.amdgcn.wmma.scale.f32.32x16x128.f4.v16f32.v16i32.v8i32"
        wmma_16 = "@llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32"
        default = _llvm(_spec())
        self.assertNotIn(wmma_16, default)
        spec = _spec(mma="16x16")
        self.assertTrue(spec.kernel_name().endswith("_mma16"))
        self.assertEqual((spec.wave_repeat_m, spec.wave_repeat_n), (8, 8))
        self.assertEqual((spec.a_scale_reads, spec.reads_per_step), (2, 36))
        llvm = _llvm(spec)
        self.assertIn('"s_wait_dscnt 36"', llvm)
        self.assertNotIn(wmma_32, llvm)
        calls = re.findall(_WMMA + r"[^\n]*", llvm)
        self.assertEqual(len(calls), 2 * len(re.findall(_WMMA, default)))
        for call in calls:
            self.assertIn(f"<8 x float> {wmma_16}(i32 4, <16 x i32> %", call)
            self.assertIn(", i32 4, <16 x i32> %", call)
        narrow = _spec(
            1024,
            16,
            2048,
            tile_m=256,
            tile_n=16,
            wave_group=(4, 1),
            half_plr="none",
            cluster_dim=(1, 2, 1),
            mma="16x16",
        )
        self.assertIsNone(validate(narrow))
        self.assertEqual((narrow.wave_repeat_m, narrow.wave_repeat_n), (4, 1))
        self.assertEqual(narrow.reads_per_step, 12)
        self.assertEqual(narrow.store_width, 1)
        llvm = _llvm(narrow)
        self.assertIn('"s_wait_dscnt 12"', llvm)
        self.assertRegex(llvm, r"\bstore i8 [^\n]*addrspace\(1\)")
        self.assertNotRegex(llvm, r"\bstore <[^\n]*addrspace\(1\)")

    def test_8bit_atoms(self):
        """FP8/BF8 issue only the 16x16x128 f8f6f4 WMMA with each operand's
        format (FP8 0, BF8 1) and 16 live words, as many as the FP4 16x16
        kernel of the same geometry (k_steps * rm * rn per tile), and wait
        for four reads per fragment row."""
        wmma_16 = "@llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32"
        lds_read = r"load <4 x i32>, ptr addrspace\(3\)"
        geometry = {"tile_m": 128, "tile_n": 128, "tile_k": 256}
        fp4_spec = _spec(mma="16x16", **geometry)
        fp4_llvm = _llvm(fp4_spec)
        fp4 = re.findall(_WMMA, fp4_llvm)
        # Every read is 16 B at rm = rn = 4; count the K-step read groups.
        groups, rest = divmod(
            len(re.findall(lds_read, fp4_llvm)), fp4_spec.reads_per_step
        )
        self.assertEqual(rest, 0)
        for a_dtype, b_dtype in _PAIRS_8BIT:
            spec = _spec8(a_dtype=a_dtype, b_dtype=b_dtype, **geometry)
            rm, rn = spec.wave_repeat_m, spec.wave_repeat_n
            with self.subTest(a_dtype=a_dtype, b_dtype=b_dtype):
                self.assertEqual(
                    spec.atom.op_id,
                    f"wmma_gfx1250_f32_16x16x128_{a_dtype}_{b_dtype}"
                    "_scale_e8m0_e8m0_k32",
                )
                self.assertEqual(
                    spec.reads_per_step,
                    spec.a_scale_reads + spec.b_scale_reads + 4 * (rm + rn),
                )
                llvm = _llvm(spec)
                calls = re.findall(_WMMA + r"[^\n]*", llvm)
                self.assertEqual(len(calls), len(fp4))
                self.assertEqual(len(calls) % (spec.k_steps * rm * rn), 0)
                fa, fb = _FORMATS[a_dtype], _FORMATS[b_dtype]
                for call in calls:
                    self.assertIn(f"<8 x float> {wmma_16}(i32 {fa}, <16 x i32> %", call)
                    self.assertIn(f", i32 {fb}, <16 x i32> %", call)
                self.assertIn(f'"s_wait_dscnt {spec.reads_per_step}"', llvm)
                self.assertEqual(
                    len(re.findall(lds_read, llvm)), groups * spec.reads_per_step
                )

    def test_fp6_atom_reads_and_scale_swap(self):
        """FP6 issues the 16x16x128 f8f6f4 WMMA with format 2 on both
        operands, as many as FP8 of the same geometry. "chunked" reads each
        row as six 8 B LDS reads (waited for as three pairs), "contiguous" as
        three 16 B reads plus one byte swap per scale word; the scales are
        16 B reads either way."""
        wmma_16 = "@llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32"
        read_8, read_16 = (rf"load <{n} x i32>, ptr addrspace\(3\)" for n in (2, 4))
        geometry = {"tile_m": 128, "tile_n": 128, "tile_k": 256}
        fp8 = _spec8(**geometry)
        fp8_llvm = _llvm(fp8)
        groups, rest = divmod(len(re.findall(read_16, fp8_llvm)), fp8.reads_per_step)
        self.assertEqual(rest, 0)
        for k_reads, loads, width in (("chunked", 6, 2), ("contiguous", 3, 4)):
            spec = _spec6(k_reads=k_reads, **geometry)
            rm, rn = spec.wave_repeat_m, spec.wave_repeat_n
            with self.subTest(k_reads=k_reads):
                self.assertEqual(
                    spec.atom.op_id,
                    "wmma_gfx1250_f32_16x16x128_fp6_fp6_scale_e8m0_e8m0_k32",
                )
                self.assertEqual(spec.reads_per_step, 2 + 3 * (rm + rn))
                llvm = _llvm(spec)
                calls = re.findall(_WMMA + r"[^\n]*", llvm)
                self.assertEqual(len(calls), len(re.findall(_WMMA, fp8_llvm)))
                for call in calls:
                    self.assertIn(f"<8 x float> {wmma_16}(i32 2, <16 x i32> %", call)
                    self.assertIn(", i32 2, <16 x i32> %", call)
                self.assertIn(f'"s_wait_dscnt {spec.reads_per_step}"', llvm)
                data_reads = groups * loads * (rm + rn)
                if width == 2:
                    self.assertEqual(len(re.findall(read_8, llvm)), data_reads)
                    self.assertEqual(len(re.findall(read_16, llvm)), groups * 2)
                    self.assertNotIn("llvm.amdgcn.perm", llvm)
                else:
                    self.assertNotRegex(llvm, read_8)
                    self.assertEqual(
                        len(re.findall(read_16, llvm)), data_reads + groups * 2
                    )
                    swaps = re.findall(
                        r"call i32 @llvm\.amdgcn\.perm\([^\n]*i32 50397696\)", llvm
                    )
                    self.assertEqual(len(swaps), groups * (rm + rn))

    def test_scale_swaps_follow_the_k_step_wait(self):
        """With ``scale_swap`` "wait" each K-step's scale swaps come right
        after its first wait, behind a barrier VALU may not cross, in every
        compute path; by default no barrier comes before them."""
        barrier = "call void @llvm.amdgcn.sched.barrier(i32 2044)"
        swap = re.compile(r"call i32 @llvm\.amdgcn\.perm\([^\n]*i32 50397696\)")
        for path in (
            _DIRECT,
            {"half_plr": "none"},
            {"half_plr": "b"},
            {"half_plr": "b", "half_b_order": "late"},
        ):
            spec = _spec6(k_reads="contiguous", scale_swap="wait", **path)
            lines = _llvm(spec).splitlines()
            words = spec.wave_repeat_m + spec.wave_repeat_n
            swaps = [i for i, line in enumerate(lines) if swap.search(line)]
            barriers = [i for i, line in enumerate(lines) if barrier in line]
            wmmas = sum(1 for line in lines if re.search(_WMMA, line))
            with self.subTest(**path):
                self.assertEqual(
                    len(barriers), wmmas // (spec.wave_repeat_m * spec.wave_repeat_n)
                )
                self.assertEqual(len(swaps), len(barriers) * words)
                for at in barriers:
                    self.assertIn('asm sideeffect "s_wait_dscnt', lines[at - 1])
                    self.assertEqual(
                        [i for i in swaps if at < i <= at + words],
                        list(range(at + 1, at + 1 + words)),
                    )
            for other in (_spec6(k_reads="contiguous", **path), _spec8(**path)):
                with self.subTest(other.kernel_name()):
                    self.assertNotIn(barrier, _llvm(other))

    def test_l_order(self):
        """Each WMMA appears once, and part ``i`` (the WMMAs whose larger
        index is ``i``) follows parts ``0..i-1``: it needs read groups
        ``0..i`` only."""
        for rm, rn in itertools.product((1, 2, 4, 8), (1, 2, 4, 8)):
            order = TileCompute.l_order(rm, rn)
            with self.subTest(rm=rm, rn=rn):
                self.assertEqual(
                    sorted(order), sorted(itertools.product(range(rm), range(rn)))
                )
                parts = [max(pair) for pair in order]
                self.assertEqual(parts, sorted(parts))
        self.assertEqual(TileCompute.l_order(2, 2), [(0, 0), (1, 0), (1, 1), (0, 1)])

    def test_interleaved_reads_follow_the_wmma_chunks(self):
        """With ``read_issue`` "interleave" each K-step issues its WMMAs in
        ``read_groups`` chunks; read group ``g`` of the next K-step follows
        chunk ``min(g, groups - 2)``, and the K-step ends with a use of its
        operands (and, with the swaps after the wait, of the scale reads they
        took) after its last WMMA."""
        load = re.compile(r"= load (<\d+ x i32>|i32), ptr addrspace\(3\)")
        hold = re.compile(r'call void asm sideeffect "", "v[v,]*"')
        ahead = {"local_prefetch": 1, "half_plr": "none", "read_issue": "interleave"}
        for spec in (
            _spec6(
                tile_m=128,
                tile_n=128,
                tile_k=512,
                k_reads="contiguous",
                scale_swap="wait",
                **ahead,
            ),
            _spec6(
                1024,
                16,
                2048,
                tile_m=128,
                tile_n=16,
                wave_group=(4, 1),
                cluster_dim=(1, 2, 1),
                **ahead,
            ),
            _spec8(tile_m=64, tile_n=128, **ahead),
        ):
            tokens = []
            for line in _llvm(spec).splitlines():
                if re.search(_WMMA, line):
                    tokens.append("W")
                elif load.search(line):
                    tokens.append("R")
                elif hold.search(line):
                    tokens.append("H")
            # Tile 0's K-step 0 reads come before any WMMA.
            first = tokens.index("W")
            self.assertEqual(set(tokens[:first]), {"R"})
            order = TileCompute.l_order(spec.wave_repeat_m, spec.wave_repeat_n)
            groups = spec.read_groups
            size, extra = divmod(len(order), groups)
            step = []
            for c in range(groups):
                step += ["W"] * (size + (c < extra))
                for g in range(groups):
                    if min(g, max(groups - 2, 0)) == c:
                        # IR loads: every 8 B FP6 read is its own load.
                        rows = (g < spec.wave_repeat_m) + (g < spec.wave_repeat_n)
                        scales = spec.a_scale_reads + spec.b_scale_reads
                        step += ["R"] * (
                            scales * (g == 0) + rows * len(spec.lane_reads)
                        )
            step.append("H")
            with self.subTest(spec.kernel_name()):
                rest = tokens[first:]
                steps, extra_tokens = divmod(len(rest), len(step))
                self.assertEqual(extra_tokens, 0)
                self.assertEqual(rest, step * steps)

    def test_panel_walk_ir(self):
        """A panel walk's loop runs two K tiles of four passes of the 128x128
        compute (512 WMMAs); the prologue loads tile 0's four panels and each
        tile loads the next one's four (data and scales per wave); the TDM
        waits follow :meth:`PanelWalk.waits`."""
        tdm = r"call void @llvm\.amdgcn\.tensor\.load\.to\.lds\("
        tile_wait = r"call void @llvm\.amdgcn\.s\.wait\.tensorcnt\(i16 (\d+)\)"
        waits = PanelWalk.waits()
        expected = [waits["prologue"]] + [
            waits[key]
            for key in ((0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (0, 0))
            if waits[key] is not None
        ]
        for fields in (
            {"half_plr": "none"},
            {"half_plr": "b"},
            {
                "half_plr": "none",
                "k_reads": "contiguous",
                "scale_swap": "wait",
                "read_issue": "interleave",
            },
        ):
            spec = _spec6(512, 512, 2048, cluster_dim=(2, 2, 1), **_PANELS, **fields)
            with self.subTest(spec.kernel_name()):
                self.assertIsNone(validate(spec))
                llvm = _llvm(spec)
                self.assertEqual(len(re.findall(_WMMA, llvm)), 2 * 4 * 4 * 16)
                self.assertEqual(len(re.findall(tdm, llvm)), (4 + 2 * 4) * 2)
                waited = [int(w) for w in re.findall(tile_wait, llvm)]
                self.assertEqual(waited, expected + [0])

    def test_half_walk_ir(self):
        """A half walk's loop runs two K tiles of two halves of the 128x256
        compute (512 WMMAs); the prologue loads tile 0's three half-0 panels,
        its A1 and tile 1's B, and each tile four panels (data and scales
        per wave); the TDM waits follow :meth:`HalfWalk.waits`."""
        tdm = r"call void @llvm\.amdgcn\.tensor\.load\.to\.lds\("
        tile_wait = r"call void @llvm\.amdgcn\.s\.wait\.tensorcnt\(i16 (\d+)\)"
        waits = HalfWalk.waits()
        expected = [waits["prologue"]] + [waits[1], waits[0]] * 2 + [0]
        for fields in (
            {"half_plr": "none"},
            {"half_plr": "b"},
            {
                "half_plr": "none",
                "k_reads": "contiguous",
                "scale_swap": "wait",
                "read_issue": "interleave",
            },
        ):
            spec = _spec6(512, 512, 2048, cluster_dim=(2, 2, 1), **_HALVES, **fields)
            with self.subTest(spec.kernel_name()):
                self.assertIsNone(validate(spec))
                llvm = _llvm(spec)
                self.assertEqual(len(re.findall(_WMMA, llvm)), 2 * 2 * 4 * 32)
                self.assertEqual(len(re.findall(tdm, llvm)), (6 + 2 * 4) * 2)
                waited = [int(w) for w in re.findall(tile_wait, llvm)]
                self.assertEqual(waited, expected)

    def test_fixed_read_bases_ir(self):
        """Fixed read bases: one opaque register per buffer (pass slot) and A
        row, B fragment and scale group, made before the K loop (a
        v_perm_b32 through an identity selector from asm); the default
        makes none."""
        selector = re.compile(
            r'(%\w+) = call i32 asm sideeffect "s_mov_b32 \$0, 0x3020100"'
        )
        ahead = {"local_prefetch": 1, "half_plr": "none"}
        cases = (
            # (spec, buffers or slots of A and B, A rows, B fragments)
            (
                _spec6(
                    tile_m=128,
                    tile_n=128,
                    tile_k=512,
                    lds_buffers=3,
                    k_reads="contiguous",
                    scale_swap="wait",
                    read_issue="interleave",
                    **ahead,
                ),
                (3, 3),
                4,
                4,
            ),
            (
                _spec6(
                    512, 512, 2048, cluster_dim=(2, 2, 1), **_PANELS, half_plr="none"
                ),
                (2, 2),
                4,
                4,
            ),
            (
                _spec6(
                    512, 512, 2048, cluster_dim=(2, 2, 1), **_HALVES, half_plr="none"
                ),
                (2, 2),
                4,
                8,
            ),
            (
                _spec6(
                    1024,
                    16,
                    2048,
                    tile_m=128,
                    tile_n=16,
                    tile_k=512,
                    wave_group=(4, 1),
                    cluster_dim=(1, 2, 1),
                    **_DIRECT,
                ),
                (2, 2),
                2,
                1,
            ),
        )
        for spec, (a_units, b_units), rm, rn in cases:
            with self.subTest(spec.kernel_name()):
                self.assertIsNone(selector.search(_llvm(spec)))
                text = _llvm(replace(spec, read_bases="fixed"))
                (name,) = selector.findall(text)
                kept = re.findall(
                    rf"@llvm\.amdgcn\.perm\((i32 %\w+), \1, i32 {name}\)", text
                )
                self.assertEqual(len(kept), a_units * (rm + 1) + b_units * (rn + 1))

    def test_pair_map_ir(self):
        """The pair map stores each group of a lane's N-repeats as its own
        vector, reads the B scales per group (counted in the LDS waits) and
        keeps the WMMA count."""
        for fields, store, b_scale_load in (
            (_PAIRS, "<4 x bfloat>", "load <4 x i32>"),
            (
                {"tile_m": 128, "tile_n": 128, "tile_k": 512, "wave_map": "pairs"},
                "<2 x bfloat>",
                "load <2 x i32>",
            ),
        ):
            spec = _spec6(out_dtype="bf16", **fields)
            rows = replace(spec, wave_map="rows")
            with self.subTest(spec.kernel_name()):
                self.assertIsNone(validate(spec))
                self.assertEqual(spec.store_width, spec.n_group)
                self.assertEqual(spec.b_scale_reads, max(2, spec.wave_repeat_n // 4))
                llvm = _llvm(spec)
                self.assertIn(f"store {store}", llvm)
                self.assertNotIn(f"store <{2 * spec.n_group} x bfloat>", llvm)
                self.assertIn(b_scale_load, llvm)
                for count in spec.dscnt_waits:
                    self.assertIn(f'"s_wait_dscnt {min(count, 63)}"', llvm)
                self.assertEqual(
                    len(re.findall(_WMMA, llvm)), len(re.findall(_WMMA, _llvm(rows)))
                )

    def test_tiled_operands_descriptors_and_prefetch(self):
        """A tiled operand's data loads copy 256 B lines in the iterate blocks
        of its row-major layout; the other operand's loads and the scale loads
        are unchanged. The GL2 prefetch takes a tiled operand's whole lines:
        for 192 B FP6 rows at cluster 4x4, 768 lines per operand where the
        row-major rows take 1024, which with the scales' 64 fits one round
        of the cluster's 2048 threads instead of two."""

        def descriptors(spec):
            calls = []
            original = IRBuilder.tdm_descriptor_2d

            def record(builder, *args, **kwargs):
                calls.append(kwargs)
                return original(builder, *args, **kwargs)

            with mock.patch.object(IRBuilder, "tdm_descriptor_2d", record):
                build_mx_gemm(spec)
            return {
                (
                    c["tile_dim0"],
                    c["tile_dim1"],
                    c.get("iterate_count", 0),
                    c.get("lds_increment", 0),
                )
                for c in calls
            }

        base = _spec6(
            1024, 1024, 2048, tile_m=256, tile_n=128, tile_k=512, half_b_order="late"
        )
        # A: 128 rows of 384 B, 8 per block (16 B pad); B: 64 rows, 4 per block.
        row_major = {"a": (384, 8, 16, 3088), "b": (384, 4, 16, 1552)}
        tiled = {"a": (256, 12, 16, 3088), "b": (256, 6, 16, 1552)}
        loads = descriptors(base)
        self.assertLessEqual(set(row_major.values()), loads)
        scales = loads - set(row_major.values())
        for which in ("a", "b", "ab"):
            spec = replace(base, tiled_operands=which)
            expected = scales | {
                (tiled if spec.tiled(name) else row_major)[name] for name in ("a", "b")
            }
            with self.subTest(tiled=which):
                self.assertEqual(descriptors(spec), expected)

        def prefetches(spec):
            count = [0]
            original = IRBuilder.global_prefetch

            def record(builder, *args, **kwargs):
                count[0] += 1
                return original(builder, *args, **kwargs)

            with mock.patch.object(IRBuilder, "global_prefetch", record):
                build_mx_gemm(spec)
            return count[0]

        maf = _spec6(
            *_MAF_MED[0],
            tile_m=256,
            tile_n=256,
            lds_buffers=3,
            half_b_order="late",
            k_reads="contiguous",
            scale_swap="wait",
        )
        rounds = prefetches(maf)
        self.assertGreater(rounds, 0)
        self.assertEqual(prefetches(replace(maf, tiled_operands="ab")) * 2, rounds)

    def test_inst_prefetch_ir(self):
        """An instruction prefetch before every inst_prefetch-th WMMA of the
        kernel, counted over every tile and pass; none by default."""
        prefetch = "s_prefetch_inst_pc_rel 0, null, 31"
        for fields in (
            {"tile_k": 256, "local_prefetch": 1, "half_b_order": "late"},
            {
                **_PANELS,
                "half_plr": "none",
                "read_issue": "interleave",
                "k_reads": "contiguous",
                "scale_swap": "wait",
            },
            _DIRECT,
        ):
            base = _spec6(512, 512, 2048, cluster_dim=(1, 1, 1), **fields)
            self.assertNotIn(prefetch, _llvm(base))
            for every in (16, 64):
                spec = replace(base, inst_prefetch=every)
                with self.subTest(kernel=spec.kernel_name()):
                    llvm = _llvm(spec)
                    wmmas = len(re.findall(_WMMA, llvm))
                    self.assertEqual(llvm.count(prefetch), -(-wmmas // every))
                    self.assertLess(
                        llvm.index(prefetch), re.search(_WMMA, llvm).start()
                    )

    def test_three_buffers_issue_wait_and_peel(self):
        """With three buffers the prologue loads three tiles and every tile
        body one; a tile waits until at most the two tiles after it are in
        flight (with read-ahead, the next tile's read waits for one); the
        loop runs three tiles and peels the full tiles three does not
        divide, then the K tail."""
        tdm = r"call void @llvm\.amdgcn\.tensor\.load\.to\.lds\("
        tile_wait = r"call void @llvm\.amdgcn\.s\.wait\.tensorcnt\(i16 ([1-9])\)"
        geometry = {"tile_m": 128, "tile_n": 128, "tile_k": 512, "lds_buffers": 3}
        for k_dim, fields in itertools.product(
            (2048, 2560, 3072, 3072 + 256, 1024),
            (_DIRECT, {"local_prefetch": 1, "half_plr": "b"}),
        ):
            spec = _spec6(1024, 1024, k_dim, **geometry, **fields)
            with self.subTest(K=k_dim, local_prefetch=spec.local_prefetch):
                self.assertIsNone(validate(spec))
                llvm = _llvm(spec)
                full = k_dim // spec.tile_k
                bodies = full % 3 + 3
                per_tile = spec.k_steps * spec.wave_repeat_m * spec.wave_repeat_n
                tail = spec.tail_steps * spec.wave_repeat_m * spec.wave_repeat_n
                self.assertEqual(len(re.findall(_WMMA, llvm)), bodies * per_tile + tail)
                tiles = bodies + (1 if spec.tail_steps else 0)
                self.assertEqual(
                    len(re.findall(tdm, llvm)), spec.tdm_loads * (3 + tiles)
                )
                waits = {"4"} | ({"2"} if spec.local_prefetch else set())
                self.assertEqual(set(re.findall(tile_wait, llvm)), waits)

    def test_wmma_reuse_orders(self):
        """The K-step order of each reuse option, and the operand each WMMA
        keeps for the next one."""
        T, F = True, False
        expected = {
            "none": (
                [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
                [F] * 6,
                [F] * 6,
            ),
            "a": (
                [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
                [T, T, F, T, T, F],
                [F] * 6,
            ),
            "b": (
                [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)],
                [F] * 6,
                [T, F, T, F, T, F],
            ),
            "ab": (
                [(0, 0), (1, 0), (1, 1), (0, 1), (0, 2), (1, 2)],
                [F, T, F, T, F, F],
                [T, F, T, F, T, F],
            ),
        }
        for reuse, (order, keep_a, keep_b) in expected.items():
            with self.subTest(reuse):
                self.assertEqual(
                    TileCompute.wmma_order(reuse, 2, 3), (order, keep_a, keep_b)
                )

    def test_wmma_reuse_orders_and_flags_the_wmmas(self):
        """Each K-step's WMMAs follow :meth:`TileCompute.wmma_order`, and a
        WMMA flags A (B) only when the next WMMA takes the same A (B)
        fragment. Without reuse no WMMA flags an operand."""
        call = re.compile(
            _WMMA + r"[^(]*\(i32 \d+, <16 x i32> (%[\w.]+), i32 \d+, "
            r"<16 x i32> (%[\w.]+),[^\n]*i1 (true|false), i1 (true|false)\)"
        )
        direct = {"local_prefetch": 0, "half_plr": "none"}
        for base in (
            _spec6(k_reads="contiguous", **direct),
            _spec8(**direct),
            _spec6(scales="registers", tile_m=128, **direct),
        ):
            rm, rn = base.wave_repeat_m, base.wave_repeat_n
            for reuse in WMMA_REUSE:
                spec = replace(base, wmma_reuse=reuse)
                calls = call.findall(_llvm(spec))
                order, keep_a, keep_b = TileCompute.wmma_order(reuse, rm, rn)
                with self.subTest(spec.kernel_name()):
                    self.assertGreater(len(calls), 0)
                    self.assertEqual(len(calls) % (rm * rn), 0)
                    for at in range(0, len(calls), rm * rn):
                        step = calls[at : at + rm * rn]
                        self.assertEqual([c[2] == "true" for c in step], keep_a)
                        self.assertEqual([c[3] == "true" for c in step], keep_b)
                        for i, j in itertools.combinations(range(rm * rn), 2):
                            self.assertEqual(
                                step[i][0] == step[j][0], order[i][0] == order[j][0]
                            )
                            self.assertEqual(
                                step[i][1] == step[j][1], order[i][1] == order[j][1]
                            )

    def test_half_b_reuse_flags_each_b_fragment_but_its_last(self):
        """With read-ahead and B in halves, wmma_reuse "b" flags B on exactly
        the WMMAs whose next WMMA takes the same B fragment (every M-repeat
        of a B fragment but the last) and never A; without it no WMMA flags
        an operand."""
        call = re.compile(
            _WMMA + r"[^(]*\(i32 \d+, <16 x i32> (%[\w.]+), i32 \d+, "
            r"<16 x i32> (%[\w.]+),[^\n]*i1 (true|false), i1 (true|false)\)"
        )
        swait = {"k_reads": "contiguous", "scale_swap": "wait"}
        for base in (
            _spec6(tile_m=256, tile_n=128, tile_k=512, half_b_order="late", **swait),
            _spec6(lds_buffers=3, half_b_order="late", **swait),
            _spec6(half_b_order="ahead"),
            _spec8(half_b_order="late"),
            _spec6(
                256,
                512,
                2048,
                tile_m=64,
                tile_n=128,
                tile_k=512,
                lds_buffers=3,
                half_b_order="late",
                cluster_dim=(4, 2, 1),
                **swait,
            ),
        ):
            rm = base.wave_repeat_m
            for reuse in ("none", "b"):
                spec = replace(base, wmma_reuse=reuse)
                with self.subTest(spec.kernel_name()):
                    self.assertIsNone(validate(spec))
                    calls = call.findall(_llvm(spec))
                    self.assertGreater(len(calls), 0)
                    self.assertFalse(any(c[2] == "true" for c in calls))
                    flags = [c[3] == "true" for c in calls]
                    if reuse == "none":
                        self.assertFalse(any(flags))
                        continue
                    shared = [x[1] == y[1] for x, y in zip(calls, calls[1:])] + [False]
                    self.assertEqual(flags, shared)
                    self.assertEqual(sum(flags) * rm, len(calls) * (rm - 1))

    def test_register_scales_load_ahead_and_select_halves(self):
        """Register scales: no TDM scale loads and no LDS scale reads. Tile 0's
        words load before the loop and each tile loads the next tile's, one
        global load of up to 4 words per lane, operand and K-step (one per
        word when M or N is not a multiple of the words); each WMMA
        names the half holding its repeat, and two repeats share a word
        register exactly when they share the word. Contiguous FP6 swaps each
        word once per tile."""
        wmma = re.compile(
            _WMMA + r"[^\n]*i32 (\d), i32 \d, i32 (%[\w.]+), "
            r"i32 (\d), i32 \d, i32 (%[\w.]+), i1"
        )
        scale_load = r"= load (?:<(\d) x i32>|i32), ptr addrspace\(1\)"
        lds_read = r"load (?:<\d+ x )?i(?:32|64)>?, ptr addrspace\(3\)"
        tdm = r"call void @llvm\.amdgcn\.tensor\.load\.to\.lds\("
        tile_wait = r"call void @llvm\.amdgcn\.s\.wait\.tensorcnt\(i16 ([1-9])\)"
        swap = r"call i32 @llvm\.amdgcn\.perm\([^\n]*i32 50397696\)"
        direct = {**_DIRECT, "scales": "registers"}
        for spec in (
            _spec6(k_reads="contiguous", tile_m=128, tile_n=128, **direct),
            _spec6(**direct),
            _spec8(wmma_reuse="b", **direct),
            _spec6(64, 64, 2048, tile_m=64, tile_n=64, cluster_dim=(1, 1, 1), **direct),
            _spec6(
                1024,
                16,
                2048,
                tile_m=256,
                tile_n=16,
                wave_group=(4, 1),
                cluster_dim=(1, 2, 1),
                k_reads="contiguous",
                **direct,
            ),
            _spec6(1001, 1024, 2048, tile_m=128, cluster_dim=(1, 1, 1), **direct),
        ):
            rm, rn = spec.wave_repeat_m, spec.wave_repeat_n
            ha, hb = ScaleRegisters.half(rm), ScaleRegisters.half(rn)
            lds_spec = replace(spec, scales="lds")
            llvm, lds_llvm = _llvm(spec), _llvm(lds_spec)
            calls = wmma.findall(llvm)
            with self.subTest(spec.kernel_name()):
                self.assertEqual(spec.tail_steps, 0)
                bodies, rest = divmod(len(calls), spec.k_steps * rm * rn)
                self.assertEqual(rest, 0)
                self.assertEqual(
                    len(re.findall(tdm, llvm)) * 2, len(re.findall(tdm, lds_llvm))
                )
                # Each tile waits for its own loads only: one per wave here,
                # data and scales with LDS scales.
                self.assertEqual((spec.tdm_loads, lds_spec.tdm_loads), (1, 2))
                self.assertEqual(set(re.findall(tile_wait, llvm)), {"1"})
                self.assertEqual(set(re.findall(tile_wait, lds_llvm)), {"2"})
                scale_reads = lds_spec.a_scale_reads + lds_spec.b_scale_reads
                self.assertEqual(
                    len(re.findall(lds_read, llvm)),
                    len(re.findall(lds_read, lds_llvm))
                    - bodies * spec.k_steps * scale_reads,
                )
                self.assertEqual(re.findall(scale_load, lds_llvm), [])
                words = sorted(int(n or 1) for n in re.findall(scale_load, llvm))
                per_set = [
                    min(4, h - j) if extent % h == 0 else 1
                    for h, extent in ((ha, spec.M), (hb, spec.N))
                    for j in range(0, h, 4 if extent % h == 0 else 1)
                ] * spec.k_steps
                self.assertEqual(words, sorted(per_set * (bodies + 1)))
                self.assertEqual(
                    len(re.findall(swap, llvm)),
                    bodies * spec.k_steps * (ha + hb) if spec.contiguous_k else 0,
                )
                order = TileCompute.wmma_order(spec.wmma_reuse, rm, rn)[0]
                for at in range(0, len(calls), rm * rn):
                    step = calls[at : at + rm * rn]
                    for (i, j), (sel_a, word_a, sel_b, word_b) in zip(order, step):
                        self.assertEqual(int(sel_a), ScaleRegisters.place(rm, i)[1])
                        self.assertEqual(int(sel_b), ScaleRegisters.place(rn, j)[1])
                    for x, y in itertools.combinations(range(rm * rn), 2):
                        (ix, jx), (iy, jy) = order[x], order[y]
                        self.assertEqual(
                            step[x][1] == step[y][1],
                            ScaleRegisters.place(rm, ix)[0]
                            == ScaleRegisters.place(rm, iy)[0],
                        )
                        self.assertEqual(
                            step[x][3] == step[y][3],
                            ScaleRegisters.place(rn, jx)[0]
                            == ScaleRegisters.place(rn, jy)[0],
                        )

    def test_register_scale_loads_take_sgpr_bases(self):
        """Register scales: a tile's K-step bases (A then B per K-step) come
        from one asm with an SGPR output each, and each load adds to one of
        them the lane's offset, copied once per tile through v_perm_b32 with
        the identity selector."""
        bases = re.compile(r'(%[\w.]+) = call \{[^}\n]*\} asm "", "((?:=s,)*=s),')
        selector = re.compile(
            r'(%\w+) = call i32 asm sideeffect "s_mov_b32 \$0, 0x3020100"'
        )
        direct = {**_DIRECT, "scales": "registers"}
        for spec in (
            _spec6(k_reads="contiguous", tile_m=128, tile_n=128, **direct),
            _spec8(wmma_reuse="b", **direct),
            _spec6(
                1024,
                16,
                2048,
                tile_m=256,
                tile_n=16,
                wave_group=(4, 1),
                cluster_dim=(1, 2, 1),
                k_reads="contiguous",
                **direct,
            ),
            _spec6(1001, 1024, 2048, tile_m=128, cluster_dim=(1, 1, 1), **direct),
        ):
            llvm = _llvm(spec)
            rm, rn = spec.wave_repeat_m, spec.wave_repeat_n
            ha, hb = ScaleRegisters.half(rm), ScaleRegisters.half(rn)
            with self.subTest(spec.kernel_name()):
                bodies = len(re.findall(_WMMA, llvm)) // (spec.k_steps * rm * rn)
                asms = dict(bases.findall(llvm))
                outputs = [constraints.count("=s") for constraints in asms.values()]
                self.assertEqual(outputs, [2 * spec.k_steps] * (bodies + 1))
                (identity,) = selector.findall(llvm)
                copies = re.findall(
                    rf"@llvm\.amdgcn\.perm\(i32 %[\w.]+, i32 %[\w.]+, i32 {identity}\)",
                    llvm,
                )
                offsets = (1 if spec.M % ha == 0 else ha) + (
                    1 if spec.N % hb == 0 else hb
                )
                self.assertEqual(len(copies), offsets * (bodies + 1))
                base_values = {
                    value
                    for value, asm in re.findall(
                        r"(%[\w.]+) = extractvalue [^\n]* (%[\w.]+), \d+$",
                        llvm,
                        re.M,
                    )
                    if asm in asms
                }
                lane_ptrs = re.findall(
                    r"= getelementptr inbounds i8, ptr addrspace\(1\) (%[\w.]+), i64",
                    llvm,
                )
                self.assertEqual(
                    sum(ptr in base_values for ptr in lane_ptrs),
                    offsets * spec.k_steps * (bodies + 1),
                )

    def test_fp6_k_tail_and_edge_tiles_build(self):
        """FP6 at tile_k 384 always has a K tail at these K: its GL2 line
        prefetch is clamped to the row's end (smin); without a tail or edge
        tile there is no clamp. Partial M and N tiles build."""
        whole = _llvm(_spec6(1024, 1024, 4096, cluster_dim=(1, 1, 1)))
        self.assertNotIn("@llvm.smin.i32(", whole)
        for k_dim in (2048, 32768, 65536):
            spec = _spec6(1024, 1024, k_dim, tile_k=384, cluster_dim=(1, 1, 1))
            with self.subTest(K=k_dim):
                self.assertEqual(spec.tail_steps, (k_dim % 384) // 128)
                llvm = _llvm(spec)
                self.assertIn("@llvm.smin.i32(", llvm)
                # The peel and the two-tile loop body, then the tail tile.
                tiles = k_dim // 384 % 2 + 2
                per_step = spec.wave_repeat_m * spec.wave_repeat_n
                self.assertEqual(
                    len(re.findall(_WMMA, llvm)),
                    (tiles * spec.k_steps + spec.tail_steps) * per_step,
                )
        for problem in ((1000, 1024, 2048), (1024, 1000, 2048), (300, 16, 2048)):
            fields = {"cluster_dim": (1, 1, 1)}
            if problem[1] == 16:
                fields.update(tile_n=16, wave_group=(4, 1), half_plr="none")
            spec = _spec6(*problem, **fields)
            with self.subTest(problem=problem):
                self.assertIsNone(validate(spec))
                self.assertIn("f8f6f4", _llvm(spec))

    def test_partial_n_tile_walks_only_the_rows_left(self):
        """B data is in TDM iterate mode (8 rows a walk). With a partial N
        tile its walk count comes from the rows left; otherwise it is the
        static 16 (``15 << 16`` in group 2)."""
        static_count = f"readfirstlane.i32(i32 {15 << 16})"
        for (m, n), runtime in (
            ((1024, 1040), True),
            ((1040, 1024), False),
            ((1024, 1024), False),
        ):
            spec = _spec(m, n, 2048, cluster_dim=(1, 1, 1))
            self.assertEqual(spec.lds_layout.b.iterate_rows, 8)
            with self.subTest(M=m, N=n):
                llvm = _llvm(spec)
                self.assertEqual(static_count in llvm, not runtime)
                self.assertEqual("@llvm.smin.i32(" in llvm, runtime)

    def test_peeled_and_looped_tiles(self):
        """The odd-count peel and the two-tile loop body are emitted once
        each; the loop covers the rest."""
        for tiles in (4, 5):
            spec = _spec(1024, 1024, tiles * 512)
            emitted = tiles % 2 + 2
            with self.subTest(tiles=tiles):
                llvm = _llvm(spec)
                self.assertEqual(len(re.findall(_WMMA, llvm)), emitted * 4 * 32)
                # The first K-step of an odd-count peel reads the zero
                # accumulators directly.
                zero_c = len(re.findall(_WMMA + r"[^\n]*<16 x float> %cz", llvm))
                self.assertEqual(zero_c, 32 if tiles % 2 else 0)

    def test_k_tail_tile(self):
        """K not a multiple of tile_k: the full tiles as before, then one
        tile of the remaining K-steps (loaded clipped at K)."""
        for tail_steps in (1, 2, 3):
            spec = _spec(1024, 1024, 4 * 512 + tail_steps * 128)
            self.assertEqual(spec.tail_steps, tail_steps)
            emitted = 2
            with self.subTest(tail_steps=tail_steps):
                llvm = _llvm(spec)
                self.assertEqual(
                    len(re.findall(_WMMA, llvm)), (emitted * 4 + tail_steps) * 32
                )
                # Every tile, the tail included, signals the cluster once.
                self.assertEqual(llvm.count("s_cmp_eq_u32 $0, 0"), emitted + 1)

    def test_k_partial_step(self):
        """K a multiple of 32 but not 128: the tail rounds up to a whole
        K-step, read clipped at K, and the scale loads count ceil(K/128)."""
        for extra in (32, 64, 96, 128 + 32, 384 + 96):
            spec = _spec(1024, 1024, 4 * 512 + extra)
            with self.subTest(K=spec.K):
                self.assertIsNone(validate(spec))
                self.assertEqual(spec.tail_steps, -(-extra // 128))
                llvm = _llvm(spec)
                emitted = 2
                self.assertEqual(
                    len(re.findall(_WMMA, llvm)),
                    (emitted * 4 + spec.tail_steps) * 32,
                )
                self.assertRegex(llvm, r"add nsw i32 %K, 127|add i32 %K, 127")
        self.assertNotIn("%K, 127", _llvm(_spec(1024, 1024, 4 * 512 + 128)))

    def test_k_not_a_scale_block_multiple(self):
        """K % 32 != 0 is rejected before any other K or size check."""
        for M, N, K in (
            (1024, 1024, 2048 + 16),
            (160, 16840, 16840),
            (2**20, 2**20, 2**20 + 1),
        ):
            with self.subTest(K=K):
                reason = validate(_spec(M, N, K))
                self.assertEqual(
                    reason,
                    f"K must be a multiple of 32: each E8M0 scale covers 32 K values (got K={K})",
                )
                with self.assertRaisesRegex(ValueError, "multiple of 32"):
                    build_mx_gemm(_spec(M, N, K))

    def test_fp6_k_is_a_multiple_of_16(self):
        """FP6 takes K a multiple of 16 (rows of whole 16-value packs, the
        last E8M0 scale covering what is left of K) and K shorter than
        tile_k. The other types keep K a multiple of 32 and a full tile; the
        tiled operands and the ring keep their K rules."""
        small = {"tile_m": 64, "tile_n": 64, "cluster_dim": (1, 1, 1)}
        for K, tile_k in itertools.product(
            (16, 32, 80, 112, 144, 240, 528, 1840), (128, 256, 512)
        ):
            spec = _spec6(256, 2048, K, tile_k=tile_k, **small)
            with self.subTest(K=K, tile_k=tile_k):
                self.assertIsNone(validate(spec))
                self.assertEqual(spec.tail_steps, -(-(K % tile_k) // 128))
        for K in (8, 24, 92, 104, 2048 + 8):
            with self.subTest(K=K):
                self.assertEqual(
                    validate(_spec6(256, 2048, K, tile_k=128, **small)),
                    "FP6 K must be a multiple of 16: A and B rows are whole "
                    f"16-value packs (got K={K})",
                )
        self.assertRegex(
            validate(_spec8(256, 2048, 80, tile_k=128, **small)), "multiple of 32"
        )
        self.assertEqual(
            validate(_spec8(256, 2048, 96, tile_k=128, **small)),
            "K must hold at least one full tile",
        )
        self.assertRegex(
            validate(_spec6(256, 2048, 80, tile_k=128, tiled_operands="ab", **small)),
            "multiple of tile_k",
        )
        self.assertIsNotNone(validate(_spec6(512, 512, 2048 + 16, **_RING)))

    def test_fp6_short_k_is_one_partial_tile(self):
        """FP6 K shorter than tile_k has no full tile: the peel and the loop
        run zero times (the loop body is still emitted) and the K tail is the
        only tile read. Loads past K reload tile 0, so no offset is K -
        tile_k, which would be negative; with a full tile they reload the
        last whole tile as before."""
        small = {"tile_m": 64, "tile_n": 64, "cluster_dim": (1, 1, 1)}
        for K, tile_k, buffers in (
            (32, 128, 2),
            (80, 256, 2),
            (112, 512, 3),
            (240, 256, 3),
        ):
            spec = _spec6(256, 2048, K, tile_k=tile_k, lds_buffers=buffers, **small)
            with self.subTest(K=K, tile_k=tile_k, buffers=buffers):
                llvm = _llvm(spec)
                steps = buffers * spec.k_steps + spec.tail_steps
                per_step = spec.wave_repeat_m * spec.wave_repeat_n
                self.assertEqual(len(re.findall(_WMMA, llvm)), steps * per_step)
                self.assertNotRegex(llvm, rf"= sub (nsw )?i32 %K, {tile_k}\b")
        full = _llvm(_spec6(256, 2048, 512 + 32, tile_k=256, **small))
        self.assertRegex(full, r"= sub (nsw )?i32 %K, 256\b")

    def test_short_k(self):
        """One or two full tiles, with or without a tail. The loop body is
        emitted (it runs zero times) after the odd-count peel."""
        for k_dim, tile_k in itertools.product(
            (128, 256, 384, 512, 640), (128, 256, 512)
        ):
            spec = _spec(1024, 1024, k_dim, tile_m=128, tile_n=128, tile_k=tile_k)
            if validate(spec) is not None:
                self.assertLess(k_dim, tile_k)
                continue
            full = k_dim // tile_k
            if full > 2:
                continue
            with self.subTest(K=k_dim, tile_k=tile_k):
                llvm = _llvm(spec)
                per_step = spec.wave_repeat_m * spec.wave_repeat_n
                tiles = full % 2 + 2
                steps = tiles * spec.k_steps + spec.tail_steps
                self.assertEqual(len(re.findall(_WMMA, llvm)), steps * per_step)

    def test_one_step_tiles_signal_the_cluster(self):
        """tile_k 128 has no second-to-last K-step: each tile still signals
        once, before its cluster wait."""
        for fields in (
            {},
            {"half_plr": "none"},
            {"half_plr": "none", "local_prefetch": 0},
        ):
            spec = _spec(tile_k=128, **fields)
            with self.subTest(**fields):
                llvm = _llvm(spec)
                self.assertEqual(len(re.findall(_WMMA, llvm)), 2 * 32)
                self.assertEqual(llvm.count("s_cmp_eq_u32 $0, 0"), 2)
                self.assertEqual(llvm.count("s_barrier_wait -3"), 2 + 2)

    def test_direct_reads_refill_at_every_signal_point(self):
        """local_prefetch 0 issues each tile's refill TDM and cluster wait
        for every signal point up to the K-step's WMMA count, as read-ahead
        does."""
        tdm = r"call [^\n]*tensor\.load"
        for tile_m, mma in ((256, "16x16"), (128, "32x16")):
            fields = {
                "tile_m": tile_m,
                "tile_n": 16,
                "wave_group": (4, 1),
                "cluster_dim": (1, 4, 1),
                "half_plr": "none",
                "mma": mma,
            }
            spec = _spec(1024, 16, 4096, **fields)
            llvm = _llvm(spec)
            expected = (len(re.findall(tdm, llvm)), llvm.count("s_barrier_wait -3"))
            for at in range(spec.signal_limit + 1):
                direct = _spec(
                    1024, 16, 4096, local_prefetch=0, cluster_signal_wmmas=at, **fields
                )
                with self.subTest(mma=mma, signal_wmmas=at):
                    self.assertIsNone(validate(direct))
                    llvm = _llvm(direct)
                    self.assertEqual(
                        (
                            len(re.findall(tdm, llvm)),
                            llvm.count("s_barrier_wait -3"),
                        ),
                        expected,
                    )

    def test_schedule_variants_emit_their_waits_and_barriers(self):
        """Two tiles are emitted at the smoke shape (the loop body; see the
        peel test), four K-steps each; every tile adds one wait before its
        refill TDM."""
        default = _llvm(_spec())
        self.assertEqual(default.count('"s_wait_dscnt 35"'), 2 * (4 * 2 + 1))
        self.assertEqual(default.count("s_cmp_eq_u32 $0, 0"), 2)
        none = _llvm(_spec(half_plr="none"))
        self.assertEqual(none.count('"s_wait_dscnt 35"'), 2 * (4 + 1))
        direct = _llvm(_spec(half_plr="none", local_prefetch=0))
        self.assertEqual(direct.count('"s_wait_dscnt 0"'), 2 * (4 + 1))
        # The whole cluster barrier opens and closes the kernel; the wave-0
        # signal adds no workgroup barrier of its own.
        self.assertEqual(default.count("s_barrier_signal_isfirst"), 2)
        no_cluster = _llvm(_spec(cluster_dim=(1, 1, 1)))
        self.assertNotIn("s_barrier_signal -3", no_cluster)
        self.assertEqual(no_cluster.count('"s_wait_dscnt 35"'), 2 * (4 * 2 + 1))
        # half_b_order "late": per K-step a wait leaving B's second half (8
        # reads) and one leaving the next scales and A (19); the late wait
        # also leaves the next tile's scales and A.
        late = _llvm(_spec(half_b_order="late"))
        self.assertNotIn('"s_wait_dscnt 35"', late)
        self.assertEqual(late.count('"s_wait_dscnt 8"'), 2 * 4)
        self.assertEqual(late.count('"s_wait_dscnt 19"'), 2 * (4 + 1))
        self.assertEqual(len(re.findall(_WMMA, late)), len(re.findall(_WMMA, default)))
        lds_read = r"load <\d+ x i32>, ptr addrspace\(3\)"
        self.assertEqual(
            len(re.findall(lds_read, late)), len(re.findall(lds_read, default))
        )
        no_prefetch = _llvm(_spec(gl2_prefetch_distance=0))
        self.assertNotIn("global.prefetch", no_prefetch)
        # Four prefetches (tiles 2 and 3 up front, one per emitted tile) over
        # 2048 lanes of the 4x4 cluster: 256 B rows give 2176 lines (two
        # rounds) per tile. 128 B rows give 2112 lines for an even tile and
        # only the 64 scale lines (one round) for an odd one.
        prefetch = "call void @llvm.amdgcn.global.prefetch"
        self.assertEqual(default.count(prefetch), 4 * 2)
        self.assertEqual(_llvm(_spec(tile_k=256)).count(prefetch), 2 * (2 + 1))

    def test_two_n_repeats(self):
        """wave_repeat_n 2 (1x4 waves, tile_n 128): one 8 B B-scale read per
        K-step, counted in the LDS waits, and 2-wide stores."""
        for tile_m, out_dtype in itertools.product((128, 256), OUT_DTYPES):
            spec = _spec(
                tile_m=tile_m, tile_n=128, wave_group=(1, 4), out_dtype=out_dtype
            )
            with self.subTest(tile_m=tile_m, out_dtype=out_dtype):
                self.assertIsNone(validate(spec))
                self.assertEqual(spec.wave_repeat_n, 2)
                self.assertEqual(spec.b_scale_reads, 1)
                llvm = _llvm(spec)
                self.assertIn("load <2 x i32>", llvm)
                self.assertIn(f'"s_wait_dscnt {spec.reads_per_step}"', llvm)
                store = {
                    "fp8e4m3": "<2 x i8>",
                    "fp16": "<2 x half>",
                    "bf16": "<2 x bfloat>",
                }
                self.assertIn(f"store {store[out_dtype]}", llvm)

    def test_one_column_stores_keep_the_fp32_product(self):
        """One D column per store (tile_n 64, 1x4 waves): each FP32 result
        passes an optimization barrier before its FP16 or BF16 conversion,
        so alpha * acc is not converted by one fused multiply-convert;
        whole and partial M tiles. Two-column stores convert in pairs
        without it."""
        barrier = 'asm "", "=v,0"(float'
        for m, out_dtype in itertools.product((64, 40), ("fp16", "bf16")):
            lone = _spec6(
                m,
                256,
                384,
                tile_m=16,
                tile_n=64,
                tile_k=384,
                wave_group=(1, 4),
                cluster_dim=(1, 1, 1),
                out_dtype=out_dtype,
                **_DIRECT,
            )
            pair = replace(lone, tile_n=128)
            with self.subTest(m=m, out_dtype=out_dtype):
                self.assertIsNone(validate(lone))
                self.assertEqual(lone.store_width, 1)
                self.assertEqual(pair.store_width, 2)
                llvm = _llvm(lone)
                converts = len(re.findall(r"= fptrunc float ", llvm))
                self.assertGreater(converts, 0)
                self.assertEqual(llvm.count(barrier), converts)
                self.assertNotIn(barrier, _llvm(pair))

    def test_output_types(self):
        for out_dtype, store in (("fp16", "<8 x half>"), ("bf16", "<8 x bfloat>")):
            with self.subTest(out_dtype):
                llvm = _llvm(_spec(out_dtype=out_dtype))
                self.assertIn(f"store {store}", llvm)
                self.assertNotIn("cvt.pk.fp8", llvm)
                self.assertNotIn("cvt.scalef32.pk8.fp8", llvm)
        self.assertIn("@llvm.amdgcn.cvt.scalef32.pk8.fp8.f32", _llvm(_spec()))
        self.assertIn(
            "@llvm.amdgcn.cvt.pk.fp8.f32", _llvm(_spec(tile_m=128, tile_n=128))
        )

    def test_ring_kernel(self):
        """walk "ring": a loop pass is a period, 16 K-steps of 64 WMMAs; a
        wave issues four TDM loads before the loop (lines 0 and 1, scale
        tiles 0 and 1) and eight in it (six lines, two scale tiles), waits
        for them with :meth:`RingWalk.waits`' counts in K-step order, and
        with a cluster signals and waits for it once per refill K-step. The
        LDS reads are the scales and rows of the first K-step and of the 16
        of a pass."""
        load = re.compile(r"= load (<\d+ x i32>|i32), ptr addrspace\(3\)")
        signal = '"s_cmp_eq_u32 $0, 0\\0As_cbranch_scc0 1\\0As_barrier_signal -3"'
        for (m, n, k), fields in (
            (_MAF_MED[0], {"scale_swap": "wait"}),
            (
                _MAF_MED[1],
                {"half_b_order": "ahead", "cluster_dim": (2, 4, 1)},
            ),
            ((512, 512, 4096), {"cluster_dim": (1, 1, 1), "wmma_reuse": "b"}),
        ):
            spec = _spec6(m, n, k, **_RING | fields)
            with self.subTest(spec.kernel_name()):
                self.assertIsNone(validate(spec))
                llvm = _llvm(spec)
                self.assertEqual(len(re.findall(_WMMA, llvm)), 16 * 64)
                self.assertEqual(
                    len(
                        re.findall(
                            r"call void @llvm\.amdgcn\.tensor\.load\.to\.lds", llvm
                        )
                    ),
                    12,
                )
                waits = [
                    int(w) for w in re.findall(r"s\.wait\.tensorcnt\(i16 (\d+)\)", llvm)
                ]
                counts = RingWalk.waits()
                loop = [
                    counts[q] for q in range(RingWalk.PERIOD) if counts[q] is not None
                ]
                self.assertEqual(waits, [counts["prologue"], *loop, 0])
                per_step = (
                    spec.a_scale_reads
                    + spec.b_scale_reads
                    + len(spec.lane_reads) * (spec.wave_repeat_m + spec.wave_repeat_n)
                )
                self.assertEqual(len(load.findall(llvm)), 17 * per_step)
                refills = sum(1 for q in range(RingWalk.PERIOD) if RingWalk.refills(q))
                cluster = spec.has_cluster
                self.assertEqual(
                    llvm.count('asm sideeffect "s_barrier_wait -3"'), refills * cluster
                )
                self.assertEqual(llvm.count(signal), refills * cluster)


class MxGemmBenchReferenceTest(unittest.TestCase):
    def test_block_reference_is_the_full_reference_sliced(self):
        """``--verify block`` checks the first and last tiles plus random
        rows and columns, against the same values the full reference has."""
        for out_dtype, beta in itertools.product(OUT_DTYPES, (0.0, 0.5)):
            spec = _spec(384, 640, 512, tile_m=128, tile_n=128, out_dtype=out_dtype)
            inputs = check_inputs(spec)
            rows, cols = verify_block(spec, 37)
            with self.subTest(out_dtype=out_dtype, beta=beta):
                self.assertTrue(np.array_equal(rows[:128], np.arange(128)))
                self.assertTrue(np.array_equal(rows[-128:], np.arange(256, 384)))
                self.assertTrue(np.array_equal(cols[-128:], np.arange(512, 640)))
                self.assertGreater(len(cols), 256)
                alpha = np.float32(1.0 / 512)
                full = reference(spec, inputs, alpha=alpha, beta=beta)
                block = reference(
                    spec, inputs, alpha=alpha, beta=beta, rows=rows, cols=cols
                )
                self.assertEqual(full[np.ix_(rows, cols)].tobytes(), block.tobytes())

    def test_scales_pad_to_whole_k_steps(self):
        """K/32 not a multiple of 4: each MN's last step is zero-padded."""
        spec = _spec(256, 384, 512 + 96, tile_m=128, tile_n=128)
        inputs = check_inputs(spec)
        _, _, a_scale, b_scale, _ = device_inputs(inputs)
        for swizzled, canonical in ((a_scale, inputs[2]), (b_scale, inputs[3].T)):
            self.assertEqual(swizzled.shape, (5, canonical.shape[0], 4))
            self.assertTrue((swizzled[4, :, 3] == 0).all())
            self.assertTrue(np.array_equal(swizzled[4, :, :3], canonical[:, 16:19]))
            self.assertTrue(np.array_equal(swizzled[0], canonical[:, :4]))

    def test_fp6_partial_scale_group(self):
        """FP6 K % 32 != 0: ceil(K / 32) scale groups, the last covering the
        K % 32 values left. The reference equals that of K padded to a
        multiple of 32 with zero codes."""
        for K, data in itertools.product((16, 80, 112, 144), ("exact", "sparse")):
            spec = _spec6(
                64, 48, K, tile_m=64, tile_n=16, wave_group=(4, 1), half_plr="none"
            )
            inputs = check_inputs(spec) if data == "exact" else sparse_inputs(spec)
            groups = -(-K // 32)
            padded_k = groups * 32
            padded = replace(spec, K=padded_k)
            codes = [np.pad(c, ((0, 0), (0, padded_k - K))) for c in inputs[5:]]
            padded_inputs = (*inputs[:5], *codes)
            with self.subTest(K=K, data=data):
                self.assertEqual(inputs[2].shape, (64, groups))
                self.assertEqual(inputs[3].shape, (groups, 48))
                self.assertEqual(inputs[0].shape, (64, K // 4 * 3))
                for beta in (0.0, 0.5):
                    self.assertEqual(
                        reference(spec, inputs, alpha=1.0, beta=beta).tobytes(),
                        reference(
                            padded, padded_inputs, alpha=1.0, beta=beta
                        ).tobytes(),
                    )
                _, _, a_scale, b_scale, _ = device_inputs(inputs)
                self.assertEqual(a_scale.shape, (-(-K // 128), 64, 4))
                self.assertEqual(b_scale.shape, (-(-K // 128), 48, 4))


class _FakeEvent:
    def __init__(self, gpu):
        self.gpu, self.at, self.destroyed = gpu, None, False

    def record(self):
        self.at = self.gpu.clock

    def synchronize(self):
        pass

    def elapsed_to(self, end):
        return float(end.at - self.at)

    def destroy(self):
        self.destroyed = True


class _FakeGpu:
    """``_time``'s launcher and its runtime in one: a GPU clock in ms that
    ``launch`` advances by 2 and a prepared enqueue by 3, so each time shows
    which path it measured. ``fail_on``: the enqueue call that raises."""

    def __init__(self, fail_on=None):
        self.runtime = self
        self.clock = 0
        self.events = []
        self.prepared_with = []
        self.enqueues = 0
        self.fail_on = fail_on

    def event(self):
        self.events.append(_FakeEvent(self))
        return self.events[-1]

    def sync(self):
        pass

    def launch(self, arguments):
        self.clock += 2

    def prepare_launch(self, arguments):
        self.prepared_with.append(arguments)

        def enqueue():
            self.enqueues += 1
            if self.enqueues == self.fail_on:
                raise RuntimeError("enqueue failed")
            self.clock += 3

        return enqueue


class MxGemmBenchTimingTest(unittest.TestCase):
    """The bench's timing and ``--out`` without a device."""

    def test_event_batch_times_prepared_enqueues(self):
        """event_batch_us brackets ``--batch`` prepared enqueues, built once,
        with events and divides by the batch; median_us times ``launch``."""
        gpu = _FakeGpu()
        times = _time(gpu, b"args", warmup=5, iters=4, batch=10)
        self.assertEqual(gpu.prepared_with, [b"args"])
        self.assertEqual((times.event_batch, times.median), (3000.0, 2000.0))
        self.assertIsInstance(times.batched, float)
        self.assertIsInstance(times.enqueue, float)
        self.assertTrue(all(event.destroyed for event in gpu.events))

    def test_zero_counts_time_nothing(self):
        gpu = _FakeGpu()
        self.assertEqual(
            _time(gpu, b"args", warmup=1, iters=0, batch=0), (None, None, None, None)
        )
        self.assertEqual((gpu.prepared_with, gpu.events), ([], []))

    def test_failed_enqueue_destroys_the_batch_events(self):
        gpu = _FakeGpu(fail_on=2)
        with self.assertRaisesRegex(RuntimeError, "enqueue failed"):
            _time(gpu, b"args", warmup=1, iters=1, batch=10)
        self.assertEqual(len(gpu.events), 2)
        self.assertTrue(all(event.destroyed for event in gpu.events))

    def test_hold_keeps_the_prepared_launch_running(self):
        """``--hold-seconds`` enqueues the prepared launch, built once, in
        groups of about 20 ms of device time until the window has passed;
        no hold enqueues nothing."""
        gpu = _FakeGpu()
        hold = _hold(gpu, b"args", 0.05, 1000.0)
        self.assertEqual(gpu.prepared_with, [b"args"])
        self.assertGreaterEqual(hold.end - hold.start, 0.05)
        self.assertGreater(hold.launches, 0)
        self.assertEqual(hold.launches % 20, 0)
        self.assertEqual(gpu.enqueues, hold.launches)
        idle = _FakeGpu()
        self.assertEqual(_hold(idle, b"args", 0.0, 1000.0), (None, None, 0))
        self.assertEqual(idle.prepared_with, [])

    def test_failed_hold_enqueue_propagates(self):
        gpu = _FakeGpu(fail_on=3)
        with self.assertRaisesRegex(RuntimeError, "enqueue failed"):
            _hold(gpu, b"args", 1.0, None)

    def test_out_holds_every_row_without_base(self):
        """``--out`` is written row by row; without ``--base`` nothing
        rewrites it at the end."""
        import contextlib
        import csv
        import io
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "rows" / "bench.csv"
            argv = ["--host-only", "--tile", "48x256x512,256x256x64", "--out", str(out)]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(argv), 0)
            with out.open(newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
        self.assertEqual(reader.fieldnames, FIELDS)
        self.assertEqual(
            [row["config"] for row in rows], ["tile=48x256x512", "tile=256x256x64"]
        )
        for row in rows:
            self.assertTrue(row["check"].startswith("INVALID:"), row["check"])


class MxGemmBenchCodeDirTest(unittest.TestCase):
    def test_code_dir_stores_loads_and_refuses_another_comgr(self):
        """``--host-only --code-dir`` compiles and stores each code object; a
        second run loads them all, byte for byte; a directory stamped by
        another COMGR library is refused."""
        import contextlib
        import csv
        import io
        import tempfile
        from pathlib import Path

        argv = ["--host-only", "--mma", "16x16", "--a-dtype", "fp6", "--b-dtype", "fp6"]
        argv += [
            "--tile",
            "128x128x256",
            "--local-prefetch",
            "0,1",
            "--half-plr",
            "none",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            code = Path(tmp) / "code"
            runs = []
            for attempt in range(2):
                out = Path(tmp) / f"bench{attempt}.csv"
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(
                        main([*argv, "--code-dir", str(code), "--out", str(out)]), 0
                    )
                with out.open(newline="") as handle:
                    runs.append(list(csv.DictReader(handle)))
            stored = sorted(code.glob("*.hsaco"))
            snapshot = {path.name: path.read_bytes() for path in stored}
            self.assertEqual([row["code"] for row in runs[0]], ["compiled"] * 2)
            self.assertEqual([row["code"] for row in runs[1]], ["stored"] * 2)
            self.assertEqual(len(stored), 2)
            self.assertEqual(
                [row["vgpr"] for row in runs[0]], [row["vgpr"] for row in runs[1]]
            )
            self.assertEqual({p.name: p.read_bytes() for p in stored}, snapshot)
            (code / "COMGR_MD5").write_text("0" * 32 + "\n")
            with (
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(SystemExit, "another COMGR library"),
            ):
                main([*argv, "--code-dir", str(code)])


class MxGemmSparseCheckTest(unittest.TestCase):
    """All E2M1 values in a few K-steps: exact FP32 sums at any K."""

    def _values(self, spec, data):
        """Float64 A and B with their scales applied."""
        _, _, a_scale, b_scale, _, a_codes, b_codes = data
        a_values = decode_table(spec.a_dtype).astype(np.float64)
        b_values = decode_table(spec.b_dtype).astype(np.float64)
        x = a_values[a_codes] * np.repeat(decode_e8m0(a_scale), spec.block_k, axis=1)
        y = b_values[b_codes] * np.repeat(decode_e8m0(b_scale).T, spec.block_k, axis=1)
        return x, y

    @staticmethod
    def _specs(**fields):
        """The FP4 spec of these tests, its 8-bit and its FP6 counterparts."""
        problem = (64, 48, 2048 + 96)
        yield _spec(*problem, tile_m=128, tile_n=128, **fields)
        for a_dtype, b_dtype in _PAIRS_8BIT:
            yield _spec8(
                *problem, a_dtype=a_dtype, b_dtype=b_dtype, tile_m=128, **fields
            )
        yield _spec6(*problem, tile_m=128, **fields)

    def test_steps_hold_the_k_edges_and_tile_boundaries(self):
        spec = _spec(64, 48, 2048 + 96, tile_m=128, tile_n=128)
        steps = sparse_steps(spec)
        self.assertEqual(len(steps), SPARSE_STEPS)
        self.assertTrue({0, 3, 4, 15, 16} <= set(steps.tolist()))
        self.assertEqual(sparse_steps(_spec(64, 48, 256, tile_k=128)).tolist(), [0, 1])

    def test_inputs_cover_all_codes_and_are_zero_elsewhere(self):
        spec = _spec(64, 48, 2048 + 96, tile_m=128, tile_n=128)
        data = sparse_inputs(spec)
        active = np.zeros(spec.K, dtype=bool)
        for step in sparse_steps(spec):
            active[step * 128 : (step + 1) * 128] = True
        for codes in data[5:]:
            self.assertEqual(set(np.unique(codes[:, active])), set(range(16)))
            self.assertFalse(codes[:, ~active].any())
        self.assertEqual(
            set(np.unique(data[2])) | set(np.unique(data[3])), {125, 126, 127, 128}
        )

    def test_8bit_inputs_are_narrow_and_zero_elsewhere(self):
        """8-bit sparse values are +-[2^-1, 1.875] in steps of 2^-4 with
        E8M0 2^-1..2^0, and the exact check is FP4's 0 / +1 re-encoded."""
        for spec in (s for s in self._specs() if s.elem_bits == 8):
            data = sparse_inputs(spec)
            active = np.zeros(spec.K, dtype=bool)
            for step in sparse_steps(spec):
                active[step * 128 : (step + 1) * 128] = True
            with self.subTest(spec.kernel_name()):
                for codes, dtype, device in (
                    (data[5], spec.a_dtype, data[0]),
                    (data[6], spec.b_dtype, data[1]),
                ):
                    self.assertIs(device.dtype, np.dtype(np.uint8))
                    self.assertEqual(device.shape, codes.shape)
                    values = np.abs(decode_table(dtype)[codes[:, active]])
                    self.assertGreaterEqual(values.min(), 0.5)
                    self.assertLessEqual(values.max(), 1.875)
                    self.assertFalse((values * 16 % 1).any())
                    self.assertFalse(codes[:, ~active].any())
                scales = set(np.unique(data[2])) | set(np.unique(data[3]))
                self.assertEqual(scales, {126, 127})
                fp4 = check_inputs(_spec(spec.M, spec.N, spec.K))
                exact = check_inputs(spec)
                for index, dtype in ((5, spec.a_dtype), (6, spec.b_dtype)):
                    self.assertTrue(
                        np.array_equal(
                            decode_table(dtype)[exact[index]], _E2M1[fp4[index]]
                        )
                    )
                self.assertTrue(np.array_equal(exact[0], exact[5]))

    def test_fp6_inputs(self):
        """FP6 sparse data holds all 64 E2M3 codes in the sparse K-steps with
        A scales 2^-1..2^0 and B scales 2^0; the exact check is FP4's 0 / +1
        re-encoded; the device operands are the codes packed four in three
        bytes, as ``pack_fp6_codes`` packs them and ``decode_fp6`` reads them."""
        spec = list(self._specs())[-1]
        self.assertEqual(spec.a_dtype, "fp6")
        data = sparse_inputs(spec)
        active = np.zeros(spec.K, dtype=bool)
        for step in sparse_steps(spec):
            active[step * 128 : (step + 1) * 128] = True
        for codes, device in ((data[5], data[0]), (data[6], data[1])):
            self.assertEqual(set(np.unique(codes[:, active])), set(range(64)))
            self.assertFalse(codes[:, ~active].any())
            self.assertEqual(device.shape, (codes.shape[0], spec.K * 3 // 4))
            self.assertTrue(np.array_equal(device, pack_fp6_rows(codes)))
        self.assertEqual(set(np.unique(data[2])), {126, 127})
        self.assertEqual(set(np.unique(data[3])), {127})
        exact = check_inputs(spec)
        fp4 = check_inputs(_spec(spec.M, spec.N, spec.K))
        for index in (5, 6):
            self.assertTrue(
                np.array_equal(decode_table("fp6")[exact[index]], _E2M1[fp4[index]])
            )
            self.assertTrue(
                np.array_equal(exact[index - 5], pack_fp6_rows(exact[index]))
            )
        codes = np.random.default_rng(3).integers(0, 64, (5, 96), dtype=np.uint8)
        packed = pack_fp6_rows(codes)
        self.assertTrue(np.array_equal(packed, pack_fp6_codes(codes)))
        self.assertTrue(np.array_equal(decode_fp6(packed), decode_table("fp6")[codes]))

    def test_decode_tables_and_timed_inputs(self):
        fp8, bf8 = decode_table("fp8"), decode_table("bf8")
        self.assertEqual((fp8[0x38], bf8[0x3C], fp8[0xC0], bf8[0xC0]), (1, 1, -2, -2))
        self.assertTrue(np.isnan(fp8[0x7F]) and np.isinf(bf8[0x7C]))
        fp6 = decode_table("fp6")
        self.assertEqual(len(fp6), 64)
        self.assertEqual(
            (fp6[0x08], fp6[0x28], fp6[0x1F], fp6[0x01], fp6[0x07]),
            (1, -1, 7.5, 0.125, 0.875),
        )
        # FP6 timing data: codes with the exponent MSB clear (0..1.875, both
        # signs), packed, and E8M0 scales 2^-3..2^3.
        spec = _spec6(256, 128, 512)
        a, b, a_scale, b_scale = timed_inputs(spec)
        self.assertEqual((a.shape, b.shape), ((256, 384), (128, 384)))
        self.assertEqual((a_scale.shape, b_scale.shape), ((256, 16), (16, 128)))
        for packed in (a, b):
            values = decode_fp6(packed)
            self.assertLessEqual(np.abs(values).max(), 1.875)
            self.assertTrue((values < 0).any() and (values > 1).any())
            # 32 codes; +0 and -0 compare equal.
            self.assertEqual(len(np.unique(values)), 31)
        self.assertEqual(
            set(np.unique(a_scale)) | set(np.unique(b_scale)), set(range(124, 131))
        )
        for a_dtype, b_dtype in _PAIRS_8BIT:
            spec = _spec8(256, 128, 512, a_dtype=a_dtype, b_dtype=b_dtype)
            a, b, a_scale, b_scale = timed_inputs(spec)
            with self.subTest(a_dtype=a_dtype, b_dtype=b_dtype):
                self.assertEqual((a.shape, b.shape), ((256, 512), (128, 512)))
                self.assertEqual((a_scale.shape, b_scale.shape), ((256, 16), (16, 128)))
                for codes, dtype in ((a, a_dtype), (b, b_dtype)):
                    values = decode_table(dtype)[codes]
                    self.assertTrue(np.isfinite(values).all())
                    self.assertLess(np.abs(values).max(), 2)

    def test_fp32_sums_are_exact_in_any_order(self):
        self.assertLess(SPARSE_STEPS * 128 * 144 * 2**6, 2**24)
        self.assertLess(SPARSE_STEPS * 128 * 1.875**2 * 2**10, 2**24)
        self.assertLess(SPARSE_STEPS * 128 * 7.5**2 * 2**7, 2**24)
        for spec in self._specs():
            x, y = self._values(spec, sparse_inputs(spec))
            exact = x @ y.T
            for order in (1, -1):
                total = np.zeros(exact.shape, dtype=np.float32)
                for j in range(spec.K)[::order]:
                    total += np.outer(x[:, j], y[:, j]).astype(np.float32)
                with self.subTest(spec.kernel_name(), order=order):
                    self.assertTrue(np.array_equal(total, exact))

    def test_reference_is_the_exact_result_rounded(self):
        for spec, beta in itertools.product(
            (
                spec
                for out_dtype in OUT_DTYPES
                for spec in self._specs(out_dtype=out_dtype)
            ),
            (0.0, 0.5),
        ):
            out_dtype = spec.out_dtype
            data = sparse_inputs(spec)
            x, y = self._values(spec, data)
            alpha = np.float32(SPARSE_ALPHA)
            value = (alpha * (x @ y.T)).astype(np.float32) + np.float32(beta) * data[
                4
            ].astype(np.float32)
            if out_dtype == "fp8e4m3":
                value = np.clip(value, -448, 448)
            expected = value.astype(reference(spec, data, alpha=alpha, beta=beta).dtype)
            with self.subTest(spec.kernel_name(), beta=beta):
                got = reference(spec, data, alpha=alpha, beta=beta)
                self.assertEqual(got.tobytes(), expected.tobytes())

    def test_a_dropped_last_step_changes_most_outputs_at_large_k(self):
        """U8's K (109760, last step 64 wide), bf16 out."""
        spec = _spec(64, 48, 109760, out_dtype="bf16")
        data = list(sparse_inputs(spec))
        alpha = np.float32(SPARSE_ALPHA)
        good = reference(spec, data, alpha=alpha, beta=0.0)
        data[5] = data[5].copy()
        data[5][:, (spec.K - 1) // 128 * 128 :] = 0
        bad = reference(spec, data, alpha=alpha, beta=0.0)
        self.assertGreater(np.mean(good.view(np.uint16) != bad.view(np.uint16)), 0.9)


@unittest.skipIf(shutil.which("llvm-objdump") is None, "needs llvm-objdump")
class MxGemmHostBuildTest(unittest.TestCase):
    def test_resources(self):
        """Host compiles: VGPR file and LDS limits, no spills."""
        import tempfile
        from pathlib import Path

        from rocke.analysis.isa import analyze_hsaco
        from rocke.helpers.compile import compile_kernel

        for spec in (
            _spec(*_BENCH),
            _spec(tile_m=128, tile_n=256, tile_k=256, wave_group=(4, 1)),
            _spec(cluster_dim=(1, 1, 1), half_plr="none", out_dtype="fp16"),
            _spec(1024, 1024, 2048 + 384, tile_m=128, tile_n=128),
            _spec(tile_m=128, tile_n=128, tile_k=128),
            _spec(1000, 1000, 2048),
            _spec(1024, 1024, 2048 + 96, tile_m=128, tile_n=128),
            _spec(1024, 1001, 2048, tile_m=128, tile_n=128, out_dtype="bf16"),
            _spec(mma="16x16"),
            _spec(
                1024,
                16,
                2048,
                tile_m=256,
                tile_n=16,
                wave_group=(4, 1),
                half_plr="none",
                cluster_dim=(1, 2, 1),
                mma="16x16",
            ),
            _spec(
                1024,
                16,
                2048,
                tile_m=64,
                tile_n=16,
                wave_group=(4, 1),
                half_plr="none",
                cluster_dim=(1, 1, 1),
                mma="16x16",
            ),
            _spec(
                1024,
                16,
                2048,
                tile_m=128,
                tile_n=16,
                wave_group=(4, 1),
                half_plr="none",
                cluster_dim=(1, 2, 1),
            ),
            # hipBLASLt's MXFP8 MAF and MAB tunings.
            _spec8(*_BENCH, local_prefetch=0, half_plr="none"),
            _spec8(*_BENCH, half_b_order="late"),
            _spec(*_BENCH, half_b_order="late"),
            _spec8(
                65536,
                16,
                32768,
                tile_m=256,
                tile_n=16,
                wave_group=(4, 1),
                half_plr="none",
                cluster_dim=(1, 1, 1),
            ),
            _spec8(*_BENCH, tile_m=128, a_dtype="bf8", b_dtype="bf8"),
            _spec8(1000, 1000, 2048 + 96, tile_m=128, tile_n=128, b_dtype="bf8"),
            # FP6 at its large tiles, both reads, and an edge tile with a K tail.
            *(
                _spec6(*shape, tile_m=tm, tile_n=tn, tile_k=tk, **fields)
                for shape, (tm, tn, tk) in itertools.product(_MAF_MED, _FP6_TILES)
                for fields in (
                    {"local_prefetch": 0, "half_plr": "none"},
                    {"local_prefetch": 0, "half_plr": "none", "k_reads": "contiguous"},
                    {
                        "local_prefetch": 0,
                        "half_plr": "none",
                        "k_reads": "contiguous",
                        "a_lds_pad": LdsPad(),
                        "b_lds_pad": LdsPad(),
                    },
                )
            ),
            _spec6(
                49152,
                16,
                32768,
                tile_m=256,
                tile_n=16,
                wave_group=(4, 1),
                half_plr="none",
                cluster_dim=(1, 1, 1),
            ),
            _spec6(1000, 1000, 2048 + 96, tile_m=128, tile_n=128, tile_k=384),
            # B reuse at the MAF/MED tiles.
            *(
                make(
                    *_MAF_MED[0], local_prefetch=0, half_plr="none", wmma_reuse="b", **f
                )
                for make, f in (
                    (_spec6, {"tile_k": 384, "k_reads": "contiguous"}),
                    (_spec8, {}),
                )
            ),
            # Three LDS buffers at the reference design's tile, with and
            # without read-ahead.
            *(
                _spec6(
                    *_MAF_MED[0],
                    tile_m=128,
                    tile_n=128,
                    tile_k=512,
                    cluster_dim=(2, 4, 1),
                    k_reads="contiguous",
                    lds_buffers=3,
                    **f,
                )
                for f in (_DIRECT, {"half_plr": "b"})
            ),
            # Scale swaps after the K-step's wait on every read-ahead path
            # at the reference design's tile.
            *(
                _spec6(
                    *_MAF_MED[0],
                    tile_m=128,
                    tile_n=128,
                    tile_k=512,
                    cluster_dim=(2, 4, 1),
                    k_reads="contiguous",
                    scale_swap="wait",
                    **f,
                )
                for f in (
                    {"half_plr": "b"},
                    {"half_plr": "b", "half_b_order": "late"},
                    {"half_plr": "none"},
                    {"half_plr": "b", "lds_buffers": 3},
                    {"half_plr": "none", "read_issue": "interleave"},
                    {"half_plr": "none", "read_issue": "interleave", "lds_buffers": 3},
                )
            ),
            # Panel walks at the MAF shape: block and interleaved reads, half-B.
            *(
                _spec6(
                    *_MAF_MED[0],
                    cluster_dim=(4, 4, 1),
                    k_reads="contiguous",
                    scale_swap="wait",
                    **_PANELS,
                    **f,
                )
                for f in (
                    {"half_plr": "none", "read_issue": "interleave"},
                    {"half_plr": "none"},
                    {"half_plr": "b"},
                )
            ),
            # Half walks at the MAF shape: block reads, half-B ahead and late,
            # fixed read bases (interleaved reads hold two K-steps of the
            # half's operands and spill).
            *(
                _spec6(
                    *_MAF_MED[0],
                    cluster_dim=(4, 4, 1),
                    k_reads="contiguous",
                    scale_swap="wait",
                    **_HALVES,
                    **f,
                )
                for f in (
                    {"half_plr": "none"},
                    {"half_plr": "none", "read_bases": "fixed"},
                    {"half_plr": "b"},
                    {"half_plr": "b", "half_b_order": "late"},
                )
            ),
            # The ring at MAF and MED: B in halves late and ahead, scale swaps
            # after the wait and at the read.
            *(
                _spec6(*shape, cluster_dim=cluster, **_RING | f)
                for shape, cluster in zip(_MAF_MED, ((4, 4, 1), (2, 4, 1)))
                for f in (
                    {"scale_swap": "wait", "wmma_reuse": "b"},
                    {"half_b_order": "ahead"},
                )
            ),
            # Interleaved reads at the MAB tiling (one N-repeat, one-word B
            # scales).
            _spec6(
                49152,
                16,
                32768,
                tile_m=256,
                tile_n=16,
                tile_k=512,
                wave_group=(4, 1),
                cluster_dim=(1, 2, 1),
                k_reads="contiguous",
                scale_swap="wait",
                local_prefetch=1,
                half_plr="none",
                read_issue="interleave",
            ),
            # Fixed read bases at the MED, MAF (panel walk), MAB and SM tilings.
            _spec6(
                *_MAF_MED[1],
                tile_m=128,
                tile_n=128,
                tile_k=512,
                cluster_dim=(2, 4, 1),
                lds_buffers=3,
                k_reads="contiguous",
                scale_swap="wait",
                local_prefetch=1,
                half_plr="none",
                read_issue="interleave",
                read_bases="fixed",
            ),
            _spec6(
                *_MAF_MED[0],
                cluster_dim=(4, 4, 1),
                k_reads="contiguous",
                scale_swap="wait",
                **_PANELS,
                half_plr="none",
                read_issue="interleave",
                read_bases="fixed",
            ),
            _spec6(
                49152,
                16,
                32768,
                tile_m=256,
                tile_n=16,
                tile_k=512,
                wave_group=(4, 1),
                cluster_dim=(1, 2, 1),
                k_reads="contiguous",
                read_bases="fixed",
                **_DIRECT,
            ),
            _spec6(
                256,
                2048,
                2048,
                tile_m=64,
                tile_n=64,
                tile_k=512,
                cluster_dim=(4, 2, 1),
                k_reads="contiguous",
                scale_swap="wait",
                half_b_order="late",
                read_bases="fixed",
            ),
            # Register scales at the MAF/MED, MAB and SM tilings.
            *(
                _spec6(
                    *_MAF_MED[0],
                    tile_k=384,
                    cluster_dim=(2, 4, 1),
                    k_reads="contiguous",
                    scales="registers",
                    wmma_reuse=reuse,
                    **_DIRECT,
                )
                for reuse in WMMA_REUSE
            ),
            _spec8(
                49152,
                16,
                32768,
                tile_m=256,
                tile_n=64,
                wave_group=(4, 1),
                cluster_dim=(1, 2, 1),
                scales="registers",
                wmma_reuse="ab",
                **_DIRECT,
            ),
            _spec6(
                49152,
                16,
                32768,
                tile_m=256,
                tile_n=16,
                tile_k=512,
                wave_group=(4, 1),
                cluster_dim=(1, 2, 1),
                k_reads="contiguous",
                scales="registers",
                **_DIRECT,
            ),
            _spec6(
                256,
                2048,
                2048,
                tile_m=64,
                tile_n=64,
                tile_k=512,
                cluster_dim=(4, 2, 1),
                scales="registers",
                wmma_reuse="ab",
                **_DIRECT,
            ),
        ):
            with self.subTest(spec.kernel_name()):
                artifact = compile_kernel(
                    build_mx_gemm(spec), arch="gfx1250", backend="python"
                )
                with tempfile.NamedTemporaryFile(suffix=".hsaco") as handle:
                    handle.write(artifact.hsaco)
                    handle.flush()
                    info = analyze_hsaco(Path(handle.name), keep_text=True)
                self.assertLessEqual(info.resources.vgpr_count, 1024)
                self.assertEqual(info.resources.lds_bytes, spec.lds_bytes)
                self.assertEqual(info.resources.scratch_bytes, 0)
                self.assertEqual(loop_scratch_ops(info.objdump_text), 0)


class MxGemmDeviceTest(unittest.TestCase):
    def test_matches_reference_on_gfx1250(self):
        if get_device_arch(0) != "gfx1250":
            self.skipTest("visible HIP device 0 is not gfx1250")
        for spec in (
            list(_valid_geometries())
            + list(_valid_geometries("16x16"))
            + [
                _spec(
                    1000,
                    16,
                    2048,
                    tile_m=64,
                    tile_n=16,
                    wave_group=(4, 1),
                    half_plr="none",
                    cluster_dim=(1, 1, 1),
                    mma="16x16",
                ),
                _spec(out_dtype="fp16"),
                _spec(out_dtype="bf16"),
                _spec(cluster_dim=(1, 1, 1)),
                _spec(half_plr="none"),
                _spec(half_plr="none", local_prefetch=0),
                _spec(half_b_order="late"),
                _spec(tile_k=128, half_b_order="late"),
            ]
            + list(_valid_geometries("16x16", a_dtype="fp8", b_dtype="fp8"))
            + [
                _spec8(a_dtype=a_dtype, b_dtype=b_dtype, **fields)
                for a_dtype, b_dtype in _PAIRS_8BIT
                for fields in (
                    {"tile_m": 128},
                    {"tile_m": 128, "tile_k": 128, "half_plr": "none"},
                    {"local_prefetch": 0, "half_plr": "none"},
                    {"half_b_order": "late"},
                    {"tile_k": 128, "half_b_order": "late"},
                    {
                        "tile_m": 256,
                        "tile_n": 16,
                        "wave_group": (4, 1),
                        "half_plr": "none",
                        "cluster_dim": (1, 2, 1),
                    },
                )
            ]
            + [
                _spec8(1000, 1000, 2048 + 96, tile_m=128, tile_n=128, b_dtype="bf8"),
                _spec8(tile_m=128, out_dtype="bf16"),
                _spec8(local_prefetch=0, half_plr="none", wmma_reuse="b"),
            ]
            + list(_valid_geometries("16x16", a_dtype="fp6", b_dtype="fp6"))
            + [
                _spec6(**fields)
                for fields in (
                    {"k_reads": "contiguous"},
                    {"k_reads": "contiguous", "tile_k": 384},
                    {"k_reads": "contiguous", "half_b_order": "late"},
                    {"local_prefetch": 0, "half_plr": "none"},
                    {"a_lds_pad": LdsPad(), "b_lds_pad": LdsPad()},
                    {"tile_k": 384, "local_prefetch": 0, "half_plr": "none"},
                    {
                        "tile_k": 384,
                        "local_prefetch": 0,
                        "half_plr": "none",
                        "k_reads": "contiguous",
                        "wmma_reuse": "b",
                    },
                    {
                        "tile_m": 256,
                        "tile_n": 16,
                        "wave_group": (4, 1),
                        "half_plr": "none",
                        "cluster_dim": (1, 2, 1),
                    },
                )
            ]
            + [
                _spec6(1000, 1000, 2048 + 96, tile_m=128, tile_n=128, tile_k=384),
                _spec6(
                    1000,
                    1000,
                    2048 + 96,
                    tile_m=128,
                    tile_n=128,
                    k_reads="contiguous",
                    out_dtype="bf16",
                ),
            ]
            # FP6 K a multiple of 16: partial last scale blocks, K shorter
            # than tile_k, M = 1 and 128 (partial M tiles).
            + [
                _spec6(M, 512, K, tile_m=tile_m, tile_k=tile_k, **fields)
                for M, tile_m, fields in (
                    (1, 16, {"wave_group": (1, 4), "cluster_dim": (2, 1, 1)}),
                    (128, 64, {"cluster_dim": (1, 1, 1)}),
                )
                for K, tile_k in (
                    (32, 128),
                    (80, 256),
                    (112, 128),
                    (144, 512),
                    (240, 256),
                    (528, 256),
                    (1840, 512),
                )
            ]
            + [
                _spec6(128, 512, K, tile_m=64, tile_n=64, tile_k=256, **fields)
                for K in (80, 528)
                for fields in (
                    _DIRECT | {"cluster_dim": (1, 1, 1)},
                    {"half_b_order": "late", "k_reads": "contiguous"}
                    | {
                        "scale_swap": "wait",
                        "lds_buffers": 3,
                        "cluster_dim": (2, 2, 1),
                    },
                    {"half_plr": "none", "k_reads": "contiguous", "scale_swap": "wait"}
                    | {"read_issue": "interleave", "cluster_dim": (1, 1, 1)},
                    {"half_b_order": "late", "wmma_reuse": "b", "read_bases": "fixed"}
                    | {"cluster_dim": (1, 1, 1)},
                    _DIRECT | {"scales": "registers", "cluster_dim": (1, 1, 1)},
                )
            ]
            # One D column per store at an alpha (1 / 384) whose products
            # round to FP32 ties between two FP16/BF16 values: whole and
            # partial M tiles.
            + [
                _spec6(
                    M,
                    256,
                    384,
                    tile_m=16,
                    tile_n=64,
                    tile_k=384,
                    wave_group=(1, 4),
                    cluster_dim=(1, 1, 1),
                    out_dtype=out_dtype,
                    **_DIRECT,
                )
                for M in (64, 40)
                for out_dtype in ("fp16", "bf16")
            ]
            # Three LDS buffers: every peel, a K tail, read-ahead, FP8.
            + [
                _spec6(1024, 1024, k_dim, **fields)
                for k_dim in (2048, 2560, 3072, 3072 + 256)
                for fields in (
                    {"tile_m": 128, "tile_n": 128, "tile_k": 512, "lds_buffers": 3}
                    | _DIRECT,
                    {
                        "tile_m": 128,
                        "tile_n": 128,
                        "tile_k": 512,
                        "lds_buffers": 3,
                        "k_reads": "contiguous",
                    },
                )
            ]
            + [
                _spec8(tile_m=128, tile_n=128, lds_buffers=3),
                _spec8(tile_m=128, tile_n=128, lds_buffers=3, **_DIRECT),
            ]
            # Scale swaps after the K-step's wait: every compute path, three
            # buffers, edge tiles and a K tail.
            + [
                _spec6(*problem, k_reads="contiguous", scale_swap="wait", **fields)
                for problem, fields in (
                    ((), _DIRECT),
                    ((), {"tile_m": 128, "tile_n": 128, "half_plr": "none"}),
                    ((), {"tile_m": 128, "tile_n": 128}),
                    ((), {"tile_m": 128, "tile_n": 128, "half_b_order": "late"}),
                    (
                        (1024, 1024, 3072 + 256),
                        {"tile_m": 128, "tile_n": 128, "tile_k": 512, "lds_buffers": 3},
                    ),
                    (
                        (1000, 1000, 2048 + 96),
                        {"tile_m": 128, "tile_n": 128, "tile_k": 384},
                    ),
                )
            ]
            # Interleaved reads: three buffers, edge tiles and a
            # K tail, a one-K-step tail, narrow N, FP8 and chunked FP6.
            + [
                make(*problem, local_prefetch=1, half_plr="none", **fields)
                for make, problem, fields in (
                    (_spec6, (), {"k_reads": "contiguous", "scale_swap": "wait"}),
                    (
                        _spec6,
                        (1024, 1024, 3072 + 256),
                        {
                            "tile_m": 128,
                            "tile_n": 128,
                            "tile_k": 512,
                            "lds_buffers": 3,
                            "k_reads": "contiguous",
                            "scale_swap": "wait",
                        },
                    ),
                    (
                        _spec6,
                        (1000, 1000, 2048 + 96),
                        {"tile_m": 128, "tile_n": 128, "tile_k": 384},
                    ),
                    (
                        _spec6,
                        (1024, 1024, 2048 + 128),
                        {"tile_m": 64, "tile_n": 128, "tile_k": 512},
                    ),
                    (
                        _spec6,
                        (1024, 16, 2048),
                        {
                            "tile_m": 128,
                            "tile_n": 16,
                            "wave_group": (4, 1),
                            "cluster_dim": (1, 2, 1),
                            "k_reads": "contiguous",
                            "scale_swap": "wait",
                        },
                    ),
                    (_spec8, (), {"tile_m": 128, "tile_n": 128}),
                )
                for fields in ({**fields, "read_issue": "interleave"},)
            ]
            # Panel walks: every read-ahead path, clusters, two loop passes,
            # chunked FP6.
            + [
                _spec6(*problem, **_PANELS, **fields)
                for problem, fields in (
                    (
                        (512, 512, 2048),
                        {
                            "half_plr": "none",
                            "k_reads": "contiguous",
                            "scale_swap": "wait",
                            "read_issue": "interleave",
                            "cluster_dim": (2, 2, 1),
                        },
                    ),
                    ((512, 512, 2048), {"half_plr": "none", "cluster_dim": (1, 1, 1)}),
                    (
                        (512, 512, 4096),
                        {
                            "half_plr": "b",
                            "k_reads": "contiguous",
                            "scale_swap": "wait",
                            "cluster_dim": (2, 2, 1),
                        },
                    ),
                    (
                        (512, 256, 2048),
                        {"half_b_order": "late", "cluster_dim": (1, 1, 1)},
                    ),
                )
            ]
            # Half walks: every read-ahead path, clusters, two and four loop
            # passes, chunked FP6, fixed read bases.
            + [
                _spec6(*problem, **_HALVES, **fields)
                for problem, fields in (
                    (
                        (512, 512, 2048),
                        {
                            "half_plr": "none",
                            "k_reads": "contiguous",
                            "scale_swap": "wait",
                            "read_issue": "interleave",
                            "cluster_dim": (2, 2, 1),
                        },
                    ),
                    ((512, 512, 2048), {"half_plr": "none", "cluster_dim": (1, 1, 1)}),
                    (
                        (512, 512, 4096),
                        {
                            "half_plr": "b",
                            "k_reads": "contiguous",
                            "scale_swap": "wait",
                            "cluster_dim": (2, 2, 1),
                        },
                    ),
                    (
                        (256, 512, 2048),
                        {"half_b_order": "late", "cluster_dim": (2, 1, 1)},
                    ),
                    (
                        (512, 256, 2048),
                        {
                            "half_plr": "none",
                            "k_reads": "contiguous",
                            "scale_swap": "wait",
                            "read_issue": "interleave",
                            "read_bases": "fixed",
                            "cluster_dim": (1, 2, 1),
                        },
                    ),
                )
            ]
            # B reuse with B in halves: half-B late with two and three
            # buffers and a K tail, half-B ahead, FP8, the 64x128x512 tile.
            + [
                make(*problem, wmma_reuse="b", **fields)
                for make, problem, fields in (
                    (
                        _spec6,
                        (512, 512, 2048 + 512),
                        {
                            "tile_m": 256,
                            "tile_n": 128,
                            "tile_k": 512,
                            "half_b_order": "late",
                            "k_reads": "contiguous",
                            "scale_swap": "wait",
                            "cluster_dim": (2, 2, 1),
                        },
                    ),
                    (
                        _spec6,
                        (1000, 1000, 2048 + 256),
                        {
                            "half_b_order": "late",
                            "lds_buffers": 3,
                            "k_reads": "contiguous",
                            "scale_swap": "wait",
                            "cluster_dim": (1, 1, 1),
                        },
                    ),
                    (_spec6, (512, 512, 2048), {"cluster_dim": (2, 2, 1)}),
                    (
                        _spec8,
                        (512, 512, 2048),
                        {"half_b_order": "late", "cluster_dim": (2, 2, 1)},
                    ),
                    (
                        _spec6,
                        (256, 512, 2048),
                        {
                            "tile_m": 64,
                            "tile_n": 128,
                            "tile_k": 512,
                            "half_b_order": "late",
                            "lds_buffers": 3,
                            "k_reads": "contiguous",
                            "scale_swap": "wait",
                            "cluster_dim": (4, 2, 1),
                        },
                    ),
                )
            ]
            # Instruction prefetch: the panel walk, half-B late with three
            # buffers and a K tail, FP8 half-B late.
            + [
                make(*problem, inst_prefetch=every, **fields)
                for make, problem, every, fields in (
                    (
                        _spec6,
                        (512, 512, 2048),
                        16,
                        {
                            **_PANELS,
                            "half_plr": "none",
                            "k_reads": "contiguous",
                            "scale_swap": "wait",
                            "read_issue": "interleave",
                            "cluster_dim": (2, 2, 1),
                        },
                    ),
                    (
                        _spec6,
                        (1024, 1024, 2048 + 256),
                        8,
                        {
                            "tile_m": 256,
                            "tile_n": 256,
                            "local_prefetch": 1,
                            "half_b_order": "late",
                            "lds_buffers": 3,
                            "k_reads": "contiguous",
                            "scale_swap": "wait",
                        },
                    ),
                    (
                        _spec8,
                        (512, 512, 2048),
                        32,
                        {"half_b_order": "late", "cluster_dim": (2, 2, 1)},
                    ),
                )
            ]
            # Pair map: half-B late and ahead with two and three buffers,
            # chunked and contiguous FP6, edge tiles and a K tail, interleaved
            # reads with fixed read bases, groups of one and two N-repeats,
            # clusters, FP8.
            + [
                make(*problem, **fields)
                for make, problem, fields in (
                    (
                        _spec6,
                        (512, 512, 2048),
                        {
                            **_PAIRS,
                            "lds_buffers": 3,
                            "k_reads": "contiguous",
                            "scale_swap": "wait",
                            "cluster_dim": (2, 2, 1),
                        },
                    ),
                    (
                        _spec6,
                        (512, 512, 2048),
                        {**_PAIRS, "half_b_order": "ahead", "cluster_dim": (1, 1, 1)},
                    ),
                    (
                        _spec6,
                        (1000, 1000, 2048 + 96),
                        {**_PAIRS, "k_reads": "contiguous", "cluster_dim": (1, 1, 1)},
                    ),
                    (
                        _spec6,
                        (1024, 1024, 3072 + 256),
                        {
                            "tile_m": 128,
                            "tile_n": 128,
                            "tile_k": 512,
                            "local_prefetch": 1,
                            "half_plr": "none",
                            "lds_buffers": 3,
                            "k_reads": "contiguous",
                            "scale_swap": "wait",
                            "read_issue": "interleave",
                            "read_bases": "fixed",
                            "wave_map": "pairs",
                        },
                    ),
                    (
                        _spec6,
                        (256, 512, 2048),
                        {
                            "tile_m": 64,
                            "tile_n": 64,
                            "tile_k": 512,
                            "half_b_order": "late",
                            "cluster_dim": (4, 2, 1),
                            "wave_map": "pairs",
                        },
                    ),
                    (_spec8, (512, 512, 2048), {**_DIRECT, "wave_map": "pairs"}),
                    (
                        _spec8,
                        (512, 512, 2048),
                        {**_PAIRS, "cluster_dim": (2, 2, 1), "out_dtype": "fp16"},
                    ),
                )
            ]
            # Fixed read bases: three buffers with interleaved reads and a K
            # tail, edge tiles with half-B, a panel walk, one N-repeat, FP8.
            + [
                _spec6(
                    1024,
                    1024,
                    2048 + 128,
                    tile_m=128,
                    tile_n=128,
                    tile_k=512,
                    lds_buffers=3,
                    k_reads="contiguous",
                    scale_swap="wait",
                    local_prefetch=1,
                    half_plr="none",
                    read_issue="interleave",
                    read_bases="fixed",
                ),
                _spec6(
                    1000,
                    1000,
                    2048,
                    tile_m=128,
                    tile_n=128,
                    tile_k=512,
                    k_reads="contiguous",
                    scale_swap="wait",
                    read_bases="fixed",
                ),
                _spec6(
                    512,
                    512,
                    2048,
                    **_PANELS,
                    half_plr="none",
                    k_reads="contiguous",
                    scale_swap="wait",
                    read_issue="interleave",
                    cluster_dim=(2, 2, 1),
                    read_bases="fixed",
                ),
                _spec6(
                    1024,
                    16,
                    2048,
                    tile_m=128,
                    tile_n=16,
                    tile_k=512,
                    wave_group=(4, 1),
                    cluster_dim=(1, 2, 1),
                    k_reads="contiguous",
                    read_bases="fixed",
                    **_DIRECT,
                ),
                _spec6(
                    256,
                    512,
                    2048,
                    tile_m=64,
                    tile_n=64,
                    tile_k=512,
                    cluster_dim=(4, 2, 1),
                    half_b_order="late",
                    k_reads="contiguous",
                    scale_swap="wait",
                    read_bases="fixed",
                ),
                _spec8(
                    tile_m=128,
                    tile_n=128,
                    local_prefetch=1,
                    half_plr="none",
                    read_issue="interleave",
                    read_bases="fixed",
                ),
            ]
            # Register scales: every half packing, the reuse orders, edge
            # tiles and K tails.
            + [
                _spec6(scales="registers", **_DIRECT, **fields)
                for fields in (
                    {},
                    {"k_reads": "contiguous", "tile_k": 384, "wmma_reuse": "b"},
                    {"k_reads": "contiguous", "wmma_reuse": "ab"},
                    {"wmma_reuse": "a"},
                    {"tile_m": 64, "tile_n": 64},
                    {
                        "tile_m": 256,
                        "tile_n": 16,
                        "wave_group": (4, 1),
                        "cluster_dim": (1, 2, 1),
                    },
                )
            ]
            + [
                _spec6(
                    1000,
                    1000,
                    2048 + 96,
                    tile_m=128,
                    tile_n=128,
                    tile_k=384,
                    scales="registers",
                    **_DIRECT,
                ),
                _spec6(
                    1000,
                    1000,
                    2048 + 96,
                    tile_m=128,
                    tile_n=128,
                    k_reads="contiguous",
                    out_dtype="bf16",
                    scales="registers",
                    wmma_reuse="ab",
                    **_DIRECT,
                ),
                _spec8(scales="registers", wmma_reuse="ab", **_DIRECT),
                _spec8(
                    1000,
                    1000,
                    2048 + 96,
                    tile_m=128,
                    tile_n=128,
                    b_dtype="bf8",
                    scales="registers",
                    **_DIRECT,
                ),
            ]
            # Tiled operands: A, B or both on 256x256x256 and 256x128x512
            # half-B late (rows of 192 and 384 B, 6 and 12 lines per block),
            # 128x128 interleaved reads, the N 16 tile, 64x128 with three
            # buffers; MXFP8 in iterate blocks and as padded whole-line rows.
            + [
                _spec6(*problem, tiled_operands=tiled, **fields)
                for problem, fields, tiled_set in (
                    (
                        (1024, 1024, 2048),
                        {
                            "tile_m": 256,
                            "tile_n": 256,
                            "lds_buffers": 3,
                            "half_b_order": "late",
                            "k_reads": "contiguous",
                            "scale_swap": "wait",
                        },
                        ("a", "b", "ab"),
                    ),
                    (
                        (1024, 1024, 2048),
                        {
                            "tile_m": 256,
                            "tile_n": 128,
                            "tile_k": 512,
                            "half_b_order": "late",
                            "k_reads": "contiguous",
                            "scale_swap": "wait",
                            "wmma_reuse": "b",
                        },
                        ("a", "b", "ab"),
                    ),
                    (
                        (512, 512, 2048),
                        {
                            "tile_m": 128,
                            "tile_n": 128,
                            "tile_k": 512,
                            "lds_buffers": 3,
                            "half_plr": "none",
                            "k_reads": "contiguous",
                            "scale_swap": "wait",
                            "read_issue": "interleave",
                            "cluster_dim": (2, 4, 1),
                        },
                        ("ab",),
                    ),
                    (
                        (1024, 64, 2048),
                        {
                            "tile_n": 16,
                            "tile_k": 512,
                            "wave_group": (4, 1),
                            "k_reads": "contiguous",
                            "cluster_dim": (1, 2, 1),
                            **_DIRECT,
                        },
                        ("a", "ab"),
                    ),
                    (
                        (256, 512, 2048),
                        {
                            "tile_m": 64,
                            "tile_n": 128,
                            "tile_k": 512,
                            "lds_buffers": 3,
                            "half_b_order": "late",
                            "k_reads": "contiguous",
                            "scale_swap": "wait",
                            "cluster_dim": (4, 2, 1),
                        },
                        ("ab",),
                    ),
                )
                for tiled in tiled_set
            ]
            # FP6 K tile 1024 (768 B rows): the N 16 tile direct (contiguous and
            # chunked) and with read-ahead, a K tail on N 64, 64x64 half-B late.
            + [
                _spec6(*problem, tile_k=1024, **fields)
                for problem, fields in (
                    (
                        (1024, 64, 4096),
                        {
                            "tile_m": 128,
                            "tile_n": 16,
                            "wave_group": (4, 1),
                            "k_reads": "contiguous",
                            "cluster_dim": (1, 2, 1),
                            **_DIRECT,
                        },
                    ),
                    (
                        (1024, 64, 4096),
                        {
                            "tile_m": 128,
                            "tile_n": 16,
                            "wave_group": (4, 1),
                            "cluster_dim": (1, 4, 1),
                            **_DIRECT,
                        },
                    ),
                    (
                        (1024, 64, 4096),
                        {
                            "tile_m": 128,
                            "tile_n": 16,
                            "wave_group": (4, 1),
                            "local_prefetch": 1,
                            "half_plr": "none",
                            "k_reads": "contiguous",
                            "scale_swap": "wait",
                            "cluster_dim": (1, 2, 1),
                        },
                    ),
                    (
                        (1024, 128, 4096 + 384),
                        {
                            "tile_m": 128,
                            "tile_n": 64,
                            "wave_group": (4, 1),
                            "k_reads": "contiguous",
                            "cluster_dim": (1, 1, 1),
                            **_DIRECT,
                        },
                    ),
                    (
                        (512, 512, 4096),
                        {
                            "tile_m": 64,
                            "tile_n": 64,
                            "half_b_order": "late",
                            "k_reads": "contiguous",
                            "scale_swap": "wait",
                            "cluster_dim": (2, 2, 1),
                        },
                    ),
                )
            ]
            + [
                _spec8(half_b_order="late", wmma_reuse="b", tiled_operands="ab"),
                _spec8(
                    1024,
                    64,
                    2048,
                    tile_n=64,
                    wave_group=(4, 1),
                    cluster_dim=(1, 2, 1),
                    tiled_operands="ab",
                    **_DIRECT,
                ),
            ]
            # The ring: one and two periods, clusters 1x1, 2x2 and 4x4, B in
            # halves late and ahead, scale swaps after the wait and at the
            # read, without GL2 prefetch.
            + [
                _spec6(*problem, **_RING | fields)
                for problem, fields in (
                    (
                        (512, 512, 4096),
                        {
                            "cluster_dim": (1, 1, 1),
                            "scale_swap": "wait",
                            "wmma_reuse": "b",
                        },
                    ),
                    ((512, 512, 4096), {"cluster_dim": (2, 2, 1)}),
                    (
                        (512, 512, 2048),
                        {
                            "cluster_dim": (2, 2, 1),
                            "half_b_order": "ahead",
                            "scale_swap": "wait",
                        },
                    ),
                    ((1024, 1024, 4096), {"cluster_dim": (4, 4, 1)}),
                    (
                        (1024, 1024, 4096),
                        {
                            "cluster_dim": (4, 4, 1),
                            "half_b_order": "ahead",
                            "gl2_prefetch_distance": 0,
                        },
                    ),
                )
            ]
        ):
            with self.subTest(spec.kernel_name()):
                run_case(spec)


if __name__ == "__main__":
    unittest.main()
