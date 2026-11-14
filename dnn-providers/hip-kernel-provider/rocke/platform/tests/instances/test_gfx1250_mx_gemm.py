# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Coverage for the flag-free gfx1250 MX GEMM (``mx_gemm``)."""

from __future__ import annotations

import itertools
import re
import shutil
import unittest
from dataclasses import replace

import numpy as np
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
    timed_inputs,
    verify_block,
)
from rocke.helpers.compile import _comgr_options_for_kernel
from rocke.instances.gfx1250.mx_gemm import (
    K_READS,
    OUT_DTYPES,
    TILE_K,
    TILE_MN,
    WAVE_GROUPS,
    WMMA_REUSE,
    IntOps,
    LdsPad,
    MxGemmSpec,
    build_mx_gemm,
    derived_lds_pads,
    validate,
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
        """FP6 takes the 16x16 atom and tile_k 128/256/384/512 (512 only
        below 256x256: LDS), packs four K values in three bytes, and is the
        only type with contiguous reads. 256x256x256 and 256x256x384 are
        valid at two large problems with the 4x4 cluster."""
        self.assertEqual(
            len(list(_valid_geometries("16x16", a_dtype="fp6", b_dtype="fp6"))), 133
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
        for shape, (tm, tn, tk) in itertools.product(_MAF_MED, _FP6_TILES):
            for k_reads in K_READS:
                fields = {"tile_m": tm, "tile_n": tn, "tile_k": tk, "k_reads": k_reads}
                with self.subTest(shape=shape, tile=(tm, tn, tk), k_reads=k_reads):
                    self.assertIsNone(
                        validate(_spec6(*shape, out_dtype="bf16", **fields))
                    )
        bad = {
            "32x16": ({"mma": "32x16"}, "mma must be one of"),
            "tile_k 1024": ({"tile_k": 1024}, "tile_k must be in"),
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
        """B reuse runs with local_prefetch 0 and the 16x16 WMMA, and tags
        the kernel name."""
        direct = {"local_prefetch": 0, "half_plr": "none"}
        self.assertEqual(WMMA_REUSE, ("none", "b"))
        for spec in (
            _spec6(wmma_reuse="b", **direct),
            _spec8(wmma_reuse="b", **direct),
        ):
            with self.subTest(spec.kernel_name()):
                self.assertIsNone(validate(spec))
                self.assertIn("_rb", spec.kernel_name())
        self.assertNotIn("_rb", _spec6(**direct).kernel_name())
        bad = {
            "value": (_spec6(wmma_reuse="a", **direct), "wmma_reuse must be one of"),
            "prefetch": (_spec6(wmma_reuse="b"), "needs local_prefetch 0"),
            "atom": (_spec(wmma_reuse="b", **direct), "is for the 16x16 WMMA"),
        }
        for label, (spec, reason) in bad.items():
            with self.subTest(label):
                self.assertIn(reason, validate(spec) or "")

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

    def test_fp6_rows_take_a_pad_every_bank_cycle_of_rows(self):
        """FP6 rows (96 B per 128 K) take 16 B after every 8, 4, 8 or 2 rows,
        the rows after which a row's first bank repeats; the blocks are no
        power of two, so they are TDM iterate mode. A power-of-two block
        that cuts a row stays rejected; "no pad" stays valid."""
        for tile_k, rows in ((128, 8), (256, 4), (384, 8), (512, 2)):
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
                # Row r of a lane half starts at bank (r * row words + pad
                # words) mod 64: 16 rows, 16 different 16 B bank groups.
                op = spec.lds_layout.a
                starts = {
                    spec.lds_layout.row_offset(IntOps(), "a", r) // 4 % 64
                    for r in range(16)
                }
                self.assertEqual(len(starts), 16)
                self.assertTrue(all(s % 4 == 0 for s in starts))
        self.assertIsNone(validate(_spec6(a_lds_pad=LdsPad(), b_lds_pad=LdsPad())))
        self.assertIsNone(validate(_spec6(a_lds_pad=LdsPad(32, 1536))))
        for pad in (LdsPad(16, 512), LdsPad(16, 1000), LdsPad(16, 192 * 3)):
            with self.subTest(pad=pad):
                self.assertIsNotNone(validate(_spec6(b_lds_pad=pad)))

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

    def test_wmma_reuse_b_orders_and_flags_the_wmmas(self):
        """With B reuse each K-step's WMMAs run N-repeat outer; every WMMA
        but the last of an N-repeat flags B and hands it to the next WMMA,
        which takes the same B. Without it no WMMA flags an operand."""
        call = re.compile(
            _WMMA + r"[^(]*\(i32 \d+, <16 x i32> (%[\w.]+), i32 \d+, "
            r"<16 x i32> (%[\w.]+),[^\n]*i1 (true|false), i1 (true|false)\)"
        )
        direct = {"local_prefetch": 0, "half_plr": "none"}
        for spec in (
            _spec6(k_reads="contiguous", wmma_reuse="b", **direct),
            _spec8(wmma_reuse="b", **direct),
        ):
            rm, rn = spec.wave_repeat_m, spec.wave_repeat_n
            calls = call.findall(_llvm(spec))
            plain = call.findall(_llvm(replace(spec, wmma_reuse="none")))
            with self.subTest(spec.kernel_name()):
                self.assertEqual(len(calls), len(plain))
                self.assertEqual(len(calls) % (rm * rn), 0)
                self.assertTrue(all(c[2:] == ("false", "false") for c in plain))
                self.assertTrue(all(c[2] == "false" for c in calls))
                flagged = [c[3] == "true" for c in calls]
                self.assertEqual(sum(flagged), len(calls) // (rm * rn) * (rm - 1) * rn)
                for i, keep in enumerate(flagged):
                    if keep:
                        self.assertEqual(calls[i][1], calls[i + 1][1])
                for i in range(0, len(calls), rm):
                    self.assertEqual(flagged[i : i + rm], [True] * (rm - 1) + [False])

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
        ):
            with self.subTest(spec.kernel_name()):
                run_case(spec)


if __name__ == "__main__":
    unittest.main()
