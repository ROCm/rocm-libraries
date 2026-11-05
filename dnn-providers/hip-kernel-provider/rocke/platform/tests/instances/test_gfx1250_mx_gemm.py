# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Coverage for the flag-free gfx1250 MX GEMM (``mx_gemm``)."""

from __future__ import annotations

import itertools
import re
import shutil
import unittest

import numpy as np
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.examples.common.scaled_gemm_reference import decode_e8m0
from rocke.examples.gfx1250.gemm.mx_gemm_bench import (
    _E2M1,
    SPARSE_ALPHA,
    SPARSE_STEPS,
    check_inputs,
    device_inputs,
    loop_scratch_ops,
    reference,
    run_case,
    sparse_inputs,
    sparse_steps,
    verify_block,
)
from rocke.examples.gfx1250.gemm.mxfp4_tdm_gemm import YAML_TILE
from rocke.helpers.compile import _comgr_options_for_kernel
from rocke.instances.gfx1250.mx_gemm import (
    OUT_DTYPES,
    TILE_K,
    TILE_MN,
    WAVE_GROUPS,
    IntOps,
    LdsPad,
    MxGemmSpec,
    build_mx_gemm,
    derived_lds_pads,
    validate,
)
from rocke.instances.gfx1250.mxfp4_tdm_gemm import build_mxfp4_tdm_gemm
from rocke.instances.gfx1250.mxfp4_tdm_gemm_variants import variant_spec
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


def _valid_geometries(mma: str = "32x16"):
    for tm, tn, tk, wg in itertools.product(TILE_MN, TILE_MN, TILE_K, WAVE_GROUPS):
        spec = _spec(tile_m=tm, tile_n=tn, tile_k=tk, wave_group=wg, mma=mma)
        if validate(spec) is None:
            yield spec


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
        self.assertIsNone(validate(_spec(*_BENCH, cluster_order="aid")))
        self.assertIsNone(validate(_spec(cluster_dim=(1, 1, 1), out_dtype="bf16")))
        self.assertIsNone(validate(_spec(half_plr="none", local_prefetch=0)))
        self.assertEqual(len(list(_valid_geometries())), 75)
        self.assertEqual(len(list(_valid_geometries("16x16"))), 102)
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
            "signal range": {"cluster_signal_wmmas": 17},
            "signal form": {"cluster_signal": "early"},
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
        self.assertEqual(spec.signal_form, "wave0")
        self.assertEqual(spec.signal_wmmas, 6)
        self.assertEqual(spec.fp8_convert_width, 8)
        self.assertEqual(spec.reads_per_step, 35)
        small = _spec(tile_m=128, tile_n=128)
        self.assertEqual(small.signal_form, "drained")
        self.assertEqual(small.signal_wmmas, 4)
        self.assertEqual(small.fp8_convert_width, 4)
        self.assertIsNone(_spec(cluster_dim=(1, 1, 1)).signal_form)

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

    def test_kernel_name_encodes_non_default_tuning(self):
        base = _spec().kernel_name()
        self.assertTrue(base.endswith("_t256x256x512_c4x4"), base)
        tuned = _spec(cluster_dim=(2, 2, 1), gl2_prefetch_distance=0, init_c_peel=False)
        self.assertIn("_c2x2_pf0", tuned.kernel_name())
        self.assertTrue(tuned.kernel_name().endswith("_noinitc"))


class MxGemmLayoutTest(unittest.TestCase):
    def test_reads_land_where_tdm_writes(self):
        """Row and scale offsets agree with the TDM's padded writes, every
        lane read is contiguous, and the regions are disjoint and in bounds."""
        specs = (
            list(_valid_geometries())
            + list(_valid_geometries("16x16"))
            + [
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


class MxGemmIrTest(unittest.TestCase):
    def test_benchmark_config_matches_step6_14(self):
        """Gate 1: byte-identical LLVM IR (kernel name aside) at the benchmark
        problem, and Step 6.14's compile options plus the regclass-priority
        allocator option."""
        old = variant_spec("step6_14_mxfp4_gemm", *_BENCH, **YAML_TILE)
        new = _spec(*_BENCH, cluster_order="aid")
        old_kernel = build_mxfp4_tdm_gemm(old, arch="gfx1250")
        new_kernel = build_mx_gemm(new)
        old_llvm = _lower_kernel_to_llvm_python(
            old_kernel, arch="gfx1250", llvm_flavor="llvm23"
        )
        new_llvm = _lower_kernel_to_llvm_python(
            new_kernel, arch="gfx1250", llvm_flavor="llvm23"
        )
        self.assertEqual(
            new_llvm.replace(new.kernel_name(), "KERNEL"),
            old_llvm.replace(old.kernel_name(), "KERNEL"),
        )
        self.assertEqual(
            _comgr_options_for_kernel(new_kernel),
            _comgr_options_for_kernel(old_kernel)
            + ["-mllvm", "-greedy-regclass-priority-trumps-globalness"],
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
            cluster_order="aid",
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
        """The init-C peel, the odd-count peel and the two-tile loop body are
        emitted once each; the loop covers the rest."""
        for tiles, init_c in itertools.product((4, 5), (True, False)):
            spec = _spec(1024, 1024, tiles * 512, init_c_peel=init_c)
            emitted = int(init_c) + (tiles - int(init_c)) % 2 + 2
            with self.subTest(tiles=tiles, init_c_peel=init_c):
                llvm = _llvm(spec)
                self.assertEqual(len(re.findall(_WMMA, llvm)), emitted * 4 * 32)
                # The first K-step of the init-C peel (or, without it, of an
                # odd-count peel) reads the zero vector directly.
                zero_c = len(re.findall(_WMMA + r"[^\n]*<16 x float> %cz", llvm))
                self.assertEqual(zero_c, 32 if init_c or tiles % 2 else 0)

    def test_k_tail_tile(self):
        """K not a multiple of tile_k: the full tiles as before, then one
        tile of the remaining K-steps (loaded clipped at K)."""
        for tail_steps, init_c in itertools.product((1, 2, 3), (True, False)):
            spec = _spec(1024, 1024, 4 * 512 + tail_steps * 128, init_c_peel=init_c)
            self.assertEqual(spec.tail_steps, tail_steps)
            emitted = int(init_c) + (4 - int(init_c)) % 2 + 2
            with self.subTest(tail_steps=tail_steps, init_c_peel=init_c):
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
                emitted = 1 + 3 % 2 + 2
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
        emitted (it runs zero times) after the init-C and odd-count peels."""
        for k_dim, tile_k, init_c in itertools.product(
            (128, 256, 384, 512, 640), (128, 256, 512), (True, False)
        ):
            spec = _spec(
                1024,
                1024,
                k_dim,
                tile_m=128,
                tile_n=128,
                tile_k=tile_k,
                init_c_peel=init_c,
            )
            if validate(spec) is not None:
                self.assertLess(k_dim, tile_k)
                continue
            full = k_dim // tile_k
            if full > 2:
                continue
            with self.subTest(K=k_dim, tile_k=tile_k, init_c_peel=init_c):
                llvm = _llvm(spec)
                per_step = spec.wave_repeat_m * spec.wave_repeat_n
                tiles = int(init_c) + (full - int(init_c)) % 2 + 2
                steps = tiles * spec.k_steps + spec.tail_steps
                self.assertEqual(len(re.findall(_WMMA, llvm)), steps * per_step)

    def test_one_step_tiles_signal_the_cluster(self):
        """tile_k 128 has no second-to-last K-step: each tile still signals
        once, before its cluster wait."""
        for fields in (
            {"cluster_signal": "wave0"},
            {"cluster_signal": "wave0", "half_plr": "none"},
            {"cluster_signal": "drained"},
            {"cluster_signal": "drained", "half_plr": "none", "local_prefetch": 0},
        ):
            spec = _spec(tile_k=128, **fields)
            with self.subTest(**fields):
                llvm = _llvm(spec)
                self.assertEqual(len(re.findall(_WMMA, llvm)), 4 * 32)
                if fields["cluster_signal"] == "drained":
                    self.assertEqual(llvm.count("s_barrier_signal_isfirst"), 2 + 4)
                else:
                    self.assertEqual(llvm.count("s_cmp_eq_u32 $0, 0"), 4)
                self.assertEqual(llvm.count("s_barrier_wait -3"), 2 + 4)

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
        """Four tiles are emitted at the smoke shape (see the peel test), four
        K-steps each; every tile adds one wait before its refill TDM."""
        default = _llvm(_spec())
        self.assertEqual(default.count('"s_wait_dscnt 35"'), 4 * (4 * 2 + 1))
        self.assertEqual(default.count("s_cmp_eq_u32 $0, 0"), 4)
        none = _llvm(_spec(half_plr="none"))
        self.assertEqual(none.count('"s_wait_dscnt 35"'), 4 * (4 + 1))
        direct = _llvm(_spec(half_plr="none", local_prefetch=0))
        self.assertEqual(direct.count('"s_wait_dscnt 0"'), 4 * (4 + 1))
        # The whole cluster barrier opens and closes the kernel; the drained
        # signal adds its first half once per tile.
        self.assertEqual(default.count("s_barrier_signal_isfirst"), 2)
        drained = _llvm(_spec(cluster_signal="drained"))
        self.assertNotIn("s_cmp_eq_u32 $0, 0", drained)
        self.assertEqual(drained.count("s_barrier_signal_isfirst"), 2 + 4)
        no_cluster = _llvm(_spec(cluster_dim=(1, 1, 1)))
        self.assertNotIn("s_barrier_signal -3", no_cluster)
        self.assertEqual(no_cluster.count('"s_wait_dscnt 35"'), 4 * (4 * 2 + 1))
        no_prefetch = _llvm(_spec(gl2_prefetch_distance=0))
        self.assertNotIn("global.prefetch", no_prefetch)
        # Six prefetches (tiles 2 and 3 up front, one per emitted tile) over
        # 2048 lanes of the 4x4 cluster: 256 B rows give 2176 lines (two
        # rounds) per tile. 128 B rows give 2112 lines for an even tile and
        # only the 64 scale lines (one round) for an odd one.
        prefetch = "call void @llvm.amdgcn.global.prefetch"
        self.assertEqual(default.count(prefetch), 6 * 2)
        self.assertEqual(_llvm(_spec(tile_k=256)).count(prefetch), 3 * (2 + 1))

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


class MxGemmSparseCheckTest(unittest.TestCase):
    """All E2M1 values in a few K-steps: exact FP32 sums at any K."""

    def _values(self, spec, data):
        """Float64 A and B with their scales applied."""
        _, _, a_scale, b_scale, _, a_codes, b_codes = data
        e2m1 = _E2M1.astype(np.float64)
        x = e2m1[a_codes] * np.repeat(decode_e8m0(a_scale), spec.block_k, axis=1)
        y = e2m1[b_codes] * np.repeat(decode_e8m0(b_scale).T, spec.block_k, axis=1)
        return x, y

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

    def test_fp32_sums_are_exact_in_any_order(self):
        self.assertLess(SPARSE_STEPS * 128 * 144 * 2**6, 2**24)
        spec = _spec(64, 48, 2048 + 96, tile_m=128, tile_n=128)
        x, y = self._values(spec, sparse_inputs(spec))
        exact = x @ y.T
        for order in (1, -1):
            total = np.zeros(exact.shape, dtype=np.float32)
            for j in range(spec.K)[::order]:
                total += np.outer(x[:, j], y[:, j]).astype(np.float32)
            with self.subTest(order=order):
                self.assertTrue(np.array_equal(total, exact))

    def test_reference_is_the_exact_result_rounded(self):
        for out_dtype, beta in itertools.product(OUT_DTYPES, (0.0, 0.5)):
            spec = _spec(64, 48, 2048 + 96, tile_m=128, tile_n=128, out_dtype=out_dtype)
            data = sparse_inputs(spec)
            x, y = self._values(spec, data)
            alpha = np.float32(SPARSE_ALPHA)
            value = (alpha * (x @ y.T)).astype(np.float32) + np.float32(beta) * data[
                4
            ].astype(np.float32)
            if out_dtype == "fp8e4m3":
                value = np.clip(value, -448, 448)
            expected = value.astype(reference(spec, data, alpha=alpha, beta=beta).dtype)
            with self.subTest(out_dtype=out_dtype, beta=beta):
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
            _spec(*_BENCH, cluster_order="aid"),
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
                cluster_order="aid",
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
                _spec(cluster_signal="drained"),
                _spec(init_c_peel=False),
            ]
        ):
            with self.subTest(spec.kernel_name()):
                run_case(spec)


if __name__ == "__main__":
    unittest.main()
