# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Coverage for the flag-free gfx1250 MX FP4 GEMM (``mxfp4_gemm``)."""

from __future__ import annotations

import itertools
import re
import shutil
import unittest

from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.examples.gfx1250.gemm.mxfp4_gemm_bench import loop_scratch_ops, run_case
from rocke.examples.gfx1250.gemm.mxfp4_tdm_gemm import YAML_TILE
from rocke.helpers.compile import _comgr_options_for_kernel
from rocke.instances.gfx1250.mxfp4_gemm import (
    TILE_K,
    TILE_MN,
    WAVE_GROUPS,
    IntOps,
    LdsPad,
    MxFp4GemmSpec,
    build_mxfp4_gemm,
    derived_lds_pads,
    validate,
)
from rocke.instances.gfx1250.mxfp4_tdm_gemm import build_mxfp4_tdm_gemm
from rocke.instances.gfx1250.mxfp4_tdm_gemm_variants import variant_spec
from rocke.runtime.hip_module import get_device_arch

_BENCH = (4096, 4096, 65536)
_SMOKE = (1024, 1024, 2048)
_WMMA = r"call [^\n]*@llvm\.amdgcn\.wmma\.scale\."


def _spec(*problem, **fields) -> MxFp4GemmSpec:
    m, n, k = problem or _SMOKE
    return MxFp4GemmSpec(name="t", M=m, N=n, K=k, **fields)


def _llvm(spec: MxFp4GemmSpec) -> str:
    return _lower_kernel_to_llvm_python(
        build_mxfp4_gemm(spec), arch="gfx1250", llvm_flavor="llvm23"
    )


def _valid_geometries():
    for tm, tn, tk, wg in itertools.product(TILE_MN, TILE_MN, TILE_K, WAVE_GROUPS):
        spec = _spec(tile_m=tm, tile_n=tn, tile_k=tk, wave_group=wg)
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


class MxFp4GemmSpecTest(unittest.TestCase):
    def test_validator_accepts_tuning_space_and_rejects_bad_specs(self):
        self.assertIsNone(validate(_spec()))
        self.assertIsNone(validate(_spec(*_BENCH, cluster_order="aid")))
        self.assertIsNone(validate(_spec(cluster_dim=(1, 1, 1), out_dtype="bf16")))
        self.assertIsNone(validate(_spec(half_plr="none", local_prefetch=0)))
        self.assertEqual(len(list(_valid_geometries())), 20)
        bad = {
            "tile_m": {"tile_m": 64},
            "tile_k": {"tile_k": 128},
            "wave_group": {"wave_group": (1, 2)},
            "wave_repeat_n": {"tile_n": 128, "wave_group": (1, 4)},
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
        for label, fields in bad.items():
            problem = (1024, 1024, 512) if label == "short K" else _SMOKE
            with self.subTest(label):
                self.assertIsNotNone(validate(_spec(*problem, **fields)))

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


class MxFp4GemmLayoutTest(unittest.TestCase):
    def test_reads_land_where_tdm_writes(self):
        """Row and scale offsets agree with the TDM's padded writes, every
        lane read is contiguous, and the regions are disjoint and in bounds."""
        specs = list(_valid_geometries()) + [
            _spec(a_lds_pad=LdsPad(), b_lds_pad=LdsPad()),
            _spec(a_lds_pad=LdsPad(32, 2048), b_scale_lds_pad=LdsPad()),
            _spec(
                tile_m=256,
                tile_n=128,
                wave_group=(1, 4),
                a_scale_lds_pad=LdsPad(16, 512),
            ),
        ]
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
                    ("b_scale", 16),
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


class MxFp4GemmIrTest(unittest.TestCase):
    def test_benchmark_config_matches_step6_14(self):
        """Gate 1: byte-identical LLVM IR (kernel name aside) and the same
        compile options as Step 6.14 at the benchmark problem."""
        old = variant_spec("step6_14_mxfp4_gemm", *_BENCH, **YAML_TILE)
        new = _spec(*_BENCH, cluster_order="aid")
        old_kernel = build_mxfp4_tdm_gemm(old, arch="gfx1250")
        new_kernel = build_mxfp4_gemm(new)
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
            _comgr_options_for_kernel(new_kernel), _comgr_options_for_kernel(old_kernel)
        )

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


@unittest.skipIf(shutil.which("llvm-objdump") is None, "needs llvm-objdump")
class MxFp4GemmHostBuildTest(unittest.TestCase):
    def test_resources(self):
        """Host compiles: VGPR file and LDS limits, no scratch in the loop."""
        import tempfile
        from pathlib import Path

        from rocke.analysis.isa import analyze_hsaco
        from rocke.helpers.compile import compile_kernel

        for spec in (
            _spec(*_BENCH, cluster_order="aid"),
            _spec(tile_m=128, tile_n=256, tile_k=256, wave_group=(4, 1)),
            _spec(cluster_dim=(1, 1, 1), half_plr="none", out_dtype="fp16"),
        ):
            with self.subTest(spec.kernel_name()):
                artifact = compile_kernel(
                    build_mxfp4_gemm(spec), arch="gfx1250", backend="python"
                )
                with tempfile.NamedTemporaryFile(suffix=".hsaco") as handle:
                    handle.write(artifact.hsaco)
                    handle.flush()
                    info = analyze_hsaco(Path(handle.name), keep_text=True)
                self.assertLessEqual(info.resources.vgpr_count, 1024)
                self.assertEqual(info.resources.lds_bytes, spec.lds_bytes)
                self.assertEqual(loop_scratch_ops(info.objdump_text), 0)


class MxFp4GemmDeviceTest(unittest.TestCase):
    def test_matches_reference_on_gfx1250(self):
        if get_device_arch(0) != "gfx1250":
            self.skipTest("visible HIP device 0 is not gfx1250")
        for spec in list(_valid_geometries()) + [
            _spec(out_dtype="fp16"),
            _spec(out_dtype="bf16"),
            _spec(cluster_dim=(1, 1, 1)),
            _spec(half_plr="none"),
            _spec(half_plr="none", local_prefetch=0),
            _spec(cluster_signal="drained"),
            _spec(init_c_peel=False),
        ]:
            with self.subTest(spec.kernel_name()):
                run_case(spec)


if __name__ == "__main__":
    unittest.main()
