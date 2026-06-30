# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for rocke multi-arch support (Phase F gates).

Covers the polymorphic-core foundation (ArchTarget + MMA catalog + ISA backend),
the GEMM family policy, the gfx950 byte-identical guarantee, the hybrid-layout
back-compat shim, and the architectural isolation rules from
``dsl_docs/architecture/multi_arch_data_layout.md`` ("Review Rules").

Run:  PYTHONPATH=Python python3 tests/instances/test_rocke_multiarch.py
These tests need no GPU.
"""

from __future__ import annotations

import pathlib
import unittest

from rocke.core.arch import ArchTarget, known_arches, normalize_dtype
from rocke.core.isa import Gfx9MfmaBackend, Gfx950Backend, backend_for

_ROCKE_ROOT = pathlib.Path(__file__).resolve().parents[2] / "Python" / "rocke"


class TestArchTarget(unittest.TestCase):
    def test_known_arches(self):
        # CDNA MFMA (gfx90a CDNA2, gfx942/gfx950 CDNA3/4) + RDNA WMMA/wave32
        # (gfx1151 RDNA3.5, gfx1201 RDNA4, gfx11-generic) + gfx1250-class
        # CDNA/GFX12 WMMA.
        self.assertEqual(
            set(known_arches()),
            {
                "gfx90a",
                "gfx942",
                "gfx950",
                "gfx1151",
                "gfx1201",
                "gfx1250",
                "gfx11-generic",
            },
        )

    def test_gfx1250_facts(self):
        t = ArchTarget.from_gfx("gfx1250")
        self.assertEqual(t.wave_size, 32)
        self.assertEqual(t.family, "cdna")
        self.assertEqual(t.target_family, "gfx12_cdna")
        self.assertEqual(t.isa_triple, "amdgcn-amd-amdhsa--gfx1250")
        # WMMA catalog (no MFMA): primary fp16/bf16 atom is 16x16x32 (K=32),
        # distinct from gfx1201's 16x16x16.
        self.assertEqual(
            t.mma.enumerate(a_dtype="fp16", b_dtype="fp16", c_dtype="fp32"), []
        )
        wmma = t.mma.enumerate(
            family="wmma", a_dtype="fp16", b_dtype="fp16", c_dtype="fp32"
        )
        self.assertEqual([o.shape for o in wmma], [(16, 16, 32)])
        from rocke.core.isa import backend_for
        from rocke.core.isa.backend import Gfx1250Backend

        self.assertIsInstance(backend_for("gfx1250"), Gfx1250Backend)

    def test_gfx1151_rdna_facts(self):
        t = ArchTarget.from_gfx("gfx1151")
        self.assertEqual(t.wave_size, 32)
        self.assertEqual(t.family, "rdna")
        # WMMA catalog, not MFMA: no 'mma'-family atoms, has 'wmma' 16x16x16.
        self.assertEqual(
            t.mma.enumerate(a_dtype="fp16", b_dtype="fp16", c_dtype="fp32"), []
        )
        wmma = t.mma.enumerate(
            family="wmma", a_dtype="fp16", b_dtype="fp16", c_dtype="fp32"
        )
        self.assertEqual([o.shape for o in wmma], [(16, 16, 16)])
        from rocke.core.isa import backend_for, Gfx11RdnaBackend

        self.assertIsInstance(backend_for("gfx1151"), Gfx11RdnaBackend)

    def test_unknown_arch_raises(self):
        with self.assertRaises(KeyError):
            ArchTarget.from_gfx("gfx999")

    def test_facts(self):
        t942 = ArchTarget.from_gfx("gfx942")
        t950 = ArchTarget.from_gfx("gfx950")
        self.assertEqual(t942.isa_triple, "amdgcn-amd-amdhsa--gfx942")
        self.assertEqual(t942.wave_size, 64)
        self.assertEqual(t942.lds_capacity_bytes, 65536)
        self.assertEqual(t950.lds_capacity_bytes, 163840)
        self.assertEqual(t942.vmcnt_bits, 4)
        self.assertEqual(t950.vmcnt_bits, 6)
        self.assertFalse(t942.memory.has_ds_read_tr)
        self.assertTrue(t950.memory.has_ds_read_tr)

    def test_predicates(self):
        t942 = ArchTarget.from_gfx("gfx942")
        self.assertTrue(t942.fits_lds(65536))
        self.assertFalse(t942.fits_lds(65537))
        self.assertTrue(t942.supports_dtype_combo("fp16", "fp16", "fp32"))
        # gfx942 (CDNA3) ships fp8/bf8 MFMA, so the fp8 dtype combo IS
        # supported on it. The fp4/MX MFMA family is the gfx950-only
        # combo, so use it for the negative case.
        self.assertTrue(t942.supports_dtype_combo("fp8", "fp8", "fp32"))
        self.assertFalse(t942.supports_dtype_combo("fp4", "fp4", "fp32"))
        self.assertTrue(
            ArchTarget.from_gfx("gfx950").supports_dtype_combo("fp4", "fp4", "fp32")
        )
        self.assertEqual(t942.max_threads_per_block, 1024)

    def test_dtype_normalization(self):
        self.assertEqual(normalize_dtype("f16"), "fp16")
        self.assertEqual(normalize_dtype("half"), "fp16")
        self.assertEqual(normalize_dtype("bf16"), "bf16")
        self.assertEqual(normalize_dtype("f32"), "fp32")


class TestMmaCatalog(unittest.TestCase):
    def test_f16_atom_divergence(self):
        f = lambda gfx: sorted(  # noqa: E731
            op.shape
            for op in ArchTarget.from_gfx(gfx).mma.enumerate(
                a_dtype="fp16", b_dtype="fp16", c_dtype="fp32"
            )
        )
        # gfx942 narrow CDNA3 atoms only; gfx950 adds the wide CDNA4 atoms.
        self.assertEqual(f("gfx942"), [(16, 16, 16), (32, 32, 8)])
        self.assertEqual(
            f("gfx950"), [(16, 16, 16), (16, 16, 32), (32, 32, 8), (32, 32, 16)]
        )

    def test_wide_atom_absent_on_gfx942(self):
        t = ArchTarget.from_gfx("gfx942")
        self.assertFalse(
            t.mma.has_shape(
                a_dtype="fp16", b_dtype="fp16", c_dtype="fp32", m=16, n=16, k=32
            )
        )
        self.assertTrue(
            ArchTarget.from_gfx("gfx950").mma.has_shape(
                a_dtype="fp16", b_dtype="fp16", c_dtype="fp32", m=16, n=16, k=32
            )
        )

    def test_select_largest_k(self):
        op942 = ArchTarget.from_gfx("gfx942").mma.select_largest_k(
            a_dtype="fp16", b_dtype="fp16", c_dtype="fp32", m=16, n=16
        )
        op950 = ArchTarget.from_gfx("gfx950").mma.select_largest_k(
            a_dtype="fp16", b_dtype="fp16", c_dtype="fp32", m=16, n=16
        )
        self.assertEqual(op942.k, 16)  # gfx942 max is 16x16x16
        self.assertEqual(op950.k, 32)  # gfx950 has the wide 16x16x32
        self.assertEqual(op950.op_id, "mfma_f32_16x16x32_f16")

    def test_select_returns_none_for_unsupported(self):
        t = ArchTarget.from_gfx("gfx942")
        # fp8 IS on gfx942 now (16x16x32 / 32x32x16); fp4/MX is the
        # genuinely gfx950-only family, so it must select to None on gfx942.
        self.assertIsNone(
            t.mma.select_largest_k(
                a_dtype="fp4", b_dtype="fp4", c_dtype="fp32", m=16, n=16
            )
        )
        # ...and resolve to the fp8 hero atom on gfx942 (it's shipped).
        op_fp8 = t.mma.select_largest_k(
            a_dtype="fp8", b_dtype="fp8", c_dtype="fp32", m=16, n=16
        )
        self.assertIsNotNone(op_fp8)
        self.assertEqual(op_fp8.op_id, "mfma_f32_16x16x32_fp8")


class TestISABackend(unittest.TestCase):
    def test_backend_selection(self):
        self.assertIsInstance(backend_for("gfx942"), Gfx9MfmaBackend)
        self.assertIsInstance(backend_for("gfx950"), Gfx950Backend)

    def test_unknown_backend_raises(self):
        with self.assertRaises(KeyError):
            backend_for("gfx999")

    def test_preamble_and_waitcnt_shared_for_cdna(self):
        from rocke.core.lower_llvm import LLVM_FLAVOR_LLVM22

        b942, b950 = backend_for("gfx942"), backend_for("gfx950")
        # datalayout/triple/waitcnt are hardware-verified identical across these
        # CDNA targets today (see multi_arch_data_layout.md "ISA Backend").
        # datalayout is LLVM-keyed (not gfx-keyed), so compare under the same flavor.
        self.assertEqual(
            b942.module_preamble(LLVM_FLAVOR_LLVM22),
            b950.module_preamble(LLVM_FLAVOR_LLVM22),
        )
        self.assertIn("target datalayout", b942.module_preamble(LLVM_FLAVOR_LLVM22))
        self.assertIn("amdgcn-amd-amdhsa", b942.triple)
        self.assertEqual(b942.encode_waitcnt(0, -1, 0), b950.encode_waitcnt(0, -1, 0))


class TestGfx950ByteIdentical(unittest.TestCase):
    def _kernel(self):
        from rocke.instances.common.gemm_universal import (
            UniversalGemmSpec,
            TileSpec,
            TraitSpec,
            DataSpec,
            build_universal_gemm,
        )

        spec = UniversalGemmSpec(
            name="byteid",
            tile=TileSpec(
                tile_m=128,
                tile_n=128,
                tile_k=32,
                warp_m=2,
                warp_n=2,
                warp_k=1,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=16,
            ),
            trait=TraitSpec(
                pipeline="compv4", scheduler="intrawave", epilogue="cshuffle"
            ),
            data=DataSpec(dtype_a="fp16", dtype_b="fp16", dtype_c="fp16"),
        )
        return build_universal_gemm(spec)

    def test_legacy_equals_gfx950(self):
        from rocke.core.lower_llvm import lower_kernel_to_llvm

        k = self._kernel()
        self.assertEqual(
            lower_kernel_to_llvm(k), lower_kernel_to_llvm(k, arch="gfx950")
        )

    def test_arch_param_default_is_cdna_byte_identical(self):
        # The new ``arch`` parameter on build_universal_gemm must default to the
        # CDNA behaviour: building with arch="gfx950" (and the gfx942-legal
        # 16x16x16 atom for gfx942) must produce byte-identical LLVM to the
        # historical no-arg build, across the (pipeline, epilogue) matrix.
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.instances.common.gemm_universal import (
            UniversalGemmSpec,
            TileSpec,
            TraitSpec,
            DataSpec,
            build_universal_gemm,
        )

        for pl in ("mem", "compv3", "compv4"):
            for ep in ("default", "cshuffle"):
                spec = UniversalGemmSpec(
                    name="archid",
                    tile=TileSpec(
                        tile_m=128,
                        tile_n=128,
                        tile_k=32,
                        warp_m=2,
                        warp_n=2,
                        warp_k=1,
                        warp_tile_m=16,
                        warp_tile_n=16,
                        warp_tile_k=16,
                    ),
                    trait=TraitSpec(pipeline=pl, scheduler="intrawave", epilogue=ep),
                    data=DataSpec(dtype_a="fp16", dtype_b="fp16", dtype_c="fp16"),
                )
                legacy = lower_kernel_to_llvm(build_universal_gemm(spec), arch="gfx950")
                explicit = lower_kernel_to_llvm(
                    build_universal_gemm(spec, arch="gfx950"), arch="gfx950"
                )
                self.assertEqual(legacy, explicit, f"{pl}/{ep} drifted on gfx950")

    def test_emit_sched_hints_arch_resolution(self):
        # emit_sched_hints=None (default) is arch-resolved: gfx950 OMITS the
        # compv4 sched_group_barrier/sched_barrier schedule (the measured +2%
        # uplift; see optimization/utilities/skills/empirical-case-studies.md
        # Case Study 7) while gfx942 keeps the historical emission. Explicit
        # True/False overrides the arch default on either arch.
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.instances.common.gemm_universal import (
            UniversalGemmSpec,
            TileSpec,
            TraitSpec,
            DataSpec,
            build_universal_gemm,
        )

        def n_hints(arch, hints):
            spec = UniversalGemmSpec(
                name="sched",
                tile=TileSpec(
                    tile_m=128,
                    tile_n=128,
                    tile_k=32,
                    warp_m=2,
                    warp_n=2,
                    warp_k=1,
                    warp_tile_m=16,
                    warp_tile_n=16,
                    warp_tile_k=16,
                ),
                trait=TraitSpec(
                    pipeline="compv4",
                    scheduler="intrawave",
                    epilogue="default",
                    emit_sched_hints=hints,
                ),
                data=DataSpec(dtype_a="fp16", dtype_b="fp16", dtype_c="fp16"),
            )
            ir = lower_kernel_to_llvm(build_universal_gemm(spec, arch=arch), arch=arch)
            return ir.count("sched.group.barrier") + ir.count("sched.barrier")

        # gfx950 default takes the uplift (no schedule hints emitted).
        self.assertEqual(n_hints("gfx950", None), 0)
        # gfx942 default preserves the historical compv4 schedule.
        self.assertGreater(n_hints("gfx942", None), 0)
        # Explicit override forces the choice regardless of the arch default.
        self.assertGreater(n_hints("gfx950", True), 0)
        self.assertEqual(n_hints("gfx950", False), 0)
        self.assertEqual(n_hints("gfx942", False), 0)


class TestUnifiedGfx1151Gemm(unittest.TestCase):
    """The MMA-contract unification: the SAME GEMM body emits WMMA on gfx1151."""

    def _spec(self, pipeline="mem", epilogue="default"):
        from rocke.instances.common.gemm_universal import (
            UniversalGemmSpec,
            TileSpec,
            TraitSpec,
            DataSpec,
        )

        return UniversalGemmSpec(
            name="ugemm1151",
            tile=TileSpec(
                tile_m=32,
                tile_n=32,
                tile_k=16,
                warp_m=2,
                warp_n=2,
                warp_k=1,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=16,
            ),
            trait=TraitSpec(
                pipeline=pipeline,
                scheduler="intrawave",
                epilogue=epilogue,
                pad_m=True,
                pad_n=True,
                pad_k=True,
            ),
            data=DataSpec(dtype_a="fp16", dtype_b="fp16", dtype_c="fp16"),
            wave_size=32,
        )

    def test_unified_body_emits_wmma(self):
        # build_universal_gemm(spec, arch="gfx1151") must produce a wave32 WMMA
        # kernel (the same builder that emits MFMA on CDNA).
        from rocke.core.lower_llvm import lower_kernel_to_llvm
        from rocke.instances.common.gemm_universal import build_universal_gemm

        ir = lower_kernel_to_llvm(
            build_universal_gemm(self._spec(), arch="gfx1151"), arch="gfx1151"
        )
        self.assertIn("wmma.f32.16x16x16.f16", ir)
        self.assertIn("<16 x half>", ir)  # WMMA operand fragment
        self.assertIn("<8 x float>", ir)  # WMMA accumulator fragment

    def test_unified_body_rejects_unsupported_wmma_configs(self):
        # The WMMA path is gated to mem + default + 16x16x16; richer configs
        # must reject cleanly (no silent wrong-ISA emission).
        from rocke.instances.common.gemm_universal import is_valid_spec

        ok, why = is_valid_spec(self._spec(pipeline="compv4"), arch="gfx1151")
        self.assertFalse(ok)
        self.assertIn("WMMA", why)
        ok, why = is_valid_spec(self._spec(epilogue="cshuffle"), arch="gfx1151")
        self.assertFalse(ok)
        self.assertIn("WMMA", why)

    def test_wave64_spec_rejected_on_gfx1151(self):
        # A wave64 spec must not build on the wave32 target.
        from rocke.instances.common.gemm_universal import (
            UniversalGemmSpec,
            TileSpec,
            TraitSpec,
            DataSpec,
            is_valid_spec,
        )

        spec = UniversalGemmSpec(
            name="w64",
            wave_size=64,
            tile=TileSpec(
                tile_m=32,
                tile_n=32,
                tile_k=16,
                warp_m=2,
                warp_n=2,
                warp_k=1,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=16,
            ),
            trait=TraitSpec(pipeline="mem", epilogue="default"),
            data=DataSpec(dtype_a="fp16", dtype_b="fp16", dtype_c="fp16"),
        )
        ok, why = is_valid_spec(spec, arch="gfx1151")
        self.assertFalse(ok)
        self.assertIn("wave_size", why)


class TestGemmPolicy(unittest.TestCase):
    def _spec(self, wtm, wtn, wtk, dt="fp16"):
        from rocke.instances.common.gemm_universal import (
            UniversalGemmSpec,
            TileSpec,
            TraitSpec,
            DataSpec,
        )

        return UniversalGemmSpec(
            name="p",
            tile=TileSpec(
                tile_m=4 * wtm,
                tile_n=4 * wtn,
                tile_k=max(32, wtk),
                warp_m=2,
                warp_n=2,
                warp_k=1,
                warp_tile_m=wtm,
                warp_tile_n=wtn,
                warp_tile_k=wtk,
            ),
            trait=TraitSpec(pipeline="mem", epilogue="default"),
            data=DataSpec(dtype_a=dt, dtype_b=dt, dtype_c=dt),
        )

    def test_arch_aware_validate(self):
        from rocke.instances.common.gemm_policy import GemmPipelinePolicy

        pol = GemmPipelinePolicy()
        t942, t950 = ArchTarget.from_gfx("gfx942"), ArchTarget.from_gfx("gfx950")
        # 16x16x16 legal on both; the wide 16x16x32 only on gfx950.
        self.assertTrue(pol.validate(t942, self._spec(16, 16, 16)).ok)
        self.assertTrue(pol.validate(t950, self._spec(16, 16, 16)).ok)
        self.assertFalse(pol.validate(t942, self._spec(16, 16, 32)).ok)
        self.assertTrue(pol.validate(t950, self._spec(16, 16, 32)).ok)

    def test_valid_warp_tiles_from_catalog(self):
        from rocke.instances.common.gemm_policy import GemmPipelinePolicy

        pol = GemmPipelinePolicy()
        self.assertEqual(
            pol.valid_warp_tiles(
                ArchTarget.from_gfx("gfx942"), ("fp16", "fp16", "fp32")
            ),
            [(16, 16, 16), (32, 32, 8)],
        )


class TestHybridLayoutShim(unittest.TestCase):
    def test_canonical_layout_public_api(self):
        # The back-compat shims at the old flat paths have been removed; the
        # canonical hybrid-layout module is the single source of truth and the
        # public package API re-exports from it.
        from rocke.instances import build_universal_gemm as public_build
        from rocke.instances.common import gemm_universal as impl

        self.assertIs(public_build, impl.build_universal_gemm)
        # private helpers still live on the canonical module (sibling instances
        # depend on them)
        for name in ("_emit_mfma", "_choose_load_vec", "_storage_dtype"):
            self.assertTrue(hasattr(impl, name))
        # the old flat shim path must be gone
        import importlib

        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("rocke.instances.gemm_universal")


class TestDeviceArchAndFusionTargeting(unittest.TestCase):
    """The launch-path fix: device-arch detection + arch-aware fusion atoms."""

    def test_get_device_arch_importable(self):
        from rocke.runtime.hip_module import get_device_arch

        v = get_device_arch()  # None off-GPU, "gfxNNN" on a device — never raises
        self.assertTrue(v is None or (isinstance(v, str) and v.startswith("gfx")))

    def test_fusion_gemm_atoms_are_catalog_legal_per_arch(self):
        # The GemmEpilogueLowerer must pick warp-tile K from the target's MMA
        # catalog so it never emits a gfx950-only atom (e.g. 32x32x16 f16) on
        # gfx942. Drive candidates() with the device resolved to each arch and
        # assert every emitted spec validates on that arch.
        import unittest.mock as mock
        from rocke.helpers import (
            FusionOp,
            FusionTensor,
            GemmEpilogueLowerer,
            GreedyFusionScheduler,
            build_graph,
        )
        from rocke.instances.common.gemm_universal import is_valid_spec

        graph = build_graph(
            tensors=[
                FusionTensor("A", (256, 256), "fp16", is_input=True),
                FusionTensor("B", (256, 256), "fp16", is_input=True),
                FusionTensor("mm", (256, 256), "fp16", is_output=True),
            ],
            ops=[FusionOp("mm0", "matmul", ("A", "B"), ("mm",))],
            inputs=("A", "B"),
            outputs=("mm",),
        )
        region = GreedyFusionScheduler().schedule(graph).regions[0]
        low = GemmEpilogueLowerer()
        for gfx in ("gfx942", "gfx950"):
            with mock.patch(
                "rocke.runtime.hip_module.get_device_arch", return_value=gfx
            ):
                cfgs = low.candidates(graph, region)
            self.assertTrue(cfgs, f"no candidates for {gfx}")
            for c in cfgs:
                ok, why = is_valid_spec(c.spec, arch=gfx)
                self.assertTrue(ok, f"{gfx}: illegal spec emitted: {why}")
            # On gfx942 none may use the gfx950-only wide f16 atoms.
            if gfx == "gfx942":
                for c in cfgs:
                    self.assertNotIn(
                        (
                            c.spec.tile.warp_tile_m,
                            c.spec.tile.warp_tile_n,
                            c.spec.tile.warp_tile_k,
                        ),
                        {(16, 16, 32), (32, 32, 16)},
                    )


class TestArchitecturalIsolation(unittest.TestCase):
    """Review-rule gates from the design doc."""

    def _read(self, rel):
        return (_ROCKE_ROOT / rel).read_text()

    def test_core_arch_no_dispatcher_import(self):
        # Catch real imports of dispatcher, not the word in docstrings/comments.
        import re

        pat = re.compile(r"^\s*(from|import)\s+\S*dispatcher", re.MULTILINE)
        for p in (_ROCKE_ROOT / "core" / "arch").rglob("*.py"):
            self.assertIsNone(
                pat.search(p.read_text()), f"{p} must not import from dispatcher/"
            )

    def test_core_arch_no_pipeline_vocabulary(self):
        # Pipeline/scheduler names are instance-side policy, never in core/arch.
        blob = "\n".join(
            p.read_text() for p in (_ROCKE_ROOT / "core" / "arch").rglob("*.py")
        )
        for tok in ("compv4", "compv3", "intrawave", "interwave", "qr_ks_vs"):
            self.assertNotIn(tok, blob, f"pipeline token {tok!r} leaked into core/arch")

    def test_core_arch_no_llvm_intrinsic_text(self):
        blob = "\n".join(
            p.read_text() for p in (_ROCKE_ROOT / "core" / "arch").rglob("*.py")
        )
        self.assertNotIn(
            "llvm.amdgcn", blob, "core/arch must not contain intrinsic text"
        )


class TestDatalayoutDriftGuard(unittest.TestCase):
    """Drift guard: assert rocke's hardcoded datalayout matches the toolchain.

    The LLVM IR datalayout is LLVM-version-keyed (not gfx-keyed): ROCm 7.2
    ships ``p8:128:128:128:48``, while 7.0/7.1 shipped ``p8:128:128``. The
    difference is auto-upgraded away when compiling textual IR through comgr,
    so a wrong-but-well-formed hardcoded string compiles fine — but relying
    on that parser leniency is fragile across ingestion paths.

    This test re-derives the ground truth from the project's own ``hipcc``:
    it lowers a tiny kernel to HIP C++, runs ``hipcc --offload-arch=<arch>
    -S -emit-llvm --cuda-device-only`` (the same toolchain that compiles
    real kernels), extracts the ``target datalayout`` line, and asserts
    rocke's flavor-keyed constant matches byte-for-byte.

    When ROCm bumps the canonical datalayout (or a new drift item appears),
    this fails with a clear "regenerate the constant" message on any dev/CI
    box with hipcc, instead of silently leaning on comgr's override.

    Skipped cleanly when hipcc is absent (e.g., pure-Python test envs).
    """

    def _find_hipcc(self) -> bool:
        """Return whether hipcc is in PATH."""
        import shutil

        return shutil.which("hipcc") is not None

    def _extract_datalayout_from_ir(self, ir_text: str) -> str:
        """Extract 'target datalayout = "..."' from LLVM IR text."""
        import re

        for line in ir_text.splitlines():
            m = re.match(r'^target datalayout = "([^"]+)"', line)
            if m:
                return m.group(1)
        raise ValueError("no target datalayout line found in IR")

    def test_datalayout_matches_hipcc_emitted_ir(self):
        """Assert rocke's hardcoded datalayout matches hipcc's emitted IR."""
        if not self._find_hipcc():
            self.skipTest("hipcc not in PATH")

        from rocke.core.ir import F32, KernelDef, Param, PtrType, Region
        from rocke.core.isa.backend import wired_arches
        from rocke.core.lower_llvm import (
            _datalayout_for_flavor,
            _detect_llvm_flavor,
            _flavor_for_rocm,
            _system_rocm_version,
        )
        from rocke.helpers.compile import emit_device_llvm_ir_via_hipcc

        # Build a minimal kernel (any wired arch works; datalayout is gfx-invariant).
        kernel = KernelDef(
            name="drift_guard",
            params=[Param("p", PtrType(F32, "global"))],
            body=Region([]),
            attrs={},
        )

        # Validate the constant for the flavor of the REFERENCE toolchain -- the
        # `hipcc` on PATH (the system /opt/rocm) that emits the IR below. That can
        # be a DIFFERENT LLVM vintage than the comgr lib `_detect_llvm_flavor()`
        # resolves at runtime (e.g. a torch-bundled comgr 7.2/llvm22 alongside a
        # system hipcc 7.0/llvm20). We can only statically validate the constant
        # whose toolchain is actually present; the comgr flavor is exercised
        # dynamically by every GPU compile (a wrong datalayout aborts codegen).
        sys_ver = _system_rocm_version()
        detected_flavor = (
            _flavor_for_rocm(*sys_ver) if sys_ver else _detect_llvm_flavor()
        )
        rocke_dl = _datalayout_for_flavor(detected_flavor)

        # Test across all wired arches to confirm datalayout really is gfx-invariant
        # (the assumption the flavor split rests on).
        for arch in wired_arches():
            with self.subTest(arch=arch):
                try:
                    ir = emit_device_llvm_ir_via_hipcc(kernel, arch=arch, timeout_s=60)
                except Exception as e:
                    # hipcc on this box may not support every wired arch (e.g. a
                    # newer GFX12 target like gfx1250 on an older ROCm toolchain).
                    # That is a toolchain limitation, NOT a datalayout drift, so
                    # skip this arch rather than fail -- the other arches still
                    # prove the gfx-invariant datalayout the flavor split needs.
                    self.skipTest(f"hipcc cannot target {arch} on this toolchain: {e}")
                toolchain_dl = self._extract_datalayout_from_ir(ir)
                self.assertEqual(
                    rocke_dl,
                    toolchain_dl,
                    f"Datalayout drift detected for {arch} under {detected_flavor}.\n"
                    f"  rocke constant: {rocke_dl}\n"
                    f"  hipcc emitted:   {toolchain_dl}\n"
                    f"Regenerate _DATALAYOUT_{detected_flavor.upper().replace('llvm', '')} "
                    f"in core/lower_llvm.py by running:\n"
                    f'  echo "" | /opt/rocm/llvm/bin/clang -target amdgcn-amd-amdhsa '
                    f"-mcpu={arch} -emit-llvm -S - -o - | grep 'target datalayout'",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
