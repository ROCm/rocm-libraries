# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""No-GPU tests for the gfx1201 WMMA K-loop traits.

``lds_store_after_mma`` (LDS stores after the WMMAs in the compv4 double
buffer), ``group_m`` (GROUP_M raster) and ``k_stagger`` / ``k_stagger_step``
(per-workgroup K start offset). Covered here: the gates (each trait is inert
outside its loop, so admitting it there would rename a kernel without changing
it), default-off byte identity, distinct kernel names, the loop shape of the
store placement, and agreement between the Python and C++ engines. Numerical
correctness on hardware is ``numeric/test_gfx1201_wmma_kloop_traits_numeric.py``.
"""

from __future__ import annotations

import itertools
import re
import unittest
from dataclasses import replace

from rocke.tests.arch_marks import arch_specific

ARCH = "gfx1201"
_ON = dict(lds_store_after_mma=True, group_m=4, k_stagger=8, k_stagger_step=4)


def _spec(**trait):
    from rocke.instances.common.gemm_universal import (
        DataSpec,
        TileSpec,
        TraitSpec,
        UniversalGemmSpec,
    )

    return UniversalGemmSpec(
        name="kloop_traits",
        tile=TileSpec(128, 128, 32, 2, 2, 1, 16, 16, 16),
        trait=TraitSpec(
            **{
                "pipeline": "compv4",
                "epilogue": "default",
                "lds_swizzle": True,
                **trait,
            }
        ),
        data=DataSpec("fp16", "fp16", "fp16"),
        wave_size=32,
    )


def _mfma_spec(**trait):
    from rocke.instances.common.gemm_universal import (
        DataSpec,
        TileSpec,
        TraitSpec,
        UniversalGemmSpec,
    )

    return UniversalGemmSpec(
        name="kloop_traits_mfma",
        tile=TileSpec(128, 128, 32, 2, 2, 1, 32, 32, 16),
        trait=TraitSpec(**{"pipeline": "compv4", "epilogue": "default", **trait}),
        data=DataSpec("fp16", "fp16", "fp16"),
    )


def _lower(spec, arch=ARCH):
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
    from rocke.instances.common.gemm_universal import build_universal_gemm

    return _lower_kernel_to_llvm_python(
        build_universal_gemm(spec, arch=arch), arch=arch
    )


def _loop_body(ir: str) -> list:
    """The K loop's body block (``for.body.N:`` up to the next label). The
    compv4 kernel has one ``scf.for``: the double-buffered K loop."""
    lines = ir.splitlines()
    start = next(i for i, l in enumerate(lines) if re.match(r"for\.body\.\d+:", l))
    end = next(
        i for i in range(start + 1, len(lines)) if re.match(r"[\w.]+:", lines[i])
    )
    return lines[start:end]


# Rejected configurations, one per gate, with the reason both engines give.
_REJECTS = {
    "store placement on mem": (
        dict(pipeline="mem", lds_store_after_mma=True),
        "lds_store_after_mma needs the WMMA compv4 double-buffered K loop",
    ),
    "store placement on cshuffle": (
        dict(epilogue="cshuffle", lds_store_after_mma=True),
        None,  # the cshuffle epilogue is rejected first on gfx1201
    ),
    "stagger on mem": (
        dict(pipeline="mem", k_stagger=8),
        "k_stagger needs the WMMA compv4 double-buffered K loop",
    ),
    "stagger with split_k": (
        dict(k_stagger=8, split_k=2),
        "k_stagger does not support split_k",
    ),
    "stagger with pad_k": (
        dict(k_stagger=8, pad_k=True),
        "k_stagger does not support pad_k",
    ),
    "negative stagger": (dict(k_stagger=-1), "k_stagger must be >= 0"),
    "zero stagger step": (dict(k_stagger=2, k_stagger_step=0), "k_stagger_step >= 1"),
    "negative group_m": (dict(group_m=-1), "group_m must be >= 0"),
}


@arch_specific((ARCH,), reason="the WMMA K-loop traits are admitted on gfx1201 only")
class TestKloopTraitGates(unittest.TestCase):
    def test_each_trait_and_all_together_are_admitted(self):
        from rocke.instances.common.gemm_universal import is_valid_spec

        for label, kw in (
            ("store placement", dict(lds_store_after_mma=True)),
            ("group_m", dict(group_m=8)),
            ("stagger", dict(k_stagger=3, k_stagger_step=2)),
            ("all", _ON),
        ):
            with self.subTest(label):
                self.assertEqual(is_valid_spec(_spec(**kw), arch=ARCH), (True, "ok"))

    def test_rejected_outside_their_loop(self):
        from rocke.instances.common.gemm_universal import is_valid_spec

        for label, (kw, reason) in _REJECTS.items():
            with self.subTest(label):
                ok, why = is_valid_spec(_spec(**kw), arch=ARCH)
                self.assertFalse(ok, f"{label} admitted")
                if reason is not None:
                    self.assertIn(reason, why)

    def test_mfma_rejects_every_trait(self):
        """The MFMA compv4 loop keeps its HotLoop schedule untouched."""
        from rocke.instances.common.gemm_universal import is_valid_spec

        self.assertTrue(is_valid_spec(_mfma_spec(), arch="gfx950")[0])
        for kw in (dict(lds_store_after_mma=True), dict(group_m=4), dict(k_stagger=4)):
            with self.subTest(**kw):
                self.assertFalse(is_valid_spec(_mfma_spec(**kw), arch="gfx950")[0])


@arch_specific((ARCH,), reason="the WMMA K-loop traits are admitted on gfx1201 only")
class TestKloopTraitEmission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.off = _lower(_spec())

    def test_default_is_unchanged(self):
        """Explicit defaults emit the same kernel as an untouched TraitSpec."""
        self.assertEqual(
            self.off,
            _lower(
                _spec(
                    lds_store_after_mma=False, group_m=0, k_stagger=0, k_stagger_step=1
                )
            ),
        )

    def test_each_trait_changes_the_kernel(self):
        for kw in (dict(lds_store_after_mma=True), dict(group_m=4), dict(k_stagger=8)):
            with self.subTest(**kw):
                self.assertNotEqual(_lower(_spec(**kw)), self.off)

    def test_k_stagger_step_without_stagger_is_inert(self):
        """``k_stagger_step`` only scales the offset; with no stagger the
        kernel and its name are those of the default."""
        spec = _spec(k_stagger_step=4)
        self.assertEqual(_lower(spec), self.off)
        self.assertEqual(spec.kernel_name(), _spec().kernel_name())

    def test_stores_follow_the_wmmas_in_the_loop_body(self):
        """With the placement on, every LDS store of the next tile comes after
        the last WMMA of the current tile; off, the stores precede the WMMAs."""
        wmma = "llvm.amdgcn.wmma"
        store = re.compile(r"^\s*store <8 x half> %\S+, ptr addrspace\(3\)")
        on = _lower(_spec(lds_store_after_mma=True))
        for label, ir, after in (("off", self.off, False), ("on", on, True)):
            body = _loop_body(ir)
            w = [i for i, l in enumerate(body) if wmma in l]
            s = [i for i, l in enumerate(body) if store.match(l)]
            with self.subTest(label):
                self.assertTrue(w and s, "loop body has no WMMA or LDS store")
                if after:
                    self.assertGreater(min(s), max(w))
                else:
                    self.assertLess(max(s), min(w))


class TestKloopTraitNames(unittest.TestCase):
    """Kernels that differ only in a K-loop trait must not share a name."""

    def test_tags_name_only_the_traits_that_are_on(self):
        base = _spec().kernel_name()
        for tag in ("_lsam", "_gm", "_kst"):
            self.assertNotIn(tag, base)
        on = _spec(**_ON).kernel_name()
        for tag in ("_lsam", "_gm4", "_kst8x4"):
            self.assertIn(tag, on)

    def test_every_setting_has_its_own_name(self):
        settings = list(
            itertools.product((False, True), (0, 4, 8), ((0, 1), (8, 4), (8, 1)))
        )
        names = {
            _spec(
                lds_store_after_mma=lsam, group_m=gm, k_stagger=s, k_stagger_step=st
            ).kernel_name()
            for lsam, gm, (s, st) in settings
        }
        self.assertEqual(len(names), len(settings))


@arch_specific((ARCH,), reason="the WMMA K-loop traits are admitted on gfx1201 only")
class TestKloopTraitsCppEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import rocke_engine  # noqa: F401
        except ImportError as e:
            raise unittest.SkipTest(f"rocke_engine C++ binding not importable: {e}")

    def test_both_engines_emit_the_same_ir(self):
        import rocke_engine

        from rocke.core.backend import universal_gemm_spec_to_dict

        for lsam, gm, (s, st) in itertools.product(
            (False, True), (0, 4), ((0, 1), (8, 4), (3, 2))
        ):
            spec = _spec(
                lds_store_after_mma=lsam, group_m=gm, k_stagger=s, k_stagger_step=st
            )
            with self.subTest(lsam=lsam, group_m=gm, k_stagger=(s, st)):
                cpp = rocke_engine.gemm_lower_llvm(
                    universal_gemm_spec_to_dict(spec), ARCH
                )
                self.assertEqual(cpp, _lower(spec))

    def test_both_engines_agree_on_every_gate(self):
        import rocke_engine

        from rocke.core.backend import universal_gemm_spec_to_dict
        from rocke.instances.common.gemm_universal import is_valid_spec

        cases = [(label, _spec(**kw), ARCH) for label, (kw, _) in _REJECTS.items()]
        cases += [("all on", _spec(**_ON), ARCH)]
        cases += [
            (f"mfma {k}", _mfma_spec(**{k: v}), "gfx950")
            for k, v in (
                ("lds_store_after_mma", True),
                ("group_m", 4),
                ("k_stagger", 4),
            )
        ]
        for label, spec, arch in cases:
            with self.subTest(label):
                cpp = rocke_engine.gemm_is_valid(
                    universal_gemm_spec_to_dict(spec), arch
                )
                self.assertEqual(tuple(cpp), tuple(is_valid_spec(spec, arch=arch)))


_DB1WG_TILES = (
    (128, 128, 64, 2, 4),
    (256, 128, 32, 4, 2),
    (128, 256, 32, 2, 4),
)
_DB1WG_REASON = "lds_db_allow_1wg needs the gfx1201 WMMA compv4 double-buffered K loop"


def _db1wg_spec(tile=_DB1WG_TILES[0], **traits):
    from rocke.instances.common.gemm_universal import TileSpec

    return replace(
        _spec(**{"lds_db_allow_1wg": True, **traits}),
        tile=TileSpec(*tile, 1, 16, 16, 16),
        block_size=0,
    )


def _db1wg_gate_cases():
    yield "64 KiB", _db1wg_spec(), ARCH
    yield "48 KiB wide M", _db1wg_spec(_DB1WG_TILES[1]), ARCH
    yield "48 KiB wide N", _db1wg_spec(_DB1WG_TILES[2]), ARCH
    yield "padded overflow", _db1wg_spec(lds_k_pad=8), ARCH
    yield "bare overflow", _db1wg_spec((256, 128, 64, 4, 2)), ARCH
    yield "mem", _db1wg_spec(pipeline="mem"), ARCH
    yield "cshuffle", _db1wg_spec(epilogue="cshuffle"), ARCH
    yield "direct to LDS", _db1wg_spec(direct_to_lds=True), ARCH
    yield "other WMMA", _spec(
        pipeline="mem", lds_swizzle=False, lds_db_allow_1wg=True
    ), "gfx1151"
    other = _spec(pipeline="mem", lds_swizzle=False, lds_db_allow_1wg=True)
    yield "gfx1250 WMMA", replace(
        other, tile=replace(other.tile, warp_tile_k=32)
    ), "gfx1250"
    yield "MFMA", _mfma_spec(lds_db_allow_1wg=True), "gfx950"


@arch_specific((ARCH,), reason="the one-workgroup LDS policy targets gfx1201")
class TestOneWorkgroupLdsPolicy(unittest.TestCase):
    def test_enlarged_tiles_fit_without_exceeding_capacity(self):
        from rocke.instances.common.gemm_universal import _ab_lds_plan, is_valid_spec

        for tile, expected_bytes in zip(_DB1WG_TILES, (65536, 49152, 49152)):
            for dtype in ("fp16", "bf16"):
                spec = _db1wg_spec(tile, **_ON)
                spec = replace(
                    spec,
                    data=replace(
                        spec.data, dtype_a=dtype, dtype_b=dtype, dtype_c=dtype
                    ),
                )
                with self.subTest(tile=tile, dtype=dtype):
                    self.assertEqual(is_valid_spec(spec, arch=ARCH), (True, "ok"))
                    single, db, two_buf = _ab_lds_plan(spec, ARCH)
                    self.assertEqual(
                        (single * 2, db, two_buf), (expected_bytes, True, True)
                    )
                    off = replace(
                        spec, trait=replace(spec.trait, lds_db_allow_1wg=False)
                    )
                    self.assertFalse(_ab_lds_plan(off, ARCH)[1])
                    self.assertFalse(is_valid_spec(off, arch=ARCH)[0])

    def test_padding_is_charged_without_rejecting_a_fitting_double_buffer(self):
        from rocke.instances.common.gemm_universal import _ab_lds_plan, is_valid_spec

        spec = _db1wg_spec(_DB1WG_TILES[1], lds_k_pad=8)
        self.assertEqual(is_valid_spec(spec, arch=ARCH), (True, "ok"))
        self.assertEqual(_ab_lds_plan(spec, ARCH), (30720, True, True))

    def test_oversize_and_wrong_paths_reject_without_single_buffer_fallback(self):
        from rocke.instances.common.gemm_universal import is_valid_spec

        # Earlier architecture/path validators keep their existing precedence.
        for label, spec, arch in _db1wg_gate_cases():
            if label in ("64 KiB", "48 KiB wide M", "48 KiB wide N"):
                continue
            with self.subTest(label):
                ok, why = is_valid_spec(spec, arch=arch)
                self.assertFalse(ok)
                if label not in ("cshuffle", "direct to LDS"):
                    self.assertEqual(why, _DB1WG_REASON)

    def test_opt_in_is_allowed_when_two_workgroups_already_fit(self):
        from rocke.instances.common.gemm_universal import is_valid_spec

        self.assertEqual(
            is_valid_spec(_spec(lds_db_allow_1wg=True), arch=ARCH), (True, "ok")
        )

    def test_default_preserves_name_and_ir(self):
        implicit = _spec()
        explicit = _spec(lds_db_allow_1wg=False)
        self.assertEqual(implicit.kernel_name(), explicit.kernel_name())
        self.assertEqual(_lower(implicit), _lower(explicit))
        # The policy alone does not change the loop of an already fitting tile.
        on = _spec(lds_db_allow_1wg=True)
        self.assertTrue(on.kernel_name().endswith("_db1wg"))
        self.assertEqual(
            _lower(on).replace(on.kernel_name(), implicit.kernel_name()),
            _lower(implicit),
        )

    def test_native_validity_and_ir_agree(self):
        try:
            import rocke_engine
        except ImportError as e:
            self.skipTest(f"rocke_engine C++ binding not importable: {e}")
        from rocke.core.backend import universal_gemm_spec_to_dict
        from rocke.instances.common.gemm_universal import is_valid_spec

        for label, spec, arch in _db1wg_gate_cases():
            with self.subTest(label):
                payload = universal_gemm_spec_to_dict(spec)
                valid = is_valid_spec(spec, arch=arch)
                self.assertEqual(
                    tuple(rocke_engine.gemm_is_valid(payload, arch)), valid
                )
                if valid[0]:
                    self.assertEqual(
                        rocke_engine.gemm_lower_llvm(payload, arch), _lower(spec, arch)
                    )


_READS_REASON = "global_addressing needs unpadded nonbatched gfx1201 FP16 RCR"


def _reads_gate_cases():
    """(label, spec, arch, expected reason fragment or None when admitted)."""
    for mode in ("auto", "global64"):
        yield f"{mode} plain", _spec(global_addressing=mode), ARCH, None
        yield f"{mode} all traits", _spec(global_addressing=mode, **_ON), ARCH, None
        yield f"{mode} one workgroup", _db1wg_spec(
            global_addressing=mode, **_ON
        ), ARCH, None
    yield "unknown mode", _spec(global_addressing="buffer"), ARCH, "must be legacy"
    yield "empty mode", _spec(global_addressing=""), ARCH, "must be legacy"
    yield "mem", _spec(pipeline="mem", global_addressing="auto"), ARCH, _READS_REASON
    yield "pad_k", _spec(pad_k=True, global_addressing="auto"), ARCH, _READS_REASON
    yield "wsp3", _spec(pipeline="wsp3", global_addressing="auto"), ARCH, _READS_REASON
    bf16 = _spec(global_addressing="auto")
    bf16 = replace(
        bf16, data=replace(bf16.data, dtype_a="bf16", dtype_b="bf16", dtype_c="bf16")
    )
    yield "bf16", bf16, ARCH, _READS_REASON
    yield "batched", replace(
        _spec(global_addressing="auto"), batched=True
    ), ARCH, _READS_REASON
    yield "MFMA", _mfma_spec(global_addressing="auto"), "gfx950", _READS_REASON


@arch_specific((ARCH,), reason="prepared global reads are admitted on gfx1201 only")
class TestPreparedGlobalReads(unittest.TestCase):
    def test_gates(self):
        from rocke.instances.common.gemm_universal import is_valid_spec

        for label, spec, arch, reason in _reads_gate_cases():
            with self.subTest(label):
                ok, why = is_valid_spec(spec, arch=arch)
                if reason is None:
                    self.assertEqual((ok, why), (True, "ok"))
                else:
                    self.assertFalse(ok)
                    self.assertIn(reason, why)

    def test_build_refuses_what_the_gate_refuses(self):
        from rocke.instances.common.gemm_universal import build_universal_gemm

        # wsp3 is routed to its own emitter before the ordinary gate runs.
        with self.assertRaisesRegex(ValueError, _READS_REASON):
            build_universal_gemm(
                _spec(pipeline="wsp3", global_addressing="auto"), arch=ARCH
            )

    def test_each_mode_has_its_own_name_and_kernel(self):
        specs = {
            m: _spec(global_addressing=m, **_ON) for m in ("legacy", "auto", "global64")
        }
        self.assertEqual(len({s.kernel_name() for s in specs.values()}), 3)
        self.assertEqual(len({_lower(s) for s in specs.values()}), 3)

    def test_native_validity_and_ir_agree(self):
        try:
            import rocke_engine
        except ImportError as e:
            self.skipTest(f"rocke_engine C++ binding not importable: {e}")
        from rocke.core.backend import universal_gemm_spec_to_dict
        from rocke.instances.common.gemm_universal import is_valid_spec

        for label, spec, arch, _ in _reads_gate_cases():
            with self.subTest(label):
                payload = universal_gemm_spec_to_dict(spec)
                valid = is_valid_spec(spec, arch=arch)
                self.assertEqual(
                    tuple(rocke_engine.gemm_is_valid(payload, arch)), valid
                )
                if valid[0]:
                    self.assertEqual(
                        rocke_engine.gemm_lower_llvm(payload, arch), _lower(spec, arch)
                    )


if __name__ == "__main__":
    unittest.main()
