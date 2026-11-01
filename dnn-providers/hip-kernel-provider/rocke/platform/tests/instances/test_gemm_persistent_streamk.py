# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only checks of the persistent / stream-K / pad universal-GEMM levers.

* ``persistent_ctas`` on the in-stream (fragment-prefetch / interleave) bodies;
* ``streamk``: stream-K tail balancing in the persistent loop (DP + 2-tile SK
  partition in the kernel, linear fixup with self-cleaning ready flags);
* ``dtl_tight_rsrc``: DTL buffer resources based at the tile origin with the
  operand's remaining bytes as range (rows past M / N read zeros);
* ``pad_fast_tile``: a uniform full-tile branch to the unguarded epilogue;
* the optional ``ordering="monotonic"`` of ``global_flag_wait_eq``.

Every lever is default-off (the default trait keeps the historical emission);
the C++ engine mirror is checked through ``backend="both"`` when it is built.
"""

from __future__ import annotations

import re
import unittest
import warnings
from dataclasses import replace

from rocke.core.ir import I32, IRBuilder, PtrType
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.instances.common.gemm_universal import (
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
    is_valid_spec,
    launch_shape_error,
    mono_data_spec,
    streamk_launch_error,
    streamk_resident_ctas,
    streamk_workspace_bytes,
)

_T32 = TileSpec(256, 256, 32, 2, 2, 1, 16, 16, 32)  # 4-wave AGPR interleave geometry
_F_TILE = TileSpec(256, 256, 64, 4, 4, 1, 16, 16, 32)  # 16-wave ring
_A32 = TileSpec(256, 256, 64, 2, 2, 1, 32, 32, 16)  # 4-wave 32x32x16
_F942 = TileSpec(256, 256, 32, 4, 4, 1, 16, 16, 16)
_BASE = TraitSpec(
    pipeline="compv4",
    epilogue="default",
    direct_to_lds=True,
    dtl_prefetch=True,
    lds_swizzle=True,
    dtl_cache_a=0,
    dtl_cache_b=0,
    chiplet_swizzle=True,
)
_AGPR_ILV = dict(
    dtl_stages=3,
    interwave_setprio=True,
    dtl_frag_prefetch=True,
    acc_agpr=True,
    mfma_interleave=True,
    ilv_barrier_after=2,
    ilv_merge=True,
    lds_swizzle_r=1,
    dtl_hoist=True,
)
_PAD = dict(pad_m=True, pad_n=True, dtl_tight_rsrc=True)


def _spec(tile=_T32, dtype="fp16", **trait):
    return UniversalGemmSpec(
        name="t",
        tile=tile,
        trait=replace(_BASE, **trait),
        data=mono_data_spec(dtype),
        wave_size=64,
    )


def _build(s, arch="gfx950"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_universal_gemm(s, arch=arch)


def _ll(s, arch="gfx950"):
    return lower_kernel_to_llvm(_build(s, arch), arch=arch)


def _ops(region, name, out=None):
    out = [] if out is None else out
    for op in region.ops:
        if op.name == name:
            out.append(op)
        for r in op.regions:
            _ops(r, name, out)
    return out


class TestDefaults(unittest.TestCase):
    def test_default_off(self):
        tr = TraitSpec()
        self.assertFalse(tr.streamk)
        self.assertFalse(tr.dtl_tight_rsrc)
        self.assertFalse(tr.pad_fast_tile)

    def test_no_new_params_or_flags_by_default(self):
        k = _build(_spec(**_AGPR_ILV))
        self.assertEqual([p.name for p in k.params], ["A", "B", "C", "M", "N", "K"])
        name = _spec(**_AGPR_ILV).kernel_name()
        for flag in ("_sk", "_trs", "_pft"):
            self.assertNotIn(flag, name)


class TestPersistentInStream(unittest.TestCase):
    def test_valid_on_the_interleave_body(self):
        s = _spec(**_AGPR_ILV, persistent_ctas=256)
        ok, why = is_valid_spec(s, "gfx950")
        self.assertTrue(ok, why)
        self.assertIn("_pc256", s.kernel_name())
        # one persistent tile loop around the K loop
        self.assertTrue(_ops(_build(s).body, "scf.for"))

    def test_stream_barrier_body_valid_too(self):
        s = _spec(_F_TILE, dtl_stages=2, dtl_stream_barrier=True, persistent_ctas=256)
        self.assertTrue(is_valid_spec(s, "gfx950")[0])
        _build(s)

    def test_still_needs_the_ring(self):
        ok, why = is_valid_spec(_spec(persistent_ctas=256), "gfx950")
        self.assertFalse(ok)
        self.assertEqual(why, "persistent_ctas requires dtl_stages >= 2")


class TestStreamK(unittest.TestCase):
    def test_validity(self):
        ok, why = is_valid_spec(_spec(**_AGPR_ILV, streamk=True), "gfx950")
        self.assertFalse(ok)
        self.assertEqual(
            why, "streamk requires persistent_ctas > 0 (the stream-K pool)"
        )
        for s in (
            _spec(**_AGPR_ILV, persistent_ctas=256, streamk=True),
            _spec(_F_TILE, dtl_stages=2, persistent_ctas=256, streamk=True),
            _spec(
                _A32,
                dtl_stages=2,
                dtl_frag_prefetch=True,
                acc_agpr=True,
                mfma_interleave=True,
                persistent_ctas=256,
                streamk=True,
            ),
        ):
            with self.subTest(name=s.kernel_name()):
                ok, why = is_valid_spec(s, "gfx950")
                self.assertTrue(ok, why)
                self.assertTrue(s.kernel_name().endswith("_sk"))

    def test_params_and_workspace(self):
        s = _spec(**_AGPR_ILV, persistent_ctas=256, streamk=True)
        k = _build(s)
        self.assertEqual(
            [p.name for p in k.params],
            ["A", "B", "C", "M", "N", "K", "SkFlags", "SkPart"],
        )
        # flags: 256 workgroups x 4 waves x 4 B; partials: 256 x 256x256 f32
        self.assertEqual(streamk_workspace_bytes(s), (4096, 256 * 256 * 256 * 4))
        s16 = _spec(_F_TILE, dtl_stages=2, persistent_ctas=256, streamk=True)
        self.assertEqual(streamk_workspace_bytes(s16)[0], 256 * 16 * 4)
        self.assertEqual(streamk_workspace_bytes(_spec(**_AGPR_ILV)), (0, 0))

    def test_persistent_grid_launches_the_whole_pool(self):
        from rocke.dispatch.gemm.common import persistent_grid

        s = _spec(**_AGPR_ILV, persistent_ctas=256)
        sk = _spec(**_AGPR_ILV, persistent_ctas=256, streamk=True)
        # 36 output tiles: the plain persistent loop launches 36, stream-K 256
        self.assertEqual(persistent_grid(s, 1536, 1536), (36, 1, 1))
        self.assertEqual(persistent_grid(sk, 1536, 1536), (256, 1, 1))

    def test_relaxed_spin_one_acquire_per_partner(self):
        s = _spec(**_AGPR_ILV, persistent_ctas=256, streamk=True)
        k = _build(s)
        waits = _ops(k.body, "memref.global_flag_wait_eq")
        self.assertEqual(len(waits), 1)
        self.assertEqual(waits[0].attrs.get("ordering"), "monotonic")
        fences = [
            (f.attrs["scope"], f.attrs["ordering"])
            for f in _ops(k.body, "memref.fence")
        ]
        self.assertEqual(sorted(fences), [("agent", "acquire"), ("agent", "release")])
        stores = [f.attrs["ordering"] for f in _ops(k.body, "memref.global_flag_store")]
        self.assertEqual(stores, ["monotonic", "monotonic"])  # flag reset + publish
        ll = lower_kernel_to_llvm(k, arch="gfx950")
        self.assertIn('syncscope("agent") monotonic, align 4', ll)
        self.assertNotIn('syncscope("agent") acquire, align 4', ll)
        self.assertIn('fence syncscope("agent") acquire', ll)

    def test_partials_use_vector_buffer_ops(self):
        ll = _ll(_spec(**_AGPR_ILV, persistent_ctas=256, streamk=True))
        self.assertIn("llvm.amdgcn.raw.ptr.buffer.store.v4", ll.replace("i32", "f32"))
        # 32x32 atoms (16 floats per lane) store / load four 16-byte quads each
        a32 = _spec(
            _A32,
            dtl_stages=2,
            dtl_frag_prefetch=True,
            acc_agpr=True,
            mfma_interleave=True,
            persistent_ctas=256,
            streamk=True,
        )
        _ll(a32)


class TestStreamKResidency(unittest.TestCase):
    """an owner spins on higher-index workgroups, so all P of the
    stream-K pool must be co-resident; validity rejects P above the static
    bound (CUs x workgroups per CU at the LDS use and the 512-register worst
    case), the launch helper checks the live device."""

    def test_agpr_ilv_one_workgroup_per_cu(self):
        ok, why = is_valid_spec(
            _spec(**_AGPR_ILV, persistent_ctas=256, streamk=True), "gfx950"
        )
        self.assertTrue(ok, why)
        for p in (264, 512, 1024, 4096, 100000):
            with self.subTest(P=p):
                ok, why = is_valid_spec(
                    _spec(**_AGPR_ILV, persistent_ctas=p, streamk=True), "gfx950"
                )
                self.assertFalse(ok)
                self.assertEqual(
                    why,
                    f"streamk needs all persistent_ctas = {p} workgroups co-resident, "
                    "but at most 256 fit on gfx950 (256 CUs x 1 per CU: LDS 98304 of "
                    "163840 B, 4 waves at the 512-register worst case)",
                )
                # the plain persistent loop (no spin) keeps any pool size
                self.assertTrue(
                    is_valid_spec(_spec(**_AGPR_ILV, persistent_ctas=p), "gfx950")[0]
                )

    def test_lds_and_register_limits(self):
        # LDS-limited: 1 wave, 64 KiB -> 2 per CU; register-limited: 2 waves at
        # 32 KiB -> 2 per CU (4 SIMDs, one 512-register wave each)
        for tile, lds in (
            (TileSpec(128, 128, 64, 1, 1, 1, 16, 16, 32), 65536),
            (TileSpec(128, 128, 32, 2, 1, 1, 16, 16, 32), 32768),
        ):
            with self.subTest(tile=tile):
                ok512 = _spec(tile, dtl_stages=2, persistent_ctas=512, streamk=True)
                self.assertTrue(is_valid_spec(ok512, "gfx950")[0])
                self.assertEqual(streamk_resident_ctas(ok512, "gfx950", lds), (512, 2))
                bad = _spec(tile, dtl_stages=2, persistent_ctas=1024, streamk=True)
                ok, why = is_valid_spec(bad, "gfx950")
                self.assertFalse(ok)
                self.assertIn("at most 512 fit on gfx950 (256 CUs x 2 per CU", why)

    def test_gfx942_cu_count(self):
        sk = dict(dtl_stages=2, streamk=True, **_PAD)
        self.assertTrue(
            is_valid_spec(_spec(_F942, persistent_ctas=304, **sk), "gfx942")[0]
        )
        ok, why = is_valid_spec(_spec(_F942, persistent_ctas=312, **sk), "gfx942")
        self.assertFalse(ok)
        self.assertIn("at most 304 fit on gfx942 (304 CUs x 1 per CU", why)

    def test_launch_error_on_the_live_device(self):
        s = _spec(**_AGPR_ILV, persistent_ctas=256, streamk=True)
        self.assertIsNone(streamk_launch_error(s, 256, 1))
        # a CPX partition of the same device (32 CUs) cannot hold the pool
        self.assertIn("streamk pool 256 > 32", streamk_launch_error(s, 32, 1))
        self.assertIsNone(
            streamk_launch_error(_spec(**_AGPR_ILV, persistent_ctas=256), 32, 1)
        )

    def test_warp_k_rule_and_workspace_guard(self):
        s = _wk2_spec()
        ok, why = is_valid_spec(s, "gfx950")
        self.assertFalse(ok)
        self.assertEqual(
            why,
            "streamk requires warp_k == 1 (got 2): the partial-tile slot is sized "
            "tile_m x tile_n",
        )
        no_sk = replace(s, trait=replace(s.trait, streamk=False))
        self.assertTrue(is_valid_spec(no_sk, "gfx950")[0])
        with self.assertRaisesRegex(ValueError, "warp_k 2 != 1"):
            streamk_workspace_bytes(s)


def _wk2_spec():
    return UniversalGemmSpec(
        name="t",
        tile=TileSpec(256, 256, 32, 2, 2, 2, 16, 16, 32),
        trait=replace(_BASE, **_AGPR_ILV, persistent_ctas=256, streamk=True),
        data=mono_data_spec("fp16"),
        wave_size=64,
        block_size=256,
    )


class TestFlagWaitOrdering(unittest.TestCase):
    def _kernel(self, **kw):
        b = IRBuilder("k")
        f = b.param("F", PtrType(I32, "global"), align=4)
        b.global_flag_wait_eq(f, b.const_i32(0), b.const_i32(1), **kw)
        return b

    def test_default_has_no_ordering_attr(self):
        b = self._kernel()
        (op,) = _ops(b.kernel.body, "memref.global_flag_wait_eq")
        self.assertEqual(op.attrs, {"scope": "agent"})

    def test_monotonic(self):
        b = self._kernel(ordering="monotonic")
        (op,) = _ops(b.kernel.body, "memref.global_flag_wait_eq")
        self.assertEqual(op.attrs, {"scope": "agent", "ordering": "monotonic"})
        from rocke.core.verify import verify

        self.assertEqual([d for d in verify(b.kernel) if d.severity == "error"], [])
        ll = lower_kernel_to_llvm(b.kernel, arch="gfx950")
        self.assertRegex(
            ll,
            r'load atomic i32, ptr addrspace\(1\) \S+ syncscope\("agent"\) monotonic',
        )
        from rocke.core.lower_hip import lower_kernel_to_hip

        self.assertIn("__ATOMIC_RELAXED", lower_kernel_to_hip(b.kernel, arch="gfx950"))

    def test_rejects_other_orderings(self):
        with self.assertRaisesRegex(ValueError, "unknown flag wait ordering"):
            self._kernel(ordering="release")


class TestTightRsrc(unittest.TestCase):
    def test_validity_and_name(self):
        ok, why = is_valid_spec(_spec(dtl_tight_rsrc=True), "gfx950")
        self.assertFalse(ok)
        self.assertEqual(
            why, "dtl_tight_rsrc requires dtl_stages >= 2 and a non-batched body"
        )
        s = _spec(**_AGPR_ILV, **_PAD)
        self.assertTrue(is_valid_spec(s, "gfx950")[0])
        self.assertTrue(s.kernel_name().endswith("_trs"))
        self.assertIn("_pad_", s.kernel_name())

    def test_resource_based_at_the_tile(self):
        base = _ll(_spec(**_AGPR_ILV))
        tight = _ll(_spec(**_AGPR_ILV, dtl_tight_rsrc=True))
        # every hoisted resource is rebased (one byte GEP per operand per issue
        # set) and soffset is 0
        n_gep = len(re.findall(r"getelementptr inbounds i8, ptr addrspace\(1\)", tight))
        self.assertGreater(
            n_gep,
            len(re.findall(r"getelementptr inbounds i8, ptr addrspace\(1\)", base)),
        )
        self.assertIn("llvm.smax.i32", tight)

    def test_launch_contract(self):
        s = _spec(**_AGPR_ILV, **_PAD)
        self.assertIsNone(launch_shape_error(s, 1536, 1440, 576))
        self.assertIn(
            "do not fit the i32 buffer range", launch_shape_error(s, 65536, 1024, 16384)
        )
        # without pad_n a ragged N stays outside the contract
        s2 = _spec(**_AGPR_ILV, dtl_tight_rsrc=True)
        self.assertIn("N=1440", launch_shape_error(s2, 1536, 1440, 576))


class TestPadFastTile(unittest.TestCase):
    def test_validity_and_name(self):
        for kw, ok_want in (
            (dict(pad_fast_tile=True), False),
            (dict(pad_fast_tile=True, pad_n=True, epilogue="cshuffle"), False),
            (dict(pad_fast_tile=True, pad_n=True, split_k=2), False),
            (dict(pad_fast_tile=True, pad_n=True), True),
            (dict(pad_fast_tile=True, pad_m=True), True),
        ):
            with self.subTest(**kw):
                ok, why = is_valid_spec(_spec(**_AGPR_ILV, **kw), "gfx950")
                self.assertEqual(ok, ok_want, why)
                if not ok:
                    self.assertTrue(why.startswith("pad_fast_tile requires"))
        self.assertTrue(
            _spec(**_AGPR_ILV, **_PAD, pad_fast_tile=True)
            .kernel_name()
            .endswith("_pft")
        )

    def test_two_epilogue_bodies(self):
        slow = _ll(_spec(**_AGPR_ILV, **_PAD))
        fast = _ll(_spec(**_AGPR_ILV, **_PAD, pad_fast_tile=True))
        n_slow = slow.count("store half")
        n_fast = fast.count("store half")
        # the guarded body plus the unguarded copy
        self.assertEqual(n_fast, 2 * n_slow)


# (candidate dtype, shape) -> one dispatched spec per DTL ring candidate body
# ; the parity configs gemm[57..64] pin the same specs byte for
# byte (TestDispatchedSpecsPinned)
DISPATCHED_SHAPES = (
    ("fp16", (8192, 8200, 64)),  # small ring, 2-stage (ragged small K)
    ("bf16", (2048, 2048, 2048)),  # small ring, 3-stage interleave (one wave)
    ("fp16", (2048, 2048, 96)),  # 128x128x32 interleave ring (K % 64 == 32)
    ("bf16", (4608, 4608, 160)),  # 256x128x32 AGPR ring (K % 64 == 32)
    # (shapes the row-interleave bodies do not take, so they still dispatch
    # the DTL ring bodies)
    ("fp16", (10240, 10248, 128)),  # pad ring, 16 waves, persistent
    ("bf16", (24576, 24584, 64)),  # AGPR interleave big-grid pad body
    ("fp16", (24576, 24576, 128)),  # AGPR interleave big-grid aligned body
    ("bf16", (46080, 46080, 256)),  # 16-wave ring, wgm 4 + iglp1
)
# DISPATCHED_SHAPES[i] <-> tests/instances/parity/gemm_emit.py config 57 + i
DISPATCHED_PARITY_BASE = 57
_T1W = TileSpec(128, 128, 64, 1, 1, 1, 16, 16, 32)  # 1 wave, 64 KiB LDS
_T2W = TileSpec(128, 128, 32, 2, 1, 1, 16, 16, 32)  # 2 waves, 32 KiB LDS


def _dispatched_specs():
    from rocke.dispatch import dispatch_gemm_bf16, dispatch_gemm_fp16
    from rocke.dispatch.gemm.common import GemmRequest

    out = []
    for dt, (M, N, K) in DISPATCHED_SHAPES:
        fn = dispatch_gemm_fp16 if dt == "fp16" else dispatch_gemm_bf16
        r = fn(GemmRequest(M=M, N=N, K=K, arch="gfx950", dtype=dt))
        out.append(((dt, M, N, K), r.spec))
    return out


class TestDispatchedSpecsPinned(unittest.TestCase):
    def test_dispatched_specs_are_the_parity_configs(self):
        import importlib.util
        import os
        import sys

        par = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parity")
        sys.path.insert(0, par)
        try:
            spec_ = importlib.util.spec_from_file_location(
                "gemm_emit_r4", os.path.join(par, "gemm_emit.py")
            )
            mod = importlib.util.module_from_spec(spec_)
            spec_.loader.exec_module(mod)
        finally:
            sys.path.remove(par)
        for i, ((dt, M, N, K), s) in enumerate(_dispatched_specs()):
            with self.subTest(shape=(dt, M, N, K)):
                p = mod._spec(DISPATCHED_PARITY_BASE + i)
                self.assertEqual(s.tile, p.tile)
                self.assertEqual(s.trait, p.trait)
                self.assertEqual(s.data, p.data)
                self.assertEqual(
                    (s.block_size, s.wave_size), (p.block_size, p.wave_size)
                )
                self.assertTrue(is_valid_spec(s, "gfx950")[0])


def _engine():
    try:
        import rocke_engine  # type: ignore
    except ImportError:
        return None
    return rocke_engine


@unittest.skipIf(_engine() is None, "rocke_engine (C++ pybind engine) not built")
class TestEngineMirrorPersistent(unittest.TestCase):
    """The C++ engine builds the same kernel as Python for every persistent / stream-K lever and
    gives the same validity verdicts and reasons."""

    _SPECS = (
        ("agpr_ilv_pc", _spec(**_AGPR_ILV, persistent_ctas=256)),
        ("agpr_ilv_pc_sk", _spec(**_AGPR_ILV, persistent_ctas=256, streamk=True)),
        (
            "s2_pc_sk_bf16",
            _spec(
                _F_TILE, dtype="bf16", dtl_stages=2, persistent_ctas=256, streamk=True
            ),
        ),
        (
            "agpr_ilv_pc_sk_pad_pft",
            _spec(
                **_AGPR_ILV,
                **_PAD,
                persistent_ctas=256,
                streamk=True,
                pad_fast_tile=True,
            ),
        ),
        ("s2_pad_pft", _spec(_F_TILE, dtl_stages=2, **_PAD, pad_fast_tile=True)),
        (
            "a32_pc_sk",
            _spec(
                _A32,
                dtl_stages=2,
                dtl_frag_prefetch=True,
                acc_agpr=True,
                mfma_interleave=True,
                persistent_ctas=256,
                streamk=True,
            ),
        ),
    )

    @staticmethod
    def _dispatched():
        """The exact specs the DTL ring dispatcher candidates emit."""
        return [
            (f"disp_{dt}_{M}x{N}x{K}", s) for (dt, M, N, K), s in _dispatched_specs()
        ]

    def test_both_engines_agree(self):
        from rocke.core.backend import lower_universal_gemm

        for name, s in tuple(self._SPECS) + tuple(self._dispatched()):
            with self.subTest(name=name):
                self.assertTrue(is_valid_spec(s, "gfx950")[0])
                r = lower_universal_gemm(s, arch="gfx950", backend="both", want_ir=True)
                self.assertIn(s.kernel_name(), r.llvm_text)

    def test_batched_wrapper_carries_the_levers(self):
        # the batched converter (_gemm_subspec_to_dict) sends every lever: a
        # batched ring body with pad + fast tile + tight-free DTL builds the
        # same kernel in both engines, and the stream-K / persistent rules
        # (non-batched only) reject the same batched specs
        from rocke.core.backend import batched_gemm_spec_to_dict, lower_batched_gemm
        from rocke.instances.common.batched_gemm import (
            BatchedGemmSpec,
            is_valid_spec as bvalid,
        )

        eng = _engine()
        ok_spec = BatchedGemmSpec(
            name="b",
            tile=_F_TILE,
            trait=replace(
                _BASE,
                dtl_stages=2,
                interwave_setprio=True,
                dtl_hoist=True,
                pad_m=True,
                pad_n=True,
                pad_fast_tile=True,
            ),
            dtype="bf16",
        )
        self.assertTrue(bvalid(ok_spec, arch="gfx950")[0])
        r = lower_batched_gemm(ok_spec, arch="gfx950", backend="both", want_ir=True)
        self.assertIn("_pft", r.llvm_text)
        for tr in (
            replace(_BASE, **_AGPR_ILV, streamk=True),
            replace(_BASE, dtl_stages=2, persistent_ctas=256, streamk=True),
            replace(_BASE, dtl_stages=2, dtl_tight_rsrc=True),
        ):
            s = BatchedGemmSpec(name="b", tile=_T32, trait=tr, dtype="fp16")
            with self.subTest(name=s.to_universal_spec().kernel_name()):
                py_ok, py_why = bvalid(s, arch="gfx950")
                c_ok, c_why = eng.batched_gemm_is_valid(
                    batched_gemm_spec_to_dict(s), "gfx950"
                )
                self.assertFalse(py_ok)
                self.assertEqual(bool(c_ok), py_ok, (py_why, c_why))

    def test_same_validity(self):
        from rocke.core.backend import universal_gemm_spec_to_dict

        eng = _engine()
        cases = [(s, "gfx950") for _n, s in self._SPECS]
        cases += [
            (_spec(**_AGPR_ILV, streamk=True), "gfx950"),
            (_spec(dtl_tight_rsrc=True), "gfx950"),
            (_spec(**_AGPR_ILV, pad_fast_tile=True), "gfx950"),
            (
                _spec(**_AGPR_ILV, pad_n=True, pad_fast_tile=True, epilogue="cshuffle"),
                "gfx950",
            ),
            (
                _spec(_F942, dtl_stages=2, persistent_ctas=256, streamk=True, **_PAD),
                "gfx942",
            ),
            # stream-K co-residency and warp_k rules
            (_spec(**_AGPR_ILV, persistent_ctas=264, streamk=True), "gfx950"),
            (_spec(**_AGPR_ILV, persistent_ctas=4096, streamk=True), "gfx950"),
            (_spec(_T1W, dtl_stages=2, persistent_ctas=512, streamk=True), "gfx950"),
            (_spec(_T1W, dtl_stages=2, persistent_ctas=1024, streamk=True), "gfx950"),
            (_spec(_T2W, dtl_stages=2, persistent_ctas=1024, streamk=True), "gfx950"),
            (
                _spec(_F942, dtl_stages=2, persistent_ctas=312, streamk=True, **_PAD),
                "gfx942",
            ),
            (_wk2_spec(), "gfx950"),
        ]
        cases += [(s, "gfx950") for _n, s in self._dispatched()]
        for s, arch in cases:
            with self.subTest(name=s.kernel_name(), arch=arch):
                py_ok, py_why = is_valid_spec(s, arch)
                c_ok, c_why = eng.gemm_is_valid(universal_gemm_spec_to_dict(s), arch)
                self.assertEqual(bool(c_ok), py_ok, (py_why, c_why))
                if not py_ok:
                    self.assertEqual(c_why, py_why)


if __name__ == "__main__":
    unittest.main()
