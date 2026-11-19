# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Wgrad and wgrad Stage 2 spec gates, and two opt-in Stage 1 epilogue barriers.

No GPU needed. Every spec here is one the builders cannot honour: an LDS pool
over the cap, a register-prefetch loop past the staged-load limit, a knob on a
loop or epilogue it never reaches (which would only rename the kernel), or a
loader geometry the k-packed line decode cannot take. The Python validator
must refuse each one (``is_valid_wgrad_spec`` and, where the rule is spec-only,
``validate()``), and so must the C++ engine with the same reason, while the
near twins that do build stay admitted. The Stage 2 fold refuses what its
fixed-size C++ folds cannot hold.

The last class checks two Stage 1 epilogues in the built IR: the ws_fuse final
reader only clears slab elements it owns, and the ws_lc tile, which shares LDS
bytes with the K loop's tiles, is written only after a barrier when the
prefetch_regs=2 loop ends without one.
"""

from __future__ import annotations

import dataclasses
import importlib
import os
import sys
import unittest
from dataclasses import replace

_PARITY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parity")


def _emitter_spec(idx: int):
    """(spec, arch) of wgrad parity config ``idx``."""
    if _PARITY not in sys.path:
        sys.path.insert(0, _PARITY)
    return importlib.import_module("conv_implicit_gemm_wgrad_emit")._spec(idx)


def _engine():
    from rocke.core.backend import BackendError, import_engine

    from kernels.conv.engine_bridge import ENGINE_MODULE

    try:
        return import_engine(ENGINE_MODULE)
    except BackendError as e:
        raise unittest.SkipTest(f"C++ engine not importable: {e}") from None


def _spec_dict(spec) -> dict:
    """A WgradConvSpec as the flat dict the C++ binding reads."""
    d = {
        f.name: getattr(spec, f.name)
        for f in dataclasses.fields(spec)
        if f.name not in ("problem", "data", "acc_epilogue", "lds_layout")
        and getattr(spec, f.name) is not None
    }
    d["problem"] = {
        k: v for k, v in dataclasses.asdict(spec.problem).items() if v is not None
    }
    d["dtype_a"] = spec.data.dtype_a
    d["dtype_b"] = spec.data.dtype_b
    d["dtype_d"] = spec.data.dtype_d
    return d


def _xwin_3d_problem(K: int = 72):
    from kernels.common._conv_implicit_gemm_common import ConvProblem

    return ConvProblem(
        N=1, Di=6, Hi=10, Wi=32, C=64, K=K, Z=3, Y=3, X=3, sD=1, pD=1, pH=1, pW=1, dD=1
    )


def _xwin_spec(**kw):
    """A gfx942 3-D bf16 xwin=3 two-stage spec (64x192x32, 1x2 waves)."""
    from kernels.common._conv_implicit_gemm_common import ConvDataSpec
    from kernels.common.conv_implicit_gemm_wgrad import WgradConvSpec

    base = dict(
        problem=_xwin_3d_problem(),
        data=ConvDataSpec(dtype_a="bf16", dtype_b="bf16", dtype_d="bf16"),
        tile_m=64,
        tile_n=192,
        tile_k=32,
        warp_m=1,
        warp_n=2,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=16,
        vector_size_a=8,
        vector_size_b=8,
        vector_size_c=1,
        lds_k_pad=4,
        split_k=2,
        two_stage=True,
        prefetch_regs=1,
        lds_kpack=2,
        xwin=3,
    )
    base.update(kw)
    return WgradConvSpec(**base)


def _gated():
    """(label, spec, arch, reason substring) of specs both engines refuse."""
    s109, a109 = _emitter_spec(109)
    s110, a110 = _emitter_spec(110)
    s111, a111 = _emitter_spec(111)
    s127, a127 = _emitter_spec(127)
    s16, a16 = _emitter_spec(16)
    s11, a11 = _emitter_spec(11)
    s21, a21 = _emitter_spec(21)
    s5, a5 = _emitter_spec(5)
    out = [
        ("pfr2_lds", replace(s109, prefetch_regs=2), a109, "LDS budget 73728"),
        (
            "pfr2_lds_plain",
            replace(s16, tile_m=128, tile_n=128, tile_k=64, prefetch_regs=2),
            a16,
            "LDS budget 73728",
        ),
        (
            "xwin_gn2_lds",
            replace(
                s110,
                tile_m=256,
                warp_m=4,
                warp_n=1,
                xwin=3,
                xwin_groups=2,
                xwin_pp=8,
                xwin_gn=2,
                prefetch_regs=1,
                lds_kpack=2,
            ),
            a110,
            "LDS budget 65664",
        ),
        (
            "two_stage_cshuffle",
            replace(s16, epilogue="cshuffle"),
            a16,
            "epilogue='cshuffle' is invalid",
        ),
        (
            "pfr_sched_xwin_pp",
            replace(s127, pfr_sched=1),
            a127,
            "not the xwin_pp ping-pong",
        ),
        (
            "esf_wave32",
            replace(s5, epi_store_fast=True),
            a5,
            "epi_store_fast (wgrad)",
        ),
        (
            "kp_line_undecodable",
            replace(
                s16,
                tile_m=32,
                tile_n=64,
                warp_m=1,
                warp_n=2,
                vector_size_a=4,
                vector_size_b=4,
                prefetch_regs=1,
                lds_kpack=2,
                kp_line=8,
                split_k=1,
                two_stage=False,
            ),
            a16,
            "kp_line=8 needs whole 64-lane groups",
        ),
    ]
    for pfr in (1, 3):
        out.append(
            (
                f"pfr{pfr}_staged_loads",
                replace(
                    s11,
                    tile_k=256,
                    vector_size_a=1,
                    vector_size_b=1,
                    prefetch_regs=pfr,
                ),
                a11,
                "prefetch_regs carries at most 32 global loads",
            )
        )
    for pfr in (1, 2, 3):
        out.append(
            (
                f"streamk_pfr{pfr}",
                replace(s21, prefetch_regs=pfr),
                a21,
                "prefetch_addr/regs rewrites",
            )
        )
    for kw in (
        dict(xwin_gn=1),
        dict(xwin_bq=1),
        dict(xwin_aq=1),
        dict(xwin_aq=3),
        dict(xwin_pps=1),
        dict(xwin_rot=1),
        dict(xwin_kord=1),
        dict(xwin_kord=5),
    ):
        out.append(
            (f"no_xwin_{kw}", replace(s111, **kw), a111, "need xwin"),
        )
    return out


def _admitted():
    """(label, spec, arch): the near twins of the gated specs, which build."""
    s110, a110 = _emitter_spec(110)
    s16, a16 = _emitter_spec(16)
    s11, a11 = _emitter_spec(11)
    s109, a109 = _emitter_spec(109)
    out = [
        (
            "pfr2_fits",
            replace(s16, tile_m=128, tile_n=128, tile_k=32, prefetch_regs=2),
            a16,
        ),
        ("pfr1_big_tile", replace(s109, prefetch_regs=1), a109),
        (
            "pfr1_staged_32",
            replace(s11, tile_k=128, vector_size_a=1, vector_size_b=1, prefetch_regs=1),
            a11,
        ),
        (
            "kp_line_decodable",
            replace(
                s16,
                tile_m=256,
                tile_n=256,
                tile_k=32,
                warp_m=2,
                warp_n=4,
                prefetch_regs=1,
                lds_kpack=2,
                sep_addr=2,
                kp_line=8,
                split_k=1,
                two_stage=False,
            ),
            a16,
        ),
    ]
    for gn in (0, 1):
        out.append(
            (
                f"xwin_gn{gn}_lds",
                replace(
                    s110,
                    tile_m=256,
                    warp_m=4,
                    warp_n=1,
                    xwin=3,
                    xwin_groups=2,
                    xwin_pp=8,
                    xwin_gn=gn,
                    prefetch_regs=1,
                    lds_kpack=2,
                ),
                a110,
            )
        )
    for idx in (113, 120, 127, 133, 137, 139):
        spec, arch = _emitter_spec(idx)
        out.append((f"parity_{idx}", spec, arch))
    return out


class TestPythonWgradGates(unittest.TestCase):
    def test_gated_specs_are_refused(self):
        from kernels.common.conv_implicit_gemm_wgrad import is_valid_wgrad_spec

        for label, spec, arch, why in _gated():
            with self.subTest(label):
                ok, reason = is_valid_wgrad_spec(spec, arch)
                self.assertFalse(ok)
                self.assertIn(why, reason)

    def test_near_twins_stay_admitted(self):
        from kernels.common.conv_implicit_gemm_wgrad import is_valid_wgrad_spec

        for label, spec, arch in _admitted():
            with self.subTest(label):
                self.assertEqual(is_valid_wgrad_spec(spec, arch), (True, "ok"))

    def test_spec_only_rules_raise_in_validate(self):
        # The knob rules live in the gate validate() shares, so a builder call
        # fails with the reason rather than building a renamed kernel.
        for label, spec, _arch, why in _gated():
            if "lds" in label:
                continue  # the LDS budget needs the arch
            with self.subTest(label):
                with self.assertRaisesRegex(ValueError, ".+") as cm:
                    spec.validate()
                self.assertIn(why, str(cm.exception))

    def test_lds_bswz_needs_a_k_packed_b_tile(self):
        # 64x32 over 256 threads is one 8-wide chunk per thread: the B store
        # cannot take row_kpack 2, and the builder would raise.
        from kernels.common._conv_implicit_gemm_common import ConvDataSpec
        from kernels.common.conv_implicit_gemm_wgrad import is_valid_wgrad_spec

        s16, arch = _emitter_spec(16)
        spec = replace(
            s16,
            data=ConvDataSpec(dtype_a="bf16", dtype_b="bf16", dtype_d="bf16"),
            tile_m=64,
            tile_n=64,
            tile_k=32,
            prefetch_regs=1,
            lds_kpack=2,
            lds_bswz=1,
            split_k=1,
            two_stage=False,
        )
        ok, reason = is_valid_wgrad_spec(spec, arch)
        self.assertFalse(ok)
        self.assertIn("lds_bswz needs the k-packed row-mode B store", reason)
        # A taller B tile (two chunks per thread) takes it.
        self.assertEqual(
            is_valid_wgrad_spec(replace(spec, tile_n=128), arch), (True, "ok")
        )

    def test_kp_line_gate_matches_the_loader(self):
        # The gate mirrors CoalescedTileLoader's own kp_line rule: for every
        # k-packed loader the builder would make, the gate admits the spec
        # exactly when the loader can be constructed.
        from rocke.helpers.loads import CoalescedTileLoader

        from kernels.common.conv_implicit_gemm_wgrad import (
            _kp_line_conflict,
            _sync_kpack_operands,
        )

        s16, _ = _emitter_spec(16)
        checked = 0
        for tm, tn, tk, wm, wn, vec in (
            (32, 64, 64, 1, 2, 4),
            (64, 64, 32, 2, 2, 8),
            (128, 128, 32, 2, 2, 8),
            (256, 256, 32, 2, 4, 8),
            (64, 128, 64, 1, 2, 8),
            (128, 64, 32, 4, 2, 4),
        ):
            spec = replace(
                s16,
                tile_m=tm,
                tile_n=tn,
                tile_k=tk,
                warp_m=wm,
                warp_n=wn,
                vector_size_a=vec,
                vector_size_b=vec,
                prefetch_regs=1,
                lds_kpack=2,
                kp_line=8,
                split_k=1,
                two_stage=False,
            )
            ops = _sync_kpack_operands(spec)
            built = True
            for rows, cols, threads, lv, kp in ops:
                if kp == 1:
                    continue
                try:
                    CoalescedTileLoader(
                        tile_rows=rows,
                        tile_cols=cols,
                        block_size=threads,
                        load_vec=lv,
                        vector_axis="row",
                        row_kpack=kp,
                        kp_line=8,
                    )
                except ValueError:
                    built = False
            if any(op[4] > 1 for op in ops):
                checked += 1
                with self.subTest(tile=(tm, tn, tk, wm, wn, vec)):
                    self.assertEqual(_kp_line_conflict(spec) == "", built)
        self.assertGreaterEqual(checked, 4)

    def test_ws_lc_and_ws_gred_tiles_are_charged(self):
        # The epilogue tile aliases the K loop's tiles, so it is charged on its
        # own: 16 waves x 16 rows x (64 + 4) f32 is 69632 bytes.
        from kernels.common.conv_implicit_gemm_wgrad import is_valid_wgrad_spec

        prob = _xwin_3d_problem(K=256)
        for kw in (
            dict(tile_m=128, warp_m=8, warp_n=1, xwin_groups=2),
            dict(tile_m=64, warp_m=4, warp_n=1, xwin_groups=4),
        ):
            with self.subTest(kw):
                ok, reason = is_valid_wgrad_spec(
                    _xwin_spec(problem=prob, ws_lc=1, **kw), "gfx942"
                )
                self.assertFalse(ok)
                self.assertIn("LDS budget 69632 bytes (the ws_lc", reason)
        for kw in (
            dict(ws_lc=1),
            dict(ws_lc=1, tile_m=64, warp_m=4, warp_n=1, xwin_groups=2),
            dict(ws_gred=1, tile_m=128, warp_m=4, warp_n=1, xwin_groups=2),
        ):
            with self.subTest(kw):
                self.assertEqual(
                    is_valid_wgrad_spec(_xwin_spec(problem=prob, **kw), "gfx942"),
                    (True, "ok"),
                )


class TestCppAgreesWithPython(unittest.TestCase):
    def test_gated_and_admitted_verdicts_agree(self):
        from kernels.common.conv_implicit_gemm_wgrad import is_valid_wgrad_spec

        eng = _engine()
        cases = [(label, spec, arch) for label, spec, arch, _ in _gated()]
        cases += _admitted()
        for label, spec, arch in cases:
            with self.subTest(label):
                self.assertEqual(
                    tuple(eng.conv_wgrad_is_valid_spec(_spec_dict(spec), arch)),
                    is_valid_wgrad_spec(spec, arch),
                )


def _reduce_spec(**kw):
    from kernels.common._conv_implicit_gemm_common import ConvProblem
    from kernels.common.conv_wgrad_workspace_reduce import WgradReduceSpec

    base = dict(
        problem=ConvProblem(N=1, Hi=4, Wi=4, C=64, K=64, Y=3, X=3), dtype_d="bf16"
    )
    base.update(kw)
    return WgradReduceSpec(**base)


def _reduce_dict(spec) -> dict:
    """A WgradReduceSpec as the dict the C++ conv_wgrad_reduce binding reads."""
    from kernels.common.conv_wgrad_workspace_reduce import _wg_M, _wg_N

    return dict(
        wg_M=_wg_M(spec.problem),
        wg_N=_wg_N(spec.problem),
        problem_short=spec.problem.short(),
        dtype_d=spec.dtype_d,
        tile_m=spec.tile_m,
        tile_n=spec.tile_n,
        name=spec.name,
        groups=spec.groups,
        ws_replicas=spec.ws_replicas,
        vec=spec.vec,
        rezero=spec.rezero,
        ws_bf16=spec.ws_bf16,
        fold_loop=spec.fold_loop,
        slab_par=spec.slab_par,
    )


_REDUCE_GATED = (
    (dict(ws_bf16=True), "ws_bf16 needs vec > 1"),
    (dict(vec=4, slab_par=4), "slab_par needs fold_loop"),
    (dict(vec=16), "vec must be 1, 2, 4 or 8"),
    (dict(vec=3), "vec must be 1, 2, 4 or 8"),
    (dict(vec=4, ws_replicas=256, fold_loop=128, tile_m=1), "fold_loop must be 0..64"),
    (dict(vec=4, slab_par=0), "slab_par must be >= 1"),
    (
        dict(vec=8, problem=None),
        "must divide the dW row length",
    ),
)


def _reduce_gated():
    from kernels.common._conv_implicit_gemm_common import ConvProblem

    for kw, why in _REDUCE_GATED:
        kw = dict(kw)
        if kw.get("problem", 1) is None:
            # wg_N = 3 * 3 * 4 = 36: not a multiple of 8
            kw["problem"] = ConvProblem(N=1, Hi=4, Wi=4, C=4, K=64, Y=3, X=3)
        yield kw, why


class TestWgradReduceGates(unittest.TestCase):
    def test_python_refuses(self):
        from kernels.common.conv_wgrad_workspace_reduce import (
            build_conv_wgrad_workspace_reduce,
        )

        for kw, why in _reduce_gated():
            with self.subTest(kw):
                with self.assertRaisesRegex(ValueError, why):
                    build_conv_wgrad_workspace_reduce(_reduce_spec(**kw), arch="gfx942")

    def test_cpp_refuses_alike(self):
        # The C++ folds hold one value per vector lane and per looped slab
        # load in fixed arrays: past them, the validator must refuse before
        # the builder runs.
        eng = _engine()
        for kw, why in _reduce_gated():
            with self.subTest(kw):
                with self.assertRaisesRegex(RuntimeError, why):
                    eng.conv_wgrad_reduce_lower_llvm(
                        _reduce_dict(_reduce_spec(**kw)), "gfx942"
                    )

    def test_admitted_folds_still_build(self):
        from kernels.common.conv_wgrad_workspace_reduce import (
            build_conv_wgrad_workspace_reduce,
        )

        for kw in (
            dict(),
            dict(vec=4),
            dict(vec=8),
            dict(vec=4, ws_bf16=True),
            dict(vec=4, ws_replicas=256, fold_loop=64, tile_m=1),
            dict(vec=4, ws_replicas=48, fold_loop=4, slab_par=4, tile_m=1),
        ):
            with self.subTest(kw):
                build_conv_wgrad_workspace_reduce(_reduce_spec(**kw), arch="gfx942")


def _ops(region, ancestors=()):
    """Every op under ``region`` in program order, with its ancestor op names."""
    for op in region.ops:
        yield op, ancestors
        for r in op.regions:
            yield from _ops(r, ancestors + (op.name,))


class TestStage1EpilogueBarriers(unittest.TestCase):
    def test_ws_fuse_clears_only_in_bounds_slab_elements(self):
        # An out-of-bounds element's slab offset is clamped to 0, tile (0, 0)'s
        # first element: the final reader of an edge tile must not zero it.
        from kernels.common._conv_implicit_gemm_common import ConvProblem
        from kernels.common.conv_implicit_gemm_wgrad import (
            build_implicit_gemm_conv_wgrad,
        )

        p = ConvProblem(
            N=1,
            Di=8,
            Hi=16,
            Wi=16,
            C=24,
            K=72,
            Z=3,
            Y=3,
            X=3,
            sD=1,
            pD=1,
            pH=1,
            pW=1,
            dD=1,
        )
        spec = _xwin_spec(
            problem=p,
            tile_n=64,
            warp_m=2,
            warp_n=2,
            xwin=0,
            ws_replicas=1,
            sep_addr=2,
            ws_fuse=1,
        )
        k = build_implicit_gemm_conv_wgrad(spec, arch="gfx942")
        zeroing = [
            anc
            for op, anc in _ops(k.body)
            if op.name == "tile.inline_asm"
            and "global_store_dword" in str(op.attrs.get("template", ""))
        ]
        self.assertTrue(zeroing)
        for anc in zeroing:
            # the final-arriver branch, then the element's own bounds branch
            self.assertGreaterEqual(anc.count("scf.if"), 2)

    def test_ws_lc_tile_is_written_after_a_barrier(self):
        from kernels.common.conv_implicit_gemm_wgrad import (
            build_implicit_gemm_conv_wgrad,
        )

        for pfr in (1, 2, 3):
            with self.subTest(prefetch_regs=pfr):
                k = build_implicit_gemm_conv_wgrad(
                    _xwin_spec(prefetch_regs=pfr, ws_lc=1), arch="gfx942"
                )
                names = []
                alloc = None
                for op, _ in _ops(k.body):
                    if (
                        op.name == "tile.smem_alloc"
                        and op.results
                        and "ws_lc" in str(getattr(op.results[0], "name", ""))
                    ):
                        alloc = len(names)
                    names.append(op.name)
                self.assertIsNotNone(alloc)
                last_mma = max(i for i in range(alloc) if names[i] == "tile.mma")
                self.assertIn("tile.sync_lds_only", names[last_mma:alloc])


if __name__ == "__main__":
    unittest.main(verbosity=2)
