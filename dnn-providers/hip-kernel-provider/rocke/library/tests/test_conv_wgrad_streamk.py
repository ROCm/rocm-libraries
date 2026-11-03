# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-side tests for stream-K implicit-GEMM backward-weight convolution.

No GPU needed. Covers the spec surface (defaults, kernel-name tags), the
validator gates and their agreement between ``validate()``,
``is_valid_wgrad_spec`` and the C++ port, admits-implies-builds over the
mode x reduction x arch x groups matrix, the AOT kernarg ABI and launch values
(the partition block), the host-side launch plan (grid, workspace layout,
determinism) against the C port, and byte-identical lowering between the two
engines through the ``rocke_engine`` binding.

GPU numerics live in ``test_conv_wgrad_streamk_gpu.py``.
"""

from __future__ import annotations

import dataclasses
import unittest
from dataclasses import replace

from kernels.common._conv_implicit_gemm_common import ConvDataSpec, ConvProblem
from kernels.common.conv_abi import _STREAMK_SCALARS, conv_arg_names
from kernels.common.conv_args import ConvArgs, StreamKLaunch
from kernels.common.conv_implicit_gemm_wgrad import (
    WgradConvSpec,
    build_implicit_gemm_conv_wgrad,
    is_valid_wgrad_spec,
    wgrad_streamk_grid,
    wgrad_streamk_partition,
    wgrad_streamk_signature,
    wgrad_streamk_workspace_layout,
    wgrad_streamk_workspace_nbytes,
)

_DENSE = ConvProblem(N=2, Hi=12, Wi=12, C=32, K=96, Y=3, X=3, pH=1, pW=1)
# kpg = 40 is not a multiple of tile_m, so the last M tile of every group is
# partial: the case CK's own GemmM * GemmBatch fold leaves untested.
_GROUPED = ConvProblem(N=2, Hi=10, Wi=10, C=48, K=120, Y=3, X=3, pH=1, pW=1, groups=3)
# Depthwise: the only shape group merging admits, so its stream-K gate is reached.
_DW = ConvProblem(N=2, Hi=8, Wi=8, C=16, K=16, Y=3, X=3, pH=1, pW=1, groups=16)
# 17 x 27 = 459 output tiles at 64x64: more than the default pool of either
# stream-K arch (256 CTAs on gfx950, 304 on gfx942 for linear/tree), so an
# auto-pool launch really splits tiles and needs flags and partial slots.
_SPLITS = ConvProblem(N=2, Hi=12, Wi=12, C=192, K=1056, Y=3, X=3, pH=1, pW=1)


def _spec(problem=_DENSE, *, dtype_d="fp16", edge=32, **kw) -> WgradConvSpec:
    base = dict(
        problem=problem,
        data=ConvDataSpec(dtype_a="fp16", dtype_b="fp16", dtype_d=dtype_d),
        tile_m=64,
        tile_n=64,
        tile_k=32,
        warp_m=2,
        warp_n=2,
        warp_tile_m=edge,
        warp_tile_n=edge,
        warp_tile_k=16,
        streamk="dp_sk",
        streamk_ctas=6,
    )
    base.update(kw)
    return WgradConvSpec(**base)


def _engine():
    """The ``rocke_engine`` binding, or skip the calling test."""
    from rocke.core.backend import BackendError, _import_engine

    try:
        return _import_engine()
    except BackendError as e:
        raise unittest.SkipTest(f"C++ engine not importable: {e}") from None


def _spec_dict(spec: WgradConvSpec) -> dict:
    """Flatten a WgradConvSpec into the dict ``conv_wgrad_build_spec`` reads."""
    d = {
        f.name: getattr(spec, f.name)
        for f in dataclasses.fields(spec)
        if f.name not in ("problem", "data", "acc_epilogue", "lds_layout")
    }
    d = {k: v for k, v in d.items() if v is not None}
    d["problem"] = {
        k: v for k, v in dataclasses.asdict(spec.problem).items() if v is not None
    }
    d["dtype_a"] = spec.data.dtype_a
    d["dtype_b"] = spec.data.dtype_b
    d["dtype_d"] = spec.data.dtype_d
    return d


class TestStreamKSpecSurface(unittest.TestCase):
    def test_default_is_off_and_name_unchanged(self):
        plain = replace(_spec(), streamk="off", streamk_ctas=-1)
        self.assertNotIn("_sk", plain.kernel_name())
        self.assertEqual(
            plain.kernel_name(),
            WgradConvSpec(
                **{
                    f.name: getattr(plain, f.name)
                    for f in dataclasses.fields(plain)
                    if not f.name.startswith("streamk")
                }
            ).kernel_name(),
        )

    def test_name_tags_every_knob_that_changes_the_body(self):
        names = {
            _spec().kernel_name(),
            _spec(streamk="persistent").kernel_name(),
            _spec(streamk_reduction="tree").kernel_name(),
            _spec(streamk_reduction="workspace").kernel_name(),
            _spec(dtype_d="fp32", streamk_reduction="atomic").kernel_name(),
        }
        self.assertEqual(len(names), 5, names)
        self.assertTrue(_spec().kernel_name().endswith("_sk"))

    def test_pool_is_a_launch_parameter(self):
        # The pool only sizes the host partition: the name and the kernel body
        # are the same for every pool, like the split-K degree.
        from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

        def ll(spec):
            k = build_implicit_gemm_conv_wgrad(spec, arch="gfx950")
            return _lower_kernel_to_llvm_python(k, arch="gfx950")

        base = ll(_spec())
        for ctas in (8, -1):
            spec = _spec(streamk_ctas=ctas)
            self.assertEqual(spec.kernel_name(), _spec().kernel_name())
            self.assertEqual(ll(spec), base)

    def test_auto_pool_depends_on_the_reduction(self):
        from kernels.common.conv_implicit_gemm_wgrad import wgrad_streamk_default_ctas

        # linear/tree wait on later CTAs: one per CU, always resident.
        for red in ("linear", "tree"):
            spec = _spec(streamk_ctas=-1, streamk_reduction=red)
            self.assertEqual(wgrad_streamk_default_ctas(spec, "gfx950"), 256)
            self.assertEqual(wgrad_streamk_default_ctas(spec, "gfx942"), 304)
        # workspace/atomic never wait: four CTAs per CU.
        for red, dd in (("workspace", "fp16"), ("atomic", "fp32")):
            spec = _spec(streamk_ctas=-1, streamk_reduction=red, dtype_d=dd)
            self.assertEqual(wgrad_streamk_default_ctas(spec, "gfx950"), 1024)
            self.assertEqual(wgrad_streamk_default_ctas(spec, "gfx942"), 1216)
        # Counted in CTAs, not waves: an 8-wave CTA gets the same pool.
        wide = _spec(streamk_ctas=-1, streamk_reduction="workspace", warp_m=4)
        self.assertEqual(wgrad_streamk_default_ctas(wide, "gfx950"), 1024)

    def test_problem_is_not_baked_in(self):
        # AOT: the problem shape is a kernarg, so two problems with the same
        # build-time knobs (here: grouping and 2-D) lower to the same body.
        from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

        other = ConvProblem(N=8, Hi=20, Wi=6, C=48, K=24, Y=3, X=3, pH=1, pW=1)
        lls = []
        for problem in (_DENSE, other):
            k = build_implicit_gemm_conv_wgrad(_spec(problem), arch="gfx950")
            body = _lower_kernel_to_llvm_python(k, arch="gfx950")
            lls.append(body.replace(k.name, "KERNEL"))
        self.assertEqual(lls[0], lls[1])


class TestStreamKValidator(unittest.TestCase):
    """Every gate rejects in ``validate()`` and ``is_valid_wgrad_spec`` alike."""

    _REJECTS = [
        (dict(streamk="sideways"), "streamk must be"),
        (dict(streamk_reduction="ring"), "streamk_reduction must be"),
        (dict(streamk_ctas=0), "streamk_ctas must be"),
        (dict(streamk_ctas=-3), "streamk_ctas must be"),
        (dict(split_k=4), "both partition K_wg"),
        (dict(split_k=-1), "both partition K_wg"),
        (dict(two_stage=True, split_k=2), "both partition K_wg"),
        (dict(unroll_k=True), "one plain K loop"),
        (dict(epilogue="cshuffle"), "epilogue='default'"),
        (dict(chiplet_swizzle=True), "chiplet_swizzle"),
        (dict(streamk_reduction="atomic"), "needs dtype_d='fp32'"),
    ]

    def test_rejects(self):
        for kw, needle in self._REJECTS:
            spec = _spec(**kw)
            with self.subTest(kw=kw):
                ok, why = is_valid_wgrad_spec(spec, "gfx950")
                self.assertFalse(ok)
                self.assertIn(needle, why)
                with self.assertRaises(ValueError) as cm:
                    spec.validate()
                self.assertEqual(str(cm.exception), why)

    def test_group_merge_and_acc_epilogue_rejected(self):
        from kernels.common._conv_implicit_gemm_common import ConvAccumulatorEpilogue

        for spec, needle in (
            (_spec(_DW, group_merge=2), "group_merge=1"),
            (_spec(acc_epilogue=ConvAccumulatorEpilogue(relu=True)), "acc_epilogue"),
        ):
            with self.subTest(needle=needle):
                ok, why = is_valid_wgrad_spec(spec, "gfx950")
                self.assertFalse(ok)
                self.assertIn(needle, why)

    def test_arch_gate(self):
        ok, why = is_valid_wgrad_spec(_spec(), "gfx90a")
        self.assertFalse(ok)
        self.assertIn("stream-K wgrad supports only gfx942, gfx950", why)
        # Wave32 targets already fail the wave-size check; either way, rejected.
        for arch in ("gfx1151", "gfx1201", "gfx1250"):
            with self.subTest(arch=arch):
                self.assertFalse(is_valid_wgrad_spec(_spec(), arch)[0])

    def test_cpp_rejects_with_the_same_reason(self):
        eng = _engine()
        for kw, _needle in self._REJECTS:
            spec = _spec(**kw)
            _ok, why = is_valid_wgrad_spec(spec, "gfx950")
            with self.subTest(kw=kw), self.assertRaises(RuntimeError) as cm:
                eng.conv_wgrad_lower_llvm(_spec_dict(spec), "gfx950")
            self.assertIn(why, str(cm.exception))

    # (label, problem, arch, spec overrides). Each rejected config trips one
    # gate only; _MULTI below covers configs that trip several.
    _AGREEMENT = [
        ("ok_dp_sk_linear", _DENSE, "gfx950", {}),
        (
            "ok_pers_tree",
            _GROUPED,
            "gfx950",
            dict(streamk="persistent", streamk_reduction="tree"),
        ),
        (
            "ok_atomic_fp32",
            _DENSE,
            "gfx950",
            dict(dtype_d="fp32", streamk_reduction="atomic"),
        ),
        ("ok_workspace", _GROUPED, "gfx950", dict(streamk_reduction="workspace")),
        ("ok_gfx942", _DENSE, "gfx942", dict(edge=16, streamk="persistent")),
        ("ok_auto_ctas", _DENSE, "gfx950", dict(streamk_ctas=-1)),
        ("ok_off", _DENSE, "gfx950", dict(streamk="off")),
        ("split_k", _DENSE, "gfx950", dict(split_k=4)),
        ("split_k_auto", _DENSE, "gfx950", dict(split_k=-1)),
        ("two_stage", _DENSE, "gfx950", dict(two_stage=True, split_k=2)),
        ("cshuffle", _DENSE, "gfx950", dict(epilogue="cshuffle")),
        ("unroll_k", _DENSE, "gfx950", dict(unroll_k=True)),
        ("chiplet", _DENSE, "gfx950", dict(chiplet_swizzle=True)),
        ("atomic_fp16", _DENSE, "gfx950", dict(streamk_reduction="atomic")),
        (
            "atomic_bf16",
            _DENSE,
            "gfx950",
            dict(dtype_d="bf16", streamk_reduction="atomic"),
        ),
        ("ctas_zero", _DENSE, "gfx950", dict(streamk_ctas=0)),
        ("ctas_minus_two", _DENSE, "gfx950", dict(streamk_ctas=-2)),
        ("bad_mode", _DENSE, "gfx950", dict(streamk="sideways")),
        ("bad_reduction", _DENSE, "gfx950", dict(streamk_reduction="sideways")),
        ("gfx90a", _DENSE, "gfx90a", {}),
        # wave32 so Python's wave-size check passes and the arch gate decides.
        ("gfx1250", _DENSE, "gfx1250", dict(wave_size=32, warp_tile_k=32)),
        ("group_merge", _DW, "gfx950", dict(group_merge=2)),
    ]

    def test_cpp_validity_agrees(self):
        eng = _engine()
        for label, problem, arch, kw in self._AGREEMENT:
            spec = _spec(problem, **kw)
            want = is_valid_wgrad_spec(spec, arch)
            with self.subTest(label):
                self.assertEqual(label.startswith("ok_"), want[0], want[1])
                got = eng.conv_wgrad_is_valid_spec(_spec_dict(spec), arch)
                self.assertEqual(tuple(got), want)

    # Configs that fail more than one gate: the C port runs the gates in
    # Python's order, so both report the first one Python reaches.
    _MULTI = [
        ("ctas_zero+tile_m", _DENSE, "gfx950", dict(streamk_ctas=0, tile_m=96)),
        (
            "block_4096+ctas_zero",
            _DENSE,
            "gfx950",
            dict(warp_m=8, warp_n=8, tile_m=256, tile_n=256, streamk_ctas=0),
        ),
        ("unknown_arch", _DENSE, "gfx9999", {}),
        ("unknown_arch+tile_m", _DENSE, "gfx9999", dict(tile_m=96)),
        (
            "wave32_on_gfx950+ctas",
            _DENSE,
            "gfx950",
            dict(wave_size=32, streamk_ctas=-5),
        ),
        ("wave64_on_gfx1250", _DENSE, "gfx1250", {}),
        ("async_dma", _DENSE, "gfx950", dict(async_dma=True)),
        ("async_dma_off", _DENSE, "gfx950", dict(async_dma=True, streamk="off")),
        ("vec_c+ctas", _DENSE, "gfx950", dict(vector_size_c=4, streamk_ctas=-5)),
        ("vec_c_off", _DENSE, "gfx950", dict(vector_size_c=4, streamk="off")),
        ("group_merge_degree", _DW, "gfx950", dict(group_merge=3)),
        ("group_merge+tile_m", _DW, "gfx950", dict(group_merge=2, tile_m=96)),
        ("gfx90a+split_k", _DENSE, "gfx90a", dict(split_k=4)),
    ]

    def test_cpp_validity_agrees_multi_failure(self):
        eng = _engine()
        for label, problem, arch, kw in self._MULTI:
            spec = _spec(problem, **kw)
            want = is_valid_wgrad_spec(spec, arch)
            with self.subTest(label):
                self.assertFalse(want[0])
                got = eng.conv_wgrad_is_valid_spec(_spec_dict(spec), arch)
                self.assertEqual(tuple(got), want)

    def test_cpp_rejects_grouped_non_streamk(self):
        # Intentional divergence: grouped wgrad is only ported on the stream-K
        # path, so the C port rejects a grouped spec Python accepts -- but only
        # after every gate the two share.
        eng = _engine()
        spec = _spec(_GROUPED, streamk="off")
        self.assertEqual(is_valid_wgrad_spec(spec, "gfx950"), (True, "ok"))
        ok, why = eng.conv_wgrad_is_valid_spec(_spec_dict(spec), "gfx950")
        self.assertFalse(ok)
        self.assertIn("grouped convolution (groups=3 > 1)", why)
        bad = _spec(_GROUPED, streamk="off", tile_m=96)
        self.assertEqual(
            tuple(eng.conv_wgrad_is_valid_spec(_spec_dict(bad), "gfx950")),
            is_valid_wgrad_spec(bad, "gfx950"),
        )

    def test_cpp_rejects_group_merge_with_the_same_reason(self):
        eng = _engine()
        spec = _spec(_DW, group_merge=2)
        _ok, why = is_valid_wgrad_spec(spec, "gfx950")
        with self.assertRaises(RuntimeError) as cm:
            eng.conv_wgrad_lower_llvm(_spec_dict(spec), "gfx950")
        self.assertIn(why, str(cm.exception))


class TestStreamKAdmitsImpliesBuilds(unittest.TestCase):
    def test_matrix(self):
        built = 0
        for arch, edge in (("gfx950", 32), ("gfx942", 16)):
            for problem in (_DENSE, _GROUPED):
                for mode in ("dp_sk", "persistent"):
                    for reduction, dtype_d in (
                        ("linear", "fp16"),
                        ("tree", "bf16"),
                        ("atomic", "fp32"),
                        ("workspace", "fp16"),
                    ):
                        for ctas in (1, 4, 7, 64):
                            spec = _spec(
                                problem,
                                dtype_d=dtype_d,
                                edge=edge,
                                streamk=mode,
                                streamk_reduction=reduction,
                                streamk_ctas=ctas,
                            )
                            ok, why = is_valid_wgrad_spec(spec, arch)
                            with self.subTest(
                                arch=arch,
                                g=problem.groups,
                                mode=mode,
                                r=reduction,
                                w=ctas,
                            ):
                                self.assertTrue(ok, why)
                                build_implicit_gemm_conv_wgrad(spec, arch=arch)
                                built += 1
        self.assertEqual(built, 2 * 2 * 2 * 4 * 4)


class TestStreamKHostPlan(unittest.TestCase):
    def test_grid_matches_partition(self):
        for mode in ("dp_sk", "persistent"):
            spec = _spec(streamk=mode, streamk_ctas=4)
            part = wgrad_streamk_partition(spec)
            with self.subTest(mode=mode):
                want = 4 if mode == "persistent" else part.dp_tiles + part.sk_ctas
                self.assertEqual(wgrad_streamk_grid(spec), (want, 1, 1))

    def test_group_fold_tile_count(self):
        # m_tiles is per group: ceil(kpg / tile_m) * groups, never
        # ceil(kpg * groups / tile_m).
        part = wgrad_streamk_partition(_spec(_GROUPED))
        self.assertEqual(part.m_tiles, 3)  # ceil(40 / 64) * 3
        self.assertEqual(part.n_tiles, 3)  # ceil(3*3*16 / 64)

    def test_workspace_layout(self):
        spec = _spec(streamk_ctas=7)
        self.assertEqual(wgrad_streamk_partition(spec).sk_ctas, 7)
        flags, partials = wgrad_streamk_workspace_layout(spec)
        self.assertEqual(flags, 256)  # 7 CTAs * 4 waves * 4 B, 256-aligned
        self.assertEqual(partials, 7 * 64 * 64 * 4)
        self.assertEqual(wgrad_streamk_workspace_nbytes(spec), flags + partials)
        # 10 tiles over a pool of 5 split evenly: all data-parallel, no slots.
        self.assertEqual(wgrad_streamk_workspace_layout(_spec(streamk_ctas=5)), (0, 0))
        atomic = _spec(dtype_d="fp32", streamk_reduction="atomic")
        self.assertEqual(wgrad_streamk_workspace_nbytes(atomic), 0)
        ws = _spec(streamk_reduction="workspace")
        self.assertEqual(wgrad_streamk_workspace_nbytes(ws), 8 * 96 * 288 * 4)

    def test_launch_rejects_bad_pool(self):
        from kernels.common.conv_implicit_gemm_wgrad import wgrad_streamk_launch

        for ctas in (0, -2, -3):
            with self.subTest(ctas=ctas), self.assertRaisesRegex(
                ValueError, "streamk_ctas must be -1"
            ):
                wgrad_streamk_launch(_spec(streamk_ctas=ctas))
        self.assertEqual(wgrad_streamk_launch(_spec(streamk_ctas=-1)).pool, 256)
        self.assertEqual(
            wgrad_streamk_launch(_spec(streamk_ctas=-1), arch="gfx942").pool, 304
        )

    def test_cpp_host_rejects_bad_pool(self):
        eng = _engine()
        for ctas in (0, -3):
            with self.subTest(ctas=ctas), self.assertRaises(ValueError):
                eng.conv_wgrad_streamk_host(
                    _spec_dict(_spec(streamk_ctas=ctas)), "gfx950"
                )
        with self.assertRaises(ValueError):
            eng.conv_wgrad_streamk_host(_spec_dict(_spec(streamk="off")), "gfx950")

    def test_signature_tracks_reduction(self):
        names = lambda s: [p["name"] for p in wgrad_streamk_signature(s)]  # noqa: E731
        plain = [n for n, _ in conv_arg_names(direction="wgrad")]
        block = list(_STREAMK_SCALARS)
        self.assertEqual(names(_spec()), plain + ["sk_flags", "sk_partials"] + block)
        ws = plain[:-2] + ["ws_ptr", "ws_bytes"] + plain[-2:] + block
        self.assertEqual(names(_spec(streamk_reduction="workspace")), ws)
        self.assertEqual(
            names(_spec(dtype_d="fp32", streamk_reduction="atomic")), plain + block
        )

    def test_cpp_host_plan_agrees(self):
        eng = _engine()
        for problem in (_DENSE, _GROUPED):
            for mode in ("dp_sk", "persistent"):
                for reduction, dtype_d in (
                    ("linear", "fp16"),
                    ("tree", "fp16"),
                    ("atomic", "fp32"),
                    ("workspace", "fp16"),
                ):
                    spec = _spec(
                        problem,
                        dtype_d=dtype_d,
                        streamk=mode,
                        streamk_reduction=reduction,
                        streamk_ctas=7,
                    )
                    got = eng.conv_wgrad_streamk_host(_spec_dict(spec), "gfx950")
                    flags, partials = wgrad_streamk_workspace_layout(spec)
                    with self.subTest(g=problem.groups, mode=mode, r=reduction):
                        self.assertEqual(tuple(got["grid"]), wgrad_streamk_grid(spec))
                        self.assertEqual(got["flags_bytes"], flags)
                        self.assertEqual(got["partials_bytes"], partials)
                        self.assertEqual(
                            got["workspace_bytes"], wgrad_streamk_workspace_nbytes(spec)
                        )
                        self.assertEqual(
                            got["is_deterministic"], reduction in ("linear", "tree")
                        )

    def test_cpp_default_pool_agrees(self):
        from kernels.common.conv_implicit_gemm_wgrad import wgrad_streamk_default_pool
        from rocke.helpers.split_k import _ARCH_NUM_CUS

        from rocke.core.arch.target import known_arches

        eng = _engine()
        # Both directions: every arch of the Python CU table, and every arch
        # rocke knows of plus a few it does not -- a key only the C table has
        # would make the C pool differ from Python's 256-CU fallback there.
        arches = {*_ARCH_NUM_CUS, *known_arches(), "gfx90a", "gfx908", "gfx1100"}
        arches |= {"gfx1200", "gfx9999"}
        for arch in sorted(arches):
            for reduction in ("linear", "tree", "atomic", "workspace"):
                with self.subTest(arch=arch, r=reduction):
                    self.assertEqual(
                        eng.conv_wgrad_streamk_default_pool(reduction, arch),
                        wgrad_streamk_default_pool(reduction, arch),
                    )

    def test_cpp_host_plan_agrees_auto_pool(self):
        # streamk_ctas=-1 (the spec default) resolves to the arch's CU-based
        # pool in both engines.
        eng = _engine()
        split = 0
        for arch, edge in (("gfx950", 32), ("gfx942", 16)):
            for problem in (_DENSE, _GROUPED, _SPLITS):
                for mode in ("dp_sk", "persistent"):
                    for reduction, dtype_d in (
                        ("linear", "fp16"),
                        ("tree", "fp16"),
                        ("atomic", "fp32"),
                        ("workspace", "fp16"),
                    ):
                        spec = _spec(
                            problem,
                            dtype_d=dtype_d,
                            edge=edge,
                            streamk=mode,
                            streamk_reduction=reduction,
                            streamk_ctas=-1,
                        )
                        got = eng.conv_wgrad_streamk_host(_spec_dict(spec), arch)
                        flags, partials = wgrad_streamk_workspace_layout(
                            spec, arch=arch
                        )
                        with self.subTest(
                            arch=arch, g=problem.groups, mode=mode, r=reduction
                        ):
                            self.assertEqual(
                                tuple(got["grid"]), wgrad_streamk_grid(spec, arch=arch)
                            )
                            self.assertEqual(got["flags_bytes"], flags)
                            self.assertEqual(got["partials_bytes"], partials)
                            self.assertEqual(
                                got["workspace_bytes"],
                                wgrad_streamk_workspace_nbytes(spec, arch=arch),
                            )
                            if problem is _SPLITS and reduction in ("linear", "tree"):
                                # 459 tiles over the default pool: every CTA is a
                                # stream-K CTA, so both regions are non-empty.
                                pool = 256 if arch == "gfx950" else 304
                                self.assertEqual(got["grid"][0], pool)
                                self.assertEqual(flags, pool * 4 * 4)
                                self.assertEqual(partials, pool * 64 * 64 * 4)
                                split += 1
        self.assertEqual(split, 2 * 2 * 2)
        # The Python helpers default to gfx950; the C++ binding takes no default
        # target, so it has to be named, and an empty one is refused.
        spec = _spec(streamk="persistent", streamk_ctas=-1)
        with self.assertRaisesRegex(ValueError, "explicit gfx target"):
            eng.conv_wgrad_streamk_host(_spec_dict(spec), "")
        got = eng.conv_wgrad_streamk_host(_spec_dict(spec), "gfx950")
        self.assertEqual(tuple(got["grid"]), wgrad_streamk_grid(spec))


class TestStreamKLaunchAbi(unittest.TestCase):
    """The stream-K kernarg ABI and the launch values that fill it."""

    def _args(self, problem=_DENSE):
        return ConvArgs.from_problem(
            problem, direction="wgrad", tile_m=64, tile_n=64, tile_k=32
        )

    def test_extras_follow_the_unchanged_prefix(self):
        plain = conv_arg_names(direction="wgrad", is_3d=True)
        for red in ("atomic", "workspace", "linear", "tree"):
            sk = conv_arg_names(direction="wgrad", is_3d=True, streamk=red)
            with self.subTest(reduction=red):
                head = plain[:-2]
                if red == "workspace":
                    head = head + [("ws_ptr", "f32*"), ("ws_bytes", "i32")]
                self.assertEqual(sk[: len(head) + 2], head + plain[-2:])
                tail = [n for n, _ in sk[len(head) + 2 :]]
                bufs = ["sk_flags", "sk_partials"] if red in ("linear", "tree") else []
                self.assertEqual(tail, bufs + list(_STREAMK_SCALARS))

    def test_abi_rejects(self):
        with self.assertRaises(ValueError):
            conv_arg_names(direction="fwd", streamk="linear")
        with self.assertRaises(ValueError):
            conv_arg_names(direction="wgrad", streamk="sideways")
        with self.assertRaises(ValueError):
            conv_arg_names(direction="wgrad", two_stage=True, streamk="workspace")

    def test_values_are_the_partition(self):
        for problem in (_DENSE, _GROUPED):
            for persistent in (False, True):
                for pool in (5, 6, 7, 1024):
                    sk = StreamKLaunch(pool=pool, persistent=persistent)
                    args = self._args(problem)
                    v = args.to_launch_values(
                        1, 2, 3, 4, 5, 6, streamk=sk, sk_flags=7, sk_partials=8
                    )
                    part = args.streamk_partition(pool, persistent=persistent)
                    with self.subTest(g=problem.groups, pers=persistent, pool=pool):
                        self.assertEqual(v["sk_iters_per_tile"], part.iters_per_tile)
                        self.assertEqual(v["sk_dp_tiles"], part.dp_tiles)
                        self.assertEqual(v["sk_ctas"], part.sk_ctas)
                        self.assertEqual(v["sk_iters_per_cta"], part.iters_per_sk_cta)
                        self.assertEqual(v["sk_extra_iters"], part.extra_iters)
                        self.assertEqual(v["sk_pool"], pool)
                        self.assertEqual(v["sk_tree_rounds"], part.tree_rounds)
                        self.assertEqual((v["sk_flags"], v["sk_partials"]), (7, 8))
                        self.assertEqual(args.streamk_grid(sk), (part.grid_size, 1, 1))

    def test_launch_value_contract(self):
        args = self._args()
        with self.assertRaisesRegex(ValueError, "sk_flags must be passed"):
            args.to_launch_values(1, 2, 3, 4, 5, 6, streamk=StreamKLaunch(pool=6))
        with self.assertRaisesRegex(ValueError, "ws_ptr must be passed"):
            args.to_launch_values(
                1, 2, 3, 4, 5, 6, streamk=StreamKLaunch(pool=6, reduction="workspace")
            )
        with self.assertRaisesRegex(ValueError, "split_k=1"):
            args.to_launch_values(
                1,
                2,
                3,
                4,
                5,
                6,
                split_k=2,
                streamk=StreamKLaunch(pool=6, reduction="atomic"),
            )
        with self.assertRaisesRegex(ValueError, "not arguments"):
            args.to_launch_values(
                1,
                2,
                3,
                4,
                5,
                6,
                sk_flags=7,
                streamk=StreamKLaunch(pool=6, reduction="atomic"),
            )
        with self.assertRaises(ValueError):
            StreamKLaunch(pool=0)

    def test_launch_shape_overrides_the_spec_problem(self):
        # One binary, many shapes: the helpers size the launch from `problem`.
        other = ConvProblem(N=8, Hi=16, Wi=16, C=32, K=96, Y=3, X=3, pH=1, pW=1)
        spec = _spec()
        self.assertEqual(
            wgrad_streamk_partition(spec, problem=other),
            self._args(other).streamk_partition(6),
        )
        self.assertNotEqual(
            wgrad_streamk_partition(spec, problem=other), wgrad_streamk_partition(spec)
        )

    def test_launch_shape_must_keep_the_built_modes(self):
        # The kernel bakes grouping, 2-D/3-D, pointwise addressing and its
        # channel vector widths; a launch shape that changes any is rejected.
        base = {"N": 2, "Hi": 12, "Wi": 12, "Y": 3, "X": 3, "pH": 1, "pW": 1}
        spec = _spec()  # _DENSE: groups=1, 2-D, 3x3, kpg=96, cpg=32
        for needle, other in (
            ("grouping", ConvProblem(C=32, K=96, groups=2, **base)),
            ("is_3d", ConvProblem(C=32, K=96, Di=4, Z=1, sD=1, pD=0, dD=1, **base)),
            ("is_pointwise", ConvProblem(N=2, Hi=12, Wi=12, C=32, K=96, Y=1, X=1)),
            (
                "kpg=36 is not a multiple of the built 8-wide dY",
                ConvProblem(C=32, K=36, **base),
            ),
            (
                "cpg=20 is not a multiple of the built 8-wide X",
                ConvProblem(C=20, K=96, **base),
            ),
        ):
            with self.subTest(needle=needle):
                for helper in (wgrad_streamk_partition, wgrad_streamk_workspace_nbytes):
                    with self.assertRaises(ValueError) as cm:
                        helper(spec, problem=other)
                    self.assertIn(needle, str(cm.exception))
        # X and dW both run along cpg; the error names the operand that failed.
        narrow_b = _spec(vector_size_b=4)
        with self.assertRaises(ValueError) as cm:
            wgrad_streamk_partition(narrow_b, problem=ConvProblem(C=20, K=96, **base))
        self.assertIn(
            "cpg=20 is not a multiple of the built 8-wide dW", str(cm.exception)
        )
        # Wider channel runs than the build keep the same code path.
        wgrad_streamk_partition(spec, problem=ConvProblem(C=64, K=192, **base))


class TestStreamKDualEngine(unittest.TestCase):
    """The C++ builder lowers every stream-K variant to the Python engine's bytes."""

    def test_lowered_ir_is_byte_identical(self):
        from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

        eng = _engine()
        cases = [
            ("gfx950", _DENSE, "dp_sk", "linear", "fp16", 32),
            ("gfx950", _DENSE, "dp_sk", "tree", "fp16", 32),
            ("gfx950", _GROUPED, "persistent", "linear", "bf16", 32),
            ("gfx950", _GROUPED, "dp_sk", "atomic", "fp32", 32),
            ("gfx942", _GROUPED, "persistent", "tree", "fp16", 16),
            ("gfx942", _DENSE, "persistent", "workspace", "fp16", 16),
        ]
        for arch, problem, mode, reduction, dtype_d, edge in cases:
            spec = _spec(
                problem,
                dtype_d=dtype_d,
                edge=edge,
                streamk=mode,
                streamk_reduction=reduction,
            )
            with self.subTest(arch=arch, g=problem.groups, mode=mode, r=reduction):
                py_ll = _lower_kernel_to_llvm_python(
                    build_implicit_gemm_conv_wgrad(spec, arch=arch), arch=arch
                )
                self.assertEqual(
                    eng.conv_wgrad_lower_llvm(_spec_dict(spec), arch), py_ll
                )

    def test_cpp_builds_auto_pool(self):
        # The pool is not part of the kernel, so the C++ builder takes -1 too.
        from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

        eng = _engine()
        spec = _spec(streamk_ctas=-1)
        py_ll = _lower_kernel_to_llvm_python(
            build_implicit_gemm_conv_wgrad(spec, arch="gfx950"), arch="gfx950"
        )
        self.assertEqual(eng.conv_wgrad_lower_llvm(_spec_dict(spec), "gfx950"), py_ll)


if __name__ == "__main__":
    unittest.main(verbosity=2)
