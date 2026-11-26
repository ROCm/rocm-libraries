# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Conv + per-output-channel bias forward (``ImplicitGemmConvSpec.bias``).

The kernel computes ``round(acc_fp32 + bias[g*kpg + k])``: the bias is added
to the fp32 accumulators before the accumulator epilogue and the one rounding
to the output dtype.

CPU: the kernarg tail, the launch values, the name tag, the identity hash and
label, the validator refusals (both engines, reason strings included) and the
fragment layouts the bias add relies on. GPU (gfx942 / gfx950): 2-D and 3-D,
grouped, depthwise-merged and pointwise problems, fp16 / bf16 / fp32, the
default, cshuffle and register-transpose stores, against the fp32 torch
convolution plus the bias, rounded once to the output dtype.
"""

from __future__ import annotations

import ctypes
import importlib.util
import unittest
from dataclasses import replace

import pytest

_HAS_TORCH = importlib.util.find_spec("torch") is not None


def _problem(**kw):
    from kernels.common._conv_implicit_gemm_common import ConvProblem

    base = dict(N=2, Hi=9, Wi=7, C=32, K=40, Y=3, X=3, pH=1, pW=1)
    base.update(kw)
    return ConvProblem(**base)


def _spec(problem, *, dtype="fp16", tile=(64, 64, 32, 2, 2, 16, 16), **kw):
    from kernels.common.conv_implicit_gemm import ConvDataSpec, ImplicitGemmConvSpec

    tm, tn, tk, wm, wn, wt, wk = tile
    kw.setdefault("pipeline", "mem")
    kw.setdefault("epilogue", "cshuffle")
    if kw["epilogue"] == "default":
        kw.setdefault("vector_size_c", 1)
    return ImplicitGemmConvSpec(
        problem=problem,
        name="test_fwd_bias",
        data=ConvDataSpec(dtype_a=dtype, dtype_b=dtype, dtype_d=dtype),
        tile_m=tm,
        tile_n=tn,
        tile_k=tk,
        warp_m=wm,
        warp_n=wn,
        warp_tile_m=wt,
        warp_tile_n=wt,
        warp_tile_k=wk,
        groups=problem.groups,
        bias=True,
        **kw,
    )


def _cpp_is_valid(spec, arch: str):
    from kernels.conv.engine_bridge import (
        ENGINE_MODULE,
        conv_implicit_gemm_spec_to_dict,
    )
    from rocke.core.backend import BackendError, import_engine

    try:
        eng = import_engine(ENGINE_MODULE)
    except BackendError as e:
        raise unittest.SkipTest(f"C++ engine not importable: {e}") from None
    return tuple(
        eng.conv_implicit_gemm_is_valid_spec(
            conv_implicit_gemm_spec_to_dict(spec), arch
        )
    )


class TestFwdBiasAbi(unittest.TestCase):
    def test_tail_follows_every_other_arg(self):
        from kernels.common.conv_abi import conv_arg_names, conv_bias_arg_names

        tail = [("Bias", "bias"), ("Bias_bytes", "i32")]
        self.assertEqual(conv_bias_arg_names(), tail)
        for kw in (
            {},
            dict(is_3d=True),
            dict(two_stage=True),
            dict(is_3d=True, split_k=True, split_tail=True, split_det=True),
        ):
            with self.subTest(**kw):
                plain = conv_arg_names(direction="fwd", **kw)
                self.assertEqual(
                    conv_arg_names(direction="fwd", bias=True, **kw), plain + tail
                )

    def test_signature_types(self):
        from kernels.common.conv_abi import conv_args_signature

        for dt, ir in (("fp16", "f16"), ("bf16", "bf16"), ("fp32", "f32")):
            sig = conv_args_signature(dt, bias=True)
            self.assertEqual(
                sig[-2:],
                [
                    {"name": "Bias", "type": f"ptr<{ir}, global>", "size_bytes": 8},
                    {"name": "Bias_bytes", "type": "i32", "size_bytes": 4},
                ],
            )
            self.assertEqual(sig[:-2], conv_args_signature(dt))

    def test_bias_is_forward_only(self):
        from kernels.common.conv_abi import conv_arg_names

        for d in ("wgrad", "dgrad"):
            with self.assertRaises(ValueError):
                conv_arg_names(direction=d, bias=True)

    def test_launch_values(self):
        from kernels.common.conv_args import ConvArgs

        args = ConvArgs.from_problem(_problem(), tile_m=64, tile_n=64)
        plain = args.to_launch_values(1, 2, 3, 10, 20, 30)
        biased = args.to_launch_values(1, 2, 3, 10, 20, 30, bias_ptr=4, bias_bytes=80)
        self.assertEqual(list(biased)[-2:], ["Bias", "Bias_bytes"])
        self.assertEqual((biased["Bias"], biased["Bias_bytes"]), (4, 80))
        self.assertEqual({k: biased[k] for k in plain}, plain)
        self.assertNotIn("Bias", plain)

    def test_launch_values_refuse_a_short_bias(self):
        """Bias_bytes bounds the kernel's bias reads (a channel past it reads
        0), so it must cover K values of the narrowest (2-byte) output."""
        from kernels.common.conv_args import ConvArgs

        args = ConvArgs.from_problem(_problem(), tile_m=64, tile_n=64)
        for short in (None, 0, 2 * 40 - 1, 1 << 31):
            with self.subTest(bias_bytes=short):
                with self.assertRaisesRegex(ValueError, "K=40 channels needs"):
                    args.to_launch_values(
                        1, 2, 3, 10, 20, 30, bias_ptr=4, bias_bytes=short
                    )
        with self.assertRaisesRegex(ValueError, "Bias_bytes"):
            args.to_launch_values(1, 2, 3, 10, 20, 30, bias_bytes=80)


class TestFwdBiasSpec(unittest.TestCase):
    def test_name_tag_after_the_acc_epilogue_tag(self):
        from kernels.common.conv_implicit_gemm import ConvAccumulatorEpilogue

        spec = _spec(_problem())
        plain = replace(spec, bias=False)
        self.assertNotIn("cbias", plain.kernel_name())
        self.assertEqual(
            spec.kernel_name(),
            plain.kernel_name().replace("_mem_cshuffle", "_mem_cshuffle_cbias"),
        )
        epi = replace(spec, acc_epilogue=ConvAccumulatorEpilogue(relu=True))
        self.assertIn("_mem_cshuffle_epi_relu_cbias", epi.kernel_name())

    def test_params(self):
        from kernels.common.conv_implicit_gemm import build_implicit_gemm_conv

        spec = _spec(_problem())
        names = [p.name for p in build_implicit_gemm_conv(spec, arch="gfx942").params]
        plain = [
            p.name
            for p in build_implicit_gemm_conv(
                replace(spec, bias=False), arch="gfx942"
            ).params
        ]
        self.assertEqual(names, plain + ["Bias", "Bias_bytes"])

    def test_identity_hash_and_label(self):
        from kernels.conv.kernel_identity import KernelIdentity

        base = dict(
            arch="gfx942",
            direction="fwd",
            algorithm="implicit_gemm",
            dtype_a="fp16",
            dtype_b="fp16",
            dtype_d="fp16",
            tile_m=64,
            tile_n=64,
            tile_k=32,
            warp_m=2,
            warp_n=2,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=16,
            pipeline="mem",
            epilogue="cshuffle",
            wave_size=64,
            vector_size_a=8,
            vector_size_b=8,
            vector_size_c=8,
        )
        plain = KernelIdentity(**base)
        biased = KernelIdentity(**base, bias=True)
        self.assertNotEqual(plain.stable_hash(), biased.stable_hash())
        self.assertNotIn("cbias", plain.label())
        self.assertIn("-cbias-", biased.label())
        self.assertEqual(KernelIdentity.from_label(biased.label()), biased)
        self.assertEqual(KernelIdentity.from_label(plain.label()), plain)

    def test_engine_bridge_forwards_bias(self):
        from kernels.conv.engine_bridge import conv_implicit_gemm_spec_to_dict

        spec = _spec(_problem())
        self.assertIs(conv_implicit_gemm_spec_to_dict(spec)["bias"], True)
        self.assertNotIn(
            "bias", conv_implicit_gemm_spec_to_dict(replace(spec, bias=False))
        )


class TestFwdBiasKernelCache(unittest.TestCase):
    """A cached conv + bias binary (the opt-in ROCKE_AOT_FWD_BIAS axis) is
    never offered for a cached problem: the problem carries no bias tensor, so
    its Bias / Bias_bytes kernargs would be left unset."""

    def test_the_cache_offers_the_plain_binary_only(self):
        import tempfile
        from pathlib import Path

        from benchmarks.common.kernel_cache import KernelCache
        from kernels.conv.kernel_identity import (
            KernelIdentity,
            current_llvm_flavor,
            identity_capability_key,
            identity_supports_problem,
        )

        plain = KernelIdentity(
            arch="gfx942",
            direction="fwd",
            algorithm="implicit_gemm",
            dtype_a="fp16",
            dtype_b="fp16",
            dtype_d="fp16",
            tile_m=64,
            tile_n=64,
            tile_k=32,
            warp_m=2,
            warp_n=2,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=16,
            pipeline="mem",
            epilogue="cshuffle",
            wave_size=64,
            vector_size_a=8,
            vector_size_b=8,
            vector_size_c=8,
            llvm_flavor=current_llvm_flavor(),
        )
        biased = replace(plain, bias=True)
        problem = _problem()
        # The two are separate cache groups (the cache answers once per key).
        self.assertNotEqual(
            identity_capability_key(plain), identity_capability_key(biased)
        )
        with tempfile.TemporaryDirectory() as root:
            cache = KernelCache(Path(root), "gfx942")
            # The plain binary's answer is the shape predicate's, unchanged.
            self.assertEqual(
                cache.supports_problem(plain, problem),
                identity_supports_problem(plain, problem),
            )
            self.assertTrue(cache.supports_problem(plain, problem)[0])
            ok, why = cache.supports_problem(biased, problem)
            self.assertFalse(ok)
            self.assertIn("no bias tensor", why)
            # The bias twin heads its own group even when cached first.
            cache.put(biased, b"bias", {"kernel_name": "k_bias"})
            cache.put(plain, b"plain", {"kernel_name": "k_plain"})
            picked = [i for i, _, _ in cache.compatible(problem, direction="fwd")]
            self.assertEqual(picked, [plain])


class TestFwdBiasValidator(unittest.TestCase):
    def _refused(self, spec, arch, needle):
        from kernels.common.conv_implicit_gemm import is_valid_spec

        ok, why = is_valid_spec(spec, arch)
        self.assertFalse(ok)
        self.assertIn(needle, why)
        with self.assertRaisesRegex(ValueError, needle):
            spec.validate()
        self.assertEqual(_cpp_is_valid(spec, arch), (False, why))

    def test_split_k_without_an_owner_refused(self):
        p3 = _problem(Di=5, Z=3, sD=1, pD=1, dD=1, C=64, K=64)
        # The 3-D split is the atomic one; split_det gives it an owner.
        for kw in (dict(split_k=2), dict(split_k=-1)):
            with self.subTest(**kw):
                self._refused(
                    _spec(p3, dtype="bf16", **kw), "gfx942", "atomic split-K forward"
                )
        # The 2-D atomic split (split_atomic).
        self._refused(
            _spec(_problem(), split_k=2, split_atomic=True),
            "gfx942",
            "atomic split-K forward",
        )
        # The stream-K / hybrid segment loops, split_det or not.
        for kw in (
            dict(split_hybrid=4, split_det=1),
            dict(split_hybrid=4, split_det=3),
            dict(split_streamk=True, split_det=1),
        ):
            with self.subTest(**kw):
                self._refused(
                    _spec(p3, dtype="bf16", tile=_T256, split_k=2, **kw),
                    "gfx942",
                    "stream-K split-K forward",
                )

    def test_plain_spec_accepted_by_both_engines(self):
        from kernels.common.conv_implicit_gemm import is_valid_spec

        spec = _spec(_problem())
        self.assertTrue(is_valid_spec(spec, "gfx942")[0])
        self.assertTrue(_cpp_is_valid(spec, "gfx942")[0])


class TestFwdBiasFragmentLayouts(unittest.TestCase):
    """The bias add loads one value per (fragment column, lane): every slot of
    a lane's fp32 accumulator fragment must sit in one output column, and the
    columns of an atom must be 0..n-1, for every MMA op the forward uses."""

    class _IntBuilder:
        def const_i32(self, v):
            return v

        def add(self, a, b):
            return a + b

        def sub(self, a, b):
            return a - b

        def mul(self, a, b):
            return a * b

        def div(self, a, b):
            return a // b

        def mod(self, a, b):
            return a % b

        def land(self, a, b):
            return a & b

        def lshr(self, a, b):
            return a >> b

        def shl(self, a, b):
            return a << b

    def test_slot_invariant_columns(self):
        from kernels.common.conv_implicit_gemm import _resolve_conv_op

        cases = [
            ("gfx942", "fp16", (16, 16, 16)),
            ("gfx942", "bf16", (32, 32, 8)),
            ("gfx950", "fp16", (32, 32, 16)),
            ("gfx950", "bf16", (16, 16, 32)),
            ("gfx1250", "fp16", (16, 16, 32)),
            ("gfx1250", "bf16", (16, 16, 32)),
            ("gfx1151", "fp16", (16, 16, 16)),
        ]
        ib = self._IntBuilder()
        for arch, dt, (m, n, k) in cases:
            with self.subTest(arch=arch, dtype=dt, atom=(m, n, k)):
                wave = 32 if arch in ("gfx1250", "gfx1151") else 64
                spec = _spec(
                    _problem(),
                    dtype=dt,
                    tile=(2 * m, 2 * n, 2 * k, 1, 1, m, k),
                    wave_size=wave,
                )
                op = _resolve_conv_op(spec, arch)
                lay = op.dst_layout()
                cols = []
                for lane in range(lay.wave_size):
                    c = {lay.coord(ib, lane, i)[1] for i in range(lay.frag_len)}
                    self.assertEqual(len(c), 1, f"lane {lane}: columns {c}")
                    cols.append(c.pop())
                self.assertEqual(sorted(set(cols)), list(range(n)))


def _p3_det(**kw):
    return _problem(Di=5, Z=3, sD=1, pD=1, dD=1, C=64, K=64, **kw)


def _strip_bias_tail(ll: str, name: str, plain_name: str) -> str:
    """``ll`` of a bias kernel with its name and its trailing Bias /
    Bias_bytes parameters taken out."""
    import re

    ll = ll.replace(name, plain_name)
    return re.sub(r", ptr addrspace\(1\)[^,]*%Bias, i32 %Bias_bytes\)", ")", ll)


class TestFwdBiasSplitK(unittest.TestCase):
    """The split-K forward forms that have one owner of every output value
    take the bias, added exactly once (fwd_bias_stage)."""

    def _forms(self):
        p2, p3 = _problem(), _p3_det()
        wmma = dict(wave_size=32, tile=(32, 32, 32, 1, 1, 16, 32))
        return [
            # (spec, arch, stage)
            (_spec(p2, split_k=2), "gfx942", "reduce"),
            (_spec(p2, split_k=2, epilogue="default"), "gfx950", "reduce"),
            (_spec(p2, split_k=-1), "gfx950", "reduce"),
            (_spec(p2, dtype="bf16", split_k=2, **wmma), "gfx1250", "reduce"),
            (
                _spec(
                    _problem(K=64), dtype="bf16", split_k=2, split_k_cluster=4, **wmma
                ),
                "gfx1250",
                "cluster",
            ),
            (
                _spec(p3, dtype="bf16", tile=_T256, split_k=2, split_det=1),
                "gfx942",
                "owner",
            ),
            (
                _spec(
                    p3,
                    dtype="bf16",
                    tile=_T256,
                    split_k=2,
                    split_det=2,
                    split_tail=True,
                ),
                "gfx942",
                "owner",
            ),
            (
                _spec(p3, dtype="bf16", tile=_T256, split_k=2, split_det=3),
                "gfx942",
                "reduce",
            ),
        ]

    def test_stage(self):
        from kernels.common.conv_implicit_gemm import fwd_bias_stage

        self.assertEqual(fwd_bias_stage(_spec(_problem())), "epilogue")
        self.assertEqual(
            fwd_bias_stage(replace(_spec(_problem(), split_k=2), bias=False)), ""
        )
        for spec, _, stage in self._forms():
            with self.subTest(name=spec.kernel_name()):
                self.assertEqual(fwd_bias_stage(spec), stage)

    def test_accepted_by_both_engines(self):
        from kernels.common.conv_implicit_gemm import is_valid_spec

        for spec, arch, _ in self._forms():
            if spec.split_k == -1:
                continue  # resolved before validation
            with self.subTest(name=spec.kernel_name(), arch=arch):
                self.assertEqual(is_valid_spec(spec, arch), (True, "ok"))
                self.assertEqual(_cpp_is_valid(spec, arch), (True, "ok"))
                spec.validate()

    def test_split_forms_take_the_bias_tail(self):
        from kernels.common.conv_implicit_gemm import build_implicit_gemm_conv

        for spec, arch, _ in self._forms():
            if spec.split_k == -1 or spec.split_k_cluster:
                continue  # cluster_barrier needs llvm23; parity covers it
            with self.subTest(name=spec.kernel_name()):
                names = [
                    p.name for p in build_implicit_gemm_conv(spec, arch=arch).params
                ]
                self.assertEqual(names[-2:], ["Bias", "Bias_bytes"])

    def test_reduce_form_kernel_does_not_read_the_bias(self):
        # The two-stage and split_det 3 kernels store raw partial sums: the
        # bias tail is in their ABI (one kernarg block for both launches), the
        # code is the kernel's without bias.
        from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

        from kernels.common.conv_implicit_gemm import build_implicit_gemm_conv

        for spec, arch, stage in self._forms():
            if stage != "reduce" or spec.split_k == -1 or arch == "gfx1250":
                continue
            with self.subTest(name=spec.kernel_name()):
                plain = replace(spec, bias=False)

                def ll(s):
                    return _lower_kernel_to_llvm_python(
                        build_implicit_gemm_conv(s, arch=arch), arch=arch
                    )

                biased = ll(spec)
                self.assertIn("%Bias_bytes", biased)
                self.assertEqual(
                    _strip_bias_tail(biased, spec.kernel_name(), plain.kernel_name()),
                    ll(plain),
                )

    def test_owner_form_adds_the_bias(self):
        from kernels.common.conv_implicit_gemm import build_implicit_gemm_conv
        from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

        for spec, arch, stage in self._forms():
            if stage != "owner":
                continue
            with self.subTest(name=spec.kernel_name()):
                ll = _lower_kernel_to_llvm_python(
                    build_implicit_gemm_conv(spec, arch=arch), arch=arch
                )
                self.assertIn("%Bias_bytes", ll.split("{", 1)[1])

    def test_launch_values_end_with_the_bias(self):
        from kernels.common.conv_args import ConvArgs

        p3 = _p3_det()
        args = ConvArgs.from_problem(p3, tile_m=256, tile_n=64, tile_k=32)
        v = args.to_launch_values(
            1,
            2,
            3,
            10,
            20,
            30,
            split_k=2,
            sk_flags=5,
            sk_partials=6,
            bias_ptr=7,
            bias_bytes=2 * p3.K,
        )
        self.assertEqual(
            list(v)[-4:], ["sk_flags", "sk_partials", "Bias", "Bias_bytes"]
        )
        args2 = ConvArgs.from_problem(_problem(), tile_m=64, tile_n=64, tile_k=32)
        v2 = args2.to_launch_values(
            1,
            2,
            3,
            10,
            20,
            30,
            split_k=2,
            ws_ptr=8,
            ws_bytes=9,
            bias_ptr=7,
            bias_bytes=2 * 40,
        )
        self.assertEqual(list(v2)[-6:-2], ["ws_ptr", "ws_bytes", "ks", "ks_count"])
        self.assertEqual(list(v2)[-2:], ["Bias", "Bias_bytes"])


class TestFwdBiasReduceKernels(unittest.TestCase):
    """The fold kernels of the "reduce" forms: Stage 2 of the two-stage split
    (conv_fwd_split_k_reduce) and the split_det 3 fold
    (conv_wgrad_workspace_reduce) add the bias to the folded fp32 sum."""

    def test_stage2_name_abi(self):
        from kernels.common.conv_fwd_split_k_reduce import (
            FwdSplitKReduceSpec,
            build_conv_fwd_split_k_reduce,
            fwd_split_k_reduce_signature,
        )

        spec = FwdSplitKReduceSpec(dtype_d="fp16", vec=4, degree=3, bias=True)
        plain = replace(spec, bias=False)
        self.assertEqual(
            spec.kernel_name(), plain.kernel_name().replace("_fp16", "_cbias_fp16")
        )
        names = [p.name for p in build_conv_fwd_split_k_reduce(spec).params]
        plain_names = [p.name for p in build_conv_fwd_split_k_reduce(plain).params]
        self.assertEqual(names, plain_names + ["K", "Bias", "Bias_bytes"])
        sig = fwd_split_k_reduce_signature(spec)
        self.assertEqual([e["name"] for e in sig], names)
        self.assertEqual(sig[-2]["type"], "ptr<f16, global>")

    def test_det3_fold_name_abi(self):
        from kernels.common._conv_implicit_gemm_common import ConvProblem
        from kernels.common.conv_wgrad_workspace_reduce import (
            WgradReduceSpec,
            build_conv_wgrad_workspace_reduce,
            wgrad_reduce_signature,
        )

        prob = ConvProblem(N=1, Hi=1, Wi=1, C=96, K=144, Y=1, X=1)
        for kw in (dict(), dict(vec=4), dict(vec=4, ws_replicas=9, fold_loop=4)):
            with self.subTest(**kw):
                spec = WgradReduceSpec(
                    problem=prob, dtype_d="bf16", ws_replicas=3, bias=True
                )
                spec = replace(spec, **kw)
                plain = replace(spec, bias=False)
                self.assertTrue(spec.kernel_name().endswith("_cbias_bf16"))
                names = [p.name for p in build_conv_wgrad_workspace_reduce(spec).params]
                plain_names = [
                    p.name for p in build_conv_wgrad_workspace_reduce(plain).params
                ]
                self.assertEqual(names, plain_names + ["Bias", "Bias_bytes"])
                self.assertEqual(
                    [e["name"] for e in wgrad_reduce_signature(spec)], names
                )

    def test_det3_fold_refusals(self):
        from kernels.common._conv_implicit_gemm_common import ConvProblem
        from kernels.common.conv_wgrad_workspace_reduce import (
            WgradReduceSpec,
            build_conv_wgrad_workspace_reduce,
        )

        prob = ConvProblem(N=1, Hi=1, Wi=1, C=96, K=144, Y=1, X=1)
        for kw in (dict(groups=2), dict(vec=4, rezero=1)):
            with self.subTest(**kw):
                spec = WgradReduceSpec(problem=prob, dtype_d="bf16", bias=True, **kw)
                with self.assertRaisesRegex(ValueError, "split_det 3 fold"):
                    build_conv_wgrad_workspace_reduce(spec)

    def test_cpp_binding_agrees(self):
        from rocke.core.backend import BackendError, import_engine
        from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

        from kernels.common.conv_fwd_split_k_reduce import (
            FwdSplitKReduceSpec,
            build_conv_fwd_split_k_reduce,
        )
        from kernels.conv.engine_bridge import ENGINE_MODULE

        try:
            eng = import_engine(ENGINE_MODULE)
        except BackendError as e:
            raise unittest.SkipTest(f"C++ engine not importable: {e}") from None
        for kw in (dict(vec=4), dict(vec=1, dtype_d="fp16"), dict(vec=2, degree=3)):
            with self.subTest(**kw):
                spec = FwdSplitKReduceSpec(bias=True, **kw)
                d = dict(
                    dtype_d=spec.dtype_d,
                    vec=spec.vec,
                    block_size=spec.block_size,
                    degree=spec.degree,
                    bias=True,
                )
                self.assertEqual(
                    eng.conv_fwd_split_k_reduce_lower_llvm(d, "gfx942"),
                    _lower_kernel_to_llvm_python(
                        build_conv_fwd_split_k_reduce(spec, arch="gfx942"),
                        arch="gfx942",
                    ),
                )


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------


def _gpu_skip() -> str:
    if not _HAS_TORCH:
        return "torch not importable"
    try:
        from rocke.runtime.hip_module import get_device_arch

        arch = get_device_arch()
    except Exception as e:  # pragma: no cover - no device
        return f"no GPU: {e}"
    if not arch or arch.split(":")[0] not in ("gfx942", "gfx950"):
        return f"the bias GPU cases use MFMA tiles of gfx942 / gfx950; device is {arch}"
    return ""


_GPU_SKIP = _gpu_skip()
# Peak-relative error bound of the forward correctness tests.
_TOL = {"fp16": 5e-2, "bf16": 5e-2, "fp32": 1e-4}


def _u8(t):
    return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())


def _run(spec):
    """Build, launch and verify ``spec``; returns the peak-relative error
    against ``round(conv_fp32 + bias)`` and the same error of a reference
    without the bias (which must be large, or the case would not tell)."""
    import torch
    from builders.common.conv_reference import conv_reference
    from kernels.common.conv_abi import conv_args_signature
    from kernels.common.conv_args import ConvArgs
    from kernels.common.conv_implicit_gemm import (
        build_implicit_gemm_conv,
        is_valid_spec_for_problem,
    )
    from rocke import compile_kernel
    from rocke.runtime.hip_module import Runtime, get_device_arch
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    arch = get_device_arch().split(":")[0]
    p = spec.problem
    ok, why = is_valid_spec_for_problem(spec, p, arch)
    assert ok, why
    art = compile_kernel(build_implicit_gemm_conv(spec, arch=arch), arch=arch)
    dt = spec.data.dtype_d
    tdt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dt]
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=art.kernel_name,
        signature=conv_args_signature(dt, is_3d=p.is_3d, bias=True),
    )
    depth = (p.Di,) if p.is_3d else ()
    filt = (p.Z,) if p.is_3d else ()
    out_depth = (p.Do,) if p.is_3d else ()
    torch.manual_seed(0)
    A = torch.empty(p.N, *depth, p.Hi, p.Wi, p.C).uniform_(-1, 1).to(tdt)
    B = torch.empty(p.K, *filt, p.Y, p.X, p.cpg).uniform_(-1, 1).to(tdt)
    bias = torch.empty(p.K).uniform_(-8, 8).to(tdt)
    D = torch.empty(p.N, *out_depth, p.Ho, p.Wo, p.K, dtype=tdt)
    conv = conv_reference(A, B, p).cpu().float()
    ref = (conv + bias.float()).to(tdt).float()
    acc = spec.acc_epilogue
    if not acc.is_identity():
        t = conv + bias.float() + acc.bias
        t = t * acc.scale
        if acc.relu:
            t = t.clamp(min=0.0)
        ref = t.to(tdt).float()
    rt = Runtime()
    ptrs = [rt.alloc(t.nbytes) for t in (A, B, D, bias)]
    try:
        rt.memcpy_h2d(ptrs[0], _u8(A), A.nbytes)
        rt.memcpy_h2d(ptrs[1], _u8(B), B.nbytes)
        rt.memcpy_h2d(ptrs[3], _u8(bias), bias.nbytes)
        # NaN in every 16-bit dtype: an output never written fails.
        rt.memset(ptrs[2], 0xFF, D.nbytes)
        args = ConvArgs.from_problem(p, tile_m=spec.tile_m, tile_n=spec.tile_n)
        values = args.to_launch_values(
            *ptrs[:3],
            A.nbytes,
            B.nbytes,
            D.nbytes,
            bias_ptr=ptrs[3],
            bias_bytes=bias.nbytes,
        )
        grid = (
            -(-spec.grid_N_gemm // spec.tile_n),
            -(-spec.grid_M // spec.tile_m),
            spec.grid_groups,
        )
        launcher(
            values,
            config=LaunchConfig(
                grid=grid, block=(spec.launch_block_size, 1, 1), fence=True
            ),
        )
        out = torch.empty_like(D)
        rt.memcpy_d2h(_u8(out), ptrs[2], D.nbytes)
    finally:
        for q in ptrs:
            rt.free(q)
    out = out.float()
    scale = ref.abs().max().clamp(min=1.0)
    err = float((out - ref).abs().max() / scale)
    no_bias = float((out - conv.to(tdt).float()).abs().max() / scale)
    return err, no_bias


def _p3(**kw):
    base = dict(N=1, Hi=6, Wi=5, C=64, K=64, Di=5, Z=3, sD=1, pD=1, dD=1)
    base.update(kw)
    return _problem(**base)


_T256 = (256, 64, 32, 4, 1, 16, 16)


@pytest.mark.gpu
@unittest.skipIf(_GPU_SKIP, _GPU_SKIP)
class TestFwdBiasGpu(unittest.TestCase):
    def _check(self, spec):
        err, no_bias = _run(spec)
        tol = _TOL[spec.data.dtype_d]
        self.assertLess(err, tol, f"{spec.kernel_name()}: rel err {err:.3e}")
        print(f"  {spec.kernel_name()}: rel err {err:.2e} (without bias {no_bias:.2e})")
        self.assertGreater(no_bias, 3 * tol, "the bias did not change the output")

    def test_2d(self):
        for dtype in ("fp16", "bf16"):
            for epi in ("default", "cshuffle"):
                with self.subTest(dtype=dtype, epilogue=epi):
                    self._check(_spec(_problem(), dtype=dtype, epilogue=epi))
        # The 32x32 atom (16 slots per lane) and a K past the tile (40 of 64).
        self._check(_spec(_problem(), dtype="bf16", tile=(64, 64, 32, 2, 2, 32, 8)))

    def test_2d_grouped(self):
        for groups, k in ((2, 40), (4, 64)):
            with self.subTest(groups=groups):
                self._check(_spec(_problem(groups=groups, K=k), dtype="bf16"))
        # Grouped pointwise: the flat fast path.
        self._check(_spec(_problem(Y=1, X=1, pH=0, pW=0, groups=2, K=64)))

    def test_depthwise_group_merge(self):
        p = _problem(C=64, K=64, groups=64)
        self._check(_spec(p, tile=(64, 64, 64, 2, 2, 16, 16), group_merge=8))

    def test_fp32(self):
        self._check(_spec(_problem(), dtype="fp32", tile=(64, 64, 32, 2, 2, 16, 4)))

    def test_acc_epilogue_after_bias(self):
        from kernels.common.conv_implicit_gemm import ConvAccumulatorEpilogue

        self._check(
            _spec(
                _problem(),
                acc_epilogue=ConvAccumulatorEpilogue(bias=0.5, scale=0.5, relu=True),
            )
        )

    def test_3d(self):
        for dtype in ("fp16", "bf16"):
            for groups in (1, 2):
                with self.subTest(dtype=dtype, groups=groups):
                    self._check(_spec(_p3(groups=groups), dtype=dtype, tile=_T256))
        self._check(
            _spec(
                _p3(groups=2),
                dtype="bf16",
                tile=(128, 128, 32, 2, 2, 32, 8),
                epilogue="default",
            )
        )

    def test_3d_store_variants(self):
        # The register quad-transpose store and the fast packed cshuffle store.
        self._check(
            _spec(
                _p3(groups=2, Y=1, X=1, Z=1, pH=0, pW=0, pD=0),
                dtype="bf16",
                tile=(128, 64, 32, 2, 2, 16, 16),
                epi_quad_xpose=1,
            )
        )
        self._check(_spec(_p3(groups=2), dtype="bf16", tile=_T256, epi_store_fast=True))


def _run_split(spec, degree, *, full_tiles=None, launches=2):
    """Build and launch a split-K forward with bias at ``degree`` (the
    two-stage split with its bias Stage 2, or a split_det form with its bias
    fold) and verify it; returns the error pair of :func:`_run`. Every launch
    takes the one kernarg block (bias tail included) the form shares between
    its launches; D is poisoned before each launch."""
    import torch
    from builders.common.conv_reference import conv_reference
    from kernels.common._conv_implicit_gemm_common import ConvProblem
    from kernels.common.conv_abi import conv_args_signature
    from kernels.common.conv_args import ConvArgs
    from kernels.common.conv_fwd_split_k_reduce import (
        FwdSplitKReduceSpec,
        build_conv_fwd_split_k_reduce,
        fwd_split_k_reduce_grid,
        fwd_split_k_reduce_signature,
        fwd_split_k_reduce_vec,
        fwd_split_k_workspace_nbytes,
    )
    from kernels.common.conv_implicit_gemm import (
        build_implicit_gemm_conv,
        fwd_bias_stage,
        fwd_split_det_slabs,
        fwd_split_det_slots,
        is_valid_spec,
    )
    from kernels.common.conv_wgrad_workspace_reduce import (
        WgradReduceSpec,
        build_conv_wgrad_workspace_reduce,
        wgrad_reduce_grid,
        wgrad_reduce_signature,
    )
    from rocke import compile_kernel
    from rocke.runtime.hip_module import Runtime, get_device_arch
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    arch = get_device_arch().split(":")[0]
    ok, why = is_valid_spec(spec, arch)
    assert ok, why
    p = spec.problem
    det = spec.split_det
    stage = fwd_bias_stage(spec)
    dt = spec.data.dtype_d
    tdt = {"fp16": torch.float16, "bf16": torch.bfloat16}[dt]
    art = compile_kernel(build_implicit_gemm_conv(spec, arch=arch), arch=arch)
    conv_k = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=art.kernel_name,
        signature=conv_args_signature(
            dt,
            is_3d=p.is_3d,
            two_stage=not det,
            split_k=bool(det),
            split_tail=spec.split_tail,
            split_det=bool(det),
            bias=True,
        ),
    )
    depth = (p.Di,) if p.is_3d else ()
    filt = (p.Z,) if p.is_3d else ()
    out_depth = (p.Do,) if p.is_3d else ()
    torch.manual_seed(0)
    A = torch.empty(p.N, *depth, p.Hi, p.Wi, p.C).uniform_(-1, 1).to(tdt)
    B = torch.empty(p.K, *filt, p.Y, p.X, p.cpg).uniform_(-1, 1).to(tdt)
    # Wide enough to stand out of the deeper 3-D sums (a missing or doubled
    # add must miss the tolerance by far).
    bias = torch.empty(p.K).uniform_(-16, 16).to(tdt)
    D = torch.empty(p.N, *out_depth, p.Ho, p.Wo, p.K, dtype=tdt)
    conv = conv_reference(A, B, p).cpu().float()
    ref = (conv + bias.float()).to(tdt).float()
    args = ConvArgs.from_problem(
        p, tile_m=spec.tile_m, tile_n=spec.tile_n, tile_k=spec.tile_k
    )
    M = args.gemm_m
    tiles = args.num_pid_m * args.num_pid_n
    if det:
        flag_bytes = 8 * tiles * max(1, p.groups)
        if det == 3:
            slabs = fwd_split_det_slabs(tiles, degree)
            part_bytes = 4 * M * p.K * slabs
        else:
            part_bytes = (
                4
                * spec.tile_m
                * spec.tile_n
                * max(
                    1,
                    fwd_split_det_slots(
                        tiles,
                        max(1, p.groups),
                        degree,
                        split_tail=spec.split_tail,
                        full_tiles=full_tiles or 0,
                    ),
                )
            )
        ws_bytes = flag_bytes + part_bytes
    else:
        ws_bytes = fwd_split_k_workspace_nbytes(M, p.K, degree)
    rt = Runtime()
    ptrs = [rt.alloc(t.nbytes) for t in (A, B, D, bias)]
    ws = rt.alloc(ws_bytes)
    outs = []
    try:
        rt.memcpy_h2d(ptrs[0], _u8(A), A.nbytes)
        rt.memcpy_h2d(ptrs[1], _u8(B), B.nbytes)
        rt.memcpy_h2d(ptrs[3], _u8(bias), bias.nbytes)
        rt.memset(ws, 0, ws_bytes)
        split = (
            dict(
                sk_flags=int(ws),
                sk_partials=int(ws) + flag_bytes,
                full_tiles=full_tiles if spec.split_tail else None,
            )
            if det
            else dict(ws_ptr=int(ws), ws_bytes=ws_bytes)
        )
        values = args.to_launch_values(
            *ptrs[:3],
            A.nbytes,
            B.nbytes,
            D.nbytes,
            split_k=degree,
            bias_ptr=ptrs[3],
            bias_bytes=bias.nbytes,
            **split,
        )
        fold = None
        if stage == "reduce" and det == 3:
            fspec = WgradReduceSpec(
                problem=ConvProblem(N=1, Hi=1, Wi=1, C=p.K, K=M, Y=1, X=1),
                dtype_d=dt,
                ws_replicas=slabs,
                vec=4,
                bias=True,
            )
            fart = compile_kernel(build_conv_wgrad_workspace_reduce(fspec), arch=arch)
            fold = (
                KernelLauncher(
                    hsaco=fart.hsaco,
                    kernel_name=fart.kernel_name,
                    signature=wgrad_reduce_signature(fspec),
                ),
                dict(
                    ws_ptr=int(ws) + flag_bytes,
                    dw_ptr=ptrs[2],
                    wg_M=fspec.wg_M,
                    wg_N=fspec.wg_N,
                    ws_bytes=part_bytes,
                    dw_bytes=D.nbytes,
                    groups=1,
                    Bias=ptrs[3],
                    Bias_bytes=bias.nbytes,
                ),
                LaunchConfig(
                    grid=wgrad_reduce_grid(fspec),
                    block=(fspec.block_size, 1, 1),
                    fence=True,
                ),
            )
        elif stage == "reduce":
            elems = M * p.K
            rspec = FwdSplitKReduceSpec(
                dtype_d=dt, vec=fwd_split_k_reduce_vec(elems), degree=degree, bias=True
            )
            rart = compile_kernel(build_conv_fwd_split_k_reduce(rspec), arch=arch)
            fold = (
                KernelLauncher(
                    hsaco=rart.hsaco,
                    kernel_name=rart.kernel_name,
                    signature=fwd_split_k_reduce_signature(rspec),
                ),
                dict(
                    ws_ptr=int(ws),
                    d_ptr=ptrs[2],
                    elems=elems,
                    ws_bytes=ws_bytes,
                    d_bytes=D.nbytes,
                    ks_count=degree,
                    K=p.K,
                    Bias=ptrs[3],
                    Bias_bytes=bias.nbytes,
                ),
                LaunchConfig(
                    grid=fwd_split_k_reduce_grid(rspec, elems),
                    block=(rspec.block_size, 1, 1),
                    fence=True,
                ),
            )
        for _ in range(launches):
            rt.memset(ptrs[2], 0xFF, D.nbytes)
            conv_k(
                values,
                config=LaunchConfig(
                    grid=args.grid(degree),
                    block=(spec.launch_block_size, 1, 1),
                    fence=True,
                ),
            )
            if fold is not None:
                fold[0](fold[1], config=fold[2])
            out = torch.empty_like(D)
            rt.memcpy_d2h(_u8(out), ptrs[2], D.nbytes)
            outs.append(out.float())
    finally:
        for q in ptrs + [ws]:
            rt.free(q)
    for o in outs[1:]:
        assert bool((o == outs[0]).all()), "split-K bias output not reproducible"
    out = outs[0]
    scale = ref.abs().max().clamp(min=1.0)
    err = float((out - ref).abs().max() / scale)
    no_bias = float((out - conv.to(tdt).float()).abs().max() / scale)
    return err, no_bias


@pytest.mark.gpu
@unittest.skipIf(_GPU_SKIP, _GPU_SKIP)
class TestFwdBiasSplitKGpu(unittest.TestCase):
    """Each split form adds the bias exactly once: a second add (or none)
    is off by the bias, far past the tolerance."""

    def _check(self, spec, degrees, **kw):
        tol = _TOL[spec.data.dtype_d]
        for degree in degrees:
            with self.subTest(name=spec.kernel_name(), degree=degree):
                err, no_bias = _run_split(spec, degree, **kw)
                print(
                    f"  {spec.kernel_name()} s{degree}: rel err {err:.2e} "
                    f"(without bias {no_bias:.2e})"
                )
                self.assertLess(err, tol)
                self.assertGreater(no_bias, 3 * tol)

    def test_two_stage(self):
        # Stage 2 adds bias[idx % K] per element, with vec from the element
        # count as the benchmark plans it; K = 40 (vec 4 over channel quads).
        for dtype in ("fp16", "bf16"):
            for epi in ("cshuffle", "default"):
                with self.subTest(dtype=dtype, epilogue=epi):
                    self._check(
                        _spec(_problem(), dtype=dtype, epilogue=epi, split_k=2),
                        (2, 3, 5),
                    )
        # Grouped: the packed [M][G*kpg] row; K = 2 * 21 = 42 with vec 4
        # (M * K = 126 * 42): a thread's four elements cross a row end.
        self._check(
            _spec(_problem(groups=2, K=42, C=32), dtype="bf16", split_k=2), (2, 4)
        )

    def test_split_det_owner(self):
        for mode in (1, 2):
            with self.subTest(mode=mode):
                self._check(
                    _spec(
                        _p3_det(), dtype="bf16", tile=_T256, split_k=2, split_det=mode
                    ),
                    (2, 3),
                )
        # split_tail: whole-K tiles store plainly (and add the bias), the
        # tail tiles go through the owner's fold.
        self._check(
            _spec(
                _p3_det(N=2),
                dtype="bf16",
                tile=_T256,
                split_k=2,
                split_det=2,
                split_tail=True,
            ),
            (3,),
            full_tiles=1,
        )

    def test_split_det_slabs(self):
        self._check(
            _spec(_p3_det(), dtype="bf16", tile=_T256, split_k=2, split_det=3), (2, 3)
        )
        self._check(
            _spec(_p3_det(groups=2), dtype="bf16", tile=_T256, split_k=2, split_det=3),
            (2,),
        )


if __name__ == "__main__":
    unittest.main()
