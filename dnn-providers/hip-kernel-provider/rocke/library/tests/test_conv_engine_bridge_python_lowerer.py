# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""``py_fn`` of every ``kernels.conv.engine_bridge.lower_*`` uses the Python lowerer.

``rocke.core.lower_family`` runs ``py_fn`` for ``backend="python"`` and as the
Python half of ``backend="both"``. If ``py_fn`` called the public
``lower_kernel_to_llvm`` it would re-resolve the backend from the environment,
so ``"python"`` could silently run the C++ lowerer and ``"both"`` would compare
C++ against C++. The bridge therefore lowers through
``rocke.core.lower_kernel_python``, which calls ``_lower_kernel_to_llvm_python``.

* :class:`TestPyFnIsPythonOnly` replaces the dispatching lowerer and the C++
  engine with stubs that fail the test, then lowers one spec per family with
  ``backend="python"``: it must succeed. It needs no engine.
* :class:`TestBothRunsBothLowerers` (needs ``rocke_library_engine``) checks that
  ``backend="both"`` calls the Python lowerer once and the C++ binding, and
  that the two engines agree (serialized IR and ``.ll``) for every family.
* :func:`test_no_py_fn_uses_the_dispatching_lowerer` covers every ``lower_*``,
  including the families without a spec in the table below.

Run:
  PYTHONPATH=<engine build>/cpp/bindings:rocke/platform/python:rocke/library \\
      <python> -m pytest rocke/library/tests/test_conv_engine_bridge_python_lowerer.py -v
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import os
import unittest
from unittest import mock

from kernels.common._conv_implicit_gemm_common import ConvDataSpec, ConvProblem
from kernels.common.conv_dgrad_patch_wmma import DgradPatchWmmaSpec
from kernels.common.conv_dgrad_smallc_wmma import DgradSmallcWmmaSpec
from kernels.common.conv_direct_dgrad_pp_mfma import (
    DgradPpMfmaSpec,
    dgrad_pp_mfma_problem,
)
from kernels.common.conv_direct_dgrad_wmma import DirectDgradWmmaSpec
from kernels.common.conv_direct_dw_fwd_wmma import DwFwdWmmaSpec
from kernels.common.conv_direct_fwd_flex_mfma import FwdFlexMfmaSpec
from kernels.common.conv_direct_fwd_gstream_wmma import FwdGstreamWmmaSpec
from kernels.common.conv_direct_fwd_pp_mfma import FwdPpMfmaSpec
from kernels.common.conv_direct_fwd_tapouter_wmma import DirectFwdTapOuterSpec
from kernels.common.conv_direct_fwd_wmma import DirectFwdWmmaSpec
from kernels.common.conv_direct_grouped import (
    DirectConv4cSpec,
    DirectConv16cSpec,
    DirectConvProblem,
    DirectConvSpec,
)
from kernels.common.conv_direct_nongrouped import DirectNongroupedConvSpec
from kernels.common.conv_fwd_small_mfma import FwdSmallcMfmaSpec, FwdSmallkMfmaSpec
from kernels.common.conv_fwd_small_wmma import FwdSmallcWmmaSpec, FwdSmallkWmmaSpec
from kernels.common.conv_implicit_gemm import ImplicitGemmConvSpec
from kernels.common.conv_winograd_fwd import WinogradFwdSpec
from kernels.common.conv_winograd_gemm_wmma import (
    WinogradGemmReduceSpec,
    WinogradGemmWmmaSpec,
)
from kernels.common.img2col import Img2ColSpec
from kernels.conv import engine_bridge as eb
from rocke.core import backend as bk
from rocke.core import lower_llvm

_HAS_ENGINE = importlib.util.find_spec("rocke_library_engine") is not None
_SKIP = "" if _HAS_ENGINE else "rocke_library_engine extension not importable"


def _p(
    C, K, KH=3, S=1, PAD=1, groups=1, H=32, W=48, dtype="bf16", N=2, **kw
) -> DirectConvProblem:
    return DirectConvProblem(
        N=N,
        H=H,
        W=W,
        groups=groups,
        cpg=C,
        kpg=K,
        KH=KH,
        KW=KH,
        PAD=PAD,
        stride=S,
        dtype=dtype,
        **kw,
    )


def _cases():
    """``(label, lower_fn, positional args, keyword args, arch)`` per family."""
    p4 = DirectConvProblem(
        N=2, H=13, W=17, groups=32, cpg=4, kpg=4, KH=3, KW=3, PAD=1, dtype="fp16"
    )
    p16 = DirectConvProblem(
        N=2, H=9, W=17, groups=8, cpg=16, kpg=16, KH=3, KW=3, PAD=1, dtype="fp16"
    )
    wino = ConvProblem(N=1, Hi=9, Wi=7, C=12, K=20, Y=3, X=3, pH=1, pW=1, groups=2)
    ig = ImplicitGemmConvSpec(
        problem=ConvProblem(N=2, Hi=14, Wi=14, C=32, K=64, Y=3, X=3, pH=1, pW=1),
        data=ConvDataSpec("fp16", "fp16", "fp16"),
        tile_m=64,
        tile_n=64,
        tile_k=32,
        warp_m=2,
        warp_n=2,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=32,
        epilogue="cshuffle",
    )
    ng = DirectNongroupedConvSpec(
        problem=DirectConvProblem(
            N=1, H=8, W=32, groups=1, cpg=32, kpg=64, dtype="fp16"
        ),
        tile_h=4,
        tile_w=32,
        tile_k=64,
        ck=16,
        waves_m=2,
        waves_n=2,
        lds_pad=8,
    )
    return (
        ("conv_implicit_gemm", eb.lower_conv_implicit_gemm, (ig,), {}, "gfx950"),
        (
            "conv_direct_grouped/16c",
            eb.lower_conv_direct_grouped,
            (DirectConv16cSpec(problem=p16),),
            {"kind": "16c"},
            "gfx950",
        ),
        (
            "conv_direct_grouped/4c",
            eb.lower_conv_direct_grouped,
            (DirectConv4cSpec(problem=p4),),
            {"kind": "4c"},
            "gfx950",
        ),
        (
            "conv_direct_grouped/generic",
            eb.lower_conv_direct_grouped,
            (DirectConvSpec(problem=p16),),
            {"kind": "generic"},
            "gfx950",
        ),
        (
            "conv_direct_nongrouped",
            eb.lower_conv_direct_nongrouped,
            (ng,),
            {},
            "gfx950",
        ),
        (
            "conv_direct_fwd_smallc_wmma",
            eb.lower_conv_direct_fwd_smallc_wmma,
            (FwdSmallcWmmaSpec(problem=_p(3, 64)),),
            {},
            "gfx1250",
        ),
        (
            "conv_direct_fwd_smallk_wmma",
            eb.lower_conv_direct_fwd_smallk_wmma,
            (FwdSmallkWmmaSpec(problem=_p(64, 4)),),
            {},
            "gfx1250",
        ),
        (
            "conv_direct_fwd_smallc_mfma",
            eb.lower_conv_direct_fwd_smallc_mfma,
            (FwdSmallcMfmaSpec(problem=_p(3, 64)),),
            {},
            "gfx950",
        ),
        (
            "conv_direct_fwd_smallk_mfma",
            eb.lower_conv_direct_fwd_smallk_mfma,
            (FwdSmallkMfmaSpec(problem=_p(64, 4)),),
            {},
            "gfx950",
        ),
        (
            "conv_direct_fwd_pp_mfma",
            eb.lower_conv_direct_fwd_pp_mfma,
            (FwdPpMfmaSpec(problem=_p(96, 136, H=37, W=41)),),
            {},
            "gfx950",
        ),
        (
            "conv_direct_fwd_flex_mfma",
            eb.lower_conv_direct_fwd_flex_mfma,
            (FwdFlexMfmaSpec(problem=_p(192, 136, S=2, H=37, W=41)),),
            {},
            "gfx950",
        ),
        (
            "conv_direct_dw_fwd_wmma",
            eb.lower_conv_direct_dw_fwd_wmma,
            (DwFwdWmmaSpec(problem=_p(1, 1, groups=64, H=20, W=37)),),
            {},
            "gfx1250",
        ),
        (
            "conv_direct_fwd_gstream_wmma",
            eb.lower_conv_direct_fwd_gstream_wmma,
            (FwdGstreamWmmaSpec(problem=_p(16, 16, groups=8, H=28, W=28)),),
            {},
            "gfx1250",
        ),
        (
            "conv_direct_fwd_tapouter_wmma",
            eb.lower_conv_direct_fwd_tapouter_wmma,
            (
                DirectFwdTapOuterSpec(
                    problem=_p(64, 96, PAD=12, H=29, W=37, dil_h=12, dil_w=12)
                ),
            ),
            {},
            "gfx1250",
        ),
        (
            "conv_dgrad_patch_wmma",
            eb.lower_conv_dgrad_patch_wmma,
            (
                DgradPatchWmmaSpec(
                    problem=_p(3, 64, KH=4, S=4, PAD=0),
                    waves_c=3,
                    waves_p=1,
                    tiles_c=1,
                    r_blk=4,
                ),
            ),
            {},
            "gfx1250",
        ),
        (
            "conv_dgrad_smallc_wmma",
            eb.lower_conv_dgrad_smallc_wmma,
            (
                DgradSmallcWmmaSpec(
                    problem=_p(3, 64),
                    waves_c=1,
                    waves_p=4,
                    tiles_c=2,
                    tiles_p=6,
                    out_h=16,
                    out_w=16,
                ),
            ),
            {},
            "gfx1250",
        ),
        (
            "conv_direct_dgrad_wmma",
            eb.lower_conv_direct_dgrad_wmma,
            (DirectDgradWmmaSpec(problem=_p(64, 64, H=28, W=28)),),
            {},
            "gfx1250",
        ),
        (
            "conv_direct_dgrad_pp_mfma",
            eb.lower_conv_direct_dgrad_pp_mfma,
            (DgradPpMfmaSpec(problem=dgrad_pp_mfma_problem(_p(136, 96, H=37, W=41))),),
            {},
            "gfx950",
        ),
        (
            "conv_direct_fwd_wmma",
            eb.lower_conv_direct_fwd_wmma,
            (DirectFwdWmmaSpec(problem=_p(64, 64, H=28, W=28)),),
            {},
            "gfx1250",
        ),
        (
            "conv_winograd_fwd",
            eb.lower_conv_winograd_fwd,
            (WinogradFwdSpec(problem=wino, m=2), "K1"),
            {},
            "gfx1250",
        ),
        (
            "conv_winograd_gemm_wmma",
            eb.lower_conv_winograd_gemm_wmma,
            (
                WinogradGemmWmmaSpec(
                    waves_m=1, waves_n=2, frags_m=2, frags_n=1, tile_k=32, stages=2
                ),
            ),
            {},
            "gfx1250",
        ),
        (
            "conv_winograd_gemm_reduce",
            eb.lower_conv_winograd_gemm_reduce,
            (WinogradGemmReduceSpec("bf16"),),
            {},
            "gfx1250",
        ),
        (
            "img2col",
            eb.lower_img2col,
            (
                Img2ColSpec(
                    problem=ConvProblem(N=1, Hi=8, Wi=8, C=16, K=16, Y=3, X=3),
                    block_tile_m=4,
                    block_tile_k=64,
                    vec_k=1,
                ),
            ),
            {},
            "gfx950",
        ),
    )


class _EngineMustNotRun:
    """Stands in for the binding: the bridge imports it for every backend, but
    ``backend="python"`` must never call into it."""

    def __getattr__(self, name):
        def call(*_a, **_k):
            raise AssertionError(f"{name}: the C++ engine ran for backend='python'")

        return call


def _engine_must_not_run(*_a, **_k):
    return _EngineMustNotRun()


def _dispatch_must_not_run(*_a, **_k):
    raise AssertionError(
        "py_fn went through the backend-resolving lowerer; it must call "
        "rocke.core.lower_kernel_python"
    )


# The gfx1250 builders lower tensor_load_to_lds only at llvm23, so pin the
# flavor rather than inherit the host's.
@mock.patch.dict(os.environ, {"ROCKE_LLVM_FLAVOR": "llvm23"})
class TestPyFnIsPythonOnly(unittest.TestCase):
    def test_backend_python_never_reaches_the_cpp_lowerer(self):
        for label, fn, args, kw, arch in _cases():
            with self.subTest(family=label):
                with (
                    mock.patch.object(
                        lower_llvm, "lower_kernel_to_llvm", _dispatch_must_not_run
                    ),
                    mock.patch.object(
                        bk, "lower_kernel_via_backend", _dispatch_must_not_run
                    ),
                    mock.patch.object(eb, "import_engine", _engine_must_not_run),
                ):
                    res = fn(*args, arch=arch, backend="python", want_ir=True, **kw)
                self.assertEqual(res.backend, "python")
                self.assertTrue(res.llvm_text)
                self.assertTrue(res.ir_text)


@unittest.skipIf(bool(_SKIP), _SKIP)
@mock.patch.dict(os.environ, {"ROCKE_LLVM_FLAVOR": "llvm23"})
class TestBothRunsBothLowerers(unittest.TestCase):
    def test_backend_both_calls_python_lowerer_and_binding(self):
        real_py = lower_llvm._lower_kernel_to_llvm_python
        eng = eb.import_engine(eb.ENGINE_MODULE)
        for label, fn, args, kw, arch in _cases():
            with self.subTest(family=label):
                py_calls = []
                cpp_calls = []

                def counting_py(kernel, **k):
                    py_calls.append(1)
                    return real_py(kernel, **k)

                class _Counting:
                    def __getattr__(self, name):
                        attr = getattr(eng, name)
                        if not name.endswith("_lower_llvm"):
                            return attr

                        def call(*a, **k):
                            cpp_calls.append(name)
                            return attr(*a, **k)

                        return call

                with (
                    mock.patch.object(
                        lower_llvm, "_lower_kernel_to_llvm_python", counting_py
                    ),
                    mock.patch.object(
                        eb, "import_engine", lambda *_a, **_k: _Counting()
                    ),
                ):
                    res = fn(*args, arch=arch, backend="both", **kw)
                self.assertEqual(len(py_calls), 1)
                self.assertEqual(len(cpp_calls), 1, cpp_calls)
                self.assertEqual(res.backend, "both")


def test_no_py_fn_uses_the_dispatching_lowerer():
    """Every ``lower_*`` of the bridge, including families without a spec in
    :func:`_cases`: the module must not name the backend-resolving
    ``lower_kernel_to_llvm`` and every nested ``py_fn`` must call the shared
    Python-lowerer helper."""
    tree = ast.parse(inspect.getsource(eb))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {
        a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names
    }
    assert "lower_kernel_to_llvm" not in names
    lowers = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("lower_")
    ]
    assert len(lowers) >= 24
    for top in lowers:
        fns = [
            n
            for n in ast.walk(top)
            if isinstance(n, ast.FunctionDef) and n.name == "py_fn"
        ]
        assert len(fns) == 1, top.name
        called = {
            c.func.id
            for c in ast.walk(fns[0])
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
        }
        assert "lower_kernel_python_ll_ir" in called, top.name
        # The shared helper is the whole body: no per-family (ll, ir) block.
        rets = [n for n in ast.walk(fns[0]) if isinstance(n, ast.Return)]
        assert len(rets) == 1 and isinstance(rets[0].value, ast.Call), top.name
        assert rets[0].value.func.id == "lower_kernel_python_ll_ir", top.name
        assert "serialize" not in {
            a.name
            for n in ast.walk(fns[0])
            if isinstance(n, ast.ImportFrom)
            for a in n.names
        }, top.name


if __name__ == "__main__":
    unittest.main()
