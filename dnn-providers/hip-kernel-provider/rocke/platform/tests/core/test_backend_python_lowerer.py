# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The Python path of every ``rocke.core.backend`` family uses the Python lowerer.

``backend="python"`` (and the Python half of ``backend="both"``) must not go
through the public ``lower_kernel_to_llvm``: it re-resolves the backend from
``ROCKE_BACKEND`` and, with ``ROCKE_BACKEND=cpp``, runs the C++ lowerer while
the result is labelled ``python``. Every family lowers through
``rocke.core.lower_kernel_python`` instead.

CPU only; no engine needed.
"""

from __future__ import annotations

import ast
import inspect
import os
from unittest import mock

import rocke.core as core
from rocke.core import backend as bk
from rocke.core import lower_llvm
from rocke.instances.common.batched_gemm import BatchedGemmSpec
from rocke.instances.common.gemm_universal import (
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
)

_TILE = TileSpec(32, 32, 32, 1, 1, 1, 16, 16, 16)


def _dispatch_must_not_run(*_a, **_k):
    raise AssertionError(
        "the Python path went through the backend-resolving lowerer; it must "
        "call rocke.core.lower_kernel_python"
    )


class _EngineMustNotRun:
    def __getattr__(self, name):
        def call(*_a, **_k):
            raise AssertionError(f"{name}: the C++ engine ran for backend='python'")

        return call


def _no_dispatch():
    return (
        mock.patch.dict(os.environ, {"ROCKE_BACKEND": "cpp"}),
        mock.patch.object(lower_llvm, "lower_kernel_to_llvm", _dispatch_must_not_run),
        mock.patch.object(bk, "lower_kernel_via_backend", _dispatch_must_not_run),
        mock.patch.object(bk, "import_engine", lambda *_a, **_k: _EngineMustNotRun()),
    )


def _run_python(fn, spec):
    p0, p1, p2, p3 = _no_dispatch()
    with p0, p1, p2, p3:
        res = fn(spec, arch="gfx942", backend="python", want_ir=True)
    assert res.backend == "python"
    assert "amdgcn" in res.llvm_text
    assert res.ir_text


def test_universal_gemm_python_path_with_cpp_default():
    spec = UniversalGemmSpec(
        name="py_only",
        tile=_TILE,
        trait=TraitSpec(pipeline="mem", epilogue="default"),
        data=DataSpec("fp16", "fp16", "fp16", "fp32"),
    )
    _run_python(bk.lower_universal_gemm, spec)


def test_family_python_path_with_cpp_default():
    spec = BatchedGemmSpec(
        name="py_only_batched",
        tile=_TILE,
        trait=TraitSpec(pipeline="mem", epilogue="default"),
    )
    _run_python(bk.lower_batched_gemm, spec)


def test_lower_kernel_python_is_public():
    assert core.lower_kernel_python is bk.lower_kernel_python


def test_lower_kernel_python_ll_ir_is_public():
    assert core.lower_kernel_python_ll_ir is bk.lower_kernel_python_ll_ir


def test_no_python_path_uses_the_dispatching_lowerer():
    """Every ``lower_*`` family wrapper's ``py_fn`` and the universal GEMM's
    ``_lower_python`` go through ``lower_kernel_python_ll_ir`` (which calls
    ``lower_kernel_python``); nothing in the module calls
    ``lower_kernel_to_llvm``."""
    tree = ast.parse(inspect.getsource(bk))

    def called(node):
        out = set()
        for c in ast.walk(node):
            if isinstance(c, ast.Call):
                f = c.func
                out.add(f.id if isinstance(f, ast.Name) else getattr(f, "attr", ""))
        return out

    assert "lower_kernel_to_llvm" not in called(tree)
    top = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert "lower_kernel_python_ll_ir" in called(top["_lower_python"])
    assert "lower_kernel_python" in called(top["lower_kernel_python_ll_ir"])
    wrappers = [
        n
        for n in top.values()
        if any(
            isinstance(f, ast.FunctionDef) and f.name == "py_fn" for f in ast.walk(n)
        )
    ]
    assert len(wrappers) >= 10
    for w in wrappers:
        py_fn = next(
            f
            for f in ast.walk(w)
            if isinstance(f, ast.FunctionDef) and f.name == "py_fn"
        )
        assert "lower_kernel_python_ll_ir" in called(py_fn), w.name
