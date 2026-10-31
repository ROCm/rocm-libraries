# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Codegen in a worker process must leave the solutions as in-process codegen does.

writeSolutionsAndKernels generates the kernels with ParallelMap2. In-process (one process,
e.g. CpuThreads: 1) the kernel objects are the solutions themselves, so whatever codegen writes
onto a kernel lands on its solution. In a worker the kernel is a pickled copy: only what
passPostKernelInfoToSolution copies back from the KernelCodeGenResult reaches the solutions,
and from them Data/*.yaml.

Both ways are run here on kernels of real configs and the complete state of every solution
(all attributes, compared deeply) must match, so a value codegen writes onto the kernel that
the copy-back misses fails the test, not only the ones known when it was written (LdsNumBytes
from the subtile LDS layout, LoopIters from ForceUnrollSubIter, _subtileUnitScaleVgpr from
plain-FP8 subtile kernels, LdsOffsetBias from the GlobalSplitU epilogue).
"""

import enum
import pickle

import pytest

from codegen_harness import _init_rocisa_for
from config_harness import (
    _isolated_globals_with_isa,
    _solutions_from_config_unguarded,
    _toolchain_for,
)
from Tensile.Common.Types import DebugConfig
from Tensile.KernelWriterAssembly import KernelWriterAssembly
from Tensile.TensileCreateLibrary.Run import (
    groupKernelRecords,
    passPostKernelInfoToSolution,
    processKernelSource,
)

pytestmark = pytest.mark.unit

_COMMON = "tensilelite/Tests/common/"

# Toolchain objects every solution shares by reference; they are not solution state and are
# not serialized.
_SHARED_ATTRS = {"assembler", "isaInfoMap"}


def _flatten(obj, path, out, seen):
    """Record every leaf of ``obj`` as ``out[path] = repr``, following containers and object
    attributes (``__dict__`` and ``__slots__``)."""
    if obj is None or isinstance(obj, (bool, int, float, str, bytes, enum.Enum)):
        out[path] = repr(obj)
        return
    if id(obj) in seen:
        out[path] = "<cycle>"
        return
    seen = seen | {id(obj)}
    if isinstance(obj, dict):
        out[path] = "dict"
        items = [(repr(key), value) for key, value in obj.items()]
    elif isinstance(obj, (list, tuple)):
        out[path] = "%s[%d]" % (type(obj).__name__, len(obj))
        items = [(str(index), value) for index, value in enumerate(obj)]
    elif isinstance(obj, (set, frozenset)):
        out[path] = repr(sorted(repr(value) for value in obj))
        return
    else:
        attrs = dict(getattr(obj, "__dict__", {}))
        for cls in type(obj).__mro__:
            for slot in getattr(cls, "__slots__", ()):
                if hasattr(obj, slot):
                    attrs[slot] = getattr(obj, slot)
        if not attrs:
            out[path] = repr(obj)
            return
        out[path] = type(obj).__name__
        items = [(key, value) for key, value in attrs.items() if key not in _SHARED_ATTRS]
    for key, value in items:
        _flatten(value, "%s/%s" % (path, key), out, seen)


def _solutionStates(solutions):
    out = {}
    for index, solution in enumerate(solutions):
        _flatten(solution, str(index), out, frozenset())
    return out


def _codegen(solutions, assembler, inWorker):
    """The codegen + copy-back part of writeSolutionsAndKernels, with ParallelMap2 running
    in-process (``inWorker`` False) or handing each task pickled arguments, as joblib does for
    a worker process (``inWorker`` True)."""
    kernels = [kernel for solution in solutions for kernel in solution.getKernels()]
    representatives = [group.representative for group in groupKernelRecords(kernels, False)]
    writer = KernelWriterAssembly(assembler, DebugConfig())
    results = []
    for kernel in representatives:
        rocIsa = _init_rocisa_for(kernel)
        args = (writer, rocIsa.getData(), rocIsa.getOutputOptions(), False, kernel)
        if inWorker:
            args = pickle.loads(pickle.dumps(args))
        result = processKernelSource(*args)
        assert result.err == 0, result.name
        results.append(result)
    passPostKernelInfoToSolution(results, representatives, solutions, False)


def _solutions(config, problemIndex, limit, assembler, isaInfoMap):
    return _solutions_from_config_unguarded(
        _COMMON + config, assembler, isaInfoMap, limit, problemIndex
    )


@pytest.mark.parametrize(
    "config,problemIndex,limit,changedKey",
    [
        # Subtile kernels: SubtileLdsLayout.applyLdsLayout sets LdsNumBytes.
        ("gemm/gfx950/subtile_mxfp4.yaml", 0, 2, "LdsNumBytes"),
        # The f8 problem's kernel takes ForceUnrollSubIter: KernelWriter sets LoopIters.
        ("gemm/gfx950/custom_mainloop_scheduling.yaml", 6, None, "LoopIters"),
        # Plain FP8 subtile kernel (MatrixInstK 128, no MX scale): Components/Subtile/Kernel.py
        # stores the unit-scale VGPR as _subtileUnitScaleVgpr.
        ("gemm/gfx950/subtile_fp8.yaml", 0, None, "_subtileUnitScaleVgpr"),
        # GlobalSplitU 2 kernels with a bias epilogue: the epilogue codegen switches
        # LdsOffsetBias between LdsOffsetBiasGSU / LdsOffsetBiasNonGSU and leaves the last.
        ("sparse/gfx94x/bf16_activation.yaml", 0, None, "LdsOffsetBias"),
    ],
)
def test_worker_codegen_leaves_solutions_as_in_process(config, problemIndex, limit, changedKey):
    assembler, isaInfoMap = _toolchain_for("gfx950")
    with _isolated_globals_with_isa(isaInfoMap):
        inProcess = _solutions(config, problemIndex, limit, assembler, isaInfoMap)
        inWorker = _solutions(config, problemIndex, limit, assembler, isaInfoMap)
        assert inProcess
        before = _solutionStates(inProcess)
        assert _solutionStates(inWorker) == before

        _codegen(inProcess, assembler, inWorker=False)
        _codegen(inWorker, assembler, inWorker=True)

    after = _solutionStates(inProcess)
    # The case is only meaningful if codegen changed the key it was chosen for.
    changed = {path for path in after if after[path] != before.get(path)}
    assert any(path.endswith("/'%s'" % changedKey) for path in changed), sorted(changed)
    assert _solutionStates(inWorker) == after
