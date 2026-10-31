# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for what TensileCreateLibrary's assembly codegen workers return:
each worker writes and assembles its kernel's source itself, so the result sent
back to the parent carries the kernel info but not the source."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import tensilelite.TensileCreateLibrary.Run as Run

pytestmark = pytest.mark.unit

_ISA = (9, 4, 2)


def _result():
    return Run.KernelCodeGenResult(
        err=0,
        src="s_endpgm\n",
        header="// header\n",
        name="kernel0",
        targetObjFilename=None,
        isa=_ISA,
        wavefrontSize=64,
        cuoccupancy=2,
        pgr=1,
        mathclk=7,
    )


@pytest.fixture
def written(monkeypatch, tmp_path):
    """Runs writeSolutionsAndKernelsTCL on one fake kernel with real composed
    codegen, recording the result each worker writes to disk."""
    written = []
    kernel = {"KernelLanguage": "Assembly", "ISA": list(_ISA)}

    def writeAssembly(asmPath, result):
        written.append(result)
        return Path(asmPath) / f"{result.name}.s", result.isa, result.wavefrontSize, None

    monkeypatch.setattr(Run, "rocisa", MagicMock())
    monkeypatch.setattr(
        Run,
        "_normalizeKernelGroups",
        lambda kernels, splitGSU: [SimpleNamespace(representative=k, aliases=[k]) for k in kernels],
    )
    monkeypatch.setattr(Run, "processKernelSource", lambda *a, **kw: _result())
    monkeypatch.setattr(Run, "writeAssembly", writeAssembly)
    monkeypatch.setattr(Run, "_alignAmdgcnTargetToStepping", lambda *a, **kw: None)
    monkeypatch.setattr(Run, "_stinky_asm_verify_wanted", lambda isa: False)
    monkeypatch.setattr(Run, "ParallelMap2", lambda fn, items, *a, **kw: [fn(k) for k in items])
    monkeypatch.setattr(Run, "buildAssemblyCodeObjectFiles", lambda *a, **kw: [])
    monkeypatch.setattr(Run, "buildSourceCodeObjectFiles", lambda *a, **kw: [])
    monkeypatch.setattr(Run, "writeHelpers", lambda *a, **kw: None)

    _, _, results = Run.writeSolutionsAndKernelsTCL(
        str(tmp_path),
        MagicMock(),  # asmToolchain
        MagicMock(),  # srcToolchain
        [],  # solutions
        [kernel],
        [],  # kernelHelperObjs
        MagicMock(),  # kernelWriterAssembly
        ["gfx942"],
        removeTemporaries=False,
    )
    return written, results


def test_worker_writes_the_full_source(written):
    written, _ = written
    assert [r.src for r in written] == ["s_endpgm\n"]


def test_result_returned_to_the_parent_has_no_source(written):
    _, results = written
    assert [(r.src, r.header) for r in results] == [(None, None)]
    assert results[0]._replace(src=_result().src, header=_result().header) == _result()
