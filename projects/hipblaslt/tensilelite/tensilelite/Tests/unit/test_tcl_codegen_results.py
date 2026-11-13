# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for what TensileCreateLibrary's assembly codegen workers return:
each worker writes and assembles its kernel's source itself, so the result sent
back to the parent carries the kernel info but not the source."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import tensilelite.create_library.run as Run

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
        # Not the default, so that a field lost on the way back to the parent shows.
        customKernelDef={"name": "custom0"},
    )


@pytest.fixture
def run(monkeypatch, tmp_path):
    """Runs writeSolutionsAndKernelsTCL on one fake kernel through the real
    codegen composition: generating the source is faked, but the worker writes
    the real .s file and calls the (mock) assembler on it."""
    sourceCalls = []
    kernel = {"KernelLanguage": "Assembly", "ISA": list(_ISA)}

    def processKernelSource(*args, **kwargs):
        sourceCalls.append(kwargs)
        return _result()

    monkeypatch.setattr(Run, "rocisa", MagicMock())
    monkeypatch.setattr(
        Run,
        "_normalizeKernelGroups",
        lambda kernels, splitGSU: [SimpleNamespace(representative=k, aliases=[k]) for k in kernels],
    )
    monkeypatch.setattr(Run, "processKernelSource", processKernelSource)
    monkeypatch.setattr(Run, "_alignAmdgcnTargetToStepping", lambda *a, **kw: None)
    monkeypatch.setattr(Run, "_stinky_asm_verify_wanted", lambda isa: False)
    monkeypatch.setattr(Run, "ParallelMap2", lambda fn, items, *a, **kw: [fn(k) for k in items])
    monkeypatch.setattr(Run, "buildAssemblyCodeObjectFiles", lambda *a, **kw: [])
    monkeypatch.setattr(Run, "buildSourceCodeObjectFiles", lambda *a, **kw: [])
    monkeypatch.setattr(Run, "writeHelpers", lambda *a, **kw: None)

    asmToolchain = MagicMock()
    _, _, results = Run.writeSolutionsAndKernelsTCL(
        str(tmp_path),
        asmToolchain,
        MagicMock(),  # srcToolchain
        [],  # solutions
        [kernel],
        [],  # kernelHelperObjs
        MagicMock(),  # kernelWriterAssembly
        ["gfx942"],
        removeTemporaries=False,
    )
    return SimpleNamespace(
        tmp=tmp_path, assembler=asmToolchain.assembler, sourceCalls=sourceCalls, results=results
    )


def test_worker_writes_and_assembles_the_full_source(run):
    (asmFile,) = run.tmp.rglob("kernel0.s")
    assert asmFile.read_text() == "s_endpgm\n"
    run.assembler.assert_called_once()
    assert run.assembler.call_args.args[2] == str(asmFile)


def test_source_is_not_compressed_for_the_parent(run):
    assert run.sourceCalls == [{}]


def test_result_returned_to_the_parent_has_no_source(run):
    assert [(r.src, r.header) for r in run.results] == [(None, None)]
    assert run.results[0]._replace(src=_result().src, header=_result().header) == _result()
