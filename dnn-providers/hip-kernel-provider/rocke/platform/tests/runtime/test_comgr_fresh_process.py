# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``build_hsaco_from_llvm_ir(..., fresh_process=True)`` and its child.

CPU-only: the child process is stubbed except in the last test, which compiles
a trivial kernel both ways with the comgr library when one loads (skipped
otherwise).
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from rocke.core.ir import KernelDef, Region
from rocke.helpers import compile as compile_module
from rocke.runtime import _comgr_fresh_process as child
from rocke.runtime import comgr

PKG_ROOT = str(Path(comgr.__file__).resolve().parents[2])


@pytest.fixture
def no_lib(monkeypatch: pytest.MonkeyPatch) -> None:
    """No comgr is loaded in this process: the parent's own checks pass."""
    lib = SimpleNamespace(_name="/opt/x/libamd_comgr.so.3")
    monkeypatch.setattr(comgr, "_resolve_lib", lambda: lib)
    monkeypatch.setattr(comgr, "_assert_ir_flavor_matches_lib", lambda ir: None)


def _fake_child(rc: int, *, payload: bytes | None = b"HSACO", stderr: bytes = b""):
    """A ``subprocess.run`` stand-in that answers like the child would."""
    calls = []

    def run(cmd, *, input, stdout, stderr: object, env):  # noqa: A002
        calls.append(SimpleNamespace(cmd=cmd, input=input, env=env))
        header = json.loads(input.partition(b"\n")[0])
        if payload is not None:
            meta = {
                "size": len(payload),
                "bc": 1.0,
                "relocatable": 2.0,
                "executable": 3.0,
            }
            with open(header["out"], "wb") as fh:
                fh.write(json.dumps(meta).encode() + b"\n" + payload)
        return SimpleNamespace(returncode=rc, stdout=b"", stderr=err)

    err = stderr
    return run, calls


def test_fresh_process_round_trip(no_lib: None) -> None:
    run, calls = _fake_child(0)
    with mock.patch.object(subprocess, "run", run):
        hsaco, t = comgr.build_hsaco_from_llvm_ir(
            "IR TEXT",
            isa="amdgcn-amd-amdhsa--gfx942",
            options=["-O3", "-x"],
            fresh_process=True,
        )
    assert hsaco == b"HSACO"
    assert (t.bc, t.relocatable, t.executable) == (1.0, 2.0, 3.0)
    (call,) = calls
    assert call.cmd == [sys.executable, "-m", "rocke.runtime._comgr_fresh_process"]
    head, _, body = call.input.partition(b"\n")
    header = json.loads(head)
    assert header["isa"] == "amdgcn-amd-amdhsa--gfx942"
    assert header["options"] == ["-O3", "-x"]
    assert body == b"IR TEXT"
    # the child loads this process's library and imports this rocke package
    assert call.env["ROCKE_COMGR_LIB"] == "/opt/x/libamd_comgr.so.3"
    # ... compiles the module itself (no object from comgr's on-disk cache)
    assert call.env["AMD_COMGR_CACHE"] == "0"
    assert call.env["PYTHONPATH"].split(os.pathsep)[0] == PKG_ROOT


def test_in_process_compile_is_the_default(no_lib: None) -> None:
    with (
        mock.patch.object(subprocess, "run", side_effect=AssertionError("spawned")),
        mock.patch.object(comgr, "_destroy_data_set"),
        mock.patch.object(
            comgr, "_create_data_set", side_effect=RuntimeError("in-process")
        ),
    ):
        with pytest.raises(RuntimeError, match="in-process"):
            comgr.build_hsaco_from_llvm_ir("IR", isa="amdgcn-amd-amdhsa--gfx942")


@pytest.mark.parametrize(
    ("rc", "payload", "error", "match"),
    [
        (2, None, comgr.ComgrError, "do_action failed"),
        (1, None, comgr.ComgrError, "fresh compile process failed"),
        (-11, None, comgr.ComgrProcessDied, "died"),
        (3221225477, None, comgr.ComgrProcessDied, "died"),
        (0, None, comgr.ComgrProcessDied, "died"),
    ],
)
def test_fresh_process_errors(no_lib, rc, payload, error, match) -> None:
    run, _ = _fake_child(rc, payload=payload, stderr=b"do_action failed")
    with mock.patch.object(subprocess, "run", run):
        with pytest.raises(error, match=match) as info:
            comgr.build_hsaco_from_llvm_ir("IR", fresh_process=True)
    # a compile error is not reported as a dead process
    assert (info.type is comgr.ComgrProcessDied) == (rc not in (1, 2))


def test_truncated_result_is_a_dead_process(no_lib: None) -> None:
    def run(cmd, *, input, stdout, stderr, env):  # noqa: A002
        out = json.loads(input.partition(b"\n")[0])["out"]
        with open(out, "wb") as fh:
            fh.write(b'{"size": 9, "bc": 0, "relocatable": 0, "executable": 0}\nABC')
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    with mock.patch.object(subprocess, "run", run):
        with pytest.raises(comgr.ComgrProcessDied, match="3 of 9"):
            comgr.build_hsaco_from_llvm_ir("IR", fresh_process=True)


def _stdin(header: dict, body: bytes) -> SimpleNamespace:
    return SimpleNamespace(
        buffer=io.BytesIO(json.dumps(header).encode() + b"\n" + body)
    )


def test_child_writes_the_protocol(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out.bin"
    seen = {}

    def build(ir, *, isa, options):
        seen.update(ir=ir, isa=isa, options=options)
        return b"\x7fELF..", comgr.ComgrTimings(
            bc=0.5, relocatable=0.25, executable=0.125
        )

    monkeypatch.setattr(comgr, "build_hsaco_from_llvm_ir", build)
    hdr = {"isa": "amdgcn-amd-amdhsa--gfx950", "options": ["-O3"], "out": str(out)}
    monkeypatch.setattr(sys, "stdin", _stdin(hdr, "ü IR".encode()))
    assert child.main() == 0
    # the child compiles in its own process: it never asks for another one
    assert seen == {
        "ir": "ü IR",
        "isa": "amdgcn-amd-amdhsa--gfx950",
        "options": ["-O3"],
    }
    head, _, blob = out.read_bytes().partition(b"\n")
    assert json.loads(head) == {
        "size": 6,
        "bc": 0.5,
        "relocatable": 0.25,
        "executable": 0.125,
    }
    assert blob == b"\x7fELF.."


def test_child_reports_a_compile_error(tmp_path, monkeypatch, capsys) -> None:
    def build(ir, *, isa, options):
        raise comgr.ComgrError("bad IR")

    monkeypatch.setattr(comgr, "build_hsaco_from_llvm_ir", build)
    hdr = {"isa": "i", "options": [], "out": str(tmp_path / "o")}
    monkeypatch.setattr(sys, "stdin", _stdin(hdr, b"IR"))
    assert child.main() == 2
    assert capsys.readouterr().err == "bad IR"
    assert not (tmp_path / "o").exists()


@pytest.mark.parametrize("marked", [False, True])
def test_compile_kernel_follows_the_kernel_attribute(marked: bool) -> None:
    kernel = KernelDef("fresh_contract", [], Region("entry"))
    if marked:
        kernel.attrs[compile_module.COMGR_FRESH_PROCESS_ATTR] = True
    with (
        mock.patch.object(compile_module, "_lower_llvm_via_backend", return_value="ll"),
        mock.patch.object(
            compile_module,
            "build_hsaco_from_llvm_ir",
            return_value=(b"hsaco", comgr.ComgrTimings()),
        ) as build,
    ):
        compile_module.compile_kernel(kernel, arch="gfx942")
    assert build.call_args.kwargs["fresh_process"] is marked


def test_fresh_process_builds_the_in_process_object() -> None:
    """With a loadable comgr: a trivial kernel compiles to the same bytes in
    this process and in a fresh one (the child really runs)."""
    try:
        comgr._resolve_lib()
    except comgr.ComgrError as ex:
        pytest.skip(f"no comgr: {ex}")
    kernel = KernelDef("fresh_process_probe", [], Region("entry"))
    here = compile_module.compile_kernel(kernel, arch="gfx942", backend="python")
    kernel.attrs[compile_module.COMGR_FRESH_PROCESS_ATTR] = True
    fresh = compile_module.compile_kernel(kernel, arch="gfx942", backend="python")
    assert fresh.llvm_text == here.llvm_text
    assert fresh.hsaco == here.hsaco and fresh.hsaco[:4] == b"\x7fELF"
