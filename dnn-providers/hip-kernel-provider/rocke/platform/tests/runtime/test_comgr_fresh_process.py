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
import re
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
from rocke.tests.arch_marks import COVERED_ARCHES

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
    assert header["memory_cap_bytes"] == comgr.FRESH_PROCESS_MEMORY_CAP_BYTES
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


# What a ROCm 7.2.4 child wrote to stderr when the compile of an
# iglp_opt(1) region without LDS reads ran into the cap, before it aborted.
_LLVM_OUT_OF_MEMORY_STDERR = b"LLVM ERROR: out of memory\nAllocation failed\n"


@pytest.mark.parametrize("capped", [True, False])
@pytest.mark.parametrize(
    ("rc", "stderr", "reported"),
    [
        # the compiler: LLVM reports the failed allocation and aborts
        (-6, _LLVM_OUT_OF_MEMORY_STDERR, "LLVM ERROR: out of memory"),
        (0xC0000409, _LLVM_OUT_OF_MEMORY_STDERR, "LLVM ERROR: out of memory"),
        # Python in the child: a MemoryError
        (child.OUT_OF_MEMORY_EXIT_STATUS, b"MemoryError", "MemoryError"),
    ],
    ids=["posix-abort", "windows-abort", "python"],
)
def test_running_out_of_memory_names_the_cause(
    no_lib, monkeypatch, rc, stderr, reported, capped
) -> None:
    monkeypatch.setattr(child, "ADDRESS_SPACE_CAP_APPLIES", capped)
    run, _ = _fake_child(rc, payload=None, stderr=stderr)
    cap = " under its 4 GiB address-space cap" if capped else ""
    expected = re.escape(
        f"the fresh compile process ran out of memory{cap} ({reported})."
    )
    with (
        mock.patch.object(subprocess, "run", run),
        pytest.raises(comgr.ComgrOutOfMemory, match=f"^{expected}") as info,
    ):
        comgr.build_hsaco_from_llvm_ir("IR", fresh_process=True)
    message = str(info.value)
    assert "576e1affab35" in message and "gfx_support_audit.md" in message
    # a dead process to callers that catch that
    assert isinstance(info.value, comgr.ComgrProcessDied)


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
    # capping this test process would cap pytest: record the cap instead (a real
    # child applies it: test_the_child_caps_its_own_address_space)
    caps = []
    monkeypatch.setattr(child, "limit_address_space", caps.append)
    hdr = {
        "isa": "amdgcn-amd-amdhsa--gfx950",
        "options": ["-O3"],
        "out": str(out),
        "memory_cap_bytes": 123 << 20,
    }
    monkeypatch.setattr(sys, "stdin", _stdin(hdr, "ü IR".encode()))
    assert child.main() == 0
    assert caps == [123 << 20]
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
    monkeypatch.setattr(child, "limit_address_space", lambda cap_bytes: None)
    hdr = {"isa": "i", "options": [], "out": str(tmp_path / "o"), "memory_cap_bytes": 1}
    monkeypatch.setattr(sys, "stdin", _stdin(hdr, b"IR"))
    assert child.main() == 2
    assert capsys.readouterr().err == "bad IR"
    assert not (tmp_path / "o").exists()


def test_child_reports_running_out_of_memory(tmp_path, monkeypatch, capsys) -> None:
    def build(ir, *, isa, options):
        raise MemoryError

    monkeypatch.setattr(comgr, "build_hsaco_from_llvm_ir", build)
    monkeypatch.setattr(child, "limit_address_space", lambda cap_bytes: None)
    hdr = {"isa": "i", "options": [], "out": str(tmp_path / "o"), "memory_cap_bytes": 1}
    monkeypatch.setattr(sys, "stdin", _stdin(hdr, b"IR"))
    assert child.main() == child.OUT_OF_MEMORY_EXIT_STATUS
    assert capsys.readouterr().err == "MemoryError"
    assert not (tmp_path / "o").exists()


def _run_in_python(script: str) -> list[str]:
    """Run ``script`` in a new Python process that imports this rocke
    package; return its stdout lines."""
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": PKG_ROOT},
        check=False,
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.splitlines()


def _tightest(*limits: int) -> int:
    """The lowest of ``limits``, an unlimited one counting as no limit."""
    import resource

    finite = [limit for limit in limits if limit != resource.RLIM_INFINITY]
    return min(finite) if finite else resource.RLIM_INFINITY


@pytest.mark.skipif(not child.ADDRESS_SPACE_CAP_APPLIES, reason="no RLIMIT_AS")
def test_the_child_caps_its_own_address_space() -> None:
    """In a real process the cap becomes the RLIMIT_AS soft limit (or a lower
    limit the process inherited stays) and an allocation past it raises
    MemoryError; a soft limit below a later cap stays, and the hard limit is
    left as it is. The expected limits follow whatever limits this test runs
    under (the child process inherits them)."""
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    cap, lower = 1 << 30, 3 << 29
    inherited = _tightest(soft, hard)
    # RLIM_INFINITY is -1 here: an unlimited address space is no reason to skip.
    if inherited != resource.RLIM_INFINITY and inherited < lower:
        pytest.skip(f"inherited RLIMIT_AS {soft, hard} is below {lower} bytes")
    lines = _run_in_python(
        "import resource\n"
        "from rocke.runtime._comgr_fresh_process import limit_address_space\n"
        f"limit_address_space({cap})\n"
        "print(*resource.getrlimit(resource.RLIMIT_AS))\n"
        "try:\n"
        f"    bytearray({2 * cap})\n"
        "except MemoryError:\n"
        "    print('MemoryError')\n"
        f"resource.setrlimit(resource.RLIMIT_AS, ({lower}, {lower}))\n"
        f"limit_address_space({4 * cap})\n"
        "print(*resource.getrlimit(resource.RLIMIT_AS))\n"
    )
    assert lines == [
        f"{_tightest(cap, soft, hard)} {hard}",
        "MemoryError",
        f"{lower} {lower}",
    ]


def test_the_child_runs_without_a_cap_where_none_applies(monkeypatch) -> None:
    """Where the platform cannot cap the address space, the child does not
    reach for ``resource`` at all (here an import of it would fail)."""
    monkeypatch.setattr(child, "ADDRESS_SPACE_CAP_APPLIES", False)
    monkeypatch.setitem(sys.modules, "resource", None)
    child.limit_address_space(1 << 20)


@pytest.mark.parametrize("arch", COVERED_ARCHES)
@pytest.mark.parametrize("marked", [False, True])
def test_compile_kernel_follows_the_kernel_attribute(marked: bool, arch: str) -> None:
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
        compile_module.compile_kernel(kernel, arch=arch)
    assert build.call_args.kwargs["fresh_process"] is marked


@pytest.mark.parametrize("arch", COVERED_ARCHES)
@pytest.mark.parametrize("marked", [False, True])
def test_a_lowered_kernel_compiles_as_compile_kernel_compiles_it(
    marked: bool, arch: str
) -> None:
    """compile_comgr_input hands COMGR what compile_kernel hands it, fresh process
    included, and returns the code object."""
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
        compile_module.compile_kernel(kernel, arch=arch)
        whole = build.call_args
        hsaco = compile_module.compile_comgr_input(
            compile_module.lower_kernel_for_comgr(kernel, arch=arch)
        )
    assert build.call_args == whole
    assert hsaco == b"hsaco"


@pytest.mark.parametrize("arch", COVERED_ARCHES)
def test_fresh_process_builds_the_in_process_object(arch: str) -> None:
    """With a loadable comgr: a trivial kernel compiles to the same bytes in
    this process and in a fresh one (the child really runs)."""
    try:
        comgr._resolve_lib()
    except comgr.ComgrError as ex:
        pytest.skip(f"no comgr: {ex}")
    kernel = KernelDef("fresh_process_probe", [], Region("entry"))
    here = compile_module.compile_kernel(kernel, arch=arch, backend="python")
    kernel.attrs[compile_module.COMGR_FRESH_PROCESS_ATTR] = True
    fresh = compile_module.compile_kernel(kernel, arch=arch, backend="python")
    assert fresh.llvm_text == here.llvm_text
    assert fresh.hsaco == here.hsaco and fresh.hsaco[:4] == b"\x7fELF"
