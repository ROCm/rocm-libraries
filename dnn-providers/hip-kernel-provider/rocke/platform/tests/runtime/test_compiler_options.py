# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The measured table of ``-mllvm`` options each compiler accepts, and the
check that keeps an option the loaded compiler does not know away from COMGR.

LLVM's option parser exits the whole process on an unknown option, so the
tests that compile do it in a child process: a regression that lets the option
through ends the child, and the test reports it instead of pytest dying.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from rocke.runtime import comgr
from rocke.runtime import compiler_options as table
from rocke.runtime.comgr import CompilerInfo
from rocke.runtime.compiler_options import CompilerOptionNotAccepted

OPTION = "-amdgpu-mfma-vgpr-form"
PLATFORM = Path(__file__).resolve().parents[2]
PKG_ROOT = str(PLATFORM / "python")


def _compiler(*version: int) -> CompilerInfo:
    """A compiler identity as ``loaded_compiler_info`` reports one."""
    return CompilerInfo(
        llvm_version=version or None,
        source="LLVMGetVersion",
        requested_comgr="libamd_comgr.so",
        comgr_path="/opt/rocm/lib/libamd_comgr.so",
        query_library_path=None,
    )


def _measured_major(accepted: bool) -> int:
    """An LLVM major whose committed answer for OPTION is ``accepted``."""
    for major in json.loads(table.TABLE_PATH.read_text())["options"][OPTION]:
        if table.lookup(OPTION, int(major)).accepted is accepted:
            return int(major)
    pytest.fail(f"the table has no LLVM version that {accepted=} for {OPTION}")


# --- the table ------------------------------------------------------------------


def test_table_answers_are_measured_facts() -> None:
    doc = json.loads(table.TABLE_PATH.read_text(encoding="utf-8"))
    assert doc["schema"] == table.SCHEMA
    assert OPTION in table.measured_options()
    for name, column in doc["options"].items():
        assert column, f"{name} has no measured compiler"
        for major, cell in column.items():
            assert isinstance(cell["accepted"], bool)
            assert cell["verified_on"].startswith(f"LLVM {major}.")
            # A refusal carries the compiler's own words; an acceptance none.
            assert bool(cell["evidence"]) is not cell["accepted"], (name, major)


def test_backend_options_reads_only_mllvm_values() -> None:
    assert table.backend_options(["-O3"]) == ()
    assert table.backend_options(["-O3", "-mllvm", f"{OPTION}=1"]) == (OPTION,)
    # COMGR also takes the joined spelling, and it reaches the same parser.
    assert table.backend_options(["-O3", f"-mllvm={OPTION}=1"]) == (OPTION,)
    with pytest.raises(ValueError, match="'-mllvm' is the last"):
        table.backend_options(["-O3", "-mllvm"])
    with pytest.raises(ValueError, match="'-mllvm=' names no option"):
        table.backend_options(["-O3", "-mllvm="])


# --- the check, against the committed answers -------------------------------------


def test_refuses_an_option_the_compiler_does_not_accept() -> None:
    major = _measured_major(accepted=False)
    with pytest.raises(CompilerOptionNotAccepted) as err:
        table.require_accepted(
            ["-O3", "-mllvm", OPTION],
            compiler=_compiler(major, 0, 0),
            needed_by="kernel 'k_vgpr_form'",
        )
    text = str(err.value)
    assert "kernel 'k_vgpr_form'" in text
    assert f"'-mllvm {OPTION}'" in text
    assert f"LLVM {major}.0.0" in text
    assert isinstance(err.value, ValueError)


def test_accepts_an_option_the_compiler_accepts() -> None:
    major = _measured_major(accepted=True)
    table.require_accepted(
        ["-O3", "-mllvm", OPTION], compiler=_compiler(major, 0, 0), needed_by="k"
    )


def test_an_unmeasured_compiler_is_an_error_not_a_guess() -> None:
    with pytest.raises(ValueError, match="not been measured on LLVM 999") as err:
        table.require_accepted(
            ["-mllvm", OPTION], compiler=_compiler(999, 0, 0), needed_by="k"
        )
    assert not isinstance(err.value, CompilerOptionNotAccepted)


def test_an_unmeasured_option_is_an_error_not_a_guess() -> None:
    with pytest.raises(ValueError, match="never been measured"):
        table.require_accepted(
            ["-mllvm", "-amdgpu-no-such-option"],
            compiler=_compiler(_measured_major(accepted=True), 0, 0),
            needed_by="k",
        )


def test_options_without_mllvm_need_no_compiler_identity() -> None:
    table.require_accepted(["-O3"], compiler=_compiler(), needed_by="k")
    with pytest.raises(ValueError, match="LLVM version is unknown"):
        table.require_accepted(["-mllvm", OPTION], compiler=_compiler(), needed_by="k")


# --- real COMGR, in a child process -------------------------------------------------


def _loaded_major() -> int:
    info = comgr.loaded_compiler_info()
    if info is None or info.llvm_version is None:
        pytest.skip("no queryable COMGR library loads on this host")
    return info.llvm_version[0]


def _child_env() -> dict:
    """This process's environment, with the COMGR it loaded named explicitly.

    A child that resolved COMGR on its own could load a different compiler (for
    example the system one, where this process loaded torch's), and then answer
    for a compiler other than the one the expected outcome was read for.
    """
    info = comgr.loaded_compiler_info()
    env = dict(os.environ, AMD_COMGR_CACHE="0")
    if info is not None and (info.comgr_path or info.requested_comgr):
        env["ROCKE_COMGR_LIB"] = info.comgr_path or info.requested_comgr
    return env


def _run_child(body: str) -> subprocess.CompletedProcess:
    env = _child_env()
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (PKG_ROOT, env.get("PYTHONPATH", "")) if p
    )
    script = textwrap.dedent(
        """
        from rocke.runtime.compiler_options import CompilerOptionNotAccepted
        try:
            n = run()
        except CompilerOptionNotAccepted as ex:
            print("REFUSED", ex)
        except ValueError as ex:
            print("UNMEASURED", ex)
        else:
            print("COMPILED", n)
        """
    )
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body) + script],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
        check=False,
    )


def _expected_outcome(major: int) -> str:
    answer = table.lookup(OPTION, major)
    if answer is None:
        return "UNMEASURED"
    return "COMPILED" if answer.accepted else "REFUSED"


_KERNEL = """
from rocke.core.ir import KernelDef, Region
from rocke.helpers.compile import compile_kernel

def run():
    kernel = KernelDef("k_vgpr_form", [], Region("entry"))
    kernel.attrs["mfma_vgpr_form"] = True
    return len(compile_kernel(kernel, arch="gfx942", backend="python").hsaco)
"""


def _raw_compile(options: list) -> str:
    """A child body that hands ``options`` to the public COMGR entry point."""
    return f"""
from rocke.runtime.comgr import build_hsaco_from_llvm_ir

def run():
    ir = "define amdgpu_kernel void @k() {{\\n  ret void\\n}}\\n"
    hsaco, _ = build_hsaco_from_llvm_ir(
        ir, isa="amdgcn-amd-amdhsa--gfx942", options={options!r}
    )
    return len(hsaco)
"""


_RAW_COMPILE = _raw_compile(["-O3", "-mllvm", OPTION])
_RAW_COMPILE_JOINED = _raw_compile(["-O3", f"-mllvm={OPTION}"])


@pytest.mark.parametrize(
    "body",
    [_KERNEL, _RAW_COMPILE, _RAW_COMPILE_JOINED],
    ids=["compile_kernel", "build_hsaco_from_llvm_ir", "joined_mllvm_spelling"],
)
def test_compile_follows_the_table_and_never_exits_the_process(body: str) -> None:
    """On a compiler that accepts the option the kernel compiles as before; on
    one that does not, the compile is refused with the kernel, option and
    compiler named. Either way the process survives to say so."""
    expected = _expected_outcome(_loaded_major())
    proc = _run_child(body)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.startswith(expected), proc.stdout + proc.stderr
    if expected == "REFUSED":
        assert f"'-mllvm {OPTION}'" in proc.stdout
    if expected == "REFUSED" and body is _KERNEL:
        assert "kernel 'k_vgpr_form'" in proc.stdout


def test_committed_column_matches_this_compiler() -> None:
    """Re-measuring this host's compiler gives the committed answers."""
    tool = PLATFORM / "tools" / "gen_compiler_options.py"
    if not tool.is_file():
        pytest.skip(f"source-tree check: {tool.name} is not installed")
    major = _loaded_major()
    if table.lookup(OPTION, major) is None:
        pytest.skip(f"no committed column for LLVM {major}")
    proc = subprocess.run(
        [sys.executable, str(tool), "--check"],
        capture_output=True,
        text=True,
        env=_child_env(),
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
