# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The measured table of ``-mllvm`` options each compiler accepts, and the
check that keeps an option the loaded compiler does not know away from COMGR.

Most tests run in this process and need no COMGR: every public compile entry
point is called with the loaded compiler's identity pinned to one the table
records as refusing the option, and must refuse before COMGR is reached.

LLVM's option parser exits the whole process on an unknown option, so the
tests that compile with the real loaded compiler do it in a child process: a
regression that lets the option through ends the child, and the test reports
it instead of pytest dying.

Whether the committed answers match the loaded compiler is the CTest gate
``rocke_compiler_options_reproducibility`` (``tools/gen_compiler_options.py
--check``), not a test here.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import textwrap
from collections.abc import Callable, Iterator
from unittest import mock

import pytest
from rocke.core.ir import KernelDef, Region
from rocke.helpers.compile import (
    KernelModule,
    accepted_comgr_options,
    compile_kernel,
    compile_module,
    lower_kernel_for_comgr,
)
from rocke.runtime import CompilerOptionNotAccepted, comgr
from rocke.runtime import compiler_options as table
from rocke.runtime.comgr import (
    CompilerInfo,
    build_hsaco_from_llvm_ir,
    build_hsaco_from_llvm_modules,
)
from rocke.tests.subprocess_launcher import rocke_python_command, rocke_subprocess_env

OPTION = "-amdgpu-mfma-vgpr-form"
ISA = "amdgcn-amd-amdhsa--gfx942"


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
    for major, answer in table.measured_answers()[OPTION].items():
        if answer.accepted is accepted:
            return major
    pytest.fail(f"the table has no LLVM version that {accepted=} for {OPTION}")


def _vgpr_form_kernel(name: str) -> KernelDef:
    """An empty kernel that asks for VGPR-form MFMAs, so it needs OPTION."""
    kernel = KernelDef(name, [], Region("entry"))
    kernel.attrs["mfma_vgpr_form"] = True
    return kernel


def _empty_ir(name: str) -> str:
    return f"define amdgpu_kernel void @{name}() {{\n  ret void\n}}\n"


# --- the table ------------------------------------------------------------------


def test_table_answers_are_measured_facts() -> None:
    answers = table.measured_answers()
    assert OPTION in answers
    for name, column in answers.items():
        assert column, f"{name} has no measured compiler"
        for major, answer in column.items():
            assert answer == table.lookup(name, major)
            assert answer.verified_on.startswith(f"LLVM {major}.")
            # A refusal carries the compiler's own words; an acceptance none.
            assert bool(answer.evidence) is not answer.accepted, (name, major)


def test_committed_file_is_what_format_table_writes() -> None:
    """The tool writes with format_table, so a regenerated table differs from
    the committed one only where a measurement changed."""
    text = table.TABLE_PATH.read_text(encoding="utf-8")
    assert table.format_table(table.parse_table(text)) == text
    assert table.parse_table(text) == table.measured_answers()


def _committed_with(old: str, new: str) -> str:
    """The committed table text with its first ``old`` replaced by ``new``."""
    text = table.TABLE_PATH.read_text(encoding="utf-8")
    assert old in text
    return text.replace(old, new, 1)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            table.SCHEMA,
            "rocke.compiler_options/v0",
            (
                r"schema='rocke\.compiler_options/v0', expected "
                r"'rocke\.compiler_options/v1'"
            ),
        ),
        # bool("false") is True: a string here must be refused, not read.
        (
            '"accepted": false',
            '"accepted": "false"',
            r"on LLVM 20 accepted='false', expected true or false",
        ),
        (
            '"verified_on": "LLVM 20.0.0"',
            '"verified_on": 20',
            r"on LLVM 20 evidence=.* and verified_on=20, expected strings",
        ),
        (
            '"evidence": ',
            '"reason": ',
            (
                r"on LLVM 20 has missing keys \['evidence'\] and unexpected keys "
                r"\['reason'\]"
            ),
        ),
        ('"20": {', '"llvm20": {', r"has the keys \['llvm20'\], expected LLVM major"),
        # A JSON syntax error names the file too, not only a line and column.
        ('"options": {', '"options": [', r"not valid JSON: "),
    ],
    ids=["schema", "accepted_not_bool", "verified_on_not_str", "key", "major", "json"],
)
def test_parse_table_refuses_a_malformed_table(
    old: str, new: str, message: str
) -> None:
    """Each refusal names the file and the entry that is wrong."""
    with pytest.raises(ValueError, match=message) as err:
        table.parse_table(_committed_with(old, new), source="edited.json")
    assert str(err.value).startswith("edited.json: ")


@pytest.mark.parametrize(
    ("path", "message"),
    [
        (("options",), r"options=\[\], expected an object mapping each option"),
        (
            ("options", OPTION),
            rf"'{OPTION}' is \[\], expected an object keyed by LLVM major",
        ),
        (
            ("options", OPTION, "20"),
            (
                rf"'{OPTION}' on LLVM 20 is \[\], expected an object with the keys "
                "accepted, evidence, verified_on"
            ),
        ),
    ],
    ids=["options", "column", "cell"],
)
def test_parse_table_refuses_a_list_where_an_object_belongs(
    path: tuple[str, ...], message: str
) -> None:
    """Every level of the table must be a JSON object; a list at any level is
    refused with the file and the entry named, not read or indexed."""
    doc = json.loads(table.TABLE_PATH.read_text(encoding="utf-8"))
    parent = doc
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = []
    with pytest.raises(ValueError, match=message) as err:
        table.parse_table(json.dumps(doc), source="edited.json")
    assert str(err.value).startswith("edited.json: ")


def test_parse_table_names_the_committed_file_by_default() -> None:
    with pytest.raises(ValueError) as err:
        table.parse_table(_committed_with(table.SCHEMA, "rocke.compiler_options/v0"))
    assert str(err.value).startswith(f"{table.TABLE_PATH}: ")


# --- the drift the reproducibility gate reports -----------------------------------


def test_column_drift_is_empty_when_the_column_reproduces() -> None:
    committed = table.measured_answers()
    for major in committed[OPTION]:
        measured = {
            option: column[major].accepted
            for option, column in committed.items()
            if major in column
        }
        assert table.column_drift(committed, measured, llvm_major=major) == []


def test_column_drift_reports_a_changed_answer() -> None:
    major = _measured_major(accepted=False)
    drift = table.column_drift(
        table.measured_answers(), {OPTION: True}, llvm_major=major
    )
    assert drift == [
        f"{OPTION} on LLVM {major}: committed accepted=False, measured accepted=True"
    ]


def test_column_drift_reports_a_missing_answer() -> None:
    drift = table.column_drift(
        table.measured_answers(),
        {OPTION: True, "-amdgpu-no-such-option": False},
        llvm_major=999,
    )
    assert drift == [
        f"{OPTION} on LLVM 999: no committed answer",
        "-amdgpu-no-such-option on LLVM 999: no committed answer",
    ]


def test_backend_options_reads_only_mllvm_values() -> None:
    assert table.backend_options(["-O3"]) == ()
    assert table.backend_options(["-O3", "-mllvm", f"{OPTION}=1"]) == (OPTION,)
    # COMGR also takes the joined spelling, and it reaches the same parser.
    assert table.backend_options(["-O3", f"-mllvm={OPTION}=1"]) == (OPTION,)
    with pytest.raises(ValueError, match="'-mllvm' is the last"):
        table.backend_options(["-O3", "-mllvm"])
    with pytest.raises(ValueError, match="'-mllvm=' names no option"):
        table.backend_options(["-O3", "-mllvm="])


def test_the_refusal_is_public_in_rocke_runtime() -> None:
    import rocke.runtime

    assert "CompilerOptionNotAccepted" in rocke.runtime.__all__
    assert CompilerOptionNotAccepted is table.CompilerOptionNotAccepted
    assert issubclass(CompilerOptionNotAccepted, ValueError)


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


# --- every public entry point refuses, in this process ------------------------------


@pytest.fixture
def refusing_compiler_loaded() -> Iterator[CompilerInfo]:
    """Report, for this test, a loaded compiler the table records as refusing
    OPTION.

    The identity is the only thing replaced: nothing reaches COMGR while the
    check holds, so this runs on any host. An entry point that lost its check
    would go on to the real COMGR, and whatever happens there (a compile, a
    ``ComgrError``, or the process ending where the real compiler refuses the
    option) fails the test.
    """
    refusing = _compiler(_measured_major(accepted=False), 0, 0)
    with mock.patch.object(comgr, "loaded_compiler_info", return_value=refusing):
        yield refusing


# Each public entry point that can hand OPTION to COMGR, called as a user
# would, and the name its refusal gives to what is being compiled.
_ENTRY_POINTS: dict[str, tuple[Callable[[], object], str]] = {
    "compile_kernel": (
        lambda: compile_kernel(
            _vgpr_form_kernel("k_vgpr_form"), arch="gfx942", backend="python"
        ),
        "kernel 'k_vgpr_form'",
    ),
    "compile_module": (
        lambda: compile_module(
            KernelModule(
                primary=_vgpr_form_kernel("k_primary"),
                companions={"merge": _vgpr_form_kernel("k_merge")},
            ),
            arch="gfx942",
            backend="python",
        ),
        "KernelModule kernels ['k_primary', 'k_merge']",
    ),
    "lower_kernel_for_comgr": (
        lambda: lower_kernel_for_comgr(
            _vgpr_form_kernel("k_vgpr_form"), arch="gfx942", backend="python"
        ),
        "kernel 'k_vgpr_form'",
    ),
    "accepted_comgr_options": (
        lambda: accepted_comgr_options(_vgpr_form_kernel("k_vgpr_form")),
        "kernel 'k_vgpr_form'",
    ),
    "build_hsaco_from_llvm_ir": (
        lambda: build_hsaco_from_llvm_ir(
            _empty_ir("k"), isa=ISA, options=["-O3", "-mllvm", OPTION]
        ),
        "this COMGR compile",
    ),
    "build_hsaco_from_llvm_ir_fresh_process": (
        lambda: build_hsaco_from_llvm_ir(
            _empty_ir("k"),
            isa=ISA,
            options=["-O3", "-mllvm", OPTION],
            fresh_process=True,
        ),
        "this COMGR compile",
    ),
    "build_hsaco_from_llvm_ir_joined_spelling": (
        lambda: build_hsaco_from_llvm_ir(
            _empty_ir("k"), isa=ISA, options=["-O3", f"-mllvm={OPTION}"]
        ),
        "this COMGR compile",
    ),
    "build_hsaco_from_llvm_modules": (
        lambda: build_hsaco_from_llvm_modules(
            [_empty_ir("k_a"), _empty_ir("k_b")],
            isa=ISA,
            options=["-O3", "-mllvm", OPTION],
        ),
        "this COMGR compile",
    ),
}


@pytest.mark.parametrize("entry_point", list(_ENTRY_POINTS))
def test_every_entry_point_refuses_before_comgr(
    entry_point: str, refusing_compiler_loaded: CompilerInfo
) -> None:
    call, needed_by = _ENTRY_POINTS[entry_point]
    with pytest.raises(CompilerOptionNotAccepted) as err:
        call()
    text = str(err.value)
    assert text.startswith(f"{needed_by} needs the compiler option '-mllvm {OPTION}'")
    assert refusing_compiler_loaded.describe() in text


# --- the child-process environment --------------------------------------------------


def _loaded_compiler() -> CompilerInfo:
    info = comgr.loaded_compiler_info()
    if info is None or info.llvm_version is None:
        pytest.skip("no queryable COMGR library loads on this host")
    return info


@pytest.mark.parametrize(
    ("comgr_path", "expected_lib"),
    [
        ("/opt/rocm/lib/libamd_comgr.so", "/opt/rocm/lib/libamd_comgr.so"),
        # The loader could not say which file it mapped: the name it was asked for.
        (None, "libamd_comgr.so"),
    ],
    ids=["mapped_path", "requested_name"],
)
def test_comgr_child_env_names_the_loaded_library(
    comgr_path: str | None, expected_lib: str
) -> None:
    loaded = dataclasses.replace(_compiler(20, 0, 0), comgr_path=comgr_path)
    with mock.patch.object(comgr, "loaded_compiler_info", return_value=loaded):
        env = comgr.comgr_child_env({"KEPT": "1", "AMD_COMGR_CACHE": "1"})
    assert env == {
        "KEPT": "1",
        "ROCKE_COMGR_LIB": expected_lib,
        "AMD_COMGR_CACHE": "0",
    }


def test_comgr_child_env_refuses_without_a_loaded_library() -> None:
    with (
        mock.patch.object(comgr, "loaded_compiler_info", return_value=None),
        pytest.raises(comgr.ComgrError, match="COMGR could not be loaded"),
    ):
        comgr.comgr_child_env()


# --- real COMGR, in a child process -------------------------------------------------


def _run_child(body: str) -> subprocess.CompletedProcess:
    script = textwrap.dedent(
        """
        from rocke.runtime import CompilerOptionNotAccepted
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
        rocke_python_command("-c", textwrap.dedent(body) + script),
        capture_output=True,
        text=True,
        env=rocke_subprocess_env(comgr.comgr_child_env()),
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

_KERNEL_OPTIONS = """
from rocke.core.ir import KernelDef, Region
from rocke.helpers.compile import accepted_comgr_options

def run():
    kernel = KernelDef("k_vgpr_form", [], Region("entry"))
    kernel.attrs["mfma_vgpr_form"] = True
    return len(accepted_comgr_options(kernel))
"""


def _raw_compile(options: list[str]) -> str:
    """A child body that hands ``options`` to the public COMGR entry point."""
    return f"""
from rocke.runtime.comgr import build_hsaco_from_llvm_ir

def run():
    ir = {_empty_ir("k")!r}
    hsaco, _ = build_hsaco_from_llvm_ir(ir, isa={ISA!r}, options={options!r})
    return len(hsaco)
"""


_RAW_COMPILE = _raw_compile(["-O3", "-mllvm", OPTION])
_RAW_COMPILE_JOINED = _raw_compile(["-O3", f"-mllvm={OPTION}"])


@pytest.mark.parametrize(
    "body",
    [_KERNEL, _KERNEL_OPTIONS, _RAW_COMPILE, _RAW_COMPILE_JOINED],
    ids=[
        "compile_kernel",
        "accepted_comgr_options",
        "build_hsaco_from_llvm_ir",
        "joined_mllvm_spelling",
    ],
)
def test_compile_follows_the_table_and_never_exits_the_process(body: str) -> None:
    """On a compiler that accepts the option the kernel compiles as before; on
    one that does not, the compile is refused with the kernel, option and
    compiler named. Either way the process survives to say so."""
    expected = _expected_outcome(_loaded_compiler().llvm_version[0])
    proc = _run_child(body)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.startswith(expected), proc.stdout + proc.stderr
    if expected == "REFUSED":
        assert f"'-mllvm {OPTION}'" in proc.stdout
    if expected == "REFUSED" and body in (_KERNEL, _KERNEL_OPTIONS):
        assert "kernel 'k_vgpr_form'" in proc.stdout
