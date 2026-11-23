# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""How the library's compile tests apply the platform's compiler-tool policy
(``rocke.tests.compiler_tools``, through ``_compiler_test_utils``).

CPU only. Each error is a real one from the committed option table: a
compiler whose LLVM major the table records as refusing the option
(``CompilerOptionNotAccepted``), or one the table has no answer for (a plain
``ValueError``). These tests pin that a refusal skips, that it fails when
``ROCKE_REQUIRE_COMPILER_TOOLS=1`` requires the tools, and that no other error
is turned into a skip. They also pin that ``compile_or_skip``, the compile
helper of the attention backward tests, applies the same policy to a real
``compile_kernel`` call, and that ``compiled_resources``, the GDN/KDA one,
applies it where no ``llvm-objdump`` can read the code object.
"""

from __future__ import annotations

import unittest

import pytest
from rocke.core.ir import KernelDef, Op, Region
from rocke.helpers import compile as compile_module
from rocke.helpers.compile import compile_kernel
from rocke.runtime import comgr
from rocke.runtime.comgr import CompilerInfo
from rocke.runtime.compiler_options import measured_answers, require_accepted
from rocke.tests.compiler_tools import REQUIRE_COMPILER_TOOLS_ENV, require_tools_or_skip
from rocke.tests.lanes import CI_TEST_TOOLS, skip_until_reason

from . import _attention_bwd_resources, _parallel_compile
from ._attention_bwd_resources import compile_or_skip
from ._compiler_test_utils import (
    skip_where_compiler_refuses_option,
    skip_where_compiler_refuses_processor,
)
from ._gdn_decode_compile import compile_each, compiled_resources

OPTION = "-amdgpu-mfma-vgpr-form"
# How the refusal starts, and how the policy introduces it.
REFUSAL = f"kernel 'probe' needs the compiler option '-mllvm {OPTION}'"
SKIP_REASON = f"loaded compiler cannot build this kernel: {REFUSAL}"
# A missing or refusing tool skips as waiting for the CI test environment.
TOOLS_SKIP = skip_until_reason(CI_TEST_TOOLS, reason=SKIP_REASON)


def _compiler(llvm_major: int) -> CompilerInfo:
    """A compiler identity as ``loaded_compiler_info`` reports one."""
    return CompilerInfo(
        llvm_version=(llvm_major, 0, 0),
        source="LLVMGetVersion",
        requested_comgr="libamd_comgr.so",
        comgr_path=None,
        query_library_path=None,
    )


def _compile_on(llvm_major: int) -> None:
    """Ask the committed table whether LLVM ``llvm_major`` accepts ``OPTION``,
    as ``compile_kernel`` does before compiling a kernel that needs it."""
    require_accepted(
        ["-O3", "-mllvm", OPTION],
        compiler=_compiler(llvm_major),
        needed_by="kernel 'probe'",
    )


def _refusing_major() -> int:
    """An LLVM major the table records as refusing ``OPTION``."""
    return next(
        major
        for major, answer in measured_answers()[OPTION].items()
        if not answer.accepted
    )


def _unmeasured_major() -> int:
    """An LLVM major newer than every one the table has an answer for."""
    return 1 + max(measured_answers()[OPTION])


def test_refused_option_skips(monkeypatch):
    monkeypatch.delenv(REQUIRE_COMPILER_TOOLS_ENV, raising=False)
    with (
        pytest.raises(unittest.SkipTest, match=f"^{TOOLS_SKIP}"),
        skip_where_compiler_refuses_option(),
    ):
        _compile_on(_refusing_major())


def test_refused_option_fails_where_tools_are_required(monkeypatch):
    monkeypatch.setenv(REQUIRE_COMPILER_TOOLS_ENV, "1")
    required = f"^{REQUIRE_COMPILER_TOOLS_ENV}=1 but {SKIP_REASON}"
    with (
        pytest.raises(AssertionError, match=required),
        skip_where_compiler_refuses_option(),
    ):
        _compile_on(_refusing_major())


@pytest.mark.parametrize("required", ["0", "1"])
def test_other_errors_propagate(monkeypatch, required):
    # A compiler the table has no answer for raises a plain ValueError (the
    # base class of the refusal): that is a missing measurement, not a skip.
    monkeypatch.setenv(REQUIRE_COMPILER_TOOLS_ENV, required)
    with (
        pytest.raises(
            ValueError, match="which has not been measured on LLVM"
        ) as caught,
        skip_where_compiler_refuses_option(),
    ):
        _compile_on(_unmeasured_major())
    assert type(caught.value) is ValueError


def _vgpr_form_kernel() -> KernelDef:
    """An empty kernel that asks for VGPR-form MFMAs, so it needs ``OPTION``."""
    kernel = KernelDef("probe", [], Region("entry"))
    kernel.attrs["mfma_vgpr_form"] = True
    return kernel


@pytest.mark.parametrize(
    "toolchain_gap",
    [None, "flavor llvm20"],
    ids=["gate-toolchain", "other-toolchain"],
)
@pytest.mark.parametrize(
    ("required", "outcome", "reason"),
    [
        ("0", unittest.SkipTest, f"^{TOOLS_SKIP}"),
        ("1", AssertionError, f"^{REQUIRE_COMPILER_TOOLS_ENV}=1 but {SKIP_REASON}"),
    ],
    ids=["skips", "fails-where-tools-are-required"],
)
def test_compile_or_skip_applies_the_policy(
    monkeypatch, toolchain_gap, required, outcome, reason
):
    # The loaded compiler's identity, as the option check sees it, is replaced
    # by one the table records as refusing OPTION, so compile_kernel refuses
    # before COMGR on any host. compile_or_skip's own rule depends on whether
    # the host is the resource gate's toolchain, so both answers are pinned.
    refusing = _compiler(_refusing_major())
    monkeypatch.setattr(comgr, "loaded_compiler_info", lambda: refusing)
    monkeypatch.setattr(
        _attention_bwd_resources, "gate_toolchain_gap", lambda arch: toolchain_gap
    )
    monkeypatch.setenv(REQUIRE_COMPILER_TOOLS_ENV, required)
    try:
        with pytest.raises(outcome, match=reason):
            # The Python backend: a helper that lost the refusal must fail here
            # without needing the C++ engine.
            compile_or_skip(_vgpr_form_kernel(), arch="gfx942", backend="python")
    except pytest.skip.Exception as skipped:
        # Its own toolchain-gap skip would hide the refusal here, and pytest
        # would report this test as skipped rather than failed.
        pytest.fail(f"compile_or_skip skipped outside the shared policy: {skipped}")


# The ValueError rocke.core.arch.target raises for a processor the loaded
# compiler cannot target (the text test_arch_target_identity.py pins).
PROCESSOR_REFUSAL = (
    "the llvm20 compiler does not accept the processor 'gfx1250', and its "
    "arch_specs.json row points at no generic target that compiler accepts -- "
    "compile with an LLVM flavor whose compiler accepts 'gfx1250'"
)


def test_refused_processor_skips(monkeypatch):
    monkeypatch.delenv(REQUIRE_COMPILER_TOOLS_ENV, raising=False)
    with pytest.raises(unittest.SkipTest, match="cannot target this processor"):
        with skip_where_compiler_refuses_processor():
            raise ValueError(PROCESSOR_REFUSAL)


def test_refused_processor_fails_where_tools_are_required(monkeypatch):
    monkeypatch.setenv(REQUIRE_COMPILER_TOOLS_ENV, "1")
    with pytest.raises(AssertionError, match=REQUIRE_COMPILER_TOOLS_ENV):
        with skip_where_compiler_refuses_processor():
            raise ValueError(PROCESSOR_REFUSAL)


@pytest.mark.parametrize("required", ["0", "1"])
@pytest.mark.parametrize(
    "message",
    [
        "variant rejected",
        # One of the two phrases alone is not the refusal.
        "the llvm20 compiler does not accept the processor 'gfx1250'",
    ],
)
def test_other_value_errors_propagate_from_the_processor_policy(
    monkeypatch, required, message
):
    monkeypatch.setenv(REQUIRE_COMPILER_TOOLS_ENV, required)
    with pytest.raises(ValueError, match="variant rejected|does not accept"):
        with skip_where_compiler_refuses_processor():
            raise ValueError(message)


@pytest.mark.parametrize(
    ("required", "outcome", "reason"),
    [
        (
            "0",
            unittest.SkipTest,
            "^"
            + skip_until_reason(
                CI_TEST_TOOLS, reason="HSACO introspection unavailable: missing tool"
            ),
        ),
        (
            "1",
            AssertionError,
            f"^{REQUIRE_COMPILER_TOOLS_ENV}=1 but HSACO introspection unavailable",
        ),
    ],
    ids=["skips", "fails-where-tools-are-required"],
)
def test_gdn_compiled_resources_applies_the_policy(
    monkeypatch, tmp_path, required, outcome, reason
):
    # The GDN/KDA resource checks compile through COMGR, then read the code
    # object with the llvm-objdump on PATH. An empty PATH is a host without it.
    kernel = KernelDef("probe", [], Region("entry"))
    try:
        compile_kernel(kernel, arch="gfx950", capture_ir_text=False)
    except ImportError as exc:
        # Before the parametrized requirement is set: the run's own policy
        # decides whether a host without COMGR skips or fails.
        require_tools_or_skip(f"comgr toolchain unavailable: {exc}")
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv(REQUIRE_COMPILER_TOOLS_ENV, required)
    with pytest.raises(outcome, match=reason):
        compiled_resources(kernel)


def test_a_batch_reports_each_kernels_failure_on_that_kernel(monkeypatch):
    # One kernel lowers and compiles, one fails to lower, one fails to compile:
    # each outcome names its own kernel, and the failures raise only when their
    # own kernel is asked. One worker, so the compile stand-in applies.
    good = KernelDef("batch_good", [], Region("entry"))
    unlowerable = KernelDef("batch_unlowerable", [], Region("entry", [Op("no.op")]))
    uncompilable = KernelDef("batch_uncompilable", [], Region("entry"))
    try:
        compile_kernel(good, arch="gfx950", capture_ir_text=False)
    except ImportError as exc:
        require_tools_or_skip(f"comgr toolchain unavailable: {exc}")
    compile_one = compile_module.compile_comgr_input

    def compile_or_refuse(comgr_input):
        if comgr_input.kernel_name == uncompilable.name:
            raise comgr.ComgrError("refused")
        return compile_one(comgr_input)

    monkeypatch.setattr(compile_module, "compile_comgr_input", compile_or_refuse)
    monkeypatch.setattr(_parallel_compile, "available_cpus", lambda: 1)
    outcomes = compile_each([good, unlowerable, uncompilable])
    assert [o.kernel_name for o in outcomes] == [
        good.name,
        unlowerable.name,
        uncompilable.name,
    ]
    assert outcomes[0].error is None and outcomes[0].hsaco[:4] == b"\x7fELF"
    with pytest.raises(NotImplementedError, match="no.op"):
        outcomes[1].scratch_bytes()
    with pytest.raises(comgr.ComgrError, match="refused"):
        outcomes[2].scratch_bytes()
