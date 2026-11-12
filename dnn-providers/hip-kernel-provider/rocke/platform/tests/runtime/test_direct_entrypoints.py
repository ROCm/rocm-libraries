# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Entry-point contract of ``ROCKE_CODEOBJ_BACKEND=direct``.

The LLVM-IR entries always use COMGR (the direct compiler reads rocKE IR), and
``compile_kernel`` / ``compile_module`` report which compiler built each kernel
through ``backend_usage()`` and ``ROCKE_DIRECT_LOG``. Host-only: lowering,
COMGR and the direct compiler are mocked.
"""

from __future__ import annotations

from unittest import mock

import pytest
from rocke.core.ir import KernelDef, Region
from rocke.core.lower_llvm import _detect_llvm_flavor
from rocke.helpers import KernelModule, compile_kernel, compile_module
from rocke.helpers import compile as compile_impl
from rocke.runtime import code_object, comgr
from rocke.runtime.code_object import CodeObjectError, DirectUnsupported

_REFUSAL = (
    "direct rocKE IR compiler rejected the kernel: "
    "ROCKE_DIRECT_UNSUPPORTED: inline asm instruction v_mov_b32_sdwa"
)


@pytest.fixture(autouse=True)
def direct_environment(monkeypatch):
    for name in (
        "ROCKE_DIRECT_STRICT",
        "ROCKE_DIRECT_FALLBACK",
        "ROCKE_DIRECT_ALLOW",
        "ROCKE_DIRECT_DENY",
        "ROCKE_DIRECT_LOG",
        "ROCKE_LLVM_FLAVOR",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ROCKE_CODEOBJ_BACKEND", "direct")
    code_object.reset_backend_usage()
    yield
    code_object.reset_backend_usage()


@pytest.fixture
def comgr_build(monkeypatch):
    """COMGR (lowering and code generation) replaced by mocks."""
    monkeypatch.setattr(compile_impl, "_lower_llvm_via_backend", lambda *a, **k: "ll")
    monkeypatch.setattr(
        compile_impl, "_code_object_llvm_flavor", lambda backend: "llvm22"
    )
    build = mock.Mock(return_value=(b"comgr-hsaco", code_object.CodeObjectTimings()))
    monkeypatch.setattr(compile_impl, "build_hsaco_from_llvm_ir", build)
    link = mock.Mock(return_value=(b"comgr-module", code_object.CodeObjectTimings()))
    monkeypatch.setattr(compile_impl, "build_hsaco_from_llvm_modules", link)
    return build, link


def _direct(monkeypatch, outcome):
    """``_compile_kernel_direct`` returning an artifact or raising ``outcome``."""

    def compile_direct(kernel, **_):
        if isinstance(outcome, Exception):
            raise outcome
        return compile_impl.KernelArtifact(
            kernel=kernel,
            ir_text="",
            llvm_text="",
            hsaco=b"direct-hsaco",
            code_object_backend="direct",
        )

    monkeypatch.setattr(compile_impl, "_compile_kernel_direct", compile_direct)


def _kernel(name: str = "entry_probe") -> KernelDef:
    return KernelDef(name, [], Region("entry"))


def test_llvm_entries_use_comgr_whatever_the_environment_says(monkeypatch) -> None:
    timings = comgr.ComgrTimings(bc=1.0)
    single = mock.Mock(return_value=(b"one", timings))
    linked = mock.Mock(return_value=(b"two", timings))
    monkeypatch.setattr(comgr, "build_hsaco_from_llvm_ir", single)
    monkeypatch.setattr(comgr, "build_hsaco_from_llvm_modules", linked)

    assert code_object.build_hsaco_from_llvm_ir("IR")[0] == b"one"
    assert code_object.build_hsaco_from_llvm_modules(["a", "b"])[0] == b"two"
    assert single.call_count == linked.call_count == 1


def test_an_explicit_direct_request_of_an_llvm_entry_is_an_error() -> None:
    with pytest.raises(CodeObjectError, match="not LLVM IR"):
        code_object.build_hsaco_from_llvm_ir("IR", backend="direct")
    with pytest.raises(CodeObjectError, match="not LLVM IR"):
        code_object.build_hsaco_from_llvm_modules(["a", "b"], backend="direct")


def test_llvm_flavor_detection_asks_comgr_not_the_direct_compiler(monkeypatch) -> None:
    info = code_object.CodeObjectCompilerInfo(
        backend="comgr",
        llvm_version=(22, 0, 0),
        source="test",
        identity="comgr:test",
    )
    monkeypatch.setattr(code_object, "_comgr_compiler_info", lambda: info)
    assert _detect_llvm_flavor() == "llvm22"


def test_kernels_built_by_direct_are_counted_and_logged(monkeypatch, capsys) -> None:
    monkeypatch.setenv("ROCKE_DIRECT_LOG", "1")
    _direct(monkeypatch, None)

    artifact = compile_kernel(_kernel("counted"), arch="gfx950")

    assert artifact.code_object_backend == "direct"
    usage = code_object.backend_usage()
    assert (usage["direct"], usage["comgr"], usage["fallback"]) == (1, 0, 0)
    assert capsys.readouterr().err == (
        "rocke-direct: kernel=counted backend=direct fallback=-\n"
    )


def test_a_refusal_falls_back_to_comgr_and_is_counted(
    monkeypatch, capsys, comgr_build
) -> None:
    monkeypatch.setenv("ROCKE_DIRECT_LOG", "1")
    _direct(monkeypatch, DirectUnsupported(_REFUSAL))

    artifact = compile_kernel(_kernel("refused"), arch="gfx950")

    assert artifact.code_object_backend == "comgr"
    assert artifact.hsaco == b"comgr-hsaco"
    assert artifact.direct_fallback == _REFUSAL
    reason = "ROCKE_DIRECT_UNSUPPORTED: inline asm instruction v_mov_b32_sdwa"
    usage = code_object.backend_usage()
    assert (usage["direct"], usage["comgr"], usage["fallback"]) == (0, 1, 1)
    assert usage["fallback_reasons"] == {reason: 1}
    assert capsys.readouterr().err == (
        f"rocke-direct: kernel=refused backend=comgr fallback={reason}\n"
    )


def test_strict_mode_raises_a_refusal_and_counts_it(monkeypatch, comgr_build) -> None:
    monkeypatch.setenv("ROCKE_DIRECT_STRICT", "1")
    _direct(monkeypatch, DirectUnsupported(_REFUSAL))

    with pytest.raises(DirectUnsupported):
        compile_kernel(_kernel(), arch="gfx950")

    usage = code_object.backend_usage()
    assert (usage["refused"], usage["comgr"]) == (1, 0)
    comgr_build[0].assert_not_called()


def test_other_direct_errors_fall_back_only_when_asked(
    monkeypatch, comgr_build
) -> None:
    _direct(monkeypatch, CodeObjectError("ROCKE_DIRECT_REGISTERS: out of VGPRs"))

    with pytest.raises(CodeObjectError, match="out of VGPRs"):
        compile_kernel(_kernel(), arch="gfx950")
    assert code_object.backend_usage()["refused"] == 1

    monkeypatch.setenv("ROCKE_DIRECT_FALLBACK", "all")
    artifact = compile_kernel(_kernel(), arch="gfx950")
    assert artifact.code_object_backend == "comgr"
    assert "out of VGPRs" in artifact.direct_fallback

    monkeypatch.setenv("ROCKE_DIRECT_STRICT", "1")
    with pytest.raises(CodeObjectError):
        compile_kernel(_kernel(), arch="gfx950")

    monkeypatch.delenv("ROCKE_DIRECT_STRICT")
    monkeypatch.setenv("ROCKE_DIRECT_FALLBACK", "sometimes")
    with pytest.raises(ValueError, match="ROCKE_DIRECT_FALLBACK"):
        compile_kernel(_kernel(), arch="gfx950")


@pytest.mark.parametrize(
    ("allow", "deny", "name", "direct"),
    [
        ("", "", "attn_fwd_d128", True),
        ("attn_*", "", "attn_fwd_d128", True),
        ("attn_*", "", "conv_fprop", False),
        ("", "attn_*d128", "attn_fwd_d128", False),
        ("attn_*,conv_*", "conv_wgrad*", "conv_wgrad_bf16", False),
        ("attn_*,conv_*", "conv_wgrad*", "conv_fprop", True),
    ],
)
def test_allow_and_deny_lists_pick_the_kernels_direct_compiles(
    monkeypatch, comgr_build, allow, deny, name, direct
) -> None:
    monkeypatch.setenv("ROCKE_DIRECT_ALLOW", allow)
    monkeypatch.setenv("ROCKE_DIRECT_DENY", deny)
    monkeypatch.setenv(
        "ROCKE_DIRECT_STRICT", "1"
    )  # a listed-out kernel is not a refusal
    _direct(monkeypatch, None)

    artifact = compile_kernel(_kernel(name), arch="gfx950")

    assert artifact.code_object_backend == ("direct" if direct else "comgr")
    assert bool(artifact.direct_fallback) is not direct
    usage = code_object.backend_usage()
    assert (usage["direct"], usage["comgr"], usage["fallback"]) == (
        (1, 0, 0) if direct else (0, 1, 1)
    )


def test_a_module_compiles_with_comgr_under_the_direct_environment(
    monkeypatch, comgr_build
) -> None:
    monkeypatch.setenv("ROCKE_DIRECT_STRICT", "1")
    module = KernelModule(_kernel("primary"), {"merge": _kernel("merge")})

    artifact = compile_module(module, arch="gfx950")

    assert artifact.hsaco == b"comgr-module"
    assert artifact.direct_fallback == ""
    comgr_build[1].assert_called_once()
    assert comgr_build[1].call_args.kwargs["backend"] == "comgr"
    usage = code_object.backend_usage()
    assert (usage["comgr"], usage["fallback"]) == (1, 0)


def test_an_explicit_direct_module_request_is_a_refusal(
    monkeypatch, comgr_build
) -> None:
    module = KernelModule(_kernel("primary"), {"merge": _kernel("merge")})

    artifact = compile_module(module, arch="gfx950", code_object_backend="direct")
    assert artifact.hsaco == b"comgr-module"
    assert "one kernel per code object" in artifact.direct_fallback
    assert code_object.backend_usage()["fallback"] == 1

    monkeypatch.setenv("ROCKE_DIRECT_STRICT", "1")
    with pytest.raises(DirectUnsupported, match="one kernel per code object"):
        compile_module(module, arch="gfx950", code_object_backend="direct")
    assert code_object.backend_usage()["refused"] == 1


def test_reset_clears_the_counters(monkeypatch) -> None:
    _direct(monkeypatch, None)
    compile_kernel(_kernel(), arch="gfx950")
    assert code_object.backend_usage()["direct"] == 1
    code_object.reset_backend_usage()
    assert code_object.backend_usage() == {
        "direct": 0,
        "comgr": 0,
        "fallback": 0,
        "refused": 0,
        "fallback_reasons": {},
    }
