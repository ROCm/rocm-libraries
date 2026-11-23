# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rocke.core.native_engine, the C++ engine's serialized-IR lowering over its C
ABI: the same text as the Python lowerer, and each refusal with the engine's
status, its reason and the documented exception type.

The lowering cases need the shared engine and skip, waiting for the
shared-library lane, where it is not found. The cases about finding the
library run a copy of the module and the modules it imports, all standard
library only, in a child process with a package tree of its own, so the
library it finds (or does not) is decided by that tree and its environment
alone.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from rocke import assets
from rocke.core import engine_provenance, native_engine
from rocke.core.arch import domain as arch_domain
from rocke.core.ir import F32, IRBuilder, PtrType
from rocke.core.ir_serialize import serialize
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import (
    LLVM_FLAVORS,
    ArchDomainError,
    _lower_kernel_to_llvm_python,
)
from rocke.tests.engines import native_engine_or_skip
from rocke.tests.lanes import BINDING_LANE, import_or_skip_until

_ARCH = "gfx942"


def _copy_kernel():
    """A kernel with a parameter, a load, a store and arithmetic."""
    b = IRBuilder("native_engine_copy")
    src = b.param("src", PtrType(F32, "global"), noalias=True, readonly=True)
    dst = b.param("dst", PtrType(F32, "global"), noalias=True)
    tid = b.thread_id_x()
    value = b.global_load(src, tid, F32, align=4)
    b.global_store(dst, tid, b.fmul(value, b.const_f32(2.0)), align=4)
    b.ret()
    return b.kernel


@pytest.mark.parametrize("flavor", LLVM_FLAVORS)
def test_llvm_is_the_python_lowerers_text(flavor):
    native = native_engine_or_skip()
    kernel = _copy_kernel()
    assert native.lower_serialized_ir(
        serialize(kernel), _ARCH, flavor
    ) == _lower_kernel_to_llvm_python(kernel, arch=_ARCH, llvm_flavor=flavor)


def test_hip_is_the_python_lowerers_text():
    native = native_engine_or_skip()
    kernel = _copy_kernel()
    assert native.lower_serialized_ir_to_hip(
        serialize(kernel), _ARCH
    ) == lower_kernel_to_hip(kernel, arch=_ARCH)


# (input, arch, status, reason) of refusals both lowerings make; the LLVM one
# also refuses a target it has no backend for (the HIP one reads an unknown
# target as its gfx950 default).
_SHARED_REFUSALS = [
    pytest.param("", _ARCH, 1, "empty IR input", id="empty"),
    pytest.param(
        "not serialized IR", _ARCH, 1, "bad header format name", id="not-a-kernel"
    ),
]
_UNKNOWN_ARCH = pytest.param(
    None, "gfx000", 3, "unknown arch backend gfx000", id="unknown-arch"
)


def _assert_refused(lower, c_entry: str, ir_text, arch, status, reason) -> None:
    if ir_text is None:
        ir_text = serialize(_copy_kernel())
    with pytest.raises(RuntimeError) as refused:
        lower(ir_text, arch)
    assert type(refused.value) is RuntimeError
    assert str(refused.value) == (
        f"{c_entry}: lower failed for arch {arch!r} (status {status}): {reason}"
    )


@pytest.mark.parametrize(
    "ir_text, arch, status, reason", [*_SHARED_REFUSALS, _UNKNOWN_ARCH]
)
def test_an_llvm_refusal_keeps_the_engines_status_and_reason(
    ir_text, arch, status, reason
):
    native = native_engine_or_skip()
    _assert_refused(
        lambda text, target: native.lower_serialized_ir(text, target, "llvm20"),
        "rocke_lower_serialized_ir_to_llvm",
        ir_text,
        arch,
        status,
        reason,
    )


@pytest.mark.parametrize("ir_text, arch, status, reason", _SHARED_REFUSALS)
def test_a_hip_refusal_keeps_the_engines_status_and_reason(
    ir_text, arch, status, reason
):
    native = native_engine_or_skip()
    _assert_refused(
        native.lower_serialized_ir_to_hip,
        "rocke_lower_serialized_ir_to_hip_text",
        ir_text,
        arch,
        status,
        reason,
    )


def _refusing_flavor(key: str, arch: str) -> str:
    """The first committed flavor whose arch-domain column refuses ``key`` on
    ``arch`` without a waiver."""
    for flavor in LLVM_FLAVORS:
        table = arch_domain.load(flavor)
        cell = table.lookup(key, arch) if table is not None else None
        if (
            cell is not None
            and arch_domain.is_negative(cell.status)
            and not arch_domain.is_waived(key, arch, flavor)
        ):
            return flavor
    raise AssertionError(f"no committed column refuses {key} on {arch}")


def test_an_arch_domain_refusal_is_its_own_type_with_the_lanes_message(monkeypatch):
    """``mov.dpp8.i32`` is a gfx10+ intrinsic the committed columns measure
    arch_absent on gfx9, so gfx950 is a real negative to refuse."""
    native = native_engine_or_skip()
    flavor = _refusing_flavor("mov.dpp8.i32", "gfx950")
    b = IRBuilder("native_engine_dpp8")
    b.kernel.attrs["max_workgroup_size"] = 64
    b.mov_dpp8(b.const_i32(1), 0x765432)
    monkeypatch.setenv(arch_domain.MODE_ENV, "error")
    with pytest.raises(native.ArchDomainRefusal) as refused:
        native.lower_serialized_ir(serialize(b.kernel), "gfx950", flavor)
    with pytest.raises(ArchDomainError) as python_refusal:
        _lower_kernel_to_llvm_python(b.kernel, arch="gfx950", llvm_flavor=flavor)
    assert str(refused.value) == str(python_refusal.value)
    assert isinstance(refused.value, NotImplementedError)


@pytest.mark.parametrize(
    "lower",
    [
        lambda ir: native_engine.lower_serialized_ir(ir, "", "llvm20"),
        lambda ir: native_engine.lower_serialized_ir_to_hip(ir, ""),
    ],
    ids=["llvm", "hip"],
)
def test_no_target_is_refused_before_the_engine_is_needed(lower):
    with pytest.raises(ValueError) as refused:
        lower(serialize(_copy_kernel()))
    assert str(refused.value) == native_engine.NO_TARGET_MESSAGE
    assert native_engine.NO_TARGET_MESSAGE == (
        "lowering requires an explicit gfx target; pass arch=... (there is no default)"
    )


@pytest.mark.parametrize("entry", ["lower_serialized_ir", "lower_serialized_ir_hip"])
def test_the_binding_refuses_no_target_with_the_same_text(entry):
    """The rocke_engine binding spells the refusal itself, in C++."""
    binding = import_or_skip_until("rocke_engine", BINDING_LANE)
    with pytest.raises(RuntimeError) as refused:
        getattr(binding, entry)(serialize(_copy_kernel()), arch="")
    assert str(refused.value) == (
        f"rocke_engine.{entry}: {native_engine.NO_TARGET_MESSAGE}"
    )


@pytest.mark.skipif(sys.platform != "linux", reason="loads the C library by name")
def test_a_library_without_an_entry_point_predates_it():
    """A library at the right binary ABI that lacks an entry point is older
    than this package: the C runtime, which has none of the engine's."""
    c_runtime = ctypes.CDLL(None)
    with pytest.raises(native_engine.SharedEngineAbiMismatch) as refused:
        native_engine.bind_function(
            c_runtime, "libc", "rocke_lower_serialized_ir_to_llvm", [], ctypes.c_int
        )
    assert str(refused.value) == (
        "librocke at 'libc' predates the entry point "
        "rocke_lower_serialized_ir_to_llvm; rebuild the library from this "
        "package's source"
    )


def test_an_unknown_flavor_names_the_engines_flavors():
    native = native_engine_or_skip()
    expected = ", ".join(repr(flavor) for flavor in LLVM_FLAVORS)
    with pytest.raises(ValueError) as refused:
        native.lower_serialized_ir(serialize(_copy_kernel()), _ARCH, "llvm99")
    assert str(refused.value) == (
        f"lower_serialized_ir failed - flavor='llvm99', expected one of ({expected})"
    )


def _run_isolated_copy(tmp_path: Path, environment: dict[str, str], code: str) -> str:
    """Run ``code`` in a child Python whose only ``rocke`` package is a copy of
    this module and the standard-library modules it imports, in a tree of its
    own under ``tmp_path``; return its stdout."""
    package = tmp_path / "tree" / "rocke"
    (package / "core").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "core" / "__init__.py").write_text("")
    shutil.copy(native_engine.__file__, package / "core" / "native_engine.py")
    shutil.copy(engine_provenance.__file__, package / "core" / "engine_provenance.py")
    shutil.copy(assets.__file__, package / "assets.py")
    run = subprocess.run(
        [sys.executable, "-S", "-c", code],
        cwd=package.parent,
        env={**environment, "PYTHONPATH": str(package.parent)},
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert run.returncode == 0, run.stderr
    return run.stdout


_REPORT_MISSING_LIBRARY = """
from rocke.core import native_engine
print(native_engine.native_lowering_available())
try:
    native_engine.lower_serialized_ir("ir", "gfx942", "llvm20")
except native_engine.SharedEngineNotFound as missing:
    print(missing)
"""


def test_no_library_is_reported_with_where_it_was_looked_for(tmp_path):
    environment = {"PATH": ""}
    out = _run_isolated_copy(tmp_path, environment, _REPORT_MISSING_LIBRARY)
    tree = (tmp_path / "tree").resolve()
    reason = (
        f"no shared C++ engine: {native_engine.SHARED_ENGINE_ENV} is unset and no "
        f"{native_engine.shared_library_name()} is installed beside the rocke "
        f"package ({tree}). Build the rocke_shared target "
        f"(ROCKE_BUILD_SHARED_ENGINE=ON) and set {native_engine.SHARED_ENGINE_ENV} "
        "to the library it makes"
    )
    assert out.splitlines() == ["False", reason]


def test_a_named_library_that_does_not_exist_is_an_error(tmp_path):
    missing = tmp_path / "nowhere" / native_engine.shared_library_name()
    environment = {native_engine.SHARED_ENGINE_ENV: str(missing)}
    code = """
from rocke.core import native_engine
try:
    native_engine.native_lowering_available()
except FileNotFoundError as error:
    print(error)
"""
    out = _run_isolated_copy(tmp_path, environment, code)
    assert out.strip() == (
        f"librocke not found at {str(missing)!r} (from "
        f"{native_engine.SHARED_ENGINE_ENV}). Refusing to fall back to another library"
    )


def test_none_says_the_process_has_no_shared_engine(tmp_path):
    """ROCKE_ONLINE_LIB=none hides even an installed library: the copy's tree
    has one beside the package, as the provider's test tree does."""
    (tmp_path / "tree").mkdir()
    (tmp_path / "tree" / native_engine.shared_library_name()).touch()
    environment = {native_engine.SHARED_ENGINE_ENV: native_engine.NO_SHARED_ENGINE}
    out = _run_isolated_copy(tmp_path, environment, _REPORT_MISSING_LIBRARY)
    reason = (
        f"no shared C++ engine: {native_engine.SHARED_ENGINE_ENV}="
        f"{native_engine.NO_SHARED_ENGINE} says this process has none"
    )
    assert out.splitlines() == ["False", reason]
