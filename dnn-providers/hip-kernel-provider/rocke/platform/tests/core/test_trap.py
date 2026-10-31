# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The ``trap`` op: abort the kernel.

``trap()`` lowers to ``call void @llvm.trap()`` (HIP ``__builtin_trap()``),
which the AMDGPU backend emits as ``s_trap 2``. A taken trap ends the kernel
with a hardware exception; depending on the HIP runtime the launch reports an
error or the runtime aborts the process, and either way the process cannot use
the GPU afterwards. That is the runtime's behaviour, not rocKE's, so no test
takes the trap on a GPU: ``s_trap 2`` in the code object is the check that the
trap is emitted.

Covers the LLVM IR and HIP source of both engines (the same text at every LLVM
flavor), the refusal of a trap with operands or results by the verifier and
every lowering, the instruction in the compiled code object, and, when a GPU is
present, on the LLVM and the hipcc route, in a child process: a launch whose
trap branch is not taken runs to its store.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path

import pytest
from rocke.core.ir import I32, IRBuilder, PtrType, Region
from rocke.core.verify import verify
from rocke.tests.engines import (
    assert_both_engines_reject,
    assert_both_engines_reject_hip,
    lower_hip_with_both_engines,
    lower_with_both_engines,
)
from rocke.tests.subprocess_launcher import (
    rocke_python_command,
    rocke_subprocess_env,
)

FLAVORS = ("llvm20", "llvm22", "llvm23")


def _trap_kernel(name):
    """One lane: trap when the i32 parameter ``fail`` is non-zero, else store
    7 to ``Y[0]``."""
    b = IRBuilder(name)
    b.kernel.attrs["max_workgroup_size"] = 1
    y = b.param("Y", PtrType(I32, "global"))
    fail = b.param("fail", I32)
    with b.scf_if(b.cmp_ne(fail, b.const_i32(0))):
        b.trap()
    b.global_store(y, b.const_i32(0), b.const_i32(7))
    b.ret()
    return b.kernel


# --------------------------------------------------------------------------
# LLVM IR and HIP source
# --------------------------------------------------------------------------


def test_both_engines_emit_one_llvm_trap_call_under_the_branch():
    kernel = _trap_kernel("trap_llvm")
    for flavor in FLAVORS:
        text = lower_with_both_engines(kernel, "gfx90a", llvm_flavor=flavor)
        assert text.count("  call void @llvm.trap()\n") == 1, flavor
        assert text.count("declare void @llvm.trap()\n") == 1, flavor


def test_both_engines_emit_the_hip_trap_builtin():
    hip = lower_hip_with_both_engines(_trap_kernel("trap_hip"), "gfx90a")
    assert hip.count("__builtin_trap();") == 1


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------

MALFORMED_MESSAGE = "trap takes no operands, results or regions"


def _trap_with_a_result_kernel():
    """A tile.trap that defines an i32 result, written past the builder (as
    serialized IR can reach the verifier and the lowerers), and a store that
    uses the result afterwards: the lowerings emit only the trap call, so the
    store would read an undefined value."""
    b = IRBuilder("trap_result")
    y = b.param("Y", PtrType(I32, "global"))
    result = b._op("tile.trap", [], [I32], result_name_hint="t").result
    b.global_store(y, b.const_i32(0), result)
    b.ret()
    return b.kernel


def _trap_with_an_operand_kernel():
    """A tile.trap that takes an i32 operand, written past the builder."""
    b = IRBuilder("trap_operand")
    b.param("Y", PtrType(I32, "global"))
    b._op("tile.trap", [b.const_i32(1)], [])
    b.ret()
    return b.kernel


def _trap_with_a_region_kernel():
    """A tile.trap that carries an empty region, written past the builder."""
    b = IRBuilder("trap_region")
    b.param("Y", PtrType(I32, "global"))
    b._op("tile.trap", [], [], regions=[Region("body")])
    b.ret()
    return b.kernel


@pytest.mark.parametrize(
    "make_kernel",
    [
        _trap_with_a_result_kernel,
        _trap_with_an_operand_kernel,
        _trap_with_a_region_kernel,
    ],
    ids=["result_used_afterwards", "operand", "region"],
)
def test_verifier_and_every_lowering_refuse_a_trap_that_is_not_bare(make_kernel):
    kernel = make_kernel()
    assert MALFORMED_MESSAGE in [d.message for d in verify(kernel)]
    assert_both_engines_reject(kernel, "gfx90a", ValueError, MALFORMED_MESSAGE)
    assert_both_engines_reject_hip(kernel, "gfx90a", ValueError, MALFORMED_MESSAGE)


# --------------------------------------------------------------------------
# The instruction in the code object
# --------------------------------------------------------------------------


def _objdump():
    for candidate in (
        os.environ.get("LLVM_OBJDUMP"),
        os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "llvm/bin/llvm-objdump"),
        shutil.which("llvm-objdump"),
    ):
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def test_code_object_holds_s_trap_2_on_gfx90a():
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime import comgr

    objdump = _objdump()
    if objdump is None:
        pytest.skip("llvm-objdump not found (LLVM_OBJDUMP / ROCM_PATH / PATH)")
    try:
        comgr._resolve_lib()
    except comgr.ComgrError as error:
        pytest.skip(f"COMGR not loadable: {str(error)[:200]}")
    hsaco = compile_kernel(_trap_kernel("trap_isa"), arch="gfx90a").hsaco
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trap.hsaco"
        path.write_bytes(hsaco)
        isa = subprocess.run(
            [objdump, "-d", "--mcpu=gfx90a", str(path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    traps = [ln.split("//")[0].split() for ln in isa.splitlines() if "s_trap" in ln]
    assert traps == [["s_trap", "2"]], traps


# --------------------------------------------------------------------------
# GPU: the trap branch not taken, in a child process
# --------------------------------------------------------------------------


def _gpu_arch():
    try:
        from rocke.runtime.hip_module import get_device_arch
    except ImportError:
        return None
    try:
        return get_device_arch()
    except Exception:  # noqa: BLE001 - no HIP runtime means no GPU test
        return None


# Runs in the child: compiles the kernel on ``route``, launches it with
# fail = 0 and prints the stored value. A child, so that a regression which
# takes the trap cannot end or poison the test process.
_CHILD = textwrap.dedent(
    """
    import ctypes, struct, sys
    import numpy as np
    from rocke.runtime.hip_module import Runtime
    sys.path.insert(0, sys.argv[3])
    from test_trap import _trap_kernel

    route, arch = sys.argv[1], sys.argv[2]
    kernel = _trap_kernel(f"trap_gpu_{route}")
    if route == "llvm":
        from rocke.helpers.compile import compile_kernel
        blob = compile_kernel(kernel, arch=arch).hsaco
    else:
        from rocke.helpers.compile import compile_kernel_via_hipcc
        blob = compile_kernel_via_hipcc(kernel, arch=arch).hsaco
    runtime = Runtime()
    function = runtime.load_module(blob).get_function(kernel.name)
    y_dev = runtime.alloc(4)
    runtime.memset(y_dev, 0, 4)
    runtime.launch_blocking(function, (1, 1, 1), (1, 1, 1), struct.pack("Qi", y_dev, 0))
    y = np.zeros(1, dtype=np.int32)
    runtime.memcpy_d2h((ctypes.c_ubyte * 4).from_buffer(y), y_dev, 4)
    print(f"not taken: {int(y[0])}", flush=True)
    """
)


@pytest.mark.parametrize("route", ["llvm", "hipcc"])
def test_gpu_kernel_runs_to_its_store_when_the_trap_is_not_taken(route):
    arch = _gpu_arch()
    if not arch:
        pytest.skip("no GPU")
    if route == "hipcc" and shutil.which("hipcc") is None:
        pytest.skip("hipcc not found")
    child = subprocess.run(
        rocke_python_command("-c", _CHILD, route, arch, str(Path(__file__).parent)),
        check=False,  # the child's output is the result, read below
        capture_output=True,
        text=True,
        timeout=600,
        env=rocke_subprocess_env(),
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert "not taken: 7" in child.stdout.splitlines(), child.stdout + child.stderr
