# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared helpers for the bound-check, access-entry and run tests of tensor_transforms.

- ``runtime_length`` -- a runtime length as the tiling layer's ``runtime_value`` returns it: a
  kernel parameter behind a range marker stating its facts, read back as a ``RuntimeValue``.
- ``RuntimeTensor`` -- a tensor description with runtime lengths (the entries read only
  ``lengths``, ``strides`` and ``dtype``).
- ``gfx90a_isa`` -- the disassembled gfx90a ISA of a kernel (compiled offline; no GPU needed).
- ``run_lanes`` -- build a test-local kernel, compile it for the GPU present, run it, and return
  what every lane stored.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pytest

from rocke.core.ir import I1, I32, I64, IRBuilder, PtrType, Value
from rocke.helpers.tiling.tensor_transforms import RuntimeValue
from rocke.helpers.tiling.tensor_transforms.walk import runtime_value_of
from rocke.tests.gpu import current_device_arch

WAVE = 64


def builder(workgroup_size: int = WAVE) -> IRBuilder:
    b = IRBuilder("bound_check_test")
    b.kernel.attrs["max_workgroup_size"] = workgroup_size
    return b


def raises(exc: type[Exception], message: str):
    return pytest.raises(exc, match=f"^{re.escape(message)}$")


def runtime_length(
    b: IRBuilder,
    name: str,
    *,
    at_least: int = 0,
    at_most: int = (1 << 31) - 1,
    multiple_of: int = 1,
) -> RuntimeValue:
    """A kernel parameter ``name`` stated ``at_least <= value <= at_most`` (a multiple of
    ``multiple_of``), the way the tiling layer's ``runtime_value`` states a length it checked.
    """
    marked = b._assume_range(
        b.param(name, I32), lo=at_least, hi=at_most + 1, multiple_of=multiple_of
    )
    return runtime_value_of(marked)


@dataclass(frozen=True)
class RuntimeTensor:
    """A tensor description whose lengths may be runtime lengths."""

    lengths: tuple[int | RuntimeValue, ...]
    strides: tuple[int, ...]
    dtype: Any


def ops(b: IRBuilder) -> list:
    def walk(body):
        for op in body:
            yield op
            for region in op.regions:
                yield from walk(region.ops)

    return list(walk(b.kernel.body.ops))


def count(b: IRBuilder, name: str, **attrs) -> int:
    return sum(
        1
        for op in ops(b)
        if op.name == name and all(op.attrs.get(k) == v for k, v in attrs.items())
    )


def store_values(b: IRBuilder, values: Sequence[Value], *, per_block: bool) -> int:
    """Store every value (as i64) to a kernel parameter ``out``: one slot per value, per lane
    (or per workgroup, for wave-uniform values). Returns the number of slots."""
    out = b.param("out", PtrType(I64, "global"))
    base = b.block_id_x()
    if not per_block:
        base = b.add(b.mul(base, b.const_i32(WAVE)), b.thread_id_x())
    base = b.mul(base, b.const_i32(len(values)))
    for slot, value in enumerate(values):
        if value.type == I1:
            wide = b.zext(value, I64)
        elif value.type == I32:
            wide = b.sext(value, I64)
        else:
            wide = value
        b.global_store(out, b.add(base, b.const_i32(slot)), wide)
    return len(values)


def gfx90a_isa(b: IRBuilder) -> str:
    """The kernel compiled for gfx90a and disassembled (skips when the tools are missing)."""
    compile_kernel = pytest.importorskip("rocke.helpers.compile").compile_kernel
    objdump = shutil.which("llvm-objdump") or shutil.which(
        "llvm-objdump",
        path=os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "llvm", "bin"),
    )
    if objdump is None:
        pytest.skip("llvm-objdump is not available")
    artifact = compile_kernel(b.kernel, arch="gfx90a")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "kernel.hsaco")
        with open(path, "wb") as f:
            f.write(artifact.hsaco)
        text = subprocess.run(
            [objdump, "-d", path], capture_output=True, text=True, check=True
        ).stdout
    return text


def isa_count(isa: str, pattern: str) -> int:
    """How many instructions of the disassembly match ``pattern`` (a mnemonic regex)."""
    return len(re.findall(rf"^\s*({pattern})\b", isa, flags=re.MULTILINE))


def run_lanes(
    build: Callable[[IRBuilder], Sequence[Value]],
    *,
    scalars: Mapping[str, int] | None = None,
    buffers: Mapping[str, Any] | None = None,
    n_blocks: int = 1,
    per_block: bool = False,
):
    """Run a kernel whose lanes store the values ``build`` returns, compiled for the GPU
    present; return them as an int64 array ``[block, lane, slot]`` (``[block, slot]`` with
    ``per_block``). A block is ``WAVE`` (64) threads whatever the GPU's wave size.

    ``scalars`` are i32 kernel parameters (declared by the test with ``runtime_length`` or
    ``b.param``, in this order, before ``build`` returns); ``buffers`` are read-only i32 arrays,
    declared here and reached in ``build`` through ``b.get_param``.
    """
    import numpy as np

    from rocke.helpers.compile import compile_kernel
    from rocke.helpers.spec import SignatureBuilder
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.host_buffers import as_u8_buffer
    from rocke.runtime.launcher import (
        DeviceMem,
        KernelLauncher,
        LaunchConfig,
        synchronize_and_release,
    )

    scalars = dict(scalars or {})
    buffers = {
        name: np.ascontiguousarray(a, dtype=np.int32)
        for name, a in (buffers or {}).items()
    }
    b = builder()
    for name in buffers:
        b.param(name, PtrType(I32, "global"), readonly=True, noalias=True)
    values = build(b)
    slots = store_values(b, values, per_block=per_block)
    signature = SignatureBuilder()
    for param in b.kernel.params:
        if param.name in buffers:
            signature = signature.ptr(param.name, "i32")
        elif param.name != "out":
            signature = signature.scalar(param.name, "i32")
    signature = signature.ptr("out", "i64")
    artifact = compile_kernel(b.kernel, arch=current_device_arch())
    launcher = KernelLauncher(
        hsaco=artifact.hsaco,
        kernel_name=artifact.kernel_name,
        signature=signature.build(),
    )
    rows = n_blocks if per_block else n_blocks * WAVE
    host = np.zeros(rows * slots, dtype=np.int64)
    rt = Runtime()
    out = DeviceMem(host.nbytes)
    args: dict[str, Any] = {"out": out}
    for name, array in buffers.items():
        mem = DeviceMem(array.nbytes)
        rt.memcpy_h2d(mem.ptr(), as_u8_buffer(array), array.nbytes)
        args[name] = mem
    for param in b.kernel.params:
        if param.name not in buffers and param.name != "out":
            args[param.name] = scalars[param.name]
    launcher(args, config=LaunchConfig(grid=(n_blocks, 1, 1), block=(WAVE, 1, 1)))
    synchronize_and_release()
    rt.memcpy_d2h(as_u8_buffer(host), out.ptr(), host.nbytes)
    if per_block:
        return host.reshape(n_blocks, slots)
    return host.reshape(n_blocks, WAVE, slots)
