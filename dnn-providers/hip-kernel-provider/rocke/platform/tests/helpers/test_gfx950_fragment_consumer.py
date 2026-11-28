# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Full gfx950 consumer migration: native authoring parity and public I/O path."""

import ast
import inspect
import os
from pathlib import Path
import subprocess

import pytest

from rocke.core.ir_serialize import serialize
from rocke.core.verify import verify
from rocke.instances.gfx950.block_scaled_gemm import (
    BlockScaledGemmSpec,
    build_block_scaled_gemm,
)

_FORMATS = ("fp8e4m3", "bf8e5m2", "fp6e2m3", "fp6e3m2", "fp4e2m1")


@pytest.mark.parametrize("index", range(50))
def test_complete_descriptor_consumer_has_native_authoring_identity(index):
    a, b = _FORMATS[(index % 25) // 5], _FORMATS[index % 5]
    c = "bf16" if index < 25 else "fp16"
    shape = (32, 64, 128) if index < 25 else (64, 96, 256)
    spec = BlockScaledGemmSpec("gemm_io", *shape, a, b, c)
    kernel = build_block_scaled_gemm(spec)
    assert not verify(kernel)
    executable = os.environ.get("ROCKE_FRAGMENT_IO_TEST")
    if not executable:
        pytest.skip("set ROCKE_FRAGMENT_IO_TEST to the built native fragment emitter")
    assert Path(executable).is_file()
    result = subprocess.run(
        [executable, "--gemm", a, b, c, *map(str, shape)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.stdout == serialize(kernel)


def test_kernel_body_does_not_bypass_fragment_io():
    tree = ast.parse(inspect.getsource(build_block_scaled_gemm))
    calls = [node.func for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(
        isinstance(call, ast.Attribute)
        and call.attr in ("global_load", "global_load_vN", "global_store")
        for call in calls
    )
    assert not any(
        isinstance(call, ast.Name) and call.id == "load_matrix_fragment"
        for call in calls
    )
