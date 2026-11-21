# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""TF32 diagnostic consumers retain native identity and alias-visible I/O order."""

import ast
import inspect
from itertools import product
import os
from pathlib import Path
import subprocess

import pytest

from rocke.core.ir_serialize import serialize
from rocke.core.verify import verify
from rocke.instances.gfx942.tf32_mma_probe import (
    PREPARATIONS,
    Tf32MmaProbeSpec,
    build_tf32_mma_probe,
)


@pytest.mark.parametrize("m,mode", list(product((16, 32), PREPARATIONS)))
def test_tf32_descriptor_consumer_has_native_authoring_identity(m, mode):
    kernel = build_tf32_mma_probe(Tf32MmaProbeSpec(m, mode))
    assert not verify(kernel)
    executable = os.environ.get("ROCKE_FRAGMENT_IO_TEST")
    if not executable:
        pytest.skip("set ROCKE_FRAGMENT_IO_TEST to the built native fragment emitter")
    assert Path(executable).is_file()
    result = subprocess.run(
        [executable, "--tf32", str(m), mode],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.stdout == serialize(kernel)


@pytest.mark.parametrize("mode", PREPARATIONS)
def test_probe_preserves_read_prepare_store_instruction_order(mode):
    kernel = build_tf32_mma_probe(Tf32MmaProbeSpec(16, mode))
    events = []
    for op in kernel.body.ops:
        if op.name == "tile.mma":
            events.append("mma")
        elif op.name.startswith("memref.global_load") and op.operands[0].name in (
            "%A",
            "%B",
        ):
            events.append("read " + op.operands[0].name[1:])
        elif op.name == "memref.global_store_typed" and op.operands[0].name in (
            "%PA",
            "%PB",
        ):
            events.append("write " + op.operands[0].name[1:])
    if mode == "fp32":
        assert events == ["read A", "read B", "write PA", "write PB", "mma"] * 2
    else:
        assert events == [
            "read A",
            "write PA",
            "write PA",
            "read B",
            "write PB",
            "write PB",
            "mma",
        ]


def test_probe_has_no_auxiliary_or_direct_memory_bypass():
    tree = ast.parse(inspect.getsource(build_tf32_mma_probe))
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
