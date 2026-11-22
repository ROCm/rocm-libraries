# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shared test-side oracles for the retained loop-exit tests."""
import copy
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from tensilelite.Common.GlobalParameters import defaultSolution

_DEFAULT = copy.deepcopy(defaultSolution)


@pytest.fixture(autouse=True)
def pristine_default_solution():
    saved = copy.deepcopy(defaultSolution)
    defaultSolution.clear()
    defaultSolution.update(copy.deepcopy(_DEFAULT))
    yield
    defaultSolution.clear()
    defaultSolution.update(saved)


def emit(tmp_path, name, logic, overrides):
    from codegen_harness import emit_kernels_from_logic, solutions_from_logic

    document = copy.deepcopy(yaml.safe_load(logic.read_text()))
    document["Solutions"][0].update(overrides)
    path = tmp_path / (name + ".yaml")
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    solutions = solutions_from_logic(path)
    assert len(solutions) == 1 and solutions[0]["Valid"], "recipe must derive successfully"
    solution = solutions[0]
    for field in ("AssertSummationElementMultiple", "PrefetchGlobalRead",
                  "ExpandPointerSwap", "UnrollLoopSwapGlobalReadOrder",
                  "UseCustomMainLoopSchedule"):
        if field in overrides:
            assert solution[field] == overrides[field], (field, solution[field], overrides[field])
    results = emit_kernels_from_logic(path, canonical=False)
    assert len(results) == 1
    _, source, error = results[0]
    assert error == 0, "valid loop recipe must emit successfully"
    return source


def instructions(source):
    """Ignore prose: return assembly opcode/operand text only."""
    return [line.split("//", 1)[0].strip() for line in source.splitlines()
            if re.match(r"^\s*[sv]_[a-z0-9_]+\s", line)]


def dec_blocks(source):
    code = instructions(source)
    prefix = "s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], "
    return [(line[len(prefix):], code[index + 1])
            for index, line in enumerate(code) if line.startswith(prefix)]


def has_single_iteration_escape(source):
    code = instructions(source)
    return any(line == "s_cmp_eq_u32 s[sgprLoopCounterL], 0x1"
               and code[index + 1].startswith("s_cbranch_scc1 label_toPGR1")
               for index, line in enumerate(code[:-1]))


def assemble(tmp_path, name, source):
    compiler = shutil.which("amdclang++") or "/opt/rocm/bin/amdclang++"
    assembly = tmp_path / (name + ".s")
    assembly.write_text(source)
    result = subprocess.run([compiler, "-x", "assembler", "-target", "amdgcn-amd-amdhsa",
                             "-mcpu=gfx950", "-c", str(assembly), "-o",
                             str(tmp_path / (name + ".o"))], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-4000:]
