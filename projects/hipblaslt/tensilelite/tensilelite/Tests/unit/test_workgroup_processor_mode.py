# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""WorkGroupProcessorMode: the kernel descriptor selects CU mode or workgroup-processor mode."""
import re

import pytest

from config_harness import assert_assembles, emit_kernels_from_config, solutions_from_config
from test_scalar_load_b_codegen import _ARCH, _config_with, _emit

pytestmark = pytest.mark.unit
_DIRECTIVE = re.compile(r"^\s*\.amdhsa_workgroup_processor_mode (\d+)", re.M)


def _modes(results):
    return [_DIRECTIVE.findall(source) for _, source, _ in results]


def test_default_leaves_the_mode_to_the_assembler():
    assert _modes(_emit("NN")) == [[]] * 4


def test_cu_mode_is_emitted_and_assembles():
    results = _emit("NT")
    assert _modes(results) == [["0"]] * 4
    for name, source, _ in results:
        assert_assembles(source, name)


def test_workgroup_processor_mode_is_emitted(tmp_path):
    config = _config_with(tmp_path, WorkGroupProcessorMode=[1])
    results = emit_kernels_from_config(config, limit=1, arch=_ARCH, canonical=False, expected_fork_count=1)
    assert _modes(results) == [["1"]]


def test_reset_without_workgroup_processors(tmp_path):
    # gfx9 has no workgroup processors: the request is dropped instead of rejecting the solution
    config = _config_with(tmp_path, WorkGroupProcessorMode=[0], ScalarLoadB=[0], UseDualFMAC=[False],
                          WorkGroup=[[16, 16, 1]], ThreadTile=[[4, 4]], AssertFree1ElementMultiple=[1])
    solutions = [s for s in solutions_from_config(config, arch="gfx942") if s["Valid"]]
    assert solutions
    assert {s["WorkGroupProcessorMode"] for s in solutions} == {-1}
