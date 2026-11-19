# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The ``gpu`` marker splits this tree between its two installed CTest entries.

``conftest.py`` marks every test that asks for a GPU fixture, and the installed entries select by
``-m gpu`` and ``-m "not gpu"``. Collecting two of the tree's modules each way proves the split:
a test asking for ``any_gpu`` (``tensor_transforms/test_runs.py``) or ``gfx90a_gpu``
(``test_leading_axes.py``) is selected only by ``-m gpu``, and every other test only by
``-m "not gpu"``. Collecting needs no GPU.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from rocke.tests.subprocess_launcher import rocke_python_command, rocke_subprocess_env

_TREE = Path(__file__).resolve().parent
_MODULES = (
    _TREE / "tensor_transforms" / "test_runs.py",
    _TREE / "test_leading_axes.py",
)
# The tests of those modules that ask for a GPU fixture.
_GPU_TESTS = {
    "test_runs.py::test_gpu_a_vector_straddling_a_runtime_length_has_an_exact_valid_span",
    "test_runs.py::test_gpu_the_conv_k_vector_span_is_exact_against_the_k_tail",
    "test_leading_axes.py::test_rank_3_load_then_store_is_bit_exact",
}


def _collected_tests(tmp_path: Path, marker_expression: str) -> set[str]:
    """``file::function`` of every test the marker expression selects from ``_MODULES``."""
    # The tree's own configuration: importlib mode for its relative imports, no other options.
    configuration = tmp_path / "pytest.ini"
    configuration.write_text("[pytest]\naddopts = --import-mode=importlib\n")
    result = subprocess.run(
        rocke_python_command(
            "-m",
            "pytest",
            "-c",
            configuration,
            "--rootdir",
            _TREE,
            "-p",
            "no:cacheprovider",
            "--collect-only",
            "-q",
            "-m",
            marker_expression,
            *_MODULES,
        ),
        capture_output=True,
        text=True,
        env=rocke_subprocess_env(),
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return {
        f"{Path(path).name}::{test.split('[')[0]}"
        for path, _, test in (
            line.partition("::") for line in result.stdout.splitlines() if "::" in line
        )
    }


def test_the_gpu_marker_selects_exactly_the_tests_that_ask_for_a_gpu_fixture(
    tmp_path: Path,
) -> None:
    gpu = _collected_tests(tmp_path, "gpu")
    not_gpu = _collected_tests(tmp_path, "not gpu")
    assert gpu == _GPU_TESTS
    assert not_gpu
    assert not gpu & not_gpu
