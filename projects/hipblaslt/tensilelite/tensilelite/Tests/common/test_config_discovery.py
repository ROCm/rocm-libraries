# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""findConfigs walks every YAML under Tests/, so it must tell a TensileLite
config from the other YAML that lives there."""

import os

import pytest

from config_helpers import configMarks, findConfigs

_COMMON_DIR = os.path.dirname(os.path.abspath(__file__))
_TESTS_ROOT = os.path.dirname(_COMMON_DIR)

_CONFIG = """\
GlobalParameters: {}
BenchmarkProblems:
  - - OperationType: GEMM
      DataType: b
    - BenchmarkFinalParameters:
        - ProblemSizes:
          - Exact: [256, 256, 1, 256]
"""

pytestmark = pytest.mark.filterwarnings("ignore::pytest.PytestUnknownMarkWarning")


@pytest.mark.parametrize(
    "body",
    [
        "kernels: {a: [1, 2]}\n",
        "GlobalParameters: {}\n",
        "- [1, 2]\n",
    ],
    ids=["data-mapping", "no-benchmark-problems", "sequence"],
)
def test_yaml_that_is_not_a_config_is_skipped(tmp_path, body):
    path = tmp_path / "data.yaml"
    path.write_text(body)
    assert configMarks(str(path), str(tmp_path), ["gfx950"]) is None


@pytest.mark.parametrize(
    "rel",
    [
        "unit/data/rocroller_predicate_manifest.yaml",
    ],
)
def test_unit_test_data_is_not_a_config(rel):
    path = os.path.join(_TESTS_ROOT, rel)
    assert os.path.isfile(path), path
    assert configMarks(path, _TESTS_ROOT, ["gfx950"]) is None


def test_find_configs_keeps_configs_next_to_data(tmp_path):
    (tmp_path / "config.yaml").write_text(_CONFIG)
    (tmp_path / "data.yaml").write_text("kernels: {a: [1, 2]}\n")

    params = findConfigs(str(tmp_path), ["gfx950"], deviceTags=[])

    assert [p.id for p in params] == ["config.yaml"]
