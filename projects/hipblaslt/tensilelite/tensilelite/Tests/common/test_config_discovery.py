# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""findConfigs walks every YAML under Tests/, so it must tell a TensileLite
config from the other YAML that lives there."""

import pytest

from config_helpers import configMarks, findConfigs

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


def test_find_configs_keeps_configs_next_to_data(tmp_path):
    (tmp_path / "config.yaml").write_text(_CONFIG)
    (tmp_path / "data.yaml").write_text("kernels: {a: [1, 2]}\n")

    params = findConfigs(str(tmp_path), ["gfx950"], deviceTags=[])

    assert [p.id for p in params] == ["config.yaml"]
