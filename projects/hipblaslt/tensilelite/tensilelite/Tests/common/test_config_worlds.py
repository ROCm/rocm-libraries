# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

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
%s
"""


def _write(tmp_path, name, exacts):
    path = tmp_path / name
    path.write_text(_CONFIG % "\n".join("          - Exact: %s" % e for e in exacts))
    return path


def _worldMarks(marks):
    return [m.args for m in marks if m.name == "world"]


pytestmark = pytest.mark.filterwarnings("ignore::pytest.PytestUnknownMarkWarning")


def test_fused_a2a_config_is_marked_with_its_worlds():
    path = os.path.join(_COMMON_DIR, "comm", "gfx950", "fused_a2a.yaml")
    assert _worldMarks(configMarks(path, _TESTS_ROOT, ["gfx950"])) == [(2, 4, 8)]


def test_config_without_worlds_has_no_world_mark(tmp_path):
    path = _write(tmp_path, "plain.yaml", ["[256, 256, 1, 256]"])
    assert _worldMarks(configMarks(str(path), str(tmp_path), ["gfx950"])) == []


def test_find_configs_splits_one_param_per_world(tmp_path):
    _write(tmp_path, "multi.yaml", ["{sizes: [256, 256, 1, 256], world: 4}",
                                    "{sizes: [256, 256, 1, 256], world: 2}",
                                    "{sizes: [512, 256, 1, 256], world: 2}"])
    _write(tmp_path, "plain.yaml", ["[256, 256, 1, 256]"])

    params = sorted(findConfigs(str(tmp_path), ["gfx950"]), key=lambda p: p.id)

    assert [p.id for p in params] == ["multi.yaml-world2", "multi.yaml-world4", "plain.yaml"]
    assert [_worldMarks(p.marks) for p in params] == [[(2,)], [(4,)], []]


def test_find_configs_keeps_one_param_when_not_splitting(tmp_path):
    _write(tmp_path, "multi.yaml", ["{sizes: [256, 256, 1, 256], world: 4}",
                                    "{sizes: [256, 256, 1, 256], world: 2}"])

    params = findConfigs(str(tmp_path), ["gfx950"], splitWorlds=False)

    assert [p.id for p in params] == ["multi.yaml"]
    assert _worldMarks(params[0].marks) == [(2, 4)]
