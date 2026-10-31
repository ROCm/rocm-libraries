# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for scripts/sparsify_fork_grid.py that need no compiler or device."""

import itertools
from pathlib import Path
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
import sparsify_fork_grid as sparsify  # noqa: E402

pytestmark = pytest.mark.unit

CONFIG = """\
BenchmarkProblems:
  -
    - OperationType: GEMM
    - # BenchmarkProblemSizeGroup
      ForkParameters:
        - TDMInst: [3]
        # A comment that belongs to the next entry.
        - MatrixInstruction:
          - [16, 16, 128, 1, 1, 2, 4, 2, 2]
          - [16, 16, 128, 1, 1, 4, 4, 1, 4]
        - DepthU: [256, 512, 1024]
        - SourceSwap: [False, True]
        - ClusterDim: [[1, 1], [2, 4]]
        - WavefrontSize: [32]
      BenchmarkFinalParameters:
        - ProblemSizes:
          - Exact: [1024, 512, 1, 256]
"""


def full_product(forks):
    names = list(forks)
    return [dict(zip(names, values)) for values in itertools.product(*forks.values())]


def pairs_of(rows):
    axes = sorted({axis for row in rows for axis in row})
    result = set()
    for row in rows:
        result |= sparsify.canonical_pairs(row, axes)
    return result


def parsed_forks():
    doc = yaml.safe_load(CONFIG)
    return {
        k: v for item in doc["BenchmarkProblems"][0][1]["ForkParameters"] for k, v in item.items()
    }


def chosen_rows(forks):
    return [
        {
            "MatrixInstruction": forks["MatrixInstruction"][0],
            "DepthU": 256,
            "SourceSwap": False,
            "ClusterDim": [1, 1],
        },
        {
            "MatrixInstruction": forks["MatrixInstruction"][1],
            "DepthU": 1024,
            "SourceSwap": True,
            "ClusterDim": [2, 4],
        },
    ]


def test_cover_pairs_covers_every_pair_with_fewer_rows_than_the_product():
    rows = full_product({"A": [0, 1, 2], "B": [0, 1, 2], "C": [0, 1], "D": [0, 1, 2, 3]})
    chosen = [rows[i] for i in sparsify.cover_pairs(rows)]
    assert pairs_of(chosen) == pairs_of(rows)
    assert len(chosen) < len(rows) // 4


def test_cover_pairs_only_covers_pairs_that_occur_together():
    # A=1 and B=1 never occur together, so that pair does not force an extra row.
    rows = [{"A": 0, "B": 0}, {"A": 0, "B": 1}, {"A": 1, "B": 0}]
    assert sparsify.cover_pairs(rows) == [0, 1, 2]
    rows = [{"A": 0, "B": 0, "C": 0}, {"A": 0, "B": 0, "C": 1}]
    assert sparsify.cover_pairs(rows) == [0, 1]


def test_cover_pairs_is_deterministic():
    rows = full_product({"A": [0, 1, 2], "B": [0, 1, 2], "C": [0, 1, 2]})
    assert sparsify.cover_pairs(rows) == sparsify.cover_pairs(list(rows))


def test_render_replaces_multi_valued_entries_and_keeps_everything_else():
    forks = parsed_forks()
    axes = sparsify.multi_valued_axes(forks)
    assert axes == {"MatrixInstruction", "DepthU", "SourceSwap", "ClusterDim"}
    rows = chosen_rows(forks)
    new_group = yaml.safe_load(sparsify.render(CONFIG, 0, axes, rows))["BenchmarkProblems"][0][1]
    original = yaml.safe_load(CONFIG)["BenchmarkProblems"][0][1]
    assert new_group["BenchmarkFinalParameters"] == original["BenchmarkFinalParameters"]
    single = [item for item in new_group["ForkParameters"] if "Groups" not in item]
    assert single == [{"TDMInst": [3]}, {"WavefrontSize": [32]}]
    (groups,) = [item["Groups"] for item in new_group["ForkParameters"] if "Groups" in item]
    assert len(groups) == 1 and len(groups[0]) == len(rows)


def test_rendered_block_expands_to_exactly_the_chosen_rows():
    from tensilelite.BenchmarkStructs import _expandGroupedParameters, constructForkPermutations

    forks = parsed_forks()
    axes = sparsify.multi_valued_axes(forks)
    rows = chosen_rows(forks)
    new = yaml.safe_load(sparsify.render(CONFIG, 0, axes, rows))
    new_forks = {
        k: v for item in new["BenchmarkProblems"][0][1]["ForkParameters"] for k, v in item.items()
    }
    groups = _expandGroupedParameters(new_forks.pop("Groups"))
    expanded = list(constructForkPermutations(new_forks, groups))
    assert len(expanded) == len(rows)
    assert {sparsify.canonical({a: p[a] for a in axes}) for p in expanded} == {
        sparsify.canonical(r) for r in rows
    }


def test_keep_leaves_a_parameter_as_an_ordinary_entry():
    forks = parsed_forks()
    axes = sparsify.multi_valued_axes(forks)
    rows = [
        {"MatrixInstruction": forks["MatrixInstruction"][0], "DepthU": 256, "ClusterDim": [1, 1]}
    ]
    new = yaml.safe_load(sparsify.render(CONFIG, 0, axes, rows, keep=("SourceSwap",)))
    names = [next(iter(item)) for item in new["BenchmarkProblems"][0][1]["ForkParameters"]]
    assert "SourceSwap" in names and "DepthU" not in names and "Groups" in names
