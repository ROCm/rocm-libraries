# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for opt-in sharding of long-running common-test YAML configs."""

import itertools
from collections import Counter, defaultdict
from pathlib import Path

import pytest
import yaml

from Tensile.Tests.common.artifact_helpers import artifact_name_for_config
from Tensile.Tests.common.config_helpers import (
    ConfigSpec,
    configForSpec,
    configSpecs,
    materializeConfig,
)


_COMMON_DIR = Path(__file__).resolve().parents[1] / "common"

_SHARDED_CONFIGS = (
    ("gemm/gfx12/gsu_gfx1250.yaml", (18,)),
    ("gemm/gfx12/subtile_bf16_gfx1250_bench.yaml", (12,)),
    ("gemm/gfx12/decouple_pgr_tdm_fuse_gfx1250.yaml", (36,)),
    ("streamk/gfx1250/core/sk_sgemm_quick.yaml", (4, 16)),
)


def _load(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _freeze(value):
    if isinstance(value, dict):
        return tuple((key, _freeze(item)) for key, item in value.items())
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _fork_product(group):
    parameters = {}
    for entry in group["ForkParameters"]:
        assert len(entry) == 1
        name, values = next(iter(entry.items()))
        parameters[name] = values

    names = tuple(parameters)
    return {
        tuple((name, _freeze(value)) for name, value in zip(names, values))
        for values in itertools.product(*(parameters[name] for name in names))
    }


@pytest.mark.parametrize(
    "relative_path, expected_group_counts", _SHARDED_CONFIGS
)
def test_real_config_shards_are_disjoint_and_exhaustive(
    relative_path, expected_group_counts
):
    """Each original fork permutation occurs in exactly one generated shard."""
    path = _COMMON_DIR / relative_path
    original = _load(path)
    specs = configSpecs(str(path))

    counts = Counter(spec.group_index for spec in specs)
    assert tuple(counts.values()) == expected_group_counts

    observed = defaultdict(set)
    for spec in specs:
        sharded = configForSpec(spec)
        assert len(sharded["BenchmarkProblems"]) == 1

        original_problem = original["BenchmarkProblems"][spec.problem_index]
        original_group = original_problem[spec.group_index + 1]
        shard_problem = sharded["BenchmarkProblems"][0]
        shard_group = shard_problem[1]

        assert shard_problem[0] == original_problem[0]
        assert shard_group["BenchmarkFinalParameters"] \
            == original_group["BenchmarkFinalParameters"]
        for key, value in original.items():
            if key != "BenchmarkProblems":
                assert sharded[key] == value
        for key, value in original_group.items():
            if key != "ForkParameters":
                assert shard_group[key] == value

        permutations = _fork_product(shard_group)
        group_key = (spec.problem_index, spec.group_index)
        assert observed[group_key].isdisjoint(permutations)
        observed[group_key].update(permutations)

    for problem_index, problem in enumerate(original["BenchmarkProblems"]):
        for group_index, group in enumerate(problem[1:]):
            group_key = (problem_index, group_index)
            assert observed[group_key] == _fork_product(group)


@pytest.mark.parametrize("relative_path, _", _SHARDED_CONFIGS)
def test_shard_ids_and_artifact_names_are_unique(relative_path, _):
    path = _COMMON_DIR / relative_path
    specs = configSpecs(str(path))

    labels = [spec.shard_label for spec in specs]
    artifacts = [
        artifact_name_for_config(spec.source_path, spec.shard_label)
        for spec in specs
    ]
    assert len(labels) == len(set(labels))
    assert len(artifacts) == len(set(artifacts))


def test_materialize_config_writes_only_the_selected_shard(tmp_path):
    path = _COMMON_DIR / "gemm/gfx12/subtile_bf16_gfx1250_bench.yaml"
    spec = configSpecs(str(path))[5]

    materialized = materializeConfig(spec, tmp_path)

    assert Path(materialized).parent == tmp_path
    assert Path(materialized).name == spec.shard_label + ".yaml"
    assert _load(materialized) == configForSpec(spec)


def test_config_without_shard_metadata_keeps_its_source_path(tmp_path):
    path = tmp_path / "plain.yaml"
    with open(path, "w") as f:
        yaml.safe_dump({"GlobalParameters": {}, "BenchmarkProblems": []}, f)

    specs = configSpecs(str(path))

    assert specs == [ConfigSpec(str(path))]
    assert materializeConfig(specs[0], tmp_path) == str(path)


@pytest.mark.parametrize(
    "shard_by, match",
    (
        ([], "non-empty list"),
        (["Axis", "Axis"], "unique"),
        (["Missing"], "names missing ForkParameters"),
    ),
)
def test_invalid_shard_metadata_fails_closed(tmp_path, shard_by, match):
    path = tmp_path / "invalid.yaml"
    doc = {
        "TestParameters": {"shard_by": shard_by},
        "BenchmarkProblems": [[
            {"OperationType": "GEMM", "DataType": "s"},
            {
                "ForkParameters": [{"Axis": [1, 2]}],
                "BenchmarkFinalParameters": [{"ProblemSizes": [[1, 1, 1]]}],
            },
        ]],
    }
    with open(path, "w") as f:
        yaml.safe_dump(doc, f)

    with pytest.raises(ValueError, match=match):
        configSpecs(str(path))
