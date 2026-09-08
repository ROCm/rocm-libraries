#!/usr/bin/env python3
"""Focused unit tests for find_exact.py logic extraction/writing."""

from __future__ import annotations

from pathlib import Path

import yaml

from find_exact import fetchDataFromLogic, yamlListInfo


def _mk_info(local_idx: int, splitk: int = 0, tflops: float = 123.4) -> yamlListInfo:
    info = yamlListInfo()
    info.problemSizes = [16, 16, 1, 16, 16, 16, 16, 16]
    info.localSolutionIndex = local_idx
    info.splitK = splitk
    info.tflops = tflops
    return info


def _mk_solution(index: int, gsu: int = 0) -> dict:
    return {
        "SolutionIndex": index,
        "SolutionNameMin": f"S_{index}_GSU{gsu}",
        "GlobalSplitU": gsu,
        "AssertFree0ElementMultiple": 1,
        "AssertFree1ElementMultiple": 1,
        "AssertSummationElementMultiple": 1,
    }


def test_fetch_data_from_logic_list_format(tmp_path: Path) -> None:
    src = tmp_path / "src_list.yaml"
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    list_logic = [
        "v0",            # 0
        "sched",         # 1
        "gfx950",        # 2
        ["Device 0049"], # 3
        {"OperationType": "GEMM"},   # 4
        [_mk_solution(0, 0)],          # 5 Solutions
        [2, 3, 0, 1],                  # 6
        [],                            # 7 ExactLogic
        None,                          # 8 RangeLogic
        None,                          # 9
        "DeviceEfficiency",           # 10
        "Equality",                   # 11 LibraryType
    ]
    src.write_text(yaml.safe_dump(list_logic, sort_keys=False))

    fetchDataFromLogic(str(src), str(out_dir), [_mk_info(0)], logicType="GridBased")

    out = yaml.safe_load((out_dir / src.name).read_text())
    assert isinstance(out, list)
    assert out[11] == "GridBased"
    assert out[8] is None
    assert len(out[5]) == 1
    assert out[5][0]["SolutionIndex"] == 0
    assert out[7][0][0] == [16, 16, 1, 16, 16, 16, 16, 16]


def test_fetch_data_from_logic_dict_format(tmp_path: Path) -> None:
    src = tmp_path / "src_dict.yaml"
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    dict_logic = {
        "ScheduleName": "sched",
        "ArchitectureName": "gfx950",
        "DeviceNames": ["Device 0049"],
        "ProblemType": {"OperationType": "GEMM"},
        "Solutions": [_mk_solution(0, 0)],
        "IndexOrder": [2, 3, 0, 1],
        "ExactLogic": [],
        "RangeLogic": [[1, 2, 3]],
        "PerfMetric": "DeviceEfficiency",
        "LibraryType": "Equality",
    }
    src.write_text(yaml.safe_dump(dict_logic, sort_keys=False))

    fetchDataFromLogic(str(src), str(out_dir), [_mk_info(0)], logicType="GridBased")

    out = yaml.safe_load((out_dir / src.name).read_text())
    assert isinstance(out, dict)
    assert out["LibraryType"] == "GridBased"
    assert out["RangeLogic"] is None
    assert len(out["Solutions"]) == 1
    assert out["Solutions"][0]["SolutionIndex"] == 0
    assert out["ExactLogic"][0][0] == [16, 16, 1, 16, 16, 16, 16, 16]
