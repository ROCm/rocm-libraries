# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from pathlib import Path

import pandas as pd
import yaml

from geko.bench.workload_report import (
    compare_runs,
    datatype_tokens,
    drop_tuned_sizes,
    stage_merge_tuned_library,
)
from geko.library.operations import load_library

_STOCK = "library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx1250"
TN = "gfx1250_Cijk_Alik_Bljk_BBS_BH_BiasSB_HAS_SAV_UserArgs.yaml"
NN = "gfx1250_Cijk_Ailk_Bljk_BBS_BH_BiasSB_HAS_SAV_UserArgs.yaml"


def _logic(
    library_type: str,
    sizes: list | None,
    cu: int | None = None,
    devices: tuple[str, ...] = ("Device 75c7",),
    trans: tuple[bool, bool] = (True, False),
) -> dict:
    return {
        "MinimumRequiredVersion": "4.33.0",
        "ScheduleName": "gfx1250",
        "ArchitectureName": "gfx1250",
        "CUCount": cu,
        "DeviceNames": list(devices),
        "ProblemType": {"TransposeA": trans[0], "TransposeB": trans[1]},
        "DefaultSolution": {},
        "Solutions": [{"SolutionIndex": 0}],
        "IndexOrder": [0, 1, 2, 3],
        "ExactLogic": sizes,
        "RangeLogic": None,
        "TileSelectionIndices": None,
        "LibraryType": library_type,
    }


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def _no_merge(*args, **kwargs):
    raise AssertionError("TensileMergeLibrary must not run for prediction-only branches")


def test_datatype_tokens_follow_logic_filenames() -> None:
    names = [
        "gfx1250_Cijk_Alik_Bljk_BBS_BH_UserArgs.yaml",
        "gfx1250_Cijk_Ailk_Bjlk_BBS_BH_BiasSB_HAS_SAV_UserArgs.yaml",
        "gfx950_Cijk_Alik_Bljk_HHS_BH_UserArgs.yaml",
    ]
    assert datatype_tokens(names) == ["BBS", "HHS"]


def test_prediction_branch_gets_sibling_exact_file_with_branch_hardware(
    tmp_path: Path, monkeypatch
) -> None:
    hip = tmp_path / "hip"
    stock = hip / _STOCK
    _write(stock / "gfx1250_96cu/Origami" / TN, _logic("Prediction", None, cu=96))
    _write(stock / "gfx1250/Origami" / TN, _logic("Prediction", None, devices=("Device 73f0",)))
    _write(stock / "gfx1250/Equality/gfx1250_Cijk_Alik_Bljk_F8BS_BH_UserArgs.yaml", _logic("Equality", []))
    tuned = tmp_path / "final_libs"
    _write(tuned / TN, _logic("Equality", [[[256, 1024, 1, 128], [0, 1.0]]]))
    monkeypatch.setattr("geko.bench.workload_report.merge_logic", _no_merge)

    logic = stage_merge_tuned_library(hip, tuned, tmp_path / "stage" / "logic", arch="gfx1250")

    sibling = load_library(logic / "gfx1250_96cu/Equality" / TN)
    assert sibling.data["CUCount"] == 96
    assert sibling.data["DeviceNames"] == ["Device 75c7"]
    assert sibling.sizes == [[[256, 1024, 1, 128], [0, 1.0]]]
    assert load_library(logic / "gfx1250_96cu/Origami" / TN).type == "Prediction"
    generic = load_library(logic / "gfx1250/Equality" / TN)
    assert generic.data["CUCount"] is None
    assert (logic / "gfx1250/Equality/gfx1250_Cijk_Alik_Bljk_F8BS_BH_UserArgs.yaml").is_file()


def test_branch_for_other_device_is_left_alone(tmp_path: Path, monkeypatch) -> None:
    hip = tmp_path / "hip"
    _write(hip / _STOCK / "gfx1250/Origami" / TN, _logic("Prediction", None, devices=("Device 73f0",)))
    tuned = tmp_path / "final_libs"
    _write(tuned / TN, _logic("Equality", [[[64, 64, 1, 64], [0, 1.0]]]))
    monkeypatch.setattr("geko.bench.workload_report.merge_logic", _no_merge)

    logic = stage_merge_tuned_library(hip, tuned, tmp_path / "logic", arch="gfx1250")

    assert load_library(logic / "gfx1250/Origami" / TN).data["DeviceNames"] == ["Device 73f0"]
    assert load_library(logic / "gfx1250/Equality" / TN).data["DeviceNames"] == ["Device 75c7"]


def test_stock_equality_file_is_merged_not_replaced(tmp_path: Path, monkeypatch) -> None:
    hip = tmp_path / "hip"
    _write(hip / _STOCK / "gfx1250/Equality" / NN, _logic("Equality", [[[8, 8, 1, 8], [0, 1.0]]], trans=(False, False)))
    tuned = tmp_path / "final_libs"
    _write(tuned / NN, _logic("Equality", [[[16, 16, 1, 16], [0, 1.0]]], trans=(False, False)))
    calls = []

    def fake_merge(hip_path, orig_dir, inc_dir, output_dir, eff=False, force=True):
        calls.append(Path(orig_dir).name)
        (Path(output_dir) / NN).write_text("merged")

    monkeypatch.setattr("geko.bench.workload_report.merge_logic", fake_merge)

    logic = stage_merge_tuned_library(hip, tuned, tmp_path / "logic", arch="gfx1250")

    assert calls == ["Equality"]
    assert (logic / "gfx1250/Equality" / NN).read_text() == "merged"


def test_datatype_filter_limits_stock_copy(tmp_path: Path, monkeypatch) -> None:
    hip = tmp_path / "hip"
    f8 = "gfx1250_Cijk_Alik_Bljk_F8BS_BH_UserArgs.yaml"
    _write(hip / _STOCK / "gfx1250/Equality" / f8, _logic("Equality", []))
    tuned = tmp_path / "final_libs"
    _write(tuned / TN, _logic("Equality", [[[1, 1, 1, 1], [0, 1.0]]]))
    monkeypatch.setattr("geko.bench.workload_report.merge_logic", _no_merge)

    full = stage_merge_tuned_library(hip, tuned, tmp_path / "full", arch="gfx1250")
    filtered = stage_merge_tuned_library(hip, tuned, tmp_path / "filtered", arch="gfx1250", datatype_filter=True)

    assert (full / "gfx1250/Equality" / f8).is_file()
    assert not (filtered / "gfx1250/Equality" / f8).exists()


def _bench(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["transA", "transB", "m", "n", "k", "batch_count", "mean_us", "us", "solution"]
    )


def test_compare_runs_classifies_outcomes() -> None:
    def timed(rows: list[tuple]) -> pd.DataFrame:
        df = _bench(rows)
        df["time_us"] = df["mean_us"]
        return df

    before = timed([
        ("T", "N", 1, 1, 1, 1, 10.0, 10.0, "a"),
        ("T", "N", 2, 2, 2, 1, 10.0, 10.0, "a"),
        ("N", "N", 3, 3, 3, 1, 10.0, 10.0, "a"),
    ])
    after = timed([
        ("T", "N", 1, 1, 1, 1, 10.4, 10.4, "a"),
        ("T", "N", 2, 2, 2, 1, 8.0, 8.0, "tuned"),
        ("N", "N", 3, 3, 3, 1, 12.0, 12.0, "tuned"),
    ])
    table = compare_runs(before, after).set_index("m")
    assert table.loc[1, "outcome"] == "same_kernel"
    assert table.loc[2, "outcome"] == "tuned_faster"
    assert table.loc[3, "outcome"] == "tuned_slower"


def test_drop_tuned_sizes_removes_only_rejected_transpose_and_size(tmp_path: Path) -> None:
    tuned = tmp_path / "tuned"
    _write(tuned / TN, _logic("Equality", [
        [[256, 1024, 1, 128], [0, 1.0]],
        [[512, 1024, 1, 128], [0, 1.0]],
    ]))
    _write(tuned / NN, _logic("Equality", [[[256, 1024, 1, 128], [0, 1.0]]], trans=(False, False)))
    rejected = _bench([("T", "N", 256, 1024, 128, 1, 0.0, 0.0, "x")])

    removed = drop_tuned_sizes(tuned, tmp_path / "out", rejected)

    assert removed == 1
    assert load_library(tmp_path / "out" / TN).sizes == [[[512, 1024, 1, 128], [0, 1.0]]]
    assert load_library(tmp_path / "out" / NN).sizes == [[[256, 1024, 1, 128], [0, 1.0]]]
