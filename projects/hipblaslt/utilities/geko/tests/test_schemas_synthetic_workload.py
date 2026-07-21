# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tensile and hipBLASLt workload rows; compatibility with bench.log.parse."""

from pathlib import Path

import pytest
import yaml

from geko.bench.log import parse
from geko import schemas
from geko.constants import GEMM_LOG_FIELDS
from geko.schemas import GemmConfig, GemmType, RunState


@pytest.mark.parametrize(
    "dt,dd,cd",
    [
        ("B", "B", "S"),
        ("H", "H", "S"),
        ("X", "S", "S"),
    ],
)
def test_tensile_roundtrip_through_hipblaslt(dt, dd, cd):
    a_t, b_t, c_t, comp = GemmType._tensile_triple_to_hipblaslt(dt, dd, cd)
    gt = GemmType.from_hipblaslt("N", "T", a_t, b_t, c_t, comp)
    assert (gt.data_type, gt.dest_data_type, gt.compute_data_type) == (dt, dd, cd)


def test_tensile_triple_mixed_bh():
    a_t, b_t, c_t, comp = GemmType._tensile_triple_to_hipblaslt("BH", "B", "S")
    assert a_t == "bf16_r" and b_t == "f16_r" and c_t == "bf16_r" and comp == "f32_r"


def test_workload_log_rows_keys_and_sample_values():
    gt = GemmType.from_tensile("N", "T", "B", "B", "S")
    row = GemmConfig(gt, [[1024, 1024, 1, 1024]]).workload_log_rows()[0]
    assert set(row) == set(GEMM_LOG_FIELDS)
    assert row["M"] == 1024 and row["transB"] == "T"


def test_single_gemm_workload_parseable(tmp_path: Path):
    gt = GemmType.from_tensile("N", "T", "B", "B", "S")
    rows = GemmConfig(gt, [[128, 256, 2, 512]]).workload_log_rows()
    assert len(rows) == 1
    ypath = tmp_path / "w.yaml"
    with ypath.open("w") as f:
        yaml.safe_dump(rows, f, default_flow_style=None, sort_keys=False, width=5000)
    df = parse(ypath, as_df=True)
    assert len(df) == 1
    assert df["M"].iloc[0] == 128 and df["batch_count"].iloc[0] == 2


def test_workload_log_rows_tensile_only():
    gt = GemmType.from_tensile("N", "N", "B", "B", "S")
    rows = GemmConfig(gt, [[64, 64, 1, 64]]).workload_log_rows()
    assert len(rows) == 1
    assert set(rows[0]) == set(GEMM_LOG_FIELDS)


def test_workload_log_rows_with_logical():
    gt = GemmType.from_hipblaslt("T", "N", "bf16_r", "bf16_r", "bf16_r", "f32_r")
    rows = GemmConfig(gt, [[32, 32, 1, 32]]).workload_log_rows()
    assert rows[0]["a_type"] == "bf16_r" and rows[0]["transA"] == "T"


def test_workload_log_rows_multi_sizes():
    gt = GemmType.from_tensile("N", "N", "B", "B", "S")
    rows = GemmConfig(gt, [[64, 64, 1, 64], [128, 128, 1, 128]]).workload_log_rows()
    assert len(rows) == 2
    assert rows[0]["M"] == 64 and rows[1]["M"] == 128


def test_workload_log_rows_concat_multiple_configs():
    g1 = GemmConfig(GemmType.from_tensile("N", "N", "B", "B", "S"), [[8, 8, 1, 8]])
    g2 = GemmConfig(GemmType.from_tensile("N", "T", "H", "H", "S"), [[16, 16, 1, 16]])
    rows: list[dict] = []
    for gc in (g1, g2):
        rows.extend(gc.workload_log_rows())
    assert len(rows) == 2
    assert rows[0]["a_type"] == "bf16_r" and rows[1]["a_type"] == "f16_r"


def test_compute_type_for_workload_log_scalar_and_non_scalar() -> None:
    assert schemas._compute_type_for_workload_log("f32_r") == "c_f32_r"
    assert schemas._compute_type_for_workload_log("f16_r") == "f16_r"


def test_gemmtype_validation_errors() -> None:
    with pytest.raises(ValueError, match="Invalid transA"):
        GemmType("X", "N", data_type="B", dest_data_type="B", compute_data_type="S")

    with pytest.raises(ValueError, match="must be all set or all None"):
        GemmType("N", "N", a_type="f16_r", data_type="B", dest_data_type="B", compute_data_type="S")

    with pytest.raises(ValueError, match="Invalid a_type"):
        GemmType(
            "N",
            "N",
            a_type="bad",
            b_type="f16_r",
            c_type="f16_r",
            compute_type="f32_r",
            data_type="B",
            dest_data_type="B",
            compute_data_type="S",
        )

    with pytest.raises(ValueError, match="must be non-empty strings"):
        GemmType("N", "N", data_type="", dest_data_type="B", compute_data_type="S")


def test_tensile_mapper_error_paths() -> None:
    with pytest.raises(ValueError, match="Unknown Tensile DataType letter"):
        GemmType._tensile_triple_to_hipblaslt("Q", "B", "S")

    with pytest.raises(ValueError, match="must be 1 or 2 letters"):
        GemmType._tensile_triple_to_hipblaslt("ABC", "B", "S")

    with pytest.raises(ValueError, match="Unknown Tensile DestDataType letter"):
        GemmType._tensile_triple_to_hipblaslt("B", "Q", "S")

    with pytest.raises(ValueError, match="Unknown Tensile ComputeDataType letter"):
        GemmType._tensile_triple_to_hipblaslt("B", "B", "Q")


def test_hipblaslt_to_tensile_tf32_invalid_combo_raises() -> None:
    with pytest.raises(NotImplementedError, match="TF32 not implemented"):
        GemmType._hipblaslt_to_tensile("f16_r", "f16_r", "f16_r", "xf32_r")


def test_gemmconfig_validation_and_row_key_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    gt = GemmType.from_tensile("N", "N", "B", "B", "S")
    with pytest.raises(ValueError, match="non-empty list"):
        GemmConfig(gt, [])

    with pytest.raises(ValueError, match="four positive integers"):
        GemmConfig(gt, [[1, 2, 3]])

    gc = GemmConfig(gt, [[8, 8, 1, 8]])
    monkeypatch.setattr("geko.schemas.GEMM_LOG_FIELDS", tuple(list(GEMM_LOG_FIELDS) + ["extra"]))
    with pytest.raises(ValueError, match="must match GEMM_LOG_FIELDS exactly"):
        gc.workload_log_rows()


def test_runstate_dump_load_and_verify_failures(tmp_path: Path) -> None:
    input_file = tmp_path / "in.yaml"
    input_file.write_text("x\n")

    state = RunState.create(input_file)
    out = tmp_path / "work" / "run_state.json"
    state.dump(out)
    loaded = RunState.load(out)
    assert loaded.input_path == str(input_file)

    with pytest.raises(ValueError, match="Workdir belongs to input"):
        loaded.verify(tmp_path / "other.yaml")

    bad = RunState(
        input_sha256="not-a-real-hash",
        input_path=str(input_file),
        created_at=loaded.created_at,
        last_modified=loaded.last_modified,
        configured=False,
        optimized=False,
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        bad.verify(input_file)
