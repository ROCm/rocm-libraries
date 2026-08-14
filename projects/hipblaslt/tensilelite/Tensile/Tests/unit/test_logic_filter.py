# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for Tensile.LogicFilter."""

import argparse
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from Tensile.LogicFilter import (
    buildPredicates,
    iterLogicFiles,
    loadLogicHeader,
    main,
    resolveAlias,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _makeArgs(**kwargs):
    """Build a minimal Namespace with all LogicFilter fields set to no-op defaults."""
    defaults = dict(
        subtile=None,
        input_type=None,
        input_type_a=None,
        input_type_b=None,
        dest_type=None,
        dquant_type=None,
        partial_rms=False,
        residual_add=False,
        residual_out=False,
        arch=None,
        schedule_name=None,
        field=None,
        solution_field=None,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _makeRecord(problemType=None, solutions=None, arch="gfx950", scheduleName="s"):
    """Build a minimal record dict suitable for predicate tests."""
    return {
        "path": Path("test.yaml"),
        "scheduleName": scheduleName,
        "arch": arch,
        "problemType": problemType or {},
        "solutions": solutions or [],
    }


# ---------------------------------------------------------------------------
# 1. Dict-form parse
# ---------------------------------------------------------------------------

def test_dictFormParse(tmp_path):
    """loadLogicHeader correctly parses the modern dict-form YAML."""
    content = {
        "ScheduleName": "mySchedule",
        "ArchitectureName": "gfx950",
        "CUCount": None,
        "DeviceNames": ["Device 75a0"],
        "ProblemType": {
            "DataTypeA": 7,
            "DataTypeB": 7,
            "DestDataType": 7,
            "DQuantType": "None",
        },
        "Solutions": [{"UseSubtileImpl": True}],
    }
    f = tmp_path / "test.yaml"
    f.write_text(yaml.dump(content))

    record = loadLogicHeader(f)

    assert record is not None
    assert record["scheduleName"] == "mySchedule"
    assert record["arch"] == "gfx950"
    assert record["problemType"]["DataTypeA"] == 7
    assert record["solutions"][0]["UseSubtileImpl"] is True
    assert record["path"] == f


# ---------------------------------------------------------------------------
# 2. List-form parse
# ---------------------------------------------------------------------------

def test_listFormParse(tmp_path):
    """loadLogicHeader correctly parses the legacy list-form YAML."""
    yaml_str = """\
- MinimumRequiredVersion: 5.0.0
- listSchedule
- gfx942
- [Device 0049]
- DataType: 7
  DataTypeA: 7
  DataTypeB: 7
  DestDataType: 7
  DQuantType: None
- []
"""
    f = tmp_path / "list.yaml"
    f.write_text(yaml_str)

    record = loadLogicHeader(f)

    assert record is not None
    assert record["scheduleName"] == "listSchedule"
    assert record["arch"] == "gfx942"
    assert record["problemType"]["DataTypeA"] == 7
    assert record["solutions"] == []


# ---------------------------------------------------------------------------
# 3. Alias resolution
# ---------------------------------------------------------------------------

def test_aliasResolution():
    """resolveAlias maps known aliases to numeric codes."""
    assert resolveAlias("bf16") == 7
    assert resolveAlias("f8_r") == 15
    assert resolveAlias("7") == 7
    assert resolveAlias("fp8_fnuz") == 11
    assert resolveAlias("f32_r") == 0
    assert resolveAlias("e8") == 22


def test_aliasUnknown():
    """resolveAlias raises ValueError for unknown aliases."""
    with pytest.raises(ValueError):
        resolveAlias("unknown_type")


# ---------------------------------------------------------------------------
# 4. --subtile predicate
# ---------------------------------------------------------------------------

def test_subtilePredMatch():
    """--subtile matches a record where any solution has UseSubtileImpl=True."""
    record = _makeRecord(solutions=[{"UseSubtileImpl": False}, {"UseSubtileImpl": True}])
    preds = buildPredicates(_makeArgs(subtile=True))
    assert all(p(record) for p in preds)


def test_subtilePredNoMatch():
    """--subtile does not match a record with no UseSubtileImpl=True solutions."""
    record = _makeRecord(solutions=[{"UseSubtileImpl": False}])
    preds = buildPredicates(_makeArgs(subtile=True))
    assert not all(p(record) for p in preds)


def test_noSubtilePred():
    """--no-subtile matches only when no solution has UseSubtileImpl."""
    record_none = _makeRecord(solutions=[{"UseSubtileImpl": False}])
    record_has = _makeRecord(solutions=[{"UseSubtileImpl": True}])

    preds = buildPredicates(_makeArgs(subtile=False))
    assert all(p(record_none) for p in preds)
    assert not all(p(record_has) for p in preds)


# ---------------------------------------------------------------------------
# 5. --input-type predicate
# ---------------------------------------------------------------------------

def test_inputTypePredMatch():
    """--input-type bf16 matches when DataTypeA is 7."""
    record = _makeRecord(problemType={"DataTypeA": 7, "DataTypeB": 15})
    preds = buildPredicates(_makeArgs(input_type="bf16"))
    assert all(p(record) for p in preds)


def test_inputTypePredNoMatch():
    """--input-type bf16 does not match when neither DataTypeA nor DataTypeB is 7."""
    record = _makeRecord(problemType={"DataTypeA": 15, "DataTypeB": 15})
    preds = buildPredicates(_makeArgs(input_type="bf16"))
    assert not all(p(record) for p in preds)


def test_inputTypeAPred():
    """--input-type-a only checks DataTypeA."""
    record_a7 = _makeRecord(problemType={"DataTypeA": 7, "DataTypeB": 15})
    record_b7 = _makeRecord(problemType={"DataTypeA": 15, "DataTypeB": 7})

    preds = buildPredicates(_makeArgs(input_type_a="bf16"))
    assert all(p(record_a7) for p in preds)
    assert not all(p(record_b7) for p in preds)


# ---------------------------------------------------------------------------
# 6. --dquant-type predicate
# ---------------------------------------------------------------------------

def test_dquantTypePredMatch():
    """--dquant-type MXFP8 matches DQuantType: MXFP8 (case-insensitive)."""
    record = _makeRecord(problemType={"DQuantType": "MXFP8"})
    preds = buildPredicates(_makeArgs(dquant_type="MXFP8"))
    assert all(p(record) for p in preds)


def test_dquantTypePredNoMatch():
    """--dquant-type MXFP8 does not match DQuantType: None."""
    record = _makeRecord(problemType={"DQuantType": "None"})
    preds = buildPredicates(_makeArgs(dquant_type="MXFP8"))
    assert not all(p(record) for p in preds)


def test_dquantTypeCaseInsensitive():
    """--dquant-type matching is case-insensitive."""
    record = _makeRecord(problemType={"DQuantType": "MXFP8"})
    preds = buildPredicates(_makeArgs(dquant_type="mxfp8"))
    assert all(p(record) for p in preds)


# ---------------------------------------------------------------------------
# 7. --partial-rms predicate
# ---------------------------------------------------------------------------

def test_partialRmsMatchResidualAdd():
    """--partial-rms matches when PartialRMSResidualAdd is true."""
    record = _makeRecord(problemType={
        "PartialRMSResidualAdd": True,
        "PartialRMSQuant": False,
        "PartialRMSStoreBf16D": False,
    })
    preds = buildPredicates(_makeArgs(partial_rms=True))
    assert all(p(record) for p in preds)


def test_partialRmsMatchQuant():
    """--partial-rms matches when PartialRMSQuant is true."""
    record = _makeRecord(problemType={
        "PartialRMSResidualAdd": False,
        "PartialRMSQuant": True,
        "PartialRMSStoreBf16D": False,
    })
    preds = buildPredicates(_makeArgs(partial_rms=True))
    assert all(p(record) for p in preds)


def test_partialRmsNoMatch():
    """--partial-rms does not match when all three PartialRMS fields are false."""
    record = _makeRecord(problemType={
        "PartialRMSResidualAdd": False,
        "PartialRMSQuant": False,
        "PartialRMSStoreBf16D": False,
    })
    preds = buildPredicates(_makeArgs(partial_rms=True))
    assert not all(p(record) for p in preds)


# ---------------------------------------------------------------------------
# 8. --field generic coercion
# ---------------------------------------------------------------------------

def test_fieldBoolCoercion():
    """--field correctly coerces bool values."""
    record = _makeRecord(problemType={"HighPrecisionAccumulate": True, "Batched": False})

    preds_true = buildPredicates(_makeArgs(field=["HighPrecisionAccumulate=true"]))
    assert all(p(record) for p in preds_true)

    preds_false = buildPredicates(_makeArgs(field=["Batched=false"]))
    assert all(p(record) for p in preds_false)

    preds_mismatch = buildPredicates(_makeArgs(field=["HighPrecisionAccumulate=false"]))
    assert not all(p(record) for p in preds_mismatch)


def test_fieldIntCoercion():
    """--field correctly coerces integer values."""
    record = _makeRecord(problemType={"DataType": 7, "ComputeDataType": 0})

    preds = buildPredicates(_makeArgs(field=["DataType=7"]))
    assert all(p(record) for p in preds)

    preds_no = buildPredicates(_makeArgs(field=["DataType=4"]))
    assert not all(p(record) for p in preds_no)


def test_fieldStringCoercion():
    """--field correctly matches string values."""
    record = _makeRecord(problemType={"OperationType": "GEMM", "DQuantType": "None"})

    preds = buildPredicates(_makeArgs(field=["OperationType=GEMM"]))
    assert all(p(record) for p in preds)

    preds_no = buildPredicates(_makeArgs(field=["OperationType=CONV"]))
    assert not all(p(record) for p in preds_no)


def test_fieldMissingKey():
    """--field returns False when the key is absent from problemType."""
    record = _makeRecord(problemType={"DataType": 7})
    preds = buildPredicates(_makeArgs(field=["NonExistentKey=something"]))
    assert not all(p(record) for p in preds)


# ---------------------------------------------------------------------------
# 9. AND combination
# ---------------------------------------------------------------------------

def test_andCombination():
    """All predicates must match; a single failing predicate rejects the record."""
    record = _makeRecord(problemType={
        "DataTypeA": 7,
        "DataTypeB": 7,
        "DQuantType": "MXFP8",
    })

    args_both_match = _makeArgs(input_type="bf16", dquant_type="MXFP8")
    preds = buildPredicates(args_both_match)
    assert all(p(record) for p in preds)

    args_one_fails = _makeArgs(input_type="bf16", dquant_type="Tile")
    preds2 = buildPredicates(args_one_fails)
    assert not all(p(record) for p in preds2)


# ---------------------------------------------------------------------------
# 10. Malformed file skip
# ---------------------------------------------------------------------------

def test_malformedFileReturnsNone(tmp_path):
    """loadLogicHeader returns None for a non-YAML file."""
    bad = tmp_path / "garbage.yaml"
    bad.write_bytes(b"\x00\x01\x02binary garbage \xff\xfe")
    assert loadLogicHeader(bad) is None


def test_nonYamlTextReturnsNone(tmp_path):
    """loadLogicHeader returns None when the content is not valid YAML structure."""
    bad = tmp_path / "notlogic.yaml"
    # A plain scalar string is valid YAML but not a dict or list — should return None.
    bad.write_text("just a plain string with no structure\n")
    assert loadLogicHeader(bad) is None


# ---------------------------------------------------------------------------
# 11. Exit codes
# ---------------------------------------------------------------------------

def test_exitCodeNoMatch(monkeypatch, tmp_path):
    """Exit code 1 when no matching files are found."""
    monkeypatch.setattr(sys, "argv", ["TensileLogicFilter", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1


def test_exitCodeMatch(monkeypatch, tmp_path):
    """Exit code 0 when at least one file matches."""
    content = {
        "ScheduleName": "s",
        "ArchitectureName": "gfx950",
        "CUCount": None,
        "DeviceNames": [],
        "ProblemType": {"DataTypeA": 7, "DataTypeB": 7, "DestDataType": 7},
        "Solutions": [],
    }
    (tmp_path / "test.yaml").write_text(yaml.dump(content))
    monkeypatch.setattr(sys, "argv", ["TensileLogicFilter", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0


def test_exitCodeUsageError(monkeypatch, tmp_path):
    """Exit code 2 when an unrecognised data-type alias is given."""
    monkeypatch.setattr(sys, "argv",
        ["TensileLogicFilter", str(tmp_path), "--input-type", "totally_unknown"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
