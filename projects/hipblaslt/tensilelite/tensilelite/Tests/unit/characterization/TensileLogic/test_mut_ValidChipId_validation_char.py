# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Mutation-targeted tests for chip-ID placement and validation helpers."""

from pathlib import Path

import pytest

import Tensile.TensileLogic.ValidChipId as V
from Tensile.Common.Architectures import ArchInfo, LogicFileError

pytestmark = pytest.mark.unit


def test_report_chip_id_failure_writes_exact_stderr(capsys):
    V._reportChipIdFailure(Path("logic.yaml"), "bad chip")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Error: bad chip (file: logic.yaml)\n"


_PLACEMENT_CASES = [
    (
        "base-valid-arch-ids",
        "gfx950",
        {"id=75a3", "id=75b0"},
        Path("gfx950/Equality/logic.yaml"),
        None,
    ),
    (
        "base-rejects-cross-arch-id",
        "gfx950",
        {"id=74a0"},
        Path("gfx950/Equality/logic.yaml"),
        "base gfx950 logic may only declare chip IDs available for gfx950 "
        "['id=75a0', 'id=75a2', 'id=75a3', 'id=75a8', 'id=75b0', "
        "'id=75b2', 'id=75b3', 'id=75b8']; found ['id=74a0']",
    ),
    (
        "malformed-chip-id-directory",
        "gfx950",
        {"id=75a3"},
        Path("gfx950_75a3/logic.yaml"),
        "chip-ID directory 'gfx950_75a3' must use gfx950_id<chip> format",
    ),
    (
        "non-source-chip-id-directory",
        "gfx950",
        {"id=75a0"},
        Path("gfx950_id75a0/logic.yaml"),
        "gfx950_id directory uses non-source chip ID id=75a0",
    ),
    (
        "source-directory-must-declare-own-id",
        "gfx950",
        {"id=75b0"},
        Path("gfx950_id75a3/logic.yaml"),
        "id=75a3 directory must contain id=75a3 in the YAML Device list",
    ),
    (
        "source-directory-rejects-default-id",
        "gfx950",
        {"id=75a3", "id=75a0"},
        Path("gfx950_id75a3/logic.yaml"),
        "id=75a3 directory may not declare default fallback chip IDs ['id=75a0']",
    ),
    (
        "source-directory-accepts-source-family-sibling",
        "gfx950",
        {"id=75a3", "id=75b0"},
        Path("gfx950_id75a3/logic.yaml"),
        None,
    ),
    (
        "source-directory-rejects-outside-family",
        "gfx950",
        {"id=75a3", "id=74a0"},
        Path("gfx950_id75a3/logic.yaml"),
        "id=75a3 directory may only declare chip IDs in fallback family "
        "['id=75a0', 'id=75a2', 'id=75a3', 'id=75a8', 'id=75b0', "
        "'id=75b2', 'id=75b3', 'id=75b8']; found ['id=74a0', 'id=75a3']",
    ),
]


@pytest.mark.parametrize(
    "name,gfx,device_ids,filepath,expected",
    _PLACEMENT_CASES,
    ids=[case[0] for case in _PLACEMENT_CASES],
)
def test_validate_chip_id_placement_branches(name, gfx, device_ids, filepath, expected):
    assert V._validateChipIdPlacement(gfx, device_ids, filepath) == expected


def test_validate_chip_id_passes_parser_args_and_logic_relative_path(monkeypatch):
    calls = {}

    def fake_extract(filepath, validateDeviceIds=True):
        calls["extract"] = (filepath, validateDeviceIds)
        return ArchInfo(
            Name="gfx950",
            Gfx="gfx950",
            DeviceIds={"id=75a3"},
        )

    def fake_verify(device_id, gfx):
        calls.setdefault("verify", []).append((device_id, gfx))
        return device_id

    def fake_placement(gfx, device_ids, filepath):
        calls["placement"] = (gfx, device_ids, filepath)
        return None

    monkeypatch.setattr(V, "_extractArchInfo", fake_extract)
    monkeypatch.setattr(V, "_verifyPredicate", fake_verify)
    monkeypatch.setattr(V, "_validateChipIdPlacement", fake_placement)

    file_path = Path("actual.yaml")
    relative_path = Path("gfx950_id75a3/logic.yaml")
    assert V._validateChipId(file_path, logic_relative_path=relative_path) is True
    assert calls["extract"] == (file_path, False)
    assert calls["verify"] == [("id=75a3", "gfx950")]
    assert calls["placement"] == ("gfx950", {"id=75a3"}, relative_path)


def test_validate_chip_id_non_chip_arch_returns_before_device_checks(monkeypatch):
    calls = {"verify": 0, "placement": 0}

    monkeypatch.setattr(
        V,
        "_extractArchInfo",
        lambda filepath, validateDeviceIds=True: ArchInfo(
            Name="gfx942",
            Gfx="gfx942",
            DeviceIds={"id=74a0"},
        ),
    )
    monkeypatch.setattr(V, "_verifyPredicate", lambda device_id, gfx: calls.__setitem__("verify", 1))
    monkeypatch.setattr(
        V,
        "_validateChipIdPlacement",
        lambda gfx, device_ids, filepath: calls.__setitem__("placement", 1),
    )

    assert V._validateChipId(Path("gfx942.yaml")) is True
    assert calls == {"verify": 0, "placement": 0}


def test_validate_chip_id_reports_parse_error(monkeypatch, capsys):
    def fake_extract(filepath, validateDeviceIds=True):
        raise LogicFileError("bad header")

    monkeypatch.setattr(V, "_extractArchInfo", fake_extract)

    assert V._validateChipId(Path("bad.yaml"), report_path=Path("report.yaml")) is False
    assert capsys.readouterr().err == (
        "Error: Chip ID validation failed: bad header (file: report.yaml)\n"
    )


def test_validate_chip_id_reports_missing_device_ids(monkeypatch, capsys):
    monkeypatch.setattr(
        V,
        "_extractArchInfo",
        lambda filepath, validateDeviceIds=True: ArchInfo(
            Name="gfx950",
            Gfx="gfx950",
            DeviceIds=set(),
        ),
    )

    assert V._validateChipId(Path("empty.yaml"), report_path=Path("report.yaml")) is False
    assert capsys.readouterr().err == (
        "Error: gfx950 logic must declare at least one Device chip ID "
        "(file: report.yaml)\n"
    )


def test_validate_chip_id_reports_predicate_error(monkeypatch, capsys):
    monkeypatch.setattr(
        V,
        "_extractArchInfo",
        lambda filepath, validateDeviceIds=True: ArchInfo(
            Name="gfx950",
            Gfx="gfx950",
            DeviceIds={"id=74a0"},
        ),
    )
    monkeypatch.setattr(
        V,
        "_verifyPredicate",
        lambda device_id, gfx: (_ for _ in ()).throw(ValueError("wrong chip")),
    )

    assert V._validateChipId(Path("bad-device.yaml"), report_path=Path("report.yaml")) is False
    assert capsys.readouterr().err == (
        "Error: ValidChipId failed (ValueError): wrong chip (file: report.yaml)\n"
    )


def test_validate_chip_id_reports_placement_error(monkeypatch, capsys):
    monkeypatch.setattr(
        V,
        "_extractArchInfo",
        lambda filepath, validateDeviceIds=True: ArchInfo(
            Name="gfx950",
            Gfx="gfx950",
            DeviceIds={"id=75a3"},
        ),
    )
    monkeypatch.setattr(V, "_verifyPredicate", lambda device_id, gfx: device_id)
    monkeypatch.setattr(
        V,
        "_validateChipIdPlacement",
        lambda gfx, device_ids, filepath: "bad placement",
    )

    assert V._validateChipId(Path("placed.yaml"), report_path=Path("report.yaml")) is False
    assert capsys.readouterr().err == "Error: bad placement (file: report.yaml)\n"


def test_validate_chip_id_accepts_dict_format_logic(tmp_path):
    logic = tmp_path / "logic.yaml"
    logic.write_text(
        "\n".join(
            [
                "MinimumRequiredVersion: 5.0.0",
                "ScheduleName: gfx950",
                "ArchitectureName: gfx950",
                "CUCount: 256",
                "DeviceNames: [Device 75a0]",
                "ProblemType: {}",
                "DefaultSolution: {}",
                "Solutions: []",
                "IndexOrder: [2, 3, 0, 1]",
                "ExactLogic: []",
                "RangeLogic: null",
                "TileSelectionIndices: null",
                "PerfMetric: DeviceEfficiency",
                "LibraryType: Equality",
                "",
            ]
        ),
        encoding="utf-8",
    )

    assert V._validateChipId(logic, logic_relative_path=Path("gfx950/Equality/logic.yaml")) is True
