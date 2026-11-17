# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Contract tests for the published compiler capability study."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_PLATFORM = Path(__file__).resolve().parents[2]
_TOOLS = _PLATFORM / "tools"
_CAPABILITIES_DIR = _TOOLS / "compiler_capabilities"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from compiler_capabilities import check_historical_study as study


def test_published_study_matches_machine_readable_observations() -> None:
    study.check(
        _CAPABILITIES_DIR / "historical_observations.json",
        _CAPABILITIES_DIR / "historical_study.md",
        _CAPABILITIES_DIR / "churn_report.md",
    )


def _check_changed(data: dict[str, object], tmp_path: Path) -> None:
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(data))
    study.check(
        changed,
        _CAPABILITIES_DIR / "historical_study.md",
        _CAPABILITIES_DIR / "churn_report.md",
    )


def test_study_checker_rejects_a_changed_feature(
    tmp_path: Path,
) -> None:
    data = json.loads((_CAPABILITIES_DIR / "historical_observations.json").read_text())
    data["weekly"][0]["features"]["Complete data layout"] = "changed"
    with pytest.raises(study.StudyError, match="published matrix differs for W06"):
        _check_changed(data, tmp_path)


def test_study_checker_rejects_local_paths_and_provenance_drift(
    tmp_path: Path,
) -> None:
    original = json.loads(
        (_CAPABILITIES_DIR / "historical_observations.json").read_text()
    )
    local_path = json.loads(json.dumps(original))
    local_path["weekly"][0]["compiler"][
        "version_text"
    ] += "\nInstalledDir: /home/user/sdk"
    with pytest.raises(study.StudyError, match="machine-local path"):
        _check_changed(local_path, tmp_path)

    opt_local_path = json.loads(json.dumps(original))
    opt_local_path["weekly"][0]["opt"]["version_text"] += " /home/user/libLLVM.so"
    with pytest.raises(study.StudyError, match="machine-local path"):
        _check_changed(opt_local_path, tmp_path)

    non_public_url = json.loads(json.dumps(original))
    non_public_url["weekly"][0]["index_url"] = "file:///tmp/index/"
    non_public_url["weekly"][0]["artifact_url"] = (
        "file:///tmp/index/" + non_public_url["weekly"][0]["artifact"]
    )
    with pytest.raises(study.StudyError, match="not public HTTPS"):
        _check_changed(non_public_url, tmp_path)

    wrong_repository = json.loads(json.dumps(original))
    wrong_repository["weekly"][0]["compiler"][
        "repository"
    ] = "https://example.com/llvm-project"
    with pytest.raises(study.StudyError, match="repository disagrees"):
        _check_changed(wrong_repository, tmp_path)
