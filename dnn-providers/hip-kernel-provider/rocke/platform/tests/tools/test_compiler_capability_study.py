# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Contracts for historical COMGR replay, evidence, and published claims."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

_SOURCE_TOOLS = Path(__file__).resolve().parents[2] / "tools"
_INSTALLED_TOOLS = Path(__file__).resolve().parent
_TOOLS = (
    _SOURCE_TOOLS
    if (_SOURCE_TOOLS / "compiler_capabilities").is_dir()
    else _INSTALLED_TOOLS
)
_CAPABILITIES = _TOOLS / "compiler_capabilities"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from compiler_capabilities import check_historical_study as study
from compiler_capabilities import replay_historical as replay
from rocke.compiler import evidence


def _paths() -> tuple[Path, Path, Path, Path]:
    return (
        _CAPABILITIES / "historical_inputs.json",
        _CAPABILITIES / "historical_observations.json",
        _CAPABILITIES / "historical_study.md",
        _CAPABILITIES / "churn_report.md",
    )


def _changed_json(path: Path, data: dict[str, object], tmp_path: Path) -> Path:
    changed = tmp_path / path.name
    changed.write_text(json.dumps(data), encoding="utf-8")
    return changed


def test_published_comgr_study_matches_machine_evidence() -> None:
    study.check(*_paths())


def test_example_report_matches_the_installed_capability_contract() -> None:
    example = json.loads(
        (_CAPABILITIES / "example.capabilities.json").read_text(encoding="utf-8")
    )
    evidence.validate_report_structure(example)


@pytest.mark.parametrize(
    ("section", "field", "message"),
    [
        ("compiler", "sha256", "COMGR digest"),
        ("probe", "protocol_sha256", "protocol_sha256"),
        ("probe", "schema_sha256", "schema_sha256"),
    ],
)
def test_checker_rejects_missing_provenance_digest(
    tmp_path: Path, section: str, field: str, message: str
) -> None:
    inputs, observations_path, published, churn = _paths()
    observations = json.loads(observations_path.read_text())
    if section == "compiler":
        observations["weekly"][0]["compiler"][field] = ""
    else:
        observations["probe"][field] = ""
    changed = _changed_json(observations_path, observations, tmp_path)
    with pytest.raises(study.StudyError, match=message):
        study.check(inputs, changed, published, churn)


@pytest.mark.parametrize("field", ["protocol_sha256", "schema_sha256"])
def test_checker_rejects_probe_source_digest_drift(tmp_path: Path, field: str) -> None:
    inputs, observations_path, published, churn = _paths()
    observations = json.loads(observations_path.read_text())
    observations["probe"][field] = "f" * 64
    changed = _changed_json(observations_path, observations, tmp_path)
    with pytest.raises(study.StudyError, match=f"historical {field} differs"):
        study.check(inputs, changed, published, churn)


def test_checker_rejects_machine_local_paths(tmp_path: Path) -> None:
    inputs, observations_path, published, churn = _paths()
    for index, local_path in enumerate(("/opt/rocm/lib", "/root/sdk/lib")):
        observations = json.loads(observations_path.read_text())
        observations["weekly"][0]["compiler"]["version_source"] = local_path
        changed = tmp_path / f"observations-{index}.json"
        changed.write_text(json.dumps(observations), encoding="utf-8")
        with pytest.raises(study.StudyError, match="machine-local path"):
            study.check(inputs, changed, published, churn)


def test_checker_rejects_public_source_id_and_artifact_drift(tmp_path: Path) -> None:
    inputs_path, observations_path, published, churn = _paths()
    inputs = json.loads(inputs_path.read_text())
    inputs["public_sources"].pop()
    changed_inputs = _changed_json(inputs_path, inputs, tmp_path)
    with pytest.raises(study.StudyError, match="three public HTTPS sources"):
        study.check(changed_inputs, observations_path, published, churn)

    observations = json.loads(observations_path.read_text())
    observations["public_sources"].pop()
    changed = _changed_json(observations_path, observations, tmp_path)
    with pytest.raises(study.StudyError, match="public sources"):
        study.check(inputs_path, changed, published, churn)

    observations = json.loads(observations_path.read_text())
    observations["weekly"][1]["id"] = observations["weekly"][0]["id"]
    changed = _changed_json(observations_path, observations, tmp_path)
    with pytest.raises(study.StudyError, match="selected inputs|identifiers"):
        study.check(inputs_path, changed, published, churn)

    observations = json.loads(observations_path.read_text())
    observations["weekly"][0]["artifact_url"] = "file:///tmp/compiler.whl"
    changed = _changed_json(observations_path, observations, tmp_path)
    with pytest.raises(study.StudyError, match="selected inputs|public HTTPS"):
        study.check(inputs_path, changed, published, churn)


def test_checker_rejects_corpus_digest_drift(tmp_path: Path) -> None:
    inputs, observations_path, published, churn = _paths()
    observations = json.loads(observations_path.read_text())
    reference = observations["weekly"][0]["reports"]["gfx950"]
    capabilities = observations["vectors"][reference["vector"]]
    capability = next(iter(capabilities))
    capabilities[capability].pop()
    changed = _changed_json(observations_path, observations, tmp_path)
    with pytest.raises(
        study.StudyError, match="vector .* digest differs|corpus differs"
    ):
        study.check(inputs, changed, published, churn)


def test_checker_rejects_stale_published_boundaries(tmp_path: Path) -> None:
    inputs, observations, published, churn = _paths()
    changed = tmp_path / "historical_study.md"
    changed.write_text(
        published.read_text().replace("| W34 → W35 | 4 |", "| W34 → W35 | 3 |"),
        encoding="utf-8",
    )
    with pytest.raises(study.StudyError, match="published structural boundaries"):
        study.check(inputs, observations, changed, churn)


def test_compiler_member_selection_requires_comgr_and_llvm_dependencies() -> None:
    names = [
        "pkg/lib/libamd_comgr.so.3",
        "pkg/lib/llvm/lib/libLLVM.so.24",
        "pkg/lib/llvm/lib/libclang-cpp.so.24",
        "pkg/lib/rocm_sysdeps/lib/librocm_sysdeps_z.so.1",
        "pkg/lib/rocm_sysdeps/lib/librocm_sysdeps_zstd.so.1",
        "pkg/lib/llvm/bin/clang",
    ]
    assert replay._compiler_members(names) == names[:-1]
    with pytest.raises(replay.support.ProbeError, match="libamd_comgr"):
        replay._compiler_members(names[1:])


def test_replay_is_offline_unless_acquisition_is_explicit(tmp_path: Path) -> None:
    row = {
        "id": "W01",
        "kind": "wheel",
        "artifact_url": "https://example.com/compiler.whl",
    }
    with pytest.raises(replay.support.ProbeError, match="--fetch-missing"):
        replay._cached_comgr(
            row,
            tmp_path / "artifacts",
            fetch_missing=False,
            block_bytes=1024,
        )


def test_explicit_acquisition_populates_persistent_artifact_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = {
        "id": "W01",
        "kind": "wheel",
        "artifact_url": "https://example.com/compiler.whl",
    }
    extracts = 0

    def extract(url: str, destination: Path, *, block_bytes: int) -> Path:
        nonlocal extracts
        del url, block_bytes
        extracts += 1
        library = destination / "pkg" / "lib" / "libamd_comgr.so.3"
        library.parent.mkdir(parents=True)
        library.write_bytes(b"comgr")
        return library

    monkeypatch.setattr(replay, "_extract_wheel_comgr", extract)
    cache = tmp_path / "artifacts"
    acquired = replay._cached_comgr(row, cache, fetch_missing=True, block_bytes=1024)
    reused = replay._cached_comgr(row, cache, fetch_missing=False, block_bytes=1024)

    assert acquired == reused
    assert acquired.read_bytes() == b"comgr"
    assert extracts == 1


def test_replay_reuses_duplicate_artifact_and_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    url = "https://example.com/compiler.whl"
    inputs = {
        "format": "rocke-comgr-historical-inputs/v1",
        "public_sources": ["https://example.com/index"],
        "weekly": [
            {
                "id": identifier,
                "kind": "wheel",
                "artifact": "compiler.whl",
                "index_url": "https://example.com/",
                "artifact_url": url,
                "processors": ["gfx950"],
            }
            for identifier in ("W01", "W02")
        ],
    }
    inputs_path = tmp_path / "inputs.json"
    inputs_path.write_text(json.dumps(inputs), encoding="utf-8")
    output = tmp_path / "observations.json"
    example = json.loads(
        (_CAPABILITIES / "example.capabilities.json").read_text(encoding="utf-8")
    )
    extracts = 0
    probes = 0

    def cached(*args: object, **kwargs: object) -> Path:
        nonlocal extracts
        extracts += 1
        return tmp_path / "libamd_comgr.so.3"

    def probe_once(**kwargs: object) -> tuple[dict[str, object], bool]:
        nonlocal probes
        probes += 1
        report = copy.deepcopy(example)
        processor = str(kwargs["processor"])
        report["probe"]["processor"] = processor
        report["probe"]["effective_processor"] = processor
        report["probe"]["protocol_sha256"] = replay.corpus.protocol_sha256(processor)
        report["probe"]["schema_sha256"] = replay.evidence._schema_sha256()
        return report, False

    monkeypatch.setattr(replay, "_cached_comgr", cached)
    monkeypatch.setattr(replay.evidence, "load_or_probe", probe_once)
    document = replay.replay(
        inputs_path,
        output,
        processors=None,
        artifact_cache_dir=tmp_path / "artifacts",
        fetch_missing=False,
        jobs=1,
        timeout=1,
        block_bytes=1024,
    )

    assert extracts == 1
    assert probes == 1
    assert [row["id"] for row in document["weekly"]] == ["W01", "W02"]
    assert json.loads(output.read_text()) == document

    resumed = replay.replay(
        inputs_path,
        output,
        processors=None,
        artifact_cache_dir=tmp_path / "artifacts",
        fetch_missing=False,
        jobs=1,
        timeout=1,
        block_bytes=1024,
    )
    assert resumed == document
    assert extracts == 1
    assert probes == 1
