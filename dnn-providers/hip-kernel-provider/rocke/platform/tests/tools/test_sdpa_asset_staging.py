# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""External SDPA test assets do not bind the SDK to its embedding repository."""

from __future__ import annotations

import importlib.util
import json
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PLATFORM = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    not (PLATFORM / "CMakeLists.txt").is_file(),
    reason="source-only CMake installation contracts require the SDK sources",
)


def _inputs(root):
    root.mkdir(parents=True)
    template = root / "caller-template.json"
    template.write_text('{"template": true}\n', encoding="utf-8")
    bundles = root / "caller-bundles"
    files = {
        "family/SdpaFwdCase/graph.json": '{"graph": 1}\n',
        "family/SdpaFwdOther/nested/graph.json": '{"graph": 2}\n',
        "family/SdpaFwdCase/Default/graph.json": "{}\n",
        "family/SdpaFwdCase/meta.json": "{}\n",
        "family/SdpaFwdCase/x_support.json": "{}\n",
        "family/SdpaFwdCase/x_sweep.json": "{}\n",
        "family/OtherOp/graph.json": "{}\n",
    }
    for name, content in files.items():
        path = bundles / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return template, bundles, files


def _configure_stage(tmp_path, template=None, bundles=None):
    source = tmp_path / "copied-sdk"
    source.mkdir()
    shutil.copy2(
        PLATFORM / "cmake/StageHipdnnSdpaTestAssets.cmake",
        source / "StageHipdnnSdpaTestAssets.cmake",
    )
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\n"
        "project(AssetStage NONE)\n"
        'set(ROCKE_TEST_INSTALL_DIR "bin/renamed-provider")\n'
        'include("${CMAKE_CURRENT_LIST_DIR}/StageHipdnnSdpaTestAssets.cmake")\n'
        'file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/asset-state.txt" "${_ROCKE_SDPA_HIPDNN_ASSETS}")\n',
        encoding="utf-8",
    )
    build = tmp_path / "build"
    command = ["cmake", "-S", str(source), "-B", str(build)]
    if template is not None:
        command.append(f"-DROCKE_HIPDNN_SDPA_TEMPLATE={template}")
    if bundles is not None:
        command.append(f"-DROCKE_HIPDNN_SDPA_BUNDLES_DIR={bundles}")
    result = subprocess.run(command, capture_output=True, text=True)
    return build, result


def test_platform_copyability_guard_is_green():
    runner = runpy.run_path(str(PLATFORM / "tests/run_all.py"))
    assert runner["relative_path_guard"]() == 0


def test_copied_asset_module_stages_only_selected_graphs_and_relocates(
    tmp_path, monkeypatch
):
    template, bundles, files = _inputs(tmp_path / "external inputs")
    build, result = _configure_stage(tmp_path, template, bundles)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (build / "asset-state.txt").read_text() == "ON"
    prefix = tmp_path / "prefix"
    subprocess.run(
        ["cmake", "--install", str(build), "--prefix", str(prefix)], check=True
    )
    relocated = tmp_path / "relocated"
    prefix.rename(relocated)
    assets = relocated / "bin/renamed-provider/tests/library/tests/sdpa/hipdnn"
    expected = {
        "family/SdpaFwdCase/graph.json",
        "family/SdpaFwdOther/nested/graph.json",
    }
    assert {
        p.relative_to(assets / "integration-test-bundles").as_posix()
        for p in assets.rglob("graph.json")
    } == expected
    assert (
        assets / "attention_sdpa_fwd.graph.json"
    ).read_bytes() == template.read_bytes()
    for name in expected:
        assert (assets / "integration-test-bundles" / name).read_text() == files[name]

    # The shipped reader uses only the installed copy after both input roots disappear.
    shutil.copy2(
        PLATFORM.parent / "library/tests/sdpa/hipdnn_assets.py",
        assets.parent / "hipdnn_assets.py",
    )
    shutil.rmtree(template.parent)
    spec = importlib.util.spec_from_file_location(
        "relocated_sdpa_assets", assets.parent / "hipdnn_assets.py"
    )
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    monkeypatch.setattr(reader, "library_root", lambda: tmp_path / "no-source-library")
    assert reader.sdpa_graph_template() == assets / "attention_sdpa_fwd.graph.json"
    assert {
        p.relative_to(assets / "integration-test-bundles").as_posix()
        for p in reader.sdpa_fwd_golden_graphs()
    } == expected
    (assets / "attention_sdpa_fwd.graph.json").unlink()
    assert reader.sdpa_graph_template() is None


def test_no_inputs_or_no_selected_graphs_leave_assets_optional(tmp_path):
    build, result = _configure_stage(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (build / "asset-state.txt").read_text() == "OFF"
    other = tmp_path / "no-sdpa"
    other.mkdir()
    template, bundles, _ = _inputs(other / "inputs")
    shutil.rmtree(bundles / "family")
    bundles.mkdir(exist_ok=True)
    build, result = _configure_stage(other, template, bundles)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (build / "asset-state.txt").read_text() == "OFF"


@pytest.mark.parametrize(
    "problem",
    [
        "template_only",
        "bundles_only",
        "missing_file",
        "directory_as_file",
        "missing_directory",
        "file_as_directory",
    ],
)
def test_explicit_invalid_inputs_fail_instead_of_losing_coverage(tmp_path, problem):
    template, bundles, _ = _inputs(tmp_path / "inputs")
    options = {
        "template_only": (template, None),
        "bundles_only": (None, bundles),
        "missing_file": (tmp_path / "missing.json", bundles),
        "directory_as_file": (bundles, bundles),
        "missing_directory": (template, tmp_path / "missing-dir"),
        "file_as_directory": (template, template),
    }
    _, result = _configure_stage(tmp_path, *options[problem])
    assert result.returncode != 0
    assert "ROCKE_HIPDNN_SDPA_" in result.stdout + result.stderr


def test_platform_copy_configures_without_an_embedding_repository(tmp_path):
    copied = tmp_path / "standalone-sdk"
    copied.mkdir()
    for path in PLATFORM.iterdir():
        if path.is_file():
            shutil.copy2(path, copied / path.name)
    for name in ("cpp", "cmake", "python", "tools"):
        shutil.copytree(
            PLATFORM / name,
            copied / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    result = subprocess.run(
        [
            "cmake",
            "-S",
            str(copied),
            "-B",
            str(tmp_path / "copy-build"),
            "-DBUILD_TESTING=OFF",
            "-DROCKE_INSTALL_TESTS=OFF",
            "-DROCKE_BUILD_PYBIND=OFF",
            f"-DPython3_EXECUTABLE={sys.executable}",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (copied.parent / "library").exists()


def test_platform_registers_required_assets_from_explicit_inputs(tmp_path):
    template, bundles, _ = _inputs(tmp_path / "external")
    build = tmp_path / "sdk-build"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(PLATFORM),
            "-B",
            str(build),
            "-DBUILD_TESTING=OFF",
            "-DROCKE_BUILD_PYBIND=OFF",
            "-DROCKE_INSTALL_TEST_GPU_REFERENCES=OFF",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DROCKE_HIPDNN_SDPA_TEMPLATE={template}",
            f"-DROCKE_HIPDNN_SDPA_BUNDLES_DIR={bundles}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = build / "CTestTestfile.cmake.install"
    listing = tmp_path / "manifest"
    listing.mkdir()
    shutil.copy2(manifest, listing / "CTestTestfile.cmake")
    result = subprocess.run(
        ["ctest", "--test-dir", str(listing), "--show-only=json-v1"],
        check=True,
        capture_output=True,
        text=True,
    )
    entry = next(
        t
        for t in json.loads(result.stdout)["tests"]
        if t["name"] == "rocke_library_sdpa_pytest"
    )
    assert "--rocke-require-hipdnn-assets" in entry["command"]
