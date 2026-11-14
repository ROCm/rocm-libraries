# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CMake contracts for standalone platform and embedded rocKE test assets."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path
from types import ModuleType

import pytest
from rocke.assets import library_root, platform_root

pytestmark = pytest.mark.integration


@pytest.fixture
def platform() -> Path:
    root = platform_root()
    if not (root / "CMakeLists.txt").is_file():
        pytest.skip("source CMake project is absent from the installed test artifact")
    if shutil.which("cmake") is None:
        pytest.skip("cmake is required for the source configuration contract")
    return root


@pytest.fixture
def rocke(platform: Path) -> Path:
    root = platform.parent
    if not (root / "CMakeLists.txt").is_file():
        pytest.skip("rocKE umbrella project is absent from this platform copy")
    return root


def _configure(
    source: Path, build: Path, *options: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build),
            "-DBUILD_TESTING=OFF",
            "-DROCKE_BUILD_PYENV=OFF",
            "-DROCKE_INSTALL_TESTS=ON",
            "-DROCKE_INSTALL_PYTHON_TESTS=ON",
            "-DROCKE_INSTALL_TEST_GPU_REFERENCES=OFF",
            *options,
        ],
        capture_output=True,
        check=False,
        text=True,
    )


def _configure_embedded(
    rocke: Path, build: Path, repository: Path | None = None
) -> subprocess.CompletedProcess[str]:
    embedding = build.parent / "embedding source"
    embedding.mkdir(parents=True, exist_ok=True)
    (embedding / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.25.2)\n"
        "project(rocke_embedding_contract LANGUAGES C CXX)\n"
        f'add_subdirectory("{rocke.as_posix()}" rocke)\n'
    )
    options = ["-DROCKE_INSTALL_PROVIDER_CTEST_SUBDIR=hip_kernel_provider"]
    if repository is not None:
        options.append(f"-DROCM_LIBRARIES_ROOT={repository}")
    return _configure(embedding, build, *options)


def _hipdnn_assets_module() -> ModuleType:
    helper = library_root() / "tests" / "sdpa" / "hipdnn_assets.py"
    spec = importlib.util.spec_from_file_location(
        "rocke_hipdnn_assets_contract", helper
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_repository_assets(repository: Path) -> dict[str, Path]:
    module = _hipdnn_assets_module()
    source_repository = module.repo_root()
    assert source_repository is not None
    source_packs = (
        library_root().parent.parent / "src/engines/kernel_ingestor_engine/packs"
    )

    template = repository / module._SDPA_GRAPH_TEMPLATE
    template.parent.mkdir(parents=True)
    template.write_text("{}\n")

    bundles = repository / module._GOLDEN_BUNDLES
    graph_dir = bundles / "case" / "SdpaFwdSmoke" / "nested"
    graph_dir.mkdir(parents=True)
    graph = graph_dir / "graph.json"
    graph.write_text("{}\n")
    excluded = graph_dir / "support.json"
    excluded.write_text("{}\n")
    default = bundles / "case" / "SdpaFwdSmoke" / "Default" / "graph.json"
    default.parent.mkdir(parents=True)
    default.write_text("{}\n")

    headers = repository / source_packs.relative_to(source_repository)
    headers.mkdir(parents=True)
    dense = headers / "Gfx950AttentionDenseGeometry.hpp"
    unified = headers / "Gfx950AttentionUnified3DGeometry.hpp"
    dense.write_text("// dense\n")
    unified.write_text("// unified\n")
    return {
        "template": template,
        "graph": graph,
        "excluded": excluded,
        "default": default,
        "dense": dense,
        "unified": unified,
    }


def _generated_text(build: Path, name: str) -> str:
    generated = list(build.rglob(name))
    assert generated
    return "\n".join(path.read_text() for path in generated)


def test_standalone_platform_copy_configures(platform: Path, tmp_path: Path) -> None:
    copied_platform = tmp_path / "standalone source" / "platform"
    shutil.copytree(
        platform,
        copied_platform,
        ignore=shutil.ignore_patterns("__pycache__", ".venv*", "build*"),
    )

    result = _configure(copied_platform, tmp_path / "build directory")

    assert result.returncode == 0, result.stdout + result.stderr
    runner_path = copied_platform / "tests" / "run_all.py"
    spec = importlib.util.spec_from_file_location("rocke_copied_run_all", runner_path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    assert runner.relative_path_guard() == 0


def test_embedded_build_stages_available_assets(rocke: Path, tmp_path: Path) -> None:
    repository = tmp_path / "repository root"
    assets = _write_repository_assets(repository)
    build = tmp_path / "build directory"

    result = _configure_embedded(rocke, build, repository)

    assert result.returncode == 0, result.stdout + result.stderr
    install_text = _generated_text(build, "cmake_install.cmake")
    for name in ("template", "graph", "excluded", "default", "dense", "unified"):
        assert assets[name].as_posix() in install_text
    assert (
        "tests/library/tests/sdpa/hipdnn/integration-test-bundles/"
        "case/SdpaFwdSmoke/nested"
    ) in install_text
    assert "tests/library/tests/hkp_packs" in install_text
    ctest_text = _generated_text(build, "CTestTestfile.cmake.install")
    assert "--rocke-require-hipdnn-assets" in ctest_text
    assert "--rocke-require-hkp-pack-headers" in ctest_text


def test_source_relative_fallback_stages_available_headers(
    rocke: Path, tmp_path: Path
) -> None:
    build = tmp_path / "build directory"

    result = _configure_embedded(rocke, build)

    assert result.returncode == 0, result.stdout + result.stderr
    install_text = _generated_text(build, "cmake_install.cmake")
    for name in (
        "Gfx950AttentionDenseGeometry.hpp",
        "Gfx950AttentionUnified3DGeometry.hpp",
    ):
        assert name in install_text
    ctest_text = _generated_text(build, "CTestTestfile.cmake.install")
    assert "--rocke-require-hkp-pack-headers" in ctest_text


def test_embedded_build_without_assets_remains_optional(
    rocke: Path, tmp_path: Path
) -> None:
    repository = tmp_path / "empty repository root"
    repository.mkdir()
    build = tmp_path / "build directory"

    result = _configure_embedded(rocke, build, repository)

    assert result.returncode == 0, result.stdout + result.stderr
    install_text = _generated_text(build, "cmake_install.cmake")
    assert "tests/library/tests/hkp_packs" not in install_text
    ctest_text = _generated_text(build, "CTestTestfile.cmake.install")
    assert "--rocke-require-hipdnn-assets" not in ctest_text
    # Without the pack headers, the flag would fail every installed run of
    # rocke_library_attention_cpu_pytest on the missing copy.
    assert "--rocke-require-hkp-pack-headers" not in ctest_text
