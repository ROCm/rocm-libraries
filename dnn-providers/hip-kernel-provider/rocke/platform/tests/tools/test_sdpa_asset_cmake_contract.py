# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CMake contracts for standalone platform and embedded rocKE test assets."""

from __future__ import annotations

import importlib.util
import re
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
    rocke: Path, build: Path, *options: str
) -> subprocess.CompletedProcess[str]:
    embedding = build.parent / "embedding source"
    embedding.mkdir(parents=True, exist_ok=True)
    (embedding / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.25.2)\n"
        "project(rocke_embedding_contract LANGUAGES C CXX)\n"
        f'add_subdirectory("{rocke.as_posix()}" rocke)\n'
    )
    return _configure(
        embedding,
        build,
        "-DROCKE_INSTALL_PROVIDER_CTEST_SUBDIR=hip_kernel_provider",
        *options,
    )


def _write_external_assets(
    template: Path, bundles: Path, headers: Path
) -> tuple[Path, Path, Path]:
    template.parent.mkdir(parents=True, exist_ok=True)
    template.write_text("{}\n")

    eligible = bundles / "case" / "SdpaFwdSmoke" / "nested" / "graph.json"
    eligible.parent.mkdir(parents=True)
    eligible.write_text("{}\n")
    for excluded in ("meta.json", "support.json", "sweep.json"):
        (eligible.parent / excluded).write_text("{}\n")
    default = bundles / "case" / "SdpaFwdSmoke" / "Default" / "graph.json"
    default.parent.mkdir(parents=True)
    default.write_text("{}\n")
    other = bundles / "case" / "OtherOp" / "graph.json"
    other.parent.mkdir(parents=True)
    other.write_text("{}\n")

    headers.mkdir(parents=True)
    (headers / "Gfx950AttentionDenseGeometry.hpp").write_text("// dense\n")
    (headers / "Gfx950AttentionUnified3DGeometry.hpp").write_text("// unified\n")
    return template, bundles, headers


def _external_assets(assets: Path) -> tuple[list[str], Path, Path, Path]:
    template, bundles, headers = _write_external_assets(
        assets / "template source.json",
        assets / "bundle source",
        assets / "pack headers",
    )
    options = [
        f"-DROCKE_HIPDNN_SDPA_TEMPLATE={template}",
        f"-DROCKE_HIPDNN_SDPA_BUNDLES={bundles}",
        f"-DROCKE_PROVIDER_PACK_HEADERS={headers}",
    ]
    return options, template, bundles, headers


def _hipdnn_assets_module() -> ModuleType:
    helper = library_root() / "tests" / "sdpa" / "hipdnn_assets.py"
    spec = importlib.util.spec_from_file_location(
        "rocke_hipdnn_assets_contract", helper
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _repository_assets(repository: Path) -> tuple[Path, Path, Path]:
    module = _hipdnn_assets_module()
    source_repository = module.repo_root()
    assert source_repository is not None
    source_packs = (
        library_root().parent.parent / "src/engines/kernel_ingestor_engine/packs"
    )
    return _write_external_assets(
        repository / module._SDPA_GRAPH_TEMPLATE,
        repository / module._GOLDEN_BUNDLES,
        repository / source_packs.relative_to(source_repository),
    )


def _generated_install_text(build: Path) -> str:
    scripts = list(build.rglob("cmake_install.cmake"))
    assert scripts
    return "\n".join(script.read_text() for script in scripts)


def _generated_ctest_text(build: Path) -> str:
    manifests = list(build.rglob("CTestTestfile.cmake.install"))
    assert manifests
    return "\n".join(manifest.read_text() for manifest in manifests)


def _python_selected_graphs(bundles: Path) -> set[Path]:
    module = _hipdnn_assets_module()
    module._INSTALLED_BUNDLES = bundles
    module.repo_root = lambda: None
    return set(module.sdpa_fwd_golden_graphs())


def _install_rule(install_text: str, source: Path) -> str:
    source_position = install_text.index(source.as_posix())
    rule_start = install_text.rfind("if(CMAKE_INSTALL_COMPONENT", 0, source_position)
    rule_end = install_text.index("endif()", source_position)
    assert rule_start >= 0
    return install_text[rule_start:rule_end]


def _assert_default_install_rule(
    install_text: str, source: Path, destination: str
) -> None:
    rule = _install_rule(install_text, source)
    assert 'CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified"' in rule
    assert f'DESTINATION "${{CMAKE_INSTALL_PREFIX}}/{destination}"' in rule


def _installed_source(install_text: str, expected: Path) -> Path:
    pattern = rf'"([^"\n]*[/\\]{re.escape(expected.name)})"'
    matches = [Path(match) for match in re.findall(pattern, install_text)]
    resolved_expected = expected.resolve()
    matching = [source for source in matches if source.resolve() == resolved_expected]
    assert len(matching) == 1
    return matching[0]


def test_standalone_platform_copy_configures_with_paths_containing_spaces(
    platform: Path, tmp_path: Path
) -> None:
    copied_platform = tmp_path / "standalone source" / "platform"
    shutil.copytree(
        platform,
        copied_platform,
        ignore=shutil.ignore_patterns("__pycache__", ".venv*", "build*"),
    )

    result = _configure(copied_platform, tmp_path / "build directory")

    assert result.returncode == 0, result.stdout + result.stderr


def test_platform_passes_the_existing_relative_path_guard(platform: Path) -> None:
    runner_path = platform / "tests" / "run_all.py"
    spec = importlib.util.spec_from_file_location(
        "rocke_copyability_runner", runner_path
    )
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    assert runner.relative_path_guard() == 0


def _assert_sdpa_and_pack_assets_staged(
    build: Path, template: Path, bundles: Path, headers: Path
) -> None:
    install_text = _generated_install_text(build)
    sdpa_destination = "bin/hip_kernel_provider/tests/library/tests/sdpa/hipdnn"
    _assert_default_install_rule(install_text, template, sdpa_destination)
    template_rule = _install_rule(install_text, template)
    assert 'RENAME "attention_sdpa_fwd.graph.json"' in template_rule

    selected = _python_selected_graphs(bundles)
    expected_graph = bundles / "case" / "SdpaFwdSmoke" / "nested" / "graph.json"
    assert selected == {expected_graph}
    graph_destination = (
        f"{sdpa_destination}/integration-test-bundles/case/SdpaFwdSmoke/nested"
    )
    _assert_default_install_rule(install_text, expected_graph, graph_destination)
    for graph in bundles.rglob("*.json"):
        assert (graph.as_posix() in install_text) == (graph == expected_graph)

    header_destination = "bin/hip_kernel_provider/tests/library/tests/hkp_packs"
    for header in headers.iterdir():
        _assert_default_install_rule(install_text, header, header_destination)
    assert "--rocke-require-hipdnn-assets" in _generated_ctest_text(build)
    assert "rocke_external_test_assets" not in install_text


def test_embedded_configure_stages_explicit_sdpa_and_pack_assets(
    rocke: Path, tmp_path: Path
) -> None:
    options, template, bundles, headers = _external_assets(tmp_path / "external assets")
    build = tmp_path / "build directory"

    result = _configure_embedded(rocke, build, *options)

    assert result.returncode == 0, result.stdout + result.stderr
    _assert_sdpa_and_pack_assets_staged(build, template, bundles, headers)


def test_rocm_libraries_root_defaults_stage_sdpa_and_pack_assets(
    rocke: Path, tmp_path: Path
) -> None:
    repository = tmp_path / "repository root"
    template, bundles, headers = _repository_assets(repository)
    build = tmp_path / "build directory"

    result = _configure_embedded(rocke, build, f"-DROCM_LIBRARIES_ROOT={repository}")

    assert result.returncode == 0, result.stdout + result.stderr
    _assert_sdpa_and_pack_assets_staged(build, template, bundles, headers)


def test_source_relative_defaults_stage_available_pack_headers(
    rocke: Path, tmp_path: Path
) -> None:
    headers = rocke.parent / "src/engines/kernel_ingestor_engine/packs"
    build = tmp_path / "build directory"

    result = _configure_embedded(rocke, build)

    assert result.returncode == 0, result.stdout + result.stderr
    install_text = _generated_install_text(build)
    assert headers.is_dir()
    header_destination = "bin/hip_kernel_provider/tests/library/tests/hkp_packs"
    for name in (
        "Gfx950AttentionDenseGeometry.hpp",
        "Gfx950AttentionUnified3DGeometry.hpp",
    ):
        source = _installed_source(install_text, headers / name)
        _assert_default_install_rule(install_text, source, header_destination)


def test_pack_headers_stage_without_sdpa_assets(rocke: Path, tmp_path: Path) -> None:
    options, template, bundles, headers = _external_assets(tmp_path / "external assets")
    template.unlink()
    shutil.rmtree(bundles)
    build = tmp_path / "build directory"

    result = _configure_embedded(rocke, build, *options)

    assert result.returncode == 0, result.stdout + result.stderr
    install_text = _generated_install_text(build)
    for header in headers.iterdir():
        assert header.as_posix() in install_text
    assert "--rocke-require-hipdnn-assets" not in _generated_ctest_text(build)


@pytest.mark.parametrize(
    "missing_header",
    ("Gfx950AttentionDenseGeometry.hpp", "Gfx950AttentionUnified3DGeometry.hpp"),
)
def test_sdpa_assets_stage_when_either_pack_header_is_missing(
    rocke: Path, tmp_path: Path, missing_header: str
) -> None:
    options, template, _, headers = _external_assets(tmp_path / "external assets")
    (headers / missing_header).unlink()
    build = tmp_path / "build directory"

    result = _configure_embedded(rocke, build, *options)

    assert result.returncode == 0, result.stdout + result.stderr
    install_text = _generated_install_text(build)
    assert template.as_posix() in install_text
    assert "Gfx950AttentionDenseGeometry.hpp" not in install_text
    assert "Gfx950AttentionUnified3DGeometry.hpp" not in install_text
    assert "--rocke-require-hipdnn-assets" in _generated_ctest_text(build)


@pytest.mark.parametrize(
    "disabled", ("ROCKE_INSTALL_TESTS", "ROCKE_INSTALL_PYTHON_TESTS")
)
def test_disabled_test_installation_remains_permissive(
    rocke: Path, tmp_path: Path, disabled: str
) -> None:
    options, _, _, _ = _external_assets(tmp_path / "external assets")

    result = _configure_embedded(
        rocke,
        tmp_path / "build directory",
        *options,
        f"-D{disabled}=OFF",
    )

    assert result.returncode == 0, result.stdout + result.stderr
