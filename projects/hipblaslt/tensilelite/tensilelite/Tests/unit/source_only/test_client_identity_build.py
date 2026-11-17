# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Compile the production metadata module without HIP or a GPU."""

import json
import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit
_ROOT = Path(__file__).resolve().parents[5]
_SHA = "0123456789abcdef0123456789abcdef01234567"


def _run(*args, cwd=None):
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("TENSILELITE_SOURCE_REVISION", "ROCM_PATH", "ROCM_VERSION")
    }
    result = subprocess.run(args, cwd=cwd, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.fixture
def project(tmp_path):
    src = tmp_path / "src"
    files = [
        "cmake/TensileLiteMetadata.cmake",
        "cmake/GenerateTensileLiteMetadata.cmake",
        "cmake/TensileLiteIdentityTests.cmake",
        "tensilelite/release_metadata.py",
        "tensilelite/VERSION",
        "tensilelite/setup.py",
        "tensilelite/tensilelite/_source_revision.py",
        "tensilelite/tensilelite/_build_rocm.py",
        "tensilelite/client/TensileLiteClientVersion.hpp.in",
        "tensilelite/tests/check_client_version.cmake",
    ]
    for name in files:
        dest = src / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_ROOT / name, dest)
    rocm = tmp_path / "rocm"
    (rocm / ".info").mkdir(parents=True)
    (rocm / ".info/version").write_text("10.3.0\n", encoding="utf-8")
    (src / "CMakeLists.txt").write_text(
        """
cmake_minimum_required(VERSION 3.25)
project(client_identity LANGUAGES CXX)
include(GNUInstallDirs)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(HIPBLASLT_BUILD_ROCM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../rocm")
include(cmake/TensileLiteMetadata.cmake)
hipblaslt_configure_tensilelite_metadata("${CMAKE_CURRENT_SOURCE_DIR}/tensilelite"
    "${CMAKE_CURRENT_BINARY_DIR}/metadata")
add_executable(tensilelite-client probe.cpp)
hipblaslt_attach_tensilelite_metadata(tensilelite-client)
enable_testing()
set(TENSILELITE_TESTS_INSTALL_SUBDIR hipblaslt-tensilelite)
set(TENSILELITE_TESTS_CTEST_INSTALL_FILE "${CMAKE_CURRENT_BINARY_DIR}/install_CTestTestfile.cmake")
file(WRITE "${TENSILELITE_TESTS_CTEST_INSTALL_FILE}" "")
include(cmake/TensileLiteIdentityTests.cmake)
hipblaslt_add_tensilelite_identity_tests()
install(TARGETS tensilelite-client RUNTIME DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/hipblaslt/tensilelite" COMPONENT tests)
install(FILES "${TENSILELITE_TESTS_CTEST_INSTALL_FILE}" DESTINATION "${CMAKE_INSTALL_BINDIR}/${TENSILELITE_TESTS_INSTALL_SUBDIR}" RENAME CTestTestfile.cmake COMPONENT tests)
""",
        encoding="utf-8",
    )
    (src / "probe.cpp").write_text(
        """
#include "TensileLiteClientVersion.hpp"
#include <iostream>
#include <string>
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    std::string query(argv[1]);
    if (query == "--version") std::cout << TENSILELITE_CLIENT_VERSION;
    else if (query == "--source-revision") std::cout << TENSILELITE_CLIENT_SOURCE_REVISION;
    else if (query == "--build-rocm") std::cout << TENSILELITE_CLIENT_BUILD_ROCM;
    else return 2;
    std::cout << "\\n";
}
""",
        encoding="utf-8",
    )
    _run("git", "init", "-q", cwd=src)
    _git(src, "add", ".")
    _git(src, "commit", "-qm", "initial")
    return src, tmp_path / "build"


def _git(src, *args):
    return _run(
        "git",
        "-c",
        "user.name=Identity Test",
        "-c",
        "user.email=identity@example.invalid",
        "-c",
        "commit.gpgsign=false",
        *args,
        cwd=src,
    ).strip()


def _configure(src, build, *args, cmake="cmake"):
    _run(
        cmake,
        "-S",
        str(src),
        "-B",
        str(build),
        "-G",
        "Ninja",
        f"-DPython3_EXECUTABLE={sys.executable}",
        *args,
    )


def _build(build, cmake="cmake"):
    return _run(cmake, "--build", str(build), "--target", "tensilelite-client")


def _copy_build_cmake(tmp_path):
    # Copy the native executable: a symlink or Python launcher can resolve back
    # to the runner's CMake, leaving the configure-time path available after relocation.
    script = tmp_path / "cmake-paths.cmake"
    paths = tmp_path / "cmake-paths.txt"
    script.write_text(
        'file(WRITE "${OUTPUT}" "${CMAKE_COMMAND}\\n${CMAKE_ROOT}\\n")\n',
        encoding="utf-8",
    )
    _run("cmake", f"-DOUTPUT={paths}", "-P", str(script))
    executable, resources = (Path(path) for path in paths.read_text().splitlines())
    toolchain = tmp_path / "build tools"
    (toolchain / "bin").mkdir(parents=True)
    cmake = toolchain / "bin" / executable.name
    shutil.copy2(executable, cmake)
    (toolchain / "share").mkdir()
    (toolchain / "share" / resources.name).symlink_to(resources, target_is_directory=True)
    _run(cmake, f"-DOUTPUT={paths}", "-P", str(script))
    assert Path(paths.read_text().splitlines()[0]) == cmake
    return cmake


def _query(build, option="--source-revision"):
    return _run(str(build / "tensilelite-client"), option).strip()


def _current(src):
    return _run(
        sys.executable, str(src / "tensilelite/release_metadata.py"), "source-revision"
    ).strip()


def test_identity_tracks_dirty_edits_reverts_and_unchanged_builds(project):
    src, build = project
    _configure(src, build)
    _build(build)
    clean = _git(src, "rev-parse", "HEAD")
    assert _query(build) == clean
    header = build / "metadata/TensileLiteClientVersion.hpp"
    executable = build / "tensilelite-client"
    times = header.stat().st_mtime_ns, executable.stat().st_mtime_ns
    _build(build)
    assert (header.stat().st_mtime_ns, executable.stat().st_mtime_ns) == times
    probe = src / "probe.cpp"
    original = probe.read_text(encoding="utf-8")
    probe.write_text(original + "// first edit\n", encoding="utf-8")
    first = _current(src)
    _build(build)
    assert _query(build) == first
    assert first.startswith(clean + ".dirty.")
    probe.write_text(original + "// second edit\n", encoding="utf-8")
    second = _current(src)
    assert second != first
    _build(build)
    assert _query(build) == second
    probe.write_text(original, encoding="utf-8")
    _build(build)
    assert _query(build) == clean


def test_identity_tracks_commit_after_packed_ref_configure(project):
    src, build = project
    _git(src, "pack-refs", "--all", "--prune")
    ref = _git(src, "symbolic-ref", "HEAD")
    assert not (src / ".git" / ref).exists()
    _configure(src, build)
    _build(build)
    old = _query(build)
    probe = src / "probe.cpp"
    probe.write_text(probe.read_text() + "// new commit\n")
    _git(src, "add", "probe.cpp")
    _git(src, "commit", "-qm", "next")
    current = _git(src, "rev-parse", "HEAD")
    assert current != old
    _build(build)
    assert _query(build) == current


def test_installed_identity_tests_run_after_relocation_and_detect_mismatch(project, tmp_path):
    src, build = project
    cmake = _copy_build_cmake(tmp_path)
    _configure(src, build, cmake=cmake)
    _build(build, cmake)
    # Install after an edit: expectations must come from the last build, not configure.
    probe = src / "probe.cpp"
    probe.write_text(probe.read_text() + "// after configure\n")
    _build(build, cmake)
    _run("ctest", "--test-dir", str(build), "--output-on-failure")
    install = tmp_path / "install"
    _run(cmake, "--install", str(build), "--prefix", str(install), "--component", "tests")
    relocated = tmp_path / "relocated"
    install.rename(relocated)
    shutil.rmtree(src)
    shutil.rmtree(build)
    # The installed suite must use the runner's tools, without the build machine's CMake.
    shutil.rmtree(cmake.parent.parent)
    suite = relocated / "bin/hipblaslt-tensilelite"
    listing = json.loads(_run("ctest", "--test-dir", str(suite), "--show-only=json-v1"))
    assert {test["name"] for test in listing["tests"]} == {
        "TensileLiteClient.version",
        "TensileLiteClient.source-revision",
        "TensileLiteClient.build-rocm",
    }
    _run(
        "ctest",
        "--test-dir",
        str(suite),
        "-L",
        "standard",
        "--output-on-failure",
        "--no-tests=error",
    )
    metadata = suite / "metadata.json"
    values = json.loads(metadata.read_text())
    values["source_revision"] = _SHA
    metadata.write_text(json.dumps(values))
    result = subprocess.run(
        [
            "ctest",
            "--test-dir",
            str(suite),
            "-R",
            "source-revision",
            "--output-on-failure",
            "--no-tests=error",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "mismatch" in result.stdout + result.stderr


def test_source_export_override_survives_rebuild(project):
    src, build = project
    shutil.rmtree(src / ".git")
    _configure(src, build, f"-DTENSILELITE_SOURCE_REVISION={_SHA}")
    _build(build)
    assert _query(build) == _SHA
    assert _query(build, "--version") == "5.0.0+g" + _SHA
    assert _query(build, "--build-rocm") == "10.3.0 base"


def test_canonical_wheel_and_client_stay_paired_through_incremental_builds(project):
    src, build = project
    # Use the real distributable package and production wheel build module.
    stager = _ROOT / "tensilelite/scripts/stage_release_source.py"
    _run(
        sys.executable,
        str(stager),
        "--source",
        str(_ROOT / "tensilelite"),
        "--destination",
        str(src / "tensilelite"),
    )
    shutil.copyfile(_ROOT / "cmake/TensileLiteWheel.cmake", src / "cmake/TensileLiteWheel.cmake")
    cmake = src / "CMakeLists.txt"
    cmake.write_text(cmake.read_text() + """
include(cmake/TensileLiteWheel.cmake)
hipblaslt_add_tensilelite_release_wheel("${CMAKE_CURRENT_SOURCE_DIR}/tensilelite")
""")
    _git(src, "add", ".")
    _git(src, "commit", "-qm", "wheel fixture")
    _configure(src, build)
    manifest = build / "tensilelite-release-wheels/.published"

    def build_and_check():
        _run(
            "cmake",
            "--build",
            str(build),
            "--target",
            "tensilelite-client",
            "tensilelite-canonical-release-wheel",
            "--parallel",
            "2",
        )
        import zipfile

        publication = json.loads(manifest.read_text())
        wheel = manifest.parent / publication["wheel"]
        revision = _query(build)
        assert revision == _current(src)
        assert publication["metadata"]["source_revision"] == revision
        assert publication["metadata"]["distribution_version"] == _query(build, "--version")
        assert wheel.name == "tensilelite-" + _query(build, "--version") + "-py3-none-any.whl"
        assert list(manifest.parent.glob("*.whl")) == [wheel]
        with zipfile.ZipFile(wheel) as archive:
            stamp = archive.read("tensilelite/_build_info.py").decode()
            assignment = next(
                node for node in ast.parse(stamp).body if isinstance(node, ast.Assign)
            )
            assert ast.literal_eval(assignment.value) == revision
        return revision

    clean = build_and_check()
    before = manifest.stat().st_mtime_ns, (build / "tensilelite-client").stat().st_mtime_ns
    build_and_check()
    assert (
        manifest.stat().st_mtime_ns,
        (build / "tensilelite-client").stat().st_mtime_ns,
    ) == before
    source = src / "tensilelite/_tensilelite_client_binding.py"
    original = source.read_text()
    source.write_text(original + "\n# first edit\n")
    first = build_and_check()
    assert first.startswith(clean + ".dirty.")
    source.write_text(original + "\n# second edit\n")
    assert build_and_check() != first
    source.write_text(original)
    assert build_and_check() == clean
    # A commit outside the package must change its stamp and version too.
    (src / "other-component.txt").write_text("new commit outside tensilelite\n")
    _git(src, "add", ".")
    _git(src, "commit", "-qm", "other component")
    assert build_and_check() == _git(src, "rev-parse", "HEAD")
    unpacked = build / "unpacked"
    _run(
        sys.executable,
        str(src / "tensilelite/scripts/install_release_wheels.py"),
        "--manifest",
        str(manifest),
        "--destination",
        str(unpacked),
        "--copy-wheels",
        str(unpacked / "wheels"),
    )
    assert list((unpacked / "wheels").glob("*.whl"))
    assert (unpacked / "_tensilelite_client_binding.py").is_file()
    assert (unpacked / "tensilelite_configure_client.py").is_file()
    assert _git(src, "rev-parse", "HEAD") in (unpacked / "tensilelite/_build_info.py").read_text()
