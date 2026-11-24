# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Exercise the production artifact handoff and build graph without a GPU SDK.

The wheel is real. Only native compilation and the two expensive generation
commands use host fixtures; these tests do not establish GPU/native API support.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import pytest

pytestmark = pytest.mark.unit
_HIPBLASLT = Path(__file__).resolve().parents[5]
_SOURCE = _HIPBLASLT / "tensilelite"
_GENERATOR = _SOURCE / "cmake/TensileLiteGenerator.cmake"
_STAGED_WHEEL = _HIPBLASLT.parent / "hipsparselt/cmake/hipsparselt_tensilelite_wheel.cmake"


def _run(*args, success=True):
    result = subprocess.run(args, capture_output=True, text=True)
    if success:
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0, result.stdout + result.stderr
    return result.stdout + result.stderr


def _configure(source, build, *args, success=True):
    return _run("cmake", "-G", "Ninja", "-S", str(source), "-B", str(build), *args, success=success)


def _build(build, target):
    return _run("cmake", "--build", str(build), "--target", target)


@pytest.fixture(scope="module")
def producer(tmp_path_factory):
    source = tmp_path_factory.mktemp("packaged-generator-producer")
    (source / "native.cpp").write_text("int native() { return 1; }\n")
    (source / "CMakeLists.txt").write_text(f"""cmake_minimum_required(VERSION 3.25)
project(generator_producer LANGUAGES CXX)
include(GNUInstallDirs)
set(hipblaslt_SOURCE_DIR "{_HIPBLASLT.as_posix()}")
set(Python3_EXECUTABLE "{Path(sys.executable).as_posix()}")
set(HIPBLASLT_ENABLE_DEVICE OFF)
set(HIPBLASLT_INSTALL_TENSILELITE_TEST_ARTIFACTS OFF)
set(HIPBLASLT_PREPARE_TENSILELITE_GENERATOR ON)
set(hip_VERSION "7.2.0")
set(HIPBLASLT_BUILD_ROCM_ROOT "/selected-sdk")
include("{_HIPBLASLT.as_posix()}/cmake/TensileLiteMetadata.cmake")
hipblaslt_configure_tensilelite_metadata("{_SOURCE.as_posix()}" "${{CMAKE_BINARY_DIR}}/metadata")
include("{_HIPBLASLT.as_posix()}/cmake/TensileLiteWheel.cmake")
hipblaslt_add_tensilelite_release_wheel("{_SOURCE.as_posix()}")
add_library(_rocisa MODULE native.cpp)
set_target_properties(_rocisa PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${{CMAKE_BINARY_DIR}}/native/rocisa")
include("{_HIPBLASLT.as_posix()}/cmake/hipblaslt_python.cmake")
block()
    hipblaslt_configure_tensilelite_python("")
endblock()
""")
    build = source / "build"
    _configure(source, build, "-DCMAKE_CXX_COMPILER=c++")
    _build(build, "all")
    assert (build / "tensilelite-generator/unpacked.stamp").exists()
    return source, build


def _wheel(producer):
    _, build = producer
    manifest = build / "tensilelite-release-wheels/.published"
    return manifest.parent / json.loads(manifest.read_text())["wheel"]


def _rewrite_wheel(source, destination, replacements):
    with zipfile.ZipFile(source) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    entries.update(replacements)
    with zipfile.ZipFile(destination, "w") as archive:
        for name, content in entries.items():
            if content is not None:
                archive.writestr(name, content)


@pytest.mark.parametrize("selection", ["wheel", "manifest"])
def test_selected_package_tracks_native_wheel_and_logic_changes(tmp_path, producer, selection):
    wheel = tmp_path / _wheel(producer).name
    runs = tmp_path / "runs"
    driver = (
        "import sys, os, tensilelite\nfrom pathlib import Path\n"
        f"assert Path(tensilelite.__file__).is_relative_to(Path({str(tmp_path / 'build')!r}))\n"
        "assert os.environ['ROCM_VERSION'] == '7.2.0'\n"
        f"with Path({str(runs)!r}).open('a') as output:\n"
        "    output.write(sys.argv[1] + '\\n')\n"
    )
    _rewrite_wheel(_wheel(producer), wheel, {"tensilelite/__main__.py": driver})
    manifest = tmp_path / "published.json"
    manifest.write_text(json.dumps({"wheel": wheel.name}))
    metadata = producer[1] / "metadata/metadata.json"
    logic = tmp_path / "logic"
    logic.mkdir()
    (logic / "one.yaml").write_text("initial")
    native = tmp_path / "native.cpp"
    native.write_text("int native() { return 1; }\n")
    wrapper = tmp_path / "rocisa/__init__.py"
    wrapper.parent.mkdir()
    wrapper.write_text("# native wrapper\n")
    argument = f'WHEEL "{wheel}"' if selection == "wheel" else f'WHEEL_MANIFEST "{manifest}"'
    (tmp_path / "CMakeLists.txt").write_text(f"""cmake_minimum_required(VERSION 3.25)
project(generator_consumer LANGUAGES CXX)
add_library(_rocisa MODULE native.cpp)
set_target_properties(_rocisa PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${{CMAKE_BINARY_DIR}}/rocisa")
include("{_GENERATOR.as_posix()}")
block()
    tensilelite_prepare_generator(TARGET selected {argument} METADATA "{metadata}"
        PYTHON_EXECUTABLE "{Path(sys.executable).as_posix()}" ENVIRONMENT "ROCM_VERSION=7.2.0")
endblock()
create_device_library(TARGET generated GENERATOR selected LOGIC_PATH "{logic}"
    OUTPUT_DIR "${{CMAKE_BINARY_DIR}}/out" ARCHES gfx942)
""")
    build = tmp_path / "build"
    _configure(tmp_path, build, "-DCMAKE_CXX_COMPILER=c++")
    _build(build, "generated")
    expected = ["logic", "create-library"]
    assert runs.read_text().splitlines() == expected
    _build(build, "generated")
    assert runs.read_text().splitlines() == expected
    for change in ("native", "native-python", "wheel", "edit", "add", "remove"):
        time.sleep(1.05)
        if change == "native":
            native.write_text("int native() { return 2; }\n")
        elif change == "native-python":
            wrapper.write_text("# changed native wrapper\n")
        elif change == "wheel":
            _rewrite_wheel(wheel, wheel, {"tensilelite/new_resource.py": "CHANGED = True\n"})
            manifest.touch()  # local producer publishes the changed artifact
        elif change == "edit":
            (logic / "one.yaml").write_text("edited")
        elif change == "add":
            added = logic / "two.yaml"
            added.write_text("added")
            os.utime(added, (1, 1))
        else:
            (logic / "two.yaml").unlink()
        _build(build, "generated")
        expected += ["logic", "create-library"]
        assert runs.read_text().splitlines() == expected, change
        _build(build, "generated")
        assert runs.read_text().splitlines() == expected, change


@pytest.mark.parametrize(
    "case, therock_version, expected",
    [
        ("producer identity", None, "identity=7.2.0"),
        ("matching version", "7.2.0", "identity=7.2.0"),
        ("different version", "7.3.0", "does not match the staged hipBLASLt"),
        ("development build", "git", "does not match the staged hipBLASLt"),
        ("different source", None, "does not match the staged hipBLASLt"),
    ],
)
def test_staged_wheel_identity_must_match_the_producer(tmp_path, case, therock_version, expected):
    metadata = json.loads(
        _run(sys.executable, "-B", str(_SOURCE / "release_metadata.py"), "metadata",
             "--rocm-identity", "7.2.0")
    )
    if case == "different source":
        metadata["source_revision"] = "f" * 40
    metadata_file = tmp_path / "metadata.json"
    metadata_file.write_text(json.dumps(metadata))
    (tmp_path / "CMakeLists.txt").write_text(f"""cmake_minimum_required(VERSION 3.25)
project(identity NONE)
set(Python3_EXECUTABLE "{Path(sys.executable).as_posix()}")
include("{_STAGED_WHEEL.as_posix()}")
hipsparselt_add_staged_tensilelite_wheel("{_SOURCE.as_posix()}" "{metadata_file.as_posix()}" identity)
message(STATUS "identity=${{identity}}")
""")
    args = [] if therock_version is None else [f"-DTHEROCK_PACKAGE_VERSION={therock_version}"]
    success = expected.startswith("identity=")
    output = _configure(tmp_path, tmp_path / "build", *args, success=success)
    assert expected in output, case


def test_test_artifact_installs_the_explicit_wheel(tmp_path, producer):
    wheel = tmp_path / _wheel(producer).name
    _rewrite_wheel(_wheel(producer), wheel, {"tensilelite/explicit_marker.py": "MARKER = 1\n"})
    (tmp_path / "CMakeLists.txt").write_text(f"""cmake_minimum_required(VERSION 3.25)
project(test_artifact NONE)
set(Python3_EXECUTABLE "{Path(sys.executable).as_posix()}")
set(TENSILELITE_METADATA_FILE "{(producer[1] / 'metadata/metadata.json').as_posix()}")
set(TENSILELITE_GENERATOR_WHEEL "{wheel.as_posix()}")
include("{_HIPBLASLT.as_posix()}/cmake/TensileLiteWheel.cmake")
hipblaslt_install_tensilelite_test_wheel("{_SOURCE.as_posix()}" share/tests)
""")
    build = tmp_path / "build"
    _configure(tmp_path, build)
    prefix = tmp_path / "prefix"
    _run("cmake", "--install", str(build), "--prefix", str(prefix), "--component", "tests")
    assert (prefix / "share/tests/tensilelite/explicit_marker.py").is_file()
    assert [path.name for path in (prefix / "share/tests/wheels").iterdir()] == [wheel.name]
    assert not (build / "tensilelite-release-wheels").exists()


@pytest.mark.parametrize("failure", ["missing", "unreadable", "name", "version", "revision"])
def test_explicit_artifact_rejection(tmp_path, producer, failure):
    wheel = tmp_path / _wheel(producer).name
    replacements = {}
    metadata = json.loads((producer[1] / "metadata/metadata.json").read_text())
    if failure == "version":
        metadata["distribution_version"] = "0.0.0"
    elif failure == "revision":
        metadata["source_revision"] = "f" * 40
    elif failure == "name":
        with zipfile.ZipFile(_wheel(producer)) as archive:
            entry = next(
                name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
            )
            replacements[entry] = archive.read(entry).replace(b"Name: tensilelite", b"Name: other")
    if failure == "unreadable":
        wheel.write_bytes(b"not a wheel archive")
    elif failure != "missing":
        _rewrite_wheel(_wheel(producer), wheel, replacements)
    metadata_file = tmp_path / "metadata.json"
    metadata_file.write_text(json.dumps(metadata))
    output = _run(
        sys.executable,
        str(_SOURCE / "scripts/prepare_generator.py"),
        "--wheel",
        str(wheel),
        "--metadata",
        str(metadata_file),
        "--check-only",
        success=False,
    )
    assert "Invalid TensileLite generator wheel" in output


def test_archive_without_revision_can_consume_a_stamped_wheel(tmp_path, producer):
    metadata = json.loads((producer[1] / "metadata/metadata.json").read_text())
    metadata["source_revision"] = "none"
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata))
    _run(
        sys.executable,
        str(_SOURCE / "scripts/prepare_generator.py"),
        "--wheel",
        str(_wheel(producer)),
        "--metadata",
        str(path),
        "--check-only",
    )


@pytest.mark.parametrize("selected_native_exists", [True, False])
def test_selected_native_precedes_editable_install_finders(tmp_path, selected_native_exists):
    native = tmp_path / "native"
    native.mkdir()
    if selected_native_exists:
        (native / "rocisa.py").write_text("SELECTED = True\n")
    # Model an installed editable finder ahead of Python's ordinary PathFinder.
    # Falling through to it must never substitute another rocisa for our choice.
    script = f"""
import os, runpy, sys
class Editable:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "rocisa":
            raise AssertionError("ambient editable rocisa was selected")
sys.meta_path.insert(0, Editable())
os.environ["TENSILELITE_GENERATOR_PYTHON_ROOT"] = {str(tmp_path)!r}
os.environ["TENSILELITE_GENERATOR_NATIVE_ROOT"] = {str(native)!r}
runpy.run_path({str(_SOURCE / "scripts/generator-bootstrap/sitecustomize.py")!r})
import rocisa
assert rocisa.SELECTED
"""
    output = _run(sys.executable, "-c", script, success=selected_native_exists)
    if not selected_native_exists:
        assert "Selected generator does not provide rocisa" in output
        assert "ambient editable rocisa was selected" not in output


def test_selection_is_inherited_by_python_workers(tmp_path):
    (tmp_path / "rocisa.py").write_text("SELECTED = True\n")
    environment = dict(os.environ)
    environment.update(
        PYTHONPATH=str(_SOURCE / "scripts/generator-bootstrap"),
        TENSILELITE_GENERATOR_PYTHON_ROOT=str(tmp_path),
        TENSILELITE_GENERATOR_NATIVE_ROOT=str(tmp_path),
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import rocisa, subprocess, sys; assert rocisa.SELECTED; "
            "subprocess.run([sys.executable, '-c', 'import rocisa; assert rocisa.SELECTED'], "
            "check=True)",
        ],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
