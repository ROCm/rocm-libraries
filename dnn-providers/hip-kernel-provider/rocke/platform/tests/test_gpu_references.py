# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Configure/install the published reference path without a compiler or GPU."""

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

_PLATFORM = Path(__file__).resolve().parents[1]
_LIBRARY = _PLATFORM.parent / "library"
_CMAKE = shutil.which("cmake")
pytestmark = pytest.mark.skipif(_CMAKE is None, reason="CMake is required")
# Operation packages by the convention CMake and reference_common/artifact.py use.
_OPERATIONS = sorted(
    registry.parents[1].name.removesuffix("_reference")
    for registry in (_LIBRARY / "tests").glob("*_reference/architectures/registry.json")
)


@pytest.fixture
def project(tmp_path):
    library = tmp_path / "library"
    common = library / "tests/reference_common"
    common.mkdir(parents=True)
    shutil.copyfile(
        _LIBRARY / "tests/reference_common/artifact.py", common / "artifact.py"
    )
    cmake = tmp_path / "cmake"
    cmake.mkdir()
    shutil.copyfile(
        _PLATFORM / "cmake/InstallGpuReferences.cmake",
        cmake / "InstallGpuReferences.cmake",
    )
    publish(tmp_path)
    for operation in _OPERATIONS:
        # Installation needs each operation's identity, not its qualification code.
        package = library / f"tests/{operation}_reference"
        (package / "architectures").mkdir(parents=True)
        for name in ("operation.json", "architectures/registry.json"):
            shutil.copyfile(
                _LIBRARY / f"tests/{operation}_reference" / name, package / name
            )
        declared = json.loads((package / "operation.json").read_text())
        bundle = tmp_path / operation
        (bundle / "payload").mkdir(parents=True)
        payload = operation.encode()
        (bundle / "payload/kernel.hsaco").write_bytes(payload)
        manifest = {
            "schema": 2,
            "baseline_revision": "a" * 40,
            "operation": declared["manifest_operation"],
            "files": {"kernel.hsaco": hashlib.sha256(payload).hexdigest()},
        }
        encoded = json.dumps(manifest).encode()
        (bundle / "manifest.json").write_bytes(encoded)
        lock = package / "architectures/gfx942/baseline_lock.json"
        lock.parent.mkdir(parents=True)
        lock.write_text(
            json.dumps(
                {
                    "schema": 2,
                    "baseline_revision": manifest["baseline_revision"],
                    "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
                }
            )
        )
        archive = library / f"tests/reference_bundles/{operation}/gfx942.tar.gz"
        archive.parent.mkdir(parents=True)
        subprocess.run(
            [
                sys.executable,
                "-I",
                str(common / "artifact.py"),
                "pack",
                "--operation",
                operation,
                "--bundle",
                str(bundle),
                "--lock",
                str(lock),
                "--archive",
                str(archive),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    (tmp_path / "CMakeLists.txt").write_text(
        f"""cmake_minimum_required(VERSION 3.25)
project(reference_install NONE)
set(_ROCKE_LIBRARY_DIR "${{CMAKE_CURRENT_SOURCE_DIR}}/library")
set(ROCKE_TEST_INSTALL_DIR "tests")
set(Python3_EXECUTABLE "{Path(sys.executable).as_posix()}")
include("${{CMAKE_CURRENT_SOURCE_DIR}}/cmake/InstallGpuReferences.cmake")
set(installed "")
foreach(operation IN LISTS _ROCKE_REFERENCE_OPERATIONS)
  string(TOUPPER "${{operation}}" upper)
  string(APPEND installed "${{operation}}=${{_ROCKE_${{upper}}_INSTALLED_ARCHITECTURES}}\\n")
endforeach()
file(WRITE "${{CMAKE_CURRENT_BINARY_DIR}}/installed.txt" "${{installed}}")
"""
    )
    return tmp_path


def publish(project, **architectures):
    """Publish gfx942 for every operation unless overridden; return the lists."""
    published = {operation: ["gfx942"] for operation in _OPERATIONS} | architectures
    # An empty set() would unset the list; "" publishes nothing explicitly.
    values = {
        operation: " ".join(names) or '""' for operation, names in published.items()
    }
    (project / "cmake/PublishedGpuReferences.cmake").write_text(
        "".join(
            f"set(ROCKE_PUBLISHED_{operation.upper()}_ARCHITECTURES {value})\n"
            for operation, value in values.items()
        )
    )
    return published


def register(project, operation):
    """Add an operation package that has qualified no architecture yet.

    Tests name it "fictional" so that it cannot collide with a real package.
    """
    package = project / f"library/tests/{operation}_reference"
    (package / "architectures").mkdir(parents=True)
    (package / "architectures/registry.json").write_text('["gfx942"]\n')
    (package / "operation.json").write_text(
        json.dumps({"manifest_operation": f"{operation}-fwd"})
    )


def _architecture_lists(text):
    """Parse `operation=arch;arch` lines into lists, as CMake would split them."""
    lists = (line.split("=", 1) for line in text.splitlines())
    return {
        operation: list(filter(None, names.split(";"))) for operation, names in lists
    }


def installed(project):
    return _architecture_lists((project / "build/installed.txt").read_text())


def _artifact_operations(library):
    """Return the operations the artifact CLI under `library` discovers."""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import runpy, sys; print(*runpy.run_path(sys.argv[1])['OPERATIONS'])",
            str(library / "tests/reference_common/artifact.py"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.split()


def configure(project, *, enabled=True, python=True):
    return subprocess.run(
        [
            _CMAKE,
            "-S",
            str(project),
            "-B",
            str(project / "build"),
            f"-DROCKE_INSTALL_TEST_GPU_REFERENCES={'ON' if enabled else 'OFF'}",
            f"-DPython3_Interpreter_FOUND={'TRUE' if python else 'FALSE'}",
        ],
        capture_output=True,
        text=True,
    )


def test_published_operations_install_separately(project):
    result = configure(project)
    assert result.returncode == 0, result.stdout + result.stderr
    assert installed(project) == {operation: ["gfx942"] for operation in _OPERATIONS}
    prefix = project / "installed"
    subprocess.run(
        [
            _CMAKE,
            "--install",
            str(project / "build"),
            "--prefix",
            str(prefix),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    for operation in _OPERATIONS:
        payload = (
            prefix
            / f"tests/engines/test_arch_content/rocke/{operation}/gfx942/payload/kernel.hsaco"
        )
        assert payload.read_bytes() == operation.encode()


def test_disabled_references_need_neither_python_nor_bundles(project):
    # Operations are still discovered so their host CTests stay registered.
    tests = project / "library/tests"
    shutil.rmtree(tests / "reference_bundles")
    for path in tests.glob("*_reference/architectures/*/baseline_lock.json"):
        path.unlink()
    (tests / "reference_common/artifact.py").unlink()
    result = configure(project, enabled=False, python=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert installed(project) == {operation: [] for operation in _OPERATIONS}


def test_shared_code_payloads_install_under_separate_processor_directories(project):
    """Artifact routing follows the qualification processor, not shared code bytes."""
    tests = project / "library/tests"
    for architecture in ("gfx1200", "gfx1201"):
        lock_dir = tests / f"sdpa_reference/architectures/{architecture}"
        lock_dir.mkdir()
        shutil.copyfile(
            tests / "sdpa_reference/architectures/gfx942/baseline_lock.json",
            lock_dir / "baseline_lock.json",
        )
        shutil.copyfile(
            tests / "reference_bundles/sdpa/gfx942.tar.gz",
            tests / f"reference_bundles/sdpa/{architecture}.tar.gz",
        )
    published = publish(project, sdpa=["gfx1200", "gfx1201"])
    result = configure(project)
    assert result.returncode == 0, result.stdout + result.stderr
    assert installed(project) == published
    prefix = project / "installed"
    subprocess.run(
        [_CMAKE, "--install", str(project / "build"), "--prefix", str(prefix)],
        check=True,
        capture_output=True,
        text=True,
    )
    root = prefix / "tests/engines/test_arch_content/rocke/sdpa"
    assert {p.name for p in root.iterdir()} == {"gfx1200", "gfx1201"}
    for architecture in ("gfx1200", "gfx1201"):
        assert (root / architecture / "payload/kernel.hsaco").read_bytes() == b"sdpa"


@pytest.mark.parametrize("problem", ["archive", "lock", "integrity", "python"])
def test_required_reference_prerequisites_fail_configuration(project, problem):
    library = project / "library/tests"
    lock = library / "conv_reference/architectures/gfx942/baseline_lock.json"
    if problem == "archive":
        (library / "reference_bundles/conv/gfx942.tar.gz").unlink()
    elif problem == "lock":
        lock.unlink()
    elif problem == "integrity":
        data = json.loads(lock.read_text())
        data["manifest_sha256"] = "0" * 64
        lock.write_text(json.dumps(data))
    result = configure(project, python=problem != "python")
    assert result.returncode != 0
    expected = {
        "archive": "reference archive missing",
        "lock": "No qualified conv baseline lock for gfx942",
        "integrity": "does not match the pinned lock",
        "python": "requires a Python3 interpreter",
    }
    output = result.stdout + result.stderr
    assert expected[problem] in output
    if problem == "archive":
        # CMake wraps message text at spaces, so a path with a space spans lines.
        pointer = str(library / "reference_bundles/conv/gfx942.tar.gz.dvc")
        assert " ".join(pointer.split()) in " ".join(output.split())
    if problem == "lock":
        assert "baseline_lock.json" in output
        assert "Traceback" not in output


@pytest.mark.parametrize("missing", ["package", "publication", "metadata"])
def test_half_wired_operation_fails_configuration(project, missing):
    """Each half-wired state fails instead of silently losing coverage."""
    if missing == "package":
        publish(project, fictional=["gfx942"])
        expected = [
            "ROCKE_PUBLISHED_FICTIONAL_ARCHITECTURES names no reference operation",
            "fictional_reference/architectures/registry.json does not exist",
        ]
    elif missing == "publication":
        register(project, "fictional")
        expected = ["Reference operation fictional has no publication list"]
    else:
        register(project, "fictional")
        (project / "library/tests/fictional_reference/operation.json").unlink()
        publish(project, fictional=[])
        expected = [
            "Reference operation fictional has no",
            "fictional_reference/operation.json recording the manifest_operation",
        ]
    result = configure(project)
    assert result.returncode != 0
    # CMake wraps long messages.
    output = " ".join((result.stdout + result.stderr).split())
    assert all(message in output for message in expected), output


def test_registered_operation_may_publish_no_architecture(project):
    register(project, "fictional")
    published = publish(project, fictional=[])
    result = configure(project)
    assert result.returncode == 0, result.stdout + result.stderr
    assert installed(project) == published


def test_operations_extending_a_name_keep_the_artifact_cli_order(project):
    """The glob finds fictional_bwd_reference/ first, but fictional sorts first."""
    for operation in ("fictional", "fictional_bwd"):
        register(project, operation)
    publish(project, fictional=[], fictional_bwd=[])
    result = configure(project)
    assert result.returncode == 0, result.stdout + result.stderr
    # CTests register in CMake's order.
    expected = sorted([*_OPERATIONS, "fictional", "fictional_bwd"])
    assert _artifact_operations(project / "library") == expected
    assert list(installed(project)) == expected


def _cmake_script(tmp_path, body):
    """Run a CMake script and return the value it leaves in `result`."""
    script = tmp_path / "script.cmake"
    output = tmp_path / "result.txt"
    script.write_text(body + f'file(WRITE "{output.as_posix()}" "${{result}}")\n')
    subprocess.run(
        [_CMAKE, "-P", str(script)], check=True, capture_output=True, text=True
    )
    return output.read_text()


def _published(tmp_path):
    """Evaluate every publication list in the real CMake manifest."""
    return _architecture_lists(
        _cmake_script(
            tmp_path,
            f'include("{(_PLATFORM / "cmake/PublishedGpuReferences.cmake").as_posix()}")\n'
            'set(result "")\n'
            "get_cmake_property(variables VARIABLES)\n"
            "foreach(variable IN LISTS variables)\n"
            '  if(variable MATCHES "^ROCKE_PUBLISHED_(.+)_ARCHITECTURES$")\n'
            '    string(TOLOWER "${CMAKE_MATCH_1}" operation)\n'
            '    string(APPEND result "${operation}=${${variable}}\\n")\n'
            "  endif()\n"
            "endforeach()\n",
        )
    )


def test_reference_operations_are_defined_once_by_their_packages(tmp_path):
    """CMake, the artifact CLI and publication all follow the operation packages."""
    discovered = _cmake_script(
        tmp_path,
        f'set(_ROCKE_LIBRARY_DIR "{_LIBRARY.as_posix()}")\n'
        f'include("{(_PLATFORM / "cmake/InstallGpuReferences.cmake").as_posix()}")\n'
        'set(result "${_ROCKE_REFERENCE_OPERATIONS}")\n',
    )
    assert discovered.split(";") == _OPERATIONS
    assert _artifact_operations(_LIBRARY) == _OPERATIONS
    assert sorted(_published(tmp_path)) == _OPERATIONS
    tests = _LIBRARY / "tests"
    owners = {}
    for operation in _OPERATIONS:
        metadata = json.loads(
            (tests / f"{operation}_reference/operation.json").read_text()
        )
        declared = metadata["manifest_operation"]
        assert isinstance(declared, str)
        # An id names one operation's bundles; a copied package keeps the original's.
        assert declared not in owners, (
            f"{operation} and {owners[declared]} both declare manifest_operation "
            f"{declared!r}"
        )
        owners[declared] = operation
        # CTest registration and the workflow address these files by convention.
        for name in (
            f"test_{operation}_reference_contract.py",
            f"test_{operation}_pinned_reference.py",
            f"run_{operation}_reference.py",
        ):
            assert (tests / name).is_file(), f"{operation} has no {name}"


def test_published_bundles_have_supported_cohorts_and_committed_locks(tmp_path):
    """Evaluate the real CMake manifest and check its qualification prerequisites."""
    tests = _LIBRARY / "tests"
    for operation, architectures in _published(tmp_path).items():
        assert len(architectures) == len(set(architectures))
        root = tests / f"{operation}_reference/architectures"
        supported = json.loads((root / "registry.json").read_text())
        assert set(architectures) <= set(supported)
        for architecture in architectures:
            lock = json.loads((root / architecture / "baseline_lock.json").read_text())
            assert lock["schema"] == 2
            assert len(bytes.fromhex(lock["baseline_revision"])) == 20
            assert len(bytes.fromhex(lock["manifest_sha256"])) == 32


def _reference_ctests(published):
    """Expand the CTest names that platform/CMakeLists.txt registers per operation.

    A name is registered once per operation or, when it also names the
    architecture, once per published pair.
    """
    text = (_PLATFORM / "CMakeLists.txt").read_text()
    names = []
    for template in re.findall(r"add_test\((\S*\$\{_operation\}\S*)", text):
        for operation, architectures in published.items():
            name = template.replace("${_operation}", operation)
            if "${_arch}" in name:
                names += [name.replace("${_arch}", arch) for arch in architectures]
            else:
                names.append(name)
    return names


def test_reference_ctests_keep_the_shared_reference_labels(tmp_path):
    """The provider category patterns must keep matching the registered names.

    Every rocke_<operation>_reference_unit_pytest and published
    rocke_<operation>_gpu_<arch>_pytest needs each category label that the
    shared rocke_reference_common_pytest gets. A name that no longer matches
    the patterns keeps only the quick label baked into the installed tests.
    """
    # The provider YAML sits beside rocke/ and the monorepo's shared parser
    # applies it; a standalone rocKE tree has neither.
    provider = _PLATFORM.parent.parent
    categories = provider / "ROCKE_ENGINE_test_categories_external.yaml"
    parser = provider.parent.parent / "shared/ctest/parse_ctest_categories.py"
    for path in (categories, parser):
        if not path.is_file():
            pytest.skip(f"no provider CTest categories to check: {path} does not exist")
    pytest.importorskip("yaml", reason="the shared category parser needs PyYAML")
    published = _published(tmp_path)
    names = _reference_ctests(published)
    # One host suite per operation and one GPU suite per published pair.
    assert len(names) == len(published) + sum(map(len, published.values())), names
    shared = "rocke_reference_common_pytest"
    # Label an installed CTest file the way the provider build does.
    ctests = tmp_path / "CTestTestfile.cmake"
    ctests.write_text(
        "".join(f"add_test({name} python)\n" for name in [shared, *names])
    )
    result = subprocess.run(
        [sys.executable, str(parser), str(categories), str(ctests)],
        check=True,
        capture_output=True,
        text=True,
    )
    labels = {
        name: set(value.split(";"))
        for name, value in re.findall(
            r'set_tests_properties\("([^"]+)" PROPERTIES LABELS "([^"]*)"\)',
            result.stdout,
        )
    }
    assert labels.get(shared), f"{categories.name} no longer labels {shared}"
    for name in names:
        missing = labels[shared] - labels.get(name, set())
        assert not missing, (
            f"{name} lacks the {shared} labels {sorted(missing)}: the patterns in "
            f"{categories.name} no longer match it"
        )
