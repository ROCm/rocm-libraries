"""Exercise the real CMake wheel lifecycle and consumer environment with real pip.

Only the wheels and pack payload are synthetic: CMake owns provisioning, digest
invalidation, ordering, and launching the consumer. All installs belong to
isolated temporary environments, never the pytest interpreter or its user site.
"""

import _ctypes
import base64
import hashlib
import json
import os
import shutil
import sysconfig
import time
import zipfile
from pathlib import Path

import pytest
from cmake_harness import PKG, SuppliedPython, consumer_preamble, write_module

pytestmark = pytest.mark.quick

# Executed both outside packaging (parent inventory) and by the pack consumer.
PROBE = """
import importlib
import importlib.metadata
import importlib.util
import json
import os
import site
import sys
from pathlib import Path

names = ('rocke', 'kernels', 'msgpack', 'zstandard', 'ambient', 'second_only',
         'script_choice', 'removed_module')
modules = {}
for name in names:
    spec = importlib.util.find_spec(name)
    if spec is not None:
        module = importlib.import_module(name)
        modules[name] = {'origin': str(Path(module.__file__).resolve()),
                         'value': getattr(module, 'VALUE', None)}
result = {
    'modules': modules,
    'inventory': sorted((d.metadata['Name'], d.version, str(d.locate_file('')))
                        for d in importlib.metadata.distributions()),
    'python': sys.executable,
    'path': sys.path,
    'user_site_enabled': site.ENABLE_USER_SITE,
    'env': {key: os.environ.get(key) for key in
            ('ROCKE_BACKEND', 'ROCKE_CPP_STRICT', 'ROCKE_COMGR_LIB',
             'HKP_PACK_JOBS', 'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE')},
}
"""

CONSUMER = (
    PROBE
    + """
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--out-root', type=Path)
parser.add_argument('--kpack-python-dir')
parser.add_argument('--rocke-wheel-stamp', type=Path)
args, _ = parser.parse_known_args()
# A successful consumer requires all four imports, not just find_spec results.
import rocke, kernels, msgpack, zstandard
result['kpack'] = args.kpack_python_dir
result['digest'] = args.rocke_wheel_stamp.read_text().strip()
args.out_root.mkdir(parents=True, exist_ok=True)
(args.out_root / 'imports.json').write_text(json.dumps(result), encoding='utf-8')
"""
)


def _wheel(directory, distribution, files):
    """Write a valid, deterministic pure-Python wheel, including RECORD hashes."""
    directory.mkdir(parents=True, exist_ok=True)
    info = f"{distribution}-0.1.0.dist-info"
    contents = {name: text.encode() for name, text in files.items()}
    contents[f"{info}/METADATA"] = (
        f"Metadata-Version: 2.1\nName: {distribution}\nVersion: 0.1.0\n"
        # Dropping --no-deps must fail rather than accidentally finding a host dep.
        "Requires-Dist: hkp-unavailable-fixture-dependency\n\n"
    ).encode()
    contents[f"{info}/WHEEL"] = (
        "Wheel-Version: 1.0\nGenerator: hkp-test\nRoot-Is-Purelib: true\n"
        "Tag: py3-none-any\n"
    ).encode()
    records = []
    for name, data in contents.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
        records.append(f"{name},sha256={digest.decode()},{len(data)}\n")
    records.append(f"{info}/RECORD,,\n")
    contents[f"{info}/RECORD"] = "".join(records).encode()
    path = directory / f"{distribution}-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as wheel:
        for name, data in contents.items():
            wheel.writestr(zipfile.ZipInfo(name), data)
    return path


class _Environment(SuppliedPython):
    def __init__(self, root, *, cmake, make_program, pip=True, user_site=False):
        self.cmake = cmake
        self.make_program = make_program
        self.source = root / "source with spaces"
        self.build_dir = root / "build with spaces"
        self.wheels = root / "local wheels"
        self.source.mkdir(parents=True)
        super().__init__(root, pip=pip, user_site=user_site)
        self.private = self.build_dir / "hkp-rocke-python"
        self.ready = self.private / ".installed"
        self.digest = self.build_dir / "hkp-rocke-wheels.sha256"
        self.write_wheels()
        (self.source / "consumer.py").write_text(CONSUMER, encoding="utf-8")
        (self.source / "authored").mkdir()
        (self.source / "kpack with spaces" / "rocm_kpack").mkdir(parents=True)
        (self.source / "CMakeLists.txt").write_text(
            consumer_preamble("HkpPythonEnvironment")
            + """set(ROCKE_WHEEL_VERSION 0.1.0)
hkp_rocke_wheel_stamp(wheel_stamp)
hkp_rocke_wheel_python_interp(interp ready python_dir "${wheel_stamp}")
set(HKP_TOOL "${CMAKE_CURRENT_SOURCE_DIR}/consumer.py")
foreach(label first second)
    hkp_wire_pack_target(
        NAME "${label}"
        SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/authored"
        OUT_ROOT "${CMAKE_CURRENT_BINARY_DIR}/${label}"
        ARCHES gfx942 HIPCC unused
        ROCM_KPACK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kpack with spaces"
        ENABLE_ROCKE ON ROCKE_INTERP "${interp}" ROCKE_READY "${ready}"
        ROCKE_PYTHON_DIR "${python_dir}" ROCKE_WHEEL_STAMP "${wheel_stamp}"
        ROCKE_COMGR_LIB "${HIPKERNELPROVIDER_ROCKE_COMGR_LIB}" PACK_JOBS 2)
endforeach()
""",
            encoding="utf-8",
        )

    def write_wheels(self, value="wheel-one", *, removed=True, invalid=False):
        files = {"rocke/__init__.py": f"VALUE = {value!r}\n"}
        if invalid:
            files["rocke/__init__.py"] = "raise ImportError('invalid fixture wheel')\n"
        if removed:
            files["removed_module.py"] = "VALUE = 'old-only'\n"
        self.platform_wheel = _wheel(self.wheels, "rocke", files)
        self.library_wheel = _wheel(
            self.wheels,
            "rocke_library",
            {"kernels/__init__.py": f"VALUE = {value!r}\n"},
        )

    def configure(self, *, python=None, comgr_lib=None, success=True):
        # CMAKE_MAKE_PROGRAM is supplied rather than discovered so the Ninja the
        # surrounding build uses drives this sub-build too.
        if comgr_lib is None:
            comgr_lib = self.root / "comgr with spaces"
        return self.run(
            self.cmake,
            "-S",
            self.source,
            "-B",
            self.build_dir,
            "-G",
            "Ninja",
            f"-DCMAKE_MAKE_PROGRAM={self.make_program}",
            f"-DPython3_EXECUTABLE={python or self.python}",
            f"-DROCKE_WHEEL_DIR={self.wheels}",
            f"-DHIPKERNELPROVIDER_ROCKE_COMGR_LIB={comgr_lib}",
            success=success,
        )

    def build(self, *, success=True):
        # Both consumers share the actual provisioning producer. Let Ninja select
        # its normal parallelism instead of serializing this regression.
        return self.run(
            self.cmake,
            "--build",
            self.build_dir,
            "--target",
            "hkp_packaging_first",
            "hkp_packaging_second",
            success=success,
        )

    def snapshot(self):
        return json.loads(
            self.run(self.python, "-c", PROBE + "\nprint(json.dumps(result))").stdout
        )

    def consumer(self, name="first"):
        return json.loads(
            (self.build_dir / name / "imports.json").read_text(encoding="utf-8")
        )

    def advance_file_clock(self):
        # Observe a later filesystem tick instead of guessing a sleep duration or
        # aging readiness behind unrelated inputs (which would force a rebuild).
        newest = max(self.ready.stat().st_mtime_ns, self.digest.stat().st_mtime_ns)
        clock = self.root / "filesystem-clock"
        deadline = time.monotonic() + 10
        while True:
            clock.touch()
            if clock.stat().st_mtime_ns > newest:
                return
            assert time.monotonic() < deadline, "filesystem clock did not advance"
            time.sleep(0.01)


def _assert_wheels(environment, value):
    assert environment.ready.is_file()
    for label in ("first", "second"):
        result = environment.consumer(label)
        for name in ("rocke", "kernels"):
            module = result["modules"][name]
            assert module["value"] == value
            assert Path(module["origin"]).is_relative_to(environment.private)
        assert Path(result["python"]) == environment.python
        assert result["digest"] == environment.digest.read_text().strip()


@pytest.fixture
def build_environment(tmp_path, cmake, cmake_make_program):
    """Build an _Environment wired to the CMake and build tool of this build."""

    def factory(**kwargs):
        return _Environment(
            tmp_path, cmake=cmake, make_program=cmake_make_program, **kwargs
        )

    return factory


def _configure_comgr_consumer(environment, *, sdk, package=True):
    """Use the production resolver with isolated metadata and a loadable fixture.

    Copy only runtime modules beneath an empty package initializer so this probe
    needs neither NumPy nor the authoring surface. A Python shared library stands
    in for a loadable COMGR file; no compiler function or GPU action is exercised.
    """
    runtime = environment.source / "rocke/platform/python/rocke/runtime"
    runtime.mkdir(parents=True)
    (runtime.parent / "__init__.py").write_text("", encoding="utf-8")
    (runtime / "__init__.py").write_text("", encoding="utf-8")
    source_runtime = PKG.parent / "rocke/platform/python/rocke/runtime"
    runtime_files = {
        "rocke/__init__.py": "",
        "rocke/runtime/__init__.py": "",
    }
    for name in (
        "comgr.py",
        "_ctypes_bind.py",
        "runtime_coexistence.py",
        "_sdk_comgr.py",
    ):
        shutil.copyfile(source_runtime / name, runtime / name)
        runtime_files[f"rocke/runtime/{name}"] = (source_runtime / name).read_text(
            encoding="utf-8"
        )
    _wheel(environment.wheels, "rocke", runtime_files)
    # Only inspect CTest registration here; no pytest subprocess is executed.
    write_module(environment.runtime, "pytest", "registration-only")
    relative = (
        "_rocm_sdk_core/bin/amd_comgr.dll"
        if os.name == "nt"
        else "_rocm_sdk_core/lib/libamd_comgr.so.3"
    )
    sdk_library = environment.site / relative
    # Standalone Python builds may embed _ctypes rather than ship an extension.
    loadable = getattr(_ctypes, "__file__", None)
    if loadable is None:
        loadable = Path(sysconfig.get_config_var("LIBDIR")) / sysconfig.get_config_var(
            "LDLIBRARY"
        )
    if sdk:
        sdk_library.parent.mkdir(parents=True)
        shutil.copyfile(loadable, sdk_library)
        (sdk_library.parent.parent / "__init__.py").write_text(
            "raise AssertionError('SDK initialization must not run')\n",
            encoding="utf-8",
        )
        metadata = environment.site / "rocm_sdk_core-7.9.0rc1.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: rocm-sdk-core\nVersion: 7.9.0rc1\n",
            encoding="utf-8",
        )
        (metadata / "RECORD").write_text(f"{relative},,\n", encoding="utf-8")
    competing = environment.source / (
        "package-comgr.dll" if os.name == "nt" else "package-comgr.so"
    )
    shutil.copyfile(loadable, competing)
    # A no-package fixture must stay negative even on SDK-provisioned CI hosts.
    package_target = "set(CMAKE_DISABLE_FIND_PACKAGE_amd_comgr TRUE)\n"
    if package:
        package_target = f"""add_library(amd_comgr SHARED IMPORTED)
set_target_properties(amd_comgr PROPERTIES IMPORTED_LOCATION "{competing.as_posix()}")
"""
        if package == "config":
            package_dir = environment.source / "comgr config"
            package_dir.mkdir()
            (package_dir / "amd_comgr-config.cmake").write_text(
                package_target, encoding="utf-8"
            )
            package_target = f'set(amd_comgr_DIR "{package_dir.as_posix()}")\n'
    (environment.source / "consumer.py").write_text(
        CONSUMER
        + """
from rocke.runtime import comgr
result['comgr'] = comgr._resolve_lib()._name
(args.out_root / 'imports.json').write_text(json.dumps(result), encoding='utf-8')
""",
        encoding="utf-8",
    )
    (environment.source / "CMakeLists.txt").write_text(
        consumer_preamble("HkpComgrSelection")
        + """set(ROCKE_WHEEL_VERSION 0.1.0)
set(HKP_PKG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/descriptor-packaging")
"""
        + package_target
        + """
_hkp_resolve_rocke_args(args comgr_lib)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/comgr-selection.txt" "${comgr_lib}\\n${args}\\n")
set(HKP_TOOL "${CMAKE_CURRENT_SOURCE_DIR}/consumer.py")
hkp_wire_pack_target(
    NAME first SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/authored"
    OUT_ROOT "${CMAKE_CURRENT_BINARY_DIR}/first"
    ARCHES gfx942 HIPCC unused
    ROCM_KPACK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kpack with spaces"
    ENABLE_ROCKE ON ${args})
enable_testing()
set(HIPKERNELPROVIDER_ENABLE_TESTS ON)
hkp_register_tests("${CMAKE_CURRENT_SOURCE_DIR}/kpack with spaces" unused "${comgr_lib}")
""",
        encoding="utf-8",
    )
    return loadable


def test_configure_uses_sdk_without_imported_comgr(build_environment):
    env = build_environment()
    _configure_comgr_consumer(env, sdk=True, package=False)
    env.configure(comgr_lib="")
    selection = (env.build_dir / "comgr-selection.txt").read_text(encoding="utf-8")
    assert selection.startswith("\n")
    assert "ROCKE_COMGR_LIB" not in selection
    assert "package-comgr" not in selection


def _assert_ctest_comgr_defaults(env, expected):
    # CTest parses native path separators and CMake escaping before comparison.
    ctest = Path(env.cmake).with_name("ctest" + Path(env.cmake).suffix)
    proc = env.run(ctest, "--test-dir", env.build_dir, "--show-only=json-v1")
    tests = json.loads(proc.stdout)["tests"]
    assert {test["name"] for test in tests} == {
        "hip-kernel-provider-hkp-pack-quick",
        "hip-kernel-provider-hkp-pack",
    }
    for test in tests:
        command = test["command"]
        assert command.count("--default-comgr") == 1
        assert Path(command[command.index("--default-comgr") + 1]) == expected


# Exercise both a missing SDK (TheRock's build interpreter) and a competing SDK:
# the build's exported compiler must be the one configure, packing and CTest use.
@pytest.mark.parametrize("sdk", [False, True])
@pytest.mark.parametrize("package", [True, "config"], ids=["target", "package"])
def test_configure_and_pack_use_imported_comgr(build_environment, sdk, package):
    env = build_environment()
    _configure_comgr_consumer(env, sdk=sdk, package=package)
    env.configure(comgr_lib="")
    expected = next(env.source.glob("package-comgr.*"))
    selection = (env.build_dir / "comgr-selection.txt").read_text(encoding="utf-8")
    assert selection.splitlines()[0] == expected.as_posix()
    assert f"ROCKE_COMGR_LIB;{expected.as_posix()}" in selection
    # The default must stay build-local rather than become a cached user override.
    cache = (env.build_dir / "CMakeCache.txt").read_text(encoding="utf-8")
    assert "HIPKERNELPROVIDER_ROCKE_COMGR_LIB:PATH=\n" in cache
    env.run(env.cmake, "--build", env.build_dir, "--target", "hkp_packaging_first")
    result = env.consumer()
    assert Path(result["comgr"]) == expected
    assert Path(result["env"]["ROCKE_COMGR_LIB"]) == expected
    _assert_ctest_comgr_defaults(env, expected)
    tests = (env.build_dir / "CTestTestfile.cmake").read_text(encoding="utf-8")
    assert "ROCKE_COMGR_LIB=" not in tests


@pytest.mark.parametrize(
    "properties,configuration,selected",
    [
        (
            "IMPORTED_CONFIGURATIONS DEBUG\\;RELEASE "
            "IMPORTED_LOCATION_DEBUG {debug} IMPORTED_LOCATION_RELEASE {release}",
            "Debug",
            "debug",
        ),
        (
            "IMPORTED_CONFIGURATIONS DEBUG\\;RELEASE "
            "IMPORTED_LOCATION_DEBUG {missing} IMPORTED_LOCATION_RELEASE {release}",
            "Debug",
            "release",
        ),
        (
            "IMPORTED_CONFIGURATIONS DEBUG\\;RELEASE "
            "IMPORTED_LOCATION_DEBUG {debug} IMPORTED_LOCATION_RELEASE {release}",
            "",
            "release",
        ),
        ("IMPORTED_LOCATION {release}", "", "release"),
        ("IMPORTED_IMPLIB {release}", "", None),
        (
            "IMPORTED_CONFIGURATIONS CUSTOM IMPORTED_LOCATION_CUSTOM {debug}",
            "",
            "debug",
        ),
    ],
    ids=[
        "configured",
        "missing-config",
        "multi-config",
        "unconfigured",
        "implib-is-not-loadable",
        "custom-config",
    ],
)
def test_imported_comgr_loadable_location(
    build_environment, properties, configuration, selected
):
    env = build_environment()
    paths = {name: env.source / name for name in ("debug", "release", "missing")}
    for name in ("debug", "release"):
        paths[name].touch()
    properties = properties.format(
        **{name: f'"{path.as_posix()}"' for name, path in paths.items()}
    )
    (env.source / "CMakeLists.txt").write_text(
        consumer_preamble("HkpImportedComgr")
        + f"""set(CMAKE_BUILD_TYPE "{configuration}")
add_library(amd_comgr SHARED IMPORTED)
set_target_properties(amd_comgr PROPERTIES {properties})
hkp_imported_library_location(amd_comgr selection)
file(WRITE "${{CMAKE_CURRENT_BINARY_DIR}}/selection.txt" "${{selection}}")
""",
        encoding="utf-8",
    )
    env.configure(comgr_lib="")
    result = (env.build_dir / "selection.txt").read_text(encoding="utf-8")
    assert result == (paths[selected].as_posix() if selected else "")


def test_configure_rejects_missing_comgr(build_environment):
    env = build_environment()
    _configure_comgr_consumer(env, sdk=False, package=False)
    proc = env.configure(comgr_lib="", success=False)
    diagnostic = " ".join((proc.stdout + proc.stderr).split())
    assert "no candidate path" in diagnostic
    assert "rocke[sdk-core]" in diagnostic


@pytest.mark.parametrize("loadable", [True, False])
@pytest.mark.parametrize("sdk", [False, True])
def test_configure_explicit_comgr_is_sole_candidate(build_environment, loadable, sdk):
    env = build_environment()
    fixture_library = _configure_comgr_consumer(env, sdk=sdk)
    explicit = env.root / ("explicit.dll" if os.name == "nt" else "explicit.so")
    if loadable:
        shutil.copyfile(fixture_library, explicit)
    proc = env.configure(comgr_lib=explicit, success=loadable)
    if loadable:
        selection = (env.build_dir / "comgr-selection.txt").read_text(encoding="utf-8")
        assert Path(selection.splitlines()[0]) == explicit
        assert Path(selection.split("ROCKE_COMGR_LIB;", 1)[1].strip()) == explicit
    else:
        diagnostic = " ".join((proc.stdout + proc.stderr).split())
        assert explicit.as_posix() in diagnostic
        assert "1 candidate(s) failed" in diagnostic


@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("alias", [False, True])
def test_configure_and_pack_preserve_runtime_comgr_override(
    build_environment, relative, alias
):
    env = build_environment()
    fixture_library = _configure_comgr_consumer(env, sdk=True)
    explicit = (
        env.root
        / "env compiler with spaces"
        / ("env-override.dll" if os.name == "nt" else "env-override.so")
    )
    explicit.parent.mkdir()
    shutil.copyfile(fixture_library, explicit)
    if alias:
        link = explicit.with_name("compiler alias" + explicit.suffix)
        try:
            link.symlink_to(explicit)
        except OSError as exc:
            pytest.skip(f"compiler alias fixture requires symlink support: {exc}")
        explicit = link
    env.env["ROCKE_COMGR_LIB"] = (
        "./" + explicit.relative_to(env.root).as_posix() if relative else str(explicit)
    )
    env.configure(comgr_lib="")
    selection = (env.build_dir / "comgr-selection.txt").read_text(encoding="utf-8")
    cache = (env.build_dir / "CMakeCache.txt").read_text(encoding="utf-8")
    assert "HIPKERNELPROVIDER_ROCKE_COMGR_LIB:PATH=\n" in cache
    # The build must use the configure-time selection even without that override
    # in its ambient environment, and with a different working directory.
    env.env.pop("ROCKE_COMGR_LIB")
    env.run(env.cmake, "--build", env.build_dir, "--target", "hkp_packaging_first")
    result = env.consumer()
    assert Path(selection.splitlines()[0]) == explicit
    assert Path(selection.split("ROCKE_COMGR_LIB;", 1)[1].strip()) == explicit
    assert Path(result["env"]["ROCKE_COMGR_LIB"]) == explicit
    assert Path(result["comgr"]) == explicit
    _assert_ctest_comgr_defaults(env, explicit)


def test_pth_runtime_and_parent_inventory_preserved(tmp_path, build_environment):
    env = build_environment()
    parent_wheels = tmp_path / "parent wheels"
    parent_rocke = _wheel(
        parent_wheels, "rocke", {"rocke/__init__.py": "VALUE = 'parent'\n"}
    )
    parent_kernels = _wheel(
        parent_wheels, "rocke_library", {"kernels/__init__.py": "VALUE = 'parent'\n"}
    )
    env.run(
        env.python,
        "-m",
        "pip",
        "--disable-pip-version-check",
        "install",
        "--no-index",
        "--no-deps",
        parent_rocke,
        parent_kernels,
    )
    before = env.snapshot()
    assert before["modules"]["rocke"]["value"] == "parent"
    for name in ("msgpack", "zstandard"):
        assert Path(before["modules"][name]["origin"]).parent == env.runtime
    env.configure()
    env.build()
    _assert_wheels(env, "wheel-one")
    for name in ("msgpack", "zstandard"):
        assert env.consumer()["modules"][name] == before["modules"][name]
    assert env.snapshot() == before


def test_enabled_user_site_runtime_preserved(tmp_path, build_environment):
    env = build_environment(user_site=True)
    (env.site / "hkp_fixture_runtime.pth").unlink()
    user_site = Path(
        env.run(
            env.python, "-c", "import site; print(site.getusersitepackages())"
        ).stdout.strip()
    )
    assert user_site.is_relative_to(tmp_path)
    for name in ("msgpack", "zstandard"):
        write_module(user_site, name, "user-site")
    before = env.snapshot()
    assert before["user_site_enabled"] is True
    for name in ("msgpack", "zstandard"):
        assert Path(before["modules"][name]["origin"]).parent == user_site
    env.configure()
    env.build()
    _assert_wheels(env, "wheel-one")
    for name in ("msgpack", "zstandard"):
        assert env.consumer()["modules"][name] == before["modules"][name]
    assert env.snapshot() == before


def test_execution_time_pythonpath_and_startup_precedence(tmp_path, build_environment):
    env = build_environment()
    first, second = tmp_path / "ambient first", tmp_path / "ambient second"
    write_module(first, "ambient", "first")
    write_module(second, "ambient", "second")
    write_module(second, "second_only", "preserved")
    write_module(first, "rocke", "ambient-conflict")
    write_module(first, "kernels", "ambient-conflict")
    write_module(first, "script_choice", "ambient")
    write_module(env.source, "script_choice", "script-directory")
    env.configure()
    # Change after configure: the production command must preserve execution-time
    # PYTHONPATH, not reconstruct a configure-time snapshot of the environment.
    env.env["PYTHONPATH"] = os.pathsep.join(map(str, (first, second)))
    env.build()
    _assert_wheels(env, "wheel-one")
    result = env.consumer()
    assert result["modules"]["ambient"]["value"] == "first"
    assert result["modules"]["second_only"]["value"] == "preserved"
    assert result["modules"]["script_choice"]["value"] == "script-directory"
    assert list(map(Path, result["env"]["PYTHONPATH"].split(os.pathsep))) == [
        env.private,
        first,
        second,
    ]
    assert result["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert result["env"]["ROCKE_BACKEND"] == "python"
    assert result["env"]["ROCKE_CPP_STRICT"] == "1"
    assert result["env"]["HKP_PACK_JOBS"] == "2"
    assert Path(result["env"]["ROCKE_COMGR_LIB"]) == tmp_path / "comgr with spaces"
    assert Path(result["kpack"]) == env.source / "kpack with spaces"


@pytest.mark.parametrize("missing", ["interpreter", "pip", "msgpack", "zstandard"])
def test_missing_supplied_prerequisite_fails_early(
    tmp_path, missing, build_environment
):
    env = build_environment(pip=missing != "pip")
    python = env.python
    if missing == "interpreter":
        python = tmp_path / "absent interpreter"
    elif missing in ("msgpack", "zstandard"):
        (env.runtime / f"{missing}.py").unlink()
    proc = env.configure(python=python, success=False)
    diagnostic = " ".join((proc.stdout + proc.stderr).split())
    assert missing in diagnostic
    assert str(python) in diagnostic
    # Assert actionable prerequisites, not full diagnostic prose.
    assert "Python3_EXECUTABLE" in diagnostic or "install" in diagnostic
    assert not env.ready.exists()
    assert not (env.build_dir / "first" / "imports.json").exists()


def test_same_version_refresh_removes_old_modules(build_environment):
    env = build_environment()
    env.configure()
    env.build()
    assert env.consumer()["modules"]["removed_module"]["value"] == "old-only"
    old_digest = env.digest.read_text()
    env.advance_file_clock()
    before = env.ready.stat().st_mtime_ns
    env.write_wheels("wheel-two", removed=False)
    env.build()
    _assert_wheels(env, "wheel-two")
    assert "removed_module" not in env.consumer()["modules"]
    assert env.digest.read_text() != old_digest
    assert env.ready.stat().st_mtime_ns != before


def test_identical_wheels_do_not_reinstall(build_environment):
    env = build_environment()
    env.configure()
    env.build()
    marker = env.private / "installation-must-survive"
    marker.write_text("untouched", encoding="utf-8")
    watched = (env.digest, env.ready, env.private / "rocke" / "__init__.py")
    before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in watched]
    env.advance_file_clock()
    for wheel in (env.platform_wheel, env.library_wheel):
        wheel.write_bytes(wheel.read_bytes())
    env.build()
    assert [(path.read_bytes(), path.stat().st_mtime_ns) for path in watched] == before
    assert marker.read_text() == "untouched"
    _assert_wheels(env, "wheel-one")


def test_deleted_private_directory_recovers(build_environment):
    env = build_environment()
    env.configure()
    env.build()
    sibling = env.build_dir / "unrelated-build-output"
    sibling.write_text("keep", encoding="utf-8")
    shutil.rmtree(env.private)
    assert not env.ready.exists()
    env.build()
    _assert_wheels(env, "wheel-one")
    assert sibling.read_text() == "keep"


@pytest.mark.parametrize("failure", ["corrupt-wheel", "invalid-import"])
def test_failed_refresh_clears_readiness_and_recovers(failure, build_environment):
    env = build_environment()
    env.configure()
    env.build()
    outputs = [env.build_dir / label / "imports.json" for label in ("first", "second")]
    before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in outputs]
    env.advance_file_clock()
    if failure == "corrupt-wheel":
        env.platform_wheel.write_bytes(b"not a wheel archive")
    else:
        env.write_wheels("broken", invalid=True)
    env.build(success=False)
    assert not env.ready.exists()
    # Neither consumer may run after failed provisioning.
    assert [(path.read_bytes(), path.stat().st_mtime_ns) for path in outputs] == before
    env.write_wheels("repaired", removed=False)
    env.build()
    _assert_wheels(env, "repaired")
    assert "removed_module" not in env.consumer()["modules"]


@pytest.mark.parametrize(
    "override", ["unset", "empty", "valid", "missing", "unloadable"]
)
def test_ctest_comgr_default_preserves_runtime_override(build_environment, override):
    env = build_environment()
    fixture_library = _configure_comgr_consumer(env, sdk=True)
    default = next(env.source.glob("package-comgr.*"))
    alternate = env.root / ("caller.dll" if os.name == "nt" else "caller.so")
    if override == "valid":
        shutil.copyfile(fixture_library, alternate)
    elif override == "unloadable":
        alternate.write_text("not a shared library", encoding="utf-8")
    # Execute a small real resolver probe through the production CTest command.
    # The fixture's pytest module needs no test dependencies or GPU toolchain.
    (env.runtime / "pytest.py").write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(env.source / 'rocke/platform/python')!r})\n"
        "from rocke.runtime import comgr\n"
        "print('COMGR_SELECTED=' + comgr._resolve_lib()._name)\n",
        encoding="utf-8",
    )
    env.configure(comgr_lib="")
    if override != "unset":
        env.env["ROCKE_COMGR_LIB"] = "" if override == "empty" else str(alternate)
    ctest = Path(env.cmake).with_name("ctest" + Path(env.cmake).suffix)
    proc = env.run(
        ctest,
        "--test-dir",
        env.build_dir,
        "-R",
        "hkp-pack-quick$",
        "-V",
        success=override not in ("missing", "unloadable"),
    )
    if override in ("missing", "unloadable"):
        assert "1 candidate(s) failed" in proc.stdout + proc.stderr
    else:
        expected = alternate if override == "valid" else default
        selected = next(
            line.split("COMGR_SELECTED=", 1)[1]
            for line in proc.stdout.splitlines()
            if "COMGR_SELECTED=" in line
        )
        assert Path(selected) == expected
