"""Exercise the real CMake wheel lifecycle and consumer environment with real pip.

Only the wheels and pack payload are synthetic: CMake owns provisioning, digest
invalidation, ordering, and launching the consumer. All installs belong to
isolated temporary environments, never the pytest interpreter or its user site.
"""

import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.quick
PKG = Path(__file__).resolve().parent.parent
MODULE = PKG / "cmake" / "HkpPackaging.cmake"

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


def _module(directory, name, value):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.py").write_text(f"VALUE = {value!r}\n", encoding="utf-8")


class _Environment:
    def __init__(self, root, *, cmake, make_program, pip=True, user_site=False):
        self.cmake = cmake
        self.make_program = make_program
        self.root = root
        self.source = root / "source with spaces"
        self.build_dir = root / "build with spaces"
        self.wheels = root / "local wheels"
        self.parent = root / "supplied python"
        self.source.mkdir(parents=True)
        self.env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("PYTHON", "PIP_", "CMAKE_"))
            and key not in ("VIRTUAL_ENV", "CONDA_PREFIX")
        }
        self.env.update(
            {
                "PYTHONUSERBASE": str(root / "isolated user base"),
                "PYTHONDONTWRITEBYTECODE": "1",
                "PIP_CONFIG_FILE": os.devnull,
                "PIP_NO_INDEX": "1",
                "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            }
        )
        command = [sys.executable, "-m", "venv", "--copies"]
        if not pip:
            command.append("--without-pip")
        if user_site:
            # Normal venv startup permits user-site only in this mode. All writes
            # still target this venv or the isolated PYTHONUSERBASE below.
            command.append("--system-site-packages")
        self.run(*command, self.parent)
        executable = (
            f"Scripts/{Path(sys._base_executable).name}"
            if os.name == "nt"
            else "bin/python"
        )
        self.python = self.parent / executable
        self.site = Path(
            self.run(
                self.python,
                "-c",
                "import sysconfig; print(sysconfig.get_path('purelib'))",
            ).stdout.strip()
        )
        self.private = self.build_dir / "hkp-rocke-python"
        self.ready = self.private / ".installed"
        self.digest = self.build_dir / "hkp-rocke-wheels.sha256"
        self.runtime = root / "pth runtime only"
        _module(self.runtime, "msgpack", "parent-msgpack")
        _module(self.runtime, "zstandard", "parent-zstandard")
        (self.site / "hkp_fixture_runtime.pth").write_text(
            str(self.runtime) + "\n", encoding="utf-8"
        )
        self.write_wheels()
        (self.source / "consumer.py").write_text(CONSUMER, encoding="utf-8")
        (self.source / "authored").mkdir()
        (self.source / "kpack with spaces" / "rocm_kpack").mkdir(parents=True)
        # Include and call production functions, not a copy of their commands.
        (self.source / "CMakeLists.txt").write_text(
            f"""cmake_minimum_required(VERSION 3.25.2)
project(HkpPythonEnvironment NONE)
list(APPEND CMAKE_MODULE_PATH "{(PKG.parent / 'cmake').as_posix()}")
include("{MODULE.as_posix()}")
set(ROCKE_WHEEL_VERSION 0.1.0)
hkp_rocke_wheel_stamp(wheel_stamp)
hkp_rocke_wheel_python_interp(interp ready python_dir "${{wheel_stamp}}")
set(HKP_TOOL "${{CMAKE_CURRENT_SOURCE_DIR}}/consumer.py")
foreach(label first second)
    hkp_wire_pack_target(
        NAME "${{label}}"
        SOURCE_ROOT "${{CMAKE_CURRENT_SOURCE_DIR}}/authored"
        OUT_ROOT "${{CMAKE_CURRENT_BINARY_DIR}}/${{label}}"
        ARCHES gfx942 HIPCC unused
        ROCM_KPACK_DIR "${{CMAKE_CURRENT_SOURCE_DIR}}/kpack with spaces"
        ROCKE_INTERP "${{interp}}" ROCKE_READY "${{ready}}"
        ROCKE_PYTHON_DIR "${{python_dir}}" ROCKE_WHEEL_STAMP "${{wheel_stamp}}"
        ROCKE_COMGR_LIB "${{HIPKERNELPROVIDER_ROCKE_COMGR_LIB}}" PACK_JOBS 2)
endforeach()
""",
            encoding="utf-8",
        )

    def run(self, *command, success=True):
        proc = subprocess.run(
            [str(arg) for arg in command],
            cwd=self.root,
            env=self.env,
            capture_output=True,
            text=True,
        )
        if success:
            assert proc.returncode == 0, proc.stdout + proc.stderr
        else:
            assert proc.returncode != 0, proc.stdout + proc.stderr
        return proc

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

    def configure(self, *, python=None, success=True):
        # CMAKE_MAKE_PROGRAM is supplied rather than discovered so the Ninja the
        # surrounding build uses drives this sub-build too.
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
            f"-DHIPKERNELPROVIDER_ROCKE_COMGR_LIB={self.root / 'comgr with spaces'}",
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
        _module(user_site, name, "user-site")
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
    _module(first, "ambient", "first")
    _module(second, "ambient", "second")
    _module(second, "second_only", "preserved")
    _module(first, "rocke", "ambient-conflict")
    _module(first, "kernels", "ambient-conflict")
    _module(first, "script_choice", "ambient")
    _module(env.source, "script_choice", "script-directory")
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
