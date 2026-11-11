# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
from pathlib import Path
import runpy
import subprocess
import sys
import zipfile

import pytest


pytestmark = pytest.mark.unit
_SOURCE_ROOT = Path(__file__).resolve().parents[4]
_INSTALLER = _SOURCE_ROOT / "scripts/install_release_wheels.py"
_STAGER = _SOURCE_ROOT / "scripts/stage_release_source.py"
_VALIDATOR = _SOURCE_ROOT / "scripts/check_release_wheel_contents.py"
_SHA = "0123456789abcdef0123456789abcdef01234567"


def _isolated_source(tmp_path):
    destination = tmp_path / "source"
    runpy.run_path(str(_STAGER))["stage_source"](_SOURCE_ROOT, destination)
    return destination


def _write_minimal_wheel(path, *, name, package_root, scripts, requirements=()):
    dist_info = path.name.removesuffix("-py3-none-any.whl") + ".dist-info"
    metadata = [f"Name: {name}", "Version: 1.0.0+rocm7.2.4", "Requires-Python: >=3.10"]
    metadata.extend(f"Requires-Dist: {requirement}" for requirement in requirements)
    entry_points = ["[console_scripts]"]
    entry_points.extend(f"{key} = {value}" for key, value in scripts.items())
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{package_root}/__init__.py", "")
        archive.writestr(f"{dist_info}/METADATA", "\n".join(metadata) + "\n")
        archive.writestr(f"{dist_info}/WHEEL", "Tag: py3-none-any\n")
        archive.writestr(f"{dist_info}/entry_points.txt", "\n".join(entry_points) + "\n")


def test_release_source_staging_excludes_shared_build_state(tmp_path):
    source = tmp_path / "input"
    destination = tmp_path / "staged"
    ignored_directories = (
        ".agents",
        ".codex",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "_skbuild",
        "dist",
        "htmlcov",
        "mutants",
    )
    (source / "package").mkdir(parents=True)
    (source / "package/module.py").write_text("kept", encoding="utf-8")
    (source / "build/lib").mkdir(parents=True)
    (source / "build/lib/stale.py").write_text("stale", encoding="utf-8")
    (source / "package/__pycache__").mkdir()
    (source / "package/__pycache__/module.pyc").write_bytes(b"stale")
    for ignored_directory in ignored_directories:
        (source / ignored_directory).mkdir()
        (source / ignored_directory / "local-state").write_text("stale", encoding="utf-8")
    (source / ".coverage").write_text("stale", encoding="utf-8")
    (source / "package/_build_info.py").write_text("stale", encoding="utf-8")

    stager = runpy.run_path(str(_STAGER))
    stager["stage_source"](source, destination)

    assert (destination / "package/module.py").read_text(encoding="utf-8") == "kept"
    assert not (destination / "build").exists()
    assert not (destination / "package/__pycache__").exists()
    assert not (destination / ".coverage").exists()
    assert not (destination / "package/_build_info.py").exists()
    for ignored_directory in ignored_directories:
        assert not (destination / ignored_directory).exists()


def _validate_cli(wheel, source_root, *extra):
    return subprocess.run(
        [
            sys.executable,
            str(_VALIDATOR),
            "--wheel",
            str(wheel),
            "--expected-version",
            f"5.0.0+g{_SHA}",
            "--source-root",
            str(source_root),
            *extra,
        ],
        capture_output=True,
        text=True,
    )


def test_release_wheel_records_the_given_revision_and_validates(tmp_path):
    """Build the wheel as hipBLASLt's CMake does: staged source, explicit commit."""
    source_root = _isolated_source(tmp_path)
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in ("ROCM_PATH", "ROCM_VERSION", "TENSILELITE_ROCM_VERSION")
    }
    environment.update(TENSILELITE_SOURCE_REVISION=_SHA, SOURCE_DATE_EPOCH="1700000000")

    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    build = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--disable-pip-version-check",
            "--no-build-isolation",
            "--no-deps",
            "--wheel-dir",
            str(wheel_dir),
            str(source_root),
        ],
        cwd=source_root,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr
    (wheel,) = wheel_dir.glob("tensilelite-*.whl")
    assert wheel.name == f"tensilelite-5.0.0+g{_SHA}-py3-none-any.whl"
    custom_kernel_root = source_root / "tensilelite/CustomKernels"
    assert any(
        path.parent != custom_kernel_root for path in custom_kernel_root.rglob("*.s")
    ), "the release-wheel check must exercise nested custom-kernel resources"

    validation = _validate_cli(wheel, source_root, "--expected-source-revision", _SHA)
    assert validation.returncode == 0, validation.stderr

    other = _validate_cli(wheel, source_root, "--expected-source-revision", "f" * 40)
    assert other.returncode != 0
    assert "source revision must be" in other.stderr
    unrecorded = _validate_cli(wheel, source_root, "--expected-source-revision", "none")
    assert unrecorded.returncode != 0

    installed_root = tmp_path / "installed"
    installation = subprocess.run(
        [sys.executable, str(_INSTALLER), "--destination", str(installed_root), str(wheel)],
        capture_output=True,
        text=True,
    )
    assert installation.returncode == 0, installation.stderr
    expected_module = installed_root / "tensilelite/KernelWriterAssembly.py"
    assert expected_module.is_file()
    import_check = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.util; "
            "spec = importlib.util.find_spec('tensilelite.KernelWriterAssembly'); "
            "print(spec.origin if spec else '')",
        ],
        cwd=tmp_path,
        env=dict(environment, PYTHONPATH=str(installed_root)),
        capture_output=True,
        text=True,
    )
    assert import_check.returncode == 0, import_check.stderr
    assert Path(import_check.stdout.strip()) == expected_module
    assert f"SOURCE_REVISION = {_SHA!r}" in (
        installed_root / "tensilelite/_build_info.py"
    ).read_text(encoding="utf-8")

    with zipfile.ZipFile(wheel, "a") as archive:
        metadata = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        archive.writestr(metadata.rsplit("/", 1)[0] + "/client.json", "/tmp/client")
    bound_validation = _validate_cli(wheel, source_root)
    assert bound_validation.returncode != 0
    assert "wheel must not contain client bindings" in bound_validation.stderr


def test_installer_removes_stale_wheel_owned_files_only(tmp_path):
    destination = tmp_path / "installed"
    (destination / "tensilelite").mkdir(parents=True)
    (destination / "tensilelite/retired.py").write_text("stale", encoding="utf-8")
    (destination / "_tensilelite_client_binding.py").write_text("stale", encoding="utf-8")
    (destination / "tensilelite-0.0.0.dist-info").mkdir()
    (destination / "rocisa").mkdir()
    (destination / "rocisa/keep.py").write_text("keep", encoding="utf-8")

    wheel = tmp_path / "tensilelite-1.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("tensilelite/__init__.py", "current")
        archive.writestr("tensilelite-1.0.0.dist-info/METADATA", "Name: tensilelite\n")

    installer = runpy.run_path(str(_INSTALLER))
    installer["install_wheels"]([wheel], destination)

    assert not (destination / "tensilelite/retired.py").exists()
    assert not (destination / "_tensilelite_client_binding.py").exists()
    assert not (destination / "tensilelite-0.0.0.dist-info").exists()
    assert (destination / "tensilelite/__init__.py").read_text(encoding="utf-8") == "current"
    assert (destination / "rocisa/keep.py").read_text(encoding="utf-8") == "keep"


def _canonical_wheel(tmp_path, *, scripts=None, build_info=f"SOURCE_REVISION = {'0' * 40!r}\n"):
    wheel = tmp_path / "tensilelite-1.0.0+rocm7.2.4-py3-none-any.whl"
    _write_minimal_wheel(
        wheel,
        name="tensilelite",
        package_root="tensilelite",
        scripts=scripts or {
            "tensilelite": "tensilelite.cli:main",
            "tensilelite-configure-client": "tensilelite_configure_client:main",
        },
        requirements=(
            "filelock",
            "joblib",
            "msgpack",
            "numpy",
            "packaging",
            "pandas",
            "pyyaml",
        ),
    )
    if build_info is not None:
        with zipfile.ZipFile(wheel, "a") as archive:
            archive.writestr("tensilelite/_build_info.py", build_info)
    return wheel


def test_validator_rejects_missing_runtime_dependencies(tmp_path):
    validate = runpy.run_path(str(_VALIDATOR))["validate"]

    problems = validate(_canonical_wheel(tmp_path), tmp_path, "1.0.0+rocm7.2.4")

    assert any("runtime dependencies must be exactly" in problem for problem in problems)


def test_validator_rejects_legacy_command_names(tmp_path):
    validate = runpy.run_path(str(_VALIDATOR))["validate"]
    scripts = {
        "tensilelite": "tensilelite.cli:main",
        "tensilelite-configure-client": "tensilelite_configure_client:main",
        "TensileCreateLibrary": "tensilelite.create_library:main",
    }

    problems = validate(_canonical_wheel(tmp_path, scripts=scripts), tmp_path, "1.0.0+rocm7.2.4")

    assert any("unexpected console scripts" in problem for problem in problems)


@pytest.mark.parametrize(
    ("build_info", "expected", "message"),
    [
        (None, None, "missing tensilelite/_build_info.py"),
        ("VALUE = 1\n", None, "does not define SOURCE_REVISION"),
        (f"SOURCE_REVISION = {'0' * 40!r}\n", "1" * 40, "source revision must be"),
        ("SOURCE_REVISION = None\n", "1" * 40, "got none"),
        (f"SOURCE_REVISION = {'0' * 40!r}\n", "none", "source revision must be none"),
    ],
)
def test_validator_checks_the_stamped_revision(tmp_path, build_info, expected, message):
    validate = runpy.run_path(str(_VALIDATOR))["validate"]
    wheel = _canonical_wheel(tmp_path, build_info=build_info)

    problems = validate(wheel, tmp_path, "1.0.0+rocm7.2.4", expected)

    assert any(message in problem for problem in problems), problems


def test_validator_accepts_a_wheel_without_a_revision_when_none_is_expected(tmp_path):
    validate = runpy.run_path(str(_VALIDATOR))["validate"]
    wheel = _canonical_wheel(tmp_path, build_info="SOURCE_REVISION = None\n")

    problems = validate(wheel, tmp_path, "1.0.0+rocm7.2.4", "none")

    assert not any("source revision" in problem for problem in problems), problems
