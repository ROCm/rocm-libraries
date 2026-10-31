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

    stager = runpy.run_path(str(_STAGER))
    stager["stage_source"](source, destination)

    assert (destination / "package/module.py").read_text(encoding="utf-8") == "kept"
    assert not (destination / "build").exists()
    assert not (destination / "package/__pycache__").exists()
    assert not (destination / ".coverage").exists()
    for ignored_directory in ignored_directories:
        assert not (destination / ignored_directory).exists()


def test_canonical_and_compatibility_release_wheels_validate_independently(tmp_path):
    source_root = _isolated_source(tmp_path)
    rocm_root = tmp_path / "rocm"
    (rocm_root / ".info").mkdir(parents=True)
    (rocm_root / ".info/version").write_text("7.2.4\n", encoding="utf-8")
    environment = dict(
        os.environ,
        ROCM_PATH=str(rocm_root),
        # PR 8 derives release identity from the selected ROCm root. A later
        # TheRock build-input migration deliberately changes this contract.
        TENSILELITE_ROCM_VERSION="8.0.0",
    )

    for mode, source, pattern in (
        ("canonical", source_root, "tensilelite-*.whl"),
        ("compatibility", source_root / "compat", "tensilelite_tensile_compat-*.whl"),
    ):
        wheel_dir = tmp_path / mode
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
                str(source),
            ],
            cwd=source_root,
            env=environment,
            capture_output=True,
            text=True,
        )
        assert build.returncode == 0, build.stderr
        wheel = next(wheel_dir.glob(pattern))
        if mode == "canonical":
            custom_kernel_root = source_root / "tensilelite/CustomKernels"
            assert any(
                path.parent != custom_kernel_root for path in custom_kernel_root.rglob("*.s")
            ), "the release-wheel check must exercise nested custom-kernel resources"
        validation = subprocess.run(
            [
                sys.executable,
                str(_VALIDATOR),
                "--mode",
                mode,
                "--wheel",
                str(wheel),
                "--expected-version",
                "5.0.0+rocm7.2.4",
                "--source-root",
                str(source_root),
            ],
            capture_output=True,
            text=True,
        )
        assert validation.returncode == 0, validation.stderr

        installed_root = tmp_path / "installed"
        installation = subprocess.run(
            [
                sys.executable,
                str(_INSTALLER),
                "--destination",
                str(installed_root),
                str(wheel),
            ],
            capture_output=True,
            text=True,
        )
        assert installation.returncode == 0, installation.stderr
        expected_module = {
            "canonical": installed_root / "tensilelite/KernelWriterAssembly.py",
            "compatibility": installed_root / "tensilelite_tensile_compat/commands.py",
        }[mode]
        assert expected_module.is_file()
        module_name = {
            "canonical": "tensilelite.KernelWriterAssembly",
            "compatibility": "tensilelite_tensile_compat.commands",
        }[mode]
        import_check = subprocess.run(
            [
                sys.executable,
                "-c",
                "import importlib.util; "
                f"spec = importlib.util.find_spec({module_name!r}); "
                "print(spec.origin if spec else '')",
            ],
            cwd=tmp_path,
            env=dict(environment, PYTHONPATH=str(installed_root)),
            capture_output=True,
            text=True,
        )
        assert import_check.returncode == 0, import_check.stderr
        assert Path(import_check.stdout.strip()) == expected_module

        with zipfile.ZipFile(wheel, "a") as archive:
            metadata = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
            archive.writestr(metadata.rsplit("/", 1)[0] + "/client.json", "/tmp/client")
        bound_validation = subprocess.run(
            [
                sys.executable,
                str(_VALIDATOR),
                "--mode",
                mode,
                "--wheel",
                str(wheel),
                "--expected-version",
                "5.0.0+rocm7.2.4",
                "--source-root",
                str(source_root),
            ],
            capture_output=True,
            text=True,
        )
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


def test_compatibility_sdist_builds_a_self_contained_wheel(tmp_path):
    source_root = _isolated_source(tmp_path)
    rocm_root = tmp_path / "rocm"
    (rocm_root / ".info").mkdir(parents=True)
    (rocm_root / ".info/version").write_text("7.2.4\n", encoding="utf-8")
    environment = dict(os.environ, ROCM_PATH=str(rocm_root))
    sdist_dir = tmp_path / "sdist"
    wheel_dir = tmp_path / "wheel"
    sdist_dir.mkdir()
    wheel_dir.mkdir()

    sdist = subprocess.run(
        [sys.executable, "setup.py", "sdist", "--dist-dir", str(sdist_dir)],
        cwd=source_root / "compat",
        env=environment,
        capture_output=True,
        text=True,
    )
    assert sdist.returncode == 0, sdist.stderr
    archive = next(sdist_dir.glob("*.tar.gz"))

    wheel = subprocess.run(
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
            str(archive),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert wheel.returncode == 0, wheel.stderr
    assert next(wheel_dir.glob("tensilelite_tensile_compat-*.whl")).is_file()


def test_validator_rejects_cross_package_leaks_and_missing_runtime_dependencies(tmp_path):
    validator = runpy.run_path(str(_VALIDATOR))
    validate = validator["validate"]
    canonical_scripts = {
        "tensilelite": "tensilelite.cli:main",
        "tensilelite-configure-client": "tensilelite_configure_client:main",
    }
    canonical = tmp_path / "tensilelite-1.0.0+rocm7.2.4-py3-none-any.whl"
    _write_minimal_wheel(
        canonical,
        name="tensilelite",
        package_root="tensilelite",
        scripts=canonical_scripts,
        requirements=("filelock", "joblib", "msgpack", "numpy", "packaging", "pyyaml"),
    )
    with zipfile.ZipFile(canonical, "a") as archive:
        archive.writestr("tensilelite_tensile_compat/leak.py", "")

    canonical_problems = validate(canonical, "canonical", tmp_path, "1.0.0+rocm7.2.4")
    assert any("compatibility package entries" in problem for problem in canonical_problems)
    assert any("runtime dependencies must be exactly" in problem for problem in canonical_problems)

    compatibility = tmp_path / "tensilelite_tensile_compat-1.0.0+rocm7.2.4-py3-none-any.whl"
    _write_minimal_wheel(
        compatibility,
        name="tensilelite-tensile-compat",
        package_root="tensilelite_tensile_compat",
        scripts=validator["_COMPATIBILITY_SCRIPTS"],
        requirements=("tensilelite==1.0.0+rocm7.2.4",),
    )
    with zipfile.ZipFile(compatibility, "a") as archive:
        archive.writestr("tensilelite/leak.py", "")

    compatibility_problems = validate(
        compatibility, "compatibility", tmp_path, "1.0.0+rocm7.2.4"
    )
    assert any("canonical package entries" in problem for problem in compatibility_problems)
