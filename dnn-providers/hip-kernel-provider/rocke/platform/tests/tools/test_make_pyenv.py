# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The rocke-pyenv helper checks the build tooling in the env that uses it.

``cmake/make_pyenv.py`` creates the build-local venv and holds the minimum
pip, setuptools and wheel the build needs offline. Each case runs the helper
with a real interpreter: a virtualenv whose ``pip``, ``setuptools`` and
``wheel`` are stand-ins (a ``setuptools`` that only carries a version), so a
case controls exactly what the helper sees.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import venv
from pathlib import Path

import pytest
from rocke.tests.lanes import SOURCE_TREE, skip_until

# Creating a venv takes well under a second; the bound is for a slow host.
_TIMEOUT_S = 60
# Where a venv keeps its interpreter.
_SCRIPTS = "Scripts" if os.name == "nt" else "bin"
# The source checkout's rocke/ directory, which holds cmake/ and library/.
_ROCKE = Path(__file__).resolve().parents[3]


def _make_pyenv_script() -> Path:
    """The helper in a source checkout; skip in a platform-only tree."""
    script = _ROCKE / "cmake" / "make_pyenv.py"
    if not script.is_file():
        skip_until(
            SOURCE_TREE,
            reason="cmake/make_pyenv.py is absent from this platform-only tree",
        )
    return script


def _python(env: Path) -> Path:
    return env / _SCRIPTS / ("python.exe" if os.name == "nt" else "python")


def _site_packages(python: Path) -> Path:
    return Path(
        subprocess.run(
            [
                str(python),
                "-c",
                "import sysconfig; print(sysconfig.get_paths()['purelib'])",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
        ).stdout.strip()
    )


def _env_with(
    tmp_path: Path,
    name: str,
    setuptools_version: str | None,
    has_wheel: bool,
    has_pip: bool = True,
) -> Path:
    """A virtualenv, without the base installation's packages, holding a
    stand-in ``setuptools`` of ``setuptools_version`` (none when None), and
    stand-in ``wheel`` and ``pip`` packages when asked; returns its interpreter."""
    env = tmp_path / name
    # Linked, as the venv module links by default off Windows: some relocatable
    # interpreter builds cannot run from a copied binary.
    venv.create(env, with_pip=False, symlinks=os.name != "nt")
    python = _python(env)
    site = _site_packages(python)
    packages = {
        "setuptools": (
            None
            if setuptools_version is None
            else f"__version__ = {setuptools_version!r}\n"
        ),
        "wheel": "" if has_wheel else None,
        "pip": "" if has_pip else None,
    }
    for package, init in packages.items():
        if init is not None:
            (site / package).mkdir()
            (site / package / "__init__.py").write_text(init)
    return python


def _run_check(python: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(python), str(_make_pyenv_script()), "--check"],
        capture_output=True,
        text=True,
        check=False,
        timeout=_TIMEOUT_S,
    )


def _create(parent: Path, child: Path) -> subprocess.CompletedProcess[str]:
    """Run the helper with ``parent`` to create the venv ``child`` offline."""
    return subprocess.run(
        [str(parent), str(_make_pyenv_script()), "--offline", str(child)],
        capture_output=True,
        text=True,
        check=False,
        timeout=_TIMEOUT_S,
    )


@pytest.mark.parametrize(
    "setuptools_version, has_wheel, has_pip, missing",
    [
        (None, False, True, "setuptools>=64.0"),
        ("63.4.3", True, True, "setuptools>=64.0 (found 63.4.3)"),
        ("69.5.1", False, True, "wheel (or setuptools>=70.1, which includes it)"),
        ("70.1.0", False, False, "pip"),
    ],
    ids=["no-setuptools", "no-editable-builds", "no-bdist-wheel", "no-pip"],
)
def test_the_check_names_the_missing_tooling(
    tmp_path, setuptools_version, has_wheel, has_pip, missing
):
    python = _env_with(tmp_path, "env", setuptools_version, has_wheel, has_pip)
    result = _run_check(python)
    assert result.returncode == 1
    assert result.stderr == (
        f"rocke: {python} cannot build rocke without a package index; it needs "
        f"{missing}. Install what is missing for the interpreter CMake uses "
        "(Python3_EXECUTABLE), or point Python3_EXECUTABLE at one that has it.\n"
    )


@pytest.mark.parametrize(
    "setuptools_version, has_wheel",
    [("64.0.0", True), ("70.1.0", False)],
    ids=["wheel-package", "bundled-bdist-wheel"],
)
def test_the_check_accepts_enough_tooling(tmp_path, setuptools_version, has_wheel):
    result = _run_check(_env_with(tmp_path, "env", setuptools_version, has_wheel))
    assert (result.returncode, result.stderr) == (0, "")


def test_the_venv_sees_the_creating_virtualenvs_packages_first(tmp_path):
    """--system-site-packages alone would show the base installation under the
    creating virtualenv, not the virtualenv itself; an offline venv also gets
    no setuptools of its own that could hide the virtualenv's."""
    parent = _env_with(tmp_path, "parent", "70.1.0", has_wheel=False)
    child = tmp_path / "child"
    created = _create(parent, child)
    assert created.returncode == 0, created.stderr
    seen = subprocess.run(
        [
            str(_python(child)),
            "-c",
            "import setuptools; print(setuptools.__version__, setuptools.__file__)",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_S,
    ).stdout.split()
    assert seen == [
        "70.1.0",
        str(_site_packages(parent) / "setuptools" / "__init__.py"),
    ]


def test_the_check_after_creation_runs_with_the_venvs_interpreter(tmp_path):
    """The refusal names the new venv's interpreter, not the creating one."""
    parent = _env_with(tmp_path, "parent", "63.4.3", has_wheel=True)
    created = _create(parent, tmp_path / "child")
    assert created.returncode == 1
    reported = created.stderr.removeprefix("rocke: ").split(" cannot build")[0]
    assert Path(reported).parent == tmp_path / "child" / _SCRIPTS
    assert "it needs setuptools>=64.0 (found 63.4.3)." in created.stderr


@pytest.mark.parametrize("pyproject", ["platform", "library"])
def test_the_pyproject_setuptools_floor_is_the_checked_one(pyproject):
    """Each package's build requirement asks for the setuptools the build
    checks for, so an isolated build and the offline build agree."""
    spec = importlib.util.spec_from_file_location("make_pyenv", _make_pyenv_script())
    make_pyenv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(make_pyenv)
    text = (_ROCKE / pyproject / "pyproject.toml").read_text()
    build_system = text.split("[build-system]", 1)[1].split("\n[", 1)[0]
    floors = re.findall(r'"setuptools>=(\d+)(?:\.(\d+))?"', build_system)
    assert [(int(major), int(minor or 0)) for major, minor in floors] == [
        make_pyenv.EDITABLE_SETUPTOOLS
    ]
