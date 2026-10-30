# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Fresh Python children import the same rocKE package as their parent."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import rocke
from rocke.tests.subprocess_launcher import (
    rocke_python_command,
    rocke_subprocess_env,
)


def test_child_import_uses_package_under_test_instead_of_cwd_or_pythonpath(
    monkeypatch, tmp_path
):
    decoy = tmp_path / "rocke"
    decoy.mkdir()
    (decoy / "__init__.py").write_text("raise RuntimeError('imported cwd decoy')\n")
    ambient = tmp_path / "ambient"
    (ambient / "rocke").mkdir(parents=True)
    (ambient / "rocke" / "__init__.py").write_text(
        "raise RuntimeError('imported PYTHONPATH decoy')\n"
    )
    (tmp_path / "startup_value.py").write_text("VALUE = 'cwd'\n")
    (ambient / "startup_value.py").write_text("VALUE = 'ambient'\n")
    (ambient / "sitecustomize.py").write_text(
        "import os, startup_value, sys\n"
        f"sys.path.insert(0, {str(ambient)!r})\n"
        "import rocke\n"
        "os.environ['ROCKE_STARTUP_VALUE'] = startup_value.VALUE\n"
        "os.environ['ROCKE_STARTUP_PACKAGE'] = rocke.__file__\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(ambient))

    child = subprocess.run(
        rocke_python_command(
            "-c",
            "import json, os, pathlib, rocke; "
            "print(json.dumps([str(pathlib.Path(rocke.__file__).resolve()), "
            "os.environ['ROCKE_STARTUP_VALUE'], "
            "str(pathlib.Path(os.environ['ROCKE_STARTUP_PACKAGE']).resolve())]))",
        ),
        cwd=tmp_path,
        env=rocke_subprocess_env(),
        check=True,
        capture_output=True,
        text=True,
    )

    package = str(Path(rocke.__file__).resolve())
    assert json.loads(child.stdout) == [package, "cwd", package]


def test_launcher_preserves_code_arguments(tmp_path):
    child = subprocess.run(
        rocke_python_command("-c", "import sys; print('|'.join(sys.argv))", "a", "b"),
        cwd=tmp_path,
        env=rocke_subprocess_env(),
        check=True,
        capture_output=True,
        text=True,
    )

    assert child.stdout.strip() == "-c|a|b"


def test_module_execution_preserves_cwd_imports_and_arguments(tmp_path):
    (tmp_path / "rocke").mkdir()
    (tmp_path / "rocke" / "__init__.py").write_text(
        "raise RuntimeError('imported module decoy')\n"
    )
    (tmp_path / "local_probe.py").write_text(
        "import json, pathlib, rocke, sys\n"
        "print(json.dumps([str(pathlib.Path(rocke.__file__).resolve()), sys.argv[1:]]))\n"
    )
    child = subprocess.run(
        rocke_python_command("-m", "local_probe", "value"),
        cwd=tmp_path,
        env=rocke_subprocess_env(),
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(child.stdout) == [str(Path(rocke.__file__).resolve()), ["value"]]


def test_script_execution_preserves_sibling_imports_and_arguments(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "sibling.py").write_text("VALUE = 'sibling'\n")
    ambient = tmp_path / "ambient"
    ambient.mkdir()
    (ambient / "sibling.py").write_text("VALUE = 'ambient'\n")
    (scripts / "rocke").mkdir()
    (scripts / "rocke" / "__init__.py").write_text(
        "raise RuntimeError('imported script decoy')\n"
    )
    (scripts / "probe.py").write_text(
        "import json, pathlib, rocke, sibling, sys\n"
        "print(json.dumps([str(pathlib.Path(rocke.__file__).resolve()), "
        "sibling.VALUE, sys.argv[1:]]))\n"
    )
    child = subprocess.run(
        rocke_python_command(scripts / "probe.py", "value"),
        cwd=tmp_path,
        env=rocke_subprocess_env({"PYTHONPATH": str(ambient)}),
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(child.stdout) == [
        str(Path(rocke.__file__).resolve()),
        "sibling",
        ["value"],
    ]


def test_explicit_python_path_precedes_cwd_but_keeps_cwd_importable(tmp_path):
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    (explicit / "selected.py").write_text("VALUE = 'explicit'\n")
    (explicit / "rocke").mkdir()
    (explicit / "rocke" / "__init__.py").write_text(
        "raise RuntimeError('imported explicit rocke decoy')\n"
    )
    (tmp_path / "selected.py").write_text("VALUE = 'cwd'\n")
    (tmp_path / "local.py").write_text("VALUE = 'local'\n")
    ambient = tmp_path / "ambient"
    ambient.mkdir()
    (ambient / "selected.py").write_text("VALUE = 'ambient'\n")
    (ambient / "local.py").write_text("VALUE = 'ambient'\n")

    child = subprocess.run(
        rocke_python_command(
            "-c",
            "import local, pathlib, rocke, selected; "
            "print(local.VALUE, selected.VALUE, pathlib.Path(rocke.__file__).resolve())",
        ),
        cwd=tmp_path,
        env=rocke_subprocess_env(
            {"PYTHONPATH": str(ambient)}, python_paths=(explicit,)
        ),
        check=True,
        capture_output=True,
        text=True,
    )

    assert child.stdout.strip() == (f"local explicit {Path(rocke.__file__).resolve()}")


def test_subprocess_environment_preserves_callers_environment(tmp_path):
    base = {"PYTHONPATH": os.pathsep.join(("first", "second")), "MARKER": "kept"}
    explicit = tmp_path / "explicit"

    child_env = rocke_subprocess_env(base, python_paths=(explicit,))

    package_parent = str(Path(rocke.__file__).resolve().parent.parent)
    assert child_env["PYTHONPATH"].split(os.pathsep) == [
        package_parent,
        str(explicit),
        "first",
        "second",
    ]
    assert child_env["MARKER"] == "kept"
    assert base == {
        "PYTHONPATH": os.pathsep.join(("first", "second")),
        "MARKER": "kept",
    }
