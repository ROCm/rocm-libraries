"""The supplied interpreter and project preamble of a synthetic CMake consumer.

The tests that drive real sub-configures and sub-builds of `HkpPackaging.cmake`
share these: the rocKE wheel lifecycle in `tests/rocke/`, and the rocKE-free
wiring in `tests/`. A plain module rather than conftest content, for the reason
`synthesised_objects.py` gives.
"""

import os
import subprocess
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
MODULE = PKG / "cmake" / "HkpPackaging.cmake"


def consumer_preamble(project):
    """The head of a consumer's CMakeLists.txt: it includes and calls the
    production module rather than a copy of its commands."""
    return f"""cmake_minimum_required(VERSION 3.25.2)
project({project} NONE)
list(APPEND CMAKE_MODULE_PATH "{(PKG.parent / 'cmake').as_posix()}")
include("{MODULE.as_posix()}")
"""


def write_module(directory, name, value):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.py").write_text(f"VALUE = {value!r}\n", encoding="utf-8")


class SuppliedPython:
    """An isolated venv standing in for the Python3_EXECUTABLE a build is given.

    msgpack and zstandard, which every pack imports, are stub modules on a `.pth`
    path, so the interpreter imports them though pip never installed them. pip
    itself is present only when asked for. Commands run with the calling
    environment's Python, pip and CMake variables stripped, and every install
    stays inside `root`, never the pytest interpreter or its user site.
    """

    def __init__(self, root, *, pip=True, user_site=False):
        self.root = root
        self.parent = root / "supplied python"
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
            # still target this venv or the isolated PYTHONUSERBASE above.
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
        self.runtime = root / "pth runtime only"
        write_module(self.runtime, "msgpack", "parent-msgpack")
        write_module(self.runtime, "zstandard", "parent-zstandard")
        (self.site / "hkp_fixture_runtime.pth").write_text(
            str(self.runtime) + "\n", encoding="utf-8"
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
