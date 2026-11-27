#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Create the build-local rocke-pyenv and check it can build rocke.

It holds the minimum Python build tooling the build checks for: pip, setuptools
64 or newer (the first with PEP 660 editable builds), and, to build a wheel
without pip's isolated build environment, the wheel package unless setuptools
is 70.1 or newer (the first that ships its own bdist_wheel). The pyproject.toml
build requirements state the same setuptools floor, which a test keeps equal to
EDITABLE_SETUPTOOLS.

Usage:
    make_pyenv.py --check                 check the running interpreter
    make_pyenv.py [--offline] <venv-dir>  create the venv, then check it

The venv is created with --system-site-packages, so it reuses the packages the
interpreter running this script was given (a ROCm torch, the build's
setuptools). When that interpreter is itself a virtualenv, --system-site-packages
alone would expose only the base installation under it, not the virtualenv's own
packages; a .pth file in the new venv adds the virtualenv's site directories,
ahead of the base installation's.

Without --offline the venv gets its own pip (ensurepip), then pip, setuptools
and wheel are upgraded from the package index. With --offline nothing is
downloaded and the venv gets no tooling of its own: ensurepip would also install
its bundled setuptools on older Pythons, ahead of the creating interpreter's.
The venv then uses the pip, setuptools and wheel it sees through the site
directories above. Either way the venv's own interpreter then runs the check, so
it holds for the interpreter that runs the installs.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
import sysconfig
import venv
from pathlib import Path
from types import SimpleNamespace

# (major, minor) of the oldest setuptools with PEP 660 editable builds.
EDITABLE_SETUPTOOLS = (64, 0)
# (major, minor) of the oldest setuptools that ships bdist_wheel itself.
BUNDLED_WHEEL_SETUPTOOLS = (70, 1)

# The file in the new venv that adds the creating virtualenv's site directories.
_PARENT_PTH = "_rocke_parent_env.pth"


def _version(text: str) -> tuple[int, int]:
    """The (major, minor) of a version string such as ``"70.1.0"``."""
    match = re.match(r"(\d+)(?:\.(\d+))?", text)
    if match is None:
        raise ValueError(
            f"_version failed - text={text!r}, expected a version such as '70.1.0'"
        )
    return int(match.group(1)), int(match.group(2) or 0)


def missing_build_tools() -> list[str]:
    """What the running interpreter lacks to build rocke offline, as
    requirement strings; empty when it has everything."""
    missing = [] if importlib.util.find_spec("pip") else ["pip"]
    if importlib.util.find_spec("setuptools") is None:
        return [*missing, f"setuptools>={_dotted(EDITABLE_SETUPTOOLS)}"]
    import setuptools

    version = _version(setuptools.__version__)
    if version < EDITABLE_SETUPTOOLS:
        missing.append(
            f"setuptools>={_dotted(EDITABLE_SETUPTOOLS)} "
            f"(found {setuptools.__version__})"
        )
    if version < BUNDLED_WHEEL_SETUPTOOLS and importlib.util.find_spec("wheel") is None:
        missing.append(
            f"wheel (or setuptools>={_dotted(BUNDLED_WHEEL_SETUPTOOLS)}, "
            "which includes it)"
        )
    return missing


def _dotted(version: tuple[int, int]) -> str:
    return f"{version[0]}.{version[1]}"


def check() -> int:
    """Report what the running interpreter lacks; 0 when nothing."""
    missing = missing_build_tools()
    if not missing:
        return 0
    print(
        f"rocke: {sys.executable} cannot build rocke without a package index; it "
        f"needs {' and '.join(missing)}. Install what is missing for the "
        "interpreter CMake uses (Python3_EXECUTABLE), or point Python3_EXECUTABLE "
        "at one that has it.",
        file=sys.stderr,
    )
    return 1


class _PyenvBuilder(venv.EnvBuilder):
    """A venv that also sees the site directories of the virtualenv creating it."""

    _python: str | None = None

    @property
    def python(self) -> str:
        """The new venv's interpreter; set once :meth:`create` has run."""
        if self._python is None:
            raise RuntimeError(
                "_PyenvBuilder.python failed - the venv is not created yet, "
                "expected create() first"
            )
        return self._python

    def post_setup(self, context: SimpleNamespace) -> None:
        self._python = context.env_exe
        site = Path(
            subprocess.run(
                [
                    context.env_exe,
                    "-c",
                    "import sysconfig; print(sysconfig.get_paths()['purelib'])",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        pth = site / _PARENT_PTH
        if sys.prefix == sys.base_prefix:
            pth.unlink(missing_ok=True)
            return
        paths = sysconfig.get_paths()
        parent_sites = dict.fromkeys((paths["purelib"], paths["platlib"]))
        # addsitedir, not a bare path line, so the parent's own .pth files
        # (editable installs, a further parent) are processed as well.
        pth.write_text(
            "".join(
                f"import site; site.addsitedir({directory!r})\n"
                for directory in parent_sites
            )
        )


def create(venv_dir: Path, *, offline: bool) -> int:
    """Create (or refresh) the venv at ``venv_dir`` and check its tooling."""
    builder = _PyenvBuilder(
        system_site_packages=True, symlinks=False, with_pip=not offline
    )
    builder.create(venv_dir)
    python = builder.python
    if not offline:
        subprocess.run(
            [
                python,
                "-m",
                "pip",
                "install",
                "-q",
                "--upgrade",
                "pip",
                f"setuptools>={_dotted(EDITABLE_SETUPTOOLS)}",
                "wheel",
            ],
            check=True,
        )
    return subprocess.run(
        [python, str(Path(__file__).resolve()), "--check"], check=False
    ).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="check the running interpreter's build tooling and exit",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="download nothing: the creating interpreter must provide the tooling",
    )
    parser.add_argument("venv_dir", nargs="?", type=Path)
    args = parser.parse_args()
    if args.check:
        if args.venv_dir is not None or args.offline:
            parser.error("--check takes no other argument")
        return check()
    if args.venv_dir is None:
        parser.error("give the venv directory, or --check")
    return create(args.venv_dir, offline=args.offline)


if __name__ == "__main__":
    sys.exit(main())
