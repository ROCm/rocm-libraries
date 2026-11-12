# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import configparser
from pathlib import Path
import re

import pytest


pytestmark = pytest.mark.unit

_TENSILELITE_ROOT = Path(__file__).resolve().parents[4]

# `pip install ... -e {toxinidir}` with no sub-directory: this checkout itself.
_EDITABLE_PACKAGE = re.compile(r"-e\s+\{toxinidir\}(?=\s|$)")
_USES_THE_PACKAGE = re.compile(r"\b(invoke|pytest)\b")


def _environments():
    parser = configparser.RawConfigParser(strict=False, interpolation=None)
    parser.read(_TENSILELITE_ROOT / "tox.ini", encoding="utf-8")
    base_skips_install = parser.getboolean(
        "testenv", "skip_install", fallback=False
    )
    for section in parser.sections():
        if not section.startswith("testenv:"):
            continue
        skips_install = parser.getboolean(
            section, "skip_install", fallback=base_skips_install
        )
        commands = [
            line.strip()
            for line in parser.get(section, "commands", fallback="").splitlines()
            if line.strip()
        ]
        yield section.split(":", 1)[1], skips_install, commands


@pytest.mark.parametrize(
    "name, commands",
    [
        (name, commands)
        for name, skips_install, commands in _environments()
        if skips_install and _USES_THE_PACKAGE.search("\n".join(commands))
    ],
)
def test_environments_that_skip_tox_install_install_the_package_first(name, commands):
    # tasks.py, and the tests, import tensilelite, which needs its installed
    # metadata and runtime. An environment that skips tox's own install must
    # install this checkout before the first invoke or pytest command.
    first_use = next(
        index for index, command in enumerate(commands)
        if _USES_THE_PACKAGE.search(command)
    )
    installs = [
        index for index, command in enumerate(commands)
        if _EDITABLE_PACKAGE.search(command)
    ]

    assert installs, f"tox environment {name!r} never installs this checkout"
    assert installs[0] <= first_use, (
        f"tox environment {name!r} runs invoke or pytest before installing this checkout"
    )


def test_the_environments_are_found():
    names = {name for name, skips_install, commands in _environments() if skips_install}
    # Guards the parametrization above against silently selecting nothing.
    assert {"unit", "logic-corpus", "coverage-unit", "coverage-cpp"} <= names
