"""Shared pytest configuration and fixtures for the parser test suite.

Adds the parent directory (``shared/ctest``) to ``sys.path`` so the test
modules can ``import parse_test_categories`` and ``import parse_catch2_categories``
without requiring the package to be installed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

THIS_DIR = Path(__file__).resolve().parent
PARSER_DIR = THIS_DIR.parent
FIXTURES_DIR = THIS_DIR / "fixtures"

if str(PARSER_DIR) not in sys.path:
    sys.path.insert(0, str(PARSER_DIR))


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Absolute path to the ``tests/fixtures`` directory."""
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def parser_dir() -> Path:
    """Absolute path to the parser scripts (``shared/ctest``)."""
    return PARSER_DIR


def extract_add_test_blocks(cmake_text: str) -> dict[str, dict]:
    """Parse ``add_test(...)`` / ``set_tests_properties(...)`` pairs from
    parser stdout into a dict keyed by test name.

    The parser emits CMake in a stable, well-known shape (one ``add_test``
    immediately followed by one ``set_tests_properties``), so a small regex
    is enough -- we deliberately avoid pulling in a full CMake AST.

    Returned dict shape per test::

        {
            "<test_name>": {
                "command": "<rest of COMMAND line>",
                "working_directory": "<dir>",
                "labels": ["label1", "label2", ...],
                "timeout": 300,
                "environment": "VAR=val;...",  # or None if absent
            },
            ...
        }
    """
    tests: dict[str, dict] = {}

    add_test_re = re.compile(
        r"add_test\(\s*NAME\s+(?P<name>\S+)\s+"
        r"COMMAND\s+(?P<command>[^\n]+?)\s+"
        r"WORKING_DIRECTORY\s+(?P<wd>\S+)\s*\)",
        re.MULTILINE,
    )
    props_re = re.compile(
        r"set_tests_properties\((?P<name>\S+)\s+PROPERTIES\s*\n"
        r"\s*LABELS\s+\"(?P<labels>[^\"]*)\"\s*\n"
        r"\s*TIMEOUT\s+(?P<timeout>\d+)"
        r"(?:\s*\n\s*ENVIRONMENT\s+\"(?P<env>[^\"]*)\")?"
        r"\s*\n\s*\)",
        re.MULTILINE,
    )

    for m in add_test_re.finditer(cmake_text):
        tests[m.group("name")] = {
            "command": m.group("command").strip(),
            "working_directory": m.group("wd"),
        }

    for m in props_re.finditer(cmake_text):
        name = m.group("name")
        if name not in tests:
            tests[name] = {}
        tests[name]["labels"] = (
            m.group("labels").split(";") if m.group("labels") else []
        )
        tests[name]["timeout"] = int(m.group("timeout"))
        tests[name]["environment"] = m.group("env")

    return tests


def parse_install_file(install_path: Path) -> dict[str, dict]:
    """Parse the install-time CTestTestfile.cmake (relative-path style) into
    the same dict shape as ``extract_add_test_blocks``.

    Install-tree format differs from build-tree CMake -- it uses the single-line
    ``add_test(<name> "../<exe>" <args>...)`` form expected by CTest's script
    interpreter, with no ``NAME``/``COMMAND``/``WORKING_DIRECTORY`` keywords.
    """
    text = install_path.read_text()
    tests: dict[str, dict] = {}

    add_test_re = re.compile(
        r"^add_test\((?P<name>\S+)\s+(?P<rest>.*)\)\s*$",
        re.MULTILINE,
    )
    props_re = re.compile(
        r"^set_tests_properties\((?P<name>\S+)\s+PROPERTIES\s+"
        r"LABELS\s+\"(?P<labels>[^\"]*)\"\s+"
        r"TIMEOUT\s+(?P<timeout>\d+)"
        r"(?:\s+ENVIRONMENT\s+\"(?P<env>[^\"]*)\")?\s*\)\s*$",
        re.MULTILINE,
    )

    for m in add_test_re.finditer(text):
        tests[m.group("name")] = {"command_line": m.group("rest").strip()}

    for m in props_re.finditer(text):
        name = m.group("name")
        if name not in tests:
            tests[name] = {}
        tests[name]["labels"] = (
            m.group("labels").split(";") if m.group("labels") else []
        )
        tests[name]["timeout"] = int(m.group("timeout"))
        tests[name]["environment"] = m.group("env")

    return tests
