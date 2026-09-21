# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Standalone regression test for tasks.py helpers -- run directly with
# `pytest shared/stinkytofu/tests/test_tasks.py`. Not wired into CTest: the
# python_module/tests/ glob in tests/CMakeLists.txt is scoped to tests that
# import the compiled stinkytofu module; this one only exercises tasks.py
# itself and needs no build.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tasks


def test_parse_vcvars_env_parses_key_value_lines():
    stdout = (
        "VSINSTALLDIR=C:\\VS\\2022\\BuildTools\\\nINCLUDE=C:\\Some\\Include\\Path\n"
    )
    env = tasks._parse_vcvars_env(stdout)
    assert env["VSINSTALLDIR"] == "C:\\VS\\2022\\BuildTools\\"
    assert env["INCLUDE"] == "C:\\Some\\Include\\Path"


def test_parse_vcvars_env_tolerates_replacement_characters():
    # Simulates what errors="replace" produces when vcvarsall.bat's banner
    # text contains a byte undecodable under the active code page (e.g. a
    # JIS/Shift-JIS system locale with an English-language VS install) --
    # the exact scenario _setup_msvc_env()'s errors="replace" is meant to
    # survive instead of crashing on.
    stdout = (
        "Copyright �� 2022 banner text\n"
        "� garbage line with no equals sign\n"
        "\n"
        "VSINSTALLDIR=C:\\VS\\2022\\BuildTools\\\n"
        "INCLUDE=C:\\Some\\Include\\Path\n"
    )
    env = tasks._parse_vcvars_env(stdout)
    assert env["VSINSTALLDIR"] == "C:\\VS\\2022\\BuildTools\\"
    assert env["INCLUDE"] == "C:\\Some\\Include\\Path"


def test_parse_vcvars_env_empty_stdout():
    assert tasks._parse_vcvars_env("") == {}
