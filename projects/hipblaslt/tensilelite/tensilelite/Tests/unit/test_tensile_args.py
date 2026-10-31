# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The test harness runs the client at ClientLogLevel 0 unless the user overrides it.

tensileArgs builds the Tensile arguments of every test-suite Tensile call (the
tensile_args fixture). It passes ClientLogLevel=0 before any user-supplied
--global-parameters, so a user-supplied ClientLogLevel wins.
"""

import argparse

import pytest

from Tensile import Tensile
from Tensile.Tests.tensile_args import tensileArgs

pytestmark = pytest.mark.unit


def _clientLogLevel(args):
    """ClientLogLevel that Tensile's command-line parsing gives for ``args``."""
    argParser = argparse.ArgumentParser()
    Tensile.addCommonArguments(argParser)
    return Tensile.argUpdatedGlobalParameters(argParser.parse_args(args)).get("ClientLogLevel")


def test_client_log_level_0_without_options():
    args = tensileArgs({}.get, None)

    assert args == ["--global-parameters", "ClientLogLevel=0"]
    assert _clientLogLevel(args) == 0


def test_all_options():
    options = {
        "--tensile-options": "--cxx-compiler,/usr/bin/amdclang++",
        "--global-parameters": "CheckASMCodeSize=True",
        "--prebuilt-client": "/path/to/client",
        "--gpu-targets": "gfx950",
    }

    assert tensileArgs(options.get, "/tmp/lock") == [
        "--client-lock", "/tmp/lock",
        "--global-parameters", "ClientLogLevel=0",
        "--cxx-compiler", "/usr/bin/amdclang++",
        "--global-parameters", "CheckASMCodeSize=True",
        "--prebuilt-client", "/path/to/client",
        "--gpu-targets", "gfx950",
    ]


def test_no_common_build_drops_prebuilt_client():
    options = {"--no-common-build": True, "--prebuilt-client": "/path/to/client"}

    assert "--prebuilt-client" not in tensileArgs(options.get, None)


@pytest.mark.parametrize("option,value", [
    ("--global-parameters", "ClientLogLevel=3"),
    ("--tensile-options", "--global-parameters,ClientLogLevel=3"),
])
def test_user_client_log_level_comes_after_and_wins(option, value):
    args = tensileArgs({option: value}.get, None)

    assert args.index("ClientLogLevel=0") < args.index("ClientLogLevel=3")
    assert _clientLogLevel(args) == 3
