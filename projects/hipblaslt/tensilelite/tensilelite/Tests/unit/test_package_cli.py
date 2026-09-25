# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import importlib
import subprocess
import sys

import pytest

import tensilelite
from tensilelite import cli

pytestmark = pytest.mark.unit


def test_help_lists_only_supported_commands(capsys):
    assert cli.main(["--help"]) == 0
    output = capsys.readouterr().out
    for command in cli._COMMAND_HELP:
        assert f"  {command}" in output
    assert "--version" in output
    assert "TensileCreateLibrary" not in output


def test_dispatch_forwards_arguments(monkeypatch):
    seen = []

    def handler(argv):
        seen.extend(argv)
        return 7

    monkeypatch.setattr(cli, "_handler", lambda command: handler)

    assert cli.main(["logic", "path", "--check-all"]) == 7
    assert seen == ["path", "--check-all"]


def test_invalid_command_uses_argparse_error():
    with pytest.raises(SystemExit, match="2"):
        cli.main(["unknown"])


@pytest.mark.parametrize(
    "command,module_name,attribute",
    [
        ("create-library", "tensilelite.tensilelite_create_library.run", "run"),
        ("generate-summations", "tensilelite.GenerateSummations", "main"),
        ("logic", "tensilelite.tensilelite_logic.run", "main"),
        ("run", "tensilelite.tensilelite", "main"),
    ],
)
def test_direct_handler_mapping(command, module_name, attribute):
    expected = getattr(importlib.import_module(module_name), attribute)
    assert cli._handler(command) is expected


@pytest.mark.parametrize(
    "command,module_name",
    [
        ("benchmark-cluster", "tensilelite.benchmark_cluster"),
        ("logic-to-yaml", "tensilelite.lib_logic_to_yaml"),
        ("merge-library", "tensilelite.merge_library"),
        ("retune-library", "tensilelite.retune_library"),
        ("update-library", "tensilelite.update_library"),
    ],
)
def test_sys_argv_handler_mapping_and_restoration(monkeypatch, command, module_name):
    module = importlib.import_module(module_name)
    seen = []

    def entry_point():
        seen.append(list(sys.argv))
        return 3

    monkeypatch.setattr(module, "main", entry_point)
    original_argv = ["outer", "argument"]
    monkeypatch.setattr(sys, "argv", original_argv)

    assert cli._handler(command)(["--flag"]) == 3
    assert seen == [[f"tensilelite {command}", "--flag"]]
    assert sys.argv is original_argv


def test_sys_argv_is_restored_when_a_wrapped_handler_raises(monkeypatch):
    original_argv = ["outer"]
    monkeypatch.setattr(sys, "argv", original_argv)

    def fail():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        cli._invoke_sys_argv_main(fail, "broken", ["--flag"])

    assert sys.argv is original_argv


def test_version_uses_the_package_version(monkeypatch, capsys):
    monkeypatch.setattr(tensilelite, "__version__", "1.2.3+rocm4.5.6")

    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out == "1.2.3+rocm4.5.6\n"


def test_version_rejects_additional_arguments():
    with pytest.raises(SystemExit, match="2"):
        cli.main(["--version", "unexpected"])


def test_generate_summations_help_is_available_without_running_the_command(capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["generate-summations", "--help"])

    assert error.value.code == 0
    assert "logic_path" in capsys.readouterr().out


@pytest.mark.parametrize("command", tuple(cli._COMMAND_HELP))
def test_every_installed_command_has_canonical_help(command):
    result = subprocess.run(
        [sys.executable, "-m", "tensilelite", command, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert f"usage: tensilelite {command}" in result.stdout
