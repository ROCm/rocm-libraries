# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest

from tensilelite import cli


pytestmark = pytest.mark.unit


def test_help_lists_only_supported_commands(capsys):
    assert cli.main(["--help"]) == 0
    output = capsys.readouterr().out
    assert "create-library" in output
    assert "logic" in output
    assert "run" in output
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
