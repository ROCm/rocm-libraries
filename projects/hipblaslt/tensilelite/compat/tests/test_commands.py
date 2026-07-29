# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import sys

from tensilelite_tensile_compat import commands


def test_warning_is_emitted_once(capsys):
    commands._warned = False
    commands._warn("Tensile")
    commands._warn("Tensile")

    output = capsys.readouterr().err
    assert output.count("DEPRECATED") == 1
    assert "ROCm 9.0" in output


def test_canonical_dispatch_preserves_arguments(monkeypatch):
    from tensilelite import cli

    seen = []
    commands._warned = True
    monkeypatch.setattr(sys, "argv", ["TensileLogic", "logic.yaml", "--check-all"])
    monkeypatch.setattr(cli, "main", lambda argv: seen.extend(argv) or 3)

    assert commands.logic() == 3
    assert seen == ["logic", "logic.yaml", "--check-all"]
