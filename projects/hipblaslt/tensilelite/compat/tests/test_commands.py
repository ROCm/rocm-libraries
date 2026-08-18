# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import sys
import types
from pathlib import Path

import pytest

from tensilelite_tensile_compat import commands


pytestmark = pytest.mark.compat


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


@pytest.mark.parametrize(
    ("function", "subcommand"),
    [
        (commands.tensile, "run"),
        (commands.create_library, "create-library"),
        (commands.logic, "logic"),
    ],
)
def test_canonical_wrappers_delegate_exact_subcommand(monkeypatch, function, subcommand):
    seen = []
    monkeypatch.setattr(commands, "_canonical", lambda value: seen.append(value) or 7)
    assert function() == 7
    assert seen == [subcommand]


@pytest.mark.parametrize(
    ("function", "module_name", "passes_argv"),
    [
        (commands.benchmark_cluster, "tensilelite.benchmark_cluster", False),
        (commands.logic_to_yaml, "tensilelite.lib_logic_to_yaml", False),
        (commands.merge_library, "tensilelite.merge_library", False),
        (commands.retune_library, "tensilelite.retune_library", False),
        (commands.update_library, "tensilelite.update_library", False),
        (commands.verify_stinky_elf, "tensilelite.verify_stinky_comment_vs_elf_text", True),
    ],
)
def test_legacy_wrappers_preserve_arguments_and_return_code(
    monkeypatch, function, module_name, passes_argv
):
    seen = []
    module = types.ModuleType(module_name)
    module.main = lambda *args: seen.append(args) or 9
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setattr(sys, "argv", [function.__name__, "one", "two"])
    commands._warned = True

    assert function() == 9
    expected = [(["one", "two"],)] if passes_argv else [()]
    assert seen == expected


def test_generate_summations_forwards_argument_values(monkeypatch):
    seen = []
    module = types.ModuleType("tensilelite.GenerateSummations")
    module.GenerateSummations = lambda args: seen.extend(args) or 11
    monkeypatch.setitem(sys.modules, "tensilelite.GenerateSummations", module)
    monkeypatch.setattr(sys, "argv", ["TensileGenerateSummations", "logic", "output"])
    commands._warned = True

    assert commands.generate_summations() == 11
    assert seen == ["logic", "output"]


def test_get_path_prints_installed_package_without_newline(capsys):
    import tensilelite

    commands._warned = True
    assert commands.get_path() == 0
    assert capsys.readouterr().out == str(Path(tensilelite.__file__).resolve().parent)
