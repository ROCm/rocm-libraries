# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A local pre-commit hook loads from the config and takes the paths pre-commit gives it.

``rocke/tools/precommit_hooks.py`` reads one ``repo: local`` hook of a
pre-commit config and applies pre-commit's path rule to it: the config's
top-level ``files`` and ``exclude`` first, then the hook's own.

The last cases hold the rocKE hooks of the monorepo's own config to their
rules.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from rocke.assets import platform_root
from rocke.tests.import_closure import script_import_closure
from rocke.tests.lanes import SOURCE_TREE, skip_until
from rocke.tests.source_modules import load_module_from_path

pytest.importorskip("yaml", reason="reading a pre-commit config needs PyYAML")

MODULE = Path(__file__).resolve().parents[3] / "tools" / "precommit_hooks.py"
if not MODULE.is_file():
    skip_until(
        SOURCE_TREE, reason="rocke/tools/precommit_hooks.py is absent from this tree"
    )
precommit_hooks = load_module_from_path("precommit_hooks_under_test", MODULE)

PLATFORM = platform_root().resolve()
# rocke/platform -> rocke -> hip-kernel-provider -> dnn-providers -> monorepo.
MONOREPO = PLATFORM.parents[3]
MONOREPO_CONFIG = MONOREPO / ".pre-commit-config.yaml"

CONFIG = """\
exclude: ^vendored/
repos:
  - repo: https://example.com/remote-hooks
    rev: v1
    hooks:
      - id: remote-only
  - repo: local
    hooks:
      - id: checks
        name: checks
        entry: python -B run_checks.py --strict
        language: python
        args: [--quiet]
        additional_dependencies: [pytest]
        files: ^lib/
        exclude: \\.md$
        pass_filenames: false
      - id: plain
        name: plain
        entry: python plain.py
        language: python
      - id: twice
        name: twice
        entry: python one.py
        language: python
  - repo: local
    hooks:
      - id: twice
        name: twice
        entry: python two.py
        language: python
"""


@pytest.fixture
def config(tmp_path):
    path = tmp_path / ".pre-commit-config.yaml"
    path.write_text(CONFIG)
    return path


def test_a_hook_loads_its_command_and_settings(config):
    hook = precommit_hooks.load_local_hook(config, "checks")
    assert hook.entry == "python -B run_checks.py --strict"
    assert hook.args == ("--quiet",)
    assert hook.language == "python"
    assert hook.additional_dependencies == ("pytest",)
    assert hook.pass_filenames is False


def test_a_hook_that_sets_nothing_gets_pre_commits_defaults(config):
    hook = precommit_hooks.load_local_hook(config, "plain")
    assert hook.args == ()
    assert hook.additional_dependencies == ()
    assert hook.pass_filenames is True
    assert hook.takes("any/path.py")
    assert not hook.takes("vendored/any/path.py")


@pytest.mark.parametrize(
    ("path", "taken"),
    [
        ("lib/module.py", True),
        # The hook's exclude.
        ("lib/README.md", False),
        # Outside the hook's files.
        ("tools/module.py", False),
        # The config's top-level exclude holds for every hook.
        ("vendored/lib/module.py", False),
    ],
)
def test_a_path_reaches_a_hook_through_both_levels_of_files_and_exclude(
    config, path, taken
):
    assert precommit_hooks.load_local_hook(config, "checks").takes(path) is taken


@pytest.mark.parametrize("hook_id", ["no-such-hook", "remote-only"])
def test_an_id_no_local_hook_has_is_refused(config, hook_id):
    with pytest.raises(
        KeyError,
        match=rf"loading a pre-commit hook failed - id='{hook_id}', expected a "
        r"local hook of .* with that id",
    ):
        precommit_hooks.load_local_hook(config, hook_id)


def test_an_id_two_local_hooks_share_is_refused(config):
    with pytest.raises(
        ValueError,
        match=r"loading a pre-commit hook failed - id='twice' names 2 local hooks "
        r"of .*, expected one",
    ):
        precommit_hooks.load_local_hook(config, "twice")


def test_a_config_that_is_not_yaml_is_refused(tmp_path):
    config = tmp_path / ".pre-commit-config.yaml"
    config.write_text("repos: [unclosed\n")
    with pytest.raises(
        ValueError,
        match=r"(?s)loading a pre-commit hook failed - config=.* does not parse .*, "
        r"expected a YAML pre-commit config",
    ):
        precommit_hooks.load_local_hook(config, "checks")


@pytest.fixture(scope="module")
def monorepo_config():
    if not MONOREPO_CONFIG.is_file():
        skip_until(
            SOURCE_TREE,
            reason="reads the monorepo's pre-commit config, which only a monorepo "
            "checkout has",
        )
    return MONOREPO_CONFIG


def test_the_hook_checks_take_every_module_their_tests_import_beyond_rocke(
    monorepo_config, tmp_path
):
    # A change to a module one of rocke-hook-checks' tests imports can change
    # that test's outcome, so the hook must take it. What import rocke loads
    # is left out: a break there fails rocke-pending-lanes, which runs on every
    # change to rocKE but its documentation.
    hook = precommit_hooks.load_local_hook(monorepo_config, "rocke-hook-checks")
    test_files = [
        MONOREPO / argument for argument in hook.args if argument.endswith(".py")
    ]
    assert test_files
    search_roots = [PLATFORM / "python"]
    import_rocke = tmp_path / "import_rocke.py"
    import_rocke.write_text("import rocke\n")
    loaded_by_rocke = script_import_closure(import_rocke, search_roots).files
    imported = {
        path.relative_to(MONOREPO).as_posix()
        for test_file in test_files
        for path in script_import_closure(test_file, search_roots).files
        - loaded_by_rocke
        if path.is_relative_to(MONOREPO)
    }
    missing = sorted(path for path in imported if not hook.takes(path))
    assert not missing, (
        "rocke-hook-checks in .pre-commit-config.yaml does not take these files "
        "its tests import; add them to its files: pattern:\n  " + "\n  ".join(missing)
    )
