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

import os
import shlex
import subprocess
import sys
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
ROCKE_DIR = "dnn-providers/hip-kernel-provider/rocke/"
# The rocKE hooks of the monorepo's config. The config is outside rocke/, so a
# change there that drops or renames a rocKE gate gets no rocKE review; this
# list is under rocke/, so changing it does.
ROCKE_HOOK_IDS = frozenset(
    {
        "rocke-arch-table-engine-parity",
        "rocke-compliance",
        "rocke-compliance-rules",
        "rocke-emitter-pair-files",
        "rocke-emitter-pair-parity",
        "rocke-gemm-spec-dict-binding-parity",
        "rocke-hook-checks",
        "rocke-library-reach",
        "rocke-native-unit-tests",
        "rocke-no-torch",
        "rocke-on-deletion",
        "rocke-pending-lanes",
        "rocke-toolchain-pins",
    }
)

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


def test_the_local_hooks_load_in_the_order_the_config_lists_them(config):
    hooks = precommit_hooks.load_local_hooks(config)
    assert [hook.hook_id for hook in hooks] == ["checks", "plain", "twice", "twice"]
    assert hooks[0] == precommit_hooks.load_local_hook(config, "checks")


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


@pytest.fixture(scope="module")
def rocke_hooks(monorepo_config):
    return [
        hook
        for hook in precommit_hooks.load_local_hooks(monorepo_config)
        if hook.hook_id.startswith("rocke-")
    ]


def test_the_monorepo_config_holds_exactly_the_rocke_hooks(rocke_hooks):
    ids = [hook.hook_id for hook in rocke_hooks]
    assert len(ids) == len(set(ids)), "a rocKE hook id is listed twice"
    assert set(ids) == ROCKE_HOOK_IDS, (
        "the rocKE hooks of .pre-commit-config.yaml changed: missing "
        f"{sorted(ROCKE_HOOK_IDS - set(ids))}, new {sorted(set(ids) - ROCKE_HOOK_IDS)}; "
        "update ROCKE_HOOK_IDS here when the change is meant"
    )


def test_the_config_wide_files_and_exclude_leave_rocke_visible(rocke_hooks):
    for hook in rocke_hooks:
        assert hook.config_files.search(ROCKE_DIR), hook.hook_id
        assert not hook.config_exclude.search(ROCKE_DIR), hook.hook_id


def test_every_file_a_rocke_hook_runs_exists(rocke_hooks):
    named = {
        (hook.hook_id, token)
        for hook in rocke_hooks
        for token in (*shlex.split(hook.entry), *hook.args)
        if token.startswith(ROCKE_DIR) and token.endswith(".py")
    }
    assert {hook_id for hook_id, _ in named} == ROCKE_HOOK_IDS
    missing = sorted(
        f"{hook_id}: {path}"
        for hook_id, path in named
        if not (MONOREPO / path).is_file()
    )
    assert not missing, "rocKE hooks run files that do not exist:\n  " + "\n  ".join(
        missing
    )


@pytest.mark.parametrize(
    "hook_id", ["rocke-pending-lanes", "rocke-on-deletion", "rocke-hook-checks"]
)
def test_a_wrapped_hook_says_skip_and_passes_where_rocke_is_not_checked_out(
    rocke_hooks, hook_id, tmp_path
):
    # CI checks out only the directories a change touches; from a directory
    # without rocKE the wrapper runs nothing, and says so.
    (hook,) = [hook for hook in rocke_hooks if hook.hook_id == hook_id]
    program, *arguments = shlex.split(hook.entry)
    assert program == "python"
    result = subprocess.run(
        [sys.executable, *arguments, *hook.args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout == f"SKIP: {hook.args[0]} is not checked out\n"


@pytest.mark.parametrize(
    "path",
    [
        # The shared CTest category parser the provider's categories go through.
        "shared/ctest/parse_ctest_categories.py",
        # The teams each registry record must name.
        ".github/CODEOWNERS",
        "dnn-providers/hip-kernel-provider/ROCKE_ENGINE_test_categories_external.yaml",
    ],
)
def test_the_registry_hook_takes_the_files_it_reads_outside_rocke(rocke_hooks, path):
    (hook,) = [hook for hook in rocke_hooks if hook.hook_id == "rocke-pending-lanes"]
    assert hook.takes(path)


def _pytest_test_files(hook) -> list[Path]:
    """The test files ``hook`` runs through pytest, as monorepo paths."""
    tokens = [*shlex.split(hook.entry), *hook.args]
    if "pytest" not in tokens:
        return []
    return [
        MONOREPO / token
        for token in tokens
        if token.startswith(ROCKE_DIR) and token.endswith(".py")
    ]


def test_a_hook_whose_tests_configure_the_platform_requires_cmake_and_ctest(
    rocke_hooks,
):
    # A test that configures the platform takes the build_programs fixture,
    # which skips it without cmake or ctest; a hook that ran it so would pass
    # having checked nothing, so the hook makes the fixture fail instead.
    configuring = {
        hook.hook_id: hook
        for hook in rocke_hooks
        for test_file in _pytest_test_files(hook)
        if "build_programs" in test_file.read_text(encoding="utf-8")
    }
    assert configuring, "no rocKE hook runs a test that takes build_programs"
    lax = sorted(
        hook_id
        for hook_id, hook in configuring.items()
        if "--require-build-programs" not in (*shlex.split(hook.entry), *hook.args)
    )
    assert not lax, (
        "these rocKE hooks run tests that take build_programs without "
        f"--require-build-programs, so a missing cmake would pass them: {lax}"
    )


@pytest.mark.parametrize(
    ("hook_option", "returncode", "outcome"),
    [
        ([], 0, "1 skipped"),
        (
            ["--require-build-programs"],
            1,
            (
                "--require-build-programs: missing=['cmake', 'ctest'], expected "
                "cmake and ctest on PATH"
            ),
        ),
    ],
)
def test_a_test_that_configures_the_platform_fails_without_cmake_under_the_hook_option(
    tmp_path, hook_option, returncode, outcome
):
    # One case of the build-tree runner's test that configures the platform,
    # run with no cmake or ctest on PATH: it skips in a plain run and fails in
    # a hook's run.
    no_programs = tmp_path / "empty-path"
    no_programs.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-q",
            "--basetemp",
            str(tmp_path / "basetemp"),
            *hook_option,
            str(Path(__file__).with_name("test_run_build_tree_ctest.py")),
            "-k",
            "test_an_unregistered_entry_is_refused_by_name",
        ],
        env={**os.environ, "PATH": str(no_programs)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == returncode, result.stdout + result.stderr
    assert outcome in result.stdout, result.stdout
