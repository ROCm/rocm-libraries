# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The rocke-emitter-pair-parity hook takes every file its entries read.

The hook builds two emitter programs and runs two build-tree CTest entries
(``tools/run_build_tree_ctest.py``). Its ``files:`` pattern in the root
``.pre-commit-config.yaml`` names what can change their outcome, so a change to
any other file skips it. This test keeps the pattern whole as the entries grow.
The hook must take:

* every file of the monorepo that the platform's configure reads;
* every source of the emitter programs and of each target they link, and every
  file in those targets' include directories in the monorepo;
* every Python file the entries' scripts and the hook's runner can import; and
  an import of theirs that resolves nowhere in the platform's Python tree must
  be one ``UNRESOLVED_IMPORTS`` lists;

and it must take no test module that none of those scripts can import, so a
change to another test leaves the hook out. The pattern also names the build
steps' generator scripts and their inputs, and the platform's cmake/ helpers:
CMake does not report the files a build step's command reads, so no check here
can see them.

The rocke-native-unit-tests hook runs the same runner, which builds every
program of the tree for the native unit entries. It must take every file the
configure reads, every source of every target and every file in their include
directories in the monorepo, and every file the runner imports beyond what
import rocke loads (a break there fails rocke-pending-lanes). The files those
programs read at run time are named in its pattern by hand: CTest lists an
entry's arguments only once its program is built.
"""

from __future__ import annotations

import shlex
import shutil
from itertools import pairwise
from pathlib import Path

import pytest
from rocke.assets import platform_root
from rocke.tests.cmake_file_api import read_codemodel, request_codemodel
from rocke.tests.ctest_listing import list_ctest_entries
from rocke.tests.import_closure import script_import_closure
from rocke.tests.lanes import SOURCE_TREE, skip_until_reason
from rocke.tests.platform_build import configure_platform
from rocke.tests.source_modules import load_module_from_path

HOOK_ID = "rocke-emitter-pair-parity"
NATIVE_HOOK_ID = "rocke-native-unit-tests"
PLATFORM = platform_root().resolve()
# rocke/platform -> rocke -> hip-kernel-provider -> dnn-providers -> monorepo.
MONOREPO = PLATFORM.parents[3]
PRECOMMIT_CONFIG = MONOREPO / ".pre-commit-config.yaml"
PRECOMMIT_HOOKS = PLATFORM.parent / "tools" / "precommit_hooks.py"

# The top-level names the scripts import that the platform's Python tree does
# not hold, each with why the hook need not take a file for it. A new name
# here is a module the hook cannot see; burn entries down, do not add lightly.
UNRESOLVED_IMPORTS = {
    "numpy": "an installed dependency",
    "torch": "an optional installed dependency",
    "rocke_engine": "the engine's Python binding, built from cpp/bindings, which "
    "neither emitter program links",
    "kernels": "the library's kernel package; the platform-only configure leaves "
    "the library out, and rocke-library-reach holds the platform's imports of it",
}

pytestmark = [
    pytest.mark.skipif(
        not (PRECOMMIT_CONFIG.is_file() and PRECOMMIT_HOOKS.is_file()),
        reason=skip_until_reason(
            SOURCE_TREE,
            reason="reads the monorepo's pre-commit config, which only a monorepo "
            "checkout has",
        ),
    ),
    pytest.mark.skipif(
        shutil.which("cmake") is None or shutil.which("ctest") is None,
        reason="CMake and CTest required",
    ),
]


def _load_hook(hook_id: str):
    pytest.importorskip("yaml", reason="reading the pre-commit config needs PyYAML")
    precommit_hooks = load_module_from_path("rocke_precommit_hooks", PRECOMMIT_HOOKS)
    return precommit_hooks.load_local_hook(PRECOMMIT_CONFIG, hook_id)


@pytest.fixture(scope="module")
def hook():
    return _load_hook(HOOK_ID)


@pytest.fixture(scope="module")
def native_hook():
    return _load_hook(NATIVE_HOOK_ID)


def _runner_script(hook) -> Path:
    """The one Python script the hook's command runs."""
    (runner,) = [
        MONOREPO / token for token in shlex.split(hook.entry) if token.endswith(".py")
    ]
    return runner


def _option_values(arguments: list[str], option: str) -> list[str]:
    return [value for flag, value in pairwise(arguments) if flag == option]


@pytest.fixture(scope="module")
def hook_command(hook):
    """The runner script, CMake targets and CTest entries the hook names."""
    arguments = shlex.split(hook.entry)
    return (
        _runner_script(hook),
        _option_values(arguments, "--target"),
        _option_values(arguments, "--test"),
    )


@pytest.fixture(scope="module")
def configured_tree(hook_command, tmp_path_factory):
    """The platform configured as the hook's runner configures it, unbuilt,
    with the CMake file API's replies."""
    runner_path, _, _ = hook_command
    runner = load_module_from_path("rocke_run_build_tree_ctest", runner_path)
    build_root = tmp_path_factory.mktemp("build")
    request_codemodel(build_root)
    configure_platform(build_root, *runner.PLATFORM_ONLY, config=runner.CONFIG)
    return build_root, runner.CONFIG


@pytest.fixture(scope="module")
def script_imports(hook_command, configured_tree):
    """What the hook's runner and its CTest entries' scripts can import: the
    files, and the names found nowhere."""
    runner, _, tests = hook_command
    build_root, config = configured_tree
    entries = list_ctest_entries(
        build_root,
        config=config,
        name_pattern="^(" + "|".join(tests) + ")$",
    )
    assert sorted(entry["name"] for entry in entries) == sorted(tests)
    scripts = {runner} | {
        Path(argument).resolve()
        for entry in entries
        for argument in entry["command"]
        if argument.endswith(".py")
    }
    closures = [
        script_import_closure(script, [PLATFORM / "python"]) for script in scripts
    ]
    files = frozenset().union(*(closure.files for closure in closures))
    unresolved = frozenset().union(*(closure.unresolved for closure in closures))
    return files, unresolved


def _repo_relative(paths) -> list[str]:
    """``paths`` inside the monorepo, relative to it, sorted."""
    return sorted(
        path.relative_to(MONOREPO).as_posix()
        for path in paths
        if path.is_relative_to(MONOREPO)
    )


def _not_taken(hook, paths) -> list[str]:
    """``paths`` inside the monorepo that ``hook`` does not take, repo-relative."""
    return [path for path in _repo_relative(paths) if not hook.takes(path)]


def _missing(paths: list[str], what: str, hook_id: str = HOOK_ID) -> str:
    return (
        f"the {hook_id} hook in .pre-commit-config.yaml does not take {what}; add "
        "to its files: pattern:\n  " + "\n  ".join(paths)
    )


def test_the_hook_takes_every_file_the_configure_reads(hook, configured_tree):
    build_root, config = configured_tree
    model = read_codemodel(build_root, config)
    assert model.input_files
    missing = _not_taken(hook, model.input_files)
    assert not missing, _missing(missing, "these files the configure reads")


def test_the_hook_takes_every_file_the_emitter_programs_compile(
    hook, hook_command, configured_tree
):
    _, emitter_targets, _ = hook_command
    build_root, config = configured_tree
    model = read_codemodel(build_root, config)
    targets = [model.targets[name] for name in model.linked_targets(emitter_targets)]
    sources = {source for target in targets for source in target.sources}
    headers = {
        path
        for target in targets
        for include_dir in target.include_dirs
        if include_dir.is_relative_to(MONOREPO)
        for path in include_dir.rglob("*")
        if path.is_file()
    }
    assert sources and headers
    missing = _not_taken(hook, sources | headers)
    assert not missing, _missing(missing, "these files the emitter programs compile")


def test_the_hook_takes_every_python_file_its_scripts_can_import(hook, script_imports):
    files, _ = script_imports
    missing = _not_taken(hook, files)
    assert not missing, _missing(missing, "these files its scripts can import")


def test_every_import_of_its_scripts_resolves_or_is_listed(script_imports):
    _, unresolved = script_imports
    unlisted = sorted(unresolved - UNRESOLVED_IMPORTS.keys())
    assert not unlisted, (
        f"the scripts of the {HOOK_ID} hook import {unlisted}, found nowhere in "
        "the platform's Python tree; if the module is in the monorepo (a "
        "sys.path entry a script adds, for one), put it where the scripts' "
        "imports resolve and its file in the hook's files: pattern, else list it "
        "in UNRESOLVED_IMPORTS with the reason"
    )
    stale = sorted(UNRESOLVED_IMPORTS.keys() - unresolved)
    assert (
        not stale
    ), f"no script imports {stale} any more; drop it from UNRESOLVED_IMPORTS"


def test_the_hook_takes_no_test_module_its_scripts_cannot_import(hook, script_imports):
    files, _ = script_imports
    test_modules = {
        path
        for root in (PLATFORM / "tests", PLATFORM / "python")
        for path in root.rglob("test_*.py")
    }
    needless = [
        path for path in _repo_relative(test_modules - files) if hook.takes(path)
    ]
    assert not needless, (
        f"the {HOOK_ID} hook in .pre-commit-config.yaml takes test modules its "
        "scripts cannot import, so a change to them runs it for nothing; narrow its "
        "files: pattern:\n  " + "\n  ".join(needless)
    )


def test_the_native_hook_runs_the_runner_it_shares_with_the_emitter_pair_hook(
    hook, native_hook
):
    # The configured tree the cases below check is the runner's own configure.
    assert _runner_script(native_hook) == _runner_script(hook)


def test_the_native_hook_takes_every_file_the_configure_reads(
    native_hook, configured_tree
):
    build_root, config = configured_tree
    model = read_codemodel(build_root, config)
    missing = _not_taken(native_hook, model.input_files)
    assert not missing, _missing(
        missing, "these files the configure reads", NATIVE_HOOK_ID
    )


def test_the_native_hook_takes_every_file_the_programs_compile(
    native_hook, configured_tree
):
    build_root, config = configured_tree
    model = read_codemodel(build_root, config)
    targets = list(model.targets.values())
    sources = {source for target in targets for source in target.sources}
    headers = {
        path
        for target in targets
        for include_dir in target.include_dirs
        if include_dir.is_relative_to(MONOREPO)
        for path in include_dir.rglob("*")
        if path.is_file()
    }
    assert sources and headers
    missing = _not_taken(native_hook, sources | headers)
    assert not missing, _missing(
        missing, "these files the tree's programs compile", NATIVE_HOOK_ID
    )


def test_the_native_hook_takes_what_its_runner_imports_beyond_rocke(
    native_hook, tmp_path
):
    search_roots = [PLATFORM / "python"]
    import_rocke = tmp_path / "import_rocke.py"
    import_rocke.write_text("import rocke\n")
    loaded_by_rocke = script_import_closure(import_rocke, search_roots).files
    imported = script_import_closure(_runner_script(native_hook), search_roots).files
    missing = _not_taken(native_hook, imported - loaded_by_rocke)
    assert not missing, _missing(
        missing, "these files its runner imports", NATIVE_HOOK_ID
    )
