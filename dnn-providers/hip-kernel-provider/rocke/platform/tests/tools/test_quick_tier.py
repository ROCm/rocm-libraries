# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The quick tier keeps to its rules: unit tests and basic enablement checks only.

TESTING.md section 5.0.3 says what each tier is for. This check configures the
platform (nothing is built), labels its entries with the provider categories as
CI does, and holds the quick tier to the rules of
``rocke/tools/check_entry_times.py``. The installed entries, which CI runs, are
judged on this operating system's column of ``entry_times.json``: every quick
entry has a recorded time, the tier fits its budget and no item of an entry is
over the item budget, so a new quick entry fails here until its time is
recorded from a CI log. Neither the installed nor the build-tree entries of
quick may be registered as GPU work unless ``GPU_ENTRY_ALLOWLIST`` names them,
and no registration may carry the quick label itself: the provider categories'
quick list is the one place that puts an entry in quick.
"""

from __future__ import annotations

import dataclasses
import shutil
from collections.abc import Sequence
from pathlib import Path

import pytest
from rocke.tests.ctest_listing import (
    PROVIDER_CTEST_SUBDIR,
    configure_installed_ctest,
    list_ctest_entries,
    list_installed_ctest_entries,
)
from rocke.tests.lanes import SOURCE_TREE, skip_until
from rocke.tests.provider_categories import provider_category_labels
from rocke.tests.source_modules import load_entry_time_check

check = load_entry_time_check(Path(__file__).resolve().parent)
if check is None:
    skip_until(
        SOURCE_TREE,
        reason="rocke/tools/check_entry_times.py is absent from this tree",
    )
[QUICK] = [tier for tier in check.CI_TIERS if tier.label == "quick"]


@dataclasses.dataclass(frozen=True)
class _Registration:
    """The configured entries: as registered (``registered_*``), and with the
    labels CI selects them by, their own and the provider categories'."""

    registered_installed: tuple[check.RegisteredEntry, ...]
    registered_build_tree: tuple[check.RegisteredEntry, ...]
    installed: tuple[check.RegisteredEntry, ...]
    build_tree: tuple[check.RegisteredEntry, ...]


def _registered(listed: list[dict]) -> tuple[check.RegisteredEntry, ...]:
    # Nothing is built, so CTest lists no command for the C++ test binaries,
    # whose programs do not exist yet; each runs as one program, not pytest.
    return tuple(
        check.registered_entries(
            test if "command" in test else {**test, "command": [test["name"]]}
            for test in listed
        )
    )


def _labelled(
    registered: tuple[check.RegisteredEntry, ...], scratch: Path
) -> tuple[check.RegisteredEntry, ...]:
    tier_labels = provider_category_labels([e.name for e in registered], scratch)
    return tuple(
        dataclasses.replace(
            entry, labels=entry.labels | tier_labels.get(entry.name, set())
        )
        for entry in registered
    )


@pytest.fixture(scope="module")
def registration(tmp_path_factory: pytest.TempPathFactory) -> _Registration:
    if shutil.which("cmake") is None or shutil.which("ctest") is None:
        pytest.skip("CMake and CTest required")
    scratch = tmp_path_factory.mktemp("quick-tier")
    build, ctest_dir = configure_installed_ctest(
        scratch, f"-DROCKE_INSTALL_PROVIDER_CTEST_SUBDIR={PROVIDER_CTEST_SUBDIR}"
    )
    installed = _registered(list_installed_ctest_entries(ctest_dir, scratch))
    build_tree = _registered(list_ctest_entries(build))
    return _Registration(
        registered_installed=installed,
        registered_build_tree=build_tree,
        installed=_labelled(installed, scratch),
        build_tree=_labelled(build_tree, scratch),
    )


def _entries_registered_as_quick(
    registered: Sequence[check.RegisteredEntry],
) -> list[str]:
    """The entries whose own registration carries the quick label: they would
    join quick without an edit to the provider categories' quick list."""
    return sorted(entry.name for entry in registered if QUICK.label in entry.labels)


def test_no_registration_bakes_in_the_quick_label(registration) -> None:
    """The provider categories' quick list is the one place that puts an entry
    in quick."""
    baked = _entries_registered_as_quick(
        [*registration.registered_installed, *registration.registered_build_tree]
    )
    assert not baked, (
        f"registered with the quick label: {baked}; register them without it "
        "and list the ones quick runs in ROCKE_ENGINE_test_categories_external.yaml"
    )


def test_a_registration_with_the_quick_label_is_found(registration) -> None:
    """The check above finds an entry whose own registration carries the quick
    label, so its empty result means none does."""
    [entry, *_] = registration.registered_installed
    planted = dataclasses.replace(entry, labels=entry.labels | {QUICK.label})
    assert _entries_registered_as_quick([planted]) == [entry.name]


def _quick_tier_breaches(installed: Sequence[check.RegisteredEntry]) -> list[str]:
    """Everything the quick tier's rules refuse among the installed entries,
    judged on this operating system's recorded times, with the tier's
    projection first."""
    column = check.host_column()
    recorded = check.load_entry_times()
    assert column in recorded, f"entry_times.json has no {column} column"
    entry_times = {column: recorded[column]}
    [verdict] = check.evaluate_tiers(installed, entry_times, tiers=[QUICK])
    breaches = check.tier_breaches(installed, entry_times, [verdict], tiers=[QUICK])
    return [verdict.summary(), *breaches] if breaches else []


def test_the_installed_quick_entries_keep_to_the_tier_rules(registration) -> None:
    installed = registration.installed
    assert any(entry.runs_in(QUICK) for entry in installed), "no entry runs in quick"
    breaches = _quick_tier_breaches(installed)
    assert not breaches, "\n".join(breaches)


def test_no_build_tree_quick_entry_is_registered_as_gpu_work(registration) -> None:
    breaches = check.gpu_entry_breaches(registration.build_tree, tiers=[QUICK])
    assert not breaches, "\n".join(breaches)


def test_a_gpu_entry_moved_into_quick_is_refused(registration) -> None:
    """The registration carries what marks a GPU entry, so moving one into
    quick fails here."""
    gpu_entry = "rocke_tiling_gpu_pytest"
    planted = [
        (
            dataclasses.replace(entry, labels=entry.labels | {QUICK.label})
            if entry.name == gpu_entry
            else entry
        )
        for entry in registration.installed
    ]
    assert any(entry.name == gpu_entry for entry in planted), gpu_entry
    refused = f"quick: {gpu_entry} uses the GPU"
    assert any(refused in breach for breach in _quick_tier_breaches(planted))


# Coverage that runs on every pull request in standard, never in quick: the
# golden checks, the HIP emitter pair, the oracle-free GPU checks and a pinned
# reference operation's contracts and GPU lane (named as the CMake registers
# them, so the category patterns are what is checked).
_STANDARD_COVERAGE = (
    "rocke_golden_static",
    "rocke_differential_golden_ll",
    "rocke_differential_golden_canonical",
    "rocke_optimization_barrier_hip_pair",
    "rocke_oracle_free_pytest",
    "rocke_reference_common_pytest",
    "rocke_conv_reference_unit_pytest",
    "rocke_conv_gpu_gfx942_pytest",
)


def test_golden_parity_gpu_and_reference_entries_run_in_standard_not_quick(
    tmp_path: Path,
) -> None:
    labels = provider_category_labels(list(_STANDARD_COVERAGE), tmp_path)
    misplaced = {
        name: sorted(labels.get(name, set()))
        for name in _STANDARD_COVERAGE
        if "standard" not in labels.get(name, set())
        or QUICK.label in labels.get(name, set())
    }
    assert not misplaced, misplaced
