# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The quick tier keeps to its rules: unit tests and basic enablement checks only.

TESTING.md section 5.0.3 says what each tier is for. This check configures the
platform (nothing is built), labels its entries with the provider categories as
CI does, and holds the quick tier to the rules of
``rocke/tools/check_entry_times.py``. The installed entries, which CI runs, are
judged on each operating system's column of ``entry_times.json`` against that
system's registration: every entry a CI tier runs has a recorded time, quick
fits its budget and no item of a quick entry is over the item budget, so a new
entry fails here until its time is recorded from a CI log. Neither the
installed nor the build-tree entries of quick may be registered as GPU work
unless ``GPU_ENTRY_ALLOWLIST`` names them, and no registration may carry the
quick label itself: the provider categories' quick list is the one place that
puts an entry in quick.
"""

from __future__ import annotations

import dataclasses
import re
import shutil
import sys
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

# The configure arguments that make a registration another operating system's.
# rocKE's CMake registers or leaves out an entry for Windows only in `if(WIN32)`
# blocks (UNIX and MSVC pick link and compile options), so setting WIN32 lists
# the Windows registration on any other host. A Windows host cannot unset it,
# so it lists only its own.
_REGISTRATION_ARGUMENTS = {"linux": (), "windows": ("-DWIN32=ON",)}

# What each column fails today, by entry: on Windows, the entries registered
# since the Windows times were recorded, until a Windows CI log records them
# (check_entry_times.py record --column windows), and rocke_ir_serialize_roundtrip,
# recorded at 3.7 s there, over quick's 2 s item budget until it gets cheaper
# there or moves to standard. A Windows rocke_entry_time_check fails on each of
# these. Burn entries down, never add one: a new gap fails here, and so does a
# listed one that no longer fails.
_KNOWN_GAPS = {
    "linux": frozenset(),
    "windows": frozenset(
        {
            "rocke_attention_dense_gfx950_spec",
            "rocke_contracts_pytest",
            "rocke_engine_modules_check",
            "rocke_entry_time_check",
            "rocke_ir_serialize_roundtrip",
            "rocke_library_dsa_pytest",
            "rocke_library_host_pytest",
            "rocke_platform_gpu_pytest",
            "rocke_sha256",
            "rocke_spec_refusal_guards",
            "rocke_tiling_backend_width_pytest",
            "rocke_value_names",
        }
    ),
}


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


def _configured(scratch: Path, column: str) -> _Registration:
    """The registration of ``column``'s operating system, configured under
    ``scratch``."""
    build, ctest_dir = configure_installed_ctest(
        scratch,
        f"-DROCKE_INSTALL_PROVIDER_CTEST_SUBDIR={PROVIDER_CTEST_SUBDIR}",
        *_REGISTRATION_ARGUMENTS[column],
    )
    installed = _registered(list_installed_ctest_entries(ctest_dir, scratch))
    build_tree = _registered(list_ctest_entries(build))
    return _Registration(
        registered_installed=installed,
        registered_build_tree=build_tree,
        installed=_labelled(installed, scratch),
        build_tree=_labelled(build_tree, scratch),
    )


@pytest.fixture(scope="module")
def registrations(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, _Registration]:
    """Each column's registration this host can list: all of them, except a
    Windows host's, which lists only its own."""
    if shutil.which("cmake") is None or shutil.which("ctest") is None:
        pytest.skip("CMake and CTest required")
    host = check.host_column()
    listable = [host] if sys.platform == "win32" else list(_REGISTRATION_ARGUMENTS)
    return {
        column: _configured(tmp_path_factory.mktemp(f"quick-tier-{column}"), column)
        for column in listable
    }


@pytest.fixture(scope="module")
def registration(registrations: dict[str, _Registration]) -> _Registration:
    """This operating system's registration."""
    return registrations[check.host_column()]


@pytest.fixture(params=sorted(_REGISTRATION_ARGUMENTS))
def column(request: pytest.FixtureRequest, registrations) -> str:
    """Each column of ``entry_times.json`` whose registration this host lists."""
    if request.param not in registrations:
        pytest.skip(
            f"a {check.host_column()} host cannot list the {request.param} registration"
        )
    return request.param


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


def _column_times(column: str) -> dict[str, dict]:
    recorded = check.load_entry_times()
    assert column in recorded, f"entry_times.json has no {column} column"
    return {column: recorded[column]}


def _quick_tier_judgement(
    installed: Sequence[check.RegisteredEntry], column: str
) -> tuple[check.TierVerdict, list[str]]:
    """The quick tier's projection among the installed entries on ``column``'s
    recorded times, and everything the tier's rules refuse there."""
    entry_times = _column_times(column)
    [verdict] = check.evaluate_tiers(installed, entry_times, tiers=[QUICK])
    breaches = check.tier_breaches(installed, entry_times, [verdict], tiers=[QUICK])
    return verdict, breaches


def _names_entry(breach: str, name: str) -> bool:
    return re.search(rf"\b{re.escape(name)}\b", breach) is not None


def _beyond_known_gaps(breaches: Sequence[str], column: str) -> list[str]:
    """The ``breaches`` that name no entry of ``column``'s known gaps, then one
    line per known gap that no breach names any more."""
    known = _KNOWN_GAPS[column]
    new = [b for b in breaches if not any(_names_entry(b, n) for n in known)]
    closed = [
        f"{column}: {name} no longer fails; remove it from _KNOWN_GAPS"
        for name in sorted(known)
        if not any(_names_entry(b, name) for b in breaches)
    ]
    return [*new, *closed]


def test_the_installed_quick_entries_keep_to_the_tier_rules(
    registrations, column: str
) -> None:
    installed = registrations[column].installed
    assert any(entry.runs_in(QUICK) for entry in installed), "no entry runs in quick"
    verdict, breaches = _quick_tier_judgement(installed, column)
    standard_untimed = [
        breach
        for breach in check.untimed_entries(installed, _column_times(column))
        if breach not in breaches
    ]
    beyond = _beyond_known_gaps([*breaches, *standard_untimed], column)
    assert not beyond, "\n".join([verdict.summary(), *beyond])


def test_the_windows_registration_leaves_out_its_non_windows_entries(
    registrations,
) -> None:
    """The WIN32 configure lists the Windows registration: an entry the CMake
    registers only outside Windows is not in it."""
    if set(registrations) != set(_REGISTRATION_ARGUMENTS):
        pytest.skip("lists the Windows registration from another host")
    windows = {entry.name for entry in registrations["windows"].installed}
    linux = {entry.name for entry in registrations["linux"].installed}
    not_on_windows = "rocke_library_gdn_pytest"
    assert not_on_windows in linux
    assert not_on_windows not in windows


def test_a_new_untimed_entry_is_found_beyond_the_known_gaps(
    registrations, column: str
) -> None:
    """The per-column check fails an entry a CI tier runs whose time is not
    recorded, so its empty result means each such entry is a known gap."""
    entry = next(e for e in registrations[column].installed if e.runs_in(QUICK))
    planted = dataclasses.replace(entry, name="rocke_planted_untimed_entry")
    beyond = _beyond_known_gaps(
        check.untimed_entries([planted], _column_times(column)), column
    )
    assert any("rocke_planted_untimed_entry has no time" in line for line in beyond)


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
    _, breaches = _quick_tier_judgement(planted, check.host_column())
    assert any(refused in breach for breach in breaches)


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
