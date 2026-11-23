# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The CI lanes a test can wait for, and how a skipped case says which one.

A lane is something the pull-request and push CI runs lack: a job that does
not exist yet (a scheduled run, a pull request that has not landed), or, for a
file CI does run, something its cases need there (torch, a gfx950 runner, the
C++ engine's Python modules). ``rocke.tests.pending_lanes`` records which test
files wait for which lane, with an owner and an expiry.

A case that skips because CI lacks what a lane would give it says so where it
skips, with the lane itself rather than with words a reader must interpret. A
case that needs a GPU states the arches it needs, and this module books the
runner lane they wait for (:func:`arch_skip_reason`)::

    skip_needing_arch(("gfx950",), reason="needs a gfx950 GPU, found gfx942")
    torch = import_or_skip_until("torch", TORCH_IN_CI)
    @pytest.mark.skipif(no_engine, reason=skip_until_reason(BINDING_LANE,
                                                            reason="no rocke_engine"))

The skip reason then starts with ``waits for <lane>: `` (``skip_until_reason``),
which the run records keep and the limbo audit reads back
(:func:`lanes_waited_for`). A skip for any other reason (a defect, a refused
configuration, a module that is installed but fails to import) names no lane,
and no registry entry can account for it: it stays in limbo until it is fixed
or quarantined. ``platform/tests/tools/test_pending_lanes.py`` holds the tree
to this: no test names a runner lane or writes a declaration by hand. The
shared helpers declare for their callers (``rocke.tests.gpu``,
``rocke.tests.engines``, ``rocke.tests.compiler_tools``, ...), and a case a CI
entry deselects by a lane's marker (``exhaustive``, ``integration``) is
recorded as waiting for that lane (``rocke.tests.run_records``).
"""

from __future__ import annotations

import importlib
import os
import re
from collections.abc import Collection
from dataclasses import dataclass
from types import ModuleType
from typing import NoReturn

import pytest

from ..core.arch.target import known_arches

__all__ = [
    "BINDING_LANE",
    "CI_TEST_TOOLS",
    "COMPREHENSIVE_CASES",
    "ENROLMENT",
    "EXHAUSTIVE_CASES",
    "GFX90A_RUNNER",
    "GFX950_RUNNER",
    "GFX1151_GFX12_RUNNER",
    "GFX1250_RUNNER",
    "LANES",
    "MULTI_GPU_RUNNER",
    "NIGHTLY_GPU",
    "NIGHTLY_SOURCE",
    "OWNER_DECISION",
    "PR_16482",
    "REQUIRE_CPP_ENGINE_ENV",
    "SHARED_LIBRARY_LANE",
    "SOURCE_TREE_CASES",
    "TORCH_IN_CI",
    "WEEKLY_EXHAUSTIVE",
    "Lane",
    "arch_skip_reason",
    "cpp_engine_required",
    "import_or_skip_until",
    "lanes_waited_for",
    "runner_lanes_for",
    "skip_needing_arch",
    "skip_until",
    "skip_until_reason",
    "skip_until_shared_engine",
]


@dataclass(frozen=True)
class Lane:
    """A CI lane that pending tests wait for.

    ``enrolled`` is False for a lane no CI-run entry has yet: every case of its
    files waits for it. It is True for something the CI-run entries lack: its
    files are in a CI-run entry, and a case waits for it only when it says so
    (:func:`skip_until`). ``waits_for_torch`` says each of its files imports
    torch. ``arches`` names the GPU arches of a runner lane, whose cases the GPU
    helpers declare (:func:`runner_lanes_for`); ``marker`` names the pytest
    marker by which the CI-run entries deselect the lane's cases.
    """

    name: str
    description: str
    enrolled: bool
    waits_for_torch: bool = False
    arches: tuple[str, ...] = ()
    marker: str = ""

    def __post_init__(self) -> None:
        if not _LANE_NAME.fullmatch(self.name):
            raise ValueError(
                f"Lane failed - name={self.name!r}, expected lowercase words "
                "joined by '-'"
            )
        if (self.arches or self.marker) and not self.enrolled:
            raise ValueError(
                f"Lane failed - name={self.name!r}, arches={self.arches!r}, "
                f"marker={self.marker!r}, expected neither for a lane whose files "
                "are in no CI-run entry: every case of its files waits for it"
            )
        if not isinstance(self.arches, tuple) or not set(self.arches) <= set(
            known_arches()
        ):
            raise ValueError(
                f"Lane failed - name={self.name!r}, arches={self.arches!r}, "
                "expected a tuple of known gfx arches"
            )


_LANE_NAME = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")

PR_16482 = Lane(
    "pr-16482",
    "pull request 16482, which adds the attention backward CTest entries (a draft)",
    enrolled=False,
)
ENROLMENT = Lane(
    "enrolment",
    "a registration in an existing quick or standard entry, proposed by triage and "
    "waiting for the owner's budget or tier decision",
    enrolled=False,
)
NIGHTLY_SOURCE = Lane(
    "nightly-source",
    "a scheduled source-tree run (run_all.py; the comprehensive tier), which no "
    "job runs yet",
    enrolled=False,
)
WEEKLY_EXHAUSTIVE = Lane(
    "weekly-exhaustive",
    "a scheduled run of the exhaustive cohort (run_all.py --exhaustive), which no "
    "job runs yet",
    enrolled=False,
)
NIGHTLY_GPU = Lane(
    "nightly-gpu",
    "a scheduled GPU stage on gfx942 and gfx950, which does not exist yet",
    enrolled=False,
)
OWNER_DECISION = Lane(
    "owner-decision",
    "an owner decision between a lane with optional dependencies, a rewrite and a drop",
    enrolled=False,
)
BINDING_LANE = Lane(
    "binding-lane",
    "the C++ engine's Python modules in the CI test tree: the file is in a CI-run "
    "entry, but its cases that need a kernel family's C++ builder skip where the "
    "build's Python had no pybind11",
    enrolled=True,
)
TORCH_IN_CI = Lane(
    "torch-in-ci",
    "torch in the CI test environment: the file is in a CI-run entry, but its "
    "torch-checked cases skip there",
    enrolled=True,
    waits_for_torch=True,
)
GFX950_RUNNER = Lane(
    "gfx950-runner",
    "a CI GPU stage on gfx950: the file is in a CI-run entry, but its gfx950 cases "
    "skip on the gfx942 and gfx110X runners",
    enrolled=True,
    arches=("gfx950",),
)
GFX90A_RUNNER = Lane(
    "gfx90a-runner",
    "a CI GPU stage on gfx90a: the file is in a CI-run entry, but its gfx90a cases "
    "skip on the gfx942 and gfx110X runners",
    enrolled=True,
    arches=("gfx90a",),
)
GFX1250_RUNNER = Lane(
    "gfx1250-runner",
    "a CI GPU stage on gfx1250: the file is in a CI-run entry, but its gfx1250 "
    "cases skip on the gfx942 and gfx110X runners, and they have not run on "
    "gfx1250 hardware",
    enrolled=True,
    arches=("gfx1250",),
)
GFX1151_GFX12_RUNNER = Lane(
    "gfx1151-gfx12-runner",
    "a CI GPU stage on gfx1151, gfx1200 or gfx1201: the file is in a CI-run entry, "
    "but its cases for those arches skip on the gfx942 and gfx110X runners",
    enrolled=True,
    arches=("gfx1151", "gfx1200", "gfx1201"),
)
SOURCE_TREE_CASES = Lane(
    "source-tree-cases",
    "a scheduled source-tree run (run_all.py), which no job runs yet: the file is "
    "in a CI-run entry of the installed test tree, but its cases that read the "
    "source tree (platform tools, C++ sources, docs, git history) skip there, or "
    "its integration-marked cases are deselected there",
    enrolled=True,
    marker="integration",
)
EXHAUSTIVE_CASES = Lane(
    "exhaustive-cases",
    "a scheduled exhaustive run (run_all.py --exhaustive), which no job runs yet: "
    "the file is in a CI-run entry, but its exhaustive cases are deselected there",
    enrolled=True,
    marker="exhaustive",
)
COMPREHENSIVE_CASES = Lane(
    "comprehensive-cases",
    "a scheduled CI run of the comprehensive tier's installed entries, which no "
    "job runs for rocKE (TheRock's nightly does not build it): the file is in a "
    "CI-run entry, but the test runs only in a comprehensive-tier entry",
    enrolled=True,
)
MULTI_GPU_RUNNER = Lane(
    "multi-gpu-runner",
    "a CI GPU stage with two GPUs: the file is in a CI-run entry, but its cases "
    "that need two GPUs skip on the one-GPU runners",
    enrolled=True,
)
SHARED_LIBRARY_LANE = Lane(
    "shared-library-lane",
    "the shared librocke C library in the CI test tree: the file is in a CI-run "
    "entry, but its cases that load the library (the C++ engine's "
    "family-agnostic lowering, the portable-IR C ABI) skip where the build "
    "installs none (the Windows provider build) and the lane does not build "
    "one on demand",
    enrolled=True,
)
CI_TEST_TOOLS = Lane(
    "ci-test-tools",
    "the optional packages and LLVM tools these cases use, in the CI test "
    "environment: the file is in a CI-run entry, but TheRock's test step installs "
    "no jsonschema or matplotlib and puts no LLVM tools on PATH, so they skip",
    enrolled=True,
)

LANES: tuple[Lane, ...] = (
    PR_16482,
    ENROLMENT,
    NIGHTLY_SOURCE,
    WEEKLY_EXHAUSTIVE,
    NIGHTLY_GPU,
    OWNER_DECISION,
    BINDING_LANE,
    TORCH_IN_CI,
    GFX950_RUNNER,
    GFX90A_RUNNER,
    GFX1250_RUNNER,
    GFX1151_GFX12_RUNNER,
    SOURCE_TREE_CASES,
    EXHAUSTIVE_CASES,
    COMPREHENSIVE_CASES,
    MULTI_GPU_RUNNER,
    SHARED_LIBRARY_LANE,
    CI_TEST_TOOLS,
)
_BY_NAME = {lane.name: lane for lane in LANES}

# The prefix a declared skip reason starts with: the lane names, then ": ".
_WAITS_FOR = "waits for "
_DECLARED = re.compile(r"waits for ([a-z0-9-]+(?:, [a-z0-9-]+)*): ")


def skip_until_reason(*lanes: Lane, reason: str) -> str:
    """The skip reason of a case that waits for ``lanes`` (lanes CI lacks), for
    ``reason``: ``waits for <lane>[, <lane>...]: <reason>``. For a skip pytest
    or unittest raises from a reason: ``skipif``, ``skipIf``, a parameter's
    ``skip`` mark, a recorded deselection."""
    if not lanes:
        raise ValueError(
            f"skip_until_reason failed - reason={reason!r}, expected at least one lane"
        )
    for lane in lanes:
        if _BY_NAME.get(getattr(lane, "name", None)) is not lane:
            raise ValueError(
                f"skip_until_reason failed - lane={lane!r}, expected one of "
                "rocke.tests.lanes.LANES"
            )
        if not lane.enrolled:
            raise ValueError(
                f"skip_until_reason failed - lane={lane.name!r}, expected a lane CI "
                "lacks: every case of a file waiting for a lane no CI entry runs "
                "waits for it without saying so"
            )
    if not reason.strip():
        raise ValueError(
            "skip_until_reason failed - reason is empty, expected why the case skips"
        )
    return f"{_WAITS_FOR}{', '.join(lane.name for lane in lanes)}: {reason}"


def skip_until(*lanes: Lane, reason: str) -> NoReturn:
    """Skip the running case (or, called at import, its module) because CI lacks
    what ``lanes`` would give it, for ``reason``. Works in a pytest test,
    fixture or module and in a ``unittest`` test, ``setUp`` or ``setUpClass``."""
    __tracebackhide__ = True  # pytest reports the skip at the caller
    raise pytest.skip.Exception(
        skip_until_reason(*lanes, reason=reason), allow_module_level=True
    )


#: Set to ``1`` where a run requires the C++ engine: a case whose shared engine
#: is missing fails instead of skipping (:func:`skip_until_shared_engine`).
#: The ``rocke.tests.require_cpp_engine`` pytest plugin sets it.
REQUIRE_CPP_ENGINE_ENV = "ROCKE_REQUIRE_CPP_ENGINE"


def cpp_engine_required() -> bool:
    """Whether this run requires the C++ engine (``ROCKE_REQUIRE_CPP_ENGINE``)."""
    return os.environ.get(REQUIRE_CPP_ENGINE_ENV, "").strip() not in ("", "0")


def skip_until_shared_engine(reason: str) -> NoReturn:
    """Skip the running case (or class or module), for ``reason``, as waiting
    for :data:`SHARED_LIBRARY_LANE`, because no shared C++ engine was found;
    fail it instead where the run requires the C++ engine
    (:func:`cpp_engine_required`), so such a run cannot pass without it."""
    __tracebackhide__ = True  # pytest reports the outcome at the caller
    if cpp_engine_required():
        pytest.fail(
            f"{REQUIRE_CPP_ENGINE_ENV}=1 requires the C++ engine: {reason}",
            pytrace=False,
        )
    skip_until(SHARED_LIBRARY_LANE, reason=reason)


def import_or_skip_until(module_name: str, lane: Lane) -> ModuleType:
    """The module ``module_name``, or skip the running case (or module) as
    waiting for ``lane`` when it is not installed: ``pytest.importorskip``
    that names the lane. A module that is installed but fails to import (a
    broken build) raises."""
    __tracebackhide__ = True  # pytest reports the skip at the caller
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if not _is_missing_module(error, module_name):
            raise
        skip_until(lane, reason=f"could not import {module_name!r}: {error}")


def _is_missing_module(error: ModuleNotFoundError, module_name: str) -> bool:
    """Whether ``error`` says ``module_name`` (or a package it is in) is not
    installed, rather than that a module it imports is missing."""
    return error.name is not None and (
        module_name == error.name or module_name.startswith(f"{error.name}.")
    )


def runner_lanes_for(arches: Collection[str]) -> tuple[Lane, ...]:
    """The runner lanes a case that needs a GPU of one of ``arches`` waits for:
    each lane with a GPU of them, when every one of them has a runner lane (no
    CI runner has any of them). Empty when one of ``arches`` has none: a CI
    runner has it (gfx942), and the case runs there, or no lane does."""
    lanes = []
    for arch in arches:
        lane = next((lane for lane in LANES if arch in lane.arches), None)
        if lane is None:
            return ()
        if lane not in lanes:
            lanes.append(lane)
    return tuple(lanes)


def arch_skip_reason(arches: Collection[str], reason: str) -> str:
    """The skip reason of a case that needs a GPU of one of ``arches`` and runs
    on none here: declaring their runner lanes (:func:`runner_lanes_for`) when
    no CI runner has any of them, else ``reason`` as is, for the audit to leave
    the case to the runner that has one."""
    lanes = runner_lanes_for(arches)
    return skip_until_reason(*lanes, reason=reason) if lanes else reason


def skip_needing_arch(arches: Collection[str], reason: str) -> NoReturn:
    """Skip the running case (or module) that needs a GPU of one of ``arches``
    and runs on none here, for ``reason``, as :func:`arch_skip_reason` books
    it. A test states the arches it needs; which lane that waits for is this
    module's fact."""
    __tracebackhide__ = True  # pytest reports the skip at the caller
    raise pytest.skip.Exception(
        arch_skip_reason(arches, reason), allow_module_level=True
    )


def lanes_waited_for(skip_reason: str) -> tuple[Lane, ...]:
    """The lanes a skip or deselection reason says its case waits for (see
    :func:`skip_until_reason`); empty when it names none."""
    declared = _DECLARED.match(skip_reason)
    if declared is None:
        return ()
    names = declared.group(1).split(", ")
    unknown = [name for name in names if name not in _BY_NAME]
    if unknown:
        raise ValueError(
            f"lanes_waited_for failed - skip_reason={skip_reason!r}, lanes "
            f"{unknown} are not in rocke.tests.lanes.LANES, expected the names of "
            "lanes it lists (a renamed lane needs its declarations renamed)"
        )
    return tuple(_BY_NAME[name] for name in names)
