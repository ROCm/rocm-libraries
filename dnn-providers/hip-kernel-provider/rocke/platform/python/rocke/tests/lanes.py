# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The four kinds of thing a test can wait for, and how a skipped case says which.

A lane is what pull-request CI lacks that a test needs, one of four kinds
(:data:`LANES`):

- ``runner-hardware``: a GPU of an arch no pull-request runner has (gfx942 and
  gfx110X have one), or a second GPU;
- ``build-variant``: a build or test environment CI does not provide (the C++
  engine's Python modules, the shared librocke on the Windows build, optional
  packages and LLVM tools in the test step);
- ``ticket``: work a tracked issue owns (a CTest entry or tier not added yet,
  a scheduled run, a torch-free rewrite);
- ``source-tree``: a source checkout, which the installed test tree is not.

``rocke.tests.pending_lanes`` records which test files, or tests, wait for which
lane, with a team, an issue and an expiry.

A case that skips because CI lacks what a lane would give it says so where it
skips, with the lane itself rather than with words a reader must interpret. A
case that needs a GPU states the arches it needs, and this module books the
lane they wait for (:func:`arch_skip_reason`)::

    skip_needing_arch(("gfx950",), reason="needs a gfx950 GPU, found gfx942")
    engine = import_or_skip_until("rocke_engine", BUILD_VARIANT)
    @pytest.mark.skipif(no_checkout, reason=skip_until_reason(SOURCE_TREE,
                                                              reason="no cpp/"))

The skip reason then starts with ``waits for <lane>: `` (``skip_until_reason``),
which the run records keep and the limbo audit reads back
(:func:`lanes_waited_for`). A skip for any other reason (a defect, a refused
configuration, a module that is installed but fails to import) names no lane,
and no registry entry can account for it: it stays in limbo until it is fixed
or quarantined. ``platform/tests/tools/test_pending_lanes.py`` holds the tree
to this: no test names the runner lane or writes a lane reason by hand. The
shared helpers declare for their callers (``rocke.tests.gpu``,
``rocke.tests.engines``, ``rocke.tests.compiler_tools``, ...), and a case a CI
entry deselects by a lane's marker (``exhaustive``, ``integration``) is
recorded as waiting for that lane (``rocke.tests.run_records``).
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import re
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import NoReturn

import pytest
from packaging.requirements import Requirement

from ..core.arch.target import known_arches

__all__ = [
    "BUILD_VARIANT",
    "LANES",
    "REQUIRE_CPP_ENGINE_ENV",
    "RUNNER_HARDWARE",
    "SOURCE_TREE",
    "TICKET",
    "Lane",
    "arch_skip_reason",
    "cpp_engine_required",
    "import_or_skip_until",
    "lanes_waited_for",
    "runner_lanes_for",
    "skip_needing_arch",
    "skip_until",
    "skip_until_installed",
    "skip_until_reason",
    "skip_until_shared_engine",
]


@dataclass(frozen=True)
class Lane:
    """One kind of thing pull-request CI lacks that pending tests wait for.

    ``arches`` names the GPU arches no pull-request runner has, whose cases
    the GPU helpers declare (:func:`runner_lanes_for`); ``marker`` names the
    pytest marker by which the CI-run entries deselect the lane's cases.
    """

    name: str
    description: str
    arches: tuple[str, ...] = ()
    marker: str = ""

    def __post_init__(self) -> None:
        if not _LANE_NAME.fullmatch(self.name):
            raise ValueError(
                f"Lane failed - name={self.name!r}, expected lowercase words "
                "joined by '-'"
            )
        if not isinstance(self.arches, tuple) or not set(self.arches) <= set(
            known_arches()
        ):
            raise ValueError(
                f"Lane failed - name={self.name!r}, arches={self.arches!r}, "
                "expected a tuple of known gfx arches"
            )


_LANE_NAME = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")

RUNNER_HARDWARE = Lane(
    "runner-hardware",
    "hardware no pull-request CI runner has: a GPU of an arch other than gfx942 "
    "and gfx110X, or a second GPU",
    arches=("gfx90a", "gfx950", "gfx1151", "gfx1200", "gfx1201", "gfx1250"),
)
BUILD_VARIANT = Lane(
    "build-variant",
    "a build or test environment pull-request CI does not provide: the C++ "
    "engine's Python modules (a build whose Python has pybind11), the shared "
    "librocke (the Windows provider build installs none), or the optional "
    "packages and LLVM tools the CI test step does not install",
)
TICKET = Lane(
    "ticket",
    "work its record's issue tracks: a CTest entry or tier the tests need, a "
    "scheduled run (the comprehensive tier, the exhaustive cohort), or a "
    "torch-free rewrite; the CI-run entries deselect exhaustive cases with "
    "-m 'not exhaustive'",
    marker="exhaustive",
)
SOURCE_TREE = Lane(
    "source-tree",
    "a source checkout (run_all.py): the cases read platform tools, C++ "
    "sources, docs or the git history, which the installed test tree omits, or "
    "carry the integration marker the installed entries deselect with "
    "-m 'not integration'",
    marker="integration",
)

LANES: tuple[Lane, ...] = (RUNNER_HARDWARE, BUILD_VARIANT, TICKET, SOURCE_TREE)
_BY_NAME = {lane.name: lane for lane in LANES}

# The prefix a declared skip reason starts with: the lane names, then ": ".
_WAITS_FOR = "waits for "
_DECLARED = re.compile(r"waits for ([a-z0-9-]+(?:, [a-z0-9-]+)*): ")


def skip_until_reason(*lanes: Lane, reason: str) -> str:
    """The skip reason of a case that waits for ``lanes`` (what CI lacks), for
    ``reason``: ``waits for <lane>[, <lane>...]: <reason>``. For a skip pytest
    or unittest raises from a reason: ``skipif``, ``skipIf``, a parameter's
    ``skip`` mark, a recorded deselection."""
    if not lanes:
        raise ValueError(
            f"skip_until_reason failed - reason={reason!r}, expected at least one lane"
        )
    if len(set(lanes)) != len(lanes):
        raise ValueError(
            f"skip_until_reason failed - lanes={[lane.name for lane in lanes]}, "
            "expected each lane once"
        )
    for lane in lanes:
        if _BY_NAME.get(getattr(lane, "name", None)) is not lane:
            raise ValueError(
                f"skip_until_reason failed - lane={lane!r}, expected one of "
                "rocke.tests.lanes.LANES"
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
    for :data:`BUILD_VARIANT`, because no shared C++ engine was found;
    fail it instead where the run requires the C++ engine
    (:func:`cpp_engine_required`), so such a run cannot pass without it."""
    __tracebackhide__ = True  # pytest reports the outcome at the caller
    if cpp_engine_required():
        pytest.fail(
            f"{REQUIRE_CPP_ENGINE_ENV}=1 requires the C++ engine: {reason}",
            pytrace=False,
        )
    skip_until(BUILD_VARIANT, reason=reason)


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


def skip_until_installed(requirements: Path, lane: Lane) -> None:
    """Skip the running case (or module) as waiting for ``lane`` unless every
    distribution the pip requirements file ``requirements`` lists is installed
    at a version its specifier allows. The file holds one requirement per line
    (``name>=1.0``); ``#`` starts a comment."""
    __tracebackhide__ = True  # pytest reports the skip at the caller
    unmet = []
    for line in requirements.read_text(encoding="utf-8").splitlines():
        text = line.partition("#")[0].strip()
        if not text:
            continue
        requirement = Requirement(text)
        try:
            found = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            unmet.append(f"{requirement.name} is not installed")
            continue
        if not requirement.specifier.contains(found, prereleases=True):
            unmet.append(f"{requirement.name} {found} does not satisfy {requirement}")
    if unmet:
        skip_until(lane, reason=f"{requirements.name}: {'; '.join(unmet)}")


def _is_missing_module(error: ModuleNotFoundError, module_name: str) -> bool:
    """Whether ``error`` says ``module_name`` (or a package it is in) is not
    installed, rather than that a module it imports is missing."""
    return error.name is not None and (
        module_name == error.name or module_name.startswith(f"{error.name}.")
    )


def runner_lanes_for(arches: Collection[str]) -> tuple[Lane, ...]:
    """The lanes a case that needs a GPU of one of ``arches`` waits for: the
    lane of their GPUs (:data:`RUNNER_HARDWARE`) when no pull-request runner has
    any of them. Empty when one of ``arches`` has none: a CI runner has it
    (gfx942), and the case runs there."""
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
    on none here: declaring the lane of their GPUs (:func:`runner_lanes_for`) when
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
