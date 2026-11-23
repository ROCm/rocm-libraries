# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Every test that does not run normally, with its recorded reason: test files
that wait for a CI lane, measured allowances that wait for a decision, and
quarantined tests that are known to fail.

A test is out of limbo only when a CI lane runs it, or when it has an explicit,
recorded reason not to yet. This module is that record: one entry per test file,
or test, whose lane does not run it in pull-request or push CI (the quick and standard
tiers), with the lane it waits for, why, who owns it and when the entry must be
re-decided. ``platform/tests/tools/test_pending_lanes.py`` holds the tree to it:
every test file is selected by a CI-run CTest entry or listed here, and no entry
here is stale.

Stale and expired are treated differently. A stale entry (its file is gone, or
a file waiting for a new lane is already in CI, or one waiting for something CI
lacks has left CI) always fails the check. An expired entry fails only when the
change under test touches this module or the expired entry's file; otherwise the
check reports it as a warning, so an entry reaching its date does not turn an
unrelated pull request red. A scheduled run passes ``--strict-expiry`` to fail
on every expired entry.

There are two kinds of lane. A lane with ``enrolled=False`` does not exist yet
(a scheduled run, a pull request that has not landed, a registration not yet
applied): its files are in no CI-run entry, and every case of them waits for
it. A lane with ``enrolled=True`` is something the CI-run entries lack (torch in
the test environment, a gfx950 or gfx1250 runner): its files are selected by a
CI-run entry, but their cases that need it skip there. Such a lane names the
skip reasons of the cases that wait for it (``Lane.skip_reasons``), so an entry
for a file covers those cases only, and a case that skips for another reason
needs a record of its own. A file whose cases lack several such things (a
gfx950 case and a gfx1250 case) has one entry per lane; a file waiting for a
lane that does not exist yet has exactly one entry. A file listed under a lane
that waits for torch must import torch.

    PendingTest(
        path="library/tests/test_example.py",  # relative to the rocke directory
        lane=WEEKLY_EXHAUSTIVE,
        reason="an exhaustive sweep, too slow for a pull-request tier",
        owner="rocKE library; issue: none filed yet",
        expires="2027-01-31",
    )

An entry may name one test of its file instead (``node``, as a quarantine
does), under a lane CI lacks: it covers that test's cases whatever their skip
reason, for a test whose reason does not name what it waits for, or that its
entry deselects:

    PendingTest(
        path="library/tests/test_example.py",
        lane=COMPREHENSIVE_CASES,
        reason="its sweep runs only in a comprehensive-tier entry",
        owner="rocKE library; issue: none filed yet",
        expires="2027-01-31",
        node="test_sweep",
    )

The file-level check cannot see single cases. ``tools/limbo_audit.py`` checks
them: from the run records (``rocke.tests.run_records``) of CI lanes, it fails
on each case no lane ran that :func:`waiting_entry` finds no record for.

A measured budget is the other kind of record: a test's resource check allows
one kernel the bytes one compiler is measured to spill, while the kernel owners
decide its retune. The allowance applies only on that compiler (matched by LLVM
major); on any other the test's strict rule holds. It carries the same reason,
owner and expiry, and the same check reports it stale or expired.

    MeasuredBudget(
        path="library/tests/test_example.py",  # the test file that applies it
        kernel_name="rocke_example_kernel",
        scratch_bytes=40,
        llvm_major=24,
        reason="spills 40 B at its VGPR cap on LLVM 24; LLVM 20 spills none",
        owner="kernel owners; issue: none filed yet",
        expires="2027-01-31",
    )

A quarantined test is known to fail (a flake, or a defect that turns develop
red) and is held out of every run until it is fixed. It names one pytest item,
or every parameter of one test function, and the issue that tracks the fix
(``NO_ISSUE_FILED`` until one is filed). A test that fails only on some GPU
arches names them, and is held out on those arches only. The pytest roots of
``platform/tests`` and ``library/tests`` skip it with the reason
``quarantined: <issue>`` (``rocke.tests.quarantine``), so it stays visible in
every report; ``ROCKE_QUARANTINE=ignore`` runs it anyway. Rules stricter than a
pending entry's hold it: an expired quarantine always fails the check, and at
most ``QUARANTINE_CAP`` tests may be quarantined at once.

    QuarantinedTest(
        test_id="platform/tests/test_example.py::test_case[gfx942]",
        issue="AICK-1234",  # a Jira id, ROCm/<repo>#<number> or NO_ISSUE_FILED
        reason="fails on every run since the recorder change",
        owner="rocKE platform",
        expires="2027-01-31",
        arches=("gfx950",),  # optional: the GPU arches it fails on
        runs_files=(),  # optional: test files it runs in a child process
    )

Paths, owners and issues are plain text: no links.
"""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..core.arch.target import known_arches

if TYPE_CHECKING:
    from ..runtime.comgr import CompilerInfo

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
    "MEASURED_BUDGETS",
    "MULTI_GPU_RUNNER",
    "NIGHTLY_GPU",
    "NIGHTLY_SOURCE",
    "NO_ISSUE_FILED",
    "OWNER_DECISION",
    "PENDING",
    "PR_16482",
    "QUARANTINED",
    "QUARANTINE_CAP",
    "QUARANTINE_SKIP_PREFIX",
    "QUARANTINE_TREES",
    "SHARED_LIBRARY_LANE",
    "SOURCE_TREE_CASES",
    "TEST_FILE_ROOTS",
    "TORCH_IN_CI",
    "WEEKLY_EXHAUSTIVE",
    "Lane",
    "MeasuredBudget",
    "PendingTest",
    "QuarantinedTest",
    "measured_budget_kernels",
    "measured_scratch_allowance",
    "quarantined_test",
    "tree_test_files",
    "waiting_entry",
]


@dataclass(frozen=True)
class Lane:
    """A CI lane that pending tests wait for."""

    name: str
    description: str
    # True when the lane's files are already selected by a CI-run entry and
    # wait for something that entry lacks; False when no CI-run entry selects
    # them.
    enrolled: bool
    # True when what the lane waits for is torch: each of its files imports
    # torch (``platform/tests/tools/test_pending_lanes.py`` checks it).
    waits_for_torch: bool = False
    # For a lane CI lacks (enrolled): regular expressions, one of which a case
    # waiting for it prints in its skip or deselection reason where CI runs
    # its file, so an entry for the whole file covers those cases only. A lane
    # CI lacks with none takes entries for single tests only. A lane that no
    # CI entry runs yet has none: every case of its files waits for it.
    skip_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.skip_reasons and not self.enrolled:
            raise ValueError(
                f"Lane failed - name={self.name!r}, skip_reasons="
                f"{self.skip_reasons!r}, expected none for a lane whose files are "
                "in no CI-run entry: every case of its files waits for it"
            )
        for pattern in self.skip_reasons:
            try:
                re.compile(pattern)
            except re.error as error:
                raise ValueError(
                    f"Lane failed - name={self.name!r}, skip reason {pattern!r} is "
                    f"not a regular expression: {error}"
                ) from None

    def explains(self, reason: str | None) -> bool:
        """Whether a case of a file listed under this lane waits for it, given
        why a CI lane did not run the case: its skip or deselection reason, or
        None when no CI lane collected it. A lane that no CI entry runs yet
        explains every case of its files; a lane CI lacks, the cases whose
        reason one of its ``skip_reasons`` matches."""
        if not self.enrolled:
            return True
        return reason is not None and any(
            re.search(pattern, reason) for pattern in self.skip_reasons
        )


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
BINDING_LANE = Lane(
    "binding-lane",
    "the C++ engine's Python modules in the CI test tree: the file is in a CI-run "
    "entry, but its cases skip where the build's Python had no pybind11",
    enrolled=True,
    skip_reasons=(r"rocke_(library_)?engine", r"C\+\+ engine", r"rocke_ir_lower"),
)
OWNER_DECISION = Lane(
    "owner-decision",
    "an owner decision between a lane with optional dependencies, a rewrite and a drop",
    enrolled=False,
)
TORCH_IN_CI = Lane(
    "torch-in-ci",
    "torch in the CI test environment: the file is in a CI-run entry, but its "
    "torch-checked cases skip there",
    enrolled=True,
    waits_for_torch=True,
    skip_reasons=(r"\btorch\b",),
)
GFX950_RUNNER = Lane(
    "gfx950-runner",
    "a CI GPU stage on gfx950: the file is in a CI-run entry, but its gfx950 cases "
    "skip on the gfx942 and gfx110X runners",
    enrolled=True,
    skip_reasons=(r"\bgfx950\b",),
)
GFX90A_RUNNER = Lane(
    "gfx90a-runner",
    "a CI GPU stage on gfx90a: the file is in a CI-run entry, but its gfx90a cases "
    "skip on the gfx942 and gfx110X runners",
    enrolled=True,
    skip_reasons=(r"\bgfx90a\b",),
)
GFX1250_RUNNER = Lane(
    "gfx1250-runner",
    "a CI GPU stage on gfx1250: the file is in a CI-run entry, but its gfx1250 "
    "cases skip on the gfx942 and gfx110X runners, and they have not run on "
    "gfx1250 hardware",
    enrolled=True,
    skip_reasons=(r"\bgfx1250\b",),
)
GFX1151_GFX12_RUNNER = Lane(
    "gfx1151-gfx12-runner",
    "a CI GPU stage on gfx1151, gfx1200 or gfx1201: the file is in a CI-run entry, "
    "but its cases for those arches skip on the gfx942 and gfx110X runners",
    enrolled=True,
    skip_reasons=(r"\bgfx1151\b", r"\bgfx120[01]\b", r"\bgfx12\b", r"\bRDNA ?4\b"),
)
SOURCE_TREE_CASES = Lane(
    "source-tree-cases",
    "a scheduled source-tree run (run_all.py), which no job runs yet: the file is "
    "in a CI-run entry of the installed test tree, but its cases that read the "
    "source tree (platform tools, C++ sources, docs, git history) skip there, or "
    "its integration-marked cases are deselected there",
    enrolled=True,
    skip_reasons=(
        r"source[- ]tree",
        r"source checkout",
        r"installed (test )?(layout|tree|payload|artifact)",
        r"\bomits\b",
        r"platform tools not found",
        r"platform-only tree",
        r"absent (here|from this)",
        r"deselected by -m '[^']*\bnot integration\b",
    ),
)
EXHAUSTIVE_CASES = Lane(
    "exhaustive-cases",
    "a scheduled exhaustive run (run_all.py --exhaustive), which no job runs yet: "
    "the file is in a CI-run entry, but its exhaustive cases are deselected there",
    enrolled=True,
    skip_reasons=(r"deselected by -m '[^']*\bnot exhaustive\b",),
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
    skip_reasons=(r"\btwo (\w+ )?GPUs\b",),
)
SHARED_LIBRARY_LANE = Lane(
    "shared-library-lane",
    "the shared librocke C library in the CI test tree: the file is in a CI-run "
    "entry, but its cases that load the library skip there, as the provider "
    "build installs none and the lane does not build one on demand",
    enrolled=True,
    skip_reasons=(r"\bshared librocke\b",),
)
CI_TEST_TOOLS = Lane(
    "ci-test-tools",
    "the optional packages and LLVM tools these cases use, in the CI test "
    "environment: the file is in a CI-run entry, but TheRock's test step installs "
    "no jsonschema or matplotlib and puts no LLVM tools on PATH, so they skip",
    enrolled=True,
    skip_reasons=(
        r"\bjsonschema\b",
        r"\bmatplotlib\b",
        r"\bllvm-objdump\b",
        r"\bllc\b",
    ),
)

LANES: tuple[Lane, ...] = (
    PR_16482,
    ENROLMENT,
    NIGHTLY_SOURCE,
    WEEKLY_EXHAUSTIVE,
    NIGHTLY_GPU,
    BINDING_LANE,
    OWNER_DECISION,
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


def _require_test_path(kind: str, path: str) -> None:
    if not path.endswith(".py") or path.startswith("/"):
        raise ValueError(
            f"{kind} failed - path={path!r}, expected a test file path relative "
            "to the rocke directory"
        )


def _require_recorded(
    kind: str, path: str, reason: str, owner: str, expires: str
) -> None:
    """Fail unless the reason and owner are plain text and expires is an ISO date."""
    for field_name, text in (("reason", reason), ("owner", owner)):
        if not text.strip():
            raise ValueError(
                f"{kind} failed - path={path!r}, {field_name} is empty, expected "
                "the recorded text"
            )
    if "://" in f"{reason} {owner}":
        raise ValueError(
            f"{kind} failed - path={path!r}, the reason or owner holds a link, "
            "expected plain text"
        )
    try:
        datetime.date.fromisoformat(expires)
    except ValueError:
        raise ValueError(
            f"{kind} failed - path={path!r}, expires={expires!r}, expected an ISO "
            "date (YYYY-MM-DD)"
        ) from None


@dataclass(frozen=True)
class PendingTest:
    """One test file, or one test of it, waiting for a lane, with its recorded
    reason.

    ``node`` names tests below the file (one case ``test_x[case]``, each case
    of ``test_x``, each test of a class ``TestX``; any of them below a class),
    as a quarantine does: the entry covers those cases, whatever their skip
    reason, under a lane CI lacks. Empty (the
    default), the entry covers the file, as its lane's kind says
    (:meth:`Lane.explains`).
    """

    path: str
    lane: Lane
    reason: str
    owner: str
    expires: str
    node: str = ""

    def __post_init__(self) -> None:
        _require_test_path("PendingTest", self.path)
        if self.lane not in LANES:
            raise ValueError(
                f"PendingTest failed - path={self.path!r}, lane={self.lane.name!r}, "
                "expected one of pending_lanes.LANES"
            )
        if self.node and not self.lane.enrolled:
            raise ValueError(
                f"PendingTest failed - path={self.path!r}, node={self.node!r}, "
                f"lane={self.lane.name!r}, expected a lane CI lacks for one test: "
                "a lane no CI entry runs yet takes whole files"
            )
        if not self.node and self.lane.enrolled and not self.lane.skip_reasons:
            raise ValueError(
                f"PendingTest failed - path={self.path!r}, lane={self.lane.name!r}, "
                "expected a node: the lane has no skip reasons, so it takes "
                "entries for single tests only"
            )
        _require_recorded(
            "PendingTest", self.path, self.reason, self.owner, self.expires
        )

    def covers(self, path: str, node_name: str, reason: str | None) -> bool:
        """Whether this entry records why no CI lane ran the case ``node_name``
        of ``path``, given ``reason`` (see :func:`waiting_entry`)."""
        if path != self.path:
            return False
        if self.node:
            return _names_case(self.node, node_name)
        return self.lane.explains(reason)

    @property
    def expiry_date(self) -> datetime.date:
        return datetime.date.fromisoformat(self.expires)


@dataclass(frozen=True)
class MeasuredBudget:
    """The scratch bytes one test allows one kernel on one compiler, pending a
    retune. What the bytes bound is the test's own check: total spill, or spill
    above a reference build of the same kernel."""

    path: str
    kernel_name: str
    scratch_bytes: int
    # The LLVM major of the compiler the bytes were measured on; on any other
    # compiler the allowance is 0.
    llvm_major: int
    reason: str
    owner: str
    expires: str

    def __post_init__(self) -> None:
        _require_test_path("MeasuredBudget", self.path)
        if not self.kernel_name.strip():
            raise ValueError(
                f"MeasuredBudget failed - path={self.path!r}, kernel_name is empty, "
                "expected the exact kernel name"
            )
        for field_name in ("scratch_bytes", "llvm_major"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"MeasuredBudget failed - kernel_name={self.kernel_name!r}, "
                    f"{field_name}={value!r}, expected a positive integer"
                )
        _require_recorded(
            "MeasuredBudget", self.path, self.reason, self.owner, self.expires
        )

    @property
    def expiry_date(self) -> datetime.date:
        return datetime.date.fromisoformat(self.expires)


# The pytest roots that skip quarantined tests, relative to the rocke directory.
QUARANTINE_TREES: tuple[str, ...] = ("platform/tests", "library/tests")
# At most this many tests are quarantined at once, so quarantine cannot become
# the backlog: at the cap, a new entry needs one cleared first.
QUARANTINE_CAP = 10
# The issue of a quarantine whose tracking issue is not filed yet.
NO_ISSUE_FILED = "none filed yet"
# A quarantined test's skip reason: this, then its issue.
QUARANTINE_SKIP_PREFIX = "quarantined: "
# Where test files live, relative to the rocke directory.
TEST_FILE_ROOTS: tuple[str, ...] = ("platform", "library")
# A Jira id (AICK-1234), a GitHub issue (ROCm/rocm-libraries#123) or
# NO_ISSUE_FILED.
_ISSUE = re.compile(
    rf"[A-Z][A-Z0-9]+-[0-9]+|ROCm/[A-Za-z0-9._-]+#[0-9]+|{re.escape(NO_ISSUE_FILED)}"
)


@dataclass(frozen=True)
class QuarantinedTest:
    """One test known to fail, skipped in every run until its issue is fixed.

    ``test_id`` is the test file relative to the rocke directory, ``::`` and the
    pytest node name: ``...::test_x[gfx942]`` names one parameter of
    ``test_x``, ``...::test_x`` all of them, ``...::TestX`` each test of a
    class. ``arches`` names the GPU arches it fails on, where it is held out;
    empty (the default) holds it out everywhere. ``runs_files`` names the test
    files the quarantined test runs in a child process, which it holds out
    with it.
    """

    test_id: str
    issue: str
    reason: str
    owner: str
    expires: str
    arches: tuple[str, ...] = ()
    runs_files: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        path, separator, node_name = self.test_id.partition("::")
        if not separator or not node_name:
            raise ValueError(
                f"QuarantinedTest failed - test_id={self.test_id!r}, expected "
                "<test file>::<pytest node name>"
            )
        _require_test_path("QuarantinedTest", path)
        if not path.startswith(tuple(f"{tree}/" for tree in QUARANTINE_TREES)):
            raise ValueError(
                f"QuarantinedTest failed - test_id={self.test_id!r}, expected a "
                f"test under one of {QUARANTINE_TREES}, whose pytest roots skip "
                "quarantined tests"
            )
        if not _ISSUE.fullmatch(self.issue):
            raise ValueError(
                f"QuarantinedTest failed - test_id={self.test_id!r}, "
                f"issue={self.issue!r}, expected a Jira id (AICK-1234), "
                f"ROCm/<repo>#<number> or {NO_ISSUE_FILED!r}"
            )
        _require_recorded(
            "QuarantinedTest", path, self.reason, self.owner, self.expires
        )
        if not isinstance(self.arches, tuple) or not set(self.arches) <= set(
            known_arches()
        ):
            raise ValueError(
                f"QuarantinedTest failed - test_id={self.test_id!r}, "
                f"arches={self.arches!r}, expected a tuple of known gfx arches"
            )
        if not isinstance(self.runs_files, tuple):
            raise TypeError(
                f"QuarantinedTest failed - test_id={self.test_id!r}, runs_files="
                f"{self.runs_files!r}, expected a tuple of test file paths"
            )
        for child in self.runs_files:
            _require_test_path("QuarantinedTest", child)

    @property
    def path(self) -> str:
        """The test file, relative to the rocke directory."""
        return self.test_id.partition("::")[0]

    @property
    def node_name(self) -> str:
        """The pytest node name after the file: ``[Class::]test[params]``."""
        return self.test_id.partition("::")[2]

    @property
    def expiry_date(self) -> datetime.date:
        return datetime.date.fromisoformat(self.expires)

    def holds_out_on(self, arch: str | None) -> bool:
        """Whether this entry holds its test out on a GPU of ``arch`` (None: no
        GPU): always when it names no arches, else only on one it names."""
        return not self.arches or arch in self.arches

    def covers(self, path: str, node_name: str) -> bool:
        """Whether this entry quarantines the item ``node_name`` of ``path``."""
        return path == self.path and _names_case(self.node_name, node_name)


def _names_case(test: str, node_name: str) -> bool:
    """Whether the test node name ``test`` names the pytest item ``node_name``:
    ``test_x[case]`` that case, ``test_x`` each of its cases, a class each test
    below it (any of them below a class)."""
    return (
        node_name == test
        or node_name.startswith(f"{test}::")
        or ("[" not in test and node_name.startswith(f"{test}["))
    )


def quarantined_test(path: str, node_name: str) -> QuarantinedTest | None:
    """The ``QUARANTINED`` entry for the pytest item ``node_name`` of the test
    file ``path`` (relative to the rocke directory), or None."""
    return next((entry for entry in QUARANTINED if entry.covers(path, node_name)), None)


def _waiting(
    lane: Lane, reason: str, owner: str, expires: str, *paths: str
) -> tuple[PendingTest, ...]:
    """One entry per path, all waiting for ``lane`` for the same reason."""
    return tuple(PendingTest(path, lane, reason, owner, expires) for path in paths)


PENDING: tuple[PendingTest, ...] = (
    *_waiting(
        PR_16482,
        "never registered; the draft adds them to its attention backward CPU or "
        "gfx942 GPU entry, with the slow tests cut down",
        "rocKE library attention; issues AICK-3259, AICK-3218",
        "2026-12-31",
        "library/tests/dispatch/attention_bwd/test_dense_tuning_registry.py",
        "library/tests/dispatch/attention_bwd/test_gfx950_tuning.py",
        "library/tests/dispatch/attention_bwd/test_known_wrong_knobs.py",
        "library/tests/dispatch/attention_bwd/test_registry.py",
        "library/tests/dispatch/attention_bwd/test_serving_policy.py",
        "library/tests/dispatch/attention_bwd/test_tuned_table_import_check.py",
        "library/tests/dispatch/attention_bwd/test_tuned_tables.py",
        "library/tests/dispatch/attention_bwd/test_tuning_registry.py",
        "library/tests/dispatch/attention_bwd/test_unified_tuning.py",
        "library/tests/test_attention_bwd.py",
        "library/tests/test_attention_bwd_access.py",
        "library/tests/test_attention_bwd_addr.py",
        "library/tests/test_attention_bwd_aux.py",
        "library/tests/test_attention_bwd_band.py",
        "library/tests/test_attention_bwd_barrier_merge.py",
        "library/tests/test_attention_bwd_bench.py",
        "library/tests/test_attention_bwd_caps.py",
        "library/tests/test_attention_bwd_comgr.py",
        "library/tests/test_attention_bwd_compile.py",
        "library/tests/test_attention_bwd_dispatch.py",
        "library/tests/test_attention_bwd_dispatch_numeric.py",
        "library/tests/test_attention_bwd_dq_levers.py",
        "library/tests/test_attention_bwd_frag.py",
        "library/tests/test_attention_bwd_fwd_chain.py",
        "library/tests/test_attention_bwd_host_checks.py",
        "library/tests/test_attention_bwd_hsaco_cache.py",
        "library/tests/test_attention_bwd_kernel_cases.py",
        "library/tests/test_attention_bwd_lds.py",
        "library/tests/test_attention_bwd_mirror.py",
        "library/tests/test_attention_bwd_mutations.py",
        "library/tests/test_attention_bwd_noatomic.py",
        "library/tests/test_attention_bwd_numeric.py",
        "library/tests/test_attention_bwd_paged_guard.py",
        "library/tests/test_attention_bwd_perf_gate.py",
        "library/tests/test_attention_bwd_plan.py",
        "library/tests/test_attention_bwd_refusals.py",
        "library/tests/test_attention_bwd_rocke_arm.py",
        "library/tests/test_attention_bwd_run.py",
        "library/tests/test_attention_bwd_sched.py",
        "library/tests/test_attention_bwd_shared_helpers.py",
        "library/tests/test_attention_bwd_sinks.py",
        "library/tests/test_attention_bwd_sweep.py",
        "library/tests/test_attention_bwd_tiles.py",
        "library/tests/test_attention_bwd_unified_mutations.py",
        "library/tests/test_attention_bwd_unified_numeric.py",
        "library/tests/test_attention_bwd_unified_plan.py",
        "library/tests/test_attention_bwd_unit_ops.py",
        "library/tests/test_attention_bwd_worklist.py",
        "library/tests/test_attention_bwd_xt.py",
        "library/tests/test_attention_dense_bwd.py",
        "library/tests/test_attention_dense_bwd_ir.py",
        "library/tests/test_attention_dense_bwd_numeric.py",
        "library/tests/test_attention_dense_bwd_split.py",
        "library/tests/test_lse_store_ir.py",
        "library/tests/test_lse_store_numeric.py",
    ),
    *_waiting(
        WEEKLY_EXHAUSTIVE,
        "in the exhaustive cohort (library/tests/_affordable_lane.py): too slow for "
        "a pull-request tier; pull request 16482 adds a sample of it to standard",
        "rocKE library attention; issue AICK-3259",
        "2027-01-31",
        "library/tests/test_attention_bwd_agpr_probe.py",
        "library/tests/test_attention_bwd_barrier_fence.py",
        "library/tests/test_attention_bwd_dq.py",
        "library/tests/test_attention_bwd_dq_sched.py",
        "library/tests/test_attention_bwd_epilogue.py",
        "library/tests/test_attention_bwd_ir.py",
        "library/tests/test_attention_bwd_ir_golden.py",
        "library/tests/test_attention_bwd_ring.py",
        "library/tests/test_attention_bwd_ring_sched.py",
        "library/tests/test_attention_bwd_sched_body.py",
        "library/tests/test_attention_bwd_split.py",
        "library/tests/test_attention_bwd_split_sched.py",
        "library/tests/test_attention_bwd_transpose_probe.py",
        "library/tests/test_attention_bwd_unified.py",
        "library/tests/test_attention_bwd_valu.py",
        "library/tests/test_attention_bwd_xt_probe.py",
        "library/tests/test_barrier_fence_parity_emit.py",
    ),
    *_waiting(
        NIGHTLY_GPU,
        "per-lever device twins of the backward kernel, left out of pull request "
        "16482's GPU entry; they need a GPU run on gfx942 and gfx950 first",
        "rocKE library attention; issue AICK-3259",
        "2027-01-31",
        "library/tests/test_attention_bwd_aux_numeric.py",
        "library/tests/test_attention_bwd_gemm_numeric.py",
        "library/tests/test_attention_bwd_ring_numeric.py",
        "library/tests/test_attention_bwd_sched_numeric.py",
        "library/tests/test_attention_bwd_split_numeric.py",
        "library/tests/test_attention_bwd_tiles_numeric.py",
        "library/tests/test_attention_bwd_valu_numeric.py",
    ),
    *_waiting(
        BINDING_LANE,
        "C++ engine parity checks that need the rocke_engine module; every case "
        "skips without it, until TheRock's build environment has pybind11",
        "rocKE platform engine; issue: none filed yet",
        "2027-01-31",
        "library/tests/test_attention_ir_cpp_parity.py",
        "library/tests/test_gdn_decode_ir_cpp_parity.py",
        "library/tests/test_selector_cpp_parity.py",
    ),
    *_waiting(
        NIGHTLY_SOURCE,
        "source-tree only: it reads cpp/, tools/ or dsl_docs/, or needs a native "
        "build, so the install leaves it out or it skips there; run_all.py runs it",
        "rocKE test infrastructure; issue: none filed yet",
        "2027-01-31",
        "library/tests/test_attention_bwd_counted_sync.py",
        "library/tests/test_attention_bwd_resources.py",
        "library/tests/test_dsl_probe_attention_demos.py",
        "platform/python/rocke/benchmark/perf/tests/test_wavescope_pmc.py",
        "platform/tests/instances/differential/test_canonical_text.py",
        "platform/tests/instances/differential/test_check_byte_identity_progress.py",
        "platform/tests/instances/differential/test_gate_verdict.py",
        "platform/tests/instances/differential/test_gfx950_scaled_mfma_emit.py",
        "platform/tests/instances/differential/test_golden_store.py",
        "platform/tests/test_engine_linkage.py",
        "platform/tests/test_gpu_references.py",
        "platform/tests/tools/test_arch_domain_artifact.py",
        "platform/tests/tools/test_check_mma_catalog.py",
        "platform/tests/tools/test_dsl_probes.py",
        "platform/tests/tools/test_lds_analyzer.py",
        "platform/tests/tools/test_wavescope_sidecar.py",
    ),
    *_waiting(
        NIGHTLY_SOURCE,
        "every test carries the integration marker (pytestmark), which the "
        "installed entries deselect with -m 'not integration'; run_all.py runs it",
        "rocKE test infrastructure; issue: none filed yet",
        "2027-01-31",
        "platform/tests/tools/test_provider_install_contract.py",
        "platform/tests/tools/test_sdpa_asset_cmake_contract.py",
    ),
    *_waiting(
        NIGHTLY_SOURCE,
        "registered only in rocke_tiling_visualization_pytest, a "
        "comprehensive-tier entry (about 44 s, low importance); no scheduled "
        "job runs that tier yet",
        "rocKE tiling; issue: none filed yet",
        "2027-01-31",
        "platform/python/rocke/tests/helpers/tiling/visualization/test_auto_pipeline.py",
        "platform/python/rocke/tests/helpers/tiling/visualization/test_block_diagram.py",
        "platform/python/rocke/tests/helpers/tiling/visualization/test_layout_render.py",
        "platform/python/rocke/tests/helpers/tiling/visualization/test_layout_visualizer.py",
    ),
    *_waiting(
        ENROLMENT,
        "registered only in comprehensive-tier entries (runner contract, worker "
        "integration); triage proposes the standard tier",
        "rocKE test infrastructure; issue: none filed yet",
        "2026-12-15",
        "library/tests/test_library_test_tiers.py",
    ),
    *_waiting(
        OWNER_DECISION,
        "needs pandas, scikit-learn, lightgbm and pyarrow, which no CI lane "
        "installs: a nightly lane with the dependencies, or leave them out",
        "rocKE heuristics; issue: none filed yet",
        "2026-12-15",
        "platform/python/rocke/heuristics/tests/test_data_pipeline.py",
        "platform/python/rocke/heuristics/tests/test_evaluate.py",
        "platform/python/rocke/heuristics/tests/test_feature_engine.py",
        "platform/python/rocke/heuristics/tests/test_feature_engine_grouped_conv.py",
        "platform/python/rocke/heuristics/tests/test_feature_parity.py",
        "platform/python/rocke/heuristics/tests/test_predict.py",
        "platform/python/rocke/heuristics/tests/test_search.py",
        "platform/python/rocke/heuristics/tests/test_train.py",
    ),
    *_waiting(
        TORCH_IN_CI,
        "its GPU numerics compare against a torch reference; without torch in the "
        "CI test environment those cases skip",
        "rocKE library; torch in the CI test environment or a numpy reference "
        "(issue AICK-3334 for attention)",
        "2027-01-31",
        "library/tests/sdpa/test_dense_sdpa_numeric.py",
        "library/tests/sdpa/test_unified_sdpa_numeric.py",
    ),
    *_waiting(
        TORCH_IN_CI,
        "its graph-replay case captures through torch; without torch in the CI "
        "test environment that case skips",
        "rocKE library attention",
        "2027-01-31",
        "library/tests/test_attention_unified_lse_numeric.py",
    ),
    *_waiting(
        TORCH_IN_CI,
        "its torch-caller cases (graph replay, the concurrent workspace on torch "
        "streams, the prepared decode lifecycle) take torch tensors; without torch "
        "in the CI test environment those cases skip",
        "rocKE library attention",
        "2027-01-31",
        "library/tests/test_strided_kv_decode_numeric.py",
    ),
    *_waiting(
        GFX950_RUNNER,
        "its device cases need gfx950, which no CI GPU stage has",
        "rocKE library; issue: none filed yet",
        "2027-01-31",
        "library/tests/test_attention_dense_gfx950_numeric.py",
        "library/tests/test_attention_tiled_2d_gfx950_biased_numeric.py",
        "library/tests/test_attention_tiled_3d_kv_lds_swizzle.py",
        "library/tests/test_attention_tiled_3d_wave_page_id_load.py",
        "library/tests/test_attention_unified_lse_numeric.py",
        "library/tests/test_attention_unified_mask_gfx950_numeric.py",
        "library/tests/test_attention_unified_swa_sink_gfx950_numeric.py",
        "library/tests/test_attn_tiled_2d_gfx950_fp8_k_lds_numeric.py",
        "library/tests/test_conv_dgrad_4c.py",
        "library/tests/test_conv_dgrad_correctness.py",
        "library/tests/test_conv_dgrad_depthwise_windowed.py",
        "library/tests/test_conv_fwd_split_k.py",
        "library/tests/test_conv_wgrad_correctness.py",
        "library/tests/test_convolution_forward_aot_numeric.py",
        "library/tests/test_direct_conv_correctness.py",
        "library/tests/test_direct_dgrad_pp_mfma.py",
        "library/tests/test_direct_fwd_flex_mfma.py",
        "library/tests/test_direct_fwd_pp_mfma.py",
        "library/tests/test_dsa_indexer_numeric.py",
        "library/tests/test_extended_parity_attention.py",
        "library/tests/test_fwd_small_mfma.py",
        "library/tests/test_gdn_decode_fused_gfx950_numeric.py",
        "library/tests/test_gdn_decode_gfx950_numeric.py",
        "library/tests/test_gdn_decode_knobs_gfx950_numeric.py",
        "library/tests/test_kda_chunkwise_gfx950_numeric.py",
        "library/tests/test_kda_decode_gfx950_numeric.py",
        "library/tests/test_kda_gdn_gfx950_numeric.py",
        "platform/python/rocke/tests/helpers/tiling/mma/test_mma_k_coupling.py",
    ),
    *_waiting(
        GFX950_RUNNER,
        "its device cases need gfx950, which no CI GPU stage has. Known defect: "
        "four test_additive_bias_with_features_matches_reference cases "
        "(tiled_window_1seq, bf16) fail on gfx950, with torch and without: the "
        "selector routes them to transposed32 where the route pin expects 16x16; "
        "the selector owner decides which is right",
        "rocKE library attention selector; issue: none filed yet",
        "2026-12-15",
        "library/tests/test_attention_additive_bias_gfx950_numeric.py",
    ),
    *_waiting(
        GFX950_RUNNER,
        "its device cases need gfx950 (the multi-GPU sweep, two of them), which no "
        "CI GPU stage has",
        "rocKE platform; issue: none filed yet",
        "2027-01-31",
        "platform/python/rocke/benchmark/gemm/tests/test_fp16_rcr_multigpu.py",
        "platform/tests/core/test_f8f6f4_16x16x128.py",
        "platform/tests/core/test_xf32_bf16x3_device.py",
        "platform/tests/instances/test_gfx950_mixed_scaled_gemm_numeric.py",
        "platform/tests/instances/test_gfx950_mxfp4_numeric.py",
        "platform/tests/instances/test_gfx950_mxfp6_numeric.py",
        "platform/tests/instances/test_gfx950_mxfp8_numeric.py",
    ),
    *_waiting(
        GFX1250_RUNNER,
        "its device cases need gfx1250, which no CI GPU stage has",
        "rocKE library; issue: none filed yet",
        "2027-01-31",
        "library/tests/test_conv_dgrad_correctness.py",
        "library/tests/test_conv_dgrad_depthwise_windowed.py",
        "library/tests/test_conv_fwd_split_k.py",
        "library/tests/test_conv_wgrad_correctness.py",
        "library/tests/test_conv_winograd_dispatch_gpu.py",
        "library/tests/test_conv_winograd_fwd_gpu.py",
        "library/tests/test_conv_winograd_fwd_rt_gpu.py",
        "library/tests/test_conv_winograd_gemm_wmma.py",
        "library/tests/test_conv_winograd_gen_gpu.py",
        "library/tests/test_dgrad_stem_wmma.py",
        "library/tests/test_direct_conv_correctness.py",
        "library/tests/test_direct_depthwise_fwd_cpl.py",
        "library/tests/test_direct_dgrad_wmma.py",
        "library/tests/test_direct_dw_fwd_wmma.py",
        "library/tests/test_direct_fwd_gstream_wmma.py",
        "library/tests/test_direct_fwd_tapouter_wmma.py",
        "library/tests/test_direct_fwd_wmma.py",
        "library/tests/test_fwd_small_wmma.py",
        "library/tests/test_gfx1250_attention.py",
    ),
    *_waiting(
        GFX1250_RUNNER,
        "its device cases need gfx1250, which no CI GPU stage has",
        "rocKE platform; issue: none filed yet",
        "2027-01-31",
        "platform/tests/instances/test_gfx1250_mixed_scaled_gemm_numeric.py",
        "platform/tests/instances/test_gfx1250_mxfp4_numeric.py",
        "platform/tests/instances/test_gfx1250_mxfp6_numeric.py",
        "platform/tests/instances/test_gfx1250_mxfp8_numeric.py",
        "platform/tests/instances/test_gfx1250_mxfp_examples.py",
        "platform/tests/instances/test_gfx1250_nvfp4_numeric.py",
        "platform/tests/instances/test_gfx1250_scale_formats_numeric.py",
    ),
    *_waiting(
        GFX1151_GFX12_RUNNER,
        "its device case runs on gfx1151 or gfx1201 only, which no CI GPU stage has",
        "rocKE platform; issue: none filed yet",
        "2027-01-31",
        "platform/tests/instances/test_matmul_nbits_numeric.py",
    ),
    *_waiting(
        GFX90A_RUNNER,
        "its GPU cases compile for gfx90a only (39 cases in these files); they "
        "pass on a gfx90a node",
        "rocKE tiling; issue: none filed yet",
        "2027-01-31",
        "platform/python/rocke/tests/helpers/tiling/kernels/test_tiling_gemm_interleaved_demo.py",
        "platform/python/rocke/tests/helpers/tiling/kernels/test_tiling_gemm_manual_demo.py",
        "platform/python/rocke/tests/helpers/tiling/test_fragment_storage.py",
        "platform/python/rocke/tests/helpers/tiling/test_lds_descriptor_addressing.py",
        "platform/python/rocke/tests/helpers/tiling/test_leading_axes.py",
    ),
    *_waiting(
        GFX90A_RUNNER,
        "its GPU case checks gfx90a instructions on gfx90a only",
        "rocKE platform; issue: none filed yet",
        "2027-01-31",
        "platform/tests/core/test_u8_arith.py",
        "platform/tests/numeric/test_gfx90a_mfma_ops.py",
    ),
)


_ATTENTION_BUILDS = "library/tests/test_attention_builds.py"
_GDN_DECODE_FUSED = "library/tests/test_gdn_decode_fused.py"
_COMPILER_20 = "ROCm 7.1 (AMD clang 20.0.0git, roc-7.1.0 25425)"
_COMPILER_24 = "AMD clang 24.0.0git (llvm-project 424729dba653)"
_SPILL_RETUNE_OWNER = "kernel owners; issue: none filed yet"
_SPILL_RETUNE_EXPIRES = "2026-12-31"


def _measured(
    path: str, llvm_major: int, reason: str, *allowances: tuple[str, int]
) -> tuple[MeasuredBudget, ...]:
    """One spill-retune budget per ``(kernel name, bytes)``, all for one reason."""
    return tuple(
        MeasuredBudget(
            path,
            kernel_name,
            scratch_bytes,
            llvm_major,
            reason,
            _SPILL_RETUNE_OWNER,
            _SPILL_RETUNE_EXPIRES,
        )
        for kernel_name, scratch_bytes in allowances
    )


MEASURED_BUDGETS: tuple[MeasuredBudget, ...] = (
    *_measured(
        _ATTENTION_BUILDS,
        20,
        "gfx942 fp16 D128 tiled-2d, the shipped default geometry (wpe2): "
        f"{_COMPILER_20} caps VGPR at 256 for its 2-wave tuning and spills this "
        f"many bytes; {_COMPILER_24} compiles it with no scratch",
        (
            "rocke_uattn2d_tiled_d128_b64_g4_fp16_w4_wpe2_mw32_mfma32x8_stqk_s1_mask1_hoist_mlim_kvcpall_cfvst_ksring_rd2",
            44,
        ),
    ),
    *_measured(
        _ATTENTION_BUILDS,
        20,
        "gfx942 fp16 D128 paged prefill at the BN=64 tile (wpe2; paged V takes the "
        f"DMA path): {_COMPILER_20} caps VGPR at 256 and spills this many bytes; "
        f"{_COMPILER_24} compiles it with no scratch",
        (
            "rocke_attention_dense_d128_hq32_kv8_bn64_fp16_sq8192_sk8192_causal_swa4096_pgd16_nb512_lazyrs_gfx942_b1_wpe2",
            28,
        ),
    ),
    *_measured(
        _ATTENTION_BUILDS,
        24,
        "gfx942 D128 paged prefill at the BN=32 tile (wpe2, window skip): "
        f"{_COMPILER_24} caps VGPR at 256 for its 2-wave tuning and spills this "
        "many bytes (fp16, bf16); ROCm 7.1 (LLVM 20) compiles both with no scratch",
        (
            "rocke_attention_dense_d128_hq16_kv4_bn32_fp16_bm128_sq1024_sk1024_causal_swa256_pgd16_nb128_lazyrs_gfx942_b1_wpe2_wskip",
            44,
        ),
        (
            "rocke_attention_dense_d128_hq16_kv4_bn32_bf16_bm128_sq1024_sk1024_causal_swa256_pgd16_nb128_lazyrs_gfx942_b1_wpe2_wskip",
            24,
        ),
    ),
    *_measured(
        _ATTENTION_BUILDS,
        24,
        "gfx942 dispatched bf16 D64 causal prefill (wpe4, K padded to 8): "
        f"{_COMPILER_24} caps VGPR at 128 and spills this many bytes (without, "
        "with LSE); ROCm 7.1 (LLVM 20) compiles both with no scratch",
        (
            "rocke_attention_dense_d64_bn64_bf16_kpad8_causal_lazyrs_npbthkvmin_gfx942_wpe4",
            40,
        ),
        (
            "rocke_attention_dense_d64_bn64_bf16_kpad8_causal_lazyrs_npbthkvmin_gfx942_wpe4_lse",
            44,
        ),
    ),
    *_measured(
        _GDN_DECODE_FUSED,
        24,
        "gfx950 single-wave conv_once decode, f32 state, WTK 4, conv only: "
        f"{_COMPILER_24} spills this many bytes more loading the rows first than "
        "late (0 B late); ROCm 7.1 (LLVM 20) spills neither",
        (
            "rocke_gdn_decode_bf16_kh16_vh16_dk128_dv128_w1k4b1_stf32_lhnt_shnt_l2_cv_rn_xcd_sr_co",
            36,
        ),
        (
            "rocke_gdn_decode_bf16_kh16_vh16_dk128_dv128_w1k4b1_stf32_lhnt_shnt_l2_cv_rn_xcd_sr_ic_co",
            36,
        ),
    ),
    *_measured(
        _GDN_DECODE_FUSED,
        24,
        "gfx950 single-wave conv_once decode, f32 state, WTK 4, KDA gate, conv and "
        f"norm with out_lds and norm_gate_once: {_COMPILER_24} spills this many "
        "bytes more loading the rows first than late (0 B late); ROCm 7.1 "
        "(LLVM 20) spills neither",
        (
            "rocke_gdn_decode_bf16_kh16_vh16_dk128_dv128_w1k4b1_stf32_kda_lhnt_shnt_l2_cv_rn_xcd_sr_co_ngo_olds",
            24,
        ),
        (
            "rocke_gdn_decode_bf16_kh16_vh16_dk128_dv128_w1k4b1_stf32_kda_lhnt_shnt_l2_cv_rn_xcd_sr_ic_co_ngo_olds",
            24,
        ),
    ),
)


def measured_budget_kernels(path: str) -> frozenset[str]:
    """The kernel names ``MEASURED_BUDGETS`` holds for the test file ``path``,
    whatever the compiler, so the test can fail a name it no longer builds."""
    return frozenset(
        budget.kernel_name for budget in MEASURED_BUDGETS if budget.path == path
    )


def measured_scratch_allowance(
    path: str, kernel_name: str, compiler: CompilerInfo | None
) -> int:
    """The scratch bytes the test file ``path`` allows ``kernel_name`` when
    built by ``compiler`` (``rocke.runtime.comgr.loaded_compiler_info()``): its
    measured budget when the compiler's LLVM major is the measured one, else 0."""
    if compiler is None or compiler.llvm_version is None:
        return 0
    return next(
        (
            budget.scratch_bytes
            for budget in MEASURED_BUDGETS
            if budget.path == path
            and budget.kernel_name == kernel_name
            and budget.llvm_major == compiler.llvm_version[0]
        ),
        0,
    )


def _quarantined(
    issue: str,
    reason: str,
    owner: str,
    expires: str,
    *test_ids: str,
    arches: tuple[str, ...] = (),
) -> tuple[QuarantinedTest, ...]:
    """One entry per test id, all quarantined for the same issue and reason (on
    ``arches`` only, when given)."""
    return tuple(
        QuarantinedTest(test_id, issue, reason, owner, expires, arches)
        for test_id in test_ids
    )


_RECORDER_REFUSES_IF_ELSE = (
    "the portable-IR recorder refuses scf.if_else with results, which the "
    "three conv_direct_*_mfma kernels use, so recording and replaying every "
    "production kernel fails"
)

QUARANTINED: tuple[QuarantinedTest, ...] = (
    QuarantinedTest(
        "platform/tests/portable_ir/test_portable_ir.py::test_unit[test_recording_builder]",
        "AICK-2863",
        _RECORDER_REFUSES_IF_ELSE,
        "rocKE platform portable IR",
        "2027-01-31",
        runs_files=(
            "platform/python/rocke/portable_ir/tests/test_recording_builder.py",
        ),
    ),
    *_quarantined(
        "AICK-2863",
        _RECORDER_REFUSES_IF_ELSE,
        "rocKE platform portable IR",
        "2027-01-31",
        "platform/tests/portable_ir/test_portable_ir.py::test_record_coverage",
        "platform/tests/portable_ir/test_portable_ir.py::test_parity_matrix",
    ),
    *_quarantined(
        NO_ISSUE_FILED,
        "built by the ROCm core-10.1 COMGR, the gfx950-only dgrad kouter and ukd "
        "kernels (16x16x32 and 32x32x16 atoms) fault with an illegal memory "
        "access or miscompare (rel_err 1.17, depthwise 11x3); the same kernels "
        "verify when built by the ROCm 7.1 or 7.2 COMGR on the same gfx950 node "
        "(a COMGR bisect of this file's tests)",
        "kernel owners",
        "2026-12-15",
        "library/tests/test_conv_dgrad_correctness.py::TestConvDgradCorrectness::test_bf16_depthwise_large_filter",
        "library/tests/test_conv_dgrad_correctness.py::TestConvDgradCorrectness::test_bf16_stride1",
        "library/tests/test_conv_dgrad_correctness.py::TestConvDgradCorrectness::test_fp16_resnet_shape",
        "library/tests/test_conv_dgrad_correctness.py::TestConvDgradCorrectness::test_fp16_stride1",
        arches=("gfx950",),
    ),
)


def waiting_entry(
    path: str, node_name: str, reason: str | None
) -> PendingTest | QuarantinedTest | None:
    """The record of why no CI lane ran the case ``node_name`` of the test file
    ``path`` (relative to the rocke directory), or None when nothing records it.

    ``reason`` is why a CI lane that collected the case did not run it: its
    skip or deselection reason, or None when no CI lane collected it. The case
    is recorded by its quarantine, or that of the test that runs its file
    (one that names arches, only where it was skipped as quarantined), or by
    an entry that covers it
    (:meth:`PendingTest.covers`): one for the test, or one for its file whose
    lane explains ``reason``.
    """
    quarantine = quarantined_test(path, node_name) or next(
        (entry for entry in QUARANTINED if path in entry.runs_files), None
    )
    if quarantine is not None and (
        not quarantine.arches
        or (reason is not None and reason.startswith(QUARANTINE_SKIP_PREFIX))
    ):
        return quarantine
    return next(
        (entry for entry in PENDING if entry.covers(path, node_name, reason)), None
    )


def tree_test_files(rocke_dir: Path) -> set[str]:
    """Every test file of the source tree at ``rocke_dir``, relative to it."""
    return {
        file.relative_to(rocke_dir).as_posix()
        for root in TEST_FILE_ROOTS
        for file in (rocke_dir / root).rglob("test_*.py")
        if "__pycache__" not in file.parts
    }
