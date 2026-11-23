# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Every test that does not run normally, with its recorded reason: test files
that wait for a CI lane, measured allowances that wait for a decision, and
quarantined tests that are known to fail.

A test is out of limbo only when a CI lane runs it, or when it has an explicit,
recorded reason not to yet. This module is that record: one entry per test file,
or test, whose lane (``rocke.tests.lanes``) does not run it in pull-request or
push CI (the quick and standard tiers), with the lane it waits for, why, who
owns it and when the entry must be re-decided.
``platform/tests/tools/test_pending_lanes.py`` holds the tree to it: every test
file is selected by a CI-run CTest entry or listed here, and no entry here is
stale.

Stale and expired are treated differently. A stale entry (its file is gone, or
a file waiting for a new lane is already in CI, or one waiting for something CI
lacks has left CI) always fails the check. An expired entry fails only when the
change under test touches this module or the expired entry's file; otherwise the
check reports it as a warning, so an entry reaching its date does not turn an
unrelated pull request red. A scheduled run passes ``--strict-expiry`` to fail
on every expired entry.

An entry for a file under a lane no CI entry runs yet (``Lane.enrolled`` False)
covers every case of the file. One under a lane CI lacks (torch, a gfx950
runner) covers only the cases that declare that lane where they skip
(``rocke.tests.lanes.skip_until``): a case that skips for any other reason is
not covered, whatever its reason says. A file whose cases wait for several
things CI lacks has one entry per lane; a file waiting for a lane that does not
exist yet has exactly one entry. A file listed under a lane that waits for
torch must import torch.

    PendingTest(
        path="library/tests/test_example.py",  # relative to the rocke directory
        lane=GFX950_RUNNER,
        reason="its device cases need gfx950, which no CI GPU stage has",
        owner="rocKE library; issue: none filed yet",
        expires="2027-01-31",
    )

An entry may name one test of its file instead (``node``, as a quarantine
does), under a lane CI lacks: it covers that test's cases whatever their skip
reason, for a test that CI does not select, or whose skip cannot name its lane:

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

Measured budgets and quarantines apply to some compiler builds only; both name
them the same way (:class:`Compilers`): by LLVM major, by the llvm-project
commits of the builds, or by both.

A measured budget is the other kind of record: a test's resource check allows
one kernel the bytes the named compilers are measured to spill, while the
kernel owners decide its retune. With any other compiler the test's strict rule
holds. It carries the same reason, owner and expiry, and the same check reports
it stale or expired.

    MeasuredBudget(
        path="library/tests/test_example.py",  # the test file that applies it
        kernel_name="rocke_example_kernel",
        scratch_bytes=40,
        compilers=Compilers(llvm_major=24),
        reason="spills 40 B at its VGPR cap on LLVM 24; LLVM 20 spills none",
        owner="kernel owners; issue: none filed yet",
        expires="2027-01-31",
    )

A quarantined test is known to fail (a flake, or a defect that turns develop
red) and is held out of every run until it is fixed. It names one pytest item,
or every parameter of one test function, and the issue that tracks the fix
(``NO_ISSUE_FILED`` until one is filed). A test that fails only on some GPU
arches names them, and is held out on those arches only; one that fails only
when built by some compilers (a defect of one compiler build) names them, and
is held out only while the loaded COMGR's compiler is one of them. The pytest
roots of ``platform/tests`` and ``library/tests`` skip it with the reason
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
        compilers=Compilers(llvm_commits=("424729dba653",)),  # optional
    )

Paths, owners and issues are plain text: no links.
"""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass
from pathlib import Path

from ..core.arch.target import known_arches
from .comgr import CompilerBuild
from .lanes import (
    BINDING_LANE,
    CI_TEST_TOOLS,
    COMPREHENSIVE_CASES,
    ENROLMENT,
    EXHAUSTIVE_CASES,
    GFX90A_RUNNER,
    GFX950_RUNNER,
    GFX1151_GFX12_RUNNER,
    GFX1250_RUNNER,
    LANES,
    MULTI_GPU_RUNNER,
    NIGHTLY_GPU,
    NIGHTLY_SOURCE,
    OWNER_DECISION,
    PR_16482,
    SHARED_LIBRARY_LANE,
    SOURCE_TREE_CASES,
    TORCH_IN_CI,
    WEEKLY_EXHAUSTIVE,
    Lane,
    lanes_waited_for,
)

__all__ = [
    "MEASURED_BUDGETS",
    "NO_ISSUE_FILED",
    "PENDING",
    "QUARANTINED",
    "QUARANTINE_CAP",
    "QUARANTINE_SKIP_PREFIX",
    "QUARANTINE_TREES",
    "TEST_FILE_ROOTS",
    "Compilers",
    "MeasuredBudget",
    "PendingTest",
    "QuarantinedTest",
    "measured_budget_kernels",
    "measured_scratch_allowance",
    "quarantined_test",
    "tree_test_files",
    "waiting_entry",
]


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
    as a quarantine does, under a lane CI lacks: the entry covers those cases
    where no CI entry selects them (not collected, or deselected), or where
    they declare its lane. Empty (the default), the entry covers each case of
    the file under a lane no CI entry runs yet, and each case that declares its
    lane under a lane CI lacks (:meth:`covers`). No entry covers a case that
    skipped without declaring its lane.
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
                "expected one of rocke.tests.lanes.LANES"
            )
        if self.node and not self.lane.enrolled:
            raise ValueError(
                f"PendingTest failed - path={self.path!r}, node={self.node!r}, "
                f"lane={self.lane.name!r}, expected a lane CI lacks for one test: "
                "a lane no CI entry runs yet takes whole files"
            )
        _require_recorded(
            "PendingTest", self.path, self.reason, self.owner, self.expires
        )

    def covers(
        self,
        path: str,
        node_name: str,
        reason: str | None,
        *,
        deselected: bool = False,
    ) -> bool:
        """Whether this entry records why no CI lane ran the case ``node_name``
        of ``path``, given ``reason`` and whether a CI lane ``deselected`` it
        (see :func:`waiting_entry`): under a lane no CI entry runs yet, any
        case of its file; else a case whose reason declares its lane
        (``rocke.tests.lanes.lanes_waited_for``), of its file or of the test it
        names, or a case of that test no CI entry selects."""
        if path != self.path:
            return False
        if not self.lane.enrolled:
            return True
        declared = reason is not None and self.lane in lanes_waited_for(reason)
        if not self.node:
            return declared
        return _names_case(self.node, node_name) and (
            declared or reason is None or deselected
        )

    @property
    def expiry_date(self) -> datetime.date:
        return datetime.date.fromisoformat(self.expires)


# An llvm-project commit: at least 12 of its lowercase hex digits.
_LLVM_COMMIT = re.compile(r"[0-9a-f]{12,40}")


@dataclass(frozen=True)
class Compilers:
    """The compiler builds a measured budget or a quarantine applies to: those
    of LLVM major ``llvm_major``, those built from one of the llvm-project
    commits ``llvm_commits`` (each at least 12 hex digits), or, when both are
    given, those that are both. Two builds of one LLVM major can differ (one
    spills where the other does not), so a record measured on one build names
    its commit."""

    llvm_major: int | None = None
    llvm_commits: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.llvm_major is None and not self.llvm_commits:
            raise ValueError(
                "Compilers failed - llvm_major=None, llvm_commits=(), expected an "
                "LLVM major, llvm-project commits or both"
            )
        if self.llvm_major is not None and (
            isinstance(self.llvm_major, bool)
            or not isinstance(self.llvm_major, int)
            or self.llvm_major <= 0
        ):
            raise ValueError(
                f"Compilers failed - llvm_major={self.llvm_major!r}, expected a "
                "positive integer"
            )
        if not isinstance(self.llvm_commits, tuple) or not all(
            isinstance(commit, str) and _LLVM_COMMIT.fullmatch(commit)
            for commit in self.llvm_commits
        ):
            raise ValueError(
                f"Compilers failed - llvm_commits={self.llvm_commits!r}, expected a "
                "tuple of llvm-project commits, each 12 to 40 lowercase hex digits"
            )

    def include(self, build: CompilerBuild) -> bool:
        """Whether ``build`` (``rocke.tests.comgr.loaded_compiler_build()``) is
        one of these compilers; a build whose major or commit is unknown is
        none of those that name it."""
        if self.llvm_major is not None and build.llvm_major != self.llvm_major:
            return False
        return not self.llvm_commits or (
            build.llvm_commit is not None
            and build.llvm_commit.startswith(self.llvm_commits)
        )


@dataclass(frozen=True)
class MeasuredBudget:
    """The scratch bytes one test allows one kernel on some compilers, pending a
    retune. What the bytes bound is the test's own check: total spill, or spill
    above a reference build of the same kernel."""

    path: str
    kernel_name: str
    scratch_bytes: int
    # The compilers the bytes were measured on; on any other the allowance is 0.
    compilers: Compilers
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
        if (
            isinstance(self.scratch_bytes, bool)
            or not isinstance(self.scratch_bytes, int)
            or self.scratch_bytes <= 0
        ):
            raise ValueError(
                f"MeasuredBudget failed - kernel_name={self.kernel_name!r}, "
                f"scratch_bytes={self.scratch_bytes!r}, expected a positive integer"
            )
        if not isinstance(self.compilers, Compilers):
            raise TypeError(
                f"MeasuredBudget failed - kernel_name={self.kernel_name!r}, "
                f"compilers={self.compilers!r}, expected a Compilers"
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
    with it. ``compilers`` names the compiler builds it fails with, and holds
    it out only while the loaded COMGR's compiler is one of them
    (``rocke.tests.comgr.loaded_compiler_build``); None (the default) holds it
    out whatever the compiler. ``params`` narrows a parametrized test to the
    cases whose parameter ids include every one of them: ``("bf16",)`` on
    ``...::test_x`` holds out ``test_x[case_a-bf16-mha]`` and
    ``test_x[case_b-bf16-gqa]``, one record for one defect across several
    cases. Each id is matched whole, between the ``-`` separators of the case's
    brackets; empty (the default) holds out every case ``test_id`` names.
    """

    test_id: str
    issue: str
    reason: str
    owner: str
    expires: str
    arches: tuple[str, ...] = ()
    runs_files: tuple[str, ...] = ()
    compilers: Compilers | None = None
    params: tuple[str, ...] = ()

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
        if self.compilers is not None and not isinstance(self.compilers, Compilers):
            raise TypeError(
                f"QuarantinedTest failed - test_id={self.test_id!r}, "
                f"compilers={self.compilers!r}, expected a Compilers or None"
            )
        if not isinstance(self.params, tuple) or any(
            not isinstance(param, str) or not param or "-" in param or "[" in param
            for param in self.params
        ):
            raise ValueError(
                f"QuarantinedTest failed - test_id={self.test_id!r}, "
                f"params={self.params!r}, expected a tuple of whole parameter ids"
            )
        if self.params and "[" in node_name:
            raise ValueError(
                f"QuarantinedTest failed - test_id={self.test_id!r}, "
                f"params={self.params!r}, expected a test without parameters to "
                "narrow"
            )

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

    def holds_out_with(self, build: CompilerBuild) -> bool:
        """Whether this entry holds its test out when the loaded compiler is
        ``build``: always when it names no compilers, else only for one of
        them (:meth:`Compilers.include`)."""
        return self.compilers is None or self.compilers.include(build)

    def covers(self, path: str, node_name: str) -> bool:
        """Whether this entry quarantines the item ``node_name`` of ``path``."""
        if path != self.path or not _names_case(self.node_name, node_name):
            return False
        if not self.params:
            return True
        _, bracket, ids = node_name.partition("[")
        return bool(bracket) and set(self.params) <= set(ids[:-1].split("-"))


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
        "library/tests/test_selector_cpp_parity.py",
    ),
    *_waiting(
        SHARED_LIBRARY_LANE,
        "its cases that lower through the C++ engine's shared library "
        "(librocke) skip where the build installs none (the Windows provider "
        "build)",
        "rocKE platform engine; issue: none filed yet",
        "2027-01-31",
        "library/tests/test_attention_ir_cpp_parity.py",
        "library/tests/test_gdn_decode_ir_cpp_parity.py",
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
        "its device cases need gfx950, which no CI GPU stage has",
        "rocKE library attention; issue: none filed yet",
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
        GFX950_RUNNER,
        "its gfx950 reference cases wait for a gfx950 runner: the gfx942 reference "
        "entry deselects them, and the gfx950 one exits on the CI runners",
        "rocKE library; issue: none filed yet",
        "2027-01-31",
        "library/tests/test_conv_pinned_reference.py",
        "library/tests/test_sdpa_pinned_reference.py",
    ),
    *_waiting(
        GFX1151_GFX12_RUNNER,
        "its gfx1151, gfx1200 and gfx1201 reference cases wait for those runners: "
        "the gfx942 reference entry deselects them, and theirs exit on the CI "
        "runners",
        "rocKE library; issue: none filed yet",
        "2027-01-31",
        "library/tests/test_sdpa_pinned_reference.py",
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
    *_waiting(
        BINDING_LANE,
        "its cases that lower through the C++ engine's Python modules "
        "(rocke_engine, rocke_library_engine) skip in CI, whose build's Python has "
        "no pybind11 to build them",
        "rocKE library; issue: none filed yet",
        "2027-01-31",
        "library/tests/test_attention_tiled_3d_kv_cache_policy.py",
        "library/tests/test_attention_tiled_3d_kv_lds_swizzle.py",
        "library/tests/test_attention_tiled_3d_running_max.py",
        "library/tests/test_attention_tiled_3d_wave_page_id_load.py",
        "library/tests/test_attention_tiled_cpp_entry_guards.py",
        "library/tests/test_attn_tiled_2d_gfx950_k_lds_read.py",
        "library/tests/test_conv_direct_grouped_backend_parity.py",
        "library/tests/test_conv_direct_nongrouped_tf32.py",
        "library/tests/test_conv_direct_tf32_mfma.py",
        "library/tests/test_conv_engine_bridge_sched.py",
        "library/tests/test_conv_fwd_split_k.py",
        "library/tests/test_conv_fwd_tf32.py",
        "library/tests/test_conv_implicit_gemm_vector_width.py",
        "library/tests/test_conv_wgrad_prefetch_global_regs.py",
        "library/tests/test_conv_wgrad_streamk.py",
        "library/tests/test_conv_winograd_gemm_wmma.py",
        "library/tests/test_conv_winograd_gen_cpu.py",
        "library/tests/test_direct_dgrad_pp_mfma.py",
        "library/tests/test_direct_fwd_flex_mfma.py",
        "library/tests/test_direct_fwd_pp_mfma.py",
        "library/tests/test_strided_kv_decode.py",
    ),
    *_waiting(
        SHARED_LIBRARY_LANE,
        "its cases that lower through the C++ engine's shared library "
        "(librocke) skip where the build installs none (the Windows provider "
        "build)",
        "rocKE library; issue: none filed yet",
        "2027-01-31",
        "library/tests/test_attention_bwd_lds_alignment.py",
        "library/tests/test_conv_direct_grouped_backend_parity.py",
    ),
    *_waiting(
        BINDING_LANE,
        "its cases that lower through the C++ engine's Python modules "
        "(rocke_engine, rocke_library_engine) skip in CI, whose build's Python has "
        "no pybind11 to build them",
        "rocKE platform; issue: none filed yet",
        "2027-01-31",
        "platform/tests/core/test_arch_domain_lane.py",
        "platform/tests/core/test_f64_mma.py",
        "platform/tests/core/test_gfx950_scaled_mfma.py",
        "platform/tests/core/test_mx_scale_decode.py",
        "platform/tests/core/test_tf32.py",
        "platform/tests/instances/test_gemm_tdm_padding.py",
        "platform/tests/instances/test_gemm_tdm_traits_cpp_engine.py",
        "platform/tests/instances/test_mx_gemm_scale_range.py",
        "platform/tests/support/test_engine_modules.py",
        "platform/tests/test_binding_compiler_selection.py",
        "platform/tests/test_compiler_detection_native.py",
        "platform/tests/test_rocke.py",
    ),
    *_waiting(
        SHARED_LIBRARY_LANE,
        "its cases that lower through the C++ engine's shared library "
        "(librocke) skip where the build installs none (the Windows provider "
        "build)",
        "rocKE platform; issue: none filed yet",
        "2027-01-31",
        "platform/tests/core/test_arch_domain_lane.py",
        "platform/tests/core/test_barrier_fence.py",
        "platform/tests/core/test_bitcast_width.py",
        "platform/tests/core/test_buffer_access_widths.py",
        "platform/tests/core/test_buffer_load_scalar.py",
        "platform/tests/core/test_buffer_rsrc_operands.py",
        "platform/tests/core/test_cdna_mfma_ops.py",
        "platform/tests/core/test_constant_space_global_ops.py",
        "platform/tests/core/test_debug_info.py",
        "platform/tests/core/test_ds_read_tr16_b128_gate.py",
        "platform/tests/core/test_dtypes.py",
        "platform/tests/core/test_f8f6f4_16x16x128.py",
        "platform/tests/core/test_float_format.py",
        "platform/tests/core/test_fp8_scale_conversion_gate.py",
        "platform/tests/core/test_gfx11_wmma_ops.py",
        "platform/tests/core/test_gfx1250_f8f6f4_wmma_ops.py",
        "platform/tests/core/test_gfx1250_llvm_bridge.py",
        "platform/tests/core/test_gfx1250_scaled_wmma.py",
        "platform/tests/core/test_gfx1250_scaled_wmma_operand_forms.py",
        "platform/tests/core/test_gfx1250_wmma_ops.py",
        "platform/tests/core/test_gfx12_wmma_ops.py",
        "platform/tests/core/test_gfx950_f8f6f4_mfma_ops.py",
        "platform/tests/core/test_gfx950_scaled_mfma.py",
        "platform/tests/core/test_global_access_alignment.py",
        "platform/tests/core/test_global_ptr_offset.py",
        "platform/tests/core/test_hip_lowering_parity.py",
        "platform/tests/core/test_integer_wmma_signedness.py",
        "platform/tests/core/test_ldexp.py",
        "platform/tests/core/test_lds_alias_scopes.py",
        "platform/tests/core/test_loop_carried_descriptor.py",
        "platform/tests/core/test_lowbit_loop_types.py",
        "platform/tests/core/test_lshr.py",
        "platform/tests/core/test_magic_division_sum.py",
        "platform/tests/core/test_matrix_calls.py",
        "platform/tests/core/test_mfma_lowbit_ops_removed.py",
        "platform/tests/core/test_mfma_merged_operand_contract.py",
        "platform/tests/core/test_mfma_operand_contract.py",
        "platform/tests/core/test_mma_io.py",
        "platform/tests/core/test_mma_operand_register_type.py",
        "platform/tests/core/test_mma_result_dtype.py",
        "platform/tests/core/test_mx_scale_decode.py",
        "platform/tests/core/test_native_engine.py",
        "platform/tests/core/test_nontemporal_lowering.py",
        "platform/tests/core/test_optimization_barrier.py",
        "platform/tests/core/test_packed_only.py",
        "platform/tests/core/test_param_facts.py",
        "platform/tests/core/test_scf_if_else_results.py",
        "platform/tests/core/test_serialized_ir_name_scopes.py",
        "platform/tests/core/test_smem_access_alignment.py",
        "platform/tests/core/test_target_fact_gates.py",
        "platform/tests/core/test_tf32.py",
        "platform/tests/core/test_trap.py",
        "platform/tests/core/test_u8_arith.py",
        "platform/tests/core/test_u8_memory.py",
        "platform/tests/core/test_unrolled_for_nested_regions.py",
        "platform/tests/core/test_unsigned_int_ops.py",
        "platform/tests/core/test_value_names.py",
        "platform/tests/core/test_wmma_operand_contract.py",
        "platform/tests/core/test_workgroup_size.py",
        "platform/tests/core/test_xf32_bf16x3_lowering.py",
        "platform/tests/core/test_zero_vec_and_lds_element_types.py",
        "platform/tests/instances/differential/test_hip_source_identity.py",
        "platform/tests/instances/test_gfx950_grouped_gemm.py",
        "platform/tests/numeric/test_warp_shuffle_xor_dpp_numeric.py",
        "platform/tests/numeric/test_wave_vote_numeric.py",
        "platform/tests/support/test_engine_modules.py",
        "platform/tests/support/test_engines.py",
        "platform/tests/support/test_require_cpp_engine.py",
        "platform/tests/test_rocke.py",
    ),
    *_waiting(
        SHARED_LIBRARY_LANE,
        "its cases that lower through the C++ engine's shared library "
        "(librocke) skip where the build installs none (the Windows provider "
        "build)",
        "rocKE tiling; issue: none filed yet",
        "2027-01-31",
        "platform/python/rocke/tests/helpers/tiling/mma/test_mma_dtypes.py",
        "platform/python/rocke/tests/helpers/tiling/mma/test_mma_issuable_pairs.py",
        "platform/python/rocke/tests/helpers/tiling/mma/test_mma_issue.py",
        "platform/python/rocke/tests/helpers/tiling/mma/test_mma_k_coupling.py",
        "platform/python/rocke/tests/helpers/tiling/mma/test_mma_operand_feed.py",
        "platform/python/rocke/tests/helpers/tiling/tensor_transforms/test_emit.py",
        "platform/python/rocke/tests/helpers/tiling/test_fragment_storage.py",
        "platform/python/rocke/tests/helpers/tiling/test_global_load_addressing.py",
    ),
    *_waiting(
        TORCH_IN_CI,
        "its torch-checked cases skip without torch in the CI test environment",
        "rocKE library; torch in the CI test environment or a numpy reference",
        "2027-01-31",
        "library/tests/sdpa/test_dense_sdpa_binding.py",
        "library/tests/sdpa/test_sdpa_bwd_case_table.py",
        "library/tests/sdpa/test_sdpa_case_table.py",
        "library/tests/sdpa/test_sdpa_reference.py",
        "library/tests/sdpa/test_unified_sdpa_binding.py",
        "library/tests/test_attention_builds.py",
        "library/tests/test_benchmark_tf32_plumbing.py",
        "library/tests/test_conv_batch_and_miopen_parse.py",
        "library/tests/test_extended_parity_attention.py",
        "library/tests/test_strided_kv_decode.py",
    ),
    *_waiting(
        TORCH_IN_CI,
        "its torch-checked cases skip without torch in the CI test environment",
        "rocKE platform; torch in the CI test environment or a numpy reference",
        "2027-01-31",
        "platform/tests/instances/test_gfx1250_qwen_integration.py",
        "platform/tests/instances/test_rocke_numeric.py",
        "platform/tests/runtime/test_device_array.py",
        "platform/tests/test_rocke.py",
    ),
    *_waiting(
        GFX950_RUNNER,
        "its device cases need gfx950, which no CI GPU stage has",
        "rocKE library; issue: none filed yet",
        "2027-01-31",
        "library/tests/test_attention_tiled_runtime_shape_numeric.py",
        "library/tests/test_attention_unified_2d_masked_tile_numeric.py",
    ),
    *_waiting(
        GFX950_RUNNER,
        "its device cases need gfx950, which no CI GPU stage has",
        "rocKE platform; issue: none filed yet",
        "2027-01-31",
        "platform/tests/instances/test_flatmm_numeric.py",
        "platform/tests/instances/test_gfx950_grouped_gemm.py",
        "platform/tests/numeric/test_warp_shuffle_xor_dpp_numeric.py",
    ),
    PendingTest(
        "library/tests/test_conv_dgrad_correctness.py",
        GFX950_RUNNER,
        "its cases use the (32, 32, 16) bf16 and fp16 warp tile, which gfx942 does not "
        "support, so they run on gfx950",
        "rocKE library; issue: none filed yet",
        "2027-01-31",
        node="TestConvDgradUniformKDecode",
    ),
    *_waiting(
        GFX1250_RUNNER,
        "its device cases need gfx1250, which no CI GPU stage has",
        "rocKE library; issue: none filed yet",
        "2027-01-31",
        "library/tests/test_conv_fwd_correctness.py",
    ),
    *_waiting(
        GFX1250_RUNNER,
        "its device cases need gfx1250, which no CI GPU stage has",
        "rocKE platform; issue: none filed yet",
        "2027-01-31",
        "platform/tests/core/test_gpu_checked_value.py",
    ),
    *_waiting(
        GFX1250_RUNNER,
        "its device cases need gfx1250, which no CI GPU stage has",
        "rocKE tiling; issue: none filed yet",
        "2027-01-31",
        "platform/python/rocke/tests/helpers/tiling/mma/test_mma_k_coupling.py",
    ),
    PendingTest(
        "library/tests/test_conv_wgrad_correctness.py",
        GFX1250_RUNNER,
        "WMMA split-K wgrad: its atomics are not enabled on gfx942, so it runs on gfx1250",
        "rocKE library; issue: none filed yet",
        "2027-01-31",
        node="TestConvWgradCorrectness::test_split_k_wmma",
    ),
    PendingTest(
        "library/tests/test_conv_wgrad_correctness.py",
        GFX1250_RUNNER,
        "WMMA split-K wgrad: its atomics are not enabled on gfx942, so it runs on gfx1250",
        "rocKE library; issue: none filed yet",
        "2027-01-31",
        node="TestConvWgradCorrectness::test_grouped_split_k_wmma",
    ),
    PendingTest(
        "library/tests/test_conv_wgrad_correctness.py",
        GFX1250_RUNNER,
        "WMMA split-K wgrad: its atomics are not enabled on gfx942, so it runs on gfx1250",
        "rocKE library; issue: none filed yet",
        "2027-01-31",
        node="TestConvWgradCorrectness::test_wide_tile_spec_at_degree_cap_gfx1250",
    ),
    *_waiting(
        GFX1151_GFX12_RUNNER,
        "its device cases need gfx1151, gfx1200 or gfx1201, which no CI GPU stage has",
        "rocKE library; issue: none filed yet",
        "2027-01-31",
        "library/tests/dispatch/attention/test_gfx1151_attention_dispatch_numeric.py",
        "library/tests/test_wmma_fmha_fwd_v2_features.py",
        "library/tests/test_wmma_fmha_fwd_v2_numeric.py",
    ),
    *_waiting(
        GFX1151_GFX12_RUNNER,
        "its device cases need gfx1151, gfx1200 or gfx1201, which no CI GPU stage has",
        "rocKE platform; issue: none filed yet",
        "2027-01-31",
        "platform/tests/instances/test_attention_fwd_ext_numeric.py",
    ),
    *_waiting(
        SOURCE_TREE_CASES,
        "its source-tree cases (they read platform tools, C++ sources, docs or the "
        "git history, which the installed test tree omits) skip in CI; run_all.py "
        "runs them",
        "rocKE library; issue: none filed yet",
        "2027-01-31",
        "library/tests/test_reference_common.py",
    ),
    *_waiting(
        SOURCE_TREE_CASES,
        "its source-tree cases (they read platform tools, C++ sources, docs or the "
        "git history, which the installed test tree omits) skip in CI; run_all.py "
        "runs them",
        "rocKE platform portable IR; issue: none filed yet",
        "2027-01-31",
        "platform/python/rocke/portable_ir/tests/test_freshness.py",
        "platform/tests/portable_ir/test_recipe_roller.py",
    ),
    *_waiting(
        SOURCE_TREE_CASES,
        "its source-tree cases (they read platform tools, C++ sources, docs or the "
        "git history, which the installed test tree omits) skip in CI; run_all.py "
        "runs them",
        "rocKE platform; issue: none filed yet",
        "2027-01-31",
        "platform/tests/core/test_arch_domain_lane.py",
        "platform/tests/core/test_integer_wmma_signedness.py",
        "platform/tests/instances/test_gemm_kloop_tdm.py",
        "platform/tests/instances/test_gfx950_attention_tiled_2d_prologue.py",
        "platform/tests/numeric/test_f64_gemm_numeric.py",
        "platform/tests/test_library_reach.py",
        "platform/tests/tools/test_cmake_test_registration.py",
        "platform/tests/tools/test_gen_arch_domain_cpp.py",
        "platform/tests/tools/test_gen_cxx_keywords_cpp.py",
        "platform/tests/tools/test_golden_tool.py",
        "platform/tests/tools/test_run_all.py",
        "platform/tests/tools/test_run_checks.py",
    ),
    *_waiting(
        SOURCE_TREE_CASES,
        "its source-tree cases (they read platform tools, C++ sources, docs or the "
        "git history, which the installed test tree omits) skip in CI; run_all.py "
        "runs them",
        "rocKE tiling; issue: none filed yet",
        "2027-01-31",
        "platform/python/rocke/tests/helpers/tiling/mma/test_mma_dtypes.py",
        "platform/python/rocke/tests/helpers/tiling/mma/test_mma_issuable_pairs.py",
        "platform/python/rocke/tests/helpers/tiling/mma/test_mma_issue.py",
        "platform/python/rocke/tests/helpers/tiling/mma/test_mma_k_coupling.py",
        "platform/python/rocke/tests/helpers/tiling/mma/test_mma_operand_feed.py",
        "platform/python/rocke/tests/helpers/tiling/tensor_transforms/test_accesses.py",
        "platform/python/rocke/tests/helpers/tiling/traits/test_mma_backend_coverage.py",
    ),
    *_waiting(
        EXHAUSTIVE_CASES,
        "its exhaustive cases are deselected in CI (-m 'not exhaustive'); no job runs the "
        "exhaustive cohort yet",
        "rocKE library; issue: none filed yet",
        "2027-01-31",
        "library/tests/test_convolution_forward_aot_packaging.py",
        "library/tests/test_kernel_label.py",
    ),
    PendingTest(
        "library/tests/test_conv_direct_nongrouped_tf32.py",
        COMPREHENSIVE_CASES,
        "the sweep sampler runs only in rocke_library_conv_sweep_gpu_pytest, a "
        "comprehensive-tier entry",
        "rocKE library",
        "2027-01-31",
        node="TestNongroupedF32Gpu::test_sweep_candidates",
    ),
    PendingTest(
        "library/tests/test_attention_tiled_cache_key.py",
        COMPREHENSIVE_CASES,
        "the invariant sweep runs only in rocke_library_attention_cache_key_sweep_pytest, "
        "a comprehensive-tier entry",
        "rocKE library attention",
        "2027-01-31",
        node="test_key_matches_build_identity",
    ),
    PendingTest(
        "library/tests/test_attention_tiled_2d_gfx950_biased_numeric.py",
        COMPREHENSIVE_CASES,
        "the score-modifier mixes run only in "
        "rocke_library_attention_biased_mixes_gpu_pytest, a comprehensive-tier entry, "
        "and need gfx950",
        "rocKE library attention",
        "2027-01-31",
        node="test_modifier_mixes",
    ),
    *_waiting(
        MULTI_GPU_RUNNER,
        "its case follows a change of the current device, which needs two GPUs",
        "rocKE platform; issue: none filed yet",
        "2027-01-31",
        "platform/tests/support/test_gpu.py",
    ),
    *_waiting(
        SHARED_LIBRARY_LANE,
        "its cases that load the shared librocke (the portable-IR C ABI) skip where "
        "the build installs none (the Windows provider build)",
        "rocKE platform portable IR; issue: none filed yet",
        "2027-01-31",
        "platform/python/rocke/portable_ir/tests/test_abi.py",
        "platform/python/rocke/portable_ir/tests/test_guard.py",
        "platform/python/rocke/portable_ir/tests/test_launch.py",
    ),
    *_waiting(
        CI_TEST_TOOLS,
        "its cases that read code objects need llvm-objdump, which the CI test step "
        "does not put on PATH",
        "rocKE library; issue: none filed yet",
        "2027-01-31",
        "library/tests/test_attention_tiled_3d_reduce_lds.py",
    ),
    *_waiting(
        CI_TEST_TOOLS,
        "its cases need jsonschema or llc, which the CI test environment lacks",
        "rocKE platform; issue: none filed yet",
        "2027-01-31",
        "platform/tests/analysis/test_lds_serialization.py",
        "platform/tests/core/test_debug_info.py",
    ),
    *_waiting(
        CI_TEST_TOOLS,
        "its figure cases need matplotlib, which the CI test environment lacks",
        "rocKE tiling; issue: none filed yet",
        "2027-01-31",
        "platform/python/rocke/tests/helpers/tiling/test_lds_conflict.py",
    ),
    *_waiting(
        CI_TEST_TOOLS,
        "its cases that read the code object need llvm-objdump, which the CI test "
        "step does not put on PATH",
        "rocKE platform; issue: none filed yet",
        "2027-01-31",
        "platform/tests/core/test_nontemporal_lowering.py",
        "platform/tests/core/test_trap.py",
    ),
    *_waiting(
        CI_TEST_TOOLS,
        "its occupancy cases build code objects with llc and read them with "
        "llvm-readelf, which the CI test step does not put on PATH",
        "rocKE platform benchmark; issue: none filed yet",
        "2027-01-31",
        "platform/python/rocke/benchmark/perf/tests/test_cli.py",
        "platform/python/rocke/benchmark/perf/tests/test_occupancy.py",
        "platform/tests/benchmark/test_occupancy_targets.py",
    ),
    *_waiting(
        CI_TEST_TOOLS,
        "its cases that read the code object need llvm-objdump, which the CI test "
        "step does not put on PATH",
        "rocKE library; issue: none filed yet",
        "2027-01-31",
        "library/tests/test_attention_tiled_3d_kv_cache_policy.py",
        "library/tests/test_conv_direct_nongrouped_tf32.py",
        "library/tests/test_conv_direct_tf32_mfma.py",
        "library/tests/test_conv_fwd_tf32.py",
    ),
    *_waiting(
        BINDING_LANE,
        "its cases that lower through the C++ engine's library module "
        "(rocke_library_engine) skip in CI, whose build's Python has no pybind11 "
        "to build it",
        "rocKE library; issue: none filed yet",
        "2027-01-31",
        "library/tests/test_dgrad_stem_wmma.py",
        "library/tests/test_direct_dgrad_wmma.py",
        "library/tests/test_direct_dw_fwd_wmma.py",
        "library/tests/test_direct_fwd_gstream_wmma.py",
        "library/tests/test_direct_fwd_tapouter_wmma.py",
        "library/tests/test_direct_fwd_wmma.py",
        "library/tests/test_fwd_small_mfma.py",
        "library/tests/test_fwd_small_wmma.py",
    ),
    *_waiting(
        GFX1151_GFX12_RUNNER,
        "its gfx1151 cases need a gfx1151 GPU, which no CI GPU stage has",
        "rocKE platform; issue: none filed yet",
        "2027-01-31",
        "platform/tests/instances/test_rocke_numeric.py",
    ),
)


_ATTENTION_BUILDS = "library/tests/test_attention_builds.py"
_GDN_DECODE_FUSED = "library/tests/test_gdn_decode_fused.py"
_MLA_PREFILL = "library/tests/test_mla_prefill_gfx942.py"
_COMPILER_20 = "ROCm 7.1 (AMD clang 20.0.0git, roc-7.1.0 25425)"
# The llvm-project commits of two LLVM 24 compiler builds: the one CI pins, and
# one a ROCm install ships.
_LLVM_24_CI_COMMIT = "424729dba653"
_LLVM_24_ROCM_INSTALL_COMMIT = "d6f6cb691863"
_COMPILER_24 = f"AMD clang 24.0.0git (llvm-project {_LLVM_24_CI_COMMIT})"
_LLVM_20 = Compilers(llvm_major=20)
_LLVM_24 = Compilers(llvm_major=24)
_LLVM_24_CI = Compilers(llvm_major=24, llvm_commits=(_LLVM_24_CI_COMMIT,))
_LLVM_24_ROCM_INSTALL = Compilers(
    llvm_major=24, llvm_commits=(_LLVM_24_ROCM_INSTALL_COMMIT,)
)
_SPILL_RETUNE_OWNER = "kernel owners; issue: none filed yet"
_SPILL_RETUNE_EXPIRES = "2026-12-31"


def _measured(
    path: str, compilers: Compilers, reason: str, *allowances: tuple[str, int]
) -> tuple[MeasuredBudget, ...]:
    """One spill-retune budget per ``(kernel name, bytes)``, all for one reason."""
    return tuple(
        MeasuredBudget(
            path,
            kernel_name,
            scratch_bytes,
            compilers,
            reason,
            _SPILL_RETUNE_OWNER,
            _SPILL_RETUNE_EXPIRES,
        )
        for kernel_name, scratch_bytes in allowances
    )


MEASURED_BUDGETS: tuple[MeasuredBudget, ...] = (
    *_measured(
        _ATTENTION_BUILDS,
        _LLVM_20,
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
        _LLVM_20,
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
        _LLVM_24,
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
        _LLVM_24,
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
        _LLVM_24,
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
        _LLVM_24,
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
    # Keyed by MlaPrefillSpec.fwd_kernel_identity(), the name without the
    # device's XCD count: the 6- and 8-XCD variants spill the same bytes.
    *_measured(
        _MLA_PREFILL,
        _LLVM_24_CI,
        "gfx942 bf16 MLA prefill forward, the shipped layout (4 waves, VGPR cap "
        f"256): {_COMPILER_24} spills this many bytes; the LLVM 24 build of "
        f"llvm-project {_LLVM_24_ROCM_INSTALL_COMMIT} compiles it with no scratch, "
        "and LLVM 20 does not accept the -amdgpu-mfma-vgpr-form option it needs",
        (
            "mla_prefill_fwd_gfx942_h128_n128r64v128_rkv512_q32k16_rt32_w4_hpw2_qpw2_bf16",
            12,
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
    path: str, kernel_name: str, build: CompilerBuild
) -> int:
    """The scratch bytes the test file ``path`` allows ``kernel_name`` when
    built by ``build`` (``rocke.tests.comgr.loaded_compiler_build()``): its
    measured budget when ``build`` is one of the compilers it was measured on,
    else 0."""
    return next(
        (
            budget.scratch_bytes
            for budget in MEASURED_BUDGETS
            if budget.path == path
            and budget.kernel_name == kernel_name
            and budget.compilers.include(build)
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
    compilers: Compilers | None = None,
) -> tuple[QuarantinedTest, ...]:
    """One entry per test id, all quarantined for the same issue and reason (on
    ``arches`` and with ``compilers`` only, when given)."""
    return tuple(
        QuarantinedTest(
            test_id,
            issue,
            reason,
            owner,
            expires,
            arches=arches,
            compilers=compilers,
        )
        for test_id in test_ids
    )


_RECORDER_REFUSES_IF_ELSE = (
    "the portable-IR recorder refuses scf.if_else with results, which the "
    "three conv_direct_*_mfma kernels use, so recording and replaying every "
    "production kernel fails"
)

_DGRAD_CORRECTNESS = (
    "library/tests/test_conv_dgrad_correctness.py::TestConvDgradCorrectness"
)
# What the LLVM 24 compiler builds below get wrong in the gfx950 dgrad kernels.
_SGPR_SPLIT_SPILL = (
    "the register allocator splits an SGPR tuple filled by one scalar load, "
    "copies only some of its pieces, spills the whole tuple and later reads a piece "
    "it never copied (a compiler defect: rocKE's IR is the same for every "
    "LLVM 24 build, and LLVM 20 builds these kernels correctly)"
)
# The compiler issue that tracks _SGPR_SPLIT_SPILL.
_SGPR_SPLIT_SPILL_ISSUE = "LCOMPILER-3317"


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
        _SGPR_SPLIT_SPILL_ISSUE,
        "built by the LLVM 24 compiler a ROCm install ships, the gfx950 dgrad kouter "
        "and ukd kernels (16x16x32 and 32x32x16 atoms) fault with an illegal "
        "memory access or miscompare (rel_err 1.17, depthwise 11x3): "
        f"{_SGPR_SPLIT_SPILL}",
        "kernel owners",
        "2026-12-15",
        f"{_DGRAD_CORRECTNESS}::test_bf16_depthwise_large_filter",
        f"{_DGRAD_CORRECTNESS}::test_bf16_stride1",
        f"{_DGRAD_CORRECTNESS}::test_fp16_resnet_shape",
        f"{_DGRAD_CORRECTNESS}::test_fp16_stride1",
        arches=("gfx950",),
        compilers=_LLVM_24_ROCM_INSTALL,
    ),
    *_quarantined(
        _SGPR_SPLIT_SPILL_ISSUE,
        "built by the LLVM 24 compiler CI pins, the gfx950 dgrad kernel at tile "
        "256x64x32, warp tile 32x32x16, compv4 miscompares on the fp16 depthwise "
        f"stride-2 shape (rel_err 0.88): {_SGPR_SPLIT_SPILL}",
        "kernel owners",
        "2026-12-15",
        f"{_DGRAD_CORRECTNESS}::test_fp16_depthwise_stride2",
        arches=("gfx950",),
        compilers=_LLVM_24_CI,
    ),
    QuarantinedTest(
        "library/tests/test_attention_additive_bias_gfx950_numeric.py::"
        "test_additive_bias_with_features_matches_reference",
        "AICK-3375",
        "on gfx950 the attention selector routes the single-sequence bf16 "
        "sliding-window prefill (384 queries over 640 keys, window 64) to the "
        "transposed 32x32 score path, where the test's route pin expects 16x16, "
        "so the route assertion fails before launch, with torch and without, in "
        "each of its four bias-layout and head cases; the selector owners decide "
        "whether the pin or the selector is right",
        "rocKE library attention selector",
        "2026-12-31",
        arches=("gfx950",),
        params=("tiled_window_1seq", "bf16"),
    ),
)


def waiting_entry(
    path: str, node_name: str, reason: str | None, *, deselected: bool = False
) -> PendingTest | QuarantinedTest | None:
    """The record of why no CI lane ran the case ``node_name`` of the test file
    ``path`` (relative to the rocke directory), or None when nothing records it.

    ``reason`` is why a CI lane that collected the case did not run it: its
    skip or deselection reason (``deselected`` says which), or None when no CI
    lane collected it. The case is recorded by its quarantine, or that of the
    test that runs its file (one that names arches or compilers, only where it
    was skipped as quarantined), or by an entry that covers it
    (:meth:`PendingTest.covers`).
    """
    quarantine = quarantined_test(path, node_name) or next(
        (entry for entry in QUARANTINED if path in entry.runs_files), None
    )
    if quarantine is not None and (
        not (quarantine.arches or quarantine.compilers)
        or (reason is not None and reason.startswith(QUARANTINE_SKIP_PREFIX))
    ):
        return quarantine
    return next(
        (
            entry
            for entry in PENDING
            if entry.covers(path, node_name, reason, deselected=deselected)
        ),
        None,
    )


def tree_test_files(rocke_dir: Path) -> set[str]:
    """Every test file of the source tree at ``rocke_dir``, relative to it."""
    return {
        file.relative_to(rocke_dir).as_posix()
        for root in TEST_FILE_ROOTS
        for file in (rocke_dir / root).rglob("test_*.py")
        if "__pycache__" not in file.parts
    }
