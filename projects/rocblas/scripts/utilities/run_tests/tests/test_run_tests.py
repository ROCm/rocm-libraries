# Copyright (C) 2016-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""Tests for run_tests.py helper functions.

Run with:  pytest scripts/utilities/tests/
"""
import os
import sys
from typing import Optional, Sequence

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from run_tests import (
    _archive_log,
    _build_exclusion_filter,
    _FILTER_LIST_THRESHOLD,
    _list_remaining_tests,
    _parse_completed_tests,
    _parse_test_counts,
    _record_from_dict,
    build_all_groups,
    build_arg_parser,
    recover_interrupted_jobs,
    JobRecord,
    JobSpec,
    RunState,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _write_log(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def _gtest_log(
    total: Optional[int] = None,
    passed: Sequence[str] = (),
    failed: Sequence[str] = (),
    summary: bool = False,
) -> str:
    """Build a realistic GTest output string."""
    lines = []
    if total is not None:
        lines.append(f"[==========] Running {total} tests from 10 test cases.")
    for name in passed:
        lines.append(f"[       OK ] suite.{name} (5 ms)")
    for name in failed:
        lines.append(f"[  FAILED  ] suite.{name} (3 ms)")
    if summary:
        lines.append(f"[  PASSED  ] {len(passed)} tests.")
        if failed:
            lines.append(f"[  FAILED  ] {len(failed)} tests, listed below:")
            for name in failed:
                # Suite-level repeat — no timing suffix; must NOT be parsed as test-level
                lines.append(f"[  FAILED  ] suite.{name}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# _parse_completed_tests
# ---------------------------------------------------------------------------

class TestParseCompletedTests:
    def test_basic_pass_and_fail(self, tmp_path):
        log = tmp_path / "job.txt"
        _write_log(str(log), _gtest_log(
            total=10,
            passed=("alpha", "beta"),
            failed=("gamma",),
        ))
        passed, failed = _parse_completed_tests(str(log))
        assert passed == {"suite.alpha", "suite.beta"}
        assert failed == {"suite.gamma"}

    def test_suite_level_failure_line_not_counted(self, tmp_path):
        """The summary repeat '[  FAILED  ] suite.name' (no timing) must be ignored."""
        log = tmp_path / "job.txt"
        _write_log(str(log), _gtest_log(
            total=5,
            passed=("p1", "p2"),
            failed=("f1",),
            summary=True,
        ))
        passed, failed = _parse_completed_tests(str(log))
        # f1 appears twice in the log: once with timing (test-level), once without (summary).
        # It should appear exactly once in the failed set.
        assert failed == {"suite.f1"}
        assert passed == {"suite.p1", "suite.p2"}

    def test_empty_log(self, tmp_path):
        log = tmp_path / "job.txt"
        _write_log(str(log), "")
        passed, failed = _parse_completed_tests(str(log))
        assert passed == set()
        assert failed == set()

    def test_missing_file(self, tmp_path):
        passed, failed = _parse_completed_tests(str(tmp_path / "nonexistent.txt"))
        assert passed == set()
        assert failed == set()

    def test_preamble_lines_ignored(self, tmp_path):
        """Preamble written by run_tests.py itself must not produce false matches."""
        content = (
            "[run_tests.py] PARTIAL RESUME — excluding 50 tests total\n"
            "[run_tests.py] Filter: *asum*quick*-test1:test2\n"
            "\n"
            "[==========] Running 100 tests from 3 test cases.\n"
            "[       OK ] suite.real_test (2 ms)\n"
        )
        log = tmp_path / "job.txt"
        _write_log(str(log), content)
        passed, failed = _parse_completed_tests(str(log))
        assert passed == {"suite.real_test"}
        assert failed == set()


# ---------------------------------------------------------------------------
# _parse_test_counts
# ---------------------------------------------------------------------------

class TestParseTestCounts:
    def test_finished_log_uses_summary(self, tmp_path):
        """Authoritative '[  PASSED  ] N tests.' overrides OK-line count."""
        log = tmp_path / "job.txt"
        _write_log(str(log), _gtest_log(
            total=100,
            passed=tuple(f"t{i}" for i in range(97)),
            failed=("f1", "f2", "f3"),
            summary=True,
        ))
        passed, _, total, _ = _parse_test_counts(str(log))
        assert passed == 97
        assert total == 100

    def test_inprogress_log_counts_ok_lines(self, tmp_path):
        """No summary line yet — fall back to counting [       OK ] lines."""
        log = tmp_path / "job.txt"
        _write_log(str(log), _gtest_log(total=500, passed=tuple(f"t{i}" for i in range(42))))
        passed, _, total, _ = _parse_test_counts(str(log))
        assert passed == 42
        assert total == 500

    def test_total_zero_when_header_absent(self, tmp_path):
        log = tmp_path / "job.txt"
        _write_log(str(log), "[       OK ] suite.t1 (1 ms)\n[       OK ] suite.t2 (1 ms)\n")
        passed, _, total, _ = _parse_test_counts(str(log))
        assert passed == 2
        assert total == 0

    def test_missing_file(self, tmp_path):
        passed, _, total, _ = _parse_test_counts(str(tmp_path / "nonexistent.txt"))
        assert passed == 0
        assert total == 0


# ---------------------------------------------------------------------------
# _build_exclusion_filter
# ---------------------------------------------------------------------------

class TestBuildExclusionFilter:
    def test_empty_set_returns_none(self):
        assert _build_exclusion_filter("*asum*quick*", set()) is None

    def test_appends_to_filter_without_negative(self):
        result = _build_exclusion_filter("*asum*quick*", {"suite.t1", "suite.t2"})
        assert result is not None
        assert result.startswith("*asum*quick*-")
        negative = result.split("-", 1)[1]
        assert "suite.t1" in negative
        assert "suite.t2" in negative

    def test_appends_to_filter_with_existing_negative(self):
        original = "*asum*quick*-*_batched*:*_ex*"
        result = _build_exclusion_filter(original, {"suite.t1"})
        assert result is not None
        positive, negative = result.split("-", 1)
        assert positive == "*asum*quick*"
        # Original negative clauses must be preserved
        assert "*_batched*" in negative
        assert "*_ex*" in negative
        # New exclusion appended
        assert "suite.t1" in negative

    def test_excluded_names_are_sorted(self):
        result = _build_exclusion_filter("*f*", {"suite.z", "suite.a", "suite.m"})
        assert result is not None
        negative = result.split("-", 1)[1]
        names = [p for p in negative.split(":") if not p.startswith("*")]
        assert names == sorted(names)


# ---------------------------------------------------------------------------
# _archive_log
# ---------------------------------------------------------------------------

class TestArchiveLog:
    def test_creates_prev_on_first_archive(self, tmp_path):
        log = tmp_path / "job.txt"
        _write_log(str(log), "content")
        archived = _archive_log(str(log))
        assert archived is not None
        assert archived == str(log) + ".prev"
        assert os.path.exists(archived)
        assert not os.path.exists(str(log))

    def test_sequences_when_prev_exists(self, tmp_path):
        log = tmp_path / "job.txt"
        _write_log(str(log) + ".prev", "old")  # .prev already occupied
        _write_log(str(log), "new")
        archived = _archive_log(str(log))
        assert archived is not None
        assert archived == str(log) + ".prev.1"
        assert os.path.exists(archived)

    def test_sequences_skip_occupied_numbers(self, tmp_path):
        log = tmp_path / "job.txt"
        _write_log(str(log) + ".prev", "old")
        _write_log(str(log) + ".prev.1", "older")
        _write_log(str(log), "newest")
        archived = _archive_log(str(log))
        assert archived == str(log) + ".prev.2"

    def test_returns_none_when_file_absent(self, tmp_path):
        result = _archive_log(str(tmp_path / "nonexistent.txt"))
        assert result is None

    def test_original_content_preserved(self, tmp_path):
        log = tmp_path / "job.txt"
        _write_log(str(log), "hello world")
        archived = _archive_log(str(log))
        assert archived is not None
        with open(archived) as f:
            assert f.read() == "hello world"


# ---------------------------------------------------------------------------
# Multi-run partial resume — cumulative exclusion chain
# ---------------------------------------------------------------------------

class TestCumulativeResumeChain:
    """Simulate three consecutive interrupted runs of the same job and verify
    that the exclusion set, cumulative passed count, and estimated total all
    accumulate correctly across each restart."""

    ORIGINAL_FILTER = "*trsv_batched*quick*-*_ex*"
    FULL_TOTAL = 4354

    def _simulate_run(
        self,
        tmp_path,
        run_index: int,
        gtest_total: int,
        passed_names: list,
        failed_names: list,
        prev_excluded: set,
        prev_passed_offset: int,
        skip_failed: bool = False,
    ) -> dict:
        """Write a partial-run log and compute the next state, mirroring _execute."""
        log = tmp_path / f"job.run{run_index}.txt"
        _write_log(str(log), _gtest_log(
            total=gtest_total,
            passed=passed_names,
            failed=failed_names,
        ))

        new_passed, new_failed = _parse_completed_tests(str(log))
        _, _, partial_total, _ = _parse_test_counts(str(log))

        new_excluded = new_passed | (new_failed if skip_failed else set())
        all_excluded = prev_excluded | new_excluded
        cumulative_passed = prev_passed_offset + len(new_passed)
        estimated_original = partial_total + len(prev_excluded) if partial_total else 0

        return {
            "all_excluded": all_excluded,
            "cumulative_passed": cumulative_passed,
            "estimated_original": estimated_original,
            "partial_total": partial_total,
            "log": str(log),
        }

    def test_single_interrupt_offset_and_total(self, tmp_path):
        """After one interruption, display offset and total are correct."""
        state = self._simulate_run(
            tmp_path, 1,
            gtest_total=self.FULL_TOTAL,
            passed_names=[f"t{i}" for i in range(638)],
            failed_names=[f"f{i}" for i in range(12)],
            prev_excluded=set(),
            prev_passed_offset=0,
        )
        assert state["cumulative_passed"] == 638
        assert state["estimated_original"] == self.FULL_TOTAL
        assert len(state["all_excluded"]) == 638  # only passed (skip_failed=False)

    def test_chain_preserves_all_passes_across_two_interrupts(self, tmp_path):
        """Run 1 → interrupt → Run 2 → interrupt → Run 3 starts with full history."""
        # Run 1: full job, interrupted after 638 pass + 12 fail
        s1 = self._simulate_run(
            tmp_path, 1,
            gtest_total=self.FULL_TOTAL,
            passed_names=[f"p1_{i}" for i in range(638)],
            failed_names=[f"f1_{i}" for i in range(12)],
            prev_excluded=set(),
            prev_passed_offset=0,
        )
        assert s1["estimated_original"] == self.FULL_TOTAL

        # Run 2: partial resume of run 1, GTest runs 3716 = 4354-638 tests
        gtest_total_run2 = self.FULL_TOTAL - len(s1["all_excluded"])  # 3716
        s2 = self._simulate_run(
            tmp_path, 2,
            gtest_total=gtest_total_run2,
            passed_names=[f"p2_{i}" for i in range(68)],
            failed_names=[f"f2_{i}" for i in range(5)],
            prev_excluded=s1["all_excluded"],
            prev_passed_offset=s1["cumulative_passed"],
        )
        # cumulative: 638 + 68 = 706
        assert s2["cumulative_passed"] == 706
        # estimated_original: 3716 + 638 = 4354
        assert s2["estimated_original"] == self.FULL_TOTAL
        # exclusion set grows: only new passed added (skip_failed=False)
        assert len(s2["all_excluded"]) == 638 + 68

    def test_chain_third_interrupt_still_correct(self, tmp_path):
        """Three interruptions: the third run still knows about all previous passes."""
        # Run 1
        s1 = self._simulate_run(
            tmp_path, 1,
            gtest_total=self.FULL_TOTAL,
            passed_names=[f"p1_{i}" for i in range(638)],
            failed_names=[f"f1_{i}" for i in range(12)],
            prev_excluded=set(),
            prev_passed_offset=0,
        )
        # Run 2
        s2 = self._simulate_run(
            tmp_path, 2,
            gtest_total=self.FULL_TOTAL - len(s1["all_excluded"]),
            passed_names=[f"p2_{i}" for i in range(68)],
            failed_names=[f"f2_{i}" for i in range(5)],
            prev_excluded=s1["all_excluded"],
            prev_passed_offset=s1["cumulative_passed"],
        )
        # Run 3
        s3 = self._simulate_run(
            tmp_path, 3,
            gtest_total=self.FULL_TOTAL - len(s2["all_excluded"]),
            passed_names=[f"p3_{i}" for i in range(10)],
            failed_names=[],
            prev_excluded=s2["all_excluded"],
            prev_passed_offset=s2["cumulative_passed"],
        )
        assert s3["cumulative_passed"] == 638 + 68 + 10  # = 716
        assert s3["estimated_original"] == self.FULL_TOTAL
        # Exclusion filter must be non-empty
        f = _build_exclusion_filter(self.ORIGINAL_FILTER, s3["all_excluded"])
        assert f is not None
        assert f != self.ORIGINAL_FILTER

    def test_skip_failed_also_excludes_failed_tests(self, tmp_path):
        """With skip_failed=True, failed tests join the exclusion set too."""
        s1 = self._simulate_run(
            tmp_path, 1,
            gtest_total=100,
            passed_names=[f"p{i}" for i in range(30)],
            failed_names=[f"f{i}" for i in range(5)],
            prev_excluded=set(),
            prev_passed_offset=0,
            skip_failed=True,
        )
        # Excluded = passed + failed = 35
        assert len(s1["all_excluded"]) == 35
        # But cumulative_passed counts only passed tests
        assert s1["cumulative_passed"] == 30

    def test_total_is_sticky_and_never_shrinks(self, tmp_path):
        """tests_total (estimated_original) must never decrease across runs."""
        s1 = self._simulate_run(
            tmp_path, 1,
            gtest_total=self.FULL_TOTAL,
            passed_names=[f"p{i}" for i in range(500)],
            failed_names=[],
            prev_excluded=set(),
            prev_passed_offset=0,
        )
        s2 = self._simulate_run(
            tmp_path, 2,
            gtest_total=self.FULL_TOTAL - 500,
            passed_names=[f"q{i}" for i in range(20)],
            failed_names=[],
            prev_excluded=s1["all_excluded"],
            prev_passed_offset=s1["cumulative_passed"],
        )
        # estimated_original at each stage should equal FULL_TOTAL
        assert s1["estimated_original"] == self.FULL_TOTAL
        assert s2["estimated_original"] == self.FULL_TOTAL


# ---------------------------------------------------------------------------
# GTest filter ambiguity — prefix-name exclusions
# ---------------------------------------------------------------------------

class TestFilterAmbiguity:
    """Verify that functions whose names are prefixes of other function names
    have explicit negative exclusions in their plain-variant GTest filters,
    so that e.g. *rotm* does not accidentally run rotmg tests.

    Only the plain (non-batched) variant needs the extra exclusions; the
    batched/strided_batched variants use a more specific positive pattern
    (e.g. *rotm_batched*) that cannot match the longer sibling names.
    """

    @pytest.fixture(scope="class")
    def groups(self):
        return build_all_groups("/tmp")

    def _filter(self, groups, group_id, variant):
        for spec in groups[group_id]:
            if spec.job_id == f"{group_id}.{variant}":
                return spec.gtest_filter
        raise KeyError(f"{group_id}.{variant}")

    # -- L1_BLAS plain variants -------------------------------------------

    @pytest.mark.parametrize("excluded", ["*rotg*", "*rotm*"])
    def test_rot_excludes_longer_siblings(self, groups, excluded):
        assert excluded in self._filter(groups, "L1_BLAS", "rot")

    def test_rotm_excludes_rotmg(self, groups):
        assert "*rotmg*" in self._filter(groups, "L1_BLAS", "rotm")

    def test_dot_excludes_dotc(self, groups):
        assert "*dotc*" in self._filter(groups, "L1_BLAS", "dot")

    # -- L2_BLAS plain variants -------------------------------------------

    def test_her_excludes_her2(self, groups):
        assert "*her2*" in self._filter(groups, "L2_BLAS", "her")

    def test_hpr_excludes_hpr2(self, groups):
        assert "*hpr2*" in self._filter(groups, "L2_BLAS", "hpr")

    def test_spr_excludes_spr2(self, groups):
        assert "*spr2*" in self._filter(groups, "L2_BLAS", "spr")

    def test_syr_excludes_syr2(self, groups):
        assert "*syr2*" in self._filter(groups, "L2_BLAS", "syr")

    @pytest.mark.parametrize("excluded", ["*geru*", "*gerc*"])
    def test_ger_excludes_longer_siblings(self, groups, excluded):
        assert excluded in self._filter(groups, "L2_BLAS", "ger")

    def test_trsv_excludes_tbsv(self, groups):
        assert "*tbsv*" in self._filter(groups, "L2_BLAS", "trsv")

    @pytest.mark.parametrize("excluded", ["*tpmv*", "*tbmv*"])
    def test_trmv_excludes_longer_siblings(self, groups, excluded):
        assert excluded in self._filter(groups, "L2_BLAS", "trmv")

    # -- Batched/strided variants must NOT carry extra exclusions ----------
    # Their positive pattern (e.g. *rotm_batched*) is already unambiguous.

    @pytest.mark.parametrize("variant", ["rotm_batched", "rotm_strided_batched"])
    def test_rotm_batched_variants_clean(self, groups, variant):
        assert "*rotmg*" not in self._filter(groups, "L1_BLAS", variant)

    @pytest.mark.parametrize("variant", ["rot_batched", "rot_strided_batched"])
    def test_rot_batched_variants_clean(self, groups, variant):
        f = self._filter(groups, "L1_BLAS", variant)
        assert "*rotg*" not in f
        assert "*rotmg*" not in f


# ---------------------------------------------------------------------------
# _parse_test_counts — failed count and current_test
# ---------------------------------------------------------------------------

class TestParseTestCountsExtended:
    def test_failed_count_from_test_level_lines(self, tmp_path):
        """[  FAILED  ] Suite.Test (N ms) lines increment the failed count."""
        log = tmp_path / "job.txt"
        _write_log(str(log), _gtest_log(
            total=10,
            passed=("p1", "p2"),
            failed=("f1", "f2", "f3"),
            summary=True,
        ))
        _, failed, _, _ = _parse_test_counts(str(log))
        assert failed == 3

    def test_suite_level_failed_lines_not_counted(self, tmp_path):
        """Summary repeat '[  FAILED  ] Suite.Test' (no timing) must not inflate count."""
        log = tmp_path / "job.txt"
        _write_log(str(log), _gtest_log(
            total=5,
            passed=("p1",),
            failed=("f1", "f2"),
            summary=True,
        ))
        _, failed, _, _ = _parse_test_counts(str(log))
        # f1 and f2 appear twice in log (once with timing, once without)
        assert failed == 2

    def test_current_test_from_run_line(self, tmp_path):
        """current_test is the name from the last [ RUN      ] line."""
        content = (
            "[==========] Running 3 tests from 1 test suite.\n"
            "[ RUN      ] suite.alpha\n"
            "[       OK ] suite.alpha (1 ms)\n"
            "[ RUN      ] suite.beta\n"
        )
        log = tmp_path / "job.txt"
        _write_log(str(log), content)
        _, _, _, current = _parse_test_counts(str(log))
        assert current == "suite.beta"

    def test_current_test_empty_when_no_run_line(self, tmp_path):
        log = tmp_path / "job.txt"
        _write_log(str(log), "[==========] Running 1 tests from 1 test suite.\n")
        _, _, _, current = _parse_test_counts(str(log))
        assert current == ""

    def test_current_test_updates_to_last_run_line(self, tmp_path):
        """Only the last [ RUN      ] line is kept."""
        content = "\n".join(
            f"[ RUN      ] suite.t{i}\n[       OK ] suite.t{i} (1 ms)"
            for i in range(5)
        ) + "\n[ RUN      ] suite.t5\n"
        log = tmp_path / "job.txt"
        _write_log(str(log), content)
        _, _, _, current = _parse_test_counts(str(log))
        assert current == "suite.t5"


# ---------------------------------------------------------------------------
# build_all_groups — structure and counts
# ---------------------------------------------------------------------------

class TestBuildAllGroups:
    @pytest.fixture(scope="class")
    def groups(self):
        return build_all_groups("/tmp/test_output")

    def test_expected_groups_present(self, groups):
        assert list(groups.keys()) == ["AUXILIARY", "L1_BLAS", "L1_BLAS_EX", "L2_BLAS"]

    def test_auxiliary_count(self, groups):
        assert len(groups["AUXILIARY"]) == 13

    def test_l1_blas_count(self, groups):
        assert len(groups["L1_BLAS"]) == 42   # 14 functions × 3 variants

    def test_l1_blas_ex_count(self, groups):
        assert len(groups["L1_BLAS_EX"]) == 18  # 6 functions × 3 variants

    def test_l2_blas_count(self, groups):
        assert len(groups["L2_BLAS"]) == 72   # 24 functions × 3 variants

    def test_all_job_ids_unique(self, groups):
        all_ids = [sp.job_id for specs in groups.values() for sp in specs]
        assert len(all_ids) == len(set(all_ids))

    def test_log_file_path_uses_output_dir(self, groups):
        for specs in groups.values():
            for sp in specs:
                assert sp.log_file.startswith("/tmp/test_output/")

    def test_log_file_name_matches_job_id(self, groups):
        for specs in groups.values():
            for sp in specs:
                expected = f"{sp.job_id}.txt"
                assert sp.log_file.endswith(expected)

    def test_group_id_matches_group_key(self, groups):
        for gid, specs in groups.items():
            for sp in specs:
                assert sp.group_id == gid

    def test_auxiliary_filters_contain_quick(self, groups):
        for sp in groups["AUXILIARY"]:
            assert "*quick*" in sp.gtest_filter

    def test_l1_plain_variants_exclude_batched_and_ex(self, groups):
        """Plain L1_BLAS variants must exclude _batched and _ex."""
        for sp in groups["L1_BLAS"]:
            if not ("_batched" in sp.job_id or "_strided_batched" in sp.job_id):
                assert "*_batched*" in sp.gtest_filter
                assert "*_ex*" in sp.gtest_filter

    def test_l1_batched_variants_exclude_ex(self, groups):
        """L1_BLAS batched/strided variants must exclude _ex."""
        for sp in groups["L1_BLAS"]:
            if "_batched" in sp.job_id:
                assert "*_ex*" in sp.gtest_filter

    def test_l1_ex_plain_variants_exclude_batched(self, groups):
        """L1_BLAS_EX plain _ex variants must exclude _batched."""
        for sp in groups["L1_BLAS_EX"]:
            # plain variants end in exactly _ex (not _batched_ex or _strided_batched_ex)
            fn = sp.job_id.split(".", 1)[1]
            if fn.endswith("_ex") and "_batched_" not in fn:
                assert "*_batched*" in sp.gtest_filter

    def test_l1_ex_batched_and_strided_no_extra_exclusion(self, groups):
        """L1_BLAS_EX batched_ex and strided_batched_ex have no negative clause."""
        for sp in groups["L1_BLAS_EX"]:
            if "_batched_ex" in sp.job_id or "_strided_batched_ex" in sp.job_id:
                assert "-" not in sp.gtest_filter

    def test_l2_plain_variants_exclude_batched_and_ex(self, groups):
        for sp in groups["L2_BLAS"]:
            if not ("_batched" in sp.job_id or "_strided_batched" in sp.job_id):
                assert "*_batched*" in sp.gtest_filter
                assert "*_ex*" in sp.gtest_filter


# ---------------------------------------------------------------------------
# _record_from_dict — defaults for missing fields
# ---------------------------------------------------------------------------

class TestRecordFromDict:
    def test_minimal_dict_uses_defaults(self):
        rec = _record_from_dict({})
        assert rec.status == "not_started"
        assert rec.result == "unknown"
        assert rec.pid is None
        assert rec.tests_passed == 0
        assert rec.tests_failed == 0
        assert rec.tests_total == 0
        assert rec.tests_passed_offset == 0
        assert rec.tests_failed_offset == 0
        assert rec.excluded_tests == []
        assert rec.prev_log is None

    def test_explicit_values_round_trip(self):
        d = {
            "status": "finished",
            "result": "pass",
            "pid": None,
            "start_time": 1000.0,
            "end_time": 2000.0,
            "exit_code": 0,
            "tests_passed": 42,
            "tests_failed": 3,
            "tests_total": 100,
            "tests_passed_offset": 10,
            "tests_failed_offset": 1,
            "excluded_tests": ["suite.t1", "suite.t2"],
            "prev_log": "/tmp/job.txt.prev",
        }
        rec = _record_from_dict(d)
        assert rec.status == "finished"
        assert rec.result == "pass"
        assert rec.tests_passed == 42
        assert rec.tests_failed == 3
        assert rec.tests_total == 100
        assert rec.tests_passed_offset == 10
        assert rec.excluded_tests == ["suite.t1", "suite.t2"]
        assert rec.prev_log == "/tmp/job.txt.prev"

    def test_incomplete_result_preserved(self):
        rec = _record_from_dict({"result": "incomplete"})
        assert rec.result == "incomplete"


# ---------------------------------------------------------------------------
# recover_interrupted_jobs
# ---------------------------------------------------------------------------

def _dead_pid() -> int:
    """Return a PID that is guaranteed not to exist by starting and immediately
    waiting on a subprocess, then returning its (now-released) PID."""
    import subprocess
    p = subprocess.Popen(["true"])
    p.wait()
    return p.pid


def _make_state(job_ids: list, status: str = "running", pid=None) -> RunState:
    if pid is None:
        pid = _dead_pid()
    records = {
        jid: JobRecord(
            status=status, result="unknown", pid=pid,
            start_time=1000.0, end_time=None, exit_code=None,
        )
        for jid in job_ids
    }
    return RunState(version=1, executable="", output_dir="", max_parallel=4,
                    records=records)


def _make_all_jobs(tmp_path, job_ids: list) -> dict:
    return {
        jid: JobSpec(
            job_id=jid, group_id="G",
            gtest_filter="*quick*",
            log_file=str(tmp_path / f"{jid}.txt"),
        )
        for jid in job_ids
    }


class TestRecoverInterruptedJobs:
    def test_dead_pid_resets_to_not_started(self, tmp_path):
        """A job with a dead PID is reset so it will be re-run."""
        state = _make_state(["G.job1"])  # uses a guaranteed-dead PID
        all_jobs = _make_all_jobs(tmp_path, ["G.job1"])
        reattach = []
        recover_interrupted_jobs(state, all_jobs, reattach)
        rec = state.records["G.job1"]
        assert rec.status == "not_started"
        assert rec.result == "unknown"
        assert rec.pid is None
        assert reattach == []

    def test_dead_pid_archives_existing_log(self, tmp_path):
        """The old log is renamed to .prev before the job is re-run."""
        state = _make_state(["G.job1"], pid=_dead_pid())
        all_jobs = _make_all_jobs(tmp_path, ["G.job1"])
        log = tmp_path / "G.job1.txt"
        log.write_text("old output")
        reattach = []
        recover_interrupted_jobs(state, all_jobs, reattach)
        assert not log.exists()
        assert (tmp_path / "G.job1.txt.prev").exists()
        assert state.records["G.job1"].prev_log == str(tmp_path / "G.job1.txt.prev")

    def test_dead_pid_no_log_prev_log_is_none(self, tmp_path):
        """If there is no log file, prev_log stays None."""
        state = _make_state(["G.job1"], pid=_dead_pid())
        all_jobs = _make_all_jobs(tmp_path, ["G.job1"])
        reattach = []
        recover_interrupted_jobs(state, all_jobs, reattach)
        assert state.records["G.job1"].prev_log is None

    def test_not_running_jobs_are_ignored(self, tmp_path):
        """Jobs not in 'running' status are left untouched."""
        state = _make_state(["G.job1"], status="finished", pid=_dead_pid())
        all_jobs = _make_all_jobs(tmp_path, ["G.job1"])
        reattach = []
        recover_interrupted_jobs(state, all_jobs, reattach)
        assert state.records["G.job1"].status == "finished"
        assert reattach == []

    def test_pid_none_treated_as_dead(self, tmp_path):
        """A running job with pid=None (should not normally happen) is reset."""
        state = RunState(version=1, executable="", output_dir="", max_parallel=4,
                         records={"G.job1": JobRecord(
                             status="running", result="unknown", pid=None,
                             start_time=1000.0, end_time=None, exit_code=None,
                         )})
        all_jobs = _make_all_jobs(tmp_path, ["G.job1"])
        reattach = []
        recover_interrupted_jobs(state, all_jobs, reattach)
        assert state.records["G.job1"].status == "not_started"
        assert reattach == []

    def test_mixed_dead_and_not_running(self, tmp_path):
        """Only running jobs with dead PIDs are reset; others unchanged."""
        state = _make_state(["G.a", "G.b"], pid=_dead_pid())
        state.records["G.b"].status = "finished"
        state.records["G.b"].result = "pass"
        all_jobs = _make_all_jobs(tmp_path, ["G.a", "G.b"])
        reattach = []
        recover_interrupted_jobs(state, all_jobs, reattach)
        assert state.records["G.a"].status == "not_started"
        assert state.records["G.b"].status == "finished"   # untouched


# ---------------------------------------------------------------------------
# Result determination — pass / fail / incomplete
# ---------------------------------------------------------------------------

class TestResultDetermination:
    """Verify the pass/fail/incomplete logic that _run_one applies after a job
    finishes.  We replicate the decision directly rather than launching a real
    subprocess, since the logic only depends on exit_code, log content, and the
    known_total / offset values preserved in the JobRecord."""

    def _determine_result(self, exit_code: int, log_content: str, tmp_path,
                          known_total: int = 0, offset: int = 0) -> str:
        """Mirror the result logic in _run_one."""
        log = tmp_path / "job.txt"
        _write_log(str(log), log_content)
        _passed, failed_count, _total, _ = _parse_test_counts(str(log))
        if exit_code == 0:
            if _passed == 0 and _total == 0 and known_total > 0 and offset < known_total:
                return "incomplete"
            return "pass"
        elif failed_count > 0:
            return "fail"
        else:
            return "incomplete"

    def test_exit0_is_pass(self, tmp_path):
        result = self._determine_result(0, _gtest_log(
            total=5, passed=("t1", "t2"), summary=True,
        ), tmp_path)
        assert result == "pass"

    def test_exit_nonzero_with_failures_is_fail(self, tmp_path):
        result = self._determine_result(-1, _gtest_log(
            total=5, passed=("t1",), failed=("f1", "f2"), summary=True,
        ), tmp_path)
        assert result == "fail"

    def test_sigint_no_failures_is_incomplete(self, tmp_path):
        """A job killed mid-run (e.g. SIGINT, exit_code=-2) with no recorded
        test failures is 'incomplete', not 'fail'."""
        result = self._determine_result(-2, _gtest_log(
            total=100, passed=tuple(f"t{i}" for i in range(30)),
        ), tmp_path)
        assert result == "incomplete"

    def test_crash_no_failures_is_incomplete(self, tmp_path):
        """SIGSEGV (exit_code=-11) with no test failures is 'incomplete'."""
        result = self._determine_result(-11, _gtest_log(
            total=50, passed=("t1",),
        ), tmp_path)
        assert result == "incomplete"

    def test_nonzero_exit_with_failures_is_fail_not_incomplete(self, tmp_path):
        """If any [  FAILED  ] test lines exist, result is 'fail' regardless
        of whether the run was also interrupted."""
        result = self._determine_result(-2, _gtest_log(
            total=10, passed=("p1",), failed=("f1",),
        ), tmp_path)
        assert result == "fail"

    def test_exit0_empty_log_is_pass(self, tmp_path):
        """exit_code=0 always means pass, even if the log is empty."""
        result = self._determine_result(0, "", tmp_path)
        assert result == "pass"

    # ------------------------------------------------------------------
    # False-pass detection: exit_code=0 but GTest ran 0 tests
    # ------------------------------------------------------------------

    def test_zero_tests_with_remaining_is_incomplete(self, tmp_path):
        """exit_code=0 with 0 tests run and offset < known_total is a false
        pass — the exclusion filter silently swallowed remaining tests."""
        result = self._determine_result(
            0, "[==========] Running 0 tests from 0 test suites.\n[  PASSED  ] 0 tests.\n",
            tmp_path, known_total=4354, offset=707,
        )
        assert result == "incomplete"

    def test_zero_tests_offset_equals_total_is_pass(self, tmp_path):
        """exit_code=0 with 0 tests run is a genuine pass when offset covers
        the full known total (every test already passed in previous runs)."""
        result = self._determine_result(
            0, "[==========] Running 0 tests from 0 test suites.\n[  PASSED  ] 0 tests.\n",
            tmp_path, known_total=707, offset=707,
        )
        assert result == "pass"

    def test_zero_tests_no_known_total_is_pass(self, tmp_path):
        """When known_total is 0 (never seen a non-zero run), we cannot detect
        a false pass — default to pass to avoid false negatives."""
        result = self._determine_result(
            0, "[==========] Running 0 tests from 0 test suites.\n[  PASSED  ] 0 tests.\n",
            tmp_path, known_total=0, offset=0,
        )
        assert result == "pass"

    def test_zero_tests_offset_exceeds_total_is_pass(self, tmp_path):
        """Defensive: if offset somehow exceeds known_total (stale estimate),
        do not mark incomplete — assume the job completed normally."""
        result = self._determine_result(
            0, "[==========] Running 0 tests from 0 test suites.\n[  PASSED  ] 0 tests.\n",
            tmp_path, known_total=100, offset=105,
        )
        assert result == "pass"

    def test_nonzero_tests_exit0_not_affected_by_false_pass_check(self, tmp_path):
        """When GTest actually ran tests and exit_code=0, false-pass logic
        must not interfere — result is always pass."""
        result = self._determine_result(
            0, _gtest_log(total=50, passed=tuple(f"t{i}" for i in range(50)), summary=True),
            tmp_path, known_total=500, offset=450,
        )
        assert result == "pass"


# ---------------------------------------------------------------------------
# tests_total refresh behaviour (poll thread logic)
# ---------------------------------------------------------------------------

def _apply_total_update(rec: JobRecord, current_total: int) -> None:
    """Replicate the poll thread's tests_total update logic."""
    offset = rec.tests_passed_offset
    if current_total > 0:
        rec.tests_total = current_total + offset


class TestTotalRefresh:
    """Verify that tests_total is always updated from the current run's
    GTest header, and that a 0-test run leaves the last known value intact."""

    def test_total_set_on_first_run(self):
        """When tests_total is 0 and GTest reports N tests, total is set."""
        rec = JobRecord(status="running", result="unknown", pid=None,
                        start_time=None, end_time=None, exit_code=None,
                        tests_passed_offset=0)
        _apply_total_update(rec, current_total=1474)
        assert rec.tests_total == 1474

    def test_total_updated_even_when_nonzero(self):
        """Stale tests_total from a prior run is replaced on the next run.
        This is the her_batched / hpmv_strided_batched fix."""
        rec = JobRecord(status="running", result="unknown", pid=None,
                        start_time=None, end_time=None, exit_code=None,
                        tests_passed_offset=122, tests_total=621)
        _apply_total_update(rec, current_total=548)
        assert rec.tests_total == 548 + 122  # = 670, not stale 621

    def test_zero_test_run_preserves_last_known_total(self):
        """When GTest runs 0 tests (all filtered out), the previous
        tests_total is preserved so false-pass detection can use it."""
        rec = JobRecord(status="running", result="unknown", pid=None,
                        start_time=None, end_time=None, exit_code=None,
                        tests_passed_offset=707, tests_total=4354)
        _apply_total_update(rec, current_total=0)
        assert rec.tests_total == 4354  # unchanged

    def test_total_includes_offset(self):
        """tests_total accounts for already-excluded (passed) tests so it
        represents the original unfiltered job size."""
        rec = JobRecord(status="running", result="unknown", pid=None,
                        start_time=None, end_time=None, exit_code=None,
                        tests_passed_offset=122, tests_total=0)
        _apply_total_update(rec, current_total=548)
        assert rec.tests_total == 670  # 548 remaining + 122 already done

    def test_successive_runs_always_use_current_count(self):
        """Across multiple resume cycles, tests_total tracks the shrinking
        remaining count + offset rather than sticking to the first value."""
        rec = JobRecord(status="running", result="unknown", pid=None,
                        start_time=None, end_time=None, exit_code=None,
                        tests_passed_offset=0, tests_total=0)
        _apply_total_update(rec, current_total=1474)   # run 1: all tests
        assert rec.tests_total == 1474
        rec.tests_passed_offset = 250
        _apply_total_update(rec, current_total=1224)   # run 2: 250 excluded
        assert rec.tests_total == 1224 + 250           # = 1474 (consistent)
        rec.tests_passed_offset = 500
        _apply_total_update(rec, current_total=974)    # run 3: 500 excluded
        assert rec.tests_total == 974 + 500            # = 1474 (still consistent)


# ---------------------------------------------------------------------------
# --exclude-pattern  CLI option and filter injection
# ---------------------------------------------------------------------------

def _apply_exclude_patterns(original_filter: str,
                             patterns: Sequence[str]) -> str:
    """Replicate the JobRunner exclude_patterns injection logic."""
    if not patterns:
        return original_filter
    return _build_exclusion_filter(original_filter, set(patterns)) or original_filter


def _full_filter_pipeline(original_filter: str,
                           exclude_patterns: Sequence[str],
                           per_test_excluded: set) -> str:
    """Replicate the fixed two-step filter pipeline in JobRunner._execute:
      1. Apply --exclude-pattern additions to the job's original filter.
      2. Apply accumulated per-test exclusions on top of step 1's result.
    This matches the fix for the silent-drop bug (Bug 1).
    """
    gtest_filter = _apply_exclude_patterns(original_filter, exclude_patterns)
    augmented = _build_exclusion_filter(gtest_filter, per_test_excluded)
    return augmented if augmented is not None else gtest_filter


class TestExcludePattern:
    """Tests for the --exclude-pattern CLI option and its effect on GTest filters."""

    # -- CLI parsing -------------------------------------------------------

    def test_no_exclude_pattern_defaults_to_none(self):
        """When --exclude-pattern is absent, args.exclude_patterns is None."""
        args = build_arg_parser().parse_args([])
        assert args.exclude_patterns is None

    def test_single_exclude_pattern_parsed(self):
        """A single --exclude-pattern value is stored as a one-element list."""
        args = build_arg_parser().parse_args(["--exclude-pattern", "*f32_c_LNN*"])
        assert args.exclude_patterns == ["*f32_c_LNN*"]

    def test_multiple_exclude_patterns_accumulated(self):
        """Repeated --exclude-pattern flags accumulate into a list."""
        args = build_arg_parser().parse_args([
            "--exclude-pattern", "*f32_c_LNN*",
            "--exclude-pattern", "*f64_c_LTN*",
        ])
        assert args.exclude_patterns == ["*f32_c_LNN*", "*f64_c_LTN*"]

    # -- Filter injection --------------------------------------------------

    def test_pattern_appended_to_filter_without_negative(self):
        """A filter with no existing negative section gets one added."""
        result = _apply_exclude_patterns("*trsv*quick*", ["*f32_c_LNN*"])
        assert result == "*trsv*quick*-*f32_c_LNN*"

    def test_pattern_appended_to_filter_with_existing_negative(self):
        """Exclude pattern is colon-joined after the existing negative section."""
        result = _apply_exclude_patterns("*trsv*quick*-*_ex*", ["*f32_c_LNN*"])
        assert result.startswith("*trsv*quick*-")
        negative = result.split("-", 1)[1]
        assert "*_ex*" in negative
        assert "*f32_c_LNN*" in negative

    def test_multiple_patterns_all_appear_in_negative(self):
        """All supplied exclude patterns appear in the filter's negative section."""
        result = _apply_exclude_patterns(
            "*trsv*quick*-*_ex*",
            ["*f32_c_LNN*", "*f64_c_LTN*"],
        )
        negative = result.split("-", 1)[1]
        assert "*f32_c_LNN*" in negative
        assert "*f64_c_LTN*" in negative

    def test_positive_section_unchanged(self):
        """Exclude patterns must never alter the positive part of the filter."""
        original = "*trsv_batched*quick*-*_ex*"
        result = _apply_exclude_patterns(original, ["*f32_c_LNN*"])
        positive = result.split("-", 1)[0]
        assert positive == "*trsv_batched*quick*"

    def test_no_patterns_returns_filter_unchanged(self):
        """With an empty exclude list the original filter is returned as-is."""
        original = "*trsv*quick*-*_ex*"
        assert _apply_exclude_patterns(original, []) == original

    def test_pattern_does_not_affect_excluded_tests_accumulation(self):
        """Exclude patterns are separate from the per-test exclusion set used
        for partial resume; _build_exclusion_filter with per-test names still
        works correctly when patterns are also present."""
        base = "*trsv*quick*-*f32_c_LNN*"   # already has exclude pattern applied
        per_test = {"_/trsv.suite/test_alpha", "_/trsv.suite/test_beta"}
        result = _build_exclusion_filter(base, per_test)
        assert result is not None
        negative = result.split("-", 1)[1]
        assert "*f32_c_LNN*" in negative
        assert "_/trsv.suite/test_alpha" in negative
        assert "_/trsv.suite/test_beta" in negative

    # -- Bug 1 fix: exclude-pattern must survive per-test exclusion step -----

    def test_exclude_pattern_preserved_with_per_test_exclusions(self):
        """--exclude-pattern must not be silently dropped when per-test
        exclusions from state are also applied (the two-step pipeline bug)."""
        original = "*trsv*quick*-*_batched*:*_ex*:*tbsv*:"
        per_test = {"_/trsv.suite/test_alpha", "_/trsv.suite/test_beta"}
        result = _full_filter_pipeline(original, ["*f32_c_LNN*"], per_test)
        negative = result.split("-", 1)[1]
        assert "*f32_c_LNN*" in negative, "exclude-pattern was silently dropped"
        assert "_/trsv.suite/test_alpha" in negative
        assert "_/trsv.suite/test_beta" in negative

    def test_exclude_pattern_preserved_with_many_per_test_exclusions(self):
        """With a large per-test exclusion set the pattern still survives."""
        original = "*trsv*quick*-*_batched*:*_ex*:"
        per_test = {f"_/trsv.suite/test_{i}" for i in range(500)}
        result = _full_filter_pipeline(original, ["*quick_trsv_fortran_f32_c_*_F*"], per_test)
        negative = result.split("-", 1)[1]
        assert "*quick_trsv_fortran_f32_c_*_F*" in negative

    def test_no_exclude_pattern_with_per_test_exclusions(self):
        """With no --exclude-pattern the pipeline still produces the correct
        per-test exclusion filter."""
        original = "*trsv*quick*-*_batched*:*_ex*:"
        per_test = {"_/trsv.suite/test_alpha"}
        result = _full_filter_pipeline(original, [], per_test)
        assert "_/trsv.suite/test_alpha" in result
        assert result.startswith("*trsv*quick*-")

    # -- Bug 2 fix: long filters must not use @file syntax -------------------

    def test_short_filter_threshold(self):
        """A filter under the threshold fits inline (no GTEST_FILTER env needed)."""
        short = "*trsv*quick*-*_ex*"
        assert len(short) <= 32768

    def test_long_filter_exceeds_threshold(self):
        """A filter with thousands of per-test exclusions exceeds _FILTER_LIST_THRESHOLD chars."""
        original = "*trsv*quick*-*_batched*:*_ex*:*tbsv*:"
        per_test = {f"_/trsv.suite/quick_trsv_small_f64_r_LNN_{i}_{i}_{i}" for i in range(800)}
        result = _full_filter_pipeline(original, [], per_test)
        assert len(result) > _FILTER_LIST_THRESHOLD, "expected filter to exceed inline threshold"


# ---------------------------------------------------------------------------
# _list_remaining_tests
# ---------------------------------------------------------------------------

def _make_list_output(test_map: dict) -> str:
    """Build a --gtest_list_tests output string.

    test_map: {suite_name: [test_name, ...]}  (suite_name WITHOUT trailing dot)
    """
    lines = []
    for suite, tests in test_map.items():
        lines.append(f"{suite}.")
        for t in tests:
            lines.append(f"  {t}")
    return "\n".join(lines) + "\n"


class TestListRemainingTests:
    """Tests for _list_remaining_tests() — the --gtest_list_tests-based fallback."""

    def _run(self, test_map: dict, excluded: Optional[set] = None,
             patterns: Optional[list] = None, returncode: int = 0):
        """Call _list_remaining_tests with a mocked subprocess."""
        import unittest.mock as mock

        stdout = _make_list_output(test_map)
        completed = mock.MagicMock()
        completed.returncode = returncode
        completed.stdout = stdout

        with mock.patch("run_tests._subprocess_run", return_value=completed):
            return _list_remaining_tests(
                "/fake/rocblas-test",
                "*trsv*quick*-*_ex*",
                excluded or set(),
                patterns or [],
            )

    def test_returns_all_tests_when_no_exclusions(self):
        """All listed tests are returned when nothing is excluded."""
        test_map = {"blas2/quick_trsv": ["test_a", "test_b", "test_c"]}
        result = self._run(test_map)
        assert result == [
            "blas2/quick_trsv.test_a",
            "blas2/quick_trsv.test_b",
            "blas2/quick_trsv.test_c",
        ]

    def test_excluded_set_removes_tests(self):
        """Tests listed in ``excluded`` are removed from the result."""
        test_map = {"suite": ["test_a", "test_b", "test_c"]}
        excluded = {"suite.test_b"}
        result = self._run(test_map, excluded=excluded)
        assert result is not None
        assert "suite.test_b" not in result
        assert "suite.test_a" in result
        assert "suite.test_c" in result

    def test_exclude_pattern_removes_matching_tests(self):
        """Tests matching an --exclude-pattern glob are removed."""
        test_map = {"blas2/trsv": [
            "quick_trsv_f32_c_LNN_128",
            "quick_trsv_f32_c_LNU_128",
            "quick_trsv_f64_r_LNN_128",
        ]}
        result = self._run(test_map, patterns=["*f32_c_*"])
        assert result is not None
        names = set(result)
        assert "blas2/trsv.quick_trsv_f32_c_LNN_128" not in names
        assert "blas2/trsv.quick_trsv_f32_c_LNU_128" not in names
        assert "blas2/trsv.quick_trsv_f64_r_LNN_128" in names

    def test_both_excluded_and_pattern_applied(self):
        """Both the exclusion set and pattern globs are applied simultaneously."""
        test_map = {"suite": ["test_a", "test_b", "test_c"]}
        excluded = {"suite.test_a"}
        result = self._run(test_map, excluded=excluded, patterns=["*test_b*"])
        assert result == ["suite.test_c"]

    def test_returns_empty_list_when_all_excluded(self):
        """Returns empty list (not None) when every listed test is excluded."""
        test_map = {"suite": ["test_a", "test_b"]}
        excluded = {"suite.test_a", "suite.test_b"}
        result = self._run(test_map, excluded=excluded)
        assert result is not None
        assert result == []

    def test_returns_none_on_subprocess_failure(self):
        """Returns None when the subprocess exits with a non-zero code."""
        test_map = {"suite": ["test_a"]}
        result = self._run(test_map, returncode=1)
        assert result is None

    def test_returns_none_on_subprocess_exception(self):
        """Returns None when the subprocess raises an exception."""
        import unittest.mock as mock

        with mock.patch("run_tests._subprocess_run", side_effect=OSError("no such file")):
            result = _list_remaining_tests("/bad/path", "*filter*", set(), [])
        assert result is None

    def test_multiple_suites_parsed_correctly(self):
        """Test names from multiple suites are all collected with correct suite prefix."""
        test_map = {
            "suite_a": ["alpha", "beta"],
            "suite_b": ["gamma"],
        }
        result = self._run(test_map)
        assert result is not None
        assert "suite_a.alpha" in result
        assert "suite_a.beta" in result
        assert "suite_b.gamma" in result
        assert len(result) == 3

    def test_threshold_constant_matches_usage(self):
        """_FILTER_LIST_THRESHOLD is the documented 32768 byte limit."""
        assert _FILTER_LIST_THRESHOLD == 32768

    def test_strips_get_param_annotation_from_test_name(self):
        """GTest --gtest_list_tests annotates parameterized tests with
        '  # GetParam() = ...'.  These annotations must be stripped, otherwise
        the resulting filter contains spaces/parens that GTest can't match."""
        import unittest.mock as mock

        stdout = (
            "_/trsv.\n"
            "  blas2_tensile/quick_trsv_f64_r_LNN  # GetParam() = (1, 2, 3)\n"
            "  blas2_tensile/quick_trsv_f64_r_LNT  # GetParam() = (4, 5, 6)\n"
        )
        completed = mock.MagicMock()
        completed.returncode = 0
        completed.stdout = stdout

        with mock.patch("run_tests._subprocess_run", return_value=completed):
            result = _list_remaining_tests("/x", "*trsv*", set(), [])

        assert result is not None
        # Names must NOT include the "# GetParam()..." annotation
        for name in result:
            assert "#" not in name, f"annotation leaked into test name: {name}"
            assert "GetParam" not in name
        assert "_/trsv.blas2_tensile/quick_trsv_f64_r_LNN" in result
        assert "_/trsv.blas2_tensile/quick_trsv_f64_r_LNT" in result

    def test_excluded_set_matches_stripped_names_for_parameterized(self):
        """Per-test exclusions use the bare name (no annotation), so stripping
        must happen *before* the excluded-set check."""
        import unittest.mock as mock

        stdout = (
            "suite.\n"
            "  test_a  # GetParam() = (1)\n"
            "  test_b  # GetParam() = (2)\n"
        )
        completed = mock.MagicMock()
        completed.returncode = 0
        completed.stdout = stdout

        with mock.patch("run_tests._subprocess_run", return_value=completed):
            result = _list_remaining_tests("/x", "*", {"suite.test_a"}, [])

        assert result == ["suite.test_b"]
