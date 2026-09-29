"""Tests for harvest_support_claims.py."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harvest_support_claims import (
    DEVELOP,
    DROP_BAD_PATH,
    DROP_MIN_RUNS,
    DROP_NO_DEPTH,
    DROP_NOT_PASSED,
    DROP_SHORTFALL,
    Cell,
    FetchError,
    MalformedSummary,
    block_cells,
    extract_blocks,
    gtest_name,
    harvest,
    main,
    read_log,
    run_gh,
    run_git,
    split_streams,
)
from support_sidecar import Claim, load, render

_TESTDATA = Path(__file__).resolve().parent / "testdata"
_TS = "2026-09-28T05:47:38.5917461Z "
_SWEEP = "integration-test-bundles/quick/ConvolutionFwdPointwise/Default/sweep.json"
_SINGLE = (
    "integration-test-bundles/quick/BatchnormFwdInference/nchw/fp32/Small/Small.json"
)


def _summary(**overrides: object) -> dict:
    summary = {
        "schema_version": 1,
        "mode": "enforcing",
        "run": {"engine": "MIOPEN_ENGINE", "arch": "gfx1030", "platform": "windows"},
        "graphs": {"found": 3, "with_claims": 0, "selected": 3, "ran": 3, "queried": 3},
        "verdicts": {
            "confirmed": 0,
            "accepted": 0,
            "failed_in_use": 0,
            "broken": 0,
            "errored": 0,
            "unclaimed": 3,
        },
        "counters_consistent": True,
        "unenforced": {
            "no_applicable_claim": 0,
            "not_opened": 0,
            "not_selected": 0,
            "skipped_before_run": 0,
        },
        "harness_defects": {"missed_query": 0},
        "claim_failures": [],
        "failed_in_use": [],
        "unclaimed_support": [
            {
                "bundle": _SWEEP,
                "reached": "verified",
                "required": "verified",
                "cases": ["2_8_3_3_fp32_nchw", "2_8_3_3_fp16_nchw"],
            },
            {"bundle": _SINGLE, "reached": "verified", "required": "verified"},
        ],
    }
    summary.update(overrides)
    return summary


def _gtest_output(
    summary: object,
    failed: Optional[List[str]] = None,
    skipped: Optional[List[str]] = None,
    ran_line: bool = True,
    mode: str = "ENFORCING",
) -> List[str]:
    """Output of one bundle-harness run, as the harness prints it."""
    lines = ["[==========] Running 3 tests from 2 test suites."]
    for name in skipped or []:
        lines.append(f"[  SKIPPED ] {name} (0 ms)")
    for name in failed or []:
        lines.append(f"[  FAILED  ] {name} (12 ms)")
    if ran_line:
        lines.append("[==========] 3 tests from 2 test suites ran. (40 ms total)")
    lines.append("[  PASSED  ] 1 test.")
    if failed:
        lines.append(f"[  FAILED  ] {len(failed)} tests, listed below:")
        lines.extend(f"[  FAILED  ] {name}" for name in failed)
    lines.append(f"==== SUPPORT CLAIM SUMMARY ({mode}) ====")
    if isinstance(summary, str):
        lines.extend(summary.splitlines())
    else:
        lines.extend(
            json.dumps({"support_claim_summary": summary}, indent=2).splitlines()
        )
    return lines


def _log(*streams: tuple) -> str:
    """Joins (ctest_number or None, lines) pairs into a GitHub job log."""
    out = []
    for number, lines in streams:
        prefix = f"{number}: " if number is not None else ""
        out.extend(f"{_TS}{prefix}{line}" for line in lines)
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Stream splitting
# ---------------------------------------------------------------------------


class TestSplitStreams(unittest.TestCase):
    def test_timestamp_prefix_and_ansi_are_stripped(self) -> None:
        text = f"{_TS}23: \x1b[0;32m[       OK ]\x1b[m a.b (1 ms)\n{_TS}plain\n"
        self.assertEqual(
            split_streams(text), {"23": ["[       OK ] a.b (1 ms)"], "": ["plain"]}
        )

    def test_interleaved_prefixes_keep_per_stream_order(self) -> None:
        text = _log((3, ["a1"]), (23, ["b1"]), (3, ["a2"]), (23, ["b2"]))
        self.assertEqual(split_streams(text), {"3": ["a1", "a2"], "23": ["b1", "b2"]})


# ---------------------------------------------------------------------------
# Block extraction
# ---------------------------------------------------------------------------


class TestExtractBlocks(unittest.TestCase):
    def test_single_block_parses(self) -> None:
        blocks = extract_blocks(_log((23, _gtest_output(_summary()))), log="job.log")
        self.assertEqual(len(blocks), 1)
        self.assertIsNone(blocks[0].error)
        self.assertEqual(blocks[0].stream, "23")
        self.assertEqual(blocks[0].log, "job.log")
        self.assertEqual(blocks[0].summary["run"]["arch"], "gfx1030")

    def test_ctest_reprint_yields_identical_second_block(self) -> None:
        output = _gtest_output(_summary(), failed=["quick_A.x"])
        blocks = extract_blocks(_log((23, output), (None, output)))
        self.assertEqual([b.stream for b in blocks], ["23", ""])
        self.assertEqual(blocks[0].summary, blocks[1].summary)
        self.assertEqual(blocks[0].not_passed, blocks[1].not_passed)

    def test_failed_and_skipped_names_are_collected(self) -> None:
        output = _gtest_output(
            _summary(), failed=["quick_A.x", "quick_A.y"], skipped=["quick_B.z"]
        )
        (block,) = extract_blocks(_log((23, output)))
        self.assertEqual(block.not_passed, {"quick_A.x", "quick_A.y", "quick_B.z"})

    def test_failed_count_line_is_not_a_test_name(self) -> None:
        output = _gtest_output(_summary(), failed=["quick_A.x"])
        (block,) = extract_blocks(_log((23, output)))
        self.assertEqual(block.not_passed, {"quick_A.x"})

    def test_results_of_other_ctest_streams_are_not_attached(self) -> None:
        other = [
            "[  SKIPPED ] TestGpuPlanBuilder.Fusion (0 ms)",
            "[==========] 1 test from 1 test suite ran. (0 ms total)",
        ]
        (block,) = extract_blocks(_log((3, other), (23, _gtest_output(_summary()))))
        self.assertEqual(block.not_passed, frozenset())

    def test_interleaved_streams_still_parse(self) -> None:
        output = _gtest_output(_summary())
        noise = [f"noise {i}" for i in range(len(output))]
        pairs = []
        for line, other in zip(output, noise):
            pairs.append((23, [line]))
            pairs.append((13, [other]))
        (block,) = extract_blocks(_log(*pairs))
        self.assertIsNone(block.error)

    def test_warning_mode_header_is_recognised(self) -> None:
        output = _gtest_output(
            _summary(mode="warning_only"), mode="WARNING ONLY -- NOT ENFORCED"
        )
        (block,) = extract_blocks(_log((23, output)))
        self.assertIsNone(block.error)

    def test_log_without_block_yields_nothing(self) -> None:
        self.assertEqual(
            extract_blocks(
                _log((23, ["[==========] 1 test from 1 test suite ran. (0 ms total)"]))
            ),
            [],
        )

    def test_trailing_lines_after_json_are_ignored(self) -> None:
        output = _gtest_output(_summary()) + ["[  PASSED  ] extra", "}"]
        (block,) = extract_blocks(_log((23, output)))
        self.assertIsNone(block.error)


# ---------------------------------------------------------------------------
# Block rejection
# ---------------------------------------------------------------------------


class TestRejectBlocks(unittest.TestCase):
    def _error(self, summary: object, **kwargs: object) -> Optional[str]:
        (block,) = extract_blocks(_log((23, _gtest_output(summary, **kwargs))))
        if block.error is not None:
            self.assertIsNone(block.summary)
        return block.error

    def test_missing_ran_line(self) -> None:
        self.assertIn("ran.", self._error(_summary(), ran_line=False))

    def test_truncated_json(self) -> None:
        text = json.dumps({"support_claim_summary": _summary()}, indent=2)
        self.assertIn("does not parse", self._error(text[: len(text) // 2]))

    def test_missing_summary_object(self) -> None:
        self.assertIn("support_claim_summary", self._error('{"other": {}}'))

    def test_wrong_schema_version(self) -> None:
        self.assertIn("schema_version", self._error(_summary(schema_version=2)))

    def test_bad_or_missing_arch(self) -> None:
        for arch in ("", "unknown", "GFX942", "gfx942:sramecc+", None):
            summary = _summary()
            summary["run"] = dict(summary["run"], arch=arch)
            with self.subTest(arch=arch):
                self.assertIn("run.arch", self._error(summary))

    def test_bad_platform(self) -> None:
        summary = _summary()
        summary["run"] = dict(summary["run"], platform="macos")
        self.assertIn("run.platform", self._error(summary))

    def test_empty_engine(self) -> None:
        summary = _summary()
        summary["run"] = dict(summary["run"], engine="")
        self.assertIn("run.engine", self._error(summary))

    def test_missing_run(self) -> None:
        summary = copy.deepcopy(_summary())
        del summary["run"]
        self.assertIn("run", self._error(summary))

    def test_inconsistent_counters(self) -> None:
        self.assertIn(
            "counters_consistent", self._error(_summary(counters_consistent=False))
        )

    def test_unknown_or_missing_mode(self) -> None:
        for mode in ("enforce", "WARNING ONLY", "", None):
            with self.subTest(mode=mode):
                self.assertIn("mode", self._error(_summary(mode=mode)))

    def test_harness_defect(self) -> None:
        for defects in (
            {"missed_query": 2},
            {"missed_query": 0, "other": 1},
            {"missed_query": False},
            {"missed_query": "0"},
        ):
            with self.subTest(defects=defects):
                self.assertIn(
                    "harness_defects.", self._error(_summary(harness_defects=defects))
                )

    def test_harness_defects_without_missed_query(self) -> None:
        for defects in ({}, 0, None, [0]):
            with self.subTest(defects=defects):
                self.assertIn(
                    "missed_query", self._error(_summary(harness_defects=defects))
                )


# ---------------------------------------------------------------------------
# Gtest names
# ---------------------------------------------------------------------------


class TestGtestName(unittest.TestCase):
    def test_sweep_case(self) -> None:
        self.assertEqual(
            gtest_name(_SWEEP, "2_8_3_3_fp32_nchw_dil1x1_postpad1x1_prepad1x1"),
            "quick_ConvolutionFwdPointwise_Default"
            ".2_8_3_3_fp32_nchw_dil1x1_postpad1x1_prepad1x1",
        )

    def test_single_graph(self) -> None:
        self.assertEqual(
            gtest_name(_SINGLE, ""),
            "quick_BatchnormFwdInference_nchw_fp32_Small.Small",
        )

    def test_segments_and_case_are_sanitized(self) -> None:
        self.assertEqual(
            gtest_name("integration-test-bundles/quick/Op-A/b.c/sweep.json", "x-1.5"),
            "quick_Op_A_b_c.x_1_5",
        )

    def test_non_ascii_becomes_one_underscore_per_byte(self) -> None:
        self.assertEqual(
            gtest_name("integration-test-bundles/quick/Op/\u00e9.json", ""),
            "quick_Op.__",
        )

    def test_paths_outside_bundle_root_have_no_name(self) -> None:
        for bundle in (
            "/abs/integration-test-bundles/quick/A/B.json",
            "integration-test-bundles/B.json",
            "integration-test-bundles/quick/../A/B.json",
        ):
            with self.subTest(bundle=bundle):
                self.assertIsNone(gtest_name(bundle, ""))


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------


def _block(summary: dict, **kwargs: object):
    (block,) = extract_blocks(_log((23, _gtest_output(summary, **kwargs))))
    assert block.error is None, block.error
    return block


def _cell(bundle: str, case: str = "", arch: str = "gfx1030") -> Cell:
    return Cell(bundle, case, arch, "windows", "MIOPEN_ENGINE")


class TestBlockCells(unittest.TestCase):
    def test_one_cell_per_case_and_per_single_graph(self) -> None:
        cells = block_cells(_block(_summary()))
        self.assertEqual(
            cells.kept,
            {
                _cell(_SWEEP, "2_8_3_3_fp32_nchw"),
                _cell(_SWEEP, "2_8_3_3_fp16_nchw"),
                _cell(_SINGLE),
            },
        )
        self.assertEqual(cells.dropped, {})
        self.assertEqual(cells.kept.pop().lane, "gfx1030/windows")

    def test_failed_sweep_case_is_dropped(self) -> None:
        failed = "quick_ConvolutionFwdPointwise_Default.2_8_3_3_fp32_nchw"
        cells = block_cells(_block(_summary(), failed=[failed]))
        self.assertEqual(
            cells.dropped, {_cell(_SWEEP, "2_8_3_3_fp32_nchw"): DROP_NOT_PASSED}
        )
        self.assertEqual(cells.drops, {DROP_NOT_PASSED: 1})
        self.assertIn(_cell(_SWEEP, "2_8_3_3_fp16_nchw"), cells.kept)

    def test_skipped_single_graph_is_dropped(self) -> None:
        skipped = "quick_BatchnormFwdInference_nchw_fp32_Small.Small"
        cells = block_cells(_block(_summary(), skipped=[skipped]))
        self.assertEqual(cells.dropped, {_cell(_SINGLE): DROP_NOT_PASSED})
        self.assertEqual(cells.drops, {DROP_NOT_PASSED: 1})

    def test_depth_shortfall_and_missing_depth_are_dropped(self) -> None:
        summary = _summary(
            unclaimed_support=[
                {"bundle": _SINGLE, "reached": "executed", "required": "verified"},
                {"bundle": _SWEEP, "reached": "verified", "cases": ["a"]},
                {
                    "bundle": _SWEEP,
                    "reached": "unknown",
                    "required": "applicable",
                    "cases": ["b"],
                },
            ]
        )
        cells = block_cells(_block(summary))
        self.assertEqual(cells.kept, set())
        self.assertEqual(cells.drops, {DROP_SHORTFALL: 1, DROP_NO_DEPTH: 2})

    def test_reached_above_required_is_kept(self) -> None:
        summary = _summary(
            unclaimed_support=[
                {"bundle": _SINGLE, "reached": "verified", "required": "executed"}
            ]
        )
        self.assertEqual(block_cells(_block(summary)).kept, {_cell(_SINGLE)})

    def test_bundle_outside_root_is_dropped(self) -> None:
        summary = _summary(
            unclaimed_support=[
                {
                    "bundle": "/abs/A/B.json",
                    "reached": "verified",
                    "required": "verified",
                }
            ]
        )
        self.assertEqual(block_cells(_block(summary)).drops, {DROP_BAD_PATH: 1})

    def test_entry_lane_overrides_run(self) -> None:
        summary = _summary(
            unclaimed_support=[
                {
                    "bundle": _SINGLE,
                    "arch": "gfx942",
                    "reached": "verified",
                    "required": "verified",
                }
            ]
        )
        self.assertEqual(
            block_cells(_block(summary)).kept, {_cell(_SINGLE, arch="gfx942")}
        )

    def test_malformed_entries_raise(self) -> None:
        bad_entries = (
            [{"bundle": _SINGLE, "arch": "unknown", "reached": "verified"}],
            [{"bundle": _SWEEP, "cases": "a", "reached": "verified"}],
            [{"reached": "verified"}],
            "not a list",
        )
        for entries in bad_entries:
            with self.subTest(entries=entries):
                with self.assertRaises(MalformedSummary):
                    block_cells(_block(_summary(unclaimed_support=entries)))

    def test_failures_are_carried_with_their_lane(self) -> None:
        failure = {
            "bundle": _SINGLE,
            "verdict": "broken",
            "reason": "no engine",
            "reached": "not-reached",
            "required": "verified",
        }
        cells = block_cells(
            _block(_summary(claim_failures=[failure], failed_in_use=[failure]))
        )
        expected = dict(
            failure, arch="gfx1030", platform="windows", engine="MIOPEN_ENGINE"
        )
        self.assertEqual(cells.claim_failures, [expected])
        self.assertEqual(cells.failed_in_use, [expected])


# ---------------------------------------------------------------------------
# Logs and runs
# ---------------------------------------------------------------------------

_FP32 = "quick_ConvolutionFwdPointwise_Default.2_8_3_3_fp32_nchw"
_GFX942 = {"engine": "MIOPEN_ENGINE", "arch": "gfx942", "platform": "linux"}


def _doubled(summary: dict, **kwargs: object) -> str:
    """A failed CTest: the block, then CTest's unprefixed reprint of it."""
    lines = _gtest_output(summary, **kwargs)
    return _log((23, lines), (None, lines))


def _single_log(summary: dict, **kwargs: object) -> str:
    return _log((23, _gtest_output(summary, **kwargs)))


def _failure(**overrides: object) -> dict:
    return {"bundle": _SINGLE, "verdict": "broken", "reason": "no engine", **overrides}


class TestReadLog(unittest.TestCase):
    def test_reprinted_block_counts_once(self) -> None:
        report, cells = read_log("a.log", _doubled(_summary(), failed=[_FP32]))
        self.assertEqual((report.blocks, report.unique), (2, 1))
        self.assertEqual(report.lanes, {"gfx1030/windows"})
        self.assertEqual(
            report.status(), "2 blocks (1 unique) gfx1030/windows enforcing"
        )
        self.assertIsNone(report.error)
        self.assertEqual(len(cells), 2)

    def test_warning_only_block_is_harvested_and_shows_its_mode(self) -> None:
        output = _gtest_output(
            _summary(mode="warning_only"), mode="WARNING ONLY -- NOT ENFORCED"
        )
        report, cells = read_log("a.log", _log((23, output)))
        self.assertEqual(report.modes, {"warning_only"})
        self.assertEqual(
            report.status(), "1 block (1 unique) gfx1030/windows warning_only"
        )
        _, enforcing = read_log("b.log", _single_log(_summary()))
        self.assertEqual(cells, enforcing)

    def test_no_block(self) -> None:
        report, cells = read_log("a.log", _log((3, ["[  PASSED  ] 5 tests."])))
        self.assertEqual((report.status(), cells), ("no block", []))

    def test_any_rejected_block_rejects_the_log(self) -> None:
        text = _log(
            (13, _gtest_output(_summary())),
            (23, _gtest_output(_summary(counters_consistent=False))),
        )
        report, cells = read_log("a.log", text)
        self.assertEqual(cells, [])
        self.assertEqual(report.error, "stream 23: counters_consistent is not true")
        self.assertTrue(report.status().startswith("REJECTED: stream 23"))

    def test_malformed_entry_rejects_the_log(self) -> None:
        text = _log((None, _gtest_output(_summary(unclaimed_support="x"))))
        report, cells = read_log("a.log", text)
        self.assertEqual(cells, [])
        self.assertIn("unprefixed stream: unclaimed_support", report.error)

    def test_block_printed_by_the_harness(self) -> None:
        # An excerpt of a CI job log: the gtest results before the block, and
        # the block exactly as SupportClaimReport.cpp printed it.
        path = _TESTDATA / "support_claim_summary_hip_mlops_gfx942_linux.log"
        text = path.read_bytes().decode("utf-8")
        (block,) = extract_blocks(text)
        self.assertEqual(len(block.not_passed), 4)
        self.assertIn(
            "quick_SdpaFwdRuntimeScale_bhsd_fp8_hd128_nomask_batch_Small.Small",
            block.not_passed,
        )
        report, (cells,) = read_log(path.name, text)
        self.assertIsNone(report.error)
        self.assertEqual(report.lanes, {"gfx942/linux"})
        self.assertEqual(cells.dropped, {})
        self.assertEqual(len(cells.kept), 48)
        self.assertEqual({c.engine for c in cells.kept}, {"HIP_MLOPS_ENGINE"})
        bundles = Counter(c.bundle for c in cells.kept)
        self.assertEqual(
            bundles,
            {
                "integration-test-bundles/quick/ResampleFwd/Default/sweep.json": 24,
                "integration-test-bundles/quick/ResampleFwd/Variant2/sweep.json": 24,
            },
        )


class TestHarvest(unittest.TestCase):
    def test_single_run_claims_every_kept_cell(self) -> None:
        result = harvest([[("a.log", _single_log(_summary()))]])
        self.assertEqual(
            set(result.claims),
            {
                _cell(_SWEEP, "2_8_3_3_fp32_nchw"),
                _cell(_SWEEP, "2_8_3_3_fp16_nchw"),
                _cell(_SINGLE),
            },
        )
        self.assertEqual(result.drops, {})

    def test_a_drop_in_any_run_vetoes_the_cell(self) -> None:
        result = harvest(
            [
                [("a.log", _single_log(_summary()))],
                [("b.log", _doubled(_summary(), failed=[_FP32]))],
            ]
        )
        self.assertNotIn(_cell(_SWEEP, "2_8_3_3_fp32_nchw"), result.claims)
        self.assertEqual(len(result.claims), 2)
        self.assertEqual(result.drops, {"gfx1030/windows": {DROP_NOT_PASSED: 1}})

    def test_min_runs_counts_distinct_runs(self) -> None:
        only_single = _summary(
            unclaimed_support=[
                {"bundle": _SINGLE, "reached": "verified", "required": "verified"}
            ]
        )
        result = harvest(
            [
                [("a.log", _single_log(_summary())), ("b.log", _doubled(_summary()))],
                [("c.log", _single_log(only_single))],
            ],
            min_runs=2,
        )
        self.assertEqual(result.claims, [_cell(_SINGLE)])
        self.assertEqual(result.drops, {"gfx1030/windows": {DROP_MIN_RUNS: 2}})

    def test_rejected_log_contributes_nothing(self) -> None:
        rejected = _log(
            (23, _gtest_output(_summary(), failed=[_FP32])),
            (13, _gtest_output(_summary(schema_version=2))),
        )
        result = harvest([[("a.log", _single_log(_summary())), ("b.log", rejected)]])
        self.assertEqual(len(result.claims), 3)
        self.assertEqual(result.drops, {})
        self.assertIsNotNone(result.logs[1].error)

    def test_lane_filter(self) -> None:
        other = _summary(run=_GFX942, claim_failures=[_failure()])
        result = harvest(
            [
                [
                    ("a.log", _single_log(_summary(), failed=[_FP32])),
                    ("b.log", _single_log(other)),
                ]
            ],
            lane="gfx942/linux",
        )
        self.assertEqual({c.lane for c in result.claims}, {"gfx942/linux"})
        self.assertEqual(len(result.claims), 3)
        self.assertEqual(result.drops, {})
        self.assertEqual([_failure(**_GFX942)], result.claim_failures)

    def test_failures_are_deduplicated(self) -> None:
        summary = _summary(claim_failures=[_failure()], failed_in_use=[_failure()])
        result = harvest(
            [[("a.log", _doubled(summary))], [("b.log", _single_log(summary))]]
        )
        lane = {"engine": "MIOPEN_ENGINE", "arch": "gfx1030", "platform": "windows"}
        self.assertEqual(result.claim_failures, [_failure(**lane)])
        self.assertEqual(result.failed_in_use, [_failure(**lane)])


class TestMain(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.root = Path(self._dir.name)

    def _write(self, name: str, text: str) -> str:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return str(path)

    def _main(self, *argv: str, gh: Optional["FakeGh"] = None) -> tuple:
        gh = gh or FakeGh()
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = main(list(argv), gh)
            except SystemExit as exc:
                code = exc.code
        return code, out.getvalue(), err.getvalue()

    def test_table(self) -> None:
        log = self._write("run/a.log", _doubled(_summary(), failed=[_FP32]))
        code, out, _ = self._main(str(self.root / "run"))
        self.assertEqual(code, 0)
        self.assertIn(f"{log}  2 blocks (1 unique) gfx1030/windows enforcing\n", out)
        self.assertIn(
            "gfx1030/windows        2 claims; dropped: 1 gtest failed or skipped", out
        )
        self.assertIn(f"gfx1030/windows MIOPEN_ENGINE {_SINGLE}\n", out)
        self.assertIn(f"{_SWEEP} [2_8_3_3_fp16_nchw]", out)

    def test_json(self) -> None:
        log = self._write("a.log", _single_log(_summary()))
        code, out, _ = self._main(log, "--format", "json")
        self.assertEqual(code, 0)
        report = json.loads(out)
        self.assertEqual(report["logs"][0]["modes"], ["enforcing"])
        self.assertEqual(
            report["lanes"], {"gfx1030/windows": {"claims": 3, "drops": {}}}
        )
        self.assertIn(
            {
                "bundle": _SINGLE,
                "arch": "gfx1030",
                "platform": "windows",
                "engine": "MIOPEN_ENGINE",
            },
            report["claims"],
        )

    def test_rejected_log_exits_1(self) -> None:
        log = self._write("a.log", _single_log(_summary(schema_version=2)))
        code, out, _ = self._main(log)
        self.assertEqual(code, 1)
        self.assertIn("REJECTED: stream 23: schema_version 2", out)
        self.assertIn("Missing (0)\n", out)

    def test_log_without_block_is_missing(self) -> None:
        self._write("run/a.log", _single_log(_summary()))
        empty = self._write("run/b.log", "no block here\n")
        code, out, err = self._main(str(self.root / "run"))
        self.assertEqual(code, 0)
        self.assertIn(f"Missing (1)\n  {empty}\n", out)
        self.assertIn("gfx1030/windows        3 claims", out)
        self.assertIn("warning: 1 log has no support block", err)

    def test_usage_errors_exit_2(self) -> None:
        log = self._write("a.log", _single_log(_summary()))
        self.root.joinpath("empty").mkdir()
        cases = {
            "missing path": [str(self.root / "missing.log")],
            "directory without logs": [str(self.root / "empty")],
            "min-runs above runs": [log, "--min-runs", "2"],
            "min-runs zero": [log, "--min-runs", "0"],
            "lane without platform": [log, "--lane", "gfx942"],
            "lane with bad platform": [log, "--lane", "gfx942/macos"],
            "min-runs above the default nightly": ["--min-runs", "2"],
            "run ID zero": ["--run", "0"],
            "run ID twice": ["--run", "7", "--run", "7"],
            "jobs not a regex": ["--run", "7", "--jobs", "("],
            "write without lane": [log, "--write"],
        }
        for name, argv in cases.items():
            with self.subTest(name):
                gh = FakeGh()
                self.assertEqual(self._main(*argv, gh=gh)[0], 2)
                self.assertEqual(gh.calls, [])


class FakeGit:
    """Serves the git commands that read sidecars from the tip of develop."""

    REV = "0123abc"

    def __init__(self, develop: Optional[Dict[str, str]] = None, fail: bool = False):
        self.develop = develop or {}
        self.fail = fail
        self.calls: List[List[str]] = []

    def __call__(self, args: List[str]) -> bytes:
        self.calls.append(args)
        command = args[2:]  # after "-C ROOT"
        if self.fail:
            raise FetchError(f"git {' '.join(args)}: fatal: unable to access")
        if command[0] == "fetch":
            return b""
        if command[0] == "rev-parse":
            return f"{self.REV}\n".encode()
        if command[:4] == ["ls-tree", "-z", "--name-only", self.REV]:
            names = [n for n in command[5:] if n in self.develop]
            return "".join(n + "\0" for n in names).encode()
        if command[:2] == ["cat-file", "blob"]:
            return self.develop[command[2].split(":./", 1)[1]].encode()
        raise AssertionError(f"unexpected git {args}")


class TestWrite(unittest.TestCase):
    _LANE = ("--lane", "gfx1030/windows")
    _WRITE = (*_LANE, "--write")

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        for bundle in (_SWEEP, _SINGLE):
            path = self.root / bundle
            path.parent.mkdir(parents=True)
            path.write_text("{}\n", encoding="utf-8")
        self.sweep = self.root / _SWEEP.replace("sweep.json", "support.json")
        self.single = self.root / _SINGLE.replace(".json", ".support.json")
        self.git = FakeGit()

    def _main(self, *argv: str, summary: Optional[dict] = None) -> tuple:
        log = self.root / "a.log"
        log.write_text(_single_log(summary or _summary()), encoding="utf-8")
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            code = main([str(log), *argv], FakeGh(), self.root, self.git)
        return code, err.getvalue()

    def _claim(self, case: str = "") -> Claim:
        return Claim(case, "MIOPEN_ENGINE", "gfx1030", "windows")

    def _rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def test_writes_new_sidecars(self) -> None:
        code, err = self._main(*self._WRITE)
        self.assertEqual(code, 0)
        self.assertEqual(
            load(self.sweep),
            {self._claim("2_8_3_3_fp16_nchw"), self._claim("2_8_3_3_fp32_nchw")},
        )
        self.assertEqual(load(self.single), {self._claim()})
        self.assertIn(f"changed {self._rel(self.sweep)} (+2)\n", err)
        self.assertIn(f"changed {self._rel(self.single)} (+1)\n", err)
        self.assertIn("2 sidecars changed, +3 claims", err)

    def test_without_write_names_the_changes_and_writes_nothing(self) -> None:
        old = {self._claim("2_8_3_3_fp16_nchw")}
        self.sweep.write_bytes(render(self.sweep, old).encode("utf-8"))
        code, err = self._main(*self._LANE)
        self.assertEqual(code, 0)
        self.assertIn(f"would change {self._rel(self.sweep)} (+1)\n", err)
        self.assertIn(f"would change {self._rel(self.single)} (+1)\n", err)
        self.assertIn("2 sidecars would change, +2 claims", err)
        self.assertEqual(load(self.sweep), old)
        self.assertFalse(self.single.exists())

    def test_adds_to_an_existing_sidecar_and_keeps_every_old_claim(self) -> None:
        old = {
            Claim("x", "MIOPEN_ENGINE", "gfx942", "linux"),
            Claim("2_8_3_3_fp16_nchw", "OTHER_ENGINE", "gfx1030", "windows"),
        }
        self.sweep.write_bytes(render(self.sweep, old).encode("utf-8"))
        # Develop has one of the old claims: the sidecar is ahead, not behind.
        develop = {Claim("x", "MIOPEN_ENGINE", "gfx942", "linux")}
        self.git.develop = {self._rel(self.sweep): render(self.sweep, develop)}
        self.assertEqual(self._main(*self._WRITE)[0], 0)
        self.assertEqual(
            load(self.sweep),
            old | {self._claim("2_8_3_3_fp16_nchw"), self._claim("2_8_3_3_fp32_nchw")},
        )

    def test_reads_each_sidecar_from_the_fetched_develop(self) -> None:
        self._main(*self._WRITE)
        root = ["-C", str(self.root)]
        self.assertEqual(
            self.git.calls[:3],
            [
                [*root, "fetch", "--quiet", "--no-tags", DEVELOP[0], "develop"],
                [*root, "rev-parse", "FETCH_HEAD"],
                [
                    *root,
                    *("ls-tree", "-z", "--name-only", FakeGit.REV, "--"),
                    self._rel(self.single),
                    self._rel(self.sweep),
                ],
            ],
        )

    def test_sidecar_behind_develop_is_refused(self) -> None:
        claim = Claim("", "MIOPEN_ENGINE", "gfx942", "linux")
        self.git.develop = {self._rel(self.single): render(self.single, {claim})}
        for argv, suffix in ((self._LANE, "\n"), (self._WRITE, "; wrote nothing\n")):
            with self.subTest(argv=argv):
                code, err = self._main(*argv)
                self.assertEqual(code, 1)
                self.assertIn(
                    f"{self._rel(self.single)}: lacks 1 of develop's claims;"
                    f" rebase onto develop{suffix}",
                    err,
                )
                self.assertFalse(self.sweep.exists() or self.single.exists())

    def test_unfetchable_develop_exits_2_and_writes_nothing(self) -> None:
        self.git = FakeGit(fail=True)
        code, err = self._main(*self._WRITE)
        self.assertEqual(code, 2)
        self.assertIn("unable to access; wrote nothing", err)
        self.assertFalse(self.sweep.exists() or self.single.exists())

    def test_second_run_changes_nothing(self) -> None:
        self._main(*self._WRITE)
        before = self.sweep.read_bytes(), self.single.read_bytes()
        code, err = self._main(*self._WRITE)
        self.assertEqual(code, 0)
        self.assertIn("0 sidecars changed, +0 claims", err)
        self.assertEqual((self.sweep.read_bytes(), self.single.read_bytes()), before)

    def test_other_lane_writes_nothing_and_fetches_nothing(self) -> None:
        self.assertEqual(self._main("--lane", "gfx942/linux", "--write")[0], 0)
        self.assertFalse(self.sweep.exists() or self.single.exists())
        self.assertEqual(self.git.calls, [])

    def test_without_lane_fetches_nothing(self) -> None:
        self.assertEqual(self._main()[0], 0)
        self.assertEqual(self.git.calls, [])

    def test_refused_sidecar_writes_nothing(self) -> None:
        text = '{"version": 1, "claims": {}}\n'
        self.single.write_bytes(text.encode("utf-8"))
        code, err = self._main(*self._WRITE)
        self.assertEqual(code, 1)
        self.assertIn("not in the form this writer renders; wrote nothing", err)
        self.assertFalse(self.sweep.exists())
        self.assertEqual(self.single.read_bytes(), text.encode("utf-8"))

    def test_missing_bundle_writes_nothing(self) -> None:
        (self.root / _SINGLE).unlink()
        code, err = self._main(*self._WRITE)
        self.assertEqual(code, 1)
        self.assertIn(f"{_SINGLE}: no such bundle", err)
        self.assertFalse(self.sweep.exists() or self.single.exists())

    def test_rejected_log_writes_nothing(self) -> None:
        code, err = self._main(*self._WRITE, summary=_summary(schema_version=2))
        self.assertEqual(code, 1)
        self.assertIn("a log was rejected; wrote nothing", err)
        self.assertFalse(self.sweep.exists() or self.single.exists())
        self.assertEqual(self.git.calls, [])

    def test_run_git_errors(self) -> None:
        with mock.patch("subprocess.run", side_effect=FileNotFoundError):
            with self.assertRaisesRegex(FetchError, "not installed"):
                run_git(["fetch"])
        failed = mock.Mock(returncode=128, stdout=b"", stderr=b"fatal: no remote\n")
        with mock.patch("subprocess.run", return_value=failed):
            with self.assertRaisesRegex(FetchError, "^git fetch x: fatal: no remote$"):
                run_git(["fetch", "x"])

    def test_run_git_returns_stdout(self) -> None:
        done = mock.Mock(returncode=0, stdout=b"abc\n", stderr=b"")
        with mock.patch("subprocess.run", return_value=done) as run:
            self.assertEqual(run_git(["rev-parse", "HEAD"]), b"abc\n")
        self.assertEqual(run.call_args.args[0], ["git", "rev-parse", "HEAD"])


_JOB = "Test miopenprovider (gfx1030, windows)"


def _nightly(
    run_id: int,
    event: str = "schedule",
    branch: str = "develop",
    status: str = "completed",
) -> dict:
    return {"id": run_id, "event": event, "head_branch": branch, "status": status}


class FakeGh:
    """Serves `gh api` requests from in-memory runs, jobs and logs."""

    def __init__(
        self,
        nightly: Optional[List[dict]] = None,
        runs: Optional[Dict[int, Tuple[str, List[Tuple[int, str, str]]]]] = None,
        logs: Optional[Dict[int, str]] = None,
    ) -> None:
        self.nightly = nightly or []
        self.runs = runs or {}
        self.logs = logs or {}
        self.calls: List[List[str]] = []

    def __call__(self, args: List[str]) -> bytes:
        self.calls.append(args)
        path = args[0].split("?")[0].split("/")
        if path[-1] == "runs" and path[-2] == "therock-multi-arch-ci-nightly.yml":
            return json.dumps({"workflow_runs": self.nightly}).encode()
        if path[-2] == "runs" and int(path[-1]) in self.runs:
            return json.dumps({"status": self.runs[int(path[-1])][0]}).encode()
        if path[-1] == "jobs" and int(path[-2]) in self.runs:
            jobs = self.runs[int(path[-2])][1]
            return "".join(
                json.dumps({"id": i, "name": n, "conclusion": c}) + "\n"
                for i, n, c in jobs
            ).encode()
        if path[-1] == "logs" and int(path[-2]) in self.logs:
            return self.logs[int(path[-2])].encode()
        raise FetchError(f"gh api {args[0]}: HTTP 404: Not Found")

    def fetched_logs(self) -> List[str]:
        return [a[0].split("/")[-2] for a in self.calls if a[0].endswith("/logs")]


class TestFetch(unittest.TestCase):
    def _gh(self, status: str = "completed", **overrides: str) -> FakeGh:
        conclusions = {"11": "success", "12": "success", "13": "cancelled"}
        conclusions.update(overrides)
        jobs = [
            (13, _JOB + " retry", conclusions["13"]),
            (11, _JOB, conclusions["11"]),
            (12, "Build miopenprovider", conclusions["12"]),
        ]
        text = _single_log(_summary())
        nightly = [
            _nightly(10, event="workflow_dispatch"),
            _nightly(9, status="in_progress"),
            _nightly(8, branch="users/x/nightly-toggle"),
            _nightly(7),
        ]
        return FakeGh(nightly, {7: (status, jobs)}, {11: text, 12: text, 13: text})

    def _main(self, gh: FakeGh, *argv: str) -> tuple:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(list(argv), gh)
        return code, out.getvalue(), err.getvalue()

    def test_no_runs_fetches_latest_nightly(self) -> None:
        gh = self._gh()
        code, out, err = self._main(gh)
        self.assertEqual(code, 0)
        self.assertIn(
            "latest nightly: run 7\n"
            "  https://github.com/ROCm/rocm-libraries/actions/runs/7\n",
            err,
        )
        self.assertIn(
            "https://github.com/ROCm/rocm-libraries/actions/runs/7/job/11  1 block",
            out,
        )
        self.assertIn(
            "Missing (1)\n"
            f"  https://github.com/ROCm/rocm-libraries/actions/runs/7/job/13  cancelled  {_JOB} retry\n",
            out,
        )
        self.assertIn("warning: 1 log has no support block", err)
        self.assertIn("gfx1030/windows        3 claims", out)
        self.assertEqual(gh.fetched_logs(), ["11"])
        self.assertTrue(gh.calls[0][0].endswith("/runs?per_page=50"))

    def test_failed_job_without_block_is_missing(self) -> None:
        # A timed-out job fails before it prints its block.
        gh = self._gh(**{"12": "failure"})
        gh.logs[12] = "##[error]The job has exceeded the maximum execution time\n"
        code, out, err = self._main(gh, "--run", "7", "--jobs", "miopenprovider")
        self.assertEqual(code, 0)
        self.assertIn(
            "Missing (2)\n"
            "  https://github.com/ROCm/rocm-libraries/actions/runs/7/job/12  failure  Build miopenprovider\n"
            f"  https://github.com/ROCm/rocm-libraries/actions/runs/7/job/13  cancelled  {_JOB} retry\n",
            out,
        )
        self.assertIn("gfx1030/windows        3 claims", out)
        self.assertIn("warning: 2 logs have no support block", err)

    def test_missing_in_json(self) -> None:
        code, out, _ = self._main(self._gh(), "--run", "7", "--format", "json")
        self.assertEqual(code, 0)
        self.assertEqual(
            json.loads(out)["missing"],
            [
                {
                    "log": "https://github.com/ROCm/rocm-libraries/actions/runs/7/job/13",
                    "conclusion": "cancelled",
                    "name": _JOB + " retry",
                }
            ],
        )

    def test_no_warning_when_nothing_is_missing(self) -> None:
        gh = self._gh(**{"13": "success"})
        code, out, err = self._main(gh, "--run", "7")
        self.assertEqual(code, 0)
        self.assertIn("Missing (0)\n", out)
        self.assertNotIn("warning", err)

    def test_run_is_fetched_by_id(self) -> None:
        gh = self._gh()
        code, out, err = self._main(gh, "--run", "7")
        self.assertEqual(code, 0)
        self.assertNotIn("latest nightly", err)
        self.assertIn("actions/runs/7/job/11", out)
        self.assertFalse(any("workflows" in a[0] for a in gh.calls))

    def test_fetched_and_read_runs_count_together(self) -> None:
        log = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False)
        self.addCleanup(Path(log.name).unlink)
        with log:
            log.write(_single_log(_summary()))
        code, out, _ = self._main(self._gh(), log.name, "--run", "7", "--min-runs", "2")
        self.assertEqual(code, 0)
        self.assertIn("gfx1030/windows        3 claims", out)

    def test_jobs_selects_by_name(self) -> None:
        gh = self._gh()
        code, _, _ = self._main(gh, "--run", "7", "--jobs", "^Build")
        self.assertEqual(code, 0)
        self.assertEqual(gh.fetched_logs(), ["12"])

    def test_default_jobs_are_every_provider_test(self) -> None:
        names = {
            21: "Linux::release / Test gfx94X-dcgpu / Test miopenprovider / "
            "Test miopenprovider (shard 1/1) (gfx94X-dcgpu)",
            22: "Linux::release / Test gfx94X-dcgpu / Test hipkernelprovider / "
            "Test hipkernelprovider (shard 1/1) (gfx94X-dcgpu)",
            23: "Windows::release / Test gfx1151 / Test hipblasltprovider / "
            "Test hipblasltprovider (shard 1/1) (gfx1151)",
            24: "Linux::release / Test gfx94X-dcgpu / Test hipblaslt / "
            "Test hipblaslt (shard 1/6) (gfx94X-dcgpu)",
            25: "Linux::release / Test gfx94X-dcgpu / Test hipdnn / "
            "Test hipdnn (shard 1/1) (gfx94X-dcgpu)",
            26: "Build miopenprovider",
        }
        text = _single_log(_summary())
        gh = FakeGh(
            runs={7: ("completed", [(i, n, "success") for i, n in names.items()])},
            logs={i: text for i in names},
        )
        self.assertEqual(self._main(gh, "--run", "7")[0], 0)
        self.assertEqual(sorted(gh.fetched_logs()), ["21", "22", "23"])

    def test_failed_job_is_fetched(self) -> None:
        gh = self._gh(**{"11": "failure"})
        self.assertEqual(self._main(gh, "--run", "7")[0], 0)
        self.assertEqual(gh.fetched_logs(), ["11"])

    def test_unfetchable_runs_exit_2(self) -> None:
        cases = {
            "no nightly": (FakeGh(), [], "no completed"),
            "no completed develop nightly": (
                FakeGh(
                    [
                        _nightly(10, event="workflow_dispatch"),
                        _nightly(9, status="queued"),
                        _nightly(8, branch="main"),
                    ]
                ),
                [],
                "no completed",
            ),
            "unknown run": (self._gh(), ["--run", "8"], "HTTP 404"),
            "run in progress": (
                self._gh("in_progress"),
                ["--run", "7"],
                "not completed",
            ),
            "no matching job": (self._gh(), ["--run", "7", "--jobs", "Lint"], "no job"),
            "no finished job": (
                self._gh(**{"11": "skipped"}),
                ["--run", "7"],
                "no succeeded or failed",
            ),
        }
        for name, (gh, argv, message) in cases.items():
            with self.subTest(name):
                code, out, err = self._main(gh, *argv)
                self.assertEqual(code, 2)
                self.assertEqual(out, "")
                self.assertIn(message, err)

    def test_run_gh_errors(self) -> None:
        with mock.patch("subprocess.run", side_effect=FileNotFoundError):
            with self.assertRaisesRegex(FetchError, "not installed"):
                run_gh(["repos/x"])
        failed = mock.Mock(returncode=1, stdout=b"", stderr=b"HTTP 403: Forbidden\n")
        with mock.patch("subprocess.run", return_value=failed):
            with self.assertRaisesRegex(FetchError, "^gh api repos/x: HTTP 403"):
                run_gh(["repos/x"])

    def test_run_gh_returns_stdout(self) -> None:
        done = mock.Mock(returncode=0, stdout=b"{}", stderr=b"")
        with mock.patch("subprocess.run", return_value=done) as run:
            self.assertEqual(run_gh(["repos/x", "--paginate"]), b"{}")
        self.assertEqual(run.call_args.args[0], ["gh", "api", "repos/x", "--paginate"])


if __name__ == "__main__":
    unittest.main()
