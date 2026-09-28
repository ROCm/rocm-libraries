"""Tests for harvest_support_claims.py."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harvest_support_claims import (
    DROP_BAD_PATH,
    DROP_MIN_RUNS,
    DROP_NO_DEPTH,
    DROP_NOT_PASSED,
    DROP_SHORTFALL,
    Cell,
    MalformedSummary,
    block_cells,
    extract_blocks,
    gtest_name,
    harvest,
    main,
    read_log,
    split_streams,
)

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
        "unenforced": 0,
        "harness_defects": 0,
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
        self.assertEqual(report.status(), "2 blocks (1 unique) gfx1030/windows")
        self.assertIsNone(report.error)
        self.assertEqual(len(cells), 2)

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

    def _main(self, *argv: str) -> tuple:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = main(list(argv))
            except SystemExit as exc:
                code = exc.code
        return code, out.getvalue(), err.getvalue()

    def test_table(self) -> None:
        log = self._write("run/a.log", _doubled(_summary(), failed=[_FP32]))
        code, out, _ = self._main(str(self.root / "run"))
        self.assertEqual(code, 0)
        self.assertIn(f"{log}  2 blocks (1 unique) gfx1030/windows", out)
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
            "no runs": [],
        }
        for name, argv in cases.items():
            with self.subTest(name):
                self.assertEqual(self._main(*argv)[0], 2)


if __name__ == "__main__":
    unittest.main()
