"""Tests for harvest_support_claims.py."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harvest_support_claims import extract_blocks, split_streams

_TS = "2026-09-28T05:47:38.5917461Z "
_SWEEP = "integration-test-bundles/quick/ConvolutionFwdPointwise/Default/sweep.json"
_SINGLE = (
    "integration-test-bundles/quick/BatchnormFwdInference/nchw/fp32/Small/Small.json"
)


def _summary(**overrides: object) -> dict:
    summary = {
        "schema_version": 1,
        "mode": "ENFORCING",
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


if __name__ == "__main__":
    unittest.main()
