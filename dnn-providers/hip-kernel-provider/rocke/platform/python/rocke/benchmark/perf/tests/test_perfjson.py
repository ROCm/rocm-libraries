# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the PerfJSON launcher line (pure, no GPU)."""
import io
import json
import unittest

from rocke.benchmark.perf import harness, perfjson


class TestPayload(unittest.TestCase):
    def test_absent_fields_are_dropped(self):
        self.assertEqual(perfjson.payload(ms=1.5), {"ms": 1.5})

    def test_non_finite_timing_is_dropped_not_serialized(self):
        # json.dumps would emit bare NaN/Infinity, which is not valid JSON and would
        # make the whole line unparseable rather than one field missing.
        line = perfjson.format_line(ms=float("nan"), tflops=float("inf"), gbps=500.0)
        self.assertEqual(
            json.loads(line.removeprefix(perfjson.PREFIX)), {"gbps": 500.0}
        )

    def test_extra_fields_pass_through(self):
        self.assertEqual(
            perfjson.payload(ms=1.0, variant="tile128")["variant"], "tile128"
        )


class TestEmit(unittest.TestCase):
    def test_emitted_line_is_what_the_harness_parses(self):
        buf = io.StringIO()
        perfjson.emit(
            stream=buf, ms=2.0, tflops=12.0, gbps=500.0, bad_count=0, total=64
        )
        out = harness._perf_from_stdout("some log noise\n" + buf.getvalue())
        self.assertEqual(out, {"ms_median": 2.0, "tflops": 12.0, "gbs": 500.0})

    def test_verification_fields_round_trip_under_verify(self):
        buf = io.StringIO()
        perfjson.emit(stream=buf, ms=2.0, max_abs_diff=0.0, bad_count=0, total=64)
        self.assertEqual(
            harness._verification_from_stdout(buf.getvalue(), verified=True),
            {"max_abs_diff": 0.0, "bad_count": 0, "total": 64, "ok": True},
        )

    def test_emit_returns_the_line_it_printed(self):
        buf = io.StringIO()
        line = perfjson.emit(stream=buf, ms=1.0)
        self.assertEqual(buf.getvalue(), line + "\n")
        self.assertTrue(line.startswith(perfjson.PREFIX))


if __name__ == "__main__":
    unittest.main()
