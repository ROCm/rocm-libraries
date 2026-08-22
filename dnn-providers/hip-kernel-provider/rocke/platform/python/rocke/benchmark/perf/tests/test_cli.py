# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the CLI wiring (no-GPU paths, pure)."""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout

from rocke.benchmark.perf import schema
from rocke.benchmark.perf.tool import cli, store
import rocke.benchmark.perf.tool.cli as cli_mod


def _rec(arch="gfx950", kernel="k", shape=None, *, busy=None):
    counters = {"busy_cycles": busy, "total_clocks": 1000} if busy is not None else {}
    return {
        "schema": schema.SCHEMA_VERSION,
        "run": {"run_id": "r", "arch": arch, "timestamp": "t"},
        "kernel": {"kernel_name": kernel, "op": "gemm", "shape": shape or {"M": 8}},
        "wall": {},
        "counters": counters,
        "resources": {},
        "derived": {"busy_fraction": busy / 1000} if busy is not None else {},
        "captured_counters": sorted(counters),
        "verify": {},
    }


def _run(argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = cli.main(argv)
    return code, buf.getvalue()


class TestCompareCmd(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _seed(self, *busies, kernel="k", shape=None):
        for b in busies:
            store.append(_rec(kernel=kernel, shape=shape, busy=b), cache=self.cache)

    def test_compare_regressed_exit_1(self):
        self._seed(1000, 1200)  # +20%
        code, out = _run(
            [
                "--cache",
                self.cache,
                "compare",
                "--arch",
                "gfx950",
                "--op",
                "gemm",
                "--kernel-name",
                "k",
                "--shape",
                '{"M":8}',
            ]
        )
        self.assertEqual(code, 1)
        self.assertIn("REGRESSED", out)

    def test_compare_within_noise_exit_0(self):
        self._seed(1000, 1010)  # +1%
        code, out = _run(
            [
                "--cache",
                self.cache,
                "compare",
                "--arch",
                "gfx950",
                "--op",
                "gemm",
                "--kernel-name",
                "k",
                "--shape",
                '{"M":8}',
            ]
        )
        self.assertEqual(code, 0)
        self.assertIn("within noise", out)

    def test_compare_json_global_position(self):
        self._seed(1000, 1200)
        code, out = _run(
            [
                "--json",
                "--cache",
                self.cache,
                "compare",
                "--arch",
                "gfx950",
                "--op",
                "gemm",
                "--kernel-name",
                "k",
                "--shape",
                '{"M":8}',
            ]
        )
        data = json.loads(out)
        self.assertEqual(data[0]["verdict"], "regressed")

    def test_json_and_cache_after_subcommand(self):
        # --json / --cache must also work AFTER the subcommand (position-independent)
        self._seed(1000, 1200)
        code, out = _run(
            [
                "compare",
                "--arch",
                "gfx950",
                "--op",
                "gemm",
                "--kernel-name",
                "k",
                "--shape",
                '{"M":8}',
                "--cache",
                self.cache,
                "--json",
            ]
        )
        data = json.loads(out)
        self.assertEqual(data[0]["verdict"], "regressed")

    def test_compare_all_scans_every_identity(self):
        self._seed(1000, 1200, kernel="a")
        self._seed(500, 505, kernel="b")
        code, out = _run(["--json", "--cache", self.cache, "compare", "--all"])
        verdicts = {d["verdict"] for d in json.loads(out)}
        self.assertEqual(verdicts, {"regressed", "within_noise"})
        self.assertEqual(code, 1)  # any regression -> nonzero

    def test_compare_no_history(self):
        code, out = _run(["--cache", self.cache, "compare", "--all"])
        self.assertEqual(code, 0)
        self.assertIn("no history", out)

    def test_compare_requires_identity_or_all(self):
        with self.assertRaises(SystemExit):
            _run(["--cache", self.cache, "compare"])


class TestOccupancyCmd(unittest.TestCase):
    def test_occupancy_reads_and_prints(self):
        fake = {"vgpr": 24, "sgpr": 32, "lds_bytes": 2048, "occupancy": 12}
        orig = cli_mod._occupancy.resources
        cli_mod._occupancy.resources = lambda data, arch: fake
        try:
            with tempfile.NamedTemporaryFile(suffix=".hsaco") as f:
                f.write(b"\x7fELF fake")
                f.flush()
                code, out = _run(["--json", "occupancy", f.name, "--arch", "gfx950"])
        finally:
            cli_mod._occupancy.resources = orig
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["occupancy"], 12)

    def test_occupancy_empty_errors(self):
        orig = cli_mod._occupancy.resources
        cli_mod._occupancy.resources = lambda data, arch: {}
        try:
            with tempfile.NamedTemporaryFile(suffix=".hsaco") as f:
                f.write(b"x")
                f.flush()
                with self.assertRaises(SystemExit):
                    _run(["occupancy", f.name, "--arch", "gfx950"])
        finally:
            cli_mod._occupancy.resources = orig


class TestProfileCmd(unittest.TestCase):
    """profile wiring, with the GPU harness monkeypatched."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = self._tmp.name
        self._orig = cli_mod._harness.profile
        self.calls = []

        def fake_profile(
            cmd,
            arch,
            *,
            match=None,
            label=None,
            op="unknown",
            shape=None,
            warmup=0,
            per_dispatch=False,
            warn=None,
        ):
            self.calls.append(
                {
                    "cmd": cmd,
                    "match": match,
                    "label": label,
                    "op": op,
                    "shape": shape,
                    "warmup": warmup,
                    "per_dispatch": per_dispatch,
                }
            )
            return _rec(
                arch=arch, kernel=(label or "dispatched_sym"), shape=shape, busy=1000
            )

        cli_mod._harness.profile = fake_profile

    def tearDown(self):
        cli_mod._harness.profile = self._orig
        self._tmp.cleanup()

    def test_double_dash_is_stripped(self):
        _run(
            [
                "--cache",
                self.cache,
                "profile",
                "--arch",
                "gfx950",
                "--",
                "python3",
                "-m",
                "run_it",
            ]
        )
        self.assertEqual(self.calls[0]["cmd"], ["python3", "-m", "run_it"])

    def test_no_double_dash_also_works(self):
        _run(["--cache", self.cache, "profile", "--arch", "gfx950", "myprog", "--flag"])
        self.assertEqual(self.calls[0]["cmd"], ["myprog", "--flag"])

    def test_label_and_match_passthrough(self):
        _run(
            [
                "--cache",
                self.cache,
                "profile",
                "--arch",
                "gfx950",
                "--kernel-name",
                "mygemm",
                "--match-kernel",
                "gemm_tile",
                "--shape",
                '{"M":8}',
                "--",
                "prog",
            ]
        )
        c = self.calls[0]
        self.assertEqual(c["label"], "mygemm")  # identity
        self.assertEqual(c["match"], "gemm_tile")  # profiler filter
        self.assertEqual(c["shape"], {"M": 8})

    def test_profile_stores_and_reports(self):
        code, out = _run(
            [
                "--json",
                "--cache",
                self.cache,
                "profile",
                "--arch",
                "gfx950",
                "--kernel-name",
                "k",
                "--shape",
                '{"M":8}',
                "--",
                "prog",
            ]
        )
        # first run -> no baseline, exit 0, and a record landed in the store
        self.assertEqual(code, 0)
        self.assertEqual(len(store.load(cache=self.cache)), 1)
        self.assertEqual(json.loads(out)["selfcheck"]["verdict"], "no_baseline")

    def test_profile_without_timing_is_not_stored(self):
        def fail_profile(*args, **kwargs):
            raise RuntimeError(
                "kernel command did not emit valid PerfJSON timing (finite ms)"
            )

        cli_mod._harness.profile = fail_profile
        with self.assertRaisesRegex(SystemExit, "valid PerfJSON timing"):
            _run(
                [
                    "--cache",
                    self.cache,
                    "profile",
                    "--arch",
                    "gfx950",
                    "--",
                    "prog",
                ]
            )
        self.assertEqual(store.load(cache=self.cache), [])


class TestParser(unittest.TestCase):
    def test_missing_subcommand_errors(self):
        with self.assertRaises(SystemExit):
            _run([])

    def test_bad_shape_json_errors(self):
        with self.assertRaises(SystemExit):
            _run(
                [
                    "compare",
                    "--arch",
                    "gfx950",
                    "--op",
                    "gemm",
                    "--kernel-name",
                    "k",
                    "--shape",
                    "not-json",
                ]
            )

    def test_negative_warmup_errors(self):
        with self.assertRaises(SystemExit):
            _run(["profile", "--arch", "gfx950", "--warmup", "-1", "--", "prog"])

    def test_nonpositive_repeats_error(self):
        for repeats in ("0", "-1"):
            with self.subTest(repeats=repeats), self.assertRaises(SystemExit):
                _run(
                    [
                        "profile",
                        "--arch",
                        "gfx950",
                        "--repeats",
                        repeats,
                        "--",
                        "prog",
                    ]
                )

    def test_threshold_and_noise_k_reject_invalid_values(self):
        parser = cli_mod._build_parser()
        for option in ("--threshold", "--noise-k"):
            for value in ("-1", "nan", "inf", "-inf"):
                commands = (
                    [
                        "profile",
                        "--arch",
                        "gfx950",
                        f"{option}={value}",
                        "--",
                        "prog",
                    ],
                    ["compare", "--all", f"{option}={value}"],
                )
                for command in commands:
                    with self.subTest(option=option, value=value, command=command[0]):
                        with self.assertRaises(SystemExit):
                            parser.parse_args(command)

    def test_threshold_and_noise_k_accept_zero_and_positive_values(self):
        parser = cli_mod._build_parser()
        for command in ("profile", "compare"):
            prefix = (
                [command, "--arch", "gfx950"]
                if command == "profile"
                else [command, "--all"]
            )
            for value in ("0", "1.5"):
                args = parser.parse_args(
                    [*prefix, "--threshold", value, "--noise-k", value]
                )
                with self.subTest(command=command, value=value):
                    self.assertEqual(args.threshold, float(value))
                    self.assertEqual(args.noise_k, float(value))


if __name__ == "__main__":
    unittest.main()
