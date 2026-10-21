# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import json
import random
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import run_random_hipblaslt_gemms as runner
import sample_hipblaslt_gemms as sampler


def bench_result(problem, *, norm="0", atol="1e-6"):
    keys = (
        "M",
        "N",
        "K",
        "batch_count",
        "a_type",
        "b_type",
        "c_type",
        "d_type",
        "compute_type",
        "transA",
        "transB",
        "alpha",
        "beta",
        "scaleA",
        "scaleB",
        "bias_vector",
        "activation_type",
        "bias_type",
    )
    header = [k.lower() if k in ("M", "N", "K") else k for k in keys]
    values = []
    for key in keys:
        value = problem[key]
        if key == "bias_vector":
            value = int(value)
        elif key == "compute_type":
            value = value[2:]
        values.append(str(value))
    return (
        "[0]:"
        + ",".join(header)
        + ",norm_error,atol,rtol\n"
        + ",".join(values)
        + f",{norm},{atol},1e-6\n"
        + "    --kernel name: gemm_kernel\n"
    )


class RandomGemmTests(unittest.TestCase):
    def test_explicit_cases_run_in_addition_to_random_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            problem = sampler.sample_gemm(random.Random(10))
            cases = root / "explicit.json"
            cases.write_text(
                json.dumps(
                    [
                        {
                            "id": "required",
                            "description": "Regression input",
                            "problem": problem,
                        }
                    ]
                )
            )
            config = root / "config.json"
            config.write_text(
                json.dumps({"vm": {"gpu": {"device": {"gfx_target_version": 90500}}}})
            )
            executed = []

            def run_case(args, directory, problem, config):
                executed.append(directory.name)
                return {
                    "outcome": (
                        "failed" if directory.name.startswith("explicit-") else "passed"
                    ),
                    "reason": (
                        "race reported"
                        if directory.name.startswith("explicit-")
                        else None
                    ),
                    "exit_code": 0,
                }

            with patch.object(runner, "run_case", run_case):
                status = runner.main(
                    [
                        "--count",
                        "3",
                        "--threads",
                        "2",
                        "--explicit-cases",
                        str(cases),
                        "--bench",
                        sys.executable,
                        "--rocjitsu",
                        sys.executable,
                        "--config",
                        str(config),
                        "--output",
                        str(root / "all"),
                    ]
                )
            self.assertEqual(status, 1)
            self.assertEqual(
                set(executed),
                {"explicit-required", "case-0001", "case-0002", "case-0003"},
            )
            summary = json.loads((root / "all/summary.json").read_text())
            self.assertEqual(
                (
                    summary["requested"],
                    summary["sampled"],
                    summary["failed"],
                    summary["passed"],
                ),
                (4, 3, 1, 3),
            )
            self.assertEqual(summary["sources"]["explicit"]["failed"], 1)
            self.assertTrue(summary["complete"])
            report = json.loads((root / "all/cases.json").read_text())
            self.assertEqual(len(report), 4)
            self.assertIn("race reported", (root / "all/cases.md").read_text())
            with patch.object(runner, "run_case", side_effect=OSError("cannot launch")):
                self.assertEqual(
                    runner.main(
                        [
                            "--count", "2", "--bench", sys.executable,
                            "--rocjitsu", sys.executable, "--config", str(config),
                            "--output", str(root / "launch-failure"),
                        ]
                    ),
                    1,
                )
            failure = json.loads((root / "launch-failure/summary.json").read_text())
            self.assertTrue(failure["complete"])
            self.assertEqual(failure["failed"], 2)
            self.assertEqual(
                json.loads((root / "launch-failure/cases.json").read_text())[0]["result"]["reason"],
                "execution error: cannot launch",
            )
            runner.main(["--count", "3", "--dry-run", "--output", str(root / "random")])
            for iteration in range(1, 4):
                name = f"case-{iteration:04d}/input.yaml"
                self.assertEqual(
                    (root / "all" / name).read_text(),
                    (root / "random" / name).read_text(),
                )
            runner.main(
                [
                    "--count",
                    "0",
                    "--explicit-cases",
                    str(cases),
                    "--dry-run",
                    "--output",
                    str(root / "explicit"),
                ]
            )
            self.assertFalse((root / "explicit/case-0001").exists())
            runner.main(
                [
                    "--explicit-case",
                    "required",
                    "--explicit-cases",
                    str(cases),
                    "--dry-run",
                    "--output",
                    str(root / "one"),
                ]
            )
            self.assertEqual(
                (root / "one/explicit-required/input.yaml").read_text(),
                (root / "explicit/explicit-required/input.yaml").read_text(),
            )
            runner.main(
                [
                    "--case",
                    "2",
                    "--count",
                    "3",
                    "--explicit-cases",
                    str(cases),
                    "--dry-run",
                    "--output",
                    str(root / "single-random"),
                ]
            )
            self.assertFalse((root / "single-random/explicit-required").exists())
            with patch.object(runner, "run_case", run_case):
                status = runner.main(
                    [
                        "--count",
                        "3",
                        "--explicit-cases",
                        str(cases),
                        "--stop-on-failure",
                        "--bench",
                        sys.executable,
                        "--rocjitsu",
                        sys.executable,
                        "--config",
                        str(config),
                        "--output",
                        str(root / "stopped"),
                    ]
                )
            self.assertEqual(status, 1)
            stopped = json.loads((root / "stopped/summary.json").read_text())
            self.assertEqual(stopped["sources"]["random"]["started"], 0)
            self.assertFalse(stopped["complete"])

    def test_explicit_cases_reject_duplicates_and_bench_control_overrides(self):
        cases = runner.load_explicit_cases(
            Path(__file__).parent.parent / "hipblaslt_explicit_gemms.json"
        )
        self.assertEqual(len({case["id"] for case in cases}), len(cases))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.json"
            for data in (
                [cases[0], cases[0]],
                [{**cases[0], "id": "../escape"}],
                [
                    {
                        **cases[0],
                        "problem": {**cases[0]["problem"], "norm_check_assert": False},
                    }
                ],
                [{**cases[0], "problem": {**cases[0]["problem"], "M": 0}}],
            ):
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    runner.load_explicit_cases(path)

    def test_dry_run_and_single_case_reproduce_the_same_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner.main(["--dry-run", "--count", "100", "--output", str(root / "all")])
            runner.main(
                [
                    "--dry-run",
                    "--count",
                    "100",
                    "--case",
                    "42",
                    "--output",
                    str(root / "one"),
                ]
            )
            self.assertEqual(
                (root / "all/case-0042/input.yaml").read_text(),
                (root / "one/case-0042/input.yaml").read_text(),
            )
            summary = json.loads((root / "all/summary.json").read_text())
            self.assertEqual(summary["sampled"], 100)
            self.assertEqual(summary["passed"], 0)
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["architecture"], "gfx950")
            for counts in summary["coverage"].values():
                self.assertEqual(sum(counts.values()), 100)

    def test_sampler_is_seeded_aligned_and_preserves_legal_profiles(self):
        def inputs(count, seed):
            rng = random.Random(seed)
            return [sampler.sample_gemm(rng) for _ in range(count)]

        self.assertEqual(inputs(3, 0), inputs(7, 0)[:3])
        self.assertNotEqual(inputs(3, 0), inputs(3, 42))
        problems = inputs(1000, 0)
        self.assertEqual(
            {p["family"] for p in problems}, set(sampler.NUMERICAL_FAMILIES)
        )
        self.assertEqual(
            {p["transA"] + p["transB"] for p in problems}, {"NN", "NT", "TN", "TT"}
        )
        self.assertEqual(
            {(p["bias_vector"], p["activation_type"]) for p in problems},
            set(sampler.EPILOGUES),
        )
        self.assertTrue(any(p["M"] != p["N"] != p["K"] for p in problems))
        for problem in problems:
            for key in ("M", "N", "K"):
                self.assertIn(problem[key], range(32, 500, 32))
            self.assertEqual(problem["a_type"], problem["b_type"])
            self.assertEqual(problem["c_type"], problem["d_type"])
            self.assertEqual(problem["compute_type"], "c_f32_r")
            self.assertNotEqual(problem["alpha"], 0)
            if problem["family"].startswith("mx"):
                self.assertEqual((problem["transA"], problem["transB"]), ("T", "N"))
                self.assertEqual((problem["scaleA"], problem["scaleB"]), (1001, 1001))
            else:
                self.assertEqual((problem["scaleA"], problem["scaleB"]), (0, 0))
            self.assertNotIn("family", runner.bench_input(problem))

    def test_missing_wrong_failed_or_nonfinite_results_cannot_pass(self):
        problem = sampler.sample_gemm(random.Random(0))
        text = bench_result(problem)
        race = '[rocjitsu] Kernel dispatch: "gemm_kernel" symbol="gemm_kernel"\n'
        self.assertIsNone(runner.check_result(problem, text, race))
        for key, value in (
            ("a_type", "f16_r"),
            ("compute_type", "c_f16_r"),
            ("transA", "T" if problem["transA"] == "N" else "N"),
            ("alpha", 3),
            ("beta", 3),
            ("scaleA", 1001),
            ("scaleB", 1001),
            ("bias_vector", not problem["bias_vector"]),
            (
                "activation_type",
                "none" if problem["activation_type"] != "none" else "relu",
            ),
        ):
            with self.subTest(key=key):
                self.assertIsNotNone(
                    runner.check_result(
                        problem, bench_result({**problem, key: value}), race
                    )
                )
        biased = {**problem, "bias_vector": True}
        self.assertIsNotNone(
            runner.check_result(
                biased, bench_result({**biased, "bias_type": "bf16_r"}), race
            )
        )
        self.assertEqual(
            runner.check_result(
                problem, "error: NO solution found!", race + "RACE detected\n"
            ),
            "race reported",
        )
        for bad_text, bad_race in (
            ("", race),
            (text + text, race),
            (
                text.replace(
                    f"{problem['M']},{problem['N']},{problem['K']},1", "1,1,1,1"
                ),
                race,
            ),
            (text.replace(",norm_error", ",missing_norm_error"), race),
            (bench_result(problem, norm="nan"), race),
            (bench_result(problem, atol="failed"), race),
            (text, ""),
            (text, 'symbol="fill_kernel"'),
            (text, race + "RACE detected\n"),
            (text + "\n[rj warn] unsupported instruction\n", race),
        ):
            with self.subTest(text=bad_text, race=bad_race):
                self.assertIsNotNone(runner.check_result(problem, bad_text, bad_race))

    def test_full_loop_preserves_failures_and_later_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench = root / "hipblaslt-bench"
            bench.write_text("#!/bin/sh\nexit 0\n")
            bench.chmod(0o755)
            (root / "hipblaslt_gentest.py").write_text(
                "import shutil, sys\nshutil.copyfile(sys.argv[-1], sys.argv[-2])\n"
            )
            emulator = root / "rocjitsu"
            emulator.write_text(
                f"#!{sys.executable}\n"
                "import json, sys, time\nfrom pathlib import Path\n"
                "p = {k: json.loads(v) for k, v in (line.lstrip('- ').split(': ', 1) "
                "for line in Path(sys.argv[-1]).read_text().splitlines())}\n"
                "config = json.loads(Path(sys.argv[2]).read_text())\n"
                "d = Path(config['sinks']['dir'])\n"
                "started = time.monotonic()\ntime.sleep(0.1)\n"
                "(d / 'interval.json').write_text(json.dumps([started, time.monotonic()]))\n"
                "(d / 'race.log').write_text('[rocjitsu] Kernel dispatch: symbol=\"gemm_kernel\"\\n' "
                "+ ('RACE found\\n' if d.name == 'case-0002' and config.get('failure', 'race') == 'race' else ''))\n"
                f"sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
                "from test_random_hipblaslt_gemms import bench_result\n"
                "if d.name == 'case-0002':\n"
                "    mode = config.get('failure', 'race')\n"
                "    if mode == 'numerical-exit': sys.exit(1)\n"
                "    if mode == 'emulator-exit': sys.exit(7)\n"
                "    if mode == 'no-solution':\n"
                "        print('error: NO solution found!'); sys.exit(0)\n"
                "    if mode == 'timeout': time.sleep(60)\n"
                "    if mode == 'allclose':\n"
                "        print(bench_result(p, atol='failed')); sys.exit(0)\n"
                "print(bench_result(p))\n"
            )
            emulator.chmod(0o755)
            config = root / "config.json"
            config.write_text(
                json.dumps({"vm": {"gpu": {"device": {"gfx_target_version": 90500}}}})
            )
            output = root / "results"
            status = runner.main(
                [
                    "--count",
                    "3",
                    "--bench",
                    str(bench),
                    "--rocjitsu",
                    str(emulator),
                    "--config",
                    str(config),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(status, 1)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(
                (
                    summary["sampled"],
                    summary["passed"],
                    summary["failed"],
                    summary["known_failures"],
                ),
                (3, 2, 1, 0),
            )
            self.assertIn(
                "race reported", (output / "case-0002/result.json").read_text()
            )
            self.assertIn("passed", (output / "case-0003/result.json").read_text())

            for threads, early_stop in ((3, False), (2, True)):
                with self.subTest(threads=threads, early_stop=early_stop):
                    output = root / f"threads-{threads}"
                    command = [
                        "--count",
                        "8",
                        "--threads",
                        str(threads),
                        "--bench",
                        str(bench),
                        "--rocjitsu",
                        str(emulator),
                        "--config",
                        str(config),
                        "--output",
                        str(output),
                    ]
                    if early_stop:
                        command.append("--stop-on-failure")
                    self.assertEqual(runner.main(command), 1)
                    summary = json.loads((output / "summary.json").read_text())
                    self.assertEqual(summary["failed"], 1)
                    self.assertEqual(
                        summary["sampled"], summary["passed"] + summary["failed"]
                    )
                    self.assertEqual(summary["complete"], not early_stop)
                    if early_stop:
                        self.assertLess(summary["sampled"], 8)
                    else:
                        self.assertEqual(summary["sampled"], 8)
                    intervals = [
                        json.loads(p.read_text())
                        for p in output.glob("case-*/interval.json")
                    ]
                    self.assertTrue(
                        any(
                            max(a[0], b[0]) < min(a[1], b[1])
                            for i, a in enumerate(intervals)
                            for b in intervals[i + 1 :]
                        )
                    )

            for mode, reason, exit_code in (
                ("numerical-exit", "bench exited 1", 1),
                ("allclose", "missing or failed numerical checks", 0),
                ("emulator-exit", "bench exited 7", 7),
                ("no-solution", "no heuristic solution", 0),
                ("timeout", "bench exited 124", 124),
            ):
                for early_stop in (False, True):
                    with self.subTest(mode=mode, early_stop=early_stop):
                        config.write_text(
                            json.dumps(
                                {
                                    "failure": mode,
                                    "vm": {
                                        "gpu": {"device": {"gfx_target_version": 90500}}
                                    },
                                }
                            )
                        )
                        output = root / f"{mode}-{early_stop}"
                        command = [
                            "--count",
                            "3",
                            "--timeout",
                            "1",
                            "--bench",
                            str(bench),
                            "--rocjitsu",
                            str(emulator),
                            "--config",
                            str(config),
                            "--output",
                            str(output),
                        ]
                        if early_stop:
                            command.append("--stop-on-failure")
                        self.assertEqual(runner.main(command), 1)
                        summary = json.loads((output / "summary.json").read_text())
                        unsupported = int(mode == "no-solution")
                        self.assertEqual(summary["failed"], 1 - unsupported)
                        self.assertEqual(summary["unsupported"], unsupported)
                        self.assertEqual(
                            summary["random_no_solution_rate"],
                            unsupported / (2 if early_stop else 3),
                        )
                        stopped = early_stop
                        self.assertEqual(summary["passed"], 1 if stopped else 2)
                        self.assertEqual(summary["sampled"], 2 if stopped else 3)
                        self.assertEqual(summary["complete"], not stopped)
                        self.assertEqual(summary["known_failures"], 0)
                        result = json.loads(
                            (output / "case-0002/result.json").read_text()
                        )
                        self.assertEqual(result["reason"], reason)
                        self.assertEqual(result["exit_code"], exit_code)
                        self.assertEqual(
                            result["outcome"],
                            "unsupported" if unsupported else "failed",
                        )
                        if stopped:
                            self.assertFalse((output / "case-0003").exists())
                        else:
                            result = json.loads(
                                (output / "case-0003/result.json").read_text()
                            )
                            self.assertEqual(result["outcome"], "passed")

    def test_threads_do_not_change_generated_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for threads in (1, 4):
                runner.main(
                    [
                        "--dry-run",
                        "--count",
                        "8",
                        "--threads",
                        str(threads),
                        "--output",
                        str(root / str(threads)),
                    ]
                )
            for iteration in range(1, 9):
                name = f"case-{iteration:04d}/input.yaml"
                self.assertEqual(
                    (root / "1" / name).read_text(), (root / "4" / name).read_text()
                )

    def test_timeout_terminates_case_and_returns_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                runner.run_command(
                    [sys.executable, "-c", "import time; time.sleep(60)"],
                    Path(tmp),
                    0.1,
                    "timeout.log",
                ),
                124,
            )

    def test_execution_requires_gfx950(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config.json"
            config.write_text(
                json.dumps({"vm": {"gpu": {"device": {"gfx_target_version": 90402}}}})
            )
            with self.assertRaises(SystemExit) as error:
                runner.main(
                    ["--config", str(config), "--output", str(root / "results")]
                )
            self.assertEqual(error.exception.code, 2)
            self.assertFalse((root / "results").exists())


if __name__ == "__main__":
    unittest.main()
