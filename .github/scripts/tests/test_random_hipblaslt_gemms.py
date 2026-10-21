# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import run_random_hipblaslt_gemms as runner


def bench_result(problem, *, norm="0", atol="1e-6"):
    return (
        "[0]:m,n,k,batch_count,a_type,b_type,c_type,d_type,transA,transB,norm_error,atol,rtol\n"
        f"{problem['M']},{problem['N']},{problem['K']},1,"
        f"{problem['a_type']},{problem['b_type']},{problem['c_type']},{problem['d_type']},"
        f"N,N,{norm},{atol},1e-6\n"
        "    --kernel name: gemm_kernel\n"
    )


class RandomGemmTests(unittest.TestCase):
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

    def test_sampler_prefix_and_type_cycle(self):
        def inputs(count, seed):
            rng = random.Random(seed)
            return [runner.sample_gemm(i, rng) for i in range(1, count + 1)]

        self.assertEqual(inputs(3, 0), inputs(7, 0)[:3])
        self.assertEqual(inputs(3, 0), inputs(3, 42))
        self.assertEqual(
            [(p["M"], p["N"], p["K"], p["a_type"]) for p in inputs(3, 0)],
            [(11, 11, 11, "f32_r"), (12, 12, 12, "bf16_r"), (13, 13, 13, "f32_r")],
        )

    def test_missing_wrong_failed_or_nonfinite_results_cannot_pass(self):
        problem = runner.sample_gemm(1, random.Random(0))
        text = bench_result(problem)
        race = '[rocjitsu] Kernel dispatch: "gemm_kernel" symbol="gemm_kernel"\n'
        self.assertIsNone(runner.check_result(problem, text, race))
        for bad_text, bad_race in (
            ("", race),
            (text + text, race),
            (text.replace("11,11,11,1", "12,11,11,1"), race),
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
                "+ ('RACE found\\n' if p['M'] == 12 and config.get('failure', 'race') == 'race' else ''))\n"
                f"sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
                "from test_random_hipblaslt_gemms import bench_result\n"
                "if p['M'] == 12:\n"
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
            config.write_text("{}")
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
                ("no-solution", "expected one completed bench result", 0),
                ("timeout", "bench exited 124", 124),
            ):
                for early_stop in (False, True):
                    with self.subTest(mode=mode, early_stop=early_stop):
                        config.write_text(json.dumps({"failure": mode}))
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
                        self.assertEqual(summary["failed"], 1)
                        self.assertEqual(summary["passed"], 1 if early_stop else 2)
                        self.assertEqual(summary["sampled"], 2 if early_stop else 3)
                        self.assertEqual(summary["complete"], not early_stop)
                        self.assertEqual(summary["known_failures"], 0)
                        result = json.loads(
                            (output / "case-0002/result.json").read_text()
                        )
                        self.assertEqual(result["reason"], reason)
                        self.assertEqual(result["exit_code"], exit_code)
                        self.assertEqual(result["outcome"], "failed")
                        if early_stop:
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


if __name__ == "__main__":
    unittest.main()
