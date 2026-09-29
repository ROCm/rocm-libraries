#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import copy
import csv
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import zlib

import msgpack
import rocjitsu_race_sweep as sweep


def solution(index):
    return {
        "index": index,
        "name": f"solution-{index}",
        "kernelName": f"kernel-{index}",
        "problemType": {
            "aType": "Float",
            "bType": "Float",
            "cType": "Float",
            "dType": "Float",
        },
        "hardwarePredicate": {"type": "Processor", "value": "gfx942"},
    }


def job():
    return {
        "id": 0,
        "solutions": [{"index": 7, "name": "solution-7", "kernel": "kernel-7"}],
    }


def success_log():
    stream = io.StringIO()
    for n, shape in enumerate(sweep.SHAPES):
        stream.write('[rocjitsu] Kernel dispatch: "kernel-7" symbol="kernel-7"\n')
        csv.writer(stream, lineterminator="\n").writerow(
            [
                0,
                f"{n}/3",
                "7/7",
                "Contraction",
                "(" + ",".join(map(str, shape)) + ")",
                "None",
                "",
                "None",
                "solution-7",
                "PASSED",
            ]
        )
    return stream.getvalue()


class SweepTests(unittest.TestCase):
    def test_native_and_compressed_messagepack(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, data in [
                (
                    "lib.dat",
                    msgpack.packb({"solutions": [solution(7)]}, use_bin_type=True),
                ),
                (
                    "lib.dat.zlib",
                    zlib.compress(
                        msgpack.packb({"solutions": [solution(7)]}, use_bin_type=True)
                    ),
                ),
            ]:
                path = Path(tmp) / name
                path.write_bytes(data)
                self.assertEqual(sweep.read_library(path)["solutions"][0]["index"], 7)

    def test_hardware_constraints_are_not_bypassed(self):
        device = {"device_id": 0x74A1, "simd_count": 1216, "simd_per_cu": 4}
        predicate = {
            "type": "AMDGPU",
            "value": {
                "type": "And",
                "value": [
                    {"type": "Processor", "value": "gfx942"},
                    {"type": "CUCount", "value": 304},
                ],
            },
        }
        self.assertTrue(sweep.matches_hardware(predicate, "gfx942", device))
        self.assertFalse(sweep.matches_hardware(predicate, "gfx950", device))
        self.assertFalse(sweep.matches_hardware({"type": "Unknown"}, "gfx942", device))
        device["simd_count"] = 256 * 4
        self.assertFalse(sweep.matches_hardware(predicate, "gfx942", device))

    def test_batches_are_contiguous_and_distinct(self):
        groups = {("a.dat", "type", "hw"): [solution(i) for i in range(30)]}
        result = sweep.select_batches(groups, 20, 10)
        self.assertEqual(
            [[s["index"] for s in j["solutions"]] for j in result],
            [list(range(10)), list(range(10, 20))],
        )
        aliased = copy.deepcopy(groups)
        aliased[("a.dat", "type", "hw")][10]["kernelName"] = "kernel-0"
        result = sweep.select_batches(aliased, 20, 10)
        self.assertEqual(len({s["kernel"] for j in result for s in j["solutions"]}), 20)
        self.assertEqual(result[1]["solutions"][0]["index"], 11)

    def test_missing_inventory_is_not_silently_a_smaller_sweep(self):
        with self.assertRaisesRegex(ValueError, "selected only"):
            sweep.select_batches(
                {("a.dat", "type", "hw"): [solution(i) for i in range(9)]}, 100, 10
            )

    def test_numerical_and_dispatch_evidence_required(self):
        text = success_log()
        self.assertFalse(sweep.classify(job(), text, 0)["failed"])
        self.assertTrue(
            sweep.classify(job(), text.replace(",PASSED", ",NO_CHECK", 1), 0)["failed"]
        )
        self.assertTrue(
            sweep.classify(
                job(),
                text.replace(
                    '[rocjitsu] Kernel dispatch: "kernel-7" symbol="kernel-7"\n', "", 1
                ),
                0,
            )["failed"]
        )
        self.assertTrue(sweep.classify(job(), text, 1)["failed"])

    def test_missing_duplicate_and_rejected_cases_fail(self):
        text = success_log()
        record = next(line for line in text.splitlines() if ",PASSED" in line)
        for changed in [
            text.replace(record + "\n", "", 1),
            text + record + "\n",
            text.replace(",PASSED", ",DID_NOT_SATISFY_ASSERTS", 1),
        ]:
            self.assertTrue(sweep.classify(job(), changed, 0)["failed"])

    def test_races_and_warnings_remain_visible_and_fail(self):
        for suffix in [
            "RACE kernel=? dispatch=9\nEND_RACE\n",
            "[rj warn] unsupported HW_ID\n",
        ]:
            result = sweep.classify(job(), success_log() + suffix, 0)
            self.assertTrue(result["failed"])
            self.assertTrue(result["race_headers"] or result["warnings"])

    def test_four_workers_and_no_duplicate_assignment(self):
        gate = threading.Barrier(4)

        def execute(task, slot):
            if task["id"] < 4:
                gate.wait(timeout=2)
            time.sleep(0.01)
            return {"id": task["id"], "failed": False, "cases": []}

        result = sweep.run_queue(
            [{"id": i} for i in range(10)], 4, execute, lambda _: None
        )
        self.assertEqual(result["max_active"], 4)
        self.assertEqual([r["id"] for r in result["results"]], list(range(10)))

    def test_failure_stops_new_jobs_but_finishes_inflight(self):
        gate = threading.Barrier(4)

        def execute(task, slot):
            gate.wait(timeout=2)
            if task["id"]:
                time.sleep(0.05)
            return {"id": task["id"], "failed": task["id"] == 0, "cases": []}

        result = sweep.run_queue(
            [{"id": i} for i in range(10)], 4, execute, lambda _: None
        )
        self.assertEqual(result["unstarted"], list(range(4, 10)))
        self.assertEqual(result["stop"]["in_flight"], 3)
        self.assertEqual(len(result["results"]), 4)

    def test_worker_exception_is_a_failure(self):
        def execute(task, slot):
            raise ValueError("malformed output")

        result = sweep.run_queue(
            [{"id": i} for i in range(2)], 1, execute, lambda _: None
        )
        self.assertEqual(result["unstarted"], [1])
        self.assertTrue(result["results"][0]["failed"])

    def test_timeout_kills_child_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            childfile = Path(tmp) / "child"
            code = (
                'import subprocess,time,pathlib; p=subprocess.Popen(["sleep","60"]); pathlib.Path('
                + repr(str(childfile))
                + ").write_text(str(p.pid)); time.sleep(60)"
            )
            result = sweep.execute_command(
                [sys.executable, "-c", code], Path(tmp) / "log", 0.3, dict(os.environ)
            )
            self.assertEqual(result, 124)
            stat = Path("/proc") / childfile.read_text() / "stat"
            try:
                state = stat.read_text().split(") ")[1].split()[0]
            except FileNotFoundError:
                state = "exited"
            self.assertIn(state, {"exited", "Z"})

    def test_progress_failure_stops_queue_and_is_reported(self):
        def progress(value):
            raise OSError("No space left")

        result = sweep.run_queue(
            [{"id": i} for i in range(2)],
            1,
            lambda task, slot: {"id": task["id"], "failed": False, "cases": []},
            progress,
        )
        self.assertEqual(result["unstarted"], [1])
        self.assertEqual(result["stop"]["trigger_job"], 0)
        self.assertTrue(result["results"][0]["failed"])
        self.assertIn("No space left", result["results"][0]["errors"][0])

    def test_driver_runs_all_stages_and_propagates_each_failure(self):
        driver = Path(__file__).with_name("run_rocjitsu_hipblaslt_race_check.sh")
        # Exercise the real post-setup stage sequence with lightweight workloads.
        stages = driver.read_text().split("\ncheck_status=0\n", 1)[1]
        setup = """
set -euo pipefail
check_status=0
ROCJITSU_BIN=emulator
TENSILELITE_CLIENT=client
ROCJITSU_CONFIG=config
ROCJITSU_GPU_TARGET=gfx942
ROCM_PATH=artifact
RACE_REPORT_DIR=reports
run_timed() { echo "STAGE: $1"; shift; "$@"; }
run_hipblaslt_bench_check() { return "$BENCH_STATUS"; }
run_tensilelite_client_check() { return "$CLIENT_STATUS"; }
python3() { printf 'ARG: %s\n' "$@"; return "$SWEEP_STATUS"; }
"""
        for bench, client, sweep_status in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            with self.subTest(statuses=(bench, client, sweep_status)):
                result = subprocess.run(
                    ["bash", "-c", setup + stages],
                    env={
                        **os.environ,
                        "BENCH_STATUS": str(bench),
                        "CLIENT_STATUS": str(client),
                        "SWEEP_STATUS": str(sweep_status),
                    },
                    text=True,
                    capture_output=True,
                    timeout=10,
                )
                self.assertEqual(
                    result.returncode, int(any((bench, client, sweep_status)))
                )
                self.assertEqual(
                    [
                        line
                        for line in result.stdout.splitlines()
                        if line.startswith("STAGE:")
                    ],
                    [
                        "STAGE: hipblaslt-bench race check",
                        "STAGE: tensilelite-client race check",
                        "STAGE: 100-kernel toy race sweep",
                    ],
                )
                for flag, value in (
                    ("--workers", "4"),
                    ("--kernels", "100"),
                    ("--target", "gfx942"),
                ):
                    self.assertIn(f"ARG: {flag}\nARG: {value}\n", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
