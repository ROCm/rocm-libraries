# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Observable artifact contracts, using profiler fixtures without a GPU."""
import hashlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import patch

from rocke.benchmark.perf import harness, schema
from rocke.benchmark.perf.tool import cli

RAW = (
    b"Kernel_Name,Dispatch_Id,Counter_Name,Counter_Value,Start_Timestamp,End_Timestamp\r\n"
    b"gemm,0,GRBM_GUI_ACTIVE,900,0,1000\r\n"
    b"gemm,1,GRBM_GUI_ACTIVE,100,1000,2000\r\n"
)


class TestArtifactExport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cache = self.root / "cache"
        self.launches = 0
        self.discover = patch.object(
            harness._counters,
            "discover",
            return_value={"busy_cycles": "GRBM_GUI_ACTIVE"},
        ).start()
        self.profiler = patch.object(
            harness, "_run_rocprofv3", side_effect=self.capture
        ).start()
        self.addCleanup(patch.stopall)

    def capture(self, cmd, pmc, outdir, env, timeout):
        self.launches += 1
        path = outdir / "pmc_1" / "results_counter_collection.csv"
        path.parent.mkdir(parents=True)
        path.write_bytes(RAW)
        return True, ""

    def run_cli(self, dest, *extra):
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            code = cli.main(
                [
                    "profile",
                    "--arch",
                    "gfx950",
                    "--op",
                    "gemm",
                    "--kernel-name",
                    "gemm",
                    "--shape",
                    '{"M":1}',
                    "--warmup",
                    "1",
                    "--cache",
                    str(self.cache),
                    "--artifacts-dir",
                    str(dest),
                    "--json",
                    *extra,
                    "--",
                    sys.executable,
                    "-c",
                    "print('PerfJSON: {\"ms\": 1}')",
                ]
            )
        return code, json.loads(out.getvalue())

    def read_json(self, path):
        return json.loads(path.read_text())

    def test_repeats_preserve_raw_bytes_and_portable_json_without_history(self):
        dest = self.root / "bundle"
        code, stdout = self.run_cli(
            dest, "--repeats", "2", "--per-dispatch", "--no-store"
        )
        self.assertEqual(code, 0)
        relocated = self.root / "relocated"
        shutil.move(dest, relocated)
        manifest = self.read_json(relocated / "manifest.json")
        record = self.read_json(relocated / manifest["measurement"])
        schema.validate(record)
        self.assertEqual(record, stdout["record"])
        self.assertEqual(record["counters"]["busy_cycles"], 100)
        self.assertEqual(record["n_samples"], 2)
        self.assertNotIn("profile_capture", record)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["semantics"]["att_association"], "unbound")
        self.assertEqual(len(manifest["samples"]), 2)
        self.assertEqual({s["sample_index"] for s in record["counter_samples"]}, {0, 1})
        self.assertFalse((self.cache / "history.jsonl").exists())
        csvs = [f for f in manifest["files"] if f["kind"] == "pmc_csv"]
        self.assertEqual(len(csvs), 2)
        for item in manifest["files"]:
            path = relocated / item["path"]
            self.assertFalse(Path(item["path"]).is_absolute())
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), item["sha256"]
            )
        for item in csvs:
            self.assertEqual((relocated / item["path"]).read_bytes(), RAW)
        for sample in manifest["samples"]:
            one = self.read_json(relocated / sample["measurement"])
            self.assertEqual(sample["run_id"], one["run"]["run_id"])
            self.assertEqual(one["profile_capture"]["warmup_per_pass"], 1)
            self.assertEqual(
                one["profile_capture"]["counter_map"],
                {"busy_cycles": "GRBM_GUI_ACTIVE"},
            )

    def test_existing_destination_refused_before_launch_and_preserved(self):
        dest = self.root / "existing"
        dest.mkdir()
        (dest / "mine").write_text("untouched")
        with self.assertRaises(SystemExit):
            self.run_cli(dest)
        self.assertEqual(self.launches, 0)
        self.assertEqual((dest / "mine").read_text(), "untouched")
        self.assertEqual(list(dest.iterdir()), [dest / "mine"])

    def test_failure_preserves_partial_raw_without_complete_claim(self):
        def fail(*args):
            self.capture(*args)
            raise RuntimeError("capture failed")

        self.profiler.side_effect = fail
        dest = self.root / "failure"
        with self.assertRaises(SystemExit):
            self.run_cli(dest)
        manifest = self.read_json(dest / "manifest.json")
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["samples"][0]["status"], "failed")
        self.assertFalse((dest / "measurement.json").exists())
        self.assertEqual(
            (
                dest / "samples/0000/raw/prof/pmc_1/results_counter_collection.csv"
            ).read_bytes(),
            RAW,
        )

    def test_warmup_failure_preserves_raw_and_no_record(self):
        dest = self.root / "warmup"
        with self.assertRaises(SystemExit):
            self.run_cli(dest, "--warmup", "2")
        self.assertEqual(self.read_json(dest / "manifest.json")["status"], "failed")
        self.assertFalse((dest / "measurement.json").exists())

    def test_wall_only_bundle_reports_unavailable_profiler(self):
        self.discover.return_value = {}
        dest = self.root / "wall"
        code, result = self.run_cli(dest, "--no-store")
        self.assertEqual(code, 0)
        manifest = self.read_json(dest / "manifest.json")
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(
            manifest["samples"][0]["profile_capture"]["status"], "unavailable"
        )
        self.assertEqual(result["record"]["wall"]["ms_median"], 1)
        self.assertFalse(any(f["kind"] == "pmc_csv" for f in manifest["files"]))

    def test_regression_exit_still_publishes_complete_bundle(self):
        self.run_cli(self.root / "baseline")

        def slower(*args):
            self.capture(*args)
            path = args[2] / "pmc_1/results_counter_collection.csv"
            path.write_bytes(RAW.replace(b",100,1000,2000", b",200,1000,2000"))
            return True, ""

        self.profiler.side_effect = slower
        dest = self.root / "candidate"
        code, result = self.run_cli(dest)
        self.assertEqual(code, 1)
        self.assertEqual(result["selfcheck"]["verdict"], "regressed")
        manifest = self.read_json(dest / "manifest.json")
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(self.read_json(dest / "comparison.json"), result["selfcheck"])

    def test_second_repeat_failure_preserves_first_record(self):
        def fail_second(*args):
            self.capture(*args)
            if self.launches == 2:
                raise RuntimeError("second capture failed")
            return True, ""

        self.profiler.side_effect = fail_second
        dest = self.root / "partial"
        with self.assertRaises(SystemExit):
            self.run_cli(dest, "--repeats", "2")
        manifest = self.read_json(dest / "manifest.json")
        self.assertEqual(
            [s["status"] for s in manifest["samples"]], ["complete", "failed"]
        )
        schema.validate(self.read_json(dest / "samples/0000/measurement.json"))
        self.assertFalse((dest / "measurement.json").exists())
