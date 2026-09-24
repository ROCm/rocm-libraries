# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Standalone WaveScope capture contracts, with an executable GPU-free profiler."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


PLATFORM = Path(__file__).resolve().parents[5]
UTILITY = (
    PLATFORM
    / "dsl_docs/optimization/utilities/tools/wavescope/capture_wavescope_pmc.py"
)
RAW = (
    b"Kernel_Name,Dispatch_Id,Counter_Name,Counter_Value,Start_Timestamp,End_Timestamp\r\n"
    b"gemm,0,GRBM_GUI_ACTIVE,900,0,1000\r\n"
    b"gemm,1,GRBM_GUI_ACTIVE,100,1000,2000\r\n"
    b"other,2,GRBM_GUI_ACTIVE,700,2000,3000\r\n"
)


@unittest.skipUnless(os.name == "posix", "executable profiler fixture requires POSIX")
class TestWaveScopePmc(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.cwd = self.root / "launch directory"
        self.cwd.mkdir()
        self.trace = self.root / "trace"
        self.trace.mkdir()
        for name in ("code.json", "filenames.json", "occupancy.json"):
            (self.trace / name).write_text("{}\n")
        self.cache = self.root / "history"
        self.launch_log = self.root / "launches.jsonl"
        self.profiler_log = self.root / "profiler.jsonl"
        self.raw = self.root / "original.csv"
        self.raw.write_bytes(RAW)
        imports = self.root / "launcher imports"
        imports.mkdir()
        (imports / "capture_dependency.py").write_text("VALUE = 'import preserved'\n")
        (self.cwd / "launcher.py").write_text(
            textwrap.dedent(
                """\
            import json
            import os
            import sys
            from pathlib import Path
            import capture_dependency
            from rocke.benchmark.perf import schema

            entry = {
                "cwd": os.getcwd(),
                "argv": sys.argv[1:],
                "marker": os.environ["CAPTURE_MARKER"],
                "pythonpath": os.environ["PYTHONPATH"],
                "dependency": capture_dependency.VALUE,
                "schema": schema.__file__,
            }
            with open(os.environ["CAPTURE_LAUNCH_LOG"], "a") as stream:
                stream.write(json.dumps(entry) + "\\n")
            print("launcher diagnostic, not JSON")
            print('PerfJSON: {"ms": 1}')
            """
            )
        )
        bindir = self.root / "bin"
        bindir.mkdir()
        profiler = bindir / "rocprofv3"
        profiler.write_text(
            f"#!{sys.executable}\n"
            + textwrap.dedent(
                """\
            import json
            import os
            import subprocess
            import sys
            from pathlib import Path

            args = sys.argv[1:]
            with open(os.environ["CAPTURE_PROFILER_LOG"], "a") as stream:
                stream.write(json.dumps(args) + "\\n")
            if args == ["--list-avail"]:
                print("Counter_Name : GRBM_GUI_ACTIVE")
                raise SystemExit(0)
            result = subprocess.run(args[args.index("--") + 1:])
            if result.returncode:
                raise SystemExit(result.returncode)
            output = Path(args[args.index("-d") + 1])
            csv = output / "pmc_1/results_counter_collection.csv"
            csv.parent.mkdir(parents=True)
            csv.write_bytes(Path(os.environ["CAPTURE_RAW"]).read_bytes())
            if os.environ.get("CAPTURE_SECOND_PASS"):
                csv2 = output / "pmc_2/results_counter_collection.csv"
                csv2.parent.mkdir(parents=True)
                csv2.write_bytes(Path(os.environ["CAPTURE_RAW"]).read_bytes())
            sample = next(part for part in output.parts if part.isdigit())
            if sample in os.environ.get("CAPTURE_FAIL_SAMPLES", "").split(","):
                raise SystemExit(7)
            if os.environ.get("CAPTURE_NO_CSV"):
                csv.unlink()
            """
            )
        )
        profiler.chmod(0o755)
        self.env = {
            **os.environ,
            "PATH": str(bindir) + os.pathsep + os.environ.get("PATH", ""),
            "PYTHONPATH": str(imports),
            "PYTHONNOUSERSITE": "1",
            "CAPTURE_MARKER": "environment preserved",
            "CAPTURE_LAUNCH_LOG": str(self.launch_log),
            "CAPTURE_PROFILER_LOG": str(self.profiler_log),
            "CAPTURE_RAW": str(self.raw),
            "HOME": str(self.root),
            "XDG_CACHE_HOME": str(self.root / "xdg-cache"),
        }
        self.command = [
            sys.executable,
            "launcher.py",
            "--output-dir",
            "launcher-owned",
            "--store-history",
            "argument with spaces",
        ]

    def invoke(self, args):
        return subprocess.run(
            [sys.executable, str(UTILITY), *args],
            cwd=self.cwd,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def capture(self, destination, *options):
        return self.invoke(
            [
                "--output-dir",
                str(destination),
                "--arch",
                "gfx950",
                "--op",
                "gemm",
                "--kernel-name",
                "named-gemm",
                "--match-kernel",
                "gemm",
                "--shape",
                '{"M":1}',
                "--warmup",
                "1",
                "--cache",
                str(self.cache),
                *options,
                "--",
                *self.command,
            ]
        )

    def read_json(self, path):
        return json.loads(path.read_text())

    def test_repeats_export_both_formats_and_preserve_launcher_context(self):
        result = self.capture("bundle", "--repeats", "2", "--per-dispatch", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        stdout = json.loads(result.stdout)
        self.assertIn("WaveScope PMC bundle:", result.stderr)
        self.assertNotIn("launcher diagnostic", result.stdout)
        self.assertFalse((self.cache / "history.jsonl").exists())
        self.assertFalse((self.cwd / "launcher-owned").exists())
        launches = [
            json.loads(line) for line in self.launch_log.read_text().splitlines()
        ]
        self.assertEqual(len(launches), 4)
        for launch in launches:
            self.assertEqual(launch["cwd"], str(self.cwd))
            self.assertEqual(launch["argv"], self.command[2:])
            self.assertEqual(launch["marker"], self.env["CAPTURE_MARKER"])
            self.assertEqual(launch["dependency"], "import preserved")
            self.assertEqual(
                launch["pythonpath"],
                str(PLATFORM / "python") + os.pathsep + self.env["PYTHONPATH"],
            )
            self.assertEqual(
                Path(launch["schema"]),
                PLATFORM / "python/rocke/benchmark/perf/schema.py",
            )
        destination = self.root / "relocated"
        shutil.move(self.cwd / "bundle", destination)
        manifest = self.read_json(destination / "manifest.json")
        record = self.read_json(destination / manifest["measurement"])
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["semantics"]["att_association"], "unbound")
        self.assertEqual(record, stdout["record"])
        self.assertEqual(record["n_samples"], 2)
        self.assertEqual(record["kernel"]["kernel_name"], "named-gemm")
        self.assertEqual(record["kernel"]["shape"], {"M": 1})
        self.assertEqual(record["counters"]["busy_cycles"], 100)
        self.assertEqual({s["sample_index"] for s in record["counter_samples"]}, {0, 1})
        self.assertEqual(
            self.read_json(destination / "comparison.json"), stdout["selfcheck"]
        )
        csvs = [item for item in manifest["files"] if item["kind"] == "pmc_csv"]
        self.assertEqual(len(csvs), 2)
        self.assertEqual(len(manifest["samples"]), 2)
        for item in manifest["files"]:
            self.assertFalse(Path(item["path"]).is_absolute())
            data = (destination / item["path"]).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), item["sha256"])
        for item in csvs:
            self.assertEqual((destination / item["path"]).read_bytes(), RAW)
        for sample in manifest["samples"]:
            measurement = self.read_json(destination / sample["measurement"])
            self.assertEqual(sample["run_id"], measurement["run"]["run_id"])
            self.assertEqual(measurement["profile_capture"]["warmup_per_pass"], 1)
            self.assertEqual(measurement["profile_capture"]["status"], "complete")

    def test_human_stdout_still_exports_json_and_csv(self):
        destination = self.root / "human"
        result = self.capture(destination)
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = self.read_json(destination / "manifest.json")
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(
            self.read_json(destination / manifest["measurement"])["counters"][
                "busy_cycles"
            ],
            100,
        )
        csvs = [item for item in manifest["files"] if item["kind"] == "pmc_csv"]
        self.assertEqual(len(csvs), 1)
        self.assertEqual((destination / csvs[0]["path"]).read_bytes(), RAW)
        self.assertFalse((self.cache / "history.jsonl").exists())

    def test_trace_dir_publishes_one_repeat_all_replay_passes(self):
        self.env["CAPTURE_SECOND_PASS"] = "1"
        destination = self.root / "pmc-bundle"
        result = self.capture(
            destination,
            "--trace-dir",
            str(self.trace),
            "--repeats",
            "2",
            "--json",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        sidecars = sorted(self.trace.glob("rocke_pmc_*_counter_collection.csv"))
        self.assertEqual(
            [path.name for path in sidecars],
            [
                "rocke_pmc_1_counter_collection.csv",
                "rocke_pmc_2_counter_collection.csv",
            ],
        )
        self.assertTrue(all(path.read_bytes() == RAW for path in sidecars))
        manifest = self.read_json(destination / "manifest.json")
        self.assertEqual(
            len([item for item in manifest["files"] if item["kind"] == "pmc_csv"]),
            4,
        )
        self.assertIn(f"WaveScope-ready trace folder: {self.trace}", result.stderr)
        self.assertNotIn("then upload these CSVs", result.stderr)

    def test_trace_dir_refuses_existing_sidecar_without_overwrite(self):
        sidecar = self.trace / "rocke_pmc_1_counter_collection.csv"
        sidecar.write_bytes(b"keep me")
        result = self.capture(
            self.root / "collision-bundle",
            "--trace-dir",
            str(self.trace),
            "--json",
        )
        self.assertEqual(result.returncode, 2)
        self.assertFalse((self.root / "collision-bundle").exists())
        self.assertEqual(sidecar.read_bytes(), b"keep me")
        self.assertIn("already contains PMC sidecars", result.stderr)

    def test_trace_dir_refuses_nested_bundle(self):
        destination = self.trace / "pmc_bundle"
        result = self.capture(
            destination,
            "--trace-dir",
            str(self.trace),
        )
        self.assertEqual(result.returncode, 2)
        self.assertFalse(destination.exists())
        self.assertIn("must be separate sibling trees", result.stderr)

    def test_trace_dir_requires_wavescope_dispatch_files(self):
        not_a_trace = self.root / "not-a-trace"
        not_a_trace.mkdir()
        destination = self.root / "invalid-trace-bundle"
        result = self.capture(
            destination,
            "--trace-dir",
            str(not_a_trace),
        )
        self.assertEqual(result.returncode, 2)
        self.assertFalse(destination.exists())
        self.assertIn("not a WaveScope dispatch folder", result.stderr)

    def test_failed_profiler_csv_is_retained_without_upload_guidance(self):
        self.env["CAPTURE_FAIL_SAMPLES"] = "0000"
        destination = self.root / "failed-profiler"
        result = self.capture(destination, "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = self.read_json(destination / "manifest.json")
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["samples"][0]["profile_capture"]["status"], "failed")
        csvs = [item for item in manifest["files"] if item["kind"] == "pmc_csv"]
        self.assertEqual(len(csvs), 1)
        self.assertEqual((destination / csvs[0]["path"]).read_bytes(), RAW)
        self.assertNotIn(str(destination / csvs[0]["path"]), result.stderr)
        self.assertNotIn("upload a CSV", result.stderr)
        self.assertIn("diagnosis", result.stderr)
        self.assertEqual(json.loads(result.stdout)["record"]["wall"]["ms_median"], 1)

    def test_mixed_repeats_recommend_only_successful_profiler_csv(self):
        self.env["CAPTURE_FAIL_SAMPLES"] = "0000"
        destination = self.root / "mixed-profiler"
        result = self.capture(destination, "--repeats", "2", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = self.read_json(destination / "manifest.json")
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(
            [sample["profile_capture"]["status"] for sample in manifest["samples"]],
            ["failed", "complete"],
        )
        csvs = [item for item in manifest["files"] if item["kind"] == "pmc_csv"]
        self.assertEqual(len(csvs), 2)
        for item in csvs:
            self.assertEqual((destination / item["path"]).read_bytes(), RAW)
            csv_path = str(destination / item["path"])
            if Path(manifest["samples"][0]["raw_dir"]) in Path(item["path"]).parents:
                self.assertNotIn(csv_path, result.stderr)
            else:
                self.assertIn(csv_path, result.stderr)
        self.assertIn("upload these CSVs", result.stderr)

    def test_wall_only_export_has_no_csv_upload_guidance(self):
        self.env["CAPTURE_NO_CSV"] = "1"
        destination = self.root / "wall-only"
        result = self.capture(destination, "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = self.read_json(destination / "manifest.json")
        self.assertEqual(manifest["status"], "complete")
        self.assertFalse(any(item["kind"] == "pmc_csv" for item in manifest["files"]))
        self.assertEqual(json.loads(result.stdout)["record"]["wall"]["ms_median"], 1)
        self.assertNotIn("upload a CSV", result.stderr)

    def test_opt_in_history_preserves_regression_exit_and_complete_bundle(self):
        baseline = self.capture(self.root / "baseline", "--store-history", "--json")
        self.assertEqual(baseline.returncode, 0, baseline.stderr)
        self.assertEqual(
            json.loads(baseline.stdout)["selfcheck"]["verdict"], "no_baseline"
        )
        history = self.cache / "history.jsonl"
        self.assertEqual(len(history.read_text().splitlines()), 1)
        self.raw.write_bytes(RAW.replace(b",100,1000,2000", b",200,1000,2000"))
        destination = self.root / "candidate"
        result = self.capture(
            destination,
            "--store-history",
            "--threshold",
            "0.1",
            "--noise-k",
            "0",
            "--json",
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        stdout = json.loads(result.stdout)
        self.assertEqual(stdout["selfcheck"]["verdict"], "regressed")
        self.assertEqual(len(history.read_text().splitlines()), 2)
        self.assertEqual(
            self.read_json(destination / "manifest.json")["status"], "complete"
        )
        self.assertEqual(
            self.read_json(destination / "comparison.json"), stdout["selfcheck"]
        )
        before = history.read_bytes()
        result = self.capture(self.root / "not-stored", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(history.read_bytes(), before)

    def test_existing_destinations_are_preserved_without_launch(self):
        directory = self.root / "existing"
        directory.mkdir()
        marker = directory / "mine"
        marker.write_bytes(b"untouched")
        file = self.root / "file"
        file.write_bytes(b"original")
        dangling = self.root / "dangling"
        dangling.symlink_to(self.root / "missing")
        for destination in (directory, file, dangling):
            with self.subTest(destination=destination.name):
                result = self.capture(destination)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("already exists", result.stderr)
                self.assertFalse(self.launch_log.exists())
                self.assertFalse(self.profiler_log.exists())
        self.assertEqual(list(directory.iterdir()), [marker])
        self.assertEqual(marker.read_bytes(), b"untouched")
        self.assertEqual(file.read_bytes(), b"original")
        self.assertTrue(dangling.is_symlink())
        self.assertFalse((self.root / "missing").exists())

    def test_invalid_arguments_are_rejected_without_launch(self):
        destination = self.root / "invalid"
        for options in (
            ["--repeats", "0"],
            ["--warmup", "-1"],
            ["--shape", "[]"],
            ["--unknown-option"],
            ["--no-store"],
            ["--artifacts-dir", str(self.root / "other")],
            ["--artifacts-dir=" + str(self.root / "other")],
        ):
            with self.subTest(options=options):
                result = self.capture(destination, *options)
                self.assertNotEqual(result.returncode, 0)
                self.assertRegex(result.stderr, r"error:|--shape must be a JSON object")
                self.assertFalse(destination.exists())
                self.assertFalse(self.launch_log.exists())
                self.assertFalse(self.profiler_log.exists())

    def test_missing_destination_or_launch_is_rejected(self):
        destination = self.root / "missing-launch"
        for args in (
            ["--arch", "gfx950", "--", *self.command],
            ["--output-dir", str(destination), "--arch", "gfx950"],
            ["--output-dir", str(destination), "--arch", "gfx950", "--"],
        ):
            with self.subTest(args=args):
                result = self.invoke(args)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("error:", result.stderr)
                self.assertFalse(destination.exists())
                self.assertFalse(self.launch_log.exists())
                self.assertFalse(self.profiler_log.exists())


if __name__ == "__main__":
    unittest.main()
