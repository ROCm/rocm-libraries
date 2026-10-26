# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import copy
import csv
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rocjitsu_compare import (
    audit,
    audit_item,
    compare,
    resource_statistics,
    solution_digest,
    workload_coverage,
)
from rocjitsu_evidence import digest, retain
import rocjitsu_experiment
from rocjitsu_experiment import commands
from rocjitsu_pytest import write_json


class CompareTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "tensilelite/Tests/common/sample.yaml"
        self.source.parent.mkdir(parents=True)
        self.source.write_text("example")
        self.solutions = [
            dict(
                ProblemType={"DataType": "S"},
                SolutionIndex=i,
                SolutionNameMin=f"kernel{i}",
                BaseName=f"library-label-{i}",
                KernelNameMin=f"kernel{i}",
                StreamK=i,
                CustomKernel={"args": ["AddressFlags"]},
            )
            for i in range(2)
        ]
        self.env = dict(
            TENSILELITE_ROOT="/artifact/tensilelite",
            TENSILELITE_CLIENT="/artifact/client",
            ROCM_PATH="/artifact",
            ROCJITSU_TARGET="gfx1250",
            MIRAGE_PROFILE="mi450x",
            CASE_FILTER="sample",
        )

    def item(self, root, nodeid, sizes, candidate):
        output = root / "output" / nodeid.replace("/", "_")
        group = "sample_00_00" if candidate else "sample_01_02"
        data = output / "1_BenchmarkProblems" / group / "Data/00_Final.yaml"
        data.parent.mkdir(parents=True)
        data.write_text(
            yaml.safe_dump(
                [
                    {"MinimumRequiredVersion": "5.0.0"},
                    {"ProblemSizes": []},
                    *self.solutions,
                ]
            )
        )
        ini = (
            output
            / "1_BenchmarkProblems"
            / group
            / "00_Final/caches/id/source/ClientParameters.ini"
        )
        ini.parent.mkdir(parents=True)
        ini.write_text(
            "init-seed=261025\nnum-elements-to-validate=-1\ncheck-synchronizer=True\nPrintWinnersOnly=False\nnum-benchmarks=1\nlog-level=Debug\n"
            + "".join(f"problem-size={n},8,1,16\na-strides=-1,{n},-1\n" for n in sizes)
        )
        directory = (
            root / "jobs" / nodeid / "evidence" / "item"
            if candidate
            else root / "evidence/item"
        )
        directory.mkdir(parents=True)
        phases = [
            dict(
                name=phase,
                output_dir=str(output),
                completed=True,
                wall_seconds=1.0,
                evidence_seconds=0.01,
                files=retain(output, directory / phase),
            )
            for phase in ("build", "run")
        ]
        document = dict(
            schema=1,
            nodeid=nodeid,
            source="tensilelite/Tests/common/sample.yaml",
            source_sha256=digest(self.source),
            phases=phases,
            test_sha256="test",
            helper_sha256="helper",
        )
        stream = io.StringIO()
        stream.write(f"loading config file {ini}\n")
        writer = csv.writer(stream)
        writer.writerow(
            [
                "run",
                "problem-progress",
                "solution-progress",
                "operation",
                "problem-sizes",
                "bias-type",
                "factor-dim",
                "activation-type",
                "solution",
                "validation",
            ]
        )
        for p, n in enumerate(sizes):
            for s in range(2):
                writer.writerow(
                    [
                        0,
                        f"{p}/{len(sizes)-1}",
                        f"{s}/1",
                        "Contraction",
                        f"({n},8,1,16)",
                        "None",
                        "",
                        "None",
                        f"kernel{s}",
                        "PASSED" if s == 0 else "DID_NOT_SATISFY_ASSERTS",
                    ]
                )
        (directory / "call-stdout.log").write_text(stream.getvalue())
        write_json(directory / "item.json", document)
        return directory, document

    def report(self, name, candidate):
        root = self.root / name
        root.mkdir()
        source = dict(
            nodeid="whole",
            path=str(self.source),
            sha256=digest(self.source),
            markers=[],
        )
        jobs = []
        (root / "shape-yamls").mkdir()
        for n in (4, 12):
            generated = root / "shape-yamls" / f"shape{n}.yaml"
            generated.write_text(str(n))
            jobs.append(
                dict(
                    id=f"shape{n}",
                    nodeid=f"shape{n}",
                    work=n * 8 * 16,
                    source=str(self.source),
                    source_sha256=digest(self.source),
                    yaml=str(generated),
                    yaml_sha256=digest(generated),
                    problem_type_index=1,
                    group_index=2,
                    problems=[dict(sizes=[n, 8, 1, 16])],
                )
            )
        tests = {}
        for nodeid, sizes in (
            [(j["nodeid"], [j["problems"][0]["sizes"][0]]) for j in jobs]
            if candidate
            else [("whole", [4, 12])]
        ):
            directory, item = self.item(root, nodeid, sizes, candidate)
            if candidate:
                item["job"] = next(j for j in jobs if j["nodeid"] == nodeid)
                write_json(directory / "item.json", item)
                (root / "jobs" / nodeid / "test.log").write_text(
                    f"mirage: session session-{nodeid}\n"
                )
            collection = directory.parent if candidate else root / "evidence"
            write_json(
                collection / "collection-serial.json",
                dict(
                    selected=[dict(nodeid=nodeid)],
                    collected=[dict(nodeid=nodeid)],
                    deselected=[],
                ),
            )
            tests[nodeid] = dict(
                outcome="passed",
                phases={
                    phase: dict(outcome="passed", seconds=1)
                    for phase in ("setup", "call", "teardown")
                },
                wall_seconds=3,
            )
        write_json(
            root / "shape-plan.json",
            dict(version=1, ceiling=4096**3, sources=[source], jobs=jobs, excluded=[]),
        )
        write_json(root / "progress.json", dict(selected=len(tests), tests=tests))
        write_json(
            root / "execution.json",
            dict(complete=True, exit_code=0, wall_seconds=10 if candidate else 20),
        )
        write_json(
            root / "runtime.json",
            dict(
                strategy="shape" if candidate else "reference",
                **{
                    k: "frozen"
                    for k in (
                        "artifact_run_id",
                        "client_sha256",
                        "rocjitsu_sha256",
                        "mirage_sha256",
                        "compiler_sha256",
                        "seed",
                        "target",
                        "profile",
                    )
                },
            ),
        )
        return root, audit(root)

    def test_whole_and_split_cover_same_semantic_shapes_solutions_and_verdicts(self):
        _, reference = self.report("reference", False)
        _, candidate = self.report("candidate", True)
        self.assertEqual(reference["audit_errors"], [])
        self.assertEqual(candidate["audit_errors"], [])
        result = compare(reference, candidate)
        self.assertTrue(result["coverage_equal"], result["audit_errors"])
        self.assertEqual(result["reference_over_candidate_ratio"], 2)
        self.assertFalse(result["statistical_claim"])
        self.assertEqual(
            candidate["client_verdicts"], {"PASSED": 2, "DID_NOT_SATISFY_ASSERTS": 2}
        )

    def test_coverage_counts_partial_files_and_arithmetic_separately(self):
        def job(name, source, work, problems):
            return dict(
                nodeid=name,
                source=f"tensilelite/Tests/common/{source}.yaml",
                source_sha256="frozen",
                work=work,
                problems=[dict(sizes=[n, 1, 1, 1]) for n in problems],
                reason="logical work exceeds 10",
            )

        coverage = workload_coverage(
            dict(
                ceiling=10,
                jobs=[
                    job("small", "partial", 2, [2]),
                    job("grouped", "kept", 8, [3, 5]),
                ],
                excluded=[
                    job("large", "partial", 45, [45]),
                    job("only", "removed", 45, [45]),
                ],
            )
        )
        self.assertEqual(coverage["selected_yaml_files"], 3)
        self.assertEqual(coverage["admitted_queue_jobs"], 2)
        self.assertEqual(coverage["admitted_problem_descriptors"], 3)
        self.assertEqual(coverage["threshold_excluded_queue_jobs"], 2)
        self.assertEqual(coverage["threshold_affected_yaml_files"], 2)
        self.assertEqual(coverage["fully_excluded_yaml_files"], 1)
        self.assertEqual(coverage["excluded_declared_work_fraction"], 0.9)
        self.assertEqual(len(coverage["exclusions"]), 2)

    def test_changed_gate_or_emulator_model_prevents_an_elapsed_ratio(self):
        _, reference = self.report("reference", False)
        _, candidate = self.report("candidate", True)
        for field in ("gate", "model", "capacity"):
            with self.subTest(field=field):
                changed = copy.deepcopy(candidate)
                if field == "gate":
                    changed["coverage"]["ceiling"] *= 2
                elif field == "model":
                    changed["runtime"]["config"] = {"max_ticks": 10}
                else:
                    changed["runtime"]["available_cpus"] = 48
                result = compare(reference, changed)
                self.assertFalse(result["coverage_equal"])
                self.assertIsNone(result["reference_over_candidate_ratio"])

    def test_child_cpu_accounting_and_missing_resource_evidence(self):
        runtime = dict(available_cpus=4)
        self.assertIsNone(resource_statistics(self.root, runtime))
        usage = dict(wall_seconds=20, user_seconds=8, system_seconds=2, exit_code=0)
        write_json(self.root / "resource-time.json", usage)
        stats = resource_statistics(self.root, runtime)
        self.assertEqual(stats["mean_cpu_cores"], 0.5)
        self.assertEqual(stats["cpu_capacity_fraction"], 0.125)
        _, reference = self.report("reference", False)
        _, candidate = self.report("candidate", True)
        reference["resource_time"] = stats
        self.assertIsNone(
            compare(reference, candidate)["reference_over_candidate_ratio"]
        )
        candidate["resource_time"] = dict(stats, wall_seconds=5)
        self.assertEqual(
            compare(reference, candidate)["reference_over_candidate_ratio"], 4
        )
        for wall in (0, -1, float("nan")):
            with self.subTest(wall=wall):
                write_json(
                    self.root / "resource-time.json", dict(usage, wall_seconds=wall)
                )
                with self.assertRaises(ValueError):
                    resource_statistics(self.root, runtime)

    def test_matrix_retains_failed_arm_and_only_reports_audited_ratios(self):
        seed, _ = self.report("seed", False)
        write_json(
            self.root / "shape-plan.json",
            json.loads((seed / "shape-plan.json").read_text()),
        )
        calls = []

        def trial(strategy, parent, env, plan, workers, seconds, suite, label):
            calls.append((label, strategy, workers))
            _, report = self.report(label, strategy == "shape")
            report["runtime"].update(
                workers=workers, cpu_thread_budget=1 if strategy == "shape" else 6
            )
            if label == "shape-96":
                report["audit_errors"].append("interrupted numerical verdict sweep")
                report["counts"]["job_timeout"] = 1
                report["ledger"].pop()
            return report

        command = [
            "rocjitsu_experiment",
            "--report-dir",
            str(self.root),
            "--strategy",
            "matrix",
            "--workers",
            "96",
            "--per-job-seconds",
            "300",
            "--suite-seconds",
            "500",
        ]
        with (
            patch.object(sys, "argv", command),
            patch.object(os, "sched_getaffinity", return_value=set(range(96))),
            patch.dict(os.environ, COHORT_MANIFEST="", GITHUB_STEP_SUMMARY=""),
            patch.object(rocjitsu_experiment, "trial", side_effect=trial),
        ):
            self.assertEqual(rocjitsu_experiment.main(), 1)
        self.assertEqual(
            set(calls),
            {
                ("reference", "reference", 16),
                ("shape-48", "shape", 48),
                ("shape-96", "shape", 96),
            },
        )
        benchmark = json.loads((self.root / "benchmark.json").read_text())
        self.assertEqual(len(benchmark["strategies"]), 3)
        self.assertEqual(len(benchmark["comparisons"]), 3)
        self.assertFalse(benchmark["statistical_claim"])
        self.assertTrue(
            all(r["mean_cpu_cores"] is None for r in benchmark["strategies"])
        )
        for pair in benchmark["comparisons"]:
            if "shape-96" in (pair["left"], pair["right"]):
                self.assertIsNone(pair["reference_over_candidate_ratio"])
            else:
                self.assertEqual(pair["reference_over_candidate_ratio"], 2)
        self.assertFalse(
            json.loads((self.root / "experiment.json").read_text())["complete"]
        )
        self.assertIn("shape-96", (self.root / "benchmark.md").read_text())

    def test_drift_in_kernel_semantics_and_unknown_metadata_is_not_ignored(self):
        original = self.solutions[0]
        renamed = original | dict(
            BaseName="new",
            SolutionIndex=99,
            KernelNameMin="short",
            SolutionNameMin="short",
        )
        self.assertEqual(solution_digest(original), solution_digest(renamed))
        for changed in (
            original | dict(StreamK=8),
            original | dict(FutureSafetySetting=False),
            original | dict(CustomKernel={"args": ["AddressWorkspace"]}),
        ):
            self.assertNotEqual(solution_digest(original), solution_digest(changed))

    def test_missing_or_duplicate_processed_rows_invalidates_sweep(self):
        for mutation in ("missing", "duplicate", "malformed", "unknown"):
            root = self.root / mutation
            root.mkdir()
            directory, item = self.item(root, "example", [4, 12], False)
            log = directory / "call-stdout.log"
            lines = log.read_text().splitlines()
            if mutation == "missing":
                lines.pop()
            elif mutation == "duplicate":
                lines.append(lines[-1])
            elif mutation == "malformed":
                lines[-1] += ",extra"
            else:
                lines[-1] = lines[-1].replace("DID_NOT_SATISFY_ASSERTS", "NEW_STATUS")
            log.write_text("\n".join(lines))
            with self.assertRaises(ValueError, msg=mutation):
                audit_item(root, directory, item)

    def test_queue_journal_recovers_stale_snapshot_and_rejects_truncated_event(self):
        root, expected = self.report("journal", True)
        progress = json.loads((root / "progress.json").read_text())
        journal = "".join(
            json.dumps(dict(schema=1, nodeid=nodeid, record=record)) + "\n"
            for nodeid, record in progress["tests"].items()
        )
        write_json(
            root / "progress.json",
            dict(
                selected=2,
                tests={
                    nodeid: dict(outcome="incomplete", phases={})
                    for nodeid in progress["tests"]
                },
            ),
        )
        (root / "queue-events.jsonl").write_text(journal)
        recovered = audit(root)
        self.assertEqual(recovered["audit_errors"], [])
        self.assertEqual(recovered["counts"], expected["counts"])
        (root / "queue-events.jsonl").write_text(journal[:-5])
        incomplete = audit(root)
        self.assertTrue(any("queue event" in e for e in incomplete["audit_errors"]))
        self.assertFalse(compare(expected, incomplete)["coverage_equal"])

    def test_interrupted_phase_without_final_timing_still_produces_a_failed_audit(self):
        root, _ = self.report("interrupted", True)
        metadata = next((root / "jobs").glob("*/evidence/*/item.json"))
        item = json.loads(metadata.read_text())
        item["phases"][1] = dict(
            name="run", completed=False, output_dir=item["phases"][1]["output_dir"]
        )
        write_json(metadata, item)
        report = audit(root)
        self.assertTrue(any("interrupted" in e for e in report["audit_errors"]))

    def test_hidden_verdicts_tampered_ini_and_changed_validation_invalidate_claim(self):
        root, reference = self.report("reference", False)
        _, candidate = self.report("candidate", True)
        changed = copy.deepcopy(candidate)
        changed["inventory"][0]["controls"]["check-synchronizer"] = ["False"]
        self.assertFalse(compare(reference, changed)["coverage_equal"])
        changed = copy.deepcopy(candidate)
        changed["ledger"][0]["validation"] = "NO_CHECK"
        self.assertFalse(compare(reference, changed)["coverage_equal"])
        changed = copy.deepcopy(candidate)
        changed["counts"]["job_timeout"] = 1
        self.assertIsNone(compare(reference, changed)["reference_over_candidate_ratio"])
        ini = next((root / "evidence").glob("*/run/**/ClientParameters.ini"))
        ini.write_text(
            ini.read_text().replace("PrintWinnersOnly=False", "PrintWinnersOnly=True")
        )
        self.assertTrue(audit(root)["audit_errors"])

    def test_reference_preserves_generation_allocation_and_rejects_gate_exclusions(
        self,
    ):
        plan = dict(excluded=[])
        root = self.root / "trial"
        reference = commands("reference", root, plan, self.env, 16, 300, 600)
        candidate = commands("shape", root, plan, self.env, 96, 300, 600)
        self.assertFalse(any("CpuThreads=1" in value for value in reference))
        self.assertTrue(any("CpuThreads=1" in value for value in candidate))
        self.assertIn("--capture=fd", reference)
        self.assertIn("--capture=fd", candidate)
        for command in (reference, candidate):
            self.assertEqual(
                command[command.index("-c") + 1], "/artifact/tensilelite/pytest.ini"
            )
        self.assertTrue(any("DataInitSeed=261025" in value for value in reference))
        with self.assertRaises(ValueError):
            commands(
                "reference", root, dict(excluded=["giant"]), self.env, 16, 300, 600
            )


if __name__ == "__main__":
    unittest.main()
