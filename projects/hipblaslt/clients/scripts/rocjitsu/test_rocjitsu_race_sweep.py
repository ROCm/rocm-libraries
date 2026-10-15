#!/usr/bin/env python3
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
import threading
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import zlib

import msgpack
import yaml
import rocjitsu_race_sweep as sweep
import rocjitsu_sweep_plan as plan
from rocjitsu_known_failures import apply_known_failures, known_failures


def solution(index, tile=(64, 96, 1), depth=32):
    return {
        "index": index,
        "name": f"solution-{index}",
        "kernelName": f"kernel-{index}",
        "problemType": {
            "operationIdentifier": "Contraction_l_Ailk_Bjlk_Cijk_Dijk",
            "aType": "Float",
            "bType": "Float",
            "cType": "Float",
            "dType": "Float",
            "computeType": "Float",
            "transA": False,
            "transB": True,
            "biasDataTypeWhiteList": [],
            "biasSrcWhiteList": [3],
        },
        "hardwarePredicate": {"type": "Processor", "value": "gfx942"},
        "problemPredicate": {"type": "And", "value": []},
        "sizeMapping": {
            "macroTile": list(tile),
            "depthU": depth,
            "globalSplitU": 1,
            "streamK": 0,
        },
    }


def job():
    s = solution(7)
    return {
        "id": 0,
        "solutions": [{"index": 7, "name": "solution-7", "kernel": "kernel-7"}],
        "cases": plan.derive_cases(s),
        "problem_type": s["problemType"],
        "library": "library.dat",
    }


def success_log(j=None):
    j = job() if j is None else j
    s = j["solutions"][0]
    stream = io.StringIO()
    for n, case in enumerate(j["cases"]):
        stream.write(sweep.target_dispatch_line(j) + "\n")
        csv.writer(stream, lineterminator="\n").writerow(
            [
                0,
                f"{n}/3",
                f"{s['index']}/{s['index']}",
                "Contraction",
                "(" + ",".join(map(str, case["shape"])) + ")",
                "None",
                "",
                "None",
                s["name"],
                "PASSED",
            ]
        )
    return stream.getvalue()


def bench_log(j=None):
    j = job() if j is None else j
    s = j["solutions"][0]
    lines = []
    for case in j["cases"]:
        m, n, batch, k = case["shape"]
        lines += [sweep.target_dispatch_line(j)] * 2
        lines += [
            "[0]:m,n,batch_count,k,norm_error,atol,rtol",
            f"    {m},{n},{batch},{k},0,0.00001,0.00001",
            f"    --Solution index: {s['index']}",
            f"    --Solution name:  {s['name']}",
            f"    --kernel name:    {s['kernel']}",
        ]
    return "\n".join(lines) + "\n"


def failure_list():
    return {
        "version": 3,
        "rocjitsu_revision": "a" * 40,
        "case_policy": plan.POLICY,
        "failures": [
            {
                "id": "test-hang",
                "reason": "Unsupported instruction leaves the client waiting",
                "tracking": ["https://github.com/ROCm/rocm-systems/pull/12963"],
                # Review dates are reminders; an old date must still apply skips.
                "review_by": "2000-01-01",
                "target": "gfx942",
                "backends": ["tensile", "bench"],
                "cases": [c["id"] for c in job()["cases"]],
                "kernel_names": ["kernel-7"],
            }
        ],
    }


class SweepTests(unittest.TestCase):
    def test_known_failures_follow_kernel_names_across_artifact_changes(self):
        j = job()
        j["code_objects"] = ["kernel.co"]
        original = {
            "target": "gfx942",
            "backend": "tensile",
            "policy": plan.POLICY,
            "input_sha256": {"library.dat": "1" * 64, "kernel.co": "2" * 64},
            "jobs": [j],
        }
        data = failure_list()
        data["failures"][0]["cases"] = ["tile"]
        data["failures"][0]["backends"] = ["tensile"]
        for difference in (
            None,
            "index",
            "metadata",
            "code",
            "solution",
            "kernel",
            "target",
            "backend",
            "planning",
        ):
            with self.subTest(difference=difference):
                manifest = copy.deepcopy(original)
                if difference == "index":
                    manifest["jobs"][0]["solutions"][0]["index"] = 8
                elif difference == "metadata":
                    manifest["input_sha256"]["library.dat"] = "3" * 64
                elif difference == "code":
                    manifest["input_sha256"]["kernel.co"] = "3" * 64
                elif difference == "solution":
                    manifest["jobs"][0]["solutions"][0]["name"] = "different-settings"
                elif difference == "kernel":
                    manifest["jobs"][0]["solutions"][0]["kernel"] = "different-kernel"
                elif difference == "target":
                    manifest["target"] = "gfx950"
                elif difference == "backend":
                    manifest["backend"] = "bench"
                elif difference == "planning":
                    manifest["jobs"][0]["planning_error"] = "Missing code object"
                before = copy.deepcopy(manifest["jobs"])
                apply_known_failures(manifest, yaml.safe_dump(data), "a" * 40)
                cases = manifest["jobs"][0]["cases"]
                self.assertEqual(
                    [c["id"] for c in cases if c.get("skip")],
                    (
                        ["tile"]
                        if difference in (None, "index", "metadata", "code", "solution")
                        else []
                    ),
                )
                for case in cases:
                    case.pop("skip", None)
                self.assertEqual(
                    manifest["jobs"],
                    before,
                    "skips must not replace or reshape selections",
                )

    def test_kernel_skip_applies_to_every_solution_using_the_name(self):
        jobs = [job(), job(), job()]
        for i, j in enumerate(jobs):
            j["id"] = i
            j["solutions"][0].update(index=7 + i, name=f"solution-{7 + i}")
        jobs[1]["cases"][0]["shape"] = [128, 192, 1, 256]
        jobs[2]["solutions"][0]["kernel"] = "different-kernel"
        manifest = {
            "target": "gfx942",
            "backend": "tensile",
            "policy": plan.POLICY,
            "jobs": jobs,
        }
        data = failure_list()
        data["failures"][0]["cases"] = ["tile"]
        apply_known_failures(manifest, yaml.safe_dump(data), "a" * 40)
        self.assertEqual(manifest["known_failures"]["skipped_cases"], 2)
        self.assertEqual(
            [[c["id"] for c in j["cases"] if c.get("skip")] for j in jobs],
            [["tile"], ["tile"], []],
        )

    def test_known_failure_schema_revision_and_overlap_guards(self):
        for change in (
            "missing_tracking",
            "bad_case",
            "bad_name",
            "unknown_key",
        ):
            with self.subTest(change=change):
                data = failure_list()
                entry = data["failures"][0]
                if change == "missing_tracking":
                    entry["tracking"] = []
                elif change == "bad_case":
                    entry["cases"] = ["typo"]
                elif change == "bad_name":
                    entry["kernel_names"] = [7]
                else:
                    entry["case"] = ["tile"]
                with self.assertRaises(ValueError):
                    known_failures(yaml.safe_dump(data))
        for change in ("revision", "policy", "overlap"):
            with self.subTest(change=change):
                data = failure_list()
                manifest = {
                    "target": "gfx942",
                    "backend": "tensile",
                    "policy": plan.POLICY,
                    "input_sha256": {"library.dat": "1" * 64, "kernel.co": "2" * 64},
                    "jobs": [{**job(), "code_objects": ["kernel.co"]}],
                }
                revision = "a" * 40
                if change == "revision":
                    revision = "b" * 40
                elif change == "policy":
                    manifest["policy"] = "new-policy"
                else:
                    duplicate = copy.deepcopy(data["failures"][0])
                    duplicate["id"] = "overlap"
                    data["failures"].append(duplicate)
                with self.assertRaises(ValueError):
                    apply_known_failures(manifest, yaml.safe_dump(data), revision)

    def exercise_skips(
        self, root, backend, cases, indices=(7, 8), recheck=False, failure=None
    ):
        """Use the real planner/list/runner; only the native subprocess is simulated."""
        library = root / "library_gfx942.dat"
        library.write_bytes(
            msgpack.packb(
                {"solutions": [solution(i) for i in indices]}, use_bin_type=True
            )
        )
        code_object = root / "library_gfx942.co"
        code_object.write_bytes(b"kernel")
        config = root / "config.json"
        config.write_text(json.dumps({"vm": {"gpu": {"device": {}}}}))
        data = failure_list()
        entry = data["failures"][0]
        entry["cases"] = cases
        path = root / "failures.yaml"
        path.write_text(yaml.safe_dump(data))
        args = SimpleNamespace(
            reports=root / "reports",
            backend=backend,
            workers=1,
            timeout=120,
            suite_timeout=1500,
            seed="test",
            target="gfx942",
            library_dir=root,
            solutions=100,
            solution_indices=list(indices),
            config=config,
            rocjitsu=code_object,
            client=code_object,
            known_failures=path,
            rocjitsu_revision=("b" if recheck else "a") * 40,
            run_known_failures=recheck,
        )
        invocations = []

        def execute(command, log, timeout, env):
            native_input = Path(command[-1]).read_text()
            if backend == "tensile":
                options = [line.split("=", 1) for line in native_input.splitlines()]
                index = int(next(v for k, v in options if k == "solution-start-idx"))
                shapes = [
                    list(map(int, v.split(",")))
                    for k, v in options
                    if k == "problem-size"
                ]
            else:
                rows = yaml.safe_load(native_input)
                self.assertEqual(len({r["solution_index"] for r in rows}), 1)
                index = rows[0]["solution_index"]
                shapes = [[r[k] for k in ("M", "N", "batch_count", "K")] for r in rows]
            expected = [
                c
                for c in plan.derive_cases(solution(index))
                if recheck or index != 7 or c["id"] not in cases
            ]
            self.assertEqual(
                shapes,
                [c["shape"] for c in expected],
                "known failures reached native input",
            )
            invocations.append(index)
            if failure == "exception":
                raise ValueError("native adapter failed")
            native_job = {
                "solutions": [
                    {
                        "index": index,
                        "name": f"solution-{index}",
                        "kernel": f"kernel-{index}",
                    }
                ],
                "cases": expected,
            }
            text = (
                success_log(native_job)
                if backend == "tensile"
                else bench_log(native_job)
            )
            if failure == "race":
                text += "RACE new finding\nEND_RACE\n"
            elif failure == "warning":
                text += "[rj warn] new unsupported instruction\n"
            elif failure == "dispatch":
                text = text.replace(
                    sweep.target_dispatch_line(native_job) + "\n", "", 1
                )
            elif failure == "numerical":
                text = (
                    text.replace(",PASSED", ",FAILED", 1)
                    if backend == "tensile"
                    else text.replace("0.00001", "nan", 1)
                )
            log.write_text(text)
            return 0

        with (
            patch.object(sweep, "physical_cpus", return_value=[0]),
            patch.object(sweep, "execute_command", side_effect=execute),
        ):
            self.assertEqual(sweep.run(args), int(failure is not None))
        summary = json.loads((args.reports / "summary.json").read_text())
        manifest = json.loads((args.reports / "manifest.json").read_text())
        self.assertEqual(manifest["planned_cases"], 4 * len(indices))
        self.assertEqual(summary["accounted_cases"], 4 * len(indices))
        self.assertEqual(
            (args.reports / "known_failures.yaml").read_bytes(), path.read_bytes()
        )
        self.assertEqual(manifest["known_failures"]["sha256"], plan.sha256(path))
        return args.reports, summary, invocations

    def test_full_and_partial_skips_filter_inputs_and_allow_following_jobs(self):
        for backend in ("tensile", "bench"):
            for cases, indices in (
                ([c["id"] for c in job()["cases"]], (7,)),
                ([c["id"] for c in job()["cases"]], (7, 8)),
                (["tile", "mn_edge"], (7, 8)),
            ):
                with (
                    self.subTest(backend=backend, cases=cases, indices=indices),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    reports, summary, invocations = self.exercise_skips(
                        Path(tmp), backend, cases, indices
                    )
                    self.assertEqual(
                        invocations, [i for i in indices if i != 7 or len(cases) < 4]
                    )
                    self.assertEqual(
                        summary["counts"].get("PASSED", 0),
                        4 * len(indices) - len(cases),
                    )
                    self.assertEqual(summary["counts"]["SKIPPED"], len(cases))
                    self.assertEqual(summary["missing"], [])
                    self.assertEqual(summary["unstarted_cases"], [])
                    self.assertTrue(summary["passed"])
                    self.assertFalse(summary["fully_validated"])
                    self.assertIsNone(summary["stop"])
                    self.assertIn(
                        "Run status: PASS WITH SKIPS.",
                        (reports / "summary.md").read_text(),
                    )
                    for case in summary["results"][0]["cases"]:
                        if case["status"] == "SKIPPED":
                            self.assertEqual(case["target_dispatches"], 0)
                            self.assertIn("12963", case["known_failure"]["tracking"][0])
                    if len(cases) == 4:
                        self.assertIsNone(summary["results"][0]["returncode"])
                        self.assertEqual(summary["results"][0]["target_dispatches"], 0)
                        self.assertFalse((reports / "batch-000.command.json").exists())

    def test_remaining_case_failures_still_stop_the_sweep(self):
        for backend in ("tensile", "bench"):
            for failure in ("race", "warning", "dispatch", "numerical", "exception"):
                with (
                    self.subTest(backend=backend, failure=failure),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    _, summary, invocations = self.exercise_skips(
                        Path(tmp), backend, ["tile", "mn_edge"], failure=failure
                    )
                    self.assertEqual(invocations, [7])
                    self.assertFalse(summary["passed"])
                    self.assertFalse(summary["fully_validated"])
                    self.assertEqual(summary["stop"]["reason"], "job-failure")
                    self.assertEqual(summary["counts"]["SKIPPED"], 2)
                    self.assertEqual(len(summary["unstarted_cases"]), 4)
                    if failure == "exception":
                        self.assertEqual(len(summary["missing"]), 2)

    def test_recheck_executes_known_cases_and_keeps_strict_validation(self):
        for backend in ("tensile", "bench"):
            for failure in (None, "race"):
                with (
                    self.subTest(backend=backend, failure=failure),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    _, summary, invocations = self.exercise_skips(
                        Path(tmp),
                        backend,
                        [c["id"] for c in job()["cases"]],
                        recheck=True,
                        failure=failure,
                    )
                    self.assertEqual(invocations, [7] if failure else [7, 8])
                    self.assertEqual(summary["counts"].get("SKIPPED", 0), 0)
                    self.assertEqual(summary["fully_validated"], failure is None)

    def test_shipped_known_failures_are_valid(self):
        data = known_failures(
            Path(sweep.__file__).with_name("known_failures.yaml").read_text()
        )
        self.assertTrue(data["rocjitsu_revision"])

    def test_native_and_compressed_messagepack(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = msgpack.packb({"solutions": [solution(7)]}, use_bin_type=True)
            for name, data in [
                ("lib.dat", payload),
                ("lib.dat.zlib", zlib.compress(payload)),
            ]:
                path = Path(tmp) / name
                path.write_bytes(data)
                self.assertEqual(plan.read_library(path)["solutions"][0]["index"], 7)

    def test_corrupt_metadata_is_rejected_before_inventory(self):
        payload = msgpack.packb({"solutions": [solution(7)]}, use_bin_type=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lib.dat.zlib"
            for data in (
                zlib.compress(payload) + b"trailing",
                zlib.compress(payload)[:-1],
            ):
                with self.subTest(data=data[-8:]):
                    path.write_bytes(data)
                    with self.assertRaises(zlib.error):
                        plan.read_library(path)
            path = Path(tmp) / "lib.dat"
            path.write_bytes(payload + payload)
            with self.assertRaises(msgpack.ExtraData):
                plan.read_library(path)

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
        self.assertTrue(plan.matches_hardware(predicate, "gfx942", device))
        self.assertFalse(plan.matches_hardware(predicate, "gfx950", device))
        self.assertFalse(plan.matches_hardware({"type": "Unknown"}, "gfx942", device))

    def test_sample_is_seeded_order_independent_and_reaches_every_index(self):
        solutions = [solution(i) for i in range(100)]
        alias = copy.deepcopy(solutions[0])
        alias["index"] = 1000
        solutions.append(alias)
        same_kernel = copy.deepcopy(alias)
        same_kernel.update(index=1001, name="another-solution")
        same_kernel["sizeMapping"]["globalSplitU"] = 4
        solutions.append(same_kernel)

        def choose(seed, values, count=10):
            entries = ((Path("library.dat"), s) for s in values)
            return [s["index"] for _, s in plan.select_solutions(entries, count, seed)]

        self.assertEqual(choose("0", solutions), choose("0", list(reversed(solutions))))
        self.assertNotEqual(choose("0", solutions), choose("1", solutions))
        self.assertEqual(choose("0", solutions), choose("0", solutions, 20)[:10])
        all_indices = choose("0", solutions, None)
        self.assertEqual(set(all_indices), {s["index"] for s in solutions})
        self.assertEqual(len(all_indices), len(solutions))
        self.assertEqual(choose("0", solutions, 102), all_indices)
        self.assertEqual(choose("0", solutions, 101), all_indices[:101])
        with self.assertRaisesRegex(ValueError, "found only 102"):
            choose("0", solutions, 103)
        with self.assertRaisesRegex(ValueError, "No eligible solutions"):
            choose("0", [], None)

    def test_selection_survives_renumbering_and_repackaging_with_repeated_names(self):
        solutions = [solution(i) for i in range(7, 16)]
        for s in solutions:
            s["internalArgsSupport"] = {"version": 0}
            s["taskPredicate"] = {"type": "True"}
        for i, s in enumerate(solutions[1:], 1):
            s["name"] = solutions[0]["name"]
            s["kernelName"] = solutions[0]["kernelName"]
            # A solution name need not describe all execution metadata.
            if i == 1:
                s["sizeMapping"]["globalSplitU"] = 4
            elif i == 2:
                s["problemType"]["useScaleAB"] = "Vector"
            elif i == 3:
                s["hardwarePredicate"] = {"type": "True"}
            elif i == 4:
                s["problemPredicate"] = {"type": "True"}
            elif i == 5:
                s["kernelName"] = "alternate-kernel"
            elif i == 6:
                s["internalArgsSupport"]["version"] = 1
            elif i == 7:
                s["taskPredicate"] = {"type": "False"}
            # The last entry is identical apart from its numeric index.

        def configurations(values, library, count):
            return [
                {
                    k: v
                    for k, v in s.items()
                    if k not in ("index", "libraryLogicIndex", "ideals", "linearModel")
                }
                for _, s in plan.select_solutions(
                    ((Path(library), s) for s in values), count, "0"
                )
            ]

        renamed = copy.deepcopy(solutions)
        for i, s in enumerate(renamed):
            s["index"] = 100 - i
            s["libraryLogicIndex"] = i * 10
            s["ideals"] = {1: float(i)}
            s["linearModel"] = {"slope": i}
            s["sizeMapping"] = dict(reversed(list(s["sizeMapping"].items())))
        for count in range(1, len(solutions) + 1):
            with self.subTest(count=count):
                self.assertEqual(
                    configurations(solutions, "first.dat", count),
                    configurations(reversed(renamed), "repacked.dat.zlib", count),
                )

    def test_cli_accepts_larger_counts_and_all_solution_selection(self):
        required = [
            "sweep",
            "--backend",
            "tensile",
            "--target",
            "gfx942",
            "--seed",
            "0",
            "--rocjitsu-revision",
            "a" * 40,
        ]
        for name in ("rocjitsu", "client", "config", "library-dir", "reports"):
            required += ["--" + name, "."]
        for flags, count in (
            ([], 100),
            (["--solutions", "101"], 101),
            (["--all-solutions"], None),
        ):
            with (
                self.subTest(flags=flags),
                patch.object(sys, "argv", required + flags),
                patch.object(sweep, "run", return_value=0) as run,
            ):
                self.assertEqual(sweep.main(), 0)
                self.assertEqual(run.call_args.args[0].solutions, count)
        for flags in (
            ["--solutions", "0"],
            ["--all-solutions", "--solution-index", "7"],
        ):
            with (
                self.subTest(flags=flags),
                patch.object(sys, "argv", required + flags),
                patch.object(sys, "stderr", io.StringIO()),
                patch.object(sweep, "run") as run,
            ):
                with self.assertRaises(SystemExit):
                    sweep.main()
                run.assert_not_called()

    def test_shapes_follow_each_kernels_tile_and_depth(self):
        first, second = plan.derive_cases(solution(0)), plan.derive_cases(
            solution(1, (128, 32, 1), 64)
        )
        self.assertEqual(first[0]["shape"], [64, 96, 1, 32])
        self.assertEqual(second[0]["shape"], [128, 32, 1, 64])
        self.assertTrue(
            first[2]["m_tail"] and first[2]["n_tail"] and first[3]["k_tail"]
        )
        self.assertEqual(len({tuple(c["shape"]) for c in first}), 4)

    def test_size_constraints_adjust_shapes_without_bypassing_them(self):
        s = solution(0)
        s["problemPredicate"]["value"] = [
            {"type": "LeadingFree0SizesGreaterOrEqual", "value": 128},
            {"type": "Free0SizeMultiple", "index": 0, "value": 8},
            {"type": "BoundSizeMultiple", "index": -1, "value": 32},
            {"type": "GlobalSplitUCheckMinK", "value": [32, 4]},
        ]
        cases = plan.derive_cases(s)
        self.assertTrue(
            all(
                c["shape"][0] >= 128 and c["shape"][0] % 8 == 0 and c["shape"][3] >= 128
                for c in cases
            )
        )
        self.assertTrue(all(not c["k_tail"] for c in cases))
        s["problemPredicate"]["value"].append(
            {"type": "SizeLessThan", "index": 0, "value": 64}
        )
        with self.assertRaisesRegex(ValueError, "constraints"):
            plan.derive_cases(s)

    def test_jit_plan_uses_current_metadata_and_keeps_unschedulable_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "library_gfx942.dat.zlib"
            (root / "library_gfx942.co").write_bytes(b"object")
            s = solution(7)

            def write(s):
                path.write_bytes(
                    zlib.compress(msgpack.packb({"solutions": [s]}, use_bin_type=True))
                )

            write(s)
            device = {"device_id": 0x74A1, "simd_count": 1216, "simd_per_cu": 4}
            first = plan.make_plan(root, "gfx942", device, 1, "seed")
            s["sizeMapping"]["depthU"] = 64
            write(s)
            second = plan.make_plan(root, "gfx942", device, 1, "seed")
            self.assertNotEqual(
                first["artifact_fingerprint"], second["artifact_fingerprint"]
            )
            self.assertNotEqual(first["jobs"][0]["cases"], second["jobs"][0]["cases"])
            s["sizeMapping"]["macroTile"] = [8192, 8192, 1]
            write(s)
            failed = plan.make_plan(root, "gfx942", device, 1, "seed")
            self.assertEqual(failed["jobs"][0]["solutions"][0]["index"], 7)
            self.assertIn("planning_error", failed["jobs"][0])
            self.assertEqual(len(failed["jobs"][0]["cases"]), 4)

    def test_explicit_indices_include_aliases_and_keep_unsupported_solutions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, alias, unsupported = solution(7), solution(8), solution(9)
            alias["kernelName"] = first["kernelName"]
            alias["name"] = first["name"]
            unsupported["problemType"]["groupedGemm"] = True
            (root / "library_gfx942.dat").write_bytes(
                msgpack.packb(
                    {"solutions": [first, alias, unsupported]}, use_bin_type=True
                )
            )
            (root / "library_gfx942.co").write_bytes(b"object")
            result = plan.make_plan(root, "gfx942", {}, 100, "seed", [9, 8, 7])
            self.assertEqual(result["selection"], "solution-indices")
            self.assertEqual(result["kernels"], 2)
            self.assertEqual(result["planned_cases"], 12)
            self.assertEqual(
                [j["solutions"][0]["index"] for j in result["jobs"]], [7, 8, 9]
            )
            self.assertIn(
                "outside the client adapter scope", result["jobs"][2]["planning_error"]
            )
            with self.assertRaisesRegex(ValueError, "indices not found:.*10"):
                plan.make_plan(root, "gfx942", {}, 100, "seed", [7, 10])
            all_eligible = plan.make_plan(root, "gfx942", {}, None, "seed")
            self.assertEqual(all_eligible["selection"], "all-eligible-solutions")
            self.assertEqual(all_eligible["solutions"], 2)
            self.assertEqual(all_eligible["kernels"], 1)
            self.assertEqual(all_eligible["planned_cases"], 8)
            self.assertEqual(
                {j["solutions"][0]["index"] for j in all_eligible["jobs"]}, {7, 8}
            )
            self.assertEqual(all_eligible["inventory"]["outside_adapter_scope"], 1)

    def test_native_metadata_stays_authoritative_for_each_adapter(self):
        j = job()
        self.assertEqual(
            sweep.client_options(j, Path("results"))["solution-start-idx"], 7
        )
        rows = sweep.bench_options(j)
        self.assertEqual([r["K"] for r in rows], [c["shape"][3] for c in j["cases"]])
        self.assertTrue(
            all(r["solution_index"] == 7 and r["algo_method"] == 2 for r in rows)
        )
        self.assertTrue(
            all(
                r["norm_check"] and r["norm_check_assert"] and r["allclose_check"]
                for r in rows
            )
        )

    def test_bench_preserves_fnuz_compute_inputs_with_half_storage(self):
        # Sampled solution 236890 stores Half but computes with two FNUZ formats.
        j = job()
        j["problem_type"].update(
            aType="Half",
            bType="Half",
            cType="Half",
            dType="Half",
            computeInputTypeA="Float8_fnuz",
            computeInputTypeB="BFloat8_fnuz",
        )
        rows = sweep.bench_options(j)
        self.assertTrue(all(r["a_type"] == r["b_type"] == "f16_r" for r in rows))
        self.assertTrue(all(r["compute_input_typeA"] == "f8_fnuz_r" for r in rows))
        self.assertTrue(all(r["compute_input_typeB"] == "bf8_fnuz_r" for r in rows))

    def test_integer_bench_inputs_use_native_compute_type(self):
        # Bench accepts INT8 storage, but rejects an explicit INT8 compute-input
        # override. Its unset default selects the integer path from compute_type.
        for output_type, bench_type in (("Int8", "i8_r"), ("Int32", "i32_r")):
            with self.subTest(output_type=output_type):
                j = job()
                j["problem_type"].update(
                    aType="Int8",
                    bType="Int8",
                    cType=output_type,
                    dType=output_type,
                    computeType="Int32",
                    computeInputTypeA="Int8",
                    computeInputTypeB="Int8",
                )
                for row in sweep.bench_options(j):
                    self.assertEqual(row["a_type"], "i8_r")
                    self.assertEqual(row["b_type"], "i8_r")
                    self.assertEqual(row["d_type"], bench_type)
                    self.assertEqual(row["compute_type"], "c_i32_r")
                    self.assertNotIn("compute_input_typeA", row)
                    self.assertNotIn("compute_input_typeB", row)

    def test_duplicate_artifact_indices_fail_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for shard in ("first", "second"):
                (root / f"{shard}_gfx942.dat").write_bytes(
                    msgpack.packb({"solutions": [solution(7)]}, use_bin_type=True)
                )
            with self.assertRaisesRegex(ValueError, "Duplicate solution index 7"):
                plan.make_plan(root, "gfx942", {}, 1, "seed")

    def test_run_accounts_for_unsupported_and_unstarted_cases(self):
        jobs = [job(), job()]
        jobs[0]["planning_error"] = "No bounded shapes"
        jobs[0]["cases"] = [{"id": c["id"], "shape": None} for c in jobs[0]["cases"]]
        jobs[1]["id"] = 1
        manifest = {
            "jobs": jobs,
            "library_dir": "unused",
            "artifact_fingerprint": "test",
            "planned_cases": 8,
            "seed": "test",
            "policy": plan.POLICY,
            "preparation_seconds": 0,
        }
        for backend in ("tensile", "bench"):
            manifest["backend"] = backend
            with tempfile.TemporaryDirectory() as tmp:
                args = SimpleNamespace(
                    reports=Path(tmp) / "reports",
                    backend=backend,
                    workers=1,
                    timeout=1,
                    suite_timeout=1500,
                    seed="test",
                )
                with (
                    patch.object(sweep, "prepare", return_value=(manifest, None)),
                    patch.object(sweep, "physical_cpus", return_value=[0]),
                    patch.object(sweep, "execute_command") as execute,
                ):
                    self.assertEqual(sweep.run(args), 1)
                    execute.assert_not_called()
                summary = json.loads((args.reports / "summary.json").read_text())
                self.assertEqual(summary["counts"], {"UNSUPPORTED_CASE": 4})
                self.assertEqual(len(summary["unstarted_cases"]), 4)
                self.assertEqual(summary["missing"], [])
                self.assertEqual(summary["accounted_cases"], 8)
                self.assertFalse(summary["passed"])
                report = (args.reports / "summary.md").read_text()
                self.assertIn("| 7 | FAIL | 0/4 | 0/", report)
                self.assertIn("| 7 | NOT RUN |", report)
                self.assertIn("Run status: FAIL.", report)

    def test_suite_budget_charges_preparation_and_preserves_accounting(self):
        for backend in ("tensile", "bench"):
            for phase in ("preparation", "process", "completed", "within_budget"):
                with (
                    self.subTest(backend=backend, phase=phase),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    jobs = [
                        job()
                        for _ in range(2 if phase in ("preparation", "process") else 1)
                    ]
                    for n, j in enumerate(jobs):
                        j.update(id=n, code_objects=[])
                    manifest = {
                        "jobs": jobs,
                        "backend": backend,
                        "library_dir": "unused",
                        "artifact_fingerprint": "test",
                        "planned_cases": 4 * len(jobs),
                        "seed": "test",
                        "policy": plan.POLICY,
                        "preparation_seconds": 2,
                    }
                    args = SimpleNamespace(
                        reports=Path(tmp) / "reports",
                        backend=backend,
                        workers=1,
                        timeout=120,
                        suite_timeout=5,
                        seed="test",
                        rocjitsu=Path("emulator"),
                        client=Path("client"),
                    )
                    clock = [0]

                    def prepare(_):
                        clock[0] = 6 if phase == "preparation" else 2
                        return manifest, Path("config")

                    def execute(command, log, timeout, env):
                        # Preparation leaves only three of the five seconds.
                        self.assertEqual(timeout, 3)
                        clock[0] = 4 if phase == "within_budget" else 5
                        log.write_text(
                            ""
                            if phase == "process"
                            else success_log() if backend == "tensile" else bench_log()
                        )
                        return 124 if phase == "process" else 0

                    with (
                        patch.object(
                            sweep.time, "monotonic", side_effect=lambda: clock[0]
                        ),
                        patch.object(sweep, "prepare", side_effect=prepare),
                        patch.object(sweep, "physical_cpus", return_value=[0]),
                        patch.object(
                            sweep, "execute_command", side_effect=execute
                        ) as launch,
                    ):
                        self.assertEqual(sweep.run(args), int(phase != "within_budget"))
                    self.assertEqual(launch.call_count, int(phase != "preparation"))
                    summary = json.loads((args.reports / "summary.json").read_text())
                    self.assertEqual(
                        summary["accounted_cases"], manifest["planned_cases"]
                    )
                    self.assertEqual(summary["suite_timeout"], 5)
                    self.assertEqual(summary["suite_seconds"], clock[0])
                    settings = json.loads((args.reports / "settings.json").read_text())
                    self.assertEqual(settings["suite_timeout"], 5)
                    report = (args.reports / "summary.md").read_text()
                    if phase == "within_budget":
                        self.assertTrue(summary["passed"])
                        self.assertIsNone(summary["stop"])
                    else:
                        self.assertFalse(summary["passed"])
                        self.assertEqual(summary["stop"]["reason"], "suite-timeout")
                        self.assertIn("Stopped: suite-timeout.", report)
                    self.assertEqual(
                        len(summary["unstarted_cases"]),
                        8 if phase == "preparation" else 4 if phase == "process" else 0,
                    )
                    self.assertEqual(
                        len(summary["missing"]), 4 if phase == "process" else 0
                    )
                    if phase == "process":
                        self.assertIn(
                            "Suite time budget exhausted",
                            summary["results"][0]["errors"],
                        )
                    if phase in ("completed", "within_budget"):
                        self.assertEqual(summary["counts"], {"PASSED": 4})
                    else:
                        self.assertIn("| 7 | NOT RUN |", report)

    def test_report_distinguishes_numerical_pass_from_race_failure(self):
        manifest = {
            "backend": "tensile",
            "seed": "a|b",
            "policy": plan.POLICY,
            "preparation_seconds": 0,
            "artifact_fingerprint": "test",
            "jobs": [job()],
        }
        result = sweep.classify_tensile(job(), success_log(), 0)
        report = sweep.render_report(manifest, {"results": [result], "passed": True})
        self.assertIn("| 7 | PASS | 4/4 | 4/4 |", report)
        self.assertIn("a&#124;b", report)
        result = sweep.classify_tensile(job(), success_log() + "RACE example\n", 0)
        report = sweep.render_report(manifest, {"results": [result], "passed": False})
        self.assertIn("| 7 | FAIL | 4/4 | 4/4 |", report)
        self.assertIn("1 race reports", report)
        self.assertNotIn("| PASS |", report)

    def test_per_case_dispatch_proof_rejects_redistributed_or_wrong_symbols(self):
        dispatch = sweep.target_dispatch_line(job()) + "\n"
        for classify, text in [
            (sweep.classify_tensile, success_log()),
            (sweep.classify_bench, bench_log()),
        ]:
            # Same total launches but zero for the first case, extras for the next.
            count = 2 if classify is sweep.classify_bench else 1
            changed = text.replace(dispatch * count, "", 1)
            changed = changed.replace(dispatch * count, dispatch * (2 * count), 1)
            self.assertTrue(classify(job(), changed, 0)["failed"])
            changed = text.replace('symbol="kernel-7"', 'symbol="wrong"', 1)
            self.assertTrue(classify(job(), changed, 0)["failed"])

    def test_numerical_and_dispatch_evidence_required(self):
        for classify, text in [
            (sweep.classify_tensile, success_log()),
            (sweep.classify_bench, bench_log()),
        ]:
            self.assertFalse(classify(job(), text, 0)["failed"])
            self.assertTrue(classify(job(), text, 1)["failed"])
            self.assertTrue(
                classify(
                    job(), text.replace("[rocjitsu] Kernel dispatch:", "missing:", 1), 0
                )["failed"]
            )
            for suffix in [
                "RACE kernel=? dispatch=9\nEND_RACE\n",
                "[rj warn] unsupported HW_ID\n",
            ]:
                self.assertTrue(classify(job(), text + suffix, 0)["failed"])
        self.assertTrue(
            sweep.classify_tensile(
                job(), success_log().replace(",PASSED", ",NO_CHECK", 1), 0
            )["failed"]
        )

    def test_missing_duplicate_and_rejected_cases_fail(self):
        text = success_log()
        record = next(line for line in text.splitlines() if ",PASSED" in line)
        for changed in [
            text.replace(record + "\n", "", 1),
            text + record + "\n",
            text.replace(",PASSED", ",DID_NOT_SATISFY_ASSERTS", 1),
        ]:
            self.assertTrue(sweep.classify_tensile(job(), changed, 0)["failed"])
        text = bench_log()
        for changed in [
            text.replace("index: 7", "index: 8", 1),
            text.replace("kernel name:    kernel-7", "kernel name:    wrong", 1),
            text.replace("0.00001", "failed", 1),
            text.replace("0.00001", "nan", 1),
            text + text,
        ]:
            self.assertTrue(sweep.classify_bench(job(), changed, 0)["failed"])

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

    def test_suite_expiry_stops_assignment_and_drains_four_workers(self):
        clock = [0]
        gate = threading.Barrier(4, action=lambda: clock.__setitem__(0, 5))

        def execute(task, slot):
            gate.wait(timeout=2)
            return {"id": task["id"], "failed": False, "cases": []}

        with patch.object(sweep.time, "monotonic", side_effect=lambda: clock[0]):
            result = sweep.run_queue(
                [{"id": i} for i in range(10)], 4, execute, lambda _: None, deadline=5
            )
        self.assertEqual(result["stop"]["reason"], "suite-timeout")
        self.assertEqual(result["stop"]["in_flight"], 3)
        self.assertEqual(result["unstarted"], list(range(4, 10)))
        self.assertEqual([r["id"] for r in result["results"]], list(range(4)))
        self.assertTrue(all(not r["failed"] for r in result["results"]))

    def test_expired_command_budget_never_launches(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(sweep.subprocess, "Popen") as launch,
        ):
            result = sweep.execute_command(["unused"], Path(tmp) / "log", 0, {})
            self.assertEqual(result, 124)
            launch.assert_not_called()

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
            except (FileNotFoundError, ProcessLookupError):
                state = "exited"
            self.assertIn(state, {"exited", "Z", "X"})

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
