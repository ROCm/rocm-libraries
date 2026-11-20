# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only tests: python -m unittest discover -s clients/scripts/kernel_corpus."""

from __future__ import annotations

import datetime
import io
import json
import os
import sys
import tempfile
import textwrap
import unittest
import zlib
from pathlib import Path
from typing import Optional
from unittest import mock

import msgpack

sys.path.insert(0, str(Path(__file__).resolve().parent))

import corpus_index  # noqa: E402
import known_bugs  # noqa: E402
import plan  # noqa: E402
import refusals  # noqa: E402
import report  # noqa: E402
import run  # noqa: E402


def solution(
    index,
    *,
    a="Float",
    c=None,
    compute="Float",
    compute_input=None,
    trans=(False, False),
    mt=(64, 32),
    du=16,
    gsu=1,
    extra_predicates=(),
    cu=None,
    **problem,
):
    c = c or a
    predicates = [
        {"type": "OperationIdentifierEqual", "value": "Contraction_l_Ailk_Bljk_Cijk_Dijk"},
        {"type": "LeadingFree0SizesGreaterOrEqual", "value": 2},
        {"type": "GlobalSplitUCheckMinK", "value": [du, gsu]},
        *extra_predicates,
    ]
    hardware = {"type": "Processor", "value": "gfx90a"}
    if cu:
        hardware = {"type": "And", "value": [hardware, {"type": "CUCount", "value": cu}]}
    return {
        "index": index,
        "name": f"Cijk_test_MT{mt[0]}x{mt[1]}x{du}_GSU{gsu}_{index}",
        "kernelName": f"Cijk_test_MT{mt[0]}x{mt[1]}x{du}",
        "problemType": {
            "aType": a,
            "bType": a,
            "cType": c,
            "dType": c,
            "computeType": compute,
            "computeInputTypeA": compute_input or a,
            "transA": trans[0],
            "transB": trans[1],
            "stridedBatched": True,
            "f32XdlMathOp": "Float",
            "useScaleAB": "",
            **problem,
        },
        "sizeMapping": {"macroTile": [mt[0], mt[1], 1], "depthU": du, "globalSplitU": gsu},
        "problemPredicate": {"type": "And", "value": predicates},
        "hardwarePredicate": {"type": "AMDGPU", "value": hardware},
        "taskPredicate": {"type": "And", "value": []},
    }


def write_library(lib_dir: Path, solutions: list[dict]) -> None:
    lib_dir.mkdir(parents=True, exist_ok=True)
    half = len(solutions) // 2
    packed = msgpack.packb({"solutions": solutions[:half], "library": {}})
    (lib_dir / "TensileLibrary_A_Type_gfx90a.dat.zlib").write_bytes(zlib.compress(packed))
    packed = msgpack.packb({"solutions": solutions[half:], "library": {}})
    (lib_dir / "TensileLibrary_B_Type_gfx90a.dat").write_bytes(packed)
    # Lazy-loading index files carry no solutions and must be ignored.
    (lib_dir / "TensileLibrary_lazy_gfx90a.dat").write_bytes(msgpack.packb({"library": {}}))


def record(sol: dict) -> dict:
    return corpus_index.solution_record(sol, "test.dat", "gfx90a")


def mx_solution(index, *, a="Float6", k=65536, swizzle_b=True, scale_format=3, **problem):
    """An exact-size MX FP6 custom kernel as the rocKE Equality logic registers it: TN, bf16 D,
    M = N = 4096, one K, alpha 1, beta 0 and scales in 128x4 tiles."""
    exact = [
        {"type": "SizeEqual", "index": 0, "value": 4096},
        {"type": "SizeEqual", "index": 1, "value": 4096},
        {"type": "SizeEqual", "index": 2, "value": 1},
        {"type": "SizeEqual", "index": 3, "value": k},
        {"type": "StrideAEqual", "index": 1, "value": k},
        {"type": "StrideBEqual", "index": 1, "value": k},
        {"type": "StrideDEqual", "index": 1, "value": 4096},
        {"type": "AlphaValue", "value": "1"},
        {"type": "BetaZero"},
        {"type": "MXScaleFormatEquals", "value": scale_format},
    ]
    fields = {
        "c": "BFloat16",
        "trans": (True, False),
        "mt": (256, 256),
        "du": 128,
        "extra_predicates": exact,
        "mxBlockA": 32,
        "mxBlockB": 32,
        "swizzleTensorA": True,
        "swizzleTensorB": swizzle_b,
        "useBeta": False,
        **problem,
    }
    sol = solution(index, a=a, **fields)
    sol["name"] = f"rocke_gemm_mx_{a}_4096x4096x{k}_{index}"
    return sol


class CorpusIndexTests(unittest.TestCase):
    def test_reads_plain_and_compressed_files_and_sorts_by_index(self):
        with tempfile.TemporaryDirectory() as d:
            lib = Path(d) / "gfx90a"
            write_library(lib, [solution(3), solution(1), solution(2, a="Half")])
            records = corpus_index.build_corpus(lib, "gfx90a")
        self.assertEqual([r["index"] for r in records], [1, 2, 3])
        self.assertEqual(records[0]["sizes"]["macroTile"], [64, 32, 1])

    def test_size_predicates_skip_problem_type_restatements(self):
        r = record(
            solution(0, extra_predicates=[{"type": "Free0SizeMultiple", "index": 0, "value": 8}])
        )
        self.assertNotIn("OperationIdentifierEqual", r["predicates"])
        self.assertEqual(r["predicates"]["Free0SizeMultiple[0]"], 8)
        self.assertEqual(r["predicates"]["GlobalSplitUCheckMinK"], [16, 1])

    def test_hardware_requirements_read_nested_cu_count(self):
        self.assertEqual(
            record(solution(0, cu=104))["hardware"], {"processor": "gfx90a", "cu_count": 104}
        )
        self.assertEqual(record(solution(0))["hardware"], {"processor": "gfx90a"})

    def test_device_specific_libraries_keep_every_chip_id(self):
        sol = solution(0)
        sol["hardwarePredicate"] = {
            "type": "AMDGPU",
            "value": {
                "type": "And",
                "value": [
                    {"type": "Processor", "value": "gfx950"},
                    {
                        "type": "Or",
                        "value": [
                            {"type": "PciChipId", "value": 0x75A3},
                            {"type": "PciChipId", "value": 0x75A2},
                        ],
                    },
                ],
            },
        }
        self.assertEqual(record(sol)["hardware"]["pci_chip_ids"], [0x75A3, 0x75A2])

    def test_default_rocm_dir_follows_therock_bin_dir(self):
        with mock.patch.dict(os.environ, {"THEROCK_BIN_DIR": "/x/build/bin"}, clear=True):
            self.assertEqual(corpus_index.default_rocm_dir(), Path("/x/build"))


class PlanTests(unittest.TestCase):
    def types(self, **kw):
        return plan.problem_types(record(solution(0, **kw)))

    def test_type_mapping(self):
        self.assertEqual(self.types()[0]["compute_type"], "c_f32_r")
        self.assertEqual(
            self.types(a="Half", compute_input="BFloat16")[0]["compute_type"], "c_f32_fast_bf16_r"
        )
        t, _ = self.types(a="Int8", c="Int32", compute="Int32")
        self.assertEqual(
            (t["a_type"], t["d_type"], t["compute_type"], t["scale_type"]),
            ("i8_r", "i32_r", "c_i32_r", "i32_r"),
        )
        self.assertEqual(self.types(a="Double", compute="Double")[0]["compute_type"], "c_f64_r")
        self.assertTrue(
            self.types(a="ComplexFloat", compute="ComplexFloat")[1].startswith("unsupported")
        )

    def test_scale_factors_the_problem_type_requires(self):
        t, _ = self.types(a="Float8_fnuz", c="BFloat16", useScaleAB="Scalar")
        self.assertEqual(
            (t["a_type"], t["compute_type"], t["scaleA"], t["scaleB"]),
            ("f8_fnuz_r", "c_f32_r", 1, 1),
        )
        self.assertEqual(self.types(a="Float8", c="Half", useScaleAB="Vector")[0]["scaleA"], 2)
        t, _ = self.types(a="Float8", useScaleCD=True)
        self.assertEqual((t["scaleC"], t["scaleD"]), (1, 1))
        t, _ = self.types(a="Half", compute_input="Float8_fnuz")
        self.assertEqual((t["compute_type"], t["compute_input_typeA"]), ("c_f32_r", "f8_fnuz_r"))
        r = record(solution(0, a="Float8", c="Half", compute_input="Half"))
        r["problem"]["bType"] = "Half"
        t, _ = plan.problem_types(r)
        self.assertEqual(
            (t["a_type"], t["b_type"], t["compute_input_typeA"], t["compute_input_typeB"]),
            ("f8_r", "f16_r", "f16_r", "f16_r"),
        )

    def test_unsupported_problem_types_are_reported_not_dropped(self):
        self.assertEqual(self.types(groupedGemm=True), (None, "deferred: grouped GEMM"))
        self.assertTrue(self.types(useScaleAB="Block")[1].startswith("deferred"))
        self.assertIn("unmapped", self.types(a="Float6")[1])

    def test_smoke_shapes_honor_tile_split_k_and_predicates(self):
        r = record(
            solution(
                0,
                mt=(64, 32),
                du=16,
                gsu=3,
                extra_predicates=[{"type": "Free0SizeMultiple", "index": 0, "value": 8}],
            )
        )
        shapes = {s.tag: s for s in plan.smoke_shapes(r)}
        self.assertEqual(set(shapes), set(plan.SMOKE_TAGS))
        for s in shapes.values():
            self.assertEqual(s.m % 8, 0)
            self.assertGreaterEqual(s.k, 16 * 3)
        self.assertEqual((shapes["exact"].m, shapes["exact"].n, shapes["exact"].k), (128, 64, 96))
        self.assertNotEqual(shapes["ragged"].m % 64, 0)
        self.assertNotEqual(shapes["ragged"].n % 32, 0)
        self.assertNotEqual(shapes["ktail"].k % 16, 0)
        self.assertGreater(shapes["batch"].batch, 1)

    def test_yaml_names_round_trip_to_case_ids(self):
        cases, _ = plan.plan_kernel(record(solution(812, trans=(True, False))))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "c.yaml"
            plan.write_cases_yaml(cases, path, Path(d) / "hipblaslt_common.yaml")
            text = path.read_text()
        self.assertIn(f"include: {Path(d).resolve()}/hipblaslt_common.yaml", text)
        self.assertIn("solution_index: 812", text)
        self.assertIn('transA: "T"', text)
        for case in cases:
            gtest = f"_/matmul_test.matmul/pre_checkin_{case.yaml_name}_f32_r_NN_1_2_APIAlgoIndex"
            self.assertEqual(plan.case_id_from_test_name(gtest), case.case_id)

    def test_batched_cases_get_non_overlapping_strides(self):
        cases, _ = plan.plan_kernel(record(solution(5)))
        entry = plan.case_yaml_entry(next(c for c in cases if c.shape.batch > 1))
        s = entry["matrix_size"][0]
        self.assertGreaterEqual(entry["stride_a"], s["lda"] * s["K"])
        self.assertGreaterEqual(entry["stride_d"], s["ldd"] * s["N"])


class MXPlanTests(unittest.TestCase):
    def test_fp6_with_128x4_scales_maps_to_scale_mode_1018_and_its_orders(self):
        t, reason = plan.problem_types(record(mx_solution(0)))
        self.assertIsNone(reason)
        self.assertEqual(
            (t["a_type"], t["b_type"], t["d_type"], t["compute_type"], t["scaleA"], t["scaleB"]),
            ("f6_r", "f6_r", "bf16_r", "c_f32_r", 1018, 1018),
        )
        self.assertEqual((t["swizzle_a"], t["swizzle_b"]), (True, True))
        t, _ = plan.problem_types(record(mx_solution(0, a="BFloat6", swizzle_b=False)))
        self.assertEqual((t["a_type"], t["swizzle_b"]), ("bf6_r", False))

    def test_other_mx_kernels_stay_deferred_with_the_reason(self):
        for sol, what in (
            (mx_solution(0, a="Float4"), "Float4 A"),
            (mx_solution(0, scale_format=1), "scale format 1"),
            (mx_solution(0, swizzleTensorA=False), "only FP6 TN"),
            (mx_solution(0, trans=(False, False)), "only FP6 TN"),
        ):
            with self.subTest(what=what):
                types, reason = plan.problem_types(record(sol))
                self.assertIsNone(types)
                self.assertTrue(reason.startswith("deferred: MX block scales"), reason)
                self.assertIn(what, reason)
        # Swizzled operands without MX are still deferred.
        self.assertEqual(
            plan.problem_types(record(solution(0, swizzleTensorA=True)))[1], "deferred: swizzled A"
        )

    def test_an_exact_size_kernel_gets_its_registered_problem_and_the_cpu_reference(self):
        cases, reason = plan.plan_kernel(record(mx_solution(7)))
        self.assertIsNone(reason)
        self.assertEqual([c.shape.tag for c in cases], [plan.REGISTERED_TAG])
        case = cases[0]
        self.assertTrue(case.cpu_reference)
        entry = plan.case_yaml_entry(case)
        self.assertEqual(
            entry["matrix_size"],
            [
                {
                    "M": 4096,
                    "N": 4096,
                    "K": 65536,
                    "lda": 65536,
                    "ldb": 65536,
                    "ldc": 4096,
                    "ldd": 4096,
                }
            ],
        )
        self.assertEqual(
            {k: entry[k] for k in ("alpha", "beta", "use_ext", "use_ext_setproblem")},
            {"alpha": 1.0, "beta": 0.0, "use_ext": 1, "use_ext_setproblem": 1},
        )
        self.assertEqual(
            {k: entry[k] for k in ("initialization", "fast_check", "unit_check", "norm_check")},
            {"initialization": "rand_int", "fast_check": 0, "unit_check": 1, "norm_check": 1},
        )

    def test_e3m2_past_exact_fp32_sums_is_checked_within_tolerances(self):
        exact = plan.plan_kernel(record(mx_solution(0, a="BFloat6", k=65536)))[0][0].args
        self.assertEqual((exact["initialization"], exact["unit_check"]), ("rand_int", 1))
        far = plan.plan_kernel(record(mx_solution(0, a="BFloat6", k=131584)))[0][0].args
        self.assertEqual(
            (far["initialization"], far["unit_check"], far["norm_check"], far["allclose_check"]),
            ("trig_float", 0, 1, 1),
        )
        e2m3 = plan.plan_kernel(record(mx_solution(0, k=131584)))[0][0].args
        self.assertEqual(e2m3["initialization"], "rand_int")

    def test_the_fast_check_mx_oracle_uses_sparse_k_integer_exact_data_at_any_k(self):
        for a, k in (("Float6", 2048), ("BFloat6", 131584)):
            cases, reason = plan.plan_kernel(
                record(mx_solution(4, a=a, k=k)), "smoke", "fast_check"
            )
            self.assertIsNone(reason)
            case = cases[0]
            self.assertFalse(case.cpu_reference)
            entry = plan.case_yaml_entry(case)
            self.assertEqual(
                {
                    key: entry.get(key)
                    for key in (
                        "initialization",
                        "integer_exact_pattern",
                        "fast_check",
                        "unit_check",
                        "norm_check",
                        "allclose_check",
                    )
                },
                {
                    "initialization": "integer_exact",
                    "integer_exact_pattern": "sparse_k",
                    "fast_check": 1,
                    "unit_check": 0,
                    "norm_check": None,
                    "allclose_check": None,
                },
            )
            self.assertEqual(entry["matrix_size"][0]["K"], k)
        # Kernels without MX scales are unchanged, and edges still check only for a refusal.
        plain = plan.plan_kernel(record(solution(5)), "smoke", "fast_check")[0]
        self.assertEqual(plain, plan.plan_kernel(record(solution(5)))[0])
        edges = plan.plan_kernel(record(mx_solution(3)), "edges", "fast_check")[0]
        self.assertTrue(all(c.expect == "refused" and not c.cpu_reference for c in edges))
        with self.assertRaises(ValueError):
            plan.plan_kernel(record(mx_solution(3)), "smoke", "cpu")

    def test_edges_break_one_predicate_each_and_expect_a_refusal(self):
        cases, reason = plan.plan_kernel(record(mx_solution(3)), "edges")
        self.assertIsNone(reason)
        by_tag = {c.shape.tag: c for c in cases}
        self.assertEqual(set(by_tag), set(plan.EDGE_TAGS))
        self.assertTrue(all(c.expect == "refused" and not c.cpu_reference for c in cases))
        size = {tag: plan.case_yaml_entry(c)["matrix_size"][0] for tag, c in by_tag.items()}
        self.assertEqual((size["kup"]["K"], size["kup"]["lda"]), (65536 + 128, 65536 + 128))
        self.assertEqual((size["kdown"]["K"], size["khalf"]["K"]), (65536 - 128, 32768))
        self.assertEqual((size["ldapad"]["K"], size["ldapad"]["lda"]), (65536, 65536 + 128))
        self.assertEqual((size["mup"]["M"], size["mup"]["ldd"]), (4096 + 256, 4096 + 256))
        self.assertEqual((size["lddpad"]["M"], size["lddpad"]["ldd"]), (4096, 4096 + 256))
        self.assertEqual(plan.case_yaml_entry(by_tag["batch2"])["batch_count"], 2)
        self.assertEqual(by_tag["alpha2"].args["alpha"], 2.0)
        self.assertEqual(by_tag["beta1"].args["beta"], 1.0)
        self.assertEqual(by_tag["border"].types["swizzle_b"], False)
        self.assertEqual(by_tag["kup"].types["swizzle_b"], True)

    def test_edges_and_mx_need_an_exact_size_solution(self):
        self.assertEqual(
            plan.plan_kernel(record(solution(0)), "edges"),
            ([], "deferred: the edges profile needs an exact-size (SizeEqual) solution"),
        )
        tiled = mx_solution(0)
        tiled["problemPredicate"]["value"] = [
            p for p in tiled["problemPredicate"]["value"] if p["type"] != "SizeEqual"
        ]
        self.assertTrue(plan.plan_kernel(record(tiled))[1].startswith("deferred: the smoke"))

    def test_new_tags_round_trip_through_gtest_names(self):
        for tag in (plan.REGISTERED_TAG, "batch2", "batch", "kup"):
            gtest = f"_/matmul_test.matmul/pre_checkin_kc_9_{tag}_f6_r_TN_1_2_APIAlgoIndex"
            self.assertEqual(plan.case_id_from_test_name(gtest), f"9:{tag}")


class ParseTests(unittest.TestCase):
    def chunk(self):
        cases = {f"{i}:exact": None for i in (1, 2, 3, 4)}
        return run.ChunkRun(cases, 60, io.StringIO())

    def test_outcomes(self):
        c = self.chunk()
        name = "_/matmul_test.matmul/pre_checkin_kc_{}_exact_f32_r_APIAlgoIndex"
        for line in [
            f"[ RUN      ] {name.format(1)}",
            "hipblaslt-bench --api_method c -m 1 --algo_method index --solution_index 1",
            f"[       OK ] {name.format(1)} (5 ms)",
            f"[ RUN      ] {name.format(2)}",
            "NO solution found!",
            f"[  FAILED  ] {name.format(2)}, where GetParam() = {{ x: 1, devices:  (3 ms)",
            f"[ RUN      ] {name.format(3)}",
            "fast_check: D mismatch at batch 0 row 3 col 4: expected 7, got 3",
            f"[  FAILED  ] {name.format(3)}, where GetParam() = {{ y }} (9 ms)",
            f"[ RUN      ] {name.format(4)}",
            f"[  SKIPPED ] {name.format(4)} (0 ms)",
            f"[  FAILED  ] {name.format(3)}, where GetParam() = {{ y }}",
        ]:
            c.feed(line + "\n")
        got = {r["case_id"]: r for r in c.results}
        self.assertEqual(
            [got[f"{i}:exact"]["status"] for i in (1, 2, 3, 4)], ["pass", "refused", "fail", "skip"]
        )
        self.assertIn("--solution_index 1", got["1:exact"]["bench_cmd"])
        self.assertIn("expected 7", got["3:exact"]["message"])
        self.assertEqual(len(c.results), 4)

    def test_fast_check_refusing_a_configuration_is_not_a_kernel_failure(self):
        c = self.chunk()
        c.feed("[ RUN      ] x/pre_checkin_kc_1_exact_f32\n")
        c.feed("Failed\nfast_check does not support data type 4\n")
        c.feed("[  FAILED  ] x/pre_checkin_kc_1_exact_f32, where GetParam() = { } (2 ms)\n")
        c.feed("[ RUN      ] x/pre_checkin_kc_2_exact_f32\n")
        c.feed("fast_check found a wrong element\n")
        c.feed("[  FAILED  ] x/pre_checkin_kc_2_exact_f32, where GetParam() = { } (2 ms)\n")
        self.assertEqual([r["status"] for r in c.results], ["unsupported", "fail"])

    def test_fault_is_flagged(self):
        c = self.chunk()
        c.feed("[ RUN      ] x/pre_checkin_kc_1_exact_f32\n")
        c.feed("Warning: Queue error - HSA_STATUS_ERROR_MEMORY_FAULT\n")
        self.assertTrue(c.fault)


class ExpectationTests(unittest.TestCase):
    def test_a_case_planned_to_be_refused_passes_only_when_refused(self):
        edge = plan.plan_kernel(record(mx_solution(2)), "edges")[0][0]
        got = {
            status: run.apply_expectation(edge, {"status": status, "message": "m"})
            for status in ("refused", "pass", "crash", "skip")
        }
        self.assertEqual(
            {s: r["status"] for s, r in got.items()},
            {"refused": "expected_refusal", "pass": "fail", "crash": "crash", "skip": "skip"},
        )
        self.assertTrue(got["pass"]["message"].startswith(run.ACCEPTED_OUTSIDE_PREDICATES))
        self.assertIn("[runner]", report.failure_reason({**got["pass"], "fault": False}))
        registered = plan.plan_kernel(record(mx_solution(2)))[0][0]
        self.assertEqual(
            run.apply_expectation(registered, {"status": "refused"})["status"], "refused"
        )

    def test_relaunch_command_replaces_timing_and_adds_the_relaunch_check(self):
        cmd = run.relaunch_command(
            "hipblaslt-bench --api_method c -m 64 -i 10 --cold_iters 2 --solution_index 4",
            Path("/r/bin/hipblaslt-bench"),
            25,
        )
        self.assertEqual(
            cmd,
            [
                "/r/bin/hipblaslt-bench",
                "--api_method",
                "c",
                "-m",
                "64",
                "--solution_index",
                "4",
                "-j",
                "1",
                "-i",
                "1",
                "--post_bench_verify_iters",
                "25",
                "--verify_strict",
            ],
        )

    def test_host_memory_estimate_matches_the_gtest_measurement(self):
        case = plan.plan_kernel(record(mx_solution(0, k=131584)))[0][0]
        self.assertAlmostEqual(run.host_reference_bytes(case) / 2**30, 12, delta=0.5)

    def test_a_repro_keeps_leading_dimensions_arguments_and_expectation(self):
        edge = next(
            c
            for c in plan.plan_kernel(record(mx_solution(2)), "edges")[0]
            if c.shape.tag == "ldapad"
        )
        row = {
            "index": 2,
            "shape": "ldapad",
            "m": edge.shape.m,
            "n": edge.shape.n,
            "k": edge.shape.k,
            "batch": 1,
            "types": edge.types,
            "trans": "TN",
            "ld": [edge.shape.lda, edge.shape.ldb, edge.shape.ldd],
            "args": edge.args,
            "expect": edge.expect,
        }
        rebuilt = report.case_from_row(json.loads(json.dumps(row)))
        self.assertEqual(plan.case_yaml_entry(rebuilt), plan.case_yaml_entry(edge))
        self.assertEqual(rebuilt.expect, "refused")


class ReachabilityTests(unittest.TestCase):
    def test_chip_id_and_cu_count(self):
        r = record(solution(0, cu=104))
        r["hardware"]["pci_chip_ids"] = [0x75A3, 0x75A2]
        self.assertEqual(run.unreachable_reason(r, 104, 0x75A2), "")
        self.assertIn("0x75a3, 0x75a2", run.unreachable_reason(r, 104, 0x75A0))
        self.assertIn("104 CUs", run.unreachable_reason(r, 110, 0x75A2))
        # MI355X (0x75a3) also runs the MI350X (0x75a0) solutions; MI350X does not run its.
        mi350 = record(solution(1))
        mi350["hardware"]["pci_chip_ids"] = [0x75A0]
        self.assertEqual(run.unreachable_reason(mi350, None, 0x75A3), "")
        self.assertEqual(run.unreachable_reason(mi350, None, 0x75A0), "")
        # Unknown device facts leave the decision to the library's own check.
        self.assertEqual(run.unreachable_reason(r, None, None), "")

    def test_rocminfo_gpu_agent(self):
        text = (
            "Agent 1\n  Name: AMD EPYC\n  Device Type: CPU\n  Compute Unit: 96\n"
            "Agent 2\n  Name: gfx950\n  Chip ID:                 30112(0x75a0)\n"
            "  Device Type:             GPU\n  Compute Unit:            256\n"
        )
        self.assertEqual(run.parse_rocminfo_gpu(text), {"cu_count": 256, "pci_chip_id": 0x75A0})

    def test_a_result_from_one_device_outranks_unreachable_on_another(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / "shard-1-of-2").mkdir()
            (out / "shard-2-of-2").mkdir()
            ran = {"case_id": "7:exact", "index": 7, "shape": "exact", "status": "pass"}
            not_here = {"case_id": "7:-", "index": 7, "shape": None, "status": "unreachable"}
            never = {"case_id": "8:-", "index": 8, "shape": None, "status": "unreachable"}
            (out / "shard-1-of-2" / "results.jsonl").write_text(json.dumps(ran) + "\n")
            (out / "shard-2-of-2" / "results.jsonl").write_text(
                json.dumps(not_here) + "\n" + json.dumps(never) + "\n"
            )
            self.assertEqual(set(report.load_results(out)), {"7:exact", "8:-"})


class ShardSelectionTests(unittest.TestCase):
    def test_shards_partition_the_corpus(self):
        records = [record(solution(i)) for i in range(103)]
        seen = []
        for i in range(1, 8):
            args = run.parse_args(
                ["--rocm-dir", "/x", "--out", "/y", "--shard-index", str(i), "--total-shards", "7"]
            )
            seen += [r["index"] for r in run.select_records(records, args)]
        self.assertEqual(sorted(seen), list(range(103)))

    def test_shard_defaults_come_from_therock_environment(self):
        with mock.patch.dict(os.environ, {"SHARD_INDEX": "2", "TOTAL_SHARDS": "4"}):
            args = run.parse_args(["--rocm-dir", "/x", "--out", "/y"])
        self.assertEqual((args.shard_index, args.total_shards), (2, 4))


# Stands in for hipblaslt-test. Every case passes except those whose YAML name starts with BAD,
# which fault the GPU or hang (MODE), and those REFUSE matches, which the pinned solution
# declines. With SICKEN set, the first bad case leaves the GPU unable to run anything, so every
# later process fails at once, the runner's health check included.
FAKE_TEST_BODY = textwrap.dedent(
    r"""
    import os, re, sys, time
    print("hipBLASLt git version: fake123")
    if DEVICE_LINE:
        print("Device ID 0 : AMD Fake GPU gfx90a:sramecc+:xnack-")
    args = sys.argv[1:]
    if "--yaml" not in args:
        sys.exit(0)
    if os.path.exists(SICK_MARKER):
        print("hipErrorNoDevice", flush=True)
        sys.exit(1)
    text = open(args[args.index("--yaml") + 1]).read()
    names = re.findall(r"^- name: (\S+)", text, re.M)
    for name in names:
        test = "_/matmul_test.matmul/pre_checkin_" + name + "_f32_r_APIAlgoIndex"
        print("[ RUN      ] " + test, flush=True)
        if name.startswith(BAD):
            if SICKEN:
                open(SICK_MARKER, "w").close()
            if MODE == "hang":
                # Long past the 1 s case timeout, but short enough that a runner which misses
                # the hang fails the test instead of stalling it.
                time.sleep(30)
            print("Warning: Queue error - HSA_STATUS_ERROR_MEMORY_FAULT", flush=True)
            sys.exit(134)
        if REFUSE and re.search(REFUSE, name):
            print("NO solution found!", flush=True)
            print("[  FAILED  ] " + test + ", where GetParam() = { } (1 ms)", flush=True)
            continue
        index = name.split("_")[1]
        print("hipblaslt-bench -m 64 --algo_method index --solution_index " + index + " -i 10")
        print("[       OK ] " + test + " (1 ms)", flush=True)
    print("[  PASSED  ] %d test%s." % (len(names), "" if len(names) == 1 else "s"))
    """
)

# Stands in for hipblaslt-bench --post_bench_verify_iters: logs its arguments and fails the
# relaunch check, with exit code 2, for solution index BENCH_FAIL.
FAKE_BENCH_BODY = textwrap.dedent(
    r"""
    import sys
    args = sys.argv[1:]
    with open(BENCH_LOG, "a") as f:
        f.write(" ".join(args) + "\n")
    if args[args.index("--solution_index") + 1] == BENCH_FAIL:
        print("post_bench_verify_iters failed for solution: 1 of 3 relaunches gave a different D")
        sys.exit(2)
    """
)


def fake_rocm(
    root: Path,
    bad: str = "kc_4_",
    mode: str = "fault",
    sicken: bool = False,
    device: bool = True,
    refuse: str = "",
    solutions: Optional[list[dict]] = None,
    bench_fail: str = "",
) -> Path:
    """An install tree with the fake hipblaslt-test and hipblaslt-bench and, by default, a
    9-solution gfx90a library: solutions 0 to 7 plan 4 cases each, and solution 8 is a
    deferred grouped GEMM."""
    rocm = root / "rocm"
    (rocm / "bin").mkdir(parents=True)
    fake = rocm / "bin" / "hipblaslt-test"
    settings = (
        f"BAD = {bad!r}\nMODE = {mode!r}\nSICKEN = {sicken!r}\nDEVICE_LINE = {device!r}\n"
        f"SICK_MARKER = {str(root / 'gpu-is-sick')!r}\nREFUSE = {refuse!r}\n"
    )
    fake.write_text(f"#!{sys.executable}\n{settings}{FAKE_TEST_BODY}")
    fake.chmod(0o755)
    bench = rocm / "bin" / "hipblaslt-bench"
    settings = f"BENCH_LOG = {str(root / 'bench.log')!r}\nBENCH_FAIL = {bench_fail!r}\n"
    bench.write_text(f"#!{sys.executable}\n{settings}{FAKE_BENCH_BODY}")
    bench.chmod(0o755)
    (rocm / "bin" / "hipblaslt_common.yaml").write_text("---\n")
    if solutions is None:
        solutions = [solution(i) for i in range(8)] + [solution(8, groupedGemm=True)]
    write_library(rocm / "lib" / "hipblaslt" / "library" / "gfx90a", solutions)
    return rocm


def corpus_argv(rocm: Path, out: Path, *extra: str) -> list[str]:
    # Explicit device facts keep the test from reading the real GPU through rocminfo, and the
    # bundled known-bugs list, whose review dates pass, stays out unless a test names a list.
    if "--known-bugs" not in extra:
        extra += ("--no-known-bugs",)
    return [
        "--rocm-dir",
        str(rocm),
        "--out",
        str(out),
        "--chunk-cases",
        "10",
        "--cu-count",
        "104",
        "--pci-chip-id",
        "0x740f",
        *extra,
    ]


class EndToEndTests(unittest.TestCase):
    def test_a_faulting_kernel_ends_only_its_case_and_the_run_resumes(self):
        with tempfile.TemporaryDirectory() as d:
            rocm = fake_rocm(Path(d))
            out = Path(d) / "out"
            argv = corpus_argv(rocm, out)
            with mock.patch("sys.stdout", io.StringIO()), mock.patch("time.sleep"):
                code = run.main(argv)
            self.assertEqual(code, run.EXIT_FAILURES)
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["case_status"], {"pass": 28, "crash": 4, "deferred": 1})
            self.assertEqual({f["index"] for f in summary["failures"]}, {4})
            self.assertEqual({f["reason"] for f in summary["failures"]}, {"GPU memory fault"})
            self.assertTrue((out / "repro" / "4_exact.yaml").exists())

            # A rerun has nothing left to do and reports the same results.
            with mock.patch("sys.stdout", io.StringIO()) as stdout:
                self.assertEqual(run.main(argv), run.EXIT_FAILURES)
            self.assertIn("0 cases to run", stdout.getvalue())
            again = json.loads((out / "summary.json").read_text())
            self.assertEqual(again["case_status"], summary["case_status"])

    def test_a_hung_case_times_out_and_the_rest_still_run(self):
        # The runner polls every 5 s, so a 1 s case timeout costs about 5 s of wall time.
        with tempfile.TemporaryDirectory() as d:
            rocm = fake_rocm(Path(d), bad="kc_4_exact", mode="hang")
            out = Path(d) / "out"
            with mock.patch("sys.stdout", io.StringIO()), mock.patch("time.sleep"):
                code = run.main(corpus_argv(rocm, out, "--case-timeout", "1"))
            self.assertEqual(code, run.EXIT_FAILURES)
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["case_status"], {"pass": 31, "timeout": 1, "deferred": 1})
            self.assertEqual([f["case_id"] for f in summary["failures"]], ["4:exact"])

    def test_a_gpu_that_stops_working_stops_the_shard_and_shows_why(self):
        with tempfile.TemporaryDirectory() as d:
            rocm = fake_rocm(Path(d), sicken=True)
            out = Path(d) / "out"
            stderr = io.StringIO()
            with mock.patch("sys.stdout", io.StringIO()), mock.patch(
                "sys.stderr", stderr
            ), mock.patch("time.sleep"):
                code = run.main(corpus_argv(rocm, out))
            self.assertEqual(code, run.EXIT_GPU_UNHEALTHY)
            self.assertIn("hipErrorNoDevice", stderr.getvalue())
            summary = json.loads((out / "summary.json").read_text())
            # Solutions 0 to 3 ran before the fault; nothing ran on the sick GPU after it.
            self.assertEqual(summary["case_status"], {"pass": 16, "crash": 1, "deferred": 1})

    def test_a_device_query_that_never_names_a_gpu_is_retried_then_reported(self):
        with tempfile.TemporaryDirectory() as d:
            rocm = fake_rocm(Path(d), device=False)
            stderr = io.StringIO()
            with mock.patch("sys.stderr", stderr), mock.patch("time.sleep") as sleep:
                code = run.main(corpus_argv(rocm, Path(d) / "out"))
            self.assertEqual(code, run.EXIT_GPU_UNHEALTHY)
            self.assertEqual(sleep.call_count, 2)
            self.assertIn("hipBLASLt git version: fake123", stderr.getvalue())


class MXEndToEndTests(unittest.TestCase):
    """Two exact-size MX kernels, one with B tiled and one with B in plain columns."""

    SOLUTIONS = [mx_solution(0), mx_solution(1, swizzle_b=False)]

    def test_edges_pass_when_refused_and_fail_when_the_kernel_runs(self):
        with tempfile.TemporaryDirectory() as d:
            # Solution 1 wrongly runs a K one DepthU past its registered size.
            rocm = fake_rocm(
                Path(d), bad="none", refuse=r"^kc_0_|^kc_1_(?!kup)", solutions=self.SOLUTIONS
            )
            out = Path(d) / "out"
            with mock.patch("sys.stdout", io.StringIO()), mock.patch("time.sleep"):
                code = run.main(corpus_argv(rocm, out, "--profile", "edges"))
            self.assertEqual(code, run.EXIT_FAILURES)
            summary = json.loads((out / "summary.json").read_text())
            n = len(plan.EDGE_TAGS)
            self.assertEqual(summary["case_status"], {"fail": 1, "expected_refusal": 2 * n - 1})
            self.assertEqual(summary["kernel_outcomes"], {"passing": 1, "failing": 1})
            failure = summary["failures"][0]
            self.assertEqual(failure["case_id"], "1:kup")
            self.assertEqual(failure["reason"], run.ACCEPTED_OUTSIDE_PREDICATES)
            self.assertIn('"lda": 65664', (out / "repro" / "1_kup.yaml").read_text())

    def test_cpu_reference_cases_run_alone_and_relaunch_checks_each_solution(self):
        with tempfile.TemporaryDirectory() as d:
            rocm = fake_rocm(Path(d), bad="none", solutions=self.SOLUTIONS, bench_fail="1")
            out = Path(d) / "out"
            with mock.patch("sys.stdout", io.StringIO()), mock.patch("time.sleep"):
                code = run.main(corpus_argv(rocm, out, "--relaunch", "3"))
            self.assertEqual(code, run.EXIT_FAILURES)
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["case_status"], {"pass": 3, "fail": 1})
            self.assertEqual([f["case_id"] for f in summary["failures"]], ["1:relaunch"])
            self.assertIn("post_bench_verify_iters failed", summary["failures"][0]["reason"])
            log = (out / "shard-1-of-1" / "hipblaslt-test.log").read_text()
            self.assertEqual(log.count("(1 cases)"), 2)
            bench = (Path(d) / "bench.log").read_text().splitlines()
            self.assertEqual(len(bench), 2)
            self.assertTrue(
                all(line.endswith("--post_bench_verify_iters 3 --verify_strict") for line in bench)
            )
            self.assertNotIn("-i 10", bench[0])

            # A rerun relaunches nothing again.
            with mock.patch("sys.stdout", io.StringIO()):
                run.main(corpus_argv(rocm, out, "--relaunch", "3"))
            self.assertEqual(len((Path(d) / "bench.log").read_text().splitlines()), 2)


# Stands in for refusal_probe: accepts every registered problem and refuses the rest, except
# labels ACCEPT matches, which it accepts. It names each solution by the planner's convention.
FAKE_PROBE_BODY = textwrap.dedent(
    r"""
    import re, sys
    for line in sys.stdin:
        label, index, a_type = line.split()[:3]
        accept = label.endswith("registered") or (ACCEPT and re.search(ACCEPT, label))
        name = "rocke_gemm_mx_%s_4096x4096x65536_%s" % (
            "Float6" if a_type == "f6_r" else "BFloat6", index)
        print("RESULT %s %s isAlgoSupported %s %s" % (
            label, "accepted" if accept else "refused",
            "SUCCESS" if accept else "INVALID_VALUE", name), flush=True)
    """
)


def fake_probe(root: Path, accept: str = "") -> Path:
    probe = root / "fake_refusal_probe"
    probe.write_text(f"#!{sys.executable}\nACCEPT = {accept!r}\n{FAKE_PROBE_BODY}")
    probe.chmod(0o755)
    return probe


class RefusalProbeTests(unittest.TestCase):
    def test_probe_lines_carry_the_problems_hipblaslt_test_cannot_build(self):
        cases = {c.shape.tag: c for c in refusals.probe_cases(record(mx_solution(4)))}
        self.assertEqual(set(cases), {plan.REGISTERED_TAG, *plan.EDGE_TAGS})
        line = dict((tag, refusals.probe_line(c).split()) for tag, c in cases.items())
        # label index a_type tiled_b m n k batch lda ldb ldd alpha beta
        self.assertEqual(
            line["registered"],
            "4:probe-registered 4 f6_r 1 4096 4096 65536 1 65536 65536 4096 1.0 0.0".split(),
        )
        self.assertEqual(line["batch2"][7], "2")
        self.assertEqual((line["ldapad"][8], line["ldbpad"][9]), ("65664", "65664"))
        self.assertEqual(line["border"][3], "0")
        self.assertEqual(refusals.probe_cases(record(solution(0))), [])

    def test_answers_are_judged_against_the_plan(self):
        rec = record(mx_solution(4))
        cases = {c.shape.tag: c for c in refusals.probe_cases(rec)}
        answer = {"verdict": "refused", "probe_name": rec["name"], "line": "RESULT x"}
        accepted = {**answer, "verdict": "accepted"}
        judge = lambda tag, a: refusals.classify(cases[tag], rec, a)[0]  # noqa: E731
        self.assertEqual(judge("registered", accepted), "pass")
        self.assertEqual(judge("registered", answer), "fail")
        self.assertEqual(judge("batch2", answer), "expected_refusal")
        self.assertEqual(judge("batch2", accepted), "fail")
        self.assertEqual(judge("batch2", None), "crash")
        renamed = {**accepted, "probe_name": "Cijk_something_else"}
        status, message = refusals.classify(cases["registered"], rec, renamed)
        self.assertEqual(status, "fail")
        self.assertIn("Cijk_something_else", message)

    def test_parse_results_reads_only_result_lines(self):
        got = refusals.parse_results(
            "rocblaslt warning\nRESULT 4:probe-kup refused isAlgoSupported INVALID_VALUE n\n"
            "RESULT 4:probe-kdown refused setProblem INVALID_VALUE -\n"
        )
        self.assertEqual(set(got), {"4:probe-kup", "4:probe-kdown"})
        self.assertIsNone(got["4:probe-kdown"]["probe_name"])

    def test_a_missing_header_stops_the_build_with_the_reason(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError, "blas dev artifact"):
                refusals.build_probe(Path(d), Path(d) / "out")


class RefusalEndToEndTests(unittest.TestCase):
    def test_refusals_run_once_and_report_an_accepted_edge(self):
        with tempfile.TemporaryDirectory() as d:
            rocm = fake_rocm(Path(d), bad="none", solutions=MXEndToEndTests.SOLUTIONS)
            probe = fake_probe(Path(d), accept=r"^1:probe-ldapad$")
            out = Path(d) / "out"
            argv = corpus_argv(
                rocm, out, "--profile", "edges", "--refusals", "--refusal-probe", str(probe)
            )
            with mock.patch("sys.stdout", io.StringIO()), mock.patch("time.sleep"):
                code = run.main(argv)
            self.assertEqual(code, run.EXIT_FAILURES)
            summary = json.loads((out / "summary.json").read_text())
            n = len(plan.EDGE_TAGS)
            # The fake hipblaslt-test runs every edge (2n fails); the probe refuses 2n - 1.
            self.assertEqual(
                summary["case_status"],
                {"pass": 2, "fail": 2 * n + 1, "expected_refusal": 2 * n - 1},
            )
            probe_failures = [f for f in summary["failures"] if "probe" in f["case_id"]]
            self.assertEqual([f["case_id"] for f in probe_failures], ["1:probe-ldapad"])
            self.assertEqual(probe_failures[0]["reason"], refusals.ACCEPTED_WITHOUT_DATA)

            log = out / "shard-1-of-1" / "hipblaslt-test.log"
            before = log.read_text().count("refusal probe")
            with mock.patch("sys.stdout", io.StringIO()):
                run.main(argv)
            self.assertEqual(log.read_text().count("refusal probe"), before)


class ReportTests(unittest.TestCase):
    def test_rows_written_before_a_classification_rule_are_classified_again(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / "shard-1-of-1").mkdir()
            old = {
                "case_id": "3:exact",
                "index": 3,
                "shape": "exact",
                "status": "fail",
                "message": "fast_check does not support data type 4",
            }
            (out / "shard-1-of-1" / "results.jsonl").write_text(json.dumps(old) + "\n")
            self.assertEqual(report.load_results(out)["3:exact"]["status"], "unsupported")

    def test_a_fault_is_a_kernel_failure_even_when_fast_check_complains(self):
        message = "fast_check does not support data type 4"
        self.assertEqual(report.classify_failure(message, fault=True), "fail")
        self.assertEqual(report.classify_failure("NO solution found!"), "refused")

    def test_a_layout_the_client_cannot_build_is_unsupported_not_a_kernel_failure(self):
        for message in (
            "Received uncaught exception: HIPBLASLT_ORDER_COL256_R128_EXT requires batch count 1",
            "Received uncaught exception: MX scales require lda equal to the rows stored",
        ):
            self.assertEqual(report.classify_failure(message), "unsupported")
        self.assertEqual(report.classify_failure("Received uncaught exception: boom"), "fail")

    def test_unsupported_cases_are_listed_as_cases_with_the_line_that_explains_them(self):
        why = "Received uncaught exception: MX scales require lda equal to the rows stored"
        results = {
            "0:ldapad": {
                "case_id": "0:ldapad",
                "index": 0,
                "shape": "ldapad",
                "status": "unsupported",
                "message": f"hipblaslt_test.cpp:256: Failure\n{why}",
            },
            "1:-": {"case_id": "1:-", "index": 1, "shape": None, "status": "deferred"},
        }
        results["1:-"]["message"] = "deferred: grouped GEMM"
        summary = report.summarize(results, [], today=TODAY)
        self.assertEqual(summary["unsupported_reasons"], {why: 1})
        self.assertEqual(summary["not_run_reasons"], {"deferred: grouped GEMM": 1})
        text = report.markdown(summary)
        self.assertIn(f"## Unsupported cases\n\n- 1 cases: {why}", text)
        self.assertIn("## Solutions not planned\n\n- 1 solutions: deferred: grouped GEMM", text)


TODAY = datetime.date(2026, 11, 1)


def known_entry(name: str, **fields) -> dict:
    return {
        "arch": "gfx90a",
        "solution_name": name,
        "ticket": "ROCM-1",
        "added": "2026-10-01",
        "review_by": "2027-01-31",
        "reason": "wrong answers",
        **fields,
    }


class KnownBugsParseTests(unittest.TestCase):
    def test_bad_entries_are_rejected_with_the_reason(self):
        bad = [
            ({"ticket": None}, "missing ticket"),
            ({"review_by": None}, "missing review_by"),
            ({"owner": "me"}, "unknown field owner"),
            ({"ticket": "fix later"}, "not a tracker key"),
            ({"added": "Oct 1"}, "added must be a date"),
            ({"review_by": "2026-09-01"}, "review_by is before added"),
        ]
        for fields, message in bad:
            with self.subTest(fields=fields):
                with self.assertRaisesRegex(ValueError, message):
                    known_bugs.parse({"entries": [known_entry("k", **fields)]})

    def test_a_duplicate_entry_is_rejected(self):
        entry = known_entry("k", shape="ktail")
        with self.assertRaisesRegex(ValueError, "duplicates"):
            known_bugs.parse({"entries": [entry, {**entry, "ticket": "ROCM-2"}]})
        # The same kernel narrowed to another shape is a different entry.
        self.assertEqual(
            len(known_bugs.parse({"entries": [entry, {**entry, "shape": "exact"}]})), 2
        )

    def test_the_bundled_list_parses(self):
        entries = known_bugs.load(known_bugs.DEFAULT_PATH)
        self.assertTrue(entries)
        self.assertTrue(all(e.ticket.startswith("ROCM-") for e in entries))


class KnownBugsReportTests(unittest.TestCase):
    """report.write_reports on hand-made shard results: two solutions with the same name that
    differ only in split-K, as custom-GSU solutions do, and one other solution."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        recs = [record(solution(1, gsu=19)), record(solution(2, gsu=7)), record(solution(3))]
        recs[0]["name"] = recs[1]["name"] = "Cijk_custom_gsu"
        corpus_index.write_jsonl(recs, self.out / "corpus.jsonl")
        self.names = {r["index"]: r["name"] for r in recs}

    def tearDown(self):
        self.tmp.cleanup()

    def results(self, statuses: dict[str, str]) -> None:
        shard = self.out / "shard-1-of-1"
        shard.mkdir(exist_ok=True)
        rows = []
        for case_id, status in statuses.items():
            index, shape = case_id.split(":")
            rows.append(
                {
                    "case_id": case_id,
                    "index": int(index),
                    "shape": shape,
                    "status": status,
                    "arch": "gfx90a",
                    "device": "AMD Instinct MI210",
                    "name": self.names[int(index)],
                    "problem_type": "Float/Float/Float/Float/Float/Float/NN",
                    "types": {"a_type": "f32_r"},
                    "trans": "NN",
                    "m": 64,
                    "n": 64,
                    "k": 64,
                    "message": "fast_check: 3 elements of D are wrong",
                }
            )
        (shard / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    def report(self, *entries: dict) -> dict:
        known = known_bugs.parse({"entries": list(entries)})
        return report.write_reports(self.out, Path("common.yaml"), known, TODAY)

    def test_a_listed_failure_is_known_and_does_not_fail_the_run(self):
        self.results({"1:exact": "fail", "1:ktail": "fail", "3:exact": "pass"})
        summary = self.report(known_entry("Cijk_custom_gsu", global_split_u=19))
        self.assertEqual(summary["case_status"], {"pass": 1, "known_fail": 2})
        self.assertEqual((summary["failing_cases"], summary["known_failing_cases"]), (0, 2))
        self.assertEqual(summary["known_bugs"][0]["status"], known_bugs.KNOWN)
        self.assertEqual(summary["known_bugs"][0]["age_days"], 31)
        self.assertFalse(report.run_failed(summary))
        self.assertTrue((self.out / "repro" / "1_exact.yaml").exists())
        self.assertIn("ROCM-1", (self.out / "summary.md").read_text())

    def test_global_split_u_and_shape_narrow_the_match(self):
        # Solution 2 shares the name but not the split-K value; its failure is not covered.
        # Solution 1's exact failure is outside the entry's ktail shape.
        self.results({"1:ktail": "fail", "1:exact": "fail", "2:ktail": "fail"})
        summary = self.report(known_entry("Cijk_custom_gsu", global_split_u=19, shape="ktail"))
        self.assertEqual(summary["case_status"], {"fail": 2, "known_fail": 1})
        self.assertEqual({f["case_id"] for f in summary["failures"]}, {"1:exact", "2:ktail"})
        self.assertTrue(report.run_failed(summary))

    def test_an_entry_whose_cases_all_pass_fails_the_run(self):
        self.results({"3:exact": "pass", "3:ktail": "pass"})
        summary = self.report(known_entry(self.names[3]))
        self.assertEqual(summary["case_status"], {"pass": 2})
        self.assertEqual(summary["known_bugs"][0]["status"], known_bugs.FIXED)
        self.assertTrue(report.run_failed(summary))

    def test_an_entry_past_its_review_date_fails_the_run_even_while_failing(self):
        self.results({"3:exact": "fail"})
        summary = self.report(known_entry(self.names[3], review_by="2026-10-31"))
        self.assertEqual(summary["case_status"], {"known_fail": 1})
        self.assertEqual(summary["known_bugs"][0]["status"], known_bugs.EXPIRED)
        self.assertTrue(report.run_failed(summary))

    def test_an_entry_with_no_results_is_reported_but_does_not_fail_the_run(self):
        # A shard that never reached the kernel, or a retune that renamed it, says nothing about
        # the bug. Entries for another architecture are left out of this report entirely.
        self.results({"3:exact": "pass"})
        summary = self.report(
            known_entry("Cijk_retuned_away"), known_entry("Cijk_other", arch="gfx942")
        )
        self.assertEqual(
            [(k["solution_name"], k["status"]) for k in summary["known_bugs"]],
            [("Cijk_retuned_away", known_bugs.NOT_SEEN)],
        )
        self.assertFalse(report.run_failed(summary))

    def test_report_main_exits_2_on_a_malformed_list(self):
        self.results({"3:exact": "pass"})
        bad = self.out / "bad.yaml"
        bad.write_text("entries:\n  - arch: gfx90a\n")
        with mock.patch("sys.stderr", io.StringIO()) as stderr:
            self.assertEqual(report.main(["--out", str(self.out), "--known-bugs", str(bad)]), 2)
        self.assertIn("missing solution_name", stderr.getvalue())


class KnownBugsEndToEndTests(unittest.TestCase):
    def test_a_listed_faulting_kernel_runs_and_is_reported_as_known(self):
        with tempfile.TemporaryDirectory() as d:
            rocm = fake_rocm(Path(d))
            listed = Path(d) / "known.yaml"
            # run.main judges review dates against the real date, so this one never passes.
            entry = known_entry(solution(4)["name"], review_by="9999-12-31")
            listed.write_text(f"entries:\n  - {json.dumps(entry)}\n")
            out = Path(d) / "out"
            with mock.patch("sys.stdout", io.StringIO()), mock.patch("time.sleep"):
                code = run.main(corpus_argv(rocm, out, "--known-bugs", str(listed)))
            self.assertEqual(code, 0)
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["case_status"], {"pass": 28, "known_fail": 4, "deferred": 1})


if __name__ == "__main__":
    unittest.main()
