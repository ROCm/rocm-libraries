# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only tests: python -m unittest discover -s clients/scripts/kernel_corpus."""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import textwrap
import unittest
import zlib
from pathlib import Path
from unittest import mock

import msgpack

sys.path.insert(0, str(Path(__file__).resolve().parent))

import corpus_index  # noqa: E402
import plan  # noqa: E402
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
        self.assertEqual(set(shapes), set(plan.SHAPE_TAGS))
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


FAKE_TEST = textwrap.dedent(
    """\
    #!{python}
    # Stands in for hipblaslt-test: passes every case except solution {bad}, which "faults".
    import re, sys
    print("hipBLASLt git version: fake123")
    print("Device ID 0 : AMD Fake GPU gfx90a:sramecc+:xnack-")
    args = sys.argv[1:]
    if "--yaml" not in args:
        sys.exit(0)
    text = open(args[args.index("--yaml") + 1]).read()
    names = re.findall(r"^- name: (\\S+)", text, re.M)
    for name in names:
        test = "_/matmul_test.matmul/pre_checkin_" + name + "_f32_r_APIAlgoIndex"
        print("[ RUN      ] " + test, flush=True)
        if name.startswith("kc_{bad}_"):
            print("Warning: Queue error - HSA_STATUS_ERROR_MEMORY_FAULT", flush=True)
            sys.exit(134)
        print("[       OK ] " + test + " (1 ms)", flush=True)
    print("[  PASSED  ] %d test%s." % (len(names), "" if len(names) == 1 else "s"))
    """
)


class EndToEndTests(unittest.TestCase):
    def test_a_faulting_kernel_ends_only_its_case_and_the_run_resumes(self):
        with tempfile.TemporaryDirectory() as d:
            rocm = Path(d) / "rocm"
            (rocm / "bin").mkdir(parents=True)
            fake = rocm / "bin" / "hipblaslt-test"
            fake.write_text(FAKE_TEST.format(python=sys.executable, bad=4))
            fake.chmod(0o755)
            (rocm / "bin" / "hipblaslt_common.yaml").write_text("---\n")
            write_library(
                rocm / "lib" / "hipblaslt" / "library" / "gfx90a",
                [solution(i) for i in range(8)] + [solution(8, groupedGemm=True)],
            )
            out = Path(d) / "out"
            argv = [
                "--rocm-dir",
                str(rocm),
                "--out",
                str(out),
                "--chunk-cases",
                "10",
                "--cu-count",
                "104",
            ]
            with mock.patch("sys.stdout", io.StringIO()), mock.patch("time.sleep"):
                code = run.main(argv)
            self.assertEqual(code, run.EXIT_FAILURES)
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["case_status"], {"pass": 28, "crash": 4, "deferred": 1})
            self.assertEqual({f["index"] for f in summary["failures"]}, {4})
            self.assertTrue((out / "repro" / "4_exact.yaml").exists())

            # A rerun has nothing left to do and reports the same results.
            with mock.patch("sys.stdout", io.StringIO()) as stdout:
                self.assertEqual(run.main(argv), run.EXIT_FAILURES)
            self.assertIn("0 cases to run", stdout.getvalue())
            again = json.loads((out / "summary.json").read_text())
            self.assertEqual(again["case_status"], summary["case_status"])


if __name__ == "__main__":
    unittest.main()
