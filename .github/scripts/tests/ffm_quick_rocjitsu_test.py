# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import sys
import tempfile
import textwrap
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import ffm_quick_rocjitsu as sweep  # noqa: E402


SAMPLE = textwrap.dedent(
    """
    add_test(test_hiprand_kernel "../test_hiprand_kernel")
    set_tests_properties("test_hiprand_kernel" PROPERTIES LABELS "quick;ffm-quick;ffm-full")
    add_test(rocblas-test_ffm-quick_suite python3 "../rocblas_rtest.py" -t ctest_ffm-quick)
    set_tests_properties(rocblas-test_ffm-quick_suite PROPERTIES LABELS "ffm-quick" TIMEOUT 7200)
    add_test(test_hip_api_gpus_ffm-quick_suite [[../test_hip_api]] --gtest_filter=Smoke*)
    set_tests_properties(test_hip_api_gpus_ffm-quick_suite PROPERTIES LABELS "ffm-quick;ffm-full" TIMEOUT 7200)
    add_test("bilinear ffm-quick" "../bilinear" -y configs/quick.yaml)
    set_tests_properties("bilinear ffm-quick" PROPERTIES LABELS "ffm-quick" SKIP_REGULAR_EXPRESSION "unsupported host device")
    add_test(excluded "../excluded")
    set_tests_properties("excluded" PROPERTIES LABELS "ffm-quick;ex_gpu_gfx110X")
    add_test(quick_only "../quick_only")
    set_tests_properties("quick_only" PROPERTIES LABELS "quick")
    """
)


def write_ctest(path: Path, labels: list[str], names: list[str] | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    names = names or ["kernel"]
    joined = ";".join(labels)
    body = []
    for name in names:
        body.append(f'add_test({name} "../{name}")')
        body.append(f'set_tests_properties("{name}" PROPERTIES LABELS "{joined}")')
    path.joinpath("CTestTestfile.cmake").write_text(
        "\n".join(body) + "\n", encoding="utf-8"
    )


class ParseInstalledCTestTest(unittest.TestCase):
    def test_selects_ffm_quick_commands(self):
        tests = sweep.parse_ctest_file(SAMPLE)
        selected = [test for test in tests.values() if sweep.selects_ffm_quick(test)]
        names = [test["name"] for test in selected]
        self.assertEqual(
            names,
            [
                "test_hiprand_kernel",
                "rocblas-test_ffm-quick_suite",
                "test_hip_api_gpus_ffm-quick_suite",
                "bilinear ffm-quick",
            ],
        )
        self.assertEqual(
            tests["rocblas-test_ffm-quick_suite"]["command"],
            ["python3", "../rocblas_rtest.py", "-t", "ctest_ffm-quick"],
        )
        self.assertEqual(
            tests["test_hip_api_gpus_ffm-quick_suite"]["command"][:2],
            ["../test_hip_api", "--gtest_filter=Smoke*"],
        )
        self.assertEqual(sweep.timeout_seconds(tests["test_hiprand_kernel"]), 3600)
        self.assertEqual(
            sweep.timeout_seconds(tests["rocblas-test_ffm-quick_suite"]), 3600
        )

    def test_share_ctest_directory_uses_short_component_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_ctest(
                root / "share" / "hsa-amd-aqlprofile",
                ["unittests"],
                ["legacy-tests"],
            )
            found = sweep.discover(root, label="^quick$", components=["aqlprofile"])
            self.assertEqual([item["component"] for item in found], ["aqlprofile"])
            self.assertEqual(found[0]["tests"][0]["name"], "legacy-tests")

    def test_named_rocjpeg_uses_quick_sample_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "share" / "rocjpeg" / "samples" / "jpegDecode").mkdir(parents=True)
            (root / "share" / "rocjpeg" / "images").mkdir()
            found = sweep.discover(root, label="^quick$", components=["rocjpeg"])
            self.assertEqual([item["component"] for item in found], ["rocjpeg"])
            self.assertEqual(
                [test["name"] for test in found[0]["tests"]],
                [name for name, _extra in sweep.JPEG_QUICK_CASES],
            )
            self.assertEqual(sweep.discover(root, label="^quick$"), [])

    def test_missing_components_accepts_source_quick_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "share" / "rocdecode" / "samples" / "videoDecodeRaw").mkdir(
                parents=True
            )
            (root / "share" / "rocdecode" / "video").mkdir()
            self.assertEqual(
                sweep.missing_components(root, "^quick$", ["rocdecode", "rocjpeg"]),
                ["rocjpeg"],
            )

    def test_sample_binary_path_is_absolute(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "share" / "rocdecode" / "samples" / "videoDecodeRaw").mkdir(
                parents=True
            )
            (root / "share" / "rocdecode" / "video").mkdir()
            found = sweep.discover(root, label="^quick$", components=["rocdecode"])
            self.assertTrue(Path(found[0]["tests"][0]["command"][0]).is_absolute())

    def test_rocjitsu_timeout_is_one_hour(self):
        self.assertEqual(sweep.timeout_seconds({"timeout": 300}), 3600)
        self.assertEqual(sweep.timeout_seconds({"timeout": 7200}), 3600)
        self.assertEqual(sweep.timeout_seconds({"timeout": None}), 3600)
        self.assertEqual(sweep.timeout_seconds({}), 3600)

    def test_skip_regex_matches_ctest_output(self):
        regex = sweep.parse_ctest_file(SAMPLE)["bilinear ffm-quick"]["skip_regex"]
        self.assertTrue(sweep.output_skips("unsupported host device\n", regex))
        self.assertFalse(sweep.output_skips("result mismatch\n", regex))

    def test_discover_orders_small_directories_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_ctest(root / "bin" / "hipRAND", ["ffm-quick"], ["kernel"])
            write_ctest(root / "bin" / "rocprim", ["ffm-quick"], ["t0", "t1", "t2"])
            found = sweep.discover(root)
            self.assertEqual(
                [item["component"] for item in found], ["hipRAND", "rocprim"]
            )

    def test_discover_all_quick_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_ctest(root / "bin" / "hipRAND", ["quick", "standard"], ["api"])
            write_ctest(root / "bin" / "rocBLAS", ["quick"], ["suite"])
            write_ctest(root / "bin" / "ffm_only", ["ffm-quick"], ["ffm"])
            found = sweep.discover(root, label="^quick$")
            self.assertEqual(
                [item["component"] for item in found], ["hipRAND", "rocBLAS"]
            )

    def test_discover_named_component_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_ctest(root / "bin" / "hipRAND", ["quick"], ["api"])
            write_ctest(root / "bin" / "rocBLAS", ["quick"], ["suite"])
            write_ctest(root / "bin" / "rocprim", ["quick"], ["t0"])
            found = sweep.discover(
                root, label="^quick$", components=["hipRAND", "rocprim"]
            )
            self.assertEqual(
                [item["component"] for item in found], ["hipRAND", "rocprim"]
            )
            one = sweep.discover(root, label="^quick$", component="rocBLAS")
            self.assertEqual([item["component"] for item in one], ["rocBLAS"])

    def test_missing_components_lists_unknown_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_ctest(root / "bin" / "hipRAND", ["quick"], ["api"])
            self.assertEqual(
                sweep.missing_components(root, "^quick$", ["hipRAND", "missing"]),
                ["missing"],
            )

    def test_list_json_stdout_is_only_the_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_ctest(root / "bin" / "hipsolver", ["quick"], ["api"])
            captured = StringIO()
            with redirect_stdout(captured):
                code = sweep.main(
                    [
                        "--rocm-root",
                        str(root),
                        "--label",
                        "quick",
                        "--components",
                        "hipsolver",
                        "--list-json",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertEqual(
                json.loads(captured.getvalue()),
                {"include": [{"component": "hipsolver"}]},
            )

    def test_matrix_payload_lists_components(self):
        groups = [{"component": "hipRAND"}, {"component": "rocBLAS"}]
        self.assertEqual(
            sweep.matrix_payload(groups),
            {"include": [{"component": "hipRAND"}, {"component": "rocBLAS"}]},
        )

    def test_parse_component_names(self):
        self.assertEqual(
            sweep.parse_component_names(" hipRAND, rocBLAS , "),
            ["hipRAND", "rocBLAS"],
        )
        self.assertEqual(sweep.parse_component_names(""), [])

    def test_label_pattern_anchors_bare_names(self):
        pattern = sweep.label_pattern("quick")
        self.assertTrue(pattern.search("quick"))
        self.assertFalse(pattern.search("ffm-quick"))

    def test_rocjitsu_arg_starting_with_dashes_is_attached(self):
        joined = sweep.attach_option_values(
            ["--rocjitsu-arg", "--cpu-thread-budget=1", "--list-only"],
            "--rocjitsu-arg",
        )
        self.assertEqual(
            joined,
            ["--rocjitsu-arg=--cpu-thread-budget=1", "--list-only"],
        )


if __name__ == "__main__":
    unittest.main()
