import copy
from pathlib import Path
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import therock_matrix


class TheRockMatrixTest(unittest.TestCase):
    def test_collect_projects_to_run_without_additional_option(self):
        subtrees = ["projects/hipblaslt"]

        project_to_run = therock_matrix.collect_projects_to_run(subtrees)
        self.assertEqual(len(project_to_run), 1)
        blas_entry = project_to_run[0]
        self.assertIn(
            "hipsparselt",
            blas_entry["projects_to_test"].split(","),
        )
        self.assertTrue(blas_entry["run_rocjitsu_race_check"])

    def test_rocjitsu_race_check_does_not_run_for_rocblas_only(self):
        project_to_run = therock_matrix.collect_projects_to_run(["projects/rocblas"])
        self.assertEqual(len(project_to_run), 1)
        self.assertFalse(project_to_run[0]["run_rocjitsu_race_check"])

    def test_rocjitsu_race_check_does_not_run_for_provider_rows(self):
        # These rows exercise the names that motivated the explicit selection
        # marker. In particular, `hipblasltprovider` must not match merely
        # because it contains the `hipblaslt` substring.
        provider_subtrees = [
            "dnn-providers/hipblaslt-provider",
            "dnn-providers/hip-kernel-provider",
        ]

        for subtree in provider_subtrees:
            with self.subTest(subtree=subtree):
                project_to_run = therock_matrix.collect_projects_to_run([subtree])
                self.assertGreater(len(project_to_run), 0)
                self.assertFalse(
                    any(
                        project["run_rocjitsu_race_check"] for project in project_to_run
                    )
                )

    def test_rocjitsu_race_check_follows_hipblaslt_into_merged_row(self):
        project_to_run = therock_matrix.collect_projects_to_run(
            ["projects/hipblaslt", "projects/miopen"]
        )
        matching_rows = [
            row for row in project_to_run if row["run_rocjitsu_race_check"]
        ]
        self.assertEqual(len(matching_rows), 1)
        self.assertIn(
            "tensilelite",
            matching_rows[0]["projects_to_test"].split(","),
        )

    def test_collect_projects_to_run_hipthreads(self):
        subtrees = ["projects/hipthreads"]

        project_to_run = therock_matrix.collect_projects_to_run(subtrees)
        self.assertEqual(len(project_to_run), 1)
        hipthreads_entry = project_to_run[0]
        self.assertIn(
            "hipthreads",
            hipthreads_entry["projects_to_test"].split(","),
        )

    def test_collect_projects_to_run(self):
        subtrees = ["projects/rocsparse", "projects/hipblaslt"]

        project_to_run = therock_matrix.collect_projects_to_run(subtrees)
        self.assertEqual(len(project_to_run), 1)

    def test_collect_projects_to_run_additional_option(self):
        subtrees = ["projects/rocsparse"]

        project_to_run = therock_matrix.collect_projects_to_run(subtrees)
        self.assertEqual(len(project_to_run), 1)

    def test_collect_projects_to_run_hiptensor_linux(self):
        # On Linux CK links rocRAND + roctracer/rocprofiler-sdk, so hipTensor
        # must enable RAND and ROCPROFV3 in addition to CK itself.
        with mock.patch.dict(os.environ, {"PLATFORM": "linux"}):
            project_to_run = therock_matrix.collect_projects_to_run(
                ["projects/hiptensor"]
            )
        self.assertEqual(len(project_to_run), 1)
        options = project_to_run[0]["cmake_options"].split(" ")
        self.assertIn("hiptensor", project_to_run[0]["projects_to_test"].split(","))
        self.assertIn("-DTHEROCK_ENABLE_HIPTENSOR=ON", options)
        self.assertIn("-DTHEROCK_ENABLE_COMPOSABLE_KERNEL=ON", options)
        self.assertIn("-DTHEROCK_ENABLE_RAND=ON", options)
        self.assertIn("-DTHEROCK_ENABLE_ROCPROFV3=ON", options)

    def test_collect_projects_to_run_hiptensor_windows(self):
        # The profiler (ROCPROFV3) is not built on Windows, so it must NOT be
        # passed there; RAND is still required by CK.
        with mock.patch.dict(os.environ, {"PLATFORM": "windows"}):
            project_to_run = therock_matrix.collect_projects_to_run(
                ["projects/hiptensor"]
            )
        self.assertEqual(len(project_to_run), 1)
        options = project_to_run[0]["cmake_options"].split(" ")
        self.assertIn("-DTHEROCK_ENABLE_HIPTENSOR=ON", options)
        self.assertIn("-DTHEROCK_ENABLE_COMPOSABLE_KERNEL=ON", options)
        self.assertIn("-DTHEROCK_ENABLE_RAND=ON", options)
        self.assertNotIn("-DTHEROCK_ENABLE_ROCPROFV3=ON", options)

    def test_collect_projects_to_run_rpp_linux(self):
        with mock.patch.dict(os.environ, {"PLATFORM": "linux"}):
            project_to_run = therock_matrix.collect_projects_to_run(["projects/rpp"])
        self.assertEqual(len(project_to_run), 1)
        rpp_entry = project_to_run[0]
        options = rpp_entry["cmake_options"].split(" ")
        self.assertEqual(rpp_entry["projects_to_test"], "rpp")
        self.assertIn("-DTHEROCK_ENABLE_RPP=ON", options)
        self.assertIn("-DTHEROCK_ENABLE_ALL=OFF", options)
        # The platform restriction is a matrix-selection detail and must not
        # leak into the row consumed by the workflow.
        self.assertNotIn("platforms", rpp_entry)

    def test_collect_projects_to_run_rpp_windows(self):
        # RPP is experimental on Windows in TheRock and its test job is
        # Linux-only, so the Windows matrix must not carry an rpp row at all.
        with mock.patch.dict(os.environ, {"PLATFORM": "windows"}):
            project_to_run = therock_matrix.collect_projects_to_run(["projects/rpp"])
        self.assertEqual(project_to_run, [])

    def test_collect_projects_to_run_dependency_graph(self):
        subtrees = ["projects/miopen", "projects/hipblaslt"]

        project_to_run = therock_matrix.collect_projects_to_run(subtrees)
        self.assertEqual(len(project_to_run), 1)

    def test_collect_projects_to_run_dependency_graph_diff_projects(self):
        subtrees = ["projects/miopen", "projects/rocwmma"]

        project_to_run = therock_matrix.collect_projects_to_run(subtrees)
        # rocwmma only contributes via blas under additional_options; miopen absorbs blas.
        self.assertEqual(len(project_to_run), 1)
        combined = project_to_run[0]
        self.assertIn("rocwmma", combined["projects_to_test"].split(","))
        self.assertIn("miopen", combined["projects_to_test"].split(","))

    def test_collect_projects_to_run_does_not_mutate_module_state(self):
        # Snapshot module-level dicts, run a series of representative calls, and
        # confirm the originals are untouched. This guards against the
        # mutate-globals regression that previously required importlib.reload
        # between tests.
        project_map_before = copy.deepcopy(therock_matrix.project_map)
        additional_options_before = copy.deepcopy(therock_matrix.additional_options)

        therock_matrix.collect_projects_to_run(["projects/hipblaslt"])
        therock_matrix.collect_projects_to_run(
            ["projects/rocsparse", "projects/hipblaslt"]
        )
        therock_matrix.collect_projects_to_run(
            ["projects/miopen", "projects/hipblaslt"]
        )
        therock_matrix.collect_projects_to_run(["projects/miopen", "projects/rocwmma"])
        with mock.patch.dict(os.environ, {"PLATFORM": "linux"}):
            therock_matrix.collect_projects_to_run(["projects/rpp"])

        self.assertEqual(therock_matrix.project_map, project_map_before)
        self.assertEqual(therock_matrix.additional_options, additional_options_before)


if __name__ == "__main__":
    unittest.main()
