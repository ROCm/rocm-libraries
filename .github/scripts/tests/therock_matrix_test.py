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

        self.assertEqual(therock_matrix.project_map, project_map_before)
        self.assertEqual(therock_matrix.additional_options, additional_options_before)

    @staticmethod
    def _gated(project, *options):
        return {"ci:test-flag": {"project": project, "cmake_options": list(options)}}

    @staticmethod
    def _all_options(project_to_run):
        options = []
        for job in project_to_run:
            options.extend(job["cmake_options"].split(" "))
        return options

    def _run_gated(self, project, subtrees, labels=("ci:test-flag",)):
        with mock.patch.dict(
            therock_matrix.LABEL_GATED_CMAKE_OPTIONS,
            self._gated(project, "-DTHEROCK_FLAG_TEST=ON"),
            clear=True,
        ):
            return therock_matrix.collect_projects_to_run(subtrees, list(labels))

    def test_label_gated_cmake_option_injected_when_label_and_project_present(self):
        project_to_run = self._run_gated("hipthreads", ["projects/hipthreads"])
        self.assertEqual(len(project_to_run), 1)
        self.assertIn("-DTHEROCK_FLAG_TEST=ON", self._all_options(project_to_run))

    def test_label_gated_cmake_option_absent_without_label(self):
        project_to_run = self._run_gated(
            "hipthreads", ["projects/hipthreads"], labels=()
        )
        self.assertEqual(len(project_to_run), 1)
        self.assertNotIn("-DTHEROCK_FLAG_TEST=ON", self._all_options(project_to_run))

    def test_label_gated_cmake_option_absent_when_project_not_built(self):
        # Label is present, but its target project is not in the build set, so the
        # flag must not leak into any job that is being built.
        project_to_run = self._run_gated("hipdnn", ["projects/hipthreads"])
        self.assertEqual(len(project_to_run), 1)
        self.assertNotIn("-DTHEROCK_FLAG_TEST=ON", self._all_options(project_to_run))

    def test_label_gated_options_injected_for_additional_options_project(self):
        # hipdnn is an optional component: it lives in additional_options and is
        # never a project_map key, so it merges into its project_to_add parent.
        project_to_run = self._run_gated("hipdnn", ["projects/hipdnn"])
        self.assertEqual(len(project_to_run), 1)
        self.assertIn("-DTHEROCK_FLAG_TEST=ON", self._all_options(project_to_run))

    def test_label_gated_options_injected_when_parent_already_present(self):
        # Same as above but the project_to_add parent is also being built, which
        # takes the extend branch of the merge instead of the assign branch.
        project_to_run = self._run_gated(
            "hipdnn", ["projects/hipdnn", "projects/miopen"]
        )
        self.assertIn("-DTHEROCK_FLAG_TEST=ON", self._all_options(project_to_run))

    def test_label_gated_options_injected_when_target_absorbed_by_dependency(self):
        # blas is absorbed into miopen when both are built, so the flag has to
        # follow blas into the surviving miopen job rather than being dropped.
        project_to_run = self._run_gated(
            "blas", ["projects/miopen", "projects/rocblas"]
        )
        self.assertIn("-DTHEROCK_FLAG_TEST=ON", self._all_options(project_to_run))

    def test_label_gated_option_overriding_default_is_ordered_last(self):
        # cmake takes the last -D for a given name, so an injected option that
        # contradicts a default must survive dedup in last position.
        default = "-DTHEROCK_FLAG_HIPKERNELPROVIDER_ENABLE_ROCKE=ON"
        override = "-DTHEROCK_FLAG_HIPKERNELPROVIDER_ENABLE_ROCKE=OFF"
        with mock.patch.dict(
            therock_matrix.LABEL_GATED_CMAKE_OPTIONS,
            self._gated("hip-kernel-provider", override),
            clear=True,
        ):
            project_to_run = therock_matrix.collect_projects_to_run(
                ["dnn-providers/hip-kernel-provider"], ["ci:test-flag"]
            )
        options = self._all_options(project_to_run)
        self.assertIn(override, options)
        self.assertGreater(options.index(override), options.index(default))
        # Pin the whole sequence, not just the relative position of the two
        # options above. Emitted order has to be a pure function of the input:
        # a set()-based dedup would satisfy the assertion above on some
        # PYTHONHASHSEEDs by luck, but almost never reproduces the full order.
        # The defaults are read from the module rather than spelled out, so
        # adding a real option to hip-kernel-provider does not break this test.
        defaults = therock_matrix.project_map["hip-kernel-provider"]["cmake_options"]
        self.assertEqual(options, [*defaults, override, "-DTHEROCK_ENABLE_ALL=OFF"])

    def test_label_gated_options_injected_for_every_known_project(self):
        # Every project reachable from subtree_to_project_map must accept an
        # injection without raising and without silently dropping the option.
        reverse_map = {}
        for subtree, project in therock_matrix.subtree_to_project_map.items():
            reverse_map.setdefault(project, subtree)
        self.assertTrue(reverse_map)
        for project, subtree in sorted(reverse_map.items()):
            with self.subTest(project=project):
                project_to_run = self._run_gated(project, [subtree])
                self.assertIn(
                    "-DTHEROCK_FLAG_TEST=ON", self._all_options(project_to_run)
                )

    def test_validate_label_gated_cmake_options_rejects_unknown_project(self):
        with self.assertRaisesRegex(ValueError, "unknown project"):
            therock_matrix.validate_label_gated_cmake_options(
                self._gated("not-a-project", "-DTHEROCK_FLAG_TEST=ON")
            )

    def test_validate_label_gated_cmake_options_rejects_missing_keys(self):
        with self.assertRaisesRegex(ValueError, "missing required key 'cmake_options'"):
            therock_matrix.validate_label_gated_cmake_options(
                {"ci:test-flag": {"project": "miopen"}}
            )
        with self.assertRaisesRegex(ValueError, "missing required key 'project'"):
            therock_matrix.validate_label_gated_cmake_options(
                {"ci:test-flag": {"cmake_options": ["-DTHEROCK_FLAG_TEST=ON"]}}
            )

    def test_validate_label_gated_cmake_options_rejects_string_options(self):
        with self.assertRaisesRegex(ValueError, "must be a list"):
            therock_matrix.validate_label_gated_cmake_options(
                {
                    "ci:test-flag": {
                        "project": "miopen",
                        "cmake_options": "-DTHEROCK_FLAG_TEST=ON",
                    }
                }
            )


if __name__ == "__main__":
    unittest.main()
