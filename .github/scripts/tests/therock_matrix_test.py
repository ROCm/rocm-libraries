import copy
from pathlib import Path
import os
import sys
import unittest

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


if __name__ == "__main__":
    unittest.main()
