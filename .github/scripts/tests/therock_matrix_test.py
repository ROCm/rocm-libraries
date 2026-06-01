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
        self.assertEqual(len(project_to_run), 1)

    def test_collect_projects_to_run_sets_hipdnn_clang_tidy_project(self):
        subtrees = ["projects/hipdnn"]

        project_to_run = therock_matrix.collect_projects_to_run(subtrees)

        self.assertEqual(len(project_to_run), 1)
        self.assertEqual(project_to_run[0]["clang_tidy_projects"], "hipdnn")

    def test_collect_projects_to_run_sets_stinkytofu_clang_tidy_project(self):
        subtrees = ["shared/stinkytofu"]

        project_to_run = therock_matrix.collect_projects_to_run(subtrees)

        self.assertEqual(len(project_to_run), 1)
        self.assertEqual(project_to_run[0]["clang_tidy_projects"], "stinkytofu")

    def test_collect_projects_to_run_carries_clang_tidy_projects_to_parent(self):
        subtrees = ["projects/miopen", "shared/stinkytofu"]

        project_to_run = therock_matrix.collect_projects_to_run(subtrees)

        self.assertEqual(len(project_to_run), 1)
        self.assertEqual(project_to_run[0]["clang_tidy_projects"], "stinkytofu")


if __name__ == "__main__":
    unittest.main()
