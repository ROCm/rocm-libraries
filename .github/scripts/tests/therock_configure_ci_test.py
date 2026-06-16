from pathlib import Path
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import therock_configure_ci


class ConfigureCITest(unittest.TestCase):
    @patch("subprocess.run")
    def test_pull_request(self, mock_run):
        args = {
            "is_pull_request": True,
        }

        mock_process = MagicMock()
        mock_process.stdout = "projects/rocprim/src/main.cpp\nprojects/hipcub/src/main.cpp\nprojects/rocwmma/src/main.cpp"
        mock_run.return_value = mock_process

        project_to_run, test_type = therock_configure_ci.retrieve_projects(args)
        self.assertIn("rocprim", str(project_to_run))
        self.assertIn("hipcub", str(project_to_run))
        self.assertIn("rocwmma", str(project_to_run))
        self.assertEqual(test_type, "standard")

    @patch("subprocess.run")
    def test_pull_request_empty(self, mock_run):
        args = {"is_pull_request": True, "input_subtrees": ""}

        mock_process = MagicMock()
        mock_process.stdout = ""
        mock_run.return_value = mock_process

        project_to_run, test_type = therock_configure_ci.retrieve_projects(args)
        self.assertEqual(len(project_to_run), 0)

    @patch("subprocess.run")
    def test_workflow_dispatch(self, mock_run):
        args = {
            "is_workflow_dispatch": True,
            "input_projects": "projects/rocprim projects/hipcub",
        }

        mock_process = MagicMock()
        mock_process.stdout = ""
        mock_run.return_value = mock_process

        project_to_run, test_type = therock_configure_ci.retrieve_projects(args)
        self.assertIn("rocprim", str(project_to_run))
        self.assertIn("hipcub", str(project_to_run))
        self.assertEqual(test_type, "standard")

    @patch("subprocess.run")
    def test_workflow_dispatch_bad_input(self, mock_run):
        args = {
            "is_workflow_dispatch": True,
            "input_projects": "projects/rocprim$$projects/hipcub",
        }

        mock_process = MagicMock()
        mock_process.stdout = ""
        mock_run.return_value = mock_process

        project_to_run, test_type = therock_configure_ci.retrieve_projects(args)
        self.assertEqual(len(project_to_run), 0)

    @patch("subprocess.run")
    def test_workflow_dispatch_all(self, mock_run):
        args = {"is_workflow_dispatch": True, "input_projects": "all"}

        mock_process = MagicMock()
        mock_process.stdout = ""
        mock_run.return_value = mock_process

        project_to_run, test_type = therock_configure_ci.retrieve_projects(args)
        self.assertGreaterEqual(len(project_to_run), 3)
        self.assertEqual(test_type, "standard")

    @patch("subprocess.run")
    def test_workflow_dispatch_empty(self, mock_run):
        args = {"is_workflow_dispatch": True, "input_projects": ""}

        mock_process = MagicMock()
        mock_process.stdout = ""
        mock_run.return_value = mock_process

        project_to_run, test_type = therock_configure_ci.retrieve_projects(args)
        self.assertEqual(len(project_to_run), 0)

    @patch("subprocess.run")
    def test_is_push(self, mock_run):
        args = {
            "is_push": True,
        }

        mock_process = MagicMock()
        mock_process.stdout = "projects/rocprim/src/main.cpp"
        mock_run.return_value = mock_process

        project_to_run, test_type = therock_configure_ci.retrieve_projects(args)
        self.assertIn("rocprim", str(project_to_run))
        self.assertEqual(test_type, "standard")

    def test_is_path_workflow_file_related_to_ci(self):
        workflow_path = ".github/workflows/therocktest.yml"
        self.assertTrue(
            therock_configure_ci.is_path_workflow_file_related_to_ci(workflow_path)
        )
        script_path = ".github/scripts/therocktest.py"
        self.assertTrue(
            therock_configure_ci.is_path_workflow_file_related_to_ci(script_path)
        )
        bad_path = ".github/workflows/test.yml"
        self.assertFalse(
            therock_configure_ci.is_path_workflow_file_related_to_ci(bad_path)
        )

    def test_is_path_skippable(self):
        # Skippable paths
        self.assertTrue(therock_configure_ci.is_path_skippable("README.md"))
        self.assertTrue(therock_configure_ci.is_path_skippable("docs/guide.rst"))
        self.assertTrue(
            therock_configure_ci.is_path_skippable("projects/rocprim/.gitignore")
        )
        self.assertTrue(
            therock_configure_ci.is_path_skippable("projects/hipcub/CHANGELOG.md")
        )
        self.assertTrue(
            therock_configure_ci.is_path_skippable(
                "projects/rocwmma/docs/sphinx/requirements.in"
            )
        )
        self.assertTrue(
            therock_configure_ci.is_path_skippable(
                "shared/tensile/docs/sphinx/requirements.in"
            )
        )
        # dnn-providers paths
        self.assertTrue(
            therock_configure_ci.is_path_skippable(
                "dnn-providers/miopen-provider/docs/OperationSupport.md"
            )
        )
        self.assertTrue(
            therock_configure_ci.is_path_skippable(
                "dnn-providers/miopen-provider/.gitignore"
            )
        )
        # AI assistant config files
        self.assertTrue(
            therock_configure_ci.is_path_skippable("projects/hipdnn/.clinerules")
        )
        self.assertTrue(
            therock_configure_ci.is_path_skippable(
                "dnn-providers/miopen-provider/.clinerules"
            )
        )
        self.assertTrue(
            therock_configure_ci.is_path_skippable("projects/rocblas/.cursorrules")
        )
        self.assertTrue(
            therock_configure_ci.is_path_skippable(
                "projects/hipdnn/.cursor/rules/ai-rules.mdc"
            )
        )
        # Non-skippable paths
        self.assertFalse(
            therock_configure_ci.is_path_skippable("projects/rocprim/src/main.cpp")
        )
        self.assertFalse(therock_configure_ci.is_path_skippable("CMakeLists.txt"))

    def test_check_for_non_skippable_path(self):
        # All skippable
        self.assertFalse(
            therock_configure_ci.check_for_non_skippable_path(
                ["README.md", "docs/guide.rst", ".gitignore"]
            )
        )
        # Contains non-skippable
        self.assertTrue(
            therock_configure_ci.check_for_non_skippable_path(
                ["README.md", "projects/rocprim/src/main.cpp"]
            )
        )
        # None and empty
        self.assertFalse(therock_configure_ci.check_for_non_skippable_path(None))
        self.assertFalse(therock_configure_ci.check_for_non_skippable_path([]))

    @patch("therock_configure_ci.get_modified_paths")
    def test_retrieve_projects_skips_ci_for_skippable_paths(self, mock_get_modified):
        mock_get_modified.return_value = [
            "README.md",
            "docs/guide.rst",
            "projects/rocprim/.gitignore",
        ]

        projects, test_type = therock_configure_ci.retrieve_projects(
            {"is_pull_request": True, "base_ref": "HEAD^"}
        )

        self.assertEqual(projects, [])
        self.assertEqual(test_type, "standard")

    @patch("therock_configure_ci.get_modified_paths")
    def test_retrieve_projects_runs_ci_for_non_skippable_paths(self, mock_get_modified):
        mock_get_modified.return_value = ["README.md", "projects/rocprim/src/main.cpp"]

        projects, test_type = therock_configure_ci.retrieve_projects(
            {"is_pull_request": True, "base_ref": "HEAD^"}
        )

        self.assertIn("rocprim", str(projects))
        self.assertEqual(test_type, "standard")

    @patch("therock_configure_ci.get_modified_paths")
    def test_retrieve_projects_runs_ci_for_two_projects(self, mock_get_modified):
        mock_get_modified.return_value = [
            "README.md",
            "projects/rocprim/src/main.cpp",
            "projects/hipcub/src/main.cpp",
        ]

        projects, test_type = therock_configure_ci.retrieve_projects(
            {"is_pull_request": True, "base_ref": "HEAD^"}
        )

        self.assertIn("rocprim", str(projects))
        self.assertIn("hipcub", str(projects))
        self.assertEqual(test_type, "standard")

    @patch("therock_configure_ci.get_modified_paths")
    def test_retrieve_projects_skips_ci_for_ai_config_files(self, mock_get_modified):
        mock_get_modified.return_value = [
            "projects/hipdnn/.clinerules",
            "dnn-providers/miopen-provider/.clinerules",
            "projects/rocblas/.cursorrules",
            "projects/hipdnn/.cursor/rules/ai-rules.mdc",
        ]

        projects, test_type = therock_configure_ci.retrieve_projects(
            {"is_pull_request": True, "base_ref": "HEAD^"}
        )

        self.assertEqual(projects, [])
        self.assertEqual(test_type, "standard")

    @patch("therock_configure_ci.get_modified_paths")
    def test_retrieve_projects_runs_ci_for_workflow_paths(self, mock_get_modified):
        mock_get_modified.return_value = [".github/workflows/therock-ci.yml"]

        projects, test_type = therock_configure_ci.retrieve_projects(
            {"is_pull_request": True, "base_ref": "HEAD^"}
        )

        self.assertGreaterEqual(len(projects), 3)
        self.assertEqual(test_type, "quick")

    def test_parse_test_labels_single_project(self):
        labels = ["test:rocblas"]
        projects, test_type = therock_configure_ci.parse_test_labels(labels)
        self.assertIn("blas", projects)
        self.assertIsNone(test_type)

    def test_parse_test_labels_with_test_type(self):
        labels = ["test:rocblas", "test_type:comprehensive"]
        projects, test_type = therock_configure_ci.parse_test_labels(labels)
        self.assertIn("blas", projects)
        self.assertEqual(test_type, "comprehensive")

    @patch("therock_configure_ci.get_modified_paths")
    def test_retrieve_projects_with_test_label(self, mock_get_modified):
        mock_get_modified.return_value = []

        pr_labels_json = '{"labels": [{"name": "test:rocblas"}]}'
        projects, test_type = therock_configure_ci.retrieve_projects(
            {"is_pull_request": True, "base_ref": "HEAD^", "pr_labels": pr_labels_json}
        )

        self.assertGreater(len(projects), 0)
        self.assertIn("BLAS", str(projects))
        self.assertEqual(test_type, "standard")

    @patch("therock_configure_ci.get_modified_paths")
    def test_retrieve_projects_with_test_label_and_type(self, mock_get_modified):
        mock_get_modified.return_value = []

        pr_labels_json = '{"labels": [{"name": "test:rocblas"}, {"name": "test_type:comprehensive"}]}'
        projects, test_type = therock_configure_ci.retrieve_projects(
            {"is_pull_request": True, "base_ref": "HEAD^", "pr_labels": pr_labels_json}
        )

        self.assertGreater(len(projects), 0)
        self.assertIn("BLAS", str(projects))
        self.assertEqual(test_type, "comprehensive")

    @patch("therock_configure_ci.get_modified_paths")
    def test_retrieve_projects_with_multiple_test_labels(self, mock_get_modified):
        mock_get_modified.return_value = []

        pr_labels_json = '{"labels": [{"name": "test:rocblas"}, {"name": "test:miopen"}, {"name": "test_type:invalid_type"}]}'
        projects, test_type = therock_configure_ci.retrieve_projects(
            {"is_pull_request": True, "base_ref": "HEAD^", "pr_labels": pr_labels_json}
        )

        # Should test both blas and miopen
        self.assertGreaterEqual(len(projects), 1)
        projects_str = str(projects)
        self.assertIn("BLAS", projects_str)
        self.assertIn("MIOPEN", projects_str)
        # Invalid test_type labels are ignored, so test_type falls back to standard
        self.assertEqual(test_type, "standard")

    @patch("therock_configure_ci.get_modified_paths")
    def test_retrieve_projects_label_overrides_skippable_paths(self, mock_get_modified):
        # Only skippable paths modified
        mock_get_modified.return_value = ["README.md", "docs/guide.rst"]

        pr_labels_json = '{"labels": [{"name": "test:rocblas"}]}'
        projects, test_type = therock_configure_ci.retrieve_projects(
            {"is_pull_request": True, "base_ref": "HEAD^", "pr_labels": pr_labels_json}
        )

        # Should run tests even with only skippable paths because of label
        self.assertGreater(len(projects), 0)
        self.assertIn("BLAS", str(projects))

    @patch("therock_configure_ci.get_modified_paths")
    def test_retrieve_projects_label_combines_with_file_changes(
        self, mock_get_modified
    ):
        # File change in rocprim
        mock_get_modified.return_value = ["projects/rocprim/src/main.cpp"]

        pr_labels_json = '{"labels": [{"name": "test:rocblas"}]}'
        projects, test_type = therock_configure_ci.retrieve_projects(
            {"is_pull_request": True, "base_ref": "HEAD^", "pr_labels": pr_labels_json}
        )

        # Should test both rocprim (from files) and rocblas (from label)
        self.assertGreaterEqual(len(projects), 2)
        projects_str = str(projects)
        self.assertIn("PRIM", projects_str)  # rocprim
        self.assertIn("BLAS", projects_str)  # rocblas

    @patch("therock_configure_ci.get_modified_paths")
    def test_retrieve_projects_nightly_ignores_labels(self, mock_get_modified):
        # Test labels only apply to pull requests, not nightly runs
        mock_get_modified.return_value = []

        pr_labels_json = (
            '{"labels": [{"name": "test:rocblas"}, {"name": "test_type:comprehensive"}]}'
        )
        projects, test_type = therock_configure_ci.retrieve_projects(
            {"is_nightly": True, "base_ref": "HEAD^", "pr_labels": pr_labels_json}
        )

        # Nightly should test all projects with comprehensive tests (labels ignored)
        self.assertGreater(len(projects), 0)
        self.assertEqual(test_type, "comprehensive")


if __name__ == "__main__":
    unittest.main()
