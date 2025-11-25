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
            "input_subtrees": "projects/rocprim\nprojects/hipcub"
        }
        
        mock_process = MagicMock()
        mock_process.stdout = ""
        mock_run.return_value = mock_process

        project_to_run = therock_configure_ci.retrieve_projects(args)
        self.assertEqual(len(project_to_run), 1)

    @patch("subprocess.run")
    def test_pull_request_empty(self, mock_run):
        args = {
            "is_pull_request": True,
            "input_subtrees": ""
        }
        
        mock_process = MagicMock()
        mock_process.stdout = ""
        mock_run.return_value = mock_process

        project_to_run = therock_configure_ci.retrieve_projects(args)
        self.assertEqual(len(project_to_run), 0)

    def test_workflow_dispatch(self):
        args = {
            "is_workflow_dispatch": True,
            "input_projects": "projects/rocprim projects/hipcub"
        }

        project_to_run = therock_configure_ci.retrieve_projects(args)
        self.assertEqual(len(project_to_run), 1)

    def test_workflow_dispatch_bad_input(self):
        args = {
            "is_workflow_dispatch": True,
            "input_projects": "projects/rocprim$$projects/hipcub"
        }

        project_to_run = therock_configure_ci.retrieve_projects(args)
        self.assertEqual(len(project_to_run), 0)

    def test_workflow_dispatch_all(self):
        args = {
            "is_workflow_dispatch": True,
            "input_projects": "all"
        }

        project_to_run = therock_configure_ci.retrieve_projects(args)
        self.assertGreaterEqual(len(project_to_run), 1)

    def test_workflow_dispatch_empty(self):
        args = {
            "is_workflow_dispatch": True,
            "input_projects": ""
        }

        project_to_run = therock_configure_ci.retrieve_projects(args)
        self.assertEqual(len(project_to_run), 0)

    @patch("subprocess.run")
    def test_is_push(self, mock_run):
        args = {
            "is_push": True,
        }
        
        mock_process = MagicMock()
        mock_process.stdout = ""
        mock_run.return_value = mock_process

        project_to_run = therock_configure_ci.retrieve_projects(args)
        self.assertGreaterEqual(len(project_to_run), 1)

    def test_is_path_workflow_file_related_to_ci(self):
        workflow_path = ".github/workflows/therocktest.yml"
        self.assertTrue(therock_configure_ci.is_path_workflow_file_related_to_ci(workflow_path))
        script_path = ".github/scripts/therocktest.py"
        self.assertTrue(therock_configure_ci.is_path_workflow_file_related_to_ci(script_path))
        bad_path = ".github/workflows/test.yml"
        self.assertFalse(therock_configure_ci.is_path_workflow_file_related_to_ci(bad_path))

if __name__ == "__main__":
    unittest.main()
