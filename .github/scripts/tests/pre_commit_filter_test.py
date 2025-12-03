#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import pre_commit_filter


class TestPreCommitFilter(unittest.TestCase):
    @patch("pre_commit_filter.Path.exists")
    def test_filter_files_opted_in_shared_project(self, mock_exists):
        mock_exists.return_value = True
        changed_files = ["shared/hipdnn/some_file.cpp"]
        filtered, projects = pre_commit_filter.filter_files(changed_files, {"hipdnn"})
        self.assertEqual(filtered, ["shared/hipdnn/some_file.cpp"])
        self.assertEqual(projects, {"hipdnn"})

    @patch("pre_commit_filter.Path.exists")
    def test_filter_files_non_opted_in_project(self, mock_exists):
        mock_exists.return_value = True
        changed_files = ["projects/otherproject/some_file.cpp"]
        filtered, projects = pre_commit_filter.filter_files(changed_files, {"hipdnn"})
        self.assertEqual(filtered, [])
        self.assertEqual(projects, set())

    @patch("pre_commit_filter.Path.exists")
    def test_filter_files_non_opted_in_shared_project(self, mock_exists):
        mock_exists.return_value = True
        changed_files = ["shared/otherproject/some_file.cpp"]
        filtered, projects = pre_commit_filter.filter_files(changed_files, {"hipdnn"})
        self.assertEqual(filtered, [])
        self.assertEqual(projects, set())

    @patch("pre_commit_filter.Path.exists")
    def test_filter_files_outside_projects(self, mock_exists):
        mock_exists.return_value = True
        changed_files = [".github/workflows/pre-commit.yml", "README.md"]
        filtered, projects = pre_commit_filter.filter_files(changed_files, {"hipdnn"})
        self.assertEqual(filtered, [".github/workflows/pre-commit.yml", "README.md"])
        self.assertEqual(projects, set())

    @patch("pre_commit_filter.Path.exists")
    def test_filter_files_mixed(self, mock_exists):
        mock_exists.return_value = True
        changed_files = [
            "projects/hipdnn/file1.cpp",
            "shared/hipdnn/file2.cpp",
            "projects/otherproject/file3.cpp",
            "shared/otherproject/file4.cpp",
            "README.md",
        ]
        filtered, projects = pre_commit_filter.filter_files(changed_files, {"hipdnn"})
        expected_filtered = [
            "projects/hipdnn/file1.cpp",
            "shared/hipdnn/file2.cpp",
            "README.md",
        ]
        self.assertEqual(sorted(filtered), sorted(expected_filtered))
        self.assertEqual(projects, {"hipdnn"})

    @patch("pre_commit_filter.Path.exists")
    def test_filter_files_with_spaces(self, mock_exists):
        mock_exists.return_value = True
        changed_files = ["projects/hipdnn/file with spaces.cpp", "file with spaces.md"]
        filtered, projects = pre_commit_filter.filter_files(changed_files, {"hipdnn"})
        self.assertEqual(
            filtered, ["file with spaces.md", "projects/hipdnn/file with spaces.cpp"]
        )
        self.assertEqual(projects, {"hipdnn"})

    def test_parse_arguments_with_spaces(self):
        # Simulate command line arguments: script.py --file-list "file 1" "file 2"
        test_args = ["pre_commit_filter.py", "--file-list", "file 1", "file 2"]
        with patch.object(sys, "argv", test_args):
            args = pre_commit_filter.parse_arguments()
            self.assertEqual(args.file_list, ["file 1", "file 2"])

    @patch("pre_commit_filter.Path.exists")
    def test_filter_files_deleted_file(self, mock_exists):
        mock_exists.return_value = False
        changed_files = ["projects/hipdnn/deleted_file.cpp"]
        filtered, projects = pre_commit_filter.filter_files(changed_files, {"hipdnn"})
        self.assertEqual(filtered, [])
        self.assertEqual(projects, set())

    @patch("pre_commit_filter.subprocess.run")
    def test_get_changed_files(self, mock_run):
        mock_result = MagicMock()
        mock_result.stdout = "file1.txt\0file2.txt\0"
        mock_run.return_value = mock_result

        files = pre_commit_filter.get_changed_files("base", "head")
        self.assertEqual(files, ["file1.txt", "file2.txt"])
        mock_run.assert_called_with(
            ["git", "diff", "-z", "--name-only", "base...head"],
            capture_output=True,
            text=True,
            check=True,
        )

    @patch("pre_commit_filter.os.environ", {})
    @patch("builtins.open", new_callable=unittest.mock.mock_open)
    def test_write_github_output(self, mock_file):
        # Mock GITHUB_OUTPUT environment variable
        with patch.dict(os.environ, {"GITHUB_OUTPUT": "output.txt"}):
            pre_commit_filter.write_github_output("key", "value")
            mock_file.assert_called_with("output.txt", "a")
            mock_file().write.assert_called_with("key=value\n")

    @patch("pre_commit_filter.os.environ", {})
    @patch("builtins.open", new_callable=unittest.mock.mock_open)
    def test_write_github_output_multiline(self, mock_file):
        with patch.dict(os.environ, {"GITHUB_OUTPUT": "output.txt"}):
            pre_commit_filter.write_github_output("key", "line1\nline2")
            mock_file.assert_called_with("output.txt", "a")
            mock_file().write.assert_called_with("key<<EOF\nline1\nline2\nEOF\n")

    @patch("pre_commit_filter.write_github_output")
    @patch("pre_commit_filter.Path.exists")
    def test_main_flow(self, mock_exists, mock_write_output):
        mock_exists.return_value = True

        # Simulate command line arguments
        test_args = [
            "pre_commit_filter.py",
            "hipdnn",
            "--file-list",
            "projects/hipdnn/file1.cpp",
            "projects/other/file2.cpp",
            "shared/other/file3.cpp",
            "shared/file4.cpp",
            "README.md",
        ]

        with patch.object(sys, "argv", test_args):
            pre_commit_filter.main()

        # Verify expected calls to write_github_output
        # 1. should_run = true
        # 2. files = list of files
        # 3. hipdnn_changed = true

        # Check calls
        calls = mock_write_output.call_args_list

        # We expect at least 3 calls
        self.assertTrue(len(calls) >= 3)

        # Verify specific calls
        mock_write_output.assert_any_call("should_run", "true")
        mock_write_output.assert_any_call("hipdnn_changed", "true")

        # Verify files output
        # Expected filtered files: projects/hipdnn/file1.cpp, README.md (sorted)
        expected_files = "README.md\nprojects/hipdnn/file1.cpp"
        mock_write_output.assert_any_call("files", expected_files)


if __name__ == "__main__":
    unittest.main()
