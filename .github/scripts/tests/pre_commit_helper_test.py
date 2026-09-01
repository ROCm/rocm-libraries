from pathlib import Path
import os
import sys
import subprocess
import unittest
from unittest.mock import call, patch, MagicMock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import pre_commit_helper


class PreCommitHelperTest(unittest.TestCase):
    def test_pr_base_ref_handling(self):
        from_ref = "origin/release/therock-foo"
        default_ref = "origin/develop"
        from_ref = pre_commit_helper.select_diff_base(from_ref, default_ref)
        self.assertEqual(from_ref, "origin/release/therock-foo")

        pushed_branch_name = "feature/x"
        base_branch, target_ref = pre_commit_helper.resolve_fetch_target(
            from_ref,
            pushed_branch_name,
        )
        self.assertEqual(base_branch, "release/therock-foo")
        self.assertEqual(target_ref, "refs/remotes/origin/release/therock-foo")

    def test_new_branch_default_fallback(self):
        from_ref = "0" * 40
        default_ref = "origin/develop"

        from_ref = pre_commit_helper.select_diff_base(from_ref, default_ref)
        self.assertEqual(from_ref, default_ref)

    def test_main_normalizes_new_branch_before_pipeline(self):
        null_ref = "0" * 40

        with patch.object(
            sys,
            "argv",
            [
                "pre_commit_helper.py",
                "--from-ref",
                null_ref,
                "--default-ref",
                "origin/develop",
                "--pushed-branch-name",
                "feature/from-elsewhere",
            ],
        ), patch("pre_commit_helper.fetch_diff_base") as mock_fetch, patch(
            "pre_commit_helper.sparse_checkout_changed_projects",
            return_value=["projects/hipdnn/src/foo.cpp"],
        ) as mock_sparse_checkout, patch(
            "pre_commit_helper.set_github_output"
        ) as mock_set_github_output:
            pre_commit_helper.main()

        mock_fetch.assert_called_once_with("origin/develop", "feature/from-elsewhere")
        mock_sparse_checkout.assert_called_once_with("origin/develop")
        mock_set_github_output.assert_any_call({"diff_ref": "origin/develop"})
        mock_set_github_output.assert_any_call(
            {"changed_files": "projects/hipdnn/src/foo.cpp"}
        )

    def test_main_runs_pre_commit_checkout_pipeline(self):
        changed_files = [
            "projects/hipdnn/src/foo.cpp",
            "projects/hipdnn/include/foo.hpp",
            "shared/hipdnn/common.py",
            "README.md",
        ]

        with patch.object(
            sys,
            "argv",
            [
                "pre_commit_helper.py",
                "--from-ref",
                "origin/develop",
                "--default-ref",
                "origin/develop",
                "--pushed-branch-name",
                "feature/pre-commit-helper",
            ],
        ), patch("pre_commit_helper.is_shallow_repo", return_value=False), patch(
            "pre_commit_helper.get_modified_paths", return_value=changed_files
        ) as mock_get_modified_paths, patch(
            "pre_commit_helper.subprocess.run"
        ) as mock_run, patch(
            "pre_commit_helper.set_github_output"
        ) as mock_set_github_output:
            pre_commit_helper.main()

        mock_get_modified_paths.assert_called_once_with("origin/develop...HEAD")
        mock_run.assert_has_calls(
            [
                call(
                    [
                        "git",
                        "fetch",
                        "--no-tags",
                        "--prune",
                        "--no-recurse-submodules",
                        "--filter=blob:none",
                        "origin",
                        "+refs/heads/develop:refs/remotes/origin/develop",
                    ],
                    check=True,
                ),
                call(
                    [
                        "git",
                        "sparse-checkout",
                        "set",
                        "--cone",
                        "projects/hipdnn",
                        "shared/hipdnn",
                    ],
                    check=True,
                ),
            ]
        )
        mock_set_github_output.assert_any_call({"diff_ref": "origin/develop"})
        mock_set_github_output.assert_any_call(
            {"changed_files": "\n".join(sorted(changed_files))}
        )

    @patch("pre_commit_helper.is_shallow_repo", return_value=True)
    @patch("pre_commit_helper.subprocess.run")
    def test_multi_commit_shallow_push(self, mock_run, mock_is_shallow_repo):
        pre_commit_helper.fetch_diff_base("abc123", "develop")

        mock_run.assert_called_once_with(
            [
                "git",
                "fetch",
                "--no-tags",
                "--prune",
                "--no-recurse-submodules",
                "--filter=blob:none",
                "--unshallow",
                "origin",
                "+refs/heads/develop:refs/remotes/origin/develop",
            ],
            check=True,
        )

    @patch("pre_commit_helper.get_modified_paths")
    def test_absent_merge_base_failure_output(self, mock_get_modified_paths):
        from_ref = "origin/develop"
        mock_get_modified_paths.side_effect = subprocess.CalledProcessError(
            returncode=128,
            cmd=["git", "diff", "--name-only", f"{from_ref}...HEAD"],
            stderr="fatal: origin/develop...HEAD: no merge base",
        )

        with self.assertRaises(subprocess.CalledProcessError):
            pre_commit_helper.compute_three_dot_diff(from_ref)

        mock_get_modified_paths.assert_called_once_with(f"{from_ref}...HEAD")


if __name__ == "__main__":
    unittest.main()
