# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Run with the pinned TheRock build_tools/github_actions on PYTHONPATH."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import get_changed_projects as selection

CONFIG = Path(__file__).resolve().parents[2] / "repos-config.json"
SOURCE = "projects/hipblaslt/clients/tests/src/hipblaslt_test.cpp"
TENSILELITE = "projects/hipblaslt/tensilelite/tensilelite/KernelWriterAssembly.py"
ASAN_WORKFLOW = ".github/workflows/hipblaslt-asan-ci.yml"
THEROCK_WORKFLOW = ".github/workflows/therock-multi-arch-ci.yml"


class ChangedProjectsTest(unittest.TestCase):
    def assert_selection(self, paths, projects, run_all=False, config=CONFIG):
        result = selection.get_changed_projects(paths, config)
        self.assertEqual(result.changed_projects, ",".join(sorted(projects)))
        self.assertEqual(result.run_all_tests, run_all)
        self.assertEqual(result.test_type, "quick" if run_all else "standard")

    def test_irrelevant_files_do_not_broaden_source_selection(self):
        # An independent workflow change must not select unrelated libraries.
        ignored = [
            ASAN_WORKFLOW,
            ".github/workflows/pre-commit.yml",
            "projects/rocfft/docs/design.rst",
            ".github/dependabot.yml",  # Comes from TheRock's base skip policy.
        ]
        for extra in [set(), *({path} for path in ignored), set(ignored)]:
            with self.subTest(extra=extra):
                self.assert_selection(
                    {SOURCE, TENSILELITE} | extra,
                    ["projects/hipblaslt", "projects/hipblaslt/tensilelite"],
                )

    def test_nested_source_root_keeps_its_identity(self):
        self.assert_selection({TENSILELITE}, ["projects/hipblaslt/tensilelite"])

    def test_only_irrelevant_files_select_no_projects(self):
        for paths in [
            set(),
            {ASAN_WORKFLOW},
            {"README.md", "projects/rocfft/docs/design.rst"},
        ]:
            with self.subTest(paths=paths):
                self.assert_selection(paths, [])

    def test_unknown_and_shared_changes_retain_broad_coverage(self):
        for extra in [
            "unknown-build.cfg",
            "test/therock/test_hipblaslt.py",
            "CMakeLists.txt",
            "shared/ctest/CTestCustom.cmake",
            ".github/repos-config.json",
            ".github/scripts/get_changed_projects.py",
            ".github/scripts/component_ci.py",
        ]:
            with self.subTest(extra=extra):
                self.assert_selection({SOURCE, ASAN_WORKFLOW, extra}, [], run_all=True)
        self.assert_selection({THEROCK_WORKFLOW, "unknown-build.cfg"}, [], run_all=True)

    def test_shared_workflow_retains_configured_and_native_projects(self):
        config = json.loads(CONFIG.read_text())
        configured = {
            f"{entry['category']}/{entry['name']}" for entry in config["repositories"]
        }
        self.assert_selection({THEROCK_WORKFLOW, ASAN_WORKFLOW}, configured)
        self.assert_selection(
            {THEROCK_WORKFLOW, "projects/hipccl/CMakeLists.txt"},
            configured | {"projects/hipccl"},
        )

    def test_platform_only_skips_do_not_remove_projects(self):
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "repos-config.json"
            config.write_text(CONFIG.read_text())
            config.with_name("skip-ci-config.toml").write_text(
                "[skip_ci]\ncommon = []\n"
                'linux = ["projects/hipblaslt/*"]\n'
                'windows = ["projects/rocfft/*"]\n'
            )
            self.assert_selection(
                {SOURCE, "projects/rocfft/CMakeLists.txt"},
                ["projects/hipblaslt", "projects/rocfft"],
                config=config,
            )

    def test_missing_or_invalid_configuration_retains_broad_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "repos-config.json"
            config.write_text(CONFIG.read_text())
            policy = config.with_name("skip-ci-config.toml")
            self.assert_selection({SOURCE}, [], run_all=True, config=config)
            for content in ["not valid TOML", '[skip_ci]\ncommon = "*"']:
                policy.write_text(content)
                self.assert_selection({SOURCE}, [], run_all=True, config=config)
            policy.write_text("[skip_ci]\ncommon = []")
            config.unlink()
            self.assert_selection({SOURCE}, [], run_all=True, config=config)

    def test_workflow_inputs_and_outputs(self):
        for event, base, head, api_paths, expected in [
            (
                "pull_request",
                "base",
                "head",
                [SOURCE, ASAN_WORKFLOW],
                "projects/hipblaslt",
            ),
            ("push", "base", "head", [TENSILELITE], "projects/hipblaslt/tensilelite"),
            ("pull_request", "base", "head", None, None),
            ("pull_request", "", "head", [], None),
            ("push", "base", "", [], None),
            ("schedule", "base", "head", [], None),
            ("workflow_dispatch", "base", "head", [], None),
        ]:
            with self.subTest(event=event, base=base, head=head, paths=api_paths):
                with patch.object(
                    selection, "get_modified_paths_via_api", return_value=api_paths
                ) as get_paths:
                    with patch.object(selection.therock, "set_github_output") as output:
                        selection.main(
                            [
                                "--event-name",
                                event,
                                "--github-repo",
                                "ROCm/rocm-libraries",
                                "--base-sha",
                                base,
                                "--head-sha",
                                head,
                                "--config-path",
                                str(CONFIG),
                            ]
                        )
                if event in ("pull_request", "push") and base and head:
                    get_paths.assert_called_once_with("ROCm/rocm-libraries", base, head)
                else:
                    get_paths.assert_not_called()
                output.assert_called_once_with(
                    {
                        "changed_projects": expected or "",
                        "run_all_tests": "true" if expected is None else "false",
                        "test_type": "quick" if expected is None else "standard",
                    }
                )


if __name__ == "__main__":
    unittest.main()
