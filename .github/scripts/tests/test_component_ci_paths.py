# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Check component CI path patterns and the selection they produce."""

import fnmatch
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))

from component_ci import (
    COMPONENTS,
    EXCLUSION_EXCEPTIONS,
    EXCLUSIONS,
    INFRA_PATHS,
    detect_changed_components,
)


class ComponentCIPathsTest(unittest.TestCase):
    def test_every_pattern_matches_a_tracked_file(self):
        repo_root = Path(__file__).resolve().parents[3]
        patterns = {path for group in COMPONENTS.values() for path in group}
        patterns.update(INFRA_PATHS)
        for table in (EXCLUSIONS, EXCLUSION_EXCEPTIONS):
            patterns.update(path for group in table.values() for path in group)

        # Git can list the complete tree even in component CI's sparse checkout.
        tracked = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()

        # Same matcher as component_ci.matches_paths, so a pattern that matches
        # nothing here can never trigger, exclude, or except anything.
        unmatched = sorted(
            p for p in patterns if not any(fnmatch.fnmatch(f, p) for f in tracked)
        )
        self.assertEqual(
            unmatched, [], f"Component CI patterns matching no file: {unmatched}"
        )


class MiopenProseSelectionTest(unittest.TestCase):
    DOCS = {
        "projects/miopen/README.md",
        "projects/miopen/CHANGELOG.md",
        "projects/miopen/docs/index.rst",
        "projects/miopen/docs/how-to/install.rst",
    }

    def test_docs_only_change_skips_miopen(self):
        self.assertFalse(detect_changed_components(self.DOCS)["miopen"])

    def test_docs_with_other_component_code_skips_miopen(self):
        changed = detect_changed_components(self.DOCS | {"shared/stinkytofu/a.cpp"})
        self.assertFalse(changed["miopen"])
        self.assertTrue(changed["stinkytofu"])

    def test_docs_with_miopen_source_runs_miopen(self):
        changed = self.DOCS | {"projects/miopen/src/convolution.cpp"}
        self.assertTrue(detect_changed_components(changed)["miopen"])

    def test_jit_library_source_runs_jit(self):
        changed = {
            "projects/hipblaslt/library/src/amd_detail/hipblaslt-jit-library.cpp"
        }
        selected = detect_changed_components(changed)
        self.assertTrue(selected["hipblaslt_jit"])
        self.assertFalse(selected["geko"])

    def test_unrelated_file_skips_jit(self):
        changed = {"projects/hipblaslt/utilities/geko/pyproject.toml"}
        selected = detect_changed_components(changed)
        self.assertFalse(selected["hipblaslt_jit"])
        self.assertTrue(selected["geko"])

    def test_package_license_runs_miopen(self):
        changed = {"projects/miopen/LICENSE.md"}
        self.assertTrue(detect_changed_components(changed)["miopen"])

    def test_infra_change_runs_miopen_despite_docs_only_component_change(self):
        changed = self.DOCS | {".github/scripts/component_ci.py"}
        self.assertTrue(all(detect_changed_components(changed).values()))

    def test_exclusions_do_not_apply_to_other_components(self):
        changed = {"projects/hipblaslt/tensilelite/README.md"}
        self.assertTrue(detect_changed_components(changed)["tensilelite_coverage"])

    def test_source_renamed_into_docs_runs_miopen(self):
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("GIT_")
            and k not in ("GITHUB_OUTPUT", "GITHUB_EVENT_NAME")
        }
        with tempfile.TemporaryDirectory() as repo:

            def git(*args):
                subprocess.run(
                    [
                        "git",
                        "-c",
                        "user.name=ci",
                        "-c",
                        "user.email=ci@ci",
                        "-c",
                        "commit.gpgsign=false",
                        *args,
                    ],
                    cwd=repo,
                    env=env,
                    check=True,
                    capture_output=True,
                )

            source = Path(repo, "projects/miopen/src/sample.cpp")
            source.parent.mkdir(parents=True)
            source.write_text("int sample() { return 1; }\n" * 20)
            git("init", "-q")
            git("add", "-A")
            git("commit", "-qm", "base")
            Path(repo, "projects/miopen/docs").mkdir()
            git(
                "mv", "projects/miopen/src/sample.cpp", "projects/miopen/docs/sample.md"
            )
            git("commit", "-qm", "move")

            env["BASE_REF"] = "HEAD^"
            output = subprocess.run(
                [sys.executable, str(SCRIPTS_DIR / "component_ci.py")],
                cwd=repo,
                env=env,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.splitlines()
        self.assertIn("miopen=true", output)


if __name__ == "__main__":
    unittest.main()
