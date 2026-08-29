"""Unit tests for check_dependency_pins.py.

Run locally:
    python -m unittest tools/lint/test_check_dependency_pins.py -v
    # or
    pytest tools/lint/test_check_dependency_pins.py
"""

import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

THIS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import check_dependency_pins as cdp  # noqa: E402

SHA = "0123456789abcdef0123456789abcdef01234567"


def findings_for(text: str) -> list:
    """Run the CMake rule set over a snippet."""
    return cdp.check_text(text)


class StripComments(unittest.TestCase):
    def test_comment_is_blanked_but_positions_survive(self):
        line = 'set(X "a") # GIT_TAG master'
        stripped = cdp.strip_comments(line)
        self.assertNotIn("GIT_TAG", stripped)
        self.assertEqual(len(stripped), len(line))

    def test_hash_inside_a_quoted_string_is_kept(self):
        stripped = cdp.strip_comments('set(URL "https://h/x#frag")')
        self.assertIn("#frag", stripped)

    def test_line_count_is_preserved(self):
        self.assertEqual(cdp.strip_comments("a\n# c\nb").count("\n"), 2)


class CommentedOutDeclarations(unittest.TestCase):
    def test_declaration_inside_a_comment_block_is_not_a_finding(self):
        text = "\n".join(
            [
                "# FetchContent_Declare(",
                "#   example",
                "#   GIT_REPOSITORY https://example.invalid/e.git",
                "#   GIT_TAG master",
                "# )",
            ]
        )
        self.assertEqual(findings_for(text), [])


class GitTagRules(unittest.TestCase):
    def test_mutable_git_tag_is_a_finding(self):
        findings = findings_for(
            "FetchContent_Declare(dep\n"
            "  GIT_REPOSITORY https://example.invalid/d.git\n"
            "  GIT_TAG master)\n"
        )
        self.assertEqual(len(findings), 1)
        line, name, reason = findings[0]
        self.assertEqual((line, name), (1, "dep"))
        self.assertIn("not a 40-character commit SHA", reason)

    def test_sha_git_tag_passes(self):
        self.assertEqual(
            findings_for(
                "FetchContent_Declare(dep\n"
                "  GIT_REPOSITORY https://example.invalid/d.git\n"
                f"  GIT_TAG {SHA})\n"
            ),
            [],
        )

    def test_missing_git_tag_is_a_finding(self):
        findings = findings_for(
            "FetchContent_Declare(dep GIT_REPOSITORY https://example.invalid/d.git)\n"
        )
        self.assertEqual(len(findings), 1)
        self.assertIn("no GIT_TAG", findings[0][2])

    def test_git_shallow_with_a_sha_is_a_finding(self):
        findings = findings_for(
            "FetchContent_Declare(dep\n"
            "  GIT_REPOSITORY https://example.invalid/d.git\n"
            f"  GIT_TAG {SHA}\n"
            "  GIT_SHALLOW TRUE)\n"
        )
        self.assertEqual(len(findings), 1)
        self.assertIn("GIT_SHALLOW", findings[0][2])

    def test_git_shallow_with_a_mutable_tag_reports_the_tag_first(self):
        findings = findings_for(
            "FetchContent_Declare(dep\n"
            "  GIT_REPOSITORY https://example.invalid/d.git\n"
            "  GIT_TAG v1.2.3\n"
            "  GIT_SHALLOW TRUE)\n"
        )
        self.assertEqual(len(findings), 1)
        self.assertIn("v1.2.3", findings[0][2])


class VariableResolution(unittest.TestCase):
    def test_variable_defaulting_to_a_sha_counts_as_pinned(self):
        self.assertEqual(
            findings_for(
                f'set(DEP_TAG "{SHA}" CACHE STRING "")\n'
                "FetchContent_Declare(dep\n"
                "  GIT_REPOSITORY https://example.invalid/d.git\n"
                "  GIT_TAG ${DEP_TAG})\n"
            ),
            [],
        )

    def test_variable_defaulting_to_a_branch_is_a_finding(self):
        findings = findings_for(
            'set(DEP_TAG "main" CACHE STRING "")\n'
            "FetchContent_Declare(dep\n"
            "  GIT_REPOSITORY https://example.invalid/d.git\n"
            "  GIT_TAG ${DEP_TAG})\n"
        )
        self.assertEqual(len(findings), 1)

    def test_undefined_variable_is_a_finding(self):
        findings = findings_for(
            "FetchContent_Declare(dep\n"
            "  GIT_REPOSITORY https://example.invalid/d.git\n"
            "  GIT_TAG ${UNSET_TAG})\n"
        )
        self.assertEqual(len(findings), 1)

    def test_url_hash_supplied_through_a_variable_counts_as_pinned(self):
        self.assertEqual(
            findings_for(
                "set(DEP_ARGS URL_HASH SHA256=abc)\n"
                "FetchContent_Declare(dep\n"
                "  URL https://example.invalid/d.tar.gz\n"
                "  ${DEP_ARGS})\n"
            ),
            [],
        )


class UrlRules(unittest.TestCase):
    def test_url_without_a_hash_is_a_finding(self):
        findings = findings_for(
            "FetchContent_Declare(dep URL https://example.invalid/d.tar.gz)\n"
        )
        self.assertEqual(len(findings), 1)
        self.assertIn("no URL_HASH", findings[0][2])

    def test_url_hash_passes(self):
        self.assertEqual(
            findings_for(
                "FetchContent_Declare(dep\n"
                "  URL https://example.invalid/d.tar.gz\n"
                "  URL_HASH SHA256=abc)\n"
            ),
            [],
        )

    def test_url_md5_passes(self):
        self.assertEqual(
            findings_for(
                "FetchContent_Declare(dep\n"
                "  URL https://example.invalid/d.tar.gz\n"
                "  URL_MD5 abc)\n"
            ),
            [],
        )

    def test_source_dir_only_declaration_is_not_a_fetch(self):
        self.assertEqual(
            findings_for(
                "FetchContent_Declare(dep SOURCE_DIR ${CMAKE_SOURCE_DIR}/vendor)\n"
            ),
            [],
        )


class FileDownloadRules(unittest.TestCase):
    def test_download_without_expected_hash_is_a_finding(self):
        findings = findings_for(
            "file(DOWNLOAD https://example.invalid/d.tar.gz ${CMAKE_BINARY_DIR}/d.tar.gz)\n"
        )
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0][1], "file(DOWNLOAD)")

    def test_download_with_expected_hash_passes(self):
        self.assertEqual(
            findings_for(
                "file(DOWNLOAD https://example.invalid/d.tar.gz ${CMAKE_BINARY_DIR}/d.tar.gz\n"
                "  EXPECTED_HASH SHA256=abc)\n"
            ),
            [],
        )


class OtherCallForms(unittest.TestCase):
    def test_external_project_add_is_checked(self):
        findings = findings_for(
            "ExternalProject_Add(dep\n"
            "  GIT_REPOSITORY https://example.invalid/d.git\n"
            "  GIT_TAG develop)\n"
        )
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0][1], "dep")

    def test_download_project_proj_keyword_names_the_target(self):
        findings = findings_for(
            "download_project(PROJ dep\n"
            "  GIT_REPOSITORY https://example.invalid/d.git\n"
            "  GIT_TAG master)\n"
        )
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0][1], "dep")

    def test_similarly_named_call_is_not_matched(self):
        self.assertEqual(
            findings_for(
                "my_FetchContent_Declare(dep GIT_REPOSITORY x GIT_TAG main)\n"
            ),
            [],
        )

    def test_nested_parentheses_do_not_truncate_the_block(self):
        findings = findings_for(
            "FetchContent_Declare(dep\n"
            "  GIT_REPOSITORY https://example.invalid/d.git\n"
            f"  GIT_TAG {SHA}\n"
            "  $<$<BOOL:${X}>:GIT_SHALLOW>)\n"
        )
        self.assertEqual(len(findings), 1)
        self.assertIn("GIT_SHALLOW", findings[0][2])


class ReadSources(unittest.TestCase):
    """CI sparse-checkouts leave tracked files absent from disk."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = pathlib.Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.git("init", "-q")
        self.git("config", "user.email", "t@t")
        self.git("config", "user.name", "t")
        (self.repo / "present.cmake").write_text("present\n", encoding="utf-8")
        (self.repo / "absent.cmake").write_text("staged\n", encoding="utf-8")
        self.git("add", "-A")
        self.cwd = pathlib.Path.cwd()
        self.addCleanup(os.chdir, self.cwd)
        os.chdir(self.repo)

    def git(self, *args):
        subprocess.run(["git", "-C", str(self.repo), *args], check=True)

    def test_worktree_file_is_read_from_disk(self):
        (self.repo / "present.cmake").write_text("edited\n", encoding="utf-8")
        sources = dict(cdp.read_sources([pathlib.Path("present.cmake")]))
        self.assertEqual(sources[pathlib.Path("present.cmake")], "edited\n")

    def test_unmaterialized_file_falls_back_to_the_staged_blob(self):
        (self.repo / "absent.cmake").unlink()
        sources = dict(cdp.read_sources([pathlib.Path("absent.cmake")]))
        self.assertEqual(sources[pathlib.Path("absent.cmake")], "staged\n")

    def test_mixed_batch_returns_both(self):
        (self.repo / "absent.cmake").unlink()
        paths = [pathlib.Path("present.cmake"), pathlib.Path("absent.cmake")]
        self.assertEqual(len(cdp.read_sources(paths)), 2)

    def test_untracked_missing_file_is_skipped_not_fatal(self):
        self.assertEqual(cdp.read_sources([pathlib.Path("nope.cmake")]), [])


class Baseline(unittest.TestCase):
    def test_comments_and_blank_lines_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline = pathlib.Path(tmp) / "baseline.txt"
            baseline.write_text("# header\n\na/b.cmake:dep # why\n", encoding="utf-8")
            original = cdp.BASELINE
            cdp.BASELINE = baseline
            try:
                self.assertEqual(cdp.load_baseline(), {"a/b.cmake:dep"})
            finally:
                cdp.BASELINE = original

    def test_missing_baseline_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = cdp.BASELINE
            cdp.BASELINE = pathlib.Path(tmp) / "absent.txt"
            try:
                self.assertEqual(cdp.load_baseline(), set())
            finally:
                cdp.BASELINE = original


class MergeBaseline(unittest.TestCase):
    def _with_baseline(self, text: str) -> pathlib.Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        baseline = pathlib.Path(tmp.name) / "baseline.txt"
        baseline.write_text(text, encoding="utf-8")
        original = cdp.BASELINE
        cdp.BASELINE = baseline
        self.addCleanup(setattr, cdp, "BASELINE", original)
        return baseline

    def test_existing_comments_and_entries_survive(self):
        baseline = self._with_baseline(
            "# why this one cannot be pinned\na/b.cmake:dep\n"
        )
        added = cdp.merge_baseline({"a/b.cmake:dep", "c/d.cmake:other"})
        self.assertEqual(added, ["c/d.cmake:other"])
        text = baseline.read_text(encoding="utf-8")
        self.assertIn("# why this one cannot be pinned", text)
        self.assertIn("a/b.cmake:dep", text)
        self.assertIn("c/d.cmake:other", text)

    def test_nothing_new_leaves_the_file_byte_identical(self):
        baseline = self._with_baseline("# header\na/b.cmake:dep\n")
        before = baseline.read_bytes()
        self.assertEqual(cdp.merge_baseline({"a/b.cmake:dep"}), [])
        self.assertEqual(baseline.read_bytes(), before)

    def test_missing_trailing_newline_does_not_join_lines(self):
        baseline = self._with_baseline("a/b.cmake:dep")
        cdp.merge_baseline({"c/d.cmake:other"})
        self.assertIn(
            "a/b.cmake:dep\nc/d.cmake:other", baseline.read_text(encoding="utf-8")
        )


class CommittedBaseline(unittest.TestCase):
    def test_every_baseline_entry_is_well_formed(self):
        for entry in cdp.load_baseline():
            self.assertRegex(entry, r"^[^:]+:.+$", entry)


if __name__ == "__main__":
    unittest.main()
