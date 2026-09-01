#!/usr/bin/env python3
"""
Baseline unit tests for pr_merge_sync_patches.py.

These tests document the CURRENT behavior of every testable function,
including known-buggy behavior. Tests that capture a known bug are decorated
with @pytest.mark.xfail(strict=True) and assert the DESIRED behavior.
When a bug is fixed:
  1. Remove the @pytest.mark.xfail decorator.
  2. Run the suite — the test should now pass.
"""

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import pr_merge_sync_patches as sut
from repo_config_model import RepoEntry

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_entry(
    name: str,
    category: str = "projects",
    url: str = None,
    branch: str = "develop",
) -> RepoEntry:
    return RepoEntry(
        name=name,
        url=url or f"ROCm/{name}",
        branch=branch,
        category=category,
        auto_subtree_pull=False,
        auto_subtree_push=True,
        monorepo_source_of_truth=True,
    )


def _git(args, cwd=None):
    """Thin wrapper around git for test repo setup."""
    subprocess.run(
        ["git"] + args,
        cwd=cwd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def make_git_repo(path: Path) -> None:
    """Create a minimal initialised git repository at path."""
    _git(["init", str(path)])
    _git(["config", "user.name", "Test"], cwd=path)
    _git(["config", "user.email", "test@test.com"], cwd=path)


def make_initial_commit(
    repo: Path, filename: str = "README.md", content: str = "hello\n"
) -> str:
    """Write a file, commit it, and return the commit SHA."""
    (repo / filename).write_text(content)
    _git(["add", filename], cwd=repo)
    _git(["commit", "-m", "initial commit"], cwd=repo)
    return (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo).decode().strip()
    )


# ---------------------------------------------------------------------------
# get_subtree_info
# ---------------------------------------------------------------------------


class GetSubtreeInfoTest(unittest.TestCase):

    def setUp(self):
        self.config = [
            make_entry("rocblas"),
            make_entry("hipcub"),
            make_entry("tensile", category="shared"),
        ]

    def test_single_matching_subtree_returned(self):
        result = sut.get_subtree_info(self.config, ["projects/rocblas"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "rocblas")

    def test_multiple_matching_subtrees_returned(self):
        result = sut.get_subtree_info(
            self.config, ["projects/rocblas", "projects/hipcub"]
        )
        names = {e.name for e in result}
        self.assertEqual(names, {"rocblas", "hipcub"})

    def test_shared_category_matched_correctly(self):
        result = sut.get_subtree_info(self.config, ["shared/tensile"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "tensile")

    @pytest.mark.xfail(
        reason="BUG: when no subtrees match the config the caller silently exits 0 — no error is surfaced; after the fix this should raise or trigger a non-zero exit",
        strict=True,
    )
    def test_unmatched_subtree_returns_empty_list(self):
        with self.assertRaises(SystemExit):
            sut.get_subtree_info(self.config, ["projects/doesnotexist"])

    def test_unmatched_subtree_logs_warning(self):
        with self.assertLogs("pr_merge_sync_patches", level="WARNING") as cm:
            sut.get_subtree_info(self.config, ["projects/doesnotexist"])
        self.assertTrue(any("doesnotexist" in msg for msg in cm.output))

    def test_mix_of_matched_and_unmatched(self):
        result = sut.get_subtree_info(
            self.config, ["projects/rocblas", "projects/doesnotexist"]
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "rocblas")

    def test_empty_subtrees_list_returns_empty(self):
        result = sut.get_subtree_info(self.config, [])
        self.assertEqual(result, [])

    def test_empty_config_returns_empty(self):
        result = sut.get_subtree_info([], ["projects/rocblas"])
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# _extract_commit_message_from_patch
# ---------------------------------------------------------------------------


class ExtractCommitMessageTest(unittest.TestCase):
    """
    Tests for _extract_commit_message_from_patch(patch_path: Path) -> str.

    Each test documents current behavior. Tests marked BUG capture
    known bugs in commit-message extraction.

    Fixtures (tests/fixtures/):
      simple.patch          - clean [PATCH] prefix + (#42) PR ref, multi-line body,
                              full diff section. The baseline happy-path input.
      no_pr_ref.patch       - [PATCH] prefix with no (#NN) suffix. Confirms the
                              PR-ref regex doesn't corrupt clean subjects.
      multi_part.patch      - [PATCH 2/5] prefix. Exposes BUG: only the bare
                              "[PATCH]" token is stripped; "[PATCH N/M]" leaks through.
      mime_encoded.patch    - Subject encoded as =?UTF-8?q?...?=. Exposes BUG:
                              no MIME decoding is performed.
      folded_subject.patch  - Long subject folded across two lines (RFC 2822).
                              Exposes BUG: only the first line is captured.
      body_with_dashes.patch - Commit body contains a "---...---" visual separator.
                              Exposes BUG: any line starting with "---" stops
                              extraction, truncating the rest of the body.
    """

    def test_simple_patch_subject_and_body_are_present(self):
        # Checks overall shape: subject is clean, body text is preserved.
        # Individual properties (prefix stripping, PR-ref stripping, diff exclusion)
        # are each covered by their own dedicated tests below.
        result = sut._extract_commit_message_from_patch(FIXTURES / "simple.patch")
        self.assertTrue(result.startswith("Fix the off-by-one error"))
        self.assertIn("loop bounds", result)
        self.assertIn("Signed-off-by", result)

    def test_simple_patch_does_not_include_diff_lines(self):
        result = sut._extract_commit_message_from_patch(FIXTURES / "simple.patch")
        self.assertNotIn("diff --git", result)
        self.assertNotIn("@@", result)

    def test_pr_ref_is_stripped_from_subject(self):
        result = sut._extract_commit_message_from_patch(FIXTURES / "simple.patch")
        self.assertNotIn("(#42)", result)

    def test_patch_prefix_stripped_for_single_patch(self):
        result = sut._extract_commit_message_from_patch(FIXTURES / "simple.patch")
        self.assertFalse(result.startswith("[PATCH]"))

    def test_no_pr_ref_leaves_subject_unchanged(self):
        result = sut._extract_commit_message_from_patch(FIXTURES / "no_pr_ref.patch")
        self.assertTrue(result.startswith("Fix the thing without a PR ref"))

    @pytest.mark.xfail(
        reason="BUG: [PATCH N/M] prefixes are not stripped; only the literal '[PATCH]' token is matched so '[PATCH 2/5]' leaks through",
        strict=True,
    )
    def test_multi_part_patch_prefix_is_NOT_stripped(self):
        result = sut._extract_commit_message_from_patch(FIXTURES / "multi_part.patch")
        self.assertFalse(result.startswith("[PATCH"))

    @pytest.mark.xfail(
        reason="BUG: MIME-encoded words (=?UTF-8?q?...?=) in the Subject line are left as raw encoded text; the current code performs no MIME decoding",
        strict=True,
    )
    def test_mime_encoded_subject_is_NOT_decoded(self):
        result = sut._extract_commit_message_from_patch(FIXTURES / "mime_encoded.patch")
        self.assertNotIn("=?UTF-8?q?", result)
        self.assertIn("mémoire", result)

    @pytest.mark.xfail(
        reason="BUG: when git folds a long Subject across multiple lines (RFC 2822) the current code reads only the first line; the continuation is picked up as body text instead of being joined into the subject",
        strict=True,
    )
    def test_folded_subject_continuation_is_included_verbatim(self):
        result = sut._extract_commit_message_from_patch(
            FIXTURES / "folded_subject.patch"
        )
        self.assertIn("across two lines", result.splitlines()[0])

    def test_body_stops_at_patch_separator(self):
        # The "---" line that separates the commit message from the stat/diff
        # should NOT appear in the extracted message.
        result = sut._extract_commit_message_from_patch(FIXTURES / "simple.patch")
        # The separator itself should not appear.
        self.assertNotIn("\n---\n", result)

    @pytest.mark.xfail(
        reason="BUG: any line starting with '---' is treated as the patch separator and stops extraction, truncating legitimate body content that contains visual separator lines",
        strict=True,
    )
    def test_body_stops_at_first_line_starting_with_dashes(self):
        result = sut._extract_commit_message_from_patch(
            FIXTURES / "body_with_dashes.patch"
        )
        self.assertIn("second paragraph", result)

    def test_patch_with_no_subject_line_returns_empty_string(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
            f.write(
                "From: Someone <s@example.com>\nDate: Mon, 18 Aug 2026 10:00:00 -0400\n\nBody\n---\n"
            )
            tmp = Path(f.name)
        self.addCleanup(tmp.unlink)
        result = sut._extract_commit_message_from_patch(tmp)
        self.assertEqual(result, "")

    def test_returns_stripped_string(self):
        # Leading/trailing whitespace should be stripped from the final result.
        result = sut._extract_commit_message_from_patch(FIXTURES / "simple.patch")
        self.assertEqual(result, result.strip())


# ---------------------------------------------------------------------------
# _format_commit_message
# ---------------------------------------------------------------------------


class FormatCommitMessageTest(unittest.TestCase):

    def test_annotation_prepended_to_original_message(self):
        result = sut._format_commit_message(
            "ROCm/rocm-libraries", 42, "abc1234xyz", "Fix the thing"
        )
        self.assertTrue(result.startswith("[rocm-libraries] ROCm/rocm-libraries#42"))

    def test_sha_truncated_to_7_characters(self):
        result = sut._format_commit_message(
            "ROCm/rocm-libraries", 99, "abc1234xyz9999999", "Fix"
        )
        self.assertIn("commit abc1234", result)
        self.assertNotIn("abc1234x", result)

    def test_original_message_appears_after_blank_line(self):
        result = sut._format_commit_message(
            "ROCm/rocm-libraries", 1, "a" * 40, "My message"
        )
        lines = result.splitlines()
        # annotation line, blank line, original message
        self.assertEqual(lines[1], "")
        self.assertEqual(lines[2], "My message")

    def test_multiline_original_message_preserved(self):
        original = "Subject line\n\nBody paragraph."
        result = sut._format_commit_message(
            "ROCm/rocm-libraries", 1, "a" * 40, original
        )
        self.assertIn("Subject line", result)
        self.assertIn("Body paragraph.", result)

    def test_pr_number_appears_in_annotation(self):
        result = sut._format_commit_message(
            "ROCm/rocm-libraries", 1234, "a" * 40, "msg"
        )
        self.assertIn("#1234", result)


# ---------------------------------------------------------------------------
# resolve_patch_author — PR body parsing logic
# ---------------------------------------------------------------------------


class ResolvePatchAuthorTest(unittest.TestCase):
    """
    Tests for the author-resolution logic inside resolve_patch_author().
    We mock GitHubCLIClient so no real HTTP calls are made.
    """

    def _make_client(
        self, pr_body: str, pr_author: str, user_name: str, user_email: str
    ):
        client = MagicMock()
        client.get_pr_by_number.return_value = {
            "body": pr_body,
            "user": {"login": pr_author},
        }
        client.get_user.return_value = (user_name, user_email)
        return client

    def test_uses_explicitly_listed_original_author(self):
        client = self._make_client(
            pr_body="Originally authored by @jdoe\nOther text",
            pr_author="bot-user",
            user_name="Jane Doe",
            user_email="jane@example.com",
        )
        name, email = sut.resolve_patch_author(client, "ROCm/rocm-libraries", 42)
        client.get_user.assert_called_once_with("jdoe")
        self.assertEqual(name, "Jane Doe")
        self.assertEqual(email, "jane@example.com")

    def test_falls_back_to_pr_author_when_no_explicit_original_author(self):
        client = self._make_client(
            pr_body="Regular PR description",
            pr_author="actual-author",
            user_name="Actual Author",
            user_email="actual@example.com",
        )
        name, email = sut.resolve_patch_author(client, "ROCm/rocm-libraries", 42)
        client.get_user.assert_called_once_with("actual-author")
        self.assertEqual(name, "Actual Author")

    def test_falls_back_to_pr_author_when_body_is_none(self):
        client = self._make_client(
            pr_body=None,
            pr_author="pr-author",
            user_name="PR Author",
            user_email="pr@example.com",
        )
        # Should not raise even with None body.
        name, email = sut.resolve_patch_author(client, "ROCm/rocm-libraries", 42)
        client.get_user.assert_called_once_with("pr-author")

    def test_falls_back_to_pr_author_when_body_is_empty_string(self):
        client = self._make_client(
            pr_body="",
            pr_author="pr-author",
            user_name="PR Author",
            user_email="pr@example.com",
        )
        name, email = sut.resolve_patch_author(client, "ROCm/rocm-libraries", 42)
        client.get_user.assert_called_once_with("pr-author")

    def test_author_regex_accepts_hyphens_in_username(self):
        client = self._make_client(
            pr_body="Originally authored by @john-doe",
            pr_author="bot",
            user_name="John Doe",
            user_email="john@example.com",
        )
        sut.resolve_patch_author(client, "ROCm/rocm-libraries", 42)
        client.get_user.assert_called_once_with("john-doe")

    def test_author_regex_accepts_underscores_in_username(self):
        client = self._make_client(
            pr_body="Originally authored by @john_doe",
            pr_author="bot",
            user_name="John Doe",
            user_email="john@example.com",
        )
        sut.resolve_patch_author(client, "ROCm/rocm-libraries", 42)
        client.get_user.assert_called_once_with("john_doe")

    def test_get_user_name_falls_back_to_username_when_none(self):
        client = MagicMock()
        client.get_pr_by_number.return_value = {
            "body": "",
            "user": {"login": "myuser"},
        }
        # get_user returns (None, email) — name should fall back to the login.
        client.get_user.return_value = (None, "myuser@example.com")
        name, email = sut.resolve_patch_author(client, "ROCm/rocm-libraries", 1)
        self.assertEqual(name, "myuser")


# ---------------------------------------------------------------------------
# _stage_changes — git-level behavior
# ---------------------------------------------------------------------------


class StageChangesTest(unittest.TestCase):
    """
    Tests for _stage_changes(repo_path). These use a real local git repo
    to exercise the actual git invocation rather than mocking _run_git.
    """

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmpdir.name)
        make_git_repo(self.repo)
        make_initial_commit(self.repo)

    def tearDown(self):
        self._tmpdir.cleanup()

    def _staged_files(self):
        out = (
            subprocess.check_output(
                ["git", "diff", "--cached", "--name-only"], cwd=self.repo
            )
            .decode()
            .strip()
        )
        return set(out.splitlines()) if out else set()

    def test_new_file_is_staged(self):
        (self.repo / "new_file.cpp").write_text("int main() {}")
        sut._stage_changes(self.repo)
        self.assertIn("new_file.cpp", self._staged_files())

    def test_modified_tracked_file_is_staged(self):
        (self.repo / "README.md").write_text("modified content")
        sut._stage_changes(self.repo)
        self.assertIn("README.md", self._staged_files())

    @pytest.mark.xfail(
        reason="BUG: 'git add .' respects .gitignore and silently drops new untracked files that match ignore patterns",
        strict=True,
    )
    def test_gitignored_file_is_NOT_staged(self):
        (self.repo / ".gitignore").write_text("*.log\n")
        _git(["add", ".gitignore"], cwd=self.repo)
        _git(["commit", "-m", "add gitignore"], cwd=self.repo)

        (self.repo / "output.log").write_text("some log content")
        sut._stage_changes(self.repo)

        self.assertIn("output.log", self._staged_files())

    def test_previously_force_tracked_file_modifications_are_staged(self):
        # `git add .` DOES stage modifications to already-tracked files even when
        # they match .gitignore -- git only skips *new untracked* files that match.
        # This documents that boundary: modifications to force-tracked files are
        # NOT lost by the current implementation.
        (self.repo / ".gitignore").write_text("generated/\n")
        _git(["add", ".gitignore"], cwd=self.repo)
        _git(["commit", "-m", "add gitignore"], cwd=self.repo)

        gen_dir = self.repo / "generated"
        gen_dir.mkdir()
        (gen_dir / "output.h").write_text("// generated")
        _git(["add", "--force", "generated/output.h"], cwd=self.repo)
        _git(["commit", "-m", "force-add generated file"], cwd=self.repo)

        (gen_dir / "output.h").write_text("// updated generated")
        sut._stage_changes(self.repo)

        # Modifications to already-tracked files ARE staged even under .gitignore.
        self.assertIn("generated/output.h", self._staged_files())

    @pytest.mark.xfail(
        reason="BUG: if a patch introduces a brand-new file whose path matches .gitignore, 'git add .' silently skips it; the sync appears to succeed but the file is never committed to the sub-repo",
        strict=True,
    )
    def test_new_gitignored_file_added_by_patch_is_NOT_staged(self):
        (self.repo / ".gitignore").write_text("generated/\n")
        _git(["add", ".gitignore"], cwd=self.repo)
        _git(["commit", "-m", "add gitignore"], cwd=self.repo)

        # Simulate a patch dropping a brand-new file that matches .gitignore.
        gen_dir = self.repo / "generated"
        gen_dir.mkdir()
        (gen_dir / "new_output.h").write_text("// brand new generated file")

        sut._stage_changes(self.repo)

        self.assertIn("generated/new_output.h", self._staged_files())


# ---------------------------------------------------------------------------
# _apply_patch — git-level behavior
# ---------------------------------------------------------------------------


class ApplyPatchTest(unittest.TestCase):
    """
    Tests for _apply_patch(repo_path, patch_path). Uses real local git repos.
    """

    def setUp(self):
        # Strategy: we need a real .patch file and a real target repo to apply it
        # to. We build both from scratch in a temp directory so these tests have
        # no dependency on the state of the working tree.
        #
        # Layout:
        #   <tmpdir>/repo/       - the "sub-repo" we will apply the patch to
        #   <tmpdir>/patch_repo/ - a scratch repo used only to generate the patch
        #   <tmpdir>/change.patch
        self._tmpdir = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmpdir.name) / "repo"
        self.repo.mkdir()
        make_git_repo(self.repo)
        make_initial_commit(self.repo, "file.txt", "line1\nline2\nline3\n")

        patch_repo = Path(self._tmpdir.name) / "patch_repo"
        patch_repo.mkdir()
        make_git_repo(patch_repo)
        make_initial_commit(patch_repo, "file.txt", "line1\nline2\nline3\n")
        (patch_repo / "file.txt").write_text("line1\nline2 modified\nline3\n")
        _git(["add", "file.txt"], cwd=patch_repo)
        _git(["commit", "-m", "modify line2"], cwd=patch_repo)

        self.patch_file = Path(self._tmpdir.name) / "change.patch"
        subprocess.run(
            ["git", "format-patch", "-1", "HEAD", "--output", str(self.patch_file)],
            cwd=patch_repo,
            check=True,
        )

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_clean_patch_applied_successfully(self):
        sut._apply_patch(self.repo, self.patch_file)
        content = (self.repo / "file.txt").read_text()
        self.assertIn("line2 modified", content)

    @pytest.mark.xfail(
        reason="BUG: applying a patch a second time raises RuntimeError; there is no already-synced guard and no --3way flag so it errors out instead of being a no-op",
        strict=True,
    )
    def test_applying_same_patch_twice_raises(self):
        sut._apply_patch(self.repo, self.patch_file)
        _git(["add", "."], cwd=self.repo)
        _git(["commit", "-m", "apply patch"], cwd=self.repo)

        sut._apply_patch(self.repo, self.patch_file)  # should be a no-op, not raise

    def test_patch_failure_raises_runtime_error(self):
        # Corrupt the working tree so the patch cannot apply cleanly.
        (self.repo / "file.txt").write_text("completely different content\n")
        with self.assertRaises(RuntimeError):
            sut._apply_patch(self.repo, self.patch_file)


# ---------------------------------------------------------------------------
# _push_changes — behavior on failure
# ---------------------------------------------------------------------------


FAKE_SHA = "abc1234def5678901234567890123456789abcdef"


class PushChangesTest(unittest.TestCase):
    """Tests for _push_changes(repo_path, branch).

    _run_git is always mocked. The call sequence for a clean push is:
      1. rev-parse HEAD          -> local SHA
      2. push origin <branch>    -> ""
      3. ls-remote origin <ref>  -> "<sha>\trefs/heads/<branch>"
    """

    def _make_git_mock(self, *, push_side_effect=None):
        """Return a side_effect function that routes by git sub-command."""

        def _git(args, **kwargs):
            sub = args[0]
            if sub == "rev-parse":
                return FAKE_SHA
            if sub == "push":
                if push_side_effect:
                    return push_side_effect(args, **kwargs)
                return ""
            if sub == "ls-remote":
                return f"{FAKE_SHA}\trefs/heads/develop"
            if sub in ("fetch", "rebase"):
                return ""
            raise AssertionError(f"Unexpected git call: {args}")

        return _git

    def test_successful_push_calls_git_push_origin_branch(self):
        calls = []

        def side_effect(args, **kwargs):
            calls.append((args, kwargs))
            return self._make_git_mock()(args, **kwargs)

        with patch.object(sut, "_run_git", side_effect=side_effect):
            sut._push_changes(Path("/tmp/repo"), "develop")

        push_call = next(((args, kw) for args, kw in calls if args[0] == "push"), None)
        self.assertIsNotNone(push_call, "expected a push call but none was made")
        push_args, push_kwargs = push_call
        self.assertEqual(push_args, ["push", "origin", "develop"])
        self.assertEqual(push_kwargs.get("cwd"), Path("/tmp/repo"))

    def test_transient_failure_is_retried_and_succeeds(self):
        push_calls = []

        def push_side_effect(args, **kwargs):
            push_calls.append(args)
            if len(push_calls) == 1:
                raise RuntimeError(
                    "Git command failed: push\nremote: Internal Server Error"
                )
            return ""

        with patch.object(
            sut,
            "_run_git",
            side_effect=self._make_git_mock(push_side_effect=push_side_effect),
        ), patch.object(time, "sleep"):
            sut._push_changes(Path("/tmp/fake"), "develop")  # should not raise

        self.assertEqual(len(push_calls), 2)

    def test_transient_failure_exhausts_retries_and_raises(self):
        def push_side_effect(args, **kwargs):
            raise RuntimeError(
                "Git command failed: push\nremote: Internal Server Error"
            )

        with patch.object(
            sut,
            "_run_git",
            side_effect=self._make_git_mock(push_side_effect=push_side_effect),
        ), patch.object(time, "sleep"):
            with self.assertRaises(RuntimeError):
                sut._push_changes(Path("/tmp/fake"), "develop", max_attempts=1)

    def test_non_fast_forward_triggers_rebase_and_retry(self):
        push_calls = []
        extra_calls = []

        def push_side_effect(args, **kwargs):
            push_calls.append(args)
            if len(push_calls) == 1:
                raise RuntimeError(
                    "Git command failed: push\nerror: failed to push some refs (non-fast-forward)"
                )
            return ""

        def side_effect(args, **kwargs):
            result = self._make_git_mock(push_side_effect=push_side_effect)(
                args, **kwargs
            )
            if args[0] in ("fetch", "rebase"):
                extra_calls.append(args[0])
            return result

        with patch.object(sut, "_run_git", side_effect=side_effect):
            sut._push_changes(Path("/tmp/fake"), "develop")  # should not raise

        self.assertEqual(len(push_calls), 2)
        self.assertEqual(extra_calls, ["fetch", "rebase"])

    def test_non_fast_forward_exhausts_retries_and_raises(self):
        def push_side_effect(args, **kwargs):
            raise RuntimeError(
                "Git command failed: push\nerror: failed to push some refs (non-fast-forward)"
            )

        with patch.object(
            sut,
            "_run_git",
            side_effect=self._make_git_mock(push_side_effect=push_side_effect),
        ):
            with self.assertRaises(RuntimeError):
                sut._push_changes(Path("/tmp/fake"), "develop", max_attempts=1)

    def test_auth_failure_propagates_immediately(self):
        push_calls = []

        def push_side_effect(args, **kwargs):
            push_calls.append(args)
            raise RuntimeError("Git command failed: push\nerror: authentication failed")

        with patch.object(
            sut,
            "_run_git",
            side_effect=self._make_git_mock(push_side_effect=push_side_effect),
        ):
            with self.assertRaises(RuntimeError):
                sut._push_changes(Path("/tmp/fake"), "develop")

        self.assertEqual(len(push_calls), 1)  # no retry on auth failure

    def test_post_push_verification_uses_ls_remote(self):
        calls = []

        def side_effect(args, **kwargs):
            calls.append(args[0])
            return self._make_git_mock()(args, **kwargs)

        with patch.object(sut, "_run_git", side_effect=side_effect):
            sut._push_changes(Path("/tmp/repo"), "develop")

        self.assertIn("ls-remote", calls)

    def test_verification_fails_if_remote_sha_does_not_match(self):
        wrong_sha = "0" * 40

        def side_effect(args, **kwargs):
            if args[0] == "rev-parse":
                return FAKE_SHA
            if args[0] == "push":
                return ""
            if args[0] == "ls-remote":
                return f"{wrong_sha}\trefs/heads/develop"
            return ""

        with patch.object(sut, "_run_git", side_effect=side_effect):
            with self.assertRaises(RuntimeError):
                sut._push_changes(Path("/tmp/repo"), "develop")

    def test_ls_remote_empty_output_raises(self):
        def side_effect(args, **kwargs):
            if args[0] == "rev-parse":
                return FAKE_SHA
            if args[0] == "push":
                return ""
            if args[0] == "ls-remote":
                return ""  # branch not found on remote
            return ""

        with patch.object(sut, "_run_git", side_effect=side_effect):
            with self.assertRaises(RuntimeError):
                sut._push_changes(Path("/tmp/repo"), "develop")


# ---------------------------------------------------------------------------
# apply_patch_to_subrepo — dry-run gate
# ---------------------------------------------------------------------------


class ApplyPatchToSubrepoTest(unittest.TestCase):

    def test_dry_run_skips_all_git_operations(self):
        """In dry-run mode the function must log and return without cloning."""
        with patch.object(sut, "_clone_subrepo") as mock_clone, patch.object(
            sut, "_apply_patch"
        ) as mock_apply, patch.object(sut, "_push_changes") as mock_push:
            sut.apply_patch_to_subrepo(
                entry=make_entry("rocblas"),
                monorepo_url="ROCm/rocm-libraries",
                monorepo_pr=1,
                patch_path=FIXTURES / "simple.patch",
                author_name="Jane Doe",
                author_email="jane@example.com",
                merge_sha="abc1234",
                dry_run=True,
            )
        # The production code clones before checking --dry-run, so _clone_subrepo
        # is called even in dry-run mode. If that changes, update this assertion.
        mock_clone.assert_called_once()
        mock_apply.assert_not_called()
        mock_push.assert_not_called()

    def test_non_dry_run_commits_and_pushes(self):
        """The full pipeline runs commit and push exactly once."""
        with patch.object(sut, "_clone_subrepo"), patch.object(
            sut, "_configure_git_user"
        ), patch.object(sut, "_apply_patch"), patch.object(
            sut, "_stage_changes"
        ), patch.object(
            sut, "_commit_changes"
        ) as mock_commit, patch.object(
            sut, "_set_authenticated_remote"
        ), patch.object(
            sut, "_push_changes"
        ) as mock_push:
            sut.apply_patch_to_subrepo(
                entry=make_entry("rocblas"),
                monorepo_url="ROCm/rocm-libraries",
                monorepo_pr=99,
                patch_path=FIXTURES / "simple.patch",
                author_name="Jane Doe",
                author_email="jane@example.com",
                merge_sha="abc1234",
                dry_run=False,
            )
        mock_commit.assert_called_once()
        mock_push.assert_called_once()


if __name__ == "__main__":
    unittest.main()
