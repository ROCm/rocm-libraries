# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Hermetic transaction self-tests for ``pyproject-mutmut.sh``."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "pyproject-mutmut.sh"
ORIGINAL = (
    b"[build-system]\r\n"
    b'requires = ["setuptools"]\r\n'
    b"\r\n"
    b"[tool.mutmut]\r\n"
    b'source_paths = ["Tensile"]\r\n'
    b"only_mutate = [\r\n"
    b'    "Tensile/Common/Utilities.py",\r\n'
    b"]\r\n"
    b'do_not_mutate = ["Tensile/Tests/*"]\r\n'
    b"pytest_add_cli_args_test_selection = [\r\n"
    b'    "Tensile/Tests/unit/Common",\r\n'
    b"]\r\n"
    b"# unrelated comment must survive set\r\n"
    b"\r\n"
    b"[tool.other]\r\n"
    b'key = "value"\r\n'
)


def git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def make_repo(path: Path, contents: bytes = ORIGINAL) -> Path:
    path.mkdir()
    git(path, "init", "-q")
    git(path, "config", "user.name", "Mutation Test")
    git(path, "config", "user.email", "mutation@example.invalid")
    (path / "pyproject.toml").write_bytes(contents)
    git(path, "add", "pyproject.toml")
    git(path, "commit", "-q", "-m", "fixture")
    return path


def run_tool(
    repo: Path, backup: Path, command: str, *args: str
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "bash",
            str(SCRIPT),
            command,
            "--src",
            str(repo),
            "--backup",
            str(backup),
            *args,
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ.copy(),
        cwd=repo,
    )


def require_success(result: subprocess.CompletedProcess) -> None:
    assert result.returncode == 0, result.stdout + result.stderr


def require_failure(result: subprocess.CompletedProcess, message: str) -> None:
    assert result.returncode != 0, result.stdout + result.stderr
    assert message in result.stderr, result.stdout + result.stderr


def metadata_path(backup: Path) -> Path:
    return Path(str(backup) + ".meta.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transaction_temporaries(*directories: Path):
    return [path for directory in directories for path in directory.glob(".*.tmp.*")]


def test_transaction_records_identity_restores_exact_bytes_and_clears(tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source = repo / "pyproject.toml"
    backup = tmp_path / "state" / "pyproject.toml.bak"
    metadata = metadata_path(backup)
    original_mode = source.stat().st_mode & 0o7777

    require_success(run_tool(repo, backup, "backup"))
    record = json.loads(metadata.read_text())
    expected_hash = hashlib.sha256(ORIGINAL).hexdigest()
    assert backup.read_bytes() == ORIGINAL
    assert record == {
        "backup_file": str(backup.resolve()),
        "generated_mode": original_mode,
        "generated_sha256": expected_hash,
        "head_oid": git(repo, "rev-parse", "HEAD").stdout.strip(),
        "original_mode": original_mode,
        "original_sha256": expected_hash,
        "pending_generated_sha256": None,
        "schema": "pyproject-mutmut-backup/1",
        "source_dir": str(repo.resolve()),
        "source_file": str(source.resolve()),
    }

    repeated = run_tool(repo, backup, "backup")
    require_failure(repeated, "backup transaction already exists")
    assert backup.read_bytes() == ORIGINAL

    first_set = run_tool(
        repo,
        backup,
        "set",
        "--only-mutate",
        "Tensile/LibraryIO.py",
        "--test-selection",
        "Tensile/Tests/unit/characterization/LibraryIO",
    )
    require_success(first_set)
    generated = source.read_bytes()
    assert generated != ORIGINAL
    assert b'    "Tensile/LibraryIO.py",\r\n' in generated
    assert b"# unrelated comment must survive set\r\n" in generated
    record = json.loads(metadata.read_text())
    assert record["original_sha256"] == expected_hash
    assert record["generated_sha256"] == sha256(source)
    assert record["pending_generated_sha256"] is None

    # Repeating the same set is byte-idempotent and remains transaction-owned.
    require_success(
        run_tool(
            repo,
            backup,
            "set",
            "--only-mutate",
            "Tensile/LibraryIO.py",
            "--test-selection",
            "Tensile/Tests/unit/characterization/LibraryIO",
        )
    )
    assert source.read_bytes() == generated

    require_success(run_tool(repo, backup, "restore"))
    assert source.read_bytes() == ORIGINAL
    assert source.stat().st_mode & 0o7777 == original_mode
    assert not backup.exists()
    assert not metadata.exists()

    # A completed restore ends the lifecycle, so another transaction may start.
    require_success(run_tool(repo, backup, "backup"))


@pytest.mark.parametrize("staged", [False, True])
def test_backup_rejects_dirty_source_without_creating_state(
    tmp_path: Path, staged: bool
):
    repo = make_repo(tmp_path / "repo")
    source = repo / "pyproject.toml"
    backup = tmp_path / "state" / "pyproject.toml.bak"
    source.write_bytes(source.read_bytes() + b"# unrelated edit\n")
    if staged:
        git(repo, "add", "pyproject.toml")

    result = run_tool(repo, backup, "backup")
    require_failure(result, "staged or unstaged changes")
    assert not backup.exists()
    assert not metadata_path(backup).exists()


@pytest.mark.parametrize(
    ("enable_flag", "tag_option", "expected_tag", "message"),
    [
        (
            "--assume-unchanged",
            "-v",
            "h",
            "unsafe assume-unchanged index flag",
        ),
        ("--skip-worktree", "-t", "S", "unsafe skip-worktree index flag"),
    ],
)
def test_hidden_index_flags_block_backup_and_assert_clean(
    tmp_path: Path,
    enable_flag: str,
    tag_option: str,
    expected_tag: str,
    message: str,
):
    repo = make_repo(tmp_path / "repo")
    source = repo / "pyproject.toml"
    backup = tmp_path / "state" / "pyproject.toml.bak"
    metadata = metadata_path(backup)
    git(repo, "update-index", enable_flag, "--", "pyproject.toml")
    hidden_edit = source.read_bytes() + b"# hidden from ordinary git diff\n"
    source.write_bytes(hidden_edit)

    backup_result = run_tool(repo, backup, "backup")
    require_failure(backup_result, message)
    clean_result = run_tool(repo, backup, "assert-clean")
    require_failure(clean_result, message)

    assert source.read_bytes() == hidden_edit
    assert not backup.exists()
    assert not metadata.exists()
    tag = git(repo, "ls-files", tag_option, "--", "pyproject.toml").stdout[:1]
    assert tag == expected_tag


def test_backup_rejects_stale_content_and_incomplete_state(tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source = repo / "pyproject.toml"
    backup = tmp_path / "state" / "pyproject.toml.bak"
    require_success(run_tool(repo, backup, "backup"))

    backup.write_bytes(backup.read_bytes() + b"stale")
    result = run_tool(repo, backup, "set", "--only-mutate", "Tensile/A.py")
    require_failure(result, "backup content hash does not match")
    assert source.read_bytes() == ORIGINAL

    metadata_path(backup).unlink()
    result = run_tool(repo, backup, "restore")
    require_failure(result, "incomplete or missing backup transaction")
    assert source.read_bytes() == ORIGINAL


def test_backup_is_bound_to_source_and_head(tmp_path: Path):
    first = make_repo(tmp_path / "first")
    second = make_repo(tmp_path / "second")
    backup = tmp_path / "state" / "pyproject.toml.bak"
    require_success(run_tool(first, backup, "backup"))

    wrong_source = run_tool(
        second, backup, "set", "--only-mutate", "Tensile/LibraryIO.py"
    )
    require_failure(wrong_source, "source_dir mismatch")
    assert (second / "pyproject.toml").read_bytes() == ORIGINAL

    (first / "unrelated.txt").write_text("new commit\n")
    git(first, "add", "unrelated.txt")
    git(first, "commit", "-q", "-m", "advance head")
    wrong_head = run_tool(first, backup, "set", "--only-mutate", "Tensile/LibraryIO.py")
    require_failure(wrong_head, "head_oid mismatch")
    restore = run_tool(first, backup, "restore")
    require_failure(restore, "head_oid mismatch")
    assert (first / "pyproject.toml").read_bytes() == ORIGINAL


def test_post_backup_edit_blocks_set_and_restore(tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source = repo / "pyproject.toml"
    backup = tmp_path / "state" / "pyproject.toml.bak"
    require_success(run_tool(repo, backup, "backup"))
    edited = source.read_bytes() + b"# user edit after backup\n"
    source.write_bytes(edited)

    set_result = run_tool(repo, backup, "set", "--only-mutate", "Tensile/LibraryIO.py")
    require_failure(
        set_result, "source content changed outside this backup transaction"
    )
    restore_result = run_tool(repo, backup, "restore")
    require_failure(
        restore_result, "source content changed outside this backup transaction"
    )
    assert source.read_bytes() == edited


def test_post_set_edit_blocks_restore(tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source = repo / "pyproject.toml"
    backup = tmp_path / "state" / "pyproject.toml.bak"
    require_success(run_tool(repo, backup, "backup"))
    require_success(
        run_tool(repo, backup, "set", "--only-mutate", "Tensile/LibraryIO.py")
    )
    edited = source.read_bytes() + b"# user edit after set\n"
    source.write_bytes(edited)

    result = run_tool(repo, backup, "restore")
    require_failure(result, "source content changed outside this backup transaction")
    assert source.read_bytes() == edited
    assert backup.exists()
    assert metadata_path(backup).exists()


@pytest.mark.parametrize("after_set", [False, True])
def test_mode_only_edit_blocks_set_and_restore(tmp_path: Path, after_set: bool):
    repo = make_repo(tmp_path / "repo")
    source = repo / "pyproject.toml"
    backup = tmp_path / "state" / "pyproject.toml.bak"
    require_success(run_tool(repo, backup, "backup"))
    if after_set:
        require_success(
            run_tool(repo, backup, "set", "--only-mutate", "Tensile/LibraryIO.py")
        )
    owned_contents = source.read_bytes()
    owned_mode = source.stat().st_mode & 0o7777
    changed_mode = owned_mode ^ 0o100
    source.chmod(changed_mode)

    set_result = run_tool(
        repo, backup, "set", "--only-mutate", "Tensile/Configuration.py"
    )
    require_failure(set_result, "source mode changed outside this backup transaction")
    restore_result = run_tool(repo, backup, "restore")
    require_failure(
        restore_result, "source mode changed outside this backup transaction"
    )
    assert source.read_bytes() == owned_contents
    assert source.stat().st_mode & 0o7777 == changed_mode


def test_backup_mode_edit_is_rejected(tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source = repo / "pyproject.toml"
    backup = tmp_path / "state" / "pyproject.toml.bak"
    require_success(run_tool(repo, backup, "backup"))
    backup.chmod((backup.stat().st_mode & 0o7777) ^ 0o100)

    result = run_tool(repo, backup, "restore")
    require_failure(result, "backup mode does not match backup metadata")
    assert source.read_bytes() == ORIGINAL
    assert backup.exists()
    assert metadata_path(backup).exists()


def test_comma_parser_preserves_glob_characters_and_unicode(tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source = repo / "pyproject.toml"
    backup = tmp_path / "state" / "pyproject.toml.bak"
    (repo / "Tensile").mkdir()
    (repo / "Tensile" / "matcheda.py").write_text("would expose glob expansion\n")
    literal_glob = "Tensile/*?[ab].py"
    unicode_path = "Tensile/Δelta.py"

    require_success(run_tool(repo, backup, "backup"))
    require_success(
        run_tool(
            repo,
            backup,
            "set",
            "--only-mutate",
            f"{literal_glob},{unicode_path}",
        )
    )

    generated = source.read_text()
    assert f'    "{literal_glob}",' in generated
    assert f'    "{unicode_path}",' in generated
    assert "matcheda.py" not in generated
    assert "\\u0394" not in generated


def test_metadata_only_interrupted_restore_is_safely_finalized(tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source = repo / "pyproject.toml"
    backup = tmp_path / "state" / "pyproject.toml.bak"
    metadata = metadata_path(backup)
    require_success(run_tool(repo, backup, "backup"))
    require_success(
        run_tool(repo, backup, "set", "--only-mutate", "Tensile/LibraryIO.py")
    )
    record = json.loads(metadata.read_text())

    # State after restore installed and verified the original, unlinked the
    # backup, and was interrupted before unlinking the metadata sidecar.
    source.write_bytes(ORIGINAL)
    source.chmod(record["original_mode"])
    backup.unlink()

    result = run_tool(repo, backup, "restore")
    require_success(result)
    assert "restore already complete" in result.stdout
    assert source.read_bytes() == ORIGINAL
    assert not backup.exists()
    assert not metadata.exists()


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("source", "source_dir mismatch"),
        ("head", "head_oid mismatch"),
        ("content", "not the recorded original file"),
        ("mode", "not the recorded original file"),
    ],
)
def test_metadata_only_restore_rejects_untrusted_state(
    tmp_path: Path, fault: str, message: str
):
    repo = make_repo(tmp_path / "repo")
    source = repo / "pyproject.toml"
    backup = tmp_path / "state" / "pyproject.toml.bak"
    metadata = metadata_path(backup)
    require_success(run_tool(repo, backup, "backup"))
    record = json.loads(metadata.read_text())
    backup.unlink()
    command_repo = repo

    if fault == "source":
        command_repo = make_repo(tmp_path / "other")
    elif fault == "head":
        (repo / "unrelated.txt").write_text("advance HEAD\n")
        git(repo, "add", "unrelated.txt")
        git(repo, "commit", "-q", "-m", "advance head")
    elif fault == "content":
        source.write_bytes(ORIGINAL + b"# unrelated edit\n")
    elif fault == "mode":
        source.chmod(record["original_mode"] ^ 0o100)

    result = run_tool(command_repo, backup, "restore")
    require_failure(result, message)
    assert metadata.exists()


def test_backup_only_interrupted_state_remains_fail_closed(tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source = repo / "pyproject.toml"
    backup = tmp_path / "state" / "pyproject.toml.bak"
    metadata = metadata_path(backup)
    require_success(run_tool(repo, backup, "backup"))
    metadata.unlink()

    result = run_tool(repo, backup, "restore")
    require_failure(result, "incomplete or missing backup transaction")
    assert source.read_bytes() == ORIGINAL
    assert backup.read_bytes() == ORIGINAL
    assert not metadata.exists()


def test_failed_rewrite_is_atomic_and_cleans_temporary_files(tmp_path: Path):
    malformed = ORIGINAL.replace(b"only_mutate", b"different_key")
    repo = make_repo(tmp_path / "repo", malformed)
    source = repo / "pyproject.toml"
    backup = tmp_path / "state" / "pyproject.toml.bak"
    metadata = metadata_path(backup)
    require_success(run_tool(repo, backup, "backup"))
    metadata_before = metadata.read_bytes()

    result = run_tool(repo, backup, "set", "--only-mutate", "Tensile/A.py")
    require_failure(result, "key 'only_mutate' not found")
    assert source.read_bytes() == malformed
    assert backup.read_bytes() == malformed
    assert metadata.read_bytes() == metadata_before
    assert transaction_temporaries(repo, backup.parent) == []


def test_interrupted_set_journal_recovers_both_atomic_outcomes(tmp_path: Path):
    repo = make_repo(tmp_path / "repo")
    source = repo / "pyproject.toml"
    backup = tmp_path / "state" / "pyproject.toml.bak"
    metadata = metadata_path(backup)
    require_success(run_tool(repo, backup, "backup"))
    require_success(
        run_tool(repo, backup, "set", "--only-mutate", "Tensile/LibraryIO.py")
    )

    # Simulate interruption after the journal was written and the source was
    # atomically replaced, but before generated_sha256 was finalized.
    generated_hash = sha256(source)
    record = json.loads(metadata.read_text())
    record["generated_sha256"] = record["original_sha256"]
    record["pending_generated_sha256"] = generated_hash
    metadata.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    require_success(run_tool(repo, backup, "restore"))
    assert source.read_bytes() == ORIGINAL

    # An interrupted restore may have already installed the exact original
    # bytes and mode while leaving the transaction artifacts behind. Retrying
    # restore must recognize that trusted state and finish cleanup.
    require_success(run_tool(repo, backup, "backup"))
    require_success(
        run_tool(repo, backup, "set", "--only-mutate", "Tensile/LibraryIO.py")
    )
    source.write_bytes(ORIGINAL)
    source.chmod(json.loads(metadata.read_text())["original_mode"])
    require_success(run_tool(repo, backup, "restore"))
    assert source.read_bytes() == ORIGINAL
    assert not backup.exists()
    assert not metadata.exists()

    # Simulate interruption before replacement: current content still matches
    # generated_sha256 and the pending candidate can be safely discarded.
    require_success(run_tool(repo, backup, "backup"))
    record = json.loads(metadata.read_text())
    record["pending_generated_sha256"] = "a" * 64
    metadata.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    require_success(
        run_tool(repo, backup, "set", "--only-mutate", "Tensile/LibraryIO.py")
    )
    require_success(run_tool(repo, backup, "restore"))
    assert source.read_bytes() == ORIGINAL
