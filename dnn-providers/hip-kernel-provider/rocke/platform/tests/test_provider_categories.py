# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""``shared_category_parser`` gives the checked-out parser in a full checkout and
the git index's copy in a sparse checkout that leaves ``shared/ctest`` out; it
gives None for a directory that is no git checkout and for a checkout whose
index lacks the parser, and raises when the index holds the parser but git
cannot read it."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from rocke.tests.provider_categories import shared_category_parser

GIT = shutil.which("git")

pytestmark = pytest.mark.skipif(GIT is None, reason="needs git on PATH")

PARSER = Path("shared/ctest/parse_ctest_categories.py")
PARSER_SOURCE = b"print('the committed parser')\n"
# A file outside shared/, so a sparse checkout of its directory leaves the parser out.
PROVIDER_FILE = Path("provider/provider.txt")
# Keep the developer's git configuration (signing, hooks, templates) out of the
# repositories these tests make.
ISOLATED_GIT_ENVIRONMENT = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
}


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        [GIT, *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        env=ISOLATED_GIT_ENVIRONMENT,
    ).stdout.strip()


def _committed_tree(root: Path, files: dict[Path, bytes]) -> Path:
    """A git work tree at ``root`` with ``files`` committed."""
    for relative, content in files.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_bytes(content)
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=rocke",
        "-c",
        "user.email=rocke@example.com",
        "commit",
        "-q",
        "-m",
        "tree",
    )
    return root


@pytest.fixture
def monorepo(tmp_path: Path) -> Path:
    """A git work tree holding the parser and one provider file, committed."""
    return _committed_tree(
        tmp_path / "monorepo",
        {PARSER: PARSER_SOURCE, PROVIDER_FILE: b"x\n"},
    )


@pytest.fixture
def scratch(tmp_path: Path) -> Path:
    directory = tmp_path / "scratch"
    directory.mkdir()
    return directory


def test_a_full_checkout_gives_the_checked_out_parser(monorepo, scratch):
    assert shared_category_parser(monorepo, scratch) == monorepo / PARSER
    assert not any(scratch.iterdir())


def test_a_sparse_checkout_without_the_parser_gives_its_indexed_copy(monorepo, scratch):
    _git(monorepo, "sparse-checkout", "set", "--cone", PROVIDER_FILE.parent.as_posix())
    assert not (monorepo / PARSER).exists()
    parser = shared_category_parser(monorepo, scratch)
    assert parser == scratch / PARSER.name
    assert parser.read_bytes() == PARSER_SOURCE


def test_an_indexed_parser_git_cannot_read_raises_with_git_s_message(monorepo, scratch):
    # As in a blobless clone whose on-demand fetch fails: the index names the
    # blob, and the object store lacks it.
    _git(monorepo, "sparse-checkout", "set", "--cone", PROVIDER_FILE.parent.as_posix())
    blob = _git(monorepo, "rev-parse", f":{PARSER.as_posix()}")
    (monorepo / ".git" / "objects" / blob[:2] / blob[2:]).unlink()
    with pytest.raises(
        RuntimeError,
        match=r"reading the shared category parser from the git index failed - "
        r"path='shared/ctest/parse_ctest_categories\.py', git said '.+'",
    ):
        shared_category_parser(monorepo, scratch)


def test_a_checkout_whose_index_lacks_the_parser_gives_none(tmp_path, scratch):
    checkout = _committed_tree(tmp_path / "other", {PROVIDER_FILE: b"x\n"})
    assert shared_category_parser(checkout, scratch) is None


def test_a_directory_that_is_no_git_checkout_gives_none(tmp_path, scratch):
    outside = tmp_path / "outside"
    outside.mkdir()
    assert shared_category_parser(outside, scratch) is None
