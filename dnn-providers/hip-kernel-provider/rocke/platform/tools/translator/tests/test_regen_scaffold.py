# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""tr.regen writes a first draft and never writes over an existing file."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from trtest import CPP_MODULES, GOLDEN, ROCKE, TR, normalize

SRC = Path("platform/cpp/instances/common")
INC = Path("platform/cpp/include/rocke")
HEADERS = [INC / f"helper_rocke.instances.common.{m}.h" for m in CPP_MODULES]
SUPPORT = [
    SRC / "rocke_py.hpp",
    SRC / "rocke_py_adapters.hpp",
    # valu uses eval_layout, the one adapter that needs the backward headers
    SRC / "rocke_py_adapters_attention_bwd.hpp",
]
TARGETS = [SRC / f"{m}.cpp" for m in CPP_MODULES] + SUPPORT + HEADERS
OWN = [t for t in TARGETS if t not in SUPPORT]  # what --overwrite replaces
EDIT = "// edited by hand\n"


def regen(fixture_tr, *args: str, rocke: Path = ROCKE) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "tr.regen", *CPP_MODULES, "--rocke", str(rocke), *args],
        cwd=TR,
        env=fixture_tr.env(),
        capture_output=True,
        text=True,
    )


def snapshot(root: Path) -> dict[Path, bytes]:
    return {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.fixture(scope="module")
def addr_abi(tmp_path_factory, fixture_tr, clang_format) -> Path:
    """addr's generated header, which valu and counted call into."""
    d = tmp_path_factory.mktemp("abi")
    fixture_tr.run("abi", "_attention_bwd_addr", "--header", str(d))
    return d


@pytest.fixture(scope="module")
def first(tmp_path_factory, fixture_tr, addr_abi) -> Path:
    """One regen into an empty tree."""
    into = tmp_path_factory.mktemp("first")
    p = regen(fixture_tr, "--into", str(into), "--abi-dir", str(addr_abi))
    assert p.returncode == 0, p.stdout + p.stderr
    assert p.stdout.count("wrote   ") == len(TARGETS), p.stdout
    assert "skip" not in p.stdout
    return into


def edited_copy(first: Path, tmp_path: Path) -> Path:
    into = tmp_path / "tree"
    shutil.copytree(first, into)
    for t in TARGETS:
        with open(into / t, "a") as f:
            f.write(EDIT)
    return into


def test_first_regen_writes_everything(first):
    assert sorted(snapshot(first)) == sorted(TARGETS)
    for m in CPP_MODULES:
        text = (first / SRC / f"{m}.cpp").read_text()
        assert normalize(text) == normalize((GOLDEN / f"{m}.cpp.golden").read_text())
        assert "first draft" in text.splitlines()[2]
        assert "hand-maintained from here" in text.splitlines()[2]
    for h in HEADERS:
        assert "hand-maintained from here" in "".join(
            (first / h).read_text().splitlines()[:6]
        )
    assert (first / SRC / "rocke_py.hpp").read_bytes() == (
        TR / "tr" / "prelude" / "rocke_py.hpp"
    ).read_bytes()


def test_second_regen_keeps_hand_edits(first, fixture_tr, addr_abi, tmp_path):
    into = edited_copy(first, tmp_path)
    before = snapshot(into)
    p = regen(fixture_tr, "--into", str(into), "--abi-dir", str(addr_abi))
    assert p.returncode == 0, p.stdout + p.stderr
    for t in TARGETS:
        assert f"skip    {into / t} (exists)" in p.stdout
    assert "wrote" not in p.stdout
    assert snapshot(into) == before


def test_overwrite(first, fixture_tr, addr_abi, tmp_path):
    """--overwrite replaces the modules' sources and headers; a hand fix to a
    shared support header survives it."""
    into = edited_copy(first, tmp_path)
    edited = snapshot(into)
    p = regen(
        fixture_tr, "--into", str(into), "--abi-dir", str(addr_abi), "--overwrite"
    )
    assert p.returncode == 0, p.stdout + p.stderr
    assert p.stdout.count("wrote   ") == len(OWN), p.stdout
    for t in SUPPORT:
        assert f"skip    {into / t} (exists)" in p.stdout
    after = snapshot(into)
    assert {t: after[t] for t in OWN} == {t: snapshot(first)[t] for t in OWN}
    assert {t: after[t] for t in SUPPORT} == {t: edited[t] for t in SUPPORT}


def test_overwrite_support(first, fixture_tr, addr_abi, tmp_path):
    """--overwrite-support replaces the shared support headers, and only them."""
    into = edited_copy(first, tmp_path)
    edited = snapshot(into)
    p = regen(
        fixture_tr,
        "--into",
        str(into),
        "--abi-dir",
        str(addr_abi),
        "--overwrite-support",
    )
    assert p.returncode == 0, p.stdout + p.stderr
    assert p.stdout.count("wrote   ") == len(SUPPORT), p.stdout
    after = snapshot(into)
    assert {t: after[t] for t in SUPPORT} == {t: snapshot(first)[t] for t in SUPPORT}
    assert {t: after[t] for t in OWN} == {t: edited[t] for t in OWN}
    p = regen(
        fixture_tr,
        "--into",
        str(into),
        "--abi-dir",
        str(addr_abi),
        "--overwrite",
        "--overwrite-support",
    )
    assert p.returncode == 0, p.stdout + p.stderr
    assert p.stdout.count("wrote   ") == len(TARGETS), p.stdout
    assert snapshot(into) == snapshot(first)


@pytest.mark.parametrize("overwrite", [False, True])
def test_pretend_writes_nothing(first, fixture_tr, addr_abi, tmp_path, overwrite):
    empty = tmp_path / "empty"
    empty.mkdir()
    into = edited_copy(first, tmp_path)
    before = snapshot(into)
    extra = ["--overwrite"] if overwrite else []
    for tree, writes in (
        (empty, len(TARGETS)),
        (into, len(OWN) if overwrite else 0),
    ):
        p = regen(
            fixture_tr,
            "--into",
            str(tree),
            "--abi-dir",
            str(addr_abi),
            "--pretend",
            *extra,
        )
        assert p.returncode == 0, p.stdout + p.stderr
        assert p.stdout.count("(pretend)") == writes, p.stdout
        assert p.stdout.count("(exists)") == len(TARGETS) - writes, p.stdout
        assert "wrote" not in p.stdout
    assert snapshot(empty) == {}
    assert snapshot(into) == before


def _tree(first: Path, addr_abi: Path, tmp_path: Path, headers) -> Path:
    """A rocKE-shaped tree to read: the live headers plus the given module headers."""
    root = tmp_path / "rocke"
    inc = root / INC
    shutil.copytree(ROCKE / INC, inc)
    (root / "platform" / "python").symlink_to(ROCKE / "platform" / "python")
    style = next(
        p / ".clang-format" for p in ROCKE.parents if (p / ".clang-format").exists()
    )
    shutil.copy(style, root / ".clang-format")
    shutil.copy(addr_abi / "helper_rocke.instances.common._attention_bwd_addr.h", inc)
    for h in headers:
        shutil.copy(first / h, inc)
    return root


def test_existing_header_selects_its_abi(first, fixture_tr, addr_abi, tmp_path):
    """Either origin: one header as written by an earlier draft, one edited by hand."""
    root = _tree(first, addr_abi, tmp_path, HEADERS)
    with open(root / HEADERS[0], "a") as f:
        f.write(EDIT)
    p = regen(fixture_tr, "--pretend", rocke=root)
    for m, h in zip(CPP_MODULES, HEADERS):
        assert (
            f"{m}: header {root / h} exists; using its ABI (--generated-abi to override)"
            in p.stdout
        ), (p.stdout + p.stderr)
    assert "header  " not in p.stdout  # no header generated
    # hand-ABI translation may leave holes; then the message names the way out
    assert p.returncode == 0 or (
        "headers were picked automatically; try --generated-abi" in p.stdout
    ), (p.stdout + p.stderr)
    abi = tmp_path / "abi"
    shutil.copytree(addr_abi, abi)  # valu and counted call into addr's header
    p = regen(
        fixture_tr, "--pretend", "--generated-abi", "--abi-dir", str(abi), rocke=root
    )
    assert p.returncode == 0, p.stdout + p.stderr
    assert "exists; using its ABI" not in p.stdout
    assert p.stdout.count("header  ") == len(CPP_MODULES), p.stdout + p.stderr


def test_rerun_into_the_tree_read(first, fixture_tr, addr_abi, tmp_path):
    """--into defaulted to --rocke: the second run finds its own first draft
    (headers included) and skips every file instead of translating again."""
    root = _tree(first, addr_abi, tmp_path, [])
    abi = tmp_path / "abi"
    shutil.copytree(addr_abi, abi)
    p = regen(fixture_tr, "--abi-dir", str(abi), rocke=root)
    assert p.returncode == 0, p.stdout + p.stderr
    assert p.stdout.count("wrote   ") == len(TARGETS), p.stdout
    with open(root / TARGETS[0], "a") as f:
        f.write(EDIT)
    before = {t: (root / t).read_bytes() for t in TARGETS}
    p = regen(fixture_tr, "--abi-dir", str(abi), rocke=root)
    assert p.returncode == 0, p.stdout + p.stderr
    for t in TARGETS:
        assert f"skip    {root / t} (exists)" in p.stdout
    assert p.stdout.count("(exists)") == len(TARGETS), p.stdout
    assert "wrote" not in p.stdout and "holes" not in p.stdout
    assert {t: (root / t).read_bytes() for t in TARGETS} == before


def test_some_headers_exist_is_an_error(first, fixture_tr, addr_abi, tmp_path):
    root = _tree(first, addr_abi, tmp_path, HEADERS[:1])
    p = regen(fixture_tr, "--pretend", rocke=root)
    assert p.returncode != 0
    assert "pass --hand-abi or --generated-abi" in p.stderr
    abi = tmp_path / "abi"
    shutil.copytree(addr_abi, abi)
    p = regen(
        fixture_tr, "--pretend", "--generated-abi", "--abi-dir", str(abi), rocke=root
    )
    assert p.returncode == 0, p.stdout + p.stderr


def test_no_check_flag(fixture_tr):
    p = regen(fixture_tr, "--check")
    assert p.returncode == 2 and "--check" in p.stderr
