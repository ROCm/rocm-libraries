#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Byte-identity gate for the quant codegen generators, with zero committed fixtures.

A refactor of a code generator is only safe if the bytes it emits do not move:
the ``KERNEL_NAME`` string decides which Old-TE kernel a parity row is compared
against, and the emitted C++ decides what runs.  The obvious way to prove that is
a golden-file test -- and that was tried and reverted here for good reasons: 124
files / 1.4 MB of ``.hpp`` fixtures, which CI's ``clang-format`` scan of ``*.hpp``
then reformats out from under the test.

This is the fixture-free form.  The **reference** generator is materialized on
demand from git (the pre-refactor revision), executed into one temporary
directory, the current generator into another, and the two trees compared with
:mod:`filecmp`.  Nothing is committed, nothing is formatted, and the reference
cannot drift because it *is* the old revision.

Skips cleanly when git history is unavailable (source tarball, shallow export).
"""

import filecmp
import subprocess
import sys
from pathlib import Path

import pytest

_DISP = Path(__file__).resolve().parent.parent
_CODEGEN = _DISP / "codegen"
_REPO_REL = "projects/composablekernel/dispatcher/codegen"

# The revision the current generators must remain byte-identical to.  Bump it only
# together with an intentional, reviewed change to emitted output.
#
# ``fec98c1c0a`` is a *locally created* merge commit, so it does not exist in a
# fresh clone of the PR branch.  That is why the resolution below is ordered and
# why an unresolvable baseline is a FAILURE, not a skip: a gate that skips is not
# a gate, and the previous revision of this file skipped every parametrization
# silently in exactly the environment (clean CI checkout) where it matters most.
#
# Order: explicit override -> the pinned SHA -> the merge base with the upstream
# integration branch, which is reachable in any non-shallow clone.
_BASELINE_REV_PIN = "fec98c1c0a"
_BASELINE_REV_ENV = "CK_CODEGEN_BASELINE_REV"
_BASELINE_FALLBACK_UPSTREAMS = ("origin/develop", "upstream/develop", "develop")

# Generators under the byte-identity gate.  All ten quant generators -- the five
# grouped ones the refactor rewrote and the five non-grouped ones it touched
# through ``codegen_common``, which is shared and therefore just as able to move
# their bytes.
_GENERATORS = [
    "unified_gemm_tensor_quant_codegen.py",
    "unified_gemm_rowcolquant_codegen.py",
    "unified_gemm_aquant_codegen.py",
    "unified_gemm_abquant_codegen.py",
    "unified_gemm_bquant_codegen.py",
    "unified_grouped_gemm_rowcolquant_codegen.py",
    "unified_grouped_gemm_tensorquant_codegen.py",
    "unified_grouped_gemm_aquant_codegen.py",
    "unified_grouped_gemm_abquant_codegen.py",
    "unified_grouped_gemm_bquant_codegen.py",
]


def _repo_root() -> Path:
    path = _DISP
    while path != path.parent:
        if (path / ".git").exists():
            return path
        path = path.parent
    return None


def _rev_exists(root: Path, rev: str) -> bool:
    return subprocess.run(
        ["git", "-C", str(root), "cat-file", "-e", f"{rev}^{{commit}}"],
        capture_output=True, timeout=60,
    ).returncode == 0


def _resolve_baseline_rev(root: Path):
    """(rev, how) for the reference revision, or (None, why-not)."""
    import os
    override = os.environ.get(_BASELINE_REV_ENV)
    if override:
        if not _rev_exists(root, override):
            return None, f"{_BASELINE_REV_ENV}={override} is not a commit"
        return override, f"{_BASELINE_REV_ENV} override"
    if _rev_exists(root, _BASELINE_REV_PIN):
        return _BASELINE_REV_PIN, "pinned revision"
    for upstream in _BASELINE_FALLBACK_UPSTREAMS:
        result = subprocess.run(
            ["git", "-C", str(root), "merge-base", "HEAD", upstream],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip(), f"merge-base with {upstream}"
    return None, (
        f"pinned revision {_BASELINE_REV_PIN} is not in this clone and no "
        f"merge base with {_BASELINE_FALLBACK_UPSTREAMS} could be computed; "
        f"set {_BASELINE_REV_ENV} to a reachable pre-refactor commit"
    )


def _git_show(root: Path, rev: str, rel: str):
    result = subprocess.run(
        ["git", "-C", str(root), "show", f"{rev}:{rel}"],
        capture_output=True, timeout=60,
    )
    return result.stdout if result.returncode == 0 else None


@pytest.fixture(scope="module")
def baseline_rev():
    root = _repo_root()
    if root is None:
        pytest.skip("not a git checkout; cannot materialize the reference generator")
    rev, how = _resolve_baseline_rev(root)
    assert rev is not None, how
    return rev


@pytest.fixture(scope="module")
def baseline_codegen_dir(tmp_path_factory, baseline_rev):
    """A checkout of the whole codegen/ directory at the baseline revision.

    The whole directory, not just the generator: the generators import
    ``codegen_common``, and the reference must run against the reference support
    code as well.
    """
    root = _repo_root()
    listing = subprocess.run(
        ["git", "-C", str(root), "ls-tree", "--name-only", f"{baseline_rev}:{_REPO_REL}"],
        capture_output=True, text=True, timeout=60,
    )
    assert listing.returncode == 0, (
        f"revision {baseline_rev} has no {_REPO_REL}: {listing.stderr.strip()}"
    )

    out = tmp_path_factory.mktemp("baseline_codegen")
    for name in listing.stdout.split():
        if not name.endswith((".py", ".json")):
            continue
        blob = _git_show(root, baseline_rev, f"{_REPO_REL}/{name}")
        if blob is not None:
            (out / name).write_bytes(blob)
    return out


def _generate(codegen_dir: Path, generator: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [sys.executable, str(codegen_dir / generator), "--output-dir", str(out_dir)],
        capture_output=True, text=True, timeout=600, cwd=str(codegen_dir),
    )
    return result


def _tree_diff(left: Path, right: Path):
    """Files that differ, are only on one side, or could not be compared."""
    left_files = {p.name for p in left.rglob("*") if p.is_file()}
    right_files = {p.name for p in right.rglob("*") if p.is_file()}
    problems = []
    for name in sorted(left_files ^ right_files):
        side = "reference only" if name in left_files else "current only"
        problems.append(f"{name}: {side}")
    common = sorted(left_files & right_files)
    match, mismatch, errors = filecmp.cmpfiles(left, right, common, shallow=False)
    problems += [f"{name}: bytes differ" for name in mismatch]
    problems += [f"{name}: could not compare" for name in errors]
    return problems, len(common)


@pytest.mark.parametrize("generator", _GENERATORS)
def test_emitted_headers_are_byte_identical(generator, baseline_codegen_dir,
                                            baseline_rev, tmp_path):
    """The refactored generator must emit the reference bytes, exactly."""
    assert (baseline_codegen_dir / generator).exists(), (
        f"{generator} did not exist at {baseline_rev}; it is new, so either drop "
        "it from the gate or re-anchor the baseline"
    )
    if not (_CODEGEN / generator).exists():
        pytest.fail(f"{generator} has been removed but is still under the gate")

    reference_out = tmp_path / "reference"
    current_out = tmp_path / "current"

    reference = _generate(baseline_codegen_dir, generator, reference_out)
    if reference.returncode != 0:
        pytest.skip(
            f"reference {generator} does not run in this environment: "
            f"{reference.stderr.strip()[-400:]}"
        )

    current = _generate(_CODEGEN, generator, current_out)
    assert current.returncode == 0, (
        f"current {generator} failed:\n{current.stderr[-2000:]}"
    )

    problems, compared = _tree_diff(reference_out, current_out)
    assert compared > 0, f"{generator} emitted no headers on either side"
    assert not problems, (
        f"{generator} no longer emits the {baseline_rev} bytes "
        f"({compared} headers compared):\n" + "\n".join(problems[:40])
    )


@pytest.mark.parametrize("generator", _GENERATORS)
def test_listed_kernel_names_are_byte_identical(generator, baseline_codegen_dir,
                                               baseline_rev, tmp_path):
    """``--list-names`` is the parity contract; it must not move either.

    Cheaper than the full emit, and it is the string a parity harness keys on to
    pick the Old-TE kernel a bridge row is compared against.
    """
    assert (baseline_codegen_dir / generator).exists(), (
        f"{generator} did not exist at {baseline_rev}"
    )

    def names(codegen_dir, label):
        # --output-dir is passed because the pre-migration CLIs require it even
        # for --list-names; nothing is written on that path.
        unused = tmp_path / label
        unused.mkdir(parents=True, exist_ok=True)
        return subprocess.run(
            [sys.executable, str(codegen_dir / generator),
             "--list-names", "--output-dir", str(unused)],
            capture_output=True, text=True, timeout=120, cwd=str(codegen_dir),
        )

    reference = names(baseline_codegen_dir, "ref_names")
    if reference.returncode != 0:
        pytest.skip(f"reference {generator} --list-names unavailable")
    current = names(_CODEGEN, "cur_names")
    assert current.returncode == 0, current.stderr[-2000:]
    assert current.stdout == reference.stdout, (
        f"{generator} --list-names changed vs {baseline_rev}"
    )


def test_baseline_revision_is_reachable():
    """A gate whose reference cannot be materialized is a gate that never runs."""
    root = _repo_root()
    if root is None:
        pytest.skip("not a git checkout")
    rev, how = _resolve_baseline_rev(root)
    assert rev is not None, how


def test_gate_covers_every_quant_generator():
    """Every quant generator on disk must be under the byte-identity gate."""
    on_disk = {
        p.name for p in _CODEGEN.glob("unified_*quant*_codegen.py")
    }
    missing = on_disk - set(_GENERATORS)
    assert not missing, (
        f"quant codegen scripts not under the byte-identity gate: {sorted(missing)}"
    )
