#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Integration test for the receipt-4 (PyTorch) deterministic fmha_bwd instances.

Receipt 4 is the preset PyTorch builds CK's memory-efficient SDPA backward
against. It used to carry

    cond &= deterministic == "f"

which dropped every *deterministic* backward instance from codegen, so
`libtorch_hip.so` shipped only the `atomicAdd` `dq` accumulation. That makes
`q.grad` run-to-run non-reproducible, and `torch.use_deterministic_algorithms`
has no instance to dispatch to.

This test drives the *real* `generate.py` entry point -- the same one CMake and
PyTorch's `ck/CMakeLists.txt` invoke -- and asserts four things:

1. The receipt-4 branch no longer filters on `deterministic`, while receipts 3
   and 801 still do. (Source-level; catches an over-broad edit.)
2. Receipt 4 emits deterministic backward instances at all.
3. Every non-deterministic instance has exactly one deterministic twin and vice
   versa -- no missing twins, no orphans. Coverage is complete rather than
   partial, so there is no configuration that builds at `deterministic=false`
   and silently misses at `deterministic=true`.
4. The change is purely additive. The test reconstructs the pre-change
   generator in a temp tree by re-inserting the filter line, runs it, and
   requires the old blob set to be a strict subset of the new one with the
   difference being exactly the deterministic instances. Receipts 3 and 801 are
   required to be byte-identical before and after.

Pure Python: no GPU, no HIP toolchain, no build. Only the codegen step runs.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


# --------------------------------------------------------------------------- #
# Locate the fmha example directory (holds generate.py + codegen/).
# --------------------------------------------------------------------------- #

_HERE = os.path.dirname(os.path.abspath(__file__))
_FMHA_EX = os.path.normpath(
    os.path.join(_HERE, "..", "..", "..", "example", "ck_tile", "01_fmha")
)
_GENERATE_PY = os.path.join(_FMHA_EX, "generate.py")
_CODEGEN_REL = os.path.join("codegen", "ops", "fmha_bwd.py")

# The line receipt 4 used to carry, and the receipts that must still carry it.
_FILTER_LINE = 'cond &= deterministic == "f"'
_RECEIPTS_STILL_FILTERED = {"3", "801"}
_RECEIPT_UNDER_TEST = "4"

# `deterministic` is spelled into every blob name for *both* values, as
# `_deterministic_` and `_ndeterministic_`. The two tokens are disjoint, but
# a substring search for "deterministic" matches both -- which is why this
# test never counts with a bare `grep -c deterministic`.
_DET = "_deterministic_"
_NDET = "_ndeterministic_"


def _receipts_owning_filter_line(src_path):
    """Map each occurrence of the filter line to the receipt that governs it.

    Walks backwards from the line to the enclosing `if/elif receipt == N:`
    rather than matching on line numbers, which drift.
    """
    lines = Path(src_path).read_text().split("\n")
    owners = []
    for i, line in enumerate(lines):
        if line.strip() != _FILTER_LINE:
            continue
        for j in range(i, -1, -1):
            m = re.match(r"\s*(el)?if receipt == (\d+)\s*:", lines[j])
            if m:
                owners.append((i, m.group(2)))
                break
        else:
            owners.append((i, None))
    return lines, owners


def _run_generate(cwd, receipt, out_txt):
    """`generate.py --api bwd --receipt N --list_blobs`.

    Deliberately passes no `--targets`, no `--optdim` and no `--filter`: that
    is exactly how PyTorch invokes it, and nothing downstream prunes the list.
    """
    cmd = [
        sys.executable,
        os.path.join(cwd, "generate.py"),
        "--api", "bwd",
        "--receipt", str(receipt),
        "--list_blobs", str(out_txt),
    ]
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def _blob_names(list_txt):
    """Basenames from a --list_blobs file, deduplicated and sorted."""
    return sorted(
        {
            os.path.basename(ln.strip())
            for ln in Path(list_txt).read_text().splitlines()
            if ln.strip()
        }
    )


class _RevertedTree:
    """A temp copy of the example dir with the receipt-4 filter line restored.

    Reconstructs the pre-change generator so the test can diff old against new
    without a second checkout. On an unpatched tree the re-insertion is a
    no-op, the two blob sets come out equal, and the additive-change test
    fails -- which is the behaviour we want from a regression test.
    """

    def __init__(self):
        self._tmp = None
        self.path = ""

    def __enter__(self):
        self._tmp = tempfile.mkdtemp()
        self.path = os.path.join(self._tmp, "01_fmha")
        shutil.copytree(_FMHA_EX, self.path)
        src = os.path.join(self.path, _CODEGEN_REL)
        lines, owners = _receipts_owning_filter_line(src)

        # Find the receipt-4 branch and the `if not cond:` that closes it,
        # then put the filter line back immediately above that guard.
        anchor = None
        for i, line in enumerate(lines):
            m = re.match(r"\s*(el)?if receipt == (\d+)\s*:", line)
            if m and m.group(2) == _RECEIPT_UNDER_TEST:
                anchor = i
                break
        assert anchor is not None, "no `receipt == 4` branch in fmha_bwd.py"

        guard = None
        for i in range(anchor + 1, len(lines)):
            if lines[i].strip() == "if not cond:":
                guard = i
                break
        assert guard is not None, "no `if not cond:` closing the receipt-4 branch"

        indent = lines[guard][: len(lines[guard]) - len(lines[guard].lstrip())]
        lines.insert(guard, indent + _FILTER_LINE)
        Path(src).write_text("\n".join(lines))
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._tmp:
            shutil.rmtree(self._tmp, ignore_errors=True)


class TestFmhaBwdReceipt4Deterministic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(_GENERATE_PY):
            raise unittest.SkipTest(f"generate.py not found at {_GENERATE_PY}")

    # ---- 1. source-level: only receipt 4 lost the filter ------------------ #

    def test_only_receipt4_dropped_the_deterministic_filter(self):
        src = os.path.join(_FMHA_EX, _CODEGEN_REL)
        _, owners = _receipts_owning_filter_line(src)
        receipts = {r for _, r in owners}
        self.assertNotIn(
            _RECEIPT_UNDER_TEST,
            receipts,
            msg=(
                "receipt 4 still filters out deterministic instances; PyTorch "
                "will ship only the atomicAdd dq accumulation and "
                "use_deterministic_algorithms will have nothing to dispatch to"
            ),
        )
        self.assertEqual(
            receipts,
            _RECEIPTS_STILL_FILTERED,
            msg=(
                "receipts still carrying the deterministic filter changed. "
                f"Expected {sorted(_RECEIPTS_STILL_FILTERED)}, got "
                f"{sorted(r for r in receipts if r)}. Receipts 3 (aiter) and "
                "801 are out of scope for this change and must keep it."
            ),
        )

    # ---- 2. codegen actually emits deterministic instances ---------------- #

    def test_receipt4_emits_deterministic_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "bwd_blob_list.txt")
            res = _run_generate(_FMHA_EX, 4, out)
            self.assertEqual(
                res.returncode,
                0,
                msg=f"generate.py failed\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}",
            )
            names = _blob_names(out)
            det = [n for n in names if _DET in n]
            self.assertTrue(
                det,
                msg=(
                    "receipt 4 emitted zero deterministic bwd instances out of "
                    f"{len(names)} blobs"
                ),
            )

    # ---- 3. every instance has a twin, both directions -------------------- #

    def test_deterministic_instances_are_a_bijection_with_nondeterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "bwd_blob_list.txt")
            self.assertEqual(_run_generate(_FMHA_EX, 4, out).returncode, 0)
            names = _blob_names(out)

            det = {n for n in names if _DET in n}
            ndet = {n for n in names if _NDET in n}
            self.assertFalse(det & ndet, msg="a blob name matched both tokens")

            missing = {n for n in ndet if n.replace(_NDET, _DET) not in det}
            orphans = {n for n in det if n.replace(_DET, _NDET) not in ndet}
            self.assertEqual(
                (len(missing), len(orphans)),
                (0, 0),
                msg=(
                    "deterministic coverage is partial, so some configuration "
                    "builds at deterministic=false and misses at "
                    f"deterministic=true.\nmissing twins: {sorted(missing)[:5]}"
                    f"\norphans: {sorted(orphans)[:5]}"
                ),
            )
            self.assertEqual(len(det), len(ndet))

    # ---- 4. the change is purely additive --------------------------------- #

    def test_change_is_additive_and_leaves_other_receipts_alone(self):
        with _RevertedTree() as old, tempfile.TemporaryDirectory() as tmp:
            for receipt in (4, 3, 801):
                new_txt = os.path.join(tmp, f"new_{receipt}.txt")
                old_txt = os.path.join(tmp, f"old_{receipt}.txt")
                self.assertEqual(_run_generate(_FMHA_EX, receipt, new_txt).returncode, 0)
                self.assertEqual(_run_generate(old.path, receipt, old_txt).returncode, 0)
                new, before = set(_blob_names(new_txt)), set(_blob_names(old_txt))

                self.assertFalse(
                    before - new,
                    msg=(
                        f"receipt {receipt}: {len(before - new)} instance(s) that "
                        "used to be generated no longer are, e.g. "
                        f"{sorted(before - new)[:3]}"
                    ),
                )
                added = new - before
                if receipt == 4:
                    self.assertTrue(added, msg="receipt 4 gained nothing")
                    self.assertTrue(
                        all(_DET in n for n in added),
                        msg=(
                            "receipt 4 gained non-deterministic instances too: "
                            f"{sorted(n for n in added if _DET not in n)[:3]}"
                        ),
                    )
                    self.assertEqual(
                        added,
                        {n for n in new if _DET in n},
                        msg="the added set is not exactly the deterministic set",
                    )
                else:
                    self.assertFalse(
                        added,
                        msg=(
                            f"receipt {receipt} is out of scope but changed: "
                            f"{sorted(added)[:3]}"
                        ),
                    )


if __name__ == "__main__":
    unittest.main()
