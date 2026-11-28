# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""tools/dg_summary.py: the differential golden's checks summarized per flavor. The checks kept only the log's last
lines, the last flavor's (a last "NEW (unblessed): 15" can be llvm24's alone beside llvm20-23's hundreds).
"""

import pathlib
import subprocess
import sys
import tempfile
import unittest

TOOL = pathlib.Path(__file__).resolve().parents[1] / "tools" / "dg_summary.py"


def block(
    where: str,
    blessed: int,
    changed: int,
    gone: int,
    new: int,
    verdict: str = "GOLDEN OK",
) -> str:
    return (
        f"mode=ll  flavor=x  families=115  jobs=8  python-only\n\n=== GOLDEN CHECK ({where}) ===\n"
        f"  blessed configs : {blessed}\n  UNBLESSED CHANGE: {changed}\n  MISSING (gone)  : {gone}\n"
        f"  NEW (unblessed) : {new}\n\n{verdict}\n"
    )


def summary(text: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
        f.write(text)
    try:
        return subprocess.run(
            [sys.executable, "-B", str(TOOL), f.name],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    finally:
        pathlib.Path(f.name).unlink()


class TestDgSummary(unittest.TestCase):
    def test_every_flavor_is_shown(self):
        text = "".join(
            block(f"ll[{fl}]", 4235, 0, 0, n)
            for fl, n in (
                ("llvm20", 210),
                ("llvm22", 210),
                ("llvm23", 275),
                ("llvm24", 15),
            )
        )
        self.assertEqual(
            summary(text),
            "ll[llvm20]: changed 0, gone 0, new 210; ll[llvm22]: changed 0, gone 0, new 210; "
            "ll[llvm23]: changed 0, gone 0, new 275; ll[llvm24]: changed 0, gone 0, new 15",
        )

    def test_a_mismatch_or_a_missing_golden_fails(self):
        text = block(
            "ll[llvm22]",
            4235,
            3,
            0,
            0,
            "GOLDEN MISMATCH: emitted output changed vs the committed anchor.",
        )
        self.assertEqual(summary(text), "ll[llvm22]: changed 3, gone 0, new 0 · FAIL")
        text = block("ll[llvm23]", 4356, 0, 0, 1) + (
            "\nNO GOLDEN for ll[llvm24]. Bless first: run_diff.py --mode ll "
            "--flavor llvm24 --record-golden\n"
        )
        self.assertEqual(
            summary(text),
            "ll[llvm23]: changed 0, gone 0, new 1; ll[llvm24]: no golden · FAIL",
        )

    def test_canonical_and_an_empty_log(self):
        self.assertEqual(
            summary(block("ir_canonical", 3547, 0, 0, 275)),
            "ir_canonical: changed 0, gone 0, new 275",
        )
        self.assertEqual(
            summary("Traceback (most recent call last):\n"),
            "no golden check in the log · FAIL",
        )

    def test_fields_split_on_blanks_and_a_missing_log_prints_nothing(self):
        """awk's fields: runs of spaces and tabs; a "NEW" line with no check before it still prints. A log that is not
        there prints nothing and exits 2, as awk does."""
        text = "=== GOLDEN CHECK (x[(y)])\t===\n  UNBLESSED CHANGE:\t 4 \t\n  MISSING (gone)\n  NEW (unblessed) : 7"
        self.assertEqual(summary(text), "x[y]: changed 4, gone (gone), new 7")
        self.assertEqual(summary("  NEW (unblessed) : 9\n"), ": changed , gone , new 9")
        p = subprocess.run(
            [sys.executable, "-B", str(TOOL), str(TOOL.parent / "no_such.log")],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual((p.returncode, p.stdout), (2, ""))


if __name__ == "__main__":
    unittest.main()
