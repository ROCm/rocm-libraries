# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Gate the generated intrinsic arch-domain artifacts against drift.

``tools/gen_arch_domain.py`` measures, per LLVM flavor, which intrinsic
declarations actually link for which gfx target, and commits the answer as
``python/rocke/core/arch/data/intrinsic_arch_domain.<flavor>.json``. A generated
artifact that is not gated drifts from its generator, and a stale availability
table is worse than none because it is trusted.

Two gates, because they fail for different reasons and are available in
different places:

* **Structure** runs everywhere — no LLVM, no GPU, no built C++ engine. It reads
  the committed JSON and checks it against the decl table it claims to describe.
  This is the gate that catches the realistic drift: someone adds, renames or
  removes an intrinsic key and the artifact silently stops covering it. The
  artifact would still be internally consistent, still parse, and still look
  authoritative, while saying nothing about the new key.

* **Regeneration** re-probes the toolchain and asserts the committed column for
  *this host's* flavor comes back byte-identical. It is the only check that can
  catch a wrong measurement rather than a missing one. A full sweep is ~150 keys
  x 7 targets and finishes in seconds, so it is a test rather than a nightly.

A host can only ever measure its own LLVM, so a flavor with no matching
toolchain is skipped, never failed — "we did not get an answer" must not be
recorded as "the answer is no". Same reason the artifact distinguishes
``arch_absent`` (the target genuinely cannot lower it) from ``target_unsupported``
and ``toolchain_crash`` (no data).
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from rocke.core.isa.backend import wired_arches
from rocke.core.lower_llvm import _resolve_llvm_flavor

_HERE = Path(__file__).resolve().parent
_ROCKE = _HERE.parents[1]  # tests/core -> platform
_TOOL = _ROCKE / "tools" / "gen_arch_domain.py"
_DATA = _ROCKE / "python" / "rocke" / "core" / "arch" / "data"

# Import the generator for its constants and its decl-table merge rule, so the
# gate cannot disagree with the tool about what "the decl table for a flavor"
# means. Re-deriving the merge here would just create a second place to forget
# a rung when flavor resolution gains one.
if str(_ROCKE / "tools") not in sys.path:
    sys.path.insert(0, str(_ROCKE / "tools"))
import gen_arch_domain as G

_STATUSES = {
    G.STATUS_OK,
    G.STATUS_NAME_ABSENT,
    G.STATUS_ARCH_ABSENT,
    G.STATUS_TARGET_UNSUPPORTED,
    G.STATUS_TOOLCHAIN_CRASH,
    G.STATUS_PROBE_ERROR,
}

# Every status except `ok` is a claim about something that went wrong, and a
# claim with no diagnostic behind it cannot be acted on by whoever hits it.
_NEEDS_EVIDENCE = _STATUSES - {G.STATUS_OK}


def _columns() -> list[tuple[str, Path]]:
    """Committed artifacts as ``(flavor, path)``, flavor taken from the name."""
    out = []
    for path in sorted(_DATA.glob("intrinsic_arch_domain.*.json")):
        out.append((path.name.split(".")[1], path))
    return out


class ArchDomainArtifactStructureTest(unittest.TestCase):
    """Checks that need no toolchain, so they run on every host."""

    def test_at_least_one_column_is_committed(self):
        self.assertTrue(_columns(), f"no arch-domain artifact under {_DATA}")

    def test_columns_are_well_formed(self):
        for flavor, path in _columns():
            with self.subTest(flavor=flavor):
                doc = json.loads(path.read_text())
                self.assertEqual(doc.get("schema"), G.SCHEMA)

                # Provenance must agree with the filename. An artifact filed
                # under the wrong LLVM vintage is worse than a missing one: the
                # measurement is real, just attributed to a toolchain that never
                # produced it, and nothing downstream can tell.
                tc = doc.get("toolchain", {})
                self.assertEqual(tc.get("flavor"), flavor)
                clang_flavor = G._flavor_of_clang(tc.get("clang", ""))
                if clang_flavor is not None:
                    self.assertEqual(clang_flavor, flavor, tc.get("clang"))

                self.assertEqual(sorted(tc.get("arches", [])), sorted(wired_arches()))

    def test_every_decl_key_is_covered_on_every_wired_arch(self):
        """The drift this file exists to catch.

        The artifact answers a question *about the decl table*. Add a key and
        the artifact does not grow a row; it just quietly has no opinion on the
        one intrinsic nobody has measured yet. Nothing else in the tree notices,
        because the JSON is still valid and every row it does have is still
        right.

        Enforced asymmetrically, for the same reason the artifact separates
        ``arch_absent`` from ``target_unsupported``:

        * a **stale** key -- in the artifact, gone from the decl table -- is
          always a defect, and is caught on every column from any host. Nothing
          about it needs a toolchain to fix.
        * an **unmeasured** key can only be answered on a host running that
          flavor's toolchain. Failing every column here would mean one new decl
          key reds the build on every machine until someone has run all three
          toolchains -- "we did not get an answer" recorded as "the answer is
          no", which is exactly what the design forbids. So full coverage is
          required of this host's flavor, and reported as a named skip for the
          rest.
        """
        host = _resolve_llvm_flavor()
        arches = sorted(wired_arches())
        for flavor, path in _columns():
            with self.subTest(flavor=flavor):
                doc = json.loads(path.read_text())
                expected = set(G._decl_table(flavor))
                got = set(doc["keys"])
                for key, row in doc["keys"].items():
                    self.assertEqual(sorted(row), arches, key)
                self.assertEqual(
                    sorted(got - expected),
                    [],
                    f"{path.name} measures keys the decl table no longer has "
                    "-- re-run tools/gen_arch_domain.py and commit the result",
                )
                missing = sorted(expected - got)
                if not missing:
                    continue
                why = (
                    f"{path.name} has no answer for {len(missing)} decl "
                    f"key(s): {missing} -- re-run tools/gen_arch_domain.py on "
                    f"a {flavor} toolchain and commit the result"
                )
                if flavor == host:
                    self.fail(why)
                self.skipTest(why)

    def test_every_cell_carries_an_actionable_answer(self):
        for flavor, path in _columns():
            doc = json.loads(path.read_text())
            for key, row in doc["keys"].items():
                for arch, cell in row.items():
                    with self.subTest(flavor=flavor, key=key, arch=arch):
                        status = cell.get("status")
                        self.assertIn(status, _STATUSES)
                        # `probe_error` means our own probe module was
                        # malformed -- the one status that is a defect in the
                        # generator rather than a fact about the toolchain, and
                        # so must never survive into a committed column.
                        self.assertNotEqual(status, G.STATUS_PROBE_ERROR)
                        self.assertEqual(cell.get("verified_on"), flavor)
                        if status in _NEEDS_EVIDENCE:
                            self.assertTrue(cell.get("evidence"), "no diagnostic")


class ArchDomainRegenerationTest(unittest.TestCase):
    """Re-probe this host's flavor and require the committed column to match."""

    def test_regenerating_this_hosts_column_is_a_no_op(self):
        proc = subprocess.run(
            [sys.executable, str(_TOOL), "--check"],
            capture_output=True,
            text=True,
            check=False,
            timeout=1800,
        )
        out = proc.stdout + proc.stderr

        # Three distinct kinds of "we cannot answer here", all skips. Failing
        # any of them would red every CI machine whose ROCm differs from the one
        # that blessed the artifact, which is most of them.
        if "UNVALIDATED" in out:
            self.skipTest(out.strip().splitlines()[-1])
        if proc.returncode == 2 and "clang reports" in out:
            self.skipTest("toolchain flavor disagrees with the resolved flavor")

        self.assertEqual(proc.returncode, 0, out)


if __name__ == "__main__":
    unittest.main()
