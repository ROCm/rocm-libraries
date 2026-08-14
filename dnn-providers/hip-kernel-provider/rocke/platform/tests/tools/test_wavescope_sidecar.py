# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for the WaveScope inline-frames sidecar producer.

The join the sidecar feeds is keyed on ``"codeobj:addr"``, and virtual
addresses repeat across code objects, so attributing a dispatch to the wrong
object does not fail -- it silently reports another kernel's source against
these instructions. Everything here is about refusing to guess:

  * a dispatch is matched to the code object it actually ran, independently of
    the other dispatches in the trace;
  * a trace whose objects cannot be told apart is reported, not guessed at;
  * rows belonging to another object are skipped rather than joined on address;
  * a sidecar from an earlier run over the same folder is gone before this run
    decides anything, so an unresolvable dispatch cannot keep serving the
    previous run's attribution.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_TOOL = (
    Path(__file__).resolve().parents[2]
    / "dsl_docs/optimization/utilities/tools/wavescope/emit_inline_frames.py"
)
_spec = importlib.util.spec_from_file_location("emit_inline_frames", _TOOL)
efi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(efi)


def dumped(tmp: Path, ident: int, size: int = 1024) -> Path:
    """A stand-in for a code object rocprofv3 dumped, named the way it names."""
    path = tmp / f"kernel_code_object_id_{ident}.out"
    path.write_bytes(b"\0" * size)
    return path


def row(isa: str, codeobj: int, vaddr: int) -> list:
    """A code.json row: only the ISA, code object and address columns matter."""
    out = [isa, "", "", "", codeobj, vaddr]
    assert out[efi.CODEOBJ_COL] == codeobj and out[efi.VADDR_COL] == vaddr
    return out


def frame(name: str, lo: int, hi: int, depth: int = 1, line: int = 7) -> dict:
    return {
        "depth": depth,
        "name": name,
        "ranges": [(lo, hi)],
        "call_file": "/src/gemm.py",
        "call_line": line,
        "call_col": 3,
    }


class TestSelectCodeObject(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_matches_on_the_id_the_dispatch_ran(self):
        objs = [dumped(self.tmp, 1), dumped(self.tmp, 2)]
        path, ident, problem = efi.select_code_object(objs, {"2"}, None)
        self.assertIsNone(problem)
        self.assertEqual(ident, "2")
        self.assertEqual(path.name, "kernel_code_object_id_2.out")

    def test_a_dispatch_that_ran_another_object_is_not_guessed_at(self):
        """The regression: one dumped object, a dispatch that did not run it.

        Taking the dispatch's single id and pairing it with whatever DWARF was
        dumped attributes one kernel's source to another's instructions, and
        the addresses overlap often enough that it looks plausible.
        """
        path, ident, problem = efi.select_code_object(
            [dumped(self.tmp, 1)], {"2"}, None
        )
        self.assertIsNone(path)
        self.assertIsNone(ident)
        self.assertIn("--code-object", problem)

    def test_largest_object_does_not_win_across_dispatches(self):
        """Selection is per dispatch, so a big unrelated object is ignored."""
        big = dumped(self.tmp, 1, size=1 << 20)
        small = dumped(self.tmp, 2, size=16)
        for present, expected in (({"1"}, big), ({"2"}, small)):
            path, ident, problem = efi.select_code_object([big, small], present, None)
            self.assertIsNone(problem)
            self.assertEqual(path, expected)
            self.assertEqual({ident}, present)

    def test_several_matching_objects_are_ambiguous(self):
        objs = [dumped(self.tmp, 1), dumped(self.tmp, 2)]
        _, _, problem = efi.select_code_object(objs, {"1", "2"}, None)
        self.assertIn("--code-object", problem)

    def test_an_unlabelled_explicit_object_is_trusted_when_unambiguous(self):
        """An .hsaco from the build carries no id, but the caller named it."""
        hsaco = self.tmp / "kernel.hsaco"
        hsaco.write_bytes(b"\0")
        path, ident, problem = efi.select_code_object([hsaco], {"3"}, hsaco)
        self.assertIsNone(problem)
        self.assertEqual((path, ident), (hsaco, "3"))

    def test_an_unlabelled_explicit_object_is_refused_when_ambiguous(self):
        hsaco = self.tmp / "kernel.hsaco"
        hsaco.write_bytes(b"\0")
        _, _, problem = efi.select_code_object([hsaco], {"3", "4"}, hsaco)
        self.assertIsNotNone(problem)


class TestBuildSidecar(unittest.TestCase):
    def test_rows_from_another_object_are_skipped(self):
        rows = [row("v_mov", 1, 100), row("v_add", 2, 100)]
        sidecar = efi.build_sidecar(rows, [frame("load_a", 0, 200)], "1")
        self.assertEqual(sidecar["resolved"], 1)
        self.assertEqual(sidecar["skipped_other_object"], 1)
        self.assertEqual(list(sidecar["stacks"]), ["1:100"])

    def test_keys_carry_the_code_object(self):
        """Two objects share address 100; the key is what keeps them apart."""
        rows = [row("v_mov", 1, 100), row("v_add", 2, 100)]
        sidecar = efi.build_sidecar(rows, [frame("load_a", 0, 200)], None)
        self.assertEqual(sorted(sidecar["stacks"]), ["1:100", "2:100"])

    def test_version_is_stamped(self):
        sidecar = efi.build_sidecar([row("v_mov", 1, 8)], [frame("f", 0, 16)], "1")
        self.assertEqual(sidecar["version"], efi.SIDECAR_VERSION)
        self.assertEqual(sidecar["code_object_id"], "1")


class TwoObjectTrace:
    """A trace holding two code objects that both place a symbol at address 100.

    The shared address is the point: a dispatch attributed to the wrong object
    still joins, so every wrong answer here looks like a right one.
    """

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.objs = {i: dumped(self.tmp, i) for i in (1, 2)}
        # Both objects place a symbol at 100, which is what makes a wrong
        # attribution join cleanly instead of coming back empty.
        self.frames = {
            self.objs[1]: [frame("from_object_one", 0, 200)],
            self.objs[2]: [frame("from_object_two", 0, 200)],
        }
        self.enterContext(
            mock.patch.object(efi, "find_dwarfdump", lambda: "llvm-dwarfdump")
        )
        self.enterContext(
            mock.patch.object(
                efi, "parse_inline_frames", lambda obj, _dd: self.frames[obj]
            )
        )

    def dispatch(self, name: str, codeobj: int) -> Path:
        d = self.tmp / f"ui_output_kernel_dispatch_{name}"
        d.mkdir()
        (d / "code.json").write_text(json.dumps({"code": [row("v_mov", codeobj, 100)]}))
        return d

    def sidecar_of(self, d: Path) -> dict:
        return json.loads((d / efi.SIDECAR).read_text())


class TestMainPerDispatch(TwoObjectTrace, unittest.TestCase):
    """End to end over a trace with two dispatches of two different kernels."""

    def test_each_dispatch_gets_its_own_object(self):
        one = self.dispatch("1", 1)
        two = self.dispatch("2", 2)
        self.assertEqual(efi.main([str(self.tmp)]), 0)
        self.assertEqual(self.sidecar_of(one)["functions"], ["from_object_one"])
        self.assertEqual(self.sidecar_of(two)["functions"], ["from_object_two"])

    def test_a_dispatch_with_no_matching_object_is_skipped_not_mislabelled(self):
        """Skipping one dispatch is still a successful run.

        The dispatches that did resolve have their sidecars, so reporting
        failure would have the caller tell the user no sidecar was written and
        that the Source tab has lost its inlining, when most of the trace has
        it.
        """
        good = self.dispatch("1", 1)
        orphan = self.dispatch("9", 9)
        self.assertEqual(efi.main([str(self.tmp)]), 0)
        self.assertTrue((good / efi.SIDECAR).is_file())
        self.assertFalse((orphan / efi.SIDECAR).is_file())

    def test_no_usable_dispatch_is_an_error(self):
        self.dispatch("9", 9)
        with self.assertRaises(SystemExit):
            efi.main([str(self.tmp)])


class TestRerunOverAnExistingSidecar(TwoObjectTrace, unittest.TestCase):
    """A trace folder gets re-decoded and re-run over; stale answers must go.

    The dangerous case is a dispatch that resolved on a previous run and cannot
    on this one -- a different ``--code-object``, a re-decode that dropped the
    matching dump. Leaving the old file behind means the viewer keeps reading
    it, and because addresses repeat across code objects it joins cleanly and
    reports the wrong kernel's source rather than falling back.
    """

    def stale(self, d: Path) -> Path:
        """A sidecar a previous run left, naming a function from another object."""
        path = d / efi.SIDECAR
        path.write_text(
            json.dumps(
                {
                    "version": efi.SIDECAR_VERSION,
                    "functions": ["from_a_previous_run"],
                    "files": ["/src/old.py"],
                    "stacks": {"9:100": [[0, 0, 1, 0]]},
                    "resolved": 1,
                }
            )
        )
        return path

    def test_stale_sidecar_is_gone_when_the_dispatch_cannot_be_resolved(self):
        orphan = self.dispatch("9", 9)
        good = self.dispatch("1", 1)
        self.stale(orphan)
        self.assertEqual(efi.main([str(self.tmp)]), 0)
        self.assertFalse((orphan / efi.SIDECAR).exists())
        self.assertEqual(self.sidecar_of(good)["functions"], ["from_object_one"])

    def test_stale_sidecar_is_replaced_when_the_dispatch_resolves(self):
        d = self.dispatch("1", 1)
        self.stale(d)
        self.assertEqual(efi.main([str(self.tmp)]), 0)
        self.assertEqual(self.sidecar_of(d)["functions"], ["from_object_one"])
        self.assertEqual(list(self.sidecar_of(d)["stacks"]), ["1:100"])

    def test_stale_sidecar_is_gone_even_when_the_whole_run_fails(self):
        """Nothing resolves, so the run reports failure -- with no stale file
        left to be read as though it had succeeded."""
        orphan = self.dispatch("9", 9)
        self.stale(orphan)
        with self.assertRaises(SystemExit):
            efi.main([str(self.tmp)])
        self.assertFalse((orphan / efi.SIDECAR).exists())

    def test_no_temporary_file_is_left_beside_the_sidecar(self):
        """The write goes through a temp name; it must not survive the run."""
        d = self.dispatch("1", 1)
        self.assertEqual(efi.main([str(self.tmp)]), 0)
        self.assertEqual(
            [p.name for p in d.iterdir() if p.name.startswith(efi.SIDECAR)],
            [efi.SIDECAR],
        )


if __name__ == "__main__":
    unittest.main()
