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

That last one is a lifecycle invariant rather than a single behaviour: every way
a run can end -- no code object, no llvm-dwarfdump, an unreadable object, a
failed write, a capture that deliberately writes no sidecar at all -- has to
leave each dispatch holding a complete sidecar from this run or none, never an
earlier one and never a partial file.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_TOOLS = (
    Path(__file__).resolve().parents[2]
    / "dsl_docs/optimization/utilities/tools/wavescope"
)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


efi = _load("emit_inline_frames")
cwt = _load("capture_wavescope_trace")


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

    def leftovers(self, d: Path) -> list[str]:
        """Everything sidecar-shaped in ``d``, temporary files included."""
        return sorted(p.name for p in d.iterdir() if p.name.startswith(efi.SIDECAR))


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
        self.assertEqual(self.leftovers(d), [efi.SIDECAR])


class TestNothingSurvivesAFailedRun(TwoObjectTrace, unittest.TestCase):
    """Invalidation has to happen before anything that can end the run.

    Ordering is the whole finding here. Removing stale sidecars inside the
    per-dispatch loop looks equivalent and is not: the loop is reached only
    after a code object has been found and llvm-dwarfdump located, so on a host
    missing either, the run exits reporting that no sidecar was written while
    every previous sidecar is still sitting in the folder for the viewer to
    read. The wrapper's warning then actively misleads -- it says the Source tab
    has fallen back to innermost frames when it is in fact showing another
    build's call stacks.
    """

    def test_a_stale_sidecar_goes_even_when_no_code_object_is_found(self):
        d = self.dispatch("1", 1)
        self.stale(d)
        for obj in self.objs.values():
            obj.unlink()
        with self.assertRaises(SystemExit):
            efi.main([str(self.tmp)])
        self.assertEqual(self.leftovers(d), [])

    def test_a_stale_sidecar_goes_even_when_dwarfdump_is_missing(self):
        d = self.dispatch("1", 1)
        self.stale(d)

        def missing():
            raise SystemExit("llvm-dwarfdump not found")

        with mock.patch.object(efi, "find_dwarfdump", missing):
            with self.assertRaises(SystemExit):
                efi.main([str(self.tmp)])
        self.assertEqual(self.leftovers(d), [])

    def test_a_stale_sidecar_goes_even_when_the_object_cannot_be_read(self):
        """A dwarfdump failure ends the loop part way through the dispatches."""
        first = self.dispatch("1", 1)
        second = self.dispatch("2", 2)
        self.stale(second)

        def unreadable(obj, _dd):
            raise SystemExit(f"llvm-dwarfdump failed on {obj}")

        with mock.patch.object(efi, "parse_inline_frames", unreadable):
            with self.assertRaises(SystemExit):
                efi.main([str(self.tmp)])
        self.assertEqual(self.leftovers(first), [])
        self.assertEqual(self.leftovers(second), [])

    def test_an_interrupted_write_leaves_neither_the_new_nor_the_old_sidecar(self):
        """A half-written file is unreadable; the old one is worse -- it parses."""
        d = self.dispatch("1", 1)
        self.stale(d)
        with mock.patch.object(Path, "replace", side_effect=OSError("no space")):
            with self.assertRaises(OSError):
                efi.main([str(self.tmp)])
        self.assertEqual(self.leftovers(d), [])

    def test_a_temporary_file_from_an_interrupted_run_is_cleaned_up(self):
        """An interrupt leaves a `.tmp`; the next run must not leave it there."""
        d = self.dispatch("9", 9)
        (d / f"{efi.SIDECAR}{efi.TMP_SUFFIX}").write_text('{"version": 2, "stac')
        with self.assertRaises(SystemExit):
            efi.main([str(self.tmp)])
        self.assertEqual(self.leftovers(d), [])


class TestInvalidateOnly(TwoObjectTrace, unittest.TestCase):
    """The mode the wrapper uses after a capture that writes no sidecar."""

    def test_it_drops_sidecars_without_needing_a_code_object_or_the_tool(self):
        d = self.dispatch("1", 1)
        self.stale(d)
        for obj in self.objs.values():
            obj.unlink()

        def missing():
            raise SystemExit("llvm-dwarfdump not found")

        with mock.patch.object(efi, "find_dwarfdump", missing):
            self.assertEqual(efi.main([str(self.tmp), "--invalidate-only"]), 0)
        self.assertEqual(self.leftovers(d), [])

    def test_a_folder_with_nothing_to_drop_is_not_an_error(self):
        self.dispatch("1", 1)
        self.assertEqual(efi.main([str(self.tmp), "--invalidate-only"]), 0)


class TestCaptureWithoutSource(unittest.TestCase):
    """``--no-source`` writes no sidecar, so it has to remove the one it finds.

    The trace in the folder is new after a recapture and the sidecar beside it
    describes a build that no longer exists. Skipping the sidecar step is not
    the same as leaving the folder alone.
    """

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.out = self.tmp / "att_out"
        self.dispatch_dir = self.out / "ui_output_kernel_dispatch_0"
        self.dispatch_dir.mkdir(parents=True)
        (self.dispatch_dir / "code.json").write_text(
            json.dumps({"code": [row("v_mov", 1, 100)]})
        )
        self.sidecar = self.dispatch_dir / efi.SIDECAR
        self.sidecar.write_text(json.dumps({"version": 2, "stacks": {"9:100": []}}))
        self.enterContext(mock.patch.object(cwt, "run_capture", lambda *a, **kw: None))

    def run_wrapper(self, *flags: str) -> int:
        return cwt.main(["--output-dir", str(self.out), *flags, "--", "true"])

    def test_a_recapture_without_source_removes_the_old_sidecar(self):
        self.assertEqual(self.run_wrapper("--no-source"), 0)
        self.assertFalse(self.sidecar.exists())

    def test_a_capture_with_source_still_delegates_to_the_sidecar_tool(self):
        seen = []
        with mock.patch.object(
            cwt, "run_sidecar", lambda out, obj: seen.append(Path(out)) or True
        ):
            self.assertEqual(self.run_wrapper(), 0)
        self.assertEqual(seen, [self.out.resolve()])


if __name__ == "__main__":
    unittest.main()
