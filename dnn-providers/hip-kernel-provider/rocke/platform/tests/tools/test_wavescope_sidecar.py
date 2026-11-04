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
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_TOOLS = Path(__file__).resolve().parents[2] / "dsl_docs/optimization/utilities/tools"


def _load(name: str, subdir: str = "wavescope"):
    spec = importlib.util.spec_from_file_location(name, _TOOLS / subdir / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


efi = _load("emit_inline_frames")
cwt = _load("capture_wavescope_trace")
cat = _load("capture_att_trace", "stage2_capture")
tp = _load("trace_provenance")

IDENTITY = dict(
    trace_id="trace-test",
    instruction_listing_hash_value="sha256:" + "0" * 64,
    code_object_hash=None,
)


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
        sidecar = efi.build_sidecar(rows, [frame("load_a", 0, 200)], "1", **IDENTITY)
        self.assertEqual(sidecar["resolved"], 1)
        self.assertEqual(sidecar["skipped_other_object"], 1)
        self.assertEqual(list(sidecar["stacks"]), ["1:100"])

    def test_keys_carry_the_code_object(self):
        """Two objects share address 100; the key is what keeps them apart."""
        rows = [row("v_mov", 1, 100), row("v_add", 2, 100)]
        sidecar = efi.build_sidecar(rows, [frame("load_a", 0, 200)], None, **IDENTITY)
        self.assertEqual(sorted(sidecar["stacks"]), ["1:100", "2:100"])

    def test_version_is_stamped(self):
        sidecar = efi.build_sidecar(
            [row("v_mov", 1, 8)], [frame("f", 0, 16)], "1", **IDENTITY
        )
        self.assertEqual(sidecar["version"], efi.SIDECAR_VERSION)
        self.assertEqual(sidecar["code_object_id"], "1")
        self.assertEqual(sidecar["traceId"], IDENTITY["trace_id"])
        self.assertEqual(
            sidecar["instructionListingHash"],
            IDENTITY["instruction_listing_hash_value"],
        )


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
        code_json = d / "code.json"
        code_json.write_text(json.dumps({"code": [row("v_mov", codeobj, 100)]}))
        tp.write_trace_sentinel(
            d,
            trace_id="capture-test",
            instruction_listing_hash=tp.instruction_listing_hash_file(code_json),
            capture=tp.CAPTURE_COMPLETE,
        )
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
        return sorted(
            p.name
            for p in d.iterdir()
            if p.name.startswith(efi.SIDECAR) or p.name.startswith(tp.TRACE_SENTINEL)
        )


class TestMainPerDispatch(TwoObjectTrace, unittest.TestCase):
    """End to end over a trace with two dispatches of two different kernels."""

    def test_each_dispatch_gets_its_own_object(self):
        one = self.dispatch("1", 1)
        two = self.dispatch("2", 2)
        self.assertEqual(efi.main([str(self.tmp)]), 0)
        for d, func in ((one, "from_object_one"), (two, "from_object_two")):
            sidecar = self.sidecar_of(d)
            self.assertEqual(sidecar["functions"], [func])
            self.assertEqual(sidecar["version"], efi.SIDECAR_VERSION)
            self.assertEqual(
                sidecar["instructionListingHash"],
                tp.instruction_listing_hash_file(d / "code.json"),
            )
            sentinel = tp.read_trace_sentinel(d)
            self.assertEqual(
                sentinel["instructionListingHash"], sidecar["instructionListingHash"]
            )
            self.assertEqual(sentinel["codeObjectHash"], sidecar["codeObjectHash"])
            self.assertEqual(sentinel["traceId"], sidecar["traceId"])

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
        self.assertEqual(self.leftovers(d), [efi.SIDECAR, tp.TRACE_SENTINEL])


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
        self.assertEqual(self.leftovers(d), [tp.TRACE_SENTINEL])

    def test_a_stale_sidecar_goes_even_when_dwarfdump_is_missing(self):
        d = self.dispatch("1", 1)
        self.stale(d)

        def missing():
            raise SystemExit("llvm-dwarfdump not found")

        with mock.patch.object(efi, "find_dwarfdump", missing):
            with self.assertRaises(SystemExit):
                efi.main([str(self.tmp)])
        self.assertEqual(self.leftovers(d), [tp.TRACE_SENTINEL])

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
        self.assertEqual(self.leftovers(first), [tp.TRACE_SENTINEL])
        self.assertEqual(self.leftovers(second), [tp.TRACE_SENTINEL])

    def test_an_interrupted_write_leaves_neither_the_new_nor_the_old_sidecar(self):
        """A half-written file is unreadable; the old one is worse -- it parses."""
        d = self.dispatch("1", 1)
        self.stale(d)
        with mock.patch.object(Path, "replace", side_effect=OSError("no space")):
            with self.assertRaises(OSError):
                efi.main([str(self.tmp)])
        self.assertEqual(self.leftovers(d), [tp.TRACE_SENTINEL])

    def test_a_temporary_file_from_an_interrupted_run_is_cleaned_up(self):
        """An interrupt leaves a `.tmp`; the next run must not leave it there."""
        d = self.dispatch("9", 9)
        (d / f"{efi.SIDECAR}{efi.TMP_SUFFIX}").write_text('{"version": 2, "stac')
        with self.assertRaises(SystemExit):
            efi.main([str(self.tmp)])
        self.assertEqual(self.leftovers(d), [tp.TRACE_SENTINEL])


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
        self.assertEqual(self.leftovers(d), [tp.TRACE_SENTINEL])

    def test_a_folder_with_nothing_to_drop_is_not_an_error(self):
        self.dispatch("1", 1)
        self.assertEqual(efi.main([str(self.tmp), "--invalidate-only"]), 0)

    def test_a_dispatch_whose_code_json_is_gone_is_still_cleaned(self):
        """The folder nothing else will clean up.

        A re-decode that removed or never finished writing ``code.json`` leaves
        a dispatch that cannot be resolved and therefore never reaches the loop
        -- while still holding the sidecar the last decode produced for it.
        Deciding what to clean from what is *resolvable* would skip exactly the
        folder most likely to be serving an answer for a trace that is gone.
        """
        d = self.dispatch("1", 1)
        self.stale(d)
        (d / "code.json").unlink()
        self.assertEqual(efi.main([str(self.tmp), "--invalidate-only"]), 0)
        self.assertEqual(self.leftovers(d), [tp.TRACE_SENTINEL])

    def test_a_sidecar_that_cannot_be_removed_stops_the_run(self):
        d = self.dispatch("1", 1)
        self.stale(d)
        with mock.patch.object(Path, "unlink", side_effect=OSError("read-only")):
            with self.assertRaises(SystemExit):
                efi.main([str(self.tmp), "--invalidate-only"])


class TestEmitterGenerationRouting(unittest.TestCase):
    """An output root names several captures, never one implicit trace."""

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.out = self.tmp / "att_out"
        self.first = self.out / "capture-first"
        self.second = self.out / "capture-second"
        self.dispatch = self.first / "ui_output_kernel_dispatch_0"
        self.dispatch.mkdir(parents=True)
        self.second.mkdir()
        self.sidecar = self.dispatch / efi.SIDECAR
        self.sidecar.write_text("{}")

    def test_output_root_lists_generations_instead_of_recursing(self):
        with self.assertRaises(SystemExit) as raised:
            efi.main([str(self.out)])

        message = str(raised.exception)
        self.assertIn("capture output root", message)
        self.assertIn(str(self.first), message)
        self.assertIn(str(self.second), message)
        self.assertIn(f"python emit_inline_frames.py {self.first}", message)
        self.assertTrue(self.sidecar.exists())

    def test_output_root_does_not_auto_select_its_only_generation(self):
        self.second.rmdir()
        with self.assertRaisesRegex(SystemExit, "capture output root"):
            efi.main([str(self.out)])
        self.assertTrue(self.sidecar.exists())

    def test_invalidate_only_refuses_before_crossing_generation_boundary(self):
        with self.assertRaisesRegex(SystemExit, "capture output root"):
            efi.main([str(self.out), "--invalidate-only"])
        self.assertTrue(self.sidecar.exists())


class TestCaptureGenerationIsolation(unittest.TestCase):
    """Every capture owns an empty generation, never a reused trace directory."""

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.out = self.tmp / "att_out"
        self.old_generation = self.out / "capture-old"
        self.old_dispatch = self.write_dispatch(
            self.old_generation, "v_mov", 1, trace_id="old"
        )
        self.old_sidecar = self.old_dispatch / efi.SIDECAR
        self.old_sidecar.write_text('{"version":2}')
        self.capture_id = "current"
        self.generation = tp.capture_generation(self.out, self.capture_id)
        self.enterContext(
            mock.patch.object(cat, "_preflight", lambda: Path("/opt/rocm/lib"))
        )
        self.enterContext(mock.patch.object(cat, "report", lambda d: None))

    def write_dispatch(
        self, generation: Path, isa: str, codeobj: int, *, trace_id: str | None = None
    ) -> Path:
        d = generation / "ui_output_kernel_dispatch_0"
        d.mkdir(parents=True)
        code = d / "code.json"
        code.write_text(json.dumps({"code": [row(isa, codeobj, 100)]}))
        if trace_id is not None:
            tp.write_trace_sentinel(
                d,
                trace_id=trace_id,
                instruction_listing_hash=tp.instruction_listing_hash_file(code),
                capture=tp.CAPTURE_COMPLETE,
            )
        return d

    def run_capture(self, fake_capture) -> int:
        argv = [
            "capture_att_trace.py",
            "--kernel-regex",
            "k",
            "--output-dir",
            str(self.out),
            "--capture-id",
            self.capture_id,
            "--",
            "true",
        ]
        with mock.patch.object(sys, "argv", argv):
            with mock.patch.object(cat, "capture", fake_capture):
                return cat.main()

    def current_dispatch(self) -> Path:
        return self.generation / "ui_output_kernel_dispatch_0"

    def test_partial_failure_is_truncated_without_touching_the_old_generation(self):
        def partial_then_fail(_command, _regex, output, *_args, **_kwargs):
            self.write_dispatch(output, "v_add", 2)
            raise cat.CaptureError("rocprofv3 exited 1")

        with self.assertRaises(cat.CaptureError):
            self.run_capture(partial_then_fail)
        current = self.current_dispatch()
        doc = tp.read_trace_sentinel(current)
        self.assertEqual(doc["capture"], tp.CAPTURE_TRUNCATED)
        self.assertEqual(doc["traceId"], self.capture_id)
        self.assertFalse((current / efi.SIDECAR).exists())
        self.assertTrue(self.old_sidecar.exists())
        self.assertEqual(tp.read_trace_sentinel(self.old_dispatch)["traceId"], "old")

    def test_keyboard_interrupt_marks_partial_output_truncated(self):
        def interrupted(_command, _regex, output, *_args, **_kwargs):
            self.write_dispatch(output, "v_add", 2)
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            self.run_capture(interrupted)
        self.assertEqual(
            tp.read_trace_sentinel(self.current_dispatch())["capture"],
            tp.CAPTURE_TRUNCATED,
        )

    def test_unexpected_programming_error_is_not_reclassified(self):
        def broken(_command, _regex, output, *_args, **_kwargs):
            self.write_dispatch(output, "v_add", 2)
            raise RuntimeError("bug")

        with self.assertRaisesRegex(RuntimeError, "bug"):
            self.run_capture(broken)
        self.assertIsNone(tp.read_trace_sentinel(self.current_dispatch()))

    def test_success_publishes_only_the_new_generation(self):
        def write_current(_command, _regex, output, *_args, **_kwargs):
            self.write_dispatch(output, "v_add", 2)

        self.assertEqual(self.run_capture(write_current), 0)
        doc = tp.read_trace_sentinel(self.current_dispatch())
        self.assertEqual(doc["capture"], tp.CAPTURE_COMPLETE)
        self.assertEqual(doc["traceId"], self.capture_id)
        self.assertTrue(self.old_sidecar.exists())

    def test_no_current_dispatch_is_an_error_even_when_an_old_one_exists(self):
        with self.assertRaisesRegex(SystemExit, "no current"):
            self.run_capture(lambda *_args, **_kwargs: None)
        self.assertFalse(self.generation.exists())
        self.assertTrue(self.old_sidecar.exists())

    def test_empty_expected_failures_remove_the_current_generation(self):
        failures = (
            cat.CaptureError("rocprofv3 exited 1"),
            OSError("cannot launch profiler"),
            KeyboardInterrupt(),
        )
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                with self.assertRaises(type(failure)):
                    self.run_capture(mock.Mock(side_effect=failure))
                self.assertFalse(self.generation.exists())
                self.assertTrue(self.old_generation.exists())

    def test_nonempty_generation_without_dispatch_is_retained(self):
        diagnostic = self.generation / "profiler.log"

        def write_diagnostic(_command, _regex, output, *_args, **_kwargs):
            (output / diagnostic.name).write_text("capture diagnostics")

        with self.assertRaisesRegex(SystemExit, "retained"):
            self.run_capture(write_diagnostic)
        self.assertEqual(diagnostic.read_text(), "capture diagnostics")
        self.assertTrue(self.old_generation.exists())

    def test_partial_directory_without_code_json_is_retained(self):
        partial = self.generation / "ui_output_kernel_dispatch_0"

        def write_partial(_command, _regex, output, *_args, **_kwargs):
            (output / partial.name).mkdir()

        with self.assertRaisesRegex(SystemExit, "retained"):
            self.run_capture(write_partial)
        self.assertTrue(partial.is_dir())

    def test_cleanup_failure_does_not_mask_the_capture_error(self):
        original = cat.CaptureError("primary profiler failure")

        def fail_without_output(*_args, **_kwargs):
            raise original

        with mock.patch.object(Path, "rmdir", side_effect=OSError("read-only")):
            with self.assertRaisesRegex(cat.CaptureError, "primary profiler failure"):
                self.run_capture(fail_without_output)
        self.assertTrue(self.generation.exists())

    def test_same_code_json_is_still_current_in_a_new_generation(self):
        old_text = (self.old_dispatch / "code.json").read_text()

        def write_same(_command, _regex, output, *_args, **_kwargs):
            d = output / "ui_output_kernel_dispatch_0"
            d.mkdir()
            (d / "code.json").write_text(old_text)

        self.assertEqual(self.run_capture(write_same), 0)
        self.assertEqual(
            tp.read_trace_sentinel(self.current_dispatch())["capture"],
            tp.CAPTURE_COMPLETE,
        )


class TestTraceProvenance(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_instruction_listing_hash_matches_wavescope_shape(self):
        rows = [
            ["; kernel", 0, 0, "", 1, 4096, 0, 0, 0, 0],
            ["v_mov_b32 v0, v1", 0, 1, "/a.py:2", 1, 4096, 1, 0, 0, 0],
        ]
        expected_input = (
            f'{json.dumps(["; kernel", 1, 4096], separators=(",", ":"))}\n'
            f'{json.dumps(["v_mov_b32 v0, v1", 1, 4096], separators=(",", ":"))}\n'
        )
        self.assertEqual(
            tp.instruction_listing_hash(rows), tp.sha256_text(expected_input)
        )

    def test_instruction_listing_hash_is_stable_across_json_formatting(self):
        rows = [["v_mov", 0, 1, "", 1, 100, 1, 0, 0, 0]]
        compact = json.dumps({"code": rows})
        spaced = json.dumps({"code": rows}, indent=2)
        path = self.tmp / "code.json"
        path.write_text(compact)
        hash_compact = tp.instruction_listing_hash_file(path)
        path.write_text(spaced)
        hash_spaced = tp.instruction_listing_hash_file(path)
        self.assertEqual(hash_compact, hash_spaced)

    def test_sha256_matches_exact_bytes(self):
        path = self.tmp / "code.json"
        path.write_text('{"code":[]}')
        self.assertEqual(tp.sha256_file(path), tp.sha256_text('{"code":[]}'))
        path.write_text('{"code":[]}\n')
        self.assertNotEqual(tp.sha256_file(path), tp.sha256_text('{"code":[]}'))

    def test_capture_generation_rejects_path_components(self):
        with self.assertRaises(ValueError):
            tp.capture_generation(self.tmp, "../outside")

    def test_sidecar_invalidation_preserves_capture_sentinels(self):
        d = self.tmp / "ui_output_k_dispatch_0"
        d.mkdir()
        (d / efi.SIDECAR).write_text("{}")
        (d / tp.TRACE_SENTINEL).write_text("{}")
        (d / f"{efi.SIDECAR}{efi.TMP_SUFFIX}").write_text("{}")
        self.assertEqual(tp.invalidate_sidecars(self.tmp), 2)
        self.assertEqual(self.leftovers(d), [tp.TRACE_SENTINEL])

    def leftovers(self, d: Path) -> list[str]:
        return sorted(
            p.name
            for p in d.iterdir()
            if p.name.startswith(efi.SIDECAR) or p.name.startswith(tp.TRACE_SENTINEL)
        )


class TestMonotonicCaptureState(TwoObjectTrace, unittest.TestCase):
    def test_truncated_capture_cannot_be_promoted_by_the_emitter(self):
        d = self.dispatch("1", 1)
        sentinel = tp.read_trace_sentinel(d)
        sentinel["capture"] = tp.CAPTURE_TRUNCATED
        tp.write_json_atomic(d / tp.TRACE_SENTINEL, sentinel)

        with self.assertRaisesRegex(SystemExit, "no sidecar written"):
            efi.main([str(self.tmp)])
        self.assertEqual(tp.read_trace_sentinel(d)["capture"], tp.CAPTURE_TRUNCATED)
        self.assertFalse((d / efi.SIDECAR).exists())

    def test_regeneration_preserves_capture_identity_and_status(self):
        d = self.dispatch("1", 1)
        before = tp.read_trace_sentinel(d)
        self.assertEqual(efi.main([str(self.tmp)]), 0)
        after = tp.read_trace_sentinel(d)
        self.assertEqual(after["traceId"], before["traceId"])
        self.assertEqual(after["capture"], before["capture"])
        self.assertEqual(
            after["instructionListingHash"], before["instructionListingHash"]
        )
        self.assertIn("codeObjectHash", after)

    def test_missing_sentinel_requires_explicit_legacy_opt_in(self):
        d = self.dispatch("1", 1)
        (d / tp.TRACE_SENTINEL).unlink()
        with self.assertRaisesRegex(SystemExit, "no sidecar written"):
            efi.main([str(self.tmp)])
        self.assertEqual(efi.main([str(self.tmp), "--assume-complete"]), 0)
        self.assertEqual(tp.read_trace_sentinel(d)["capture"], tp.CAPTURE_COMPLETE)

    def test_changed_code_json_is_refused_without_changing_capture_status(self):
        d = self.dispatch("1", 1)
        (d / "code.json").write_text(json.dumps({"code": [row("v_add", 1, 100)]}))
        with self.assertRaisesRegex(SystemExit, "no sidecar written"):
            efi.main([str(self.tmp)])
        self.assertEqual(tp.read_trace_sentinel(d)["capture"], tp.CAPTURE_COMPLETE)


class TestCaptureWrapper(unittest.TestCase):
    """The wrapper adds the environment and the sidecar step, nothing else.

    Invalidation deliberately does not live here: it belongs to the capture the
    wrapper delegates to, so that a direct capture behaves the same way.
    """

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.out = self.tmp / "att_out"
        self.generation = self.out / "capture-wrapper"
        (self.generation / "ui_output_kernel_dispatch_0").mkdir(parents=True)

    def run_wrapper(self, *flags: str) -> int:
        seen = []
        with (
            mock.patch.object(cwt, "run_capture", lambda *a, **kw: self.generation),
            mock.patch.object(
                cwt, "run_sidecar", lambda out, obj: seen.append(Path(out)) or True
            ),
        ):
            code = cwt.main(["--output-dir", str(self.out), *flags, "--", "true"])
        return code, seen

    def test_the_source_path_delegates_to_the_sidecar_tool(self):
        code, seen = self.run_wrapper()
        self.assertEqual(code, 0)
        self.assertEqual(seen, [self.generation])

    def test_no_source_skips_the_sidecar_step(self):
        code, seen = self.run_wrapper("--no-source")
        self.assertEqual(code, 0)
        self.assertEqual(seen, [])

    def test_capture_and_wrapper_agree_on_the_generation_path(self):
        capture_id = "11111111-2222-3333-4444-555555555555"
        completed = mock.Mock(returncode=0)
        with (
            mock.patch.object(cwt.uuid, "uuid4", return_value=capture_id),
            mock.patch.object(cwt.subprocess, "run", return_value=completed) as run,
        ):
            generation = cwt.run_capture(["true"], self.out, [], with_source=False)

        self.assertEqual(generation, self.out / f"capture-{capture_id}")
        argv = run.call_args.args[0]
        self.assertEqual(argv[argv.index("--capture-id") + 1], capture_id)
        self.assertEqual(argv[argv.index("--output-dir") + 1], str(self.out))


if __name__ == "__main__":
    unittest.main()
