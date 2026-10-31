# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""No-GPU tests that machine-state snapshots reach disk on a run that dies.

The snapshots exist to explain a run that went wrong, and a run that goes wrong
is the one that does not reach the end of ``main``. A device fault latches onto
the HIP context, so the next allocation raises and the process dies with the
``<stage>:at_fault`` snapshot still only in memory. That is not hypothetical:
the one sweep that actually faulted produced neither ``machine_state.json`` nor
``summary.json``, and an absent file is indistinguishable from "this run
predates the feature" or "the feature was off".

So the property under test is not "the snapshot is taken" -- that is six lines
of dict building -- but "the snapshot survives". These tests drive
``flush_machine_state`` directly rather than through a GPU run, which is why
they are honest about what they do not cover: whether a real sticky HIP fault
leaves enough of a process to write at all. ``flush_machine_state`` shells out
to nothing and only touches the filesystem, so it should, but that is an
argument rather than a measurement.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


def _states(*labels):
    return [{"label": x, "clean": True, "kfd_count": 0} for x in labels]


class TestMachineStateFlush(unittest.TestCase):
    def _flush(self, out, states):
        # By package name, not by source path: the installed test cohort has the
        # importable rocke package but no python/rocke source tree beside it.
        from rocke.examples.gfx1250.gemm_perf_fp8 import fp8_gemm_sweep as mod

        mod.flush_machine_state(out, states)
        return mod

    def test_snapshots_land_before_a_later_stage_can_raise(self):
        """The file must exist as soon as a stage has contributed snapshots."""
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            self._flush(out, _states("run:start", "tile:before", "tile:after"))
            f = out / "machine_state.json"
            self.assertTrue(f.exists(), "snapshots were not flushed to disk")
            got = json.loads(f.read_text())
            self.assertEqual(
                [s["label"] for s in got], ["run:start", "tile:before", "tile:after"]
            )

    def test_an_at_fault_snapshot_is_preserved(self):
        """The fault snapshot is the whole point; it must be on disk."""
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            states = _states("run:start", "trait:before")
            states.append(
                {
                    "label": "trait:at_fault",
                    "clean": True,
                    "faulted_candidate": "some_kernel",
                    "faulted_index": 53,
                    "candidates_total": 800,
                }
            )
            self._flush(out, states)
            got = json.loads((out / "machine_state.json").read_text())
            fault = [s for s in got if s["label"].endswith(":at_fault")]
            self.assertEqual(len(fault), 1, "at_fault snapshot did not survive")
            self.assertEqual(fault[0]["faulted_index"], 53)
            self.assertEqual(fault[0]["faulted_candidate"], "some_kernel")

    def test_a_later_flush_supersedes_an_earlier_one(self):
        """Each stage rewrites the file, so the last write wins and is complete."""
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            mod = self._flush(out, _states("run:start"))
            mod.flush_machine_state(out, _states("run:start", "tile:after"))
            got = json.loads((out / "machine_state.json").read_text())
            self.assertEqual(len(got), 2)

    def test_a_failed_write_does_not_mask_the_real_error(self):
        """Diagnostics must never replace the failure being diagnosed."""
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "file_not_a_dir"
            out.write_text("blocks mkdir")
            # Must not raise even though the path cannot hold a file.
            self._flush(out, _states("run:start"))


if __name__ == "__main__":
    unittest.main()
