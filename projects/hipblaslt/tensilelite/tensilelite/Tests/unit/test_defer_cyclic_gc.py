# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import unittest
from unittest.mock import patch

import Tensile.TensileCreateLibrary.Run as run_module


class RecordingGC:
    def __init__(self, enabled):
        self.enabled = enabled
        self.events = []
        self.unfreeze = lambda: self.events.append("unfreeze")

    def isenabled(self):
        self.events.append("isenabled")
        return self.enabled

    def freeze(self):
        self.events.append("freeze")

    def disable(self):
        self.events.append("disable")
        self.enabled = False

    def enable(self):
        self.events.append("enable")
        self.enabled = True


class TestDeferCyclicGC(unittest.TestCase):
    def assertOrdering(self, events, initially_enabled):
        """Assert the orderings the contract depends on, and only those.

        Whether the entry freeze happens before or after disabling, and where
        the atexit registration lands relative to re-enabling, are both
        arbitrary; pinning them would turn harmless refactors into failures.
        """
        body = events.index("body")
        freezes = [i for i, event in enumerate(events) if event == "freeze"]

        # The prior state has to be sampled before we clobber it, and
        # collection has to be off for the whole body.
        self.assertLess(events.index("isenabled"), events.index("disable"))
        self.assertLess(events.index("disable"), body)

        # Freeze once on entry, once on exit -- the exit freeze has to happen
        # even when the body raises, and before collection is switched back on.
        self.assertEqual(len(freezes), 2)
        entryFreeze, exitFreeze = freezes
        self.assertLess(entryFreeze, body)
        self.assertGreater(exitFreeze, body)

        self.assertGreater(events.index("register"), body)

        if initially_enabled:
            self.assertEqual(events.count("enable"), 1)
            self.assertLess(exitFreeze, events.index("enable"))
        else:
            self.assertNotIn("enable", events)

    def exercise(self, initially_enabled, raises):
        fake_gc = RecordingGC(initially_enabled)
        registered = []

        def register(callback):
            fake_gc.events.append("register")
            registered.append(callback)
            return callback

        with (
            patch.object(run_module, "gc", fake_gc),
            patch.object(run_module.atexit, "register", side_effect=register),
        ):
            if raises:
                with self.assertRaisesRegex(RuntimeError, "parse failed"):
                    with run_module.deferCyclicGC():
                        fake_gc.events.append("body")
                        self.assertFalse(fake_gc.enabled)
                        raise RuntimeError("parse failed")
            else:
                with run_module.deferCyclicGC():
                    fake_gc.events.append("body")
                    self.assertFalse(fake_gc.enabled)

        self.assertOrdering(fake_gc.events, initially_enabled)
        self.assertEqual(fake_gc.enabled, initially_enabled)

        # What was deferred is the unfreeze itself: running the registered
        # callback has to reach gc.unfreeze, not some no-op wrapper.
        self.assertEqual(len(registered), 1)
        registered[0]()
        self.assertEqual(fake_gc.events[-1], "unfreeze")

    def test_restores_prior_gc_state_on_normal_and_exceptional_exit(self):
        for initially_enabled in (True, False):
            for raises in (False, True):
                with self.subTest(initially_enabled=initially_enabled, raises=raises):
                    self.exercise(initially_enabled=initially_enabled, raises=raises)
