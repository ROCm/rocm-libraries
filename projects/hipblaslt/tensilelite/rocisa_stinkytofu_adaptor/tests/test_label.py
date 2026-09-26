# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Standalone tests for ``rocisa_stinkytofu_adaptor.label``."""

from __future__ import annotations

import copy
import os
import pickle
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.normpath(os.path.join(_HERE, ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from rocisa_stinkytofu_adaptor.label import LabelManager  # noqa: E402


class TestLabelManagerAddName(unittest.TestCase):
    def test_first_call_inserts_at_zero(self):
        lm = LabelManager()
        lm.addName("foo")
        self.assertEqual(lm._labels, {"foo": 0})

    def test_second_call_bumps_to_one(self):
        lm = LabelManager()
        lm.addName("foo")
        lm.addName("foo")
        self.assertEqual(lm._labels["foo"], 1)

    def test_repeated_calls_increment(self):
        lm = LabelManager()
        for _ in range(5):
            lm.addName("foo")
        self.assertEqual(lm._labels["foo"], 4)

    def test_distinct_names_track_independently(self):
        lm = LabelManager()
        lm.addName("foo")
        lm.addName("foo")
        lm.addName("bar")
        self.assertEqual(lm._labels, {"foo": 1, "bar": 0})


class TestLabelManagerGetName(unittest.TestCase):
    def test_first_call_inserts_at_zero_and_returns_bare_name(self):
        lm = LabelManager()
        self.assertEqual(lm.getName("foo"), "foo")
        self.assertEqual(lm._labels, {"foo": 0})

    def test_idempotent_for_existing_name(self):
        lm = LabelManager()
        lm.getName("foo")
        lm.getName("foo")
        self.assertEqual(lm._labels["foo"], 0)

    def test_returns_suffixed_name_when_count_nonzero(self):
        lm = LabelManager()
        lm.addName("foo")
        lm.addName("foo")
        self.assertEqual(lm.getName("foo"), "foo_1")
        lm.addName("foo")
        self.assertEqual(lm.getName("foo"), "foo_2")


class TestLabelManagerGetNameInc(unittest.TestCase):
    def test_first_call_returns_bare_name(self):
        lm = LabelManager()
        self.assertEqual(lm.getNameInc("foo"), "foo")
        self.assertEqual(lm._labels["foo"], 0)

    def test_second_call_returns_underscore_one(self):
        lm = LabelManager()
        lm.getNameInc("foo")
        self.assertEqual(lm.getNameInc("foo"), "foo_1")
        self.assertEqual(lm._labels["foo"], 1)

    def test_third_call_returns_underscore_two(self):
        lm = LabelManager()
        lm.getNameInc("foo")
        lm.getNameInc("foo")
        self.assertEqual(lm.getNameInc("foo"), "foo_2")

    def test_inc_after_getName_first_returns_underscore_one(self):
        lm = LabelManager()
        lm.getName("foo")
        self.assertEqual(lm.getNameInc("foo"), "foo_1")


class TestLabelManagerGetNameIndex(unittest.TestCase):
    def test_index_zero_returns_bare_name(self):
        lm = LabelManager()
        lm.addName("foo")
        self.assertEqual(lm.getNameIndex("foo", 0), "foo")

    def test_index_within_range_returns_suffixed_name(self):
        lm = LabelManager()
        for _ in range(3):
            lm.addName("foo")
        self.assertEqual(lm.getNameIndex("foo", 1), "foo_1")
        self.assertEqual(lm.getNameIndex("foo", 2), "foo_2")

    def test_missing_name_raises_runtime_error(self):
        lm = LabelManager()
        with self.assertRaises(RuntimeError) as cm:
            lm.getNameIndex("never_added", 0)
        self.assertIn("You have to add a label first", str(cm.exception))

    def test_index_exceeds_count_raises_runtime_error(self):
        lm = LabelManager()
        lm.addName("foo")
        with self.assertRaises(RuntimeError) as cm:
            lm.getNameIndex("foo", 1)
        self.assertIn("The index 1 exceeded.", str(cm.exception))
        self.assertIn("(> 0)", str(cm.exception))

    def test_getNameIndex_does_not_mutate_state(self):
        lm = LabelManager()
        lm.addName("foo")
        lm.addName("foo")
        before = dict(lm._labels)
        lm.getNameIndex("foo", 1)
        self.assertEqual(lm._labels, before)


class TestLabelManagerGetUniqueName(unittest.TestCase):
    def test_monotonic_shared_counter(self):
        lm = LabelManager()
        self.assertEqual(
            [lm.getUniqueNamePrefix("X") for _ in range(3)],
            ["X_0", "X_1", "X_2"],
        )
        self.assertEqual(lm.getUniqueName(), "label_3")
        self.assertEqual(lm.getUniqueNamePrefix("Y"), "Y_4")

    def test_inserts_into_registry(self):
        lm = LabelManager()
        name = lm.getUniqueName()
        self.assertEqual(name, "label_0")
        self.assertIn(name, lm._labels)
        self.assertEqual(lm._labels[name], 0)

    def test_skips_manually_added_collision(self):
        lm = LabelManager()
        lm.addName("X_0")
        self.assertEqual(lm.getUniqueNamePrefix("X"), "X_1")


class TestLabelManagerGetUniqueNamePrefix(unittest.TestCase):
    def test_starts_with_prefix(self):
        lm = LabelManager()
        name = lm.getUniqueNamePrefix("loop")
        self.assertEqual(name, "loop_0")

    def test_inserts_into_registry(self):
        lm = LabelManager()
        name = lm.getUniqueNamePrefix("loop")
        self.assertIn(name, lm._labels)
        self.assertEqual(lm._labels[name], 0)


class TestLabelManagerDeepCopy(unittest.TestCase):
    def test_deepcopy_preserves_counters(self):
        lm = LabelManager()
        lm.addName("foo")
        lm.addName("foo")
        lm.addName("bar")
        clone = copy.deepcopy(lm)
        self.assertEqual(clone._labels, {"foo": 1, "bar": 0})
        self.assertEqual(clone._counter, 0)

    def test_deepcopy_is_independent(self):
        lm = LabelManager()
        lm.addName("foo")
        clone = copy.deepcopy(lm)
        clone.addName("foo")
        clone.addName("baz")
        self.assertEqual(lm._labels, {"foo": 0})
        self.assertEqual(clone._labels, {"foo": 1, "baz": 0})

    def test_deepcopy_empty_manager(self):
        clone = copy.deepcopy(LabelManager())
        self.assertEqual(clone._labels, {})
        self.assertEqual(clone._counter, 0)

    def test_deepcopy_preserves_unique_counter(self):
        lm = LabelManager()
        lm.getUniqueNamePrefix("X")
        lm.getUniqueNamePrefix("X")
        clone = copy.deepcopy(lm)
        self.assertEqual(clone.getUniqueNamePrefix("X"), "X_2")
        self.assertEqual(lm.getUniqueNamePrefix("X"), "X_2")


class TestLabelManagerPickle(unittest.TestCase):
    def test_pickle_roundtrip_preserves_counters(self):
        lm = LabelManager()
        lm.addName("foo")
        lm.addName("foo")
        lm.addName("bar")
        clone = pickle.loads(pickle.dumps(lm))
        self.assertEqual(clone._labels, lm._labels)
        self.assertEqual(clone._counter, lm._counter)

    def test_pickle_empty_manager(self):
        clone = pickle.loads(pickle.dumps(LabelManager()))
        self.assertEqual(clone._labels, {})
        self.assertEqual(clone._counter, 0)

    def test_pickle_state_is_isolated(self):
        lm = LabelManager()
        lm.addName("foo")
        state = lm.__getstate__()
        lm.addName("foo")
        self.assertEqual(state, ({"foo": 0}, 0))

    def test_pickle_roundtrip_independent(self):
        lm = LabelManager()
        lm.addName("foo")
        clone = pickle.loads(pickle.dumps(lm))
        clone.addName("foo")
        self.assertEqual(lm._labels, {"foo": 0})
        self.assertEqual(clone._labels, {"foo": 1})

    def test_counter_survives_pickle(self):
        lm = LabelManager()
        lm.getUniqueNamePrefix("X")
        lm.getUniqueNamePrefix("X")
        clone = pickle.loads(pickle.dumps(lm))
        self.assertEqual(clone.getUniqueNamePrefix("X"), "X_2")


class TestLabelManagerRepr(unittest.TestCase):
    def test_repr_contains_class_name_and_state(self):
        lm = LabelManager()
        lm.addName("foo")
        r = repr(lm)
        self.assertIn("LabelManager", r)
        self.assertIn("foo", r)


class TestLabelManagerPublicSurface(unittest.TestCase):
    def test_no_public_getData_method(self):
        self.assertFalse(hasattr(LabelManager(), "getData"))

    def test_no_dict_taking_ctor(self):
        with self.assertRaises(TypeError):
            LabelManager({"foo": 0})

    def test_no_magicGenerator(self):
        from rocisa_stinkytofu_adaptor import label as _label

        self.assertFalse(hasattr(_label, "magicGenerator"))


class TestLabelManagerScenarios(unittest.TestCase):
    def test_loop_label_collision_disambiguation(self):
        lm = LabelManager()
        outer = lm.getNameInc("LoopTop")
        inner = lm.getNameInc("LoopTop")
        self.assertEqual(outer, "LoopTop")
        self.assertEqual(inner, "LoopTop_1")

    def test_get_name_then_get_name_index_round_trip(self):
        lm = LabelManager()
        emitted = [lm.getNameInc("foo") for _ in range(3)]
        self.assertEqual(emitted, ["foo", "foo_1", "foo_2"])
        for i, expected in enumerate(emitted):
            self.assertEqual(lm.getNameIndex("foo", i), expected)

    def test_deepcopy_after_use_independent(self):
        lm = LabelManager()
        for _ in range(3):
            lm.addName("base")
        worker = copy.deepcopy(lm)
        worker.addName("base")
        worker.addName("worker_only")
        self.assertEqual(lm._labels, {"base": 2})
        self.assertEqual(worker._labels, {"base": 3, "worker_only": 0})


if __name__ == "__main__":
    unittest.main()
