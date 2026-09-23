# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Standalone tests for ``rocisa_stinkytofu_adaptor.label``.

Run from any working directory:

    python3 projects/hipblaslt/tensilelite/rocisa_stinkytofu_adaptor/tests/test_label.py

Or with pytest:

    pytest projects/hipblaslt/tensilelite/rocisa_stinkytofu_adaptor/tests/test_label.py

These tests pin behaviour that KernelWriter depends on for label
naming -- ``LabelManager`` produces unique-by-construction asm
labels and routes EVERY emitted label name through one of its
``getName / getNameInc / getUniqueName*`` accessors. Counter
semantics, error messages, and pickle / deepcopy round-trips must
all match ``rocisa::LabelManager`` byte-for-byte so swapping the
adaptor in cannot silently change emitted label names.
"""

from __future__ import annotations

import copy
import os
import pickle
import string
import sys
import unittest
from unittest import mock

# ---------------------------------------------------------------------------
# Self-contained sys.path bootstrap (mirrors test_code.py).
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.normpath(os.path.join(_HERE, ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)


from rocisa_stinkytofu_adaptor import label as _label  # noqa: E402
from rocisa_stinkytofu_adaptor.label import (  # noqa: E402
    LabelManager,
    magicGenerator,
)


# ===========================================================================
# magicGenerator -- 16-char random ``[A-Z0-9]`` string.
# ===========================================================================


class TestMagicGeneratorOutput(unittest.TestCase):
    """``magicGenerator`` returns a fresh 16-character string drawn from
    the rocisa alphabet (uppercase letters + digits, no punctuation,
    no lowercase). Mirror of label.hpp:34-43."""

    def test_length_is_16(self):
        # rocisa's ``LABEL_NAME_LENGTH = 17`` is misleading -- the
        # loop iterates ``LABEL_NAME_LENGTH - 1 == 16`` times. We
        # mirror the actual byte count.
        for _ in range(50):
            self.assertEqual(len(magicGenerator()), 16)

    def test_alphabet_is_uppercase_alnum(self):
        # 36-char alphabet: A-Z + 0-9. No lowercase, no punctuation,
        # no whitespace.
        allowed = set(string.ascii_uppercase + string.digits)
        for _ in range(50):
            chars = set(magicGenerator())
            self.assertTrue(
                chars.issubset(allowed),
                msg=f"unexpected chars: {chars - allowed}",
            )

    def test_uniqueness_in_a_small_batch(self):
        # 50 draws from a 36^16 space -- collision probability is
        # astronomically low (~1e-22). Any collision indicates the
        # RNG is broken.
        batch = {magicGenerator() for _ in range(50)}
        self.assertEqual(len(batch), 50)


class TestMagicGeneratorMonkeypatch(unittest.TestCase):
    """``magicGenerator`` is a module-level free function and must
    therefore be monkeypatchable via ``label._MAGIC_CHARS`` -- this is
    the lever tests use to pin output to a deterministic value."""

    def test_can_pin_alphabet(self):
        # Patching the alphabet to a single character makes the output
        # fully deterministic ("AAAAAAAAAAAAAAAA") -- proves the
        # constant is the only source of randomness besides Python's
        # global ``random`` state.
        with mock.patch.object(_label, "_MAGIC_CHARS", "A"):
            self.assertEqual(magicGenerator(), "A" * 16)


# ===========================================================================
# LabelManager.addName -- counter management.
# ===========================================================================


class TestLabelManagerAddName(unittest.TestCase):
    """``addName`` inserts at 0 on first call, then bumps by 1 per
    subsequent call. Mirror of label.hpp:54-65."""

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
        # First call set 0; remaining 4 bumps brought it to 4. The
        # counter is "occurrences MINUS one", NOT raw occurrences.
        self.assertEqual(lm._labels["foo"], 4)

    def test_distinct_names_track_independently(self):
        lm = LabelManager()
        lm.addName("foo")
        lm.addName("foo")
        lm.addName("bar")
        self.assertEqual(lm._labels, {"foo": 1, "bar": 0})


# ===========================================================================
# LabelManager.getName -- non-bumping lookup with insert-on-miss.
# ===========================================================================


class TestLabelManagerGetName(unittest.TestCase):
    """``getName`` returns ``name`` (count 0) or ``name_<count>``,
    inserts on miss but does NOT bump. Mirror of label.hpp:67-74."""

    def test_first_call_inserts_at_zero_and_returns_bare_name(self):
        lm = LabelManager()
        self.assertEqual(lm.getName("foo"), "foo")
        # Side effect: the missing-key insert MUST persist.
        self.assertEqual(lm._labels, {"foo": 0})

    def test_idempotent_for_existing_name(self):
        # Calling ``getName`` twice must NOT bump the counter -- it's
        # the non-bumping accessor. ``getNameInc`` is the bumping
        # variant.
        lm = LabelManager()
        lm.getName("foo")
        lm.getName("foo")
        self.assertEqual(lm._labels["foo"], 0)

    def test_returns_suffixed_name_when_count_nonzero(self):
        lm = LabelManager()
        lm.addName("foo")
        lm.addName("foo")  # count is now 1
        self.assertEqual(lm.getName("foo"), "foo_1")
        # Another addName brings count to 2.
        lm.addName("foo")
        self.assertEqual(lm.getName("foo"), "foo_2")


# ===========================================================================
# LabelManager.getNameInc -- bumping accessor.
# ===========================================================================


class TestLabelManagerGetNameInc(unittest.TestCase):
    """``getNameInc`` bumps the counter then returns the qualified
    name. Mirror of label.hpp:76-82."""

    def test_first_call_returns_bare_name(self):
        # ``addName`` on a fresh name sets count to 0 -> returns the
        # bare name.
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
        # ``getName`` inserts at 0 (returns bare); a subsequent
        # ``getNameInc`` bumps to 1 -- the *first* getNameInc on a
        # name already-known-to-getName therefore returns ``foo_1``.
        # This is the asymmetry between the two accessors.
        lm = LabelManager()
        lm.getName("foo")
        self.assertEqual(lm.getNameInc("foo"), "foo_1")


# ===========================================================================
# LabelManager.getNameIndex -- specific-occurrence accessor.
# ===========================================================================


class TestLabelManagerGetNameIndex(unittest.TestCase):
    """``getNameIndex`` returns the qualified name for a specific
    occurrence index, raising ``RuntimeError`` (matching C++
    ``std::runtime_error``) on missing-name or out-of-range index.
    Mirror of label.hpp:84-102."""

    def test_index_zero_returns_bare_name(self):
        lm = LabelManager()
        lm.addName("foo")
        self.assertEqual(lm.getNameIndex("foo", 0), "foo")

    def test_index_within_range_returns_suffixed_name(self):
        lm = LabelManager()
        for _ in range(3):
            lm.addName("foo")  # final count: 2
        self.assertEqual(lm.getNameIndex("foo", 1), "foo_1")
        self.assertEqual(lm.getNameIndex("foo", 2), "foo_2")

    def test_missing_name_raises_runtime_error(self):
        lm = LabelManager()
        with self.assertRaises(RuntimeError) as cm:
            lm.getNameIndex("never_added", 0)
        # Byte-for-byte text parity with rocisa's exception message
        # so any consumer that string-matches on it keeps working.
        self.assertIn("You have to add a label first", str(cm.exception))

    def test_index_exceeds_count_raises_runtime_error(self):
        lm = LabelManager()
        lm.addName("foo")  # count is 0
        with self.assertRaises(RuntimeError) as cm:
            lm.getNameIndex("foo", 1)
        self.assertIn("The index 1 exceeded.", str(cm.exception))
        self.assertIn("(> 0)", str(cm.exception))

    def test_getNameIndex_does_not_mutate_state(self):
        # The method is a pure read -- it must not mutate the
        # counter (unlike ``addName`` / ``getNameInc``).
        lm = LabelManager()
        lm.addName("foo")
        lm.addName("foo")  # count 1
        before = dict(lm._labels)
        lm.getNameIndex("foo", 1)
        self.assertEqual(lm._labels, before)


# ===========================================================================
# LabelManager.getUniqueName -- magicGenerator-seeded unique names.
# ===========================================================================


class TestLabelManagerGetUniqueName(unittest.TestCase):
    """``getUniqueName`` loops on ``magicGenerator`` until the
    candidate is absent from the registry, then routes through
    ``getName`` (which inserts at count 0 and returns the bare
    candidate). Mirror of label.hpp:104-114."""

    def test_returns_a_16_char_name(self):
        lm = LabelManager()
        name = lm.getUniqueName()
        self.assertEqual(len(name), 16)

    def test_inserts_into_registry(self):
        # The returned candidate goes through ``getName`` which
        # inserts at 0 -- so the registry must contain it post-call.
        lm = LabelManager()
        name = lm.getUniqueName()
        self.assertIn(name, lm._labels)
        self.assertEqual(lm._labels[name], 0)

    def test_loops_past_collisions(self):
        # Pin ``magicGenerator`` to return a sequence where the
        # first two outputs are already in the registry; the third
        # is fresh. ``getUniqueName`` must skip the dupes and pick
        # the third.
        lm = LabelManager()
        lm.addName("AAA")
        lm.addName("BBB")
        outputs = iter(["AAA", "BBB", "CCC"])
        with mock.patch.object(_label, "magicGenerator", lambda: next(outputs)):
            name = lm.getUniqueName()
        self.assertEqual(name, "CCC")
        # ``CCC`` is now tracked at count 0; ``AAA`` / ``BBB`` were
        # untouched (still at 0 from the addName calls above).
        self.assertEqual(lm._labels["CCC"], 0)
        self.assertEqual(lm._labels["AAA"], 0)


# ===========================================================================
# LabelManager.getUniqueNamePrefix -- prefix-scoped unique names.
# ===========================================================================


class TestLabelManagerGetUniqueNamePrefix(unittest.TestCase):
    """``getUniqueNamePrefix(prefix)`` is the readable variant of
    ``getUniqueName`` -- builds ``<prefix>_<magic>`` candidates.
    Mirror of label.hpp:116-126."""

    def test_starts_with_prefix(self):
        lm = LabelManager()
        name = lm.getUniqueNamePrefix("loop")
        # Length: 4 (prefix) + 1 (underscore) + 16 (magic) = 21.
        self.assertEqual(len(name), 21)
        self.assertTrue(name.startswith("loop_"))

    def test_inserts_into_registry(self):
        lm = LabelManager()
        name = lm.getUniqueNamePrefix("loop")
        self.assertIn(name, lm._labels)
        self.assertEqual(lm._labels[name], 0)

    def test_loops_past_collisions(self):
        # Same shape as TestLabelManagerGetUniqueName.test_loops_
        # past_collisions but with a prefixed candidate space.
        lm = LabelManager()
        lm.addName("loop_AAA")
        outputs = iter(["AAA", "BBB"])
        with mock.patch.object(_label, "magicGenerator", lambda: next(outputs)):
            name = lm.getUniqueNamePrefix("loop")
        self.assertEqual(name, "loop_BBB")


# ===========================================================================
# LabelManager.__deepcopy__ / pickle round-trip.
# ===========================================================================


class TestLabelManagerDeepCopy(unittest.TestCase):
    """``copy.deepcopy(lm)`` produces an isolated clone seeded with a
    COPY of the underlying map. Mirror of label.cpp:54-58."""

    def test_deepcopy_preserves_counters(self):
        lm = LabelManager()
        lm.addName("foo")
        lm.addName("foo")  # count 1
        lm.addName("bar")  # count 0
        clone = copy.deepcopy(lm)
        self.assertEqual(clone._labels, {"foo": 1, "bar": 0})

    def test_deepcopy_is_independent(self):
        lm = LabelManager()
        lm.addName("foo")
        clone = copy.deepcopy(lm)
        # Mutate clone; original must be untouched.
        clone.addName("foo")
        clone.addName("baz")
        self.assertEqual(lm._labels, {"foo": 0})
        self.assertEqual(clone._labels, {"foo": 1, "baz": 0})

    def test_deepcopy_empty_manager(self):
        clone = copy.deepcopy(LabelManager())
        self.assertEqual(clone._labels, {})


class TestLabelManagerPickle(unittest.TestCase):
    """Pickle round-trip uses the 1-tuple ``(dict,)`` shape from
    rocisa's ``__getstate__`` / ``__setstate__`` (label.cpp:59-64)."""

    def test_pickle_roundtrip_preserves_counters(self):
        lm = LabelManager()
        lm.addName("foo")
        lm.addName("foo")
        lm.addName("bar")
        clone = pickle.loads(pickle.dumps(lm))
        self.assertEqual(clone._labels, lm._labels)

    def test_pickle_empty_manager(self):
        clone = pickle.loads(pickle.dumps(LabelManager()))
        self.assertEqual(clone._labels, {})

    def test_pickle_state_is_isolated(self):
        # ``__getstate__`` must copy the inner dict so post-pickle
        # mutation of the live manager does NOT leak into the
        # already-serialised bytes.
        lm = LabelManager()
        lm.addName("foo")
        state = lm.__getstate__()
        lm.addName("foo")  # bump live to 1
        # State tuple must still reflect the pre-bump snapshot.
        self.assertEqual(state, ({"foo": 0},))

    def test_pickle_roundtrip_independent(self):
        lm = LabelManager()
        lm.addName("foo")
        clone = pickle.loads(pickle.dumps(lm))
        clone.addName("foo")
        # Bumping the clone must NOT affect the original (the inner
        # dict was copied during __setstate__).
        self.assertEqual(lm._labels, {"foo": 0})
        self.assertEqual(clone._labels, {"foo": 1})


# ===========================================================================
# LabelManager misc -- repr, no public getData, public-API shape.
# ===========================================================================


class TestLabelManagerRepr(unittest.TestCase):
    """``repr`` is adaptor-only convenience (rocisa has none) but it's
    additive -- exposing a Python repr does not change observable
    behaviour against rocisa."""

    def test_repr_contains_class_name_and_state(self):
        lm = LabelManager()
        lm.addName("foo")
        r = repr(lm)
        self.assertIn("LabelManager", r)
        self.assertIn("foo", r)


class TestLabelManagerPublicSurface(unittest.TestCase):
    """The public Python API matches what rocisa's nanobind binding
    exposes (label.cpp:38-64). Anything beyond is a parity break."""

    def test_no_public_getData_method(self):
        # rocisa intentionally does NOT bind ``getData`` to Python --
        # it's only used internally by ``__deepcopy__`` /
        # ``__getstate__``. We must mirror that by not having a
        # public ``getData`` either.
        self.assertFalse(hasattr(LabelManager(), "getData"))

    def test_no_dict_taking_ctor(self):
        # rocisa's nanobind binding only exposes ``nb::init<>()`` (no
        # arg). The dict-taking ctor is internal. ``LabelManager``
        # must reject any positional arg.
        with self.assertRaises(TypeError):
            LabelManager({"foo": 0})


# ===========================================================================
# Integration scenarios -- representative KernelWriter usage shapes.
# ===========================================================================


class TestLabelManagerScenarios(unittest.TestCase):
    """End-to-end scenarios that mirror typical KernelWriter call
    sequences -- exercises the accessors in combination."""

    def test_loop_label_collision_disambiguation(self):
        # Two nested loops both want "LoopTop"; the manager
        # disambiguates the second occurrence by suffix.
        lm = LabelManager()
        outer = lm.getNameInc("LoopTop")
        inner = lm.getNameInc("LoopTop")
        self.assertEqual(outer, "LoopTop")
        self.assertEqual(inner, "LoopTop_1")

    def test_get_name_then_get_name_index_round_trip(self):
        # Caller emits 3 occurrences of "foo" then asks for each by
        # index -- the i-th occurrence must round-trip via
        # ``getNameIndex(name, i)``.
        lm = LabelManager()
        emitted = [lm.getNameInc("foo") for _ in range(3)]
        self.assertEqual(emitted, ["foo", "foo_1", "foo_2"])
        for i, expected in enumerate(emitted):
            self.assertEqual(lm.getNameIndex("foo", i), expected)

    def test_deepcopy_after_use_independent(self):
        # Real-world parallel-map worker: clone a populated manager,
        # both sides keep using their own copy. Mutations must NOT
        # cross.
        lm = LabelManager()
        for _ in range(3):
            lm.addName("base")  # count 2
        worker = copy.deepcopy(lm)
        worker.addName("base")  # worker count 3
        worker.addName("worker_only")  # worker count 0
        self.assertEqual(lm._labels, {"base": 2})
        self.assertEqual(worker._labels, {"base": 3, "worker_only": 0})


if __name__ == "__main__":
    unittest.main()
