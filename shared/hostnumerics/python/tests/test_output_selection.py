# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import ctypes
import unittest

import hostnumerics as hn


SIZE_MAX = (1 << (8 * ctypes.sizeof(ctypes.c_size_t))) - 1


class OutputSelectionTests(unittest.TestCase):
    def assert_selection(self, selection, logical_elements, expected):
        self.assertEqual(selection.indices(logical_elements), expected)
        self.assertEqual(selection.selected_count(logical_elements), len(expected))

    def test_all_and_empty_outputs(self):
        selection = hn.OutputSelection.all()
        self.assertEqual(selection.kind, hn.OutputSelectionKind.All)
        self.assertTrue(selection.selects_all)
        self.assertEqual(selection.index_order, [])
        for logical in (0, 1, 7):
            self.assert_selection(selection, logical, list(range(logical)))
        self.assertEqual(selection.selected_count(SIZE_MAX), SIZE_MAX)

    def test_strided_and_limited_outputs(self):
        selection = hn.OutputSelection.strided(2, 3, max_elements=2)
        self.assertEqual(selection.kind, hn.OutputSelectionKind.Strided)
        self.assertFalse(selection.selects_all)
        self.assertEqual(
            (selection.first, selection.stride, selection.max_elements), (2, 3, 2)
        )
        for logical in range(11):
            for first in (0, 2, logical, logical + 1):
                for stride in (1, 2, 3, 11):
                    expected = list(range(first, logical, stride))
                    self.assert_selection(
                        hn.OutputSelection.strided(first, stride), logical, expected
                    )
                    for limit in (0, 1, 2, 20):
                        self.assert_selection(
                            hn.OutputSelection.strided(first, stride, limit),
                            logical,
                            expected[:limit],
                        )
        with self.assertRaisesRegex(RuntimeError, "stride must be nonzero"):
            hn.OutputSelection.strided(0, 0, max_elements=0)

    def test_explicit_normalization_and_bounds(self):
        source = [7, 0, 3, 7, 3]
        selection = hn.OutputSelection.explicit_indices(source)
        source[:] = [99]
        self.assertEqual(selection.kind, hn.OutputSelectionKind.Explicit)
        self.assertFalse(selection.selects_all)
        self.assert_selection(selection, 8, [0, 3, 7])
        returned = selection.indices(8)
        returned[:] = [99]
        self.assert_selection(selection, 12, [0, 3, 7])
        for logical in (0, 3, 7):
            for query in (selection.indices, selection.selected_count):
                with self.assertRaisesRegex(RuntimeError, "index exceeds output shape"):
                    query(logical)
        empty = hn.OutputSelection.explicit_indices([])
        for logical in (0, 1, SIZE_MAX):
            self.assert_selection(empty, logical, [])

    def test_prime_sampling_is_approximate(self):
        for logical, allocated, requested, stride in (
            (10, 10, 3, 3),
            (20, 9, 1, 11),
            (60, 49, 1, 53),
            (128, 121, 1, 127),
            (20, 1, 3, 2),
            (20, 0, 3, 2),
        ):
            selection = hn.OutputSelection.prime_stride(logical, allocated, requested)
            self.assertEqual(selection.stride, stride)
            self.assert_selection(selection, logical, list(range(0, logical, stride)))
        self.assertEqual(
            hn.OutputSelection.prime_stride(10, 10, 3).selected_count(10), 4
        )

    def test_prime_zero_request_and_all_output_behavior(self):
        for logical, requested in ((10, 0), (10, 10), (10, 11), (0, 0), (0, 1)):
            selection = hn.OutputSelection.prime_stride(logical, SIZE_MAX, requested)
            self.assertTrue(selection.selects_all)
            self.assert_selection(selection, logical, list(range(logical)))

    def test_size_t_boundaries_and_overflow(self):
        self.assert_selection(
            hn.OutputSelection.strided(SIZE_MAX - 2, 3), SIZE_MAX, [SIZE_MAX - 2]
        )
        self.assert_selection(
            hn.OutputSelection.strided(1, SIZE_MAX - 2), SIZE_MAX, [1, SIZE_MAX - 1]
        )
        self.assert_selection(hn.OutputSelection.strided(SIZE_MAX, 1), SIZE_MAX, [])
        self.assertEqual(
            hn.OutputSelection.strided(0, 1).selected_count(SIZE_MAX), SIZE_MAX
        )
        invalid = hn.OutputSelection.explicit_indices([SIZE_MAX])
        for query in (invalid.indices, invalid.selected_count):
            with self.assertRaises(RuntimeError):
                query(SIZE_MAX)
        with self.assertRaisesRegex(
            RuntimeError, "Prime-stride output selection overflow"
        ):
            hn.OutputSelection.prime_stride(2, SIZE_MAX, 1)

    def test_selections_own_order_sequences(self):
        for source in ([2, 1, 0], (2, 1, 0)):
            selections = (
                hn.OutputSelection.all(source),
                hn.OutputSelection.strided(0, 2, index_order=source),
                hn.OutputSelection.explicit_indices([1, 7], source),
                hn.OutputSelection.prime_stride(24, 24, 5, source),
                hn.OutputSelection.prime_stride(24, 24, 0, source),
            )
            if isinstance(source, list):
                source[:] = [0, 1, 2]
            del source
            for selection in selections:
                self.assertEqual(selection.index_order, [2, 1, 0])
                returned_order = selection.index_order
                returned_order[:] = [0, 1, 2]
                self.assertEqual(selection.index_order, [2, 1, 0])
        self.assertEqual(hn.OutputSelection.all([7, 7]).index_order, [7, 7])

    def test_non_size_t_arguments_are_rejected(self):
        for value in (-1, SIZE_MAX + 1):
            for factory in (
                lambda: hn.OutputSelection.strided(value, 1),
                lambda: hn.OutputSelection.strided(0, value),
                lambda: hn.OutputSelection.explicit_indices([value]),
                lambda: hn.OutputSelection.all([value]),
                lambda: hn.OutputSelection.prime_stride(10, value, 1),
            ):
                with self.assertRaises(TypeError):
                    factory()
            for query in (
                hn.OutputSelection.all().indices,
                hn.OutputSelection.all().selected_count,
            ):
                with self.assertRaises(TypeError):
                    query(value)


if __name__ == "__main__":
    unittest.main()
