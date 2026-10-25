# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import ctypes
import unittest

import hostnumerics as hn


SIZE_MAX = (1 << (8 * ctypes.sizeof(ctypes.c_size_t))) - 1


class IndexSelectionTests(unittest.TestCase):
    def assert_selection(self, selection, logical_elements, expected):
        self.assertEqual(selection.indices(logical_elements), expected)
        self.assertEqual(selection.selected_count(logical_elements), len(expected))

    def test_factory_choices_are_required(self):
        for factory in (
            lambda: hn.IndexSelection.all(),
            lambda: hn.IndexSelection.strided(0, 2, index_order=[0]),
            lambda: hn.IndexSelection.strided(0, 2, max_elements=SIZE_MAX),
            lambda: hn.IndexSelection.explicit_indices([1]),
            lambda: hn.IndexSelection.prime_stride(10, 10, 3),
        ):
            with self.assertRaises(TypeError):
                factory()

    def test_all_and_empty_indices(self):
        selection = hn.IndexSelection.all(index_order=[0])
        self.assertEqual(selection.kind, hn.IndexSelectionKind.All)
        self.assertTrue(selection.selects_all)
        self.assertEqual(selection.index_order(1), [0])
        for logical in (0, 1, 7):
            self.assert_selection(selection, logical, list(range(logical)))
        self.assertEqual(selection.selected_count(SIZE_MAX), SIZE_MAX)

    def test_strided_and_limited_indices(self):
        selection = hn.IndexSelection.strided(2, 3, max_elements=2, index_order=[0])
        self.assertEqual(selection.kind, hn.IndexSelectionKind.Strided)
        self.assertFalse(selection.selects_all)
        self.assertEqual(
            (selection.first, selection.stride, selection.max_elements), (2, 3, 2)
        )
        for logical in range(11):
            for first in (0, 2, logical, logical + 1):
                for stride in (1, 2, 3, 11):
                    expected = list(range(first, logical, stride))
                    self.assert_selection(
                        hn.IndexSelection.strided(
                            first, stride, max_elements=SIZE_MAX, index_order=[0]
                        ),
                        logical,
                        expected,
                    )
                    for limit in (0, 1, 2, 20):
                        self.assert_selection(
                            hn.IndexSelection.strided(
                                first, stride, limit, index_order=[0]
                            ),
                            logical,
                            expected[:limit],
                        )
        with self.assertRaisesRegex(RuntimeError, "stride must be nonzero"):
            hn.IndexSelection.strided(0, 0, max_elements=0, index_order=[0])

    def test_explicit_normalization_and_bounds(self):
        source = [7, 0, 3, 7, 3]
        selection = hn.IndexSelection.explicit_indices(source, index_order=[0])
        source[:] = [99]
        self.assertEqual(selection.kind, hn.IndexSelectionKind.Explicit)
        self.assertFalse(selection.selects_all)
        self.assert_selection(selection, 8, [0, 3, 7])
        returned = selection.indices(8)
        returned[:] = [99]
        self.assert_selection(selection, 12, [0, 3, 7])
        for logical in (0, 3, 7):
            for query in (selection.indices, selection.selected_count):
                with self.assertRaisesRegex(
                    RuntimeError, "index exceeds logical element count"
                ):
                    query(logical)
        empty = hn.IndexSelection.explicit_indices([], index_order=[0])
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
            selection = hn.IndexSelection.prime_stride(
                logical, allocated, requested, index_order=[0]
            )
            self.assertEqual(selection.stride, stride)
            self.assert_selection(selection, logical, list(range(0, logical, stride)))
        self.assertEqual(
            hn.IndexSelection.prime_stride(10, 10, 3, index_order=[0]).selected_count(
                10
            ),
            4,
        )

    def test_prime_zero_request_and_all_indices(self):
        for logical, requested in ((10, 0), (10, 10), (10, 11), (0, 0), (0, 1)):
            selection = hn.IndexSelection.prime_stride(
                logical, SIZE_MAX, requested, index_order=[0]
            )
            self.assertTrue(selection.selects_all)
            self.assert_selection(selection, logical, list(range(logical)))

    def test_size_t_boundaries_and_overflow(self):
        self.assert_selection(
            hn.IndexSelection.strided(
                SIZE_MAX - 2, 3, max_elements=SIZE_MAX, index_order=[0]
            ),
            SIZE_MAX,
            [SIZE_MAX - 2],
        )
        self.assert_selection(
            hn.IndexSelection.strided(
                1, SIZE_MAX - 2, max_elements=SIZE_MAX, index_order=[0]
            ),
            SIZE_MAX,
            [1, SIZE_MAX - 1],
        )
        self.assert_selection(
            hn.IndexSelection.strided(
                SIZE_MAX, 1, max_elements=SIZE_MAX, index_order=[0]
            ),
            SIZE_MAX,
            [],
        )
        self.assertEqual(
            hn.IndexSelection.strided(
                0, 1, max_elements=SIZE_MAX, index_order=[0]
            ).selected_count(SIZE_MAX),
            SIZE_MAX,
        )
        invalid = hn.IndexSelection.explicit_indices([SIZE_MAX], index_order=[0])
        for query in (invalid.indices, invalid.selected_count):
            with self.assertRaises(RuntimeError):
                query(SIZE_MAX)
        with self.assertRaisesRegex(
            RuntimeError, "Prime-stride index selection overflow"
        ):
            hn.IndexSelection.prime_stride(2, SIZE_MAX, 1, index_order=[0])

    def test_selections_own_order_sequences(self):
        for source in ([2, 1, 0], (2, 1, 0)):
            selections = (
                hn.IndexSelection.all(source),
                hn.IndexSelection.strided(
                    0, 2, index_order=source, max_elements=SIZE_MAX
                ),
                hn.IndexSelection.explicit_indices([1, 7], source),
                hn.IndexSelection.prime_stride(24, 24, 5, source),
                hn.IndexSelection.prime_stride(24, 24, 0, source),
            )
            if isinstance(source, list):
                source[:] = [0, 1, 2]
            del source
            for selection in selections:
                self.assertEqual(selection.index_order(3), [2, 1, 0])
                returned_order = selection.index_order(3)
                returned_order[:] = [0, 1, 2]
                self.assertEqual(selection.index_order(3), [2, 1, 0])

    def test_strict_and_named_orders(self):
        with self.assertRaises(TypeError):
            hn.IndexSelection()
        scalar = hn.IndexSelection.all([])
        self.assertEqual(scalar.index_order(0), [])
        with self.assertRaisesRegex(RuntimeError, "rank does not match"):
            scalar.index_order(2)
        for order in ([0, 0], [1], [0, 2]):
            for factory in (
                lambda: hn.IndexSelection.all(order),
                lambda: hn.IndexSelection.strided(0, 2, 3, order),
                lambda: hn.IndexSelection.explicit_indices([1], order),
                lambda: hn.IndexSelection.prime_stride(6, 6, 0, order),
            ):
                with self.assertRaisesRegex(RuntimeError, "permutation"):
                    factory()
        for selection in (
            hn.IndexSelection.all_last_dimension_fastest(),
            hn.IndexSelection.strided_last_dimension_fastest(0, 2, 3),
            hn.IndexSelection.explicit_indices_last_dimension_fastest([1]),
            hn.IndexSelection.prime_stride_last_dimension_fastest(6, 6, 2),
        ):
            self.assertEqual(selection.index_order(0), [])
            self.assertEqual(selection.index_order(2), [0, 1])
            self.assertEqual(selection.index_order(3), [0, 1, 2])

    def test_non_size_t_arguments_are_rejected(self):
        for value in (-1, SIZE_MAX + 1):
            for factory in (
                lambda: hn.IndexSelection.strided(
                    value, 1, max_elements=SIZE_MAX, index_order=[0]
                ),
                lambda: hn.IndexSelection.strided(
                    0, value, max_elements=SIZE_MAX, index_order=[0]
                ),
                lambda: hn.IndexSelection.explicit_indices([value], index_order=[0]),
                lambda: hn.IndexSelection.all([value]),
                lambda: hn.IndexSelection.prime_stride(10, value, 1, index_order=[0]),
            ):
                with self.assertRaises(TypeError):
                    factory()
            for query in (
                hn.IndexSelection.all(index_order=[0]).indices,
                hn.IndexSelection.all(index_order=[0]).selected_count,
            ):
                with self.assertRaises(TypeError):
                    query(value)


if __name__ == "__main__":
    unittest.main()
