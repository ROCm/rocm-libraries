# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import ctypes
import itertools
import unittest

import numpy as np
import hostnumerics as hn

SIZE_MAX = (1 << (8 * ctypes.sizeof(ctypes.c_size_t))) - 1


class IndexSelectionTests(unittest.TestCase):
    def test_required_range_fields_and_order(self):
        for construct in (
            lambda: hn.IndexRange(0, 2),
            lambda: hn.IndexSelection(hn.IndexRange.all()),
            lambda: hn.IndexSelection([1]),
        ):
            with self.assertRaises(TypeError):
                construct()
        with self.assertRaisesRegex(RuntimeError, "stride must be nonzero"):
            hn.IndexRange(0, 0, 3)

    def test_coordinates_against_numpy(self):
        dimensions = (2, 3, 4)
        shape = hn.Shape(dimensions)
        for order in itertools.permutations(range(3)):
            for first, stride, limit in (
                (0, 1, SIZE_MAX),
                (1, 3, 5),
                (2, 5, 0),
                (24, 1, 2),
            ):
                flat = list(range(first, 24, stride))[:limit]
                expected = []
                for index in flat:
                    ordered = np.unravel_index(
                        index, tuple(dimensions[a] for a in order)
                    )
                    point = [0] * 3
                    for axis, coordinate in zip(order, ordered):
                        point[axis] = int(coordinate)
                    expected.append(point)
                for indices in (hn.IndexRange(first, stride, limit), flat[::-1] + flat):
                    selection = hn.IndexSelection(indices, order)
                    self.assertEqual(shape.selected_indices(selection), flat)
                    self.assertEqual(shape.selected_count(selection), len(flat))
                    self.assertEqual(shape.selected_coordinates(selection), expected)

    def test_normalization_ownership_and_bounds(self):
        indices, order = [7, 0, 3, 7], [0, 1]
        selection = hn.IndexSelection(indices, order)
        indices[:] = [99]
        order[:] = [1, 0]
        shape = hn.Shape([2, 4])
        self.assertEqual(shape.selected_indices(selection), [0, 3, 7])
        self.assertEqual(shape.index_order(selection), [0, 1])
        invalid = hn.IndexSelection([8], [0, 1])
        for query in (
            shape.selected_indices,
            shape.selected_count,
            shape.selected_coordinates,
            shape.selects_all,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "index exceeds logical element count"
            ):
                query(invalid)
        for bad in ([0, 0], [1], [0, 2]):
            with self.assertRaisesRegex(RuntimeError, "permutation"):
                hn.IndexSelection([], bad)
        for empty_shape in (hn.Shape(), hn.Shape([2, 0])):
            wrong_rank = hn.IndexSelection([], [0])
            with self.assertRaisesRegex(RuntimeError, "rank does not match"):
                empty_shape.selected_coordinates(wrong_rank)

    def test_all_is_full_coverage(self):
        shape = hn.Shape([2, 4])
        for indices in (hn.IndexRange.all(), hn.IndexRange(0, 1, 8), list(range(8))):
            self.assertTrue(
                shape.selects_all(hn.IndexSelection(indices, hn.LastDimensionFastest()))
            )
        self.assertFalse(
            shape.selects_all(hn.IndexSelection(hn.IndexRange(0, 1, 7), [0, 1]))
        )
        for dimensions, expected in (([], [[]]), ([0], [])):
            shape = hn.Shape(dimensions)
            selection = hn.IndexSelection(
                hn.IndexRange.all(), hn.LastDimensionFastest()
            )
            self.assertEqual(shape.selected_coordinates(selection), expected)

    def test_range_counts_and_size_t_boundaries(self):
        for total in range(11):
            shape = hn.Shape([total])
            for first, stride, limit in itertools.product(
                (0, 2, total, total + 1), (1, 2, 3, 11), (0, 1, 2, 20)
            ):
                selection = hn.IndexSelection(hn.IndexRange(first, stride, limit), [0])
                expected = list(range(first, total, stride))[:limit]
                self.assertEqual(shape.selected_indices(selection), expected)
                self.assertEqual(shape.selected_count(selection), len(expected))
        shape = hn.Shape([SIZE_MAX])
        selection = hn.IndexSelection(hn.IndexRange(1, SIZE_MAX - 2, 2), [0])
        self.assertEqual(shape.selected_indices(selection), [1, SIZE_MAX - 1])
        self.assertEqual(
            shape.selected_count(hn.IndexSelection(hn.IndexRange.all(), [0])), SIZE_MAX
        )
        for value in (-1, SIZE_MAX + 1):
            with self.assertRaises(TypeError):
                hn.IndexRange(value, 1, 1)
            with self.assertRaises(TypeError):
                hn.IndexSelection([value], [0])

    def test_inherited_prime_sampling(self):
        for logical, allocated, requested, stride in (
            (10, 10, 3, 3),
            (20, 9, 1, 11),
            (60, 49, 1, 53),
            (128, 121, 1, 127),
            (20, 0, 3, 2),
        ):
            indices = hn.prime_index_range(logical, allocated, requested)
            self.assertEqual(indices.stride, stride)
            selection = hn.IndexSelection(indices, [0])
            self.assertEqual(
                hn.Shape([logical]).selected_indices(selection),
                list(range(0, logical, stride)),
            )
        for requested in (0, 10, 11):
            selection = hn.IndexSelection(
                hn.prime_index_range(10, SIZE_MAX, requested), [0]
            )
            self.assertTrue(hn.Shape([10]).selects_all(selection))
        with self.assertRaisesRegex(
            RuntimeError, "Prime-stride index selection overflow"
        ):
            hn.prime_index_range(2, SIZE_MAX, 1)


if __name__ == "__main__":
    unittest.main()
