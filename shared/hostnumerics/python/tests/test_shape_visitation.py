# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import ctypes
import itertools
import unittest

import numpy as np
import hostnumerics as hn

SIZE_MAX = (1 << (8 * ctypes.sizeof(ctypes.c_size_t))) - 1


class ShapeVisitationTests(unittest.TestCase):
    def test_required_range_fields_and_order(self):
        with self.assertRaises(TypeError):
            hn.IndexRange(0, 2)
        with self.assertRaises(TypeError):
            hn.Shape([2]).selected_coordinates(hn.IndexRange.all())
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
                indices = set(flat)
                interval = hn.IndexRange(first, stride, limit)
                self.assertEqual(interval.indices(shape.element_count), flat)
                self.assertEqual(interval.count(shape.element_count), len(flat))
                for positions in (interval, indices):
                    self.assertEqual(
                        shape.selected_coordinates(positions, order), expected
                    )

    def test_sets_and_bounds(self):
        shape = hn.Shape([2, 4])
        self.assertEqual(
            shape.selected_coordinates({7, 0, 3, 7}, [0, 1]), [[0, 0], [0, 3], [1, 3]]
        )
        with self.assertRaisesRegex(
            RuntimeError, "index exceeds logical element count"
        ):
            shape.selected_coordinates({0, 8}, [0, 1])
        for bad in ([0, 0], [1], [0, 2]):
            with self.assertRaises(RuntimeError):
                shape.selected_coordinates(set(), bad)
            with self.assertRaises(RuntimeError):
                shape.selected_coordinates(hn.IndexRange(0, 1, 0), bad)
        for empty_shape in (hn.Shape(), hn.Shape([2, 0])):
            with self.assertRaisesRegex(RuntimeError, "rank"):
                empty_shape.selected_coordinates(set(), [0])

    def test_all_is_full_coverage(self):
        shape = hn.Shape([2, 4])
        for interval in (hn.IndexRange.all(), hn.IndexRange(0, 1, 8)):
            self.assertEqual(interval.count(shape.element_count), shape.element_count)
        self.assertEqual(hn.IndexRange(0, 1, 7).count(shape.element_count), 7)
        for dimensions, expected in (([], [[]]), ([0], [])):
            shape = hn.Shape(dimensions)
            self.assertEqual(
                shape.selected_coordinates_last_dimension_fastest(hn.IndexRange.all()),
                expected,
            )
            self.assertEqual(
                shape.index_order_last_dimension_fastest(), list(range(len(dimensions)))
            )

    def test_range_counts_and_size_t_boundaries(self):
        for total in range(11):
            for first, stride, limit in itertools.product(
                (0, 2, total, total + 1), (1, 2, 3, 11), (0, 1, 2, 20)
            ):
                interval = hn.IndexRange(first, stride, limit)
                expected = list(range(first, total, stride))[:limit]
                self.assertEqual(interval.indices(total), expected)
                self.assertEqual(interval.count(total), len(expected))
        self.assertEqual(
            hn.IndexRange(1, SIZE_MAX - 2, 2).indices(SIZE_MAX), [1, SIZE_MAX - 1]
        )
        self.assertEqual(hn.IndexRange.all().count(SIZE_MAX), SIZE_MAX)
        for value in (-1, SIZE_MAX + 1):
            with self.assertRaises(TypeError):
                hn.IndexRange(value, 1, 1)
            with self.assertRaises(TypeError):
                hn.Shape([1]).selected_coordinates({value}, [0])
            with self.assertRaises(TypeError):
                hn.IndexRange.all().count(value)

    def test_inherited_prime_sampling(self):
        for logical, allocated, requested, stride in (
            (10, 10, 3, 3),
            (20, 9, 1, 11),
            (60, 49, 1, 53),
            (128, 121, 1, 127),
            (20, 0, 3, 2),
        ):
            interval = hn.prime_index_range(logical, allocated, requested)
            self.assertEqual(interval.stride, stride)
            self.assertEqual(
                interval.indices(logical),
                list(range(0, logical, stride)),
            )
        for requested in (0, 10, 11):
            self.assertEqual(
                hn.prime_index_range(10, SIZE_MAX, requested).count(10), 10
            )
        with self.assertRaisesRegex(RuntimeError, "Prime-stride"):
            hn.prime_index_range(2, SIZE_MAX, 1)


if __name__ == "__main__":
    unittest.main()
