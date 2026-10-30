# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import ctypes
import itertools
import sys
import unittest

import numpy as np
import hostnumerics as hn

SIZE_MAX = (1 << (8 * ctypes.sizeof(ctypes.c_size_t))) - 1


class ShapeVisitationTests(unittest.TestCase):
    def test_required_range_fields_and_order(self):
        with self.assertRaises(TypeError):
            hn.IndexRange(0, 2)
        with self.assertRaises(TypeError):
            hn.IndexRange.all()
        with self.assertRaises(TypeError):
            hn.Shape([2]).selected_coordinates(hn.IndexRange.all(2))
        with self.assertRaisesRegex(RuntimeError, "stride must be nonzero"):
            hn.IndexRange(0, 3, 0)
        interval = hn.IndexRange(start=1, stop=7, stride=2)
        self.assertEqual((interval.start, interval.stop, interval.stride), (1, 7, 2))
        self.assertEqual(interval.indices(), [1, 3, 5])

    def test_coordinates_against_numpy(self):
        dimensions = (2, 3, 4)
        shape = hn.Shape(dimensions)
        for order in itertools.permutations(range(3)):
            for start, end, stride in (
                (0, 24, 1),
                (1, 16, 3),
                (2, 2, 5),
                (24, 30, 1),
                (1, 31, 4),
            ):
                full_range = np.arange(start, end, stride).tolist()
                flat = [index for index in full_range if index < shape.element_count]
                expected = []
                for index in flat:
                    ordered = np.unravel_index(
                        index, tuple(dimensions[a] for a in order)
                    )
                    point = [0] * 3
                    for axis, coordinate in zip(order, ordered):
                        point[axis] = int(coordinate)
                    expected.append(point)
                interval = hn.IndexRange(start, end, stride)
                self.assertEqual(interval.indices(), full_range)
                self.assertEqual(interval.count(), len(full_range))
                self.assertEqual(
                    shape.selected_coordinates(flat, hn.Permutation(order)), expected
                )
                if end <= shape.element_count:
                    self.assertEqual(
                        shape.selected_coordinates(interval, hn.Permutation(order)),
                        expected,
                    )
                else:
                    with self.assertRaisesRegex(
                        RuntimeError, "exceeds the element count"
                    ):
                        shape.selected_coordinates(interval, hn.Permutation(order))

    def test_explicit_indices_and_bounds(self):
        shape = hn.Shape([2, 4])
        self.assertEqual(
            shape.selected_coordinates([7, 0, 3, 7], hn.Permutation([0, 1])),
            [[1, 3], [0, 0], [0, 3], [1, 3]],
        )
        self.assertEqual(
            shape.selected_coordinates((7, 0), hn.Permutation([1, 0])), [[1, 3], [0, 0]]
        )
        with self.assertRaisesRegex(RuntimeError, r"index 8 is outside \[0, 8\)"):
            shape.selected_coordinates([0, 8, 1], hn.Permutation([0, 1]))
        for bad in ([0, 0], [1], [0, 2]):
            with self.assertRaises(RuntimeError):
                shape.selected_coordinates([], hn.Permutation(bad))
            with self.assertRaises(RuntimeError):
                shape.selected_coordinates(hn.IndexRange(0, 0, 1), hn.Permutation(bad))
        # Orders must be Permutations, so a list of indices is never read as one.
        with self.assertRaises(TypeError):
            shape.selected_coordinates([1, 0], [0, 1])
        for empty_shape in (hn.Shape(), hn.Shape([2, 0])):
            with self.assertRaisesRegex(RuntimeError, "rank"):
                empty_shape.selected_coordinates([], hn.Permutation([0]))

    def test_all_is_full_coverage(self):
        shape = hn.Shape([2, 4])
        for interval in (
            hn.IndexRange.all(shape.element_count),
            hn.IndexRange(0, 8, 1),
        ):
            self.assertEqual(interval.count(), shape.element_count)
        self.assertEqual(hn.IndexRange(0, 7, 1).count(), 7)
        for dimensions, expected in (([], [[]]), ([0], [])):
            shape = hn.Shape(dimensions)
            self.assertEqual(
                shape.selected_coordinates(
                    hn.IndexRange.all(shape.element_count),
                    hn.Permutation.identity(shape.rank),
                ),
                expected,
            )

    def test_range_counts_and_size_t_boundaries(self):
        for end in range(11):
            for start, stride in itertools.product((0, 2, end, end + 1), (1, 2, 3, 11)):
                interval = hn.IndexRange(start, end, stride)
                expected = list(range(start, end, stride))
                self.assertEqual(interval.indices(), expected)
                self.assertEqual(interval.count(), len(expected))
        self.assertEqual(
            hn.IndexRange(1, SIZE_MAX, SIZE_MAX - 2).indices(), [1, SIZE_MAX - 1]
        )
        self.assertEqual(hn.IndexRange.all(SIZE_MAX).count(), SIZE_MAX)
        for value in (-1, SIZE_MAX + 1):
            for args in ((value, 1, 1), (0, value, 1), (0, 1, value)):
                with self.assertRaises(TypeError):
                    hn.IndexRange(*args)
            with self.assertRaises(TypeError):
                hn.Shape([1]).selected_coordinates([value], hn.Permutation([0]))
            with self.assertRaises(TypeError):
                hn.IndexRange.all(value)

    def test_prime_sampling(self):
        def next_prime(value):
            candidate = max(value, 2)
            while any(candidate % d == 0 for d in range(2, int(candidate**0.5) + 1)):
                candidate += 1
            return candidate

        for logical in range(1, 200):
            for requested in range(1, logical):
                stride = next_prime(logical // requested)
                interval = hn.prime_index_range(
                    logical_elements=logical, requested_elements=requested
                )
                self.assertEqual(
                    (interval.start, interval.stop, interval.stride),
                    (0, logical, stride),
                )
        self.assertEqual(
            hn.prime_index_range(logical_elements=10, requested_elements=0).count(), 0
        )
        for requested in (10, 11):
            self.assertEqual(
                hn.prime_index_range(
                    logical_elements=10, requested_elements=requested
                ).count(),
                10,
            )
        with self.assertRaisesRegex(RuntimeError, "Prime-stride"):
            hn.prime_index_range(logical_elements=SIZE_MAX, requested_elements=1)
        with self.assertRaises(TypeError):
            hn.prime_index_range(10, 3)

    def test_ranges_are_values(self):
        interval = hn.IndexRange(1, 8, 3)
        self.assertEqual(list(interval), [1, 4, 7])
        self.assertEqual(len(interval), 3)
        self.assertEqual(len(hn.IndexRange(5, 5, 1)), 0)
        self.assertEqual(repr(interval), "IndexRange(start=1, stop=8, stride=3)")
        self.assertEqual(interval, hn.IndexRange(1, 8, 3))
        self.assertNotEqual(interval, hn.IndexRange(1, 9, 3))
        self.assertEqual(len({interval, hn.IndexRange(1, 8, 3)}), 1)
        self.assertEqual(list(hn.IndexRange(SIZE_MAX - 2, SIZE_MAX, 2)), [SIZE_MAX - 2])

    def test_len_is_limited_like_python_range(self):
        self.assertEqual(len(hn.IndexRange.all(sys.maxsize)), sys.maxsize)
        for stop in (sys.maxsize + 1, SIZE_MAX):
            with self.assertRaises(OverflowError):
                len(range(stop))
            interval = hn.IndexRange.all(stop)
            with self.assertRaises(OverflowError):
                len(interval)
            self.assertEqual(interval.count(), stop)


if __name__ == "__main__":
    unittest.main()
