# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import ctypes
import itertools
import math
import unittest

import hostnumerics as hn


SIZE_MAX = (1 << (8 * ctypes.sizeof(ctypes.c_size_t))) - 1
ORDERS = (hn.IndexOrder.FirstDimensionFastest, hn.IndexOrder.LastDimensionFastest)


class ShapeTests(unittest.TestCase):
    def test_scalar_and_owned_extents(self):
        scalar = hn.Shape()
        self.assertEqual(scalar, hn.Shape([]))
        self.assertEqual(
            (scalar.rank, scalar.dimensions, scalar.element_count), (0, [], 1)
        )
        for extents in ([2, 3, 4], (2, 3, 4)):
            shape = hn.Shape(extents)
            self.assertEqual(shape.rank, 3)
            self.assertEqual(shape.dimensions, [2, 3, 4])
            self.assertEqual(shape[1], 3)
            self.assertEqual(shape.extent(2), 4)
            self.assertEqual(shape, hn.Shape([2, 3, 4]))
            self.assertNotEqual(shape, hn.Shape([2, 4, 3]))
            self.assertNotEqual(shape, None)
        source = [2, 3, 4]
        shape = hn.Shape(source)
        source[0] = 9
        returned = shape.dimensions
        returned[1] = 9
        self.assertEqual(shape.dimensions, [2, 3, 4])
        for attribute, value in (
            ("rank", 9),
            ("dimensions", [9]),
            ("element_count", 9),
        ):
            with self.assertRaises(AttributeError):
                setattr(shape, attribute, value)

    def test_products(self):
        shape = hn.Shape([2, 3, 4])
        self.assertEqual(shape.element_count, 24)
        self.assertEqual(shape.element_count_in_range(1, 3), 12)
        self.assertEqual(shape.element_count_in_range(3, 3), 1)
        self.assertEqual(shape.element_count_excluding(1), 8)
        self.assertEqual(hn.Shape([7]).element_count_excluding(0), 1)
        # Compare every selected range/exclusion against Python's exact integers.
        for dimensions in itertools.product((0, 1, 2, SIZE_MAX), repeat=3):
            shape = hn.Shape(dimensions)
            for first in range(4):
                for end in range(first, 4):
                    expected = math.prod(dimensions[first:end])
                    with self.subTest(dimensions=dimensions, first=first, end=end):
                        if expected > SIZE_MAX:
                            with self.assertRaises(OverflowError):
                                shape.element_count_in_range(first, end)
                        else:
                            self.assertEqual(
                                shape.element_count_in_range(first, end), expected
                            )
            for excluded in range(3):
                expected = math.prod(
                    d for i, d in enumerate(dimensions) if i != excluded
                )
                with self.subTest(dimensions=dimensions, excluded=excluded):
                    if expected > SIZE_MAX:
                        with self.assertRaises(OverflowError):
                            shape.element_count_excluding(excluded)
                    else:
                        self.assertEqual(
                            shape.element_count_excluding(excluded), expected
                        )

    def test_descriptive_bounds_errors(self):
        shape = hn.Shape([2, 3, 4])
        for operation in (lambda: shape.extent(3), lambda: shape[3]):
            with self.assertRaisesRegex(IndexError, r"dimension 3 is outside \[0, 3\)"):
                operation()
        for first, end, reason in (
            (2, 1, "start exceeds the end"),
            (0, 4, "end exceeds the shape rank"),
        ):
            with self.assertRaisesRegex(
                IndexError,
                rf"range \[{first}, {end}\) is invalid for rank 3: the {reason}",
            ):
                shape.element_count_in_range(first, end)
        with self.assertRaisesRegex(IndexError, r"dimension 3 is outside \[0, 3\)"):
            shape.element_count_excluding(3)

    def test_coordinate_orders(self):
        shape = hn.Shape([2, 3, 4])
        self.assertEqual(shape.linear_index([1, 0, 2], ORDERS[0]), 13)
        self.assertEqual(shape.linear_index([1, 0, 2], ORDERS[1]), 14)
        self.assertEqual(shape.coordinates(17, ORDERS[0]), [1, 2, 2])
        self.assertEqual(shape.coordinates(17, ORDERS[1]), [1, 1, 1])
        for dimensions in ([], [7], [2, 3, 4], [2, 1, 3, 2]):
            shape = hn.Shape(dimensions)
            for order in ORDERS:
                axes = range(shape.rank)
                if order == hn.IndexOrder.FirstDimensionFastest:
                    axes = reversed(axes)
                axes = list(axes)
                # Independently enumerate coordinates, rather than just round-tripping.
                positions = itertools.product(
                    *(range(dimensions[axis]) for axis in axes)
                )
                for index, position in enumerate(positions):
                    coordinates = [0] * shape.rank
                    for axis, coordinate in zip(axes, position):
                        coordinates[axis] = coordinate
                    with self.subTest(dimensions=dimensions, order=order, index=index):
                        self.assertEqual(shape.coordinates(index, order), coordinates)
                        self.assertEqual(shape.linear_index(coordinates, order), index)

    def test_coordinate_errors(self):
        shape = hn.Shape([2, 3, 4])
        for order in ORDERS:
            with self.assertRaisesRegex(
                ValueError, "coordinate rank 0 does not match shape rank 3"
            ):
                shape.linear_index([], order)
            with self.assertRaisesRegex(
                IndexError, r"coordinate 3 in dimension 1 is outside \[0, 3\)"
            ):
                shape.linear_index([1, 3, 0], order)
            with self.assertRaisesRegex(
                IndexError, r"logical index 24 is outside \[0, 24\)"
            ):
                shape.coordinates(24, order)
            with self.assertRaises(IndexError):
                hn.Shape().coordinates(1, order)
            for dimensions in (
                [0],
                [0, SIZE_MAX, 2],
                [SIZE_MAX, 0, 2],
                [SIZE_MAX, 2, 0],
            ):
                empty = hn.Shape(dimensions)
                self.assertEqual(empty.element_count, 0)
                with self.assertRaises(IndexError):
                    empty.coordinates(0, order)
                with self.assertRaises(IndexError):
                    empty.linear_index([0] * empty.rank, order)

    def test_overflow(self):
        self.assertEqual(hn.Shape([SIZE_MAX, 1]).element_count, SIZE_MAX)
        for dimensions in ([SIZE_MAX, 2], [SIZE_MAX, 2, 1, 1]):
            with self.assertRaisesRegex(OverflowError, "element count exceeds size_t"):
                _ = hn.Shape(dimensions).element_count
        oversized = hn.Shape([SIZE_MAX, 2])
        self.assertEqual(oversized.linear_index([0, 1], ORDERS[1]), 1)
        for order in ORDERS:
            with self.assertRaises(OverflowError):
                oversized.linear_index([SIZE_MAX - 1, 1], order)
            with self.assertRaises(OverflowError):
                oversized.coordinates(0, order)

    def test_invalid_python_inputs(self):
        for dimensions in ([-1], [SIZE_MAX + 1], [1.5], 3):
            with self.subTest(dimensions=dimensions), self.assertRaises(TypeError):
                hn.Shape(dimensions)
        shape = hn.Shape([2])
        for index in (-1, SIZE_MAX + 1, 1.5):
            with self.subTest(index=index), self.assertRaises(TypeError):
                shape.extent(index)
        for order in (-1, 2, "row-major"):
            with self.subTest(order=order), self.assertRaises(TypeError):
                shape.coordinates(0, order)
            with self.subTest(order=order), self.assertRaises(TypeError):
                shape.linear_index([0], order)
