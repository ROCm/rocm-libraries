# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import ctypes
import itertools
import math
import unittest

import hostnumerics as hn
import numpy as np


SIZE_MAX = (1 << (8 * ctypes.sizeof(ctypes.c_size_t))) - 1
ORDERS = ([2, 1, 0], [0, 1, 2])


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
        self.assertEqual(shape.project([0, 2]).element_count, 8)
        self.assertEqual(hn.Shape([7]).project([]).element_count, 1)
        for dimensions in itertools.product((0, 1, 2, SIZE_MAX), repeat=3):
            if math.prod(d for d in dimensions if d != 0) > SIZE_MAX:
                with self.assertRaisesRegex(
                    RuntimeError, "product of nonzero extents exceeds size_t"
                ):
                    hn.Shape(dimensions)
                continue
            shape = hn.Shape(dimensions)
            for first in range(4):
                for end in range(first, 4):
                    expected = math.prod(dimensions[first:end])
                    with self.subTest(dimensions=dimensions, first=first, end=end):
                        self.assertEqual(
                            shape.element_count_in_range(first, end), expected
                        )
            for excluded in range(3):
                expected = math.prod(
                    d for i, d in enumerate(dimensions) if i != excluded
                )
                with self.subTest(dimensions=dimensions, excluded=excluded):
                    self.assertEqual(
                        shape.project(
                            [i for i in range(3) if i != excluded]
                        ).element_count,
                        expected,
                    )
            for axes in itertools.permutations(range(3)):
                projected = shape.project(axes)
                self.assertEqual(projected.dimensions, [dimensions[a] for a in axes])
                self.assertEqual(projected.element_count, math.prod(dimensions))
            for selection in (
                slice(None),
                slice(None, None, -1),
                slice(0, 2),
                slice(1, None),
            ):
                sliced = shape[selection]
                self.assertEqual(sliced.dimensions, list(dimensions[selection]))
                self.assertEqual(sliced.element_count, math.prod(dimensions[selection]))

    def test_descriptive_bounds_errors(self):
        shape = hn.Shape([2, 3, 4])
        with self.assertRaisesRegex(RuntimeError, r"dimension 3 is outside \[0, 3\)"):
            shape.extent(3)
        with self.assertRaisesRegex(IndexError, "tuple index out of range"):
            shape[3]
        for first, end, reason in (
            (2, 1, "start exceeds the end"),
            (0, 4, "end exceeds the shape rank"),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                rf"range \[{first}, {end}\) is invalid for rank 3: the {reason}",
            ):
                shape.element_count_in_range(first, end)
        with self.assertRaisesRegex(RuntimeError, r"dimension 3 is outside \[0, 3\)"):
            shape.project([3])

    def test_coordinate_orders(self):
        shape = hn.Shape([2, 3, 4])
        self.assertEqual(shape.linear_index([1, 0, 2], ORDERS[0]), 13)
        self.assertEqual(shape.linear_index([1, 0, 2], ORDERS[1]), 14)
        self.assertEqual(shape.coordinates(17, ORDERS[0]), [1, 2, 2])
        self.assertEqual(shape.coordinates(17, ORDERS[1]), [1, 1, 1])
        for dimensions in ([], [7], [2, 3, 4], [2, 1, 3, 2]):
            shape = hn.Shape(dimensions)
            for order in itertools.permutations(range(shape.rank)):
                positions = itertools.product(
                    *(range(dimensions[axis]) for axis in order)
                )
                for index, position in enumerate(positions):
                    coordinates = [0] * shape.rank
                    for axis, coordinate in zip(order, position):
                        coordinates[axis] = coordinate
                    with self.subTest(dimensions=dimensions, order=order, index=index):
                        self.assertEqual(shape.coordinates(index, order), coordinates)
                        self.assertEqual(shape.linear_index(coordinates, order), index)
                        if dimensions:
                            ordered_shape = tuple(dimensions[axis] for axis in order)
                            ordered_point = tuple(coordinates[axis] for axis in order)
                            self.assertEqual(
                                index,
                                np.ravel_multi_index(
                                    ordered_point, ordered_shape, order="C"
                                ),
                            )
                            self.assertEqual(
                                ordered_point,
                                np.unravel_index(index, ordered_shape, order="C"),
                            )
            for index in range(shape.element_count):
                self.assertEqual(
                    shape.coordinates(index),
                    shape.coordinates(index, list(range(shape.rank))),
                )
        example = hn.Shape([2, 3, 5])
        self.assertEqual(example.linear_index([1, 2, 3], [1, 2, 0]), 27)
        self.assertEqual(example.coordinates(27, [1, 2, 0]), [1, 2, 3])

    def test_large_coordinate_arithmetic(self):
        # Sample boundary coordinates; Python integers provide an independent
        # oracle without enumerating SIZE_MAX-sized dimensions or using NumPy's
        # more restrictive native integer limits.
        for dimensions in itertools.product((0, 1, 2, SIZE_MAX), repeat=3):
            count = math.prod(dimensions)
            if math.prod(d for d in dimensions if d != 0) > SIZE_MAX:
                with self.assertRaises(RuntimeError):
                    hn.Shape(dimensions)
                continue
            shape = hn.Shape(dimensions)
            for order in itertools.permutations(range(shape.rank)):
                if count == 0:
                    indices = [max(extent - 1, 0) for extent in dimensions]
                    with self.subTest(dimensions=dimensions, order=order):
                        with self.assertRaises(RuntimeError):
                            shape.linear_index(indices, order)
                    continue
                samples = [
                    sorted({0, min(1, extent - 1), extent - 1}) for extent in dimensions
                ]
                for indices in itertools.product(*samples):
                    expected = sum(
                        indices[axis]
                        * math.prod(
                            dimensions[later] for later in order[position + 1 :]
                        )
                        for position, axis in enumerate(order)
                    )
                    with self.subTest(
                        dimensions=dimensions, order=order, indices=indices
                    ):
                        self.assertEqual(shape.linear_index(indices, order), expected)
                        self.assertEqual(
                            shape.coordinates(expected, order), list(indices)
                        )

    def test_coordinate_bounds_before_arithmetic(self):
        shape = hn.Shape([SIZE_MAX, 1, 0])
        for order in itertools.permutations(range(shape.rank)):
            for indices, dimension, coordinate, extent in (
                ([SIZE_MAX - 1, 0, 0], 2, 0, 0),
                ([0, 0, 5], 2, 5, 0),
                ([SIZE_MAX, 0, 0], 0, SIZE_MAX, SIZE_MAX),
            ):
                with self.subTest(order=order, indices=indices):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        rf"coordinate {coordinate} in dimension {dimension} is outside "
                        rf"\[0, {extent}\)",
                    ):
                        shape.linear_index(indices, order)

    def test_projection(self):
        shape = hn.Shape([2, 3, 4])
        self.assertEqual(shape.project([2, 0]).dimensions, [4, 2])
        self.assertEqual(shape.project([]), hn.Shape())
        with self.assertRaisesRegex(RuntimeError, "repeats dimension 1"):
            shape.project([1, 1])
        for order, message in (
            ([0], "index order rank 1 does not match shape rank 3"),
            ([0, 0, 2], r"index order must be a permutation of \[0, 3\)"),
            ([0, 1, 3], r"index order must be a permutation of \[0, 3\)"),
        ):
            with self.subTest(order=order):
                with self.assertRaisesRegex(RuntimeError, message):
                    shape.coordinates(0, order)
                with self.assertRaisesRegex(RuntimeError, message):
                    shape.linear_index([0, 0, 0], order)

    def test_coordinate_errors(self):
        shape = hn.Shape([2, 3, 4])
        scalar = hn.Shape()
        self.assertEqual(scalar.linear_index([]), 0)
        self.assertEqual(scalar.coordinates(0), [])
        for operation in (
            lambda: scalar.coordinates(0, [0]),
            lambda: scalar.linear_index([], [0]),
            lambda: shape.linear_index([], [0]),
            lambda: shape.linear_index([], [0, 0, 2]),
        ):
            with self.assertRaises(RuntimeError):
                operation()
        for order in ORDERS:
            with self.assertRaisesRegex(
                RuntimeError, "coordinate rank 0 does not match shape rank 3"
            ):
                shape.linear_index([], order)
            with self.assertRaisesRegex(
                RuntimeError, r"coordinate 3 in dimension 1 is outside \[0, 3\)"
            ):
                shape.linear_index([1, 3, 0], order)
            with self.assertRaisesRegex(
                RuntimeError, r"logical index 24 is outside \[0, 24\)"
            ):
                shape.coordinates(24, order)
            with self.assertRaises(RuntimeError):
                hn.Shape().coordinates(1)
            for dimensions in (
                [0],
                [0, SIZE_MAX, 1],
                [SIZE_MAX, 0, 1],
                [SIZE_MAX, 1, 0],
            ):
                empty = hn.Shape(dimensions)
                self.assertEqual(empty.element_count, 0)
                with self.assertRaises(RuntimeError):
                    empty.coordinates(0)
                with self.assertRaises(RuntimeError):
                    empty.linear_index([0] * empty.rank)

    def test_overflow(self):
        boundary = hn.Shape([SIZE_MAX, 1])
        self.assertEqual(boundary.element_count, SIZE_MAX)
        for order in ([0, 1], [1, 0]):
            self.assertEqual(
                boundary.linear_index([SIZE_MAX - 1, 0], order), SIZE_MAX - 1
            )
            self.assertEqual(
                boundary.coordinates(SIZE_MAX - 1, order), [SIZE_MAX - 1, 0]
            )
        for dimensions in (
            [SIZE_MAX, 2],
            [SIZE_MAX, 2, 1, 1],
            [0, SIZE_MAX, 2],
            [SIZE_MAX, 0, 2],
            [SIZE_MAX, 2, 0],
        ):
            for extents in (dimensions, tuple(dimensions)):
                with self.subTest(extents=extents), self.assertRaisesRegex(
                    RuntimeError, "product of nonzero extents exceeds size_t"
                ):
                    hn.Shape(extents)
        empty = hn.Shape([SIZE_MAX, 1, 0])
        self.assertEqual(empty.element_count, 0)
        self.assertEqual(empty.element_count_in_range(0, 2), SIZE_MAX)
        self.assertEqual(empty.project([0, 1]).element_count, SIZE_MAX)
        self.assertEqual(empty[:2].element_count, SIZE_MAX)
        self.assertEqual(empty[::-1][1:].element_count, SIZE_MAX)
        self.assertEqual(empty.project([1, 2, 0]).element_count, 0)
        self.assertEqual(empty[::-1].element_count, 0)

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
        shape = hn.Shape([2, 3, 5])
        for axis in (-1, -(2**63)):
            for operation in (
                lambda: shape.extent(axis),
                lambda: shape.project([axis]),
                lambda: shape.coordinates(0, [0, 1, axis]),
                lambda: shape.linear_index([0, 0, 0], [0, 1, axis]),
                lambda: shape.element_count_in_range(axis, 3),
            ):
                with self.subTest(axis=axis), self.assertRaises(TypeError):
                    operation()

    def test_tuple_sequence_operations(self):
        for extents in ((), (0,), (2, 3, 2, 5), (SIZE_MAX, 1)):
            with self.subTest(extents=extents):
                shape = hn.Shape(extents)
                self.assertEqual(len(shape), len(extents))
                self.assertEqual(tuple(shape), extents)
                self.assertEqual(list(reversed(shape)), list(reversed(extents)))
                self.assertEqual(bool(shape), bool(extents))
                self.assertEqual(repr(shape), f"Shape({extents!r})")
                for value in (0, 2, 3.0, SIZE_MAX, "2", None):
                    self.assertEqual(value in shape, value in extents)
                    self.assertEqual(shape.count(value), extents.count(value))
                    for bounds in ((), (1,), (-3, 3), (0, 10**100)):
                        try:
                            expected = extents.index(value, *bounds)
                        except ValueError:
                            with self.assertRaises(ValueError):
                                shape.index(value, *bounds)
                        else:
                            self.assertEqual(shape.index(value, *bounds), expected)

    def test_tuple_integer_and_slice_indexing(self):
        class LastIndex:
            def __index__(self):
                return -1

        for extents in ((), (0,), (2, 3, 5), tuple(range(20)), (SIZE_MAX, 1)):
            shape = hn.Shape(extents)
            for key in (0, -1, True, np.int64(-1), LastIndex(), 10**100, -(10**100)):
                with self.subTest(extents=extents, key=key):
                    try:
                        expected = extents[key]
                    except IndexError:
                        with self.assertRaises(IndexError):
                            shape[key]
                    else:
                        self.assertEqual(shape[key], expected)
            for start, stop, step in itertools.product(
                (None, -30, -1, 0, 2, 30, 10**100),
                (None, -30, -1, 0, 2, 30, 10**100),
                (None, -3, -1, 1, 2, 5, 10**100, -(10**100)),
            ):
                key = slice(start, stop, step)
                with self.subTest(extents=extents, key=key):
                    result = shape[key]
                    self.assertIsInstance(result, hn.Shape)
                    self.assertEqual(tuple(result), extents[key])
            self.assertEqual(shape[:][::-1][::2], hn.Shape(extents[::-1][::2]))
        self.assertEqual(hn.Shape(range(20))[0:16:5], hn.Shape([0, 5, 10, 15]))
        shape = hn.Shape([2, 3, 5])
        for key in (None, Ellipsis, (slice(None), slice(None)), [0], 1.5):
            with self.assertRaises(TypeError):
                shape[key]
        with self.assertRaises(ValueError):
            shape[::0]

        class InvalidIndex:
            def __index__(self):
                raise RuntimeError("index conversion failed")

        with self.assertRaisesRegex(RuntimeError, "index conversion failed"):
            shape[InvalidIndex()]

    def test_tuple_hash_and_immutability(self):
        for extents in ((), (0,), (2, 3, 5), (SIZE_MAX, 1)):
            shape = hn.Shape(extents)
            self.assertEqual(hash(shape), hash(extents))
            self.assertEqual({shape: "found"}[hn.Shape(extents)], "found")
            self.assertEqual(len({shape, hn.Shape(extents)}), 1)
            self.assertNotEqual(shape, extents)
        shape = hn.Shape([2, 3, 5])
        iterator = iter(shape)
        del shape
        self.assertEqual(tuple(iterator), (2, 3, 5))
        shape = hn.Shape([2, 3, 5])
        with self.assertRaises(TypeError):
            shape[0] = 9
        for method in ("append", "pop", "sort"):
            self.assertFalse(hasattr(shape, method))
