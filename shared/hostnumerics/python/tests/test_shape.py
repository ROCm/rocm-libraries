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

    def test_index_conversions_require_order_or_named_choice(self):
        shape = hn.Shape([2, 3])
        with self.assertRaises(TypeError):
            shape.coordinates(1)
        with self.assertRaises(TypeError):
            shape.linear_index([0, 1])
        self.assertEqual(shape.coordinates_last_dimension_fastest(1), [0, 1])
        self.assertEqual(shape.linear_index_last_dimension_fastest([0, 1]), 1)

    def test_products_and_dimension_selection(self):
        shape = hn.Shape([2, 3, 5])
        self.assertEqual(shape.element_count, 30)
        for axes, extents in (
            ([0, 2], [2, 5]),
            ([2, 0], [5, 2]),
            ([1, 1], [3, 3]),
            ([], []),
        ):
            selected = shape.from_dimensions(axes)
            self.assertEqual(selected.dimensions, extents)
            self.assertEqual(selected.element_count, math.prod(extents))
        for dimensions in itertools.product((0, 1, 2, SIZE_MAX), repeat=3):
            count = math.prod(dimensions)
            if count > SIZE_MAX:
                with self.assertRaisesRegex(
                    RuntimeError, "element count exceeds size_t"
                ):
                    hn.Shape(dimensions)
                continue
            shape = hn.Shape(dimensions)
            self.assertEqual(shape.element_count, count)
            for length in range(4):
                for axes in itertools.product(range(3), repeat=length):
                    extents = [dimensions[a] for a in axes]
                    expected = math.prod(extents)
                    with self.subTest(dimensions=dimensions, axes=axes):
                        if expected > SIZE_MAX:
                            with self.assertRaisesRegex(
                                RuntimeError, "element count exceeds size_t"
                            ):
                                shape.from_dimensions(axes)
                        else:
                            selected = shape.from_dimensions(axes)
                            self.assertEqual(selected.dimensions, extents)
                            self.assertEqual(selected.element_count, expected)

    def test_descriptive_bounds_errors(self):
        shape = hn.Shape([2, 3, 4])
        with self.assertRaisesRegex(RuntimeError, r"dimension 3 is outside \[0, 3\)"):
            shape.extent(3)
        with self.assertRaisesRegex(RuntimeError, r"dimension 3 is outside \[0, 3\)"):
            shape[3]
        with self.assertRaisesRegex(RuntimeError, r"dimension 3 is outside \[0, 3\)"):
            shape.from_dimensions([3])

    def test_coordinate_orders(self):
        shape = hn.Shape([2, 3, 4])
        self.assertEqual(shape.linear_index([1, 0, 2], hn.Permutation(ORDERS[0])), 13)
        self.assertEqual(shape.linear_index([1, 0, 2], hn.Permutation(ORDERS[1])), 14)
        self.assertEqual(shape.coordinates(17, hn.Permutation(ORDERS[0])), [1, 2, 2])
        self.assertEqual(shape.coordinates(17, hn.Permutation(ORDERS[1])), [1, 1, 1])
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
                        self.assertEqual(
                            shape.coordinates(index, hn.Permutation(order)), coordinates
                        )
                        self.assertEqual(
                            shape.linear_index(coordinates, hn.Permutation(order)),
                            index,
                        )
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
                    shape.coordinates_last_dimension_fastest(index),
                    shape.coordinates(index, hn.Permutation(list(range(shape.rank)))),
                )
                self.assertEqual(
                    shape.linear_index_last_dimension_fastest(
                        shape.coordinates_last_dimension_fastest(index)
                    ),
                    index,
                )
        example = hn.Shape([2, 3, 5])
        self.assertEqual(example.linear_index([1, 2, 3], hn.Permutation([1, 2, 0])), 27)
        self.assertEqual(example.coordinates(27, hn.Permutation([1, 2, 0])), [1, 2, 3])

    def test_large_coordinate_arithmetic(self):
        # Sample boundary coordinates; Python integers provide an independent
        # oracle without enumerating SIZE_MAX-sized dimensions or using NumPy's
        # more restrictive native integer limits.
        for dimensions in itertools.product((0, 1, 2, SIZE_MAX), repeat=3):
            count = math.prod(dimensions)
            if count > SIZE_MAX:
                with self.assertRaises(RuntimeError):
                    hn.Shape(dimensions)
                continue
            shape = hn.Shape(dimensions)
            for order in itertools.permutations(range(shape.rank)):
                if count == 0:
                    indices = [max(extent - 1, 0) for extent in dimensions]
                    with self.subTest(dimensions=dimensions, order=order):
                        with self.assertRaises(RuntimeError):
                            shape.linear_index(indices, hn.Permutation(order))
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
                        self.assertEqual(
                            shape.linear_index(indices, hn.Permutation(order)), expected
                        )
                        self.assertEqual(
                            shape.coordinates(expected, hn.Permutation(order)),
                            list(indices),
                        )

    def test_coordinate_bounds_before_arithmetic(self):
        shape = hn.Shape([SIZE_MAX, 2, 0])
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
                        shape.linear_index(indices, hn.Permutation(order))

    def test_index_order_errors(self):
        shape = hn.Shape([2, 3, 4])
        for order in ([], [0], [0, 1, 2, 3]):
            message = f"index order rank {len(order)} does not match shape rank 3"
            with self.subTest(order=order):
                with self.assertRaisesRegex(RuntimeError, message):
                    shape.coordinates(0, hn.Permutation(order))
                with self.assertRaisesRegex(RuntimeError, message):
                    shape.linear_index([0, 0, 0], hn.Permutation(order))
        # A plain list is never accepted as an order, even a valid permutation.
        for order in ([0, 1, 2], (2, 1, 0)):
            with self.subTest(order=order), self.assertRaises(TypeError):
                shape.coordinates(0, order)
            with self.subTest(order=order), self.assertRaises(TypeError):
                shape.linear_index([0, 0, 0], order)

    def test_coordinate_errors(self):
        shape = hn.Shape([2, 3, 4])
        scalar = hn.Shape()
        self.assertEqual(scalar.linear_index_last_dimension_fastest([]), 0)
        self.assertEqual(scalar.coordinates_last_dimension_fastest(0), [])
        for operation in (
            lambda: scalar.coordinates(0, hn.Permutation([0])),
            lambda: scalar.linear_index([], hn.Permutation([0])),
            lambda: shape.linear_index([], hn.Permutation([0])),
            lambda: shape.linear_index([], hn.Permutation([0, 0, 2])),
        ):
            with self.assertRaises(RuntimeError):
                operation()
        for order in ORDERS:
            for coordinates in ([], [0, 0], [0, 0, 0, 0]):
                with self.assertRaisesRegex(
                    RuntimeError, "coordinate rank .* does not match shape rank 3"
                ):
                    shape.linear_index(coordinates, hn.Permutation(order))
            for coordinates, axis, extent in (
                ([2, 0, 0], 0, 2),
                ([1, 3, 0], 1, 3),
                ([1, 0, 4], 2, 4),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    rf"coordinate {extent} in dimension {axis} is outside \[0, {extent}\)",
                ):
                    shape.linear_index(coordinates, hn.Permutation(order))
            with self.assertRaisesRegex(
                RuntimeError, r"logical index 24 is outside \[0, 24\)"
            ):
                shape.coordinates(24, hn.Permutation(order))
            with self.assertRaises(RuntimeError):
                hn.Shape().coordinates_last_dimension_fastest(1)
            for dimensions in (
                [0],
                [0, SIZE_MAX, 1],
                [SIZE_MAX, 0, 1],
                [SIZE_MAX, 1, 0],
            ):
                empty = hn.Shape(dimensions)
                self.assertEqual(empty.element_count, 0)
                with self.assertRaises(RuntimeError):
                    empty.coordinates_last_dimension_fastest(0)
                with self.assertRaises(RuntimeError):
                    empty.linear_index_last_dimension_fastest([0] * empty.rank)

    def test_overflow(self):
        boundary = hn.Shape([SIZE_MAX, 1])
        self.assertEqual(boundary.element_count, SIZE_MAX)
        for order in ([0, 1], [1, 0]):
            self.assertEqual(
                boundary.linear_index([SIZE_MAX - 1, 0], hn.Permutation(order)),
                SIZE_MAX - 1,
            )
            self.assertEqual(
                boundary.coordinates(SIZE_MAX - 1, hn.Permutation(order)),
                [SIZE_MAX - 1, 0],
            )
        for dimensions in (
            [SIZE_MAX, 2],
            [SIZE_MAX, 2, 1, 1],
        ):
            for extents in (dimensions, tuple(dimensions)):
                with self.subTest(extents=extents), self.assertRaisesRegex(
                    RuntimeError, "element count exceeds size_t"
                ):
                    hn.Shape(extents)
        for dimensions in ([0, SIZE_MAX, 2], [SIZE_MAX, 0, 2], [SIZE_MAX, 2, 0]):
            empty = hn.Shape(dimensions)
            self.assertEqual(empty.element_count, 0)
            nonzero = [axis for axis, extent in enumerate(dimensions) if extent]
            with self.assertRaisesRegex(RuntimeError, "element count exceeds size_t"):
                empty.from_dimensions(nonzero)
            with self.assertRaises(RuntimeError):
                empty.coordinates_last_dimension_fastest(0)
        empty = hn.Shape([SIZE_MAX, 2, 0])
        with self.assertRaisesRegex(RuntimeError, "element count exceeds size_t"):
            empty.from_dimensions([0, 1])
        self.assertEqual(empty.from_dimensions([2, 0, 0]).element_count, 0)
        with self.assertRaisesRegex(RuntimeError, "element count exceeds size_t"):
            hn.Shape([SIZE_MAX]).from_dimensions([0, 0])

    def test_invalid_python_inputs(self):
        for dimensions in ([-1], [SIZE_MAX + 1], [1.5], 3):
            with self.subTest(dimensions=dimensions), self.assertRaises(TypeError):
                hn.Shape(dimensions)
        shape = hn.Shape([2])
        for index in (-1, -(10**100), SIZE_MAX + 1, 1.5):
            with self.subTest(index=index), self.assertRaises(TypeError):
                shape.extent(index)
            with self.subTest(index=index), self.assertRaises(TypeError):
                shape[index]
        for order in (-1, 2, "row-major", [0]):
            with self.subTest(order=order), self.assertRaises(TypeError):
                shape.coordinates(0, order)
            with self.subTest(order=order), self.assertRaises(TypeError):
                shape.linear_index([0], order)
        shape = hn.Shape([2, 3, 5])
        for axis in (-1, -(2**63)):
            for operation in (
                lambda: shape.extent(axis),
                lambda: shape.from_dimensions([axis]),
                lambda: shape.coordinates(0, [0, 1, axis]),
                lambda: shape.linear_index([0, 0, 0], [0, 1, axis]),
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

    def test_checked_integer_indexing(self):
        class FirstIndex:
            def __index__(self):
                return 0

        for extents in ((), (0,), (2, 3, 5), tuple(range(20)), (SIZE_MAX, 1)):
            shape = hn.Shape(extents)
            for key in (*range(len(extents)), True, np.int64(0), FirstIndex()):
                with self.subTest(extents=extents, key=key):
                    try:
                        expected = extents[key]
                    except IndexError:
                        with self.assertRaises(RuntimeError):
                            shape[key]
                    else:
                        self.assertEqual(shape[key], expected)
                        self.assertEqual(shape[key], shape.extent(key))
            for key in (len(extents), SIZE_MAX):
                with self.subTest(extents=extents, key=key), self.assertRaisesRegex(
                    RuntimeError, rf"dimension {key} is outside \[0, {len(extents)}\)"
                ):
                    shape[key]
        shape = hn.Shape([2, 3, 5])
        for key in (
            -1,
            np.int64(-1),
            None,
            Ellipsis,
            (slice(None), slice(None)),
            [0],
            1.5,
        ):
            with self.assertRaises(TypeError):
                shape[key]

    def test_slices_are_rejected(self):
        for extents in ((), (0,), (2, 3, 5)):
            shape = hn.Shape(extents)
            for key in (
                slice(None),
                slice(1, None),
                slice(None, None, -1),
                slice(0, 3, 2),
                slice(None, None, 0),
            ):
                with self.subTest(extents=extents, key=key), self.assertRaises(
                    TypeError
                ):
                    shape[key]

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
