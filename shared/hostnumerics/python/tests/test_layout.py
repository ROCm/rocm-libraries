# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import ctypes
import gc
import itertools
import unittest

import hostnumerics as hn
import numpy as np


PTRDIFF_MAX = (1 << (8 * ctypes.sizeof(ctypes.c_ssize_t) - 1)) - 1
PTRDIFF_MIN = -PTRDIFF_MAX - 1
SIZE_MAX = (1 << (8 * ctypes.sizeof(ctypes.c_size_t))) - 1


class LayoutTests(unittest.TestCase):
    def test_scalar_and_owned_metadata(self):
        scalar = hn.Layout()
        self.assertEqual(scalar, hn.Layout(hn.Shape(), [], 0))
        self.assertEqual(scalar.shape, hn.Shape())
        self.assertEqual(scalar.strides, [])
        self.assertEqual(scalar.element_offset([]), 0)
        self.assertEqual(hn.Layout(hn.Shape(), [], -7).element_offset([]), -7)
        source = [5, -1]
        shape = hn.Shape([2, 3])
        layout = hn.Layout(shape, source, 2)
        source[0] = 9
        returned = layout.strides
        returned[1] = 9
        del shape, source
        gc.collect()
        self.assertEqual(layout.shape.dimensions, [2, 3])
        self.assertEqual(layout.strides, [5, -1])
        self.assertEqual(
            (layout.stride(0), layout.stride(1), layout.offset), (5, -1, 2)
        )
        returned_shape = layout.shape
        del layout
        gc.collect()
        self.assertEqual(returned_shape.dimensions, [2, 3])
        layout = hn.Layout(hn.Shape([2, 3]), (5, -1), 2)
        self.assertEqual(layout, hn.Layout(hn.Shape([2, 3]), [5, -1], 2))
        self.assertNotEqual(layout, hn.Layout(hn.Shape([2, 3]), [5, -1], 3))
        self.assertNotEqual(layout, hn.Layout(hn.Shape([2, 3]), [5, 1], 2))
        self.assertNotEqual(layout, hn.Layout(hn.Shape([3, 2]), [5, -1], 2))
        self.assertNotEqual(layout, None)
        for attribute, value in (("shape", hn.Shape()), ("strides", []), ("offset", 9)):
            with self.assertRaises(AttributeError):
                setattr(layout, attribute, value)

    def test_value_hashing(self):
        left = hn.Layout(hn.Shape([2, 3]), [5, -1], 2)
        equal = hn.Layout(hn.Shape([2, 3]), [5, -1], 2)
        different = hn.Layout(hn.Shape([2, 3]), [5, -1], 3)
        self.assertEqual(hash(left), hash(equal))
        self.assertEqual(len({left, equal, different}), 2)
        self.assertEqual({left: "value"}[equal], "value")

    def test_repr(self):
        self.assertEqual(
            repr(hn.Layout(hn.Shape([2, 3]), [5, -1], 2)),
            "Layout(Shape((2, 3)), (5, -1), 2)",
        )
        self.assertEqual(repr(hn.Layout()), "Layout(Shape(()), (), 0)")

    def test_contiguous_orders_match_numpy(self):
        factories = (
            ("C", hn.Layout.contiguous_last_dimension_fastest),
            ("F", hn.Layout.contiguous_first_dimension_fastest),
        )
        for dimensions in ([], [7], [2, 3], [2, 3, 4], [2, 1, 3, 2]):
            for order, factory in factories:
                layout = factory(hn.Shape(dimensions))
                array = np.empty(dimensions, dtype=np.int64, order=order)
                with self.subTest(dimensions=dimensions, order=order):
                    self.assertEqual(layout.shape.dimensions, dimensions)
                    self.assertEqual(
                        layout.strides, [s // array.itemsize for s in array.strides]
                    )
                    self.assertEqual(layout.offset, 0)
                    for point in itertools.product(*(range(d) for d in dimensions)):
                        expected = (
                            0
                            if not dimensions
                            else int(
                                np.ravel_multi_index(point, dimensions, order=order)
                            )
                        )
                        self.assertEqual(layout.element_offset(point), expected)

    def test_all_contiguous_permutations(self):
        for dimensions in ([], [7], [2, 3], [2, 3, 5], [2, 1, 3, 2]):
            shape = hn.Shape(dimensions)
            self.assertEqual(
                hn.Layout.contiguous(
                    shape, hn.Permutation(list(range(len(dimensions))))
                ),
                hn.Layout.contiguous_last_dimension_fastest(shape),
            )
            for order in itertools.permutations(range(len(dimensions))):
                layout = hn.Layout.contiguous(shape, hn.Permutation(order))
                storage = np.empty([dimensions[axis] for axis in order], dtype=np.int64)
                expected_strides = [0] * len(dimensions)
                for axis, stride in zip(order, storage.strides):
                    expected_strides[axis] = stride // storage.itemsize
                self.assertEqual(layout.strides, expected_strides)
                for point in itertools.product(*(range(d) for d in dimensions)):
                    expected = (
                        0
                        if not dimensions
                        else int(
                            np.ravel_multi_index(
                                tuple(point[axis] for axis in order), storage.shape
                            )
                        )
                    )
                    self.assertEqual(layout.element_offset(point), expected)
                    self.assertEqual(
                        layout.element_offset(point),
                        shape.linear_index(point, hn.Permutation(order)),
                    )

    def test_invalid_contiguous_orders(self):
        shape = hn.Shape([2, 3, 5])
        with self.assertRaises(TypeError):
            hn.Layout.contiguous(shape)
        for order in ([], [0, 1], [0, 1, 2, 3]):
            with self.assertRaisesRegex(
                RuntimeError, f"order rank {len(order)}.*shape rank 3"
            ):
                hn.Layout.contiguous(shape, hn.Permutation(order))
        with self.assertRaises(RuntimeError):
            hn.Layout.contiguous(hn.Shape(), hn.Permutation([0]))
        # Orders must be Permutations; plain sequences are rejected before any check.
        for order in ([0, 1, 2], [0, 0, 2], [0, 1, -1]):
            with self.assertRaises(TypeError):
                hn.Layout.contiguous(shape, order)

    def test_signed_zero_and_overlapping_strides(self):
        # Python integers provide an independent, exact oracle for each address.
        for dimensions in ([], [3], [2, 3], [2, 1, 2]):
            for strides in itertools.product((-7, -1, 0, 1, 5), repeat=len(dimensions)):
                for offset in (-3, 0, 11):
                    layout = hn.Layout(hn.Shape(dimensions), strides, offset)
                    for point in itertools.product(*(range(d) for d in dimensions)):
                        expected = offset + sum(i * s for i, s in zip(point, strides))
                        self.assertEqual(layout.element_offset(point), expected)

    def test_numpy_strided_views(self):
        storage = np.arange(30, dtype=np.int64)
        matrix = storage.reshape(5, 6)
        for view in (
            matrix,
            matrix.T,
            matrix[1:4, ::2],
            matrix[::-1, ::-2],
            np.broadcast_to(matrix[2:3, :1], (3, 4)),
        ):
            offset = (view.ctypes.data - storage.ctypes.data) // storage.itemsize
            layout = hn.Layout(
                hn.Shape(view.shape), [s // view.itemsize for s in view.strides], offset
            )
            for point in np.ndindex(view.shape):
                self.assertEqual(layout.element_offset(point), int(view[point]))

    def test_rank_and_coordinate_errors(self):
        with self.assertRaisesRegex(
            RuntimeError, "stride count 1 does not match shape rank 2"
        ):
            hn.Layout(hn.Shape([2, 3]), [1], 0)
        layout = hn.Layout(hn.Shape([2, 3]), [3, 1], 0)
        for point in ([], [0], [0, 0, 0]):
            with self.assertRaisesRegex(
                RuntimeError,
                f"coordinate rank {len(point)} does not match shape rank 2",
            ):
                layout.element_offset(point)
        for point, message in (
            ([2, 0], r"coordinate 2 in dimension 0 is outside \[0, 2\)"),
            ([0, 3], r"coordinate 3 in dimension 1 is outside \[0, 3\)"),
        ):
            with self.assertRaisesRegex(RuntimeError, message):
                layout.element_offset(point)
        with self.assertRaisesRegex(
            RuntimeError, r"layout dimension 2 is outside \[0, 2\)"
        ):
            layout.stride(2)
        for dimensions in ([0], [2, 0, 3]):
            empty = hn.Layout(hn.Shape(dimensions), [1] * len(dimensions), 0)
            with self.assertRaises(RuntimeError):
                empty.element_offset([0] * len(dimensions))

    def test_coordinates_are_validated_before_arithmetic(self):
        for strides, offset, point in (
            ([PTRDIFF_MAX, 1], 0, [2, 3]),
            ([1, 1], PTRDIFF_MAX, [1, 3]),
        ):
            layout = hn.Layout(hn.Shape([3, 3]), strides, offset)
            with self.assertRaisesRegex(
                RuntimeError, r"coordinate 3 in dimension 1 is outside \[0, 3\)"
            ):
                layout.element_offset(point)
        for dimensions in ([SIZE_MAX, 2, 0], [3, 0]):
            layout = hn.Layout(hn.Shape(dimensions), [PTRDIFF_MAX] * len(dimensions), 0)
            point = [d - 1 if d else 0 for d in dimensions]
            axis = dimensions.index(0)
            with self.assertRaisesRegex(
                RuntimeError, f"coordinate 0 in dimension {axis} is outside"
            ):
                layout.element_offset(point)

    def test_offset_arithmetic_boundaries(self):
        for stride, offset, index in (
            (PTRDIFF_MAX, PTRDIFF_MIN, 1),
            (PTRDIFF_MIN, PTRDIFF_MAX, 1),
            (-1, 0, PTRDIFF_MAX + 1),
            (0, PTRDIFF_MIN, SIZE_MAX - 1),
        ):
            layout = hn.Layout(hn.Shape([SIZE_MAX]), [stride], offset)
            self.assertEqual(layout.element_offset([index]), offset + index * stride)
        for stride, offset, index, message in (
            (PTRDIFF_MAX, 0, 2, "multiplication"),
            (PTRDIFF_MIN, 0, 2, "multiplication"),
            (1, PTRDIFF_MAX, 1, "addition"),
            (-1, PTRDIFF_MIN, 1, "addition"),
        ):
            with self.subTest(stride=stride, offset=offset), self.assertRaisesRegex(
                RuntimeError, f"offset {message} overflow"
            ):
                hn.Layout(hn.Shape([3]), [stride], offset).element_offset([index])
        # Checking each step keeps arithmetic in ptrdiff_t without a wider accumulator;
        # later cancellation cannot rescue an overflowing partial sum.
        with self.assertRaises(RuntimeError):
            hn.Layout(hn.Shape([2, 2]), [1, -1], PTRDIFF_MAX).element_offset([1, 1])
        self.assertEqual(
            hn.Layout(hn.Shape([2, 2]), [-1, 1], PTRDIFF_MAX).element_offset([1, 1]),
            PTRDIFF_MAX,
        )

    def test_element_bounds(self):
        for dimensions in ([], [3], [2, 3], [2, 0, 3]):
            for strides in itertools.product((-3, 0, 5), repeat=len(dimensions)):
                layout = hn.Layout(hn.Shape(dimensions), strides, 7)
                offsets = [
                    7 + sum(i * s for i, s in zip(point, strides))
                    for point in itertools.product(*(range(d) for d in dimensions))
                ]
                expected = (min(offsets), max(offsets)) if offsets else (0, -1)
                self.assertEqual(layout.element_bounds(), expected)
        for layout, message in (
            (
                hn.Layout(hn.Shape([2, 3]), [0, PTRDIFF_MAX], 0),
                "multiplication.*dimension 1",
            ),
            (hn.Layout(hn.Shape([2, 2]), [0, 1], PTRDIFF_MAX), "addition.*dimension 1"),
        ):
            with self.assertRaisesRegex(RuntimeError, message):
                layout.element_bounds()

    def test_contiguous_overflow_and_empty_shapes(self):
        factories = (
            hn.Layout.contiguous_last_dimension_fastest,
            hn.Layout.contiguous_first_dimension_fastest,
        )
        for factory in factories:
            self.assertEqual(factory(hn.Shape([PTRDIFF_MAX])).strides, [1])
            for dimensions in ([PTRDIFF_MAX + 1], [PTRDIFF_MAX, 2]):
                with self.assertRaises(RuntimeError):
                    factory(hn.Shape(dimensions))
            # The final product is checked even when every actual stride and offset fits.
            with self.assertRaisesRegex(RuntimeError, "contiguous stride overflow"):
                factory(hn.Shape([2, (PTRDIFF_MAX + 1) // 2]))
            for dimensions in ([0], [2, 0, 3]):
                layout = factory(hn.Shape(dimensions))
                self.assertEqual(layout.shape.dimensions, dimensions)
                with self.assertRaises(RuntimeError):
                    layout.element_offset([0] * len(dimensions))
        self.assertEqual(
            hn.Layout.contiguous_last_dimension_fastest(hn.Shape([2, 0, 3])).strides,
            [0, 3, 1],
        )
        self.assertEqual(
            hn.Layout.contiguous_first_dimension_fastest(hn.Shape([2, 0, 3])).strides,
            [1, 2, 0],
        )
        # Overflow depends on traversal order: a zero extent visited first makes
        # every slower stride zero, so a later huge extent cannot overflow.
        self.assertEqual(
            hn.Layout.contiguous_first_dimension_fastest(
                hn.Shape([0, PTRDIFF_MAX, 2])
            ).strides,
            [1, 0, 0],
        )
        self.assertEqual(
            hn.Layout.contiguous_last_dimension_fastest(
                hn.Shape([2, PTRDIFF_MAX, 0])
            ).strides,
            [0, 0, 1],
        )
        with self.assertRaises(RuntimeError):
            hn.Layout.contiguous_last_dimension_fastest(hn.Shape([0, PTRDIFF_MAX, 2]))
        with self.assertRaises(RuntimeError):
            hn.Layout.contiguous_first_dimension_fastest(hn.Shape([2, PTRDIFF_MAX, 0]))

    def test_invalid_python_inputs(self):
        for strides in ([PTRDIFF_MIN - 1], [PTRDIFF_MAX + 1], [1.5]):
            with self.assertRaises(TypeError):
                hn.Layout(hn.Shape([2]), strides, 0)
        for offset in (PTRDIFF_MIN - 1, PTRDIFF_MAX + 1, 1.5):
            with self.assertRaises(TypeError):
                hn.Layout(hn.Shape([2]), [1], offset)
        layout = hn.Layout(hn.Shape([2]), [1], 0)
        for axis in (-1, -(1 << 63), SIZE_MAX + 1, 1.5, slice(None)):
            with self.assertRaises(TypeError):
                layout.stride(axis)
        for point in ([-1], [SIZE_MAX + 1], [1.5], 3):
            with self.assertRaises(TypeError):
                hn.Layout(hn.Shape([2]), [1], 0).element_offset(point)
