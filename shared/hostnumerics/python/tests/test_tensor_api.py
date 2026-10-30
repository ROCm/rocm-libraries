# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from _hostnumerics_test_support import *  # noqa: F403


class TensorApiTests(unittest.TestCase):  # noqa: F405
    def test_public_api_uses_matmul_instead_of_fused_gemm(self):
        self.assertFalse(hasattr(hv, "reference_gemm"))
        self.assertFalse(hasattr(hv, "reference_gemm_into"))
        self.assertFalse(hasattr(native, "_GemmOptions"))
        self.assertFalse(hasattr(native, "_reference_gemm"))
        self.assertFalse(hasattr(native, "_reference_gemm_into"))

    def test_numpy_round_trip(self):
        values = np.arange(12, dtype=np.float32).reshape(3, 4)
        tensor = hv.from_numpy(values)
        self.assertEqual(tensor.type, hv.ScalarType.Float32)
        self.assertEqual(tensor.shape, [3, 4])
        np.testing.assert_array_equal(hv.to_numpy(tensor, dtype=None), values)

    def test_numpy_tensor_preserves_positive_stride_and_gaps(self):
        storage = np.asarray(
            [
                1.0,
                np.inf,
                2.0,
                np.inf,
                3.0,
                4.0,
                np.inf,
                5.0,
                np.inf,
                6.0,
            ],
            dtype=np.float32,
        )
        values = storage.reshape(2, 5)[:, ::2]
        view = hv.Tensor.from_numpy(values)
        expected = hv.from_numpy(
            np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        )

        self.assertEqual(view.type, hv.ScalarType.Float32)
        self.assertEqual(view.shape, [2, 3])
        self.assertEqual(view.strides, [5, 2])
        self.assertEqual(view.offset, 0)
        self.assertEqual(len(view.storage), storage.nbytes)
        np.testing.assert_array_equal(hv.to_numpy(view, dtype=None), values)
        self.assertTrue(
            hv.compare(view, expected, options=hv.ComparisonOptions()).passed
        )

        tolerance = hv.find_allclose_tolerance(
            view,
            expected,
            [0.0],
            [0.0],
            options=hv.allclose_comparison_options(1e-8, 1e-5, False),
        )
        self.assertIsNotNone(tolerance)
        self.assertEqual(tolerance.absolute, 0.0)
        self.assertTrue(
            hv.check_unused_tensor_storage(
                view,
                allocated_elements=storage.size,
                region=hv.SentinelRegion.Inside,
                max_reported_mismatches=10,
            ).passed
        )

    def test_numpy_tensor_preserves_negative_strides(self):
        values = np.arange(24, dtype=np.float32).reshape(4, 6)[::-1, ::-2]
        view = hv.Tensor.from_numpy(values)

        self.assertEqual(view.shape, [4, 3])
        self.assertEqual(view.strides, [-6, -2])
        self.assertEqual(view.offset, 22)
        self.assertEqual(len(view.storage), 23 * values.itemsize)
        np.testing.assert_array_equal(hv.to_numpy(view, dtype=None), values)
        self.assertTrue(
            hv.compare(
                view, hv.from_numpy(values), options=hv.ComparisonOptions()
            ).passed
        )

    def test_scalar_item_and_broadcast_view(self):
        scalar = hv.from_numpy(np.asarray(3.5, dtype=np.float32))
        self.assertEqual(scalar.item(), 3.5)
        self.assertEqual(
            hv.from_numpy(np.asarray(2**63 - 1, dtype=np.int64)).item(),
            2**63 - 1,
        )
        self.assertEqual(
            hv.from_numpy(np.asarray(2**64 - 1, dtype=np.uint64)).item(),
            2**64 - 1,
        )
        self.assertEqual(
            hv.from_numpy(np.asarray(1.25 - 2.5j, dtype=np.complex64)).item(),
            complex(1.25, -2.5),
        )
        self.assertEqual(
            hv.from_numpy(
                np.asarray(1.5, dtype=np.float32), hv.ScalarType.Float4E2M1
            ).item(),
            1.5,
        )

        broadcast = scalar.broadcast_to([2, 3])
        self.assertEqual(broadcast.shape, [2, 3])
        self.assertEqual(broadcast.strides, [0, 0])
        np.testing.assert_array_equal(
            hv.to_numpy(broadcast, dtype=None), np.full((2, 3), 3.5, dtype=np.float32)
        )

        vector = hv.from_numpy(np.asarray([1.0, 2.0], dtype=np.float32))
        expanded = vector.expand_dims(1)
        self.assertEqual(expanded.shape, [2, 1])
        self.assertEqual(expanded.strides, [1, 0])
        np.testing.assert_array_equal(
            hv.to_numpy(expanded, dtype=None),
            np.asarray([[1.0], [2.0]], dtype=np.float32),
        )

        with self.assertRaisesRegex(RuntimeError, "rank-zero"):
            hv.from_numpy(np.asarray([1.0], dtype=np.float32)).item()

    def test_tensor_conversion_preserves_layout(self):
        values = np.arange(24, dtype=np.float32).reshape(4, 6)[::-1, ::-2]
        view = hv.Tensor.from_numpy(values)

        copied = view.to(hv.ScalarType.Float32)
        self.assertEqual(copied.strides, view.strides)
        self.assertEqual(copied.offset, view.offset)
        np.testing.assert_array_equal(hv.to_numpy(copied, dtype=None), values)

        converted = view.to(hv.ScalarType.Float16)
        self.assertEqual(converted.strides, view.strides)
        self.assertEqual(converted.offset, view.offset)
        np.testing.assert_array_equal(
            hv.to_numpy(converted, dtype=None), values.astype(np.float16)
        )

        bfloat_source = hv.from_numpy(np.asarray([1.1, -2.25], dtype=np.float32))
        np.testing.assert_array_equal(
            hv.to_numpy(bfloat_source.to(hv.ScalarType.BFloat16), dtype=None),
            quantize_bfloat16(np.asarray([1.1, -2.25], dtype=np.float32)),
        )

        packed = hv.from_numpy(
            np.asarray([-6.0, -0.5, 1.5, 6.0], dtype=np.float32),
            hv.ScalarType.Float4E2M1,
        )
        np.testing.assert_array_equal(
            hv.to_numpy(packed.to(hv.ScalarType.Float32), dtype=None),
            np.asarray([-6.0, -0.5, 1.5, 6.0], dtype=np.float32),
        )

    def test_numpy_tensor_owns_an_independent_copy(self):
        values = np.arange(6, dtype=np.float32).reshape(2, 3)
        view = hv.Tensor.from_numpy(values)
        expected = hv.from_numpy(values.copy())
        self.assertTrue(
            hv.compare(view, expected, options=hv.ComparisonOptions()).passed
        )

        values[1, 2] = -17.0
        np.testing.assert_array_equal(
            hv.to_numpy(view, dtype=None), np.arange(6, dtype=np.float32).reshape(2, 3)
        )
        self.assertTrue(
            hv.compare(view, expected, options=hv.ComparisonOptions()).passed
        )

    def test_numpy_tensor_does_not_retain_source_owner(self):
        values = np.arange(6, dtype=np.float64)
        owner = weakref.ref(values)
        view = hv.Tensor.from_numpy(values)

        del values
        gc.collect()
        self.assertIsNone(owner())
        np.testing.assert_array_equal(
            hv.to_numpy(view, dtype=None), np.arange(6, dtype=np.float64)
        )

    def test_tensor_clone_is_independent(self):
        tensor = hv.from_numpy(np.arange(6, dtype=np.float32))
        cloned = tensor.clone()
        recipe = real_generation_recipe_last_dimension_fastest(
            hv.GenerationRecipe.constant(hv.ConstantGenerationParameters(value=-17.0)),
            seed=0,
        )
        hv.generate_at(cloned, 5, recipe)
        np.testing.assert_array_equal(
            hv.to_numpy(tensor, dtype=None), np.arange(6, dtype=np.float32)
        )
        self.assertEqual(hv.to_numpy(cloned, dtype=None)[5], -17.0)

    def test_numpy_tensor_accepts_read_only_array(self):
        values = np.arange(6, dtype=np.int32).reshape(2, 3)
        values.flags.writeable = False
        view = hv.Tensor.from_numpy(values)

        self.assertEqual(view.type, hv.ScalarType.Int32)
        np.testing.assert_array_equal(hv.to_numpy(view, dtype=None), values)

    def test_numpy_tensor_rejects_storage_conversion(self):
        with self.assertRaises(TypeError):
            hv.Tensor.from_numpy([1.0, 2.0])
        with self.assertRaises(RuntimeError):
            hv.Tensor.from_numpy(np.arange(4, dtype=np.dtype(">f4")))
        with self.assertRaises(RuntimeError):
            hv.Tensor.from_numpy(
                np.arange(4, dtype=np.float64),
                hv.ScalarType.Float32,
            )
        with self.assertRaises(RuntimeError):
            hv.Tensor.from_numpy(
                np.arange(4, dtype=np.uint8),
                hv.ScalarType.Float8E4M3,
            )

        byte_storage = np.arange(16, dtype=np.uint8)
        byte_strided = np.ndarray(
            shape=(3,),
            dtype=np.int16,
            buffer=byte_storage,
            strides=(3,),
        )
        with self.assertRaises(RuntimeError):
            hv.Tensor.from_numpy(byte_strided)

    def test_affine_layout_decode(self):
        storage = np.asarray(
            [-99.0, 1.0, 2.0, -99.0, 3.0, 4.0, -99.0],
            dtype=np.float32,
        )
        tensor = hv.Tensor.from_storage(
            hv.ScalarType.Float32,
            [2, 2],
            storage.tobytes(),
            strides=[1, 3],
            offset=1,
        )
        np.testing.assert_array_equal(
            hv.to_numpy(tensor, dtype=None),
            np.asarray([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32),
        )

    def test_multidimensional_slice_view(self):
        values = np.arange(120, dtype=np.float32).reshape(2, 5, 3, 4)
        tensor = hv.from_numpy(values)
        view = tensor[:, ::2, -1, :]

        self.assertIsInstance(view, hv.Tensor)
        self.assertEqual(view.type, tensor.type)
        self.assertEqual(view.shape, [2, 3, 4])
        self.assertEqual(view.strides, [60, 24, 1])
        self.assertEqual(view.offset, 8)
        np.testing.assert_array_equal(
            hv.to_numpy(view, dtype=None), values[:, ::2, -1, :]
        )
        np.testing.assert_array_equal(hv.to_numpy(tensor[-1], dtype=None), values[-1])
        np.testing.assert_array_equal(hv.to_numpy(tensor[()], dtype=None), values)
        np.testing.assert_array_equal(
            hv.to_numpy(view * 2.0 + 1.0, dtype=None), values[:, ::2, -1, :] * 2.0 + 1.0
        )
        scalar = tensor[-1, -1, -1, -1]
        self.assertEqual(scalar.shape, [])
        self.assertEqual(scalar.item(), values[-1, -1, -1, -1].item())

    def test_slice_bounds_steps_and_empty_results(self):
        for extent in (0, 1, 2, 5):
            values = np.arange(extent, dtype=np.int32)
            tensor = hv.from_numpy(values)
            for key in (
                slice(None),
                slice(None, None, 2),
                slice(None, None, -1),
                slice(None, None, -2),
                slice(1, 4),
                slice(4, 1),
                slice(-100, 100, 3),
                slice(100, -100, -3),
                slice(-(10**100), 10**100, 2),
                slice(None, None, 10**100),
                slice(None, None, -(10**100)),
            ):
                with self.subTest(extent=extent, key=key):
                    view = tensor[key]
                    expected = values[key]
                    self.assertEqual(view.shape, list(expected.shape))
                    np.testing.assert_array_equal(
                        hv.to_numpy(view, dtype=None), expected
                    )
                    if expected.size > 1:
                        self.assertEqual(
                            view.strides,
                            [stride // values.itemsize for stride in expected.strides],
                        )
        values = np.arange(6, dtype=np.int32).reshape(2, 3)
        tensor = hv.from_numpy(values)
        self.assertEqual(tensor[0, ::-3].strides, [-3])
        for key in ((slice(0, 0), -1), (slice(None), slice(3, 3, -1))):
            np.testing.assert_array_equal(
                hv.to_numpy(tensor[key], dtype=None), values[key]
            )

    def test_slices_of_reversed_gapped_and_broadcast_views(self):
        values = np.arange(240, dtype=np.float32).reshape(2, 5, 3, 8)
        reversed_values = values[::-1, ::-1, :, ::2]
        tensor = hv.Tensor.from_numpy(reversed_values)
        view = tensor[:, ::-2, -1, ::-1]
        np.testing.assert_array_equal(
            hv.to_numpy(view, dtype=None), reversed_values[:, ::-2, -1, ::-1]
        )
        np.testing.assert_array_equal(
            hv.to_numpy(view[::-1, 1:, ::2], dtype=None),
            reversed_values[:, ::-2, -1, ::-1][::-1, 1:, ::2],
        )

        scalar = hv.from_numpy(np.asarray(3.5, dtype=np.float32))
        broadcast = scalar.broadcast_to([2, 5, 3, 4])[:, ::2, -1, :]
        self.assertEqual(broadcast.strides, [0, 0, 0])
        np.testing.assert_array_equal(
            hv.to_numpy(broadcast, dtype=None),
            np.full((2, 3, 4), 3.5, dtype=np.float32),
        )

    def test_integer_index_protocol_and_scalar_tensors(self):
        class LastIndex:
            def __index__(self):
                return -1

        for values in (
            np.asarray([2**64 - 2, 2**64 - 1], dtype=np.uint64),
            np.asarray([1.0 + 2.0j, 3.0 - 4.0j], dtype=np.complex128),
            np.asarray([False, True], dtype=np.bool_),
        ):
            tensor = hv.from_numpy(values)
            for key in (-1, np.int64(-1), LastIndex()):
                scalar = tensor[key]
                self.assertIsInstance(scalar, hv.Tensor)
                self.assertEqual(scalar.type, tensor.type)
                self.assertEqual(scalar.shape, [])
                self.assertEqual(scalar.item(), values[-1].item())
                self.assertEqual(scalar[()].item(), scalar.item())

    def test_slice_views_share_writes_and_retain_storage(self):
        values = np.arange(120, dtype=np.float32).reshape(2, 5, 3, 4)
        tensor = hv.from_numpy(values)
        view = tensor[:, ::2, -1, :]
        recipe = real_generation_recipe_last_dimension_fastest(
            hv.GenerationRecipe.constant(hv.ConstantGenerationParameters(value=-17.0)),
            seed=0,
        )
        hv.generate_at(view, 5, recipe)
        values[0, 2, 2, 1] = -17.0
        np.testing.assert_array_equal(hv.to_numpy(tensor, dtype=None), values)

        hv.generate_at(tensor, 119, recipe)
        values[1, 4, 2, 3] = -17.0
        np.testing.assert_array_equal(
            hv.to_numpy(view, dtype=None), values[:, ::2, -1, :]
        )
        del tensor
        gc.collect()
        np.testing.assert_array_equal(
            hv.to_numpy(view, dtype=None), values[:, ::2, -1, :]
        )
        hv.generate_at(view, 0, recipe)
        self.assertEqual(view[0, 0, 0].item(), -17.0)

    def test_iteration_selects_shared_rows_and_stops(self):
        values = np.arange(24, dtype=np.float32).reshape(4, 6)[::-2, ::-2]
        tensor = hv.Tensor.from_numpy(values)
        self.assertEqual(len(tensor), len(values))
        iterator = iter(tensor)
        first = next(iterator)
        np.testing.assert_array_equal(hv.to_numpy(first, dtype=None), values[0])
        recipe = real_generation_recipe_last_dimension_fastest(
            hv.GenerationRecipe.constant(hv.ConstantGenerationParameters(value=-17.0)),
            seed=0,
        )
        hv.generate_at(first, 0, recipe)
        self.assertEqual(tensor[0, 0].item(), -17.0)
        del tensor
        gc.collect()
        np.testing.assert_array_equal(
            hv.to_numpy(next(iterator), dtype=None), values[1]
        )
        for _ in range(2):
            with self.assertRaises(StopIteration):
                next(iterator)
        self.assertEqual(list(hv.from_numpy(np.empty((0, 3), dtype=np.float32))), [])
        self.assertEqual(len(hv.from_numpy(np.empty((0, 3), dtype=np.float32))), 0)
        self.assertEqual(
            [
                value.item()
                for value in hv.from_numpy(np.asarray([2, 3], dtype=np.int32))
            ],
            [2, 3],
        )
        with self.assertRaises(TypeError):
            iter(hv.from_numpy(np.asarray(0, dtype=np.int32)))
        with self.assertRaises(TypeError):
            len(hv.from_numpy(np.asarray(0, dtype=np.int32)))

    def test_truth_value_uses_the_sole_element(self):
        for value, dtype in (
            (0, np.int32),
            (2**64 - 1, np.uint64),
            (-0.0, np.float64),
            (np.nan, np.float64),
            (0j, np.complex128),
            (2j, np.complex128),
            (False, np.bool_),
            (True, np.bool_),
        ):
            for shape in ((), (1,), (1, 1)):
                with self.subTest(value=value, shape=shape):
                    array = np.asarray(value, dtype=dtype).reshape(shape)
                    self.assertEqual(bool(hv.from_numpy(array)), bool(array))
        packed = hv.from_numpy(
            np.asarray([0.0, 1.5], dtype=np.float32), hv.ScalarType.Float4E2M1
        )
        self.assertFalse(bool(packed[0]))
        self.assertTrue(bool(packed[1:]))
        for shape in ((0,), (1, 0), (2,), (2, 3)):
            with self.subTest(shape=shape), self.assertRaisesRegex(
                Exception, "ambiguous"
            ):
                bool(hv.from_numpy(np.zeros(shape, dtype=np.int32)))

    def test_packed_slice_views_preserve_neighboring_bits(self):
        for scalar_type, samples in (
            (hv.ScalarType.Int4, [-8, -3, 0, 2, 7]),
            (hv.ScalarType.Float4E2M1, [-6.0, -0.5, 0.0, 1.5, 6.0]),
            (hv.ScalarType.Float6E2M3, [-2.0, -0.5, 0.0, 1.5, 2.0]),
        ):
            with self.subTest(scalar_type=scalar_type):
                values = np.resize(np.asarray(samples, dtype=np.float32), 100)
                tensor = hv.from_numpy(values.reshape(2, 5, 2, 5), scalar_type)
                expected = hv.to_numpy(tensor, dtype=None)
                original_storage = tensor.storage
                view = tensor[:, ::2, -1, :]
                self.assertEqual(view.offset, 5)
                np.testing.assert_array_equal(
                    hv.to_numpy(view, dtype=None), expected[:, ::2, -1, :]
                )
                recipe = real_generation_recipe_last_dimension_fastest(
                    hv.GenerationRecipe.constant(
                        hv.ConstantGenerationParameters(value=2.0)
                    ),
                    seed=0,
                )
                hv.generate_at(view, 0, recipe)
                expected[0, 0, 1, 0] = 2
                np.testing.assert_array_equal(hv.to_numpy(tensor, dtype=None), expected)
                bits = hv.scalar_type_info(scalar_type).storage_bits
                changed_bits = range(5 * bits, 6 * bits)
                updated_storage = tensor.storage
                for bit in range(len(original_storage) * 8):
                    if bit not in changed_bits:
                        self.assertEqual(
                            (original_storage[bit // 8] >> (bit % 8)) & 1,
                            (updated_storage[bit // 8] >> (bit % 8)) & 1,
                        )

    def test_invalid_tensor_indices(self):
        tensor = hv.from_numpy(np.arange(6, dtype=np.float32).reshape(2, 3))
        for index in (-3, 2, -(10**100), 10**100):
            with self.subTest(index=index), self.assertRaisesRegex(
                Exception, "Tensor index .*axis 0"
            ):
                tensor[index]
        for key in (
            1.5,
            True,
            np.bool_(True),
            np.asarray(0),
            np.asarray([0, 1]),
            [0, 1],
            "0",
        ):
            with self.subTest(key=key), self.assertRaisesRegex(
                Exception, "integer positions and slices"
            ):
                tensor[key]
        with self.assertRaisesRegex(Exception, "3 indices.*rank 2"):
            tensor[:, :, :]
        with self.assertRaisesRegex(Exception, "only one ellipsis"):
            tensor[..., ...]
        with self.assertRaisesRegex(Exception, "slice step cannot be zero"):
            tensor[::0]
        with self.assertRaisesRegex(Exception, "Tensor index .*axis 1"):
            tensor[:0, 3]
        with self.assertRaisesRegex(Exception, "Tensor index .*axis 0"):
            hv.from_numpy(np.empty((0,), dtype=np.float32))[-1]
        with self.assertRaisesRegex(Exception, "1 indices.*rank 0"):
            hv.from_numpy(np.asarray(1.0, dtype=np.float32))[:]
        huge_extent = hv.from_numpy(np.asarray(1.0, dtype=np.float32)).broadcast_to(
            [int(np.iinfo(np.uintp).max)]
        )
        with self.assertRaisesRegex(Exception, "extent exceeds Python"):
            huge_extent[:]

        class InvalidIndex:
            def __index__(self):
                raise RuntimeError("index conversion failed")

        with self.assertRaisesRegex(Exception, "index conversion failed"):
            tensor[InvalidIndex()]
        with self.assertRaisesRegex(Exception, "slice indices must be integers"):
            tensor[1.5:]
        with self.assertRaisesRegex(Exception, "Tensor index exceeds Python"):
            tensor[10**5000]

    def test_slicing_degenerate_axes_avoids_unused_stride_overflow(self):
        tensor = hv.Tensor.from_storage(
            hv.ScalarType.Float32,
            [1],
            np.asarray([3.5], dtype=np.float32).tobytes(),
            strides=[int(np.iinfo(np.intp).min)],
            offset=0,
        )
        self.assertEqual(tensor[::-1][0].item(), 3.5)
        self.assertEqual(tensor[:0:-1].shape, [0])
        empty = hv.Tensor.from_storage(
            hv.ScalarType.Float32,
            [0, 3],
            b"",
            strides=[1, int(np.iinfo(np.intp).max)],
            offset=0,
        )
        self.assertEqual(empty[:, ::2].shape, [0, 2])

    def test_ellipsis_and_newaxis_select_shared_views(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        tensor = hv.Tensor.from_numpy(values)
        for key in (
            Ellipsis,
            None,
            (Ellipsis, -1),
            (None, Ellipsis, -1, None),
            (slice(None, None, -1), None, Ellipsis, slice(None, None, 2)),
            (0, 1, 2, Ellipsis),
            (0, None, 1, None, 2, None),
            (slice(0, 0), Ellipsis, None),
        ):
            with self.subTest(key=key):
                view = tensor[key]
                self.assertEqual(view.shape, list(np.asarray(values[key]).shape))
                np.testing.assert_array_equal(
                    hv.to_numpy(view, dtype=None), values[key]
                )
        scalar = tensor[0, 0, 0]
        self.assertEqual(scalar[...].item(), 0)
        self.assertEqual(scalar[None, ..., None].shape, [1, 1])
        recipe = real_generation_recipe_last_dimension_fastest(
            hv.GenerationRecipe.constant(hv.ConstantGenerationParameters(value=-17.0)),
            seed=0,
        )
        hv.generate_at(tensor[None, ..., None], 23, recipe)
        self.assertEqual(tensor[-1, -1, -1].item(), -17.0)

    def test_expand_dims_normalizes_negative_axes(self):
        for shape in ((), (3,), (2, 3), (0, 3)):
            values = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
            tensor = hv.Tensor.from_numpy(values)
            destination_rank = len(shape) + 1
            for axis in range(-destination_rank, destination_rank):
                with self.subTest(shape=shape, axis=axis):
                    np.testing.assert_array_equal(
                        hv.to_numpy(tensor.expand_dims(axis), dtype=None),
                        np.expand_dims(values, axis),
                    )
            for axis in (-destination_rank - 1, destination_rank):
                with self.assertRaisesRegex(Exception, "expansion axis"):
                    tensor.expand_dims(axis)
