# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from _host_numerics_test_support import *  # noqa: F403


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
        np.testing.assert_array_equal(hv.to_numpy(tensor), values)

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
        np.testing.assert_array_equal(hv.to_numpy(view), values)
        self.assertTrue(hv.compare(view, expected).passed)

        tolerance = hv.find_allclose_tolerance(view, expected, [0.0], [0.0])
        self.assertIsNotNone(tolerance)
        self.assertEqual(tolerance.absolute, 0.0)
        self.assertTrue(
            hv.check_unused_tensor_storage(view, allocated_elements=storage.size).passed
        )

    def test_numpy_tensor_preserves_negative_strides(self):
        values = np.arange(24, dtype=np.float32).reshape(4, 6)[::-1, ::-2]
        view = hv.Tensor.from_numpy(values)

        self.assertEqual(view.shape, [4, 3])
        self.assertEqual(view.strides, [-6, -2])
        self.assertEqual(view.offset, 22)
        self.assertEqual(len(view.storage), 23 * values.itemsize)
        np.testing.assert_array_equal(hv.to_numpy(view), values)
        self.assertTrue(hv.compare(view, hv.from_numpy(values)).passed)

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
            hv.to_numpy(broadcast), np.full((2, 3), 3.5, dtype=np.float32)
        )

        vector = hv.from_numpy(np.asarray([1.0, 2.0], dtype=np.float32))
        expanded = vector.expand_dims(1)
        self.assertEqual(expanded.shape, [2, 1])
        self.assertEqual(expanded.strides, [1, 0])
        np.testing.assert_array_equal(
            hv.to_numpy(expanded), np.asarray([[1.0], [2.0]], dtype=np.float32)
        )

        with self.assertRaisesRegex(ValueError, "rank-zero"):
            hv.from_numpy(np.asarray([1.0], dtype=np.float32)).item()

    def test_tensor_conversion_preserves_layout(self):
        values = np.arange(24, dtype=np.float32).reshape(4, 6)[::-1, ::-2]
        view = hv.Tensor.from_numpy(values)

        copied = view.to(hv.ScalarType.Float32)
        self.assertEqual(copied.strides, view.strides)
        self.assertEqual(copied.offset, view.offset)
        np.testing.assert_array_equal(hv.to_numpy(copied), values)

        converted = view.to(hv.ScalarType.Float16)
        self.assertEqual(converted.strides, view.strides)
        self.assertEqual(converted.offset, view.offset)
        np.testing.assert_array_equal(hv.to_numpy(converted), values.astype(np.float16))

        bfloat_source = hv.from_numpy(np.asarray([1.1, -2.25], dtype=np.float32))
        np.testing.assert_array_equal(
            hv.to_numpy(bfloat_source.to(hv.ScalarType.BFloat16)),
            quantize_bfloat16(np.asarray([1.1, -2.25], dtype=np.float32)),
        )

        packed = hv.from_numpy(
            np.asarray([-6.0, -0.5, 1.5, 6.0], dtype=np.float32),
            hv.ScalarType.Float4E2M1,
        )
        np.testing.assert_array_equal(
            hv.to_numpy(packed.to(hv.ScalarType.Float32)),
            np.asarray([-6.0, -0.5, 1.5, 6.0], dtype=np.float32),
        )

    def test_numpy_tensor_owns_an_independent_copy(self):
        values = np.arange(6, dtype=np.float32).reshape(2, 3)
        view = hv.Tensor.from_numpy(values)
        expected = hv.from_numpy(values.copy())
        self.assertTrue(hv.compare(view, expected).passed)

        values[1, 2] = -17.0
        np.testing.assert_array_equal(
            hv.to_numpy(view), np.arange(6, dtype=np.float32).reshape(2, 3)
        )
        self.assertTrue(hv.compare(view, expected).passed)

    def test_numpy_tensor_does_not_retain_source_owner(self):
        values = np.arange(6, dtype=np.float64)
        owner = weakref.ref(values)
        view = hv.Tensor.from_numpy(values)

        del values
        gc.collect()
        self.assertIsNone(owner())
        np.testing.assert_array_equal(hv.to_numpy(view), np.arange(6, dtype=np.float64))

    def test_tensor_clone_is_independent(self):
        tensor = hv.from_numpy(np.arange(6, dtype=np.float32))
        cloned = tensor.clone()
        recipe = real_generation_recipe(
            hv.GenerationRecipe.constant(hv.ConstantGenerationParameters(value=-17.0))
        )
        hv.generate_at(cloned, 5, recipe)
        np.testing.assert_array_equal(
            hv.to_numpy(tensor), np.arange(6, dtype=np.float32)
        )
        self.assertEqual(hv.to_numpy(cloned)[5], -17.0)

    def test_numpy_tensor_accepts_read_only_array(self):
        values = np.arange(6, dtype=np.int32).reshape(2, 3)
        values.flags.writeable = False
        view = hv.Tensor.from_numpy(values)

        self.assertEqual(view.type, hv.ScalarType.Int32)
        np.testing.assert_array_equal(hv.to_numpy(view), values)

    def test_numpy_tensor_rejects_storage_conversion(self):
        with self.assertRaises(TypeError):
            hv.Tensor.from_numpy([1.0, 2.0])
        with self.assertRaises(TypeError):
            hv.Tensor.from_numpy(np.arange(4, dtype=np.dtype(">f4")))
        with self.assertRaises(ValueError):
            hv.Tensor.from_numpy(
                np.arange(4, dtype=np.float64),
                hv.ScalarType.Float32,
            )
        with self.assertRaises(ValueError):
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
        with self.assertRaises(ValueError):
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
            hv.to_numpy(tensor),
            np.asarray([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32),
        )
