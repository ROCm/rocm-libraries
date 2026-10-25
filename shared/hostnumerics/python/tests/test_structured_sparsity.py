# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from _hostnumerics_test_support import *  # noqa: F403


class StructuredSparsityTests(unittest.TestCase):  # noqa: F405
    def test_oversized_empty_shapes_do_no_sparsity_or_metadata_work(self):
        maximum = int(np.iinfo(np.intp).max)
        for shape in ([maximum, 3, 0], [3, maximum, 0]):
            with self.subTest(shape=shape):
                # Excluding the zero axis would overflow even though no work exists.
                tensor = hv.Tensor.from_storage(
                    hv.ScalarType.Float32, shape, b"", strides=[0, 0, 0], offset=0
                )
                pattern = hv.StructuredSparsityPattern()
                pattern.index_order = [0, 1, 2]
                pattern.axis = 2
                pattern.fixed_positions = [1, 3]
                result = hv.apply_structured_sparsity(tensor, pattern, True)
                for output in (
                    result.pruned,
                    result.compressed,
                    result.retained_indices,
                    result.two_of_four_metadata,
                    hv.encode_two_of_four_metadata(result.retained_indices, 2),
                ):
                    self.assertEqual(output.shape, shape)
                    self.assertEqual(output.size, 0)
                    self.assertEqual(output.storage, b"")
                pattern.fixed_positions = [1, 4]
                with self.assertRaisesRegex(RuntimeError, "fixed position"):
                    hv.apply_structured_sparsity(tensor, pattern, True)
                with self.assertRaisesRegex(RuntimeError, "input type must be UInt8"):
                    hv.encode_two_of_four_metadata(tensor, 2)

    def test_structured_sparsity_all_fixed_two_of_four_patterns(self):
        values = np.arange(1, 17, dtype=np.float32).reshape(2, 8)
        retained_position_sets = (
            (0, 1),
            (0, 2),
            (1, 2),
            (0, 3),
            (1, 3),
            (2, 3),
        )

        for retained_positions in retained_position_sets:
            with self.subTest(retained_positions=retained_positions):
                pattern = hv.StructuredSparsityPattern()
                pattern.index_order = [0, 1]
                pattern.axis = 1
                pattern.fixed_positions = list(retained_positions)
                result = hv.apply_structured_sparsity(
                    hv.from_numpy(values), pattern, True
                )

                expected_pruned = np.zeros_like(values)
                expected_compressed = np.empty((2, 4), dtype=np.float32)
                expected_indices = np.empty((2, 4), dtype=np.uint8)
                for row in range(values.shape[0]):
                    for group in range(2):
                        source = values[row, group * 4 : (group + 1) * 4]
                        for retained_index, position in enumerate(retained_positions):
                            expected_pruned[row, group * 4 + position] = source[
                                position
                            ]
                            expected_compressed[row, group * 2 + retained_index] = (
                                source[position]
                            )
                            expected_indices[row, group * 2 + retained_index] = position

                np.testing.assert_array_equal(
                    hv.to_numpy(result.pruned, dtype=None), expected_pruned
                )
                np.testing.assert_array_equal(
                    hv.to_numpy(result.compressed, dtype=None), expected_compressed
                )
                np.testing.assert_array_equal(
                    hv.to_numpy(result.retained_indices, dtype=None), expected_indices
                )
                metadata = hv.encode_two_of_four_metadata(result.retained_indices, 1)
                nibble = retained_positions[0] | (retained_positions[1] << 2)
                np.testing.assert_array_equal(
                    hv.to_numpy(metadata, dtype=None),
                    np.full(
                        (2, 1),
                        nibble | (nibble << 4),
                        dtype=np.uint8,
                    ),
                )
                np.testing.assert_array_equal(
                    hv.to_numpy(result.two_of_four_metadata, dtype=None),
                    hv.to_numpy(metadata, dtype=None),
                )

    def test_structured_sparsity_random_is_deterministic_and_self_consistent(self):
        values = np.arange(1, 65, dtype=np.float32).reshape(4, 16)
        pattern = hv.StructuredSparsityPattern()
        pattern.index_order = [0, 1]
        pattern.axis = 1
        pattern.selection = hv.StructuredSparsitySelection.Random
        pattern.seed = 0x12345678

        first = hv.apply_structured_sparsity(hv.from_numpy(values), pattern, True)
        second = hv.apply_structured_sparsity(hv.from_numpy(values), pattern, True)
        np.testing.assert_array_equal(
            hv.to_numpy(first.pruned, dtype=None),
            hv.to_numpy(second.pruned, dtype=None),
        )
        np.testing.assert_array_equal(
            hv.to_numpy(first.compressed, dtype=None),
            hv.to_numpy(second.compressed, dtype=None),
        )
        np.testing.assert_array_equal(
            hv.to_numpy(first.retained_indices, dtype=None),
            hv.to_numpy(second.retained_indices, dtype=None),
        )

        pruned = hv.to_numpy(first.pruned, dtype=None)
        compressed = hv.to_numpy(first.compressed, dtype=None)
        retained_indices = hv.to_numpy(first.retained_indices, dtype=None)
        metadata = hv.to_numpy(first.two_of_four_metadata, dtype=None)
        observed_position_sets = set()
        for row in range(values.shape[0]):
            for group in range(values.shape[1] // 4):
                positions = retained_indices[row, group * 2 : (group + 1) * 2]
                self.assertLess(positions[0], positions[1])
                observed_position_sets.add(tuple(int(x) for x in positions))
                expected_group = np.zeros(4, dtype=np.float32)
                for retained_index, position in enumerate(positions):
                    expected_group[position] = values[row, group * 4 + position]
                    self.assertEqual(
                        compressed[row, group * 2 + retained_index],
                        values[row, group * 4 + position],
                    )
                np.testing.assert_array_equal(
                    pruned[row, group * 4 : (group + 1) * 4],
                    expected_group,
                )
                nibble = int(positions[0]) | (int(positions[1]) << 2)
                metadata_byte = metadata[row, group // 2]
                observed_nibble = (
                    metadata_byte & 0xF if group % 2 == 0 else metadata_byte >> 4
                )
                self.assertEqual(observed_nibble, nibble)
        self.assertGreater(len(observed_position_sets), 1)

    def test_structured_sparsity_handles_strided_input_and_packed_values(self):
        storage = np.full(20, -99.0, dtype=np.float32)
        logical = np.arange(1, 17, dtype=np.float32).reshape(2, 8)
        storage[0:8] = logical[0]
        storage[10:18] = logical[1]
        strided = hv.Tensor.from_storage(
            hv.ScalarType.Float32, [2, 8], storage.tobytes(), [10, 1], offset=0
        )
        pattern = hv.StructuredSparsityPattern()
        pattern.index_order = [0, 1]
        pattern.axis = 1
        pattern.fixed_positions = [1, 3]
        observed = hv.apply_structured_sparsity(
            strided, pattern, emit_two_of_four_metadata=False
        )
        expected = np.zeros_like(logical)
        expected[:, 1::4] = logical[:, 1::4]
        expected[:, 3::4] = logical[:, 3::4]
        np.testing.assert_array_equal(
            hv.to_numpy(observed.pruned, dtype=None), expected
        )
        self.assertEqual(observed.pruned.strides, [8, 1])
        self.assertEqual(observed.compressed.strides, [4, 1])
        self.assertEqual(observed.retained_indices.strides, [4, 1])

        fp4_values = np.asarray(
            [[-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.5]],
            dtype=np.float32,
        )
        fp4 = hv.apply_structured_sparsity(
            hv.from_numpy(fp4_values, hv.ScalarType.Float4E2M1),
            pattern,
            emit_two_of_four_metadata=False,
        )
        expected_fp4 = np.zeros_like(fp4_values)
        expected_fp4[:, 1::4] = fp4_values[:, 1::4]
        expected_fp4[:, 3::4] = fp4_values[:, 3::4]
        np.testing.assert_array_equal(hv.to_numpy(fp4.pruned, dtype=None), expected_fp4)
