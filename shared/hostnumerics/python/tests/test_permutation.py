# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import ctypes
import itertools
import unittest

import hostnumerics as hn

SIZE_MAX = (1 << (8 * ctypes.sizeof(ctypes.c_size_t))) - 1


class PermutationTests(unittest.TestCase):
    def test_every_permutation_against_itertools(self):
        for size in range(5):
            values = [f"v{index}" for index in range(size)]
            for entries in itertools.permutations(range(size)):
                with self.subTest(entries=entries):
                    permutation = hn.Permutation(list(entries))
                    self.assertEqual(len(permutation), size)
                    self.assertEqual(list(permutation), list(entries))
                    self.assertEqual(permutation.entries, list(entries))
                    self.assertEqual(
                        [permutation[position] for position in range(size)],
                        list(entries),
                    )
                    self.assertEqual(
                        permutation.apply(values), [values[entry] for entry in entries]
                    )
                    inverse = permutation.inverse()
                    self.assertEqual(
                        list(inverse),
                        sorted(range(size), key=lambda position: entries[position]),
                    )
                    self.assertEqual(inverse.apply(permutation.apply(values)), values)

    def test_named_orders(self):
        for size in range(5):
            self.assertEqual(list(hn.Permutation.identity(size)), list(range(size)))
            self.assertEqual(
                list(hn.Permutation.reverse(size)), list(reversed(range(size)))
            )
        self.assertEqual(repr(hn.Permutation.reverse(3)), "Permutation([2, 1, 0])")
        self.assertEqual(repr(hn.Permutation([])), "Permutation([])")

    def test_invalid_entries(self):
        for entries in ([0, 0], [1, 2], [1], [0, 1, SIZE_MAX]):
            with self.subTest(entries=entries), self.assertRaisesRegex(
                RuntimeError, "each value must appear exactly once"
            ):
                hn.Permutation(entries)
        with self.assertRaisesRegex(
            RuntimeError, r"entry 2 at position 2 is repeated or outside \[0, 3\)"
        ):
            hn.Permutation([0, 2, 2])
        for entries in ([0, -1], [0.5], [SIZE_MAX + 1], "01"):
            with self.subTest(entries=entries), self.assertRaises(TypeError):
                hn.Permutation(entries)
        with self.assertRaises(TypeError):
            hn.Permutation()
        permutation = hn.Permutation([1, 0])
        with self.assertRaises(IndexError):
            permutation[2]
        with self.assertRaisesRegex(RuntimeError, "size 2 to 3 values"):
            permutation.apply([1, 2, 3])

    def test_values(self):
        permutation = hn.Permutation([2, 0, 1])
        self.assertEqual(permutation, hn.Permutation([2, 0, 1]))
        self.assertNotEqual(permutation, permutation.inverse())
        self.assertEqual(len({permutation, hn.Permutation([2, 0, 1])}), 1)
        self.assertNotEqual(permutation, [2, 0, 1])
        with self.assertRaises(AttributeError):
            permutation.entries = [0, 1, 2]


if __name__ == "__main__":
    unittest.main()
