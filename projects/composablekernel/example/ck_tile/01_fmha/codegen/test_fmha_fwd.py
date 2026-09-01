# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import unittest

from codegen.ops.fmha_fwd import get_fwd_blobs


def all_traits(pool):
    for by_arch in pool.pool.values():
        for by_dtype in by_arch.values():
            for bucket in by_dtype.values():
                yield from bucket


class FmhaFwdCodegenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pool, _ = get_fwd_blobs(
            ["gfx950"],
            kernel_filter="",
            receipt=4,
            optdim_list=[-1],
            mask_impl="simplified",
        )

    def test_unpadded_trload_instances_require_aligned_sequence_lengths(self):
        q_guards = 0
        k_guards = 0

        for trait in all_traits(self.pool):
            if trait.pipeline_tag != "qr_async_trload" or trait.mode == "group":
                continue

            if trait.spad == "f":
                self.assertEqual(
                    trait.scheck,
                    f"a.seqlen_q % {trait.bm0} == 0",
                )
                q_guards += 1

            if trait.skpad == "f":
                self.assertEqual(
                    trait.skcheck,
                    "(a.cu_seqlen_k_ptr == nullptr) && "
                    f"(a.seqlen_k != 0 && a.seqlen_k % {trait.bn0} == 0)",
                )
                k_guards += 1

        self.assertGreater(q_guards, 0)
        self.assertGreater(k_guards, 0)


if __name__ == "__main__":
    unittest.main()
