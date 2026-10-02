# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import dataclasses
import os
import unittest
from unittest import mock

from codegen.ops.fmha_fwd import (
    CompatibilityRuleFactoryGfx950,
    FmhaFwdPipeline,
    FmhaFwdTileSize,
    KernelContext,
    ProblemContext,
    get_fwd_blobs,
    is_compatible,
)


def all_traits(pool):
    for by_arch in pool.pool.values():
        for by_dtype in by_arch.values():
            for bucket in by_dtype.values():
                yield from bucket


def trload_pipeline(spad, skpad):
    return FmhaFwdPipeline(
        "qr_async_trload", "row", spad, skpad, "f", "f", "f", "no", "t", "f", "no", "s_no", "f", "t", "f"
    )  # fmt: skip


TRLOAD_TILE = FmhaFwdTileSize(16, 32, 64, 128, 32, 128, 1, 1, 1, 1, 1, 1, 16, 16, 32, 16, 16, 32, -1)  # fmt: skip


class FmhaFwdCodegenTest(unittest.TestCase):
    # (receipt, optdim_list, expect batch trload, expect group trload)
    RECEIPT_CONFIGS = (
        (0, [32, 64, 80, 128, 256], True, True),
        (2, [-1], True, True),
        (3, [-1], True, True),
        (4, [-1], True, True),
        (100, [-1], True, False),
        (200, [-1], False, True),
        (600, [-1], True, True),
        (700, [-1], True, True),
    )

    def get_traits(self, receipt, optdim_list):
        with mock.patch.dict(os.environ, {"CK_TILE_FMHA_FWD_CUSTOM_FACTORY": "0"}):
            pool, _ = get_fwd_blobs(
                ["gfx950"],
                kernel_filter="",
                receipt=receipt,
                optdim_list=optdim_list,
                mask_impl="simplified",
            )
        return [t for t in all_traits(pool) if t.pipeline_tag == "qr_async_trload"]

    def test_trload_sequence_padding_only_in_group_mode(self):
        for receipt, optdim_list, expect_batch, expect_group in self.RECEIPT_CONFIGS:
            with self.subTest(receipt=receipt, optdim_list=optdim_list):
                traits = self.get_traits(receipt, optdim_list)
                batch = [t for t in traits if t.mode == "batch"]
                group = [t for t in traits if t.mode == "group"]

                self.assertEqual(len(batch) > 0, expect_batch)
                self.assertEqual(len(group) > 0, expect_group)

                for trait in batch:
                    self.assertEqual((trait.spad, trait.skpad), ("f", "f"))
                    self.assertEqual(trait.scheck, "true")
                    self.assertEqual(trait.skcheck, "true")
                for trait in group:
                    self.assertEqual((trait.spad, trait.skpad), ("t", "t"))

    def test_batch_trload_scheck_rejects_sequence_padding(self):
        trait = self.get_traits(4, [-1])
        trait = next(t for t in trait if t.mode == "batch")
        with self.assertRaises(ValueError):
            dataclasses.replace(trait, spad="t").scheck
        with self.assertRaises(ValueError):
            dataclasses.replace(trait, skpad="t").skcheck

    def test_compatibility_keeps_padded_trload_for_group_mode_only(self):
        rules = CompatibilityRuleFactoryGfx950.get_rules()
        padded = KernelContext(
            tile=TRLOAD_TILE, pipeline=trload_pipeline("t", "t"), mask_impl="simplified"
        )
        unpadded = KernelContext(
            tile=TRLOAD_TILE, pipeline=trload_pipeline("f", "f"), mask_impl="simplified"
        )

        def problem(mode):
            return ProblemContext(dtype="bf16", mode=mode, hdim=128, hdim_v=128)

        self.assertFalse(is_compatible(problem("batch"), padded, rules))
        self.assertTrue(is_compatible(problem("batch"), unpadded, rules))
        self.assertTrue(is_compatible(problem("group"), padded, rules))


if __name__ == "__main__":
    unittest.main()
