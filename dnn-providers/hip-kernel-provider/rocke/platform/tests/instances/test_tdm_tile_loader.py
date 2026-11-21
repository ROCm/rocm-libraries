# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host tests of TDM window validation and descriptor preparation."""

from __future__ import annotations

import unittest
from unittest.mock import Mock

from rocke.core.ir import F16, I8, I64, IRBuilder, PtrType
from rocke.helpers.loads import TdmSourceWindow2D, TdmTileLoader2D


class TestTdmTileLoader(unittest.TestCase):
    def test_prepare_rejects_invalid_source_before_emitting(self):
        loader = TdmTileLoader2D(16, 16, 2)
        for ptr_type, msg in (
            (PtrType(I8, "global"), "pointee width"),
            (PtrType(F16, "constant"), "global pointer"),
            (I64, "global pointer"),
        ):
            with self.subTest(ptr_type=ptr_type):
                ptr = IRBuilder("tdm_source").param("ptr", ptr_type)
                source = TdmSourceWindow2D(ptr, 0, 1, 4, 16, 12, 16)
                b = Mock()
                with self.assertRaisesRegex(ValueError, msg):
                    loader.prepare(b, source, "lds_slot", zero=0)
                self.assertEqual(b.mock_calls, [])

    def test_prepare_rebases_window_and_does_not_issue(self):
        b = Mock()
        b.zext.side_effect = lambda value, _: value
        b.mul.side_effect = lambda a, c: a * c
        b.add.side_effect = lambda a, c: a + c
        b.sub.side_effect = lambda a, c: a - c
        b.smax.side_effect = max
        b.global_addr_of.return_value = "rebased"
        groups = ("d0", "d1", "d2", "d3", "d4")
        b.tdm_descriptor_2d.return_value = groups
        ptr = IRBuilder("tdm_source").param("ptr", PtrType(F16, "global"))
        source = TdmSourceWindow2D(
            ptr=ptr,
            base_offset=2**32,
            row_origin=2**16,
            col_origin=32,
            row_extent=2**16 + 9,
            cols_remaining=16,
            row_stride=2**16 + 16,
        )
        loader = TdmTileLoader2D(32, 32, 2, pad_interval=2, pad_amount=16)

        load = loader.prepare(b, source, "lds_slot", zero=0)

        self.assertEqual(
            [call.args for call in b.zext.call_args_list],
            [(source.row_origin, I64), (source.row_stride, I64), (32, I64)],
        )
        b.global_addr_of.assert_called_once_with(
            ptr, 2**32 + (2**16) * (2**16 + 16) + 32
        )
        b.tdm_descriptor_2d.assert_called_once_with(
            "rebased",
            "lds_slot",
            elem_bytes=2,
            tensor_dim0=16,
            tensor_dim1=9,
            row_stride=2**16 + 16,
            tile_dim0=32,
            tile_dim1=32,
            pad_interval=2,
            pad_amount=16,
        )
        b.tensor_load_to_lds.assert_not_called()
        load.issue(b, cachepolicy=0)
        b.tensor_load_to_lds.assert_called_once_with(*groups, cachepolicy=0)
        b.s_wait_tensorcnt.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
