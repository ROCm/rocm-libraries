# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for StreamK=5 hybrid mode codegen intent.

These tests import Tensile modules directly and inspect emitted rocisa
instructions rather than matching Python source text. Signature layout
is covered by sk_hybrid.yaml kernel tests.
"""

# Prime the component registry before StreamK imports (avoids circular import).
from Tensile.KernelWriterAssembly import KernelWriterAssembly  # noqa: F401

from rocisa.instruction import SAndB32, SLShiftRightB32

from Tensile.Common.ValidParameters import validParameters
from Tensile.Components.StreamK import (
    StreamK,
    StreamKHybrid,
    StreamKTwoTileDPFirst,
    streamKVariantClass,
)

from streamk5_test_helpers import (
    SK5_KERNARG_ALIASES,
    SK5_PERSISTENT_ALIASES,
    collect_sk5_regset_aliases,
    emit_mode_extraction_module,
    reg_name,
)


class TestStreamK5ValidParameters:
    def test_streamk_enum_includes_5(self):
        assert 5 in validParameters["StreamK"]
        assert validParameters["StreamK"] == [0, 1, 2, 3, 4, 5]


class TestStreamK5Component:
    def test_streamk_hybrid_is_registered_variant(self):
        assert streamKVariantClass(5) is StreamKHybrid
        assert StreamKHybrid.kernel == {"StreamK": 5}

    def test_component_dispatches_streamk_5_to_hybrid(self, mock_streamk_writer):
        impl = StreamK.find(mock_streamk_writer(5))
        assert isinstance(impl, StreamKHybrid)

    def test_component_dispatches_streamk_3_to_static_path(self, mock_streamk_writer):
        impl = StreamK.find(mock_streamk_writer(3))
        assert isinstance(impl, StreamKTwoTileDPFirst)


class TestStreamK5ModeExtraction:
    def test_mode_extraction_shifts_bit_30(self):
        module = emit_mode_extraction_module()
        shift_inst = next(
            inst for inst in module.flatitems() if isinstance(inst, SLShiftRightB32)
        )
        params = list(shift_inst.getParams())
        assert reg_name(params[0]) == "sgprStreamKHybridMode"
        assert reg_name(params[1]) == "sgprMagicShiftItersPerTile"
        assert params[2] == hex(30)

    def test_mode_extraction_masks_magic_shift_with_bfffffff(self):
        module = emit_mode_extraction_module()
        mask_insts = [
            inst for inst in module.flatitems() if isinstance(inst, SAndB32)
        ]
        clear_inst = next(
            inst
            for inst in mask_insts
            if reg_name(list(inst.getParams())[0]) == "sgprMagicShiftItersPerTile"
        )
        params = list(clear_inst.getParams())
        assert params[2] == hex(0xBFFFFFFF)

    def test_mode_extraction_does_not_use_bit_31(self):
        module = emit_mode_extraction_module()
        for inst in module.flatitems():
            if isinstance(inst, SLShiftRightB32):
                assert list(inst.getParams())[2] != hex(31)


class TestStreamK5RegSetAliasing:
    def test_hybrid_variant_requests_parallel_reduction_aliases(self):
        variant = streamKVariantClass(5)()
        assert variant.emitsParallelReductionSgprAliases is True

    def test_macro_and_set_emits_sk4_to_sk3_kernarg_aliases(self):
        aliases = collect_sk5_regset_aliases()
        for expected in SK5_KERNARG_ALIASES:
            assert expected in aliases

    def test_macro_and_set_emits_persistent_slot_aliases(self):
        aliases = collect_sk5_regset_aliases()
        for expected in SK5_PERSISTENT_ALIASES:
            assert expected in aliases
