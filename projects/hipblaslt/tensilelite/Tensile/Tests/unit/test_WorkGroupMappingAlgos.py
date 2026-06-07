# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Unit tests for Tensile/Components/WorkGroupMappingAlgos.py
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from types import SimpleNamespace
from conftest import create_mock_writer


@pytest.mark.unit
class TestScalarUInt24DivideAndRemainderPair:
    """Test scalarUInt24DivideAndRemainderPair function"""

    def test_generates_division_module(self):
        """Should generate module with quotient and remainder instructions"""
        from Tensile.Components.WorkGroupMappingAlgos import scalarUInt24DivideAndRemainderPair

        qReg = [0, 1]
        dReg = [2, 3]
        divReg = [4, 5]
        rReg = [6, 7]
        tmpVgprRes = SimpleNamespace(idx=10)
        wavewidth = 64

        module = scalarUInt24DivideAndRemainderPair(
            qReg, dReg, divReg, rReg, tmpVgprRes, wavewidth,
            doRemainder=True, doQuotient=True
        )

        # Verify module contains instructions
        instructions = module.items()
        assert len(instructions) > 0

        # Verify quotient registers are written (VReadlaneB32 to qReg)
        quotient_writes = [inst for inst in instructions
                          if hasattr(inst, 'dst') and
                          hasattr(inst.dst, 'regIdx') and
                          inst.dst.regIdx in qReg]
        assert len(quotient_writes) == 2, "Should write both quotient registers"

        # Verify remainder registers are written (VReadlaneB32 to rReg)
        remainder_writes = [inst for inst in instructions
                           if hasattr(inst, 'dst') and
                           hasattr(inst.dst, 'regIdx') and
                           inst.dst.regIdx in rReg]
        assert len(remainder_writes) == 2, "Should write both remainder registers"

    def test_quotient_only_mode(self):
        """Should generate module for quotient-only division"""
        from Tensile.Components.WorkGroupMappingAlgos import scalarUInt24DivideAndRemainderPair

        qReg = [0, 1]
        dReg = [2, 3]
        divReg = [4, 5]
        rReg = [6, 7]
        tmpVgprRes = SimpleNamespace(idx=10)
        wavewidth = 64

        module = scalarUInt24DivideAndRemainderPair(
            qReg, dReg, divReg, rReg, tmpVgprRes, wavewidth,
            doRemainder=False, doQuotient=True
        )

        instructions = module.items()

        # Verify quotient registers are written
        quotient_writes = [inst for inst in instructions
                          if hasattr(inst, 'dst') and
                          hasattr(inst.dst, 'regIdx') and
                          inst.dst.regIdx in qReg]
        assert len(quotient_writes) == 2, "Should write both quotient registers"

        # Verify remainder registers are NOT written (doRemainder=False)
        remainder_writes = [inst for inst in instructions
                           if hasattr(inst, 'dst') and
                           hasattr(inst.dst, 'regIdx') and
                           inst.dst.regIdx in rReg]
        assert len(remainder_writes) == 0, "Should not write remainder registers when doRemainder=False"

    def test_remainder_only_mode(self):
        """Should generate module for remainder-only division"""
        from Tensile.Components.WorkGroupMappingAlgos import scalarUInt24DivideAndRemainderPair

        qReg = [0, 1]
        dReg = [2, 3]
        divReg = [4, 5]
        rReg = [6, 7]
        tmpVgprRes = SimpleNamespace(idx=10)
        wavewidth = 64

        module = scalarUInt24DivideAndRemainderPair(
            qReg, dReg, divReg, rReg, tmpVgprRes, wavewidth,
            doRemainder=True, doQuotient=False
        )

        instructions = module.items()

        # Verify remainder registers are written
        remainder_writes = [inst for inst in instructions
                           if hasattr(inst, 'dst') and
                           hasattr(inst.dst, 'regIdx') and
                           inst.dst.regIdx in rReg]
        assert len(remainder_writes) == 2, "Should write both remainder registers"

        # Verify quotient registers are NOT written (doQuotient=False)
        quotient_writes = [inst for inst in instructions
                          if hasattr(inst, 'dst') and
                          hasattr(inst.dst, 'regIdx') and
                          inst.dst.regIdx in qReg]
        assert len(quotient_writes) == 0, "Should not write quotient registers when doQuotient=False"

    def test_wavewidth_32(self):
        """Should handle wavewidth 32"""
        from Tensile.Components.WorkGroupMappingAlgos import scalarUInt24DivideAndRemainderPair

        qReg = [0, 1]
        dReg = [2, 3]
        divReg = [4, 5]
        rReg = [6, 7]
        tmpVgprRes = SimpleNamespace(idx=10)
        wavewidth = 32  # Different wavewidth

        module = scalarUInt24DivideAndRemainderPair(
            qReg, dReg, divReg, rReg, tmpVgprRes, wavewidth
        )

        instructions = module.items()

        # Verify quotient and remainder registers are written
        quotient_writes = [inst for inst in instructions
                          if hasattr(inst, 'dst') and
                          hasattr(inst.dst, 'regIdx') and
                          inst.dst.regIdx in qReg]
        assert len(quotient_writes) == 2, "Should write both quotient registers"

        remainder_writes = [inst for inst in instructions
                           if hasattr(inst, 'dst') and
                           hasattr(inst.dst, 'regIdx') and
                           inst.dst.regIdx in rReg]
        assert len(remainder_writes) == 2, "Should write both remainder registers"



