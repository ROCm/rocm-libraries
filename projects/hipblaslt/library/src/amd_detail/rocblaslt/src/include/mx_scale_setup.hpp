// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// hipBLASLt supplies compact MX scales to external custom kernels, except for a
// pre-swizzled operand, while the TensileLite client allocates scales from the
// padded descriptor. Attach the descriptor and its caller-layout flag together
// so both describe the same input.

#include <Tensile/ContractionProblem.hpp>

namespace rocblaslt
{
    /// Attach hipBLASLt's MX block scale for A and declare its buffer layout.
    /// scalesArePacked is false only for the pre-swizzled gfx950 buffer.
    inline void setHostMXScaleA(TensileLite::ContractionProblemGemm& problem,
                                rocisa::DataType                     mxType,
                                int                                  mxBlock,
                                TensileLite::MXScalePadding          padding,
                                bool                                 scalesArePacked = true)
    {
        problem.setMXScaleA(mxType, mxBlock, {}, padding);
        problem.setCustomKernelMXScaleAIsPacked(scalesArePacked);
    }

    /// Attach hipBLASLt's MX block scale for B and declare its buffer layout.
    /// scalesArePacked is false only for the pre-swizzled gfx950 buffer.
    inline void setHostMXScaleB(TensileLite::ContractionProblemGemm& problem,
                                rocisa::DataType                     mxType,
                                int                                  mxBlock,
                                TensileLite::MXScalePadding          padding,
                                bool                                 scalesArePacked = true)
    {
        problem.setMXScaleB(mxType, mxBlock, {}, padding);
        problem.setCustomKernelMXScaleBIsPacked(scalesArePacked);
    }
}
