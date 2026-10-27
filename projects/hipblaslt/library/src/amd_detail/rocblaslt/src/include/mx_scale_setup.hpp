// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Attaching an MX block-scale descriptor and declaring the layout of the buffer
// behind it are one operation, not two.
//
// ContractionProblemGemm::setMXScaleA/B shape the descriptor from the padding
// mode. On gfx950 a plain Block_32_UE8M0 caller allocated a packed buffer, so
// the mode is Packed. A Block_32_UE8M0_32_8_EXT caller pre-swizzled into the
// padded gfx950 layout, so the mode is Gfx950 and scalesArePacked is false.
// generateCustomCall reads customKernelMXScalesArePacked() when it has to
// rebuild a stride the descriptor does not already describe. The TensileLite
// benchmark client leaves that flag false when it allocated the padded buffer.
//
// Keeping the two calls paired here is what stops the second one from being
// dropped on its own: a caller that attaches hipBLASLt's scales gets the
// declaration with them, and the pairing is pinned by
// clients/tests/unit/mx_scale_setup_gtest.cpp. Only tensile_host.cpp includes
// this header from the library side, so the "only tensile_host.cpp includes
// Tensile headers" rule at the top of that file still holds.

#include <Tensile/ContractionProblem.hpp>

namespace rocblaslt
{
    /// Attach hipBLASLt's MX block scale for A.
    /// scalesArePacked is false only for the pre-swizzled gfx950 buffer.
    inline void setHostMXScaleA(TensileLite::ContractionProblemGemm& problem,
                                rocisa::DataType                     mxType,
                                int                                  mxBlock,
                                TensileLite::MXScalePadding          padding,
                                bool                                 scalesArePacked = true)
    {
        problem.setMXScaleA(mxType, mxBlock, {}, padding);
        problem.setCustomKernelMXScalesArePacked(scalesArePacked);
    }

    /// Attach hipBLASLt's MX block scale for B.
    /// scalesArePacked is false only for the pre-swizzled gfx950 buffer.
    inline void setHostMXScaleB(TensileLite::ContractionProblemGemm& problem,
                                rocisa::DataType                     mxType,
                                int                                  mxBlock,
                                TensileLite::MXScalePadding          padding,
                                bool                                 scalesArePacked = true)
    {
        problem.setMXScaleB(mxType, mxBlock, {}, padding);
        problem.setCustomKernelMXScalesArePacked(scalesArePacked);
    }
}
