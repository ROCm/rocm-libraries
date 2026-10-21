// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Attaching an MX block-scale descriptor and declaring the layout of the buffer
// behind it are one operation, not two.
//
// ContractionProblemGemm::setMXScaleA/B pad the descriptor on gfx950 for the
// Tensile-generated HostPreSwizzle kernels. hipBLASLt's own scale pointer is the
// user's plain Block_32_UE8M0 buffer, which is packed at ceil(K/mxBlock), so
// ContractionSolution::generateCustomCall has to hand an external custom kernel
// the packed pitch rather than the descriptor's padded one -- it asks
// customKernelMXScalesArePacked(), which defaults to false for the TensileLite
// benchmark client, whose scale buffer really is the padded one.
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
    /// Attach hipBLASLt's MX block scale for A and declare the buffer packed.
    inline void setHostMXScaleA(TensileLite::ContractionProblemGemm& problem,
                                rocisa::DataType                     mxType,
                                int                                  mxBlock,
                                bool                                 padScaleTensorFreeDim)
    {
        problem.setMXScaleA(mxType, mxBlock, {}, padScaleTensorFreeDim);
        problem.setCustomKernelMXScalesArePacked(true);
    }

    /// Attach hipBLASLt's MX block scale for B and declare the buffer packed.
    inline void setHostMXScaleB(TensileLite::ContractionProblemGemm& problem,
                                rocisa::DataType                     mxType,
                                int                                  mxBlock,
                                bool                                 padScaleTensorFreeDim)
    {
        problem.setMXScaleB(mxType, mxBlock, {}, padScaleTensorFreeDim);
        problem.setCustomKernelMXScalesArePacked(true);
    }
}
