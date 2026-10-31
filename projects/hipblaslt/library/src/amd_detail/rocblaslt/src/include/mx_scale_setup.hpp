// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// hipBLASLt supplies compact MX scales to external custom kernels, while the
// TensileLite client allocates scales from the padded descriptor. Attach the
// descriptor and its caller-layout flag together so both describe the same input.

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
