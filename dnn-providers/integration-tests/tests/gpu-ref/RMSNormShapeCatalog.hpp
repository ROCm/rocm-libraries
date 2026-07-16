// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "RMSNormTestCase.hpp"

namespace gpu_rmsnorm_ref_test
{

using hipdnn_data_sdk::utilities::TensorLayout;

inline std::vector<RMSNormTestCase> getRMSnormSmall4DTestCases()
{
    return {{{2, 3, 4, 4}, {1, 3, 4, 4}, TensorLayout::NCHW},
            {{32, 3, 8, 8}, {1, 1, 8, 8}, TensorLayout::NCHW},
            {{1, 3, 14, 14}, {1, 1, 1, 14}, TensorLayout::NCHW},
            {{16, 32, 24, 16}, {1, 32, 24, 16}, TensorLayout::NCHW},
            {{16, 40, 48, 32}, {1, 1, 48, 32}, TensorLayout::NCHW},

            {{2, 3, 4, 4}, {1, 3, 4, 4}, TensorLayout::NHWC},
            {{32, 3, 8, 8}, {1, 1, 8, 8}, TensorLayout::NHWC},
            {{1, 3, 14, 14}, {1, 1, 1, 14}, TensorLayout::NHWC},
            {{16, 32, 24, 16}, {1, 32, 24, 16}, TensorLayout::NHWC},
            {{16, 40, 48, 32}, {1, 1, 48, 32}, TensorLayout::NHWC}};
}

inline std::vector<RMSNormTestCase> getRMSnormSmall5DTestCases()
{
    return {{{2, 3, 3, 1, 1}, {1, 3, 3, 1, 1}, TensorLayout::NCDHW},
            {{2, 3, 4, 2, 2}, {1, 1, 4, 2, 2}, TensorLayout::NCDHW},
            {{2, 3, 4, 2, 2}, {1, 1, 1, 2, 2}, TensorLayout::NCDHW},
            {{2, 3, 4, 2, 2}, {1, 1, 1, 1, 2}, TensorLayout::NCDHW},
            {{4, 8, 2, 4, 4}, {1, 8, 2, 4, 4}, TensorLayout::NCDHW},

            {{2, 3, 3, 1, 1}, {1, 3, 3, 1, 1}, TensorLayout::NDHWC},
            {{2, 3, 4, 2, 2}, {1, 1, 4, 2, 2}, TensorLayout::NDHWC},
            {{2, 3, 4, 2, 2}, {1, 1, 1, 2, 2}, TensorLayout::NDHWC},
            {{2, 3, 4, 2, 2}, {1, 1, 1, 1, 2}, TensorLayout::NDHWC},
            {{4, 8, 2, 4, 4}, {1, 8, 2, 4, 4}, TensorLayout::NDHWC}};
}

inline std::vector<RMSNormTestCase> getRMSnormMedium4DTestCases()
{
    return {{{1, 3, 14, 14}, {1, 3, 14, 14}, TensorLayout::NCHW},
            {{1, 3, 14, 14}, {1, 1, 14, 14}, TensorLayout::NCHW},
            {{1, 3, 14, 14}, {1, 1, 1, 14}, TensorLayout::NCHW},
            {{1, 256, 1, 1}, {1, 256, 1, 1}, TensorLayout::NCHW},
            {{2, 3, 1, 1}, {1, 3, 1, 1}, TensorLayout::NCHW},
            {{32, 1, 14, 14}, {1, 1, 14, 14}, TensorLayout::NCHW},
            {{32, 3, 1, 14}, {1, 3, 1, 14}, TensorLayout::NCHW},
            {{32, 3, 14, 1}, {1, 3, 14, 1}, TensorLayout::NCHW},
            {{32, 3, 14, 1}, {1, 1, 14, 1}, TensorLayout::NCHW},
            {{16, 32, 192, 128}, {1, 32, 192, 128}, TensorLayout::NCHW},
            {{16, 32, 192, 128}, {1, 1, 192, 128}, TensorLayout::NCHW},
            {{16, 64, 225, 225}, {1, 64, 225, 225}, TensorLayout::NCHW},
            {{16, 64, 225, 225}, {1, 1, 1, 225}, TensorLayout::NCHW},
            {{16, 128, 56, 56}, {1, 128, 56, 56}, TensorLayout::NCHW},
            {{16, 128, 56, 56}, {1, 1, 56, 56}, TensorLayout::NCHW},

            {{1, 3, 14, 14}, {1, 3, 14, 14}, TensorLayout::NHWC},
            {{1, 3, 14, 14}, {1, 1, 14, 14}, TensorLayout::NHWC},
            {{1, 3, 14, 14}, {1, 1, 1, 14}, TensorLayout::NHWC},
            {{1, 256, 1, 1}, {1, 256, 1, 1}, TensorLayout::NHWC},
            {{2, 3, 1, 1}, {1, 3, 1, 1}, TensorLayout::NHWC},
            {{32, 1, 14, 14}, {1, 1, 14, 14}, TensorLayout::NHWC},
            {{32, 3, 1, 14}, {1, 3, 1, 14}, TensorLayout::NHWC},
            {{32, 3, 14, 1}, {1, 3, 14, 1}, TensorLayout::NHWC},
            {{32, 3, 14, 1}, {1, 1, 14, 1}, TensorLayout::NHWC},
            {{16, 32, 192, 128}, {1, 32, 192, 128}, TensorLayout::NHWC},
            {{16, 32, 192, 128}, {1, 1, 192, 128}, TensorLayout::NHWC},
            {{16, 64, 225, 225}, {1, 64, 225, 225}, TensorLayout::NHWC},
            {{16, 64, 225, 225}, {1, 1, 1, 225}, TensorLayout::NHWC},
            {{16, 128, 56, 56}, {1, 128, 56, 56}, TensorLayout::NHWC},
            {{16, 128, 56, 56}, {1, 1, 56, 56}, TensorLayout::NHWC}};
}

inline std::vector<RMSNormTestCase> getRMSnormMedium5DTestCases()
{
    return {{{16, 3, 8, 14, 14}, {1, 3, 8, 14, 14}, TensorLayout::NCDHW},
            {{16, 32, 4, 48, 32}, {1, 32, 4, 48, 32}, TensorLayout::NCDHW},
            {{16, 32, 4, 48, 32}, {1, 1, 1, 48, 32}, TensorLayout::NCDHW},
            {{8, 64, 4, 28, 28}, {1, 64, 4, 28, 28}, TensorLayout::NCDHW},
            {{8, 64, 4, 28, 28}, {1, 1, 4, 28, 28}, TensorLayout::NCDHW},

            {{16, 3, 8, 14, 14}, {1, 3, 8, 14, 14}, TensorLayout::NDHWC},
            {{16, 32, 4, 48, 32}, {1, 32, 4, 48, 32}, TensorLayout::NDHWC},
            {{16, 32, 4, 48, 32}, {1, 1, 1, 48, 32}, TensorLayout::NDHWC},
            {{8, 64, 4, 28, 28}, {1, 64, 4, 28, 28}, TensorLayout::NDHWC},
            {{8, 64, 4, 28, 28}, {1, 1, 4, 28, 28}, TensorLayout::NDHWC}};
}

inline std::vector<RMSNormTestCase> getRMSnormLarge4DTestCases()
{
    return {{{16, 288, 48, 32}, {1, 288, 48, 32}, TensorLayout::NCHW},
            {{16, 288, 48, 32}, {1, 1, 48, 32}, TensorLayout::NCHW},
            {{16, 576, 1, 30}, {1, 576, 1, 30}, TensorLayout::NCHW},
            {{16, 576, 1, 30}, {1, 1, 1, 30}, TensorLayout::NCHW},
            {{16, 2048, 16, 32}, {1, 2048, 16, 32}, TensorLayout::NCHW},
            {{16, 2048, 16, 32}, {1, 1, 16, 32}, TensorLayout::NCHW},
            {{128, 35, 48, 32}, {1, 35, 48, 32}, TensorLayout::NCHW},
            {{128, 512, 24, 48}, {1, 512, 24, 48}, TensorLayout::NCHW},
            {{128, 512, 24, 48}, {1, 1, 24, 48}, TensorLayout::NCHW},

            {{16, 288, 48, 32}, {1, 288, 48, 32}, TensorLayout::NHWC},
            {{16, 288, 48, 32}, {1, 1, 48, 32}, TensorLayout::NHWC},
            {{16, 576, 1, 30}, {1, 576, 1, 30}, TensorLayout::NHWC},
            {{16, 576, 1, 30}, {1, 1, 1, 30}, TensorLayout::NHWC},
            {{16, 2048, 16, 32}, {1, 2048, 16, 32}, TensorLayout::NHWC},
            {{16, 2048, 16, 32}, {1, 1, 16, 32}, TensorLayout::NHWC},
            {{128, 35, 48, 32}, {1, 35, 48, 32}, TensorLayout::NHWC},
            {{128, 512, 24, 48}, {1, 512, 24, 48}, TensorLayout::NHWC},
            {{128, 512, 24, 48}, {1, 1, 24, 48}, TensorLayout::NHWC}};
}

inline std::vector<RMSNormTestCase> getRMSnormLarge5DTestCases()
{
    return {{{16, 128, 8, 24, 16}, {1, 128, 8, 24, 16}, TensorLayout::NCDHW},
            {{16, 256, 4, 32, 32}, {1, 256, 4, 32, 32}, TensorLayout::NCDHW},
            {{16, 256, 4, 32, 32}, {1, 1, 4, 32, 32}, TensorLayout::NCDHW},
            {{32, 128, 8, 16, 16}, {1, 128, 8, 16, 16}, TensorLayout::NCDHW},
            {{32, 128, 8, 16, 16}, {1, 1, 1, 16, 16}, TensorLayout::NCDHW},

            {{16, 128, 8, 24, 16}, {1, 128, 8, 24, 16}, TensorLayout::NDHWC},
            {{16, 256, 4, 32, 32}, {1, 256, 4, 32, 32}, TensorLayout::NDHWC},
            {{16, 256, 4, 32, 32}, {1, 1, 4, 32, 32}, TensorLayout::NDHWC},
            {{32, 128, 8, 16, 16}, {1, 128, 8, 16, 16}, TensorLayout::NDHWC},
            {{32, 128, 8, 16, 16}, {1, 1, 1, 16, 16}, TensorLayout::NDHWC}};
}

} // namespace gpu_rmsnorm_ref_test
