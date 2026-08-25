// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "MatmulTestCase.hpp"

namespace gpu_matmul_ref_test
{

using hipdnn_data_sdk::utilities::TensorLayout;

inline std::vector<MatmulTestCase> getMatmulSmall2DTestCases()
{
    return {{{2, 2}, {2, 2}}, {{3, 2}, {2, 5}}, {{2, 5}, {5, 3}}};
}

inline std::vector<MatmulTestCase> getMatmulSmall3DTestCases()
{
    return {{{2, 2, 2}, {2, 2, 2}},
            {{7, 3, 2}, {7, 2, 5}},
            {{7, 2, 5}, {7, 5, 3}},
            {{2, 2, 2}, {4, 2, 3}},
            {{6, 2, 2}, {3, 2, 3}}};
}

inline std::vector<MatmulTestCase> getMatmulSmall4DTestCases()
{
    return {{{2, 2, 2, 2}, {2, 2, 2, 2}},
            {{2, 3, 3, 2}, {2, 3, 2, 5}},
            {{2, 3, 2, 5}, {2, 3, 5, 3}},
            {{1, 6, 2, 2}, {3, 3, 2, 3}}};
}

inline std::vector<MatmulTestCase> getMatmulSmall5DTestCases()
{
    return {{{2, 2, 2, 2, 2}, {2, 2, 2, 2, 2}},
            {{2, 2, 2, 3, 2}, {2, 2, 2, 2, 3}},
            {{2, 2, 2, 2, 3}, {2, 2, 2, 3, 2}},
            {{1, 6, 3, 2, 3}, {3, 2, 6, 3, 2}}};
}

inline std::vector<MatmulTestCase> getMatmulMedium2DTestCases()
{
    return {{{64, 64}, {64, 64}}, {{15, 65}, {65, 17}}, {{63, 17}, {17, 65}}};
}

inline std::vector<MatmulTestCase> getMatmulMedium3DTestCases()
{
    return {{{2, 64, 64}, {2, 64, 64}},
            {{2, 15, 65}, {2, 65, 17}},
            {{2, 63, 17}, {2, 17, 65}},
            {{2, 16, 16}, {4, 16, 16}},
            {{6, 16, 16}, {3, 16, 16}}};
}

inline std::vector<MatmulTestCase> getMatmulMedium4DTestCases()
{
    return {{{2, 2, 2, 64, 64}, {2, 2, 2, 64, 64}},
            {{2, 2, 2, 15, 65}, {2, 2, 2, 65, 17}},
            {{2, 2, 2, 63, 17}, {2, 2, 2, 17, 65}},
            {{1, 6, 2, 16, 16}, {3, 2, 6, 16, 16}}};
}

inline std::vector<MatmulTestCase> getMatmulMedium5DTestCases()
{
    return {{{2, 2, 64, 64}, {2, 2, 64, 64}},
            {{2, 2, 15, 65}, {2, 2, 65, 17}},
            {{2, 2, 63, 17}, {2, 2, 17, 65}},
            {{1, 6, 16, 16}, {3, 3, 16, 16}}};
}

inline std::vector<MatmulTestCase> getMatmulLarge2DTestCases()
{
    return {{{512, 1}, {1, 256}}, {{129, 127}, {127, 128}}, {{128, 129}, {129, 127}}};
}

inline std::vector<MatmulTestCase> getMatmulLarge3DTestCases()
{
    return {{{16, 512, 1}, {16, 1, 256}},
            {{16, 129, 127}, {16, 127, 128}},
            {{16, 128, 129}, {16, 129, 127}},
            {{16, 128, 129}, {2, 129, 127}},
            {{1, 128, 129}, {4, 129, 127}}};
}

inline std::vector<MatmulTestCase> getMatmulLarge4DTestCases()
{
    return {{{4, 16, 512, 1}, {4, 16, 1, 256}},
            {{4, 16, 129, 127}, {4, 16, 127, 128}},
            {{4, 16, 128, 129}, {4, 16, 129, 127}},
            {{1, 16, 128, 129}, {4, 2, 129, 127}}};
}

inline std::vector<MatmulTestCase> getMatmulLarge5DTestCases()
{
    return {{{8, 4, 16, 512, 1}, {8, 4, 16, 1, 256}},
            {{8, 4, 16, 129, 127}, {8, 4, 16, 127, 128}},
            {{8, 4, 16, 128, 129}, {8, 4, 16, 129, 127}},
            {{6, 1, 16, 128, 129}, {3, 4, 2, 129, 127}}};
}

} // namespace gpu_matmul_ref_test
