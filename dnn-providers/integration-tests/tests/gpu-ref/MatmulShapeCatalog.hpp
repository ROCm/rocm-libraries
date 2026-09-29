// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "MatmulTestCase.hpp"

namespace gpu_matmul_ref_test
{

using hipdnn_data_sdk::utilities::generateStrides;
using hipdnn_data_sdk::utilities::TensorLayout;

inline std::vector<MatmulTestCase> getMatmulSmall2DTestCases()
{
    return {{{2, 2}, {2, 2}},
            {{3, 2}, {2, 5}},
            {{3, 2}, {2, 5}, {3, 5}, {1, 3}, {1, 2}, {1, 3}},
            {{2, 5}, {5, 3}},
            {{2, 5}, {5, 3}, {2, 3}, {5, 1}, {3, 1}, {40, 10}}};
}

inline std::vector<MatmulTestCase> getMatmulSmall3DTestCases()
{
    return {{{2, 2, 2}, {2, 2, 2}},
            {{7, 3, 2}, {7, 2, 5}},
            {{7, 2, 5}, {7, 5, 3}},
            {{2, 2, 2}, {4, 2, 3}},
            {{6, 2, 2}, {3, 2, 3}},
            {{6, 2, 2}, {3, 2, 3}, {6, 2, 3}, {4, 2, 1}, {15, 5, 1}, {1, 40, 10}}};
}

inline std::vector<MatmulTestCase> getMatmulSmall4DTestCases()
{
    return {{{2, 2, 2, 2}, {2, 2, 2, 2}},
            {{2, 3, 3, 2}, {2, 3, 2, 5}},
            {{2, 3, 2, 5}, {2, 3, 5, 3}},
            {{2, 3, 2, 5},
             {2, 3, 5, 3},
             {2, 3, 2, 3},
             generateStrides({2, 3, 2, 5}, TensorLayout::NHWC.strideOrder),
             generateStrides({2, 3, 5, 3}, TensorLayout::NHWC.strideOrder),
             generateStrides({2, 3, 2, 3}, TensorLayout::NHWC.strideOrder)},
            {{1, 6, 2, 2}, {3, 3, 2, 3}},
            {{1, 6, 2, 2},
             {3, 3, 2, 3},
             {3, 6, 2, 3},
             generateStrides({1, 6, 2, 2}, TensorLayout::NHWC.strideOrder),
             generateStrides({3, 3, 2, 3}),
             generateStrides({3, 6, 2, 3}, TensorLayout::NHWC.strideOrder)}};
}

inline std::vector<MatmulTestCase> getMatmulSmall5DTestCases()
{
    return {{{2, 2, 2, 2, 2}, {2, 2, 2, 2, 2}},
            {{2, 2, 2, 3, 2}, {2, 2, 2, 2, 3}},
            {{2, 2, 2, 2, 3}, {2, 2, 2, 3, 2}},
            {{1, 6, 3, 2, 3}, {3, 2, 6, 3, 2}},
            {{1, 2, 3, 3, 2},
             {7, 6, 6, 2, 3},
             {7, 6, 6, 3, 3},
             {256, 1, 16, 3, 64},
             {1, 42, 512, 21, 7},
             {1, 7, 42, 252, 756}}};
}

inline std::vector<MatmulTestCase> getMatmulSmallTestCases()
{
    auto testCases = getMatmulSmall2DTestCases();
    auto testCases3d = getMatmulSmall3DTestCases();
    auto testCases4d = getMatmulSmall4DTestCases();
    auto testCases5d = getMatmulSmall5DTestCases();
    testCases.insert(testCases.end(), testCases3d.begin(), testCases3d.end());
    testCases.insert(testCases.end(), testCases4d.begin(), testCases4d.end());
    testCases.insert(testCases.end(), testCases5d.begin(), testCases5d.end());
    return testCases;
}

inline std::vector<MatmulTestCase> getMatmulMedium2DTestCases()
{
    return {{{64, 64}, {64, 64}},
            {{15, 65}, {65, 17}},
            {{15, 65}, {65, 17}, {15, 17}, {1, 65}, {1, 65}, {1, 17}},
            {{63, 17}, {17, 65}},
            {{63, 17}, {17, 65}, {63, 65}, {17, 1}, {65, 1}, {500, 5}}};
}

inline std::vector<MatmulTestCase> getMatmulMedium3DTestCases()
{
    return {{{2, 64, 64}, {2, 64, 64}},
            {{2, 15, 65}, {2, 65, 17}},
            {{2, 63, 17}, {2, 17, 65}},
            {{2, 16, 16}, {4, 16, 16}},
            {{6, 16, 16}, {3, 16, 16}},
            {{6, 16, 16}, {3, 16, 16}, {6, 16, 16}, {256, 16, 1}, {400, 20, 1}, {1, 200, 10}}};
}

inline std::vector<MatmulTestCase> getMatmulMedium4DTestCases()
{
    return {{{2, 2, 64, 64}, {2, 2, 64, 64}},
            {{2, 2, 15, 65}, {2, 2, 65, 17}},
            {{2, 2, 63, 17}, {2, 2, 17, 65}},
            {{2, 2, 63, 17},
             {2, 2, 17, 65},
             {2, 2, 63, 65},
             generateStrides({2, 2, 63, 17}, TensorLayout::NHWC.strideOrder),
             generateStrides({2, 2, 17, 65}, TensorLayout::NHWC.strideOrder),
             generateStrides({2, 2, 63, 65}, TensorLayout::NHWC.strideOrder)},
            {{1, 6, 16, 16}, {3, 3, 16, 16}},
            {{1, 6, 16, 16},
             {3, 3, 16, 16},
             {3, 6, 16, 16},
             generateStrides({1, 6, 16, 16}, TensorLayout::NHWC.strideOrder),
             generateStrides({3, 3, 16, 16}),
             generateStrides({3, 6, 16, 16}, TensorLayout::NHWC.strideOrder)}};
}

inline std::vector<MatmulTestCase> getMatmulMedium5DTestCases()
{
    return {{{2, 2, 2, 64, 64}, {2, 2, 2, 64, 64}},
            {{2, 2, 2, 15, 65}, {2, 2, 2, 65, 17}},
            {{2, 2, 2, 63, 17}, {2, 2, 2, 17, 65}},
            {{1, 6, 2, 16, 16}, {3, 2, 6, 16, 16}},
            {{1, 2, 3, 63, 17},
             {7, 6, 6, 17, 65},
             {7, 6, 6, 63, 65},
             {24576, 1, 200, 3, 1024},
             {1, 7735, 65536, 455, 7},
             {1, 7, 42, 252, 15876}}};
}

inline std::vector<MatmulTestCase> getMatmulMediumTestCases()
{
    auto testCases = getMatmulMedium2DTestCases();
    auto testCases3d = getMatmulMedium3DTestCases();
    auto testCases4d = getMatmulMedium4DTestCases();
    auto testCases5d = getMatmulMedium5DTestCases();
    testCases.insert(testCases.end(), testCases3d.begin(), testCases3d.end());
    testCases.insert(testCases.end(), testCases4d.begin(), testCases4d.end());
    testCases.insert(testCases.end(), testCases5d.begin(), testCases5d.end());
    return testCases;
}

inline std::vector<MatmulTestCase> getMatmulLarge2DTestCases()
{
    return {{{512, 1}, {1, 256}},
            {{129, 127}, {127, 128}},
            {{129, 127}, {127, 128}, {129, 128}, {1, 129}, {1, 127}, {1, 129}},
            {{128, 129}, {129, 127}},
            {{128, 129}, {129, 127}, {128, 127}, {129, 1}, {127, 1}, {65536, 256}}};
}

inline std::vector<MatmulTestCase> getMatmulLarge3DTestCases()
{
    return {{{16, 512, 1}, {16, 1, 256}},
            {{16, 129, 127}, {16, 127, 128}},
            {{16, 128, 129}, {16, 129, 127}},
            {{16, 128, 129}, {2, 129, 127}},
            {{1, 128, 129}, {4, 129, 127}},
            {{16, 128, 129},
             {2, 129, 127},
             {16, 128, 127},
             {16512, 129, 1},
             {768, 256, 1},
             {1, 4000, 20}}};
}

inline std::vector<MatmulTestCase> getMatmulLarge4DTestCases()
{
    return {{{4, 16, 512, 1}, {4, 16, 1, 256}},
            {{4, 16, 129, 127}, {4, 16, 127, 128}},
            {{4, 16, 128, 129}, {4, 16, 129, 127}},
            {{4, 16, 128, 129},
             {4, 16, 129, 127},
             {4, 16, 128, 127},
             generateStrides({4, 16, 128, 129}, TensorLayout::NHWC.strideOrder),
             generateStrides({4, 16, 129, 127}, TensorLayout::NHWC.strideOrder),
             generateStrides({4, 16, 128, 127}, TensorLayout::NHWC.strideOrder)},
            {{1, 16, 128, 129}, {4, 2, 129, 127}},
            {{1, 16, 128, 129},
             {4, 2, 129, 127},
             {4, 16, 128, 127},
             generateStrides({1, 16, 128, 129}, TensorLayout::NHWC.strideOrder),
             generateStrides({4, 2, 129, 127}),
             generateStrides({4, 16, 128, 127}, TensorLayout::NHWC.strideOrder)}};
}

inline std::vector<MatmulTestCase> getMatmulLarge5DTestCases()
{
    return {{{8, 4, 16, 512, 1}, {8, 4, 16, 1, 256}},
            {{8, 4, 16, 129, 127}, {8, 4, 16, 127, 128}},
            {{8, 4, 16, 128, 129}, {8, 4, 16, 129, 127}},
            {{6, 1, 16, 128, 129}, {3, 4, 2, 129, 127}},
            {{1, 2, 3, 128, 129},
             {7, 6, 6, 129, 127},
             {7, 6, 6, 128, 127},
             {524288, 1, 512, 3, 2048},
             {1, 114681, 1048576, 889, 7},
             {1, 7, 42, 252, 32256}}};
}

inline std::vector<MatmulTestCase> getMatmulLargeTestCases()
{
    auto testCases = getMatmulLarge2DTestCases();
    auto testCases3d = getMatmulLarge3DTestCases();
    auto testCases4d = getMatmulLarge4DTestCases();
    auto testCases5d = getMatmulLarge5DTestCases();
    testCases.insert(testCases.end(), testCases3d.begin(), testCases3d.end());
    testCases.insert(testCases.end(), testCases4d.begin(), testCases4d.end());
    testCases.insert(testCases.end(), testCases5d.begin(), testCases5d.end());
    return testCases;
}

} // namespace gpu_matmul_ref_test
