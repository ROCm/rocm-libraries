// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <initializer_list>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/TensorDescriptor.hpp>

#include <stdexcept>
#include <vector>

#include "BatchPointerLayout.hpp"
#include "DataInitializationTestUtils.hpp"

using namespace TensileLite;
using namespace TensileLite::Client;

namespace
{
    TensorDescriptor makeDescriptor(std::initializer_list<size_t> sizes,
                                    std::initializer_list<size_t> strides)
    {
        return TensorDescriptor("batch-pointer-layout",
                                rocisa::DataType::Float,
                                sizes,
                                strides);
    }
} // namespace

TEST(BatchPointerLayout, EmptyBatchIndicesProduceSingleZeroOffset)
{
    auto const tensor = makeDescriptor({4}, {13});

    auto const layout = makeBatchPointerLayout(tensor, std::vector<size_t>{});

    EXPECT_EQ(layout.offsets, std::vector<size_t>({0}));
}

TEST(BatchPointerLayout, SingleBatchDimensionUsesTensorStride)
{
    auto const tensor = makeDescriptor({4}, {13});

    auto const layout = makeBatchPointerLayout(tensor, std::vector<size_t>{0});

    EXPECT_EQ(layout.offsets, std::vector<size_t>({0, 13, 26, 39}));
}

TEST(BatchPointerLayout, MultipleBatchDimensionsPreserveCoordNumberedOrder)
{
    auto const tensor = makeDescriptor({2, 3, 4, 5}, {1, 10, 100, 1000});

    auto const layout = makeBatchPointerLayout(tensor, std::vector<size_t>{2, 3});

    EXPECT_EQ(layout.offsets,
              std::vector<size_t>({0,    100,  200,  300,  1000, 1100, 1200, 1300, 2000,
                                   2100, 2200, 2300, 3000, 3100, 3200, 3300, 4000, 4100,
                                   4200, 4300}));
}

TEST(BatchPointerLayout, TensorBatchIndicesMapABCD)
{
    ContractionProblemGemm::BatchIndices batchIndices{{2, 3, 4, 5}, {6, 7, 8, 9}};

    EXPECT_EQ(batchPointerTensorBatchIndices(batchIndices, ContractionProblemGemm::TENSOR::A),
              std::vector<size_t>({2, 6}));
    EXPECT_EQ(batchPointerTensorBatchIndices(batchIndices, ContractionProblemGemm::TENSOR::B),
              std::vector<size_t>({3, 7}));
    EXPECT_EQ(batchPointerTensorBatchIndices(batchIndices, ContractionProblemGemm::TENSOR::C),
              std::vector<size_t>({4, 8}));
    EXPECT_EQ(batchPointerTensorBatchIndices(batchIndices, ContractionProblemGemm::TENSOR::D),
              std::vector<size_t>({5, 9}));
}

TEST(BatchPointerLayout, InvalidNonABCDTensorThrows)
{
    ContractionProblemGemm::BatchIndices batchIndices{{2, 3, 4, 5}};

    EXPECT_THROW(batchPointerTensorBatchIndices(batchIndices, ContractionProblemGemm::TENSOR::BIAS),
                 std::invalid_argument);
    EXPECT_THROW(batchPointerTensorBatchIndices(ContractionProblemGemm::BatchIndices{},
                                               ContractionProblemGemm::TENSOR::BIAS),
                 std::invalid_argument);
}

TEST(BatchPointerLayout, DestinationLayoutUsesSourceBatchIndicesButDestinationStrides)
{
    ContractionProblemGemm::BatchIndices batchIndices{{2, 3, 4, 5}, {6, 7, 8, 9}};

    auto const sourceBatchIdx
        = batchPointerTensorBatchIndices(batchIndices, ContractionProblemGemm::TENSOR::A);
    EXPECT_EQ(sourceBatchIdx, std::vector<size_t>({2, 6}));

    auto const destination = makeDescriptor({1, 1, 2, 1, 1, 1, 3},
                                            {TensorDescriptor::UseDefaultStride,
                                             TensorDescriptor::UseDefaultStride,
                                             1000,
                                             TensorDescriptor::UseDefaultStride,
                                             TensorDescriptor::UseDefaultStride,
                                             TensorDescriptor::UseDefaultStride,
                                             7});

    auto const layout = makeBatchPointerLayout(destination, sourceBatchIdx);

    EXPECT_EQ(layout.offsets, std::vector<size_t>({0, 1000, 7, 1007, 14, 1014}));
}

TEST(BatchPointerLayout, ProblemStrideChangesProduceDifferentLayouts)
{
    constexpr size_t BATCH = 4;

    auto const smallProblem = TensileLite::testing::makeBatchedProblem(32, 32, 32, BATCH);
    auto const largeProblem = TensileLite::testing::makeBatchedProblem(64, 64, 64, BATCH);

    auto const smallBatchIdx
        = batchPointerTensorBatchIndices(smallProblem.batchIndices(),
                                         ContractionProblemGemm::TENSOR::A);
    auto const largeBatchIdx
        = batchPointerTensorBatchIndices(largeProblem.batchIndices(),
                                         ContractionProblemGemm::TENSOR::A);

    auto const smallLayout = makeBatchPointerLayout(smallProblem.a(), smallBatchIdx);
    auto const largeLayout = makeBatchPointerLayout(largeProblem.a(), largeBatchIdx);

    ASSERT_GE(smallLayout.offsets.size(), 2u);
    ASSERT_GE(largeLayout.offsets.size(), 2u);

    EXPECT_EQ(smallLayout.offsets[1] - smallLayout.offsets[0], size_t(32 * 32));
    EXPECT_EQ(largeLayout.offsets[1] - largeLayout.offsets[0], size_t(64 * 64));
}

TEST(BatchPointerLayout, InvalidBatchDimensionThrows)
{
    auto const tensor = makeDescriptor({2, 3, 4, 5}, {1, 10, 100, 1000});

    EXPECT_THROW(makeBatchPointerLayout(tensor, std::vector<size_t>{4}), std::out_of_range);
}
