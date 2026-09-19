// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>

#include <atomic>
#include <chrono>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

using namespace hipdnn_test_sdk::detail;

class TestCpuFpReferenceUtilities : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Setup code if needed
    }

    void TearDown() override
    {
        // Cleanup code if needed
    }
};

TEST_F(TestCpuFpReferenceUtilities, ParallelTensorFunctorDynamic1DIndexCalculation)
{
    // Test 1D tensor index calculation
    auto functor = makeParallelTensorFunctor(
        [](const std::vector<int64_t>& indices) { (void)indices; }, std::vector<int64_t>{10});

    std::vector<int64_t> indices0;
    functor.fillNdIndices(0, indices0);
    EXPECT_EQ(indices0.size(), 1);
    EXPECT_EQ(indices0[0], 0);

    std::vector<int64_t> indices5;
    functor.fillNdIndices(5, indices5);
    EXPECT_EQ(indices5.size(), 1);
    EXPECT_EQ(indices5[0], 5);

    std::vector<int64_t> indices9;
    functor.fillNdIndices(9, indices9);
    EXPECT_EQ(indices9.size(), 1);
    EXPECT_EQ(indices9[0], 9);
}

TEST_F(TestCpuFpReferenceUtilities, ParallelTensorFunctorDynamic2DIndexCalculation)
{
    auto functor = makeParallelTensorFunctor(
        [](const std::vector<int64_t>& indices) { (void)indices; }, std::vector<int64_t>{3, 4});

    std::vector<int64_t> indices0;
    functor.fillNdIndices(0, indices0); // Should be (0, 0)
    EXPECT_EQ(indices0.size(), 2);
    EXPECT_EQ(indices0[0], 0);
    EXPECT_EQ(indices0[1], 0);

    std::vector<int64_t> indices3;
    functor.fillNdIndices(3, indices3); // Should be (0, 3)
    EXPECT_EQ(indices3[0], 0);
    EXPECT_EQ(indices3[1], 3);

    std::vector<int64_t> indices4;
    functor.fillNdIndices(4, indices4); // Should be (1, 0)
    EXPECT_EQ(indices4[0], 1);
    EXPECT_EQ(indices4[1], 0);

    std::vector<int64_t> indices7;
    functor.fillNdIndices(7, indices7); // Should be (1, 3)
    EXPECT_EQ(indices7[0], 1);
    EXPECT_EQ(indices7[1], 3);

    std::vector<int64_t> indices11;
    functor.fillNdIndices(11, indices11); // Should be (2, 3)
    EXPECT_EQ(indices11[0], 2);
    EXPECT_EQ(indices11[1], 3);
}

TEST_F(TestCpuFpReferenceUtilities, ParallelTensorFunctorDynamic3DIndexCalculation)
{
    auto functor = makeParallelTensorFunctor(
        [](const std::vector<int64_t>& indices) { (void)indices; }, std::vector<int64_t>{2, 3, 4});

    std::vector<int64_t> indices0;
    functor.fillNdIndices(0, indices0); // Should be (0, 0, 0)
    EXPECT_EQ(indices0.size(), 3);
    EXPECT_EQ(indices0[0], 0);
    EXPECT_EQ(indices0[1], 0);
    EXPECT_EQ(indices0[2], 0);

    std::vector<int64_t> indices12;
    functor.fillNdIndices(12, indices12); // Should be (1, 0, 0)
    EXPECT_EQ(indices12[0], 1);
    EXPECT_EQ(indices12[1], 0);
    EXPECT_EQ(indices12[2], 0);

    std::vector<int64_t> indices23;
    functor.fillNdIndices(23, indices23); // Should be (1, 2, 3)
    EXPECT_EQ(indices23[0], 1);
    EXPECT_EQ(indices23[1], 2);
    EXPECT_EQ(indices23[2], 3);
}

TEST_F(TestCpuFpReferenceUtilities, ParallelTensorFunctorDynamic4DIndexCalculation)
{
    auto functor
        = makeParallelTensorFunctor([](const std::vector<int64_t>& indices) { (void)indices; },
                                    std::vector<int64_t>{2, 2, 2, 2});

    std::vector<int64_t> indices0;
    functor.fillNdIndices(0, indices0); // Should be (0, 0, 0, 0)
    EXPECT_EQ(indices0.size(), 4);
    EXPECT_EQ(indices0[0], 0);
    EXPECT_EQ(indices0[1], 0);
    EXPECT_EQ(indices0[2], 0);
    EXPECT_EQ(indices0[3], 0);

    std::vector<int64_t> indices8;
    functor.fillNdIndices(8, indices8); // Should be (1, 0, 0, 0)
    EXPECT_EQ(indices8[0], 1);
    EXPECT_EQ(indices8[1], 0);
    EXPECT_EQ(indices8[2], 0);
    EXPECT_EQ(indices8[3], 0);

    std::vector<int64_t> indices15;
    functor.fillNdIndices(15, indices15); // Should be (1, 1, 1, 1)
    EXPECT_EQ(indices15[0], 1);
    EXPECT_EQ(indices15[1], 1);
    EXPECT_EQ(indices15[2], 1);
    EXPECT_EQ(indices15[3], 1);
}

TEST_F(TestCpuFpReferenceUtilities, ParallelTensorFunctorDynamicSingleThreadExecution)
{
    std::atomic<int> sum{0};

    auto sumFunction
        = [&sum](const std::vector<int64_t>& indices) { sum += static_cast<int>(indices[0]); };

    auto functor = makeParallelTensorFunctor(sumFunction, std::vector<int64_t>{10});
    functor(1); // Single thread

    EXPECT_EQ(sum.load(), 45);
}

TEST_F(TestCpuFpReferenceUtilities, ParallelTensorFunctorDynamicMultiThreadExecution)
{
    std::atomic<int> sum{0};

    auto sumFunction
        = [&sum](const std::vector<int64_t>& indices) { sum += static_cast<int>(indices[0]); };

    auto functor = makeParallelTensorFunctor(sumFunction, std::vector<int64_t>{100});
    functor(4); // Four threads

    // Sum should be 0+1+2+...+99 = 4950
    EXPECT_EQ(sum.load(), 4950);
}

TEST_F(TestCpuFpReferenceUtilities, ParallelTensorFunctorDynamicElementCoverage)
{
    constexpr size_t TENSOR_SIZE = 50;
    std::vector<std::atomic<int>> counts(TENSOR_SIZE);

    for(auto& count : counts)
    {
        count = 0;
    }

    auto countFunction = [&counts](const std::vector<int64_t>& indices) {
        counts[static_cast<size_t>(indices[0])]++;
    };

    auto functor = makeParallelTensorFunctor(countFunction, std::vector<int64_t>{TENSOR_SIZE});
    functor(3);

    for(size_t i = 0; i < TENSOR_SIZE; ++i)
    {
        EXPECT_EQ(counts[i].load(), 1) << "Element " << i << " was not processed exactly once";
    }
}

TEST_F(TestCpuFpReferenceUtilities, ParallelTensorFunctorDynamic2DElementCoverage)
{
    constexpr size_t HEIGHT = 5;
    constexpr size_t WIDTH = 6;
    std::set<std::pair<size_t, size_t>> processedElements;
    std::mutex elementsMutex;

    auto recordFunction
        = [&processedElements, &elementsMutex](const std::vector<int64_t>& indices) {
              const std::lock_guard<std::mutex> lock(elementsMutex);
              processedElements.insert({indices[0], indices[1]});
          };

    auto functor = makeParallelTensorFunctor(recordFunction, std::vector<int64_t>{HEIGHT, WIDTH});
    functor(2); // Two threads

    EXPECT_EQ(processedElements.size(), HEIGHT * WIDTH);

    for(size_t i = 0; i < HEIGHT; ++i)
    {
        for(size_t j = 0; j < WIDTH; ++j)
        {
            EXPECT_TRUE(processedElements.count({i, j}) == 1)
                << "Element (" << i << ", " << j << ") was not processed";
        }
    }
}

TEST_F(TestCpuFpReferenceUtilities, ParallelTensorFunctorDynamicStrideSizesValidation)
{
    // 2D tensor (3x4)
    auto functor2D = makeParallelTensorFunctor(
        [](const std::vector<int64_t>& indices) { (void)indices; }, std::vector<int64_t>{3, 4});
    EXPECT_EQ(functor2D.totalElements, 12);
    EXPECT_EQ(functor2D.strides[0], 4); // stride for first dimension
    EXPECT_EQ(functor2D.strides[1], 1); // stride for second dimension

    // 3D tensor (2x3x4)
    auto functor3D = makeParallelTensorFunctor(
        [](const std::vector<int64_t>& indices) { (void)indices; }, std::vector<int64_t>{2, 3, 4});
    EXPECT_EQ(functor3D.totalElements, 24);
    EXPECT_EQ(functor3D.strides[0], 12); // stride for first dimension
    EXPECT_EQ(functor3D.strides[1], 4); // stride for second dimension
    EXPECT_EQ(functor3D.strides[2], 1); // stride for third dimension

    // 4D tensor (2x2x3x4)
    auto functor4D
        = makeParallelTensorFunctor([](const std::vector<int64_t>& indices) { (void)indices; },
                                    std::vector<int64_t>{2, 2, 3, 4});
    EXPECT_EQ(functor4D.totalElements, 48);
    EXPECT_EQ(functor4D.strides[0], 24); // stride for first dimension
    EXPECT_EQ(functor4D.strides[1], 12); // stride for second dimension
    EXPECT_EQ(functor4D.strides[2], 4); // stride for third dimension
    EXPECT_EQ(functor4D.strides[3], 1); // stride for fourth dimension
}

TEST_F(TestCpuFpReferenceUtilities, ParallelTensorFunctorDynamicEdgeCases)
{
    std::atomic<int> count{0};
    auto countFunction = [&count](const std::vector<int64_t>& indices) {
        (void)indices;
        count++;
    };

    auto functor1x1 = makeParallelTensorFunctor(countFunction, std::vector<int64_t>{1});
    functor1x1(1);
    EXPECT_EQ(count.load(), 1);

    count = 0;
    auto functor1x10 = makeParallelTensorFunctor(
        [&count](const std::vector<int64_t>& indices) {
            (void)indices;
            count++;
        },
        std::vector<int64_t>{1, 10});
    functor1x10(2);
    EXPECT_EQ(count.load(), 10);

    count = 0;
    auto functorSmall = makeParallelTensorFunctor(
        [&count](const std::vector<int64_t>& indices) {
            (void)indices;
            count++;
        },
        std::vector<int64_t>{3});
    functorSmall(10); // 10 threads for 3 elements
    EXPECT_EQ(count.load(), 3);
}

TEST_F(TestCpuFpReferenceUtilities, ParallelTensorFunctorDynamicZeroThreadsFallsBackToSingleThread)
{
    std::atomic<int> count{0};
    auto countFunction = [&count](const std::vector<int64_t>& indices) {
        (void)indices;
        count++;
    };

    auto functor = makeParallelTensorFunctor(countFunction, std::vector<int64_t>{5});
    functor(0);

    EXPECT_EQ(count.load(), 5);
}

TEST_F(TestCpuFpReferenceUtilities, ParallelTensorFunctorDynamicLargerTensorPerformance)
{
    constexpr size_t TENSOR_SIZE = 10000;
    std::atomic<size_t> sum{0};

    auto sumFunction
        = [&sum](const std::vector<int64_t>& indices) { sum += static_cast<size_t>(indices[0]); };

    auto functor = makeParallelTensorFunctor(sumFunction, std::vector<int64_t>{TENSOR_SIZE});

    auto start = std::chrono::high_resolution_clock::now();
    functor(std::thread::hardware_concurrency());
    auto end = std::chrono::high_resolution_clock::now();

    // Verify correctness: sum of 0 to 9999 = 49995000
    EXPECT_EQ(sum.load(), 49995000);

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    if(duration.count() >= 1000)
    {
        FAIL() << "Parallel execution took too long: " << duration.count() << "ms";
    }
}

// Additional edge case tests for ParallelTensorFunctorDynamic
TEST_F(TestCpuFpReferenceUtilities, ParallelTensorFunctorDynamicEmptyTensor)
{
    std::atomic<int> count{0};
    auto countFunction = [&count](const std::vector<int64_t>& indices) {
        (void)indices;
        count++;
    };

    // Test with empty dimensions vector
    auto functorEmpty = makeParallelTensorFunctor(countFunction, std::vector<int64_t>{});
    functorEmpty(2);
    EXPECT_EQ(count.load(), 0); // No elements to process

    count = 0;
    // Test with zero-sized dimension
    auto functorZero = makeParallelTensorFunctor(countFunction, std::vector<int64_t>{0});
    functorZero(2);
    EXPECT_EQ(count.load(), 0); // No elements to process
}

TEST_F(TestCpuFpReferenceUtilities, ParallelTensorFunctorDynamic5DIndexCalculation)
{
    // Test 5D tensor to ensure we support higher dimensions
    auto functor
        = makeParallelTensorFunctor([](const std::vector<int64_t>& indices) { (void)indices; },
                                    std::vector<int64_t>{2, 2, 2, 2, 2});

    std::vector<int64_t> indices0;
    functor.fillNdIndices(0, indices0); // Should be (0, 0, 0, 0, 0)
    EXPECT_EQ(indices0.size(), 5);
    EXPECT_EQ(indices0[0], 0);
    EXPECT_EQ(indices0[1], 0);
    EXPECT_EQ(indices0[2], 0);
    EXPECT_EQ(indices0[3], 0);
    EXPECT_EQ(indices0[4], 0);

    std::vector<int64_t> indices16;
    functor.fillNdIndices(16, indices16); // Should be (1, 0, 0, 0, 0)
    EXPECT_EQ(indices16[0], 1);
    EXPECT_EQ(indices16[1], 0);
    EXPECT_EQ(indices16[2], 0);
    EXPECT_EQ(indices16[3], 0);
    EXPECT_EQ(indices16[4], 0);

    std::vector<int64_t> indices31;
    functor.fillNdIndices(31, indices31); // Should be (1, 1, 1, 1, 1)
    EXPECT_EQ(indices31[0], 1);
    EXPECT_EQ(indices31[1], 1);
    EXPECT_EQ(indices31[2], 1);
    EXPECT_EQ(indices31[3], 1);
    EXPECT_EQ(indices31[4], 1);
}

TEST_F(TestCpuFpReferenceUtilities, ParallelTensorFunctorDynamicLargeDimensions)
{
    // Test with large dimension sizes
    std::atomic<int> count{0};
    auto countFunction = [&count](const std::vector<int64_t>& indices) {
        (void)indices;
        count++;
    };

    // Test with large single dimension
    auto functorLarge = makeParallelTensorFunctor(countFunction, std::vector<int64_t>{1000});
    functorLarge(4);
    EXPECT_EQ(count.load(), 1000);

    count = 0;
    // Test with multiple large dimensions
    auto functorMultiLarge
        = makeParallelTensorFunctor(countFunction, std::vector<int64_t>{10, 100});
    functorMultiLarge(2);
    EXPECT_EQ(count.load(), 1000);
}

TEST_F(TestCpuFpReferenceUtilities, ParallelTensorFunctorDynamicIrregularShapes)
{
    // Test with irregular tensor shapes (different dimension sizes)
    std::set<std::tuple<size_t, size_t, size_t>> processedElements;
    std::mutex elementsMutex;

    auto recordFunction
        = [&processedElements, &elementsMutex](const std::vector<int64_t>& indices) {
              const std::lock_guard<std::mutex> lock(elementsMutex);
              processedElements.insert({indices[0], indices[1], indices[2]});
          };

    // Irregular 3D tensor: 7x3x5
    auto functor = makeParallelTensorFunctor(recordFunction, std::vector<int64_t>{7, 3, 5});
    functor(3);

    EXPECT_EQ(processedElements.size(), 7 * 3 * 5);

    // Verify all elements were processed
    for(size_t i = 0; i < 7; ++i)
    {
        for(size_t j = 0; j < 3; ++j)
        {
            for(size_t k = 0; k < 5; ++k)
            {
                EXPECT_TRUE(processedElements.count({i, j, k}) == 1)
                    << "Element (" << i << ", " << j << ", " << k << ") was not processed";
            }
        }
    }
}

TEST_F(TestCpuFpReferenceUtilities, ParallelTensorFunctorDynamicSingleElementDimensions)
{
    // Test with dimensions that have size 1 (broadcasting-like scenarios)
    std::atomic<int> count{0};
    auto countFunction = [&count](const std::vector<int64_t>& indices) {
        (void)indices;
        count++;
    };

    // Test 4D tensor with some dimensions of size 1: [1, 5, 1, 3]
    auto functor = makeParallelTensorFunctor(countFunction, std::vector<int64_t>{1, 5, 1, 3});
    functor(2);
    EXPECT_EQ(count.load(), 15); // 1 * 5 * 1 * 3 = 15

    // Verify index calculation for this shape
    std::vector<int64_t> indices0;
    functor.fillNdIndices(0, indices0); // Should be (0, 0, 0, 0)
    EXPECT_EQ(indices0[0], 0);
    EXPECT_EQ(indices0[1], 0);
    EXPECT_EQ(indices0[2], 0);
    EXPECT_EQ(indices0[3], 0);

    std::vector<int64_t> indices7;
    functor.fillNdIndices(7, indices7); // Should be (0, 2, 0, 1)
    EXPECT_EQ(indices7[0], 0);
    EXPECT_EQ(indices7[1], 2);
    EXPECT_EQ(indices7[2], 0);
    EXPECT_EQ(indices7[3], 1);
}

TEST_F(TestCpuFpReferenceUtilities, ParallelTensorFunctorDynamicThreadSafety)
{
    // Test thread safety with concurrent access to shared data
    constexpr size_t TENSOR_SIZE = 1000;
    std::vector<std::atomic<int>> elementCounts(TENSOR_SIZE);

    for(auto& count : elementCounts)
    {
        count = 0;
    }

    auto threadSafeFunction = [&elementCounts](const std::vector<int64_t>& indices) {
        // Each element should be processed exactly once across all threads
        elementCounts[static_cast<size_t>(indices[0])]++;
    };

    auto functor = makeParallelTensorFunctor(threadSafeFunction, std::vector<int64_t>{TENSOR_SIZE});
    functor(8); // Use many threads to stress test

    // Verify each element was processed exactly once
    for(size_t i = 0; i < TENSOR_SIZE; ++i)
    {
        EXPECT_EQ(elementCounts[i].load(), 1) << "Element " << i << " was processed "
                                              << elementCounts[i].load() << " times instead of 1";
    }
}

// ============================================================================
// buildDenseOffsets / flatOffset
// ============================================================================

TEST_F(TestCpuFpReferenceUtilities, BuildDenseOffsetsIsRowMajor)
{
    // The normalization references replaced an iterateAlongDimensions walk with this
    // table, so the order has to match: last dimension varies fastest.
    const std::vector<int64_t> strides{10, 1};
    const auto offsets = buildDenseOffsets({2, 3}, strides.data());

    EXPECT_EQ(offsets, (std::vector<int64_t>{0, 1, 2, 10, 11, 12}));
}

TEST_F(TestCpuFpReferenceUtilities, BuildDenseOffsetsHonoursNonPackedStrides)
{
    // A permuted (channels-last) stride set must yield the addresses the strides
    // describe, not the packed ones the extents imply.
    const std::vector<int64_t> strides{1, 4};
    const auto offsets = buildDenseOffsets({2, 3}, strides.data());

    EXPECT_EQ(offsets, (std::vector<int64_t>{0, 4, 8, 1, 5, 9}));
}

TEST_F(TestCpuFpReferenceUtilities, BuildDenseOffsetsTreatsZeroStrideAsBroadcast)
{
    // A zero stride is how an axis the walk does not address contributes nothing: the
    // walk still visits every position, but they alias onto the same offsets.
    const std::vector<int64_t> strides{0, 1};
    const auto offsets = buildDenseOffsets({3, 2}, strides.data());

    EXPECT_EQ(offsets, (std::vector<int64_t>{0, 1, 0, 1, 0, 1}));
}

TEST_F(TestCpuFpReferenceUtilities, BuildDenseOffsetsOfNoExtentsIsOnePosition)
{
    // The scalar walk is one position at offset 0, not an empty table - a reference
    // looping over an empty table would silently skip its only element.
    const std::vector<int64_t> strides{};
    const auto offsets = buildDenseOffsets({}, strides.data());

    EXPECT_EQ(offsets, (std::vector<int64_t>{0}));
}

TEST_F(TestCpuFpReferenceUtilities, FlatOffsetReducesOnlyTheLeadingIndices)
{
    // Each normalization pass holds one index space fixed and walks the other, so it
    // reduces a prefix of the index vector, never all of it.
    const std::vector<int64_t> indices{2, 3, 4};
    const std::vector<int64_t> strides{100, 10, 1};

    EXPECT_EQ(flatOffset(indices.data(), strides.data(), 0), 0);
    EXPECT_EQ(flatOffset(indices.data(), strides.data(), 1), 200);
    EXPECT_EQ(flatOffset(indices.data(), strides.data(), 2), 230);
    EXPECT_EQ(flatOffset(indices.data(), strides.data(), 3), 234);
}

// ============================================================================
// ConvolutionWindow
// ============================================================================

TEST_F(TestCpuFpReferenceUtilities, ConvolutionWindowEmitsTapsInRowMajorWindowOrder)
{
    // Convolution accumulates in tap order, so this ordering is what keeps results
    // bit-identical to the iterateAlongDimensions formulation it replaced.
    const std::vector<int64_t> extents{2, 3};
    const std::vector<int64_t> windowStrides{3, 1};
    const std::vector<int64_t> sourceStrides{100, 10};

    ConvolutionWindow window;
    window.build(extents.size(),
                 extents.data(),
                 windowStrides.data(),
                 sourceStrides.data(),
                 [](size_t, int64_t index) { return index; });

    const std::vector<int64_t> expectedWindow{0, 1, 2, 3, 4, 5};
    const std::vector<int64_t> expectedSource{0, 10, 20, 100, 110, 120};

    ASSERT_EQ(window.taps().size(), expectedWindow.size());
    for(size_t i = 0; i < expectedWindow.size(); ++i)
    {
        EXPECT_EQ(window.taps()[i].windowOffset, expectedWindow[i]) << "tap " << i;
        EXPECT_EQ(window.taps()[i].sourceOffset, expectedSource[i]) << "tap " << i;
    }
}

TEST_F(TestCpuFpReferenceUtilities, ConvolutionWindowDropsTapsWithNoSourceElement)
{
    // A tap in the padding region has no source element. It must disappear while the
    // surviving taps keep the window offsets they would have had.
    const std::vector<int64_t> extents{3};
    const std::vector<int64_t> strides{1};

    ConvolutionWindow window;
    window.build(
        extents.size(), extents.data(), strides.data(), strides.data(), [](size_t, int64_t index) {
            return index == 0 ? -1 : index - 1;
        });

    ASSERT_EQ(window.taps().size(), 2u);
    EXPECT_EQ(window.taps()[0].windowOffset, 1);
    EXPECT_EQ(window.taps()[0].sourceOffset, 0);
    EXPECT_EQ(window.taps()[1].windowOffset, 2);
    EXPECT_EQ(window.taps()[1].sourceOffset, 1);
}

TEST_F(TestCpuFpReferenceUtilities, ConvolutionWindowFactorsValidityPerDimension)
{
    // The whole optimization rests on validity factoring per dimension: dropping one
    // index of the outer dimension must remove that entire slice of the product, not a
    // single tap.
    const std::vector<int64_t> extents{2, 2};
    const std::vector<int64_t> windowStrides{2, 1};
    const std::vector<int64_t> sourceStrides{10, 1};

    ConvolutionWindow window;
    window.build(extents.size(),
                 extents.data(),
                 windowStrides.data(),
                 sourceStrides.data(),
                 [](size_t dim, int64_t index) { return (dim == 0 && index == 1) ? -1 : index; });

    ASSERT_EQ(window.taps().size(), 2u);
    EXPECT_EQ(window.taps()[0].windowOffset, 0);
    EXPECT_EQ(window.taps()[1].windowOffset, 1);
}

TEST_F(TestCpuFpReferenceUtilities, ConvolutionWindowIsEmptyWhenNoTapHasASource)
{
    // A fully padded window must accumulate nothing rather than address anything.
    const std::vector<int64_t> extents{2, 2};
    const std::vector<int64_t> strides{2, 1};

    ConvolutionWindow window;
    window.build(
        extents.size(), extents.data(), strides.data(), strides.data(), [](size_t, int64_t) {
            return -1;
        });

    EXPECT_TRUE(window.taps().empty());
}

TEST_F(TestCpuFpReferenceUtilities, ConvolutionWindowRebuildReplacesPreviousTaps)
{
    // One window is reused for every output position, so a rebuild must replace the
    // previous position's taps rather than append to or leak them.
    const std::vector<int64_t> extents{4};
    const std::vector<int64_t> strides{1};

    ConvolutionWindow window;
    window.build(
        extents.size(), extents.data(), strides.data(), strides.data(), [](size_t, int64_t index) {
            return index;
        });
    ASSERT_EQ(window.taps().size(), 4u);

    window.build(
        extents.size(), extents.data(), strides.data(), strides.data(), [](size_t, int64_t index) {
            return index < 2 ? index : -1;
        });

    ASSERT_EQ(window.taps().size(), 2u);
    EXPECT_EQ(window.taps()[0].windowOffset, 0);
    EXPECT_EQ(window.taps()[1].windowOffset, 1);
}

// ============================================================================
// ParallelTensorFunctorWithScratch
// ============================================================================

namespace
{

// Counts its own construction so a test can tell per-thread scratch from per-work-item
// scratch, which is the entire difference between this functor and the plain one.
struct CountingScratch
{
    static inline std::atomic<int> constructions{0};

    CountingScratch()
    {
        constructions++;
    }

    int itemsSeen{0};
};

} // namespace

TEST_F(TestCpuFpReferenceUtilities, ParallelTensorFunctorWithScratchConstructsOnePerThread)
{
    // The point of the scratch functor: construction is paid once per worker thread, not
    // once per work item. A per-item scratch would report 64 constructions here.
    CountingScratch::constructions = 0;

    constexpr int64_t ELEMENT_COUNT = 64;
    constexpr std::size_t THREAD_COUNT = 4;

    std::atomic<int> visits{0};
    auto functor = makeParallelTensorFunctorWithScratch<CountingScratch>(
        [&visits](CountingScratch& scratch, const std::vector<int64_t>& indices) {
            (void)indices;
            scratch.itemsSeen++;
            visits++;
        },
        std::vector<int64_t>{ELEMENT_COUNT});
    functor(THREAD_COUNT);

    EXPECT_EQ(visits.load(), static_cast<int>(ELEMENT_COUNT));
    EXPECT_EQ(CountingScratch::constructions.load(), static_cast<int>(THREAD_COUNT));
}

TEST_F(TestCpuFpReferenceUtilities, ParallelTensorFunctorWithScratchReusesScratchAcrossWorkItems)
{
    // Single thread, so every work item must land on the same instance. A per-item
    // scratch would leave itemsSeen at 1 and defeat the buffer reuse this exists for.
    CountingScratch::constructions = 0;

    constexpr int64_t ELEMENT_COUNT = 10;
    std::atomic<int> maxItemsSeen{0};

    auto functor = makeParallelTensorFunctorWithScratch<CountingScratch>(
        [&maxItemsSeen](CountingScratch& scratch, const std::vector<int64_t>& indices) {
            (void)indices;
            scratch.itemsSeen++;
            maxItemsSeen = std::max(maxItemsSeen.load(), scratch.itemsSeen);
        },
        std::vector<int64_t>{ELEMENT_COUNT});
    functor(1);

    EXPECT_EQ(CountingScratch::constructions.load(), 1);
    EXPECT_EQ(maxItemsSeen.load(), static_cast<int>(ELEMENT_COUNT));
}

TEST_F(TestCpuFpReferenceUtilities, ParallelTensorFunctorWithScratchIsolatesThreads)
{
    // Threads must not share one scratch; sharing would be a data race in every caller.
    // All workers are alive concurrently, so their addresses are necessarily distinct.
    constexpr int64_t ELEMENT_COUNT = 1000;
    constexpr std::size_t THREAD_COUNT = 4;

    std::mutex mutex;
    std::set<const void*> addresses;

    auto functor = makeParallelTensorFunctorWithScratch<CountingScratch>(
        [&mutex, &addresses](CountingScratch& scratch, const std::vector<int64_t>& indices) {
            (void)indices;
            const std::lock_guard<std::mutex> lock(mutex);
            addresses.insert(&scratch);
        },
        std::vector<int64_t>{ELEMENT_COUNT});
    functor(THREAD_COUNT);

    EXPECT_EQ(addresses.size(), THREAD_COUNT);
}

TEST_F(TestCpuFpReferenceUtilities, ParallelTensorFunctorWithScratchVisitsEveryIndexOnce)
{
    // Same coverage contract as the plain functor: the scratch parameter must not
    // perturb how the index space is decomposed or divided.
    constexpr int64_t HEIGHT = 5;
    constexpr int64_t WIDTH = 7;

    std::vector<std::atomic<int>> visitCounts(static_cast<size_t>(HEIGHT * WIDTH));
    for(auto& count : visitCounts)
    {
        count = 0;
    }

    auto functor = makeParallelTensorFunctorWithScratch<CountingScratch>(
        [&visitCounts](CountingScratch& scratch, const std::vector<int64_t>& indices) {
            (void)scratch;
            visitCounts[static_cast<size_t>((indices[0] * WIDTH) + indices[1])]++;
        },
        std::vector<int64_t>{HEIGHT, WIDTH});
    functor(3);

    for(size_t i = 0; i < visitCounts.size(); ++i)
    {
        EXPECT_EQ(visitCounts[i].load(), 1)
            << "index " << i << " was visited the wrong number of times";
    }
}

TEST_F(TestCpuFpReferenceUtilities, ParallelTensorFunctorWithScratchStopsOnFalse)
{
    // The bool early-exit contract carries over from the plain functor.
    std::atomic<int> visits{0};

    auto functor = makeParallelTensorFunctorWithScratch<CountingScratch>(
        [&visits](CountingScratch& scratch, const std::vector<int64_t>& indices) {
            (void)scratch;
            visits++;
            return indices[0] < 3;
        },
        std::vector<int64_t>{100});
    functor(1);

    // Indices 0, 1 and 2 continue; index 3 returns false and stops the thread.
    EXPECT_EQ(visits.load(), 4);
}

TEST_F(TestCpuFpReferenceUtilities, ParallelTensorFunctorWithScratchDoesNothingForAnEmptyRange)
{
    // A zero-sized dimension must spawn no threads and construct no scratch.
    CountingScratch::constructions = 0;
    std::atomic<int> visits{0};

    auto functor = makeParallelTensorFunctorWithScratch<CountingScratch>(
        [&visits](CountingScratch& scratch, const std::vector<int64_t>& indices) {
            (void)scratch;
            (void)indices;
            visits++;
        },
        std::vector<int64_t>{0});
    functor(4);

    EXPECT_EQ(visits.load(), 0);
    EXPECT_EQ(CountingScratch::constructions.load(), 0);
}

// A bool-returning functor taking a non-const lvalue reference. It binds fine at the call
// site either way, so an early-exit probe that models the argument as an rvalue or as a
// `const&` silently drops the bool and runs the whole range.
namespace
{

struct MutableIndexEarlyExit
{
    std::atomic<int>* visits;

    bool operator()(std::vector<int64_t>& indices) const
    {
        (*visits)++;
        return indices[0] < 3;
    }
};

} // namespace

TEST_F(TestCpuFpReferenceUtilities, ParallelTensorFunctorDynamicStopsOnFalseForMutableIndices)
{
    std::atomic<int> visits{0};

    auto functor
        = makeParallelTensorFunctor(MutableIndexEarlyExit{&visits}, std::vector<int64_t>{100});
    functor(1);

    // Indices 0, 1 and 2 continue; index 3 returns false and stops the thread.
    EXPECT_EQ(visits.load(), 4);
}

TEST_F(TestCpuFpReferenceUtilities, ParallelTensorFunctorWithScratchStopsOnFalseForMutableIndices)
{
    std::atomic<int> visits{0};

    auto functor = makeParallelTensorFunctorWithScratch<CountingScratch>(
        [&visits](CountingScratch& scratch, std::vector<int64_t>& indices) {
            (void)scratch;
            visits++;
            return indices[0] < 3;
        },
        std::vector<int64_t>{100});
    functor(1);

    EXPECT_EQ(visits.load(), 4);
}
