// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceUtilities.hpp>

#include <atomic>
#include <chrono>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

using namespace hipdnn_sdk::test_utilities;

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

    auto indices0 = functor.getNdIndices(0);
    EXPECT_EQ(indices0.size(), 1);
    EXPECT_EQ(indices0[0], 0);

    auto indices5 = functor.getNdIndices(5);
    EXPECT_EQ(indices5.size(), 1);
    EXPECT_EQ(indices5[0], 5);

    auto indices9 = functor.getNdIndices(9);
    EXPECT_EQ(indices9.size(), 1);
    EXPECT_EQ(indices9[0], 9);
}

TEST_F(TestCpuFpReferenceUtilities, ParallelTensorFunctorDynamic2DIndexCalculation)
{
    auto functor = makeParallelTensorFunctor(
        [](const std::vector<int64_t>& indices) { (void)indices; }, std::vector<int64_t>{3, 4});

    auto indices0 = functor.getNdIndices(0); // Should be (0, 0)
    EXPECT_EQ(indices0.size(), 2);
    EXPECT_EQ(indices0[0], 0);
    EXPECT_EQ(indices0[1], 0);

    auto indices3 = functor.getNdIndices(3); // Should be (0, 3)
    EXPECT_EQ(indices3[0], 0);
    EXPECT_EQ(indices3[1], 3);

    auto indices4 = functor.getNdIndices(4); // Should be (1, 0)
    EXPECT_EQ(indices4[0], 1);
    EXPECT_EQ(indices4[1], 0);

    auto indices7 = functor.getNdIndices(7); // Should be (1, 3)
    EXPECT_EQ(indices7[0], 1);
    EXPECT_EQ(indices7[1], 3);

    auto indices11 = functor.getNdIndices(11); // Should be (2, 3)
    EXPECT_EQ(indices11[0], 2);
    EXPECT_EQ(indices11[1], 3);
}

TEST_F(TestCpuFpReferenceUtilities, ParallelTensorFunctorDynamic3DIndexCalculation)
{
    auto functor = makeParallelTensorFunctor(
        [](const std::vector<int64_t>& indices) { (void)indices; }, std::vector<int64_t>{2, 3, 4});

    auto indices0 = functor.getNdIndices(0); // Should be (0, 0, 0)
    EXPECT_EQ(indices0.size(), 3);
    EXPECT_EQ(indices0[0], 0);
    EXPECT_EQ(indices0[1], 0);
    EXPECT_EQ(indices0[2], 0);

    auto indices12 = functor.getNdIndices(12); // Should be (1, 0, 0)
    EXPECT_EQ(indices12[0], 1);
    EXPECT_EQ(indices12[1], 0);
    EXPECT_EQ(indices12[2], 0);

    auto indices23 = functor.getNdIndices(23); // Should be (1, 2, 3)
    EXPECT_EQ(indices23[0], 1);
    EXPECT_EQ(indices23[1], 2);
    EXPECT_EQ(indices23[2], 3);
}

TEST_F(TestCpuFpReferenceUtilities, ParallelTensorFunctorDynamic4DIndexCalculation)
{
    auto functor
        = makeParallelTensorFunctor([](const std::vector<int64_t>& indices) { (void)indices; },
                                    std::vector<int64_t>{2, 2, 2, 2});

    auto indices0 = functor.getNdIndices(0); // Should be (0, 0, 0, 0)
    EXPECT_EQ(indices0.size(), 4);
    EXPECT_EQ(indices0[0], 0);
    EXPECT_EQ(indices0[1], 0);
    EXPECT_EQ(indices0[2], 0);
    EXPECT_EQ(indices0[3], 0);

    auto indices8 = functor.getNdIndices(8); // Should be (1, 0, 0, 0)
    EXPECT_EQ(indices8[0], 1);
    EXPECT_EQ(indices8[1], 0);
    EXPECT_EQ(indices8[2], 0);
    EXPECT_EQ(indices8[3], 0);

    auto indices15 = functor.getNdIndices(15); // Should be (1, 1, 1, 1)
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
              std::lock_guard<std::mutex> lock(elementsMutex);
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

TEST_F(TestCpuFpReferenceUtilities, ParallelTensorFunctorDynamicZeroThreads)
{
    std::atomic<int> count{0};
    auto countFunction = [&count](const std::vector<int64_t>& indices) {
        (void)indices;
        count++;
    };

    auto functor = makeParallelTensorFunctor(countFunction, std::vector<int64_t>{5});
    functor(0); // Zero threads

    // With zero threads, no processing should occur
    EXPECT_EQ(count.load(), 0);
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

    auto indices0 = functor.getNdIndices(0); // Should be (0, 0, 0, 0, 0)
    EXPECT_EQ(indices0.size(), 5);
    EXPECT_EQ(indices0[0], 0);
    EXPECT_EQ(indices0[1], 0);
    EXPECT_EQ(indices0[2], 0);
    EXPECT_EQ(indices0[3], 0);
    EXPECT_EQ(indices0[4], 0);

    auto indices16 = functor.getNdIndices(16); // Should be (1, 0, 0, 0, 0)
    EXPECT_EQ(indices16[0], 1);
    EXPECT_EQ(indices16[1], 0);
    EXPECT_EQ(indices16[2], 0);
    EXPECT_EQ(indices16[3], 0);
    EXPECT_EQ(indices16[4], 0);

    auto indices31 = functor.getNdIndices(31); // Should be (1, 1, 1, 1, 1)
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
              std::lock_guard<std::mutex> lock(elementsMutex);
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
    auto indices0 = functor.getNdIndices(0); // Should be (0, 0, 0, 0)
    EXPECT_EQ(indices0[0], 0);
    EXPECT_EQ(indices0[1], 0);
    EXPECT_EQ(indices0[2], 0);
    EXPECT_EQ(indices0[3], 0);

    auto indices7 = functor.getNdIndices(7); // Should be (0, 2, 0, 1)
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
