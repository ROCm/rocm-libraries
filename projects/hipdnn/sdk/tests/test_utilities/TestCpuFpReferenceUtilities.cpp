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

TEST_F(TestCpuFpReferenceUtilities, ParallelTensorFunctor1DIndexCalculation)
{
    // Test 1D tensor index calculation
    auto functor = makeParallelTensorFunctor([](auto i) { (void)i; }, 10);

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

TEST_F(TestCpuFpReferenceUtilities, ParallelTensorFunctor2DIndexCalculation)
{
    auto functor = makeParallelTensorFunctor(
        [](auto i, auto j) {
            (void)i;
            (void)j;
        },
        3,
        4);

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

TEST_F(TestCpuFpReferenceUtilities, ParallelTensorFunctor3DIndexCalculation)
{
    auto functor = makeParallelTensorFunctor(
        [](auto i, auto j, auto k) {
            (void)i;
            (void)j;
            (void)k;
        },
        2,
        3,
        4);

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

TEST_F(TestCpuFpReferenceUtilities, ParallelTensorFunctor4DIndexCalculation)
{
    auto functor = makeParallelTensorFunctor(
        [](auto i, auto j, auto k, auto l) {
            (void)i;
            (void)j;
            (void)k;
            (void)l;
        },
        2,
        2,
        2,
        2);

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

TEST_F(TestCpuFpReferenceUtilities, ParallelTensorFunctorSingleThreadExecution)
{
    std::atomic<int> sum{0};

    auto sumFunction = [&sum](auto i) { sum += static_cast<int>(i); };

    auto functor = makeParallelTensorFunctor(sumFunction, 10);
    functor(1); // Single thread

    EXPECT_EQ(sum.load(), 45);
}

TEST_F(TestCpuFpReferenceUtilities, ParallelTensorFunctorMultiThreadExecution)
{
    std::atomic<int> sum{0};

    auto sumFunction = [&sum](auto i) { sum += static_cast<int>(i); };

    auto functor = makeParallelTensorFunctor(sumFunction, 100);
    functor(4); // Four threads

    // Sum should be 0+1+2+...+99 = 4950
    EXPECT_EQ(sum.load(), 4950);
}

TEST_F(TestCpuFpReferenceUtilities, ParallelTensorFunctorElementCoverage)
{
    constexpr size_t TENSOR_SIZE = 50;
    std::vector<std::atomic<int>> counts(TENSOR_SIZE);

    for(auto& count : counts)
    {
        count = 0;
    }

    auto countFunction = [&counts](auto i) { counts[static_cast<size_t>(i)]++; };

    auto functor = makeParallelTensorFunctor(countFunction, TENSOR_SIZE);
    functor(3);

    for(size_t i = 0; i < TENSOR_SIZE; ++i)
    {
        EXPECT_EQ(counts[i].load(), 1) << "Element " << i << " was not processed exactly once";
    }
}

TEST_F(TestCpuFpReferenceUtilities, ParallelTensorFunctor2DElementCoverage)
{
    constexpr size_t HEIGHT = 5;
    constexpr size_t WIDTH = 6;
    std::set<std::pair<size_t, size_t>> processedElements;
    std::mutex elementsMutex;

    auto recordFunction = [&processedElements, &elementsMutex](auto i, auto j) {
        std::lock_guard<std::mutex> lock(elementsMutex);
        processedElements.insert({static_cast<size_t>(i), static_cast<size_t>(j)});
    };

    auto functor = makeParallelTensorFunctor(recordFunction, HEIGHT, WIDTH);
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

TEST_F(TestCpuFpReferenceUtilities, ParallelTensorFunctorStrideSizesValidation)
{

    // 2D tensor (3x4)
    auto functor2D = makeParallelTensorFunctor(
        [](auto i, auto j) {
            (void)i;
            (void)j;
        },
        3,
        4);
    EXPECT_EQ(functor2D._totalElements, 12);
    EXPECT_EQ(functor2D._strides[0], 4); // stride for first dimension
    EXPECT_EQ(functor2D._strides[1], 1); // stride for second dimension

    // 3D tensor (2x3x4)
    auto functor3D = makeParallelTensorFunctor(
        [](auto i, auto j, auto k) {
            (void)i;
            (void)j;
            (void)k;
        },
        2,
        3,
        4);
    EXPECT_EQ(functor3D._totalElements, 24);
    EXPECT_EQ(functor3D._strides[0], 12); // stride for first dimension
    EXPECT_EQ(functor3D._strides[1], 4); // stride for second dimension
    EXPECT_EQ(functor3D._strides[2], 1); // stride for third dimension

    // 4D tensor (2x2x3x4)
    auto functor4D = makeParallelTensorFunctor(
        [](auto i, auto j, auto k, auto l) {
            (void)i;
            (void)j;
            (void)k;
            (void)l;
        },
        2,
        2,
        3,
        4);
    EXPECT_EQ(functor4D._totalElements, 48);
    EXPECT_EQ(functor4D._strides[0], 24); // stride for first dimension
    EXPECT_EQ(functor4D._strides[1], 12); // stride for second dimension
    EXPECT_EQ(functor4D._strides[2], 4); // stride for third dimension
    EXPECT_EQ(functor4D._strides[3], 1); // stride for fourth dimension
}

TEST_F(TestCpuFpReferenceUtilities, ParallelTensorFunctorEdgeCases)
{
    std::atomic<int> count{0};
    auto countFunction = [&count](auto i) {
        (void)i;
        count++;
    };

    auto functor1x1 = makeParallelTensorFunctor(countFunction, 1);
    functor1x1(1);
    EXPECT_EQ(count.load(), 1);

    count = 0;
    auto functor1x10 = makeParallelTensorFunctor(
        [&count](auto i, auto j) {
            (void)i;
            (void)j;
            count++;
        },
        1,
        10);
    functor1x10(2);
    EXPECT_EQ(count.load(), 10);

    count = 0;
    auto functorSmall = makeParallelTensorFunctor(
        [&count](auto i) {
            (void)i;
            count++;
        },
        3);
    functorSmall(10); // 10 threads for 3 elements
    EXPECT_EQ(count.load(), 3);
}

TEST_F(TestCpuFpReferenceUtilities, ParallelTensorFunctorZeroThreads)
{
    std::atomic<int> count{0};
    auto countFunction = [&count](auto i) {
        (void)i;
        count++;
    };

    auto functor = makeParallelTensorFunctor(countFunction, 5);
    functor(0); // Zero threads

    // With zero threads, no processing should occur
    EXPECT_EQ(count.load(), 0);
}

TEST_F(TestCpuFpReferenceUtilities, ParallelTensorFunctorLargerTensorPerformance)
{
    constexpr size_t TENSOR_SIZE = 10000;
    std::atomic<size_t> sum{0};

    auto sumFunction = [&sum](auto i) { sum += static_cast<size_t>(i); };

    auto functor = makeParallelTensorFunctor(sumFunction, TENSOR_SIZE);

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
