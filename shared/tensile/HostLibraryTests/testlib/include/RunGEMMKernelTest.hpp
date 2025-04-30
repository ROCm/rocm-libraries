/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef RUN_GEMM_KERNEL_TEST_HPP
#define RUN_GEMM_KERNEL_TEST_HPP

#include "GEMMKernelTest.hpp"
#include <gtest/gtest.h>

/* Generates the specific test parameters
 * that will be used in the GEMM gtests.
 * Uses testing components as specified
 * in the GEMMKernelTest interface.
 */
template <typename DeviceBackend>
struct RunGEMMKernelTestParams
{
    // The interface we are creating input params for
    using TestInterface = GEMMKernelTest<DeviceBackend>;

    // Extract test interface components
    using ContractionProblem  = typename TestInterface::ContractionProblem;
    using ContractionSolution = typename TestInterface::ContractionSolution;
    using MemoryPageAlignment = typename TestInterface::MemoryPageAlignment;
    using SolutionAdapter     = typename TestInterface::SolutionAdapter;
    using SolutionLibrary     = typename TestInterface::SolutionLibrary;

    // Test input params
    using ProblemParams  = typename TestInterface::ProblemParams;
    using SolutionParams = typename TestInterface::SolutionParams;

    // Runtime input generation
    static std::vector<std::shared_ptr<TestInterface>> TypedTests();
    static std::vector<ProblemParams>                  TestProblems();
    static std::vector<ProblemParams>                  TestProblemsExtended();
    static std::vector<SolutionParams>                 TestLibraries();
    static std::vector<MemoryPageAlignment>            TestMemoryAlignments();

private:
    static std::vector<SolutionParams> TestLibraries_Impl();
};

/* Testing harness:
*  This ensures each test receives each
*  combination of inputs.
*  Tests for different backends can
*  be defined in each their own backend
*  ecosystems.
*  See hip/RunGEMMKernel_test.cpp
*/
template <typename DeviceBackend>
struct RunGEMMKernelTest
    : public ::testing::TestWithParam<
          std::tuple<std::shared_ptr<GEMMKernelTest<DeviceBackend>>,
                     typename GEMMKernelTest<DeviceBackend>::ProblemParams,
                     typename GEMMKernelTest<DeviceBackend>::SolutionParams,
                     typename GEMMKernelTest<DeviceBackend>::MemoryPageAlignment>>
{
    using Base = ::testing::TestWithParam<
        std::tuple<std::shared_ptr<GEMMKernelTest<DeviceBackend>>,
                   typename GEMMKernelTest<DeviceBackend>::ProblemParams,
                   typename GEMMKernelTest<DeviceBackend>::SolutionParams,
                   typename GEMMKernelTest<DeviceBackend>::MemoryPageAlignment>>;
    void SetUp() override;
    void TearDown() override;
};

#include "RunGEMMKernelTest_impl.hpp"

#endif // RUN_GEMM_KERNEL_TEST_HPP
