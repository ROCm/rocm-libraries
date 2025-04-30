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
#ifndef RUN_GEMM_KERNEL_TEST_HPP_IMPL
#define RUN_GEMM_KERNEL_TEST_HPP_IMPL

#include <Tensile/Tensile.hpp>

#include "GEMMKernelTest.hpp"
#include "RunGEMMKernelTest.hpp"

///////////////////////////////////////////////////////////////
// template<typename DeviceBackend>
// class RunGEMMKernelTestParams
///////////////////////////////////////////////////////////////

template <typename DeviceBackend>
auto RunGEMMKernelTestParams<DeviceBackend>::TypedTests()
    -> std::vector<std::shared_ptr<TestInterface>>
{
    static auto testFloat
        = std::make_shared<TypedGEMMKernelTest<ContractionInputs_S_S_S, DeviceBackend>>();
    //static auto testDouble
    //    = std::make_shared<TypedGEMMKernelTest<ContractionInputs_D_D_D>>();
    //     static auto testCFloat = std::make_shared<TypedGEMMKernelTest<ComplexContractionInputs_S_S_S>>();
    //     static auto testCDouble = std::make_shared<TypedGEMMKernelTest<ComplexContractionInputs_D_D_D>>();
    //     static auto testInt8x4 = std::make_shared<TypedGEMMKernelTest<ContractionInputs_I8_I32_I32>>();
    //     static auto testInt32 = std::make_shared<TypedGEMMKernelTest<ContractionInputs_I32_I32_I32>>();
    //     static auto testHalf = std::make_shared<TypedGEMMKernelTest<ContractionInputs_H_H_H>>();
    // #ifdef TENSILE_USE_BF16
    //     static auto testBF16 = std::make_shared<TypedGEMMKernelTest<ContractionInputs_B_B_S>>();
    // #endif
    return std::vector<std::shared_ptr<TestInterface>>{
        testFloat,
        // testDouble,
        //         testCFloat,
        //         testCDouble,
        //         testInt8x4,
        //         testInt32,
        //         testHalf,
        // #ifdef TENSILE_USE_BF16
        //         testBF16,
        // #endif
    };
}

template <typename DeviceBackend>
auto RunGEMMKernelTestParams<DeviceBackend>::TestProblems() -> std::vector<ProblemParams>
{
    return std::vector<ProblemParams>{

        //{false, false, 5760, 5760, 5760, 5760, 5760, 5760, 1.5, 4},
        //{false,  true, 5760, 5760, 5760, 5760, 5760, 5760, 1.5, 4},
        //{ true, false, 5760, 5760, 5760, 5760, 5760, 5760, 1.5, 4},
        //{ true,  true, 5760, 5760, 5760, 5760, 5760, 5760, 1.5, 4},
        std::make_tuple(false, false, 4, 4, 6, 4, 6, 4, 1.5, 2),
        std::make_tuple(false, true, 4, 4, 6, 4, 4, 4, 1.5, 2),
        std::make_tuple(true, false, 4, 4, 6, 6, 6, 4, 1.5, 2),
        std::make_tuple(true, true, 4, 4, 6, 6, 4, 4, 1.5, 2),

        std::make_tuple(false, false, 15, 15, 15, 15, 15, 15, 1.5, 1),
        std::make_tuple(false, true, 15, 15, 15, 15, 15, 15, 1.5, 1),
        std::make_tuple(true, false, 15, 15, 15, 15, 15, 15, 1.5, 1),
        std::make_tuple(true, true, 15, 15, 15, 15, 15, 15, 1.5, 1),

        std::make_tuple(false, false, 16, 16, 16, 16, 16, 16, 1.5, 1),
        std::make_tuple(false, true, 16, 16, 16, 16, 16, 16, 1.5, 1),
        std::make_tuple(true, false, 16, 16, 16, 16, 16, 16, 1.5, 1),
        std::make_tuple(true, true, 16, 16, 16, 16, 16, 16, 1.5, 1),

        std::make_tuple(false, false, 17, 17, 17, 17, 17, 17, 1.5, 1),
        std::make_tuple(false, true, 17, 17, 17, 17, 17, 17, 1.5, 1),
        std::make_tuple(true, false, 17, 17, 17, 17, 17, 17, 1.5, 1),
        std::make_tuple(true, true, 17, 17, 17, 17, 17, 17, 1.5, 1),

        std::make_tuple(false, false, 31, 31, 31, 31, 31, 31, 1.5, 1),
        std::make_tuple(false, true, 31, 31, 31, 31, 31, 31, 1.5, 1),
        std::make_tuple(true, false, 31, 31, 31, 31, 31, 31, 1.5, 1),
        std::make_tuple(true, true, 31, 31, 31, 31, 31, 31, 1.5, 1),

        std::make_tuple(false, false, 32, 32, 32, 32, 32, 32, 1.5, 1),
        std::make_tuple(false, true, 32, 32, 32, 32, 32, 32, 1.5, 1),
        std::make_tuple(true, false, 32, 32, 32, 32, 32, 32, 1.5, 1),
        std::make_tuple(true, true, 32, 32, 32, 32, 32, 32, 1.5, 1),

        std::make_tuple(false, false, 33, 33, 33, 33, 33, 33, 1.5, 1),
        std::make_tuple(false, true, 33, 33, 33, 33, 33, 33, 1.5, 1),
        std::make_tuple(true, false, 33, 33, 33, 33, 33, 33, 1.5, 1),
        std::make_tuple(true, true, 33, 33, 33, 33, 33, 33, 1.5, 1),

        std::make_tuple(false, false, 34, 34, 34, 34, 34, 34, 1.5, 1),
        std::make_tuple(false, true, 34, 34, 34, 34, 34, 34, 1.5, 1),
        std::make_tuple(true, false, 34, 34, 34, 34, 34, 34, 1.5, 1),
        std::make_tuple(true, true, 34, 34, 34, 34, 34, 34, 1.5, 1),

        std::make_tuple(false, false, 234, 123, 634, 234, 634, 234, 1.5, 1),
        std::make_tuple(false, false, 234, 123, 634, 245, 768, 249, 1.5, 12),
        std::make_tuple(false, true, 234, 123, 634, 245, 768, 249, 1.5, 12),
        std::make_tuple(true, false, 234, 123, 634, 768, 768, 249, 1.5, 12),
        std::make_tuple(true, true, 234, 123, 634, 768, 768, 249, 1.5, 12),

        std::make_tuple(false, false, 1, 4, 6, 1, 6, 1, 1.5, 1),
        std::make_tuple(false, false, 4, 1, 6, 4, 6, 4, 1.5, 1),
        std::make_tuple(false, false, 4, 4, 1, 4, 1, 4, 1.5, 1),

        std::make_tuple(false, true, 1, 4, 6, 1, 4, 1, 1.5, 1),
        std::make_tuple(false, true, 4, 1, 6, 4, 1, 4, 1.5, 1),
        std::make_tuple(false, true, 4, 4, 1, 4, 4, 4, 1.5, 1),

        std::make_tuple(true, false, 1, 4, 6, 6, 6, 1, 1.5, 1),
        std::make_tuple(true, false, 4, 1, 6, 6, 6, 4, 1.5, 1),
        std::make_tuple(true, false, 4, 4, 1, 1, 1, 4, 1.5, 1),

        std::make_tuple(true, true, 1, 4, 6, 6, 4, 1, 1.5, 1),
        std::make_tuple(true, true, 4, 1, 6, 6, 1, 4, 1.5, 1),
        std::make_tuple(true, true, 4, 4, 1, 1, 4, 4, 1.5, 1),

        TestInterface::RandomGEMMParams,
        TestInterface::RandomGEMMParams};
}

template <typename DeviceBackend>
auto RunGEMMKernelTestParams<DeviceBackend>::TestProblemsExtended() -> std::vector<ProblemParams>
{
    return std::vector<ProblemParams>{
        TestInterface::RandomGEMMParams,
        TestInterface::RandomGEMMParams,
        TestInterface::RandomGEMMParams,
        TestInterface::RandomGEMMParams,
        TestInterface::RandomGEMMParams,
        TestInterface::RandomGEMMParams,
        std::make_tuple(false, true, 1, 128, 256, 1, 270, 49928, 1.5, 1),
        std::make_tuple(false, true, 384, 1, 384, 384, 270, 49928, 1.5, 1),
        std::make_tuple(true, true, 4, 4, 1, 1, 4, 4, 1.5, 1),

        std::make_tuple(false, false, 16328, 384, 384, 16328, 384, 16328, 2.0, 1),
        std::make_tuple(false, true, 16328, 384, 384, 16328, 16328, 16328, 2.0, 1),
        std::make_tuple(true, false, 16328, 384, 384, 384, 384, 16328, 2.0, 1),
        std::make_tuple(true, true, 16328, 384, 384, 384, 16328, 16328, 2.0, 1)};
}

template <typename DeviceBackend>
auto RunGEMMKernelTestParams<DeviceBackend>::TestLibraries_Impl() -> std::vector<SolutionParams>
{
    bool debug = Debug::Instance().printKernelArguments();

    std::vector<SolutionParams> rv;

    {
        auto library
            = EmbeddedLibrary<ContractionProblem, ContractionSolution>::Get("kernels_lite");
        auto adapter = std::make_shared<SolutionAdapter>(debug, "kernels_lite");
        adapter->loadEmbeddedCodeObjects("kernels_lite");
        rv.emplace_back(library, adapter, false);
    }

    {
        auto library
            = EmbeddedLibrary<ContractionProblem, ContractionSolution>::Get("kernels_lite_mixed");
        auto adapter = std::make_shared<SolutionAdapter>(debug, "kernels_lite_mixed");
        adapter->loadEmbeddedCodeObjects("kernels_lite_mixed");
        rv.emplace_back(library, adapter, true);
    }

    {
        auto library = LoadLibraryFile<ContractionProblem, ContractionSolution>(
            TestData::Instance().file("test_kernels_lite/library/TensileLibrary").native());
        auto adapter = std::make_shared<SolutionAdapter>(debug, "kernels_lite (file)");
        for(auto file : TestData::Instance().glob("test_kernels_lite/library/*.*co"))
            adapter->loadCodeObjectFile(file.native());

        rv.emplace_back(library, adapter, false);
    }

    {
        auto library = LoadLibraryFile<ContractionProblem, ContractionSolution>(
            TestData::Instance().file("test_kernels_lite_mixed/library/TensileLibrary").native());
        auto adapter = std::make_shared<SolutionAdapter>(debug, "kernels_lite_mixed (file)");
        for(auto file : TestData::Instance().glob("test_kernels_lite_mixed/library/*.*co"))
            adapter->loadCodeObjectFile(file.native());

        rv.emplace_back(library, adapter, true);
    }

    {
        auto library = LoadLibraryFile<ContractionProblem, ContractionSolution>(
            TestData::Instance().file("test_tile_aware_selection/library/TensileLibrary").native());

        auto adapter = std::make_shared<SolutionAdapter>(debug, "tile_aware_selection");
        for(auto file : TestData::Instance().glob("test_tile_aware_selection/library/*.*co"))
            adapter->loadCodeObjectFile(file.native());

        for(auto file : TestData::Instance().glob("test_tile_aware_selection/library/*.*hsaco"))
            adapter->loadCodeObjectFile(file.native());

        rv.emplace_back(library, adapter, false);
    }

    auto envDir = TestData::Env("TENSILE_TEST_LIBRARY");
    if(envDir)
    {
        auto library = LoadLibraryFile<ContractionProblem, ContractionSolution>(
            envDir.file("TensileLibrary").native());
        auto adapter = std::make_shared<SolutionAdapter>(debug, "TENSILE_TEST_LIBRARY");
        auto device  = std::dynamic_pointer_cast<AMDGPU>(DeviceBackend::getCurrentDevice());
        auto arch    = device->processor;

        for(auto file : envDir.glob(concatenate("*-", arch, ".co")))
        {
            adapter->loadCodeObjectFile(file.native());
        }

        for(auto file : envDir.glob(concatenate("*-", arch, ".hsaco")))
        {
            try
            {
                adapter->loadCodeObjectFile(file.native());
            }
            catch(std::logic_error& exc)
            {
            }
        }

        rv.emplace_back(library, adapter, false);
    }

    return rv;
}

template <typename DeviceBackend>
auto RunGEMMKernelTestParams<DeviceBackend>::TestLibraries() -> std::vector<SolutionParams>
{
    // Prevent the libraries from being loaded twice.
    static auto rv = TestLibraries_Impl();
    return rv;
}

template <typename DeviceBackend>
auto RunGEMMKernelTestParams<DeviceBackend>::TestMemoryAlignments()
    -> std::vector<MemoryPageAlignment>
{
    return std::vector<MemoryPageAlignment>{MemoryPageAlignment::BEGIN, MemoryPageAlignment::END};
}

///////////////////////////////////////////////////////////////
// template<typename DeviceBackend>
// class RunGEMMKernelTestParams
///////////////////////////////////////////////////////////////

template <typename DeviceBackend>
void RunGEMMKernelTest<DeviceBackend>::SetUp()
{
    auto param          = Base::GetParam();
    auto typedTest      = std::get<0>(param);
    auto problemParams  = std::get<1>(param);
    auto solutionParams = std::get<2>(param);
    auto pageAlignment  = std::get<3>(param);
    typedTest->SetUp(problemParams, solutionParams, pageAlignment);
}

template <typename DeviceBackend>
void RunGEMMKernelTest<DeviceBackend>::TearDown()
{
    auto param     = Base::GetParam();
    auto typedTest = std::get<0>(param);
    typedTest->TearDown();
}

#endif // RUN_GEMM_KERNEL_TEST_HPP_IMPL
