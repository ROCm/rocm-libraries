/* ************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */

#include <gtest/gtest.h>
#include <boost/filesystem.hpp>

#include <Tensile/ContractionLibrary.hpp>
#ifdef TENSILE_YAML
#include <Tensile/llvm/YAML.hpp>
#endif
#include <TestUtils.hpp>

#include "TestData.hpp"

using namespace TensileLite;

struct RangeLibraryTest
    : public ::testing::TestWithParam<std::tuple<AMDGPU, std::string, bool, bool>>
{
    AMDGPU                                                   hardware;
    std::string                                              filename;
    bool                                                     hasNavi, solutionRequired;
    std::shared_ptr<SolutionLibrary<ContractionProblemGemm>> library;

    static std::map<std::string, std::shared_ptr<SolutionLibrary<ContractionProblemGemm>>>
        libraryCache;

    void SetUp() override
    {
        std::tie(hardware, filename, hasNavi, solutionRequired) = GetParam();

        if(hardware.processor == AMDGPU::Processor::gfx1010 && !hasNavi)
            GTEST_SKIP();

        library = loadLibrary();

        if(library == nullptr)
        {
            std::cout << libraryPath().native() << std::endl;
            if(!boost::filesystem::is_regular_file(libraryPath()))
                GTEST_SKIP();
            else
                ASSERT_NE(library, nullptr);
        }
    }

    boost::filesystem::path libraryPath()
    {
        return TestData::Instance().file(filename);
    }

    std::shared_ptr<SolutionLibrary<ContractionProblemGemm>> loadLibrary(bool cache = true)
    {
        if(!cache)
            return loadLibraryNoCache();

        auto pathStr = libraryPath().native();

        auto iter = libraryCache.find(pathStr);
        if(iter != libraryCache.end())
            return iter->second;

        return libraryCache[pathStr] = loadLibraryNoCache();
    }

    std::shared_ptr<SolutionLibrary<ContractionProblemGemm>> loadLibraryNoCache()
    {
        auto path = libraryPath();

        if(boost::filesystem::is_regular_file(path))
            return LoadLibraryFile<ContractionProblemGemm>(path.native());

        return nullptr;
    }
};

std::map<std::string, std::shared_ptr<SolutionLibrary<ContractionProblemGemm>>>
    RangeLibraryTest::libraryCache;


TEST_P(RangeLibraryTest, SpecificSizes)
{
    // Solution in Range library should be picked for
    // transposeA: true, transposeB false, datatype BFloat16,
    // M in [128, 768], N in [1024, 1408], batch_count in [-1, -1], K in [49024, 49280]

    std::vector<std::tuple<size_t, size_t, size_t, size_t, bool>> MNKB = 
        {{128, 1024, 1,  49024, true},
         {768, 1024, 1,  49024, true},
         {128, 1408, 1,  49024, true},
         {128, 1024, 1,  49280, true},
         {10,  1024, 1,  49024, false},
         {769, 1024, 1,  49024, false},
         {128, 10,   1,  49024, false},
         {128, 1409, 1,  49024, false},
         {128, 1024, 1,  16,    false},
         {128, 1024, 1,  49285, false}};
         

    for(auto [M, N, B, K, in_range] : MNKB)
    {
        auto problem = ContractionProblemGemm::GEMM_Strides(true,
                                                            false,
                                                            rocisa::DataType::BFloat16,
                                                            rocisa::DataType::BFloat16,
                                                            rocisa::DataType::BFloat16,
                                                            rocisa::DataType::BFloat16,
                                                            M,
                                                            N,
                                                            K,
                                                            B,
                                                            K,    // lda
                                                            M*K,  // stridea
                                                            K,    // ldb
                                                            K*N,  // strideb
                                                            M,    // ldc
                                                            M*N,  // stridec
                                                            M,    // ldd
                                                            M*N,  // strided
                                                            2.0); // beta
        problem.setComputeInputType(rocisa::DataType::BFloat16);
        problem.setHighPrecisionAccumulate(true);
        problem.setWorkspaceSize(120324096);

        auto solution = library->findBestSolution(problem, hardware);
        auto solutions = library->findTopSolutions(problem, hardware, 1);
        if(in_range)
        {
            ASSERT_NE(solution, nullptr) << problem;
            ASSERT_EQ(solutions.size(), 1);
        }
        else
        {
            ASSERT_EQ(solution, nullptr) << problem;
            ASSERT_EQ(solutions.size(), 0);
        }
    }
}

std::vector<RangeLibraryTest::ParamType> GetRangeLibrary(std::string const& ext)
{
    std::vector<RangeLibraryTest::ParamType> rv;

    AMDGPU gpu(AMDGPU::Processor::gfx950, 256, "MI350");

    rv.push_back(std::make_tuple(gpu, "Kernels_gfx950." + ext, false, false));

    return rv;
}

std::vector<RangeLibraryTest::ParamType> GetRangeLibraryParams()
{
    std::vector<RangeLibraryTest::ParamType> rv;

#ifdef TENSILE_YAML
    auto yamlParams = GetRangeLibrary("yaml");
    rv.insert(rv.end(), yamlParams.begin(), yamlParams.end());
#endif

#ifdef TENSILE_MSGPACK
    auto datParams = GetRangeLibrary("dat");
    rv.insert(rv.end(), datParams.begin(), datParams.end());
#endif

    return rv;
}

INSTANTIATE_TEST_SUITE_P(RangeLibraryTest, RangeLibraryTest, ::testing::ValuesIn(GetRangeLibraryParams()));
