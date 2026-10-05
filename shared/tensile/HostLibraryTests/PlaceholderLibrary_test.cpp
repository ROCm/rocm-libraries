// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/PlaceholderLibrary.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>

using namespace Tensile;

namespace
{
    class PlaceholderLibraryTest : public ::testing::Test
    {
    protected:
        std::filesystem::path                  directory;
        PlaceholderLibrary<ContractionProblem> placeholder;
        SolutionMap<ContractionSolution>       solutions;
        std::mutex                             solutionsGuard;
        AMDGPU                                 hardware;
        ContractionProblem                     problem
            = ContractionProblem::GEMM(false, false, 4, 4, 4, 4, 4, 4, 1.5, false, 2);

        void SetUp() override
        {
            const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
            for(unsigned attempt = 0;; ++attempt)
            {
                auto candidate = std::filesystem::temp_directory_path()
                                 / ("tensile-placeholder-" + std::to_string(stamp) + "-"
                                    + std::to_string(attempt));
                if(std::filesystem::create_directory(candidate))
                {
                    directory = std::move(candidate);
                    break;
                }
            }
            placeholder.libraryDirectory = directory.generic_string();
            placeholder.filePrefix       = "TensileLibrary_Type_SS_fallback";
            placeholder.suffix           = ".dat";
            placeholder.masterSolutions  = &solutions;
            placeholder.solutionsGuard   = &solutionsGuard;
        }

        void TearDown() override
        {
            if(!directory.empty())
                std::filesystem::remove_all(directory);
        }

        std::filesystem::path filename() const
        {
            return directory / (placeholder.filePrefix + placeholder.suffix);
        }

        void writeValidLibrary()
        {
            // MessagePack: {solutions: [], library: {type: Hardware, rows: []}}.
            // A real serialized library with no applicable solutions; no GPU needed.
            const std::string data
                = "\x82\xa9solutions\x90\xa7library\x82\xa4type\xa8Hardware\xa4rows\x90";
            std::ofstream out(filename(), std::ios::binary);
            out.write(data.data(), data.size());
            ASSERT_TRUE(out.good());
        }
    };
}

TEST_F(PlaceholderLibraryTest, MissingFileReportsRequestedPath)
{
    try
    {
        placeholder.findBestSolution(problem, hardware);
        FAIL() << "Missing fallback metadata must report an error";
    }
    catch(const std::runtime_error& error)
    {
        EXPECT_NE(std::string(error.what()).find(filename().generic_string()), std::string::npos);
    }
    EXPECT_EQ(placeholder.library, nullptr);
    EXPECT_TRUE(solutions.empty());
}

TEST_F(PlaceholderLibraryTest, MissingFileFailsAllSolutionQueries)
{
    EXPECT_THROW(placeholder.findAllSolutions(problem, hardware), std::runtime_error);
    EXPECT_THROW(placeholder.findAllSolutionsMatchingType(problem, hardware), std::runtime_error);
    EXPECT_EQ(placeholder.library, nullptr);
    EXPECT_TRUE(solutions.empty());
}

TEST_F(PlaceholderLibraryTest, MalformedFileReportsError)
{
    {
        std::ofstream out(filename(), std::ios::binary);
        out.put('\xc1');
    }
    EXPECT_THROW(placeholder.loadPlaceholderLibrary(), std::runtime_error);
    EXPECT_EQ(placeholder.library, nullptr);
    EXPECT_TRUE(solutions.empty());
}

TEST_F(PlaceholderLibraryTest, ValidLibraryLoadsAndRemainsCached)
{
    writeValidLibrary();
    ASSERT_TRUE(placeholder.loadPlaceholderLibrary(&hardware));
    ASSERT_NE(placeholder.library, nullptr);
    EXPECT_EQ(placeholder.findBestSolution(problem, hardware), nullptr);
    EXPECT_TRUE(placeholder.findAllSolutions(problem, hardware).empty());
    EXPECT_TRUE(placeholder.findAllSolutionsMatchingType(problem, hardware).empty());
    std::filesystem::remove(filename());
    EXPECT_FALSE(placeholder.loadPlaceholderLibrary(&hardware));
}

TEST_F(PlaceholderLibraryTest, FailedLoadCanBeRetriedAfterFileIsRestored)
{
    EXPECT_THROW(placeholder.loadPlaceholderLibrary(), std::runtime_error);
    writeValidLibrary();
    EXPECT_TRUE(placeholder.loadPlaceholderLibrary());
    EXPECT_NE(placeholder.library, nullptr);
}
