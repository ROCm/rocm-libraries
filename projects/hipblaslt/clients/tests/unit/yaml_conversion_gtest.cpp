// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt_parse_data.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

namespace
{
    std::filesystem::path& inputDirectory()
    {
        static std::filesystem::path directory;
        return directory;
    }

    void removeInputDirectory()
    {
        std::error_code error;
        std::filesystem::remove_all(inputDirectory(), error);
    }

    void parseYamlAndExit(const char* contents)
    {
        const auto suffix  = std::chrono::steady_clock::now().time_since_epoch().count();
        const auto root    = std::filesystem::temp_directory_path();
        bool       created = false;
        for(int attempt = 0; attempt < 100 && !created; ++attempt)
        {
            inputDirectory() = root
                               / ("hipblaslt-yaml-test-" + std::to_string(suffix) + "-"
                                  + std::to_string(attempt));
            created = std::filesystem::create_directory(inputDirectory());
        }
        if(!created)
            std::exit(2);

        // The parser calls exit on conversion failure, so stack cleanup cannot
        // remove the input fixture in that case.
        if(std::atexit(removeInputDirectory) != 0)
        {
            removeInputDirectory();
            std::exit(2);
        }

        const auto input = inputDirectory() / "input data.yaml";
        {
            std::ofstream stream(input);
            stream << contents;
            stream.close();
            if(!stream)
                std::exit(2);
        }

        char        executable[] = "yaml-conversion-test";
        char        option[]     = "--yaml";
        std::string filename     = "\"" + input.string() + "\"";
        char*       argv[]       = {executable, option, filename.data(), nullptr};
        int         argc         = 3;
        const bool  parsed       = hipblaslt_parse_data(argc, argv);
        std::exit(parsed && argc == 1 && argv[1] == nullptr ? EXIT_SUCCESS : 2);
    }
}

TEST(HostUnitYamlConversionDeathTest, smoke_RejectsMalformedYaml)
{
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_EXIT(parseYamlAndExit("Tests: [}\n"), ::testing::ExitedWithCode(EXIT_FAILURE), "");
}

TEST(HostUnitYamlConversionDeathTest, smoke_AcceptsValidYaml)
{
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_EXIT(parseYamlAndExit("Tests: []\n"), ::testing::ExitedWithCode(EXIT_SUCCESS), "");
}
