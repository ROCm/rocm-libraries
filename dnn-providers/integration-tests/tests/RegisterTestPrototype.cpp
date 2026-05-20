// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// ALMIOPEN-1946 Prototype: Validates that ::testing::RegisterTest with
// a static initializer can auto-discover golden reference bundles from
// disk and register them as individual, named gtest cases.
//
// What this proves:
//   1. Static initializer + RegisterTest registers tests before main()
//   2. --gtest_list_tests shows human-readable suite/test names
//   3. value_param surfaces the bundle path in output
//   4. Dropping a new .json file adds a test on next run
//   5. Zero bundles triggers a hard failure (not a silent pass)
//
// Usage:
//   ./register_test_prototype --gtest_list_tests
//   ./register_test_prototype
//   ./register_test_prototype --gtest_filter=BatchnormFwdInference_nchw_fp32.*

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace
{

// ---------------------------------------------------------------------------
// Test fixture for dynamically registered golden reference tests.
// Each instance receives a bundle path via the factory lambda.
// TestBody() verifies the bundle structure — a real implementation
// would load the graph and validate tensors against CPU reference.
// ---------------------------------------------------------------------------
class GoldenRefTest : public ::testing::Test
{
public:
    std::filesystem::path bundlePath;

    void TestBody() override
    {
        ASSERT_FALSE(bundlePath.empty()) << "Bundle path was not set";

        ASSERT_TRUE(std::filesystem::exists(bundlePath))
            << "Bundle not found: " << bundlePath;

        // Verify the JSON file is non-empty
        const auto fileSize = std::filesystem::file_size(bundlePath);
        EXPECT_GT(fileSize, 0U) << "Bundle JSON is empty: " << bundlePath;

        // Verify at least one .bin tensor file exists alongside the JSON
        const auto bundleDir = bundlePath.parent_path();
        bool hasTensorFile = false;
        for(const auto& entry : std::filesystem::directory_iterator(bundleDir))
        {
            if(entry.path().extension() == ".bin")
            {
                hasTensorFile = true;
                break;
            }
        }
        EXPECT_TRUE(hasTensorFile)
            << "No .bin tensor files found in: " << bundleDir;
    }
};

// ---------------------------------------------------------------------------
// Sentinel test that FAILs when no bundles are discovered.
// Prevents silent green CI when data directory is missing.
// ---------------------------------------------------------------------------
class NoBundlesFailTest : public ::testing::Test
{
public:
    std::string message;

    void TestBody() override
    {
        FAIL() << message;
    }
};

// ---------------------------------------------------------------------------
// Derive a gtest-safe suite name from the directory structure.
// golden_reference_data/BatchnormFwdInference/nchw/fp32/Small/Small.json
//                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                        -> "BatchnormFwdInference_nchw_fp32"
// ---------------------------------------------------------------------------
std::string deriveSuiteName(const std::filesystem::path& jsonPath,
                            const std::filesystem::path& dataRoot)
{
    auto relative = std::filesystem::relative(jsonPath.parent_path(), dataRoot);
    auto comboPath = relative.parent_path();
    auto suiteName = comboPath.string();
    std::replace(suiteName.begin(), suiteName.end(), '/', '_');
    std::replace(suiteName.begin(), suiteName.end(), '\\', '_');
    std::replace(suiteName.begin(), suiteName.end(), '-', '_');
    return suiteName;
}

std::string sanitizeTestName(const std::string& name)
{
    std::string result = name;
    for(auto& ch : result)
    {
        if(std::isalnum(static_cast<unsigned char>(ch)) == 0 && ch != '_')
        {
            ch = '_';
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Scan golden_reference_data/ and register one test per .json bundle.
// Uses static initialization -- same mechanism INSTANTIATE_TEST_SUITE_P uses.
// ---------------------------------------------------------------------------
int registerGoldenRefTests()
{
    namespace fs = std::filesystem;

    const auto exeDir = fs::canonical("/proc/self/exe").parent_path();
    const std::vector<fs::path> searchPaths = {
        exeDir / "../lib/golden_reference_data",
        fs::path("golden_reference_data"),
    };

    fs::path dataRoot;
    for(const auto& candidate : searchPaths)
    {
        if(fs::exists(candidate) && fs::is_directory(candidate))
        {
            dataRoot = fs::canonical(candidate);
            break;
        }
    }

    if(dataRoot.empty())
    {
        std::cerr << "[RegisterTest] ERROR: golden_reference_data not found. Searched:\n";
        for(const auto& p : searchPaths)
        {
            std::cerr << "  - " << p << "\n";
        }
        testing::RegisterTest(
            "GoldenRefDiscovery",
            "FAIL_NoDataDirectory",
            nullptr,
            nullptr,
            __FILE__,
            __LINE__,
            []() -> NoBundlesFailTest* {
                auto* test = new NoBundlesFailTest();
                test->message = "golden_reference_data directory not found";
                return test;
            });
        return 0;
    }

    std::cerr << "[RegisterTest] Scanning: " << dataRoot << "\n";

    int count = 0;
    for(const auto& entry : fs::recursive_directory_iterator(dataRoot))
    {
        if(!entry.is_regular_file() || entry.path().extension() != ".json")
        {
            continue;
        }

        const auto jsonPath = entry.path();
        const auto suiteName = deriveSuiteName(jsonPath, dataRoot);
        const auto testName = sanitizeTestName(jsonPath.stem().string());
        const auto pathStr = jsonPath.string();

        testing::RegisterTest(
            suiteName.c_str(),
            testName.c_str(),
            nullptr,
            pathStr.c_str(),
            __FILE__,
            __LINE__,
            [jsonPath]() -> GoldenRefTest* {
                auto* test = new GoldenRefTest();
                test->bundlePath = jsonPath;
                return test;
            });
        ++count;
    }

    std::cerr << "[RegisterTest] Registered " << count << " tests from " << dataRoot << "\n";

    if(count == 0)
    {
        testing::RegisterTest(
            "GoldenRefDiscovery",
            "FAIL_NoBundlesFound",
            nullptr,
            dataRoot.c_str(),
            __FILE__,
            __LINE__,
            [dataRoot]() -> NoBundlesFailTest* {
                auto* test = new NoBundlesFailTest();
                test->message = "No .json bundles found in " + dataRoot.string();
                return test;
            });
    }

    return count;
}

// Static initializer -- runs before main()
const int g_registeredCount = registerGoldenRefTests();

} // namespace

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
