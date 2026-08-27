// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

// getpid() below stamps the temp path per process. MSVC ships no <unistd.h>;
// it spells the same call _getpid() in <process.h>.
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

#include "harness/SupportMatrixCollector.hpp"

using hipdnn_integration_tests::SupportMatrixCollector;

namespace
{
/// This process's id. MSVC has no <unistd.h> and spells the call _getpid().
int currentProcessId()
{
#ifdef _WIN32
    return _getpid();
#else
    return ::getpid();
#endif
}
} // namespace

// NOLINTBEGIN(readability-identifier-naming) -- gtest macro-generated names

class TestSupportMatrixCollector : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SupportMatrixCollector::get().reset();
    }

    void TearDown() override
    {
        SupportMatrixCollector::get().reset();
    }
};

TEST_F(TestSupportMatrixCollector, SingletonIdentity)
{
    auto& a = SupportMatrixCollector::get();
    auto& b = SupportMatrixCollector::get();
    EXPECT_EQ(&a, &b);
}

TEST_F(TestSupportMatrixCollector, DefaultDisabled)
{
    EXPECT_FALSE(SupportMatrixCollector::get().isEnabled());
}

TEST_F(TestSupportMatrixCollector, SetEnabled)
{
    auto& collector = SupportMatrixCollector::get();
    collector.setEnabled(true);
    EXPECT_TRUE(collector.isEnabled());
    collector.setEnabled(false);
    EXPECT_FALSE(collector.isEnabled());
}

TEST_F(TestSupportMatrixCollector, SetOutputPath)
{
    auto& collector = SupportMatrixCollector::get();
    collector.setOutputPath("test_output.md");
    EXPECT_EQ(collector.getOutputPath(), "test_output.md");
}

TEST_F(TestSupportMatrixCollector, RecordSkippedWhenDisabled)
{
    auto& collector = SupportMatrixCollector::get();
    collector.recordGraphSupport("Conv", "ConvFprop fp32", "Test1", {0});
    EXPECT_TRUE(collector.getRecords().empty());
}

TEST_F(TestSupportMatrixCollector, RecordGraphSupport)
{
    auto& collector = SupportMatrixCollector::get();
    collector.setEnabled(true);
    collector.recordGraphSupport("Conv", "ConvFprop fp32", "Test1", {}, "note1", "NHWC");

    auto records = collector.getRecords();
    ASSERT_EQ(records.size(), 1u);
    EXPECT_EQ(records[0].graphName, "Conv");
    EXPECT_EQ(records[0].graphDescription, "ConvFprop fp32");
    EXPECT_EQ(records[0].testName, "Test1");
    EXPECT_EQ(records[0].note, "note1");
    EXPECT_EQ(records[0].layout, "NHWC");
}

TEST_F(TestSupportMatrixCollector, UnknownEngineId)
{
    auto& collector = SupportMatrixCollector::get();
    collector.setEnabled(true);
    collector.recordGraphSupport("Conv", "ConvFprop fp32", "Test1", {999999});

    auto records = collector.getRecords();
    ASSERT_EQ(records.size(), 1u);
    ASSERT_EQ(records[0].supportingEngines.size(), 1u);
    // An unregistered ID is rendered as its zero-padded hexadecimal, which is the
    // same spelling the backend logs use, so a matrix entry can be grepped for.
    auto engineName = *records[0].supportingEngines.begin();
    EXPECT_EQ(engineName, "0x00000000000F423F");
}

TEST_F(TestSupportMatrixCollector, EngineIdSuppliedByNameMapRendersAsItsName)
{
    auto& collector = SupportMatrixCollector::get();
    collector.setEnabled(true);
    collector.setEngineNames({{999999, "hipkernel:Whatever"}});
    collector.recordGraphSupport("Conv", "ConvFprop fp32", "Test1", {999999});

    auto records = collector.getRecords();
    ASSERT_EQ(records.size(), 1u);
    ASSERT_EQ(records[0].supportingEngines.size(), 1u);
    // The same id the fallback case renders as hexadecimal, so this pins the map lookup
    // rather than a difference between the two ids.
    auto engineName = *records[0].supportingEngines.begin();
    EXPECT_EQ(engineName, "hipkernel:Whatever");
}

TEST_F(TestSupportMatrixCollector, ResetClearsState)
{
    auto& collector = SupportMatrixCollector::get();
    collector.setEnabled(true);
    collector.setOutputPath("custom.md");
    collector.recordGraphSupport("Conv", "ConvFprop fp32", "Test1", {});

    collector.reset();

    EXPECT_FALSE(collector.isEnabled());
    EXPECT_EQ(collector.getOutputPath(), "support_matrix.md");
    EXPECT_TRUE(collector.getRecords().empty());
}

TEST_F(TestSupportMatrixCollector, WriteMarkdownProducesFile)
{
    auto& collector = SupportMatrixCollector::get();
    collector.setEnabled(true);

    // A unique path, not a bare name in the shared CWD. ctest invokes this binary from
    // several tiered suites (see _add_test_target_internal in dnn-providers/cmake/
    // Tests.cmake -- the category suites all run the same executable), so under -j N two
    // copies share a working directory. A fixed name means one copy can truncate or
    // rewrite the file between this test's writeMarkdown() and its read, which surfaces
    // as a file that opens fine but holds the wrong contents.
    const auto tmpPath
        = (std::filesystem::temp_directory_path()
           / ("test_support_matrix_" + std::to_string(currentProcessId()) + "_"
              + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())
              + ".md"))
              .string();
    collector.setOutputPath(tmpPath);
    collector.recordGraphSupport("Conv", "ConvFprop fp32", "Test1", {}, "", "NHWC");

    collector.writeMarkdown({"TestEngine"});

    std::ifstream inFile(tmpPath);
    ASSERT_TRUE(inFile.is_open()) << "collector wrote nothing to " << tmpPath;

    const std::string content((std::istreambuf_iterator<char>(inFile)),
                              std::istreambuf_iterator<char>());
    inFile.close();
    std::remove(tmpPath.c_str());

    EXPECT_TRUE(content.find("# Engine Support Matrix") != std::string::npos);
    EXPECT_TRUE(content.find("ConvFprop fp32") != std::string::npos);
    EXPECT_TRUE(content.find("TestEngine") != std::string::npos);
}

// NOLINTEND(readability-identifier-naming)
