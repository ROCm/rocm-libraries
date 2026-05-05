// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <string>

#include "harness/SupportMatrixCollector.hpp"

using hipdnn_integration_tests::SupportMatrixCollector;

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
    auto engineName = *records[0].supportingEngines.begin();
    EXPECT_TRUE(engineName.find("Unknown(") != std::string::npos);
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

    std::string tmpPath = "test_support_matrix_output.md";
    collector.setOutputPath(tmpPath);
    collector.recordGraphSupport("Conv", "ConvFprop fp32", "Test1", {}, "", "NHWC");

    collector.writeMarkdown({"TestEngine"});

    std::ifstream inFile(tmpPath);
    ASSERT_TRUE(inFile.is_open());

    std::string content((std::istreambuf_iterator<char>(inFile)), std::istreambuf_iterator<char>());
    inFile.close();
    std::remove(tmpPath.c_str());

    EXPECT_TRUE(content.find("# Engine Support Matrix") != std::string::npos);
    EXPECT_TRUE(content.find("ConvFprop fp32") != std::string::npos);
    EXPECT_TRUE(content.find("TestEngine") != std::string::npos);
}

// NOLINTEND(readability-identifier-naming)
