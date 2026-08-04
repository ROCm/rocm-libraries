// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestFeatureExtractor.cpp
 * @brief Tests for the UHD feature extraction system.
 */

#include "heuristics/uhd/FeatureExtractor.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <unordered_set>

using hipdnn_backend::heuristics::uhd::FeatureExtractionContext;
using hipdnn_backend::heuristics::uhd::FeatureExtractor;

namespace
{

class TestFeatureExtractor : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Set up a basic device context
        FeatureExtractionContext::ValueMap deviceVars = {
            {"cu_count", 120.0},
            {"warp_size", int64_t{64}},
            {"total_global_mem", int64_t{68719476736}}, // 64 GB
        };
        _ctx.bindDeviceVars(deviceVars);

        // Set up kernel metadata
        FeatureExtractionContext::ValueMap kernelVars = {
            {"tile_m", 64.0},
            {"tile_n", 64.0},
            {"tile_k", 16.0},
            {"split_k", 1.0},
        };
        _ctx.bindKernelVars(kernelVars);

        // Set up query properties
        FeatureExtractionContext::ValueMap queryVars = {
            {"batch", 32.0},
            {"seqlen", 512.0},
            {"heads", 8.0},
        };
        _ctx.bindQueryVars(queryVars);
    }

    FeatureExtractionContext _ctx;
};

// ========== Basic feature extraction ==========

TEST_F(TestFeatureExtractor, ExtractsSingleFeature)
{
    std::vector<std::string> signature = {"\"$device.cu_count\""};
    FeatureExtractor extractor(signature);

    auto features = extractor.extract(_ctx);
    ASSERT_EQ(features.size(), 1u);
    EXPECT_DOUBLE_EQ(features[0], 120.0);
}

TEST_F(TestFeatureExtractor, ExtractsMultipleFeatures)
{
    std::vector<std::string> signature = {
        "\"$device.cu_count\"",
        "\"$kernel.tile_m\"",
        "\"$q.batch\"",
    };
    FeatureExtractor extractor(signature);

    auto features = extractor.extract(_ctx);
    ASSERT_EQ(features.size(), 3u);
    EXPECT_DOUBLE_EQ(features[0], 120.0);
    EXPECT_DOUBLE_EQ(features[1], 64.0);
    EXPECT_DOUBLE_EQ(features[2], 32.0);
}

TEST_F(TestFeatureExtractor, ExtractsComputedFeatures)
{
    std::vector<std::string> signature = {
        R"({"+": ["$kernel.tile_m", "$kernel.tile_n"]})",    // 64 + 64 = 128
        R"({"*": ["$q.batch", "$q.seqlen"]})",               // 32 * 512 = 16384
        R"({"ceil_div": ["$device.cu_count", "$kernel.tile_m"]})", // ceil(120/64) = 2
    };
    FeatureExtractor extractor(signature);

    auto features = extractor.extract(_ctx);
    ASSERT_EQ(features.size(), 3u);
    EXPECT_DOUBLE_EQ(features[0], 128.0);
    EXPECT_DOUBLE_EQ(features[1], 16384.0);
    EXPECT_DOUBLE_EQ(features[2], 2.0);
}

// ========== Feature count ==========

TEST_F(TestFeatureExtractor, ReportsCorrectFeatureCount)
{
    std::vector<std::string> signature = {
        "\"$device.cu_count\"",
        "\"$kernel.tile_m\"",
        "\"$kernel.tile_n\"",
        "\"$q.batch\"",
    };
    FeatureExtractor extractor(signature);
    EXPECT_EQ(extractor.featureCount(), 4u);
}

// ========== Variable references ==========

TEST_F(TestFeatureExtractor, CollectsVariableReferences)
{
    std::vector<std::string> signature = {
        "\"$device.cu_count\"",
        R"({"*": ["$kernel.tile_m", "$q.batch"]})",
    };
    FeatureExtractor extractor(signature);

    const auto& refs = extractor.getVariableRefs();
    EXPECT_EQ(refs.size(), 3u);
    EXPECT_TRUE(refs.count("$device.cu_count") > 0);
    EXPECT_TRUE(refs.count("$kernel.tile_m") > 0);
    EXPECT_TRUE(refs.count("$q.batch") > 0);
}

// ========== Context validation ==========

TEST_F(TestFeatureExtractor, ValidatesCompleteContext)
{
    std::vector<std::string> signature = {
        "\"$device.cu_count\"",
        "\"$kernel.tile_m\"",
    };
    FeatureExtractor extractor(signature);
    EXPECT_TRUE(extractor.validateContext(_ctx));
}

TEST_F(TestFeatureExtractor, DetectsIncompleteContext)
{
    std::vector<std::string> signature = {
        "\"$device.cu_count\"",
        "\"$kernel.missing_field\"",
    };
    FeatureExtractor extractor(signature);
    EXPECT_FALSE(extractor.validateContext(_ctx));
}

TEST_F(TestFeatureExtractor, ReportsMissingVariables)
{
    std::vector<std::string> signature = {
        "\"$device.cu_count\"",
        "\"$kernel.missing_field\"",
        "\"$q.unknown\"",
    };
    FeatureExtractor extractor(signature);

    auto missing = extractor.getMissingVariables(_ctx);
    EXPECT_EQ(missing.size(), 2u);

    std::unordered_set<std::string> missingSet(missing.begin(), missing.end());
    EXPECT_TRUE(missingSet.count("$kernel.missing_field") > 0);
    EXPECT_TRUE(missingSet.count("$q.unknown") > 0);
}

// ========== Signature hash ==========

TEST_F(TestFeatureExtractor, ComputesConsistentHash)
{
    std::vector<std::string> signature = {
        "\"$device.cu_count\"",
        "\"$kernel.tile_m\"",
    };
    FeatureExtractor extractor1(signature);
    FeatureExtractor extractor2(signature);

    EXPECT_EQ(extractor1.getSignatureHash(), extractor2.getSignatureHash());
}

TEST_F(TestFeatureExtractor, DifferentSignaturesDifferentHash)
{
    std::vector<std::string> sig1 = {"\"$device.cu_count\""};
    std::vector<std::string> sig2 = {"\"$kernel.tile_m\""};

    FeatureExtractor extractor1(sig1);
    FeatureExtractor extractor2(sig2);

    EXPECT_NE(extractor1.getSignatureHash(), extractor2.getSignatureHash());
}

TEST_F(TestFeatureExtractor, HashLengthIsConsistent)
{
    std::vector<std::string> signature = {"\"$device.cu_count\""};
    FeatureExtractor extractor(signature);

    // Hash should be 64 chars (padded to SHA-256 length)
    EXPECT_EQ(extractor.getSignatureHash().length(), 64u);
}

// ========== KMD field validation ==========

TEST_F(TestFeatureExtractor, ValidatesKnownKmdFields)
{
    std::vector<std::string> signature = {
        "\"$kernel.tile_m\"",
        "\"$kernel.tile_n\"",
    };
    FeatureExtractor extractor(signature);

    std::unordered_set<std::string> kmdFields = {"tile_m", "tile_n", "tile_k", "split_k"};
    EXPECT_TRUE(extractor.validateAgainstKmdFields(kmdFields));
}

TEST_F(TestFeatureExtractor, DetectsMissingKmdFields)
{
    std::vector<std::string> signature = {
        "\"$kernel.tile_m\"",
        "\"$kernel.unknown_field\"",
    };
    FeatureExtractor extractor(signature);

    std::unordered_set<std::string> kmdFields = {"tile_m", "tile_n"};
    EXPECT_FALSE(extractor.validateAgainstKmdFields(kmdFields));
}

TEST_F(TestFeatureExtractor, ReportsMissingKmdFields)
{
    std::vector<std::string> signature = {
        "\"$kernel.tile_m\"",
        "\"$kernel.unknown1\"",
        "\"$kernel.unknown2\"",
    };
    FeatureExtractor extractor(signature);

    std::unordered_set<std::string> kmdFields = {"tile_m"};
    auto missing = extractor.getMissingKmdFields(kmdFields);

    EXPECT_EQ(missing.size(), 2u);
    std::unordered_set<std::string> missingSet(missing.begin(), missing.end());
    EXPECT_TRUE(missingSet.count("unknown1") > 0);
    EXPECT_TRUE(missingSet.count("unknown2") > 0);
}

TEST_F(TestFeatureExtractor, NonKernelVarsIgnoredInKmdValidation)
{
    std::vector<std::string> signature = {
        "\"$device.cu_count\"",
        "\"$q.batch\"",
    };
    FeatureExtractor extractor(signature);

    // Empty KMD fields should still pass because no $kernel.* refs exist
    std::unordered_set<std::string> emptyKmdFields;
    EXPECT_TRUE(extractor.validateAgainstKmdFields(emptyKmdFields));
}

// ========== Context binding ==========

TEST_F(TestFeatureExtractor, ClearResetsContext)
{
    _ctx.clear();

    std::vector<std::string> signature = {"\"$device.cu_count\""};
    FeatureExtractor extractor(signature);
    EXPECT_FALSE(extractor.validateContext(_ctx));
}

TEST_F(TestFeatureExtractor, SingleBindAddsVariable)
{
    FeatureExtractionContext ctx;
    ctx.bind("$custom.value", 42.0);

    std::vector<std::string> signature = {"\"$custom.value\""};
    FeatureExtractor extractor(signature);

    auto features = extractor.extract(ctx);
    ASSERT_EQ(features.size(), 1u);
    EXPECT_DOUBLE_EQ(features[0], 42.0);
}

} // namespace
