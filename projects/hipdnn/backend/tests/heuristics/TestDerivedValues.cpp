// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <backend/src/heuristics/uhd/FeatureExtractor.hpp>
#include <gtest/gtest.h>

using namespace hipdnn_backend::heuristics::uhd;

class TestDerivedValues : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ctx.bindDeviceVars({{"cu_count", 110.0}});
        // RFC 0019 §6.1: Dims are positional
        // Assuming rank-4 tensor: (batch, heads, sequence, head_dim)
        ctx.bindQueryVars({{"dims[0]", 16.0},    // batch
                           {"dims[1]", 32.0},    // heads
                           {"dims[2]", 2048.0}}  // sequence (seqlen_q)
        );
    }

    FeatureExtractionContext ctx;
};

// Test basic derived value evaluation
TEST_F(TestDerivedValues, BasicDerivedValue)
{
    std::vector<std::pair<std::string, std::string>> derived = {
        {"num_tiles", "{\"ceil_div\": [\"$q.dims[2]\", 64]}"}};

    std::vector<std::string> signature = {"$derived.num_tiles", "$q.dims[0]"};

    FeatureExtractor extractor(signature, derived);

    const auto features = extractor.extract(ctx);
    ASSERT_EQ(features.size(), 2);
    EXPECT_DOUBLE_EQ(features[0], 32.0); // ceil(2048 / 64) = 32
    EXPECT_DOUBLE_EQ(features[1], 16.0); // $q.dims[0]
}

// Test derived value referencing earlier derived value
TEST_F(TestDerivedValues, ChainedDerivedValues)
{
    std::vector<std::pair<std::string, std::string>> derived
        = {{"num_tiles_m", "{\"ceil_div\": [\"$q.dims[2]\", 64]}"},
           {"num_tiles_k", "{\"ceil_div\": [\"$q.dims[1]\", 4]}"},
           {"total_tiles", "{\"*\": [\"$derived.num_tiles_m\", \"$derived.num_tiles_k\"]}"}};

    std::vector<std::string> signature
        = {"$derived.num_tiles_m", "$derived.num_tiles_k", "$derived.total_tiles"};

    FeatureExtractor extractor(signature, derived);

    const auto features = extractor.extract(ctx);
    ASSERT_EQ(features.size(), 3);
    EXPECT_DOUBLE_EQ(features[0], 32.0);  // ceil(2048 / 64)
    EXPECT_DOUBLE_EQ(features[1], 8.0);   // ceil(32 / 4)
    EXPECT_DOUBLE_EQ(features[2], 256.0); // 32 * 8
}

// Test kernel-independent derived value cached in extractSharedRow
TEST_F(TestDerivedValues, KernelIndependentDerivedCached)
{
    std::vector<std::pair<std::string, std::string>> derived
        = {{"total_threads", "{\"*\": [\"$q.dims[0]\", \"$q.dims[1]\"]}"}};

    std::vector<std::string> signature = {"$derived.total_threads", "$q.dims[2]"};

    FeatureExtractor extractor(signature, derived);

    // Extract shared row (should evaluate kernel-independent derived)
    auto sharedRow = extractor.extractSharedRow(ctx);
    ASSERT_EQ(sharedRow.size(), 2);
    EXPECT_DOUBLE_EQ(sharedRow[0], 512.0); // 16 * 32
    EXPECT_DOUBLE_EQ(sharedRow[1], 2048.0);
}

// Test kernel-dependent derived value re-evaluated per candidate
TEST_F(TestDerivedValues, KernelDependentDerivedReEvaluated)
{
    std::vector<std::pair<std::string, std::string>> derived
        = {{"tiles_per_block", "{\"ceil_div\": [\"$q.dims[2]\", \"$kernel.tile_m\"]}"}};

    std::vector<std::string> signature = {"$derived.tiles_per_block", "$q.dims[0]"};

    FeatureExtractor extractor(signature, derived);

    // Get shared row (derived value slot should be 0 because it's kernel-dependent)
    auto row = extractor.extractSharedRow(ctx);
    ASSERT_EQ(row.size(), 2);
    EXPECT_DOUBLE_EQ(row[0], 0.0);   // Kernel-dependent, not yet evaluated
    EXPECT_DOUBLE_EQ(row[1], 16.0);  // $q.dims[0]

    // Bind kernel vars for candidate 1: tile_m=64
    ctx.bindKernelVars({{"tile_m", 64.0}});
    extractor.extractKernelInto(ctx, row);
    EXPECT_DOUBLE_EQ(row[0], 32.0);  // ceil(2048 / 64) = 32
    EXPECT_DOUBLE_EQ(row[1], 16.0);

    // Clear and bind for candidate 2: tile_m=128
    ctx.clearKernelVars();
    ctx.bindKernelVars({{"tile_m", 128.0}});
    extractor.extractKernelInto(ctx, row);
    EXPECT_DOUBLE_EQ(row[0], 16.0);  // ceil(2048 / 128) = 16
    EXPECT_DOUBLE_EQ(row[1], 16.0);
}

// Test mixed kernel-dependent and independent derived values
TEST_F(TestDerivedValues, MixedDerivedDependencies)
{
    std::vector<std::pair<std::string, std::string>> derived = {
        {"total_elems", "{\"*\": [\"$q.dims[0]\", \"$q.dims[2]\"]}"},  // Independent
        {"tiles_per_block", "{\"ceil_div\": [\"$q.dims[2]\", \"$kernel.tile_m\"]}"},  // Dependent
        {"elems_per_tile", "{\"ceil_div\": [\"$derived.total_elems\", \"$derived.tiles_per_block\"]}"} // Mixed
    };

    std::vector<std::string> signature
        = {"$derived.total_elems", "$derived.tiles_per_block", "$derived.elems_per_tile"};

    FeatureExtractor extractor(signature, derived);

    auto row = extractor.extractSharedRow(ctx);
    ASSERT_EQ(row.size(), 3);
    EXPECT_DOUBLE_EQ(row[0], 32768.0);  // 16 * 2048 (independent)
    EXPECT_DOUBLE_EQ(row[1], 0.0);      // Kernel-dependent, not yet evaluated
    EXPECT_DOUBLE_EQ(row[2], 0.0);      // References kernel-dependent derived, also deferred

    // Now bind kernel and fill in dependent slots
    ctx.bindKernelVars({{"tile_m", 64.0}});
    extractor.extractKernelInto(ctx, row);
    EXPECT_DOUBLE_EQ(row[0], 32768.0);
    EXPECT_DOUBLE_EQ(row[1], 32.0);   // ceil(2048 / 64)
    EXPECT_DOUBLE_EQ(row[2], 1024.0); // ceil(32768 / 32)
}

// Test empty derived block (backward compatibility)
TEST_F(TestDerivedValues, EmptyDerivedBlock)
{
    std::vector<std::pair<std::string, std::string>> derived = {};
    std::vector<std::string> signature = {"$q.dims[0]", "$q.dims[2]"};

    FeatureExtractor extractor(signature, derived);

    const auto features = extractor.extract(ctx);
    ASSERT_EQ(features.size(), 2);
    EXPECT_DOUBLE_EQ(features[0], 16.0);
    EXPECT_DOUBLE_EQ(features[1], 2048.0);
}

// Test derived value with complex JsonLogic expression
TEST_F(TestDerivedValues, ComplexDerivedExpression)
{
    std::vector<std::pair<std::string, std::string>> derived = {
        {"score", "{\"if\": [{\">\": [\"$q.dims[0]\", 10]}, {\"*\": [\"$q.dims[0]\", 2]}, \"$q.dims[0]\"]}"}
    };

    std::vector<std::string> signature = {"$derived.score"};

    FeatureExtractor extractor(signature, derived);

    const auto features = extractor.extract(ctx);
    ASSERT_EQ(features.size(), 1);
    EXPECT_DOUBLE_EQ(features[0], 32.0); // batch=16 > 10, so 16 * 2 = 32
}
