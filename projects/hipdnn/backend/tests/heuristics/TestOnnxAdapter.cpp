// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestOnnxAdapter.cpp
 * @brief Tests for OnnxAdapter (ONNX Runtime inference) per RFC 0019 §7.3.
 *
 * Current status: DEPENDENCY-GATED STUB
 * - ONNX Runtime is not available in the build environment
 * - Tests verify the stub behavior: load() returns nullptr with clear error log
 * - This closes the contract gap (ONNX is schema-valid, not silently rejected)
 * - When ONNX Runtime is added, expand these tests to cover:
 *   - Loading .onnx models
 *   - Scoring feature vectors via ONNX inference
 *   - Features hash validation
 *   - Input shape validation
 */

#include "heuristics/uhd/adapters/OnnxAdapter.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

using namespace hipdnn_backend::heuristics::uhd;

namespace
{
constexpr const char* TEST_HASH = "sha256:test_hash_12345678";
} // namespace

class TestOnnxAdapter : public ::testing::Test
{
};

// DEPENDENCY-GATED STUB TESTS
// These tests verify that the ONNX adapter correctly reports unavailability
// when ONNX Runtime is not present, rather than silently failing or crashing.

TEST_F(TestOnnxAdapter, LoadReturnsNullptrWhenDependencyUnavailable)
{
    // ONNX Runtime is not available in the current build environment.
    // load() should return nullptr with a clear error message.
    auto adapter = OnnxAdapter::load("/nonexistent/model.onnx", TEST_HASH);
    EXPECT_EQ(adapter, nullptr);
}

TEST_F(TestOnnxAdapter, LoadReturnsNullptrForValidPath)
{
    // Even with a plausible path, load should return nullptr when ONNX Runtime is unavailable.
    auto adapter = OnnxAdapter::load("/tmp/model.onnx", TEST_HASH);
    EXPECT_EQ(adapter, nullptr);
}

TEST_F(TestOnnxAdapter, LoadReturnsNullptrForEmptyPath)
{
    // Empty path should also return nullptr (same dependency-unavailable behavior).
    auto adapter = OnnxAdapter::load("", TEST_HASH);
    EXPECT_EQ(adapter, nullptr);
}

// TODO: When ONNX Runtime dependency is added, expand with these tests:
//
// TEST_F(TestOnnxAdapter, LoadAndScoreSimpleModel)
// {
//     // Load a simple .onnx model with 2 inputs, 1 output
//     auto adapter = OnnxAdapter::load("test_models/linear.onnx", TEST_HASH);
//     ASSERT_NE(adapter, nullptr);
//     EXPECT_EQ(adapter->type(), UhdAdapterType::ONNX);
//     EXPECT_EQ(adapter->expectedFeatureCount(), 2U);
//
//     // Score a feature vector
//     double score = adapter->score({1.0, 2.0});
//     EXPECT_GT(score, 0.0);
// }
//
// TEST_F(TestOnnxAdapter, LoadFailsOnMissingFile)
// {
//     auto adapter = OnnxAdapter::load("/nonexistent/model.onnx", TEST_HASH);
//     EXPECT_EQ(adapter, nullptr);
// }
//
// TEST_F(TestOnnxAdapter, LoadFailsOnInvalidOnnxFile)
// {
//     auto adapter = OnnxAdapter::load("test_models/not_an_onnx.txt", TEST_HASH);
//     EXPECT_EQ(adapter, nullptr);
// }
//
// TEST_F(TestOnnxAdapter, LoadFailsOnFeatureHashMismatch)
// {
//     // Model embeds features_hash, but load() is called with different hash
//     auto adapter = OnnxAdapter::load("test_models/linear.onnx", "sha256:wrong_hash");
//     EXPECT_EQ(adapter, nullptr);
// }
//
// TEST_F(TestOnnxAdapter, ScoreThrowsOnFeatureCountMismatch)
// {
//     auto adapter = OnnxAdapter::load("test_models/linear.onnx", TEST_HASH);
//     ASSERT_NE(adapter, nullptr);
//
//     // Model expects 2 features, provide 3
//     EXPECT_THROW(adapter->score({1.0, 2.0, 3.0}), std::invalid_argument);
// }
//
// TEST_F(TestOnnxAdapter, ScoreBatch)
// {
//     auto adapter = OnnxAdapter::load("test_models/linear.onnx", TEST_HASH);
//     ASSERT_NE(adapter, nullptr);
//
//     const std::vector<std::vector<double>> batch = {{1.0, 2.0}, {3.0, 4.0}};
//     auto scores = adapter->scoreBatch(batch);
//     ASSERT_EQ(scores.size(), 2U);
//     EXPECT_GT(scores[0], 0.0);
//     EXPECT_GT(scores[1], 0.0);
// }
