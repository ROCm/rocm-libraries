// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestEngineRegistryArchResolution.cpp
 * @brief Unit tests for RFC 0019 §8.3 architecture resolution in EngineEntry.
 *
 * Tests the resolveUhd() static method which implements the fallback chain:
 * exact arch match → "default" → nullopt
 */

#include <heuristics/uhd/EngineRegistry.hpp>

#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <unordered_map>

namespace hipdnn_backend::heuristics::uhd
{

namespace
{

/// Helper: Build a minimal UhdConfig for testing
UhdConfig makeTestConfig(const std::string& uhdId, const std::string& transform = "identity")
{
    UhdConfig cfg;
    cfg.uhdId            = uhdId;
    cfg.name             = "Test UHD";
    cfg.adapterType      = "static_order";
    cfg.objective        = "max";
    cfg.scoreTransform   = transform;
    cfg.scoreUnits       = "tflops";
    cfg.scoreCalibrated  = false;
    return cfg;
}

} // namespace

// ========== RFC 0019 §8.3: Architecture Resolution Tests ==========

TEST(TestEngineEntryArchResolution, ExactArchMatch)
{
    // Setup: Register UHDs for gfx942 and default
    std::unordered_map<std::string, UhdConfig> roleMap;
    roleMap["gfx942"]  = makeTestConfig("uhd_gfx942", "log1p");
    roleMap["default"] = makeTestConfig("uhd_default", "identity");

    // Test: Request gfx942 (exact match exists)
    auto result = EngineEntry::resolveUhd(roleMap, "gfx942");

    // Verify: Should return gfx942 config
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->uhdId, "uhd_gfx942");
    EXPECT_EQ(result->scoreTransform, "log1p");
}

TEST(TestEngineEntryArchResolution, FallbackToDefault)
{
    // Setup: Register only default (no gfx1100 entry)
    std::unordered_map<std::string, UhdConfig> roleMap;
    roleMap["default"] = makeTestConfig("uhd_default", "identity");

    // Test: Request gfx1100 (no exact match, fallback to default)
    auto result = EngineEntry::resolveUhd(roleMap, "gfx1100");

    // Verify: Should return default config
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->uhdId, "uhd_default");
    EXPECT_EQ(result->scoreTransform, "identity");
}

TEST(TestEngineEntryArchResolution, NoMatchReturnsNullopt)
{
    // Setup: Register only gfx942 (no default, no gfx950)
    std::unordered_map<std::string, UhdConfig> roleMap;
    roleMap["gfx942"] = makeTestConfig("uhd_gfx942", "log1p");

    // Test: Request gfx950 (no exact match, no default)
    auto result = EngineEntry::resolveUhd(roleMap, "gfx950");

    // Verify: Should return nullopt
    EXPECT_FALSE(result.has_value());
}

TEST(TestEngineEntryArchResolution, EmptyMapReturnsNullopt)
{
    // Setup: Empty role map
    const std::unordered_map<std::string, UhdConfig> roleMap;

    // Test: Request any arch
    auto result = EngineEntry::resolveUhd(roleMap, "gfx942");

    // Verify: Should return nullopt
    EXPECT_FALSE(result.has_value());
}

TEST(TestEngineEntryArchResolution, ExactMatchPrefersOverDefault)
{
    // Setup: Register both exact match and default with different configs
    std::unordered_map<std::string, UhdConfig> roleMap;
    roleMap["gfx950"]  = makeTestConfig("uhd_gfx950_tuned", "sqrt");
    roleMap["default"] = makeTestConfig("uhd_default_conservative", "identity");

    // Test: Request gfx950 (both exact and default exist)
    auto result = EngineEntry::resolveUhd(roleMap, "gfx950");

    // Verify: Should return exact match, NOT default
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->uhdId, "uhd_gfx950_tuned");
    EXPECT_EQ(result->scoreTransform, "sqrt");
    EXPECT_NE(result->uhdId, "uhd_default_conservative");
}

TEST(TestEngineEntryArchResolution, DefaultKeyResolvesToDefault)
{
    // Setup: Register default and gfx942
    std::unordered_map<std::string, UhdConfig> roleMap;
    roleMap["gfx942"]  = makeTestConfig("uhd_gfx942", "log1p");
    roleMap["default"] = makeTestConfig("uhd_default", "identity");

    // Test: Explicitly request "default" arch
    auto result = EngineEntry::resolveUhd(roleMap, "default");

    // Verify: Should return default config
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->uhdId, "uhd_default");
    EXPECT_EQ(result->scoreTransform, "identity");
}

TEST(TestEngineEntryArchResolution, MultipleArchsAllResolveCorrectly)
{
    // Setup: Register multiple archs with distinct configs
    std::unordered_map<std::string, UhdConfig> roleMap;
    roleMap["gfx942"]  = makeTestConfig("uhd_gfx942", "log1p");
    roleMap["gfx950"]  = makeTestConfig("uhd_gfx950", "sqrt");
    roleMap["gfx1100"] = makeTestConfig("uhd_gfx1100", "log");
    roleMap["default"] = makeTestConfig("uhd_default", "identity");

    // Test: Each arch resolves to its own config
    auto r942  = EngineEntry::resolveUhd(roleMap, "gfx942");
    auto r950  = EngineEntry::resolveUhd(roleMap, "gfx950");
    auto r1100 = EngineEntry::resolveUhd(roleMap, "gfx1100");

    ASSERT_TRUE(r942.has_value());
    EXPECT_EQ(r942->uhdId, "uhd_gfx942");
    EXPECT_EQ(r942->scoreTransform, "log1p");

    ASSERT_TRUE(r950.has_value());
    EXPECT_EQ(r950->uhdId, "uhd_gfx950");
    EXPECT_EQ(r950->scoreTransform, "sqrt");

    ASSERT_TRUE(r1100.has_value());
    EXPECT_EQ(r1100->uhdId, "uhd_gfx1100");
    EXPECT_EQ(r1100->scoreTransform, "log");

    // Test: Unknown arch falls back to default
    auto rUnknown = EngineEntry::resolveUhd(roleMap, "gfx9999");
    ASSERT_TRUE(rUnknown.has_value());
    EXPECT_EQ(rUnknown->uhdId, "uhd_default");
}

// ========== EngineEntry Helper Method Tests ==========

TEST(TestEngineEntryHelpers, ResolveSortKernelCatalogDelegatesToResolveUhd)
{
    // Setup: Create an EngineEntry with sortKernelCatalog configs
    EngineEntry entry;
    entry.engineId   = 1000;
    entry.engineName = "TestEngine";
    entry.sortKernelCatalog["gfx942"]  = makeTestConfig("uhd_sort_gfx942", "log1p");
    entry.sortKernelCatalog["default"] = makeTestConfig("uhd_sort_default", "identity");

    // Test: Exact match
    auto r942 = entry.resolveSortKernelCatalog("gfx942");
    ASSERT_TRUE(r942.has_value());
    EXPECT_EQ(r942->uhdId, "uhd_sort_gfx942");

    // Test: Fallback to default
    auto rUnknown = entry.resolveSortKernelCatalog("gfx1100");
    ASSERT_TRUE(rUnknown.has_value());
    EXPECT_EQ(rUnknown->uhdId, "uhd_sort_default");

    // Test: No match
    entry.sortKernelCatalog.clear();
    entry.sortKernelCatalog["gfx942"] = makeTestConfig("uhd_942_only");
    auto rNoMatch = entry.resolveSortKernelCatalog("gfx950");
    EXPECT_FALSE(rNoMatch.has_value());
}

TEST(TestEngineEntryHelpers, ResolvePredictEngineTflopsIndependent)
{
    // Setup: sortKernelCatalog and predictEngineTflops have different archs
    EngineEntry entry;
    entry.engineId   = 1000;
    entry.engineName = "TestEngine";

    entry.sortKernelCatalog["gfx942"] = makeTestConfig("uhd_sort_gfx942");
    entry.predictEngineTflops["gfx950"] = makeTestConfig("uhd_predict_gfx950");
    entry.sortKernelCatalog["default"] = makeTestConfig("uhd_sort_default");

    // Test: predictEngineTflops resolves independently
    auto predictResult = entry.resolvePredictEngineTflops("gfx950");
    ASSERT_TRUE(predictResult.has_value());
    EXPECT_EQ(predictResult->uhdId, "uhd_predict_gfx950");

    // Test: sortKernelCatalog still works for gfx942
    auto sortResult = entry.resolveSortKernelCatalog("gfx942");
    ASSERT_TRUE(sortResult.has_value());
    EXPECT_EQ(sortResult->uhdId, "uhd_sort_gfx942");

    // Test: predictEngineTflops falls back to default when gfx942 not registered
    auto predictFallback = entry.resolvePredictEngineTflops("gfx942");
    EXPECT_FALSE(predictFallback.has_value()); // No default in predictEngineTflops
}

TEST(TestEngineEntryHelpers, ResolvePredictApplicableKernelsFutureRole)
{
    // Setup: Register only in predictApplicableKernels (future role)
    EngineEntry entry;
    entry.engineId   = 1000;
    entry.engineName = "TestEngine";
    entry.predictApplicableKernels["gfx942"]  = makeTestConfig("uhd_applicable_gfx942");
    entry.predictApplicableKernels["default"] = makeTestConfig("uhd_applicable_default");

    // Test: Exact match
    auto result = entry.resolvePredictApplicableKernels("gfx942");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->uhdId, "uhd_applicable_gfx942");

    // Test: Fallback
    auto fallback = entry.resolvePredictApplicableKernels("gfx1100");
    ASSERT_TRUE(fallback.has_value());
    EXPECT_EQ(fallback->uhdId, "uhd_applicable_default");
}

} // namespace hipdnn_backend::heuristics::uhd
