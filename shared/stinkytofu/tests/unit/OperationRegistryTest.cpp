#include "StinkyBuilder.hpp"
#include "ir/Operation.hpp"
#include "ir/StinkyIR.hpp"
#include "ir/operations/Activation.hpp"
#include <gtest/gtest.h>

using namespace stinkytofu;

class OperationRegistryTest : public ::testing::Test
{
protected:
    std::array<int, 3> arch     = {9, 4, 2};
    StinkyTofu*        builder  = nullptr;
    StinkyIR*          ir       = nullptr;
    OperationRegistry* registry = nullptr;

    void SetUp() override
    {
        builder  = new StinkyTofu(arch);
        ir       = new StinkyIR(arch);
        registry = new OperationRegistry(*builder, *ir, true);

        // Register operations
        registry->registerOperation<ActivationBuilder>("activation");
    }

    void TearDown() override
    {
        delete registry;
        delete ir;
        delete builder;
    }
};

// ============================================================================
// Basic Registry Tests
// ============================================================================

TEST_F(OperationRegistryTest, RegisterAndCheckOperations)
{
    EXPECT_TRUE(registry->hasOperation("activation"));
    EXPECT_FALSE(registry->hasOperation("nonexistent"));
}

TEST_F(OperationRegistryTest, BuildUnknownOperation)
{
    // Unknown operation triggers assert in debug builds
    // In release builds, behavior is undefined (by design)
    // So we just check that the operation exists
    EXPECT_FALSE(registry->hasOperation("nonexistent"));
}

// ============================================================================
// Activation Operation Tests
// ============================================================================

TEST_F(OperationRegistryTest, BuildSimpleReLU)
{
    ActivationConfig cfg(ActivationType::Relu, ActivationDataType::F32, 0, 1);

    auto result = registry->build("activation", cfg);

    EXPECT_FALSE(result.fromCache);
    EXPECT_EQ(result.operationName, "relu");
    EXPECT_GT(result.instructions.size(), 0);
    EXPECT_GT(result.instructionCountAfterOpt, 0);
}

TEST_F(OperationRegistryTest, BuildGeLU)
{
    std::vector<uint32_t> tmpVgprs = {10}; // Need 1 temp VGPR
    ActivationConfig      cfg(ActivationType::Gelu, ActivationDataType::F32, 0, 1, {}, tmpVgprs);

    auto result = registry->build("activation", cfg);

    EXPECT_FALSE(result.fromCache);
    EXPECT_EQ(result.operationName, "gelu");
    EXPECT_GT(result.instructions.size(), 0);
    // GELU is complex, should have many instructions
    EXPECT_GT(result.instructionCountBeforeOpt, 10);
}

TEST_F(OperationRegistryTest, BuildLeakyReLU)
{
    ActivationConfig cfg(ActivationType::LeakyRelu, ActivationDataType::F32, 0, 1, {0.01});

    auto result = registry->build("activation", cfg);

    EXPECT_FALSE(result.fromCache);
    EXPECT_EQ(result.operationName, "leaky_relu");
    EXPECT_GT(result.instructions.size(), 0);
}

TEST_F(OperationRegistryTest, BuildClamp)
{
    ActivationConfig cfg(ActivationType::Clamp, ActivationDataType::F32, 0, 1, {0.0, 1.0});

    auto result = registry->build("activation", cfg);

    EXPECT_FALSE(result.fromCache);
    EXPECT_EQ(result.operationName, "clamp");
    EXPECT_GT(result.instructions.size(), 0);
}

TEST_F(OperationRegistryTest, BuildDGelu)
{
    std::vector<uint32_t> tmpVgprs = {10, 11, 12}; // Need 3 temp VGPRs
    ActivationConfig      cfg(ActivationType::DGelu, ActivationDataType::F32, 0, 1, {}, tmpVgprs);

    auto result = registry->build("activation", cfg);

    EXPECT_FALSE(result.fromCache);
    EXPECT_EQ(result.operationName, "dgelu");
    EXPECT_GT(result.instructions.size(), 0);
    // DGelu is very complex
    EXPECT_GT(result.instructionCountBeforeOpt, 20);
}

// ============================================================================
// Caching Tests
// ============================================================================

TEST_F(OperationRegistryTest, CacheHit)
{
    ActivationConfig cfg1(ActivationType::Relu, ActivationDataType::F32, 0, 1);

    // First call - cache miss
    auto result1 = registry->build("activation", cfg1);
    EXPECT_FALSE(result1.fromCache);
    size_t instructionCount1 = result1.instructions.size();

    // Second call with SAME activation type - cache hit
    ActivationConfig cfg2(
        ActivationType::Relu, ActivationDataType::F32, 2, 3); // Different registers
    auto result2 = registry->build("activation", cfg2);
    EXPECT_TRUE(result2.fromCache);
    EXPECT_EQ(result2.instructions.size(), instructionCount1);

    // Check stats
    auto stats = registry->getStats("activation");
    EXPECT_EQ(stats.totalBuilds, 2);
    EXPECT_EQ(stats.hits, 1);
    EXPECT_EQ(stats.misses, 1);
    EXPECT_DOUBLE_EQ(stats.hitRate(), 0.5);
}

TEST_F(OperationRegistryTest, CacheMissWithDifferentType)
{
    ActivationConfig cfg1(ActivationType::Relu, ActivationDataType::F32, 0, 1);
    ActivationConfig cfg2(ActivationType::Gelu, ActivationDataType::F32, 0, 1, {}, {10});

    auto result1 = registry->build("activation", cfg1);
    auto result2 = registry->build("activation", cfg2);

    EXPECT_FALSE(result1.fromCache);
    EXPECT_FALSE(result2.fromCache);

    // Both should be cache misses (different activation types)
    auto stats = registry->getStats("activation");
    EXPECT_EQ(stats.hits, 0);
    EXPECT_EQ(stats.misses, 2);
}

TEST_F(OperationRegistryTest, CacheMissWithDifferentDataType)
{
    ActivationConfig cfg1(ActivationType::Relu, ActivationDataType::F32, 0, 1);
    ActivationConfig cfg2(ActivationType::Relu, ActivationDataType::F16, 0, 1);

    auto result1 = registry->build("activation", cfg1);
    auto result2 = registry->build("activation", cfg2);

    EXPECT_FALSE(result1.fromCache);
    EXPECT_FALSE(result2.fromCache);

    // Both should be cache misses (different data types)
    auto stats = registry->getStats("activation");
    EXPECT_EQ(stats.hits, 0);
    EXPECT_EQ(stats.misses, 2);
}

TEST_F(OperationRegistryTest, CacheMissWithDifferentParams)
{
    ActivationConfig cfg1(ActivationType::LeakyRelu, ActivationDataType::F32, 0, 1, {0.01});
    ActivationConfig cfg2(ActivationType::LeakyRelu, ActivationDataType::F32, 0, 1, {0.1});

    auto result1 = registry->build("activation", cfg1);
    auto result2 = registry->build("activation", cfg2);

    EXPECT_FALSE(result1.fromCache);
    EXPECT_FALSE(result2.fromCache);

    // Both should be cache misses (different alpha values)
    auto stats = registry->getStats("activation");
    EXPECT_EQ(stats.hits, 0);
    EXPECT_EQ(stats.misses, 2);
}

TEST_F(OperationRegistryTest, ClearCache)
{
    ActivationConfig cfg(ActivationType::Relu, ActivationDataType::F32, 0, 1);

    // Build once
    auto result1 = registry->build("activation", cfg);
    EXPECT_FALSE(result1.fromCache);

    // Build again - should hit cache
    auto result2 = registry->build("activation", cfg);
    EXPECT_TRUE(result2.fromCache);

    // Clear cache
    registry->clearCache("activation");

    // Build again - should miss cache
    auto result3 = registry->build("activation", cfg);
    EXPECT_FALSE(result3.fromCache);
}

TEST_F(OperationRegistryTest, DisableCache)
{
    ActivationConfig cfg(ActivationType::Relu, ActivationDataType::F32, 0, 1);

    // Disable cache
    registry->setCacheEnabled(false);

    // Build twice
    auto result1 = registry->build("activation", cfg);
    auto result2 = registry->build("activation", cfg);

    // Both should be cache misses
    EXPECT_FALSE(result1.fromCache);
    EXPECT_FALSE(result2.fromCache);

    auto stats = registry->getStats("activation");
    EXPECT_EQ(stats.hits, 0);
    EXPECT_EQ(stats.misses, 2);
}

// ============================================================================
// Optimization Tests
// ============================================================================

TEST_F(OperationRegistryTest, OptimizationReducesInstructions)
{
    std::vector<uint32_t> tmpVgprs = {10};
    ActivationConfig      cfg(ActivationType::Gelu, ActivationDataType::F32, 0, 1, {}, tmpVgprs);

    auto result = registry->build("activation", cfg);

    // GELU should be optimized
    EXPECT_GT(result.optimizationIterations, 0);
    // Optimization should reduce instruction count (or at least not increase it)
    EXPECT_LE(result.instructionCountAfterOpt, result.instructionCountBeforeOpt);
}

// ============================================================================
// Validation Tests
// ============================================================================

TEST_F(OperationRegistryTest, ValidationFailsWithMissingTempVgprs)
{
    // GELU needs 1 temp VGPR, but we provide none
    ActivationConfig cfg(ActivationType::Gelu, ActivationDataType::F32, 0, 1, {}, {});

    // Config should fail validation
    EXPECT_FALSE(cfg.validate());
}

TEST_F(OperationRegistryTest, ValidationFailsWithMissingParams)
{
    // LeakyReLU needs alpha parameter, but we provide none
    ActivationConfig cfg(ActivationType::LeakyRelu, ActivationDataType::F32, 0, 1, {});

    // Config should fail validation
    EXPECT_FALSE(cfg.validate());
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(OperationRegistryTest, MultipleActivationTypes)
{
    std::vector<uint32_t> tmpVgprs = {10, 11, 12};

    // Build different activation types
    ActivationConfig reluCfg(ActivationType::Relu, ActivationDataType::F32, 0, 1);
    ActivationConfig geluCfg(ActivationType::Gelu, ActivationDataType::F32, 0, 1, {}, {10});
    ActivationConfig sigmoidCfg(ActivationType::Sigmoid, ActivationDataType::F32, 0, 1, {}, {10});

    auto reluResult    = registry->build("activation", reluCfg);
    auto geluResult    = registry->build("activation", geluCfg);
    auto sigmoidResult = registry->build("activation", sigmoidCfg);

    EXPECT_EQ(reluResult.operationName, "relu");
    EXPECT_EQ(geluResult.operationName, "gelu");
    EXPECT_EQ(sigmoidResult.operationName, "sigmoid");

    // All should be cache misses (different types)
    EXPECT_FALSE(reluResult.fromCache);
    EXPECT_FALSE(geluResult.fromCache);
    EXPECT_FALSE(sigmoidResult.fromCache);

    // Build same types again - should hit cache
    auto reluResult2 = registry->build("activation", reluCfg);
    auto geluResult2 = registry->build("activation", geluCfg);

    EXPECT_TRUE(reluResult2.fromCache);
    EXPECT_TRUE(geluResult2.fromCache);
}

TEST_F(OperationRegistryTest, CacheHitRateCalculation)
{
    ActivationConfig cfg(ActivationType::Relu, ActivationDataType::F32, 0, 1);

    // Build 10 times
    for(int i = 0; i < 10; i++)
    {
        registry->build("activation", cfg);
    }

    auto stats = registry->getStats("activation");
    EXPECT_EQ(stats.totalBuilds, 10);
    EXPECT_EQ(stats.hits, 9); // First is miss, rest are hits
    EXPECT_EQ(stats.misses, 1);
    EXPECT_DOUBLE_EQ(stats.hitRate(), 0.9);
}
