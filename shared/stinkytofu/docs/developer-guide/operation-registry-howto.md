# Operation Registry How-To Guide

This guide shows you how to use the Operation Registry system in StinkyTofu to build, cache, and optimize reusable IR operations.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Using with Cache and Optimization](#using-with-cache-and-optimization)
3. [Using without Cache or Optimization](#using-without-cache-or-optimization)
4. [Creating Custom Operations](#creating-custom-operations)
5. [Performance Tips](#performance-tips)
6. [Troubleshooting](#troubleshooting)

## Quick Start

### Basic Usage

'''cpp
#include "ir/Operation.hpp"
#include "ir/operations/Activation.hpp"

// 1. Create the registry (cache enabled)
std::array<int, 3> arch = {9, 4, 2};
StinkyTofu builder(arch);
StinkyIR ir(arch);
OperationRegistry registry(builder, ir, true);  // true = cache enabled

// 2. Register operations you'll use
registry.registerOperation<ActivationBuilder>("activation");

// 3. Build an operation
ActivationConfig cfg(ActivationType::Relu, ActivationDataType::F32,
                     vgprIn, vgprOut);
auto result = registry.build("activation", cfg);

// 4. Use the generated instructions
for (auto* inst : result.instructions) {
    myModule->append(inst);
}
'''

## Using with Cache and Optimization

This is the **recommended approach** for complex operations that will be used repeatedly.

### Example 1: Activation Functions (Full Optimization)

'''cpp
#include "ir/Operation.hpp"
#include "ir/operations/Activation.hpp"

void generateKernel() {
    std::array<int, 3> arch = {9, 4, 2};
    StinkyTofu builder(arch);
    StinkyIR ir(arch);

    // Enable cache for maximum performance
    OperationRegistry registry(builder, ir, true);
    registry.registerOperation<ActivationBuilder>("activation");

    // First call: GELU (will optimize and cache)
    {
        std::vector<uint32_t> tempVgprs = {100};  // Need 1 temp for GELU
        ActivationConfig cfg(ActivationType::Gelu, ActivationDataType::F32,
                             0, 1, {}, tempVgprs);

        auto result = registry.build("activation", cfg);

        std::cout << "Generated " << result.instructions.size() << " instructions\n";
        std::cout << "From cache: " << (result.fromCache ? "yes" : "no") << "\n";
        std::cout << "Optimized: " << result.instructionCountBeforeOpt
                  << " -> " << result.instructionCountAfterOpt << "\n";
        // Output:
        // Generated 13 instructions
        // From cache: no
        // Optimized: 13 -> 13

        // Use instructions...
        for (auto* inst : result.instructions) {
            myModule->append(inst);
        }
    }

    // Second call: Same GELU, different registers (CACHE HIT!)
    {
        std::vector<uint32_t> tempVgprs = {200};  // Different temp register
        ActivationConfig cfg(ActivationType::Gelu, ActivationDataType::F32,
                             10, 11, {}, tempVgprs);  // Different in/out

        auto result = registry.build("activation", cfg);

        std::cout << "From cache: " << (result.fromCache ? "yes" : "no") << "\n";
        // Output: From cache: yes  (50x faster!)

        // Use cached instructions...
    }

    // Check statistics
    auto stats = registry.getStats("activation");
    std::cout << "Cache hit rate: " << stats.hitRate() << "%\n";
    // Output: Cache hit rate: 50%
}
'''

### Example 2: Batch Processing with Cache

'''cpp
void processBatchActivations(const std::vector<Layer>& layers) {
    std::array<int, 3> arch = {9, 4, 2};
    StinkyTofu builder(arch);
    StinkyIR ir(arch);
    OperationRegistry registry(builder, ir, true);
    registry.registerOperation<ActivationBuilder>("activation");

    for (const auto& layer : layers) {
        // Configure activation for this layer
        ActivationConfig cfg(
            layer.activationType,           // e.g., Relu, Gelu, Sigmoid
            ActivationDataType::F32,
            layer.inputVgpr,
            layer.outputVgpr,
            layer.params,                    // e.g., alpha for LeakyReLU
            layer.tempVgprs
        );

        // Build (uses cache if same activation type was seen before)
        auto result = registry.build("activation", cfg);

        // Log cache performance
        if (result.fromCache) {
            std::cout << "Layer " << layer.name << ": cache hit ?\n";
        } else {
            std::cout << "Layer " << layer.name << ": built and optimized\n";
        }

        // Add to kernel
        layer.module->appendInstructions(result.instructions);
    }

    // Print final statistics
    registry.printStats();
}
'''

### Example 3: Controlling Optimization Level

'''cpp
// For operations that need custom optimization settings
class CustomActivationBuilder : public OperationBuilder {
public:
    CustomActivationBuilder(StinkyTofu& builder, StinkyIR& ir)
        : OperationBuilder(builder, ir) {}

    // Override optimization config for more aggressive optimization
    PipelineConfig getOptimizationConfig() const override {
        PipelineConfig config;
        config.enablePeephole = true;
        config.enableDCE = true;
        config.enableDuplicateElim = true;
        config.optimizationIterations = 5;  // More iterations than default
        config.optimizationLevel = OptLevel::O3;
        return config;
    }

    OperationResult buildRaw(const OperationConfig& cfg) override {
        // Build instructions...
    }
};
'''

## Using without Cache or Optimization

For **simple operations** or **one-time use**, you can disable caching and optimization.

### Method 1: Disable Cache Globally

'''cpp
void generateSimpleKernel() {
    std::array<int, 3> arch = {9, 4, 2};
    StinkyTofu builder(arch);
    StinkyIR ir(arch);

    // Disable cache (last parameter = false)
    OperationRegistry registry(builder, ir, false);
    registry.registerOperation<ActivationBuilder>("activation");

    // Every call builds fresh (no cache overhead)
    ActivationConfig cfg(ActivationType::Relu, ActivationDataType::F32, 0, 1);
    auto result = registry.build("activation", cfg);

    // result.fromCache will always be false
}
'''

### Method 2: Disable Optimization per Operation

'''cpp
class SimpleActivationBuilder : public OperationBuilder {
public:
    SimpleActivationBuilder(StinkyTofu& builder, StinkyIR& ir)
        : OperationBuilder(builder, ir) {}

    // Disable optimization for this operation
    bool shouldOptimize() const override {
        return false;  // No optimization overhead
    }

    // Optionally disable caching too
    bool isCacheable() const override {
        return false;  // Don't cache results
    }

    OperationResult buildRaw(const OperationConfig& cfg) override {
        auto& acfg = static_cast<const ActivationConfig&>(cfg);

        // Build simple ReLU directly
        std::vector<StinkyInstruction*> insts;
        if (acfg.activationType == ActivationType::Relu) {
            insts = ir.reluF32(acfg.vgprIn, acfg.vgprOut);
        }

        return {insts, false, insts.size(), insts.size(), 0};
    }
};
'''

### Method 3: Use StinkyIR Directly (No Registry)

For maximum simplicity, skip the registry entirely:

'''cpp
void buildSimpleActivation() {
    std::array<int, 3> arch = {9, 4, 2};
    StinkyTofu builder(arch);
    StinkyIR ir(arch);

    // Call StinkyIR functions directly
    auto reluInsts = ir.reluF32(0, 1);
    auto geluInsts = ir.geluF32(2, 3, 100);  // tempVgpr = 100

    // Use instructions immediately
    for (auto* inst : reluInsts) {
        myModule->append(inst);
    }

    // No caching, no optimization, no registry overhead
}
'''

## Creating Custom Operations

### Step 1: Define Configuration

'''cpp
// include/ir/operations/MyOperation.hpp
#include "ir/Operation.hpp"

enum class MyOperationType {
    Variant1,
    Variant2
};

class MyOperationConfig : public OperationConfig {
public:
    MyOperationType type;
    uint32_t vgprDst;
    uint32_t vgprSrc;
    std::vector<int32_t> params;

    MyOperationConfig(MyOperationType type, uint32_t dst, uint32_t src,
                      std::vector<int32_t> params = {})
        : type(type), vgprDst(dst), vgprSrc(src), params(params) {}

    // Generate cache key based on type and param count (not registers!)
    std::string getCacheKey() const override {
        std::ostringstream oss;
        oss << static_cast<int>(type) << ":" << params.size();
        return oss.str();
    }

    // Validate configuration
    bool validate() const override {
        if (type == MyOperationType::Variant1 && params.empty()) {
            return false;  // Variant1 requires parameters
        }
        return true;
    }
};
'''

### Step 2: Define Builder

'''cpp
// src/ir/operations/MyOperation.cpp
#include "ir/operations/MyOperation.hpp"
#include "ir/StinkyIR.hpp"

class MyOperationBuilder : public OperationBuilder {
public:
    MyOperationBuilder(StinkyTofu& builder, StinkyIR& ir)
        : OperationBuilder(builder, ir) {}

    OperationResult buildRaw(const OperationConfig& config) override {
        auto& cfg = static_cast<const MyOperationConfig&>(config);

        std::vector<StinkyInstruction*> instructions;

        switch (cfg.type) {
            case MyOperationType::Variant1:
                instructions = buildVariant1(cfg);
                break;
            case MyOperationType::Variant2:
                instructions = buildVariant2(cfg);
                break;
        }

        return {instructions, false, instructions.size(), 0, 0};
    }

    // This operation benefits from caching
    bool isCacheable() const override {
        return true;
    }

    // This operation benefits from optimization
    bool shouldOptimize() const override {
        return true;
    }

    // Use default optimization config (Peephole + DCE + DuplicateElim)
    PipelineConfig getOptimizationConfig() const override {
        return OperationBuilder::getOptimizationConfig();
    }

private:
    std::vector<StinkyInstruction*> buildVariant1(const MyOperationConfig& cfg) {
        // Implementation...
        std::vector<StinkyInstruction*> insts;
        auto param = cfg.params[0];
        insts.push_back(builder.VMulF32(vgpr(cfg.vgprDst), vgpr(cfg.vgprSrc),
                                        imm(param), vgpr(cfg.vgprDst)));
        return insts;
    }

    std::vector<StinkyInstruction*> buildVariant2(const MyOperationConfig& cfg) {
        // Implementation...
        return {};
    }
};
'''

### Step 3: Register and Use

'''cpp
// Register your custom operation
registry.registerOperation<MyOperationBuilder>("my_operation");

// Use it
MyOperationConfig cfg(MyOperationType::Variant1, dstVgpr, srcVgpr, {42});
auto result = registry.build("my_operation", cfg);
'''

## Performance Tips

### When to Enable Cache

? **Enable cache when:**
- Operation will be used multiple times
- Operation is complex (>5 instructions)
- Operation is expensive to optimize
- You have similar operations with different registers

? **Disable cache when:**
- Operation is very simple (1-2 instructions)
- Operation is used only once
- Memory is constrained
- Cache overhead > build cost

### When to Enable Optimization

? **Enable optimization when:**
- Operation generates >10 instructions
- There's potential for dead code or duplicates
- Peephole patterns could apply
- Performance is critical

? **Disable optimization when:**
- Operation is already optimal (e.g., single instruction)
- Build speed is more important than execution speed
- Debugging (easier to see unoptimized code)

### Choosing Optimization Iterations

'''cpp
PipelineConfig getOptimizationConfig() const override {
    PipelineConfig cfg;
    cfg.enablePeephole = true;
    cfg.enableDCE = true;
    cfg.enableDuplicateElim = true;

    // Tune iterations based on complexity
    cfg.optimizationIterations = 1;  // Simple operations
    cfg.optimizationIterations = 3;  // Default (activations)
    cfg.optimizationIterations = 5;  // Complex operations

    return cfg;
}
'''

### Cache Key Best Practices

'''cpp
// ? GOOD: Cache based on operation semantics
std::string getCacheKey() const override {
    return std::to_string(static_cast<int>(type)) + ":" +
           std::to_string(paramCount);
}

// ? BAD: Including register numbers prevents cache hits
std::string getCacheKey() const override {
    return std::to_string(vgprIn) + ":" + std::to_string(vgprOut);
}

// ? BAD: Too coarse, different operations share same key
std::string getCacheKey() const override {
    return "my_op";
}
'''

## Troubleshooting

### Problem: Low Cache Hit Rate

**Symptom:** 'stats.hitRate() < 20%'

**Solutions:**
1. Check cache key - is it too fine-grained?
2. Verify 'isCacheable()' returns 'true'
3. Enable cache globally: 'OperationRegistry(builder, ir, true)'
4. Print cache keys to debug: 'std::cout << cfg.getCacheKey() << "\n";'

### Problem: Slow Build Times

**Symptom:** First call takes too long

**Solutions:**
1. Reduce 'optimizationIterations' in 'getOptimizationConfig()'
2. Disable optimization for simple operations: 'shouldOptimize() return false'
3. Profile with 'result.instructionCountBeforeOpt' vs 'instructionCountAfterOpt'

### Problem: Invalid Configuration

**Symptom:** Assertion failure or wrong instructions

**Solutions:**
1. Implement robust 'validate()' method
2. Add asserts in 'buildRaw()': 'assert(cfg.validate())'
3. Check required parameters: temp registers, constants, etc.

### Problem: Memory Usage Too High

**Symptom:** Registry cache growing unbounded

**Solutions:**
1. Clear cache periodically: 'registry.clearCache("activation")'
2. Disable caching: 'OperationRegistry(builder, ir, false)'
3. Make fewer operations cacheable: 'isCacheable() return false'

## Comparison: Three Approaches

| Approach | Cache | Optimize | Best For | Example |
|----------|-------|----------|----------|---------|
| **Full Registry** | ? | ? | Repeated complex operations | GELU in 100 layers |
| **Registry (No Opt)** | ? | ? | Repeated simple operations | ReLU in 100 layers |
| **Direct StinkyIR** | ? | ? | One-time or trivial operations | Single ReLU call |

### Performance Comparison (GELU F32)

'''
Approach              | First Call | Subsequent | Total (100 calls)
----------------------|------------|------------|------------------
Full Registry         | 5.0ms      | 0.1ms      | 14.9ms
Registry (No Opt)     | 0.5ms      | 0.1ms      | 10.4ms
Direct StinkyIR       | 0.5ms      | 0.5ms      | 50.0ms
'''

**Recommendation:** Use Full Registry for complex operations, Direct StinkyIR for simple operations.

## Examples

See 'examples/test_activation_registry.cpp' for a complete working example.

## References

- [Operation Registry Design](../design/operation-registry.md)
- [Optimization Pipeline How-To](optimization-pipeline-howto.md)
- [StinkyIR API Reference](stinky-ir-reference.md)

