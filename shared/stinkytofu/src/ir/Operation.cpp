#include "ir/Operation.hpp"
#include "ErrorHandling.hpp"
#include "ir/StinkyIR.hpp"
#include "ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu.hpp"
#include <cassert>
#include <sstream>

//==============================================================================
// Error Handling Strategy
//==============================================================================
//
// 1. Validation Errors (config.validate() fails):
//    - Use assert() - caller's responsibility to validate before calling build()
//    - Example: Missing required parameters, insufficient temp VGPRs
//
// 2. Unknown Operations (opName not registered):
//    - Use assert() - programmer error (forgot to register operation)
//
// 3. Unsupported Code Paths (e.g., unsupported activation type):
//    - Use STINKY_UNREACHABLE() - should never happen if validate() works correctly
//
// This approach has zero overhead in release builds while catching errors in debug.
//
//==============================================================================

namespace stinkytofu
{

    // ========================================================================
    // OperationBuilder
    // ========================================================================

    PipelineConfig OperationBuilder::getOptimizationConfig() const
    {
        // Default: no optimization (most operations don't need it)
        // Override this in specific builders (like ActivationBuilder) if needed
        PipelineConfig cfg;
        cfg.enablePeephole         = false;
        cfg.enableDCE              = false;
        cfg.enableDuplicateElim    = false;
        cfg.optimizationIterations = 0;

        // Set architecture from StinkyIR (required by passes that need arch info)
        cfg.gemmTileConfig       = std::make_unique<GemmTileConfig>();
        cfg.gemmTileConfig->arch = ir.getArch();

        return cfg;
    }

    // ========================================================================
    // VariantOperationBuilder
    // ========================================================================

    void VariantOperationBuilder::registerVariant(OperationVariant variant)
    {
        variants.push_back(std::move(variant));
        // Sort by priority (higher priority first)
        std::sort(variants.begin(), variants.end(), [](const auto& a, const auto& b) {
            return a.priority > b.priority;
        });
    }

    OperationResult VariantOperationBuilder::buildRaw(const OperationConfig& config)
    {
        // Find matching variant
        for(const auto& variant : variants)
        {
            if(variant.matcher(config))
            {
                return variant.generator(*this, config);
            }
        }

        // No matching variant found - this should never happen if variants are registered correctly
        STINKY_UNREACHABLE("No matching operation variant found for configuration");
    }

    // ========================================================================
    // OperationRegistry
    // ========================================================================

    OperationResult OperationRegistry::build(const std::string&     opName,
                                             const OperationConfig& config)
    {
        stats[opName].totalBuilds++;

        // Validate config
        assert(config.validate() && "Invalid operation configuration");

        // Get builder factory
        auto it = builderFactories.find(opName);
        assert(it != builderFactories.end() && "Unknown operation");

        auto builderInstance = it->second(builder, ir);

        // Check cache if enabled
        if(cacheEnabled && builderInstance->isCacheable())
        {
            std::string cacheKey = config.getCacheKey();
            if(!cacheKey.empty())
            {
                auto cacheIt = cache.find({opName, cacheKey});
                if(cacheIt != cache.end())
                {
                    // Cache hit!
                    stats[opName].hits++;

                    OperationResult result;
                    result.instructions             = cacheIt->second;
                    result.fromCache                = true;
                    result.operationName            = opName;
                    result.instructionCountAfterOpt = result.instructions.size();
                    return result;
                }
            }
        }

        // Cache miss - build raw
        stats[opName].misses++;
        auto result                      = builderInstance->buildRaw(config);
        result.instructionCountBeforeOpt = result.instructions.size();

        // Optimize if requested
        if(builderInstance->shouldOptimize())
        {
            auto optConfig                  = builderInstance->getOptimizationConfig();
            result.instructions             = optimizeInstructions(result.instructions, optConfig);
            result.instructionCountAfterOpt = result.instructions.size();
            result.optimizationIterations   = optConfig.optimizationIterations;
        }
        else
        {
            result.instructionCountAfterOpt = result.instructions.size();
        }

        // Store in cache if cacheable
        if(cacheEnabled && builderInstance->isCacheable())
        {
            std::string cacheKey = config.getCacheKey();
            if(!cacheKey.empty())
            {
                cache[{opName, cacheKey}] = result.instructions;
            }
        }

        return result;
    }

    bool OperationRegistry::hasOperation(const std::string& name) const
    {
        return builderFactories.find(name) != builderFactories.end();
    }

    void OperationRegistry::clearCache(const std::string& opName)
    {
        if(opName.empty())
        {
            cache.clear();
        }
        else
        {
            auto it = cache.begin();
            while(it != cache.end())
            {
                if(it->first.first == opName)
                {
                    it = cache.erase(it);
                }
                else
                {
                    ++it;
                }
            }
        }
    }

    const OperationRegistry::CacheStats&
        OperationRegistry::getStats(const std::string& opName) const
    {
        static CacheStats empty;
        auto              it = stats.find(opName);
        return it != stats.end() ? it->second : empty;
    }

    void OperationRegistry::printStats() const
    {
        printf("=== Operation Registry Statistics ===\n");
        for(const auto& [name, stat] : stats)
        {
            printf("%s: %zu builds, %zu hits, %zu misses (%.1f%% hit rate)\n",
                   name.c_str(),
                   stat.totalBuilds,
                   stat.hits,
                   stat.misses,
                   stat.hitRate() * 100.0);
        }
    }

    std::vector<StinkyInstruction*>
        OperationRegistry::optimizeInstructions(const std::vector<StinkyInstruction*>& instructions,
                                                const PipelineConfig&                  config)
    {
        // Create a temporary function for optimization
        Function func;
        auto*    entryBlock = func.createBasicBlock("entry");
        func.setEntryBlock(entryBlock);

        // Add instructions to the block
        IRList& irList = entryBlock->getIR();
        for(auto* inst : instructions)
        {
            irList.push_back(inst);
        }

        // Run optimization pipeline
        OptimizationPipeline::run(func, config);

        // Extract optimized instructions
        std::vector<StinkyInstruction*> result;
        for(auto& irBase : entryBlock->getIR())
        {
            result.push_back(static_cast<StinkyInstruction*>(&irBase));
        }

        // Clean up (remove instructions from block but don't delete them - owned by StinkyTofu)
        irList.clear();

        return result;
    }

} // namespace stinkytofu
