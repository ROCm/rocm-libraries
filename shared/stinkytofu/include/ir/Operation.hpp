#pragma once

#include "StinkyBuilder.hpp"
#include "ir/asm/OptimizationPipeline.hpp"
#include "ir/asm/StinkyAsmIR.hpp"
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace stinkytofu
{
    // Forward declarations
    class StinkyIR;

    // ========================================================================
    // Base Configuration
    // ========================================================================

    /**
     * @brief Base class for operation configurations
     * Each operation type defines its own config subclass
     */
    class OperationConfig
    {
    public:
        virtual ~OperationConfig() = default;

        /**
         * @brief Generate cache key for this configuration
         * Return empty string if operation is not cacheable
         * Cache key should NOT include register allocations
         */
        virtual std::string getCacheKey() const = 0;

        /**
         * @brief Validate configuration parameters
         */
        virtual bool validate() const = 0;
    };

    // ========================================================================
    // Result
    // ========================================================================

    /**
     * @brief Result of building an operation
     */
    struct OperationResult
    {
        // Generated instructions
        std::vector<StinkyInstruction*> instructions;

        // Cache info
        bool fromCache = false;

        // Metadata
        std::string                        operationName;
        std::map<std::string, std::string> metadata;

        // Stats (for profiling/debugging)
        size_t instructionCountBeforeOpt = 0;
        size_t instructionCountAfterOpt  = 0;
        int    optimizationIterations    = 0;
    };

    // ========================================================================
    // Builder Base Class
    // ========================================================================

    /**
     * @brief Base class for operation builders
     */
    class OperationBuilder
    {
    protected:
        StinkyTofu& builder;
        StinkyIR&   ir;

    public:
        OperationBuilder(StinkyTofu& builder, StinkyIR& ir)
            : builder(builder)
            , ir(ir)
        {
        }
        virtual ~OperationBuilder() = default;

        /**
         * @brief Build operation (called by registry, not directly by user)
         * This generates the RAW instructions without optimization
         */
        virtual OperationResult buildRaw(const OperationConfig& config) = 0;

        /**
         * @brief Whether this operation type should be cached
         * Default: false (developer must opt-in)
         */
        virtual bool isCacheable() const
        {
            return false;
        }

        /**
         * @brief Whether this operation type should be optimized
         * Default: false (developer must opt-in)
         */
        virtual bool shouldOptimize() const
        {
            return false;
        }

        /**
         * @brief Optimization configuration for this operation type
         * Only called if shouldOptimize() returns true
         */
        virtual PipelineConfig getOptimizationConfig() const;

        // Accessors
        StinkyTofu& getBuilder()
        {
            return builder;
        }
        const StinkyTofu& getBuilder() const
        {
            return builder;
        }
        StinkyIR& getIR()
        {
            return ir;
        }
        const StinkyIR& getIR() const
        {
            return ir;
        }
    };

    // ========================================================================
    // Variant Builder (for operations with multiple implementations)
    // ========================================================================

    /**
     * @brief Dispatch table entry for operation variants
     */
    struct OperationVariant
    {
        std::string                                                               name;
        std::function<bool(const OperationConfig&)>                               matcher;
        std::function<OperationResult(OperationBuilder&, const OperationConfig&)> generator;
        uint32_t    priority = 0; // Higher priority variants checked first
        std::string description;
    };

    /**
     * @brief Builder that dispatches to variants based on config
     * Example: MFMA has variants for f32_16x16x4, f16_32x32x8, etc.
     */
    class VariantOperationBuilder : public OperationBuilder
    {
    private:
        std::vector<OperationVariant> variants;

    protected:
        void registerVariant(OperationVariant variant);

    public:
        using OperationBuilder::OperationBuilder;

        OperationResult buildRaw(const OperationConfig& config) override;
    };

    // ========================================================================
    // Registry with Integrated Opt+Cache
    // ========================================================================

    /**
     * @brief Central registry for all operations
     * Handles operation dispatch, caching, and optimization
     */
    class OperationRegistry
    {
    private:
        StinkyTofu& builder;
        StinkyIR&   ir;

        // Builder factories
        std::map<std::string,
                 std::function<std::unique_ptr<OperationBuilder>(StinkyTofu&, StinkyIR&)>>
            builderFactories;

        // Cache: (operation_name, cache_key) -> optimized instructions
        std::map<std::pair<std::string, std::string>, std::vector<StinkyInstruction*>> cache;

        // Global cache enable/disable
        bool cacheEnabled;

        // Stats
        struct CacheStats
        {
            size_t hits        = 0;
            size_t misses      = 0;
            size_t totalBuilds = 0;

            double hitRate() const
            {
                return totalBuilds > 0 ? (double)hits / totalBuilds : 0.0;
            }
        };
        std::map<std::string, CacheStats> stats;

        // Helper: optimize instructions
        std::vector<StinkyInstruction*>
            optimizeInstructions(const std::vector<StinkyInstruction*>& instructions,
                                 const PipelineConfig&                  config);

    public:
        OperationRegistry(StinkyTofu& builder, StinkyIR& ir, bool enableCache = true)
            : builder(builder)
            , ir(ir)
            , cacheEnabled(enableCache)
        {
        }

        /**
         * @brief Register an operation type
         */
        template <typename BuilderType>
        void registerOperation(const std::string& name)
        {
            builderFactories[name]
                = [](StinkyTofu& b, StinkyIR& ir) { return std::make_unique<BuilderType>(b, ir); };
        }

        /**
         * @brief Build operation with automatic opt+cache
         * This is the main entry point
         *
         * @pre config.validate() must return true (checked with assert)
         * @pre opName must be a registered operation (checked with assert)
         *
         * Error handling strategy:
         * - Validation errors: assert() - caller must validate config first
         * - Unknown operations: assert() - programmer error (operation not registered)
         * - Unsupported paths: STINKY_UNREACHABLE() - impossible code paths
         *
         * @param opName Name of the registered operation
         * @param config Configuration for the operation (must pass validate())
         * @return OperationResult with generated instructions, cache info, and stats
         */
        OperationResult build(const std::string& opName, const OperationConfig& config);

        /**
         * @brief Check if operation exists
         */
        bool hasOperation(const std::string& name) const;

        /**
         * @brief Clear cache for specific operation or all
         */
        void clearCache(const std::string& opName = "");

        /**
         * @brief Enable/disable caching globally
         */
        void setCacheEnabled(bool enabled)
        {
            cacheEnabled = enabled;
        }

        /**
         * @brief Get cache statistics
         */
        const CacheStats& getStats(const std::string& opName) const;

        /**
         * @brief Print cache statistics (for debugging)
         */
        void printStats() const;
    };

} // namespace stinkytofu
