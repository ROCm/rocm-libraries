#pragma once

#include "ir/Operation.hpp"
#include <sstream>
#include <vector>

namespace stinkytofu
{

    // ========================================================================
    // Activation Types
    // ========================================================================

    enum class ActivationType
    {
        Relu,
        LeakyRelu,
        Gelu,
        Sigmoid,
        Tanh,
        Abs,
        Clamp,
        Silu,
        Swish,
        ClippedRelu,
        DGelu
    };

    enum class ActivationDataType
    {
        F16,
        F32,
        F64,
        I32
    };

    // ========================================================================
    // Activation Configuration
    // ========================================================================

    /**
     * @brief Configuration for activation operations
     */
    class ActivationConfig : public OperationConfig
    {
    public:
        ActivationType        type;
        ActivationDataType    dtype;
        uint32_t              vgprIn;
        uint32_t              vgprOut;
        std::vector<double>   params; // alpha, beta, etc.
        std::vector<uint32_t> tmpVgprs; // Temporary VGPRs for complex activations

        ActivationConfig(ActivationType               type,
                         ActivationDataType           dtype,
                         uint32_t                     vgprIn,
                         uint32_t                     vgprOut,
                         const std::vector<double>&   params   = {},
                         const std::vector<uint32_t>& tmpVgprs = {})
            : type(type)
            , dtype(dtype)
            , vgprIn(vgprIn)
            , vgprOut(vgprOut)
            , params(params)
            , tmpVgprs(tmpVgprs)
        {
        }

        std::string getCacheKey() const override;
        bool        validate() const override;

        // Helper to get required temp VGPRs for each activation type
        static size_t getRequiredTempVgprs(ActivationType type);
    };

    // ========================================================================
    // Activation Builder
    // ========================================================================

    /**
     * @brief Builder for activation operations
     * Cacheable + Optimized (matches Activation.py behavior)
     */
    class ActivationBuilder : public OperationBuilder
    {
    public:
        using OperationBuilder::OperationBuilder;

        // Opt-in to caching (like Activation.py)
        bool isCacheable() const override
        {
            return true;
        }

        // Opt-in to optimization (like Activation.py)
        bool shouldOptimize() const override
        {
            return true;
        }

        // Custom optimization config for activations
        PipelineConfig getOptimizationConfig() const override;

        OperationResult buildRaw(const OperationConfig& config) override;

    private:
        // Helper to get activation name string
        static std::string getActivationName(ActivationType type);
    };

} // namespace stinkytofu
