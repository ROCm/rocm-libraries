#include "ir/operations/Activation.hpp"
#include "ErrorHandling.hpp"
#include "StinkyBuilder.hpp"
#include "ir/StinkyIR.hpp"
#include "stinkytofu.hpp"
#include <cassert>
#include <sstream>

namespace stinkytofu
{
    namespace
    {
        // Helper functions to create immediate StinkyRegisters
        inline StinkyRegister imm(int32_t value)
        {
            return StinkyRegister(value);
        }

        inline StinkyRegister imm(double value)
        {
            return StinkyRegister(value);
        }
    } // anonymous namespace

    // ========================================================================
    // ActivationConfig
    // ========================================================================

    std::string ActivationConfig::getCacheKey() const
    {
        // Cache key based on activation type and data type only
        // NOT registers, so we can reuse across different register allocations
        std::stringstream ss;
        ss << static_cast<int>(type) << "_" << static_cast<int>(dtype);

        // Include parameters in cache key (e.g., alpha for leaky relu)
        for(auto p : params)
        {
            ss << "_" << p;
        }

        return ss.str();
    }

    bool ActivationConfig::validate() const
    {
        // Check that we have enough temp VGPRs for this activation
        size_t required = getRequiredTempVgprs(type);
        if(tmpVgprs.size() < required)
        {
            return false;
        }

        // Validate parameters
        switch(type)
        {
        case ActivationType::LeakyRelu:
            return params.size() >= 1; // Need alpha

        case ActivationType::Clamp:
        case ActivationType::ClippedRelu:
            return params.size() >= 2; // Need alpha and beta

        case ActivationType::Swish:
            return params.size() >= 1; // Need beta

        default:
            return true;
        }
    }

    size_t ActivationConfig::getRequiredTempVgprs(ActivationType type)
    {
        switch(type)
        {
        case ActivationType::Relu:
        case ActivationType::Abs:
            return 0; // No temps needed

        case ActivationType::LeakyRelu:
            return 0; // Implemented with VCmpGE + VCndMask, no temps

        case ActivationType::Gelu:
        case ActivationType::Sigmoid:
        case ActivationType::Tanh:
            return 1; // Need 1 temp

        case ActivationType::Clamp:
            return 0; // No temps needed

        case ActivationType::Silu:
            return 1; // Need temp for sigmoid

        case ActivationType::Swish:
            return 2; // Need 2 temps

        case ActivationType::ClippedRelu:
            return 1; // Need 1 temp

        case ActivationType::DGelu:
            return 3; // Need 3 temps

        default:
            return 0;
        }
    }

    // ========================================================================
    // ActivationBuilder
    // ========================================================================

    PipelineConfig ActivationBuilder::getOptimizationConfig() const
    {
        // Activation-specific: Only Peephole + DCE + DuplicateElim
        PipelineConfig cfg;
        cfg.enablePeephole         = true;
        cfg.enableDCE              = true;
        cfg.enableDuplicateElim    = true;
        cfg.optimizationIterations = 3;
        // All other passes disabled (CFG, Scheduling, WaitCnt)

        // Set architecture from StinkyIR (required by passes that need arch info)
        cfg.gemmTileConfig       = std::make_unique<GemmTileConfig>();
        cfg.gemmTileConfig->arch = getIR().getArch();

        return cfg;
    }

    OperationResult ActivationBuilder::buildRaw(const OperationConfig& config)
    {
        auto& cfg = static_cast<const ActivationConfig&>(config);

        OperationResult result;
        result.operationName = getActivationName(cfg.type);

        // Dispatch to appropriate StinkyIR function
        switch(cfg.type)
        {
        case ActivationType::Relu:
            if(cfg.dtype == ActivationDataType::F32)
                result.instructions = ir.reluF32(builder, cfg.vgprIn, cfg.vgprOut);
            else if(cfg.dtype == ActivationDataType::F16)
                result.instructions = ir.reluF16(builder, cfg.vgprIn, cfg.vgprOut);
            else if(cfg.dtype == ActivationDataType::F64)
                result.instructions = ir.reluF64(builder, cfg.vgprIn, cfg.vgprOut);
            else if(cfg.dtype == ActivationDataType::I32)
                result.instructions = ir.reluI32(builder, cfg.vgprIn, cfg.vgprOut);
            break;

        case ActivationType::LeakyRelu:
            if(cfg.dtype == ActivationDataType::F32)
                result.instructions
                    = ir.leakyReluF32(builder, cfg.vgprIn, cfg.vgprOut, imm(cfg.params[0]));
            else if(cfg.dtype == ActivationDataType::F16)
                result.instructions
                    = ir.leakyReluF16(builder, cfg.vgprIn, cfg.vgprOut, imm(cfg.params[0]));
            break;

        case ActivationType::Gelu:
            if(cfg.dtype == ActivationDataType::F32)
                result.instructions = ir.geluF32(builder, cfg.vgprIn, cfg.vgprOut, cfg.tmpVgprs[0]);
            else if(cfg.dtype == ActivationDataType::F16)
                result.instructions = ir.geluF16(builder, cfg.vgprIn, cfg.vgprOut, cfg.tmpVgprs[0]);
            break;

        case ActivationType::Sigmoid:
            if(cfg.dtype == ActivationDataType::F32)
                result.instructions
                    = ir.sigmoidF32(builder, cfg.vgprIn, cfg.vgprOut, cfg.tmpVgprs[0]);
            else if(cfg.dtype == ActivationDataType::F16)
                result.instructions
                    = ir.sigmoidF16(builder, cfg.vgprIn, cfg.vgprOut, cfg.tmpVgprs[0]);
            break;

        case ActivationType::Abs:
            if(cfg.dtype == ActivationDataType::F32)
                result.instructions = ir.absF32(builder, cfg.vgprIn, cfg.vgprOut);
            else if(cfg.dtype == ActivationDataType::F16)
                result.instructions = ir.absF16(builder, cfg.vgprIn, cfg.vgprOut);
            break;

        case ActivationType::Clamp:
            if(cfg.dtype == ActivationDataType::F32)
                result.instructions = ir.clampF32(
                    builder, cfg.vgprIn, cfg.vgprOut, imm(cfg.params[0]), imm(cfg.params[1]));
            else if(cfg.dtype == ActivationDataType::F16)
                result.instructions = ir.clampF16(
                    builder, cfg.vgprIn, cfg.vgprOut, imm(cfg.params[0]), imm(cfg.params[1]));
            break;

        case ActivationType::Silu:
            if(cfg.dtype == ActivationDataType::F32)
                result.instructions = ir.siluF32(builder, cfg.vgprIn, cfg.vgprOut, cfg.tmpVgprs[0]);
            else if(cfg.dtype == ActivationDataType::F16)
                result.instructions = ir.siluF16(builder, cfg.vgprIn, cfg.vgprOut, cfg.tmpVgprs[0]);
            break;

        case ActivationType::Swish:
            if(cfg.dtype == ActivationDataType::F32)
                result.instructions = ir.swishF32(builder,
                                                  cfg.vgprIn,
                                                  cfg.vgprOut,
                                                  imm(cfg.params[0]),
                                                  cfg.tmpVgprs[0],
                                                  cfg.tmpVgprs[1]);
            else if(cfg.dtype == ActivationDataType::F16)
                result.instructions = ir.swishF16(builder,
                                                  cfg.vgprIn,
                                                  cfg.vgprOut,
                                                  imm(cfg.params[0]),
                                                  cfg.tmpVgprs[0],
                                                  cfg.tmpVgprs[1]);
            break;

        case ActivationType::ClippedRelu:
            if(cfg.dtype == ActivationDataType::F32)
                result.instructions = ir.clippedReluF32(this->builder,
                                                        cfg.vgprIn,
                                                        cfg.vgprOut,
                                                        imm(cfg.params[0]),
                                                        imm(cfg.params[1]),
                                                        cfg.tmpVgprs[0]);
            else if(cfg.dtype == ActivationDataType::F16)
                result.instructions = ir.clippedReluF16(this->builder,
                                                        cfg.vgprIn,
                                                        cfg.vgprOut,
                                                        imm(cfg.params[0]),
                                                        imm(cfg.params[1]),
                                                        cfg.tmpVgprs[0]);
            else if(cfg.dtype == ActivationDataType::F64)
                result.instructions = ir.clippedReluF64(this->builder,
                                                        cfg.vgprIn,
                                                        cfg.vgprOut,
                                                        imm(cfg.params[0]),
                                                        imm(cfg.params[1]),
                                                        cfg.tmpVgprs[0]);
            else if(cfg.dtype == ActivationDataType::I32)
                result.instructions = ir.clippedReluI32(this->builder,
                                                        cfg.vgprIn,
                                                        cfg.vgprOut,
                                                        imm(static_cast<int32_t>(cfg.params[0])),
                                                        imm(static_cast<int32_t>(cfg.params[1])),
                                                        cfg.tmpVgprs[0]);
            break;

        case ActivationType::DGelu:
            if(cfg.dtype == ActivationDataType::F32)
                result.instructions = ir.dgeluF32(this->builder,
                                                  cfg.vgprIn,
                                                  cfg.vgprOut,
                                                  cfg.tmpVgprs[0],
                                                  cfg.tmpVgprs[1],
                                                  cfg.tmpVgprs[2]);
            break;

        default:
            // This should never happen if validate() is called properly
            STINKY_UNREACHABLE("Unsupported activation type");
        }

        return result; // Never reached, but silence warning
    }

    std::string ActivationBuilder::getActivationName(ActivationType type)
    {
        switch(type)
        {
        case ActivationType::Relu:
            return "relu";
        case ActivationType::LeakyRelu:
            return "leaky_relu";
        case ActivationType::Gelu:
            return "gelu";
        case ActivationType::Sigmoid:
            return "sigmoid";
        case ActivationType::Tanh:
            return "tanh";
        case ActivationType::Abs:
            return "abs";
        case ActivationType::Clamp:
            return "clamp";
        case ActivationType::Silu:
            return "silu";
        case ActivationType::Swish:
            return "swish";
        case ActivationType::ClippedRelu:
            return "clipped_relu";
        case ActivationType::DGelu:
            return "dgelu";
        default:
            return "unknown";
        }
    }

} // namespace stinkytofu
