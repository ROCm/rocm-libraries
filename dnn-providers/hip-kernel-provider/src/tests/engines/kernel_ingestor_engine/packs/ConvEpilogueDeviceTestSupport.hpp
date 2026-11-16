// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <variant>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceConvolution.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvEpilogueTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file ConvEpilogueDeviceTestSupport.hpp
 * @brief Fused conv + pointwise graphs on the device, through any conv engine's match, prepare
 *        and launch, against the CPU reference convolution (fp32 output) followed by the chain
 *        in double on the host: the one-rounding semantics the fused kernels implement.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing::conv_epilogue_test
{

using Half = hipdnn_data_sdk::types::half;
using BFloat16 = hipdnn_data_sdk::types::bfloat16;

template <typename T>
DataType elementTypeOf()
{
    if constexpr(std::is_same_v<T, float>)
    {
        return DataType::FLOAT;
    }
    else if constexpr(std::is_same_v<T, Half>)
    {
        return DataType::HALF;
    }
    else
    {
        return DataType::BFLOAT16;
    }
}

/// Unit roundoff of T's storage.
template <typename T>
double unitRoundoff()
{
    if constexpr(std::is_same_v<T, float>)
    {
        return 6.0e-8;
    }
    else if constexpr(std::is_same_v<T, Half>)
    {
        return 4.9e-4;
    }
    else
    {
        return 3.9e-3;
    }
}

/// Special values a case plants.
enum class Specials
{
    NONE,
    /// NaN and +-infinity in the per-channel operands of a few channels.
    NAN_INF,
    /// x all zero: the convolution is exactly 0 and every output is the chain of the operands
    /// alone, which sit on the clip bounds.
    CLIP_BOUNDARY,
};

/// One device-side value a step reads: the per-channel or scalar tensor, or a runtime scalar's
/// host slot.
struct OperandData
{
    int64_t uid = 0;
    std::vector<double> values; ///< per channel, or one
    std::unique_ptr<hipdnn_data_sdk::utilities::TensorBase<float>> f32;
    std::unique_ptr<hipdnn_data_sdk::utilities::TensorBase<Half>> f16;
    std::unique_ptr<hipdnn_data_sdk::utilities::TensorBase<BFloat16>> bf16;
    float host = 0.0F; ///< a runtime scalar's value, read at execute
};

/**
 * Runs one engine on a fused graph. Called with the graph's context and device buffers; it
 * matches, prepares and launches each config it wants checked, and calls @p checkAfterLaunch
 * after each launch (synchronised). Returns the number of configs it launched.
 */
using EngineRunner = std::function<int(const GraphFixture& fixture,
                                       const std::vector<hipdnnPluginDeviceBuffer_t>& buffers,
                                       const std::function<void(const std::string&)>& check)>;

/// A kernel's values of @p fields, as one string: configs with equal keys run the same code
/// paths for the fields named.
inline std::string metadataKey(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
                               const std::vector<std::string>& fields)
{
    std::string key;
    for(const auto& field : fields)
    {
        const auto value = kernel.tryGetMetadata(field);
        key += field + "=";
        if(value.has_value())
        {
            std::visit(
                [&key](const auto& v) {
                    using V = std::decay_t<decltype(v)>;
                    if constexpr(std::is_same_v<V, std::string>)
                    {
                        key += v;
                    }
                    else if constexpr(std::is_same_v<V, std::vector<int64_t>>)
                    {
                        for(const auto e : v)
                        {
                            key += std::to_string(e) + ",";
                        }
                    }
                    else
                    {
                        key += std::to_string(v);
                    }
                },
                *value);
        }
        key += ";";
    }
    return key;
}

/// The runner for an ingestor pack: every kernel of @p kernels the pack's kernel match admits,
/// or, with @p distinctBy, the first admitted kernel of each distinct value of those metadata
/// fields (every code path of the fields named, at a bounded compile cost).
inline EngineRunner
    catalogRunner(PackSymbols pack,
                  const std::vector<hipdnn_plugin_sdk::ingestor::KernelDefinition>& kernels,
                  const std::vector<std::string>& distinctBy = {},
                  bool requireMatch = true)
{
    return [pack, kernels, distinctBy, requireMatch](
               const GraphFixture& fixture,
               const std::vector<hipdnnPluginDeviceBuffer_t>& buffers,
               const std::function<void(const std::string&)>& check) {
        const auto bound = matchesGraph(pack, fixture.context());
        if(!bound.has_value())
        {
            if(requireMatch)
            {
                ADD_FAILURE() << "the engine's graph match declines the fused graph";
            }
            return 0;
        }
        const auto& handler = dispatchHandler(pack);
        const Handle handle;
        int ran = 0;
        std::vector<std::string> seen;
        for(const auto& kernel : kernels)
        {
            if(!matchesKernel(pack, fixture.context(), kernel, *bound))
            {
                continue;
            }
            if(!distinctBy.empty())
            {
                auto key = metadataKey(kernel, distinctBy);
                if(std::find(seen.begin(), seen.end(), key) != seen.end())
                {
                    continue;
                }
                seen.push_back(std::move(key));
            }
            const auto prepared = handler.prepare(fixture.context(), *bound, kernel);
            const size_t workspaceBytes = handler.workspaceBytes(fixture.context(), *bound, kernel);
            void* workspace = nullptr;
            if(workspaceBytes > 0 && hipMalloc(&workspace, workspaceBytes) != hipSuccess)
            {
                ADD_FAILURE() << "workspace allocation failed for " << kernel.name;
                continue;
            }
            handler.launch(handle,
                           *prepared,
                           buffers.data(),
                           static_cast<uint32_t>(buffers.size()),
                           workspace);
            EXPECT_EQ(hipStreamSynchronize(handle.getStream()), hipSuccess);
            if(workspace != nullptr)
            {
                EXPECT_EQ(hipFree(workspace), hipSuccess);
            }
            ++ran;
            check(kernel.name);
        }
        return ran;
    };
}

/// Product of a step chain's error gains, for the tolerance.
inline double chainGain(const std::vector<Step>& steps, const std::vector<double>& inputMagnitudes)
{
    double gain = 1.0;
    for(size_t i = 0; i < steps.size(); ++i)
    {
        gain *= stepGain(steps[i], inputMagnitudes[i]);
    }
    return gain;
}

/**
 * Builds the fused graph of @p shape (forward) and @p steps in element type T, fills x, w and
 * the operands, runs @p runner and checks every launch against the host reference.
 *
 * @param convTolerance The convolution's own absolute error bound for an output whose
 *        reduction has @p reduction terms each at most 1 in magnitude (x and w in [-1, 1]).
 * @return the number of configs launched.
 */
template <typename T>
int checkFusedCase(
    ConvShape shape,
    const std::vector<Step>& steps,
    const EngineRunner& runner,
    Specials specials = Specials::NONE,
    const std::function<double(int64_t reduction)>& convTolerance
    = [](int64_t reduction) { return 1e-6 * static_cast<double>(reduction); },
    unsigned int seed = 1234)
{
    using hipdnn_data_sdk::utilities::Tensor;
    shape.dataType = elementTypeOf<T>();
    const auto spec = fusedSpecFor(shape, steps, DataType::FLOAT);
    const GraphFixture fixture(buildFusedGraph(spec), currentDeviceProperties());

    Tensor<T> x(spec.conv.x.dims, spec.conv.x.strides);
    Tensor<T> w(spec.conv.w.dims, spec.conv.w.strides);
    const auto& outSpec
        = entryOf(const_cast<FusedGraphSpec&>(spec), outputUid(steps.size())).tensor;
    Tensor<T> y(outSpec.dims, outSpec.strides);
    if(specials == Specials::CLIP_BOUNDARY)
    {
        x.fillWithValue(static_cast<T>(0.0F));
    }
    else
    {
        x.fillTensorWithRandomValues(-1.0F, 1.0F, seed);
    }
    w.fillTensorWithRandomValues(-1.0F, 1.0F, seed + 1);
    y.fillWithSentinelValue();

    // Operands: per-channel and device scalars in their tensors' types, runtime scalars on the
    // host. Values in [-1, 1] (scales in [0.5, 1.5]), with NaN / infinity planted on request,
    // or sitting on the clip bounds of the chain's relu for the boundary case.
    std::mt19937 random(seed + 2);
    std::uniform_real_distribution<float> unit(-1.0F, 1.0F);
    std::vector<OperandData> operands(steps.size());
    std::vector<double> magnitudes(steps.size(), 1.0);
    float boundaryLower = 0.0F;
    float boundaryUpper = 1.0F;
    for(const auto& step : steps)
    {
        if(step.mode == PointwiseMode::RELU_FWD)
        {
            boundaryLower = step.lowerClip.value_or(0.0F);
            boundaryUpper = step.upperClip.value_or(boundaryLower + 1.0F);
        }
    }
    std::vector<hipdnnPluginDeviceBuffer_t> buffers{{conv_naive_test::X_UID, x.rawDeviceData()},
                                                    {conv_naive_test::W_UID, w.rawDeviceData()},
                                                    {outputUid(steps.size()), y.rawDeviceData()}};
    for(size_t i = 0; i < steps.size(); ++i)
    {
        const auto& step = steps[i];
        if(step.operand == OperandKind::NONE)
        {
            continue;
        }
        auto& data = operands[i];
        data.uid = OPERAND_UID + static_cast<int64_t>(i);
        const bool isScale = step.mode == PointwiseMode::MUL;
        const auto draw = [&](int64_t channel) -> double {
            if(specials == Specials::CLIP_BOUNDARY)
            {
                // Below, on, between and above the bounds, by channel.
                const std::array<double, 5> picks{
                    static_cast<double>(boundaryLower) - 0.5,
                    static_cast<double>(boundaryLower),
                    0.5 * static_cast<double>(boundaryLower + boundaryUpper),
                    static_cast<double>(boundaryUpper),
                    static_cast<double>(boundaryUpper) + 0.5};
                return isScale ? 1.0 : picks[static_cast<size_t>(channel) % picks.size()];
            }
            if(specials == Specials::NAN_INF && !isScale)
            {
                if(channel == 1)
                {
                    return std::numeric_limits<double>::quiet_NaN();
                }
                if(channel == 2)
                {
                    return std::numeric_limits<double>::infinity();
                }
                if(channel == 3)
                {
                    return -std::numeric_limits<double>::infinity();
                }
            }
            const auto u = static_cast<double>(unit(random));
            return isScale ? 1.0 + 0.5 * u : u;
        };
        if(step.operand == OperandKind::CONSTANT || step.operand == OperandKind::RUNTIME)
        {
            data.values = {static_cast<double>(step.value)};
            if(step.operand == OperandKind::RUNTIME)
            {
                data.host = step.value;
                buffers.push_back({data.uid, &data.host});
            }
            magnitudes[i] = std::fabs(static_cast<double>(step.value));
            continue;
        }
        const int64_t count = step.operand == OperandKind::PER_CHANNEL ? shape.k : 1;
        const auto& entry = entryOf(const_cast<FusedGraphSpec&>(spec), data.uid).tensor;
        double largest = 0.0;
        for(int64_t c = 0; c < count; ++c)
        {
            data.values.push_back(draw(c));
        }
        // Rounded to the operand's type, as the kernel reads it.
        switch(step.operandType)
        {
        case DataType::HALF:
        {
            data.f16 = std::make_unique<Tensor<Half>>(entry.dims, entry.strides);
            auto* host = static_cast<Half*>(data.f16->rawHostData());
            for(int64_t c = 0; c < count; ++c)
            {
                host[c]
                    = static_cast<Half>(static_cast<float>(data.values[static_cast<size_t>(c)]));
                data.values[static_cast<size_t>(c)]
                    = static_cast<double>(static_cast<float>(host[c]));
            }
            data.f16->markHostModified();
            buffers.push_back({data.uid, data.f16->rawDeviceData()});
            break;
        }
        case DataType::BFLOAT16:
        {
            data.bf16 = std::make_unique<Tensor<BFloat16>>(entry.dims, entry.strides);
            auto* host = static_cast<BFloat16*>(data.bf16->rawHostData());
            for(int64_t c = 0; c < count; ++c)
            {
                host[c] = static_cast<BFloat16>(
                    static_cast<float>(data.values[static_cast<size_t>(c)]));
                data.values[static_cast<size_t>(c)]
                    = static_cast<double>(static_cast<float>(host[c]));
            }
            data.bf16->markHostModified();
            buffers.push_back({data.uid, data.bf16->rawDeviceData()});
            break;
        }
        default:
        {
            data.f32 = std::make_unique<Tensor<float>>(entry.dims, entry.strides);
            auto* host = static_cast<float*>(data.f32->rawHostData());
            for(int64_t c = 0; c < count; ++c)
            {
                host[c] = static_cast<float>(data.values[static_cast<size_t>(c)]);
                data.values[static_cast<size_t>(c)] = static_cast<double>(host[c]);
            }
            data.f32->markHostModified();
            buffers.push_back({data.uid, data.f32->rawDeviceData()});
            break;
        }
        }
        for(const double v : data.values)
        {
            if(std::isfinite(v))
            {
                largest = std::max(largest, std::fabs(v));
            }
        }
        magnitudes[i] = largest;
    }

    // The reference: the convolution in fp32 from the T inputs, then the chain in double.
    Tensor<float> convOut(outSpec.dims, outSpec.strides);
    hipdnn_test_sdk::utilities::CpuFpReferenceConvolution::fprop(
        x, w, convOut, shape.stride, shape.dilation, shape.prePadding, shape.postPadding);

    int64_t taps = 1;
    for(const auto f : shape.filter)
    {
        taps *= f;
    }
    const int64_t reduction = (shape.c / shape.groups) * taps;
    const double convBound = convTolerance(reduction);
    const double gain = chainGain(steps, magnitudes);

    const auto& dims = outSpec.dims;
    const auto& strides = outSpec.strides;
    const auto check = [&](const std::string& config) {
        y.markDeviceModified();
        const auto* actual = static_cast<const T*>(y.rawHostData());
        const auto* reference = static_cast<const float*>(convOut.rawHostData());
        int64_t mismatches = 0;
        std::vector<int64_t> index(dims.size(), 0);
        while(true)
        {
            int64_t offset = 0;
            for(size_t a = 0; a < dims.size(); ++a)
            {
                offset += index[a] * strides[a];
            }
            const int64_t channel = index[1];
            auto v = static_cast<double>(reference[offset]);
            double magnitude = std::fabs(v);
            for(size_t i = 0; i < steps.size(); ++i)
            {
                double input = 0.0;
                if(steps[i].operand != OperandKind::NONE)
                {
                    const auto& values = operands[i].values;
                    input = values.size() == 1 ? values[0] : values[static_cast<size_t>(channel)];
                }
                v = applyStep(steps[i], v, input);
                magnitude = std::max(magnitude, std::fabs(v));
            }
            const auto want
                = static_cast<double>(static_cast<float>(static_cast<T>(static_cast<float>(v))));
            const auto got = static_cast<double>(static_cast<float>(actual[offset]));
            const double tolerance = convBound * gain + 4e-6 * magnitude
                                     + 2.0 * unitRoundoff<T>() * (std::fabs(want) + 1e-3);
            bool ok = false;
            if(std::isnan(want))
            {
                ok = std::isnan(got);
            }
            else if(std::isinf(want))
            {
                ok = got == want;
            }
            else
            {
                ok = std::fabs(got - want) <= tolerance;
            }
            if(!ok && ++mismatches <= 5)
            {
                ADD_FAILURE() << config << ": at offset " << offset << " (channel " << channel
                              << "): got " << got << ", want " << want << " (unrounded " << v
                              << "), tolerance " << tolerance;
            }
            size_t axis = dims.size();
            bool done = true;
            while(axis > 0)
            {
                --axis;
                if(++index[axis] < dims[axis])
                {
                    done = false;
                    break;
                }
                index[axis] = 0;
            }
            if(done)
            {
                break;
            }
        }
        EXPECT_EQ(mismatches, 0) << config;
        // The next config starts from the sentinel again.
        y.fillWithSentinelValue();
        static_cast<void>(y.rawDeviceData());
    };

    return runner(fixture, buffers, check);
}

/// The menu patterns every fusing engine is checked with.
inline std::vector<std::pair<std::string, std::vector<Step>>> menuChains()
{
    Step swish = unary(PointwiseMode::SWISH_FWD);
    swish.swishBeta = 1.5F;
    Step elu = unary(PointwiseMode::ELU_FWD);
    elu.eluAlpha = 0.7F;
    Step runtimeScale = scale();
    runtimeScale.operand = OperandKind::RUNTIME;
    runtimeScale.value = 1.25F;
    Step constantBias = bias();
    constantBias.operand = OperandKind::CONSTANT;
    constantBias.value = -0.125F;
    Step deviceScalar = scale(DataType::HALF);
    deviceScalar.operand = OperandKind::DEVICE_SCALAR;
    return {
        {"BiasRelu", {bias(DataType::HALF), relu()}},
        {"ScaleBiasClamp", {scale(DataType::BFLOAT16), bias(), relu(-0.5F, 0.5F)}},
        {"BiasClippedRelu", {bias(), relu(std::nullopt, 0.25F)}},
        {"LeakyRelu", {relu(std::nullopt, std::nullopt, 0.1F)}},
        {"BiasGelu", {bias(), unary(PointwiseMode::GELU_FWD)}},
        {"GeluTanh", {unary(PointwiseMode::GELU_APPROX_TANH_FWD)}},
        {"BiasSwish", {bias(DataType::HALF), swish}},
        {"Sigmoid", {unary(PointwiseMode::SIGMOID_FWD)}},
        {"Tanh", {unary(PointwiseMode::TANH_FWD)}},
        {"Elu", {elu}},
        {"ScalarScaleBiasSilu", {runtimeScale, constantBias, unary(PointwiseMode::SWISH_FWD)}},
        {"DeviceScalarScale", {deviceScalar}},
    };
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing::conv_epilogue_test

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_CONV
