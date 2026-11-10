// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see TestFlydslRmsNormEngine.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hip/hip_runtime_api.h>

#include <hipdnn_data_sdk/types/Bfloat16.hpp>
#include <hipdnn_data_sdk/types/Half.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceCausalConv1d.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslCausalConv1dTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslNormDeviceTestSupport.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslRmsNormTestGraphs.hpp"

/**
 * @file TestFlydslCausalConv1dObjects.cpp
 * @brief Every FlyDSL causal-conv1d object this device's shard carries, launched through the
 *        pack's own matchers and dispatch handler against hipDNN's CPU reference
 *        (CpuFpReferenceCausalConv1d): a decode step, a decode step on a pool updated in
 *        place, a prefill across many token tiles, a varlen batch with empty and short
 *        sequences (pool, in place, int64 offsets and indices, bias), a stateless prefill,
 *        and states and weight stored channels-innermost. y starts as NaN, so an output
 *        nothing writes fails; every final-state row the reference leaves alone must come
 *        back untouched.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using data_objects::DataType;
using hipdnn_data_sdk::utilities::Tensor;
using Bf16 = hipdnn_data_sdk::types::bfloat16;
using F16 = hipdnn_data_sdk::types::half;
using Reference = hipdnn_test_sdk::utilities::CpuFpReferenceCausalConv1d;

/// One launch: the sequence lengths and what else the graph binds.
struct ConvCase
{
    std::string name;
    std::vector<int64_t> lengths;
    int64_t channels = 2048;
    bool bias = false;
    bool initialState = true;
    bool finalState = true;
    /// Pool rows beyond N; 0 means no pool (no state_indices).
    int64_t extraPoolRows = 0;
    /// final_state bound to initial_state's device buffer.
    bool inPlace = false;
    bool int64Offsets = false;
    bool int64Indices = false;
    /// Weight stored [W, C] and states [rows, W - 1, C]: the kernel's strided path.
    bool channelsInnermost = false;
};

std::vector<ConvCase> convCases()
{
    std::vector<ConvCase> cases;
    {
        const ConvCase decode{"Decode", std::vector<int64_t>(8, 1)};
        cases.push_back(decode);
    }
    {
        ConvCase pool{"DecodePoolInPlace", std::vector<int64_t>(8, 1)};
        pool.extraPoolRows = 5;
        pool.inPlace = true;
        pool.bias = true;
        cases.push_back(pool);
    }
    {
        ConvCase prefill{"PrefillFromZeros", {300}};
        prefill.initialState = false;
        cases.push_back(prefill);
    }
    {
        ConvCase varlen{"VarlenEmptyAndShortSequences", {0, 5, 1, 0, 2, 77, 3, 0}};
        varlen.channels = 256;
        varlen.bias = true;
        varlen.extraPoolRows = 3;
        varlen.inPlace = true;
        varlen.int64Offsets = true;
        varlen.int64Indices = true;
        cases.push_back(varlen);
    }
    {
        ConvCase stateless{"PrefillWithoutState", {33, 31, 64}};
        stateless.channels = 512;
        stateless.bias = true;
        stateless.initialState = false;
        stateless.finalState = false;
        cases.push_back(stateless);
    }
    {
        ConvCase strided{"ChannelsInnermostStatesAndWeight", {3, 1, 40}};
        strided.channels = 128;
        strided.channelsInnermost = true;
        cases.push_back(strided);
    }
    {
        // C not a multiple of 4: only the 1-lane objects serve it.
        ConvCase odd{"OddChannels", {2, 9}};
        odd.channels = 37;
        odd.bias = true;
        cases.push_back(odd);
    }
    return cases;
}

std::string stringField(const KernelDefinition& kernel, const char* field)
{
    const auto* value = tryGetMetadataField<std::string>(kernel.metadata, field);
    return value == nullptr ? std::string() : *value;
}

int64_t intField(const KernelDefinition& kernel, const char* field)
{
    const auto* value = tryGetMetadataField<int64_t>(kernel.metadata, field);
    return value == nullptr ? 0 : *value;
}

double toleranceFor(const std::string& dtype)
{
    if(dtype == "bf16")
    {
        return 1e-2;
    }
    return dtype == "f16" ? 2e-3 : 1e-5;
}

/// [rows, C, W - 1] logically; stored [rows, W - 1, C] when channels are innermost.
std::vector<int64_t> stateStrides(const std::vector<int64_t>& dims, bool channelsInnermost)
{
    if(!channelsInnermost)
    {
        return flydslCausalConv1dPackedStrides(dims);
    }
    return {dims[1] * dims[2], 1, dims[1]};
}

template <typename T>
void copyInto(Tensor<T>& destination, const Tensor<T>& source)
{
    const auto& dims = source.dims();
    for(int64_t n = 0; n < dims[0]; ++n)
    {
        for(int64_t c = 0; c < dims[1]; ++c)
        {
            for(int64_t k = 0; k < dims[2]; ++k)
            {
                destination.setHostValue(source.getHostValue(n, c, k), n, c, k);
            }
        }
    }
    destination.memory().markHostModified();
}

template <typename T, typename S, typename I>
void expectObjectMatches(const KernelDefinition& kernel,
                         DataType dataType,
                         const ConvCase& convCase)
{
    SCOPED_TRACE(kernel.name + " on " + convCase.name);
    const auto dtype = stringField(kernel, "dtype");
    const int64_t width = intField(kernel, "width");
    const bool silu = stringField(kernel, "activation") == "silu";

    const auto sequences = static_cast<int64_t>(convCase.lengths.size());
    const int64_t tokens
        = std::accumulate(convCase.lengths.begin(), convCase.lengths.end(), int64_t{0});
    const int64_t channels = convCase.channels;
    const bool pool = convCase.extraPoolRows > 0;
    const int64_t stateRows = sequences + convCase.extraPoolRows;

    CausalConv1dGraphSpec spec;
    spec.dataType = dataType;
    spec.tokens = tokens;
    spec.channels = channels;
    spec.width = width;
    spec.sequences = sequences;
    spec.activation = silu ? data_objects::CausalConv1dActivation::SILU
                           : data_objects::CausalConv1dActivation::IDENTITY;
    spec.cuSeqlensDataType = convCase.int64Offsets ? DataType::INT64 : DataType::INT32;
    spec.stateIndicesDataType = convCase.int64Indices ? DataType::INT64 : DataType::INT32;
    spec.bias = convCase.bias;
    spec.initialState = convCase.initialState;
    spec.finalState = convCase.finalState;
    if(pool)
    {
        spec.poolRows = stateRows;
    }
    const std::vector<int64_t> stateDims{stateRows, channels, width - 1};
    const auto stateStride = stateStrides(stateDims, convCase.channelsInnermost);
    const std::vector<int64_t> weightDims{channels, width};
    const std::vector<int64_t> weightStride = convCase.channelsInnermost
                                                  ? std::vector<int64_t>{1, channels}
                                                  : flydslCausalConv1dPackedStrides(weightDims);
    spec.weightStrides = weightStride;
    spec.initialStrides = stateStride;
    spec.finalStrides = stateStride;

    const GraphFixture fixture(buildFlydslCausalConv1dGraph(spec), packedArchDeviceProperties());
    const auto bound = matchesGraph(FLYDSL_CAUSAL_CONV1D, fixture.context());
    ASSERT_TRUE(bound.has_value()) << "the graph this object is meant for is declined";
    if(channels % intField(kernel, "lanes") != 0)
    {
        EXPECT_FALSE(matchesKernel(FLYDSL_CAUSAL_CONV1D, fixture.context(), kernel, *bound))
            << "an object accepts C = " << channels << " off its lanes";
        return;
    }
    ASSERT_TRUE(matchesKernel(FLYDSL_CAUSAL_CONV1D, fixture.context(), kernel, *bound));

    const auto& handler = dispatchHandler(FLYDSL_CAUSAL_CONV1D);
    const auto prepared = handler.prepare(fixture.context(), *bound, kernel);
    ASSERT_NE(prepared, nullptr);

    // Operands.
    const std::vector<int64_t> rowDims{tokens, channels};
    Tensor<T> x(rowDims);
    Tensor<T> weight(weightDims, weightStride);
    Tensor<T> bias({channels});
    Tensor<T> y(rowDims);
    Tensor<T> expectedY(rowDims);
    Tensor<S> cuSeqlens({sequences + 1});
    Tensor<I> stateIndices({sequences});
    Tensor<T> initialState(stateDims, stateStride);
    Tensor<T> finalState(stateDims, stateStride);
    Tensor<T> expectedFinal(stateDims, stateStride);
    x.fillWithRandomValues(T(-1.0F), T(1.0F), 1234U);
    weight.fillWithRandomValues(T(-0.5F), T(0.5F), 2345U);
    bias.fillWithRandomValues(T(-0.5F), T(0.5F), 3456U);
    initialState.fillWithRandomValues(T(-1.0F), T(1.0F), 4567U);
    finalState.fillWithRandomValues(T(-1.0F), T(1.0F), 5678U);
    y.fillWithSentinelValue();

    S offset = 0;
    cuSeqlens.setHostValue(offset, 0);
    for(int64_t n = 0; n < sequences; ++n)
    {
        offset = static_cast<S>(offset + convCase.lengths[static_cast<size_t>(n)]);
        cuSeqlens.setHostValue(offset, n + 1);
    }
    cuSeqlens.memory().markHostModified();
    std::vector<I> rows(static_cast<size_t>(stateRows));
    std::iota(rows.begin(), rows.end(), I{0});
    std::shuffle(rows.begin(), rows.end(), std::mt19937(7U));
    for(int64_t n = 0; n < sequences; ++n)
    {
        stateIndices.setHostValue(rows[static_cast<size_t>(n)], n);
    }
    stateIndices.memory().markHostModified();

    // The reference, on its own copy of the state it writes: the initial state itself when
    // in place, else the final state's starting contents.
    Tensor<T> referenceInitial(stateDims, stateStride);
    copyInto(referenceInitial, initialState);
    copyInto(expectedFinal, convCase.inPlace ? initialState : finalState);
    auto* const initialStatePtr = convCase.inPlace ? &expectedFinal : &referenceInitial;
    Reference::forward<T, S, I>(x,
                                weight,
                                cuSeqlens,
                                expectedY,
                                spec.activation,
                                convCase.bias ? &bias : nullptr,
                                convCase.initialState ? initialStatePtr : nullptr,
                                pool ? &stateIndices : nullptr,
                                convCase.finalState ? &expectedFinal : nullptr);

    auto& gpuFinal = convCase.inPlace ? initialState : finalState;
    std::vector<hipdnnPluginDeviceBuffer_t> arguments{
        {FLYDSL_CAUSAL_CONV1D_X_UID, x.memory().deviceData()},
        {FLYDSL_CAUSAL_CONV1D_WEIGHT_UID, weight.memory().deviceData()},
        {FLYDSL_CAUSAL_CONV1D_CU_SEQLENS_UID, cuSeqlens.memory().deviceData()},
        {FLYDSL_CAUSAL_CONV1D_Y_UID, y.memory().deviceData()}};
    if(convCase.bias)
    {
        arguments.push_back({FLYDSL_CAUSAL_CONV1D_BIAS_UID, bias.memory().deviceData()});
    }
    if(convCase.initialState)
    {
        arguments.push_back(
            {FLYDSL_CAUSAL_CONV1D_INITIAL_STATE_UID, initialState.memory().deviceData()});
    }
    if(pool)
    {
        arguments.push_back(
            {FLYDSL_CAUSAL_CONV1D_STATE_INDICES_UID, stateIndices.memory().deviceData()});
    }
    if(convCase.finalState)
    {
        arguments.push_back({FLYDSL_CAUSAL_CONV1D_FINAL_STATE_UID, gpuFinal.memory().deviceData()});
    }
    const Handle handle;
    handler.launch(
        handle, *prepared, arguments.data(), static_cast<uint32_t>(arguments.size()), nullptr);
    ASSERT_EQ(hipSuccess, hipDeviceSynchronize());
    y.memory().markDeviceModified();
    gpuFinal.memory().markDeviceModified();
    // Pull the results back: the const reads below do not migrate.
    static_cast<void>(y.memory().hostData());
    static_cast<void>(gpuFinal.memory().hostData());

    const double tolerance = toleranceFor(dtype);
    size_t mismatches = 0;
    for(int64_t t = 0; t < tokens; ++t)
    {
        for(int64_t c = 0; c < channels; ++c)
        {
            const auto got = static_cast<double>(static_cast<float>(y.getHostValue(t, c)));
            const auto want = static_cast<double>(static_cast<float>(expectedY.getHostValue(t, c)));
            if(!std::isfinite(got) || std::abs(got - want) > tolerance * (1.0 + std::abs(want)))
            {
                if(++mismatches <= 5)
                {
                    ADD_FAILURE() << "y[" << t << ", " << c << "] = " << got << ", expected "
                                  << want;
                }
            }
        }
    }
    EXPECT_EQ(mismatches, 0U);

    if(!convCase.finalState)
    {
        return;
    }
    // The final state is a copy of inputs: exact, rows the reference leaves alone included.
    mismatches = 0;
    for(int64_t n = 0; n < stateRows; ++n)
    {
        for(int64_t c = 0; c < channels; ++c)
        {
            for(int64_t k = 0; k < width - 1; ++k)
            {
                const auto got = static_cast<float>(gpuFinal.getHostValue(n, c, k));
                const auto want = static_cast<float>(expectedFinal.getHostValue(n, c, k));
                if(got != want && ++mismatches <= 5)
                {
                    ADD_FAILURE() << "final_state[" << n << ", " << c << ", " << k << "] = " << got
                                  << ", expected " << want;
                }
            }
        }
    }
    EXPECT_EQ(mismatches, 0U);
}

template <typename T>
void expectObjectMatchesEveryCase(const KernelDefinition& kernel, DataType dataType)
{
    for(const auto& convCase : convCases())
    {
        if(convCase.int64Offsets && convCase.int64Indices)
        {
            expectObjectMatches<T, int64_t, int64_t>(kernel, dataType, convCase);
        }
        else
        {
            expectObjectMatches<T, int32_t, int32_t>(kernel, dataType, convCase);
        }
    }
}

TEST(TestGpuFlydslCausalConv1dObjects, EveryObjectMatchesTheCpuReference)
{
    SKIP_IF_NO_DEVICES();
    const auto kernels = stagedKernels(FLYDSL_CAUSAL_CONV1D.engineName);
    if(kernels.empty())
    {
        reportUnstaged("causal_conv1d",
                       "this build staged no causal-conv1d objects for this device");
        return;
    }

    for(const auto& kernel : kernels)
    {
        const auto dtype = stringField(kernel, "dtype");
        if(dtype == "bf16")
        {
            expectObjectMatchesEveryCase<Bf16>(kernel, DataType::BFLOAT16);
        }
        else if(dtype == "f16")
        {
            expectObjectMatchesEveryCase<F16>(kernel, DataType::HALF);
        }
        else if(dtype == "f32")
        {
            expectObjectMatchesEveryCase<float>(kernel, DataType::FLOAT);
        }
        else
        {
            ADD_FAILURE() << kernel.name << " declares dtype '" << dtype << "'";
        }
    }
    RecordProperty("objects", static_cast<int>(kernels.size()));
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
