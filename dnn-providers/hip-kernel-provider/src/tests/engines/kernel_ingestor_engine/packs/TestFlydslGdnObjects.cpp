// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see TestFlydslRmsNormEngine.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
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
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceGdn.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslGdnTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslNormDeviceTestSupport.hpp"

/**
 * @file TestFlydslGdnObjects.cpp
 * @brief Every FlyDSL GDN object this device's shard carries, launched through its pack's
 *        dispatch handler and compared with hipDNN's CPU reference (`CpuFpReferenceGdn`,
 *        f32 compute).
 *
 * Each object runs, with Qwen3.8's flags (safe gate with a_log and dt_bias, sigmoid beta,
 * q/k L2 norm, default scale) and GQA head replication:
 *  - a varlen prefill (sequences of 5, 0 and 17 tokens, the empty one's final state its
 *    initial state) and a decode step (four one-token sequences);
 *  - carried-state equivalence: a prefill of each sequence's first tokens, then a decode
 *    of its next token from the prefill's final state, against one reference run over the
 *    whole sequences -- the decode output and the state it leaves must match the
 *    reference's last token and final state.
 * A `state_strided` 1 object runs these with K-major states. Each vector-state K = V = 128
 * object also runs Qwen3.8's own head counts (16 q/k heads, 48 v heads) and the flag and
 * layout variants: LOG and LINEAR gates without the safe gate, raw beta, no L2 norm and an
 * explicit scale, allow_neg_eigval, f16 and bf16 gates and parameters, no initial or final
 * state, a state pool through state_indices written in place over the initial state
 * (overwrite_initial_state), and q/k/v as strided slices of one projection row.
 *
 * Outputs start as NaN (0xFF bytes) so an element nothing wrote fails; a pool's rows no
 * sequence names must come back untouched.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using data_objects::DataType;
using data_objects::GateDomain;
using Bf16 = hipdnn_data_sdk::types::bfloat16;
using F16 = hipdnn_data_sdk::types::half;
using hipdnn_data_sdk::utilities::Tensor;
using hipdnn_test_sdk::utilities::CpuFpReferenceGdn;
using hipdnn_test_sdk::utilities::GdnConfig;
using hipdnn_test_sdk::utilities::GdnTensors;

int64_t intField(const KernelDefinition& kernel, const char* field)
{
    const auto* value = tryGetMetadataField<int64_t>(kernel.metadata, field);
    return value == nullptr ? -1 : *value;
}

std::string stringField(const KernelDefinition& kernel, const char* field)
{
    const auto* value = tryGetMetadataField<std::string>(kernel.metadata, field);
    return value == nullptr ? std::string() : *value;
}

/// A device buffer freed on scope exit.
class DeviceBuffer
{
public:
    explicit DeviceBuffer(size_t bytes)
        : _bytes(std::max<size_t>(bytes, 1))
    {
        EXPECT_EQ(hipSuccess, hipMalloc(&_pointer, _bytes));
    }

    DeviceBuffer(const void* host, size_t bytes)
        : DeviceBuffer(bytes)
    {
        EXPECT_EQ(hipSuccess, hipMemcpy(_pointer, host, bytes, hipMemcpyHostToDevice));
    }

    ~DeviceBuffer()
    {
        static_cast<void>(hipFree(_pointer));
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&&) = delete;
    DeviceBuffer& operator=(DeviceBuffer&&) = delete;

    void* get() const
    {
        return _pointer;
    }

    void read(void* host, size_t bytes) const
    {
        EXPECT_EQ(hipSuccess, hipMemcpy(host, _pointer, bytes, hipMemcpyDeviceToHost));
    }

private:
    size_t _bytes;
    void* _pointer = nullptr;
};

/// Every index of @p dims in row-major order, with its packed position.
void forEachIndex(const std::vector<int64_t>& dims,
                  const std::function<void(const std::vector<int64_t>&, size_t)>& visit)
{
    std::vector<int64_t> index(dims.size(), 0);
    const auto count = static_cast<size_t>(
        std::accumulate(dims.begin(), dims.end(), int64_t{1}, std::multiplies<>()));
    for(size_t linear = 0; linear < count; ++linear)
    {
        visit(index, linear);
        for(auto axis = dims.size(); axis-- > 0;)
        {
            if(++index[axis] < dims[axis])
            {
                break;
            }
            index[axis] = 0;
        }
    }
}

/// The value @p value becomes when stored as @p type.
float quantize(float value, DataType type)
{
    if(type == DataType::BFLOAT16)
    {
        return static_cast<float>(Bf16(value));
    }
    if(type == DataType::HALF)
    {
        return static_cast<float>(F16(value));
    }
    return value;
}

/// A host operand: logical dims, packed f32 values (already representable in its type).
struct Operand
{
    std::vector<int64_t> dims;
    std::vector<float> values;
};

/// Everything one run needs: the graph spec (dtypes, flags, layouts) and its operands.
struct Problem
{
    GdnGraphSpec spec;
    std::vector<int32_t> cuSeqlens;
    std::vector<int32_t> stateIndices;
    Operand q, k, v, g, beta, aLog, dtBias, initialState;
    /// The final-state buffer's contents before the run (a pool's untouched rows keep them).
    Operand finalStatePrior;
};

struct Outputs
{
    std::vector<float> o;
    std::vector<float> finalState;
};

std::vector<int64_t> stridesFor(const GdnGraphSpec& spec, int64_t uid, const Operand& operand)
{
    const auto it = spec.strides.find(uid);
    return it == spec.strides.end() ? flydslPackedStrides(operand.dims) : it->second;
}

/// @p operand laid out with @p strides in type T, as the device will read it.
template <typename T>
Tensor<T> stridedTensor(const Operand& operand, const std::vector<int64_t>& strides)
{
    Tensor<T> tensor(operand.dims, strides);
    // Padding between strided elements is never read; fill it with something visible.
    std::fill_n(tensor.memory().hostData(), tensor.elementSpace(), T(-777.0F));
    forEachIndex(operand.dims, [&](const std::vector<int64_t>& index, size_t linear) {
        tensor.setHostValue(T(operand.values[linear]), index);
    });
    return tensor;
}

template <typename T>
std::vector<float> packedValues(const Tensor<T>& tensor, const std::vector<int64_t>& dims)
{
    std::vector<float> values;
    forEachIndex(dims, [&](const std::vector<int64_t>& index, size_t /*linear*/) {
        values.push_back(static_cast<float>(tensor.getHostValue(index)));
    });
    return values;
}

/// One operand on the device in its graph type and layout.
class DeviceOperand
{
public:
    DeviceOperand(const Operand& operand, DataType type, const std::vector<int64_t>& strides)
        : _dims(operand.dims)
        , _strides(strides)
        , _type(type)
    {
        switch(type)
        {
        case DataType::BFLOAT16:
            upload(stridedTensor<Bf16>(operand, strides));
            break;
        case DataType::HALF:
            upload(stridedTensor<F16>(operand, strides));
            break;
        default:
            upload(stridedTensor<float>(operand, strides));
            break;
        }
    }

    void* get() const
    {
        return _buffer->get();
    }

    /// The device contents, read back through the layout as packed values.
    std::vector<float> read() const
    {
        switch(_type)
        {
        case DataType::BFLOAT16:
            return readAs<Bf16>();
        case DataType::HALF:
            return readAs<F16>();
        default:
            return readAs<float>();
        }
    }

private:
    template <typename T>
    void upload(const Tensor<T>& tensor)
    {
        _bytes = tensor.elementSpace() * sizeof(T);
        _buffer = std::make_unique<DeviceBuffer>(tensor.memory().hostData(), _bytes);
    }

    template <typename T>
    std::vector<float> readAs() const
    {
        Tensor<T> tensor(_dims, _strides);
        _buffer->read(tensor.memory().hostData(), _bytes);
        tensor.memory().markHostModified();
        return packedValues(tensor, _dims);
    }

    std::vector<int64_t> _dims;
    std::vector<int64_t> _strides;
    DataType _type;
    size_t _bytes = 0;
    std::unique_ptr<DeviceBuffer> _buffer;
};

/// Raw int32 values on the device.
DeviceBuffer int32Buffer(const std::vector<int32_t>& values)
{
    return {values.data(), values.size() * sizeof(int32_t)};
}

/// The NaN pattern every output starts as.
Operand nanOperand(const std::vector<int64_t>& dims)
{
    Operand operand;
    operand.dims = dims;
    operand.values.assign(static_cast<size_t>(std::accumulate(
                              dims.begin(), dims.end(), int64_t{1}, std::multiplies<>())),
                          std::numeric_limits<float>::quiet_NaN());
    return operand;
}

/// Launches @p kernel on @p problem through the pack's dispatch handler.
Outputs runObject(const KernelDefinition& kernel, const Problem& problem)
{
    const auto& spec = problem.spec;
    const GraphFixture fixture(buildFlydslGdnGraph(spec), packedArchDeviceProperties());
    const auto bound = matchesGraph(FLYDSL_GDN, fixture.context());
    EXPECT_TRUE(bound.has_value()) << "the graph this object is meant for is declined";
    if(!bound.has_value())
    {
        return {};
    }
    // The object's own metadata has to admit the graph built for it; one that does not would
    // never be selected, which a launch by name would otherwise hide.
    EXPECT_TRUE(matchesKernel(FLYDSL_GDN, fixture.context(), kernel, *bound));
    const auto& handler = dispatchHandler(FLYDSL_GDN);
    const auto prepared = handler.prepare(fixture.context(), *bound, kernel);
    EXPECT_NE(prepared, nullptr);
    if(prepared == nullptr)
    {
        return {};
    }

    const auto device = [&](int64_t uid, const Operand& operand, DataType type) {
        return std::make_unique<DeviceOperand>(operand, type, stridesFor(spec, uid, operand));
    };
    const auto q = device(FLYDSL_GDN_Q_UID, problem.q, spec.ioType);
    const auto k = device(FLYDSL_GDN_K_UID, problem.k, spec.ioType);
    const auto v = device(FLYDSL_GDN_V_UID, problem.v, spec.ioType);
    const auto g = device(FLYDSL_GDN_G_UID, problem.g, spec.gType);
    const auto beta = device(FLYDSL_GDN_BETA_UID, problem.beta, spec.betaType);
    const auto oDims = std::vector<int64_t>{spec.tokens, spec.headsOut(), spec.valueDim};
    const auto o = device(FLYDSL_GDN_O_UID, nanOperand(oDims), spec.ioType);
    const auto cu = int32Buffer(problem.cuSeqlens);

    std::vector<hipdnnPluginDeviceBuffer_t> arguments{{FLYDSL_GDN_Q_UID, q->get()},
                                                      {FLYDSL_GDN_K_UID, k->get()},
                                                      {FLYDSL_GDN_V_UID, v->get()},
                                                      {FLYDSL_GDN_G_UID, g->get()},
                                                      {FLYDSL_GDN_BETA_UID, beta->get()},
                                                      {FLYDSL_GDN_CU_SEQLENS_UID, cu.get()},
                                                      {FLYDSL_GDN_O_UID, o->get()}};
    std::unique_ptr<DeviceOperand> aLog;
    std::unique_ptr<DeviceOperand> dtBias;
    std::unique_ptr<DeviceOperand> initialState;
    std::unique_ptr<DeviceOperand> finalState;
    std::unique_ptr<DeviceBuffer> indices;
    if(spec.aLog)
    {
        aLog = device(FLYDSL_GDN_A_LOG_UID, problem.aLog, spec.paramType);
        arguments.push_back({FLYDSL_GDN_A_LOG_UID, aLog->get()});
    }
    if(spec.dtBias)
    {
        dtBias = device(FLYDSL_GDN_DT_BIAS_UID, problem.dtBias, spec.paramType);
        arguments.push_back({FLYDSL_GDN_DT_BIAS_UID, dtBias->get()});
    }
    if(spec.initialState)
    {
        initialState = device(FLYDSL_GDN_INITIAL_STATE_UID, problem.initialState, spec.stateType);
        arguments.push_back({FLYDSL_GDN_INITIAL_STATE_UID, initialState->get()});
    }
    if(spec.finalState)
    {
        // In place: one buffer bound to both uids, as overwrite_initial_state asks.
        if(spec.overwriteInitialState)
        {
            arguments.push_back({FLYDSL_GDN_FINAL_STATE_UID, initialState->get()});
        }
        else
        {
            finalState
                = device(FLYDSL_GDN_FINAL_STATE_UID, problem.finalStatePrior, spec.stateType);
            arguments.push_back({FLYDSL_GDN_FINAL_STATE_UID, finalState->get()});
        }
    }
    if(spec.stateIndices)
    {
        indices = std::make_unique<DeviceBuffer>(problem.stateIndices.data(),
                                                 problem.stateIndices.size() * sizeof(int32_t));
        arguments.push_back({FLYDSL_GDN_STATE_INDICES_UID, indices->get()});
    }

    const Handle handle;
    handler.launch(
        handle, *prepared, arguments.data(), static_cast<uint32_t>(arguments.size()), nullptr);
    EXPECT_EQ(hipSuccess, hipDeviceSynchronize());

    Outputs outputs;
    outputs.o = o->read();
    if(spec.finalState)
    {
        outputs.finalState = spec.overwriteInitialState ? initialState->read() : finalState->read();
    }
    return outputs;
}

template <typename T>
Tensor<T> packedTensor(const Operand& operand)
{
    return stridedTensor<T>(operand, flydslPackedStrides(operand.dims));
}

/// hipDNN's CPU reference on @p problem, f32 compute.
template <typename Io>
Outputs runReference(const Problem& problem)
{
    const auto& spec = problem.spec;
    auto q = packedTensor<Io>(problem.q);
    auto k = packedTensor<Io>(problem.k);
    auto v = packedTensor<Io>(problem.v);
    auto g = packedTensor<float>(problem.g);
    auto beta = packedTensor<float>(problem.beta);
    const Operand cuOperand{{static_cast<int64_t>(problem.cuSeqlens.size())}, {}};
    Tensor<int32_t> cu(cuOperand.dims);
    std::copy(problem.cuSeqlens.begin(), problem.cuSeqlens.end(), cu.memory().hostData());
    const auto oDims = std::vector<int64_t>{spec.tokens, spec.headsOut(), spec.valueDim};
    Tensor<Io> o(oDims);
    auto aLog = packedTensor<float>(problem.aLog);
    auto dtBias = packedTensor<float>(problem.dtBias);
    auto initialState = packedTensor<float>(problem.initialState);
    auto finalState = packedTensor<float>(spec.overwriteInitialState ? problem.initialState
                                                                     : problem.finalStatePrior);
    Tensor<int32_t> indices({std::max<int64_t>(1, spec.sequences)});
    std::copy(
        problem.stateIndices.begin(), problem.stateIndices.end(), indices.memory().hostData());

    GdnTensors<Io> tensors;
    tensors.q = &q;
    tensors.k = &k;
    tensors.v = &v;
    tensors.g = &g;
    tensors.beta = &beta;
    tensors.cuSeqlens = &cu;
    tensors.o = &o;
    tensors.aLog = spec.aLog ? &aLog : nullptr;
    tensors.dtBias = spec.dtBias ? &dtBias : nullptr;
    tensors.initialState = spec.initialState ? &initialState : nullptr;
    tensors.finalState = spec.finalState ? &finalState : nullptr;
    tensors.stateIndices = spec.stateIndices ? &indices : nullptr;

    GdnConfig config;
    config.scale = spec.scale;
    config.useQkL2norm = spec.useQkL2norm;
    config.useBetaSigmoid = spec.useBetaSigmoid;
    config.allowNegEigval = spec.allowNegEigval;
    config.safeGate = spec.safeGate;
    config.gateDomain = spec.gateDomain;
    CpuFpReferenceGdn::forward<float>(tensors, config);

    Outputs outputs;
    outputs.o = packedValues(o, oDims);
    if(spec.finalState)
    {
        outputs.finalState = packedValues(finalState, problem.finalStatePrior.dims);
    }
    return outputs;
}

/// Worst error relative to max(|expected|, floor * max|expected|); NaN is infinite.
void expectClose(const std::vector<float>& actual,
                 const std::vector<float>& expected,
                 double tolerance,
                 const std::string& what)
{
    ASSERT_EQ(actual.size(), expected.size()) << what;
    double peak = 0.0;
    for(const auto value : expected)
    {
        peak = std::max(peak, std::abs(static_cast<double>(value)));
    }
    const double floor = std::max(peak * 0.05, 1e-6);
    double worst = 0.0;
    size_t where = 0;
    for(size_t i = 0; i < actual.size(); ++i)
    {
        const auto got = static_cast<double>(actual[i]);
        const auto want = static_cast<double>(expected[i]);
        const double error = std::isfinite(got)
                                 ? std::abs(got - want) / std::max(std::abs(want), floor)
                                 : std::numeric_limits<double>::infinity();
        if(error > worst)
        {
            worst = error;
            where = i;
        }
    }
    EXPECT_LE(worst, tolerance) << what << ": worst at " << where << " got " << actual[where]
                                << " want " << expected[where];
}

std::vector<float>
    randomValues(std::mt19937& generator, size_t count, float low, float high, DataType type)
{
    std::uniform_real_distribution<float> uniform(low, high);
    std::vector<float> values(count);
    for(auto& value : values)
    {
        value = quantize(uniform(generator), type);
    }
    return values;
}

/// Operands for @p spec with @p lengths: values in ranges where the recurrence stays
/// bounded for the spec's flags.
Problem makeProblem(GdnGraphSpec spec, const std::vector<int32_t>& lengths, uint32_t seed)
{
    std::mt19937 generator(seed);
    Problem problem;
    problem.cuSeqlens = {0};
    for(const auto length : lengths)
    {
        problem.cuSeqlens.push_back(problem.cuSeqlens.back() + length);
    }
    spec.tokens = problem.cuSeqlens.back();
    spec.sequences = static_cast<int64_t>(lengths.size());
    const auto t = spec.tokens;
    const auto ho = spec.headsOut();
    const auto rows = spec.stateRows();

    // Without the norm, k is kept near unit length so I - beta k k^T stays contractive.
    const float qkRange
        = spec.useQkL2norm ? 1.0F : std::sqrt(3.0F / static_cast<float>(spec.keyDim));
    const auto make = [&](std::vector<int64_t> dims, float low, float high, DataType type) {
        const auto count = static_cast<size_t>(
            std::accumulate(dims.begin(), dims.end(), int64_t{1}, std::multiplies<>()));
        return Operand{std::move(dims), randomValues(generator, count, low, high, type)};
    };
    problem.q = make({t, spec.headsQ, spec.keyDim}, -qkRange, qkRange, spec.ioType);
    problem.k = make({t, spec.headsK, spec.keyDim}, -qkRange, qkRange, spec.ioType);
    problem.v = make({t, spec.headsV, spec.valueDim}, -1.0F, 1.0F, spec.ioType);
    if(spec.safeGate)
    {
        problem.g = make({t, ho}, -2.0F, 2.0F, spec.gType);
    }
    else if(spec.gateDomain == GateDomain::LINEAR)
    {
        problem.g = make({t, ho}, 0.5F, 1.0F, spec.gType);
    }
    else
    {
        problem.g = make({t, ho}, -0.5F, 0.0F, spec.gType);
    }
    problem.beta = spec.useBetaSigmoid ? make({t, ho}, -2.0F, 2.0F, spec.betaType)
                                       : make({t, ho}, 0.0F, 1.0F, spec.betaType);
    if(spec.allowNegEigval)
    {
        // 2 sigmoid(beta) up to 2 with a unit k: keep beta below 0 so 2 sigmoid < 1.
        problem.beta = make({t, ho}, -3.0F, 0.0F, spec.betaType);
    }
    problem.aLog = make({ho}, -1.0F, 0.5F, spec.paramType);
    problem.dtBias = make({ho}, -1.0F, 1.0F, spec.paramType);
    problem.initialState
        = make({rows, ho, spec.valueDim, spec.keyDim}, -0.5F, 0.5F, DataType::FLOAT);
    problem.finalStatePrior = Operand{problem.initialState.dims,
                                      std::vector<float>(problem.initialState.values.size(), 7.0F)};
    if(spec.stateIndices)
    {
        // Distinct rows of the pool, in no particular order.
        std::vector<int32_t> pool(static_cast<size_t>(rows));
        std::iota(pool.begin(), pool.end(), 0);
        std::shuffle(pool.begin(), pool.end(), generator);
        problem.stateIndices.assign(pool.begin(), pool.begin() + spec.sequences);
    }
    problem.spec = spec;
    return problem;
}

double outputTolerance(const GdnGraphSpec& spec)
{
    // o is rounded to bf16 (8 bits) or f16 (11 bits) of mantissa.
    return spec.ioType == DataType::BFLOAT16 ? 1.5e-2 : 4e-3;
}

constexpr double STATE_TOLERANCE = 1e-3;

template <typename Io>
void expectProblemMatches(const KernelDefinition& kernel, const Problem& problem, const char* what)
{
    SCOPED_TRACE(kernel.name + ": " + what);
    const auto gpu = runObject(kernel, problem);
    const auto reference = runReference<Io>(problem);
    expectClose(gpu.o, reference.o, outputTolerance(problem.spec), "o");
    if(problem.spec.finalState)
    {
        expectClose(gpu.finalState, reference.finalState, STATE_TOLERANCE, "final_state");
    }
}

/// The graph spec @p kernel is built for, with Qwen3.8's flags and a small GQA head layout:
/// 2 q/k heads, 4 v heads, so each q/k head serves two output heads.
GdnGraphSpec specFor(const KernelDefinition& kernel)
{
    GdnGraphSpec spec;
    spec.ioType = stringField(kernel, FLYDSL_GDN_DTYPE_FIELD) == "f16" ? DataType::HALF
                                                                       : DataType::BFLOAT16;
    spec.keyDim = intField(kernel, FLYDSL_GDN_HEAD_DIM_FIELD);
    spec.valueDim = intField(kernel, FLYDSL_GDN_HEAD_DIM_V_FIELD);
    spec.headsQ = 2;
    spec.headsK = 2;
    spec.headsV = 4;
    if(intField(kernel, FLYDSL_GDN_STATE_STRIDED_FIELD) == 1)
    {
        const auto kMajor
            = flydslGdnKMajorStateStrides(spec.headsOut(), spec.valueDim, spec.keyDim);
        spec.strides[FLYDSL_GDN_INITIAL_STATE_UID] = kMajor;
        spec.strides[FLYDSL_GDN_FINAL_STATE_UID] = kMajor;
    }
    return spec;
}

/// The packed rows @p tokens of @p operand (axis 0 is the token axis).
Operand tokenRows(const Operand& operand, const std::vector<int32_t>& tokens)
{
    Operand rows;
    rows.dims = operand.dims;
    rows.dims[0] = static_cast<int64_t>(tokens.size());
    const auto width = operand.values.size() / static_cast<size_t>(operand.dims[0]);
    for(const auto token : tokens)
    {
        const auto first = operand.values.begin()
                           + static_cast<std::ptrdiff_t>(static_cast<size_t>(token) * width);
        rows.values.insert(rows.values.end(), first, first + static_cast<std::ptrdiff_t>(width));
    }
    return rows;
}

/// @p whole restricted to @p tokens, sequence by sequence.
Problem subProblem(const Problem& whole,
                   const std::vector<int32_t>& tokens,
                   const std::vector<int32_t>& lengths)
{
    Problem part = whole;
    part.cuSeqlens = {0};
    for(const auto length : lengths)
    {
        part.cuSeqlens.push_back(part.cuSeqlens.back() + length);
    }
    part.spec.tokens = part.cuSeqlens.back();
    part.q = tokenRows(whole.q, tokens);
    part.k = tokenRows(whole.k, tokens);
    part.v = tokenRows(whole.v, tokens);
    part.g = tokenRows(whole.g, tokens);
    part.beta = tokenRows(whole.beta, tokens);
    return part;
}

/// A prefill of every sequence but its last token, then a decode of that token from the
/// prefill's final state, against one reference run over the whole sequences.
template <typename Io>
void expectCarriedStateMatches(const KernelDefinition& kernel, const std::vector<int32_t>& lengths)
{
    SCOPED_TRACE(kernel.name + ": prefill then decode from its state");
    const auto whole = makeProblem(specFor(kernel), lengths, 31);
    const auto reference = runReference<Io>(whole);

    std::vector<int32_t> prefillTokens;
    std::vector<int32_t> prefillLengths;
    std::vector<int32_t> decodeTokens;
    for(size_t n = 0; n + 1 < whole.cuSeqlens.size(); ++n)
    {
        for(auto token = whole.cuSeqlens[n]; token + 1 < whole.cuSeqlens[n + 1]; ++token)
        {
            prefillTokens.push_back(token);
        }
        prefillLengths.push_back(whole.cuSeqlens[n + 1] - whole.cuSeqlens[n] - 1);
        decodeTokens.push_back(whole.cuSeqlens[n + 1] - 1);
    }

    const auto prefill = subProblem(whole, prefillTokens, prefillLengths);
    const auto prefillOut = runObject(kernel, prefill);
    ASSERT_FALSE(prefillOut.finalState.empty());

    auto decode = subProblem(whole, decodeTokens, std::vector<int32_t>(decodeTokens.size(), 1));
    decode.initialState.values = prefillOut.finalState;
    const auto decodeOut = runObject(kernel, decode);

    const auto rowWidth = static_cast<size_t>(whole.spec.headsOut() * whole.spec.valueDim);
    std::vector<float> wantPrefill;
    std::vector<float> wantDecode;
    for(const auto token : prefillTokens)
    {
        const auto first = reference.o.begin()
                           + static_cast<std::ptrdiff_t>(static_cast<size_t>(token) * rowWidth);
        wantPrefill.insert(wantPrefill.end(), first, first + static_cast<std::ptrdiff_t>(rowWidth));
    }
    for(const auto token : decodeTokens)
    {
        const auto first = reference.o.begin()
                           + static_cast<std::ptrdiff_t>(static_cast<size_t>(token) * rowWidth);
        wantDecode.insert(wantDecode.end(), first, first + static_cast<std::ptrdiff_t>(rowWidth));
    }
    const auto tolerance = outputTolerance(whole.spec);
    expectClose(prefillOut.o, wantPrefill, tolerance, "prefill o");
    expectClose(decodeOut.o, wantDecode, tolerance, "decode o");
    expectClose(decodeOut.finalState, reference.finalState, STATE_TOLERANCE, "carried state");
}

template <typename Io>
void expectVariantsMatch(const KernelDefinition& kernel)
{
    const auto variant = [&](const char* what,
                             const std::function<void(GdnGraphSpec&)>& change,
                             const std::vector<int32_t>& lengths = {6, 1, 9}) {
        auto spec = specFor(kernel);
        change(spec);
        expectProblemMatches<Io>(kernel, makeProblem(spec, lengths, 7), what);
    };

    variant("Qwen3.8 heads, prefill",
            [](GdnGraphSpec& spec) {
                spec.headsQ = 16;
                spec.headsK = 16;
                spec.headsV = 48;
            },
            {33, 2});
    variant("Qwen3.8 heads, decode",
            [](GdnGraphSpec& spec) {
                spec.headsQ = 16;
                spec.headsK = 16;
                spec.headsV = 48;
            },
            {1, 1, 1});
    variant("LOG gate, raw beta, no norm, explicit scale", [](GdnGraphSpec& spec) {
        spec.safeGate = false;
        spec.aLog = false;
        spec.dtBias = false;
        spec.useBetaSigmoid = false;
        spec.useQkL2norm = false;
        spec.scale = 0.3F;
    });
    variant("LINEAR gate", [](GdnGraphSpec& spec) {
        spec.safeGate = false;
        spec.aLog = false;
        spec.dtBias = false;
        spec.gateDomain = GateDomain::LINEAR;
    });
    variant("allow_neg_eigval", [](GdnGraphSpec& spec) { spec.allowNegEigval = true; });
    variant("safe gate without a_log or dt_bias", [](GdnGraphSpec& spec) {
        spec.aLog = false;
        spec.dtBias = false;
    });
    variant("bf16 gates and parameters", [](GdnGraphSpec& spec) {
        spec.gType = DataType::BFLOAT16;
        spec.betaType = DataType::BFLOAT16;
        spec.paramType = DataType::BFLOAT16;
    });
    variant("f16 gates and parameters", [](GdnGraphSpec& spec) {
        spec.gType = DataType::HALF;
        spec.betaType = DataType::FLOAT;
        spec.paramType = DataType::HALF;
    });
    variant("no initial state", [](GdnGraphSpec& spec) { spec.initialState = false; });
    variant("no final state", [](GdnGraphSpec& spec) { spec.finalState = false; });
    variant("state pool, in place", [](GdnGraphSpec& spec) {
        spec.stateIndices = true;
        spec.poolRows = 7;
        spec.overwriteInitialState = true;
    });
    variant("state pool", [](GdnGraphSpec& spec) {
        spec.stateIndices = true;
        spec.poolRows = 5;
    });
    variant("q, k and v sliced from one projection row", [](GdnGraphSpec& spec) {
        // [T, 2 * H * K + HV * V]: q, then k, then v.
        const auto row = 2 * spec.headsQ * spec.keyDim + spec.headsV * spec.valueDim;
        spec.strides[FLYDSL_GDN_Q_UID] = {row, spec.keyDim, 1};
        spec.strides[FLYDSL_GDN_K_UID] = {row, spec.keyDim, 1};
        spec.strides[FLYDSL_GDN_V_UID] = {row, spec.valueDim, 1};
        // g and beta as [T, 2 * HO] interleaved columns.
        spec.strides[FLYDSL_GDN_G_UID] = {2 * spec.headsOut(), 2};
        spec.strides[FLYDSL_GDN_BETA_UID] = {2 * spec.headsOut(), 2};
    });
}

template <typename Io>
void expectObjectMatches(const KernelDefinition& kernel)
{
    expectProblemMatches<Io>(kernel, makeProblem(specFor(kernel), {5, 0, 17}, 3), "varlen prefill");
    expectProblemMatches<Io>(kernel, makeProblem(specFor(kernel), {1, 1, 1, 1}, 5), "decode");
    expectCarriedStateMatches<Io>(kernel, {9, 4});

    if(intField(kernel, FLYDSL_GDN_STATE_STRIDED_FIELD) == 0
       && intField(kernel, FLYDSL_GDN_HEAD_DIM_FIELD) == 128
       && intField(kernel, FLYDSL_GDN_HEAD_DIM_V_FIELD) == 128)
    {
        expectVariantsMatch<Io>(kernel);
    }
}

std::vector<KernelDefinition> stagedGdnKernels()
{
    auto kernels = stagedKernels(FLYDSL_GDN.engineName);
    std::sort(kernels.begin(), kernels.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.name < rhs.name;
    });
    return kernels;
}

TEST(TestGpuFlydslGdnObjects, EveryObjectMatchesTheCpuReference)
{
    SKIP_IF_NO_DEVICES();
    const auto kernels = stagedGdnKernels();
    if(kernels.empty())
    {
        reportUnstaged("gdn", "this build staged no gdn objects for this device");
        return;
    }
    for(const auto& kernel : kernels)
    {
        const auto dtype = stringField(kernel, FLYDSL_GDN_DTYPE_FIELD);
        if(dtype == "bf16")
        {
            expectObjectMatches<Bf16>(kernel);
        }
        else if(dtype == "f16")
        {
            expectObjectMatches<F16>(kernel);
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
