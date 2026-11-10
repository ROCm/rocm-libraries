// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see TestFlydslRmsNormEngine.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hip/hip_runtime_api.h>

#include <hipdnn_data_sdk/types/Bfloat16.hpp>
#include <hipdnn_data_sdk/types/Half.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/pointwise/CpuReferencePointwise.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslNormDeviceTestSupport.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslPointwiseTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslRmsNormTestGraphs.hpp"

/**
 * @file TestFlydslPointwiseObjects.cpp
 * @brief The flyDSL pointwise objects launched on this device against the hipDNN CPU
 *        reference.
 *
 * Two suites:
 *
 *  - `TestGpuFlydslPointwiseDispatch` selects the object for a graph the way the engine
 *    does -- graph match, then the one staged object whose metadata the kernel matcher
 *    admits -- and runs every operation and the broadcast patterns the Qwen3.8 graphs use:
 *    [1,T,H] + [1,T,H], [1,T,k,H] * [1,T,k,1] (bf16 and f32 weights), [T,H] * [T,H],
 *    [T,k] / [T,1], a broadcast first input, a lower-rank input, a BSHD output view, the
 *    dtype casts and the swish / softplus betas.
 *  - `TestGpuFlydslPointwiseObjects` launches each staged object by name over a dense
 *    shape (the vector path, with a partial last vector), a broadcast one and a strided
 *    one (the per-element path). `OnePerForm` runs one object per (operation, input
 *    dtype); `Full/...Every` runs all of them.
 *
 * The reference is hipDNN's CPU pointwise (CpuReferencePointwiseImpl), in double. It does
 * not implement EXP or SOFTPLUS_FWD, so those two are computed here in double from the
 * same quantised inputs. Outputs start as NaN, so an element no thread wrote fails.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

using data_objects::DataType;
using data_objects::PointwiseMode;
using Bf16 = hipdnn_data_sdk::types::bfloat16;
using F16 = hipdnn_data_sdk::types::half;
template <typename T>
using HostTensor = hipdnn_data_sdk::utilities::Tensor<T>;

constexpr const char* OP = "pointwise";

/// One operand's layout; empty strides mean packed row-major.
struct Layout
{
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;

    std::vector<int64_t> resolvedStrides() const
    {
        return strides.empty() ? flydslPointwisePackedStrides(dims) : strides;
    }
};

/// One launch: an operation over its operands' layouts.
struct Problem
{
    PointwiseMode mode = PointwiseMode::ADD;
    Layout in0;
    std::optional<Layout> in1;
    Layout out;
    std::optional<float> beta;
};

template <typename T>
constexpr DataType dataTypeOf()
{
    if constexpr(std::is_same_v<T, Bf16>)
    {
        return DataType::BFLOAT16;
    }
    else if constexpr(std::is_same_v<T, F16>)
    {
        return DataType::HALF;
    }
    else
    {
        return DataType::FLOAT;
    }
}

/// The tolerance for an output type: a little over one unit in the last place, relative to
/// max(|expected|, 1). The objects compute in f32 and round once, the reference in double.
template <typename T>
double toleranceFor()
{
    if constexpr(std::is_same_v<T, Bf16>)
    {
        return 1e-2;
    }
    else if constexpr(std::is_same_v<T, F16>)
    {
        return 2e-3;
    }
    else
    {
        return 1e-5;
    }
}

/// Calls @p visit with every index of @p dims, outermost axis slowest.
template <typename Visit>
void forEachIndex(const std::vector<int64_t>& dims, Visit&& visit)
{
    std::vector<int64_t> index(dims.size(), 0);
    if(std::any_of(dims.begin(), dims.end(), [](int64_t extent) { return extent <= 0; }))
    {
        return;
    }
    while(true)
    {
        visit(index);
        size_t axis = dims.size();
        while(axis > 0)
        {
            --axis;
            if(++index[axis] < dims[axis])
            {
                break;
            }
            index[axis] = 0;
            if(axis == 0)
            {
                return;
            }
        }
        if(dims.empty())
        {
            return;
        }
    }
}

/// @p index of the output, as the index of an input of @p dims (aligned at the innermost
/// axis, broadcast where the input's extent is 1).
std::vector<int64_t> broadcastIndex(const std::vector<int64_t>& index,
                                    const std::vector<int64_t>& dims)
{
    std::vector<int64_t> result(dims.size());
    const auto offset = index.size() - dims.size();
    for(size_t axis = 0; axis < dims.size(); ++axis)
    {
        result[axis] = dims[axis] == 1 ? 0 : index[offset + axis];
    }
    return result;
}

double asDouble(float value)
{
    return static_cast<double>(value);
}

/// The CPU reference into @p expected.
template <typename A, typename B, typename O>
void computeReference(const Problem& problem,
                      HostTensor<O>& expected,
                      const HostTensor<A>& in0,
                      const HostTensor<B>* in1)
{
    using hipdnn_test_sdk::utilities::CpuReferencePointwiseImpl;
    const auto beta = static_cast<double>(problem.beta.value_or(1.0F));
    switch(problem.mode)
    {
    case PointwiseMode::EXP:
    case PointwiseMode::SOFTPLUS_FWD:
        // Not in the CPU reference: the same double-precision formula it would apply.
        forEachIndex(problem.out.dims, [&](const std::vector<int64_t>& index) {
            const double x = asDouble(
                static_cast<float>(in0.getHostValue(broadcastIndex(index, problem.in0.dims))));
            const double y = problem.mode == PointwiseMode::EXP
                                 ? std::exp(x)
                                 : std::log1p(std::exp(beta * x)) / beta;
            expected.setHostValue(static_cast<O>(static_cast<float>(y)), index);
        });
        expected.memory().markHostModified();
        break;
    case PointwiseMode::SWISH_FWD:
        CpuReferencePointwiseImpl<O, A>::pointwiseCompute(
            problem.mode, expected, in0, 0.0F, 0.0F, 0.0F, static_cast<float>(beta));
        break;
    case PointwiseMode::ADD:
    case PointwiseMode::SUB:
    case PointwiseMode::MUL:
    case PointwiseMode::DIV:
        CpuReferencePointwiseImpl<O, A, B>::pointwiseCompute(problem.mode, expected, in0, *in1);
        break;
    default:
        CpuReferencePointwiseImpl<O, A>::pointwiseCompute(problem.mode, expected, in0);
        break;
    }
}

/// The staged objects the kernel matcher admits for @p fixture's graph.
std::vector<KernelDefinition> admittedKernels(const GraphFixture& fixture,
                                              const BoundTokens& bound,
                                              const std::vector<KernelDefinition>& staged)
{
    std::vector<KernelDefinition> admitted;
    for(const auto& kernel : staged)
    {
        if(matchesKernel(FLYDSL_POINTWISE, fixture.context(), kernel, bound))
        {
            admitted.push_back(kernel);
        }
    }
    return admitted;
}

/**
 * @brief Runs @p problem on the device and compares it with the CPU reference.
 *
 * @param named The object to launch; when absent, the one the kernel matcher admits among
 *        the staged objects, which must be exactly one.
 */
template <typename A, typename B, typename O>
void expectMatchesReference(const Problem& problem,
                            const std::optional<KernelDefinition>& named = std::nullopt)
{
    const bool binary = problem.in1.has_value();
    PointwiseGraphSpec spec;
    spec.mode = problem.mode;
    spec.in0 = flydslOperand(dataTypeOf<A>(), problem.in0.dims, problem.in0.resolvedStrides());
    if(binary)
    {
        spec.in1
            = flydslOperand(dataTypeOf<B>(), problem.in1->dims, problem.in1->resolvedStrides());
    }
    spec.out = flydslOperand(dataTypeOf<O>(), problem.out.dims, problem.out.resolvedStrides());
    if(problem.mode == PointwiseMode::SWISH_FWD)
    {
        spec.swishBeta = problem.beta;
    }
    if(problem.mode == PointwiseMode::SOFTPLUS_FWD)
    {
        spec.softplusBeta = problem.beta;
    }

    const GraphFixture fixture(buildFlydslPointwiseGraph(spec), packedArchDeviceProperties());
    const auto bound = matchesGraph(FLYDSL_POINTWISE, fixture.context());
    ASSERT_TRUE(bound.has_value()) << "the graph is declined";

    KernelDefinition kernel;
    if(named.has_value())
    {
        kernel = *named;
        ASSERT_TRUE(matchesKernel(FLYDSL_POINTWISE, fixture.context(), kernel, *bound))
            << kernel.name << " does not admit the graph built from its own metadata";
    }
    else
    {
        const auto staged = stagedKernels(FLYDSL_POINTWISE.engineName);
        if(staged.empty())
        {
            reportUnstaged(OP, "this build staged no pointwise objects for this device");
            return;
        }
        const auto admitted = admittedKernels(fixture, *bound, staged);
        ASSERT_EQ(admitted.size(), 1U) << "exactly one object serves each graph";
        kernel = admitted.front();
    }
    SCOPED_TRACE(kernel.name);

    HostTensor<A> in0(problem.in0.dims, problem.in0.resolvedStrides());
    std::optional<HostTensor<B>> in1;
    HostTensor<O> out(problem.out.dims, problem.out.resolvedStrides());
    HostTensor<O> expected(problem.out.dims, problem.out.resolvedStrides());
    const auto seed = static_cast<unsigned int>(problem.out.dims.back() * 31 + 7);
    in0.fillWithRandomValues(A(-4.0F), A(4.0F), seed);
    if(binary)
    {
        in1.emplace(problem.in1->dims, problem.in1->resolvedStrides());
        // Away from zero, so DIV's quotients stay representable.
        in1->fillWithRandomValues(B(0.25F), B(2.0F), seed + 1);
        if(problem.mode != PointwiseMode::DIV)
        {
            in1->fillWithRandomValues(B(-2.0F), B(2.0F), seed + 1);
        }
    }

    const auto& handler = dispatchHandler(FLYDSL_POINTWISE);
    const auto prepared = handler.prepare(fixture.context(), *bound, kernel);
    ASSERT_NE(prepared, nullptr);

    void* outDevice = out.memory().deviceData();
    ASSERT_EQ(hipSuccess, hipMemset(outDevice, 0xFF, out.memory().count() * sizeof(O)));
    std::vector<hipdnnPluginDeviceBuffer_t> buffers{
        {FLYDSL_POINTWISE_IN0_UID, in0.memory().deviceData()},
        {FLYDSL_POINTWISE_OUT_UID, outDevice}};
    if(binary)
    {
        buffers.push_back({FLYDSL_POINTWISE_IN1_UID, in1->memory().deviceData()});
    }
    const Handle handle;
    handler.launch(
        handle, *prepared, buffers.data(), static_cast<uint32_t>(buffers.size()), nullptr);
    ASSERT_EQ(hipSuccess, hipDeviceSynchronize());
    out.memory().markDeviceModified();
    // Non-const access copies the device result back; the reads below are const.
    static_cast<void>(out.memory().hostData());

    computeReference<A, B, O>(problem, expected, in0, binary ? &*in1 : nullptr);

    double worst = 0.0;
    std::vector<int64_t> worstIndex;
    float worstGot = 0.0F;
    float worstExpected = 0.0F;
    forEachIndex(problem.out.dims, [&](const std::vector<int64_t>& index) {
        const auto got = static_cast<float>(out.getHostValue(index));
        const auto want = static_cast<float>(expected.getHostValue(index));
        const double error = std::isfinite(got) ? std::abs(asDouble(got) - asDouble(want))
                                                      / std::max(std::abs(asDouble(want)), 1.0)
                                                : std::numeric_limits<double>::infinity();
        if(error > worst || worstIndex.empty())
        {
            worst = std::max(worst, error);
            worstIndex = index;
            worstGot = got;
            worstExpected = want;
        }
    });
    std::string where;
    for(const auto coordinate : worstIndex)
    {
        where += std::to_string(coordinate) + ",";
    }
    EXPECT_LE(worst, toleranceFor<O>())
        << "worst at [" << where << "]: got " << worstGot << ", expected " << worstExpected;
}

Problem unary(PointwiseMode mode, const std::vector<int64_t>& dims)
{
    return {mode, {dims, {}}, std::nullopt, {dims, {}}, std::nullopt};
}

Problem binary(PointwiseMode mode,
               const std::vector<int64_t>& in0,
               const std::vector<int64_t>& in1,
               const std::vector<int64_t>& out)
{
    return {mode, {in0, {}}, Layout{in1, {}}, {out, {}}, std::nullopt};
}

constexpr int64_t T = 37;
constexpr int64_t H = 1031;
constexpr int64_t K = 8;

// ---------------------------------------------------------------------------
// TestGpuFlydslPointwiseDispatch
// ---------------------------------------------------------------------------

TEST(TestGpuFlydslPointwiseDispatch, EveryUnaryOperationInBf16)
{
    SKIP_IF_NO_DEVICES();
    for(const auto mode : {PointwiseMode::SIGMOID_FWD,
                           PointwiseMode::SWISH_FWD,
                           PointwiseMode::EXP,
                           PointwiseMode::SOFTPLUS_FWD,
                           PointwiseMode::NEG,
                           PointwiseMode::IDENTITY})
    {
        SCOPED_TRACE(static_cast<int>(mode));
        expectMatchesReference<Bf16, Bf16, Bf16>(unary(mode, {1, T, H}));
    }
}

TEST(TestGpuFlydslPointwiseDispatch, EveryUnaryOperationInF32AndF16)
{
    SKIP_IF_NO_DEVICES();
    for(const auto mode : {PointwiseMode::SIGMOID_FWD,
                           PointwiseMode::SWISH_FWD,
                           PointwiseMode::EXP,
                           PointwiseMode::SOFTPLUS_FWD,
                           PointwiseMode::NEG,
                           PointwiseMode::IDENTITY})
    {
        SCOPED_TRACE(static_cast<int>(mode));
        expectMatchesReference<float, float, float>(unary(mode, {T, 129}));
        expectMatchesReference<F16, F16, F16>(unary(mode, {T, 129}));
    }
}

TEST(TestGpuFlydslPointwiseDispatch, EveryBinaryOperationOverTwoSameShapeRows)
{
    SKIP_IF_NO_DEVICES();
    for(const auto mode :
        {PointwiseMode::ADD, PointwiseMode::SUB, PointwiseMode::MUL, PointwiseMode::DIV})
    {
        SCOPED_TRACE(static_cast<int>(mode));
        expectMatchesReference<Bf16, Bf16, Bf16>(binary(mode, {T, H}, {T, H}, {T, H}));
        expectMatchesReference<float, float, float>(binary(mode, {T, H}, {T, H}, {T, H}));
    }
}

TEST(TestGpuFlydslPointwiseDispatch, AddsTheResidual)
{
    SKIP_IF_NO_DEVICES();
    expectMatchesReference<Bf16, Bf16, Bf16>(
        binary(PointwiseMode::ADD, {1, T, H}, {1, T, H}, {1, T, H}));
}

TEST(TestGpuFlydslPointwiseDispatch, ScalesEachExpertRowByItsRoutingWeight)
{
    SKIP_IF_NO_DEVICES();
    const auto problem = binary(PointwiseMode::MUL, {1, T, K, H}, {1, T, K, 1}, {1, T, K, H});
    expectMatchesReference<Bf16, Bf16, Bf16>(problem);
    expectMatchesReference<Bf16, float, Bf16>(problem);
    expectMatchesReference<float, float, float>(problem);
}

TEST(TestGpuFlydslPointwiseDispatch, MultipliesTheGate)
{
    SKIP_IF_NO_DEVICES();
    expectMatchesReference<Bf16, Bf16, Bf16>(binary(PointwiseMode::MUL, {T, H}, {T, H}, {T, H}));
}

TEST(TestGpuFlydslPointwiseDispatch, RenormalisesTheTopKWeights)
{
    SKIP_IF_NO_DEVICES();
    expectMatchesReference<float, float, float>(binary(PointwiseMode::DIV, {T, K}, {T, 1}, {T, K}));
}

TEST(TestGpuFlydslPointwiseDispatch, BroadcastsTheFirstInput)
{
    SKIP_IF_NO_DEVICES();
    expectMatchesReference<Bf16, Bf16, Bf16>(binary(PointwiseMode::SUB, {1, H}, {T, H}, {T, H}));
}

TEST(TestGpuFlydslPointwiseDispatch, BroadcastsALowerRankInputAndBothInputs)
{
    SKIP_IF_NO_DEVICES();
    expectMatchesReference<F16, F16, F16>(binary(PointwiseMode::ADD, {T, H}, {H}, {T, H}));
    expectMatchesReference<float, Bf16, float>(binary(PointwiseMode::MUL, {T, 1}, {1, H}, {T, H}));
}

TEST(TestGpuFlydslPointwiseDispatch, WritesABshdViewOfABhsdOutput)
{
    SKIP_IF_NO_DEVICES();
    Problem problem = binary(PointwiseMode::MUL, {2, 4, 5, 64}, {2, 4, 5, 64}, {2, 4, 5, 64});
    problem.out.strides = {int64_t{4} * 5 * 64, 64, int64_t{4} * 64, 1};
    problem.in1->strides = {int64_t{4} * 5 * 64, 64, int64_t{4} * 64, 1};
    expectMatchesReference<Bf16, Bf16, Bf16>(problem);
}

TEST(TestGpuFlydslPointwiseDispatch, CastsBetweenEveryPairOfTypes)
{
    SKIP_IF_NO_DEVICES();
    const auto problem = unary(PointwiseMode::IDENTITY, {T, H});
    expectMatchesReference<Bf16, Bf16, float>(problem);
    expectMatchesReference<float, float, Bf16>(problem);
    expectMatchesReference<F16, F16, Bf16>(problem);
    expectMatchesReference<Bf16, Bf16, F16>(problem);
    expectMatchesReference<float, float, F16>(problem);
    expectMatchesReference<F16, F16, float>(problem);
}

TEST(TestGpuFlydslPointwiseDispatch, AppliesTheSwishAndSoftplusBetas)
{
    SKIP_IF_NO_DEVICES();
    auto swish = unary(PointwiseMode::SWISH_FWD, {T, 129});
    swish.beta = 1.702F;
    expectMatchesReference<float, float, float>(swish);
    auto softplus = unary(PointwiseMode::SOFTPLUS_FWD, {T, 129});
    softplus.beta = 2.5F;
    expectMatchesReference<float, float, float>(softplus);
}

TEST(TestGpuFlydslPointwiseDispatch, ComputesASingleElement)
{
    SKIP_IF_NO_DEVICES();
    expectMatchesReference<float, float, float>(binary(PointwiseMode::DIV, {1}, {1}, {1}));
}

// ---------------------------------------------------------------------------
// TestGpuFlydslPointwiseObjects
// ---------------------------------------------------------------------------

enum class ObjectSelection
{
    ONE_PER_FORM,
    EVERY
};

std::string stringField(const KernelDefinition& kernel, const char* field)
{
    const auto* value = tryGetMetadataField<std::string>(kernel.metadata, field);
    return value == nullptr ? std::string() : *value;
}

std::optional<PointwiseMode> modeOf(const std::string& operation)
{
    const std::vector<std::pair<const char*, PointwiseMode>> modes{
        {"sigmoid", PointwiseMode::SIGMOID_FWD},
        {"swish", PointwiseMode::SWISH_FWD},
        {"exp", PointwiseMode::EXP},
        {"softplus", PointwiseMode::SOFTPLUS_FWD},
        {"neg", PointwiseMode::NEG},
        {"identity", PointwiseMode::IDENTITY},
        {"add", PointwiseMode::ADD},
        {"sub", PointwiseMode::SUB},
        {"mul", PointwiseMode::MUL},
        {"div", PointwiseMode::DIV}};
    for(const auto& [name, mode] : modes)
    {
        if(operation == name)
        {
            return mode;
        }
    }
    return std::nullopt;
}

/// The problems each object runs: dense (vector path, partial last vector), broadcast and
/// strided (per-element path).
std::vector<Problem> objectProblems(PointwiseMode mode, bool isBinary)
{
    if(isBinary)
    {
        Problem strided = binary(mode, {2, 4, 5, 33}, {2, 1, 5, 33}, {2, 4, 5, 33});
        strided.out.strides = {int64_t{4} * 5 * 33, 33, int64_t{4} * 33, 1};
        return {binary(mode, {3, H}, {3, H}, {3, H}),
                binary(mode, {1, 5, 3, 129}, {1, 5, 3, 1}, {1, 5, 3, 129}),
                strided};
    }
    Problem strided = unary(mode, {2, 4, 5, 33});
    strided.out.strides = {int64_t{4} * 5 * 33, 33, int64_t{4} * 33, 1};
    auto withBeta = [mode](Problem problem) {
        if(mode == PointwiseMode::SWISH_FWD || mode == PointwiseMode::SOFTPLUS_FWD)
        {
            problem.beta = 1.5F;
        }
        return problem;
    };
    return {unary(mode, {3, H}), withBeta(unary(mode, {5, 129})), strided};
}

template <typename A, typename B, typename O>
void runObject(const KernelDefinition& kernel, PointwiseMode mode, bool isBinary)
{
    for(const auto& problem : objectProblems(mode, isBinary))
    {
        expectMatchesReference<A, B, O>(problem, kernel);
    }
}

/// Calls @p visit with a value of the host type a metadata dtype names.
template <typename Visit>
void withHostType(const std::string& dtype, Visit&& visit)
{
    if(dtype == "bf16")
    {
        visit(Bf16{});
    }
    else if(dtype == "f16")
    {
        visit(F16{});
    }
    else if(dtype == "f32")
    {
        visit(float{});
    }
    else
    {
        ADD_FAILURE() << "unknown dtype '" << dtype << "'";
    }
}

class TestGpuFlydslPointwiseObjects : public ::testing::TestWithParam<ObjectSelection>
{
};

TEST_P(TestGpuFlydslPointwiseObjects, MatchTheReference)
{
    SKIP_IF_NO_DEVICES();
    auto staged = stagedKernels(FLYDSL_POINTWISE.engineName);
    if(staged.empty())
    {
        reportUnstaged(OP, "this build staged no pointwise objects for this device");
        return;
    }
    std::sort(staged.begin(), staged.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.name < rhs.name;
    });

    std::vector<std::pair<std::string, std::string>> seen;
    int launched = 0;
    for(const auto& kernel : staged)
    {
        const auto operation = stringField(kernel, FLYDSL_POINTWISE_OPERATION_FIELD);
        const auto in0 = stringField(kernel, FLYDSL_POINTWISE_DTYPE_FIELD);
        const auto in1 = stringField(kernel, FLYDSL_POINTWISE_IN1_DTYPE_FIELD);
        const auto out = stringField(kernel, FLYDSL_POINTWISE_OUT_DTYPE_FIELD);
        if(GetParam() == ObjectSelection::ONE_PER_FORM)
        {
            const auto form = std::make_pair(operation, in0);
            if(std::find(seen.begin(), seen.end(), form) != seen.end())
            {
                continue;
            }
            seen.push_back(form);
        }
        const auto mode = modeOf(operation);
        ASSERT_TRUE(mode.has_value()) << kernel.name << " declares operation '" << operation << "'";
        const bool isBinary = in1 != "none";
        SCOPED_TRACE(kernel.name);
        withHostType(in0, [&](auto a) {
            withHostType(isBinary ? in1 : in0, [&](auto b) {
                withHostType(out, [&](auto o) {
                    runObject<decltype(a), decltype(b), decltype(o)>(kernel, *mode, isBinary);
                });
            });
        });
        ++launched;
    }
    RecordProperty("objects", launched);
}

std::string selectionName(const ::testing::TestParamInfo<ObjectSelection>& info)
{
    return info.param == ObjectSelection::EVERY ? "Every" : "OnePerForm";
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuFlydslPointwiseObjects,
                         ::testing::Values(ObjectSelection::ONE_PER_FORM),
                         selectionName);

INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuFlydslPointwiseObjects,
                         ::testing::Values(ObjectSelection::EVERY),
                         selectionName);

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
