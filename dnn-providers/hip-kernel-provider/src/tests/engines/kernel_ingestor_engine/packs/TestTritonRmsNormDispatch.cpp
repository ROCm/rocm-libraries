// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Adapted from dnn-providers/hip-kernel-provider/src/tests/engines/kernel_ingestor_engine/packs/TestFlydslRmsNormEngine.cpp at commit 03911535bd0 (FlyDSL provider)

// Redundant with the CMake gate on purpose: if this file lands on an ungated source list
// it compiles to an empty translation unit instead of a suite reading an unstaged shard.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_TRITON)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hip/hip_runtime_api.h>

#include <hipdnn_data_sdk/types/Bfloat16.hpp>
#include <hipdnn_data_sdk/types/Half.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/packs/TritonRmsNormTestGraphs.hpp"

/**
 * @file TestTritonRmsNormDispatch.cpp
 * @brief Launches the Triton RMS-norm kernels from the staged archive and compares the
 *        output with a host reference.
 *
 * Kernels are chosen from the staged descriptors as the engine chooses them (kernel
 * matcher, then score). Every case skips without a device or without a staged object for
 * the device's arch.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

using data_objects::DataType;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

BoundTokens bindingsFor(const MatchContext& context)
{
    auto bound = matchesGraph(TRITON_RMSNORM, context);
    if(!bound.has_value())
    {
        throw std::logic_error("test graph does not match the pack it is dispatched against");
    }
    return std::move(*bound);
}

/// The current device with the feature suffix stripped: hipGetDeviceProperties reports
/// `gfx942:sramecc+:xnack-`, but shards are named by the bare arch and prepare() passes
/// this string to the archive lookup.
DeviceProperties packedArchDeviceProperties()
{
    auto properties = currentDeviceProperties();
    const auto colon = properties.gcnArchName.find(':');
    if(colon != std::string::npos)
    {
        properties.gcnArchName.resize(colon);
    }
    return properties;
}

/// Why no staged kernel is available for this device, or empty when some is.
std::string whyNoStagedKernel()
{
    const auto* staged = findLoadedSet(TRITON_RMSNORM.engineName);
    if(staged == nullptr)
    {
        return "this build staged no Triton RMSNorm descriptor set";
    }
    const auto device = packedArchDeviceProperties().gcnArchName;
    for(const auto& pack : staged->packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const auto& arches = kernel.arch.empty() ? pack.arch : kernel.arch;
            if(std::find(arches.begin(), arches.end(), device) != arches.end())
            {
                return {};
            }
        }
    }
    return "this build's descriptor root has no Triton RMSNorm object for " + device;
}

/// The kernel the engine would run for @p context: the highest-scoring of this device's
/// staged kernels that the kernel matcher admits. Built from the descriptor so every source
/// field is the packer's own.
std::optional<KernelDefinition> selectStagedKernel(const MatchContext& context,
                                                   const BoundTokens& bound)
{
    const auto* staged = findLoadedSet(TRITON_RMSNORM.engineName);
    if(staged == nullptr)
    {
        return std::nullopt;
    }
    const auto device = context.deviceProperties.gcnArchName;

    std::optional<KernelDefinition> best;
    double bestScore = 0.0;
    for(const auto& pack : staged->packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            // An empty kernel arch list means the pack's.
            const auto& arches = kernel.arch.empty() ? pack.arch : kernel.arch;
            if(std::find(arches.begin(), arches.end(), device) == arches.end())
            {
                continue;
            }
            const KernelDefinition candidate{kernel.id,
                                             pack.id,
                                             pack.dispatchId,
                                             kernel.source,
                                             kernel.metadata,
                                             kernel.priority,
                                             arches,
                                             kernel.originDirectory,
                                             kernel.name,
                                             kernel.treeRoot};
            if(!matchesKernel(TRITON_RMSNORM, context, candidate, bound))
            {
                continue;
            }
            const auto score = scoreKernel(TRITON_RMSNORM, context, candidate, bound);
            if(!best.has_value() || score > bestScore)
            {
                best = candidate;
                bestScore = score;
            }
        }
    }
    return best;
}

/// Input magnitude factor for the epsilon-sensitive cases (epsilon = 1e-2).
///
/// The base fill has magnitudes 0.5 + k/8, k = 0..7, so mean(x^2) is about 0.96 and epsilon
/// 1e-2 would move rstd by only ~0.5%, inside the bf16 tolerance: a kernel that ignored
/// epsilon would pass. Scaled by 0.03 the squares shrink by 9e-4, so mean(x^2) is about
/// 8.6e-4 for wide rows and at most 1.70e-3 (N = 1, k = 7). Then
/// rstd(eps=1e-5) / rstd(eps=1e-2) = sqrt((ms + 1e-2) / (ms + 1e-5)) is at least 2.61 (at
/// the maximal ms): a relative output change of at least 1.61, i.e. >= 80x the bf16
/// tolerance (2e-2) and >= 320x the f16 one (5e-3). For wide rows the change is ~2.52
/// (~126x and ~504x). The epsilon = 1e-5 cases keep the unscaled fill.
constexpr double SMALL_INPUT_SCALE = 0.03;
constexpr double EPSILON_SENSITIVE_THRESHOLD = 1e-3;

/// Host and device x, scale and y for one RMS-norm, freed on scope exit.
///
/// The reference is computed from the same bf16/f16 values the device reads, which keeps
/// input quantisation out of the error budget. Signs vary so the product is signed while
/// the squares do not cancel.
template <typename T>
class RmsNormBuffers
{
public:
    RmsNormBuffers(int64_t rows, int64_t columns, double epsilon, double inputScale)
        : _rows(rows)
        , _columns(columns)
        , _epsilon(epsilon)
        , _x(static_cast<size_t>(rows * columns))
        , _scale(static_cast<size_t>(columns))
    {
        for(int64_t row = 0; row < rows; ++row)
        {
            for(int64_t column = 0; column < columns; ++column)
            {
                // Eighths are exact in bf16 at scale 1; a scaled value is quantised, and the
                // reference reads the quantised one.
                const float magnitude = 0.5F + static_cast<float>((row + column) % 8) * 0.125F;
                const float sign = ((row + 2 * column) % 3 == 0) ? -1.0F : 1.0F;
                _x[static_cast<size_t>(row * columns + column)]
                    = T(static_cast<float>(inputScale) * sign * magnitude);
            }
        }
        for(int64_t column = 0; column < columns; ++column)
        {
            _scale[static_cast<size_t>(column)] = T(0.5F + static_cast<float>(column % 4) * 0.25F);
        }

        EXPECT_EQ(hipSuccess, hipMalloc(&_xDevice, _x.size() * sizeof(T)));
        EXPECT_EQ(hipSuccess, hipMalloc(&_scaleDevice, _scale.size() * sizeof(T)));
        EXPECT_EQ(hipSuccess, hipMalloc(&_yDevice, _x.size() * sizeof(T)));
        EXPECT_EQ(hipSuccess,
                  hipMemcpy(_xDevice, _x.data(), _x.size() * sizeof(T), hipMemcpyHostToDevice));
        EXPECT_EQ(
            hipSuccess,
            hipMemcpy(
                _scaleDevice, _scale.data(), _scale.size() * sizeof(T), hipMemcpyHostToDevice));
        EXPECT_EQ(hipSuccess, hipMemset(_yDevice, 0, _x.size() * sizeof(T)));
    }

    ~RmsNormBuffers()
    {
        static_cast<void>(hipFree(_xDevice));
        static_cast<void>(hipFree(_scaleDevice));
        static_cast<void>(hipFree(_yDevice));
    }

    RmsNormBuffers(const RmsNormBuffers&) = delete;
    RmsNormBuffers& operator=(const RmsNormBuffers&) = delete;

    /// x, scale and y, plus the epsilon slot when @p withRuntimeEpsilon. A pass-by-value
    /// scalar is read from host memory, so that slot points at a member of this object.
    std::vector<hipdnnPluginDeviceBuffer_t> descriptors(bool withRuntimeEpsilon)
    {
        std::vector<hipdnnPluginDeviceBuffer_t> buffers{
            hipdnnPluginDeviceBuffer_t{TRITON_X_UID, _xDevice},
            hipdnnPluginDeviceBuffer_t{TRITON_SCALE_UID, _scaleDevice},
            hipdnnPluginDeviceBuffer_t{TRITON_Y_UID, _yDevice}};
        if(withRuntimeEpsilon)
        {
            _epsilonHost = static_cast<float>(_epsilon);
            buffers.push_back(hipdnnPluginDeviceBuffer_t{TRITON_EPSILON_UID, &_epsilonHost});
        }
        return buffers;
    }

    std::vector<T> readOutput() const
    {
        std::vector<T> result(_x.size());
        EXPECT_EQ(hipSuccess,
                  hipMemcpy(result.data(), _yDevice, _x.size() * sizeof(T), hipMemcpyDeviceToHost));
        return result;
    }

    /// y = x * rsqrt(mean(x^2) + eps) * scale, in double from the quantised inputs.
    std::vector<double> reference() const
    {
        std::vector<double> expected(_x.size());
        for(int64_t row = 0; row < _rows; ++row)
        {
            double sumOfSquares = 0.0;
            for(int64_t column = 0; column < _columns; ++column)
            {
                const auto value = static_cast<double>(
                    static_cast<float>(_x[static_cast<size_t>(row * _columns + column)]));
                sumOfSquares += value * value;
            }
            const double rstd
                = 1.0 / std::sqrt(sumOfSquares / static_cast<double>(_columns) + _epsilon);
            for(int64_t column = 0; column < _columns; ++column)
            {
                const auto index = static_cast<size_t>(row * _columns + column);
                expected[index] = static_cast<double>(static_cast<float>(_x[index])) * rstd
                                  * static_cast<double>(
                                      static_cast<float>(_scale[static_cast<size_t>(column)]));
            }
        }
        return expected;
    }

private:
    int64_t _rows;
    int64_t _columns;
    double _epsilon;
    float _epsilonHost = 0.0F;
    std::vector<T> _x;
    std::vector<T> _scale;
    void* _xDevice = nullptr;
    void* _scaleDevice = nullptr;
    void* _yDevice = nullptr;
};

/// Worst element-wise relative error, with the reference magnitude floored at 1e-3 so a
/// near-zero output does not turn rounding into a large ratio.
template <typename T>
std::pair<double, size_t> worstRelativeError(const std::vector<T>& actual,
                                             const std::vector<double>& expected)
{
    double worst = 0.0;
    size_t worstIndex = 0;
    for(size_t i = 0; i < actual.size(); ++i)
    {
        const auto got = static_cast<double>(static_cast<float>(actual[i]));
        // NaN never compares greater, so a non-finite value is made an infinite error.
        const double error
            = std::isfinite(got) && std::isfinite(expected[i])
                  ? std::abs(got - expected[i]) / std::max(std::abs(expected[i]), 1e-3)
                  : std::numeric_limits<double>::infinity();
        if(error > worst)
        {
            worst = error;
            worstIndex = i;
        }
    }
    return {worst, worstIndex};
}

/// Runs the engine-selected staged kernel over a graph with x dims @p xDims and checks every
/// element against the host reference. Rows are the product of every axis but the last;
/// columns are the last axis.
///
/// @param runtimeEpsilon true to supply epsilon at execute as a pass-by-value operand,
///        false to bake it into the graph.
/// @param scaleDims scale's dims; the rank-1 `{columns}` when unset.
template <typename T>
void expectRmsNormMatchesReference(DataType dataType,
                                   const std::vector<int64_t>& xDims,
                                   double epsilon,
                                   bool runtimeEpsilon,
                                   double relativeTolerance,
                                   const std::optional<std::vector<int64_t>>& scaleDims
                                   = std::nullopt)
{
    const int64_t columns = xDims.back();
    int64_t rows = 1;
    for(size_t axis = 0; axis + 1 < xDims.size(); ++axis)
    {
        rows *= xDims[axis];
    }

    TritonRmsNormGraphOptions options;
    options.dataType = dataType;
    options.xDims = xDims;
    options.scaleDims = scaleDims;
    options.epsilon = static_cast<float>(epsilon);
    options.runtimeEpsilon = runtimeEpsilon;
    const GraphFixture fixture(buildTritonRmsNormGraph(options), packedArchDeviceProperties());

    const auto bound = bindingsFor(fixture.context());
    const auto kernel = selectStagedKernel(fixture.context(), bound);
    ASSERT_TRUE(kernel.has_value())
        << "no staged kernel covers N=" << columns << " for this device, though the shard has "
        << "objects for it";

    const auto& handler = dispatchHandler(TRITON_RMSNORM);
    const auto prepared = handler.prepare(fixture.context(), bound, *kernel);
    ASSERT_NE(prepared, nullptr);

    const double inputScale = epsilon > EPSILON_SENSITIVE_THRESHOLD ? SMALL_INPUT_SCALE : 1.0;
    RmsNormBuffers<T> buffers(rows, columns, epsilon, inputScale);
    const auto descriptors = buffers.descriptors(runtimeEpsilon);
    const Handle handle;

    handler.launch(
        handle, *prepared, descriptors.data(), static_cast<uint32_t>(descriptors.size()), nullptr);
    ASSERT_EQ(hipSuccess, hipDeviceSynchronize());

    const auto actual = buffers.readOutput();
    const auto expected = buffers.reference();
    ASSERT_EQ(actual.size(), expected.size());

    const auto [worst, worstIndex] = worstRelativeError(actual, expected);
    EXPECT_LE(worst, relativeTolerance)
        << "kernel " << kernel->name << ", rows=" << rows << ", N=" << columns
        << ", epsilon=" << epsilon << ": at element " << worstIndex << " got "
        << static_cast<float>(actual[worstIndex]) << ", expected " << expected[worstIndex];
}

// ---------------------------------------------------------------------------
// dispatch on device: a real launch out of the staged archive
// ---------------------------------------------------------------------------

struct NumericsCase
{
    int64_t columns;
    int64_t rows;
    double epsilon;
};

/// Odd widths, each block_n boundary and boundary+1, and the largest width; 1, 7 and 130
/// rows; epsilon 1e-5 and a large 1e-2 that dominates the mean square of the scaled inputs.
std::vector<NumericsCase> numericsCases()
{
    std::vector<NumericsCase> cases;
    for(const int64_t columns : {1, 7, 1000, 1001, 1024, 1025, 2048, 2049, 3000, 4096, 4097, 8192})
    {
        for(const int64_t rows : {1, 7, 130})
        {
            for(const double epsilon : {1e-5, 1e-2})
            {
                cases.push_back({columns, rows, epsilon});
            }
        }
    }
    return cases;
}

std::string numericsCaseName(const ::testing::TestParamInfo<NumericsCase>& info)
{
    return "N" + std::to_string(info.param.columns) + "_rows" + std::to_string(info.param.rows)
           + (info.param.epsilon > 1e-3 ? "_epsLarge" : "_epsSmall");
}

class TestGpuTritonRmsNormBf16 : public ::testing::TestWithParam<NumericsCase>
{
};

TEST_P(TestGpuTritonRmsNormBf16, MatchesTheHostReferenceThroughARuntimeEpsilon)
{
    SKIP_IF_NO_DEVICES();
    if(const auto why = whyNoStagedKernel(); !why.empty())
    {
        GTEST_SKIP() << why;
    }

    // bf16 keeps 8 mantissa bits (~4e-3 relative); 2e-2 leaves room for the kernel's fp32
    // reduction order differing from the reference's.
    const auto& c = GetParam();
    expectRmsNormMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        DataType::BFLOAT16, {c.rows, c.columns}, c.epsilon, true, 2e-2);
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuTritonRmsNormBf16,
                         ::testing::ValuesIn(numericsCases()),
                         numericsCaseName);

class TestGpuTritonRmsNormF16 : public ::testing::TestWithParam<NumericsCase>
{
};

TEST_P(TestGpuTritonRmsNormF16, MatchesTheHostReferenceThroughARuntimeEpsilon)
{
    SKIP_IF_NO_DEVICES();
    if(const auto why = whyNoStagedKernel(); !why.empty())
    {
        GTEST_SKIP() << why;
    }

    const auto& c = GetParam();
    expectRmsNormMatchesReference<hipdnn_data_sdk::types::half>(
        DataType::HALF, {c.rows, c.columns}, c.epsilon, true, 5e-3);
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuTritonRmsNormF16,
                         ::testing::ValuesIn(numericsCases()),
                         numericsCaseName);

TEST(TestGpuTritonRmsNormDispatch, HonoursAnEpsilonBakedIntoTheGraph)
{
    SKIP_IF_NO_DEVICES();
    if(const auto why = whyNoStagedKernel(); !why.empty())
    {
        GTEST_SKIP() << why;
    }

    // A graph constant reaches the kernarg like a runtime value. The fill is scaled down
    // for epsilon 1e-2 (see SMALL_INPUT_SCALE), so a kernarg ignoring it would miss the
    // reference by >= 80x the bf16 tolerance.
    expectRmsNormMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        DataType::BFLOAT16, {7, 1024}, 1e-2, false, 2e-2);
}

TEST(TestGpuTritonRmsNormDispatch, CollapsesRank3LeadingAxesIntoTheGrid)
{
    SKIP_IF_NO_DEVICES();
    if(const auto why = whyNoStagedKernel(); !why.empty())
    {
        GTEST_SKIP() << why;
    }

    // 15 rows; a grid sized from the leading axis alone would leave 12 unnormalised.
    expectRmsNormMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        DataType::BFLOAT16, {3, 5, 1001}, 1e-5, true, 2e-2);
}

TEST(TestGpuTritonRmsNormDispatch, CollapsesRank4LeadingAxesIntoTheGrid)
{
    SKIP_IF_NO_DEVICES();
    if(const auto why = whyNoStagedKernel(); !why.empty())
    {
        GTEST_SKIP() << why;
    }

    // 64 rows under a rank-4 scale; a grid sized from the leading axis alone would leave
    // 62 unnormalised.
    expectRmsNormMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        DataType::BFLOAT16, {2, 8, 4, 4096}, 1e-5, true, 2e-2, std::vector<int64_t>{1, 1, 1, 4096});
}

TEST(TestGpuTritonRmsNormDispatch, LoadsTheModuleOnceAcrossTwoPrepares)
{
    SKIP_IF_NO_DEVICES();
    if(const auto why = whyNoStagedKernel(); !why.empty())
    {
        GTEST_SKIP() << why;
    }

    const GraphFixture fixture(buildTritonRmsNormGraph(), packedArchDeviceProperties());
    const auto bound = bindingsFor(fixture.context());
    const auto kernel = selectStagedKernel(fixture.context(), bound);
    ASSERT_TRUE(kernel.has_value());

    const auto& handler = dispatchHandler(TRITON_RMSNORM);

    // Warm first: the module may already be resident from an earlier case, so only a
    // second plan over the same kernel is expected to add nothing.
    const auto warm = handler.prepare(fixture.context(), bound, *kernel);
    ASSERT_NE(warm, nullptr);

    const size_t resident = tritonRmsNormKpackModuleCache().size();
    ASSERT_GT(resident, 0U);

    const auto second = handler.prepare(fixture.context(), bound, *kernel);
    ASSERT_NE(second, nullptr);

    EXPECT_EQ(tritonRmsNormKpackModuleCache().size(), resident);
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_TRITON
