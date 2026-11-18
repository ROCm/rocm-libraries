// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see TestFlydslRmsNormEngine.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hip/hip_runtime_api.h>

#include <hipdnn_data_sdk/types/Bfloat16.hpp>
#include <hipdnn_data_sdk/types/Half.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslNormDeviceTestSupport.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslRmsNormBwdTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslRmsNormTestGraphs.hpp"

/**
 * @file TestFlydslRmsNormObjects.cpp
 * @brief Every FlyDSL RMS-norm object this device's shard carries, launched by name.
 *
 * The dispatch suites launch the objects their cases name, and selection picks one object
 * per graph, so most of the shipped objects are never run by either. An object whose bytes
 * are wrong -- a regeneration that went astray, a file copied over its neighbour -- still
 * passes `gen_descriptors.py --check` and the shard check once its manifest agrees, and
 * only a launch tells it apart. So each object here is launched directly, through its own
 * pack's dispatch handler, against a double-precision host reference:
 *
 *  - at its own width when baked; when generic, at its bound, just below it and at a
 *    mid-range width that is not a multiple of 8 (the forward generic tier has no bound,
 *    so the largest baked width stands in for one, plus a width below one vector);
 *  - over 3 and 37 rows, so a partial last workgroup and more rows than programs occur;
 *  - with every row at its own scale, so every row's inv_rms differs;
 *  - into outputs that start as NaN, with a non-finite element an infinite error;
 *  - the backward twice, the two launches compared bit for bit.
 *
 * Two tiers. `OnePerForm` runs one object per (dtype, gamma form, inv_rms, tier) and is
 * the pre-merge case. `Full/...Every` runs all of them and is the nightly one: about a
 * second per architecture today, but the object count grows with every op.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

using data_objects::DataType;
using data_objects::NormFwdPhase;
using Bf16 = hipdnn_data_sdk::types::bfloat16;
using F16 = hipdnn_data_sdk::types::half;

/// Away from the 1e-5 every other case uses, so an object that ignored its argument fails.
constexpr float OBJECT_EPSILON = 0.01F;

/// The row counts every object runs at.
constexpr std::array<int64_t, 2> OBJECT_ROWS{3, 37};

/// Which of the staged objects a case launches.
enum class ObjectSelection
{
    ONE_PER_FORM,
    EVERY
};

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

/// The graph dtype for a kernel's metadata spelling of one.
DataType graphDataType(const std::string& dtype)
{
    if(dtype == "bf16")
    {
        return DataType::BFLOAT16;
    }
    return dtype == "f16" ? DataType::HALF : DataType::FLOAT;
}

/// The tolerances the dispatch suites use for each dtype.
double toleranceFor(const std::string& dtype)
{
    if(dtype == "bf16")
    {
        return 2e-2;
    }
    return dtype == "f16" ? 5e-3 : 1e-4;
}

/// Row @p row's scale: 0.25 to 4.4, so mean squares differ by over two orders of magnitude.
double rowScale(int64_t row)
{
    return 0.25 * std::pow(2.0, static_cast<double>(row % 5))
           * (1.0 + 0.1 * static_cast<double>(row % 3));
}

template <typename V>
double asDouble(V value)
{
    return static_cast<double>(static_cast<float>(value));
}

/// @p kernels, or one per distinct value of @p form when @p selection asks for that: the
/// first of each in name order, so the choice is the same on every run.
template <typename Form>
std::vector<KernelDefinition>
    select(std::vector<KernelDefinition> kernels, ObjectSelection selection, Form form)
{
    std::sort(kernels.begin(), kernels.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.name < rhs.name;
    });
    if(selection == ObjectSelection::EVERY)
    {
        return kernels;
    }
    std::vector<KernelDefinition> chosen;
    std::vector<decltype(form(kernels.front()))> seen;
    for(auto& kernel : kernels)
    {
        const auto key = form(kernel);
        if(std::find(seen.begin(), seen.end(), key) == seen.end())
        {
            seen.push_back(key);
            chosen.push_back(std::move(kernel));
        }
    }
    return chosen;
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

    template <typename V>
    explicit DeviceBuffer(const std::vector<V>& host)
        : DeviceBuffer(host.size() * sizeof(V))
    {
        EXPECT_EQ(hipSuccess,
                  hipMemcpy(_pointer, host.data(), host.size() * sizeof(V), hipMemcpyHostToDevice));
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

    /// Every byte 0xFF: NaN in every float type, so an element nothing writes fails.
    void poison() const
    {
        EXPECT_EQ(hipSuccess, hipMemset(_pointer, 0xFF, _bytes));
    }

    template <typename V>
    std::vector<V> read(size_t count) const
    {
        std::vector<V> host(count);
        EXPECT_EQ(hipSuccess,
                  hipMemcpy(host.data(), _pointer, count * sizeof(V), hipMemcpyDeviceToHost));
        return host;
    }

private:
    size_t _bytes;
    void* _pointer = nullptr;
};

// ---------------------------------------------------------------------------
// forward
// ---------------------------------------------------------------------------

template <typename T, typename G>
void expectForwardObjectMatches(const KernelDefinition& kernel, int64_t rows, int64_t columns)
{
    SCOPED_TRACE(kernel.name + " over " + std::to_string(rows) + "x" + std::to_string(columns));
    const auto dtype = stringField(kernel, FLYDSL_DTYPE_FIELD);
    const bool training = intField(kernel, FLYDSL_STORE_RSTD_FIELD) == 1;
    const bool f32Scale = intField(kernel, FLYDSL_SCALE_F32_FIELD) == 1;

    std::mt19937 generator(static_cast<uint32_t>(rows * 131 + columns));
    std::uniform_real_distribution<float> uniform(-1.0F, 1.0F);
    std::vector<T> x(static_cast<size_t>(rows * columns));
    std::vector<G> gamma(static_cast<size_t>(columns));
    for(int64_t row = 0; row < rows; ++row)
    {
        for(int64_t column = 0; column < columns; ++column)
        {
            x[static_cast<size_t>(row * columns + column)]
                = T(static_cast<float>(rowScale(row) * static_cast<double>(uniform(generator))));
        }
    }
    for(auto& value : gamma)
    {
        value = G(1.5F * uniform(generator));
    }

    // y = x * rsqrt(mean(x^2) + eps) * gamma and inv_rms = rsqrt(mean(x^2) + eps), in double
    // from the quantised inputs.
    std::vector<double> expectedY(x.size());
    std::vector<double> expectedInvRms(static_cast<size_t>(rows));
    for(int64_t row = 0; row < rows; ++row)
    {
        double sumOfSquares = 0.0;
        for(int64_t column = 0; column < columns; ++column)
        {
            const auto value = asDouble(x[static_cast<size_t>(row * columns + column)]);
            sumOfSquares += value * value;
        }
        const double invRms = 1.0
                              / std::sqrt(sumOfSquares / static_cast<double>(columns)
                                          + static_cast<double>(OBJECT_EPSILON));
        expectedInvRms[static_cast<size_t>(row)] = invRms;
        for(int64_t column = 0; column < columns; ++column)
        {
            const auto index = static_cast<size_t>(row * columns + column);
            expectedY[index]
                = asDouble(x[index]) * invRms * asDouble(gamma[static_cast<size_t>(column)]);
        }
    }

    const GraphFixture fixture(
        buildFlydslRmsNormGraph(graphDataType(dtype),
                                {rows, columns},
                                OBJECT_EPSILON,
                                false,
                                std::nullopt,
                                f32Scale ? std::optional<DataType>(DataType::FLOAT) : std::nullopt,
                                std::nullopt,
                                false,
                                training,
                                training ? NormFwdPhase::TRAINING : NormFwdPhase::NOT_SET),
        packedArchDeviceProperties());
    const auto bound = matchesGraph(FLYDSL_RMSNORM, fixture.context());
    ASSERT_TRUE(bound.has_value()) << "the graph this object is meant for is declined";
    // The object's own metadata has to admit the graph built from it; one that does not
    // would never be selected, which a launch by name would otherwise hide.
    ASSERT_TRUE(matchesKernel(FLYDSL_RMSNORM, fixture.context(), kernel, *bound));

    const auto& handler = dispatchHandler(FLYDSL_RMSNORM);
    const auto prepared = handler.prepare(fixture.context(), *bound, kernel);
    ASSERT_NE(prepared, nullptr);

    const DeviceBuffer xDevice(x);
    const DeviceBuffer gammaDevice(gamma);
    const DeviceBuffer yDevice(x.size() * sizeof(T));
    const DeviceBuffer invRmsDevice(static_cast<size_t>(rows) * sizeof(float));
    yDevice.poison();
    invRmsDevice.poison();
    std::vector<hipdnnPluginDeviceBuffer_t> arguments{{FLYDSL_X_UID, xDevice.get()},
                                                      {FLYDSL_SCALE_UID, gammaDevice.get()},
                                                      {FLYDSL_Y_UID, yDevice.get()}};
    if(training)
    {
        arguments.push_back({FLYDSL_INV_RMS_UID, invRmsDevice.get()});
    }
    const Handle handle;
    handler.launch(
        handle, *prepared, arguments.data(), static_cast<uint32_t>(arguments.size()), nullptr);
    ASSERT_EQ(hipSuccess, hipDeviceSynchronize());

    const auto y = yDevice.read<T>(x.size());
    const auto worstY = worstRelativeError(y, expectedY, 1e-3);
    EXPECT_LE(worstY.error, toleranceFor(dtype))
        << "y at element " << worstY.index << ": got " << static_cast<float>(y[worstY.index])
        << ", expected " << expectedY[worstY.index];
    if(training)
    {
        const auto invRms = invRmsDevice.read<float>(static_cast<size_t>(rows));
        const auto worstInvRms = worstRelativeError(invRms, expectedInvRms, 0.0);
        EXPECT_LE(worstInvRms.error, 1e-4) << "inv_rms of row " << worstInvRms.index;
    }
}

template <typename T>
void expectForwardObjectMatchesAnyGamma(const KernelDefinition& kernel,
                                        int64_t rows,
                                        int64_t columns)
{
    if(intField(kernel, FLYDSL_SCALE_F32_FIELD) == 1)
    {
        expectForwardObjectMatches<T, float>(kernel, rows, columns);
    }
    else
    {
        expectForwardObjectMatches<T, T>(kernel, rows, columns);
    }
}

// ---------------------------------------------------------------------------
// backward
// ---------------------------------------------------------------------------

template <typename T, typename G>
void expectBackwardObjectMatches(const KernelDefinition& kernel, int64_t rows, int64_t columns)
{
    SCOPED_TRACE(kernel.name + " over " + std::to_string(rows) + "x" + std::to_string(columns));
    const auto dtype = stringField(kernel, FLYDSL_DTYPE_FIELD);
    const bool f32Scale = intField(kernel, FLYDSL_SCALE_F32_FIELD) == 1;

    std::mt19937 generator(static_cast<uint32_t>(rows * 977 + columns));
    std::uniform_real_distribution<float> uniform(-1.0F, 1.0F);
    const auto count = static_cast<size_t>(rows * columns);
    std::vector<T> x(count);
    std::vector<T> dy(count);
    std::vector<G> gamma(static_cast<size_t>(columns));
    std::vector<float> invRms(static_cast<size_t>(rows));
    for(int64_t row = 0; row < rows; ++row)
    {
        double sumOfSquares = 0.0;
        for(int64_t column = 0; column < columns; ++column)
        {
            const auto index = static_cast<size_t>(row * columns + column);
            x[index]
                = T(static_cast<float>(rowScale(row) * static_cast<double>(uniform(generator))));
            dy[index] = T(uniform(generator));
            sumOfSquares += asDouble(x[index]) * asDouble(x[index]);
        }
        invRms[static_cast<size_t>(row)]
            = static_cast<float>(1.0
                                 / std::sqrt(sumOfSquares / static_cast<double>(columns)
                                             + static_cast<double>(OBJECT_EPSILON)));
    }
    for(auto& value : gamma)
    {
        value = G(1.5F * uniform(generator));
    }

    // dx = (dy*g - xhat*c1) * rstd with xhat = x*rstd and c1 = mean(xhat*dy*g); dgamma is
    // the sum over rows of dy*xhat. In double, from the quantised inputs and f32 inv_rms.
    std::vector<double> expectedDx(count);
    std::vector<double> expectedDgamma(static_cast<size_t>(columns), 0.0);
    for(int64_t row = 0; row < rows; ++row)
    {
        const auto rstd = static_cast<double>(invRms[static_cast<size_t>(row)]);
        double c1 = 0.0;
        for(int64_t column = 0; column < columns; ++column)
        {
            const auto index = static_cast<size_t>(row * columns + column);
            c1 += asDouble(x[index]) * rstd * asDouble(dy[index])
                  * asDouble(gamma[static_cast<size_t>(column)]);
        }
        c1 /= static_cast<double>(columns);
        for(int64_t column = 0; column < columns; ++column)
        {
            const auto index = static_cast<size_t>(row * columns + column);
            const double xhat = asDouble(x[index]) * rstd;
            expectedDx[index]
                = (asDouble(dy[index]) * asDouble(gamma[static_cast<size_t>(column)]) - xhat * c1)
                  * rstd;
            expectedDgamma[static_cast<size_t>(column)] += asDouble(dy[index]) * xhat;
        }
    }

    RmsNormBwdGraphSpec spec;
    spec.dataType = graphDataType(dtype);
    spec.xDims = {rows, columns};
    if(f32Scale)
    {
        spec.scaleDataType = DataType::FLOAT;
    }
    const GraphFixture fixture(buildFlydslRmsNormBwdGraph(spec), packedArchDeviceProperties());
    const auto bound = matchesGraph(FLYDSL_RMSNORM_BWD, fixture.context());
    ASSERT_TRUE(bound.has_value()) << "the graph this object is meant for is declined";
    ASSERT_TRUE(matchesKernel(FLYDSL_RMSNORM_BWD, fixture.context(), kernel, *bound));

    const auto& handler = dispatchHandler(FLYDSL_RMSNORM_BWD);
    const auto workspaceBytes = handler.workspaceBytes(fixture.context(), *bound, kernel);
    const auto prepared = handler.prepare(fixture.context(), *bound, kernel);
    ASSERT_NE(prepared, nullptr);

    const DeviceBuffer xDevice(x);
    const DeviceBuffer dyDevice(dy);
    const DeviceBuffer gammaDevice(gamma);
    const DeviceBuffer invRmsDevice(invRms);
    const DeviceBuffer dxDevice(count * sizeof(T));
    const DeviceBuffer dgammaDevice(gamma.size() * sizeof(G));
    const DeviceBuffer workspace(workspaceBytes);
    const std::vector<hipdnnPluginDeviceBuffer_t> arguments{
        {FLYDSL_BWD_DY_UID, dyDevice.get()},
        {FLYDSL_BWD_X_UID, xDevice.get()},
        {FLYDSL_BWD_SCALE_UID, gammaDevice.get()},
        {FLYDSL_BWD_INV_RMS_UID, invRmsDevice.get()},
        {FLYDSL_BWD_DX_UID, dxDevice.get()},
        {FLYDSL_BWD_DSCALE_UID, dgammaDevice.get()}};
    const Handle handle;
    const auto run = [&] {
        dxDevice.poison();
        dgammaDevice.poison();
        workspace.poison();
        handler.launch(handle,
                       *prepared,
                       arguments.data(),
                       static_cast<uint32_t>(arguments.size()),
                       workspace.get());
        EXPECT_EQ(hipSuccess, hipDeviceSynchronize());
        return std::make_pair(dxDevice.read<T>(count), dgammaDevice.read<G>(gamma.size()));
    };

    const auto [dx, dgamma] = run();
    const auto worstDx = worstRelativeError(dx, expectedDx, 1e-1);
    EXPECT_LE(worstDx.error, toleranceFor(dtype)) << "dx at element " << worstDx.index;
    // dgamma sums `rows` products, so its scale grows with the row count.
    const auto worstDgamma = worstRelativeError(dgamma, expectedDgamma, 1.0);
    EXPECT_LE(worstDgamma.error, toleranceFor(dtype)) << "dgamma at element " << worstDgamma.index;

    // Deterministic: the same programs, rows and reduction order on every launch.
    const auto [dxAgain, dgammaAgain] = run();
    EXPECT_EQ(0, std::memcmp(dx.data(), dxAgain.data(), dx.size() * sizeof(T)));
    EXPECT_EQ(0, std::memcmp(dgamma.data(), dgammaAgain.data(), dgamma.size() * sizeof(G)));
}

template <typename T>
void expectBackwardObjectMatchesAnyGamma(const KernelDefinition& kernel,
                                         int64_t rows,
                                         int64_t columns)
{
    if(intField(kernel, FLYDSL_SCALE_F32_FIELD) == 1)
    {
        expectBackwardObjectMatches<T, float>(kernel, rows, columns);
    }
    else
    {
        expectBackwardObjectMatches<T, T>(kernel, rows, columns);
    }
}

// ---------------------------------------------------------------------------
// the cases
// ---------------------------------------------------------------------------

class TestGpuFlydslRmsNormObjects : public ::testing::TestWithParam<ObjectSelection>
{
};

TEST_P(TestGpuFlydslRmsNormObjects, ForwardObjectsMatchTheReference)
{
    SKIP_IF_NO_DEVICES();
    const auto staged = stagedKernels(FLYDSL_RMSNORM.engineName);
    if(staged.empty())
    {
        reportUnstaged("rmsnorm", "this build staged no RMSNorm objects for this device");
        return;
    }

    // A form is what changes the code a kernel runs: element type, gamma's, whether it
    // writes inv_rms, and baked or generic.
    const auto kernels = select(staged, GetParam(), [](const KernelDefinition& kernel) {
        return std::make_tuple(stringField(kernel, FLYDSL_DTYPE_FIELD),
                               intField(kernel, FLYDSL_SCALE_F32_FIELD),
                               intField(kernel, FLYDSL_STORE_RSTD_FIELD),
                               intField(kernel, FLYDSL_N_FIELD) == FLYDSL_GENERIC_N);
    });
    for(const auto& kernel : kernels)
    {
        const auto dtype = stringField(kernel, FLYDSL_DTYPE_FIELD);
        const auto bakedN = intField(kernel, FLYDSL_N_FIELD);
        const auto widths = bakedN == FLYDSL_GENERIC_N ? std::vector<int64_t>{16384, 16383, 1001, 7}
                                                       : std::vector<int64_t>{bakedN};
        for(const auto rows : OBJECT_ROWS)
        {
            for(const auto columns : widths)
            {
                if(dtype == "bf16")
                {
                    expectForwardObjectMatchesAnyGamma<Bf16>(kernel, rows, columns);
                }
                else if(dtype == "f16")
                {
                    expectForwardObjectMatchesAnyGamma<F16>(kernel, rows, columns);
                }
                else if(dtype == "f32")
                {
                    expectForwardObjectMatches<float, float>(kernel, rows, columns);
                }
                else
                {
                    ADD_FAILURE() << kernel.name << " declares dtype '" << dtype << "'";
                }
            }
        }
    }
    RecordProperty("objects", static_cast<int>(kernels.size()));
}

TEST_P(TestGpuFlydslRmsNormObjects, BackwardObjectsMatchTheReference)
{
    SKIP_IF_NO_DEVICES();
    const auto staged = stagedKernels(FLYDSL_RMSNORM_BWD.engineName);
    if(staged.empty())
    {
        reportUnstaged("rmsnorm_bwd",
                       "this build staged no RMSNorm backward objects for this device");
        return;
    }

    // The generic tiers are told apart by their bound, which sizes their registers.
    const auto kernels = select(staged, GetParam(), [](const KernelDefinition& kernel) {
        const bool generic = intField(kernel, FLYDSL_N_FIELD) == FLYDSL_GENERIC_N;
        return std::make_tuple(stringField(kernel, FLYDSL_DTYPE_FIELD),
                               intField(kernel, FLYDSL_SCALE_F32_FIELD),
                               generic ? intField(kernel, FLYDSL_N_MAX_FIELD) : int64_t{0});
    });
    for(const auto& kernel : kernels)
    {
        const auto dtype = stringField(kernel, FLYDSL_DTYPE_FIELD);
        const auto bakedN = intField(kernel, FLYDSL_N_FIELD);
        const auto nMax = intField(kernel, FLYDSL_N_MAX_FIELD);
        const auto widths = bakedN == FLYDSL_GENERIC_N
                                ? std::vector<int64_t>{nMax, nMax - 1, nMax / 2 + 3}
                                : std::vector<int64_t>{bakedN};
        for(const auto rows : OBJECT_ROWS)
        {
            for(const auto columns : widths)
            {
                if(dtype == "bf16")
                {
                    expectBackwardObjectMatchesAnyGamma<Bf16>(kernel, rows, columns);
                }
                else if(dtype == "f16")
                {
                    expectBackwardObjectMatchesAnyGamma<F16>(kernel, rows, columns);
                }
                else if(dtype == "f32")
                {
                    expectBackwardObjectMatches<float, float>(kernel, rows, columns);
                }
                else
                {
                    ADD_FAILURE() << kernel.name << " declares dtype '" << dtype << "'";
                }
            }
        }
    }
    RecordProperty("objects", static_cast<int>(kernels.size()));
}

std::string selectionName(const ::testing::TestParamInfo<ObjectSelection>& info)
{
    return info.param == ObjectSelection::EVERY ? "Every" : "OnePerForm";
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuFlydslRmsNormObjects,
                         ::testing::Values(ObjectSelection::ONE_PER_FORM),
                         selectionName);

INSTANTIATE_TEST_SUITE_P(Full,
                         TestGpuFlydslRmsNormObjects,
                         ::testing::Values(ObjectSelection::EVERY),
                         selectionName);

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
