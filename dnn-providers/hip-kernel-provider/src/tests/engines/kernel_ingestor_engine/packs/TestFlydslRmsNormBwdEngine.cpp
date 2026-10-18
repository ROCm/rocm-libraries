// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see TestFlydslRmsNormEngine.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
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
#include "engines/kernel_ingestor_engine/packs/FlydslRmsNormBwdTestGraphs.hpp"

/**
 * @file TestFlydslRmsNormBwdEngine.cpp
 * @brief The flyDSL RMS-norm backward pack across its native seams, plus real launches.
 *
 * The census over what the pack ships is TestFlydslRmsNormBwdPacks.cpp. The pack's
 * particulars, each with cases here:
 *
 *  - Widths are BAKED or BOUNDED: a baked object serves its own N, a generic one every N
 *    up to its `n_max`, and wider graphs are declined.
 *  - dgamma is DETERMINISTIC: a partial kernel leaves one f32 row per program in the
 *    workspace and a reduce kernel sums them, so two launches agree bit for bit.
 *  - No epsilon (the forward folded it into inv_rms) and no dbias.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

using data_objects::DataType;

bool matches(const RmsNormBwdGraphSpec& spec)
{
    const GraphFixture fixture(buildFlydslRmsNormBwdGraph(spec));
    return matchesGraph(FLYDSL_RMSNORM_BWD, fixture.context()).has_value();
}

BoundTokens bindingsFor(const MatchContext& context)
{
    auto bound = matchesGraph(FLYDSL_RMSNORM_BWD, context);
    if(!bound.has_value())
    {
        throw std::logic_error("test graph does not match the pack it is dispatched against");
    }
    return std::move(*bound);
}

/// The current device with its arch's feature suffix stripped; see the forward suite.
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

KernelDefinition makeBwdKernel(const std::string& dtype,
                               int64_t bakedN,
                               int64_t nMax,
                               int64_t scaleF32 = 0,
                               int64_t priority = 100)
{
    KernelDefinition kernel;
    kernel.name = "rmsnorm_bwd_" + dtype + "_n" + std::to_string(bakedN);
    kernel.priority = priority;
    kernel.metadata
        = {{std::string(FLYDSL_DTYPE_FIELD), dtype},
           {std::string(FLYDSL_N_FIELD), bakedN},
           {std::string(FLYDSL_N_MAX_FIELD), nMax},
           {std::string(FLYDSL_SCALE_F32_FIELD), scaleF32},
           {std::string(FLYDSL_REDUCE_SYMBOL_FIELD), std::string(FLYDSL_BWD_REDUCE_SYMBOL)}};
    return kernel;
}

/// A staged backward kernel by (dtype, N, n_max, scale_f32), promoted from the loaded
/// descriptors as the forward suite does. False when this build staged none for the
/// current device.
bool findStagedBwdKernel(
    const std::string& dtype, int64_t bakedN, int64_t nMax, int64_t scaleF32, KernelDefinition& out)
{
    const auto* staged = findLoadedSet(FLYDSL_RMSNORM_BWD.engineName);
    if(staged == nullptr)
    {
        return false;
    }
    const auto device = packedArchDeviceProperties().gcnArchName;
    for(const auto& pack : staged->packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const auto& arches = kernel.arch.empty() ? pack.arch : kernel.arch;
            if(!device.empty() && !arches.empty()
               && std::find(arches.begin(), arches.end(), device) == arches.end())
            {
                continue;
            }
            const auto* name
                = tryGetMetadataField<std::string>(kernel.metadata, FLYDSL_DTYPE_FIELD);
            const auto* n = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_N_FIELD);
            const auto* bound = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_N_MAX_FIELD);
            const auto* f32Scale
                = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_SCALE_F32_FIELD);
            if(name == nullptr || n == nullptr || bound == nullptr || f32Scale == nullptr
               || *name != dtype || *n != bakedN || *bound != nMax || *f32Scale != scaleF32)
            {
                continue;
            }
            out = KernelDefinition{kernel.id,
                                   pack.id,
                                   pack.dispatchId,
                                   kernel.source,
                                   kernel.metadata,
                                   kernel.priority,
                                   kernel.arch.empty() ? pack.arch : kernel.arch,
                                   kernel.originDirectory,
                                   kernel.name,
                                   kernel.treeRoot};
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// graph_match
// ---------------------------------------------------------------------------

struct BwdCase
{
    std::string name;
    RmsNormBwdGraphSpec (*spec)();
};

std::string bwdCaseName(const ::testing::TestParamInfo<BwdCase>& info)
{
    return info.param.name;
}

class TestFlydslRmsNormBwdGraphMatcherAcceptance : public ::testing::TestWithParam<BwdCase>
{
};

TEST_P(TestFlydslRmsNormBwdGraphMatcherAcceptance, Accepts)
{
    EXPECT_TRUE(matches(GetParam().spec()));
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestFlydslRmsNormBwdGraphMatcherAcceptance,
    ::testing::Values(
        BwdCase{"Bfloat16", [] { return RmsNormBwdGraphSpec{}; }},
        BwdCase{"Half",
                [] {
                    RmsNormBwdGraphSpec spec;
                    spec.dataType = DataType::HALF;
                    return spec;
                }},
        BwdCase{"Float",
                [] {
                    RmsNormBwdGraphSpec spec;
                    spec.dataType = DataType::FLOAT;
                    return spec;
                }},
        // gamma and dgamma in f32 beside bf16 x, dy and dx.
        BwdCase{"Bfloat16WithAnF32Scale",
                [] {
                    RmsNormBwdGraphSpec spec;
                    spec.scaleDataType = DataType::FLOAT;
                    return spec;
                }},
        BwdCase{"Rank3",
                [] {
                    RmsNormBwdGraphSpec spec;
                    spec.xDims = {2, 3, 4096};
                    return spec;
                }},
        // cuDNN's layout: x {B*S, D, 1, 1}, gamma {1, D, 1, 1}, inv_rms {B*S, 1, 1, 1}.
        BwdCase{"CudnnLayout",
                [] {
                    RmsNormBwdGraphSpec spec;
                    spec.xDims = {32, 4096, 1, 1};
                    spec.scaleDims = std::vector<int64_t>{1, 4096, 1, 1};
                    return spec;
                }},
        // Any width; whether an object serves it is the kernel matcher's question.
        BwdCase{"AnOddWidth",
                [] {
                    RmsNormBwdGraphSpec spec;
                    spec.xDims = {8, 4097};
                    return spec;
                }},
        BwdCase{"ComputeTypeUnset",
                [] {
                    RmsNormBwdGraphSpec spec;
                    spec.computeDataType = DataType::UNSET;
                    return spec;
                }}),
    bwdCaseName);

class TestFlydslRmsNormBwdGraphMatcherRefusal : public ::testing::TestWithParam<BwdCase>
{
};

TEST_P(TestFlydslRmsNormBwdGraphMatcherRefusal, Refuses)
{
    EXPECT_FALSE(matches(GetParam().spec()));
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestFlydslRmsNormBwdGraphMatcherRefusal,
    ::testing::Values(
        // No object sums dy into a dbias.
        BwdCase{"WithADbias",
                [] {
                    RmsNormBwdGraphSpec spec;
                    spec.withDbias = true;
                    return spec;
                }},
        BwdCase{"OverrideShapeEnabled",
                [] {
                    RmsNormBwdGraphSpec spec;
                    spec.overrideShapeEnabled = true;
                    return spec;
                }},
        BwdCase{"ComputeTypeDouble",
                [] {
                    RmsNormBwdGraphSpec spec;
                    spec.computeDataType = DataType::DOUBLE;
                    return spec;
                }},
        BwdCase{"AnUnsupportedDtype",
                [] {
                    RmsNormBwdGraphSpec spec;
                    spec.dataType = DataType::DOUBLE;
                    return spec;
                }},
        // x, dy and dx share one element type.
        BwdCase{"DyOfAnotherDtype",
                [] {
                    RmsNormBwdGraphSpec spec;
                    spec.dyDataType = DataType::HALF;
                    return spec;
                }},
        BwdCase{"DxOfAnotherDtype",
                [] {
                    RmsNormBwdGraphSpec spec;
                    spec.dxDataType = DataType::FLOAT;
                    return spec;
                }},
        // gamma is x's dtype or f32 beside 16-bit x, and dgamma is gamma's.
        BwdCase{"GammaAtAnother16BitDtype",
                [] {
                    RmsNormBwdGraphSpec spec;
                    spec.scaleDataType = DataType::HALF;
                    return spec;
                }},
        BwdCase{"Bfloat16GammaBesideFloat",
                [] {
                    RmsNormBwdGraphSpec spec;
                    spec.dataType = DataType::FLOAT;
                    spec.scaleDataType = DataType::BFLOAT16;
                    return spec;
                }},
        BwdCase{"DgammaUnlikeGamma",
                [] {
                    RmsNormBwdGraphSpec spec;
                    spec.scaleDataType = DataType::FLOAT;
                    spec.dscaleDataType = DataType::BFLOAT16;
                    return spec;
                }},
        // inv_rms is one f32 per row, dense.
        BwdCase{"InvRmsInHalf",
                [] {
                    RmsNormBwdGraphSpec spec;
                    spec.invRmsDataType = DataType::HALF;
                    return spec;
                }},
        BwdCase{"InvRmsPerColumn",
                [] {
                    RmsNormBwdGraphSpec spec;
                    spec.invRmsDims = std::vector<int64_t>{1, 4096};
                    return spec;
                }},
        BwdCase{"InvRmsStrided",
                [] {
                    RmsNormBwdGraphSpec spec;
                    spec.strides[FLYDSL_BWD_INV_RMS_UID] = {2, 1};
                    return spec;
                }},
        // Every row operand packed: the kernels pass the width as the row stride.
        BwdCase{"PaddedRowsInX",
                [] {
                    RmsNormBwdGraphSpec spec;
                    spec.strides[FLYDSL_BWD_X_UID] = {4160, 1};
                    return spec;
                }},
        BwdCase{"PaddedRowsInDy",
                [] {
                    RmsNormBwdGraphSpec spec;
                    spec.strides[FLYDSL_BWD_DY_UID] = {4160, 1};
                    return spec;
                }},
        BwdCase{"DxOfAnotherShape",
                [] {
                    RmsNormBwdGraphSpec spec;
                    spec.dxDims = std::vector<int64_t>{8, 2048};
                    return spec;
                }},
        BwdCase{"VirtualDx",
                [] {
                    RmsNormBwdGraphSpec spec;
                    spec.virtualTensors.insert(FLYDSL_BWD_DX_UID);
                    return spec;
                }},
        BwdCase{"DyAlignedTo8",
                [] {
                    RmsNormBwdGraphSpec spec;
                    spec.alignment[FLYDSL_BWD_DY_UID] = 8;
                    return spec;
                }},
        BwdCase{"GammaNotTheNormalisedShape",
                [] {
                    RmsNormBwdGraphSpec spec;
                    spec.scaleDims = std::vector<int64_t>{8, 4096};
                    return spec;
                }},
        // 2^30 bf16 elements is one byte past one buffer view's 32-bit offsets.
        BwdCase{"PastThirtyTwoBitAddressing",
                [] {
                    RmsNormBwdGraphSpec spec;
                    spec.xDims = {1 << 20, 1 << 10};
                    return spec;
                }}),
    bwdCaseName);

TEST(TestFlydslRmsNormBwdBinding, BindsEveryTokenTheDispatchReads)
{
    RmsNormBwdGraphSpec spec;
    spec.xDims = {2, 3, 4096};
    const GraphFixture fixture(buildFlydslRmsNormBwdGraph(spec));
    const auto bound = matchesGraph(FLYDSL_RMSNORM_BWD, fixture.context());
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_RMSNORM_BWD.inputAToken), FLYDSL_BWD_X_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_RMSNORM_BWD.inputBToken), FLYDSL_BWD_SCALE_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_RMSNORM_BWD.outputToken), FLYDSL_BWD_DX_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_BWD_DY_TOKEN), FLYDSL_BWD_DY_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_BWD_INV_RMS_TOKEN), FLYDSL_BWD_INV_RMS_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_BWD_DSCALE_TOKEN), FLYDSL_BWD_DSCALE_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_BWD_ROWS_TOKEN), 6);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_BWD_COLUMNS_TOKEN), 4096);
}

// ---------------------------------------------------------------------------
// kernel_match and score
// ---------------------------------------------------------------------------

bool kernelServes(const RmsNormBwdGraphSpec& spec, const KernelDefinition& kernel)
{
    const GraphFixture fixture(buildFlydslRmsNormBwdGraph(spec));
    return matchesKernel(
        FLYDSL_RMSNORM_BWD, fixture.context(), kernel, bindingsFor(fixture.context()));
}

TEST(TestFlydslRmsNormBwdKernelMatcher, ABakedObjectServesItsOwnWidthOnly)
{
    const RmsNormBwdGraphSpec spec;
    EXPECT_TRUE(kernelServes(spec, makeBwdKernel("bf16", 4096, 4096)));
    EXPECT_FALSE(kernelServes(spec, makeBwdKernel("bf16", 8192, 8192)));
}

TEST(TestFlydslRmsNormBwdKernelMatcher, AGenericObjectServesUpToItsBound)
{
    RmsNormBwdGraphSpec spec;
    spec.xDims = {8, 1024};
    EXPECT_TRUE(kernelServes(spec, makeBwdKernel("bf16", 0, 1024, 0, 30)));
    spec.xDims = {8, 1025};
    // One column past the bound: the object's register accumulators do not reach it.
    EXPECT_FALSE(kernelServes(spec, makeBwdKernel("bf16", 0, 1024, 0, 30)));
    EXPECT_TRUE(kernelServes(spec, makeBwdKernel("bf16", 0, 8192, 0, 20)));
}

TEST(TestFlydslRmsNormBwdKernelMatcher, PairsDtypeAndGammaForm)
{
    const RmsNormBwdGraphSpec plain;
    EXPECT_FALSE(kernelServes(plain, makeBwdKernel("f16", 4096, 4096)));
    EXPECT_FALSE(kernelServes(plain, makeBwdKernel("bf16", 4096, 4096, 1)));

    RmsNormBwdGraphSpec mixed;
    mixed.scaleDataType = DataType::FLOAT;
    EXPECT_TRUE(kernelServes(mixed, makeBwdKernel("bf16", 4096, 4096, 1)));
    EXPECT_FALSE(kernelServes(mixed, makeBwdKernel("bf16", 4096, 4096, 0)));
}

TEST(TestFlydslRmsNormBwdScore, RanksBakedThenTheNarrowerGenericTier)
{
    const GraphFixture fixture(buildFlydslRmsNormBwdGraph());
    const auto bound = bindingsFor(fixture.context());
    const auto baked = scoreKernel(
        FLYDSL_RMSNORM_BWD, fixture.context(), makeBwdKernel("bf16", 4096, 4096), bound);
    const auto narrow = scoreKernel(
        FLYDSL_RMSNORM_BWD, fixture.context(), makeBwdKernel("bf16", 0, 8192, 0, 20), bound);
    const auto wide = scoreKernel(
        FLYDSL_RMSNORM_BWD, fixture.context(), makeBwdKernel("bf16", 0, 16384, 0, 10), bound);
    EXPECT_GT(baked, narrow);
    EXPECT_GT(narrow, wide);
}

TEST(TestFlydslRmsNormBwdDispatch, SizesOneF32DgammaRowPerProgram)
{
    // At most one program per row: 8 rows on a 64-CU device is 8 programs. Above 1024 rows
    // the scalar path takes 2 per CU, so 4096 rows is 128.
    auto properties = testDeviceProperties();
    properties.multiProcessorCount = 64;
    const GraphFixture fewRows(buildFlydslRmsNormBwdGraph(), properties);
    EXPECT_EQ(dispatchHandler(FLYDSL_RMSNORM_BWD)
                  .workspaceBytes(fewRows.context(),
                                  bindingsFor(fewRows.context()),
                                  makeBwdKernel("bf16", 4096, 4096)),
              size_t{8} * 4096 * sizeof(float));

    RmsNormBwdGraphSpec spec;
    spec.xDims = {4096, 1024};
    const GraphFixture manyRows(buildFlydslRmsNormBwdGraph(spec), properties);
    EXPECT_EQ(dispatchHandler(FLYDSL_RMSNORM_BWD)
                  .workspaceBytes(manyRows.context(),
                                  bindingsFor(manyRows.context()),
                                  makeBwdKernel("bf16", 0, 1024, 0, 30)),
              size_t{128} * 1024 * sizeof(float));

    // The vec8 path (bf16 at a baked 4096) takes 1.5 per CU above 2048 rows: 96.
    RmsNormBwdGraphSpec vec;
    vec.xDims = {4096, 4096};
    const GraphFixture vecRows(buildFlydslRmsNormBwdGraph(vec), properties);
    EXPECT_EQ(dispatchHandler(FLYDSL_RMSNORM_BWD)
                  .workspaceBytes(vecRows.context(),
                                  bindingsFor(vecRows.context()),
                                  makeBwdKernel("bf16", 4096, 4096)),
              size_t{96} * 4096 * sizeof(float));
}

// ---------------------------------------------------------------------------
// dispatch on device
// ---------------------------------------------------------------------------

/// Device x, dy, gamma, inv_rms (f32), dx and dgamma for one backward, freed on scope
/// exit. Inputs are exact in bf16; inv_rms is the forward's statistic of the same x. dx
/// and dgamma start as NaN, so an element the kernels do not write fails the comparison.
template <typename T, typename G = T>
class RmsNormBwdBuffers
{
public:
    RmsNormBwdBuffers(int64_t rows, int64_t columns, size_t workspaceBytes)
        : _rows(rows)
        , _columns(columns)
        , _x(static_cast<size_t>(rows * columns))
        , _dy(_x.size())
        , _gamma(static_cast<size_t>(columns))
        , _invRms(static_cast<size_t>(rows))
    {
        for(int64_t row = 0; row < rows; ++row)
        {
            double sumOfSquares = 0.0;
            for(int64_t column = 0; column < columns; ++column)
            {
                const auto index = static_cast<size_t>(row * columns + column);
                _x[index] = T(0.5F + static_cast<float>((row + column) % 8) * 0.125F);
                _dy[index] = T(-1.0F + static_cast<float>((3 * row + column) % 16) * 0.125F);
                const auto value = static_cast<double>(static_cast<float>(_x[index]));
                sumOfSquares += value * value;
            }
            _invRms[static_cast<size_t>(row)] = static_cast<float>(
                1.0 / std::sqrt(sumOfSquares / static_cast<double>(columns) + 1e-5));
        }
        for(int64_t column = 0; column < columns; ++column)
        {
            _gamma[static_cast<size_t>(column)] = G(0.5F + static_cast<float>(column % 4) * 0.25F);
        }
        upload(_xDevice, _x);
        upload(_dyDevice, _dy);
        upload(_gammaDevice, _gamma);
        upload(_invRmsDevice, _invRms);
        EXPECT_EQ(hipSuccess, hipMalloc(&_dxDevice, _x.size() * sizeof(T)));
        EXPECT_EQ(hipSuccess, hipMalloc(&_dgammaDevice, _gamma.size() * sizeof(G)));
        EXPECT_EQ(hipSuccess, hipMalloc(&_workspace, std::max<size_t>(workspaceBytes, 1)));
        poison();
    }

    ~RmsNormBwdBuffers()
    {
        for(void* pointer : {_xDevice,
                             _dyDevice,
                             _gammaDevice,
                             _invRmsDevice,
                             _dxDevice,
                             _dgammaDevice,
                             _workspace})
        {
            static_cast<void>(hipFree(pointer));
        }
    }

    RmsNormBwdBuffers(const RmsNormBwdBuffers&) = delete;
    RmsNormBwdBuffers& operator=(const RmsNormBwdBuffers&) = delete;

    /// dx, dgamma and the workspace back to NaN.
    void poison() const
    {
        EXPECT_EQ(hipSuccess, hipMemset(_dxDevice, 0xFF, _x.size() * sizeof(T)));
        EXPECT_EQ(hipSuccess, hipMemset(_dgammaDevice, 0xFF, _gamma.size() * sizeof(G)));
    }

    std::vector<hipdnnPluginDeviceBuffer_t> descriptors() const
    {
        return {hipdnnPluginDeviceBuffer_t{FLYDSL_BWD_DY_UID, _dyDevice},
                hipdnnPluginDeviceBuffer_t{FLYDSL_BWD_X_UID, _xDevice},
                hipdnnPluginDeviceBuffer_t{FLYDSL_BWD_SCALE_UID, _gammaDevice},
                hipdnnPluginDeviceBuffer_t{FLYDSL_BWD_INV_RMS_UID, _invRmsDevice},
                hipdnnPluginDeviceBuffer_t{FLYDSL_BWD_DX_UID, _dxDevice},
                hipdnnPluginDeviceBuffer_t{FLYDSL_BWD_DSCALE_UID, _dgammaDevice}};
    }

    void* workspace() const
    {
        return _workspace;
    }

    std::vector<T> readDx() const
    {
        return download<T>(_dxDevice, _x.size());
    }

    std::vector<G> readDgamma() const
    {
        return download<G>(_dgammaDevice, _gamma.size());
    }

    /// dx = (dy*g - xhat*c1) * rstd with xhat = x*rstd, c1 = mean(xhat*dy*g); dgamma =
    /// sum over rows of dy*xhat. In double, from the quantised inputs and the f32 inv_rms.
    std::pair<std::vector<double>, std::vector<double>> reference() const
    {
        std::vector<double> dx(_x.size());
        std::vector<double> dgamma(_gamma.size(), 0.0);
        for(int64_t row = 0; row < _rows; ++row)
        {
            const auto rstd = static_cast<double>(_invRms[static_cast<size_t>(row)]);
            double c1 = 0.0;
            for(int64_t column = 0; column < _columns; ++column)
            {
                const auto index = static_cast<size_t>(row * _columns + column);
                c1 += value(_x[index]) * rstd * value(_dy[index])
                      * value(_gamma[static_cast<size_t>(column)]);
            }
            c1 /= static_cast<double>(_columns);
            for(int64_t column = 0; column < _columns; ++column)
            {
                const auto index = static_cast<size_t>(row * _columns + column);
                const auto xhat = value(_x[index]) * rstd;
                dx[index]
                    = (value(_dy[index]) * value(_gamma[static_cast<size_t>(column)]) - xhat * c1)
                      * rstd;
                dgamma[static_cast<size_t>(column)] += value(_dy[index]) * xhat;
            }
        }
        return {dx, dgamma};
    }

private:
    template <typename V>
    static double value(V element)
    {
        return static_cast<double>(static_cast<float>(element));
    }

    template <typename V>
    static void upload(void*& device, const std::vector<V>& host)
    {
        EXPECT_EQ(hipSuccess, hipMalloc(&device, host.size() * sizeof(V)));
        EXPECT_EQ(hipSuccess,
                  hipMemcpy(device, host.data(), host.size() * sizeof(V), hipMemcpyHostToDevice));
    }

    template <typename V>
    static std::vector<V> download(void* device, size_t count)
    {
        std::vector<V> host(count);
        EXPECT_EQ(hipSuccess,
                  hipMemcpy(host.data(), device, count * sizeof(V), hipMemcpyDeviceToHost));
        return host;
    }

    int64_t _rows;
    int64_t _columns;
    std::vector<T> _x;
    std::vector<T> _dy;
    std::vector<G> _gamma;
    std::vector<float> _invRms;
    void* _xDevice = nullptr;
    void* _dyDevice = nullptr;
    void* _gammaDevice = nullptr;
    void* _invRmsDevice = nullptr;
    void* _dxDevice = nullptr;
    void* _dgammaDevice = nullptr;
    void* _workspace = nullptr;
};

/// Largest error relative to max(|expected|, floor).
template <typename V>
double worstRelativeError(const std::vector<V>& actual,
                          const std::vector<double>& expected,
                          double floor)
{
    double worst = 0.0;
    for(size_t i = 0; i < actual.size(); ++i)
    {
        const auto got = static_cast<double>(static_cast<float>(actual[i]));
        const double error = std::isfinite(got) ? std::abs(got - expected[i])
                                                      / std::max(std::abs(expected[i]), floor)
                                                : std::numeric_limits<double>::infinity();
        worst = std::max(worst, error);
    }
    return worst;
}

/// Runs the staged (dtype, N, n_max, form) object over a rows x columns graph twice,
/// checks dx and dgamma against the host reference and that the two runs agree bitwise.
template <typename T, typename G = T>
void expectBackwardMatchesReference(DataType dataType,
                                    const std::string& dtype,
                                    int64_t bakedN,
                                    int64_t nMax,
                                    int64_t rows,
                                    int64_t columns,
                                    double relativeTolerance)
{
    constexpr bool F32_SCALE = std::is_same_v<G, float> && !std::is_same_v<T, float>;
    KernelDefinition kernel;
    if(!findStagedBwdKernel(dtype, bakedN, nMax, F32_SCALE ? 1 : 0, kernel))
    {
        GTEST_SKIP() << "nothing packed for dtype=" << dtype << " N=" << bakedN << " n_max=" << nMax
                     << " in this build's descriptor root";
    }

    RmsNormBwdGraphSpec spec;
    spec.dataType = dataType;
    spec.xDims = {rows, columns};
    if(F32_SCALE)
    {
        spec.scaleDataType = DataType::FLOAT;
    }
    const GraphFixture fixture(buildFlydslRmsNormBwdGraph(spec), packedArchDeviceProperties());
    const auto& handler = dispatchHandler(FLYDSL_RMSNORM_BWD);
    const auto bound = bindingsFor(fixture.context());
    ASSERT_TRUE(matchesKernel(FLYDSL_RMSNORM_BWD, fixture.context(), kernel, bound));
    const auto workspaceBytes = handler.workspaceBytes(fixture.context(), bound, kernel);
    const auto prepared = handler.prepare(fixture.context(), bound, kernel);
    ASSERT_NE(prepared, nullptr);

    const RmsNormBwdBuffers<T, G> buffers(rows, columns, workspaceBytes);
    const auto descriptors = buffers.descriptors();
    const Handle handle;
    const auto run = [&] {
        buffers.poison();
        handler.launch(handle,
                       *prepared,
                       descriptors.data(),
                       static_cast<uint32_t>(descriptors.size()),
                       buffers.workspace());
        EXPECT_EQ(hipSuccess, hipDeviceSynchronize());
        return std::make_pair(buffers.readDx(), buffers.readDgamma());
    };

    const auto [dx, dgamma] = run();
    const auto [expectedDx, expectedDgamma] = buffers.reference();
    EXPECT_LE(worstRelativeError(dx, expectedDx, 1e-1), relativeTolerance) << "dx";
    // dgamma sums `rows` products, so its scale grows with the row count.
    EXPECT_LE(worstRelativeError(dgamma, expectedDgamma, 1.0), relativeTolerance) << "dgamma";

    // Deterministic: the same programs, rows and reduction order on every launch.
    const auto [dxAgain, dgammaAgain] = run();
    EXPECT_EQ(0, std::memcmp(dx.data(), dxAgain.data(), dx.size() * sizeof(T)));
    EXPECT_EQ(0, std::memcmp(dgamma.data(), dgammaAgain.data(), dgamma.size() * sizeof(G)));
}

TEST(TestGpuFlydslRmsNormBwdDispatch, ComputesAtABakedWidth)
{
    SKIP_IF_NO_DEVICES();
    // The vec8 path (N >= 4096) and the unrolled scalar one (3072).
    expectBackwardMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        DataType::BFLOAT16, "bf16", 4096, 4096, 64, 4096, 2e-2);
    expectBackwardMatchesReference<hipdnn_data_sdk::types::half>(
        DataType::HALF, "f16", 3072, 3072, 37, 3072, 5e-3);
}

TEST(TestGpuFlydslRmsNormBwdDispatch, ComputesThroughTheGenericTiers)
{
    SKIP_IF_NO_DEVICES();
    expectBackwardMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        DataType::BFLOAT16, "bf16", 0, 1024, 50, 128, 2e-2);
    expectBackwardMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        DataType::BFLOAT16, "bf16", 0, 8192, 37, 4097, 2e-2);
    expectBackwardMatchesReference<float>(DataType::FLOAT, "f32", 0, 8192, 11, 1000, 1e-4);
}

TEST(TestGpuFlydslRmsNormBwdDispatch, ComputesAnF32GammaBesideBf16AndF16)
{
    SKIP_IF_NO_DEVICES();
    expectBackwardMatchesReference<hipdnn_data_sdk::types::half, float>(
        DataType::HALF, "f16", 8192, 8192, 16, 8192, 5e-3);
    expectBackwardMatchesReference<hipdnn_data_sdk::types::bfloat16, float>(
        DataType::BFLOAT16, "bf16", 0, 8192, 9, 2000, 2e-2);
}

TEST(TestGpuFlydslRmsNormBwdDispatch, ComputesWithManyRowsPerProgram)
{
    SKIP_IF_NO_DEVICES();
    // Far more rows than programs, so each program walks many rows and the reduce kernel
    // sums every program's row.
    expectBackwardMatchesReference<hipdnn_data_sdk::types::bfloat16>(
        DataType::BFLOAT16, "bf16", 0, 1024, 4096, 256, 2e-2);
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
