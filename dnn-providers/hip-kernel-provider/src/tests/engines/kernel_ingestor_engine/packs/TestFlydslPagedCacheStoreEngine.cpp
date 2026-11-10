// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see TestFlydslRmsNormEngine.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "engines/kernel_ingestor_engine/packs/FlydslPagedCacheStoreTestGraphs.hpp"

/**
 * @file TestFlydslPagedCacheStoreEngine.cpp
 * @brief hipkernel:flydsl_paged_cache_store's graph match, kernel match and score, on the
 *        CPU: what it accepts, and every refusal.
 *
 * The real launches against the CPU reference are TestFlydslPagedCacheStoreObjects.cpp; the
 * census of what ships is TestFlydslPagedCacheStorePacks.cpp.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using data_objects::DataType;

std::optional<BoundTokens> match(const PagedCacheStoreGraphSpec& spec)
{
    const GraphFixture fixture(buildFlydslPagedCacheStoreGraph(spec));
    return matchesGraph(FLYDSL_PAGED_CACHE_STORE, fixture.context());
}

bool accepts(const PagedCacheStoreGraphSpec& spec)
{
    return match(spec).has_value();
}

KernelDefinition
    makeStoreKernel(const std::string& dtype, const std::string& cacheDtype, int64_t priority = 10)
{
    KernelDefinition kernel;
    kernel.name = "paged_cache_store_" + dtype + "_cache" + cacheDtype;
    kernel.priority = priority;
    kernel.metadata = {{std::string(FLYDSL_DTYPE_FIELD), dtype},
                       {std::string(FLYDSL_CACHE_DTYPE_FIELD), cacheDtype},
                       {std::string(FLYDSL_BLOCK_THREADS_FIELD), int64_t{256}}};
    return kernel;
}

bool kernelAccepts(const PagedCacheStoreGraphSpec& spec, const KernelDefinition& kernel)
{
    const GraphFixture fixture(buildFlydslPagedCacheStoreGraph(spec));
    return matchesKernel(FLYDSL_PAGED_CACHE_STORE, fixture.context(), kernel);
}

/// SDPA-native x [B, H, S, D] viewed out of a [B, S, H, D] projection buffer.
PagedCacheStoreGraphSpec sdpaNative()
{
    PagedCacheStoreGraphSpec spec;
    spec.xDims = {2, 4, 3, 64};
    spec.xStrides = std::vector<int64_t>{int64_t{3} * 4 * 64, 64, int64_t{4} * 64, 1};
    spec.containerDims = {6, 4, 16, 64};
    return spec;
}

// ---------------------------------------------------------------------------
// Accepted
// ---------------------------------------------------------------------------

TEST(TestFlydslPagedCacheStoreEngine, BindsTheOperandsOfATokenPackedGraph)
{
    const auto bound = match({});
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_PAGED_CACHE_STORE.inputAToken), FLYDSL_PCS_X_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_PAGED_CACHE_STORE.inputBToken), FLYDSL_PCS_SLOT_UID);
    EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_PAGED_CACHE_STORE.outputToken),
              FLYDSL_PCS_CONTAINER_UID);
    EXPECT_FALSE(tryGetBoundInt(*bound, FLYDSL_PCS_SCALE_TOKEN).has_value());
}

TEST(TestFlydslPagedCacheStoreEngine, AcceptsSdpaNativeXWithAnyStrides)
{
    EXPECT_TRUE(accepts(sdpaNative()));

    // Strided slot mapping, padded container rows, and a non-unit innermost x stride: all
    // runtime strides of the scalar path.
    auto spec = sdpaNative();
    spec.slotStrides = std::vector<int64_t>{8, 2};
    spec.containerStrides = std::vector<int64_t>{int64_t{4} * 16 * 80, int64_t{16} * 80, 80, 1};
    EXPECT_TRUE(accepts(spec));
    spec.xStrides
        = std::vector<int64_t>{int64_t{2} * 3 * 4 * 64, int64_t{2} * 64, int64_t{2} * 4 * 64, 2};
    EXPECT_TRUE(accepts(spec));

    // A head-major container: [nb, H, bs, D] dims over [nb, bs, H, D] memory.
    spec.containerStrides = std::vector<int64_t>{int64_t{16} * 4 * 64, 64, int64_t{4} * 64, 1};
    EXPECT_TRUE(accepts(spec));
}

TEST(TestFlydslPagedCacheStoreEngine, AcceptsEveryXDtypeIntoItselfAndIntoFp8)
{
    for(const auto xType : {DataType::BFLOAT16, DataType::HALF, DataType::FLOAT})
    {
        for(const auto cacheType : {xType,
                                    DataType::FP8_E4M3,
                                    DataType::FP8_E5M2,
                                    DataType::FP8_E4M3_FNUZ,
                                    DataType::FP8_E5M2_FNUZ})
        {
            PagedCacheStoreGraphSpec spec;
            spec.xDataType = xType;
            spec.containerDataType = cacheType;
            EXPECT_TRUE(accepts(spec))
                << static_cast<int>(xType) << " into " << static_cast<int>(cacheType);
        }
    }
}

TEST(TestFlydslPagedCacheStoreEngine, AcceptsInt64SlotsAndEveryScaleForm)
{
    PagedCacheStoreGraphSpec spec;
    spec.slotDataType = DataType::INT64;
    EXPECT_TRUE(accepts(spec));

    for(const auto scale : {PcsScale::DEVICE, PcsScale::BAKED, PcsScale::RUNTIME})
    {
        spec.scale = scale;
        const auto bound = match(spec);
        ASSERT_TRUE(bound.has_value()) << static_cast<int>(scale);
        EXPECT_EQ(tryGetBoundInt(*bound, FLYDSL_PCS_SCALE_TOKEN), FLYDSL_PCS_SCALE_UID);
    }

    // A one-element scale of any rank.
    spec.scale = PcsScale::DEVICE;
    spec.scaleDims = {1, 1, 1, 1};
    EXPECT_TRUE(accepts(spec));

    spec.computeDataType = DataType::UNSET;
    EXPECT_TRUE(accepts(spec));
}

// ---------------------------------------------------------------------------
// Declined
// ---------------------------------------------------------------------------

TEST(TestFlydslPagedCacheStoreEngine, DeclinesGraphsThatAreNotOneStoreNode)
{
    PagedCacheStoreGraphSpec spec;
    spec.secondNode = true;
    EXPECT_FALSE(accepts(spec));

    spec = {};
    spec.overrideShapeEnabled = true;
    EXPECT_FALSE(accepts(spec));

    // The rmsnorm graph is a different union arm.
    const GraphFixture rmsnorm(buildFlydslRmsNormGraph());
    EXPECT_FALSE(matchesGraph(FLYDSL_PAGED_CACHE_STORE, rmsnorm.context()).has_value());
}

TEST(TestFlydslPagedCacheStoreEngine, DeclinesDtypesNoObjectIsBuiltFor)
{
    PagedCacheStoreGraphSpec spec;
    spec.computeDataType = DataType::HALF;
    EXPECT_FALSE(accepts(spec)) << "the product is formed in f32";

    spec = {};
    spec.xDataType = DataType::FP8_E4M3;
    EXPECT_FALSE(accepts(spec)) << "fp8 x";

    spec = {};
    spec.xDataType = DataType::HALF;
    spec.containerDataType = DataType::BFLOAT16;
    EXPECT_FALSE(accepts(spec)) << "a 16-bit container of another type";

    spec = {};
    spec.containerDataType = DataType::FLOAT;
    EXPECT_FALSE(accepts(spec)) << "a widening container";

    spec = {};
    spec.containerDataType = DataType::INT8;
    EXPECT_FALSE(accepts(spec));

    spec = {};
    spec.slotDataType = DataType::FLOAT;
    EXPECT_FALSE(accepts(spec));

    spec = {};
    spec.scale = PcsScale::DEVICE;
    spec.scaleDataType = DataType::HALF;
    EXPECT_FALSE(accepts(spec));
}

TEST(TestFlydslPagedCacheStoreEngine, DeclinesShapesTheSchemaDoesNotDefine)
{
    PagedCacheStoreGraphSpec spec;
    spec.xDims = {5, 128};
    spec.slotDims = std::vector<int64_t>{5};
    EXPECT_FALSE(accepts(spec)) << "rank-2 x";

    spec = {};
    spec.slotDims = std::vector<int64_t>{4};
    EXPECT_FALSE(accepts(spec)) << "slot_mapping shorter than T";

    spec = sdpaNative();
    spec.slotDims = std::vector<int64_t>{int64_t{2} * 3};
    EXPECT_FALSE(accepts(spec)) << "[T] slots beside [B, H, S, D] x";

    spec = {};
    spec.containerDims = {4, 3, 8, 64};
    EXPECT_FALSE(accepts(spec)) << "container H differs";

    spec = {};
    spec.containerDims = {4, 2, 8, 128};
    EXPECT_FALSE(accepts(spec)) << "container D differs";

    spec = {};
    spec.containerDims = {4, 2, 8};
    EXPECT_FALSE(accepts(spec)) << "rank-3 container";

    spec = {};
    spec.xDims = {0, 2, 64};
    EXPECT_FALSE(accepts(spec)) << "no tokens";

    spec = {};
    spec.scale = PcsScale::DEVICE;
    spec.scaleDims = {2};
    EXPECT_FALSE(accepts(spec)) << "a scale of two elements";
}

TEST(TestFlydslPagedCacheStoreEngine, DeclinesOperandsTheKernelCannotAddress)
{
    PagedCacheStoreGraphSpec spec;
    spec.virtualContainer = true;
    EXPECT_FALSE(accepts(spec));

    spec = {};
    spec.xStrides = std::vector<int64_t>{128, -64, 1};
    EXPECT_FALSE(accepts(spec)) << "negative stride";

    // Two blocks on one address, and two heads on one address.
    spec = {};
    spec.containerStrides = std::vector<int64_t>{0, 512, 64, 1};
    EXPECT_FALSE(accepts(spec));
    spec.containerStrides = std::vector<int64_t>{1024, 32, 64, 1};
    EXPECT_FALSE(accepts(spec));

    // One slot's row past the 32-bit offsets the kernel addresses it with.
    spec = {};
    spec.containerStrides = std::vector<int64_t>{int64_t{1} << 32, int64_t{1} << 30, 64, 1};
    EXPECT_FALSE(accepts(spec));

    // More slots than an int32 counts.
    spec = {};
    spec.containerDims = {int64_t{1} << 28, 2, 16, 64};
    EXPECT_FALSE(accepts(spec));

    // A batch past the grid's y.
    spec = sdpaNative();
    spec.xDims = {70000, 4, 1, 64};
    spec.xStrides = std::nullopt;
    EXPECT_FALSE(accepts(spec));
}

// ---------------------------------------------------------------------------
// Kernel match and score
// ---------------------------------------------------------------------------

TEST(TestFlydslPagedCacheStoreEngine, PairsEachGraphWithItsDtypePairOnly)
{
    PagedCacheStoreGraphSpec spec;
    EXPECT_TRUE(kernelAccepts(spec, makeStoreKernel("bf16", "bf16")));
    EXPECT_FALSE(kernelAccepts(spec, makeStoreKernel("f16", "f16")));
    EXPECT_FALSE(kernelAccepts(spec, makeStoreKernel("bf16", "fp8_e4m3")));

    spec.containerDataType = DataType::FP8_E4M3;
    EXPECT_TRUE(kernelAccepts(spec, makeStoreKernel("bf16", "fp8_e4m3")));
    EXPECT_FALSE(kernelAccepts(spec, makeStoreKernel("bf16", "fp8_e4m3_fnuz")));
    EXPECT_FALSE(kernelAccepts(spec, makeStoreKernel("bf16", "bf16")));

    spec.containerDataType = DataType::FP8_E5M2_FNUZ;
    spec.xDataType = DataType::FLOAT;
    EXPECT_TRUE(kernelAccepts(spec, makeStoreKernel("f32", "fp8_e5m2_fnuz")));
    EXPECT_FALSE(kernelAccepts(spec, makeStoreKernel("f32", "fp8_e5m2")));
}

TEST(TestFlydslPagedCacheStoreEngine, ScoresByPriority)
{
    const GraphFixture fixture(buildFlydslPagedCacheStoreGraph());
    EXPECT_EQ(scoreKernel(
                  FLYDSL_PAGED_CACHE_STORE, fixture.context(), makeStoreKernel("bf16", "bf16", 10)),
              10.0);
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
