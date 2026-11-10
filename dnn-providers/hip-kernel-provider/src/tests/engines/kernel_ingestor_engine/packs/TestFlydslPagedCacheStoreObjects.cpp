// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see TestFlydslRmsNormEngine.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <hip/hip_runtime_api.h>

#include <hipdnn_data_sdk/types/Bfloat16.hpp>
#include <hipdnn_data_sdk/types/Fp8E4M3.hpp>
#include <hipdnn_data_sdk/types/Fp8E4M3Fnuz.hpp>
#include <hipdnn_data_sdk/types/Fp8E5M2.hpp>
#include <hipdnn_data_sdk/types/Fp8E5M2Fnuz.hpp>
#include <hipdnn_data_sdk/types/Half.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferencePagedCacheStore.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslNormDeviceTestSupport.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslPagedCacheStoreTestGraphs.hpp"

/**
 * @file TestFlydslPagedCacheStoreObjects.cpp
 * @brief Every FlyDSL paged-cache-store object this device's shard carries, launched through
 *        its pack's dispatch against hipDNN's CPU reference.
 *
 * Each object runs four graphs: token-packed x on the vector path; SDPA-native x out of a
 * [B, S, H, D] buffer with INT64 slots into a head-major, padded container; a scalar-path
 * graph (odd D, a strided innermost x axis); and an FP8-range one whose scale drives values
 * past the format's largest finite and below its smallest normal. The scale rotates over
 * absent, device tensor and runtime pass-by-value.
 *
 * The container starts as random bytes, and every slot nothing addresses -- including the
 * tokens with negative slots, and the padding the strides skip -- must keep them: the whole
 * allocation is compared byte for byte with the reference's, which casts the same f32
 * product once with round-to-nearest-even and saturation.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
namespace types = hipdnn_data_sdk::types;
using data_objects::DataType;
using hipdnn_data_sdk::utilities::Tensor;

std::string stringField(const KernelDefinition& kernel, const char* field)
{
    const auto* value = tryGetMetadataField<std::string>(kernel.metadata, field);
    return value == nullptr ? std::string() : *value;
}

template <typename T>
struct GraphType;
template <>
struct GraphType<types::bfloat16>
{
    static constexpr DataType VALUE = DataType::BFLOAT16;
};
template <>
struct GraphType<types::half>
{
    static constexpr DataType VALUE = DataType::HALF;
};
template <>
struct GraphType<float>
{
    static constexpr DataType VALUE = DataType::FLOAT;
};
template <>
struct GraphType<types::fp8_e4m3>
{
    static constexpr DataType VALUE = DataType::FP8_E4M3;
};
template <>
struct GraphType<types::fp8_e5m2>
{
    static constexpr DataType VALUE = DataType::FP8_E5M2;
};
template <>
struct GraphType<types::fp8_e4m3_fnuz>
{
    static constexpr DataType VALUE = DataType::FP8_E4M3_FNUZ;
};
template <>
struct GraphType<types::fp8_e5m2_fnuz>
{
    static constexpr DataType VALUE = DataType::FP8_E5M2_FNUZ;
};
template <>
struct GraphType<int32_t>
{
    static constexpr DataType VALUE = DataType::INT32;
};
template <>
struct GraphType<int64_t>
{
    static constexpr DataType VALUE = DataType::INT64;
};

/// One graph an object runs.
struct StoreCase
{
    std::string name;
    std::vector<int64_t> xDims;
    std::optional<std::vector<int64_t>> xStrides;
    std::optional<std::vector<int64_t>> slotStrides;
    std::vector<int64_t> containerDims;
    std::optional<std::vector<int64_t>> containerStrides;
    bool int64Slots = false;
    PcsScale scale = PcsScale::NONE;
    float scaleValue = 1.0F;
    /// x is drawn from [-xRange, xRange].
    float xRange = 4.0F;
};

std::vector<int64_t> packedStrides(const std::vector<int64_t>& dims)
{
    return flydslPackedStrides(dims);
}

template <typename T>
void fillRandomBytes(Tensor<T>& tensor, std::mt19937& generator)
{
    auto* bytes = reinterpret_cast<unsigned char*>(tensor.memory().hostData());
    std::uniform_int_distribution<int> byte(0, 255);
    for(size_t i = 0; i < tensor.elementSpace() * sizeof(T); ++i)
    {
        bytes[i] = static_cast<unsigned char>(byte(generator));
    }
    tensor.memory().markHostModified();
}

template <typename XT, typename CT, typename ST>
void expectStoreMatches(const KernelDefinition& kernel, const StoreCase& storeCase)
{
    SCOPED_TRACE(kernel.name + " / " + storeCase.name);
    std::mt19937 generator(
        static_cast<uint32_t>(storeCase.xDims[0] * 7919 + storeCase.xDims.back()));

    const bool tokenPacked = storeCase.xDims.size() == 3;
    const std::vector<int64_t> slotDims
        = tokenPacked ? std::vector<int64_t>{storeCase.xDims[0]}
                      : std::vector<int64_t>{storeCase.xDims[0], storeCase.xDims[2]};
    const auto xStrides = storeCase.xStrides.value_or(packedStrides(storeCase.xDims));
    const auto slotStrides = storeCase.slotStrides.value_or(packedStrides(slotDims));
    const auto containerStrides
        = storeCase.containerStrides.value_or(packedStrides(storeCase.containerDims));

    Tensor<XT> x(storeCase.xDims, xStrides);
    fillRandomBytes(x, generator); // the gaps the strides skip
    std::uniform_real_distribution<float> value(-storeCase.xRange, storeCase.xRange);
    for(auto* element : x)
    {
        *static_cast<XT*>(element) = static_cast<XT>(value(generator));
    }
    x.memory().markHostModified();

    // A distinct slot per token from a shuffled range, about one token in five skipped
    // with a negative slot (INT64 slots also with one far below -2^32).
    const auto slotCount = storeCase.containerDims[0] * storeCase.containerDims[2];
    std::vector<int64_t> pool(static_cast<size_t>(slotCount));
    std::iota(pool.begin(), pool.end(), int64_t{0});
    std::shuffle(pool.begin(), pool.end(), generator);
    Tensor<ST> slots(slotDims, slotStrides);
    slots.fillWithValue(ST{-7});
    size_t next = 0;
    size_t token = 0;
    for(auto* element : slots)
    {
        int64_t slot = pool.at(next++);
        if(token % 5 == 3)
        {
            slot = (std::is_same_v<ST, int64_t> && token % 10 == 3) ? -(int64_t{1} << 40) : -1;
        }
        *static_cast<ST*>(element) = static_cast<ST>(slot);
        ++token;
    }
    slots.memory().markHostModified();

    Tensor<CT> expected(storeCase.containerDims, containerStrides);
    fillRandomBytes(expected, generator);
    Tensor<CT> container(storeCase.containerDims, containerStrides);
    std::memcpy(container.memory().hostData(),
                expected.memory().hostData(),
                expected.elementSpace() * sizeof(CT));
    container.memory().markHostModified();

    Tensor<float> scale({1});
    scale.fillWithValue(storeCase.scaleValue);
    hipdnn_test_sdk::utilities::CpuFpReferencePagedCacheStore::store(
        x, slots, expected, storeCase.scale == PcsScale::NONE ? nullptr : &scale);

    PagedCacheStoreGraphSpec spec;
    spec.xDataType = GraphType<XT>::VALUE;
    spec.containerDataType = GraphType<CT>::VALUE;
    spec.slotDataType = GraphType<ST>::VALUE;
    spec.xDims = storeCase.xDims;
    spec.xStrides = xStrides;
    spec.slotStrides = slotStrides;
    spec.containerDims = storeCase.containerDims;
    spec.containerStrides = containerStrides;
    spec.scale = storeCase.scale;
    spec.bakedScale = storeCase.scaleValue;
    const GraphFixture fixture(buildFlydslPagedCacheStoreGraph(spec), packedArchDeviceProperties());
    const auto bound = matchesGraph(FLYDSL_PAGED_CACHE_STORE, fixture.context());
    ASSERT_TRUE(bound.has_value()) << "the graph this object is meant for is declined";
    ASSERT_TRUE(matchesKernel(FLYDSL_PAGED_CACHE_STORE, fixture.context(), kernel, *bound));

    const auto& handler = dispatchHandler(FLYDSL_PAGED_CACHE_STORE);
    ASSERT_EQ(handler.workspaceBytes(fixture.context(), *bound, kernel), 0U);
    const auto prepared = handler.prepare(fixture.context(), *bound, kernel);
    ASSERT_NE(prepared, nullptr);

    std::vector<hipdnnPluginDeviceBuffer_t> arguments{
        {FLYDSL_PCS_X_UID, x.memory().deviceData()},
        {FLYDSL_PCS_SLOT_UID, slots.memory().deviceData()},
        {FLYDSL_PCS_CONTAINER_UID, container.memory().deviceData()}};
    float hostScale = storeCase.scaleValue;
    if(storeCase.scale == PcsScale::DEVICE)
    {
        arguments.push_back({FLYDSL_PCS_SCALE_UID, scale.memory().deviceData()});
    }
    else if(storeCase.scale == PcsScale::RUNTIME)
    {
        arguments.push_back({FLYDSL_PCS_SCALE_UID, &hostScale});
    }
    const Handle handle;
    handler.launch(
        handle, *prepared, arguments.data(), static_cast<uint32_t>(arguments.size()), nullptr);
    ASSERT_EQ(hipSuccess, hipDeviceSynchronize());
    container.memory().markDeviceModified();

    const auto* got = reinterpret_cast<const unsigned char*>(container.memory().hostData());
    const auto* want = reinterpret_cast<const unsigned char*>(expected.memory().hostData());
    const auto bytes = expected.elementSpace() * sizeof(CT);
    size_t mismatches = 0;
    size_t first = bytes;
    for(size_t i = 0; i < bytes; ++i)
    {
        if(got[i] != want[i])
        {
            first = std::min(first, i);
            ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0U) << "first differing byte " << first << " of " << bytes << " (element "
                              << first / sizeof(CT) << "): got "
                              << static_cast<int>(first < bytes ? got[first] : 0) << ", expected "
                              << static_cast<int>(first < bytes ? want[first] : 0);
}

/// The graphs every object runs; @p fp8 adds the range case.
std::vector<StoreCase> storeCases(bool fp8)
{
    std::vector<StoreCase> cases;
    {
        StoreCase c;
        c.name = "token-packed, vector path";
        c.xDims = {37, 3, 128};
        c.containerDims = {6, 3, 8, 128};
        c.scale = PcsScale::NONE;
        cases.push_back(c);
    }
    {
        // x [B, H, S, D] viewed out of a [B, S, H, D] projection buffer; the container
        // [nb, H, bs, D] laid out as [nb, bs, H, D] rows padded from 72 to 80.
        StoreCase c;
        c.name = "SDPA-native, int64 slots, strided container";
        c.xDims = {2, 4, 5, 72};
        c.xStrides = std::vector<int64_t>{int64_t{5} * 4 * 72, 72, int64_t{4} * 72, 1};
        c.slotStrides = std::vector<int64_t>{16, 2};
        c.containerDims = {3, 4, 8, 72};
        c.containerStrides = std::vector<int64_t>{int64_t{8} * 4 * 80, 80, int64_t{4} * 80, 1};
        c.int64Slots = true;
        c.scale = PcsScale::DEVICE;
        c.scaleValue = 0.75F;
        cases.push_back(c);
    }
    {
        StoreCase c;
        c.name = "scalar path: odd D, strided innermost x";
        c.xDims = {11, 2, 37};
        c.xStrides = std::vector<int64_t>{int64_t{2} * 2 * 40, int64_t{2} * 40, 2};
        c.containerDims = {2, 2, 7, 37};
        c.containerStrides = std::vector<int64_t>{int64_t{2} * 7 * 41, int64_t{7} * 41, 41, 1};
        c.scale = PcsScale::RUNTIME;
        c.scaleValue = 1.5F;
        cases.push_back(c);
    }
    if(fp8)
    {
        for(const float scaleValue : {2.0e4F, 4.0e-4F})
        {
            StoreCase c;
            c.name = "fp8 range, scale " + std::to_string(scaleValue);
            c.xDims = {1, 4, 3, 64};
            c.containerDims = {1, 4, 4, 64};
            c.int64Slots = true;
            c.scale = PcsScale::RUNTIME;
            c.scaleValue = scaleValue;
            cases.push_back(c);
        }
    }
    return cases;
}

template <typename XT, typename CT>
void runCases(const KernelDefinition& kernel, bool fp8)
{
    for(const auto& storeCase : storeCases(fp8))
    {
        if(storeCase.int64Slots)
        {
            expectStoreMatches<XT, CT, int64_t>(kernel, storeCase);
        }
        else
        {
            expectStoreMatches<XT, CT, int32_t>(kernel, storeCase);
        }
    }
}

template <typename XT>
void runForCache(const KernelDefinition& kernel, const std::string& cacheDtype)
{
    if(cacheDtype == "fp8_e4m3")
    {
        runCases<XT, types::fp8_e4m3>(kernel, true);
    }
    else if(cacheDtype == "fp8_e5m2")
    {
        runCases<XT, types::fp8_e5m2>(kernel, true);
    }
    else if(cacheDtype == "fp8_e4m3_fnuz")
    {
        runCases<XT, types::fp8_e4m3_fnuz>(kernel, true);
    }
    else if(cacheDtype == "fp8_e5m2_fnuz")
    {
        runCases<XT, types::fp8_e5m2_fnuz>(kernel, true);
    }
    else
    {
        runCases<XT, XT>(kernel, false);
    }
}

TEST(TestGpuFlydslPagedCacheStoreObjects, EveryObjectMatchesTheReference)
{
    SKIP_IF_NO_DEVICES();
    auto kernels = stagedKernels(FLYDSL_PAGED_CACHE_STORE.engineName);
    if(kernels.empty())
    {
        reportUnstaged("paged_cache_store",
                       "this build staged no paged-cache-store objects for this device");
        return;
    }
    std::sort(kernels.begin(), kernels.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.name < rhs.name;
    });
    for(const auto& kernel : kernels)
    {
        const auto dtype = stringField(kernel, FLYDSL_DTYPE_FIELD);
        const auto cacheDtype = stringField(kernel, FLYDSL_CACHE_DTYPE_FIELD);
        if(cacheDtype != dtype && cacheDtype.rfind("fp8_", 0) != 0)
        {
            ADD_FAILURE() << kernel.name << " declares cache_dtype '" << cacheDtype << "'";
            continue;
        }
        if(dtype == "bf16")
        {
            runForCache<types::bfloat16>(kernel, cacheDtype);
        }
        else if(dtype == "f16")
        {
            runForCache<types::half>(kernel, cacheDtype);
        }
        else if(dtype == "f32")
        {
            runForCache<float>(kernel, cacheDtype);
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
