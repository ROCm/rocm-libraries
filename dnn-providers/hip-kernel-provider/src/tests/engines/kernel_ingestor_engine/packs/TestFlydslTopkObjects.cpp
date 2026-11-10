// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose; see TestFlydslRmsNormEngine.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hip/hip_runtime_api.h>

#include <hipdnn_data_sdk/types/Bfloat16.hpp>
#include <hipdnn_data_sdk/types/Half.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceTopk.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslNormDeviceTestSupport.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslTopkTestGraphs.hpp"

/**
 * @file TestFlydslTopkObjects.cpp
 * @brief Every FlyDSL top-k object this device's shard carries, launched through its pack's
 *        dispatch handler and compared exactly with hipDNN's CPU reference
 *        (`CpuFpReferenceTopk`): indices equal, values equal bit for bit.
 *
 * Each object runs every k it serves among 1, 2, 8, 10, 64, 257 and 1024 (as the row
 * allows), largest and smallest, with and without values, over:
 *  - three kinds of rows: normal noise; four distinct values, so nearly every selection
 *    breaks ties by position; and rows of NaNs of both signs, infinities, signed zeros and
 *    small integers, where NaN must rank above every number and -0.0 tie with +0.0;
 *  - row widths 1, 7, 512 (the MoE router), 3001 (odd, so rows are not 16-byte aligned)
 *    and 5000 (past the radix select's LDS key cache), and the greedy-decoding vocabulary,
 *    248320, at k 1 and 8;
 *  - x at a 16-byte aligned base (128-bit row loads) and one element past it (element by
 *    element);
 * into outputs that start as 0xFF bytes, so a slot nothing wrote fails.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using data_objects::DataType;
using Bf16 = hipdnn_data_sdk::types::bfloat16;
using F16 = hipdnn_data_sdk::types::half;
using hipdnn_data_sdk::utilities::Tensor;

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

DataType graphDataType(const std::string& dtype)
{
    if(dtype == "bf16")
    {
        return DataType::BFLOAT16;
    }
    return dtype == "f16" ? DataType::HALF : DataType::FLOAT;
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

    ~DeviceBuffer()
    {
        static_cast<void>(hipFree(_pointer));
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&&) = delete;
    DeviceBuffer& operator=(DeviceBuffer&&) = delete;

    unsigned char* get() const
    {
        return static_cast<unsigned char*>(_pointer);
    }

    void fill(int byte) const
    {
        EXPECT_EQ(hipSuccess, hipMemset(_pointer, byte, _bytes));
    }

    std::vector<unsigned char> read(size_t bytes) const
    {
        std::vector<unsigned char> host(bytes);
        EXPECT_EQ(hipSuccess, hipMemcpy(host.data(), _pointer, bytes, hipMemcpyDeviceToHost));
        return host;
    }

private:
    size_t _bytes;
    void* _pointer = nullptr;
};

/// How a case's row elements are drawn.
enum class RowKind
{
    NOISE,
    TIES,
    SPECIALS,
};

const char* rowKindName(RowKind kind)
{
    switch(kind)
    {
    case RowKind::NOISE:
        return "noise";
    case RowKind::TIES:
        return "ties";
    default:
        return "specials";
    }
}

float drawValue(RowKind kind, std::mt19937& generator)
{
    switch(kind)
    {
    case RowKind::NOISE:
        return std::normal_distribution<float>(0.0F, 3.0F)(generator);
    case RowKind::TIES:
        return static_cast<float>(generator() % 4) - 1.5F;
    default:
        break;
    }
    constexpr float NAN_VALUE = std::numeric_limits<float>::quiet_NaN();
    constexpr float INF_VALUE = std::numeric_limits<float>::infinity();
    switch(generator() % 12)
    {
    case 0:
        return NAN_VALUE;
    case 1:
        return -NAN_VALUE;
    case 2:
        return INF_VALUE;
    case 3:
        return -INF_VALUE;
    case 4:
        return 0.0F;
    case 5:
        return -0.0F;
    default:
        return static_cast<float>(generator() % 3) - 1.0F;
    }
}

/// One launch's shape and options.
struct TopkCase
{
    int64_t rows = 1;
    int64_t columns = 1;
    int64_t k = 1;
    bool largest = true;
    bool withValues = true;
    RowKind kind = RowKind::NOISE;
    /// x starts one element past a 16-byte aligned base, so the kernel reads it element by
    /// element.
    bool offsetBase = false;
};

/// Runs @p kernel on @p topkCase and compares with the CPU reference.
template <typename T, typename I>
void expectCaseMatchesReference(const KernelDefinition& kernel, const TopkCase& topkCase)
{
    SCOPED_TRACE(kernel.name + " rows " + std::to_string(topkCase.rows) + " n "
                 + std::to_string(topkCase.columns) + " k " + std::to_string(topkCase.k)
                 + (topkCase.largest ? " largest" : " smallest")
                 + (topkCase.withValues ? " +values " : " ") + rowKindName(topkCase.kind)
                 + (topkCase.offsetBase ? " offset base" : ""));

    const std::vector<int64_t> xDims{topkCase.rows, topkCase.columns};
    const std::vector<int64_t> outDims{topkCase.rows, topkCase.k};
    Tensor<T> x(xDims, hipdnn_data_sdk::utilities::generateStrides(xDims));
    Tensor<I> expectedIndices(outDims, hipdnn_data_sdk::utilities::generateStrides(outDims));
    Tensor<T> expectedValues(outDims, hipdnn_data_sdk::utilities::generateStrides(outDims));

    std::mt19937 generator(static_cast<uint32_t>(topkCase.columns * 131 + topkCase.k * 7
                                                 + static_cast<int64_t>(topkCase.kind)));
    auto* xData = x.memory().hostData();
    for(size_t i = 0; i < x.elementCount(); ++i)
    {
        xData[i] = T(drawValue(topkCase.kind, generator));
    }
    hipdnn_test_sdk::utilities::CpuFpReferenceTopk::topk<T, I>(
        x, expectedIndices, &expectedValues, topkCase.k, topkCase.largest);

    TopkGraphSpec spec;
    spec.dataType = graphDataType(stringField(kernel, FLYDSL_TOPK_DTYPE_FIELD));
    spec.indexDataType = sizeof(I) == 8 ? DataType::INT64 : DataType::INT32;
    spec.xDims = xDims;
    spec.k = topkCase.k;
    spec.largest = topkCase.largest;
    spec.withValues = topkCase.withValues;
    if(topkCase.offsetBase)
    {
        spec.alignment = {{FLYDSL_TOPK_X_UID, static_cast<int64_t>(sizeof(T))}};
    }

    const GraphFixture fixture(buildFlydslTopkGraph(spec), packedArchDeviceProperties());
    const auto bound = matchesGraph(FLYDSL_TOPK, fixture.context());
    ASSERT_TRUE(bound.has_value()) << "the graph this object is meant for is declined";
    // The object's own metadata has to admit the graph built for it; one that does not would
    // never be selected, which a launch by name would otherwise hide.
    ASSERT_TRUE(matchesKernel(FLYDSL_TOPK, fixture.context(), kernel, *bound));

    const auto& handler = dispatchHandler(FLYDSL_TOPK);
    const auto prepared = handler.prepare(fixture.context(), *bound, kernel);
    ASSERT_NE(prepared, nullptr);

    const auto xBytes = x.elementCount() * sizeof(T);
    const size_t offset = topkCase.offsetBase ? sizeof(T) : 0;
    const DeviceBuffer xDevice(xBytes + offset);
    ASSERT_EQ(hipSuccess, hipMemcpy(xDevice.get() + offset, xData, xBytes, hipMemcpyHostToDevice));
    const auto indicesBytes = expectedIndices.elementCount() * sizeof(I);
    const auto valuesBytes = expectedValues.elementCount() * sizeof(T);
    const DeviceBuffer indicesDevice(indicesBytes);
    const DeviceBuffer valuesDevice(valuesBytes);
    indicesDevice.fill(0xFF);
    valuesDevice.fill(0xFF);

    std::vector<hipdnnPluginDeviceBuffer_t> arguments{
        {FLYDSL_TOPK_X_UID, xDevice.get() + offset},
        {FLYDSL_TOPK_INDICES_UID, indicesDevice.get()}};
    if(topkCase.withValues)
    {
        arguments.push_back({FLYDSL_TOPK_VALUES_UID, valuesDevice.get()});
    }
    const Handle handle;
    handler.launch(
        handle, *prepared, arguments.data(), static_cast<uint32_t>(arguments.size()), nullptr);
    ASSERT_EQ(hipSuccess, hipDeviceSynchronize());

    const auto indices = indicesDevice.read(indicesBytes);
    const auto* wantIndices
        = reinterpret_cast<const unsigned char*>(expectedIndices.memory().hostData());
    const auto indexMismatch = std::mismatch(indices.begin(), indices.end(), wantIndices);
    if(indexMismatch.first != indices.end())
    {
        const auto slot = static_cast<size_t>(indexMismatch.first - indices.begin()) / sizeof(I);
        I got{};
        std::memcpy(&got, &indices[slot * sizeof(I)], sizeof(I));
        ADD_FAILURE() << "indices differ first at row " << slot / static_cast<size_t>(topkCase.k)
                      << " slot " << slot % static_cast<size_t>(topkCase.k) << ": got "
                      << static_cast<int64_t>(got) << ", want "
                      << static_cast<int64_t>(expectedIndices.memory().hostData()[slot]);
    }

    const auto values = valuesDevice.read(valuesBytes);
    if(topkCase.withValues)
    {
        const auto* wantValues
            = reinterpret_cast<const unsigned char*>(expectedValues.memory().hostData());
        const auto valueMismatch = std::mismatch(values.begin(), values.end(), wantValues);
        EXPECT_EQ(valueMismatch.first, values.end())
            << "values differ first at byte " << (valueMismatch.first - values.begin());
    }
    else
    {
        // Without a values output nothing may be written through the values slots.
        EXPECT_TRUE(std::all_of(
            values.begin(), values.end(), [](unsigned char byte) { return byte == 0xFF; }));
    }
}

template <typename T, typename I>
void expectObjectMatches(const KernelDefinition& kernel)
{
    const auto kMax = intField(kernel, FLYDSL_TOPK_K_MAX_FIELD);
    ASSERT_GE(kMax, 1) << kernel.name;
    const std::vector<int64_t> ks{1, 2, 8, 10, 64, 257, 1024};

    int variant = 0;
    for(const int64_t columns :
        {int64_t{1}, int64_t{7}, int64_t{512}, int64_t{3001}, int64_t{5000}})
    {
        for(const auto kind : {RowKind::NOISE, RowKind::TIES, RowKind::SPECIALS})
        {
            for(const auto k : ks)
            {
                if(k > columns || k > kMax)
                {
                    continue;
                }
                for(const bool largest : {true, false})
                {
                    TopkCase topkCase;
                    topkCase.rows = columns >= 3001 ? 3 : 9;
                    topkCase.columns = columns;
                    topkCase.k = k;
                    topkCase.largest = largest;
                    topkCase.kind = kind;
                    topkCase.withValues = (variant % 2) == 0;
                    topkCase.offsetBase = (variant % 3) == 1;
                    ++variant;
                    expectCaseMatchesReference<T, I>(kernel, topkCase);
                }
            }
        }
    }

    // Greedy decoding over the vocabulary, and a small k over it.
    for(const int64_t k : {int64_t{1}, int64_t{8}})
    {
        if(k > kMax)
        {
            continue;
        }
        for(const auto kind : {RowKind::NOISE, RowKind::SPECIALS})
        {
            TopkCase topkCase;
            topkCase.rows = 2;
            topkCase.columns = 248320;
            topkCase.k = k;
            topkCase.kind = kind;
            topkCase.largest = kind == RowKind::NOISE;
            topkCase.withValues = k == 8;
            topkCase.offsetBase = kind == RowKind::SPECIALS;
            expectCaseMatchesReference<T, I>(kernel, topkCase);
        }
    }
}

template <typename T>
void expectObjectMatchesAnyIndex(const KernelDefinition& kernel)
{
    if(stringField(kernel, FLYDSL_TOPK_INDEX_DTYPE_FIELD) == "i64")
    {
        expectObjectMatches<T, int64_t>(kernel);
    }
    else
    {
        expectObjectMatches<T, int32_t>(kernel);
    }
}

TEST(TestGpuFlydslTopkObjects, EveryObjectMatchesTheCpuReference)
{
    SKIP_IF_NO_DEVICES();
    auto kernels = stagedKernels(FLYDSL_TOPK.engineName);
    if(kernels.empty())
    {
        reportUnstaged("topk", "this build staged no topk objects for this device");
        return;
    }
    std::sort(kernels.begin(), kernels.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.name < rhs.name;
    });
    for(const auto& kernel : kernels)
    {
        const auto dtype = stringField(kernel, FLYDSL_TOPK_DTYPE_FIELD);
        if(dtype == "bf16")
        {
            expectObjectMatchesAnyIndex<Bf16>(kernel);
        }
        else if(dtype == "f16")
        {
            expectObjectMatchesAnyIndex<F16>(kernel);
        }
        else if(dtype == "f32")
        {
            expectObjectMatchesAnyIndex<float>(kernel);
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
