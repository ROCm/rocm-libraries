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
#include <hipdnn_test_sdk/utilities/CpuFpReferenceGather.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslGatherTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslNormDeviceTestSupport.hpp"

/**
 * @file TestFlydslGatherObjects.cpp
 * @brief Every FlyDSL gather object this device's shard carries, launched through its
 *        pack's dispatch handler and compared bit for bit with hipDNN's CPU reference
 *        (`CpuFpReferenceGather`).
 *
 * Each object runs:
 *  - an embedding lookup (x `[V, H]`, rank-2 token ids, axis 0) and a gather along a middle
 *    axis, at row widths with a partial last chunk, the indices including 0, V - 1 and
 *    repeats; the per-element objects also at row widths no vector object serves and along
 *    the last axis (one element per row);
 *  - indices outside `[0, V)`, which the reference refuses and the kernel writes as zero
 *    rows (the pack's documented behaviour);
 * into outputs that start as 0xFF bytes, so an element nothing wrote fails.
 *
 * `AddressesRowsPastFourGiB` gathers the last rows of a Qwen-sized table (248320 x 5120)
 * in f32 -- 4.7 GiB, past any 32-bit byte offset -- and in bf16, through every staged
 * object of those dtypes.
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

/// Matches, prepares and launches @p kernel on the graph @p spec describes; returns y's
/// bytes. Fails the case if the graph or the object declines.
std::vector<unsigned char> runObject(const KernelDefinition& kernel,
                                     const GatherGraphSpec& spec,
                                     const void* xDevice,
                                     const void* indicesDevice,
                                     size_t yBytes)
{
    const GraphFixture fixture(buildFlydslGatherGraph(spec), packedArchDeviceProperties());
    const auto bound = matchesGraph(FLYDSL_GATHER, fixture.context());
    EXPECT_TRUE(bound.has_value()) << "the graph this object is meant for is declined";
    if(!bound.has_value())
    {
        return {};
    }
    // The object's own metadata has to admit the graph built for it; one that does not would
    // never be selected, which a launch by name would otherwise hide.
    EXPECT_TRUE(matchesKernel(FLYDSL_GATHER, fixture.context(), kernel, *bound));

    const auto& handler = dispatchHandler(FLYDSL_GATHER);
    const auto prepared = handler.prepare(fixture.context(), *bound, kernel);
    EXPECT_NE(prepared, nullptr);
    if(prepared == nullptr)
    {
        return {};
    }

    const DeviceBuffer yDevice(yBytes);
    yDevice.fill(0xFF);
    const std::vector<hipdnnPluginDeviceBuffer_t> arguments{
        {FLYDSL_GATHER_X_UID, const_cast<void*>(xDevice)},
        {FLYDSL_GATHER_INDICES_UID, const_cast<void*>(indicesDevice)},
        {FLYDSL_GATHER_Y_UID, yDevice.get()}};
    const Handle handle;
    handler.launch(
        handle, *prepared, arguments.data(), static_cast<uint32_t>(arguments.size()), nullptr);
    EXPECT_EQ(hipSuccess, hipDeviceSynchronize());
    return yDevice.read(yBytes);
}

/// One case: x of @p xDims gathered along @p axis by @p indexDims indices, compared with the
/// CPU reference. Every index in range; 0 and the axis's last row always among them.
template <typename T, typename I>
void expectGatherMatchesReference(const KernelDefinition& kernel,
                                  const std::vector<int64_t>& xDims,
                                  const std::vector<int64_t>& indexDims,
                                  int64_t axis)
{
    std::string shape;
    for(const auto extent : xDims)
    {
        shape += std::to_string(extent) + ",";
    }
    SCOPED_TRACE(kernel.name + " x [" + shape + "] axis " + std::to_string(axis));

    Tensor<T> x(xDims, hipdnn_data_sdk::utilities::generateStrides(xDims));
    Tensor<I> indices(indexDims, hipdnn_data_sdk::utilities::generateStrides(indexDims));
    const auto yDims = flydslGatherOutputDims(xDims, indexDims, axis);
    Tensor<T> expected(yDims, hipdnn_data_sdk::utilities::generateStrides(yDims));

    std::mt19937 generator(static_cast<uint32_t>(xDims.back() * 977 + axis));
    std::uniform_real_distribution<float> uniform(-8.0F, 8.0F);
    auto* xData = x.memory().hostData();
    for(size_t i = 0; i < x.elementCount(); ++i)
    {
        xData[i] = T(uniform(generator));
    }
    const auto extent = xDims[static_cast<size_t>(axis)];
    std::uniform_int_distribution<int64_t> pick(0, extent - 1);
    auto* indexData = indices.memory().hostData();
    for(size_t i = 0; i < indices.elementCount(); ++i)
    {
        I value = 0;
        if(i == 0)
        {
            value = static_cast<I>(extent - 1);
        }
        else if(i != 1)
        {
            value = static_cast<I>(pick(generator));
        }
        indexData[i] = value;
    }
    hipdnn_test_sdk::utilities::CpuFpReferenceGather::gather(x, indices, expected, axis);

    GatherGraphSpec spec;
    spec.dataType = graphDataType(stringField(kernel, FLYDSL_GATHER_DTYPE_FIELD));
    spec.indexDataType = sizeof(I) == 8 ? DataType::INT64 : DataType::INT32;
    spec.xDims = xDims;
    spec.indexDims = indexDims;
    spec.axis = axis;

    const DeviceBuffer xDevice(xData, x.elementCount() * sizeof(T));
    const DeviceBuffer indicesDevice(indexData, indices.elementCount() * sizeof(I));
    const auto yBytes = expected.elementCount() * sizeof(T);
    const auto y = runObject(kernel, spec, xDevice.get(), indicesDevice.get(), yBytes);
    ASSERT_EQ(y.size(), yBytes);

    // A copy: bit for bit.
    const auto* want = reinterpret_cast<const unsigned char*>(expected.memory().hostData());
    const auto mismatch = std::mismatch(y.begin(), y.end(), want);
    EXPECT_EQ(mismatch.first, y.end())
        << "first differing byte at " << (mismatch.first - y.begin()) << " of " << yBytes;
}

/// Indices outside [0, V): the reference throws, and the kernel writes their rows as zeros
/// without reading outside x. The in-range rows are still gathered.
template <typename T, typename I>
void expectOutOfRangeRowsAreZero(const KernelDefinition& kernel, int64_t width)
{
    SCOPED_TRACE(kernel.name + " out-of-range indices, width " + std::to_string(width));
    constexpr int64_t ROWS = 9;
    std::vector<T> x(static_cast<size_t>(ROWS * width));
    for(size_t i = 0; i < x.size(); ++i)
    {
        x[i] = T(static_cast<float>(i % 251) - 125.0F);
    }
    // -1 and V, and 2^32 + 1 for INT64 (a truncation to 32 bits would read row 1).
    std::vector<I> indices{
        static_cast<I>(3), static_cast<I>(-1), static_cast<I>(ROWS - 1), static_cast<I>(ROWS)};
    if constexpr(sizeof(I) == 8)
    {
        indices.push_back(static_cast<I>((int64_t{1} << 32) + 1));
    }
    else
    {
        indices.push_back(static_cast<I>(std::numeric_limits<int32_t>::min()));
    }

    GatherGraphSpec spec;
    spec.dataType = graphDataType(stringField(kernel, FLYDSL_GATHER_DTYPE_FIELD));
    spec.indexDataType = sizeof(I) == 8 ? DataType::INT64 : DataType::INT32;
    spec.xDims = {ROWS, width};
    spec.indexDims = {static_cast<int64_t>(indices.size())};

    const DeviceBuffer xDevice(x.data(), x.size() * sizeof(T));
    const DeviceBuffer indicesDevice(indices.data(), indices.size() * sizeof(I));
    const auto rowBytes = static_cast<size_t>(width) * sizeof(T);
    const auto y
        = runObject(kernel, spec, xDevice.get(), indicesDevice.get(), indices.size() * rowBytes);
    ASSERT_EQ(y.size(), indices.size() * rowBytes);

    for(size_t slot = 0; slot < indices.size(); ++slot)
    {
        const auto index = static_cast<int64_t>(indices[slot]);
        std::vector<unsigned char> want(rowBytes, 0);
        if(index >= 0 && index < ROWS)
        {
            std::memcpy(want.data(), &x[static_cast<size_t>(index * width)], rowBytes);
        }
        EXPECT_EQ(0, std::memcmp(&y[slot * rowBytes], want.data(), rowBytes))
            << "row " << slot << " (index " << index << ")";
    }
}

/// The row widths an object runs at: whole vectors with a partial last chunk for a vector
/// object; for a per-element one also odd widths and one element.
std::vector<int64_t> widthsFor(const KernelDefinition& kernel, size_t elementBytes)
{
    const auto vecBytes = intField(kernel, FLYDSL_GATHER_VEC_BYTES_FIELD);
    const auto lane = static_cast<int64_t>(16 / elementBytes);
    std::vector<int64_t> widths{5120, lane, 2048 / static_cast<int64_t>(elementBytes) + lane};
    if(vecBytes != 16)
    {
        widths.insert(widths.end(), {1001, 7, 1});
    }
    return widths;
}

template <typename T, typename I>
void expectObjectMatches(const KernelDefinition& kernel)
{
    for(const auto width : widthsFor(kernel, sizeof(T)))
    {
        // The embedding lookup: a [37, H] table and rank-2 token ids.
        expectGatherMatchesReference<T, I>(kernel, {37, width}, {1, 5}, 0);
        // A middle axis: outer 3, rank-2 indices.
        expectGatherMatchesReference<T, I>(kernel, {3, 11, width}, {2, 3}, 1);
    }
    if(intField(kernel, FLYDSL_GATHER_VEC_BYTES_FIELD) != 16)
    {
        // The last axis: one element per gathered row, outer 20.
        expectGatherMatchesReference<T, I>(kernel, {4, 5, 13}, {6}, 2);
    }
    expectOutOfRangeRowsAreZero<T, I>(kernel, widthsFor(kernel, sizeof(T)).front());
}

template <typename T>
void expectObjectMatchesAnyIndex(const KernelDefinition& kernel)
{
    if(stringField(kernel, FLYDSL_GATHER_INDEX_DTYPE_FIELD) == "i64")
    {
        expectObjectMatches<T, int64_t>(kernel);
    }
    else
    {
        expectObjectMatches<T, int32_t>(kernel);
    }
}

std::vector<KernelDefinition> stagedGatherKernels()
{
    auto kernels = stagedKernels(FLYDSL_GATHER.engineName);
    std::sort(kernels.begin(), kernels.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.name < rhs.name;
    });
    return kernels;
}

TEST(TestGpuFlydslGatherObjects, EveryObjectMatchesTheCpuReference)
{
    SKIP_IF_NO_DEVICES();
    const auto kernels = stagedGatherKernels();
    if(kernels.empty())
    {
        reportUnstaged("gather", "this build staged no gather objects for this device");
        return;
    }
    for(const auto& kernel : kernels)
    {
        const auto dtype = stringField(kernel, FLYDSL_GATHER_DTYPE_FIELD);
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

/// Gathers the rows at the far end of a Qwen-sized table through @p kernel: rows past
/// 2 GiB and (in f32) past 4 GiB of the table's base, where a 32-bit byte offset wraps.
template <typename T, typename I>
void expectFarRowsGathered(const KernelDefinition& kernel,
                           const DeviceBuffer& table,
                           int64_t vocabulary,
                           int64_t width)
{
    SCOPED_TRACE(kernel.name + " far rows");
    const auto rowBytes = static_cast<size_t>(width) * sizeof(T);
    // The last row, the first rows past 2^31 and 2^32 bytes, and row 0.
    const auto pastBytes = [&](int64_t bytes) {
        return std::min<int64_t>(vocabulary - 1, bytes / static_cast<int64_t>(rowBytes) + 1);
    };
    const std::vector<I> indices{static_cast<I>(vocabulary - 1),
                                 static_cast<I>(pastBytes(int64_t{1} << 31)),
                                 static_cast<I>(pastBytes(int64_t{1} << 32)),
                                 0,
                                 static_cast<I>(vocabulary - 2)};

    // Each gathered row gets its own contents; the rest of the table stays as filled.
    std::vector<std::vector<T>> rows;
    for(size_t slot = 0; slot < indices.size(); ++slot)
    {
        std::vector<T> row(static_cast<size_t>(width));
        for(size_t i = 0; i < row.size(); ++i)
        {
            const auto value
                = (static_cast<int64_t>(i) * 7 + static_cast<int64_t>(slot) * 13) % 199;
            row[i] = T(static_cast<float>(value) - 99.0F);
        }
        auto* destination = static_cast<unsigned char*>(table.get())
                            + static_cast<size_t>(indices[slot]) * rowBytes;
        ASSERT_EQ(hipSuccess, hipMemcpy(destination, row.data(), rowBytes, hipMemcpyHostToDevice));
        rows.push_back(std::move(row));
    }

    GatherGraphSpec spec;
    spec.dataType = graphDataType(stringField(kernel, FLYDSL_GATHER_DTYPE_FIELD));
    spec.indexDataType = sizeof(I) == 8 ? DataType::INT64 : DataType::INT32;
    spec.xDims = {vocabulary, width};
    spec.indexDims = {1, static_cast<int64_t>(indices.size())};

    const DeviceBuffer indicesDevice(indices.data(), indices.size() * sizeof(I));
    const auto y
        = runObject(kernel, spec, table.get(), indicesDevice.get(), indices.size() * rowBytes);
    ASSERT_EQ(y.size(), indices.size() * rowBytes);
    // A repeated index (pastBytes clamps on a small table) holds the last row written.
    for(size_t slot = 0; slot < indices.size(); ++slot)
    {
        size_t writer = slot;
        for(size_t later = slot + 1; later < indices.size(); ++later)
        {
            if(indices[later] == indices[slot])
            {
                writer = later;
            }
        }
        EXPECT_EQ(0, std::memcmp(&y[slot * rowBytes], rows[writer].data(), rowBytes))
            << "row " << slot << " (index " << static_cast<int64_t>(indices[slot]) << ")";
    }
}

template <typename T>
void expectFarRowsGatheredByEveryObject(const std::vector<KernelDefinition>& kernels,
                                        const std::string& dtype)
{
    constexpr int64_t VOCABULARY = 248320;
    constexpr int64_t WIDTH = 5120;
    const auto bytes = static_cast<size_t>(VOCABULARY * WIDTH) * sizeof(T);
    size_t freeBytes = 0;
    size_t totalBytes = 0;
    ASSERT_EQ(hipSuccess, hipMemGetInfo(&freeBytes, &totalBytes));
    if(freeBytes < bytes + (size_t{1} << 30))
    {
        GTEST_SKIP() << "a " << bytes << "-byte table does not fit this device's free memory";
    }
    const DeviceBuffer table(bytes);
    table.fill(0x5A);
    int launched = 0;
    for(const auto& kernel : kernels)
    {
        if(stringField(kernel, FLYDSL_GATHER_DTYPE_FIELD) != dtype)
        {
            continue;
        }
        if(stringField(kernel, FLYDSL_GATHER_INDEX_DTYPE_FIELD) == "i64")
        {
            expectFarRowsGathered<T, int64_t>(kernel, table, VOCABULARY, WIDTH);
        }
        else
        {
            expectFarRowsGathered<T, int32_t>(kernel, table, VOCABULARY, WIDTH);
        }
        ++launched;
    }
    // Two index dtypes x two access widths.
    EXPECT_EQ(launched, 4) << dtype << " objects run on the far rows";
}

TEST(TestGpuFlydslGatherObjects, AddressesRowsPastFourGiB)
{
    SKIP_IF_NO_DEVICES();
    const auto kernels = stagedGatherKernels();
    if(kernels.empty())
    {
        reportUnstaged("gather", "this build staged no gather objects for this device");
        return;
    }
    expectFarRowsGatheredByEveryObject<float>(kernels, "f32");
    expectFarRowsGatheredByEveryObject<Bf16>(kernels, "bf16");
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
