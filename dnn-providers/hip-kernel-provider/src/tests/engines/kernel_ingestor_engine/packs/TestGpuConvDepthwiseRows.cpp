// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceConvolution.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <nlohmann/json.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestGpuConvDepthwiseRows.cpp
 * @brief The channel-row depthwise kernels (CONV_DEPTHWISE_ROWS_FWD, CONV_DEPTHWISE_ROWS_BWD)
 *        on the device, through the engines' own graph match, kernel match, prepare and launch,
 *        against hipDNN's CPU reference convolution.
 *
 * Every config of the shipped catalog that the kernel match admits runs on every case, in fp32,
 * fp16 and bf16. The cases cover every compile-time filter and the runtime-filter configs:
 * dilations on and off the lanes' dilation grid, strided forward, odd extents, planes taller
 * and wider than a tile or a band, unit-stride backward data (the forward form with the filter
 * reflected), strided 1-D backward data in both layouts at every catalog stride, rows of a
 * height-1 filter, channel counts below and above a wave, and true convolution. A host-only test
 * checks that every config serves some case. The CPU reference is cross-correlation only, so
 * true convolution reflects the filter around it.
 */
namespace
{

using namespace hip_kernel_provider;
using namespace hip_kernel_provider::kernel_ingestor_engine;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing::conv_naive_test;
using conv_naive::ConvDirection;
using hipdnn_data_sdk::utilities::Tensor;
using hipdnn_flatbuffers_sdk::data_objects::ConvMode;
using hipdnn_flatbuffers_sdk::data_objects::DataType;
using hipdnn_test_sdk::utilities::CpuFpReferenceConvolution;

using Half = hipdnn_data_sdk::types::half;
using BFloat16 = hipdnn_data_sdk::types::bfloat16;

PackSymbols packFor(ConvDirection direction)
{
    return {"",
            direction == ConvDirection::FORWARD ? "hipkernel.conv_depthwise_rows_fwd.graph_match"
                                                : "hipkernel.conv_depthwise_rows_bwd.graph_match",
            "",
            "hipkernel.conv_depthwise_rows.kernel_match",
            "",
            "hipkernel.conv_depthwise_rows.dispatch",
            "",
            "",
            ""};
}

/// The shipped catalog of one direction, as kernel definitions: read from the production
/// descriptor tree beside the provider (the unit tests' own descriptor root holds only test
/// sets), so these cases run exactly the configs that ship.
std::vector<hipdnn_plugin_sdk::ingestor::KernelDefinition> catalog(ConvDirection direction)
{
    static const auto s_catalogs = [] {
        // The provider's resolved tree, else the copy beside this binary's tree (the tests link
        // the provider in, so its module-relative lookup can fall through to the install prefix).
        std::vector<std::filesystem::path> roots;
        {
            const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter unset(
                "HIPDNN_DESCRIPTOR_DIR", "");
            roots.push_back(descriptorSearchDirectory());
        }
        std::error_code exeError;
        for(auto dir = std::filesystem::read_symlink("/proc/self/exe", exeError).parent_path();
            !exeError && !dir.empty() && dir != dir.root_path();
            dir = dir.parent_path())
        {
            roots.push_back(dir / "lib" / "hipdnn_plugins" / "engines"
                            / HIPKERNELPROVIDER_DESCRIPTOR_SUBDIR);
        }
        std::array<std::vector<hipdnn_plugin_sdk::ingestor::KernelDefinition>, 2> catalogs;
        for(size_t i = 0; i < catalogs.size(); ++i)
        {
            const std::string file
                = i == 0 ? "conv_depthwise_rows_fwd.kdp.json" : "conv_depthwise_rows_bwd.kdp.json";
            std::filesystem::path found;
            for(const auto& root : roots)
            {
                std::error_code error;
                for(auto it = std::filesystem::recursive_directory_iterator(root, error);
                    !error && found.empty()
                    && it != std::filesystem::recursive_directory_iterator();
                    it.increment(error))
                {
                    if(it->path().filename() == file)
                    {
                        found = it->path();
                    }
                }
                if(!found.empty())
                {
                    break;
                }
            }
            if(found.empty())
            {
                throw std::runtime_error("no " + file + " beside the provider or this test binary");
            }
            std::ifstream stream(found);
            const auto pack = nlohmann::json::parse(stream);
            for(const auto& descriptor : pack.at("kernelDescriptors"))
            {
                hipdnn_plugin_sdk::ingestor::KernelDefinition kernel;
                kernel.name = descriptor.at("name").get<std::string>();
                kernel.source.kind = hipdnn_plugin_sdk::ingestor::KernelSourceKind::EMBEDDED_SOURCE;
                kernel.source.sourceFile
                    = descriptor.at("kernel_source").at("source_file").get<std::string>();
                kernel.source.entryPoint
                    = descriptor.at("kernel_source").at("entry_point").get<std::string>();
                for(const auto& [key, value] : descriptor.at("metadata").items())
                {
                    if(value.is_string())
                    {
                        kernel.metadata[key] = value.get<std::string>();
                    }
                    else
                    {
                        kernel.metadata[key] = value.get<int64_t>();
                    }
                }
                kernel.priority = descriptor.at("priority").get<int64_t>();
                catalogs[i].push_back(std::move(kernel));
            }
        }
        return catalogs;
    }();
    return s_catalogs[direction == ConvDirection::FORWARD ? 0 : 1];
}

void forEachIndex(const std::vector<int64_t>& dims,
                  const std::function<void(const std::vector<int64_t>&)>& visit)
{
    std::vector<int64_t> index(dims.size(), 0);
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
    }
}

int64_t offsetOf(const std::vector<int64_t>& index, const std::vector<int64_t>& strides)
{
    int64_t offset = 0;
    for(size_t axis = 0; axis < index.size(); ++axis)
    {
        offset += index[axis] * strides[axis];
    }
    return offset;
}

template <typename T>
void reflectFilter(Tensor<T>& filter, Tensor<T>& reflected)
{
    const auto& dims = filter.dims();
    const auto* source = static_cast<const T*>(filter.rawHostData());
    auto* target = static_cast<T*>(reflected.rawHostData());
    forEachIndex(dims, [&](const std::vector<int64_t>& index) {
        auto mirror = index;
        for(size_t axis = 2; axis < dims.size(); ++axis)
        {
            mirror[axis] = dims[axis] - 1 - index[axis];
        }
        target[offsetOf(mirror, reflected.strides())] = source[offsetOf(index, filter.strides())];
    });
    reflected.markHostModified();
}

template <typename T>
float storageEpsilon()
{
    if constexpr(std::is_same_v<T, float>)
    {
        return 1.2e-7f;
    }
    else if constexpr(std::is_same_v<T, Half>)
    {
        return 9.8e-4f;
    }
    else
    {
        return 7.9e-3f;
    }
}

template <typename T>
DataType dataTypeOf()
{
    if constexpr(std::is_same_v<T, float>)
    {
        return DataType::FLOAT;
    }
    else if constexpr(std::is_same_v<T, Half>)
    {
        return DataType::HALF;
    }
    else
    {
        return DataType::BFLOAT16;
    }
}

struct DeviceCase
{
    std::string name;
    ConvShape shape;
    ConvDirection direction;
};

ConvShape depthwise(int64_t channels,
                    std::vector<int64_t> input,
                    std::vector<int64_t> filter,
                    std::vector<int64_t> pad,
                    std::vector<int64_t> dilation,
                    bool channelsLast = false)
{
    ConvShape shape;
    shape.n = 2;
    shape.c = shape.k = shape.groups = channels;
    shape.input = std::move(input);
    shape.filter = std::move(filter);
    shape.prePadding = pad;
    shape.postPadding = pad;
    shape.stride.assign(shape.input.size(), 1);
    shape.dilation = std::move(dilation);
    shape.channelsLast = channelsLast;
    return shape;
}

/// ConvTranspose1d of @p channels channels: filter @p k, stride @p s, padding @p pad.
ConvShape transposed(int64_t channels, int64_t length, int64_t k, int64_t s, int64_t pad, bool last)
{
    auto shape = depthwise(channels, {length}, {k}, {pad}, {1}, last);
    shape.stride = {s};
    return shape;
}

std::vector<DeviceCase> deviceCases()
{
    constexpr auto FWD = ConvDirection::FORWARD;
    constexpr auto BWD = ConvDirection::BACKWARD_DATA;
    std::vector<DeviceCase> cases;
    // 1-D forward, channels-first: every compile-time filter, dilation on the 256-lane grid (8,
    // 128) and off it (3), a tile edge past 1024 outputs, stride 2.
    cases.push_back({"Nchw1dK3D8", depthwise(3, {1301}, {3}, {8}, {8}), FWD});
    cases.push_back({"Nchw1dK3D128", depthwise(2, {1301}, {3}, {128}, {128}), FWD});
    cases.push_back({"Nchw1dK3D3", depthwise(3, {601}, {3}, {3}, {3}), FWD});
    cases.push_back({"Nchw1dK7", depthwise(4, {601}, {7}, {3}, {1}), FWD});
    cases.push_back({"Nchw1dK9", depthwise(4, {601}, {9}, {4}, {1}), FWD});
    cases.push_back({"Nchw1dK31", depthwise(3, {777}, {31}, {15}, {1}), FWD});
    cases.push_back({"Nchw1dK15", depthwise(3, {500}, {15}, {7}, {1}), FWD});
    auto strided1d = depthwise(3, {777}, {31}, {15}, {1});
    strided1d.stride = {2};
    cases.push_back({"Nchw1dK31Stride2", strided1d, FWD});
    // 2-D forward, channels-first: dilation 2 and 4 (on the grid of every tile and band), 3 (off
    // it), anisotropic, 5x5 dilation 2, 9x9, stride 2, a plane taller than the 72-row band and
    // wider than a tile.
    cases.push_back({"Nchw3x3D2", depthwise(3, {17, 19}, {3, 3}, {2, 2}, {2, 2}), FWD});
    cases.push_back({"Nchw3x3D4", depthwise(2, {33, 35}, {3, 3}, {4, 4}, {4, 4}), FWD});
    cases.push_back({"Nchw3x3D3", depthwise(2, {33, 35}, {3, 3}, {3, 3}, {3, 3}), FWD});
    cases.push_back({"Nchw3x3D1x2", depthwise(2, {21, 23}, {3, 3}, {1, 2}, {1, 2}), FWD});
    cases.push_back({"Nchw3x3Tall", depthwise(2, {77, 70}, {3, 3}, {1, 1}, {1, 1}), FWD});
    cases.push_back({"Nchw5x5D2", depthwise(3, {29, 31}, {5, 5}, {4, 4}, {2, 2}), FWD});
    cases.push_back({"Nchw9x9", depthwise(2, {23, 21}, {9, 9}, {4, 4}, {1, 1}), FWD});
    cases.push_back({"Nchw5x4Asymmetric",
                     [] {
                         auto shape = depthwise(3, {12, 13}, {5, 4}, {2, 1}, {1, 1});
                         shape.postPadding = {0, 3};
                         return shape;
                     }(),
                     FWD});
    auto strided2d = depthwise(3, {33, 35}, {3, 3}, {1, 1}, {1, 1});
    strided2d.stride = {2, 2};
    cases.push_back({"Nchw3x3Stride2", strided2d, FWD});
    auto convolution = depthwise(3, {17, 19}, {3, 3}, {2, 2}, {2, 2});
    convolution.mode = ConvMode::CONVOLUTION;
    cases.push_back({"Nchw3x3D2ConvolutionMode", convolution, FWD});
    // Unit-stride backward data runs the forward form (filter reflected): every unit-stride
    // forward case again.
    const size_t forwardCases = cases.size();
    for(size_t i = 0; i < forwardCases; ++i)
    {
        const auto& stride = cases[i].shape.stride;
        if(std::all_of(stride.begin(), stride.end(), [](int64_t s) { return s == 1; }))
        {
            cases.push_back({cases[i].name, cases[i].shape, BWD});
        }
    }
    // Strided 1-D backward data (ConvTranspose1d), both layouts, every catalog stride and
    // compile-time filter.
    for(const bool last : {false, true})
    {
        const std::string layout = last ? "Nhwc" : "Nchw";
        cases.push_back({layout + "1dK12S2", transposed(3, 515, 12, 2, 5, last), BWD});
        cases.push_back({layout + "1dK12S2Pad0", transposed(3, 516, 12, 2, 0, last), BWD});
        cases.push_back({layout + "1dK29S2", transposed(2, 1001, 29, 2, 13, last), BWD});
        cases.push_back({layout + "1dK43S3", transposed(2, 1003, 43, 3, 20, last), BWD});
        cases.push_back({layout + "1dK7S3", transposed(5, 1003, 7, 3, 4, last), BWD});
        cases.push_back({layout + "1dK8S4", transposed(3, 801, 8, 4, 2, last), BWD});
        cases.push_back({layout + "1dK10S5", transposed(3, 801, 10, 5, 3, last), BWD});
        cases.push_back({layout + "1dK16S8", transposed(3, 1601, 16, 8, 4, last), BWD});
        cases.push_back({layout + "1dK12S2C70", transposed(70, 257, 12, 2, 5, last), BWD});
        auto rows = transposed(3, 515, 12, 2, 5, last);
        rows.input = {4, 515};
        rows.filter = {1, 12};
        rows.prePadding = rows.postPadding = {0, 5};
        rows.stride = {1, 2};
        rows.dilation = {1, 1};
        cases.push_back({layout + "RowsK12S2", rows, BWD});
        auto reflected = transposed(3, 515, 12, 2, 5, last);
        reflected.mode = ConvMode::CONVOLUTION;
        cases.push_back({layout + "1dK12S2ConvolutionMode", reflected, BWD});
        // dy one wide (hipconv #15149 review: a stride normalized for an output extent of 1):
        // x 13 from a K12 S2 filter, its last element reached by no tap.
        cases.push_back({layout + "1dK12S2OutputOne", transposed(3, 13, 12, 2, 0, last), BWD});
    }
    return cases;
}

template <typename T>
void runCase(ConvShape shape, ConvDirection direction)
{
    shape.dataType = dataTypeOf<T>();
    const auto spec = specFor(shape, direction);
    const GraphFixture fixture(buildGraph(spec), currentDeviceProperties());
    const auto pack = packFor(direction);
    const auto bound = matchesGraph(pack, fixture.context());
    ASSERT_TRUE(bound.has_value()) << "the engine refused a case it should serve";

    Tensor<T> x(spec.x.dims, spec.x.strides);
    Tensor<T> w(spec.w.dims, spec.w.strides);
    Tensor<T> y(spec.y.dims, spec.y.strides);
    constexpr unsigned int SEED = 4321;
    const bool forward = direction == ConvDirection::FORWARD;
    (forward ? x : y).fillTensorWithRandomValues(-1.0f, 1.0f, SEED);
    w.fillTensorWithRandomValues(-1.0f, 1.0f, SEED + 1);
    Tensor<T>& written = forward ? y : x;

    const bool reflect = shape.mode == ConvMode::CONVOLUTION;
    Tensor<T> reflectedW(spec.w.dims, spec.w.strides);
    if(reflect)
    {
        reflectFilter(w, reflectedW);
    }
    const Tensor<T>& filterForReference = reflect ? reflectedW : w;
    Tensor<T> reference(written.dims(), written.strides());
    if(forward)
    {
        CpuFpReferenceConvolution::fprop(x,
                                         filterForReference,
                                         reference,
                                         shape.stride,
                                         shape.dilation,
                                         shape.prePadding,
                                         shape.postPadding);
    }
    else
    {
        CpuFpReferenceConvolution::dgrad(reference,
                                         filterForReference,
                                         y,
                                         shape.stride,
                                         shape.dilation,
                                         shape.prePadding,
                                         shape.postPadding);
    }
    int64_t taps = 1;
    for(const auto extent : shape.filter)
    {
        taps *= extent;
    }
    const float accumulationBound = 1e-6f * static_cast<float>(taps);

    const auto& handler = dispatchHandler(pack);
    int ran = 0;
    for(const auto& kernel : catalog(direction))
    {
        if(!matchesKernel(pack, fixture.context(), kernel, *bound))
        {
            continue;
        }
        SCOPED_TRACE(kernel.name);
        ++ran;
        const auto prepared = handler.prepare(fixture.context(), *bound, kernel);
        ASSERT_NE(prepared, nullptr);

        written.fillWithSentinelValue();
        const std::array<hipdnnPluginDeviceBuffer_t, 3> buffers{
            hipdnnPluginDeviceBuffer_t{X_UID, x.rawDeviceData()},
            hipdnnPluginDeviceBuffer_t{W_UID, w.rawDeviceData()},
            hipdnnPluginDeviceBuffer_t{Y_UID, y.rawDeviceData()}};
        const Handle handle;
        handler.launch(
            handle, *prepared, buffers.data(), static_cast<uint32_t>(buffers.size()), nullptr);
        ASSERT_EQ(hipStreamSynchronize(handle.getStream()), hipSuccess);
        written.markDeviceModified();

        const auto* actual = static_cast<const T*>(written.rawHostData());
        const auto* expected = static_cast<const T*>(reference.rawHostData());
        int64_t mismatches = 0;
        forEachIndex(written.dims(), [&](const std::vector<int64_t>& index) {
            const float got = static_cast<float>(actual[offsetOf(index, written.strides())]);
            const float want = static_cast<float>(expected[offsetOf(index, reference.strides())]);
            const float tolerance
                = accumulationBound + 2.0f * storageEpsilon<T>() * (std::fabs(want) + 1e-3f);
            if(!(std::fabs(got - want) <= tolerance) && ++mismatches <= 5)
            {
                ADD_FAILURE() << "at linear index " << offsetOf(index, written.strides())
                              << ": got " << got << ", want " << want << ", tolerance "
                              << tolerance;
            }
        });
        EXPECT_EQ(mismatches, 0);
    }
    EXPECT_GE(ran, 1) << "no catalog config serves this case";
}

class TestGpuConvDepthwiseRows : public ::testing::TestWithParam<DeviceCase>
{
};

TEST_P(TestGpuConvDepthwiseRows, Float)
{
    SKIP_IF_NO_DEVICES();
    runCase<float>(GetParam().shape, GetParam().direction);
}

TEST_P(TestGpuConvDepthwiseRows, Half)
{
    SKIP_IF_NO_DEVICES();
    runCase<Half>(GetParam().shape, GetParam().direction);
}

TEST_P(TestGpuConvDepthwiseRows, BFloat16)
{
    SKIP_IF_NO_DEVICES();
    runCase<BFloat16>(GetParam().shape, GetParam().direction);
}

INSTANTIATE_TEST_SUITE_P(
    , TestGpuConvDepthwiseRows, ::testing::ValuesIn(deviceCases()), [](const auto& info) {
        return info.param.name + (info.param.direction == ConvDirection::FORWARD ? "Fwd" : "Bwd");
    });

// Host-only: the launch is sized from the problem alone, so a device that reports no units
// (hipconv #15149 review: zero CUs threw) is served exactly as a counted one, without throwing.
TEST(TestConvDepthwiseRowsCatalog, AnUncountedDeviceIsServedAsACountedOne)
{
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        auto shape = direction == ConvDirection::FORWARD ? depthwise(3, {601}, {7}, {3}, {1})
                                                         : transposed(3, 515, 12, 2, 5, false);
        shape.dataType = DataType::FLOAT;
        auto counted = testDeviceProperties();
        counted.gcnArchName = "gfx942:sramecc+:xnack-";
        counted.multiProcessorCount = 64;
        auto uncounted = counted;
        uncounted.multiProcessorCount = 0;
        const auto pack = packFor(direction);
        const GraphFixture withUnits(buildGraph(specFor(shape, direction)), counted);
        const GraphFixture withoutUnits(buildGraph(specFor(shape, direction)), uncounted);
        const auto bound = matchesGraph(pack, withUnits.context());
        ASSERT_TRUE(bound.has_value());
        int served = 0;
        for(auto kernel : catalog(direction))
        {
            kernel.metadata["dtype"] = std::string("FLOAT");
            const bool admitted = matchesKernel(pack, withUnits.context(), kernel, *bound);
            served += admitted ? 1 : 0;
            EXPECT_EQ(matchesKernel(pack, withoutUnits.context(), kernel, *bound), admitted)
                << kernel.name;
        }
        EXPECT_GT(served, 0);
    }
}

// Host-only: every shipped config is admitted by some device case, so the device suite runs
// every config.
TEST(TestConvDepthwiseRowsCatalog, EveryConfigServesADeviceCase)
{
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        const auto pack = packFor(direction);
        ASSERT_FALSE(catalog(direction).empty());
        for(const auto& kernel : catalog(direction))
        {
            bool served = false;
            for(const auto& deviceCase : deviceCases())
            {
                if(deviceCase.direction != direction)
                {
                    continue;
                }
                auto shape = deviceCase.shape;
                shape.dataType = DataType::FLOAT;
                const GraphFixture fixture(buildGraph(specFor(shape, direction)));
                const auto bound = matchesGraph(pack, fixture.context());
                // Every dtype runs the same cases: match the config's own.
                auto asFloat = kernel;
                asFloat.metadata["dtype"] = std::string("FLOAT");
                if(bound.has_value() && matchesKernel(pack, fixture.context(), asFloat, *bound))
                {
                    served = true;
                    break;
                }
            }
            EXPECT_TRUE(served) << kernel.name << " serves no device case";
        }
    }
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_CONV
