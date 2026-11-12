// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
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
#include "engines/kernel_ingestor_engine/packs/ConvFftProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestGpuConvFft.cpp
 * @brief The FFT convolution kernels (CONV_FFT_FWD / _BWD / _WRW) on the device, through the
 *        engine's own graph match, kernel match, prepare and launch, against hipDNN's CPU
 *        reference convolution accumulated in double (fprop / dgrad / wgrad).
 *
 * Every config of the shipped catalog whose transform fits the case runs on every case in fp32
 * (the admission floor is switched off, so small cases reach every code path), and every
 * transform size and block (256-thread, 64-channel product configs) in fp16 and bf16:
 * MIOpen's FFT shapes (5x5, padding 2, 28 / 14 / 7), odd extents that are no multiple of the tile,
 * 11x11 and 31x31 filters, asymmetric filters with per-axis dilation and groups, 1-D rows (with
 * dilation and groups), depthwise, outputs smaller than one tile, both layouts, both conv modes,
 * and several chunks of tiles (a workspace cap forced low; backward weights adds the chunks'
 * sums in order).
 *
 * Tolerance, the FFT error model: each output is off by at most 8 log2(th tw) units of fp32
 * rounding of the largest reference magnitude (FFT error is relative to the whole transform,
 * not to the output), plus two units of the element type's rounding of the output itself.
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
using hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter;

using Half = hipdnn_data_sdk::types::half;
using BFloat16 = hipdnn_data_sdk::types::bfloat16;

PackSymbols packFor(ConvDirection direction)
{
    return {"",
            direction == ConvDirection::FORWARD         ? "hipkernel.conv_fft_fwd.graph_match"
            : direction == ConvDirection::BACKWARD_DATA ? "hipkernel.conv_fft_bwd.graph_match"
                                                        : "hipkernel.conv_fft_wrw.graph_match",
            "",
            "hipkernel.conv_fft.kernel_match",
            "",
            "hipkernel.conv_fft.dispatch",
            "",
            "",
            ""};
}

int directionIndex(ConvDirection direction)
{
    return direction == ConvDirection::FORWARD         ? 0
           : direction == ConvDirection::BACKWARD_DATA ? 1
                                                       : 2;
}

/// The shipped catalog of @p direction, read from the production descriptor tree beside the
/// provider (or beside this binary), so the cases run exactly the configs that ship.
std::vector<hipdnn_plugin_sdk::ingestor::KernelDefinition> catalog(ConvDirection direction)
{
    static const auto s_catalogs = [] {
        std::vector<std::filesystem::path> roots;
        {
            const ScopedEnvironmentVariableSetter unset("HIPDNN_DESCRIPTOR_DIR", "");
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
        std::array<std::vector<hipdnn_plugin_sdk::ingestor::KernelDefinition>, 3> catalogs;
        const std::array<std::string, 3> files{
            "conv_fft_fwd.kdp.json", "conv_fft_bwd.kdp.json", "conv_fft_wrw.kdp.json"};
        for(size_t i = 0; i < catalogs.size(); ++i)
        {
            std::filesystem::path found;
            for(const auto& root : roots)
            {
                std::error_code error;
                for(auto it = std::filesystem::recursive_directory_iterator(root, error);
                    !error && found.empty()
                    && it != std::filesystem::recursive_directory_iterator();
                    it.increment(error))
                {
                    if(it->path().filename() == files[i])
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
                throw std::runtime_error("no " + files[i]
                                         + " beside the provider or this test binary");
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
    return s_catalogs[static_cast<size_t>(directionIndex(direction))];
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

/// Copies @p filter into @p reflected with every spatial axis reversed.
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
    ConvDirection direction = ConvDirection::FORWARD;
    /// HIP_KERNEL_PROVIDER_CONV_FFT_WORKSPACE_CAP for the case (0: the default cap).
    long long workspaceCap = 0;
};

ConvShape conv(int64_t n,
               int64_t c,
               int64_t k,
               int64_t groups,
               std::vector<int64_t> input,
               std::vector<int64_t> filter,
               std::vector<int64_t> pad,
               std::vector<int64_t> dilation,
               bool channelsLast)
{
    ConvShape shape;
    shape.n = n;
    shape.c = c;
    shape.k = k;
    shape.groups = groups;
    shape.input = std::move(input);
    shape.filter = std::move(filter);
    shape.prePadding = pad;
    shape.postPadding = pad;
    shape.stride.assign(shape.input.size(), 1);
    shape.dilation = std::move(dilation);
    shape.channelsLast = channelsLast;
    return shape;
}

std::vector<DeviceCase> deviceCases()
{
    std::vector<DeviceCase> cases;
    const auto add = [&cases](const std::string& name, const ConvShape& shape, long long cap = 0) {
        cases.push_back({"Fwd" + name, shape, ConvDirection::FORWARD, cap});
        cases.push_back({"Bwd" + name, shape, ConvDirection::BACKWARD_DATA, cap});
        cases.push_back({"Wrw" + name, shape, ConvDirection::BACKWARD_WEIGHTS, cap});
    };
    // MIOpen's FFT special cases (GoogLeNet 5x5 branches), as references.
    add("Miopen5x5Img28", conv(2, 16, 32, 1, {28, 28}, {5, 5}, {2, 2}, {1, 1}, false));
    add("Miopen5x5Img14", conv(2, 24, 64, 1, {14, 14}, {5, 5}, {2, 2}, {1, 1}, false));
    add("Miopen5x5Img7", conv(4, 32, 48, 1, {7, 7}, {5, 5}, {2, 2}, {1, 1}, false));
    // Odd extents, no tile multiples, both layouts.
    add("Odd11x11Nchw", conv(2, 8, 12, 1, {37, 41}, {11, 11}, {5, 5}, {1, 1}, false));
    add("Odd11x11Nhwc", conv(2, 8, 12, 1, {37, 41}, {11, 11}, {5, 5}, {1, 1}, true));
    // Asymmetric filter, per-axis dilation, groups, asymmetric padding.
    auto asym = conv(2, 12, 8, 2, {29, 31}, {7, 3}, {3, 2}, {1, 2}, true);
    asym.postPadding = {2, 3};
    add("Asym7x3DilGroupsNhwc", asym);
    // A 31x31 filter (64x64 transforms only).
    add("Large31x31", conv(1, 4, 4, 1, {45, 45}, {31, 31}, {15, 15}, {1, 1}, false));
    // 1-D rows: long filter; dilated and grouped channels-last.
    add("Row1x31", conv(2, 16, 24, 1, {300}, {31}, {15}, {1}, false));
    add("Row1x15Dil3Groups", conv(2, 8, 8, 4, {1001}, {15}, {21}, {3}, true));
    // A 2-D image with a 1 x s filter (rows independent: the row transforms serve it too).
    add("OneByS", conv(2, 8, 16, 1, {9, 77}, {1, 15}, {0, 7}, {1, 1}, false));
    // True convolution.
    auto convolution = conv(2, 6, 10, 1, {33, 33}, {13, 13}, {6, 6}, {1, 1}, true);
    convolution.mode = ConvMode::CONVOLUTION;
    add("ConvolutionMode", convolution);
    // Depthwise (the fused path: transform, filter multiply, inverse in one launch; backward
    // weights as partition sums folded in order), a 31x31 one channels-last, and a depth
    // multiplier of 2 (the general path: two output channels a group).
    add("Depthwise13x13", conv(2, 16, 16, 16, {40, 40}, {13, 13}, {6, 6}, {1, 1}, false));
    add("Depthwise31x31Nhwc", conv(1, 8, 8, 8, {45, 45}, {31, 31}, {15, 15}, {1, 1}, true));
    add("DepthwiseMult2", conv(2, 8, 16, 8, {23, 27}, {9, 9}, {4, 4}, {1, 1}, false));
    // An output smaller than any tile, no padding.
    add("ValidSmall", conv(3, 8, 8, 1, {9, 9}, {5, 5}, {0, 0}, {1, 1}, false));
    // Several chunks: a 2 MB workspace cap leaves room for about a hundred 16x16 tiles; 405
    // tiles in chunks of 102, so the last chunk is short (the grids stay sized for a full one).
    add("Chunked5x5", conv(5, 8, 8, 1, {100, 100}, {5, 5}, {2, 2}, {1, 1}, false), 2 << 20);
    return cases;
}

template <typename T>
void runCase(ConvShape shape, ConvDirection direction, long long workspaceCap)
{
    const ScopedEnvironmentVariableSetter floor("HIP_KERNEL_PROVIDER_CONV_FFT_FLOOR", "0");
    const ScopedEnvironmentVariableSetter cap("HIP_KERNEL_PROVIDER_CONV_FFT_WORKSPACE_CAP",
                                              workspaceCap > 0 ? std::to_string(workspaceCap) : "");
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
    const bool reflect = shape.mode == ConvMode::CONVOLUTION;
    Tensor<T> reflectedW(spec.w.dims, spec.w.strides);
    std::optional<Tensor<T>> reference;
    Tensor<T>* out = nullptr;
    if(direction == ConvDirection::FORWARD)
    {
        x.fillTensorWithRandomValues(-1.0f, 1.0f, SEED);
        w.fillTensorWithRandomValues(-1.0f, 1.0f, SEED + 1);
        if(reflect)
        {
            reflectFilter(w, reflectedW);
        }
        reference.emplace(y.dims(), y.strides());
        CpuFpReferenceConvolution::fprop<T, T, T, double>(x,
                                                          reflect ? reflectedW : w,
                                                          *reference,
                                                          shape.stride,
                                                          shape.dilation,
                                                          shape.prePadding,
                                                          shape.postPadding);
        out = &y;
    }
    else if(direction == ConvDirection::BACKWARD_DATA)
    {
        y.fillTensorWithRandomValues(-1.0f, 1.0f, SEED);
        w.fillTensorWithRandomValues(-1.0f, 1.0f, SEED + 1);
        if(reflect)
        {
            reflectFilter(w, reflectedW);
        }
        reference.emplace(x.dims(), x.strides());
        CpuFpReferenceConvolution::dgrad<T, T, T, double>(*reference,
                                                          reflect ? reflectedW : w,
                                                          y,
                                                          shape.stride,
                                                          shape.dilation,
                                                          shape.prePadding,
                                                          shape.postPadding);
        out = &x;
    }
    else
    {
        x.fillTensorWithRandomValues(-1.0f, 1.0f, SEED);
        y.fillTensorWithRandomValues(-1.0f, 1.0f, SEED + 1);
        reference.emplace(w.dims(), w.strides());
        if(reflect)
        {
            // The graph's filter is applied reflected, so its gradient is the reflection.
            Tensor<T> dw(w.dims(), w.strides());
            CpuFpReferenceConvolution::wgrad<T, T, T, double>(
                x, dw, y, shape.stride, shape.dilation, shape.prePadding, shape.postPadding);
            reflectFilter(dw, *reference);
        }
        else
        {
            CpuFpReferenceConvolution::wgrad<T, T, T, double>(x,
                                                              *reference,
                                                              y,
                                                              shape.stride,
                                                              shape.dilation,
                                                              shape.prePadding,
                                                              shape.postPadding);
        }
        out = &w;
    }

    const auto* expected = static_cast<const T*>(reference->rawHostData());
    float largest = 0.0f;
    forEachIndex(reference->dims(), [&](const std::vector<int64_t>& index) {
        largest = std::max(
            largest,
            std::fabs(static_cast<float>(expected[offsetOf(index, reference->strides())])));
    });

    const auto& handler = dispatchHandler(pack);
    int ran = 0;
    for(const auto& kernel : catalog(direction))
    {
        if(!matchesKernel(pack, fixture.context(), kernel, *bound))
        {
            continue;
        }
        // 16-bit tensors change only the loads and stores: run them on every transform size and
        // block (the 256-thread, 64-channel product configs), fp32 on every config.
        if(!std::is_same_v<T, float>
           && (kernel.getIntMetadata("tb") != 256 || kernel.getIntMetadata("gn") != 64
               || kernel.getIntMetadata("tm") != 4))
        {
            continue;
        }
        SCOPED_TRACE(kernel.name);
        ++ran;
        const auto config = conv_fft::configOf(kernel);
        ASSERT_TRUE(config.has_value());
        const auto prepared = handler.prepare(fixture.context(), *bound, kernel);
        ASSERT_NE(prepared, nullptr);
        const size_t workspaceBytes = handler.workspaceBytes(fixture.context(), *bound, kernel);
        void* workspace = nullptr;
        ASSERT_GT(workspaceBytes, 0u);
        ASSERT_EQ(hipMalloc(&workspace, workspaceBytes), hipSuccess);
        out->fillWithSentinelValue();
        const std::array<hipdnnPluginDeviceBuffer_t, 3> buffers{
            hipdnnPluginDeviceBuffer_t{X_UID, x.rawDeviceData()},
            hipdnnPluginDeviceBuffer_t{W_UID, w.rawDeviceData()},
            hipdnnPluginDeviceBuffer_t{Y_UID, y.rawDeviceData()}};
        const Handle handle;
        handler.launch(
            handle, *prepared, buffers.data(), static_cast<uint32_t>(buffers.size()), workspace);
        ASSERT_EQ(hipStreamSynchronize(handle.getStream()), hipSuccess);
        ASSERT_EQ(hipFree(workspace), hipSuccess);
        out->markDeviceModified();

        const float fftBound
            = 8.0f * std::log2(static_cast<float>(config->th * config->tw)) * 1.2e-7f * largest;
        const auto* actual = static_cast<const T*>(out->rawHostData());
        int64_t mismatches = 0;
        forEachIndex(out->dims(), [&](const std::vector<int64_t>& index) {
            const float got = static_cast<float>(actual[offsetOf(index, out->strides())]);
            const float want = static_cast<float>(expected[offsetOf(index, reference->strides())]);
            const float tolerance = fftBound + 2.0f * storageEpsilon<T>() * std::fabs(want);
            if(!(std::fabs(got - want) <= tolerance) && ++mismatches <= 5)
            {
                ADD_FAILURE() << "at linear index " << offsetOf(index, out->strides()) << ": got "
                              << got << ", want " << want << ", tolerance " << tolerance;
            }
        });
        EXPECT_EQ(mismatches, 0);
    }
    EXPECT_GE(ran, 1) << "no catalog config serves this case";
}

class TestGpuConvFft : public ::testing::TestWithParam<DeviceCase>
{
};

TEST_P(TestGpuConvFft, Float)
{
    SKIP_IF_NO_DEVICES();
    runCase<float>(GetParam().shape, GetParam().direction, GetParam().workspaceCap);
}

TEST_P(TestGpuConvFft, Half)
{
    SKIP_IF_NO_DEVICES();
    runCase<Half>(GetParam().shape, GetParam().direction, GetParam().workspaceCap);
}

TEST_P(TestGpuConvFft, BFloat16)
{
    SKIP_IF_NO_DEVICES();
    runCase<BFloat16>(GetParam().shape, GetParam().direction, GetParam().workspaceCap);
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuConvFft,
                         ::testing::ValuesIn(deviceCases()),
                         [](const auto& info) { return info.param.name; });

// ---- Host-only: the shipped catalog ----

TEST(TestConvFftCatalog, EveryConfigParsesAndNamesItsDirection)
{
    const std::array<std::pair<ConvDirection, const char*>, 3> directions{
        std::pair{ConvDirection::FORWARD, "FWD"},
        std::pair{ConvDirection::BACKWARD_DATA, "BWD"},
        std::pair{ConvDirection::BACKWARD_WEIGHTS, "WRW"}};
    for(const auto& [direction, name] : directions)
    {
        ASSERT_FALSE(catalog(direction).empty());
        for(const auto& kernel : catalog(direction))
        {
            SCOPED_TRACE(kernel.name);
            EXPECT_TRUE(conv_fft::configOf(kernel).has_value());
            EXPECT_EQ(kernel.getStringMetadata("direction"), name);
        }
    }
}

} // namespace

#endif
