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
 * @file TestGpuConvDepthwiseMfma.cpp
 * @brief The matrix-core depthwise kernels (CONV_DEPTHWISE_MFMA_FWD, CONV_DEPTHWISE_MFMA_BWD) on
 *        the device, through the engines' own graph match, kernel match, prepare and launch,
 *        against hipDNN's CPU reference convolution.
 *
 * Every config of the shipped catalog that the kernel match admits on the current device runs on
 * every case, in fp16 and bf16 (a device without the configs' MFMA admits none and the case is
 * skipped). The cases cover every catalog filter at 128 channels (every wave count), narrow and
 * odd extents against the 32-column tile and the folded images, a batch the fold does not
 * divide, channel counts of one wave, asymmetric and zero padding, unit-stride backward data (the
 * forward form with the filter reflected), true convolution, stride 2 forward, and channels-first
 * cases for its own core. A host-only test checks that
 * every config serves some case. The CPU reference is cross-correlation only, so true convolution
 * reflects the filter around it.
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
            direction == ConvDirection::FORWARD ? "hipkernel.conv_depthwise_mfma_fwd.graph_match"
                                                : "hipkernel.conv_depthwise_mfma_bwd.graph_match",
            "",
            "hipkernel.conv_depthwise_mfma.kernel_match",
            "",
            "hipkernel.conv_depthwise_mfma.dispatch",
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
                = i == 0 ? "conv_depthwise_mfma_fwd.kdp.json" : "conv_depthwise_mfma_bwd.kdp.json";
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

std::vector<DeviceCase> deviceCases()
{
    constexpr auto FWD = ConvDirection::FORWARD;
    constexpr auto BWD = ConvDirection::BACKWARD_DATA;
    std::vector<DeviceCase> cases;
    // Every catalog filter at 128 channels: every wave count divides them.
    cases.push_back({"K3C128", depthwise(128, {11, 37}, {3, 3}, {1, 1}, {1, 1}, true), FWD});
    cases.push_back({"K5C128", depthwise(128, {9, 14}, {5, 5}, {2, 2}, {1, 1}, true), FWD});
    cases.push_back({"K7C128", depthwise(128, {13, 15}, {7, 7}, {3, 3}, {1, 1}, true), FWD});
    cases.push_back({"K9C128", depthwise(128, {10, 12}, {9, 9}, {4, 4}, {1, 1}, true), FWD});
    // One wave of channels; extents odd against the 32-column tile and its 16- and 8-wide folds.
    cases.push_back({"K3C8", depthwise(8, {13, 37}, {3, 3}, {1, 1}, {1, 1}, true), FWD});
    cases.push_back({"K5C24", depthwise(24, {6, 5}, {5, 5}, {2, 2}, {1, 1}, true), FWD});
    cases.push_back({"K7C16Wide", depthwise(16, {5, 70}, {7, 7}, {3, 3}, {1, 1}, true), FWD});
    // Zero padding; batches of 3 and 5 against folds of 2 and 4.
    auto unpadded = depthwise(64, {33, 33}, {3, 3}, {0, 0}, {1, 1}, true);
    unpadded.n = 3;
    cases.push_back({"K3Pad0N3", unpadded, FWD});
    auto odd = depthwise(32, {9, 7}, {7, 7}, {3, 3}, {1, 1}, true);
    odd.n = 5;
    cases.push_back({"K7N5", odd, FWD});
    // Asymmetric padding; a single pixel.
    auto asymmetric = depthwise(32, {12, 13}, {7, 7}, {2, 1}, {1, 1}, true);
    asymmetric.postPadding = {4, 5};
    cases.push_back({"K7Asymmetric", asymmetric, FWD});
    cases.push_back({"K7OnePixel", depthwise(128, {1, 1}, {7, 7}, {3, 3}, {1, 1}, true), FWD});
    auto convolution = depthwise(32, {17, 19}, {3, 3}, {1, 1}, {1, 1}, true);
    convolution.mode = ConvMode::CONVOLUTION;
    cases.push_back({"K3ConvolutionMode", convolution, FWD});
    // Channels-first: every catalog filter, odd channel counts and a single pair, rows wider than
    // a 128-column tile, the first plane's padding (reached before the buffer's start), folds
    // of up to 8 images over a batch of 9.
    const auto first = [](ConvShape shape) {
        shape.channelsLast = false;
        return shape;
    };
    cases.push_back({"NchwK3C5", first(depthwise(5, {13, 37}, {3, 3}, {1, 1}, {1, 1})), FWD});
    cases.push_back({"NchwK5C3Wide", first(depthwise(3, {9, 150}, {5, 5}, {2, 2}, {1, 1})), FWD});
    cases.push_back({"NchwK7C2", first(depthwise(2, {6, 5}, {7, 7}, {3, 3}, {1, 1})), FWD});
    cases.push_back({"NchwK9C4", first(depthwise(4, {11, 12}, {9, 9}, {4, 4}, {1, 1})), FWD});
    auto nchwBatch = first(depthwise(6, {7, 14}, {3, 3}, {1, 1}, {1, 1}));
    nchwBatch.n = 9;
    cases.push_back({"NchwK3N9", nchwBatch, FWD});
    // Planes smaller than the padding plus a tile, every catalog filter, both layouts: a k x k
    // image to a single pixel (unpadded) and a single pixel padded by k - 1 to a k x k image.
    // Backward data of the first is MobileFaceNet's 7 x 7 depthwise backward (dy 1 x 1): with a
    // 1-element plane, loads of several channel pairs reach before the buffer's start or past
    // its end, not only the first and the last pair's.
    for(const int64_t k : {3, 5, 7, 9})
    {
        const std::string tag = "K" + std::to_string(k);
        for(const bool channelsLast : {false, true})
        {
            const std::string layout = channelsLast ? "Nhwc" : "Nchw";
            auto toPixel = depthwise(128, {k, k}, {k, k}, {0, 0}, {1, 1}, channelsLast);
            toPixel.n = 16;
            cases.push_back({layout + tag + "ToPixel", toPixel, FWD});
            auto fromPixel = depthwise(128, {1, 1}, {k, k}, {k - 1, k - 1}, {1, 1}, channelsLast);
            fromPixel.n = 5;
            cases.push_back({layout + tag + "FromPixel", fromPixel, FWD});
        }
    }
    // Unit-stride backward data runs the forward form (filter reflected): every case again.
    const size_t forwardCases = cases.size();
    for(size_t i = 0; i < forwardCases; ++i)
    {
        cases.push_back({cases[i].name, cases[i].shape, BWD});
    }
    // Stride 2, forward only (strided backward data is a gather the VALU engines run): every
    // strided catalog filter at 128 channels, one wave of channels with odd extents, a batch of 3
    // without padding.
    const auto strided = [](ConvShape shape) {
        shape.stride = {2, 2};
        return shape;
    };
    cases.push_back(
        {"S2K3C128", strided(depthwise(128, {17, 37}, {3, 3}, {1, 1}, {1, 1}, true)), FWD});
    cases.push_back(
        {"S2K5C128", strided(depthwise(128, {13, 20}, {5, 5}, {2, 2}, {1, 1}, true)), FWD});
    cases.push_back({"S2K5C8", strided(depthwise(8, {9, 11}, {5, 5}, {2, 2}, {1, 1}, true)), FWD});
    auto stridedUnpadded = strided(depthwise(64, {16, 66}, {3, 3}, {0, 0}, {1, 1}, true));
    stridedUnpadded.n = 3;
    cases.push_back({"S2K3Pad0N3", stridedUnpadded, FWD});
    auto nchwStrided = strided(depthwise(3, {17, 300}, {3, 3}, {1, 1}, {1, 1}));
    nchwStrided.channelsLast = false;
    cases.push_back({"NchwS2K3C3", nchwStrided, FWD});
    auto nchwStrided5 = strided(depthwise(5, {9, 31}, {5, 5}, {2, 2}, {1, 1}));
    nchwStrided5.channelsLast = false;
    nchwStrided5.n = 9;
    cases.push_back({"NchwS2K5N9", nchwStrided5, FWD});
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
    float accumulationBound = 1e-6f * static_cast<float>(taps);
    // gfx90a's fp16 MFMA flushes denormal inputs to zero (a hardware property, not this
    // kernel's): each tap may lose up to the smallest normal fp16 times |w| <= 1.
    if constexpr(std::is_same_v<T, Half>)
    {
        if(std::string(fixture.context().deviceProperties.gcnArchName).rfind("gfx90a", 0) == 0)
        {
            accumulationBound += 6.11e-5f * static_cast<float>(taps);
        }
    }

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
    if(ran == 0)
    {
        GTEST_SKIP() << "no catalog config runs on this device";
    }
}

class TestGpuConvDepthwiseMfma : public ::testing::TestWithParam<DeviceCase>
{
};

TEST_P(TestGpuConvDepthwiseMfma, Half)
{
    SKIP_IF_NO_DEVICES();
    runCase<Half>(GetParam().shape, GetParam().direction);
}

TEST_P(TestGpuConvDepthwiseMfma, BFloat16)
{
    SKIP_IF_NO_DEVICES();
    runCase<BFloat16>(GetParam().shape, GetParam().direction);
}

INSTANTIATE_TEST_SUITE_P(
    , TestGpuConvDepthwiseMfma, ::testing::ValuesIn(deviceCases()), [](const auto& info) {
        return info.param.name + (info.param.direction == ConvDirection::FORWARD ? "Fwd" : "Bwd");
    });

// Host-only: every shipped config is admitted by some device case on a device that runs its
// MFMA, so the device suite runs every config there.
TEST(TestConvDepthwiseMfmaCatalog, EveryConfigServesADeviceCase)
{
    auto properties = testDeviceProperties();
    properties.gcnArchName = "gfx950:sramecc+:xnack-";
    properties.multiProcessorCount = 256;
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
                shape.dataType = kernel.getStringMetadata("dtype") == "HALF" ? DataType::HALF
                                                                             : DataType::BFLOAT16;
                const GraphFixture fixture(buildGraph(specFor(shape, direction)), properties);
                const auto bound = matchesGraph(pack, fixture.context());
                if(bound.has_value() && matchesKernel(pack, fixture.context(), kernel, *bound))
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
