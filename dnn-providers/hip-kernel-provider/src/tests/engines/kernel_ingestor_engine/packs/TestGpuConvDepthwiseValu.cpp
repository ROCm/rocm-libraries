// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceConvolution.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestGpuConvDepthwiseValu.cpp
 * @brief The VALU depthwise kernels on the device, through the engines' own graph match,
 *        kernel match, prepare and launch, against hipDNN's CPU reference convolution.
 *
 * Every kernel config of the catalog (mirrored from descriptors/conv/depthwise_valu, generated
 * from the same list) runs on every shape whose kernel match admits it, forward and backward
 * data, in fp32, fp16 and bf16; a host-only test checks that every config serves some case. The CPU
 * reference is cross-correlation only, so true convolution reflects the filter around it.
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

/// One catalog config: KMD metadata minus dtype and direction.
struct Config
{
    Config(std::string layoutName, int64_t rank, std::string variantName)
        : layout(std::move(layoutName))
        , ndims(rank)
        , variant(std::move(variantName))
    {
    }

    std::string layout;
    int64_t ndims;
    std::string variant;
    int64_t kd = 0, kh = 0, kw = 0, wstrip = 0, th = 0, tw = 0;
    int64_t bd = 0, bh = 0, bw = 0, bk = 0, rd = 0, rh = 0, rw = 0;

    std::string tag() const
    {
        return layout + std::to_string(ndims) + "d_" + variant + "_k" + std::to_string(kd) + "x"
               + std::to_string(kh) + "x" + std::to_string(kw) + "_t" + std::to_string(th) + "x"
               + std::to_string(tw) + "_b" + std::to_string(bd) + "x" + std::to_string(bh) + "x"
               + std::to_string(bw) + "x" + std::to_string(bk);
    }
};

Config floorConfig(const std::string& layout, int64_t ndims, int64_t wstrip)
{
    Config config{layout, ndims, "floor"};
    config.wstrip = wstrip;
    return config;
}

Config microtileConfig(int64_t th, int64_t tw)
{
    Config config{"NHWC", 2, "microtile"};
    config.kh = config.kw = 3;
    config.th = th;
    config.tw = tw;
    return config;
}

Config blockedConfig(const std::string& variant, int64_t k, int64_t block, int64_t bk, int64_t r)
{
    Config config{"NHWC", 2, variant};
    config.kh = config.kw = k;
    config.bh = config.bw = block;
    config.bk = bk;
    config.rh = config.rw = r;
    return config;
}

// The catalog of descriptors/conv/depthwise_valu (~/scratch/conv-dw-gen/gen.py CONFIGS).
const std::vector<Config>& catalog()
{
    static const std::vector<Config> s_configs = [] {
        std::vector<Config> configs{floorConfig("NHWC", 2, 4),
                                    floorConfig("NCHW", 2, 0),
                                    floorConfig("NHWC", 3, 4),
                                    floorConfig("NCHW", 3, 0),
                                    microtileConfig(2, 4),
                                    microtileConfig(2, 8),
                                    microtileConfig(4, 4),
                                    blockedConfig("fused", 5, 16, 32, 4),
                                    blockedConfig("fused", 5, 8, 32, 2),
                                    blockedConfig("fused", 7, 16, 32, 4),
                                    blockedConfig("fused", 7, 8, 32, 2),
                                    blockedConfig("lds", 9, 8, 16, 2)};
        Config lds3d = blockedConfig("lds", 3, 8, 8, 2);
        lds3d.ndims = 3;
        lds3d.kd = 3;
        lds3d.bd = 4;
        lds3d.rd = 2;
        configs.push_back(lds3d);
        return configs;
    }();
    return s_configs;
}

PackSymbols packFor(ConvDirection direction)
{
    return {"",
            direction == ConvDirection::FORWARD ? "hipkernel.conv_depthwise_valu_fwd.graph_match"
                                                : "hipkernel.conv_depthwise_valu_bwd.graph_match",
            "",
            "hipkernel.conv_depthwise_valu.kernel_match",
            "",
            "hipkernel.conv_depthwise_valu.dispatch",
            "",
            "",
            ""};
}

hipdnn_plugin_sdk::ingestor::KernelDefinition
    kernelFor(const Config& config, DataType dataType, ConvDirection direction)
{
    hipdnn_plugin_sdk::ingestor::KernelDefinition kernel;
    kernel.name = "conv_depthwise_valu_test." + config.tag();
    kernel.source.kind = hipdnn_plugin_sdk::ingestor::KernelSourceKind::EMBEDDED_SOURCE;
    kernel.source.sourceFile = "kernels/ConvDepthwiseValu.cpp";
    kernel.source.entryPoint = "ConvDepthwiseValu";
    kernel.metadata
        = {{"dtype", std::string(hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(dataType))},
           {"direction", std::string(direction == ConvDirection::FORWARD ? "FWD" : "BWD")},
           {"layout", config.layout},
           {"ndims", config.ndims},
           {"variant", config.variant},
           {"kd", config.kd},
           {"kh", config.kh},
           {"kw", config.kw},
           {"wstrip", config.wstrip},
           {"th", config.th},
           {"tw", config.tw},
           {"bd", config.bd},
           {"bh", config.bh},
           {"bw", config.bw},
           {"bk", config.bk},
           {"rd", config.rd},
           {"rh", config.rh},
           {"rw", config.rw}};
    return kernel;
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
};

ConvShape depthwise(int64_t channels,
                    std::vector<int64_t> input,
                    std::vector<int64_t> filter,
                    std::vector<int64_t> pad,
                    bool channelsLast)
{
    ConvShape shape;
    shape.n = 2;
    shape.c = shape.k = shape.groups = channels;
    shape.input = std::move(input);
    shape.filter = std::move(filter);
    shape.prePadding = pad;
    shape.postPadding = pad;
    shape.stride.assign(shape.input.size(), 1);
    shape.dilation.assign(shape.input.size(), 1);
    shape.channelsLast = channelsLast;
    return shape;
}

// Channel counts are not multiples of the 32- or 16-wide channel blocks, and planes are not
// multiples of the tiles, so every edge path runs.
std::vector<DeviceCase> deviceCases()
{
    std::vector<DeviceCase> cases;
    cases.push_back({"Nhwc3x3", depthwise(37, {13, 11}, {3, 3}, {1, 1}, true)});
    cases.push_back({"Nhwc5x5", depthwise(37, {19, 17}, {5, 5}, {2, 2}, true)});
    cases.push_back({"Nhwc7x7", depthwise(20, {18, 21}, {7, 7}, {3, 3}, true)});
    cases.push_back({"Nhwc9x9", depthwise(19, {15, 14}, {9, 9}, {4, 4}, true)});
    cases.push_back({"Nchw3x3", depthwise(9, {13, 11}, {3, 3}, {1, 1}, false)});
    cases.push_back({"Ndhwc3x3x3", depthwise(11, {5, 9, 10}, {3, 3, 3}, {1, 1, 1}, true)});
    cases.push_back({"Ncdhw3x3x3", depthwise(7, {5, 6, 7}, {3, 3, 3}, {1, 1, 1}, false)});
    cases.push_back({"Nhwc1d", depthwise(13, {23}, {5}, {2}, true)});

    auto asymmetric = depthwise(13, {12, 13}, {3, 5}, {0, 2}, true);
    asymmetric.prePadding = {2, 1};
    asymmetric.postPadding = {0, 3};
    asymmetric.dilation = {2, 1};
    cases.push_back({"NhwcAsymmetricPaddingDilation", asymmetric});

    auto nchwDilated = depthwise(6, {12, 13}, {3, 3}, {2, 1}, false);
    nchwDilated.dilation = {2, 1};
    cases.push_back({"NchwDilated", nchwDilated});

    auto convolution = depthwise(37, {13, 11}, {3, 3}, {1, 1}, true);
    convolution.mode = ConvMode::CONVOLUTION;
    cases.push_back({"Nhwc3x3ConvolutionMode", convolution});

    // Forward only: backward data is served for unit stride.
    auto strided = depthwise(21, {17, 15}, {3, 3}, {1, 1}, true);
    strided.stride = {2, 2};
    cases.push_back({"Nhwc3x3Stride2", strided});
    auto nchwStrided = depthwise(5, {17, 15}, {5, 3}, {2, 1}, false);
    nchwStrided.stride = {2, 3};
    cases.push_back({"NchwStrided", nchwStrided});
    return cases;
}

template <typename T>
void runCase(ConvShape shape, ConvDirection direction)
{
    shape.dataType = dataTypeOf<T>();
    const bool strided = std::any_of(
        shape.stride.begin(), shape.stride.end(), [](int64_t value) { return value != 1; });
    const auto spec = specFor(shape, direction);
    const GraphFixture fixture(buildGraph(spec), currentDeviceProperties());
    const auto pack = packFor(direction);
    const auto bound = matchesGraph(pack, fixture.context());
    if(direction == ConvDirection::BACKWARD_DATA && strided)
    {
        EXPECT_FALSE(bound.has_value()) << "strided backward data is not served";
        return;
    }
    ASSERT_TRUE(bound.has_value()) << "the engine refused a case it should serve";

    Tensor<T> x(spec.x.dims, spec.x.strides);
    Tensor<T> w(spec.w.dims, spec.w.strides);
    Tensor<T> y(spec.y.dims, spec.y.strides);
    constexpr unsigned int SEED = 4321;
    const bool forward = direction == ConvDirection::FORWARD;
    (forward ? x : y).fillTensorWithRandomValues(-1.0f, 1.0f, SEED);
    w.fillTensorWithRandomValues(-1.0f, 1.0f, SEED + 1);
    Tensor<T>& written = forward ? y : x;

    // The reference, once per case.
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
    for(const auto& config : catalog())
    {
        const auto kernel = kernelFor(config, shape.dataType, direction);
        if(!matchesKernel(pack, fixture.context(), kernel, *bound))
        {
            continue;
        }
        SCOPED_TRACE(config.tag());
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

class TestGpuConvDepthwiseValu
    : public ::testing::TestWithParam<std::tuple<DeviceCase, ConvDirection>>
{
};

TEST_P(TestGpuConvDepthwiseValu, Float)
{
    SKIP_IF_NO_DEVICES();
    runCase<float>(std::get<0>(GetParam()).shape, std::get<1>(GetParam()));
}

TEST_P(TestGpuConvDepthwiseValu, Half)
{
    SKIP_IF_NO_DEVICES();
    runCase<Half>(std::get<0>(GetParam()).shape, std::get<1>(GetParam()));
}

TEST_P(TestGpuConvDepthwiseValu, BFloat16)
{
    SKIP_IF_NO_DEVICES();
    runCase<BFloat16>(std::get<0>(GetParam()).shape, std::get<1>(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuConvDepthwiseValu,
                         ::testing::Combine(::testing::ValuesIn(deviceCases()),
                                            ::testing::Values(ConvDirection::FORWARD,
                                                              ConvDirection::BACKWARD_DATA)),
                         [](const auto& info) {
                             return std::get<0>(info.param).name
                                    + (std::get<1>(info.param) == ConvDirection::FORWARD ? "Fwd"
                                                                                         : "Bwd");
                         });

// Host-only: every config of the catalog is admitted by some device case, so the device suite
// runs every config (forward; a halo config's backward data runs on the same unit-stride case).
TEST(TestConvDepthwiseValuCatalog, EveryConfigServesADeviceCase)
{
    const auto pack = packFor(ConvDirection::FORWARD);
    for(const auto& config : catalog())
    {
        bool served = false;
        for(const auto& deviceCase : deviceCases())
        {
            auto shape = deviceCase.shape;
            shape.dataType = DataType::HALF;
            const GraphFixture fixture(buildGraph(specFor(shape, ConvDirection::FORWARD)));
            const auto bound = matchesGraph(pack, fixture.context());
            if(bound.has_value()
               && matchesKernel(pack,
                                fixture.context(),
                                kernelFor(config, DataType::HALF, ConvDirection::FORWARD),
                                *bound))
            {
                served = true;
                break;
            }
        }
        EXPECT_TRUE(served) << config.tag() << " serves no device case";
    }
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_CONV
