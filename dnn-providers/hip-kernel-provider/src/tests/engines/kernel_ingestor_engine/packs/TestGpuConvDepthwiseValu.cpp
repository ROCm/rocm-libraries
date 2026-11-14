// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
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
 * reference is cross-correlation only, so true convolution reflects the filter around it. The
 * backward-weights engine (CONV_DEPTHWISE_VALU_WRW) runs every config of its catalog on the same
 * cases, from a workspace its handler sizes, in all three element types.
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
    /// Phase blocks: the baked stride; vec -1 = 16 bytes of the element type (channels-last).
    int64_t stride = 0, vec = 0;

    std::string tag() const
    {
        return layout + std::to_string(ndims) + "d_" + variant + "_k" + std::to_string(kd) + "x"
               + std::to_string(kh) + "x" + std::to_string(kw) + "_t" + std::to_string(th) + "x"
               + std::to_string(tw) + "_b" + std::to_string(bd) + "x" + std::to_string(bh) + "x"
               + std::to_string(bw) + "x" + std::to_string(bk) + "_r" + std::to_string(rh) + "x"
               + std::to_string(rw) + "_s" + std::to_string(stride) + "_v" + std::to_string(vec);
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

Config phaseConfig(const std::string& layout, int64_t k, int64_t bk)
{
    Config config{layout, 2, "phase"};
    config.kh = config.kw = k;
    config.stride = 2;
    config.bk = bk;
    config.vec = layout == "NHWC" ? -1 : 0;
    return config;
}

Config rowtileConfig(int64_t k, int64_t th, int64_t tw, int64_t vec, int64_t stride = 1)
{
    Config config{"NHWC", 2, "rowtile"};
    config.kh = config.kw = k;
    config.th = th;
    config.tw = tw;
    config.vec = vec;
    config.stride = stride;
    return config;
}

/// The rowtile tiles the generator ships (gen_depthwise_valu.py ROWTILE_TILES, ROWTILE_S2_TILES),
/// each with 1 and 2 channels per lane.
const std::vector<Config>& rowtileConfigs()
{
    static const std::vector<Config> s_configs = [] {
        std::vector<Config> configs;
        const std::vector<std::array<int64_t, 3>> tiles{{3, 2, 8},
                                                        {3, 4, 4},
                                                        {3, 4, 8},
                                                        {3, 7, 7},
                                                        {5, 4, 4},
                                                        {5, 4, 8},
                                                        {5, 7, 7},
                                                        {7, 2, 14},
                                                        {7, 4, 4},
                                                        {7, 4, 8},
                                                        {7, 7, 7}};
        for(const auto& [k, th, tw] : tiles)
        {
            for(const int64_t vec : {1, 2})
            {
                configs.push_back(rowtileConfig(k, th, tw, vec));
            }
        }
        // Stride 2, forward only.
        const std::vector<std::array<int64_t, 3>> stride2{
            {3, 2, 8}, {3, 4, 4}, {3, 4, 8}, {5, 2, 8}, {5, 4, 4}};
        for(const auto& [k, th, tw] : stride2)
        {
            for(const int64_t vec : {1, 2})
            {
                configs.push_back(rowtileConfig(k, th, tw, vec, 2));
            }
        }
        return configs;
    }();
    return s_configs;
}

Config planeConfig(int64_t k, int64_t bh, int64_t bw, int64_t bk, int64_t rh, int64_t rw)
{
    Config config{"NCHW", 2, "lds"};
    config.kh = config.kw = k;
    config.bh = bh;
    config.bw = bw;
    config.bk = bk;
    config.rh = rh;
    config.rw = rw;
    return config;
}

// The catalog of descriptors/conv/depthwise_valu (conv/generators/gen_depthwise_valu.py).
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
        // Backward data only: the gather over the taps.
        for(const auto* layout : {"NHWC", "NCHW"})
        {
            for(const int64_t ndims : {2, 3})
            {
                configs.emplace_back(layout, ndims, "gather");
            }
        }
        // Backward data only: stride-2 phase blocks.
        for(const int64_t k : {3, 5, 7})
        {
            for(const int64_t bk : {8, 16, 32})
            {
                configs.push_back(phaseConfig("NHWC", k, bk));
            }
            configs.push_back(phaseConfig("NCHW", k, 0));
        }
        for(const auto& plane : {planeConfig(3, 16, 64, 1, 2, 4),
                                 planeConfig(3, 16, 32, 2, 2, 4),
                                 planeConfig(3, 8, 16, 8, 2, 2),
                                 planeConfig(5, 16, 32, 2, 2, 4),
                                 planeConfig(5, 8, 8, 8, 2, 2),
                                 planeConfig(7, 16, 32, 2, 2, 4),
                                 planeConfig(7, 8, 8, 8, 2, 2)})
        {
            configs.push_back(plane);
        }
        configs.insert(configs.end(), rowtileConfigs().begin(), rowtileConfigs().end());
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
           {"rw", config.rw},
           {"stride", config.stride},
           {"vec",
            config.vec == -1 ? int64_t{16} / (dataType == DataType::FLOAT ? 4 : 2) : config.vec}};
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
                    const std::vector<int64_t>& pad,
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
    // Planes past one wide tile in both axes, and channel counts past the small tiles' blocks.
    cases.push_back({"Nchw3x3Wide", depthwise(3, {37, 70}, {3, 3}, {1, 1}, false)});
    cases.push_back({"Nchw5x5", depthwise(11, {19, 37}, {5, 5}, {2, 2}, false)});
    cases.push_back({"Nchw7x7", depthwise(10, {18, 35}, {7, 7}, {3, 3}, false)});
    // Rows narrower than the backward-weights register strip (8 columns), the filter wider
    // than the row.
    cases.push_back({"Nhwc7x7Narrow", depthwise(20, {9, 5}, {7, 7}, {3, 3}, true)});
    cases.push_back({"Nchw7x7Narrow", depthwise(10, {9, 5}, {7, 7}, {3, 3}, false)});
    // Whole 8-channel vectors (the matrix-core backward weights): a partial channel block,
    // several column tiles (width past 64), heights not whole multiples of the row step,
    // padding below the filter's half and at it.
    cases.push_back({"Nhwc3x3C40", depthwise(40, {13, 21}, {3, 3}, {1, 1}, true)});
    cases.push_back({"Nhwc5x5C72Pad1", depthwise(72, {11, 37}, {5, 5}, {1, 1}, true)});
    cases.push_back({"Nhwc7x7C136Wide", depthwise(136, {9, 70}, {7, 7}, {3, 3}, true)});
    cases.push_back({"Nhwc7x7C16Narrow", depthwise(16, {10, 5}, {7, 7}, {3, 3}, true)});
    // Register row tiles: two channels per lane over 65 groups (a second block of one lane),
    // padding past the filter's half (output rows and columns of padding only), one-row images.
    cases.push_back({"Nhwc5x5C130Pad3", depthwise(130, {9, 11}, {5, 5}, {3, 3}, true)});
    cases.push_back({"Nhwc3x3C64H1", depthwise(64, {1, 9}, {3, 3}, {1, 1}, true)});
    auto stride5 = depthwise(40, {19, 17}, {5, 5}, {2, 2}, true);
    stride5.stride = {2, 2};
    cases.push_back({"Nhwc5x5Stride2C40", stride5});
    // Stride 2 on the matrix core: a row wider than one 32-column tile (two tiles, odd widths),
    // and 7x7 (two tap groups a side, the odd one starting two halves into its word).
    auto stride3 = depthwise(24, {13, 71}, {3, 3}, {1, 1}, true);
    stride3.stride = {2, 2};
    cases.push_back({"Nhwc3x3Stride2C24Wide", stride3});
    auto stride7 = depthwise(16, {18, 21}, {7, 7}, {3, 3}, true);
    stride7.stride = {2, 2};
    cases.push_back({"Nhwc7x7Stride2C16", stride7});
    cases.push_back({"Ndhwc3x3x3", depthwise(11, {5, 9, 10}, {3, 3, 3}, {1, 1, 1}, true)});
    cases.push_back({"Ncdhw3x3x3", depthwise(7, {5, 6, 7}, {3, 3, 3}, {1, 1, 1}, false)});
    cases.push_back({"Nhwc1d", depthwise(13, {23}, {5}, {2}, true)});
    // 1-D filters the backward-weights register tiles bake: Mamba's 4 (causal padding) and
    // Conformer's 31.
    cases.push_back({"Nhwc1dK4", depthwise(37, {50}, {4}, {3}, true)});
    cases.push_back({"Nchw1dK4", depthwise(7, {50}, {4}, {3}, false)});
    cases.push_back({"Nhwc1dK31", depthwise(9, {70}, {31}, {15}, true)});
    cases.push_back({"Nchw1dK31", depthwise(5, {70}, {31}, {15}, false)});

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

    // Strided: backward data runs only through the gather.
    auto strided = depthwise(21, {17, 15}, {3, 3}, {1, 1}, true);
    strided.stride = {2, 2};
    cases.push_back({"Nhwc3x3Stride2", strided});
    auto nchwStrided = depthwise(5, {17, 15}, {5, 3}, {2, 1}, false);
    nchwStrided.stride = {2, 3};
    cases.push_back({"NchwStrided", nchwStrided});
    // Strided layers whose output extent is 1 (hipconv #15149 review: a stride normalized for a
    // size-1 axis): 4x4 3x3 stride 2 -> 1x1, and 1 in H only (4x9 -> 1x5).
    auto toOne = depthwise(16, {4, 4}, {3, 3}, {0, 0}, true);
    toOne.stride = {2, 2};
    cases.push_back({"NhwcStride2OutputOne", toOne});
    auto toOneH = depthwise(6, {4, 9}, {3, 3}, {0, 1}, false);
    toOneH.stride = {2, 2};
    cases.push_back({"NchwStride2OutputOneHigh", toOneH});
    // Stride-2 filters the phase blocks bake, both layouts; channels that no vector or block
    // divides, odd planes, padding below and at the filter's half, and none.
    auto phase = [&](const char* name,
                     int64_t c,
                     std::vector<int64_t> in,
                     int64_t k,
                     int64_t pad,
                     bool channelsLast) {
        auto shape = depthwise(c, std::move(in), {k, k}, {pad, pad}, channelsLast);
        shape.stride = {2, 2};
        cases.push_back({name, shape});
    };
    phase("NchwPhase3x3", 19, {17, 14}, 3, 1, false);
    phase("NhwcPhase5x5", 43, {19, 21}, 5, 2, true);
    phase("NchwPhase5x5", 9, {16, 19}, 5, 2, false);
    phase("NhwcPhase7x7", 24, {23, 18}, 7, 3, true);
    phase("NchwPhase7x7Unpadded", 5, {21, 16}, 7, 0, false);
    phase("NhwcPhase3x3Unpadded", 64, {15, 16}, 3, 0, true);
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
            const auto got = static_cast<float>(actual[offsetOf(index, written.strides())]);
            const auto want = static_cast<float>(expected[offsetOf(index, reference.strides())]);
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

// ---- Backward weights (CONV_DEPTHWISE_VALU_WRW) ----

/// A device that reports @p units dispatch units: the backward-weights partitions are sized
/// from them.
hipdnn_plugin_sdk::ingestor::DeviceProperties countedDevice(int units = 64,
                                                            const std::string& arch = "gfx000")
{
    auto properties = testDeviceProperties();
    properties.multiProcessorCount = units;
    properties.gcnArchName = arch;
    return properties;
}

/// One backward-weights catalog config (conv/generators/gen_depthwise_valu_wrw.py): KMD
/// metadata minus dtype.
struct WrwConfig
{
    std::string layout;
    std::string variant;
    int64_t kd = 0;
    int64_t kh = 0;
    int64_t kw = 0;
    int64_t q = 0;
    int64_t g = 0;
    int64_t nwg = 0;
    int64_t bpc = 0;
    /// The depth multiplier (output channels per input channel).
    int64_t mult = 1;
    /// The hankel configs' stride (both axes); 0 for the VALU variants.
    int64_t s = 0;
    /// The vector tiles' channels a thread and output columns a strip; 0 otherwise.
    int64_t cv = 0;
    int64_t strip = 0;

    std::string tag() const
    {
        return layout + "_" + variant + "_k" + std::to_string(kd) + "x" + std::to_string(kh) + "x"
               + std::to_string(kw) + "_q" + std::to_string(q) + "g" + std::to_string(g) + "w"
               + std::to_string(nwg) + "b" + std::to_string(bpc) + "m" + std::to_string(mult) + "s"
               + std::to_string(s) + "v" + std::to_string(cv) + "x" + std::to_string(strip);
    }
};

/// The matrix-core (hankel) schedules (q, g, nwg, bpc) the generator ships for 3x3, 5x5, 7x7.
const std::vector<std::array<int64_t, 4>>& hankelSchedules()
{
    static const std::vector<std::array<int64_t, 4>> s_schedules{{32, 1, 1, 2},
                                                                 {32, 2, 1, 2},
                                                                 {32, 1, 1, 4},
                                                                 {32, 2, 1, 4},
                                                                 {32, 1, 2, 2},
                                                                 {16, 2, 1, 2},
                                                                 {16, 2, 2, 1},
                                                                 {16, 1, 2, 2},
                                                                 {16, 4, 1, 2},
                                                                 {16, 4, 1, 1},
                                                                 {32, 4, 1, 1}};
    return s_schedules;
}

const std::vector<WrwConfig>& wrwCatalog()
{
    static const std::vector<WrwConfig> s_configs = [] {
        std::vector<WrwConfig> configs;
        for(const auto* layout : {"NHWC", "NCHW"})
        {
            configs.push_back({layout, "floor", 0, 0, 0});
            for(const auto& [kd, kh, kw] : std::vector<std::array<int64_t, 3>>{
                    {1, 3, 3}, {1, 5, 5}, {1, 7, 7}, {1, 1, 4}, {1, 1, 31}, {3, 3, 3}})
            {
                configs.push_back({layout, "tile", kd, kh, kw});
            }
            // Depth multiplier 2: the floor and the 2-D tiles.
            configs.push_back({layout, "floor", 0, 0, 0, 0, 0, 0, 0, 2});
            for(const int64_t k : {3, 5, 7})
            {
                configs.push_back({layout, "tile", 1, k, k, 0, 0, 0, 0, 2});
            }
        }
        // Vector tiles: (filter, cv, strip).
        for(const auto& [k, cv, strip] : std::vector<std::array<int64_t, 3>>{{3, 4, 8},
                                                                             {3, 8, 4},
                                                                             {3, 2, 8},
                                                                             {5, 4, 4},
                                                                             {5, 2, 8},
                                                                             {7, 2, 8},
                                                                             {7, 2, 4},
                                                                             {7, 1, 8}})
        {
            WrwConfig config{"NHWC", "vtile", 1, k, k};
            config.cv = cv;
            config.strip = strip;
            configs.push_back(config);
        }
        for(const int64_t stride : {1, 2})
        {
            for(const int64_t k : {3, 5, 7})
            {
                for(const auto& [q, g, nwg, bpc] : hankelSchedules())
                {
                    configs.push_back({"NHWC", "hankel", 1, k, k, q, g, nwg, bpc, 1, stride});
                }
            }
        }
        return configs;
    }();
    return s_configs;
}

PackSymbols wrwPack()
{
    return {"",
            "hipkernel.conv_depthwise_valu_wrw.graph_match",
            "",
            "hipkernel.conv_depthwise_valu_wrw.kernel_match",
            "",
            "hipkernel.conv_depthwise_valu_wrw.dispatch",
            "",
            "",
            ""};
}

hipdnn_plugin_sdk::ingestor::KernelDefinition wrwKernelFor(const WrwConfig& config,
                                                           DataType dataType)
{
    hipdnn_plugin_sdk::ingestor::KernelDefinition kernel;
    kernel.name = "conv_depthwise_valu_wrw_test." + config.tag();
    kernel.source.kind = hipdnn_plugin_sdk::ingestor::KernelSourceKind::EMBEDDED_SOURCE;
    kernel.source.sourceFile = "kernels/ConvDepthwiseValuWrw.cpp";
    kernel.source.entryPoint = "ConvDepthwiseValuWrwPartial";
    kernel.metadata
        = {{"dtype", std::string(hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(dataType))},
           {"layout", config.layout},
           {"variant", config.variant},
           {"kd", config.kd},
           {"kh", config.kh},
           {"kw", config.kw},
           {"q", config.q},
           {"g", config.g},
           {"nwg", config.nwg},
           {"bpc", config.bpc},
           {"mult", config.mult},
           {"s", config.s},
           {"cv", config.cv},
           {"strip", config.strip}};
    return kernel;
}

/// Depth multiplier 2 (two output channels per input channel; MobileNet-style expansion and
/// separable layers): the backward-weights VALU variants. Odd channel counts, a strided and a
/// dilated floor case, 1-D and 3-D.
std::vector<DeviceCase> multiplierCases()
{
    const auto times2 = [](ConvShape shape) {
        shape.k = 2 * shape.c;
        return shape;
    };
    std::vector<DeviceCase> cases;
    cases.push_back({"Mult2Nhwc3x3", times2(depthwise(37, {13, 11}, {3, 3}, {1, 1}, true))});
    cases.push_back({"Mult2Nchw3x3", times2(depthwise(9, {13, 11}, {3, 3}, {1, 1}, false))});
    cases.push_back({"Mult2Nhwc5x5", times2(depthwise(21, {19, 17}, {5, 5}, {2, 2}, true))});
    cases.push_back({"Mult2Nchw7x7", times2(depthwise(10, {18, 21}, {7, 7}, {3, 3}, false))});
    cases.push_back({"Mult2Nhwc7x7", times2(depthwise(20, {18, 21}, {7, 7}, {3, 3}, true))});
    cases.push_back({"Mult2Nchw5x5", times2(depthwise(11, {19, 37}, {5, 5}, {2, 2}, false))});
    auto strided = times2(depthwise(19, {15, 14}, {3, 3}, {1, 1}, true));
    strided.stride = {2, 2};
    cases.push_back({"Mult2NhwcStride2", strided});
    auto dilated = times2(depthwise(6, {12, 13}, {3, 3}, {2, 2}, false));
    dilated.dilation = {2, 2};
    cases.push_back({"Mult2NchwDilation2", dilated});
    cases.push_back({"Mult2Nhwc1d", times2(depthwise(13, {23}, {5}, {2}, true))});
    cases.push_back(
        {"Mult2Ndhwc3x3x3", times2(depthwise(11, {5, 9, 10}, {3, 3, 3}, {1, 1, 1}, true))});
    return cases;
}

/// Every config that admits @p shape runs it, from a workspace the handler sizes, against the
/// CPU reference (reflected for true convolution).
template <typename T>
void runWrwCase(ConvShape shape)
{
    shape.dataType = dataTypeOf<T>();
    const auto spec = specFor(shape, ConvDirection::BACKWARD_WEIGHTS);
    const GraphFixture fixture(buildGraph(spec), currentDeviceProperties());
    const auto pack = wrwPack();
    const auto bound = matchesGraph(pack, fixture.context());
    ASSERT_TRUE(bound.has_value()) << "the engine refused a case it should serve";

    Tensor<T> x(spec.x.dims, spec.x.strides);
    Tensor<T> w(spec.w.dims, spec.w.strides);
    Tensor<T> y(spec.y.dims, spec.y.strides);
    constexpr unsigned int SEED = 4321;
    x.fillTensorWithRandomValues(-1.0f, 1.0f, SEED);
    y.fillTensorWithRandomValues(-1.0f, 1.0f, SEED + 1);

    Tensor<T> reference(spec.w.dims, spec.w.strides);
    const bool reflect = shape.mode == ConvMode::CONVOLUTION;
    Tensor<T> crossCorrelation(spec.w.dims, spec.w.strides);
    CpuFpReferenceConvolution::wgrad(x,
                                     reflect ? crossCorrelation : reference,
                                     y,
                                     shape.stride,
                                     shape.dilation,
                                     shape.prePadding,
                                     shape.postPadding);
    if(reflect)
    {
        // True convolution's dw is cross-correlation's reflected.
        reflectFilter(crossCorrelation, reference);
    }
    int64_t reduction = spec.y.dims[0];
    for(size_t axis = 2; axis < spec.y.dims.size(); ++axis)
    {
        reduction *= spec.y.dims[axis];
    }
    const float accumulationBound = 1e-6f * static_cast<float>(reduction);

    const auto& handler = dispatchHandler(pack);
    int ran = 0;
    for(const auto& config : wrwCatalog())
    {
        const auto kernel = wrwKernelFor(config, shape.dataType);
        if(!matchesKernel(pack, fixture.context(), kernel, *bound))
        {
            continue;
        }
        SCOPED_TRACE(config.tag());
        ++ran;
        const auto prepared = handler.prepare(fixture.context(), *bound, kernel);
        ASSERT_NE(prepared, nullptr);
        const size_t workspaceBytes = handler.workspaceBytes(fixture.context(), *bound, kernel);
        ASSERT_GT(workspaceBytes, 0u);
        void* workspace = nullptr;
        ASSERT_EQ(hipMalloc(&workspace, workspaceBytes), hipSuccess);

        w.fillWithSentinelValue();
        const std::array<hipdnnPluginDeviceBuffer_t, 3> buffers{
            hipdnnPluginDeviceBuffer_t{X_UID, x.rawDeviceData()},
            hipdnnPluginDeviceBuffer_t{W_UID, w.rawDeviceData()},
            hipdnnPluginDeviceBuffer_t{Y_UID, y.rawDeviceData()}};
        const Handle handle;
        handler.launch(
            handle, *prepared, buffers.data(), static_cast<uint32_t>(buffers.size()), workspace);
        ASSERT_EQ(hipStreamSynchronize(handle.getStream()), hipSuccess);
        ASSERT_EQ(hipFree(workspace), hipSuccess);
        w.markDeviceModified();

        const auto* actual = static_cast<const T*>(w.rawHostData());
        const auto* expected = static_cast<const T*>(reference.rawHostData());
        int64_t mismatches = 0;
        forEachIndex(w.dims(), [&](const std::vector<int64_t>& index) {
            const auto got = static_cast<float>(actual[offsetOf(index, w.strides())]);
            const auto want = static_cast<float>(expected[offsetOf(index, reference.strides())]);
            const float tolerance
                = accumulationBound + 2.0f * storageEpsilon<T>() * (std::fabs(want) + 1e-3f);
            if(!(std::fabs(got - want) <= tolerance) && ++mismatches <= 5)
            {
                ADD_FAILURE() << "at linear index " << offsetOf(index, w.strides()) << ": got "
                              << got << ", want " << want << ", tolerance " << tolerance;
            }
        });
        EXPECT_EQ(mismatches, 0);
    }
    EXPECT_GE(ran, 1) << "no catalog config serves this case";
}

class TestGpuConvDepthwiseValuWrw : public ::testing::TestWithParam<DeviceCase>
{
};

TEST_P(TestGpuConvDepthwiseValuWrw, Float)
{
    SKIP_IF_NO_DEVICES();
    runWrwCase<float>(GetParam().shape);
}

TEST_P(TestGpuConvDepthwiseValuWrw, Half)
{
    SKIP_IF_NO_DEVICES();
    runWrwCase<Half>(GetParam().shape);
}

TEST_P(TestGpuConvDepthwiseValuWrw, BFloat16)
{
    SKIP_IF_NO_DEVICES();
    runWrwCase<BFloat16>(GetParam().shape);
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuConvDepthwiseValuWrw,
                         ::testing::ValuesIn(deviceCases()),
                         [](const auto& info) { return info.param.name; });

INSTANTIATE_TEST_SUITE_P(Multiplier,
                         TestGpuConvDepthwiseValuWrw,
                         ::testing::ValuesIn(multiplierCases()),
                         [](const auto& info) { return info.param.name; });

/// An inf in x that some filter taps never pair with a dy element: a column past wo of the last
/// tile, an image's first row under its bottom filter row, or (stride 2) a row or column of the
/// other parity. Every matrix-core (hankel) config keeps those taps finite, as the reference does.
struct InfCase
{
    std::string name;
    ConvShape shape;
    std::vector<int64_t> at; // (n, c, y, x)
};

std::vector<InfCase> infCases()
{
    std::vector<InfCase> cases;
    // wo 21: the last 32- or 16-column tile runs past it; x column 20 under tap 0 meets ox 21.
    cases.push_back(
        {"Stride1RaggedColumn", depthwise(16, {9, 21}, {3, 3}, {1, 1}, true), {1, 3, 4, 20}});
    // x row 0 under filter row 2 meets dy row -1.
    cases.push_back(
        {"Stride1FirstRow", depthwise(16, {9, 21}, {3, 3}, {1, 1}, true), {0, 5, 0, 10}});
    // wo 32 (whole tiles), 7x7: the first row under filter rows 4 to 6.
    cases.push_back(
        {"Stride1FirstRow7x7", depthwise(16, {9, 32}, {7, 7}, {3, 3}, true), {1, 9, 0, 13}});
    auto odd = depthwise(16, {13, 21}, {3, 3}, {1, 1}, true);
    odd.stride = {2, 2};
    // Stride 2: x row 1 under filter row 1 meets dy row 1/2.
    cases.push_back({"Stride2OddRow", odd, {1, 7, 1, 10}});
    auto ragged = depthwise(16, {13, 22}, {3, 3}, {1, 1}, true);
    ragged.stride = {2, 2};
    // Stride 2, wo 11: x column 21 under tap 0 meets ox 11.
    cases.push_back({"Stride2RaggedColumn", ragged, {0, 2, 6, 21}});
    return cases;
}

template <typename T>
void runWrwInfCase(const InfCase& infCase)
{
    auto shape = infCase.shape;
    shape.dataType = dataTypeOf<T>();
    const auto spec = specFor(shape, ConvDirection::BACKWARD_WEIGHTS);
    const GraphFixture fixture(buildGraph(spec), currentDeviceProperties());
    const auto pack = wrwPack();
    const auto bound = matchesGraph(pack, fixture.context());
    ASSERT_TRUE(bound.has_value());

    Tensor<T> x(spec.x.dims, spec.x.strides);
    Tensor<T> w(spec.w.dims, spec.w.strides);
    Tensor<T> y(spec.y.dims, spec.y.strides);
    constexpr unsigned int SEED = 2468;
    x.fillTensorWithRandomValues(-1.0f, 1.0f, SEED);
    y.fillTensorWithRandomValues(-1.0f, 1.0f, SEED + 1);
    static_cast<T*>(x.rawHostData())[offsetOf(infCase.at, x.strides())]
        = static_cast<T>(std::numeric_limits<float>::infinity());
    x.markHostModified();
    Tensor<T> reference(spec.w.dims, spec.w.strides);
    CpuFpReferenceConvolution::wgrad(
        x, reference, y, shape.stride, shape.dilation, shape.prePadding, shape.postPadding);
    const auto* expected = static_cast<const T*>(reference.rawHostData());
    int64_t finiteTaps = 0;
    forEachIndex(w.dims(), [&](const std::vector<int64_t>& index) {
        if(std::isfinite(static_cast<float>(expected[offsetOf(index, reference.strides())])))
        {
            ++finiteTaps;
        }
    });
    // The case is only a test when the inf's channel has both finite and non-finite taps.
    ASSERT_GT(finiteTaps, 0);
    ASSERT_LT(finiteTaps, static_cast<int64_t>(w.elementCount()));

    int64_t reduction = spec.y.dims[0];
    for(size_t axis = 2; axis < spec.y.dims.size(); ++axis)
    {
        reduction *= spec.y.dims[axis];
    }
    const float accumulationBound = 1e-6f * static_cast<float>(reduction);
    const auto& handler = dispatchHandler(pack);
    int ran = 0;
    for(const auto& config : wrwCatalog())
    {
        const auto kernel = wrwKernelFor(config, shape.dataType);
        if(config.variant != "hankel" || !matchesKernel(pack, fixture.context(), kernel, *bound))
        {
            continue;
        }
        SCOPED_TRACE(config.tag());
        ++ran;
        const auto prepared = handler.prepare(fixture.context(), *bound, kernel);
        const size_t workspaceBytes = handler.workspaceBytes(fixture.context(), *bound, kernel);
        void* workspace = nullptr;
        ASSERT_EQ(hipMalloc(&workspace, workspaceBytes), hipSuccess);
        w.fillWithSentinelValue();
        const std::array<hipdnnPluginDeviceBuffer_t, 3> buffers{
            hipdnnPluginDeviceBuffer_t{X_UID, x.rawDeviceData()},
            hipdnnPluginDeviceBuffer_t{W_UID, w.rawDeviceData()},
            hipdnnPluginDeviceBuffer_t{Y_UID, y.rawDeviceData()}};
        const Handle handle;
        handler.launch(
            handle, *prepared, buffers.data(), static_cast<uint32_t>(buffers.size()), workspace);
        ASSERT_EQ(hipStreamSynchronize(handle.getStream()), hipSuccess);
        ASSERT_EQ(hipFree(workspace), hipSuccess);
        w.markDeviceModified();
        const auto* actual = static_cast<const T*>(w.rawHostData());
        int64_t mismatches = 0;
        forEachIndex(w.dims(), [&](const std::vector<int64_t>& index) {
            const auto got = static_cast<float>(actual[offsetOf(index, w.strides())]);
            const auto want = static_cast<float>(expected[offsetOf(index, reference.strides())]);
            const bool ok = std::isfinite(want)
                                ? std::fabs(got - want) <= accumulationBound
                                                               + 2.0f * storageEpsilon<T>()
                                                                     * (std::fabs(want) + 1e-3f)
                                : !std::isfinite(got);
            if(!ok && ++mismatches <= 5)
            {
                ADD_FAILURE() << "at linear index " << offsetOf(index, w.strides()) << ": got "
                              << got << ", want " << want;
            }
        });
        EXPECT_EQ(mismatches, 0);
    }
    if(ran == 0)
    {
        GTEST_SKIP() << "no matrix-core config serves this case on this device";
    }
}

class TestGpuConvDepthwiseValuWrwHankelInf : public ::testing::TestWithParam<InfCase>
{
};

TEST_P(TestGpuConvDepthwiseValuWrwHankelInf, Half)
{
    SKIP_IF_NO_DEVICES();
    runWrwInfCase<Half>(GetParam());
}

TEST_P(TestGpuConvDepthwiseValuWrwHankelInf, BFloat16)
{
    SKIP_IF_NO_DEVICES();
    runWrwInfCase<BFloat16>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuConvDepthwiseValuWrwHankelInf,
                         ::testing::ValuesIn(infCases()),
                         [](const auto& info) { return info.param.name; });

// Host-only: the kernel's multiplier is the graph's; the graph's output channels a whole
// multiple of its input channels; no matrix-core (hankel) config takes a multiplier.
TEST(TestConvDepthwiseValuWrwMultiplier, KernelsServeTheirOwnMultiplier)
{
    const auto pack = wrwPack();
    const auto admits = [&](ConvShape shape, const WrwConfig& config, const std::string& arch) {
        shape.dataType = DataType::HALF;
        const GraphFixture fixture(buildGraph(specFor(shape, ConvDirection::BACKWARD_WEIGHTS)),
                                   countedDevice(64, arch));
        const auto bound = matchesGraph(pack, fixture.context());
        return bound.has_value()
               && matchesKernel(
                   pack, fixture.context(), wrwKernelFor(config, DataType::HALF), *bound);
    };
    auto one = depthwise(40, {13, 21}, {3, 3}, {1, 1}, true);
    auto two = one;
    two.k = 80;
    const WrwConfig floor1{"NHWC", "floor", 0, 0, 0};
    const WrwConfig floor2{"NHWC", "floor", 0, 0, 0, 0, 0, 0, 0, 2};
    EXPECT_TRUE(admits(one, floor1, "gfx942"));
    EXPECT_FALSE(admits(one, floor2, "gfx942"));
    EXPECT_TRUE(admits(two, floor2, "gfx942"));
    EXPECT_FALSE(admits(two, floor1, "gfx942"));
    // 60 output channels from 40: no whole multiplier, no graph.
    auto ragged = one;
    ragged.k = 60;
    ragged.dataType = DataType::HALF;
    const GraphFixture fixture(buildGraph(specFor(ragged, ConvDirection::BACKWARD_WEIGHTS)),
                               countedDevice(64, "gfx942"));
    EXPECT_FALSE(matchesGraph(pack, fixture.context()).has_value());
    // The matrix-core kernel walks input channels: never with a multiplier.
    const WrwConfig hankel1{"NHWC", "hankel", 1, 3, 3, 32, 1, 1, 2, 1, 1};
    WrwConfig hankel2 = hankel1;
    hankel2.mult = 2;
    EXPECT_TRUE(admits(one, hankel1, "gfx942"));
    EXPECT_FALSE(admits(two, hankel2, "gfx942"));
    // Output channels times taps within 32 bits: 2^28 output channels from 2^18 (3x3).
    auto wide = depthwise(int64_t{1} << 18, {4, 4}, {3, 3}, {1, 1}, true);
    wide.k = int64_t{1} << 28;
    wide.n = 1;
    wide.dataType = DataType::HALF;
    const GraphFixture wideFixture(buildGraph(specFor(wide, ConvDirection::BACKWARD_WEIGHTS)),
                                   countedDevice(64, "gfx942"));
    EXPECT_FALSE(matchesGraph(pack, wideFixture.context()).has_value());
    wide.k = int64_t{1} << 26;
    const GraphFixture fitFixture(buildGraph(specFor(wide, ConvDirection::BACKWARD_WEIGHTS)),
                                  countedDevice(64, "gfx942"));
    EXPECT_TRUE(matchesGraph(pack, fitFixture.context()).has_value());
}

// Host-only: the partitions and the partial sums run over output channels, so 40 channels at
// multiplier 2 take the workspace of 80 plain depthwise channels.
TEST(TestConvDepthwiseValuWrwMultiplier, TheWorkspaceCountsOutputChannels)
{
    const auto pack = wrwPack();
    const auto workspaceOf = [&](ConvShape shape, const WrwConfig& config) {
        shape.dataType = DataType::HALF;
        const GraphFixture fixture(buildGraph(specFor(shape, ConvDirection::BACKWARD_WEIGHTS)),
                                   countedDevice(64, "gfx942"));
        const auto bound = matchesGraph(pack, fixture.context());
        EXPECT_TRUE(bound.has_value());
        const auto kernel = wrwKernelFor(config, DataType::HALF);
        EXPECT_TRUE(matchesKernel(pack, fixture.context(), kernel, *bound));
        return dispatchHandler(pack).workspaceBytes(fixture.context(), *bound, kernel);
    };
    for(const bool channelsLast : {true, false})
    {
        const std::string layout = channelsLast ? "NHWC" : "NCHW";
        auto two = depthwise(40, {13, 21}, {3, 3}, {1, 1}, channelsLast);
        two.k = 80;
        const auto eighty = depthwise(80, {13, 21}, {3, 3}, {1, 1}, channelsLast);
        const auto withTwo = workspaceOf(two, WrwConfig{layout, "floor", 0, 0, 0, 0, 0, 0, 0, 2});
        const auto withEighty = workspaceOf(eighty, WrwConfig{layout, "floor", 0, 0, 0});
        EXPECT_GT(withTwo, 0u) << layout;
        EXPECT_EQ(withTwo, withEighty) << layout;
    }
}

// Host-only: every backward-weights config serves some device case.
TEST(TestConvDepthwiseValuCatalog, EveryWrwConfigServesADeviceCase)
{
    const auto pack = wrwPack();
    auto allCases = deviceCases();
    for(const auto& deviceCase : multiplierCases())
    {
        allCases.push_back(deviceCase);
    }
    for(const auto& config : wrwCatalog())
    {
        bool served = false;
        // The matrix-core configs run on CDNA; gfx950 has the LDS every one of them fits.
        const std::string arch = config.variant == "hankel" ? "gfx950" : "gfx000";
        for(const auto& deviceCase : allCases)
        {
            auto shape = deviceCase.shape;
            shape.dataType = DataType::HALF;
            const GraphFixture fixture(buildGraph(specFor(shape, ConvDirection::BACKWARD_WEIGHTS)),
                                       countedDevice(64, arch));
            const auto bound = matchesGraph(pack, fixture.context());
            if(bound.has_value()
               && matchesKernel(
                   pack, fixture.context(), wrwKernelFor(config, DataType::HALF), *bound))
            {
                served = true;
                break;
            }
        }
        EXPECT_TRUE(served) << config.tag() << " serves no device case";
    }
}

// Host-only: the partitions are sized from the device's units, so a device that reports none is
// declined (not divided by), while the forward and backward-data configs, which do not read the
// count, serve it unchanged.
TEST(TestConvDepthwiseValuCatalog, AnUncountedDeviceDeclinesOnlyTheUnitSizedWrw)
{
    auto shape = depthwise(37, {13, 11}, {3, 3}, {1, 1}, true);
    shape.dataType = DataType::HALF;
    const auto wrw = wrwPack();
    const auto floor = wrwKernelFor(WrwConfig{"NHWC", "floor", 0, 0, 0}, DataType::HALF);
    for(const int units : {64, 0, -1})
    {
        const GraphFixture fixture(buildGraph(specFor(shape, ConvDirection::BACKWARD_WEIGHTS)),
                                   countedDevice(units));
        const auto bound = matchesGraph(wrw, fixture.context());
        ASSERT_TRUE(bound.has_value());
        EXPECT_EQ(matchesKernel(wrw, fixture.context(), floor, *bound), units > 0) << units;
    }
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        const auto pack = packFor(direction);
        const GraphFixture counted(buildGraph(specFor(shape, direction)), countedDevice());
        const GraphFixture uncounted(buildGraph(specFor(shape, direction)), countedDevice(0));
        const auto bound = matchesGraph(pack, counted.context());
        ASSERT_TRUE(bound.has_value());
        for(const auto& config : catalog())
        {
            const auto kernel = kernelFor(config, DataType::HALF, direction);
            EXPECT_EQ(matchesKernel(pack, counted.context(), kernel, *bound),
                      matchesKernel(pack, uncounted.context(), kernel, *bound))
                << config.tag();
        }
    }
}

// Host-only: what a phase-block config admits, one refusal per field it gates.
bool phaseAdmits(const Config& config,
                 ConvShape shape,
                 ConvDirection direction = ConvDirection::BACKWARD_DATA,
                 DataType dataType = DataType::HALF)
{
    shape.dataType = dataType;
    const auto pack = packFor(direction);
    const GraphFixture fixture(buildGraph(specFor(shape, direction)));
    const auto bound = matchesGraph(pack, fixture.context());
    return bound.has_value()
           && matchesKernel(
               pack, fixture.context(), kernelFor(config, dataType, direction), *bound);
}

ConvShape phaseShape(bool channelsLast)
{
    auto shape = depthwise(40, {17, 14}, {3, 3}, {1, 1}, channelsLast);
    shape.stride = {2, 2};
    return shape;
}

TEST(TestConvDepthwiseValuPhase, ServesStridedBackwardDataOfItsFilterAndStride)
{
    EXPECT_TRUE(phaseAdmits(phaseConfig("NHWC", 3, 16), phaseShape(true)));
    EXPECT_TRUE(phaseAdmits(phaseConfig("NCHW", 3, 0), phaseShape(false)));
    EXPECT_TRUE(phaseAdmits(phaseConfig("NHWC", 3, 16),
                            phaseShape(true),
                            ConvDirection::BACKWARD_DATA,
                            DataType::FLOAT));
}

TEST(TestConvDepthwiseValuPhase, RefusesOneFieldAtATime)
{
    const auto base = phaseConfig("NHWC", 3, 16);
    ASSERT_TRUE(phaseAdmits(base, phaseShape(true)));
    // Forward.
    EXPECT_FALSE(phaseAdmits(base, phaseShape(true), ConvDirection::FORWARD));
    // Stride: another graph stride, or one axis only.
    auto stride3 = phaseShape(true);
    stride3.stride = {3, 3};
    EXPECT_FALSE(phaseAdmits(base, stride3));
    auto mixed = phaseShape(true);
    mixed.stride = {2, 1};
    EXPECT_FALSE(phaseAdmits(base, mixed));
    auto unit = phaseShape(true);
    unit.stride = {1, 1};
    EXPECT_FALSE(phaseAdmits(base, unit));
    // Filter.
    auto k5 = phaseShape(true);
    k5.filter = {5, 5};
    k5.prePadding = k5.postPadding = {2, 2};
    EXPECT_FALSE(phaseAdmits(base, k5));
    // Dilation.
    auto dilated = phaseShape(true);
    dilated.dilation = {2, 2};
    dilated.prePadding = dilated.postPadding = {2, 2};
    EXPECT_FALSE(phaseAdmits(base, dilated));
    // Rank.
    auto threeD = depthwise(8, {5, 9, 10}, {3, 3, 3}, {1, 1, 1}, true);
    threeD.stride = {2, 2, 2};
    EXPECT_FALSE(phaseAdmits(base, threeD));
    // Vector: past 16 bytes, not whole words, absent on channels-last, set on channels-first.
    auto wide = base;
    wide.vec = 16;
    EXPECT_FALSE(phaseAdmits(wide, phaseShape(true)));
    auto odd = base;
    odd.vec = 3;
    EXPECT_FALSE(phaseAdmits(odd, phaseShape(true)));
    auto none = base;
    none.vec = 0;
    EXPECT_FALSE(phaseAdmits(none, phaseShape(true)));
    auto nchwVec = phaseConfig("NCHW", 3, 0);
    nchwVec.vec = 8;
    EXPECT_FALSE(phaseAdmits(nchwVec, phaseShape(false)));
    auto nchwBk = phaseConfig("NCHW", 3, 0);
    nchwBk.bk = 8;
    EXPECT_FALSE(phaseAdmits(nchwBk, phaseShape(false)));
    // Channel vectors per block: positive, dividing the 256-thread block.
    auto bk24 = base;
    bk24.bk = 24;
    EXPECT_FALSE(phaseAdmits(bk24, phaseShape(true)));
    auto bk0 = base;
    bk0.bk = 0;
    EXPECT_FALSE(phaseAdmits(bk0, phaseShape(true)));
    // Layout.
    EXPECT_FALSE(phaseAdmits(base, phaseShape(false)));
}

// Host-only: every config of the catalog is admitted by some device case, so the device suite
// runs every config (forward; a halo config's backward data runs on the same unit-stride case).
TEST(TestConvDepthwiseValuCatalog, EveryConfigServesADeviceCase)
{
    for(const auto& config : catalog())
    {
        // The gather and the phase blocks are backward-data configs; the rest serve forward too.
        const auto direction = config.variant == "gather" || config.variant == "phase"
                                   ? ConvDirection::BACKWARD_DATA
                                   : ConvDirection::FORWARD;
        const auto pack = packFor(direction);
        bool served = false;
        for(const auto& deviceCase : deviceCases())
        {
            auto shape = deviceCase.shape;
            shape.dataType = DataType::HALF;
            const GraphFixture fixture(buildGraph(specFor(shape, direction)));
            const auto bound = matchesGraph(pack, fixture.context());
            if(bound.has_value()
               && matchesKernel(
                   pack, fixture.context(), kernelFor(config, DataType::HALF, direction), *bound))
            {
                served = true;
                break;
            }
        }
        EXPECT_TRUE(served) << config.tag() << " serves no device case";
    }
}

// Host-only: what a forward rowtile config admits, one refusal per field it gates.
bool rowtileAdmits(const Config& config, ConvShape shape, DataType dataType = DataType::HALF)
{
    shape.dataType = dataType;
    const auto pack = packFor(ConvDirection::FORWARD);
    const GraphFixture fixture(buildGraph(specFor(shape, ConvDirection::FORWARD)));
    const auto bound = matchesGraph(pack, fixture.context());
    return bound.has_value()
           && matchesKernel(pack,
                            fixture.context(),
                            kernelFor(config, dataType, ConvDirection::FORWARD),
                            *bound);
}

TEST(TestConvDepthwiseValuRowtile, RefusesOneFieldAtATime)
{
    const auto base = rowtileConfig(3, 4, 4, 2);
    const auto shape = depthwise(40, {13, 11}, {3, 3}, {1, 1}, true);
    ASSERT_TRUE(rowtileAdmits(base, shape));
    ASSERT_TRUE(rowtileAdmits(base, shape, DataType::FLOAT));
    ASSERT_TRUE(rowtileAdmits(base, shape, DataType::BFLOAT16));
    // Layout and rank.
    EXPECT_FALSE(rowtileAdmits(base, depthwise(40, {13, 11}, {3, 3}, {1, 1}, false)));
    auto threeD = base;
    threeD.ndims = 3;
    EXPECT_FALSE(rowtileAdmits(threeD, depthwise(40, {5, 13, 11}, {3, 3, 3}, {1, 1, 1}, true)));
    EXPECT_FALSE(rowtileAdmits(base, depthwise(40, {5, 13, 11}, {3, 3, 3}, {1, 1, 1}, true)));
    // Stride and dilation, either axis; a stride-2 tile only stride 2 on both axes.
    auto strided = shape;
    strided.stride = {1, 2};
    EXPECT_FALSE(rowtileAdmits(base, strided));
    const auto base2 = rowtileConfig(3, 4, 4, 2, 2);
    auto stride2 = shape;
    stride2.stride = {2, 2};
    EXPECT_TRUE(rowtileAdmits(base2, stride2));
    EXPECT_FALSE(rowtileAdmits(base, stride2));
    EXPECT_FALSE(rowtileAdmits(base2, shape));
    EXPECT_FALSE(rowtileAdmits(base2, strided));
    auto stride21 = shape;
    stride21.stride = {2, 1};
    EXPECT_FALSE(rowtileAdmits(base2, stride21));
    for(const int64_t stride : {0, -1})
    {
        auto other = base;
        other.stride = stride;
        EXPECT_FALSE(rowtileAdmits(other, shape)) << stride;
    }
    auto dilated = shape;
    dilated.dilation = {2, 1};
    dilated.prePadding = dilated.postPadding = {2, 1};
    EXPECT_FALSE(rowtileAdmits(base, dilated));
    // Filter height and width.
    EXPECT_FALSE(rowtileAdmits(base, depthwise(40, {13, 11}, {5, 3}, {2, 1}, true)));
    EXPECT_FALSE(rowtileAdmits(base, depthwise(40, {13, 11}, {3, 5}, {1, 2}, true)));
    // Channels per lane: 1 or 2, dividing the channels.
    EXPECT_FALSE(rowtileAdmits(base, depthwise(37, {13, 11}, {3, 3}, {1, 1}, true)));
    EXPECT_TRUE(
        rowtileAdmits(rowtileConfig(3, 4, 4, 1), depthwise(37, {13, 11}, {3, 3}, {1, 1}, true)));
    for(const int64_t vec : {0, 3, 4})
    {
        auto other = base;
        other.vec = vec;
        EXPECT_FALSE(rowtileAdmits(other, shape)) << vec;
    }
    // A positive output tile.
    auto noHeight = base;
    noHeight.th = 0;
    EXPECT_FALSE(rowtileAdmits(noHeight, shape));
    auto noWidth = base;
    noWidth.tw = 0;
    EXPECT_FALSE(rowtileAdmits(noWidth, shape));
    // Every field the variant does not use is 0.
    for(auto field : {&Config::kd,
                      &Config::wstrip,
                      &Config::bd,
                      &Config::bh,
                      &Config::bw,
                      &Config::bk,
                      &Config::rd,
                      &Config::rh,
                      &Config::rw})
    {
        auto other = base;
        other.*field = 2;
        EXPECT_FALSE(rowtileAdmits(other, shape)) << other.tag();
    }
    // x and y rows below 2^31 bytes (fp16: 2^30 elements), at the limit and one past it.
    const int64_t rowElements = (int64_t{1} << 30) - 1;
    const int64_t c = 64;
    const int64_t widest = (rowElements - 1) / c;
    EXPECT_TRUE(rowtileAdmits(base, depthwise(c, {1, widest}, {3, 3}, {1, 1}, true)));
    EXPECT_FALSE(rowtileAdmits(base, depthwise(c, {1, widest + 1}, {3, 3}, {1, 1}, true)));
    // The grid's channel-block and N x H-tile extents below 2^16.
    auto rows = depthwise(40, {int64_t{4} * 32767, 3}, {3, 3}, {1, 1}, true);
    EXPECT_TRUE(rowtileAdmits(base, rows));
    rows.input = {int64_t{4} * 32768, 3};
    EXPECT_FALSE(rowtileAdmits(base, rows));
    const auto one = rowtileConfig(3, 4, 4, 1);
    EXPECT_TRUE(rowtileAdmits(one, depthwise(int64_t{64} * 65535, {3, 3}, {3, 3}, {1, 1}, true)));
    EXPECT_FALSE(rowtileAdmits(one, depthwise(int64_t{64} * 65536, {3, 3}, {3, 3}, {1, 1}, true)));
}

// The rowtile cores keep f16 operands packed and let the FMA widen them (v_fma_mix_f32). This
// must compute exactly what converting both operands to fp32 and calling fmaf computes: the f16
// output of every rowtile config, bit for bit, against that sequence on the host (taps in (ky, kx)
// order, one fp32 rounding per tap, a padding tap adding w * 0, one rounding to f16 at the end).
// sampleStride 1 checks every output; a larger one checks every sampleStride-th output (in
// logical (n, c, h, w) order) and every output of the last image row.
void runRowtileBitwise(ConvShape shape,
                       const std::vector<Config>& configs = rowtileConfigs(),
                       int64_t sampleStride = 1)
{
    shape.dataType = DataType::HALF;
    const auto direction = ConvDirection::FORWARD;
    const auto spec = specFor(shape, direction);
    const GraphFixture fixture(buildGraph(spec), currentDeviceProperties());
    const auto pack = packFor(direction);
    const auto bound = matchesGraph(pack, fixture.context());
    ASSERT_TRUE(bound.has_value());

    Tensor<Half> x(spec.x.dims, spec.x.strides);
    Tensor<Half> w(spec.w.dims, spec.w.strides);
    Tensor<Half> y(spec.y.dims, spec.y.strides);
    if(sampleStride == 1)
    {
        x.fillTensorWithRandomValues(-1.0f, 1.0f, 97);
        w.fillTensorWithRandomValues(-1.0f, 1.0f, 98);
    }
    else
    {
        // Hashed f16 bit patterns of magnitude below 1 (biased exponent 0-14, either sign): as
        // varied as random values, and quick for 2^30 elements.
        for(auto* tensor : {&x, &w})
        {
            auto* data = static_cast<Half*>(tensor->rawHostData());
            const auto count = tensor->elementCount();
            const uint32_t salt = tensor == &x ? 0x9e3779b9u : 0x85ebca6bu;
            for(size_t i = 0; i < count; ++i)
            {
                const uint32_t h = (static_cast<uint32_t>(i) ^ salt) * 0x27d4eb2fu;
                data[i].data = static_cast<uint16_t>(((h >> 16) & 0x8000u)
                                                     | (((h >> 11) % 15u) << 10) | (h & 0x3ffu));
            }
            tensor->markHostModified();
        }
    }
    const auto* xs = static_cast<const Half*>(x.rawHostData());
    const auto* ws = static_cast<const Half*>(w.rawHostData());
    const int64_t kh = shape.filter[0];
    const int64_t kw = shape.filter[1];
    const bool flip = shape.mode == ConvMode::CONVOLUTION;
    const auto expected = [&](const std::vector<int64_t>& o) {
        float acc = 0.0f;
        for(int64_t ky = 0; ky < kh; ++ky)
        {
            for(int64_t kx = 0; kx < kw; ++kx)
            {
                const int64_t ih = o[2] * shape.stride[0] - shape.prePadding[0] + ky;
                const int64_t iw = o[3] * shape.stride[1] - shape.prePadding[1] + kx;
                const bool inside
                    = ih >= 0 && ih < shape.input[0] && iw >= 0 && iw < shape.input[1];
                const float a
                    = inside ? static_cast<float>(xs[offsetOf({o[0], o[1], ih, iw}, x.strides())])
                             : 0.0f;
                const float b = static_cast<float>(ws[offsetOf(
                    {o[1], 0, flip ? kh - 1 - ky : ky, flip ? kw - 1 - kx : kx}, w.strides())]);
                acc = std::fma(b, a, acc);
            }
        }
        return Half(acc).data;
    };
    // The outputs checked, in logical (n, c, h, w) order.
    const auto& dims = y.dims();
    std::vector<std::vector<int64_t>> checked;
    if(sampleStride == 1)
    {
        forEachIndex(dims, [&](const std::vector<int64_t>& o) { checked.push_back(o); });
    }
    else
    {
        const int64_t total = dims[0] * dims[1] * dims[2] * dims[3];
        const auto at = [&](int64_t i) {
            return std::vector<int64_t>{i / (dims[1] * dims[2] * dims[3]),
                                        i / (dims[2] * dims[3]) % dims[1],
                                        i / dims[3] % dims[2],
                                        i % dims[3]};
        };
        for(int64_t i = 0; i < total; i += sampleStride)
        {
            checked.push_back(at(i));
        }
        for(int64_t c = 0; c < dims[1]; ++c)
        {
            for(int64_t q = 0; q < dims[3]; ++q)
            {
                checked.push_back({dims[0] - 1, c, dims[2] - 1, q});
            }
        }
    }
    std::vector<uint16_t> want;
    want.reserve(checked.size());
    for(const auto& o : checked)
    {
        want.push_back(expected(o));
    }

    const auto& handler = dispatchHandler(pack);
    int ran = 0;
    for(const auto& config : configs)
    {
        const auto kernel = kernelFor(config, DataType::HALF, direction);
        if(!matchesKernel(pack, fixture.context(), kernel, *bound))
        {
            continue;
        }
        SCOPED_TRACE(config.tag());
        ++ran;
        const auto prepared = handler.prepare(fixture.context(), *bound, kernel);
        ASSERT_NE(prepared, nullptr);
        y.fillWithSentinelValue();
        const std::array<hipdnnPluginDeviceBuffer_t, 3> buffers{
            hipdnnPluginDeviceBuffer_t{X_UID, x.rawDeviceData()},
            hipdnnPluginDeviceBuffer_t{W_UID, w.rawDeviceData()},
            hipdnnPluginDeviceBuffer_t{Y_UID, y.rawDeviceData()}};
        const Handle handle;
        handler.launch(
            handle, *prepared, buffers.data(), static_cast<uint32_t>(buffers.size()), nullptr);
        ASSERT_EQ(hipStreamSynchronize(handle.getStream()), hipSuccess);
        y.markDeviceModified();
        const auto* actual = static_cast<const Half*>(y.rawHostData());
        int64_t mismatches = 0;
        for(size_t i = 0; i < checked.size(); ++i)
        {
            const auto& o = checked[i];
            const uint16_t got = actual[offsetOf(o, y.strides())].data;
            if(got != want[i] && ++mismatches <= 5)
            {
                ADD_FAILURE() << "at output (" << o[0] << ", " << o[1] << ", " << o[2] << ", "
                              << o[3] << "): bits " << got << ", single-rounding fmaf " << want[i];
            }
        }
        EXPECT_EQ(mismatches, 0);
    }
    EXPECT_GE(ran, 1) << "no rowtile config serves this case";
}

TEST(TestGpuConvDepthwiseValuRowtile, HalfIsBitwiseTheFmafOfConvertedOperands)
{
    SKIP_IF_NO_DEVICES();
    // Even and odd channel counts (1 and 2 channels per lane, partial channel blocks), planes
    // that no tile divides, padding past the filter's half, a narrow row, true convolution.
    runRowtileBitwise(depthwise(130, {9, 11}, {5, 5}, {3, 3}, true));
    runRowtileBitwise(depthwise(37, {13, 11}, {3, 3}, {1, 1}, true));
    runRowtileBitwise(depthwise(136, {9, 21}, {7, 7}, {3, 3}, true));
    runRowtileBitwise(depthwise(16, {10, 5}, {7, 7}, {3, 3}, true));
    auto convolution = depthwise(40, {13, 11}, {3, 3}, {1, 1}, true);
    convolution.mode = ConvMode::CONVOLUTION;
    runRowtileBitwise(convolution);
    auto stride3 = depthwise(40, {13, 11}, {3, 3}, {1, 1}, true);
    stride3.stride = {2, 2};
    runRowtileBitwise(stride3);
    auto stride5 = depthwise(37, {19, 17}, {5, 5}, {2, 2}, true);
    stride5.stride = {2, 2};
    runRowtileBitwise(stride5);
}

// The rowtile core has one body for every tensor size: x and y of 2^30 + 2^18 f16 elements (past
// 2 GiB), rows rebased past 2^31 bytes, checked at sampled outputs and the whole last row.
TEST(TestGpuConvDepthwiseValuRowtile, HalfServesTensorsPastTwoGiB)
{
    SKIP_IF_NO_DEVICES();
    size_t freeBytes = 0;
    size_t totalBytes = 0;
    ASSERT_EQ(hipMemGetInfo(&freeBytes, &totalBytes), hipSuccess);
    if(freeBytes < (size_t{6} << 30))
    {
        GTEST_SKIP() << "needs 6 GiB of free device memory";
    }
    // 4097 rows of 4096 columns of 64 channels (512 KiB rows).
    auto shape = depthwise(64, {4097, 4096}, {3, 3}, {1, 1}, true);
    shape.n = 1;
    runRowtileBitwise(shape, {rowtileConfig(3, 4, 4, 2)}, 4099);
}

// Host-only: the matrix-core (hankel) backward-weights gates, one field at a time from an
// admitted base (fp16 channels-last 3x3, 40 channels, on gfx942).
TEST(TestConvDepthwiseValuWrwHankel, RefusesOneFieldAtATime)
{
    struct Case
    {
        std::string name;
        std::function<void(ConvShape&, WrwConfig&, std::string&)> mutate;
        bool admitted;
    };
    const auto base = [] { return depthwise(40, {13, 21}, {3, 3}, {1, 1}, true); };
    const std::vector<Case> cases{
        {"Base", [](ConvShape&, WrwConfig&, std::string&) {}, true},
        {"Float",
         [](ConvShape& s, WrwConfig&, std::string&) { s.dataType = DataType::FLOAT; },
         false},
        {"Bf16",
         [](ConvShape& s, WrwConfig&, std::string&) { s.dataType = DataType::BFLOAT16; },
         true},
        {"Bf16OnGfx908",
         [](ConvShape& s, WrwConfig&, std::string& arch) {
             s.dataType = DataType::BFLOAT16;
             arch = "gfx908";
         },
         false},
        {"Fp16OnGfx908", [](ConvShape&, WrwConfig&, std::string& arch) { arch = "gfx908"; }, true},
        {"Gfx90a", [](ConvShape&, WrwConfig&, std::string& arch) { arch = "gfx90a"; }, true},
        {"Gfx1100", [](ConvShape&, WrwConfig&, std::string& arch) { arch = "gfx1100"; }, false},
        {"Gfx906", [](ConvShape&, WrwConfig&, std::string& arch) { arch = "gfx906"; }, false},
        {"ChannelsFirst",
         [](ConvShape& s, WrwConfig& c, std::string&) {
             s.channelsLast = false;
             c.layout = "NCHW";
         },
         false},
        {"Stride2", [](ConvShape& s, WrwConfig&, std::string&) { s.stride = {2, 2}; }, false},
        {"Stride2OfAStride2Config",
         [](ConvShape& s, WrwConfig& c, std::string&) {
             s.stride = {2, 2};
             c.s = 2;
         },
         true},
        {"UnitStrideOfAStride2Config",
         [](ConvShape&, WrwConfig& c, std::string&) { c.s = 2; },
         false},
        {"WidthStride2OfAStride2Config",
         [](ConvShape& s, WrwConfig& c, std::string&) {
             s.stride = {1, 2};
             c.s = 2;
         },
         false},
        {"HeightStride2OfAStride2Config",
         [](ConvShape& s, WrwConfig& c, std::string&) {
             s.stride = {2, 1};
             c.s = 2;
         },
         false},
        {"Stride3",
         [](ConvShape& s, WrwConfig& c, std::string&) {
             s.stride = {3, 3};
             c.s = 3;
         },
         false},
        {"StrideZero", [](ConvShape&, WrwConfig& c, std::string&) { c.s = 0; }, false},
        {"Stride2Dilation2",
         [](ConvShape& s, WrwConfig& c, std::string&) {
             s.stride = {2, 2};
             s.dilation = {2, 2};
             s.prePadding = s.postPadding = {2, 2};
             c.s = 2;
         },
         false},
        {"WidthStride2", [](ConvShape& s, WrwConfig&, std::string&) { s.stride = {1, 2}; }, false},
        {"HeightStride2", [](ConvShape& s, WrwConfig&, std::string&) { s.stride = {2, 1}; }, false},
        {"HeightDilation2",
         [](ConvShape& s, WrwConfig&, std::string&) {
             s.dilation = {2, 1};
             s.prePadding = s.postPadding = {2, 1};
         },
         false},
        {"Dilation2",
         [](ConvShape& s, WrwConfig&, std::string&) {
             s.dilation = {2, 2};
             s.prePadding = s.postPadding = {2, 2};
         },
         false},
        {"WidthDilation2",
         [](ConvShape& s, WrwConfig&, std::string&) {
             s.dilation = {1, 2};
             s.prePadding = s.postPadding = {1, 2};
         },
         false},
        {"ChannelsNotVectors",
         [](ConvShape& s, WrwConfig&, std::string&) { s.c = s.k = s.groups = 36; },
         false},
        {"FilterNotTheConfigs",
         [](ConvShape& s, WrwConfig&, std::string&) {
             s.filter = {5, 5};
             s.prePadding = s.postPadding = {2, 2};
         },
         false},
        {"FilterHeightNotTheConfigs",
         [](ConvShape& s, WrwConfig&, std::string&) {
             s.filter = {5, 3};
             s.prePadding = s.postPadding = {2, 1};
         },
         false},
        {"FilterWidthNotTheConfigs",
         [](ConvShape& s, WrwConfig&, std::string&) {
             s.filter = {3, 5};
             s.prePadding = s.postPadding = {1, 2};
         },
         false},
        {"ConfigFilterNotTheGraphs",
         [](ConvShape&, WrwConfig& c, std::string&) { c.kw = 5; },
         false},
        {"ConfigDepthNotOne", [](ConvShape&, WrwConfig& c, std::string&) { c.kd = 3; }, false},
        {"ThreeD",
         [](ConvShape& s, WrwConfig&, std::string&) {
             s = depthwise(40, {4, 13, 21}, {1, 3, 3}, {0, 1, 1}, true);
         },
         false},
        {"QNotSteps", [](ConvShape&, WrwConfig& c, std::string&) { c.q = 30; }, false},
        {"QZero", [](ConvShape&, WrwConfig& c, std::string&) { c.q = 0; }, false},
        {"Q68", [](ConvShape&, WrwConfig& c, std::string&) { c.q = 68; }, false},
        {"Q64", [](ConvShape&, WrwConfig& c, std::string&) { c.q = 64; }, true},
        {"GZero", [](ConvShape&, WrwConfig& c, std::string&) { c.g = 0; }, false},
        // On gfx950, whose LDS the next two would fit.
        {"GFive",
         [](ConvShape&, WrwConfig& c, std::string& arch) {
             c.g = 5;
             arch = "gfx950";
         },
         false},
        {"GFour",
         [](ConvShape&, WrwConfig& c, std::string& arch) {
             c.g = 4;
             arch = "gfx950";
         },
         true},
        {"NwgThree",
         [](ConvShape&, WrwConfig& c, std::string& arch) {
             c.nwg = 3;
             arch = "gfx950";
         },
         false},
        // 256 channels a block: 97664 bytes of LDS, gfx950's.
        {"NwgFour",
         [](ConvShape&, WrwConfig& c, std::string& arch) {
             c.nwg = 4;
             arch = "gfx950";
         },
         true},
        {"NwgFourPastGfx942", [](ConvShape&, WrwConfig& c, std::string&) { c.nwg = 4; }, false},
        {"BpcThree", [](ConvShape&, WrwConfig& c, std::string&) { c.bpc = 3; }, false},
        {"BpcSixteen", [](ConvShape&, WrwConfig& c, std::string&) { c.bpc = 16; }, false},
        // 7x7, 16 columns, 4 rows a step, 2 groups a wave: 87264 bytes of LDS.
        {"LdsPastGfx942",
         [](ConvShape& s, WrwConfig& c, std::string&) {
             s.filter = {7, 7};
             s.prePadding = s.postPadding = {3, 3};
             c = {"NHWC", "hankel", 1, 7, 7, 16, 4, 2, 1, 1, 1};
         },
         false},
        {"LdsWithinGfx950",
         [](ConvShape& s, WrwConfig& c, std::string& arch) {
             s.filter = {7, 7};
             s.prePadding = s.postPadding = {3, 3};
             c = {"NHWC", "hankel", 1, 7, 7, 16, 4, 2, 1, 1, 1};
             arch = "gfx950";
         },
         true},
        {"ValuTileWithMatrixFields",
         [](ConvShape&, WrwConfig& c, std::string&) { c.variant = "tile"; },
         false},
        {"ValuTileWithoutThem",
         [](ConvShape&, WrwConfig& c, std::string&) { c = {"NHWC", "tile", 1, 3, 3}; },
         true},
        {"ValuTileWithAStride",
         [](ConvShape&, WrwConfig& c, std::string&) {
             c = {"NHWC", "tile", 1, 3, 3};
             c.s = 1;
         },
         false},
    };
    const auto pack = wrwPack();
    for(const auto& c : cases)
    {
        SCOPED_TRACE(c.name);
        auto shape = base();
        shape.dataType = DataType::HALF;
        WrwConfig config{"NHWC", "hankel", 1, 3, 3, 32, 1, 1, 2, 1, 1};
        std::string arch = "gfx942";
        c.mutate(shape, config, arch);
        const GraphFixture fixture(buildGraph(specFor(shape, ConvDirection::BACKWARD_WEIGHTS)),
                                   countedDevice(64, arch));
        const auto bound = matchesGraph(pack, fixture.context());
        ASSERT_TRUE(bound.has_value());
        EXPECT_EQ(
            matchesKernel(pack, fixture.context(), wrwKernelFor(config, shape.dataType), *bound),
            c.admitted);
    }
}

// Host-only: the matrix-core configs load x and dy 16 bytes at a time, so each pointer must be
// declared 16-byte aligned; an 8-byte one declines (to the VALU variants).
TEST(TestConvDepthwiseValuWrwHankel, DeclinesPointersBelowSixteenBytes)
{
    const auto pack = wrwPack();
    const WrwConfig config{"NHWC", "hankel", 1, 3, 3, 32, 1, 1, 2, 1, 1};
    const auto admits = [&](int64_t xAlign, int64_t yAlign) {
        auto shape = depthwise(40, {13, 21}, {3, 3}, {1, 1}, true);
        shape.dataType = DataType::HALF;
        auto spec = specFor(shape, ConvDirection::BACKWARD_WEIGHTS);
        spec.x.alignment = xAlign;
        spec.y.alignment = yAlign;
        const GraphFixture fixture(buildGraph(spec), countedDevice(64, "gfx942"));
        const auto bound = matchesGraph(pack, fixture.context());
        return bound.has_value()
               && matchesKernel(
                   pack, fixture.context(), wrwKernelFor(config, DataType::HALF), *bound);
    };
    EXPECT_TRUE(admits(16, 16));
    EXPECT_TRUE(admits(32, 64));
    EXPECT_FALSE(admits(8, 16));
    EXPECT_FALSE(admits(16, 8));
    EXPECT_FALSE(admits(2, 2));
}

// Host-only: the vector-tile backward-weights gates, one field at a time from an admitted base
// (fp16 channels-last 3x3, 40 channels, 4 channels a thread, strips of 8; 16-byte pointers).
TEST(TestConvDepthwiseValuWrwVtile, RefusesOneFieldAtATime)
{
    struct Align
    {
        int64_t x = 16;
        int64_t y = 16;
    };
    struct Case
    {
        std::string name;
        std::function<void(ConvShape&, WrwConfig&, Align&)> mutate;
        bool admitted;
    };
    const auto base = [] { return depthwise(40, {13, 21}, {3, 3}, {1, 1}, true); };
    const std::vector<Case> cases{
        {"Base", [](ConvShape&, WrwConfig&, Align&) {}, true},
        {"Float", [](ConvShape& s, WrwConfig&, Align&) { s.dataType = DataType::FLOAT; }, true},
        {"Bf16", [](ConvShape& s, WrwConfig&, Align&) { s.dataType = DataType::BFLOAT16; }, true},
        {"ChannelsFirst",
         [](ConvShape& s, WrwConfig& c, Align&) {
             s.channelsLast = false;
             c.layout = "NCHW";
         },
         false},
        {"HeightStride2", [](ConvShape& s, WrwConfig&, Align&) { s.stride = {2, 1}; }, true},
        {"WidthStride2", [](ConvShape& s, WrwConfig&, Align&) { s.stride = {1, 2}; }, false},
        {"HeightDilation2",
         [](ConvShape& s, WrwConfig&, Align&) {
             s.dilation = {2, 1};
             s.prePadding = s.postPadding = {2, 1};
         },
         true},
        {"WidthDilation2",
         [](ConvShape& s, WrwConfig&, Align&) {
             s.dilation = {1, 2};
             s.prePadding = s.postPadding = {1, 2};
         },
         false},
        {"ChannelsNotVectors",
         [](ConvShape& s, WrwConfig&, Align&) { s.c = s.k = s.groups = 42; },
         false},
        {"ChannelsOfTwoVectors",
         [](ConvShape& s, WrwConfig& c, Align&) {
             s.c = s.k = s.groups = 42;
             c.cv = 2;
         },
         true},
        {"FilterNotTheConfigs",
         [](ConvShape& s, WrwConfig&, Align&) {
             s.filter = {5, 5};
             s.prePadding = s.postPadding = {2, 2};
         },
         false},
        {"FilterWidthNotTheConfigs",
         [](ConvShape& s, WrwConfig&, Align&) {
             s.filter = {3, 5};
             s.prePadding = s.postPadding = {1, 2};
         },
         false},
        {"ConfigDepthNotOne", [](ConvShape&, WrwConfig& c, Align&) { c.kd = 3; }, false},
        {"ThreeD",
         [](ConvShape& s, WrwConfig&, Align&) {
             s = depthwise(40, {4, 13, 21}, {1, 3, 3}, {0, 1, 1}, true);
         },
         false},
        {"Multiplier",
         [](ConvShape& s, WrwConfig& c, Align&) {
             s.k = 80;
             c.mult = 2;
         },
         false},
        {"CvThree", [](ConvShape&, WrwConfig& c, Align&) { c.cv = 3; }, false},
        {"CvZero", [](ConvShape&, WrwConfig& c, Align&) { c.cv = 0; }, false},
        {"CvSixteen", [](ConvShape&, WrwConfig& c, Align&) { c.cv = 16; }, false},
        {"StripThree", [](ConvShape&, WrwConfig& c, Align&) { c.strip = 3; }, false},
        {"StripZero", [](ConvShape&, WrwConfig& c, Align&) { c.strip = 0; }, false},
        {"XAlignedToFourBytes", [](ConvShape&, WrwConfig&, Align& a) { a.x = 4; }, false},
        {"DyAlignedToFourBytes", [](ConvShape&, WrwConfig&, Align& a) { a.y = 4; }, false},
        {"FourBytesFitTwoHalves",
         [](ConvShape&, WrwConfig& c, Align& a) {
             a.x = a.y = 4;
             c.cv = 2;
         },
         true},
        {"WithMatrixFields", [](ConvShape&, WrwConfig& c, Align&) { c.q = 32; }, false},
        {"WithAStride", [](ConvShape&, WrwConfig& c, Align&) { c.s = 1; }, false},
        {"TileWithVectorFields",
         [](ConvShape&, WrwConfig& c, Align&) { c.variant = "tile"; },
         false},
    };
    const auto pack = wrwPack();
    for(const auto& c : cases)
    {
        SCOPED_TRACE(c.name);
        auto shape = base();
        shape.dataType = DataType::HALF;
        WrwConfig config{"NHWC", "vtile", 1, 3, 3};
        config.cv = 4;
        config.strip = 8;
        Align align;
        c.mutate(shape, config, align);
        auto spec = specFor(shape, ConvDirection::BACKWARD_WEIGHTS);
        spec.x.alignment = align.x;
        spec.y.alignment = align.y;
        const GraphFixture fixture(buildGraph(spec), countedDevice(64, "gfx942"));
        const auto bound = matchesGraph(pack, fixture.context());
        ASSERT_TRUE(bound.has_value());
        EXPECT_EQ(
            matchesKernel(pack, fixture.context(), wrwKernelFor(config, shape.dataType), *bound),
            c.admitted);
    }
}

TEST(TestConvDepthwiseValuWrwComputeType, ServesCudnnsComputeTypesPerElementType)
{
    expectCudnnComputeTypes(depthwise(37, {13, 11}, {3, 3}, {1, 1}, true),
                            {DataType::FLOAT, DataType::HALF, DataType::BFLOAT16},
                            [](const ConvShape& s) {
                                const GraphFixture fixture(
                                    buildGraph(specFor(s, ConvDirection::BACKWARD_WEIGHTS)));
                                return matchesGraph(wrwPack(), fixture.context()).has_value();
                            });
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_CONV
