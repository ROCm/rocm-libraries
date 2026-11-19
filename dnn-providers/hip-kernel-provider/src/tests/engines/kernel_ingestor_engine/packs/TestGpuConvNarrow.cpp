// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <set>
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
#include "engines/kernel_ingestor_engine/packs/ConvDeterminismTestSupport.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvNarrowNative.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestGpuConvNarrow.cpp
 * @brief The narrow convolution kernels (CONV_NARROW_FWD: output layers of at most 8 channels;
 *        CONV_NARROW_BWD: input gradients of at most 8 channels; CONV_NARROW_WRW: backward
 *        weights of both) on the device, through the engine's own graph match, kernel match,
 *        prepare and launch, against hipDNN's CPU reference convolution (fprop / dgrad / wgrad,
 *        fp64 accumulation).
 *
 * Every config of the installed catalog that the kernel match admits on this device runs on every
 * case, in fp32, fp16 and bf16: the shipped filter classes (3x3x3, 3x3, 1x1, 1 x 7 forward;
 * 7x7 and 3x3 at stride 2, 3x7x7, 3x3 and 3x3x3 at stride 1, 1x1 backward), narrow channel counts
 * 1 to 8 (fewer than a config's bound), reduction channel counts that are and are not whole
 * vectors, padding 0 to 3, dilation, odd extents spanning several strips and bands, both layouts,
 * true convolution, and several images. Backward weights: both sides (an output layer's K <= 8,
 * a stem's C <= 8), the same classes and edges, several channel blocks and splits. Infinities and
 * NaN in the streamed operand propagate as the reference's do (forward and backward data never
 * gather products of padding; an output layer's backward weights make no product for a tap that
 * reaches no dy pixel, planted in x's border pixels; a stem's are planted away from the borders,
 * see ConvNarrowWrw.cpp). A tensor past 2 GiB of images each under it runs on 64-bit image bases.
 * Every config relaunches bitwise identical.
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
    if(direction == ConvDirection::BACKWARD_WEIGHTS)
    {
        return {"",
                "hipkernel.conv_narrow_wrw.graph_match",
                "",
                "hipkernel.conv_narrow_wrw.kernel_match",
                "",
                "hipkernel.conv_narrow_wrw.dispatch",
                "",
                "",
                ""};
    }
    return {"",
            direction == ConvDirection::BACKWARD_DATA ? "hipkernel.conv_narrow_bwd.graph_match"
                                                      : "hipkernel.conv_narrow_fwd.graph_match",
            "",
            "hipkernel.conv_narrow.kernel_match",
            "",
            "hipkernel.conv_narrow.dispatch",
            "",
            "",
            ""};
}

/// The shipped catalogs of a direction (every dtype's pack), as kernel definitions: read from the
/// production descriptor tree beside the provider, so these cases run exactly the configs that
/// ship.
std::vector<hipdnn_plugin_sdk::ingestor::KernelDefinition> catalog(ConvDirection direction)
{
    static const auto s_catalogs = [] {
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
        std::array<std::vector<hipdnn_plugin_sdk::ingestor::KernelDefinition>, 3> catalogs;
        for(size_t i = 0; i < catalogs.size(); ++i)
        {
            const std::array<const char*, 3> prefixes{
                "conv_narrow_fwd_", "conv_narrow_bwd_", "conv_narrow_wrw_"};
            const std::string prefix = prefixes.at(i);
            std::set<std::filesystem::path> found;
            for(const auto& root : roots)
            {
                std::error_code error;
                for(auto it = std::filesystem::recursive_directory_iterator(root, error);
                    !error && it != std::filesystem::recursive_directory_iterator();
                    it.increment(error))
                {
                    const auto name = it->path().filename().string();
                    if(name.rfind(prefix, 0) == 0 && it->path().extension() == ".json"
                       && name.find(".kdp.") != std::string::npos)
                    {
                        found.insert(it->path());
                    }
                }
                if(!found.empty())
                {
                    break;
                }
            }
            if(found.empty())
            {
                throw std::runtime_error("no " + prefix
                                         + "*.kdp.json beside the provider or this test binary");
            }
            for(const auto& file : found)
            {
                std::ifstream stream(file);
                const auto pack = nlohmann::json::parse(stream);
                for(const auto& descriptor : pack.at("kernelDescriptors"))
                {
                    hipdnn_plugin_sdk::ingestor::KernelDefinition kernel;
                    kernel.name = descriptor.at("name").get<std::string>();
                    kernel.source.kind
                        = hipdnn_plugin_sdk::ingestor::KernelSourceKind::EMBEDDED_SOURCE;
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
        }
        return catalogs;
    }();
    if(direction == ConvDirection::BACKWARD_WEIGHTS)
    {
        return s_catalogs[2];
    }
    return s_catalogs[direction == ConvDirection::BACKWARD_DATA ? 1 : 0];
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
    ConvDirection direction = ConvDirection::FORWARD;
    /// Infinities and NaN planted into the streamed operand.
    bool specials = false;
    /// Run only the first admitted config (a large case).
    bool oneConfig = false;
    /// Served only with 160 KiB of LDS (gfx950): a 7x7 stem's 147 or a 3x7x7 stem's 441
    /// (narrow channel, tap) columns of products outgrow 64 KiB.
    bool largeLds = false;
    /// Backward weights of an output layer: infinities and NaN planted in x's border pixels,
    /// whose taps that reach no dy pixel make no product (an explicitly padded GEMM's result).
    bool borderSpecials = false;
};

ConvShape conv(int64_t c,
               int64_t k,
               std::vector<int64_t> input,
               std::vector<int64_t> filter,
               std::vector<int64_t> stride,
               const std::vector<int64_t>& pad,
               bool channelsLast,
               int64_t n = 2)
{
    ConvShape shape;
    shape.n = n;
    shape.c = c;
    shape.k = k;
    shape.groups = 1;
    shape.input = std::move(input);
    shape.filter = std::move(filter);
    shape.prePadding = pad;
    shape.postPadding = pad;
    shape.stride = std::move(stride);
    shape.dilation.assign(shape.input.size(), 1);
    shape.channelsLast = channelsLast;
    return shape;
}

// Extents span several strips and bands and are not multiples of them; reduction channels are
// whole 16-byte vectors in some cases and not in others.
std::vector<DeviceCase> deviceCases()
{
    std::vector<DeviceCase> cases;
    for(const bool cl : {true, false})
    {
        const std::string l = cl ? "Nhwc" : "Nchw";
        cases.push_back({l + "3x3K3", conv(32, 3, {37, 75}, {3, 3}, {1, 1}, {1, 1}, cl)});
        cases.push_back({l + "3x3K3OddC", conv(19, 3, {13, 41}, {3, 3}, {1, 1}, {1, 1}, cl)});
        cases.push_back({l + "3x3K4", conv(40, 4, {21, 33}, {3, 3}, {1, 1}, {1, 1}, cl, 1)});
        cases.push_back({l + "3x3K8", conv(24, 8, {11, 19}, {3, 3}, {1, 1}, {1, 1}, cl, 1)});
        cases.push_back({l + "3x3K2Pad0", conv(48, 2, {17, 23}, {3, 3}, {1, 1}, {0, 0}, cl, 1)});
        cases.push_back(
            {l + "3x3x3K3", conv(24, 3, {5, 9, 37}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, cl, 1)});
        cases.push_back(
            {l + "3x3x3K3Pad0", conv(96, 3, {6, 11, 70}, {3, 3, 3}, {1, 1, 1}, {0, 0, 0}, cl, 1)});
        cases.push_back({l + "1x1K3", conv(64, 3, {9, 17}, {1, 1}, {1, 1}, {0, 0}, cl)});
        cases.push_back({l + "1x7K1", conv(32, 1, {150}, {7}, {1}, {3}, cl)});
        auto dilated = conv(40, 3, {19, 27}, {3, 3}, {1, 1}, {2, 2}, cl, 1);
        dilated.dilation = {2, 2};
        cases.push_back({l + "3x3K3Dilation2", dilated});
        auto convolution = conv(32, 3, {15, 21}, {3, 3}, {1, 1}, {1, 1}, cl, 1);
        convolution.mode = ConvMode::CONVOLUTION;
        cases.push_back({l + "3x3K3ConvolutionMode", convolution});

        // Backward data: conv(c = dx channels, k = dy channels, dx extents, ...).
        const auto bwd = ConvDirection::BACKWARD_DATA;
        cases.push_back({l + "Bwd7x7S2C3", conv(3, 64, {45, 61}, {7, 7}, {2, 2}, {3, 3}, cl), bwd});
        cases.back().largeLds = true;
        cases.push_back(
            {l + "Bwd7x7S2C1", conv(1, 24, {20, 21}, {7, 7}, {2, 2}, {3, 3}, cl, 1), bwd});
        cases.push_back(
            {l + "Bwd3x3S2C3", conv(3, 24, {31, 39}, {3, 3}, {2, 2}, {1, 1}, cl, 1), bwd});
        cases.push_back(
            {l + "Bwd3x3S2C3Pad0", conv(3, 32, {29, 35}, {3, 3}, {2, 2}, {0, 0}, cl, 1), bwd});
        cases.push_back(
            {l + "Bwd3x3S1C4", conv(4, 40, {19, 37}, {3, 3}, {1, 1}, {1, 1}, cl, 1), bwd});
        cases.push_back({l + "Bwd3x3x3C1",
                         conv(1, 32, {7, 9, 11}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, cl, 1),
                         bwd});
        cases.push_back({l + "Bwd3x7x7S122C3",
                         conv(3, 24, {6, 30, 30}, {3, 7, 7}, {1, 2, 2}, {1, 3, 3}, cl, 1),
                         bwd});
        cases.back().largeLds = true;
        cases.push_back(
            {l + "Bwd1x1C3", conv(3, 48, {13, 17}, {1, 1}, {1, 1}, {0, 0}, cl, 1), bwd});
        auto bwdConvolution = conv(3, 40, {21, 19}, {7, 7}, {2, 2}, {3, 3}, cl, 1);
        bwdConvolution.mode = ConvMode::CONVOLUTION;
        cases.push_back({l + "Bwd7x7S2ConvolutionMode", bwdConvolution, bwd});
        cases.back().largeLds = true;

        cases.push_back({l + "3x3K3Specials",
                         conv(32, 3, {13, 37}, {3, 3}, {1, 1}, {1, 1}, cl, 1),
                         ConvDirection::FORWARD,
                         true});
        cases.push_back({l + "Bwd7x7S2Specials",
                         conv(3, 24, {23, 30}, {7, 7}, {2, 2}, {3, 3}, cl, 1),
                         bwd,
                         true});
        cases.back().largeLds = true;
    }
    // Backward weights: an output layer's (K <= 8) and a stem's (C <= 8), conv(c, k, x extents).
    const auto wrw = ConvDirection::BACKWARD_WEIGHTS;
    for(const bool cl : {true, false})
    {
        const std::string l = std::string("Wrw") + (cl ? "Nhwc" : "Nchw");
        cases.push_back({l + "3x3K3", conv(32, 3, {37, 75}, {3, 3}, {1, 1}, {1, 1}, cl), wrw});
        cases.push_back(
            {l + "3x3K3OddC", conv(19, 3, {13, 41}, {3, 3}, {1, 1}, {1, 1}, cl, 1), wrw});
        cases.push_back(
            {l + "3x3K4ManyChannels", conv(200, 4, {9, 17}, {3, 3}, {1, 1}, {1, 1}, cl, 1), wrw});
        cases.push_back(
            {l + "3x3x3K3", conv(24, 3, {5, 9, 37}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, cl, 1), wrw});
        cases.push_back({l + "3x3x3K3Pad0",
                         conv(96, 3, {6, 11, 70}, {3, 3, 3}, {1, 1, 1}, {0, 0, 0}, cl, 1),
                         wrw});
        cases.push_back({l + "1x1K2", conv(48, 2, {9, 17}, {1, 1}, {1, 1}, {0, 0}, cl), wrw});
        cases.push_back({l + "1x7K1", conv(32, 1, {150}, {7}, {1}, {3}, cl), wrw});
        auto dilated = conv(40, 3, {19, 27}, {3, 3}, {1, 1}, {2, 2}, cl, 1);
        dilated.dilation = {2, 2};
        cases.push_back({l + "3x3K3Dilation2", dilated, wrw});
        cases.push_back(
            {l + "3x3K3Stride2", conv(24, 3, {13, 37}, {3, 3}, {2, 2}, {1, 1}, cl, 1), wrw});
        cases.push_back({l + "7x7S2C3", conv(3, 64, {45, 61}, {7, 7}, {2, 2}, {3, 3}, cl), wrw});
        cases.push_back({l + "3x3S2C3", conv(3, 24, {31, 39}, {3, 3}, {2, 2}, {1, 1}, cl, 1), wrw});
        cases.push_back(
            {l + "3x3x3C1", conv(1, 32, {7, 9, 11}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, cl, 1), wrw});
        cases.push_back({l + "3x7x7S122C3",
                         conv(3, 24, {6, 30, 30}, {3, 7, 7}, {1, 2, 2}, {1, 3, 3}, cl, 1),
                         wrw});
        cases.back().largeLds = true;
        cases.push_back({l + "4x4S4C3", conv(3, 48, {36, 44}, {4, 4}, {4, 4}, {0, 0}, cl, 1), wrw});
        auto wrwConvolution = conv(3, 40, {21, 19}, {7, 7}, {2, 2}, {3, 3}, cl, 1);
        wrwConvolution.mode = ConvMode::CONVOLUTION;
        cases.push_back({l + "7x7S2ConvolutionMode", wrwConvolution, wrw});
        cases.push_back(
            {l + "3x3K3Specials", conv(32, 3, {13, 37}, {3, 3}, {1, 1}, {1, 1}, cl, 1), wrw, true});
        for(const auto& [name, shape] : std::vector<std::pair<std::string, ConvShape>>{
                {"3x3K3BorderSpecials", conv(32, 3, {13, 37}, {3, 3}, {1, 1}, {1, 1}, cl, 1)},
                {"3x3K3Stride2UnusedSpecials",
                 conv(32, 3, {14, 38}, {3, 3}, {2, 2}, {0, 0}, cl, 1)},
                {"3x3x3K3BorderSpecials",
                 conv(24, 3, {5, 9, 21}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, cl, 1)}})
        {
            cases.push_back({l + name, shape, wrw});
            cases.back().borderSpecials = true;
        }
        cases.push_back({l + "7x7S2C3Specials",
                         conv(3, 24, {23, 30}, {7, 7}, {2, 2}, {3, 3}, cl, 1),
                         wrw,
                         true});
    }
    // Five channels-last images of 2^28 bf16 elements each: 2.5 GiB in all, on 64-bit image
    // bases.
    cases.push_back({"NhwcLargeTensor1x1K1",
                     conv(64, 1, {2048, 2048}, {1, 1}, {1, 1}, {0, 0}, true, 5),
                     ConvDirection::FORWARD,
                     false,
                     true});
    return cases;
}

/// Plants +inf at the centre of every image of a packed @p tensor (no tap of it reaches padding:
/// backward weights' wide operand).
template <typename T>
void plantCentreInfinity(Tensor<T>& tensor)
{
    auto* data = static_cast<T*>(tensor.rawHostData());
    const auto& dims = tensor.dims();
    std::vector<int64_t> index(dims.size(), 0);
    for(size_t axis = 2; axis < dims.size(); ++axis)
    {
        index[axis] = dims[axis] / 2;
    }
    data[offsetOf(index, tensor.strides())]
        = static_cast<T>(std::numeric_limits<float>::infinity());
    tensor.markHostModified();
}

/// Plants +inf, -inf and NaN at fixed positions of @p tensor.
template <typename T>
void plantBorderSpecials(Tensor<T>& tensor)
{
    auto* data = static_cast<T*>(tensor.rawHostData());
    const auto& dims = tensor.dims();
    std::vector<int64_t> first(dims.size(), 0);
    std::vector<int64_t> last(dims.size(), 0);
    for(size_t axis = 2; axis < dims.size(); ++axis)
    {
        last[axis] = dims[axis] - 1;
    }
    first[1] = 1;
    data[offsetOf(first, tensor.strides())]
        = static_cast<T>(std::numeric_limits<float>::infinity());
    last[1] = 2;
    data[offsetOf(last, tensor.strides())]
        = static_cast<T>(-std::numeric_limits<float>::infinity());
    last[1] = dims[1] - 1;
    last.back() = 0;
    data[offsetOf(last, tensor.strides())]
        = static_cast<T>(std::numeric_limits<float>::quiet_NaN());
    tensor.markHostModified();
}

template <typename T>
void plantSpecials(Tensor<T>& tensor)
{
    auto* data = static_cast<T*>(tensor.rawHostData());
    const auto count = static_cast<int64_t>(tensor.elementSpace());
    const std::array<float, 3> values{std::numeric_limits<float>::infinity(),
                                      -std::numeric_limits<float>::infinity(),
                                      std::numeric_limits<float>::quiet_NaN()};
    for(size_t i = 0; i < values.size(); ++i)
    {
        data[(count / 7) * static_cast<int64_t>(i + 1) + 3] = static_cast<T>(values.at(i));
    }
    tensor.markHostModified();
}

template <typename T>
void runCase(const DeviceCase& deviceCase)
{
    auto shape = deviceCase.shape;
    const auto direction = deviceCase.direction;
    const bool backward = direction == ConvDirection::BACKWARD_DATA;
    const bool weights = direction == ConvDirection::BACKWARD_WEIGHTS;
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
    Tensor<T>& in = backward ? y : x;
    Tensor<T>& convOut = backward ? x : y;
    Tensor<T>& out = weights ? w : convOut;
    in.fillTensorWithRandomValues(-1.0f, 1.0f, SEED);
    if(weights)
    {
        y.fillTensorWithRandomValues(-1.0f, 1.0f, SEED + 1);
    }
    else
    {
        w.fillTensorWithRandomValues(-1.0f, 1.0f, SEED + 1);
    }
    if(deviceCase.borderSpecials)
    {
        plantBorderSpecials(x);
    }
    if(deviceCase.specials)
    {
        if(weights)
        {
            // The wide operand: x of an output layer, dy of a stem.
            plantCentreInfinity(shape.k < shape.c ? x : y);
        }
        else
        {
            plantSpecials(in);
        }
    }

    const bool reflect = shape.mode == ConvMode::CONVOLUTION;
    Tensor<T> reflectedW(spec.w.dims, spec.w.strides);
    if(reflect)
    {
        reflectFilter(w, reflectedW);
    }
    Tensor<T> reference(out.dims(), out.strides());
    if(weights)
    {
        Tensor<T> crossCorrelation(spec.w.dims, spec.w.strides);
        CpuFpReferenceConvolution::wgrad<T, T, T, double>(x,
                                                          reflect ? crossCorrelation : reference,
                                                          y,
                                                          shape.stride,
                                                          shape.dilation,
                                                          shape.prePadding,
                                                          shape.postPadding);
        if(reflect)
        {
            crossCorrelation.markHostModified();
            reflectFilter(crossCorrelation, reference);
        }
    }
    else if(backward)
    {
        CpuFpReferenceConvolution::dgrad<T, T, T, double>(reference,
                                                          reflect ? reflectedW : w,
                                                          y,
                                                          shape.stride,
                                                          shape.dilation,
                                                          shape.prePadding,
                                                          shape.postPadding);
    }
    else
    {
        CpuFpReferenceConvolution::fprop<T, T, T, double>(x,
                                                          reflect ? reflectedW : w,
                                                          reference,
                                                          shape.stride,
                                                          shape.dilation,
                                                          shape.prePadding,
                                                          shape.postPadding);
    }
    int64_t reductionLength = backward ? shape.k : shape.c;
    for(const auto extent : shape.filter)
    {
        reductionLength *= extent;
    }
    if(weights)
    {
        // Every output position of every image.
        reductionLength = shape.n;
        for(const auto extent : shape.output())
        {
            reductionLength *= extent;
        }
    }
    const float accumulationBound = 2e-6f * static_cast<float>(reductionLength);

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
        const size_t workspaceBytes = handler.workspaceBytes(fixture.context(), *bound, kernel);
        ASSERT_GT(workspaceBytes, 0U);
        void* workspace = nullptr;
        ASSERT_EQ(hipMalloc(&workspace, workspaceBytes), hipSuccess);
        // Stale contents must not leak into the result.
        ASSERT_EQ(hipMemset(workspace, 0x7f, workspaceBytes), hipSuccess);
        out.fillWithSentinelValue();
        const std::array<hipdnnPluginDeviceBuffer_t, 3> buffers{
            hipdnnPluginDeviceBuffer_t{X_UID, x.rawDeviceData()},
            hipdnnPluginDeviceBuffer_t{W_UID, w.rawDeviceData()},
            hipdnnPluginDeviceBuffer_t{Y_UID, y.rawDeviceData()}};
        const Handle handle;
        handler.launch(
            handle, *prepared, buffers.data(), static_cast<uint32_t>(buffers.size()), workspace);
        ASSERT_EQ(hipStreamSynchronize(handle.getStream()), hipSuccess);
        // Deterministic: a second launch gives the same bits (the UED's RFC 0025 claim).
        EXPECT_TRUE(conv_determinism_test::relaunchReproducesOutput(
            [&] {
                handler.launch(handle,
                               *prepared,
                               buffers.data(),
                               static_cast<uint32_t>(buffers.size()),
                               workspace);
            },
            handle.getStream(),
            out,
            workspace,
            workspaceBytes));
        ASSERT_EQ(hipFree(workspace), hipSuccess);
        out.markDeviceModified();

        const auto* actual = static_cast<const T*>(out.rawHostData());
        const auto* expected = static_cast<const T*>(reference.rawHostData());
        int64_t mismatches = 0;
        forEachIndex(out.dims(), [&](const std::vector<int64_t>& index) {
            const auto got = static_cast<float>(actual[offsetOf(index, out.strides())]);
            const auto want = static_cast<float>(expected[offsetOf(index, reference.strides())]);
            bool ok = false;
            if(std::isnan(want))
            {
                ok = std::isnan(got);
            }
            else if(std::isinf(want))
            {
                ok = got == want;
            }
            else
            {
                const float tolerance
                    = accumulationBound + 2.0f * storageEpsilon<T>() * (std::fabs(want) + 1e-3f);
                ok = std::fabs(got - want) <= tolerance;
            }
            if(!ok && ++mismatches <= 5)
            {
                ADD_FAILURE() << "at linear index " << offsetOf(index, out.strides()) << ": got "
                              << got << ", want " << want;
            }
        });
        EXPECT_EQ(mismatches, 0);
        if(deviceCase.oneConfig)
        {
            break;
        }
    }
    // The vector-ALU configs ship everywhere: every case is served on every device, but those
    // whose products outgrow 64 KiB of LDS, which the 160 KiB devices serve.
    hipDeviceProp_t properties{};
    ASSERT_EQ(hipGetDeviceProperties(&properties, 0), hipSuccess);
    if(!deviceCase.largeLds
       || conv_narrow::ldsLimitOf(properties.gcnArchName) >= int64_t{160} * 1024)
    {
        EXPECT_GE(ran, 1) << "no catalog config serves this case";
    }
}

class TestGpuConvNarrow : public ::testing::TestWithParam<DeviceCase>
{
};

TEST_P(TestGpuConvNarrow, Float)
{
    SKIP_IF_NO_DEVICES();
    runCase<float>(GetParam());
}

TEST_P(TestGpuConvNarrow, Half)
{
    SKIP_IF_NO_DEVICES();
    runCase<Half>(GetParam());
}

TEST_P(TestGpuConvNarrow, BFloat16)
{
    SKIP_IF_NO_DEVICES();
    runCase<BFloat16>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuConvNarrow,
                         ::testing::ValuesIn(deviceCases()),
                         [](const auto& info) { return info.param.name; });

// ---- Host-only: the shipped catalog ----

TEST(TestConvNarrowCatalog, EveryConfigParsesAndIsDistinct)
{
    const auto hashOf = [](const std::string& text) {
        return static_cast<int64_t>(std::hash<std::string>{}(text));
    };
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        ASSERT_FALSE(catalog(direction).empty());
        std::set<std::vector<int64_t>> knobs;
        for(const auto& kernel : catalog(direction))
        {
            SCOPED_TRACE(kernel.name);
            const auto config = conv_narrow::configOf(kernel);
            ASSERT_TRUE(config.has_value());
            EXPECT_EQ(config->direction,
                      direction == ConvDirection::FORWARD ? std::string("FWD")
                                                          : std::string("BWD"));
            // The integer knobs and the strings that fix the graph and the device (dtype,
            // layout, ship list) tell every config apart.
            const std::vector<int64_t> key{kernel.getIntMetadata("atom_id"),
                                           config->kmin,
                                           config->ko,
                                           config->fd,
                                           config->fr,
                                           config->fs,
                                           config->sw,
                                           config->ds,
                                           config->cmin,
                                           config->cm,
                                           config->pd,
                                           config->sf,
                                           config->wr,
                                           config->rb,
                                           config->nw,
                                           config->op,
                                           config->layout == "nchw" ? 1 : 0,
                                           hashOf(std::string(kernel.getStringMetadata("dtype"))),
                                           hashOf(std::string(kernel.getStringMetadata("archs")))};
            EXPECT_TRUE(knobs.insert(key).second) << "two configs with the same knobs";
        }
    }
    ASSERT_FALSE(catalog(ConvDirection::BACKWARD_WEIGHTS).empty());
    std::set<std::vector<int64_t>> knobs;
    for(const auto& kernel : catalog(ConvDirection::BACKWARD_WEIGHTS))
    {
        SCOPED_TRACE(kernel.name);
        const auto config = conv_narrow::wrwConfigOf(kernel);
        ASSERT_TRUE(config.has_value());
        const std::vector<int64_t> key{kernel.getIntMetadata("atom_id"),
                                       config->side == "k" ? 1 : 0,
                                       config->kmin,
                                       config->ko,
                                       config->fd,
                                       config->fr,
                                       config->fs,
                                       config->wr,
                                       config->rb,
                                       config->nw,
                                       config->mpw,
                                       config->layout == "nchw" ? 1 : 0,
                                       hashOf(std::string(kernel.getStringMetadata("dtype"))),
                                       hashOf(std::string(kernel.getStringMetadata("archs")))};
        EXPECT_TRUE(knobs.insert(key).second) << "two configs with the same knobs";
    }
}

} // namespace

#endif
