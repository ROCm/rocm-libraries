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
#include "engines/kernel_ingestor_engine/packs/ConvNaiveTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvSmallcProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestGpuConvSmallc.cpp
 * @brief The small-input-channel convolution kernels (CONV_SMALLC_FWD, and CONV_SMALLC_BWD:
 *        stride-1 backward data with few output-gradient channels) on the device, through the
 *        engine's own graph match, kernel match, prepare and launch, against hipDNN's CPU
 *        reference convolution (fprop / dgrad).
 *
 * Every config of the installed catalog that the kernel match admits on this device runs on every
 * case, in fp32, fp16 and bf16: each instantiated geometry (7x7, 3x3 and 4x4 at their strides;
 * 3x7x7, 1x7x7, 3x3x3), channel counts 1 to 4 (fewer than a config's bound pads with
 * zeros), output channels that are not tile multiples, padding 0 to 3, odd extents, both layouts,
 * several tiles per image, and true convolution (the pre-pass reflects the filter). Both
 * epilogues run (LDS slots; registers on channels-fastest outputs, with aligned vector and scalar
 * channel runs). Backward cases: dy of 1 to 4 channels, dx channels not tile multiples, padding
 * below, at and past the filter (negative forward padding), 2-D and 3-D, both layouts, both modes.
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
            direction == ConvDirection::BACKWARD_DATA ? "hipkernel.conv_smallc_bwd.graph_match"
                                                      : "hipkernel.conv_smallc_fwd.graph_match",
            "",
            "hipkernel.conv_smallc.kernel_match",
            "",
            "hipkernel.conv_smallc.dispatch",
            "",
            "",
            ""};
}

/// The shipped catalog, as kernel definitions: read from the production
/// descriptor tree beside the provider (the unit tests' own descriptor root holds only test
/// sets), so these cases run exactly the configs that ship.
std::vector<hipdnn_plugin_sdk::ingestor::KernelDefinition> catalog(ConvDirection direction
                                                                   = ConvDirection::FORWARD)
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
                = i == 0 ? "conv_smallc_fwd.kdp.json" : "conv_smallc_bwd.kdp.json";
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
};

ConvShape stem(int64_t c,
               int64_t k,
               std::vector<int64_t> input,
               std::vector<int64_t> filter,
               std::vector<int64_t> stride,
               std::vector<int64_t> pad,
               bool channelsLast)
{
    ConvShape shape;
    shape.n = 2;
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

// Extents are not tile multiples and K is not a multiple of the 16-channel fragments, so every
// edge runs.
std::vector<DeviceCase> deviceCases()
{
    std::vector<DeviceCase> cases;
    cases.push_back({"Nchw7x7S2", stem(3, 40, {37, 29}, {7, 7}, {2, 2}, {3, 3}, false)});
    cases.push_back({"Nhwc7x7S2", stem(3, 70, {33, 45}, {7, 7}, {2, 2}, {3, 3}, true)});
    cases.push_back({"Nchw7x7S2C1", stem(1, 24, {20, 21}, {7, 7}, {2, 2}, {3, 3}, false)});
    cases.push_back({"Nhwc3x3S2", stem(3, 24, {31, 19}, {3, 3}, {2, 2}, {1, 1}, true)});
    cases.push_back({"Nchw3x3S2Pad0", stem(3, 32, {29, 35}, {3, 3}, {2, 2}, {0, 0}, false)});
    cases.push_back({"Nchw3x3S1C4", stem(4, 20, {19, 37}, {3, 3}, {1, 1}, {1, 1}, false)});
    cases.push_back({"Nhwc3x3S1C2", stem(2, 18, {17, 18}, {3, 3}, {1, 1}, {2, 2}, true)});
    // Channels-fastest y with K a multiple of 4: the register epilogue's vector stores (and the
    // scalar tail of a partial channel block).
    cases.push_back({"Nhwc3x3S1C4K40", stem(4, 40, {19, 37}, {3, 3}, {1, 1}, {1, 1}, true)});
    cases.push_back({"Nhwc4x4S4", stem(3, 48, {36, 44}, {4, 4}, {4, 4}, {0, 0}, true)});
    cases.push_back(
        {"Ncdhw3x7x7S122", stem(3, 64, {6, 30, 30}, {3, 7, 7}, {1, 2, 2}, {1, 3, 3}, false)});
    cases.push_back(
        {"Ncdhw1x7x7S122", stem(3, 45, {4, 22, 26}, {1, 7, 7}, {1, 2, 2}, {0, 3, 3}, false)});
    cases.push_back(
        {"Ncdhw3x3x3C1", stem(1, 32, {7, 9, 11}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, false)});
    cases.push_back(
        {"Ncdhw3x3x3S2", stem(4, 16, {9, 10, 11}, {3, 3, 3}, {2, 2, 2}, {1, 1, 1}, false)});
    auto many = stem(3, 64, {150, 170}, {7, 7}, {2, 2}, {3, 3}, false);
    many.n = 3;
    cases.push_back({"Nchw7x7S2ManyTiles", many});
    auto convolution = stem(3, 40, {21, 19}, {7, 7}, {2, 2}, {3, 3}, true);
    convolution.mode = ConvMode::CONVOLUTION;
    cases.push_back({"Nhwc7x7S2ConvolutionMode", convolution});

    // Backward data: stem(c = dx channels, k = dy channels, dx extents, ...).
    const auto bwd = ConvDirection::BACKWARD_DATA;
    cases.push_back({"BwdNhwc3x3K3", stem(72, 3, {37, 29}, {3, 3}, {1, 1}, {1, 1}, true), bwd});
    cases.push_back(
        {"BwdNhwc3x3K3C130", stem(130, 3, {21, 19}, {3, 3}, {1, 1}, {1, 1}, true), bwd});
    cases.push_back({"BwdNhwc3x3K4Pad0", stem(40, 4, {21, 33}, {3, 3}, {1, 1}, {0, 0}, true), bwd});
    cases.push_back({"BwdNchw3x3K3", stem(33, 3, {19, 23}, {3, 3}, {1, 1}, {1, 1}, false), bwd});
    cases.push_back({"BwdNhwc3x3K1", stem(24, 1, {17, 20}, {3, 3}, {1, 1}, {1, 1}, true), bwd});
    auto past = stem(36, 2, {18, 21}, {3, 3}, {1, 1}, {3, 2}, true);
    past.postPadding = {0, 1}; // pre-padding past the filter: negative forward padding
    cases.push_back({"BwdNhwc3x3PadPastFilter", past, bwd});
    cases.push_back(
        {"BwdNdhwc3x3x3K3", stem(40, 3, {5, 18, 22}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, true), bwd});
    cases.push_back(
        {"BwdNcdhw3x3x3K4", stem(20, 4, {4, 9, 11}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, false), bwd});
    auto bwdConvolution = stem(48, 3, {23, 17}, {3, 3}, {1, 1}, {1, 1}, true);
    bwdConvolution.mode = ConvMode::CONVOLUTION;
    cases.push_back({"BwdNhwc3x3ConvolutionMode", bwdConvolution, bwd});
    auto bwdMany = stem(128, 3, {70, 90}, {3, 3}, {1, 1}, {1, 1}, true);
    bwdMany.n = 3;
    cases.push_back({"BwdNhwc3x3ManyTiles", bwdMany, bwd});
    return cases;
}

template <typename T>
void runCase(ConvShape shape, ConvDirection direction)
{
    const bool backward = direction == ConvDirection::BACKWARD_DATA;
    shape.dataType = dataTypeOf<T>();
    const auto spec = specFor(shape, direction);
    const GraphFixture fixture(buildGraph(spec), currentDeviceProperties());
    const auto pack = packFor(direction);
    const auto bound = matchesGraph(pack, fixture.context());
    ASSERT_TRUE(bound.has_value()) << "the engine refused a case it should serve";

    Tensor<T> x(spec.x.dims, spec.x.strides);
    Tensor<T> w(spec.w.dims, spec.w.strides);
    Tensor<T> y(spec.y.dims, spec.y.strides);
    constexpr unsigned int SEED = 1234;
    // The kernel reads x forward and dy backward, and writes the other one.
    Tensor<T>& in = backward ? y : x;
    Tensor<T>& out = backward ? x : y;
    in.fillTensorWithRandomValues(-1.0f, 1.0f, SEED);
    w.fillTensorWithRandomValues(-1.0f, 1.0f, SEED + 1);

    const bool reflect = shape.mode == ConvMode::CONVOLUTION;
    Tensor<T> reflectedW(spec.w.dims, spec.w.strides);
    if(reflect)
    {
        reflectFilter(w, reflectedW);
    }
    Tensor<T> reference(out.dims(), out.strides());
    if(backward)
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
        CpuFpReferenceConvolution::fprop(x,
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
        void* workspace = nullptr;
        if(workspaceBytes > 0)
        {
            ASSERT_EQ(hipMalloc(&workspace, workspaceBytes), hipSuccess);
        }
        out.fillWithSentinelValue();
        const std::array<hipdnnPluginDeviceBuffer_t, 3> buffers{
            hipdnnPluginDeviceBuffer_t{X_UID, x.rawDeviceData()},
            hipdnnPluginDeviceBuffer_t{W_UID, w.rawDeviceData()},
            hipdnnPluginDeviceBuffer_t{Y_UID, y.rawDeviceData()}};
        const Handle handle;
        handler.launch(
            handle, *prepared, buffers.data(), static_cast<uint32_t>(buffers.size()), workspace);
        ASSERT_EQ(hipStreamSynchronize(handle.getStream()), hipSuccess);
        if(workspace != nullptr)
        {
            ASSERT_EQ(hipFree(workspace), hipSuccess);
        }
        out.markDeviceModified();

        const auto* actual = static_cast<const T*>(out.rawHostData());
        const auto* expected = static_cast<const T*>(reference.rawHostData());
        int64_t mismatches = 0;
        forEachIndex(out.dims(), [&](const std::vector<int64_t>& index) {
            const float got = static_cast<float>(actual[offsetOf(index, out.strides())]);
            const float want = static_cast<float>(expected[offsetOf(index, reference.strides())]);
            const float tolerance
                = accumulationBound + 2.0f * storageEpsilon<T>() * (std::fabs(want) + 1e-3f);
            if(!(std::fabs(got - want) <= tolerance) && ++mismatches <= 5)
            {
                ADD_FAILURE() << "at linear index " << offsetOf(index, out.strides()) << ": got "
                              << got << ", want " << want << ", tolerance " << tolerance;
            }
        });
        EXPECT_EQ(mismatches, 0);
    }
    // Forward fp32 has FMA configs everywhere; 16-bit has MMA configs on the matrix-core GPUs (no
    // bf16 ones on gfx908). Backward ships to the matrix-core GPUs only (fp32 by MFMA on CDNA,
    // by FMA on gfx11 / gfx12).
    const auto arch = std::string(currentDeviceProperties().gcnArchName);
    const bool mma = (arch.rfind("gfx9", 0) == 0 && arch.rfind("gfx906", 0) != 0)
                     || arch.rfind("gfx11", 0) == 0 || arch.rfind("gfx12", 0) == 0;
    const bool bf16On908 = arch.rfind("gfx908", 0) == 0 && std::is_same_v<T, BFloat16>;
    if((std::is_same_v<T, float> && (!backward || mma)) || (mma && !bf16On908))
    {
        EXPECT_GE(ran, 1) << "no catalog config serves this case";
    }
}

class TestGpuConvSmallc : public ::testing::TestWithParam<DeviceCase>
{
};

TEST_P(TestGpuConvSmallc, Float)
{
    SKIP_IF_NO_DEVICES();
    runCase<float>(GetParam().shape, GetParam().direction);
}

TEST_P(TestGpuConvSmallc, Half)
{
    SKIP_IF_NO_DEVICES();
    runCase<Half>(GetParam().shape, GetParam().direction);
}

TEST_P(TestGpuConvSmallc, BFloat16)
{
    SKIP_IF_NO_DEVICES();
    runCase<BFloat16>(GetParam().shape, GetParam().direction);
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuConvSmallc,
                         ::testing::ValuesIn(deviceCases()),
                         [](const auto& info) { return info.param.name; });

// ---- Host-only: the shipped catalog ----

TEST(TestConvSmallcCatalog, EveryConfigParses)
{
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        ASSERT_FALSE(catalog(direction).empty());
        for(const auto& kernel : catalog(direction))
        {
            SCOPED_TRACE(kernel.name);
            EXPECT_TRUE(conv_smallc::configOf(kernel).has_value());
        }
    }
}

TEST(TestConvSmallcCatalog, BackwardConfigsAreUnitStride)
{
    for(const auto& kernel : catalog(ConvDirection::BACKWARD_DATA))
    {
        SCOPED_TRACE(kernel.name);
        const auto config = conv_smallc::configOf(kernel);
        ASSERT_TRUE(config.has_value());
        EXPECT_EQ(config->sd * config->sh * config->sw, 1);
        EXPECT_EQ(kernel.getStringMetadata("direction"), "BWD");
    }
}

} // namespace

#endif
