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
#include "engines/kernel_ingestor_engine/packs/ConvDirectProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestGpuConvDirect.cpp
 * @brief The stride-1 halo-tile direct convolution kernels (CONV_DIRECT_FWD, CONV_DIRECT_BWD) on
 *        the device, through the engines' own graph match, kernel match, prepare and launch,
 *        against hipDNN's CPU reference convolution.
 *
 * Every config of the installed catalog that the kernel match admits on this device runs on
 * every case, forward and backward data, in fp32, fp16 and bf16 (and, with TF32 permission, the
 * three-bf16 fp32 configs). The cases cover every instantiated filter (3x3, 5x5, 7x7, 1x7, 7x1,
 * 3x3x3), both layouts (channels-first runs through the staging transposes), padding 0 to 3,
 * asymmetric padding, odd extents, channel counts that are not tile multiples (and 3, which only
 * the scalar configs serve), maps wide and deep enough for several blocks, depth splitting, and
 * true convolution (the filter reflected by the weight pre-pass). The CPU reference is
 * cross-correlation only, so true convolution reflects the filter around it.
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
            direction == ConvDirection::FORWARD ? "hipkernel.conv_direct_fwd.graph_match"
                                                : "hipkernel.conv_direct_bwd.graph_match",
            "",
            "hipkernel.conv_direct.kernel_match",
            "",
            "hipkernel.conv_direct.dispatch",
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
                = i == 0 ? "conv_direct_fwd.kdp.json" : "conv_direct_bwd.kdp.json";
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
};

ConvShape dense(int64_t c,
                int64_t k,
                std::vector<int64_t> input,
                std::vector<int64_t> filter,
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
    shape.stride.assign(shape.input.size(), 1);
    shape.dilation.assign(shape.input.size(), 1);
    shape.channelsLast = channelsLast;
    return shape;
}

ConvShape padded(ConvShape shape, std::vector<int64_t> pre, std::vector<int64_t> post)
{
    shape.prePadding = std::move(pre);
    shape.postPadding = std::move(post);
    return shape;
}

// Extents are not multiples of the tiles and channel counts are not multiples of the column
// tiles or channel chunks, so every edge runs; 24, 40 and 136 channels admit the 8-wide vectors,
// 3 only the scalar ones.
std::vector<DeviceCase> deviceCases()
{
    std::vector<DeviceCase> cases;
    cases.push_back({"Nhwc3x3", dense(24, 40, {13, 11}, {3, 3}, {1, 1}, true)});
    cases.push_back({"Nchw3x3", dense(24, 40, {13, 11}, {3, 3}, {1, 1}, false)});
    cases.push_back({"Nhwc3x3Pad0", dense(40, 24, {9, 11}, {3, 3}, {0, 0}, true)});
    cases.push_back({"Nhwc3x3Pad2", dense(24, 72, {9, 11}, {3, 3}, {2, 2}, true)});
    cases.push_back({"Nhwc3x3Pad3", dense(16, 24, {6, 9}, {3, 3}, {3, 3}, true)});
    cases.push_back({"NhwcSmallChannels", dense(3, 16, {17, 15}, {3, 3}, {1, 1}, true)});
    cases.push_back({"NhwcWide", dense(32, 48, {5, 70}, {3, 3}, {1, 1}, true)});
    cases.push_back({"NhwcMultiBlock", dense(136, 152, {17, 23}, {3, 3}, {1, 1}, true)});
    cases.push_back({"Nhwc5x5", dense(32, 48, {19, 23}, {5, 5}, {2, 2}, true)});
    cases.push_back({"Nchw5x5Asymmetric",
                     padded(dense(24, 40, {11, 13}, {5, 5}, {1, 1}, false), {2, 1}, {1, 2})});
    cases.push_back({"Nhwc7x7", dense(16, 32, {20, 21}, {7, 7}, {3, 3}, true)});
    cases.push_back({"Nhwc1x7", dense(32, 32, {11, 30}, {1, 7}, {0, 3}, true)});
    cases.push_back({"Nchw7x1", dense(32, 40, {30, 11}, {7, 1}, {3, 0}, false)});
    cases.push_back({"Ndhwc3x3x3", dense(32, 40, {5, 7, 9}, {3, 3, 3}, {1, 1, 1}, true)});
    cases.push_back({"Ncdhw3x3x3", dense(16, 24, {5, 6, 7}, {3, 3, 3}, {1, 1, 1}, false)});
    cases.push_back({"Ndhwc3x3x3Pad0", dense(16, 16, {4, 6, 6}, {3, 3, 3}, {0, 0, 0}, true)});
    cases.push_back(
        {"Ndhwc3x3x3Pad2Asymmetric",
         padded(dense(24, 56, {3, 5, 6}, {3, 3, 3}, {2, 1, 2}, true), {2, 1, 2}, {1, 2, 0})});
    auto deep = dense(64, 64, {9, 8, 8}, {3, 3, 3}, {1, 1, 1}, true);
    deep.n = 1;
    cases.push_back({"Ndhwc3x3x3Deep", deep});

    auto convolution = dense(24, 40, {13, 11}, {3, 3}, {1, 1}, true);
    convolution.mode = ConvMode::CONVOLUTION;
    cases.push_back({"Nhwc3x3ConvolutionMode", convolution});
    auto convolution3d = dense(16, 24, {4, 5, 6}, {3, 3, 3}, {1, 1, 1}, false);
    convolution3d.mode = ConvMode::CONVOLUTION;
    cases.push_back({"Ncdhw3x3x3ConvolutionMode", convolution3d});
    auto convolution1x7 = padded(dense(16, 24, {9, 20}, {1, 7}, {0, 3}, true), {0, 1}, {0, 4});
    convolution1x7.mode = ConvMode::CONVOLUTION;
    cases.push_back({"Nhwc1x7AsymmetricConvolutionMode", convolution1x7});
    // Output extent 1 (hipconv #15149 review): the engine runs stride 1 only, so a "valid" 3x3
    // over 3 rows (one output row) and the 3-D whole-filter window.
    cases.push_back({"Nhwc3x3OutputOneHigh", dense(24, 40, {3, 11}, {3, 3}, {0, 0}, true)});
    cases.push_back({"Ndhwc3x3x3OutputOne", dense(16, 24, {3, 3, 3}, {3, 3, 3}, {0, 0, 0}, true)});
    // More tiles than a 256-CU device runs at once: the rows kernel's blocks walk several tiles,
    // prefetching the next tile's first chunk; 136 channels make an odd chunk count (the walk
    // alternates its starting stage) and 8 a single chunk.
    auto manyTiles = dense(136, 8, {160, 160}, {3, 3}, {1, 1}, true);
    manyTiles.n = 6;
    cases.push_back({"Nhwc3x3ManyTiles", manyTiles});
    return cases;
}

/// Which configs a run takes: the native ones, or (fp32 with TF32 permission) the three-bf16 ones.
enum class Math
{
    NATIVE,
    THREE_BF16,
};

/// +inf and -inf planted into @p input (the tensor the reduction reads, besides the filter).
template <typename T>
void plantInfinities(Tensor<T>& input)
{
    int64_t count = 1;
    for(const auto extent : input.dims())
    {
        count *= extent;
    }
    auto* data = static_cast<T*>(input.rawHostData());
    data[0] = static_cast<T>(INFINITY);
    data[count / 2] = static_cast<T>(-INFINITY);
    input.markHostModified();
}

/// A result matches its reference: within @p tolerance where the reference is finite, the same
/// infinity, or NaN where the reference is NaN.
bool matches(float got, float want, float tolerance)
{
    if(std::isnan(want))
    {
        return std::isnan(got);
    }
    if(std::isinf(want))
    {
        return got == want;
    }
    return std::fabs(got - want) <= tolerance;
}

template <typename T>
void runCase(ConvShape shape,
             ConvDirection direction,
             Math math = Math::NATIVE,
             bool infinities = false)
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
    constexpr unsigned int SEED = 1234;
    const bool forward = direction == ConvDirection::FORWARD;
    (forward ? x : y).fillTensorWithRandomValues(-1.0f, 1.0f, SEED);
    w.fillTensorWithRandomValues(-1.0f, 1.0f, SEED + 1);
    if(infinities)
    {
        plantInfinities(forward ? x : y);
    }
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
    // Each output sums (taps x reduction channels) products of values in [-1, 1].
    int64_t reductionLength = (forward ? shape.c : shape.k) / shape.groups;
    for(const auto extent : shape.filter)
    {
        reductionLength *= extent;
    }
    // Three-bf16 drops about 2^-16 of each product (the lo x lo term and the lo parts' rounding).
    const float accumulationBound
        = (math == Math::THREE_BF16 ? 3e-5f : 2e-6f) * static_cast<float>(reductionLength);

    const auto& handler = dispatchHandler(pack);
    int ran = 0;
    for(const auto& kernel : catalog(direction))
    {
        if((kernel.getIntMetadata("math") == 1) != (math == Math::THREE_BF16)
           || !matchesKernel(pack, fixture.context(), kernel, *bound))
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

        written.fillWithSentinelValue();
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
        written.markDeviceModified();

        const auto* actual = static_cast<const T*>(written.rawHostData());
        const auto* expected = static_cast<const T*>(reference.rawHostData());
        int64_t mismatches = 0;
        forEachIndex(written.dims(), [&](const std::vector<int64_t>& index) {
            const float got = static_cast<float>(actual[offsetOf(index, written.strides())]);
            const float want = static_cast<float>(expected[offsetOf(index, reference.strides())]);
            const float tolerance
                = accumulationBound + 2.0f * storageEpsilon<T>() * (std::fabs(want) + 1e-3f);
            if(!matches(got, want, tolerance) && ++mismatches <= 5)
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

class TestGpuConvDirect : public ::testing::TestWithParam<std::tuple<DeviceCase, ConvDirection>>
{
};

TEST_P(TestGpuConvDirect, Float)
{
    SKIP_IF_NO_DEVICES();
    runCase<float>(std::get<0>(GetParam()).shape, std::get<1>(GetParam()));
}

TEST_P(TestGpuConvDirect, Half)
{
    SKIP_IF_NO_DEVICES();
    runCase<Half>(std::get<0>(GetParam()).shape, std::get<1>(GetParam()));
}

TEST_P(TestGpuConvDirect, BFloat16)
{
    SKIP_IF_NO_DEVICES();
    runCase<BFloat16>(std::get<0>(GetParam()).shape, std::get<1>(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuConvDirect,
                         ::testing::Combine(::testing::ValuesIn(deviceCases()),
                                            ::testing::Values(ConvDirection::FORWARD,
                                                              ConvDirection::BACKWARD_DATA)),
                         [](const auto& info) {
                             return std::get<0>(info.param).name
                                    + (std::get<1>(info.param) == ConvDirection::FORWARD ? "Fwd"
                                                                                         : "Bwd");
                         });

/// The three-bf16 fp32 configs (TF32 permission): matrix-core devices only.
class TestGpuConvDirectThreeBf16
    : public ::testing::TestWithParam<std::tuple<DeviceCase, ConvDirection>>
{
};

TEST_P(TestGpuConvDirectThreeBf16, Float)
{
    SKIP_IF_NO_DEVICES();
    const auto properties = currentDeviceProperties();
    const auto arch = std::string(properties.gcnArchName);
    if(arch.rfind("gfx9", 0) != 0 && arch.rfind("gfx11", 0) != 0 && arch.rfind("gfx12", 0) != 0)
    {
        GTEST_SKIP() << "no bf16 matrix cores";
    }
    if(arch.rfind("gfx906", 0) == 0)
    {
        GTEST_SKIP() << "no bf16 matrix cores";
    }
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter tf32(
        "HIP_KERNEL_PROVIDER_CONV_TF32", "1");
    runCase<float>(std::get<0>(GetParam()).shape, std::get<1>(GetParam()), Math::THREE_BF16);
}

/// +inf and -inf in the input: every output an infinity reaches is that infinity (NaN where both
/// meet), as the exact fp32 configs give; the three-bf16 split must not turn them into NaN.
TEST_P(TestGpuConvDirectThreeBf16, FloatInfinities)
{
    SKIP_IF_NO_DEVICES();
    const auto properties = currentDeviceProperties();
    const auto arch = std::string(properties.gcnArchName);
    if((arch.rfind("gfx9", 0) != 0 && arch.rfind("gfx11", 0) != 0 && arch.rfind("gfx12", 0) != 0)
       || arch.rfind("gfx906", 0) == 0)
    {
        GTEST_SKIP() << "no bf16 matrix cores";
    }
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter tf32(
        "HIP_KERNEL_PROVIDER_CONV_TF32", "1");
    runCase<float>(std::get<0>(GetParam()).shape, std::get<1>(GetParam()), Math::THREE_BF16, true);
}

TEST_P(TestGpuConvDirectThreeBf16, FloatInfinitiesNative)
{
    SKIP_IF_NO_DEVICES();
    runCase<float>(std::get<0>(GetParam()).shape, std::get<1>(GetParam()), Math::NATIVE, true);
}

std::vector<DeviceCase> threeBf16Cases()
{
    std::vector<DeviceCase> cases;
    cases.push_back({"Nhwc3x3", dense(24, 40, {13, 11}, {3, 3}, {1, 1}, true)});
    cases.push_back({"Nchw3x3MultiBlock", dense(136, 72, {17, 23}, {3, 3}, {1, 1}, false)});
    cases.push_back({"Ndhwc3x3x3", dense(32, 40, {5, 7, 9}, {3, 3, 3}, {1, 1, 1}, true)});
    auto convolution = dense(24, 40, {13, 11}, {3, 3}, {2, 0}, true);
    convolution.mode = ConvMode::CONVOLUTION;
    cases.push_back({"Nhwc3x3ConvolutionMode", convolution});
    return cases;
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuConvDirectThreeBf16,
                         ::testing::Combine(::testing::ValuesIn(threeBf16Cases()),
                                            ::testing::Values(ConvDirection::FORWARD,
                                                              ConvDirection::BACKWARD_DATA)),
                         [](const auto& info) {
                             return std::get<0>(info.param).name
                                    + (std::get<1>(info.param) == ConvDirection::FORWARD ? "Fwd"
                                                                                         : "Bwd");
                         });

// ---- Host-only: the shipped catalog ----

TEST(TestConvDirectCatalog, EveryConfigParsesAndRunsSomewhere)
{
    static constexpr std::array<const char*, 8> ARCHS{
        "gfx906", "gfx908", "gfx90a", "gfx942", "gfx950", "gfx1100", "gfx1151", "gfx1201"};
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        ASSERT_FALSE(catalog(direction).empty());
        for(const auto& kernel : catalog(direction))
        {
            SCOPED_TRACE(kernel.name);
            const auto config = conv_direct::configOf(kernel);
            ASSERT_TRUE(config.has_value());
            const bool somewhere = std::any_of(ARCHS.begin(), ARCHS.end(), [&](const char* arch) {
                return conv_direct::configRunsOn(*config, arch);
            });
            EXPECT_TRUE(somewhere);
        }
    }
}

TEST(TestConvDirectCatalog, EveryFilterAndDtypeHasAConfigOnEveryArchitecture)
{
    static constexpr std::array<const char*, 8> ARCHS{
        "gfx906", "gfx908", "gfx90a", "gfx942", "gfx950", "gfx1030", "gfx1151", "gfx1201"};
    for(const auto& filter : conv_direct::FILTERS)
    {
        for(const char* dtype : {"FLOAT", "HALF", "BFLOAT16"})
        {
            for(const char* arch : ARCHS)
            {
                const auto& kernels = catalog(ConvDirection::FORWARD);
                const bool served = std::any_of(kernels.begin(), kernels.end(), [&](const auto& k) {
                    const auto config = conv_direct::configOf(k);
                    return config.has_value() && k.getStringMetadata("dtype") == dtype
                           && !config->math && config->kd == filter[0] && config->r == filter[1]
                           && config->s == filter[2] && conv_direct::configRunsOn(*config, arch);
                });
                EXPECT_TRUE(served) << dtype << " " << filter[0] << "x" << filter[1] << "x"
                                    << filter[2] << " on " << arch;
            }
        }
    }
}

} // namespace

#endif
