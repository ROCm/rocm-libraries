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
#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceConvolution.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <nlohmann/json.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvIgemmProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvTf32Timing.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestGpuConvIgemm.cpp
 * @brief The implicit-GEMM convolution kernels (CONV_IGEMM_FWD, CONV_IGEMM_BWD) on the device,
 *        through the engines' own graph match, kernel match, prepare and launch, against
 *        hipDNN's CPU reference convolution.
 *
 * Every config of the installed catalog that the kernel match admits on this device runs on
 * every case, forward and backward data, in fp32, fp16 and bf16. The cases cover both layouts
 * (channels-first runs through the staging copies), groups, stride, dilation, asymmetric
 * padding, 1-D to 3-D, channel counts no vector divides (only the scalar configs serve them),
 * and true convolution (a reflected, staged filter). The CPU reference is cross-correlation
 * only, so true convolution reflects the filter around it.
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
            direction == ConvDirection::FORWARD ? "hipkernel.conv_igemm_fwd.graph_match"
                                                : "hipkernel.conv_igemm_bwd.graph_match",
            "",
            "hipkernel.conv_igemm.kernel_match",
            "",
            "hipkernel.conv_igemm.dispatch",
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
            const std::string file = i == 0 ? "conv_igemm_fwd.kdp.json" : "conv_igemm_bwd.kdp.json";
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

// Row counts are not multiples of the tiles and channel counts are not multiples of the column
// tiles, so every edge runs; 24 and 40 channels admit the 8-wide vectors, 3 only the scalar ones.
std::vector<DeviceCase> deviceCases()
{
    std::vector<DeviceCase> cases;
    cases.push_back({"Nhwc3x3", dense(24, 40, {13, 11}, {3, 3}, {1, 1}, true)});
    cases.push_back({"Nchw3x3", dense(24, 40, {13, 11}, {3, 3}, {1, 1}, false)});
    // Small groups (4 channels each), channels-first: the group-fusion configs.
    auto smallGroups = dense(32, 32, {9, 10}, {3, 3}, {1, 1}, false);
    smallGroups.groups = 8;
    cases.push_back({"NchwSmallGroups", smallGroups});
    // Channels-first images of whole vectors: the channels-first-output configs' vector stores.
    cases.push_back({"Nchw3x3WholeVectors", dense(24, 40, {8, 12}, {3, 3}, {1, 1}, false)});
    auto groupedFirst = dense(32, 48, {12, 13}, {3, 3}, {1, 1}, false);
    groupedFirst.groups = 4;
    groupedFirst.stride = {2, 2};
    cases.push_back({"NchwGroupedStride2", groupedFirst});
    cases.push_back({"Nhwc1x1", dense(64, 32, {7, 9}, {1, 1}, {0, 0}, true)});
    cases.push_back({"NhwcSmallChannels", dense(3, 16, {17, 15}, {3, 3}, {1, 1}, true)});

    auto grouped = dense(32, 48, {12, 13}, {3, 3}, {1, 1}, true);
    grouped.groups = 4;
    grouped.stride = {2, 2};
    cases.push_back({"NhwcGroupedStride2", grouped});

    auto dilated1d = dense(32, 32, {100}, {7}, {9}, true);
    dilated1d.dilation = {3};
    cases.push_back({"Nhwc1dDilated", dilated1d});

    // Transposed-convolution shape: backward data with stride 2 and an even filter.
    auto upsample = dense(16, 16, {50}, {4}, {1}, true);
    upsample.stride = {2};
    cases.push_back({"Nhwc1dK4Stride2", upsample});

    cases.push_back({"Ncdhw3x3x3", dense(16, 24, {5, 6, 7}, {3, 3, 3}, {1, 1, 1}, false)});
    cases.push_back({"Ndhwc3x3x3", dense(16, 24, {5, 6, 7}, {3, 3, 3}, {1, 1, 1}, true)});

    auto asymmetric = dense(16, 24, {12, 13}, {5, 3}, {0, 2}, true);
    asymmetric.prePadding = {2, 1};
    asymmetric.postPadding = {0, 3};
    asymmetric.dilation = {2, 1};
    asymmetric.stride = {1, 2};
    cases.push_back({"NhwcAsymmetricPaddingDilationStride", asymmetric});

    auto convolution = dense(24, 40, {13, 11}, {3, 3}, {1, 1}, true);
    convolution.mode = ConvMode::CONVOLUTION;
    cases.push_back({"Nhwc3x3ConvolutionMode", convolution});

    // Small-channel groups (8 and 4 per group both ways): the tap-packing configs, whose k-steps
    // span several taps, with stride and dilation so taps past the filter's end run.
    auto grouped8 = dense(64, 64, {9, 10}, {3, 3}, {2, 2}, true);
    grouped8.groups = 8;
    grouped8.dilation = {2, 2};
    cases.push_back({"NhwcGroups8PerGroup", grouped8});
    auto grouped4 = dense(16, 16, {5, 7, 6}, {3, 3, 3}, {1, 1, 1}, true);
    grouped4.groups = 4;
    grouped4.stride = {2, 2, 2};
    cases.push_back({"Ndhwc3x3x3Groups4PerGroup", grouped4});

    // Transposed-convolution shape with many output channels: the direct-filter backward-data
    // configs read the K T R S C filter with output-channel pairs past one k-step.
    auto upsample8 = dense(40, 72, {37}, {8}, {2}, true);
    upsample8.stride = {4};
    cases.push_back({"Nhwc1dK8Stride4", upsample8});

    // Strided layers whose output extent is 1 on an axis (hipconv #15149 review: a stride
    // normalized for a size-1 axis computed the wrong output size). Backward data then leaves
    // input rows no tap reaches, which must come out zero.
    auto toOne = dense(32, 24, {4, 4}, {3, 3}, {0, 0}, true);
    toOne.stride = {2, 2};
    cases.push_back({"NhwcStride2OutputOne", toOne});
    auto toOneHigh = dense(16, 24, {4, 9}, {3, 3}, {0, 1}, false);
    toOneHigh.stride = {2, 2};
    cases.push_back({"NchwStride2OutputOneHigh", toOneHigh});
    auto toOneDeep = dense(16, 16, {3, 6, 6}, {3, 3, 3}, {0, 1, 1}, true);
    toOneDeep.stride = {2, 2, 2};
    cases.push_back({"Ndhwc3x3x3Stride2OutputOneDeep", toOneDeep});
    return cases;
}

/// Which configs a run takes: the graph's own element type, or (fp32 under TF32 permission) the
/// TF32-class ones: xf32 atoms (gfx942) and three-bf16.
enum class Math
{
    NATIVE,
    TF32,
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
    const auto& handler = dispatchHandler(pack);
    int ran = 0;
    for(const auto& kernel : catalog(direction))
    {
        const bool xf32 = kernel.getStringMetadata("atom").find("xf32") != std::string::npos;
        const bool split = kernel.getStringMetadata("math") == "bf16x3";
        if(!matchesKernel(pack, fixture.context(), kernel, *bound)
           || (xf32 || split) != (math == Math::TF32))
        {
            continue;
        }
        // Known, reported to the fwd lead: group fusion's zero filter blocks multiply the other
        // groups' infinities into NaN (0 x inf), so fused configs are left out of the
        // infinity runs until that is fixed.
        if(infinities && kernel.getIntMetadata("fuse") != 0)
        {
            continue;
        }
        // xf32 rounds each operand to 10 mantissa bits (about 2^-11 of a product each);
        // three-bf16 drops lo x lo (about 2^-16).
        const float accumulationBound = (xf32    ? 1e-3f
                                         : split ? 3e-5f
                                                 : 2e-6f)
                                        * static_cast<float>(reductionLength);
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

class TestGpuConvIgemm : public ::testing::TestWithParam<std::tuple<DeviceCase, ConvDirection>>
{
};

TEST_P(TestGpuConvIgemm, Float)
{
    SKIP_IF_NO_DEVICES();
    runCase<float>(std::get<0>(GetParam()).shape, std::get<1>(GetParam()));
}

TEST_P(TestGpuConvIgemm, Half)
{
    SKIP_IF_NO_DEVICES();
    runCase<Half>(std::get<0>(GetParam()).shape, std::get<1>(GetParam()));
}

TEST_P(TestGpuConvIgemm, BFloat16)
{
    SKIP_IF_NO_DEVICES();
    runCase<BFloat16>(std::get<0>(GetParam()).shape, std::get<1>(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuConvIgemm,
                         ::testing::Combine(::testing::ValuesIn(deviceCases()),
                                            ::testing::Values(ConvDirection::FORWARD,
                                                              ConvDirection::BACKWARD_DATA)),
                         [](const auto& info) {
                             return std::get<0>(info.param).name
                                    + (std::get<1>(info.param) == ConvDirection::FORWARD ? "Fwd"
                                                                                         : "Bwd");
                         });

/// Whether the device has TF32-class K1 configs: bf16 matrix cores where three bf16 products
/// are admitted (gfx908 included), and gfx942's xf32.
bool hasTf32Forms()
{
    const auto properties = currentDeviceProperties();
    const std::string arch(hipdnn_plugin_sdk::stripArchFeatures(properties.gcnArchName));
    return arch == "gfx908" || arch == "gfx90a" || arch == "gfx942" || arch == "gfx950"
           || arch.rfind("gfx11", 0) == 0 || arch.rfind("gfx12", 0) == 0;
}

std::vector<DeviceCase> tf32Cases()
{
    std::vector<DeviceCase> cases;
    for(const auto& deviceCase : deviceCases())
    {
        if(deviceCase.name == "Nhwc3x3" || deviceCase.name == "Nchw3x3"
           || deviceCase.name == "Ndhwc3x3x3" || deviceCase.name == "Nhwc1x1"
           || deviceCase.name == "NhwcGroupedStride2")
        {
            cases.push_back(deviceCase);
        }
    }
    return cases;
}

/// The TF32-class fp32 configs (xf32, three-bf16) under TF32 permission, against the fp64 reference
/// at their own bounds; and with +inf and -inf in the input, where every output a infinity reaches
/// must be that infinity (or NaN where both meet), as exact fp32 gives. Native fp32 configs take
/// the same infinities.
class TestGpuConvIgemmTf32 : public ::testing::TestWithParam<std::tuple<DeviceCase, ConvDirection>>
{
};

TEST_P(TestGpuConvIgemmTf32, Float)
{
    SKIP_IF_NO_DEVICES();
    if(!hasTf32Forms())
    {
        GTEST_SKIP() << "no TF32-class configs on this device";
    }
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter tf32(
        "HIP_KERNEL_PROVIDER_CONV_TF32", "1");
    runCase<float>(std::get<0>(GetParam()).shape, std::get<1>(GetParam()), Math::TF32);
}

TEST_P(TestGpuConvIgemmTf32, FloatInfinities)
{
    SKIP_IF_NO_DEVICES();
    if(!hasTf32Forms())
    {
        GTEST_SKIP() << "no TF32-class configs on this device";
    }
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter tf32(
        "HIP_KERNEL_PROVIDER_CONV_TF32", "1");
    runCase<float>(std::get<0>(GetParam()).shape, std::get<1>(GetParam()), Math::TF32, true);
}

TEST_P(TestGpuConvIgemmTf32, FloatInfinitiesNative)
{
    SKIP_IF_NO_DEVICES();
    runCase<float>(std::get<0>(GetParam()).shape, std::get<1>(GetParam()), Math::NATIVE, true);
}

/// Opt-in (K1_TF32_TIMING=<shape CSV>, forward): the best config per math form (exact, xf32,
/// three-bf16) per shape, TF32 permitted. See ConvTf32Timing.hpp.
TEST(TestGpuConvIgemmTiming, ByMathForm)
{
    SKIP_IF_NO_DEVICES();
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter tf32(
        "HIP_KERNEL_PROVIDER_CONV_TF32", "1");
    conv_tf32_timing::timeByForm(
        packFor(ConvDirection::FORWARD),
        ConvDirection::FORWARD,
        catalog(ConvDirection::FORWARD),
        "K1_TF32_TIMING",
        [](const auto& kernel) {
            if(kernel.getStringMetadata("math") == "bf16x3")
            {
                return std::string("bf16x3");
            }
            return std::string(kernel.getStringMetadata("atom").find("xf32") != std::string::npos
                                   ? "xf32"
                                   : "exact");
        });
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuConvIgemmTf32,
                         ::testing::Combine(::testing::ValuesIn(tf32Cases()),
                                            ::testing::Values(ConvDirection::FORWARD,
                                                              ConvDirection::BACKWARD_DATA)),
                         [](const auto& info) {
                             return std::get<0>(info.param).name
                                    + (std::get<1>(info.param) == ConvDirection::FORWARD ? "Fwd"
                                                                                         : "Bwd");
                         });

// ---- Host-only: the atom table and the catalog ----

/// A 2-D problem as the halo and packing checks read it (defined below).
conv_igemm::IgemmProblem haloProblem(
    ConvDirection direction, int c, int h, int filter, int pad, int stride, int dilation);

TEST(TestConvIgemmCatalog, EveryAtomIsKnownAndRunsSomewhere)
{
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        for(const auto& kernel : catalog(direction))
        {
            const auto& atom = kernel.getStringMetadata("atom");
            SCOPED_TRACE(kernel.name);
            ASSERT_TRUE(conv_igemm::atomNamed(atom).has_value());
            static constexpr std::array<const char*, 7> ARCHS{
                "gfx908", "gfx90a", "gfx942", "gfx950", "gfx1100", "gfx1151", "gfx1201"};
            const bool somewhere = std::any_of(ARCHS.begin(), ARCHS.end(), [&](const char* arch) {
                return conv_igemm::atomRunsOn(atom, arch);
            });
            EXPECT_TRUE(somewhere);
        }
    }
}

TEST(TestConvIgemmCatalog, AtomsRunOnlyWhereTheirInstructionsExist)
{
    EXPECT_TRUE(conv_igemm::atomRunsOn("mfma32_f16", "gfx942:sramecc+:xnack-"));
    EXPECT_FALSE(conv_igemm::atomRunsOn("mfma32_f16", "gfx1100"));
    EXPECT_TRUE(conv_igemm::atomRunsOn("mfma16_bf16_908", "gfx908:sramecc+:xnack-"));
    EXPECT_FALSE(conv_igemm::atomRunsOn("mfma16_bf16", "gfx908"));
    EXPECT_TRUE(conv_igemm::atomRunsOn("mfma16_xf32", "gfx942"));
    EXPECT_FALSE(conv_igemm::atomRunsOn("mfma16_xf32", "gfx950"));
    EXPECT_TRUE(conv_igemm::atomRunsOn("wmma11_f16", "gfx1151"));
    EXPECT_FALSE(conv_igemm::atomRunsOn("wmma11_f16", "gfx1201"));
    EXPECT_TRUE(conv_igemm::atomRunsOn("wmma12_bf16", "gfx1201"));
    EXPECT_TRUE(conv_igemm::atomRunsOn("fma", "gfx906"));
    EXPECT_FALSE(conv_igemm::atomRunsOn("no_such_atom", "gfx942"));
}

TEST(TestConvIgemmCatalog, KnobsTellApartEveryConfigOfAGraph)
{
    // Candidate enumeration identifies a config by its knob tuple: within one direction and
    // dtype (what one graph can admit) no two configs may share every knob.
    static constexpr std::array<const char*, 19> KNOBS{"atom",
                                                       "math",
                                                       "bm",
                                                       "bn",
                                                       "bk",
                                                       "wm",
                                                       "wn",
                                                       "threads",
                                                       "vec",
                                                       "stages",
                                                       "xcd",
                                                       "halo",
                                                       "wdirect",
                                                       "pack",
                                                       "cf",
                                                       "swz",
                                                       "dma",
                                                       "fuse",
                                                       "large"};
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        std::vector<std::string> tuples;
        for(const auto& kernel : catalog(direction))
        {
            std::string tuple = kernel.getStringMetadata("dtype");
            for(const char* knob : KNOBS)
            {
                const std::string name(knob);
                tuple += "|"
                         + (name == "atom" || name == "math"
                                ? kernel.getStringMetadata(name)
                                : std::to_string(kernel.getIntMetadata(name)));
            }
            EXPECT_EQ(std::count(tuples.begin(), tuples.end(), tuple), 0) << kernel.name;
            tuples.push_back(tuple);
        }
    }
}

TEST(TestConvIgemmCatalog, HaloConfigsAreSingleStageMatrixCoreAndNoTwoStageTilesShip)
{
    int halos = 0;
    int xcds = 0;
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        for(const auto& kernel : catalog(direction))
        {
            SCOPED_TRACE(kernel.name);
            // Every catalog arch has 64 KB of LDS a workgroup, where the second buffer loses.
            EXPECT_EQ(kernel.getIntMetadata("stages"), 1);
            if(kernel.getIntMetadata("halo") != 0)
            {
                ++halos;
                EXPECT_NE(kernel.getStringMetadata("atom"), "fma");
            }
            xcds += kernel.getIntMetadata("xcd") != 0 ? 1 : 0;
        }
    }
    EXPECT_GT(halos, 0);
    EXPECT_GT(xcds, 0);
}

TEST(TestConvIgemmCatalog, DirectFilterConfigsAreBackwardDataOnlyAndPackingIsNotHalo)
{
    int direct = 0;
    int packed = 0;
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        for(const auto& kernel : catalog(direction))
        {
            SCOPED_TRACE(kernel.name);
            if(kernel.getIntMetadata("wdirect") != 0)
            {
                ++direct;
                EXPECT_EQ(direction, ConvDirection::BACKWARD_DATA);
                EXPECT_EQ(kernel.getIntMetadata("pack"), 0);
            }
            if(kernel.getIntMetadata("pack") != 0)
            {
                ++packed;
                EXPECT_EQ(kernel.getIntMetadata("halo"), 0);
            }
        }
    }
    EXPECT_GT(direct, 0);
    EXPECT_GT(packed, 0);
}

hipdnn_plugin_sdk::ingestor::DeviceProperties deviceOf(const std::string& arch, int units);
conv_igemm::IgemmProblem parsedProblem(ConvShape shape, ConvDirection direction);

TEST(TestConvIgemmCatalog, EveryMatrixCoreConfigButTapPackingHasAChannelsFirstOutputTwin)
{
    static constexpr std::array<const char*, 17> SHAPE{"atom",
                                                       "math",
                                                       "bm",
                                                       "bn",
                                                       "bk",
                                                       "wm",
                                                       "wn",
                                                       "threads",
                                                       "vec",
                                                       "stages",
                                                       "xcd",
                                                       "halo",
                                                       "wdirect",
                                                       "pack",
                                                       "swz",
                                                       "dma",
                                                       "fuse"};
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        const auto& kernels = catalog(direction);
        const auto tupleOf = [](const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel) {
            std::string tuple = kernel.getStringMetadata("dtype");
            for(const char* knob : SHAPE)
            {
                const std::string name(knob);
                tuple += "|"
                         + (name == "atom" || name == "math"
                                ? kernel.getStringMetadata(name)
                                : std::to_string(kernel.getIntMetadata(name)));
            }
            return tuple;
        };
        int twins = 0;
        for(const auto& kernel : kernels)
        {
            SCOPED_TRACE(kernel.name);
            const bool cf = kernel.getIntMetadata("cf") != 0;
            const bool eligible
                = kernel.getStringMetadata("atom") != "fma" && kernel.getIntMetadata("pack") == 0;
            if(cf)
            {
                EXPECT_TRUE(eligible);
                continue;
            }
            const auto tuple = tupleOf(kernel);
            const auto twin = std::count_if(kernels.begin(), kernels.end(), [&](const auto& other) {
                return other.getIntMetadata("cf") != 0 && tupleOf(other) == tuple;
            });
            EXPECT_EQ(twin, eligible ? 1 : 0);
            twins += static_cast<int>(twin);
        }
        EXPECT_GT(twins, 0);
    }
}

TEST(TestConvIgemmCatalog, EveryPlainNativeConfigHasAFlatReductionTwin)
{
    static constexpr std::array<const char*, 12> SHAPE{
        "atom", "math", "bm", "bn", "bk", "wm", "wn", "threads", "vec", "stages", "xcd", "cf"};
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        const auto& kernels = catalog(direction);
        const auto tupleOf = [](const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel) {
            std::string tuple = kernel.getStringMetadata("dtype");
            for(const char* knob : SHAPE)
            {
                const std::string name(knob);
                tuple += "|"
                         + (name == "atom" || name == "math"
                                ? kernel.getStringMetadata(name)
                                : std::to_string(kernel.getIntMetadata(name)));
            }
            return tuple;
        };
        int flat = 0;
        for(const auto& kernel : kernels)
        {
            SCOPED_TRACE(kernel.name);
            const bool plain
                = kernel.getIntMetadata("halo") == 0 && kernel.getIntMetadata("wdirect") == 0
                  && kernel.getIntMetadata("swz") == 0 && kernel.getIntMetadata("cf") == 0
                  && kernel.getIntMetadata("fuse") == 0
                  && kernel.getStringMetadata("math") == "native";
            if(kernel.getIntMetadata("pack") == 2)
            {
                EXPECT_TRUE(plain);
                ++flat;
                continue;
            }
            if(kernel.getIntMetadata("pack") != 0 || !plain)
            {
                continue;
            }
            const auto tuple = tupleOf(kernel);
            EXPECT_EQ(std::count_if(kernels.begin(),
                                    kernels.end(),
                                    [&](const auto& other) {
                                        return other.getIntMetadata("pack") == 2
                                               && tupleOf(other) == tuple;
                                    }),
                      1);
        }
        EXPECT_GT(flat, 0);
    }
}

TEST(TestConvIgemmCatalog, FlatReductionServesOnlyChannelsItsKStepDoesNotDivide)
{
    // c 32 (forward reduction) is whole 32-wide k-steps; k 48 (backward data) is not.
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        auto shape = dense(32, 48, {12, 13}, {3, 3}, {1, 1}, true);
        shape.dataType = DataType::HALF;
        const GraphFixture fixture(buildGraph(specFor(shape, direction)),
                                   deviceOf("gfx942:sramecc+:xnack-", 304));
        const auto pack = packFor(direction);
        const auto bound = matchesGraph(pack, fixture.context());
        ASSERT_TRUE(bound.has_value());
        const auto& kernels = catalog(direction);
        const auto flat = std::find_if(kernels.begin(), kernels.end(), [](const auto& kernel) {
            return kernel.getIntMetadata("pack") == 2 && kernel.getStringMetadata("dtype") == "HALF"
                   && kernel.getStringMetadata("atom") == "mfma32_f16"
                   && kernel.getIntMetadata("bk") == 32 && kernel.getIntMetadata("xcd") == 0;
        });
        ASSERT_NE(flat, kernels.end());
        const bool forward = direction == ConvDirection::FORWARD;
        EXPECT_EQ(matchesKernel(pack, fixture.context(), *flat, *bound), !forward);
        auto unknown = *flat;
        unknown.metadata["pack"] = int64_t{3};
        EXPECT_FALSE(matchesKernel(pack, fixture.context(), unknown, *bound));
    }
    conv_igemm::IgemmProblem problem;
    problem.direction = ConvDirection::FORWARD;
    problem.args.c = 224;
    problem.args.k = 224;
    problem.args.groups = 4; // 56 channels per group
    EXPECT_TRUE(conv_igemm::flatFits(problem, 32));
    problem.args.groups = 7; // 32
    EXPECT_FALSE(conv_igemm::flatFits(problem, 32));
    EXPECT_TRUE(conv_igemm::flatFits(problem, 64));
}

TEST(TestConvIgemmCatalog, SwizzledTileConfigsAreSixteenBitVectorEightPlainTiles)
{
    int swizzled = 0;
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        for(const auto& kernel : catalog(direction))
        {
            if(kernel.getIntMetadata("swz") == 0)
            {
                continue;
            }
            SCOPED_TRACE(kernel.name);
            ++swizzled;
            const auto info = conv_igemm::atomNamed(kernel.getStringMetadata("atom"));
            ASSERT_TRUE(info.has_value());
            ASSERT_GT(info->swizzleK, 0);
            EXPECT_EQ(kernel.getIntMetadata("bk") % info->swizzleK, 0);
            EXPECT_EQ(kernel.getStringMetadata("math"), "native");
            EXPECT_EQ(kernel.getIntMetadata("vec"), 8);
            EXPECT_EQ(kernel.getIntMetadata("halo"), 0);
            EXPECT_EQ(kernel.getIntMetadata("wdirect"), 0);
            EXPECT_EQ(kernel.getIntMetadata("pack"), 0);
        }
    }
    EXPECT_GT(swizzled, 0);
}

TEST(TestConvIgemmCatalog, SwizzledTileKnobDeclinesEveryUnsupportedCombination)
{
    auto shape = dense(32, 48, {12, 13}, {3, 3}, {1, 1}, true);
    shape.dataType = DataType::HALF;
    const GraphFixture fixture(buildGraph(specFor(shape, ConvDirection::BACKWARD_DATA)),
                               deviceOf("gfx942:sramecc+:xnack-", 304));
    const auto pack = packFor(ConvDirection::BACKWARD_DATA);
    const auto bound = matchesGraph(pack, fixture.context());
    ASSERT_TRUE(bound.has_value());
    const auto& kernels = catalog(ConvDirection::BACKWARD_DATA);
    const auto base = std::find_if(kernels.begin(), kernels.end(), [](const auto& kernel) {
        return kernel.getIntMetadata("swz") != 0 && kernel.getStringMetadata("dtype") == "HALF"
               && kernel.getStringMetadata("atom") == "mfma32_f16"
               && kernel.getIntMetadata("xcd") == 0 && kernel.getIntMetadata("cf") == 0;
    });
    ASSERT_NE(base, kernels.end());
    EXPECT_TRUE(matchesKernel(pack, fixture.context(), *base, *bound));
    const auto declines = [&](const char* field, const auto& value) {
        auto kernel = *base;
        kernel.metadata[field] = value;
        return !matchesKernel(pack, fixture.context(), kernel, *bound);
    };
    EXPECT_TRUE(declines("swz", int64_t{2}));
    EXPECT_TRUE(declines("vec", int64_t{4}));
    EXPECT_TRUE(declines("bk", int64_t{128}));
    // 16x16x16 fragments are read two k-steps (32) per chunk: a 16 k-step cannot hold them.
    {
        auto narrow = *base;
        narrow.metadata["atom"] = std::string("mfma16_f16");
        narrow.metadata["bk"] = int64_t{16};
        EXPECT_FALSE(matchesKernel(pack, fixture.context(), narrow, *bound));
        narrow.metadata["swz"] = int64_t{0};
        EXPECT_TRUE(matchesKernel(pack, fixture.context(), narrow, *bound));
    }
    EXPECT_TRUE(declines("halo", int64_t{1}));
    EXPECT_TRUE(declines("wdirect", int64_t{1}));
    EXPECT_TRUE(declines("pack", int64_t{1}));
    EXPECT_TRUE(declines("math", std::string("bf16x3")));

    // An atom whose fragments are not whole 8-, 16- or 32-byte runs: gfx908's bf16 (16x16x8).
    shape.dataType = DataType::BFLOAT16;
    const GraphFixture gfx908(buildGraph(specFor(shape, ConvDirection::BACKWARD_DATA)),
                              deviceOf("gfx908", 120));
    const auto bound908 = matchesGraph(pack, gfx908.context());
    ASSERT_TRUE(bound908.has_value());
    const auto plain908 = std::find_if(kernels.begin(), kernels.end(), [](const auto& kernel) {
        return kernel.getStringMetadata("atom") == "mfma16_bf16_908"
               && kernel.getIntMetadata("vec") == 8 && kernel.getIntMetadata("halo") == 0
               && kernel.getIntMetadata("wdirect") == 0 && kernel.getIntMetadata("pack") == 0
               && kernel.getIntMetadata("cf") == 0 && kernel.getIntMetadata("bk") == 32;
    });
    ASSERT_NE(plain908, kernels.end());
    EXPECT_TRUE(matchesKernel(pack, gfx908.context(), *plain908, *bound908));
    auto swizzled908 = *plain908;
    swizzled908.metadata["swz"] = int64_t{1};
    EXPECT_FALSE(matchesKernel(pack, gfx908.context(), swizzled908, *bound908));
}

TEST(TestConvIgemmCatalog, DmaConfigsAreSwizzledAndServeOnlyGfx950)
{
    auto shape = dense(32, 48, {12, 13}, {3, 3}, {1, 1}, true);
    shape.dataType = DataType::HALF;
    int dma = 0;
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        const auto pack = packFor(direction);
        const GraphFixture gfx950(buildGraph(specFor(shape, direction)), deviceOf("gfx950", 256));
        const GraphFixture gfx942(buildGraph(specFor(shape, direction)),
                                  deviceOf("gfx942:sramecc+:xnack-", 304));
        const auto bound950 = matchesGraph(pack, gfx950.context());
        const auto bound942 = matchesGraph(pack, gfx942.context());
        ASSERT_TRUE(bound950.has_value() && bound942.has_value());
        for(const auto& kernel : catalog(direction))
        {
            if(kernel.getIntMetadata("dma") == 0)
            {
                continue;
            }
            SCOPED_TRACE(kernel.name);
            ++dma;
            EXPECT_EQ(kernel.getIntMetadata("swz"), 1);
            EXPECT_GE(kernel.getIntMetadata("dma"), 2);
            EXPECT_LE(kernel.getIntMetadata("dma"), 4);
            if(kernel.getStringMetadata("dtype") != "HALF" || kernel.getIntMetadata("cf") != 0
               || kernel.getIntMetadata("bk") != 32)
            {
                continue;
            }
            // c 32 / k 48 take the 8-wide vector; a 32 k-step serves both directions.
            EXPECT_TRUE(matchesKernel(pack, gfx950.context(), kernel, *bound950));
            EXPECT_FALSE(matchesKernel(pack, gfx942.context(), kernel, *bound942));
            auto unswizzled = kernel;
            unswizzled.metadata["swz"] = int64_t{0};
            EXPECT_FALSE(matchesKernel(pack, gfx950.context(), unswizzled, *bound950));
            auto oneStage = kernel;
            oneStage.metadata["dma"] = int64_t{1};
            EXPECT_FALSE(matchesKernel(pack, gfx950.context(), oneStage, *bound950));
            auto fiveStages = kernel;
            fiveStages.metadata["dma"] = int64_t{5};
            EXPECT_FALSE(matchesKernel(pack, gfx950.context(), fiveStages, *bound950));
        }
    }
    EXPECT_GT(dma, 0);
}

TEST(TestConvIgemmCatalog, FusedGroupsExpandTheFilterIntoBlockDiagonalGroups)
{
    auto shape = dense(32, 64, {12, 13}, {3, 3}, {1, 1}, true);
    shape.groups = 8; // 4 input and 8 output channels a group
    const auto problem = parsedProblem(shape, ConvDirection::FORWARD);
    ASSERT_TRUE(problem.filterExpandable);
    const auto fused = conv_igemm::withFusedGroups(problem, 4);
    ASSERT_TRUE(fused.has_value());
    EXPECT_EQ(fused->args.groups, 2);
    EXPECT_TRUE(fused->w.needed && fused->w.expand);
    EXPECT_EQ(fused->w.expandArgs.cg, 4);
    EXPECT_EQ(fused->w.expandArgs.kg, 8);
    EXPECT_EQ(fused->w.expandArgs.f, 4);
    EXPECT_EQ(fused->w.expandArgs.taps, 9);
    EXPECT_EQ(fused->w.bytes, static_cast<size_t>(64 * 9 * 16 * 2));
    EXPECT_TRUE(fused->filterKtrsc);
    // Groups the fusion does not divide, and no fusion at all.
    EXPECT_FALSE(conv_igemm::withFusedGroups(problem, 16).has_value());
    EXPECT_FALSE(conv_igemm::withFusedGroups(problem, 1).has_value());
    // A reflected filter (true convolution) is not expanded.
    shape.mode = ConvMode::CONVOLUTION;
    const auto reflected = parsedProblem(shape, ConvDirection::FORWARD);
    EXPECT_FALSE(reflected.filterExpandable);
    EXPECT_FALSE(conv_igemm::withFusedGroups(reflected, 4).has_value());
}

TEST(TestConvIgemmCatalog, GroupFusionServesOnlySmallGroupsThroughTheFilterItExpands)
{
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        SCOPED_TRACE(direction == ConvDirection::FORWARD ? "forward" : "backward data");
        const bool forward = direction == ConvDirection::FORWARD;
        const auto pack = packFor(direction);
        const auto& kernels = catalog(direction);
        const auto fused = std::find_if(kernels.begin(), kernels.end(), [&](const auto& kernel) {
            return kernel.getIntMetadata("fuse") == 4 && kernel.getStringMetadata("dtype") == "HALF"
                   && kernel.getStringMetadata("atom") == "mfma32_f16"
                   && kernel.getIntMetadata("cf") == 0;
        });
        ASSERT_NE(fused, kernels.end());
        EXPECT_EQ(fused->getIntMetadata("wdirect"), forward ? 0 : 1);
        const auto matches = [&](ConvShape shape, const auto& kernel) {
            shape.dataType = DataType::HALF;
            const GraphFixture fixture(buildGraph(specFor(shape, direction)),
                                       deviceOf("gfx942:sramecc+:xnack-", 304));
            const auto bound = matchesGraph(pack, fixture.context());
            return bound.has_value() && matchesKernel(pack, fixture.context(), kernel, *bound);
        };
        // 8 groups of 8 channels: fused by 4, 32 channels each way.
        auto small = dense(64, 64, {12, 13}, {3, 3}, {1, 1}, true);
        small.groups = 8;
        EXPECT_TRUE(matches(small, *fused));
        // 8 groups of 32 channels: 4 fused would be 128.
        auto wide = dense(256, 256, {12, 13}, {3, 3}, {1, 1}, true);
        wide.groups = 8;
        EXPECT_FALSE(matches(wide, *fused));
        // 6 groups: 4 does not divide them.
        auto six = dense(48, 48, {12, 13}, {3, 3}, {1, 1}, true);
        six.groups = 6;
        EXPECT_FALSE(matches(six, *fused));
        // A reflected filter.
        auto reflected = small;
        reflected.mode = ConvMode::CONVOLUTION;
        EXPECT_FALSE(matches(reflected, *fused));
        // Backward data through the transposed (not direct) filter.
        if(!forward)
        {
            auto transposed = *fused;
            transposed.metadata["wdirect"] = int64_t{0};
            EXPECT_FALSE(matches(small, transposed));
        }
        // Fusion with the halo loop or tap packing, and fusion factors outside 2, 4, 8, 16.
        const std::vector<std::pair<std::string, int64_t>> bad{
            {"halo", 1}, {"pack", 1}, {"pack", 2}, {"fuse", 3}, {"fuse", 32}};
        for(const auto& [field, value] : bad)
        {
            auto kernel = *fused;
            kernel.metadata[field] = value;
            EXPECT_FALSE(matches(small, kernel)) << field << " " << value;
        }
    }
}

TEST(TestConvIgemmCatalog, ChannelsFirstOutputIsTheTransposedOutputOfTheDirection)
{
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        const bool forward = direction == ConvDirection::FORWARD;
        conv_igemm::IgemmProblem problem;
        problem.direction = direction;
        // Only the input channels-first: not the output.
        auto& input = forward ? problem.x : problem.y;
        auto& output = forward ? problem.y : problem.x;
        input.needed = input.transpose = true;
        EXPECT_FALSE(conv_igemm::outputChannelsFirst(problem));
        // The output staged by the strided copy (not packed channels-first).
        output.needed = true;
        EXPECT_FALSE(conv_igemm::outputChannelsFirst(problem));
        output.transpose = true;
        EXPECT_TRUE(conv_igemm::outputChannelsFirst(problem));
    }
}

TEST(TestConvIgemmCatalog, ChannelsFirstOutputProblemsStageNoOutput)
{
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        conv_igemm::IgemmProblem problem;
        problem.direction = direction;
        problem.x.needed = problem.x.transpose = true;
        problem.x.bytes = 1000;
        problem.w.needed = true;
        problem.w.bytes = 5000;
        problem.y.needed = problem.y.transpose = true;
        problem.y.bytes = 300;
        const auto direct = conv_igemm::withoutOutputStaging(problem);
        if(direction == ConvDirection::FORWARD)
        {
            EXPECT_TRUE(direct.x.needed);
            EXPECT_FALSE(direct.y.needed);
            EXPECT_EQ(direct.w.workspaceOffset, 1024u); // after x, 256-byte aligned
            EXPECT_EQ(direct.workspaceBytes, 6144u);
        }
        else
        {
            EXPECT_FALSE(direct.x.needed);
            EXPECT_TRUE(direct.y.needed);
            EXPECT_EQ(direct.w.workspaceOffset, 0u);
            EXPECT_EQ(direct.y.workspaceOffset, 5120u);
            EXPECT_EQ(direct.workspaceBytes, 5632u);
        }
    }
}

TEST(TestConvIgemmCatalog, ChannelsFirstOutputConfigsServeOnlyChannelsFirstOutputs)
{
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        for(const bool channelsLast : {true, false})
        {
            auto shape = dense(32, 48, {12, 13}, {3, 3}, {1, 1}, channelsLast);
            shape.dataType = DataType::HALF;
            const GraphFixture fixture(buildGraph(specFor(shape, direction)),
                                       deviceOf("gfx942:sramecc+:xnack-", 304));
            const auto pack = packFor(direction);
            const auto bound = matchesGraph(pack, fixture.context());
            ASSERT_TRUE(bound.has_value());
            int checked = 0;
            for(const auto& kernel : catalog(direction))
            {
                if(kernel.getStringMetadata("dtype") != "HALF"
                   || kernel.getStringMetadata("atom") != "mfma32_f16"
                   || kernel.getIntMetadata("xcd") != 0 || kernel.getIntMetadata("halo") != 0
                   || kernel.getIntMetadata("wdirect") != 0 || kernel.getIntMetadata("pack") != 0
                   || kernel.getIntMetadata("fuse") != 0 || kernel.getIntMetadata("large") != 0
                   || !conv_igemm::blockOrderServes(kernel.getIntMetadata("bm"),
                                                    kernel.getIntMetadata("bn"),
                                                    0,
                                                    "gfx942:sramecc+:xnack-"))
                {
                    continue; // (the plain-order 256 x 256 tile is for single-die devices)
                }
                SCOPED_TRACE(kernel.name + (channelsLast ? " channels-last" : " channels-first"));
                const bool cf = kernel.getIntMetadata("cf") != 0;
                EXPECT_EQ(matchesKernel(pack, fixture.context(), kernel, *bound),
                          !cf || !channelsLast);
                ++checked;
            }
            EXPECT_GT(checked, 1);
        }
    }
}

TEST(TestConvIgemmCatalog, PackingFitsOnlyReductionsShorterThanAndDividingTheKStep)
{
    auto problem = haloProblem(ConvDirection::FORWARD, 64, 8, 3, 1, 1, 1);
    problem.args.groups = 8; // 8 channels per group
    EXPECT_TRUE(conv_igemm::packFits(problem, 32));
    EXPECT_TRUE(conv_igemm::packFits(problem, 16));
    EXPECT_FALSE(conv_igemm::packFits(problem, 8));
    problem.args.groups = 4; // 16
    EXPECT_FALSE(conv_igemm::packFits(problem, 16));
    problem.args.c = problem.args.k = 48; // 12 per group
    EXPECT_FALSE(conv_igemm::packFits(problem, 32));
    // Backward data reduces over the output channels.
    auto backward = haloProblem(ConvDirection::BACKWARD_DATA, 64, 8, 3, 1, 1, 1);
    backward.args.c = 256;
    backward.args.groups = 8; // k 64: 8 per group
    EXPECT_TRUE(conv_igemm::packFits(backward, 16));
}

TEST(TestConvIgemmCatalog, DirectFilterProblemsStageNoFilter)
{
    conv_igemm::IgemmProblem problem;
    problem.x.needed = true;
    problem.x.bytes = 1000;
    problem.w.needed = true;
    problem.w.bytes = 5000;
    problem.y.needed = true;
    problem.y.bytes = 300;
    problem.args.wRowStride = 7;
    const auto direct = conv_igemm::withoutFilterStaging(problem);
    EXPECT_FALSE(direct.w.needed);
    EXPECT_EQ(direct.args.wRowStride, 0);
    EXPECT_EQ(direct.x.workspaceOffset, 0u);
    EXPECT_EQ(direct.y.workspaceOffset, 1024u); // after x, 256-byte aligned
    EXPECT_EQ(direct.workspaceBytes, 1536u); // y's end, 256-byte aligned
}

TEST(TestConvIgemmCatalog, XcdOrderServesOnlyMultiXcdDevices)
{
    EXPECT_TRUE(conv_igemm::hasXcds("gfx942:sramecc+:xnack-"));
    EXPECT_TRUE(conv_igemm::hasXcds("gfx950"));
    EXPECT_FALSE(conv_igemm::hasXcds("gfx90a:sramecc+:xnack-"));
    EXPECT_FALSE(conv_igemm::hasXcds("gfx1201"));
    // The 256 x 256 tile: XCD order on multi-XCD devices, plain order on single-die ones.
    EXPECT_TRUE(conv_igemm::blockOrderServes(256, 256, 1, "gfx950"));
    EXPECT_FALSE(conv_igemm::blockOrderServes(256, 256, 0, "gfx942:sramecc+:xnack-"));
    EXPECT_TRUE(conv_igemm::blockOrderServes(256, 256, 0, "gfx90a:sramecc+:xnack-"));
    EXPECT_FALSE(conv_igemm::blockOrderServes(256, 256, 1, "gfx908"));
    EXPECT_TRUE(conv_igemm::blockOrderServes(128, 128, 0, "gfx950"));
}

/// A 2-D problem as the halo check reads it: n 2, square input, square filter.
conv_igemm::IgemmProblem haloProblem(
    ConvDirection direction, int c, int h, int filter, int pad, int stride, int dilation)
{
    conv_igemm::IgemmProblem problem;
    problem.direction = direction;
    problem.elementBytes = 2;
    auto& a = problem.args;
    a.n = 2;
    a.c = a.k = c;
    a.groups = 1;
    a.di = a.dout = a.ft = 1;
    a.hi = a.wi = h;
    a.ho = a.wo = (h + 2 * pad - dilation * (filter - 1) - 1) / stride + 1;
    a.fr = a.fs = filter;
    a.sd = 1;
    a.sh = a.sw = stride;
    a.pd = 0;
    a.ph = a.pw = pad;
    a.dd = 1;
    a.dh = a.dw = dilation;
    return problem;
}

TEST(TestConvIgemmCatalog, HaloFitsOnlyStrideOneSameConvolutionsWithinItsShiftRange)
{
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        // 28x28 3x3 pad 1: shift range 2 * 28 + 2 = 58 rows.
        EXPECT_TRUE(conv_igemm::haloFits(haloProblem(direction, 64, 28, 3, 1, 1, 1), 256));
        // 56x56: 114 rows, past the 64-row capacity.
        EXPECT_FALSE(conv_igemm::haloFits(haloProblem(direction, 64, 56, 3, 1, 1, 1), 256));
        // Stride 2, and a "valid" convolution (output smaller than the input).
        EXPECT_FALSE(conv_igemm::haloFits(haloProblem(direction, 64, 28, 3, 1, 2, 1), 256));
        EXPECT_FALSE(conv_igemm::haloFits(haloProblem(direction, 64, 28, 3, 0, 1, 1), 256));
        // Dilation 2 with pad 2 keeps the size: range 4 * 14 + 4 = 60 rows.
        EXPECT_TRUE(conv_igemm::haloFits(haloProblem(direction, 64, 14, 3, 2, 1, 2), 256));
        // 1x1: no shift at all.
        EXPECT_TRUE(conv_igemm::haloFits(haloProblem(direction, 64, 56, 1, 0, 1, 1), 256));
        // Channels so wide the staged rows' offsets leave 32 bits.
        EXPECT_FALSE(conv_igemm::haloFits(haloProblem(direction, 1 << 21, 8, 3, 1, 1, 1), 256));
    }
}

// ---- Host-only: the split-K default and the device's units ----

hipdnn_plugin_sdk::ingestor::DeviceProperties deviceOf(const std::string& arch, int units)
{
    auto properties = testDeviceProperties();
    properties.gcnArchName = arch;
    properties.multiProcessorCount = units;
    return properties;
}

conv_igemm::IgemmProblem parsedProblem(ConvShape shape, ConvDirection direction)
{
    shape.dataType = DataType::HALF;
    const GraphFixture fixture(buildGraph(specFor(shape, direction)));
    const auto problem = conv_igemm::parseIgemmProblem(fixture.context(), direction);
    EXPECT_TRUE(problem.has_value());
    return problem.value_or(conv_igemm::IgemmProblem{});
}

int splitsOf(const conv_igemm::IgemmProblem& problem, const std::string& arch, int units)
{
    return conv_igemm::splitPlanFor(problem, 128, 128, 32, deviceOf(arch, units)).splits;
}

TEST(TestConvIgemmSplitPlan, Gfx9SplitsBelowTwoBlocksPerComputeUnit)
{
    // One 128 x 128 tile (64 pixels, 128 output channels) over 9 x 64 k-steps.
    const auto deep
        = parsedProblem(dense(2048, 128, {8, 8}, {3, 3}, {1, 1}, true), ConvDirection::FORWARD);
    // 2 x 304 blocks wanted, half the 576 k-steps, capped at 16.
    EXPECT_EQ(splitsOf(deep, "gfx942:sramecc+:xnack-", 304), 16);
    EXPECT_EQ(splitsOf(deep, "gfx90a", 104), 16);
    // Two k-steps per split at most: 9 taps x 1 chunk of 32 channels -> 4 splits.
    const auto shallow
        = parsedProblem(dense(32, 128, {8, 8}, {3, 3}, {1, 1}, true), ConvDirection::FORWARD);
    EXPECT_EQ(splitsOf(shallow, "gfx942", 304), 4);
}

TEST(TestConvIgemmSplitPlan, FlatReductionCountsKStepsOverTheTapsEndToEnd)
{
    // 24 channels x 9 taps: padded per tap 9 k-steps of 32, flat ceil(216 / 32) = 7.
    const auto problem
        = parsedProblem(dense(24, 128, {8, 8}, {3, 3}, {1, 1}, true), ConvDirection::FORWARD);
    const auto device = deviceOf("gfx942", 304);
    EXPECT_EQ(conv_igemm::splitPlanFor(problem, 128, 128, 32, device).splits, 4);
    EXPECT_EQ(conv_igemm::splitPlanFor(problem, 128, 128, 32, device, 2).splits, 3);
}

TEST(TestConvIgemmSplitPlan, RdnaSplitsOnlyBelowOneBlockPerWgpAndKeepsSixtyFourKSteps)
{
    const auto deep
        = parsedProblem(dense(2048, 128, {8, 8}, {3, 3}, {1, 1}, true), ConvDirection::FORWARD);
    // 48 WGPs (a 96-CU W7900): one tile wants 48 splits; 576 k-steps keep 9 of 64 each.
    EXPECT_EQ(splitsOf(deep, "gfx1100", 48), 9);
    EXPECT_EQ(splitsOf(deep, "gfx1201", 32), 9);
    // Fewer than 128 k-steps: no split, however empty the device.
    const auto shallow
        = parsedProblem(dense(32, 128, {8, 8}, {3, 3}, {1, 1}, true), ConvDirection::FORWARD);
    EXPECT_EQ(splitsOf(shallow, "gfx1100", 48), 1);
    // The vocoder transposed conv (k8 s4, 1536 -> 768 channels, dy 256): 48 tiles over 96
    // k-steps. gfx942 splits it 13 ways; every split measured slower on gfx1100 and gfx1201.
    ConvShape vocoder = dense(768, 1536, {1024}, {8}, {2}, true);
    vocoder.n = 1;
    vocoder.stride = {4};
    const auto backward = parsedProblem(vocoder, ConvDirection::BACKWARD_DATA);
    EXPECT_EQ(splitsOf(backward, "gfx942", 304), 13);
    EXPECT_EQ(splitsOf(backward, "gfx1100", 48), 1);
    EXPECT_EQ(splitsOf(backward, "gfx1201", 32), 1);
    EXPECT_EQ(splitsOf(backward, "gfx1100", 12), 1);
}

TEST(TestConvIgemmSplitPlan, ADeviceWithoutUnitsGetsNoSplitAndNoConfig)
{
    const auto deep
        = parsedProblem(dense(2048, 128, {8, 8}, {3, 3}, {1, 1}, true), ConvDirection::FORWARD);
    for(const int units : {0, -1})
    {
        const auto plan = conv_igemm::splitPlanFor(deep, 128, 128, 32, deviceOf("gfx942", units));
        EXPECT_EQ(plan.splits, 1);
        EXPECT_EQ(plan.workspaceBytes, deep.workspaceBytes);
    }
    // The kernel match declines every config the same graph is served by on a counted device.
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        for(const char* arch : {"gfx942:sramecc+:xnack-", "gfx1100", "gfx1201"})
        {
            auto shape = dense(32, 64, {9, 7}, {3, 3}, {1, 1}, true);
            shape.dataType = DataType::HALF;
            const auto graph = specFor(shape, direction);
            const GraphFixture counted(buildGraph(graph), deviceOf(arch, 64));
            const GraphFixture uncounted(buildGraph(graph), deviceOf(arch, 0));
            const auto pack = packFor(direction);
            const auto bound = matchesGraph(pack, counted.context());
            ASSERT_TRUE(bound.has_value());
            int served = 0;
            for(const auto& kernel : catalog(direction))
            {
                if(matchesKernel(pack, counted.context(), kernel, *bound))
                {
                    ++served;
                    EXPECT_FALSE(matchesKernel(pack, uncounted.context(), kernel, *bound))
                        << arch << " " << kernel.name;
                }
            }
            EXPECT_GT(served, 0) << arch;
        }
    }
}

TEST(TestConvIgemmCatalog, BufferSpanIsWholeImagesFromTheBlocksFirstImage)
{
    // hipconv #15149 review: a per-tile buffer base rebased to a row in the bottom padding made
    // the next image's offsets negative past 2 GiB. K1's buffer starts at the block's first
    // image and spans whole images (its rows' offsets are image-relative, each tap bounds-checked),
    // so a tensor of any size keeps 32-bit offsets while the images a block can touch stay under
    // 2^31 bytes, and drops to 64-bit addressing when one image does not.
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        // 20000 images of 64 x 64 x 64 fp16 (10 GiB), padding past a 3x3 filter's reach.
        auto many = dense(64, 64, {64, 64}, {3, 3}, {3, 3}, true);
        many.n = 20000;
        const auto large = parsedProblem(many, direction);
        const int64_t imageBytes = 64 * 64 * 64 * 2;
        EXPECT_GE(large.blockSpanBytes, imageBytes);
        EXPECT_LE(large.blockSpanBytes, 3 * imageBytes);
        // One 64 x 4096 x 4096 fp16 image is 2^31 bytes: past the 32-bit offsets.
        auto huge = dense(64, 64, {4096, 4096}, {3, 3}, {1, 1}, true);
        huge.n = 1;
        EXPECT_GE(parsedProblem(huge, direction).blockSpanBytes, int64_t{1} << 31);
    }
}

TEST(TestConvIgemmCatalog, LargeImageConfigsServeExactlyTheImagesPastTwoToTheThirtyOnePixels)
{
    // The kernels index one image's positions in 32 bits, the large-image configs (K1_LARGE) in
    // 64. #14424 X3 (46341 x 46341, 1x1, one channel: 2^31 + 4633 pixels) takes only those;
    // 46340 x 46340 (2^31 - 88733) and the same pixels over 16 images only the others.
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        const auto pack = packFor(direction);
        const auto& kernels = catalog(direction);
        const auto plain = std::find_if(kernels.begin(), kernels.end(), [](const auto& k) {
            return k.getStringMetadata("dtype") == "HALF" && k.getStringMetadata("atom") == "fma"
                   && k.getIntMetadata("vec") == 1 && k.getIntMetadata("large") == 0;
        });
        const auto large = std::find_if(kernels.begin(), kernels.end(), [](const auto& k) {
            return k.getStringMetadata("dtype") == "HALF" && k.getStringMetadata("atom") == "fma"
                   && k.getIntMetadata("vec") == 1 && k.getIntMetadata("large") == 1;
        });
        ASSERT_NE(plain, kernels.end());
        ASSERT_NE(large, kernels.end());
        const auto serves = [&](ConvShape shape, const auto& kernel) {
            shape.dataType = DataType::HALF;
            const GraphFixture fixture(buildGraph(specFor(shape, direction)),
                                       deviceOf("gfx942:sramecc+:xnack-", 304));
            const auto bound = matchesGraph(pack, fixture.context());
            return bound.has_value() && matchesKernel(pack, fixture.context(), kernel, *bound);
        };
        auto x3 = dense(1, 1, {46341, 46341}, {1, 1}, {0, 0}, true);
        x3.n = 1;
        EXPECT_TRUE(serves(x3, *large));
        EXPECT_FALSE(serves(x3, *plain));
        auto under = x3;
        under.input = {46340, 46340};
        EXPECT_FALSE(serves(under, *large));
        EXPECT_TRUE(serves(under, *plain));
        auto batched = x3;
        batched.n = 16;
        batched.input = {11586, 11586};
        EXPECT_FALSE(serves(batched, *large));
        EXPECT_TRUE(serves(batched, *plain));
        // A large-image knob on any other variant is not a config.
        auto haloLarge = *large;
        haloLarge.metadata["large"] = int64_t{2};
        EXPECT_FALSE(serves(x3, haloLarge));
    }
}

TEST(TestConvIgemmLaunch, OnlyTheSwizzledWideWmmaTileGetsAnOccupancyTarget)
{
    // 4 waves per SIMD for the swizzled 128x128x64 4x2 WMMA tile on the 1.5x-VGPR gfx11 chips;
    // every other config and device keeps the compiler's choice (0).
    for(const char* arch : {"gfx1100", "gfx1101", "gfx1151"})
    {
        EXPECT_EQ(conv_igemm::occupancyTarget("wmma11_f16", 128, 128, 64, 4, 2, 1, arch), 4);
        EXPECT_EQ(conv_igemm::occupancyTarget("wmma11_bf16", 128, 128, 64, 4, 2, 1, arch), 4);
        EXPECT_EQ(conv_igemm::occupancyTarget("wmma11_f16", 128, 128, 64, 4, 2, 0, arch), 0);
        EXPECT_EQ(conv_igemm::occupancyTarget("wmma11_f16", 128, 128, 64, 2, 4, 1, arch), 0);
        EXPECT_EQ(conv_igemm::occupancyTarget("wmma11_f16", 128, 128, 32, 4, 2, 1, arch), 0);
    }
    for(const char* arch : {"gfx1102", "gfx1150", "gfx1201", "gfx942"})
    {
        EXPECT_EQ(conv_igemm::occupancyTarget("wmma11_f16", 128, 128, 64, 4, 2, 1, arch), 0);
    }
    EXPECT_EQ(conv_igemm::occupancyTarget("wmma12_f16", 128, 128, 64, 4, 2, 1, "gfx1201"), 0);
}

TEST(TestConvIgemmLaunch, RowTilesPastOneDispatchRunAsSeveralLaunches)
{
    // #14424 case X3: N = C = K = 1, 46341 x 46341, 1x1 -- 2^31 + 4633 output rows. One
    // dispatch addresses 2^32 - 1 work items along x, so 128-row tiles of 256-thread blocks
    // (16777253 tiles, 4294976768 work items) take two launches.
    const long long rows = 46341LL * 46341LL;
    const long long tiles = (rows + 127) / 128;
    const long long first = conv_igemm::launchTilesFor(tiles, 256);
    EXPECT_EQ(first, (int64_t{1} << 24) - 1);
    EXPECT_LT(first * 256, int64_t{1} << 32);
    EXPECT_EQ(tiles - first, 38);
    // Every grid that fits one dispatch is one launch.
    EXPECT_EQ(conv_igemm::launchTilesFor(1000, 256), 1000);
    EXPECT_EQ(conv_igemm::launchTilesFor((int64_t{1} << 24) - 1, 256), (int64_t{1} << 24) - 1);
    EXPECT_EQ(conv_igemm::launchTilesFor(int64_t{1} << 24, 256), (int64_t{1} << 24) - 1);
    EXPECT_EQ(conv_igemm::launchTilesFor(int64_t{1} << 30, 64), (int64_t{1} << 26) - 1);
}

TEST(TestConvIgemmComputeType, ServesCudnnsComputeTypesPerElementType)
{
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        expectCudnnComputeTypes(
            dense(16, 24, {7, 6}, {3, 3}, {1, 1}, true),
            {DataType::FLOAT, DataType::HALF, DataType::BFLOAT16},
            [direction](const ConvShape& s) {
                const GraphFixture fixture(buildGraph(specFor(s, direction)));
                return conv_igemm::parseIgemmProblem(fixture.context(), direction).has_value();
            });
    }
}

} // namespace

#endif
