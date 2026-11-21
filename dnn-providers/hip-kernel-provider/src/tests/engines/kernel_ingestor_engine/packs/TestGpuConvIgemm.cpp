// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceConvolution.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <nlohmann/json.hpp>

#include "compilation/ProgramFetchRecording.hpp"
#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvDeterminismTestSupport.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvEpilogueDeviceTestSupport.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvEpilogueTiming.hpp"
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
        // Each direction's exact, TF32 and architecture-qualified catalogs.
        for(size_t i = 0; i < catalogs.size(); ++i)
        {
            const std::string prefix = i == 0 ? "conv_igemm_fwd" : "conv_igemm_bwd";
            const std::string suffix = ".kdp.json";
            std::vector<std::filesystem::path> found;
            for(const auto& root : roots)
            {
                std::error_code error;
                for(auto it = std::filesystem::recursive_directory_iterator(root, error);
                    !error && it != std::filesystem::recursive_directory_iterator();
                    it.increment(error))
                {
                    const auto file = it->path().filename().string();
                    if(file.rfind(prefix, 0) == 0 && file.size() >= suffix.size()
                       && file.compare(file.size() - suffix.size(), suffix.size(), suffix) == 0)
                    {
                        found.push_back(it->path());
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
            std::sort(found.begin(), found.end());
            catalogs[i] = readDistinctEmbeddedKernelCatalog(found);
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
                const std::vector<int64_t>& pad,
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

/// A patch convolution: filter @p f on every axis at stride f, no padding.
ConvShape patchShape(int64_t c, int64_t k, std::vector<int64_t> input, int64_t f, bool channelsLast)
{
    const std::vector<int64_t> filter(input.size(), f);
    const std::vector<int64_t> pad(input.size(), 0);
    auto shape = dense(c, k, std::move(input), filter, pad, channelsLast);
    shape.stride.assign(shape.input.size(), f);
    return shape;
}

std::vector<DeviceCase> patchCases()
{
    std::vector<DeviceCase> cases;
    cases.push_back({"NhwcPatch4x4Rgb", patchShape(3, 24, {12, 20}, 4, true)});
    cases.push_back({"NhwcPatch16x16Rgb", patchShape(3, 40, {32, 48}, 16, true)});
    cases.push_back({"NhwcPatch2x2", patchShape(16, 24, {6, 10}, 2, true)});
    cases.push_back({"NchwPatch2x2", patchShape(16, 24, {6, 10}, 2, false)});
    cases.push_back({"NdhwcPatch2x2x2", patchShape(8, 16, {4, 6, 8}, 2, true)});
    cases.push_back({"NhwcPatch1d", patchShape(8, 16, {40}, 4, true)});
    auto convolution = patchShape(16, 24, {6, 10}, 2, true);
    convolution.mode = ConvMode::CONVOLUTION;
    cases.push_back({"NhwcPatch2x2ConvolutionMode", convolution});
    cases.push_back({"NhwcPatch2x2RaggedWidth", patchShape(16, 16, {8, 11}, 2, true)});
    return cases;
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
    // Pointwise layers with a narrow side (6 channels): forward reduces 6 channels into 64,
    // backward data 6 into 64 the other way, so a 1-wide reduction vector writes wide rows.
    cases.push_back({"Nhwc1x1NarrowReduction", dense(64, 6, {9, 11}, {1, 1}, {0, 0}, true)});
    cases.push_back({"Nchw1x1NarrowReduction", dense(64, 6, {9, 11}, {1, 1}, {0, 0}, false)});

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

    // Patch convolutions (filter equal to the stride, no padding): the patch configs. RGB runs
    // of 4 x 3 and 16 x 3 channels (4- and 8-wide vectors; 48 is a flat reduction for 32- and
    // 64-wide k-steps), wide channels, 3-D, 1-D, channels-first (staged), true convolution, and
    // a width the stride does not divide (served by the plain configs only).
    for(const auto& patchCase : patchCases())
    {
        cases.push_back(patchCase);
    }
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

/// Which configs a case must run: any, or at least one patch config (and only those), or at least
/// one channels-first-input config (and only those), or the flex or col2im configs alone.
enum class Configs
{
    ANY,
    PATCH,
    CFX,
    FLEX,
    COL2IM,
};

template <typename T>
void runCase(ConvShape shape,
             ConvDirection direction,
             Math math = Math::NATIVE,
             bool infinities = false,
             Configs configs = Configs::ANY,
             unsigned int seed = 1234,
             bool noSplitOnly = false,
             bool channelsFirstInputOnly = false,
             bool filterInfinities = false)
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
    const bool forward = direction == ConvDirection::FORWARD;
    (forward ? x : y).fillTensorWithRandomValues(-1.0f, 1.0f, seed);
    w.fillTensorWithRandomValues(-1.0f, 1.0f, seed + 1);
    if(infinities)
    {
        plantInfinities(forward ? x : y);
    }
    if(filterInfinities)
    {
        plantInfinities(w);
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
        if(noSplitOnly && kernel.getIntMetadata("split_policy") != 1)
        {
            continue;
        }
        if(channelsFirstInputOnly && kernel.getIntMetadata("cf") != 2)
        {
            continue;
        }
        if(!matchesKernel(pack, fixture.context(), kernel, *bound)
           || (xf32 || split) != (math == Math::TF32))
        {
            continue;
        }
        // Known open parity item: group fusion's zero filter blocks multiply the other
        // groups' infinities into NaN (0 x inf), so fused configs are left out of the
        // infinity runs until that is fixed.
        if(infinities && kernel.getIntMetadata("fuse") != 0)
        {
            continue;
        }
        if((configs == Configs::PATCH && kernel.getIntMetadata("patch") == 0)
           || (configs == Configs::COL2IM && kernel.getIntMetadata("patch") != 3)
           || (configs == Configs::CFX && kernel.getIntMetadata("cfx") == 0)
           || (configs == Configs::FLEX && kernel.getIntMetadata("flex") == 0))
        {
            continue;
        }
        // xf32 rounds each operand to 10 mantissa bits (about 2^-11 of a product each);
        // three-bf16 drops lo x lo (about 2^-16).
        float perTermBound = 2e-6f;
        if(xf32)
        {
            perTermBound = 1e-3f;
        }
        else if(split)
        {
            perTermBound = 3e-5f;
        }
        const float accumulationBound = perTermBound * static_cast<float>(reductionLength);
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
            written,
            workspace,
            workspaceBytes));
        if(workspace != nullptr)
        {
            ASSERT_EQ(hipFree(workspace), hipSuccess);
        }
        written.markDeviceModified();

        const auto* actual = static_cast<const T*>(written.rawHostData());
        const auto* expected = static_cast<const T*>(reference.rawHostData());
        int64_t mismatches = 0;
        forEachIndex(written.dims(), [&](const std::vector<int64_t>& index) {
            const auto got = static_cast<float>(actual[offsetOf(index, written.strides())]);
            const auto want = static_cast<float>(expected[offsetOf(index, reference.strides())]);
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

/// Patch convolutions through the patch configs only (every architecture has them: the FMA kernel
/// takes every type).
class TestGpuConvIgemmPatch : public ::testing::TestWithParam<std::tuple<DeviceCase, ConvDirection>>
{
};

// The patch configs alone, with +inf and -inf in the input: every output an infinity reaches is
// that infinity (NaN where both meet), as the reference gives. No padded run or ragged lane may
// multiply a zero into an infinity.
TEST_P(TestGpuConvIgemmPatch, FloatInfinities)
{
    SKIP_IF_NO_DEVICES();
    runCase<float>(
        std::get<0>(GetParam()).shape, std::get<1>(GetParam()), Math::NATIVE, true, Configs::PATCH);
}

TEST_P(TestGpuConvIgemmPatch, HalfInfinities)
{
    SKIP_IF_NO_DEVICES();
    runCase<Half>(
        std::get<0>(GetParam()).shape, std::get<1>(GetParam()), Math::NATIVE, true, Configs::PATCH);
}

TEST_P(TestGpuConvIgemmPatch, BFloat16Infinities)
{
    SKIP_IF_NO_DEVICES();
    runCase<BFloat16>(
        std::get<0>(GetParam()).shape, std::get<1>(GetParam()), Math::NATIVE, true, Configs::PATCH);
}

std::vector<DeviceCase> servedPatchCases()
{
    std::vector<DeviceCase> cases;
    for(const auto& patchCase : patchCases())
    {
        if(patchCase.name != "NhwcPatch2x2RaggedWidth")
        {
            cases.push_back(patchCase);
        }
    }
    return cases;
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuConvIgemmPatch,
                         ::testing::Combine(::testing::ValuesIn(servedPatchCases()),
                                            ::testing::Values(ConvDirection::FORWARD,
                                                              ConvDirection::BACKWARD_DATA)),
                         [](const auto& info) {
                             return std::get<0>(info.param).name
                                    + (std::get<1>(info.param) == ConvDirection::FORWARD ? "Fwd"
                                                                                         : "Bwd");
                         });

/// Col2im backward data (patch 3): layers whose dy has fewer positions than dx ("valid" or partly
/// padded), as one GEMM over dy's positions plus the column sum into dx. 3-D in both layouts
/// (channels-first: dy and dx staged, the K C T R S filter's columns in its own order), two
/// images, padding on one side only, stride 2 (taps that skip dx positions, rows no tap reaches),
/// dilation, six channels (not whole vectors of four) in both layouts, and a 1-D filter.
std::vector<DeviceCase> col2imCases()
{
    std::vector<DeviceCase> cases;
    cases.push_back({"Ndhwc3x3x3Valid", dense(16, 24, {6, 5, 7}, {3, 3, 3}, {0, 0, 0}, true)});
    cases.push_back({"Ncdhw3x3x3Valid", dense(16, 24, {6, 5, 7}, {3, 3, 3}, {0, 0, 0}, false)});
    auto partly = dense(24, 32, {9, 8}, {3, 3}, {0, 1}, true);
    partly.postPadding = {0, 0};
    cases.push_back({"Nhwc3x3PaddedOneSide", partly});
    auto strided = dense(16, 16, {13, 12}, {3, 3}, {0, 0}, true);
    strided.stride = {2, 2};
    cases.push_back({"Nhwc3x3Stride2Valid", strided});
    auto dilated = dense(16, 24, {11, 12}, {3, 3}, {0, 0}, true);
    dilated.dilation = {2, 2};
    cases.push_back({"Nhwc3x3Dilation2Valid", dilated});
    cases.push_back({"Nhwc2x2SixChannels", dense(6, 24, {7, 6}, {2, 2}, {0, 0}, true)});
    cases.push_back({"Nchw2x2SixChannels", dense(6, 24, {7, 6}, {2, 2}, {0, 0}, false)});
    cases.push_back({"Nhwc1dK5Valid", dense(16, 24, {12}, {5}, {0}, true)});
    return cases;
}

/// Infinities in dy (the reduction's input) or in the filter. Col2im sums only the products of dy
/// positions that exist, as the reference defines backward data: an infinite filter tap meets no
/// padding, so dx elements whose taps reach past dy stay finite (a GEMM over dx rows multiplies
/// the padding's zeros by the filter, 0 x inf = NaN, which is why no other suite plants infinities
/// in the filter).
enum class Infinities
{
    NONE,
    DY,
    FILTER,
};

class TestGpuConvIgemmCol2im : public ::testing::TestWithParam<std::tuple<DeviceCase, Infinities>>
{
protected:
    template <typename T>
    void run()
    {
        SKIP_IF_NO_DEVICES();
        // The col2im configs run on matrix cores: none take fp32 on RDNA or anything on gfx906.
        const auto device = currentDeviceProperties();
        const auto dtype
            = std::string(hipdnn_flatbuffers_sdk::data_objects::EnumNameDataType(dataTypeOf<T>()));
        bool served = false;
        for(const auto& kernel : catalog(ConvDirection::BACKWARD_DATA))
        {
            served = served
                     || (kernel.getIntMetadata("patch") == 3
                         && kernel.getStringMetadata("dtype") == dtype
                         && conv_igemm::atomRunsOn(kernel.getStringMetadata("atom"),
                                                   device.gcnArchName));
        }
        if(!served)
        {
            GTEST_SKIP() << "no col2im config runs this element type on this device";
        }
        const auto& [deviceCase, infinities] = GetParam();
        runCase<T>(deviceCase.shape,
                   ConvDirection::BACKWARD_DATA,
                   Math::NATIVE,
                   infinities == Infinities::DY,
                   Configs::COL2IM,
                   1234,
                   false,
                   false,
                   infinities == Infinities::FILTER);
    }
};

TEST_P(TestGpuConvIgemmCol2im, Float)
{
    run<float>();
}

TEST_P(TestGpuConvIgemmCol2im, Half)
{
    run<Half>();
}

TEST_P(TestGpuConvIgemmCol2im, BFloat16)
{
    run<BFloat16>();
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestGpuConvIgemmCol2im,
    ::testing::Combine(::testing::ValuesIn(col2imCases()),
                       ::testing::Values(Infinities::NONE, Infinities::DY, Infinities::FILTER)),
    [](const auto& info) {
        const auto inf = std::get<1>(info.param);
        return std::get<0>(info.param).name
               + (inf == Infinities::DY       ? "DyInfinities"
                  : inf == Infinities::FILTER ? "FilterInfinities"
                                              : "");
    });

/// 1-D channels-first (N C L) graphs: PyTorch's only conv1d layout, read in place by the
/// channels-first-input configs (cfx). Lengths that 8 does not divide (vectors straddling the
/// row end, unaligned rows), two images (blocks crossing an image boundary), "same", "valid" and
/// strided geometry, dilation, groups, a pointwise layer, an output of extent 1, a row shorter
/// than a vector, true convolution.
std::vector<DeviceCase> channelsFirst1dCases()
{
    std::vector<DeviceCase> cases;
    cases.push_back({"NclK3Same", dense(24, 40, {100}, {3}, {1}, false)});
    // The same layer as a 2-D graph of height 1 (how cuDNN-style callers send conv1d).
    cases.push_back({"Nchw1xLK3Same", dense(24, 40, {1, 100}, {1, 3}, {0, 1}, false)});
    auto dilated = dense(32, 32, {128}, {11}, {15}, false);
    dilated.dilation = {3};
    cases.push_back({"NclK11Dilation3", dilated});
    cases.push_back({"NclK7Valid", dense(16, 24, {64}, {7}, {0}, false)});
    auto stride2 = dense(16, 24, {101}, {3}, {1}, false);
    stride2.stride = {2};
    cases.push_back({"NclK3Stride2", stride2});
    auto k2s2 = dense(32, 16, {99}, {2}, {0}, false);
    k2s2.stride = {2};
    cases.push_back({"NclK2Stride2", k2s2});
    cases.push_back({"NclPointwise", dense(64, 32, {200}, {1}, {0}, false)});
    auto grouped = dense(32, 48, {50}, {5}, {2}, false);
    grouped.groups = 4;
    cases.push_back({"NclK5Groups4", grouped});
    auto toOne = dense(16, 24, {4}, {3}, {0}, false);
    toOne.stride = {2};
    cases.push_back({"NclStride2OutputOne", toOne});
    cases.push_back({"NclShortRow", dense(8, 16, {3}, {3}, {1}, false)});
    // Pointwise 2-D and 3-D channels-first layers, as the 1-D line of each channel's pixels.
    cases.push_back({"Nchw1x1Line", dense(64, 32, {12, 13}, {1, 1}, {0, 0}, false)});
    cases.push_back({"Ncdhw1x1x1Line", dense(16, 24, {3, 5, 7}, {1, 1, 1}, {0, 0, 0}, false)});
    auto convolution = dense(24, 40, {40}, {3}, {1}, false);
    convolution.mode = ConvMode::CONVOLUTION;
    cases.push_back({"NclK3ConvolutionMode", convolution});
    return cases;
}

class TestGpuConvIgemmChannelsFirst1d
    : public ::testing::TestWithParam<std::tuple<DeviceCase, ConvDirection>>
{
};

TEST_P(TestGpuConvIgemmChannelsFirst1d, Float)
{
    SKIP_IF_NO_DEVICES();
    runCase<float>(
        std::get<0>(GetParam()).shape, std::get<1>(GetParam()), Math::NATIVE, false, Configs::CFX);
}

TEST_P(TestGpuConvIgemmChannelsFirst1d, Half)
{
    SKIP_IF_NO_DEVICES();
    runCase<Half>(
        std::get<0>(GetParam()).shape, std::get<1>(GetParam()), Math::NATIVE, false, Configs::CFX);
}

TEST_P(TestGpuConvIgemmChannelsFirst1d, BFloat16)
{
    SKIP_IF_NO_DEVICES();
    runCase<BFloat16>(
        std::get<0>(GetParam()).shape, std::get<1>(GetParam()), Math::NATIVE, false, Configs::CFX);
}

// +inf and -inf in the gathered input: every output they reach is that infinity (NaN where both
// meet), as the reference gives; padding and the zeroed halo rows never multiply into them.
TEST_P(TestGpuConvIgemmChannelsFirst1d, FloatInfinities)
{
    SKIP_IF_NO_DEVICES();
    runCase<float>(
        std::get<0>(GetParam()).shape, std::get<1>(GetParam()), Math::NATIVE, true, Configs::CFX);
}

TEST_P(TestGpuConvIgemmChannelsFirst1d, HalfInfinities)
{
    SKIP_IF_NO_DEVICES();
    runCase<Half>(
        std::get<0>(GetParam()).shape, std::get<1>(GetParam()), Math::NATIVE, true, Configs::CFX);
}

TEST_P(TestGpuConvIgemmChannelsFirst1d, BFloat16Infinities)
{
    SKIP_IF_NO_DEVICES();
    runCase<BFloat16>(
        std::get<0>(GetParam()).shape, std::get<1>(GetParam()), Math::NATIVE, true, Configs::CFX);
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuConvIgemmChannelsFirst1d,
                         ::testing::Combine(::testing::ValuesIn(channelsFirst1dCases()),
                                            ::testing::Values(ConvDirection::FORWARD,
                                                              ConvDirection::BACKWARD_DATA)),
                         [](const auto& info) {
                             return std::get<0>(info.param).name
                                    + (std::get<1>(info.param) == ConvDirection::FORWARD ? "Fwd"
                                                                                         : "Bwd");
                         });

TEST(TestGpuConvIgemmNoSplit, IndependentlySeededBackwardDataTails)
{
    SKIP_IF_NO_DEVICES();
    if(hipdnn_plugin_sdk::stripArchFeatures(currentDeviceProperties().gcnArchName) != "gfx1250")
    {
        GTEST_SKIP() << "native no-split candidate requires gfx1250";
    }
    for(const int k : {40, 264})
    {
        auto shape = dense(40, k, {7, 9}, {3, 3}, {1, 1}, false);
        shape.n = 1;
        for(const unsigned int seed : {7919u, 104729u})
        {
            runCase<Half>(
                shape, ConvDirection::BACKWARD_DATA, Math::NATIVE, false, Configs::ANY, seed, true);
        }
    }
}

TEST(TestGpuConvIgemmChannelsFirstInput, PointwiseSpatialChannelAndSplitTails)
{
    SKIP_IF_NO_DEVICES();
    if(hipdnn_plugin_sdk::stripArchFeatures(currentDeviceProperties().gcnArchName) != "gfx1250")
    {
        GTEST_SKIP() << "native channels-first input candidate requires gfx1250";
    }
    for(const int64_t c : {40, 520})
    {
        for(const auto& input : {std::vector<int64_t>{40}, {5, 8}, {2, 5, 4}})
        {
            auto shape = dense(c,
                               49,
                               input,
                               std::vector<int64_t>(input.size(), 1),
                               std::vector<int64_t>(input.size(), 0),
                               false);
            shape.n = 1;
            shape.mode = ConvMode::CONVOLUTION;
            runCase<Half>(shape,
                          ConvDirection::FORWARD,
                          Math::NATIVE,
                          false,
                          Configs::ANY,
                          7919u,
                          false,
                          true);
        }
    }
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

/// The flex tile (gfx950 LDS-DMA form, gfx942 register-staged form): flattened-position tiles whose 256 rows straddle planes and images,
/// reduction channels and output channels that are not whole k-steps or tiles, odd depths, depth
/// strides, groups, dilation, filters past 3 taps, and a deep reduction over a tiny grid (split-K).
struct FlexCase
{
    std::string name;
    ConvShape shape;
    bool backward; ///< backward data at unit stride is served too
};

std::vector<FlexCase> flexCases()
{
    std::vector<FlexCase> cases;
    auto odd = dense(72, 136, {5, 9, 11}, {3, 3, 3}, {1, 1, 1}, true);
    odd.n = 3;
    cases.push_back({"Ndhwc3x3x3OddDepthRaggedChannels", odd, true});
    cases.push_back(
        {"Ndhwc3x3x3NoPadding", dense(64, 64, {7, 6, 9}, {3, 3, 3}, {0, 0, 0}, true), true});
    auto depthStride = dense(40, 48, {9, 6, 7}, {3, 3, 3}, {1, 1, 1}, true);
    depthStride.stride = {2, 1, 1};
    cases.push_back({"Ndhwc3x3x3DepthStride2", depthStride, false});
    auto split = dense(512, 256, {3, 4, 5}, {3, 3, 3}, {1, 1, 1}, true);
    split.n = 1;
    cases.push_back({"Ndhwc3x3x3DeepReductionSplit", split, true});
    auto groups = dense(64, 80, {9, 10}, {3, 3}, {1, 1}, true);
    groups.groups = 2;
    cases.push_back({"Nhwc3x3Groups2", groups, true});
    auto dilated = dense(32, 40, {12, 13}, {3, 3}, {2, 2}, true);
    dilated.dilation = {2, 2};
    cases.push_back({"Nhwc3x3Dilation2", dilated, true});
    auto wide = dense(24, 264, {11, 14}, {5, 3}, {0, 2}, true);
    wide.prePadding = {2, 1};
    wide.postPadding = {0, 3};
    cases.push_back({"Nhwc5x3AsymmetricPadding", wide, true});
    auto strided = dense(48, 32, {13, 11}, {3, 3}, {1, 1}, true);
    strided.stride = {2, 2};
    cases.push_back({"Nhwc3x3Stride2", strided, false});
    cases.push_back({"Ndhwc1x1x1", dense(64, 136, {3, 5, 7}, {1, 1, 1}, {0, 0, 0}, true), true});
    auto convolution = dense(32, 40, {6, 7, 5}, {3, 3, 3}, {1, 1, 1}, true);
    convolution.mode = ConvMode::CONVOLUTION;
    cases.push_back({"Ndhwc3x3x3ConvolutionMode", convolution, true});
    cases.push_back(
        {"Ncdhw3x3x3Staged", dense(32, 40, {6, 7, 5}, {3, 3, 3}, {1, 1, 1}, false), true});
    // A few row tiles past whole rounds of 228 and of 304 units (MI300A, MI325X): the
    // register-staged form splits only those (the tail split) on the matching device.
    for(const int64_t rows : {236, 310})
    {
        auto tail = dense(128, 16, {rows, 256}, {3, 3}, {1, 1}, true);
        tail.n = 1;
        cases.push_back({"Nhwc3x3TailSplit" + std::to_string(rows) + "RowTiles", tail, false});
    }
    // 300 row tiles past one round of 256 units (MI355X) with a reduction deep enough to pay a
    // split: the LDS-DMA form splits only the last 44.
    auto deepTail = dense(2048, 16, {300, 256}, {1, 1}, {0, 0}, true);
    deepTail.n = 1;
    cases.push_back({"Nhwc1x1TailSplit300RowTiles", deepTail, false});
    return cases;
}

class TestGpuConvIgemmFlex
    : public ::testing::TestWithParam<std::tuple<FlexCase, ConvDirection, bool>>
{
protected:
    template <typename T>
    void run()
    {
        SKIP_IF_NO_DEVICES();
        // The properties outlive the arch view into their name.
        const auto device = currentDeviceProperties();
        const auto arch = hipdnn_plugin_sdk::stripArchFeatures(device.gcnArchName);
        if(arch != "gfx950" && arch != "gfx942")
        {
            GTEST_SKIP() << "flex tiles run on gfx950 and gfx942 only";
        }
        if(std::is_same_v<T, float> && arch != "gfx942")
        {
            GTEST_SKIP() << "fp32 flex tiles are the gfx942 form's";
        }
        const auto& [flexCase, direction, infinities] = GetParam();
        if(direction == ConvDirection::BACKWARD_DATA && !flexCase.backward)
        {
            GTEST_SKIP() << "strided backward data runs in stride phases, which flex tiles decline";
        }
        runCase<T>(flexCase.shape, direction, Math::NATIVE, infinities, Configs::FLEX);
    }
};

TEST_P(TestGpuConvIgemmFlex, Half)
{
    run<Half>();
}

TEST_P(TestGpuConvIgemmFlex, BFloat16)
{
    run<BFloat16>();
}

TEST_P(TestGpuConvIgemmFlex, Float)
{
    run<float>();
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuConvIgemmFlex,
                         ::testing::Combine(::testing::ValuesIn(flexCases()),
                                            ::testing::Values(ConvDirection::FORWARD,
                                                              ConvDirection::BACKWARD_DATA),
                                            ::testing::Bool()),
                         [](const auto& info) {
                             return std::get<0>(info.param).name
                                    + (std::get<1>(info.param) == ConvDirection::FORWARD ? "Fwd"
                                                                                         : "Bwd")
                                    + (std::get<2>(info.param) ? "Infinities" : "");
                         });

/// Whether the device has TF32-class K1 configs: bf16 matrix cores where three bf16 products
/// are admitted (gfx908 included), and gfx942's xf32.
bool hasTf32Forms()
{
    const auto properties = currentDeviceProperties();
    const std::string arch(hipdnn_plugin_sdk::stripArchFeatures(properties.gcnArchName));
    return conv_igemm::atomRunsOn("mfma16_bf16_908", arch)
           || conv_igemm::atomRunsOn("mfma16_bf16", arch)
           || conv_igemm::atomRunsOn("wmma11_bf16", arch)
           || conv_igemm::atomRunsOn("wmma12_bf16", arch)
           || conv_igemm::atomRunsOn("wmma125_bf16", arch);
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

/// The TF32-class fp32 configs (xf32, three-bf16; the CONV_IGEMM_*_TF32 engines'), against the fp64 reference
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
    runCase<float>(std::get<0>(GetParam()).shape, std::get<1>(GetParam()), Math::TF32);
}

TEST_P(TestGpuConvIgemmTf32, FloatInfinities)
{
    SKIP_IF_NO_DEVICES();
    if(!hasTf32Forms())
    {
        GTEST_SKIP() << "no TF32-class configs on this device";
    }
    runCase<float>(std::get<0>(GetParam()).shape, std::get<1>(GetParam()), Math::TF32, true);
}

TEST_P(TestGpuConvIgemmTf32, FloatInfinitiesNative)
{
    SKIP_IF_NO_DEVICES();
    runCase<float>(std::get<0>(GetParam()).shape, std::get<1>(GetParam()), Math::NATIVE, true);
}

/// Opt-in (K1_TF32_TIMING=<shape CSV>, forward): the best config per math form (exact, xf32,
/// three-bf16) per shape. See ConvTf32Timing.hpp.
TEST(TestGpuConvIgemmTiming, ByMathForm)
{
    SKIP_IF_NO_DEVICES();
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

class TestConvIgemmCatalogReader : public ::testing::Test
{
protected:
    hipdnn_test_sdk::utilities::ScopedDirectory _directory
        = hipdnn_test_sdk::utilities::claimScratchDirectory("igemm-catalog");

    static nlohmann::json record(const std::string& id)
    {
        return {{"version", "1.0"},
                {"id", id},
                {"name", "catalog-contract"},
                {"kernel_source",
                 {{"kind", "embedded_source"},
                  {"source_file", "kernels/ConvIgemm.cpp"},
                  {"entry_point", "ConvIgemm"}}},
                {"metadata", {{"dtype", "HALF"}, {"vec", 8}}},
                {"priority", 1000},
                {"arch", {"gfx942"}}};
    }

    std::filesystem::path shard(const std::string& name, const nlohmann::json& records) const
    {
        const auto path = _directory.path() / (name + ".kdp.json");
        std::ofstream stream;
        stream.exceptions(std::ios::failbit | std::ios::badbit);
        stream.open(path);
        stream << nlohmann::json{{"kernelDescriptors", records}};
        return path;
    }
};

TEST_F(TestConvIgemmCatalogReader, SameIdShardsRetainEveryDistinctKernelAndArch)
{
    const std::string firstId = "00000000-0000-0000-0000-000000000001";
    const std::string secondId = "00000000-0000-0000-0000-000000000002";
    const std::string thirdId = "00000000-0000-0000-0000-000000000003";
    auto first = record(firstId);
    auto second = record(secondId);
    second["name"] = "distinct-id-same-metadata";
    auto third = record(thirdId);
    third["metadata"]["vec"] = 1;
    const auto a = shard("gfx942", nlohmann::json::array({first, second}));
    first["arch"] = {"gfx950"};
    const auto b = shard("gfx950", nlohmann::json::array({first, third}));

    const auto kernels = readDistinctEmbeddedKernelCatalog({a, b});
    ASSERT_EQ(kernels.size(), 3u);
    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::toString(kernels[0].kernelId), firstId);
    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::toString(kernels[1].kernelId), secondId);
    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::toString(kernels[2].kernelId), thirdId);
    EXPECT_EQ(kernels[0].arch, (std::vector<std::string>{"gfx942", "gfx950"}));
    EXPECT_EQ(kernels[1].getIntMetadata("vec"), 8);
    EXPECT_EQ(kernels[2].getIntMetadata("vec"), 1);
}

TEST_F(TestConvIgemmCatalogReader, AnUnqualifiedCopyRemainsUnrestricted)
{
    auto kernel = record("00000000-0000-0000-0000-000000000001");
    const auto a = shard("gfx942", nlohmann::json::array({kernel}));
    kernel.erase("arch");
    const auto b = shard("unqualified", nlohmann::json::array({kernel}));
    kernel["arch"] = {"gfx950"};
    const auto c = shard("gfx950", nlohmann::json::array({kernel}));
    const auto kernels = readDistinctEmbeddedKernelCatalog({a, b, c});
    ASSERT_EQ(kernels.size(), 1u);
    EXPECT_TRUE(kernels[0].arch.empty());
}

TEST_F(TestConvIgemmCatalogReader, ConflictingCopiesFailInsteadOfChoosingTheFirst)
{
    const auto original = record("00000000-0000-0000-0000-000000000001");
    const auto a = shard("first", nlohmann::json::array({original}));
    for(const char* field : {"name", "kernel_source", "metadata", "priority"})
    {
        SCOPED_TRACE(field);
        auto changed = original;
        if(std::string_view(field) == "name")
        {
            changed["name"] = "conflicting-name";
        }
        else if(std::string_view(field) == "kernel_source")
        {
            changed["kernel_source"]["entry_point"] = "ConflictingKernel";
        }
        else if(std::string_view(field) == "metadata")
        {
            changed["metadata"]["vec"] = 4;
        }
        else
        {
            changed["priority"] = 999;
        }
        const auto b = shard("conflicting", nlohmann::json::array({changed}));
        EXPECT_THROW(readDistinctEmbeddedKernelCatalog({a, b}), std::runtime_error);
    }
}

TEST_F(TestConvIgemmCatalogReader, MalformedCopiesAreValidatedBeforeDeduplication)
{
    const auto original = record("00000000-0000-0000-0000-000000000001");
    const auto a = shard("first", nlohmann::json::array({original}));
    for(const char* field : {"id", "kernel_source", "metadata", "priority", "arch"})
    {
        SCOPED_TRACE(field);
        auto malformed = original;
        malformed[field] = nullptr;
        const auto b = shard("malformed", nlohmann::json::array({malformed}));
        EXPECT_THROW(readDistinctEmbeddedKernelCatalog({a, b}), std::exception);
    }
    for(const auto& value : {nlohmann::json(true), nlohmann::json(8.5)})
    {
        auto malformed = original;
        malformed["metadata"]["vec"] = value;
        const auto b = shard("malformed-knob", nlohmann::json::array({malformed}));
        EXPECT_THROW(readDistinctEmbeddedKernelCatalog({a, b}), std::runtime_error);
    }
}

TEST(TestConvIgemmCatalog, EveryAtomIsKnownAndRunsSomewhere)
{
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        for(const auto& kernel : catalog(direction))
        {
            const auto& atom = kernel.getStringMetadata("atom");
            SCOPED_TRACE(kernel.name);
            ASSERT_TRUE(conv_igemm::atomNamed(atom).has_value());
            static constexpr std::array<const char*, 8> ARCHS{
                "gfx908", "gfx90a", "gfx942", "gfx950", "gfx1100", "gfx1151", "gfx1201", "gfx1250"};
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
    EXPECT_FALSE(conv_igemm::atomRunsOn("wmma12_bf16", "gfx1250"));
    for(const auto* atom : {"wmma125_f16", "wmma125_bf16", "wmma125_f32"})
    {
        EXPECT_TRUE(conv_igemm::atomRunsOn(atom, "gfx1250:xnack-"));
        EXPECT_FALSE(conv_igemm::atomRunsOn(atom, "gfx1201"));
        EXPECT_FALSE(conv_igemm::atomRunsOn(atom, "gfx942"));
    }
    EXPECT_TRUE(conv_igemm::atomRunsOn("fma", "gfx906"));
    EXPECT_FALSE(conv_igemm::atomRunsOn("no_such_atom", "gfx942"));
}

TEST(TestConvIgemmCatalog, KnobsTellApartEveryConfigOfAGraph)
{
    // Candidate enumeration identifies a config by its knob tuple: within one direction and
    // dtype (what one graph can admit) no two configs may share every knob.
    static constexpr std::array<const char*, 23> KNOBS{
        "atom", "math",   "bm",    "bn",    "bk",           "wm",   "wn",  "threads",
        "vec",  "stages", "xcd",   "halo",  "wdirect",      "pack", "cf",  "swz",
        "dma",  "fuse",   "large", "patch", "split_policy", "cfx",  "flex"};
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
    static constexpr std::array<const char*, 18> SHAPE{"atom",
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
                                                       "fuse",
                                                       "patch"};
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
            // Channels-first-input configs are twins of channels-first-output ones (their own test);
            // the no-split and channels-first-input-and-output candidates are single configs.
            if(kernel.getIntMetadata("cfx") != 0 || kernel.getIntMetadata("split_policy") != 0
               || kernel.getIntMetadata("cf") == 2)
            {
                continue;
            }
            const bool cf = kernel.getIntMetadata("cf") != 0;
            // Patch backward data writes channels-last dx only; the patch tile has no twins.
            // Flex tiles write channels-last outputs only.
            const bool eligible
                = kernel.getStringMetadata("atom") != "fma" && kernel.getIntMetadata("pack") == 0
                  && kernel.getIntMetadata("patch") != 2 && kernel.getIntMetadata("flex") == 0
                  && (direction != ConvDirection::BACKWARD_DATA
                      || kernel.getIntMetadata("patch") == 0);
            if(cf)
            {
                EXPECT_TRUE(eligible);
                continue;
            }
            const auto tuple = tupleOf(kernel);
            const auto twin = std::count_if(kernels.begin(), kernels.end(), [&](const auto& other) {
                return other.getIntMetadata("cf") == 1 && other.getIntMetadata("cfx") == 0
                       && other.getIntMetadata("split_policy") == 0 && tupleOf(other) == tuple;
            });
            EXPECT_EQ(twin, eligible ? 1 : 0);
            twins += static_cast<int>(twin);
        }
        EXPECT_GT(twins, 0);
    }
}

TEST(TestConvIgemmCatalog, EveryPlainNativeConfigHasAFlatReductionTwin)
{
    static constexpr std::array<const char*, 15> SHAPE{"atom",
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
                                                       "cf",
                                                       "swz",
                                                       "dma",
                                                       "patch"};
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
            // Split-policy twins have flat twins of their own policy.
            return tuple + "|" + std::to_string(kernel.getIntMetadata("split_policy"));
        };
        int flat = 0;
        for(const auto& kernel : kernels)
        {
            SCOPED_TRACE(kernel.name);
            // Patch forward also runs the flat reduction on swizzled tiles (DMA included), for
            // runs such as ViT's 16 x 3 that its k-steps do not divide.
            const bool swizzledPatch
                = kernel.getIntMetadata("patch") != 0 && direction == ConvDirection::FORWARD;
            const bool plain
                = kernel.getIntMetadata("halo") == 0 && kernel.getIntMetadata("wdirect") == 0
                  && (kernel.getIntMetadata("swz") == 0 || swizzledPatch)
                  && kernel.getIntMetadata("cf") == 0 && kernel.getIntMetadata("fuse") == 0
                  && kernel.getIntMetadata("patch") != 2 && kernel.getIntMetadata("flex") == 0
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

/// Fixed native matcher contracts, independent of which GPU catalogs this install staged.
/// Installed-catalog inventory and knob uniqueness are checked separately above.
hipdnn_plugin_sdk::ingestor::KernelDefinition nativeContractKernel(ConvDirection direction)
{
    hipdnn_plugin_sdk::ingestor::KernelDefinition kernel{};
    const bool forward = direction == ConvDirection::FORWARD;
    kernel.name = forward ? "channels-first-input-contract" : "no-split-contract";
    kernel.metadata = {{"direction", std::string(forward ? "FWD" : "BWD")},
                       {"dtype", std::string("HALF")},
                       {"atom", std::string("wmma125_f16")},
                       {"math", std::string("native")},
                       {"bm", int64_t{64}},
                       {"bn", int64_t{64}},
                       {"bk", int64_t{32}},
                       {"wm", int64_t{2}},
                       {"wn", int64_t{2}},
                       {"threads", int64_t{0}},
                       {"vec", int64_t{8}},
                       {"stages", int64_t{1}},
                       {"xcd", int64_t{0}},
                       {"halo", int64_t{0}},
                       {"wdirect", int64_t{0}},
                       {"pack", int64_t{0}},
                       {"cf", int64_t{forward ? 2 : 1}},
                       {"swz", int64_t{0}},
                       {"dma", int64_t{0}},
                       {"fuse", int64_t{0}},
                       {"large", int64_t{0}},
                       {"patch", int64_t{0}},
                       {"split_policy", int64_t{forward ? 0 : 1}},
                       {"cfx", int64_t{0}},
                       {"flex", int64_t{0}}};
    return kernel;
}

TEST(TestConvIgemmCatalog, ChannelsFirstInputRejectsUnsupportedGeometryAndAlignment)
{
    const auto kernel = nativeContractKernel(ConvDirection::FORWARD);
    const auto pack = packFor(ConvDirection::FORWARD);
    const auto base = dense(40, 49, {5, 8}, {1, 1}, {0, 0}, false);
    const auto accepts
        = [&](ConvShape shape, int64_t alignment = 16, int64_t weightAlignment = 16) {
              shape.dataType = DataType::HALF;
              auto spec = specFor(shape, ConvDirection::FORWARD);
              spec.x.alignment = alignment;
              spec.w.alignment = weightAlignment;
              const GraphFixture fixture(buildGraph(spec), deviceOf("gfx1250", 256));
              const auto bound = matchesGraph(pack, fixture.context());
              return bound.has_value() && matchesKernel(pack, fixture.context(), kernel, *bound);
          };
    EXPECT_TRUE(accepts(base));
    EXPECT_FALSE(accepts(base, 2));
    auto changed = base;
    changed.channelsLast = true;
    EXPECT_FALSE(accepts(changed));
    changed = base;
    changed.input = {5, 7};
    EXPECT_FALSE(accepts(changed));
    changed = base;
    changed.filter = {3, 3};
    changed.prePadding = changed.postPadding = {1, 1};
    EXPECT_FALSE(accepts(changed));
    changed = base;
    changed.stride = {2, 2};
    EXPECT_FALSE(accepts(changed));
    changed = base;
    changed.postPadding = {1, 0};
    EXPECT_FALSE(accepts(changed));
    changed = base;
    changed.k = 48;
    changed.groups = 2;
    EXPECT_FALSE(accepts(changed));
    changed = base;
    changed.input = {65536, 65536};
    EXPECT_FALSE(accepts(changed));
    changed = base;
    changed.c = int64_t{2147483640};
    EXPECT_FALSE(accepts(changed)); // The loader's channel cursor must not overflow int.
    changed = base;
    changed.input = {536870912, 1};
    EXPECT_TRUE(accepts(changed)); // Large packed tensors use the 64-bit input loader.
    EXPECT_FALSE(accepts(changed, 16, 2)); // Its vector filter loads also need aligned pointers.
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
            // Swizzled flat reductions: patch forward only.
            if(kernel.getIntMetadata("pack") != 0)
            {
                EXPECT_EQ(kernel.getIntMetadata("pack"), 2);
                EXPECT_EQ(kernel.getIntMetadata("patch"), 1);
                EXPECT_EQ(direction, ConvDirection::FORWARD);
            }
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
            // (Split-policy twins are served only where their plan differs from the default's.)
            if(kernel.getStringMetadata("dtype") != "HALF" || kernel.getIntMetadata("cf") != 0
               || kernel.getIntMetadata("bk") != 32 || kernel.getIntMetadata("patch") != 0
               || kernel.getIntMetadata("split_policy") != 0)
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
        const auto direct = conv_igemm::withoutActivationStaging(problem, false);
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
                   || kernel.getIntMetadata("patch") != 0
                   || kernel.getIntMetadata("split_policy") != 0
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

TEST(TestConvIgemmCatalog, ChannelsFirstInputIsTheTransposedInputOfTheDirection)
{
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        const bool forward = direction == ConvDirection::FORWARD;
        conv_igemm::IgemmProblem problem;
        problem.direction = direction;
        auto& input = forward ? problem.x : problem.y;
        auto& output = forward ? problem.y : problem.x;
        output.needed = output.transpose = true;
        EXPECT_FALSE(conv_igemm::inputChannelsFirst(problem));
        input.needed = true; // staged by the strided copy: not packed channels-first
        EXPECT_FALSE(conv_igemm::inputChannelsFirst(problem));
        input.transpose = true;
        EXPECT_TRUE(conv_igemm::inputChannelsFirst(problem));
    }
}

TEST(TestConvIgemmCatalog, ChannelsFirstInputProblemsStageNoInput)
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
        const auto direct
            = conv_igemm::withoutInputStaging(conv_igemm::withoutActivationStaging(problem, false));
        EXPECT_FALSE(direct.x.needed);
        EXPECT_FALSE(direct.y.needed);
        EXPECT_TRUE(direct.w.needed);
        EXPECT_EQ(direct.w.workspaceOffset, 0u);
        EXPECT_EQ(direct.workspaceBytes, 5120u); // the filter alone
    }
}

conv_igemm::CfxTile cfxTile(int64_t bm, int64_t halo, long long ldsBytes = 64LL * 1024)
{
    conv_igemm::CfxTile tile;
    tile.bm = bm;
    tile.bn = 128;
    tile.bk = 32;
    tile.wm = tile.wn = 2;
    tile.halo = halo;
    tile.ldsBytes = ldsBytes;
    return tile;
}

ConvShape ncl(int64_t length, int64_t filter, int64_t pad, int64_t stride, int64_t dilation)
{
    auto shape = dense(64, 64, {length}, {filter}, {pad}, false);
    shape.stride = {stride};
    shape.dilation = {dilation};
    return shape;
}

TEST(TestConvIgemmCatalog, ChannelsFirstInputFitsOneDimensionalGathersItsHaloHolds)
{
    const auto fwd = ConvDirection::FORWARD;
    const auto bwd = ConvDirection::BACKWARD_DATA;
    // Stride 1: one staged pixel a row.
    EXPECT_TRUE(conv_igemm::cfxFits(parsedProblem(ncl(100, 3, 1, 1, 1), fwd), cfxTile(128, 1)));
    EXPECT_TRUE(conv_igemm::cfxFits(parsedProblem(ncl(100, 3, 1, 1, 1), bwd), cfxTile(128, 1)));
    // Stride 2 forward reaches two pixels a row: halo 2, not halo 1. Backward data gathers dy at
    // unit stride in each phase, so halo 1 holds it.
    EXPECT_FALSE(conv_igemm::cfxFits(parsedProblem(ncl(100, 3, 1, 2, 1), fwd), cfxTile(128, 1)));
    EXPECT_TRUE(conv_igemm::cfxFits(parsedProblem(ncl(100, 3, 1, 2, 1), fwd), cfxTile(128, 2)));
    EXPECT_TRUE(conv_igemm::cfxFits(parsedProblem(ncl(100, 3, 1, 2, 1), bwd), cfxTile(128, 1)));
    // Taps reaching 70 pixels: 127 + 70 + 1 + 7 = 205 rows, past halo 1's 200 (reach 60: 195).
    EXPECT_FALSE(conv_igemm::cfxFits(parsedProblem(ncl(400, 11, 35, 1, 7), fwd), cfxTile(128, 1)));
    EXPECT_TRUE(conv_igemm::cfxFits(parsedProblem(ncl(400, 11, 30, 1, 6), fwd), cfxTile(128, 1)));
    // Outputs past the input's end (a 1-wide filter padded by 2): an image's rows reach past the
    // next image's first pixel, and the run is bounded by the steps, taps and padding.
    EXPECT_TRUE(conv_igemm::cfxFits(parsedProblem(ncl(100, 1, 2, 1, 1), fwd), cfxTile(128, 1)));
    // One row per image at stride 2 (gap 2 per crossing): 254 + 127 x 2 + 2 + 8 = 518 rows, past
    // halo 2's 328.
    EXPECT_FALSE(conv_igemm::cfxFits(parsedProblem(ncl(4, 3, 0, 2, 1), fwd), cfxTile(128, 2)));
    // LDS: 256 rows at halo 2 and 64-deep k-steps need 100 KiB (gfx950's 160 KiB only).
    auto deep = cfxTile(256, 2);
    deep.bk = 64;
    EXPECT_FALSE(conv_igemm::cfxFits(parsedProblem(ncl(1000, 3, 1, 2, 1), fwd), deep));
    deep.ldsBytes = 160LL * 1024;
    EXPECT_TRUE(conv_igemm::cfxFits(parsedProblem(ncl(1000, 3, 1, 2, 1), fwd), deep));
    // The FMA kernel's in-place tile (no halo) serves any stride.
    EXPECT_TRUE(conv_igemm::cfxFits(parsedProblem(ncl(100, 10, 0, 5, 1), fwd), cfxTile(64, 0)));
    // 2-D and 3-D problems are not served.
    EXPECT_FALSE(conv_igemm::cfxFits(
        parsedProblem(dense(64, 64, {12, 13}, {3, 3}, {1, 1}, false), fwd), cfxTile(128, 1)));
    EXPECT_FALSE(conv_igemm::cfxFits(
        parsedProblem(dense(16, 24, {5, 6, 7}, {3, 3, 3}, {1, 1, 1}, false), fwd),
        cfxTile(128, 1)));
}

TEST(TestConvIgemmCatalog, ChannelsFirstInputConfigsServeOnlyOneDimensionalChannelsFirstGraphs)
{
    struct Graph
    {
        const char* name;
        ConvShape shape;
        bool served;
    };
    auto stride5 = ncl(300, 5, 0, 5, 1);
    const std::vector<Graph> graphs{
        {"ncl", ncl(300, 3, 1, 1, 1), true},
        {"nchw height 1", dense(64, 64, {1, 300}, {1, 3}, {0, 1}, false), true},
        {"nlc", dense(64, 64, {300}, {3}, {1}, true), false},
        {"nchw 3x3", dense(64, 64, {12, 13}, {3, 3}, {1, 1}, false), false},
        {"ncdhw", dense(16, 24, {5, 6, 7}, {3, 3, 3}, {1, 1, 1}, false), false},
        // Pointwise, unit stride, unpadded: the 1-D line of D x H x W pixels a channel.
        {"nchw 1x1", dense(64, 64, {12, 13}, {1, 1}, {0, 0}, false), true},
        {"ncdhw 1x1x1", dense(16, 24, {5, 6, 7}, {1, 1, 1}, {0, 0, 0}, false), true},
        {"nhwc 1x1", dense(64, 64, {12, 13}, {1, 1}, {0, 0}, true), false},
        {"nchw 1x1 padded", dense(64, 64, {12, 13}, {1, 1}, {1, 1}, false), false},
    };
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        for(auto graph : graphs)
        {
            graph.shape.dataType = DataType::HALF;
            const GraphFixture fixture(buildGraph(specFor(graph.shape, direction)),
                                       deviceOf("gfx942:sramecc+:xnack-", 304));
            const auto pack = packFor(direction);
            const auto bound = matchesGraph(pack, fixture.context());
            ASSERT_TRUE(bound.has_value());
            int served = 0;
            for(const auto& kernel : catalog(direction))
            {
                if(kernel.getIntMetadata("cfx") != 0
                   && matchesKernel(pack, fixture.context(), kernel, *bound))
                {
                    ++served;
                }
            }
            SCOPED_TRACE(graph.name);
            EXPECT_EQ(served > 0, graph.served);
        }
    }
    // Forward stride 5 reaches five pixels a row, past every halo tile: only the FMA kernel's
    // in-place tile.
    stride5.dataType = DataType::HALF;
    const GraphFixture fixture(buildGraph(specFor(stride5, ConvDirection::FORWARD)),
                               deviceOf("gfx942:sramecc+:xnack-", 304));
    const auto pack = packFor(ConvDirection::FORWARD);
    const auto bound = matchesGraph(pack, fixture.context());
    ASSERT_TRUE(bound.has_value());
    int served = 0;
    for(const auto& kernel : catalog(ConvDirection::FORWARD))
    {
        if(kernel.getIntMetadata("cfx") != 0
           && matchesKernel(pack, fixture.context(), kernel, *bound))
        {
            EXPECT_EQ(kernel.getStringMetadata("atom"), "fma") << kernel.name;
            ++served;
        }
    }
    EXPECT_GT(served, 0);
}

TEST(TestConvIgemmCatalog, PointwiseChannelsFirstProblemsViewAsTheirOneDimensionalLine)
{
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        const auto planar
            = parsedProblem(dense(64, 32, {12, 13}, {1, 1}, {0, 0}, false), direction);
        const auto line = conv_igemm::withChannelsFirstLineView(planar);
        EXPECT_EQ(line.args.di, 1);
        EXPECT_EQ(line.args.hi, 1);
        EXPECT_EQ(line.args.wi, 12 * 13);
        EXPECT_EQ(line.args.dout, 1);
        EXPECT_EQ(line.args.ho, 1);
        EXPECT_EQ(line.args.wo, 12 * 13);
        EXPECT_EQ(line.args.m, planar.args.m);
        EXPECT_EQ(line.base.args.inSpatial[2], 12 * 13);
        EXPECT_EQ(line.base.args.outSpatial[1], 1);
        EXPECT_TRUE(conv_igemm::cfxFits(line, cfxTile(128, 1)));
        EXPECT_FALSE(conv_igemm::cfxFits(planar, cfxTile(128, 1)));
        const auto volume
            = parsedProblem(dense(16, 24, {3, 5, 7}, {1, 1, 1}, {0, 0, 0}, false), direction);
        EXPECT_EQ(conv_igemm::withChannelsFirstLineView(volume).args.wi, 3 * 5 * 7);
        // Not pointwise unit-stride unpadded channels-first: unchanged.
        for(const auto& shape : {dense(64, 32, {12, 13}, {3, 3}, {1, 1}, false),
                                 dense(64, 32, {12, 13}, {1, 1}, {1, 1}, false),
                                 dense(64, 32, {12, 13}, {1, 1}, {0, 0}, true)})
        {
            const auto problem = parsedProblem(shape, direction);
            const auto same = conv_igemm::withChannelsFirstLineView(problem);
            EXPECT_EQ(same.args.hi, problem.args.hi);
            EXPECT_EQ(same.args.wi, problem.args.wi);
        }
        auto strided = dense(64, 32, {12, 14}, {1, 1}, {0, 0}, false);
        strided.stride = {2, 2};
        const auto s2 = parsedProblem(strided, direction);
        EXPECT_EQ(conv_igemm::withChannelsFirstLineView(s2).args.hi, s2.args.hi);
    }
}

TEST(TestConvIgemmCatalog, ChannelsFirstInputConfigsArePlainHaloTilesOrFma)
{
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        int count = 0;
        for(const auto& kernel : catalog(direction))
        {
            if(kernel.getIntMetadata("cfx") == 0)
            {
                continue;
            }
            SCOPED_TRACE(kernel.name);
            ++count;
            EXPECT_EQ(kernel.getStringMetadata("math"), "native");
            EXPECT_EQ(kernel.getIntMetadata("swz"), 0);
            EXPECT_EQ(kernel.getIntMetadata("dma"), 0);
            EXPECT_EQ(kernel.getIntMetadata("pack"), 0);
            EXPECT_EQ(kernel.getIntMetadata("wdirect"), 0);
            EXPECT_EQ(kernel.getIntMetadata("fuse"), 0);
            EXPECT_EQ(kernel.getIntMetadata("large"), 0);
            EXPECT_EQ(kernel.getIntMetadata("patch"), 0);
            if(kernel.getStringMetadata("atom") == "fma")
            {
                EXPECT_EQ(kernel.getIntMetadata("halo"), 0);
            }
            else
            {
                EXPECT_EQ(kernel.getIntMetadata("cf"), 1);
                const auto halo = kernel.getIntMetadata("halo");
                EXPECT_TRUE(halo == 1 || (halo == 2 && direction == ConvDirection::FORWARD));
            }
        }
        EXPECT_GT(count, 0);
    }
}

std::optional<conv_igemm::IgemmProblem> patchView(ConvShape shape, ConvDirection direction)
{
    shape.dataType = DataType::HALF;
    const GraphFixture fixture(buildGraph(specFor(shape, direction)));
    const auto problem = conv_igemm::parseIgemmProblem(fixture.context(), direction);
    if(!problem.has_value())
    {
        ADD_FAILURE() << "the plain parse declined a patch test case";
        return std::nullopt;
    }
    return conv_igemm::withPatchView(*problem);
}

std::optional<conv_igemm::IgemmProblem> col2imView(ConvShape shape)
{
    shape.dataType = DataType::HALF;
    const GraphFixture fixture(buildGraph(specFor(shape, ConvDirection::BACKWARD_DATA)));
    const auto problem
        = conv_igemm::parseIgemmProblem(fixture.context(), ConvDirection::BACKWARD_DATA);
    if(!problem.has_value())
    {
        ADD_FAILURE() << "the plain parse declined a col2im test case";
        return std::nullopt;
    }
    return conv_igemm::withCol2imView(*problem);
}

TEST(TestConvIgemmCatalog, Col2imIsOneGemmOverDyPositionsWithColumnsInTheWorkspace)
{
    // 3x3x3 "valid" over 6 x 5 x 7: dy is 4 x 3 x 5 a image, 27 taps of 16 channels a column row.
    const auto valid = dense(16, 24, {6, 5, 7}, {3, 3, 3}, {0, 0, 0}, true);
    const auto view = col2imView(valid);
    ASSERT_TRUE(view.has_value());
    EXPECT_EQ(view->args.c, 16 * 27);
    EXPECT_EQ(view->args.k, 24);
    EXPECT_EQ(view->args.di * view->args.hi * view->args.wi, 4 * 3 * 5);
    EXPECT_EQ(view->args.ft * view->args.fr * view->args.fs, 1);
    EXPECT_EQ(view->args.sd * view->args.sh * view->args.sw, 1);
    EXPECT_EQ(view->args.pd + view->args.ph + view->args.pw, 0);
    EXPECT_EQ(view->phases, 1);
    EXPECT_EQ(view->rows, 2 * 4 * 3 * 5);
    EXPECT_FALSE(view->w.needed); // the K T R S C filter is the GEMM's K-major B, read in place
    EXPECT_TRUE(view->col2imTapMajor);
    EXPECT_EQ(view->col2imBytes, sizeof(float) * 2 * 4 * 3 * 5 * 16 * 27);
    EXPECT_GE(view->col2imOffset, 0U);
    EXPECT_EQ(view->workspaceBytes % 256, 0U);
    EXPECT_GE(view->workspaceBytes, view->col2imOffset + view->col2imBytes);
    // Channels-first dy and dx are staged; the columns follow them, in the K C T R S filter's
    // (channel, tap) order.
    const auto staged = col2imView(dense(16, 24, {6, 5, 7}, {3, 3, 3}, {0, 0, 0}, false));
    ASSERT_TRUE(staged.has_value());
    EXPECT_TRUE(staged->x.needed);
    EXPECT_FALSE(staged->w.needed);
    EXPECT_FALSE(staged->col2imTapMajor);
    EXPECT_GE(staged->col2imOffset, staged->x.workspaceOffset + staged->x.bytes);
}

TEST(TestConvIgemmCatalog, Col2imDeclinesEachFieldItDoesNotServe)
{
    const auto valid = dense(16, 24, {6, 5, 7}, {3, 3, 3}, {0, 0, 0}, true);
    ASSERT_TRUE(col2imView(valid).has_value());
    // "Same" padding: dy has as many positions as dx, so the GEMM over dx rows is no larger.
    EXPECT_FALSE(col2imView(dense(16, 24, {6, 5, 7}, {3, 3, 3}, {1, 1, 1}, true)).has_value());
    // Groups.
    auto grouped = valid;
    grouped.groups = 2;
    EXPECT_FALSE(col2imView(grouped).has_value());
    // Patch convolutions: the patch view writes their dx with no columns in between.
    EXPECT_FALSE(col2imView(patchShape(16, 24, {6, 10}, 2, true)).has_value());
    // True convolution: the reflected filter is not the K-major B the configs read in place.
    auto convolution = valid;
    convolution.mode = ConvMode::CONVOLUTION;
    EXPECT_FALSE(col2imView(convolution).has_value());
    // One tap: nothing to sum (the plain GEMM over dx rows is the same GEMM).
    EXPECT_FALSE(col2imView(dense(16, 24, {6, 5, 7}, {1, 1, 1}, {0, 0, 0}, true)).has_value());
    // Columns past 1 GiB of fp32.
    auto huge = dense(256, 24, {66, 66, 66}, {3, 3, 3}, {0, 0, 0}, true);
    huge.n = 1;
    EXPECT_FALSE(col2imView(huge).has_value());
    auto under = huge;
    under.input = {34, 34, 34};
    EXPECT_TRUE(col2imView(under).has_value());
    // Forward graphs never reach the view: the kernel match declines col2im configs there.
}

TEST(TestConvIgemmCatalog, Col2imConfigsAreBackwardDataDirectFilterPlainTiles)
{
    int count = 0;
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        for(const auto& kernel : catalog(direction))
        {
            if(kernel.getIntMetadata("patch") != 3)
            {
                continue;
            }
            SCOPED_TRACE(kernel.name);
            ++count;
            EXPECT_EQ(direction, ConvDirection::BACKWARD_DATA);
            EXPECT_EQ(kernel.getIntMetadata("wdirect"), 1);
            EXPECT_EQ(kernel.getStringMetadata("math"), "native");
            EXPECT_NE(kernel.getStringMetadata("atom"), "fma");
            for(const auto* knob : {"halo",
                                    "pack",
                                    "cf",
                                    "swz",
                                    "dma",
                                    "fuse",
                                    "large",
                                    "cfx",
                                    "flex",
                                    "split_policy"})
            {
                EXPECT_EQ(kernel.getIntMetadata(knob), 0) << knob;
            }
        }
    }
    EXPECT_GT(count, 0);
}

TEST(TestConvIgemmCatalog, PatchForwardFoldsTheWidthTapsIntoChannels)
{
    // 16x16 stride 16 over RGB: W / 16 pixels of 48 channels, filter rows of one tap.
    const auto vit = patchView(patchShape(3, 64, {32, 64}, 16, true), ConvDirection::FORWARD);
    ASSERT_TRUE(vit.has_value());
    EXPECT_EQ(vit->args.c, 48);
    EXPECT_EQ(vit->args.wi, 4);
    EXPECT_EQ(vit->args.wo, 4);
    EXPECT_EQ(vit->args.fs, 1);
    EXPECT_EQ(vit->args.sw, 1);
    EXPECT_EQ(vit->args.fr, 16);
    EXPECT_EQ(vit->args.sh, 16);
    EXPECT_FALSE(vit->w.needed); // K T R S C is K T R 1 (S C): read as given
    // Only the width needs to fold: any height geometry stays a gather.
    auto widthOnly = patchShape(8, 16, {9, 12}, 4, true);
    widthOnly.filter = {3, 4};
    widthOnly.stride = {1, 4};
    widthOnly.prePadding = {1, 0};
    widthOnly.postPadding = {1, 0};
    EXPECT_TRUE(patchView(widthOnly, ConvDirection::FORWARD).has_value());
    // Channels-first activations are staged packed channels-last, which the view reads alike.
    EXPECT_TRUE(patchView(patchShape(16, 24, {6, 10}, 2, false), ConvDirection::FORWARD));
}

TEST(TestConvIgemmCatalog, PatchBackwardDataIsOneGemmOverDyPositions)
{
    const auto view = patchView(patchShape(16, 24, {6, 10}, 2, true), ConvDirection::BACKWARD_DATA);
    ASSERT_TRUE(view.has_value());
    EXPECT_EQ(view->args.c, 64); // 2 x 2 x 16 columns
    EXPECT_EQ(view->args.k, 24);
    EXPECT_EQ(view->args.hi, 3);
    EXPECT_EQ(view->args.wi, 5);
    EXPECT_EQ(view->args.fr * view->args.fs * view->args.ft, 1);
    EXPECT_EQ(view->args.sh * view->args.sw * view->args.sd, 1);
    EXPECT_EQ(view->args.patchT, 1);
    EXPECT_EQ(view->args.patchR, 2);
    EXPECT_EQ(view->phases, 1);
    EXPECT_EQ(view->rows, 2 * 3 * 5);
    // A packed K R S C filter: the parse's transpose to [(r, s, c)][k].
    EXPECT_TRUE(view->w.needed && view->w.transpose);
    EXPECT_TRUE(view->filterKtrsc);
    // A K C R S filter, or a reflected one: the strided copy to [t][r][s][c][k].
    const auto kcrs
        = patchView(patchShape(16, 24, {6, 10}, 2, false), ConvDirection::BACKWARD_DATA);
    ASSERT_TRUE(kcrs.has_value());
    EXPECT_TRUE(kcrs->w.needed && !kcrs->w.transpose);
    EXPECT_EQ(kcrs->w.copy.total, 2 * 2 * 16 * 24);
    auto convolution = patchShape(16, 24, {6, 10}, 2, true);
    convolution.mode = ConvMode::CONVOLUTION;
    const auto reflected = patchView(convolution, ConvDirection::BACKWARD_DATA);
    ASSERT_TRUE(reflected.has_value());
    EXPECT_TRUE(reflected->w.needed && !reflected->w.transpose);
    EXPECT_NE(reflected->w.graphOffset, 0);
    // 3-D: the depth taps join the columns too.
    const auto deep
        = patchView(patchShape(8, 16, {4, 6, 8}, 2, true), ConvDirection::BACKWARD_DATA);
    ASSERT_TRUE(deep.has_value());
    EXPECT_EQ(deep->args.c, 64);
    EXPECT_EQ(deep->args.patchT, 2);
}

TEST(TestConvIgemmCatalog, PatchViewDeclinesEveryNonPatchField)
{
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        SCOPED_TRACE(direction == ConvDirection::FORWARD ? "forward" : "backward data");
        const auto base = patchShape(16, 24, {8, 12}, 4, true);
        EXPECT_TRUE(patchView(base, direction).has_value());
        auto grouped = base;
        grouped.groups = 2;
        EXPECT_FALSE(patchView(grouped, direction).has_value());
        auto overlap = base; // the width stride below the filter
        overlap.stride = {4, 2};
        EXPECT_FALSE(patchView(overlap, direction).has_value());
        auto padded = base;
        padded.prePadding = {0, 2};
        padded.postPadding = {0, 2};
        EXPECT_FALSE(patchView(padded, direction).has_value());
        auto dilated = base; // (a dilated filter never tiles the width exactly either)
        dilated.input = {8, 28};
        dilated.dilation = {1, 2};
        EXPECT_FALSE(patchView(dilated, direction).has_value());
        auto ragged = base; // the width not a whole number of tiles
        ragged.input = {8, 14};
        EXPECT_FALSE(patchView(ragged, direction).has_value());
        auto unit = base; // nothing to fold along the width
        unit.filter = {4, 1};
        unit.stride = {4, 1};
        EXPECT_EQ(patchView(unit, direction).has_value(), direction != ConvDirection::FORWARD);
    }
    // Backward data folds every axis: the height must tile exactly too.
    auto rowsOverlap = patchShape(16, 24, {8, 12}, 4, true);
    rowsOverlap.stride = {2, 4};
    EXPECT_TRUE(patchView(rowsOverlap, ConvDirection::FORWARD).has_value());
    EXPECT_FALSE(patchView(rowsOverlap, ConvDirection::BACKWARD_DATA).has_value());
    auto raggedRows = patchShape(16, 24, {10, 12}, 4, true);
    EXPECT_TRUE(patchView(raggedRows, ConvDirection::FORWARD).has_value());
    EXPECT_FALSE(patchView(raggedRows, ConvDirection::BACKWARD_DATA).has_value());
    auto paddedRows = patchShape(16, 24, {8, 12}, 4, true);
    paddedRows.prePadding = {2, 0};
    paddedRows.postPadding = {2, 0};
    EXPECT_FALSE(patchView(paddedRows, ConvDirection::BACKWARD_DATA).has_value());
    // A 1x1 filter folds nothing.
    EXPECT_FALSE(patchView(patchShape(16, 24, {8, 12}, 1, true), ConvDirection::BACKWARD_DATA));
    // Images past 2^31 - 1 pixels take the large-image configs, never a patch config.
    auto huge = patchShape(1, 1, {46342, 46342}, 2, true);
    huge.n = 1;
    EXPECT_FALSE(patchView(huge, ConvDirection::FORWARD).has_value());
}

TEST(TestConvIgemmCatalog, PatchConfigsServeOnlyPatchConvolutions)
{
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        const auto pack = packFor(direction);
        const auto serves = [&](ConvShape shape, const auto& kernel) {
            shape.dataType = DataType::HALF;
            const GraphFixture fixture(buildGraph(specFor(shape, direction)),
                                       deviceOf("gfx942:sramecc+:xnack-", 304));
            const auto bound = matchesGraph(pack, fixture.context());
            return bound.has_value() && matchesKernel(pack, fixture.context(), kernel, *bound);
        };
        const auto& kernels = catalog(direction);
        const auto patch = std::find_if(kernels.begin(), kernels.end(), [](const auto& k) {
            return k.getStringMetadata("dtype") == "HALF"
                   && k.getStringMetadata("atom") == "mfma32_f16" && k.getIntMetadata("patch") == 1
                   && k.getIntMetadata("pack") == 0 && k.getIntMetadata("swz") == 0
                   && k.getIntMetadata("cf") == 0 && k.getIntMetadata("xcd") == 0
                   && k.getIntMetadata("wdirect") == 0 && k.getIntMetadata("vec") == 8;
        });
        ASSERT_NE(patch, kernels.end());
        SCOPED_TRACE(patch->name);
        EXPECT_TRUE(serves(patchShape(16, 24, {8, 12}, 2, true), *patch));
        EXPECT_FALSE(serves(dense(16, 24, {8, 12}, {3, 3}, {1, 1}, true), *patch));
        // Runs of S C channels (forward's reduction channels, backward data's dx stores) take the
        // channel vector: 2 x 3 is not a whole 8-wide vector, 2 x 4 is.
        EXPECT_FALSE(serves(patchShape(3, 24, {8, 12}, 2, true), *patch));
        EXPECT_TRUE(serves(patchShape(4, 24, {8, 12}, 2, true), *patch));
        // The patch tile serves the same convolutions on its 16-bit 16x16 atoms, every variant
        // knob off.
        const auto tile = std::find_if(kernels.begin(), kernels.end(), [](const auto& k) {
            return k.getStringMetadata("dtype") == "HALF"
                   && k.getStringMetadata("atom") == "mfma16_f16" && k.getIntMetadata("patch") == 2
                   && k.getIntMetadata("vec") == 8;
        });
        ASSERT_NE(tile, kernels.end());
        EXPECT_TRUE(serves(patchShape(16, 24, {8, 12}, 2, true), *tile));
        EXPECT_FALSE(serves(dense(16, 24, {8, 12}, {3, 3}, {1, 1}, true), *tile));
        EXPECT_FALSE(serves(patchShape(3, 24, {8, 12}, 2, true), *tile));
        for(const char* knob : {"xcd", "halo", "pack", "cf", "swz", "dma"})
        {
            auto variant = *tile;
            variant.metadata[knob] = int64_t{1};
            EXPECT_FALSE(serves(patchShape(16, 24, {8, 12}, 2, true), variant)) << knob;
        }
        // The in-place filter read is backward data's only.
        auto inPlace = *tile;
        inPlace.metadata["wdirect"] = int64_t{1};
        EXPECT_EQ(serves(patchShape(16, 24, {8, 12}, 2, true), inPlace),
                  direction == ConvDirection::BACKWARD_DATA);
        auto fp32Atom = *tile;
        fp32Atom.metadata["atom"] = std::string("mfma16_f32");
        EXPECT_FALSE(serves(patchShape(16, 24, {8, 12}, 2, true), fp32Atom));
        auto wide = *tile;
        wide.metadata["atom"] = std::string("mfma32_f16");
        EXPECT_FALSE(serves(patchShape(16, 24, {8, 12}, 2, true), wide));
        // A patch knob other than 0 or 1, or with large images or group fusion, is not a config.
        auto two = *patch;
        two.metadata["patch"] = int64_t{2};
        EXPECT_FALSE(serves(patchShape(16, 24, {8, 12}, 2, true), two));
        auto fused = *patch;
        fused.metadata["fuse"] = int64_t{2};
        EXPECT_FALSE(serves(patchShape(16, 24, {8, 12}, 2, true), fused));
        auto large = *patch;
        large.metadata["large"] = int64_t{1};
        EXPECT_FALSE(serves(patchShape(16, 24, {8, 12}, 2, true), large));
        if(direction == ConvDirection::BACKWARD_DATA)
        {
            // dx is written channels-last: no channels-first output.
            auto channelsFirst = *patch;
            channelsFirst.metadata["cf"] = int64_t{1};
            EXPECT_FALSE(serves(patchShape(16, 24, {8, 12}, 2, false), channelsFirst));
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
    // Patch configs run one launch whatever the fill.
    EXPECT_EQ(conv_igemm::splitPlanFor(deep, 128, 128, 32, deviceOf("gfx942", 304), 0, true).splits,
              1);
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

TEST(TestConvIgemmSplitPlan, ExplicitNoSplitPreservesStagingAcrossOccupancyAndReduction)
{
    for(const bool channelsLast : {false, true})
    {
        for(const int k : {32, 512})
        {
            auto shape = dense(40, k, {7, 9}, {1, 1}, {0, 0}, channelsLast);
            shape.n = 1;
            const auto problem = parsedProblem(shape, ConvDirection::BACKWARD_DATA);
            for(const int units : {1, 128})
            {
                const auto device = deviceOf("gfx1250:xnack-", units);
                const auto automatic = conv_igemm::splitPlanFor(problem, 64, 64, 32, device);
                const auto explicitPlan
                    = conv_igemm::splitPlanFor(problem, 64, 64, 32, device, 0, false, 1);
                const int expectedSplits = units == 128 && k == 512 ? 2 : 1;
                EXPECT_EQ(automatic.splits, expectedSplits);
                EXPECT_EQ(explicitPlan.splits, 1);
                EXPECT_EQ(explicitPlan.partialOffset, 0);
                EXPECT_EQ(explicitPlan.workspaceBytes, problem.workspaceBytes);
                EXPECT_EQ(explicitPlan.outElems, 7 * 9 * 40);
                if(expectedSplits > 1)
                {
                    EXPECT_GE(automatic.partialOffset, problem.workspaceBytes);
                    EXPECT_EQ(automatic.workspaceBytes,
                              automatic.partialOffset
                                  + 2 * static_cast<size_t>(explicitPlan.outElems) * sizeof(float));
                }
                else
                {
                    EXPECT_EQ(automatic.workspaceBytes, explicitPlan.workspaceBytes);
                }
            }
        }
    }
}

TEST(TestConvIgemmCatalog, NoSplitRejectsUnsupportedPolicyAndExecutionDomains)
{
    // 512 output channels: the default plan splits backward data on gfx1250 (16 k-steps), so the
    // no-split candidate is a distinct plan and served.
    auto shape = dense(40, 512, {7, 9}, {1, 1}, {0, 0}, false);
    shape.dataType = DataType::HALF;
    const auto direction = ConvDirection::BACKWARD_DATA;
    const auto pack = packFor(direction);
    const GraphFixture supported(buildGraph(specFor(shape, direction)), deviceOf("gfx1250", 128));
    const auto bound = matchesGraph(pack, supported.context());
    ASSERT_TRUE(bound.has_value());
    const auto candidate = nativeContractKernel(direction);
    ASSERT_TRUE(matchesKernel(pack, supported.context(), candidate, *bound));
    for(const int policy : {-1, static_cast<int>(conv_igemm::MAX_SPLIT_POLICY) + 1})
    {
        auto invalid = candidate;
        invalid.metadata["split_policy"] = int64_t{policy};
        EXPECT_FALSE(matchesKernel(pack, supported.context(), invalid, *bound));
    }
    auto wrongDirection = candidate;
    wrongDirection.metadata["direction"] = std::string("FWD");
    const auto forwardPack = packFor(ConvDirection::FORWARD);
    const GraphFixture forward(buildGraph(specFor(shape, ConvDirection::FORWARD)),
                               deviceOf("gfx1250", 128));
    const auto forwardBound = matchesGraph(forwardPack, forward.context());
    ASSERT_TRUE(forwardBound.has_value());
    EXPECT_FALSE(matchesKernel(forwardPack, forward.context(), wrongDirection, *forwardBound));
    auto wrongType = candidate;
    wrongType.metadata["dtype"] = std::string("FLOAT");
    shape.dataType = DataType::FLOAT;
    const GraphFixture fp32(buildGraph(specFor(shape, direction)), deviceOf("gfx1250", 128));
    const auto fp32Bound = matchesGraph(pack, fp32.context());
    ASSERT_TRUE(fp32Bound.has_value());
    EXPECT_FALSE(matchesKernel(pack, fp32.context(), wrongType, *fp32Bound));
    shape.dataType = DataType::HALF;
    const GraphFixture older(buildGraph(specFor(shape, direction)), deviceOf("gfx1201", 64));
    const auto olderBound = matchesGraph(pack, older.context());
    ASSERT_TRUE(olderBound.has_value());
    EXPECT_FALSE(matchesKernel(pack, older.context(), candidate, *olderBound));
    shape.channelsLast = true;
    const GraphFixture channelsLast(buildGraph(specFor(shape, direction)),
                                    deviceOf("gfx1250", 128));
    const auto channelsLastBound = matchesGraph(pack, channelsLast.context());
    ASSERT_TRUE(channelsLastBound.has_value());
    EXPECT_FALSE(matchesKernel(pack, channelsLast.context(), candidate, *channelsLastBound));
}

TEST(TestConvIgemmSplitPlan, ExplicitSplitCountsAreExactWhereFeasible)
{
    // 2048 channels x 9 taps: 576 k-steps of 32.
    const auto deep
        = parsedProblem(dense(2048, 128, {8, 8}, {3, 3}, {1, 1}, true), ConvDirection::FORWARD);
    const auto device = deviceOf("gfx942", 304);
    for(const int policy : {2, 32, 64})
    {
        const auto plan = conv_igemm::splitPlanFor(deep, 128, 128, 32, device, 0, false, policy);
        EXPECT_EQ(plan.splits, policy);
        EXPECT_GE(plan.partialOffset, deep.workspaceBytes);
        EXPECT_EQ(plan.workspaceBytes,
                  plan.partialOffset
                      + static_cast<size_t>(policy) * static_cast<size_t>(plan.outElems)
                            * sizeof(float));
    }
    EXPECT_EQ(conv_igemm::splitPlanFor(deep, 128, 128, 32, device).splits, 16);
    EXPECT_EQ(conv_igemm::splitPlanFor(deep, 128, 128, 32, device, 0, false, 1).splits, 1);
    // 32 channels x 9 taps: 9 k-steps, so 16 splits would leave some without one: none.
    const auto shallow
        = parsedProblem(dense(32, 128, {8, 8}, {3, 3}, {1, 1}, true), ConvDirection::FORWARD);
    const auto infeasible = conv_igemm::splitPlanFor(shallow, 128, 128, 32, device, 0, false, 16);
    EXPECT_EQ(infeasible.splits, 1);
    EXPECT_EQ(infeasible.workspaceBytes, shallow.workspaceBytes);
    EXPECT_EQ(conv_igemm::splitPlanFor(shallow, 128, 128, 32, device, 0, false, 9).splits, 9);
    // A filled grid (640 tiles against 608 wanted, 144 k-steps, partials in bounds): none.
    const auto filled
        = parsedProblem(dense(512, 256, {128, 160}, {3, 3}, {1, 1}, true), ConvDirection::FORWARD);
    for(const int policy : {2, 8})
    {
        const auto plan = conv_igemm::splitPlanFor(filled, 128, 128, 32, device, 0, false, policy);
        EXPECT_EQ(plan.splits, 1);
        EXPECT_EQ(plan.workspaceBytes, filled.workspaceBytes);
    }
    // Patch configs never split, whatever the policy.
    EXPECT_EQ(conv_igemm::splitPlanFor(deep, 128, 128, 32, device, 0, true, 32).splits, 1);
    EXPECT_THROW(conv_igemm::splitPlanFor(
                     deep, 128, 128, 32, device, 0, false, conv_igemm::MAX_SPLIT_POLICY + 1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_THROW(conv_igemm::splitPlanFor(deep, 128, 128, 32, device, 0, false, -1),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestConvIgemmSplitPlan, ResidentBlocksFollowThreadsLdsAndAccumulators)
{
    // 512 threads (8 waves), 256 x 128, fp16 on gfx950: 128 registers (4 waves a SIMD, 16 a CU)
    // and 30 KiB of LDS: 2 blocks.
    EXPECT_EQ(conv_igemm::residentBlocksOf(256, 128, 32, 512, 64, 1, 0, 2, "gfx950"), 2);
    // 512 threads, 256 x 256: 192 registers, 2 waves a SIMD, 8 a CU: one block.
    EXPECT_EQ(conv_igemm::residentBlocksOf(256, 256, 32, 512, 64, 1, 0, 2, "gfx942"), 1);
    // The same on gfx942's 64 KiB with three 64-deep DMA-like stages (3 x 512 x 72 x 2 = 216
    // KiB): over the LDS, still one block.
    EXPECT_EQ(conv_igemm::residentBlocksOf(256, 256, 64, 512, 64, 3, 0, 2, "gfx942"), 1);
    // Fewer than 8 waves: the default's 2 (0 here), however large the tile or its LDS.
    EXPECT_EQ(conv_igemm::residentBlocksOf(128, 128, 32, 256, 64, 1, 0, 2, "gfx942"), 0);
    EXPECT_EQ(conv_igemm::residentBlocksOf(256, 256, 32, 256, 64, 1, 0, 2, "gfx950"), 0);
    EXPECT_EQ(conv_igemm::residentBlocksOf(128, 128, 64, 256, 64, 1, 3, 2, "gfx950"), 0);
    // Not a gfx9 CU, or the FMA kernel: the split plan's arch default.
    EXPECT_EQ(conv_igemm::residentBlocksOf(256, 256, 32, 512, 32, 1, 0, 2, "gfx1201"), 0);
    EXPECT_EQ(conv_igemm::residentBlocksOf(256, 256, 32, 512, 32, 1, 0, 2, "gfx1250"), 0);
    EXPECT_EQ(conv_igemm::residentBlocksOf(64, 64, 16, 512, 0, 1, 0, 4, "gfx942"), 0);
    // A config one block a CU holds fills the device with half the default's blocks: 40 tiles
    // on 304 CUs split 16 ways at two blocks a CU (the cap), 8 at one.
    const auto wide
        = parsedProblem(dense(1024, 1280, {16, 16}, {3, 3}, {1, 1}, true), ConvDirection::FORWARD);
    const auto device = deviceOf("gfx942", 304);
    EXPECT_EQ(conv_igemm::splitPlanFor(wide, 128, 128, 32, device).splits, 16);
    EXPECT_EQ(conv_igemm::splitPlanFor(wide, 128, 128, 32, device, 0, false, 0, 1).splits, 8);
    EXPECT_EQ(conv_igemm::splitPlanFor(wide, 128, 128, 32, device, 0, false, 0, 3).splits, 16);
}

TEST(TestConvIgemmCatalog, SplittableMatrixCoreConfigsHaveSplitPolicyTwins)
{
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        const auto& kernels = catalog(direction);
        // Every field but the split policy.
        const auto knobsOf = [](const auto& kernel) {
            auto fields = kernel.metadata;
            fields.erase("split_policy");
            return fields;
        };
        int nosplit = 0;
        int wide = 0;
        for(const auto& kernel : kernels)
        {
            const auto atom = kernel.getStringMetadata("atom");
            if(kernel.getIntMetadata("split_policy") != 0 || atom.rfind("mfma", 0) != 0
               || atom.find("xf32") != std::string::npos
               || kernel.getStringMetadata("math") != "native"
               || kernel.getIntMetadata("patch") != 0 || kernel.getIntMetadata("large") != 0)
            {
                continue;
            }
            SCOPED_TRACE(kernel.name);
            const auto knobs = knobsOf(kernel);
            const auto twin = [&](int64_t policy) {
                return std::count_if(kernels.begin(), kernels.end(), [&](const auto& other) {
                    return other.getIntMetadata("split_policy") == policy
                           && knobsOf(other) == knobs;
                });
            };
            EXPECT_EQ(twin(1), 1);
            nosplit += 1;
            if(kernel.getIntMetadata("bm") * kernel.getIntMetadata("bn") >= int64_t{256} * 128)
            {
                EXPECT_EQ(twin(32), 1);
                ++wide;
            }
        }
        EXPECT_GT(nosplit, 0);
        EXPECT_GT(wide, 0);
    }
}

TEST(TestConvIgemmCatalog, SplitPolicyTwinsServeOnlyPlansTheDefaultDoesNotRun)
{
    const auto direction = ConvDirection::FORWARD;
    const auto pack = packFor(direction);
    const auto& kernels = catalog(direction);
    const auto find = [&](int64_t policy) {
        return std::find_if(kernels.begin(), kernels.end(), [&](const auto& kernel) {
            return kernel.getStringMetadata("atom") == "mfma32_f16"
                   && kernel.getIntMetadata("bm") == 256 && kernel.getIntMetadata("bn") == 128
                   && kernel.getIntMetadata("cf") == 0 && kernel.getIntMetadata("cfx") == 0
                   && kernel.getIntMetadata("xcd") == 0 && kernel.getIntMetadata("swz") == 0
                   && kernel.getIntMetadata("pack") == 0 && kernel.getIntMetadata("fuse") == 0
                   && kernel.getIntMetadata("split_policy") == policy;
        });
    };
    const auto plainAt = find(0);
    const auto noneAt = find(1);
    const auto split32At = find(32);
    ASSERT_NE(plainAt, kernels.end());
    ASSERT_NE(noneAt, kernels.end());
    ASSERT_NE(split32At, kernels.end());
    const auto& plain = *plainAt;
    const auto& none = *noneAt;
    const auto& split32 = *split32At;
    const auto served = [&](const ConvShape& shape, const auto& kernel) {
        auto s = shape;
        s.dataType = DataType::HALF;
        const GraphFixture fixture(buildGraph(specFor(s, direction)), deviceOf("gfx942", 304));
        const auto bound = matchesGraph(pack, fixture.context());
        return bound.has_value() && matchesKernel(pack, fixture.context(), kernel, *bound);
    };
    // Deep and under-filled (one tile, 576 k-steps): the default splits 16 ways, so no-split and
    // 32 splits are distinct plans.
    const auto deep = dense(2048, 128, {8, 8}, {3, 3}, {1, 1}, true);
    EXPECT_TRUE(served(deep, plain));
    EXPECT_TRUE(served(deep, none));
    EXPECT_TRUE(served(deep, split32));
    // Filled (1024 tiles of 256 x 128 against 608 wanted): the default does not split, so the
    // no-split twin is the same plan.
    const auto filled = dense(256, 256, {256, 256}, {3, 3}, {1, 1}, true);
    EXPECT_TRUE(served(filled, plain));
    EXPECT_FALSE(served(filled, none));
    // Filled and narrow (640 tiles of 256 x 128 on 8 output channels, 144 k-steps, 32 splits'
    // partials 168 MB): 32 splits fit but the grid needs none, so the twin is not served and its
    // partials do not size the engine's workspace.
    const auto narrow = dense(512, 8, {256, 320}, {3, 3}, {1, 1}, true);
    EXPECT_TRUE(served(narrow, plain));
    EXPECT_FALSE(served(narrow, split32));
    // 9 k-steps cannot take 32 splits.
    const auto shallow = dense(32, 128, {8, 8}, {3, 3}, {1, 1}, true);
    EXPECT_FALSE(served(shallow, split32));
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

TEST(TestConvIgemmSplitPlan, FlexLdsDmaFormTakesTheTailSplitAtItsSplitCost)
{
    EXPECT_EQ(conv_igemm::flexSplitCost(1), 20);
    EXPECT_EQ(conv_igemm::flexSplitCost(2), 14);
    // 300 row tiles of 216 k-steps on 256 units: 256 row tiles fill one round and the other 44
    // split 5 ways into one more round (216 + 44 + 20 k-steps, not 2 x 216).
    const auto tail
        = conv_igemm::flexTailSplit(300, 1, 216, 256, true, conv_igemm::flexSplitCost(1));
    EXPECT_EQ(tail.splitRowTile, 256);
    EXPECT_EQ(tail.splits, 5);
    // A fused epilogue: every row tile splits, 4 rounds of 72 + 20.
    const auto whole
        = conv_igemm::flexTailSplit(300, 1, 216, 256, false, conv_igemm::flexSplitCost(1));
    EXPECT_EQ(whole.splitRowTile, 0);
    EXPECT_EQ(whole.splits, 3);
    // Split policy 0 applies it to the flex tile's grid: one 256 x 256 tile of 27 x 8 k-steps of
    // 64 splits 16 ways (14 + 20 k-steps); policies 1 and N plan as for any config.
    auto deep = dense(512, 256, {3, 4, 5}, {3, 3, 3}, {1, 1, 1}, true);
    deep.n = 1;
    const auto problem = parsedProblem(deep, ConvDirection::FORWARD);
    const auto device = deviceOf("gfx950", 256);
    const auto flexPlan = [&](int64_t policy) {
        return conv_igemm::splitPlanFor(problem, 256, 256, 64, device, 0, false, policy, 1, 1);
    };
    EXPECT_EQ(flexPlan(0).splits, 16);
    EXPECT_EQ(flexPlan(0).splitRowTile, 0);
    EXPECT_EQ(flexPlan(0).partialElems, flexPlan(0).outElems);
    EXPECT_EQ(flexPlan(1).splits, 1);
    EXPECT_EQ(flexPlan(4).splits, 4);
    // 88 tiles (11 x 32 x 64 positions) on 256 units: 2 splits in one round (108 + 20 k-steps
    // beat two rounds of any larger split) where the default rule would take 3 splits.
    auto wide = dense(512, 256, {11, 32, 64}, {3, 3, 3}, {1, 1, 1}, true);
    wide.n = 1;
    const auto wideProblem = parsedProblem(wide, ConvDirection::FORWARD);
    EXPECT_EQ(conv_igemm::splitPlanFor(wideProblem, 256, 256, 64, device, 0, false, 0, 1, 1).splits,
              2);
    EXPECT_EQ(conv_igemm::splitPlanFor(wideProblem, 256, 256, 64, device, 0, false, 0, 1).splits,
              3);
    // 300 x 256 positions (300 row tiles of one 256-channel column tile): the tail split, its
    // partials covering only the 44 split row tiles.
    auto rows = dense(512, 256, {300, 256}, {3, 3}, {1, 1}, true);
    rows.n = 1;
    const auto rowsPlan = conv_igemm::splitPlanFor(
        parsedProblem(rows, ConvDirection::FORWARD), 256, 256, 64, device, 0, false, 0, 1, 1);
    EXPECT_EQ(rowsPlan.splitRowTile, 256);
    EXPECT_GT(rowsPlan.splits, 1);
    EXPECT_EQ(rowsPlan.partialElems, rowsPlan.outElems - 256LL * 256 * 256);
}

TEST(TestConvIgemmSplitPlan, FlexTailSplitSplitsTheRowTilesPastTheWholeRounds)
{
    // 256 row tiles of 2 tiles, 216 k-steps, 228 units: 228 row tiles fill two rounds and the
    // other 56 tiles split 4 ways into one more round (2 x 216 + 54 + 14 k-steps).
    const auto tail
        = conv_igemm::flexTailSplit(256, 2, 216, 228, true, conv_igemm::flexSplitCost(2));
    EXPECT_EQ(tail.splitRowTile, 228);
    EXPECT_EQ(tail.splits, 4);
    // With the tail kept in the launch (a fused epilogue), every row tile splits or none does:
    // 7 rounds of 72 + 14 beat 3 rounds of 216.
    const auto whole
        = conv_igemm::flexTailSplit(256, 2, 216, 228, false, conv_igemm::flexSplitCost(2));
    EXPECT_EQ(whole.splitRowTile, 0);
    EXPECT_EQ(whole.splits, 3);
    // 128 tiles under one round of 228 units, 432 k-steps: 5 splits run 3 rounds of 87 + 14.
    const auto under
        = conv_igemm::flexTailSplit(32, 4, 432, 228, true, conv_igemm::flexSplitCost(2));
    EXPECT_EQ(under.splitRowTile, 0);
    EXPECT_EQ(under.splits, 5);
    // Whole rounds exactly, or a reduction too short to pay a split: unsplit.
    EXPECT_EQ(
        conv_igemm::flexTailSplit(228, 2, 216, 228, true, conv_igemm::flexSplitCost(2)).splits, 1);
    EXPECT_EQ(conv_igemm::flexTailSplit(32, 4, 16, 228, true, conv_igemm::flexSplitCost(2)).splits,
              1);
    // The register-staged plan of a single row tile splits all of it, partials of the whole
    // output.
    auto deep = dense(512, 256, {3, 4, 5}, {3, 3, 3}, {1, 1, 1}, true);
    deep.n = 1;
    const auto problem = parsedProblem(deep, ConvDirection::FORWARD);
    const auto plan = conv_igemm::splitPlanFor(
        problem, 256, 128, 64, deviceOf("gfx942", 228), 0, false, 0, 1, 2);
    EXPECT_GT(plan.splits, 1);
    EXPECT_EQ(plan.splitRowTile, 0);
    EXPECT_EQ(plan.partialElems, plan.outElems);
    // An explicit split policy splits every row tile: partials of the whole output.
    const auto explicitPlan = conv_igemm::splitPlanFor(
        problem, 256, 128, 64, deviceOf("gfx942", 228), 0, false, 4, 1, 2);
    EXPECT_EQ(explicitPlan.splits, 4);
    EXPECT_EQ(explicitPlan.splitRowTile, 0);
    EXPECT_EQ(explicitPlan.partialElems, explicitPlan.outElems);
}

TEST(TestConvIgemmCatalog, FlexWideRegisterTileRunsUnsplitOnGridsOfFewerTilesThanUnits)
{
    // 200 row tiles of one 256-wide column tile over 432 k-steps (512 channels): on 200 units
    // the default plan does not split and is served (its no-split twin, the same plan, is not);
    // on 400 the default plan splits and declines, and the no-split twin is served.
    auto wide = dense(512, 48, {5, 40, 256}, {3, 3, 3}, {1, 1, 1}, true);
    wide.n = 1;
    wide.dataType = DataType::HALF;
    const auto pack = packFor(ConvDirection::FORWARD);
    int checked = 0;
    for(const auto& kernel : catalog(ConvDirection::FORWARD))
    {
        const auto policy = kernel.getIntMetadata("split_policy");
        if(kernel.getIntMetadata("flex") == 0 || kernel.getIntMetadata("bn") != 256
           || kernel.getStringMetadata("atom") != "mfma16_f16" || policy > 1)
        {
            continue;
        }
        ++checked;
        for(const auto& [units, filled] :
            std::initializer_list<std::pair<int, bool>>{{200, true}, {400, false}})
        {
            SCOPED_TRACE(units);
            const GraphFixture fixture(buildGraph(specFor(wide, ConvDirection::FORWARD)),
                                       deviceOf("gfx942:sramecc+:xnack-", units));
            const auto bound = matchesGraph(pack, fixture.context());
            ASSERT_TRUE(bound.has_value());
            EXPECT_EQ(matchesKernel(pack, fixture.context(), kernel, *bound),
                      policy == 0 ? filled : !filled);
        }
    }
    EXPECT_EQ(checked, 2);
}

TEST(TestConvIgemmCatalog, FlexWideRegisterTileKeepsBothPlansInBackwardData)
{
    // Backward data of 512 into 256 channels over 200 x 256 rows on 400 units: the default plan
    // splits and stays served, and so does its no-split twin.
    auto wide = dense(256, 512, {5, 40, 256}, {3, 3, 3}, {1, 1, 1}, true);
    wide.n = 1;
    wide.dataType = DataType::HALF;
    const auto pack = packFor(ConvDirection::BACKWARD_DATA);
    const GraphFixture fixture(buildGraph(specFor(wide, ConvDirection::BACKWARD_DATA)),
                               deviceOf("gfx942:sramecc+:xnack-", 400));
    const auto bound = matchesGraph(pack, fixture.context());
    ASSERT_TRUE(bound.has_value());
    int checked = 0;
    for(const auto& kernel : catalog(ConvDirection::BACKWARD_DATA))
    {
        const auto policy = kernel.getIntMetadata("split_policy");
        if(kernel.getIntMetadata("flex") == 0 || kernel.getIntMetadata("bn") != 256
           || kernel.getStringMetadata("atom") != "mfma16_f16" || policy > 1)
        {
            continue;
        }
        SCOPED_TRACE(policy);
        ++checked;
        EXPECT_TRUE(matchesKernel(pack, fixture.context(), kernel, *bound));
    }
    EXPECT_EQ(checked, 2);
}

TEST(TestConvIgemmCatalog, FlexConfigsAreGfx950OrGfx942TilesThatDeclineEveryOtherKnob)
{
    auto shape = dense(32, 48, {5, 6, 7}, {3, 3, 3}, {1, 1, 1}, true);
    shape.dataType = DataType::HALF;
    int flex = 0;
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        const auto pack = packFor(direction);
        const GraphFixture gfx950(buildGraph(specFor(shape, direction)), deviceOf("gfx950", 256));
        const GraphFixture gfx942(buildGraph(specFor(shape, direction)),
                                  deviceOf("gfx942:sramecc+:xnack-", 304));
        // The register-staged 256 x 256 tile serves grids of a block a unit or more: 320 row
        // tiles.
        auto wide = dense(32, 48, {5, 64, 256}, {3, 3, 3}, {1, 1, 1}, true);
        wide.dataType = DataType::HALF;
        const GraphFixture gfx942Wide(buildGraph(specFor(wide, direction)),
                                      deviceOf("gfx942:sramecc+:xnack-", 304));
        const auto bound950 = matchesGraph(pack, gfx950.context());
        const auto bound942 = matchesGraph(pack, gfx942.context());
        const auto bound942Wide = matchesGraph(pack, gfx942Wide.context());
        ASSERT_TRUE(bound950.has_value() && bound942.has_value() && bound942Wide.has_value());
        for(const auto& kernel : catalog(direction))
        {
            // Split-policy twins are served only where their plan differs from the default's.
            if(kernel.getIntMetadata("flex") == 0 || kernel.getIntMetadata("split_policy") != 0)
            {
                continue;
            }
            SCOPED_TRACE(kernel.name);
            // The LDS-DMA form (16x16x32 atoms) is gfx950's, the register-staged form gfx942's:
            // 128-byte k-steps, or 64-byte ones on the register-staged 16-bit 256-wide tile.
            const bool dma = kernel.getStringMetadata("atom").rfind("mfma16x32_", 0) == 0;
            const bool fp32 = kernel.getStringMetadata("dtype") == "FLOAT";
            const bool wideReg = !dma && kernel.getIntMetadata("bn") == 256;
            EXPECT_EQ(kernel.getIntMetadata("flex"), 1);
            EXPECT_EQ(kernel.getIntMetadata("bm"), 256);
            EXPECT_EQ(kernel.getIntMetadata("bk"), wideReg || fp32 ? 32 : 64);
            EXPECT_TRUE(!wideReg || !fp32);
            EXPECT_EQ(kernel.getIntMetadata("xcd"), 1);
            // The register-staged form's backward data may read the graph's filter in place.
            const bool direct = kernel.getIntMetadata("wdirect") != 0;
            EXPECT_TRUE(!direct || (!dma && direction == ConvDirection::BACKWARD_DATA));
            if(kernel.getStringMetadata("dtype") != "HALF")
            {
                continue;
            }
            ++flex;
            const auto& home = dma ? gfx950 : wideReg ? gfx942Wide : gfx942;
            const auto& homeBound = dma ? *bound950 : wideReg ? *bound942Wide : *bound942;
            EXPECT_TRUE(matchesKernel(pack, home.context(), kernel, homeBound));
            EXPECT_FALSE(matchesKernel(
                pack, (dma ? gfx942 : gfx950).context(), kernel, dma ? *bound942 : *bound950));
            for(const auto& [field, value] :
                std::initializer_list<std::pair<const char*, int64_t>>{{"flex", 2},
                                                                       {"bm", 128},
                                                                       {"bn", 64},
                                                                       {"bk", 16},
                                                                       {"wm", 2},
                                                                       {"wn", 4},
                                                                       {"vec", 4},
                                                                       {"stages", 2},
                                                                       {"xcd", 0},
                                                                       {"halo", 1},
                                                                       {"wdirect", 1},
                                                                       {"pack", 2},
                                                                       {"cf", 1},
                                                                       {"swz", 1},
                                                                       {"dma", 2},
                                                                       {"fuse", 2},
                                                                       {"large", 1},
                                                                       {"patch", 1},
                                                                       {"cfx", 1}})
            {
                SCOPED_TRACE(field);
                if(std::string_view(field) == "wdirect" && !dma
                   && direction == ConvDirection::BACKWARD_DATA)
                {
                    continue;
                }
                auto mutated = kernel;
                mutated.metadata[field] = value;
                EXPECT_FALSE(matchesKernel(pack, home.context(), mutated, homeBound));
            }
            auto atom = kernel;
            atom.metadata["atom"] = std::string("mfma32x16_f16");
            EXPECT_FALSE(matchesKernel(pack, home.context(), atom, homeBound));
        }
        // Backward data in stride phases and filters past ten taps on an axis decline.
        auto strided = shape;
        strided.stride = {2, 2, 2};
        auto longFilter = dense(32, 48, {5, 6, 13}, {1, 1, 11}, {0, 0, 5}, true);
        longFilter.dataType = DataType::HALF;
        for(const auto& [declined, forwardToo] :
            std::initializer_list<std::pair<ConvShape, bool>>{{strided, false}, {longFilter, true}})
        {
            if(!forwardToo && direction == ConvDirection::FORWARD)
            {
                continue;
            }
            for(const auto& device :
                {deviceOf("gfx950", 256), deviceOf("gfx942:sramecc+:xnack-", 304)})
            {
                const GraphFixture fixture(buildGraph(specFor(declined, direction)), device);
                const auto bound = matchesGraph(pack, fixture.context());
                ASSERT_TRUE(bound.has_value());
                for(const auto& kernel : catalog(direction))
                {
                    if(kernel.getIntMetadata("flex") != 0)
                    {
                        EXPECT_FALSE(matchesKernel(pack, fixture.context(), kernel, *bound))
                            << kernel.name;
                    }
                }
            }
        }
        // A block whose images span past 2^31 bytes declines (32-bit buffer offsets).
        auto huge = dense(64, 64, {4096, 4096}, {3, 3}, {1, 1}, true);
        huge.n = 1;
        huge.dataType = DataType::HALF;
        for(const auto& device : {deviceOf("gfx950", 256), deviceOf("gfx942:sramecc+:xnack-", 304)})
        {
            const GraphFixture big(buildGraph(specFor(huge, direction)), device);
            const auto bigBound = matchesGraph(pack, big.context());
            ASSERT_TRUE(bigBound.has_value());
            for(const auto& kernel : catalog(direction))
            {
                if(kernel.getIntMetadata("flex") != 0)
                {
                    EXPECT_FALSE(matchesKernel(pack, big.context(), kernel, *bigBound))
                        << kernel.name;
                }
            }
        }
    }
    EXPECT_GT(flex, 0);
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
        const int64_t imageBytes = int64_t{64} * 64 * 64 * 2;
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

// ---------------------------------------------------------------------------
// Fused pointwise epilogues (forward): every code path that writes y -- the LDS-staged and
// channels-first epilogues, the FMA store, the split-K reduce, group fusion, halo, tap packing,
// the TF32 forms -- once per distinct path among the configs the kernel match admits.
// ---------------------------------------------------------------------------

using conv_epilogue_test::Specials;
using conv_epilogue_test::Step;
using FusedChain = std::pair<std::string, std::vector<Step>>;

/// The config fields that select a different store path.
const std::vector<std::string>& fusedPathFields()
{
    static const std::vector<std::string> s_fields{
        "atom", "math", "cf", "pack", "halo", "fuse", "dma", "swz", "wdirect", "large", "patch"};
    return s_fields;
}

struct FusedShape
{
    std::string name;
    ConvShape shape;
};

std::vector<FusedShape> fusedShapes()
{
    std::vector<FusedShape> shapes;
    shapes.push_back({"NhwcEdge", dense(24, 40, {9, 7}, {3, 3}, {1, 1}, true)});
    shapes.push_back({"NchwEdge", dense(24, 40, {9, 7}, {3, 3}, {1, 1}, false)});
    // A long reduction over few rows: the split-K plan splits, and the reduce writes y.
    ConvShape split = dense(512, 48, {4, 4}, {3, 3}, {1, 1}, true);
    split.n = 1;
    shapes.push_back({"SplitK", split});
    ConvShape splitCf = split;
    splitCf.channelsLast = false;
    shapes.push_back({"SplitKChannelsFirst", splitCf});
    ConvShape grouped = dense(32, 32, {6, 6}, {3, 3}, {1, 1}, true);
    grouped.groups = 4;
    shapes.push_back({"Grouped", grouped});
    shapes.push_back({"OneD", dense(16, 24, {13}, {3}, {1}, true)});
    shapes.push_back({"ThreeD", dense(8, 16, {3, 5, 6}, {3, 3, 3}, {1, 1, 1}, true)});
    // Patch convolutions: the patch view of the general loop and the patch tile (8- and 4-wide
    // channel runs), channels-last and channels-first.
    shapes.push_back({"Patch16x16Rgb", patchShape(3, 40, {32, 48}, 16, true)});
    shapes.push_back({"Patch4x4Rgb", patchShape(3, 24, {12, 20}, 4, true)});
    shapes.push_back({"Patch2x2ChannelsFirst", patchShape(16, 24, {6, 10}, 2, false)});
    // Native gfx1250 cf=2: no activation staging, with channel/output tails. The long
    // reduction also exercises the fused split-K reduce on its channels-first output.
    for(const int64_t c : {40, 520})
    {
        auto pointwise = dense(c, 49, {5, 8}, {1, 1}, {0, 0}, false);
        pointwise.n = 1;
        shapes.push_back(
            {c == 40 ? "PointwiseChannelsFirst" : "PointwiseChannelsFirstSplitK", pointwise});
    }
    return shapes;
}

template <typename T>
void runFused(const ConvShape& shape, const std::vector<Step>& steps, Specials specials)
{
    const auto runner = conv_epilogue_test::catalogRunner(
        packFor(ConvDirection::FORWARD), catalog(ConvDirection::FORWARD), fusedPathFields());
    // The TF32 forms (three-bf16, xf32) carry their own error: 2^-11 relative per product.
    const auto tolerance = [](int64_t reduction) {
        return (std::is_same_v<T, float> ? 1e-3 : 1e-6) * static_cast<double>(reduction);
    };
    EXPECT_GE(conv_epilogue_test::checkFusedCase<T>(shape, steps, runner, specials, tolerance), 1)
        << "no catalog config serves this fused case";
}

class TestGpuConvIgemmFused : public ::testing::TestWithParam<std::tuple<FusedShape, FusedChain>>
{
};

TEST_P(TestGpuConvIgemmFused, Float)
{
    SKIP_IF_NO_DEVICES();
    runFused<float>(std::get<0>(GetParam()).shape, std::get<1>(GetParam()).second, Specials::NONE);
}

TEST_P(TestGpuConvIgemmFused, Half)
{
    SKIP_IF_NO_DEVICES();
    runFused<Half>(std::get<0>(GetParam()).shape, std::get<1>(GetParam()).second, Specials::NONE);
}

TEST_P(TestGpuConvIgemmFused, BFloat16)
{
    SKIP_IF_NO_DEVICES();
    runFused<BFloat16>(
        std::get<0>(GetParam()).shape, std::get<1>(GetParam()).second, Specials::NONE);
}

std::vector<FusedChain> fusedChainsAllShapes()
{
    const auto menu = conv_epilogue_test::menuChains();
    return {menu[0], menu[6]}; // bias + relu, bias + swish
}

INSTANTIATE_TEST_SUITE_P(Shapes,
                         TestGpuConvIgemmFused,
                         ::testing::Combine(::testing::ValuesIn(fusedShapes()),
                                            ::testing::ValuesIn(fusedChainsAllShapes())),
                         [](const auto& info) {
                             return std::get<0>(info.param).name + "_"
                                    + std::get<1>(info.param).first;
                         });

INSTANTIATE_TEST_SUITE_P(Menu,
                         TestGpuConvIgemmFused,
                         ::testing::Combine(::testing::Values(fusedShapes()[0], fusedShapes()[1]),
                                            ::testing::ValuesIn(conv_epilogue_test::menuChains())),
                         [](const auto& info) {
                             return std::get<0>(info.param).name + "_"
                                    + std::get<1>(info.param).first;
                         });

class TestGpuConvIgemmFusedSpecials : public ::testing::TestWithParam<FusedShape>
{
};

TEST_P(TestGpuConvIgemmFusedSpecials, NanAndInfinityOperands)
{
    SKIP_IF_NO_DEVICES();
    runFused<float>(GetParam().shape,
                    {conv_epilogue_test::bias(), conv_epilogue_test::relu()},
                    Specials::NAN_INF);
    runFused<Half>(
        GetParam().shape,
        {conv_epilogue_test::bias(DataType::HALF),
         conv_epilogue_test::unary(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SWISH_FWD)},
        Specials::NAN_INF);
}

TEST_P(TestGpuConvIgemmFusedSpecials, ClipBoundaries)
{
    SKIP_IF_NO_DEVICES();
    runFused<Half>(
        GetParam().shape,
        {conv_epilogue_test::bias(DataType::HALF), conv_epilogue_test::relu(-0.5F, 0.5F)},
        Specials::CLIP_BOUNDARY);
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuConvIgemmFusedSpecials,
                         ::testing::ValuesIn(fusedShapes()),
                         [](const auto& info) { return info.param.name; });

// A fused plan's staging helpers (copy, transpose) are built from the same options with the
// epilogue defines removed, so they must reuse the unfused plan's module rather than compile the
// same source again under a new key: after the unfused plan, the fused one compiles exactly one
// program, its fused convolution.
TEST(TestGpuConvIgemmFused, StagingHelpersReuseTheUnfusedModule)
{
    SKIP_IF_NO_DEVICES();
    // Channels-first, so x and y stage through the helpers.
    const auto shape = dense(24, 40, {9, 7}, {3, 3}, {1, 1}, false);
    const auto pack = packFor(ConvDirection::FORWARD);
    const auto& handler = dispatchHandler(pack);

    const GraphFixture plain(buildGraph(specFor(shape, ConvDirection::FORWARD)),
                             currentDeviceProperties());
    const GraphFixture fused(conv_epilogue_test::buildFusedGraph(conv_epilogue_test::fusedSpecFor(
                                 shape, {conv_epilogue_test::bias(), conv_epilogue_test::relu()})),
                             currentDeviceProperties());
    const auto plainBound = matchesGraph(pack, plain.context());
    const auto fusedBound = matchesGraph(pack, fused.context());
    ASSERT_TRUE(plainBound.has_value() && fusedBound.has_value());

    // The first config both graphs admit.
    const auto& kernels = catalog(ConvDirection::FORWARD);
    const auto kernel = std::find_if(kernels.begin(), kernels.end(), [&](const auto& candidate) {
        return matchesKernel(pack, plain.context(), candidate, *plainBound)
               && matchesKernel(pack, fused.context(), candidate, *fusedBound);
    });
    ASSERT_NE(kernel, kernels.end());

    {
        const compilation::ProgramFetchRecording recording;
        ASSERT_NE(handler.prepare(plain.context(), *plainBound, *kernel), nullptr);
        ASSERT_GE(recording.fetches().size(), 2u) << "the plan stages nothing";
    }
    const compilation::ProgramFetchRecording recording;
    ASSERT_NE(handler.prepare(fused.context(), *fusedBound, *kernel), nullptr);
    std::string sources;
    size_t built = 0;
    for(const auto& fetch : recording.fetches())
    {
        sources
            += fetch.program + " " + std::string(compilation::codeSourceName(fetch.source)) + "; ";
        built += fetch.source == compilation::CodeSource::IN_PROCESS ? 0 : 1;
    }
    EXPECT_LE(built, 1u) << kernel->name << ": " << sources;
    EXPECT_GE(recording.fetches().size(), 2u) << sources;
}

// Opt-in: each config's time with and without a fused bias + relu (shapes in $K1_EPI_TIMING).
TEST(TestGpuConvIgemmFusedTiming, BareAgainstFused)
{
    SKIP_IF_NO_DEVICES();
    conv_epilogue_timing::timeFusedAgainstBare(
        packFor(ConvDirection::FORWARD), catalog(ConvDirection::FORWARD), "K1_EPI_TIMING");
}

} // namespace

#endif
