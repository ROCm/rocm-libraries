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
#include "engines/kernel_ingestor_engine/packs/ConvGroupProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvTf32TestSupport.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvTf32Timing.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestGpuConvGroup.cpp
 * @brief The small-group convolution kernels (CONV_GROUP_FWD, CONV_GROUP_BWD, CONV_GROUP_WRW) on
 *        the device,
 *        through the engines' own graph match, kernel match, prepare and launch, against
 *        hipDNN's CPU reference convolution; and their graph and kernel gates on the host.
 *
 * Every config of the installed catalog that the kernel match admits on this device runs on
 * every case, forward, backward data and backward weights (groups up to 16 channels), in fp32,
 * fp16 and bf16: groups of 2, 4, 8 and 16
 * channels and wide groups of 24 to 56 (padded to whole 16-channel blocks), both layouts (channels-first staged and, cf configs, in place), 2-D and 3-D (ResNeXt3D's 3x3x3 and 1x3x3 filters), stride 2 (backward
 * data by stride phases, depth included), asymmetric padding, extents that are not tile
 * multiples, several channel blocks, and true convolution (the expansion reflects the filter).
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
            direction == ConvDirection::FORWARD         ? "hipkernel.conv_group_fwd.graph_match"
            : direction == ConvDirection::BACKWARD_DATA ? "hipkernel.conv_group_bwd.graph_match"
                                                        : "hipkernel.conv_group_wrw.graph_match",
            "",
            "hipkernel.conv_group.kernel_match",
            "",
            "hipkernel.conv_group.dispatch",
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
        std::array<std::vector<hipdnn_plugin_sdk::ingestor::KernelDefinition>, 3> catalogs;
        for(size_t i = 0; i < catalogs.size(); ++i)
        {
            // A direction's packs: conv_group_<tag>.kdp.json, or (backward weights, split by
            // element type and atom family) conv_group_wrw_<suffix>.kdp.json.
            const std::string tag = i == 0 ? "fwd" : i == 1 ? "bwd" : "wrw";
            const auto isPack = [&](const std::string& name) {
                return name == "conv_group_" + tag + ".kdp.json"
                       || (name.rfind("conv_group_" + tag + "_", 0) == 0 && name.size() > 9
                           && name.substr(name.size() - 9) == ".kdp.json");
            };
            std::vector<std::filesystem::path> found;
            for(const auto& root : roots)
            {
                std::error_code error;
                for(auto it = std::filesystem::recursive_directory_iterator(root, error);
                    !error && it != std::filesystem::recursive_directory_iterator();
                    it.increment(error))
                {
                    if(isPack(it->path().filename().string()))
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
                throw std::runtime_error("no conv_group_" + tag
                                         + " pack beside the provider or this test binary");
            }
            std::sort(found.begin(), found.end());
            for(const auto& path : found)
            {
                std::ifstream stream(path);
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
    return s_catalogs[direction == ConvDirection::FORWARD         ? 0
                      : direction == ConvDirection::BACKWARD_DATA ? 1
                                                                  : 2];
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

std::vector<DeviceCase> deviceCases()
{
    std::vector<DeviceCase> cases;
    // ResNeXt-like: groups of 4, 8 and 16 channels; extents that are not tile multiples.
    auto g4 = dense(64, 64, {9, 37}, {3, 3}, {1, 1}, true);
    g4.groups = 16;
    cases.push_back({"NhwcGroups4", g4});
    auto g8 = dense(64, 64, {7, 21}, {3, 3}, {1, 1}, true);
    g8.groups = 8;
    cases.push_back({"NhwcGroups8", g8});
    auto g16 = dense(64, 64, {6, 19}, {3, 3}, {1, 1}, true);
    g16.groups = 4;
    cases.push_back({"NhwcGroups16", g16});
    auto g2 = dense(32, 32, {5, 17}, {3, 3}, {1, 1}, true);
    g2.groups = 16;
    cases.push_back({"NhwcGroups2", g2});
    // Channels-first (staged), several 16-channel blocks.
    auto nchw = dense(128, 128, {9, 13}, {3, 3}, {1, 1}, false);
    nchw.groups = 32;
    cases.push_back({"NchwGroups4", nchw});
    // Stride 2 (ResNeXt's stage entries): backward data by stride phases.
    auto strided = dense(64, 64, {13, 34}, {3, 3}, {1, 1}, true);
    strided.groups = 16;
    strided.stride = {2, 2};
    cases.push_back({"NhwcGroups4Stride2", strided});
    // Wide groups (16 < cg <= 64): 32, 56 (padded to 64), 48 at stride 2, 24 in 3-D.
    auto wide32 = dense(64, 64, {9, 21}, {3, 3}, {1, 1}, true);
    wide32.groups = 2;
    cases.push_back({"NhwcGroups32", wide32});
    auto wide56 = dense(112, 112, {7, 19}, {3, 3}, {1, 1}, false);
    wide56.groups = 2;
    cases.push_back({"NchwGroups56", wide56});
    auto wide48 = dense(96, 96, {11, 17}, {3, 3}, {1, 1}, false);
    wide48.groups = 2;
    wide48.stride = {2, 2};
    cases.push_back({"NchwGroups48Stride2", wide48});
    auto wide32s2 = dense(64, 64, {10, 17}, {3, 3}, {1, 1}, true);
    wide32s2.groups = 2;
    wide32s2.stride = {2, 2};
    cases.push_back({"NhwcGroups32Stride2", wide32s2});
    auto wide24 = dense(48, 48, {4, 6, 13}, {3, 3, 3}, {1, 1, 1}, true);
    wide24.groups = 2;
    cases.push_back({"NdhwcGroups24", wide24});
    // A dx one row high: no second row phase (the phase-fused configs decline).
    auto thin = dense(32, 32, {1, 21}, {3, 3}, {1, 1}, true);
    thin.groups = 8;
    thin.stride = {2, 2};
    cases.push_back({"NhwcGroups4Stride2OneRow", thin});
    auto stridedFirst = dense(32, 32, {12, 19}, {3, 3}, {0, 1}, false);
    stridedFirst.groups = 4;
    stridedFirst.stride = {2, 2};
    cases.push_back({"NchwGroups8Stride2", stridedFirst});
    // Asymmetric and no padding.
    auto asymmetric = dense(32, 32, {10, 18}, {3, 3}, {0, 2}, true);
    asymmetric.prePadding = {2, 0};
    asymmetric.postPadding = {0, 1};
    asymmetric.groups = 8;
    cases.push_back({"NhwcGroups4AsymmetricPadding", asymmetric});
    auto valid = dense(32, 32, {8, 20}, {3, 3}, {0, 0}, false);
    valid.groups = 4;
    cases.push_back({"NchwGroups8NoPadding", valid});
    // True convolution: the expansion reflects the filter.
    auto convolution = dense(64, 64, {9, 17}, {3, 3}, {1, 1}, true);
    convolution.groups = 8;
    convolution.mode = ConvMode::CONVOLUTION;
    cases.push_back({"NhwcGroups8ConvolutionMode", convolution});
    // 3-D (ResNeXt3D): a 3x3x3 filter over planes, a 1x3x3 one, and stride 2 on every axis
    // (backward data by phases over depth too), both layouts.
    auto deep = dense(32, 32, {4, 7, 19}, {3, 3, 3}, {1, 1, 1}, true);
    deep.groups = 8;
    cases.push_back({"NdhwcGroups4", deep});
    auto flat = dense(64, 64, {3, 6, 11}, {1, 3, 3}, {0, 1, 1}, false);
    flat.groups = 8;
    cases.push_back({"NcdhwGroups8Filter1x3x3", flat});
    auto deepStrided = dense(32, 32, {5, 9, 18}, {3, 3, 3}, {1, 1, 1}, false);
    deepStrided.groups = 4;
    deepStrided.stride = {2, 2, 2};
    cases.push_back({"NcdhwGroups8Stride2", deepStrided});
    auto deepDepthStrided = dense(32, 32, {6, 7, 9}, {3, 3, 3}, {1, 1, 0}, true);
    deepDepthStrided.groups = 16;
    deepDepthStrided.stride = {2, 1, 1};
    cases.push_back({"NdhwcGroups2DepthStride2", deepDepthStrided});
    return cases;
}

/// Whether this device has a K4 atom for the case's element type, and the direction serves it.
bool served(const ConvShape& shape, ConvDirection direction)
{
    const auto arch = hipdnn_plugin_sdk::stripArchFeatures(currentDeviceProperties().gcnArchName);
    if(direction == ConvDirection::BACKWARD_WEIGHTS)
    {
        // Small groups: a matrix-core atom for the element type, and fp32 also on FMA everywhere.
        // Wide groups (17 to 32 channels): the MFMA atoms.
        const auto element = shape.dataType == DataType::FLOAT  ? "f32"
                             : shape.dataType == DataType::HALF ? "f16"
                                                                : "bf16";
        const int64_t cg = shape.c / shape.groups;
        if(cg > 16)
        {
            return cg <= 32
                   && (shape.dataType == DataType::FLOAT
                           ? conv_igemm::atomRunsOn("mfma16_f32", arch)
                       : shape.dataType == DataType::HALF
                           ? conv_igemm::atomRunsOn("mfma16_f16", arch)
                           : conv_igemm::atomRunsOn("mfma16_bf16", arch)
                                 || conv_igemm::atomRunsOn("mfma16_bf16_908", arch));
        }
        return shape.dataType == DataType::FLOAT || conv_wrw::runsMatrixAtomFor(element, arch);
    }
    if(direction == ConvDirection::BACKWARD_DATA
       && std::any_of(shape.stride.begin(), shape.stride.end(), [](int64_t s) { return s > 2; }))
    {
        return false;
    }
    // fp32 runs everywhere for small groups (the FMA configs where no fp32 16x16 atom exists);
    // wide groups need the fp32 atom.
    if(shape.dataType == DataType::FLOAT && shape.c / shape.groups > 16)
    {
        return conv_igemm::atomRunsOn("mfma16_f32", arch);
    }
    return true;
}

/// The configs a run takes: the graph's own element type, or (fp32 under TF32 permission) the
/// three-bf16 ones.
enum class Math
{
    NATIVE,
    THREE_BF16,
};

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
    if(direction == ConvDirection::BACKWARD_WEIGHTS && shape.c / shape.groups > 32)
    {
        // Groups past 32 channels are K3's.
        EXPECT_FALSE(bound.has_value());
        return;
    }
    ASSERT_TRUE(bound.has_value()) << "the engine refused a case it should serve";

    Tensor<T> x(spec.x.dims, spec.x.strides);
    Tensor<T> w(spec.w.dims, spec.w.strides);
    Tensor<T> y(spec.y.dims, spec.y.strides);
    constexpr unsigned int SEED = 1234;
    const bool forward = direction == ConvDirection::FORWARD;
    const bool weights = direction == ConvDirection::BACKWARD_WEIGHTS;
    (forward || weights ? x : y).fillTensorWithRandomValues(-1.0f, 1.0f, SEED);
    (weights ? y : w).fillTensorWithRandomValues(-1.0f, 1.0f, SEED + 1);
    if(infinities)
    {
        conv_tf32_test::plantInfinities(forward || weights ? x : y);
    }
    Tensor<T>& written = forward ? y : weights ? w : x;

    const bool reflect = shape.mode == ConvMode::CONVOLUTION;
    Tensor<T> reflectedW(spec.w.dims, spec.w.strides);
    if(reflect)
    {
        reflectFilter(w, reflectedW);
    }
    const Tensor<T>& filterForReference = reflect ? reflectedW : w;
    Tensor<T> reference(written.dims(), written.strides());
    if(weights)
    {
        // True convolution: dw of the reflected filter, reflected back.
        Tensor<T> raw(written.dims(), written.strides());
        CpuFpReferenceConvolution::wgrad(x,
                                         reflect ? raw : reference,
                                         y,
                                         shape.stride,
                                         shape.dilation,
                                         shape.prePadding,
                                         shape.postPadding);
        if(reflect)
        {
            reflectFilter(raw, reference);
        }
    }
    else if(forward)
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
    // Each output sums (taps x reduction channels) products of values in [-1, 1]; backward
    // weights (images x output positions) of them.
    int64_t reductionLength = (forward ? shape.c : shape.k) / shape.groups;
    for(const auto extent : shape.filter)
    {
        reductionLength *= extent;
    }
    if(weights)
    {
        reductionLength = shape.n;
        for(const auto extent : shape.output())
        {
            reductionLength *= extent;
        }
    }
    const float accumulationBound
        = (math == Math::THREE_BF16 ? conv_tf32_test::THREE_BF16_BOUND : 2e-6f)
          * static_cast<float>(reductionLength);

    const auto& handler = dispatchHandler(pack);
    int ran = 0;
    for(const auto& kernel : catalog(direction))
    {
        if(!matchesKernel(pack, fixture.context(), kernel, *bound)
           || (kernel.getIntMetadata("math") == 1) != (math == Math::THREE_BF16))
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
            if(!conv_tf32_test::matches(got, want, tolerance) && ++mismatches <= 5)
            {
                ADD_FAILURE() << "at linear index " << offsetOf(index, written.strides())
                              << ": got " << got << ", want " << want << ", tolerance "
                              << tolerance;
            }
        });
        EXPECT_EQ(mismatches, 0);
    }
    if(math == Math::THREE_BF16)
    {
        // The three-bf16 subset need not cover every case (the catalog test checks it serves).
        if(ran == 0)
        {
            GTEST_SKIP() << "no three-bf16 config serves this case";
        }
        return;
    }
    // Backward data past stride 2 is not served; backward weights needs an atom for 16-bit types.
    if(served(shape, direction))
    {
        EXPECT_GE(ran, 1) << "no catalog config serves this case";
    }
}

class TestGpuConvGroup : public ::testing::TestWithParam<std::tuple<DeviceCase, ConvDirection>>
{
};

TEST_P(TestGpuConvGroup, Float)
{
    SKIP_IF_NO_DEVICES();
    runCase<float>(std::get<0>(GetParam()).shape, std::get<1>(GetParam()));
}

TEST_P(TestGpuConvGroup, Half)
{
    SKIP_IF_NO_DEVICES();
    runCase<Half>(std::get<0>(GetParam()).shape, std::get<1>(GetParam()));
}

TEST_P(TestGpuConvGroup, BFloat16)
{
    SKIP_IF_NO_DEVICES();
    runCase<BFloat16>(std::get<0>(GetParam()).shape, std::get<1>(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuConvGroup,
                         ::testing::Combine(::testing::ValuesIn(deviceCases()),
                                            ::testing::Values(ConvDirection::FORWARD,
                                                              ConvDirection::BACKWARD_DATA,
                                                              ConvDirection::BACKWARD_WEIGHTS)),
                         [](const auto& info) {
                             const auto direction = std::get<1>(info.param);
                             return std::get<0>(info.param).name
                                    + (direction == ConvDirection::FORWARD         ? "Fwd"
                                       : direction == ConvDirection::BACKWARD_DATA ? "Bwd"
                                                                                   : "Wrw");
                         });

/// The three-bf16 fp32 configs (TF32 permission; wide groups, forward and backward data) at
/// three-bf16's bound, and with +inf and -inf in the kernel's input (each must reach the output
/// as itself, NaN where both meet). Small-group cases admit none and skip.
class TestGpuConvGroupThreeBf16
    : public ::testing::TestWithParam<std::tuple<DeviceCase, ConvDirection>>
{
};

TEST_P(TestGpuConvGroupThreeBf16, Float)
{
    SKIP_IF_NO_DEVICES();
    if(!conv_tf32_test::hasThreeBf16(currentDeviceProperties().gcnArchName))
    {
        GTEST_SKIP() << "no three-bf16 configs on this device";
    }
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter tf32(conv_tf32_test::TF32_ENV,
                                                                           "1");
    runCase<float>(std::get<0>(GetParam()).shape, std::get<1>(GetParam()), Math::THREE_BF16);
}

TEST_P(TestGpuConvGroupThreeBf16, FloatInfinities)
{
    SKIP_IF_NO_DEVICES();
    const auto& shape = std::get<0>(GetParam()).shape;
    if(!conv_tf32_test::hasThreeBf16(currentDeviceProperties().gcnArchName))
    {
        GTEST_SKIP() << "no three-bf16 configs on this device";
    }
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter tf32(conv_tf32_test::TF32_ENV,
                                                                           "1");
    runCase<float>(shape, std::get<1>(GetParam()), Math::THREE_BF16, true);
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuConvGroupThreeBf16,
                         ::testing::Combine(::testing::ValuesIn(deviceCases()),
                                            ::testing::Values(ConvDirection::FORWARD,
                                                              ConvDirection::BACKWARD_DATA)),
                         [](const auto& info) {
                             return std::get<0>(info.param).name
                                    + (std::get<1>(info.param) == ConvDirection::FORWARD ? "Fwd"
                                                                                         : "Bwd");
                         });

/// Opt-in (GROUP_TF32_TIMING=<shape CSV>, its direction column choosing the pack): the best
/// exact and the best three-bf16 config per shape, TF32 permitted. See ConvTf32Timing.hpp.
TEST(TestGpuConvGroupTiming, ExactAgainstThreeBf16)
{
    SKIP_IF_NO_DEVICES();
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter tf32(conv_tf32_test::TF32_ENV,
                                                                           "1");
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        conv_tf32_timing::timeByForm(
            packFor(direction),
            direction,
            catalog(direction),
            "GROUP_TF32_TIMING",
            [](const auto& kernel) {
                return std::string(kernel.getIntMetadata("math") == 1 ? "bf16x3" : "exact");
            },
            direction == ConvDirection::FORWARD ? "@fwd" : "@bwd");
    }
}

// ---- Host-only: the graph gates, one decline per field, and the kernel gates ----

/// Without TF32 permission no three-bf16 config is admitted; with it, fp32 wide-group graphs
/// admit some forward and backward data on every arch with bf16 matrix cores (not gfx908), and
/// bf16 graphs, small groups and backward weights none.
TEST(TestConvGroupCatalog, ThreeBf16OnlyUnderTf32Permission)
{
    // RegNet-like wide groups (32 channels), and ResNeXt-like small ones (4).
    auto wide = dense(64, 64, {12, 13}, {3, 3}, {1, 1}, true);
    wide.groups = 2;
    auto small = wide;
    small.groups = 16;
    for(const auto& item :
        std::vector<std::pair<ConvDirection, ConvShape>>{{ConvDirection::FORWARD, wide},
                                                         {ConvDirection::BACKWARD_DATA, wide},
                                                         {ConvDirection::FORWARD, small},
                                                         {ConvDirection::BACKWARD_DATA, small},
                                                         {ConvDirection::BACKWARD_WEIGHTS, wide}})
    {
        const auto direction = item.first;
        const auto& shape = item.second;
        for(const char* arch : {"gfx90a", "gfx942", "gfx950", "gfx1100", "gfx1201", "gfx908"})
        {
            for(const auto dataType : {DataType::FLOAT, DataType::BFLOAT16})
            {
                SCOPED_TRACE(std::string(arch) + (dataType == DataType::FLOAT ? " fp32" : " bf16")
                             + " direction " + std::to_string(static_cast<int>(direction))
                             + " groups " + std::to_string(shape.groups));
                auto graphShape = shape;
                graphShape.dataType = dataType;
                auto device = testDeviceProperties();
                device.gcnArchName = arch;
                device.warpSize = std::string(arch).rfind("gfx1", 0) == 0 ? 32 : 64;
                // The wrw launch planner sizes its grid from the units.
                device.multiProcessorCount = 64;
                const auto pack = packFor(direction);
                const GraphFixture fixture(buildGraph(specFor(graphShape, direction)), device);
                const auto bound = matchesGraph(pack, fixture.context());
                ASSERT_TRUE(bound.has_value());
                const auto admitted = [&] {
                    int count = 0;
                    for(const auto& kernel : catalog(direction))
                    {
                        count += kernel.getIntMetadata("math") == 1
                                 && matchesKernel(pack, fixture.context(), kernel, *bound);
                    }
                    return count;
                };
                {
                    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter off(
                        conv_tf32_test::TF32_ENV, "");
                    EXPECT_EQ(admitted(), 0);
                }
                const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter on(
                    conv_tf32_test::TF32_ENV, "1");
                const bool expected = dataType == DataType::FLOAT && std::string(arch) != "gfx908"
                                      && shape.groups == wide.groups
                                      && direction != ConvDirection::BACKWARD_WEIGHTS;
                EXPECT_EQ(admitted() > 0, expected);
            }
        }
    }
}

std::optional<conv_group::GroupProblem>
    parsed(ConvShape shape, ConvDirection direction, DataType dataType = DataType::HALF)
{
    shape.dataType = dataType;
    const GraphFixture fixture(buildGraph(specFor(shape, direction)));
    return conv_group::parseGroupProblem(fixture.context(), direction);
}

ConvShape resnext()
{
    auto shape = dense(64, 64, {12, 13}, {3, 3}, {1, 1}, true);
    shape.groups = 16; // 4 channels a group
    return shape;
}

TEST(TestConvGroupProblem, ServesWideGroupsPaddedToWholeBlocks)
{
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        auto shape = dense(112, 112, {12, 13}, {3, 3}, {1, 1}, true);
        shape.groups = 2; // 56 channels a group
        const auto problem = parsed(shape, direction);
        ASSERT_TRUE(problem.has_value());
        EXPECT_EQ(problem->nbi, 4);
        EXPECT_EQ(problem->phases.front().expand.nbi, 4);
        EXPECT_EQ(problem->phases.front().args.cg, 56);
        // Dense per group: 2 groups x 4 output blocks x 9 taps x 4 input blocks x 16 x 16.
        EXPECT_EQ(conv_group::expandedElements(*problem), 2u * 4 * 9 * 4 * 256);
        // Small groups keep the block-diagonal layout.
        const auto small = parsed(resnext(), direction);
        ASSERT_TRUE(small.has_value());
        EXPECT_EQ(small->nbi, 0);
        EXPECT_EQ(conv_group::expandedElements(*small), 4u * 9 * 256);
    }
}

TEST(TestConvGroupProblem, ServesSmallGroupsBothDirections)
{
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        const auto problem = parsed(resnext(), direction);
        ASSERT_TRUE(problem.has_value());
        ASSERT_EQ(problem->phases.size(), 1u);
        const auto& phase = problem->phases.front();
        EXPECT_EQ(phase.args.cg, 4);
        EXPECT_EQ(phase.args.c, 64);
        EXPECT_EQ(problem->filterT, 1);
        EXPECT_EQ(problem->kernelT, 1);
        EXPECT_EQ(problem->filterR, 3);
        EXPECT_EQ(problem->kernelR, 3);
        EXPECT_EQ(phase.args.dout, 1);
        EXPECT_EQ(phase.expand.tc, 1);
        EXPECT_EQ(phase.expand.swap, direction == ConvDirection::FORWARD ? 0 : 1);
        // Backward data reflects the taps: tap q reads filter tap 2 - q.
        EXPECT_EQ(phase.expand.rBase, direction == ConvDirection::FORWARD ? 0 : 2);
        EXPECT_EQ(phase.expand.rStep, direction == ConvDirection::FORWARD ? 1 : -1);
        // Backward data gathers dy with leading padding R - 1 - ph.
        EXPECT_EQ(phase.args.ph, 1);
    }
    auto asymmetric = resnext();
    asymmetric.prePadding = {0, 2};
    const auto backward = parsed(asymmetric, ConvDirection::BACKWARD_DATA);
    ASSERT_TRUE(backward.has_value());
    EXPECT_EQ(backward->phases.front().args.ph, 2);
    EXPECT_EQ(backward->phases.front().args.pw, 0);
    // Stride 2: four phases of 2 x 2 taps, each writing every other dx row and column.
    auto strided = resnext();
    strided.stride = {2, 2};
    const auto phased = parsed(strided, ConvDirection::BACKWARD_DATA);
    ASSERT_TRUE(phased.has_value());
    EXPECT_EQ(phased->phases.size(), 4u);
    EXPECT_EQ(phased->kernelR, 2);
    EXPECT_EQ(phased->kernelStride, 1);
    for(const auto& phase : phased->phases)
    {
        EXPECT_EQ(phase.args.osh, 2);
        EXPECT_EQ(phase.args.outH, 12);
        EXPECT_EQ(phase.args.ho, (12 - phase.args.ooh + 1) / 2);
        EXPECT_EQ(phase.expand.rStep, -2);
    }
    // 3-D: forward loops the depth taps at the graph's depth stride; backward data at depth
    // stride 2 runs phases over depth too (2 x 2 x 2 of them, 2 depth taps each).
    auto deep = dense(64, 64, {5, 12, 13}, {3, 3, 3}, {1, 1, 1}, true);
    deep.groups = 16;
    deep.stride = {2, 1, 1};
    const auto deepForward = parsed(deep, ConvDirection::FORWARD);
    ASSERT_TRUE(deepForward.has_value());
    ASSERT_EQ(deepForward->phases.size(), 1u);
    EXPECT_EQ(deepForward->kernelT, 3);
    EXPECT_EQ(deepForward->phases.front().args.sd, 2);
    EXPECT_EQ(deepForward->phases.front().args.dout, 3);
    EXPECT_EQ(deepForward->phases.front().expand.tc, 3);
    deep.stride = {2, 2, 2};
    const auto deepBackward = parsed(deep, ConvDirection::BACKWARD_DATA);
    ASSERT_TRUE(deepBackward.has_value());
    EXPECT_EQ(deepBackward->phases.size(), 8u);
    EXPECT_EQ(deepBackward->kernelT, 2);
    for(const auto& phase : deepBackward->phases)
    {
        EXPECT_EQ(phase.args.sd, 1);
        EXPECT_EQ(phase.args.osd, 2);
        EXPECT_EQ(phase.args.outD, 5);
        EXPECT_EQ(phase.args.dout, (5 - phase.args.ood + 1) / 2);
        EXPECT_EQ(phase.expand.tStep, -2);
    }
}

TEST(TestConvGroupProblem, ChannelsFirstInPlaceNeedsBothActivationsPackedChannelsFirst)
{
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        const auto last = parsed(resnext(), direction);
        ASSERT_TRUE(last.has_value());
        EXPECT_FALSE(conv_group::channelsFirstInPlace(*last).has_value());
        auto shape = resnext();
        shape.channelsLast = false;
        const auto first = parsed(shape, direction);
        ASSERT_TRUE(first.has_value());
        ASSERT_TRUE(first->igemm.x.needed && first->igemm.y.needed);
        const auto inPlace = conv_group::channelsFirstInPlace(*first);
        ASSERT_TRUE(inPlace.has_value());
        EXPECT_FALSE(inPlace->igemm.x.needed);
        EXPECT_FALSE(inPlace->igemm.y.needed);
        EXPECT_EQ(inPlace->phases.front().expandOffset, 0u);
        EXPECT_LT(inPlace->workspaceBytes, first->workspaceBytes);
    }
}

TEST(TestConvGroupProblem, PhaseFusedLaunchesOneADepthPhase)
{
    auto strided = resnext();
    strided.stride = {2, 2};
    strided.prePadding = {1, 0};
    const auto backward = parsed(strided, ConvDirection::BACKWARD_DATA);
    ASSERT_TRUE(backward.has_value());
    ASSERT_EQ(backward->phases.size(), 4u);
    const auto launches = conv_group::groupLaunches(*backward, true);
    ASSERT_TRUE(launches.has_value());
    ASSERT_EQ(launches->size(), 1u);
    const auto& args = launches->front().args;
    EXPECT_EQ(launches->front().phase, 0u);
    EXPECT_EQ(args.fph[0], backward->phases[0].args.ph);
    EXPECT_EQ(args.fph[1], backward->phases[2].args.ph);
    EXPECT_EQ(args.fpw[0], backward->phases[0].args.pw);
    EXPECT_EQ(args.fpw[1], backward->phases[1].args.pw);
    EXPECT_EQ(static_cast<size_t>(args.wps) * 2,
              backward->phases[1].expandOffset - backward->phases[0].expandOffset);
    // Unfused: a launch a phase.
    EXPECT_EQ(conv_group::groupLaunches(*backward, false)->size(), 4u);
    // 3-D at stride 2 everywhere: one launch a depth phase.
    auto deep = dense(64, 64, {5, 12, 13}, {3, 3, 3}, {1, 1, 1}, true);
    deep.groups = 16;
    deep.stride = {2, 2, 2};
    const auto deepBackward = parsed(deep, ConvDirection::BACKWARD_DATA);
    ASSERT_TRUE(deepBackward.has_value());
    const auto deepLaunches = conv_group::groupLaunches(*deepBackward, true);
    ASSERT_TRUE(deepLaunches.has_value());
    EXPECT_EQ(deepLaunches->size(), 2u);
    EXPECT_EQ(deepLaunches->back().args.ood, 1);
    // Not for stride 1, forward, or a dx one row high (no second row phase).
    EXPECT_FALSE(conv_group::groupLaunches(*parsed(resnext(), ConvDirection::BACKWARD_DATA), true)
                     .has_value());
    EXPECT_FALSE(
        conv_group::groupLaunches(*parsed(strided, ConvDirection::FORWARD), true).has_value());
    auto flat = dense(64, 64, {1, 13}, {3, 3}, {1, 1}, true);
    flat.groups = 16;
    flat.stride = {2, 2};
    const auto flatBackward = parsed(flat, ConvDirection::BACKWARD_DATA);
    ASSERT_TRUE(flatBackward.has_value());
    EXPECT_FALSE(conv_group::groupLaunches(*flatBackward, true).has_value());
}

TEST(TestConvGroupProblem, DeclinesEveryUnservedGraph)
{
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        SCOPED_TRACE(direction == ConvDirection::FORWARD ? "forward" : "backward data");
        // Ungrouped, depthwise (1 channel a group), 128, 72 and 20 channels a group, groups of 3
        // (16 not a multiple), and a channel multiplier (k != c).
        auto ungrouped = resnext();
        ungrouped.groups = 1;
        EXPECT_FALSE(parsed(ungrouped, direction).has_value());
        auto depthwise = resnext();
        depthwise.groups = 64;
        EXPECT_FALSE(parsed(depthwise, direction).has_value());
        auto wider = dense(256, 256, {12, 13}, {3, 3}, {1, 1}, true);
        wider.groups = 2; // 128 channels a group
        EXPECT_FALSE(parsed(wider, direction).has_value());
        auto wide72 = dense(144, 144, {12, 13}, {3, 3}, {1, 1}, true);
        wide72.groups = 2; // 72 channels a group, past 64
        EXPECT_FALSE(parsed(wide72, direction).has_value());
        auto wide20 = dense(40, 40, {12, 13}, {3, 3}, {1, 1}, true);
        wide20.groups = 2; // 20 channels a group, not whole 16-byte vectors
        EXPECT_FALSE(parsed(wide20, direction).has_value());
        auto threes = dense(48, 48, {12, 13}, {3, 3}, {1, 1}, true);
        threes.groups = 16;
        EXPECT_FALSE(parsed(threes, direction).has_value());
        auto multiplier = dense(64, 128, {12, 13}, {3, 3}, {1, 1}, true);
        multiplier.groups = 16;
        EXPECT_FALSE(parsed(multiplier, direction).has_value());
        // Channels not whole 16-channel blocks.
        auto ragged = dense(40, 40, {12, 13}, {3, 3}, {1, 1}, true);
        ragged.groups = 10;
        EXPECT_FALSE(parsed(ragged, direction).has_value());
        // Dilation along any axis.
        auto dilatedD = dense(64, 64, {4, 6, 7}, {3, 3, 3}, {1, 1, 1}, true);
        dilatedD.groups = 16;
        dilatedD.dilation = {2, 1, 1};
        EXPECT_FALSE(parsed(dilatedD, direction).has_value());
        auto dilatedH = resnext();
        dilatedH.dilation = {2, 1};
        EXPECT_FALSE(parsed(dilatedH, direction).has_value());
        auto dilatedW = resnext();
        dilatedW.dilation = {1, 2};
        EXPECT_FALSE(parsed(dilatedW, direction).has_value());
        // Unequal strides.
        auto uneven = resnext();
        uneven.stride = {1, 2};
        EXPECT_FALSE(parsed(uneven, direction).has_value());
    }
    // Backward data past stride 2; forward serves it (no kernel ships it).
    auto strided = resnext();
    strided.stride = {3, 3};
    EXPECT_TRUE(parsed(strided, ConvDirection::FORWARD).has_value());
    EXPECT_FALSE(parsed(strided, ConvDirection::BACKWARD_DATA).has_value());
    // Backward data past depth stride 2; forward serves any depth stride.
    auto deep = dense(64, 64, {7, 6, 7}, {3, 3, 3}, {1, 1, 1}, true);
    deep.groups = 16;
    deep.stride = {3, 1, 1};
    EXPECT_TRUE(parsed(deep, ConvDirection::FORWARD).has_value());
    EXPECT_FALSE(parsed(deep, ConvDirection::BACKWARD_DATA).has_value());
}

TEST(TestConvGroupCatalog, KernelGatesFilterStrideChannelTileAtomAndLds)
{
    const auto device = [](const std::string& arch) {
        auto properties = testDeviceProperties();
        properties.gcnArchName = arch;
        properties.multiProcessorCount = 304;
        return properties;
    };
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        auto shape = resnext();
        shape.dataType = DataType::HALF;
        const GraphFixture fixture(buildGraph(specFor(shape, direction)),
                                   device("gfx942:sramecc+:xnack-"));
        const auto pack = packFor(direction);
        const auto bound = matchesGraph(pack, fixture.context());
        ASSERT_TRUE(bound.has_value());
        const auto& kernels = catalog(direction);
        const auto base = std::find_if(kernels.begin(), kernels.end(), [](const auto& kernel) {
            return kernel.getStringMetadata("atom") == "mfma16_f16"
                   && kernel.getIntMetadata("t") == 1 && kernel.getIntMetadata("stride") == 1
                   && kernel.getIntMetadata("cb") == 2;
        });
        ASSERT_NE(base, kernels.end());
        EXPECT_TRUE(matchesKernel(pack, fixture.context(), *base, *bound));
        const auto declines = [&](const char* field, const auto& value) {
            auto kernel = *base;
            kernel.metadata[field] = value;
            return !matchesKernel(pack, fixture.context(), kernel, *bound);
        };
        EXPECT_TRUE(declines("t", int64_t{3})); // a 2-D graph's filter is one tap deep
        EXPECT_TRUE(declines("t", int64_t{0}));
        EXPECT_TRUE(declines("r", int64_t{5}));
        EXPECT_TRUE(declines("s", int64_t{1}));
        EXPECT_TRUE(declines("stride", int64_t{2}));
        EXPECT_TRUE(declines("cb", int64_t{8})); // 64 channels are not 8 blocks of 16
        EXPECT_TRUE(declines("th", int64_t{0}));
        EXPECT_TRUE(declines("wp", int64_t{64})); // a halo past 64 KB
        EXPECT_TRUE(declines("atom", std::string("wmma11_f16"))); // not on gfx942
        EXPECT_TRUE(declines("atom", std::string("mfma16_bf16"))); // another element type
        EXPECT_TRUE(declines("atom", std::string("fma"))); // no FMA form
        EXPECT_TRUE(declines("dtype", std::string("FLOAT")));
        // Channels-first in place: not for channels-last activations, and only 0 or 1.
        EXPECT_TRUE(declines("cf", int64_t{1}));
        EXPECT_TRUE(declines("cf", int64_t{2}));
        // Phase-fused: only backward data at stride 2 (this graph is stride 1); 0 or 1.
        EXPECT_TRUE(declines("pf", int64_t{1}));
        EXPECT_TRUE(declines("pf", int64_t{2}));
        // Wide-group knobs on a small-group graph.
        EXPECT_TRUE(declines("nbi", int64_t{2}));
        EXPECT_TRUE(declines("nbo", int64_t{1}));
        // Backward-weights knobs.
        EXPECT_TRUE(declines("tw", int64_t{16}));
        EXPECT_TRUE(declines("bpu", int64_t{4}));
        // Packed channels-first activations: the cf twin is admitted, with a smaller workspace
        // (nothing staged), and the staged config still is.
        auto first = resnext();
        first.channelsLast = false;
        first.dataType = DataType::HALF;
        const GraphFixture firstFixture(buildGraph(specFor(first, direction)),
                                        device("gfx942:sramecc+:xnack-"));
        const auto firstBound = matchesGraph(pack, firstFixture.context());
        ASSERT_TRUE(firstBound.has_value());
        auto inPlace = *base;
        inPlace.metadata["cf"] = int64_t{1};
        EXPECT_TRUE(matchesKernel(pack, firstFixture.context(), inPlace, *firstBound));
        EXPECT_TRUE(matchesKernel(pack, firstFixture.context(), *base, *firstBound));
        const auto& handler = dispatchHandler(pack);
        EXPECT_LT(handler.workspaceBytes(firstFixture.context(), *firstBound, inPlace),
                  handler.workspaceBytes(firstFixture.context(), *firstBound, *base));
    }
}

TEST(TestConvGroupCatalog, FmaKernelGatesGroupWidthChunkAndTile)
{
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        auto shape = resnext(); // 4 channels a group
        shape.dataType = DataType::FLOAT;
        auto properties = testDeviceProperties();
        properties.gcnArchName = "gfx1201";
        const GraphFixture fixture(buildGraph(specFor(shape, direction)), properties);
        const auto pack = packFor(direction);
        const auto bound = matchesGraph(pack, fixture.context());
        ASSERT_TRUE(bound.has_value());
        const auto& kernels = catalog(direction);
        const auto base = std::find_if(kernels.begin(), kernels.end(), [](const auto& kernel) {
            return kernel.getStringMetadata("atom") == "fma" && kernel.getIntMetadata("cg") == 4
                   && kernel.getIntMetadata("t") == 1 && kernel.getIntMetadata("stride") == 1
                   && kernel.getIntMetadata("cf") == 0 && kernel.getIntMetadata("th") == 4
                   && kernel.getIntMetadata("px") == 2;
        });
        ASSERT_NE(base, kernels.end());
        EXPECT_TRUE(matchesKernel(pack, fixture.context(), *base, *bound));
        const auto declines = [&](const char* field, const auto& value) {
            auto kernel = *base;
            kernel.metadata[field] = value;
            return !matchesKernel(pack, fixture.context(), kernel, *bound);
        };
        EXPECT_TRUE(declines("cg", int64_t{8})); // compiled for another group width
        EXPECT_TRUE(declines("cg", int64_t{3}));
        EXPECT_TRUE(declines("oc", int64_t{3}));
        EXPECT_TRUE(declines("oc", int64_t{2})); // a chunk's pixels not whole waves
        EXPECT_TRUE(declines("px", int64_t{0}));
        EXPECT_TRUE(declines("th", int64_t{3})); // not whole rows of the chunk's pixels
        EXPECT_TRUE(declines("wp", int64_t{1}));
        EXPECT_TRUE(declines("px", int64_t{64})); // a halo past 64 KB
        EXPECT_TRUE(declines("dtype", std::string("HALF")));
        EXPECT_TRUE(declines("pf", int64_t{1})); // no phase-fused FMA form
        // A matrix-core config carries no FMA knobs.
        const auto mma = std::find_if(kernels.begin(), kernels.end(), [](const auto& kernel) {
            return kernel.getStringMetadata("atom") == "mfma16_f32";
        });
        ASSERT_NE(mma, kernels.end());
        auto withCg = *mma;
        withCg.metadata["cg"] = int64_t{4};
        EXPECT_FALSE(matchesKernel(pack, fixture.context(), withCg, *bound));
    }
}

TEST(TestConvGroupCatalog, WideKernelGatesBlocksTileAndForm)
{
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        auto shape = dense(112, 112, {12, 13}, {3, 3}, {1, 1}, true);
        shape.groups = 2; // 56 channels a group: 4 blocks
        shape.dataType = DataType::HALF;
        auto properties = testDeviceProperties();
        properties.gcnArchName = "gfx942:sramecc+:xnack-";
        const GraphFixture fixture(buildGraph(specFor(shape, direction)), properties);
        const auto pack = packFor(direction);
        const auto bound = matchesGraph(pack, fixture.context());
        ASSERT_TRUE(bound.has_value());
        const auto& kernels = catalog(direction);
        const auto base = std::find_if(kernels.begin(), kernels.end(), [](const auto& kernel) {
            return kernel.getStringMetadata("atom") == "mfma16_f16"
                   && kernel.getIntMetadata("nbi") == 4 && kernel.getIntMetadata("nbo") == 4
                   && kernel.getIntMetadata("t") == 1 && kernel.getIntMetadata("stride") == 1
                   && kernel.getIntMetadata("cf") == 0 && kernel.getIntMetadata("th") == 4
                   && kernel.getIntMetadata("wp") == 2;
        });
        ASSERT_NE(base, kernels.end());
        EXPECT_TRUE(matchesKernel(pack, fixture.context(), *base, *bound));
        const auto declines = [&](const char* field, const auto& value) {
            auto kernel = *base;
            kernel.metadata[field] = value;
            return !matchesKernel(pack, fixture.context(), kernel, *bound);
        };
        EXPECT_TRUE(declines("nbi", int64_t{2})); // the groups span 4 blocks
        EXPECT_TRUE(declines("nbi", int64_t{0})); // a small-group config
        EXPECT_TRUE(declines("nbo", int64_t{3})); // not whole output tiles
        EXPECT_TRUE(declines("nbo", int64_t{0}));
        EXPECT_TRUE(declines("th", int64_t{2})); // not whole rows a wave
        EXPECT_TRUE(declines("wp", int64_t{0}));
        EXPECT_TRUE(declines("wp", int64_t{64})); // a halo past 64 KB
        EXPECT_TRUE(declines("cb", int64_t{1}));
        EXPECT_TRUE(declines("pf", int64_t{1}));
        EXPECT_TRUE(declines("cf", int64_t{1})); // channels-last graph
        EXPECT_TRUE(declines("cg", int64_t{56}));
        EXPECT_TRUE(declines("atom", std::string("fma")));
        EXPECT_TRUE(declines("atom", std::string("wmma11_f16")));
        EXPECT_TRUE(declines("dtype", std::string("FLOAT")));
        // A small-group config declines a wide graph.
        const auto small = std::find_if(kernels.begin(), kernels.end(), [](const auto& kernel) {
            return kernel.getStringMetadata("atom") == "mfma16_f16"
                   && kernel.getIntMetadata("nbi") == 0 && kernel.getIntMetadata("t") == 1
                   && kernel.getIntMetadata("stride") == 1;
        });
        ASSERT_NE(small, kernels.end());
        EXPECT_FALSE(matchesKernel(pack, fixture.context(), *small, *bound));
    }
}

TEST(TestConvGroupCatalog, EveryConfigFitsItsLdsAndKnobsTellConfigsApart)
{
    static constexpr std::array<const char*, 18> KNOBS{"atom",
                                                       "t",
                                                       "r",
                                                       "s",
                                                       "stride",
                                                       "th",
                                                       "wp",
                                                       "cb",
                                                       "cf",
                                                       "cg",
                                                       "oc",
                                                       "px",
                                                       "pf",
                                                       "nbi",
                                                       "nbo",
                                                       "tw",
                                                       "bpu",
                                                       "math"};
    for(const auto direction :
        {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA, ConvDirection::BACKWARD_WEIGHTS})
    {
        std::vector<std::string> tuples;
        for(const auto& kernel : catalog(direction))
        {
            SCOPED_TRACE(kernel.name);
            // Three-bf16 configs (math 1) stage bf16.
            const int bytes
                = kernel.getStringMetadata("dtype") == "FLOAT" && kernel.getIntMetadata("math") == 0
                      ? 4
                      : 2;
            std::string tuple = kernel.getStringMetadata("dtype");
            for(const char* knob : KNOBS)
            {
                const std::string name(knob);
                tuple += "|"
                         + (name == "atom" ? kernel.getStringMetadata(name)
                                           : std::to_string(kernel.getIntMetadata(name)));
            }
            EXPECT_EQ(std::count(tuples.begin(), tuples.end(), tuple), 0);
            tuples.push_back(tuple);
            if(direction == ConvDirection::BACKWARD_WEIGHTS)
            {
                EXPECT_EQ(kernel.getStringMetadata("direction"), "WRW");
                EXPECT_LE(conv_group::groupWrwLdsBytes(
                              bytes,
                              static_cast<int>(kernel.getIntMetadata("r")),
                              static_cast<int>(kernel.getIntMetadata("s")),
                              static_cast<int>(kernel.getIntMetadata("stride")),
                              static_cast<int>(kernel.getIntMetadata("th")),
                              static_cast<int>(kernel.getIntMetadata("tw")),
                              static_cast<int>(kernel.getIntMetadata(
                                  kernel.getIntMetadata("nbi") != 0 ? "nbo" : "cb")),
                              static_cast<int>(kernel.getIntMetadata("nbi"))),
                          64 * 1024);
                if(kernel.getIntMetadata("nbi") != 0)
                {
                    EXPECT_EQ(kernel.getIntMetadata("cb"), 0);
                    EXPECT_EQ(kernel.getIntMetadata("nbi") % kernel.getIntMetadata("nbo"), 0);
                }
                else
                {
                    EXPECT_EQ(kernel.getIntMetadata("nbo"), 0);
                }
                for(const char* unused : {"wp", "oc", "pf"})
                {
                    EXPECT_EQ(kernel.getIntMetadata(unused), 0) << unused;
                }
                // FMA configs: compiled for a group width, row segments dividing the columns,
                // fp32; matrix-core configs carry neither knob.
                if(kernel.getStringMetadata("atom") == "fma")
                {
                    EXPECT_EQ(kernel.getStringMetadata("dtype"), "FLOAT");
                    EXPECT_GE(kernel.getIntMetadata("cg"), 2);
                    EXPECT_GE(kernel.getIntMetadata("px"), 1);
                    EXPECT_EQ(kernel.getIntMetadata("tw") % kernel.getIntMetadata("px"), 0);
                }
                else
                {
                    EXPECT_EQ(kernel.getIntMetadata("cg"), 0);
                    EXPECT_EQ(kernel.getIntMetadata("px"), 0);
                }
                EXPECT_GE(kernel.getIntMetadata("bpu"), 1);
                EXPECT_LE(kernel.getIntMetadata("bpu"), 16);
                continue;
            }
            EXPECT_EQ(kernel.getIntMetadata("tw"), 0);
            EXPECT_EQ(kernel.getIntMetadata("bpu"), 0);
            // Backward data runs stride phases: ceil(R / stride) taps at unit stride.
            const int stride = static_cast<int>(kernel.getIntMetadata("stride"));
            const bool forward = direction == ConvDirection::FORWARD;
            const auto taps = [&](const char* field) {
                const int extent = static_cast<int>(kernel.getIntMetadata(field));
                return forward ? extent : (extent + stride - 1) / stride;
            };
            // Output columns a workgroup: 16 wp (matrix core), or the FMA kernel's 256 threads
            // in chunks of oc channels, th rows of px-column threads.
            const int th = static_cast<int>(kernel.getIntMetadata("th"));
            // Wide-group configs stage nbi blocks of a group (no channel blocks).
            const int nbi = static_cast<int>(kernel.getIntMetadata("nbi"));
            const int cb = nbi != 0 ? nbi : static_cast<int>(kernel.getIntMetadata("cb"));
            if(nbi != 0)
            {
                EXPECT_EQ(kernel.getIntMetadata("cb"), 0);
                EXPECT_EQ(nbi % kernel.getIntMetadata("nbo"), 0);
            }
            const bool fma = kernel.getStringMetadata("atom") == "fma";
            const int tw = fma ? static_cast<int>(256 * kernel.getIntMetadata("oc") / (16 * cb) / th
                                                  * kernel.getIntMetadata("px"))
                               : static_cast<int>(16 * kernel.getIntMetadata("wp"));
            if(kernel.getIntMetadata("pf") != 0)
            {
                // Phase-fused: backward data at stride 2 only, its output tile in LDS too.
                EXPECT_FALSE(forward);
                EXPECT_EQ(stride, 2);
                EXPECT_LE(
                    conv_group::groupPhasedLdsBytes(bytes,
                                                    taps("r"),
                                                    taps("s"),
                                                    th,
                                                    tw,
                                                    cb,
                                                    kernel.getIntMetadata("cf") != 0,
                                                    static_cast<int>(kernel.getIntMetadata("nbo"))),
                    64 * 1024);
            }
            else
            {
                EXPECT_LE(conv_group::groupLdsBytes(
                              bytes, taps("r"), taps("s"), forward ? stride : 1, th, tw, cb),
                          64 * 1024);
            }
            if(fma)
            {
                EXPECT_EQ(kernel.getStringMetadata("dtype"), "FLOAT");
                EXPECT_EQ(kernel.getIntMetadata("wp"), 0);
            }
            else
            {
                EXPECT_EQ(kernel.getIntMetadata("cg"), 0);
            }
            EXPECT_LE(kernel.getIntMetadata("stride"), 2);
        }
    }
}

// ---- Backward weights: graph gates and kernel gates ----

std::optional<conv_group::GroupWrwProblem> parsedWrw(const ConvGraphSpec& spec)
{
    const GraphFixture fixture(buildGraph(spec));
    return conv_group::parseGroupWrwProblem(fixture.context());
}

std::optional<conv_group::GroupWrwProblem> parsedWrw(ConvShape shape,
                                                     DataType dataType = DataType::HALF)
{
    shape.dataType = dataType;
    return parsedWrw(specFor(shape, ConvDirection::BACKWARD_WEIGHTS));
}

TEST(TestConvGroupWrwProblem, ServesSmallGroupsBothLayoutsAnyDepthStride)
{
    const auto last = parsedWrw(resnext());
    ASSERT_TRUE(last.has_value());
    EXPECT_EQ(last->cg, 4);
    EXPECT_EQ(last->filterT, 1);
    EXPECT_EQ(last->filterR, 3);
    EXPECT_EQ(last->stride, 1);
    EXPECT_TRUE(last->wrw.channelsLast);
    auto first = resnext();
    first.channelsLast = false;
    const auto firstParsed = parsedWrw(first);
    ASSERT_TRUE(firstParsed.has_value());
    EXPECT_TRUE(firstParsed->wrw.channelsFirst);
    auto strided = resnext();
    strided.stride = {2, 2};
    ASSERT_TRUE(parsedWrw(strided).has_value());
    EXPECT_EQ(parsedWrw(strided)->stride, 2);
    // 3-D: any depth stride (the kernel's runtime), 3 or 1 taps deep.
    auto deep = dense(64, 64, {7, 6, 7}, {3, 3, 3}, {1, 1, 1}, true);
    deep.groups = 16;
    deep.stride = {3, 1, 1};
    const auto deepParsed = parsedWrw(deep);
    ASSERT_TRUE(deepParsed.has_value());
    EXPECT_EQ(deepParsed->filterT, 3);
    EXPECT_EQ(deepParsed->stride, 1);
    // Groups of 2 and 16.
    auto g2 = dense(32, 32, {5, 7}, {3, 3}, {1, 1}, true);
    g2.groups = 16;
    EXPECT_EQ(parsedWrw(g2)->cg, 2);
    auto g16 = dense(64, 64, {5, 7}, {3, 3}, {1, 1}, false);
    g16.groups = 4;
    EXPECT_EQ(parsedWrw(g16)->cg, 16);
    EXPECT_EQ(parsedWrw(g16)->nbi, 0);
    // Wide groups: 24 to 64 channels (a multiple of 8), padded to whole 16-channel blocks.
    auto w32 = dense(64, 64, {5, 7}, {3, 3}, {1, 1}, false);
    w32.groups = 2;
    EXPECT_EQ(parsedWrw(w32)->cg, 32);
    EXPECT_EQ(parsedWrw(w32)->nbi, 2);
    auto w24 = dense(48, 48, {5, 7}, {3, 3}, {1, 1}, true);
    w24.groups = 2;
    EXPECT_EQ(parsedWrw(w24)->nbi, 2);
}

TEST(TestConvGroupWrwProblem, DeclinesEveryUnservedGraph)
{
    auto ungrouped = resnext();
    ungrouped.groups = 1;
    EXPECT_FALSE(parsedWrw(ungrouped).has_value());
    auto ungrouped16 = dense(16, 16, {12, 13}, {3, 3}, {1, 1}, true);
    EXPECT_FALSE(parsedWrw(ungrouped16).has_value()); // one group of 16: K3's dense case
    auto wide28 = dense(56, 56, {12, 13}, {3, 3}, {1, 1}, true);
    wide28.groups = 2; // 28 channels a group, not a multiple of 8
    EXPECT_FALSE(parsedWrw(wide28).has_value());
    auto depthwise = resnext();
    depthwise.groups = 64;
    EXPECT_FALSE(parsedWrw(depthwise).has_value());
    auto wider = dense(256, 256, {12, 13}, {3, 3}, {1, 1}, true);
    wider.groups = 2; // 128 channels a group
    EXPECT_FALSE(parsedWrw(wider).has_value());
    auto wide40 = dense(80, 80, {12, 13}, {3, 3}, {1, 1}, true);
    wide40.groups = 2; // 40 channels a group, past 32 (K3's)
    EXPECT_FALSE(parsedWrw(wide40).has_value());
    auto wide20 = dense(40, 40, {12, 13}, {3, 3}, {1, 1}, true);
    wide20.groups = 2; // 20 channels a group, not whole 16-byte vectors
    EXPECT_FALSE(parsedWrw(wide20).has_value());
    auto threes = dense(48, 48, {12, 13}, {3, 3}, {1, 1}, true);
    threes.groups = 16; // 3 channels a group
    EXPECT_FALSE(parsedWrw(threes).has_value());
    auto multiplier = dense(64, 128, {12, 13}, {3, 3}, {1, 1}, true);
    multiplier.groups = 16;
    EXPECT_FALSE(parsedWrw(multiplier).has_value());
    auto ragged = dense(40, 40, {12, 13}, {3, 3}, {1, 1}, true);
    ragged.groups = 10; // 4 channels a group, channels not whole 16-channel blocks
    EXPECT_FALSE(parsedWrw(ragged).has_value());
    auto dilatedD = dense(64, 64, {4, 6, 7}, {3, 3, 3}, {1, 1, 1}, true);
    dilatedD.groups = 16;
    dilatedD.dilation = {2, 1, 1};
    EXPECT_FALSE(parsedWrw(dilatedD).has_value());
    auto dilatedH = resnext();
    dilatedH.dilation = {2, 1};
    EXPECT_FALSE(parsedWrw(dilatedH).has_value());
    auto dilatedW = resnext();
    dilatedW.dilation = {1, 2};
    EXPECT_FALSE(parsedWrw(dilatedW).has_value());
    auto uneven = resnext();
    uneven.stride = {1, 2};
    EXPECT_FALSE(parsedWrw(uneven).has_value());
    // x channels-last, dy channels-first: the kernel reads both in one layout.
    auto mixed = resnext();
    mixed.dataType = DataType::HALF;
    auto spec = specFor(mixed, ConvDirection::BACKWARD_WEIGHTS);
    auto firstShape = mixed;
    firstShape.channelsLast = false;
    spec.y.strides = specFor(firstShape, ConvDirection::BACKWARD_WEIGHTS).y.strides;
    EXPECT_FALSE(parsedWrw(spec).has_value());
    // Not a backward-weights graph.
    auto forward = resnext();
    forward.dataType = DataType::HALF;
    const GraphFixture fixture(buildGraph(specFor(forward, ConvDirection::FORWARD)));
    EXPECT_FALSE(conv_group::parseGroupWrwProblem(fixture.context()).has_value());
}

TEST(TestConvGroupCatalog, WrwKernelGatesEveryField)
{
    const auto device = [](const std::string& arch, int units) {
        auto properties = testDeviceProperties();
        properties.gcnArchName = arch;
        properties.multiProcessorCount = units;
        return properties;
    };
    auto shape = resnext(); // 64 channels, groups of 4, channels-last
    shape.dataType = DataType::HALF;
    const auto spec = specFor(shape, ConvDirection::BACKWARD_WEIGHTS);
    const GraphFixture fixture(buildGraph(spec), device("gfx942:sramecc+:xnack-", 304));
    const auto pack = packFor(ConvDirection::BACKWARD_WEIGHTS);
    const auto bound = matchesGraph(pack, fixture.context());
    ASSERT_TRUE(bound.has_value());
    const auto& kernels = catalog(ConvDirection::BACKWARD_WEIGHTS);
    const auto base = std::find_if(kernels.begin(), kernels.end(), [](const auto& kernel) {
        return kernel.getStringMetadata("dtype") == "HALF"
               && kernel.getStringMetadata("atom") == "mfma16_f16"
               && kernel.getIntMetadata("t") == 1 && kernel.getIntMetadata("stride") == 1
               && kernel.getIntMetadata("cf") == 0 && kernel.getIntMetadata("cb") == 1;
    });
    ASSERT_NE(base, kernels.end());
    EXPECT_TRUE(matchesKernel(pack, fixture.context(), *base, *bound));
    const auto& handler = dispatchHandler(pack);
    EXPECT_GT(handler.workspaceBytes(fixture.context(), *bound, *base), 0u);
    const auto declines = [&](const char* field, const auto& value) {
        auto kernel = *base;
        kernel.metadata[field] = value;
        return !matchesKernel(pack, fixture.context(), kernel, *bound);
    };
    EXPECT_TRUE(declines("direction", std::string("FWD"))); // a forward config carries no tw
    EXPECT_TRUE(declines("direction", std::string("DATA")));
    EXPECT_TRUE(declines("dtype", std::string("FLOAT")));
    EXPECT_TRUE(declines("atom", std::string("wmma11_f16"))); // not on gfx942
    EXPECT_TRUE(declines("atom", std::string("mfma16_bf16"))); // another element type
    EXPECT_TRUE(declines("atom", std::string("mfma32_f16"))); // not a 16x16 atom
    EXPECT_TRUE(declines("atom", std::string("mfma16_xf32"))); // TF32
    EXPECT_TRUE(declines("atom", std::string("fma"))); // an FMA config names its group width
    EXPECT_TRUE(declines("t", int64_t{3})); // a 2-D graph's filter is one tap deep
    EXPECT_TRUE(declines("r", int64_t{5}));
    EXPECT_TRUE(declines("s", int64_t{1}));
    EXPECT_TRUE(declines("stride", int64_t{2}));
    EXPECT_TRUE(declines("th", int64_t{0}));
    EXPECT_TRUE(declines("tw", int64_t{0}));
    EXPECT_TRUE(declines("tw", int64_t{4000})); // a tile past 64 KB of LDS
    EXPECT_TRUE(declines("cb", int64_t{3}));
    EXPECT_TRUE(declines("cb", int64_t{8})); // 64 channels are not 8 blocks of 16
    EXPECT_TRUE(declines("cf", int64_t{1})); // channels-last graph
    EXPECT_TRUE(declines("cf", int64_t{2}));
    EXPECT_TRUE(declines("wp", int64_t{1}));
    EXPECT_TRUE(declines("cg", int64_t{4}));
    EXPECT_TRUE(declines("oc", int64_t{4}));
    EXPECT_TRUE(declines("px", int64_t{1}));
    EXPECT_TRUE(declines("pf", int64_t{1}));
    EXPECT_TRUE(declines("nbi", int64_t{2}));
    EXPECT_TRUE(declines("nbo", int64_t{1}));
    EXPECT_TRUE(declines("bpu", int64_t{0}));
    EXPECT_TRUE(declines("bpu", int64_t{17}));
    // A device that reports no dispatch units: no split plan.
    const GraphFixture unknown(buildGraph(spec), device("gfx942:sramecc+:xnack-", 0));
    EXPECT_FALSE(matchesKernel(pack, unknown.context(), *base, *bound));
    // The cf twin serves the channels-first graph, not this one; the channels-last config not
    // that one.
    auto first = resnext();
    first.channelsLast = false;
    first.dataType = DataType::HALF;
    const GraphFixture firstFixture(buildGraph(specFor(first, ConvDirection::BACKWARD_WEIGHTS)),
                                    device("gfx942:sramecc+:xnack-", 304));
    const auto firstBound = matchesGraph(pack, firstFixture.context());
    ASSERT_TRUE(firstBound.has_value());
    auto inPlace = *base;
    inPlace.metadata["cf"] = int64_t{1};
    EXPECT_TRUE(matchesKernel(pack, firstFixture.context(), inPlace, *firstBound));
    EXPECT_FALSE(matchesKernel(pack, firstFixture.context(), *base, *firstBound));
    auto cf2 = *base;
    cf2.metadata["cf"] = int64_t{2};
    EXPECT_FALSE(matchesKernel(pack, firstFixture.context(), cf2, *firstBound));
    // A tile extent past 32 bits.
    EXPECT_TRUE(declines("th", (int64_t{1} << 32) + 1));
    // cb 3: 96 channels are 2 tiles of 3 blocks, but a wave takes whole blocks or a block whole
    // waves.
    auto ninetySix = dense(96, 96, {12, 13}, {3, 3}, {1, 1}, true);
    ninetySix.groups = 24;
    ninetySix.dataType = DataType::HALF;
    const GraphFixture wideFixture(buildGraph(specFor(ninetySix, ConvDirection::BACKWARD_WEIGHTS)),
                                   device("gfx942:sramecc+:xnack-", 304));
    const auto wideBound = matchesGraph(pack, wideFixture.context());
    ASSERT_TRUE(wideBound.has_value());
    EXPECT_TRUE(matchesKernel(pack, wideFixture.context(), *base, *wideBound));
    auto cb3 = *base;
    cb3.metadata["cb"] = int64_t{3};
    EXPECT_FALSE(matchesKernel(pack, wideFixture.context(), cb3, *wideBound));
    // A forward graph's configs and this direction's are kept apart.
    EXPECT_FALSE(
        matchesGraph(pack,
                     GraphFixture(buildGraph(specFor(shape, ConvDirection::FORWARD))).context())
            .has_value());
}

TEST(TestConvGroupCatalog, WrwFmaKernelGatesGroupWidthUnitsAndSegments)
{
    auto shape = resnext(); // 4 channels a group
    shape.dataType = DataType::FLOAT;
    auto properties = testDeviceProperties();
    properties.gcnArchName = "gfx1201";
    properties.multiProcessorCount = 32;
    const GraphFixture fixture(buildGraph(specFor(shape, ConvDirection::BACKWARD_WEIGHTS)),
                               properties);
    const auto pack = packFor(ConvDirection::BACKWARD_WEIGHTS);
    const auto bound = matchesGraph(pack, fixture.context());
    ASSERT_TRUE(bound.has_value());
    const auto& kernels = catalog(ConvDirection::BACKWARD_WEIGHTS);
    const auto base = std::find_if(kernels.begin(), kernels.end(), [](const auto& kernel) {
        return kernel.getStringMetadata("atom") == "fma" && kernel.getIntMetadata("cg") == 4
               && kernel.getIntMetadata("t") == 1 && kernel.getIntMetadata("stride") == 1
               && kernel.getIntMetadata("cf") == 0 && kernel.getIntMetadata("tw") == 16
               && kernel.getIntMetadata("px") == 4;
    });
    ASSERT_NE(base, kernels.end());
    EXPECT_TRUE(matchesKernel(pack, fixture.context(), *base, *bound));
    const auto declines = [&](const char* field, const auto& value) {
        auto kernel = *base;
        kernel.metadata[field] = value;
        return !matchesKernel(pack, fixture.context(), kernel, *bound);
    };
    EXPECT_TRUE(declines("cg", int64_t{8})); // compiled for another group width
    EXPECT_TRUE(declines("cg", int64_t{3}));
    EXPECT_TRUE(declines("cg", int64_t{0}));
    EXPECT_TRUE(declines("px", int64_t{0}));
    EXPECT_TRUE(declines("px", int64_t{3})); // segments not dividing the tile's columns
    EXPECT_TRUE(declines("cb", int64_t{0}));
    // 16 channels a group at cb 2: whole channel tiles and a halo within LDS, but 512 (k, c)
    // units past 256 threads.
    auto g16 = dense(64, 64, {12, 13}, {3, 3}, {1, 1}, true);
    g16.groups = 4;
    g16.dataType = DataType::FLOAT;
    const GraphFixture g16Fixture(buildGraph(specFor(g16, ConvDirection::BACKWARD_WEIGHTS)),
                                  properties);
    const auto g16Bound = matchesGraph(pack, g16Fixture.context());
    ASSERT_TRUE(g16Bound.has_value());
    auto cg16 = *base;
    cg16.metadata["cg"] = int64_t{16};
    EXPECT_TRUE(matchesKernel(pack, g16Fixture.context(), cg16, *g16Bound));
    cg16.metadata["cb"] = int64_t{2};
    EXPECT_FALSE(matchesKernel(pack, g16Fixture.context(), cg16, *g16Bound));
    EXPECT_TRUE(declines("wp", int64_t{1}));
    EXPECT_TRUE(declines("oc", int64_t{4}));
    EXPECT_TRUE(declines("dtype", std::string("HALF")));
    EXPECT_TRUE(declines("tw", int64_t{4000})); // a tile past 64 KB of LDS
    // A matrix-core config carries no FMA knobs.
    const auto mma = std::find_if(kernels.begin(), kernels.end(), [](const auto& kernel) {
        return kernel.getStringMetadata("atom") == "mfma16_f32";
    });
    ASSERT_NE(mma, kernels.end());
    auto withCg = *mma;
    withCg.metadata["cg"] = int64_t{4};
    auto withPx = *mma;
    withPx.metadata["px"] = int64_t{4};
    properties.gcnArchName = "gfx942:sramecc+:xnack-";
    const GraphFixture cdna(buildGraph(specFor(shape, ConvDirection::BACKWARD_WEIGHTS)),
                            properties);
    const auto cdnaBound = matchesGraph(pack, cdna.context());
    ASSERT_TRUE(cdnaBound.has_value());
    EXPECT_FALSE(matchesKernel(pack, cdna.context(), withCg, *cdnaBound));
    EXPECT_FALSE(matchesKernel(pack, cdna.context(), withPx, *cdnaBound));
}

TEST(TestConvGroupCatalog, WrwWideKernelGatesBlocksAndForm)
{
    auto shape = dense(64, 64, {12, 13}, {3, 3}, {1, 1}, true);
    shape.groups = 2; // 32 channels a group: 2 blocks
    shape.dataType = DataType::HALF;
    auto properties = testDeviceProperties();
    properties.gcnArchName = "gfx942:sramecc+:xnack-";
    properties.multiProcessorCount = 304;
    const GraphFixture fixture(buildGraph(specFor(shape, ConvDirection::BACKWARD_WEIGHTS)),
                               properties);
    const auto pack = packFor(ConvDirection::BACKWARD_WEIGHTS);
    const auto bound = matchesGraph(pack, fixture.context());
    ASSERT_TRUE(bound.has_value());
    const auto& kernels = catalog(ConvDirection::BACKWARD_WEIGHTS);
    const auto base = std::find_if(kernels.begin(), kernels.end(), [](const auto& kernel) {
        return kernel.getStringMetadata("dtype") == "HALF"
               && kernel.getStringMetadata("atom") == "mfma16_f16"
               && kernel.getIntMetadata("nbi") == 2 && kernel.getIntMetadata("nbo") == 1
               && kernel.getIntMetadata("t") == 1 && kernel.getIntMetadata("stride") == 1
               && kernel.getIntMetadata("cf") == 0;
    });
    ASSERT_NE(base, kernels.end());
    EXPECT_TRUE(matchesKernel(pack, fixture.context(), *base, *bound));
    const auto declines = [&](const char* field, const auto& value) {
        auto kernel = *base;
        kernel.metadata[field] = value;
        return !matchesKernel(pack, fixture.context(), kernel, *bound);
    };
    EXPECT_TRUE(declines("nbi", int64_t{4})); // the groups span 2 blocks
    EXPECT_TRUE(declines("nbi", int64_t{0})); // a small-group config
    EXPECT_TRUE(declines("nbo", int64_t{3})); // not whole dy tiles of a group
    EXPECT_TRUE(declines("nbo", int64_t{0}));
    EXPECT_TRUE(declines("cb", int64_t{1}));
    EXPECT_TRUE(declines("cg", int64_t{32}));
    EXPECT_TRUE(declines("px", int64_t{4}));
    EXPECT_TRUE(declines("atom", std::string("fma")));
    EXPECT_TRUE(declines("atom", std::string("wmma12_f16")));
    EXPECT_TRUE(declines("atom", std::string("mfma32_f16"))); // not a 16x16 atom
    EXPECT_TRUE(declines("cf", int64_t{1}));
    // A small-group config declines the wide graph; the wide config a small graph.
    const auto small = std::find_if(kernels.begin(), kernels.end(), [](const auto& kernel) {
        return kernel.getStringMetadata("dtype") == "HALF"
               && kernel.getStringMetadata("atom") == "mfma16_f16"
               && kernel.getIntMetadata("nbi") == 0 && kernel.getIntMetadata("t") == 1
               && kernel.getIntMetadata("stride") == 1 && kernel.getIntMetadata("cf") == 0;
    });
    ASSERT_NE(small, kernels.end());
    EXPECT_FALSE(matchesKernel(pack, fixture.context(), *small, *bound));
    auto resnextShape = resnext();
    resnextShape.dataType = DataType::HALF;
    const GraphFixture smallFixture(
        buildGraph(specFor(resnextShape, ConvDirection::BACKWARD_WEIGHTS)), properties);
    const auto smallBound = matchesGraph(pack, smallFixture.context());
    ASSERT_TRUE(smallBound.has_value());
    EXPECT_TRUE(matchesKernel(pack, smallFixture.context(), *small, *smallBound));
    EXPECT_FALSE(matchesKernel(pack, smallFixture.context(), *base, *smallBound));
}

TEST(TestConvGroupWrwLaunch, SplitsFillTheDeviceAndPartialsSizeTheWorkspace)
{
    auto shape = dense(128, 128, {56, 56}, {3, 3}, {1, 1}, false);
    shape.groups = 32;
    shape.n = 16;
    const auto problem = parsedWrw(shape);
    ASSERT_TRUE(problem.has_value());
    auto properties = testDeviceProperties();
    properties.gcnArchName = "gfx950";
    properties.multiProcessorCount = 256;
    const auto launch = conv_group::groupWrwLaunch(*problem, 8, 16, 1, 4, properties);
    ASSERT_TRUE(launch.has_value());
    const auto& args = launch->args;
    EXPECT_EQ(args.tilesW, 4);
    EXPECT_EQ(args.tilesH, 7);
    EXPECT_EQ(args.tiles, 16LL * 7 * 4);
    EXPECT_EQ(args.partialElems, 128LL * 4 * 9);
    // 8 channel tiles: splits towards 4 workgroups a unit, at least 4 tiles a split.
    EXPECT_GE(args.splits, 1);
    EXPECT_LE(static_cast<long long>(args.splits) * 4, args.tiles);
    EXPECT_EQ(launch->grid, 8u * static_cast<unsigned>(args.splits));
    EXPECT_EQ(launch->reduce.slots, args.splits);
    EXPECT_EQ(launch->workspaceBytes % 256, 0u);
    EXPECT_GE(launch->workspaceBytes,
              static_cast<size_t>(args.splits) * static_cast<size_t>(args.partialElems) * 4);
    // dw packed K C T R S: the reduce's strides.
    EXPECT_EQ(launch->reduce.sK, 4 * 9);
    EXPECT_EQ(launch->reduce.sC, 9);
    EXPECT_EQ(launch->reduce.sS, 1);
    // No units, no plan; channels not a multiple of the tile's.
    properties.multiProcessorCount = 0;
    EXPECT_FALSE(conv_group::groupWrwLaunch(*problem, 8, 16, 1, 4, properties).has_value());
    properties.multiProcessorCount = 256;
    EXPECT_FALSE(conv_group::groupWrwLaunch(*problem, 8, 16, 16, 4, properties).has_value());
}

TEST(TestConvGroupComputeType, ServesCudnnsComputeTypesPerElementType)
{
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        expectCudnnComputeTypes(
            resnext(),
            {DataType::FLOAT, DataType::HALF, DataType::BFLOAT16},
            [direction](const ConvShape& s) {
                const GraphFixture fixture(buildGraph(specFor(s, direction)));
                return conv_group::parseGroupProblem(fixture.context(), direction).has_value();
            });
    }
    expectCudnnComputeTypes(
        resnext(), {DataType::FLOAT, DataType::HALF, DataType::BFLOAT16}, [](const ConvShape& s) {
            return parsedWrw(specFor(s, ConvDirection::BACKWARD_WEIGHTS)).has_value();
        });
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_CONV
