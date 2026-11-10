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
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestGpuConvGroup.cpp
 * @brief The small-group convolution kernels (CONV_GROUP_FWD, CONV_GROUP_BWD) on the device,
 *        through the engines' own graph match, kernel match, prepare and launch, against
 *        hipDNN's CPU reference convolution; and their graph and kernel gates on the host.
 *
 * Every config of the installed catalog that the kernel match admits on this device runs on
 * every case, forward and backward data, in fp32, fp16 and bf16: groups of 2, 4, 8 and 16
 * channels, both layouts (channels-first staged and, cf configs, in place), 2-D and 3-D (ResNeXt3D's 3x3x3 and 1x3x3 filters), stride 2 (backward
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
            direction == ConvDirection::FORWARD ? "hipkernel.conv_group_fwd.graph_match"
                                                : "hipkernel.conv_group_bwd.graph_match",
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
        std::array<std::vector<hipdnn_plugin_sdk::ingestor::KernelDefinition>, 2> catalogs;
        for(size_t i = 0; i < catalogs.size(); ++i)
        {
            const std::string file = i == 0 ? "conv_group_fwd.kdp.json" : "conv_group_bwd.kdp.json";
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
    if(direction == ConvDirection::BACKWARD_DATA
       && std::any_of(shape.stride.begin(), shape.stride.end(), [](int64_t s) { return s > 2; }))
    {
        return false;
    }
    // fp32 runs everywhere (the FMA configs where no fp32 16x16 atom exists).
    return true;
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
    constexpr unsigned int SEED = 1234;
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
    // Each output sums (taps x reduction channels) products of values in [-1, 1].
    int64_t reductionLength = (forward ? shape.c : shape.k) / shape.groups;
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
            if(!(std::fabs(got - want) <= tolerance) && ++mismatches <= 5)
            {
                ADD_FAILURE() << "at linear index " << offsetOf(index, written.strides())
                              << ": got " << got << ", want " << want << ", tolerance "
                              << tolerance;
            }
        });
        EXPECT_EQ(mismatches, 0);
    }
    // Backward data past stride 2 is not served.
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
                                                              ConvDirection::BACKWARD_DATA)),
                         [](const auto& info) {
                             return std::get<0>(info.param).name
                                    + (std::get<1>(info.param) == ConvDirection::FORWARD ? "Fwd"
                                                                                         : "Bwd");
                         });

// ---- Host-only: the graph gates, one decline per field, and the kernel gates ----

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
        // Ungrouped, depthwise (1 channel a group), 32 channels a group, groups of 3 (16 not a
        // multiple), and a channel multiplier (k != c).
        auto ungrouped = resnext();
        ungrouped.groups = 1;
        EXPECT_FALSE(parsed(ungrouped, direction).has_value());
        auto depthwise = resnext();
        depthwise.groups = 64;
        EXPECT_FALSE(parsed(depthwise, direction).has_value());
        auto wide = dense(64, 64, {12, 13}, {3, 3}, {1, 1}, true);
        wide.groups = 2;
        EXPECT_FALSE(parsed(wide, direction).has_value());
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

TEST(TestConvGroupCatalog, EveryConfigFitsItsLdsAndKnobsTellConfigsApart)
{
    static constexpr std::array<const char*, 13> KNOBS{
        "atom", "t", "r", "s", "stride", "th", "wp", "cb", "cf", "cg", "oc", "px", "pf"};
    for(const auto direction : {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA})
    {
        std::vector<std::string> tuples;
        for(const auto& kernel : catalog(direction))
        {
            SCOPED_TRACE(kernel.name);
            const int bytes = kernel.getStringMetadata("dtype") == "FLOAT" ? 4 : 2;
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
            const int cb = static_cast<int>(kernel.getIntMetadata("cb"));
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
                    conv_group::groupPhasedLdsBytes(
                        bytes, taps("r"), taps("s"), th, tw, cb, kernel.getIntMetadata("cf") != 0),
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
        }
    }
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_CONV
