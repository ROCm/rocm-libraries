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
#include "engines/kernel_ingestor_engine/packs/ConvIgemmProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveTestGraphs.hpp"
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
    static constexpr std::array<const char*, 15> KNOBS{"atom",
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
                                                       "cf"};
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

TEST(TestConvIgemmCatalog, EveryMatrixCoreConfigButTapPackingHasAChannelsFirstOutputTwin)
{
    static constexpr std::array<const char*, 14> SHAPE{"atom",
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
                                                       "pack"};
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

hipdnn_plugin_sdk::ingestor::DeviceProperties deviceOf(const std::string& arch, int units);

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
                   || kernel.getIntMetadata("wdirect") != 0)
                {
                    continue;
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

} // namespace

#endif
