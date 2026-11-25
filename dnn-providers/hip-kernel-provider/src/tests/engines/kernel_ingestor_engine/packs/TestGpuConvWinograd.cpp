// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_CONV)

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <optional>
#include <set>
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
#include "engines/kernel_ingestor_engine/packs/ConvDeterminismTestSupport.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvEpilogueDeviceTestSupport.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvEpilogueTiming.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvNaiveTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvWinogradProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

/**
 * @file TestGpuConvWinograd.cpp
 * @brief The Winograd convolution kernels (CONV_WINOGRAD_FWD, CONV_WINOGRAD_BWD,
 *        CONV_WINOGRAD_WRW) on the device, through the engines' own graph match, kernel match,
 *        prepare and launch, against hipDNN's CPU reference convolution.
 *
 * Every config of the installed catalog that the kernel match admits on this device runs on
 * every case, forward and backward data, in fp32, fp16 and bf16: the multipass engine in both
 * layouts, the fused VALU kernel and the fused matrix kernels (gfx11 / gfx1200-1201 / gfx1250).
 * Cases cover NCHW, NHWC and mixed layouts (staged), groups, padding 0, 1 and 2, asymmetric padding, odd
 * extents (partial tiles), deep channels and true convolution. The LLC-fit plan's last-level
 * cache is overridden to a few KiB so its configs slice the batch on every device.
 *
 * Winograd's error is relative to the magnitude of the tile it shares rounding with, not to the
 * output alone, so the tolerance scales with the largest sum of |products| among the outputs
 * of the neighbourhood (the reference convolution of |x| and |w|), by the bound of the study for
 * each dtype and tile: fp32 1e-5 (F(2,3)) and 5e-5 (F(4,3)), fp16 2e-3, bf16 1.6e-2.
 *
 * Backward weights (F(3,2), F(3,4), F(3,6)) runs every admitted wrw config on the stride-1 cases
 * plus batch-sliced and split-heavy ones, graded with the dynamic-tolerance model against a
 * double-accumulated reference: fp16 up to 12 eps sigma, bf16 the same absolute level (fp16
 * Winograd's), fp32 the base summation allowance (c 4).
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
    const char* graphMatch = "hipkernel.conv_winograd_wrw.graph_match";
    if(direction == ConvDirection::FORWARD)
    {
        graphMatch = "hipkernel.conv_winograd_fwd.graph_match";
    }
    else if(direction == ConvDirection::BACKWARD_DATA)
    {
        graphMatch = "hipkernel.conv_winograd_bwd.graph_match";
    }
    return {"",
            graphMatch,
            "",
            "hipkernel.conv_winograd.kernel_match",
            "",
            "hipkernel.conv_winograd.dispatch",
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
        // Each direction's engines' catalogs: forward and backward data the exact-tile engine's,
        // the bf16 engine's (CONV_WINOGRAD_*_BF16) and the F(4,3) engine's (CONV_WINOGRAD_*_F43).
        const std::vector<std::pair<size_t, std::string>> files = {
            {0, "conv_winograd_fwd.kdp.json"},
            {0, "conv_winograd_fwd_bf16.kdp.json"},
            {0, "conv_winograd_fwd_f43.kdp.json"},
            {1, "conv_winograd_bwd.kdp.json"},
            {1, "conv_winograd_bwd_bf16.kdp.json"},
            {1, "conv_winograd_bwd_f43.kdp.json"},
            {2, "conv_winograd_wrw.kdp.json"},
        };
        for(const auto& [i, file] : files)
        {
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
    size_t index = 2;
    if(direction == ConvDirection::FORWARD)
    {
        index = 0;
    }
    else if(direction == ConvDirection::BACKWARD_DATA)
    {
        index = 1;
    }
    return s_catalogs[index];
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
    /// y (forward-role output) in the other packed layout than x: the variants stage both.
    bool mixedLayouts = false;
};

ConvShape dense(int64_t c, int64_t k, std::vector<int64_t> input, int64_t pad, bool channelsLast)
{
    ConvShape shape;
    shape.n = 2;
    shape.c = c;
    shape.k = k;
    shape.groups = 1;
    shape.input = std::move(input);
    shape.filter = {3, 3};
    shape.prePadding = {pad, pad};
    shape.postPadding = {pad, pad};
    shape.stride = {1, 1};
    shape.dilation = {1, 1};
    shape.channelsLast = channelsLast;
    return shape;
}

// Odd extents leave partial Winograd tiles on both edges; channel counts are multiples of
// nothing the blocks use, so every edge of every block runs.
std::vector<DeviceCase> deviceCases()
{
    std::vector<DeviceCase> cases;
    cases.push_back({"Nchw3x3", dense(24, 40, {13, 11}, 1, false)});
    cases.push_back({"Nhwc3x3", dense(24, 40, {13, 11}, 1, true)});
    cases.push_back({"NhwcPad0OddChannels", dense(7, 5, {9, 10}, 0, true)});
    cases.push_back({"NchwPad2", dense(8, 6, {6, 7}, 2, false)});
    // The last image's aligned 12-byte patch rows exceed NUM_RECORDS although their
    // first one or two dwords hold real inputs. Exercises both alignment phases and
    // boundary words for every native cooperative config, with no tolerance relaxation.
    cases.push_back({"NchwAllocationBoundaryRows", dense(1, 1, {2, 2}, 1, false)});
    // A full native K32 chunk whose final image reaches a later tile block: no negative
    // chunk-0 address forces the per-element reload, so native vector-boundary checks matter.
    cases.push_back({"NchwNativeFullReductionBoundaryRows", dense(32, 1, {7, 9}, 1, false)});

    // One scalar input: short K4 reduction, row/column tails and windows entirely in padding.
    // Opposite asymmetric edges also exercise the negative effective padding in dgrad.
    auto tinyAsymmetric = dense(1, 1, {1, 1}, 0, true);
    tinyAsymmetric.n = 1;
    tinyAsymmetric.prePadding = {3, 0};
    tinyAsymmetric.postPadding = {0, 3};
    cases.push_back({"NhwcTinyAsymmetricPadding", tinyAsymmetric});

    auto grouped = dense(32, 48, {12, 13}, 1, false);
    grouped.groups = 4;
    cases.push_back({"NchwGrouped", grouped});

    // Native K32 reduction: a second chunk with one channel, odd row alignment, K16 output tails.
    cases.push_back({"NchwNativeReductionTail", dense(33, 35, {7, 9}, 1, false)});
    auto nativeGrouped = dense(34, 66, {9, 7}, 1, true);
    nativeGrouped.groups = 2; // C/g = 17, K/g = 33; dgrad swaps both tail contracts
    cases.push_back({"NhwcNativeGroupedReductionTail", nativeGrouped});

    auto asymmetric = dense(6, 10, {9, 5}, 1, true);
    asymmetric.prePadding = {2, 0};
    asymmetric.postPadding = {0, 1};
    cases.push_back({"NhwcAsymmetricPadding", asymmetric});

    auto convolution = dense(12, 20, {11, 9}, 1, false);
    convolution.mode = ConvMode::CONVOLUTION;
    cases.push_back({"NchwConvolutionMode", convolution});
    auto nhwcConvolution = dense(3, 5, {7, 9}, 1, true);
    nhwcConvolution.mode = ConvMode::CONVOLUTION;
    cases.push_back({"NhwcConvolutionModeOddChannels", nhwcConvolution});

    cases.push_back({"NchwDeepChannels", dense(136, 72, {7, 9}, 1, false)});
    cases.push_back({"MixedLayouts", dense(16, 24, {10, 9}, 1, true), true});
    // A transformed filter past 4 MiB (16 x 528 x 256 x 2 bytes): the RDNA cooperative plan runs
    // tile blocks fastest; several tile blocks and a partial last K block.
    cases.push_back({"NchwLargeFilterTileOrder", dense(256, 520, {9, 13}, 1, false)});

    // Stride 2 (F(3,2) on the phase sub-filters): odd and even extents, so phases run short on
    // either edge.
    const auto strided = [](ConvShape shape) {
        shape.stride = {2, 2};
        return shape;
    };
    cases.push_back({"NchwStride2", strided(dense(24, 40, {13, 11}, 1, false))});
    cases.push_back({"NhwcStride2Pad0OddChannels", strided(dense(7, 5, {9, 10}, 0, true))});
    cases.push_back({"NchwStride2Pad2", strided(dense(8, 6, {6, 7}, 2, false))});
    cases.push_back({"NchwStride2Grouped", strided(grouped)});
    cases.push_back({"NhwcStride2AsymmetricPadding", strided(asymmetric)});
    cases.push_back({"NchwStride2ConvolutionMode", strided(convolution)});
    cases.push_back({"NhwcStride2ConvolutionModeOddChannels", strided(nhwcConvolution)});
    cases.push_back({"NchwStride2Stem", strided(dense(3, 32, {17, 20}, 0, false))});
    cases.push_back({"NchwStride2DeepChannels", strided(dense(136, 72, {12, 9}, 1, false))});
    cases.push_back({"Stride2MixedLayouts", strided(dense(16, 24, {10, 9}, 1, true)), true});
    // Output extent 1 (hipconv #15149 review: a stride normalized for a size-1 axis): 4x4 -> 1x1,
    // and 4x9 -> 1x4 (1 in H only).
    cases.push_back({"NchwStride2OutputOne", strided(dense(16, 24, {4, 4}, 0, false))});
    cases.push_back({"NhwcStride2OutputOneHigh", strided(dense(8, 16, {4, 9}, 0, true))});
    return cases;
}

/// The largest sum of |products| among the outputs within @p reach (Winograd-tile reach) of
/// each output: the scale its error is relative to.
template <typename T>
std::vector<float> tileMagnitudes(Tensor<T>& magnitude, int64_t reach)
{
    const auto& dims = magnitude.dims();
    const auto* data = static_cast<const T*>(magnitude.rawHostData());
    std::vector<float> result;
    forEachIndex(dims, [&](const std::vector<int64_t>& index) {
        float largest = 0.0f;
        for(int64_t dh = -reach; dh <= reach; ++dh)
        {
            for(int64_t dw = -reach; dw <= reach; ++dw)
            {
                auto neighbour = index;
                neighbour[2] += dh;
                neighbour[3] += dw;
                if(neighbour[2] < 0 || neighbour[2] >= dims[2] || neighbour[3] < 0
                   || neighbour[3] >= dims[3])
                {
                    continue;
                }
                largest = std::max(
                    largest, static_cast<float>(data[offsetOf(neighbour, magnitude.strides())]));
            }
        }
        result.push_back(largest);
    });
    return result;
}

template <typename T>
void absoluteValues(Tensor<T>& source, Tensor<T>& target)
{
    const auto* from = static_cast<const T*>(source.rawHostData());
    auto* to = static_cast<T*>(target.rawHostData());
    forEachIndex(source.dims(), [&](const std::vector<int64_t>& index) {
        to[offsetOf(index, target.strides())] = static_cast<T>(
            std::fabs(static_cast<float>(from[offsetOf(index, source.strides())])));
    });
    target.markHostModified();
}

template <typename T>
float winogradBound(int64_t m)
{
    if constexpr(std::is_same_v<T, float>)
    {
        return m == 4 ? 5e-5f : 1e-5f;
    }
    else if constexpr(std::is_same_v<T, Half>)
    {
        return 2e-3f;
    }
    else
    {
        return 1.6e-2f;
    }
}

template <typename T>
void runCase(const DeviceCase& deviceCase,
             ConvDirection direction,
             bool nativeCoopOnly = false,
             std::optional<float> unrelatedInputNonfinite = std::nullopt)
{
    // A few KiB of "last-level cache": the LLC-fit configs slice the batch on every device.
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter llc(
        "HIP_KERNEL_PROVIDER_CONV_WINOGRAD_LLC_BYTES", "4096");
    auto shape = deviceCase.shape;
    shape.dataType = dataTypeOf<T>();
    auto spec = specFor(shape, direction);
    if(deviceCase.mixedLayouts)
    {
        spec.y.strides = packedStrides(spec.y.dims, !shape.channelsLast);
    }
    const GraphFixture fixture(buildGraph(spec), currentDeviceProperties());
    const bool native1250
        = hipdnn_plugin_sdk::stripArchFeatures(fixture.context().deviceProperties.gcnArchName)
          == "gfx1250";
    if(nativeCoopOnly && !native1250)
    {
        GTEST_SKIP() << "Native K32 cooperative convolution requires gfx1250";
    }
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
    Tensor<T>& filterForReference = reflect ? reflectedW : w;
    Tensor<T> reference(written.dims(), written.strides());
    // The same convolution of |input| and |w|: each output's sum of |products|.
    Tensor<T> absIn(forward ? x.dims() : y.dims(), forward ? x.strides() : y.strides());
    Tensor<T> absW(spec.w.dims, spec.w.strides);
    Tensor<T> magnitude(written.dims(), written.strides());
    absoluteValues(forward ? x : y, absIn);
    absoluteValues(filterForReference, absW);
    if(forward)
    {
        CpuFpReferenceConvolution::fprop(x,
                                         filterForReference,
                                         reference,
                                         shape.stride,
                                         shape.dilation,
                                         shape.prePadding,
                                         shape.postPadding);
        CpuFpReferenceConvolution::fprop(absIn,
                                         absW,
                                         magnitude,
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
        CpuFpReferenceConvolution::dgrad(magnitude,
                                         absW,
                                         absIn,
                                         shape.stride,
                                         shape.dilation,
                                         shape.prePadding,
                                         shape.postPadding);
    }
    // F(4,3) tiles are 4 outputs wide; a backward stride-2 F(3,2) tile covers 3 phase rows, 6 of
    // dx.
    const bool phasedBackward = !forward && shape.stride[0] == 2;
    const auto scale = tileMagnitudes(magnitude, phasedBackward ? 5 : 3);
    if(unrelatedInputNonfinite)
    {
        ASSERT_TRUE(forward);
        // The independent reference and its error scale use the original finite inputs.
        // Only later images/groups change; the first image's first group must stay finite.
        auto* input = static_cast<T*>(x.rawHostData());
        forEachIndex(x.dims(), [&](const std::vector<int64_t>& index) {
            if(index[0] != 0 || index[1] >= shape.c / shape.groups)
            {
                input[offsetOf(index, x.strides())] = static_cast<T>(*unrelatedInputNonfinite);
            }
        });
        x.markHostModified();
    }

    const auto& handler = dispatchHandler(pack);
    int ran = 0;
    unsigned nativeModesRan = 0;
    bool nativeMultipassRan = false;
    for(const auto& kernel : catalog(direction))
    {
        if(!matchesKernel(pack, fixture.context(), kernel, *bound))
        {
            continue;
        }
        if(nativeCoopOnly)
        {
            const auto config = conv_winograd::configOf(kernel);
            ASSERT_TRUE(config.has_value());
            if(config->variant != conv_winograd::Variant::COOP || config->atom != 4)
            {
                continue;
            }
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

        const float relative = winogradBound<T>(kernel.getIntMetadata("m"));
        const auto* actual = static_cast<const T*>(written.rawHostData());
        const auto* expected = static_cast<const T*>(reference.rawHostData());
        int64_t mismatches = 0;
        size_t visited = 0;
        forEachIndex(written.dims(), [&](const std::vector<int64_t>& index) {
            const float tileScale = scale[visited++];
            if(unrelatedInputNonfinite && (index[0] != 0 || index[1] >= shape.k / shape.groups))
            {
                return;
            }
            const auto got = static_cast<float>(actual[offsetOf(index, written.strides())]);
            const auto want = static_cast<float>(expected[offsetOf(index, reference.strides())]);
            const float tolerance
                = relative * tileScale + 2.0f * storageEpsilon<T>() * std::fabs(want) + 1e-6f;
            if(!(std::fabs(got - want) <= tolerance) && ++mismatches <= 5)
            {
                ADD_FAILURE() << "at linear index " << offsetOf(index, written.strides())
                              << ": got " << got << ", want " << want << ", tolerance "
                              << tolerance;
            }
        });
        EXPECT_EQ(mismatches, 0);
        if(native1250)
        {
            const auto& variant = kernel.getStringMetadata("variant");
            const auto config = conv_winograd::configOf(kernel);
            ASSERT_TRUE(config.has_value());
            if(config->variant == conv_winograd::Variant::MULTIPASS)
            {
                nativeMultipassRan |= config->atom == 5;
            }
            else if(config->atom >= 4)
            {
                if(variant == "wmma")
                {
                    nativeModesRan |= 1u;
                }
                else if(variant == "wreg")
                {
                    nativeModesRan |= 2u;
                }
                else
                {
                    nativeModesRan |= 4u;
                }
            }
            if(variant == "fused")
            {
                nativeModesRan |= 8u;
            }
        }
    }
    EXPECT_GE(ran, 1) << "no catalog config serves this case";
    // Numeric proof must exercise each intended family, not silently pass on VALU alone.
    if(native1250)
    {
        const unsigned expected = [&]() -> unsigned {
            if(nativeCoopOnly)
            {
                return 4u;
            }
            if(shape.stride[0] == 2)
            {
                return 8u;
            }
            if constexpr(std::is_same_v<T, float>)
            {
                return 9u;
            }
            return 15u;
        }();
        EXPECT_EQ(nativeModesRan, expected);
        if constexpr(std::is_same_v<T, float>)
        {
            EXPECT_TRUE(nativeMultipassRan)
                << "no exact-f32 native multipass catalog config executed";
        }
    }
}

class TestGpuConvWinograd : public ::testing::TestWithParam<std::tuple<DeviceCase, ConvDirection>>
{
};

TEST_P(TestGpuConvWinograd, Float)
{
    SKIP_IF_NO_DEVICES();
    runCase<float>(std::get<0>(GetParam()), std::get<1>(GetParam()));
}

TEST_P(TestGpuConvWinograd, Half)
{
    SKIP_IF_NO_DEVICES();
    runCase<Half>(std::get<0>(GetParam()), std::get<1>(GetParam()));
}

TEST_P(TestGpuConvWinograd, BFloat16)
{
    SKIP_IF_NO_DEVICES();
    runCase<BFloat16>(std::get<0>(GetParam()), std::get<1>(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuConvWinograd,
                         ::testing::Combine(::testing::ValuesIn(deviceCases()),
                                            ::testing::Values(ConvDirection::FORWARD,
                                                              ConvDirection::BACKWARD_DATA)),
                         [](const auto& info) {
                             return std::get<0>(info.param).name
                                    + (std::get<1>(info.param) == ConvDirection::FORWARD ? "Fwd"
                                                                                         : "Bwd");
                         });

TEST(TestGpuConvWinogradNative, HalfAllocationBoundary)
{
    SKIP_IF_NO_DEVICES();
    auto shape = dense(32, 1, {4, 4}, 0, false);
    shape.n = 1;
    runCase<Half>({"NchwNativeAllocationBoundary", shape}, ConvDirection::FORWARD, true);
}

TEST(TestGpuConvWinogradNative, HalfPartialReductionIsolatesImages)
{
    SKIP_IF_NO_DEVICES();
    // The second K32 chunk contains one live channel. A K16-based full-chunk guard
    // incorrectly reads the following image into its other 31 channels, where 0 * Inf/NaN
    // must not contaminate the first image's output.
    const auto shape = dense(33, 1, {4, 4}, 0, false);
    for(float nonfinite :
        {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()})
    {
        SCOPED_TRACE(std::isnan(nonfinite) ? "NaN in the other image" : "Inf in the other image");
        runCase<Half>(
            {"NchwNativePartialReductionImages", shape}, ConvDirection::FORWARD, true, nonfinite);
    }
}

TEST(TestGpuConvWinogradNative, HalfPartialReductionIsolatesGroups)
{
    SKIP_IF_NO_DEVICES();
    auto shape = dense(66, 2, {4, 4}, 0, false);
    shape.n = 1;
    shape.groups = 2; // C/g = 33: a full K32 chunk followed by one live channel.
    for(float nonfinite :
        {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()})
    {
        SCOPED_TRACE(std::isnan(nonfinite) ? "NaN in the other group" : "Inf in the other group");
        runCase<Half>(
            {"NchwNativePartialReductionGroups", shape}, ConvDirection::FORWARD, true, nonfinite);
    }
}

// ---- Backward weights ----

struct WrwCase
{
    std::string name;
    ConvShape shape;
    bool mixedLayouts = false;
    /// The engine's workspace limit in MiB (0: default): small enough to slice the batch.
    int workspaceMb = 0;
};

std::vector<WrwCase> wrwCases()
{
    std::vector<WrwCase> cases;
    for(const auto& deviceCase : deviceCases())
    {
        if(deviceCase.shape.stride[0] == 1)
        {
            cases.push_back({deviceCase.name, deviceCase.shape, deviceCase.mixedLayouts});
        }
    }
    // Many tiles over few channel blocks: the GEMM splits over the tiles.
    auto deep = dense(16, 24, {33, 35}, 1, false);
    deep.n = 6;
    cases.push_back({"NchwManyTilesSplit", deep});
    auto deepNhwc = deep;
    deepNhwc.channelsLast = true;
    cases.push_back({"NhwcManyTilesSplit", deepNhwc});
    // A small workspace limit: the batch runs in slices that add onto the partials (2 MiB for
    // channels-last, whose transposed x and dy copies take part of it).
    auto sliced = dense(40, 72, {12, 14}, 1, true);
    sliced.n = 9;
    cases.push_back({"NhwcSlicedBatch", sliced, false, 2});
    auto slicedNchw = sliced;
    slicedNchw.channelsLast = false;
    slicedNchw.groups = 2;
    cases.push_back({"NchwSlicedBatchGrouped", slicedNchw, false, 1});
    // Top padding 3 over one input row: dW rows 0 and 1 see only padding (exactly 0).
    auto deadTaps = dense(8, 8, {1, 5}, 1, false);
    deadTaps.prePadding = {3, 1};
    deadTaps.postPadding = {0, 1};
    cases.push_back({"NchwDeadTaps", deadTaps});
    return cases;
}

template <typename T>
void runWrwCase(const WrwCase& wrwCase)
{
    const std::string limit = std::to_string(wrwCase.workspaceMb);
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter workspaceLimit(
        "HIP_KERNEL_PROVIDER_CONV_WINOGRAD_WORKSPACE_MB", wrwCase.workspaceMb > 0 ? limit : "");
    auto shape = wrwCase.shape;
    shape.dataType = dataTypeOf<T>();
    auto spec = specFor(shape, ConvDirection::BACKWARD_WEIGHTS);
    if(wrwCase.mixedLayouts)
    {
        spec.y.strides = packedStrides(spec.y.dims, !shape.channelsLast);
    }
    const GraphFixture fixture(buildGraph(spec), currentDeviceProperties());
    const auto pack = packFor(ConvDirection::BACKWARD_WEIGHTS);
    const auto bound = matchesGraph(pack, fixture.context());
    ASSERT_TRUE(bound.has_value()) << "the engine refused a case it should serve";

    Tensor<T> x(spec.x.dims, spec.x.strides);
    Tensor<T> w(spec.w.dims, spec.w.strides);
    Tensor<T> y(spec.y.dims, spec.y.strides);
    constexpr unsigned int SEED = 4321;
    x.fillTensorWithRandomValues(-1.0f, 1.0f, SEED);
    y.fillTensorWithRandomValues(-1.0f, 1.0f, SEED + 1);

    // The cross-correlation dW (reflected for true convolution), accumulated in double.
    Tensor<float> correlation(spec.w.dims, spec.w.strides);
    Tensor<float> reference(spec.w.dims, spec.w.strides);
    CpuFpReferenceConvolution::wgrad<T, float, T, double>(
        x, correlation, y, shape.stride, shape.dilation, shape.prePadding, shape.postPadding);
    if(shape.mode == ConvMode::CONVOLUTION)
    {
        reflectFilter(correlation, reference);
    }
    else
    {
        forEachIndex(correlation.dims(), [&](const std::vector<int64_t>& index) {
            static_cast<float*>(reference.rawHostData())[offsetOf(index, reference.strides())]
                = static_cast<const float*>(
                    correlation.rawHostData())[offsetOf(index, correlation.strides())];
        });
        reference.markHostModified();
    }
    // The dynamic-tolerance model's band (dnn-providers/integration-tests docs/tolerances.md):
    // K products per dW element, sigma = sqrt(K mean(x^2) mean(dy^2)); output rounding 2 eps |ref|,
    // summation order c sqrt(K) u32 (sigma (+ |ref| for fp32)), intermediates lambda eps sigma.
    // fp16: lambda 12 (the model's half value); bf16: the same absolute level, fp16 Winograd's
    // (12 fp16 eps = 1.5 bf16 eps); fp32: the base model's c 4 (the shipped tiles measured at most
    // 2.31).
    const auto meanSquare = [](Tensor<T>& tensor) {
        double sum = 0.0;
        size_t count = 0;
        const auto* data = static_cast<const T*>(tensor.rawHostData());
        forEachIndex(tensor.dims(), [&](const std::vector<int64_t>& index) {
            const auto v
                = static_cast<double>(static_cast<float>(data[offsetOf(index, tensor.strides())]));
            sum += v * v;
            ++count;
        });
        return sum / static_cast<double>(count);
    };
    const auto reductions = static_cast<double>(spec.y.dims[0] * spec.y.dims[2] * spec.y.dims[3]);
    const double sigma = std::sqrt(reductions * meanSquare(x) * meanSquare(y));
    constexpr bool FP32 = std::is_same_v<T, float>;
    double eps = std::ldexp(1.0, -7);
    double lambda = 1.5;
    if constexpr(FP32)
    {
        eps = std::ldexp(1.0, -23);
        lambda = 0.0;
    }
    else if constexpr(std::is_same_v<T, Half>)
    {
        eps = std::ldexp(1.0, -10);
        lambda = 12.0;
    }
    const double summation = (FP32 ? 4.0 : 1.0) * std::sqrt(reductions) * std::ldexp(1.0, -24);
    const double minNormal = std::is_same_v<T, Half> ? std::ldexp(1.0, -14) : std::ldexp(1.0, -126);

    const auto& handler = dispatchHandler(pack);
    int ran = 0;
    for(const auto& kernel : catalog(ConvDirection::BACKWARD_WEIGHTS))
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
        if(wrwCase.workspaceMb > 0)
        {
            EXPECT_LE(workspaceBytes, static_cast<size_t>(wrwCase.workspaceMb) << 20);
        }
        void* workspace = nullptr;
        if(workspaceBytes > 0)
        {
            ASSERT_EQ(hipMalloc(&workspace, workspaceBytes), hipSuccess);
            // Stale partials or operands must not leak into the result.
            ASSERT_EQ(hipMemset(workspace, 0x7f, workspaceBytes), hipSuccess);
        }
        w.fillWithSentinelValue();
        const std::array<hipdnnPluginDeviceBuffer_t, 3> buffers{
            hipdnnPluginDeviceBuffer_t{X_UID, x.rawDeviceData()},
            hipdnnPluginDeviceBuffer_t{W_UID, w.rawDeviceData()},
            hipdnnPluginDeviceBuffer_t{Y_UID, y.rawDeviceData()}};
        const Handle handle;
        handler.launch(
            handle, *prepared, buffers.data(), static_cast<uint32_t>(buffers.size()), workspace);
        ASSERT_EQ(hipStreamSynchronize(handle.getStream()), hipSuccess);
        // Deterministic: a second run over a re-poisoned workspace gives the same bits.
        const size_t dwBytes = static_cast<size_t>(w.elementCount()) * sizeof(T);
        std::vector<unsigned char> first(dwBytes);
        std::vector<unsigned char> second(dwBytes);
        ASSERT_EQ(hipMemcpy(first.data(), w.rawDeviceData(), dwBytes, hipMemcpyDeviceToHost),
                  hipSuccess);
        if(workspace != nullptr)
        {
            ASSERT_EQ(hipMemset(workspace, 0x5a, workspaceBytes), hipSuccess);
        }
        handler.launch(
            handle, *prepared, buffers.data(), static_cast<uint32_t>(buffers.size()), workspace);
        ASSERT_EQ(hipStreamSynchronize(handle.getStream()), hipSuccess);
        ASSERT_EQ(hipMemcpy(second.data(), w.rawDeviceData(), dwBytes, hipMemcpyDeviceToHost),
                  hipSuccess);
        EXPECT_TRUE(first == second) << "two runs differ";
        if(workspace != nullptr)
        {
            ASSERT_EQ(hipFree(workspace), hipSuccess);
        }
        w.markDeviceModified();

        const auto* actual = static_cast<const T*>(w.rawHostData());
        const auto* expected = static_cast<const float*>(reference.rawHostData());
        int64_t mismatches = 0;
        forEachIndex(w.dims(), [&](const std::vector<int64_t>& index) {
            const auto got
                = static_cast<double>(static_cast<float>(actual[offsetOf(index, w.strides())]));
            const auto want = static_cast<double>(expected[offsetOf(index, reference.strides())]);
            const double tolerance = (2.0 * eps + (FP32 ? summation : 0.0)) * std::fabs(want)
                                     + (summation + lambda * eps) * sigma + minNormal;
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

class TestGpuConvWinogradWrw : public ::testing::TestWithParam<WrwCase>
{
};

TEST_P(TestGpuConvWinogradWrw, Float)
{
    SKIP_IF_NO_DEVICES();
    runWrwCase<float>(GetParam());
}

TEST_P(TestGpuConvWinogradWrw, Half)
{
    SKIP_IF_NO_DEVICES();
    runWrwCase<Half>(GetParam());
}

TEST_P(TestGpuConvWinogradWrw, BFloat16)
{
    SKIP_IF_NO_DEVICES();
    runWrwCase<BFloat16>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuConvWinogradWrw,
                         ::testing::ValuesIn(wrwCases()),
                         [](const auto& info) { return info.param.name; });
// ---- Host-only: the shipped catalog ----

TEST(TestConvWinogradCatalog, ApplicableCompiledContractsHaveDistinctIntegerKnobs)
{
    constexpr std::array KNOBS = {"m",
                                  "llc",
                                  "gemm_bm",
                                  "gemm_bn",
                                  "gemm_bk",
                                  "bk",
                                  "bt",
                                  "ccv",
                                  "tm",
                                  "tn",
                                  "waves",
                                  "bins",
                                  "f16math",
                                  "pf",
                                  "sb",
                                  "variant_id",
                                  "layout_id",
                                  "atom"};
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter smallCache(
        "HIP_KERNEL_PROVIDER_CONV_WINOGRAD_LLC_BYTES", "4096");
    for(const auto direction :
        {ConvDirection::FORWARD, ConvDirection::BACKWARD_DATA, ConvDirection::BACKWARD_WEIGHTS})
    {
        for(const auto* arch : {"gfx1100", "gfx1201", "gfx1250", "gfx942"})
        {
            for(const auto dtype : {DataType::FLOAT, DataType::HALF, DataType::BFLOAT16})
            {
                auto shape = dense(34, 66, {9, 7}, 1, false);
                shape.groups = 2;
                shape.dataType = dtype;
                auto properties = testDeviceProperties();
                properties.gcnArchName = arch;
                properties.multiProcessorCount = 64;
                const GraphFixture fixture(buildGraph(specFor(shape, direction)), properties);
                const auto pack = packFor(direction);
                const auto bound = matchesGraph(pack, fixture.context());
                ASSERT_TRUE(bound.has_value());
                std::set<std::vector<int64_t>> seen;
                for(const auto& kernel : catalog(direction))
                {
                    SCOPED_TRACE(kernel.name);
                    const auto config = conv_winograd::configOf(kernel);
                    ASSERT_TRUE(config.has_value());
                    if(matchesKernel(pack, fixture.context(), kernel, *bound))
                    {
                        std::vector<int64_t> key;
                        key.reserve(KNOBS.size());
                        for(const auto* knob : KNOBS)
                        {
                            key.push_back(kernel.getIntMetadata(knob));
                        }
                        EXPECT_TRUE(seen.insert(std::move(key)).second)
                            << "two applicable kernels would enroll the same config";
                    }
                    if(config->arch == "gfx12"
                       && (config->variant == conv_winograd::Variant::WMMA
                           || config->variant == conv_winograd::Variant::WREG
                           || config->variant == conv_winograd::Variant::COOP
                           || config->variant == conv_winograd::Variant::WRW))
                    {
                        EXPECT_FALSE(conv_winograd::configRunsOn(*config, "gfx1250"));
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fused pointwise epilogues (forward): every kernel that writes y -- the multipass output pass
// (both layouts), the fused VALU, register VALU, WMMA, register WMMA and cooperative kernels,
// F(2,3), F(3,2) by phases and F(4,3) -- once per distinct path among the configs the kernel
// match admits, on the menu chains. Outputs whose window lies in the padding (exact 0 from the
// convolution) get the epilogue of 0.
// ---------------------------------------------------------------------------

using conv_epilogue_test::Specials;
using conv_epilogue_test::Step;
using FusedChain = std::pair<std::string, std::vector<Step>>;

/// The config fields that select a different kernel or store path.
const std::vector<std::string>& fusedPathFields()
{
    static const std::vector<std::string> s_fields{
        "variant", "arch", "m", "layout", "bins", "f16math", "pf", "llc"};
    return s_fields;
}

std::vector<DeviceCase> fusedShapes()
{
    std::vector<DeviceCase> cases;
    cases.push_back({"Nchw", dense(8, 12, {13, 11}, 1, false)});
    cases.push_back({"Nhwc", dense(8, 12, {13, 11}, 1, true)});
    cases.push_back({"NchwPad2", dense(8, 6, {6, 7}, 2, false)});
    auto grouped = dense(16, 16, {9, 10}, 1, false);
    grouped.groups = 4;
    cases.push_back({"NchwGrouped", grouped});
    const auto strided = [](ConvShape shape) {
        shape.stride = {2, 2};
        return shape;
    };
    cases.push_back({"NchwStride2", strided(dense(8, 12, {13, 11}, 1, false))});
    cases.push_back({"NhwcStride2", strided(dense(8, 12, {13, 11}, 1, true))});
    return cases;
}

template <typename T>
void runFused(const ConvShape& shape, const std::vector<Step>& steps, Specials specials)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter llc(
        "HIP_KERNEL_PROVIDER_CONV_WINOGRAD_LLC_BYTES", "4096");
    const auto runner = conv_epilogue_test::catalogRunner(
        packFor(ConvDirection::FORWARD), catalog(ConvDirection::FORWARD), fusedPathFields());
    // Winograd's error is relative to the sum of |products| of the tile (the bound of the study:
    // F(4,3) for fp32); with x and w in [-1, 1] that sum is at most half the reduction.
    const auto tolerance = [](int64_t reduction) {
        return static_cast<double>(winogradBound<T>(4)) * 0.5 * static_cast<double>(reduction);
    };
    EXPECT_GE(conv_epilogue_test::checkFusedCase<T>(shape, steps, runner, specials, tolerance), 1)
        << "no catalog config serves this fused case";
}

class TestGpuConvWinogradFused : public ::testing::TestWithParam<std::tuple<DeviceCase, FusedChain>>
{
};

TEST_P(TestGpuConvWinogradFused, Float)
{
    SKIP_IF_NO_DEVICES();
    runFused<float>(std::get<0>(GetParam()).shape, std::get<1>(GetParam()).second, Specials::NONE);
}

TEST_P(TestGpuConvWinogradFused, Half)
{
    SKIP_IF_NO_DEVICES();
    runFused<Half>(std::get<0>(GetParam()).shape, std::get<1>(GetParam()).second, Specials::NONE);
}

TEST_P(TestGpuConvWinogradFused, BFloat16)
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
                         TestGpuConvWinogradFused,
                         ::testing::Combine(::testing::ValuesIn(fusedShapes()),
                                            ::testing::ValuesIn(fusedChainsAllShapes())),
                         [](const auto& info) {
                             return std::get<0>(info.param).name + "_"
                                    + std::get<1>(info.param).first;
                         });

INSTANTIATE_TEST_SUITE_P(Menu,
                         TestGpuConvWinogradFused,
                         ::testing::Combine(::testing::Values(fusedShapes()[0], fusedShapes()[4]),
                                            ::testing::ValuesIn(conv_epilogue_test::menuChains())),
                         [](const auto& info) {
                             return std::get<0>(info.param).name + "_"
                                    + std::get<1>(info.param).first;
                         });

class TestGpuConvWinogradFusedSpecials : public ::testing::TestWithParam<DeviceCase>
{
};

TEST_P(TestGpuConvWinogradFusedSpecials, NanAndInfinityOperands)
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

TEST_P(TestGpuConvWinogradFusedSpecials, ClipBoundaries)
{
    SKIP_IF_NO_DEVICES();
    runFused<Half>(
        GetParam().shape,
        {conv_epilogue_test::bias(DataType::HALF), conv_epilogue_test::relu(-0.5F, 0.5F)},
        Specials::CLIP_BOUNDARY);
}

INSTANTIATE_TEST_SUITE_P(,
                         TestGpuConvWinogradFusedSpecials,
                         ::testing::ValuesIn(fusedShapes()),
                         [](const auto& info) { return info.param.name; });

TEST(TestConvWinogradFusion, OnlyForwardMatchersClaimFusedGraphs)
{
    ConvShape shape = dense(8, 12, {13, 11}, 1, false);
    shape.dataType = DataType::HALF;
    const auto spec = conv_epilogue_test::fusedSpecFor(
        shape, {conv_epilogue_test::bias(DataType::HALF), conv_epilogue_test::relu()});
    const GraphFixture fixture(conv_epilogue_test::buildFusedGraph(spec));
    EXPECT_TRUE(matchesGraph(packFor(ConvDirection::FORWARD), fixture.context()).has_value());
    EXPECT_FALSE(
        matchesGraph(packFor(ConvDirection::BACKWARD_DATA), fixture.context()).has_value());
    EXPECT_FALSE(
        matchesGraph(packFor(ConvDirection::BACKWARD_WEIGHTS), fixture.context()).has_value());
    // The parse gives the epilogue only when the caller asks for it.
    EXPECT_FALSE(
        conv_winograd::parseWinogradProblem(fixture.context(), ConvDirection::FORWARD).has_value());
    const auto fused = conv_winograd::parseWinogradProblem(
        fixture.context(), ConvDirection::FORWARD, conv_naive::Fusion::EPILOGUE);
    ASSERT_TRUE(fused.has_value());
    EXPECT_FALSE(fused->base.epilogue.empty());
    EXPECT_EQ(fused->yUid, conv_epilogue_test::outputUid(2));
}

// Opt-in: each config's time with and without a fused bias + relu (shapes in $WINO_EPI_TIMING).
TEST(TestGpuConvWinogradFusedTiming, BareAgainstFused)
{
    SKIP_IF_NO_DEVICES();
    conv_epilogue_timing::timeFusedAgainstBare(
        packFor(ConvDirection::FORWARD), catalog(ConvDirection::FORWARD), "WINO_EPI_TIMING");
}

} // namespace

#endif
