// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HKP_TEST_GFX1250_CONV_FWD_AOT)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/CacheRoot.hpp>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PathSanitizer.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/ConvolutionFpropAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_frontend/knob/Knob.hpp>
#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_plugin_sdk/GlobalKnobDefines.hpp>
#include <hipdnn_plugin_sdk/ingestor/WinnerCacheFile.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_test_sdk/utilities/ScopedTestCacheDir.hpp>
#include <hipdnn_test_sdk/utilities/SdkFrontendTypeConversions.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/GraphTensorBundle.hpp>
#include <nlohmann/json.hpp>

#include "../IntegrationGraphVerificationHarness.hpp"
#include "IngestorIntegrationSupport.hpp"
#include "ScopedPluginLogCapture.hpp"

// ---- hipkernel:Gfx1250ConvFwdAot: the shipped forward catalog on the GPU ----
//
// The engine ships the instances rocKE's gfx1250 forward instance list registers
// (IngestorGenerator/configs/gfx1250_conv_fwd_aot_{fp16,bf16}.instances.json): per dtype
// implicit-GEMM instances and the direct families (WMMA, grouped-stream, tap-outer, small
// C, small K, depthwise WMMA and the depthwise pipelines). An instance is shape-generic:
// it bakes at most the filter, stride, dilation, padding and per-group channel counts (0
// is taken at run time), and the dispatcher's admission decides the rest.
//
// The UED knobs are the KMD's family knobs. Instances that differ only in baked geometry
// share every knob, so pinning a kernel's knobs on a graph selects whichever of those
// siblings serves it; no two siblings serve one graph below. The catalog suite pins every
// instance's knobs on every graph whose capability it could serve, executes each one the
// engine accepts with Y poisoned (NaN) beforehand, checks it against
// CpuReferenceGraphExecutor, and requires every instance to have run at least once.
// The per-graph suite runs an unbenchmarked plan (whole-catalog and per-family counts) and
// a benchmark under HIPDNN_INGESTOR_SHORTLIST_LIMIT.

namespace hip_kernel_provider::kernel_ingestor_engine::integration
{

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_test_sdk::utilities;
namespace ingestor = hipdnn_plugin_sdk::ingestor;

namespace
{

constexpr const char* ENGINE_NAME = "hipkernel:Gfx1250ConvFwdAot";
constexpr const char* SERVED_ARCH = "gfx1250";
constexpr const char* DESCRIPTOR_PREFIX = "gfx1250_conv_fwd_aot";
constexpr const char* ALGORITHM_KNOB = "algorithm";
/// Every kernel family (the `algorithm` KMD field) of the catalog; the order of
/// FwdGraph's counts.
constexpr std::array<const char*, 11> FAMILIES{"implicit_gemm",
                                               "direct_fwd_wmma",
                                               "direct_fwd_gstream_wmma",
                                               "direct_fwd_tapouter_wmma",
                                               "direct_fwd_smallc_wmma",
                                               "direct_fwd_smallk_wmma",
                                               "direct_dw_fwd_wmma",
                                               "direct_depthwise",
                                               "direct_depthwise_tiled",
                                               "direct_depthwise_col",
                                               "direct_depthwise_spatial"};
/// Instances per dtype in the shipped catalog.
constexpr size_t INSTANCES_PER_DTYPE = 263;
/// The shortlist limit of the capped benchmark: below the candidate total of most graphs
/// (up to 31), so the cap truncates their shortlist, and above that of others.
constexpr size_t BENCHMARK_LIMIT = 8;

/// Whether a plan build failed because no candidate carries the requested knob values: the
/// frontend checks each value against the values the engine advertises for the graph, and
/// the engine checks the combination.
bool knobsUnsatisfiable(const std::string& error, const LogRecorderBase& recorder)
{
    for(const auto* marker :
        {"no kernel satisfying the requested knob setting", "is not in the list of valid values"})
    {
        if(error.find(marker) != std::string::npos || recorder.hasLogContaining(marker))
        {
            return true;
        }
    }
    return false;
}

constexpr int64_t X_UID = 1;
constexpr int64_t W_UID = 2;
constexpr int64_t Y_UID = 3;

struct FwdProblem
{
    int64_t n = 2;
    int64_t c = 64;
    int64_t k = 64;
    int64_t hi = 14;
    int64_t wi = 14;
    int64_t y = 3;
    int64_t x = 3;
    int64_t strideH = 1;
    int64_t strideW = 1;
    int64_t padH = 1;
    int64_t padW = 1;
    int64_t dilationH = 1;
    int64_t dilationW = 1;
    // hipDNN carries no group attribute: groups = X channels / W dims[1].
    int64_t groups = 1;

    int64_t cpg() const
    {
        return c / groups;
    }
    int64_t kpg() const
    {
        return k / groups;
    }
    int64_t ho() const
    {
        return (hi + 2 * padH - dilationH * (y - 1) - 1) / strideH + 1;
    }
    int64_t wo() const
    {
        return (wi + 2 * padW - dilationW * (x - 1) - 1) / strideW + 1;
    }
};

/// One graph and how many catalog instances of each family (FAMILIES order) serve it, the
/// same in fp16 and bf16, as gfx1250_conv_fwd_aot_serves in
/// rocke/library/builders/gfx1250/conv_fwd_aot.py (the dispatcher's admission) counts them
/// over gfx1250_conv_fwd_aot_{fp16,bf16}.instances.json. The native kernel matcher is
/// checked against these counts.
struct FwdGraph
{
    std::string name;
    FwdProblem problem;
    std::array<size_t, FAMILIES.size()> counts;

    size_t count(const std::string& family) const
    {
        for(size_t i = 0; i < FAMILIES.size(); ++i)
        {
            if(family == FAMILIES[i])
            {
                return counts[i];
            }
        }
        return 0;
    }
    size_t total() const
    {
        size_t sum = 0;
        for(const auto value : counts)
        {
            sum += value;
        }
        return sum;
    }
};

// N is 2 wherever an instance's tile allows it (4 where it bakes a 4-image tile), so a
// write past the last Y row of an image lands in the next one and fails. Odd spatial sizes
// leave partial tiles. The first nine graphs are the problems of the former family-knob
// suite; the rest were chosen so that together the graphs reach every instance of the
// catalog: each direct instance's baked filter, stride, padding, dilation and channel
// counts, at extents where its padded tile grid is at least half real work.
// clang-format off
const std::vector<FwdGraph>& fwdGraphs()
{
    static const std::vector<FwdGraph> s_graphs{
        {"Dense3x3", {2, 64, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {9, 9, 2, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"Pointwise1x1Stride2", {1, 128, 128, 28, 28, 1, 1, 2, 2, 0, 0, 1, 1, 1}, {9, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"GroupedG4", {2, 64, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 4}, {9, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"Depthwise", {2, 64, 64, 28, 28, 3, 3, 1, 1, 1, 1, 1, 1, 64}, {3, 0, 0, 0, 0, 0, 17, 2, 6, 1, 0}},
        {"Dilated2", {1, 64, 64, 32, 32, 3, 3, 1, 1, 2, 2, 2, 2, 1}, {9, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0}},
        {"Row1x7PadW3", {1, 32, 32, 8, 64, 1, 7, 1, 1, 0, 3, 1, 1, 1}, {9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"Stem7x7Stride2", {2, 3, 64, 64, 64, 7, 7, 2, 2, 3, 3, 1, 1, 1}, {0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0}},
        {"Dense3x3TileN4", {4, 64, 64, 24, 24, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {9, 18, 2, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"Dense3x3Pad0TileN4", {4, 64, 64, 26, 26, 3, 3, 1, 1, 0, 0, 1, 1, 1}, {9, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"DepthwiseFilter7x7Pad3", {2, 32, 32, 14, 14, 7, 7, 1, 1, 3, 3, 1, 1, 32}, {3, 0, 0, 0, 0, 0, 13, 0, 0, 0, 0}},
        {"DepthwiseFilter3x3Stride2Pad1", {2, 32, 32, 14, 14, 3, 3, 2, 2, 1, 1, 1, 1, 32}, {3, 0, 0, 0, 0, 0, 0, 2, 8, 0, 0}},
        {"DenseFilter3x3Stride2Pad1C48K96", {2, 48, 96, 16, 64, 3, 3, 2, 2, 1, 1, 1, 1, 1}, {9, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"DepthwiseFilter5x5Stride2Pad2", {2, 32, 32, 14, 14, 5, 5, 2, 2, 2, 2, 1, 1, 32}, {3, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0}},
        {"DepthwiseFilter5x5Pad2", {2, 32, 32, 14, 14, 5, 5, 1, 1, 2, 2, 1, 1, 32}, {3, 0, 0, 0, 0, 0, 5, 0, 1, 0, 0}},
        {"GroupedG4Filter3x3Stride2Pad1C128K128", {2, 128, 128, 13, 29, 3, 3, 2, 2, 1, 1, 1, 1, 4}, {9, 1, 6, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"DenseFilter2x2Stride2Pad0C64K64", {2, 64, 64, 32, 32, 2, 2, 2, 2, 0, 0, 1, 1, 1}, {9, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"DenseFilter1x3Pad0x1W256", {2, 64, 64, 8, 256, 1, 3, 1, 1, 0, 1, 1, 1, 1}, {9, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"DenseFilter1x7Pad0x3W512", {2, 48, 96, 4, 512, 1, 7, 1, 1, 0, 3, 1, 1, 1}, {9, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"DenseFilter1x3Pad0x5Dilation1x5W512", {2, 64, 64, 4, 512, 1, 3, 1, 1, 0, 5, 1, 5, 1}, {9, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"DenseFilter3x3Pad1C48K96", {2, 48, 96, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {9, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"DenseFilter1x7Pad0x15Dilation1x5W512", {2, 48, 96, 4, 512, 1, 7, 1, 1, 0, 15, 1, 5, 1}, {9, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"GroupedG4Filter3x3Stride2Pad1C64K64", {2, 64, 64, 13, 29, 3, 3, 2, 2, 1, 1, 1, 1, 4}, {9, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"DenseFilter3x3Stride2Pad0", {2, 64, 128, 13, 29, 3, 3, 2, 2, 0, 0, 1, 1, 1}, {9, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"DenseFilter1x3Pad0x3Dilation1x3W512", {2, 64, 64, 4, 512, 1, 3, 1, 1, 0, 3, 1, 3, 1}, {9, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"DenseFilter1x7Pad0x9Dilation1x3W512", {2, 48, 96, 4, 512, 1, 7, 1, 1, 0, 9, 1, 3, 1}, {9, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"GroupedG16Filter3x3Stride2Pad1", {2, 128, 128, 14, 14, 3, 3, 2, 2, 1, 1, 1, 1, 16}, {9, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"GroupedG8Filter3x3Pad1C64K64", {2, 64, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 8}, {9, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"GroupedG2Filter3x3Pad1C64K64", {2, 64, 64, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 2}, {9, 5, 3, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"DenseFilter7x1Pad3x0", {2, 64, 64, 14, 14, 7, 1, 1, 1, 3, 0, 1, 1, 1}, {9, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"DenseFilter3x3Pad36Dilation36", {2, 48, 96, 14, 14, 3, 3, 1, 1, 36, 36, 36, 36, 1}, {9, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0}},
        {"GroupedG4Filter3x3Stride2Pad1C224K224", {2, 224, 224, 32, 32, 3, 3, 2, 2, 1, 1, 1, 1, 4}, {9, 6, 3, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"DenseFilter1x11Pad0x15Dilation1x3W512", {2, 64, 64, 4, 512, 1, 11, 1, 1, 0, 15, 1, 3, 1}, {9, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"DenseFilter1x11Pad0x5W512", {2, 64, 64, 4, 512, 1, 11, 1, 1, 0, 5, 1, 1, 1}, {9, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"DenseFilter1x11Pad0x25Dilation1x5W512", {2, 64, 64, 4, 512, 1, 11, 1, 1, 0, 25, 1, 5, 1}, {9, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"DepthwiseFilter7x7Pad0", {2, 32, 32, 14, 14, 7, 7, 1, 1, 0, 0, 1, 1, 32}, {3, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0}},
        {"DenseFilter1x1Stride2Pad0C48K96", {2, 48, 96, 28, 28, 1, 1, 2, 2, 0, 0, 1, 1, 1}, {9, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"DenseFilter3x1Pad1x0", {2, 48, 96, 14, 14, 3, 1, 1, 1, 1, 0, 1, 1, 1}, {9, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"DenseFilter3x3Pad24Dilation24", {2, 48, 96, 14, 14, 3, 3, 1, 1, 24, 24, 24, 24, 1}, {9, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0}},
        {"DenseFilter3x3Pad12Dilation12", {2, 48, 96, 14, 14, 3, 3, 1, 1, 12, 12, 12, 12, 1}, {9, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0}},
        {"DenseFilter3x3Pad4Dilation4C48K96", {2, 48, 96, 13, 29, 3, 3, 1, 1, 4, 4, 4, 4, 1}, {9, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0}},
        {"DenseFilter3x3Stride2Pad1C64K128", {2, 64, 128, 32, 32, 3, 3, 2, 2, 1, 1, 1, 1, 1}, {9, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"DenseFilter3x3Pad1C64K64", {2, 64, 64, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {9, 20, 2, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"GroupedG8Filter3x3Stride2Pad1", {2, 64, 64, 13, 29, 3, 3, 2, 2, 1, 1, 1, 1, 8}, {9, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"DepthwiseFilter11x11Pad5", {2, 32, 32, 14, 14, 11, 11, 1, 1, 5, 5, 1, 1, 32}, {3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}},
        {"GroupedG2Filter3x3Stride2Pad1", {2, 32, 32, 28, 28, 3, 3, 2, 2, 1, 1, 1, 1, 2}, {9, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"DenseFilter1x1Stride2Pad0C64K64", {2, 64, 64, 16, 64, 1, 1, 2, 2, 0, 0, 1, 1, 1}, {9, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"DenseFilter3x3Stride2Pad1C64K64", {2, 64, 64, 14, 14, 3, 3, 2, 2, 1, 1, 1, 1, 1}, {9, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"GroupedG4Filter3x3Pad1", {2, 32, 32, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 4}, {9, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"GroupedG2Filter3x3Pad1C32K32", {2, 32, 32, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 2}, {9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"DenseFilter1x7Pad0x3", {2, 64, 64, 14, 14, 1, 7, 1, 1, 0, 3, 1, 1, 1}, {9, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"DenseFilter2x2Stride2Pad0C128K256", {2, 128, 256, 14, 14, 2, 2, 2, 2, 0, 0, 1, 1, 1}, {9, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"DenseFilter1x3Pad0x1W512", {2, 64, 64, 4, 512, 1, 3, 1, 1, 0, 1, 1, 1, 1}, {9, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"DenseFilter3x3Pad4Dilation4C128K256", {2, 128, 256, 14, 14, 3, 3, 1, 1, 4, 4, 4, 4, 1}, {9, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0}},
        {"DenseFilter3x3Pad2Dilation2", {2, 128, 256, 14, 14, 3, 3, 1, 1, 2, 2, 2, 2, 1}, {9, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0}},
        {"GroupedG8Filter3x3Pad1C32K32", {2, 32, 32, 13, 29, 3, 3, 1, 1, 1, 1, 1, 1, 8}, {0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {"SmallC3Filter3x3Stride2Pad1", {2, 3, 32, 13, 29, 3, 3, 2, 2, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0}},
        {"SmallC4Filter3x3Pad1", {2, 4, 32, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0}},
        {"SmallC3Filter3x3Stride2Pad0", {2, 3, 32, 13, 29, 3, 3, 2, 2, 0, 0, 1, 1, 1}, {0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0}},
        {"SmallC3Filter4x4Stride4Pad0", {2, 3, 32, 32, 32, 4, 4, 4, 4, 0, 0, 1, 1, 1}, {0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0}},
        {"SmallC3Filter3x3Pad1", {2, 3, 32, 13, 29, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0}},
        {"SmallK4Filter3x3Pad1", {2, 32, 4, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0}},
        {"SmallK3Filter3x3Pad1", {2, 32, 3, 14, 14, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0}},
        {"SmallK1Filter1x7Pad0x3W256", {2, 32, 1, 8, 256, 1, 7, 1, 1, 0, 3, 1, 1, 1}, {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0}},
    };
    return s_graphs;
}
// clang-format on

struct FwdVariant
{
    std::string name;
    DataType dtype;
    /// Index into fwdGraphs(); the catalog suite runs every graph and leaves it empty.
    std::optional<size_t> graph;
};

// gtest prints a parameter on failure; the name says everything about it.
// NOLINTNEXTLINE(readability-identifier-naming) -- gtest customization point
void PrintTo(const FwdVariant& variant, std::ostream* stream)
{
    *stream << variant.name;
}

std::string dtypeSuffix(DataType dtype)
{
    return dtype == DataType::HALF ? "Fp16" : "Bf16";
}

std::string dtypeToken(DataType dtype)
{
    return dtype == DataType::HALF ? "fp16" : "bf16";
}

std::vector<FwdVariant> graphVariants()
{
    std::vector<FwdVariant> variants;
    for(size_t index = 0; index < fwdGraphs().size(); ++index)
    {
        for(const auto dtype : {DataType::HALF, DataType::BFLOAT16})
        {
            variants.push_back({fwdGraphs()[index].name + dtypeSuffix(dtype), dtype, index});
        }
    }
    return variants;
}

/// X stored NHWC (NCHW with @p channelsLast false), W KYXC, Y NHWK.
std::shared_ptr<Graph>
    buildFwdGraph(DataType dtype, const FwdProblem& problem, bool channelsLast = true)
{
    auto graph = std::make_shared<Graph>();
    graph->set_name("gfx1250_conv_fwd")
        .set_io_data_type(dtype)
        .set_intermediate_data_type(dtype)
        .set_compute_data_type(DataType::FLOAT);

    const auto& p = problem;
    // X [N, C, Hi, Wi].
    const auto x = std::make_shared<TensorAttributes>();
    x->set_uid(X_UID)
        .set_name("X")
        .set_dim({p.n, p.c, p.hi, p.wi})
        .set_stride(channelsLast ? std::vector<int64_t>{p.c * p.hi * p.wi, 1, p.wi * p.c, p.c}
                                 : std::vector<int64_t>{p.c * p.hi * p.wi, p.hi * p.wi, p.wi, 1})
        .set_data_type(dtype);
    // W [K, C/groups, Y, X] stored KYXC over the group's channels.
    const auto cpg = p.cpg();
    const auto w = std::make_shared<TensorAttributes>();
    w->set_uid(W_UID)
        .set_name("W")
        .set_dim({p.k, cpg, p.y, p.x})
        .set_stride({cpg * p.y * p.x, 1, p.x * cpg, cpg})
        .set_data_type(dtype);

    ConvFpropAttributes attributes;
    attributes.set_name("gfx1250_conv_fwd")
        .set_padding({p.padH, p.padW})
        .set_stride({p.strideH, p.strideW})
        .set_dilation({p.dilationH, p.dilationW})
        .set_convolution_mode(ConvolutionMode::CROSS_CORRELATION);
    const auto y = graph->conv_fprop(x, w, attributes);
    // Y [N, K, Ho, Wo] stored NHWK.
    const auto ho = p.ho();
    const auto wo = p.wo();
    y->set_uid(Y_UID)
        .set_name("Y")
        .set_dim({p.n, p.k, ho, wo})
        .set_stride({p.k * ho * wo, 1, wo * p.k, p.k})
        .set_output(true)
        .set_data_type(dtype);
    return graph;
}

KnobSetting benchmarkingKnob(bool enabled)
{
    return KnobSetting(hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME, int64_t{enabled ? 1 : 0});
}

std::string candidateCount(size_t candidates)
{
    return std::to_string(candidates) + " candidate(s)";
}

/// The kernel id following @p marker in the last log that carries it, or empty.
std::string loggedKernel(const LogRecorderBase& recorder, const std::string& marker)
{
    std::string kernel;
    for(const auto& log : recorder.getRecordedLogs())
    {
        const auto at = log.message.find(marker);
        if(at == std::string::npos)
        {
            continue;
        }
        const auto begin = at + marker.size();
        kernel = log.message.substr(begin, log.message.find(' ', begin) - begin);
    }
    return kernel;
}

std::string selectedKernel(const LogRecorderBase& recorder)
{
    return loggedKernel(recorder, detail::SELECTED_KERNEL_LOG);
}

std::string servedKernel(const LogRecorderBase& recorder)
{
    return loggedKernel(recorder, "' served kernel ");
}

std::string canonicalUuid(const std::string& text)
{
    return ingestor::toString(hipdnn_flatbuffers_sdk::utilities::parseUuid(text));
}

// ---- The shipped catalog, read from the product tree main() loads ----

/// The shape fields an instance may bake, in BakedShape's order
/// (packs/Gfx1250ConvFwdInstance.hpp).
constexpr std::array<const char*, 10> BAKED_FIELDS{"filter_h",
                                                   "filter_w",
                                                   "stride_h",
                                                   "stride_w",
                                                   "dilation_h",
                                                   "dilation_w",
                                                   "pad_h",
                                                   "pad_w",
                                                   "cpg",
                                                   "kpg"};

/// One kernel of the shipped forward bundle: what the test needs to pin it and to know
/// which graphs it could serve.
struct CatalogKernel
{
    std::string id;
    std::string name;
    std::string dtype;
    std::string algorithm;
    std::array<int64_t, BAKED_FIELDS.size()> baked{};
    /// Every UED knob, at the kernel's value: an ordinal for a string field, the
    /// descriptor value for an int one (0 unset; a bool knob 1 true, 2 false).
    std::vector<KnobSetting> knobs;
    /// The values of knobs, in the same order: equal for siblings that differ only in
    /// baked geometry.
    std::vector<int64_t> knobValues;

    /// Whether the kernel's baked capability admits the graph: every baked field is 0
    /// (taken at run time) or the graph's. The engine's admission judges the rest.
    bool capabilityAdmits(const FwdProblem& problem) const
    {
        const std::array<int64_t, BAKED_FIELDS.size()> actual{problem.y,
                                                              problem.x,
                                                              problem.strideH,
                                                              problem.strideW,
                                                              problem.dilationH,
                                                              problem.dilationW,
                                                              problem.padH,
                                                              problem.padW,
                                                              problem.cpg(),
                                                              problem.kpg()};
        for(size_t i = 0; i < baked.size(); ++i)
        {
            if(baked[i] != 0 && baked[i] != actual[i])
            {
                return false;
            }
        }
        return true;
    }
};

struct FwdCatalog
{
    std::filesystem::path directory;
    std::vector<CatalogKernel> kernels;
    /// The ordinal of every value of each non-integer KMD field, as the engine numbers
    /// them: the sorted distinct values over the engine's whole catalog, defaults applied.
    std::map<std::string, std::map<std::string, int64_t>> ordinals;

    std::optional<int64_t> ordinal(const std::string& field, const std::string& value) const
    {
        const auto domain = ordinals.find(field);
        if(domain == ordinals.end())
        {
            return std::nullopt;
        }
        const auto found = domain->second.find(value);
        return found == domain->second.end() ? std::nullopt : std::optional{found->second};
    }

    const CatalogKernel* find(const std::string& id) const
    {
        const auto found = std::find_if(
            kernels.begin(), kernels.end(), [&](const auto& kernel) { return kernel.id == id; });
        return found == kernels.end() ? nullptr : &*found;
    }
};

nlohmann::json readJson(const std::filesystem::path& path)
{
    std::ifstream stream(path);
    return nlohmann::json::parse(stream);
}

/// The shipped forward bundle under HIPDNN_DESCRIPTOR_RUNTIME_DIR, read once. Empty when it
/// is not there.
const FwdCatalog& shippedCatalog()
{
    static const FwdCatalog s_catalog = [] {
        FwdCatalog catalog;
        const std::filesystem::path root
            = hipdnn_data_sdk::utilities::getEnv("HIPDNN_DESCRIPTOR_RUNTIME_DIR");
        std::error_code error;
        if(root.empty() || !std::filesystem::is_directory(root, error))
        {
            return catalog;
        }
        const std::string kmdName = std::string(DESCRIPTOR_PREFIX) + ".kmd.json";
        for(auto it = std::filesystem::recursive_directory_iterator(root, error);
            !error && it != std::filesystem::recursive_directory_iterator();
            it.increment(error))
        {
            if(it->path().filename() == kmdName)
            {
                catalog.directory = it->path().parent_path();
                break;
            }
        }
        if(catalog.directory.empty())
        {
            return catalog;
        }

        const auto kmd = readJson(catalog.directory / kmdName);
        const auto ued
            = readJson(catalog.directory / (std::string(DESCRIPTOR_PREFIX) + ".ued.json"));
        std::vector<std::pair<std::string, nlohmann::json>> fields;
        for(const auto& field : kmd.at("fields"))
        {
            fields.emplace_back(field.at("name").get<std::string>(),
                                field.contains("default_value") ? field.at("default_value")
                                                                : nlohmann::json());
        }

        // Every kernel's metadata with the KMD defaults filled in, in a fixed order.
        std::vector<std::pair<std::string, nlohmann::json>> documents;
        const std::string prefix = std::string(DESCRIPTOR_PREFIX) + "_";
        const std::string suffix = ".kdp.json";
        std::vector<std::filesystem::path> shards;
        // Recursive: the kdp.json shards live under the bundle's artifacts/ (DVC-pulled),
        // not directly beside the kmd.json.
        for(const auto& file :
            std::filesystem::recursive_directory_iterator(catalog.directory, error))
        {
            const auto name = file.path().filename().string();
            if(name.size() > prefix.size() + suffix.size() && name.rfind(prefix, 0) == 0
               && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
            {
                shards.push_back(file.path());
            }
        }
        std::sort(shards.begin(), shards.end());
        for(const auto& shard : shards)
        {
            // Held by name: a range over a member of a temporary would dangle.
            const auto document = readJson(shard);
            for(const auto& kernel : document.at("kernelDescriptors"))
            {
                auto metadata = kernel.at("metadata");
                for(const auto& [field, defaultValue] : fields)
                {
                    if(!metadata.contains(field) && !defaultValue.is_null())
                    {
                        metadata[field] = defaultValue;
                    }
                }
                metadata["__id"] = kernel.at("id");
                metadata["__name"] = kernel.at("name");
                documents.emplace_back(kernel.at("id").get<std::string>(), std::move(metadata));
            }
        }

        for(const auto& [field, unused] : fields)
        {
            std::set<std::string> values;
            for(const auto& [id, metadata] : documents)
            {
                if(metadata.contains(field) && metadata.at(field).is_string())
                {
                    values.insert(metadata.at(field).get<std::string>());
                }
            }
            int64_t index = 0;
            for(const auto& value : values)
            {
                catalog.ordinals[field][value] = index++;
            }
        }

        for(const auto& [id, metadata] : documents)
        {
            CatalogKernel kernel;
            kernel.id = canonicalUuid(id);
            kernel.name = metadata.at("__name").get<std::string>();
            kernel.dtype = metadata.at("dtype").get<std::string>();
            kernel.algorithm = metadata.at("algorithm").get<std::string>();
            for(size_t i = 0; i < BAKED_FIELDS.size(); ++i)
            {
                kernel.baked[i] = metadata.at(BAKED_FIELDS[i]).get<int64_t>();
            }
            for(const auto& knob : ued.at("knobs"))
            {
                const auto name = knob.get<std::string>();
                const auto& value = metadata.at(name);
                const auto knobValue = value.is_string()
                                           ? catalog.ordinals.at(name).at(value.get<std::string>())
                                           : value.get<int64_t>();
                kernel.knobs.emplace_back(name, knobValue);
                kernel.knobValues.push_back(knobValue);
            }
            catalog.kernels.push_back(std::move(kernel));
        }
        return catalog;
    }();
    return s_catalog;
}

/// X and W randomized once, Y computed once on the host; the GPU side is reused by every
/// plan of the graph, with Y poisoned before each execution.
struct Reference
{
    std::unique_ptr<GraphTensorBundle> gpu;
    std::unique_ptr<GraphTensorBundle> cpu;
    std::vector<uint8_t> serialized;
};

enum class PinOutcome
{
    SERVED,
    REFUSED,
    FAILED
};

} // namespace

class IntegrationGpuGfx1250ConvFwd
    : public test_utilities::IntegrationGraphVerificationHarness<float, FwdVariant>
{
protected:
    using Base = test_utilities::IntegrationGraphVerificationHarness<float, FwdVariant>;

    void SetUp() override
    {
        Base::SetUp();
        if(HasFatalFailure() || IsSkipped())
        {
            return;
        }
        hipDeviceProp_t properties{};
        ASSERT_EQ(hipGetDeviceProperties(&properties, _deviceId), hipSuccess);
        _arch = properties.gcnArchName;
        if(!hipdnn_plugin_sdk::archMatches(
               _arch, SERVED_ARCH, hipdnn_plugin_sdk::ArchMatchMode::PREFIX))
        {
            GTEST_SKIP() << ENGINE_NAME << " ships for " << SERVED_ARCH << " only; this device is "
                         << _arch;
        }
    }

    void TearDown() override
    {
        if(_workspace != nullptr)
        {
            EXPECT_EQ(hipFree(_workspace), hipSuccess);
            _workspace = nullptr;
        }
        Base::TearDown();
    }

    static int64_t engineId()
    {
        return hipdnn_data_sdk::utilities::engineNameToId(ENGINE_NAME);
    }

    static const FwdGraph& graphOf(const FwdVariant& variant)
    {
        return fwdGraphs().at(variant.graph.value());
    }

    /// The ordinal the engine addresses @p family by; a failure when the catalog has none.
    static int64_t familyOrdinal(const std::string& family)
    {
        const auto ordinal = shippedCatalog().ordinal(ALGORITHM_KNOB, family);
        EXPECT_TRUE(ordinal) << family << " is not in the shipped catalog";
        return ordinal.value_or(-1);
    }

    /// Builds the operation graph and requires the engine among those offered for it.
    void buildOperationGraphTheEngineOffersToServe(Graph& graph)
    {
        auto result = graph.build_operation_graph(_handle);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        std::vector<int64_t> engineIds;
        result = graph.get_ranked_engine_ids(engineIds);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        ASSERT_NE(std::find(engineIds.begin(), engineIds.end(), engineId()), engineIds.end())
            << ENGINE_NAME << " did not offer to serve this graph; the production descriptors "
            << "for " << SERVED_ARCH << " must be packed";
    }

    /// Builds @p graph's plan for this engine with @p knobs. Returns the error text of the
    /// first failing stage, empty on success.
    std::string tryBuild(Graph& graph, const std::vector<KnobSetting>& knobs)
    {
        auto result = graph.build_operation_graph(_handle);
        if(result.code != ErrorCode::OK)
        {
            return "build_operation_graph: " + result.err_msg;
        }
        std::vector<int64_t> engineIds;
        result = graph.get_ranked_engine_ids(engineIds);
        if(result.code != ErrorCode::OK)
        {
            return "get_ranked_engine_ids: " + result.err_msg;
        }
        if(std::find(engineIds.begin(), engineIds.end(), engineId()) == engineIds.end())
        {
            return std::string("the plugin did not offer ") + ENGINE_NAME;
        }
        // Hard selection: a preferred-engine hint can fall back silently to another
        // provider.
        result = graph.create_execution_plan_ext(engineId(), knobs);
        if(result.code != ErrorCode::OK)
        {
            return "create_execution_plan_ext: " + result.err_msg;
        }
        result = graph.check_support();
        if(result.code != ErrorCode::OK)
        {
            return "check_support: " + result.err_msg;
        }
        result = graph.build_plans();
        if(result.code != ErrorCode::OK)
        {
            return "build_plans: " + result.err_msg;
        }
        int64_t selected = 0;
        result = graph.get_execution_plan_engine_id(selected);
        if(result.code != ErrorCode::OK || selected != engineId())
        {
            return "the plan's engine is not " + std::string(ENGINE_NAME);
        }
        return {};
    }

    void buildAndCompile(Graph& graph, const std::vector<KnobSetting>& knobs)
    {
        const auto error = tryBuild(graph, knobs);
        ASSERT_TRUE(error.empty()) << error;
    }

    Reference makeReference(const FwdVariant& variant, const FwdProblem& problem, unsigned seed)
    {
        // Any built graph of the problem has the same tensors as every plan's graph.
        const auto graph = buildFwdGraph(variant.dtype, problem);
        const auto built = graph->build_operation_graph(_handle);
        EXPECT_EQ(built.code, ErrorCode::OK) << built.err_msg;
        Reference reference;
        auto [serialized, serializationError] = graph->to_binary();
        EXPECT_TRUE(serializationError.is_good()) << serializationError.get_message();
        reference.serialized.assign(serialized.begin(), serialized.end());
        const auto wrapper
            = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper::fromSerializedBlob(
                reference.serialized.data(), reference.serialized.size());
        reference.gpu = std::make_unique<GraphTensorBundle>(wrapper.getTensorMap());
        reference.cpu = std::make_unique<GraphTensorBundle>(wrapper.getTensorMap());
        for(const auto uid : {X_UID, W_UID})
        {
            const auto tensorSeed = seed + static_cast<unsigned int>(uid);
            reference.gpu->randomizeTensor(uid, -1.0F, 1.0F, tensorSeed);
            reference.cpu->randomizeTensor(uid, -1.0F, 1.0F, tensorSeed);
        }
        reference.cpu->getTensor(Y_UID).fillWithSentinelValue();
        CpuReferenceGraphExecutor().execute(reference.serialized.data(),
                                            reference.serialized.size(),
                                            reference.cpu->toHostVariantPack());
        return reference;
    }

    /// Executes @p graph's plan with Y poisoned to NaN and compares Y with the reference.
    void executeAndVerify(Graph& graph, DataType dtype, Reference& reference)
    {
        int64_t workspaceSize = -1;
        auto result = graph.get_workspace_size(workspaceSize);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        ASSERT_GE(workspaceSize, 0);
        if(static_cast<size_t>(workspaceSize) > _workspaceSize)
        {
            if(_workspace != nullptr)
            {
                ASSERT_EQ(hipFree(_workspace), hipSuccess);
                _workspace = nullptr;
            }
            ASSERT_EQ(hipMalloc(&_workspace, static_cast<size_t>(workspaceSize)), hipSuccess);
            _workspaceSize = static_cast<size_t>(workspaceSize);
        }
        if(_workspace != nullptr)
        {
            // _workspace is reused, never reallocated, across every instance this harness
            // runs in a test process: an instance under-writing its workspace would
            // otherwise read a previous sibling's values and pass. 0xFF bytes are NaN in
            // both fp16 and bf16.
            ASSERT_EQ(hipMemsetAsync(_workspace, 0xFF, _workspaceSize, _stream), hipSuccess);
        }
        // Every Y element the kernel does not write stays NaN and fails the comparison.
        reference.gpu->getTensor(Y_UID).fillWithSentinelValue();
        auto buffers = reference.gpu->toDeviceVariantPack();
        result = graph.execute(_handle, buffers, workspaceSize > 0 ? _workspace : nullptr);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        ASSERT_EQ(hipStreamSynchronize(_stream), hipSuccess);
        reference.gpu->getTensor(Y_UID).markDeviceModified();

        // FP32 accumulation followed by a low-precision store; inputs bounded to [-1,1].
        const float relativeTolerance = dtype == DataType::HALF ? 0.002F : 0.016F;
        const auto validator
            = createAllCloseValidator(frontendToSdkDataType(dtype), 0.001F, relativeTolerance);
        EXPECT_TRUE(
            validator->allClose(reference.cpu->getTensor(Y_UID), reference.gpu->getTensor(Y_UID)));
    }

    /// Pins every knob of @p kernel on a fresh graph of @p problem. SERVED: the engine
    /// selected one of @p siblings (the kernels sharing every knob of @p kernel), stored in
    /// @p served, and its Y matched the reference; REFUSED: no candidate satisfied the
    /// knobs; FAILED: anything else (recorded as a test failure).
    PinOutcome pinAndVerify(const FwdVariant& variant,
                            const FwdProblem& problem,
                            const CatalogKernel& kernel,
                            const std::vector<const CatalogKernel*>& siblings,
                            Reference& reference,
                            IsolatedLogRecorder& recorder,
                            size_t expectedBeforeFiltering,
                            const CatalogKernel*& served)
    {
        recorder.clearLogs();
        const auto graph = buildFwdGraph(variant.dtype, problem);
        auto knobs = kernel.knobs;
        knobs.push_back(benchmarkingKnob(false));
        const auto error = tryBuild(*graph, knobs);
        if(!error.empty())
        {
            if(knobsUnsatisfiable(error, recorder))
            {
                return PinOutcome::REFUSED;
            }
            ADD_FAILURE() << kernel.name << ": " << error << "\n"
                          << recorder.getRecordedLogsAsString();
            return PinOutcome::FAILED;
        }
        const auto expected = "at rank 0 from 1 candidate(s) ("
                              + std::to_string(expectedBeforeFiltering) + " before knob filtering)";
        const auto selected = selectedKernel(recorder);
        const auto sibling
            = std::find_if(siblings.begin(), siblings.end(), [&](const CatalogKernel* candidate) {
                  return candidate->id == selected;
              });
        if(sibling == siblings.end() || !recorder.hasLogContaining(expected))
        {
            ADD_FAILURE() << kernel.name << " pinned, but expected one of its " << siblings.size()
                          << " knob sibling(s) " << expected << ": "
                          << recorder.getRecordedLogsAsString();
            return PinOutcome::FAILED;
        }
        served = *sibling;
        const auto failuresBefore
            = ::testing::UnitTest::GetInstance()->current_test_info()->result()->total_part_count();
        executeAndVerify(*graph, variant.dtype, reference);
        const auto* result = ::testing::UnitTest::GetInstance()->current_test_info()->result();
        for(int part = failuresBefore; part < result->total_part_count(); ++part)
        {
            if(result->GetTestPartResult(part).failed())
            {
                ADD_FAILURE() << "wrong Y from " << served->name << " (" << served->id << ")";
                return PinOutcome::FAILED;
            }
        }
        return PinOutcome::SERVED;
    }

    // ---- Winner-record plumbing for the benchmarked plan ----

    std::filesystem::path winnerShardPath() const
    {
        const auto engineDir = hipdnn_data_sdk::utilities::cacheRoot() / "ingestor-winners"
                               / std::string(ingestor::winnerCacheVersion())
                               / hipdnn_data_sdk::utilities::sanitizeForPath(ENGINE_NAME);
        const std::string arch(hipdnn_plugin_sdk::stripArchFeatures(_arch));
        std::vector<std::filesystem::path> shards;
        std::error_code error;
        for(auto it = std::filesystem::recursive_directory_iterator(engineDir, error);
            !error && it != std::filesystem::recursive_directory_iterator();
            it.increment(error))
        {
            const auto& path = it->path();
            if(path.filename() == "winners.jsonl" && path.parent_path().filename() == arch)
            {
                shards.push_back(path);
            }
        }
        EXPECT_LE(shards.size(), 1U) << "more than one winner shard under " << engineDir;
        return shards.empty() ? std::filesystem::path{} : shards.front();
    }

    std::optional<ingestor::WinnerRecord> persistedRanking() const
    {
        std::ifstream stream(winnerShardPath());
        std::string line;
        std::optional<ingestor::WinnerRecord> ranking;
        while(std::getline(stream, line))
        {
            if(const auto entry = ingestor::decodeWinnerRecordLine(line))
            {
                EXPECT_FALSE(ranking.has_value());
                ranking = entry->second;
            }
        }
        return ranking;
    }

    ScopedTestCacheDir _cacheDir{"gfx1250-conv-fwd", ScopedTestCacheDir::Scope::TEST};
    ScopedEnvironmentVariableSetter _forceBenchmarking{"HIPDNN_FORCE_BENCHMARKING", ""};
    ScopedEnvironmentVariableSetter _cacheEnabled{"HIPDNN_DISABLE_CACHE", ""};
    ScopedEnvironmentVariableSetter _shortlistLimit{
        hipdnn_plugin_sdk::INGESTOR_SHORTLIST_LIMIT_ENV_NAME, ""};
    std::string _arch;
    void* _workspace = nullptr;
    size_t _workspaceSize = 0;
};

// ---- Per graph: the unbenchmarked plan and a capped benchmark ----

class IntegrationGpuGfx1250ConvFwdAot : public IntegrationGpuGfx1250ConvFwd
{
};

// The whole catalog's count, and each family's count through the algorithm knob: a family
// with no instance for the graph is not offered (the knob never falls back to another
// family). The fallback score (the catalog priority) is not restated here; the selected
// kernel must come from a family that serves the graph.
TEST_P(IntegrationGpuGfx1250ConvFwdAot, UnbenchmarkedPlanServesFromTheWholeCatalog)
{
    const auto& variant = GetParam();
    const auto& graphCase = graphOf(variant);
    const auto& catalog = shippedCatalog();
    ASSERT_FALSE(catalog.kernels.empty()) << "the shipped forward bundle was not found";
    const auto total = graphCase.total();
    const ScopedPluginLogCapture capture(this);
    auto& recorder = capture.recorder();
    auto reference = makeReference(variant, graphCase.problem, 53);

    auto graph = buildFwdGraph(variant.dtype, graphCase.problem);
    ASSERT_NO_FATAL_FAILURE(buildAndCompile(*graph, {benchmarkingKnob(false)}));
    // Only "before knob filtering" (the pack-admitted total, independent of ranking) is
    // pinned here: with no knob requested, shadowByArchTier can rank fewer than `total`
    // candidates (one per arch tier), so "at rank 0 from N" does not have to equal it.
    EXPECT_TRUE(recorder.hasLogContaining("(" + std::to_string(total) + " before knob filtering)"))
        << recorder.getRecordedLogsAsString();
    const auto* selected = catalog.find(selectedKernel(recorder));
    ASSERT_NE(selected, nullptr) << recorder.getRecordedLogsAsString();
    EXPECT_EQ(selected->dtype, dtypeToken(variant.dtype));
    EXPECT_TRUE(selected->capabilityAdmits(graphCase.problem)) << selected->name;
    EXPECT_GT(graphCase.count(selected->algorithm), 0U) << selected->name;
    // The installed trained ranker orders the candidates. A model that cannot score one of
    // them logs an ERROR and ranks the whole catalog by priority (`decided_by=declared_order
    // reason=ranking_failed`), which every check above would accept.
    if(total > 1)
    {
        EXPECT_TRUE(recorder.hasLogContaining("decided_by=model"))
            << recorder.getRecordedLogsAsString();
    }
    EXPECT_EQ(recorder.countLogsAtLevel(HIPDNN_SEV_ERROR), 0U)
        << recorder.getRecordedLogsAsString();
    ASSERT_NO_FATAL_FAILURE(executeAndVerify(*graph, variant.dtype, reference));
    graph.reset();

    for(size_t i = 0; i < FAMILIES.size(); ++i)
    {
        const KnobSetting knob(ALGORITHM_KNOB, familyOrdinal(FAMILIES[i]));
        const auto count = graphCase.counts[i];
        recorder.clearLogs();
        const auto pinned = buildFwdGraph(variant.dtype, graphCase.problem);
        const auto error = tryBuild(*pinned, {knob, benchmarkingKnob(false)});
        if(count == 0)
        {
            EXPECT_TRUE(knobsUnsatisfiable(error, recorder)) << FAMILIES[i] << ": " << error;
            continue;
        }
        ASSERT_TRUE(error.empty()) << FAMILIES[i] << ": " << error;
        EXPECT_TRUE(recorder.hasLogContaining("at rank 0 from " + candidateCount(count) + " ("
                                              + std::to_string(total) + " before knob filtering)"))
            << FAMILIES[i] << ": " << recorder.getRecordedLogsAsString();
        const auto* pinnedKernel = catalog.find(selectedKernel(recorder));
        ASSERT_NE(pinnedKernel, nullptr) << recorder.getRecordedLogsAsString();
        EXPECT_EQ(pinnedKernel->algorithm, FAMILIES[i]);
        ASSERT_NO_FATAL_FAILURE(executeAndVerify(*pinned, variant.dtype, reference));
    }
}

// Benchmarking under HIPDNN_INGESTOR_SHORTLIST_LIMIT measures only the best-ranked
// candidates, and its record serves the next build of the graph.
TEST_P(IntegrationGpuGfx1250ConvFwdAot, CappedBenchmarkMatchesCpuReference)
{
    const auto& variant = GetParam();
    const auto& graphCase = graphOf(variant);
    const ScopedPluginLogCapture capture(this);
    auto& recorder = capture.recorder();
    const auto total = graphCase.total();
    auto reference = makeReference(variant, graphCase.problem, 71);
    const auto capped = std::min(BENCHMARK_LIMIT, total);
    _shortlistLimit.setValue(std::to_string(BENCHMARK_LIMIT));

    auto graph = buildFwdGraph(variant.dtype, graphCase.problem);
    ASSERT_NO_FATAL_FAILURE(buildAndCompile(*graph, {benchmarkingKnob(true)}));
    EXPECT_TRUE(recorder.hasLogContaining("will benchmark " + candidateCount(capped) + " ("
                                          + std::to_string(total) + " before knob filtering)"))
        << recorder.getRecordedLogsAsString();
    ASSERT_NO_FATAL_FAILURE(executeAndVerify(*graph, variant.dtype, reference));
    graph.reset();

    const auto ranking = persistedRanking();
    ASSERT_TRUE(ranking) << "Benchmarking did not persist its measured candidate ranking";
    ASSERT_EQ(ranking->size(), capped);
    for(const auto& candidate : *ranking)
    {
        EXPECT_TRUE(std::isfinite(candidate.timeMs));
        EXPECT_GT(candidate.timeMs, 0.0);
    }
    const std::string winner = ingestor::toString(ranking->front().kernelId);
    EXPECT_TRUE(recorder.hasLogContaining("benchmarking selected kernel " + winner))
        << recorder.getRecordedLogsAsString();

    recorder.clearLogs();
    const auto recreated = buildFwdGraph(variant.dtype, graphCase.problem);
    ASSERT_NO_FATAL_FAILURE(buildAndCompile(*recreated, {benchmarkingKnob(true)}));
    EXPECT_EQ(servedKernel(recorder), winner) << recorder.getRecordedLogsAsString();
    const auto count = std::to_string(capped);
    EXPECT_TRUE(recorder.hasLogContaining("from a benchmarked record of " + count + " entry(s) for "
                                          + count + " candidate(s)"))
        << recorder.getRecordedLogsAsString();
    EXPECT_FALSE(recorder.hasLogContaining("will benchmark"));
    ASSERT_NO_FATAL_FAILURE(executeAndVerify(*recreated, variant.dtype, reference));
}

// The engine serves channels-last storage only, so a channels-first graph is not offered
// to it at all.
TEST_F(IntegrationGpuGfx1250ConvFwdAot, DeclinesChannelsFirstStorage)
{
    const auto graph = buildFwdGraph(DataType::HALF, FwdProblem{}, /*channelsLast=*/false);
    const auto result = graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    std::vector<int64_t> engineIds;
    const auto ranked = graph->get_ranked_engine_ids(engineIds);
    EXPECT_TRUE(ranked.code == ErrorCode::OK || ranked.code == ErrorCode::GRAPH_NOT_SUPPORTED)
        << ranked.err_msg;
    EXPECT_EQ(std::find(engineIds.begin(), engineIds.end(), engineId()), engineIds.end());
}

// The engine advertises only the families that serve the graph: a dilated 3x3 has no
// direct_fwd_wmma instance, so that algorithm is not offered, and pinning it builds no plan
// and selects no kernel (the knob never falls back to another family).
TEST_F(IntegrationGpuGfx1250ConvFwdAot, AFamilyThatDoesNotServeTheGraphIsNotOffered)
{
    const ScopedPluginLogCapture capture(this);
    const auto& recorder = capture.recorder();
    const auto graph
        = buildFwdGraph(DataType::HALF, {1, 64, 64, 32, 32, 3, 3, 1, 1, 2, 2, 2, 2, 1});
    ASSERT_NO_FATAL_FAILURE(buildOperationGraphTheEngineOffersToServe(*graph));

    std::vector<Knob> knobs;
    auto result = graph->get_knobs_for_engine(engineId(), knobs);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    const auto implicitGemm = familyOrdinal("implicit_gemm");
    const auto directWmma = familyOrdinal("direct_fwd_wmma");
    ASSERT_TRUE(detail::advertises(knobs, ALGORITHM_KNOB, implicitGemm))
        << "the family that serves the graph is not advertised";
    EXPECT_FALSE(detail::advertises(knobs, ALGORITHM_KNOB, directWmma))
        << "a family with no instance for the graph is advertised";

    result
        = graph->create_execution_plan_ext(engineId(), {KnobSetting(ALGORITHM_KNOB, directWmma)});
    if(result.code == ErrorCode::OK)
    {
        result = graph->build_plans();
    }
    EXPECT_NE(result.code, ErrorCode::OK) << "a plan was built for a family that serves none";
    const auto selection = std::string("engine '") + ENGINE_NAME + detail::SELECTED_KERNEL_LOG;
    EXPECT_FALSE(recorder.hasLogContaining(selection)) << "Captured logs:\n"
                                                       << recorder.getRecordedLogsAsString();
}

// ---- Per dtype: every catalog instance pinned and checked ----

class IntegrationGpuGfx1250ConvFwdAotCatalog : public IntegrationGpuGfx1250ConvFwd
{
};

// Every instance of the dtype is pinned on every graph whose capability it could serve;
// the engine must accept exactly the counts above, each accepted instance must match the
// reference, and every instance must run at least once. Instances sharing every knob
// (siblings differing only in baked geometry) are pinned once per graph.
TEST_P(IntegrationGpuGfx1250ConvFwdAotCatalog, EveryInstanceMatchesCpuReference)
{
    const auto& variant = GetParam();
    const auto& catalog = shippedCatalog();
    ASSERT_FALSE(catalog.kernels.empty()) << "the shipped forward bundle was not found";
    std::vector<const CatalogKernel*> instances;
    for(const auto& kernel : catalog.kernels)
    {
        if(kernel.dtype == dtypeToken(variant.dtype))
        {
            instances.push_back(&kernel);
        }
    }
    EXPECT_EQ(instances.size(), INSTANCES_PER_DTYPE);

    const ScopedPluginLogCapture capture(this);
    auto& recorder = capture.recorder();
    std::set<std::string> covered;
    size_t executions = 0;
    unsigned int seed = 101;
    for(const auto& graphCase : fwdGraphs())
    {
        SCOPED_TRACE(graphCase.name);
        auto reference = makeReference(variant, graphCase.problem, seed++);
        // The instances the graph's capability admits, grouped by their knob values.
        std::map<std::vector<int64_t>, std::vector<const CatalogKernel*>> pins;
        for(const auto* kernel : instances)
        {
            if(kernel->capabilityAdmits(graphCase.problem))
            {
                pins[kernel->knobValues].push_back(kernel);
            }
        }
        std::map<std::string, size_t> served;
        for(const auto& [values, siblings] : pins)
        {
            const CatalogKernel* ran = nullptr;
            const auto outcome = pinAndVerify(variant,
                                              graphCase.problem,
                                              *siblings.front(),
                                              siblings,
                                              reference,
                                              recorder,
                                              graphCase.total(),
                                              ran);
            if(outcome != PinOutcome::SERVED)
            {
                continue;
            }
            ++executions;
            covered.insert(ran->id);
            ++served[ran->algorithm];
        }
        for(const auto* family : FAMILIES)
        {
            EXPECT_EQ(served[family], graphCase.count(family)) << family;
        }
    }

    std::vector<std::string> missing;
    for(const auto* kernel : instances)
    {
        if(covered.count(kernel->id) == 0)
        {
            missing.push_back(kernel->name);
        }
    }
    EXPECT_TRUE(missing.empty()) << missing.size() << " instance(s) never ran, first "
                                 << (missing.empty() ? "" : missing.front());
    std::cout << "[ fwd      ] " << variant.name << ": " << covered.size() << " of "
              << instances.size() << " instances ran, " << executions
              << " pinned executions checked against the CPU reference\n";
}

INSTANTIATE_TEST_SUITE_P(AotCatalog,
                         IntegrationGpuGfx1250ConvFwdAot,
                         ::testing::ValuesIn(graphVariants()),
                         [](const auto& info) { return info.param.name; });

INSTANTIATE_TEST_SUITE_P(AotInstances,
                         IntegrationGpuGfx1250ConvFwdAotCatalog,
                         ::testing::Values(FwdVariant{"Fp16", DataType::HALF, std::nullopt},
                                           FwdVariant{"Bf16", DataType::BFLOAT16, std::nullopt}),
                         [](const auto& info) { return info.param.name; });

} // namespace hip_kernel_provider::kernel_ingestor_engine::integration

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HKP_TEST_GFX1250_CONV_FWD_AOT
