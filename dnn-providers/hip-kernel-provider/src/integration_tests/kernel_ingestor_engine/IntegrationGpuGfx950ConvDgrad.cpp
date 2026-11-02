// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HKP_TEST_GFX950_CONV_DGRAD_AOT)

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
#include <sstream>
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
#include <hipdnn_frontend/attributes/ConvolutionDgradAttributes.hpp>
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
#include "ScopedPluginLogCapture.hpp"

// ---- hipkernel:Gfx950ConvDgradAot: the shipped dgrad catalog on the GPU ----
//
// The engine ships a curated list of rocKE dgrad sweep kernels
// (IngestorGenerator/configs/gfx950_conv_dgrad_aot.labels.txt): per dtype implicit-GEMM
// instances (tilde, dY-halo, tap-outer K and patch forms) and every direct dgrad pipeline
// (kernel_family 1: streaming and windowed depthwise, fused, 4c, and the MFMA pipelines with
// baked and runtime channels), plus the MFMA helper kernels the engine never offers as
// candidates. Each kernel is built once against rocKE's probe problem and matched to a
// graph by its launch plan.
//
// The UED knobs make every per-graph candidate tuple unique, so pinning all of a kernel's
// knobs selects exactly that kernel. The catalog suite pins every selectable instance of a
// dtype on every graph below whose capability it could serve, executes each one the
// engine accepts with dX poisoned (NaN) beforehand, checks it against
// CpuReferenceGraphExecutor, and requires every instance to have run at least once.
// The per-graph suite runs an unbenchmarked plan (whole-catalog and per-family counts) and
// a benchmark under HIPDNN_INGESTOR_CANDIDATE_LIMIT.

namespace hip_kernel_provider::kernel_ingestor_engine::integration
{

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_test_sdk::utilities;
namespace ingestor = hipdnn_plugin_sdk::ingestor;

namespace
{

constexpr const char* ENGINE_NAME = "hipkernel:Gfx950ConvDgradAot";
constexpr const char* DESCRIPTOR_PREFIX = "gfx950_conv_dgrad_aot";
constexpr const char* KERNEL_FAMILY_KNOB = "kernel_family";
constexpr const char* DIRECT_VARIANT_KNOB = "direct_variant";
constexpr int64_t IMPLICIT_GEMM_FAMILY = 0;
constexpr int64_t DIRECT_FAMILY = 1;
constexpr int64_t HELPER_FAMILY = 2;
/// Every direct_variant a selectable kernel carries, "none" being implicit GEMM; the order
/// of DgradGraph's counts.
constexpr std::array<const char*, 7> VARIANTS{"none",
                                              "depthwise_dgrad",
                                              "depthwise_dgrad_win",
                                              "grouped_dgrad_fused",
                                              "grouped_dgrad_4c",
                                              "grouped_dgrad_mfma",
                                              "dgrad_mfma_rtc"};
/// The single-kernel direct pipelines, which the fallback score ranks first.
constexpr std::array<const char*, 4> SINGLE_KERNEL_VARIANTS{
    "depthwise_dgrad", "depthwise_dgrad_win", "grouped_dgrad_fused", "grouped_dgrad_4c"};
/// The MFMA pipelines, which launch weight transforms before their compute pass.
constexpr std::array<const char*, 2> MFMA_VARIANTS{"grouped_dgrad_mfma", "dgrad_mfma_rtc"};
constexpr size_t BENCHMARK_LIMIT = 64;

template <size_t Count>
bool among(const std::array<const char*, Count>& names, const std::string& name)
{
    return std::find_if(
               names.begin(), names.end(), [&](const char* entry) { return name == entry; })
           != names.end();
}

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

constexpr int64_t DY_UID = 1;
constexpr int64_t W_UID = 2;
constexpr int64_t DX_UID = 3;

struct DgradProblem
{
    int64_t n = 2;
    int64_t c = 32;
    int64_t k = 32;
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
    // hipDNN carries no group attribute: groups = dX channels / W dims[1].
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

/// One graph and how many catalog instances of each direct_variant (VARIANTS order) serve
/// it in each dtype, as gfx950_conv_dgrad_aot_plan in
/// rocke/library/builders/common/convolution_backward_data.py counts them over
/// gfx950_conv_dgrad_aot_{fp16,bf16}.instances.json. The native kernel matcher is checked
/// against these counts.
struct DgradGraph
{
    std::string name;
    DgradProblem problem;
    std::array<size_t, VARIANTS.size()> fp16;
    std::array<size_t, VARIANTS.size()> bf16;
    /// The graph has an MFMA tuple whose every catalog row uses the reorganize helper
    /// (waves_k > 1) at groups > 1, which the engine must refuse: rocKE's reorganize
    /// kernel mis-addresses every group but the first.
    bool refusesCoalescedMfma = false;

    const std::array<size_t, VARIANTS.size()>& counts(DataType dtype) const
    {
        return dtype == DataType::HALF ? fp16 : bf16;
    }
    size_t count(DataType dtype, const std::string& variant) const
    {
        for(size_t i = 0; i < VARIANTS.size(); ++i)
        {
            if(variant == VARIANTS[i])
            {
                return counts(dtype)[i];
            }
        }
        return 0;
    }
    template <size_t Count>
    size_t count(DataType dtype, const std::array<const char*, Count>& variants) const
    {
        size_t total = 0;
        for(const auto* variant : variants)
        {
            total += count(dtype, variant);
        }
        return total;
    }
    size_t total(DataType dtype) const
    {
        return count(dtype, VARIANTS);
    }
};

// N is 2 wherever the kernel's grid allows it, so a write past the last dX row of an image
// lands in the next one and fails. Odd spatial sizes leave partial tiles. Together the
// graphs reach every selectable instance of the catalog.
// clang-format off
const std::vector<DgradGraph>& dgradGraphs()
{
    static const std::vector<DgradGraph> s_graphs{
        // Implicit GEMM: unit stride (one sub-GEMM), strides 2, 3x2 and 4 (several tilde
        // sub-GEMMs), dilation, a 1x1 strided filter whose sub-GEMMs leave dX holes (the
        // zero-fill), a nonsquare filter, patch embeddings (a folded width, a stride past
        // the filter), the dY-halo widths, grouped and strided grouped.
        {"Dense", {1, 32, 32, 10, 9, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {160, 0, 0, 6, 0, 0, 0}, {160, 0, 0, 6, 0, 0, 0}},
        {"Strided", {2, 32, 48, 15, 13, 3, 3, 2, 2, 1, 1, 1, 1, 1}, {146, 0, 0, 0, 0, 0, 0}, {146, 0, 0, 0, 0, 0, 0}},
        {"Stride3x2Dilation2x1", {1, 16, 24, 17, 13, 3, 3, 3, 2, 2, 1, 2, 1, 1}, {146, 0, 0, 0, 0, 0, 0}, {146, 0, 0, 0, 0, 0, 0}},
        {"PatchEmbedStride8", {2, 16, 32, 24, 16, 8, 8, 8, 8, 0, 0, 1, 1, 1}, {155, 0, 0, 0, 0, 0, 0}, {155, 0, 0, 0, 0, 0, 0}},
        {"PatchEmbedC3Stride4", {2, 3, 64, 32, 32, 4, 4, 4, 4, 0, 0, 1, 1, 1}, {19, 0, 0, 0, 0, 0, 0}, {19, 0, 0, 0, 0, 0, 0}},
        {"PatchEmbedC3Stride16", {2, 3, 64, 64, 64, 16, 16, 16, 16, 0, 0, 1, 1, 1}, {155, 0, 0, 0, 0, 0, 0}, {155, 0, 0, 0, 0, 0, 0}},
        {"PatchEmbedGapStride5", {2, 8, 64, 30, 30, 4, 4, 5, 5, 0, 0, 1, 1, 1}, {155, 0, 0, 0, 0, 0, 0}, {155, 0, 0, 0, 0, 0, 0}},
        {"FoldedStride2", {2, 64, 128, 16, 16, 2, 2, 2, 2, 0, 0, 1, 1, 1}, {155, 0, 0, 0, 0, 0, 0}, {155, 0, 0, 0, 0, 0, 0}},
        {"PointwiseStride2", {2, 32, 64, 14, 13, 1, 1, 2, 2, 0, 0, 1, 1, 1}, {155, 0, 0, 0, 0, 0, 0}, {155, 0, 0, 0, 0, 0, 0}},
        {"Dilated2", {2, 32, 32, 15, 17, 3, 3, 1, 1, 2, 2, 2, 2, 1}, {146, 0, 0, 0, 0, 0, 0}, {146, 0, 0, 0, 0, 0, 0}},
        {"Stride4Pad3Filter7", {1, 16, 16, 23, 21, 7, 7, 4, 4, 3, 3, 1, 1, 1}, {146, 0, 0, 0, 0, 0, 0}, {146, 0, 0, 0, 0, 0, 0}},
        {"NonsquareFilterStride1x2", {2, 24, 40, 11, 17, 3, 5, 1, 2, 1, 2, 1, 1, 1}, {146, 0, 0, 0, 0, 0, 0}, {146, 0, 0, 0, 0, 0, 0}},
        {"HaloWo32", {2, 128, 128, 12, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {229, 0, 0, 0, 0, 0, 0}, {229, 0, 0, 0, 0, 0, 0}},
        {"HaloWo8", {2, 128, 128, 12, 8, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {237, 0, 0, 0, 0, 0, 0}, {237, 0, 0, 0, 0, 0, 0}},
        {"HaloWo16", {2, 128, 128, 12, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {234, 0, 0, 0, 0, 0, 0}, {234, 0, 0, 0, 0, 0, 0}},
        {"HaloFilter5Wo48", {1, 128, 128, 10, 48, 5, 5, 1, 1, 2, 2, 1, 1, 1}, {196, 0, 0, 0, 0, 0, 0}, {196, 0, 0, 0, 0, 0, 0}},
        {"Filter1x7", {2, 128, 128, 9, 17, 1, 7, 1, 1, 0, 3, 1, 1, 1}, {195, 0, 0, 0, 0, 0, 0}, {195, 0, 0, 0, 0, 0, 0}},
        // Channel counts that admit only narrower vector widths.
        {"GroupedG4", {2, 64, 96, 13, 13, 3, 3, 1, 1, 1, 1, 1, 1, 4}, {146, 0, 0, 0, 0, 0, 0}, {146, 0, 0, 0, 0, 0, 0}},
        {"GroupedG3Strided", {2, 48, 96, 15, 15, 3, 3, 2, 2, 1, 1, 1, 1, 3}, {146, 0, 0, 0, 0, 0, 0}, {146, 0, 0, 0, 0, 0, 0}},
        {"GroupedG4Cpg3", {2, 12, 32, 11, 11, 3, 3, 1, 1, 1, 1, 1, 1, 4}, {17, 0, 0, 0, 0, 0, 0}, {17, 0, 0, 0, 0, 0, 0}},
        {"C12K24", {2, 12, 24, 13, 13, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {17, 0, 0, 0, 0, 0, 0}, {17, 0, 0, 0, 0, 0, 0}},
        {"C6K8Strided", {2, 6, 8, 13, 12, 3, 3, 2, 2, 1, 1, 1, 1, 1}, {17, 0, 0, 0, 0, 0, 0}, {17, 0, 0, 0, 0, 0, 0}},
        {"C3K16", {2, 3, 16, 12, 12, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {17, 0, 0, 0, 0, 0, 0}, {17, 0, 0, 0, 0, 0, 0}},
        {"C16K20", {2, 16, 20, 11, 13, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {16, 0, 0, 0, 0, 0, 0}, {16, 0, 0, 0, 0, 0, 0}},
        {"C16K5", {2, 16, 5, 11, 11, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {9, 0, 0, 0, 0, 0, 0}, {9, 0, 0, 0, 0, 0, 0}},
        // Depthwise (C = K = groups): the streaming and windowed kernels' filters, strides
        // and paddings. 96 and 200 groups leave a partial 64-lane channel tile; odd and even
        // H under stride 2.
        {"DepthwiseG96", {2, 96, 96, 14, 15, 3, 3, 1, 1, 1, 1, 1, 1, 96}, {0, 2, 19, 0, 0, 0, 0}, {0, 2, 19, 0, 0, 0, 0}},
        {"DepthwiseG200N1", {1, 200, 200, 9, 7, 3, 3, 1, 1, 1, 1, 1, 1, 200}, {0, 2, 19, 0, 0, 0, 0}, {0, 2, 19, 0, 0, 0, 0}},
        {"DepthwiseStridedG40OddH", {2, 40, 40, 17, 15, 3, 3, 2, 2, 1, 1, 1, 1, 40}, {0, 7, 0, 0, 0, 0, 0}, {0, 7, 0, 0, 0, 0, 0}},
        {"DepthwiseStridedG8EvenH", {2, 8, 8, 16, 18, 3, 3, 2, 2, 1, 1, 1, 1, 8}, {0, 7, 0, 0, 0, 0, 0}, {0, 7, 0, 0, 0, 0, 0}},
        {"Depthwise5x5G64", {3, 64, 64, 13, 11, 5, 5, 1, 1, 2, 2, 1, 1, 64}, {0, 0, 6, 0, 0, 0, 0}, {0, 0, 6, 0, 0, 0, 0}},
        {"Depthwise5x5StridedG32", {2, 32, 32, 15, 15, 5, 5, 2, 2, 2, 2, 1, 1, 32}, {0, 3, 0, 0, 0, 0, 0}, {0, 3, 0, 0, 0, 0, 0}},
        {"Depthwise7x7Pad0G64", {2, 64, 64, 16, 16, 7, 7, 1, 1, 0, 0, 1, 1, 64}, {0, 0, 2, 0, 0, 0, 0}, {0, 0, 2, 0, 0, 0, 0}},
        {"Depthwise7x7G72", {2, 72, 72, 13, 11, 7, 7, 1, 1, 3, 3, 1, 1, 72}, {0, 0, 10, 0, 0, 0, 0}, {0, 0, 10, 0, 0, 0, 0}},
        // The fused and 4c pipelines, 3x3 p1 s1 at each (C/groups, K/groups) they bake,
        // with N * ceil(Ho / block_h) at least the eight row bands an xcd_tiles row needs.
        {"FusedC4G8", {8, 32, 32, 16, 15, 3, 3, 1, 1, 1, 1, 1, 1, 8}, {0, 0, 0, 1, 0, 0, 0}, {0, 0, 0, 1, 0, 0, 0}},
        {"FusedC8G8", {8, 64, 64, 16, 13, 3, 3, 1, 1, 1, 1, 1, 1, 8}, {146, 0, 0, 3, 0, 0, 0}, {146, 0, 0, 3, 0, 0, 0}},
        {"FusedC16G4", {8, 64, 64, 14, 16, 3, 3, 1, 1, 1, 1, 1, 1, 4}, {146, 0, 0, 4, 0, 0, 0}, {146, 0, 0, 4, 0, 0, 0}},
        {"FusedC24Dense", {8, 24, 24, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {146, 0, 0, 4, 0, 0, 0}, {146, 0, 0, 4, 0, 0, 0}},
        {"FusedC32G2", {8, 64, 64, 13, 16, 3, 3, 1, 1, 1, 1, 1, 1, 2}, {160, 0, 0, 6, 0, 0, 0}, {160, 0, 0, 6, 0, 0, 0}},
        {"FourCG64", {2, 256, 256, 14, 13, 3, 3, 1, 1, 1, 1, 1, 1, 64}, {0, 0, 0, 0, 2, 0, 0}, {0, 0, 0, 0, 2, 0, 0}},
        // The MFMA pipelines: the dense c4k320 rows (coalesced weights, so groups == 1
        // only) and the runtime-channel row.
        {"MfmaC4K320Dense", {1, 4, 320, 14, 13, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {24, 0, 0, 0, 0, 3, 1}, {24, 0, 0, 0, 0, 3, 1}},
        {"MfmaRtcC12K320Dense", {1, 12, 320, 14, 13, 3, 3, 1, 1, 1, 1, 1, 1, 1}, {24, 0, 0, 0, 0, 0, 1}, {24, 0, 0, 0, 0, 0, 1}},
        {"MfmaC4K320G2Refused", {1, 8, 640, 10, 11, 3, 3, 1, 1, 1, 1, 1, 1, 2}, {24, 0, 0, 0, 0, 0, 0}, {24, 0, 0, 0, 0, 0, 0}, true},
    };
    return s_graphs;
}
// clang-format on

struct DgradVariant
{
    std::string name;
    DataType dtype;
    /// Index into dgradGraphs(); the catalog suite runs every graph and leaves it empty.
    std::optional<size_t> graph;
};

// gtest prints a parameter on failure; the name says everything about it.
void PrintTo(const DgradVariant& variant, std::ostream* stream)
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

std::vector<DgradVariant> graphVariants()
{
    std::vector<DgradVariant> variants;
    for(size_t index = 0; index < dgradGraphs().size(); ++index)
    {
        for(const auto dtype : {DataType::HALF, DataType::BFLOAT16})
        {
            variants.push_back({dgradGraphs()[index].name + dtypeSuffix(dtype), dtype, index});
        }
    }
    return variants;
}

std::shared_ptr<Graph> buildDgradGraph(DataType dtype, const DgradProblem& problem)
{
    auto graph = std::make_shared<Graph>();
    graph->set_name("gfx950_conv_dgrad")
        .set_io_data_type(dtype)
        .set_intermediate_data_type(dtype)
        .set_compute_data_type(DataType::FLOAT);

    const auto ho = problem.ho();
    const auto wo = problem.wo();
    // dY [N, K, Ho, Wo] stored NHWK.
    auto dy = std::make_shared<TensorAttributes>();
    dy->set_uid(DY_UID)
        .set_name("DY")
        .set_dim({problem.n, problem.k, ho, wo})
        .set_stride({problem.k * ho * wo, 1, wo * problem.k, problem.k})
        .set_data_type(dtype);
    // W [K, C/groups, Y, X] stored KYXC over the group's channels.
    const auto cpg = problem.cpg();
    auto w = std::make_shared<TensorAttributes>();
    w->set_uid(W_UID)
        .set_name("W")
        .set_dim({problem.k, cpg, problem.y, problem.x})
        .set_stride({cpg * problem.y * problem.x, 1, problem.x * cpg, cpg})
        .set_data_type(dtype);

    ConvDgradAttributes attributes;
    attributes.set_name("gfx950_conv_dgrad")
        .set_padding({problem.padH, problem.padW})
        .set_stride({problem.strideH, problem.strideW})
        .set_dilation({problem.dilationH, problem.dilationW});
    auto dx = graph->conv_dgrad(dy, w, attributes);
    // dX [N, C, Hi, Wi] stored NHWC.
    dx->set_uid(DX_UID)
        .set_name("DX")
        .set_dim({problem.n, problem.c, problem.hi, problem.wi})
        .set_stride({problem.c * problem.hi * problem.wi, 1, problem.wi * problem.c, problem.c})
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
    return loggedKernel(recorder, "' selected kernel ");
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

/// One kernel of the shipped dgrad bundle: what the test needs to pin it and to know which
/// graphs it could serve.
struct CatalogKernel
{
    std::string id;
    std::string name;
    std::string dtype;
    int64_t family = 0;
    std::string variant;
    int64_t filterH = 0;
    int64_t filterW = 0;
    int64_t strideH = 0;
    int64_t strideW = 0;
    int64_t padH = 0;
    int64_t padW = 0;
    int64_t cpg = 0;
    int64_t kpg = 0;
    int64_t wavesK = 0;
    int64_t runtimeKLoop = 0;
    /// Every UED knob, at the kernel's value (an ordinal for a string field).
    std::vector<KnobSetting> knobs;

    bool isMfma() const
    {
        return family == DIRECT_FAMILY && among(MFMA_VARIANTS, variant);
    }
    bool isSingleKernelDirect() const
    {
        return family == DIRECT_FAMILY && among(SINGLE_KERNEL_VARIANTS, variant);
    }
    bool usesCoalescedWeights() const
    {
        return isMfma() && (wavesK > 1 || runtimeKLoop != 0);
    }

    /// Whether the kernel's pinned build-time capability is the graph's. A direct kernel
    /// bakes its whole geometry (0 channel counts are runtime); an implicit GEMM kernel at
    /// most its filter and stride (0 is runtime), and the engine judges the rest. Kernels
    /// that differ only in baked geometry share every knob, so pinning one on a graph it
    /// cannot serve would select its sibling.
    bool capabilityAdmits(const DgradProblem& problem) const
    {
        if(family == IMPLICIT_GEMM_FAMILY)
        {
            const auto admits
                = [](int64_t baked, int64_t actual) { return baked == 0 || baked == actual; };
            return admits(filterH, problem.y) && admits(filterW, problem.x)
                   && admits(strideH, problem.strideH) && admits(strideW, problem.strideW);
        }
        const bool geometry = filterH == problem.y && filterW == problem.x
                              && strideH == problem.strideH && strideW == problem.strideW
                              && padH == problem.padH && padW == problem.padW
                              && problem.dilationH == 1 && problem.dilationW == 1;
        return geometry && (cpg == 0 || cpg == problem.cpg()) && (kpg == 0 || kpg == problem.kpg());
    }
};

struct DgradCatalog
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

/// The shipped dgrad bundle under HIPDNN_DESCRIPTOR_RUNTIME_DIR, read once. Empty when it
/// is not there.
const DgradCatalog& shippedCatalog()
{
    static const DgradCatalog s_catalog = [] {
        DgradCatalog catalog;
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
            kernel.family = metadata.at("kernel_family").get<int64_t>();
            kernel.variant = metadata.at("direct_variant").get<std::string>();
            kernel.filterH = metadata.at("filter_h").get<int64_t>();
            kernel.filterW = metadata.at("filter_w").get<int64_t>();
            kernel.strideH = metadata.at("stride_h").get<int64_t>();
            kernel.strideW = metadata.at("stride_w").get<int64_t>();
            kernel.padH = metadata.at("pad_h").get<int64_t>();
            kernel.padW = metadata.at("pad_w").get<int64_t>();
            kernel.cpg = metadata.at("cpg").get<int64_t>();
            kernel.kpg = metadata.at("kpg").get<int64_t>();
            kernel.wavesK = metadata.at("waves_k").get<int64_t>();
            kernel.runtimeKLoop = metadata.at("runtime_k_loop").get<int64_t>();
            for(const auto& knob : ued.at("knobs"))
            {
                const auto name = knob.get<std::string>();
                const auto& value = metadata.at(name);
                if(value.is_string())
                {
                    kernel.knobs.emplace_back(
                        name, catalog.ordinals.at(name).at(value.get<std::string>()));
                }
                else
                {
                    kernel.knobs.emplace_back(name, value.get<int64_t>());
                }
            }
            catalog.kernels.push_back(std::move(kernel));
        }
        return catalog;
    }();
    return s_catalog;
}

/// dY and W randomized once, dX computed once on the host; the GPU side is reused by every
/// plan of the graph, with dX poisoned before each execution.
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

class IntegrationGpuGfx950ConvDgrad
    : public test_utilities::IntegrationGraphVerificationHarness<float, DgradVariant>
{
protected:
    using Base = test_utilities::IntegrationGraphVerificationHarness<float, DgradVariant>;

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
               _arch, "gfx950", hipdnn_plugin_sdk::ArchMatchMode::PREFIX))
        {
            GTEST_SKIP() << "The packaged convolution family requires gfx950";
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

    int64_t engineId() const
    {
        return hipdnn_data_sdk::utilities::engineNameToId(ENGINE_NAME);
    }

    const DgradGraph& graphOf(const DgradVariant& variant) const
    {
        return dgradGraphs().at(variant.graph.value());
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

    Reference makeReference(const DgradVariant& variant, const DgradProblem& problem, unsigned seed)
    {
        // Any built graph of the problem has the same tensors as every plan's graph.
        auto graph = buildDgradGraph(variant.dtype, problem);
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
        for(const auto uid : {DY_UID, W_UID})
        {
            const auto tensorSeed = seed + static_cast<unsigned int>(uid);
            reference.gpu->randomizeTensor(uid, -1.0F, 1.0F, tensorSeed);
            reference.cpu->randomizeTensor(uid, -1.0F, 1.0F, tensorSeed);
        }
        reference.cpu->getTensor(DX_UID).fillWithSentinelValue();
        CpuReferenceGraphExecutor().execute(reference.serialized.data(),
                                            reference.serialized.size(),
                                            reference.cpu->toHostVariantPack());
        return reference;
    }

    /// Executes @p graph's plan with dX poisoned to NaN and compares dX with the reference.
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
        // Every dX element the kernel does not write stays NaN and fails the comparison.
        reference.gpu->getTensor(DX_UID).fillWithSentinelValue();
        auto buffers = reference.gpu->toDeviceVariantPack();
        result = graph.execute(_handle, buffers, workspaceSize > 0 ? _workspace : nullptr);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        ASSERT_EQ(hipStreamSynchronize(_stream), hipSuccess);
        reference.gpu->getTensor(DX_UID).markDeviceModified();

        // FP32 accumulation followed by a low-precision store; inputs bounded to [-1,1].
        const float relativeTolerance = dtype == DataType::HALF ? 0.002F : 0.016F;
        const auto validator
            = createAllCloseValidator(frontendToSdkDataType(dtype), 0.001F, relativeTolerance);
        EXPECT_TRUE(validator->allClose(reference.cpu->getTensor(DX_UID),
                                        reference.gpu->getTensor(DX_UID)));
    }

    /// Pins every knob of @p kernel on a fresh graph of @p problem. SERVED: the engine
    /// selected exactly that kernel and its dX matched the reference; REFUSED: no candidate
    /// satisfied the knobs; FAILED: anything else (recorded as a test failure).
    PinOutcome pinAndVerify(const DgradVariant& variant,
                            const DgradProblem& problem,
                            const CatalogKernel& kernel,
                            Reference& reference,
                            IsolatedLogRecorder& recorder,
                            size_t expectedBeforeFiltering)
    {
        recorder.clearLogs();
        auto graph = buildDgradGraph(variant.dtype, problem);
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
        if(selectedKernel(recorder) != kernel.id || !recorder.hasLogContaining(expected))
        {
            ADD_FAILURE() << kernel.name << " pinned, but expected '" << kernel.id << "' "
                          << expected << ": " << recorder.getRecordedLogsAsString();
            return PinOutcome::FAILED;
        }
        const auto failuresBefore
            = ::testing::UnitTest::GetInstance()->current_test_info()->result()->total_part_count();
        executeAndVerify(*graph, variant.dtype, reference);
        const auto* result = ::testing::UnitTest::GetInstance()->current_test_info()->result();
        for(int part = failuresBefore; part < result->total_part_count(); ++part)
        {
            if(result->GetTestPartResult(part).failed())
            {
                ADD_FAILURE() << "wrong dX from " << kernel.name << " (" << kernel.id << ")";
                return PinOutcome::FAILED;
            }
        }
        return PinOutcome::SERVED;
    }

    // ---- Winner-record plumbing for the benchmarked plan, as the forward suite ----

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

    ScopedTestCacheDir _cacheDir{"gfx950-conv-dgrad", ScopedTestCacheDir::Scope::TEST};
    ScopedEnvironmentVariableSetter _forceBenchmarking{"HIPDNN_FORCE_BENCHMARKING", ""};
    ScopedEnvironmentVariableSetter _cacheEnabled{"HIPDNN_DISABLE_CACHE", ""};
    ScopedEnvironmentVariableSetter _candidateLimit{
        hipdnn_plugin_sdk::INGESTOR_CANDIDATE_LIMIT_ENV_NAME, ""};
    std::string _arch;
    void* _workspace = nullptr;
    size_t _workspaceSize = 0;
};

// ---- Per graph: the unbenchmarked plan and a capped benchmark ----

class IntegrationGpuGfx950ConvDgradAot : public IntegrationGpuGfx950ConvDgrad
{
};

// The whole catalog's count, each family's and each pipeline's count through the
// kernel_family and direct_variant knobs, and the family an unbenchmarked plan prefers when
// the fallback score decides: a single-kernel direct pipeline, then an MFMA pipeline, then
// implicit GEMM (planScore).
TEST_P(IntegrationGpuGfx950ConvDgradAot, UnbenchmarkedPlanServesFromTheWholeCatalog)
{
    const auto& variant = GetParam();
    const auto& graphCase = graphOf(variant);
    const auto& catalog = shippedCatalog();
    ASSERT_FALSE(catalog.kernels.empty()) << "the shipped dgrad bundle was not found";
    const auto total = graphCase.total(variant.dtype);
    if(total == 0)
    {
        GTEST_SKIP() << "no catalog instance serves " << graphCase.name;
    }
    const ScopedPluginLogCapture capture(this);
    auto& recorder = capture.recorder();
    auto reference = makeReference(variant, graphCase.problem, 53);

    auto graph = buildDgradGraph(variant.dtype, graphCase.problem);
    ASSERT_NO_FATAL_FAILURE(buildAndCompile(*graph, {benchmarkingKnob(false)}));
    EXPECT_TRUE(recorder.hasLogContaining("at rank 0 from " + candidateCount(total) + " ("
                                          + std::to_string(total) + " before knob filtering)"))
        << recorder.getRecordedLogsAsString();
    const auto* selected = catalog.find(selectedKernel(recorder));
    ASSERT_NE(selected, nullptr) << recorder.getRecordedLogsAsString();
    EXPECT_EQ(selected->dtype, dtypeToken(variant.dtype));
    EXPECT_NE(selected->family, HELPER_FAMILY);
    EXPECT_GT(graphCase.count(variant.dtype, selected->variant), 0U) << selected->name;
    // The family order is the fallback score's (planScore); a trained model installed in the
    // arch root decides by its own ranking, so then only require that the selected kernel
    // comes from a pipeline that serves this graph (above).
    if(recorder.hasLogContaining("decided_by=native"))
    {
        if(graphCase.count(variant.dtype, SINGLE_KERNEL_VARIANTS) > 0)
        {
            EXPECT_TRUE(selected->isSingleKernelDirect()) << selected->name;
        }
        else if(graphCase.count(variant.dtype, MFMA_VARIANTS) > 0)
        {
            EXPECT_TRUE(selected->isMfma()) << selected->name;
        }
        else
        {
            EXPECT_EQ(selected->family, IMPLICIT_GEMM_FAMILY) << selected->name;
        }
    }
    ASSERT_NO_FATAL_FAILURE(executeAndVerify(*graph, variant.dtype, reference));
    graph.reset();

    const auto implicitGemm = graphCase.count(variant.dtype, "none");
    std::vector<std::pair<KnobSetting, size_t>> forced{
        {KnobSetting(KERNEL_FAMILY_KNOB, IMPLICIT_GEMM_FAMILY), implicitGemm},
        {KnobSetting(KERNEL_FAMILY_KNOB, DIRECT_FAMILY), total - implicitGemm},
        // Helpers are never candidates.
        {KnobSetting(KERNEL_FAMILY_KNOB, HELPER_FAMILY), 0}};
    for(size_t i = 1; i < VARIANTS.size(); ++i)
    {
        const auto ordinal = catalog.ordinal(DIRECT_VARIANT_KNOB, VARIANTS[i]);
        ASSERT_TRUE(ordinal) << VARIANTS[i];
        forced.emplace_back(KnobSetting(DIRECT_VARIANT_KNOB, *ordinal),
                            graphCase.counts(variant.dtype)[i]);
    }
    for(const auto& [knob, count] : forced)
    {
        recorder.clearLogs();
        auto pinned = buildDgradGraph(variant.dtype, graphCase.problem);
        const auto error = tryBuild(*pinned, {knob, benchmarkingKnob(false)});
        if(count == 0)
        {
            EXPECT_TRUE(knobsUnsatisfiable(error, recorder)) << knob.knobId() << ": " << error;
            continue;
        }
        ASSERT_TRUE(error.empty()) << knob.knobId() << ": " << error;
        EXPECT_TRUE(recorder.hasLogContaining("at rank 0 from " + candidateCount(count) + " ("
                                              + std::to_string(total) + " before knob filtering)"))
            << knob.knobId() << ": " << recorder.getRecordedLogsAsString();
        ASSERT_NO_FATAL_FAILURE(executeAndVerify(*pinned, variant.dtype, reference));
    }
}

// Benchmarking under HIPDNN_INGESTOR_CANDIDATE_LIMIT measures only the best-ranked
// candidates, and its record serves the next build of the graph.
TEST_P(IntegrationGpuGfx950ConvDgradAot, CappedBenchmarkMatchesCpuReference)
{
    const auto& variant = GetParam();
    const auto& graphCase = graphOf(variant);
    const ScopedPluginLogCapture capture(this);
    auto& recorder = capture.recorder();
    const auto total = graphCase.total(variant.dtype);
    if(total == 0)
    {
        GTEST_SKIP() << "no catalog instance serves " << graphCase.name;
    }
    auto reference = makeReference(variant, graphCase.problem, 71);
    const auto capped = std::min(BENCHMARK_LIMIT, total);
    _candidateLimit.setValue(std::to_string(BENCHMARK_LIMIT));

    auto graph = buildDgradGraph(variant.dtype, graphCase.problem);
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
    auto recreated = buildDgradGraph(variant.dtype, graphCase.problem);
    ASSERT_NO_FATAL_FAILURE(buildAndCompile(*recreated, {benchmarkingKnob(true)}));
    EXPECT_EQ(servedKernel(recorder), winner) << recorder.getRecordedLogsAsString();
    const auto count = std::to_string(capped);
    EXPECT_TRUE(recorder.hasLogContaining("from a benchmarked record of " + count + " entry(s) for "
                                          + count + " candidate(s)"))
        << recorder.getRecordedLogsAsString();
    EXPECT_FALSE(recorder.hasLogContaining("will benchmark"));
    ASSERT_NO_FATAL_FAILURE(executeAndVerify(*recreated, variant.dtype, reference));
}

// ---- Per dtype: every catalog instance pinned and checked ----

class IntegrationGpuGfx950ConvDgradAotCatalog : public IntegrationGpuGfx950ConvDgrad
{
};

// Every selectable instance of the dtype is pinned on every graph whose capability it
// could serve; the engine must accept exactly the counts above, each accepted instance
// must match the reference, and every instance must run at least once. On the graphs
// with groups > 1 the coalesced-weight MFMA rows of the graph's tuple must refuse.
TEST_P(IntegrationGpuGfx950ConvDgradAotCatalog, EveryInstanceMatchesCpuReference)
{
    const auto& variant = GetParam();
    const auto& catalog = shippedCatalog();
    ASSERT_FALSE(catalog.kernels.empty()) << "the shipped dgrad bundle was not found";
    std::vector<const CatalogKernel*> selectable;
    size_t helpers = 0;
    for(const auto& kernel : catalog.kernels)
    {
        if(kernel.dtype != dtypeToken(variant.dtype))
        {
            continue;
        }
        if(kernel.family == HELPER_FAMILY)
        {
            ++helpers;
            continue;
        }
        selectable.push_back(&kernel);
    }
    EXPECT_EQ(selectable.size(), 320U);
    EXPECT_EQ(helpers, 4U);

    const ScopedPluginLogCapture capture(this);
    auto& recorder = capture.recorder();
    std::set<std::string> covered;
    size_t executions = 0;
    unsigned int seed = 101;
    for(const auto& graphCase : dgradGraphs())
    {
        SCOPED_TRACE(graphCase.name);
        auto reference = makeReference(variant, graphCase.problem, seed++);
        std::map<std::string, size_t> served;
        size_t coalescedAttempted = 0;
        for(const auto* kernel : selectable)
        {
            if(!kernel->capabilityAdmits(graphCase.problem))
            {
                continue;
            }
            if(kernel->usesCoalescedWeights() && graphCase.problem.groups > 1)
            {
                ++coalescedAttempted;
            }
            const auto outcome = pinAndVerify(variant,
                                              graphCase.problem,
                                              *kernel,
                                              reference,
                                              recorder,
                                              graphCase.total(variant.dtype));
            if(outcome != PinOutcome::SERVED)
            {
                continue;
            }
            ++executions;
            covered.insert(kernel->id);
            ++served[kernel->variant];
        }
        for(const auto* name : VARIANTS)
        {
            EXPECT_EQ(served[name], graphCase.count(variant.dtype, name)) << name;
        }
        if(graphCase.refusesCoalescedMfma)
        {
            EXPECT_GT(coalescedAttempted, 0U) << "no coalesced MFMA row has this tuple";
        }
    }

    std::vector<std::string> missing;
    for(const auto* kernel : selectable)
    {
        if(covered.count(kernel->id) == 0)
        {
            missing.push_back(kernel->name);
        }
    }
    EXPECT_TRUE(missing.empty()) << missing.size() << " instance(s) never ran, first "
                                 << (missing.empty() ? "" : missing.front());
    std::cout << "[ dgrad    ] " << variant.name << ": " << covered.size() << " of "
              << selectable.size() << " instances ran, " << executions
              << " pinned executions checked against the CPU reference\n";
}

INSTANTIATE_TEST_SUITE_P(AotCatalog,
                         IntegrationGpuGfx950ConvDgradAot,
                         ::testing::ValuesIn(graphVariants()),
                         [](const auto& info) { return info.param.name; });

INSTANTIATE_TEST_SUITE_P(AotInstances,
                         IntegrationGpuGfx950ConvDgradAotCatalog,
                         ::testing::Values(DgradVariant{"Fp16", DataType::HALF, std::nullopt},
                                           DgradVariant{"Bf16", DataType::BFLOAT16, std::nullopt}),
                         [](const auto& info) { return info.param.name; });

} // namespace hip_kernel_provider::kernel_ingestor_engine::integration

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HKP_TEST_GFX950_CONV_DGRAD_AOT
