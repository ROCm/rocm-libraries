// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// heuristics/sweep/ConvCandidateSweep.cpp
//
// Standalone candidate sweep for the ck_dsl implicit-GEMM conv path.
//
// For each shape provided via --shapes:
//   1. Enumerate all buildable candidate knob combos via enumerateCandidates.
//   2. Compile + time every candidate on device via CompileServiceBridge + HIP.
//   3. Write one CSV row per successful candidate to --out.
//
// This sweep uses the Python DSL codegen path (CompileServiceBridge) to compile
// and time every candidate from scratch. It is an offline oracle tool; the
// production C++ dispatcher uses pre-compiled .hsaco kernels from ArtifactStore
// and does not share this enumeration path.
//
// CLI arguments (see main.cpp):
//   --shapes <path>   CSV of conv shapes to sweep.
//                     Columns: N,G,C,K,Hi,Wi,Y,X,stride_h,stride_w,pad_h,pad_w[,direction]
//                     G>1 grouped conv is supported; C and K are total across all groups.
//   --out    <path>   Path to write training rows (appended if the file exists).
//                     Columns: N,G,C,K,Hi,Wi,Y,X,stride_h,stride_w,pad_h,pad_w,
//                              tile_m,tile_n,tile_k,pipeline,tflops,latency_us

#include <hip/hip_runtime.h>

#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

#include "CkDslContainer.hpp"
#include "CkDslContext.hpp"
#include "CkDslHandle.hpp"
#include "adapters/conv_implicit_gemm/ConvImplicitGemmAdapter.hpp"
#include "adapters/conv_implicit_gemm/ConvImplicitGemmCandidateSelector.hpp"
#include "adapters/conv_implicit_gemm/ConvImplicitGemmPayload.hpp"
#include "adapters/conv_implicit_gemm/ConvImplicitGemmPerfKnobs.hpp"
#include "adapters/conv_implicit_gemm/ConvImplicitGemmSpec.hpp"
#include "engines/conv_implicit_gemm/ConvImplicitGemmPlan.hpp"
#include "engines/conv_implicit_gemm/ConvImplicitGemmPlanBuilder.hpp"
#include "perf/PerfMeasurement.hpp"
#include "python/CompileServiceBridge.hpp"
#include "runtime/DeviceArch.hpp"
#include "runtime/HipModule.hpp"
#include "runtime/KernelArtifact.hpp"

namespace py = pybind11;

namespace {

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
namespace flatbuf      = hipdnn_flatbuffers_sdk::flatbuffer_utilities;
namespace utilities    = hipdnn_data_sdk::utilities;

using ck_dsl_provider::CkDslContainer;
using ck_dsl_provider::CkDslContext;
using ck_dsl_provider::ConvImplicitGemmPerfKnobs;
using ck_dsl_provider::ConvImplicitGemmPlanBuilder;
using ck_dsl_provider::KernelArtifact;
using ck_dsl_provider::PerfMeasurement;
using ck_dsl_provider::PerfResult;
using ck_dsl_provider::ConvSelectionProblem;
using hipdnn_data_sdk::types::half;

// ── Training CSV writer ───────────────────────────────────────────────────────

static const char* kTrainingCsvHeader =
    "N,G,C,K,Hi,Wi,Y,X,stride_h,stride_w,pad_h,pad_w,"
    "tile_m,tile_n,tile_k,pipeline,tflops,latency_us\n";

struct TrainingCsvWriter {
    std::mutex    mu;
    std::ofstream f;
    bool          active = false;

    void open(const std::string& path) {
        std::lock_guard<std::mutex> lk(mu);
        const bool exists = std::ifstream(path).good();
        f.open(path, std::ios::app);
        if (!f) {
            std::cerr << "[TrainingCSV] ERROR: cannot open " << path << "\n";
            return;
        }
        if (!exists) f << kTrainingCsvHeader;
        active = true;
        std::cerr << "[TrainingCSV] Writing training rows to " << path << "\n";
    }

    void write(std::int64_t N, std::int64_t G, std::int64_t C, std::int64_t K,
               std::int64_t Hi, std::int64_t Wi, std::int64_t Y, std::int64_t X,
               std::int64_t sH, std::int64_t sW, std::int64_t pH, std::int64_t pW,
               const ConvImplicitGemmPerfKnobs& k, double tflops, double latency_us)
    {
        if (!active) return;
        std::lock_guard<std::mutex> lk(mu);
        f << N << ',' << G << ',' << C << ',' << K << ','
          << Hi << ',' << Wi << ',' << Y << ',' << X << ','
          << sH << ',' << sW << ',' << pH << ',' << pW << ','
          << k.tile_m << ',' << k.tile_n << ',' << k.tile_k << ','
          << k.pipeline << ','
          << tflops << ',' << latency_us << '\n';
        f.flush();
    }
};

static TrainingCsvWriter gCsvWriter;

// ── Shape descriptor ─────────────────────────────────────────────────────────

struct ConvCase {
    std::string  name;
    std::int64_t n, g, c, hi, wi;  // g: group count; c,k are totals across all groups
    std::int64_t k, r, s;
    std::int64_t strideH, strideW;
    std::int64_t padH, padW;
    std::int64_t dilH, dilW;
};

std::int64_t convOutDim(std::int64_t in, std::int64_t pad, std::int64_t dil,
                        std::int64_t filt, std::int64_t stride) {
    return (in + 2 * pad - dil * (filt - 1) - 1) / stride + 1;
}

// ── CSV shape loader ──────────────────────────────────────────────────────────

std::vector<ConvCase> loadShapesFromCsv(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        std::cerr << "[ShapeLoader] ERROR: cannot open " << path << "\n";
        return {};
    }
    std::string header;
    if (!std::getline(f, header)) return {};

    // Expected columns: N,G,C,K,Hi,Wi,Y,X,stride_h,stride_w,pad_h,pad_w[,direction]
    // C and K are totals across all groups; G=1 means a standard (non-grouped) conv.
    std::vector<ConvCase> cases;
    std::string line;
    int idx = 0;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string tok;
        std::vector<std::string> fields;
        while (std::getline(ss, tok, ',')) fields.push_back(tok);
        if (fields.size() < 12) continue;
        try {
            ConvCase c{};
            c.name    = "shape_" + std::to_string(idx);
            c.n       = std::stoll(fields[0]);
            c.g       = std::stoll(fields[1]);
            if (c.g < 1) c.g = 1;
            c.c       = std::stoll(fields[2]);
            c.k       = std::stoll(fields[3]);
            c.hi      = std::stoll(fields[4]);
            c.wi      = std::stoll(fields[5]);
            c.r       = std::stoll(fields[6]);
            c.s       = std::stoll(fields[7]);
            c.strideH = std::stoll(fields[8]);
            c.strideW = std::stoll(fields[9]);
            c.padH    = std::stoll(fields[10]);
            c.padW    = std::stoll(fields[11]);
            c.dilH    = 1; c.dilW = 1;
            // Skip shapes where per-group channels are not 8-aligned.
            if (c.c % (c.g * 8) != 0 || c.k % (c.g * 8) != 0) continue;
            cases.push_back(c);
        } catch (...) { continue; }
        ++idx;
    }
    std::cerr << "[ShapeLoader] Loaded " << cases.size()
              << " shapes from " << path << "\n";
    return cases;
}

// ── Knob formatter ────────────────────────────────────────────────────────────

std::string fmtKnobs(const ConvImplicitGemmPerfKnobs& k) {
    std::ostringstream os;
    os << "tile=" << k.tile_m << "x" << k.tile_n << "x" << k.tile_k
       << ",pipe=" << k.pipeline
       << ",blk=" << k.block_size();
    return os.str();
}

// ── Per-candidate compile + time ─────────────────────────────────────────────

std::optional<double> timeCandidate(
        CkDslContainer&                                container,
        ::CkDslHandle&                                 handle,
        const std::string&                             arch,
        const ConvImplicitGemmPerfKnobs&               knobs,
        const flatbuf::GraphWrapper&                   graph,
        const std::vector<hipdnnPluginDeviceBuffer_t>& deviceBuffers,
        double                                         flops,
        double&                                        medianUsOut)
{
    if (graph.nodeCount() == 0) return std::nullopt;
    const auto& node = graph.getNodeWrapper(0);
    if (node.attributesType() != data_objects::NodeAttributes::ConvolutionFwdAttributes)
        return std::nullopt;
    const auto& convAttr = node.attributesAs<data_objects::ConvolutionFwdAttributes>();

    ck_dsl_provider::ConvImplicitGemmSpec spec;
    try {
        spec = ck_dsl_provider::ConvImplicitGemmAdapter::buildSpec(convAttr, graph.getTensorMap());
    } catch (...) { return std::nullopt; }

    spec.tile_m      = knobs.tile_m;
    spec.tile_n      = knobs.tile_n;
    spec.tile_k      = knobs.tile_k;
    spec.warp_m      = knobs.warp_m;
    spec.warp_n      = knobs.warp_n;
    spec.warp_tile_m = knobs.warp_tile_m;
    spec.warp_tile_n = knobs.warp_tile_n;
    spec.warp_tile_k = knobs.warp_tile_k;
    spec.pipeline    = knobs.pipeline;

    KernelArtifact artifact;
    try {
        py::gil_scoped_acquire gil;
        py::dict payload = ck_dsl_provider::convImplicitGemmSpecToPayload(spec);
        artifact = container.compileServiceBridge().compile(
            ConvImplicitGemmPlanBuilder::opKind(), payload, arch);
    } catch (const std::exception& e) {
        std::cerr << "[Sweep] (" << fmtKnobs(knobs) << ") compile FAILED: " << e.what() << "\n";
        return std::nullopt;
    }

    std::shared_ptr<ck_dsl_provider::HipModule> module;
    try {
        module = std::make_shared<ck_dsl_provider::HipModule>(artifact);
    } catch (const std::exception& e) {
        std::cerr << "[Sweep] (" << fmtKnobs(knobs) << ") module load FAILED: " << e.what() << "\n";
        return std::nullopt;
    }

    const std::int64_t xBytes =
        (std::int64_t)spec.problem.N * spec.problem.C * spec.problem.Hi * spec.problem.Wi * 2;
    const std::int64_t wBytes =
        (std::int64_t)spec.problem.K * spec.problem.C * spec.problem.R * spec.problem.S * 2;
    const std::int64_t yBytes =
        (std::int64_t)spec.problem.N * spec.problem.K * spec.problem.Ho() * spec.problem.Wo() * 2;

    ck_dsl_provider::ConvImplicitGemmPlan plan(
        module, /*xUid=*/1, /*wUid=*/2, /*yUid=*/3, xBytes, wBytes, yBytes);

    const std::size_t wsBytes = plan.getWorkspaceSize(handle);
    void* workspace = nullptr;
    if (wsBytes > 0 && hipMalloc(&workspace, wsBytes) != hipSuccess) {
        std::cerr << "[Sweep] (" << fmtKnobs(knobs) << ") workspace alloc FAILED\n";
        return std::nullopt;
    }

    std::optional<double> result;
    try {
        plan.execute(handle, deviceBuffers.data(),
                     static_cast<std::uint32_t>(deviceBuffers.size()), workspace);
        if (hipDeviceSynchronize() != hipSuccess)
            throw std::runtime_error("hipDeviceSynchronize failed");

        PerfMeasurement pm;
        auto launchFn = [&]() {
            plan.execute(handle, deviceBuffers.data(),
                         static_cast<std::uint32_t>(deviceBuffers.size()), workspace);
        };
        PerfResult pr = pm.measure(launchFn, flops, handle.getStream());
        medianUsOut = pr.medianUs;
        result      = pr.tflops;
    } catch (const std::exception& e) {
        std::cerr << "[Sweep] (" << fmtKnobs(knobs) << ") launch FAILED: " << e.what() << "\n";
    }

    if (workspace) (void)hipFree(workspace);
    return result;
}

// ── Per-shape sweep ───────────────────────────────────────────────────────────

bool runConvOracleSweep(const ConvCase& cse, CkDslContainer& container,
                        ::CkDslHandle& handle, const std::string& arch)
{
    const std::int64_t kN  = cse.n, kG = cse.g, kC = cse.c, kHi = cse.hi, kWi = cse.wi;
    const std::int64_t kK  = cse.k, kR = cse.r, kS = cse.s;
    const std::int64_t kHo = convOutDim(kHi, cse.padH, cse.dilH, kR, cse.strideH);
    const std::int64_t kWo = convOutDim(kWi, cse.padW, cse.dilW, kS, cse.strideW);
    // Per-group channel counts used for filter dims and FLOPS.
    const std::int64_t kCpG = kC / kG;

    if (kHo <= 0 || kWo <= 0) {
        std::cerr << "[Sweep] " << cse.name << ": invalid output dims, skipping\n";
        return false;
    }

    // For grouped conv the filter is [K, C/G, R, S] in NHWC order.
    auto fbBuilder = hipdnn_test_sdk::utilities::createValidConvFwdGraph(
        {kN, kC, kHi, kWi}, {kC * kHi * kWi, 1, kWi * kC, kC},
        {kK, kCpG, kR, kS},  {kCpG * kR * kS, 1, kS * kCpG, kCpG},
        {kN, kK, kHo, kWo}, {kK * kHo * kWo, 1, kWo * kK, kK},
        {cse.padH, cse.padW}, {cse.padH, cse.padW},
        {cse.strideH, cse.strideW}, {cse.dilH, cse.dilW},
        data_objects::DataType::HALF);
    flatbuf::GraphWrapper graph(fbBuilder.GetBufferPointer(), fbBuilder.GetSize());

    const utilities::TensorLayout& nhwc = utilities::TensorLayout::NHWC;
    utilities::Tensor<half> tensorX({kN, kC, kHi, kWi}, nhwc);
    utilities::Tensor<half> tensorW({kK, kCpG, kR, kS},
        utilities::generateStrides({kK, kCpG, kR, kS}, nhwc.strideOrder));
    utilities::Tensor<half> tensorY({kN, kK, kHo, kWo}, nhwc);
    tensorX.fillWithRandomValues(half(-0.1f), half(0.1f), 0x4242u);
    tensorW.fillWithRandomValues(half(-0.1f), half(0.1f), 0x5555u);

    const std::vector<hipdnnPluginDeviceBuffer_t> deviceBuffers = {
        {1, tensorX.memory().deviceData()},
        {2, tensorW.memory().deviceData()},
        {3, tensorY.memory().deviceData()},
    };

    // FLOPS: each of the K output channels applies a [C/G, R, S] filter over [N, Ho, Wo].
    const double kFlops = 2.0 * static_cast<double>(kN) * static_cast<double>(kHo)
                              * static_cast<double>(kWo) * static_cast<double>(kK)
                              * static_cast<double>(kCpG) * static_cast<double>(kR)
                              * static_cast<double>(kS);

    ConvSelectionProblem selProblem;
    selProblem.N  = static_cast<std::int32_t>(kN);
    selProblem.C  = static_cast<std::int32_t>(kC);
    selProblem.K  = static_cast<std::int32_t>(kK);
    selProblem.G  = static_cast<std::int32_t>(kG);
    selProblem.Hi = static_cast<std::int32_t>(kHi);
    selProblem.Wi = static_cast<std::int32_t>(kWi);
    selProblem.R  = static_cast<std::int32_t>(kR);
    selProblem.S  = static_cast<std::int32_t>(kS);
    selProblem.sH = static_cast<std::int32_t>(cse.strideH);
    selProblem.sW = static_cast<std::int32_t>(cse.strideW);
    selProblem.pH = static_cast<std::int32_t>(cse.padH);
    selProblem.pW = static_cast<std::int32_t>(cse.padW);
    selProblem.dH = static_cast<std::int32_t>(cse.dilH);
    selProblem.dW = static_cast<std::int32_t>(cse.dilW);
    selProblem.dtype = "fp16";

    const std::vector<ConvImplicitGemmPerfKnobs> candidates =
        ck_dsl_provider::enumerateCandidates(selProblem, arch);

    if (candidates.empty()) {
        std::cerr << "[Sweep] " << cse.name << ": enumerateCandidates returned no candidates, skipping\n";
        return false;
    }

    std::cerr << "[Sweep] " << cse.name
              << " (G=" << kG << ") sweeping " << candidates.size() << " candidates\n";

    std::size_t sweptOk = 0, sweptFail = 0;
    for (const auto& cand : candidates) {
        double us = 0.0;
        auto tflops = timeCandidate(container, handle, arch, cand, graph,
                                    deviceBuffers, kFlops, us);
        if (!tflops) { ++sweptFail; continue; }
        ++sweptOk;
        gCsvWriter.write(kN, kG, kC, kK, kHi, kWi, kR, kS,
                         cse.strideH, cse.strideW, cse.padH, cse.padW,
                         cand, *tflops, us);
    }

    std::cerr << "[Sweep] " << cse.name
              << " done: " << sweptOk << " ok, " << sweptFail << " failed\n";
    return sweptOk > 0;
}

}  // namespace

// ── Entry point ───────────────────────────────────────────────────────────────

int sweepMain(const std::string& shapesPath, const std::string& outPath) {
    const std::vector<ConvCase> shapes = loadShapesFromCsv(shapesPath);
    if (shapes.empty()) {
        std::cerr << "ERROR: no shapes loaded from " << shapesPath << "\n";
        return 1;
    }

    gCsvWriter.open(outPath);
    if (!gCsvWriter.active) return 1;

    CkDslContainer container;
    ::CkDslHandle handle;
    const auto arch = ck_dsl_provider::detectDeviceArch(handle.getStream());
    if (!arch.has_value()) {
        std::cerr << "ERROR: could not detect device arch\n";
        return 1;
    }

    std::size_t shapesOk = 0, shapesFailed = 0;
    for (const auto& shape : shapes) {
        if (runConvOracleSweep(shape, container, handle, *arch))
            ++shapesOk;
        else
            ++shapesFailed;
    }

    std::cerr << "=== Sweep complete: " << shapesOk << "/" << shapes.size()
              << " shapes produced data (" << shapesFailed << " failed) ===\n";
    return shapesFailed > 0 ? 1 : 0;
}
