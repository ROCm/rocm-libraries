// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// heuristics/sweep/ConvCandidateSweep.cpp
//
// Standalone candidate sweep for the ck_dsl implicit-GEMM conv path.
//
// For each shape provided via --shapes:
//   1. Enumerate all (tile_m, tile_n, tile_k, pipeline) candidates.
//   2. JIT-compile each via CEngine::build_conv (pure C engine, no Python).
//   3. Time each candidate on device and write one CSV row per success.
//
// Uses the same CEngine / Kernel infrastructure as the production C-JIT path.
// No pybind11, no CompileServiceBridge, no ArtifactStore dependency.
//
// CLI arguments (see main.cpp):
//   --shapes <path>   CSV of conv shapes to sweep.
//                     Columns: N,G,C,K,Hi,Wi,Y,X,stride_h,stride_w,pad_h,pad_w[,direction]
//                     G>1 grouped conv is supported; C and K are totals across all groups.
//   --out    <path>   Path to write training rows (appended if the file exists).
//                     Columns: N,G,C,K,Hi,Wi,Y,X,stride_h,stride_w,pad_h,pad_w,
//                              tile_m,tile_n,tile_k,pipeline,tflops,latency_us

#include <hip/hip_runtime.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "ck_dsl_runtime/c_engine.hpp"
#include "ck_dsl_runtime/kernel.hpp"

namespace {

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
               int tile_m, int tile_n, int tile_k, const std::string& pipeline,
               double tflops, double latency_us) {
        if (!active) return;
        std::lock_guard<std::mutex> lk(mu);
        f << N << ',' << G << ',' << C << ',' << K << ','
          << Hi << ',' << Wi << ',' << Y << ',' << X << ','
          << sH << ',' << sW << ',' << pH << ',' << pW << ','
          << tile_m << ',' << tile_n << ',' << tile_k << ','
          << pipeline << ','
          << tflops << ',' << latency_us << '\n';
        f.flush();
    }
};

static TrainingCsvWriter gCsvWriter;

// ── Shape descriptor ─────────────────────────────────────────────────────────

struct ConvCase {
    std::string  name;
    std::int64_t n, g, c, hi, wi;
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
            if (c.c % (c.g * 8) != 0 || c.k % (c.g * 8) != 0) continue;
            cases.push_back(c);
        } catch (...) { continue; }
        ++idx;
    }
    std::cerr << "[ShapeLoader] Loaded " << cases.size()
              << " shapes from " << path << "\n";
    return cases;
}

// ── Candidate enumeration ─────────────────────────────────────────────────────
//
// The ck_dsl conv implicit-GEMM kernel is parameterized by (tile_m, tile_n,
// tile_k, pipeline). Warp geometry is derived from the tile as per the C engine
// defaults (2x2 warps, 32x32x16 atom). Candidates are filtered by:
//   - M = N*Ho*Wo must be >= tile_m (grid must be non-empty)
//   - K must be >= tile_n
//   - C/G*R*S must be >= tile_k (k-loop must have at least one iteration)

struct Candidate {
    int tile_m, tile_n, tile_k;
    std::string pipeline;
};

std::vector<Candidate> enumerateCandidates(const ConvCase& cse, std::int64_t Ho, std::int64_t Wo) {
    static const int kTileSizes[] = {32, 64, 128};
    static const char* kPipelines[] = {"mem", "compv3", "compv4"};

    const std::int64_t M   = cse.n * Ho * Wo;
    const std::int64_t N   = cse.k;
    const std::int64_t Kgm = (cse.c / cse.g) * cse.r * cse.s;  // K-dimension of the GEMM

    std::vector<Candidate> out;
    for (int tm : kTileSizes)
        for (int tn : kTileSizes)
            for (int tk : kTileSizes)
                for (const char* pipe : kPipelines) {
                    if (M < tm || N < tn || Kgm < tk) continue;
                    out.push_back({tm, tn, tk, pipe});
                }
    return out;
}

// ── Per-candidate compile + time ─────────────────────────────────────────────

static constexpr int kWarmup  = 3;
static constexpr int kRepeat  = 20;

// Returns latency in microseconds, or nullopt on compile/launch failure.
std::optional<double> timeCandidate(const ConvCase& cse, std::int64_t Ho, std::int64_t Wo,
                                    const Candidate& cand, const hipDeviceProp_t& props,
                                    void* devX, void* devW, void* devY) {
    ck_dsl::CEngine::ConvProblem prob;
    prob.N        = (int)cse.n;
    prob.Hi       = (int)cse.hi;
    prob.Wi       = (int)cse.wi;
    prob.C        = (int)cse.c;
    prob.K        = (int)cse.k;
    prob.R        = (int)cse.r;
    prob.S        = (int)cse.s;
    prob.sH       = (int)cse.strideH;
    prob.sW       = (int)cse.strideW;
    prob.pH       = (int)cse.padH;
    prob.pW       = (int)cse.padW;
    prob.dH       = (int)cse.dilH;
    prob.dW       = (int)cse.dilW;
    prob.tile_m   = cand.tile_m;
    prob.tile_n   = cand.tile_n;
    prob.tile_k   = cand.tile_k;
    prob.pipeline = cand.pipeline.c_str();
    // gcnArchName may carry suffixes like "gfx942:sramecc+:xnack-"; the C engine
    // lookup is an exact match so strip at the first colon, same as the provider.
    std::string arch_bare = props.gcnArchName;
    if (auto colon = arch_bare.find(':'); colon != std::string::npos)
        arch_bare.resize(colon);
    prob.arch = arch_bare.c_str();

    ck_dsl::CEngineResult r;
    try {
        r = ck_dsl::CEngine::build_conv(prob);
    } catch (const std::exception& e) {
        std::cerr << "[Sweep]   build_conv FAILED: " << e.what() << "\n";
        return std::nullopt;
    }

    ck_dsl::Kernel kernel = ck_dsl::Kernel::from_llvm_ir(
        std::move(r.llvm_ir), std::move(r.manifest), "");
    try {
        kernel.ensure_compiled(props);
    } catch (const std::exception& e) {
        std::cerr << "[Sweep]   compile FAILED: " << e.what() << "\n";
        return std::nullopt;
    }

    const std::int64_t elt  = 2;  // fp16
    const std::int64_t cpg  = cse.c / cse.g;
    uint64_t a_bytes = (uint64_t)cse.n * cse.hi * cse.wi * cse.c * elt;
    uint64_t b_bytes = (uint64_t)cse.k * cpg * cse.r * cse.s * elt;
    uint64_t d_bytes = (uint64_t)cse.n * Ho * Wo * cse.k * elt;

    const auto& m = kernel.manifest();
    long M_long    = (long)(cse.n * Ho * Wo);
    unsigned m_tiles = (unsigned)((M_long + m.block_m - 1) / m.block_m);
    unsigned n_tiles = (unsigned)((cse.k   + m.block_n - 1) / m.block_n);
    std::array<unsigned, 3> grid = (m.grid_order == "NM")
        ? std::array<unsigned, 3>{n_tiles, m_tiles, 1}
        : std::array<unsigned, 3>{m_tiles, n_tiles, 1};
    unsigned block = (unsigned)m.threads_per_block;

    if (grid[0] == 0 || grid[1] == 0) return std::nullopt;

    auto launch = [&]() {
        kernel.launch(
            {{"A", devX}, {"B", devW}, {"D", devY}},
            {{"A_bytes", a_bytes}, {"B_bytes", b_bytes}, {"D_bytes", d_bytes}},
            grid, block, nullptr);
    };

    // Warmup
    try {
        for (int i = 0; i < kWarmup; ++i) launch();
        if (hipDeviceSynchronize() != hipSuccess) return std::nullopt;
    } catch (...) {
        std::cerr << "[Sweep]   warmup FAILED\n";
        return std::nullopt;
    }

    // Timed runs
    auto t0 = std::chrono::steady_clock::now();
    try {
        for (int i = 0; i < kRepeat; ++i) launch();
        if (hipDeviceSynchronize() != hipSuccess) return std::nullopt;
    } catch (...) {
        std::cerr << "[Sweep]   timed launch FAILED\n";
        return std::nullopt;
    }
    auto t1 = std::chrono::steady_clock::now();

    double total_us =
        std::chrono::duration<double, std::micro>(t1 - t0).count();
    return total_us / kRepeat;
}

// ── Per-shape sweep ───────────────────────────────────────────────────────────

bool runConvSweep(const ConvCase& cse, const hipDeviceProp_t& props) {
    const std::int64_t Ho = convOutDim(cse.hi, cse.padH, cse.dilH, cse.r, cse.strideH);
    const std::int64_t Wo = convOutDim(cse.wi, cse.padW, cse.dilW, cse.s, cse.strideW);
    if (Ho <= 0 || Wo <= 0) {
        std::cerr << "[Sweep] " << cse.name << ": invalid output dims, skipping\n";
        return false;
    }

    const std::int64_t cpg   = cse.c / cse.g;
    const std::int64_t elt   = 2;  // fp16
    std::int64_t a_bytes = cse.n * cse.hi * cse.wi * cse.c * elt;
    std::int64_t b_bytes = cse.k * cpg * cse.r * cse.s * elt;
    std::int64_t d_bytes = cse.n * Ho * Wo * cse.k * elt;

    void* devX = nullptr;
    void* devW = nullptr;
    void* devY = nullptr;
    if (hipMalloc(&devX, (size_t)a_bytes) != hipSuccess ||
        hipMalloc(&devW, (size_t)b_bytes) != hipSuccess ||
        hipMalloc(&devY, (size_t)d_bytes) != hipSuccess) {
        std::cerr << "[Sweep] " << cse.name << ": device alloc FAILED, skipping\n";
        hipFree(devX); hipFree(devW); hipFree(devY);
        return false;
    }
    // Initialize input buffers to a benign constant.
    hipMemset(devX, 0, (size_t)a_bytes);
    hipMemset(devW, 0, (size_t)b_bytes);

    const std::vector<Candidate> candidates = enumerateCandidates(cse, Ho, Wo);
    std::cerr << "[Sweep] " << cse.name
              << " (N=" << cse.n << " G=" << cse.g << " C=" << cse.c
              << " K=" << cse.k << " Hi=" << cse.hi << " Wi=" << cse.wi
              << " R=" << cse.r << " S=" << cse.s << ")"
              << " sweeping " << candidates.size() << " candidates\n";

    // FLOPS: 2 * N * Ho * Wo * K * (C/G) * R * S
    const double kFlops = 2.0 * (double)cse.n * (double)Ho * (double)Wo
                              * (double)cse.k * (double)cpg
                              * (double)cse.r * (double)cse.s;

    std::size_t ok = 0, failed = 0;
    for (const auto& cand : candidates) {
        auto lat_us = timeCandidate(cse, Ho, Wo, cand, props, devX, devW, devY);
        if (!lat_us) { ++failed; continue; }
        ++ok;
        double tflops = kFlops / (*lat_us * 1e-6) / 1e12;
        gCsvWriter.write(cse.n, cse.g, cse.c, cse.k, cse.hi, cse.wi, cse.r, cse.s,
                         cse.strideH, cse.strideW, cse.padH, cse.padW,
                         cand.tile_m, cand.tile_n, cand.tile_k, cand.pipeline,
                         tflops, *lat_us);
    }

    hipFree(devX); hipFree(devW); hipFree(devY);
    std::cerr << "[Sweep] " << cse.name
              << " done: " << ok << " ok, " << failed << " failed\n";
    return ok > 0;
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

    // Detect device arch once; ensure_compiled(props) handles ISA derivation.
    hipDeviceProp_t props{};
    if (hipGetDeviceProperties(&props, 0) != hipSuccess) {
        std::cerr << "ERROR: hipGetDeviceProperties failed\n";
        return 1;
    }
    std::cerr << "[Sweep] device: " << props.name
              << " (" << props.gcnArchName << ")\n";

    std::size_t shapesOk = 0, shapesFailed = 0;
    for (const auto& shape : shapes) {
        if (runConvSweep(shape, props))
            ++shapesOk;
        else
            ++shapesFailed;
    }

    std::cerr << "=== Sweep complete: " << shapesOk << "/" << shapes.size()
              << " shapes produced data (" << shapesFailed << " failed) ===\n";
    return shapesFailed > 0 ? 1 : 0;
}
