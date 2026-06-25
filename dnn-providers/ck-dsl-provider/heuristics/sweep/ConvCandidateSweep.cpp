// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// heuristics/sweep/ConvCandidateSweep.cpp
//
// Standalone candidate sweep for the ck_dsl implicit-GEMM conv path.
//
// For each shape provided via --shapes:
//   1. Enumerate all (tile_m, tile_n, tile_k, pipeline) candidates that pass
//      ckc_implicit_gemm_conv_is_valid_spec() — the same gate used by build_conv.
//      Candidates that fail pre-validation are logged and excluded; the resulting
//      list is deterministic for a given (shape, arch) pair.
//   2. JIT-compile each pre-validated candidate via CEngine::build_conv.
//      Any failure here is an ERROR (not a silent skip): the validator and the
//      compiler disagree, which requires triage.
//   3. Time each compiled candidate on device and write one CSV row per success.
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
//   --dtype  <dtype>  Data type: fp16 (only supported value).
//
// Exit code: 0 if all shapes produced at least one timing row and no pre-validated
// candidate failed. Non-zero otherwise — the stderr output identifies which shapes
// and candidates need triage.

#include <hip/hip_runtime.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <exception>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "ckc/instance_conv_implicit_gemm.h"
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
static std::string gDtype = "fp16";  // set by sweepMain; reserved for future multi-dtype support

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
// tile_k, warp_tile_m, warp_tile_n, warp_tile_k, warp_m, warp_n, pipeline).
//
// Warp atom shapes depend on the GPU arch:
//   gfx942 / gfx950:  primary = {32x32x16}; fallback = {16x16x16}
//   gfx90a (CDNA2):   primary = {32x32x8};  fallback = {16x16x16}
//   (mfma_f32_32x32x16_f16 does NOT exist on CDNA2)
//
// For each (tile_m, tile_n, tile_k) combination we pick exactly ONE warp atom
// (the best valid one) to avoid duplicate CSV rows for the same tile choice.
// The selection priority is: largest-k atom first (better compute intensity),
// falling back to smaller atoms when the tile doesn't divide evenly.
//
// Candidates are pre-filtered by:
//   - M = N*Ho*Wo must be >= tile_m (grid must be non-empty)
//   - K must be >= tile_n
//   - C/G*R*S must be >= tile_k (k-loop must have at least one iteration)
//   - tile_m must be divisible by warp_m * warp_tile_m
//   - tile_n must be divisible by warp_n * warp_tile_n
//   - tile_k must be divisible by warp_tile_k

struct WarpAtom {
    int warp_tile_m, warp_tile_n, warp_tile_k;
    int warp_m, warp_n;
};

struct Candidate {
    int tile_m, tile_n, tile_k;
    int warp_tile_m, warp_tile_n, warp_tile_k;
    int warp_m, warp_n;
    std::string pipeline;
};

// Returns the warp atom candidates to try (in priority order) for the given arch.
// The first atom that divides tile_m, tile_n, tile_k evenly "wins" for that tile.
static std::vector<WarpAtom> warpAtomCandidatesForArch(const std::string& arch) {
    // Strip any suffix (e.g. "gfx90a:sramecc+:xnack-" -> "gfx90a").
    std::string bare = arch.substr(0, arch.find(':'));
    if (bare == "gfx90a") {
        // CDNA2 (MI200): mfma_f32_32x32x8_f16 and mfma_f32_16x16x16_f16 only.
        // mfma_f32_32x32x16_f16 does NOT exist on this arch.
        return {
            {32, 32, 8,  2, 2},  // preferred: 32x32x8 atom (warp_tile_k=8)
            {16, 16, 16, 2, 2},  // fallback:  16x16x16 atom
        };
    }
    // Default: CDNA3/4 (gfx942, gfx950) have mfma_f32_32x32x16_f16.
    return {
        {32, 32, 16, 2, 2},  // preferred: 32x32x16 atom (warp_tile_k=16)
        {16, 16, 16, 2, 2},  // fallback:  16x16x16 atom
    };
}

// Builds the ckc spec struct that mirrors what CEngine::build_conv fills in.
// Used by enumerateCandidates to pre-validate tile/pipeline combinations.
static ckc_implicit_gemm_conv_spec_t makeSpec(const ConvCase& cse, const Candidate& cand) {
    ckc_implicit_gemm_conv_spec_t s = ckc_implicit_gemm_conv_spec_default();
    s.problem = ckc_conv_problem_make(
        (int)cse.n, (int)cse.hi, (int)cse.wi, (int)cse.c, (int)cse.k,
        (int)cse.r, (int)cse.s,
        (int)cse.strideH, (int)cse.strideW, (int)cse.padH, (int)cse.padW,
        (int)cse.dilH, (int)cse.dilW);
    s.name        = "conv_igemm";
    s.groups      = (int)cse.g;
    s.tile_m      = cand.tile_m;
    s.tile_n      = cand.tile_n;
    s.tile_k      = cand.tile_k;
    s.warp_m      = cand.warp_m;
    s.warp_n      = cand.warp_n;
    s.warp_tile_m = cand.warp_tile_m;
    s.warp_tile_n = cand.warp_tile_n;
    s.warp_tile_k = cand.warp_tile_k;
    s.pipeline    = cand.pipeline.c_str();
    return s;
}

std::vector<Candidate> enumerateCandidates(const ConvCase& cse, std::int64_t Ho, std::int64_t Wo,
                                           const std::string& arch) {
    static const int kTileSizes[] = {32, 64, 128};
    static const char* kPipelines[] = {"mem", "compv3", "compv4"};

    const std::int64_t M   = cse.n * Ho * Wo;
    const std::int64_t N   = cse.k;
    const std::int64_t Kgm = (cse.c / cse.g) * cse.r * cse.s;  // K-dimension of the GEMM

    // Strip suffixes so ckc_implicit_gemm_conv_is_valid_spec gets a bare arch string.
    std::string arch_bare = arch.substr(0, arch.find(':'));

    const auto atoms = warpAtomCandidatesForArch(arch);

    std::vector<Candidate> out;
    std::size_t nPrerejected = 0;
    for (int tm : kTileSizes)
        for (int tn : kTileSizes)
            for (int tk : kTileSizes) {
                if (M < tm || N < tn || Kgm < tk) continue;
                // Select the first (best) warp atom that divides this tile evenly.
                // We produce exactly one candidate per (tile_m, tile_n, tile_k) tuple
                // to avoid duplicate rows in the training CSV.
                WarpAtom chosen{32, 32, 16, 2, 2};
                bool found = false;
                for (const WarpAtom& a : atoms) {
                    if (tm % (a.warp_m * a.warp_tile_m) != 0) continue;
                    if (tn % (a.warp_n * a.warp_tile_n) != 0) continue;
                    if (tk % a.warp_tile_k != 0) continue;
                    chosen = a;
                    found  = true;
                    break;
                }
                if (!found) continue;  // no valid atom for this tile combination

                for (const char* pipe : kPipelines) {
                    Candidate cand{tm, tn, tk,
                                   chosen.warp_tile_m, chosen.warp_tile_n,
                                   chosen.warp_tile_k, chosen.warp_m, chosen.warp_n,
                                   pipe};
                    // Pre-validate using the same gate as build_conv. This makes the
                    // candidate list deterministic: only combinations that will survive
                    // is_valid_spec reach the compile+time loop.
                    char reason[256] = {0};
                    ckc_implicit_gemm_conv_spec_t spec = makeSpec(cse, cand);
                    if (!ckc_implicit_gemm_conv_is_valid_spec(&spec, arch_bare.c_str(),
                                                              reason, sizeof reason)) {
                        std::cerr << "[Sweep]   pre-reject tile(" << tm << "," << tn
                                  << "," << tk << ") pipeline=" << pipe
                                  << ": " << reason << "\n";
                        ++nPrerejected;
                        continue;
                    }
                    out.push_back(std::move(cand));
                }
            }
    if (nPrerejected > 0)
        std::cerr << "[Sweep] " << cse.name << ": pre-rejected " << nPrerejected
                  << " candidates (LDS/arch/spec constraints)\n";
    return out;
}

// ── Per-candidate compile + time ─────────────────────────────────────────────

static constexpr int kWarmup  = 3;
static constexpr int kRepeat  = 20;

// Result from timeCandidate.
// latency_us is set on success; is_error is set when a pre-validated candidate
// failed to compile or launch — which should never happen and requires triage.
// timed_out is set when the candidate exceeded the per-candidate wall-clock limit.
struct TimingResult {
    std::optional<double> latency_us;
    bool is_error   = false;
    bool timed_out  = false;
};

// Compiles and times one pre-validated candidate.
// All failures are treated as errors: a candidate that passed is_valid_spec
// must succeed here. If it does not, the validator has a gap.
TimingResult timeCandidate(const ConvCase& cse, std::int64_t Ho, std::int64_t Wo,
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
    prob.tile_m      = cand.tile_m;
    prob.tile_n      = cand.tile_n;
    prob.tile_k      = cand.tile_k;
    prob.warp_m      = cand.warp_m;
    prob.warp_n      = cand.warp_n;
    prob.warp_tile_m = cand.warp_tile_m;
    prob.warp_tile_n = cand.warp_tile_n;
    prob.warp_tile_k = cand.warp_tile_k;
    prob.pipeline    = cand.pipeline.c_str();
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
        std::cerr << "[Sweep]   ERROR build_conv FAILED (pre-validated candidate): "
                  << e.what() << "\n"
                  << "[Sweep]   TRIAGE: is_valid_spec accepted this candidate but "
                     "build_conv rejected it — the validator has a gap.\n";
        return {std::nullopt, /*is_error=*/true};
    }

    ck_dsl::Kernel kernel = ck_dsl::Kernel::from_llvm_ir(
        std::move(r.llvm_ir), std::move(r.manifest), "");
    try {
        kernel.ensure_compiled(props);
    } catch (const std::exception& e) {
        std::cerr << "[Sweep]   ERROR compile FAILED (pre-validated candidate): "
                  << e.what() << "\n"
                  << "[Sweep]   TRIAGE: is_valid_spec accepted this candidate but "
                     "ISA compilation failed — check COMGR / LDS / resource limits.\n";
        return {std::nullopt, /*is_error=*/true};
    }

    const std::int64_t elt  = 2;  // fp16/bf16 are both 2 bytes
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

    if (grid[0] == 0 || grid[1] == 0) {
        std::cerr << "[Sweep]   ERROR zero-dimension grid after successful compile — "
                     "reporting as error for triage.\n";
        return {std::nullopt, /*is_error=*/true};
    }

    auto launch = [&]() {
        kernel.launch(
            {{"A", devX}, {"B", devW}, {"D", devY}},
            {{"A_bytes", a_bytes}, {"B_bytes", b_bytes}, {"D_bytes", d_bytes}},
            grid, block, nullptr);
    };

    // Warmup
    try {
        for (int i = 0; i < kWarmup; ++i) launch();
        if (hipDeviceSynchronize() != hipSuccess) {
            std::cerr << "[Sweep]   ERROR warmup sync failed — reporting as error.\n";
            return {std::nullopt, /*is_error=*/true};
        }
    } catch (const std::exception& e) {
        std::cerr << "[Sweep]   ERROR warmup FAILED: " << e.what() << "\n";
        return {std::nullopt, /*is_error=*/true};
    }

    // Timed runs
    auto t0 = std::chrono::steady_clock::now();
    try {
        for (int i = 0; i < kRepeat; ++i) launch();
        if (hipDeviceSynchronize() != hipSuccess) {
            std::cerr << "[Sweep]   ERROR timed-run sync failed — reporting as error.\n";
            return {std::nullopt, /*is_error=*/true};
        }
    } catch (const std::exception& e) {
        std::cerr << "[Sweep]   ERROR timed launch FAILED: " << e.what() << "\n";
        return {std::nullopt, /*is_error=*/true};
    }
    auto t1 = std::chrono::steady_clock::now();

    double total_us =
        std::chrono::duration<double, std::micro>(t1 - t0).count();
    return {total_us / kRepeat, /*is_error=*/false};
}

// ── Per-shape sweep ───────────────────────────────────────────────────────────

struct ShapeSweepResult {
    bool   produced_data;  // at least one timing row written
    size_t errors;         // pre-validated candidates that failed compile/run
    size_t timeouts;       // candidates that exceeded the per-candidate wall-clock limit
};

ShapeSweepResult runConvSweep(const ConvCase& cse, const hipDeviceProp_t& props,
                               int candidateTimeoutS) {
    const std::int64_t Ho = convOutDim(cse.hi, cse.padH, cse.dilH, cse.r, cse.strideH);
    const std::int64_t Wo = convOutDim(cse.wi, cse.padW, cse.dilW, cse.s, cse.strideW);
    if (Ho <= 0 || Wo <= 0) {
        std::cerr << "[Sweep] " << cse.name << ": invalid output dims, skipping\n";
        return {false, 0};
    }

    const std::int64_t cpg   = cse.c / cse.g;
    const std::int64_t elt   = 2;  // fp16/bf16
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
        return {false, 0};
    }
    // Initialize input buffers to a benign constant.
    hipMemset(devX, 0, (size_t)a_bytes);
    hipMemset(devW, 0, (size_t)b_bytes);

    const std::vector<Candidate> candidates = enumerateCandidates(cse, Ho, Wo,
                                                                    props.gcnArchName);
    std::cerr << "[Sweep] " << cse.name
              << " (N=" << cse.n << " G=" << cse.g << " C=" << cse.c
              << " K=" << cse.k << " Hi=" << cse.hi << " Wi=" << cse.wi
              << " R=" << cse.r << " S=" << cse.s << ")"
              << " sweeping " << candidates.size() << " pre-validated candidates\n";

    // FLOPS: 2 * N * Ho * Wo * K * (C/G) * R * S
    const double kFlops = 2.0 * (double)cse.n * (double)Ho * (double)Wo
                              * (double)cse.k * (double)cpg
                              * (double)cse.r * (double)cse.s;

    const auto timeoutDur = std::chrono::seconds(candidateTimeoutS);

    // Timed-out futures are moved here so their threads can drain after the
    // candidate loop rather than blocking each subsequent iteration.
    std::vector<std::future<TimingResult>> graveyard;

    std::size_t ok = 0, errors = 0, timeouts = 0;
    for (const auto& cand : candidates) {
        // Run timeCandidate on a background thread so we can enforce a wall-clock
        // limit. If the JIT compile or hipDeviceSynchronize hangs, the future
        // times out and we skip the candidate rather than stalling the shard.
        auto fut = std::async(std::launch::async,
                              timeCandidate, cse, Ho, Wo, cand,
                              std::cref(props), devX, devW, devY);

        if (fut.wait_for(timeoutDur) == std::future_status::timeout) {
            ++timeouts;
            std::cerr << "[Sweep]   TIMEOUT: " << cse.name
                      << " tile(" << cand.tile_m << "," << cand.tile_n
                      << "," << cand.tile_k << ") pipeline=" << cand.pipeline
                      << " exceeded " << candidateTimeoutS << "s limit — skipping\n";
            graveyard.push_back(std::move(fut));
            continue;
        }

        TimingResult res = fut.get();
        if (res.is_error) {
            ++errors;
            std::cerr << "[Sweep]   TRIAGE NEEDED: " << cse.name
                      << " tile(" << cand.tile_m << "," << cand.tile_n
                      << "," << cand.tile_k << ") pipeline=" << cand.pipeline
                      << " passed is_valid_spec but failed compile/run\n";
            continue;
        }
        if (!res.latency_us) continue;  // grid=0 or other non-error skip
        ++ok;
        double tflops = kFlops / (*res.latency_us * 1e-6) / 1e12;
        gCsvWriter.write(cse.n, cse.g, cse.c, cse.k, cse.hi, cse.wi, cse.r, cse.s,
                         cse.strideH, cse.strideW, cse.padH, cse.padW,
                         cand.tile_m, cand.tile_n, cand.tile_k, cand.pipeline,
                         tflops, *res.latency_us);
    }

    hipFree(devX); hipFree(devW); hipFree(devY);
    std::cerr << "[Sweep] " << cse.name
              << " done: " << ok << " ok";
    if (timeouts > 0)
        std::cerr << ", " << timeouts << " timed out (>" << candidateTimeoutS << "s)";
    if (errors > 0)
        std::cerr << ", " << errors << " ERRORS (triage required)";
    std::cerr << "\n";
    return {ok > 0, errors, timeouts};
}

}  // namespace

// ── Entry point ───────────────────────────────────────────────────────────────

int sweepMain(const std::string& shapesPath, const std::string& outPath,
              const std::string& dtype, int candidateTimeoutS) {
    if (dtype != "fp16") {
        std::cerr << "ERROR: --dtype must be fp16 (got: " << dtype << ")\n";
        return 1;
    }
    gDtype = dtype;

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
    std::cerr << "[Sweep] per-candidate timeout: " << candidateTimeoutS << "s\n";

    std::size_t shapesOk = 0, shapesNoData = 0, totalErrors = 0, totalTimeouts = 0;
    for (const auto& shape : shapes) {
        ShapeSweepResult r = runConvSweep(shape, props, candidateTimeoutS);
        if (r.produced_data)
            ++shapesOk;
        else
            ++shapesNoData;
        totalErrors   += r.errors;
        totalTimeouts += r.timeouts;
    }

    std::cerr << "=== Sweep complete: " << shapesOk << "/" << shapes.size()
              << " shapes produced data";
    if (shapesNoData > 0)
        std::cerr << ", " << shapesNoData << " produced no data";
    if (totalTimeouts > 0)
        std::cerr << ", " << totalTimeouts << " candidates timed out (>"
                  << candidateTimeoutS << "s)";
    if (totalErrors > 0)
        std::cerr << ", " << totalErrors
                  << " CANDIDATE ERRORS (pre-validated but compile/run failed — TRIAGE REQUIRED)";
    std::cerr << " ===\n";

    if (totalErrors > 0) {
        std::cerr << "ERROR: " << totalErrors << " candidate(s) passed is_valid_spec "
                  << "but failed in the compiler or on-device. This indicates a gap "
                  << "in the validator. Review the TRIAGE lines above.\n";
        return 2;  // distinct from shape-level failure (exit 1)
    }
    return shapesNoData > 0 ? 1 : 0;
}
