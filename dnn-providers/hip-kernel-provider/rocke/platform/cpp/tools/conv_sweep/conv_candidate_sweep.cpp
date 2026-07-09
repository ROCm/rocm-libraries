// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Cpp/tools/conv_sweep/conv_candidate_sweep.cpp
//
// Standalone candidate sweep for the rocKE implicit-GEMM conv path.
// Port of the ck-dsl ConvCandidateSweep.cpp using the rocke_* C API.
//
// For each shape provided via --shapes:
//   1. Enumerate all (tile_m, tile_n, tile_k, pipeline) candidates that pass
//      rocke_implicit_gemm_conv_is_valid_spec().
//   2. JIT-compile each pre-validated candidate via rocke_conv_implicit_gemm_lower_to_llvm
//      + comgr (LLVM IR -> HSACO).
//   3. Time each compiled candidate on device via HIP and write one CSV row per success.

#include <hip/hip_runtime.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "comgr.hpp"
#include "rocke/instance_conv_implicit_gemm.h"

namespace
{

// Strip target-specific features from GCN arch name (e.g. "gfx942:sramecc+:xnack-" -> "gfx942")
static std::string stripArchFeatures(const std::string& arch)
{
    auto colon = arch.find(':');
    return (colon != std::string::npos) ? arch.substr(0, colon) : arch;
}

// ── Training CSV writer ───────────────────────────────────────────────────────

static const char* kTrainingCsvHeader = "N,G,C,K,Hi,Wi,Y,X,stride_h,stride_w,pad_h,pad_w,"
                                        "tile_m,tile_n,tile_k,pipeline,tflops,latency_us\n";

struct TrainingCsvWriter
{
    std::mutex mu;
    std::ofstream f;
    bool active = false;

    void open(const std::string& path)
    {
        std::lock_guard<std::mutex> lk(mu);
        const bool exists = std::ifstream(path).good();
        f.open(path, std::ios::app);
        if(!f)
        {
            std::cerr << "[TrainingCSV] ERROR: cannot open " << path << "\n";
            return;
        }
        if(!exists)
            f << kTrainingCsvHeader;
        active = true;
        std::cerr << "[TrainingCSV] Writing training rows to " << path << "\n";
    }

    void write(std::int64_t N,
               std::int64_t G,
               std::int64_t C,
               std::int64_t K,
               std::int64_t Hi,
               std::int64_t Wi,
               std::int64_t Y,
               std::int64_t X,
               std::int64_t sH,
               std::int64_t sW,
               std::int64_t pH,
               std::int64_t pW,
               int tile_m,
               int tile_n,
               int tile_k,
               const std::string& pipeline,
               double tflops,
               double latency_us)
    {
        if(!active)
            return;
        std::lock_guard<std::mutex> lk(mu);
        f << N << ',' << G << ',' << C << ',' << K << ',' << Hi << ',' << Wi << ',' << Y << ',' << X
          << ',' << sH << ',' << sW << ',' << pH << ',' << pW << ',' << tile_m << ',' << tile_n
          << ',' << tile_k << ',' << pipeline << ',' << tflops << ',' << latency_us << '\n';
    }
};

static TrainingCsvWriter gCsvWriter;

// ── Shape descriptor ─────────────────────────────────────────────────────────

struct ConvCase
{
    std::string name;
    std::int64_t n, g, c, hi, wi;
    std::int64_t k, r, s;
    std::int64_t strideH, strideW;
    std::int64_t padH, padW;
    std::int64_t dilH, dilW;
};

std::int64_t convOutDim(
    std::int64_t in, std::int64_t pad, std::int64_t dil, std::int64_t filt, std::int64_t stride)
{
    return (in + 2 * pad - dil * (filt - 1) - 1) / stride + 1;
}

// ── CSV shape loader ──────────────────────────────────────────────────────────

std::vector<ConvCase> loadShapesFromCsv(const std::string& path)
{
    std::ifstream f(path);
    if(!f)
    {
        std::cerr << "[ShapeLoader] ERROR: cannot open " << path << "\n";
        return {};
    }
    std::string header;
    if(!std::getline(f, header))
        return {};

    std::vector<ConvCase> cases;
    std::string line;
    int idx = 0;
    while(std::getline(f, line))
    {
        if(line.empty())
            continue;
        std::istringstream ss(line);
        std::string tok;
        std::vector<std::string> fields;
        while(std::getline(ss, tok, ','))
            fields.push_back(tok);
        if(fields.size() < 12)
            continue;
        try
        {
            ConvCase c{};
            c.name = "shape_" + std::to_string(idx);
            c.n = std::stoll(fields[0]);
            c.g = std::stoll(fields[1]);
            if(c.g < 1)
                c.g = 1;
            c.c = std::stoll(fields[2]);
            c.k = std::stoll(fields[3]);
            c.hi = std::stoll(fields[4]);
            c.wi = std::stoll(fields[5]);
            c.r = std::stoll(fields[6]);
            c.s = std::stoll(fields[7]);
            c.strideH = std::stoll(fields[8]);
            c.strideW = std::stoll(fields[9]);
            c.padH = std::stoll(fields[10]);
            c.padW = std::stoll(fields[11]);
            c.dilH = 1;
            c.dilW = 1;
            if(c.c % (c.g * 8) != 0 || c.k % (c.g * 8) != 0)
                continue;
            cases.push_back(c);
        }
        catch(...)
        {
            continue;
        }
        ++idx;
    }
    std::cerr << "[ShapeLoader] Loaded " << cases.size() << " shapes from " << path << "\n";
    return cases;
}

// ── Candidate enumeration ─────────────────────────────────────────────────────

struct WarpAtom
{
    int warp_tile_m, warp_tile_n, warp_tile_k;
    int warp_m, warp_n;
};

struct Candidate
{
    int tile_m, tile_n, tile_k;
    int warp_tile_m, warp_tile_n, warp_tile_k;
    int warp_m, warp_n;
    std::string pipeline;
};

static std::vector<WarpAtom> warpAtomCandidatesForArch(const std::string& arch)
{
    std::string bare = stripArchFeatures(arch);
    if(bare == "gfx90a")
    {
        return {
            {32, 32, 8, 2, 2},
            {16, 16, 16, 2, 2},
        };
    }
    return {
        {32, 32, 16, 2, 2},
        {16, 16, 16, 2, 2},
    };
}

static rocke_implicit_gemm_conv_spec_t makeSpec(const ConvCase& cse, const Candidate& cand)
{
    rocke_implicit_gemm_conv_spec_t s = rocke_implicit_gemm_conv_spec_default();
    s.problem = rocke_conv_problem_make((int)cse.n,
                                        (int)cse.hi,
                                        (int)cse.wi,
                                        (int)cse.c,
                                        (int)cse.k,
                                        (int)cse.r,
                                        (int)cse.s,
                                        (int)cse.strideH,
                                        (int)cse.strideW,
                                        (int)cse.padH,
                                        (int)cse.padW,
                                        (int)cse.dilH,
                                        (int)cse.dilW);
    s.name = "conv_igemm";
    s.groups = (int)cse.g;
    s.tile_m = cand.tile_m;
    s.tile_n = cand.tile_n;
    s.tile_k = cand.tile_k;
    s.warp_m = cand.warp_m;
    s.warp_n = cand.warp_n;
    s.warp_tile_m = cand.warp_tile_m;
    s.warp_tile_n = cand.warp_tile_n;
    s.warp_tile_k = cand.warp_tile_k;
    s.pipeline = cand.pipeline.c_str();
    return s;
}

std::vector<Candidate> enumerateCandidates(const ConvCase& cse,
                                           std::int64_t Ho,
                                           std::int64_t Wo,
                                           const std::string& arch)
{
    static const int kTileSizes[] = {32, 64, 128};
    static const char* kPipelines[] = {"mem", "compv3", "compv4"};

    const std::int64_t M = cse.n * Ho * Wo;
    const std::int64_t N = cse.k;
    const std::int64_t Kgm = (cse.c / cse.g) * cse.r * cse.s;

    std::string arch_bare = stripArchFeatures(arch);

    const auto atoms = warpAtomCandidatesForArch(arch);

    std::vector<Candidate> out;
    std::size_t nPrerejected = 0;
    for(int tm : kTileSizes)
        for(int tn : kTileSizes)
            for(int tk : kTileSizes)
            {
                if(M < tm || N < tn || Kgm < tk)
                    continue;
                WarpAtom chosen{32, 32, 16, 2, 2};
                bool found = false;
                for(const WarpAtom& a : atoms)
                {
                    if(tm % (a.warp_m * a.warp_tile_m) != 0)
                        continue;
                    if(tn % (a.warp_n * a.warp_tile_n) != 0)
                        continue;
                    if(tk % a.warp_tile_k != 0)
                        continue;
                    chosen = a;
                    found = true;
                    break;
                }
                if(!found)
                    continue;

                for(const char* pipe : kPipelines)
                {
                    Candidate cand{tm,
                                   tn,
                                   tk,
                                   chosen.warp_tile_m,
                                   chosen.warp_tile_n,
                                   chosen.warp_tile_k,
                                   chosen.warp_m,
                                   chosen.warp_n,
                                   pipe};
                    char reason[256] = {0};
                    rocke_implicit_gemm_conv_spec_t spec = makeSpec(cse, cand);
                    if(!rocke_implicit_gemm_conv_is_valid_spec(
                           &spec, arch_bare.c_str(), reason, sizeof reason))
                    {
                        std::cerr << "[Sweep]   pre-reject tile(" << tm << "," << tn << "," << tk
                                  << ") pipeline=" << pipe << ": " << reason << "\n";
                        ++nPrerejected;
                        continue;
                    }
                    out.push_back(std::move(cand));
                }
            }
    if(nPrerejected > 0)
        std::cerr << "[Sweep] " << cse.name << ": pre-rejected " << nPrerejected
                  << " candidates (LDS/arch/spec constraints)\n";
    return out;
}

// ── Per-candidate compile + time ─────────────────────────────────────────────

static constexpr int kWarmup = 3;
static constexpr int kRepeat = 20;

struct TimingResult
{
    std::optional<double> latency_us;
    bool is_error = false;
    bool timed_out = false;
};

static inline void hip_check(hipError_t e, const char* where)
{
    if(e != hipSuccess)
        throw std::runtime_error(std::string(where) + ": " + hipGetErrorString(e));
}

static inline unsigned ceil_div(long x, int d)
{
    return d > 0 ? static_cast<unsigned>((x + d - 1) / d) : 0u;
}

// Compile a candidate to an HSACO (no timeout — runs synchronously).
// Returns: {hsaco, kernel_name, block_size} on success, or sets is_error.
struct CompileResult
{
    std::vector<std::byte> hsaco;
    std::string kernel_name;
    unsigned block_size = 0;
    std::array<unsigned, 3> grid = {0, 0, 0};
    bool is_error = false;
    double lower_ms = 0;
    double comgr_ms = 0;
};

CompileResult compileCandidate(const ConvCase& cse,
                               std::int64_t Ho,
                               std::int64_t Wo,
                               const Candidate& cand,
                               const std::string& arch_bare)
{
    CompileResult cr;

    rocke_implicit_gemm_conv_spec_t spec = makeSpec(cse, cand);

    // Phase 1: Lower to LLVM IR
    auto t0 = std::chrono::steady_clock::now();
    char* ll_raw = nullptr;
    char err_buf[512] = {};
    rocke_status_t st = rocke_conv_implicit_gemm_lower_to_llvm(
        &spec, arch_bare.c_str(), ROCKE_LLVM_FLAVOR_AUTO, &ll_raw, err_buf, sizeof err_buf);
    auto t1 = std::chrono::steady_clock::now();
    cr.lower_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if(st != ROCKE_OK || !ll_raw)
    {
        std::cerr << "[Sweep]   ERROR lower_to_llvm FAILED: " << err_buf << "\n";
        cr.is_error = true;
        return cr;
    }
    std::string llvm_ir(ll_raw);
    free(ll_raw);

    // Phase 2: comgr compile (LLVM IR -> HSACO)
    auto t2 = std::chrono::steady_clock::now();
    std::string isa = rocke::Compiler::isa_for(arch_bare);
    try
    {
        cr.hsaco = rocke::Compiler::compile(llvm_ir, isa);
    }
    catch(const rocke::ComgrError& e)
    {
        auto t3 = std::chrono::steady_clock::now();
        cr.comgr_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        std::cerr << "[Sweep]   ERROR comgr compile FAILED (" << cr.comgr_ms << "ms): " << e.what()
                  << "\n";
        cr.is_error = true;
        return cr;
    }
    auto t3 = std::chrono::steady_clock::now();
    cr.comgr_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // Kernel name + grid
    char kname[256] = {};
    rocke_implicit_gemm_conv_spec_kernel_name(&spec, kname, sizeof kname);
    cr.kernel_name = kname;
    cr.block_size = (unsigned)rocke_implicit_gemm_conv_spec_block_size(&spec);

    const long M_long = (long)(cse.n * Ho * Wo);
    unsigned m_tiles = ceil_div(M_long, cand.tile_m);
    unsigned n_tiles = ceil_div((long)cse.k, cand.tile_n);
    cr.grid = {n_tiles, m_tiles, 1};

    return cr;
}

// ── Per-shape sweep ───────────────────────────────────────────────────────────

struct ShapeSweepResult
{
    bool produced_data;
    size_t errors;
    size_t timeouts;
};

// Pre-validate: load HSACO + check resource limits without launching.
// Returns false if the kernel can't run (e.g. exceeds VGPR budget).
bool preValidateCandidate(const CompileResult& cr)
{
    hipModule_t mod = nullptr;
    if(hipModuleLoadData(&mod, cr.hsaco.data()) != hipSuccess)
        return false;
    hipFunction_t fn = nullptr;
    if(hipModuleGetFunction(&fn, mod, cr.kernel_name.c_str()) != hipSuccess)
    {
        (void)hipModuleUnload(mod);
        return false;
    }
    int maxThreads = 0;
    (void)hipFuncGetAttribute(&maxThreads, HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, fn);
    (void)hipModuleUnload(mod);
    return maxThreads >= (int)cr.block_size;
}

// Locate the rocke_kern_time helper binary (adjacent to this binary).
static std::string findHelperBinary()
{
    // Check env var first.
    const char* env = std::getenv("ROCKE_KERN_TIME_BIN");
    if(env && access(env, X_OK) == 0)
        return env;

    // Look adjacent to our own binary.
    char self[4096] = {};
    ssize_t len = readlink("/proc/self/exe", self, sizeof(self) - 1);
    if(len > 0)
    {
        self[len] = '\0';
        std::string dir(self);
        auto slash = dir.rfind('/');
        if(slash != std::string::npos)
        {
            std::string candidate = dir.substr(0, slash + 1) + "rocke_kern_time";
            if(access(candidate.c_str(), X_OK) == 0)
                return candidate;
        }
    }
    return {};
}

static std::string gHelperBin;

// Fork+exec the helper binary to time a candidate in an isolated process.
// The child gets a fresh HIP context via exec(), so hung kernels can be
// SIGKILL'd without poisoning the parent.
TimingResult execLaunchAndTime(
    const CompileResult& cr, const ConvCase& cse, std::int64_t Ho, std::int64_t Wo, int timeoutS)
{
    // Write HSACO to an in-memory file (avoids disk I/O per candidate).
    int memfd = memfd_create("rocke_hsaco", 0);
    if(memfd < 0)
        return {std::nullopt, true};
    ssize_t wr = write(memfd, cr.hsaco.data(), cr.hsaco.size());
    if(wr != (ssize_t)cr.hsaco.size())
    {
        close(memfd);
        return {std::nullopt, true};
    }
    lseek(memfd, 0, SEEK_SET);

    // Compute buffer sizes.
    const std::int64_t elt = 2;
    const std::int64_t cpg = cse.c / cse.g;
    size_t sz_a = (size_t)(cse.n * cse.hi * cse.wi * cse.c * elt);
    size_t sz_b = (size_t)(cse.k * cpg * cse.r * cse.s * elt);
    size_t sz_d = (size_t)(cse.n * Ho * Wo * cse.k * elt);

    // Set up pipe for child's stdout.
    int pipefd[2];
    if(pipe(pipefd) != 0)
    {
        close(memfd);
        return {std::nullopt, true};
    }

    pid_t pid = fork();
    if(pid < 0)
    {
        close(pipefd[0]);
        close(pipefd[1]);
        close(memfd);
        return {std::nullopt, true};
    }

    if(pid == 0)
    {
        // Child: redirect stdout to pipe, exec helper.
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        close(pipefd[1]);

        // Pass memfd path so child reads HSACO without disk I/O.
        char memfd_path[64];
        snprintf(memfd_path, sizeof(memfd_path), "/proc/self/fd/%d", memfd);

        std::string s_gx = std::to_string(cr.grid[0]);
        std::string s_gy = std::to_string(cr.grid[1]);
        std::string s_gz = std::to_string(cr.grid[2]);
        std::string s_bs = std::to_string(cr.block_size);
        std::string s_a = std::to_string(sz_a);
        std::string s_b = std::to_string(sz_b);
        std::string s_d = std::to_string(sz_d);

        execl(gHelperBin.c_str(),
              "rocke_kern_time",
              memfd_path,
              cr.kernel_name.c_str(),
              s_gx.c_str(),
              s_gy.c_str(),
              s_gz.c_str(),
              s_bs.c_str(),
              s_a.c_str(),
              s_b.c_str(),
              s_d.c_str(),
              (char*)nullptr);
        // exec failed
        printf("ERROR exec_failed\n");
        _exit(127);
    }

    // Parent: close write end and memfd, then wait with timeout via alarm.
    close(pipefd[1]);
    close(memfd);

    // Use a blocking waitpid with a separate alarm-based timeout.
    // Install a no-op SIGALRM handler so waitpid returns EINTR on timeout.
    struct sigaction sa = {}, old_sa = {};
    sa.sa_handler = [](int) {};
    sa.sa_flags = 0;
    sigaction(SIGALRM, &sa, &old_sa);
    alarm((unsigned)timeoutS);

    int status = 0;
    pid_t w = waitpid(pid, &status, 0);
    alarm(0);
    sigaction(SIGALRM, &old_sa, nullptr);

    if(w != pid)
    {
        kill(pid, SIGKILL);
        waitpid(pid, &status, 0);
        close(pipefd[0]);
        return {std::nullopt, false, /*timed_out=*/true};
    }

    // Read child's stdout.
    char buf[256] = {};
    ssize_t n = read(pipefd[0], buf, sizeof(buf) - 1);
    close(pipefd[0]);

    if(n <= 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
    {
        if(n > 0)
            buf[n] = '\0';
        return {std::nullopt, true};
    }
    buf[n] = '\0';

    // Parse "OK <latency_us>".
    double latency_us = 0;
    if(sscanf(buf, "OK %lf", &latency_us) == 1 && latency_us > 0)
        return {latency_us, false};

    return {std::nullopt, true};
}

// Bounded concurrency guard for parallel compilation.
struct CompileSemaphore
{
    std::mutex mu;
    std::condition_variable cv;
    int available;

    explicit CompileSemaphore(int n)
        : available(n)
    {
    }
    void acquire()
    {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return available > 0; });
        --available;
    }
    void release()
    {
        std::lock_guard<std::mutex> lk(mu);
        ++available;
        cv.notify_one();
    }
};

static CompileSemaphore* gCompileSem = nullptr;

static CompileResult boundedCompile(const ConvCase& cse,
                                    std::int64_t Ho,
                                    std::int64_t Wo,
                                    const Candidate& cand,
                                    const std::string& arch_bare)
{
    if(gCompileSem)
        gCompileSem->acquire();
    auto cr = compileCandidate(cse, Ho, Wo, cand, arch_bare);
    if(gCompileSem)
        gCompileSem->release();
    return cr;
}

ShapeSweepResult
    runConvSweep(const ConvCase& cse, const hipDeviceProp_t& props, int candidateTimeoutS)
{
    const std::int64_t Ho = convOutDim(cse.hi, cse.padH, cse.dilH, cse.r, cse.strideH);
    const std::int64_t Wo = convOutDim(cse.wi, cse.padW, cse.dilW, cse.s, cse.strideW);
    if(Ho <= 0 || Wo <= 0)
    {
        std::cerr << "[Sweep] " << cse.name << ": invalid output dims, skipping\n";
        return {false, 0, 0};
    }

    const std::int64_t cpg = cse.c / cse.g;

    std::string arch_bare = stripArchFeatures(props.gcnArchName);

    const std::vector<Candidate> candidates = enumerateCandidates(cse, Ho, Wo, props.gcnArchName);
    std::cerr << "[Sweep] " << cse.name << " (N=" << cse.n << " G=" << cse.g << " C=" << cse.c
              << " K=" << cse.k << " Hi=" << cse.hi << " Wi=" << cse.wi << " R=" << cse.r
              << " S=" << cse.s << ")" << " sweeping " << candidates.size()
              << " pre-validated candidates\n";

    const double kFlops = 2.0 * (double)cse.n * (double)Ho * (double)Wo * (double)cse.k
                          * (double)cpg * (double)cse.r * (double)cse.s;

    static constexpr int kExecTimeoutS = 5;
    const auto compileTimeout = std::chrono::seconds(candidateTimeoutS);

    // Launch all compiles in parallel (bounded by gCompileSem).
    // Compile is CPU-only; exec is GPU-only and runs sequentially below.
    struct CompileSlot
    {
        std::size_t idx;
        std::future<CompileResult> fut;
    };
    std::vector<CompileSlot> slots;
    slots.reserve(candidates.size());
    for(std::size_t i = 0; i < candidates.size(); ++i)
        slots.push_back(
            {i,
             std::async(
                 std::launch::async, boundedCompile, cse, Ho, Wo, candidates[i], arch_bare)});

    std::vector<std::future<CompileResult>> compileGraveyard;

    std::size_t ok = 0, errors = 0, timeouts = 0, compileTimeouts = 0;
    std::size_t prevalidRejects = 0;
    for(auto& slot : slots)
    {
        const auto& cand = candidates[slot.idx];

        if(slot.fut.wait_for(compileTimeout) == std::future_status::timeout)
        {
            ++compileTimeouts;
            std::cerr << "[Sweep]   COMPILE_TIMEOUT: " << cse.name << " tile(" << cand.tile_m << ","
                      << cand.tile_n << "," << cand.tile_k << ") pipeline=" << cand.pipeline
                      << " exceeded " << candidateTimeoutS << "s compile limit\n";
            compileGraveyard.push_back(std::move(slot.fut));
            continue;
        }

        CompileResult cr = slot.fut.get();
        std::cerr << "[Sweep]   COMPILED: " << cse.name << " tile(" << cand.tile_m << ","
                  << cand.tile_n << "," << cand.tile_k << ") pipeline=" << cand.pipeline
                  << " lower=" << cr.lower_ms << "ms comgr=" << cr.comgr_ms << "ms"
                  << (cr.is_error ? " FAILED" : "") << "\n";

        if(cr.is_error)
        {
            ++errors;
            continue;
        }
        if(cr.grid[0] == 0 || cr.grid[1] == 0)
        {
            std::cerr << "[Sweep]   SKIP: zero grid for " << cse.name << "\n";
            continue;
        }

        // Pre-validate — load HSACO + check resource limits (no GPU dispatch).
        if(!preValidateCandidate(cr))
        {
            ++prevalidRejects;
            std::cerr << "[Sweep]   PREVALID_REJECT: " << cse.name << " tile(" << cand.tile_m << ","
                      << cand.tile_n << "," << cand.tile_k << ") pipeline=" << cand.pipeline
                      << " (block_size=" << cr.block_size << " exceeds maxThreadsPerBlock)\n";
            continue;
        }

        // Time via fork+exec helper (isolated HIP context, sequential).
        TimingResult res = execLaunchAndTime(cr, cse, Ho, Wo, kExecTimeoutS);

        if(res.timed_out)
        {
            ++timeouts;
            std::cerr << "[Sweep]   EXEC_TIMEOUT: " << cse.name << " tile(" << cand.tile_m << ","
                      << cand.tile_n << "," << cand.tile_k << ") pipeline=" << cand.pipeline
                      << " exceeded " << kExecTimeoutS << "s (killed)"
                      << " (compile: lower=" << cr.lower_ms << "ms comgr=" << cr.comgr_ms
                      << "ms)\n";
            continue;
        }
        if(res.is_error)
        {
            ++errors;
            std::cerr << "[Sweep]   EXEC_ERROR: " << cse.name << " tile(" << cand.tile_m << ","
                      << cand.tile_n << "," << cand.tile_k << ") pipeline=" << cand.pipeline
                      << "\n";
            continue;
        }
        if(!res.latency_us)
            continue;
        ++ok;
        double tflops = kFlops / (*res.latency_us * 1e-6) / 1e12;
        gCsvWriter.write(cse.n,
                         cse.g,
                         cse.c,
                         cse.k,
                         cse.hi,
                         cse.wi,
                         cse.r,
                         cse.s,
                         cse.strideH,
                         cse.strideW,
                         cse.padH,
                         cse.padW,
                         cand.tile_m,
                         cand.tile_n,
                         cand.tile_k,
                         cand.pipeline,
                         tflops,
                         *res.latency_us);
    }

    std::cerr << "[Sweep] " << cse.name << " done: " << ok << " ok";
    if(prevalidRejects > 0)
        std::cerr << ", " << prevalidRejects << " pre-validation rejects";
    if(compileTimeouts > 0)
        std::cerr << ", " << compileTimeouts << " compile timeouts (>" << candidateTimeoutS << "s)";
    if(timeouts > 0)
        std::cerr << ", " << timeouts << " exec timeouts (>" << kExecTimeoutS << "s, killed)";
    if(errors > 0)
        std::cerr << ", " << errors << " errors";
    std::cerr << "\n";
    return {ok > 0, errors, timeouts + compileTimeouts};
}

} // namespace

// ── Entry point ───────────────────────────────────────────────────────────────

int sweepMain(const std::string& shapesPath,
              const std::string& outPath,
              const std::string& dtype,
              int candidateTimeoutS,
              int compileThreads)
{
    if(dtype != "fp16")
    {
        std::cerr << "ERROR: --dtype must be fp16 (got: " << dtype << ")\n";
        return 1;
    }

    gHelperBin = findHelperBinary();
    if(gHelperBin.empty())
    {
        std::cerr << "ERROR: rocke_kern_time helper binary not found. "
                  << "Set ROCKE_KERN_TIME_BIN or place it next to rocke_conv_sweep.\n";
        return 1;
    }
    std::cerr << "[Sweep] helper binary: " << gHelperBin << "\n";

    const std::vector<ConvCase> shapes = loadShapesFromCsv(shapesPath);
    if(shapes.empty())
    {
        std::cerr << "ERROR: no shapes loaded from " << shapesPath << "\n";
        return 1;
    }

    gCsvWriter.open(outPath);
    if(!gCsvWriter.active)
        return 1;

    if(compileThreads <= 0)
    {
        int hw = (int)std::thread::hardware_concurrency();
        compileThreads = std::clamp(hw / 4, 2, 8);
    }
    CompileSemaphore sem(compileThreads);
    gCompileSem = &sem;

    hipDeviceProp_t props{};
    if(hipGetDeviceProperties(&props, 0) != hipSuccess)
    {
        std::cerr << "ERROR: hipGetDeviceProperties failed\n";
        return 1;
    }
    std::cerr << "[Sweep] device: " << props.name << " (" << props.gcnArchName << ")\n";
    std::cerr << "[Sweep] per-candidate timeout: " << candidateTimeoutS << "s\n";
    std::cerr << "[Sweep] compile threads: " << compileThreads << "\n";

    // HIP context warmup
    {
        void* dummy = nullptr;
        if(hipMalloc(&dummy, 4) != hipSuccess)
            std::cerr << "[Sweep] WARNING: HIP context warmup hipMalloc failed (non-fatal)\n";
        if(hipDeviceSynchronize() != hipSuccess)
            std::cerr << "[Sweep] WARNING: HIP context warmup sync failed (non-fatal)\n";
        (void)hipFree(dummy);
    }

    std::size_t shapesOk = 0, shapesNoData = 0, totalErrors = 0, totalTimeouts = 0;
    for(const auto& shape : shapes)
    {
        ShapeSweepResult r = runConvSweep(shape, props, candidateTimeoutS);
        if(r.produced_data)
            ++shapesOk;
        else
            ++shapesNoData;
        totalErrors += r.errors;
        totalTimeouts += r.timeouts;
    }

    std::cerr << "=== Sweep complete: " << shapesOk << "/" << shapes.size()
              << " shapes produced data";
    if(shapesNoData > 0)
        std::cerr << ", " << shapesNoData << " produced no data";
    if(totalTimeouts > 0)
        std::cerr << ", " << totalTimeouts << " candidates timed out (>" << candidateTimeoutS
                  << "s)";
    if(totalErrors > 0)
        std::cerr << ", " << totalErrors
                  << " CANDIDATE ERRORS (pre-validated but compile/run failed — TRIAGE REQUIRED)";
    std::cerr << " ===\n";

    gCompileSem = nullptr;

    if(totalErrors > 0)
    {
        std::cerr << "ERROR: " << totalErrors << " candidate(s) passed is_valid_spec "
                  << "but failed in the compiler or on-device. This indicates a gap "
                  << "in the validator. Review the TRIAGE lines above.\n";
        return 2;
    }
    return shapesNoData > 0 ? 1 : 0;
}
