// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Shared CK DSL example launcher.
//
// One binary, built once via hipcc against `<hip/hip_runtime.h>` +
// `hipModule_t`. Every CK DSL example (`example/ck_tile/dsl/<N>_*`)
// produces a HSACO blob from Python plus a `manifest.json` describing
// the kernel ABI, recommended grid/block, expected accuracy/perf, and
// the verification mode. This launcher reads both, dispatches the
// kernel, and prints a `Perf:` line that matches the slow path's
// format byte-for-byte so downstream scripts compare directly.
//
// CLI:
//   ck_dsl_launcher <hsaco_path> <manifest_path> [--shape M,N,K] [--verify]

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using fp16 = _Float16;

#define HIP_CHECK(expr)                                                                      \
    do                                                                                       \
    {                                                                                        \
        hipError_t _e = (expr);                                                              \
        if(_e != hipSuccess)                                                                 \
        {                                                                                    \
            std::cerr << "HIP error: " << hipGetErrorString(_e) << " at " << __FILE__ << ":" \
                      << __LINE__ << std::endl;                                              \
            std::exit(2);                                                                    \
        }                                                                                    \
    } while(0)

// ----- tiny JSON probe (only what we need from the manifest) ----------------
//
// We don't link nlohmann/json into this tutorial binary. The manifest is
// produced by our own code so we control its shape; we scan for `"key": value`
// and parse the right side as a string, int, or array of ints.

static std::string read_file(const std::string& path)
{
    std::ifstream f(path);
    if(!f)
    {
        std::cerr << "cannot open " << path << std::endl;
        std::exit(2);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static std::vector<char> read_binary(const std::string& path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if(!f)
    {
        std::cerr << "cannot open " << path << std::endl;
        std::exit(2);
    }
    std::streamsize n = f.tellg();
    f.seekg(0);
    std::vector<char> buf(static_cast<std::size_t>(n));
    if(!f.read(buf.data(), n))
    {
        std::cerr << "cannot read " << path << std::endl;
        std::exit(2);
    }
    return buf;
}

static std::string find_str(const std::string& j, const std::string& key)
{
    auto pos = j.find("\"" + key + "\"");
    if(pos == std::string::npos)
        return {};
    pos = j.find(":", pos);
    if(pos == std::string::npos)
        return {};
    pos = j.find("\"", pos);
    if(pos == std::string::npos)
        return {};
    auto end = j.find("\"", pos + 1);
    if(end == std::string::npos)
        return {};
    return j.substr(pos + 1, end - pos - 1);
}

static long find_int(const std::string& j, const std::string& key, long fallback)
{
    auto pos = j.find("\"" + key + "\"");
    if(pos == std::string::npos)
        return fallback;
    pos = j.find(":", pos);
    if(pos == std::string::npos)
        return fallback;
    return std::strtol(j.c_str() + pos + 1, nullptr, 10);
}

static std::vector<long> find_int_array(const std::string& j, const std::string& key)
{
    std::vector<long> out;
    auto pos = j.find("\"" + key + "\"");
    if(pos == std::string::npos)
        return out;
    pos      = j.find("[", pos);
    auto end = j.find("]", pos);
    if(pos == std::string::npos || end == std::string::npos)
        return out;
    std::string inner = j.substr(pos + 1, end - pos - 1);
    std::stringstream ss(inner);
    std::string tok;
    while(std::getline(ss, tok, ','))
    {
        char* e = nullptr;
        long v  = std::strtol(tok.c_str(), &e, 10);
        if(e != tok.c_str())
            out.push_back(v);
    }
    return out;
}

// ----- problem dispatcher ---------------------------------------------------
//
// The launcher knows about a small set of "problem shapes": gemm_fp16,
// reduce_fp32, etc. The manifest tells us which one we are. Each problem
// owns its argument packing, verification, and metric computation.

struct Problem
{
    virtual ~Problem()                                                  = default;
    virtual void allocate_and_fill()                                    = 0;
    virtual std::vector<char> pack_args()                               = 0;
    virtual void verify()                                               = 0;
    virtual std::pair<unsigned, unsigned> grid_xy(int M_blk, int N_blk) = 0;
    virtual std::tuple<double, double, double> metrics(float ave_ms)    = 0;
};

struct GemmFp16Problem : Problem
{
    int M, N, K;
    std::vector<fp16> A_host, B_host, C_host;
    fp16 *A_dev = nullptr, *B_dev = nullptr, *C_dev = nullptr;
    bool do_verify = false;

    GemmFp16Problem(int m, int n, int k) : M(m), N(n), K(k) {}

    void allocate_and_fill() override
    {
        A_host.resize(static_cast<std::size_t>(M) * K);
        B_host.resize(static_cast<std::size_t>(N) * K);
        C_host.assign(static_cast<std::size_t>(M) * N, fp16(0));
        std::mt19937 rng(0xC0FFEE);
        std::uniform_real_distribution<float> d(-5.f, 5.f);
        for(auto& v : A_host)
            v = static_cast<fp16>(std::round(d(rng)));
        for(auto& v : B_host)
            v = static_cast<fp16>(std::round(d(rng)));
        HIP_CHECK(hipMalloc(&A_dev, A_host.size() * sizeof(fp16)));
        HIP_CHECK(hipMalloc(&B_dev, B_host.size() * sizeof(fp16)));
        HIP_CHECK(hipMalloc(&C_dev, C_host.size() * sizeof(fp16)));
        HIP_CHECK(
            hipMemcpy(A_dev, A_host.data(), A_host.size() * sizeof(fp16), hipMemcpyHostToDevice));
        HIP_CHECK(
            hipMemcpy(B_dev, B_host.data(), B_host.size() * sizeof(fp16), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(C_dev, 0, C_host.size() * sizeof(fp16)));
    }

    std::vector<char> pack_args() override
    {
        // signature: (ptr A, ptr B, ptr C, i32 M, i32 N, i32 K)
        std::vector<char> buf(3 * sizeof(void*) + 3 * sizeof(int));
        char* p = buf.data();
        std::memcpy(p, &A_dev, sizeof(void*));
        p += sizeof(void*);
        std::memcpy(p, &B_dev, sizeof(void*));
        p += sizeof(void*);
        std::memcpy(p, &C_dev, sizeof(void*));
        p += sizeof(void*);
        std::memcpy(p, &M, sizeof(int));
        p += sizeof(int);
        std::memcpy(p, &N, sizeof(int));
        p += sizeof(int);
        std::memcpy(p, &K, sizeof(int));
        return buf;
    }

    void verify() override
    {
        if(!do_verify)
            return;
        HIP_CHECK(
            hipMemcpy(C_host.data(), C_dev, C_host.size() * sizeof(fp16), hipMemcpyDeviceToHost));
        std::vector<fp16> ref(C_host.size(), fp16(0));
        auto t0 = std::chrono::steady_clock::now();
        for(int m = 0; m < M; ++m)
            for(int n = 0; n < N; ++n)
            {
                float a = 0;
                for(int k = 0; k < K; ++k)
                    a += static_cast<float>(A_host[m * K + k]) *
                         static_cast<float>(B_host[n * K + k]);
                ref[m * N + n] = static_cast<fp16>(a);
            }
        auto t1     = std::chrono::steady_clock::now();
        int bad     = 0;
        float worst = 0;
        for(std::size_t i = 0; i < C_host.size(); ++i)
        {
            float g = static_cast<float>(C_host[i]);
            float r = static_cast<float>(ref[i]);
            float d = std::abs(g - r);
            worst   = std::max(worst, d);
            if(d > 16.f)
                ++bad;
        }
        std::cerr << "verify cpu_ref=" << std::chrono::duration<double>(t1 - t0).count()
                  << "s max_abs_diff=" << worst << " bad=" << bad << "/" << C_host.size()
                  << std::endl;
        if(bad)
        {
            std::cerr << "VERIFY FAIL" << std::endl;
            std::exit(1);
        }
    }

    std::pair<unsigned, unsigned> grid_xy(int M_blk, int N_blk) override
    {
        return {static_cast<unsigned>((N + N_blk - 1) / N_blk),
                static_cast<unsigned>((M + M_blk - 1) / M_blk)};
    }

    std::tuple<double, double, double> metrics(float ave_ms) override
    {
        double flop  = 2.0 * (double)M * N * K;
        double bytes = (double)sizeof(fp16) * ((double)M * K + (double)N * K + (double)M * N);
        return {ave_ms, flop / 1e9 / ave_ms, bytes / 1e6 / ave_ms};
    }

    ~GemmFp16Problem() override
    {
        if(A_dev)
            hipFree(A_dev);
        if(B_dev)
            hipFree(B_dev);
        if(C_dev)
            hipFree(C_dev);
    }
};

// ----- conv_fp16: NHWC × KRSC -> NHWK direct/implicit-GEMM conv --------------
//
// One launcher class for both bake-offs:
//   - bake-off 1: implicit-GEMM convolution. Manifest packs:
//       kind = "conv_fp16"
//       conv_layout = "implicit_gemm"      (default)
//       conv = {N, Hi, Wi, C, K, R, S, sH, sW, pH, pW, dH, dW}
//       grid: (M_tiles, N_tiles, 1)  where
//          M_tiles = ceil(N*Ho*Wo / tile_m),
//          N_tiles = ceil(K_out / tile_n)
//       kernel sig: (ptr A_nhwc, ptr B_krsc, ptr D_nhwk,
//                    i32 A_bytes, i32 B_bytes, i32 D_bytes)
//
//   - bake-off 2: direct grouped convolution (handled later). Manifest packs:
//       kind = "conv_fp16"
//       conv_layout = "direct_grouped"
//       conv = {N, H, W, groups, cpg, kpg, KH, KW, PAD, BLOCK_GROUPS, BLOCK_Q}

struct ConvFp16Problem : Problem
{
    int N, Hi, Wi, C, Kc, R, S;
    int groups = 1, cpg = 0, kpg = 0;
    int sH = 1, sW = 1, pH = 0, pW = 0, dH = 1, dW = 1;
    int Ho = 0, Wo = 0;
    std::vector<fp16> A_host, B_host, D_host;
    fp16 *A_dev = nullptr, *B_dev = nullptr, *D_dev = nullptr;
    bool do_verify = false;
    // Whether the kernel signature includes (A_bytes, B_bytes, D_bytes)
    // as i32 params (implicit-GEMM does; direct conv variants may not).
    bool sig_has_bytes = true;

    ConvFp16Problem(int n,
                    int hi,
                    int wi,
                    int c,
                    int k,
                    int r,
                    int s,
                    int sh,
                    int sw,
                    int ph,
                    int pw,
                    int dh,
                    int dw,
                    int groups_ = 1,
                    int cpg_    = 0,
                    int kpg_    = 0)
        : N(n),
          Hi(hi),
          Wi(wi),
          C(c),
          Kc(k),
          R(r),
          S(s),
          groups(groups_),
          cpg(cpg_ ? cpg_ : c),
          kpg(kpg_ ? kpg_ : k),
          sH(sh),
          sW(sw),
          pH(ph),
          pW(pw),
          dH(dh),
          dW(dw)
    {
        if(groups <= 0 || cpg <= 0 || kpg <= 0 || groups * cpg != C || groups * kpg != Kc)
        {
            std::cerr << "invalid conv grouping: groups=" << groups << " cpg=" << cpg
                      << " kpg=" << kpg << " C=" << C << " K=" << Kc << std::endl;
            std::exit(2);
        }
        Ho = (Hi + 2 * pH - dH * (R - 1) - 1) / sH + 1;
        Wo = (Wi + 2 * pW - dW * (S - 1) - 1) / sW + 1;
    }

    size_t in_size() const { return (size_t)N * Hi * Wi * C; }
    size_t wei_size() const { return (size_t)Kc * R * S * cpg; }
    size_t out_size() const { return (size_t)N * Ho * Wo * Kc; }

    long long flops() const { return 2LL * N * Ho * Wo * Kc * (long long)(R * S * cpg); }

    void allocate_and_fill() override
    {
        A_host.resize(in_size());
        B_host.resize(wei_size());
        D_host.assign(out_size(), fp16(0));
        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> d(-1.f, 1.f);
        // Small values to stay inside fp16 dynamic range after R*S*C accumulations.
        const float scale = 0.02f;
        for(auto& v : A_host)
            v = static_cast<fp16>(d(rng) * scale);
        for(auto& v : B_host)
            v = static_cast<fp16>(d(rng) * scale);
        HIP_CHECK(hipMalloc(&A_dev, A_host.size() * sizeof(fp16)));
        HIP_CHECK(hipMalloc(&B_dev, B_host.size() * sizeof(fp16)));
        HIP_CHECK(hipMalloc(&D_dev, D_host.size() * sizeof(fp16)));
        HIP_CHECK(
            hipMemcpy(A_dev, A_host.data(), A_host.size() * sizeof(fp16), hipMemcpyHostToDevice));
        HIP_CHECK(
            hipMemcpy(B_dev, B_host.data(), B_host.size() * sizeof(fp16), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(D_dev, 0, D_host.size() * sizeof(fp16)));
    }

    std::vector<char> pack_args() override
    {
        // Signature: (ptr A, ptr B, ptr D, [i32 A_bytes, i32 B_bytes, i32 D_bytes])
        size_t sz = 3 * sizeof(void*);
        if(sig_has_bytes)
            sz += 3 * sizeof(int);
        std::vector<char> buf(sz);
        char* p = buf.data();
        std::memcpy(p, &A_dev, sizeof(void*));
        p += sizeof(void*);
        std::memcpy(p, &B_dev, sizeof(void*));
        p += sizeof(void*);
        std::memcpy(p, &D_dev, sizeof(void*));
        p += sizeof(void*);
        if(sig_has_bytes)
        {
            int Ab = static_cast<int>(in_size() * sizeof(fp16));
            int Bb = static_cast<int>(wei_size() * sizeof(fp16));
            int Db = static_cast<int>(out_size() * sizeof(fp16));
            std::memcpy(p, &Ab, sizeof(int));
            p += sizeof(int);
            std::memcpy(p, &Bb, sizeof(int));
            p += sizeof(int);
            std::memcpy(p, &Db, sizeof(int));
        }
        return buf;
    }

    void verify() override
    {
        if(!do_verify)
            return;
        HIP_CHECK(
            hipMemcpy(D_host.data(), D_dev, D_host.size() * sizeof(fp16), hipMemcpyDeviceToHost));
        // fp32 CPU reference, NHWC × KRSC -> NHWK
        std::vector<fp16> ref(D_host.size(), fp16(0));
        auto idxA = [&](int n, int h, int w, int c) {
            return ((size_t)n * Hi + h) * Wi * C + (size_t)w * C + c;
        };
        auto idxB = [&](int k, int r, int s, int c_local) {
            return ((size_t)k * R + r) * S * cpg + (size_t)s * cpg + c_local;
        };
        auto idxD = [&](int n, int ho, int wo, int k) {
            return ((size_t)n * Ho + ho) * Wo * Kc + (size_t)wo * Kc + k;
        };
        auto t0 = std::chrono::steady_clock::now();
#pragma omp parallel for collapse(3)
        for(int n = 0; n < N; ++n)
            for(int ho = 0; ho < Ho; ++ho)
                for(int wo = 0; wo < Wo; ++wo)
                    for(int k = 0; k < Kc; ++k)
                    {
                        int group = k / kpg;
                        float acc = 0;
                        for(int r = 0; r < R; ++r)
                        {
                            int hi = ho * sH - pH + r * dH;
                            if(hi < 0 || hi >= Hi)
                                continue;
                            for(int s = 0; s < S; ++s)
                            {
                                int wi = wo * sW - pW + s * dW;
                                if(wi < 0 || wi >= Wi)
                                    continue;
                                for(int c_local = 0; c_local < cpg; ++c_local)
                                {
                                    int c_global = group * cpg + c_local;
                                    acc += static_cast<float>(A_host[idxA(n, hi, wi, c_global)]) *
                                           static_cast<float>(B_host[idxB(k, r, s, c_local)]);
                                }
                            }
                        }
                        ref[idxD(n, ho, wo, k)] = static_cast<fp16>(acc);
                    }
        auto t1        = std::chrono::steady_clock::now();
        int bad        = 0;
        double worst   = 0;
        double sum_abs = 0;
        int printed    = 0;
        for(std::size_t i = 0; i < D_host.size(); ++i)
        {
            double g = static_cast<float>(D_host[i]);
            double r = static_cast<float>(ref[i]);
            double d = std::abs(g - r);
            worst    = std::max(worst, d);
            sum_abs += d;
            if(d > 1e-2)
            {
                ++bad;
                if(printed < 8)
                {
                    // For NHWK layout
                    size_t k_flat = i % Kc;
                    size_t hw_n   = i / Kc;
                    size_t w_     = hw_n % Wo;
                    size_t h_n    = hw_n / Wo;
                    size_t ho_    = h_n % Ho;
                    size_t n_     = h_n / Ho;
                    std::cerr << "  bad i=" << i << " (n=" << n_ << " ho=" << ho_ << " wo=" << w_
                              << " k=" << k_flat << "): gpu=" << g << " ref=" << r << " diff=" << d
                              << std::endl;
                    ++printed;
                }
            }
        }
        double mean = sum_abs / D_host.size();
        // Use `max_abs_diff=` (matching the gemm path) so the
        // example-runner test in `python/test/test_ck_dsl_examples.py`
        // can grep for a single canonical correctness sentinel.
        std::cerr << "verify cpu_ref=" << std::chrono::duration<double>(t1 - t0).count()
                  << "s max_abs_diff=" << worst << " mean_abs=" << mean << " bad=" << bad << "/"
                  << D_host.size() << std::endl;
        if(bad)
        {
            std::cerr << "VERIFY FAIL" << std::endl;
            std::exit(1);
        }
    }

    std::pair<unsigned, unsigned> grid_xy(int M_blk, int N_blk) override
    {
        // M_blk indexes the implicit-GEMM M axis (N*Ho*Wo).
        // N_blk indexes the K_out axis.
        int M = N * Ho * Wo;
        return {static_cast<unsigned>((M + M_blk - 1) / M_blk),
                static_cast<unsigned>((Kc + N_blk - 1) / N_blk)};
        // NOTE: this returns (M_tiles, N_tiles). The launcher swaps
        // (gx, gy) below based on the kernel's expectation. For
        // implicit-GEMM where block.y is the M tile and block.x is
        // the N tile, we want gx = N_tiles, gy = M_tiles. The main
        // function will permute via the "grid_order" manifest field.
    }

    std::tuple<double, double, double> metrics(float ave_ms) override
    {
        double f = (double)flops();
        // Conservative bytes: A + B + D (one-pass).
        double bytes = (double)sizeof(fp16) * (in_size() + wei_size() + out_size());
        return {ave_ms, f / 1e9 / ave_ms, bytes / 1e6 / ave_ms};
    }

    ~ConvFp16Problem() override
    {
        if(A_dev)
            hipFree(A_dev);
        if(B_dev)
            hipFree(B_dev);
        if(D_dev)
            hipFree(D_dev);
    }
};

// ---- main -----------------------------------------------------------------

int main(int argc, char* argv[])
{
    if(argc < 3)
    {
        std::cerr << "usage: launcher <hsaco_path> <manifest_path> [--shape M,N,K] [--verify]"
                  << std::endl;
        return 2;
    }
    std::string hsaco_path    = argv[1];
    std::string manifest_path = argv[2];
    int override_M = -1, override_N = -1, override_K = -1;
    bool verify_flag = false;

    for(int i = 3; i < argc; ++i)
    {
        std::string a = argv[i];
        if(a == "--verify")
            verify_flag = true;
        else if(a == "--shape" && i + 1 < argc)
        {
            std::string s = argv[++i];
            std::replace(s.begin(), s.end(), ',', ' ');
            std::stringstream ss(s);
            ss >> override_M >> override_N >> override_K;
        }
    }

    std::string m           = read_file(manifest_path);
    std::string kind        = find_str(m, "kind");
    std::string kernel_name = find_str(m, "kernel_name");
    std::string grid_order  = find_str(m, "grid_order"); // "MN" (default) or "NM"
    if(grid_order.empty())
        grid_order = "MN";
    long block_m   = find_int(m, "block_m", 32);
    long block_n   = find_int(m, "block_n", 32);
    long threads   = find_int(m, "threads_per_block", 256);
    long iters     = find_int(m, "timed_iters", 100);
    long warmup    = find_int(m, "warmup_iters", 5);
    auto def_shape = find_int_array(m, "default_shape");
    long M         = (override_M > 0) ? override_M : (def_shape.size() >= 1 ? def_shape[0] : 4096);
    long N         = (override_N > 0) ? override_N : (def_shape.size() >= 2 ? def_shape[1] : 4096);
    long K         = (override_K > 0) ? override_K : (def_shape.size() >= 3 ? def_shape[2] : 4096);

    std::cout << "*** ck_dsl launcher (kind=" << kind << ") ***" << std::endl;
    std::cout << "hsaco=" << hsaco_path << " kernel=" << kernel_name << " block_m=" << block_m
              << " block_n=" << block_n << " threads=" << threads << " verify=" << verify_flag
              << std::endl;

    auto blob          = read_binary(hsaco_path);
    hipModule_t module = nullptr;
    HIP_CHECK(hipModuleLoadData(&module, blob.data()));
    hipFunction_t fn = nullptr;
    HIP_CHECK(hipModuleGetFunction(&fn, module, kernel_name.c_str()));

    std::unique_ptr<Problem> prob;
    if(kind == "gemm_fp16")
    {
        std::cout << "shape M=" << M << " N=" << N << " K=" << K << std::endl;
        auto p = std::make_unique<GemmFp16Problem>(
            static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
        p->do_verify = verify_flag;
        prob         = std::move(p);
    }
    else if(kind == "conv_fp16")
    {
        auto cv = find_int_array(m, "conv");
        if(cv.size() < 7)
        {
            std::cerr << "manifest 'conv' array must have >=7 ints (N,Hi,Wi,C,K,R,S)" << std::endl;
            return 2;
        }
        int cN = cv[0], cHi = cv[1], cWi = cv[2], cC = cv[3];
        int cK = cv[4], cR = cv[5], cS = cv[6];
        int sH_ = 1, sW_ = 1, pH_ = 0, pW_ = 0, dH_ = 1, dW_ = 1;
        if(cv.size() >= 9)
        {
            sH_ = cv[7];
            sW_ = cv[8];
        }
        if(cv.size() >= 11)
        {
            pH_ = cv[9];
            pW_ = cv[10];
        }
        if(cv.size() >= 13)
        {
            dH_ = cv[11];
            dW_ = cv[12];
        }
        long groups_ = find_int(m, "groups", 1);
        long cpg_    = find_int(m, "cpg", cC / groups_);
        long kpg_    = find_int(m, "kpg", cK / groups_);
        std::cout << "conv N=" << cN << " H=" << cHi << " W=" << cWi << " C=" << cC << " K=" << cK
                  << " R=" << cR << " S=" << cS << " groups=" << groups_ << " cpg=" << cpg_
                  << " kpg=" << kpg_ << " s=(" << sH_ << "," << sW_ << ")" << " p=(" << pH_ << ","
                  << pW_ << ")" << " d=(" << dH_ << "," << dW_ << ")" << std::endl;
        auto p       = std::make_unique<ConvFp16Problem>(cN,
                                                   cHi,
                                                   cWi,
                                                   cC,
                                                   cK,
                                                   cR,
                                                   cS,
                                                   sH_,
                                                   sW_,
                                                   pH_,
                                                   pW_,
                                                   dH_,
                                                   dW_,
                                                   static_cast<int>(groups_),
                                                   static_cast<int>(cpg_),
                                                   static_cast<int>(kpg_));
        p->do_verify = verify_flag;
        // Default kernel signature: 3 ptrs + 3 i32 bytes. The manifest
        // can opt out for kernels that don't pass bytes.
        long sig_has_bytes = find_int(m, "sig_has_bytes", 1);
        p->sig_has_bytes   = sig_has_bytes != 0;
        prob               = std::move(p);
    }
    else
    {
        std::cerr << "unsupported manifest kind: " << kind << std::endl;
        return 2;
    }

    prob->allocate_and_fill();
    auto args                = prob->pack_args();
    std::size_t arg_size     = args.size();
    void* config[]           = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                                args.data(),
                                HIP_LAUNCH_PARAM_BUFFER_SIZE,
                                &arg_size,
                                HIP_LAUNCH_PARAM_END};
    auto [g_first, g_second] = prob->grid_xy(static_cast<int>(block_m), static_cast<int>(block_n));
    unsigned gx = g_first, gy = g_second;
    // grid_order "MN" means block_id.x indexes M tiles, block_id.y indexes N tiles.
    // grid_order "NM" means block_id.x indexes N tiles, block_id.y indexes M tiles.
    if(grid_order == "NM")
    {
        std::swap(gx, gy);
    }
    long gz = find_int(m, "grid_z", 1);
    // Manifest can also bake an explicit (gx, gy, gz) for kernels like
    // the direct-grouped conv whose grid doesn't fit the standard
    // (M_tiles, N_tiles) GEMM pattern (theirs is (Q_tiles, G_tiles, N)).
    auto grid_explicit = find_int_array(m, "grid_explicit");
    if(grid_explicit.size() == 3)
    {
        gx = static_cast<unsigned>(grid_explicit[0]);
        gy = static_cast<unsigned>(grid_explicit[1]);
        gz = grid_explicit[2];
    }

    std::cout << "grid=(" << gx << ", " << gy << ", " << gz << ") " << "block=(" << threads
              << ", 1, 1) " << "grid_order=" << grid_order << std::endl;

    for(long i = 0; i < warmup; ++i)
        HIP_CHECK(hipModuleLaunchKernel(fn,
                                        gx,
                                        gy,
                                        static_cast<unsigned>(gz),
                                        static_cast<unsigned>(threads),
                                        1,
                                        1,
                                        0,
                                        nullptr,
                                        nullptr,
                                        config));
    HIP_CHECK(hipDeviceSynchronize());

    prob->verify();

    // ----- per-launch benchmark -----
    hipEvent_t e0, e1;
    HIP_CHECK(hipEventCreate(&e0));
    HIP_CHECK(hipEventCreate(&e1));
    HIP_CHECK(hipEventRecord(e0));
    for(long i = 0; i < iters; ++i)
        HIP_CHECK(hipModuleLaunchKernel(fn,
                                        gx,
                                        gy,
                                        static_cast<unsigned>(gz),
                                        static_cast<unsigned>(threads),
                                        1,
                                        1,
                                        0,
                                        nullptr,
                                        nullptr,
                                        config));
    HIP_CHECK(hipEventRecord(e1));
    HIP_CHECK(hipEventSynchronize(e1));
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
    float ave                        = ms / iters;
    auto [ms_per_iter, tflops, gbps] = prob->metrics(ave);
    std::cout << "Perf: " << ms_per_iter << " ms, " << tflops << " TFlops, " << gbps << " GB/s"
              << std::endl;

    // ----- HIP graph benchmark (amortises launch overhead) -----
    // Captures N kernel launches into one hipGraph and replays it
    // R times. The graph path eliminates per-launch driver overhead
    // (~3-5us on MI300), which dominates for small kernels.
    auto graph_bench = [&](int iters_per_replay, int replays, const char* label) {
        // Create a hipGraph by capturing a sequence of launches.
        hipStream_t stream;
        HIP_CHECK(hipStreamCreate(&stream));
        HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
        for(int i = 0; i < iters_per_replay; ++i)
            HIP_CHECK(hipModuleLaunchKernel(fn,
                                            gx,
                                            gy,
                                            static_cast<unsigned>(gz),
                                            static_cast<unsigned>(threads),
                                            1,
                                            1,
                                            0,
                                            stream,
                                            nullptr,
                                            config));
        hipGraph_t graph{};
        HIP_CHECK(hipStreamEndCapture(stream, &graph));
        hipGraphExec_t exec{};
        HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        HIP_CHECK(hipGraphDestroy(graph));

        for(int i = 0; i < 3; ++i)
            HIP_CHECK(hipGraphLaunch(exec, stream));
        HIP_CHECK(hipStreamSynchronize(stream));

        hipEvent_t s0, s1;
        HIP_CHECK(hipEventCreate(&s0));
        HIP_CHECK(hipEventCreate(&s1));
        HIP_CHECK(hipEventRecord(s0, stream));
        for(int r = 0; r < replays; ++r)
            HIP_CHECK(hipGraphLaunch(exec, stream));
        HIP_CHECK(hipEventRecord(s1, stream));
        HIP_CHECK(hipEventSynchronize(s1));
        float total = 0;
        HIP_CHECK(hipEventElapsedTime(&total, s0, s1));
        float per        = total / (replays * iters_per_replay);
        auto [m, g, _gb] = prob->metrics(per);
        std::cout << "Perf(" << label << " " << replays << "x" << iters_per_replay << "): " << m
                  << " ms, " << g << " TFlops" << std::endl;
        HIP_CHECK(hipEventDestroy(s0));
        HIP_CHECK(hipEventDestroy(s1));
        HIP_CHECK(hipGraphExecDestroy(exec));
        HIP_CHECK(hipStreamDestroy(stream));
    };
    graph_bench(1, 1000, "graph");
    graph_bench(10, 200, "graph");
    graph_bench(200, 5, "graph");

    HIP_CHECK(hipEventDestroy(e0));
    HIP_CHECK(hipEventDestroy(e1));
    HIP_CHECK(hipModuleUnload(module));
    return 0;
}
