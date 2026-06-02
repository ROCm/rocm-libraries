// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/*
 * fp64_emul_accuracy.cpp
 *
 * FP64 emulation accuracy and runtime benchmark.
 *
 * Methodology mirrors GEMMul8 (Ozaki, Uchino, Imamura, arXiv:2504.08009):
 *
 *   Matrix distribution:
 *     A[i] = (U(0,1) - 0.5) * exp(N(0,1) * phi)   for phi >= 0
 *     A[i] = N(0,1)                                 for phi < 0
 *   phi_list = {0.5, 1, 2, 4}
 *
 *   Reference: double-double (DD) GEMM on GPU (~106-bit mantissa).
 *
 *   Error metric (per element, GEMMul8-identical):
 *     |err[i]| = |(D[i] - D_exact[i]) / D_exact[i]|   (DD arithmetic)
 *   Reports err_max (maximum) and err_med (median over all NxN elements).
 *
 *   Comparisons:
 *     1. hipBLASLt native DGEMM  (HIPBLAS_COMPUTE_64F)
 *     2. fp64EmulatedGemm() with num_moduli = min_s .. max_s  (default 2..20)
 *
 * This file must be compiled as HIP (it contains __global__ kernels).
 * The CMakeLists.txt sets LANGUAGE HIP for it.
 */

/* Internal hipBLASLt header — declares fp64EmulatedGemm, Fp64EmulationSettings,
 * fp64EmulationWorkspaceSize, etc.  Requires ROCBLASLT_INTERNAL_API define. */
#include "fp64_emulation.hpp"

#include <hipblaslt/hipblaslt.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

/* =========================================================================
 * Error-check macros
 * ========================================================================= */
#define HIP_CHECK(expr)                                                        \
    do {                                                                       \
        hipError_t _e = (expr);                                                \
        if(_e != hipSuccess) {                                                 \
            std::fprintf(stderr, "HIP error %s:%d: %s\n",                     \
                         __FILE__, __LINE__, hipGetErrorString(_e));           \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while(0)

#define HLT_CHECK(expr)                                                        \
    do {                                                                       \
        hipblasStatus_t _s = (expr);                                           \
        if(_s != HIPBLAS_STATUS_SUCCESS) {                                     \
            std::fprintf(stderr, "hipBLASLt error %s:%d: status=%d\n",        \
                         __FILE__, __LINE__, static_cast<int>(_s));            \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while(0)

/* =========================================================================
 * Double-double primitives — host + device.
 * Mirrors GEMMul8/testing/eval.hpp  (namespace dd).
 * ========================================================================= */

#ifndef M_PI
static constexpr double M_PI = 3.14159265358979323846;
#endif

/* On device, IEEE 754 is enforced by default (no -ffast-math) so the pragma
 * is both unnecessary and harmful — it marks inlined primitives as optnone,
 * making the DD GEMM kernel orders of magnitude slower.
 * On host it remains a useful safety net. */
#ifndef __HIP_DEVICE_COMPILE__
# pragma clang optimize off
#endif

__host__ __device__ __forceinline__
void two_sum(double a, double b, double& s, double& e)
{
    s = a + b;
    double v = s - a;
    double u = s - v;
    e = (a - u) + (b - v);
}

__host__ __device__ __forceinline__
void fast_two_sum(double a, double b, double& s, double& e)
{
    /* Requires |a| >= |b| */
    s = a + b;
    e = (a - s) + b;
}

__host__ __device__ __forceinline__
void two_sub(double a, double b, double& s, double& e)
{
    s = a - b;
    double v = s - a;
    double u = s - v;
    e = (a - u) - (b + v);
}

__host__ __device__ __forceinline__
void two_prod(double a, double b, double& p, double& e)
{
    p = a * b;
#ifdef __HIP_DEVICE_COMPILE__
    e = __fma_rn(a, b, -p);
#else
    e = std::fma(a, b, -p);
#endif
}

#ifndef __HIP_DEVICE_COMPILE__
# pragma clang optimize on
#endif

/* ── double2 = (hi, lo), |lo| << |hi| ── */

__host__ __device__ __forceinline__
double2 dd_add(double2 a, double2 b)
{
    double2 c;
    two_sum(a.x, b.x, c.x, c.y);
    c.y += a.y + b.y;
    fast_two_sum(c.x, c.y, c.x, c.y);
    return c;
}

/* Subtract double2 from double: returns (double - double2) as double2. */
__host__ __device__ __forceinline__
double2 dd_sub(double a, double2 b)
{
    double2 c;
    two_sub(a, b.x, c.x, c.y);
    c.y -= b.y;
    fast_two_sum(c.x, c.y, c.x, c.y);
    return c;
}

/* Multiply two doubles, returning exact product as double2. */
__host__ __device__ __forceinline__
double2 dd_mul(double a, double b)
{
    double2 c;
    two_prod(a, b, c.x, c.y);
    return c;
}

/* Divide double2 by double2 (one Newton step). */
__host__ __device__ __forceinline__
double2 dd_div(double2 a, double2 b)
{
    double q1 = a.x / b.x;
    /* r = a - q1 * b  (double-double) */
    double2 q1b;
    two_prod(q1, b.x, q1b.x, q1b.y);
    q1b.y += q1 * b.y;
    /* r.hi = a.hi - q1b.hi, r.lo = a.lo - q1b.lo (approximate) */
    double rhi, re;
    two_sub(a.x, q1b.x, rhi, re);
    double rlo = re + (a.y - q1b.y);
    double q2  = rhi / b.x + rlo / b.x;
    double2 result;
    fast_two_sum(q1, q2, result.x, result.y);
    return result;
}

/* =========================================================================
 * GPU random-number helpers
 * 64-bit xorshift + Box-Muller transform.
 * Seeds follow GEMMul8: seedA=12345, seedB=54321.
 * ========================================================================= */

static constexpr uint64_t SEED_A = 12345ULL;
static constexpr uint64_t SEED_B = 54321ULL;

__device__ __forceinline__
uint64_t xorshift64(uint64_t s)
{
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
}

/* Map 64-bit integer to double in (0, 1) exclusive. */
__device__ __forceinline__
double bits_to_uniform(uint64_t bits)
{
    /* top 53 bits → [0, 1) shifted slightly above 0 to avoid log(0) */
    return static_cast<double>(bits >> 11) * (1.0 / 9007199254740992.0)
           + 1e-300;
}

/* Box-Muller cosine branch: (u1, u2) in (0,1) → standard normal. */
__device__ __forceinline__
double box_muller(double u1, double u2)
{
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/* =========================================================================
 * randmat_kernel
 *
 * Fills n_elems doubles with the GEMMul8 distribution (phi parameter):
 *   phi < 0  → N(0, 1)
 *   phi >= 0 → (U(0,1) - 0.5) * exp(N(0,1) * phi)
 * ========================================================================= */
__global__ static void
randmat_kernel(size_t n_elems, double* __restrict__ A, double phi, uint64_t seed)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= n_elems) return;

    /* Per-element unique state — mix seed with index using a Knuth multiplier */
    uint64_t s = seed ^ (idx * 0x9e3779b97f4a7c15ULL + 1442695040888963407ULL);
    s = xorshift64(s);
    s = xorshift64(s);   /* additional mixing */

    uint64_t b0 = xorshift64(s);
    uint64_t b1 = xorshift64(b0);
    uint64_t b2 = xorshift64(b1);

    const double u0 = bits_to_uniform(b0);   /* Box-Muller input 1 */
    const double u1 = bits_to_uniform(b1);   /* Box-Muller input 2 */
    const double u2 = bits_to_uniform(b2);   /* uniform for (rand-0.5) part */

    const double randn = box_muller(u0, u1);
    A[idx] = (phi < 0.0) ? randn : ((u2 - 0.5) * exp(randn * phi));
}

/* =========================================================================
 * dd_gemm_kernel
 *
 * Computes C_dd[m × n] = A[m × k] * B[k × n] in double-double arithmetic.
 * All matrices column-major with natural leading dimensions (ld = first dim).
 * Output: double2 per element (hi, lo).
 *
 * Tiled 32×32 shared-memory kernel, mirrors GEMMul8's simple_gemm_device.
 *
 * Grid:  ((n + TILE-1)/TILE, (m + TILE-1)/TILE)
 * Block: (TILE, TILE)
 * ========================================================================= */
static constexpr int DD_TILE = 32;

__global__ static void
dd_gemm_kernel(size_t m, size_t n, size_t k,
               const double* __restrict__ A,    /* col-major, ld = m */
               const double* __restrict__ B,    /* col-major, ld = k */
               double2* __restrict__      C_dd) /* col-major, ld = m */
{
    const size_t row = static_cast<size_t>(blockIdx.y) * DD_TILE + threadIdx.y;
    const size_t col = static_cast<size_t>(blockIdx.x) * DD_TILE + threadIdx.x;

    __shared__ double Asub[DD_TILE][DD_TILE + 1]; /* +1 avoids bank conflicts */
    __shared__ double Bsub[DD_TILE][DD_TILE + 1];

    double2 sum = {0.0, 0.0};

    const int num_tiles = static_cast<int>((k + DD_TILE - 1) / DD_TILE);

    for(int t = 0; t < num_tiles; ++t) {
        /* A[row, t*TILE + threadIdx.x] */
        const size_t a_col = static_cast<size_t>(t * DD_TILE) + threadIdx.x;
        Asub[threadIdx.y][threadIdx.x] =
            (row < m && a_col < k) ? A[row + a_col * m] : 0.0;

        /* B[t*TILE + threadIdx.y, col] */
        const size_t b_row = static_cast<size_t>(t * DD_TILE) + threadIdx.y;
        Bsub[threadIdx.y][threadIdx.x] =
            (b_row < k && col < n) ? B[b_row + col * k] : 0.0;

        __syncthreads();

#pragma unroll
        for(int i = 0; i < DD_TILE; ++i) {
            /* sum += Asub[row_local][i] * Bsub[i][col_local]  (double-double) */
            sum = dd_add(sum, dd_mul(Asub[threadIdx.y][i], Bsub[i][threadIdx.x]));
        }
        __syncthreads();
    }

    if(row < m && col < n)
        C_dd[row + col * m] = sum;
}

/* =========================================================================
 * gemm_err_kernel
 *
 * Overwrites D_tmp[idx] (double, in-place) with the componentwise relative
 * error vs the double-double reference:
 *   |err[idx]| = |(D[idx] - C_exact[idx]) / C_exact[idx]|  (hi part of DD)
 *
 * Mirrors GEMMul8/testing/eval.hpp :: gemm_err_kernel.
 * ========================================================================= */
__global__ static void
gemm_err_kernel(size_t n_elems,
                double* __restrict__        D_tmp,  /* in: computed; out: |err| */
                const double2* __restrict__ C_dd)   /* double-double reference  */
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= n_elems) return;

    const double2 ref = C_dd[idx];
    if(ref.x == 0.0 && ref.y == 0.0) {
        /* Exact zero — report |D| as absolute error (edge case) */
        D_tmp[idx] = fabs(D_tmp[idx]);
        return;
    }

    const double2 gap = dd_sub(D_tmp[idx], ref);       /* D - C_exact  (DD)  */
    const double2 err = dd_div(gap, ref);              /* gap / C_exact (DD)  */
    D_tmp[idx] = fabs(err.x);                         /* |relative error| hi */
}

/* =========================================================================
 * Host launch wrappers
 * ========================================================================= */

static void launch_randmat(size_t m, size_t n, double* d_A, double phi,
                           uint64_t seed, hipStream_t stream)
{
    const size_t n_elems = m * n;
    const unsigned blk   = 256u;
    const unsigned grd   = static_cast<unsigned>((n_elems + blk - 1) / blk);
    hipLaunchKernelGGL(randmat_kernel, dim3(grd), dim3(blk), 0, stream,
                       n_elems, d_A, phi, seed);
}

static void launch_dd_gemm(size_t N,
                           const double* d_A, const double* d_B,
                           double2* d_C_dd, hipStream_t stream)
{
    const dim3 block(DD_TILE, DD_TILE);
    const dim3 grid(static_cast<unsigned>((N + DD_TILE - 1) / DD_TILE),
                    static_cast<unsigned>((N + DD_TILE - 1) / DD_TILE));
    hipLaunchKernelGGL(dd_gemm_kernel, grid, block, 0, stream,
                       N, N, N, d_A, d_B, d_C_dd);
}

/**
 * Compute errors of d_D vs d_C_dd, return {err_max, err_med}.
 * Uses d_err as a temporary device buffer (must be N*N doubles).
 * h_err is a pre-allocated host buffer of size N*N used for sorting.
 */
static std::pair<double, double>
compute_errors(size_t N,
               const double*  d_D,
               const double2* d_C_dd,
               double*        d_err,
               std::vector<double>& h_err,
               hipStream_t stream)
{
    const size_t n_elems = N * N;

    /* d_err = copy of d_D (kernel overwrites it) */
    HIP_CHECK(hipMemcpyAsync(d_err, d_D, n_elems * sizeof(double),
                             hipMemcpyDeviceToDevice, stream));

    const unsigned blk = 256u;
    const unsigned grd = static_cast<unsigned>((n_elems + blk - 1) / blk);
    hipLaunchKernelGGL(gemm_err_kernel, dim3(grd), dim3(blk), 0, stream,
                       n_elems, d_err, d_C_dd);

    HIP_CHECK(hipMemcpyAsync(h_err.data(), d_err, n_elems * sizeof(double),
                             hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    std::sort(h_err.begin(), h_err.begin() + static_cast<ptrdiff_t>(n_elems));

    const double err_max = h_err[n_elems - 1];
    const double err_med = (n_elems & 1u)
                         ? h_err[n_elems / 2]
                         : 0.5 * (h_err[n_elems / 2 - 1] + h_err[n_elems / 2]);
    return {err_max, err_med};
}

/* =========================================================================
 * Native DGEMM via hipBLASLt  (HIPBLAS_COMPUTE_64F)
 * ========================================================================= */
struct NativeDgemm {
    hipblasLtHandle_t            handle  = nullptr;
    hipblasLtMatmulDesc_t        desc    = nullptr;
    hipblasLtMatrixLayout_t      layoutA = nullptr;
    hipblasLtMatrixLayout_t      layoutB = nullptr;
    hipblasLtMatrixLayout_t      layoutD = nullptr;
    hipblasLtMatmulPreference_t  pref    = nullptr;
    hipblasLtMatmulHeuristicResult_t heur{};
    bool hasAlgo = false;

    void init(int64_t N)
    {
        HLT_CHECK(hipblasLtCreate(&handle));

        /* Matmul descriptor: native FP64 */
        HLT_CHECK(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_64F, HIP_R_64F));
        {
            hipblasOperation_t opN = HIPBLAS_OP_N;
            HLT_CHECK(hipblasLtMatmulDescSetAttribute(
                desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)));
            HLT_CHECK(hipblasLtMatmulDescSetAttribute(
                desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
        }

        /* Square N×N, column-major, ld = N */
        HLT_CHECK(hipblasLtMatrixLayoutCreate(
            &layoutA, HIP_R_64F,
            static_cast<uint64_t>(N), static_cast<uint64_t>(N), N));
        HLT_CHECK(hipblasLtMatrixLayoutCreate(
            &layoutB, HIP_R_64F,
            static_cast<uint64_t>(N), static_cast<uint64_t>(N), N));
        HLT_CHECK(hipblasLtMatrixLayoutCreate(
            &layoutD, HIP_R_64F,
            static_cast<uint64_t>(N), static_cast<uint64_t>(N), N));

        HLT_CHECK(hipblasLtMatmulPreferenceCreate(&pref));

        int cnt = 0;
        hipblasLtMatmulAlgoGetHeuristic(
            handle, desc, layoutA, layoutB, layoutD, layoutD,
            pref, 1, &heur, &cnt);
        hasAlgo = (cnt > 0);
        if(!hasAlgo)
            std::fprintf(stderr,
                "[warning] hipBLASLt: no DGEMM algorithm found — "
                "passing nullptr algo\n");
    }

    void run(const double* A, const double* B, double* D,
             hipStream_t stream) const
    {
        const double alpha = 1.0, beta = 0.0;
        HLT_CHECK(hipblasLtMatmul(
            handle, desc,
            &alpha, A, layoutA, B, layoutB,
            &beta,  D, layoutD, D, layoutD,
            hasAlgo ? &heur.algo : nullptr,
            nullptr, 0, stream));
    }

    void destroy()
    {
        if(pref)    { hipblasLtMatmulPreferenceDestroy(pref);    pref    = nullptr; }
        if(layoutD) { hipblasLtMatrixLayoutDestroy(layoutD);     layoutD = nullptr; }
        if(layoutB) { hipblasLtMatrixLayoutDestroy(layoutB);     layoutB = nullptr; }
        if(layoutA) { hipblasLtMatrixLayoutDestroy(layoutA);     layoutA = nullptr; }
        if(desc)    { hipblasLtMatmulDescDestroy(desc);          desc    = nullptr; }
        if(handle)  { hipblasLtDestroy(handle);                  handle  = nullptr; }
    }
};

/* =========================================================================
 * Timing helper
 * ========================================================================= */

/** Warmup num_warmup calls (no timing), then time num_runs calls.
 *  Returns elapsed milliseconds per run. */
template<typename Fn>
static double run_and_time(Fn fn, unsigned num_warmup, unsigned num_runs,
                           hipStream_t stream)
{
    for(unsigned i = 0; i < num_warmup; ++i) fn();
    HIP_CHECK(hipStreamSynchronize(stream));

    hipEvent_t beg, end;
    HIP_CHECK(hipEventCreate(&beg));
    HIP_CHECK(hipEventCreate(&end));

    HIP_CHECK(hipEventRecord(beg, stream));
    for(unsigned i = 0; i < num_runs; ++i) fn();
    HIP_CHECK(hipEventRecord(end, stream));
    HIP_CHECK(hipEventSynchronize(end));  /* all work through 'end' is done */

    float ms_total = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms_total, beg, end));

    HIP_CHECK(hipEventDestroy(end));
    HIP_CHECK(hipEventDestroy(beg));
    return static_cast<double>(ms_total) / num_runs;
}

/* =========================================================================
 * CRT bit capacity for num_moduli = 2..20 (from fp64_emulation.cpp)
 * Indexed directly by s: CRT_BITS[s], s in {2, ..., 20}.
 * ========================================================================= */
static constexpr double CRT_BITS[21] = {
    0.0,    /* s=0  (unused) */
    0.0,    /* s=1  (unused) */
    15.994, /* s=2  */
    23.976, /* s=3  */
    31.945, /* s=4  */
    39.894, /* s=5  */
    47.807, /* s=6  */
    55.708, /* s=7  */
    63.572, /* s=8  */
    71.411, /* s=9  */
    79.238, /* s=10 */
    87.040, /* s=11 */
    94.801, /* s=12 */
   102.522, /* s=13 */
   110.160, /* s=14 */
   117.782, /* s=15 */
   125.374, /* s=16 */
   132.949, /* s=17 */
   140.448, /* s=18 */
   147.931, /* s=19 */
   155.365, /* s=20 */
};

/* =========================================================================
 * CLI parsing
 * ========================================================================= */
struct Config {
    size_t              N            = 512;
    unsigned            num_runs     = 30;
    unsigned            min_s        = 2;
    unsigned            max_s        = 20;
    std::vector<double> phi_list     = {0.5, 1.0, 2.0, 4.0};
    bool                run_adaptive = true;  /* include adaptive-s run */
};

static void print_usage(const char* prog)
{
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "  -n N           Square matrix size M=N=K (default: 512)\n"
        "  --num-runs R   Timed iterations per config (default: 30)\n"
        "                 Warmup runs = same count as timed runs.\n"
        "  --min-s S      Minimum num_moduli for emulation (default: 2)\n"
        "  --max-s S      Maximum num_moduli for emulation (default: 20)\n"
        "  --phi-list P   Comma-separated phi values\n"
        "                 (default: -1,0,0.5,1,2,4  — same as GEMMul8)\n"
        "  --no-adaptive  Skip the adaptive-s (default settings) run\n"
        "  -h, --help     Print this help and exit\n"
        "\n"
        "Output: CSV with columns  phi,N,algo,crt_bits,err_max,err_med,ms_per_run\n"
        "  algo = 'DGEMM'              native FP64\n"
        "  algo = 'OS2-accu-adaptive'  adaptive s (default settings, s≤16)\n"
        "  algo = 'OS2-accu-sN'        fixed N moduli\n",
        prog);
}

static std::vector<double> parse_phi_list(const std::string& s)
{
    std::vector<double> v;
    size_t pos = 0;
    while(pos < s.size()) {
        size_t next = s.find(',', pos);
        if(next == std::string::npos) {
            v.push_back(std::stod(s.substr(pos)));
            break;
        }
        v.push_back(std::stod(s.substr(pos, next - pos)));
        pos = next + 1;
    }
    return v;
}

static Config parse_args(int argc, char** argv)
{
    Config cfg;
    for(int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if(a == "-h" || a == "--help") {
            print_usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else if((a == "-n") && i + 1 < argc) {
            cfg.N = static_cast<size_t>(std::stoul(argv[++i]));
        } else if((a == "--num-runs") && i + 1 < argc) {
            cfg.num_runs = static_cast<unsigned>(std::stoul(argv[++i]));
        } else if((a == "--min-s") && i + 1 < argc) {
            cfg.min_s = static_cast<unsigned>(std::stoul(argv[++i]));
        } else if((a == "--max-s") && i + 1 < argc) {
            cfg.max_s = static_cast<unsigned>(std::stoul(argv[++i]));
        } else if((a == "--phi-list") && i + 1 < argc) {
            cfg.phi_list = parse_phi_list(argv[++i]);
        } else if(a == "--no-adaptive") {
            cfg.run_adaptive = false;
        } else {
            std::fprintf(stderr, "Unknown option: %s\n", a.c_str());
            print_usage(argv[0]);
            std::exit(EXIT_FAILURE);
        }
    }

    /* Clamp moduli range to valid [2, 20] */
    if(cfg.min_s < 2u)  cfg.min_s = 2u;
    if(cfg.max_s > 20u) cfg.max_s = 20u;
    if(cfg.min_s > cfg.max_s) std::swap(cfg.min_s, cfg.max_s);

    if(cfg.N == 0) {
        std::fprintf(stderr, "N must be > 0\n");
        std::exit(EXIT_FAILURE);
    }
    if(cfg.phi_list.empty()) {
        std::fprintf(stderr, "--phi-list is empty\n");
        std::exit(EXIT_FAILURE);
    }
    return cfg;
}

/* =========================================================================
 * main
 * ========================================================================= */
int main(int argc, char** argv)
{
    Config cfg = parse_args(argc, argv);

    const size_t   N        = cfg.N;
    const size_t   N2       = N * N;
    const unsigned num_runs = cfg.num_runs;

    /* ── Device info ──────────────────────────────────────────────────── */
    {
        hipDeviceProp_t prop{};
        HIP_CHECK(hipGetDeviceProperties(&prop, 0));
        std::fprintf(stderr, "Device : %s\n", prop.name);
    }
    std::fprintf(stderr,
                 "Config : N=%zu  num_runs=%u  moduli=%u..%u  phi_list=[",
                 N, num_runs, cfg.min_s, cfg.max_s);
    for(size_t i = 0; i < cfg.phi_list.size(); ++i)
        std::fprintf(stderr, "%s%.4g", (i ? "," : ""), cfg.phi_list[i]);
    std::fprintf(stderr, "]\n\n");

    /* ── GPU allocations ──────────────────────────────────────────────── */
    double*  d_A    = nullptr;  /* N×N input matrix A          */
    double*  d_B    = nullptr;  /* N×N input matrix B          */
    double*  d_D    = nullptr;  /* N×N GEMM result             */
    double*  d_err  = nullptr;  /* N×N temp for error kernel   */
    double2* d_C_dd = nullptr;  /* N×N double-double reference */

    HIP_CHECK(hipMalloc(&d_A,    N2 * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_B,    N2 * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_D,    N2 * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_err,  N2 * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_C_dd, N2 * sizeof(double2)));

    std::vector<double> h_err(N2);  /* host buffer for sort */

    /* Emulation workspace — allocate for the maximum (20 moduli) */
    size_t ws_bytes = fp64EmulationWorkspaceSize(
        static_cast<int64_t>(N), static_cast<int64_t>(N),
        static_cast<int64_t>(N), 20u);
    void* d_ws = nullptr;
    HIP_CHECK(hipMalloc(&d_ws, ws_bytes));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    /* ── Native DGEMM setup ───────────────────────────────────────────── */
    NativeDgemm native;
    native.init(static_cast<int64_t>(N));

    /* ── CSV header ───────────────────────────────────────────────────── */
    std::printf("phi,N,algo,crt_bits,err_max,err_med,ms_per_run\n");
    std::fflush(stdout);

    /* ── Main sweep ───────────────────────────────────────────────────── */
    const double alpha = 1.0;
    const double beta  = 0.0;

    for(double phi : cfg.phi_list) {

        /* Fill A, B on GPU with GEMMul8 distribution */
        launch_randmat(N, N, d_A, phi, SEED_A, stream);
        launch_randmat(N, N, d_B, phi, SEED_B, stream);
        HIP_CHECK(hipStreamSynchronize(stream));

        /* Compute double-double reference: C_dd = A * B */
        launch_dd_gemm(N, d_A, d_B, d_C_dd, stream);
        HIP_CHECK(hipStreamSynchronize(stream));

        /* ── Native DGEMM ─────────────────────────────────────────────── */
        {
            auto fn = [&]{ native.run(d_A, d_B, d_D, stream); };

            /* Warmup + timed runs; d_D holds the last result after this. */
            double ms = run_and_time(fn, num_runs, num_runs, stream);

            auto [err_max, err_med] = compute_errors(
                N, d_D, d_C_dd, d_err, h_err, stream);

            std::printf("%.4g,%zu,DGEMM,%.1f,%.4e,%.4e,%.3f\n",
                        phi, N, 53.0 /* FP64 mantissa bits */,
                        err_max, err_med, ms);
            std::fflush(stdout);
        }

        /* ── Adaptive-s run: num_moduli=0 → fp64EmulationNumModuli()=16,
         *   then per-call adaptive selection reduces s based on actual
         *   C32i_prelim magnitudes (captures exponent range + inner products).
         *   Upper-bound CRT capacity: CRT_BITS[16] = 125.4 bits.           ── */
        if(cfg.run_adaptive) {
            Fp64EmulationSettings settings;
            settings.num_moduli      = 0u;       /* 0 = use fp64EmulationNumModuli() */
            settings.sv_mask         = 0u;       /* skip Inf/NaN detection */
            settings.workspace       = d_ws;
            settings.workspace_bytes = ws_bytes;

            auto fn = [&]{
                fp64EmulatedGemm(
                    HIPBLAS_OP_N, HIPBLAS_OP_N,
                    static_cast<int64_t>(N),
                    static_cast<int64_t>(N),
                    static_cast<int64_t>(N),
                    &alpha,
                    d_A, static_cast<int64_t>(N),
                    d_B, static_cast<int64_t>(N),
                    &beta,
                    d_D, static_cast<int64_t>(N),
                    d_D, static_cast<int64_t>(N),
                    stream, settings);
            };

            double ms = run_and_time(fn, num_runs, num_runs, stream);
            /* d_D holds the result of the last timed call */

            auto [err_max, err_med] = compute_errors(
                N, d_D, d_C_dd, d_err, h_err, stream);

            /* Report crt_bits as the upper-bound (default s=16 → 125.4 bits) */
            std::printf("%.4g,%zu,OS2-accu-adaptive,%.1f,%.4e,%.4e,%.3f\n",
                        phi, N, CRT_BITS[16], err_max, err_med, ms);
            std::fflush(stdout);
        }

        /* ── Emulation sweep over num_moduli = min_s .. max_s ─────────── */
        for(unsigned s = cfg.min_s; s <= cfg.max_s; ++s) {
            Fp64EmulationSettings settings;
            settings.num_moduli      = s;
            settings.sv_mask         = 0u;      /* skip Inf/NaN detection */
            settings.workspace       = d_ws;
            settings.workspace_bytes = ws_bytes;

            auto fn = [&]{
                fp64EmulatedGemm(
                    HIPBLAS_OP_N, HIPBLAS_OP_N,
                    static_cast<int64_t>(N),
                    static_cast<int64_t>(N),
                    static_cast<int64_t>(N),
                    &alpha,
                    d_A, static_cast<int64_t>(N),
                    d_B, static_cast<int64_t>(N),
                    &beta,
                    d_D, static_cast<int64_t>(N),
                    d_D, static_cast<int64_t>(N),
                    stream, settings);
            };

            double ms = run_and_time(fn, num_runs, num_runs, stream);
            /* d_D now holds the result of the last timed call. */

            auto [err_max, err_med] = compute_errors(
                N, d_D, d_C_dd, d_err, h_err, stream);

            char algo_name[32];
            std::snprintf(algo_name, sizeof(algo_name), "OS2-accu-s%u", s);

            std::printf("%.4g,%zu,%s,%.1f,%.4e,%.4e,%.3f\n",
                        phi, N, algo_name, CRT_BITS[s],
                        err_max, err_med, ms);
            std::fflush(stdout);
        }
    }

    /* ── Cleanup ──────────────────────────────────────────────────────── */
    native.destroy();
    HIP_CHECK(hipFree(d_ws));
    HIP_CHECK(hipFree(d_C_dd));
    HIP_CHECK(hipFree(d_err));
    HIP_CHECK(hipFree(d_D));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipStreamDestroy(stream));

    return EXIT_SUCCESS;
}
