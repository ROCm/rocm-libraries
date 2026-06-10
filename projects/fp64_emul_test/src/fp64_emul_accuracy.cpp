// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/*
 * fp64_emul_accuracy.cpp
 *
 * FP64 emulation accuracy and runtime benchmark.
 *
 * Uses only the public hipBLASLt API to invoke both native FP64 GEMM and
 * FP64 emulation via Ozaki Scheme II.  No internal headers are included.
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
 *     1. hipBLASLt native DGEMM  (HIPBLAS_COMPUTE_64F, emulation disabled)
 *     2. FP64 emulation with num_moduli = min_s .. max_s  (default 2..18)
 *     3. FP64 emulation adaptive-s (library default settings)
 *
 * This file must be compiled as HIP (it contains __global__ kernels).
 * The CMakeLists.txt sets LANGUAGE HIP for it.
 */

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

__host__ __device__ __forceinline__
double2 dd_add(double2 a, double2 b)
{
    double2 c;
    two_sum(a.x, b.x, c.x, c.y);
    c.y += a.y + b.y;
    fast_two_sum(c.x, c.y, c.x, c.y);
    return c;
}

__host__ __device__ __forceinline__
double2 dd_sub(double a, double2 b)
{
    double2 c;
    two_sub(a, b.x, c.x, c.y);
    c.y -= b.y;
    fast_two_sum(c.x, c.y, c.x, c.y);
    return c;
}

__host__ __device__ __forceinline__
double2 dd_mul(double a, double b)
{
    double2 c;
    two_prod(a, b, c.x, c.y);
    return c;
}

__host__ __device__ __forceinline__
double2 dd_div(double2 a, double2 b)
{
    double q1 = a.x / b.x;
    double2 q1b;
    two_prod(q1, b.x, q1b.x, q1b.y);
    q1b.y += q1 * b.y;
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
 * ========================================================================= */
/* GEMMul8 uses the same seed for A and B so that A == B (for square
 * k=m=n).  This avoids catastrophic cancellation in A*B that would make
 * per-element relative errors meaningless for near-zero output entries. */
static constexpr uint64_t SEED = 12345ULL;

__device__ __forceinline__
uint64_t xorshift64(uint64_t s)
{
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
}

__device__ __forceinline__
double bits_to_uniform(uint64_t bits)
{
    return static_cast<double>(bits >> 11) * (1.0 / 9007199254740992.0) + 1e-300;
}

__device__ __forceinline__
double box_muller(double u1, double u2)
{
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

__global__ static void
randmat_kernel(size_t n_elems, double* __restrict__ A, double phi, uint64_t seed)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= n_elems) return;

    uint64_t s = seed ^ (idx * 0x9e3779b97f4a7c15ULL + 1442695040888963407ULL);
    s = xorshift64(s);
    s = xorshift64(s);

    uint64_t b0 = xorshift64(s);
    uint64_t b1 = xorshift64(b0);
    uint64_t b2 = xorshift64(b1);

    const double u0 = bits_to_uniform(b0);
    const double u1 = bits_to_uniform(b1);
    const double u2 = bits_to_uniform(b2);

    const double randn = box_muller(u0, u1);
    A[idx] = (phi < 0.0) ? randn : ((u2 - 0.5) * exp(randn * phi));
}

/* =========================================================================
 * dd_gemm_kernel — double-double reference GEMM, all 4 transpose modes.
 *
 * Computes C_dd = op(A) × op(B) where op is N or T.
 * All matrices are square N×N column-major (leading dim = N).
 *
 * transA=false (N): op(A)[i,j] = A[i + j*m]   (A is m×k, lda=m)
 * transA=true  (T): op(A)[i,j] = A[j + i*k]   (A is k×m, lda=k)
 * transB=false (N): op(B)[i,j] = B[i + j*k]   (B is k×n, ldb=k)
 * transB=true  (T): op(B)[i,j] = B[j + i*n]   (B is n×k, ldb=n)
 *
 * For square N×N all leading dims equal N, so the formulas simplify to:
 *   A load: transA ? A[a_col + row*N] : A[row + a_col*N]
 *   B load: transB ? B[col  + b_row*N] : B[b_row + col*N]
 *
 * ========================================================================= */

static constexpr int DD_TILE = 32;

__global__ static void
dd_gemm_kernel(size_t m, size_t n, size_t k,
               const double* __restrict__ A,
               const double* __restrict__ B,
               double2* __restrict__      C_dd,
               bool transA, bool transB)
{
    const size_t row = static_cast<size_t>(blockIdx.y) * DD_TILE + threadIdx.y;
    const size_t col = static_cast<size_t>(blockIdx.x) * DD_TILE + threadIdx.x;

    /* Leading dims for the stored (pre-transpose) matrices (square: all = N). */
    const size_t ldA = transA ? k : m;
    const size_t ldB = transB ? n : k;

    __shared__ double Asub[DD_TILE][DD_TILE + 1];
    __shared__ double Bsub[DD_TILE][DD_TILE + 1];

    double2 sum = {0.0, 0.0};

    const int num_tiles = static_cast<int>((k + DD_TILE - 1) / DD_TILE);

    for(int t = 0; t < num_tiles; ++t) {
        /* a_col: index along the contracted k-dimension */
        const size_t a_col = static_cast<size_t>(t * DD_TILE) + threadIdx.x;
        if(row < m && a_col < k)
            Asub[threadIdx.y][threadIdx.x] = transA
                ? A[a_col + row * ldA]   /* A^T[row, a_col] = A[a_col, row] */
                : A[row   + a_col * ldA]; /* A[row, a_col] */
        else
            Asub[threadIdx.y][threadIdx.x] = 0.0;

        const size_t b_row = static_cast<size_t>(t * DD_TILE) + threadIdx.y;
        if(b_row < k && col < n)
            Bsub[threadIdx.y][threadIdx.x] = transB
                ? B[col   + b_row * ldB] /* B^T[b_row, col] = B[col, b_row] */
                : B[b_row + col   * ldB]; /* B[b_row, col] */
        else
            Bsub[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

#pragma unroll
        for(int i = 0; i < DD_TILE; ++i)
            sum = dd_add(sum, dd_mul(Asub[threadIdx.y][i], Bsub[i][threadIdx.x]));
        __syncthreads();
    }

    if(row < m && col < n)
        C_dd[row + col * m] = sum;
}

__global__ static void
gemm_err_kernel(size_t n_elems,
                double* __restrict__        D_tmp,
                const double2* __restrict__ C_dd)
{
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= n_elems) return;

    const double2 ref = C_dd[idx];
    if(ref.x == 0.0 && ref.y == 0.0) {
        D_tmp[idx] = fabs(D_tmp[idx]);
        return;
    }

    const double2 gap = dd_sub(D_tmp[idx], ref);
    const double2 err = dd_div(gap, ref);
    D_tmp[idx] = fabs(err.x);
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
                           double2* d_C_dd, hipStream_t stream,
                           bool transA, bool transB)
{
    const dim3 block(DD_TILE, DD_TILE);
    const dim3 grid(static_cast<unsigned>((N + DD_TILE - 1) / DD_TILE),
                    static_cast<unsigned>((N + DD_TILE - 1) / DD_TILE));
    hipLaunchKernelGGL(dd_gemm_kernel, grid, block, 0, stream,
                       N, N, N, d_A, d_B, d_C_dd, transA, transB);
}

static std::pair<double, double>
compute_errors(size_t N,
               const double*  d_D,
               const double2* d_C_dd,
               double*        d_err,
               std::vector<double>& h_err,
               hipStream_t stream)
{
    const size_t n_elems = N * N;

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
 * DgemmRunner — wraps a hipBLASLt handle + matmul descriptors for N×N DGEMM.
 * Used for both native (emulation disabled) and emulated (emulation enabled).
 * ========================================================================= */
struct DgemmRunner {
    hipblasLtHandle_t            handle  = nullptr;
    hipblasLtMatmulDesc_t        desc    = nullptr;
    hipblasLtMatrixLayout_t      layoutA = nullptr;
    hipblasLtMatrixLayout_t      layoutB = nullptr;
    hipblasLtMatrixLayout_t      layoutD = nullptr;
    hipblasLtMatmulPreference_t  pref    = nullptr;
    hipblasLtMatmulHeuristicResult_t heur{};
    bool hasAlgo = false;

    size_t workspaceSize() const { return heur.workspaceSize; }

    void requery()
    {
        int cnt = 0;
        hipblasLtMatmulAlgoGetHeuristic(
            handle, desc, layoutA, layoutB, layoutD, layoutD,
            pref, 1, &heur, &cnt);
        hasAlgo = (cnt > 0);
    }

    void init(int64_t N, bool emulation_enabled, size_t workspace_bytes)
    {
        HLT_CHECK(hipblasLtCreate(&handle));

        /* Enable or disable emulation for this handle */
        HLT_CHECK(hipblasLtSetEmulationEnabled(handle, emulation_enabled));

        if(emulation_enabled) {
            /* EAGER: emulate regardless of problem size (bypass AI heuristic) */
            HLT_CHECK(hipblasLtSetEmulationStrategy(handle,
                                                    HIPBLASLT_EMULATION_STRATEGY_EAGER));
            /* Disable Inf/NaN detection for speed (benchmark uses clean data) */
            HLT_CHECK(hipblasLtSetEmulationSpecialValuesSupport(handle, 0u));
        }

        /* Matmul descriptor: FP64 compute and scale type */
        HLT_CHECK(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_64F, HIP_R_64F));
        {
            hipblasOperation_t opN = HIPBLAS_OP_N;
            HLT_CHECK(hipblasLtMatmulDescSetAttribute(
                desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)));
            HLT_CHECK(hipblasLtMatmulDescSetAttribute(
                desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
        }

        /* Square N×N, column-major */
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
        HLT_CHECK(hipblasLtMatmulPreferenceSetAttribute(
            pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_bytes, sizeof(workspace_bytes)));

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

    /* Update transpose attributes on the matmul descriptor and re-query
     * the heuristic (workspace size may change for the emulation path). */
    void update_trans(hipblasOperation_t tA, hipblasOperation_t tB)
    {
        HLT_CHECK(hipblasLtMatmulDescSetAttribute(
            desc, HIPBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA)));
        HLT_CHECK(hipblasLtMatmulDescSetAttribute(
            desc, HIPBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB)));
        requery();
    }

    void run(const double* A, const double* B, double* D,
             void* workspace, size_t workspace_bytes,
             hipStream_t stream) const
    {
        const double alpha = 1.0, beta = 0.0;
        HLT_CHECK(hipblasLtMatmul(
            handle, desc,
            &alpha, A, layoutA, B, layoutB,
            &beta,  D, layoutD, D, layoutD,
            hasAlgo ? &heur.algo : nullptr,
            workspace, workspace_bytes, stream));
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
    HIP_CHECK(hipEventSynchronize(end));

    float ms_total = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms_total, beg, end));

    HIP_CHECK(hipEventDestroy(end));
    HIP_CHECK(hipEventDestroy(beg));
    return static_cast<double>(ms_total) / num_runs;
}

/* =========================================================================
 * CRT bit capacity for num_moduli = 2..18
 * Indexed directly by s: CRT_BITS[s], s in {2, ..., 18}.
 * ========================================================================= */
static constexpr double CRT_BITS[19] = {
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
};

/**
 * bits_for_moduli(s): returns the integer maxBits value that causes the
 * library to select exactly s moduli.
 *
 * The library picks the minimum s' such that CRT_BITS[s'] >= maxBits.
 * To select exactly s, we need:
 *   CRT_BITS[s-1] < maxBits <= CRT_BITS[s]
 * Using maxBits = (int)CRT_BITS[s-1] + 1 satisfies this for s=2..18.
 */
static int bits_for_moduli(unsigned s)
{
    if(s < 2u)  s = 2u;
    if(s > 18u) s = 18u;
    return static_cast<int>(CRT_BITS[s - 1]) + 1;
}

/* =========================================================================
 * CLI parsing
 * ========================================================================= */
struct Config {
    size_t              N            = 512;
    unsigned            num_runs     = 30;
    unsigned            min_s        = 2;
    unsigned            max_s        = 18;
    std::vector<double> phi_list     = {0.5, 1.0, 2.0, 4.0};
    /* Each entry is {transa, transb} where each char is 'N' or 'T'. */
    std::vector<std::pair<char,char>> trans_list = {{'N','N'},{'N','T'},{'T','N'},{'T','T'}};
    bool                run_adaptive = true;
    bool                check_errors = true;
};

static void print_usage(const char* prog)
{
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "  -n N           Square matrix size M=N=K (default: 512)\n"
        "  --num-runs R   Timed iterations per config (default: 30)\n"
        "  --min-s S      Minimum num_moduli for emulation (default: 2)\n"
        "  --max-s S      Maximum num_moduli for emulation (default: 18)\n"
        "  --phi-list P   Comma-separated phi values\n"
        "                 (default: 0.5,1,2,4  — same as GEMMul8)\n"
        "  --trans T      Comma-separated transpose combinations to run\n"
        "                 Each is two chars from {N,T}: NN,NT,TN,TT\n"
        "                 (default: NN,NT,TN,TT — all four)\n"
        "  --no-adaptive  Skip the adaptive-s (library-default) run\n"
        "  --no-check     Skip the double-double reference GEMM and error\n"
        "                 computation.  err_max and err_med are printed as 'nan'.\n"
        "                 Useful for fast timing-only sweeps at large N.\n"
        "  -h, --help     Print this help and exit\n"
        "\n"
        "Output: CSV columns: phi,N,transa,transb,algo,crt_bits,err_max,err_med,ms_per_run\n"
        "  algo = 'DGEMM'              native FP64 (emulation disabled)\n"
        "  algo = 'OS2-accu-adaptive'  adaptive s (library default, s<=16)\n"
        "  algo = 'OS2-accu-sN'        fixed N moduli\n",
        prog);
}

static std::vector<std::pair<char,char>> parse_trans_list(const std::string& s)
{
    std::vector<std::pair<char,char>> v;
    size_t pos = 0;
    while(pos <= s.size()) {
        const size_t next = s.find(',', pos);
        const std::string t = (next == std::string::npos)
                              ? s.substr(pos)
                              : s.substr(pos, next - pos);
        if(t.size() >= 2) {
            const char ta = static_cast<char>(std::toupper(static_cast<unsigned char>(t[0])));
            const char tb = static_cast<char>(std::toupper(static_cast<unsigned char>(t[1])));
            if((ta == 'N' || ta == 'T') && (tb == 'N' || tb == 'T'))
                v.push_back({ta, tb});
        }
        if(next == std::string::npos) break;
        pos = next + 1;
    }
    return v;
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
        } else if((a == "--trans") && i + 1 < argc) {
            cfg.trans_list = parse_trans_list(argv[++i]);
        } else if(a == "--no-adaptive") {
            cfg.run_adaptive = false;
        } else if(a == "--no-check") {
            cfg.check_errors = false;
        } else {
            std::fprintf(stderr, "Unknown option: %s\n", a.c_str());
            print_usage(argv[0]);
            std::exit(EXIT_FAILURE);
        }
    }

    if(cfg.min_s < 2u)  cfg.min_s = 2u;
    if(cfg.max_s > 18u) cfg.max_s = 18u;
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
    double*  d_A    = nullptr;
    double*  d_B    = nullptr;
    double*  d_D    = nullptr;
    double*  d_err  = nullptr;
    double2* d_C_dd = nullptr;

    HIP_CHECK(hipMalloc(&d_A,    N2 * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_B,    N2 * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_D,    N2 * sizeof(double)));
    if(cfg.check_errors) {
        HIP_CHECK(hipMalloc(&d_err,  N2 * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_C_dd, N2 * sizeof(double2)));
    }

    std::vector<double> h_err(cfg.check_errors ? N2 : 0u);

    /*
     * Workspace budget offered to hipblasLtMatmulAlgoGetHeuristic.
     * Setting this large lets the heuristic choose the optimal algorithm;
     * the actual memory allocated per-run equals heur.workspaceSize, which
     * the emulation library computes exactly from the problem size and s.
     */
    constexpr size_t WS_BUDGET = size_t(-1);    /* no limit — heuristic picks best algo;
                                                 * actual memory allocated = heur.workspaceSize */
    size_t ws_bytes = 0;                          /* actual bytes currently allocated */
    void*  d_ws     = nullptr;

    /* Grow the workspace buffer lazily to match what the heuristic requires. */
    auto ensure_ws = [&](size_t needed) {
        if(needed > ws_bytes) {
            HIP_CHECK(hipFree(d_ws));
            ws_bytes = needed;
            HIP_CHECK(hipMalloc(&d_ws, ws_bytes));
        }
    };

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    /* ── hipBLASLt runner setup ───────────────────────────────────────── */
    /* Native DGEMM: emulation explicitly disabled */
    DgemmRunner native;
    native.init(static_cast<int64_t>(N), /*emulation_enabled=*/false, WS_BUDGET);
    ensure_ws(native.workspaceSize());   /* native DGEMM workspace (typically 0) */

    /* Emulated DGEMM: emulation enabled, EAGER strategy, no Inf/NaN check */
    DgemmRunner emulated;
    emulated.init(static_cast<int64_t>(N), /*emulation_enabled=*/true, WS_BUDGET);
    ensure_ws(emulated.workspaceSize());   /* emulation workspace for default num_moduli */

    /* ── CSV header ───────────────────────────────────────────────────── */
    std::printf("phi,N,transa,transb,algo,crt_bits,err_max,err_med,ms_per_run,workspace_MiB\n");
    std::fflush(stdout);

    /* ── Main sweep: outer = transpose combination, inner = phi ──────── */
    for(const auto& tc : cfg.trans_list) {
        const char cTA = tc.first;
        const char cTB = tc.second;
        const hipblasOperation_t tA = (cTA == 'T') ? HIPBLAS_OP_T : HIPBLAS_OP_N;
        const hipblasOperation_t tB = (cTB == 'T') ? HIPBLAS_OP_T : HIPBLAS_OP_N;
        const bool bTA = (cTA == 'T');
        const bool bTB = (cTB == 'T');

        /* Update both runners to use the new transpose combination. */
        native.update_trans(tA, tB);
        emulated.update_trans(tA, tB);
        ensure_ws(native.workspaceSize());
        ensure_ws(emulated.workspaceSize());

        std::fprintf(stderr, "=== transA=%c transB=%c ===\n", cTA, cTB);

    for(double phi : cfg.phi_list) {

        /* Fill A, B on GPU */
        launch_randmat(N, N, d_A, phi, SEED, stream);
        launch_randmat(N, N, d_B, phi, SEED, stream);
        HIP_CHECK(hipStreamSynchronize(stream));

        /* Double-double reference (skipped when --no-check) */
        if(cfg.check_errors) {
            launch_dd_gemm(N, d_A, d_B, d_C_dd, stream, bTA, bTB);
            HIP_CHECK(hipStreamSynchronize(stream));
        }

        /* ── Native DGEMM ─────────────────────────────────────────────── */
        {
            ensure_ws(native.workspaceSize());
            auto fn = [&]{ native.run(d_A, d_B, d_D, d_ws, ws_bytes, stream); };
            double ms = run_and_time(fn, num_runs, num_runs, stream);

            double err_max = std::nan(""), err_med = std::nan("");
            if(cfg.check_errors)
                std::tie(err_max, err_med) = compute_errors(
                    N, d_D, d_C_dd, d_err, h_err, stream);

            std::printf("%.4g,%zu,%c,%c,DGEMM,%.1f,%.4e,%.4e,%.3f,0.000\n",
                        phi, N, cTA, cTB, 53.0, err_max, err_med, ms);
            std::fflush(stdout);
        }

        /* ── Adaptive-s emulation run ─────────────────────────────────── */
        if(cfg.run_adaptive) {
            HLT_CHECK(hipblasLtSetFixedPointEmulationMantissaControl(
                emulated.handle, HIPBLAS_EMULATION_MANTISSA_CONTROL_DYNAMIC));
            emulated.requery();
            const size_t emu_ws = emulated.workspaceSize();
            ensure_ws(emu_ws);

            auto fn = [&]{ emulated.run(d_A, d_B, d_D, d_ws, ws_bytes, stream); };
            double ms = run_and_time(fn, num_runs, num_runs, stream);

            double err_max = std::nan(""), err_med = std::nan("");
            if(cfg.check_errors)
                std::tie(err_max, err_med) = compute_errors(
                    N, d_D, d_C_dd, d_err, h_err, stream);

            std::printf("%.4g,%zu,%c,%c,OS2-accu-adaptive,%.1f,%.4e,%.4e,%.3f,%.3f\n",
                        phi, N, cTA, cTB, CRT_BITS[16],
                        err_max, err_med, ms, emu_ws / (1024.0 * 1024.0));
            std::fflush(stdout);
        }

        /* ── Emulation sweep over num_moduli = min_s .. max_s ─────────── */
        for(unsigned s = cfg.min_s; s <= cfg.max_s; ++s) {
            HLT_CHECK(hipblasLtSetFixedPointEmulationMantissaControl(
                emulated.handle, HIPBLAS_EMULATION_MANTISSA_CONTROL_FIXED));
            HLT_CHECK(hipblasLtSetFixedPointEmulationMaxMantissaBitCount(
                emulated.handle, bits_for_moduli(s)));
            emulated.requery();
            const size_t emu_ws = emulated.workspaceSize();
            ensure_ws(emu_ws);

            auto fn = [&]{ emulated.run(d_A, d_B, d_D, d_ws, ws_bytes, stream); };
            double ms = run_and_time(fn, num_runs, num_runs, stream);

            double err_max = std::nan(""), err_med = std::nan("");
            if(cfg.check_errors)
                std::tie(err_max, err_med) = compute_errors(
                    N, d_D, d_C_dd, d_err, h_err, stream);

            char algo_name[32];
            std::snprintf(algo_name, sizeof(algo_name), "OS2-accu-s%u", s);

            std::printf("%.4g,%zu,%c,%c,%s,%.1f,%.4e,%.4e,%.3f,%.3f\n",
                        phi, N, cTA, cTB, algo_name, CRT_BITS[s],
                        err_max, err_med, ms, emu_ws / (1024.0 * 1024.0));
            std::fflush(stdout);
        }
    } /* phi loop */
    } /* trans loop */

    /* ── Cleanup ──────────────────────────────────────────────────────── */
    emulated.destroy();
    native.destroy();
    HIP_CHECK(hipFree(d_ws));
    if(cfg.check_errors) {
        HIP_CHECK(hipFree(d_C_dd));
        HIP_CHECK(hipFree(d_err));
    }
    HIP_CHECK(hipFree(d_D));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipStreamDestroy(stream));

    return EXIT_SUCCESS;
}
