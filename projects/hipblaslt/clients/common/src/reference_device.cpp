// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "reference_device.hpp"
#include "hipblaslt_ostream.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

// This translation unit is compiled through hipcc/clang (the client common
// library links hip::device, which injects `-x hip`), so the __global__ kernels
// below build as device code.

namespace
{
    // hipMemset/return-on-failure helper. GPU-reference failures are unexpected;
    // log loudly rather than silently reporting a false "pass".
    inline bool gpu_ref_hip_check(hipError_t err, const char* what)
    {
        if(err != hipSuccess)
        {
            hipblaslt_cerr << "gpu_ref: " << what << " failed: " << hipGetErrorString(err)
                           << std::endl;
            return false;
        }
        return true;
    }

    __device__ inline void atomicMaxDouble(double* addr, double val)
    {
        auto*              as_ull  = reinterpret_cast<unsigned long long*>(addr);
        unsigned long long old     = *as_ull;
        unsigned long long assumed = old;
        do
        {
            assumed     = old;
            double cur  = __longlong_as_double(assumed);
            if(cur >= val)
                break;
            old = atomicCAS(as_ull, assumed, __double_as_longlong(val));
        } while(assumed != old);
    }

    template <typename T>
    __device__ inline float to_float(T v)
    {
        return static_cast<float>(v);
    }

    // Reproduces googletest FloatingPoint<float>::AlmostEquals (<= 4 ULP), which
    // is what unit_check_general()/ASSERT_FLOAT_EQ apply per element on the CPU
    // path (f16 outputs are promoted to float first, matching ASSERT_HALF_EQ).
    // Used only for exact (tol==0) unit_check so the GPU path is no stricter than
    // the CPU reference for f32 -- the two accumulate K in different orders and
    // are not bit-identical.
    __device__ inline bool float_almost_equals(float a, float b)
    {
        if(isnan(a) || isnan(b))
            return false;

        auto to_biased = [](float f) -> unsigned int {
            unsigned int bits;
            __builtin_memcpy(&bits, &f, sizeof(bits));
            constexpr unsigned int sign = 0x80000000u;
            return (bits & sign) ? (~bits + 1u) : (bits | sign);
        };
        const unsigned int ba   = to_biased(a);
        const unsigned int bb   = to_biased(b);
        const unsigned int dist = (ba >= bb) ? (ba - bb) : (bb - ba);
        return dist <= 4u;
    }

    //
    // Piece A: naive reference GEMM. One thread per output element, float
    // accumulate. Column-major with the same transpose/leading-dim/batch-stride
    // conventions as cblas_gemm(). Not tuned for performance -- correctness and
    // independence from the library kernel are the only goals.
    //
    template <typename Ti, typename Tc, typename To>
    __global__ void reference_gemm_kernel(bool        transA_is_n,
                                          bool        transB_is_n,
                                          int64_t     M,
                                          int64_t     N,
                                          int64_t     K,
                                          float       alpha,
                                          float       beta,
                                          const Ti*   A,
                                          int64_t     lda,
                                          int64_t     strideA,
                                          const Ti*   B,
                                          int64_t     ldb,
                                          int64_t     strideB,
                                          const Tc*   C,
                                          int64_t     ldc,
                                          int64_t     strideC,
                                          To*         D,
                                          int64_t     ldd,
                                          int64_t     strideD,
                                          int32_t     batchCount)
    {
        const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        const int64_t j = int64_t(blockIdx.y) * blockDim.y + threadIdx.y;
        const int64_t b = blockIdx.z;
        if(i >= M || j >= N || b >= batchCount)
            return;

        const Ti* Ab = A + b * strideA;
        const Ti* Bb = B + b * strideB;
        const Tc* Cb = C + b * strideC;
        To*       Db = D + b * strideD;

        float acc = 0.0f;
        for(int64_t l = 0; l < K; ++l)
        {
            const float a = transA_is_n ? to_float(Ab[i + l * lda]) : to_float(Ab[l + i * lda]);
            const float bv = transB_is_n ? to_float(Bb[l + j * ldb]) : to_float(Bb[j + l * ldb]);
            acc += a * bv;
        }

        float out = alpha * acc;
        if(beta != 0.0f) // beta==0 ignores C even if it holds inf/nan, matching GEMM semantics
            out += beta * to_float(Cb[i + j * ldc]);

        Db[i + j * ldd] = static_cast<To>(out);
    }

    // Device-side reduction accumulator; mirrors GpuRefResult. Zero-initialized
    // via hipMemset before launch (all fields have 0 as their identity).
    struct DevAccum
    {
        double             max_abs_error;
        double             sum_ref_sq;
        double             sum_diff_sq;
        double             allclose_g[GPU_REF_TOL_GRID_N];
        unsigned long long num_unit_fail;
        unsigned long long num_nan_mismatch;
        unsigned long long num_elements;
    };

    //
    // Piece B: compare GPU output against the reference over the valid
    // M x N x batch region (leading-dim padding is skipped). Grid-stride loop
    // with per-thread partials combined via atomics.
    //
    template <typename To>
    __global__ void compare_kernel(const To* gpu,
                                   const To* ref,
                                   int64_t   M,
                                   int64_t   N,
                                   int64_t   ldd,
                                   int64_t   strideD,
                                   int32_t   batchCount,
                                   DevAccum* out)
    {
        const double rtol[GPU_REF_TOL_GRID_N] = {1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1};

        double l_max = 0.0, l_sref = 0.0, l_sdiff = 0.0;
        double l_g[GPU_REF_TOL_GRID_N];
        for(int k = 0; k < GPU_REF_TOL_GRID_N; ++k)
            l_g[k] = 0.0;
        unsigned long long l_unit_fail = 0, l_nan = 0, l_cnt = 0;

        const size_t MN     = size_t(M) * size_t(N);
        const size_t total  = MN * size_t(batchCount);
        const size_t gid    = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        const size_t stride = size_t(gridDim.x) * blockDim.x;

        for(size_t t = gid; t < total; t += stride)
        {
            const size_t b   = t / MN;
            const size_t rem = t % MN;
            const size_t j   = rem / size_t(M);
            const size_t i   = rem % size_t(M);
            const size_t idx = i + j * size_t(ldd) + b * size_t(strideD);

            const float  gf = to_float(gpu[idx]);
            const float  rf = to_float(ref[idx]);
            const double g  = double(gf);
            const double r  = double(rf);
            ++l_cnt;

            if(isnan(g) || isinf(g) || isnan(r) || isinf(r))
            {
                // Matching same-signed infinities count as agreement (the CPU
                // checks early-out on a==b). Any nan, or an inf disagreement, is a
                // failure: flag it and poison the allclose grid so no atol passes.
                // Non-finite values are kept out of the norm sums so they do not
                // silently turn ||ref||_F into nan.
                if(isinf(g) && isinf(r) && g == r)
                {
                    // agreement, contributes nothing
                }
                else
                {
                    ++l_nan;
                    for(int k = 0; k < GPU_REF_TOL_GRID_N; ++k)
                        l_g[k] = INFINITY;
                }
                continue;
            }

            const double d = fabs(g - r);
            if(!float_almost_equals(gf, rf)) // per-element 4-ULP check, matches ASSERT_FLOAT_EQ
                ++l_unit_fail;
            l_max = fmax(l_max, d);
            l_sref += r * r;
            l_sdiff += d * d;
            // allclose tolerance is atol + rtol*|gpu| (matching allclose() in
            // allclose.hpp, which scales by the second/actual operand).
            const double ag = fabs(g);
            for(int k = 0; k < GPU_REF_TOL_GRID_N; ++k)
                l_g[k] = fmax(l_g[k], d - rtol[k] * ag);
        }

        atomicMaxDouble(&out->max_abs_error, l_max);
        atomicAdd(&out->sum_ref_sq, l_sref);
        atomicAdd(&out->sum_diff_sq, l_sdiff);
        for(int k = 0; k < GPU_REF_TOL_GRID_N; ++k)
            atomicMaxDouble(&out->allclose_g[k], l_g[k]);
        atomicAdd(&out->num_unit_fail, l_unit_fail);
        atomicAdd(&out->num_nan_mismatch, l_nan);
        atomicAdd(&out->num_elements, l_cnt);
    }

    // Launch the reference GEMM for concrete element types.
    template <typename Ti, typename Tc, typename To>
    void launch_reference_gemm(bool        transA_is_n,
                               bool        transB_is_n,
                               int64_t     M,
                               int64_t     N,
                               int64_t     K,
                               float       alpha,
                               float       beta,
                               const void* dA,
                               int64_t     lda,
                               int64_t     strideA,
                               const void* dB,
                               int64_t     ldb,
                               int64_t     strideB,
                               const void* dC,
                               int64_t     ldc,
                               int64_t     strideC,
                               void*       dDgold,
                               int64_t     ldd,
                               int64_t     strideD,
                               int32_t     batchCount,
                               hipStream_t stream)
    {
        const dim3 block(16, 16, 1);
        const dim3 grid(uint32_t((M + block.x - 1) / block.x),
                        uint32_t((N + block.y - 1) / block.y),
                        uint32_t(batchCount));
        reference_gemm_kernel<Ti, Tc, To><<<grid, block, 0, stream>>>(transA_is_n,
                                                                      transB_is_n,
                                                                      M,
                                                                      N,
                                                                      K,
                                                                      alpha,
                                                                      beta,
                                                                      static_cast<const Ti*>(dA),
                                                                      lda,
                                                                      strideA,
                                                                      static_cast<const Ti*>(dB),
                                                                      ldb,
                                                                      strideB,
                                                                      static_cast<const Tc*>(dC),
                                                                      ldc,
                                                                      strideC,
                                                                      static_cast<To*>(dDgold),
                                                                      ldd,
                                                                      strideD,
                                                                      batchCount);
    }

    bool is_f32_or_f16(hipDataType t)
    {
        return t == HIP_R_32F || t == HIP_R_16F;
    }
} // namespace

double GpuRefResult::norm_error() const
{
    // ||gpu - ref||_F / ||ref||_F, matching norm_check_general('F', ...). Note the
    // compare kernel omits agreeing infinities from the sums, so unlike LAPACK
    // xlange this ratio stays finite when the (matching) result contains inf.
    const double tol       = std::numeric_limits<double>::epsilon();
    const double ref_norm  = std::sqrt(sum_ref_sq);
    const double diff_norm = std::sqrt(sum_diff_sq);
    if(std::abs(ref_norm) <= tol && std::abs(diff_norm) <= tol)
        return 0.0;
    return diff_norm / ref_norm;
}

bool gpu_ref_supported(const Arguments& arg, std::string& reason)
{
    auto fail = [&](const char* r) {
        reason = r;
        return false;
    };

    if(arg.grouped_gemm != 0)
        return fail("grouped GEMM");
    if(arg.batch_mode != 0)
        return fail("pointer-array (general) batch mode");
    if(arg.a_type != arg.b_type)
        return fail("mixed A/B input types");
    if(!is_f32_or_f16(arg.a_type) || !is_f32_or_f16(arg.b_type))
        return fail("input type other than f32/f16");
    if(!is_f32_or_f16(arg.c_type) || !is_f32_or_f16(arg.d_type))
        return fail("C/D type other than f32/f16");
    if(arg.compute_type != HIPBLAS_COMPUTE_32F)
        return fail("compute type other than HIPBLAS_COMPUTE_32F");
    if(arg.compute_input_typeA != HIPBLASLT_DATATYPE_INVALID
       || arg.compute_input_typeB != HIPBLASLT_DATATYPE_INVALID)
        return fail("compute-input-type override (e.g. TF32)");
    if(arg.activation_type != hipblaslt_activation_type::none)
        return fail("activation epilogue");
    if(arg.bias_vector)
        return fail("bias epilogue");
    if(arg.gradient)
        return fail("gradient epilogue");
    if(arg.use_e)
        return fail("auxiliary (E) output");
    if(arg.scaleA != hipblaslt_scaling_format::none
       || arg.scaleB != hipblaslt_scaling_format::none)
        return fail("A/B scaling");
    if(arg.scaleC || arg.scaleD || arg.scaleE)
        return fail("C/D/E scaling");
    if(arg.scaleAlpha_vector)
        return fail("alpha vector");
    if(arg.amaxScaleA || arg.amaxScaleB || arg.amaxD)
        return fail("amax");
    if(arg.swizzle_a || arg.swizzle_b)
        return fail("tensor swizzling");
    if(arg.c_equal_d)
        return fail("in-place C==D");
    if(arg.rotating != 0)
        return fail("rotating buffers");
    if(arg.ulp_check)
        // ULP reporting needs the host reference (hD_gold), which the GPU path
        // never populates; reject rather than silently report 0 ULP.
        return fail("ULP check (--ulp)");

    reason.clear();
    return true;
}

void run_reference_gemm_device(bool        transA_is_n,
                               bool        transB_is_n,
                               int64_t     M,
                               int64_t     N,
                               int64_t     K,
                               float       alpha,
                               float       beta,
                               const void* dA,
                               hipDataType tA,
                               int64_t     lda,
                               int64_t     strideA,
                               const void* dB,
                               hipDataType tB,
                               int64_t     ldb,
                               int64_t     strideB,
                               const void* dC,
                               hipDataType tC,
                               int64_t     ldc,
                               int64_t     strideC,
                               void*       dDgold,
                               hipDataType tD,
                               int64_t     ldd,
                               int64_t     strideD,
                               int32_t     batchCount,
                               hipStream_t stream)
{
    if(M <= 0 || N <= 0 || batchCount <= 0)
        return;
    if(tA != tB)
    {
        hipblaslt_cerr << "gpu_ref: mixed A/B types not supported" << std::endl;
        return;
    }

    // Dispatch on (input type, C type, D type). Restricted to f32/f16, which
    // keeps this to a handful of instantiations.
#define GPU_REF_LAUNCH(TI, TC, TO)         \
    launch_reference_gemm<TI, TC, TO>(transA_is_n, \
                                      transB_is_n, \
                                      M,           \
                                      N,           \
                                      K,           \
                                      alpha,       \
                                      beta,        \
                                      dA,          \
                                      lda,         \
                                      strideA,     \
                                      dB,          \
                                      ldb,         \
                                      strideB,     \
                                      dC,          \
                                      ldc,         \
                                      strideC,     \
                                      dDgold,      \
                                      ldd,         \
                                      strideD,     \
                                      batchCount,  \
                                      stream)

#define GPU_REF_DISPATCH_TO(TI, TC)          \
    do                                       \
    {                                        \
        if(tD == HIP_R_32F)                  \
            GPU_REF_LAUNCH(TI, TC, float);   \
        else                                 \
            GPU_REF_LAUNCH(TI, TC, hipblasLtHalf); \
    } while(0)

#define GPU_REF_DISPATCH_TC(TI)                  \
    do                                           \
    {                                            \
        if(tC == HIP_R_32F)                      \
            GPU_REF_DISPATCH_TO(TI, float);      \
        else                                     \
            GPU_REF_DISPATCH_TO(TI, hipblasLtHalf); \
    } while(0)

    if(tA == HIP_R_32F)
        GPU_REF_DISPATCH_TC(float);
    else
        GPU_REF_DISPATCH_TC(hipblasLtHalf);

#undef GPU_REF_DISPATCH_TC
#undef GPU_REF_DISPATCH_TO
#undef GPU_REF_LAUNCH

    gpu_ref_hip_check(hipGetLastError(), "reference GEMM launch");
}

GpuRefResult compare_gemm_device(const void* dGpu,
                                 const void* dRef,
                                 hipDataType tD,
                                 int64_t     M,
                                 int64_t     N,
                                 int64_t     ldd,
                                 int64_t     strideD,
                                 int32_t     batchCount,
                                 hipStream_t stream)
{
    GpuRefResult result;
    if(M <= 0 || N <= 0 || batchCount <= 0)
        return result;

    // The accumulator is tiny and fixed-size; reuse it across comparisons rather
    // than churning the allocator on this hot verification path. It is
    // thread_local so concurrent multi-thread/multi-stream tests
    // (LAUNCH_TEST_ON_THREADS) never share it, and it is reallocated whenever the
    // active device changes so multi-device tests (launch_test_on_streams calls
    // hipSetDevice per device) always accumulate on the same device as the
    // stream. Intentionally not freed -- reclaimed at thread/process exit.
    thread_local DevAccum* dAccum       = nullptr;
    thread_local int       dAccumDevice = -1;
    int                    device       = -1;
    if(!gpu_ref_hip_check(hipGetDevice(&device), "get device"))
        return result;
    if(dAccum == nullptr || device != dAccumDevice)
    {
        if(dAccum)
            hipFree(dAccum); // frees on its owning device regardless of current device
        dAccum = nullptr;
        if(!gpu_ref_hip_check(hipMalloc(&dAccum, sizeof(DevAccum)), "accumulator alloc"))
            return result;
        dAccumDevice = device;
    }
    if(!gpu_ref_hip_check(hipMemsetAsync(dAccum, 0, sizeof(DevAccum), stream), "accumulator zero"))
        return result;

    const int    block = 256;
    const size_t total = size_t(M) * size_t(N) * size_t(batchCount);
    // Cap the grid; the grid-stride loop covers the rest.
    const int grid = int(std::min<size_t>((total + block - 1) / block, size_t(65535)));

    if(tD == HIP_R_32F)
        compare_kernel<float><<<grid, block, 0, stream>>>(static_cast<const float*>(dGpu),
                                                          static_cast<const float*>(dRef),
                                                          M,
                                                          N,
                                                          ldd,
                                                          strideD,
                                                          batchCount,
                                                          dAccum);
    else
        compare_kernel<hipblasLtHalf><<<grid, block, 0, stream>>>(
            static_cast<const hipblasLtHalf*>(dGpu),
            static_cast<const hipblasLtHalf*>(dRef),
            M,
            N,
            ldd,
            strideD,
            batchCount,
            dAccum);

    gpu_ref_hip_check(hipGetLastError(), "compare launch");

    DevAccum hAccum{};
    if(gpu_ref_hip_check(
           hipMemcpyAsync(&hAccum, dAccum, sizeof(DevAccum), hipMemcpyDeviceToHost, stream),
           "accumulator copy-back"))
    {
        gpu_ref_hip_check(hipStreamSynchronize(stream), "compare sync");
        result.max_abs_error    = hAccum.max_abs_error;
        result.sum_ref_sq       = hAccum.sum_ref_sq;
        result.sum_diff_sq      = hAccum.sum_diff_sq;
        result.num_unit_fail    = hAccum.num_unit_fail;
        result.num_nan_mismatch = hAccum.num_nan_mismatch;
        result.num_elements     = hAccum.num_elements;
        for(int k = 0; k < GPU_REF_TOL_GRID_N; ++k)
            result.allclose_g[k] = hAccum.allclose_g[k];
    }

    return result;
}
