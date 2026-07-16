// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gpu_compare.hpp"
#include "hipblaslt_ostream.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

// Compiled through hipcc/clang (the client common library links hip::device,
// which injects `-x hip`), so the __global__ kernel below builds as device code.

namespace
{
    // Log and return false on HIP error rather than silently reporting a pass.
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
            assumed    = old;
            double cur = __longlong_as_double(assumed);
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

    // googletest FloatingPoint<float>::AlmostEquals (<= 4 ULP), which ASSERT_FLOAT_EQ
    // (and thus unit_check) applies per element; f16 outputs are promoted to float
    // first. Drives the exact (tol==0) unit_check.
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

    // Device reduction accumulator; zero-initialized via hipMemset before launch.
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

    // Grid-stride compare over the valid M x N x batch region (ld padding skipped);
    // per-thread partials combined via atomics.
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
                // Matching same-signed infinities agree; any nan or inf disagreement
                // is a failure that also poisons the allclose grid. Non-finite values
                // stay out of the norm sums so ||ref||_F does not become nan.
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
            if(!float_almost_equals(gf, rf))
                ++l_unit_fail;
            l_max = fmax(l_max, d);
            l_sref += r * r;
            l_sdiff += d * d;
            // allclose tolerance is atol + rtol*|gpu| (allclose() scales by the actual operand).
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
} // namespace

double GpuRefResult::norm_error() const
{
    // ||gpu - ref||_F / ||ref||_F. Agreeing infinities are excluded from the sums,
    // so this stays finite when the (matching) result contains inf.
    const double tol       = std::numeric_limits<double>::epsilon();
    const double ref_norm  = std::sqrt(sum_ref_sq);
    const double diff_norm = std::sqrt(sum_diff_sq);
    if(std::abs(ref_norm) <= tol && std::abs(diff_norm) <= tol)
        return 0.0;
    return diff_norm / ref_norm;
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

    // Reuse a small accumulator across comparisons. thread_local keeps concurrent
    // multi-thread/multi-stream tests from sharing it; it is reallocated when the
    // active device changes so it always lives on the same device as the stream.
    // Intentionally not freed -- reclaimed at thread/process exit.
    thread_local DevAccum* dAccum       = nullptr;
    thread_local int       dAccumDevice = -1;
    int                    device       = -1;
    if(!gpu_ref_hip_check(hipGetDevice(&device), "get device"))
        return result;
    if(dAccum == nullptr || device != dAccumDevice)
    {
        if(dAccum)
            hipFree(dAccum);
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
