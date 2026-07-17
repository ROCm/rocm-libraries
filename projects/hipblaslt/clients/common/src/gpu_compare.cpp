// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gpu_compare.hpp"
#include "hipblaslt_ostream.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

// Compiled through hipcc/clang (the client common library links hip::device,
// which injects `-x hip`), so the __global__ kernel below builds as device code.

namespace
{
    constexpr int GPU_REF_BLOCK = 256; // threads per block; block reductions assume this

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

    // Per-element error in ULP of the output type, matching ulp_distance() in ulp.hpp.
    __device__ inline double ulp_distance(double exact, double approx, int mant_bits)
    {
        if(exact == approx)
            return 0.0;
        int          e = 0;
        const double m = frexp(exact, &e);
        if(fabs(m) == 0.5) // power-of-2 boundary
            --e;
        const double ulp_size = ldexp(1.0, e - mant_bits);
        return fabs(exact - approx) / ulp_size;
    }

    // Block reductions over GPU_REF_BLOCK threads. One shared buffer per variant,
    // reused across sequential calls; each call brackets its use with syncs.
    __device__ inline double block_reduce_max(double v)
    {
        __shared__ double s[GPU_REF_BLOCK];
        const unsigned    t = threadIdx.x;
        __syncthreads();
        s[t] = v;
        __syncthreads();
        for(unsigned stride = blockDim.x >> 1; stride > 0; stride >>= 1)
        {
            if(t < stride)
                s[t] = fmax(s[t], s[t + stride]);
            __syncthreads();
        }
        return s[0];
    }

    __device__ inline double block_reduce_sum(double v)
    {
        __shared__ double s[GPU_REF_BLOCK];
        const unsigned    t = threadIdx.x;
        __syncthreads();
        s[t] = v;
        __syncthreads();
        for(unsigned stride = blockDim.x >> 1; stride > 0; stride >>= 1)
        {
            if(t < stride)
                s[t] += s[t + stride];
            __syncthreads();
        }
        return s[0];
    }

    __device__ inline unsigned long long block_reduce_sum_ull(unsigned long long v)
    {
        __shared__ unsigned long long s[GPU_REF_BLOCK];
        const unsigned                t = threadIdx.x;
        __syncthreads();
        s[t] = v;
        __syncthreads();
        for(unsigned stride = blockDim.x >> 1; stride > 0; stride >>= 1)
        {
            if(t < stride)
                s[t] += s[t + stride];
            __syncthreads();
        }
        return s[0];
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

    // Device reduction accumulator for the global (cross-batch) metrics;
    // zero-initialized via hipMemset before launch. The per-batch Frobenius sums
    // live in a separate `bins` buffer so each batch's norm can be formed on the
    // host (matching the CPU per-batch ratio, see compare_kernel).
    struct DevAccum
    {
        double             max_abs_error;
        double             allclose_g[GPU_REF_TOL_GRID_N];
        double             max_ulp;
        double             sum_ulp;
        unsigned long long num_unit_fail;
        unsigned long long num_nan_mismatch;
        unsigned long long ulp_count;
    };

    // Compare over the valid M x N x batch region (ld padding skipped). Each block
    // maps to exactly one batch via blockIdx.y (gridDim.y == batchCount) and
    // grid-strides over that batch's M x N via blockIdx.x, so its per-batch
    // Frobenius block-reduce stays within a single batch. thread 0 then issues one
    // atomic per batch into `bins` (bins[b] += sum_ref_sq_b, bins[batchCount+b] +=
    // sum_diff_sq_b), letting the host form the CPU-matching per-batch ratio. The
    // global metrics (max/ulp/allclose/counts) reduce within the block and combine
    // into `out` with one atomic per block.
    template <typename To>
    __global__ void compare_kernel(const To* gpu,
                                   const To* ref,
                                   int64_t   M,
                                   int64_t   N,
                                   int64_t   ldd,
                                   int64_t   strideD,
                                   int32_t   batchCount,
                                   int       mant_bits,
                                   DevAccum* out,
                                   double*   bins)
    {
        const double rtol[GPU_REF_TOL_GRID_N] = {GPU_REF_TOL_GRID[0],
                                                 GPU_REF_TOL_GRID[1],
                                                 GPU_REF_TOL_GRID[2],
                                                 GPU_REF_TOL_GRID[3],
                                                 GPU_REF_TOL_GRID[4],
                                                 GPU_REF_TOL_GRID[5]};
        static_assert(GPU_REF_TOL_GRID_N == 6, "update the rtol initializer above");

        // Block-local metric accumulators, reduced into `out` after the M x N loop.
        double l_max = 0.0, l_max_ulp = 0.0, l_sum_ulp = 0.0;
        double l_g[GPU_REF_TOL_GRID_N];
        for(int k = 0; k < GPU_REF_TOL_GRID_N; ++k)
            l_g[k] = 0.0;
        unsigned long long l_unit_fail = 0, l_nan = 0, l_ulp_cnt = 0;

        const size_t MN      = size_t(M) * size_t(N);
        const size_t xstride = size_t(gridDim.x) * blockDim.x;

        const int64_t b      = blockIdx.y; // one block maps to exactly one batch
        double        l_sref = 0.0, l_sdiff = 0.0; // this batch only
        const size_t  base = size_t(b) * size_t(strideD);

        for(size_t t = size_t(blockIdx.x) * blockDim.x + threadIdx.x; t < MN; t += xstride)
        {
            const size_t j   = t / size_t(M);
            const size_t i   = t % size_t(M);
            const size_t idx = base + i + j * size_t(ldd);

            const float  gf = to_float(gpu[idx]);
            const float  rf = to_float(ref[idx]);
            const double g  = double(gf);
            const double r  = double(rf);

            if(isnan(g) || isinf(g) || isnan(r) || isinf(r))
            {
                // Matching same-signed infinities agree; any nan or inf disagreement
                // is a failure that also poisons the allclose grid. Non-finite values
                // stay out of the norm/ulp sums so they do not become nan.
                if(!(isinf(g) && isinf(r) && g == r))
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

            const double u = ulp_distance(r, g, mant_bits);
            l_max_ulp      = fmax(l_max_ulp, u);
            l_sum_ulp += u;
            ++l_ulp_cnt;
        }

        // Per-batch Frobenius partials: reduce within the block, one atomic per batch.
        double bsref  = block_reduce_sum(l_sref);
        double bsdiff = block_reduce_sum(l_sdiff);
        if(threadIdx.x == 0)
        {
            atomicAdd(&bins[b], bsref);
            atomicAdd(&bins[size_t(batchCount) + b], bsdiff);
        }

        const bool lead = threadIdx.x == 0;

        double bmax = block_reduce_max(l_max);
        if(lead)
            atomicMaxDouble(&out->max_abs_error, bmax);
        for(int k = 0; k < GPU_REF_TOL_GRID_N; ++k)
        {
            double bg = block_reduce_max(l_g[k]);
            if(lead)
                atomicMaxDouble(&out->allclose_g[k], bg);
        }
        double bmaxu = block_reduce_max(l_max_ulp);
        if(lead)
            atomicMaxDouble(&out->max_ulp, bmaxu);
        double bsumu = block_reduce_sum(l_sum_ulp);
        if(lead)
            atomicAdd(&out->sum_ulp, bsumu);
        unsigned long long buf = block_reduce_sum_ull(l_unit_fail);
        if(lead)
            atomicAdd(&out->num_unit_fail, buf);
        unsigned long long bnan = block_reduce_sum_ull(l_nan);
        if(lead)
            atomicAdd(&out->num_nan_mismatch, bnan);
        unsigned long long bcnt = block_reduce_sum_ull(l_ulp_cnt);
        if(lead)
            atomicAdd(&out->ulp_count, bcnt);
    }
} // namespace

double GpuRefResult::norm_error() const
{
    // Sum_b ||diff_b||_F / ||ref_b||_F, formed per batch on the host in
    // compare_gemm_device (agreeing infinities are excluded from the sums, so this
    // stays finite when the matching result contains inf).
    return norm_error_sum;
}

double GpuRefResult::avg_ulp() const
{
    return ulp_count ? sum_ulp / double(ulp_count) : 0.0;
}

GpuRefResult compare_gemm_device(const void* dGpu,
                                 const void* dRef,
                                 hipDataType tD,
                                 int64_t     M,
                                 int64_t     N,
                                 int64_t     ldd,
                                 int64_t     strideD,
                                 int32_t     batchCount,
                                 int         ulpMantBits,
                                 hipStream_t stream)
{
    GpuRefResult result;
    if(M <= 0 || N <= 0 || batchCount <= 0)
        return result;

    // Reuse a small accumulator across comparisons. thread_local keeps concurrent
    // multi-thread/multi-stream tests from sharing it; it is reallocated when the
    // active device changes so it always lives on the same device as the stream.
    // `dBins` holds 2*batchCount per-batch Frobenius sums and grows when batchCount
    // increases. Both are intentionally not freed -- reclaimed at thread/process exit.
    thread_local DevAccum* dAccum       = nullptr;
    thread_local double*   dBins        = nullptr;
    thread_local int       dAccumDevice = -1;
    thread_local int       dBinsCap     = 0;
    int                    device       = -1;
    if(!gpu_ref_hip_check(hipGetDevice(&device), "get device"))
        return result;
    if(dAccum == nullptr || device != dAccumDevice)
    {
        if(dAccum)
            hipFree(dAccum);
        if(dBins)
            hipFree(dBins);
        dAccum   = nullptr;
        dBins    = nullptr;
        dBinsCap = 0;
        if(!gpu_ref_hip_check(hipMalloc(&dAccum, sizeof(DevAccum)), "accumulator alloc"))
            return result;
        dAccumDevice = device;
    }
    if(dBins == nullptr || batchCount > dBinsCap)
    {
        if(dBins)
            hipFree(dBins);
        dBins = nullptr;
        if(!gpu_ref_hip_check(hipMalloc(&dBins, sizeof(double) * 2 * size_t(batchCount)),
                              "per-batch bins alloc"))
            return result;
        dBinsCap = batchCount;
    }
    if(!gpu_ref_hip_check(hipMemsetAsync(dAccum, 0, sizeof(DevAccum), stream), "accumulator zero"))
        return result;
    if(!gpu_ref_hip_check(
           hipMemsetAsync(dBins, 0, sizeof(double) * 2 * size_t(batchCount), stream), "bins zero"))
        return result;

    // Bound the grid: gridDim.x covers a single batch's M x N (one atomic per block,
    // each block grid-strides over the rest); gridDim.y indexes batches, one block
    // row per batch (mirrors the reference kernel's grid.z == batchCount).
    const size_t mn           = size_t(M) * size_t(N);
    const size_t blocksPerBat = (mn + GPU_REF_BLOCK - 1) / GPU_REF_BLOCK;
    dim3         grid;
    grid.x = uint32_t(std::min<size_t>(blocksPerBat, 8192));
    grid.y = uint32_t(batchCount);
    grid.z = 1;

    if(tD == HIP_R_32F)
        compare_kernel<float><<<grid, GPU_REF_BLOCK, 0, stream>>>(static_cast<const float*>(dGpu),
                                                                  static_cast<const float*>(dRef),
                                                                  M,
                                                                  N,
                                                                  ldd,
                                                                  strideD,
                                                                  batchCount,
                                                                  ulpMantBits,
                                                                  dAccum,
                                                                  dBins);
    else
        compare_kernel<hipblasLtHalf><<<grid, GPU_REF_BLOCK, 0, stream>>>(
            static_cast<const hipblasLtHalf*>(dGpu),
            static_cast<const hipblasLtHalf*>(dRef),
            M,
            N,
            ldd,
            strideD,
            batchCount,
            ulpMantBits,
            dAccum,
            dBins);

    gpu_ref_hip_check(hipGetLastError(), "compare launch");

    DevAccum            hAccum{};
    std::vector<double> hBins(2 * size_t(batchCount), 0.0);
    if(gpu_ref_hip_check(
           hipMemcpyAsync(&hAccum, dAccum, sizeof(DevAccum), hipMemcpyDeviceToHost, stream),
           "accumulator copy-back")
       && gpu_ref_hip_check(hipMemcpyAsync(hBins.data(),
                                           dBins,
                                           sizeof(double) * 2 * size_t(batchCount),
                                           hipMemcpyDeviceToHost,
                                           stream),
                            "bins copy-back"))
    {
        gpu_ref_hip_check(hipStreamSynchronize(stream), "compare sync");
        result.max_abs_error    = hAccum.max_abs_error;
        result.max_ulp          = hAccum.max_ulp;
        result.sum_ulp          = hAccum.sum_ulp;
        result.num_unit_fail    = hAccum.num_unit_fail;
        result.num_nan_mismatch = hAccum.num_nan_mismatch;
        result.ulp_count        = hAccum.ulp_count;
        for(int k = 0; k < GPU_REF_TOL_GRID_N; ++k)
            result.allclose_g[k] = hAccum.allclose_g[k];

        // Batched Frobenius: per-batch ratio with the same "both norms ~0 -> 0"
        // guard the CPU applies per batch (norm_check_general strided branch).
        const double tol       = std::numeric_limits<double>::epsilon();
        double       norm_sum  = 0.0;
        for(int b = 0; b < batchCount; ++b)
        {
            const double ref_norm  = std::sqrt(hBins[b]);
            const double diff_norm = std::sqrt(hBins[size_t(batchCount) + b]);
            if(std::abs(ref_norm) <= tol && std::abs(diff_norm) <= tol)
                continue;
            norm_sum += diff_norm / ref_norm;
        }
        result.norm_error_sum = norm_sum;
    }

    return result;
}
