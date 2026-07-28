// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gpu_compare.hpp"
#include "hipblaslt_ostream.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
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

    // googletest FloatingPoint<float>::AlmostEquals (<= 4 ULP), which ASSERT_FLOAT_EQ
    // (and thus unit_check) applies per element; f16/bf16 outputs are promoted to
    // float first. Drives the exact (tol==0) unit_check.
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
        const unsigned int biased_a = to_biased(a);
        const unsigned int biased_b = to_biased(b);
        const unsigned int dist = (biased_a >= biased_b) ? (biased_a - biased_b) : (biased_b - biased_a);
        return dist <= 4u;
    }

    // Per-element error in ULP of the output type, matching ulp_distance() in ulp.hpp.
    __device__ inline double ulp_distance(double exact, double approx, int mant_bits)
    {
        if(exact == approx)
            return 0.0;
        int          exponent = 0;
        const double mantissa = frexp(exact, &exponent);
        if(fabs(mantissa) == 0.5) // power-of-2 boundary
            --exponent;
        const double ulp_size = ldexp(1.0, exponent - mant_bits);
        return fabs(exact - approx) / ulp_size;
    }

    // Block reductions over GPU_REF_BLOCK threads. One shared buffer per variant,
    // reused across sequential calls; each call brackets its use with syncs.
    __device__ inline double block_reduce_max(double value)
    {
        __shared__ double scratch[GPU_REF_BLOCK];
        const unsigned    tid = threadIdx.x;
        __syncthreads();
        scratch[tid] = value;
        __syncthreads();
        for(unsigned stride = blockDim.x >> 1; stride > 0; stride >>= 1)
        {
            if(tid < stride)
                scratch[tid] = fmax(scratch[tid], scratch[tid + stride]);
            __syncthreads();
        }
        return scratch[0];
    }

    __device__ inline double block_reduce_sum(double value)
    {
        __shared__ double scratch[GPU_REF_BLOCK];
        const unsigned    tid = threadIdx.x;
        __syncthreads();
        scratch[tid] = value;
        __syncthreads();
        for(unsigned stride = blockDim.x >> 1; stride > 0; stride >>= 1)
        {
            if(tid < stride)
                scratch[tid] += scratch[tid + stride];
            __syncthreads();
        }
        return scratch[0];
    }

    __device__ inline unsigned long long block_reduce_sum_ull(unsigned long long value)
    {
        __shared__ unsigned long long scratch[GPU_REF_BLOCK];
        const unsigned                tid = threadIdx.x;
        __syncthreads();
        scratch[tid] = value;
        __syncthreads();
        for(unsigned stride = blockDim.x >> 1; stride > 0; stride >>= 1)
        {
            if(tid < stride)
                scratch[tid] += scratch[tid + stride];
            __syncthreads();
        }
        return scratch[0];
    }

    __device__ inline void atomicMaxDouble(double* addr, double val)
    {
        auto*              addr_bits = reinterpret_cast<unsigned long long*>(addr);
        unsigned long long old       = atomicAdd(addr_bits, 0ull); // atomic initial load
        unsigned long long assumed   = old;
        do
        {
            assumed            = old;
            double current_max = __longlong_as_double(assumed);
            if(current_max >= val)
                break;
            old = atomicCAS(addr_bits, assumed, __double_as_longlong(val));
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
    // atomic per batch into `bins` (bins[batch] += sum_ref_sq, bins[batchCount+batch]
    // += sum_diff_sq), letting the host form the CPU-matching per-batch ratio. The
    // global metrics (max/ulp/allclose/counts) reduce within the block and combine
    // into `accum` with one atomic per block.
    template <typename To, typename Tcmp>
    __global__ void compare_kernel(const To* gpu,
                                   const To* ref,
                                   int64_t   M,
                                   int64_t   N,
                                   int64_t   ldd,
                                   int64_t   strideD,
                                   int32_t   batchCount,
                                   int       mant_bits,
                                   DevAccum* accum,
                                   double*   bins)
    {
        const double rtol[GPU_REF_TOL_GRID_N] = {GPU_REF_TOL_GRID[0],
                                                 GPU_REF_TOL_GRID[1],
                                                 GPU_REF_TOL_GRID[2],
                                                 GPU_REF_TOL_GRID[3],
                                                 GPU_REF_TOL_GRID[4],
                                                 GPU_REF_TOL_GRID[5]};
        static_assert(GPU_REF_TOL_GRID_N == 6, "update the rtol initializer above");

        // Per-thread metric accumulators, block-reduced into `accum` after the loop.
        double max_abs_err = 0.0, max_ulp = 0.0, sum_ulp = 0.0;
        double required_atol[GPU_REF_TOL_GRID_N]; // smallest atol admissible per rtol[k]
        for(int k = 0; k < GPU_REF_TOL_GRID_N; ++k)
            required_atol[k] = 0.0;
        unsigned long long unit_fail = 0, nan_mismatch = 0, ulp_count = 0;

        const size_t elems_per_batch = size_t(M) * size_t(N);
        const size_t grid_stride     = size_t(gridDim.x) * blockDim.x;

        const int64_t batch      = blockIdx.y; // one block row maps to exactly one batch
        double        sum_ref_sq = 0.0, sum_diff_sq = 0.0; // this batch only
        const size_t  batch_base = size_t(batch) * size_t(strideD);

        for(size_t pos = size_t(blockIdx.x) * blockDim.x + threadIdx.x; pos < elems_per_batch;
            pos += grid_stride)
        {
            const size_t j   = pos / size_t(M);
            const size_t i   = pos % size_t(M);
            const size_t idx = batch_base + i + j * size_t(ldd);

            // Narrow To (f32/f16/bf16) promotes to Tcmp (float) for the compare.
            const Tcmp   gpu_cmp = static_cast<Tcmp>(gpu[idx]);
            const Tcmp   ref_cmp = static_cast<Tcmp>(ref[idx]);
            const double gpu_val = double(gpu_cmp);
            const double ref_val = double(ref_cmp);

            if(isnan(gpu_val) || isinf(gpu_val) || isnan(ref_val) || isinf(ref_val))
            {
                // Matching same-signed infinities and both-nan pairs agree (as the
                // CPU unit/near checks do); any other non-finite disagreement is a
                // failure that also poisons the allclose grid. Non-finite values stay
                // out of the norm/ulp sums so they do not become nan.
                const bool matching_inf = isinf(gpu_val) && isinf(ref_val) && gpu_val == ref_val;
                const bool both_nan     = isnan(gpu_val) && isnan(ref_val);
                if(!(matching_inf || both_nan))
                {
                    ++nan_mismatch;
                    for(int k = 0; k < GPU_REF_TOL_GRID_N; ++k)
                        required_atol[k] = INFINITY;
                }
                continue;
            }

            const double abs_diff = fabs(gpu_val - ref_val);
            if(!float_almost_equals(gpu_cmp, ref_cmp))
                ++unit_fail;
            max_abs_err = fmax(max_abs_err, abs_diff);
            sum_ref_sq += ref_val * ref_val;
            sum_diff_sq += abs_diff * abs_diff;
            // allclose tolerance is atol + rtol*|gpu| (allclose() scales by the actual operand).
            const double abs_gpu = fabs(gpu_val);
            for(int k = 0; k < GPU_REF_TOL_GRID_N; ++k)
                required_atol[k] = fmax(required_atol[k], abs_diff - rtol[k] * abs_gpu);

            const double ulp = ulp_distance(ref_val, gpu_val, mant_bits);
            max_ulp          = fmax(max_ulp, ulp);
            sum_ulp += ulp;
            ++ulp_count;
        }

        // Per-batch Frobenius partials: reduce within the block, one atomic per batch.
        double block_sum_ref_sq  = block_reduce_sum(sum_ref_sq);
        double block_sum_diff_sq = block_reduce_sum(sum_diff_sq);
        if(threadIdx.x == 0)
        {
            atomicAdd(&bins[batch], block_sum_ref_sq);
            atomicAdd(&bins[size_t(batchCount) + batch], block_sum_diff_sq);
        }

        const bool lead = threadIdx.x == 0;

        double block_max_abs = block_reduce_max(max_abs_err);
        if(lead)
            atomicMaxDouble(&accum->max_abs_error, block_max_abs);
        for(int k = 0; k < GPU_REF_TOL_GRID_N; ++k)
        {
            double block_atol = block_reduce_max(required_atol[k]);
            if(lead)
                atomicMaxDouble(&accum->allclose_g[k], block_atol);
        }
        double block_max_ulp = block_reduce_max(max_ulp);
        if(lead)
            atomicMaxDouble(&accum->max_ulp, block_max_ulp);
        double block_sum_ulp = block_reduce_sum(sum_ulp);
        if(lead)
            atomicAdd(&accum->sum_ulp, block_sum_ulp);
        unsigned long long block_unit_fail = block_reduce_sum_ull(unit_fail);
        if(lead)
            atomicAdd(&accum->num_unit_fail, block_unit_fail);
        unsigned long long block_nan_mismatch = block_reduce_sum_ull(nan_mismatch);
        if(lead)
            atomicAdd(&accum->num_nan_mismatch, block_nan_mismatch);
        unsigned long long block_ulp_count = block_reduce_sum_ull(ulp_count);
        if(lead)
            atomicAdd(&accum->ulp_count, block_ulp_count);
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
    {
        result.valid = true; // nothing to compare is a valid empty result
        return result;
    }

    // Per-call device scratch: the global accumulator and the 2*batchCount per-batch
    // Frobenius sums. The destructor frees both on every return path (including the
    // error exits below)
    struct DeviceScratch
    {
        DevAccum* accum = nullptr;
        double*   bins  = nullptr;
        ~DeviceScratch()
        {
            if(accum)
                hipFree(accum);
            if(bins)
                hipFree(bins);
        }
    } scratch;
    if(!gpu_ref_hip_check(hipMalloc(&scratch.accum, sizeof(DevAccum)), "accumulator alloc"))
        return result;
    if(!gpu_ref_hip_check(hipMalloc(&scratch.bins, sizeof(double) * 2 * size_t(batchCount)),
                          "per-batch bins alloc"))
        return result;
    DevAccum* dAccum = scratch.accum;
    double*   dBins  = scratch.bins;
    if(!gpu_ref_hip_check(hipMemsetAsync(dAccum, 0, sizeof(DevAccum), stream), "accumulator zero"))
        return result;
    if(!gpu_ref_hip_check(
           hipMemsetAsync(dBins, 0, sizeof(double) * 2 * size_t(batchCount), stream), "bins zero"))
        return result;

    // Bound the grid: gridDim.x covers a single batch's M x N (one atomic per block,
    // each block grid-strides over the rest); gridDim.y indexes batches, one block
    // row per batch (mirrors the reference kernel's grid.z == batchCount).
    const size_t elems_per_batch  = size_t(M) * size_t(N);
    const size_t blocks_per_batch = (elems_per_batch + GPU_REF_BLOCK - 1) / GPU_REF_BLOCK;
    dim3         grid;
    grid.x = uint32_t(std::min<size_t>(blocks_per_batch, 8192));
    grid.y = uint32_t(batchCount);
    grid.z = 1;

    // Output type dispatch; the compare type (Tcmp) is the type the kernel promotes
    // to before comparing -- float for the narrow outputs here. `launch` names the
    // kernel arguments once.
    auto launch = [&](auto to_tag, auto tcmp_tag) {
        using TO   = decltype(to_tag);
        using TCMP = decltype(tcmp_tag);
        compare_kernel<TO, TCMP><<<grid, GPU_REF_BLOCK, 0, stream>>>(
            static_cast<const TO*>(dGpu),
            static_cast<const TO*>(dRef),
            M,
            N,
            ldd,
            strideD,
            batchCount,
            ulpMantBits,
            dAccum,
            dBins);
    };
    if(tD == HIP_R_32F)
        launch(float{}, float{});
    else if(tD == HIP_R_16BF)
        launch(hip_bfloat16{}, float{});
    else // HIP_R_16F
        launch(hipblasLtHalf{}, float{});

    if(!gpu_ref_hip_check(hipGetLastError(), "compare launch"))
    {
        gpu_ref_hip_check(hipStreamSynchronize(stream), "compare drain"); // drain before returning
        return result; // invalid
    }

    DevAccum            hAccum{};
    std::vector<double> hBins(2 * size_t(batchCount), 0.0);
    const bool          copied
        = gpu_ref_hip_check(
              hipMemcpyAsync(&hAccum, dAccum, sizeof(DevAccum), hipMemcpyDeviceToHost, stream),
              "accumulator copy-back")
          && gpu_ref_hip_check(hipMemcpyAsync(hBins.data(),
                                              dBins,
                                              sizeof(double) * 2 * size_t(batchCount),
                                              hipMemcpyDeviceToHost,
                                              stream),
                               "bins copy-back");
    // Sync unconditionally so no in-flight copy into hAccum/hBins outlives this scope.
    const bool synced = gpu_ref_hip_check(hipStreamSynchronize(stream), "compare sync");
    if(copied && synced)
    {
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
        const double zero_norm_tol = std::numeric_limits<double>::epsilon();
        double       norm_sum      = 0.0;
        for(int batch = 0; batch < batchCount; ++batch)
        {
            const double ref_norm  = std::sqrt(hBins[batch]);
            const double diff_norm = std::sqrt(hBins[size_t(batchCount) + batch]);
            if(std::abs(ref_norm) <= zero_norm_tol && std::abs(diff_norm) <= zero_norm_tol)
                continue;
            norm_sum += diff_norm / ref_norm;
        }
        result.norm_error_sum = norm_sum;
        result.valid          = true;
    }

    return result;
}
