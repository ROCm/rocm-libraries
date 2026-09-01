// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_REDUCTION_FUNCTIONS_HPP
#define GUARD_REDUCTION_FUNCTIONS_HPP

#include "vector_types.hpp"
#ifndef MIOPEN_HIP_RUNTIME_COMPILE
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "configuration.hpp"
#include "static_unroll.hpp"

// NOTE: This header should be independent from batchnorm_functions.hpp
// These templates will not be compiled before calling them.
namespace miopen {
namespace reduction {

namespace detail {

const unsigned long long FULL_MASK = 0xFFFFFFFFFFFFFFFFull;

__forceinline__ __device__ unsigned int next_power_of_2(unsigned int n)
{
    unsigned int p = 1;
    while(p < n)
        p <<= 1;
    return p;
}

} // namespace detail

// This function is currently not in use, as more performant variants exist in the form of
// reduce2_welford, which selects either doing shuffles or doing DPP assembly for GFX9.
// Theoretically, this function can be used on cards before GCN3 that do not support
// cross-lane operations. But the main reason this function is left in the code is
// to further performance comparisons when applying it in different kernels.
template <typename FloatAccum, unsigned int SizeLclData>
__forceinline__ __device__ void lds_reduce2_welford(FloatAccum& mean,
                                                    FloatAccum& variance,
                                                    FloatAccum& count,
                                                    FloatAccum scale,
                                                    FloatAccum (&lcl_data_mean)[SizeLclData],
                                                    FloatAccum (&lcl_data_variance)[SizeLclData],
                                                    FloatAccum (&lcl_data_count)[SizeLclData],
                                                    unsigned int lid)
{
    lcl_data_mean[lid]     = mean;
    lcl_data_variance[lid] = variance;
    lcl_data_count[lid]    = count;
    __syncthreads();

    for(unsigned int red = detail::next_power_of_2(SizeLclData) >> 1; red > 0; red >>= 1)
    {
        if(lid < red && lid + red < SizeLclData)
        {
            FloatAccum delta = lcl_data_mean[lid + red] - lcl_data_mean[lid];
            FloatAccum n_a   = lcl_data_count[lid];
            FloatAccum n_b   = lcl_data_count[lid + red];
            FloatAccum n_new = n_a + n_b;
            FloatAccum n_new_rcp =
                n_new != 0.0f ? __builtin_amdgcn_rcpf(n_new)
                              : 0.f; // Calculates 1 / n_new; Done to handle the case where some of
                                     // the partitions of mean/variance calculation are zero
            lcl_data_mean[lid] =
                (lcl_data_mean[lid] * n_a + lcl_data_mean[lid + red] * n_b) * n_new_rcp;
            lcl_data_variance[lid] = lcl_data_variance[lid] + lcl_data_variance[lid + red] +
                                     delta * delta * (n_a * n_b * n_new_rcp);
            lcl_data_count[lid] = n_new;
        }
        __syncthreads();
    }

    mean     = lcl_data_mean[0];
    variance = lcl_data_variance[0] * scale;
}

template <typename FloatAccum, unsigned int BlockSize>
__forceinline__ __device__ void
reduce2(FloatAccum& x, FloatAccum& y, FloatAccum scale, unsigned int lid)
{
    static_assert(BlockSize > 0, "BlockSize must be positive");

    if constexpr(BlockSize == 1)
    {
        x *= scale;
        y *= scale;
        return;
    }

    if constexpr(BlockSize % 64 == 0)
    {
        for(unsigned int d = warpSize / 2; d >= 1; d >>= 1)
        {
            x += __shfl_down_sync(detail::FULL_MASK, x, d);
            y += __shfl_down_sync(detail::FULL_MASK, y, d);
        }

        if(BlockSize <= static_cast<unsigned int>(warpSize))
        {
            x = __shfl_sync(detail::FULL_MASK, x, 0) * scale;
            y = __shfl_sync(detail::FULL_MASK, y, 0) * scale;
            return;
        }

        constexpr unsigned int max_warps = BlockSize / 32;
        __shared__ FloatAccum s_x[max_warps];
        __shared__ FloatAccum s_y[max_warps];

        const unsigned int lane      = lid % static_cast<unsigned int>(warpSize);
        const unsigned int wid       = lid / static_cast<unsigned int>(warpSize);
        const unsigned int num_warps = BlockSize / static_cast<unsigned int>(warpSize);

        if(lane == 0)
        {
            s_x[wid] = x;
            s_y[wid] = y;
        }
        __syncthreads();

        if(wid == 0)
        {
            x = FloatAccum{0};
            y = FloatAccum{0};
            for(unsigned int i = lane; i < num_warps; i += static_cast<unsigned int>(warpSize))
            {
                x += s_x[i];
                y += s_y[i];
            }
            for(unsigned int d = warpSize / 2; d >= 1; d >>= 1)
            {
                x += __shfl_down_sync(detail::FULL_MASK, x, d);
                y += __shfl_down_sync(detail::FULL_MASK, y, d);
            }
        }

        if(lid == 0)
        {
            s_x[0] = x * scale;
            s_y[0] = y * scale;
        }
        __syncthreads();
        x = s_x[0];
        y = s_y[0];
    }
    else
    {
        // Slow path, mainly for the unlikely case of a 32 thread block
        __shared__ FloatAccum s_x[BlockSize];
        __shared__ FloatAccum s_y[BlockSize];

        s_x[lid] = x;
        s_y[lid] = y;
        __syncthreads();

        if(lid < static_cast<unsigned int>(warpSize))
        {
            x = FloatAccum{0};
            y = FloatAccum{0};
            for(unsigned int i = lid; i < BlockSize; i += static_cast<unsigned int>(warpSize))
            {
                x += s_x[i];
                y += s_y[i];
            }
            for(unsigned int d = warpSize / 2; d >= 1; d >>= 1)
            {
                x += __shfl_down_sync(detail::FULL_MASK, x, d);
                y += __shfl_down_sync(detail::FULL_MASK, y, d);
            }
        }

        if(lid == 0)
        {
            s_x[0] = x * scale;
            s_y[0] = y * scale;
        }
        __syncthreads();
        x = s_x[0];
        y = s_y[0];
    }
}

template <typename FloatAccumC, typename FloatAccum, unsigned int SizeLclData>
__forceinline__ __device__ void lds_reduce2_2d(FloatAccumC& x,
                                               FloatAccumC& y,
                                               FloatAccum scale,
                                               FloatAccumC (&lcl_data)[SizeLclData],
                                               unsigned int xstride,
                                               unsigned int xlid,
                                               unsigned int ylid,
                                               unsigned int size)
{
    unsigned int offset1 = 2 * (xlid + ylid * xstride);
    // store the values by pairs (so the compiler will generate
    // one instruction to read/write them)
    lcl_data[offset1 + 0] = static_cast<FloatAccumC>(x);
    lcl_data[offset1 + 1] = static_cast<FloatAccumC>(y);

    __syncthreads();

    const unsigned int red_start = detail::next_power_of_2(size) >> 1;

    for(unsigned int red = red_start; red > 0; red >>= 1)
    {
        unsigned int offset2 = offset1 + red * xstride * 2;
        if(ylid < red && ylid + red < size && offset2 + 1 < SizeLclData)
        {
            // make sure there is one read and one write
            x += lcl_data[offset2 + 0];
            y += lcl_data[offset2 + 1];
            lcl_data[offset1 + 0] = x;
            lcl_data[offset1 + 1] = y;
        }
        __syncthreads();
    }
    x = static_cast<FloatAccumC>(lcl_data[xlid * 2 + 0] * scale);
    y = static_cast<FloatAccumC>(lcl_data[xlid * 2 + 1] * scale);
}

// Caller must ensure: SizeLclData >= (blockDim.x * blockDim.y * blockDim.z + warpSize - 1) /
// warpSize
// @warning Undefined behavior if SizeLclData is too small
// Caller must ensure: All lanes must be active
// @warning Undefined behavior if lanes are masked
template <typename FloatAccum, unsigned int SizeLclData>
__forceinline__ __device__ void gcn_reduce2(FloatAccum& x,
                                            FloatAccum& y,
                                            FloatAccum scale,
                                            FloatAccum (&lcl_data_x)[SizeLclData],
                                            FloatAccum (&lcl_data_y)[SizeLclData],
                                            unsigned int lid)
{
    const unsigned int ldsidx         = lid / warpSize;
    constexpr unsigned long long mask = 0xFFFFFFFFFFFFFFFFull;
    x                                 = __reduce_add_sync(mask, x);
    y                                 = __reduce_add_sync(mask, y);
    // Last thread
    if((lid % warpSize) == warpSize - 1)
    {
        lcl_data_x[ldsidx] = x;
        lcl_data_y[ldsidx] = y;
    }

    __syncthreads();

    x = y = 0;

    static_unroll_count<unsigned int, 0, SizeLclData, 1, 2>{[&](unsigned int i) {
        x += lcl_data_x[i];
        y += lcl_data_y[i];
    }};

    x *= scale;
    y *= scale;
}

template <typename FloatAccum>
__forceinline__ __device__ void dpp_interleaved_reduction_welford(FloatAccum& temp_mean,
                                                                  FloatAccum& temp_var,
                                                                  FloatAccum& temp_count)
{
    FloatAccum delta;
    FloatAccum count_mult;
    FloatAccum n_rcp;

    // Disabling clang-format within this function, otherwise
    // the indentation gets very messy

    // clang-format off
    #define REDUCTION_STEP(dpp) \
            /* 1 */ "v_add_f32 %4 -%0 %0 " dpp "\n" /* store the deltas of the means, that's a scratch register we'll need for the math. Multiply src0 by -1 to get the proper sign of the delta. Note: the first delta will be incorrect! Will be non-zero! */ \
            /* 2 */ "v_mul_f32 %0 %0 %2\n" /* calculate mean times count */ \
            /* 3 */ "v_mul_f32 %3 %2 %2 " dpp "\n" /* get n_a * n_b */ \
            /* 4 */ "v_mul_f32 %4 %4 %4\n" /* square deltas, we need those for the variance calculation; this is put here in place of a NOP due to dependency between 2 and 5*/ \
            /* 5 */ "v_add_f32 %0 %0 %0 " dpp "\n" /* merge mean times count */ \
            /* 6 */ "v_add_f32 %2 %2 %2 " dpp "\n" /* sum up the new counts that correspond to the new mean times count values*/ \
            /* 7 */ "v_rcp_f32 %5 %2\n" /* prepare for division by n_a + n_b possibly also do v_div_scale_f32?*/ \
            /* 8 */ "v_cmp_eq_f32 vcc %2 0\n"/* Idea: move 0 if %2 is zero for specific lanes -- for the mean, the variance and the count */\
            /* 9 */ "v_cndmask_b32_e64 %5 %5 %2 vcc\n" \
            /*10 */ "v_mul_f32 %0 %0 %5\n" /* normalize mean; %5 is 1/(n_a + n_b), it's the updated counts */ \
            /*11 */ "v_add_f32 %1 %1 %1 " dpp "\n" /* part of the variance calculation -- sum up the two partitions, add the deltas in the next steps */ \
            /*12 */ "v_mul_f32 %5 %3 %5\n"\
            /*13 */ /* NOP is not necessary here, it's needed when the first instruction is non-DPP and the second one is, thus there should be no dependency between 11 and 14 or data corruption risk between 12 and 14 */ \
            /*14 */ "v_fma_f32 %1 %4 %5 %1\n" /* %4, %5 and %1 should already have been properly offset with the required number of lanes for the reduction */ \
            /*15 */ "v_nop\n"\
            /*16 */ "v_nop\n" /* NOPs necessary because the next instr needs %4 and has DPP, dependency between 14 and 1 */\

    __asm__ volatile(
        "s_nop 4\n" /* necessary because it's not guaranteed that the compiler puts a VALU instruction that writes EXEC */
        REDUCTION_STEP("row_shr:1 bound_ctrl:0")
        REDUCTION_STEP("row_shr:2 bound_ctrl:0")
        REDUCTION_STEP("row_shr:4 bank_mask:0xe")
        REDUCTION_STEP("row_shr:8 bank_mask:0xc")
        REDUCTION_STEP("row_bcast:15 row_mask:0xa")
        REDUCTION_STEP("row_bcast:31 row_mask:0xc")


        ""
            : "=v"(temp_mean), "=v"(temp_var), "=v" (temp_count), "=v" (count_mult), "=v" (delta), "=v" (n_rcp)
            : "0"(temp_mean), "1"(temp_var), "2" (temp_count), "3" (count_mult), "4" (delta), "5" (n_rcp)
            : "vcc");
    // The last list in the assembly block is the clobbers --
    // those are registers (or other special arguments) that get
    // modified in the assembly block that are not specified as either
    // inputs or outputs. In this case, the "vcc" register is the only
    // register that gets modified in this way. If "vcc" is not added,
    // this can lead to bugs in some test cases, specifically when the
    // vcc register is used to store some value needed after the execution
    // of this block. After adding this register to the clobbers, the three
    // input parameters to this function no longer need to be volatile.

    // The macro REDUCTION_STEP is necessary only for this function;
    // if left defined, it would be exposed when the header is included
    #undef REDUCTION_STEP

    // clang-format on
}

template <typename FloatAccum>
__forceinline__ __device__ void shfl_interleaved_reduction_welford(FloatAccum& temp_mean,
                                                                   FloatAccum& temp_var,
                                                                   FloatAccum& temp_count)
{
#pragma unroll
    for(unsigned int offset = warpSize >> 1; offset >= 1; offset >>= 1)
    {
        FloatAccum otherMean = __shfl_down_sync(detail::FULL_MASK, temp_mean, offset);
        FloatAccum delta     = otherMean - temp_mean;
        FloatAccum n_a       = temp_count;
        FloatAccum n_b       = __shfl_down_sync(detail::FULL_MASK, temp_count, offset);
        FloatAccum n_new     = n_a + n_b;
        FloatAccum n_rcp     = n_new != 0.0f ? (__builtin_amdgcn_rcpf(n_new)) : 0.0f;
        temp_mean            = (temp_mean * n_a + otherMean * n_b) * (n_rcp);
        temp_var             = temp_var + __shfl_down_sync(detail::FULL_MASK, temp_var, offset) +
                   (delta * delta * (n_a * n_b)) * n_rcp;
        temp_count = n_new;
    }
}

template <typename FloatAccum, unsigned int SizeLclData>
__forceinline__ __device__ void reduce2_welford(FloatAccum& mean,
                                                FloatAccum& variance,
                                                FloatAccum& count,
                                                FloatAccum scale,
                                                FloatAccum (&lcl_data_mean)[SizeLclData],
                                                FloatAccum (&lcl_data_variance)[SizeLclData],
                                                FloatAccum (&lcl_data_count)[SizeLclData],
                                                unsigned int lid)
{
    const unsigned int ldsidx =
        lid >> (warpSize == 32 ? 5 : 6); // warpSize is either 32 or 64 on AMD hardware.

    if constexpr(!miopen::batchnorm::config::use_amdgcn)
    {
        shfl_interleaved_reduction_welford<FloatAccum>(mean, variance, count);
        __builtin_amdgcn_sched_barrier(0);

        // Last thread
        if((lid % warpSize) == 0)
        {
            lcl_data_mean[ldsidx]     = mean;
            lcl_data_variance[ldsidx] = variance;
            lcl_data_count[ldsidx]    = count;
        }
    }
    else
    {
        dpp_interleaved_reduction_welford<FloatAccum>(mean, variance, count);
        __builtin_amdgcn_sched_barrier(0);

        // Last thread
        if((lid % 64) == 63)
        {
            lcl_data_mean[ldsidx]     = mean;
            lcl_data_variance[ldsidx] = variance;
            lcl_data_count[ldsidx]    = count;
        }
    }
    __syncthreads();

    // The reduction here merges partitions together in a tree-like fashion. Otherwise,
    // precision is lost in both the mean and variance
    for(unsigned int red = detail::next_power_of_2(SizeLclData) >> 1; red > 0; red >>= 1)
    {
        if(lid < red && lid + red < SizeLclData)
        {
            FloatAccum delta = lcl_data_mean[lid + red] - lcl_data_mean[lid];
            FloatAccum n_a   = lcl_data_count[lid];
            FloatAccum n_b   = lcl_data_count[lid + red];
            FloatAccum n_new = n_a + n_b;
            FloatAccum n_new_rcp =
                n_new != 0.0f ? __builtin_amdgcn_rcpf(n_new)
                              : 0.f; // Calculates 1 / n_new; Done to handle the case where some of
                                     // the partitions of mean/variance calculation are zero
            lcl_data_mean[lid] =
                (lcl_data_mean[lid] * n_a + lcl_data_mean[lid + red] * n_b) * n_new_rcp;
            lcl_data_variance[lid] = lcl_data_variance[lid] + lcl_data_variance[lid + red] +
                                     delta * delta * (n_a * n_b * n_new_rcp);
            lcl_data_count[lid] = n_new;
        }
        __syncthreads();
    }

    // No scaling of the mean, that is already kept to scale as a requirement of Welford's variance
    // calculation
    mean     = lcl_data_mean[0];
    variance = lcl_data_variance[0] * scale;
}

template <typename FpPrecType>
__forceinline__ __device__ void
welford_step_unroll4(typename mapped_vector_type<FpPrecType, 4>::type& read4Prec,
                     FpPrecType& mean,
                     FpPrecType& variance,
                     FpPrecType& curN)
{
    // Using Welford's algorithm for calculating variance
    // Manually unrolled calculation for the mean and variance of 4 elements, plus a
    // merger step

    FpPrecType mean4     = read4Prec.x;
    FpPrecType oldMean4  = mean4;
    FpPrecType variance4 = cast<FpPrecType>(0.);

    mean4 += read4Prec.y;
    mean4 *= 0.5f;
    variance4 = fma(read4Prec.y - mean4, read4Prec.y - oldMean4, variance4);

    oldMean4 = mean4;
    mean4    = mean4 * 2.0f + read4Prec.z;
    mean4 *= 0.333333333f;
    variance4 = fma(read4Prec.z - mean4, read4Prec.z - oldMean4, variance4);

    oldMean4 = mean4;
    mean4    = mean4 * 3.0f + read4Prec.w;
    mean4 *= 0.25f;
    variance4 = fma(read4Prec.w - mean4, read4Prec.w - oldMean4, variance4);

    // merge the local mean and variance with the currently computed ones

    FpPrecType delta = mean4 - mean;
    mean             = (mean4 * 4.0f + mean * curN) / (curN + 4.f);
    variance += variance4 + delta * delta * 4.f * curN / (curN + 4.f);
    curN += 4.f;
}

} // namespace reduction
} // namespace miopen

#endif
