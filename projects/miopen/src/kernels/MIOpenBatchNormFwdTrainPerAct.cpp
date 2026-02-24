/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#ifndef MIOPEN_HIP_RUNTIME_COMPILE
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

// Disable specific warnings
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wconditional-uninitialized"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsometimes-uninitialized"
#endif

#include "batchnorm_functions.hpp"

// Load the configs to this file
namespace /*anonymous*/ {
using mio_config    = miopen::config;
using mio_bn_config = miopen::batchnorm::config;
} // namespace

//==================== PER ACTIVATION =======================

#define BLOCK_SIZE (MIO_BN_GRP0 * MIO_BN_GRP1 * MIO_BN_GRP2)

extern "C" __global__ __launch_bounds__(BLOCK_SIZE) void MIOpenBatchNormFwdTrainPerActivation(
    const typename mio_bn_config::fp_type* __restrict__ in,         /* x input */
    unsigned int in_nstride,                                        /* C*H*W */
    unsigned int in_cstride,                                        /* H*W */
    typename mio_bn_config::fp_type* __restrict__ out,              /* y output */
    const typename mio_bn_config::fp_prec_type* __restrict__ scale, /* gamma 1xCxHxW */
    const typename mio_bn_config::fp_prec_type* __restrict__ bias,  /* beta  1xCxHxW */
#if(MIO_RUNNING_RESULT == 1)
    double expAvgFactor, /* input momentum */
    const typename mio_bn_config::fp_prec_type* __restrict__ prevResultRunningMean,     /* in */
    const typename mio_bn_config::fp_prec_type* __restrict__ prevResultRunningVariance, /* in */
    typename mio_bn_config::fp_prec_type* __restrict__ nextResultRunningMean,           /* out */
    typename mio_bn_config::fp_prec_type* __restrict__ nextResultRunningVariance,       /* out */
#endif
    double epsilon /* input fuzz param > 0 */
#if(MIO_SAVE_MEAN_VARIANCE == 1)
    ,
    typename mio_bn_config::fp_prec_type* __restrict__ resultSaveMean,       /* out only */
    typename mio_bn_config::fp_prec_type* __restrict__ resultSaveInvVariance /* out only */
#endif
)
{
    using fp_type         = typename mio_bn_config::fp_type;
    using fp_prec_type    = typename mio_bn_config::fp_prec_type;
    using fp_accum_c_type = typename mio_bn_config::fp_accum_c_type;
    using fp_prec_c_type  = typename mio_bn_config::fp_prec_c_type;

    unsigned int xgid = blockIdx.x;
    unsigned int ygid = blockIdx.y;
    int cidx          = MIO_BN_HW * static_cast<int>(xgid);

    const fp_prec_type invN = fp_prec_type(1) / fp_prec_type(MIO_BN_N);

    const auto* in_base = in + cidx;
    auto* out_base      = out + cidx;
    const auto* sc_base = scale + cidx;
    const auto* bs_base = bias + cidx;

    // move across the sections of the image mini_batch stack
    for(unsigned bid = ygid; bid * MIO_BN_GRP1 < in_cstride; bid += gridDim.y)
    {
        const auto blockOffset = bid * MIO_BN_GRP1;
        const auto idx         = blockOffset + threadIdx.y;

        if(idx >= in_cstride)
        {
            return;
        }

        fp_prec_type mean     = fp_prec_type(0);
        fp_prec_type variance = fp_prec_type(0);

        unsigned adjIndex = cidx + idx; // gamma and beta tensor index

        const auto* in_ptr = in_base + blockOffset;
        auto* out_ptr      = out_base + blockOffset;

        const auto getIndex = [&](unsigned int i) { return in_nstride * i + threadIdx.y; };
        const auto getInput = [&](unsigned int i) {
            return miopen::cast<fp_prec_type>(in_ptr[getIndex(i)]);
        };

        for(unsigned int n = 0; n < MIO_BN_N; n++)
        {
            mean += getInput(n);
        }
        mean *= invN;

        for(unsigned int n = 0; n < MIO_BN_N; n++)
        {
            const fp_prec_type x = getInput(n);
            const fp_prec_type d = x - mean;
            variance += d * d;
        }
        variance *= invN;

        // epsilon is double in API; cast to precision type for math
        fp_prec_type invVariance =
            static_cast<fp_prec_type>(rsqrt(variance + static_cast<fp_prec_type>(epsilon)));

        fp_prec_type pvt_scale = sc_base[idx];
        fp_prec_type pvt_bias  = bs_base[idx];

#if(MIO_RUNNING_RESULT == 1)
        // Match the newer HIP templated/namespaced pattern:
        using StashUpdater = miopen::batchnorm::StashUpdaterPA<fp_accum_c_type>;
        StashUpdater updater(static_cast<fp_accum_c_type>(mean),
                             static_cast<fp_accum_c_type>(variance),
                             static_cast<fp_accum_c_type>(expAvgFactor));

        miopen::batchnorm::running_stash<fp_accum_c_type, fp_prec_c_type, StashUpdater>(
            prevResultRunningMean,
            prevResultRunningVariance,
            nextResultRunningMean,
            nextResultRunningVariance,
            updater,
            adjIndex);
#endif

#if(MIO_SAVE_MEAN_VARIANCE == 1)
        // Save per-activation mean and inv-variance in the same (templated) style:
        miopen::batchnorm::saved_stash<fp_accum_c_type, fp_prec_c_type>(
            resultSaveMean, resultSaveInvVariance, mean, invVariance, adjIndex);
#endif

        for(unsigned int n = 0; n < MIO_BN_N; n++)
        {
            const fp_prec_type x      = miopen::cast<fp_prec_type>(in_ptr[getIndex(n)]);
            fp_prec_type inhat        = (x - mean) * invVariance;
            const fp_prec_type y_prec = fma(pvt_scale, inhat, pvt_bias);
            out_ptr[getIndex(n)]      = miopen::cast<fp_type>(y_prec);
        }
    }
}

// Restore warnings
#ifdef __clang__
#pragma clang diagnostic pop
#pragma clang diagnostic pop
#endif
