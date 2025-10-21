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

#include "batchnorm_functions.hpp"
#include "activation_functions.hpp"
#include "static_unroll.hpp"

// Load the configs to this file
namespace /*anonymous*/ {
using mio_config    = miopen::config;
using mio_bn_config = miopen::batchnorm::config;
} // namespace

//==================== PER ACTIVATION =======================
extern "C" __global__ void MIOpenBatchNormActivFwdTrainPerActivation(
    const mio_bn_config::fp_type alpha,
    const mio_bn_config::fp_type beta,
    const mio_bn_config::fp_type gamma,
    double epsilon, /* input fuzz param > 0 */
#if (MIO_RUNNING_RESULT == 1)
    double expAvgFactor,
#endif
    const typename mio_bn_config::fp_type* __restrict in,        /* x input */
    typename mio_bn_config::fp_type* __restrict out,             /* y output */
    const typename mio_bn_config::fp_prec_type* __restrict bias, /* beta 1xCxHxW */
    const typename mio_bn_config::fp_prec_type* __restrict scale /* gamma 1xCxHxW */
#if (MIO_RUNNING_RESULT == 1)
    ,
    typename mio_bn_config::fp_prec_type* __restrict runningMean,    /*input and output, same
                                                                        descriptor as bias*/
    typename mio_bn_config::fp_prec_type* __restrict runningVariance /*input and output*/
#endif
#if (MIO_SAVE_MEAN_VARIANCE == 1)
    ,
    typename mio_bn_config::fp_prec_type* __restrict savedInvVariance, /*output only*/
    typename mio_bn_config::fp_prec_type* __restrict savedMean         /*output only*/

#endif
)
{
    using fp_prec_type    = typename mio_bn_config::fp_prec_type;
    using fp_accum_type   = typename mio_bn_config::fp_accum_type;
    using fp_accum_c_type = typename mio_bn_config::fp_accum_c_type;
    using fp_prec_c_type  = typename mio_bn_config::fp_prec_c_type;

    // PER ACTIVATION
    unsigned int xgid    = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ygid    = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int yglb_sz = blockDim.y * gridDim.y;
    unsigned int Cidx    = mio_bn_config::hw * xgid;

    // move across the sections of the image mini_batch stack
    for(unsigned int img_offset = 0; img_offset < mio_bn_config::hw; img_offset += yglb_sz)
    {
        unsigned int inImgIndex = img_offset + ygid;
        if(inImgIndex >= mio_bn_config::hw)
        {
            continue;
        }

        fp_prec_type mean     = 0;
        fp_prec_type variance = 0;
        unsigned int adjIndex = Cidx + inImgIndex; // gamma and beta tensor index
        fp_prec_type pvt_scale = scale[adjIndex];
        fp_prec_type pvt_bias  = bias[adjIndex];

        for (unsigned int n = 0; n < mio_bn_config::n; ++n)
        {
            unsigned int index = mio_bn_config::chw * n + adjIndex;
            fp_prec_type xin   = miopen::batchnorm::cast<fp_prec_type>(in[index]);
            mean += xin;
            variance = fma(xin, xin, variance);
        }
        mean /= mio_bn_config::fp_prec_type(mio_bn_config::n);
        variance /= mio_bn_config::fp_prec_type(mio_bn_config::n);
        variance                 = fma(-mean, mean, variance);
        fp_prec_type invVariance = rsqrt(variance + epsilon);

        fp_prec_type bn_out, act_out;
        for (unsigned int n = 0; n < mio_bn_config::n; ++n)
        {
            // per (x-dims) channel load a block of data unsigned into LDS
            unsigned int index = mio_bn_config::chw * n + adjIndex;
            fp_prec_type inhat =
                (miopen::batchnorm::cast<fp_prec_type>(in[index]) - mean) * invVariance;
            bn_out = fma(pvt_scale, inhat, pvt_bias);
            ActivationFunction<fp_prec_type, 1>(*reinterpret_cast<fp_prec_type(*)[1]>(&act_out),
                                                *reinterpret_cast<fp_prec_type(*)[1]>(&bn_out),
                                                miopen::batchnorm::cast<fp_prec_type>(gamma),
                                                miopen::batchnorm::cast<fp_prec_type>(beta),
                                                miopen::batchnorm::cast<fp_prec_type>(alpha));
            out[index] = miopen::batchnorm::cast<fp_prec_type>(act_out);
        }
    
#if (MIO_RUNNING_RESULT == 1)
        using StashUpdater = miopen::batchnorm::StashUpdaterPA<fp_accum_c_type>;
        StashUpdater updater(static_cast<fp_accum_c_type>(mean),
                             static_cast<fp_accum_c_type>(variance),
                             static_cast<fp_accum_c_type>(expAvgFactor));

        miopen::batchnorm::running_stash<fp_accum_c_type, fp_prec_c_type, StashUpdater>(
            runningMean, runningVariance, updater, adjIndex);
#endif

#if (MIO_SAVE_MEAN_VARIANCE == 1)
        miopen::batchnorm::saved_stash<fp_accum_c_type, fp_prec_c_type>(
            savedMean, savedInvVariance, mean, invVariance, adjIndex);
#endif
    } // image mini_batch is processed
}
