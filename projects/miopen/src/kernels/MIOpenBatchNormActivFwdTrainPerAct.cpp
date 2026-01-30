// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "batchnorm_functions.hpp"
#include "activation_functions.hpp"
#include "static_unroll.hpp"

// Load the configs to this file
namespace /*anonymous*/ {
using mio_config    = miopen::config;
using mio_bn_config = miopen::batchnorm::config;
} // namespace

//==================== PER ACTIVATION =======================
extern "C" __global__ void __launch_bounds__(mio_bn_config::launch_dim.grp0)
    MIOpenBatchNormActivFwdTrainPerActivation(
        const mio_bn_config::fp_type alpha,
        const mio_bn_config::fp_type beta,
        const mio_bn_config::fp_type gamma,
        double epsilon, /* input fuzz param > 0 */
#if(MIO_RUNNING_RESULT == 1)
        double expAvgFactor,
#endif
        const typename mio_bn_config::fp_type* __restrict in,        /* x input */
        typename mio_bn_config::fp_type* __restrict out,             /* y output */
        const typename mio_bn_config::fp_prec_type* __restrict bias, /* beta 1xCxHxW */
        const typename mio_bn_config::fp_prec_type* __restrict scale /* gamma 1xCxHxW */
#if(MIO_RUNNING_RESULT == 1)
        ,
        typename mio_bn_config::fp_prec_type* __restrict runningMean,    /*input and output, same
                                                                            descriptor as bias*/
        typename mio_bn_config::fp_prec_type* __restrict runningVariance /*input and output*/
#endif
#if(MIO_SAVE_MEAN_VARIANCE == 1)
        ,
        typename mio_bn_config::fp_prec_type* __restrict savedInvVariance, /*output only*/
        typename mio_bn_config::fp_prec_type* __restrict savedMean         /*output only*/

#endif
    )
{
    using fp_type        = typename mio_bn_config::fp_type;
    using fp_prec_type   = typename mio_bn_config::fp_prec_type;
    using fp_prec_c_type = typename mio_bn_config::fp_prec_c_type;

    // PER ACTIVATION
    unsigned int xgid = blockIdx.x * blockDim.x + threadIdx.x;

    if(xgid >= mio_bn_config::chw)
    {
        return;
    }

    fp_prec_type mean      = 0;
    fp_prec_type variance  = 0;
    fp_prec_type pvt_scale = scale[xgid];
    fp_prec_type pvt_bias  = bias[xgid];

    for(unsigned int n = 0; n < mio_bn_config::n; ++n)
    {
        unsigned int index = mio_bn_config::chw * n + xgid;
        fp_prec_type xin   = miopen::cast<fp_prec_type>(in[index]);
        mean += xin;
        variance = fma(xin, xin, variance);
    }
    mean /= mio_bn_config::n;
    variance /= mio_bn_config::n;
    variance                 = fma(-mean, mean, variance);
    fp_prec_type invVariance = rsqrt(variance + epsilon);

#if(MIO_RUNNING_RESULT == 1)
    using StashUpdater = miopen::batchnorm::StashUpdaterPA<fp_prec_c_type>;
    StashUpdater updater(static_cast<fp_prec_c_type>(mean),
                         static_cast<fp_prec_c_type>(variance),
                         static_cast<fp_prec_c_type>(expAvgFactor));

    miopen::batchnorm::running_stash<fp_prec_c_type, fp_prec_c_type, StashUpdater>(
        runningMean, runningVariance, runningMean, runningVariance, updater, xgid);
#endif

#if(MIO_SAVE_MEAN_VARIANCE == 1)
    miopen::batchnorm::saved_stash<fp_prec_c_type, fp_prec_c_type>(
        savedMean, savedInvVariance, mean, invVariance, xgid);
#endif

    fp_prec_type bn_out, act_out;
    for(unsigned int n = 0; n < mio_bn_config::n; ++n)
    {
        // per (x-dims) channel load a block of data unsigned into LDS
        unsigned int index = mio_bn_config::chw * n + xgid;
        fp_prec_type inhat = (miopen::cast<fp_prec_type>(in[index]) - mean) * invVariance;
        bn_out             = fma(pvt_scale, inhat, pvt_bias);
        ActivationFunction<fp_prec_type, 1>(*reinterpret_cast<fp_prec_type(*)[1]>(&act_out),
                                            *reinterpret_cast<fp_prec_type(*)[1]>(&bn_out),
                                            miopen::cast<fp_prec_type>(gamma),
                                            miopen::cast<fp_prec_type>(beta),
                                            miopen::cast<fp_prec_type>(alpha));
        out[index] = miopen::cast<fp_type>(act_out);
    }
}
