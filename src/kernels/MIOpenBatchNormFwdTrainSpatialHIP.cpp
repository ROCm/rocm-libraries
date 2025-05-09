/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "batchnorm_functions.hpp"
#include "activation_functions.hpp"


#ifndef MIO_LAYOUT_NHWC
#define MIO_LAYOUT_NHWC 0
#endif

#if(MIO_LAYOUT_NHWC != 0) && (MIO_LAYOUT_NHWC != 1)
#error "MIO_LAYOUT_NHWC must be 0 or 1"
#endif

template<int MIoBnVariant>
struct MIOpenBatchNormFwdTrainSpatialParams{};

template<>
struct MIOpenBatchNormFwdTrainSpatialParams<1>{
        // static constexpr int MIO_MAX_READ = static(MIO_LAYOUT_NHWC)? 
};

#if (MIO_BN_VARIANT == 0)
extern "C" __global__ void __launch_bounds__(MIO_BN_GRP0* MIO_BN_GRP1* MIO_BN_GRP2)
    MIOpenBatchNormFwdTrainSpatialHIP(
        [[maybe_unused]] const FP_TYPE* __restrict in,
        [[maybe_unused]] FP_TYPE* __restrict out,
        [[maybe_unused]] const FP_TYPE_PREC* __restrict scale,
        [[maybe_unused]] const FP_TYPE_PREC* __restrict bias,
        [[maybe_unused]] FP_TYPE_PREC INHW,
#if(MIO_RUNNING_RESULT == 1)
        [[maybe_unused]] double expAvgFactor,
        [[maybe_unused]] FP_TYPE_PREC* __restrict resultRunningMean,
        [[maybe_unused]] FP_TYPE_PREC* __restrict resultRunningVariance,
#endif
        double epsilon
#if(MIO_SAVE_MEAN_VARIANCE == 1)
        ,
        [[maybe_unused]] FP_TYPE_PREC* __restrict resultSaveMean,
        [[maybe_unused]] FP_TYPE_PREC* __restrict resultSaveInvVariance
#endif
)
{
    return;
}
#elif(MIO_BN_VARIANT == 1)
extern "C" __global__ void __launch_bounds__(MIO_BN_GRP0* MIO_BN_GRP1* MIO_BN_GRP2)
MIOpenBatchNormFwdTrainSpatialHIP(
                            [[maybe_unused]] const FP_TYPE* __restrict in,
                            [[maybe_unused]] FP_TYPE* __restrict out,
                            [[maybe_unused]] const FP_TYPE_PREC* __restrict scale,
                            [[maybe_unused]] const FP_TYPE_PREC* __restrict bias,
                            [[maybe_unused]] FP_TYPE_PREC INHW,
#if(MIO_RUNNING_RESULT == 1)
                            [[maybe_unused]] double expAvgFactor,
                            [[maybe_unused]] FP_TYPE_PREC* __restrict resultRunningMean,
                            [[maybe_unused]] FP_TYPE_PREC* __restrict resultRunningVariance,
#endif
                            [[maybe_unused]] double epsilon
#if(MIO_SAVE_MEAN_VARIANCE == 1)
                            ,
                            [[maybe_unused]] FP_TYPE_PREC* __restrict resultSaveMean,
                            [[maybe_unused]] FP_TYPE_PREC* __restrict resultSaveInvVariance
#endif
){
    return;
}

#endif

