/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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

#include <miopen/batch_norm.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/tensor.hpp>

#include <cassert>
#include <iostream>

#define MIOPEN_BN_SYNCH 0

namespace miopen {

void DeriveBNTensorDescriptor(TensorDescriptor& derivedBnDesc,
                              const TensorDescriptor& xDesc,
                              miopenBatchNormMode_t bn_mode)
{

    auto lengths = xDesc.GetLengths();
    std::vector<int> newlens(lengths.size());
    newlens[1] = lengths[1];
    if(bn_mode == miopenBNSpatial)
    {
        newlens[0] = newlens[2] = newlens[3] = 1;
        if(lengths.size() == 5)
            newlens[4] = 1;
    }
    else
    {
        newlens[0] = 1;
        newlens[2] = lengths[2];
        newlens[3] = lengths[3];

        if(lengths.size() == 5)
            newlens[4] = lengths[4];
    }
    derivedBnDesc = TensorDescriptor(/* xDesc.GetType() */ miopenFloat, newlens);
}

TensorDescriptor BuildReshaped4DTensorDescriptor(const miopen::TensorDescriptor& tDesc)
{
    std::vector<size_t> dims(tDesc.GetLengths());

    auto dataType   = tDesc.GetType();
    auto layout_opt = tDesc.GetLayoutEnum();
    if(!layout_opt)
    {
        MIOPEN_THROW(miopenStatusInternalError, "Unset layout");
    }

    auto layout = layout_opt.value();
    if(layout == miopenTensorNCDHW)
    {
        layout = miopenTensorNCHW;
    }
    else if(layout == miopenTensorNDHWC)
    {
        layout = miopenTensorNHWC;
    }
    else
    {
        std::cout << "Cannot handle layout : " << layout << "\n";
        exit(EXIT_FAILURE); // NOLINT (concurrency-mt-unsafe)
    }

    // Both NCDHW and NDHWC layout store the lens in NCHDW form
    // hence : NxCxDxHxW -> NxCx(D*H)xW
    dims[2] *= dims[3];
    dims[3] = dims[4];
    dims.pop_back();

    return {dataType, layout, dims};
}

void profileSequence(const Handle& handle, unsigned char select, float* ctime)
{

    float ktime = 0.;
    assert((select < 3) && "profileSequence case incorrect");
    switch(select)
    {

    case 0:
        if(handle.IsProfilingEnabled())
        {
            ktime  = handle.GetKernelTime();
            *ctime = ktime;

#if(MIO_BN_CPP_PROF == 1)
            printf("kernel0: %7.3f ms   total: %7.3f ms\n", ktime, *ctime);
#endif
        }
#if(MIOPEN_BN_SYNCH == 1)
        else
        {
            handle.Finish();
        }
#endif
        break;
    case 1:
        if(handle.IsProfilingEnabled())
        {
            ktime = handle.GetKernelTime();
            *ctime += ktime;

#if(MIO_BN_CPP_PROF == 1)
            printf("kernel1: %7.3f ms   total: %7.3f ms\n", ktime, *ctime);
#endif
        }
#if(MIOPEN_BN_SYNCH == 1)
        else
        {
            handle.Finish();
        }
#endif
        break;

    case 2:
        if(handle.IsProfilingEnabled())
        {
            ktime = handle.GetKernelTime();
            *ctime += ktime;
            handle.ResetKernelTime();
            handle.AccumKernelTime(*ctime);
#if(MIO_BN_CPP_PROF == 1)
            printf("kernel2: %7.3f ms   total: %7.3f ms\n", ktime, *ctime);
#endif
        }
        break;
    default: assert(false);
    }
}

} // namespace miopen
