/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "hip_declarations.hpp"
#include "rpp_hip_host_decls.hpp"

/******************** dilate ********************/

RppStatus
dilate_hip ( Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{

    return RPP_SUCCESS;
}


RppStatus
dilate_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    hip_exec_dilate_batch(srcPtr, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

/******************** erode ********************/

RppStatus
erode_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    return RPP_SUCCESS;
}

RppStatus
erode_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    hip_exec_erode_batch(srcPtr, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}