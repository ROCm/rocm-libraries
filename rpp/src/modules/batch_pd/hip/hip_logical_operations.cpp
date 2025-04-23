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

/******************** bitwise_AND ********************/

RppStatus
bitwise_AND_hip(Rpp8u* srcPtr1,Rpp8u* srcPtr2, RppiSize srcSize, Rpp8u* dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, channel};

    handle.AddKernel("", "", "bitwise_AND.cpp", "bitwise_AND", vld, vgd, "")(srcPtr1,
                                                                             srcPtr2,
                                                                             dstPtr,
                                                                             srcSize.height,
                                                                             srcSize.width,
                                                                             channel);

    return RPP_SUCCESS;
}

RppStatus
bitwise_AND_hip_batch(Rpp8u* srcPtr1,Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    hip_exec_bitwise_AND_batch(srcPtr1, srcPtr2, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

/******************** bitwise_NOT ********************/

RppStatus
bitwise_NOT_hip(Rpp8u* srcPtr, RppiSize srcSize, Rpp8u* dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31,channel};

    handle.AddKernel("", "", "bitwise_NOT.cpp", "bitwise_NOT", vld, vgd, "")(srcPtr,
                                                                             dstPtr,
                                                                             srcSize.height,
                                                                             srcSize.width,
                                                                             channel);

    return RPP_SUCCESS;
}

RppStatus
bitwise_NOT_hip_batch(Rpp8u* srcPtr, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    hip_exec_bitwise_NOT_batch(srcPtr, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

/******************** exclusive_OR ********************/

RppStatus
exclusive_OR_hip(Rpp8u* srcPtr1,Rpp8u* srcPtr2, RppiSize srcSize, Rpp8u* dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, channel};

    handle.AddKernel("", "", "exclusive_OR.cpp", "exclusive_OR", vld, vgd, "")(srcPtr1,
                                                                               srcPtr2,
                                                                               dstPtr,
                                                                               srcSize.height,
                                                                               srcSize.width,
                                                                               channel);

    return RPP_SUCCESS;
}

RppStatus
exclusive_OR_hip_batch(Rpp8u* srcPtr1,Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    hip_exec_exclusive_OR_batch(srcPtr1, srcPtr2, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}

/******************** inclusive_OR ********************/

RppStatus
inclusive_OR_hip(Rpp8u* srcPtr1,Rpp8u* srcPtr2, RppiSize srcSize, Rpp8u* dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{(srcSize.width + 31) & ~31, (srcSize.height + 31) & ~31, channel};

    handle.AddKernel("", "", "inclusive_OR.cpp", "inclusive_OR", vld, vgd, "")(srcPtr1,
                                                                               srcPtr2,
                                                                               dstPtr,
                                                                               srcSize.height,
                                                                               srcSize.width,
                                                                               channel);

    return RPP_SUCCESS;
}

RppStatus
inclusive_OR_hip_batch(Rpp8u* srcPtr1,Rpp8u* srcPtr2, Rpp8u* dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);

    hip_exec_inclusive_OR_batch(srcPtr1, srcPtr2, dstPtr, handle, chnFormat, channel, plnpkdind, max_height, max_width);

    return RPP_SUCCESS;
}