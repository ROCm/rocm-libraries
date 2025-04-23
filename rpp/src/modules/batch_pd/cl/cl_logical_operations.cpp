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

#include "rpp_cl_common.hpp"
#include "cl_declarations.hpp"

/******************** bitwise_AND ********************/

RppStatus
bitwise_AND_cl(cl_mem srcPtr1,cl_mem srcPtr2, RppiSize srcSize, cl_mem dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};

    handle.AddKernel("", "", "bitwise_AND.cl", "bitwise_AND", vld, vgd, "")(srcPtr1,
                                                                            srcPtr2,
                                                                            dstPtr,
                                                                            srcSize.height,
                                                                            srcSize.width,
                                                                            channel);

    return RPP_SUCCESS;
}

RppStatus
bitwise_AND_cl_batch(cl_mem srcPtr1,cl_mem srcPtr2, cl_mem dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};

    handle.AddKernel("", "", "bitwise_AND.cl", "bitwise_AND_batch", vld, vgd, "")(srcPtr1,
                                                                                  srcPtr2,
                                                                                  dstPtr,
                                                                                  handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                                  handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                                  handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                                  handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                                  handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                                  handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                                  handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                                  handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                                  channel,
                                                                                  handle.GetInitHandle()->mem.mgpu.inc,
                                                                                  plnpkdind);
    return RPP_SUCCESS;
}

/******************** bitwise_NOT ********************/

RppStatus
bitwise_NOT_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{srcSize.width,srcSize.height,channel};

    handle.AddKernel("", "", "bitwise_NOT.cl", "bitwise_NOT", vld, vgd, "")(srcPtr,
                                                                            dstPtr,
                                                                            srcSize.height,
                                                                            srcSize.width,
                                                                            channel);

    return RPP_SUCCESS;
}

RppStatus
bitwise_NOT_cl_batch(cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};

    handle.AddKernel("", "", "bitwise_NOT.cl", "bitwise_NOT_batch", vld, vgd, "")(srcPtr,
                                                                                  dstPtr,
                                                                                  handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                                  handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                                  handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                                  handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                                  handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                                  handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                                  handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                                  handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                                  channel,
                                                                                  handle.GetInitHandle()->mem.mgpu.inc,
                                                                                  plnpkdind);

    return RPP_SUCCESS;
}

/******************** exclusive_OR ********************/

RppStatus
exclusive_OR_cl(cl_mem srcPtr1,cl_mem srcPtr2, RppiSize srcSize, cl_mem dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};

    handle.AddKernel("", "", "exclusive_OR.cl", "exclusive_OR", vld, vgd, "")(srcPtr1,
                                                                              srcPtr2,
                                                                              dstPtr,
                                                                              srcSize.height,
                                                                              srcSize.width,
                                                                              channel);

    return RPP_SUCCESS;
}

RppStatus
exclusive_OR_cl_batch(cl_mem srcPtr1,cl_mem srcPtr2, cl_mem dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};

    handle.AddKernel("", "", "exclusive_OR.cl", "exclusive_OR_batch", vld, vgd, "")(srcPtr1,
                                                                                    srcPtr2,
                                                                                    dstPtr,
                                                                                    handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                                    handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                                    handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                                    handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                                    handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                                    handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                                    handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                                    handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                                    channel,
                                                                                    handle.GetInitHandle()->mem.mgpu.inc,
                                                                                    plnpkdind);

    return RPP_SUCCESS;
}

/******************** inclusive_OR ********************/

RppStatus
inclusive_OR_cl(cl_mem srcPtr1,cl_mem srcPtr2, RppiSize srcSize, cl_mem dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};

    handle.AddKernel("", "", "inclusive_OR.cl", "inclusive_OR", vld, vgd, "")(srcPtr1,
                                                                            srcPtr2,
                                                                            dstPtr,
                                                                            srcSize.height,
                                                                            srcSize.width,
                                                                            channel);

    return RPP_SUCCESS;
}

RppStatus
inclusive_OR_cl_batch(cl_mem srcPtr1,cl_mem srcPtr2, cl_mem dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if(chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};

    handle.AddKernel("", "", "inclusive_OR.cl", "inclusive_OR_batch", vld, vgd, "")(srcPtr1,
                                                                                    srcPtr2,
                                                                                    dstPtr,
                                                                                    handle.GetInitHandle()->mem.mgpu.roiPoints.x,
                                                                                    handle.GetInitHandle()->mem.mgpu.roiPoints.roiWidth,
                                                                                    handle.GetInitHandle()->mem.mgpu.roiPoints.y,
                                                                                    handle.GetInitHandle()->mem.mgpu.roiPoints.roiHeight,
                                                                                    handle.GetInitHandle()->mem.mgpu.srcSize.height,
                                                                                    handle.GetInitHandle()->mem.mgpu.srcSize.width,
                                                                                    handle.GetInitHandle()->mem.mgpu.maxSrcSize.width,
                                                                                    handle.GetInitHandle()->mem.mgpu.srcBatchIndex,
                                                                                    channel,
                                                                                    handle.GetInitHandle()->mem.mgpu.inc,
                                                                                    plnpkdind);

    return RPP_SUCCESS;
}