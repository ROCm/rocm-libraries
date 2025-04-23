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
#define UNROLL 1

// This function takes a positive integer and rounds it up to the nearest multiple of another provided integer
unsigned int roundUp(unsigned int value, unsigned int multiple)
{
  // Determine how far past the nearest multiple the value is
  unsigned int remainder = value % multiple;

  // Add the difference to make the value a multiple
  if(remainder != 0) {
          value += (multiple-remainder);
  }

  return value;
}

/******************** dilate ********************/

RppStatus
dilate_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    cl_kernel theKernel;
    cl_program theProgram;
    if(chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width,srcSize.height,channel};

#if UNROLL
        handle.AddKernel("", "", "dilate.cl", "dilate_unroll", vld, vgd, "")(srcPtr,
                                                                             dstPtr,
                                                                             srcSize.height,
                                                                             srcSize.width,
                                                                             channel,
                                                                             kernelSize);
#else
        std::cerr << "coming in pkd" << std::endl;
        handle.AddKernel("", "", "dilate.cl", "dilate_pkd", vld, vgd, "")(srcPtr,
                                                                          dstPtr,
                                                                          srcSize.height,
                                                                          srcSize.width,
                                                                          channel,
                                                                          kernelSize);
#endif

    }
    else
    {
        std::vector<size_t> vld{16, 16, 1};
        std::vector<size_t> vgd{srcSize.width / 16 , srcSize.height ,channel};

#if UNROLL
        handle.AddKernel("", "", "dilate.cl", "dilate_optimized_trail", vld, vgd, "")(srcPtr,
                                                                                      dstPtr,
                                                                                      srcSize.height,
                                                                                      srcSize.width,
                                                                                      channel,
                                                                                      kernelSize);
#else
        handle.AddKernel("", "", "dilate.cl", "dilate_pln", vld, vgd, "")(srcPtr,
                                                                          dstPtr,
                                                                          srcSize.height,
                                                                          srcSize.width,
                                                                          channel,
                                                                          kernelSize);
#endif

    }

    return RPP_SUCCESS;
}

RppStatus
dilate_cl_batch(cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
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

    handle.AddKernel("", "", "dilate.cl", "dilate_batch", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
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

/******************** erode ********************/

RppStatus
erode_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, Rpp32u kernelSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    cl_kernel theKernel;
    cl_program theProgram;
    if(chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width,srcSize.height,channel};

        handle.AddKernel("", "", "erode.cl", "erode_pkd", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel,
                                                                        kernelSize);
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width,srcSize.height,channel};

        handle.AddKernel("", "", "erode.cl", "erode_pln", vld, vgd, "")(srcPtr,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel,
                                                                        kernelSize);
    }

    return RPP_SUCCESS;
}


RppStatus
erode_cl_batch(cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
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

    handle.AddKernel("", "", "erode.cl", "erode_batch", vld, vgd, "")(srcPtr,
                                                                      dstPtr,
                                                                      handle.GetInitHandle()->mem.mgpu.uintArr[0].uintmem,
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