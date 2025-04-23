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

/******************** absolute_difference ********************/

RppStatus
absolute_difference_cl(cl_mem srcPtr1, cl_mem srcPtr2, RppiSize srcSize, cl_mem dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle &handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};

    handle.AddKernel("", "", "absolute_difference.cl", "absolute_difference", vld, vgd, "")(srcPtr1,
                                                                                            srcPtr2,
                                                                                            dstPtr,
                                                                                            srcSize.height,
                                                                                            srcSize.width,
                                                                                            channel);

    return RPP_SUCCESS;
}

RppStatus
absolute_difference_cl_batch(cl_mem srcPtr1, cl_mem srcPtr2, cl_mem dstPtr, rpp::Handle &handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if (chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};

    handle.AddKernel("", "", "absolute_difference.cl", "absolute_difference_batch", vld, vgd, "")(srcPtr1,
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

/******************** accumulate ********************/

RppStatus
accumulate_cl(cl_mem srcPtr1, cl_mem srcPtr2, RppiSize srcSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle &handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};

    handle.AddKernel("", "", "accumulate.cl", "accumulate", vld, vgd, "")(srcPtr1,
                                                                          srcPtr2,
                                                                          srcSize.height,
                                                                          srcSize.width,
                                                                          channel);

    return RPP_SUCCESS;
}

RppStatus
accumulate_cl_batch(cl_mem srcPtr1, cl_mem srcPtr2, rpp::Handle &handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if (chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};

    handle.AddKernel("", "", "accumulate.cl", "accumulate_batch", vld, vgd, "")(srcPtr1, srcPtr2,
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

/******************** accumulate_weighted ********************/

RppStatus
accumulate_weighted_cl(cl_mem srcPtr1, cl_mem srcPtr2, RppiSize srcSize, Rpp32f alpha, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle &handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};

    handle.AddKernel("", "", "accumulate.cl", "accumulate_weighted", vld, vgd, "")(srcPtr1,
                                                                                   srcPtr2,
                                                                                   alpha,
                                                                                   srcSize.height,
                                                                                   srcSize.width,
                                                                                   channel);

    return RPP_SUCCESS;
}

RppStatus
accumulate_weighted_cl_batch(cl_mem srcPtr1, cl_mem srcPtr2, rpp::Handle &handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if (chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};

    handle.AddKernel("", "", "accumulate.cl", "accumulate_weighted_batch", vld, vgd, "")(srcPtr1,
                                                                                         srcPtr2,
                                                                                         handle.GetInitHandle()->mem.mgpu.floatArr[0].floatmem,
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

/******************** add ********************/

RppStatus
add_cl(cl_mem srcPtr1, cl_mem srcPtr2, RppiSize srcSize, cl_mem dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle &handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};

    handle.AddKernel("", "", "add.cl", "add", vld, vgd, "")(srcPtr1,
                                                            srcPtr2,
                                                            dstPtr,
                                                            srcSize.height,
                                                            srcSize.width,
                                                            channel);

    return RPP_SUCCESS;
}

RppStatus
add_cl_batch(cl_mem srcPtr1, cl_mem srcPtr2, cl_mem dstPtr, rpp::Handle &handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if (chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};

    handle.AddKernel("", "", "add.cl", "add_batch", vld, vgd, "")(srcPtr1,
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

/******************** subtract ********************/

RppStatus
subtract_cl(cl_mem srcPtr1, cl_mem srcPtr2, RppiSize srcSize, cl_mem dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle &handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};

    handle.AddKernel("", "", "subtract.cl", "subtract", vld, vgd, "")(srcPtr1,
                                                                      srcPtr2,
                                                                      dstPtr,
                                                                      srcSize.height,
                                                                      srcSize.width,
                                                                      channel);

    return RPP_SUCCESS;
}

RppStatus
subtract_cl_batch(cl_mem srcPtr1, cl_mem srcPtr2, cl_mem dstPtr, rpp::Handle &handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if (chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};

    handle.AddKernel("", "", "subtract.cl", "subtract_batch", vld, vgd, "")(srcPtr1,
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

/******************** magnitude ********************/

RppStatus
magnitude_cl(cl_mem srcPtr1, cl_mem srcPtr2, RppiSize srcSize, cl_mem dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle &handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};

    handle.AddKernel("", "", "magnitude.cl", "magnitude", vld, vgd, "")(srcPtr1,
                                                                        srcPtr2,
                                                                        dstPtr,
                                                                        srcSize.height,
                                                                        srcSize.width,
                                                                        channel);

    return RPP_SUCCESS;
}

RppStatus
magnitude_cl_batch(cl_mem srcPtr1, cl_mem srcPtr2, cl_mem dstPtr, rpp::Handle &handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if (chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};

    handle.AddKernel("", "", "magnitude.cl", "magnitude_batch", vld, vgd, "")(srcPtr1,
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

/******************** multiply ********************/

RppStatus
multiply_cl(cl_mem srcPtr1, cl_mem srcPtr2, RppiSize srcSize, cl_mem dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle &handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};

    handle.AddKernel("", "", "multiply.cl", "multiply", vld, vgd, "")(srcPtr1,
                                                                      srcPtr2,
                                                                      dstPtr,
                                                                      srcSize.height,
                                                                      srcSize.width,
                                                                      channel);

    return RPP_SUCCESS;
}

RppStatus
multiply_cl_batch(cl_mem srcPtr1, cl_mem srcPtr2, cl_mem dstPtr, rpp::Handle &handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if (chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};

    handle.AddKernel("", "", "multiply.cl", "multiply_batch", vld, vgd, "")(srcPtr1,
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

/******************** phase ********************/

RppStatus
phase_cl(cl_mem srcPtr1, cl_mem srcPtr2, RppiSize srcSize, cl_mem dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle &handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};

    handle.AddKernel("", "", "phase.cl", "phase", vld, vgd, "")(srcPtr1,
                                                                srcPtr2,
                                                                dstPtr,
                                                                srcSize.height,
                                                                srcSize.width,
                                                                channel);

    return RPP_SUCCESS;
}

RppStatus
phase_cl_batch(cl_mem srcPtr1, cl_mem srcPtr2, cl_mem dstPtr, rpp::Handle &handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if (chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};

    handle.AddKernel("", "", "phase.cl", "phase_batch", vld, vgd, "")(srcPtr1,
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

/******************** accumulate_squared ********************/

RppStatus
accumulate_squared_cl(cl_mem srcPtr, RppiSize srcSize, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle &handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};

    handle.AddKernel("", "", "accumulate.cl", "accumulate_squared", vld, vgd, "")(srcPtr,
                                                                                  srcSize.height,
                                                                                  srcSize.width,
                                                                                  channel);

    return RPP_SUCCESS;
}

RppStatus
accumulate_squared_cl_batch(cl_mem srcPtr, rpp::Handle &handle, RppiChnFormat chnFormat, unsigned int channel)
{
    int plnpkdind;
    if (chnFormat == RPPI_CHN_PLANAR)
        plnpkdind = 1;
    else
        plnpkdind = 3;
    Rpp32u max_height, max_width;
    max_size(handle.GetInitHandle()->mem.mgpu.csrcSize.height, handle.GetInitHandle()->mem.mgpu.csrcSize.width, handle.GetBatchSize(), &max_height, &max_width);
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{max_width, max_height, handle.GetBatchSize()};

    handle.AddKernel("", "", "accumulate.cl", "accumulate_squared_batch", vld, vgd, "")(srcPtr,
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

/******************** tensor_add ********************/

RppStatus
tensor_add_cl(Rpp32u tensorDimension, Rpp32u *tensorDimensionValues, cl_mem srcPtr1, cl_mem srcPtr2, cl_mem dstPtr, rpp::Handle &handle)
{
    size_t gDim3[3];
    if (tensorDimension == 1)
    {
        gDim3[0] = tensorDimensionValues[0];
        gDim3[1] = 1;
        gDim3[2] = 1;
    }
    else if (tensorDimension == 2)
    {
        gDim3[0] = tensorDimensionValues[0];
        gDim3[1] = tensorDimensionValues[1];
        gDim3[2] = 1;
    }
    else
    {
        gDim3[0] = tensorDimensionValues[0];
        gDim3[1] = tensorDimensionValues[1];
        int value = 1;
        for (int i = 2; i < tensorDimension; i++)
        {
            value *= tensorDimensionValues[i];
        }
        gDim3[2] = value;
    }
    unsigned int dim1, dim2, dim3;
    dim1 = gDim3[0];
    dim2 = gDim3[1];
    dim3 = gDim3[2];
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};

    handle.AddKernel("", "", "tensor.cl", "tensor_add", vld, vgd, "")(tensorDimension,
                                                                      srcPtr1,
                                                                      srcPtr2,
                                                                      dstPtr,
                                                                      dim1,
                                                                      dim2,
                                                                      dim3);

    return RPP_SUCCESS;
}

/******************** tensor_subtract ********************/

RppStatus
tensor_subtract_cl(Rpp32u tensorDimension, Rpp32u *tensorDimensionValues, cl_mem srcPtr1, cl_mem srcPtr2, cl_mem dstPtr, rpp::Handle &handle)
{
    size_t gDim3[3];
    if (tensorDimension == 1)
    {
        gDim3[0] = tensorDimensionValues[0];
        gDim3[1] = 1;
        gDim3[2] = 1;
    }
    else if (tensorDimension == 2)
    {
        gDim3[0] = tensorDimensionValues[0];
        gDim3[1] = tensorDimensionValues[1];
        gDim3[2] = 1;
    }
    else
    {
        gDim3[0] = tensorDimensionValues[0];
        gDim3[1] = tensorDimensionValues[1];
        int value = 1;
        for (int i = 2; i < tensorDimension; i++)
        {
            value *= tensorDimensionValues[i];
        }
        gDim3[2] = value;
    }
    unsigned int dim1, dim2, dim3;
    dim1 = gDim3[0];
    dim2 = gDim3[1];
    dim3 = gDim3[2];
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};

    handle.AddKernel("", "", "tensor.cl", "tensor_subtract", vld, vgd, "")(tensorDimension,
                                                                           srcPtr1,
                                                                           srcPtr2,
                                                                           dstPtr,
                                                                           dim1,
                                                                           dim2,
                                                                           dim3);

    return RPP_SUCCESS;
}

/******************** tensor_multiply ********************/

RppStatus
tensor_multiply_cl(Rpp32u tensorDimension, Rpp32u *tensorDimensionValues, cl_mem srcPtr1, cl_mem srcPtr2, cl_mem dstPtr, rpp::Handle &handle)
{
    size_t gDim3[3];
    if (tensorDimension == 1)
    {
        gDim3[0] = tensorDimensionValues[0];
        gDim3[1] = 1;
        gDim3[2] = 1;
    }
    else if (tensorDimension == 2)
    {
        gDim3[0] = tensorDimensionValues[0];
        gDim3[1] = tensorDimensionValues[1];
        gDim3[2] = 1;
    }
    else
    {
        gDim3[0] = tensorDimensionValues[0];
        gDim3[1] = tensorDimensionValues[1];
        int value = 1;
        for (int i = 2; i < tensorDimension; i++)
        {
            value *= tensorDimensionValues[i];
        }
        gDim3[2] = value;
    }
    unsigned int dim1, dim2, dim3;
    dim1 = gDim3[0];
    dim2 = gDim3[1];
    dim3 = gDim3[2];
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};

    handle.AddKernel("", "", "tensor.cl", "tensor_multiply", vld, vgd, "")(tensorDimension,
                                                                           srcPtr1,
                                                                           srcPtr2,
                                                                           dstPtr,
                                                                           dim1,
                                                                           dim2,
                                                                           dim3);

    return RPP_SUCCESS;
}

/******************** tensor_matrix_multiply ********************/

RppStatus
tensor_matrix_multiply_cl(cl_mem srcPtr1, cl_mem srcPtr2, Rpp32u *tensorDimensionValues1, Rpp32u *tensorDimensionValues2, cl_mem dstPtr, rpp::Handle &handle)
{
    size_t gDim3[3];
    gDim3[0] = tensorDimensionValues2[1];
    gDim3[1] = tensorDimensionValues1[0];
    gDim3[2] = 1;
    unsigned int a, b, c, d;
    a = tensorDimensionValues1[0];
    b = tensorDimensionValues1[1];
    c = tensorDimensionValues2[0];
    d = tensorDimensionValues2[1];
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};

    handle.AddKernel("", "", "tensor.cl", "tensor_matrix_multiply", vld, vgd, "")(srcPtr1,
                                                                                  srcPtr2,
                                                                                  dstPtr,
                                                                                  a,
                                                                                  b,
                                                                                  c,
                                                                                  d);

    return RPP_SUCCESS;
}
