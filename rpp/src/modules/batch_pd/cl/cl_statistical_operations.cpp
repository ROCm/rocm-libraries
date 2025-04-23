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

/******************** thresholding ********************/

RppStatus
thresholding_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, Rpp8u min, Rpp8u max, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};

    handle.AddKernel("", "", "thresholding.cl", "thresholding", vld, vgd, "")(srcPtr,
                                                                              dstPtr,
                                                                              srcSize.height,
                                                                              srcSize.width,
                                                                              channel,
                                                                              min,
                                                                              max);

    return RPP_SUCCESS;
}

RppStatus
thresholding_cl_batch(cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
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

    handle.AddKernel("", "", "thresholding.cl", "thresholding_batch", vld, vgd, "")(srcPtr,
                                                                                    dstPtr,
                                                                                    handle.GetInitHandle()->mem.mgpu.ucharArr[0].ucharmem,
                                                                                    handle.GetInitHandle()->mem.mgpu.ucharArr[1].ucharmem,
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

/******************** min ********************/

RppStatus
min_cl(cl_mem srcPtr1,cl_mem srcPtr2, RppiSize srcSize, cl_mem dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};

    handle.AddKernel("", "", "min.cl", "min", vld, vgd, "")(srcPtr1,
                                                            srcPtr2,
                                                            dstPtr,
                                                            srcSize.height,
                                                            srcSize.width,
                                                            channel);

    return RPP_SUCCESS;
}

RppStatus
min_cl_batch(cl_mem srcPtr1,cl_mem srcPtr2, cl_mem dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
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

    handle.AddKernel("", "", "min.cl", "min_batch", vld, vgd, "")(srcPtr1,
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

/******************** max ********************/

RppStatus
max_cl(cl_mem srcPtr1, cl_mem srcPtr2, RppiSize srcSize, cl_mem dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    std::vector<size_t> vld{32, 32, 1};
    std::vector<size_t> vgd{srcSize.width, srcSize.height, channel};

    handle.AddKernel("", "", "max.cl", "max", vld, vgd, "")(srcPtr1,
                                                            srcPtr2,
                                                            dstPtr,
                                                            srcSize.height,
                                                            srcSize.width,
                                                            channel);

    return RPP_SUCCESS;
}

RppStatus
max_cl_batch(cl_mem srcPtr1,cl_mem srcPtr2, cl_mem dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
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

    handle.AddKernel("", "", "max.cl", "max_batch", vld, vgd, "")(srcPtr1,
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

/******************** min_max_loc ********************/

RppStatus
min_max_loc_cl(cl_mem srcPtr, RppiSize srcSize, Rpp8u* min, Rpp8u* max, Rpp32u* minLoc, Rpp32u* maxLoc, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    int i;

    const int LIST_SIZE = srcSize.height * srcSize.width * channel;
    int numGroups = std::ceil(LIST_SIZE / 256);

    cl_context theContext;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_CONTEXT, sizeof(cl_context), &theContext, NULL);
    cl_device_id theDevice;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_DEVICE, sizeof(cl_device_id), &theDevice, NULL);

    unsigned char minElement = 255;
    unsigned int minLocation;
    unsigned char *partial_min;
    partial_min = (unsigned char *) calloc (numGroups, sizeof(unsigned char));
    unsigned int *partial_min_location;
    partial_min_location = (unsigned int *) calloc (numGroups, sizeof(unsigned int));
    cl_mem b_mem_obj = clCreateBuffer(theContext, CL_MEM_WRITE_ONLY, numGroups * sizeof(unsigned char), NULL, NULL);
    clEnqueueWriteBuffer(handle.GetStream(), b_mem_obj, CL_TRUE, 0, numGroups * sizeof(unsigned char), partial_min, 0, NULL, NULL);
    cl_mem b_mem_obj1 = clCreateBuffer(theContext, CL_MEM_WRITE_ONLY, numGroups * sizeof(unsigned int), NULL, NULL);
    clEnqueueWriteBuffer(handle.GetStream(), b_mem_obj1, CL_TRUE, 0, numGroups * sizeof(unsigned int), partial_min_location, 0, NULL, NULL);

    unsigned char maxElement = 0;
    unsigned int maxLocation;
    unsigned char *partial_max;
    partial_max = (unsigned char *) calloc (numGroups, sizeof(unsigned char));
    unsigned int *partial_max_location;
    partial_max_location = (unsigned int *) calloc (numGroups, sizeof(unsigned int));
    cl_mem c_mem_obj = clCreateBuffer(theContext, CL_MEM_WRITE_ONLY, numGroups * sizeof(unsigned char), NULL, NULL);
    clEnqueueWriteBuffer(handle.GetStream(), c_mem_obj, CL_TRUE, 0, numGroups * sizeof(unsigned char), partial_max, 0, NULL, NULL);
    cl_mem c_mem_obj1 = clCreateBuffer(theContext, CL_MEM_WRITE_ONLY, numGroups * sizeof(unsigned int), NULL, NULL);
    clEnqueueWriteBuffer(handle.GetStream(), c_mem_obj1, CL_TRUE, 0, numGroups * sizeof(unsigned int), partial_max_location, 0, NULL, NULL);
    size_t gDim3[3];
    gDim3[0] = LIST_SIZE;
    gDim3[1] = 1;
    gDim3[2] = 1;
    size_t local_item_size[3];
    local_item_size[0] = 256;
    local_item_size[1] = 1;
    local_item_size[2] = 1;
    std::vector<size_t> vld{local_item_size[0], local_item_size[1], local_item_size[2]};
    std::vector<size_t> vgd{gDim3[0],gDim3[1],gDim3[2]};

    handle.AddKernel("", "", "min_max_loc.cl", "min", vld, vgd, "")(srcPtr,
                                                                    b_mem_obj,
                                                                    b_mem_obj1);

    clEnqueueReadBuffer(handle.GetStream(), b_mem_obj, CL_TRUE, 0, numGroups * sizeof(unsigned char), partial_min, 0, NULL, NULL);
    clEnqueueReadBuffer(handle.GetStream(), b_mem_obj1, CL_TRUE, 0, numGroups * sizeof(unsigned char), partial_min_location, 0, NULL, NULL);

    for(i = 0; i < numGroups; i++)
    {
        if(minElement > partial_min[i])
        {
            minElement = partial_min[i];
            minLocation = partial_min_location[i];

        }
    }
    *min = minElement;
    *minLoc=minLocation;

    std::vector<size_t> vld1{local_item_size[0], local_item_size[1], local_item_size[2]};
    std::vector<size_t> vgd1{gDim3[0],gDim3[1],gDim3[2]};

    handle.AddKernel("", "", "min_max_loc.cl", "max", vld1, vgd1, "")(srcPtr,
                                                                      c_mem_obj,
                                                                      c_mem_obj1);

    clEnqueueReadBuffer(handle.GetStream(), c_mem_obj, CL_TRUE, 0, numGroups * sizeof(unsigned char), partial_max, 0, NULL, NULL);
    clEnqueueReadBuffer(handle.GetStream(), b_mem_obj1, CL_TRUE, 0, numGroups * sizeof(unsigned char), partial_max_location, 0, NULL, NULL);
    for(i = 0; i < numGroups; i++)
    {
        if(maxElement < partial_max[i])
        {
            maxElement = partial_max[i];
            maxLocation = partial_max_location[i];
        }
    }

    *max = maxElement;
    *maxLoc=maxLocation;

    clReleaseMemObject(b_mem_obj);
    free(partial_min);
    clReleaseMemObject(c_mem_obj);
    free(partial_max);
    clReleaseMemObject(b_mem_obj1);
    free(partial_min_location);
    clReleaseMemObject(c_mem_obj1);
    free(partial_max_location);

    return RPP_SUCCESS;
}

RppStatus
min_max_loc_cl_batch(cl_mem srcPtr, Rpp8u *min, Rpp8u *max,
                    Rpp32u *minLoc, Rpp32u *maxLoc, rpp::Handle& handle,
                    RppiChnFormat chnFormat, unsigned int channel)
{
    Rpp32u nBatchSize = handle.GetBatchSize();
    cl_context theContext;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_CONTEXT, sizeof(cl_context), &theContext, NULL);
    cl_device_id theDevice;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_DEVICE, sizeof(cl_device_id), &theDevice, NULL);
    unsigned char *partial_min;
    unsigned int *partial_min_location;
    unsigned char *partial_max;
    unsigned int *partial_max_location;

    int numGroups = 0;

    unsigned int maxHeight, maxWidth;
    maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[0];
    maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[0];

    for(int i = 0 ; i < nBatchSize ; i++)
    {
        if(maxHeight < handle.GetInitHandle()->mem.mgpu.csrcSize.height[i])
            maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        if(maxWidth < handle.GetInitHandle()->mem.mgpu.csrcSize.width[i])
            maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
        int size = 0;
        size = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] *
                handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * channel;
        int group = std::ceil(size / 256);
        if(numGroups < group)
            numGroups = group;
    }

    partial_min = (unsigned char *) calloc (numGroups, sizeof(unsigned char));
    partial_min_location = (unsigned int *) calloc (numGroups, sizeof(unsigned int));
    partial_max = (unsigned char *) calloc (numGroups, sizeof(unsigned char));
    partial_max_location = (unsigned int *) calloc (numGroups, sizeof(unsigned int));

    cl_mem b_mem_obj = clCreateBuffer(theContext, CL_MEM_WRITE_ONLY, numGroups * sizeof(unsigned char), NULL, NULL);
    cl_mem b_mem_obj1 = clCreateBuffer(theContext, CL_MEM_WRITE_ONLY, numGroups * sizeof(unsigned int), NULL, NULL);
    cl_mem c_mem_obj = clCreateBuffer(theContext, CL_MEM_WRITE_ONLY, numGroups * sizeof(unsigned char), NULL, NULL);
    cl_mem c_mem_obj1 = clCreateBuffer(theContext, CL_MEM_WRITE_ONLY, numGroups * sizeof(unsigned int), NULL, NULL);

    cl_mem srcPtr1 = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * maxHeight * maxWidth * channel, NULL, NULL);
    size_t gDim3[3];
    size_t batchIndex = 0;

    for(int x = 0 ; x < nBatchSize ; x++)
    {
        clEnqueueCopyBuffer(handle.GetStream(), srcPtr, srcPtr1, batchIndex, 0, sizeof(unsigned char) *
                            handle.GetInitHandle()->mem.mgpu.csrcSize.width[x] *
                            handle.GetInitHandle()->mem.mgpu.csrcSize.height[x] * channel, 0, NULL, NULL);

        int i;

        int LIST_SIZE = handle.GetInitHandle()->mem.mgpu.csrcSize.height[x] * handle.GetInitHandle()->mem.mgpu.csrcSize.width[x] * channel;
        numGroups = std::ceil(LIST_SIZE / 256);

        unsigned char minElement = 255;
        unsigned int minLocation;

        clEnqueueWriteBuffer(handle.GetStream(), b_mem_obj, CL_TRUE, 0, numGroups * sizeof(unsigned char), partial_min, 0, NULL, NULL);
        clEnqueueWriteBuffer(handle.GetStream(), b_mem_obj1, CL_TRUE, 0, numGroups * sizeof(unsigned int), partial_min_location, 0, NULL, NULL);

        unsigned char maxElement = 0;
        unsigned int maxLocation;

        clEnqueueWriteBuffer(handle.GetStream(), c_mem_obj, CL_TRUE, 0, numGroups * sizeof(unsigned char), partial_max, 0, NULL, NULL);
        clEnqueueWriteBuffer(handle.GetStream(), c_mem_obj1, CL_TRUE, 0, numGroups * sizeof(unsigned int), partial_max_location, 0, NULL, NULL);

        size_t gDim3[3];
        gDim3[0] = LIST_SIZE;
        gDim3[1] = 1;
        gDim3[2] = 1;
        size_t local_item_size[3];
        local_item_size[0] = 256;
        local_item_size[1] = 1;
        local_item_size[2] = 1;
        std::vector<size_t> vld{local_item_size[0], local_item_size[1], local_item_size[2]};
        std::vector<size_t> vgd{gDim3[0],gDim3[1],gDim3[2]};

        handle.AddKernel("", "", "min_max_loc.cl", "min", vld, vgd, "")(srcPtr1,
                                                                        b_mem_obj,
                                                                        b_mem_obj1);

        clEnqueueReadBuffer(handle.GetStream(), b_mem_obj, CL_TRUE, 0, numGroups * sizeof(unsigned char), partial_min, 0, NULL, NULL);
        clEnqueueReadBuffer(handle.GetStream(), b_mem_obj1, CL_TRUE, 0, numGroups * sizeof(unsigned char), partial_min_location, 0, NULL, NULL);

        for(i = 0; i < numGroups; i++)
        {
            if(minElement > partial_min[i])
            {
                minElement = partial_min[i];
                minLocation = partial_min_location[i];

            }
        }
        *min = minElement;
        *minLoc = minLocation;

        std::vector<size_t> vld1{local_item_size[0], local_item_size[1], local_item_size[2]};
        std::vector<size_t> vgd1{gDim3[0],gDim3[1],gDim3[2]};

        handle.AddKernel("", "", "min_max_loc.cl", "max", vld1, vgd1, "")(srcPtr1,
                                                                          c_mem_obj,
                                                                          c_mem_obj1);

        clEnqueueReadBuffer(handle.GetStream(), c_mem_obj, CL_TRUE, 0, numGroups * sizeof(unsigned char), partial_max, 0, NULL, NULL);
        clEnqueueReadBuffer(handle.GetStream(), b_mem_obj1, CL_TRUE, 0, numGroups * sizeof(unsigned char), partial_max_location, 0, NULL, NULL);

        for(i = 0; i < numGroups; i++)
        {
            if(maxElement < partial_max[i])
            {
                maxElement = partial_max[i];
                maxLocation = partial_max_location[i];
            }
        }
        *max = maxElement;
        *maxLoc = maxLocation;

        min++;
        max++;
        minLoc++;
        maxLoc++;

        batchIndex += handle.GetInitHandle()->mem.mgpu.csrcSize.height[x] * handle.GetInitHandle()->mem.mgpu.csrcSize.width[x] * channel * sizeof(unsigned char);
    }
    clReleaseMemObject(b_mem_obj);
    free(partial_min);
    clReleaseMemObject(c_mem_obj);
    free(partial_max);
    clReleaseMemObject(b_mem_obj1);
    free(partial_min_location);
    clReleaseMemObject(c_mem_obj1);
    free(partial_max_location);

    return RPP_SUCCESS;
}

/******************** integral ********************/

RppStatus
integral_cl(cl_mem srcPtr, RppiSize srcSize, cl_mem dstPtr, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    cl_context theContext;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_CONTEXT, sizeof(cl_context), &theContext, NULL);
    cl_device_id theDevice;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_DEVICE, sizeof(cl_device_id), &theDevice, NULL);
    cl_kernel theKernel;
    cl_program theProgram;

    /* FIRST COLUMN */
    if(chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, 1, channel};
        handle.AddKernel("", "", "integral.cl", "integral_pkd_col", vld, vgd, "")(srcPtr,
                                                                                  dstPtr,
                                                                                  srcSize.height,
                                                                                  srcSize.width,
                                                                                  channel);
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.width, 1, channel};
        handle.AddKernel("", "", "integral.cl", "integral_pln_col", vld, vgd, "")(srcPtr,
                                                                                  dstPtr,
                                                                                  srcSize.height,
                                                                                  srcSize.width,
                                                                                  channel);
    }
    /* FIRST ROW */
    if(chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.height, 1, channel};
        handle.AddKernel("", "", "integral.cl", "integral_pkd_row", vld, vgd, "")(srcPtr,
                                                                                  dstPtr,
                                                                                  srcSize.height,
                                                                                  srcSize.width,
                                                                                  channel);
    }
    else
    {
        std::vector<size_t> vld{32, 32, 1};
        std::vector<size_t> vgd{srcSize.height, 1, channel};
        handle.AddKernel("", "", "integral.cl", "integral_pln_row", vld, vgd, "")(srcPtr,
                                                                                  dstPtr,
                                                                                  srcSize.height,
                                                                                  srcSize.width,
                                                                                  channel);
    }
    Rpp32u temp;

    for(int i = 0 ; i < srcSize.height - 1 ; i++)
    {
        if(i + 1 < ((srcSize.height - 1 <= srcSize.width - 1) ? srcSize.height - 1 : srcSize.width - 1))
            temp = i + 1;
        else if(i >= ((srcSize.height - 1 >= srcSize.width - 1) ? srcSize.height - 1 : srcSize.width - 1))
            temp = srcSize.height - i + srcSize.width - 3;
        else
            temp = (srcSize.height - 1 <= srcSize.width - 1) ? srcSize.height - 1 : srcSize.width - 1;

        if(chnFormat == RPPI_CHN_PACKED)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{temp, 1, channel};
            handle.AddKernel("", "", "integral.cl", "integral_up_pkd", vld, vgd, "")(srcPtr,
                                                                                     dstPtr,
                                                                                     srcSize.height,
                                                                                     srcSize.width,
                                                                                     channel,
                                                                                     i,
                                                                                     temp);
        }
        else
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{temp, 1, channel};
            handle.AddKernel("", "", "integral.cl", "integral_up_pln", vld, vgd, "")(srcPtr,
                                                                                     dstPtr,
                                                                                     srcSize.height,
                                                                                     srcSize.width,
                                                                                     channel,
                                                                                     i,
                                                                                     temp);
        }
    }
    for(int i = 0 ; i < srcSize.width - 2 ; i++)
    {

        if(i + 1 + srcSize.height - 1 < ((srcSize.height - 1 <= srcSize.width - 1) ? srcSize.height - 1 : srcSize.width - 1))
            temp = i + 1;
        else if(i + srcSize.height - 1 >= ((srcSize.height - 1 >= srcSize.width - 1) ? srcSize.height - 1 : srcSize.width - 1))
            temp = srcSize.height - (i + srcSize.height - 1) + srcSize.width - 3;
        else
            temp = (srcSize.height - 1 <= srcSize.width - 1) ? srcSize.height - 1 : srcSize.width - 1;

        if(chnFormat == RPPI_CHN_PACKED)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{temp, 1, channel};
            handle.AddKernel("", "", "integral.cl", "integral_low_pkd", vld, vgd, "")(srcPtr,
                                                                                      dstPtr,
                                                                                      srcSize.height,
                                                                                      srcSize.width,
                                                                                      channel,
                                                                                      i,
                                                                                      temp);
        }
        else
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{temp, 1, channel};
            handle.AddKernel("", "", "integral.cl", "integral_low_pln", vld, vgd, "")(srcPtr,
                                                                                      dstPtr,
                                                                                      srcSize.height,
                                                                                      srcSize.width,
                                                                                      channel,
                                                                                      i,
                                                                                      temp);
        }
    }

    return RPP_SUCCESS;
}

RppStatus
integral_cl_batch(cl_mem srcPtr, cl_mem dstPtr, rpp::Handle& handle, RppiChnFormat chnFormat, unsigned int channel)
{
    Rpp32u nBatchSize = handle.GetBatchSize();
    cl_context theContext;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_CONTEXT, sizeof(cl_context), &theContext, NULL);
    cl_device_id theDevice;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_DEVICE, sizeof(cl_device_id), &theDevice, NULL);
    cl_kernel theKernel;
    cl_program theProgram;

    unsigned int maxHeight, maxWidth;
    maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[0];
    maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[0];
    for(int i = 0 ; i < nBatchSize ; i++)
    {
        if(maxHeight < handle.GetInitHandle()->mem.mgpu.csrcSize.height[i])
            maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        if(maxWidth < handle.GetInitHandle()->mem.mgpu.csrcSize.width[i])
            maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
    }

    cl_mem srcPtr1 = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * maxHeight * maxWidth * channel, NULL, NULL);
    cl_mem dstPtr1 = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned int) * maxHeight * maxWidth * channel, NULL, NULL);

    size_t gDim3[3];

    size_t batchIndexSrc = 0;
    size_t batchIndexDst = 0;

    for(int i = 0 ; i < nBatchSize ; i++)
    {
        clEnqueueCopyBuffer(handle.GetStream(), srcPtr, srcPtr1, batchIndexSrc, 0, sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel, 0, NULL, NULL);
        /* FIRST COLUMN */
        gDim3[0] = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
        gDim3[1] = 1;
        gDim3[2] = channel;
        if(chnFormat == RPPI_CHN_PACKED)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};
            handle.AddKernel("", "", "integral.cl", "integral_pkd_col", vld, vgd, "")(srcPtr1,
                                                                                      dstPtr1,
                                                                                      handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                      handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                      channel);
        }
        else
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};
            handle.AddKernel("", "", "integral.cl", "integral_pln_col", vld, vgd, "")(srcPtr1,
                                                                                      dstPtr1,
                                                                                      handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                      handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                      channel);
        }

        /* FIRST ROW */
        gDim3[0] = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        if(chnFormat == RPPI_CHN_PACKED)
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};
            handle.AddKernel("", "", "integral.cl", "integral_pkd_row", vld, vgd, "")(srcPtr1,
                                                                                      dstPtr1,
                                                                                      handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                      handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                      channel);
        }
        else
        {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};
            handle.AddKernel("", "", "integral.cl", "integral_pln_row", vld, vgd, "")(srcPtr1,
                                                                                      dstPtr1,
                                                                                      handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                      handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                      channel);
        }
        Rpp32u temp = 1;

        for(int x = 0 ; x < handle.GetInitHandle()->mem.mgpu.csrcSize.height[x] - 1 ; x++)
        {
            if(x + 1 < ((handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] - 1 <= handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] - 1) ? handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] - 1 : handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] - 1))
                temp = x + 1;
            else if(x >= ((handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] - 1 >= handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] - 1) ? handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] - 1 : handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] - 1))
                temp = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] - x + handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] - 3;
            else
                temp = (handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] - 1 <= handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] - 1) ? handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] - 1 : handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] - 1;
            gDim3[0] = temp;

            if(chnFormat == RPPI_CHN_PACKED)
            {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};
            handle.AddKernel("", "", "integral.cl", "integral_up_pkd", vld, vgd, "")(srcPtr1,
                                                                                     dstPtr1,
                                                                                     handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                     handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                     channel,
                                                                                     x,
                                                                                     temp);
            }
            else
            {
            std::vector<size_t> vld{32, 32, 1};
            std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};
            handle.AddKernel("", "", "integral.cl", "integral_up_pln", vld, vgd, "")(srcPtr1,
                                                                                     dstPtr1,
                                                                                     handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                     handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                     channel,
                                                                                     x,
                                                                                     temp);
            }
        }
        for(int x = 0 ; x < handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] - 2 ; x++)
        {
            if(x + 1 + handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] - 1 < ((handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] - 1 <= handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] - 1) ? handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] - 1 : handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] - 1))
                temp = x + 1;
            else if(x + handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] - 1 >= ((handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] - 1 >= handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] - 1) ? handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] - 1 : handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] - 1))
                temp = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] - (x + handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] - 1) + handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] - 3;
            else
                temp = (handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] - 1 <= handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] - 1) ? handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] - 1 : handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] - 1;
            gDim3[0] = temp;
            if(chnFormat == RPPI_CHN_PACKED)
            {
                std::vector<size_t> vld{32, 32, 1};
                std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};
                handle.AddKernel("", "", "integral.cl", "integral_low_pkd", vld, vgd, "")(srcPtr1,
                                                                                          dstPtr1,
                                                                                          handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                          handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                          channel,
                                                                                          x,
                                                                                          temp);
            }
            else
            {
                std::vector<size_t> vld{32, 32, 1};
                std::vector<size_t> vgd{gDim3[0], gDim3[1], gDim3[2]};
                handle.AddKernel("", "", "integral.cl", "integral_low_pln", vld, vgd, "")(srcPtr1,
                                                                                          dstPtr1,
                                                                                          handle.GetInitHandle()->mem.mgpu.csrcSize.height[i],
                                                                                          handle.GetInitHandle()->mem.mgpu.csrcSize.width[i],
                                                                                          channel,
                                                                                          x,
                                                                                          temp);
            }
        }
        cl_int err = clEnqueueCopyBuffer(handle.GetStream(), dstPtr1, dstPtr, 0, batchIndexDst, sizeof(unsigned int) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * channel, 0, NULL, NULL);
        batchIndexSrc += handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * channel * sizeof(unsigned char);
        batchIndexDst += handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * channel * sizeof(unsigned int);
    }

    return RPP_SUCCESS;
}

/******************** mean_stddev ********************/

RppStatus
mean_stddev_cl(cl_mem srcPtr, RppiSize srcSize, Rpp32f *mean, Rpp32f *stddev, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    int i;

    const int LIST_SIZE = srcSize.height * srcSize.width * channel;
    int numGroups = std::ceil(LIST_SIZE / 256);

    cl_context theContext;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_CONTEXT, sizeof(cl_context), &theContext, NULL);
    cl_device_id theDevice;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_DEVICE, sizeof(cl_device_id), &theDevice, NULL);

    float sum = 0;
    long *partial_sum;
    partial_sum = (long *) calloc (numGroups, sizeof(long));
    cl_mem b_mem_obj = clCreateBuffer(theContext, CL_MEM_WRITE_ONLY, numGroups * sizeof(long), NULL, NULL);
    clEnqueueWriteBuffer(handle.GetStream(), b_mem_obj, CL_TRUE, 0, numGroups * sizeof(long), partial_sum, 0, NULL, NULL);

    float mean_sum = 0;
    float *partial_mean_sum;
    partial_mean_sum = (float *) calloc (numGroups, sizeof(float));
    cl_mem c_mem_obj = clCreateBuffer(theContext, CL_MEM_WRITE_ONLY, numGroups * sizeof(float), NULL, NULL);
    clEnqueueWriteBuffer(handle.GetStream(), c_mem_obj, CL_TRUE, 0, numGroups * sizeof(float), partial_mean_sum, 0, NULL, NULL);

    size_t gDim3[3];
    gDim3[0] = LIST_SIZE;
    gDim3[1] = 1;
    gDim3[2] = 1;
    size_t local_item_size[3];
    local_item_size[0] = 256;
    local_item_size[1] = 1;
    local_item_size[2] = 1;
    std::vector<size_t> vld{local_item_size[0], local_item_size[1], local_item_size[2]};
    std::vector<size_t> vgd{gDim3[0],gDim3[1],gDim3[2]};
    handle.AddKernel("", "", "mean_stddev.cl", "sum", vld, vgd, "")(srcPtr,
                                                                    b_mem_obj);

    clEnqueueReadBuffer(handle.GetStream(), b_mem_obj, CL_TRUE, 0, numGroups * sizeof(long), partial_sum, 0, NULL, NULL);

    for(i = 0; i < numGroups; i++)
    {
        sum += (float)partial_sum[i];
    }

    *mean = (sum) / LIST_SIZE ;

    float meanCopy = *mean;

    handle.AddKernel("", "", "mean_stddev.cl", "mean_stddev", vld, vgd, "")(srcPtr,
                                                                            c_mem_obj,
                                                                            meanCopy);

    clEnqueueReadBuffer(handle.GetStream(), c_mem_obj, CL_TRUE, 0, numGroups * sizeof(float), partial_mean_sum, 0, NULL, NULL);
    for(i = 0; i < numGroups; i++)
    {
        mean_sum += partial_mean_sum[i];
    }

    mean_sum = mean_sum / LIST_SIZE ;
    *stddev = mean_sum;

    clReleaseMemObject(b_mem_obj);
    free(partial_sum);
    clReleaseMemObject(c_mem_obj);
    free(partial_mean_sum);
    return RPP_SUCCESS;
}

RppStatus
mean_stddev_cl_batch(cl_mem srcPtr, Rpp32f *mean, Rpp32f *stddev, rpp::Handle& handle,
                        RppiChnFormat chnFormat, unsigned int channel)
{
    Rpp32u nBatchSize = handle.GetBatchSize();
    cl_context theContext;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_CONTEXT, sizeof(cl_context), &theContext, NULL);
    cl_device_id theDevice;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_DEVICE, sizeof(cl_device_id), &theDevice, NULL);

    long *partial_sum;
    float *partial_mean_sum;

    int numGroups = 0;

    unsigned int maxHeight, maxWidth;
    maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[0];
    maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[0];

    for(int i = 0 ; i < nBatchSize ; i++)
    {
        if(maxHeight < handle.GetInitHandle()->mem.mgpu.csrcSize.height[i])
            maxHeight = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i];
        if(maxWidth < handle.GetInitHandle()->mem.mgpu.csrcSize.width[i])
            maxWidth = handle.GetInitHandle()->mem.mgpu.csrcSize.width[i];
        int size = 0;
        size = handle.GetInitHandle()->mem.mgpu.csrcSize.height[i] * handle.GetInitHandle()->mem.mgpu.csrcSize.width[i] * channel;
        int group = std::ceil(size / 256);
        if(numGroups < group)
            numGroups = group;
    }

    partial_sum = (long *) calloc (numGroups, sizeof(long));
    partial_mean_sum = (float *) calloc (numGroups, sizeof(float));

    cl_mem b_mem_obj = clCreateBuffer(theContext, CL_MEM_WRITE_ONLY, numGroups * sizeof(long), NULL, NULL);
    cl_mem c_mem_obj = clCreateBuffer(theContext, CL_MEM_WRITE_ONLY, numGroups * sizeof(float), NULL, NULL);

    cl_mem srcPtr1 = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned char) * maxHeight * maxWidth * channel, NULL, NULL);
    size_t gDim3[3];

    size_t batchIndex = 0;

    for(int x = 0 ; x < nBatchSize ; x++)
    {
        clEnqueueCopyBuffer(handle.GetStream(), srcPtr, srcPtr1, batchIndex, 0, sizeof(unsigned char) * handle.GetInitHandle()->mem.mgpu.csrcSize.width[x] * handle.GetInitHandle()->mem.mgpu.csrcSize.height[x] * channel, 0, NULL, NULL);
        int i;

        int LIST_SIZE = handle.GetInitHandle()->mem.mgpu.csrcSize.height[x] * handle.GetInitHandle()->mem.mgpu.csrcSize.width[x] * channel;
        numGroups = std::ceil(LIST_SIZE / 256);

        float sum = 0;
        clEnqueueWriteBuffer(handle.GetStream(), b_mem_obj, CL_TRUE, 0, numGroups * sizeof(long), partial_sum, 0, NULL, NULL);

        float mean_sum = 0;
        clEnqueueWriteBuffer(handle.GetStream(), c_mem_obj, CL_TRUE, 0, numGroups * sizeof(float), partial_mean_sum, 0, NULL, NULL);

        size_t gDim3[3];
        gDim3[0] = LIST_SIZE;
        gDim3[1] = 1;
        gDim3[2] = 1;
        size_t local_item_size[3];
        local_item_size[0] = 256;
        local_item_size[1] = 1;
        local_item_size[2] = 1;
        std::vector<size_t> vld{local_item_size[0], local_item_size[1], local_item_size[2]};
        std::vector<size_t> vgd{gDim3[0],gDim3[1],gDim3[2]};

        handle.AddKernel("", "", "mean_stddev.cl", "sum", vld, vgd, "")(srcPtr1,
                                                                        b_mem_obj);

        clEnqueueReadBuffer(handle.GetStream(), b_mem_obj, CL_TRUE, 0, numGroups * sizeof(long), partial_sum, 0, NULL, NULL);

        for(i = 0; i < numGroups; i++)
        {
            sum += (float)partial_sum[i];
        }

        *mean = (sum) / LIST_SIZE ;
        float meanCopy = *mean;

        handle.AddKernel("", "", "mean_stddev.cl", "sum", vld, vgd, "")(srcPtr1,
                                                                        c_mem_obj,
                                                                        meanCopy);

        clEnqueueReadBuffer(handle.GetStream(), c_mem_obj, CL_TRUE, 0, numGroups * sizeof(float), partial_mean_sum, 0, NULL, NULL);
        for(i = 0; i < numGroups; i++)
        {
            mean_sum += partial_mean_sum[i];
        }

        mean_sum = mean_sum / LIST_SIZE ;
        *stddev = mean_sum;

        stddev++;
        mean++;

        batchIndex += handle.GetInitHandle()->mem.mgpu.csrcSize.height[x] * handle.GetInitHandle()->mem.mgpu.csrcSize.width[x] * channel * sizeof(unsigned char);
    }

    clReleaseMemObject(b_mem_obj);
    free(partial_sum);
    clReleaseMemObject(c_mem_obj);
    free(partial_mean_sum);

    return RPP_SUCCESS;
}

/******************** histogram ********************/

RppStatus
histogram_cl(cl_mem srcPtr, RppiSize srcSize, Rpp32u* outputHistogram, Rpp32u bins, RppiChnFormat chnFormat, unsigned int channel, rpp::Handle& handle)
{
    cl_int err;

    cl_context theContext;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_CONTEXT, sizeof(cl_context), &theContext, NULL);
    cl_device_id theDevice;
    clGetCommandQueueInfo(handle.GetStream(), CL_QUEUE_DEVICE, sizeof(cl_device_id), &theDevice, NULL);

    cl_kernel theKernel;
    cl_program theProgram;
    unsigned int numGroups;

    size_t lDim3[3];
    size_t gDim3[3];
    int num_pixels_per_work_item = 16;

    gDim3[0] = srcSize.width / num_pixels_per_work_item + 1;
    gDim3[1] = srcSize.height / num_pixels_per_work_item + 1;
    lDim3[0] = num_pixels_per_work_item;
    lDim3[1] = num_pixels_per_work_item;
    gDim3[2] = 1;
    lDim3[2] = 1;

    numGroups = gDim3[0] * gDim3[1];
    gDim3[0] = srcSize.width;
    gDim3[1] = srcSize.height;

    cl_mem partialHistogram = clCreateBuffer(theContext, CL_MEM_READ_WRITE, sizeof(unsigned int)*256*channel*numGroups, NULL, NULL);
    cl_mem histogram = clCreateBuffer(theContext, CL_MEM_READ_ONLY, sizeof(unsigned int)*256*channel, NULL, NULL);

    if (chnFormat == RPPI_CHN_PLANAR)
    {
        std::vector<size_t> vld{lDim3[0], lDim3[1], lDim3[2]};
        std::vector<size_t> vgd{gDim3[0],gDim3[1],gDim3[2]};
        if(channel > 1)
        {
            handle.AddKernel("", "", "hist.cl", "partial_histogram_pln", vld, vgd, "")(srcPtr,
                                                                                       partialHistogram,
                                                                                       srcSize.width,
                                                                                       srcSize.height,
                                                                                       channel);
        }
        else
        {
           handle.AddKernel("", "", "hist.cl", "partial_histogram_pln1", vld, vgd, "")(srcPtr,
                                                                                       partialHistogram,
                                                                                       srcSize.width,
                                                                                       srcSize.height,
                                                                                       channel);
        }
    }
    else if (chnFormat == RPPI_CHN_PACKED)
    {
        std::vector<size_t> vld{lDim3[0], lDim3[1], lDim3[2]};
        std::vector<size_t> vgd{gDim3[0],gDim3[1],gDim3[2]};

        handle.AddKernel("", "", "histogram.cl", "partial_histogram_pkd", vld, vgd, "")(srcPtr,
                                                                                        partialHistogram,
                                                                                        srcSize.width,
                                                                                        srcSize.height,
                                                                                        channel);
    }
    else
    {
        std::cerr << "Internal error: Unknown Channel format";
    }

    gDim3[0] = 256 * channel;
    lDim3[0] = 256;
    gDim3[1] = 1;
    gDim3[2] = 1;
    lDim3[1] = 1;
    lDim3[2] = 1;
    std::vector<size_t> vld{lDim3[0], lDim3[1], lDim3[2]};
    std::vector<size_t> vgd{gDim3[0],gDim3[1],gDim3[2]};

    handle.AddKernel("", "", "hist.cl", "histogram_sum_partial", vld, vgd, "")(partialHistogram,
                                                                               histogram,
                                                                               numGroups, channel);

    const unsigned int totalBin = channel * 256;
    unsigned int *tempBin = (unsigned int *)calloc(totalBin, sizeof(unsigned int));

    clEnqueueReadBuffer(handle.GetStream(), histogram, CL_TRUE, 0, sizeof(unsigned int) * 256 * channel, tempBin, 0, NULL, NULL );
    int noOfValuesInBins = (256 * channel) /bins;
    for(int i = 0 ; i < bins ; i++)
    {
        *outputHistogram=0;
        for(int j = 0 ; j < noOfValuesInBins ; j++)
        {
            *outputHistogram += tempBin[i*noOfValuesInBins+j];
        }
        outputHistogram++;
    }

    clReleaseMemObject(partialHistogram);
    clReleaseMemObject(histogram);
    free(tempBin);
    return RPP_SUCCESS;
}