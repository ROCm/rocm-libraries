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
/********************** Blur ************************/
float box_3x3[] = {
0.111, 0.111, 0.111,
0.111, 0.111, 0.111,
0.111, 0.111, 0.111,
};


// cl_int
// box_filter_cl(cl_mem srcPtr, RppiSize srcSize,
//                 cl_mem dstPtr, unsigned int filterSize,
//                 RppiChnFormat chnFormat, unsigned int channel,
//                 rpp::Handle& handle)
// {
//     unsigned short counter=0;
//     cl_int err;

//     float* filterBuffer;
//     if (filterSize == 3) filterBuffer= box_3x3;
//     else  std::cerr << "Unimplemeted kernel Size";

//     cl_context theContext;
//     clGetCommandQueueInfo(  handle.GetStream(),
//                             CL_QUEUE_CONTEXT,
//                             sizeof(cl_context), &theContext, NULL);
//     cl_mem filtPtr = clCreateBuffer(theContext, CL_MEM_READ_ONLY,
//                                     sizeof(float)*filterSize*filterSize, NULL, NULL);
//     err = clEnqueueWriteBuffer(handle.GetStream(), filtPtr, CL_TRUE, 0,
//                                    sizeof(float)*filterSize*filterSize,
//                                    filterBuffer, 0, NULL, NULL);


//     cl_kernel theKernel;
//     cl_program theProgram;


//     if (chnFormat == RPPI_CHN_PLANAR)
//     {
//         CreateProgramFromBinary(handle,"convolution.cl","convolution.cl.bin","naive_convolution_planar",theProgram,theKernel);
//         clRetainKernel(theKernel);

//         // cl_kernel_initializer(  handle.GetStream(), "convolution.cl",
//         //                         "naive_convolution_planar", theProgram, theKernel);

//     }
//     else if (chnFormat == RPPI_CHN_PACKED)
//     {
//         CreateProgramFromBinary(handle,"convolution.cl","convolution.cl.bin","naive_convolution_packed",theProgram,theKernel);
//         clRetainKernel(theKernel);

//         // cl_kernel_initializer(  handle.GetStream(), "convolution.cl",
//         //                         "naive_convolution_packed", theProgram, theKernel);
//     }
//     else
//     {std::cerr << "Internal error: Unknown Channel format";}




//     err  = clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &srcPtr);
//     err |= clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &dstPtr);
//     err |= clSetKernelArg(theKernel, counter++, sizeof(cl_mem), &filtPtr);
//     err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.height);
//     err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &srcSize.width);
//     err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &channel);
//     err |= clSetKernelArg(theKernel, counter++, sizeof(unsigned int), &filterSize);

// //----
//     size_t gDim3[3];
//     gDim3[0] = srcSize.width;
//     gDim3[1] = srcSize.height;
//     gDim3[2] = channel;
//     cl_kernel_implementer (handle, gDim3, NULL/*Local*/, theProgram, theKernel);

//     clReleaseMemObject(filtPtr);
//     return RPP_SUCCESS;
// }
