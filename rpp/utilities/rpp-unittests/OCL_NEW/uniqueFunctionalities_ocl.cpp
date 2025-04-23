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

#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "rppi.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <half/half.hpp>
#include <fstream>
#include <CL/cl.h>

using namespace cv;
using namespace std;
using half_float::half;

typedef half Rpp16f;

#define RPPPIXELCHECK(pixel) (pixel < (Rpp32f)0) ? ((Rpp32f)0) : ((pixel < (Rpp32f)255) ? pixel : ((Rpp32f)255))

template <typename T>
void displayTensor(T *pArr, Rpp32u size)
{
    int p = 0, count = 0;
    for (int i = 0; i < size; i++, count++)
    {
        printf("%d\t", (int) *(pArr + p));
        p++;
        if (count == 19)
        {
            printf("\n");
            count = 0;
        }
    }
    printf("\n");
}

template <typename T>
void displayTensorF(T *pArr, Rpp32u size)
{
    int p = 0, count = 0;
    for (int i = 0; i < size; i++, count++)
    {
        printf("%0.2f\t", (Rpp32f) *(pArr + p));
        p++;
        if (count == 19)
        {
            printf("\n");
            count = 0;
        }
    }
    printf("\n");
}

template <typename T>
void displayPlanar(T *pArr, RppiSize size, Rpp32u channel)
{
    int p = 0;
    for(int c = 0; c < channel; c++)
    {
        printf("\n\nChannel %d:\n", c+1);
        for (int i = 0; i < (size.height * size.width); i++)
        {
            printf("%d\t\t", (int) *(pArr + p));
            if (((i + 1) % size.width) == 0)
            {
                printf("\n");
            }
            p += 1;
        }
    }
}

template <typename T>
void displayPacked(T *pArr, RppiSize size, Rpp32u channel)
{
    int p = 0;
    for (int i = 0; i < size.height; i++)
    {
        for (int c = 0; c < channel; c++)
        {
            for (int j = 0; j < size.width; j++)
            {
                printf("%d\t\t", (int) *(pArr + p + c + (j * channel)));
            }
            printf("\n");
        }
        printf("\n");
        p += (channel * size.width);
    }
}

int main(int argc, char **argv)
{
    const int MIN_ARG_COUNT = 3;
    printf("\nUsage: ./uniqueFunctionalities_gpu <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <case number = 0:12>\n");
    if (argc < MIN_ARG_COUNT)
    {
        printf("\nImproper Usage! Needs all arguments!\n");
        return -1;
    }

    int ip_bitDepth = atoi(argv[1]);
    int test_case = atoi(argv[2]);

    printf("\nip_bitDepth = %d\ntest_case = %d", ip_bitDepth, test_case);

    cl_mem d_srcPtr1, d_srcPtr2, d_dstPtr;
    cl_mem d_srcPtr1_16f, d_srcPtr1_32f, d_srcPtr1_8s;
    cl_mem d_dstPtr_16f, d_dstPtr_32f, d_dstPtr_8s;
    cl_mem d_dstPtr_16u, d_dstPtr_16s;
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_context theContext;
    cl_command_queue theQueue;
    cl_int err;
    err = clGetPlatformIDs(1, &platform_id, NULL);
    err |= clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    theContext = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    theQueue = clCreateCommandQueueWithProperties(theContext, device_id, 0, &err);

    rppHandle_t handle;

    RppBackend backend = RppBackend::RPP_OCL_BACKEND;
    rppCreate(&handle, 1, 0, theQueue, backend);

    clock_t start, end;
    double gpu_time_used;

    int missingFuncFlag = 0;
    string test_case_name;

    switch (test_case)
    {
    case 0:
    {
        test_case_name = "tensor_transpose";

        Rpp32u totalNumberOfElements = 36;
        Rpp32u inTensorDim[4] = {3, 3, 2, 2};
        Rpp32u perm[4] = {0, 1, 3, 2};
        Rpp8u srcPtr[36] = {255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 130, 129, 128, 127, 126, 117, 113, 121, 127, 111, 100, 108, 65, 66, 67, 68, 69, 70, 71, 72, 13, 24, 15, 16};
        Rpp8u dstPtr[36] = {0};

        d_srcPtr1 = clCreateBuffer(theContext, CL_MEM_READ_ONLY, 36 * sizeof(Rpp8u), NULL, NULL);
        d_dstPtr = clCreateBuffer(theContext, CL_MEM_READ_ONLY, 36 * sizeof(Rpp8u), NULL, NULL);
        err |= clEnqueueWriteBuffer(theQueue, d_srcPtr1, CL_TRUE, 0, 36 * sizeof(Rpp8u), srcPtr, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(theQueue, d_dstPtr, CL_TRUE, 0, 36 * sizeof(Rpp8u), dstPtr, 0, NULL, NULL);

        start = clock();

        rppi_tensor_transpose_u8_gpu(d_srcPtr1, d_dstPtr, inTensorDim, perm, handle);

        end = clock();

        if (missingFuncFlag != 1)
        {
            clEnqueueReadBuffer(theQueue, d_dstPtr, CL_TRUE, 0, 36 * sizeof(Rpp8u), dstPtr, 0, NULL, NULL);

            printf("\n\nInput:\n");
            displayTensor(srcPtr, totalNumberOfElements);
            printf("\n\nOutput of tensor_transpose:\n");
            displayTensor(dstPtr, totalNumberOfElements);

            gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
            cout << "\nGPU Time - BatchPD : " << gpu_time_used;
            printf("\n");
        }

        break;
    }
    case 1:
    {
        test_case_name = "transpose";

        // Test Case 1
        // Rpp32u totalNumberOfElements = 24;
        // Rpp32u perm[4] = {0, 3, 1, 2};
        // Rpp32u shape[4] = {2, 2, 2, 3};
        // Rpp8u srcPtr[24] = {255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 130, 129, 128, 127, 126, 125, 124, 123, 122, 121, 120, 119};
        // Rpp8u dstPtr[24] = {0};
        // Rpp16f srcPtr16f[24], dstPtr16f[24];
        // Rpp32f srcPtr32f[24], dstPtr32f[24];
        // Rpp8s srcPtr8s[24], dstPtr8s[24];

        // Test Case 2
        Rpp32u totalNumberOfElements = 120;
        Rpp32u perm[4] = {0, 3, 1, 2};
        Rpp32u shape[4] = {2, 4, 5, 3};
        Rpp8u srcPtr[120] = {
            255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 130, 129, 128, 127, 126, 125, 124, 123, 122, 121, 120, 119, 5, 4, 3, 2, 1, 0,
            27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 115, 114, 113, 112, 111, 110,
            240, 239, 238, 237, 236, 235, 234, 233, 232, 231, 230, 229, 200, 199, 198, 197, 196, 195, 194, 193, 192, 191, 190, 189, 140, 139, 138, 137, 136, 135,
            70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 170, 169, 168, 167, 166, 165, 164, 163, 162, 161, 160, 159, 15, 14, 13, 12, 11, 10
        };
        Rpp8u dstPtr[120] = {0};
        Rpp16f srcPtr16f[120], dstPtr16f[120];
        Rpp32f srcPtr32f[120], dstPtr32f[120];
        Rpp8s srcPtr8s[120], dstPtr8s[120];

        for (int i = 0; i < totalNumberOfElements; i++)
        {
            srcPtr16f[i] = (Rpp16f) srcPtr[i];
            srcPtr32f[i] = (Rpp32f) srcPtr[i];
            srcPtr8s[i] = (Rpp8s) (((Rpp32s) srcPtr[i]) - 128);
        }

        start = clock();

        if (ip_bitDepth == 0)
        {
            d_srcPtr1 = clCreateBuffer(theContext, CL_MEM_READ_ONLY, 120 * sizeof(Rpp8u), NULL, NULL);
            d_dstPtr = clCreateBuffer(theContext, CL_MEM_READ_ONLY, 120 * sizeof(Rpp8u), NULL, NULL);
            err |= clEnqueueWriteBuffer(theQueue, d_srcPtr1, CL_TRUE, 0, 120 * sizeof(Rpp8u), srcPtr, 0, NULL, NULL);
            err |= clEnqueueWriteBuffer(theQueue, d_dstPtr, CL_TRUE, 0, 120 * sizeof(Rpp8u), dstPtr, 0, NULL, NULL);
            rppi_tensor_transpose_u8_gpu(d_srcPtr1, d_dstPtr, shape, perm, handle);
        }
        else if (ip_bitDepth == 1)
        {
            d_srcPtr1_16f = clCreateBuffer(theContext, CL_MEM_READ_ONLY, 120 * sizeof(Rpp16f), NULL, NULL);
            d_dstPtr_16f = clCreateBuffer(theContext, CL_MEM_READ_ONLY, 120 * sizeof(Rpp16f), NULL, NULL);
            err |= clEnqueueWriteBuffer(theQueue, d_srcPtr1_16f, CL_TRUE, 0, 120 * sizeof(Rpp16f), srcPtr16f, 0, NULL, NULL);
            err |= clEnqueueWriteBuffer(theQueue, d_dstPtr_16f, CL_TRUE, 0, 120 * sizeof(Rpp16f), dstPtr16f, 0, NULL, NULL);
            rppi_tensor_transpose_f16_gpu(d_srcPtr1_16f, d_dstPtr_16f, shape, perm, handle);
        }
        else if (ip_bitDepth == 2)
        {
            d_srcPtr1_32f = clCreateBuffer(theContext, CL_MEM_READ_ONLY, 120 * sizeof(Rpp32f), NULL, NULL);
            d_dstPtr_32f = clCreateBuffer(theContext, CL_MEM_READ_ONLY, 120 * sizeof(Rpp32f), NULL, NULL);
            err |= clEnqueueWriteBuffer(theQueue, d_srcPtr1_32f, CL_TRUE, 0, 120 * sizeof(Rpp32f), srcPtr32f, 0, NULL, NULL);
            err |= clEnqueueWriteBuffer(theQueue, d_dstPtr_32f, CL_TRUE, 0, 120 * sizeof(Rpp32f), dstPtr32f, 0, NULL, NULL);
            rppi_tensor_transpose_f32_gpu(d_srcPtr1_32f, d_dstPtr_32f, shape, perm, handle);
        }
        else if (ip_bitDepth == 3)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 4)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 5)
        {
            d_srcPtr1_8s = clCreateBuffer(theContext, CL_MEM_READ_ONLY, 120 * sizeof(Rpp8s), NULL, NULL);
            d_dstPtr_8s = clCreateBuffer(theContext, CL_MEM_READ_ONLY, 120 * sizeof(Rpp8s), NULL, NULL);
            err |= clEnqueueWriteBuffer(theQueue, d_srcPtr1_8s, CL_TRUE, 0, 120 * sizeof(Rpp8s), srcPtr8s, 0, NULL, NULL);
            err |= clEnqueueWriteBuffer(theQueue, d_dstPtr_8s, CL_TRUE, 0, 120 * sizeof(Rpp8s), dstPtr8s, 0, NULL, NULL);
            rppi_tensor_transpose_i8_gpu(d_srcPtr1_8s, d_dstPtr_8s, shape, perm, handle);
        }
        else if (ip_bitDepth == 6)
            missingFuncFlag = 1;
        else
            missingFuncFlag = 1;

        end = clock();

        if (ip_bitDepth == 0)
        {
            clEnqueueReadBuffer(theQueue, d_dstPtr, CL_TRUE, 0, 120 * sizeof(Rpp8u), dstPtr, 0, NULL, NULL);
            printf("\n\nInput:\n");
            displayTensor(srcPtr, totalNumberOfElements);
            printf("\n\nOutput of transpose_u8:\n");
            displayTensor(dstPtr, totalNumberOfElements);
        }
        else if (ip_bitDepth == 1)
        {
            clEnqueueReadBuffer(theQueue, d_dstPtr_16f, CL_TRUE, 0, 120 * sizeof(Rpp16f), dstPtr16f, 0, NULL, NULL);
            printf("\n\nInput:\n");
            displayTensorF(srcPtr16f, totalNumberOfElements);
            printf("\n\nOutput of transpose_f16:\n");
            displayTensorF(dstPtr16f, totalNumberOfElements);
        }
        else if (ip_bitDepth == 2)
        {
            clEnqueueReadBuffer(theQueue, d_dstPtr_32f, CL_TRUE, 0, 120 * sizeof(Rpp32f), dstPtr32f, 0, NULL, NULL);
            printf("\n\nInput:\n");
            displayTensorF(srcPtr32f, totalNumberOfElements);
            printf("\n\nOutput of transpose_f32:\n");
            displayTensorF(dstPtr32f, totalNumberOfElements);
        }
        else if (ip_bitDepth == 3)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 4)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 5)
        {
            clEnqueueReadBuffer(theQueue, d_dstPtr_8s, CL_TRUE, 0, 120 * sizeof(Rpp8s), dstPtr8s, 0, NULL, NULL);
            printf("\n\nInput:\n");
            displayTensor(srcPtr8s, totalNumberOfElements);
            printf("\n\nOutput of transpose_i8:\n");
            displayTensor(dstPtr8s, totalNumberOfElements);
        }
        else if (ip_bitDepth == 6)
            missingFuncFlag = 1;
        else
            missingFuncFlag = 1;

            gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
            cout << "\nGPU Time - BatchPD : " << gpu_time_used;
            printf("\n");

        break;
    }
    case 2:
    {
        test_case_name = "tensor_add";

        Rpp8u srcPtr1[36] = {255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 130, 129, 128, 127, 126, 117, 113, 121, 127, 111, 100, 108, 65, 66, 67, 68, 69, 70, 71, 72, 13, 24, 15, 16};
        Rpp8u srcPtr2[36] = {16, 15, 24, 13, 72, 71, 70, 69, 68, 67, 66, 65, 108, 100, 111, 127, 121, 113, 117, 126, 127, 128, 129, 130, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255};

        Rpp8u dstPtr[36] = {0};

        Rpp32u tensorDimension = 4;
        Rpp32u tensorDimensionValues[4] = {3,2,2,3};

        d_srcPtr1 = clCreateBuffer(theContext, CL_MEM_READ_ONLY, 36 * sizeof(Rpp8u), NULL, NULL);
        d_srcPtr2 = clCreateBuffer(theContext, CL_MEM_READ_ONLY, 36 * sizeof(Rpp8u), NULL, NULL);
        d_dstPtr = clCreateBuffer(theContext, CL_MEM_READ_ONLY, 36 * sizeof(Rpp8u), NULL, NULL);
        err |= clEnqueueWriteBuffer(theQueue, d_srcPtr1, CL_TRUE, 0, 36 * sizeof(Rpp8u), srcPtr1, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(theQueue, d_srcPtr2, CL_TRUE, 0, 36 * sizeof(Rpp8u), srcPtr2, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(theQueue, d_dstPtr, CL_TRUE, 0, 36 * sizeof(Rpp8u), dstPtr, 0, NULL, NULL);

        start = clock();

        if (ip_bitDepth == 0)
            rppi_tensor_add_u8_gpu(d_srcPtr1, d_srcPtr2, d_dstPtr, tensorDimension, tensorDimensionValues, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 3)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 4)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 5)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 6)
            missingFuncFlag = 1;
        else
            missingFuncFlag = 1;

        end = clock();

        if (missingFuncFlag != 1)
        {
            clEnqueueReadBuffer(theQueue, d_dstPtr, CL_TRUE, 0, 36 * sizeof(Rpp8u), dstPtr, 0, NULL, NULL);

            printf("\n\nInput 1:\n");
            displayTensor(srcPtr1, 36);
            printf("\n\nInput 2:\n");
            displayTensor(srcPtr2, 36);
            printf("\n\nTensor Shape:\n");
            printf("[%d x %d x %d x %d]", tensorDimensionValues[0], tensorDimensionValues[1], tensorDimensionValues[2], tensorDimensionValues[3]);
            printf("\n\nOutput of tensor_add:\n");
            displayTensor(dstPtr, 36);

            gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
            cout << "\nGPU Time - BatchPD : " << gpu_time_used;
            printf("\n");
        }

        break;
    }
    case 3:
    {
        test_case_name = "tensor_subtract";

        Rpp8u srcPtr1[36] = {255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 130, 129, 128, 127, 126, 117, 113, 121, 127, 111, 100, 108, 65, 66, 67, 68, 69, 70, 71, 72, 13, 24, 15, 16};
        Rpp8u srcPtr2[36] = {16, 15, 24, 13, 72, 71, 70, 69, 68, 67, 66, 65, 108, 100, 111, 127, 121, 113, 117, 126, 127, 128, 129, 130, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255};

        Rpp8u dstPtr[36] = {0};

        Rpp32u tensorDimension = 4;
        Rpp32u tensorDimensionValues[4] = {3,2,2,3};

        d_srcPtr1 = clCreateBuffer(theContext, CL_MEM_READ_ONLY, 36 * sizeof(Rpp8u), NULL, NULL);
        d_srcPtr2 = clCreateBuffer(theContext, CL_MEM_READ_ONLY, 36 * sizeof(Rpp8u), NULL, NULL);
        d_dstPtr = clCreateBuffer(theContext, CL_MEM_READ_ONLY, 36 * sizeof(Rpp8u), NULL, NULL);
        err |= clEnqueueWriteBuffer(theQueue, d_srcPtr1, CL_TRUE, 0, 36 * sizeof(Rpp8u), srcPtr1, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(theQueue, d_srcPtr2, CL_TRUE, 0, 36 * sizeof(Rpp8u), srcPtr2, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(theQueue, d_dstPtr, CL_TRUE, 0, 36 * sizeof(Rpp8u), dstPtr, 0, NULL, NULL);

        start = clock();

        if (ip_bitDepth == 0)
            rppi_tensor_subtract_u8_gpu(d_srcPtr1, d_srcPtr2, d_dstPtr, tensorDimension, tensorDimensionValues, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 3)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 4)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 5)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 6)
            missingFuncFlag = 1;
        else
            missingFuncFlag = 1;

        end = clock();

        if (missingFuncFlag != 1)
        {
            clEnqueueReadBuffer(theQueue, d_dstPtr, CL_TRUE, 0, 36 * sizeof(Rpp8u), dstPtr, 0, NULL, NULL);

            printf("\n\nInput 1:\n");
            displayTensor(srcPtr1, 36);
            printf("\n\nInput 2:\n");
            displayTensor(srcPtr2, 36);
            printf("\n\nTensor Shape:\n");
            printf("[%d x %d x %d x %d]", tensorDimensionValues[0], tensorDimensionValues[1], tensorDimensionValues[2], tensorDimensionValues[3]);
            printf("\n\nOutput of tensor_subtract:\n");
            displayTensor(dstPtr, 36);

            gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
            cout << "\nGPU Time - BatchPD : " << gpu_time_used;
            printf("\n");
        }

        break;
    }
    case 4:
    {
        test_case_name = "tensor_multiply";

        Rpp8u srcPtr1[36] = {255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 130, 129, 128, 127, 126, 117, 113, 121, 127, 111, 100, 108, 65, 66, 67, 68, 69, 70, 71, 72, 13, 24, 15, 16};
        Rpp8u srcPtr2[36] = {16, 15, 24, 13, 72, 71, 70, 69, 68, 67, 66, 65, 108, 100, 111, 127, 121, 113, 117, 126, 127, 128, 129, 130, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255};

        Rpp8u dstPtr[36] = {0};

        Rpp32u tensorDimension = 4;
        Rpp32u tensorDimensionValues[4] = {3,2,2,3};

        d_srcPtr1 = clCreateBuffer(theContext, CL_MEM_READ_ONLY, 36 * sizeof(Rpp8u), NULL, NULL);
        d_srcPtr2 = clCreateBuffer(theContext, CL_MEM_READ_ONLY, 36 * sizeof(Rpp8u), NULL, NULL);
        d_dstPtr = clCreateBuffer(theContext, CL_MEM_READ_ONLY, 36 * sizeof(Rpp8u), NULL, NULL);
        err |= clEnqueueWriteBuffer(theQueue, d_srcPtr1, CL_TRUE, 0, 36 * sizeof(Rpp8u), srcPtr1, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(theQueue, d_srcPtr2, CL_TRUE, 0, 36 * sizeof(Rpp8u), srcPtr2, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(theQueue, d_dstPtr, CL_TRUE, 0, 36 * sizeof(Rpp8u), dstPtr, 0, NULL, NULL);

        start = clock();

        if (ip_bitDepth == 0)
            rppi_tensor_multiply_u8_gpu(d_srcPtr1, d_srcPtr2, d_dstPtr, tensorDimension, tensorDimensionValues, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 3)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 4)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 5)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 6)
            missingFuncFlag = 1;
        else
            missingFuncFlag = 1;

        end = clock();

        if (missingFuncFlag != 1)
        {
            clEnqueueReadBuffer(theQueue, d_dstPtr, CL_TRUE, 0, 36 * sizeof(Rpp8u), dstPtr, 0, NULL, NULL);

            printf("\n\nInput 1:\n");
            displayTensor(srcPtr1, 36);
            printf("\n\nInput 2:\n");
            displayTensor(srcPtr2, 36);
            printf("\n\nTensor Shape:\n");
            printf("[%d x %d x %d x %d]", tensorDimensionValues[0], tensorDimensionValues[1], tensorDimensionValues[2], tensorDimensionValues[3]);
            printf("\n\nOutput of tensor_multiply:\n");
            displayTensor(dstPtr, 36);

            gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
            cout << "\nGPU Time - BatchPD : " << gpu_time_used;
            printf("\n");
        }

        break;
    }
    case 5:
    {
        test_case_name = "tensor_matrix_multiply";

        Rpp32u tensorDimensionValues1[2] = {3, 2};
        Rpp32u tensorDimensionValues2[2] = {2, 4};

        Rpp8u srcPtr1[6] = {1, 2, 3, 4, 5, 6};
        Rpp8u srcPtr2[8] = {7, 8, 9, 10, 11, 12, 13, 14};
        Rpp8u dstPtr[12] = {0};

        d_srcPtr1 = clCreateBuffer(theContext, CL_MEM_READ_ONLY, 6 * sizeof(Rpp8u), NULL, NULL);
        d_srcPtr2 = clCreateBuffer(theContext, CL_MEM_READ_ONLY, 8 * sizeof(Rpp8u), NULL, NULL);
        d_dstPtr = clCreateBuffer(theContext, CL_MEM_READ_ONLY, 12 * sizeof(Rpp8u), NULL, NULL);
        err |= clEnqueueWriteBuffer(theQueue, d_srcPtr1, CL_TRUE, 0, 6 * sizeof(Rpp8u), srcPtr1, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(theQueue, d_srcPtr2, CL_TRUE, 0, 8 * sizeof(Rpp8u), srcPtr2, 0, NULL, NULL);
        err |= clEnqueueWriteBuffer(theQueue, d_dstPtr, CL_TRUE, 0, 12 * sizeof(Rpp8u), dstPtr, 0, NULL, NULL);

        start = clock();

        if (ip_bitDepth == 0)
            rppi_tensor_matrix_multiply_u8_gpu(d_srcPtr1, d_srcPtr2, d_dstPtr, tensorDimensionValues1, tensorDimensionValues2, handle);
        else if (ip_bitDepth == 1)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 2)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 3)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 4)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 5)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 6)
            missingFuncFlag = 1;
        else
            missingFuncFlag = 1;

        end = clock();

        if (missingFuncFlag != 1)
        {
            clEnqueueReadBuffer(theQueue, d_dstPtr, CL_TRUE, 0, 12 * sizeof(Rpp8u), dstPtr, 0, NULL, NULL);

            printf("\n\nInput 1:\n");
            displayTensor(srcPtr1, 6);
            printf("\n\nInput 1 Tensor Shape:\n");
            printf("[%d x %d]", tensorDimensionValues1[0], tensorDimensionValues1[1]);
            printf("\n\nInput 2:\n");
            displayTensor(srcPtr2, 8);
            printf("\n\nInput 2 Tensor Shape:\n");
            printf("[%d x %d]", tensorDimensionValues2[0], tensorDimensionValues2[1]);
            printf("\n\nOutput of tensor_matrix_multiply:\n");
            displayTensor(dstPtr, 12);

            gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
            cout << "\nGPU Time - BatchPD : " << gpu_time_used;
            printf("\n");
        }

        break;
    }
    // case 6:
    // {
    //     test_case_name = "min_max_loc";

    //     Rpp8u srcPtr[36] = {255, 130, 65, 254, 129, 66, 253, 128, 67, 252, 127, 68, 251, 126, 69, 250, 117, 70, 249, 113, 71, 248, 121, 72, 247, 127, 13, 246, 111, 24, 245, 100, 15, 244, 108, 16};

    //     d_srcPtr1 = clCreateBuffer(theContext, CL_MEM_READ_ONLY, 36 * sizeof(Rpp8u), NULL, NULL);
    //     err |= clEnqueueWriteBuffer(theQueue, d_srcPtr1, CL_TRUE, 0, 36 * sizeof(Rpp8u), srcPtr, 0, NULL, NULL);

    //     RppiSize srcSize1Channel, srcSize3Channel;
    //     srcSize1Channel.height = 6;
    //     srcSize1Channel.width  = 6;
    //     srcSize3Channel.height = 3;
    //     srcSize3Channel.width = 4;

    //     Rpp8u min, max;
    //     Rpp32u minLoc, maxLoc;

    //     for (int i = 0; i < 3; i++)
    //     {
    //         start = clock();

    //         if (ip_bitDepth == 0)
    //         {
    //             if (i == 0)
    //                 rppi_min_max_loc_u8_pln1_gpu(d_srcPtr1, srcSize1Channel, &min, &max, &minLoc, &maxLoc, handle);
    //             else if (i == 1)
    //                 rppi_min_max_loc_u8_pln3_gpu(d_srcPtr1, srcSize3Channel, &min, &max, &minLoc, &maxLoc, handle);
    //             else if  (i == 2)
    //                 rppi_min_max_loc_u8_pkd3_gpu(d_srcPtr1, srcSize3Channel, &min, &max, &minLoc, &maxLoc, handle);
    //         }
    //         else
    //             missingFuncFlag = 1;

    //         end = clock();

    //         if (missingFuncFlag != 1)
    //         {
    //             printf("\n\nInput:\n");
    //             displayTensor(srcPtr, 36);
    //             printf("\n\nInput Shape:\n");
    //             if (i == 0)
    //                 printf("[%d x %d]", srcSize1Channel.height, srcSize1Channel.width);
    //             else if (i == 1)
    //                 printf("[%d x %d x %d]", 3, srcSize3Channel.height, srcSize3Channel.width);
    //             else if  (i == 2)
    //                 printf("[%d x %d x %d]", srcSize3Channel.height, srcSize3Channel.width, 3);
    //             printf("\n\nOutput of min_max_loc operation:\n");
    //             printf("\nMin = %d, Max = %d", min, max);
    //             printf("\nMinLoc = %d, MaxLoc = %d\n", minLoc, maxLoc);

    //             gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    //             cout << "\nGPU Time - BatchPD : " << gpu_time_used;
    //             printf("\n");
    //         }
    //     }

    //     break;
    // }
    // case 7:
    // {
    //     test_case_name = "mean_stddev";

    //     Rpp8u srcPtr[36] = {255, 130, 65, 254, 129, 66, 253, 128, 67, 252, 127, 68, 251, 126, 69, 250, 117, 70, 249, 113, 71, 248, 121, 72, 247, 127, 13, 246, 111, 24, 245, 100, 15, 244, 108, 16};

    //     d_srcPtr1 = clCreateBuffer(theContext, CL_MEM_READ_ONLY, 36 * sizeof(Rpp8u), NULL, NULL);
    //     err |= clEnqueueWriteBuffer(theQueue, d_srcPtr1, CL_TRUE, 0, 36 * sizeof(Rpp8u), srcPtr, 0, NULL, NULL);

    //     RppiSize srcSize1Channel, srcSize3Channel;
    //     srcSize1Channel.height = 6;
    //     srcSize1Channel.width  = 6;
    //     srcSize3Channel.height = 3;
    //     srcSize3Channel.width = 4;

    //     Rpp8u min, max;
    //     Rpp32f mean, stddev;

    //     for (int i = 0; i < 3; i++)
    //     {
    //         start = clock();

    //         if (ip_bitDepth == 0)
    //         {
    //             if (i == 0)
    //                 rppi_mean_stddev_u8_pkd3_gpu(d_srcPtr1, srcSize3Channel, &mean, &stddev, handle);
    //             else if (i == 1)
    //                 rppi_mean_stddev_u8_pln3_gpu(d_srcPtr1, srcSize3Channel, &mean, &stddev, handle);
    //             else if  (i == 2)
    //                 rppi_mean_stddev_u8_pln1_gpu(d_srcPtr1, srcSize3Channel, &mean, &stddev, handle);
    //         }
    //         else
    //             missingFuncFlag = 1;

    //         end = clock();

    //         if (missingFuncFlag != 1)
    //         {
    //             printf("\n\nInput:\n");
    //             displayTensor(srcPtr, 36);
    //             printf("\n\nInput Shape:\n");
    //             if (i == 0)
    //                 printf("[%d x %d]", srcSize1Channel.height, srcSize1Channel.width);
    //             else if (i == 1)
    //                 printf("[%d x %d x %d]", 3, srcSize3Channel.height, srcSize3Channel.width);
    //             else if  (i == 2)
    //                 printf("[%d x %d x %d]", srcSize3Channel.height, srcSize3Channel.width, 3);
    //             printf("\n\nOutput of mean_stddev operation:\n");
    //             printf("\nMean = %0.4f, StdDev = %0.4f", mean, stddev);

    //             gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    //             cout << "\nGPU Time - BatchPD : " << gpu_time_used;
    //             printf("\n");
    //         }
    //     }

    //     break;
    // }
    case 8:
    {
        test_case_name = "control_flow";

        rppHandle_t handle;

        bool b1 = true, b2 = false;
        bool b3 =  true;
        Rpp8u u1 = 120, u2 = 100;
        Rpp8u u3 = 20;

        start = clock();

        rpp_bool_control_flow(b1, b2, &b3, RPP_SCALAR_OP_AND, handle );
        rpp_u8_control_flow(u1, u2, &u3, RPP_SCALAR_OP_ADD, handle );

        end = clock();

        if(u3 == 220)
            cout << "---PASS---" << endl;
        else
            cout << "---FAIL---" << endl;
        if(b3 == false)
            cout << "---PASS--" << endl;
        else
            cout << "---FAIL---" << endl;

        gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        cout << "\nGPU Time - BatchPD : " << gpu_time_used;
        printf("\n");

        break;
    }
    // case 9:
    // {
    //     test_case_name = "histogram";

    //     rppHandle_t handle;
    //     int count = 0;

    //     Rpp8u srcPtr[36] = {255, 254, 253, 252, 251, 250, 249, 248, 247, 246, 245, 244, 130, 129, 128, 127, 126, 117, 113, 121, 127, 111, 100, 108, 65, 66, 67, 68, 69, 70, 71, 72, 13, 24, 15, 16};
    //     Rpp32u bins = 8;
    //     RppiSize srcSize;
    //     Rpp32u *outputHistogram = (Rpp32u *) calloc (bins, sizeof(Rpp32u));
    //     Rpp32u *outputHistogramTemp;

    //     memset(outputHistogram, 0, bins * sizeof(Rpp32u));
    //     srcSize.height = 6;
    //     srcSize.width = 6;
    //     start = clock();

    //     rppi_histogram_u8_pln1_gpu(srcPtr, srcSize, outputHistogram, bins, handle);

    //     end = clock();
    //     printf("\n\nOutput of histogram_u8_pln1 for %d bins:\n", bins);
    //     outputHistogramTemp = outputHistogram;
    //     count = 1;
    //     for (int i = 0; i < bins; i++, count++)
    //     {
    //         printf("%d\t", *outputHistogramTemp);
    //         outputHistogramTemp++;
    //         if (count == 25)
    //         {
    //             printf("\n");
    //             count = 0;
    //         }
    //     }
    //     gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    //     cout << "\nGPU Time - BatchPD : " << gpu_time_used;
    //     printf("\n");

    //     memset(outputHistogram, 0, bins * sizeof(Rpp32u));
    //     srcSize.height = 3;
    //     srcSize.width = 4;
    //     start = clock();

    //     rppi_histogram_u8_pln3_gpu(srcPtr, srcSize, outputHistogram, bins, handle);

    //     end = clock();
    //     printf("\n\nOutput of histogram_u8_pln3 for %d bins:\n", bins);
    //     outputHistogramTemp = outputHistogram;
    //     count = 1;
    //     for (int i = 0; i < bins; i++, count++)
    //     {
    //         printf("%d\t", *outputHistogramTemp);
    //         outputHistogramTemp++;
    //         if (count == 25)
    //         {
    //             printf("\n");
    //             count = 0;
    //         }
    //     }
    //     gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    //     cout << "\nGPU Time - BatchPD : " << gpu_time_used;
    //     printf("\n");

    //     memset(outputHistogram, 0, bins * sizeof(Rpp32u));
    //     srcSize.height = 3;
    //     srcSize.width = 4;
    //     start = clock();

    //     rppi_histogram_u8_pkd3_gpu(srcPtr, srcSize, outputHistogram, bins, handle);

    //     end = clock();
    //     printf("\n\nOutput of histogram_u8_pkd3 for %d bins:\n", bins);
    //     outputHistogramTemp = outputHistogram;
    //     count = 1;
    //     for (int i = 0; i < bins; i++, count++)
    //     {
    //         printf("%d\t", *outputHistogramTemp);
    //         outputHistogramTemp++;
    //         if (count == 25)
    //         {
    //             printf("\n");
    //             count = 0;
    //         }
    //     }
    //     gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    //     cout << "\nGPU Time - BatchPD : " << gpu_time_used;
    //     printf("\n");

    //     break;
    // }
    // case 10:
    // {
    //     test_case_name = "convert_bit_depth";

    //     rppHandle_t handle;

    //     Rpp8u srcPtr[36] = {255, 130, 65, 254, 129, 66, 253, 128, 67, 252, 127, 68, 251, 126, 69, 250, 117, 70, 249, 113, 71, 248, 121, 72, 247, 127, 13, 246, 111, 24, 245, 100, 15, 244, 108, 16};
    //     Rpp8s dstPtr8s[36];
    //     Rpp16u dstPtr16u[36];
    //     Rpp16s dstPtr16s[36];

    //     d_srcPtr1 = clCreateBuffer(theContext, CL_MEM_READ_ONLY, 36 * sizeof(Rpp8u), NULL, NULL);
    //     d_dstPtr_8s = clCreateBuffer(theContext, CL_MEM_READ_ONLY, 36 * sizeof(Rpp8s), NULL, NULL);
    //     d_dstPtr_16u = clCreateBuffer(theContext, CL_MEM_READ_ONLY, 36 * sizeof(Rpp16u), NULL, NULL);
    //     d_dstPtr_16s = clCreateBuffer(theContext, CL_MEM_READ_ONLY, 36 * sizeof(Rpp16s), NULL, NULL);
    //     err |= clEnqueueWriteBuffer(theQueue, d_srcPtr1, CL_TRUE, 0, 36 * sizeof(Rpp8u), srcPtr, 0, NULL, NULL);

    //     RppiSize srcSize1Channel[1], srcSize3Channel[1];
    //     srcSize1Channel[0].height = 6;
    //     srcSize1Channel[0].width  = 6;
    //     srcSize3Channel[0].height = 3;
    //     srcSize3Channel[0].width = 4;

    //     RppiSize srcSizeMax1Channel, srcSizeMax3Channel;
    //     srcSizeMax1Channel.height = 6;
    //     srcSizeMax1Channel.width  = 6;
    //     srcSizeMax3Channel.height = 3;
    //     srcSizeMax3Channel.width = 4;

    //     for (int i = 0; i < 9; i++)
    //     {
    //         start = clock();

    //         if (ip_bitDepth == 0)
    //         {
    //             if (i == 0)
    //                 rppi_convert_bit_depth_u8s8_pln1_batchPD_gpu(d_srcPtr1, srcSize1Channel, srcSizeMax1Channel, d_dstPtr_8s, 1, handle);
    //             else if (i == 1)
    //                 rppi_convert_bit_depth_u8u16_pln1_batchPD_gpu(d_srcPtr1, srcSize1Channel, srcSizeMax1Channel, d_dstPtr_16u, 1, handle);
    //             else if  (i == 2)
    //                 rppi_convert_bit_depth_u8s16_pln1_batchPD_gpu(d_srcPtr1, srcSize1Channel, srcSizeMax1Channel, d_dstPtr_16s, 1, handle);
    //             else if  (i == 3)
    //                 rppi_convert_bit_depth_u8s8_pln3_batchPD_gpu(d_srcPtr1, srcSize3Channel, srcSizeMax3Channel, d_dstPtr_8s, 1, handle);
    //             else if  (i == 4)
    //                 rppi_convert_bit_depth_u8u16_pln3_batchPD_gpu(d_srcPtr1, srcSize3Channel, srcSizeMax3Channel, d_dstPtr_16u, 1, handle);
    //             else if  (i == 5)
    //                 rppi_convert_bit_depth_u8s16_pln3_batchPD_gpu(d_srcPtr1, srcSize3Channel, srcSizeMax3Channel, d_dstPtr_16s, 1, handle);
    //             else if  (i == 6)
    //                 rppi_convert_bit_depth_u8s8_pkd3_batchPD_gpu(d_srcPtr1, srcSize3Channel, srcSizeMax3Channel, d_dstPtr_8s, 1, handle);
    //             else if  (i == 7)
    //                 rppi_convert_bit_depth_u8u16_pkd3_batchPD_gpu(d_srcPtr1, srcSize3Channel, srcSizeMax3Channel, d_dstPtr_16u, 1, handle);
    //             else if  (i == 8)
    //                 rppi_convert_bit_depth_u8s16_pkd3_batchPD_gpu(d_srcPtr1, srcSize3Channel, srcSizeMax3Channel, d_dstPtr_16s, 1, handle);
    //         }
    //         else
    //             missingFuncFlag = 1;

    //         end = clock();

    //         if (missingFuncFlag != 1)
    //         {
    //             clEnqueueReadBuffer(theQueue, d_dstPtr_8s, CL_TRUE, 0, 36 * sizeof(Rpp8u), dstPtr8s, 0, NULL, NULL);
    //             clEnqueueReadBuffer(theQueue, d_dstPtr_16u, CL_TRUE, 0, 36 * sizeof(Rpp16u), dstPtr16u, 0, NULL, NULL);
    //             clEnqueueReadBuffer(theQueue, d_dstPtr_16s, CL_TRUE, 0, 36 * sizeof(Rpp16s), dstPtr16s, 0, NULL, NULL);
    //             if ((i == 0) || (i == 1) || (i == 2))
    //             {
    //                 printf("\n\nInput:\n");
    //                 displayPlanar(srcPtr, srcSize1Channel[0], 1);
    //                 printf("\n\nInput Shape:\n");
    //                 printf("[%d x %d]", srcSize1Channel[0].height, srcSize1Channel[0].width);
    //                 printf("\n\nOutput of convert_bit_depth operation:\n");
    //                 if (i == 0)
    //                     displayPlanar(dstPtr8s, srcSize1Channel[0], 1);
    //                 else if (i == 1)
    //                     displayPlanar(dstPtr16u, srcSize1Channel[0], 1);
    //                 else if (i == 2)
    //                     displayPlanar(dstPtr16s, srcSize1Channel[0], 1);
    //             }
    //             else if ((i == 3) || (i == 4) || (i == 5))
    //             {
    //                 printf("\n\nInput:\n");
    //                 displayPlanar(srcPtr, srcSize3Channel[0], 3);
    //                 printf("\n\nInput Shape:\n");
    //                 printf("[%d x %d x %d]", 3, srcSize3Channel[0].height, srcSize3Channel[0].width);
    //                 printf("\n\nOutput of convert_bit_depth operation:\n");
    //                 if (i == 3)
    //                     displayPlanar(dstPtr8s, srcSize1Channel[0], 3);
    //                 else if (i == 4)
    //                     displayPlanar(dstPtr16u, srcSize1Channel[0], 3);
    //                 else if (i == 5)
    //                     displayPlanar(dstPtr16s, srcSize1Channel[0], 3);
    //             }
    //             else if ((i == 6) || (i == 7) || (i == 8))
    //             {
    //                 printf("\n\nInput:\n");
    //                 displayPacked(srcPtr, srcSize3Channel[0], 3);
    //                 printf("\n\nInput Shape:\n");
    //                 printf("[%d x %d x %d]", srcSize3Channel[0].height, srcSize3Channel[0].width, 3);
    //                 printf("\n\nOutput of convert_bit_depth operation:\n");
    //                 if (i == 6)
    //                     displayPacked(dstPtr8s, srcSize1Channel[0], 3);
    //                 else if (i == 7)
    //                     displayPacked(dstPtr16u, srcSize1Channel[0], 3);
    //                 else if (i == 8)
    //                     displayPacked(dstPtr16s, srcSize1Channel[0], 3);
    //             }

    //                 gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    //                 cout << "\nGPU Time - BatchPD : " << gpu_time_used;
    //                 printf("\n");
    //         }
    //     }

    //     break;
    // }
    // case 11:
    // {
    //     test_case_name = "tensor_convert_bit_depth";

    //     rppHandle_t handle;

    //     Rpp8u srcPtr[36] = {255, 130, 65, 254, 129, 66, 253, 128, 67, 252, 127, 68, 251, 126, 69, 250, 117, 70, 249, 113, 71, 248, 121, 72, 247, 127, 13, 246, 111, 24, 245, 100, 15, 244, 108, 16};
    //     Rpp8s dstPtr8s[36];
    //     Rpp16u dstPtr16u[36];
    //     Rpp16s dstPtr16s[36];

    //     Rpp32u tensorDimension = 3;
    //     Rpp32u tensorDimensionValues[3] = {3, 4, 3};

    //     for (int i = 0; i < 3; i++)
    //     {
    //         start = clock();

    //         if (ip_bitDepth == 0)
    //         {
    //             if (i == 0)
    //                 rppi_tensor_convert_bit_depth_u8s8_gpu(srcPtr, dstPtr8s, tensorDimension, tensorDimensionValues);
    //             else if (i == 1)
    //                 rppi_tensor_convert_bit_depth_u8u16_gpu(srcPtr, dstPtr16u, tensorDimension, tensorDimensionValues);
    //             else if  (i == 2)
    //                 rppi_tensor_convert_bit_depth_u8s16_gpu(srcPtr, dstPtr16s, tensorDimension, tensorDimensionValues);
    //         }
    //         else
    //             missingFuncFlag = 1;

    //         end = clock();

    //         if (missingFuncFlag != 1)
    //         {
    //             printf("\n\nInput:\n");
    //             displayTensor(srcPtr, 36);
    //             printf("\n\nInput Shape:\n");
    //             printf("[%d x %d x %d]", tensorDimensionValues[0], tensorDimensionValues[1], tensorDimensionValues[2]);
    //             printf("\n\nOutput of tensor_convert_bit_depth operation:\n");
    //             if (i == 0)
    //                 displayTensor(dstPtr8s, 36);
    //             else if (i == 1)
    //                 displayTensor(dstPtr16u, 36);
    //             else if (i == 2)
    //                 displayTensor(dstPtr16s, 36);

                    // gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
                    // cout << "\nGPU Time - BatchPD : " << gpu_time_used;
                    // printf("\n");
    //         }
    //     }

    //     break;
    // }
    // case 12:
    // {
    //     test_case_name = "tensor_look_up_table";

    //     Rpp8u srcPtr[36] = {255, 130, 65, 254, 129, 66, 253, 128, 67, 252, 127, 68, 251, 126, 69, 250, 117, 70, 249, 113, 71, 248, 121, 72, 247, 127, 13, 246, 111, 24, 245, 100, 15, 244, 108, 16};
    //     Rpp8u dstPtr[36];

    //     Rpp32u tensorDimension = 3;
    //     Rpp32u tensorDimensionValues[3] = {3, 4, 3};
    //     Rpp8u lutPtr[256];

    //     for (int i = 0; i < 256; i++)
    //     {
    //         lutPtr[i] = (Rpp8u)(255 - i);
    //     }

    //     start = clock();

    //     if (ip_bitDepth == 0)
    //         rppi_tensor_look_up_table_u8_gpu(srcPtr, dstPtr, lutPtr, tensorDimension, tensorDimensionValues);
    //     else
    //         missingFuncFlag = 1;

    //     end = clock();

    //     if (missingFuncFlag != 1)
    //     {
    //         printf("\n\nInput:\n");
    //         displayTensor(srcPtr, 36);
    //         printf("\n\nInput Shape:\n");
    //         printf("[%d x %d x %d]", tensorDimensionValues[0], tensorDimensionValues[1], tensorDimensionValues[2]);
    //         printf("\n\nOutput of tensor_look_up_table operation:\n");
    //         displayTensor(dstPtr, 36);

                // gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
                // cout << "\nGPU Time - BatchPD : " << gpu_time_used;
                // printf("\n");
    //     }

    //     break;
    // }
    default:
        missingFuncFlag = 1;
        break;
    }

    rppDestroy(handle, backend);

    if (missingFuncFlag == 1)
    {
        cout << "\nThis functionality sub-type of " << test_case_name << " doesn't yet exist in RPP\n";
        return -1;
    }

    return 0;
}
