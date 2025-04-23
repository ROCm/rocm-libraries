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

using namespace cv;
using namespace std;
using half_float::half;

typedef half Rpp16f;

#define RPPPIXELCHECK(pixel) (pixel < (Rpp32f)0) ? ((Rpp32f)0) : ((pixel < (Rpp32f)255) ? pixel : ((Rpp32f)255))

std::map<int, std::vector<string>> augmentationBitDepthMap =
{
    {0, {"0"}},
    {1, {"0", "1", "2", "5"}},
    {2, {"0"}},
    {3, {"0"}},
    {4, {"0"}},
    {5, {"0"}},
    {8, {"0"}},
};

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
    if (argc < MIN_ARG_COUNT)
    {
        printf("\nImproper Usage! Needs all arguments!\n");
        printf("Usage\n");
        for (auto elem : augmentationBitDepthMap)
        {
            for(unsigned j = 0; j < elem.second.size(); j++)
                printf("./uniqueFunctionalities_hip %s %d\n", elem.second[j].c_str(), elem.first);
        }
        return -1;
    }
    if(test_case < 0 || test_case == 6 || test_case == 7 || test_case > 8)
    {
        printf("\nInvalid test_case! test_case value should be 0 / 1 / 2 / 3 / 4 / 5 / 8\n");
        return -1;
    }
    std::vector<string> supportedBitDepths = augmentationBitDepthMap[test_case];
    if (supportedBitDepths.size() == 1)
        printf("\ntest_case %d supports only %s ip_bitDepth", test_case, supportedBitDepths[0].c_str());
    else
    {
        std::string bitDepthMessage = "";
        for(int i = 0; i < supportedBitDepths.size(); i++)
            bitDepthMessage = bitDepthMessage + supportedBitDepths[i] + " ";
        printf("\ntest_case %d supports %sip_bitDepths\n", test_case, bitDepthMessage.c_str());
    }

    int ip_bitDepth = atoi(argv[1]);
    int test_case = atoi(argv[2]);

    printf("\nip_bitDepth = %d\ntest_case = %d", ip_bitDepth, test_case);

    rppHandle_t handle;
	hipStream_t stream;
	hipStreamCreate(&stream);
    RppBackend backend = RppBackend::RPP_HIP_BACKEND;
	rppCreate(&handle, 2, 0, stream, backend);

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

        Rpp8u *d_srcPtr1, *d_dstPtr;
        hipMalloc(&d_srcPtr1, 36 * sizeof(Rpp8u));
        hipMalloc(&d_dstPtr, 36 * sizeof(Rpp8u));
        hipMemcpy(d_srcPtr1, srcPtr, 36 * sizeof(Rpp8u), hipMemcpyHostToDevice);
        hipMemcpy(d_dstPtr, dstPtr, 36 * sizeof(Rpp8u), hipMemcpyHostToDevice);

        start = clock();
        if (ip_bitDepth == 0)
        {
            rppi_tensor_transpose_u8_gpu(d_srcPtr1, d_dstPtr, inTensorDim, perm, handle);
        }
        else
            missingFuncFlag = 1;

        end = clock();

        if (missingFuncFlag != 1)
        {
            hipMemcpy(dstPtr, d_dstPtr, 36 * sizeof(Rpp8u), hipMemcpyDeviceToHost);

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

        Rpp8u *d_srcPtr1, *d_dstPtr;
        Rpp16f *d_srcPtr1_16f, *d_dstPtr_16f;
        Rpp32f *d_srcPtr1_32f, *d_dstPtr_32f;
        Rpp8s *d_srcPtr1_8s, *d_dstPtr_8s;

        if (ip_bitDepth == 0)
        {
            hipMalloc(&d_srcPtr1, 120 * sizeof(Rpp8u));
            hipMalloc(&d_dstPtr, 120 * sizeof(Rpp8u));
            hipMemcpy(d_srcPtr1, srcPtr, 120 * sizeof(Rpp8u), hipMemcpyHostToDevice);
            hipMemcpy(d_dstPtr, dstPtr, 120 * sizeof(Rpp8u), hipMemcpyHostToDevice);
            start = clock();
            rppi_tensor_transpose_u8_gpu(d_srcPtr1, d_dstPtr, shape, perm, handle);
            end = clock();

        }
        else if (ip_bitDepth == 1)
        {
            hipMalloc(&d_srcPtr1_16f, 120 * sizeof(Rpp16f));
            hipMalloc(&d_dstPtr_16f, 120 * sizeof(Rpp16f));
            hipMemcpy(d_srcPtr1_16f, srcPtr16f, 120 * sizeof(Rpp16f), hipMemcpyHostToDevice);
            hipMemcpy(d_dstPtr_16f, dstPtr16f, 120 * sizeof(Rpp16f), hipMemcpyHostToDevice);
            start = clock();
            rppi_tensor_transpose_f16_gpu(d_srcPtr1_16f, d_dstPtr_16f, shape, perm, handle);
            end = clock();
        }
        else if (ip_bitDepth == 2)
        {

            hipMalloc(&d_srcPtr1_32f, 120 * sizeof(Rpp32f));
            hipMalloc(&d_dstPtr_32f, 120 * sizeof(Rpp32f));
            hipMemcpy(d_srcPtr1_32f, srcPtr32f, 120 * sizeof(Rpp32f), hipMemcpyHostToDevice);
            hipMemcpy(d_dstPtr_32f, dstPtr32f, 120 * sizeof(Rpp32f), hipMemcpyHostToDevice);
            start = clock();
            rppi_tensor_transpose_f32_gpu(d_srcPtr1_32f, d_dstPtr_32f, shape, perm, handle);
            end = clock();
        }
        else if (ip_bitDepth == 3)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 4)
            missingFuncFlag = 1;
        else if (ip_bitDepth == 5)
        {

            hipMalloc(&d_srcPtr1_8s, 120 * sizeof(Rpp8s));
            hipMalloc(&d_dstPtr_8s, 120 * sizeof(Rpp8s));
            hipMemcpy(d_srcPtr1_8s, srcPtr8s, 120 * sizeof(Rpp8s), hipMemcpyHostToDevice);
            hipMemcpy(d_dstPtr_8s, dstPtr8s, 120 * sizeof(Rpp8s), hipMemcpyHostToDevice);
            start = clock();
            rppi_tensor_transpose_i8_gpu(d_srcPtr1_8s, d_dstPtr_8s, shape, perm, handle);
            end = clock();
        }
        else if (ip_bitDepth == 6)
            missingFuncFlag = 1;
        else
            missingFuncFlag = 1;

        if (ip_bitDepth == 0)
        {
            hipMemcpy(dstPtr, d_dstPtr, 120 * sizeof(Rpp8u), hipMemcpyDeviceToHost);
            printf("\n\nInput:\n");
            displayTensor(srcPtr, totalNumberOfElements);
            printf("\n\nOutput of transpose_u8:\n");
            displayTensor(dstPtr, totalNumberOfElements);
        }
        else if (ip_bitDepth == 1)
        {
            hipMemcpy(dstPtr16f, d_dstPtr_16f, 120 * sizeof(Rpp16f), hipMemcpyDeviceToHost);
            printf("\n\nInput:\n");
            displayTensorF(srcPtr16f, totalNumberOfElements);
            printf("\n\nOutput of transpose_f16:\n");
            displayTensorF(dstPtr16f, totalNumberOfElements);
        }
        else if (ip_bitDepth == 2)
        {
            hipMemcpy(dstPtr32f, d_dstPtr_32f, 120 * sizeof(Rpp32f), hipMemcpyDeviceToHost);
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
            hipMemcpy(dstPtr8s, d_dstPtr_8s, 120 * sizeof(Rpp8s), hipMemcpyDeviceToHost);
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

        Rpp8u *d_srcPtr1, *d_srcPtr2, *d_dstPtr;
        hipMalloc(&d_srcPtr1, 36 * sizeof(Rpp8u));
        hipMalloc(&d_srcPtr2, 36 * sizeof(Rpp8u));
        hipMalloc(&d_dstPtr, 36 * sizeof(Rpp8u));
        hipMemcpy(d_srcPtr1, srcPtr1, 36 * sizeof(Rpp8u), hipMemcpyHostToDevice);
        hipMemcpy(d_srcPtr2, srcPtr2, 36 * sizeof(Rpp8u), hipMemcpyHostToDevice);
        hipMemcpy(d_dstPtr, dstPtr, 36 * sizeof(Rpp8u), hipMemcpyHostToDevice);

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
            hipMemcpy(dstPtr, d_dstPtr, 36 * sizeof(Rpp8u), hipMemcpyDeviceToHost);

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

        Rpp8u *d_srcPtr1, *d_srcPtr2, *d_dstPtr;
        hipMalloc(&d_srcPtr1, 36 * sizeof(Rpp8u));
        hipMalloc(&d_srcPtr2, 36 * sizeof(Rpp8u));
        hipMalloc(&d_dstPtr, 36 * sizeof(Rpp8u));
        hipMemcpy(d_srcPtr1, srcPtr1, 36 * sizeof(Rpp8u), hipMemcpyHostToDevice);
        hipMemcpy(d_srcPtr2, srcPtr2, 36 * sizeof(Rpp8u), hipMemcpyHostToDevice);
        hipMemcpy(d_dstPtr, dstPtr, 36 * sizeof(Rpp8u), hipMemcpyHostToDevice);

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
            hipMemcpy(dstPtr, d_dstPtr, 36 * sizeof(Rpp8u), hipMemcpyDeviceToHost);

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

        Rpp8u *d_srcPtr1, *d_srcPtr2, *d_dstPtr;
        hipMalloc(&d_srcPtr1, 36 * sizeof(Rpp8u));
        hipMalloc(&d_srcPtr2, 36 * sizeof(Rpp8u));
        hipMalloc(&d_dstPtr, 36 * sizeof(Rpp8u));
        hipMemcpy(d_srcPtr1, srcPtr1, 36 * sizeof(Rpp8u), hipMemcpyHostToDevice);
        hipMemcpy(d_srcPtr2, srcPtr2, 36 * sizeof(Rpp8u), hipMemcpyHostToDevice);
        hipMemcpy(d_dstPtr, dstPtr, 36 * sizeof(Rpp8u), hipMemcpyHostToDevice);

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
            hipMemcpy(dstPtr, d_dstPtr, 36 * sizeof(Rpp8u), hipMemcpyDeviceToHost);

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

        Rpp8u *d_srcPtr1, *d_srcPtr2, *d_dstPtr;
        hipMalloc(&d_srcPtr1, 6 * sizeof(Rpp8u));
        hipMalloc(&d_srcPtr2, 8 * sizeof(Rpp8u));
        hipMalloc(&d_dstPtr, 12 * sizeof(Rpp8u));
        hipMemcpy(d_srcPtr1, srcPtr1, 6 * sizeof(Rpp8u), hipMemcpyHostToDevice);
        hipMemcpy(d_srcPtr2, srcPtr2, 8 * sizeof(Rpp8u), hipMemcpyHostToDevice);
        hipMemcpy(d_dstPtr, dstPtr, 12 * sizeof(Rpp8u), hipMemcpyHostToDevice);

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
            hipMemcpy(dstPtr, d_dstPtr, 12 * sizeof(Rpp8u), hipMemcpyDeviceToHost);

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
