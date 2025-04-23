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
#include "rpp.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>
#include <hip/hip_fp16.h>
#include <fstream>
#include "helpers/testSuite_helper.hpp"

using namespace cv;
using namespace std;

#define RPPPIXELCHECK(pixel) (pixel < (Rpp32f)0) ? ((Rpp32f)0) : ((pixel < (Rpp32f)255) ? pixel : ((Rpp32f)255))
#define RPPMAX2(a,b) ((a > b) ? a : b)
#define RPPMIN2(a,b) ((a < b) ? a : b)

std::string get_interpolation_type(unsigned int val, RpptInterpolationType &interpolationType)
{
    switch(val)
    {
        case 0:
        {
            interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
            return "NearestNeighbor";
        }
        case 2:
        {
            interpolationType = RpptInterpolationType::BICUBIC;
            return "Bicubic";
        }
        case 3:
        {
            interpolationType = RpptInterpolationType::LANCZOS;
            return "Lanczos";
        }
        case 4:
        {
            interpolationType = RpptInterpolationType::TRIANGULAR;
            return "Triangular";
        }
        case 5:
        {
            interpolationType = RpptInterpolationType::GAUSSIAN;
            return "Gaussian";
        }
        default:
        {
            interpolationType = RpptInterpolationType::BILINEAR;
            return "Bilinear";
        }
    }
}

std::string get_noise_type(unsigned int val)
{
    switch(val)
    {
        case 0: return "SaltAndPepper";
        case 1: return "Gaussian";
        case 2: return "Shot";
        default:return "SaltAndPepper";
    }
}

int main(int argc, char **argv)
{
    // Handle inputs

    const int MIN_ARG_COUNT = 7;

    if (argc < MIN_ARG_COUNT)
    {
        printf("\nImproper Usage! Needs all arguments!\n");
        printf("\nUsage: ./Tensor_host_pln1 <src1 folder> <src2 folder (place same as src1 folder for single image functionalities)> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <outputFormatToggle (pkd->pkd = 0 / pkd->pln = 1)> <case number = 0:86> <verbosity = 0/1>\n");
        return -1;
    }
    if (atoi(argv[4]) != 0)
    {
        printf("\nPLN1 cases don't have outputFormatToggle! Please input outputFormatToggle = 0\n");
        return -1;
    }

    char *src = argv[1];
    char *src_second = argv[2];
    int ip_bitDepth = atoi(argv[3]);
    unsigned int outputFormatToggle = atoi(argv[4]);
    int test_case = atoi(argv[5]);

    bool additionalParamCase = (test_case == 8 || test_case == 21 || test_case == 23 || test_case == 24 || test_case == 40 || test_case == 41 || test_case == 49);
    bool kernelSizeCase = (test_case == 40 || test_case == 41 || test_case == 49);
    bool interpolationTypeCase = (test_case == 21 || test_case == 23 || test_case == 24);
    bool noiseTypeCase = (test_case == 8);

    unsigned int verbosity = additionalParamCase ? atoi(argv[7]) : atoi(argv[6]);
    unsigned int additionalParam = additionalParamCase ? atoi(argv[6]) : 1;

    if (verbosity == 1)
    {
        printf("\nInputs for this test case are:");
        printf("\nsrc1 = %s", argv[1]);
        printf("\nsrc2 = %s", argv[2]);
        printf("\nu8 / f16 / f32 / u8->f16 / u8->f32 / i8 / u8->i8 (0/1/2/3/4/5/6) = %s", argv[3]);
        printf("\noutputFormatToggle (pkd->pkd = 0 / pkd->pln = 1) = %s", argv[4]);
        printf("\ncase number (0:86) = %s", argv[5]);
    }

    int ip_channel = 1;

    // Set case names

    char funcType[1000] = {"Tensor_HIP_PLN1_toPLN1"};

    char funcName[1000];
    switch (test_case)
    {
    case 0:
        strcpy(funcName, "brightness");
        outputFormatToggle = 0;
        break;
    case 1:
        strcpy(funcName, "gamma_correction");
        outputFormatToggle = 0;
        break;
    case 2:
        strcpy(funcName, "blend");
        outputFormatToggle = 0;
        break;
    case 4:
        strcpy(funcName, "contrast");
        outputFormatToggle = 0;
        break;
    case 8:
        strcpy(funcName, "noise");
        outputFormatToggle = 0;
        break;
    case 13:
        strcpy(funcName, "exposure");
        outputFormatToggle = 0;
        break;
    case 20:
        strcpy(funcName, "flip");
        outputFormatToggle = 0;
        break;
    case 21:
        strcpy(funcName, "resize");
        outputFormatToggle = 0;
        break;
    case 23:
        strcpy(funcName, "rotate");
        outputFormatToggle = 0;
        break;
    case 24:
        strcpy(funcName, "warp_affine");
        outputFormatToggle = 0;
        break;
    case 30:
        strcpy(funcName, "non_linear_blend");
        outputFormatToggle = 0;
        break;
    case 31:
        strcpy(funcName, "color_cast");
        outputFormatToggle = 0;
        break;
    case 36:
        strcpy(funcName, "color_twist");
        outputFormatToggle = 0;
        break;
    case 37:
        strcpy(funcName, "crop");
        outputFormatToggle = 0;
        break;
    case 38:
        strcpy(funcName, "crop_mirror_normalize");
        outputFormatToggle = 0;
        break;
    case 39:
        strcpy(funcName, "resize_crop_mirror");
        outputFormatToggle = 0;
        break;
    case 40:
        strcpy(funcName, "erode");
        outputFormatToggle = 0;
        break;
    case 41:
        strcpy(funcName, "dilate");
        outputFormatToggle = 0;
        break;
    case 49:
        strcpy(funcName, "box_filter");
        outputFormatToggle = 0;
        break;
    case 70:
        strcpy(funcName, "copy");
        outputFormatToggle = 0;
        break;
    case 80:
        strcpy(funcName, "resize_mirror_normalize");
        outputFormatToggle = 0;
        break;
    case 83:
        strcpy(funcName, "gridmask");
        outputFormatToggle = 0;
        break;
    case 84:
        strcpy(funcName, "spatter");
        outputFormatToggle = 0;
        break;
    case 85:
        strcpy(funcName, "swap_channels");
        outputFormatToggle = 0;
        break;
    case 86:
        strcpy(funcName, "color_to_greyscale");
        outputFormatToggle = 0;
        break;
    default:
        strcpy(funcName, "test_case");
        break;
    }

    // Initialize tensor descriptors

    RpptDesc srcDesc, dstDesc;
    RpptDescPtr srcDescPtr, dstDescPtr;
    srcDescPtr = &srcDesc;
    dstDescPtr = &dstDesc;

    // Set src/dst layouts in tensor descriptors

    srcDescPtr->layout = RpptLayout::NCHW;
    dstDescPtr->layout = RpptLayout::NCHW;

    // Set src/dst data types in tensor descriptors

    if (ip_bitDepth == 0)
    {
        strcat(funcName, "_u8_");
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::U8;
    }
    else if (ip_bitDepth == 1)
    {
        strcat(funcName, "_f16_");
        srcDescPtr->dataType = RpptDataType::F16;
        dstDescPtr->dataType = RpptDataType::F16;
    }
    else if (ip_bitDepth == 2)
    {
        strcat(funcName, "_f32_");
        srcDescPtr->dataType = RpptDataType::F32;
        dstDescPtr->dataType = RpptDataType::F32;
    }
    else if (ip_bitDepth == 3)
    {
        strcat(funcName, "_u8_f16_");
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::F16;
    }
    else if (ip_bitDepth == 4)
    {
        strcat(funcName, "_u8_f32_");
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::F32;
    }
    else if (ip_bitDepth == 5)
    {
        strcat(funcName, "_i8_");
        srcDescPtr->dataType = RpptDataType::I8;
        dstDescPtr->dataType = RpptDataType::I8;
    }
    else if (ip_bitDepth == 6)
    {
        strcat(funcName, "_u8_i8_");
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::I8;
    }

    // Other initializations

    int missingFuncFlag = 0;
    int i = 0, j = 0;
    int maxHeight = 0, maxWidth = 0;
    int maxDstHeight = 0, maxDstWidth = 0;
    unsigned long long count = 0;
    unsigned long long ioBufferSize = 0;
    unsigned long long oBufferSize = 0;
    static int noOfImages = 0;
    Mat image, image_second;

    // String ops on function name

    char src1[1000];
    strcpy(src1, src);
    strcat(src1, "/");
    char src1_second[1000];
    strcpy(src1_second, src_second);
    strcat(src1_second, "/");

    char func[1000];
    strcpy(func, funcName);
    strcat(func, funcType);

    RpptInterpolationType interpolationType = RpptInterpolationType::BILINEAR;
    if (kernelSizeCase)
    {
        char additionalParam_char[2];
        std::snprintf(additionalParam_char, sizeof(additionalParam_char), "%u", additionalParam);
        strcat(func, "_kSize");
        strcat(func, additionalParam_char);
    }
    else if (interpolationTypeCase)
    {
        std::string interpolationTypeName;
        interpolationTypeName = get_interpolation_type(additionalParam, interpolationType);
        strcat(func, "_interpolationType");
        strcat(func, interpolationTypeName.c_str());
    }
    else if (noiseTypeCase)
    {
        std::string noiseTypeName;
        noiseTypeName = get_noise_type(additionalParam);
        strcat(func, "_noiseType");
        strcat(func, noiseTypeName.c_str());
    }

    // Get number of images

    struct dirent *de;
    DIR *dr = opendir(src);
    while ((de = readdir(dr)) != NULL)
    {
        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
            continue;
        noOfImages += 1;
    }
    closedir(dr);

    // Initialize ROI tensors for src/dst

    RpptROI *roiTensorPtrSrc, *roiTensorPtrDst;
    hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
    hipHostMalloc(&roiTensorPtrDst, noOfImages * sizeof(RpptROI));

    // Initialize the ImagePatch for source and destination
    RpptImagePatch *srcImgSizes, *dstImgSizes;
    hipHostMalloc(&srcImgSizes, noOfImages * sizeof(RpptImagePatch));
    hipHostMalloc(&dstImgSizes, noOfImages * sizeof(RpptImagePatch));

    // Set ROI tensors types for src/dst

    RpptRoiType roiTypeSrc, roiTypeDst;
    roiTypeSrc = RpptRoiType::XYWH;
    roiTypeDst = RpptRoiType::XYWH;

    // Set maxHeight, maxWidth and ROIs for src/dst

    const int images = noOfImages;
    char imageNames[images][1000];

    DIR *dr1 = opendir(src);
    while ((de = readdir(dr1)) != NULL)
    {
        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
            continue;
        strcpy(imageNames[count], de->d_name);
        char temp[1000];
        strcpy(temp, src1);
        strcat(temp, imageNames[count]);

        image = imread(temp, 0);

        roiTensorPtrSrc[count].xywhROI.xy.x = 0;
        roiTensorPtrSrc[count].xywhROI.xy.y = 0;
        roiTensorPtrSrc[count].xywhROI.roiWidth = image.cols;
        roiTensorPtrSrc[count].xywhROI.roiHeight = image.rows;

        roiTensorPtrDst[count].xywhROI.xy.x = 0;
        roiTensorPtrDst[count].xywhROI.xy.y = 0;
        roiTensorPtrDst[count].xywhROI.roiWidth = image.cols;
        roiTensorPtrDst[count].xywhROI.roiHeight = image.rows;

        srcImgSizes[count].width = roiTensorPtrSrc[count].xywhROI.roiWidth;
        srcImgSizes[count].height = roiTensorPtrSrc[count].xywhROI.roiHeight;
        dstImgSizes[count].width = roiTensorPtrDst[count].xywhROI.roiWidth;
        dstImgSizes[count].height = roiTensorPtrDst[count].xywhROI.roiHeight;

        maxHeight = RPPMAX2(maxHeight, roiTensorPtrSrc[count].xywhROI.roiHeight);
        maxWidth = RPPMAX2(maxWidth, roiTensorPtrSrc[count].xywhROI.roiWidth);
        maxDstHeight = RPPMAX2(maxDstHeight, roiTensorPtrDst[count].xywhROI.roiHeight);
        maxDstWidth = RPPMAX2(maxDstWidth, roiTensorPtrDst[count].xywhROI.roiWidth);

        count++;
    }
    closedir(dr1);

    // Set numDims, offset, n/c/h/w values, n/c/h/w strides for src/dst

    srcDescPtr->numDims = 4;
    dstDescPtr->numDims = 4;

    srcDescPtr->offsetInBytes = 64;
    dstDescPtr->offsetInBytes = 0;

    srcDescPtr->n = noOfImages;
    srcDescPtr->c = ip_channel;
    srcDescPtr->h = maxHeight;
    srcDescPtr->w = maxWidth;

    dstDescPtr->n = noOfImages;
    dstDescPtr->c = ip_channel;
    dstDescPtr->h = maxDstHeight;
    dstDescPtr->w = maxDstWidth;

    // Optionally set w stride as a multiple of 8 for src/dst

    srcDescPtr->w = ((srcDescPtr->w / 8) * 8) + 8;
    dstDescPtr->w = ((dstDescPtr->w / 8) * 8) + 8;

    // Set n/c/h/w strides for src/dst

    srcDescPtr->strides.nStride = ip_channel * srcDescPtr->w * srcDescPtr->h;
    srcDescPtr->strides.cStride = srcDescPtr->w * srcDescPtr->h;
    srcDescPtr->strides.hStride = srcDescPtr->w;
    srcDescPtr->strides.wStride = 1;

    if (dstDescPtr->layout == RpptLayout::NHWC)
    {
        dstDescPtr->strides.nStride = ip_channel * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = ip_channel * dstDescPtr->w;
        dstDescPtr->strides.wStride = ip_channel;
        dstDescPtr->strides.cStride = 1;
    }
    else if (dstDescPtr->layout == RpptLayout::NCHW)
    {
        dstDescPtr->strides.nStride = ip_channel * dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.cStride = dstDescPtr->w * dstDescPtr->h;
        dstDescPtr->strides.hStride = dstDescPtr->w;
        dstDescPtr->strides.wStride = 1;
    }

    // Set buffer sizes in pixels for src/dst

    ioBufferSize = (unsigned long long)srcDescPtr->h * (unsigned long long)srcDescPtr->w * (unsigned long long)ip_channel * (unsigned long long)noOfImages;
    oBufferSize = (unsigned long long)dstDescPtr->h * (unsigned long long)dstDescPtr->w * (unsigned long long)ip_channel * (unsigned long long)noOfImages;

    // Set buffer sizes in bytes for src/dst (including offsets)

    unsigned long long ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
    unsigned long long oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;
    unsigned long long ioBufferSizeInBytes_f16 = (ioBufferSize * 2) + srcDescPtr->offsetInBytes;
    unsigned long long oBufferSizeInBytes_f16 = (oBufferSize * 2) + dstDescPtr->offsetInBytes;
    unsigned long long ioBufferSizeInBytes_f32 = (ioBufferSize * 4) + srcDescPtr->offsetInBytes;
    unsigned long long oBufferSizeInBytes_f32 = (oBufferSize * 4) + dstDescPtr->offsetInBytes;
    unsigned long long ioBufferSizeInBytes_i8 = ioBufferSize + srcDescPtr->offsetInBytes;
    unsigned long long oBufferSizeInBytes_i8 = oBufferSize + dstDescPtr->offsetInBytes;

    // Initialize host buffers for src/dst

    Rpp8u *input = (Rpp8u *)calloc(ioBufferSize, sizeof(Rpp8u));
    Rpp8u *input_second = (Rpp8u *)calloc(ioBufferSize, sizeof(Rpp8u));
    Rpp8u *output = (Rpp8u *)calloc(oBufferSize, sizeof(Rpp8u));

    // Set 8u host buffers for src/dst

    DIR *dr2 = opendir(src);
    DIR *dr2_second = opendir(src_second);
    count = 0;
    i = 0;

    Rpp8u *offsetted_input, *offsetted_input_second;
    offsetted_input = input + srcDescPtr->offsetInBytes;
    offsetted_input_second = input_second + srcDescPtr->offsetInBytes;

    Rpp32u elementsInRowMax = srcDescPtr->w * ip_channel;

    while ((de = readdir(dr2)) != NULL)
    {
        Rpp8u *input_temp, *input_second_temp;
        input_temp = offsetted_input + (i * srcDescPtr->strides.nStride);
        input_second_temp = offsetted_input_second + (i * srcDescPtr->strides.nStride);

        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
            continue;

        char temp[1000];
        strcpy(temp, src1);
        strcat(temp, de->d_name);

        char temp_second[1000];
        strcpy(temp_second, src1_second);
        strcat(temp_second, de->d_name);

        image = imread(temp, 0);
        image_second = imread(temp_second, 0);

        Rpp8u *ip_image = image.data;
        Rpp8u *ip_image_second = image_second.data;

        Rpp32u elementsInRow = roiTensorPtrSrc[i].xywhROI.roiWidth * ip_channel;

        for (j = 0; j < roiTensorPtrSrc[i].xywhROI.roiHeight; j++)
        {
            memcpy(input_temp, ip_image, elementsInRow * sizeof (Rpp8u));
            memcpy(input_second_temp, ip_image_second, elementsInRow * sizeof (Rpp8u));
            ip_image += elementsInRow;
            ip_image_second += elementsInRow;
            input_temp += elementsInRowMax;
            input_second_temp += elementsInRowMax;
        }
        i++;
        count += srcDescPtr->strides.nStride;
    }
    closedir(dr2);

    // Convert inputs to test various other bit depths and copy to hip buffers


    half *inputf16, *inputf16_second, *outputf16;
    Rpp32f *inputf32, *inputf32_second, *outputf32;
    Rpp8s *inputi8, *inputi8_second, *outputi8;
    int *d_input, *d_input_second, *d_inputf16, *d_inputf16_second, *d_inputf32, *d_inputf32_second, *d_inputi8, *d_inputi8_second;
    int *d_output, *d_outputf16, *d_outputf32, *d_outputi8;

    // Factors to convert U8 data to F32, F16 data to 0-1 range and reconvert them back to 0 -255 range

    Rpp32f conversionFactor = 1.0f / 255.0;
    if(test_case == 38)
        conversionFactor = 1.0;

    if (ip_bitDepth == 0)
    {
        hipMalloc(&d_input, ioBufferSizeInBytes_u8);
        hipMalloc(&d_input_second, ioBufferSizeInBytes_u8);
        hipMalloc(&d_output, oBufferSizeInBytes_u8);
        hipMemcpy(d_input, input, ioBufferSizeInBytes_u8, hipMemcpyHostToDevice);
        hipMemcpy(d_input_second, input_second, ioBufferSizeInBytes_u8, hipMemcpyHostToDevice);
        hipMemcpy(d_output, output, oBufferSizeInBytes_u8, hipMemcpyHostToDevice);
    }
    else if (ip_bitDepth == 1)
    {
        inputf16 = (half *)calloc(ioBufferSizeInBytes_f16, 1);
        inputf16_second = (half *)calloc(ioBufferSizeInBytes_f16, 1);
        outputf16 = (half *)calloc(oBufferSizeInBytes_f16, 1);

        Rpp8u *inputTemp, *input_secondTemp;
        half *inputf16Temp, *inputf16_secondTemp;

        inputTemp = input + srcDescPtr->offsetInBytes;
        input_secondTemp = input_second + srcDescPtr->offsetInBytes;

        inputf16Temp = (half *)((Rpp8u *)inputf16 + srcDescPtr->offsetInBytes);
        inputf16_secondTemp = (half *)((Rpp8u *)inputf16_second + srcDescPtr->offsetInBytes);

        for (int i = 0; i < ioBufferSize; i++)
        {
            *inputf16Temp = (half)(((float)*inputTemp) * conversionFactor);
            *inputf16_secondTemp = (half)(((float)*input_secondTemp) * conversionFactor);
            inputTemp++;
            inputf16Temp++;
            input_secondTemp++;
            inputf16_secondTemp++;
        }

        hipMalloc(&d_inputf16, ioBufferSizeInBytes_f16);
        hipMalloc(&d_inputf16_second, ioBufferSizeInBytes_f16);
        hipMalloc(&d_outputf16, oBufferSizeInBytes_f16);
        hipMemcpy(d_inputf16, inputf16, ioBufferSizeInBytes_f16, hipMemcpyHostToDevice);
        hipMemcpy(d_inputf16_second, inputf16_second, ioBufferSizeInBytes_f16, hipMemcpyHostToDevice);
        hipMemcpy(d_outputf16, outputf16, oBufferSizeInBytes_f16, hipMemcpyHostToDevice);
    }
    else if (ip_bitDepth == 2)
    {
        inputf32 = (Rpp32f *)calloc(ioBufferSizeInBytes_f32, 1);
        inputf32_second = (Rpp32f *)calloc(ioBufferSizeInBytes_f32, 1);
        outputf32 = (Rpp32f *)calloc(oBufferSizeInBytes_f32, 1);

        Rpp8u *inputTemp, *input_secondTemp;
        Rpp32f *inputf32Temp, *inputf32_secondTemp;

        inputTemp = input + srcDescPtr->offsetInBytes;
        input_secondTemp = input_second + srcDescPtr->offsetInBytes;

        inputf32Temp = (Rpp32f *)((Rpp8u *)inputf32 + srcDescPtr->offsetInBytes);
        inputf32_secondTemp = (Rpp32f *)((Rpp8u *)inputf32_second + srcDescPtr->offsetInBytes);

        for (int i = 0; i < ioBufferSize; i++)
        {
            *inputf32Temp = ((Rpp32f)*inputTemp) * conversionFactor;
            *inputf32_secondTemp = ((Rpp32f)*input_secondTemp) * conversionFactor;
            inputTemp++;
            inputf32Temp++;
            input_secondTemp++;
            inputf32_secondTemp++;
        }

        hipMalloc(&d_inputf32, ioBufferSizeInBytes_f32);
        hipMalloc(&d_inputf32_second, ioBufferSizeInBytes_f32);
        hipMalloc(&d_outputf32, oBufferSizeInBytes_f32);
        hipMemcpy(d_inputf32, inputf32, ioBufferSizeInBytes_f32, hipMemcpyHostToDevice);
        hipMemcpy(d_inputf32_second, inputf32_second, ioBufferSizeInBytes_f32, hipMemcpyHostToDevice);
        hipMemcpy(d_outputf32, outputf32, oBufferSizeInBytes_f32, hipMemcpyHostToDevice);
    }
    else if (ip_bitDepth == 3)
    {
        outputf16 = (half *)calloc(oBufferSizeInBytes_f16, 1);
        hipMalloc(&d_input, ioBufferSizeInBytes_u8);
        hipMalloc(&d_input_second, ioBufferSizeInBytes_u8);
        hipMalloc(&d_outputf16, oBufferSizeInBytes_f16);
        hipMemcpy(d_input, input, ioBufferSizeInBytes_u8, hipMemcpyHostToDevice);
        hipMemcpy(d_input_second, input_second, ioBufferSizeInBytes_u8, hipMemcpyHostToDevice);
        hipMemcpy(d_outputf16, outputf16, oBufferSizeInBytes_f16, hipMemcpyHostToDevice);
    }
    else if (ip_bitDepth == 4)
    {
        outputf32 = (Rpp32f *)calloc(oBufferSizeInBytes_f32, 1);
        hipMalloc(&d_input, ioBufferSizeInBytes_u8);
        hipMalloc(&d_input_second, ioBufferSizeInBytes_u8);
        hipMalloc(&d_outputf32, oBufferSizeInBytes_f32);
        hipMemcpy(d_input, input, ioBufferSizeInBytes_u8, hipMemcpyHostToDevice);
        hipMemcpy(d_input_second, input_second, ioBufferSizeInBytes_u8, hipMemcpyHostToDevice);
        hipMemcpy(d_outputf32, outputf32, oBufferSizeInBytes_f32, hipMemcpyHostToDevice);
    }
    else if (ip_bitDepth == 5)
    {
        inputi8 = (Rpp8s *)calloc(ioBufferSizeInBytes_i8, 1);
        inputi8_second = (Rpp8s *)calloc(ioBufferSizeInBytes_i8, 1);
        outputi8 = (Rpp8s *)calloc(oBufferSizeInBytes_i8, 1);

        Rpp8u *inputTemp, *input_secondTemp;
        Rpp8s *inputi8Temp, *inputi8_secondTemp;

        inputTemp = input + srcDescPtr->offsetInBytes;
        input_secondTemp = input_second + srcDescPtr->offsetInBytes;

        inputi8Temp = inputi8 + srcDescPtr->offsetInBytes;
        inputi8_secondTemp = inputi8_second + srcDescPtr->offsetInBytes;

        for (int i = 0; i < ioBufferSize; i++)
        {
            *inputi8Temp = (Rpp8s) (((Rpp32s) *inputTemp) - 128);
            *inputi8_secondTemp = (Rpp8s) (((Rpp32s) *input_secondTemp) - 128);
            inputTemp++;
            inputi8Temp++;
            input_secondTemp++;
            inputi8_secondTemp++;
        }

        hipMalloc(&d_inputi8, ioBufferSizeInBytes_i8);
        hipMalloc(&d_inputi8_second, ioBufferSizeInBytes_i8);
        hipMalloc(&d_outputi8, oBufferSizeInBytes_i8);
        hipMemcpy(d_inputi8, inputi8, ioBufferSizeInBytes_i8, hipMemcpyHostToDevice);
        hipMemcpy(d_inputi8_second, inputi8_second, ioBufferSizeInBytes_i8, hipMemcpyHostToDevice);
        hipMemcpy(d_outputi8, outputi8, oBufferSizeInBytes_i8, hipMemcpyHostToDevice);
    }
    else if (ip_bitDepth == 6)
    {
        outputi8 = (Rpp8s *)calloc(oBufferSizeInBytes_i8, 1);
        hipMalloc(&d_input, ioBufferSizeInBytes_u8);
        hipMalloc(&d_input_second, ioBufferSizeInBytes_u8);
        hipMalloc(&d_outputi8, oBufferSizeInBytes_i8);
        hipMemcpy(d_input, input, ioBufferSizeInBytes_u8, hipMemcpyHostToDevice);
        hipMemcpy(d_input_second, input_second, ioBufferSizeInBytes_u8, hipMemcpyHostToDevice);
        hipMemcpy(d_outputi8, outputi8, oBufferSizeInBytes_i8, hipMemcpyHostToDevice);
    }

    // Run case-wise RPP API and measure time

    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    RppBackend backend = RppBackend::RPP_HIP_BACKEND;
    rppCreate(&handle, noOfImages, 0, stream, backend);

    clock_t start, end;
    double max_time_used = 0, min_time_used = 500, avg_time_used = 0;

    string test_case_name;

    printf("\nRunning %s 100 times (each time with a batch size of %d images) and computing mean statistics...", func, noOfImages);

    for (int perfRunCount = 0; perfRunCount < 100; perfRunCount++)
    {
        double gpu_time_used;
        switch (test_case)
        {
        case 0:
        {
            test_case_name = "brightness";

            Rpp32f alpha[images];
            Rpp32f beta[images];
            for (i = 0; i < images; i++)
            {
                alpha[i] = 1.75;
                beta[i] = 50;
            }

            // Uncomment to run test case with an xywhROI override
            /*for (i = 0; i < images; i++)
            {
                roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
            }*/

            // Uncomment to run test case with an ltrbROI override
            /*for (i = 0; i < images; i++)
            {
                roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
                dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
                dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
            }
            roiTypeSrc = RpptRoiType::LTRB;
            roiTypeDst = RpptRoiType::LTRB;*/

            start = clock();

            if (ip_bitDepth == 0)
                rppt_brightness_gpu(d_input, srcDescPtr, d_output, dstDescPtr, alpha, beta, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 1)
                rppt_brightness_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, alpha, beta, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 2)
                rppt_brightness_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, alpha, beta, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                rppt_brightness_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, alpha, beta, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            break;
        }
        case 1:
        {
            test_case_name = "gamma_correction";

            Rpp32f gammaVal[images];
            for (i = 0; i < images; i++)
            {
                gammaVal[i] = 1.9;
            }

            // Uncomment to run test case with an xywhROI override
            /*for (i = 0; i < images; i++)
            {
                roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
            }*/

            // Uncomment to run test case with an ltrbROI override
            /*for (i = 0; i < images; i++)
            {
                roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
                dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
                dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
            }
            roiTypeSrc = RpptRoiType::LTRB;
            roiTypeDst = RpptRoiType::LTRB;*/

            start = clock();

            if (ip_bitDepth == 0)
                rppt_gamma_correction_gpu(d_input, srcDescPtr, d_output, dstDescPtr, gammaVal, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 1)
                rppt_gamma_correction_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, gammaVal, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 2)
                rppt_gamma_correction_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, gammaVal, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                rppt_gamma_correction_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, gammaVal, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            break;
        }
        case 2:
        {
            test_case_name = "blend";

            Rpp32f alpha[images];
            for (i = 0; i < images; i++)
            {
                alpha[i] = 0.4;
            }

            // Uncomment to run test case with an xywhROI override
            /*for (i = 0; i < images; i++)
            {
                roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
            }*/

            // Uncomment to run test case with an ltrbROI override
            /*for (i = 0; i < images; i++)
            {
                roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
                dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
                dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
            }
            roiTypeSrc = RpptRoiType::LTRB;
            roiTypeDst = RpptRoiType::LTRB;*/

            start = clock();

            if (ip_bitDepth == 0)
                rppt_blend_gpu(d_input, d_input_second, srcDescPtr, d_output, dstDescPtr, alpha, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 1)
                rppt_blend_gpu(d_inputf16, d_inputf16_second, srcDescPtr, d_outputf16, dstDescPtr, alpha, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 2)
                rppt_blend_gpu(d_inputf32, d_inputf32_second, srcDescPtr, d_outputf32, dstDescPtr, alpha, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                rppt_blend_gpu(d_inputi8, d_inputi8_second, srcDescPtr, d_outputi8, dstDescPtr, alpha, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            break;
        }
        case 4:
        {
            test_case_name = "contrast";

            Rpp32f contrastFactor[images];
            Rpp32f contrastCenter[images];
            for (i = 0; i < images; i++)
            {
                contrastFactor[i] = 2.96;
                contrastCenter[i] = 128;
            }

            // Uncomment to run test case with an xywhROI override
            /*for (i = 0; i < images; i++)
            {
                roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
            }*/

            // Uncomment to run test case with an ltrbROI override
            /*for (i = 0; i < images; i++)
            {
                roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
                dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
                dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
            }
            roiTypeSrc = RpptRoiType::LTRB;
            roiTypeDst = RpptRoiType::LTRB;*/


            start = clock();

            if (ip_bitDepth == 0)
                rppt_contrast_gpu(d_input, srcDescPtr, d_output, dstDescPtr, contrastFactor, contrastCenter, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 1)
                rppt_contrast_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, contrastFactor, contrastCenter, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 2)
                rppt_contrast_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, contrastFactor, contrastCenter, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                rppt_contrast_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, contrastFactor, contrastCenter, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            break;
        }
        case 8:
        {
            test_case_name = "noise";

            switch(additionalParam)
            {
                case 0:
                {
                    Rpp32f noiseProbabilityTensor[images];
                    Rpp32f saltProbabilityTensor[images];
                    Rpp32f saltValueTensor[images];
                    Rpp32f pepperValueTensor[images];
                    Rpp32u seed = 1255459;
                    for (i = 0; i < images; i++)
                    {
                        noiseProbabilityTensor[i] = 0.1f;
                        saltProbabilityTensor[i] = 0.5f;
                        saltValueTensor[i] = 1.0f;
                        pepperValueTensor[i] = 0.0f;
                    }

                    // Uncomment to run test case with an xywhROI override
                    /*for (i = 0; i < images; i++)
                    {
                        roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                        roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                        dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                        dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
                    }*/

                    // Uncomment to run test case with an ltrbROI override
                    /*for (i = 0; i < images; i++)
                    {
                        roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                        roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                        roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                        roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
                        dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
                        dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
                    }
                    roiTypeSrc = RpptRoiType::LTRB;
                    roiTypeDst = RpptRoiType::LTRB;*/

                    start = clock();

                    if (ip_bitDepth == 0)
                        rppt_salt_and_pepper_noise_gpu(d_input, srcDescPtr, d_output, dstDescPtr, noiseProbabilityTensor, saltProbabilityTensor, saltValueTensor, pepperValueTensor, seed, roiTensorPtrSrc, roiTypeSrc, handle);
                    else if (ip_bitDepth == 1)
                        rppt_salt_and_pepper_noise_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, noiseProbabilityTensor, saltProbabilityTensor, saltValueTensor, pepperValueTensor, seed, roiTensorPtrSrc, roiTypeSrc, handle);
                    else if (ip_bitDepth == 2)
                        rppt_salt_and_pepper_noise_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, noiseProbabilityTensor, saltProbabilityTensor, saltValueTensor, pepperValueTensor, seed, roiTensorPtrSrc, roiTypeSrc, handle);
                    else if (ip_bitDepth == 3)
                        missingFuncFlag = 1;
                    else if (ip_bitDepth == 4)
                        missingFuncFlag = 1;
                    else if (ip_bitDepth == 5)
                        rppt_salt_and_pepper_noise_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, noiseProbabilityTensor, saltProbabilityTensor, saltValueTensor, pepperValueTensor, seed, roiTensorPtrSrc, roiTypeSrc, handle);
                    else if (ip_bitDepth == 6)
                        missingFuncFlag = 1;
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case 1:
                {
                    Rpp32f meanTensor[images];
                    Rpp32f stdDevTensor[images];
                    Rpp32u seed = 1255459;
                    for (i = 0; i < images; i++)
                    {
                        meanTensor[i] = 0.0f;
                        stdDevTensor[i] = 0.2f;
                    }

                    // Uncomment to run test case with an xywhROI override
                    /*for (i = 0; i < images; i++)
                    {
                        roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                        roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                        dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                        dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
                    }*/

                    // Uncomment to run test case with an ltrbROI override
                    /*for (i = 0; i < images; i++)
                    {
                        roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                        roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                        roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                        roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
                        dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
                        dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
                    }
                    roiTypeSrc = RpptRoiType::LTRB;
                    roiTypeDst = RpptRoiType::LTRB;*/

                    start = clock();

                    if (ip_bitDepth == 0)
                        rppt_gaussian_noise_gpu(d_input, srcDescPtr, d_output, dstDescPtr, meanTensor, stdDevTensor, seed, roiTensorPtrSrc, roiTypeSrc, handle);
                    else if (ip_bitDepth == 1)
                        rppt_gaussian_noise_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, meanTensor, stdDevTensor, seed, roiTensorPtrSrc, roiTypeSrc, handle);
                    else if (ip_bitDepth == 2)
                        rppt_gaussian_noise_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, meanTensor, stdDevTensor, seed, roiTensorPtrSrc, roiTypeSrc, handle);
                    else if (ip_bitDepth == 3)
                        missingFuncFlag = 1;
                    else if (ip_bitDepth == 4)
                        missingFuncFlag = 1;
                    else if (ip_bitDepth == 5)
                        rppt_gaussian_noise_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, meanTensor, stdDevTensor, seed, roiTensorPtrSrc, roiTypeSrc, handle);
                    else if (ip_bitDepth == 6)
                        missingFuncFlag = 1;
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case 2:
                {
                    Rpp32f shotNoiseFactorTensor[images];
                    Rpp32u seed = 1255459;
                    for (i = 0; i < images; i++)
                    {
                        shotNoiseFactorTensor[i] = 80.0f;
                    }

                    // Uncomment to run test case with an xywhROI override
                    /*for (i = 0; i < images; i++)
                    {
                        roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                        roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                        dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                        dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
                    }*/

                    // Uncomment to run test case with an ltrbROI override
                    /*for (i = 0; i < images; i++)
                    {
                        roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                        roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                        roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                        roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
                        dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
                        dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
                    }
                    roiTypeSrc = RpptRoiType::LTRB;
                    roiTypeDst = RpptRoiType::LTRB;*/

                    start = clock();

                    if (ip_bitDepth == 0)
                        rppt_shot_noise_gpu(d_input, srcDescPtr, d_output, dstDescPtr, shotNoiseFactorTensor, seed, roiTensorPtrSrc, roiTypeSrc, handle);
                    else if (ip_bitDepth == 1)
                        rppt_shot_noise_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, shotNoiseFactorTensor, seed, roiTensorPtrSrc, roiTypeSrc, handle);
                    else if (ip_bitDepth == 2)
                        rppt_shot_noise_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, shotNoiseFactorTensor, seed, roiTensorPtrSrc, roiTypeSrc, handle);
                    else if (ip_bitDepth == 3)
                        missingFuncFlag = 1;
                    else if (ip_bitDepth == 4)
                        missingFuncFlag = 1;
                    else if (ip_bitDepth == 5)
                        rppt_shot_noise_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, shotNoiseFactorTensor, seed, roiTensorPtrSrc, roiTypeSrc, handle);
                    else if (ip_bitDepth == 6)
                        missingFuncFlag = 1;
                    else
                        missingFuncFlag = 1;

                    break;
                }
                default:
                {
                    missingFuncFlag = 1;
                    break;
                }
            }

            break;
        }
        case 13:
        {
            test_case_name = "exposure";

            Rpp32f exposureFactor[images];
            for (i = 0; i < images; i++)
            {
                exposureFactor[i] = 1.4;
            }

            // Uncomment to run test case with an xywhROI override
            /*for (i = 0; i < images; i++)
            {
                roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
            }*/

            // Uncomment to run test case with an ltrbROI override
            /*for (i = 0; i < images; i++)
            {
                roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
                dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
                dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
            }
            roiTypeSrc = RpptRoiType::LTRB;
            roiTypeDst = RpptRoiType::LTRB;*/

            start = clock();

            if (ip_bitDepth == 0)
                rppt_exposure_gpu(d_input, srcDescPtr, d_output, dstDescPtr, exposureFactor, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 1)
                rppt_exposure_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, exposureFactor, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 2)
                rppt_exposure_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, exposureFactor, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                rppt_exposure_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, exposureFactor, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            break;
        }
        case 20:
        {
            test_case_name = "flip";

            Rpp32u horizontalFlag[images];
            Rpp32u verticalFlag[images];
            for (i = 0; i < images; i++)
            {
                horizontalFlag[i] = 1;
                verticalFlag[i] = 0;
            }

            // Uncomment to run test case with an xywhROI override
            /*for (i = 0; i < images; i++)
            {
                roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
            }*/

            // Uncomment to run test case with an ltrbROI override
            /*for (i = 0; i < images; i++)
            {
                roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
                dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
                dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
            }
            roiTypeSrc = RpptRoiType::LTRB;
            roiTypeDst = RpptRoiType::LTRB;*/

            start = clock();

            if (ip_bitDepth == 0)
                rppt_flip_gpu(d_input, srcDescPtr, d_output, dstDescPtr, horizontalFlag, verticalFlag, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 1)
                rppt_flip_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, horizontalFlag, verticalFlag, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 2)
                rppt_flip_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, horizontalFlag, verticalFlag, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                rppt_flip_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, horizontalFlag, verticalFlag, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            break;
        }
        case 21:
        {
            test_case_name = "resize";

            for (i = 0; i < images; i++)
            {
                dstImgSizes[i].width = roiTensorPtrDst[i].xywhROI.roiWidth = roiTensorPtrSrc[i].xywhROI.roiWidth / 1.1;
                dstImgSizes[i].height = roiTensorPtrDst[i].xywhROI.roiHeight = roiTensorPtrSrc[i].xywhROI.roiHeight / 3;
            }

            // Uncomment to run test case with an xywhROI override
            /*for (i = 0; i < images; i++)
            {
                roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
            }*/

            // Uncomment to run test case with an ltrbROI override
            /*for (i = 0; i < images; i++)
            {
                roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
                dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
                dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
            }
            roiTypeSrc = RpptRoiType::LTRB;
            roiTypeDst = RpptRoiType::LTRB;*/

            start = clock();

            if (ip_bitDepth == 0)
                rppt_resize_gpu(d_input, srcDescPtr, d_output, dstDescPtr, dstImgSizes, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 1)
                rppt_resize_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, dstImgSizes, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 2)
                rppt_resize_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, dstImgSizes, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                rppt_resize_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, dstImgSizes, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            break;
        }
        case 23:
        {
            test_case_name = "rotate";

            if ((interpolationType != RpptInterpolationType::BILINEAR) && (interpolationType != RpptInterpolationType::NEAREST_NEIGHBOR))
            {
                missingFuncFlag = 1;
                break;
            }

            Rpp32f angle[images];
            for (i = 0; i < images; i++)
            {
                angle[i] = 50;
            }

            // Uncomment to run test case with an xywhROI override
            /*for (i = 0; i < images; i++)
            {
                roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
            }*/

            // Uncomment to run test case with an ltrbROI override
            /*for (i = 0; i < images; i++)
                roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
                dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
                dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
            }
            roiTypeSrc = RpptRoiType::LTRB;
            roiTypeDst = RpptRoiType::LTRB;*/

            start = clock();

            if (ip_bitDepth == 0)
                rppt_rotate_gpu(d_input, srcDescPtr, d_output, dstDescPtr, angle, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 1)
                rppt_rotate_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, angle, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 2)
                rppt_rotate_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, angle, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                rppt_rotate_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, angle, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            break;
        }
        case 24:
        {
            test_case_name = "warp_affine";

            if ((interpolationType != RpptInterpolationType::BILINEAR) && (interpolationType != RpptInterpolationType::NEAREST_NEIGHBOR))
            {
                missingFuncFlag = 1;
                break;
            }

            Rpp32f6 affineTensor_f6[images];
            Rpp32f *affineTensor = (Rpp32f *)affineTensor_f6;
            for (i = 0; i < images; i++)
            {
                affineTensor_f6[i].data[0] = 1.23;
                affineTensor_f6[i].data[1] = 0.5;
                affineTensor_f6[i].data[2] = 0;
                affineTensor_f6[i].data[3] = -0.8;
                affineTensor_f6[i].data[4] = 0.83;
                affineTensor_f6[i].data[5] = 0;
            }

            // Uncomment to run test case with an xywhROI override
            /*for (i = 0; i < images; i++)
            {
                roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
            }*/

            // Uncomment to run test case with an ltrbROI override
            /*for (i = 0; i < images; i++)
            {
                roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
                dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
                dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
            }
            roiTypeSrc = RpptRoiType::LTRB;
            roiTypeDst = RpptRoiType::LTRB;*/

            start = clock();

            if (ip_bitDepth == 0)
                rppt_warp_affine_gpu(d_input, srcDescPtr, d_output, dstDescPtr, affineTensor, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 1)
                rppt_warp_affine_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, affineTensor, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 2)
                rppt_warp_affine_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, affineTensor, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                rppt_warp_affine_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, affineTensor, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            break;
        }
        case 30:
        {
            test_case_name = "non_linear_blend";

            Rpp32f stdDev[images];
            for (i = 0; i < images; i++)
            {
                stdDev[i] = 50.0;
            }

            // Uncomment to run test case with an xywhROI override
            /*for (i = 0; i < images; i++)
            {
                roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
            }*/

            // Uncomment to run test case with an ltrbROI override
            /*for (i = 0; i < images; i++)
            {
                roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
                dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
                dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
            }
            roiTypeSrc = RpptRoiType::LTRB;
            roiTypeDst = RpptRoiType::LTRB;*/

            start = clock();

            if (ip_bitDepth == 0)
                rppt_non_linear_blend_gpu(d_input, d_input_second, srcDescPtr, d_output, dstDescPtr, stdDev, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 1)
                rppt_non_linear_blend_gpu(d_inputf16, d_inputf16_second, srcDescPtr, d_outputf16, dstDescPtr, stdDev, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 2)
                rppt_non_linear_blend_gpu(d_inputf32, d_inputf32_second, srcDescPtr, d_outputf32, dstDescPtr, stdDev, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                rppt_non_linear_blend_gpu(d_inputi8, d_inputi8_second, srcDescPtr, d_outputi8, dstDescPtr, stdDev, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            break;
        }
        case 37:
        {
            test_case_name = "crop";

            // Uncomment to run test case with an xywhROI override
            for (i = 0; i < images; i++)
            {
                roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
            }

            // Uncomment to run test case with an ltrbROI override
            /*for (i = 0; i < images; i++)
            {
                roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
                dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
                dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
            }
            roiTypeSrc = RpptRoiType::LTRB;
            roiTypeDst = RpptRoiType::LTRB;*/

            start = clock();

            if (ip_bitDepth == 0)
                rppt_crop_gpu(d_input, srcDescPtr, d_output, dstDescPtr, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 1)
                rppt_crop_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 2)
                rppt_crop_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                rppt_crop_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            break;
        }
        case 38:
        {
            test_case_name = "crop_mirror_normalize";
            Rpp32f multiplier[images];
            Rpp32f offset[images];
            Rpp32u mirror[images];
            Rpp32f meanParam = 100.0f;
            Rpp32f stdDevParam = 0.9f;
            Rpp32f offsetParam = - meanParam / stdDevParam;
            Rpp32f multiplierParam = 1.0f / stdDevParam;

            for (i = 0; i < images; i++)
            {
                multiplier[i] = multiplierParam;
                offset[i] = offsetParam;
                mirror[i] = 1;
            }

            // Uncomment to run test case with an xywhROI override
            for (i = 0; i < images; i++)
            {
                roiTensorPtrSrc[i].xywhROI.xy.x = 50;
                roiTensorPtrSrc[i].xywhROI.xy.y = 50;
                dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 100;
            }

            // Uncomment to run test case with an ltrbROI override
            /*for (i = 0; i < images; i++)
            {
                roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
                dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
                dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
            }
            roiTypeSrc = RpptRoiType::LTRB;
            roiTypeDst = RpptRoiType::LTRB;*/

            start = clock();

            if (ip_bitDepth == 0)
                rppt_crop_mirror_normalize_gpu(d_input, srcDescPtr, d_output, dstDescPtr, offset, multiplier, mirror, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 1)
                rppt_crop_mirror_normalize_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, offset, multiplier, mirror, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 2)
                rppt_crop_mirror_normalize_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, offset, multiplier, mirror, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 3)
                rppt_crop_mirror_normalize_gpu(d_input, srcDescPtr, d_outputf16, dstDescPtr, offset, multiplier, mirror, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 4)
                rppt_crop_mirror_normalize_gpu(d_input, srcDescPtr, d_outputf32, dstDescPtr, offset, multiplier, mirror, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 5)
                rppt_crop_mirror_normalize_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, offset, multiplier, mirror, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            break;
        }
        case 39:
        {
            test_case_name = "resize_crop_mirror";

            if (interpolationType != RpptInterpolationType::BILINEAR)
            {
                missingFuncFlag = 1;
                break;
            }

            Rpp32u mirror[images];
            for (i = 0; i < images; i++)
            {
                mirror[i] = 1;
            }

            for (i = 0; i < images; i++)
            {
                roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth / 1.1;
                dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight / 3;
                roiTensorPtrSrc[i].xywhROI.roiWidth = 50;
                roiTensorPtrSrc[i].xywhROI.roiHeight = 50;
            }

            // Uncomment to run test case with an ltrbROI override
            /*for (i = 0; i < images; i++)
            {
                roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
                dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
                dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
            }
            roiTypeSrc = RpptRoiType::LTRB;
            roiTypeDst = RpptRoiType::LTRB;*/

            start = clock();
            if (ip_bitDepth == 0)
                rppt_resize_crop_mirror_gpu(d_input, srcDescPtr, d_output, dstDescPtr, dstImgSizes, interpolationType, mirror, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 1)
                rppt_resize_crop_mirror_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, dstImgSizes, interpolationType, mirror, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 2)
                rppt_resize_crop_mirror_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, dstImgSizes, interpolationType, mirror, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                rppt_resize_crop_mirror_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, dstImgSizes, interpolationType, mirror, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            break;
        }
        case 40:
        {
            test_case_name = "erode";

            Rpp32u kernelSize = additionalParam;

            // Uncomment to run test case with an xywhROI override
            /*for (i = 0; i < images; i++)
            {
                roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
            }*/

            // Uncomment to run test case with an ltrbROI override
            /*for (i = 0; i < images; i++)
            {
                roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
                dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
                dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
            }
            roiTypeSrc = RpptRoiType::LTRB;
            roiTypeDst = RpptRoiType::LTRB;*/

            start = clock();

            if (ip_bitDepth == 0)
                rppt_erode_gpu(d_input, srcDescPtr, d_output, dstDescPtr, kernelSize, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 1)
                rppt_erode_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, kernelSize, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 2)
                rppt_erode_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, kernelSize, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                rppt_erode_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, kernelSize, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            break;
        }
        case 41:
        {
            test_case_name = "dilate";

            Rpp32u kernelSize = additionalParam;

            // Uncomment to run test case with an xywhROI override
            /*for (i = 0; i < images; i++)
            {
                roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
            }*/

            // Uncomment to run test case with an ltrbROI override
            /*for (i = 0; i < images; i++)
            {
                roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
                dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
                dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
            }
            roiTypeSrc = RpptRoiType::LTRB;
            roiTypeDst = RpptRoiType::LTRB;*/

            start = clock();

            if (ip_bitDepth == 0)
                rppt_dilate_gpu(d_input, srcDescPtr, d_output, dstDescPtr, kernelSize, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 1)
                rppt_dilate_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, kernelSize, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 2)
                rppt_dilate_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, kernelSize, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                rppt_dilate_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, kernelSize, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            break;
        }
        case 49:
        {
            test_case_name = "box_filter";

            Rpp32u kernelSize = additionalParam;

            // Uncomment to run test case with an xywhROI override
            /*for (i = 0; i < images; i++)
            {
                roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
            }*/

            // Uncomment to run test case with an ltrbROI override
            /*for (i = 0; i < images; i++)
            {
                roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
                dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
                dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
            }
            roiTypeSrc = RpptRoiType::LTRB;
            roiTypeDst = RpptRoiType::LTRB;*/

            start = clock();

            if (ip_bitDepth == 0)
                rppt_box_filter_gpu(d_input, srcDescPtr, d_output, dstDescPtr, kernelSize, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 1)
                rppt_box_filter_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, kernelSize, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 2)
                rppt_box_filter_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, kernelSize, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                rppt_box_filter_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, kernelSize, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            break;
        }
        case 70:
        {
            test_case_name = "copy";

            start = clock();
            if (ip_bitDepth == 0)
                rppt_copy_gpu(d_input, srcDescPtr, d_output, dstDescPtr, handle);
            else if (ip_bitDepth == 1)
                rppt_copy_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, handle);
            else if (ip_bitDepth == 2)
                rppt_copy_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, handle);
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                rppt_copy_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, handle);
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            break;
        }
        case 80:
        {
            test_case_name = "resize_mirror_normalize";

            if (interpolationType != RpptInterpolationType::BILINEAR)
            {
                missingFuncFlag = 1;
                break;
            }

            for (i = 0; i < images; i++)
            {
                dstImgSizes[i].width = roiTensorPtrDst[i].xywhROI.roiWidth = roiTensorPtrSrc[i].xywhROI.roiWidth / 1.1;
                dstImgSizes[i].height = roiTensorPtrDst[i].xywhROI.roiHeight = roiTensorPtrSrc[i].xywhROI.roiHeight / 3;
            }

            Rpp32f mean[images];
            Rpp32f stdDev[images];
            Rpp32u mirror[images];
            for (i = 0; i < images; i++)
            {
                mean[i] = 100.0;
                stdDev[i] = 1.0;
                mirror[i] = 1;
            }

            // Uncomment to run test case with an xywhROI override
            // for (i = 0; i < images; i++)
            // {
            //     roiTensorPtrSrc[i].xywhROI.xy.x = 0;
            //     roiTensorPtrSrc[i].xywhROI.xy.y = 0;
            //     dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
            //     dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
            // }

            // Uncomment to run test case with an ltrbROI override
            /*for (i = 0; i < images; i++)
            {
                roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
                dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
                dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
            }
            roiTypeSrc = RpptRoiType::LTRB;
            roiTypeDst = RpptRoiType::LTRB;*/

            start = clock();
            if (ip_bitDepth == 0)
                rppt_resize_mirror_normalize_gpu(d_input, srcDescPtr, d_output, dstDescPtr, dstImgSizes, interpolationType, mean, stdDev, mirror, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 1)
                rppt_resize_mirror_normalize_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, dstImgSizes, interpolationType, mean, stdDev, mirror, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 2)
                rppt_resize_mirror_normalize_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, dstImgSizes, interpolationType, mean, stdDev, mirror, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 3)
                rppt_resize_mirror_normalize_gpu(d_input, srcDescPtr, d_outputf16, dstDescPtr, dstImgSizes, interpolationType, mean, stdDev, mirror, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 4)
                rppt_resize_mirror_normalize_gpu(d_input, srcDescPtr, d_outputf32, dstDescPtr, dstImgSizes, interpolationType, mean, stdDev, mirror, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 5)
                rppt_resize_mirror_normalize_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, dstImgSizes, interpolationType, mean, stdDev, mirror, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            break;
        }
        case 83:
        {
            test_case_name = "gridmask";

            Rpp32u tileWidth = 40;
            Rpp32f gridRatio = 0.6;
            Rpp32f gridAngle = 0.5;
            RpptUintVector2D translateVector;
            translateVector.x = 0.0;
            translateVector.y = 0.0;

            // Uncomment to run test case with an xywhROI override
            /*for (i = 0; i < images; i++)
            {
                roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
            }*/

            // Uncomment to run test case with an ltrbROI override
            /*for (i = 0; i < images; i++)
            {
                roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
                dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
                dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
            }
            roiTypeSrc = RpptRoiType::LTRB;
            roiTypeDst = RpptRoiType::LTRB;*/


            start = clock();

            if (ip_bitDepth == 0)
                rppt_gridmask_gpu(d_input, srcDescPtr, d_output, dstDescPtr, tileWidth, gridRatio, gridAngle, translateVector, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 1)
                rppt_gridmask_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, tileWidth, gridRatio, gridAngle, translateVector, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 2)
                rppt_gridmask_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, tileWidth, gridRatio, gridAngle, translateVector, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                rppt_gridmask_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, tileWidth, gridRatio, gridAngle, translateVector, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            break;
        }
        case 84:
        {
            test_case_name = "spatter";

            RpptRGB spatterColor;

            // Mud Spatter
            spatterColor.R = 65;
            spatterColor.G = 50;
            spatterColor.B = 23;

            // Blood Spatter
            // spatterColor.R = 98;
            // spatterColor.G = 3;
            // spatterColor.B = 3;

            // Ink Spatter
            // spatterColor.R = 5;
            // spatterColor.G = 20;
            // spatterColor.B = 64;

            // Uncomment to run test case with an xywhROI override
            /*for (i = 0; i < images; i++)
            {
                roiTensorPtrSrc[i].xywhROI.xy.x = 0;
                roiTensorPtrSrc[i].xywhROI.xy.y = 0;
                dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth = 100;
                dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight = 180;
            }*/

            // Uncomment to run test case with an ltrbROI override
            /*for (i = 0; i < images; i++)
            {
                roiTensorPtrSrc[i].ltrbROI.lt.x = 50;
                roiTensorPtrSrc[i].ltrbROI.lt.y = 30;
                roiTensorPtrSrc[i].ltrbROI.rb.x = 210;
                roiTensorPtrSrc[i].ltrbROI.rb.y = 210;
                dstImgSizes[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
                dstImgSizes[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
            }
            roiTypeSrc = RpptRoiType::LTRB;
            roiTypeDst = RpptRoiType::LTRB;*/


            start = clock();

            if (ip_bitDepth == 0)
                rppt_spatter_gpu(d_input, srcDescPtr, d_output, dstDescPtr, spatterColor, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 1)
                rppt_spatter_gpu(d_inputf16, srcDescPtr, d_outputf16, dstDescPtr, spatterColor, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 2)
                rppt_spatter_gpu(d_inputf32, srcDescPtr, d_outputf32, dstDescPtr, spatterColor, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 3)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 4)
                missingFuncFlag = 1;
            else if (ip_bitDepth == 5)
                rppt_spatter_gpu(d_inputi8, srcDescPtr, d_outputi8, dstDescPtr, spatterColor, roiTensorPtrSrc, roiTypeSrc, handle);
            else if (ip_bitDepth == 6)
                missingFuncFlag = 1;
            else
                missingFuncFlag = 1;

            break;
        }
        default:
            missingFuncFlag = 1;
            break;
        }

        hipDeviceSynchronize();
        end = clock();

        if (missingFuncFlag == 1)
        {
            printf("\nThe functionality %s doesn't yet exist in RPP\n", func);
            return -1;
        }

        // Display measured times

        gpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        if (gpu_time_used > max_time_used)
            max_time_used = gpu_time_used;
        if (gpu_time_used < min_time_used)
            min_time_used = gpu_time_used;
        avg_time_used += gpu_time_used;
    }

    avg_time_used /= 100;
    cout << fixed << "\nmax,min,avg = " << max_time_used << "," << min_time_used << "," << avg_time_used << endl;

    rppDestroy(handle, backend);

    // Free memory

    hipHostFree(roiTensorPtrSrc);
    hipHostFree(roiTensorPtrDst);
    hipHostFree(srcImgSizes);
    hipHostFree(dstImgSizes);
    free(input);
    free(input_second);
    free(output);

    if (ip_bitDepth == 0)
    {
        hipFree(d_input);
        hipFree(d_input_second);
        hipFree(d_output);
    }
    else if (ip_bitDepth == 1)
    {
        free(inputf16);
        free(inputf16_second);
        free(outputf16);
        hipFree(d_inputf16);
        hipFree(d_inputf16_second);
        hipFree(d_outputf16);
    }
    else if (ip_bitDepth == 2)
    {
        free(inputf32);
        free(inputf32_second);
        free(outputf32);
        hipFree(d_inputf32);
        hipFree(d_inputf32_second);
        hipFree(d_outputf32);
    }
    else if (ip_bitDepth == 3)
    {
        free(outputf16);
        hipFree(d_input);
        hipFree(d_input_second);
        hipFree(d_outputf16);
    }
    else if (ip_bitDepth == 4)
    {
        free(outputf32);
        hipFree(d_input);
        hipFree(d_input_second);
        hipFree(d_outputf32);
    }
    else if (ip_bitDepth == 5)
    {
        free(inputi8);
        free(inputi8_second);
        free(outputi8);
        hipFree(d_inputi8);
        hipFree(d_inputi8_second);
        hipFree(d_outputi8);
    }
    else if (ip_bitDepth == 6)
    {
        free(outputi8);
        hipFree(d_input);
        hipFree(d_input_second);
        hipFree(d_outputi8);
    }

    return 0;
}