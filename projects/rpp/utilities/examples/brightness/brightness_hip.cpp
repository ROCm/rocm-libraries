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
#include "brightness_hip.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>
#include <hip/hip_fp16.h>
#include <fstream>

using namespace cv;
using namespace std;

inline size_t get_size_of_data_type(RpptDataType dataType)
{
    if(dataType == RpptDataType::U8)
        return sizeof(Rpp8u);
    else if(dataType == RpptDataType::I8)
        return sizeof(Rpp8s);
    else if(dataType == RpptDataType::F16)
        return sizeof(Rpp16f);
    else if(dataType == RpptDataType::F32)
        return sizeof(Rpp32f);
    else
        return 0;
}

int main(int argc, char **argv)
{
    // Handle inputs
    const int MIN_ARG_COUNT = 8;

    char *src = argv[1];
    char *srcSecond = argv[2];
    string dst = argv[3];
    int inputBitDepth = atoi(argv[4]);
    unsigned int outputFormatToggle = atoi(argv[5]);
    int layoutType = atoi(argv[6]); // 0 for pkd3 / 1 for pln3 / 2 for pln1
    int decoderType = atoi(argv[7]);

    if (argc < MIN_ARG_COUNT)
    {
        printf("\nImproper Usage! Needs all arguments!\n");
        printf("\nUsage: <src1 folder> <src2 folder (place same as src1 folder for single image functionalities)> <dst folder> \
                <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> \
                <outputFormatToggle (pkd->pkd = 0 / pkd->pln = 1)> <layout type (pkd3 - 0/pln3 - 1/pln1 - 2)> \
                <decoder type (TurboJPEG - 0/ OpenCV - 1)>\n");
        return -1;
    }

    if (layoutType == 2)
    {
        if (outputFormatToggle != 0)
        {
            printf("\nPLN1 cases don't have outputFormatToggle! Please input outputFormatToggle = 0\n");
            return -1;
        }
    }

    // Determine the number of input channels based on the specified layout type
    int inputChannels = set_input_channels(layoutType);

    // TODO; Check if really needed, Determine the type of function to be used based on the specified layout type
    string funcType = set_function_type(layoutType, outputFormatToggle);

    // Initialize tensor descriptors
    RpptDesc srcDesc, dstDesc;
    RpptDescPtr srcDescPtr = &srcDesc;
    RpptDescPtr dstDescPtr = &dstDesc;

    // Set src/dst layout types in tensor descriptors
    set_descriptor_layout( srcDescPtr, dstDescPtr, layoutType, outputFormatToggle);

    // Set src/dst data types in tensor descriptors
    set_descriptor_data_type(inputBitDepth, "brightness", srcDescPtr, dstDescPtr);

    // Other initializations
    int i = 0;
    int maxHeight = 0, maxWidth = 0;
    int maxDstHeight = 0, maxDstWidth = 0;
    Rpp64u count = 0;
    Rpp64u ioBufferSize = 0;
    Rpp64u oBufferSize = 0;
    static int noOfImages = 0;
    Mat image, imageSecond;

    // String ops on input path
    string inputPath = src;
    inputPath += "/";
    string inputPathSecond = srcSecond;
    inputPathSecond += "/";

    string func = "brightness" + funcType;

    // Get number of images and image Names
    struct dirent *de;
    DIR *dr = opendir(src);
    vector<string> imageNames;
    while ((de = readdir(dr)) != NULL)
    {
        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
            continue;
        noOfImages += 1;
        imageNames.push_back(de->d_name);
    }
    closedir(dr);
    if(!noOfImages)
    {
        std::cerr<<"Not able to find any images in the folder specified. Please check the input path";
        exit(0);
    }

    // Initialize ROI tensors for src/dst
    RpptROI *roiTensorPtrSrc, *roiTensorPtrDst;
    hipHostMalloc(&roiTensorPtrSrc, noOfImages * sizeof(RpptROI));
    hipHostMalloc(&roiTensorPtrDst, noOfImages * sizeof(RpptROI));

    // Initialize the ImagePatch for dst
    RpptImagePatch *dstImgSizes;
    hipHostMalloc(&dstImgSizes, noOfImages * sizeof(RpptImagePatch));

    // Set ROI tensors types for src/dst
    RpptRoiType roiTypeSrc, roiTypeDst;
    roiTypeSrc = RpptRoiType::XYWH;
    roiTypeDst = RpptRoiType::XYWH;

    // Set maxHeight, maxWidth and ROIs for src/dst
    const int images = noOfImages;

    for(int i = 0; i < imageNames.size(); i++)
    {
        string temp = inputPath;
        temp += imageNames[i];
        if (layoutType == 0 || layoutType == 1)
            image = imread(temp, 1);
        else
            image = imread(temp, 0);

        roiTensorPtrSrc[i].xywhROI = {0, 0, image.cols, image.rows};
        roiTensorPtrDst[i].xywhROI = {0, 0, image.cols, image.rows};
        dstImgSizes[i].width = roiTensorPtrDst[i].xywhROI.roiWidth;
        dstImgSizes[i].height = roiTensorPtrDst[i].xywhROI.roiHeight;

        maxHeight = std::max(maxHeight, roiTensorPtrSrc[i].xywhROI.roiHeight);
        maxWidth = std::max(maxWidth, roiTensorPtrSrc[i].xywhROI.roiWidth);
        maxDstHeight = std::max(maxDstHeight, roiTensorPtrDst[i].xywhROI.roiHeight);
        maxDstWidth = std::max(maxDstWidth, roiTensorPtrDst[i].xywhROI.roiWidth);

        count++;
    }

     // Check if any of maxWidth and maxHeight is less than or equal to 0
    if(maxHeight <= 0 || maxWidth <= 0)
    {
        std::cerr<<"Unable to read images properly.Please check the input path of the files specified";
        exit(0);
    }

    Rpp32u outputChannels = inputChannels;
    Rpp32u offsetInBytes = 0;

    // Set numDims, offset, n/c/h/w values, strides for src/dst
    set_descriptor_dims_and_strides(srcDescPtr, noOfImages, maxHeight, maxWidth, inputChannels, offsetInBytes);
    set_descriptor_dims_and_strides(dstDescPtr, noOfImages, maxDstHeight, maxDstWidth, outputChannels, offsetInBytes);

    // Set buffer sizes in pixels for src/dst
    ioBufferSize = (Rpp64u)srcDescPtr->h * (Rpp64u)srcDescPtr->w * (Rpp64u)srcDescPtr->c * (Rpp64u)noOfImages;
    oBufferSize = (Rpp64u)dstDescPtr->h * (Rpp64u)dstDescPtr->w * (Rpp64u)dstDescPtr->c * (Rpp64u)noOfImages;

    // Set buffer sizes in bytes for src/dst (including offsets)
    Rpp64u ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
    Rpp64u oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;
    Rpp64u inputBufferSize = ioBufferSize * get_size_of_data_type(srcDescPtr->dataType) + srcDescPtr->offsetInBytes;
    Rpp64u outputBufferSize = oBufferSize * get_size_of_data_type(dstDescPtr->dataType) + dstDescPtr->offsetInBytes;

    // Initialize 8u host buffers for src/dst
    Rpp8u *inputu8 = static_cast<Rpp8u *>(calloc(ioBufferSizeInBytes_u8, 1));
    Rpp8u *inputu8Second = static_cast<Rpp8u *>(calloc(ioBufferSizeInBytes_u8, 1));
    Rpp8u *outputu8 = static_cast<Rpp8u *>(calloc(oBufferSizeInBytes_u8, 1));

    string imageNamesPath[images];
    string imageNamesPathSecond[images];
    for(int i = 0; i < images; i++)
    {
        imageNamesPath[i] = inputPath + "/" + imageNames[i];
        imageNamesPathSecond[i] = inputPathSecond + "/" + imageNames[i];
    }

    // Read images
    if(decoderType == 0)
    {
        read_image_batch_turbojpeg(inputu8, srcDescPtr, imageNamesPath);
        read_image_batch_turbojpeg(inputu8Second, srcDescPtr, imageNamesPathSecond);
        printf("OK: decoded images using TurboJpeg\n");
    }
    else
    {
        read_image_batch_opencv(inputu8, srcDescPtr, imageNamesPath);
        read_image_batch_opencv(inputu8Second, srcDescPtr, imageNamesPathSecond);
        printf("OK: decoded images using OpenCV\n");
    }

    // if the input layout requested is PLN3, convert PKD3 inputs to PLN3 for first and second input batch
    if (layoutType == 1)
    {
        convert_pkd3_to_pln3(inputu8, srcDescPtr);
        convert_pkd3_to_pln3(inputu8Second, srcDescPtr);
    }

    // Factors to convert U8 data to F32, F16 data to 0-1 range and reconvert them back to 0 -255 range
    Rpp32f conversionFactor = 1.0f / 255.0;
    Rpp32f invConversionFactor = 1.0f / conversionFactor;

    void *input, *input_second, *output;
    void *d_input, *d_input_second, *d_output;
    input = static_cast<Rpp8u *>(calloc(inputBufferSize, 1));
    input_second = static_cast<Rpp8u *>(calloc(inputBufferSize, 1));
    output = static_cast<Rpp8u *>(calloc(outputBufferSize, 1));

    // Convert inputs to correponding bit depth specified by user
    if (inputBitDepth == 0)
    {
        memcpy(input, inputu8, inputBufferSize);
        memcpy(input_second, inputu8Second, inputBufferSize);
    }
    else if (inputBitDepth == 1)
    {
        Rpp8u *inputTemp, *inputSecondTemp;
        Rpp16f *inputf16Temp, *inputf16SecondTemp;
        inputTemp = inputu8 + srcDescPtr->offsetInBytes;
        inputSecondTemp = inputu8Second + srcDescPtr->offsetInBytes;
        inputf16Temp = reinterpret_cast<Rpp16f *>(static_cast<Rpp8u *>(input) + srcDescPtr->offsetInBytes);
        inputf16SecondTemp = reinterpret_cast<Rpp16f *>(static_cast<Rpp8u *>(input_second) + srcDescPtr->offsetInBytes);

        for (int i = 0; i < ioBufferSize; i++)
        {
            *inputf16Temp++ = static_cast<Rpp16f>((static_cast<float>(*inputTemp++)) * conversionFactor);
            *inputf16SecondTemp++ = static_cast<Rpp16f>((static_cast<float>(*inputSecondTemp++)) * conversionFactor);
        }
    }
    else if (inputBitDepth == 2)
    {
        Rpp8u *inputTemp, *inputSecondTemp;
        Rpp32f *inputf32Temp, *inputf32SecondTemp;
        inputTemp = inputu8 + srcDescPtr->offsetInBytes;
        inputSecondTemp = inputu8Second + srcDescPtr->offsetInBytes;
        inputf32Temp = reinterpret_cast<Rpp32f *>(static_cast<Rpp8u *>(input) + srcDescPtr->offsetInBytes);
        inputf32SecondTemp = reinterpret_cast<Rpp32f *>(static_cast<Rpp8u *>(input_second) + srcDescPtr->offsetInBytes);

        for (int i = 0; i < ioBufferSize; i++)
        {
            *inputf32Temp++ = (static_cast<Rpp32f>(*inputTemp++)) * conversionFactor;
            *inputf32SecondTemp++ = (static_cast<Rpp32f>(*inputSecondTemp++)) * conversionFactor;
        }
    }
    else if (inputBitDepth == 5)
    {
        Rpp8u *inputTemp, *inputSecondTemp;
        Rpp8s *inputi8Temp, *inputi8SecondTemp;

        inputTemp = inputu8 + srcDescPtr->offsetInBytes;
        inputSecondTemp = inputu8Second + srcDescPtr->offsetInBytes;
        inputi8Temp = static_cast<Rpp8s *>(input) + srcDescPtr->offsetInBytes;
        inputi8SecondTemp = static_cast<Rpp8s *>(input_second) + srcDescPtr->offsetInBytes;

        for (int i = 0; i < ioBufferSize; i++)
        {
            *inputi8Temp++ = static_cast<Rpp8s>((static_cast<Rpp32s>(*inputTemp++)) - 128);
            *inputi8SecondTemp++ = static_cast<Rpp8s>((static_cast<Rpp32s>(*inputSecondTemp++)) - 128);
        }
    }

    // Allocate hip memory for src/dst and copy decoded inputs to hip buffers
    hipMalloc(&d_input, inputBufferSize);
    hipMalloc(&d_input_second, inputBufferSize);
    hipMalloc(&d_output, outputBufferSize);
    hipMemcpy(d_input, input, inputBufferSize, hipMemcpyHostToDevice);
    hipMemcpy(d_input_second, input_second, inputBufferSize, hipMemcpyHostToDevice);
    hipMemcpy(d_output, output, outputBufferSize, hipMemcpyHostToDevice);

    // Run RPP API
    rppHandle_t handle;
    hipStream_t stream;
    hipStreamCreate(&stream);
    RppBackend backend = RppBackend::RPP_HIP_BACKEND;
    rppCreate(&handle, noOfImages, 0, stream, backend);

    //parameters for brightness node
    Rpp32f alpha[images];
    Rpp32f beta[images];
    for (i = 0; i < images; i++)
    {
        alpha[i] = 1.75;
        beta[i] = 50;
    }

    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5) {
        rppt_brightness_gpu(d_input, srcDescPtr, d_output, dstDescPtr, alpha, beta, roiTensorPtrSrc, roiTypeSrc, handle);
        printf("OK: executed brightness kernel on HIP backend\n");
    }
    else {
        printf("\nThe functionality %s doesn't yet exist in RPP\n", func.c_str());
        return -1;
    }

    hipDeviceSynchronize();

    // Reconvert other bit depths to 8u for output display purposes
    if (inputBitDepth == 0)
    {
        hipMemcpy(output, d_output, outputBufferSize, hipMemcpyDeviceToHost);
        memcpy(outputu8, output, outputBufferSize);
    }
    else if ((inputBitDepth == 1) || (inputBitDepth == 3))
    {
        hipMemcpy(output, d_output, outputBufferSize, hipMemcpyDeviceToHost);
        Rpp8u *outputTemp;
        outputTemp = outputu8 + dstDescPtr->offsetInBytes;
        Rpp16f *outputf16Temp;
        outputf16Temp = reinterpret_cast<Rpp16f *>(static_cast<Rpp8u *>(output) + dstDescPtr->offsetInBytes);
        for (int i = 0; i < oBufferSize; i++)
        {
            *outputTemp = static_cast<Rpp8u>(validate_pixel_range(static_cast<float>(*outputf16Temp) * invConversionFactor));
            outputf16Temp++;
            outputTemp++;
        }
    }
    else if ((inputBitDepth == 2) || (inputBitDepth == 4))
    {
        hipMemcpy(output, d_output, outputBufferSize, hipMemcpyDeviceToHost);
        Rpp8u *outputTemp;
        outputTemp = outputu8 + dstDescPtr->offsetInBytes;
        Rpp32f *outputf32Temp;
        outputf32Temp = reinterpret_cast<Rpp32f *>(static_cast<Rpp8u *>(output) + dstDescPtr->offsetInBytes);
        for (int i = 0; i < oBufferSize; i++)
        {
            *outputTemp = static_cast<Rpp8u>(validate_pixel_range(*outputf32Temp * invConversionFactor));
            outputf32Temp++;
            outputTemp++;
        }
    }
    else if ((inputBitDepth == 5) || (inputBitDepth == 6))
    {
        hipMemcpy(output, d_output, outputBufferSize, hipMemcpyDeviceToHost);
        Rpp8u *outputTemp = outputu8 + dstDescPtr->offsetInBytes;
        Rpp8s *outputi8Temp = static_cast<Rpp8s *>(output) + dstDescPtr->offsetInBytes;
        for (int i = 0; i < oBufferSize; i++)
        {
            *outputTemp = static_cast<Rpp8u>(validate_pixel_range((static_cast<Rpp32s>(*outputi8Temp)) + 128));
            outputi8Temp++;
            outputTemp++;
        }
    }

    // Calculate exact dstROI in XYWH format for OpenCV dump
    if (roiTypeSrc == RpptRoiType::LTRB)
        convert_roi(roiTensorPtrDst, RpptRoiType::XYWH, dstDescPtr->n);

    // Check if the ROI values for each input is within the bounds of the max buffer allocated
    RpptROI roiDefault;
    RpptROIPtr roiPtrDefault = &roiDefault;
    roiPtrDefault->xywhROI =  {0, 0, static_cast<Rpp32s>(dstDescPtr->w), static_cast<Rpp32s>(dstDescPtr->h)};
    for (int i = 0; i < dstDescPtr->n; i++)
    {
        roiTensorPtrDst[i].xywhROI.roiWidth = std::min(roiPtrDefault->xywhROI.roiWidth - roiTensorPtrDst[i].xywhROI.xy.x, roiTensorPtrDst[i].xywhROI.roiWidth);
        roiTensorPtrDst[i].xywhROI.roiHeight = std::min(roiPtrDefault->xywhROI.roiHeight - roiTensorPtrDst[i].xywhROI.xy.y, roiTensorPtrDst[i].xywhROI.roiHeight);
        roiTensorPtrDst[i].xywhROI.xy.x = std::max(roiPtrDefault->xywhROI.xy.x, roiTensorPtrDst[i].xywhROI.xy.x);
        roiTensorPtrDst[i].xywhROI.xy.y = std::max(roiPtrDefault->xywhROI.xy.y, roiTensorPtrDst[i].xywhROI.xy.y);
    }

    // Convert any PLN3 outputs to the corresponding PKD3 version for OpenCV dump
    if (layoutType == 0 || layoutType == 1)
    {
        if ((dstDescPtr->c == 3) && (dstDescPtr->layout == RpptLayout::NCHW))
            convert_pln3_to_pkd3(outputu8, dstDescPtr);
    }
    rppDestroy(handle, backend);

    // OpenCV dump (if testType is unit test and QA mode is not set)
    write_image_batch_opencv(dst, outputu8, dstDescPtr, imageNames, dstImgSizes);

    // Free memory
    hipHostFree(roiTensorPtrSrc);
    hipHostFree(roiTensorPtrDst);
    hipHostFree(dstImgSizes);
    free(input);
    free(input_second);
    free(output);
    free(inputu8);
    free(inputu8Second);
    free(outputu8);
    hipFree(d_input);
    hipFree(d_input_second);
    hipFree(d_output);
    printf("OK: free memory\n");
    printf("OK: successful\n");
    return 0;
}
