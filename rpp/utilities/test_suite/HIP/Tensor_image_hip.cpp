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
#include "../rpp_test_suite_image.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>
#include <fstream>

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
    // Handle inputs
    const int MIN_ARG_COUNT = 19;

    char *src = argv[1];
    char *srcSecond = argv[2];
    string dst = argv[3];

    int inputBitDepth = atoi(argv[4]);
    unsigned int outputFormatToggle = atoi(argv[5]);
    int testCase = atoi(argv[6]);
    int numRuns = atoi(argv[8]);
    int testType = atoi(argv[9]);     // 0 for unit and 1 for performance test
    int layoutType = atoi(argv[10]); // 0 for pkd3 / 1 for pln3 / 2 for pln1
    int qaFlag = atoi(argv[12]);
    int decoderType = atoi(argv[13]);
    int batchSize = atoi(argv[14]);

    bool additionalParamCase = (additionalParamCases.find(testCase) != additionalParamCases.end());
    bool kernelSizeCase = (kernelSizeCases.find(testCase) != kernelSizeCases.end());
    bool dualInputCase = (dualInputCases.find(testCase) != dualInputCases.end());
    bool randomOutputCase = (randomOutputCases.find(testCase) != randomOutputCases.end());
    bool nonQACase = (nonQACases.find(testCase) != nonQACases.end());
    bool interpolationTypeCase = (interpolationTypeCases.find(testCase) != interpolationTypeCases.end());
    bool reductionTypeCase = (reductionTypeCases.find(testCase) != reductionTypeCases.end());
    bool noiseTypeCase = (noiseTypeCases.find(testCase) != noiseTypeCases.end());
    bool pln1OutTypeCase = (pln1OutTypeCases.find(testCase) != pln1OutTypeCases.end());

    unsigned int verbosity = atoi(argv[11]);
    unsigned int additionalParam = additionalParamCase ? atoi(argv[7]) : 1;
    int roiList[4] = {atoi(argv[15]), atoi(argv[16]), atoi(argv[17]), atoi(argv[18])};
    string scriptPath = argv[19];

    if (verbosity == 1)
    {
        cout << "\nInputs for this test case are:";
        cout << "\nsrc1 = " << argv[1];
        cout << "\nsrc2 = " << argv[2];
        if (testType == 0)
            cout << "\ndst = " << argv[3];
        cout << "\nu8 / f16 / f32 / u8->f16 / u8->f32 / i8 / u8->i8 (0/1/2/3/4/5/6) = " << argv[4];
        cout << "\noutputFormatToggle (pkd->pkd = 0 / pkd->pln = 1) = " << argv[5];
        cout << "\ncase number (0:91) = " << argv[6];
        cout << "\nnumber of times to run = " << argv[8];
        cout << "\ntest type - (0 = unit tests / 1 = performance tests) = " << argv[9];
        cout << "\nlayout type - (0 = PKD3/ 1 = PLN3/ 2 = PLN1) = " << argv[10];
        cout << "\nqa mode - 0/1 = " << argv[12];
        cout << "\ndecoder type - (0 = TurboJPEG / 1 = OpenCV) = " << argv[13];
        cout << "\nbatch size = " << argv[14];
    }

    if (argc < MIN_ARG_COUNT)
    {
        cout << "\nImproper Usage! Needs all arguments!\n";
        cout << "\nUsage: <src1 folder> <src2 folder (place same as src1 folder for single image functionalities)> <dst folder> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <outputFormatToggle (pkd->pkd = 0 / pkd->pln = 1)> <case number = 0:87> <number of runs > 0> <layout type (0 = PKD3/ 1 = PLN3/ 2 = PLN1)> <qa mode (0/1)> <decoder type (0/1)> <batch size > 1> <roiList> <verbosity = 0/1>>\n";
        return -1;
    }

    if (layoutType == 2)
    {
        if(testCase == COLOR_TWIST || testCase == COLOR_CAST || testCase == GLITCH || testCase == COLOR_TEMPERATURE || testCase == COLOR_TO_GREYSCALE)
        {
            cout << "\ncase " << testCase << " does not exist for PLN1 layout\n";
            return -1;
        }
        else if (outputFormatToggle != 0)
        {
            cout << "\nPLN1 cases don't have outputFormatToggle! Please input outputFormatToggle = 0\n";
            return -1;
        }
    }

    if(pln1OutTypeCase && outputFormatToggle != 0)
    {
        cout << "\ntest case " << testCase << " don't have outputFormatToggle! Please input outputFormatToggle = 0\n";
        return -1;
    }
    else if (reductionTypeCase && outputFormatToggle != 0)
    {
        cout << "\nReduction Kernels don't have outputFormatToggle! Please input outputFormatToggle = 0\n";
        return -1;
    }
    else if(batchSize > MAX_BATCH_SIZE)
    {
        std::cerr << "\n Batchsize should be less than or equal to "<< MAX_BATCH_SIZE << " Aborting!";
        exit(0);
    }
    else if(testCase == RICAP && batchSize < 2)
    {
        std::cerr<<"\n RICAP only works with BatchSize > 1";
        exit(0);
    }

    // Get function name
    string funcName = augmentationMap[testCase];
    if (funcName.empty())
    {
        if (testType == 0)
            cout << "\ncase " << testCase << " is not supported\n";

        return -1;
    }

    // Determine the number of input channels based on the specified layout type
    int inputChannels = set_input_channels(layoutType);

    // Determine the type of function to be used based on the specified layout type
    string funcType = set_function_type(layoutType, pln1OutTypeCase, outputFormatToggle, "HIP");

    // Initialize tensor descriptors
    RpptDesc srcDesc, dstDesc;
    RpptDescPtr srcDescPtr = &srcDesc;
    RpptDescPtr dstDescPtr = &dstDesc;

    // Set src/dst layout types in tensor descriptors
    set_descriptor_layout(srcDescPtr, dstDescPtr, layoutType, pln1OutTypeCase, outputFormatToggle);

    // Set src/dst data types in tensor descriptors
    set_descriptor_data_type(inputBitDepth, funcName, srcDescPtr, dstDescPtr);

    // Other initializations
    int missingFuncFlag = 0;
    int i = 0, j = 0;
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

    string func = funcName;
    func += funcType;

    RpptInterpolationType interpolationType = RpptInterpolationType::BILINEAR;
    std::string interpolationTypeName = "";
    std::string noiseTypeName = "";
    if (kernelSizeCase)
    {
        char additionalParam_char[2];
        std::snprintf(additionalParam_char, sizeof(additionalParam_char), "%u", additionalParam);
        func += "_kernelSize";
        func += additionalParam_char;
    }
    else if (interpolationTypeCase)
    {
        interpolationTypeName = get_interpolation_type(additionalParam, interpolationType);
        func += "_interpolationType";
        func += interpolationTypeName.c_str();
    }
    else if (noiseTypeCase)
    {
        noiseTypeName = get_noise_type(additionalParam);
        func += "_noiseType";
        func += noiseTypeName.c_str();
    }

    if(!qaFlag)
    {
        dst += "/";
        dst += func;
    }

    // Get number of images and image Names
    vector<string> imageNames, imageNamesSecond, imageNamesPath, imageNamesPathSecond;
    search_files_recursive(src, imageNames, imageNamesPath, ".jpg");
    if(dualInputCase)
    {
        search_files_recursive(srcSecond, imageNamesSecond, imageNamesPathSecond, ".jpg");
        if(imageNames.size() != imageNamesSecond.size())
        {
            std::cerr << " \n The number of images in the input folders must be the same.";
            exit(0);
        }
    }
    noOfImages = imageNames.size();

    if(noOfImages < batchSize || ((noOfImages % batchSize) != 0))
    {
        replicate_last_file_to_fill_batch(imageNamesPath[noOfImages - 1], imageNamesPath, imageNames, imageNames[noOfImages - 1], noOfImages, batchSize);
        if(dualInputCase)
            replicate_last_file_to_fill_batch(imageNamesPathSecond[noOfImages - 1], imageNamesPathSecond, imageNamesSecond, imageNamesSecond[noOfImages - 1], noOfImages, batchSize);
        noOfImages = imageNames.size();
    }

    if(!noOfImages)
    {
        std::cerr << "Not able to find any images in the folder specified. Please check the input path";
        exit(0);
    }

    if(qaFlag)
    {
        sort(imageNames.begin(), imageNames.end());
        if(dualInputCase)
            sort(imageNamesSecond.begin(), imageNamesSecond.end());
    }

    // Initialize ROI tensors for src/dst
    RpptROI *roiTensorPtrSrc, *roiTensorPtrDst;
    CHECK_RETURN_STATUS(hipHostMalloc(&roiTensorPtrSrc, batchSize * sizeof(RpptROI)));
    CHECK_RETURN_STATUS(hipHostMalloc(&roiTensorPtrDst, batchSize * sizeof(RpptROI)));

    // Initialize the ImagePatch for dst
    RpptImagePatch *dstImgSizes;
    CHECK_RETURN_STATUS(hipHostMalloc(&dstImgSizes, batchSize * sizeof(RpptImagePatch)));

    // Set ROI tensors types for src/dst
    RpptRoiType roiTypeSrc, roiTypeDst;
    roiTypeSrc = RpptRoiType::XYWH;
    roiTypeDst = RpptRoiType::XYWH;

    Rpp32u outputChannels = inputChannels;
    if(pln1OutTypeCase)
        outputChannels = 1;
    Rpp32u srcOffsetInBytes = (kernelSizeCase) ? (12 * (additionalParam / 2)) : 0;
    Rpp32u dstOffsetInBytes = 0;
    int imagesMixed = 0; // Flag used to check if all images in dataset is of same dimensions

    set_max_dimensions(imageNamesPath, maxHeight, maxWidth, imagesMixed);
    if(testCase == RICAP && imagesMixed)
    {
        std::cerr<<"\n RICAP only works with same dimension images";
        exit(0);
    }

    Rpp32s additionalStride = 0;
    if (kernelSizeCase)
        additionalStride = additionalParam / 2;

    // Set numDims, offset, n/c/h/w values, strides for src/dst
    set_descriptor_dims_and_strides(srcDescPtr, batchSize, maxHeight, maxWidth, inputChannels, srcOffsetInBytes, additionalStride);
    set_descriptor_dims_and_strides(dstDescPtr, batchSize, maxHeight, maxWidth, outputChannels, dstOffsetInBytes);

    // Factors to convert U8 data to F32, F16 data to 0-1 range and reconvert them back to 0 -255 range
    Rpp32f conversionFactor = 1.0f / 255.0;
    if(testCase == CROP_MIRROR_NORMALIZE)
        conversionFactor = 1.0;
    Rpp32f invConversionFactor = 1.0f / conversionFactor;

    // Set buffer sizes in pixels for src/dst
    ioBufferSize = (Rpp64u)srcDescPtr->h * (Rpp64u)srcDescPtr->w * (Rpp64u)srcDescPtr->c * (Rpp64u)batchSize;
    oBufferSize = (Rpp64u)dstDescPtr->h * (Rpp64u)dstDescPtr->w * (Rpp64u)dstDescPtr->c * (Rpp64u)batchSize;

    // Set buffer sizes in bytes for src/dst (including offsets)
    Rpp64u ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
    Rpp64u oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;
    Rpp64u inputBufferSize = ioBufferSize * get_size_of_data_type(srcDescPtr->dataType) + srcDescPtr->offsetInBytes;
    Rpp64u outputBufferSize = oBufferSize * get_size_of_data_type(dstDescPtr->dataType) + dstDescPtr->offsetInBytes;

    // Initialize 8u host buffers for src/dst
    Rpp8u *inputu8 = static_cast<Rpp8u *>(calloc(ioBufferSizeInBytes_u8, 1));
    Rpp8u *inputu8Second = static_cast<Rpp8u *>(calloc(ioBufferSizeInBytes_u8, 1));
    Rpp8u *outputu8 = static_cast<Rpp8u *>(calloc(oBufferSizeInBytes_u8, 1));

    Rpp8u *offsettedInput, *offsettedInputSecond;
    offsettedInput = inputu8 + srcDescPtr->offsetInBytes;
    offsettedInputSecond = inputu8Second + srcDescPtr->offsetInBytes;
    void *input, *input_second, *output;
    void *d_input, *d_input_second, *d_output;

    input = static_cast<Rpp8u *>(calloc(inputBufferSize, 1));
    input_second = static_cast<Rpp8u *>(calloc(inputBufferSize, 1));
    output = static_cast<Rpp8u *>(calloc(outputBufferSize, 1));

    Rpp32f *rowRemapTable, *colRemapTable;
    if(testCase == REMAP)
    {
        rowRemapTable = static_cast<Rpp32f *>(calloc(ioBufferSize, sizeof(Rpp32f)));
        colRemapTable = static_cast<Rpp32f *>(calloc(ioBufferSize, sizeof(Rpp32f)));
    }

    // Run case-wise RPP API and measure time
    rppHandle_t handle;
    hipStream_t stream;
    CHECK_RETURN_STATUS(hipStreamCreate(&stream));
    RppBackend backend = RppBackend::RPP_HIP_BACKEND;
    rppCreate(&handle, batchSize, 0, stream, backend);

    int noOfIterations = (int)imageNames.size() / batchSize;
    double maxWallTime = 0, minWallTime = 500, avgWallTime = 0;
    double wallTime;
    string testCaseName;

    // Initialize buffers for any reductionType functions (testCase 87 - tensor_sum alone cannot return final sum as 8u/8s due to overflow. 8u inputs return 64u sums, 8s inputs return 64s sums)
    void *reductionFuncResultArr;
    Rpp32f *mean;
    Rpp32u reductionFuncResultArrLength = srcDescPtr->n * 4;
    if (reductionTypeCase)
    {
        int bitDepthByteSize = 0;
        if ((dstDescPtr->dataType == RpptDataType::F16) || (dstDescPtr->dataType == RpptDataType::F32) || testCase == TENSOR_MEAN || testCase == TENSOR_STDDEV)
            bitDepthByteSize = sizeof(Rpp32f);  // using 32f outputs for 16f and 32f, for testCase 90, 91
        else if ((dstDescPtr->dataType == RpptDataType::U8) || (dstDescPtr->dataType == RpptDataType::I8))
            bitDepthByteSize = (testCase == TENSOR_SUM) ? sizeof(Rpp64u) : sizeof(Rpp8u);

        CHECK_RETURN_STATUS(hipHostMalloc(&reductionFuncResultArr, reductionFuncResultArrLength * bitDepthByteSize));
        if(testCase == TENSOR_STDDEV)
            CHECK_RETURN_STATUS(hipHostMalloc(&mean, reductionFuncResultArrLength * bitDepthByteSize));
    }

    // create generic descriptor and params in case of slice
    RpptGenericDesc descriptor3D;
    RpptGenericDescPtr descriptorPtr3D = &descriptor3D;
    Rpp32s *anchorTensor = NULL, *shapeTensor = NULL;
    Rpp32u *roiTensor = NULL;
    if(testCase == SLICE)
        set_generic_descriptor_slice(srcDescPtr, descriptorPtr3D, batchSize);

    // Allocate hip memory for src/dst
    CHECK_RETURN_STATUS(hipMalloc(&d_input, inputBufferSize));
    CHECK_RETURN_STATUS(hipMalloc(&d_output, outputBufferSize));
    if(dualInputCase)
        CHECK_RETURN_STATUS(hipMalloc(&d_input_second, inputBufferSize));

    RpptROI *roiPtrInputCropRegion;
    if(testCase == RICAP)
        CHECK_RETURN_STATUS(hipHostMalloc(&roiPtrInputCropRegion, 4 * sizeof(RpptROI)));

    void *d_rowRemapTable, *d_colRemapTable;
    if(testCase == LENS_CORRECTION || testCase == REMAP)
    {
        CHECK_RETURN_STATUS(hipMalloc(&d_rowRemapTable, ioBufferSize * sizeof(Rpp32u)));
        CHECK_RETURN_STATUS(hipMalloc(&d_colRemapTable, ioBufferSize * sizeof(Rpp32u)));
        CHECK_RETURN_STATUS(hipMemset(d_rowRemapTable, 0, ioBufferSize * sizeof(Rpp32u)));
        CHECK_RETURN_STATUS(hipMemset(d_colRemapTable, 0, ioBufferSize * sizeof(Rpp32u)));
    }

    Rpp32f *cameraMatrix, *distortionCoeffs;
    if(testCase == LENS_CORRECTION)
    {
        CHECK_RETURN_STATUS(hipHostMalloc(&cameraMatrix, batchSize * 9 * sizeof(Rpp32f)));
        CHECK_RETURN_STATUS(hipHostMalloc(&distortionCoeffs, batchSize * 8 * sizeof(Rpp32f)));
    }

    Rpp32u boxesInEachImage = 3;
    Rpp32f *colorBuffer;
    RpptRoiLtrb *anchorBoxInfoTensor;
    Rpp32u *numOfBoxes;
    if(testCase == ERASE)
    {
        CHECK_RETURN_STATUS(hipHostMalloc(&colorBuffer, batchSize * boxesInEachImage * sizeof(Rpp32f)));
        CHECK_RETURN_STATUS(hipMemset(colorBuffer, 0, batchSize * boxesInEachImage * sizeof(Rpp32f)));
        CHECK_RETURN_STATUS(hipHostMalloc(&anchorBoxInfoTensor, batchSize * boxesInEachImage * sizeof(RpptRoiLtrb)));
        CHECK_RETURN_STATUS(hipHostMalloc(&numOfBoxes, batchSize * sizeof(Rpp32u)));
    }

    // create cropRoi and patchRoi in case of crop_and_patch
    RpptROI *cropRoi, *patchRoi;
    if(testCase == CROP_AND_PATCH)
    {
        CHECK_RETURN_STATUS(hipHostMalloc(&cropRoi, batchSize * sizeof(RpptROI)));
        CHECK_RETURN_STATUS(hipHostMalloc(&patchRoi, batchSize * sizeof(RpptROI)));
    }
    bool invalidROI = (roiList[0] == 0 && roiList[1] == 0 && roiList[2] == 0 && roiList[3] == 0);

    Rpp32f *intensity;
    if(testCase == VIGNETTE)
        CHECK_RETURN_STATUS(hipHostMalloc(&intensity, batchSize * sizeof(Rpp32f)));

    Rpp32f *intensityFactor = nullptr;
    Rpp32f *greyFactor = nullptr;
    if(testCase == 10)
    {
        CHECK_RETURN_STATUS(hipHostMalloc(&intensityFactor, batchSize * sizeof(Rpp32f)));
        CHECK_RETURN_STATUS(hipHostMalloc(&greyFactor, batchSize * sizeof(Rpp32f)));
    }

    Rpp32u *kernelSizeTensor;
    if(testCase == JITTER)
        CHECK_RETURN_STATUS(hipHostMalloc(&kernelSizeTensor, batchSize * sizeof(Rpp32u)));

    RpptChannelOffsets *rgbOffsets;
    if(testCase == GLITCH)
        CHECK_RETURN_STATUS(hipHostMalloc(&rgbOffsets, batchSize * sizeof(RpptChannelOffsets)));

    void *d_interDstPtr;
    if(testCase == PIXELATE)
        CHECK_RETURN_STATUS(hipHostMalloc(&d_interDstPtr, srcDescPtr->strides.nStride * srcDescPtr->n * sizeof(Rpp32f)));

    Rpp32f *perspectiveTensorPtr = NULL;
    if(testCase == WARP_PERSPECTIVE)
        CHECK_RETURN_STATUS(hipHostMalloc(&perspectiveTensorPtr, batchSize * 9 * sizeof(Rpp32f)));

    Rpp32f *alpha = nullptr;
    if(testCase == RAIN)
        CHECK_RETURN_STATUS(hipHostMalloc(&alpha, batchSize * sizeof(Rpp32f)));

    Rpp32f *minTensor = nullptr, *maxTensor = nullptr;
    if(testCase == THRESHOLD)
    {
        CHECK_RETURN_STATUS(hipHostMalloc(&minTensor, batchSize * srcDescPtr->c * sizeof(Rpp32f)));
        CHECK_RETURN_STATUS(hipHostMalloc(&maxTensor, batchSize * srcDescPtr->c * sizeof(Rpp32f)));
    }

    // case-wise RPP API and measure time script for Unit and Performance test
    cout << "\nRunning " << func << " " << numRuns << " times (each time with a batch size of " << batchSize << " images) and computing mean statistics...";
    for(int iterCount = 0; iterCount < noOfIterations; iterCount++)
    {
        vector<string>::const_iterator imagesPathStart = imageNamesPath.begin() + (iterCount * batchSize);
        vector<string>::const_iterator imagesPathEnd = imagesPathStart + batchSize;
        vector<string>::const_iterator imageNamesStart = imageNames.begin() + (iterCount * batchSize);
        vector<string>::const_iterator imageNamesEnd = imageNamesStart + batchSize;
        vector<string>::const_iterator imagesPathSecondStart = imageNamesPathSecond.begin() + (iterCount * batchSize);
        vector<string>::const_iterator imagesPathSecondEnd = imagesPathSecondStart + batchSize;

        // Set ROIs for src/dst
        set_src_and_dst_roi(imagesPathStart, imagesPathEnd, roiTensorPtrSrc, roiTensorPtrDst, dstImgSizes);

        //Read images
        if(decoderType == 0)
            read_image_batch_turbojpeg(inputu8, srcDescPtr, imagesPathStart);
        else
            read_image_batch_opencv(inputu8, srcDescPtr, imagesPathStart);

        // if the input layout requested is PLN3, convert PKD3 inputs to PLN3 for first and second input batch
        if (layoutType == 1)
            convert_pkd3_to_pln3(inputu8, srcDescPtr);

        if(dualInputCase)
        {
            if(decoderType == 0)
                read_image_batch_turbojpeg(inputu8Second, srcDescPtr, imagesPathSecondStart);
            else
                read_image_batch_opencv(inputu8Second, srcDescPtr, imagesPathSecondStart);
            if (layoutType == 1)
                convert_pkd3_to_pln3(inputu8Second, srcDescPtr);
        }

        // Convert inputs to correponding bit depth specified by user
        convert_input_bitdepth(input, input_second, inputu8, inputu8Second, inputBitDepth, ioBufferSize, inputBufferSize, srcDescPtr, dualInputCase, conversionFactor);

        //copy decoded inputs to hip buffers
        CHECK_RETURN_STATUS(hipMemcpy(d_input, input, inputBufferSize, hipMemcpyHostToDevice));
        CHECK_RETURN_STATUS(hipMemcpy(d_output, output, outputBufferSize, hipMemcpyHostToDevice));
        if(dualInputCase)
            CHECK_RETURN_STATUS(hipMemcpy(d_input_second, input_second, inputBufferSize, hipMemcpyHostToDevice));

        int roiHeightList[batchSize], roiWidthList[batchSize];
        if(invalidROI)
        {
            for(int i = 0; i < batchSize ; i++)
            {
                roiList[0] = 10;
                roiList[1] = 10;
                roiWidthList[i] = roiTensorPtrSrc[i].xywhROI.roiWidth / 2;
                roiHeightList[i] = roiTensorPtrSrc[i].xywhROI.roiHeight / 2;
            }
        }
        else
        {
            for(int i = 0; i < batchSize ; i++)
            {
                roiWidthList[i] = roiList[2];
                roiHeightList[i] = roiList[3];
            }
        }

        // Uncomment to run test case with an xywhROI override
        // roi.xywhROI = {0, 0, 25, 25};
        // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, batchSize);
        // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, batchSize);

        // Uncomment to run test case with an ltrbROI override
        // roiTypeSrc = RpptRoiType::LTRB;
        // roi.ltrbROI = {10, 10, 40, 40};
        // set_roi_values(&roi, roiTensorPtrSrc, roiTypeSrc, batchSize);
        // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, batchSize);

        for (int perfRunCount = 0; perfRunCount < numRuns; perfRunCount++)
        {
            double startWallTime, endWallTime;
            switch (testCase)
            {
                case BRIGHTNESS:
                {
                    testCaseName = "brightness";

                    Rpp32f alpha[batchSize];
                    Rpp32f beta[batchSize];
                    for (i = 0; i < batchSize; i++)
                    {
                        alpha[i] = 1.75;
                        beta[i] = 50;
                    }

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_brightness_gpu(d_input, srcDescPtr, d_output, dstDescPtr, alpha, beta, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case GAMMA_CORRECTION:
                {
                    testCaseName = "gamma_correction";

                    Rpp32f gammaVal[batchSize];
                    for (i = 0; i < batchSize; i++)
                        gammaVal[i] = 1.9;

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_gamma_correction_gpu(d_input, srcDescPtr, d_output, dstDescPtr, gammaVal, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case BLEND:
                {
                    testCaseName = "blend";

                    Rpp32f alpha[batchSize];
                    for (i = 0; i < batchSize; i++)
                        alpha[i] = 0.4;

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_blend_gpu(d_input, d_input_second, srcDescPtr, d_output, dstDescPtr, alpha, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case CONTRAST:
                {
                    testCaseName = "contrast";

                    Rpp32f contrastFactor[batchSize];
                    Rpp32f contrastCenter[batchSize];
                    for (i = 0; i < batchSize; i++)
                    {
                        contrastFactor[i] = 2.96;
                        contrastCenter[i] = 128;
                    }

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_contrast_gpu(d_input, srcDescPtr, d_output, dstDescPtr, contrastFactor, contrastCenter, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case PIXELATE:
                {
                    testCaseName = "pixelate";

                    Rpp32f pixelationPercentage = 87.5;

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_pixelate_gpu(d_input, srcDescPtr, d_output, dstDescPtr, d_interDstPtr, pixelationPercentage, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case JITTER:
                {
                    testCaseName = "jitter";

                    Rpp32u seed = 1255459;
                    for (i = 0; i < batchSize; i++)
                        kernelSizeTensor[i] = 5;

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_jitter_gpu(d_input, srcDescPtr, d_output, dstDescPtr, kernelSizeTensor, seed, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case NOISE:
                {
                    testCaseName = "noise";

                    switch(additionalParam)
                    {
                        case 0:
                        {
                            Rpp32f noiseProbabilityTensor[batchSize];
                            Rpp32f saltProbabilityTensor[batchSize];
                            Rpp32f saltValueTensor[batchSize];
                            Rpp32f pepperValueTensor[batchSize];
                            Rpp32u seed = 1255459;
                            for (i = 0; i < batchSize; i++)
                            {
                                noiseProbabilityTensor[i] = 0.1f;
                                saltProbabilityTensor[i] = 0.5f;
                                saltValueTensor[i] = 1.0f;
                                pepperValueTensor[i] = 0.0f;
                            }

                            startWallTime = omp_get_wtime();
                            if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                                rppt_salt_and_pepper_noise_gpu(d_input, srcDescPtr, d_output, dstDescPtr, noiseProbabilityTensor, saltProbabilityTensor, saltValueTensor, pepperValueTensor, seed, roiTensorPtrSrc, roiTypeSrc, handle);
                            else
                                missingFuncFlag = 1;

                            break;
                        }
                        case 1:
                        {
                            Rpp32f meanTensor[batchSize];
                            Rpp32f stdDevTensor[batchSize];
                            Rpp32u seed = 1255459;
                            for (i = 0; i < batchSize; i++)
                            {
                                meanTensor[i] = 0.0f;
                                stdDevTensor[i] = 0.2f;
                            }

                            startWallTime = omp_get_wtime();
                            if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                                rppt_gaussian_noise_gpu(d_input, srcDescPtr, d_output, dstDescPtr, meanTensor, stdDevTensor, seed, roiTensorPtrSrc, roiTypeSrc, handle);
                            else
                                missingFuncFlag = 1;

                            break;
                        }
                        case 2:
                        {
                            Rpp32f shotNoiseFactorTensor[batchSize];
                            Rpp32u seed = 1255459;
                            for (i = 0; i < batchSize; i++)
                                shotNoiseFactorTensor[i] = 80.0f;

                            startWallTime = omp_get_wtime();
                            if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                                rppt_shot_noise_gpu(d_input, srcDescPtr, d_output, dstDescPtr, shotNoiseFactorTensor, seed, roiTensorPtrSrc, roiTypeSrc, handle);
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
                case FOG:
                {
                    testCaseName = "fog";

                    for (i = 0; i < batchSize; i++)
                    {
                        intensityFactor[i] = 0;
                        greyFactor[i] = 0.3;
                    }

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_fog_gpu(d_input, srcDescPtr, d_output, dstDescPtr, intensityFactor, greyFactor, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case EXPOSURE:
                {
                    testCaseName = "exposure";

                    Rpp32f exposureFactor[batchSize];
                    for (i = 0; i < batchSize; i++)
                        exposureFactor[i] = 1.4;

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_exposure_gpu(d_input, srcDescPtr, d_output, dstDescPtr, exposureFactor, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case RAIN:
                {
                    testCaseName = "rain";

                    Rpp32f rainPercentage = 7;
                    Rpp32u rainHeight = 6;
                    Rpp32u rainWidth = 1;
                    Rpp32f slantAngle = 0;
                    for (int i = 0; i < batchSize; i++)
                        alpha[i] = 0.4;

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_rain_gpu(d_input, srcDescPtr, d_output, dstDescPtr, rainPercentage, rainWidth, rainHeight, slantAngle, alpha, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case THRESHOLD:
                {
                    testCaseName = "threshold";
                    Rpp32f normFactor = 1;
                    Rpp32f subtractionFactor = 0;
                    if (inputBitDepth == 1 || inputBitDepth == 2)
                        normFactor = 255;
                    else if (inputBitDepth == 5)
                        subtractionFactor = 128;

                    for (int i = 0; i < batchSize; i++)
                    {
                        for (int j = 0, k = i * srcDescPtr->c; j < srcDescPtr->c; j++, k++)
                        {
                            minTensor[k] = (30 / normFactor) - subtractionFactor;
                            maxTensor[k] = (100 / normFactor) - subtractionFactor;
                        }
                    }

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_threshold_gpu(d_input, srcDescPtr, d_output, dstDescPtr, minTensor, maxTensor, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case FLIP:
                {
                    testCaseName = "flip";

                    Rpp32u horizontalFlag[batchSize];
                    Rpp32u verticalFlag[batchSize];
                    for (i = 0; i < batchSize; i++)
                    {
                        horizontalFlag[i] = 1;
                        verticalFlag[i] = 0;
                    }

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_flip_gpu(d_input, srcDescPtr, d_output, dstDescPtr, horizontalFlag, verticalFlag, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case RESIZE:
                {
                    testCaseName = "resize";

                    for (i = 0; i < batchSize; i++)
                    {
                        dstImgSizes[i].width = roiTensorPtrDst[i].xywhROI.roiWidth = roiTensorPtrSrc[i].xywhROI.roiWidth / 2;
                        dstImgSizes[i].height = roiTensorPtrDst[i].xywhROI.roiHeight = roiTensorPtrSrc[i].xywhROI.roiHeight / 2;
                    }

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_resize_gpu(d_input, srcDescPtr, d_output, dstDescPtr, dstImgSizes, interpolationType, roiTensorPtrDst, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case ROTATE:
                {
                    testCaseName = "rotate";

                    if ((interpolationType != RpptInterpolationType::BILINEAR) && (interpolationType != RpptInterpolationType::NEAREST_NEIGHBOR))
                    {
                        missingFuncFlag = 1;
                        break;
                    }

                    Rpp32f angle[batchSize];
                    for (i = 0; i < batchSize; i++)
                        angle[i] = 50;

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_rotate_gpu(d_input, srcDescPtr, d_output, dstDescPtr, angle, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case WARP_AFFINE:
                {
                    testCaseName = "warp_affine";

                    if ((interpolationType != RpptInterpolationType::BILINEAR) && (interpolationType != RpptInterpolationType::NEAREST_NEIGHBOR))
                    {
                        missingFuncFlag = 1;
                        break;
                    }

                    Rpp32f6 affineTensor_f6[batchSize];
                    Rpp32f *affineTensor = (Rpp32f *)affineTensor_f6;
                    for (i = 0; i < batchSize; i++)
                    {
                        affineTensor_f6[i].data[0] = 1.23;
                        affineTensor_f6[i].data[1] = 0.5;
                        affineTensor_f6[i].data[2] = 0;
                        affineTensor_f6[i].data[3] = -0.8;
                        affineTensor_f6[i].data[4] = 0.83;
                        affineTensor_f6[i].data[5] = 0;
                    }

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_warp_affine_gpu(d_input, srcDescPtr, d_output, dstDescPtr, affineTensor, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case LENS_CORRECTION:
                {
                    testCaseName = "lens_correction";

                    RpptDesc tableDesc = srcDesc;
                    RpptDescPtr tableDescPtr = &tableDesc;
                    init_lens_correction(batchSize, srcDescPtr, cameraMatrix, distortionCoeffs, tableDescPtr);

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_lens_correction_gpu(d_input, srcDescPtr, d_output, dstDescPtr, static_cast<Rpp32f *>(d_rowRemapTable), static_cast<Rpp32f *>(d_colRemapTable), tableDescPtr, cameraMatrix, distortionCoeffs, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case WARP_PERSPECTIVE:
                {
                    testCaseName = "warp_perspective";

                    if ((interpolationType != RpptInterpolationType::BILINEAR) && (interpolationType != RpptInterpolationType::NEAREST_NEIGHBOR))
                    {
                        missingFuncFlag = 1;
                        break;
                    }

                    for (i = 0, j = 0; i < batchSize; i++, j += 9)
                    {
                        perspectiveTensorPtr[j + 0] = 0.93;
                        perspectiveTensorPtr[j + 1] = 0.5;
                        perspectiveTensorPtr[j + 2] = 0.0;
                        perspectiveTensorPtr[j + 3] = -0.5;
                        perspectiveTensorPtr[j + 4] = 0.93;
                        perspectiveTensorPtr[j + 5] = 0.0;
                        perspectiveTensorPtr[j + 6] = 0.005;
                        perspectiveTensorPtr[j + 7] = 0.005;
                        perspectiveTensorPtr[j + 8] = 1;
                    }

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_warp_perspective_gpu(d_input, srcDescPtr, d_output, dstDescPtr, perspectiveTensorPtr, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case WATER:
                {
                    testCaseName = "water";

                    Rpp32f amplX[batchSize];
                    Rpp32f amplY[batchSize];
                    Rpp32f freqX[batchSize];
                    Rpp32f freqY[batchSize];
                    Rpp32f phaseX[batchSize];
                    Rpp32f phaseY[batchSize];

                    for (i = 0; i < batchSize; i++)
                    {
                        amplX[i] = 2.0f;
                        amplY[i] = 5.0f;
                        freqX[i] = 5.8f;
                        freqY[i] = 1.2f;
                        phaseX[i] = 10.0f;
                        phaseY[i] = 15.0f;
                    }

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_water_gpu(d_input, srcDescPtr, d_output, dstDescPtr, amplX, amplY, freqX, freqY, phaseX, phaseY, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case NON_LINEAR_BLEND:
                {
                    testCaseName = "non_linear_blend";

                    Rpp32f stdDev[batchSize];
                    for (i = 0; i < batchSize; i++)
                        stdDev[i] = 50.0;

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_non_linear_blend_gpu(d_input, d_input_second, srcDescPtr, d_output, dstDescPtr, stdDev, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case COLOR_CAST:
                {
                    testCaseName = "color_cast";

                    RpptRGB rgbTensor[batchSize];
                    Rpp32f alphaTensor[batchSize];

                    for (i = 0; i < batchSize; i++)
                    {
                        rgbTensor[i].R = 0;
                        rgbTensor[i].G = 0;
                        rgbTensor[i].B = 100;
                        alphaTensor[i] = 0.5;
                    }

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_color_cast_gpu(d_input, srcDescPtr, d_output, dstDescPtr, rgbTensor, alphaTensor, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case ERASE:
                {
                    testCaseName = "erase";

                    init_erase(batchSize, boxesInEachImage, numOfBoxes, anchorBoxInfoTensor, roiTensorPtrSrc, srcDescPtr->c, colorBuffer, inputBitDepth);
                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_erase_gpu(d_input, srcDescPtr, d_output, dstDescPtr, anchorBoxInfoTensor, colorBuffer, numOfBoxes, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case CROP_AND_PATCH:
                {
                    testCaseName = "crop_and_patch";
                    for (i = 0; i < batchSize; i++)
                    {
                        cropRoi[i].xywhROI.xy.x = patchRoi[i].xywhROI.xy.x = roiList[0];
                        cropRoi[i].xywhROI.xy.y = patchRoi[i].xywhROI.xy.y = roiList[1];
                        cropRoi[i].xywhROI.roiWidth = patchRoi[i].xywhROI.roiWidth = roiWidthList[i];
                        cropRoi[i].xywhROI.roiHeight = patchRoi[i].xywhROI.roiHeight = roiHeightList[i];
                    }

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_crop_and_patch_gpu(d_input, d_input_second, srcDescPtr, d_output, dstDescPtr, roiTensorPtrSrc, cropRoi, patchRoi, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case LOOK_UP_TABLE:
                {
                    testCaseName = "lut";

                    Rpp32f *lutBuffer;
                    CHECK_RETURN_STATUS(hipHostMalloc(&lutBuffer, 65536 * sizeof(Rpp32f)));
                    CHECK_RETURN_STATUS(hipMemset(lutBuffer, 0, 65536 * sizeof(Rpp32f)));
                    Rpp8u *lut8u = reinterpret_cast<Rpp8u *>(lutBuffer);
                    Rpp16f *lut16f = reinterpret_cast<Rpp16f *>(lutBuffer);
                    Rpp32f *lut32f = reinterpret_cast<Rpp32f *>(lutBuffer);
                    Rpp8s *lut8s = reinterpret_cast<Rpp8s *>(lutBuffer);
                    if (inputBitDepth == 0)
                        for (j = 0; j < 256; j++)
                            lut8u[j] = (Rpp8u)(255 - j);
                    else if (inputBitDepth == 3)
                        for (j = 0; j < 256; j++)
                            lut16f[j] = (Rpp16f)((255 - j) * ONE_OVER_255);
                    else if (inputBitDepth == 4)
                        for (j = 0; j < 256; j++)
                            lut32f[j] = (Rpp32f)((255 - j) * ONE_OVER_255);
                    else if (inputBitDepth == 5)
                        for (j = 0; j < 256; j++)
                            lut8s[j] = (Rpp8s)(255 - j - 128);

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0)
                        rppt_lut_gpu(d_input, srcDescPtr, d_output, dstDescPtr, lut8u, roiTensorPtrSrc, roiTypeSrc, handle);
                    else if (inputBitDepth == 3)
                        rppt_lut_gpu(d_input, srcDescPtr, d_output, dstDescPtr, lut16f, roiTensorPtrSrc, roiTypeSrc, handle);
                    else if (inputBitDepth == 4)
                        rppt_lut_gpu(d_input, srcDescPtr, d_output, dstDescPtr, lut32f, roiTensorPtrSrc, roiTypeSrc, handle);
                    else if (inputBitDepth == 5)
                        rppt_lut_gpu(d_input, srcDescPtr, d_output, dstDescPtr, lut8s, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;

                    CHECK_RETURN_STATUS(hipHostFree(lutBuffer));
                }
                case GLITCH:
                {
                    testCaseName = "glitch";

                    for (i = 0; i < batchSize; i++)
                    {
                        rgbOffsets[i].r.x = 10;
                        rgbOffsets[i].r.y = 10;
                        rgbOffsets[i].g.x = 0;
                        rgbOffsets[i].g.y = 0;
                        rgbOffsets[i].b.x = 5;
                        rgbOffsets[i].b.y = 5;
                    }

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_glitch_gpu(d_input, srcDescPtr, d_output, dstDescPtr, rgbOffsets, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case COLOR_TWIST:
                {
                    testCaseName = "color_twist";

                    Rpp32f brightness[batchSize];
                    Rpp32f contrast[batchSize];
                    Rpp32f hue[batchSize];
                    Rpp32f saturation[batchSize];
                    for (i = 0; i < batchSize; i++)
                    {
                        brightness[i] = 1.4;
                        contrast[i] = 0.0;
                        hue[i] = 60.0;
                        saturation[i] = 1.9;
                    }

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_color_twist_gpu(d_input, srcDescPtr, d_output, dstDescPtr, brightness, contrast, hue, saturation, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case CROP:
                {
                    testCaseName = "crop";

                    for (i = 0; i < batchSize; i++)
                    {
                        roiTensorPtrDst[i].xywhROI.xy.x = roiList[0];
                        roiTensorPtrDst[i].xywhROI.xy.y = roiList[1];
                        dstImgSizes[i].width = roiTensorPtrDst[i].xywhROI.roiWidth = roiWidthList[i];
                        dstImgSizes[i].height = roiTensorPtrDst[i].xywhROI.roiHeight = roiHeightList[i];
                    }

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_crop_gpu(d_input, srcDescPtr, d_output, dstDescPtr, roiTensorPtrDst, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case CROP_MIRROR_NORMALIZE:
                {
                    testCaseName = "crop_mirror_normalize";
                    Rpp32f multiplier[batchSize * srcDescPtr->c];
                    Rpp32f offset[batchSize * srcDescPtr->c];
                    Rpp32u mirror[batchSize];
                    if (srcDescPtr->c == 3)
                    {
                        Rpp32f meanParam[3] = { 60.0f, 80.0f, 100.0f };
                        Rpp32f stdDevParam[3] = { 0.9f, 0.9f, 0.9f };
                        Rpp32f offsetParam[3] = { - meanParam[0] / stdDevParam[0], - meanParam[1] / stdDevParam[1], - meanParam[2] / stdDevParam[2] };
                        Rpp32f multiplierParam[3] = {  1.0f / stdDevParam[0], 1.0f / stdDevParam[1], 1.0f / stdDevParam[2] };

                        for (i = 0, j = 0; i < batchSize; i++, j += 3)
                        {
                            multiplier[j] = multiplierParam[0];
                            offset[j] = offsetParam[0];
                            multiplier[j + 1] = multiplierParam[1];
                            offset[j + 1] = offsetParam[1];
                            multiplier[j + 2] = multiplierParam[2];
                            offset[j + 2] = offsetParam[2];
                            mirror[i] = 1;
                        }
                    }
                    else if(srcDescPtr->c == 1)
                    {
                        Rpp32f meanParam = 100.0f;
                        Rpp32f stdDevParam = 0.9f;
                        Rpp32f offsetParam = - meanParam / stdDevParam;
                        Rpp32f multiplierParam = 1.0f / stdDevParam;

                        for (i = 0; i < batchSize; i++)
                        {
                            multiplier[i] = multiplierParam;
                            offset[i] = offsetParam;
                            mirror[i] = 1;
                        }
                    }

                    for (i = 0; i < batchSize; i++)
                    {
                        roiTensorPtrDst[i].xywhROI.xy.x = roiList[0];
                        roiTensorPtrDst[i].xywhROI.xy.y = roiList[1];
                        dstImgSizes[i].width = roiTensorPtrDst[i].xywhROI.roiWidth = roiWidthList[i];
                        dstImgSizes[i].height = roiTensorPtrDst[i].xywhROI.roiHeight = roiHeightList[i];
                    }

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 3 || inputBitDepth == 4 || inputBitDepth == 5)
                        rppt_crop_mirror_normalize_gpu(d_input, srcDescPtr, d_output, dstDescPtr, offset, multiplier, mirror, roiTensorPtrDst, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case RESIZE_CROP_MIRROR:
                {
                    testCaseName = "resize_crop_mirror";

                    if (interpolationType != RpptInterpolationType::BILINEAR)
                    {
                        missingFuncFlag = 1;
                        break;
                    }

                    Rpp32u mirror[batchSize];
                    for (i = 0; i < batchSize; i++)
                        mirror[i] = 1;

                    for (i = 0; i < batchSize; i++)
                    {
                        roiTensorPtrSrc[i].xywhROI.xy.x = 10;
                        roiTensorPtrSrc[i].xywhROI.xy.y = 10;
                        dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth / 2;
                        dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight / 2;
                        roiTensorPtrDst[i].xywhROI.roiWidth = 50;
                        roiTensorPtrDst[i].xywhROI.roiHeight = 50;
                    }

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 3 || inputBitDepth == 4 || inputBitDepth == 5)
                        rppt_resize_crop_mirror_gpu(d_input, srcDescPtr, d_output, dstDescPtr, dstImgSizes, interpolationType, mirror, roiTensorPtrDst, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case COLOR_TEMPERATURE:
                {
                    testCaseName = "color_temperature";

                    Rpp32s adjustment[batchSize];
                    for (i = 0; i < batchSize; i++)
                        adjustment[i] = 70;

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_color_temperature_gpu(d_input, srcDescPtr, d_output, dstDescPtr, adjustment, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case VIGNETTE:
                {
                    testCaseName = "vignette";

                    for (i = 0; i < batchSize; i++)
                        intensity[i] = 6;

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_vignette_gpu(d_input, srcDescPtr, d_output, dstDescPtr, intensity, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case BOX_FILTER:
                {
                    testCaseName = "box_filter";
                    Rpp32u kernelSize = additionalParam;

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_box_filter_gpu(d_input, srcDescPtr, d_output, dstDescPtr, kernelSize, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case GAUSSIAN_FILTER:
                {
                    testCaseName = "gaussian_filter";
                    Rpp32u kernelSize = additionalParam;

                    Rpp32f stdDevTensor[batchSize];
                    for (i = 0; i < batchSize; i++)
                    {
                        stdDevTensor[i] = 5.0f;
                    }

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_gaussian_filter_gpu(d_input, srcDescPtr, d_output, dstDescPtr, stdDevTensor, kernelSize, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case MAGNITUDE:
                {
                    testCaseName = "magnitude";

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_magnitude_gpu(d_input, d_input_second, srcDescPtr, d_output, dstDescPtr, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case PHASE:
                {
                    testCaseName = "phase";

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_phase_gpu(d_input, d_input_second, srcDescPtr, d_output, dstDescPtr, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case BITWISE_AND:
                {
                    testCaseName = "bitwise_and";

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0)
                        rppt_bitwise_and_gpu(d_input, d_input_second, srcDescPtr, d_output, dstDescPtr, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case BITWISE_NOT:
                {
                    testCaseName = "bitwise_not";

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0)
                        rppt_bitwise_not_gpu(d_input, srcDescPtr, d_output, dstDescPtr, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case BITWISE_XOR:
                {
                    testCaseName = "bitwise_xor";

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0)
                        rppt_bitwise_xor_gpu(d_input, d_input_second, srcDescPtr, d_output, dstDescPtr, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case BITWISE_OR:
                {
                    testCaseName = "bitwise_or";

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0)
                        rppt_bitwise_or_gpu(d_input, d_input_second, srcDescPtr, d_output, dstDescPtr, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case COPY:
                {
                    testCaseName = "copy";

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_copy_gpu(d_input, srcDescPtr, d_output, dstDescPtr, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case REMAP:
                {
                    testCaseName = "remap";

                    RpptDesc tableDesc = srcDesc;
                    RpptDescPtr tableDescPtr = &tableDesc;
                    init_remap(tableDescPtr, srcDescPtr, roiTensorPtrSrc, rowRemapTable, colRemapTable);

                    CHECK_RETURN_STATUS(hipMemcpy(d_rowRemapTable, (void *)rowRemapTable, ioBufferSize * sizeof(Rpp32f), hipMemcpyHostToDevice));
                    CHECK_RETURN_STATUS(hipMemcpy(d_colRemapTable, (void *)colRemapTable, ioBufferSize * sizeof(Rpp32f), hipMemcpyHostToDevice));

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_remap_gpu(d_input, srcDescPtr, d_output, dstDescPtr, (Rpp32f *)d_rowRemapTable, (Rpp32f *)d_colRemapTable, tableDescPtr, interpolationType, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case RESIZE_MIRROR_NORMALIZE:
                {
                    testCaseName = "resize_mirror_normalize";

                    if (interpolationType != RpptInterpolationType::BILINEAR)
                    {
                        missingFuncFlag = 1;
                        break;
                    }

                    for (i = 0; i < batchSize; i++)
                    {
                        dstImgSizes[i].width = roiTensorPtrDst[i].xywhROI.roiWidth = roiTensorPtrSrc[i].xywhROI.roiWidth / 2;
                        dstImgSizes[i].height = roiTensorPtrDst[i].xywhROI.roiHeight = roiTensorPtrSrc[i].xywhROI.roiWidth / 2;
                    }

                    Rpp32f mean[batchSize * 3];
                    Rpp32f stdDev[batchSize * 3];
                    Rpp32u mirror[batchSize];
                    for (i = 0, j = 0; i < batchSize; i++, j += 3)
                    {
                        mean[j] = 60.0;
                        stdDev[j] = 1.0;

                        mean[j + 1] = 80.0;
                        stdDev[j + 1] = 1.0;

                        mean[j + 2] = 100.0;
                        stdDev[j + 2] = 1.0;
                        mirror[i] = 1;
                    }

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_resize_mirror_normalize_gpu(d_input, srcDescPtr, d_output, dstDescPtr, dstImgSizes, interpolationType, mean, stdDev, mirror, roiTensorPtrDst, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case RICAP:
                {
                    testCaseName = "ricap";

                    Rpp32u permutationTensor[batchSize * 4];
                    if(qaFlag)
                        init_ricap_qa(maxWidth, maxHeight, batchSize, permutationTensor, roiPtrInputCropRegion);
                    else
                        init_ricap(maxWidth, maxHeight, batchSize, permutationTensor, roiPtrInputCropRegion);

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_ricap_gpu(d_input, srcDescPtr, d_output, dstDescPtr, permutationTensor, roiPtrInputCropRegion, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;
                    break;
                }
                case GRIDMASK:
                {
                    testCaseName = "gridmask";

                    Rpp32u tileWidth = 40;
                    Rpp32f gridRatio = 0.6;
                    Rpp32f gridAngle = 0.5;
                    RpptUintVector2D translateVector;
                    translateVector.x = 0.0;
                    translateVector.y = 0.0;

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_gridmask_gpu(d_input, srcDescPtr, d_output, dstDescPtr, tileWidth, gridRatio, gridAngle, translateVector, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case SPATTER:
                {
                    testCaseName = "spatter";

                    RpptRGB spatterColor;

                    // Mud Spatter
                    spatterColor.R = 65;
                    spatterColor.G = 50;
                    spatterColor.B = 23;

                    // Blood Spatter
                    // spatterColor.R = 98;

                    // Ink Spatter
                    // spatterColor.R = 5;
                    // spatterColor.G = 20;
                    // spatterColor.B = 64;

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_spatter_gpu(d_input, srcDescPtr, d_output, dstDescPtr, spatterColor, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case SWAP_CHANNELS:
                {
                    testCaseName = "swap_channels";

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_swap_channels_gpu(d_input, srcDescPtr, d_output, dstDescPtr, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case COLOR_TO_GREYSCALE:
                {
                    testCaseName = "color_to_greyscale";

                    RpptSubpixelLayout srcSubpixelLayout = RpptSubpixelLayout::RGBtype;

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_color_to_greyscale_gpu(d_input, srcDescPtr, d_output, dstDescPtr, srcSubpixelLayout, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case TENSOR_SUM:
                {
                    testCaseName = "tensor_sum";

                    if(srcDescPtr->c == 1)
                        reductionFuncResultArrLength = srcDescPtr->n;

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_tensor_sum_gpu(d_input, srcDescPtr, reductionFuncResultArr, reductionFuncResultArrLength, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case TENSOR_MIN:
                {
                    testCaseName = "tensor_min";

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_tensor_min_gpu(d_input, srcDescPtr, reductionFuncResultArr, reductionFuncResultArrLength, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case TENSOR_MAX:
                {
                    testCaseName = "tensor_max";

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_tensor_max_gpu(d_input, srcDescPtr, reductionFuncResultArr, reductionFuncResultArrLength, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case TENSOR_MEAN:
                {
                    testCaseName = "tensor_mean";

                    if(srcDescPtr->c == 1)
                        reductionFuncResultArrLength = srcDescPtr->n;

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_tensor_mean_gpu(d_input, srcDescPtr, reductionFuncResultArr, reductionFuncResultArrLength, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case TENSOR_STDDEV:
                {
                    testCaseName = "tensor_stddev";

                    if(srcDescPtr->c == 1)
                        reductionFuncResultArrLength = srcDescPtr->n;
                    memcpy(mean, TensorMeanReferenceOutputs[inputChannels].data(), sizeof(Rpp32f) * reductionFuncResultArrLength);

                    startWallTime = omp_get_wtime();
                    if (inputBitDepth == 0 || inputBitDepth == 1 || inputBitDepth == 2 || inputBitDepth == 5)
                        rppt_tensor_stddev_gpu(d_input, srcDescPtr, reductionFuncResultArr, reductionFuncResultArrLength, mean, roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case SLICE:
                {
                    testCaseName = "slice";
                    Rpp32u numDims = descriptorPtr3D->numDims - 1; // exclude batchSize from input dims
                    if(anchorTensor == NULL)
                        CHECK_RETURN_STATUS(hipHostMalloc(&anchorTensor, batchSize * numDims * sizeof(Rpp32s)));
                    if(shapeTensor == NULL)
                        CHECK_RETURN_STATUS(hipHostMalloc(&shapeTensor, batchSize * numDims * sizeof(Rpp32s)));
                    if(roiTensor == NULL)
                        CHECK_RETURN_STATUS(hipHostMalloc(&roiTensor, batchSize * numDims * 2 * sizeof(Rpp32u)));
                    bool enablePadding = false;
                    auto fillValue = 0;
                    init_slice(descriptorPtr3D, roiTensorPtrSrc, roiTensor, anchorTensor, shapeTensor);

                    startWallTime = omp_get_wtime();
                    if((inputBitDepth == 0 || inputBitDepth == 2) && srcDescPtr->layout == dstDescPtr->layout)
                        rppt_slice_gpu(d_input, descriptorPtr3D, d_output, descriptorPtr3D, anchorTensor, shapeTensor, &fillValue, enablePadding, roiTensor, handle);
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

            CHECK_RETURN_STATUS(hipDeviceSynchronize());
            endWallTime = omp_get_wtime();
            wallTime = endWallTime - startWallTime;
            if (missingFuncFlag == 1)
            {
                cout << "\nThe functionality " << func << " doesn't yet exist in RPP\n";
                return RPP_ERROR_NOT_IMPLEMENTED;
            }

            maxWallTime = max(maxWallTime, wallTime);
            minWallTime = min(minWallTime, wallTime);
            avgWallTime += wallTime;
        }
        wallTime *= 1000;

        if (testType == 0)
        {
            cout <<"\n\n";
            if(noOfIterations > 1)
                cout <<"Execution Timings for Iteration "<< iterCount+1 <<":"<<endl;
            cout << "GPU Backend Wall Time: " << wallTime <<" ms/batch";
            // Display results for reduction functions
            if (reductionTypeCase)
            {
                if(srcDescPtr->c == 3)
                    cout << "\nReduction result (Batch of 3 channel images produces 4 results per image in batch): ";
                else if(srcDescPtr->c == 1)
                {
                    cout << "\nReduction result (Batch of 1 channel images produces 1 result per image in batch): ";
                    reductionFuncResultArrLength = srcDescPtr->n;
                }

                // print reduction functions output array based on different bit depths, and precision desired
                int precision = ((dstDescPtr->dataType == RpptDataType::F32) || (dstDescPtr->dataType == RpptDataType::F16) || testCase == TENSOR_MEAN || testCase == TENSOR_STDDEV) ? 3 : 0;
                if (dstDescPtr->dataType == RpptDataType::F32 || testCase == TENSOR_MEAN || testCase == TENSOR_STDDEV)
                    print_array(static_cast<Rpp32f *>(reductionFuncResultArr), reductionFuncResultArrLength, precision);
                else if (dstDescPtr->dataType == RpptDataType::U8)
                {
                    if (testCase == TENSOR_SUM)
                        print_array(static_cast<Rpp64u *>(reductionFuncResultArr), reductionFuncResultArrLength, precision);
                    else
                        print_array(static_cast<Rpp8u *>(reductionFuncResultArr), reductionFuncResultArrLength, precision);
                }
                else if (dstDescPtr->dataType == RpptDataType::F16)
                {
                    if (testCase == TENSOR_SUM)
                        print_array(static_cast<Rpp32f *>(reductionFuncResultArr), reductionFuncResultArrLength, precision);
                    else
                        print_array(static_cast<Rpp16f *>(reductionFuncResultArr), reductionFuncResultArrLength, precision);
                }
                else if (dstDescPtr->dataType == RpptDataType::I8)
                {
                    if (testCase == TENSOR_SUM)
                        print_array(static_cast<Rpp64s *>(reductionFuncResultArr), reductionFuncResultArrLength, precision);
                    else
                        print_array(static_cast<Rpp8s *>(reductionFuncResultArr), reductionFuncResultArrLength, precision);
                }
                cout << "\n";

                /*Compare the output of the function with golden outputs only if
                1.QA Flag is set
                2.input bit depth 0 (U8)
                3.source and destination layout are the same*/
                if(qaFlag && inputBitDepth == 0 && (srcDescPtr->layout == dstDescPtr->layout) && !(randomOutputCase) && !(nonQACase))
                {
                    if (testCase == TENSOR_SUM)
                        compare_reduction_output(static_cast<uint64_t *>(reductionFuncResultArr), testCaseName, srcDescPtr, testCase, dst, scriptPath);
                    else if (testCase == TENSOR_MEAN || testCase == TENSOR_STDDEV)
                        compare_reduction_output(static_cast<Rpp32f *>(reductionFuncResultArr), testCaseName, srcDescPtr, testCase, dst, scriptPath);
                    else
                        compare_reduction_output(static_cast<Rpp8u *>(reductionFuncResultArr), testCaseName, srcDescPtr, testCase, dst, scriptPath);
                }
            }
            else
            {
                CHECK_RETURN_STATUS(hipMemcpy(output, d_output, outputBufferSize, hipMemcpyDeviceToHost));

                // Reconvert other bit depths to 8u for output display purposes
                convert_output_bitdepth_to_u8(output, outputu8, inputBitDepth, oBufferSize, outputBufferSize, dstDescPtr, invConversionFactor);

                // if DEBUG_MODE is set to 1, the output of the first iteration will be dumped to csv files for debugging purposes.
                if(DEBUG_MODE && iterCount == 0)
                {
                    std::ofstream refFile;
                    refFile.open(func + ".csv");
                    for (int i = 0; i < oBufferSize; i++)
                        refFile << static_cast<int>(*(outputu8 + i)) << ",";
                    refFile.close();
                }

                // if test case is slice and qaFlag is set, update the dstImgSizes with shapeTensor values
                // for output display and comparision purposes
                if (testCase == SLICE)
                {
                    if (dstDescPtr->layout == RpptLayout::NCHW)
                    {
                        if (dstDescPtr->c == 3)
                        {
                            for(int i = 0; i < batchSize; i++)
                            {
                                int idx1 = i * 3;
                                dstImgSizes[i].height = shapeTensor[idx1 + 1];
                                dstImgSizes[i].width = shapeTensor[idx1 + 2];
                            }
                        }
                        else
                        {
                            for(int i = 0; i < batchSize; i++)
                            {
                                int idx1 = i * 2;
                                dstImgSizes[i].height = shapeTensor[idx1];
                                dstImgSizes[i].width = shapeTensor[idx1 + 1];
                            }
                        }
                    }
                    else if (dstDescPtr->layout == RpptLayout::NHWC)
                    {
                        for(int i = 0; i < batchSize; i++)
                        {
                            int idx1 = i * 3;
                            dstImgSizes[i].height = shapeTensor[idx1];
                            dstImgSizes[i].width = shapeTensor[idx1 + 1];
                        }
                    }
                }

                /*Compare the output of the function with golden outputs only if
                1.QA Flag is set
                2.input bit depth 0 (Input U8 && Output U8)
                3.source and destination layout are the same
                4.augmentation case does not generate random output*/
                if(qaFlag && (inputBitDepth == 0 || inputBitDepth == 2) && (!(randomOutputCase) && !(nonQACase)))
                    compare_output(output, testCaseName, srcDescPtr, dstDescPtr, dstImgSizes, batchSize, interpolationTypeName, noiseTypeName, additionalParam, testCase, dst, scriptPath);

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
                // OpenCV dump (if testType is unit test and QA mode is not set)
                if(!qaFlag)
                    write_image_batch_opencv(dst, outputu8, dstDescPtr, imageNamesStart, dstImgSizes, MAX_IMAGE_DUMP);
            }
        }
    }
    rppDestroy(handle, backend);
    if(testType == 1)
    {
        // Display measured times
        maxWallTime *= 1000;
        minWallTime *= 1000;
        avgWallTime *= 1000;
        avgWallTime /= (numRuns * noOfIterations);
        cout << fixed <<"\nmax,min,avg wall times in ms/batch = " << maxWallTime << "," << minWallTime << "," << avgWallTime << endl;
    }

    // Free memory
    CHECK_RETURN_STATUS(hipHostFree(roiTensorPtrSrc));
    CHECK_RETURN_STATUS(hipHostFree(roiTensorPtrDst));
    CHECK_RETURN_STATUS(hipHostFree(dstImgSizes));
    if(testCase == VIGNETTE)
        CHECK_RETURN_STATUS(hipHostFree(intensity));
    if(testCase == RICAP)
        CHECK_RETURN_STATUS(hipHostFree(roiPtrInputCropRegion));
    if(testCase == CROP_AND_PATCH)
    {
        CHECK_RETURN_STATUS(hipHostFree(cropRoi));
        CHECK_RETURN_STATUS(hipHostFree(patchRoi));
    }
    if(testCase == LENS_CORRECTION)
    {
        CHECK_RETURN_STATUS(hipHostFree(cameraMatrix));
        CHECK_RETURN_STATUS(hipHostFree(distortionCoeffs));
    }
    if(testCase == REMAP)
    {
        free(rowRemapTable);
        free(colRemapTable);
        CHECK_RETURN_STATUS(hipFree(d_rowRemapTable));
        CHECK_RETURN_STATUS(hipFree(d_colRemapTable));
    }
    if(testCase == GLITCH)
        CHECK_RETURN_STATUS(hipHostFree(rgbOffsets));
    if(perspectiveTensorPtr != NULL)
      CHECK_RETURN_STATUS(hipHostFree(perspectiveTensorPtr));
    if (reductionTypeCase)
    {
        CHECK_RETURN_STATUS(hipHostFree(reductionFuncResultArr));
        if(testCase == TENSOR_STDDEV)
            CHECK_RETURN_STATUS(hipHostFree(mean));
    }
    if(testCase == ERASE)
    {
        CHECK_RETURN_STATUS(hipHostFree(colorBuffer));
        CHECK_RETURN_STATUS(hipHostFree(anchorBoxInfoTensor));
        CHECK_RETURN_STATUS(hipHostFree(numOfBoxes));
    }
    if(anchorTensor != NULL)
        CHECK_RETURN_STATUS(hipHostFree(anchorTensor));
    if(shapeTensor != NULL)
        CHECK_RETURN_STATUS(hipHostFree(shapeTensor));
    if(intensityFactor != NULL)
        CHECK_RETURN_STATUS(hipHostFree(intensityFactor));
    if(greyFactor != NULL)
        CHECK_RETURN_STATUS(hipHostFree(greyFactor));
    if(roiTensor != NULL)
        CHECK_RETURN_STATUS(hipHostFree(roiTensor));
    if(testCase == JITTER)
        CHECK_RETURN_STATUS(hipHostFree(kernelSizeTensor));
    free(input);
    free(input_second);
    free(output);
    free(inputu8);
    free(inputu8Second);
    free(outputu8);
    CHECK_RETURN_STATUS(hipFree(d_input));
    if(dualInputCase)
        CHECK_RETURN_STATUS(hipFree(d_input_second));
    CHECK_RETURN_STATUS(hipFree(d_output));
    if(testCase == PIXELATE)
        CHECK_RETURN_STATUS(hipFree(d_interDstPtr));
    if(alpha != NULL)
        CHECK_RETURN_STATUS(hipHostFree(alpha));
    if (minTensor != nullptr)
        CHECK_RETURN_STATUS(hipHostFree(minTensor));
    if (maxTensor != nullptr)
        CHECK_RETURN_STATUS(hipHostFree(maxTensor));
    return 0;
}