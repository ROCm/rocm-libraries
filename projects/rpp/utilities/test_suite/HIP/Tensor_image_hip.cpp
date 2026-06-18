/*
MIT License

Copyright (c) 2019 - 2026 Advanced Micro Devices, Inc.

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

#include <dirent.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "../rpp_test_suite_image.h"
#include "rpp.h"

using namespace std;

int main(int argc, char** argv) {
    // Handle inputs
    const int MIN_ARG_COUNT = 19;

    char* src = argv[1];
    char* srcSecond = argv[2];
    string dst = argv[3];

    int BitDepthTestMode = atoi(argv[4]);
    unsigned int outputFormatToggle = atoi(argv[5]);
    int testCase = atoi(argv[6]);
    int numRuns = atoi(argv[8]);
    int testType = atoi(argv[9]);     // 0 for unit and 1 for performance test
    int layoutType = atoi(argv[10]);  // 0 for pkd3 / 1 for pln3 / 2 for pln1
    int qaFlag = atoi(argv[12]);
    int decoderType = atoi(argv[13]);
    int batchSize = atoi(argv[14]);

    bool additionalParamCase = (additionalParamCases.find(testCase) != additionalParamCases.end());
    bool kernelSizeCase = (kernelSizeCases.find(testCase) != kernelSizeCases.end());
    bool dualInputCase = (dualInputCases.find(testCase) != dualInputCases.end());
    bool randomOutputCase = (randomOutputCases.find(testCase) != randomOutputCases.end());
    bool nonQACase = (nonQACases.find(testCase) != nonQACases.end());
    bool interpolationTypeCase =
        (interpolationTypeCases.find(testCase) != interpolationTypeCases.end());
    bool reductionTypeCase = (reductionTypeCases.find(testCase) != reductionTypeCases.end());
    bool noiseTypeCase = (noiseTypeCases.find(testCase) != noiseTypeCases.end());
    bool pln1OutTypeCase = (pln1OutTypeCases.find(testCase) != pln1OutTypeCases.end());
    bool kernelSizeAndGradientCase =
        (kernelSizeAndGradientCases.find(testCase) != kernelSizeAndGradientCases.end());

    unsigned int verbosity = atoi(argv[11]);
    unsigned int additionalParam = additionalParamCase ? atoi(argv[7]) : 1;
    int roiList[4] = {atoi(argv[15]), atoi(argv[16]), atoi(argv[17]), atoi(argv[18])};
    string scriptPath = argv[19];

    if (verbosity == 1) {
        cout << "\nInputs for this test case are:";
        cout << "\nsrc1 = " << argv[1];
        cout << "\nsrc2 = " << argv[2];
        if (testType == UNIT_TEST)  // unit test mode
            cout << "\ndst = " << argv[3];
        cout << "\nu8 / f16 / f32 / u8->f16 / u8->f32 / i8 / u8->i8 (0/1/2/3/4/5/6) = " << argv[4];
        cout << "\noutputFormatToggle (pkd->pkd = 0 / pkd->pln = 1) = " << argv[5];
        cout << "\ncase number (0:101) = " << argv[6];
        cout << "\nnumber of times to run = " << argv[8];
        cout << "\ntest type - (0 = unit tests / 1 = performance tests) = " << argv[9];
        cout << "\nlayout type - (0 = PKD3/ 1 = PLN3/ 2 = PLN1) = " << argv[10];
        cout << "\nqa mode - 0/1 = " << argv[12];
        cout << "\ndecoder type - (0 = packed .rgb / 1 = OpenCV) = " << argv[13];
        cout << "\nbatch size = " << argv[14];
    }

    if (argc < MIN_ARG_COUNT) {
        cout << "\nImproper Usage! Needs all arguments!\n";
        cout << "\nUsage: <src1 folder> <src2 folder (place same as src1 folder for single image "
                "functionalities)> <dst folder> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / "
                "u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <outputFormatToggle (pkd->pkd = 0 / pkd->pln = "
                "1)> <case number = 0:87> <number of runs > 0> <layout type (0 = PKD3/ 1 = PLN3/ 2 "
                "= PLN1)> <qa mode (0/1)> <decoder type (0/1)> <batch size > 1> <roiList> "
                "<verbosity = 0/1>>\n";
        return -1;
    }

    if (layoutType == 2) {
        if (testCase == COLOR_TWIST || testCase == COLOR_CAST || testCase == GLITCH ||
            testCase == COLOR_TEMPERATURE || testCase == COLOR_TO_GREYSCALE ||
            (testCase == YUV_TO_RGB || testCase == YUV_TO_RGB_CUBIC_V ||
             testCase == YUV_TO_RGB_LINEAR_V) ||
            testCase == HUE || testCase == SATURATION) {
            cout << "\ncase " << testCase << " does not exist for PLN1 layout\n";
            return RPP_ERROR_NOT_IMPLEMENTED;
        } else if (outputFormatToggle != 0) {
            cout << "\nPLN1 cases don't have outputFormatToggle! Please input outputFormatToggle = "
                    "0\n";
            return RPP_ERROR_NOT_IMPLEMENTED;
        }
    }

    // yuv_to_rgb outputs packed RGB only; only PKD3 (layout 0) is supported
    if ((testCase == YUV_TO_RGB || testCase == YUV_TO_RGB_CUBIC_V ||
         testCase == YUV_TO_RGB_LINEAR_V) &&
        layoutType != 0) {
        cout << "\nyuv_to_rgb only supports PKD3 (packed RGB) output. Use layout type 0.\n";
        return RPP_ERROR_NOT_IMPLEMENTED;
    }

    if (pln1OutTypeCase && outputFormatToggle != 0) {
        cout << "\ntest case " << testCase
             << " don't have outputFormatToggle! Please input outputFormatToggle = 0\n";
        return RPP_ERROR_NOT_IMPLEMENTED;
    } else if (reductionTypeCase && outputFormatToggle != 0) {
        cout << "\nReduction Kernels don't have outputFormatToggle! Please input "
                "outputFormatToggle = 0\n";
        return RPP_ERROR_NOT_IMPLEMENTED;
    } else if (batchSize > MAX_BATCH_SIZE) {
        std::cerr << "\n Batchsize should be less than or equal to " << MAX_BATCH_SIZE
                  << " Aborting!";
        exit(0);
    } else if (testCase == RICAP && batchSize < 2) {
        std::cerr << "\n RICAP only works with BatchSize > 1";
        exit(0);
    }

    // Get function name
    string funcName = augmentationMap[testCase];
    if (funcName.empty()) {
        if (testType == UNIT_TEST)  // unit test mode
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
    set_descriptor_data_type(BitDepthTestMode, funcName, srcDescPtr, dstDescPtr);

    // Other initializations
    int missingFuncFlag = 0;
    int i = 0, j = 0;
    int maxHeight = 0, maxWidth = 0;
    Rpp64u ioBufferSize = 0;
    Rpp64u oBufferSize = 0;
    static int noOfImages = 0;

    // String ops on input path
    string inputPath = src;
    inputPath += "/";
    string inputPathSecond = srcSecond;
    inputPathSecond += "/";

    string func = funcName;
    func += funcType;

    RpptImageBorderType borderType = RpptImageBorderType::REPLICATE;
    RpptInterpolationType interpolationType = RpptInterpolationType::BILINEAR;
    std::string interpolationTypeName = "";
    std::string noiseTypeName = "";
    std::string kernelSizeAndGradientName = "";
    Rpp32u kernelSize, GradientType;
    if (kernelSizeCase) {
        func += "_kernelSize";
        func += std::to_string(additionalParam);
    } else if (interpolationTypeCase) {
        interpolationTypeName = get_interpolation_type(additionalParam, interpolationType);
        func += "_interpolationType";
        func += interpolationTypeName.c_str();
    } else if (noiseTypeCase) {
        noiseTypeName = get_noise_type(additionalParam);
        func += "_noiseType";
        func += noiseTypeName.c_str();
    } else if (testCase == CHANNEL_PERMUTE) {
        if (additionalParam < 0 || additionalParam > 5) {
            std::cerr << "Error: permutationIdx out of valid range (0 to 5). Received: "
                      << additionalParam << std::endl;
            exit(0);
        }
        func += "_permOrder";
        func += std::to_string(additionalParam);
    } else if (kernelSizeAndGradientCase) {
        kernelSizeAndGradientName =
            get_kernel_size_and_gradient_type(additionalParam, kernelSize, GradientType);
        func += kernelSizeAndGradientName;
    }

    if (!qaFlag) {
        dst += "/";
        dst += func;
    }

    // Get number of images and image Names
    vector<string> imageNames, imageNamesSecond, imageNamesPath, imageNamesPathSecond;
    if ((testCase == YUV_TO_RGB || testCase == YUV_TO_RGB_CUBIC_V ||
         testCase == YUV_TO_RGB_LINEAR_V))
        search_files_recursive(src, imageNames, imageNamesPath, ".yuv");
    else {
        const char* inputExt = (decoderType == 0) ? ".rgb" : ".jpg";
        search_files_recursive(src, imageNames, imageNamesPath, inputExt);
    }
    if (dualInputCase) {
        const char* inputExtSecond = (decoderType == 0) ? ".rgb" : ".jpg";
        search_files_recursive(srcSecond, imageNamesSecond, imageNamesPathSecond, inputExtSecond);
        if (imageNames.size() != imageNamesSecond.size()) {
            std::cerr << " \n The number of images in the input folders must be the same.";
            exit(0);
        }
    }
    noOfImages = imageNames.size();

    if (noOfImages < batchSize || ((noOfImages % batchSize) != 0)) {
        replicate_last_file_to_fill_batch(imageNamesPath[noOfImages - 1], imageNamesPath,
                                          imageNames, imageNames[noOfImages - 1], noOfImages,
                                          batchSize);
        if (dualInputCase)
            replicate_last_file_to_fill_batch(
                imageNamesPathSecond[noOfImages - 1], imageNamesPathSecond, imageNamesSecond,
                imageNamesSecond[noOfImages - 1], noOfImages, batchSize);
        noOfImages = imageNames.size();
    }

    if (!noOfImages) {
        std::cerr
            << "Not able to find any images in the folder specified. Please check the input path";
        exit(0);
    }

    if (qaFlag) {
        sort(imageNames.begin(), imageNames.end());
        if (dualInputCase) sort(imageNamesSecond.begin(), imageNamesSecond.end());
    }

    // Initialize ROI tensors for src/dst
    RpptROI *roiTensorPtrSrc, *roiTensorPtrDst;
    CHECK_RETURN_STATUS(hipHostMalloc(&roiTensorPtrSrc, batchSize * sizeof(RpptROI)));
    CHECK_RETURN_STATUS(hipHostMalloc(&roiTensorPtrDst, batchSize * sizeof(RpptROI)));

    // Initialize the ImagePatch for dst
    RpptImagePatch* dstImgSizes;
    CHECK_RETURN_STATUS(hipHostMalloc(&dstImgSizes, batchSize * sizeof(RpptImagePatch)));

    // Set ROI tensors types for src/dst
    RpptRoiType roiTypeSrc = RpptRoiType::XYWH;

    Rpp32u outputChannels = inputChannels;
    if (pln1OutTypeCase) outputChannels = 1;
    Rpp32u srcOffsetInBytes =
        (kernelSizeCase || kernelSizeAndGradientCase) ? (12 * (additionalParam / 2)) : 0;
    Rpp32u dstOffsetInBytes = 0;
    int imagesMixed = 0;  // Flag used to check if all images in dataset is of same dimensions

    if ((testCase == YUV_TO_RGB || testCase == YUV_TO_RGB_CUBIC_V ||
         testCase == YUV_TO_RGB_LINEAR_V))
        set_max_dimensions_yuv(imageNamesPath, maxHeight, maxWidth, imagesMixed);
    else
        set_max_dimensions(imageNamesPath, maxHeight, maxWidth, imagesMixed, decoderType);
    if (testCase == RICAP && imagesMixed) {
        std::cerr << "\n RICAP only works with same dimension images";
        exit(0);
    }

    Rpp32s additionalStride = 0;
    if (kernelSizeCase) additionalStride = additionalParam / 2;

    // Set numDims, offset, n/c/h/w values, strides for src/dst
    set_descriptor_dims_and_strides(srcDescPtr, batchSize, maxHeight, maxWidth, inputChannels,
                                    srcOffsetInBytes, additionalStride);
    set_descriptor_dims_and_strides(dstDescPtr, batchSize, maxHeight, maxWidth, outputChannels,
                                    dstOffsetInBytes);

    // Factors to convert U8 data to F32, F16 data to 0-1 range and reconvert them back to 0 -255
    // range
    Rpp32f conversionFactor = 1.0f / 255.0;
    if (testCase == CROP_MIRROR_NORMALIZE) conversionFactor = 1.0;
    Rpp32f invConversionFactor = 1.0f / conversionFactor;

    // Set buffer sizes in pixels for src/dst
    ioBufferSize =
        (Rpp64u)srcDescPtr->h * (Rpp64u)srcDescPtr->w * (Rpp64u)srcDescPtr->c * (Rpp64u)batchSize;
    oBufferSize =
        (Rpp64u)dstDescPtr->h * (Rpp64u)dstDescPtr->w * (Rpp64u)dstDescPtr->c * (Rpp64u)batchSize;

    // Set buffer sizes in bytes for src/dst (including offsets)
    Rpp64u ioBufferSizeInBytes_u8 = ioBufferSize + srcDescPtr->offsetInBytes;
    Rpp64u oBufferSizeInBytes_u8 = oBufferSize + dstDescPtr->offsetInBytes;
    Rpp64u inputBufferSize =
        ioBufferSize * get_size_of_data_type(srcDescPtr->dataType) + srcDescPtr->offsetInBytes;
    Rpp64u outputBufferSize =
        oBufferSize * get_size_of_data_type(dstDescPtr->dataType) + dstDescPtr->offsetInBytes;
    if ((testCase == YUV_TO_RGB || testCase == YUV_TO_RGB_CUBIC_V ||
         testCase == YUV_TO_RGB_LINEAR_V)) {
        inputBufferSize = batchSize * ((Rpp64u)maxWidth * maxHeight * 3 / 2);
        ioBufferSizeInBytes_u8 = inputBufferSize;
    }

    // Initialize 8u host buffers for src/dst
    Rpp8u* inputu8 = static_cast<Rpp8u*>(calloc(ioBufferSizeInBytes_u8, 1));
    Rpp8u* inputu8Second = static_cast<Rpp8u*>(calloc(ioBufferSizeInBytes_u8, 1));
    Rpp8u* outputu8 = static_cast<Rpp8u*>(calloc(oBufferSizeInBytes_u8, 1));

    void *input, *input_second, *output;
    void *d_input, *d_input_second, *d_output;

    input = static_cast<Rpp8u*>(calloc(inputBufferSize, 1));
    input_second = static_cast<Rpp8u*>(calloc(inputBufferSize, 1));
    output = static_cast<Rpp8u*>(calloc(outputBufferSize, 1));

    Rpp32f *rowRemapTable, *colRemapTable;
    if (testCase == REMAP) {
        rowRemapTable = static_cast<Rpp32f*>(calloc(ioBufferSize, sizeof(Rpp32f)));
        colRemapTable = static_cast<Rpp32f*>(calloc(ioBufferSize, sizeof(Rpp32f)));
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

    // Initialize buffers for any reductionType functions (testCase 87 - tensor_sum alone cannot
    // return final sum as 8u/8s due to overflow. 8u inputs return 64u sums, 8s inputs return 64s
    // sums)
    void* reductionFuncResultArr;
    Rpp32f* mean;
    Rpp32u reductionFuncResultArrLength = srcDescPtr->n * 4;
    if (reductionTypeCase) {
        int bitDepthByteSize = 0;
        if ((dstDescPtr->dataType == RpptDataType::F16) ||
            (dstDescPtr->dataType == RpptDataType::F32) || testCase == TENSOR_MEAN ||
            testCase == TENSOR_STDDEV)
            bitDepthByteSize =
                sizeof(Rpp32f);  // using 32f outputs for 16f and 32f, for testCase 90, 91
        else if ((dstDescPtr->dataType == RpptDataType::U8) ||
                 (dstDescPtr->dataType == RpptDataType::I8))
            bitDepthByteSize = (testCase == TENSOR_SUM) ? sizeof(Rpp64u) : sizeof(Rpp8u);

        CHECK_RETURN_STATUS(hipHostMalloc(&reductionFuncResultArr,
                                          reductionFuncResultArrLength * bitDepthByteSize));
        if (testCase == TENSOR_STDDEV)
            CHECK_RETURN_STATUS(
                hipHostMalloc(&mean, reductionFuncResultArrLength * bitDepthByteSize));
    }

    // create generic descriptor and params in case of slice
    RpptGenericDesc descriptor3D;
    RpptGenericDescPtr descriptorPtr3D = &descriptor3D;
    Rpp32s *anchorTensor = NULL, *shapeTensor = NULL;
    Rpp32u* roiTensor = NULL;
    if (testCase == SLICE) set_generic_descriptor_slice(srcDescPtr, descriptorPtr3D, batchSize);

    // Allocate hip memory for src/dst
    CHECK_RETURN_STATUS(hipMalloc(&d_input, inputBufferSize));
    CHECK_RETURN_STATUS(hipMalloc(&d_output, outputBufferSize));
    if (dualInputCase) CHECK_RETURN_STATUS(hipMalloc(&d_input_second, inputBufferSize));

    RpptROI* roiPtrInputCropRegion;
    if (testCase == RICAP)
        CHECK_RETURN_STATUS(hipHostMalloc(&roiPtrInputCropRegion, 4 * sizeof(RpptROI)));

    void *d_rowRemapTable, *d_colRemapTable;
    if (testCase == LENS_CORRECTION || testCase == REMAP) {
        CHECK_RETURN_STATUS(hipMalloc(&d_rowRemapTable, ioBufferSize * sizeof(Rpp32u)));
        CHECK_RETURN_STATUS(hipMalloc(&d_colRemapTable, ioBufferSize * sizeof(Rpp32u)));
        CHECK_RETURN_STATUS(hipMemset(d_rowRemapTable, 0, ioBufferSize * sizeof(Rpp32u)));
        CHECK_RETURN_STATUS(hipMemset(d_colRemapTable, 0, ioBufferSize * sizeof(Rpp32u)));
    }

    Rpp32f *cameraMatrix, *distortionCoeffs;
    if (testCase == LENS_CORRECTION) {
        CHECK_RETURN_STATUS(hipHostMalloc(&cameraMatrix, batchSize * 9 * sizeof(Rpp32f)));
        CHECK_RETURN_STATUS(hipHostMalloc(&distortionCoeffs, batchSize * 8 * sizeof(Rpp32f)));
    }

    Rpp32u boxesInEachImage = 3;
    Rpp32f* colorBuffer;
    RpptRoiLtrb* anchorBoxInfoTensor;
    Rpp32u* numOfBoxes;
    if (testCase == ERASE || testCase == CUTOUT_DROPOUT) {
        CHECK_RETURN_STATUS(hipHostMalloc(
            &colorBuffer, batchSize * boxesInEachImage * srcDescPtr->c * sizeof(Rpp32f)));
        CHECK_RETURN_STATUS(hipMemset(
            colorBuffer, 0, batchSize * boxesInEachImage * srcDescPtr->c * sizeof(Rpp32f)));
        CHECK_RETURN_STATUS(hipHostMalloc(&anchorBoxInfoTensor,
                                          batchSize * boxesInEachImage * sizeof(RpptRoiLtrb)));
        CHECK_RETURN_STATUS(hipHostMalloc(&numOfBoxes, batchSize * sizeof(Rpp32u)));
    }
    Rpp32u numGridsPerColumn = 10, numGridsPerRow = 10;
    if (testCase == GRID_DROPOUT)
        CHECK_RETURN_STATUS(
            hipHostMalloc(&anchorBoxInfoTensor,
                          batchSize * numGridsPerRow * numGridsPerColumn * sizeof(RpptRoiLtrb)));

    // create cropRoi and patchRoi in case of crop_and_patch
    RpptROI *cropRoi, *patchRoi;
    if (testCase == CROP_AND_PATCH) {
        CHECK_RETURN_STATUS(hipHostMalloc(&cropRoi, batchSize * sizeof(RpptROI)));
        CHECK_RETURN_STATUS(hipHostMalloc(&patchRoi, batchSize * sizeof(RpptROI)));
    }
    bool invalidROI = (roiList[0] == 0 && roiList[1] == 0 && roiList[2] == 0 && roiList[3] == 0);

    Rpp32f* intensity;
    if (testCase == VIGNETTE)
        CHECK_RETURN_STATUS(hipHostMalloc(&intensity, batchSize * sizeof(Rpp32f)));

    Rpp32f* intensityFactor = nullptr;
    Rpp32f* greyFactor = nullptr;
    if (testCase == 10) {
        CHECK_RETURN_STATUS(hipHostMalloc(&intensityFactor, batchSize * sizeof(Rpp32f)));
        CHECK_RETURN_STATUS(hipHostMalloc(&greyFactor, batchSize * sizeof(Rpp32f)));
    }

    Rpp32f* brightnessCoefficient = nullptr;
    Rpp32f* snowThreshold = nullptr;
    Rpp32s* darkMode = nullptr;
    if (testCase == SNOW) {
        CHECK_RETURN_STATUS(hipHostMalloc(&brightnessCoefficient, batchSize * sizeof(Rpp32f)));
        CHECK_RETURN_STATUS(hipHostMalloc(&snowThreshold, batchSize * sizeof(Rpp32f)));
        CHECK_RETURN_STATUS(hipHostMalloc(&darkMode, batchSize * sizeof(Rpp32s)));
    }

    Rpp32u* kernelSizeTensor;
    if (testCase == JITTER)
        CHECK_RETURN_STATUS(hipHostMalloc(&kernelSizeTensor, batchSize * sizeof(Rpp32u)));

    RpptChannelOffsets* rgbOffsets;
    if (testCase == GLITCH)
        CHECK_RETURN_STATUS(hipHostMalloc(&rgbOffsets, batchSize * sizeof(RpptChannelOffsets)));

    void* d_interDstPtr;
    if (testCase == PIXELATE)
        CHECK_RETURN_STATUS(hipHostMalloc(
            &d_interDstPtr, srcDescPtr->strides.nStride * srcDescPtr->n * sizeof(Rpp32f)));

    Rpp32f* perspectiveTensorPtr = NULL;
    if (testCase == WARP_PERSPECTIVE)
        CHECK_RETURN_STATUS(hipHostMalloc(&perspectiveTensorPtr, batchSize * 9 * sizeof(Rpp32f)));

    Rpp32f* affineTensorPtr = NULL;
    if (testCase == WARP_AFFINE)
        CHECK_RETURN_STATUS(hipHostMalloc(&affineTensorPtr, batchSize * 6 * sizeof(Rpp32f)));

    Rpp32f* alpha = nullptr;
    if (testCase == RAIN) CHECK_RETURN_STATUS(hipHostMalloc(&alpha, batchSize * sizeof(Rpp32f)));

    if (testCase == BLEND) CHECK_RETURN_STATUS(hipHostMalloc(&alpha, batchSize * sizeof(Rpp32f)));

    Rpp32f* beta = nullptr;
    if (testCase == BRIGHTNESS) {
        CHECK_RETURN_STATUS(hipHostMalloc(&alpha, batchSize * sizeof(Rpp32f)));
        CHECK_RETURN_STATUS(hipHostMalloc(&beta, batchSize * sizeof(Rpp32f)));
    }

    RpptRGB* rgbTensor = nullptr;
    if (testCase == COLOR_CAST) {
        CHECK_RETURN_STATUS(hipHostMalloc(&alpha, batchSize * sizeof(Rpp32f)));
        CHECK_RETURN_STATUS(hipHostMalloc(&rgbTensor, batchSize * sizeof(RpptRGB)));
    }

    Rpp32f* brightness = nullptr;
    Rpp32f* contrast = nullptr;
    Rpp32f* hue = nullptr;
    Rpp32f* saturation = nullptr;
    if (testCase == COLOR_TWIST) {
        CHECK_RETURN_STATUS(hipHostMalloc(&brightness, batchSize * sizeof(Rpp32f)));
        CHECK_RETURN_STATUS(hipHostMalloc(&contrast, batchSize * sizeof(Rpp32f)));
        CHECK_RETURN_STATUS(hipHostMalloc(&hue, batchSize * sizeof(Rpp32f)));
        CHECK_RETURN_STATUS(hipHostMalloc(&saturation, batchSize * sizeof(Rpp32f)));
    }

    Rpp32f* contrastFactor = nullptr;
    Rpp32f* contrastCenter = nullptr;
    if (testCase == CONTRAST) {
        CHECK_RETURN_STATUS(hipHostMalloc(&contrastFactor, batchSize * sizeof(Rpp32f)));
        CHECK_RETURN_STATUS(hipHostMalloc(&contrastCenter, batchSize * sizeof(Rpp32f)));
    }

    Rpp32f* multiplier = nullptr;
    Rpp32f* offset = nullptr;
    Rpp32u* mirror = nullptr;
    if (testCase == CROP_MIRROR_NORMALIZE) {
        CHECK_RETURN_STATUS(hipHostMalloc(&multiplier, batchSize * srcDescPtr->c * sizeof(Rpp32f)));
        CHECK_RETURN_STATUS(hipHostMalloc(&offset, batchSize * srcDescPtr->c * sizeof(Rpp32f)));
        CHECK_RETURN_STATUS(hipHostMalloc(&mirror, batchSize * sizeof(Rpp32u)));
    }

    Rpp32f* exposureFactor = nullptr;
    if (testCase == EXPOSURE)
        CHECK_RETURN_STATUS(hipHostMalloc(&exposureFactor, batchSize * sizeof(Rpp32f)));

    Rpp32f* gammaVal = nullptr;
    if (testCase == GAMMA_CORRECTION)
        CHECK_RETURN_STATUS(hipHostMalloc(&gammaVal, batchSize * sizeof(Rpp32f)));

    Rpp32f* stdDevTensor = nullptr;
    if (testCase == GAUSSIAN_FILTER)
        CHECK_RETURN_STATUS(hipHostMalloc(&stdDevTensor, batchSize * sizeof(Rpp32f)));

    Rpp32f* noiseProbabilityTensor = nullptr;
    Rpp32f* saltProbabilityTensor = nullptr;
    Rpp32f* saltValueTensor = nullptr;
    Rpp32f* pepperValueTensor = nullptr;
    Rpp32f* meanTensor = nullptr;
    Rpp32f* shotNoiseFactorTensor = nullptr;
    if (testCase == NOISE) {
        if (additionalParam == 0) {
            CHECK_RETURN_STATUS(hipHostMalloc(&noiseProbabilityTensor, batchSize * sizeof(Rpp32f)));
            CHECK_RETURN_STATUS(hipHostMalloc(&saltProbabilityTensor, batchSize * sizeof(Rpp32f)));
            CHECK_RETURN_STATUS(hipHostMalloc(&saltValueTensor, batchSize * sizeof(Rpp32f)));
            CHECK_RETURN_STATUS(hipHostMalloc(&pepperValueTensor, batchSize * sizeof(Rpp32f)));
        } else if (additionalParam == 1) {
            CHECK_RETURN_STATUS(hipHostMalloc(&meanTensor, batchSize * sizeof(Rpp32f)));
            CHECK_RETURN_STATUS(hipHostMalloc(&stdDevTensor, batchSize * sizeof(Rpp32f)));
        } else if (additionalParam == 2)
            CHECK_RETURN_STATUS(hipHostMalloc(&shotNoiseFactorTensor, batchSize * sizeof(Rpp32f)));
    }

    if (testCase == NON_LINEAR_BLEND)
        CHECK_RETURN_STATUS(hipHostMalloc(&stdDevTensor, batchSize * sizeof(Rpp32f)));

    if (testCase == RESIZE_MIRROR_NORMALIZE) {
        CHECK_RETURN_STATUS(hipHostMalloc(&meanTensor, batchSize * 3 * sizeof(Rpp32f)));
        CHECK_RETURN_STATUS(hipHostMalloc(&stdDevTensor, batchSize * 3 * sizeof(Rpp32f)));
        CHECK_RETURN_STATUS(hipHostMalloc(&mirror, batchSize * sizeof(Rpp32u)));
    }

    Rpp32f* amplX = nullptr;
    Rpp32f* amplY = nullptr;
    Rpp32f* freqX = nullptr;
    Rpp32f* freqY = nullptr;
    Rpp32f* phaseX = nullptr;
    Rpp32f* phaseY = nullptr;
    if (testCase == WATER) {
        CHECK_RETURN_STATUS(hipHostMalloc(&amplX, batchSize * sizeof(Rpp32f)));
        CHECK_RETURN_STATUS(hipHostMalloc(&amplY, batchSize * sizeof(Rpp32f)));
        CHECK_RETURN_STATUS(hipHostMalloc(&freqX, batchSize * sizeof(Rpp32f)));
        CHECK_RETURN_STATUS(hipHostMalloc(&freqY, batchSize * sizeof(Rpp32f)));
        CHECK_RETURN_STATUS(hipHostMalloc(&phaseX, batchSize * sizeof(Rpp32f)));
        CHECK_RETURN_STATUS(hipHostMalloc(&phaseY, batchSize * sizeof(Rpp32f)));
    }

    Rpp32u* horizontalFlag = nullptr;
    Rpp32u* verticalFlag = nullptr;
    if (testCase == FLIP) {
        CHECK_RETURN_STATUS(hipHostMalloc(&horizontalFlag, batchSize * sizeof(Rpp32u)));
        CHECK_RETURN_STATUS(hipHostMalloc(&verticalFlag, batchSize * sizeof(Rpp32u)));
    }

    if (testCase == RESIZE_CROP_MIRROR)
        CHECK_RETURN_STATUS(hipHostMalloc(&mirror, batchSize * sizeof(Rpp32u)));

    Rpp32s* adjustment = nullptr;
    if (testCase == COLOR_TEMPERATURE)
        CHECK_RETURN_STATUS(hipHostMalloc(&adjustment, batchSize * sizeof(Rpp32s)));

    Rpp32f* hueShift = nullptr;
    if (testCase == HUE) CHECK_RETURN_STATUS(hipHostMalloc(&hueShift, batchSize * sizeof(Rpp32f)));

    Rpp32f* saturationFactor = nullptr;
    if (testCase == SATURATION)
        CHECK_RETURN_STATUS(hipHostMalloc(&saturationFactor, batchSize * sizeof(Rpp32f)));

    Rpp32f* strength = nullptr;
    if (testCase == EMBOSS)
        CHECK_RETURN_STATUS(hipHostMalloc(&strength, batchSize * sizeof(Rpp32f)));

    Rpp32f *minTensor = nullptr, *maxTensor = nullptr;
    if (testCase == THRESHOLD) {
        CHECK_RETURN_STATUS(hipHostMalloc(&minTensor, batchSize * srcDescPtr->c * sizeof(Rpp32f)));
        CHECK_RETURN_STATUS(hipHostMalloc(&maxTensor, batchSize * srcDescPtr->c * sizeof(Rpp32f)));
    }

    Rpp32f* thresholdTensor = nullptr;
    if (testCase == SOLARIZE)
        CHECK_RETURN_STATUS(hipHostMalloc(&thresholdTensor, batchSize * sizeof(Rpp32f)));
    Rpp8u* posterizeLevelBits = nullptr;
    if (testCase == POSTERIZE)
        CHECK_RETURN_STATUS(hipHostMalloc(&posterizeLevelBits, batchSize * sizeof(Rpp8u)));
    Rpp32f* angle = nullptr;
    if (testCase == ROTATE) CHECK_RETURN_STATUS(hipHostMalloc(&angle, batchSize * sizeof(Rpp32f)));

    Rpp32s* qualityTensor = nullptr;
    if (testCase == JPEG_COMPRESSION_DISTORTION)
        CHECK_RETURN_STATUS(hipHostMalloc(&qualityTensor, batchSize * sizeof(Rpp32s)));

    Rpp32u* permutationTensor = nullptr;
    if (testCase == CHANNEL_PERMUTE)
        CHECK_RETURN_STATUS(hipHostMalloc(&permutationTensor, 3 * batchSize * sizeof(Rpp32u)));
    if (testCase == RICAP)
        CHECK_RETURN_STATUS(hipHostMalloc(&permutationTensor, 4 * batchSize * sizeof(Rpp32u)));

    Rpp8u* dropoutTensor = nullptr;
    if (testCase == CHANNEL_DROPOUT)
        CHECK_RETURN_STATUS(
            hipHostMalloc(&dropoutTensor, batchSize * srcDescPtr->c * sizeof(Rpp8u)));

    Rpp32u maxBoxesPerImage;
    if (testCase == COARSE_DROPOUT) {
        maxBoxesPerImage = 8;
        CHECK_RETURN_STATUS(hipHostMalloc(&anchorBoxInfoTensor,
                                          batchSize * maxBoxesPerImage * sizeof(RpptRoiLtrb)));
        CHECK_RETURN_STATUS(hipHostMalloc(&numOfBoxes, batchSize * sizeof(Rpp32u)));
    }
    if (testCase == RANDOM_ERASE) {
        boxesInEachImage = 1;
        CHECK_RETURN_STATUS(hipHostMalloc(&colorBuffer, RANDOM_ERASE_NOISE_BUFFER_SIDE *
                                                            RANDOM_ERASE_NOISE_BUFFER_SIDE *
                                                            srcDescPtr->c * sizeof(Rpp32f)));
        CHECK_RETURN_STATUS(hipHostMalloc(&anchorBoxInfoTensor,
                                          batchSize * boxesInEachImage * sizeof(RpptRoiLtrb)));
    }

    // case-wise RPP API and measure time script for Unit and Performance test
    cout << "\nRunning " << func << " " << numRuns << " times (each time with a batch size of "
         << batchSize << " images) and computing mean statistics...";
    for (int iterCount = 0; iterCount < noOfIterations; iterCount++) {
        vector<string>::const_iterator imagesPathStart =
            imageNamesPath.begin() + (iterCount * batchSize);
        vector<string>::const_iterator imagesPathEnd = imagesPathStart + batchSize;
        vector<string>::const_iterator imageNamesStart =
            imageNames.begin() + (iterCount * batchSize);
        vector<string>::const_iterator imageNamesEnd = imageNamesStart + batchSize;
        vector<string>::const_iterator imagesPathSecondStart =
            imageNamesPathSecond.begin() + (iterCount * batchSize);
        vector<string>::const_iterator imagesPathSecondEnd = imagesPathSecondStart + batchSize;

        // Set ROIs for src/dst
        if ((testCase == YUV_TO_RGB || testCase == YUV_TO_RGB_CUBIC_V ||
             testCase == YUV_TO_RGB_LINEAR_V))
            set_src_and_dst_roi_yuv(imagesPathStart, imagesPathEnd, roiTensorPtrSrc,
                                    roiTensorPtrDst, dstImgSizes);
        else
            set_src_and_dst_roi(imagesPathStart, imagesPathEnd, roiTensorPtrSrc, roiTensorPtrDst,
                                dstImgSizes, decoderType);

        // Read images
        if ((testCase == YUV_TO_RGB || testCase == YUV_TO_RGB_CUBIC_V ||
             testCase == YUV_TO_RGB_LINEAR_V)) {
            read_yuv_batch_nv12(inputu8, srcDescPtr, imagesPathStart);
            CHECK_RETURN_STATUS(
                hipMemcpy(d_input, inputu8, inputBufferSize, hipMemcpyHostToDevice));
        } else {
            if (decoderType == 0) read_image_batch_packed(inputu8, srcDescPtr, imagesPathStart);
#if defined(RPP_TEST_SUITE_HAVE_OPENCV) && RPP_TEST_SUITE_HAVE_OPENCV
            else
                read_image_batch_opencv(inputu8, srcDescPtr, imagesPathStart);
#else
            else if (decoderType == 1) {
                cerr << "\nError: decoder_type 1 (OpenCV) requested but Tensor_image was built "
                        "without OpenCV (RPP_TEST_SUITE_HAVE_OPENCV). Aborting.\n";
                exit(1);
            } else {
                cerr << "\nError: invalid decoder_type (expected 0 = packed .rgb or 1 = OpenCV). "
                        "Aborting.\n";
                exit(1);
            }
#endif

            // if the input layout requested is PLN3, convert PKD3 inputs to PLN3 for first and
            // second input batch
            if (layoutType == 1) convert_pkd3_to_pln3(inputu8, srcDescPtr);

            if (dualInputCase) {
                if (decoderType == 0)
                    read_image_batch_packed(inputu8Second, srcDescPtr, imagesPathSecondStart);
#if defined(RPP_TEST_SUITE_HAVE_OPENCV) && RPP_TEST_SUITE_HAVE_OPENCV
                else
                    read_image_batch_opencv(inputu8Second, srcDescPtr, imagesPathSecondStart);
#else
                else if (decoderType == 1) {
                    cerr << "\nError: decoder_type 1 (OpenCV) requested but Tensor_image was built "
                            "without OpenCV. Aborting.\n";
                    exit(1);
                } else {
                    cerr << "\nError: invalid decoder_type (expected 0 = packed .rgb or 1 = "
                            "OpenCV). Aborting.\n";
                    exit(1);
                }
#endif
                if (layoutType == 1) convert_pkd3_to_pln3(inputu8Second, srcDescPtr);
            }

            // Convert inputs to correponding bit depth specified by user
            convert_input_bitdepth(input, input_second, inputu8, inputu8Second, BitDepthTestMode,
                                   ioBufferSize, inputBufferSize, srcDescPtr, dualInputCase,
                                   conversionFactor);

            // copy decoded inputs to hip buffers
            CHECK_RETURN_STATUS(hipMemcpy(d_input, input, inputBufferSize, hipMemcpyHostToDevice));
        }
        CHECK_RETURN_STATUS(hipMemcpy(d_output, output, outputBufferSize, hipMemcpyHostToDevice));
        if (dualInputCase)
            CHECK_RETURN_STATUS(
                hipMemcpy(d_input_second, input_second, inputBufferSize, hipMemcpyHostToDevice));

        int roiHeightList[batchSize], roiWidthList[batchSize];
        if (invalidROI) {
            for (int i = 0; i < batchSize; i++) {
                roiList[0] = 10;
                roiList[1] = 10;
                roiWidthList[i] = roiTensorPtrSrc[i].xywhROI.roiWidth / 2;
                roiHeightList[i] = roiTensorPtrSrc[i].xywhROI.roiHeight / 2;
            }
        } else {
            for (int i = 0; i < batchSize; i++) {
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

        // Save the XYWH ROI before the performance loop. The HIP executor converts
        // roiTensorPtrSrc in-place from XYWH to LTRB (it is pinned host memory, so the
        // GPU write is immediately visible on the host). Without saving and restoring,
        // every subsequent rppt_* call would re-convert already-LTRB data, progressively
        // corrupting rb.x / rb.y and causing GPU out-of-bounds memory accesses.
        vector<RpptROI> savedRoiTensorPtrSrc(batchSize);
        memcpy(savedRoiTensorPtrSrc.data(), roiTensorPtrSrc, batchSize * sizeof(RpptROI));

        for (int perfRunCount = 0; perfRunCount < numRuns; perfRunCount++) {
            // Restore the original XYWH ROI before each API call so the in-place
            // XYWH→LTRB conversion inside the executor always starts from valid data.
            memcpy(roiTensorPtrSrc, savedRoiTensorPtrSrc.data(), batchSize * sizeof(RpptROI));

            RppStatus errorCodeCapture = RPP_SUCCESS;
            double startWallTime, endWallTime;
            switch (testCase) {
                case BRIGHTNESS: {
                    testCaseName = "brightness";

                    for (i = 0; i < batchSize; i++) {
                        alpha[i] = 1.75;
                        beta[i] = 50;
                    }

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_brightness(d_input, srcDescPtr, d_output, dstDescPtr, alpha, beta,
                                            roiTensorPtrSrc, roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case GAMMA_CORRECTION: {
                    testCaseName = "gamma_correction";

                    for (i = 0; i < batchSize; i++) gammaVal[i] = 1.9;

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_gamma_correction(
                            d_input, srcDescPtr, d_output, dstDescPtr, gammaVal, roiTensorPtrSrc,
                            roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case BLEND: {
                    testCaseName = "blend";

                    for (i = 0; i < batchSize; i++) alpha[i] = 0.4;

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_blend(d_input, d_input_second, srcDescPtr, d_output, dstDescPtr,
                                       alpha, roiTensorPtrSrc, roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case CONTRAST: {
                    testCaseName = "contrast";

                    for (i = 0; i < batchSize; i++) {
                        contrastFactor[i] = 2.96;
                        contrastCenter[i] = 128;
                    }

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_contrast(
                            d_input, srcDescPtr, d_output, dstDescPtr, contrastFactor,
                            contrastCenter, roiTensorPtrSrc, roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case PIXELATE: {
                    testCaseName = "pixelate";

                    Rpp32f pixelationPercentage = 87.5;

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_pixelate(d_input, srcDescPtr, d_output, dstDescPtr, d_interDstPtr,
                                          pixelationPercentage, roiTensorPtrSrc, roiTypeSrc, handle,
                                          RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case JITTER: {
                    testCaseName = "jitter";

                    Rpp32u seed = 1255459;
                    for (i = 0; i < batchSize; i++) kernelSizeTensor[i] = 5;

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_jitter(d_input, srcDescPtr, d_output, dstDescPtr, kernelSizeTensor,
                                        seed, roiTensorPtrSrc, roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case SNOW: {
                    testCaseName = "snow";

                    for (i = 0; i < batchSize; i++) {
                        brightnessCoefficient[i] = 2.5f;
                        snowThreshold[i] = 1.0f;
                        darkMode[i] = 0;
                    }

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_snow(d_input, srcDescPtr, d_output, dstDescPtr,
                                      brightnessCoefficient, snowThreshold, darkMode,
                                      roiTensorPtrSrc, roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case NOISE: {
                    testCaseName = "noise";

                    switch (additionalParam) {
                        case 0: {
                            Rpp32u seed = 1255459;
                            for (i = 0; i < batchSize; i++) {
                                noiseProbabilityTensor[i] = 0.1f;
                                saltProbabilityTensor[i] = 0.5f;
                                saltValueTensor[i] = 1.0f;
                                pepperValueTensor[i] = 0.0f;
                            }

                            startWallTime = omp_get_wtime();
                            if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                                BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                                errorCodeCapture = rppt_salt_and_pepper_noise(
                                    d_input, srcDescPtr, d_output, dstDescPtr,
                                    noiseProbabilityTensor, saltProbabilityTensor, saltValueTensor,
                                    pepperValueTensor, seed, roiTensorPtrSrc, roiTypeSrc, handle,
                                    RPP_HIP_BACKEND);
                            else
                                missingFuncFlag = 1;

                            break;
                        }
                        case 1: {
                            Rpp32u seed = 1255459;
                            for (i = 0; i < batchSize; i++) {
                                meanTensor[i] = 0.0f;
                                stdDevTensor[i] = 0.2f;
                            }

                            startWallTime = omp_get_wtime();
                            if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                                BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                                errorCodeCapture = rppt_gaussian_noise(
                                    d_input, srcDescPtr, d_output, dstDescPtr, meanTensor,
                                    stdDevTensor, seed, roiTensorPtrSrc, roiTypeSrc, handle,
                                    RPP_HIP_BACKEND);
                            else
                                missingFuncFlag = 1;

                            break;
                        }
                        case 2: {
                            Rpp32u seed = 1255459;
                            for (i = 0; i < batchSize; i++) shotNoiseFactorTensor[i] = 80.0f;

                            startWallTime = omp_get_wtime();
                            if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                                BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                                errorCodeCapture =
                                    rppt_shot_noise(d_input, srcDescPtr, d_output, dstDescPtr,
                                                    shotNoiseFactorTensor, seed, roiTensorPtrSrc,
                                                    roiTypeSrc, handle, RPP_HIP_BACKEND);
                            else
                                missingFuncFlag = 1;

                            break;
                        }
                        default: {
                            missingFuncFlag = 1;
                            break;
                        }
                    }

                    break;
                }
                case FOG: {
                    testCaseName = "fog";

                    for (i = 0; i < batchSize; i++) {
                        intensityFactor[i] = 0;
                        greyFactor[i] = 0.3;
                    }

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_fog(d_input, srcDescPtr, d_output, dstDescPtr,
                                                    intensityFactor, greyFactor, roiTensorPtrSrc,
                                                    roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case EXPOSURE: {
                    testCaseName = "exposure";

                    for (i = 0; i < batchSize; i++) exposureFactor[i] = 1.4;

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_exposure(d_input, srcDescPtr, d_output, dstDescPtr, exposureFactor,
                                          roiTensorPtrSrc, roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case RAIN: {
                    testCaseName = "rain";

                    Rpp32f rainPercentage = 7;
                    Rpp32u rainHeight = 6;
                    Rpp32u rainWidth = 1;
                    Rpp32f slantAngle = 0;
                    for (int i = 0; i < batchSize; i++) alpha[i] = 0.4;

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_rain(d_input, srcDescPtr, d_output, dstDescPtr, rainPercentage,
                                      rainWidth, rainHeight, slantAngle, alpha, roiTensorPtrSrc,
                                      roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case THRESHOLD: {
                    testCaseName = "threshold";
                    Rpp32f normFactor = 1;
                    Rpp32f subtractionFactor = 0;
                    if (BitDepthTestMode == F16_TO_F16 || BitDepthTestMode == F32_TO_F32)
                        normFactor = 255;
                    else if (BitDepthTestMode == I8_TO_I8)
                        subtractionFactor = 128;

                    for (int i = 0; i < batchSize; i++) {
                        for (int j = 0, k = i * srcDescPtr->c; j < srcDescPtr->c; j++, k++) {
                            minTensor[k] = (30 / normFactor) - subtractionFactor;
                            maxTensor[k] = (100 / normFactor) - subtractionFactor;
                        }
                    }

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_threshold(d_input, srcDescPtr, d_output, dstDescPtr,
                                                          minTensor, maxTensor, roiTensorPtrSrc,
                                                          roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case FLIP: {
                    testCaseName = "flip";

                    for (i = 0; i < batchSize; i++) {
                        horizontalFlag[i] = 1;
                        verticalFlag[i] = 0;
                    }

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_flip(d_input, srcDescPtr, d_output, dstDescPtr,
                                                     horizontalFlag, verticalFlag, roiTensorPtrSrc,
                                                     roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case RESIZE: {
                    testCaseName = "resize";

                    for (i = 0; i < batchSize; i++) {
                        dstImgSizes[i].width = roiTensorPtrDst[i].xywhROI.roiWidth =
                            roiTensorPtrSrc[i].xywhROI.roiWidth / 2;
                        dstImgSizes[i].height = roiTensorPtrDst[i].xywhROI.roiHeight =
                            roiTensorPtrSrc[i].xywhROI.roiHeight / 2;
                    }

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_resize(d_input, srcDescPtr, d_output, dstDescPtr, dstImgSizes,
                                        interpolationType, roiTensorPtrDst, roiTypeSrc, handle,
                                        RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case ROTATE: {
                    testCaseName = "rotate";

                    if ((interpolationType != RpptInterpolationType::BILINEAR) &&
                        (interpolationType != RpptInterpolationType::NEAREST_NEIGHBOR)) {
                        missingFuncFlag = 1;
                        break;
                    }

                    for (i = 0; i < batchSize; i++) angle[i] = 50;

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_rotate(d_input, srcDescPtr, d_output, dstDescPtr,
                                                       angle, interpolationType, roiTensorPtrSrc,
                                                       roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case WARP_AFFINE: {
                    testCaseName = "warp_affine";

                    if ((interpolationType != RpptInterpolationType::BILINEAR) &&
                        (interpolationType != RpptInterpolationType::NEAREST_NEIGHBOR)) {
                        missingFuncFlag = 1;
                        break;
                    }

                    for (i = 0, j = 0; i < batchSize; i++, j += 6) {
                        affineTensorPtr[j + 0] = 1.23;
                        affineTensorPtr[j + 1] = 0.5;
                        affineTensorPtr[j + 2] = 0;
                        affineTensorPtr[j + 3] = -0.8;
                        affineTensorPtr[j + 4] = 0.83;
                        affineTensorPtr[j + 5] = 0;
                    }

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_warp_affine(d_input, srcDescPtr, d_output, dstDescPtr,
                                             affineTensorPtr, interpolationType, roiTensorPtrSrc,
                                             roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case FISHEYE: {
                    testCaseName = "fisheye";

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_fisheye(d_input, srcDescPtr, d_output, dstDescPtr, roiTensorPtrSrc,
                                         roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case LENS_CORRECTION: {
                    testCaseName = "lens_correction";

                    RpptDesc tableDesc = srcDesc;
                    RpptDescPtr tableDescPtr = &tableDesc;
                    init_lens_correction(batchSize, srcDescPtr, cameraMatrix, distortionCoeffs,
                                         tableDescPtr);

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_lens_correction(
                            d_input, srcDescPtr, d_output, dstDescPtr,
                            static_cast<Rpp32f*>(d_rowRemapTable),
                            static_cast<Rpp32f*>(d_colRemapTable), tableDescPtr, cameraMatrix,
                            distortionCoeffs, roiTensorPtrSrc, roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case WARP_PERSPECTIVE: {
                    testCaseName = "warp_perspective";

                    if ((interpolationType != RpptInterpolationType::BILINEAR) &&
                        (interpolationType != RpptInterpolationType::NEAREST_NEIGHBOR)) {
                        missingFuncFlag = 1;
                        break;
                    }

                    for (i = 0, j = 0; i < batchSize; i++, j += 9) {
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
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_warp_perspective(
                            d_input, srcDescPtr, d_output, dstDescPtr, perspectiveTensorPtr,
                            interpolationType, roiTensorPtrSrc, roiTypeSrc, handle,
                            RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case WATER: {
                    testCaseName = "water";

                    for (i = 0; i < batchSize; i++) {
                        amplX[i] = 2.0f;
                        amplY[i] = 5.0f;
                        freqX[i] = 5.8f;
                        freqY[i] = 1.2f;
                        phaseX[i] = 10.0f;
                        phaseY[i] = 15.0f;
                    }

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_water(
                            d_input, srcDescPtr, d_output, dstDescPtr, amplX, amplY, freqX, freqY,
                            phaseX, phaseY, roiTensorPtrSrc, roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case NON_LINEAR_BLEND: {
                    testCaseName = "non_linear_blend";

                    for (i = 0; i < batchSize; i++) stdDevTensor[i] = 50.0;

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_non_linear_blend(
                            d_input, d_input_second, srcDescPtr, d_output, dstDescPtr, stdDevTensor,
                            roiTensorPtrSrc, roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case COLOR_CAST: {
                    testCaseName = "color_cast";

                    for (i = 0; i < batchSize; i++) {
                        rgbTensor[i].R = 0;
                        rgbTensor[i].G = 0;
                        rgbTensor[i].B = 100;
                        alpha[i] = 0.5;
                    }

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_color_cast(
                            d_input, srcDescPtr, d_output, dstDescPtr, rgbTensor, alpha,
                            roiTensorPtrSrc, roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case ERASE: {
                    testCaseName = "erase";

                    init_erase(batchSize, boxesInEachImage, numOfBoxes, anchorBoxInfoTensor,
                               roiTensorPtrSrc, srcDescPtr->c, colorBuffer, BitDepthTestMode);
                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_erase(d_input, srcDescPtr, d_output, dstDescPtr,
                                       anchorBoxInfoTensor, colorBuffer, numOfBoxes,
                                       roiTensorPtrSrc, roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case CROP_AND_PATCH: {
                    testCaseName = "crop_and_patch";
                    for (i = 0; i < batchSize; i++) {
                        cropRoi[i].xywhROI.xy.x = patchRoi[i].xywhROI.xy.x = roiList[0];
                        cropRoi[i].xywhROI.xy.y = patchRoi[i].xywhROI.xy.y = roiList[1];
                        cropRoi[i].xywhROI.roiWidth = patchRoi[i].xywhROI.roiWidth =
                            roiWidthList[i];
                        cropRoi[i].xywhROI.roiHeight = patchRoi[i].xywhROI.roiHeight =
                            roiHeightList[i];
                    }

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_crop_and_patch(d_input, d_input_second, srcDescPtr, d_output,
                                                dstDescPtr, roiTensorPtrSrc, cropRoi, patchRoi,
                                                roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case LOOK_UP_TABLE: {
                    testCaseName = "lut";

                    Rpp32f* lutBuffer;
                    CHECK_RETURN_STATUS(hipHostMalloc(&lutBuffer, 65536 * sizeof(Rpp32f)));
                    CHECK_RETURN_STATUS(hipMemset(lutBuffer, 0, 65536 * sizeof(Rpp32f)));
                    Rpp8u* lut8u = reinterpret_cast<Rpp8u*>(lutBuffer);
                    Rpp16f* lut16f = reinterpret_cast<Rpp16f*>(lutBuffer);
                    Rpp32f* lut32f = reinterpret_cast<Rpp32f*>(lutBuffer);
                    Rpp8s* lut8s = reinterpret_cast<Rpp8s*>(lutBuffer);
                    if (BitDepthTestMode == U8_TO_U8)
                        for (j = 0; j < 256; j++) lut8u[j] = (Rpp8u)(255 - j);
                    else if (BitDepthTestMode == U8_TO_F16)
                        for (j = 0; j < 256; j++) lut16f[j] = (Rpp16f)((255 - j) * ONE_OVER_255);
                    else if (BitDepthTestMode == U8_TO_F32)
                        for (j = 0; j < 256; j++) lut32f[j] = (Rpp32f)((255 - j) * ONE_OVER_255);
                    else if (BitDepthTestMode == I8_TO_I8)
                        for (j = 0; j < 256; j++) lut8s[j] = (Rpp8s)(255 - j - 128);

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8)
                        errorCodeCapture =
                            rppt_lut(d_input, srcDescPtr, d_output, dstDescPtr, lut8u,
                                     roiTensorPtrSrc, roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else if (BitDepthTestMode == U8_TO_F16)
                        errorCodeCapture =
                            rppt_lut(d_input, srcDescPtr, d_output, dstDescPtr, lut16f,
                                     roiTensorPtrSrc, roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else if (BitDepthTestMode == U8_TO_F32)
                        errorCodeCapture =
                            rppt_lut(d_input, srcDescPtr, d_output, dstDescPtr, lut32f,
                                     roiTensorPtrSrc, roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else if (BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_lut(d_input, srcDescPtr, d_output, dstDescPtr, lut8s,
                                     roiTensorPtrSrc, roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;

                    CHECK_RETURN_STATUS(hipHostFree(lutBuffer));
                }
                case GLITCH: {
                    testCaseName = "glitch";

                    for (i = 0; i < batchSize; i++) {
                        rgbOffsets[i].r.x = 10;
                        rgbOffsets[i].r.y = 10;
                        rgbOffsets[i].g.x = 0;
                        rgbOffsets[i].g.y = 0;
                        rgbOffsets[i].b.x = 5;
                        rgbOffsets[i].b.y = 5;
                    }

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_glitch(d_input, srcDescPtr, d_output, dstDescPtr, rgbOffsets,
                                        roiTensorPtrSrc, roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case COLOR_TWIST: {
                    testCaseName = "color_twist";

                    for (i = 0; i < batchSize; i++) {
                        brightness[i] = 1.4;
                        contrast[i] = 0.0;
                        hue[i] = 60.0;
                        saturation[i] = 1.9;
                    }

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_color_twist(
                            d_input, srcDescPtr, d_output, dstDescPtr, brightness, contrast, hue,
                            saturation, roiTensorPtrSrc, roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case HUE: {
                    testCaseName = "hue";

                    for (i = 0; i < batchSize; i++) hueShift[i] = 60.0;

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_hue(d_input, srcDescPtr, d_output, dstDescPtr, hueShift,
                                     roiTensorPtrSrc, roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case SATURATION: {
                    testCaseName = "saturation";

                    for (i = 0; i < batchSize; i++) saturationFactor[i] = 5;

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_saturation(
                            d_input, srcDescPtr, d_output, dstDescPtr, saturationFactor,
                            roiTensorPtrSrc, roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case CROP: {
                    testCaseName = "crop";

                    for (i = 0; i < batchSize; i++) {
                        roiTensorPtrDst[i].xywhROI.xy.x = roiList[0];
                        roiTensorPtrDst[i].xywhROI.xy.y = roiList[1];
                        dstImgSizes[i].width = roiTensorPtrDst[i].xywhROI.roiWidth =
                            roiWidthList[i];
                        dstImgSizes[i].height = roiTensorPtrDst[i].xywhROI.roiHeight =
                            roiHeightList[i];
                    }

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_crop(d_input, srcDescPtr, d_output, dstDescPtr, roiTensorPtrDst,
                                      roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case CROP_MIRROR_NORMALIZE: {
                    testCaseName = "crop_mirror_normalize";

                    if (srcDescPtr->c == 3) {
                        Rpp32f meanParam[3] = {60.0f, 80.0f, 100.0f};
                        Rpp32f stdDevParam[3] = {0.9f, 0.9f, 0.9f};
                        Rpp32f offsetParam[3] = {-meanParam[0] / stdDevParam[0],
                                                 -meanParam[1] / stdDevParam[1],
                                                 -meanParam[2] / stdDevParam[2]};
                        Rpp32f multiplierParam[3] = {1.0f / stdDevParam[0], 1.0f / stdDevParam[1],
                                                     1.0f / stdDevParam[2]};

                        for (i = 0, j = 0; i < batchSize; i++, j += 3) {
                            multiplier[j] = multiplierParam[0];
                            offset[j] = offsetParam[0];
                            multiplier[j + 1] = multiplierParam[1];
                            offset[j + 1] = offsetParam[1];
                            multiplier[j + 2] = multiplierParam[2];
                            offset[j + 2] = offsetParam[2];
                            mirror[i] = 1;
                        }
                    } else if (srcDescPtr->c == 1) {
                        Rpp32f meanParam = 100.0f;
                        Rpp32f stdDevParam = 0.9f;
                        Rpp32f offsetParam = -meanParam / stdDevParam;
                        Rpp32f multiplierParam = 1.0f / stdDevParam;

                        for (i = 0; i < batchSize; i++) {
                            multiplier[i] = multiplierParam;
                            offset[i] = offsetParam;
                            mirror[i] = 1;
                        }
                    }

                    for (i = 0; i < batchSize; i++) {
                        roiTensorPtrDst[i].xywhROI.xy.x = roiList[0];
                        roiTensorPtrDst[i].xywhROI.xy.y = roiList[1];
                        dstImgSizes[i].width = roiTensorPtrDst[i].xywhROI.roiWidth =
                            roiWidthList[i];
                        dstImgSizes[i].height = roiTensorPtrDst[i].xywhROI.roiHeight =
                            roiHeightList[i];
                    }

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == U8_TO_F16 ||
                        BitDepthTestMode == U8_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_crop_mirror_normalize(
                            d_input, srcDescPtr, d_output, dstDescPtr, offset, multiplier, mirror,
                            roiTensorPtrDst, roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case RESIZE_CROP_MIRROR: {
                    testCaseName = "resize_crop_mirror";

                    if (interpolationType != RpptInterpolationType::BILINEAR) {
                        missingFuncFlag = 1;
                        break;
                    }

                    for (i = 0; i < batchSize; i++) mirror[i] = 1;

                    for (i = 0; i < batchSize; i++) {
                        roiTensorPtrSrc[i].xywhROI.xy.x = 10;
                        roiTensorPtrSrc[i].xywhROI.xy.y = 10;
                        dstImgSizes[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth / 2;
                        dstImgSizes[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight / 2;
                        roiTensorPtrDst[i].xywhROI.roiWidth = 50;
                        roiTensorPtrDst[i].xywhROI.roiHeight = 50;
                    }

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == U8_TO_F16 ||
                        BitDepthTestMode == U8_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_resize_crop_mirror(
                            d_input, srcDescPtr, d_output, dstDescPtr, dstImgSizes,
                            interpolationType, mirror, roiTensorPtrDst, roiTypeSrc, handle,
                            RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case COLOR_TEMPERATURE: {
                    testCaseName = "color_temperature";

                    for (i = 0; i < batchSize; i++) adjustment[i] = 70;

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_color_temperature(
                            d_input, srcDescPtr, d_output, dstDescPtr, adjustment, roiTensorPtrSrc,
                            roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case VIGNETTE: {
                    testCaseName = "vignette";

                    for (i = 0; i < batchSize; i++) intensity[i] = 6;

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_vignette(d_input, srcDescPtr, d_output, dstDescPtr, intensity,
                                          roiTensorPtrSrc, roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case ERODE: {
                    testCaseName = "erode";
                    Rpp32u kernelSize = additionalParam;

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_erode(d_input, srcDescPtr, d_output, dstDescPtr, kernelSize,
                                       roiTensorPtrSrc, roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case DILATE: {
                    testCaseName = "dilate";
                    Rpp32u kernelSize = additionalParam;

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_dilate(d_input, srcDescPtr, d_output, dstDescPtr, kernelSize,
                                        roiTensorPtrSrc, roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case BOX_FILTER: {
                    testCaseName = "box_filter";
                    Rpp32u kernelSize = additionalParam;

                    if (borderType != RpptImageBorderType::REPLICATE) {
                        missingFuncFlag = 1;
                        break;
                    }

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_box_filter(
                            d_input, srcDescPtr, d_output, dstDescPtr, kernelSize, borderType,
                            roiTensorPtrSrc, roiTypeSrc, handle, RppBackend::RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case SOBEL_FILTER: {
                    testCaseName = "sobel_filter";

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_sobel_filter(
                            d_input, srcDescPtr, d_output, dstDescPtr, GradientType, kernelSize,
                            roiTensorPtrSrc, roiTypeSrc, handle, RppBackend::RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case MEDIAN_FILTER: {
                    testCaseName = "median_filter";
                    Rpp32u kernelSize = additionalParam;
                    if (borderType != RpptImageBorderType::REPLICATE) {
                        missingFuncFlag = 1;
                        break;
                    }

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_median_filter(
                            d_input, srcDescPtr, d_output, dstDescPtr, kernelSize, borderType,
                            roiTensorPtrSrc, roiTypeSrc, handle, RppBackend::RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case GAUSSIAN_FILTER: {
                    testCaseName = "gaussian_filter";
                    Rpp32u kernelSize = additionalParam;

                    for (i = 0; i < batchSize; i++) {
                        stdDevTensor[i] = 5.0f;
                    }

                    if (borderType != RpptImageBorderType::REPLICATE) {
                        missingFuncFlag = 1;
                        break;
                    }

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_gaussian_filter(
                            d_input, srcDescPtr, d_output, dstDescPtr, stdDevTensor, kernelSize,
                            borderType, roiTensorPtrSrc, roiTypeSrc, handle,
                            RppBackend::RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case MAGNITUDE: {
                    testCaseName = "magnitude";

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_magnitude(d_input, d_input_second, srcDescPtr,
                                                          d_output, dstDescPtr, roiTensorPtrSrc,
                                                          roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case PHASE: {
                    testCaseName = "phase";

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_phase(d_input, d_input_second, srcDescPtr, d_output, dstDescPtr,
                                       roiTensorPtrSrc, roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case BITWISE_AND: {
                    testCaseName = "bitwise_and";

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8)
                        errorCodeCapture = rppt_bitwise_and(d_input, d_input_second, srcDescPtr,
                                                            d_output, dstDescPtr, roiTensorPtrSrc,
                                                            roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case BITWISE_NOT: {
                    testCaseName = "bitwise_not";

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8)
                        errorCodeCapture =
                            rppt_bitwise_not(d_input, srcDescPtr, d_output, dstDescPtr,
                                             roiTensorPtrSrc, roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case BITWISE_XOR: {
                    testCaseName = "bitwise_xor";

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8)
                        errorCodeCapture = rppt_bitwise_xor(d_input, d_input_second, srcDescPtr,
                                                            d_output, dstDescPtr, roiTensorPtrSrc,
                                                            roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case BITWISE_OR: {
                    testCaseName = "bitwise_or";

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8)
                        errorCodeCapture = rppt_bitwise_or(d_input, d_input_second, srcDescPtr,
                                                           d_output, dstDescPtr, roiTensorPtrSrc,
                                                           roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case COPY: {
                    testCaseName = "copy";

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_copy(d_input, srcDescPtr, d_output, dstDescPtr,
                                                     handle, RppBackend::RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case REMAP: {
                    testCaseName = "remap";

                    RpptDesc tableDesc = srcDesc;
                    RpptDescPtr tableDescPtr = &tableDesc;
                    init_remap(tableDescPtr, srcDescPtr, roiTensorPtrSrc, rowRemapTable,
                               colRemapTable);

                    CHECK_RETURN_STATUS(hipMemcpy(d_rowRemapTable, (void*)rowRemapTable,
                                                  ioBufferSize * sizeof(Rpp32f),
                                                  hipMemcpyHostToDevice));
                    CHECK_RETURN_STATUS(hipMemcpy(d_colRemapTable, (void*)colRemapTable,
                                                  ioBufferSize * sizeof(Rpp32f),
                                                  hipMemcpyHostToDevice));

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_remap(
                            d_input, srcDescPtr, d_output, dstDescPtr, (Rpp32f*)d_rowRemapTable,
                            (Rpp32f*)d_colRemapTable, tableDescPtr, interpolationType,
                            roiTensorPtrSrc, roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case RESIZE_MIRROR_NORMALIZE: {
                    testCaseName = "resize_mirror_normalize";

                    if (interpolationType != RpptInterpolationType::BILINEAR) {
                        missingFuncFlag = 1;
                        break;
                    }

                    for (i = 0; i < batchSize; i++) {
                        dstImgSizes[i].width = roiTensorPtrDst[i].xywhROI.roiWidth =
                            roiTensorPtrSrc[i].xywhROI.roiWidth / 2;
                        dstImgSizes[i].height = roiTensorPtrDst[i].xywhROI.roiHeight =
                            roiTensorPtrSrc[i].xywhROI.roiWidth / 2;
                    }

                    for (i = 0, j = 0; i < batchSize; i++, j += 3) {
                        meanTensor[j] = 60.0;
                        stdDevTensor[j] = 1.0;

                        meanTensor[j + 1] = 80.0;
                        stdDevTensor[j + 1] = 1.0;

                        meanTensor[j + 2] = 100.0;
                        stdDevTensor[j + 2] = 1.0;
                        mirror[i] = 1;
                    }

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_resize_mirror_normalize(
                            d_input, srcDescPtr, d_output, dstDescPtr, dstImgSizes,
                            interpolationType, meanTensor, stdDevTensor, mirror, roiTensorPtrDst,
                            roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case RICAP: {
                    testCaseName = "ricap";

                    if (qaFlag)
                        init_ricap_qa(maxWidth, maxHeight, batchSize, permutationTensor,
                                      roiPtrInputCropRegion);
                    else
                        init_ricap(maxWidth, maxHeight, batchSize, permutationTensor,
                                   roiPtrInputCropRegion);

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_ricap(d_input, srcDescPtr, d_output, dstDescPtr, permutationTensor,
                                       roiPtrInputCropRegion, roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;
                    break;
                }
                case GRIDMASK: {
                    testCaseName = "gridmask";

                    Rpp32u tileWidth = 40;
                    Rpp32f gridRatio = 0.6;
                    Rpp32f gridAngle = 0.5;
                    RpptUintVector2D translateVector;
                    translateVector.x = 0.0;
                    translateVector.y = 0.0;

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_gridmask(d_input, srcDescPtr, d_output, dstDescPtr, tileWidth,
                                          gridRatio, gridAngle, translateVector, roiTensorPtrSrc,
                                          roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case SPATTER: {
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
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_spatter(d_input, srcDescPtr, d_output, dstDescPtr, spatterColor,
                                         roiTensorPtrSrc, roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case CHANNEL_PERMUTE: {
                    testCaseName = "channel_permute";

                    for (int i = 0; i < batchSize; i++)
                        fill_perm_values(&permutationTensor[i * 3], qaFlag, additionalParam);

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_channel_permute(
                            d_input, srcDescPtr, d_output, dstDescPtr, permutationTensor, handle,
                            RppBackend::RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case COLOR_TO_GREYSCALE: {
                    testCaseName = "color_to_greyscale";

                    RpptSubpixelLayout srcSubpixelLayout = RpptSubpixelLayout::RGBtype;

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_color_to_greyscale(
                            d_input, srcDescPtr, d_output, dstDescPtr, srcSubpixelLayout, handle,
                            RppBackend::RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case YUV_TO_RGB: {
                    testCaseName = "yuv_to_rgb";
                    // Per-input col_standard / color_range from each file's .info sidecar
                    // (defaults: BT.709, full range).
                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8) {
                        size_t srcOffsetBytes = 0;
                        for (int i = 0; i < batchSize; i++) {
                            RpptYuvNv12Sidecar yuvSidecar;
                            if (!parse_yuv_nv12_sidecar(*(imagesPathStart + i), yuvSidecar)) {
                                std::cerr << "\nyuv_to_rgb: missing or invalid .info for "
                                          << *(imagesPathStart + i) << std::endl;
                                errorCodeCapture = RPP_ERROR;
                                break;
                            }
                            Rpp32u width = (Rpp32u)roiTensorPtrDst[i].xywhROI.roiWidth;
                            Rpp32u height = (Rpp32u)roiTensorPtrDst[i].xywhROI.roiHeight;
                            Rpp32u src_y_pitch = width * sizeof(Rpp8u);
                            Rpp32u src_uv_pitch = src_y_pitch;  // tight NV12: same row pitch as Y
                            Rpp32u bgr_pitch = width * 3 * sizeof(Rpp8u);
                            Rpp8u* srcY = (Rpp8u*)d_input + srcOffsetBytes;
                            Rpp8u* srcUV = srcY + (size_t)height * src_y_pitch;
                            void* dstImg =
                                (Rpp8u*)d_output + (size_t)i * dstDescPtr->strides.nStride;
                            errorCodeCapture = rppt_yuv_to_rgb(
                                srcY, srcUV, srcDescPtr, dstImg, dstDescPtr, src_y_pitch,
                                src_uv_pitch, bgr_pitch, width, height, yuvSidecar.col_standard,
                                yuvSidecar.color_range, handle, RppBackend::RPP_HIP_BACKEND);
                            if (errorCodeCapture != RPP_SUCCESS) break;
                            srcOffsetBytes += (size_t)roiTensorPtrSrc[i].xywhROI.roiWidth *
                                              roiTensorPtrSrc[i].xywhROI.roiHeight * 3 / 2;
                        }
                    } else
                        missingFuncFlag = 1;
                    break;
                }
                case YUV_TO_RGB_CUBIC_V: {
                    testCaseName = "yuv_to_rgb_cubic_v";
                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8) {
                        size_t srcOffsetBytes = 0;
                        for (int i = 0; i < batchSize; i++) {
                            RpptYuvNv12Sidecar yuvSidecar;
                            if (!parse_yuv_nv12_sidecar(*(imagesPathStart + i), yuvSidecar)) {
                                std::cerr << "\nyuv_to_rgb_cubic_v: missing or invalid .info for "
                                          << *(imagesPathStart + i) << std::endl;
                                errorCodeCapture = RPP_ERROR;
                                break;
                            }
                            Rpp32u width = (Rpp32u)roiTensorPtrDst[i].xywhROI.roiWidth;
                            Rpp32u height = (Rpp32u)roiTensorPtrDst[i].xywhROI.roiHeight;
                            Rpp32u src_y_pitch = width * sizeof(Rpp8u);
                            Rpp32u src_uv_pitch = src_y_pitch;
                            Rpp32u bgr_pitch = width * 3 * sizeof(Rpp8u);
                            Rpp8u* srcY = (Rpp8u*)d_input + srcOffsetBytes;
                            Rpp8u* srcUV = srcY + (size_t)height * src_y_pitch;
                            void* dstImg =
                                (Rpp8u*)d_output + (size_t)i * dstDescPtr->strides.nStride;
                            errorCodeCapture = rppt_yuv_to_rgb_cubic_v(
                                srcY, srcUV, srcDescPtr, dstImg, dstDescPtr, src_y_pitch,
                                src_uv_pitch, bgr_pitch, width, height, yuvSidecar.col_standard,
                                yuvSidecar.color_range, handle, RppBackend::RPP_HIP_BACKEND);
                            if (errorCodeCapture != RPP_SUCCESS) break;
                            srcOffsetBytes += (size_t)roiTensorPtrSrc[i].xywhROI.roiWidth *
                                              roiTensorPtrSrc[i].xywhROI.roiHeight * 3 / 2;
                        }
                    } else
                        missingFuncFlag = 1;
                    break;
                }
                case YUV_TO_RGB_LINEAR_V: {
                    testCaseName = "yuv_to_rgb_linear_v";
                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8) {
                        size_t srcOffsetBytes = 0;
                        for (int i = 0; i < batchSize; i++) {
                            RpptYuvNv12Sidecar yuvSidecar;
                            if (!parse_yuv_nv12_sidecar(*(imagesPathStart + i), yuvSidecar)) {
                                std::cerr << "\nyuv_to_rgb_linear_v: missing or invalid .info for "
                                          << *(imagesPathStart + i) << std::endl;
                                errorCodeCapture = RPP_ERROR;
                                break;
                            }
                            Rpp32u width = (Rpp32u)roiTensorPtrDst[i].xywhROI.roiWidth;
                            Rpp32u height = (Rpp32u)roiTensorPtrDst[i].xywhROI.roiHeight;
                            Rpp32u src_y_pitch = width * sizeof(Rpp8u);
                            Rpp32u src_uv_pitch = src_y_pitch;
                            Rpp32u bgr_pitch = width * 3 * sizeof(Rpp8u);
                            Rpp8u* srcY = (Rpp8u*)d_input + srcOffsetBytes;
                            Rpp8u* srcUV = srcY + (size_t)height * src_y_pitch;
                            void* dstImg =
                                (Rpp8u*)d_output + (size_t)i * dstDescPtr->strides.nStride;
                            errorCodeCapture = rppt_yuv_to_rgb_linear_v(
                                srcY, srcUV, srcDescPtr, dstImg, dstDescPtr, src_y_pitch,
                                src_uv_pitch, bgr_pitch, width, height, yuvSidecar.col_standard,
                                yuvSidecar.color_range, handle, RppBackend::RPP_HIP_BACKEND);
                            if (errorCodeCapture != RPP_SUCCESS) break;
                            srcOffsetBytes += (size_t)roiTensorPtrSrc[i].xywhROI.roiWidth *
                                              roiTensorPtrSrc[i].xywhROI.roiHeight * 3 / 2;
                        }
                    } else
                        missingFuncFlag = 1;
                    break;
                }
                case TENSOR_SUM: {
                    testCaseName = "tensor_sum";

                    if (srcDescPtr->c == 1) reductionFuncResultArrLength = srcDescPtr->n;

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_tensor_sum(d_input, srcDescPtr, reductionFuncResultArr,
                                            reductionFuncResultArrLength, roiTensorPtrSrc,
                                            roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case TENSOR_MIN: {
                    testCaseName = "tensor_min";

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_tensor_min(d_input, srcDescPtr, reductionFuncResultArr,
                                            reductionFuncResultArrLength, roiTensorPtrSrc,
                                            roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case TENSOR_MAX: {
                    testCaseName = "tensor_max";

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_tensor_max(d_input, srcDescPtr, reductionFuncResultArr,
                                            reductionFuncResultArrLength, roiTensorPtrSrc,
                                            roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case TENSOR_MEAN: {
                    testCaseName = "tensor_mean";

                    if (srcDescPtr->c == 1) reductionFuncResultArrLength = srcDescPtr->n;

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_tensor_mean(d_input, srcDescPtr, reductionFuncResultArr,
                                             reductionFuncResultArrLength, roiTensorPtrSrc,
                                             roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case TENSOR_STDDEV: {
                    testCaseName = "tensor_stddev";

                    if (srcDescPtr->c == 1) reductionFuncResultArrLength = srcDescPtr->n;
                    if (BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == F16_TO_F16)
                        memcpy(mean, TensorMeanReferenceOutputs_F32[inputChannels].data(),
                               sizeof(Rpp32f) * reductionFuncResultArrLength);
                    else
                        memcpy(mean, TensorMeanReferenceOutputs_U8[inputChannels].data(),
                               sizeof(Rpp32f) * reductionFuncResultArrLength);

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_tensor_stddev(d_input, srcDescPtr, reductionFuncResultArr,
                                               reductionFuncResultArrLength, mean, roiTensorPtrSrc,
                                               roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case SLICE: {
                    testCaseName = "slice";
                    Rpp32u numDims =
                        descriptorPtr3D->numDims - 1;  // exclude batchSize from input dims
                    if (anchorTensor == NULL)
                        CHECK_RETURN_STATUS(
                            hipHostMalloc(&anchorTensor, batchSize * numDims * sizeof(Rpp32s)));
                    if (shapeTensor == NULL)
                        CHECK_RETURN_STATUS(
                            hipHostMalloc(&shapeTensor, batchSize * numDims * sizeof(Rpp32s)));
                    if (roiTensor == NULL)
                        CHECK_RETURN_STATUS(
                            hipHostMalloc(&roiTensor, batchSize * numDims * 2 * sizeof(Rpp32u)));
                    bool enablePadding = false;
                    auto fillValue = 0;
                    init_slice(descriptorPtr3D, roiTensorPtrSrc, roiTensor, anchorTensor,
                               shapeTensor);

                    startWallTime = omp_get_wtime();
                    if ((BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F32_TO_F32) &&
                        srcDescPtr->layout == dstDescPtr->layout)
                        errorCodeCapture =
                            rppt_slice(d_input, descriptorPtr3D, d_output, descriptorPtr3D,
                                       anchorTensor, shapeTensor, &fillValue, enablePadding,
                                       roiTensor, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case JPEG_COMPRESSION_DISTORTION: {
                    testCaseName = "jpeg_compression_distortion";
                    for (i = 0; i < batchSize; i++) qualityTensor[i] = 50;
                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_jpeg_compression_distortion(
                            d_input, srcDescPtr, d_output, dstDescPtr, qualityTensor,
                            roiTensorPtrSrc, roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case POSTERIZE: {
                    testCaseName = "posterize";

                    for (i = 0; i < batchSize; i++) posterizeLevelBits[i] = 3;

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_posterize(d_input, srcDescPtr, d_output, dstDescPtr,
                                                          posterizeLevelBits, roiTensorPtrSrc,
                                                          roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case SOLARIZE: {
                    testCaseName = "solarize";

                    for (int i = 0; i < batchSize; i++) thresholdTensor[i] = 0.5;

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_solarize(d_input, srcDescPtr, d_output, dstDescPtr,
                                                         thresholdTensor, roiTensorPtrSrc,
                                                         roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case CHANNEL_DROPOUT: {
                    testCaseName = "channel_dropout";

                    Rpp32f dropoutProbability[batchSize];
                    Rpp32u seed = qaFlag ? DROPOUT_FIXED_SEED : std::random_device{}();
                    for (i = 0; i < batchSize; i++) dropoutProbability[i] = 0.4f;
                    generate_channel_dropout_mask(dropoutTensor, dropoutProbability, batchSize,
                                                  srcDescPtr->c, seed);

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_channel_dropout(
                            d_input, srcDescPtr, d_output, dstDescPtr, dropoutTensor,
                            roiTensorPtrSrc, roiTypeSrc, handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case CUTOUT_DROPOUT: {
                    testCaseName = "cutout_dropout";
                    Rpp32u boxesInEachImage = 1;
                    Rpp32u seed = qaFlag ? DROPOUT_FIXED_SEED : std::random_device{}();
                    init_cutout_dropout(batchSize, boxesInEachImage, numOfBoxes,
                                        anchorBoxInfoTensor, roiTensorPtrSrc, srcDescPtr->c,
                                        BitDepthTestMode, seed, 1, colorBuffer);

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_cutout_dropout(
                            d_input, srcDescPtr, d_output, dstDescPtr, anchorBoxInfoTensor,
                            colorBuffer, numOfBoxes, roiTensorPtrSrc, roiTypeSrc, handle,
                            RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case GRID_DROPOUT: {
                    testCaseName = "grid_dropout";
                    Rpp32f holeRatio = 0.4f;
                    Rpp32u seed = qaFlag ? DROPOUT_FIXED_SEED : std::random_device{}();

                    Rpp32u boxesInEachImage = numGridsPerRow * numGridsPerColumn;
                    Rpp32u maxHoleW = 0, maxHoleH = 0;
                    init_grid_dropout(srcDescPtr->n, anchorBoxInfoTensor, roiTensorPtrSrc,
                                      numGridsPerRow, numGridsPerColumn, maxHoleW, maxHoleH,
                                      holeRatio, seed);

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_grid_dropout(
                            d_input, srcDescPtr, d_output, dstDescPtr, anchorBoxInfoTensor,
                            boxesInEachImage, maxHoleW, maxHoleH, roiTensorPtrSrc, roiTypeSrc,
                            handle, RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case EMBOSS: {
                    testCaseName = "emboss";
                    Rpp32u kernelSize = additionalParam;

                    for (i = 0; i < batchSize; i++) strength[i] = 1.0f;

                    if (borderType != RpptImageBorderType::REPLICATE) {
                        missingFuncFlag = 1;
                        break;
                    }

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_emboss(d_input, srcDescPtr, d_output, dstDescPtr, strength,
                                        kernelSize, borderType, roiTensorPtrSrc, roiTypeSrc, handle,
                                        RppBackend::RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case RANDOM_ERASE: {
                    testCaseName = "random_erase";
                    Rpp32u seed = qaFlag ? DROPOUT_FIXED_SEED : std::random_device{}();
                    init_dropout_random_erase(batchSize, boxesInEachImage, NULL,
                                              anchorBoxInfoTensor, roiTensorPtrSrc, srcDescPtr->c,
                                              BitDepthTestMode, seed, 3, colorBuffer);

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_random_erase(d_input, srcDescPtr, d_output, dstDescPtr,
                                              anchorBoxInfoTensor, colorBuffer, roiTensorPtrSrc,
                                              roiTypeSrc, handle, RppBackend::RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case COARSE_DROPOUT: {
                    testCaseName = "coarse";
                    Rpp32u seed = qaFlag ? DROPOUT_FIXED_SEED : std::random_device{}();
                    init_dropout_erase(batchSize, maxBoxesPerImage, numOfBoxes, anchorBoxInfoTensor,
                                       roiTensorPtrSrc, srcDescPtr->c, BitDepthTestMode, seed, 4);
                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_coarse_dropout(
                            d_input, srcDescPtr, d_output, dstDescPtr, anchorBoxInfoTensor,
                            numOfBoxes, maxBoxesPerImage, roiTensorPtrSrc, roiTypeSrc, handle,
                            RppBackend::RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case HISTOGRAM_EQUALIZE: {
                    testCaseName = "histogram_equalize";

                    startWallTime = omp_get_wtime();
                    if (BitDepthTestMode == U8_TO_U8)
                        errorCodeCapture = rppt_histogram_equalize(
                            d_input, srcDescPtr, d_output, dstDescPtr, roiTensorPtrSrc, roiTypeSrc,
                            handle, RppBackend::RPP_HIP_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                default: {
                    missingFuncFlag = 1;
                    break;
                }
            }

            CHECK_RETURN_STATUS(hipDeviceSynchronize());
            endWallTime = omp_get_wtime();
            wallTime = endWallTime - startWallTime;
            if (missingFuncFlag == 1) {
                cout << "\nThe functionality " << func << " doesn't yet exist in RPP\n";
                return RPP_ERROR_NOT_IMPLEMENTED;
            }
            if (errorCodeCapture != RPP_SUCCESS) {
                cout << "\nThe functionality " << func << " returned an error status "
                     << rppStatusToString[errorCodeCapture] << " on run number " << perfRunCount + 1
                     << " of " << numRuns << " runs.\n";
                return errorCodeCapture;
            }

            maxWallTime = max(maxWallTime, wallTime);
            minWallTime = min(minWallTime, wallTime);
            avgWallTime += wallTime;
        }
        wallTime *= 1000;

        if (testType == UNIT_TEST)  // unit test mode
        {
            cout << "\n\n";
            if (noOfIterations > 1)
                cout << "Execution Timings for Iteration " << iterCount + 1 << ":" << endl;
            cout << "GPU Backend Wall Time: " << wallTime << " ms/batch";
            // Display results for reduction functions
            if (reductionTypeCase) {
                if (srcDescPtr->c == 3)
                    cout << "\nReduction result (Batch of 3 channel images produces 4 results per "
                            "image in batch): ";
                else if (srcDescPtr->c == 1) {
                    cout << "\nReduction result (Batch of 1 channel images produces 1 result per "
                            "image in batch): ";
                    reductionFuncResultArrLength = srcDescPtr->n;
                }

                // print reduction functions output array based on different bit depths, and
                // precision desired
                int precision = ((dstDescPtr->dataType == RpptDataType::F32) ||
                                 (dstDescPtr->dataType == RpptDataType::F16) ||
                                 testCase == TENSOR_MEAN || testCase == TENSOR_STDDEV)
                                    ? 3
                                    : 0;
                if (dstDescPtr->dataType == RpptDataType::F32 || testCase == TENSOR_MEAN ||
                    testCase == TENSOR_STDDEV)
                    print_array(static_cast<Rpp32f*>(reductionFuncResultArr),
                                reductionFuncResultArrLength, precision);
                else if (dstDescPtr->dataType == RpptDataType::U8) {
                    if (testCase == TENSOR_SUM)
                        print_array(static_cast<Rpp64u*>(reductionFuncResultArr),
                                    reductionFuncResultArrLength, precision);
                    else
                        print_array(static_cast<Rpp8u*>(reductionFuncResultArr),
                                    reductionFuncResultArrLength, precision);
                } else if (dstDescPtr->dataType == RpptDataType::F16) {
                    if (testCase == TENSOR_SUM)
                        print_array(static_cast<Rpp32f*>(reductionFuncResultArr),
                                    reductionFuncResultArrLength, precision);
                    else
                        print_array(static_cast<Rpp16f*>(reductionFuncResultArr),
                                    reductionFuncResultArrLength, precision);
                } else if (dstDescPtr->dataType == RpptDataType::I8) {
                    if (testCase == TENSOR_SUM)
                        print_array(static_cast<Rpp64s*>(reductionFuncResultArr),
                                    reductionFuncResultArrLength, precision);
                    else
                        print_array(static_cast<Rpp8s*>(reductionFuncResultArr),
                                    reductionFuncResultArrLength, precision);
                }
                cout << "\n";

                /*Compare the output of the function with golden outputs only if
                1.QA Flag is set
                2.input bit depth 0 (U8)
                3.source and destination layout are the same*/
                if (qaFlag && (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F32_TO_F32) &&
                    (srcDescPtr->layout == dstDescPtr->layout) && !(randomOutputCase) &&
                    !(nonQACase)) {
                    if (testCase == TENSOR_SUM) {
                        if (BitDepthTestMode == U8_TO_U8)
                            compare_reduction_output(static_cast<uint64_t*>(reductionFuncResultArr),
                                                     testCaseName, srcDescPtr, testCase, dst,
                                                     scriptPath);
                        else if (BitDepthTestMode == F32_TO_F32)
                            compare_reduction_output(static_cast<Rpp32f*>(reductionFuncResultArr),
                                                     testCaseName, srcDescPtr, testCase, dst,
                                                     scriptPath);
                    } else if (testCase == TENSOR_MEAN || testCase == TENSOR_STDDEV)
                        compare_reduction_output(static_cast<Rpp32f*>(reductionFuncResultArr),
                                                 testCaseName, srcDescPtr, testCase, dst,
                                                 scriptPath);
                    else {
                        if (BitDepthTestMode == U8_TO_U8)
                            compare_reduction_output(static_cast<Rpp8u*>(reductionFuncResultArr),
                                                     testCaseName, srcDescPtr, testCase, dst,
                                                     scriptPath);
                        else if (BitDepthTestMode == F32_TO_F32)
                            compare_reduction_output(static_cast<Rpp32f*>(reductionFuncResultArr),
                                                     testCaseName, srcDescPtr, testCase, dst,
                                                     scriptPath);
                    }
                }
            } else {
                CHECK_RETURN_STATUS(
                    hipMemcpy(output, d_output, outputBufferSize, hipMemcpyDeviceToHost));

                // Reconvert other bit depths to 8u for output display purposes
                convert_output_bitdepth_to_u8(output, outputu8, BitDepthTestMode, oBufferSize,
                                              outputBufferSize, dstDescPtr, invConversionFactor);

                // If DEBUG_MODE is set to 1 dump the outputs to binary files for debugging
                if (DEBUG_MODE && iterCount == 0) {
                    // Build filename: {testCaseName}_{datatype}_{additional_details}.bin
                    std::string binFileName = testCaseName;

                    // For sobel_filter and color_to_greyscale:
                    // PKD3/PLN3 inputs (c=3) share one file, PLN1 input (c=1) has separate file
                    // since they produce different outputs (RGB->grayscale vs grayscale->grayscale)
                    if (pln1OutTypeCase && srcDescPtr->c == 1) {
                        binFileName += "_PLN1_to_PLN1";
                    }

                    // Add datatype suffix
                    string dataType[4] = {"_u8", "_f32", "_f16", "_i8"};
                    binFileName += dataType[dstDescPtr->dataType];

                    // Add additional details if applicable
                    if (testCase == RESIZE || testCase == ROTATE || testCase == WARP_AFFINE ||
                        testCase == WARP_PERSPECTIVE || testCase == REMAP)
                        binFileName += "_interpolationType" + interpolationTypeName;
                    else if (testCase == NOISE)
                        binFileName += "_noiseType" + noiseTypeName;
                    else if (testCase == ERODE || testCase == DILATE || testCase == BOX_FILTER ||
                             testCase == MEDIAN_FILTER || testCase == GAUSSIAN_FILTER ||
                             testCase == EMBOSS)
                        binFileName += "_kernelSize" + std::to_string(additionalParam);
                    else if (testCase == CHANNEL_PERMUTE)
                        binFileName += "_permOrder" + std::to_string(additionalParam);
                    else if (testCase == SOBEL_FILTER)
                        binFileName += kernelSizeAndGradientName;

                    binFileName += ".bin";

                    // Determine write mode based on OUTPUT layout
                    // PKD3: write to offset 0 (truncate) - PKD3 format for RGB section
                    // PLN3: skip writing (compare_outputs_pln3 converts to PKD3 for comparison)
                    // PLN1: append after PKD3 data (only if not a pln1OutTypeCase)
                    // pln1OutTypeCases (sobel_filter, color_to_greyscale) always output grayscale,
                    // so always truncate

                    // Skip writing for PLN3 (c=3, NCHW layout) since it would overwrite PKD3 data
                    bool shouldSkipWrite =
                        (!pln1OutTypeCase && dstDescPtr->layout == RpptLayout::NCHW &&
                         dstDescPtr->c == 3);

                    if (!shouldSkipWrite) {
                        std::ios_base::openmode mode = std::ios::trunc;
                        if (!pln1OutTypeCase && dstDescPtr->layout == RpptLayout::NCHW &&
                            dstDescPtr->c == 1)
                            mode = std::ios::app;

                        // Write binary data based on data type
                        std::ofstream binFile(binFileName, std::ios::binary | mode);
                        if (binFile.is_open()) {
                            if (dstDescPtr->dataType == RpptDataType::U8)
                                binFile.write(reinterpret_cast<const char*>(outputu8),
                                              oBufferSize * sizeof(Rpp8u));
                            else if (dstDescPtr->dataType == RpptDataType::F32)
                                binFile.write(
                                    reinterpret_cast<const char*>(static_cast<Rpp32f*>(output)),
                                    oBufferSize * sizeof(Rpp32f));
                            else if (dstDescPtr->dataType == RpptDataType::F16)
                                binFile.write(
                                    reinterpret_cast<const char*>(static_cast<Rpp16f*>(output)),
                                    oBufferSize * sizeof(Rpp16f));
                            else if (dstDescPtr->dataType == RpptDataType::I8)
                                binFile.write(
                                    reinterpret_cast<const char*>(static_cast<Rpp8s*>(output)),
                                    oBufferSize * sizeof(Rpp8s));
                            binFile.close();
                        }
                    }
                }

                // if test case is slice and qaFlag is set, update the dstImgSizes with shapeTensor
                // values for output display and comparision purposes
                if (testCase == SLICE) {
                    if (dstDescPtr->layout == RpptLayout::NCHW) {
                        if (dstDescPtr->c == 3) {
                            for (int i = 0; i < batchSize; i++) {
                                int idx1 = i * 3;
                                dstImgSizes[i].height = shapeTensor[idx1 + 1];
                                dstImgSizes[i].width = shapeTensor[idx1 + 2];
                            }
                        } else {
                            for (int i = 0; i < batchSize; i++) {
                                int idx1 = i * 2;
                                dstImgSizes[i].height = shapeTensor[idx1];
                                dstImgSizes[i].width = shapeTensor[idx1 + 1];
                            }
                        }
                    } else if (dstDescPtr->layout == RpptLayout::NHWC) {
                        for (int i = 0; i < batchSize; i++) {
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
                if (qaFlag && (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F32_TO_F32) &&
                    (!(randomOutputCase) && !(nonQACase))) {
                    vector<string> batchYuvPaths;
                    if ((testCase == YUV_TO_RGB || testCase == YUV_TO_RGB_CUBIC_V ||
                         testCase == YUV_TO_RGB_LINEAR_V))
                        batchYuvPaths.assign(imagesPathStart, imagesPathEnd);
                    compare_output(output, testCaseName, srcDescPtr, dstDescPtr, dstImgSizes,
                                   batchSize, interpolationTypeName, noiseTypeName,
                                   kernelSizeAndGradientName, additionalParam, testCase, dst,
                                   scriptPath,
                                   (testCase == YUV_TO_RGB || testCase == YUV_TO_RGB_CUBIC_V ||
                                    testCase == YUV_TO_RGB_LINEAR_V)
                                       ? &batchYuvPaths
                                       : nullptr);
                }

                // Calculate exact dstROI in XYWH format for OpenCV dump
                if (roiTypeSrc == RpptRoiType::LTRB)
                    convert_roi(roiTensorPtrDst, RpptRoiType::XYWH, dstDescPtr->n);

                // Check if the ROI values for each input is within the bounds of the max buffer
                // allocated
                RpptROI roiDefault;
                RpptROIPtr roiPtrDefault = &roiDefault;
                roiPtrDefault->xywhROI = {0, 0, static_cast<Rpp32s>(dstDescPtr->w),
                                          static_cast<Rpp32s>(dstDescPtr->h)};
                for (int i = 0; i < dstDescPtr->n; i++) {
                    roiTensorPtrDst[i].xywhROI.roiWidth =
                        std::min(roiPtrDefault->xywhROI.roiWidth - roiTensorPtrDst[i].xywhROI.xy.x,
                                 roiTensorPtrDst[i].xywhROI.roiWidth);
                    roiTensorPtrDst[i].xywhROI.roiHeight =
                        std::min(roiPtrDefault->xywhROI.roiHeight - roiTensorPtrDst[i].xywhROI.xy.y,
                                 roiTensorPtrDst[i].xywhROI.roiHeight);
                    roiTensorPtrDst[i].xywhROI.xy.x =
                        std::max(roiPtrDefault->xywhROI.xy.x, roiTensorPtrDst[i].xywhROI.xy.x);
                    roiTensorPtrDst[i].xywhROI.xy.y =
                        std::max(roiPtrDefault->xywhROI.xy.y, roiTensorPtrDst[i].xywhROI.xy.y);
                }

                // Convert any PLN3 outputs to the corresponding PKD3 version for OpenCV dump
                if (layoutType == 0 || layoutType == 1) {
                    if ((dstDescPtr->c == 3) && (dstDescPtr->layout == RpptLayout::NCHW))
                        convert_pln3_to_pkd3(outputu8, dstDescPtr);
                }
                // OpenCV dump (if testType is unit test and QA mode is not set)
                if (!qaFlag) {
#if defined(RPP_TEST_SUITE_HAVE_OPENCV) && RPP_TEST_SUITE_HAVE_OPENCV
                    write_image_batch_opencv(dst, outputu8, dstDescPtr, imageNamesStart,
                                             dstImgSizes, MAX_IMAGE_DUMP);
#else
                    cerr << "\nWarning: image dump skipped (qa_mode off) — Tensor_image built "
                            "without OpenCV.\n";
#endif
                }
            }
        }
    }
    rppDestroy(handle, backend);
    if (testType == PERFORMANCE_TEST)  // performance test mode
    {
        // Display measured times
        maxWallTime *= 1000;
        minWallTime *= 1000;
        avgWallTime *= 1000;
        avgWallTime /= (numRuns * noOfIterations);
        cout << fixed << "\nmax,min,avg wall times in ms/batch = " << maxWallTime << ","
             << minWallTime << "," << avgWallTime << endl;
    }

    // Free memory
    CHECK_RETURN_STATUS(hipHostFree(roiTensorPtrSrc));
    CHECK_RETURN_STATUS(hipHostFree(roiTensorPtrDst));
    CHECK_RETURN_STATUS(hipHostFree(dstImgSizes));
    if (testCase == VIGNETTE) CHECK_RETURN_STATUS(hipHostFree(intensity));
    if (testCase == RICAP) CHECK_RETURN_STATUS(hipHostFree(roiPtrInputCropRegion));
    if (testCase == CROP_AND_PATCH) {
        CHECK_RETURN_STATUS(hipHostFree(cropRoi));
        CHECK_RETURN_STATUS(hipHostFree(patchRoi));
    }
    if (testCase == LENS_CORRECTION) {
        CHECK_RETURN_STATUS(hipHostFree(cameraMatrix));
        CHECK_RETURN_STATUS(hipHostFree(distortionCoeffs));
    }
    if (testCase == REMAP) {
        free(rowRemapTable);
        free(colRemapTable);
        CHECK_RETURN_STATUS(hipFree(d_rowRemapTable));
        CHECK_RETURN_STATUS(hipFree(d_colRemapTable));
    }
    if (testCase == GLITCH) CHECK_RETURN_STATUS(hipHostFree(rgbOffsets));
    if (perspectiveTensorPtr != NULL) CHECK_RETURN_STATUS(hipHostFree(perspectiveTensorPtr));
    if (affineTensorPtr != NULL) CHECK_RETURN_STATUS(hipHostFree(affineTensorPtr));
    if (reductionTypeCase) {
        CHECK_RETURN_STATUS(hipHostFree(reductionFuncResultArr));
        if (testCase == TENSOR_STDDEV) CHECK_RETURN_STATUS(hipHostFree(mean));
    }
    if (testCase == ERASE || testCase == CUTOUT_DROPOUT) {
        CHECK_RETURN_STATUS(hipHostFree(colorBuffer));
        CHECK_RETURN_STATUS(hipHostFree(anchorBoxInfoTensor));
        CHECK_RETURN_STATUS(hipHostFree(numOfBoxes));
    }
    if (anchorTensor != NULL) CHECK_RETURN_STATUS(hipHostFree(anchorTensor));
    if (shapeTensor != NULL) CHECK_RETURN_STATUS(hipHostFree(shapeTensor));
    if (intensityFactor != NULL) CHECK_RETURN_STATUS(hipHostFree(intensityFactor));
    if (greyFactor != NULL) CHECK_RETURN_STATUS(hipHostFree(greyFactor));
    if (brightnessCoefficient != NULL) CHECK_RETURN_STATUS(hipHostFree(brightnessCoefficient));
    if (snowThreshold != NULL) CHECK_RETURN_STATUS(hipHostFree(snowThreshold));
    if (darkMode != NULL) CHECK_RETURN_STATUS(hipHostFree(darkMode));
    if (roiTensor != NULL) CHECK_RETURN_STATUS(hipHostFree(roiTensor));
    if (testCase == JITTER) CHECK_RETURN_STATUS(hipHostFree(kernelSizeTensor));
    free(input);
    free(input_second);
    free(output);
    free(inputu8);
    free(inputu8Second);
    free(outputu8);
    CHECK_RETURN_STATUS(hipFree(d_input));
    if (dualInputCase) CHECK_RETURN_STATUS(hipFree(d_input_second));
    CHECK_RETURN_STATUS(hipFree(d_output));
    if (testCase == PIXELATE) CHECK_RETURN_STATUS(hipHostFree(d_interDstPtr));
    if (alpha != NULL) CHECK_RETURN_STATUS(hipHostFree(alpha));
    if (beta != NULL) CHECK_RETURN_STATUS(hipHostFree(beta));
    if (rgbTensor != NULL) CHECK_RETURN_STATUS(hipHostFree(rgbTensor));
    if (testCase == COLOR_TWIST) {
        CHECK_RETURN_STATUS(hipHostFree(brightness));
        CHECK_RETURN_STATUS(hipHostFree(contrast));
        CHECK_RETURN_STATUS(hipHostFree(hue));
        CHECK_RETURN_STATUS(hipHostFree(saturation));
    }
    if (testCase == CONTRAST) {
        CHECK_RETURN_STATUS(hipHostFree(contrastFactor));
        CHECK_RETURN_STATUS(hipHostFree(contrastCenter));
    }
    if (testCase == CROP_MIRROR_NORMALIZE) {
        CHECK_RETURN_STATUS(hipHostFree(multiplier));
        CHECK_RETURN_STATUS(hipHostFree(offset));
    }
    if (exposureFactor != NULL) CHECK_RETURN_STATUS(hipHostFree(exposureFactor));
    if (gammaVal != NULL) CHECK_RETURN_STATUS(hipHostFree(gammaVal));
    if (stdDevTensor != NULL) CHECK_RETURN_STATUS(hipHostFree(stdDevTensor));
    if (meanTensor != NULL) CHECK_RETURN_STATUS(hipHostFree(meanTensor));
    if (testCase == NOISE) {
        if (additionalParam == 0) {
            CHECK_RETURN_STATUS(hipHostFree(noiseProbabilityTensor));
            CHECK_RETURN_STATUS(hipHostFree(saltProbabilityTensor));
            CHECK_RETURN_STATUS(hipHostFree(saltValueTensor));
            CHECK_RETURN_STATUS(hipHostFree(pepperValueTensor));
        } else if (additionalParam == 2)
            CHECK_RETURN_STATUS(hipHostFree(shotNoiseFactorTensor));
    }
    if (mirror != NULL) CHECK_RETURN_STATUS(hipHostFree(mirror));
    if (testCase == WATER) {
        CHECK_RETURN_STATUS(hipHostFree(amplX));
        CHECK_RETURN_STATUS(hipHostFree(amplY));
        CHECK_RETURN_STATUS(hipHostFree(freqX));
        CHECK_RETURN_STATUS(hipHostFree(freqY));
        CHECK_RETURN_STATUS(hipHostFree(phaseX));
        CHECK_RETURN_STATUS(hipHostFree(phaseY));
    }
    if (testCase == FLIP) {
        CHECK_RETURN_STATUS(hipHostFree(horizontalFlag));
        CHECK_RETURN_STATUS(hipHostFree(verticalFlag));
    }
    if (testCase == ROTATE) CHECK_RETURN_STATUS(hipHostFree(angle));
    if (testCase == COLOR_TEMPERATURE) CHECK_RETURN_STATUS(hipHostFree(adjustment));
    if (hueShift != NULL) CHECK_RETURN_STATUS(hipHostFree(hueShift));
    if (saturationFactor != NULL) CHECK_RETURN_STATUS(hipHostFree(saturationFactor));
    if (testCase == EMBOSS) CHECK_RETURN_STATUS(hipHostFree(strength));
    if (minTensor != nullptr) CHECK_RETURN_STATUS(hipHostFree(minTensor));
    if (maxTensor != nullptr) CHECK_RETURN_STATUS(hipHostFree(maxTensor));
    if (posterizeLevelBits != nullptr) CHECK_RETURN_STATUS(hipHostFree(posterizeLevelBits));
    if (thresholdTensor != nullptr) CHECK_RETURN_STATUS(hipHostFree(thresholdTensor));
    if (dropoutTensor != nullptr) CHECK_RETURN_STATUS(hipHostFree(dropoutTensor));
    if (permutationTensor != nullptr) CHECK_RETURN_STATUS(hipHostFree(permutationTensor));
    if (testCase == COARSE_DROPOUT) {
        CHECK_RETURN_STATUS(hipHostFree(anchorBoxInfoTensor));
        CHECK_RETURN_STATUS(hipHostFree(numOfBoxes));
    }
    if (testCase == RANDOM_ERASE) {
        CHECK_RETURN_STATUS(hipHostFree(colorBuffer));
        CHECK_RETURN_STATUS(hipHostFree(anchorBoxInfoTensor));
    }
    if (testCase == GRID_DROPOUT) CHECK_RETURN_STATUS(hipHostFree(anchorBoxInfoTensor));
    if (qualityTensor != nullptr) CHECK_RETURN_STATUS(hipHostFree(qualityTensor));
    return 0;
}
