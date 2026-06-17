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

#include <dirent.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <fstream>
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
        cout << "\nlayout type - (0 = PKD3 / 1 = PLN3 / 2 = PLN1) = " << argv[10];
        cout << "\nqa mode - 0/1 = " << argv[12];
        cout << "\ndecoder type - (0 = packed .rgb / 1 = OpenCV) = " << argv[13];
        cout << "\nbatch size = " << argv[14];
    }

    if (argc < MIN_ARG_COUNT) {
        cout << "\nImproper Usage! Needs all arguments!\n";
        cout << "\nUsage: <src1 folder> <src2 folder (place same as src1 folder for single image "
                "functionalities)> <dst folder> <u8 = 0 / f16 = 1 / f32 = 2 / u8->f16 = 3 / "
                "u8->f32 = 4 / i8 = 5 / u8->i8 = 6> <outputFormatToggle (pkd->pkd = 0 / pkd->pln = "
                "1)> <case number = 0:87> <number of runs > 0> <layout type (layout type - (0 = "
                "PKD3 / 1 = PLN3 / 2 = PLN1)> < qa mode (0/1)> <decoder type (0/1)> <batch size > "
                "1> <roiList> <verbosity = 0/1>>\n";
    }

    if (layoutType == 2) {
        if (testCase == COLOR_CAST || testCase == GLITCH || testCase == COLOR_TWIST ||
            testCase == COLOR_TEMPERATURE || testCase == COLOR_TO_GREYSCALE || testCase == HUE ||
            testCase == SATURATION) {
            cout << "\ncase " << testCase << " does not exist for PLN1 layout\n";
            return RPP_ERROR_NOT_IMPLEMENTED;
        } else if (outputFormatToggle != 0) {
            cout << "\nPLN1 cases don't have outputFormatToggle! Please input outputFormatToggle = "
                    "0\n";
            return RPP_ERROR_NOT_IMPLEMENTED;
        }
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

    if (batchSize > MAX_BATCH_SIZE) {
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
    string funcType = set_function_type(layoutType, pln1OutTypeCase, outputFormatToggle, "HOST");

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

    if (interpolationTypeCase) {
        interpolationTypeName = get_interpolation_type(additionalParam, interpolationType);
        func += "_interpolationType";
        func += interpolationTypeName.c_str();
    } else if (noiseTypeCase) {
        noiseTypeName = get_noise_type(additionalParam);
        func += "_noiseType";
        func += noiseTypeName.c_str();
    } else if (kernelSizeCase) {
        func += "_kernelSize";
        func += std::to_string(additionalParam);
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
    const char* inputExt = (decoderType == 0) ? ".rgb" : ".jpg";
    search_files_recursive(src, imageNames, imageNamesPath, inputExt);
    if (dualInputCase) {
        search_files_recursive(srcSecond, imageNamesSecond, imageNamesPathSecond, inputExt);
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

    if (qaFlag || DEBUG_MODE) {
        sort(imageNames.begin(), imageNames.end());
        if (dualInputCase) sort(imageNamesSecond.begin(), imageNamesSecond.end());
    }

    // Initialize ROI tensors for src/dst
    RpptROI* roiTensorPtrSrc = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));
    RpptROI* roiTensorPtrDst = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    // Initialize the ImagePatch for dst
    RpptImagePatch* dstImgSizes =
        static_cast<RpptImagePatch*>(calloc(batchSize, sizeof(RpptImagePatch)));

    // Set ROI tensors types for src/dst
    RpptRoiType roiTypeSrc = RpptRoiType::XYWH;

    Rpp32u outputChannels = inputChannels;
    if (pln1OutTypeCase) outputChannels = 1;
    Rpp32u offsetInBytes = 0;
    int imagesMixed = 0;  // Flag used to check if all images in dataset is of same dimensions

    set_max_dimensions(imageNamesPath, maxHeight, maxWidth, imagesMixed, decoderType);

    if (testCase == RICAP && imagesMixed) {
        std::cerr << "\n RICAP only works with same dimension images";
        exit(0);
    }

    // Set numDims, offset, n/c/h/w values, strides for src/dst
    set_descriptor_dims_and_strides(srcDescPtr, batchSize, maxHeight, maxWidth, inputChannels,
                                    offsetInBytes);
    set_descriptor_dims_and_strides(dstDescPtr, batchSize, maxHeight, maxWidth, outputChannels,
                                    offsetInBytes);

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

    // Initialize 8u host buffers for src/dst
    Rpp8u* inputu8 = static_cast<Rpp8u*>(calloc(ioBufferSizeInBytes_u8, 1));
    Rpp8u* inputu8Second = static_cast<Rpp8u*>(calloc(ioBufferSizeInBytes_u8, 1));
    Rpp8u* outputu8 = static_cast<Rpp8u*>(calloc(oBufferSizeInBytes_u8, 1));

    void *input, *input_second, *output;

    input = static_cast<Rpp8u*>(calloc(inputBufferSize, 1));
    input_second = static_cast<Rpp8u*>(calloc(inputBufferSize, 1));
    output = static_cast<Rpp8u*>(calloc(outputBufferSize, 1));

    Rpp32f *rowRemapTable, *colRemapTable;
    if (testCase == REMAP || testCase == LENS_CORRECTION) {
        rowRemapTable = static_cast<Rpp32f*>(calloc(ioBufferSize, sizeof(Rpp32f)));
        colRemapTable = static_cast<Rpp32f*>(calloc(ioBufferSize, sizeof(Rpp32f)));
    }

    // Initialize buffers for any reductionType functions (testCase 87 - tensor_sum alone cannot
    // return final sum as 8u/8s due to overflow. 8u inputs return 64u sums, 8s inputs return 64s
    // sums)
    void* reductionFuncResultArr;
    Rpp32u reductionFuncResultArrLength = srcDescPtr->n * 4;
    if (reductionTypeCase) {
        int bitDepthByteSize = 0;
        if ((dstDescPtr->dataType == RpptDataType::F16) ||
            (dstDescPtr->dataType == RpptDataType::F32) || testCase == TENSOR_MEAN ||
            testCase == TENSOR_STDDEV) {
            bitDepthByteSize =
                sizeof(Rpp32f);  // using 32f outputs for 16f and 32f, for testCase 90, 91
            reductionFuncResultArr =
                static_cast<Rpp32f*>(calloc(reductionFuncResultArrLength, bitDepthByteSize));
        } else if ((dstDescPtr->dataType == RpptDataType::U8) ||
                   (dstDescPtr->dataType == RpptDataType::I8)) {
            bitDepthByteSize = (testCase == TENSOR_SUM) ? sizeof(Rpp64u) : sizeof(Rpp8u);
            reductionFuncResultArr =
                static_cast<void*>(calloc(reductionFuncResultArrLength, bitDepthByteSize));
        }
    }

    // create generic descriptor and params in case of slice
    RpptGenericDesc descriptor3D;
    RpptGenericDescPtr descriptorPtr3D = &descriptor3D;
    Rpp32s *anchorTensor = NULL, *shapeTensor = NULL;
    Rpp32u* roiTensor = NULL;
    if (testCase == SLICE) set_generic_descriptor_slice(srcDescPtr, descriptorPtr3D, batchSize);

    // create cropRoi and patchRoi in case of crop_and_patch
    RpptROI *cropRoi, *patchRoi;
    if (testCase == CROP_AND_PATCH) {
        cropRoi = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));
        patchRoi = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));
    }
    bool invalidROI = (roiList[0] == 0 && roiList[1] == 0 && roiList[2] == 0 && roiList[3] == 0);

    void* interDstPtr;
    if (testCase == PIXELATE)
        interDstPtr = static_cast<Rpp8u*>(
            calloc(srcDescPtr->strides.nStride * srcDescPtr->n, sizeof(Rpp32f)));

    // Set the number of threads to be used by OpenMP pragma for RPP batch processing on host.
    // If numThreads value passed is 0, number of OpenMP threads used by RPP will be set to batch
    // size
    Rpp32u numThreads = 0;
    rppHandle_t handle;
    RppBackend backend = RppBackend::RPP_HOST_BACKEND;
    rppCreate(&handle, noOfImages, numThreads, nullptr, backend);

    int noOfIterations = (int)imageNames.size() / batchSize;
    double maxWallTime = 0, minWallTime = 500, avgWallTime = 0;
    double cpuTime, wallTime;
    string testCaseName;

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
        set_src_and_dst_roi(imagesPathStart, imagesPathEnd, roiTensorPtrSrc, roiTensorPtrDst,
                            dstImgSizes, decoderType);

        // Read images
        if (decoderType == 0) read_image_batch_packed(inputu8, srcDescPtr, imagesPathStart);
#if defined(RPP_TEST_SUITE_HAVE_OPENCV) && RPP_TEST_SUITE_HAVE_OPENCV
        else
            read_image_batch_opencv(inputu8, srcDescPtr, imagesPathStart);
#else
        else if (decoderType == 1) {
            cerr << "\nError: decoder_type 1 (OpenCV) requested but Tensor_image was built without "
                    "OpenCV. Aborting.\n";
            exit(1);
        } else {
            cerr << "\nError: invalid decoder_type (expected 0 = packed RAW + .info or 1 = "
                    "OpenCV). Aborting.\n";
            exit(1);
        }
#endif

        // if the input layout requested is PLN3, convert PKD3 inputs to PLN3 for first and second
        // input batch
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
                cerr << "\nError: invalid decoder_type (expected 0 = packed RAW + .info or 1 = "
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
        // convert_roi(roiTensorPtrSrc, roiTypeSrc, batchSize);
        // update_dst_sizes_with_roi(roiTensorPtrSrc, dstImgSizes, roiTypeSrc, batchSize);

        for (int perfRunCount = 0; perfRunCount < numRuns; perfRunCount++) {
            RppStatus errorCodeCapture = RPP_SUCCESS;
            clock_t startCpuTime = 0, endCpuTime;
            double startWallTime = 0.0, endWallTime;
            switch (testCase) {
                case BRIGHTNESS: {
                    testCaseName = "brightness";
                    Rpp32f alpha[batchSize];
                    Rpp32f beta[batchSize];
                    for (i = 0; i < batchSize; i++) {
                        alpha[i] = 1.75;
                        beta[i] = 50;
                    }

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_brightness(input, srcDescPtr, output, dstDescPtr, alpha, beta,
                                            roiTensorPtrSrc, roiTypeSrc, handle, RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case GAMMA_CORRECTION: {
                    testCaseName = "gamma_correction";

                    Rpp32f gammaVal[batchSize];
                    for (i = 0; i < batchSize; i++) gammaVal[i] = 1.9;

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_gamma_correction(
                            input, srcDescPtr, output, dstDescPtr, gammaVal, roiTensorPtrSrc,
                            roiTypeSrc, handle, RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case BLEND: {
                    testCaseName = "blend";

                    Rpp32f alpha[batchSize];
                    for (i = 0; i < batchSize; i++) alpha[i] = 0.4;

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_blend(input, input_second, srcDescPtr, output, dstDescPtr, alpha,
                                       roiTensorPtrSrc, roiTypeSrc, handle, RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case CONTRAST: {
                    testCaseName = "contrast";

                    Rpp32f contrastFactor[batchSize];
                    Rpp32f contrastCenter[batchSize];
                    for (i = 0; i < batchSize; i++) {
                        contrastFactor[i] = 2.96;
                        contrastCenter[i] = 128;
                    }

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_contrast(
                            input, srcDescPtr, output, dstDescPtr, contrastFactor, contrastCenter,
                            roiTensorPtrSrc, roiTypeSrc, handle, RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case PIXELATE: {
                    testCaseName = "pixelate";

                    Rpp32f pixelationPercentage = 87.5;

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_pixelate(input, srcDescPtr, output, dstDescPtr, interDstPtr,
                                          pixelationPercentage, roiTensorPtrSrc, roiTypeSrc, handle,
                                          RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case JITTER: {
                    testCaseName = "jitter";

                    Rpp32u kernelSizeTensor[batchSize];
                    Rpp32u seed = 1255459;
                    for (i = 0; i < batchSize; i++) kernelSizeTensor[i] = 5;

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_jitter(input, srcDescPtr, output, dstDescPtr,
                                                       kernelSizeTensor, seed, roiTensorPtrSrc,
                                                       roiTypeSrc, handle, RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case SNOW: {
                    testCaseName = "snow";
                    Rpp32f brightnessCoefficient[batchSize];
                    Rpp32f snowThreshold[batchSize];
                    Rpp32s darkMode[batchSize];
                    for (i = 0; i < batchSize; i++) {
                        brightnessCoefficient[i] = 2.5f;
                        snowThreshold[i] = 1.0f;
                        darkMode[i] = 0;
                    }

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_snow(input, srcDescPtr, output, dstDescPtr, brightnessCoefficient,
                                      snowThreshold, darkMode, roiTensorPtrSrc, roiTypeSrc, handle,
                                      RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case NOISE: {
                    testCaseName = "noise";

                    switch (additionalParam) {
                        case 0: {
                            Rpp32f noiseProbabilityTensor[batchSize];
                            Rpp32f saltProbabilityTensor[batchSize];
                            Rpp32f saltValueTensor[batchSize];
                            Rpp32f pepperValueTensor[batchSize];
                            Rpp32u seed = 1255459;
                            for (i = 0; i < batchSize; i++) {
                                noiseProbabilityTensor[i] = 0.1f;
                                saltProbabilityTensor[i] = 0.5f;
                                saltValueTensor[i] = 1.0f;
                                pepperValueTensor[i] = 0.0f;
                            }

                            startWallTime = omp_get_wtime();
                            startCpuTime = clock();
                            if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                                BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                                errorCodeCapture = rppt_salt_and_pepper_noise(
                                    input, srcDescPtr, output, dstDescPtr, noiseProbabilityTensor,
                                    saltProbabilityTensor, saltValueTensor, pepperValueTensor, seed,
                                    roiTensorPtrSrc, roiTypeSrc, handle, RPP_HOST_BACKEND);
                            else
                                missingFuncFlag = 1;

                            break;
                        }
                        case 1: {
                            Rpp32f meanTensor[batchSize];
                            Rpp32f stdDevTensor[batchSize];
                            Rpp32u seed = 1255459;
                            for (i = 0; i < batchSize; i++) {
                                meanTensor[i] = 0.0f;
                                stdDevTensor[i] = 0.2f;
                            }

                            startWallTime = omp_get_wtime();
                            startCpuTime = clock();
                            if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                                BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                                errorCodeCapture = rppt_gaussian_noise(
                                    input, srcDescPtr, output, dstDescPtr, meanTensor, stdDevTensor,
                                    seed, roiTensorPtrSrc, roiTypeSrc, handle, RPP_HOST_BACKEND);
                            else
                                missingFuncFlag = 1;

                            break;
                        }
                        case 2: {
                            Rpp32f shotNoiseFactorTensor[batchSize];
                            Rpp32u seed = 1255459;
                            for (i = 0; i < batchSize; i++) shotNoiseFactorTensor[i] = 80.0f;

                            startWallTime = omp_get_wtime();
                            startCpuTime = clock();
                            if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                                BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                                errorCodeCapture = rppt_shot_noise(
                                    input, srcDescPtr, output, dstDescPtr, shotNoiseFactorTensor,
                                    seed, roiTensorPtrSrc, roiTypeSrc, handle, RPP_HOST_BACKEND);
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

                    Rpp32f intensityFactor[batchSize];
                    Rpp32f grayFactor[batchSize];
                    for (i = 0; i < batchSize; i++) {
                        intensityFactor[i] = 0;
                        grayFactor[i] = 0.3;
                    }

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_fog(input, srcDescPtr, output, dstDescPtr,
                                                    intensityFactor, grayFactor, roiTensorPtrSrc,
                                                    roiTypeSrc, handle, RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case EXPOSURE: {
                    testCaseName = "exposure";

                    Rpp32f exposureFactor[batchSize];
                    for (i = 0; i < batchSize; i++) exposureFactor[i] = 1.4;

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_exposure(input, srcDescPtr, output, dstDescPtr, exposureFactor,
                                          roiTensorPtrSrc, roiTypeSrc, handle, RPP_HOST_BACKEND);
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
                    Rpp32f alpha[batchSize];
                    for (int i = 0; i < batchSize; i++) alpha[i] = 0.4;

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_rain(input, srcDescPtr, output, dstDescPtr, rainPercentage,
                                      rainWidth, rainHeight, slantAngle, alpha, roiTensorPtrSrc,
                                      roiTypeSrc, handle, RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case THRESHOLD: {
                    testCaseName = "threshold";

                    Rpp32f minTensor[batchSize * srcDescPtr->c];
                    Rpp32f maxTensor[batchSize * srcDescPtr->c];
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
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_threshold(input, srcDescPtr, output, dstDescPtr,
                                                          minTensor, maxTensor, roiTensorPtrSrc,
                                                          roiTypeSrc, handle, RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case FLIP: {
                    testCaseName = "flip";

                    Rpp32u horizontalFlag[batchSize];
                    Rpp32u verticalFlag[batchSize];
                    for (i = 0; i < batchSize; i++) {
                        horizontalFlag[i] = 1;
                        verticalFlag[i] = 0;
                    }

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_flip(input, srcDescPtr, output, dstDescPtr,
                                                     horizontalFlag, verticalFlag, roiTensorPtrSrc,
                                                     roiTypeSrc, handle, RPP_HOST_BACKEND);
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
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_resize(
                            input, srcDescPtr, output, dstDescPtr, dstImgSizes, interpolationType,
                            roiTensorPtrDst, roiTypeSrc, handle, RPP_HOST_BACKEND);
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

                    Rpp32f angle[batchSize];
                    for (i = 0; i < batchSize; i++) angle[i] = 50;

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_rotate(input, srcDescPtr, output, dstDescPtr, angle,
                                                       interpolationType, roiTensorPtrSrc,
                                                       roiTypeSrc, handle, RPP_HOST_BACKEND);
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

                    Rpp32f6 affineTensor_f6[batchSize];
                    Rpp32f* affineTensor = (Rpp32f*)affineTensor_f6;
                    for (i = 0; i < batchSize; i++) {
                        affineTensor_f6[i].data[0] = 1.23;
                        affineTensor_f6[i].data[1] = 0.5;
                        affineTensor_f6[i].data[2] = 0;
                        affineTensor_f6[i].data[3] = -0.8;
                        affineTensor_f6[i].data[4] = 0.83;
                        affineTensor_f6[i].data[5] = 0;
                    }

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_warp_affine(
                            input, srcDescPtr, output, dstDescPtr, affineTensor, interpolationType,
                            roiTensorPtrSrc, roiTypeSrc, handle, RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case FISHEYE: {
                    testCaseName = "fisheye";

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_fisheye(input, srcDescPtr, output, dstDescPtr, roiTensorPtrSrc,
                                         roiTypeSrc, handle, RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case LENS_CORRECTION: {
                    testCaseName = "lens_correction";

                    Rpp32f cameraMatrix[9 * batchSize];
                    Rpp32f distortionCoeffs[8 * batchSize];
                    RpptDesc tableDesc = srcDesc;
                    RpptDescPtr tableDescPtr = &tableDesc;
                    init_lens_correction(batchSize, srcDescPtr, cameraMatrix, distortionCoeffs,
                                         tableDescPtr);

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_lens_correction(
                            input, srcDescPtr, output, dstDescPtr, rowRemapTable, colRemapTable,
                            tableDescPtr, cameraMatrix, distortionCoeffs, roiTensorPtrSrc,
                            roiTypeSrc, handle, RPP_HOST_BACKEND);
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

                    Rpp32f9 perspectiveTensor_f9[batchSize];
                    Rpp32f* perspectiveTensor = reinterpret_cast<Rpp32f*>(perspectiveTensor_f9);
                    for (i = 0; i < batchSize; i++) {
                        perspectiveTensor_f9[i].data[0] = 0.93;
                        perspectiveTensor_f9[i].data[1] = 0.5;
                        perspectiveTensor_f9[i].data[2] = 0.0;
                        perspectiveTensor_f9[i].data[3] = -0.5;
                        perspectiveTensor_f9[i].data[4] = 0.93;
                        perspectiveTensor_f9[i].data[5] = 0.0;
                        perspectiveTensor_f9[i].data[6] = 0.005;
                        perspectiveTensor_f9[i].data[7] = 0.005;
                        perspectiveTensor_f9[i].data[8] = 1;
                    }

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_warp_perspective(
                            input, srcDescPtr, output, dstDescPtr, perspectiveTensor,
                            interpolationType, roiTensorPtrSrc, roiTypeSrc, handle,
                            RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case WATER: {
                    testCaseName = "water";

                    Rpp32f amplX[batchSize];
                    Rpp32f amplY[batchSize];
                    Rpp32f freqX[batchSize];
                    Rpp32f freqY[batchSize];
                    Rpp32f phaseX[batchSize];
                    Rpp32f phaseY[batchSize];

                    for (i = 0; i < batchSize; i++) {
                        amplX[i] = 2.0f;
                        amplY[i] = 5.0f;
                        freqX[i] = 5.8f;
                        freqY[i] = 1.2f;
                        phaseX[i] = 10.0f;
                        phaseY[i] = 15.0f;
                    }

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_water(
                            input, srcDescPtr, output, dstDescPtr, amplX, amplY, freqX, freqY,
                            phaseX, phaseY, roiTensorPtrSrc, roiTypeSrc, handle, RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case NON_LINEAR_BLEND: {
                    testCaseName = "non_linear_blend";

                    Rpp32f stdDev[batchSize];
                    for (i = 0; i < batchSize; i++) stdDev[i] = 50.0;

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_non_linear_blend(
                            input, input_second, srcDescPtr, output, dstDescPtr, stdDev,
                            roiTensorPtrSrc, roiTypeSrc, handle, RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case COLOR_CAST: {
                    testCaseName = "color_cast";

                    RpptRGB rgbTensor[batchSize];
                    Rpp32f alphaTensor[batchSize];
                    for (i = 0; i < batchSize; i++) {
                        rgbTensor[i].R = 0;
                        rgbTensor[i].G = 0;
                        rgbTensor[i].B = 100;
                        alphaTensor[i] = 0.5;
                    }

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_color_cast(input, srcDescPtr, output, dstDescPtr,
                                                           rgbTensor, alphaTensor, roiTensorPtrSrc,
                                                           roiTypeSrc, handle, RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case ERASE: {
                    testCaseName = "erase";
                    Rpp32u boxesInEachImage = 3;
                    Rpp32f colorBuffer[batchSize * boxesInEachImage];
                    RpptRoiLtrb anchorBoxInfoTensor[batchSize * boxesInEachImage];
                    Rpp32u numOfBoxes[batchSize];

                    init_erase(batchSize, boxesInEachImage, numOfBoxes, anchorBoxInfoTensor,
                               roiTensorPtrSrc, srcDescPtr->c, colorBuffer, BitDepthTestMode);

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_erase(
                            input, srcDescPtr, output, dstDescPtr, anchorBoxInfoTensor, colorBuffer,
                            numOfBoxes, roiTensorPtrSrc, roiTypeSrc, handle, RPP_HOST_BACKEND);
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
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_crop_and_patch(
                            input, input_second, srcDescPtr, output, dstDescPtr, roiTensorPtrDst,
                            cropRoi, patchRoi, roiTypeSrc, handle, RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case LOOK_UP_TABLE: {
                    testCaseName = "lut";

                    Rpp32f lutBuffer[65536] = {0};
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
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8)
                        errorCodeCapture =
                            rppt_lut(input, srcDescPtr, output, dstDescPtr, lut8u, roiTensorPtrSrc,
                                     roiTypeSrc, handle, RPP_HOST_BACKEND);
                    else if (BitDepthTestMode == U8_TO_F16)
                        errorCodeCapture =
                            rppt_lut(input, srcDescPtr, output, dstDescPtr, lut16f, roiTensorPtrSrc,
                                     roiTypeSrc, handle, RPP_HOST_BACKEND);
                    else if (BitDepthTestMode == U8_TO_F32)
                        errorCodeCapture =
                            rppt_lut(input, srcDescPtr, output, dstDescPtr, lut32f, roiTensorPtrSrc,
                                     roiTypeSrc, handle, RPP_HOST_BACKEND);
                    else if (BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_lut(input, srcDescPtr, output, dstDescPtr, lut8s, roiTensorPtrSrc,
                                     roiTypeSrc, handle, RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case GLITCH: {
                    testCaseName = "glitch";
                    RpptChannelOffsets rgbOffsets[batchSize];

                    for (i = 0; i < batchSize; i++) {
                        rgbOffsets[i].r.x = 10;
                        rgbOffsets[i].r.y = 10;
                        rgbOffsets[i].g.x = 0;
                        rgbOffsets[i].g.y = 0;
                        rgbOffsets[i].b.x = 5;
                        rgbOffsets[i].b.y = 5;
                    }

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_glitch(input, srcDescPtr, output, dstDescPtr, rgbOffsets,
                                        roiTensorPtrSrc, roiTypeSrc, handle, RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case COLOR_TWIST: {
                    testCaseName = "color_twist";

                    Rpp32f brightness[batchSize];
                    Rpp32f contrast[batchSize];
                    Rpp32f hue[batchSize];
                    Rpp32f saturation[batchSize];
                    for (i = 0; i < batchSize; i++) {
                        brightness[i] = 1.4;
                        contrast[i] = 0.0;
                        hue[i] = 60.0;
                        saturation[i] = 1.9;
                    }

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_color_twist(
                            input, srcDescPtr, output, dstDescPtr, brightness, contrast, hue,
                            saturation, roiTensorPtrSrc, roiTypeSrc, handle, RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case HUE: {
                    testCaseName = "hue";

                    Rpp32f hueShift[batchSize];
                    for (i = 0; i < batchSize; i++) hueShift[i] = 60.0;

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_hue(input, srcDescPtr, output, dstDescPtr, hueShift,
                                     roiTensorPtrSrc, roiTypeSrc, handle, RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case SATURATION: {
                    testCaseName = "saturation";

                    Rpp32f saturationFactor[batchSize];
                    for (i = 0; i < batchSize; i++) saturationFactor[i] = 5;

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_saturation(input, srcDescPtr, output, dstDescPtr, saturationFactor,
                                            roiTensorPtrSrc, roiTypeSrc, handle, RPP_HOST_BACKEND);
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
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_crop(input, srcDescPtr, output, dstDescPtr, roiTensorPtrDst,
                                      roiTypeSrc, handle, RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case CROP_MIRROR_NORMALIZE: {
                    testCaseName = "crop_mirror_normalize";
                    Rpp32f multiplier[batchSize * srcDescPtr->c];
                    Rpp32f offset[batchSize * srcDescPtr->c];
                    Rpp32u mirror[batchSize];
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
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == U8_TO_F16 ||
                        BitDepthTestMode == U8_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_crop_mirror_normalize(
                            input, srcDescPtr, output, dstDescPtr, offset, multiplier, mirror,
                            roiTensorPtrDst, roiTypeSrc, handle, RPP_HOST_BACKEND);
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

                    Rpp32u mirror[batchSize];
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
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == U8_TO_F16 ||
                        BitDepthTestMode == U8_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_resize_crop_mirror(
                            input, srcDescPtr, output, dstDescPtr, dstImgSizes, interpolationType,
                            mirror, roiTensorPtrDst, roiTypeSrc, handle, RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case ERODE: {
                    testCaseName = "erode";
                    Rpp32u kernelSize = additionalParam;

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_erode_host(input, srcDescPtr, output, dstDescPtr, kernelSize,
                                            roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case DILATE: {
                    testCaseName = "dilate";
                    Rpp32u kernelSize = additionalParam;

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_dilate_host(input, srcDescPtr, output, dstDescPtr, kernelSize,
                                             roiTensorPtrSrc, roiTypeSrc, handle);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case COLOR_TEMPERATURE: {
                    testCaseName = "color_temperature";

                    Rpp32s adjustment[batchSize];
                    for (i = 0; i < batchSize; i++) adjustment[i] = 70;

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_color_temperature(
                            input, srcDescPtr, output, dstDescPtr, adjustment, roiTensorPtrSrc,
                            roiTypeSrc, handle, RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case VIGNETTE: {
                    testCaseName = "vignette";

                    Rpp32f intensity[batchSize];
                    for (i = 0; i < batchSize; i++) intensity[i] = 6;

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_vignette(input, srcDescPtr, output, dstDescPtr, intensity,
                                          roiTensorPtrSrc, roiTypeSrc, handle, RPP_HOST_BACKEND);
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
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_box_filter(
                            input, srcDescPtr, output, dstDescPtr, kernelSize, borderType,
                            roiTensorPtrSrc, roiTypeSrc, handle, RppBackend::RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case SOBEL_FILTER: {
                    testCaseName = "sobel_filter";

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_sobel_filter(
                            input, srcDescPtr, output, dstDescPtr, GradientType, kernelSize,
                            roiTensorPtrSrc, roiTypeSrc, handle, RppBackend::RPP_HOST_BACKEND);
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
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_median_filter(
                            input, srcDescPtr, output, dstDescPtr, kernelSize, borderType,
                            roiTensorPtrSrc, roiTypeSrc, handle, RppBackend::RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case GAUSSIAN_FILTER: {
                    testCaseName = "gaussian_filter";
                    Rpp32u kernelSize = additionalParam;

                    Rpp32f stdDevTensor[batchSize];
                    for (i = 0; i < batchSize; i++) stdDevTensor[i] = 5.0f;

                    if (borderType != RpptImageBorderType::REPLICATE) {
                        missingFuncFlag = 1;
                        break;
                    }

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_gaussian_filter(
                            input, srcDescPtr, output, dstDescPtr, stdDevTensor, kernelSize,
                            borderType, roiTensorPtrSrc, roiTypeSrc, handle,
                            RppBackend::RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case MAGNITUDE: {
                    testCaseName = "magnitude";

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_magnitude(input, input_second, srcDescPtr, output, dstDescPtr,
                                           roiTensorPtrSrc, roiTypeSrc, handle, RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case PHASE: {
                    testCaseName = "phase";

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_phase(input, input_second, srcDescPtr, output, dstDescPtr,
                                       roiTensorPtrSrc, roiTypeSrc, handle, RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case BITWISE_AND: {
                    testCaseName = "bitwise_and";

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8)
                        errorCodeCapture =
                            rppt_bitwise_and(input, input_second, srcDescPtr, output, dstDescPtr,
                                             roiTensorPtrSrc, roiTypeSrc, handle, RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case BITWISE_NOT: {
                    testCaseName = "bitwise_not";

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8)
                        errorCodeCapture =
                            rppt_bitwise_not(input, srcDescPtr, output, dstDescPtr, roiTensorPtrSrc,
                                             roiTypeSrc, handle, RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case BITWISE_XOR: {
                    testCaseName = "bitwise_xor";

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8)
                        errorCodeCapture =
                            rppt_bitwise_xor(input, input_second, srcDescPtr, output, dstDescPtr,
                                             roiTensorPtrSrc, roiTypeSrc, handle, RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case BITWISE_OR: {
                    testCaseName = "bitwise_or";

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8)
                        errorCodeCapture =
                            rppt_bitwise_or(input, input_second, srcDescPtr, output, dstDescPtr,
                                            roiTensorPtrSrc, roiTypeSrc, handle, RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case COPY: {
                    testCaseName = "copy";

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_copy(input, srcDescPtr, output, dstDescPtr, handle,
                                                     RppBackend::RPP_HOST_BACKEND);
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

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_remap(input, srcDescPtr, output, dstDescPtr, rowRemapTable,
                                       colRemapTable, tableDescPtr, interpolationType,
                                       roiTensorPtrSrc, roiTypeSrc, handle, RPP_HOST_BACKEND);
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

                    Rpp32f mean[batchSize * 3];
                    Rpp32f stdDev[batchSize * 3];
                    Rpp32u mirror[batchSize];
                    for (i = 0, j = 0; i < batchSize; i++, j += 3) {
                        mean[j] = 60.0;
                        stdDev[j] = 1.0;

                        mean[j + 1] = 80.0;
                        stdDev[j + 1] = 1.0;

                        mean[j + 2] = 100.0;
                        stdDev[j + 2] = 1.0;
                        mirror[i] = 1;
                    }

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_resize_mirror_normalize(
                            input, srcDescPtr, output, dstDescPtr, dstImgSizes, interpolationType,
                            mean, stdDev, mirror, roiTensorPtrDst, roiTypeSrc, handle,
                            RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case COLOR_JITTER: {
                    testCaseName = "color_jitter";

                    Rpp32f brightness[batchSize];
                    Rpp32f contrast[batchSize];
                    Rpp32f hue[batchSize];
                    Rpp32f saturation[batchSize];
                    for (i = 0; i < batchSize; i++) {
                        brightness[i] = 1.02;
                        contrast[i] = 1.1;
                        hue[i] = 0.02;
                        saturation[i] = 1.3;
                    }

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_color_jitter(
                            input, srcDescPtr, output, dstDescPtr, brightness, contrast, hue,
                            saturation, roiTensorPtrSrc, roiTypeSrc, handle, RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case RICAP: {
                    testCaseName = "ricap";

                    Rpp32u permutationTensor[batchSize * 4];
                    RpptROI roiPtrInputCropRegion[4];

                    if (imagesMixed) {
                        std::cerr << "\n RICAP only works with same dimension images";
                        break;
                    }

                    if (qaFlag || DEBUG_MODE)
                        init_ricap_qa(maxWidth, maxHeight, batchSize, permutationTensor,
                                      roiPtrInputCropRegion);
                    else
                        init_ricap(maxWidth, maxHeight, batchSize, permutationTensor,
                                   roiPtrInputCropRegion);

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_ricap(input, srcDescPtr, output, dstDescPtr, permutationTensor,
                                       roiPtrInputCropRegion, roiTypeSrc, handle, RPP_HOST_BACKEND);
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
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_gridmask(
                            input, srcDescPtr, output, dstDescPtr, tileWidth, gridRatio, gridAngle,
                            translateVector, roiTensorPtrSrc, roiTypeSrc, handle, RPP_HOST_BACKEND);
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
                    // spatterColor.G = 3;
                    // spatterColor.B = 3;

                    // Ink Spatter
                    // spatterColor.R = 5;
                    // spatterColor.G = 20;
                    // spatterColor.B = 64;

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_spatter(input, srcDescPtr, output, dstDescPtr, spatterColor,
                                         roiTensorPtrSrc, roiTypeSrc, handle, RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case CHANNEL_PERMUTE: {
                    testCaseName = "channel_permute";

                    Rpp32u permutationTensor[batchSize * 3];
                    for (i = 0; i < batchSize; i++)
                        fill_perm_values(&permutationTensor[i * 3], qaFlag, additionalParam);

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_channel_permute(
                            input, srcDescPtr, output, dstDescPtr, permutationTensor, handle,
                            RppBackend::RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case COLOR_TO_GREYSCALE: {
                    testCaseName = "color_to_greyscale";

                    RpptSubpixelLayout srcSubpixelLayout = RpptSubpixelLayout::RGBtype;

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_color_to_greyscale(
                            input, srcDescPtr, output, dstDescPtr, srcSubpixelLayout, handle,
                            RppBackend::RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case TENSOR_SUM: {
                    testCaseName = "tensor_sum";

                    if (srcDescPtr->c == 1) reductionFuncResultArrLength = srcDescPtr->n;

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_tensor_sum(
                            input, srcDescPtr, reductionFuncResultArr, reductionFuncResultArrLength,
                            roiTensorPtrSrc, roiTypeSrc, handle, RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case TENSOR_MIN: {
                    testCaseName = "tensor_min";

                    if (srcDescPtr->c == 1) reductionFuncResultArrLength = srcDescPtr->n;

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_tensor_min(
                            input, srcDescPtr, reductionFuncResultArr, reductionFuncResultArrLength,
                            roiTensorPtrSrc, roiTypeSrc, handle, RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case TENSOR_MAX: {
                    testCaseName = "tensor_max";

                    if (srcDescPtr->c == 1) reductionFuncResultArrLength = srcDescPtr->n;

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_tensor_max(
                            input, srcDescPtr, reductionFuncResultArr, reductionFuncResultArrLength,
                            roiTensorPtrSrc, roiTypeSrc, handle, RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case TENSOR_MEAN: {
                    testCaseName = "tensor_mean";

                    if (srcDescPtr->c == 1) reductionFuncResultArrLength = srcDescPtr->n;

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_tensor_mean(
                            input, srcDescPtr, reductionFuncResultArr, reductionFuncResultArrLength,
                            roiTensorPtrSrc, roiTypeSrc, handle, RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case TENSOR_STDDEV: {
                    testCaseName = "tensor_stddev";

                    if (srcDescPtr->c == 1) reductionFuncResultArrLength = srcDescPtr->n;
                    Rpp32f* mean;
                    if (BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == F16_TO_F16)
                        mean = TensorMeanReferenceOutputs_F32[inputChannels].data();
                    else
                        mean = TensorMeanReferenceOutputs_U8[inputChannels].data();

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_tensor_stddev(
                            input, srcDescPtr, reductionFuncResultArr, reductionFuncResultArrLength,
                            mean, roiTensorPtrSrc, roiTypeSrc, handle, RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case SLICE: {
                    testCaseName = "slice";
                    Rpp32u numDims =
                        descriptorPtr3D->numDims - 1;  // exclude batchSize from input dims
                    if (anchorTensor == NULL)
                        anchorTensor =
                            static_cast<Rpp32s*>(calloc(batchSize * numDims, sizeof(Rpp32s)));
                    ;
                    if (shapeTensor == NULL)
                        shapeTensor =
                            static_cast<Rpp32s*>(calloc(batchSize * numDims, sizeof(Rpp32s)));
                    ;
                    if (roiTensor == NULL)
                        roiTensor =
                            static_cast<Rpp32u*>(calloc(batchSize * numDims * 2, sizeof(Rpp32u)));
                    ;
                    bool enablePadding = false;
                    auto fillValue = 0;
                    init_slice(descriptorPtr3D, roiTensorPtrSrc, roiTensor, anchorTensor,
                               shapeTensor);

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();

                    if ((BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F32_TO_F32) &&
                        srcDescPtr->layout == dstDescPtr->layout)
                        errorCodeCapture =
                            rppt_slice(input, descriptorPtr3D, output, descriptorPtr3D,
                                       anchorTensor, shapeTensor, &fillValue, enablePadding,
                                       roiTensor, handle, RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case JPEG_COMPRESSION_DISTORTION: {
                    testCaseName = "jpeg_compression_distortion";

                    Rpp32s qualityTensor[batchSize];
                    for (i = 0; i < batchSize; i++) qualityTensor[i] = 50;

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_jpeg_compression_distortion(
                            input, srcDescPtr, output, dstDescPtr, qualityTensor, roiTensorPtrSrc,
                            roiTypeSrc, handle, RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case POSTERIZE: {
                    testCaseName = "posterize";

                    Rpp8u posterizeLevelBits[batchSize];
                    for (i = 0; i < batchSize; i++) posterizeLevelBits[i] = 3;

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();

                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_posterize(input, srcDescPtr, output, dstDescPtr,
                                                          posterizeLevelBits, roiTensorPtrSrc,
                                                          roiTypeSrc, handle, RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case SOLARIZE: {
                    testCaseName = "solarize";

                    Rpp32f thresholdTensor[batchSize];
                    for (int i = 0; i < batchSize; i++) thresholdTensor[i] = 0.5;

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture =
                            rppt_solarize(input, srcDescPtr, output, dstDescPtr, thresholdTensor,
                                          roiTensorPtrSrc, roiTypeSrc, handle, RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case CHANNEL_DROPOUT: {
                    testCaseName = "channel_dropout";

                    Rpp32f dropoutProbability[batchSize];
                    Rpp32u seed =
                        (qaFlag || DEBUG_MODE) ? DROPOUT_FIXED_SEED : std::random_device{}();
                    for (i = 0; i < batchSize; i++) dropoutProbability[i] = 0.4f;
                    Rpp8u dropoutTensor[batchSize * srcDescPtr->c];
                    generate_channel_dropout_mask(dropoutTensor, dropoutProbability, batchSize,
                                                  srcDescPtr->c, seed);

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_channel_dropout(
                            input, srcDescPtr, output, dstDescPtr, dropoutTensor, roiTensorPtrSrc,
                            roiTypeSrc, handle, RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case CUTOUT_DROPOUT: {
                    testCaseName = "cutout_dropout";
                    Rpp32u boxesInEachImage = 1;
                    Rpp32u seed =
                        (qaFlag || DEBUG_MODE) ? DROPOUT_FIXED_SEED : std::random_device{}();

                    RpptRoiLtrb anchorBoxInfoTensor[batchSize * boxesInEachImage];
                    Rpp32u numBoxesTensor[batchSize * boxesInEachImage];
                    size_t colorBufferElementCount = batchSize * boxesInEachImage * srcDescPtr->c;
                    size_t colorBufferElementSize = sizeof(Rpp32f);
                    if (BitDepthTestMode == U8_TO_U8)
                        colorBufferElementSize = sizeof(Rpp8u);
                    else if (BitDepthTestMode == F16_TO_F16)
                        colorBufferElementSize = sizeof(Rpp16f);
                    else if (BitDepthTestMode == I8_TO_I8)
                        colorBufferElementSize = sizeof(Rpp8s);
                    unsigned char colorBuffer[colorBufferElementCount * colorBufferElementSize];

                    init_cutout_dropout(batchSize, boxesInEachImage, numBoxesTensor,
                                        anchorBoxInfoTensor, roiTensorPtrSrc, srcDescPtr->c,
                                        BitDepthTestMode, seed, 1, colorBuffer);
                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_cutout_dropout(
                            input, srcDescPtr, output, dstDescPtr, anchorBoxInfoTensor, colorBuffer,
                            numBoxesTensor, roiTensorPtrSrc, roiTypeSrc, handle, RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case GRID_DROPOUT: {
                    testCaseName = "grid_dropout";
                    Rpp32u numGridsPerColumn = 10, numGridsPerRow = 10;
                    Rpp32f holeRatio = 0.4f;
                    Rpp32u seed =
                        (qaFlag || DEBUG_MODE) ? DROPOUT_FIXED_SEED : std::random_device{}();

                    Rpp32u boxesInEachImage = numGridsPerRow * numGridsPerColumn;
                    Rpp32u totalBoxes = srcDescPtr->n * boxesInEachImage;
                    RpptRoiLtrb anchorBoxInfoTensor[totalBoxes];
                    Rpp32u maxHoleW = 0, maxHoleH = 0;
                    init_grid_dropout(srcDescPtr->n, anchorBoxInfoTensor, roiTensorPtrSrc,
                                      numGridsPerRow, numGridsPerColumn, maxHoleW, maxHoleH,
                                      holeRatio, seed);

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_grid_dropout(input, srcDescPtr, output, dstDescPtr,
                                                             anchorBoxInfoTensor, boxesInEachImage,
                                                             maxHoleW, maxHoleH, roiTensorPtrSrc,
                                                             roiTypeSrc, handle, RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case EMBOSS: {
                    testCaseName = "emboss";
                    Rpp32u kernelSize = additionalParam;

                    Rpp32f strength[batchSize];
                    for (i = 0; i < batchSize; i++) strength[i] = 1.0f;

                    if (borderType != RpptImageBorderType::REPLICATE) {
                        missingFuncFlag = 1;
                        break;
                    }

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_emboss(
                            input, srcDescPtr, output, dstDescPtr, strength, kernelSize, borderType,
                            roiTensorPtrSrc, roiTypeSrc, handle, RppBackend::RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case RANDOM_ERASE: {
                    testCaseName = "random_erase";
                    Rpp32u boxesInEachImage = 1;
                    Rpp32u seed =
                        (qaFlag || DEBUG_MODE) ? DROPOUT_FIXED_SEED : std::random_device{}();
                    Rpp32u noiseBufferSize = RANDOM_ERASE_NOISE_BUFFER_SIDE *
                                             RANDOM_ERASE_NOISE_BUFFER_SIDE * srcDescPtr->c;
                    RpptRoiLtrb anchorBoxInfoTensor[batchSize * boxesInEachImage];
                    Rpp32f* colorBuffer[noiseBufferSize];
                    init_dropout_random_erase(batchSize, boxesInEachImage, NULL,
                                              anchorBoxInfoTensor, roiTensorPtrSrc, srcDescPtr->c,
                                              BitDepthTestMode, seed, 3, colorBuffer);

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_random_erase(
                            input, srcDescPtr, output, dstDescPtr, anchorBoxInfoTensor, colorBuffer,
                            roiTensorPtrSrc, roiTypeSrc, handle, RppBackend::RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case COARSE_DROPOUT: {
                    testCaseName = "coarse";
                    Rpp32u maxBoxesPerImage = 8;
                    Rpp32u seed =
                        (qaFlag || DEBUG_MODE) ? DROPOUT_FIXED_SEED : std::random_device{}();
                    RpptRoiLtrb anchorBoxInfoTensor[batchSize * maxBoxesPerImage];
                    Rpp32u numOfBoxes[batchSize];
                    init_dropout_erase(batchSize, maxBoxesPerImage, numOfBoxes, anchorBoxInfoTensor,
                                       roiTensorPtrSrc, srcDescPtr->c, BitDepthTestMode, seed, 4);

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                        BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                        errorCodeCapture = rppt_coarse_dropout(
                            input, srcDescPtr, output, dstDescPtr, anchorBoxInfoTensor, numOfBoxes,
                            maxBoxesPerImage, roiTensorPtrSrc, roiTypeSrc, handle,
                            RppBackend::RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                case HISTOGRAM_EQUALIZE: {
                    testCaseName = "histogram_equalize";

                    startWallTime = omp_get_wtime();
                    startCpuTime = clock();
                    if (BitDepthTestMode == U8_TO_U8)
                        errorCodeCapture = rppt_histogram_equalize(
                            input, srcDescPtr, output, dstDescPtr, roiTensorPtrSrc, roiTypeSrc,
                            handle, RppBackend::RPP_HOST_BACKEND);
                    else
                        missingFuncFlag = 1;

                    break;
                }
                default: {
                    missingFuncFlag = 1;
                    break;
                }
            }

            endCpuTime = clock();
            endWallTime = omp_get_wtime();
            cpuTime = ((double)(endCpuTime - startCpuTime)) / CLOCKS_PER_SEC;
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

            maxWallTime = std::max(maxWallTime, wallTime);
            minWallTime = std::min(minWallTime, wallTime);
            avgWallTime += wallTime;
        }
        cpuTime *= 1000;
        wallTime *= 1000;

        if (testType == UNIT_TEST)  // unit test mode
        {
            cout << "\n\n";
            if (noOfIterations > 1)
                cout << "Execution Timings for Iteration " << iterCount + 1 << ":" << endl;
            cout << "CPU Backend Clock Time: " << cpuTime << " ms/batch" << endl;
            cout << "CPU Backend Wall Time: " << wallTime << " ms/batch";

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
                // Reconvert other bit depths to 8u for output display purposes
                convert_output_bitdepth_to_u8(output, outputu8, BitDepthTestMode, oBufferSize,
                                              outputBufferSize, dstDescPtr, invConversionFactor);

                // If DEBUG_MODE is set to 1 dump the outputs to binary files for debugging
                // Skip binary writes for bitdepth conversion tests (e.g., U8_TO_F32, U8_TO_F16) to
                // avoid multiple tests writing to the same output file and accumulating PLN1
                // sections
                bool isNativeBitDepthTest =
                    (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                     BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8);
                if (DEBUG_MODE && iterCount == 0 && isNativeBitDepthTest) {
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
                    }  // end if (!shouldSkipWrite)
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
                    (!(randomOutputCase) && !(nonQACase)))
                    compare_output(output, testCaseName, srcDescPtr, dstDescPtr, dstImgSizes,
                                   batchSize, interpolationTypeName, noiseTypeName,
                                   kernelSizeAndGradientName, additionalParam, testCase, dst,
                                   scriptPath);

                // Calculate exact dstROI in XYWH format for OpenCV dump
                if (roiTypeSrc == RpptRoiType::LTRB)
                    convert_roi(roiTensorPtrDst, RpptRoiType::XYWH, dstDescPtr->n);

                // Check if the ROI values for each input is within the bounds of the max buffer
                // allocated
                RpptROI roiDefault;
                RpptROIPtr roiPtrDefault = &roiDefault;
                roiPtrDefault->xywhROI = {
                    {0, 0}, static_cast<Rpp32s>(dstDescPtr->w), static_cast<Rpp32s>(dstDescPtr->h)};
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

    if (testType == PERFORMANCE_TEST)  // performance tests
    {
        // Display measured times
        maxWallTime *= 1000;
        minWallTime *= 1000;
        avgWallTime *= 1000;
        avgWallTime /= (numRuns * noOfIterations);
        cout << fixed << "\nmax,min,avg wall times in ms/batch = " << maxWallTime << ","
             << minWallTime << "," << avgWallTime;
    }

    cout << endl;

    // Free memory
    free(roiTensorPtrSrc);
    free(roiTensorPtrDst);
    free(dstImgSizes);
    if (anchorTensor != NULL) free(anchorTensor);
    if (shapeTensor != NULL) free(shapeTensor);
    if (roiTensor != NULL) free(roiTensor);
    free(input);
    free(inputu8);
    free(inputu8Second);
    free(outputu8);
    free(input_second);
    free(output);
    if (testCase == REMAP || testCase == LENS_CORRECTION) {
        free(rowRemapTable);
        free(colRemapTable);
    }
    if (reductionTypeCase) free(reductionFuncResultArr);
    if (testCase == PIXELATE) free(interDstPtr);
    if (testCase == CROP_AND_PATCH) {
        free(cropRoi);
        free(patchRoi);
    }
    return 0;
}
