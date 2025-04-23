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

#include "../rpp_test_suite_audio.h"

int main(int argc, char **argv)
{
    // handle inputs
    const int MIN_ARG_COUNT = 7;
    if (argc < MIN_ARG_COUNT)
    {
        cout << "\nImproper Usage! Needs all arguments!\n";
        cout << "\nUsage: ./Tensor_audio_host <src folder> <case number = 0:7> <test type 0/1> <numRuns> <batchSize> <dst folder>\n";
        return -1;
    }

    char *src = argv[1];
    int testCase = atoi(argv[2]);
    int testType = atoi(argv[3]);
    int numRuns = atoi(argv[4]);
    int batchSize = atoi(argv[5]);
    char *dst = argv[6];
    string scriptPath = argv[7];

    // validation checks
    if (testType == 0 && batchSize != 3)
    {
        cout << "Error! QA Mode only runs with batchsize 3" << endl;
        return -1;
    }

    // set case names
    string funcName = audioAugmentationMap[testCase];
    if (funcName.empty())
    {
        if (testType == 0)
            cout << "\ncase " << testCase << " is not supported\n";

        return -1;
    }

    // initialize tensor descriptors
    RpptDesc srcDesc, dstDesc;
    RpptDescPtr srcDescPtr, dstDescPtr;
    srcDescPtr = &srcDesc;
    dstDescPtr = &dstDesc;

    // set src/dst data types in tensor descriptors
    srcDescPtr->dataType = RpptDataType::F32;
    dstDescPtr->dataType = RpptDataType::F32;

    // other initializations
    int missingFuncFlag = 0;
    int maxSrcChannels = 0;
    int maxSrcWidth = 0, maxSrcHeight = 0;
    int maxDstWidth = 0, maxDstHeight = 0;
    Rpp64u iBufferSize = 0;
    Rpp64u oBufferSize = 0;
    static int noOfAudioFiles = 0;

    // string ops on function name
    char src1[1000];
    strcpy(src1, src);
    strcat(src1, "/");
    string func = funcName;

    // get number of audio files
    vector<string> audioNames, audioFilesPath;
    search_files_recursive(src, audioNames, audioFilesPath, ".wav");
    noOfAudioFiles = audioNames.size();
    if (noOfAudioFiles < batchSize || ((noOfAudioFiles % batchSize) != 0))
    {
        replicate_last_file_to_fill_batch(audioFilesPath[noOfAudioFiles - 1], audioFilesPath, audioNames, audioNames[noOfAudioFiles - 1], noOfAudioFiles, batchSize);
        noOfAudioFiles = audioNames.size();
    }

    // find max audio dimensions in the input dataset
    maxSrcHeight = 1;
    maxDstHeight = 1;
    set_audio_max_dimensions(audioFilesPath, maxSrcWidth, maxSrcChannels);
    maxDstWidth = maxSrcWidth;

    // set numDims, offset, n/c/h/w values for src/dst
    Rpp32u offsetInBytes = 0;
    set_audio_descriptor_dims_and_strides(srcDescPtr, batchSize, maxSrcHeight, maxSrcWidth, maxSrcChannels, offsetInBytes);
    int maxDstChannels = maxSrcChannels;
    if(testCase == DOWN_MIXING)
    {
        srcDescPtr->numDims = 3;
        maxDstChannels = 1;
    }
    set_audio_descriptor_dims_and_strides(dstDescPtr, batchSize, maxDstHeight, maxDstWidth, maxDstChannels, offsetInBytes);

    // create generic descriptor in case of slice
    RpptGenericDesc descriptor3D;
    RpptGenericDescPtr descriptorPtr3D = &descriptor3D;
    if(testCase == SLICE)
    {
        descriptorPtr3D->numDims = 2;
        descriptorPtr3D->offsetInBytes = 0;
        descriptorPtr3D->dataType = RpptDataType::F32;
        descriptorPtr3D->dims[0] = batchSize;
        descriptorPtr3D->dims[1] = maxSrcWidth;
        descriptorPtr3D->strides[0] = descriptorPtr3D->dims[1];
    }

    // set buffer sizes for src/dst
    if(testCase == MEL_FILTER_BANK)
    {
        iBufferSize = (Rpp64u)MEL_FILTER_BANK_MAX_HEIGHT * (Rpp64u)srcDescPtr->w * (Rpp64u)srcDescPtr->c * (Rpp64u)srcDescPtr->n;
        oBufferSize = (Rpp64u)MEL_FILTER_BANK_MAX_HEIGHT * (Rpp64u)dstDescPtr->w * (Rpp64u)dstDescPtr->c * (Rpp64u)dstDescPtr->n;
    }
    else
    {
        iBufferSize = (Rpp64u)srcDescPtr->h * (Rpp64u)srcDescPtr->w * (Rpp64u)srcDescPtr->c * (Rpp64u)srcDescPtr->n;
        oBufferSize = (Rpp64u)dstDescPtr->h * (Rpp64u)dstDescPtr->w * (Rpp64u)dstDescPtr->c * (Rpp64u)dstDescPtr->n;
    }

    // compute maximum possible buffer size of resample
    Rpp64u resampleMaxBufferSize = dstDescPtr->n * dstDescPtr->strides.nStride * 1.15;
    if (testCase == RESAMPLE)
        oBufferSize = resampleMaxBufferSize;

    // compute maximum possible buffer size of spectrogram
    Rpp64u spectrogramMaxBufferSize = 257 * 3754 * dstDescPtr->n;
    if (testCase == SPECTROGRAM)
        oBufferSize = spectrogramMaxBufferSize;

    // allocate host buffers for input & output
    Rpp32f *inputf32 = (Rpp32f *)calloc(iBufferSize, sizeof(Rpp32f));
    Rpp32f *outputf32 = (Rpp32f *)calloc(oBufferSize, sizeof(Rpp32f));

    // allocate the buffers for audio length and channels
    Rpp32s *srcLengthTensor = (Rpp32s *) calloc(batchSize, sizeof(Rpp32s));
    Rpp32s *channelsTensor = (Rpp32s *) calloc(batchSize, sizeof(Rpp32s));

    // allocate the buffers for src/dst dimensions for each element in batch
    RpptImagePatch *srcDims = (RpptImagePatch *) calloc(batchSize, sizeof(RpptImagePatch));
    RpptImagePatch *dstDims = (RpptImagePatch *) calloc(batchSize, sizeof(RpptImagePatch));

    // buffers used for non silent region detection
    Rpp32s detectedIndex[batchSize], detectionLength[batchSize];

    // RpptResamplingWindow instance used for resample augmentation
    RpptResamplingWindow window;

    // Set the number of threads to be used by OpenMP pragma for RPP batch processing on host.
    // If numThreads value passed is 0, number of OpenMP threads used by RPP will be set to batch size
    Rpp32u numThreads = 0;
    rppHandle_t handle;
    RppBackend backend = RppBackend::RPP_HOST_BACKEND;
    rppCreate(&handle, batchSize, numThreads, nullptr, backend);

    int noOfIterations = static_cast<int>(audioNames.size()) / batchSize;
    double maxWallTime = 0, minWallTime = 500, avgWallTime = 0;
    string testCaseName;
    cout << "\nRunning " << func << " " << numRuns << " times (each time with a batch size of " << batchSize << " audio files) and computing mean statistics...";
    for (int iterCount = 0; iterCount < noOfIterations; iterCount++)
    {
        // read and decode audio and fill the audio dim values
        read_audio_batch_and_fill_dims(srcDescPtr, inputf32, audioFilesPath, iterCount, srcLengthTensor, channelsTensor);
        for (int perfRunCount = 0; perfRunCount < numRuns; perfRunCount++)
        {
            double startWallTime, endWallTime;
            double wallTime;
            switch (testCase)
            {
                case NON_SILENT_REGION_DETECTION:
                {
                    testCaseName = "non_silent_region_detection";
                    Rpp32f cutOffDB = -60.0;
                    Rpp32s windowLength = 2048;
                    Rpp32f referencePower = 0.0f;
                    Rpp32s resetInterval = 8192;

                    startWallTime = omp_get_wtime();
                    rppt_non_silent_region_detection_host(inputf32, srcDescPtr, srcLengthTensor, detectedIndex, detectionLength, cutOffDB, windowLength, referencePower, resetInterval, handle);

                    break;
                }
                case TO_DECIBELS:
                {
                    testCaseName = "to_decibels";
                    Rpp32f cutOffDB = std::log(1e-20);
                    Rpp32f multiplier = std::log(10);
                    Rpp32f referenceMagnitude = 1.0f;

                    for (int i = 0; i < batchSize; i++)
                    {
                        srcDims[i].height = dstDims[i].height = srcLengthTensor[i];
                        srcDims[i].width = dstDims[i].width = 1;
                    }

                    startWallTime = omp_get_wtime();
                    rppt_to_decibels_host(inputf32, srcDescPtr, outputf32, dstDescPtr, srcDims, cutOffDB, multiplier, referenceMagnitude, handle);

                    break;
                }
                case PRE_EMPHASIS_FILTER:
                {
                    testCaseName = "pre_emphasis_filter";
                    Rpp32f coeff[batchSize];
                    for (int i = 0; i < batchSize; i++)
                    {
                        coeff[i] = 0.97;
                        dstDims[i].height = srcLengthTensor[i];
                        dstDims[i].width = 1;
                    }
                    RpptAudioBorderType borderType = RpptAudioBorderType::CLAMP;

                    startWallTime = omp_get_wtime();
                    rppt_pre_emphasis_filter_host(inputf32, srcDescPtr, outputf32, dstDescPtr, srcLengthTensor, coeff, borderType, handle);

                    break;
                }
                case DOWN_MIXING:
                {
                    testCaseName = "down_mixing";
                    bool normalizeWeights = false;
                    Rpp32s srcDimsTensor[batchSize * 2];

                    for (int i = 0, j = 0; i < batchSize; i++, j += 2)
                    {
                        srcDimsTensor[j] = srcLengthTensor[i];
                        srcDimsTensor[j + 1] = channelsTensor[i];
                        dstDims[i].height = srcLengthTensor[i];
                        dstDims[i].width = 1;
                    }

                    startWallTime = omp_get_wtime();
                    rppt_down_mixing_host(inputf32, srcDescPtr, outputf32, dstDescPtr, srcDimsTensor, normalizeWeights, handle);

                    break;
                }
                case SPECTROGRAM:
                {
                    testCaseName = "spectrogram";
                    bool centerWindows = true;
                    bool reflectPadding = true;
                    Rpp32f *windowFn = NULL;
                    Rpp32s power = 2;
                    Rpp32s windowLength = 320;
                    Rpp32s windowStep = 160;
                    Rpp32s nfft = 512;
                    dstDescPtr->layout = RpptLayout::NFT;

                    int windowOffset = 0;
                    if(!centerWindows)
                        windowOffset = windowLength;

                    maxDstWidth = 0;
                    maxDstHeight = 0;
                    init_spectrogram(srcDescPtr, dstDescPtr, dstDims, srcLengthTensor, windowLength,
                                     windowStep, windowOffset, nfft, maxDstHeight, maxDstWidth);

                    // check if the output buffer size is greater than predefined spectrogramMaxBufferSize
                    if (dstDescPtr->n * dstDescPtr->strides.nStride > spectrogramMaxBufferSize)
                    {
                        std::cout << "\nError! Requested spectrogram output size is greater than predefined max size for spectrogram in test suite."
                                     "\nPlease modify spectrogramMaxBufferSize value in test suite for running spectrogram kernel" << std::endl;
                        exit(0);
                    }

                    startWallTime = omp_get_wtime();
                    rppt_spectrogram_host(inputf32, srcDescPtr, outputf32, dstDescPtr, srcLengthTensor, centerWindows, reflectPadding, windowFn, nfft, power, windowLength, windowStep, handle);

                    break;
                }
                case SLICE:
                {
                    testCaseName = "slice";
                    Rpp32u nDim = 1; // testing for 1D slice
                    auto fillValue = 0;
                    bool enablePadding = true;
                    Rpp32u roiTensor[batchSize * nDim * 2];
                    Rpp32s anchorTensor[batchSize * nDim];
                    Rpp32s shapeTensor[batchSize * nDim];

                    // 1D slice arguments
                    for (int i = 0; i < batchSize; i++)
                    {
                        int idx = i * nDim * 2;
                        roiTensor[idx] = 10;
                        roiTensor[idx + 1] = srcLengthTensor[i];
                        anchorTensor[i] = 10;
                        shapeTensor[i] = dstDims[i].width = srcLengthTensor[i] / 2;
                        dstDims[i].height = 1;
                    }

                    startWallTime = omp_get_wtime();
                    rppt_slice_host(inputf32, descriptorPtr3D, outputf32, descriptorPtr3D, anchorTensor, shapeTensor, &fillValue, enablePadding, roiTensor, handle);

                    break;
                }
                case RESAMPLE:
                {
                    testCaseName = "resample";
                    Rpp32f inRateTensor[batchSize];
                    Rpp32f outRateTensor[batchSize];
                    Rpp32s srcDimsTensor[batchSize * 2];

                    maxDstWidth = 0;
                    for(int i = 0, j = 0; i < batchSize; i++, j += 2)
                    {
                        inRateTensor[i] = 16000;
                        outRateTensor[i] = 16000 * 1.15f;
                        Rpp32f scaleRatio = outRateTensor[i] / inRateTensor[i];
                        srcDimsTensor[j] = srcLengthTensor[i];
                        srcDimsTensor[j + 1] = channelsTensor[i];
                        dstDims[i].width = static_cast<int>(std::ceil(scaleRatio * srcLengthTensor[i]));
                        dstDims[i].height = 1;
                        maxDstWidth = std::max(maxDstWidth, static_cast<int>(dstDims[i].width));
                    }
                    Rpp32f quality = 50.0f;
                    Rpp32s lobes = std::round(0.007 * quality * quality - 0.09 * quality + 3);
                    Rpp32s lookupSize = lobes * 64 + 1;
                    windowed_sinc(window, lookupSize, lobes);

                    dstDescPtr->w = maxDstWidth;
                    dstDescPtr->strides.nStride = dstDescPtr->c * dstDescPtr->w * dstDescPtr->h;

                    // check if the required output buffer size is greater than predefined resampleMaxBufferSize
                    if (dstDescPtr->n * dstDescPtr->strides.nStride > resampleMaxBufferSize)
                    {
                        std::cout << "\nError! Requested resample output size is greater than predefined max size for resample in test suite."
                                     "\nPlease modify resampleMaxBufferSize value in test suite as per your requirements for running resample kernel" << std::endl;
                        exit(0);
                    }

                    startWallTime = omp_get_wtime();
                    rppt_resample_host(inputf32, srcDescPtr, outputf32, dstDescPtr, inRateTensor, outRateTensor, srcDimsTensor, window, handle);

                    break;
                }
                case MEL_FILTER_BANK:
                {
                    testCaseName = "mel_filter_bank";

                    Rpp32f sampleRate = 16000;
                    Rpp32f minFreq = 0.0;
                    Rpp32f maxFreq = sampleRate / 2;
                    RpptMelScaleFormula melFormula = RpptMelScaleFormula::SLANEY;
                    Rpp32s numFilter = 80;
                    bool normalize = true;
                    // (height, width) for each tensor in a batch for given QA inputs.
                    Rpp32s srcDimsTensor[] = {257, 225, 257, 211, 257, 214};

                    init_mel_filter_bank(&inputf32, &outputf32, srcDescPtr, dstDescPtr, dstDims, offsetInBytes, numFilter, batchSize, srcDimsTensor, scriptPath, testType);

                    startWallTime = omp_get_wtime();
                    rppt_mel_filter_bank_host(inputf32, srcDescPtr, outputf32, dstDescPtr, srcDimsTensor, maxFreq, minFreq, melFormula, numFilter, sampleRate, normalize, handle);

                    break;
                }
                default:
                {
                    missingFuncFlag = 1;
                    break;
                }
            }

            endWallTime = omp_get_wtime();
            if (missingFuncFlag == 1)
            {
                cout << "\nThe functionality " << func << " doesn't yet exist in RPP\n";
                return -1;
            }

            wallTime = endWallTime - startWallTime;
            maxWallTime = std::max(maxWallTime, wallTime);
            minWallTime = std::min(minWallTime, wallTime);
            avgWallTime += wallTime;
        }

        // QA mode - verify outputs with golden outputs. Below code doesn’t run for performance tests
        if (testType == 0)
        {
            if (testCase == NON_SILENT_REGION_DETECTION)
                verify_non_silent_region_detection(detectedIndex, detectionLength, testCaseName, batchSize, audioNames, dst);
            else
                verify_output(outputf32, dstDescPtr, dstDims, testCaseName, dst, scriptPath, "HOST");

            /* Dump the outputs to csv files for debugging
            Runs only if
            1. DEBUG_MODE is enabled
            2. Current iteration is 1st iteration
            3. Test case is not 0 */
            if (DEBUG_MODE && iterCount == 0 && testCase != 0)
            {
                std::ofstream refFile;
                refFile.open(func + ".csv");
                for (int i = 0; i < oBufferSize; i++)
                    refFile << *(outputf32 + i) << "\n";
                refFile.close();
            }
        }
    }
    rppDestroy(handle, backend);

    // performance test mode
    if (testType == 1)
    {
        // display measured times
        maxWallTime *= 1000;
        minWallTime *= 1000;
        avgWallTime *= 1000;
        avgWallTime /= (numRuns * noOfIterations);
        cout << fixed << "\nmax,min,avg wall times in ms/batch = " << maxWallTime << "," << minWallTime << "," << avgWallTime;
    }

    cout << endl;

    // free memory
    free(srcLengthTensor);
    free(channelsTensor);
    free(srcDims);
    free(dstDims);
    free(inputf32);
    free(outputf32);

    return 0;
}
