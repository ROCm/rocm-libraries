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

#include "../rpp_test_suite_misc.h"

int main(int argc, char **argv)
{
    // Handle inputs
    const int MIN_ARG_COUNT = 10;
    if (argc < MIN_ARG_COUNT)
    {
        cout << "\nImproper Usage! Needs all arguments!\n";
        cout << "\nUsage: ./Tensor_misc_hip <case number = 0:2> <test type 0/1> <toggle 0/1> <number of dimensions> <batch size> <num runs> <additional param> <dst path> <script path>\n";
        return -1;
    }
    Rpp32u testCase, testType, nDim, batchSize, numRuns, bitDepth, toggle;
    bool qaMode;

    testCase = atoi(argv[1]);
    testType = atoi(argv[2]);
    toggle = atoi(argv[3]);
    nDim = atoi(argv[4]);
    batchSize = atoi(argv[5]);
    numRuns = atoi(argv[6]);
    bitDepth = atoi(argv[7]);
    string dst = argv[9];
    string scriptPath = argv[10];
    qaMode = (testType == 0);
    bool axisMaskCase = (testCase == NORMALIZE || testCase == CONCAT);
    bool permOrderCase = (testCase == TRANSPOSE);
    int additionalParam = (axisMaskCase || permOrderCase) ? atoi(argv[8]) : 1;
    int axisMask = additionalParam, permOrder = additionalParam;

    if (qaMode && batchSize != 3)
    {
        cout<<"QA mode can only run with batchsize 3"<<std::endl;
        return -1;
    }

    string funcName = augmentationMiscMap[testCase];
    if (funcName.empty())
    {
        cout << "\ncase " << testCase << " is not supported\n";
        return -1;
    }

    string func = funcName;
    if (axisMaskCase)
    {
        char additionalParam_char[2];
        std::snprintf(additionalParam_char, sizeof(additionalParam_char), "%d", axisMask);
        func += "_" + std::to_string(nDim) + "d" + "_axisMask";
        func += additionalParam_char;
    }
    if (permOrderCase)
    {
        char additionalParam_char[2];
        std::snprintf(additionalParam_char, sizeof(additionalParam_char), "%d", permOrder);
        func += "_" + std::to_string(nDim) + "d" + "_permOrder";
        func += additionalParam_char;
    }

    // fill roi based on mode and number of dimensions
    Rpp32u *roiTensor, *dstRoiTensor, *roiTensorSecond;
    CHECK_RETURN_STATUS(hipHostMalloc(&roiTensor, nDim * 2 * batchSize, sizeof(Rpp32u)));
    CHECK_RETURN_STATUS(hipHostMalloc(&dstRoiTensor, nDim * 2 * batchSize, sizeof(Rpp32u)));
    fill_roi_values(nDim, batchSize, roiTensor, qaMode);
    memcpy(dstRoiTensor, roiTensor, nDim * 2 * batchSize * sizeof(Rpp32u));
    if(testCase == CONCAT)
    {
        roiTensorSecond = static_cast<Rpp32u *>(calloc(nDim * 2 * batchSize, sizeof(Rpp32u)));
        fill_roi_values(nDim, batchSize, roiTensorSecond, qaMode);
        dstRoiTensor[nDim + axisMask] = roiTensor[nDim + axisMask] + roiTensorSecond[nDim + axisMask]; 
    }

    // set src/dst generic tensor descriptors
    RpptGenericDescPtr srcDescriptorPtrND, srcDescriptorPtrNDSecond, dstDescriptorPtrND;
    CHECK_RETURN_STATUS(hipHostMalloc(&srcDescriptorPtrND, sizeof(RpptGenericDesc)));
    CHECK_RETURN_STATUS(hipHostMalloc(&dstDescriptorPtrND, sizeof(RpptGenericDesc)));

    // set dims and compute strides
    int offSetInBytes = 0;
    set_generic_descriptor(srcDescriptorPtrND, nDim, offSetInBytes, bitDepth, batchSize, roiTensor);
    if(testCase == LOG1P)
        set_generic_descriptor(srcDescriptorPtrND, nDim, offSetInBytes, 6, batchSize, roiTensor);
    set_generic_descriptor(dstDescriptorPtrND, nDim, offSetInBytes, bitDepth, batchSize, dstRoiTensor);
    set_generic_descriptor_layout(srcDescriptorPtrND, dstDescriptorPtrND, nDim, toggle, qaMode);
    if(testCase == CONCAT)
    {
        CHECK_RETURN_STATUS(hipHostMalloc(&srcDescriptorPtrNDSecond, sizeof(RpptGenericDesc)));
        set_generic_descriptor(srcDescriptorPtrNDSecond, nDim, offSetInBytes, bitDepth, batchSize, roiTensorSecond);
        set_generic_descriptor_layout(srcDescriptorPtrNDSecond, dstDescriptorPtrND, nDim, toggle, qaMode);
    }

    Rpp32u iBufferSize = 1;
    Rpp32u oBufferSize = 1;
    Rpp32u iBufferSizeSecond = 1;
    Rpp32u iBufferSizeInBytes = 1;
    Rpp32u oBufferSizeInBytes = 1;
    Rpp32u iBufferSizeSecondInBytes = 1;
    for(int i = 0; i <= nDim; i++)
    {
        iBufferSize *= srcDescriptorPtrND->dims[i];
        oBufferSize *= dstDescriptorPtrND->dims[i];
    }

    iBufferSizeInBytes = iBufferSize * get_size_of_data_type(srcDescriptorPtrND->dataType);
    oBufferSizeInBytes = oBufferSize * get_size_of_data_type(dstDescriptorPtrND->dataType);

    // allocate memory for input / output
    Rpp32f *inputF32 = NULL, *inputF32Second = NULL, *outputF32 = NULL;
    Rpp16s *inputI16 = NULL;
    inputF32 = static_cast<Rpp32f *>(calloc(iBufferSizeInBytes, 1));
    outputF32 = static_cast<Rpp32f *>(calloc(oBufferSizeInBytes, 1));
    if(testCase == CONCAT)
    {
        for(int i = 0; i <= nDim; i++)
            iBufferSizeSecond *= srcDescriptorPtrNDSecond->dims[i];
        iBufferSizeSecondInBytes = iBufferSizeSecond * get_size_of_data_type(srcDescriptorPtrNDSecond->dataType);
        inputF32Second = static_cast<Rpp32f *>(calloc(iBufferSizeSecondInBytes, 1));
    }

    void *input, *inputSecond, *output;
    void *d_input, *d_inputSecond, *d_inputI16, *d_output;
    input = static_cast<Rpp32f *>(calloc(iBufferSizeInBytes, 1));
    inputSecond = static_cast<Rpp32f *>(calloc(iBufferSizeSecondInBytes, 1));
    output = static_cast<Rpp32f *>(calloc(oBufferSizeInBytes, 1));
    CHECK_RETURN_STATUS(hipMalloc(&d_input, iBufferSizeInBytes));
    CHECK_RETURN_STATUS(hipMalloc(&d_output, oBufferSizeInBytes * 2));
    if(testCase == CONCAT)
        CHECK_RETURN_STATUS(hipMalloc(&d_inputSecond, iBufferSizeSecondInBytes));

    // read input data
    if(qaMode)
    {
        read_data(inputF32, nDim, 0, scriptPath, funcName);
        if(testCase == CONCAT)
            read_data(inputF32Second, nDim, 0, scriptPath, funcName);
    }
    else
    {
        std::srand(0);
        for(int i = 0; i < iBufferSize; i++)
            inputF32[i] = static_cast<float>((std::rand() % 255));
        if(testCase == CONCAT)
        {
            for(int i = 0; i < iBufferSizeSecond; i++)
                inputF32Second[i] = static_cast<float>((std::rand() % 255));
        }
    }

    if(testCase == LOG1P)
    {
        inputI16 = static_cast<Rpp16s *>(calloc(iBufferSize, sizeof(Rpp16s)));
        CHECK_RETURN_STATUS(hipMalloc(&d_inputI16, iBufferSize * sizeof(Rpp16s)));
        for(int i = 0; i < iBufferSize; i++)
            inputI16[i] = static_cast<Rpp16s>(inputF32[i]);
        CHECK_RETURN_STATUS(hipMemcpy(d_inputI16, inputI16, iBufferSize * sizeof(Rpp16s), hipMemcpyHostToDevice));
        CHECK_RETURN_STATUS(hipDeviceSynchronize());
    }

    // Convert inputs to correponding bit depth specified by user
    convert_input_bitdepth(inputF32, inputF32Second, input, inputSecond, bitDepth, iBufferSize, iBufferSizeInBytes, srcDescriptorPtrND, testCase);

    // copy data from HOST to HIP
    CHECK_RETURN_STATUS(hipMemcpy(d_input, input, iBufferSizeInBytes, hipMemcpyHostToDevice));
    if(testCase == CONCAT)
        CHECK_RETURN_STATUS(hipMemcpy(d_inputSecond, inputSecond, iBufferSizeSecondInBytes, hipMemcpyHostToDevice));
    CHECK_RETURN_STATUS(hipDeviceSynchronize());

    Rpp32u *permTensor = nullptr;
    if (testCase == TRANSPOSE)
        CHECK_RETURN_STATUS(hipHostMalloc(&permTensor, nDim * sizeof(Rpp32u)));

    rppHandle_t handle;
    hipStream_t stream;
    CHECK_RETURN_STATUS(hipStreamCreate(&stream));
    RppBackend backend = RppBackend::RPP_HIP_BACKEND;
    rppCreate(&handle, batchSize, 0, stream, backend);

    Rpp32f *meanTensor = nullptr, *stdDevTensor = nullptr;
    Rpp32f *meanTensorCPU = nullptr, *stdDevTensorCPU = nullptr;
    bool externalMeanStd = true;

    Rpp32u missingFuncFlag = 0;
    double startWallTime, endWallTime;
    double maxWallTime = 0, minWallTime = 500, avgWallTime = 0, wallTime = 0;
    string testCaseName;

    // case-wise RPP API and measure time script for Unit and Performance test
    cout << "\nRunning " << func << " " << numRuns << " times (each time with a batch size of " << batchSize << ") and computing mean statistics...";
    for(int perfCount = 0; perfCount < numRuns; perfCount++)
    {
        switch(testCase)
        {
            case TRANSPOSE:
            {
                testCaseName  = "transpose";
                fill_perm_values(nDim, permTensor, qaMode, permOrder);

                for(int i = 1; i <= nDim; i++)
                    dstDescriptorPtrND->dims[i] = roiTensor[nDim + permTensor[i - 1]];
                compute_strides(dstDescriptorPtrND);

                startWallTime = omp_get_wtime();
                if (bitDepth == 0 || bitDepth == 1 || bitDepth == 2 || bitDepth == 5)
                    rppt_transpose_gpu(d_input, srcDescriptorPtrND, d_output, dstDescriptorPtrND, permTensor, roiTensor, handle);
                else
                    missingFuncFlag = 1;

                break;
            }
            case NORMALIZE:
            {
                testCaseName  = "normalize";
                float scale = 1.0;
                float shift = 0.0;

                // computeMeanStddev set to 3 means both mean and stddev should be computed internally.
                // Wherein 0th bit used to represent computeMean and 1st bit for computeStddev.
                Rpp8u computeMeanStddev = 3;
                externalMeanStd = !computeMeanStddev; // when mean and stddev is passed from user

                Rpp32u size = 1; // length of mean and stddev tensors differ based on axisMask and nDim
                Rpp32u maxSize = 1;
                for(int batch = 0; batch < batchSize; batch++)
                {
                    size = 1;
                    for(int i = 0; i < nDim; i++)
                        size *= ((axisMask & (int)(pow(2,i))) >= 1) ? 1 : roiTensor[(nDim * 2 * batch) + nDim + i];
                    maxSize = max(maxSize, size);
                }

                // allocate memory if no memory is allocated
                if(meanTensor == nullptr)
                    CHECK_RETURN_STATUS(hipMalloc(&meanTensor, maxSize * batchSize * sizeof(Rpp32f)));

                if(stdDevTensor == nullptr)
                    CHECK_RETURN_STATUS(hipMalloc(&stdDevTensor, maxSize * batchSize * sizeof(Rpp32f)));

                if(!computeMeanStddev)
                {
                    if(meanTensorCPU == nullptr)
                        meanTensorCPU = static_cast<Rpp32f *>(malloc(maxSize * sizeof(Rpp32f)));
                    if(stdDevTensorCPU == nullptr)
                        stdDevTensorCPU = static_cast<Rpp32f *>(malloc(maxSize * sizeof(Rpp32f)));
                    fill_mean_stddev_values(nDim, maxSize, meanTensorCPU, stdDevTensorCPU, qaMode, axisMask, scriptPath);
                    CHECK_RETURN_STATUS(hipMemcpy(meanTensor, meanTensorCPU, maxSize * sizeof(Rpp32f), hipMemcpyHostToDevice));
                    CHECK_RETURN_STATUS(hipMemcpy(stdDevTensor, stdDevTensorCPU, maxSize * sizeof(Rpp32f), hipMemcpyHostToDevice));
                    CHECK_RETURN_STATUS(hipDeviceSynchronize());
                }

                startWallTime = omp_get_wtime();
                if (bitDepth == 0 || bitDepth == 1 || bitDepth == 2 || bitDepth == 5)
                    rppt_normalize_gpu(d_input, srcDescriptorPtrND, d_output, dstDescriptorPtrND, axisMask, meanTensor, stdDevTensor, computeMeanStddev, scale, shift, roiTensor, handle);
                else
                    missingFuncFlag = 1;

                break;
            }
            case LOG:
            {
                testCaseName  = "log";

                startWallTime = omp_get_wtime();
                if (bitDepth == 0 || bitDepth == 1 || bitDepth == 2 || bitDepth == 5)
                    rppt_log_gpu(d_input, srcDescriptorPtrND, d_output, dstDescriptorPtrND, roiTensor, handle);
                else
                    missingFuncFlag = 1;

                break;
            }
            case CONCAT:
            {
                testCaseName  = "concat";
                startWallTime = omp_get_wtime();
                if (bitDepth == 0 || bitDepth == 1 || bitDepth == 2 || bitDepth == 5)
                    rppt_concat_gpu(d_input, d_inputSecond, srcDescriptorPtrND, srcDescriptorPtrNDSecond, d_output, dstDescriptorPtrND, axisMask, roiTensor, roiTensorSecond, handle);
                else
                    missingFuncFlag = 1;

                break;
            }
            case LOG1P:
            {
                testCaseName  = "log1p";

                startWallTime = omp_get_wtime();
                rppt_log1p_gpu(d_inputI16, srcDescriptorPtrND, d_output, dstDescriptorPtrND, roiTensor, handle);

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

        if (missingFuncFlag == 1)
        {
            cout << "\nThe functionality " << func << " doesn't yet exist in RPP\n";
            return RPP_ERROR_NOT_IMPLEMENTED;
        }

        wallTime = endWallTime - startWallTime;
        maxWallTime = std::max(maxWallTime, wallTime);
        minWallTime = std::min(minWallTime, wallTime);
        avgWallTime += wallTime;
    }
    rppDestroy(handle,backend);

    // compare outputs if qaMode is true
    if(qaMode)
    {
        CHECK_RETURN_STATUS(hipDeviceSynchronize());
        CHECK_RETURN_STATUS(hipMemcpy(output, d_output, oBufferSize * sizeof(Rpp32f), hipMemcpyDeviceToHost));
        CHECK_RETURN_STATUS(hipDeviceSynchronize());
        convert_output_bitdepth_to_f32(output, outputF32, bitDepth, oBufferSize, oBufferSizeInBytes, dstDescriptorPtrND);
        compare_output(outputF32, nDim, batchSize, oBufferSize, dst, func, testCaseName, additionalParam, scriptPath, externalMeanStd);
    }
    else
    {
        maxWallTime *= 1000;
        minWallTime *= 1000;
        avgWallTime *= 1000;
        avgWallTime /= numRuns;
        cout << fixed << "\nmax,min,avg wall times in ms/batch = " << maxWallTime << "," << minWallTime << "," << avgWallTime;
    }


    free(inputF32);
    if(testCase == LOG1P)
        free(inputI16);
    free(outputF32);
    CHECK_RETURN_STATUS(hipHostFree(srcDescriptorPtrND));
    CHECK_RETURN_STATUS(hipHostFree(dstDescriptorPtrND));
    CHECK_RETURN_STATUS(hipHostFree(roiTensor));
    if(testCase == LOG1P)
        CHECK_RETURN_STATUS(hipFree(d_inputI16));
    CHECK_RETURN_STATUS(hipFree(d_input));
    CHECK_RETURN_STATUS(hipFree(d_output));
    if(meanTensor != nullptr)
        CHECK_RETURN_STATUS(hipFree(meanTensor));
    if(stdDevTensor != nullptr)
        CHECK_RETURN_STATUS(hipFree(stdDevTensor));
    if (permTensor != nullptr)
        CHECK_RETURN_STATUS(hipHostFree(permTensor));
    if(meanTensorCPU != nullptr)
        free(meanTensorCPU);
    if(stdDevTensorCPU != nullptr)
        free(stdDevTensorCPU);

    return 0;
}
