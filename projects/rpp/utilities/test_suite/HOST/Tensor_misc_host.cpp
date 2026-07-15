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
int main(int argc, char** argv) {
    // Handle inputs
    const int MIN_ARG_COUNT = 10;
    if (argc < MIN_ARG_COUNT) {
        cout << "\nImproper Usage! Needs all arguments!\n";
        cout << "\nUsage: ./Tensor_misc_host <case number = 0:1> <test type 0/1> <toggle 0/1> "
                "<number of dimensions> <batch size> <num runs> <additional param> <dst path> "
                "<script path>\n";
        return -1;
    }
    Rpp32u testCase, testType, nDim, batchSize, numRuns, BitDepthTestMode, toggle;
    bool qaMode;

    testCase = atoi(argv[1]);
    testType = atoi(argv[2]);
    toggle = atoi(argv[3]);
    nDim = atoi(argv[4]);
    batchSize = atoi(argv[5]);
    numRuns = atoi(argv[6]);
    BitDepthTestMode = atoi(argv[7]);
    string dst = argv[9];
    string scriptPath = argv[10];
    qaMode = (testType == 0);
    bool axisMaskCase = (testCase == NORMALIZE || testCase == CONCAT);
    bool permOrderCase = (testCase == TRANSPOSE);
    bool broadCastCase = (testCase == TENSOR_AND_TENSOR || testCase == TENSOR_OR_TENSOR ||
                          testCase == TENSOR_XOR_TENSOR || testCase == TENSOR_ADD_TENSOR ||
                          testCase == TENSOR_SUBTRACT_TENSOR ||
                          testCase == TENSOR_MULTIPLY_TENSOR || testCase == TENSOR_DIVIDE_TENSOR);
    // Golden bitwise references use a linear shifted second tensor with zero past the last sample
    // (no wrap); arithmetic goldens use wraparound (i+1) % N — see second-input fill below.
    bool bitwiseTensorCase = (testCase == TENSOR_AND_TENSOR || testCase == TENSOR_OR_TENSOR ||
                              testCase == TENSOR_XOR_TENSOR);
    int additionalParam = (axisMaskCase || permOrderCase || broadCastCase) ? atoi(argv[8]) : 1;
    int axisMask = additionalParam, permOrder = additionalParam, broadCastFlag = additionalParam;

    if (qaMode && batchSize != 3) {
        cout << "QA mode can only run with batchsize 3" << std::endl;
        return -1;
    }

    string funcName = augmentationMiscMap[testCase];
    if (funcName.empty()) {
        cout << "\ncase " << testCase << " is not supported\n";
        return -1;
    }

    std::string bitdepthStr;  // Variable to store the bit depth as a string
    switch (BitDepthTestMode) {
        case U8_TO_U8:
            bitdepthStr = "u8";
            break;
        case F16_TO_F16:
            bitdepthStr = "f16";
            break;
        case F32_TO_F32:
            bitdepthStr = "f32";
            break;
        case U8_TO_F16:
            bitdepthStr = "u8_f16";
            break;
        case U8_TO_F32:
            bitdepthStr = "u8_f32";
            break;
        case I8_TO_I8:
            bitdepthStr = "i8";
            break;
        case U8_TO_I8:
            bitdepthStr = "u8_i8";
            break;
        case I16_TO_I16:
            bitdepthStr = "i16";
            break;
        case U16_TO_U16:
            bitdepthStr = "u16";
            break;
        case I32_TO_I32:
            bitdepthStr = "i32";
            break;
        case U32_TO_U32:
            bitdepthStr = "u32";
            break;
        case I8_TO_F32:
            bitdepthStr = "i8_f32";
            break;
        case I16_TO_F32:
            bitdepthStr = "i16_f32";
            break;
        case U16_TO_F32:
            bitdepthStr = "u16_f32";
            break;
        case U32_TO_F32:
            bitdepthStr = "u32_f32";
            break;
        case I32_TO_F32:
            bitdepthStr = "i32_f32";
            break;
        default:
            bitdepthStr = "unknown";
            break;
    }

    std::string func = funcName + "_" + std::to_string(nDim) + "d_" + bitdepthStr;
    if (axisMaskCase) func += "_axisMask" + std::to_string(axisMask);
    if (permOrderCase) func += "_permOrder" + std::to_string(permOrder);
    if ((broadCastFlag == 1) && (broadCastCase))
        func += "_broadcast_input2";
    else if ((broadCastFlag == 2) && (broadCastCase))
        func += "_broadcast_input1";

    // fill roi based on mode and number of dimensions
    Rpp32u* roiTensor = static_cast<Rpp32u*>(calloc(nDim * 2 * batchSize, sizeof(Rpp32u)));
    Rpp32u* roiTensorSecond = nullptr;
    Rpp32u* dstRoiTensor = static_cast<Rpp32u*>(calloc(nDim * 2 * batchSize, sizeof(Rpp32u)));

    fill_roi_values(nDim, batchSize, roiTensor, qaMode);
    memcpy(dstRoiTensor, roiTensor, nDim * 2 * batchSize * sizeof(Rpp32u));
    if (testCase == CONCAT) {
        roiTensorSecond = static_cast<Rpp32u*>(calloc(nDim * 2 * batchSize, sizeof(Rpp32u)));
        fill_roi_values(nDim, batchSize, roiTensorSecond, qaMode);
        dstRoiTensor[nDim + axisMask] =
            roiTensor[nDim + axisMask] + roiTensorSecond[nDim + axisMask];
    }
    if (broadCastCase) {
        roiTensorSecond = static_cast<Rpp32u*>(calloc(nDim * 2 * batchSize, sizeof(Rpp32u)));
        fill_roi_values(nDim, batchSize, roiTensorSecond, qaMode, broadCastFlag);
    }

    // set src/dst generic tensor descriptors
    RpptGenericDesc srcDescriptor, srcDescriptorSecond, dstDescriptor;
    RpptGenericDescPtr srcDescriptorPtrND, srcDescriptorPtrNDSecond, dstDescriptorPtrND;
    srcDescriptorPtrND = &srcDescriptor;
    dstDescriptorPtrND = &dstDescriptor;
    int offSetInBytes = 0;

    set_generic_descriptor(srcDescriptorPtrND, nDim, offSetInBytes, BitDepthTestMode, batchSize,
                           roiTensor, false);
    set_generic_descriptor(dstDescriptorPtrND, nDim, offSetInBytes, BitDepthTestMode, batchSize,
                           dstRoiTensor, true);

    set_generic_descriptor_layout(srcDescriptorPtrND, dstDescriptorPtrND, nDim, toggle, qaMode);

    if (testCase == CONCAT || broadCastCase) {
        srcDescriptorPtrNDSecond = &srcDescriptorSecond;
        set_generic_descriptor(srcDescriptorPtrNDSecond, nDim, offSetInBytes, BitDepthTestMode,
                               batchSize, roiTensorSecond, false);
        set_generic_descriptor_layout(srcDescriptorPtrNDSecond, dstDescriptorPtrND, nDim, toggle,
                                      qaMode);
    }
    Rpp32u iBufferSize = 1;
    Rpp32u oBufferSize = 1;
    Rpp32u iBufferSizeSecond = 1;
    Rpp64u iBufferSizeInBytes = 1;
    Rpp64u oBufferSizeInBytes = 1;
    Rpp64u iBufferSizeSecondInBytes = 1;
    for (int i = 0; i <= nDim; i++) {
        iBufferSize *= srcDescriptorPtrND->dims[i];
        oBufferSize *= dstDescriptorPtrND->dims[i];
        if (testCase == CONCAT || broadCastCase)
            iBufferSizeSecond *= srcDescriptorPtrNDSecond->dims[i];
    }

    if (testCase == LOG1P && BitDepthTestMode == I16_TO_F32) {
        // LOG1P expects int16 input (we transform F32->I16 in inputI16), but the 'input' buffer
        // used here is F32 (we store F32 to then convert). So allocate as F32 to hold that data.
        iBufferSizeInBytes = iBufferSize * get_size_of_data_type(RpptDataType::F32);
        oBufferSizeInBytes = oBufferSize * get_size_of_data_type(RpptDataType::F32);
    } else {
        iBufferSizeInBytes = iBufferSize * get_size_of_data_type(srcDescriptorPtrND->dataType);
        oBufferSizeInBytes = oBufferSize * get_size_of_data_type(dstDescriptorPtrND->dataType);
    }

    // allocate memory for input / output
    void *input = nullptr, *inputSecond = nullptr, *output = nullptr;
    input = calloc(iBufferSizeInBytes, 1);
    output = calloc(oBufferSizeInBytes, 1);
    if (testCase == CONCAT || broadCastCase) {
        iBufferSizeSecondInBytes =
            iBufferSizeSecond * get_size_of_data_type(srcDescriptorPtrNDSecond->dataType);
        inputSecond = calloc(iBufferSizeSecondInBytes, 1);
    }
    // read input data
    if (qaMode) {
        if (broadCastCase)
            read_data(input, nDim, 0, scriptPath, funcName, BitDepthTestMode, broadCastFlag);
        else if (BitDepthTestMode == I16_TO_F32)  // log1p
            read_data(input, nDim, 0, scriptPath, funcName, 2);
        else if (BitDepthTestMode == U8_TO_F32)  // log
            read_data(input, nDim, 0, scriptPath, funcName, 0);
        else
            read_data(input, nDim, 0, scriptPath, funcName, BitDepthTestMode);
        if (testCase == CONCAT)
            read_data(inputSecond, nDim, 0, scriptPath, funcName, BitDepthTestMode);
        if (broadCastCase) {
            if (BitDepthTestMode == F32_TO_F32) {
                Rpp32f* inputSecondTemp = static_cast<Rpp32f*>(inputSecond);
                Rpp32f* inputU8 = static_cast<Rpp32f*>(input);
                for (int i = 0; i < iBufferSizeSecond; i++) {
                    if (bitwiseTensorCase && (iBufferSizeSecond == iBufferSize))
                        inputSecondTemp[i] = ((Rpp32u)i + 1u < iBufferSize) ? inputU8[i + 1] : 0.0f;
                    else
                        inputSecondTemp[i] = inputU8[(i + 1) % iBufferSize];
                }
            } else if ((BitDepthTestMode == U8_TO_U8) || (BitDepthTestMode == U8_TO_F32)) {
                Rpp8u* inputSecondTemp = static_cast<Rpp8u*>(inputSecond);
                Rpp8u* inputU8 = static_cast<Rpp8u*>(input);
                for (int i = 0; i < iBufferSizeSecond; i++) {
                    if (bitwiseTensorCase && (iBufferSizeSecond == iBufferSize))
                        inputSecondTemp[i] = ((Rpp32u)i + 1u < iBufferSize) ? inputU8[i + 1] : 0;
                    else
                        inputSecondTemp[i] = inputU8[(i + 1) % iBufferSize];
                }
            } else {
                Rpp8u* inputSecondTemp = static_cast<Rpp8u*>(inputSecond);
                Rpp8u* inputU8 = static_cast<Rpp8u*>(input);
                for (int i = 0; i < iBufferSizeSecond; i++) inputSecondTemp[i] = inputU8[i + 1];
            }
        }
    } else {
        // Generic random data filling based on BitDepthTestMode
        Rpp32f *inputF32 = NULL, *inputF32Second = NULL;
        inputF32 = static_cast<Rpp32f*>(calloc(iBufferSize, sizeof(Rpp32f)));
        if ((testCase == CONCAT) || (broadCastCase))
            inputF32Second = static_cast<Rpp32f*>(calloc(iBufferSizeSecond, sizeof(Rpp32f)));

        // Generate sample values in range based on number of bits for representation
        // Note : I32/U32 can represent higher range of values - Limit set just for testing purposes
        Rpp32u valLimit = 255;
        if ((BitDepthTestMode == I16_TO_I16) || (BitDepthTestMode == U16_TO_U16)) valLimit = 65535;
        if ((BitDepthTestMode == I32_TO_I32) || (BitDepthTestMode == U32_TO_U32)) valLimit = 262143;

        std::srand(0);
        for (int i = 0; i < iBufferSize; i++)
            inputF32[i] = static_cast<float>((std::rand() % valLimit));
        if ((testCase == CONCAT) || (broadCastCase)) {
            for (int i = 0; i < iBufferSizeSecond; i++)
                inputF32Second[i] = static_cast<float>((std::rand() % valLimit));
        }

        convert_input_bitdepth(inputF32, inputF32Second, input, inputSecond, BitDepthTestMode,
                               iBufferSize, iBufferSizeSecond, iBufferSizeInBytes,
                               iBufferSizeSecondInBytes, srcDescriptorPtrND,
                               srcDescriptorPtrNDSecond, testCase);
    }

    Rpp16s* inputI16 = nullptr;
    if (testCase == LOG1P) {
        inputI16 = static_cast<Rpp16s*>(calloc(iBufferSize, sizeof(Rpp16s)));
        Rpp32f* inputF32 = static_cast<Rpp32f*>(input);
        for (int i = 0; i < iBufferSize; i++) inputI16[i] = static_cast<Rpp16s>(inputF32[i]);
    }

    // Set the number of threads to be used by OpenMP pragma for RPP batch processing on host.
    // If numThreads value passed is 0, number of OpenMP threads used by RPP will be set to batch
    // size
    Rpp32u numThreads = 0;
    rppHandle_t handle;
    RppBackend backend = RppBackend::RPP_HOST_BACKEND;
    rppCreate(&handle, batchSize, numThreads, nullptr, backend);

    Rpp32f *meanTensor = nullptr, *stdDevTensor = nullptr;
    bool externalMeanStd = true;

    Rpp32u missingFuncFlag = 0;
    double startWallTime, endWallTime;
    double maxWallTime = 0, minWallTime = 500, avgWallTime = 0, wallTime = 0;
    string testCaseName;

    // case-wise RPP API and measure time script for Unit and Performance test
    cout << "\nRunning " << func << " " << numRuns << " times (each time with a batch size of "
         << batchSize << " images) and computing mean statistics...";
    for (int perfCount = 0; perfCount < numRuns; perfCount++) {
        RppStatus errorCodeCapture = RPP_SUCCESS;
        switch (testCase) {
            case TRANSPOSE: {
                testCaseName = "transpose";
                Rpp32u permTensor[nDim];
                fill_perm_values(nDim, permTensor, qaMode, permOrder);

                for (int i = 1; i <= nDim; i++)
                    dstDescriptorPtrND->dims[i] = roiTensor[nDim + permTensor[i - 1]];
                compute_strides(dstDescriptorPtrND);

                startWallTime = omp_get_wtime();
                if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                    BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                    errorCodeCapture =
                        rppt_transpose(input, srcDescriptorPtrND, output, dstDescriptorPtrND,
                                       permTensor, roiTensor, handle, RPP_HOST_BACKEND);
                else
                    missingFuncFlag = 1;

                break;
            }
            case NORMALIZE: {
                testCaseName = "normalize";
                float scale = 1.0;
                float shift = 0.0;
                // computeMeanStddev set to 3 means both mean and stddev should be computed
                // internally. Wherein 0th bit used to represent computeMean and 1st bit for
                // computeStddev.
                Rpp8u computeMeanStddev = 3;
                externalMeanStd = !computeMeanStddev;  // when mean and stddev is passed from user

                Rpp32u size =
                    1;  // length of mean and stddev tensors differ based on axisMask and nDim
                Rpp32u maxSize = 1;
                for (int batch = 0; batch < batchSize; batch++) {
                    size = 1;
                    for (int i = 0; i < nDim; i++)
                        size *= ((axisMask & (int)(pow(2, i))) >= 1)
                                    ? 1
                                    : roiTensor[(nDim * 2 * batch) + nDim + i];
                    maxSize = max(maxSize, size);
                }

                // allocate memory if no memory is allocated
                if (meanTensor == nullptr)
                    meanTensor = static_cast<Rpp32f*>(calloc(maxSize * batchSize, sizeof(Rpp32f)));

                if (stdDevTensor == nullptr)
                    stdDevTensor =
                        static_cast<Rpp32f*>(calloc(maxSize * batchSize, sizeof(Rpp32f)));

                if (!computeMeanStddev)
                    fill_mean_stddev_values(nDim, maxSize, meanTensor, stdDevTensor, qaMode,
                                            axisMask, scriptPath, BitDepthTestMode);

                startWallTime = omp_get_wtime();
                if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                    BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                    errorCodeCapture =
                        rppt_normalize(input, srcDescriptorPtrND, output, dstDescriptorPtrND,
                                       axisMask, meanTensor, stdDevTensor, computeMeanStddev, scale,
                                       shift, roiTensor, handle, RPP_HOST_BACKEND);
                else
                    missingFuncFlag = 1;

                break;
            }
            case LOG: {
                testCaseName = "log";

                startWallTime = omp_get_wtime();
                if (BitDepthTestMode == U8_TO_F32 || BitDepthTestMode == F16_TO_F16 ||
                    BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_F32)
                    errorCodeCapture =
                        rppt_log(input, srcDescriptorPtrND, output, dstDescriptorPtrND, roiTensor,
                                 handle, RPP_HOST_BACKEND);
                else
                    missingFuncFlag = 1;

                break;
            }
            case CONCAT: {
                testCaseName = "concat";

                startWallTime = omp_get_wtime();
                if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                    BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8)
                    errorCodeCapture =
                        rppt_concat(input, inputSecond, srcDescriptorPtrND,
                                    srcDescriptorPtrNDSecond, output, dstDescriptorPtrND, axisMask,
                                    roiTensor, roiTensorSecond, handle, RPP_HOST_BACKEND);
                else
                    missingFuncFlag = 1;

                break;
            }
            case LOG1P: {
                testCaseName = "log1p";

                startWallTime = omp_get_wtime();
                if (BitDepthTestMode == I16_TO_F32)
                    errorCodeCapture =
                        rppt_log1p(inputI16, srcDescriptorPtrND, output, dstDescriptorPtrND,
                                   roiTensor, handle, RPP_HOST_BACKEND);
                else
                    missingFuncFlag = 1;
                break;
            }
            case TENSOR_AND_TENSOR: {
                testCaseName = "tensor_and_tensor";

                startWallTime = omp_get_wtime();
                if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == I8_TO_I8 ||
                    BitDepthTestMode == I16_TO_I16 || BitDepthTestMode == U16_TO_U16 ||
                    BitDepthTestMode == I32_TO_I32 || BitDepthTestMode == U32_TO_U32) {
                    if (broadCastFlag == 1)
                        errorCodeCapture = rppt_tensor_and_tensor(
                            inputSecond, input, srcDescriptorPtrNDSecond, srcDescriptorPtrND,
                            output, dstDescriptorPtrND, RPP_BROADCAST_ENABLE, roiTensorSecond,
                            roiTensor, handle, RppBackend::RPP_HOST_BACKEND);
                    else if (broadCastFlag == 2)
                        errorCodeCapture = rppt_tensor_and_tensor(
                            input, inputSecond, srcDescriptorPtrND, srcDescriptorPtrNDSecond,
                            output, dstDescriptorPtrND, RPP_BROADCAST_ENABLE, roiTensor,
                            roiTensorSecond, handle, RppBackend::RPP_HOST_BACKEND);
                    else
                        errorCodeCapture = rppt_tensor_and_tensor(
                            input, inputSecond, srcDescriptorPtrND, srcDescriptorPtrNDSecond,
                            output, dstDescriptorPtrND, RPP_BROADCAST_DISABLE, roiTensor,
                            roiTensorSecond, handle, RppBackend::RPP_HOST_BACKEND);
                } else
                    missingFuncFlag = 1;

                break;
            }
            case TENSOR_OR_TENSOR: {
                testCaseName = "tensor_or_tensor";

                startWallTime = omp_get_wtime();
                if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == I8_TO_I8 ||
                    BitDepthTestMode == I16_TO_I16 || BitDepthTestMode == U16_TO_U16 ||
                    BitDepthTestMode == I32_TO_I32 || BitDepthTestMode == U32_TO_U32) {
                    if (broadCastFlag == 1)
                        errorCodeCapture = rppt_tensor_or_tensor(
                            inputSecond, input, srcDescriptorPtrNDSecond, srcDescriptorPtrND,
                            output, dstDescriptorPtrND, RPP_BROADCAST_ENABLE, roiTensorSecond,
                            roiTensor, handle, RppBackend::RPP_HOST_BACKEND);
                    else if (broadCastFlag == 2)
                        errorCodeCapture = rppt_tensor_or_tensor(
                            input, inputSecond, srcDescriptorPtrND, srcDescriptorPtrNDSecond,
                            output, dstDescriptorPtrND, RPP_BROADCAST_ENABLE, roiTensor,
                            roiTensorSecond, handle, RppBackend::RPP_HOST_BACKEND);
                    else
                        errorCodeCapture = rppt_tensor_or_tensor(
                            input, inputSecond, srcDescriptorPtrND, srcDescriptorPtrNDSecond,
                            output, dstDescriptorPtrND, RPP_BROADCAST_DISABLE, roiTensor,
                            roiTensorSecond, handle, RppBackend::RPP_HOST_BACKEND);
                } else
                    missingFuncFlag = 1;

                break;
            }
            case TENSOR_XOR_TENSOR: {
                testCaseName = "tensor_xor_tensor";

                startWallTime = omp_get_wtime();
                if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == I8_TO_I8 ||
                    BitDepthTestMode == I16_TO_I16 || BitDepthTestMode == U16_TO_U16 ||
                    BitDepthTestMode == I32_TO_I32 || BitDepthTestMode == U32_TO_U32) {
                    if (broadCastFlag == 1)
                        errorCodeCapture = rppt_tensor_xor_tensor(
                            inputSecond, input, srcDescriptorPtrNDSecond, srcDescriptorPtrND,
                            output, dstDescriptorPtrND, RPP_BROADCAST_ENABLE, roiTensorSecond,
                            roiTensor, handle, RppBackend::RPP_HOST_BACKEND);
                    else if (broadCastFlag == 2)
                        errorCodeCapture = rppt_tensor_xor_tensor(
                            input, inputSecond, srcDescriptorPtrND, srcDescriptorPtrNDSecond,
                            output, dstDescriptorPtrND, RPP_BROADCAST_ENABLE, roiTensor,
                            roiTensorSecond, handle, RppBackend::RPP_HOST_BACKEND);
                    else
                        errorCodeCapture = rppt_tensor_xor_tensor(
                            input, inputSecond, srcDescriptorPtrND, srcDescriptorPtrNDSecond,
                            output, dstDescriptorPtrND, RPP_BROADCAST_DISABLE, roiTensor,
                            roiTensorSecond, handle, RppBackend::RPP_HOST_BACKEND);
                } else
                    missingFuncFlag = 1;

                break;
            }
            case TENSOR_ADD_TENSOR: {
                testCaseName = "tensor_add_tensor";

                startWallTime = omp_get_wtime();
                if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                    BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8 ||
                    BitDepthTestMode == I16_TO_I16 || BitDepthTestMode == U16_TO_U16 ||
                    BitDepthTestMode == I32_TO_I32 || BitDepthTestMode == U32_TO_U32) {
                    if (broadCastFlag == 1)
                        rppt_tensor_add_tensor(inputSecond, input, srcDescriptorPtrNDSecond,
                                               srcDescriptorPtrND, output, dstDescriptorPtrND,
                                               RPP_BROADCAST_ENABLE, roiTensorSecond, roiTensor,
                                               handle, RPP_HOST_BACKEND);
                    else if (broadCastFlag == 2)
                        rppt_tensor_add_tensor(input, inputSecond, srcDescriptorPtrND,
                                               srcDescriptorPtrNDSecond, output, dstDescriptorPtrND,
                                               RPP_BROADCAST_ENABLE, roiTensor, roiTensorSecond,
                                               handle, RPP_HOST_BACKEND);
                    else
                        rppt_tensor_add_tensor(input, inputSecond, srcDescriptorPtrND,
                                               srcDescriptorPtrNDSecond, output, dstDescriptorPtrND,
                                               RPP_BROADCAST_DISABLE, roiTensor, roiTensorSecond,
                                               handle, RPP_HOST_BACKEND);
                } else
                    missingFuncFlag = 1;

                break;
            }
            case TENSOR_SUBTRACT_TENSOR: {
                testCaseName = "tensor_subtract_tensor";

                startWallTime = omp_get_wtime();
                if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                    BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8 ||
                    BitDepthTestMode == I16_TO_I16 || BitDepthTestMode == U16_TO_U16 ||
                    BitDepthTestMode == I32_TO_I32 || BitDepthTestMode == U32_TO_U32) {
                    if (broadCastFlag == 1)
                        rppt_tensor_subtract_tensor(inputSecond, input, srcDescriptorPtrNDSecond,
                                                    srcDescriptorPtrND, output, dstDescriptorPtrND,
                                                    RPP_BROADCAST_ENABLE, roiTensorSecond,
                                                    roiTensor, handle, RPP_HOST_BACKEND);
                    else if (broadCastFlag == 2)
                        rppt_tensor_subtract_tensor(
                            input, inputSecond, srcDescriptorPtrND, srcDescriptorPtrNDSecond,
                            output, dstDescriptorPtrND, RPP_BROADCAST_ENABLE, roiTensor,
                            roiTensorSecond, handle, RPP_HOST_BACKEND);
                    else
                        rppt_tensor_subtract_tensor(
                            input, inputSecond, srcDescriptorPtrND, srcDescriptorPtrNDSecond,
                            output, dstDescriptorPtrND, RPP_BROADCAST_DISABLE, roiTensor,
                            roiTensorSecond, handle, RPP_HOST_BACKEND);
                } else
                    missingFuncFlag = 1;

                break;
            }
            case TENSOR_MULTIPLY_TENSOR: {
                testCaseName = "tensor_multiply_tensor";

                startWallTime = omp_get_wtime();
                if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == F16_TO_F16 ||
                    BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I8_TO_I8 ||
                    BitDepthTestMode == I16_TO_I16 || BitDepthTestMode == U16_TO_U16 ||
                    BitDepthTestMode == I32_TO_I32 || BitDepthTestMode == U32_TO_U32) {
                    if (broadCastFlag == 1)
                        rppt_tensor_multiply_tensor(inputSecond, input, srcDescriptorPtrNDSecond,
                                                    srcDescriptorPtrND, output, dstDescriptorPtrND,
                                                    RPP_BROADCAST_ENABLE, roiTensorSecond,
                                                    roiTensor, handle, RPP_HOST_BACKEND);
                    else if (broadCastFlag == 2)
                        rppt_tensor_multiply_tensor(
                            input, inputSecond, srcDescriptorPtrND, srcDescriptorPtrNDSecond,
                            output, dstDescriptorPtrND, RPP_BROADCAST_ENABLE, roiTensor,
                            roiTensorSecond, handle, RPP_HOST_BACKEND);
                    else
                        rppt_tensor_multiply_tensor(
                            input, inputSecond, srcDescriptorPtrND, srcDescriptorPtrNDSecond,
                            output, dstDescriptorPtrND, RPP_BROADCAST_DISABLE, roiTensor,
                            roiTensorSecond, handle, RPP_HOST_BACKEND);
                } else
                    missingFuncFlag = 1;

                break;
            }
            case TENSOR_DIVIDE_TENSOR: {
                testCaseName = "tensor_divide_tensor";

                startWallTime = omp_get_wtime();
                if (BitDepthTestMode == U8_TO_F32 || BitDepthTestMode == F16_TO_F16 ||
                    BitDepthTestMode == U16_TO_F32 || BitDepthTestMode == U32_TO_F32 ||
                    BitDepthTestMode == I8_TO_F32 || BitDepthTestMode == I16_TO_F32 ||
                    BitDepthTestMode == I32_TO_F32 || BitDepthTestMode == F32_TO_F32) {
                    if (broadCastFlag == 1)
                        rppt_tensor_divide_tensor(inputSecond, input, srcDescriptorPtrNDSecond,
                                                  srcDescriptorPtrND, output, dstDescriptorPtrND,
                                                  RPP_BROADCAST_ENABLE, roiTensorSecond, roiTensor,
                                                  handle, RPP_HOST_BACKEND);
                    else if (broadCastFlag == 2)
                        rppt_tensor_divide_tensor(
                            input, inputSecond, srcDescriptorPtrND, srcDescriptorPtrNDSecond,
                            output, dstDescriptorPtrND, RPP_BROADCAST_ENABLE, roiTensor,
                            roiTensorSecond, handle, RPP_HOST_BACKEND);
                    else
                        rppt_tensor_divide_tensor(
                            input, inputSecond, srcDescriptorPtrND, srcDescriptorPtrNDSecond,
                            output, dstDescriptorPtrND, RPP_BROADCAST_DISABLE, roiTensor,
                            roiTensorSecond, handle, RPP_HOST_BACKEND);
                } else
                    missingFuncFlag = 1;

                break;
            }
            default: {
                missingFuncFlag = 1;
                break;
            }
        }
        endWallTime = omp_get_wtime();

        if (missingFuncFlag == 1) {
            cout << "\nThe functionality " << func << " doesn't yet exist in RPP\n";
            return RPP_ERROR_NOT_IMPLEMENTED;
        }
        if (errorCodeCapture != RPP_SUCCESS) {
            cout << "\nThe functionality " << func << " returned an error status "
                 << rppStatusToString[errorCodeCapture] << " on run number " << perfCount + 1
                 << " of " << numRuns << " runs.\n";
            return errorCodeCapture;
        }

        wallTime = endWallTime - startWallTime;
        maxWallTime = std::max(maxWallTime, wallTime);
        minWallTime = std::min(minWallTime, wallTime);
        avgWallTime += wallTime;
    }

    if (DEBUG_MODE) {
        // Misc uses F32, so filename is {func}_f32.bin
        std::string binFileName = func + "_f32.bin";
        std::ofstream binFile(binFileName, std::ios::binary | std::ios::trunc);
        if (binFile.is_open()) {
            binFile.write(reinterpret_cast<const char*>((float*)output),
                          oBufferSize * sizeof(float));
            binFile.close();
        }
    }

    if (qaMode) {
        compare_output(output, nDim, batchSize, BitDepthTestMode, oBufferSize, dst, func,
                       testCaseName, additionalParam, scriptPath, broadCastCase ? broadCastFlag : 0,
                       "HOST", externalMeanStd);
    } else {
        maxWallTime *= 1000;
        minWallTime *= 1000;
        avgWallTime *= 1000;
        avgWallTime /= numRuns;
        cout << fixed << "\nmax,min,avg wall times in ms/batch = " << maxWallTime << ","
             << minWallTime << "," << avgWallTime;
    }

    rppDestroy(handle, backend);

    free(input);
    if (inputSecond != nullptr) free(inputSecond);
    free(output);
    if (inputI16 != nullptr) free(inputI16);
    free(roiTensor);
    free(dstRoiTensor);
    if (roiTensorSecond != nullptr) free(roiTensorSecond);
    if (meanTensor != nullptr) free(meanTensor);
    if (stdDevTensor != nullptr) free(stdDevTensor);
    return 0;
}
