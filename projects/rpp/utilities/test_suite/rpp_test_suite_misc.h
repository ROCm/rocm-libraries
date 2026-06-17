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

#include <omp.h>
#include <string.h>

#include <array>
#include <iostream>
#include <map>

#include "rpp.h"
#include "rpp_test_suite_common.h"

// Cutoff values for misc kernels listed for HOST backend followed by HIP
// Each entry: {testCaseName, {HOST_cutoff, HIP_cutoff}}
static const std::map<string, std::vector<double>> miscCutOff = {
    {"transpose", {1e-6, 1e-6}},
    {"normalize", {1e-4, 1e-4}},
    {"log", {1e-6, 1e-6}},
    {"concat", {1e-6, 1e-6}},
    {"log1p", {1e-6, 1e-6}},
    {"tensor_add_tensor", {1e-6, 1e-6}},
    {"tensor_subtract_tensor", {1e-6, 1e-6}},
    {"tensor_multiply_tensor", {1e-6, 1e-6}},
    {"tensor_divide_tensor", {1e-6, 1e-6}}};

std::map<int, string> augmentationMiscMap = {{0, "transpose"},
                                             {1, "normalize"},
                                             {2, "log"},
                                             {3, "concat"},
                                             {4, "log1p"},
                                             {5, "tensor_and_tensor"},
                                             {6, "tensor_or_tensor"},
                                             {7, "tensor_xor_tensor"},
                                             {8, "tensor_add_tensor"},
                                             {9, "tensor_subtract_tensor"},
                                             {10, "tensor_multiply_tensor"},
                                             {11, "tensor_divide_tensor"}};

enum Augmentation {
    TRANSPOSE = 0,
    NORMALIZE = 1,
    LOG = 2,
    CONCAT = 3,
    LOG1P = 4,
    TENSOR_AND_TENSOR = 5,
    TENSOR_OR_TENSOR = 6,
    TENSOR_XOR_TENSOR = 7,
    TENSOR_ADD_TENSOR = 8,
    TENSOR_SUBTRACT_TENSOR = 9,
    TENSOR_MULTIPLY_TENSOR = 10,
    TENSOR_DIVIDE_TENSOR = 11
};

// Compute strides given Generic Tensor
void compute_strides(RpptGenericDescPtr descriptorPtr) {
    if (descriptorPtr->numDims > 0) {
        uint64_t v = 1;
        for (int i = descriptorPtr->numDims - 1; i > 0; i--) {
            descriptorPtr->strides[i] = v;
            v *= descriptorPtr->dims[i];
        }
        descriptorPtr->strides[0] = v;
    }
}

// Retrieve path for bin file
string get_path(Rpp32u nDim, Rpp32u readType, string scriptPath, string testCase,
                Rpp32u BitDepthTestMode, int broadCastFlag, bool isMeanStd = false) {
    string folderPath, suffix, bitDepthStr;
    if (BitDepthTestMode == U8_TO_U8)
        bitDepthStr = "u8";
    else if (BitDepthTestMode == F32_TO_F32)
        bitDepthStr = "f32";
    else if (BitDepthTestMode == U8_TO_F32)
        bitDepthStr = "u8";
    else if (BitDepthTestMode == I8_TO_F32)
        bitDepthStr = "f32";

    if (readType == 0)  // Input
    {
        folderPath = "/../TEST_MISC_FILES/";
        if (isMeanStd) {
            suffix =
                std::to_string(nDim) + "d_mean_std.bin";  // mean/std files have no bit depth suffix
        } else {
            suffix = std::to_string(nDim) + "d_input_" + bitDepthStr + ".bin";
        }
    } else if (readType == 1)  // Output
    {
        folderPath = "/../REFERENCE_OUTPUTS_MISC/" + testCase + "/";
        suffix = testCase + "_" + std::to_string(nDim) + "d_output_" + bitDepthStr + ".bin";
    }

    return scriptPath + folderPath + suffix;
}

// Read data from Bin file
template <typename T>
void read_data(T* data, Rpp32u nDim, Rpp32u readType, string scriptPath, string testCase,
               Rpp32u BitDepthTestMode, int broadCastFlag = 0, bool isMeanStd = false) {
    if (nDim < 2 || nDim > 4) {
        std::cout << "\nGolden Inputs / Outputs are generated only for 2D/3D/4D data" << std::endl;
        exit(0);
    }
    std::string dataPath =
        get_path(nDim, readType, scriptPath, testCase, BitDepthTestMode, broadCastFlag, isMeanStd);
    read_bin_file(dataPath, data);
}

// Fill the starting indices and length of ROI values
void fill_roi_values(Rpp32u nDim, Rpp32u batchSize, Rpp32u* roiTensor, bool qaMode,
                     Rpp32u broadCastFlag = 0) {
    if (qaMode) {
        switch (nDim) {
            case 2: {
                std::array<Rpp32u, 4> roi = {0, 0, 100, 100};
                if ((broadCastFlag == 1) || (broadCastFlag == 2)) roi = {0, 0, 100, 1};
                for (int i = 0, j = 0; i < batchSize; i++, j += 4)
                    std::copy(roi.begin(), roi.end(), &roiTensor[j]);
                break;
            }
            case 3: {
                std::array<Rpp32u, 6> roi = {0, 0, 0, 25, 25, 32};
                if ((broadCastFlag == 1) || (broadCastFlag == 2)) roi = {0, 0, 0, 25, 25, 1};
                for (int i = 0, j = 0; i < batchSize; i++, j += 6)
                    std::copy(roi.begin(), roi.end(), &roiTensor[j]);
                break;
            }
            case 4: {
                std::array<Rpp32u, 8> roi = {0, 0, 0, 0, 4, 10, 25, 40};
                if ((broadCastFlag == 1) || (broadCastFlag == 2)) roi = {0, 0, 0, 0, 4, 10, 25, 1};
                for (int i = 0, j = 0; i < batchSize; i++, j += 8)
                    std::copy(roi.begin(), roi.end(), &roiTensor[j]);
                break;
            }
        }
    } else {
        switch (nDim) {
            case 2: {
                std::array<Rpp32u, 4> roi = {0, 0, 1920, 1080};
                if ((broadCastFlag == 1) || (broadCastFlag == 2)) roi = {0, 0, 1920, 1};
                for (int i = 0, j = 0; i < batchSize; i++, j += 4)
                    std::copy(roi.begin(), roi.end(), &roiTensor[j]);
                break;
            }
            case 3: {
                std::array<Rpp32u, 6> roi = {0, 0, 0, 3, 1920, 1080};
                if ((broadCastFlag == 1) || (broadCastFlag == 2)) roi = {0, 0, 0, 3, 1920, 1};
                for (int i = 0, j = 0; i < batchSize; i++, j += 6)
                    std::copy(roi.begin(), roi.end(), &roiTensor[j]);
                break;
            }
            case 4: {
                std::array<Rpp32u, 8> roi = {0, 0, 0, 0, 1, 128, 128, 128};
                if ((broadCastFlag == 1) || (broadCastFlag == 2))
                    roi = {0, 0, 0, 0, 1, 128, 128, 1};
                for (int i = 0, j = 0; i < batchSize; i++, j += 8)
                    std::copy(roi.begin(), roi.end(), &roiTensor[j]);
                break;
            }
            default: {
                // if nDim is not 2/3/4 and mode choosen is not QA
                for (int i = 0; i < batchSize; i++) {
                    int startIndex = i * nDim * 2;
                    int lengthIndex = startIndex + nDim;
                    for (int j = 0; j < nDim; j++) {
                        roiTensor[startIndex + j] = 0;
                        roiTensor[lengthIndex + j] =
                            std::rand() %
                            10;  // limiting max value in a dimension to 10 for testing purposes
                    }
                    if ((broadCastFlag == 1) || (broadCastFlag == 2))
                        roiTensor[lengthIndex + nDim - 1] = 1;
                }
                break;
            }
        }
    }
}

// Set layout for generic descriptor
void set_generic_descriptor_layout(RpptGenericDescPtr srcDescriptorPtrND,
                                   RpptGenericDescPtr dstDescriptorPtrND, Rpp32u nDim, int toggle,
                                   int qaMode) {
    if (qaMode && !toggle) {
        switch (nDim) {
            case 2: {
                srcDescriptorPtrND->layout = RpptLayout::NHWC;
                dstDescriptorPtrND->layout = RpptLayout::NHWC;
                break;
            }
            case 3: {
                srcDescriptorPtrND->layout = RpptLayout::NHWC;
                dstDescriptorPtrND->layout = RpptLayout::NHWC;
                break;
            }
            case 4: {
                srcDescriptorPtrND->layout = RpptLayout::NDHWC;
                dstDescriptorPtrND->layout = RpptLayout::NDHWC;
                break;
            }
            default: {
                cout << "Error! QA mode is supported only for 2D/3D/4D inputs" << endl;
                exit(0);
            }
        }
    } else if (nDim == 3) {
        if (toggle) {
            srcDescriptorPtrND->layout = RpptLayout::NHWC;
            dstDescriptorPtrND->layout = RpptLayout::NCHW;
        }
    } else {
        srcDescriptorPtrND->layout = RpptLayout::NDHWC;
        dstDescriptorPtrND->layout = RpptLayout::NDHWC;
    }
}

// sets generic descriptor numDims, offsetInBytes, bitdepth, dims and strides
inline void set_generic_descriptor(RpptGenericDescPtr descriptorPtr3D, int nDim, int offsetInBytes,
                                   int BitDepthTestMode, int batchSize, Rpp32u* roiTensor,
                                   bool isDestination) {
    descriptorPtr3D->numDims = nDim + 1;
    descriptorPtr3D->offsetInBytes = offsetInBytes;
    switch (BitDepthTestMode) {
        case U8_TO_U8:
            descriptorPtr3D->dataType = RpptDataType::U8;
            break;
        case F16_TO_F16:
            descriptorPtr3D->dataType = RpptDataType::F16;
            break;
        case F32_TO_F32:
            descriptorPtr3D->dataType = RpptDataType::F32;
            break;
        case I8_TO_I8:
            descriptorPtr3D->dataType = RpptDataType::I8;
            break;
        case U8_TO_F32:
            descriptorPtr3D->dataType = isDestination ? RpptDataType::F32 : RpptDataType::U8;
            break;
        case U8_TO_F16:
            descriptorPtr3D->dataType = isDestination ? RpptDataType::F16 : RpptDataType::U8;
            break;
        case U8_TO_I8:
            descriptorPtr3D->dataType = isDestination ? RpptDataType::I8 : RpptDataType::U8;
            break;
        case I16_TO_I16:
            descriptorPtr3D->dataType = RpptDataType::I16;
            break;
        case U16_TO_U16:
            descriptorPtr3D->dataType = RpptDataType::U16;
            break;
        case I32_TO_I32:
            descriptorPtr3D->dataType = RpptDataType::I32;
            break;
        case U32_TO_U32:
            descriptorPtr3D->dataType = RpptDataType::U32;
            break;
        case I8_TO_F32:
            descriptorPtr3D->dataType = isDestination ? RpptDataType::F32 : RpptDataType::I8;
            break;
        case I16_TO_F32:
            descriptorPtr3D->dataType = isDestination ? RpptDataType::F32 : RpptDataType::I16;
            break;
        case U16_TO_F32:
            descriptorPtr3D->dataType = isDestination ? RpptDataType::F32 : RpptDataType::U16;
            break;
        case U32_TO_F32:
            descriptorPtr3D->dataType = isDestination ? RpptDataType::F32 : RpptDataType::U32;
            break;
        case I32_TO_F32:
            descriptorPtr3D->dataType = isDestination ? RpptDataType::F32 : RpptDataType::I32;
            break;
        default:
            descriptorPtr3D->dataType = RpptDataType::U8;
            break;
    }

    descriptorPtr3D->dims[0] = batchSize;
    for (int i = 1; i <= nDim; i++) descriptorPtr3D->dims[i] = roiTensor[nDim + i - 1];
    compute_strides(descriptorPtr3D);
}

// Strides used to locate the corresponding mean and stddev values (based on axisMask)
// within the input bin files for 2D normalization cases.
// These strides are precomputed for various combinations of dimensions and axes
// for the default QA test case: input shape = 100x100.
std::map<Rpp32s, Rpp32u> paramStrideMap2D = {{1, 0}, {2, 100}, {3, 200}};

// Strides used to locate the corresponding mean and stddev values (based on axisMask)
// within the input bin files for 3D normalization cases.
// These strides are precomputed for various combinations of dimensions and axes
// for the default QA test case: input shape = 25x25x32.
std::map<Rpp32s, Rpp32u> paramStrideMap3D = {{1, 0},    {2, 800},  {3, 1600}, {4, 1632},
                                             {5, 2257}, {6, 2282}, {7, 2307}};

// Strides used to locate the corresponding mean and stddev values (based on axisMask)
// within the input bin files for 4D normalization cases.
// These strides are precomputed for various combinations of dimensions and axes
// for the default QA test case: input shape = 4x10x25x40.
std::map<Rpp32s, Rpp32u> paramStrideMap4D = {{1, 0},      {2, 1000},   {3, 2600},   {4, 2640},
                                             {5, 6640},   {6, 6740},   {7, 6900},   {8, 6904},
                                             {9, 16904},  {10, 17154}, {11, 17554}, {12, 17564},
                                             {13, 18564}, {14, 18589}, {15, 18629}};

// fill the mean and stddev values used for normalize
void fill_mean_stddev_values(Rpp32u nDim, Rpp32u size, Rpp32f* meanTensor, Rpp32f* stdDevTensor,
                             bool qaMode, int axisMask, string scriptPath,
                             Rpp32u BitDepthTestMode) {
    if (qaMode) {
        Rpp32u numValues, paramStride;
        switch (nDim) {
            case 2: {
                numValues = 100 + 100 + 1;
                paramStride = paramStrideMap2D[axisMask];
                break;
            }
            case 3: {
                numValues = 400 + 400 + 8 + 2500 + 50 + 50 + 1;
                paramStride = paramStrideMap3D[axisMask];
                break;
            }
            case 4: {
                numValues = 18630;
                paramStride = paramStrideMap4D[axisMask];
                break;
            }
            default: {
                cout << "Error! QA mode is supported only for 2D/3D/4D inputs" << endl;
                exit(0);
            }
        }
        std::vector<Rpp32f> paramBuf(numValues * 2);
        Rpp32f* data = paramBuf.data();
        read_data(data, nDim, 0, scriptPath, "normalize", BitDepthTestMode, 0, true);
        memcpy(meanTensor, data + paramStride, size * sizeof(Rpp32f));
        memcpy(stdDevTensor, data + numValues + paramStride, size * sizeof(Rpp32f));
    } else {
        for (int j = 0; j < size; j++) {
            meanTensor[j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            stdDevTensor[j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
    }
}

// fill the permutation values used for transpose
void fill_perm_values(Rpp32u nDim, Rpp32u* permTensor, bool qaMode, int permOrder) {
    if (qaMode) {
        switch (nDim) {
            case 2: {
                // HW->WH
                permTensor[0] = 1;
                permTensor[1] = 0;
                break;
            }
            case 3: {
                // HWC->WHC
                if (permOrder == 1) {
                    permTensor[0] = 1;
                    permTensor[1] = 0;
                    permTensor[2] = 2;
                }
                // HWC->HCW
                else if (permOrder == 2) {
                    permTensor[0] = 0;
                    permTensor[1] = 2;
                    permTensor[2] = 1;
                }
                break;
            }
            case 4: {
                // NHWC -> NCHW
                if (permOrder == 1) {
                    permTensor[0] = 0;  // N
                    permTensor[1] = 3;  // C
                    permTensor[2] = 1;  // H
                    permTensor[3] = 2;  // W
                }
                // NCHW -> NHWC
                else if (permOrder == 2) {
                    permTensor[0] = 0;  // N
                    permTensor[1] = 2;  // H
                    permTensor[2] = 3;  // W
                    permTensor[3] = 1;  // C
                }
                // NHWC -> HWCN
                else if (permOrder == 3) {
                    permTensor[0] = 1;  // H
                    permTensor[1] = 2;  // W
                    permTensor[2] = 3;  // C
                    permTensor[3] = 0;  // N
                }
                // Identity permutation (no change)
                else {
                    permTensor[0] = 0;
                    permTensor[1] = 1;
                    permTensor[2] = 2;
                    permTensor[3] = 3;
                }
                break;
            }
            default: {
                cout << "Error! QA mode is supported only for 2D/3D/4D inputs" << endl;
                exit(0);
            }
        }
    } else {
        for (int i = 0; i < nDim; i++) permTensor[i] = nDim - 1 - i;
    }
}

Rpp32u get_bin_size(Rpp32u nDim, Rpp32u readType, string scriptPath, string testCase,
                    Rpp32u BitDepthTestMode, int broadCastFlag = 0) {
    string refFile =
        get_path(nDim, readType, scriptPath, testCase, BitDepthTestMode, broadCastFlag);
    std::ifstream filestream(refFile, ios_base::in | ios_base::binary);
    filestream.seekg(0, ios_base::end);
    Rpp32u filesize = filestream.tellg();
    return filesize;
}

inline size_t get_size_of_data_type(RpptDataType dataType) {
    if (dataType == RpptDataType::U8)
        return sizeof(Rpp8u);
    else if (dataType == RpptDataType::I8)
        return sizeof(Rpp8s);
    else if (dataType == RpptDataType::F16)
        return sizeof(Rpp16f);
    else if (dataType == RpptDataType::F32)
        return sizeof(Rpp32f);
    else if (dataType == RpptDataType::I16)
        return sizeof(Rpp16s);
    else if (dataType == RpptDataType::U16)
        return sizeof(Rpp16u);
    else if (dataType == RpptDataType::I32)
        return sizeof(Rpp32s);
    else if (dataType == RpptDataType::U32)
        return sizeof(Rpp32u);
    else
        return 0;
}

// Convert input from F32 to corresponding bit depth specified by user
inline void convert_input_bitdepth(Rpp32f* inputF32, Rpp32f* inputF32Second, void* output,
                                   void* outputSecond, Rpp32s BitDepthTestMode, Rpp64u ioBufferSize,
                                   Rpp64u ioBufferSizeSecond, Rpp64u outputBufferSize,
                                   Rpp64u outputBufferSizeSecond,
                                   RpptGenericDescPtr srcGenericDescPtr,
                                   RpptGenericDescPtr srcDescriptorPtrNDSecond, Rpp32s testCase) {
    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == U8_TO_F32 ||
        BitDepthTestMode == U8_TO_F16)  // U8 case
    {
        Rpp8u* outputU8 = static_cast<Rpp8u*>(output) + srcGenericDescPtr->offsetInBytes;
        for (Rpp32s i = 0; i < ioBufferSize; i++)
            outputU8[i] = static_cast<Rpp8u>(std::clamp(std::round(inputF32[i]), 0.0f, 255.0f));

        if (testCase == CONCAT || testCase == TENSOR_AND_TENSOR || testCase == TENSOR_OR_TENSOR ||
            testCase == TENSOR_XOR_TENSOR || testCase == TENSOR_ADD_TENSOR ||
            testCase == TENSOR_SUBTRACT_TENSOR || testCase == TENSOR_MULTIPLY_TENSOR ||
            testCase == TENSOR_DIVIDE_TENSOR) {
            Rpp8u* outputU8Second =
                static_cast<Rpp8u*>(outputSecond) + srcDescriptorPtrNDSecond->offsetInBytes;
            for (Rpp32s i = 0; i < ioBufferSizeSecond; i++)
                outputU8Second[i] =
                    static_cast<Rpp8u>(std::clamp(std::round(inputF32Second[i]), 0.0f, 255.0f));
        }
    } else if (BitDepthTestMode == F16_TO_F16)  // F16 case
    {
        Rpp16f* outputF16 = reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(output) +
                                                      srcGenericDescPtr->offsetInBytes);
        for (Rpp32s i = 0; i < ioBufferSize; i++)
            outputF16[i] =
                static_cast<Rpp16f>(std::clamp(inputF32[i], -65504.0f, 65504.0f));  // F16 range

        if (testCase == CONCAT || testCase == TENSOR_AND_TENSOR || testCase == TENSOR_OR_TENSOR ||
            testCase == TENSOR_XOR_TENSOR || testCase == TENSOR_ADD_TENSOR ||
            testCase == TENSOR_SUBTRACT_TENSOR || testCase == TENSOR_MULTIPLY_TENSOR ||
            testCase == TENSOR_DIVIDE_TENSOR) {
            Rpp16f* outputF16Second = reinterpret_cast<Rpp16f*>(
                static_cast<Rpp8u*>(outputSecond) + srcDescriptorPtrNDSecond->offsetInBytes);
            for (Rpp32s i = 0; i < ioBufferSizeSecond; i++)
                outputF16Second[i] =
                    static_cast<Rpp16f>(std::clamp(inputF32Second[i], -65504.0f, 65504.0f));
        }
    } else if (BitDepthTestMode == F32_TO_F32)  // F32 case (No conversion needed)
    {
        memcpy(output, inputF32, outputBufferSize);
        if (testCase == CONCAT || testCase == TENSOR_AND_TENSOR || testCase == TENSOR_OR_TENSOR ||
            testCase == TENSOR_XOR_TENSOR || testCase == TENSOR_ADD_TENSOR ||
            testCase == TENSOR_SUBTRACT_TENSOR || testCase == TENSOR_MULTIPLY_TENSOR ||
            testCase == TENSOR_DIVIDE_TENSOR)
            memcpy(outputSecond, inputF32Second, outputBufferSizeSecond);
    } else if (BitDepthTestMode == I8_TO_I8 || BitDepthTestMode == I8_TO_F32)  // I8 case
    {
        Rpp8s* outputI8 = static_cast<Rpp8s*>(output) + srcGenericDescPtr->offsetInBytes;
        for (int i = 0; i < ioBufferSize; i++)
            outputI8[i] =
                static_cast<Rpp8s>(std::clamp(std::round(inputF32[i]) - 128, -128.0f, 127.0f));

        if (testCase == CONCAT || testCase == TENSOR_AND_TENSOR || testCase == TENSOR_OR_TENSOR ||
            testCase == TENSOR_XOR_TENSOR || testCase == TENSOR_ADD_TENSOR ||
            testCase == TENSOR_SUBTRACT_TENSOR || testCase == TENSOR_MULTIPLY_TENSOR ||
            testCase == TENSOR_DIVIDE_TENSOR) {
            Rpp8s* outputI8Second =
                static_cast<Rpp8s*>(outputSecond) + srcDescriptorPtrNDSecond->offsetInBytes;
            for (int i = 0; i < ioBufferSizeSecond; i++)
                outputI8Second[i] = static_cast<Rpp8s>(
                    std::clamp(std::round(inputF32Second[i]) - 128, -128.0f, 127.0f));
        }
    } else if (BitDepthTestMode == I16_TO_I16 || BitDepthTestMode == I16_TO_F32)  // I16 case
    {
        Rpp16s* outputI16 = reinterpret_cast<Rpp16s*>(static_cast<Rpp8u*>(output) +
                                                      srcGenericDescPtr->offsetInBytes);
        for (int i = 0; i < ioBufferSize; i++)
            outputI16[i] =
                static_cast<Rpp16s>(std::clamp(std::round(inputF32[i]), -32768.0f, 32767.0f));

        if (testCase == CONCAT || testCase == TENSOR_ADD_TENSOR ||
            testCase == TENSOR_SUBTRACT_TENSOR || testCase == TENSOR_MULTIPLY_TENSOR ||
            testCase == TENSOR_DIVIDE_TENSOR) {
            Rpp16s* outputI16Second = reinterpret_cast<Rpp16s*>(
                static_cast<Rpp8u*>(outputSecond) + srcDescriptorPtrNDSecond->offsetInBytes);
            for (int i = 0; i < ioBufferSizeSecond; i++)
                outputI16Second[i] = static_cast<Rpp16s>(
                    std::clamp(std::round(inputF32Second[i]), -32768.0f, 32767.0f));
        }
    } else if (BitDepthTestMode == U16_TO_U16 || BitDepthTestMode == U16_TO_F32)  // U16 case
    {
        Rpp16u* outputU16 = reinterpret_cast<Rpp16u*>(static_cast<Rpp8u*>(output) +
                                                      srcGenericDescPtr->offsetInBytes);
        for (int i = 0; i < ioBufferSize; i++)
            outputU16[i] = static_cast<Rpp16u>(std::clamp(std::round(inputF32[i]), 0.0f, 65535.0f));

        if (testCase == CONCAT || testCase == TENSOR_ADD_TENSOR ||
            testCase == TENSOR_SUBTRACT_TENSOR || testCase == TENSOR_MULTIPLY_TENSOR ||
            testCase == TENSOR_DIVIDE_TENSOR) {
            Rpp16u* outputU16Second = reinterpret_cast<Rpp16u*>(
                static_cast<Rpp8u*>(outputSecond) + srcDescriptorPtrNDSecond->offsetInBytes);
            for (int i = 0; i < ioBufferSizeSecond; i++)
                outputU16Second[i] =
                    static_cast<Rpp16u>(std::clamp(std::round(inputF32Second[i]), 0.0f, 65535.0f));
        }
    } else if (BitDepthTestMode == I32_TO_I32 || BitDepthTestMode == I32_TO_F32)  // I32 case
    {
        Rpp32s* outputI32 = reinterpret_cast<Rpp32s*>(static_cast<Rpp8u*>(output) +
                                                      srcGenericDescPtr->offsetInBytes);
        for (int i = 0; i < ioBufferSize; i++)
            outputI32[i] = static_cast<Rpp32s>(
                std::clamp(std::round(inputF32[i]), -2147483648.0f, 2147483647.0f));

        if (testCase == CONCAT || testCase == TENSOR_ADD_TENSOR ||
            testCase == TENSOR_SUBTRACT_TENSOR || testCase == TENSOR_MULTIPLY_TENSOR ||
            testCase == TENSOR_DIVIDE_TENSOR) {
            Rpp32s* outputI32Second = reinterpret_cast<Rpp32s*>(
                static_cast<Rpp8u*>(outputSecond) + srcDescriptorPtrNDSecond->offsetInBytes);
            for (int i = 0; i < ioBufferSizeSecond; i++)
                outputI32Second[i] = static_cast<Rpp32s>(
                    std::clamp(std::round(inputF32Second[i]), -2147483648.0f, 2147483647.0f));
        }
    } else if (BitDepthTestMode == U32_TO_U32 || BitDepthTestMode == U32_TO_F32)  // U32 case
    {
        Rpp32u* outputU32 = reinterpret_cast<Rpp32u*>(static_cast<Rpp8u*>(output) +
                                                      srcGenericDescPtr->offsetInBytes);
        for (int i = 0; i < ioBufferSize; i++)
            outputU32[i] =
                static_cast<Rpp32u>(std::clamp(std::round(inputF32[i]), 0.0f, 4294967295.0f));

        if (testCase == CONCAT || testCase == TENSOR_ADD_TENSOR ||
            testCase == TENSOR_SUBTRACT_TENSOR || testCase == TENSOR_MULTIPLY_TENSOR ||
            testCase == TENSOR_DIVIDE_TENSOR) {
            Rpp32u* outputU32Second = reinterpret_cast<Rpp32u*>(
                static_cast<Rpp8u*>(outputSecond) + srcDescriptorPtrNDSecond->offsetInBytes);
            for (int i = 0; i < ioBufferSizeSecond; i++)
                outputU32Second[i] = static_cast<Rpp32u>(
                    std::clamp(std::round(inputF32Second[i]), 0.0f, 4294967295.0f));
        }
    }
}

// Reconvert other bit depths to F32
inline void convert_output_bitdepth_to_f32(void* output, Rpp32f* outputf32, int BitDepthTestMode,
                                           Rpp64u oBufferSize, Rpp64u outputBufferSize,
                                           RpptGenericDescPtr dstDescPtr) {
    if (BitDepthTestMode == F32_TO_F32 ||
        BitDepthTestMode == U8_TO_F32)  // Already F32, direct copy
    {
        memcpy(outputf32, output, outputBufferSize);
    } else if (BitDepthTestMode == U8_TO_U8)  // U8 to F32
    {
        Rpp8u* outputTemp = static_cast<Rpp8u*>(output) + dstDescPtr->offsetInBytes;
        Rpp32f* outputf32Temp = outputf32 + dstDescPtr->offsetInBytes;
        for (int i = 0; i < oBufferSize; i++) {
            *outputf32Temp = static_cast<Rpp32f>(*outputTemp);
            outputTemp++;
            outputf32Temp++;
        }
    } else if (BitDepthTestMode == F16_TO_F16 || BitDepthTestMode == U8_TO_F16)  // F16 to F32
    {
        Rpp16f* outputf16Temp =
            reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(output) + dstDescPtr->offsetInBytes);
        Rpp32f* outputf32Temp = outputf32 + dstDescPtr->offsetInBytes;
        for (int i = 0; i < oBufferSize; i++) {
            *outputf32Temp = static_cast<Rpp32f>(*outputf16Temp);
            outputf16Temp++;
            outputf32Temp++;
        }
    } else if (BitDepthTestMode == I8_TO_I8 || BitDepthTestMode == U8_TO_I8)  // I8 to F32
    {
        Rpp8s* outputi8Temp = static_cast<Rpp8s*>(output) + dstDescPtr->offsetInBytes;
        Rpp32f* outputf32Temp = outputf32 + dstDescPtr->offsetInBytes;
        for (int i = 0; i < oBufferSize; i++) {
            *outputf32Temp = static_cast<Rpp32f>(*outputi8Temp);
            outputi8Temp++;
            outputf32Temp++;
        }
    }
}

// Compares output with reference outputs and validates QA
void compare_output(void* output, Rpp32u nDim, Rpp32u batchSize, Rpp32u BitDepthTestMode,
                    Rpp32u bufferLength, std::string dst, std::string funcName,
                    std::string testCase, int additionalParam, std::string scriptPath,
                    int broadCastFlag, std::string backend, bool isMeanStd = false) {
    // Allocate and read reference data based on BitDepthTestMode
    RpptDataType dataType;
    switch (BitDepthTestMode) {
        case U8_TO_U8:
            dataType = RpptDataType::U8;
            break;
        case F16_TO_F16:
            dataType = RpptDataType::F16;
            break;
        case F32_TO_F32:
            dataType = RpptDataType::F32;
            break;
        case U8_TO_F32:
            dataType = RpptDataType::F32;
            break;
        case I8_TO_I8:
            dataType = RpptDataType::I8;
            break;
        case I8_TO_F32:
            dataType = RpptDataType::F32;
            break;
        case I16_TO_F32:
            dataType = RpptDataType::F32;
            break;
        default:
            std::cerr << "ERROR: Invalid bitDepth specified!" << std::endl;
            return;
    }
    Rpp32u goldenOutputLength;
    int refBroadCastFlag = broadCastFlag;
    // For broadcast arithmetic ops (tensor_add/subtract/multiply/divide) in 2D/3D/4D,
    // always read from a single combined reference bin (broadCastFlag = 3)
    if (((testCase == "tensor_add_tensor") || (testCase == "tensor_subtract_tensor") ||
         (testCase == "tensor_multiply_tensor") || (testCase == "tensor_divide_tensor")) &&
        (nDim >= 2) && (nDim <= 4))
        refBroadCastFlag = 3;
    if (testCase == "log")
        goldenOutputLength = get_bin_size(nDim, 1, scriptPath, testCase, F32_TO_F32);
    else if (testCase == "tensor_and_tensor" || testCase == "tensor_or_tensor" ||
             testCase == "tensor_xor_tensor")
        goldenOutputLength =
            get_bin_size(nDim, 1, scriptPath, testCase, BitDepthTestMode, broadCastFlag);
    else
        goldenOutputLength =
            get_bin_size(nDim, 1, scriptPath, testCase, BitDepthTestMode, refBroadCastFlag);
    void* refOutput = calloc(goldenOutputLength, get_size_of_data_type(dataType));
    read_data(refOutput, nDim, 1, scriptPath, testCase, BitDepthTestMode, refBroadCastFlag);
    int subVariantStride = 0;
    if (testCase == "normalize") {
        int meanStdDevOutputStride = 0, axisMaskStride = 0;
        if (isMeanStd) meanStdDevOutputStride = goldenOutputLength / (2 * sizeof(Rpp32f));
        axisMaskStride = (additionalParam - 1) * bufferLength;
        subVariantStride = meanStdDevOutputStride + axisMaskStride;
    } else if (testCase == "transpose") {
        subVariantStride = (additionalParam - 1) * bufferLength;
    } else if (testCase == "concat") {
        subVariantStride = additionalParam * bufferLength;
    } else if (((testCase == "tensor_add_tensor") || (testCase == "tensor_subtract_tensor") ||
                (testCase == "tensor_multiply_tensor") || (testCase == "tensor_and_tensor") ||
                (testCase == "tensor_or_tensor") || (testCase == "tensor_xor_tensor") ||
                (testCase == "tensor_divide_tensor")) &&
               (nDim >= 2) && (nDim <= 4)) {
        // 3 broadcast sub-variants are packed sequentially: no-broadcast, broadcast_input2,
        // broadcast_input1
        subVariantStride = broadCastFlag * bufferLength;
    }

    // Get cutoff value from miscCutOff map based on testCase and backend
    double cutoff = 1e-6;
    auto mapIterator = miscCutOff.find(testCase);
    if (mapIterator != miscCutOff.end()) {
        const auto& cutoffVector = mapIterator->second;
        cutoff = (backend == "HOST") ? cutoffVector[0] : cutoffVector[1];
    }

    int sampleLength = bufferLength / batchSize;
    int fileMatch = 0;
    for (int i = 0; i < batchSize; i++) {
        int cnt = 0;
        int sampleOffset = i * sampleLength + subVariantStride;

        if (testCase == "log" && BitDepthTestMode == U8_TO_F32) {
            Rpp32f* ref = static_cast<Rpp32f*>(refOutput) + sampleOffset;
            Rpp32f* out = static_cast<Rpp32f*>(output) + i * sampleLength;
            for (int j = 0; j < sampleLength; j++) {
                if ((out[j] < 0 && ref[j] < 0) || (std::abs(out[j] - ref[j]) < cutoff)) cnt++;
            }
        } else if (testCase == "tensor_divide_tensor" && BitDepthTestMode == U8_TO_F32) {
            Rpp32f* ref = static_cast<Rpp32f*>(refOutput) + sampleOffset;
            Rpp32f* out = static_cast<Rpp32f*>(output) + i * sampleLength;
            for (int j = 0; j < sampleLength; j++) {
                if ((std::abs(out[j] - ref[j]) < cutoff) || (std::isinf(ref[j])) ||
                    (std::isnan(ref[j])))
                    cnt++;
            }
        } else if (BitDepthTestMode == F32_TO_F32 || BitDepthTestMode == I16_TO_F32 ||
                   BitDepthTestMode == U8_TO_F32)  // F32 || I16_F32 || U8_F32
        {
            Rpp32f* ref = static_cast<Rpp32f*>(refOutput) + sampleOffset;
            Rpp32f* out = static_cast<Rpp32f*>(output) + i * sampleLength;
            for (int j = 0; j < sampleLength; j++) {
                if (std::abs(out[j] - ref[j]) < cutoff) cnt++;
            }
        } else if (BitDepthTestMode == U8_TO_U8)  // U8
        {
            Rpp8u* ref = static_cast<Rpp8u*>(refOutput) + sampleOffset;
            Rpp8u* out = static_cast<Rpp8u*>(output) + i * sampleLength;
            for (int j = 0; j < sampleLength; j++) {
                if (out[j] - ref[j] == 0) cnt++;
            }
        }

        if (cnt == sampleLength) fileMatch++;
    }

    std::string status = funcName + ": ";
    std::cout << std::endl << "Results for Test case: " << funcName << std::endl;
    if (fileMatch == batchSize) {
        std::cout << "\nPASSED!" << std::endl;
        status += "PASSED";
    } else {
        std::cout << "\nFAILED! " << fileMatch << "/" << batchSize
                  << " outputs are matching with reference outputs" << std::endl;
        status += "FAILED";
    }

    free(refOutput);

    // Append the QA results to file
    std::string qaResultsPath = dst + "/QA_results.txt";
    std::ofstream qaResults(qaResultsPath, ios_base::app);
    if (qaResults.is_open()) {
        qaResults << status << std::endl;
        qaResults.close();
    }
}
