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
#include "rpp_test_suite_common.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>
#include <fstream>
#include <turbojpeg.h>
#include <random>
#include <map>
#include <unordered_set>
#include <iomanip>

using namespace cv;
using namespace std;

#define CUTOFF 1
#define MAX_IMAGE_DUMP 20
#define MAX_BATCH_SIZE 512
#define GOLDEN_OUTPUT_MAX_HEIGHT 150    // Golden outputs are generated with MAX_HEIGHT set to 150. Changing this constant will result in QA test failures
#define GOLDEN_OUTPUT_MAX_WIDTH 150     // Golden outputs are generated with MAX_WIDTH set to 150. Changing this constant will result in QA test failures
#define LENS_CORRECTION_GOLDEN_OUTPUT_MAX_HEIGHT 480    // Lens correction golden outputs are generated with MAX_HEIGHT set to 480. Changing this constant will result in QA test failures
#define LENS_CORRECTION_GOLDEN_OUTPUT_MAX_WIDTH 640     // Lens correction golden outputs are generated with MAX_WIDTH set to 640. Changing this constant will result in QA test failures

#define CHECK_RETURN_STATUS(x) do { \
    int retval = (x); \
    if (retval != 0) { \
        fprintf(stderr, "Runtime error: %s returned %d at %s:%d", #x, retval, __FILE__, __LINE__); \
        exit(-1); \
    } \
} while (0)

std::map<int, string> augmentationMap =
{
    {0, "brightness"},
    {1, "gamma_correction"},
    {2, "blend"},
    {4, "contrast"},
    {5, "pixelate"},
    {6, "jitter"},
    {8, "noise"},
    {10, "fog"},
    {11, "rain"},
    {13, "exposure"},
    {15, "threshold"},
    {20, "flip"},
    {21, "resize"},
    {23, "rotate"},
    {24, "warp_afffine"},
    {26, "lens_correction"},
    {28, "warp_perspective"},
    {29, "water"},
    {30, "non_linear_blend"},
    {31, "color_cast"},
    {32, "erase"},
    {33, "crop_and_patch"},
    {34, "lut"},
    {35, "glitch"},
    {36, "color_twist"},
    {37, "crop"},
    {38, "crop_mirror_normalize"},
    {39, "resize_crop_mirror"},
    {45, "color_temperature"},
    {46, "vignette"},
    {49, "box_filter"},
    {54, "gaussian_filter"},
    {61, "magnitude"},
    {63, "phase"},
    {65, "bitwise_and"},
    {66, "bitwise_not"},
    {67, "bitwise_xor"},
    {68, "bitwise_or"},
    {70, "copy"},
    {79, "remap"},
    {80, "resize_mirror_normalize"},
    {81, "color_jitter"},
    {82, "ricap"},
    {83, "gridmask"},
    {84, "spatter"},
    {85, "swap_channels"},
    {86, "color_to_greyscale"},
    {87, "tensor_sum"},
    {88, "tensor_min"},
    {89, "tensor_max"},
    {90, "tensor_mean"},
    {91, "tensor_stddev"},
    {92, "slice"}
};

enum Augmentation {
    BRIGHTNESS = 0,
    GAMMA_CORRECTION = 1,
    BLEND = 2,
    CONTRAST = 4,
    PIXELATE = 5,
    JITTER = 6,
    NOISE = 8,
    FOG = 10,
    RAIN = 11,
    EXPOSURE = 13,
    THRESHOLD = 15,
    FLIP = 20,
    RESIZE = 21,
    ROTATE = 23,
    WARP_AFFINE = 24,
    LENS_CORRECTION = 26,
    WARP_PERSPECTIVE = 28,
    WATER = 29,
    NON_LINEAR_BLEND = 30,
    COLOR_CAST = 31,
    ERASE = 32,
    CROP_AND_PATCH = 33,
    LOOK_UP_TABLE = 34,
    GLITCH = 35,
    COLOR_TWIST = 36,
    CROP = 37,
    CROP_MIRROR_NORMALIZE = 38,
    RESIZE_CROP_MIRROR = 39,
    COLOR_TEMPERATURE = 45,
    VIGNETTE = 46,
    BOX_FILTER = 49,
    GAUSSIAN_FILTER = 54,
    MAGNITUDE = 61,
    PHASE = 63,
    BITWISE_AND = 65,
    BITWISE_NOT = 66,
    BITWISE_XOR = 67,
    BITWISE_OR = 68,
    COPY = 70,
    REMAP = 79,
    RESIZE_MIRROR_NORMALIZE = 80,
    COLOR_JITTER = 81,
    RICAP = 82,
    GRIDMASK = 83,
    SPATTER = 84,
    SWAP_CHANNELS = 85,
    COLOR_TO_GREYSCALE = 86,
    TENSOR_SUM = 87,
    TENSOR_MIN = 88,
    TENSOR_MAX = 89,
    TENSOR_MEAN = 90,
    TENSOR_STDDEV = 91,
    SLICE = 92
};

const unordered_set<int> additionalParamCases = {NOISE, RESIZE, ROTATE, WARP_AFFINE, WARP_PERSPECTIVE, BOX_FILTER, GAUSSIAN_FILTER, REMAP};
const unordered_set<int> kernelSizeCases = {BOX_FILTER, GAUSSIAN_FILTER};
const unordered_set<int> dualInputCases = {BLEND, NON_LINEAR_BLEND, CROP_AND_PATCH, MAGNITUDE, PHASE, BITWISE_AND, BITWISE_XOR, BITWISE_OR};
const unordered_set<int> randomOutputCases = {JITTER, NOISE, FOG, RAIN, SPATTER};
const unordered_set<int> nonQACases = {WARP_AFFINE, WARP_PERSPECTIVE, GAUSSIAN_FILTER};
const unordered_set<int> interpolationTypeCases = {RESIZE, ROTATE, WARP_AFFINE, WARP_PERSPECTIVE, REMAP};
const unordered_set<int> reductionTypeCases = {TENSOR_SUM, TENSOR_MIN, TENSOR_MAX, TENSOR_MEAN, TENSOR_STDDEV};
const unordered_set<int> noiseTypeCases = {NOISE};
const unordered_set<int> pln1OutTypeCases = {COLOR_TO_GREYSCALE};

// Golden outputs for Tensor min Kernel
std::map<int, std::vector<Rpp8u>> TensorMinReferenceOutputs =
{
    {1, {1, 1, 7}},
    {3, {0, 0, 0, 0, 2, 0, 0, 0, 7, 9, 0, 0}}
};

// Golden outputs for Tensor max Kernel
std::map<int, std::vector<Rpp8u>> TensorMaxReferenceOutputs =
{
    {1, {239, 245, 255}},
    {3, {255, 240, 236, 255, 255, 242, 241, 255, 253, 255, 255, 255}}
};

// Golden outputs for Tensor sum Kernel
std::map<int, std::vector<uint64_t>> TensorSumReferenceOutputs =
{
    {1, {334225, 813471, 2631125}},
    {3, {348380, 340992, 262616, 951988, 1056552, 749506, 507441, 2313499, 2170646, 2732368, 3320699, 8223713}}
};

// Golden outputs for Tensor mean Kernel
std::map<int, std::vector<float>> TensorMeanReferenceOutputs =
{
    {1, {133.690, 81.347, 116.939}},
    {3, {139.352, 136.397, 105.046, 126.932, 105.655, 74.951, 50.744, 77.117, 96.473, 121.439, 147.587, 121.833}}
};

// Golden outputs for Tensor stddev Kernel
std::map<int, std::vector<float>> TensorStddevReferenceOutputs =
{
    {1, {49.583, 54.623, 47.649}},
    {3, {57.416, 47.901, 53.235, 55.220, 68.471, 55.735, 46.668, 61.880, 47.462, 49.039, 67.269, 59.130}}
};

template <typename T>
inline T validate_pixel_range(T pixel)
{
    pixel = (pixel < static_cast<Rpp32f>(0)) ? (static_cast<Rpp32f>(0)) : ((pixel < static_cast<Rpp32f>(255)) ? pixel : (static_cast<Rpp32f>(255)));
    return pixel;
}

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

// returns the interpolation type used for image resizing or scaling operations.
inline std::string get_interpolation_type(unsigned int val, RpptInterpolationType &interpolationType)
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

// returns the noise type applied to an image
inline std::string get_noise_type(unsigned int val)
{
    switch(val)
    {
        case 0: return "SaltAndPepper";
        case 1: return "Gaussian";
        case 2: return "Shot";
        default:return "SaltAndPepper";
    }
}

// returns number of input channels according to layout type
inline int set_input_channels(int layoutType)
{
    if(layoutType == 0 || layoutType == 1)
        return 3;
    else
        return 1;
}

//returns function type
inline string set_function_type(int layoutType, int pln1OutTypeCase, int outputFormatToggle, string backend)
{
    string funcType;
    if(layoutType == 0)
    {
        funcType = "Tensor_" + backend + "_PKD3";
        if (pln1OutTypeCase)
            funcType += "_toPLN1";
        else
        {
            if (outputFormatToggle)
                funcType += "_toPLN3";
            else
                funcType += "_toPKD3";
        }
    }
    else if (layoutType == 1)
    {
        funcType = "Tensor_" + backend + "_PLN3";
        if (pln1OutTypeCase)
            funcType += "_toPLN1";
        else
        {
            if (outputFormatToggle)
                funcType += "_toPKD3";
            else
                funcType += "_toPLN3";
        }
    }
    else
    {
       funcType = "Tensor_" + backend + "_PLN1";
       funcType += "_toPLN1";
    }

    return funcType;
}

// sets descriptor data types of src/dst
inline void set_descriptor_data_type(int ip_bitDepth, string &funcName, RpptDescPtr srcDescPtr, RpptDescPtr dstDescPtr)
{
    if (ip_bitDepth == 0)
    {
        funcName += "_u8_";
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::U8;
    }
    else if (ip_bitDepth == 1)
    {
        funcName += "_f16_";
        srcDescPtr->dataType = RpptDataType::F16;
        dstDescPtr->dataType = RpptDataType::F16;
    }
    else if (ip_bitDepth == 2)
    {
        funcName += "_f32_";
        srcDescPtr->dataType = RpptDataType::F32;
        dstDescPtr->dataType = RpptDataType::F32;
    }
    else if (ip_bitDepth == 3)
    {
        funcName += "_u8_f16_";
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::F16;
    }
    else if (ip_bitDepth == 4)
    {
        funcName += "_u8_f32_";
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::F32;
    }
    else if (ip_bitDepth == 5)
    {
        funcName += "_i8_";
        srcDescPtr->dataType = RpptDataType::I8;
        dstDescPtr->dataType = RpptDataType::I8;
    }
    else if (ip_bitDepth == 6)
    {
        funcName += "_u8_i8_";
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::I8;
    }
}

// sets descriptor layout of src/dst
inline void set_descriptor_layout( RpptDescPtr srcDescPtr, RpptDescPtr dstDescPtr, int layoutType, bool pln1OutTypeCase, int outputFormatToggle)
{
    if(layoutType == 0)
    {
        srcDescPtr->layout = RpptLayout::NHWC;
        // Set src/dst layouts in tensor descriptors
        if (pln1OutTypeCase)
            dstDescPtr->layout = RpptLayout::NCHW;
        else
        {
            if (outputFormatToggle == 0)
                dstDescPtr->layout = RpptLayout::NHWC;
            else if (outputFormatToggle == 1)
                dstDescPtr->layout = RpptLayout::NCHW;
        }
    }
    else if(layoutType == 1)
    {
        srcDescPtr->layout = RpptLayout::NCHW;
        // Set src/dst layouts in tensor descriptors
        if (pln1OutTypeCase)
            dstDescPtr->layout = RpptLayout::NCHW;
        else
        {
            if (outputFormatToggle == 0)
                dstDescPtr->layout = RpptLayout::NCHW;
            else if (outputFormatToggle == 1)
                dstDescPtr->layout = RpptLayout::NHWC;
        }
    }
    else
    {
        // Set src/dst layouts in tensor descriptors
        srcDescPtr->layout = RpptLayout::NCHW;
        dstDescPtr->layout = RpptLayout::NCHW;
    }
}

// sets values of maxHeight and maxWidth
inline void set_max_dimensions(vector<string>imagePaths, int& maxHeight, int& maxWidth, int& imagesMixed)
{
    tjhandle tjInstance = tjInitDecompress();
    for (const std::string& imagePath : imagePaths)
    {
        FILE* jpegFile = fopen(imagePath.c_str(), "rb");
        if (!jpegFile) {
            std::cerr << "Error opening file: " << imagePath << std::endl;
            continue;
        }

        fseek(jpegFile, 0, SEEK_END);
        long fileSize = ftell(jpegFile);
        fseek(jpegFile, 0, SEEK_SET);

        std::vector<unsigned char> jpegBuffer(fileSize);
        fread(jpegBuffer.data(), 1, fileSize, jpegFile);
        fclose(jpegFile);

        int jpegSubsamp;
        int width, height;
        if (tjDecompressHeader2(tjInstance, jpegBuffer.data(), jpegBuffer.size(), &width, &height, &jpegSubsamp) == -1) {
            std::cerr << "Error decompressing file: " << imagePath << std::endl;
            continue;
        }

        if((maxWidth && maxWidth != width) || (maxHeight && maxHeight != height))
            imagesMixed = 1;

        maxWidth = max(maxWidth, width);
        maxHeight = max(maxHeight, height);
    }
    tjDestroy(tjInstance);
}

// sets roi xywh values and dstImg sizes
inline void  set_src_and_dst_roi(vector<string>::const_iterator imagePathsStart, vector<string>::const_iterator imagePathsEnd, RpptROI *roiTensorPtrSrc, RpptROI *roiTensorPtrDst, RpptImagePatchPtr dstImgSizes)
{
    tjhandle tjInstance = tjInitDecompress();
    int i = 0;
    for (auto imagePathIter = imagePathsStart; imagePathIter != imagePathsEnd; ++imagePathIter, i++)
    {
        const string& imagePath = *imagePathIter;
        FILE* jpegFile = fopen(imagePath.c_str(), "rb");
        if (!jpegFile) {
            std::cerr << "Error opening file: " << imagePath << std::endl;
            continue;
        }

        fseek(jpegFile, 0, SEEK_END);
        long fileSize = ftell(jpegFile);
        fseek(jpegFile, 0, SEEK_SET);

        std::vector<unsigned char> jpegBuffer(fileSize);
        fread(jpegBuffer.data(), 1, fileSize, jpegFile);
        fclose(jpegFile);

        int jpegSubsamp;
        int width, height;
        if (tjDecompressHeader2(tjInstance, jpegBuffer.data(), jpegBuffer.size(), &width, &height, &jpegSubsamp) == -1) {
            std::cerr << "Error decompressing file: " << imagePath << std::endl;
            continue;
        }

        roiTensorPtrSrc[i].xywhROI = {0, 0, width, height};
        roiTensorPtrDst[i].xywhROI = {0, 0, width, height};
        dstImgSizes[i].width = roiTensorPtrDst[i].xywhROI.roiWidth;
        dstImgSizes[i].height = roiTensorPtrDst[i].xywhROI.roiHeight;
    }
    tjDestroy(tjInstance);
}

// sets generic descriptor dimensions and strides of src/dst
inline void set_generic_descriptor(RpptGenericDescPtr descriptorPtr3D, int noOfImages, int maxX, int maxY, int maxZ, int numChannels, int offsetInBytes, int layoutType)
{
    descriptorPtr3D->numDims = 5;
    descriptorPtr3D->offsetInBytes = offsetInBytes;
    descriptorPtr3D->dataType = RpptDataType::F32;

    if (layoutType == 0)
    {
        descriptorPtr3D->layout = RpptLayout::NCDHW;
        descriptorPtr3D->dims[0] = noOfImages;
        descriptorPtr3D->dims[1] = numChannels;
        descriptorPtr3D->dims[2] = maxZ;
        descriptorPtr3D->dims[3] = maxY;
        descriptorPtr3D->dims[4] = maxX;
    }
    else if (layoutType == 1)
    {
        descriptorPtr3D->layout = RpptLayout::NDHWC;
        descriptorPtr3D->dims[0] = noOfImages;
        descriptorPtr3D->dims[1] = maxZ;
        descriptorPtr3D->dims[2] = maxY;
        descriptorPtr3D->dims[3] = maxX;
        descriptorPtr3D->dims[4] = numChannels;
    }

    descriptorPtr3D->strides[0] = descriptorPtr3D->dims[1] * descriptorPtr3D->dims[2] * descriptorPtr3D->dims[3] * descriptorPtr3D->dims[4];
    descriptorPtr3D->strides[1] = descriptorPtr3D->dims[2] * descriptorPtr3D->dims[3] * descriptorPtr3D->dims[4];
    descriptorPtr3D->strides[2] = descriptorPtr3D->dims[3] * descriptorPtr3D->dims[4];
    descriptorPtr3D->strides[3] = descriptorPtr3D->dims[4];
    descriptorPtr3D->strides[4] = 1;
}

// sets generic descriptor dimensions and strides of src/dst for slice functionality
inline void set_generic_descriptor_slice(RpptDescPtr srcDescPtr, RpptGenericDescPtr descriptorPtr3D, int batchSize)
{
    descriptorPtr3D->offsetInBytes = 0;
    descriptorPtr3D->dataType = srcDescPtr->dataType;
    descriptorPtr3D->layout = srcDescPtr->layout;
    if(srcDescPtr->c == 3)
    {
        descriptorPtr3D->numDims = 4;
        descriptorPtr3D->dims[0] = batchSize;
        if (srcDescPtr->layout == RpptLayout::NHWC)
        {
            descriptorPtr3D->dims[1] = srcDescPtr->h;
            descriptorPtr3D->dims[2] = srcDescPtr->w;
            descriptorPtr3D->dims[3] = srcDescPtr->c;
        }
        else
        {
            descriptorPtr3D->dims[1] = srcDescPtr->c;
            descriptorPtr3D->dims[2] = srcDescPtr->h;
            descriptorPtr3D->dims[3] = srcDescPtr->w;
        }
        descriptorPtr3D->strides[0] = descriptorPtr3D->dims[1] * descriptorPtr3D->dims[2] * descriptorPtr3D->dims[3];
        descriptorPtr3D->strides[1] = descriptorPtr3D->dims[2] * descriptorPtr3D->dims[3];
        descriptorPtr3D->strides[2] = descriptorPtr3D->dims[3];
    }
    else
    {
        descriptorPtr3D->numDims = 3;
        descriptorPtr3D->dims[0] = batchSize;
        descriptorPtr3D->dims[1] = srcDescPtr->h;
        descriptorPtr3D->dims[2] = srcDescPtr->w;
        descriptorPtr3D->strides[0] = descriptorPtr3D->dims[1] * descriptorPtr3D->dims[2];
        descriptorPtr3D->strides[1] = descriptorPtr3D->dims[2];
    }
}

// sets descriptor dimensions and strides of src/dst
inline void set_descriptor_dims_and_strides(RpptDescPtr descPtr, int noOfImages, int maxHeight, int maxWidth, int numChannels, int offsetInBytes, int additionalStride = 0)
{
    descPtr->numDims = 4;
    descPtr->offsetInBytes = offsetInBytes;
    descPtr->n = noOfImages;
    descPtr->h = maxHeight;
    descPtr->w = maxWidth;
    descPtr->c = numChannels;

    // Optionally set w stride as a multiple of 8 for src/dst
    descPtr->w = (descPtr->w / 8) * 8 + 8 + additionalStride;
    // set strides
    if (descPtr->layout == RpptLayout::NHWC)
    {
        descPtr->strides.nStride = descPtr->c * descPtr->w * descPtr->h;
        descPtr->strides.hStride = descPtr->c * descPtr->w;
        descPtr->strides.wStride = descPtr->c;
        descPtr->strides.cStride = 1;
    }
    else if(descPtr->layout == RpptLayout::NCHW)
    {
        descPtr->strides.nStride = descPtr->c * descPtr->w * descPtr->h;
        descPtr->strides.cStride = descPtr->w * descPtr->h;
        descPtr->strides.hStride = descPtr->w;
        descPtr->strides.wStride = 1;
    }
}

inline void set_roi_values(RpptROI *roi, RpptROI *roiTensorPtrSrc, RpptRoiType roiType, int batchSize)
{
    if(roiType == RpptRoiType::XYWH)
        for (int i = 0; i < batchSize; i++)
            roiTensorPtrSrc[i].xywhROI = roi->xywhROI;
    else if(roiType == RpptRoiType::LTRB)
        for (int i = 0; i < batchSize; i++)
            roiTensorPtrSrc[i].ltrbROI = roi->ltrbROI;
}

inline void convert_roi(RpptROI *roiTensorPtrSrc, RpptRoiType roiType, int batchSize)
{
    if(roiType == RpptRoiType::LTRB)
    {
        for (int i = 0; i < batchSize; i++)
        {
            RpptRoiXywh roi = roiTensorPtrSrc[i].xywhROI;
            roiTensorPtrSrc[i].ltrbROI = {roi.xy.x, roi.xy.y, roi.roiWidth - roi.xy.x, roi.roiHeight - roi.xy.y};
        }
    }
    else
    {
        for (int i = 0; i < batchSize; i++)
        {
            RpptRoiLtrb roi = roiTensorPtrSrc[i].ltrbROI;
            roiTensorPtrSrc[i].xywhROI = {roi.lt.x, roi.lt.y, roi.rb.x - roi.lt.x + 1, roi.rb.y - roi.lt.y + 1};
        }
    }
}

// Convert inputs to correponding bit depth specified by user
inline void convert_input_bitdepth(void *input, void *input_second, Rpp8u *inputu8, Rpp8u *inputu8Second, int inputBitDepth, Rpp64u ioBufferSize, Rpp64u inputBufferSize, RpptDescPtr srcDescPtr, bool dualInputCase, Rpp32f conversionFactor)
{
    if (inputBitDepth == 0 || inputBitDepth == 3 || inputBitDepth == 4)
    {
        memcpy(input, inputu8, inputBufferSize);
        if(dualInputCase)
            memcpy(input_second, inputu8Second, inputBufferSize);
    }
    else if (inputBitDepth == 1)
    {
        Rpp8u *inputTemp, *inputSecondTemp;
        Rpp16f *inputf16Temp, *inputf16SecondTemp;
        inputTemp = inputu8 + srcDescPtr->offsetInBytes;
        inputf16Temp = reinterpret_cast<Rpp16f *>(static_cast<Rpp8u *>(input) + srcDescPtr->offsetInBytes);
        for (int i = 0; i < ioBufferSize; i++)
            *inputf16Temp++ = static_cast<Rpp16f>((static_cast<float>(*inputTemp++)) * conversionFactor);

        if(dualInputCase)
        {
            inputSecondTemp = inputu8Second + srcDescPtr->offsetInBytes;
            inputf16SecondTemp = reinterpret_cast<Rpp16f *>(static_cast<Rpp8u *>(input_second) + srcDescPtr->offsetInBytes);
            for (int i = 0; i < ioBufferSize; i++)
                *inputf16SecondTemp++ = static_cast<Rpp16f>((static_cast<float>(*inputSecondTemp++)) * conversionFactor);
        }
    }
    else if (inputBitDepth == 2)
    {
        Rpp8u *inputTemp, *inputSecondTemp;
        Rpp32f *inputf32Temp, *inputf32SecondTemp;
        inputTemp = inputu8 + srcDescPtr->offsetInBytes;
        inputf32Temp = reinterpret_cast<Rpp32f *>(static_cast<Rpp8u *>(input) + srcDescPtr->offsetInBytes);
        for (int i = 0; i < ioBufferSize; i++)
            *inputf32Temp++ = (static_cast<Rpp32f>(*inputTemp++)) * conversionFactor;

        if(dualInputCase)
        {
            inputSecondTemp = inputu8Second + srcDescPtr->offsetInBytes;
            inputf32SecondTemp = reinterpret_cast<Rpp32f *>(static_cast<Rpp8u *>(input_second) + srcDescPtr->offsetInBytes);
            for (int i = 0; i < ioBufferSize; i++)
                *inputf32SecondTemp++ = (static_cast<Rpp32f>(*inputSecondTemp++)) * conversionFactor;
        }
    }
    else if (inputBitDepth == 5)
    {
        Rpp8u *inputTemp, *inputSecondTemp;
        Rpp8s *inputi8Temp, *inputi8SecondTemp;

        inputTemp = inputu8 + srcDescPtr->offsetInBytes;
        inputi8Temp = static_cast<Rpp8s *>(input) + srcDescPtr->offsetInBytes;
        for (int i = 0; i < ioBufferSize; i++)
            *inputi8Temp++ = static_cast<Rpp8s>((static_cast<Rpp32s>(*inputTemp++)) - 128);

        if(dualInputCase)
        {
            inputSecondTemp = inputu8Second + srcDescPtr->offsetInBytes;
            inputi8SecondTemp = static_cast<Rpp8s *>(input_second) + srcDescPtr->offsetInBytes;
            for (int i = 0; i < ioBufferSize; i++)
                *inputi8SecondTemp++ = static_cast<Rpp8s>((static_cast<Rpp32s>(*inputSecondTemp++)) - 128);
        }
    }
}

// Reconvert other bit depths to 8u for output display purposes
inline void convert_output_bitdepth_to_u8(void *output, Rpp8u *outputu8, int inputBitDepth, Rpp64u oBufferSize, Rpp64u outputBufferSize, RpptDescPtr dstDescPtr, Rpp32f invConversionFactor)
{
    if (inputBitDepth == 0)
    {
        memcpy(outputu8, output, outputBufferSize);
    }
    else if ((inputBitDepth == 1) || (inputBitDepth == 3))
    {
        Rpp8u *outputTemp = outputu8 + dstDescPtr->offsetInBytes;
        Rpp16f *outputf16Temp = reinterpret_cast<Rpp16f *>(static_cast<Rpp8u *>(output) + dstDescPtr->offsetInBytes);
        for (int i = 0; i < oBufferSize; i++)
        {
            *outputTemp = static_cast<Rpp8u>(validate_pixel_range(static_cast<float>(*outputf16Temp) * invConversionFactor));
            outputf16Temp++;
            outputTemp++;
        }
    }
    else if ((inputBitDepth == 2) || (inputBitDepth == 4))
    {
        Rpp8u *outputTemp = outputu8 + dstDescPtr->offsetInBytes;
        Rpp32f *outputf32Temp = reinterpret_cast<Rpp32f *>(static_cast<Rpp8u *>(output) + dstDescPtr->offsetInBytes);
        for (int i = 0; i < oBufferSize; i++)
        {
            *outputTemp = static_cast<Rpp8u>(validate_pixel_range(*outputf32Temp * invConversionFactor));
            outputf32Temp++;
            outputTemp++;
        }
    }
    else if ((inputBitDepth == 5) || (inputBitDepth == 6))
    {
        Rpp8u *outputTemp = outputu8 + dstDescPtr->offsetInBytes;
        Rpp8s *outputi8Temp = static_cast<Rpp8s *>(output) + dstDescPtr->offsetInBytes;
        for (int i = 0; i < oBufferSize; i++)
        {
            *outputTemp = static_cast<Rpp8u>(validate_pixel_range((static_cast<Rpp32s>(*outputi8Temp) + 128)));
            outputi8Temp++;
            outputTemp++;
        }
    }
}

// updates dstImg sizes
inline void update_dst_sizes_with_roi(RpptROI *roiTensorPtrSrc, RpptImagePatchPtr dstImageSize, RpptRoiType roiType, int batchSize)
{
    if(roiType == RpptRoiType::XYWH)
    {
        for (int i = 0; i < batchSize; i++)
        {
            dstImageSize[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth;
            dstImageSize[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight;
        }
    }
    else if(roiType == RpptRoiType::LTRB)
    {
        for (int i = 0; i < batchSize; i++)
        {
            dstImageSize[i].width = roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
            dstImageSize[i].height = roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
        }
    }
}

// converts image data from PLN3 to PKD3
inline void convert_pln3_to_pkd3(Rpp8u *output, RpptDescPtr descPtr)
{
    unsigned long long bufferSize = ((unsigned long long)descPtr->h * (unsigned long long)descPtr->w * (unsigned long long)descPtr->c * (unsigned long long)descPtr->n) + descPtr->offsetInBytes;
    Rpp8u *outputCopy = (Rpp8u *)calloc(bufferSize, sizeof(Rpp8u));
    memcpy(outputCopy, output, bufferSize * sizeof(Rpp8u));

    Rpp8u *outputCopyTemp;
    outputCopyTemp = outputCopy + descPtr->offsetInBytes;

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(descPtr->n)
    for (int count = 0; count < descPtr->n; count++)
    {
        Rpp8u *outputCopyTempR, *outputCopyTempG, *outputCopyTempB;
        outputCopyTempR = outputCopyTemp + count * descPtr->strides.nStride;
        outputCopyTempG = outputCopyTempR + descPtr->strides.cStride;
        outputCopyTempB = outputCopyTempG + descPtr->strides.cStride;
        Rpp8u *outputTemp = output + descPtr->offsetInBytes + count * descPtr->strides.nStride;

        for (int i = 0; i < descPtr->h; i++)
        {
            for (int j = 0; j < descPtr->w; j++)
            {
                *outputTemp = *outputCopyTempR;
                outputTemp++;
                outputCopyTempR++;
                *outputTemp = *outputCopyTempG;
                outputTemp++;
                outputCopyTempG++;
                *outputTemp = *outputCopyTempB;
                outputTemp++;
                outputCopyTempB++;
            }
        }
    }

    free(outputCopy);
}

// converts image data from PKD3 to PLN3
inline void convert_pkd3_to_pln3(Rpp8u *input, RpptDescPtr descPtr)
{
    unsigned long long bufferSize = ((unsigned long long)descPtr->h * (unsigned long long)descPtr->w * (unsigned long long)descPtr->c * (unsigned long long)descPtr->n) + descPtr->offsetInBytes;
    Rpp8u *inputCopy = (Rpp8u *)calloc(bufferSize, sizeof(Rpp8u));
    memcpy(inputCopy, input, bufferSize * sizeof(Rpp8u));

    Rpp8u *inputTemp, *inputCopyTemp;
    inputTemp = input + descPtr->offsetInBytes;

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(descPtr->n)
    for (int count = 0; count < descPtr->n; count++)
    {
        Rpp8u *inputTempR, *inputTempG, *inputTempB;
        inputTempR = inputTemp + count * descPtr->strides.nStride;
        inputTempG = inputTempR + descPtr->strides.cStride;
        inputTempB = inputTempG + descPtr->strides.cStride;
        Rpp8u *inputCopyTemp = inputCopy + descPtr->offsetInBytes + count * descPtr->strides.nStride;

        for (int i = 0; i < descPtr->h; i++)
        {
            for (int j = 0; j < descPtr->w; j++)
            {
                *inputTempR = *inputCopyTemp;
                inputCopyTemp++;
                inputTempR++;
                *inputTempG = *inputCopyTemp;
                inputCopyTemp++;
                inputTempG++;
                *inputTempB = *inputCopyTemp;
                inputCopyTemp++;
                inputTempB++;
            }
        }
    }

    free(inputCopy);
}

// Read a batch of images using the OpenCV library
inline void read_image_batch_opencv(Rpp8u *input, RpptDescPtr descPtr, vector<string>::const_iterator imagesNamesStart)
{
    for(int i = 0; i < descPtr->n; i++)
    {
        Rpp8u *inputTemp = input + (i * descPtr->strides.nStride);
        string inputImagePath = *(imagesNamesStart + i);
        Mat image, imageBgr;
        if (descPtr->c == 3)
        {
            imageBgr = imread(inputImagePath, 1);
            cvtColor(imageBgr, image, COLOR_BGR2RGB);
        }
        else if (descPtr->c == 1)
            image = imread(inputImagePath, 0);

        int width = image.cols;
        int height = image.rows;
        Rpp32u elementsInRow = width * descPtr->c;
        Rpp8u *inputImage = image.data;
        for (int j = 0; j < height; j++)
        {
            memcpy(inputTemp, inputImage, elementsInRow * sizeof(Rpp8u));
            inputImage += elementsInRow;
            inputTemp += descPtr->w * descPtr->c;;
        }
    }
}

// Read a batch of images using the turboJpeg decoder
inline void read_image_batch_turbojpeg(Rpp8u *input, RpptDescPtr descPtr, vector<string>::const_iterator imagesNamesStart)
{
    tjhandle m_jpegDecompressor = tjInitDecompress();

    // Loop through the input images
    for (int i = 0; i < descPtr->n; i++)
    {
        // Read the JPEG compressed data from a file
        std::string inputImagePath = *(imagesNamesStart + i);
        FILE* fp = fopen(inputImagePath.c_str(), "rb");
        if(!fp)
            std::cerr << "\n unable to open file : "<<inputImagePath;
        fseek(fp, 0, SEEK_END);
        long jpegSize = ftell(fp);
        rewind(fp);
        unsigned char* jpegBuf = (unsigned char*)calloc(jpegSize, sizeof(Rpp8u));
        fread(jpegBuf, 1, jpegSize, fp);
        fclose(fp);

        // Decompress the JPEG data into an RGB image buffer
        int width, height, subsamp, color_space;
        if(tjDecompressHeader2(m_jpegDecompressor, jpegBuf, jpegSize, &width, &height, &color_space) != 0)
            std::cerr << "\n Jpeg image decode failed in tjDecompressHeader2";
        Rpp8u* rgbBuf;
        int elementsInRow;
        if(descPtr->c == 3)
        {
            elementsInRow = width * descPtr->c;
            rgbBuf= (Rpp8u*)calloc(width * height * 3, sizeof(Rpp8u));
            if(tjDecompress2(m_jpegDecompressor, jpegBuf, jpegSize, rgbBuf, width, width * 3, height, TJPF_RGB, TJFLAG_ACCURATEDCT) != 0)
                std::cerr << "\n Jpeg image decode failed ";
        }
        else
        {
            elementsInRow = width;
            rgbBuf= (Rpp8u*)calloc(width * height, sizeof(Rpp8u));
            if(tjDecompress2(m_jpegDecompressor, jpegBuf, jpegSize, rgbBuf, width, width, height, TJPF_GRAY, 0) != 0)
                std::cerr << "\n Jpeg image decode failed ";
        }
        // Copy the decompressed image buffer to the RPP input buffer
        Rpp8u *inputTemp = input + descPtr->offsetInBytes + (i * descPtr->strides.nStride);
        for (int j = 0; j < height; j++)
        {
            memcpy(inputTemp, rgbBuf + j * elementsInRow, elementsInRow * sizeof(Rpp8u));
            inputTemp += descPtr->w * descPtr->c;
        }
        // Clean up
        free(jpegBuf);
        free(rgbBuf);
    }

    // Clean up
    tjDestroy(m_jpegDecompressor);
}

// Write a batch of images using the OpenCV library
inline void write_image_batch_opencv(string outputFolder, Rpp8u *output, RpptDescPtr dstDescPtr, vector<string>::const_iterator imagesNamesStart, RpptImagePatch *dstImgSizes, int maxImageDump)
{
    // create output folder
    mkdir(outputFolder.c_str(), 0700);
    outputFolder += "/";
    static int cnt = 1;
    static int imageCnt = 0;

    Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
    Rpp8u *offsettedOutput = output + dstDescPtr->offsetInBytes;
    for (int j = 0; (j < dstDescPtr->n) && (imageCnt < maxImageDump) ; j++, imageCnt++)
    {
        Rpp32u height = dstImgSizes[j].height;
        Rpp32u width = dstImgSizes[j].width;
        Rpp32u elementsInRow = width * dstDescPtr->c;
        Rpp32u outputSize = height * width * dstDescPtr->c;
        Rpp8u *tempOutput = (Rpp8u *)calloc(outputSize, sizeof(Rpp8u));
        Rpp8u *tempOutputRow = tempOutput;
        Rpp8u *outputRow = offsettedOutput + j * dstDescPtr->strides.nStride;
        for (int k = 0; k < height; k++)
        {
            memcpy(tempOutputRow, outputRow, elementsInRow * sizeof(Rpp8u));
            tempOutputRow += elementsInRow;
            outputRow += elementsInRowMax;
        }
        string outputImagePath = outputFolder + *(imagesNamesStart + j);
        Mat matOutputImage, matOutputImageRgb;
        if (dstDescPtr->c == 1)
            matOutputImage = Mat(height, width, CV_8UC1, tempOutput);
        else if (dstDescPtr->c == 2)
            matOutputImage = Mat(height, width, CV_8UC2, tempOutput);
        else if (dstDescPtr->c == 3)
        {
            matOutputImageRgb = Mat(height, width, CV_8UC3, tempOutput);
            cvtColor(matOutputImageRgb, matOutputImage, COLOR_RGB2BGR);
        }

        fs::path pathObj(outputImagePath);
        if (fs::exists(pathObj))
        {
            std::string outPath = outputImagePath.substr(0, outputImagePath.find_last_of('.')) + "_" + to_string(cnt) + outputImagePath.substr(outputImagePath.find_last_of('.'));
            imwrite(outPath, matOutputImage);
            cnt++;
        }
        else
            imwrite(outputImagePath, matOutputImage);
        free(tempOutput);
    }
}

// compares the output of PKD3-PKD3 and PLN1-PLN1 variants
void compare_outputs_pkd_and_pln1(Rpp8u* output, Rpp8u* refOutput, RpptDescPtr dstDescPtr, RpptImagePatch *dstImgSizes, int refOutputHeight, int refOutputWidth, int refOutputSize, int &fileMatch)
{
    Rpp8u *rowTemp, *rowTempRef, *outVal, *outRefVal, *outputTemp, *outputTempRef;
    for(int imageCnt = 0; imageCnt < dstDescPtr->n; imageCnt++)
    {
        outputTemp = output + imageCnt * dstDescPtr->strides.nStride;
        outputTempRef = refOutput + imageCnt * refOutputSize;
        int height = dstImgSizes[imageCnt].height;
        int width = dstImgSizes[imageCnt].width * dstDescPtr->c;
        int matchedIdx = 0;
        int refOutputHstride = refOutputWidth * dstDescPtr->c;

        for(int i = 0; i < height; i++)
        {
            rowTemp = outputTemp + i * dstDescPtr->strides.hStride;
            rowTempRef = outputTempRef + i * refOutputHstride;
            for(int j = 0; j < width; j++)
            {
                outVal = rowTemp + j;
                outRefVal = rowTempRef + j;
                int diff = abs(*outVal - *outRefVal);
                if(diff <= CUTOFF)
                    matchedIdx++;
            }
        }
        if(matchedIdx == (height * width) && matchedIdx !=0)
            fileMatch++;
    }
}

// compares the output of PKD3-PKD3 and PLN1-PLN1 variants
void compare_outputs_pkd_and_pln1(Rpp32f* output, Rpp32f* refOutput, RpptDescPtr dstDescPtr, RpptImagePatch *dstImgSizes, int refOutputHeight, int refOutputWidth, int refOutputSize, int &fileMatch)
{
    Rpp32f *rowTemp, *rowTempRef, *outVal, *outRefVal, *outputTemp, *outputTempRef;
    for(int imageCnt = 0; imageCnt < dstDescPtr->n; imageCnt++)
    {
        outputTemp = output + imageCnt * dstDescPtr->strides.nStride;
        outputTempRef = refOutput + imageCnt * refOutputSize;
        int height = dstImgSizes[imageCnt].height;
        int width = dstImgSizes[imageCnt].width * dstDescPtr->c;
        int matchedIdx = 0;
        int refOutputHstride = refOutputWidth * dstDescPtr->c;
        for(int i = 0; i < height; i++)
        {
            rowTemp = outputTemp + i * dstDescPtr->strides.hStride;
            rowTempRef = outputTempRef + i * refOutputHstride;
            for(int j = 0; j < width; j++)
            {
                outVal = rowTemp + j;
                outRefVal = rowTempRef + j;
                Rpp32f diff = abs(*outVal - *outRefVal);
                if(diff <= 1e-6)
                    matchedIdx++;
            }
        }
        if(matchedIdx == (height * width) && matchedIdx !=0)
            fileMatch++;
    }
}

// compares the output of PLN3-PLN3 variants.This function compares the output buffer of pln3 format with its reference output in pkd3 format.
void compare_outputs_pln3(Rpp8u* output, Rpp8u* refOutput, RpptDescPtr dstDescPtr, RpptImagePatch *dstImgSizes, int refOutputHeight, int refOutputWidth, int refOutputSize, int &fileMatch)
{
    Rpp8u *rowTemp, *rowTempRef, *outVal, *outRefVal, *outputTemp, *outputTempRef, *outputTempChn, *outputTempRefChn;
    for(int imageCnt = 0; imageCnt < dstDescPtr->n; imageCnt++)
    {
        outputTemp = output + imageCnt * dstDescPtr->strides.nStride;
        outputTempRef = refOutput + imageCnt * refOutputSize;
        int height = dstImgSizes[imageCnt].height;
        int width = dstImgSizes[imageCnt].width;
        int matchedIdx = 0;
        int refOutputHstride = refOutputWidth * dstDescPtr->c;

        for(int c = 0; c < dstDescPtr->c; c++)
        {
            outputTempChn = outputTemp + c * dstDescPtr->strides.cStride;
            outputTempRefChn = outputTempRef + c;
            for(int i = 0; i < height; i++)
            {
                rowTemp = outputTempChn + i * dstDescPtr->strides.hStride;
                rowTempRef = outputTempRefChn + i * refOutputHstride;
                for(int j = 0; j < width; j++)
                {
                    outVal = rowTemp + j;
                    outRefVal = rowTempRef + j * 3;
                    int diff = abs(*outVal - *outRefVal);
                    if(diff <= CUTOFF)
                        matchedIdx++;
                }
            }
        }
        if(matchedIdx == (height * width * dstDescPtr->c) && matchedIdx !=0)
            fileMatch++;
    }
}

// compares the output of PLN3-PLN3 variants.This function compares the output buffer of pln3 format with its reference output in pkd3 format.
void compare_outputs_pln3(Rpp32f* output, Rpp32f* refOutput, RpptDescPtr dstDescPtr, RpptImagePatch *dstImgSizes, int refOutputHeight, int refOutputWidth, int refOutputSize, int &fileMatch)
{
    Rpp32f *rowTemp, *rowTempRef, *outVal, *outRefVal, *outputTemp, *outputTempRef, *outputTempChn, *outputTempRefChn;
    for(int imageCnt = 0; imageCnt < dstDescPtr->n; imageCnt++)
    {
        outputTemp = output + imageCnt * dstDescPtr->strides.nStride;
        outputTempRef = refOutput + imageCnt * refOutputSize;
        int height = dstImgSizes[imageCnt].height;
        int width = dstImgSizes[imageCnt].width;
        int matchedIdx = 0;
        int refOutputHstride = refOutputWidth * dstDescPtr->c;

        for(int c = 0; c < dstDescPtr->c; c++)
        {
            outputTempChn = outputTemp + c * dstDescPtr->strides.cStride;
            outputTempRefChn = outputTempRef + c;
            for(int i = 0; i < height; i++)
            {
                rowTemp = outputTempChn + i * dstDescPtr->strides.hStride;
                rowTempRef = outputTempRefChn + i * refOutputHstride;
                for(int j = 0; j < width; j++)
                {
                    outVal = rowTemp + j;
                    outRefVal = rowTempRef + j * 3;
                    Rpp32f diff = abs(*outVal - *outRefVal);
                    if(diff <= 1e-6)
                        matchedIdx++;
                }
            }
        }
        if(matchedIdx == (height * width * dstDescPtr->c) && matchedIdx !=0)
            fileMatch++;
    }
}

inline void compare_output(void* output, string funcName, RpptDescPtr srcDescPtr, RpptDescPtr dstDescPtr, RpptImagePatch *dstImgSizes, int noOfImages, string interpolationTypeName, string noiseTypeName, int additionalParam, int testCase, string dst, string scriptPath)
{
    string func = funcName;
    string refFile = "";
    int refOutputWidth, refOutputHeight;
    if(testCase == LENS_CORRECTION)
    {
        refOutputWidth = ((LENS_CORRECTION_GOLDEN_OUTPUT_MAX_WIDTH / 8) * 8) + 8;    // obtain next multiple of 8 after GOLDEN_OUTPUT_MAX_WIDTH
        refOutputHeight = LENS_CORRECTION_GOLDEN_OUTPUT_MAX_HEIGHT;
    }
    else
    {
        refOutputWidth = ((GOLDEN_OUTPUT_MAX_WIDTH / 8) * 8) + 8;    // obtain next multiple of 8 after GOLDEN_OUTPUT_MAX_WIDTH
        refOutputHeight = GOLDEN_OUTPUT_MAX_HEIGHT;
    }
    int refOutputSize = refOutputHeight * refOutputWidth * dstDescPtr->c;
    Rpp64u binOutputSize = refOutputHeight * refOutputWidth * dstDescPtr->n * 4;
    int pln1RefStride = refOutputHeight * refOutputWidth * dstDescPtr->n * 3;

    string dataType[4] = {"_u8_", "_f32_", "_f16_", "_i8_"};

    if(srcDescPtr->dataType == dstDescPtr->dataType)
        func += dataType[srcDescPtr->dataType];
    else
    {
        func = func + dataType[srcDescPtr->dataType];
        func.resize(func.size() - 1);
        func += dataType[dstDescPtr->dataType];
    }

    std::string binFile = func + "Tensor";
    if(srcDescPtr->layout == RpptLayout::NHWC)
        func += "Tensor_PKD3";
    else
    {
        if (srcDescPtr->c == 3)
            func += "Tensor_PLN3";
        else
            func += "Tensor_PLN1";
    }
    if(dstDescPtr->layout == RpptLayout::NHWC)
        func += "_to_PKD3";
    else
    {
        if (dstDescPtr->c == 3)
            func += "_to_PLN3";
        else
        {
            func += "_to_PLN1";
            if(testCase == COLOR_TO_GREYSCALE)
                pln1RefStride = 0;
        }
    }

    if(testCase == RESIZE ||testCase == ROTATE || testCase == WARP_AFFINE || testCase == WARP_PERSPECTIVE || testCase == REMAP)
    {
        func += "_interpolationType" + interpolationTypeName;
        binFile += "_interpolationType" + interpolationTypeName;
    }
    else if(testCase == NOISE)
    {
        func += "_noiseType" + noiseTypeName;
        binFile += "_noiseType" + noiseTypeName;
    }
    else if(testCase == BOX_FILTER || testCase == GAUSSIAN_FILTER)
    {
        func += "_kernelSize" + std::to_string(additionalParam);
        binFile += "_kernelSize" + std::to_string(additionalParam);
    }
    refFile = scriptPath + "/../REFERENCE_OUTPUT/" + funcName + "/"+ binFile + ".bin";
    int fileMatch = 0;
    if(dstDescPtr->dataType == RpptDataType::U8)
    {
        Rpp8u* binaryContent = (Rpp8u *)malloc(binOutputSize * sizeof(Rpp8u));
        read_bin_file(refFile, binaryContent);

        if(dstDescPtr->layout == RpptLayout::NHWC)
            compare_outputs_pkd_and_pln1((Rpp8u*)output, binaryContent, dstDescPtr, dstImgSizes, refOutputHeight, refOutputWidth, refOutputSize, fileMatch);
        else if(dstDescPtr->layout == RpptLayout::NCHW && dstDescPtr->c == 3)
            compare_outputs_pln3((Rpp8u*)output, binaryContent, dstDescPtr, dstImgSizes, refOutputHeight, refOutputWidth, refOutputSize, fileMatch);
        else
            compare_outputs_pkd_and_pln1((Rpp8u*)output, binaryContent + pln1RefStride, dstDescPtr, dstImgSizes, refOutputHeight, refOutputWidth, refOutputSize, fileMatch);
        free(binaryContent);
    }
    else
    {
        Rpp32f* binaryContent = (Rpp32f *)malloc(binOutputSize * sizeof(Rpp32f));
        read_bin_file(refFile, binaryContent);

        if(dstDescPtr->layout == RpptLayout::NHWC)
            compare_outputs_pkd_and_pln1((Rpp32f*)output, binaryContent, dstDescPtr, dstImgSizes, refOutputHeight, refOutputWidth, refOutputSize, fileMatch);
        else if(dstDescPtr->layout == RpptLayout::NCHW && dstDescPtr->c == 3)
            compare_outputs_pln3((Rpp32f*)output, binaryContent, dstDescPtr, dstImgSizes, refOutputHeight, refOutputWidth, refOutputSize, fileMatch);
        else
            compare_outputs_pkd_and_pln1((Rpp32f*)output, binaryContent + pln1RefStride, dstDescPtr, dstImgSizes, refOutputHeight, refOutputWidth, refOutputSize, fileMatch);
        free(binaryContent);
    }

    std::cout << std::endl << "\nResults for " << func << " :" << std::endl;
    std::string status = func + ": ";
    if(fileMatch == dstDescPtr->n)
    {
        std::cout << "PASSED!";
        status += "PASSED";
    }
    else
    {
        std::cout << "FAILED! " << fileMatch << "/" << dstDescPtr->n << " outputs are matching with reference outputs";
        status += "FAILED";
    }

    // Append the QA results to file
    std::string qaResultsPath = dst + "/QA_results.txt";
    std:: ofstream qaResults(qaResultsPath, ios_base::app);
    if (qaResults.is_open())
    {
        qaResults << status << std::endl;
        qaResults.close();
    }
}

// compares reduction type functions outputs
template <typename T>
inline void compare_reduction_output(T* output, string funcName, RpptDescPtr srcDescPtr, int testCase, string dst, string scriptPath)
{
    string func = funcName;
    string dataType[4] = {"_u8_", "_f16_", "_f32_", "_i8_"};

    func += dataType[srcDescPtr->dataType];

    if(srcDescPtr->layout == RpptLayout::NHWC)
        func += "Tensor_PKD3";
    else
    {
        if (srcDescPtr->c == 3)
            func += "Tensor_PLN3";
        else
            func += "Tensor_PLN1";
    }

    int fileMatch = 0;
    int matched_values = 0;

    T *refOutput;
    int numChannels = (srcDescPtr->c == 1) ? 1 : 3;
    int numOutputs = (srcDescPtr->c == 1) ? srcDescPtr->n : srcDescPtr->n * 4;
    if(testCase == TENSOR_MIN)
        refOutput = reinterpret_cast<T*>(TensorMinReferenceOutputs[numChannels].data());
    else if(testCase == TENSOR_MAX)
        refOutput = reinterpret_cast<T*>(TensorMaxReferenceOutputs[numChannels].data());
    else if(testCase == TENSOR_SUM)
        refOutput = reinterpret_cast<T*>(TensorSumReferenceOutputs[numChannels].data());
    else if(testCase == TENSOR_MEAN)
        refOutput = reinterpret_cast<T*>(TensorMeanReferenceOutputs[numChannels].data());
    else if(testCase == TENSOR_STDDEV)
        refOutput = reinterpret_cast<T*>(TensorStddevReferenceOutputs[numChannels].data());

    if(srcDescPtr->c == 1)
    {
        for(int i = 0; i < srcDescPtr->n; i++)
        {
            int diff = abs(static_cast<int>(output[i] - refOutput[i]));
            if(diff <= CUTOFF)
                fileMatch++;
        }
    }
    else
    {
        for(int i = 0; i < srcDescPtr->n; i++)
        {
            matched_values = 0;
            for(int j = 0; j < 4; j++)
            {
                int diff = abs(static_cast<int>(output[(i * 4) + j] - refOutput[(i * 4) + j]));
                if(diff <= CUTOFF)
                    matched_values++;
            }
            if(matched_values == 4)
                fileMatch++;
        }
    }

    std::cout << std::endl << "Results for " << func << " :" << std::endl;
    std::string status = func + ": ";
    if(fileMatch == srcDescPtr->n)
    {
        std::cout << "PASSED!" << std::endl;
        status += "PASSED";
    }
    else
    {
        std::cout << "FAILED! " << fileMatch << "/" << srcDescPtr->n << " outputs are matching with reference outputs" << std::endl;
        status += "FAILED";
    }

    // Append the QA results to file
    std::string qaResultsPath = dst + "/QA_results.txt";
    std:: ofstream qaResults(qaResultsPath, ios_base::app);
    if (qaResults.is_open())
    {
        qaResults << status << std::endl;
        qaResults.close();
    }
}

// print array of any bit depth for specified length
template <typename T>
inline void print_array(T *src, Rpp32u length, Rpp32u precision)
{
    for (int i = 0; i < length; i++)
        std::cout << " " << std::fixed << std::setprecision(precision) << static_cast<Rpp32f>(src[i]) << " ";
}

// Used to randomly swap values present in array of size n
inline void randomize(unsigned int arr[], unsigned int n)
{
    // Use a different seed value each time
    srand (time(NULL));
    for (unsigned int i = n - 1; i > 0; i--)
    {
        // Pick a random index from 0 to i
        unsigned int j = rand() % (i + 1);
        std::swap(arr[i], arr[j]);
    }
}

// Generates a random value between given min and max values
int inline randrange(int min, int max)
{
    if(max < 0)
        return -1;
    return rand() % (max - min + 1) + min;
}

// RICAP Input Crop Region initializer for QA testing and golden output match
void inline init_ricap_qa(int width, int height, int batchSize, Rpp32u *permutationTensor, RpptROIPtr roiPtrInputCropRegion)
{
    Rpp32u initialPermuteArray[batchSize], permutedArray[batchSize * 4];
    int part0Width = 40; //Set part0 width around 1/3 of image width
    int part0Height = 72; //Set part0 height around 1/2 of image height

    for (uint i = 0; i < batchSize; i++)
        initialPermuteArray[i] = i;

    for(int i = 0; i < 4; i++)
        memcpy(permutedArray + (batchSize * i), initialPermuteArray, batchSize * sizeof(Rpp32u));

    for (uint i = 0, j = 0; j < batchSize * 4; i++, j += 4)
    {
        permutationTensor[j] = permutedArray[i];
        permutationTensor[j + 1] = permutedArray[i + batchSize];
        permutationTensor[j + 2] = permutedArray[i + (batchSize * 2)];
        permutationTensor[j + 3] = permutedArray[i + (batchSize * 3)];
    }

    roiPtrInputCropRegion[0].xywhROI = {width - part0Width, 0, part0Width, part0Height};
    roiPtrInputCropRegion[1].xywhROI = {part0Width, 0, width - part0Width, part0Height};
    roiPtrInputCropRegion[2].xywhROI = {0, part0Height, part0Width, height - part0Height};
    roiPtrInputCropRegion[3].xywhROI = {0, part0Height, width - part0Width, height - part0Height};
}

// RICAP Input Crop Region initializer for unit and performance testing
void inline init_ricap(int width, int height, int batchSize, Rpp32u *permutationTensor, RpptROIPtr roiPtrInputCropRegion)
{
    Rpp32u initialPermuteArray[batchSize], permutedArray[batchSize * 4];

    for (uint i = 0; i < batchSize; i++)
        initialPermuteArray[i] = i;

    std::random_device rd;
    std::mt19937 gen(rd()); // Pseudo random number generator
    static std::uniform_real_distribution<double> unif(0.3, 0.7); // Generates a uniform real distribution between 0.3 and 0.7
    double randVal = unif(gen);

    std::random_device rd1;
    std::mt19937 gen1(rd1());
    static std::uniform_real_distribution<double> unif1(0.3, 0.7);
    double randVal1 = unif1(gen1);

    for(int i = 0; i < 4; i++)
    {
        randomize(initialPermuteArray, batchSize);
        memcpy(permutedArray + (batchSize * i), initialPermuteArray, batchSize * sizeof(Rpp32u));
    }

    for (uint i = 0, j = 0; j < batchSize * 4; i++, j += 4)
    {
        permutationTensor[j] = permutedArray[i];
        permutationTensor[j + 1] = permutedArray[i + batchSize];
        permutationTensor[j + 2] = permutedArray[i + (batchSize * 2)];
        permutationTensor[j + 3] = permutedArray[i + (batchSize * 3)];
    }

    int part0Width = std::round(randVal * width);
    int part0Height = std::round(randVal1 * height);
    roiPtrInputCropRegion[0].xywhROI = {randrange(0, width - part0Width - 8), randrange(0, height - part0Height), part0Width, part0Height}; // Subtracted x coordinate by 8 to avoid corruption when HIP processes 8 pixels at once
    roiPtrInputCropRegion[1].xywhROI = {randrange(0, part0Width - 8), randrange(0, height - part0Height), width - part0Width, part0Height};
    roiPtrInputCropRegion[2].xywhROI = {randrange(0, width - part0Width - 8), randrange(0, part0Height), part0Width, height - part0Height};
    roiPtrInputCropRegion[3].xywhROI = {randrange(0, part0Width - 8), randrange(0, part0Height), width - part0Width, height - part0Height};
}

void inline init_remap(RpptDescPtr tableDescPtr, RpptDescPtr srcDescPtr, RpptROIPtr roiTensorPtrSrc, Rpp32f *rowRemapTable, Rpp32f *colRemapTable)
{
    tableDescPtr->c = 1;
    tableDescPtr->strides.nStride = srcDescPtr->h * srcDescPtr->w;
    tableDescPtr->strides.hStride = srcDescPtr->w;
    tableDescPtr->strides.wStride = tableDescPtr->strides.cStride = 1;
    Rpp32u batchSize = srcDescPtr->n;

    for (Rpp32u count = 0; count < batchSize; count++)
    {
        Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
        rowRemapTableTemp = rowRemapTable + count * tableDescPtr->strides.nStride;
        colRemapTableTemp = colRemapTable + count * tableDescPtr->strides.nStride;
        Rpp32u halfWidth = roiTensorPtrSrc[count].xywhROI.roiWidth / 2;
        for (Rpp32u i = 0; i < roiTensorPtrSrc[count].xywhROI.roiHeight; i++)
        {
            Rpp32f *rowRemapTableTempRow, *colRemapTableTempRow;
            rowRemapTableTempRow = rowRemapTableTemp + i * tableDescPtr->strides.hStride;
            colRemapTableTempRow = colRemapTableTemp + i * tableDescPtr->strides.hStride;
            Rpp32u j = 0;
            for (; j < halfWidth; j++)
            {
                *rowRemapTableTempRow = i;
                *colRemapTableTempRow = halfWidth - j;

                rowRemapTableTempRow++;
                colRemapTableTempRow++;
            }
            for (; j < roiTensorPtrSrc[count].xywhROI.roiWidth; j++)
            {
                *rowRemapTableTempRow = i;
                *colRemapTableTempRow = j;

                rowRemapTableTempRow++;
                colRemapTableTempRow++;
            }
        }
    }
}

// initialize the roi, anchor and shape values required for slice
void init_slice(RpptGenericDescPtr descriptorPtr3D, RpptROIPtr roiPtrSrc, Rpp32u *roiTensor, Rpp32s *anchorTensor, Rpp32s *shapeTensor)
{
    if(descriptorPtr3D->numDims == 4)
    {
        if (descriptorPtr3D->layout == RpptLayout::NCHW)
        {
            for(int i = 0; i < descriptorPtr3D->dims[0]; i++)
            {
                int idx1 = i * 3;
                int idx2 = i * 6;
                roiTensor[idx2] = anchorTensor[idx1] = 0;
                roiTensor[idx2 + 1] = anchorTensor[idx1 + 1] = roiPtrSrc[i].xywhROI.xy.y;
                roiTensor[idx2 + 2] = anchorTensor[idx1 + 2] = roiPtrSrc[i].xywhROI.xy.x;
                roiTensor[idx2 + 3] = descriptorPtr3D->dims[1];
                roiTensor[idx2 + 4] = roiPtrSrc[i].xywhROI.roiHeight;
                roiTensor[idx2 + 5] = roiPtrSrc[i].xywhROI.roiWidth;
                shapeTensor[idx1] = roiTensor[idx2 + 3];
                shapeTensor[idx1 + 1] = roiTensor[idx2 + 4] / 2;
                shapeTensor[idx1 + 2] = roiTensor[idx2 + 5] / 2;
            }
        }
        else if(descriptorPtr3D->layout == RpptLayout::NHWC)
        {
            for(int i = 0; i < descriptorPtr3D->dims[0]; i++)
            {
                int idx1 = i * 3;
                int idx2 = i * 6;
                roiTensor[idx2] = anchorTensor[idx1] = roiPtrSrc[i].xywhROI.xy.y;
                roiTensor[idx2 + 1] = anchorTensor[idx1 + 1] = roiPtrSrc[i].xywhROI.xy.x;
                roiTensor[idx2 + 2] = anchorTensor[idx1 + 2] = 0;
                roiTensor[idx2 + 3] = roiPtrSrc[i].xywhROI.roiHeight;
                roiTensor[idx2 + 4] = roiPtrSrc[i].xywhROI.roiWidth;
                roiTensor[idx2 + 5] = descriptorPtr3D->dims[3];
                shapeTensor[idx1] = roiTensor[idx2 + 3] / 2;
                shapeTensor[idx1 + 1] = roiTensor[idx2 + 4] / 2;
                shapeTensor[idx1 + 2] = roiTensor[idx2 + 5];
            }
        }
    }
    if(descriptorPtr3D->numDims == 3)
    {
        for(int i = 0; i < descriptorPtr3D->dims[0]; i++)
        {
            int idx1 = i * 2;
            int idx2 = i * 4;
            roiTensor[idx2] = anchorTensor[idx1] = roiPtrSrc[i].xywhROI.xy.y;
            roiTensor[idx2 + 1] = anchorTensor[idx1 + 1] = roiPtrSrc[i].xywhROI.xy.x;
            roiTensor[idx2 + 2] = roiPtrSrc[i].xywhROI.roiHeight;
            roiTensor[idx2 + 3] = roiPtrSrc[i].xywhROI.roiWidth;
            shapeTensor[idx1] = roiTensor[idx2 + 2] / 2;
            shapeTensor[idx1 + 1] = roiTensor[idx2 + 3] / 2;
        }
    }
}

// Erase Region initializer for unit and performance testing
void inline init_erase(int batchSize, int boxesInEachImage, Rpp32u* numOfBoxes, RpptRoiLtrb* anchorBoxInfoTensor, RpptROIPtr roiTensorPtrSrc, int channels, Rpp32f *colorBuffer, int inputBitDepth)
{
    Rpp8u *colors8u = reinterpret_cast<Rpp8u *>(colorBuffer);
    Rpp16f *colors16f = reinterpret_cast<Rpp16f *>(colorBuffer);
    Rpp32f *colors32f = colorBuffer;
    Rpp8s *colors8s = reinterpret_cast<Rpp8s *>(colorBuffer);
    for(int i = 0; i < batchSize; i++)
    {
        numOfBoxes[i] = boxesInEachImage;
        int idx = boxesInEachImage * i;

        anchorBoxInfoTensor[idx].lt.x = 0.125 * roiTensorPtrSrc[i].xywhROI.roiWidth;
        anchorBoxInfoTensor[idx].lt.y = 0.125 * roiTensorPtrSrc[i].xywhROI.roiHeight;
        anchorBoxInfoTensor[idx].rb.x = 0.375 * roiTensorPtrSrc[i].xywhROI.roiWidth;
        anchorBoxInfoTensor[idx].rb.y = 0.375 * roiTensorPtrSrc[i].xywhROI.roiHeight;

        idx++;
        anchorBoxInfoTensor[idx].lt.x = 0.125 * roiTensorPtrSrc[i].xywhROI.roiWidth;
        anchorBoxInfoTensor[idx].lt.y = 0.625 * roiTensorPtrSrc[i].xywhROI.roiHeight;
        anchorBoxInfoTensor[idx].rb.x = 0.875 * roiTensorPtrSrc[i].xywhROI.roiWidth;
        anchorBoxInfoTensor[idx].rb.y = 0.875 * roiTensorPtrSrc[i].xywhROI.roiHeight;

        idx++;
        anchorBoxInfoTensor[idx].lt.x = 0.75 * roiTensorPtrSrc[i].xywhROI.roiWidth;
        anchorBoxInfoTensor[idx].lt.y = 0.125 * roiTensorPtrSrc[i].xywhROI.roiHeight;
        anchorBoxInfoTensor[idx].rb.x = 0.875 * roiTensorPtrSrc[i].xywhROI.roiWidth;
        anchorBoxInfoTensor[idx].rb.y = 0.5 * roiTensorPtrSrc[i].xywhROI.roiHeight;

        if(channels == 3)
        {
            int idx = boxesInEachImage * 3 * i;
            colorBuffer[idx] = 0;
            colorBuffer[idx + 1] = 0;
            colorBuffer[idx + 2] = 240;
            colorBuffer[idx + 3] = 0;
            colorBuffer[idx + 4] = 240;
            colorBuffer[idx + 5] = 0;
            colorBuffer[idx + 6] = 240;
            colorBuffer[idx + 7] = 0;
            colorBuffer[idx + 8] = 0;
            for (int j = 0; j < 9; j++)
            {
                if (!inputBitDepth)
                    colors8u[idx + j] = (Rpp8u)(colorBuffer[idx + j]);
                else if (inputBitDepth == 1)
                    colors16f[idx + j] = (Rpp16f)(colorBuffer[idx + j] * ONE_OVER_255);
                else if (inputBitDepth == 2)
                    colors32f[idx + j] = (Rpp32f)(colorBuffer[idx + j] * ONE_OVER_255);
                else if (inputBitDepth == 5)
                    colors8s[idx + j] = (Rpp8s)(colorBuffer[idx + j] - 128);
            }
        }
        else
        {
            int idx = boxesInEachImage * i;
            colorBuffer[idx] = 240;
            colorBuffer[idx + 1] = 120;
            colorBuffer[idx + 2] = 60;
            for (int j = 0; j < 3; j++)
            {
                if (!inputBitDepth)
                    colors8u[idx + j] = (Rpp8u)(colorBuffer[idx + j]);
                else if (inputBitDepth == 1)
                    colors16f[idx + j] = (Rpp16f)(colorBuffer[idx + j] * ONE_OVER_255);
                else if (inputBitDepth == 2)
                    colors32f[idx + j] = (Rpp32f)(colorBuffer[idx + j] * ONE_OVER_255);
                else if (inputBitDepth == 5)
                    colors8s[idx + j] = (Rpp8s)(colorBuffer[idx + j] - 128);
            }
        }
    }
}

// Lens correction initializer for unit and performance testing
void inline init_lens_correction(int batchSize, RpptDescPtr srcDescPtr, Rpp32f *cameraMatrix, Rpp32f *distortionCoeffs, RpptDescPtr tableDescPtr)
{
    typedef struct { Rpp32f data[9]; } Rpp32f9;
    typedef struct { Rpp32f data[8]; } Rpp32f8;
    Rpp32f9 *cameraMatrix_f9 = reinterpret_cast<Rpp32f9 *>(cameraMatrix);
    Rpp32f8 *distortionCoeffs_f8 = reinterpret_cast<Rpp32f8 *>(distortionCoeffs);
    Rpp32f9 sampleCameraMatrix = {534.07088364, 0, 341.53407554, 0, 534.11914595, 232.94565259, 0, 0, 1};
    Rpp32f8 sampleDistortionCoeffs = {-0.29297164, 0.10770696, 0.00131038, -0.0000311, 0.0434798, 0, 0, 0};
    for (int i = 0; i < batchSize; i++)
    {
        cameraMatrix_f9[i] = sampleCameraMatrix;
        distortionCoeffs_f8[i] = sampleDistortionCoeffs;
    }

    tableDescPtr->c = 1;
    tableDescPtr->strides.nStride = srcDescPtr->h * srcDescPtr->w;
    tableDescPtr->strides.hStride = srcDescPtr->w;
    tableDescPtr->strides.wStride = tableDescPtr->strides.cStride = 1;
}
