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
#include <stdio.h>
#include <string.h>
#if defined(RPP_TEST_SUITE_HAVE_OPENCV) && RPP_TEST_SUITE_HAVE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;
#endif
#include <omp.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <unordered_set>
#include <vector>

#include "rpp.h"
#include "rpp_test_suite_common.h"

using namespace std;

#define CUTOFF 1
#define MAX_IMAGE_DUMP 20
#define MAX_BATCH_SIZE 512
#define GOLDEN_OUTPUT_MAX_HEIGHT \
    150  // Golden outputs are generated with MAX_HEIGHT set to 150. Changing this constant will
         // result in QA test failures
#define GOLDEN_OUTPUT_MAX_WIDTH \
    150  // Golden outputs are generated with MAX_WIDTH set to 150. Changing this constant will
         // result in QA test failures
#define LENS_CORRECTION_GOLDEN_OUTPUT_MAX_HEIGHT \
    480  // Lens correction golden outputs are generated with MAX_HEIGHT set to 480. Changing this
         // constant will result in QA test failures
#define LENS_CORRECTION_GOLDEN_OUTPUT_MAX_WIDTH \
    640  // Lens correction golden outputs are generated with MAX_WIDTH set to 640. Changing this
         // constant will result in QA test failures

#define CHECK_RETURN_STATUS(x)                                                              \
    do {                                                                                    \
        int retval = (x);                                                                   \
        if (retval != 0) {                                                                  \
            fprintf(stderr, "Runtime error: %s returned %d at %s:%d", #x, retval, __FILE__, \
                    __LINE__);                                                              \
            exit(-1);                                                                       \
        }                                                                                   \
    } while (0)

std::map<int, string> augmentationMap = {{0, "brightness"},
                                         {1, "gamma_correction"},
                                         {2, "blend"},
                                         {4, "contrast"},
                                         {5, "pixelate"},
                                         {6, "jitter"},
                                         {7, "snow"},
                                         {8, "noise"},
                                         {10, "fog"},
                                         {11, "rain"},
                                         {13, "exposure"},
                                         {15, "threshold"},
                                         {20, "flip"},
                                         {21, "resize"},
                                         {23, "rotate"},
                                         {24, "warp_afffine"},
                                         {25, "fisheye"},
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
                                         {40, "erode"},
                                         {41, "dilate"},
                                         {42, "hue"},
                                         {43, "saturation"},
                                         {45, "color_temperature"},
                                         {46, "vignette"},
                                         {49, "box_filter"},
                                         {50, "sobel_filter"},
                                         {51, "median_filter"},
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
                                         {85, "channel_permute"},
                                         {86, "color_to_greyscale"},
                                         {87, "tensor_sum"},
                                         {88, "tensor_min"},
                                         {89, "tensor_max"},
                                         {90, "tensor_mean"},
                                         {91, "tensor_stddev"},
                                         {92, "slice"},
                                         {93, "jpeg_compression_distortion"},
                                         {94, "posterize"},
                                         {95, "solarize"},
                                         {96, "channel_dropout"},
                                         {97, "cutout_dropout"},
                                         {98, "grid_dropout"},
                                         {99, "random_erase"},
                                         {100, "coarse_dropout"},
                                         {101, "emboss"},
                                         {102, "histogram_equalize"},
                                         {103, "yuv_to_rgb"},
                                         {104, "yuv_to_rgb_cubic_v"},
                                         {105, "yuv_to_rgb_linear_v"}};

enum Augmentation {
    BRIGHTNESS = 0,
    GAMMA_CORRECTION = 1,
    BLEND = 2,
    CONTRAST = 4,
    PIXELATE = 5,
    JITTER = 6,
    SNOW = 7,
    NOISE = 8,
    FOG = 10,
    RAIN = 11,
    EXPOSURE = 13,
    THRESHOLD = 15,
    FLIP = 20,
    RESIZE = 21,
    ROTATE = 23,
    WARP_AFFINE = 24,
    FISHEYE = 25,
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
    ERODE = 40,
    DILATE = 41,
    HUE = 42,
    SATURATION = 43,
    COLOR_TEMPERATURE = 45,
    VIGNETTE = 46,
    BOX_FILTER = 49,
    SOBEL_FILTER = 50,
    MEDIAN_FILTER = 51,
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
    CHANNEL_PERMUTE = 85,
    COLOR_TO_GREYSCALE = 86,
    TENSOR_SUM = 87,
    TENSOR_MIN = 88,
    TENSOR_MAX = 89,
    TENSOR_MEAN = 90,
    TENSOR_STDDEV = 91,
    SLICE = 92,
    JPEG_COMPRESSION_DISTORTION = 93,
    POSTERIZE = 94,
    SOLARIZE = 95,
    CHANNEL_DROPOUT = 96,
    CUTOUT_DROPOUT = 97,
    GRID_DROPOUT = 98,
    RANDOM_ERASE = 99,
    COARSE_DROPOUT = 100,
    EMBOSS = 101,
    HISTOGRAM_EQUALIZE = 102,
    YUV_TO_RGB = 103,
    YUV_TO_RGB_CUBIC_V = 104,
    YUV_TO_RGB_LINEAR_V = 105
};

// Enum for dropout types
enum DropoutType { DROPOUT_CUTOUT = 1, DROPOUT_RANDOM_ERASING = 3, DROPOUT_COARSE = 4 };

const unordered_set<int> additionalParamCases = {
    NOISE,           RESIZE, ROTATE,          WARP_AFFINE,  WARP_PERSPECTIVE,
    ERODE,           DILATE, BOX_FILTER,      SOBEL_FILTER, MEDIAN_FILTER,
    GAUSSIAN_FILTER, REMAP,  CHANNEL_PERMUTE, EMBOSS};
const unordered_set<int> kernelSizeCases = {ERODE,         DILATE,          BOX_FILTER,
                                            MEDIAN_FILTER, GAUSSIAN_FILTER, EMBOSS};
const unordered_set<int> dualInputCases = {BLEND, NON_LINEAR_BLEND, CROP_AND_PATCH, MAGNITUDE,
                                           PHASE, BITWISE_AND,      BITWISE_XOR,    BITWISE_OR};
const unordered_set<int> randomOutputCases = {JITTER, NOISE, FOG, RAIN, SPATTER};
const unordered_set<int> nonQACases = {WARP_AFFINE, WARP_PERSPECTIVE};
const unordered_set<int> interpolationTypeCases = {RESIZE, ROTATE, WARP_AFFINE, WARP_PERSPECTIVE,
                                                   REMAP};
const unordered_set<int> reductionTypeCases = {TENSOR_SUM, TENSOR_MIN, TENSOR_MAX, TENSOR_MEAN,
                                               TENSOR_STDDEV};
const unordered_set<int> noiseTypeCases = {NOISE};
const unordered_set<int> pln1OutTypeCases = {COLOR_TO_GREYSCALE, SOBEL_FILTER};
const unordered_set<int> kernelSizeAndGradientCases = {SOBEL_FILTER};

// Golden outputs for Tensor min Kernel
std::map<int, std::vector<Rpp8u>> TensorMinReferenceOutputs_U8 = {
    {1, {1, 1, 7}}, {3, {0, 0, 0, 0, 2, 0, 0, 0, 7, 9, 0, 0}}};

// Golden outputs for Tensor min Kernel
std::map<int, std::vector<Rpp32f>> TensorMinReferenceOutputs_F32 = {
    {1, {0.004, 0.004, 0.027}},
    {3, {0.000, 0.000, 0.000, 0.000, 0.008, 0.000, 0.000, 0.000, 0.027, 0.035, 0.000, 0.000}}};

// Golden outputs for Tensor max Kernel
std::map<int, std::vector<Rpp8u>> TensorMaxReferenceOutputs_U8 = {
    {1, {239, 244, 253}}, {3, {255, 240, 236, 255, 255, 242, 241, 255, 253, 255, 255, 255}}};

// Golden outputs for Tensor max Kernel
std::map<int, std::vector<Rpp32f>> TensorMaxReferenceOutputs_F32 = {
    {1, {0.937, 0.961, 1.000}},
    {3, {1.000, 0.941, 0.925, 1.000, 1.000, 0.949, 0.945, 1.000, 0.992, 1.000, 1.000, 1.000}}};

// Golden outputs for Tensor sum Kernel
std::map<int, std::vector<uint64_t>> TensorSumReferenceOutputs_U8 = {
    {1, {334272, 813961, 2631106}},
    {3,
     {348380, 340992, 262616, 951988, 1056552, 749506, 507441, 2313499, 2170646, 2732368, 3320699,
      8223713}}};

// Golden outputs for Tensor sum Kernel
std::map<int, std::vector<Rpp32f>> TensorSumReferenceOutputs_F32 = {
    {1, {1310.686, 3190.083, 10318.138}},
    {3,
     {1366.196, 1337.224, 1029.867, 3733.286, 4143.341, 2939.239, 1989.965, 9072.546, 8512.338,
      10715.169, 13022.350, 32249.857}}};

// Golden outputs for Tensor mean Kernel
std::map<int, std::vector<Rpp32f>> TensorMeanReferenceOutputs_U8 = {
    {1, {133.690, 81.347, 116.939}},
    {3,
     {139.352, 136.397, 105.046, 126.932, 105.655, 74.951, 50.744, 77.117, 96.473, 121.439, 147.587,
      121.833}}};

// Golden outputs for Tensor mean Kernel
std::map<int, std::vector<Rpp32f>> TensorMeanReferenceOutputs_F32 = {
    {1, {0.524, 0.319, 0.459}},
    {3, {0.546, 0.535, 0.412, 0.498, 0.414, 0.294, 0.199, 0.302, 0.378, 0.476, 0.579, 0.478}}};

// Golden outputs for Tensor stddev Kernel
std::map<int, std::vector<float>> TensorStddevReferenceOutputs_U8 = {
    {1, {49.583, 54.623, 47.649}},
    {3,
     {57.416, 47.901, 53.235, 55.220, 68.471, 55.735, 46.668, 61.880, 47.462, 49.039, 67.269,
      59.130}}};

// Golden outputs for Tensor stddev Kernel
std::map<int, std::vector<Rpp32f>> TensorStddevReferenceOutputs_F32 = {
    {1, {49.583, 54.623, 47.649}},
    {3,
     {57.416, 47.901, 53.235, 55.220, 68.471, 55.735, 46.668, 61.880, 47.462, 49.039, 67.269,
      59.130}}};

template <typename T>
inline T validate_pixel_range(T pixel) {
    pixel = (pixel < static_cast<Rpp32f>(0))
                ? (static_cast<Rpp32f>(0))
                : ((pixel < static_cast<Rpp32f>(255)) ? pixel : (static_cast<Rpp32f>(255)));
    return pixel;
}

// returns the gradient type applied to an image
inline std::string get_gradient_type(unsigned int val) {
    switch (val) {
        case 0:
            return "X";
        case 1:
            return "Y";
        case 2:
            return "XY";
        default:
            return "X";
    }
}

// returns the kernel size and gradient type for sobel filter operations.
inline std::string get_kernel_size_and_gradient_type(unsigned int val, Rpp32u& kernelSize,
                                                     Rpp32u& gradientType) {
    unsigned int kernelIndex = val / 3;
    gradientType = val % 3;
    switch (kernelIndex) {
        case 0:
            kernelSize = 3;
            break;
        case 1:
            kernelSize = 5;
            break;
        case 2:
            kernelSize = 7;
            break;
        default:
            kernelSize = 3;
            break;
    }
    return ("_kernelSize" + std::to_string(kernelSize) + "_gradient" +
            get_gradient_type(gradientType));
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
    else
        return 0;
}

// returns the interpolation type used for image resizing or scaling operations.
inline std::string get_interpolation_type(unsigned int val,
                                          RpptInterpolationType& interpolationType) {
    switch (val) {
        case 0: {
            interpolationType = RpptInterpolationType::NEAREST_NEIGHBOR;
            return "NearestNeighbor";
        }
        case 2: {
            interpolationType = RpptInterpolationType::BICUBIC;
            return "Bicubic";
        }
        case 3: {
            interpolationType = RpptInterpolationType::LANCZOS;
            return "Lanczos";
        }
        case 4: {
            interpolationType = RpptInterpolationType::TRIANGULAR;
            return "Triangular";
        }
        case 5: {
            interpolationType = RpptInterpolationType::GAUSSIAN;
            return "Gaussian";
        }
        default: {
            interpolationType = RpptInterpolationType::BILINEAR;
            return "Bilinear";
        }
    }
}

// returns the noise type applied to an image
inline std::string get_noise_type(unsigned int val) {
    switch (val) {
        case 0:
            return "SaltAndPepper";
        case 1:
            return "Gaussian";
        case 2:
            return "Shot";
        default:
            return "SaltAndPepper";
    }
}

// returns number of input channels according to layout type
inline int set_input_channels(int layoutType) {
    if (layoutType == 0 || layoutType == 1)
        return 3;
    else
        return 1;
}

// returns function type
inline string set_function_type(int layoutType, int pln1OutTypeCase, int outputFormatToggle,
                                string backend) {
    string funcType;
    if (layoutType == 0) {
        funcType = "Tensor_" + backend + "_PKD3";
        if (pln1OutTypeCase)
            funcType += "_toPLN1";
        else {
            if (outputFormatToggle)
                funcType += "_toPLN3";
            else
                funcType += "_toPKD3";
        }
    } else if (layoutType == 1) {
        funcType = "Tensor_" + backend + "_PLN3";
        if (pln1OutTypeCase)
            funcType += "_toPLN1";
        else {
            if (outputFormatToggle)
                funcType += "_toPKD3";
            else
                funcType += "_toPLN3";
        }
    } else {
        funcType = "Tensor_" + backend + "_PLN1";
        funcType += "_toPLN1";
    }

    return funcType;
}

// sets descriptor data types of src/dst
inline void set_descriptor_data_type(int BitDepthTestMode, string& funcName, RpptDescPtr srcDescPtr,
                                     RpptDescPtr dstDescPtr) {
    if (BitDepthTestMode == U8_TO_U8) {
        funcName += "_u8_";
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::U8;
    } else if (BitDepthTestMode == F16_TO_F16) {
        funcName += "_f16_";
        srcDescPtr->dataType = RpptDataType::F16;
        dstDescPtr->dataType = RpptDataType::F16;
    } else if (BitDepthTestMode == F32_TO_F32) {
        funcName += "_f32_";
        srcDescPtr->dataType = RpptDataType::F32;
        dstDescPtr->dataType = RpptDataType::F32;
    } else if (BitDepthTestMode == U8_TO_F16) {
        funcName += "_u8_f16_";
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::F16;
    } else if (BitDepthTestMode == U8_TO_F32) {
        funcName += "_u8_f32_";
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::F32;
    } else if (BitDepthTestMode == I8_TO_I8) {
        funcName += "_i8_";
        srcDescPtr->dataType = RpptDataType::I8;
        dstDescPtr->dataType = RpptDataType::I8;
    } else if (BitDepthTestMode == U8_TO_I8) {
        funcName += "_u8_i8_";
        srcDescPtr->dataType = RpptDataType::U8;
        dstDescPtr->dataType = RpptDataType::I8;
    }
}

// sets descriptor layout of src/dst
inline void set_descriptor_layout(RpptDescPtr srcDescPtr, RpptDescPtr dstDescPtr, int layoutType,
                                  bool pln1OutTypeCase, int outputFormatToggle) {
    if (layoutType == 0) {
        srcDescPtr->layout = RpptLayout::NHWC;
        // Set src/dst layouts in tensor descriptors
        if (pln1OutTypeCase)
            dstDescPtr->layout = RpptLayout::NCHW;
        else {
            if (outputFormatToggle == 0)
                dstDescPtr->layout = RpptLayout::NHWC;
            else if (outputFormatToggle == 1)
                dstDescPtr->layout = RpptLayout::NCHW;
        }
    } else if (layoutType == 1) {
        srcDescPtr->layout = RpptLayout::NCHW;
        // Set src/dst layouts in tensor descriptors
        if (pln1OutTypeCase)
            dstDescPtr->layout = RpptLayout::NCHW;
        else {
            if (outputFormatToggle == 0)
                dstDescPtr->layout = RpptLayout::NCHW;
            else if (outputFormatToggle == 1)
                dstDescPtr->layout = RpptLayout::NHWC;
        }
    } else {
        // Set src/dst layouts in tensor descriptors
        srcDescPtr->layout = RpptLayout::NCHW;
        dstDescPtr->layout = RpptLayout::NCHW;
    }
}

// Sidecar metadata: shared key=value format for NV12 .yuv inputs and packed-pixel .raw inputs
// (see utilities/test_suite/scripts/README.md for .raw/.info fields and the JPEG dump script).
struct RpptTestSuiteInfoSidecar {
    int width = 0;
    int height = 0;
    int channels = 0;
    bool has_channels = false;
    bool has_col_standard = false;
    bool has_color_range = false;
    int col_standard = 0;
    int color_range = 0;
};

// RGB file header for unified binary format (replaces separate .raw + .info dual files)
struct RpptImageFileHeader {
    uint32_t magic;      // 0x52474242 ("RGBB")
    uint32_t version;    // Version 1
    uint32_t width;      // Image width
    uint32_t height;     // Image height
    uint32_t channels;   // 1 for grayscale, 3 for RGB, 4 for CMYK (future)
    uint32_t reserved1;  // Reserved for future use

    static constexpr uint32_t MAGIC = 0x52474242;  // "RGBB"
    static constexpr uint32_t VERSION_1 = 1;
    static constexpr size_t HEADER_SIZE = 24;
};

// Parse .rgb file header (supports both grayscale channels=1 and RGB channels=3).
// Returns true if valid header with magic/version check.
inline bool parse_image_file_header(const std::string& filePath, RpptTestSuiteInfoSidecar& out) {
    out = RpptTestSuiteInfoSidecar();
    FILE* fp = fopen(filePath.c_str(), "rb");
    if (!fp) return false;

    RpptImageFileHeader header;
    size_t nread = fread(&header, 1, sizeof(RpptImageFileHeader), fp);
    fclose(fp);

    if (nread != sizeof(RpptImageFileHeader)) return false;

    // Validate magic
    if (header.magic != RpptImageFileHeader::MAGIC) {
        std::cerr << "Error: invalid .rgb magic number: " << filePath << std::endl;
        return false;
    }

    // Validate version
    if (header.version != RpptImageFileHeader::VERSION_1) {
        std::cerr << "Error: unsupported version " << header.version << ": " << filePath
                  << std::endl;
        return false;
    }

    out.width = (int)header.width;
    out.height = (int)header.height;
    out.channels = (int)header.channels;
    out.has_channels = true;

    // Validate dimensions (sanity check to catch corrupt headers; actual limit is available memory)
    if (out.width <= 0 || out.height <= 0) {
        std::cerr << "Error: invalid dimensions " << out.width << "x" << out.height << " in "
                  << filePath << std::endl;
        return false;
    }

    // Validate channels (1=grayscale, 3=RGB, 4=CMYK future)
    if (out.channels != 1 && out.channels != 3 && out.channels != 4) {
        std::cerr << "Error: invalid channel count (expected 1, 3, or 4), got " << out.channels
                  << " in " << filePath << std::endl;
        return false;
    }

    return true;
}

// Path to sidecar .info (strip last extension, append ".info") — used for .yuv
inline std::string test_suite_sidecar_info_path(const std::string& mediaFilePath) {
    std::string infoPath = mediaFilePath;
    size_t dot = infoPath.find_last_of('.');
    if (dot != std::string::npos) infoPath = infoPath.substr(0, dot);
    infoPath += ".info";
    return infoPath;
}

// Legacy: parse .info sidecar (now only used for YUV files)
inline bool parse_test_suite_info_sidecar(const std::string& mediaFilePath,
                                          RpptTestSuiteInfoSidecar& out) {
    out = RpptTestSuiteInfoSidecar();
    std::string infoPath = test_suite_sidecar_info_path(mediaFilePath);
    FILE* fp = fopen(infoPath.c_str(), "r");
    if (!fp) return false;
    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        int v;
        if (sscanf(line, "width=%d", &v) == 1 && v > 0) out.width = v;
        if (sscanf(line, "height=%d", &v) == 1 && v > 0) out.height = v;
        if (sscanf(line, "channels=%d", &v) == 1 && v > 0) {
            out.channels = v;
            out.has_channels = true;
        }
        if (sscanf(line, "col_standard=%d", &v) == 1) {
            out.col_standard = v;
            out.has_col_standard = true;
        }
        if (sscanf(line, "color_range=%d", &v) == 1) {
            out.color_range = v;
            out.has_color_range = true;
        }
    }
    fclose(fp);
    return (out.width > 0 && out.height > 0);
}

// sets values of maxHeight and maxWidth (decoderType 0: .rgb with embedded header; 1: OpenCV image
// files)
inline void set_max_dimensions(vector<string> imagePaths, int& maxHeight, int& maxWidth,
                               int& imagesMixed, int decoderType) {
    if (decoderType == 0) {
        for (const std::string& imagePath : imagePaths) {
            RpptTestSuiteInfoSidecar info;
            if (!parse_image_file_header(imagePath, info) || !info.has_channels ||
                info.width <= 0 || info.height <= 0) {
                std::cerr
                    << "Error: missing or invalid .rgb header (need width, height, channels) for: "
                    << imagePath << std::endl;
                continue;
            }
            int width = info.width;
            int height = info.height;
            if ((maxWidth && maxWidth != width) || (maxHeight && maxHeight != height))
                imagesMixed = 1;
            maxWidth = max(maxWidth, width);
            maxHeight = max(maxHeight, height);
        }
        return;
    }
    if (decoderType == 1) {
#if defined(RPP_TEST_SUITE_HAVE_OPENCV) && RPP_TEST_SUITE_HAVE_OPENCV
        for (const std::string& imagePath : imagePaths) {
            Mat image = imread(imagePath, IMREAD_UNCHANGED);
            if (image.empty()) {
                std::cerr << "Error: OpenCV could not read image: " << imagePath << std::endl;
                continue;
            }
            int width = image.cols;
            int height = image.rows;
            if ((maxWidth && maxWidth != width) || (maxHeight && maxHeight != height))
                imagesMixed = 1;
            maxWidth = max(maxWidth, width);
            maxHeight = max(maxHeight, height);
        }
#else
        std::cerr << "Error: decoder_type 1 (OpenCV) requires Tensor_image built with OpenCV.\n";
        std::exit(1);
#endif
        return;
    }
    std::cerr << "Error: invalid decoder_type (expected 0 = packed RAW + .info, 1 = OpenCV).\n";
    std::exit(1);
}

// NV12 QA sidecar: required width/height; optional col_standard / color_range for yuv_to_rgb (see
// RpptColorStandard / RpptColorRange in rppdefs.h). Defaults if omitted: BT.709 + full range. In
// .info use integer codes (e.g. color_range=0 studio, color_range=2 full).
struct RpptYuvNv12Sidecar {
    int width = 0;
    int height = 0;
    RpptColorStandard col_standard = RpptColorStandard_BT709;
    RpptColorRange color_range = RpptColorRange_FULL;
};

// Read full NV12 .info sidecar. Returns true when width and height are valid.
inline bool parse_yuv_nv12_sidecar(const std::string& yuvFilePath, RpptYuvNv12Sidecar& out) {
    out = RpptYuvNv12Sidecar();
    RpptTestSuiteInfoSidecar info;
    if (!parse_test_suite_info_sidecar(yuvFilePath, info)) return false;
    out.width = info.width;
    out.height = info.height;
    if (info.has_col_standard) out.col_standard = static_cast<RpptColorStandard>(info.col_standard);
    if (info.has_color_range) out.color_range = static_cast<RpptColorRange>(info.color_range);
    return (out.width > 0 && out.height > 0);
}

// Read width/height from sidecar .info only. Format: "width=3840" and "height=2160" on separate
// lines; optional col_standard / color_range ignored here.
inline bool parse_yuv_dimensions_from_sidecar(const std::string& yuvFilePath, int& width,
                                              int& height) {
    RpptYuvNv12Sidecar s;
    if (!parse_yuv_nv12_sidecar(yuvFilePath, s)) return false;
    width = s.width;
    height = s.height;
    return true;
}

// NV12/YUV dimensions: require .info sidecar next to the .yuv file. Exits on failure.
inline void parse_yuv_dimensions(const std::string& yuvFilePath, int& width, int& height) {
    if (!parse_yuv_dimensions_from_sidecar(yuvFilePath, width, height)) {
        std::string infoPath = test_suite_sidecar_info_path(yuvFilePath);
        std::cerr << "Error: no .info file for width and height.\n"
                  << "  Expected sidecar: " << infoPath << "\n"
                  << "  YUV input: " << yuvFilePath << "\n";
        std::exit(1);
    }
}

// Derive reference file basename from YUV path for per-image QA.
// Input and ref names match except extension: foo.yuv -> foo.rgb
inline std::string get_yuv_ref_basename(const std::string& yuvFilePath) {
    std::string name = yuvFilePath;
    size_t baseNameStart = name.find_last_of("/\\");
    if (baseNameStart != std::string::npos) name = name.substr(baseNameStart + 1);
    size_t dot = name.find_last_of('.');
    if (dot != std::string::npos) name = name.substr(0, dot);
    return name;
}

// Sets max dimensions from YUV inputs using required .info sidecars (parse_yuv_dimensions exits if
// missing/invalid).
inline void set_max_dimensions_yuv(vector<string> imagePaths, int& maxHeight, int& maxWidth,
                                   int& imagesMixed) {
    for (const std::string& imagePath : imagePaths) {
        int width = 0, height = 0;
        parse_yuv_dimensions(imagePath, width, height);
        if ((maxWidth && maxWidth != width) || (maxHeight && maxHeight != height)) imagesMixed = 1;
        maxWidth = std::max(maxWidth, width);
        maxHeight = std::max(maxHeight, height);
    }
}

// Sets ROI and dst sizes from YUV inputs using required .info sidecars
inline void set_src_and_dst_roi_yuv(vector<string>::const_iterator imagePathsStart,
                                    vector<string>::const_iterator imagePathsEnd,
                                    RpptROI* roiTensorPtrSrc, RpptROI* roiTensorPtrDst,
                                    RpptImagePatchPtr dstImgSizes) {
    int i = 0;
    for (auto imagePathIter = imagePathsStart; imagePathIter != imagePathsEnd;
         ++imagePathIter, i++) {
        int width = 0, height = 0;
        parse_yuv_dimensions(*imagePathIter, width, height);
        roiTensorPtrSrc[i].xywhROI = {{0, 0}, width, height};
        roiTensorPtrDst[i].xywhROI = {{0, 0}, width, height};
        dstImgSizes[i].width = width;
        dstImgSizes[i].height = height;
    }
}

// Read a batch of NV12 YUV files (Y plane then interleaved UV) into a contiguous buffer
inline void read_yuv_batch_nv12(Rpp8u* input, RpptDescPtr descPtr,
                                vector<string>::const_iterator imagesNamesStart) {
    size_t offset = 0;
    for (int i = 0; i < descPtr->n; i++) {
        std::string inputPath = *(imagesNamesStart + i);
        int width = 0, height = 0;
        parse_yuv_dimensions(inputPath, width, height);
        FILE* fp = fopen(inputPath.c_str(), "rb");
        if (!fp) {
            std::cerr << "\nUnable to open YUV file: " << inputPath;
            continue;
        }
        size_t ySize = (size_t)width * height;
        size_t uvSize = (size_t)width * height / 2;
        size_t frameSize = ySize + uvSize;
        Rpp8u* dst = input + offset;
        size_t read = fread(dst, 1, frameSize, fp);
        fclose(fp);
        if (read != frameSize) std::cerr << "\nYUV read size mismatch for " << inputPath;
        offset += frameSize;
    }
}

// sets roi xywh values and dstImg sizes (decoderType 0: .rgb with embedded header; 1: OpenCV image
// files)
inline void set_src_and_dst_roi(vector<string>::const_iterator imagePathsStart,
                                vector<string>::const_iterator imagePathsEnd,
                                RpptROI* roiTensorPtrSrc, RpptROI* roiTensorPtrDst,
                                RpptImagePatchPtr dstImgSizes, int decoderType) {
    int i = 0;
    for (auto imagePathIter = imagePathsStart; imagePathIter != imagePathsEnd;
         ++imagePathIter, i++) {
        const string& imagePath = *imagePathIter;
        int width = 0, height = 0;
        if (decoderType == 0) {
            RpptTestSuiteInfoSidecar info;
            if (!parse_image_file_header(imagePath, info) || !info.has_channels ||
                info.width <= 0 || info.height <= 0) {
                std::cerr << "Error: invalid .rgb header for ROI: " << imagePath << std::endl;
                continue;
            }
            width = info.width;
            height = info.height;
        } else if (decoderType == 1) {
#if defined(RPP_TEST_SUITE_HAVE_OPENCV) && RPP_TEST_SUITE_HAVE_OPENCV
            Mat image = imread(imagePath, IMREAD_UNCHANGED);
            if (image.empty()) {
                std::cerr << "Error: OpenCV could not read image: " << imagePath << std::endl;
                continue;
            }
            width = image.cols;
            height = image.rows;
#else
            std::cerr
                << "Error: decoder_type 1 (OpenCV) requires Tensor_image built with OpenCV.\n";
            std::exit(1);
#endif
        } else {
            std::cerr << "Error: invalid decoder_type in set_src_and_dst_roi.\n";
            std::exit(1);
        }

        roiTensorPtrSrc[i].xywhROI = {{0, 0}, width, height};
        roiTensorPtrDst[i].xywhROI = {{0, 0}, width, height};
        dstImgSizes[i].width = roiTensorPtrDst[i].xywhROI.roiWidth;
        dstImgSizes[i].height = roiTensorPtrDst[i].xywhROI.roiHeight;
    }
}

// sets generic descriptor dimensions and strides of src/dst
inline void set_generic_descriptor(RpptGenericDescPtr descriptorPtr3D, int noOfImages, int maxX,
                                   int maxY, int maxZ, int numChannels, int offsetInBytes,
                                   int layoutType) {
    descriptorPtr3D->numDims = 5;
    descriptorPtr3D->offsetInBytes = offsetInBytes;
    descriptorPtr3D->dataType = RpptDataType::F32;

    if (layoutType == 0) {
        descriptorPtr3D->layout = RpptLayout::NCDHW;
        descriptorPtr3D->dims[0] = noOfImages;
        descriptorPtr3D->dims[1] = numChannels;
        descriptorPtr3D->dims[2] = maxZ;
        descriptorPtr3D->dims[3] = maxY;
        descriptorPtr3D->dims[4] = maxX;
    } else if (layoutType == 1) {
        descriptorPtr3D->layout = RpptLayout::NDHWC;
        descriptorPtr3D->dims[0] = noOfImages;
        descriptorPtr3D->dims[1] = maxZ;
        descriptorPtr3D->dims[2] = maxY;
        descriptorPtr3D->dims[3] = maxX;
        descriptorPtr3D->dims[4] = numChannels;
    }

    descriptorPtr3D->strides[0] = descriptorPtr3D->dims[1] * descriptorPtr3D->dims[2] *
                                  descriptorPtr3D->dims[3] * descriptorPtr3D->dims[4];
    descriptorPtr3D->strides[1] =
        descriptorPtr3D->dims[2] * descriptorPtr3D->dims[3] * descriptorPtr3D->dims[4];
    descriptorPtr3D->strides[2] = descriptorPtr3D->dims[3] * descriptorPtr3D->dims[4];
    descriptorPtr3D->strides[3] = descriptorPtr3D->dims[4];
    descriptorPtr3D->strides[4] = 1;
}

// sets generic descriptor dimensions and strides of src/dst for slice functionality
inline void set_generic_descriptor_slice(RpptDescPtr srcDescPtr, RpptGenericDescPtr descriptorPtr3D,
                                         int batchSize) {
    descriptorPtr3D->offsetInBytes = 0;
    descriptorPtr3D->dataType = srcDescPtr->dataType;
    descriptorPtr3D->layout = srcDescPtr->layout;
    if (srcDescPtr->c == 3) {
        descriptorPtr3D->numDims = 4;
        descriptorPtr3D->dims[0] = batchSize;
        if (srcDescPtr->layout == RpptLayout::NHWC) {
            descriptorPtr3D->dims[1] = srcDescPtr->h;
            descriptorPtr3D->dims[2] = srcDescPtr->w;
            descriptorPtr3D->dims[3] = srcDescPtr->c;
        } else {
            descriptorPtr3D->dims[1] = srcDescPtr->c;
            descriptorPtr3D->dims[2] = srcDescPtr->h;
            descriptorPtr3D->dims[3] = srcDescPtr->w;
        }
        descriptorPtr3D->strides[0] =
            descriptorPtr3D->dims[1] * descriptorPtr3D->dims[2] * descriptorPtr3D->dims[3];
        descriptorPtr3D->strides[1] = descriptorPtr3D->dims[2] * descriptorPtr3D->dims[3];
        descriptorPtr3D->strides[2] = descriptorPtr3D->dims[3];
    } else {
        descriptorPtr3D->numDims = 3;
        descriptorPtr3D->dims[0] = batchSize;
        descriptorPtr3D->dims[1] = srcDescPtr->h;
        descriptorPtr3D->dims[2] = srcDescPtr->w;
        descriptorPtr3D->strides[0] = descriptorPtr3D->dims[1] * descriptorPtr3D->dims[2];
        descriptorPtr3D->strides[1] = descriptorPtr3D->dims[2];
    }
}

// sets descriptor dimensions and strides of src/dst
inline void set_descriptor_dims_and_strides(RpptDescPtr descPtr, int noOfImages, int maxHeight,
                                            int maxWidth, int numChannels, int offsetInBytes,
                                            int additionalStride = 0) {
    descPtr->numDims = 4;
    descPtr->offsetInBytes = offsetInBytes;
    descPtr->n = noOfImages;
    descPtr->h = maxHeight;
    descPtr->w = maxWidth;
    descPtr->c = numChannels;

    // Optionally set w stride as a multiple of 8 for src/dst
    descPtr->w = (descPtr->w / 8) * 8 + 8 + additionalStride;
    // set strides
    if (descPtr->layout == RpptLayout::NHWC) {
        descPtr->strides.nStride = descPtr->c * descPtr->w * descPtr->h;
        descPtr->strides.hStride = descPtr->c * descPtr->w;
        descPtr->strides.wStride = descPtr->c;
        descPtr->strides.cStride = 1;
    } else if (descPtr->layout == RpptLayout::NCHW) {
        descPtr->strides.nStride = descPtr->c * descPtr->w * descPtr->h;
        descPtr->strides.cStride = descPtr->w * descPtr->h;
        descPtr->strides.hStride = descPtr->w;
        descPtr->strides.wStride = 1;
    }
}

inline void set_roi_values(RpptROI* roi, RpptROI* roiTensorPtrSrc, RpptRoiType roiType,
                           int batchSize) {
    if (roiType == RpptRoiType::XYWH)
        for (int i = 0; i < batchSize; i++) roiTensorPtrSrc[i].xywhROI = roi->xywhROI;
    else if (roiType == RpptRoiType::LTRB)
        for (int i = 0; i < batchSize; i++) roiTensorPtrSrc[i].ltrbROI = roi->ltrbROI;
}

inline void convert_roi(RpptROI* roiTensorPtrSrc, RpptRoiType roiType, int batchSize) {
    if (roiType == RpptRoiType::LTRB) {
        for (int i = 0; i < batchSize; i++) {
            RpptRoiXywh roi = roiTensorPtrSrc[i].xywhROI;
            roiTensorPtrSrc[i].ltrbROI = {{roi.xy.x, roi.xy.y},
                                          {roi.roiWidth - roi.xy.x, roi.roiHeight - roi.xy.y}};
        }
    } else {
        for (int i = 0; i < batchSize; i++) {
            RpptRoiLtrb roi = roiTensorPtrSrc[i].ltrbROI;
            roiTensorPtrSrc[i].xywhROI = {
                {roi.lt.x, roi.lt.y}, roi.rb.x - roi.lt.x + 1, roi.rb.y - roi.lt.y + 1};
        }
    }
}

// Convert inputs to correponding bit depth specified by user
inline void convert_input_bitdepth(void* input, void* input_second, Rpp8u* inputu8,
                                   Rpp8u* inputu8Second, int BitDepthTestMode, Rpp64u ioBufferSize,
                                   Rpp64u inputBufferSize, RpptDescPtr srcDescPtr,
                                   bool dualInputCase, Rpp32f conversionFactor) {
    if (BitDepthTestMode == U8_TO_U8 || BitDepthTestMode == U8_TO_F16 ||
        BitDepthTestMode == U8_TO_F32) {
        memcpy(input, inputu8, inputBufferSize);
        if (dualInputCase) memcpy(input_second, inputu8Second, inputBufferSize);
    } else if (BitDepthTestMode == F16_TO_F16) {
        Rpp8u *inputTemp, *inputSecondTemp;
        Rpp16f *inputf16Temp, *inputf16SecondTemp;
        inputTemp = inputu8 + srcDescPtr->offsetInBytes;
        inputf16Temp =
            reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(input) + srcDescPtr->offsetInBytes);
        for (int i = 0; i < ioBufferSize; i++)
            *inputf16Temp++ =
                static_cast<Rpp16f>((static_cast<float>(*inputTemp++)) * conversionFactor);

        if (dualInputCase) {
            inputSecondTemp = inputu8Second + srcDescPtr->offsetInBytes;
            inputf16SecondTemp = reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(input_second) +
                                                           srcDescPtr->offsetInBytes);
            for (int i = 0; i < ioBufferSize; i++)
                *inputf16SecondTemp++ = static_cast<Rpp16f>(
                    (static_cast<float>(*inputSecondTemp++)) * conversionFactor);
        }
    } else if (BitDepthTestMode == F32_TO_F32) {
        Rpp8u *inputTemp, *inputSecondTemp;
        Rpp32f *inputf32Temp, *inputf32SecondTemp;
        inputTemp = inputu8 + srcDescPtr->offsetInBytes;
        inputf32Temp =
            reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(input) + srcDescPtr->offsetInBytes);
        for (int i = 0; i < ioBufferSize; i++)
            *inputf32Temp++ = (static_cast<Rpp32f>(*inputTemp++)) * conversionFactor;

        if (dualInputCase) {
            inputSecondTemp = inputu8Second + srcDescPtr->offsetInBytes;
            inputf32SecondTemp = reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(input_second) +
                                                           srcDescPtr->offsetInBytes);
            for (int i = 0; i < ioBufferSize; i++)
                *inputf32SecondTemp++ =
                    (static_cast<Rpp32f>(*inputSecondTemp++)) * conversionFactor;
        }
    } else if (BitDepthTestMode == I8_TO_I8) {
        Rpp8u *inputTemp, *inputSecondTemp;
        Rpp8s *inputi8Temp, *inputi8SecondTemp;

        inputTemp = inputu8 + srcDescPtr->offsetInBytes;
        inputi8Temp = static_cast<Rpp8s*>(input) + srcDescPtr->offsetInBytes;
        for (int i = 0; i < ioBufferSize; i++)
            *inputi8Temp++ = static_cast<Rpp8s>((static_cast<Rpp32s>(*inputTemp++)) - 128);

        if (dualInputCase) {
            inputSecondTemp = inputu8Second + srcDescPtr->offsetInBytes;
            inputi8SecondTemp = static_cast<Rpp8s*>(input_second) + srcDescPtr->offsetInBytes;
            for (int i = 0; i < ioBufferSize; i++)
                *inputi8SecondTemp++ =
                    static_cast<Rpp8s>((static_cast<Rpp32s>(*inputSecondTemp++)) - 128);
        }
    }
}

// Reconvert other bit depths to 8u for output display purposes
inline void convert_output_bitdepth_to_u8(void* output, Rpp8u* outputu8, int BitDepthTestMode,
                                          Rpp64u oBufferSize, Rpp64u outputBufferSize,
                                          RpptDescPtr dstDescPtr, Rpp32f invConversionFactor) {
    if (BitDepthTestMode == U8_TO_U8) {
        memcpy(outputu8, output, outputBufferSize);
    } else if ((BitDepthTestMode == F16_TO_F16) || (BitDepthTestMode == U8_TO_F16)) {
        Rpp8u* outputTemp = outputu8 + dstDescPtr->offsetInBytes;
        Rpp16f* outputf16Temp =
            reinterpret_cast<Rpp16f*>(static_cast<Rpp8u*>(output) + dstDescPtr->offsetInBytes);
        for (int i = 0; i < oBufferSize; i++) {
            *outputTemp = static_cast<Rpp8u>(
                validate_pixel_range(static_cast<float>(*outputf16Temp) * invConversionFactor));
            outputf16Temp++;
            outputTemp++;
        }
    } else if ((BitDepthTestMode == F32_TO_F32) || (BitDepthTestMode == U8_TO_F32)) {
        Rpp8u* outputTemp = outputu8 + dstDescPtr->offsetInBytes;
        Rpp32f* outputf32Temp =
            reinterpret_cast<Rpp32f*>(static_cast<Rpp8u*>(output) + dstDescPtr->offsetInBytes);
        for (int i = 0; i < oBufferSize; i++) {
            *outputTemp =
                static_cast<Rpp8u>(validate_pixel_range(*outputf32Temp * invConversionFactor));
            outputf32Temp++;
            outputTemp++;
        }
    } else if ((BitDepthTestMode == I8_TO_I8) || (BitDepthTestMode == U8_TO_I8)) {
        Rpp8u* outputTemp = outputu8 + dstDescPtr->offsetInBytes;
        Rpp8s* outputi8Temp = static_cast<Rpp8s*>(output) + dstDescPtr->offsetInBytes;
        for (int i = 0; i < oBufferSize; i++) {
            *outputTemp = static_cast<Rpp8u>(
                validate_pixel_range((static_cast<Rpp32s>(*outputi8Temp) + 128)));
            outputi8Temp++;
            outputTemp++;
        }
    }
}

// updates dstImg sizes
inline void update_dst_sizes_with_roi(RpptROI* roiTensorPtrSrc, RpptImagePatchPtr dstImageSize,
                                      RpptRoiType roiType, int batchSize) {
    if (roiType == RpptRoiType::XYWH) {
        for (int i = 0; i < batchSize; i++) {
            dstImageSize[i].width = roiTensorPtrSrc[i].xywhROI.roiWidth;
            dstImageSize[i].height = roiTensorPtrSrc[i].xywhROI.roiHeight;
        }
    } else if (roiType == RpptRoiType::LTRB) {
        for (int i = 0; i < batchSize; i++) {
            dstImageSize[i].width =
                roiTensorPtrSrc[i].ltrbROI.rb.x - roiTensorPtrSrc[i].ltrbROI.lt.x + 1;
            dstImageSize[i].height =
                roiTensorPtrSrc[i].ltrbROI.rb.y - roiTensorPtrSrc[i].ltrbROI.lt.y + 1;
        }
    }
}

// converts image data from PLN3 to PKD3
inline void convert_pln3_to_pkd3(Rpp8u* output, RpptDescPtr descPtr) {
    unsigned long long bufferSize =
        ((unsigned long long)descPtr->h * (unsigned long long)descPtr->w *
         (unsigned long long)descPtr->c * (unsigned long long)descPtr->n) +
        descPtr->offsetInBytes;
    Rpp8u* outputCopy = (Rpp8u*)calloc(bufferSize, sizeof(Rpp8u));
    memcpy(outputCopy, output, bufferSize * sizeof(Rpp8u));

    Rpp8u* outputCopyTemp;
    outputCopyTemp = outputCopy + descPtr->offsetInBytes;

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(descPtr->n)
    for (int count = 0; count < descPtr->n; count++) {
        Rpp8u *outputCopyTempR, *outputCopyTempG, *outputCopyTempB;
        outputCopyTempR = outputCopyTemp + count * descPtr->strides.nStride;
        outputCopyTempG = outputCopyTempR + descPtr->strides.cStride;
        outputCopyTempB = outputCopyTempG + descPtr->strides.cStride;
        Rpp8u* outputTemp = output + descPtr->offsetInBytes + count * descPtr->strides.nStride;

        for (int i = 0; i < descPtr->h; i++) {
            for (int j = 0; j < descPtr->w; j++) {
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
inline void convert_pkd3_to_pln3(Rpp8u* input, RpptDescPtr descPtr) {
    unsigned long long bufferSize =
        ((unsigned long long)descPtr->h * (unsigned long long)descPtr->w *
         (unsigned long long)descPtr->c * (unsigned long long)descPtr->n) +
        descPtr->offsetInBytes;
    Rpp8u* inputCopy = (Rpp8u*)calloc(bufferSize, sizeof(Rpp8u));
    memcpy(inputCopy, input, bufferSize * sizeof(Rpp8u));

    Rpp8u* inputTemp = input + descPtr->offsetInBytes;

    omp_set_dynamic(0);
#pragma omp parallel for num_threads(descPtr->n)
    for (int count = 0; count < descPtr->n; count++) {
        Rpp8u *inputTempR, *inputTempG, *inputTempB;
        inputTempR = inputTemp + count * descPtr->strides.nStride;
        inputTempG = inputTempR + descPtr->strides.cStride;
        inputTempB = inputTempG + descPtr->strides.cStride;
        Rpp8u* inputCopyTemp =
            inputCopy + descPtr->offsetInBytes + count * descPtr->strides.nStride;

        for (int i = 0; i < descPtr->h; i++) {
            for (int j = 0; j < descPtr->w; j++) {
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

#if defined(RPP_TEST_SUITE_HAVE_OPENCV) && RPP_TEST_SUITE_HAVE_OPENCV
// Read a batch of images using the OpenCV library
inline void read_image_batch_opencv(Rpp8u* input, RpptDescPtr descPtr,
                                    vector<string>::const_iterator imagesNamesStart) {
    for (int i = 0; i < descPtr->n; i++) {
        Rpp8u* inputTemp = input + (i * descPtr->strides.nStride);
        string inputImagePath = *(imagesNamesStart + i);
        Mat image, imageBgr;
        if (descPtr->c == 3) {
            imageBgr = imread(inputImagePath, 1);
            cvtColor(imageBgr, image, COLOR_BGR2RGB);
        } else if (descPtr->c == 1)
            image = imread(inputImagePath, 0);

        int width = image.cols;
        int height = image.rows;
        Rpp32u elementsInRow = width * descPtr->c;
        Rpp8u* inputImage = image.data;
        for (int j = 0; j < height; j++) {
            memcpy(inputTemp, inputImage, elementsInRow * sizeof(Rpp8u));
            inputImage += elementsInRow;
            inputTemp += descPtr->w * descPtr->c;
            ;
        }
    }
}
#endif /* RPP_TEST_SUITE_HAVE_OPENCV */

// Read a batch of packed .rgb images (row-major pixels with 24-byte binary header).
// Supports both grayscale (channels=1) and RGB (channels=3) in unified format.
// Descriptor c=1 on RGB input (channels=3) converts RGB→gray using BT.601 luma formula.
// Descriptor c=3 on gray input (channels=1) expands gray→RGB by replicating to R=G=B.
inline void read_image_batch_packed(Rpp8u* input, RpptDescPtr descPtr,
                                    vector<string>::const_iterator imagesNamesStart) {
    for (int i = 0; i < descPtr->n; i++) {
        std::string mediaPath = *(imagesNamesStart + i);
        RpptTestSuiteInfoSidecar info;

        // Parse .rgb header
        if (!parse_image_file_header(mediaPath, info) || !info.has_channels || info.width <= 0 ||
            info.height <= 0) {
            std::cerr << "\nError: invalid or missing .rgb header: " << mediaPath;
            continue;
        }

        const int outC = (int)descPtr->c;
        const int inCh = info.channels;

        // Validate channel conversion support
        if (!((inCh == outC) || (outC == 1 && inCh == 3) || (outC == 3 && inCh == 1))) {
            std::cerr << "\nUnsupported channel conversion (" << inCh << " -> " << outC << ") for "
                      << mediaPath;
            continue;
        }

        const size_t pixelBytes = (size_t)info.width * (size_t)info.height * (size_t)inCh;

        FILE* fp = fopen(mediaPath.c_str(), "rb");
        if (!fp) {
            std::cerr << "\nUnable to open file: " << mediaPath;
            continue;
        }

        // Skip 24-byte header
        if (fseek(fp, RpptImageFileHeader::HEADER_SIZE, SEEK_SET) != 0) {
            std::cerr << "\nSeek error: " << mediaPath;
            fclose(fp);
            continue;
        }

        std::vector<Rpp8u> fileBuf(pixelBytes);
        size_t nread = fread(fileBuf.data(), 1, pixelBytes, fp);
        fclose(fp);

        if (nread != pixelBytes) {
            std::cerr << "\nRead size mismatch (expected " << pixelBytes << ", got " << nread
                      << "): " << mediaPath;
            continue;
        }
        const int w = info.width;
        const int h = info.height;
        Rpp8u* inputTemp = input + descPtr->offsetInBytes + (i * descPtr->strides.nStride);

        if (inCh == outC) {
            const int elementsInRow = w * inCh;
            for (int j = 0; j < h; j++) {
                memcpy(inputTemp, fileBuf.data() + (size_t)j * (size_t)elementsInRow,
                       (size_t)elementsInRow * sizeof(Rpp8u));
                inputTemp += descPtr->w * (Rpp32u)outC;
            }
        } else if (outC == 1 && inCh == 3) {
            // RGB→gray using BT.601 luma: Y = 0.299R + 0.587G + 0.114B
            // Integer approximation: Y = (77R + 150G + 29B + 128) >> 8
            // Weights sum to 256; +128 for rounding; result guaranteed ∈ [0,255] for inputs ∈
            // [0,255]
            for (int j = 0; j < h; j++) {
                const Rpp8u* srcRow = fileBuf.data() + (size_t)j * (size_t)w * 3u;
                for (int x = 0; x < w; x++) {
                    int r = (int)srcRow[3 * x];
                    int g = (int)srcRow[3 * x + 1];
                    int b = (int)srcRow[3 * x + 2];
                    int yv = (77 * r + 150 * g + 29 * b + 128) >> 8;
                    inputTemp[x] = (Rpp8u)yv;
                }
                inputTemp += descPtr->w;
            }
        } else if (outC == 3 && inCh == 1) {
            for (int j = 0; j < h; j++) {
                const Rpp8u* srcRow = fileBuf.data() + (size_t)j * (size_t)w;
                for (int x = 0; x < w; x++) {
                    Rpp8u yv = srcRow[x];
                    inputTemp[3 * x] = yv;
                    inputTemp[3 * x + 1] = yv;
                    inputTemp[3 * x + 2] = yv;
                }
                inputTemp += descPtr->w * 3u;
            }
        }
    }
}

#if defined(RPP_TEST_SUITE_HAVE_OPENCV) && RPP_TEST_SUITE_HAVE_OPENCV
// Write a batch of images using the OpenCV library
inline void write_image_batch_opencv(string outputFolder, Rpp8u* output, RpptDescPtr dstDescPtr,
                                     vector<string>::const_iterator imagesNamesStart,
                                     RpptImagePatch* dstImgSizes, int maxImageDump) {
    // create output folder
    mkdir(outputFolder.c_str(), 0700);
    outputFolder += "/";
    static int cnt = 1;
    static int imageCnt = 0;

    Rpp32u elementsInRowMax = dstDescPtr->w * dstDescPtr->c;
    Rpp8u* offsettedOutput = output + dstDescPtr->offsetInBytes;
    for (int j = 0; (j < dstDescPtr->n) && (imageCnt < maxImageDump); j++, imageCnt++) {
        Rpp32u height = dstImgSizes[j].height;
        Rpp32u width = dstImgSizes[j].width;
        Rpp32u elementsInRow = width * dstDescPtr->c;
        Rpp32u outputSize = height * width * dstDescPtr->c;
        // When kernel writes with per-image pitch (e.g. yuv_to_rgb), row stride = width*c; else use
        // descriptor stride
        Rpp32u rowStrideBytes = (width < (Rpp32u)dstDescPtr->w) ? elementsInRow : elementsInRowMax;
        Rpp8u* tempOutput = (Rpp8u*)calloc(outputSize, sizeof(Rpp8u));
        Rpp8u* tempOutputRow = tempOutput;
        Rpp8u* outputRow = offsettedOutput + j * dstDescPtr->strides.nStride;
        for (int k = 0; k < height; k++) {
            memcpy(tempOutputRow, outputRow, elementsInRow * sizeof(Rpp8u));
            tempOutputRow += elementsInRow;
            outputRow += rowStrideBytes;
        }
        string outputImagePath = outputFolder + *(imagesNamesStart + j);
        // OpenCV imwrite does not support .yuv, .rgb, .raw; use .png for these dumps
        if (outputImagePath.size() >= 4 &&
            outputImagePath.compare(outputImagePath.size() - 4, 4, ".yuv") == 0) {
            outputImagePath = outputImagePath.substr(0, outputImagePath.size() - 4) + ".png";
        } else if (outputImagePath.size() >= 4 &&
                   outputImagePath.compare(outputImagePath.size() - 4, 4, ".raw") == 0) {
            outputImagePath = outputImagePath.substr(0, outputImagePath.size() - 4) + ".png";
        } else if (outputImagePath.size() >= 4 &&
                   outputImagePath.compare(outputImagePath.size() - 4, 4, ".rgb") == 0) {
            outputImagePath = outputImagePath.substr(0, outputImagePath.size() - 4) + ".png";
        }
        Mat matOutputImage, matOutputImageRgb;
        if (dstDescPtr->c == 1)
            matOutputImage = Mat(height, width, CV_8UC1, tempOutput);
        else if (dstDescPtr->c == 2)
            matOutputImage = Mat(height, width, CV_8UC2, tempOutput);
        else if (dstDescPtr->c == 3) {
            matOutputImageRgb = Mat(height, width, CV_8UC3, tempOutput);
            cvtColor(matOutputImageRgb, matOutputImage, COLOR_RGB2BGR);
        }

        fs::path pathObj(outputImagePath);
        if (fs::exists(pathObj)) {
            std::string outPath = outputImagePath.substr(0, outputImagePath.find_last_of('.')) +
                                  "_" + to_string(cnt) +
                                  outputImagePath.substr(outputImagePath.find_last_of('.'));
            imwrite(outPath, matOutputImage);
            cnt++;
        } else
            imwrite(outputImagePath, matOutputImage);
        free(tempOutput);
    }
}
#endif /* RPP_TEST_SUITE_HAVE_OPENCV */

// compares the output of PKD3-PKD3 and PLN1-PLN1 variants
void compare_outputs_pkd_and_pln1(Rpp8u* output, Rpp8u* refOutput, RpptDescPtr dstDescPtr,
                                  RpptImagePatch* dstImgSizes, int refOutputHeight,
                                  int refOutputWidth, int refOutputSize, int& fileMatch) {
    Rpp8u *rowTemp, *rowTempRef, *outVal, *outRefVal, *outputTemp, *outputTempRef;
    for (int imageCnt = 0; imageCnt < dstDescPtr->n; imageCnt++) {
        outputTemp = output + imageCnt * dstDescPtr->strides.nStride;
        outputTempRef = refOutput + imageCnt * refOutputSize;
        int height = dstImgSizes[imageCnt].height;
        int width = dstImgSizes[imageCnt].width * dstDescPtr->c;
        int matchedIdx = 0;
        int refOutputHstride = refOutputWidth * dstDescPtr->c;

        for (int i = 0; i < height; i++) {
            rowTemp = outputTemp + i * dstDescPtr->strides.hStride;
            rowTempRef = outputTempRef + i * refOutputHstride;
            for (int j = 0; j < width; j++) {
                outVal = rowTemp + j;
                outRefVal = rowTempRef + j;
                int diff = abs(*outVal - *outRefVal);
                if (diff <= CUTOFF) matchedIdx++;
            }
        }
        if (matchedIdx == (height * width) && matchedIdx != 0) fileMatch++;
    }
}

// compares the output of PKD3-PKD3 and PLN1-PLN1 variants
void compare_outputs_pkd_and_pln1(Rpp32f* output, Rpp32f* refOutput, RpptDescPtr dstDescPtr,
                                  RpptImagePatch* dstImgSizes, int refOutputHeight,
                                  int refOutputWidth, int refOutputSize, int& fileMatch,
                                  int testCase) {
    Rpp32f *rowTemp, *rowTempRef, *outVal, *outRefVal, *outputTemp, *outputTempRef;
    Rpp32f cutoff = ((testCase == LENS_CORRECTION) || (testCase == SOBEL_FILTER)) ? 1e-4 : 1e-5;
    for (int imageCnt = 0; imageCnt < dstDescPtr->n; imageCnt++) {
        outputTemp = output + imageCnt * dstDescPtr->strides.nStride;
        outputTempRef = refOutput + imageCnt * refOutputSize;
        int height = dstImgSizes[imageCnt].height;
        int width = dstImgSizes[imageCnt].width * dstDescPtr->c;
        int matchedIdx = 0;
        int refOutputHstride = refOutputWidth * dstDescPtr->c;
        for (int i = 0; i < height; i++) {
            rowTemp = outputTemp + i * dstDescPtr->strides.hStride;
            rowTempRef = outputTempRef + i * refOutputHstride;
            for (int j = 0; j < width; j++) {
                outVal = rowTemp + j;
                outRefVal = rowTempRef + j;
                Rpp32f diff = abs(*outVal - *outRefVal);
                if (diff <= cutoff) matchedIdx++;
            }
        }
        if (matchedIdx == (height * width) && matchedIdx != 0) fileMatch++;
    }
}

// compares the output of PLN3-PLN3 variants.This function compares the output buffer of pln3 format
// with its reference output in pkd3 format.
void compare_outputs_pln3(Rpp8u* output, Rpp8u* refOutput, RpptDescPtr dstDescPtr,
                          RpptImagePatch* dstImgSizes, int refOutputHeight, int refOutputWidth,
                          int refOutputSize, int& fileMatch) {
    Rpp8u *rowTemp, *rowTempRef, *outVal, *outRefVal, *outputTemp, *outputTempRef, *outputTempChn,
        *outputTempRefChn;
    for (int imageCnt = 0; imageCnt < dstDescPtr->n; imageCnt++) {
        outputTemp = output + imageCnt * dstDescPtr->strides.nStride;
        outputTempRef = refOutput + imageCnt * refOutputSize;
        int height = dstImgSizes[imageCnt].height;
        int width = dstImgSizes[imageCnt].width;
        int matchedIdx = 0;
        int refOutputHstride = refOutputWidth * dstDescPtr->c;

        for (int c = 0; c < dstDescPtr->c; c++) {
            outputTempChn = outputTemp + c * dstDescPtr->strides.cStride;
            outputTempRefChn = outputTempRef + c;
            for (int i = 0; i < height; i++) {
                rowTemp = outputTempChn + i * dstDescPtr->strides.hStride;
                rowTempRef = outputTempRefChn + i * refOutputHstride;
                for (int j = 0; j < width; j++) {
                    outVal = rowTemp + j;
                    outRefVal = rowTempRef + j * 3;
                    int diff = abs(*outVal - *outRefVal);
                    if (diff <= CUTOFF) matchedIdx++;
                }
            }
        }
        if (matchedIdx == (height * width * dstDescPtr->c) && matchedIdx != 0) fileMatch++;
    }
}

// compares the output of PLN3-PLN3 variants.This function compares the output buffer of pln3 format
// with its reference output in pkd3 format.
void compare_outputs_pln3(Rpp32f* output, Rpp32f* refOutput, RpptDescPtr dstDescPtr,
                          RpptImagePatch* dstImgSizes, int refOutputHeight, int refOutputWidth,
                          int refOutputSize, int& fileMatch, int testCase) {
    Rpp32f *rowTemp, *rowTempRef, *outVal, *outRefVal, *outputTemp, *outputTempRef, *outputTempChn,
        *outputTempRefChn;
    Rpp32f cutoff = ((testCase == LENS_CORRECTION) || (testCase == SOBEL_FILTER)) ? 1e-4 : 1e-5;
    for (int imageCnt = 0; imageCnt < dstDescPtr->n; imageCnt++) {
        outputTemp = output + imageCnt * dstDescPtr->strides.nStride;
        outputTempRef = refOutput + imageCnt * refOutputSize;
        int height = dstImgSizes[imageCnt].height;
        int width = dstImgSizes[imageCnt].width;
        int matchedIdx = 0;
        int refOutputHstride = refOutputWidth * dstDescPtr->c;

        for (int c = 0; c < dstDescPtr->c; c++) {
            outputTempChn = outputTemp + c * dstDescPtr->strides.cStride;
            outputTempRefChn = outputTempRef + c;
            for (int i = 0; i < height; i++) {
                rowTemp = outputTempChn + i * dstDescPtr->strides.hStride;
                rowTempRef = outputTempRefChn + i * refOutputHstride;
                for (int j = 0; j < width; j++) {
                    outVal = rowTemp + j;
                    outRefVal = rowTempRef + j * 3;
                    Rpp32f diff = abs(*outVal - *outRefVal);
                    if (diff <= cutoff) matchedIdx++;
                }
            }
        }
        if (matchedIdx == (height * width * dstDescPtr->c) && matchedIdx != 0) fileMatch++;
    }
}

inline void compare_output(void* output, string funcName, RpptDescPtr srcDescPtr,
                           RpptDescPtr dstDescPtr, RpptImagePatch* dstImgSizes, int noOfImages,
                           string interpolationTypeName, string noiseTypeName,
                           string kernelSizeAndGradientName, int additionalParam, int testCase,
                           string dst, string scriptPath,
                           const vector<string>* yuvImagePaths = nullptr) {
    string func = funcName;
    string refFile = "";
    int refOutputWidth, refOutputHeight;
    if (testCase == LENS_CORRECTION) {
        refOutputWidth = ((LENS_CORRECTION_GOLDEN_OUTPUT_MAX_WIDTH / 8) * 8) +
                         8;  // obtain next multiple of 8 after GOLDEN_OUTPUT_MAX_WIDTH
        refOutputHeight = LENS_CORRECTION_GOLDEN_OUTPUT_MAX_HEIGHT;
    } else if (testCase == YUV_TO_RGB || testCase == YUV_TO_RGB_CUBIC_V ||
               testCase == YUV_TO_RGB_LINEAR_V) {
        refOutputWidth = dstDescPtr->w;
        refOutputHeight = dstDescPtr->h;
    } else {
        refOutputWidth = ((GOLDEN_OUTPUT_MAX_WIDTH / 8) * 8) +
                         8;  // obtain next multiple of 8 after GOLDEN_OUTPUT_MAX_WIDTH
        refOutputHeight = GOLDEN_OUTPUT_MAX_HEIGHT;
    }
    int refOutputSize = refOutputHeight * refOutputWidth * dstDescPtr->c;
    Rpp64u binOutputSize = (Rpp64u)refOutputHeight * refOutputWidth * dstDescPtr->n * 4;
    int pln1RefStride = refOutputHeight * refOutputWidth * dstDescPtr->n * 3;

    string dataType[4] = {"_u8", "_f32", "_f16", "_i8"};

    if (srcDescPtr->dataType == dstDescPtr->dataType)
        func += dataType[srcDescPtr->dataType];
    else {
        func = func + dataType[srcDescPtr->dataType] + "_to" + dataType[dstDescPtr->dataType];
    }

    // Add layout suffixes to func (for test display name)
    if (srcDescPtr->layout == RpptLayout::NHWC)
        func += "_PKD3";
    else {
        if (srcDescPtr->c == 3)
            func += "_PLN3";
        else
            func += "_PLN1";
    }
    if (dstDescPtr->layout == RpptLayout::NHWC)
        func += "_to_PKD3";
    else {
        if (dstDescPtr->c == 3)
            func += "_to_PLN3";
        else
            func += "_to_PLN1";
    }

    // Initialize binFile
    // For sobel/color_to_greyscale: PKD3/PLN3 share one file, PLN1 has separate file
    // For other functions: use funcName without layout suffixes (combined PKD3+PLN1 file)
    std::string binFile = funcName;
    if (testCase == SOBEL_FILTER || testCase == COLOR_TO_GREYSCALE) {
        // Add PLN1 layout suffix only for grayscale input (c=1)
        if (srcDescPtr->c == 1) binFile += "_PLN1_to_PLN1";
        pln1RefStride = 0;
    }

    // Add datatype to binFile
    if (srcDescPtr->dataType == dstDescPtr->dataType)
        binFile += dataType[srcDescPtr->dataType];
    else {
        binFile = binFile + dataType[srcDescPtr->dataType];
        binFile.resize(binFile.size() - 1);
        binFile += dataType[dstDescPtr->dataType];
    }
    // Remove trailing underscore from datatype to match actual filename format
    if (!binFile.empty() && binFile.back() == '_') binFile.pop_back();

    if (testCase == RESIZE || testCase == ROTATE || testCase == WARP_AFFINE ||
        testCase == WARP_PERSPECTIVE || testCase == REMAP) {
        func += "_interpolationType" + interpolationTypeName;
        binFile += "_interpolationType" + interpolationTypeName;
    } else if (testCase == NOISE) {
        func += "_noiseType" + noiseTypeName;
        binFile += "_noiseType" + noiseTypeName;
    } else if (testCase == ERODE || testCase == DILATE || testCase == BOX_FILTER ||
               testCase == MEDIAN_FILTER || testCase == GAUSSIAN_FILTER || testCase == EMBOSS) {
        func += "_kernelSize" + std::to_string(additionalParam);
        binFile += "_kernelSize" + std::to_string(additionalParam);
    } else if (testCase == CHANNEL_PERMUTE) {
        func += "_permOrder" + std::to_string(additionalParam);
        binFile += "_permOrder" + std::to_string(additionalParam);
    } else if (testCase == SOBEL_FILTER) {
        Rpp32u kernelSize, gradientType;
        get_kernel_size_and_gradient_type(additionalParam, kernelSize, gradientType);

        func += kernelSizeAndGradientName;
        std::string gradientName;
        switch (gradientType) {
            case 0:
                gradientName = "_gradientX";
                break;
            case 1:
                gradientName = "_gradientY";
                break;
            case 2:
                gradientName = "_gradientXY";
                break;
            default:
                gradientName = "";
                break;
        }
        binFile += "_kernelSize" + std::to_string(kernelSize) + gradientName;
    }

    refFile = scriptPath + "/../REFERENCE_OUTPUT/" + funcName + "/" + binFile + ".bin";
    int fileMatch = 0;
    if ((testCase == YUV_TO_RGB || testCase == YUV_TO_RGB_CUBIC_V ||
         testCase == YUV_TO_RGB_LINEAR_V) &&
        yuvImagePaths != nullptr && (int)yuvImagePaths->size() >= dstDescPtr->n &&
        dstDescPtr->dataType == RpptDataType::U8) {
        // cubic_v uses CUTOFF=3 due to GPU FMA rounding in the 4-tap chroma accumulation
        int yuvCutoff = (testCase == YUV_TO_RGB_CUBIC_V) ? 3 : CUTOFF;
        std::string refDir = scriptPath + "/../REFERENCE_OUTPUT/" + augmentationMap[testCase] + "/";
        for (int imageCnt = 0; imageCnt < dstDescPtr->n; imageCnt++) {
            std::string refPath =
                refDir + get_yuv_ref_basename((*yuvImagePaths)[imageCnt]) + ".rgb";
            int imgH = dstImgSizes[imageCnt].height;
            int imgW = dstImgSizes[imageCnt].width;
            int imgSize = imgH * imgW * dstDescPtr->c;
            Rpp8u* refBuf = (Rpp8u*)malloc((size_t)imgSize * sizeof(Rpp8u));
            FILE* rfp = fopen(refPath.c_str(), "rb");
            if (rfp) {
                fread(refBuf, 1, (size_t)imgSize, rfp);
                fclose(rfp);
                Rpp8u* outSlice = (Rpp8u*)output + imageCnt * dstDescPtr->strides.nStride;
                int rowStride =
                    imgW * (int)dstDescPtr->c;  // yuv_to_rgb writes with per-image pitch
                int matchedIdx = 0;
                for (int i = 0; i < imgH; i++) {
                    Rpp8u* outRow = outSlice + i * rowStride;
                    Rpp8u* refRow = refBuf + i * imgW * (int)dstDescPtr->c;
                    for (int j = 0; j < imgW * (int)dstDescPtr->c; j++)
                        if (abs((int)outRow[j] - (int)refRow[j]) <= yuvCutoff) matchedIdx++;
                }
                if (matchedIdx == imgSize && matchedIdx != 0) fileMatch++;
            } else
                std::cerr << "\nQA yuv_to_rgb: missing reference file (expected packed RGB24, same "
                             "basename as .yuv): "
                          << refPath << std::endl;
            free(refBuf);
        }
    } else if (dstDescPtr->dataType == RpptDataType::U8) {
        // YUV_TO_RGB variants use per-image .rgb refs only; no single ref file
        if (testCase != YUV_TO_RGB && testCase != YUV_TO_RGB_CUBIC_V &&
            testCase != YUV_TO_RGB_LINEAR_V) {
            Rpp8u* binaryContent = (Rpp8u*)malloc(binOutputSize * sizeof(Rpp8u));
            read_bin_file(refFile, binaryContent);

            if (dstDescPtr->layout == RpptLayout::NHWC)
                compare_outputs_pkd_and_pln1((Rpp8u*)output, binaryContent, dstDescPtr, dstImgSizes,
                                             refOutputHeight, refOutputWidth, refOutputSize,
                                             fileMatch);
            else if (dstDescPtr->layout == RpptLayout::NCHW && dstDescPtr->c == 3)
                compare_outputs_pln3((Rpp8u*)output, binaryContent, dstDescPtr, dstImgSizes,
                                     refOutputHeight, refOutputWidth, refOutputSize, fileMatch);
            else
                compare_outputs_pkd_and_pln1((Rpp8u*)output, binaryContent + pln1RefStride,
                                             dstDescPtr, dstImgSizes, refOutputHeight,
                                             refOutputWidth, refOutputSize, fileMatch);
            free(binaryContent);
        }
    } else {
        Rpp32f* binaryContent = (Rpp32f*)malloc(binOutputSize * sizeof(Rpp32f));
        read_bin_file(refFile, binaryContent);

        if (dstDescPtr->layout == RpptLayout::NHWC)
            compare_outputs_pkd_and_pln1((Rpp32f*)output, binaryContent, dstDescPtr, dstImgSizes,
                                         refOutputHeight, refOutputWidth, refOutputSize, fileMatch,
                                         testCase);
        else if (dstDescPtr->layout == RpptLayout::NCHW && dstDescPtr->c == 3)
            compare_outputs_pln3((Rpp32f*)output, binaryContent, dstDescPtr, dstImgSizes,
                                 refOutputHeight, refOutputWidth, refOutputSize, fileMatch,
                                 testCase);
        else
            compare_outputs_pkd_and_pln1((Rpp32f*)output, binaryContent + pln1RefStride, dstDescPtr,
                                         dstImgSizes, refOutputHeight, refOutputWidth,
                                         refOutputSize, fileMatch, testCase);
        free(binaryContent);
    }

    std::cout << std::endl << "\nResults for " << func << " :" << std::endl;
    std::string status = func + ": ";
    if (fileMatch == dstDescPtr->n) {
        std::cout << "PASSED!";
        status += "PASSED";
    } else {
        std::cout << "FAILED! " << fileMatch << "/" << dstDescPtr->n
                  << " outputs are matching with reference outputs";
        status += "FAILED";
    }

    // Append the QA results to file
    std::string qaResultsPath = dst + "/QA_results.txt";
    std::ofstream qaResults(qaResultsPath, ios_base::app);
    if (qaResults.is_open()) {
        qaResults << status << std::endl;
        qaResults.close();
    }
}

// compares reduction type functions outputs
template <typename T>
inline void compare_reduction_output(T* output, string funcName, RpptDescPtr srcDescPtr,
                                     int testCase, string dst, string scriptPath) {
    string func = funcName;
    switch (srcDescPtr->dataType) {
        case RpptDataType::U8:
            func += "_u8_";
            break;
        case RpptDataType::F16:
            func += "_f16_";
            break;
        case RpptDataType::F32:
            func += "_f32_";
            break;
        case RpptDataType::I8:
            func += "_i8_";
            break;
        default:
            func += "_unknown_";
            break;
    }

    if (srcDescPtr->layout == RpptLayout::NHWC)
        func += "Tensor_PKD3";
    else {
        if (srcDescPtr->c == 3)
            func += "Tensor_PLN3";
        else
            func += "Tensor_PLN1";
    }

    int fileMatch = 0;
    int matched_values = 0;

    int inputBitDepth = srcDescPtr->dataType;
    T* refOutput;
    int numChannels = (srcDescPtr->c == 1) ? 1 : 3;
    if (inputBitDepth == RpptDataType::F32) {
        if (testCase == TENSOR_MIN)
            refOutput = reinterpret_cast<T*>(TensorMinReferenceOutputs_F32[numChannels].data());
        else if (testCase == TENSOR_MAX)
            refOutput = reinterpret_cast<T*>(TensorMaxReferenceOutputs_F32[numChannels].data());
        else if (testCase == TENSOR_SUM)
            refOutput = reinterpret_cast<T*>(TensorSumReferenceOutputs_F32[numChannels].data());
        else if (testCase == TENSOR_MEAN)
            refOutput = reinterpret_cast<T*>(TensorMeanReferenceOutputs_F32[numChannels].data());
        else if (testCase == TENSOR_STDDEV)
            refOutput = reinterpret_cast<T*>(TensorStddevReferenceOutputs_F32[numChannels].data());
    } else if (inputBitDepth == RpptDataType::U8) {
        if (testCase == TENSOR_MIN)
            refOutput = reinterpret_cast<T*>(TensorMinReferenceOutputs_U8[numChannels].data());
        else if (testCase == TENSOR_MAX)
            refOutput = reinterpret_cast<T*>(TensorMaxReferenceOutputs_U8[numChannels].data());
        else if (testCase == TENSOR_SUM)
            refOutput = reinterpret_cast<T*>(TensorSumReferenceOutputs_U8[numChannels].data());
        else if (testCase == TENSOR_MEAN)
            refOutput = reinterpret_cast<T*>(TensorMeanReferenceOutputs_U8[numChannels].data());
        else if (testCase == TENSOR_STDDEV)
            refOutput = reinterpret_cast<T*>(TensorStddevReferenceOutputs_U8[numChannels].data());
    }

    if (srcDescPtr->c == 1) {
        for (int i = 0; i < srcDescPtr->n; i++) {
            int diff = abs(static_cast<int>(output[i] - refOutput[i]));
            if (diff <= CUTOFF) fileMatch++;
        }
    } else {
        for (int i = 0; i < srcDescPtr->n; i++) {
            matched_values = 0;
            for (int j = 0; j < 4; j++) {
                int diff = abs(static_cast<int>(output[(i * 4) + j] - refOutput[(i * 4) + j]));
                if (diff <= CUTOFF) matched_values++;
            }
            if (matched_values == 4) fileMatch++;
        }
    }

    std::cout << std::endl << "Results for " << func << " :" << std::endl;
    std::string status = func + ": ";
    if (fileMatch == srcDescPtr->n) {
        std::cout << "PASSED!" << std::endl;
        status += "PASSED";
    } else {
        std::cout << "FAILED! " << fileMatch << "/" << srcDescPtr->n
                  << " outputs are matching with reference outputs" << std::endl;
        status += "FAILED";
    }

    // Append the QA results to file
    std::string qaResultsPath = dst + "/QA_results.txt";
    std::ofstream qaResults(qaResultsPath, ios_base::app);
    if (qaResults.is_open()) {
        qaResults << status << std::endl;
        qaResults.close();
    }
}

// print array of any bit depth for specified length
template <typename T>
inline void print_array(T* src, Rpp32u length, Rpp32u precision) {
    for (int i = 0; i < length; i++)
        std::cout << " " << std::fixed << std::setprecision(precision)
                  << static_cast<Rpp32f>(src[i]) << " ";
}

// Used to randomly swap values present in array of size n
inline void randomize(unsigned int arr[], unsigned int n) {
    // Use a different seed value each time
    srand(time(NULL));
    for (unsigned int i = n - 1; i > 0; i--) {
        // Pick a random index from 0 to i
        unsigned int j = rand() % (i + 1);
        std::swap(arr[i], arr[j]);
    }
}

// Generates a random value between given min and max values
int inline randrange(int min, int max) {
    if (max < 0) return -1;
    return rand() % (max - min + 1) + min;
}

// RICAP Input Crop Region initializer for QA testing and golden output match
void inline init_ricap_qa(int width, int height, int batchSize, Rpp32u* permutationTensor,
                          RpptROIPtr roiPtrInputCropRegion) {
    Rpp32u initialPermuteArray[batchSize], permutedArray[batchSize * 4];
    int part0Width = 40;   // Set part0 width around 1/3 of image width
    int part0Height = 72;  // Set part0 height around 1/2 of image height

    for (uint i = 0; i < batchSize; i++) initialPermuteArray[i] = i;

    for (int i = 0; i < 4; i++)
        memcpy(permutedArray + (batchSize * i), initialPermuteArray, batchSize * sizeof(Rpp32u));

    for (uint i = 0, j = 0; j < batchSize * 4; i++, j += 4) {
        permutationTensor[j] = permutedArray[i];
        permutationTensor[j + 1] = permutedArray[i + batchSize];
        permutationTensor[j + 2] = permutedArray[i + (batchSize * 2)];
        permutationTensor[j + 3] = permutedArray[i + (batchSize * 3)];
    }

    roiPtrInputCropRegion[0].xywhROI = {{width - part0Width, 0}, part0Width, part0Height};
    roiPtrInputCropRegion[1].xywhROI = {{part0Width, 0}, width - part0Width, part0Height};
    roiPtrInputCropRegion[2].xywhROI = {{0, part0Height}, part0Width, height - part0Height};
    roiPtrInputCropRegion[3].xywhROI = {{0, part0Height}, width - part0Width, height - part0Height};
}

// RICAP Input Crop Region initializer for unit and performance testing
void inline init_ricap(int width, int height, int batchSize, Rpp32u* permutationTensor,
                       RpptROIPtr roiPtrInputCropRegion) {
    Rpp32u initialPermuteArray[batchSize], permutedArray[batchSize * 4];

    for (uint i = 0; i < batchSize; i++) initialPermuteArray[i] = i;

    std::random_device rd;
    std::mt19937 gen(rd());  // Pseudo random number generator
    static std::uniform_real_distribution<double> unif(
        0.3, 0.7);  // Generates a uniform real distribution between 0.3 and 0.7
    double randVal = unif(gen);

    std::random_device rd1;
    std::mt19937 gen1(rd1());
    static std::uniform_real_distribution<double> unif1(0.3, 0.7);
    double randVal1 = unif1(gen1);

    for (int i = 0; i < 4; i++) {
        randomize(initialPermuteArray, batchSize);
        memcpy(permutedArray + (batchSize * i), initialPermuteArray, batchSize * sizeof(Rpp32u));
    }

    for (uint i = 0, j = 0; j < batchSize * 4; i++, j += 4) {
        permutationTensor[j] = permutedArray[i];
        permutationTensor[j + 1] = permutedArray[i + batchSize];
        permutationTensor[j + 2] = permutedArray[i + (batchSize * 2)];
        permutationTensor[j + 3] = permutedArray[i + (batchSize * 3)];
    }

    int part0Width = std::round(randVal * width);
    int part0Height = std::round(randVal1 * height);
    roiPtrInputCropRegion[0].xywhROI = {
        {randrange(0, width - part0Width - 8), randrange(0, height - part0Height)},
        part0Width,
        part0Height};  // Subtracted x coordinate by 8 to avoid corruption when HIP processes 8
                       // pixels at once
    roiPtrInputCropRegion[1].xywhROI = {
        {randrange(0, part0Width - 8), randrange(0, height - part0Height)},
        width - part0Width,
        part0Height};
    roiPtrInputCropRegion[2].xywhROI = {
        {randrange(0, width - part0Width - 8), randrange(0, part0Height)},
        part0Width,
        height - part0Height};
    roiPtrInputCropRegion[3].xywhROI = {{randrange(0, part0Width - 8), randrange(0, part0Height)},
                                        width - part0Width,
                                        height - part0Height};
}

void inline init_remap(RpptDescPtr tableDescPtr, RpptDescPtr srcDescPtr, RpptROIPtr roiTensorPtrSrc,
                       Rpp32f* rowRemapTable, Rpp32f* colRemapTable) {
    tableDescPtr->c = 1;
    tableDescPtr->strides.nStride = srcDescPtr->h * srcDescPtr->w;
    tableDescPtr->strides.hStride = srcDescPtr->w;
    tableDescPtr->strides.wStride = tableDescPtr->strides.cStride = 1;
    Rpp32u batchSize = srcDescPtr->n;

    for (Rpp32u count = 0; count < batchSize; count++) {
        Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
        rowRemapTableTemp = rowRemapTable + count * tableDescPtr->strides.nStride;
        colRemapTableTemp = colRemapTable + count * tableDescPtr->strides.nStride;
        Rpp32u halfWidth = roiTensorPtrSrc[count].xywhROI.roiWidth / 2;
        for (Rpp32u i = 0; i < roiTensorPtrSrc[count].xywhROI.roiHeight; i++) {
            Rpp32f *rowRemapTableTempRow, *colRemapTableTempRow;
            rowRemapTableTempRow = rowRemapTableTemp + i * tableDescPtr->strides.hStride;
            colRemapTableTempRow = colRemapTableTemp + i * tableDescPtr->strides.hStride;
            Rpp32u j = 0;
            for (; j < halfWidth; j++) {
                *rowRemapTableTempRow = i;
                *colRemapTableTempRow = halfWidth - j;

                rowRemapTableTempRow++;
                colRemapTableTempRow++;
            }
            for (; j < roiTensorPtrSrc[count].xywhROI.roiWidth; j++) {
                *rowRemapTableTempRow = i;
                *colRemapTableTempRow = j;

                rowRemapTableTempRow++;
                colRemapTableTempRow++;
            }
        }
    }
}

// initialize the roi, anchor and shape values required for slice
void init_slice(RpptGenericDescPtr descriptorPtr3D, RpptROIPtr roiPtrSrc, Rpp32u* roiTensor,
                Rpp32s* anchorTensor, Rpp32s* shapeTensor) {
    if (descriptorPtr3D->numDims == 4) {
        if (descriptorPtr3D->layout == RpptLayout::NCHW) {
            for (int i = 0; i < descriptorPtr3D->dims[0]; i++) {
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
        } else if (descriptorPtr3D->layout == RpptLayout::NHWC) {
            for (int i = 0; i < descriptorPtr3D->dims[0]; i++) {
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
    if (descriptorPtr3D->numDims == 3) {
        for (int i = 0; i < descriptorPtr3D->dims[0]; i++) {
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
void inline init_erase(int batchSize, int boxesInEachImage, Rpp32u* numOfBoxes,
                       RpptRoiLtrb* anchorBoxInfoTensor, RpptROIPtr roiTensorPtrSrc, int channels,
                       Rpp32f* colorBuffer, int BitDepthTestMode) {
    Rpp8u* colors8u = reinterpret_cast<Rpp8u*>(colorBuffer);
    Rpp16f* colors16f = reinterpret_cast<Rpp16f*>(colorBuffer);
    Rpp32f* colors32f = colorBuffer;
    Rpp8s* colors8s = reinterpret_cast<Rpp8s*>(colorBuffer);
    for (int i = 0; i < batchSize; i++) {
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

        if (channels == 3) {
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
            for (int j = 0; j < 9; j++) {
                if (BitDepthTestMode == U8_TO_U8)
                    colors8u[idx + j] = (Rpp8u)(colorBuffer[idx + j]);
                else if (BitDepthTestMode == F16_TO_F16)
                    colors16f[idx + j] = (Rpp16f)(colorBuffer[idx + j] * ONE_OVER_255);
                else if (BitDepthTestMode == F32_TO_F32)
                    colors32f[idx + j] = (Rpp32f)(colorBuffer[idx + j] * ONE_OVER_255);
                else if (BitDepthTestMode == I8_TO_I8)
                    colors8s[idx + j] = (Rpp8s)(colorBuffer[idx + j] - 128);
            }
        } else {
            int idx = boxesInEachImage * i;
            colorBuffer[idx] = 240;
            colorBuffer[idx + 1] = 120;
            colorBuffer[idx + 2] = 60;
            for (int j = 0; j < 3; j++) {
                if (BitDepthTestMode == U8_TO_U8)
                    colors8u[idx + j] = (Rpp8u)(colorBuffer[idx + j]);
                else if (BitDepthTestMode == F16_TO_F16)
                    colors16f[idx + j] = (Rpp16f)(colorBuffer[idx + j] * ONE_OVER_255);
                else if (BitDepthTestMode == F32_TO_F32)
                    colors32f[idx + j] = (Rpp32f)(colorBuffer[idx + j] * ONE_OVER_255);
                else if (BitDepthTestMode == I8_TO_I8)
                    colors8s[idx + j] = (Rpp8s)(colorBuffer[idx + j] - 128);
            }
        }
    }
}

void generate_channel_dropout_mask(Rpp8u* dropoutTensor, Rpp32f* dropoutProbability, int batchSize,
                                   int channels, int seed) {
    omp_set_dynamic(0);

#pragma omp parallel for num_threads(omp_get_max_threads())
    for (int batchCount = 0; batchCount < batchSize; batchCount++) {
        std::mt19937 rng(seed + batchCount);
        std::bernoulli_distribution keepDist(1.0f - dropoutProbability[batchCount]);
        Rpp8u* maskPtrTemp = dropoutTensor + (batchCount * channels);
        bool atLeastOne = false;

        for (int channel = 0; channel < channels; channel++) {
            maskPtrTemp[channel] = keepDist(rng);
            atLeastOne |= maskPtrTemp[channel];
        }

        if (!atLeastOne) maskPtrTemp[rng() % channels] = 1;
    }
}

// Dropout Region initializer for unit and performance testing
void inline init_cutout_dropout(int batchSize, int maxBoxesPerImage, Rpp32u* numOfBoxes,
                                RpptRoiLtrb* anchorBoxInfoTensor, RpptROIPtr roiTensorPtrSrc,
                                int channels, int BitDepthTestMode, int seed, int dropoutType,
                                void* colorBuffer = NULL) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> pos_ratio(0.1f, 0.9f);
    std::uniform_real_distribution<float> wh_ratio_cutout(0.4f, 0.6f);

    Rpp8u* colors8u = reinterpret_cast<Rpp8u*>(colorBuffer);
    Rpp16f* colors16f = reinterpret_cast<Rpp16f*>(colorBuffer);
    Rpp32f* colors32f = reinterpret_cast<Rpp32f*>(colorBuffer);
    Rpp8s* colors8s = reinterpret_cast<Rpp8s*>(colorBuffer);

    for (int i = 0; i < batchSize; i++) {
        const auto& roi = roiTensorPtrSrc[i].xywhROI;
        const float roiW = static_cast<float>(roi.roiWidth);
        const float roiH = static_cast<float>(roi.roiHeight);
        const float roiX = static_cast<float>(roi.xy.x);
        const float roiY = static_cast<float>(roi.xy.y);

        float boxW, boxH;

        float squareSize = wh_ratio_cutout(rng) * std::min(roiW, roiH);
        boxW = boxH = std::max(1.0f, squareSize);
        const float x_start = std::max(0.0f, std::min(pos_ratio(rng) * (roiW - boxW), roiW - boxW));
        const float y_start = std::max(0.0f, std::min(pos_ratio(rng) * (roiH - boxH), roiH - boxH));

        RpptRoiLtrb& box = anchorBoxInfoTensor[i * maxBoxesPerImage];
        box.lt.x = static_cast<Rpp32u>(roiX + x_start);
        box.lt.y = static_cast<Rpp32u>(roiY + y_start);
        box.rb.x = static_cast<Rpp32u>(roiX + x_start + boxW - 1.0f);
        box.rb.y = static_cast<Rpp32u>(roiY + y_start + boxH - 1.0f);

        if (colorBuffer != nullptr) {
            int colorOffset = (i * maxBoxesPerImage) * channels;
            Rpp32f dropoutColor = 0.0f;
            for (int c = 0; c < channels; c++) {
                if (BitDepthTestMode == U8_TO_U8)
                    colors8u[colorOffset + c] = (Rpp8u)dropoutColor;
                else if (BitDepthTestMode == F16_TO_F16)
                    colors16f[colorOffset + c] = (Rpp16f)(dropoutColor * ONE_OVER_255);
                else if (BitDepthTestMode == F32_TO_F32)
                    colors32f[colorOffset + c] = (Rpp32f)(dropoutColor);
                else if (BitDepthTestMode == I8_TO_I8)
                    colors8s[colorOffset + c] = (Rpp8s)(dropoutColor - 128);
            }
        }
        numOfBoxes[i] = 1;
    }
}

// Dropout Region initializer for unit and performance testing
void inline init_dropout_erase(int batchSize, int maxBoxesPerImage, Rpp32u* numOfBoxes,
                               RpptRoiLtrb* anchorBoxInfoTensor, RpptROIPtr roiTensorPtrSrc,
                               int channels, int BitDepthTestMode, int seed, int dropoutType) {
    // Initialize Random Number Generators
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> pos_ratio(0.1f, 0.9f);
    std::uniform_real_distribution<float> wh_ratio_cutout(0.4f, 0.6f);
    std::uniform_real_distribution<float> wh_ratio_random(0.1f, 0.5f);
    std::uniform_real_distribution<float> wh_ratio_coarse(0.05f, 0.1f);
    int minCoarseBoxes = std::max(0, std::min(5, maxBoxesPerImage));
    int maxCoarseBoxes = std::max(minCoarseBoxes, maxBoxesPerImage);
    std::uniform_int_distribution<int> coarse_box_count_dist(minCoarseBoxes, maxCoarseBoxes);

    for (int i = 0; i < batchSize; i++) {
        const auto& roi = roiTensorPtrSrc[i].xywhROI;
        const float roiW = static_cast<float>(roi.roiWidth);
        const float roiH = static_cast<float>(roi.roiHeight);
        const float roiX = static_cast<float>(roi.xy.x);
        const float roiY = static_cast<float>(roi.xy.y);

        int actualBoxCount = 1;
        std::uniform_real_distribution<float>* curr_wh_ratio = &wh_ratio_cutout;

        if (dropoutType == DROPOUT_RANDOM_ERASING)  // Random Erasing
            curr_wh_ratio = &wh_ratio_random;
        else if (dropoutType == DROPOUT_COARSE)  // Coarse Dropout
        {
            actualBoxCount = coarse_box_count_dist(rng);
            curr_wh_ratio = &wh_ratio_coarse;
        }

        int boxOffset = i * maxBoxesPerImage;
        int validBoxCount = 0;

        for (int b = 0; b < actualBoxCount; b++) {
            float boxW, boxH;

            boxW = std::max(1.0f, (*curr_wh_ratio)(rng)*roiW);
            boxH = std::max(1.0f, (*curr_wh_ratio)(rng)*roiH);

            const float x_slack = std::max(0.0f, roiW - boxW);
            const float y_slack = std::max(0.0f, roiH - boxH);

            const float x_start = std::max(0.0f, std::min(pos_ratio(rng) * x_slack, x_slack));
            const float y_start = std::max(0.0f, std::min(pos_ratio(rng) * y_slack, y_slack));

            // Set Bounding Box Coordinates
            RpptRoiLtrb& box = anchorBoxInfoTensor[boxOffset + b];
            box.lt.x = static_cast<Rpp32u>(roiX + x_start);
            box.lt.y = static_cast<Rpp32u>(roiY + y_start);
            Rpp32u boxWInt = static_cast<Rpp32u>(boxW);
            Rpp32u boxHInt = static_cast<Rpp32u>(boxH);
            box.rb.x = box.lt.x + boxWInt - 1;
            box.rb.y = box.lt.y + boxHInt - 1;
            validBoxCount++;
        }
        numOfBoxes[i] = validBoxCount;
    }
}

// Dropout Region initializer for unit and performance testing
void init_dropout_random_erase(Rpp32u batchSize, Rpp32u maxBoxesPerImage, Rpp32u* numOfBoxes,
                               RpptRoiLtrb* anchorBoxInfoTensor, RpptROIPtr roiTensorPtrSrc,
                               Rpp32u channels, Rpp8u BitDepthTestMode, int seed, Rpp8u dropoutType,
                               void* colorBuffer = NULL) {
    std::mt19937 rng(seed);
    std::mt19937 rng_noise(seed);

    if (dropoutType == DROPOUT_RANDOM_ERASING && colorBuffer != nullptr) {
        std::uniform_real_distribution<float> dist_f(0.0f, 1.0f);
        std::uniform_int_distribution<int> dist_i(0, 255);

        Rpp32u noiseSize =
            RANDOM_ERASE_NOISE_BUFFER_SIDE * RANDOM_ERASE_NOISE_BUFFER_SIDE * channels;

        if (BitDepthTestMode == U8_TO_U8)  // U8
            for (Rpp32u i = 0; i < noiseSize; i++)
                ((Rpp8u*)colorBuffer)[i] = (Rpp8u)dist_i(rng_noise);
        else if (BitDepthTestMode == F32_TO_F32)  // F32
            for (Rpp32u i = 0; i < noiseSize; i++)
                ((Rpp32f*)colorBuffer)[i] = (Rpp32f)dist_f(rng_noise);
        else if (BitDepthTestMode == F16_TO_F16)  // F16
            for (Rpp32u i = 0; i < noiseSize; i++)
                ((Rpp16f*)colorBuffer)[i] = (Rpp16f)dist_f(rng_noise);
        else if (BitDepthTestMode == I8_TO_I8)  // I8
            for (Rpp32u i = 0; i < noiseSize; i++)
                ((Rpp8s*)colorBuffer)[i] = (Rpp8s)(dist_i(rng_noise) - 128);
    }

    std::uniform_real_distribution<float> pos_ratio(0.1f, 0.9f);
    std::uniform_real_distribution<float> wh_ratio_cutout(0.4f, 0.6f);
    std::uniform_real_distribution<float> wh_ratio_random(0.1f, 0.5f);

    for (int i = 0; i < batchSize; i++) {
        const auto& roi = roiTensorPtrSrc[i].xywhROI;
        const float roiW = static_cast<float>(roi.roiWidth);
        const float roiH = static_cast<float>(roi.roiHeight);
        const float roiX = static_cast<float>(roi.xy.x);
        const float roiY = static_cast<float>(roi.xy.y);

        std::uniform_real_distribution<float>& curr_wh_ratio =
            (dropoutType == 3) ? wh_ratio_random : wh_ratio_cutout;

        float boxW, boxH;
        if (dropoutType == 1)  // Cutout: Perfect square
        {
            float squareSize = curr_wh_ratio(rng) * std::min(roiW, roiH);
            boxW = boxH = std::max(1.0f, squareSize);
        } else  // Dropout or Random Erase: Rectangular
        {
            boxW = std::max(1.0f, curr_wh_ratio(rng) * roiW);
            boxH = std::max(1.0f, curr_wh_ratio(rng) * roiH);
        }

        const float x_slack = std::max(0.0f, roiW - boxW);
        const float y_slack = std::max(0.0f, roiH - boxH);

        const float x_start = std::max(1.0f, std::min(pos_ratio(rng) * x_slack, x_slack));
        const float y_start = std::max(1.0f, std::min(pos_ratio(rng) * y_slack, y_slack));

        RpptRoiLtrb& box = anchorBoxInfoTensor[i * maxBoxesPerImage];
        box.lt.x = static_cast<Rpp32u>(roiX + x_start);
        box.lt.y = static_cast<Rpp32u>(roiY + y_start);
        box.rb.x = static_cast<Rpp32u>(roiX + x_start + boxW);
        box.rb.y = static_cast<Rpp32u>(roiY + y_start + boxH);

        if (dropoutType != 3 && colorBuffer != nullptr) {
            int colorOffset = (i * maxBoxesPerImage) * channels;
            Rpp32f dropoutColor = 0.0f;

            if (BitDepthTestMode == U8_TO_U8)
                for (int c = 0; c < channels; c++)
                    ((Rpp8u*)colorBuffer)[colorOffset + c] = (Rpp8u)dropoutColor;
            else if (BitDepthTestMode == F32_TO_F32)
                for (int c = 0; c < channels; c++)
                    ((Rpp32f*)colorBuffer)[colorOffset + c] = (Rpp32f)dropoutColor;
            else if (BitDepthTestMode == F16_TO_F16)
                for (int c = 0; c < channels; c++)
                    ((Rpp16f*)colorBuffer)[colorOffset + c] = (Rpp16f)(dropoutColor * ONE_OVER_255);
            else if (BitDepthTestMode == I8_TO_I8)
                for (int c = 0; c < channels; c++)
                    ((Rpp8s*)colorBuffer)[colorOffset + c] = (Rpp8s)(dropoutColor - 128);
        }

        // Only set numOfBoxes if it's provided
        if (numOfBoxes != nullptr) numOfBoxes[i] = 1;
    }
}

// Grid Dropout Region initializer for unit and performance testing
inline void init_grid_dropout(int batchCount, RpptRoiLtrb* anchorBoxInfoTensor,
                              RpptROIPtr roiTensorPtrSrc, Rpp32u gridH, Rpp32u gridW,
                              Rpp32u& maxHoleW, Rpp32u& maxHoleH, Rpp32f holeRatio, int seed) {
    std::mt19937 rng(seed);

    for (int i = 0; i < batchCount; i++) {
        Rpp32u roiW = roiTensorPtrSrc[i].xywhROI.roiWidth;
        Rpp32u roiH = roiTensorPtrSrc[i].xywhROI.roiHeight;
        Rpp32s x_base = roiTensorPtrSrc[i].xywhROI.xy.x;
        Rpp32s y_base = roiTensorPtrSrc[i].xywhROI.xy.y;

        Rpp32u cellW = std::max(1u, roiW / gridW);
        Rpp32u cellH = std::max(1u, roiH / gridH);
        Rpp32u holeW = std::max(1u, static_cast<Rpp32u>(cellW * holeRatio));
        Rpp32u holeH = std::max(1u, static_cast<Rpp32u>(cellH * holeRatio));
        if (holeW > maxHoleW) maxHoleW = holeW;
        if (holeH > maxHoleH) maxHoleH = holeH;

        std::uniform_int_distribution<int> distX(0, (cellW > holeW) ? cellW - holeW : 0);
        std::uniform_int_distribution<int> distY(0, (cellH > holeH) ? cellH - holeH : 0);

        int boxOffset = i * gridH * gridW;
        for (Rpp32u row = 0; row < gridH; ++row) {
            for (Rpp32u col = 0; col < gridW; ++col) {
                Rpp32s cellX = x_base + col * cellW;
                Rpp32s cellY = y_base + row * cellH;

                Rpp32s offsetX = 0, offsetY = 0;
                if ((seed != DROPOUT_FIXED_SEED) && (cellW > holeW) && (cellH > holeH)) {
                    offsetX = distX(rng);
                    offsetY = distY(rng);
                }

                Rpp32s x1 = std::min(cellX + offsetX, x_base + (Rpp32s)roiW - 1);
                Rpp32s y1 = std::min(cellY + offsetY, y_base + (Rpp32s)roiH - 1);
                Rpp32s x2 = std::min(x1 + (Rpp32s)holeW - 1, x_base + (Rpp32s)roiW - 1);
                Rpp32s y2 = std::min(y1 + (Rpp32s)holeH - 1, y_base + (Rpp32s)roiH - 1);

                int boxIdx = boxOffset + (row * gridW + col);
                anchorBoxInfoTensor[boxIdx].lt.x = x1;
                anchorBoxInfoTensor[boxIdx].lt.y = y1;
                anchorBoxInfoTensor[boxIdx].rb.x = x2;
                anchorBoxInfoTensor[boxIdx].rb.y = y2;
            }
        }
    }
}

// Lens correction initializer for unit and performance testing
void inline init_lens_correction(int batchSize, RpptDescPtr srcDescPtr, Rpp32f* cameraMatrix,
                                 Rpp32f* distortionCoeffs, RpptDescPtr tableDescPtr) {
    typedef struct {
        Rpp32f data[9];
    } Rpp32f9;
    typedef struct {
        Rpp32f data[8];
    } Rpp32f8;
    Rpp32f9* cameraMatrix_f9 = reinterpret_cast<Rpp32f9*>(cameraMatrix);
    Rpp32f8* distortionCoeffs_f8 = reinterpret_cast<Rpp32f8*>(distortionCoeffs);
    Rpp32f9 sampleCameraMatrix = {
        534.07088364, 0, 341.53407554, 0, 534.11914595, 232.94565259, 0, 0, 1};
    Rpp32f8 sampleDistortionCoeffs = {-0.29297164, 0.10770696, 0.00131038, -0.0000311,
                                      0.0434798,   0,          0,          0};
    for (int i = 0; i < batchSize; i++) {
        cameraMatrix_f9[i] = sampleCameraMatrix;
        distortionCoeffs_f8[i] = sampleDistortionCoeffs;
    }

    tableDescPtr->c = 1;
    tableDescPtr->strides.nStride = srcDescPtr->h * srcDescPtr->w;
    tableDescPtr->strides.hStride = srcDescPtr->w;
    tableDescPtr->strides.wStride = tableDescPtr->strides.cStride = 1;
}

// fill the permutation values used for transpose
void fill_perm_values(Rpp32u* permTensor, bool qaMode, int permOrder) {
    Rpp8u mapping[][3] = {
        {0, 1, 2},  // axisMask 0 → R, G, B
        {0, 2, 1},  // axisMask 1 → R, B, G
        {1, 0, 2},  // axisMask 2 → G, R, B
        {1, 2, 0},  // axisMask 3 → G, B, R
        {2, 0, 1},  // axisMask 4 → B, R, G
        {2, 1, 0}   // axisMask 5 → B, G, R
    };
    for (int i = 0; i < 3; i++) permTensor[i] = mapping[permOrder][i];
}

#if defined(RPP_TEST_SUITE_HAVE_OPENCV) && RPP_TEST_SUITE_HAVE_OPENCV
// Compare output for single image processing mode
// Only supports: BRIGHTNESS (0), BLEND (2), FLIP (20), RESIZE (21), CROP (37), BOX_FILTER (49),
// MEDIAN_FILTER (51), GAUSSIAN_FILTER (54)
inline void compare_output_single_image(const vector<cv::Mat>& outputVec,
                                        const vector<RpptDesc>& srcDescPtr,
                                        const vector<RpptDesc>& dstDescPtr, string funcName,
                                        RpptImagePatch* dstImgSizes, int noOfImages,
                                        string interpolationTypeName, string noiseTypeName,
                                        int additionalParam, int testCase, string dst,
                                        string scriptPath) {
    string func = funcName;
    string refFile = "";
    int refOutputWidth = ((GOLDEN_OUTPUT_MAX_WIDTH / 8) * 8) + 8;
    int refOutputHeight = GOLDEN_OUTPUT_MAX_HEIGHT;
    int refOutputSize = refOutputHeight * refOutputWidth * dstDescPtr[0].c;
    int pln1RefStride = refOutputHeight * refOutputWidth * noOfImages *
                        3;  // Offset to skip PKD3/PLN3 data in reference file

    string dataType[4] = {"_u8_", "_f32_", "_f16_", "_i8_"};
    func += dataType[dstDescPtr[0].dataType];

    // binFile is for reference file lookup - does NOT include layout information
    std::string binFile = func + "Tensor";

    // Determine source and destination layout names from descriptors
    string srcLayoutName, dstLayoutName;

    // Determine source layout
    if (srcDescPtr[0].layout == RpptLayout::NHWC)
        srcLayoutName = "PKD3";
    else if (srcDescPtr[0].layout == RpptLayout::NCHW && srcDescPtr[0].c == 3)
        srcLayoutName = "PLN3";
    else
        srcLayoutName = "PLN1";

    // Determine destination layout
    if (dstDescPtr[0].layout == RpptLayout::NHWC)
        dstLayoutName = "PKD3";
    else if (dstDescPtr[0].layout == RpptLayout::NCHW && dstDescPtr[0].c == 3)
        dstLayoutName = "PLN3";
    else
        dstLayoutName = "PLN1";

    // func is for display/QA results - DOES include layout information
    func += "Tensor_" + srcLayoutName + "_to_" + dstLayoutName;

    // Handle additional parameters for supported single image cases
    if (testCase == RESIZE) {
        func += "_interpolationType" + interpolationTypeName;
        binFile += "_interpolationType" + interpolationTypeName;
    } else if (testCase == BOX_FILTER || testCase == MEDIAN_FILTER || testCase == GAUSSIAN_FILTER) {
        func += "_kernelSize" + std::to_string(additionalParam);
        binFile += "_kernelSize" + std::to_string(additionalParam);
    }

    refFile = scriptPath + "/../REFERENCE_OUTPUT/" + funcName + "/" + binFile + ".bin";
    int fileMatch = 0;

    if (dstDescPtr[0].dataType == RpptDataType::U8) {
        Rpp64u binOutputSize = refOutputHeight * refOutputWidth * noOfImages * 4;
        Rpp8u* binaryContent = (Rpp8u*)malloc(binOutputSize * sizeof(Rpp8u));
        read_bin_file(refFile, binaryContent);

        // Compare each image individually
        for (int imageCnt = 0; imageCnt < noOfImages; imageCnt++) {
            // For PLN1 layouts, reference data is offset by pln1RefStride to skip PKD3/PLN3 data
            Rpp8u* refOutputImage =
                (dstDescPtr[imageCnt].c == 1 && dstDescPtr[imageCnt].layout == RpptLayout::NCHW)
                    ? binaryContent + pln1RefStride + (imageCnt * refOutputSize)
                    : binaryContent + (imageCnt * refOutputSize);
            int height = dstImgSizes[imageCnt].height;
            int width = dstImgSizes[imageCnt].width;
            int matchedPixels = 0;
            int totalPixels = height * width * dstDescPtr[imageCnt].c;

            if (dstDescPtr[imageCnt].layout == RpptLayout::NHWC) {
                // PKD3 comparison
                for (int i = 0; i < height; i++) {
                    const Rpp8u* outputRow = outputVec[imageCnt].ptr<Rpp8u>(i);
                    const Rpp8u* refRow =
                        refOutputImage + (i * refOutputWidth * dstDescPtr[imageCnt].c);
                    for (int j = 0; j < width * dstDescPtr[imageCnt].c; j++) {
                        int diff = abs(outputRow[j] - refRow[j]);
                        if (diff <= CUTOFF) matchedPixels++;
                    }
                }
            } else if (dstDescPtr[imageCnt].layout == RpptLayout::NCHW &&
                       dstDescPtr[imageCnt].c == 3) {
                // PLN3 comparison - output is planar, reference is packed
                // Channels in Mat are at actualInputHeight intervals, not outputHeight intervals
                int matChannelStride = outputVec[imageCnt].rows / 3;
                for (int c = 0; c < 3; c++) {
                    for (int i = 0; i < height; i++) {
                        const Rpp8u* outputRow =
                            outputVec[imageCnt].ptr<Rpp8u>(i + (c * matChannelStride));
                        const Rpp8u* refRow = refOutputImage + (i * refOutputWidth * 3) + c;
                        for (int j = 0; j < width; j++) {
                            int diff = abs(outputRow[j] - refRow[j * 3]);
                            if (diff <= CUTOFF) matchedPixels++;
                        }
                    }
                }
            } else {
                // PLN1 comparison
                for (int i = 0; i < height; i++) {
                    const Rpp8u* outputRow = outputVec[imageCnt].ptr<Rpp8u>(i);
                    const Rpp8u* refRow = refOutputImage + (i * refOutputWidth);
                    for (int j = 0; j < width; j++) {
                        int diff = abs(outputRow[j] - refRow[j]);
                        if (diff <= CUTOFF) matchedPixels++;
                    }
                }
            }

            if (matchedPixels == totalPixels && matchedPixels != 0) fileMatch++;
        }
        free(binaryContent);
    } else if (dstDescPtr[0].dataType == RpptDataType::F32) {
        Rpp64u binOutputSize = refOutputHeight * refOutputWidth * noOfImages * 4;
        Rpp32f* binaryContent = (Rpp32f*)malloc(binOutputSize * sizeof(Rpp32f));
        read_bin_file(refFile, binaryContent);

        // Compare each image individually
        for (int imageCnt = 0; imageCnt < noOfImages; imageCnt++) {
            // For PLN1 layouts, reference data is offset by pln1RefStride to skip PKD3/PLN3 data
            Rpp32f* refOutputImage =
                (dstDescPtr[imageCnt].c == 1 && dstDescPtr[imageCnt].layout == RpptLayout::NCHW)
                    ? binaryContent + pln1RefStride + (imageCnt * refOutputSize)
                    : binaryContent + (imageCnt * refOutputSize);
            int height = dstImgSizes[imageCnt].height;
            int width = dstImgSizes[imageCnt].width;
            int matchedPixels = 0;
            int totalPixels = height * width * dstDescPtr[imageCnt].c;

            if (dstDescPtr[imageCnt].layout == RpptLayout::NHWC) {
                // PKD3 comparison
                for (int i = 0; i < height; i++) {
                    const Rpp32f* outputRow = outputVec[imageCnt].ptr<Rpp32f>(i);
                    const Rpp32f* refRow =
                        refOutputImage + (i * refOutputWidth * dstDescPtr[imageCnt].c);
                    for (int j = 0; j < width * dstDescPtr[imageCnt].c; j++) {
                        Rpp32f diff = abs(outputRow[j] - refRow[j]);
                        if (diff <= 2e-6) matchedPixels++;
                    }
                }
            } else if (dstDescPtr[imageCnt].layout == RpptLayout::NCHW &&
                       dstDescPtr[imageCnt].c == 3) {
                // PLN3 comparison - output is planar, reference is packed
                // Channels in Mat are at actualInputHeight intervals, not outputHeight intervals
                int matChannelStride = outputVec[imageCnt].rows / 3;
                for (int c = 0; c < 3; c++) {
                    for (int i = 0; i < height; i++) {
                        const Rpp32f* outputRow =
                            outputVec[imageCnt].ptr<Rpp32f>(i + (c * matChannelStride));
                        const Rpp32f* refRow = refOutputImage + (i * refOutputWidth * 3) + c;
                        for (int j = 0; j < width; j++) {
                            Rpp32f diff = abs(outputRow[j] - refRow[j * 3]);
                            if (diff <= 2e-6) matchedPixels++;
                        }
                    }
                }
            } else {
                // PLN1 comparison
                for (int i = 0; i < height; i++) {
                    const Rpp32f* outputRow = outputVec[imageCnt].ptr<Rpp32f>(i);
                    const Rpp32f* refRow = refOutputImage + (i * refOutputWidth);
                    for (int j = 0; j < width; j++) {
                        Rpp32f diff = abs(outputRow[j] - refRow[j]);
                        if (diff <= 2e-6) matchedPixels++;
                    }
                }
            }

            if (matchedPixels == totalPixels && matchedPixels != 0) fileMatch++;
        }
        free(binaryContent);
    }

    std::cout << std::endl << "\nResults for " << func << " :" << std::endl;
    std::string status = func + ": ";
    if (fileMatch == noOfImages) {
        std::cout << "PASSED!";
        status += "PASSED";
    } else {
        std::cout << "FAILED! " << fileMatch << "/" << noOfImages
                  << " outputs are matching with reference outputs";
        status += "FAILED";
    }

    // Append the QA results to file
    std::string qaResultsPath = dst + "/QA_results.txt";
    std::ofstream qaResults(qaResultsPath, ios_base::app);
    if (qaResults.is_open()) {
        qaResults << status << std::endl;
        qaResults.close();
    }
}
#endif /* RPP_TEST_SUITE_HAVE_OPENCV */
