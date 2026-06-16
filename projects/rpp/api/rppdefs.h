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

#ifndef RPPDEFS_H
#define RPPDEFS_H

/*! \file
 * \brief RPP common HOST/GPU typedef, enum and structure definitions.
 * \defgroup group_rppdefs RPP common definitions
 * \brief RPP definitions for all common HOST/GPU typedefs, enums and structures.
 */

#include <stddef.h>

#include <cmath>
#ifdef RPP_BACKEND_HIP
#include <hip/hip_fp16.h>
#endif  // RPP_BACKEND_HIP
#if __has_include(<half/half.hpp>)
#include <half/half.hpp>
#else
#include <half.hpp>
#endif
using halfhpp = half_float::half;
typedef halfhpp Rpp16f;

#if _WIN32
#include <intrin.h>
#else
#include <immintrin.h>
#include <smmintrin.h>
#include <x86intrin.h>
#endif
#include <vector>

/*! \brief 8 bit unsigned char minimum \ingroup group_rppdefs \page subpage_rpp */
#define RPP_MIN_8U (0)
/*! \brief 8 bit unsigned char maximum \ingroup group_rppdefs \page subpage_rppi */
#define RPP_MAX_8U (255)
/*! \brief RPP maximum dimensions in tensor \ingroup group_rppdefs \page subpage_rppt */
#define RPPT_MAX_DIMS (5)
/*! \brief RPP maximum dimensions in tensor for a sample (except batch size) \ingroup group_rppdefs
 * \page subpage_rppt */
#define RPPT_MAX_DIMS_SAMPLE (4)
/*! \brief RPP maximum channels in audio tensor \ingroup group_rppdefs \page subpage_rppt */
#define RPPT_MAX_AUDIO_CHANNELS (16)

#ifdef RPP_BACKEND_HIP
#include <hip/hip_runtime.h>
#define RPP_HOST_DEVICE __host__ __device__

/*! \brief If a HIP API call fails, return \ref RPP_ERROR_HIP_RUNTIME from the enclosing function
 * and print the error message to stderr. \ingroup group_rppdefs */
#define RPP_HIP_RETURN_IF_ERROR(expr)                                                          \
    do {                                                                                       \
        hipError_t _rpp_hip_err = (expr);                                                      \
        if (_rpp_hip_err != hipSuccess) {                                                      \
            fprintf(stderr, "HIP runtime error: %s returned %d at %s:%d", #expr, _rpp_hip_err, \
                    __FILE__, __LINE__);                                                       \
            return RPP_ERROR_HIP_RUNTIME;                                                      \
        }                                                                                      \
    } while (0)

/*! \brief Check last HIP error after kernel launch; return RPP_ERROR_HIP_LAUNCH on failure. Use
 * after hipLaunchKernelGGL. \ingroup group_rppdefs */
#define HIP_CHECK_LAUNCH_RETURN()                                                              \
    do {                                                                                       \
        hipError_t status = hipGetLastError();                                                 \
        if (status != hipSuccess) {                                                            \
            fprintf(stderr, "HIP kernel launch error: returned %d at %s:%d", status, __FILE__, \
                    __LINE__);                                                                 \
            return RPP_ERROR_HIP_LAUNCH;                                                       \
        }                                                                                      \
    } while (0)
#else
#define RPP_HOST_DEVICE
#endif

const float ONE_OVER_6 = 1.0f / 6;
const float ONE_OVER_3 = 1.0f / 3;
const float TWO_OVER_3 = 2.0f / 3;
const float ONE_OVER_255 = 1.0f / 255;
const uint MMS_MAX_SCRATCH_MEMORY = 115293120;  // maximum scratch memory size (in number of floats)
                                                // needed for MMS buffer in RNNT training
const uint SPECTROGRAM_MAX_SCRATCH_MEMORY =
    372877312;  // maximum scratch memory size (in number of floats) needed for spectrogram HIP
                // kernel in RNNT training
#define DROPOUT_FIXED_SEED 42  // Constant fixed seed for reproducing the dropout output
#define RANDOM_ERASE_NOISE_BUFFER_SIDE \
    255  // Random erase spatial noise buffer height and width. Changing this constant will result
         // in QA test failures for tiled noise mapping

/******************** RPP typedefs ********************/

/*! \brief 8 bit unsigned char \ingroup group_rppdefs */
typedef unsigned char Rpp8u;
/*! \brief 8 bit signed char \ingroup group_rppdefs */
typedef signed char Rpp8s;
/*! \brief 16 bit unsigned short \ingroup group_rppdefs */
typedef unsigned short Rpp16u;
/*! \brief 16 bit signed short \ingroup group_rppdefs */
typedef short Rpp16s;
/*! \brief 32 bit unsigned int \ingroup group_rppdefs */
typedef unsigned int Rpp32u;
/*! \brief 32 bit signed int \ingroup group_rppdefs */
typedef int Rpp32s;
/*! \brief 64 bit unsigned long long \ingroup group_rppdefs */
typedef unsigned long long Rpp64u;
/*! \brief 64 bit long long \ingroup group_rppdefs */
typedef long long Rpp64s;
/*! \brief 32 bit float \ingroup group_rppdefs */
typedef float Rpp32f;
/*! \brief 64 bit double \ingroup group_rppdefs */
typedef double Rpp64f;
/*! \brief void pointer \ingroup group_rppdefs */
typedef void* RppPtr_t;
/*! \brief size_t \ingroup group_rppdefs */
typedef size_t RppSize_t;

/*! \brief RPP RppStatus type enums
 * \ingroup group_rppdefs
 */
typedef enum {
    /*! \brief No error. \ingroup group_rppdefs */
    RPP_SUCCESS = 0,
    /*! \brief Unspecified error. \ingroup group_rppdefs */
    RPP_ERROR = -1,
    /*! \brief One or more arguments invalid. (Needs to adhere to function specification.) \ingroup
       group_rppdefs */
    RPP_ERROR_INVALID_ARGUMENTS = -2,
    /*! \brief Low tensor offsetInBytes provided for src/dst tensor. \ingroup group_rppdefs */
    RPP_ERROR_LOW_OFFSET = -3,
    /*! \brief Arguments provided will result in zero division error. \ingroup group_rppdefs */
    RPP_ERROR_ZERO_DIVISION = -4,
    /*! \brief Src tensor / src ROI dimension too high. (Needs to adhere to function specification.)
       \ingroup group_rppdefs */
    RPP_ERROR_HIGH_SRC_DIMENSION = -5,
    /*! \brief Function variant requested is not implemented / unsupported. \ingroup group_rppdefs
     */
    RPP_ERROR_NOT_IMPLEMENTED = -6,
    /*! \brief Invalid src tensor number of channels. (Needs to adhere to function specification.)
       \ingroup group_rppdefs */
    RPP_ERROR_INVALID_SRC_CHANNELS = -7,
    /*! \brief Invalid dst tensor number of channels. (Needs to adhere to function specification.)
       \ingroup group_rppdefs */
    RPP_ERROR_INVALID_DST_CHANNELS = -8,
    /*! \brief Invalid src tensor layout. (Needs to adhere to function specification.) \ingroup
       group_rppdefs */
    RPP_ERROR_INVALID_SRC_LAYOUT = -9,
    /*! \brief Invalid dst tensor layout. (Needs to adhere to function specification.) \ingroup
       group_rppdefs */
    RPP_ERROR_INVALID_DST_LAYOUT = -10,
    /*! \brief Invalid src tensor datatype. (Needs to adhere to function specification.) \ingroup
       group_rppdefs */
    RPP_ERROR_INVALID_SRC_DATATYPE = -11,
    /*! \brief Invalid dst tensor datatype. (Needs to adhere to function specification.) \ingroup
       group_rppdefs */
    RPP_ERROR_INVALID_DST_DATATYPE = -12,
    /*! \brief Invalid src/dst tensor datatype. (Needs to adhere to function specification.)
       \ingroup group_rppdefs */
    RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE = -13,
    /*! \brief Insufficient dst buffer length provided. (Needs to adhere to function specification.)
       \ingroup group_rppdefs */
    RPP_ERROR_INSUFFICIENT_DST_BUFFER_LENGTH = -14,
    /*! \brief Invalid datatype \ingroup group_rppdefs */
    RPP_ERROR_INVALID_PARAMETER_DATATYPE = -15,
    /*! \brief Not enough memory to write outputs, as per dim-lengths and strides set in descriptor
       \ingroup group_rppdefs */
    RPP_ERROR_NOT_ENOUGH_MEMORY = -16,
    /*! \brief Out of bound source ROI \ingroup group_rppdefs */
    RPP_ERROR_OUT_OF_BOUND_SRC_ROI = -17,
    /*! \brief src and dst layout mismatch \ingroup group_rppdefs */
    RPP_ERROR_LAYOUT_MISMATCH = -18,
    /*! \brief Number of channels is invalid. (Needs to adhere to function specification.) \ingroup
       group_rppdefs */
    RPP_ERROR_INVALID_CHANNELS = -19,
    /*! \brief Invalid output tile length (Needs to adhere to function specification.) \ingroup
       group_rppdefs */
    RPP_ERROR_INVALID_OUTPUT_TILE_LENGTH = -20,
    /*! \brief Shared memory size needed is beyond the bounds (Needs to adhere to function
       specification.) \ingroup group_rppdefs */
    RPP_ERROR_OUT_OF_BOUND_SHARED_MEMORY_SIZE = -21,
    /*! \brief Scratch memory size needed is beyond the bounds (Needs to adhere to function
       specification.) \ingroup group_rppdefs */
    RPP_ERROR_OUT_OF_BOUND_SCRATCH_MEMORY_SIZE = -22,
    /*! \brief Number of src dims is invalid. (Needs to adhere to function specification.) \ingroup
       group_rppdefs */
    RPP_ERROR_INVALID_SRC_DIMS = -23,
    /*! \brief Number of dst dims is invalid. (Needs to adhere to function specification.) \ingroup
       group_rppdefs */
    RPP_ERROR_INVALID_DST_DIMS = -24,
    /*! \brief Length in some dimensions are invalid. (Needs to adhere to function specification.)
       \ingroup group_rppdefs */
    RPP_ERROR_INVALID_DIM_LENGTHS = -25,
    /*! \brief The specified axis is invalid or out of range. (Needs to adhere to function
       specification.) \ingroup group_rppdefs */
    RPP_ERROR_INVALID_AXIS = -26,
    /*! \brief The user specified backend is not compatible with the initialized handle \ingroup
       group_rppdefs */
    RPP_ERROR_INCOMPATIBLE_BACKEND = -27,
    /*! \brief HIP/GPU runtime or kernel launch error \ingroup group_rppdefs */
    RPP_ERROR_HIP_LAUNCH = -28,
    /*! \brief Internal HIP/GPU runtime error \ingroup group_rppdefs */
    RPP_ERROR_HIP_RUNTIME = -29,
} RppStatus;

/*! \brief RPP RppBackend type enums
 * \ingroup group_rppdefs
 */
typedef enum { RPP_HOST_BACKEND, RPP_HIP_BACKEND } RppBackend;

/*! \brief RPP rppStatus_t type enums
 * \ingroup group_rppdefs
 */
typedef enum {
    rppStatusSuccess = 0,
    rppStatusBadParm = -1,
    rppStatusUnknownError = -2,
    rppStatusNotInitialized = -3,
    rppStatusInvalidValue = -4,
    rppStatusAllocFailed = -5,
    rppStatusInternalError = -6,
    rppStatusNotImplemented = -7,
    rppStatusUnsupportedOp = -8,
} rppStatus_t;

/*! \brief RPP Tensor Operations type enum
 * \ingroup group_rppdefs
 */
typedef enum {
    RPP_TENSOR_OP_ADD,
    RPP_TENSOR_OP_SUBTRACT,
    RPP_TENSOR_OP_MULTIPLY,
    RPP_TENSOR_OP_DIVIDE
} RpptOp;

/*! \brief RPPT Broadcast Mode type enum
 * \ingroup group_rppdefs
 */
typedef enum { RPP_TENSOR_OP_AND, RPP_TENSOR_OP_OR, RPP_TENSOR_OP_XOR } RpptBitwiseOp;

/*!
 * \brief Broadcast support mode for RPP Tensor operations.
 * \ingroup group_defs
 *
 * This enum specifies whether broadcasting is enabled or disabled
 * when applying tensor operations.
 */
typedef enum {
    RPP_BROADCAST_ENABLE,
    RPP_BROADCAST_DISABLE  // Broadcasting is disabled; Requires input tensors to be of the same
                           // shape
} RpptBroadcastMode;

/*! \brief RPP layout params
 * \ingroup group_rppdefs
 */
typedef struct {
    Rpp32u channelParam;
    Rpp32u bufferMultiplier;
} RppLayoutParams;

/*! \brief RPP 6 float vector
 * \ingroup group_rppdefs
 */
typedef struct {
    Rpp32f data[6];
} Rpp32f6;

/*! \brief RPP 9 float vector
 * \ingroup group_rppdefs
 */
typedef struct {
    Rpp32f data[9];
} Rpp32f9;

/*! \brief RPP 24 signed int vector
 * \ingroup group_rppdefs
 */
typedef struct {
    Rpp32s data[24];
} Rpp32s24;

/*! \brief RPP 24 float vector
 * \ingroup group_rppdefs
 */
typedef struct {
    Rpp32f data[24];
} Rpp32f24;

/******************** RPPT typedefs ********************/

/*! \brief RPPT Tensor datatype enum
 * \ingroup group_rppdefs
 */
typedef enum { U8, F32, F16, I8, I16, U16, I32, U32 } RpptDataType;

/*! \brief RPPT Tensor layout type enum
 * \ingroup group_rppdefs
 */
typedef enum {
    NCHW,   // BatchSize-Channels-Height-Width
    NHWC,   // BatchSize-Height-Width-Channels
    NCDHW,  // BatchSize-Channels-Depth-Height-Width
    NDHWC,  // BatchSize-Depth-Height-Width-Channels
    NHW,    // BatchSize-Height-Width
    NFT,    // BatchSize-Frequency-Time -> Frequency Major used for Spectrogram / MelfilterBank
    NTF     // BatchSize-Time-Frequency -> Time Major used for Spectrogram / MelfilterBank
} RpptLayout;

/*! \brief Color space standard for NV12 to RGB (YUV/YCbCr to RGB matrix).
 * \details Selects which industry **color space standard** defines the luma coefficients (\e wr, \e
 * wb). \ingroup group_rppdefs
 */
typedef enum {
    RpptColorStandard_BT709 = 0,      /*!< ITU-R BT.709 (default wr=0.2126, wb=0.0722) */
    RpptColorStandard_FCC = 4,        /*!< FCC */
    RpptColorStandard_BT470BG = 5,    /*!< ITU-R BT.470 System B, G */
    RpptColorStandard_BT601 = 6,      /*!< ITU-R BT.601 / SMPTE 170M */
    RpptColorStandard_SMPTE240M = 7,  /*!< SMPTE 240M */
    RpptColorStandard_BT2020_NCL = 9, /*!< ITU-R BT.2020 non-constant luminance */
    RpptColorStandard_BT2020_CL = 10  /*!< ITU-R BT.2020 constant luminance */
} RpptColorStandard;

/*! \brief Color range (luma/chroma legal levels) for NV12 to RGB.
 * \details **Color range** specifies how 8-bit Y (and scaling of the matrix) is interpreted:
 * **limited** (studio / TV / MPEG) vs **full** (JPEG / PC). \ref RpptColorRange_STUDIO: luma
 * nominally 16–235, chroma centered with Y bias 16. \ref RpptColorRange_FULL: luma 0–255, Y bias 0.
 * \ingroup group_rppdefs
 */
typedef enum {
    RpptColorRange_STUDIO = 0, /*!< Limited / MPEG / TV range */
    RpptColorRange_FULL = 2    /*!< Full / JPEG / PC range */
} RpptColorRange;

/*! \brief RPPT Tensor 2D ROI type enum
 * \ingroup group_rppdefs
 */
typedef enum {
    LTRB,  // Left-Top-Right-Bottom
    XYWH   // X-Y-Width-Height
} RpptRoiType;

/*! \brief RPPT Tensor 3D ROI type enum
 * \ingroup group_rppdefs
 */
typedef enum {
    LTFRBB,  // Left-Top-Front-Right-Bottom-Back
    XYZWHD   // X-Y-Z-Width-Height-Depth
} RpptRoi3DType;

/*! \brief RPPT Tensor subpixel layout type enum
 * \ingroup group_rppdefs
 */
typedef enum { RGBtype, BGRtype } RpptSubpixelLayout;

/*! \brief RPPT Tensor interpolation type enum
 * \ingroup group_rppdefs
 */
typedef enum {
    NEAREST_NEIGHBOR = 0,
    BILINEAR,
    BICUBIC,
    LANCZOS,
    GAUSSIAN,
    TRIANGULAR
} RpptInterpolationType;

/*! \brief RPPT Audio Border Type
 * \ingroup group_rppdefs
 */
typedef enum { ZERO = 0, CLAMP, REFLECT } RpptAudioBorderType;

/*! \brief RPPT Image Border Edge type enum
 * \ingroup group_rppdefs
 */
typedef enum { TOP_EDGE = 0, BOTTOM_EDGE, LEFT_EDGE, RIGHT_EDGE } RpptImageBorderEdge;

/*! \brief RPPT Image Border Type
 * \ingroup group_rppdefs
 */
typedef enum {
    REPLICATE = 0,   // Similar to Nearest Neighbors Padding
    CONSTANT,        // Unsupported Border Type
    REFLECT_NO_EDGE  // Unsupported Border Type
} RpptImageBorderType;

/*! \brief RPPT Mel Scale Formula
 * \ingroup group_rppdefs
 */
typedef enum {
    SLANEY = 0,  // Follows Slaney’s MATLAB Auditory Modelling Work behavior
    HTK,  // Follows O’Shaughnessy’s book formula, consistent with Hidden Markov Toolkit(HTK), m =
          // 2595 * log10(1 + (f/700))
} RpptMelScaleFormula;

/*! \brief RPPT Image 2D cartesian point type struct
 * \ingroup group_rppdefs
 */
typedef struct {
    int x;
    int y;
} RpptPoint2D;

/*! \brief RPPT Image 3D point type struct
 * \ingroup group_rppdefs
 */
typedef struct {
    int x;
    int y;
    int z;
} RpptPoint3D;

/*! \brief RPPT Tensor 2D ROI LTRB struct
 * \ingroup group_rppdefs
 */
typedef struct {
    RpptPoint2D lt, rb;  // Left-Top point and Right-Bottom point

} RpptRoiLtrb;

/*! \brief RPPT Tensor Channel Offsets struct
 * \ingroup group_rppdefs
 */
typedef struct {
    RpptPoint2D r;
    RpptPoint2D g;
    RpptPoint2D b;
} RpptChannelOffsets;

/*! \brief RPPT Tensor 3D ROI LTFRBB struct
 * \ingroup group_rppdefs
 */
typedef struct {
    RpptPoint3D ltf, rbb;  // Left-Top-Front point and Right-Bottom-Back point

} RpptRoiLtfrbb;

/*! \brief RPPT Tensor 2D ROI XYWH struct
 * \ingroup group_rppdefs
 */
typedef struct {
    RpptPoint2D xy;
    int roiWidth, roiHeight;

} RpptRoiXywh;

/*! \brief RPPT Tensor 3D ROI XYZWHD struct
 * \ingroup group_rppdefs
 */
typedef struct {
    RpptPoint3D xyz;
    int roiWidth, roiHeight, roiDepth;

} RpptRoiXyzwhd;

/*! \brief RPPT Tensor 2D ROI union
 * \ingroup group_rppdefs
 */
typedef union {
    RpptRoiLtrb ltrbROI;  // ROI defined as Left-Top-Right-Bottom
    RpptRoiXywh xywhROI;  // ROI defined as X-Y-Width-Height

} RpptROI, *RpptROIPtr;

/*! \brief RPPT Tensor 3D ROI union
 * \ingroup group_rppdefs
 */
typedef union {
    RpptRoiLtfrbb ltfrbbROI;  // ROI defined as Left-Top-Front-Right-Bottom-Back
    RpptRoiXyzwhd xyzwhdROI;  // ROI defined as X-Y-Z-Width-Height-Depth

} RpptROI3D, *RpptROI3DPtr;

/*! \brief RPPT Tensor strides type struct
 * \ingroup group_rppdefs
 */
typedef struct {
    Rpp32u nStride;
    Rpp32u cStride;
    Rpp32u hStride;
    Rpp32u wStride;
} RpptStrides;

/*! \brief RPPT Tensor descriptor type struct
 * \ingroup group_rppdefs
 */
typedef struct {
    RppSize_t numDims;
    Rpp32u offsetInBytes;
    RpptDataType dataType;
    Rpp32u n, c, h, w;
    RpptStrides strides;
    RpptLayout layout;
} RpptDesc, *RpptDescPtr;

/*! \brief RPPT Tensor Generic descriptor type struct
 * \ingroup group_rppdefs
 */
typedef struct {
    RppSize_t numDims;
    Rpp32u offsetInBytes;
    RpptDataType dataType;
    Rpp32u dims[RPPT_MAX_DIMS];
    Rpp32u strides[RPPT_MAX_DIMS];
    RpptLayout layout;
} RpptGenericDesc, *RpptGenericDescPtr;

/*! \brief RPPT Tensor 8-bit uchar RGB type struct
 * \ingroup group_rppdefs
 */
typedef struct {
    Rpp8u R;
    Rpp8u G;
    Rpp8u B;
} RpptRGB;

/*! \brief RPPT Tensor 32-bit float RGB type struct
 * \ingroup group_rppdefs
 */
typedef struct {
    Rpp32f R;
    Rpp32f G;
    Rpp32f B;
} RpptFloatRGB;

/*! \brief RPPT Tensor 2D 32-bit uint vector type struct
 * \ingroup group_rppdefs
 */
typedef struct {
    Rpp32u x;
    Rpp32u y;
} RpptUintVector2D;

/*! \brief RPPT Tensor 2D 32-bit float vector type struct
 * \ingroup group_rppdefs
 */
typedef struct {
    Rpp32f x;
    Rpp32f y;
} RpptFloatVector2D;

/*! \brief RPPT Tensor 2D image patch dimensions type struct
 * \ingroup group_rppdefs
 */
typedef struct {
    Rpp32u width;
    Rpp32u height;
} RpptImagePatch, *RpptImagePatchPtr;

/*! \brief RPPT Tensor random number generator state (xorwow state) type struct
 * \ingroup group_rppdefs
 */
typedef struct {
    Rpp32u x[5];
    Rpp32u counter;
} RpptXorwowState;

/*! \brief RPPT Tensor random number generator state (xorwow box muller state) type struct
 * \ingroup group_rppdefs
 */
typedef struct {
    Rpp32s x[5];
    Rpp32s counter;
    int boxMullerFlag;
    float boxMullerExtra;
} RpptXorwowStateBoxMuller;

/*! \brief RPPT Tensor 2D bilinear neighborhood 32-bit signed int 8-length-vectors type struct
 * \ingroup group_rppdefs
 */
typedef struct {
    Rpp32s24 srcLocsTL;
    Rpp32s24 srcLocsTR;
    Rpp32s24 srcLocsBL;
    Rpp32s24 srcLocsBR;
} RpptBilinearNbhoodLocsVecLen8;

/*! \brief RPPT Tensor 2D bilinear neighborhood 32-bit float 8-length-vectors type struct
 * \ingroup group_rppdefs
 */
typedef struct {
    Rpp32f24 srcValsTL;
    Rpp32f24 srcValsTR;
    Rpp32f24 srcValsBL;
    Rpp32f24 srcValsBR;
} RpptBilinearNbhoodValsVecLen8;

/*! \brief RPPT Tensor GenericFilter type struct
 * \ingroup group_rppdefs
 */
typedef struct GenericFilter {
    Rpp32f scale = 1.0f;
    Rpp32f radius = 1.0f;
    Rpp32s size;
    GenericFilter(RpptInterpolationType interpolationType, Rpp32s in_size, Rpp32s out_size,
                  Rpp32f scaleRatio) {
        switch (interpolationType) {
            case RpptInterpolationType::BICUBIC: {
                this->radius = 2.0f;
                break;
            }
            case RpptInterpolationType::LANCZOS: {
                if (in_size > out_size) {
                    this->radius = 3.0f * scaleRatio;
                    this->scale = (1 / scaleRatio);
                } else
                    this->radius = 3.0f;
                break;
            }
            case RpptInterpolationType::GAUSSIAN: {
                if (in_size > out_size) {
                    this->radius = scaleRatio;
                    this->scale = (1 / scaleRatio);
                }
                break;
            }
            case RpptInterpolationType::TRIANGULAR: {
                if (in_size > out_size) {
                    this->radius = scaleRatio;
                    this->scale = (1 / scaleRatio);
                }
                break;
            }
            default: {
                this->radius = 1.0f;
                this->scale = 1.0f;
                break;
            }
        }
        this->size = std::ceil(2 * this->radius);
    }
} GenericFilter;

/*! \brief RPPT Tensor RpptResamplingWindow type struct
 * \ingroup group_rppdefs
 */
typedef struct RpptResamplingWindow {
    inline RPP_HOST_DEVICE void input_range(Rpp32f x, Rpp32s* loc0, Rpp32s* loc1) {
        Rpp32s xc = std::ceil(x);
        *loc0 = xc - lobes;
        *loc1 = xc + lobes;
    }

    inline Rpp32f operator()(Rpp32f x) {
        Rpp32f locRaw = x * scale + center;
        Rpp32s locFloor = std::floor(locRaw);
        Rpp32f weight = locRaw - locFloor;
        locFloor = std::max(std::min(locFloor, lookupSize - 2), 0);
        Rpp32f current = lookup[locFloor];
        Rpp32f next = lookup[locFloor + 1];
        return current + weight * (next - current);
    }

    inline __m128 operator()(__m128 x) {
        __m128 pLocRaw = _mm_add_ps(_mm_mul_ps(x, pScale), pCenter);
        __m128i pxLocFloor = _mm_cvttps_epi32(pLocRaw);
        __m128 pLocFloor = _mm_cvtepi32_ps(pxLocFloor);
        __m128 pWeight = _mm_sub_ps(pLocRaw, pLocFloor);
        Rpp32s idx[4];
        _mm_storeu_si128(reinterpret_cast<__m128i*>(idx), pxLocFloor);
        __m128 pCurrent =
            _mm_setr_ps(lookup[idx[0]], lookup[idx[1]], lookup[idx[2]], lookup[idx[3]]);
        __m128 pNext = _mm_setr_ps(lookup[idx[0] + 1], lookup[idx[1] + 1], lookup[idx[2] + 1],
                                   lookup[idx[3] + 1]);
        return _mm_add_ps(pCurrent, _mm_mul_ps(pWeight, _mm_sub_ps(pNext, pCurrent)));
    }

    Rpp32f scale = 1, center = 1;
    Rpp32s lobes = 0, coeffs = 0;
    Rpp32s lookupSize = 0;
    Rpp32f* lookupPinned = nullptr;
    std::vector<Rpp32f> lookup;
    __m128 pCenter, pScale;
} RpptResamplingWindow;

/*! \brief Base class for Mel scale conversions.
 * \ingroup group_rppdefs
 */
struct BaseMelScale {
   public:
    inline RPP_HOST_DEVICE virtual Rpp32f hz_to_mel(Rpp32f hz) = 0;
    inline RPP_HOST_DEVICE virtual Rpp32f mel_to_hz(Rpp32f mel) = 0;
    virtual ~BaseMelScale() = default;
};

/*! \brief Derived class for HTK Mel scale conversions.
 * \ingroup group_rppdefs
 */
struct HtkMelScale : public BaseMelScale {
    inline RPP_HOST_DEVICE Rpp32f hz_to_mel(Rpp32f hz) {
        return 1127.0f * std::log(1.0f + (hz / 700.0f));
    }
    inline RPP_HOST_DEVICE Rpp32f mel_to_hz(Rpp32f mel) {
        return 700.0f * (std::exp(mel / 1127.0f) - 1.0f);
    }

   public:
    ~HtkMelScale() {};
};

/*! \brief Derived class for Slaney Mel scale conversions.
 * \ingroup group_rppdefs
 */
struct SlaneyMelScale : public BaseMelScale {
    const Rpp32f freqLow = 0;
    const Rpp32f fsp = 66.666667f;
    const Rpp32f minLogHz = 1000.0;
    const Rpp32f minLogMel = (minLogHz - freqLow) / fsp;
    const Rpp32f stepLog = 0.068751777;  // Equivalent to std::log(6.4) / 27.0;

    const Rpp32f invMinLogHz = 0.001f;
    const Rpp32f invStepLog = 1.0f / stepLog;
    const Rpp32f invFsp = 1.0f / fsp;

    inline RPP_HOST_DEVICE Rpp32f hz_to_mel(Rpp32f hz) {
        Rpp32f mel = 0.0f;
        if (hz >= minLogHz)
            mel = minLogMel + std::log(hz * invMinLogHz) * invStepLog;
        else
            mel = (hz - freqLow) * invFsp;

        return mel;
    }

    inline RPP_HOST_DEVICE Rpp32f mel_to_hz(Rpp32f mel) {
        Rpp32f hz = 0.0f;
        if (mel >= minLogMel)
            hz = minLogHz * std::exp(stepLog * (mel - minLogMel));
        else
            hz = freqLow + mel * fsp;
        return hz;
    }

   public:
    ~SlaneyMelScale() {};
};

/******************** HOST memory typedefs ********************/

/*! \brief RPP HOST memory type struct
 * \ingroup group_rppdefs
 */
typedef struct {
    Rpp32f* scratchBufferHost;
} memCPU;

#ifdef RPP_BACKEND_HIP

/******************** HIP memory typedefs ********************/

/*! \brief RPP HIP 32-bit float memory
 * \ingroup group_rppdefs
 */
typedef struct {
    Rpp32f* floatmem;
} hipMemRpp32f;

/*! \brief RPP HIP memory management type struct
 * \ingroup group_rppdefs
 */
typedef struct {
    hipMemRpp32f scratchBufferHip;
    hipMemRpp32f scratchBufferPinned;
} memGPU;

/*! \brief RPP HIP-HOST memory management
 * \ingroup group_rppdefs
 */
typedef struct {
    memCPU mcpu;
    memGPU mgpu;
} memMgmt;

#else

/*! \brief RPP HOST memory management
 * \ingroup group_rppdefs
 */
typedef struct {
    memCPU mcpu;
} memMgmt;

#endif  // BACKEND

/*! \brief RPP initialize handle
 * \ingroup group_rppdefs
 */
typedef struct {
    RppPtr_t cpuHandle;
    Rpp32u nbatchSize;
    memMgmt mem;
} InitHandle;

#endif /* RPPDEFS_H */
