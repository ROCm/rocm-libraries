/*
Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef MIVISIONX_ROCAL_API_TYPES_H
#define MIVISIONX_ROCAL_API_TYPES_H

#include <cstdlib>

#ifndef ROCAL_API_CALL
#if defined(_WIN32)
#define ROCAL_API_CALL __stdcall
#else
/*!
 * rocAL API Call macro.
 */
#define ROCAL_API_CALL
#endif
#endif

/*!
 * \file
 * \brief The AMD rocAL Library - Types
 *
 * \defgroup group_rocal_types API: AMD rocAL - Types
 * \brief The AMD rocAL Types.
 */

#include <half/half.hpp>
using half_float::half;

/*! \brief typedef void* Float Param
 * \ingroup group_rocal_types
 */
typedef void* RocalFloatParam;

/*! \brief typedef void* rocAL Int Param
 * \ingroup group_rocal_types
 */
typedef void* RocalIntParam;

/*! \brief typedef void* rocAL Context
 * \ingroup group_rocal_types
 */
typedef void* RocalContext;

/*! \brief typedef std::vectors
 * \ingroup group_rocal_types
 */
///@{
typedef std::vector<int> ImageIDBatch, AnnotationIDBatch;
typedef std::vector<std::string> ImagePathBatch;
typedef std::vector<float> ScoreBatch, RotationBatch;
typedef std::vector<std::vector<float>> CenterBatch, ScaleBatch;
typedef std::vector<std::vector<std::vector<float>>> JointsBatch, JointsVisibilityBatch;
///@}

/*! \brief Timing Info struct
 * \ingroup group_rocal_types
 */
struct TimingInfo {
    long long unsigned load_time;
    long long unsigned decode_time;
    long long unsigned process_time;
    long long unsigned transfer_time;
};

// HRNet training expects meta data (joints_data) in below format, so added here as a type for exposing to user
/*! \brief rocAL Joints Data struct - HRNet training expects meta data (joints_data) in below format, so added here as a type for exposing to user
 * \ingroup group_rocal_types
 */
struct RocalJointsData {
    ImageIDBatch image_id_batch;
    AnnotationIDBatch annotation_id_batch;
    ImagePathBatch image_path_batch;
    CenterBatch center_batch;
    ScaleBatch scale_batch;
    JointsBatch joints_batch;
    JointsVisibilityBatch joints_visibility_batch;
    ScoreBatch score_batch;
    RotationBatch rotation_batch;
};

struct ROIxywh {
    unsigned x;
    unsigned y;
    unsigned w;
    unsigned h;
};

/*! \brief  rocAL Status enum
 * \ingroup group_rocal_types
 */
enum RocalStatus {
    /*! \brief AMD ROCAL_OK
     */
    ROCAL_OK = 0,
    /*! \brief AMD ROCAL_CONTEXT_INVALID
     */
    ROCAL_CONTEXT_INVALID,
    /*! \brief AMD ROCAL_RUNTIME_ERROR
     */
    ROCAL_RUNTIME_ERROR,
    /*! \brief AMD ROCAL_UPDATE_PARAMETER_FAILED
     */
    ROCAL_UPDATE_PARAMETER_FAILED,
    /*! \brief AMD ROCAL_INVALID_PARAMETER_TYPE
     */
    ROCAL_INVALID_PARAMETER_TYPE
};

/*! \brief rocAL Image Color enum
 * \ingroup group_rocal_types
 */
enum RocalImageColor {
    /*! \brief AMD ROCAL_COLOR_RGB24
     */
    ROCAL_COLOR_RGB24 = 0,
    /*! \brief AMD ROCAL_COLOR_BGR24
     */
    ROCAL_COLOR_BGR24 = 1,
    /*! \brief AMD ROCAL_COLOR_U8
     */
    ROCAL_COLOR_U8 = 2,
    /*! \brief AMD ROCAL_COLOR_RGB_PLANAR
     */
    ROCAL_COLOR_RGB_PLANAR = 3,
};

/*! \brief rocAL Process Mode enum
 * \ingroup group_rocal_types
 */
enum RocalProcessMode {
    /*! \brief AMD ROCAL_PROCESS_GPU
     */
    ROCAL_PROCESS_GPU = 0,
    /*! \brief AMD ROCAL_PROCESS_CPU
     */
    ROCAL_PROCESS_CPU = 1
};

/*! \brief rocAL Flip Axis enum
 * \ingroup group_rocal_types
 */
enum RocalFlipAxis {
    /*! \brief AMD ROCAL_FLIP_HORIZONTAL
     */
    ROCAL_FLIP_HORIZONTAL = 0,
    /*! \brief AMD ROCAL_FLIP_VERTICAL
     */
    ROCAL_FLIP_VERTICAL = 1
};

/*! \brief rocAL Image Size Evaluation Policy enum
 * \ingroup group_rocal_types
 */
enum RocalImageSizeEvaluationPolicy {
    /*! \brief AMD ROCAL_USE_MAX_SIZE
     */
    ROCAL_USE_MAX_SIZE = 0,
    /*! \brief AMD ROCAL_USE_USER_GIVEN_SIZE
     */
    ROCAL_USE_USER_GIVEN_SIZE = 1,
    /*! \brief AMD ROCAL_USE_MOST_FREQUENT_SIZE
     */
    ROCAL_USE_MOST_FREQUENT_SIZE = 2,
    /*! \brief Use the given size only if the actual decoded size is greater than the given size
     */
    ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED = 3,  // use the given size only if the actual decoded size is greater than the given size
    /*! \brief Use max size if the actual decoded size is greater than max
     */
    ROCAL_USE_MAX_SIZE_RESTRICTED = 4,         // use max size if the actual decoded size is greater than max
};

/*! \brief rocAL Decode Device enum
 * \ingroup group_rocal_types
 */
enum RocalDecodeDevice {
    /*! \brief AMD ROCAL_HW_DECODE
     */
    ROCAL_HW_DECODE = 0,
    /*! \brief AMD ROCAL_SW_DECODE
     */
    ROCAL_SW_DECODE = 1
};

/*! \brief rocAL Tensor Layout enum
 * \ingroup group_rocal_types
 */
enum RocalTensorLayout {
    /*! \brief AMD ROCAL_NHWC
     */
    ROCAL_NHWC = 0,
    /*! \brief AMD ROCAL_NCHW
     */
    ROCAL_NCHW = 1,
    /*! \brief AMD ROCAL_NFHWc
     */
    ROCAL_NFHWC = 2,
    /*! \brief AMD ROCAL_NFCHW
     */
    ROCAL_NFCHW = 3,
    /*! \brief AMD ROCAL_NHW
     */
    ROCAL_NHW = 4,
    /*! \brief AMD ROCAL_NFT
     * Spectrogram Layout FT
     */
    ROCAL_NFT = 5,
    /*! \brief AMD ROCAL_NTF
     * Spectrogram Layout TF
     */
    ROCAL_NTF = 6,
    /*! \brief AMD ROCAL_NONE
     */
    ROCAL_NONE = 7  // Layout for generic tensors (Non-Image or Non-Video)
};

/*! \brief rocAL Tensor Output Type enum
 * \ingroup group_rocal_types
 */
enum RocalTensorOutputType {
    /*! \brief AMD ROCAL_FP32
     */
    ROCAL_FP32 = 0,
    /*! \brief AMD ROCAL_FP16
     */
    ROCAL_FP16 = 1,
    /*! \brief AMD ROCAL_UINT8
     */
    ROCAL_UINT8 = 2,
    /*! \brief AMD ROCAL_INT8
     */
    ROCAL_INT8 = 3,
    /*! \brief AMD ROCAL_UINT32
     */
    ROCAL_UINT32 = 4,
    /*! \brief AMD ROCAL_INT32
     */
    ROCAL_INT32 = 5,
    /*! \brief AMD ROCAL_INT16
     */
    ROCAL_INT16 = 6
};

/*! \brief rocAL Decoder Type enum
 * \ingroup group_rocal_types
 */
enum RocalDecoderType {
    /*! \brief AMD ROCAL_DECODER_TJPEG
     */
    ROCAL_DECODER_TJPEG = 0,
    /*! \brief AMD ROCAL_DECODER_OPENCV
     */
    ROCAL_DECODER_OPENCV = 1,
    /*! \brief AMD ROCAL_DECODER_HW_JPEG
     */
    ROCAL_DECODER_HW_JPEG = 2,
    /*! \brief AMD ROCAL_DECODER_VIDEO_FFMPEG_SW
     */
    ROCAL_DECODER_VIDEO_FFMPEG_SW = 3,
    /*! \brief AMD ROCAL_DECODER_VIDEO_FFMPEG_HW
     */
    ROCAL_DECODER_VIDEO_FFMPEG_HW = 4,
    /*! \brief AMD ROCAL_DECODER_AUDIO_GENERIC
     * Uses SndFile library to read audio files
     */
    ROCAL_DECODER_AUDIO_GENERIC = 5,
    /*! \brief AMD ROCAL_DECODER_VIDEO_ROCDECODE
     * Uses rocDecode library to decode videos on hardware
     */
    ROCAL_DECODER_VIDEO_ROCDECODE = 6,
    /*! \brief AMD ROCAL_DECODER_ROCJPEG
     * Uses rocJpeg library to decode images on hardware
     */
    ROCAL_DECODER_ROCJPEG = 7
};

enum RocalOutputMemType {
    /*! \brief AMD ROCAL_MEMCPY_HOST
     */
    ROCAL_MEMCPY_HOST = 0,
    /*! \brief AMD ROCAL_MEMCPY_GPU
     */
    ROCAL_MEMCPY_GPU = 1,
    /*! \brief AMD ROCAL_MEMCPY_PINNED
     */
    ROCAL_MEMCPY_PINNED = 2
};

// rocal external memcpy flags
/*! \brief AMD rocAL external memcpy flags - force copy to user provided host memory
 * \ingroup group_rocal_types
 */
#define ROCAL_MEMCPY_TO_HOST 1    // force copy to user provided host memory
/*! \brief AMD rocAL external memcpy flags - force copy to user provided device memory (gpu)
 * \ingroup group_rocal_types
 */
#define ROCAL_MEMCPY_TO_DEVICE 2  // force copy to user provided device memory (gpu)
/*! \brief AMD rocAL external memcpy flags - for future use
 * \ingroup group_rocal_types
 */
#define ROCAL_MEMCPY_IS_PINNED 4  // for future use

/*! \brief rocAL Resize Scaling Mode enum
 * \ingroup group_rocal_types
 */
enum RocalResizeScalingMode {
    /*! \brief scales wrt specified size, if only resize width/height is provided the other dimension is scaled according to aspect ratio
     */
    ROCAL_SCALING_MODE_DEFAULT = 0,
    /*! \brief scales wrt specified size, if only resize width/height is provided the other dimension is not scaled
     */
    ROCAL_SCALING_MODE_STRETCH = 1,
    /*! \brief scales wrt to aspect ratio, so that resize width/height is not lesser than the specified size
     */
    ROCAL_SCALING_MODE_NOT_SMALLER = 2,
    /*! \brief scales wrt to aspect ratio, so that resize width/height does not exceed specified size
     */
    ROCAL_SCALING_MODE_NOT_LARGER = 3,
    /*! \brief scales wrt to aspect ratio, so that resize width/height does not exceed specified min and max size
     */
    ROCAL_SCALING_MODE_MIN_MAX = 4
};

/*! \brief rocAL Resize Interpolation Type enum
 * \ingroup group_rocal_types
 */
enum RocalResizeInterpolationType
{
    /*! \brief AMD ROCAL_NEAREST_NEIGHBOR_INTERPOLATION
     */
    ROCAL_NEAREST_NEIGHBOR_INTERPOLATION = 0,
    /*! \brief AMD ROCAL_LINEAR_INTERPOLATION
     */
    ROCAL_LINEAR_INTERPOLATION = 1,
    /*! \brief AMD ROCAL_CUBIC_INTERPOLATION
     */
    ROCAL_CUBIC_INTERPOLATION = 2,
    /*! \brief AMD ROCAL_LANCZOS_INTERPOLATION
     */
    ROCAL_LANCZOS_INTERPOLATION = 3,
    /*! \brief AMD ROCAL_GAUSSIAN_INTERPOLATION
     */
    ROCAL_GAUSSIAN_INTERPOLATION = 4,
    /*! \brief AMD ROCAL_TRIANGULAR_INTERPOLATION
     */
    ROCAL_TRIANGULAR_INTERPOLATION = 5
};

/*! \brief Tensor Backend
 * \ingroup group_rocal_types
 */
enum RocalTensorBackend {
    /*! \brief ROCAL_CPU
     */
    ROCAL_CPU = 0,
    /*! \brief ROCAL_GPU
     */
    ROCAL_GPU = 1
};

/*! \brief Tensor ROI type
 * \ingroup group_rocal_types
 */
enum class RocalROICordsType {
    /*! \brief ROCAL_LTRB
     */
    ROCAL_LTRB = 0,
    /*! \brief ROCAL_XYWH
     */
    ROCAL_XYWH = 1
};

/*! \brief RocalExternalSourceMode struct
 * \ingroup group_rocal_types
 */
enum RocalExternalSourceMode {
    /*! \brief list of filename passed as input
     */
    ROCAL_EXTSOURCE_FNAME = 0,
    /*! \brief compressed raw buffer passed as input
     */
    ROCAL_EXTSOURCE_RAW_COMPRESSED = 1,
    /*! \brief uncompressed raw buffer passed as input
     */
    ROCAL_EXTSOURCE_RAW_UNCOMPRESSED = 2,
};

/*! \brief rocAL Audio Border Type enum
 * \ingroup group_rocal_types
 */
enum RocalAudioBorderType {
    /*! \brief AMD ROCAL_ZERO
     */
    ROCAL_ZERO = 0,
    /*! \brief AMD ROCAL_CLAMP
     */
    ROCAL_CLAMP = 1,
    /*! \brief AMD ROCAL_REFLECT
     */
    ROCAL_REFLECT = 2
};

/*! \brief rocAL Out Of Bounds Policy Type enum
 * \ingroup group_rocal_types
 */
enum RocalOutOfBoundsPolicy {
    /*! \brief Pad
     */
    ROCAL_PAD = 0,
    /*! \brief Trimtoshape
     */
    ROCAL_TRIMTOSHAPE,
    /*! \brief Error
     */
    ROCAL_ERROR
};

/*! \brief rocAL MelScale formula enum
 * \ingroup group_rocal_types
 */
enum RocalMelScaleFormula {
    /*! \brief Slaney
     * Follows Slaney’s MATLAB Auditory Modelling Work behavior
     */
    ROCAL_MELSCALE_SLANEY = 0,
    /*! \brief HTK
     * Follows O’Shaughnessy’s book formula, consistent with Hidden Markov Toolkit(HTK), m = 2595 * log10(1 + (f/700))
     */
    ROCAL_MELSCALE_HTK
};

/*! \brief Tensor Last Batch Policy Type enum
 *  \ingroup group_rocal_types
 */
enum RocalLastBatchPolicy {
    /*! \brief ROCAL_LAST_BATCH_FILL - The last batch is filled by either repeating the last sample or by wrapping up the data set.
     */
    ROCAL_LAST_BATCH_FILL = 0,
    /*! \brief ROCAL_LAST_BATCH_DROP - The last batch is dropped if there are not enough samples from the current epoch.
     */
    ROCAL_LAST_BATCH_DROP = 1,
    /*! \brief ROCAL_LAST_BATCH_PARTIAL - The last batch is partially filled with the remaining data from the current epoch, keeping the rest of the samples empty. (currently this policy works similar to FILL in rocAL, PARTIAL policy needs to be handled in the python iterator)
     */
    ROCAL_LAST_BATCH_PARTIAL = 2
};

/*! \brief  rocAL RocalShardingInfo enum
 * \ingroup group_rocal_types
 */
struct RocalShardingInfo {
    RocalLastBatchPolicy last_batch_policy;
    bool pad_last_batch_repeated;
    bool stick_to_shard;
    int32_t shard_size;

    // Constructor with default values
    RocalShardingInfo()
        : last_batch_policy(RocalLastBatchPolicy::ROCAL_LAST_BATCH_FILL),
          pad_last_batch_repeated(false),
          stick_to_shard(true),
          shard_size(-1)
    {}

    // Constructor that initializes all members
    RocalShardingInfo(
        RocalLastBatchPolicy last_batch_policy,
        bool pad_last_batch_repeated,
        bool stick_to_shard,
        int shard_size
    )
        : last_batch_policy(last_batch_policy),
          pad_last_batch_repeated(pad_last_batch_repeated),
          stick_to_shard(stick_to_shard),
          shard_size(shard_size) {}
};

/*! \brief Missing components behaviour for Webdataset
 *  \ingroup group_rocal_types
 */
enum RocalMissingComponentsBehaviour {
    /*! \brief ROCAL_MISSING_COMPONENT_ERROR
     */
    ROCAL_MISSING_COMPONENT_ERROR = 0,
    /*! \brief ROCAL_MISSING_COMPONENT_SKIP
     */
    ROCAL_MISSING_COMPONENT_SKIP = 1,
    /*! \brief ROCAL_MISSING_COMPONENT_EMPTY
     */
    ROCAL_MISSING_COMPONENT_EMPTY = 2
};

#endif  // MIVISIONX_ROCAL_API_TYPES_H
