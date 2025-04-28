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

/*
 * dtypes.h
 *
 *  Created on: Jan 29, 2019
 *      Author: root
 */

#pragma once
#include <random>
#include <vector>

#include "pipeline/exception.h"
#include "pipeline/log.h"
#include "pipeline/filesystem.h"

// Calculated from the largest resize shorter dimension in imagenet validation dataset
#define MAX_ASPECT_RATIO 6.0f

/*! \brief Tensor layouts
 *
 * currently supported by Rocal SDK as input/output
 */
enum class RocalTensorlayout {
    NHWC = 0,
    NCHW,
    NFHWC,
    NFCHW,
    NHW,
    NFT,
    NTF,
    NONE
};

/*! \brief Tensor data type
 *
 * currently supported by Rocal SDK as input/output
 */
enum class RocalTensorDataType {
    FP32 = 0,
    FP16,
    UINT8,
    INT8,
    UINT32,
    INT32,
    INT16
};

enum class RocalAffinity {
    GPU = 0,
    CPU
};

/*! \brief Color formats currently supported by Rocal SDK as input/output
 *
 */
enum class RocalColorFormat {
    RGB24 = 0,
    BGR24,
    U8,
    RGB_PLANAR,
};

/*! \brief Memory type, host or device
 *
 *  Currently supports HOST and OCL, will support HIP in future
 */
enum class RocalMemType {
    HOST = 0,
    OCL,
    HIP
};

/*! \brief Decoder mode for Video decoding
 *
 *  Currently supports Software decoding, will support Hardware decoding in future
 */
enum class DecodeMode {
    HW_VAAPI = 0,
    CPU
};

/*! \brief Tensor ROI type
 *
 * currently supports following formats
 */
enum class RocalROIType {
    LTRB = 0,
    XYWH
};

/*! \brief Tensor ROI in LTRB format
 *
 */
typedef struct {
    unsigned l, t, r, b;
} RoiLtrb;

/*! \brief Tensor ROI in XYWH format
 *
 */
typedef struct {
    unsigned x, y, w, h;
} RoiXywh;

/*! \brief Tensor ROI union
 *
 * Supports LTRB and XYWH formats
 */
typedef union {
    RoiLtrb ltrb;
    RoiXywh xywh;
} Roi2DCords;

/*! \brief Tensor ROI
 *
 * Points to the begin and end in the ROI for each data
 */
typedef struct {
    unsigned *begin;
    unsigned *end;
} RoiCords;

struct Timing {
    // The following timings are accumulated timing not just the most recent activity
    long long unsigned read_time = 0;
    long long unsigned decode_time = 0;
    long long unsigned to_device_xfer_time = 0;
    long long unsigned from_device_xfer_time = 0;
    long long unsigned copy_to_output = 0;
    long long unsigned process_time = 0;
    long long unsigned bb_process_time = 0;
    long long unsigned mask_process_time = 0;
    long long unsigned label_load_time = 0;
    long long unsigned bb_load_time = 0;
    long long unsigned mask_load_time = 0;
    long long unsigned video_read_time= 0;
    long long unsigned video_decode_time= 0;
    long long unsigned video_process_time= 0;
};

/*! \brief Tensor Last Batch Policy Type enum
 These policies the last batch policies determine the behavior when there are not enough samples in the epoch to fill the last batch
        FILL - The last batch is filled by either repeating the last sample or by wrapping up the data set.
        DROP - The last batch is dropped if it cannot be fully filled with data from the current epoch.
        PARTIAL - The last batch is partially filled with the remaining data from the current epoch, keeping the rest of the samples empty. (currently this policy works similar to FILL in rocAL, PARTIAL policy needs to be handled in the pytorch iterator)
 */
enum RocalBatchPolicy {
    FILL = 0,
    DROP,
    PARTIAL
};

template <typename RNG = std::mt19937>
class BatchRNG {
   public:
    /**
     * @brief Used to keep batch of RNGs
     *
     * @param seed Used to generate seed_seq to initialize batch of RNGs
     * @param batch_size How many RNGs to store
     * @param state_size How many seed are used to initialize one RNG. Used to
     * lower probablity of collisions between seeds used to initialize RNGs in
     * different operators.
     */
    BatchRNG(int64_t seed = 1, int batch_size = 1, int state_size = 4)
        : _seed(seed) {
        std::seed_seq seq{_seed};
        std::vector<uint32_t> seeds(batch_size * state_size);
        seq.generate(seeds.begin(), seeds.end());
        _rngs.reserve(batch_size);
        for (int i = 0; i < batch_size * state_size; i += state_size) {
            std::seed_seq s(seeds.begin() + i, seeds.begin() + i + state_size);
            _rngs.emplace_back(s);
        }
    }

    /**
     * Returns engine corresponding to given sample ID
     */
    RNG &operator[](int sample) noexcept { return _rngs[sample]; }

   private:
    int64_t _seed;
    std::vector<RNG> _rngs;
};

/*! \brief MissingComponentsBehaviour for Webdataset
 *
 */
enum MissingComponentsBehaviour {
    MISSING_COMPONENT_ERROR = 0,
    MISSING_COMPONENT_SKIP,
    MISSING_COMPONENT_EMPTY
};