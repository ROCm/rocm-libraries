/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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
#pragma once
#include <vector>

#include "pipeline/commons.h"
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include "meta_data/meta_data.h"

// assume total number of boxes per batch for device memory allocation
#define MAX_NUM_BOXES_TOTAL 4096
#define HIP_ERROR_CHECK_STATUS(call)                                  \
    {                                                                 \
        hipError_t err = (call);                                      \
        if (err != hipSuccess) {                                      \
            THROW("ERROR: Hip call failed with status " + TOSTR(err)) \
        }                                                             \
    }

struct BoxEncoderSampleDesc {
    float4 *boxes_out;
    int *labels_out;
    const float4 *boxes_in;
    const int *labels_in;
    int in_box_count;
};

class BoxEncoderGpu {
   public:
    static constexpr int BlockSize = 256;  // hip kernel blocksize
    explicit BoxEncoderGpu(int batch_size, std::vector<float> &anchors, float criteria, const std::vector<float> &means, const std::vector<float> &stds, bool offset, float scale, const hipStream_t &stream, bool pinnedMem) : _anchors(anchors), _criteria(criteria), _means(means), _stds(stds), _offset(offset), _scale(scale), _stream(stream), _pinnedMem(pinnedMem) {
        if (criteria < 0.f || criteria > 1.f || means.size() != 4 || stds.size() != 4)
            THROW("BoxEncoder invalid input parameter");
        Initialize(batch_size);
        prepare_anchors(_anchors);
        prepare_mean_std(_means, _stds);
    }
    void Run(pMetaDataBatch full_batch_meta_data, float *encoded_boxes_data, int *encoded_labels_data);

    virtual ~BoxEncoderGpu() { UnInitialize(); };

   protected:
    void Initialize(int cur_batch_size) {
        _cur_batch_size = cur_batch_size;
        _anchor_count = _anchors.size() / 4;
        _best_box_idx.resize(cur_batch_size * _anchor_count);
        _best_box_iou.resize(cur_batch_size * _anchor_count);
        hipError_t err = hipHostMalloc((void **)&_samples_host_buf, cur_batch_size * sizeof(BoxEncoderSampleDesc), hipHostMallocDefault /*hipHostMallocMapped|hipHostMallocWriteCombined*/);
        if (err != hipSuccess || !_samples_host_buf) {
            THROW("hipHostMalloc failed for BoxEncoderSampleDesc" + TOSTR(err));
        }
        if (_pinnedMem)
            err = hipHostGetDevicePointer((void **)&_samples_dev_buf, _samples_host_buf, 0);
        else
            err = hipMalloc((void **)&_samples_dev_buf, cur_batch_size * sizeof(BoxEncoderSampleDesc));
        if (err != hipSuccess || !_samples_dev_buf) {
            THROW("hipMalloc failed for BoxEncoderSampleDesc" + TOSTR(err));
        }
        // allocate  _anchors_data_dev and _anchors_as_center_wh_data for device
        err = hipMalloc((void **)&_anchors_data_dev, _anchor_count * 4 * sizeof(float));
        err = hipMalloc((void **)&_anchors_as_center_wh_data_dev, _anchor_count * 4 * sizeof(float));
        err = hipMalloc((void **)&_means_dev, 4 * sizeof(float));
        err = hipMalloc((void **)&_stds_dev, 4 * sizeof(float));

        if (err != hipSuccess || !_anchors_data_dev || !_anchors_as_center_wh_data_dev || !_means_dev || !_stds_dev) {
            THROW("hipMalloc failed for BoxEncoderGPU" + TOSTR(err));
        }
        HIP_ERROR_CHECK_STATUS(hipMalloc(&_labels_in_dev, MAX_NUM_BOXES_TOTAL * sizeof(int)));
        HIP_ERROR_CHECK_STATUS(hipMalloc(&_boxes_in_dev, MAX_NUM_BOXES_TOTAL * sizeof(float) * 4));
        HIP_ERROR_CHECK_STATUS(hipMalloc(&_best_box_idx_dev, _best_box_idx.size() * sizeof(int)));
        HIP_ERROR_CHECK_STATUS(hipMalloc(&_best_box_iou_dev, _best_box_iou.size() * sizeof(float)));
    }

    void UnInitialize() {
        if (_samples_host_buf) HIP_ERROR_CHECK_STATUS(hipHostFree(_samples_host_buf));
        if (!_pinnedMem) HIP_ERROR_CHECK_STATUS(hipFree(_samples_dev_buf));
        if (_anchors_data_dev) HIP_ERROR_CHECK_STATUS(hipFree(_anchors_data_dev));
        if (_means_dev) HIP_ERROR_CHECK_STATUS(hipFree(_means_dev));
        if (_stds_dev) HIP_ERROR_CHECK_STATUS(hipFree(_stds_dev));
        if (_anchors_as_center_wh_data_dev) HIP_ERROR_CHECK_STATUS(hipFree(_anchors_as_center_wh_data_dev));
        if (_labels_in_dev) HIP_ERROR_CHECK_STATUS(hipFree(_labels_in_dev));
        if (_boxes_in_dev) HIP_ERROR_CHECK_STATUS(hipFree(_boxes_in_dev));
        if (_best_box_idx_dev) HIP_ERROR_CHECK_STATUS(hipFree(_best_box_idx_dev));
        if (_best_box_iou_dev) HIP_ERROR_CHECK_STATUS(hipFree(_best_box_iou_dev));
    }

   private:
    void prepare_anchors(const std::vector<float> &anchors);
    void prepare_mean_std(const std::vector<float> &means, const std::vector<float> &stds);
    void WriteAnchorsToOutput(float *encoded_boxes);
    void ResetLabels(int *encoded_labels_out);
    void ClearOutput(float *encoded_boxes);
    std::pair<int *, float *> ResetBuffers();

    int _cur_batch_size;
    std::vector<float> _anchors;
    const float _criteria;
    std::vector<float> _means;
    std::vector<float> _stds;
    bool _offset;
    float _scale;
    const hipStream_t _stream;
    bool _pinnedMem;

    int _anchor_count;
    int *_labels_in_dev;
    float *_boxes_in_dev;
    std::vector<int> _best_box_idx;
    std::vector<float> _best_box_iou;
    int *_best_box_idx_dev;
    float *_best_box_iou_dev;
    std::vector<BoxEncoderSampleDesc *> _samples;
    BoxEncoderSampleDesc *_samples_host_buf, *_samples_dev_buf;
    float4 *_anchors_data_dev, *_anchors_as_center_wh_data_dev;
    float *_means_dev, *_stds_dev;
    std::vector<std::vector<size_t>> _output_shape;
};