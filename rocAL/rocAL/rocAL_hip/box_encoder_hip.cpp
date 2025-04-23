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

#include "box_encoder_hip.h"

// for both device and host
__host__ __device__ inline float4 ToBoxCenterWH(const float4 &box) {
    return {
        0.5f * (box.x + box.z),
        0.5f * (box.y + box.w),
        box.z - box.x,
        box.w - box.y};
}

__device__ float4 MatchOffsets(float4 box, float4 anchor, const float *means, const float *inv_stds, float scale) {
    float4 box_out;
    box.x *= scale;
    box.y *= scale;
    box.z *= scale;
    box.w *= scale;
    anchor.x *= scale;
    anchor.y *= scale;
    anchor.z *= scale;
    anchor.w *= scale;

    box_out.x = ((box.x - anchor.x) / anchor.z - means[0]) * inv_stds[0];
    box_out.y = ((box.y - anchor.y) / anchor.w - means[1]) * inv_stds[1];
    box_out.z = (log(box.z / anchor.z) - means[2]) * inv_stds[2];
    box_out.w = (log(box.w / anchor.w) - means[3]) * inv_stds[3];

    return box_out;
}

__device__ __forceinline__ float CalculateIou(const float4 &b1, const float4 &b2) {
    float l = fmaxf(b1.x, b2.x);
    float t = fmaxf(b1.y, b2.y);
    float r = fminf(b1.z, b2.z);
    float b = fminf(b1.w, b2.w);
    float first = fmaxf(r - l, 0.0f);
    float second = fmaxf(b - t, 0.0f);
    volatile float intersection = first * second;
    volatile float area1 = (b1.w - b1.y) * (b1.z - b1.x);
    volatile float area2 = (b2.w - b2.y) * (b2.z - b2.x);

    return intersection / (area1 + area2 - intersection);
}

__device__ inline void FindBestMatch(const int N, volatile float *vals, volatile int *idx) {
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (vals[threadIdx.x] <= vals[threadIdx.x + stride]) {
                if (vals[threadIdx.x] == vals[threadIdx.x + stride]) {
                    idx[threadIdx.x] = max(idx[threadIdx.x], idx[threadIdx.x + stride]);
                } else {
                    vals[threadIdx.x] = vals[threadIdx.x + stride];
                    idx[threadIdx.x] = idx[threadIdx.x + stride];
                }
            }
        }
        __syncthreads();
    }
}

__device__ void WriteMatchesToOutput(unsigned int anchor_count, float criteria, int *labels_out, const int *labels_in,
                                     float4 *boxes_out, const float4 *boxes_in, volatile int *best_box_idx, volatile float *best_box_iou,
                                     bool offset, const float *means, const float *inv_stds, float scale, const float4 *anchors_as_cwh) {
    for (unsigned int anchor = threadIdx.x; anchor < anchor_count; anchor += blockDim.x) {
        if (best_box_iou[anchor] > criteria) {
            int box_idx = best_box_idx[anchor];
            labels_out[anchor] = labels_in[box_idx];
            float4 box = boxes_in[box_idx];

            if (!offset)
                boxes_out[anchor] = ToBoxCenterWH(box);
            else
                boxes_out[anchor] = MatchOffsets(ToBoxCenterWH(box), anchors_as_cwh[anchor], means, inv_stds, scale);
        }
    }
}

__device__ void MatchBoxWithAnchors(const float4 &box, const int box_idx, unsigned int anchor_count, const float4 *anchors,
                                    volatile int *best_anchor_idx_buf, volatile float *best_anchor_iou_buf,
                                    volatile int *best_box_idx, volatile float *best_box_iou) {
    float best_anchor_iou = -1.0f;
    int best_anchor_idx = -1;

    for (unsigned int anchor = threadIdx.x; anchor < anchor_count; anchor += blockDim.x) {
        float new_val = CalculateIou(box, anchors[anchor]);

        if (new_val >= best_anchor_iou) {
            best_anchor_iou = new_val;
            best_anchor_idx = anchor;
        }
        if (new_val >= best_box_iou[anchor]) {
            best_box_iou[anchor] = new_val;
            best_box_idx[anchor] = box_idx;
        }
    }

    best_anchor_iou_buf[threadIdx.x] = best_anchor_iou;
    best_anchor_idx_buf[threadIdx.x] = best_anchor_idx;
}

template <int BLOCK_SIZE>
__global__ void __attribute__((visibility("default")))
BoxEncode(const BoxEncoderSampleDesc *samples, const int anchor_cnt, const float4 *anchors,
          const float criteria, int *box_idx_buffer, float *box_iou_buffer, bool offset,
          const float *means, const float *inv_stds, float scale, const float4 *anchors_as_cwh) {
    const int sample_idx = blockIdx.x;
    const auto &sample = samples[sample_idx];

    __shared__ volatile int best_anchor_idx_buf[BLOCK_SIZE];
    __shared__ volatile float best_anchor_iou_buf[BLOCK_SIZE];

    volatile int *best_box_idx = box_idx_buffer + sample_idx * anchor_cnt;
    volatile float *best_box_iou = box_iou_buffer + sample_idx * anchor_cnt;

    for (int box_idx = 0; box_idx < sample.in_box_count; ++box_idx) {
        MatchBoxWithAnchors(
            sample.boxes_in[box_idx],
            box_idx,
            anchor_cnt,
            anchors,
            best_anchor_idx_buf,
            best_anchor_iou_buf,
            best_box_idx,
            best_box_iou);

        __syncthreads();

        FindBestMatch(blockDim.x, best_anchor_iou_buf, best_anchor_idx_buf);
        __syncthreads();

        if (threadIdx.x == 0) {
            int idx = best_anchor_idx_buf[0];
            best_box_idx[idx] = box_idx;
            best_box_iou[idx] = 2.f;
        }
        __syncthreads();
    }
    __syncthreads();

    WriteMatchesToOutput(
        anchor_cnt,
        criteria,
        sample.labels_out,
        sample.labels_in,
        sample.boxes_out,
        sample.boxes_in,
        best_box_idx,
        best_box_iou,
        offset,
        means,
        inv_stds,
        scale,
        anchors_as_cwh);
}

void BoxEncoderGpu::prepare_anchors(const std::vector<float> &anchors) {
    if ((anchors.size() % 4) != 0)
        THROW("BoxEncoderGpu anchors not a multiple of 4");

    int anchor_count = anchors.size() / 4;
    int anchor_data_size = anchor_count * 4 * sizeof(float);
    auto anchors_data_cpu = reinterpret_cast<const float4 *>(anchors.data());

    std::vector<float4> anchors_as_center_wh(anchor_count);
    for (unsigned int anchor = 0; anchor < anchor_count; ++anchor)
        anchors_as_center_wh[anchor] = ToBoxCenterWH(anchors_data_cpu[anchor]);

    HIP_ERROR_CHECK_STATUS(hipMemcpy((void *)_anchors_data_dev, anchors.data(), anchor_data_size, hipMemcpyHostToDevice));
    HIP_ERROR_CHECK_STATUS(hipMemcpy((void *)_anchors_as_center_wh_data_dev, anchors_as_center_wh.data(), anchor_data_size, hipMemcpyHostToDevice));
}

void BoxEncoderGpu::prepare_mean_std(const std::vector<float> &means, const std::vector<float> &stds) {
    int data_size = 4 * sizeof(float);
    auto means_data_cpu = reinterpret_cast<const float *>(means.data());
    auto stds_data_cpu = reinterpret_cast<const float *>(stds.data());

    HIP_ERROR_CHECK_STATUS(hipMemcpy((void *)_means_dev, means_data_cpu, data_size, hipMemcpyHostToDevice));
    HIP_ERROR_CHECK_STATUS(hipMemcpy((void *)_stds_dev, stds_data_cpu, data_size, hipMemcpyHostToDevice));
}

void BoxEncoderGpu::WriteAnchorsToOutput(float *encoded_boxes) {
    // Device -> device copy for all the samples
    for (int i = 0; i < _cur_batch_size; i++) {
        HIP_ERROR_CHECK_STATUS(hipMemcpyDtoDAsync((void *)(encoded_boxes + i * _anchor_count * 4), _anchors_as_center_wh_data_dev,
                                                  _anchor_count * 4 * sizeof(float), _stream));
    }
}

std::pair<int *, float *> BoxEncoderGpu::ResetBuffers() {
    HIP_ERROR_CHECK_STATUS(hipMemsetAsync(_best_box_idx_dev, 0, _cur_batch_size * _anchor_count * sizeof(int), _stream));
    HIP_ERROR_CHECK_STATUS(hipMemsetAsync(_best_box_iou_dev, 0, _cur_batch_size * _anchor_count * sizeof(float), _stream));
    return std::make_pair(_best_box_idx_dev, _best_box_iou_dev);
}

void BoxEncoderGpu::ResetLabels(int *encoded_labels_out) {
    HIP_ERROR_CHECK_STATUS(hipMemsetAsync((void *)encoded_labels_out, 0, _cur_batch_size * _anchor_count * sizeof(int), _stream));
}

void BoxEncoderGpu::ClearOutput(float *encoded_boxes) {
    HIP_ERROR_CHECK_STATUS(hipMemsetAsync(encoded_boxes, 0, _cur_batch_size * _anchor_count * 4 * sizeof(float), _stream));
}

void BoxEncoderGpu::Run(pMetaDataBatch full_batch_meta_data, float *encoded_boxes_data, int *encoded_labels_data) {
    if (_cur_batch_size != full_batch_meta_data->size() || (_cur_batch_size <= 0))
        THROW("BoxEncoderGpu::Run Invalid input metadata");
    const auto buffers = ResetBuffers();  // reset temp buffers
                                          //    auto dims = CalculateDims(boxes_input);     // todo:: if we store output in tensorlist
    int total_num_boxes = 0;
    for (int i = 0; i < _cur_batch_size; i++) {
        auto sample = &_samples_host_buf[i];
        sample->in_box_count = full_batch_meta_data->get_labels_batch()[i].size();
        total_num_boxes += sample->in_box_count;
    }
    if (total_num_boxes > MAX_NUM_BOXES_TOTAL)
        THROW("BoxEncoderGpu::Run total_num_boxes exceeds max");
    float *boxes_in_temp = _boxes_in_dev;
    int *labels_in_temp = _labels_in_dev;
    for (int sample_idx = 0; sample_idx < _cur_batch_size; sample_idx++) {
        auto sample = &_samples_host_buf[sample_idx];
        HIP_ERROR_CHECK_STATUS(hipMemcpyHtoDAsync((void *)boxes_in_temp, full_batch_meta_data->get_bb_cords_batch()[sample_idx].data(), sample->in_box_count * sizeof(float) * 4, _stream));
        HIP_ERROR_CHECK_STATUS(hipMemcpyHtoDAsync((void *)labels_in_temp, full_batch_meta_data->get_labels_batch()[sample_idx].data(), sample->in_box_count * sizeof(int), _stream));
        sample->boxes_in = reinterpret_cast<const float4 *>(boxes_in_temp);
        sample->labels_in = reinterpret_cast<const int *>(labels_in_temp);
        sample->boxes_out = reinterpret_cast<float4 *>(encoded_boxes_data + sample_idx * _anchor_count * 4);
        sample->labels_out = reinterpret_cast<int *>(encoded_labels_data + sample_idx * _anchor_count);
        boxes_in_temp += sample->in_box_count * 4;
        labels_in_temp += sample->in_box_count;
        _output_shape.push_back(std::vector<size_t>(1, _anchor_count));
    }
    const auto means_data = reinterpret_cast<const float *>(_means.data());
    const auto stds_data = reinterpret_cast<const float *>(_stds.data());

    ResetLabels(encoded_labels_data);  // sets all labels to zero for the output: true if criteria is not matched.
    if (!_offset)
        WriteAnchorsToOutput(encoded_boxes_data);
    else
        ClearOutput(encoded_boxes_data);
    // if there is no mapped memory, do explicit copy from host to device
    if (!_pinnedMem)
        HIP_ERROR_CHECK_STATUS(hipMemcpyHtoD(_samples_dev_buf, _samples_host_buf, _cur_batch_size * sizeof(BoxEncoderSampleDesc)));
    HIP_ERROR_CHECK_STATUS(hipStreamSynchronize(_stream));

    // call the kernel for box encoding
    hipLaunchKernelGGL(BoxEncode<BlockSize>, dim3(_cur_batch_size), dim3(BlockSize), 0, _stream,
                       _samples_dev_buf,
                       _anchor_count,
                       _anchors_data_dev,
                       _criteria,
                       buffers.first,
                       buffers.second,
                       _offset,
                       _means_dev,
                       _stds_dev,
                       _scale,
                       _anchors_as_center_wh_data_dev);
}