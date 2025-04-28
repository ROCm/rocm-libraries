/*
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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


#include "pipeline/commons.h"
#include <stdio.h>
#include <string.h>
#include "decoders/image/rocjpeg_decoder.h"

#if ENABLE_ROCJPEG

#include "hip/hip_runtime_api.h"
#include "hip/hip_runtime.h"
#include "rocal_hip_kernels.h"

#define CHECK_HIP(call) {                                             \
    hipError_t hip_status = (call);                                   \
    if (hip_status != hipSuccess) {                                   \
        std::cerr << "HIP failure: 'status: " << hipGetErrorName(hip_status) << "' at " << __FILE__ << ":" << __LINE__ << std::endl;\
        exit(1);                                                      \
    }                                                                 \
}

#define CHECK_ROCJPEG(call) {                                             \
    RocJpegStatus rocjpeg_status = (call);                                \
    if (rocjpeg_status != ROCJPEG_STATUS_SUCCESS) {                       \
        std::cerr << #call << " returned " << rocJpegGetErrorName(rocjpeg_status) << " at " <<  __FILE__ << ":" << __LINE__ << std::endl;\
        exit(1);                                                        \
    }                                                                     \
}

HWRocJpegDecoder::HWRocJpegDecoder() {
};

/**
 * @brief Gets the channel pitch and sizes.
 *
 * This function gets the channel pitch and sizes based on the specified output format, chroma subsampling,
 * output image, and channel sizes.
 *
 * @param output_format The output format.
 * @param subsampling The chroma subsampling.
 * @param widths The array to store the channel widths.
 * @param heights The array to store the channel heights.
 * @param num_channels The number of channels.
 * @param output_image The output image.
 * @param channel_sizes The array to store the channel sizes.
 * @return The channel pitch.
 */
int GetChannelPitchAndSizes(RocJpegDecodeParams decode_params, RocJpegChromaSubsampling subsampling, uint32_t *widths, uint32_t *heights,
                            uint32_t &num_channels, RocJpegImage &output_image, uint32_t *channel_sizes) {
    bool is_roi_valid = false;
    uint32_t roi_width = decode_params.crop_rectangle.right - decode_params.crop_rectangle.left;
    uint32_t roi_height = decode_params.crop_rectangle.bottom - decode_params.crop_rectangle.top;
    if (roi_width > 0 && roi_height > 0 && roi_width <= widths[0] && roi_height <= heights[0]) {
        is_roi_valid = true; 
    }

    switch (decode_params.output_format) {
        case ROCJPEG_OUTPUT_NATIVE:
            switch (subsampling) {
                case ROCJPEG_CSS_444:
                    num_channels = 3;
                    output_image.pitch[2] = output_image.pitch[1] = output_image.pitch[0] = is_roi_valid ? roi_width : widths[0];
                    channel_sizes[2] = channel_sizes[1] = channel_sizes[0] = output_image.pitch[0] * (is_roi_valid ? roi_height : heights[0]);
                    break;
                case ROCJPEG_CSS_440:
                    num_channels = 3;
                    output_image.pitch[2] = output_image.pitch[1] = output_image.pitch[0] = is_roi_valid ? roi_width : widths[0];
                    channel_sizes[0] = output_image.pitch[0] * (is_roi_valid ? roi_height : heights[0]);
                    channel_sizes[2] = channel_sizes[1] = output_image.pitch[0] * ((is_roi_valid ? roi_height : heights[0]) >> 1);
                    break;
                case ROCJPEG_CSS_422:
                    num_channels = 1;
                    output_image.pitch[0] = (is_roi_valid ? roi_width : widths[0]) * 2;
                    channel_sizes[0] = output_image.pitch[0] * (is_roi_valid ? roi_height : heights[0]);
                    break;
                case ROCJPEG_CSS_420:
                    num_channels = 2;
                    output_image.pitch[1] = output_image.pitch[0] = is_roi_valid ? roi_width : widths[0];
                    channel_sizes[0] = output_image.pitch[0] * (is_roi_valid ? roi_height : heights[0]);
                    channel_sizes[1] = output_image.pitch[1] * ((is_roi_valid ? roi_height : heights[0]) >> 1);
                    break;
                case ROCJPEG_CSS_400:
                    num_channels = 1;
                    output_image.pitch[0] = is_roi_valid ? roi_width : widths[0];
                    channel_sizes[0] = output_image.pitch[0] * (is_roi_valid ? roi_height : heights[0]);
                    break;
                default:
                    std::cout << "Unknown chroma subsampling!" << std::endl;
                    return EXIT_FAILURE;
            }
            break;
        case ROCJPEG_OUTPUT_YUV_PLANAR:
            if (subsampling == ROCJPEG_CSS_400) {
                num_channels = 1;
                output_image.pitch[0] = is_roi_valid ? roi_width : widths[0];
                channel_sizes[0] = output_image.pitch[0] * (is_roi_valid ? roi_height : heights[0]);
            } else {
                num_channels = 3;
                output_image.pitch[0] = is_roi_valid ? roi_width : widths[0];
                output_image.pitch[1] = is_roi_valid ? roi_width : widths[1];
                output_image.pitch[2] = is_roi_valid ? roi_width : widths[2];
                channel_sizes[0] = output_image.pitch[0] * (is_roi_valid ? roi_height : heights[0]);
                channel_sizes[1] = output_image.pitch[1] * (is_roi_valid ? roi_height : heights[1]);
                channel_sizes[2] = output_image.pitch[2] * (is_roi_valid ? roi_height : heights[2]);
            }
            break;
        case ROCJPEG_OUTPUT_Y:
            num_channels = 1;
            output_image.pitch[0] = is_roi_valid ? roi_width : widths[0];
            channel_sizes[0] = output_image.pitch[0] * (is_roi_valid ? roi_height : heights[0]);
            break;
        case ROCJPEG_OUTPUT_RGB:
            num_channels = 1;
            output_image.pitch[0] = (is_roi_valid ? roi_width : widths[0]) * 3;
            channel_sizes[0] = output_image.pitch[0] * (is_roi_valid ? roi_height : heights[0]);
            break;
        case ROCJPEG_OUTPUT_RGB_PLANAR:
            num_channels = 3;
            output_image.pitch[2] = output_image.pitch[1] = output_image.pitch[0] = is_roi_valid ? roi_width : widths[0];
            channel_sizes[2] = channel_sizes[1] = channel_sizes[0] = output_image.pitch[0] * (is_roi_valid ? roi_height : heights[0]);
            break;
        default:
            std::cout << "Unknown output format!" << std::endl;
            return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

/**
 * @brief Gets the chroma subsampling string.
 *
 * This function gets the chroma subsampling string based on the specified subsampling value.
 *
 * @param subsampling The chroma subsampling value.
 * @param chroma_sub_sampling The string to store the chroma subsampling.
 */
void GetChromaSubsamplingStr(RocJpegChromaSubsampling subsampling, std::string &chroma_sub_sampling) {
    switch (subsampling) {
        case ROCJPEG_CSS_444:
            chroma_sub_sampling = "YUV 4:4:4";
            break;
        case ROCJPEG_CSS_440:
            chroma_sub_sampling = "YUV 4:4:0";
            break;
        case ROCJPEG_CSS_422:
            chroma_sub_sampling = "YUV 4:2:2";
            break;
        case ROCJPEG_CSS_420:
            chroma_sub_sampling = "YUV 4:2:0";
            break;
        case ROCJPEG_CSS_411:
            chroma_sub_sampling = "YUV 4:1:1";
            break;
        case ROCJPEG_CSS_400:
            chroma_sub_sampling = "YUV 4:0:0";
            break;
        case ROCJPEG_CSS_UNKNOWN:
            chroma_sub_sampling = "UNKNOWN";
            break;
        default:
            chroma_sub_sampling = "";
            break;
    }
}

void HWRocJpegDecoder::initialize(int device_id, unsigned batch_size) {
    int num_devices;
    CHECK_HIP(hipGetDeviceCount(&num_devices));
    if (num_devices < 1) {
        std::cerr << "ERROR: didn't find any GPU!" << std::endl;
        return;
    }
    if (device_id >= num_devices) {
        std::cerr << "ERROR: the requested device_id is not found!" << std::endl;
        return;
    }
    CHECK_HIP(hipSetDevice(device_id));
    RocJpegBackend rocjpeg_backend = ROCJPEG_BACKEND_HARDWARE;
    // Create stream and handle
    CHECK_ROCJPEG(rocJpegCreate(rocjpeg_backend, device_id, &_rocjpeg_handle));
    _rocjpeg_streams.resize(batch_size);
    for (unsigned i = 0; i < batch_size; i++) {
        CHECK_ROCJPEG(rocJpegStreamCreate(&_rocjpeg_streams[i]));
    }

    _device_id = device_id;
    _batch_size = batch_size;
    _output_images.resize(_batch_size);
    _src_hstride.resize(_batch_size);
    _src_img_offset.resize(_batch_size);
    _decode_params.resize(_batch_size);

    // Allocate mem for width and height arrays for src and dst
    if (!_dev_src_width) CHECK_HIP(hipMalloc((void **)&_dev_src_width, _batch_size * sizeof(size_t)));
    if (!_dev_src_height) CHECK_HIP(hipMalloc((void **)&_dev_src_height, _batch_size * sizeof(size_t)));
    if (!_dev_dst_width) CHECK_HIP(hipMalloc((void **)&_dev_dst_width, _batch_size * sizeof(size_t)));
    if (!_dev_dst_height) CHECK_HIP(hipMalloc((void **)&_dev_dst_height, _batch_size * sizeof(size_t)));
    if (!_dev_src_hstride) CHECK_HIP(hipMalloc((void **)&_dev_src_hstride, _batch_size * sizeof(size_t)));
    if (!_dev_src_img_offset) CHECK_HIP(hipMalloc((void **)&_dev_src_img_offset, _batch_size * sizeof(size_t)));

}

// Obtains the decode info of the image, and modifies width and height based on the max decode params after scaling
Decoder::Status HWRocJpegDecoder::decode_info(unsigned char *input_buffer, size_t input_size, int *width, int *height, int *actual_width, 
                                              int *actual_height, int max_decoded_width, int max_decoded_height, Decoder::ColorFormat desired_decoded_color_format, int index) {
    RocJpegChromaSubsampling subsampling;
    uint8_t num_components;
    uint32_t widths[4] = {};
    uint32_t heights[4] = {};
    uint32_t channels_size = 0;
    uint32_t channel_sizes[4] = {};

    uint32_t max_widths[4] = {static_cast<uint32_t>(max_decoded_width), 0, 0, 0};
    uint32_t max_heights[4] = {static_cast<uint32_t>(max_decoded_height), 0, 0, 0};

    switch(desired_decoded_color_format) {
        case Decoder::ColorFormat::GRAY:
            _decode_params[index].output_format = ROCJPEG_OUTPUT_Y;
            _num_channels = 1;
            break;
        case Decoder::ColorFormat::RGB:
        case Decoder::ColorFormat::BGR:
            _decode_params[index].output_format = ROCJPEG_OUTPUT_RGB;
            _num_channels = 3;
            break;
    };

    if (rocJpegStreamParse(reinterpret_cast<uint8_t*>(input_buffer), input_size, _rocjpeg_streams[index]) != ROCJPEG_STATUS_SUCCESS) {
        return Status::HEADER_DECODE_FAILED;
    }
    if (rocJpegGetImageInfo(_rocjpeg_handle, _rocjpeg_streams[index], &num_components, &subsampling, widths, heights) != ROCJPEG_STATUS_SUCCESS) {
        return Status::HEADER_DECODE_FAILED;
    }

    if (widths[0] < 64 || heights[0] < 64) {
        return Status::CONTENT_DECODE_FAILED;
    }

    std::string chroma_sub_sampling = "";
    GetChromaSubsamplingStr(subsampling, chroma_sub_sampling);
    if (subsampling == ROCJPEG_CSS_411 || subsampling == ROCJPEG_CSS_UNKNOWN) {
        return Status::UNSUPPORTED;
    }

    if (width) *width = widths[0];
    if (height) *height = heights[0];
    uint scaledw = widths[0], scaledh = heights[0];
    // Scaling to be performed if width/height is greater than max decode width/height
    if (widths[0] > max_decoded_width || heights[0] > max_decoded_height) {
        for (unsigned j = 0; j < _num_scaling_factors; j++) {
            scaledw = (((widths[0]) * _scaling_factors[j].num + _scaling_factors[j].denom - 1) / _scaling_factors[j].denom);
            scaledh = (((heights[0]) * _scaling_factors[j].num + _scaling_factors[j].denom - 1) / _scaling_factors[j].denom);
            if (scaledw <= max_decoded_width && scaledh <= max_decoded_height)
                break;
        }
    }
    // If scaled width is different than original width and height, update max dims with the original width and height, to be used for decoding
    if (scaledw != widths[0] || scaledh != heights[0]) {
        _resize_batch = true;   // If the size of any image in the batch is greater than max size, resize the complete batch
        max_widths[0] = (widths[0] + 8) &~ 7;
        max_heights[0] = (heights[0] + 8) &~ 7;
    }

    if (GetChannelPitchAndSizes(_decode_params[index], subsampling, max_widths, max_heights, channels_size, _output_images[index], channel_sizes)) {
        return Status::HEADER_DECODE_FAILED;
    }
    if (actual_width) *actual_width = scaledw;
    if (actual_height) *actual_height = scaledh;

    _rocjpeg_image_buff_size += max_widths[0] * max_heights[0];

    return Status::OK;
}

Decoder::Status HWRocJpegDecoder::decode_info(unsigned char *input_buffer, size_t input_size, int *width, int *height, int *color_comps) {
    RocJpegChromaSubsampling subsampling;
    uint8_t num_components;
    uint32_t widths[4] = {};
    uint32_t heights[4] = {};
    if (rocJpegStreamParse(reinterpret_cast<uint8_t*>(input_buffer), input_size, _rocjpeg_streams[0]) != ROCJPEG_STATUS_SUCCESS) {
        return Status::HEADER_DECODE_FAILED;
    }
    if (rocJpegGetImageInfo(_rocjpeg_handle, _rocjpeg_streams[0], &num_components, &subsampling, widths, heights) != ROCJPEG_STATUS_SUCCESS) {
        return Status::HEADER_DECODE_FAILED;
    }
    *width = widths[0];
    *height = heights[0];
    _rocjpeg_image_buff_size += (((widths[0] + 8) &~ 7) * ((heights[0] + 8) &~ 7));

    if (widths[0] < 64 || heights[0] < 64) {
        return Status::CONTENT_DECODE_FAILED;
    }

    std::string chroma_sub_sampling = "";
    GetChromaSubsamplingStr(subsampling, chroma_sub_sampling);
    if (subsampling == ROCJPEG_CSS_411 || subsampling == ROCJPEG_CSS_UNKNOWN) {
        return Status::UNSUPPORTED;
    }
    return Status::OK;
}

Decoder::Status HWRocJpegDecoder::decode_batch(std::vector<unsigned char *> &output_buffer,
                                               size_t max_decoded_width, size_t max_decoded_height,
                                               std::vector<size_t> original_image_width, std::vector<size_t> original_image_height,
                                               std::vector<size_t> &actual_decoded_width, std::vector<size_t> &actual_decoded_height) {


    if (_resize_batch) {
        // Allocate memory for the itermediate decoded output
        _rocjpeg_image_buff_size *= _num_channels;
        if (!_rocjpeg_image_buff) {
            CHECK_HIP(hipMalloc((void **)&_rocjpeg_image_buff, _rocjpeg_image_buff_size));
            _prev_image_buff_size = _rocjpeg_image_buff_size;
        } else if (_rocjpeg_image_buff_size > _prev_image_buff_size) {  // Reallocate if the intermediate output exceeds the allocated memory
            CHECK_HIP(hipFree((void *)_rocjpeg_image_buff));
            CHECK_HIP(hipMalloc((void **)&_rocjpeg_image_buff, _rocjpeg_image_buff_size));
            _prev_image_buff_size = _rocjpeg_image_buff_size;
        }

        uint8_t *img_buff = reinterpret_cast<uint8_t*>(_rocjpeg_image_buff);
        size_t src_offset = 0;

        // Update RocJpegImage with the pointer
        for (unsigned i = 0; i < _batch_size; i++) {
                _output_images[i].channel[0] = static_cast<uint8_t *>(img_buff);    // For RGB
                _src_img_offset[i] = src_offset;
                unsigned pitch_width = (original_image_width[i] + 8) &~ 7;
                unsigned pitch_height = (original_image_height[i] + 8) &~ 7;
                src_offset += (pitch_width * pitch_height * _num_channels);
                img_buff += (pitch_width * pitch_height * _num_channels);
                _src_hstride[i] = pitch_width * _num_channels;
        }

        // Copy width and height args to HIP memory
        CHECK_HIP(hipMemcpyHtoD((void *)_dev_src_width, original_image_width.data(), _batch_size * sizeof(size_t)));
        CHECK_HIP(hipMemcpyHtoD((void *)_dev_src_height, original_image_height.data(), _batch_size * sizeof(size_t)));
        CHECK_HIP(hipMemcpyHtoD((void *)_dev_dst_width, actual_decoded_width.data(), _batch_size * sizeof(size_t)));
        CHECK_HIP(hipMemcpyHtoD((void *)_dev_dst_height, actual_decoded_height.data(), _batch_size * sizeof(size_t)));
        CHECK_HIP(hipMemcpyHtoD((void *)_dev_src_hstride, _src_hstride.data(), _batch_size * sizeof(size_t)));
        CHECK_HIP(hipMemcpyHtoD((void *)_dev_src_img_offset, _src_img_offset.data(), _batch_size * sizeof(size_t)));
    } else {
        for (unsigned i = 0; i < _batch_size; i++) {
            _output_images[i].channel[0] = static_cast<uint8_t *>(output_buffer[i]);    // For RGB
        }
    }

    CHECK_ROCJPEG(rocJpegDecodeBatched(_rocjpeg_handle, _rocjpeg_streams.data(), _batch_size, _decode_params.data(), _output_images.data()));

    if (_resize_batch) {
        HipExecResizeTensor(_hip_stream, (void *)_rocjpeg_image_buff, (void *)output_buffer[0], 
                            _batch_size, _dev_src_width, _dev_src_height, 
                            _dev_dst_width, _dev_dst_height, _dev_src_hstride, _dev_src_img_offset, _num_channels,
                            max_decoded_width, max_decoded_height, max_decoded_width, max_decoded_height);
    }
    _resize_batch = false;  // Need to reset this value for every batch

    return Status::OK;
}

HWRocJpegDecoder::~HWRocJpegDecoder() {
    CHECK_ROCJPEG(rocJpegDestroy(_rocjpeg_handle));
    for (auto j = 0; j < _batch_size; j++) {
        CHECK_ROCJPEG(rocJpegStreamDestroy(_rocjpeg_streams[j]));
    }
    if (_rocjpeg_image_buff) CHECK_HIP(hipFree(_rocjpeg_image_buff));
    if (_dev_src_width) CHECK_HIP(hipFree(_dev_src_width));
    if (_dev_src_height) CHECK_HIP(hipFree(_dev_src_height));
    if (_dev_dst_width) CHECK_HIP(hipFree(_dev_dst_width));
    if (_dev_dst_height) CHECK_HIP(hipFree(_dev_dst_height));
    if (_dev_src_hstride) CHECK_HIP(hipFree(_dev_src_hstride));
    if (_dev_src_img_offset) CHECK_HIP(hipFree(_dev_src_img_offset));
}
#endif
