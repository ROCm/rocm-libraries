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


#include "pipeline/commons.h"
#include <stdio.h>
#include <string.h>
#include "decoders/image/fused_crop_decoder.h"
#include "decoders/libjpeg/libjpeg_extra.h"


FusedCropTJDecoder::FusedCropTJDecoder() {
    m_jpegDecompressor = tjInitDecompress();
};

Decoder::Status FusedCropTJDecoder::decode_info(unsigned char *input_buffer, size_t input_size, int *width, int *height, int *color_comps) {
    // TODO : Use the most recent TurboJpeg API tjDecompressHeader3 which returns the color components
    if (tjDecompressHeader2(m_jpegDecompressor,
                            input_buffer,
                            input_size,
                            width,
                            height,
                            color_comps) != 0) {
        WRN("Jpeg header decode failed " + STR(tjGetErrorStr2(m_jpegDecompressor)))
        return Status::HEADER_DECODE_FAILED;
    }
    return Status::OK;
}

Decoder::Status FusedCropTJDecoder::decode(unsigned char *input_buffer, size_t input_size, unsigned char *output_buffer,
                                           size_t max_decoded_width, size_t max_decoded_height,
                                           size_t original_image_width, size_t original_image_height,
                                           size_t &actual_decoded_width, size_t &actual_decoded_height,
                                           Decoder::ColorFormat desired_decoded_color_format, DecoderConfig decoder_config, bool keep_original_size) {
    int tjpf = TJPF_RGB;
    int planes = 1;
    switch (desired_decoded_color_format) {
        case Decoder::ColorFormat::GRAY:
            tjpf = TJPF_GRAY;
            planes = 1;
            break;
        case Decoder::ColorFormat::RGB:
            tjpf = TJPF_RGB;
            planes = 3;
            break;
        case Decoder::ColorFormat::BGR:
            tjpf = TJPF_BGR;
            planes = 3;
            break;
    };
    actual_decoded_width = max_decoded_width;
    actual_decoded_height = max_decoded_height;
    // You need get the output of random bbox crop
    // check the vector size for bounding box. If its more than zero go for random bbox crop
    // else go to random crop
    unsigned int x1_diff, crop_width_diff;
    if (_bbox_coord.size() != 0) {
        // Random bbox crop returns normalized crop cordinates
        // hence bringing it back to absolute cordinates
        _crop_window.x = std::lround(_bbox_coord[0] * original_image_width);
        _crop_window.y = std::lround(_bbox_coord[1] * original_image_height);
        _crop_window.W = std::lround((_bbox_coord[2]) * original_image_width);
        _crop_window.H = std::lround((_bbox_coord[3]) * original_image_height);
    }
    _crop_window.W = std::min(_crop_window.W, (unsigned int)max_decoded_width);
    _crop_window.H = std::min(_crop_window.H, (unsigned int)max_decoded_height);
    // TODO : Turbo Jpeg supports multiple color packing and color formats, add more as an option to the API TJPF_RGB, TJPF_BGR, TJPF_RGBX, TJPF_BGRX, TJPF_RGBA, TJPF_GRAY, TJPF_CMYK , ...
    if (tjDecompress2_partial(m_jpegDecompressor,
                              input_buffer,
                              input_size,
                              output_buffer,
                              max_decoded_width,
                              max_decoded_width * planes,
                              max_decoded_height,
                              tjpf,
                              TJFLAG_ACCURATEDCT, &x1_diff, &crop_width_diff,
                              _crop_window.x, _crop_window.y, _crop_window.W, _crop_window.H) != 0) {
        WRN("Jpeg image decode failed " + STR(tjGetErrorStr2(m_jpegDecompressor)))
        return Status::CONTENT_DECODE_FAILED;
    }

    // x1-diff should be set to x offset in tensor pipeline and removed.
    if (_crop_window.x != x1_diff) {
        unsigned char *src_ptr_temp, *dst_ptr_temp;
        unsigned int elements_in_row = max_decoded_width * planes;
        unsigned int elements_in_crop_row = _crop_window.W * planes;
        unsigned int xoffs = (_crop_window.x - x1_diff) * planes;  // in case _crop_window.x gets adjusted by tjpeg decoder
        src_ptr_temp = output_buffer;
        dst_ptr_temp = output_buffer;
        for (unsigned int i = 0; i < _crop_window.H; i++) {
            memcpy(dst_ptr_temp, src_ptr_temp + xoffs, elements_in_crop_row * sizeof(unsigned char));
            src_ptr_temp += elements_in_row;
            dst_ptr_temp += elements_in_row;
        }
    }
    actual_decoded_width = _crop_window.W;
    actual_decoded_height = _crop_window.H;

    return Status::OK;
}

FusedCropTJDecoder::~FusedCropTJDecoder() {
    tjDestroy(m_jpegDecompressor);
}
