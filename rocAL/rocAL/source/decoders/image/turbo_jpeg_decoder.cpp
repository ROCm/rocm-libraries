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


#include <stdio.h>
#include "pipeline/commons.h"
#include "decoders/image/turbo_jpeg_decoder.h"
#include "decoders/libjpeg/libjpeg_extra.h"

TJDecoder::TJDecoder() {
    m_jpegDecompressor = tjInitDecompress();
    if ((_scaling_factors = tjGetScalingFactors(&_num_scaling_factors)) == NULL)
        THROW("tjDecompress2_partial_scale(): error getting scaling factors");
    for(int i = 0; i < _num_scaling_factors; i++) {
        if(_scaling_factors[i].num < _scaling_factors[i].denom) {
            INFO(STR(_scaling_factors[i].num) + "/" + STR(_scaling_factors[i].denom));
        }
    }
};

Decoder::Status TJDecoder::decode_info(unsigned char* input_buffer, size_t input_size, int* width, int* height, int* color_comps) {
    // TODO : Use the most recent TurboJpeg API tjDecompressHeader3 which returns the color components
    if (tjDecompressHeader2(m_jpegDecompressor,
                            input_buffer,
                            input_size,
                            width,
                            height,
                            color_comps) != 0) {
        // ignore "Could not determine Subsampling type error"
        if (STR(tjGetErrorStr2(m_jpegDecompressor)).find("Could not determine subsampling type for JPEG image") == std::string::npos) {
            WRN("Jpeg header decode failed " + STR(tjGetErrorStr2(m_jpegDecompressor)))
            return Status::HEADER_DECODE_FAILED;
        }
    }
    return Status::OK;
}

Decoder::Status TJDecoder::decode(unsigned char* input_buffer, size_t input_size, unsigned char* output_buffer,
                                  size_t max_decoded_width, size_t max_decoded_height,
                                  size_t original_image_width, size_t original_image_height,
                                  size_t& actual_decoded_width, size_t& actual_decoded_height,
                                  Decoder::ColorFormat desired_decoded_color_format, DecoderConfig config, bool keep_original_size) {
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

    if (!keep_original_size) {
        actual_decoded_width = max_decoded_width;
        actual_decoded_height = max_decoded_height;
        // todo:: if the scaling factor requested is more that what the decoder can support, do cropped decoding
        if (original_image_width > (_max_scaling_factor * max_decoded_width) || original_image_height > (_max_scaling_factor * max_decoded_height)) {
            // Initializing the crop width and height to the original width and height of image to avoid cropping
            unsigned int crop_width = original_image_width;
            unsigned int crop_height = original_image_height;
            float in_ratio = static_cast<float>(original_image_width) / original_image_height;
            if (original_image_width > (_max_scaling_factor * max_decoded_width)) {
                crop_width = _max_scaling_factor * max_decoded_width;
                if (crop_width > original_image_width) crop_width = original_image_width;
                crop_height = crop_width * (1.0 / in_ratio);
                if (crop_height > _max_scaling_factor * max_decoded_height) crop_height = _max_scaling_factor * max_decoded_height;
            } else if (original_image_height > (_max_scaling_factor * max_decoded_height)) {
                crop_height = _max_scaling_factor * max_decoded_height;
                if (crop_height > original_image_height) crop_height = original_image_height;
                crop_width = crop_height * in_ratio;
                if (crop_width > _max_scaling_factor * max_decoded_width) crop_width = _max_scaling_factor * max_decoded_width;
            }
            if (tjDecompress2_partial_scale(m_jpegDecompressor,
                                            input_buffer,
                                            input_size,
                                            output_buffer,
                                            max_decoded_width,
                                            max_decoded_width * planes,
                                            max_decoded_height,
                                            tjpf,
                                            TJFLAG_ACCURATEDCT,
                                            crop_width, crop_height) != 0)

            {
                WRN("Jpeg partial image decode failed " + STR(tjGetErrorStr2(m_jpegDecompressor)))
                return Status::CONTENT_DECODE_FAILED;
            }
            // Find the decoded image size using the predefined scaling factors in the turbo jpeg decoder
            uint scaledw = max_decoded_width, scaledh = max_decoded_height;
            for (int j=0; j < _num_scaling_factors; j++) {
                scaledw = TJSCALED(original_image_width, _scaling_factors[j]);
                scaledh = TJSCALED(original_image_height, _scaling_factors[j]);
                if (scaledw <= max_decoded_width && scaledh <= max_decoded_height) {
                    break;
                }
            }
            actual_decoded_width = scaledw;
            actual_decoded_height = scaledh;
            // std::cout << "actual_decoded_width: " << actual_decoded_width << " actual_decoded_height: " << actual_decoded_height  << std::endl;
        }
        // TODO : Turbo Jpeg supports multiple color packing and color formats, add more as an option to the API TJPF_RGB, TJPF_BGR, TJPF_RGBX, TJPF_BGRX, TJPF_RGBA, TJPF_GRAY, TJPF_CMYK , ...
        else {
            if (tjDecompress2(m_jpegDecompressor,
                              input_buffer,
                              input_size,
                              output_buffer,
                              max_decoded_width,
                              max_decoded_width * planes,
                              max_decoded_height,
                              tjpf,
                              TJFLAG_ACCURATEDCT) != 0) {
                // try decode to original dim and scale using OpenCV
                WRN("Jpeg image decode failed " + STR(tjGetErrorStr2(m_jpegDecompressor)))
                return Status::CONTENT_DECODE_FAILED;
            }
            // Find the decoded image size using the predefined scaling factors in the turbo jpeg decoder
            uint scaledw = max_decoded_width, scaledh = max_decoded_height;
            for (int j=0; j < _num_scaling_factors; j++) {
                scaledw = TJSCALED(original_image_width, _scaling_factors[j]);
                scaledh = TJSCALED(original_image_height, _scaling_factors[j]);
                if (scaledw <= max_decoded_width && scaledh <= max_decoded_height)
                    break;
            }
            actual_decoded_width = scaledw;
            actual_decoded_height = scaledh;
        }
    } else {
        if (original_image_width < max_decoded_width)
            actual_decoded_width = original_image_width;
        else
            actual_decoded_width = max_decoded_width;
        if (original_image_height < max_decoded_height)
            actual_decoded_height = original_image_height;
        else
            actual_decoded_height = max_decoded_height;

        if (original_image_width > (_max_scaling_factor * max_decoded_width) || original_image_height > (_max_scaling_factor * max_decoded_height)) {
            // Initializing the crop width and height to the original width and height of image to avoid cropping
            unsigned int crop_width = original_image_width;
            unsigned int crop_height = original_image_height;
            float in_ratio = static_cast<float>(original_image_width) / original_image_height;
            if (original_image_width > (_max_scaling_factor * max_decoded_width)) {
                crop_width = _max_scaling_factor * max_decoded_width;
                if (crop_width > original_image_width) crop_width = original_image_width;
                crop_height = crop_width * (1.0 / in_ratio);
                if (crop_height > _max_scaling_factor * max_decoded_height) crop_height = _max_scaling_factor * max_decoded_height;
            } else if (original_image_height > (_max_scaling_factor * max_decoded_height)) {
                crop_height = _max_scaling_factor * max_decoded_height;
                if (crop_height > original_image_height) crop_height = original_image_height;
                crop_width = crop_height * in_ratio;
                if (crop_width > _max_scaling_factor * max_decoded_width) crop_width = _max_scaling_factor * max_decoded_width;
            }
            if (tjDecompress2_partial_scale(m_jpegDecompressor,
                                            input_buffer,
                                            input_size,
                                            output_buffer,
                                            actual_decoded_width,
                                            max_decoded_width * planes,
                                            max_decoded_height,
                                            tjpf,
                                            TJFLAG_ACCURATEDCT,
                                            crop_width, crop_height) != 0)

            {
                WRN("Jpeg partial image decode failed " + STR(tjGetErrorStr2(m_jpegDecompressor)))
                return Status::CONTENT_DECODE_FAILED;
            }
            // Find the decoded image size using the predefined scaling factors in the turbo jpeg decoder
            uint scaledw = max_decoded_width, scaledh = max_decoded_height;
            for (int j=0; j < _num_scaling_factors; j++) {
                scaledw = TJSCALED(original_image_width, _scaling_factors[j]);
                scaledh = TJSCALED(original_image_height, _scaling_factors[j]);
                if (scaledw <= max_decoded_width && scaledh <= max_decoded_height) {
                    break;
                }
            }
            actual_decoded_width = scaledw;
            actual_decoded_height = scaledh;
            // std::cout << "actual_decoded_width: " << actual_decoded_width << " actual_decoded_height: " << actual_decoded_height  << std::endl;
        } else {
            // TODO : Turbo Jpeg supports multiple color packing and color formats, add more as an option to the API TJPF_RGB, TJPF_BGR, TJPF_RGBX, TJPF_BGRX, TJPF_RGBA, TJPF_GRAY, TJPF_CMYK , ...
            if (tjDecompress2(m_jpegDecompressor,
                              input_buffer,
                              input_size,
                              output_buffer,
                              actual_decoded_width,
                              max_decoded_width * planes,
                              actual_decoded_height,
                              tjpf,
                              TJFLAG_ACCURATEDCT) != 0) {
                WRN("KO::Jpeg image decode failed " + STR(tjGetErrorStr2(m_jpegDecompressor)))
                return Status::CONTENT_DECODE_FAILED;
            }
            // Find the decoded image size using the predefined scaling factors in the turbo jpeg decoder
            if ((actual_decoded_width != original_image_width) || (actual_decoded_height != original_image_height)) {
                uint scaledw = actual_decoded_width, scaledh = actual_decoded_height;
                for (int j=0; j < _num_scaling_factors; j++) {
                    scaledw = TJSCALED(original_image_width, _scaling_factors[j]);
                    scaledh = TJSCALED(original_image_height, _scaling_factors[j]);
                    if (scaledw <= max_decoded_width && scaledh <= max_decoded_height)
                        break;
                }
                actual_decoded_width = scaledw;
                actual_decoded_height = scaledh;
            }
        }
    }
    return Status::OK;
}

TJDecoder::~TJDecoder() {
    tjDestroy(m_jpegDecompressor);
}
