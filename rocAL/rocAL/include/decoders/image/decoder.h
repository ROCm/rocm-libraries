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

#pragma once

#include <cstddef>
#include <iostream>
#include <vector>
#include "parameters/parameter_factory.h"
#include "parameters/parameter_random_crop_decoder.h"
#include "pipeline/commons.h"

#if ENABLE_HIP
#include "hip/hip_runtime_api.h"
#include "hip/hip_runtime.h"
#endif

#if ENABLE_HIP
#include "hip/hip_runtime_api.h"
#include "hip/hip_runtime.h"
#endif

enum class DecoderType {
    TURBO_JPEG = 0,        //!< Can only decode
    FUSED_TURBO_JPEG = 1,  //!< FOR PARTIAL DECODING
    OPENCV_DEC = 2,        //!< for back_up decoding
    HW_JPEG_DEC = 3,       //!< for JPEG decoding using HW via FFMPEG  
    SKIP_DECODE = 4,       //!< For skipping decoding in case of uncompressed data from reader
    FFMPEG_SW_DECODE = 5,   //!< for video decoding using CPU and FFMPEG
    FFMPEG_HW_DECODE = 6,   //!< for video decoding using HW via FFMPEG
    ROCDEC_VIDEO_DECODE = 7, //!< for video decoding using HW via rocDecode
    AUDIO_SOFTWARE_DECODE = 8,  //!< Uses sndfile to decode audio files
    ROCJPEG_DEC = 9             //!< rocJpeg hardware decoder for decoding jpeg files
};

class DecoderConfig {
   public:
    DecoderConfig() {}
    explicit DecoderConfig(DecoderType type) : _type(type) {}
    virtual DecoderType type() { return _type; };
    DecoderType _type = DecoderType::TURBO_JPEG;
    void set_random_area(std::vector<float> &random_area) { _random_area = std::move(random_area); }
    void set_random_aspect_ratio(std::vector<float> &random_aspect_ratio) { _random_aspect_ratio = std::move(random_aspect_ratio); }
    void set_num_attempts(unsigned num_attempts) { _num_attempts = num_attempts; }
    std::vector<float> get_random_area() { return _random_area; }
    std::vector<float> get_random_aspect_ratio() { return _random_aspect_ratio; }
    unsigned get_num_attempts() { return _num_attempts; }
    void set_seed(int seed) { _seed = seed; }
    int get_seed() { return _seed; }
#if ENABLE_HIP
    hipStream_t &get_hip_stream() { return _hip_stream; }
    void set_hip_stream(hipStream_t &stream) { _hip_stream = stream; }
#endif

   private:
    std::vector<float> _random_area, _random_aspect_ratio;
    unsigned _num_attempts = 10;
    int _seed = std::time(0);  // seed for decoder random crop
#if ENABLE_HIP
    hipStream_t _hip_stream;
#endif
};

class Decoder {
   public:
    enum class Status {
        OK = 0,
        HEADER_DECODE_FAILED,
        CONTENT_DECODE_FAILED,
        UNSUPPORTED,
        NO_MEMORY
    };

    enum class ColorFormat {
        GRAY = 0,
        RGB,
        BGR
    };
    //! Decodes the header of the Jpeg compressed data and returns basic info about the compressed image
    /*!
     \param input_buffer  User provided buffer containig the encoded image
     \param input_size Size of the compressed data provided in the input_buffer
     \param width pointer to the user's buffer to write the width of the compressed image to
     \param height pointer to the user's buffer to write the height of the compressed image to
     \param color_comps pointer to the user's buffer to write the number of color components of the compressed image to
    */
    virtual Status decode_info(unsigned char *input_buffer,
                               size_t input_size,
                               int *width,
                               int *height,
                               int *color_comps) = 0;
    
    //! Decodes the header of the Jpeg compressed data and returns basic info about the compressed image
    //! It also scales the width and height wrt max decoded width and height
    /*!
     \param input_buffer  User provided buffer containig the encoded image
     \param input_size Size of the compressed data provided in the input_buffer
     \param width pointer to the user's buffer to write the width of the compressed image/scaled width based on max_decoded_width
     \param height pointer to the user's buffer to write the height of the compressed image/scaled height based on max_decoded_height
     \param actual_width pointer to the user's buffer to write the width of the compressed image
     \param actual_height pointer to the user's buffer to write the height of the compressed image
     \param max_decoded_width maximum width of the decoded image
     \param max_decoded_height maximum height of the decoded image
     \param desired_decoded_color_format user provided color format of the decoded image
     \param index index of the image in the batch for which decode info must be fetched
    */
    virtual Status decode_info(unsigned char *input_buffer, size_t input_size, int *width, int *height, 
                               int *actual_width, int *actual_height, int max_decoded_width, 
                               int max_decoded_height, ColorFormat desired_decoded_color_format, int index) { return Status::UNSUPPORTED; }

    // TODO: Extend the decode API if needed, color format and order can be passed to the function
    //! Decodes the actual image data
    /*!
      \param input_buffer  User provided buffer containig the encoded image
      \param output_buffer User provided buffer used to write the decoded image into
      \param input_size Size of the compressed data provided in the input_buffer
      \param max_decoded_width The maximum width user wants the decoded image to be
      \param max_decoded_height The maximum height user wants the decoded image to be.

    */
    virtual Decoder::Status decode(unsigned char *input_buffer, size_t input_size, unsigned char *output_buffer,
                                   size_t max_decoded_width, size_t max_decoded_height,
                                   size_t original_image_width, size_t original_image_height,
                                   size_t &actual_decoded_width, size_t &actual_decoded_height,
                                   Decoder::ColorFormat desired_decoded_color_format, DecoderConfig decoder_config, bool keep_original) = 0;

    //! Decodes a batch of actual image data
    /*!
      \param output_buffer User provided buffer used to write the decoded image into
      \param max_decoded_width The maximum width user wants the decoded image to be. Image will be downscaled if bigger.
      \param max_decoded_height The maximum height user wants the decoded image to be. Image will be downscaled if bigger.
      \param original_image_width The actual width of the compressed image. decoded width will be equal to this if this is smaller than max_decoded_width
      \param original_image_height The actual height of the compressed image. decoded height will be equal to this if this is smaller than max_decoded_height
      \param actual_decoded_width The width of the image after decoding and scaling if original width is greater than max decoded width
      \param actual_decoded_height The height of the image after decoding and scaling if original height is greater than max decoded height
    */
    virtual Decoder::Status decode_batch(std::vector<unsigned char *> &output_buffer,
                                         size_t max_decoded_width, size_t max_decoded_height,
                                         std::vector<size_t> original_image_width, std::vector<size_t> original_image_height,
                                         std::vector<size_t> &actual_decoded_width, std::vector<size_t> &actual_decoded_height) { return Status::UNSUPPORTED; }

    virtual ~Decoder() = default;
    virtual void initialize(int device_id) = 0;
    virtual void initialize(int device_id, unsigned batch_size) { THROW("Initialize not implemented") }
    virtual bool is_partial_decoder() = 0;
    virtual void set_bbox_coords(std::vector<float> bbox_coords) = 0;
    virtual std::vector<float> get_bbox_coords() = 0;
    virtual void set_crop_window(CropWindow &crop_window) = 0;
};
