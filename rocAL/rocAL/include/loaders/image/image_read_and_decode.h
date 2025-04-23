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
#include <dirent.h>

#include <memory>
#include <vector>

#include "pipeline/commons.h"
#include "loaders/loader_module.h"
#include "parameters/parameter_random_crop_decoder.h"
#include "readers/image/reader_factory.h"
#include "pipeline/timing_debug.h"
#include "decoders/image/turbo_jpeg_decoder.h"


class ImageReadAndDecode {
   public:
    ImageReadAndDecode();
    ~ImageReadAndDecode();
    size_t count();
    void reset();
    void create(ReaderConfig reader_config, DecoderConfig decoder_config, int batch_size, int device_id = 0);
    void set_bbox_vector(std::vector<std::vector<float>> bbox_coords) { _bbox_coords = bbox_coords; };
    void set_random_bbox_data_reader(std::shared_ptr<RandomBBoxCrop_MetaDataReader> randombboxcrop_meta_data_reader);
    std::vector<std::vector<float>> &get_batch_random_bbox_crop_coords();
    void set_batch_random_bbox_crop_coords(std::vector<std::vector<float>> batch_crop_coords);
    void feed_external_input(const std::vector<std::string>& input_images_names, const std::vector<unsigned char *>& input_buffer,
                             const std::vector<ROIxywh>& roi_xywh, unsigned int max_width, unsigned int max_height, unsigned int channels, ExternalSourceFileMode mode, bool eos);
    //! Loads a decompressed batch of images into the buffer indicated by buff
    /// \param buff User's buffer provided to be filled with decoded image data
    /// \param names User's buffer provided to be filled with name of the images decoded
    /// \param max_decoded_width User's buffer maximum width per decoded image. User expects the decoder to downscale the image if image's original width is bigger than max_width
    /// \param max_decoded_height user's buffer maximum height per decoded image. User expects the decoder to downscale the image if image's original height is bigger than max_height
    /// \param roi_width is set by the load() function tp the width of the region that decoded image is located. It's less than max_width and is either equal to the original image width if original image width is smaller than max_width or downscaled if necessary to fit the max_width criterion.
    /// \param roi_height  is set by the load() function tp the width of the region that decoded image is located.It's less than max_height and is either equal to the original image height if original image height is smaller than max_height or downscaled if necessary to fit the max_height criterion.
    /// \param output_color_format defines what color format user expects decoder to decode images into if capable of doing so supported is
    LoaderModuleStatus load(
        unsigned char *buff,
        std::vector<std::string> &names,
        const size_t max_decoded_width,
        const size_t max_decoded_height,
        std::vector<uint32_t> &roi_width,
        std::vector<uint32_t> &roi_height,
        std::vector<uint32_t> &actual_width,
        std::vector<uint32_t> &actual_height,
        RocalColorFormat output_color_format,
        bool decoder_keep_original = false);

    //! returns timing info or other status information
    Timing timing();
    size_t last_batch_padded_size();

   private:
    std::vector<std::shared_ptr<Decoder>> _decoder;
    std::shared_ptr<Decoder> _rocjpeg_decoder;
    std::shared_ptr<Reader> _reader;
    std::vector<std::vector<unsigned char>> _compressed_buff;
    std::vector<size_t> _actual_read_size;
    std::vector<std::string> _image_names;
    std::vector<size_t> _compressed_image_size;
    std::vector<unsigned char *> _decompressed_buff_ptrs;
    std::vector<size_t> _actual_decoded_width;
    std::vector<size_t> _actual_decoded_height;
    std::vector<size_t> _original_width;
    std::vector<size_t> _original_height;
    static const size_t MAX_COMPRESSED_SIZE = 1 * 1024 * 1024;  // 1 Meg
    TimingDbg _file_load_time, _decode_time;
    size_t _batch_size, _num_threads;
    DecoderConfig _decoder_config;
    std::vector<std::vector<float>> _bbox_coords, _crop_coords_batch;
    std::shared_ptr<RandomBBoxCrop_MetaDataReader> _randombboxcrop_meta_data_reader = nullptr;
    pCropCord _CropCord;
    RocalRandomCropDecParam *_random_crop_dec_param = nullptr;
    bool _is_external_source = false;
    int _device_id = 0;
    bool _set_device_id = false;
};
