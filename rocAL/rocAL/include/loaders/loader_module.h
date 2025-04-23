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
#include <memory>

#include "readers/image/image_reader.h"
#include "circular_buffer.h"
#include "pipeline/commons.h"
#include "decoders/image/decoder.h"
#include "meta_data/meta_data_graph.h"
#include "meta_data/meta_data_reader.h"
#include "pipeline/tensor.h"

enum class LoaderModuleStatus {
    OK = 0,
    DEVICE_BUFFER_SWAP_FAILED,
    HOST_BUFFER_SWAP_FAILED,
    NO_FILES_TO_READ,
    DECODE_FAILED,
    NO_MORE_DATA_TO_READ,
    NOT_INITIALIZED
};

/*! \class LoaderModule The interface defining the API and requirements of loader modules*/
class LoaderModule {
   public:
    virtual void initialize(ReaderConfig reader_config, DecoderConfig decoder_config, RocalMemType mem_type, unsigned batch_size, bool keep_orig_size) = 0;
    virtual void set_output(Tensor* output_tensor) = 0;
    virtual LoaderModuleStatus load_next() = 0;     // Loads the next image data into the Image's buffer set by calling into the set_output
    virtual void reset() = 0;                       // Resets the loader to load from the beginning of the media
    virtual size_t remaining_count() = 0;           // Returns the number of available images to be loaded
    virtual ~LoaderModule() = default;
    virtual Timing timing() = 0;                    // Returns timing info
    virtual std::vector<std::string> get_id() = 0;  // returns the id of the last batch of images/frames loaded
    virtual void start_loading() = 0;               // starts internal loading thread
    virtual DecodedDataInfo get_decode_data_info() = 0;
    virtual CropImageInfo get_crop_image_info() { return {}; }
    virtual void set_prefetch_queue_depth(size_t prefetch_queue_depth) = 0;
    // introduce meta data reader
    virtual void set_random_bbox_data_reader(std::shared_ptr<RandomBBoxCrop_MetaDataReader> randombboxcrop_meta_data_reader) { THROW("set_random_bbox_data_reader is not compatible with this implementation") }
    virtual void shut_down() = 0;
    virtual std::vector<size_t> get_sequence_start_frame_number() { return {}; }
    virtual std::vector<std::vector<float>> get_sequence_frame_timestamps() { return {}; }
    // External Source reader
    virtual void feed_external_input(const std::vector<std::string>& input_images_names, const std::vector<unsigned char*>& input_buffer,
                                     const std::vector<ROIxywh>& roi_xywh, unsigned int max_width, unsigned int max_height,
                                     unsigned int channels, ExternalSourceFileMode mode, bool eos) = 0;
    virtual size_t last_batch_padded_size() { return 0; }
   protected:
    DecodedDataInfo _decoded_data_info, _output_decoded_data_info;  // Stores the decoded data info
};

using pLoaderModule = std::shared_ptr<LoaderModule>;