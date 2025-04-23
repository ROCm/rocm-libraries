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

#include "readers/image/cifar10_data_reader.h"
#include "loaders/image/image_loader.h"
#include "readers/image/reader_factory.h"
#include "pipeline/timing_debug.h"

class CIFAR10DataLoader : public LoaderModule {
   public:
    explicit CIFAR10DataLoader(void *dev_resources);
    ~CIFAR10DataLoader() override;
    LoaderModuleStatus load_next() override;
    void initialize(ReaderConfig reader_cfg, DecoderConfig decoder_cfg, RocalMemType mem_type, unsigned batch_size, bool keep_orig_size = true) override;
    void set_output(Tensor *output_tensor) override;
    void set_random_bbox_data_reader(std::shared_ptr<RandomBBoxCrop_MetaDataReader> randombboxcrop_meta_data_reader) override;
    size_t remaining_count() override;
    void reset() override;
    void start_loading() override;
    std::vector<std::string> get_id() override;
    DecodedDataInfo get_decode_data_info() override;
    CropImageInfo get_crop_image_info() override;
    Timing timing() override;
    void set_prefetch_queue_depth(size_t prefetch_queue_depth) override;
    void shut_down() override;
    std::vector<std::vector<float>> &get_batch_random_bbox_crop_coords();
    void set_batch_random_bbox_crop_coords(std::vector<std::vector<float>> batch_crop_coords);
    void feed_external_input(const std::vector<std::string>& input_images_names, const std::vector<unsigned char *>& input_buffer,
                             const std::vector<ROIxywh>& roi_xywh, unsigned int max_width, unsigned int max_height, unsigned int channels, ExternalSourceFileMode mode, bool eos) override {}

   private:
    void increment_loader_idx();
    bool is_out_of_data();
    void de_init();
    void stop_internal_thread();
    LoaderModuleStatus update_output_image();
    LoaderModuleStatus load_routine();
    std::shared_ptr<Reader> _reader;
    void *_dev_resources;
    bool _initialized = false;
    RocalMemType _mem_type;
    size_t _output_mem_size;
    bool _internal_thread_running;
    size_t _batch_size;
    size_t _image_size;
    std::thread _load_thread;
    std::vector<unsigned char *> _load_buff;
    std::vector<size_t> _actual_read_size;
    std::vector<std::string> _output_names;
    CircularBuffer _circ_buff;
    size_t _prefetch_queue_depth;
    TimingDbg _file_load_time, _swap_handle_time;
    size_t _loader_idx;
    size_t _shard_count = 1;
    void fast_forward_through_empty_loaders();
    bool _is_initialized;
    bool _stopped = false;
    bool _loop;                     //<! If true the reader will wrap around at the end of the media (files/images/...) and wouldn't stop
    size_t _image_counter = 0;      //!< How many images have been loaded already
    size_t _remaining_image_count;  //!< How many images are there yet to be loaded
    Tensor *_output_tensor;
    std::shared_ptr<RandomBBoxCrop_MetaDataReader> _randombboxcrop_meta_data_reader = nullptr;
    std::vector<std::vector<float>> _bbox_coords, _crop_coords_batch;
    CropImageInfo _crop_image_info;
    CropImageInfo _output_cropped_image_info;
};
