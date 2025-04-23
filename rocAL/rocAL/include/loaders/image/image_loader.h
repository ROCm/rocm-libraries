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

#include <string>
#include <thread>
#include <vector>

#include "loaders/circular_buffer.h"
#include "pipeline/commons.h"
#include "image_read_and_decode.h"
#include "meta_data/meta_data_reader.h"
//
// ImageLoader runs an internal thread for loading an decoding of images asynchronously
// it uses a circular buffer to store decoded frames and images for the user
class ImageLoader : public LoaderModule {
   public:
    explicit ImageLoader(void* dev_resources);
    ~ImageLoader() override;
    LoaderModuleStatus load_next() override;
    void initialize(ReaderConfig reader_cfg, DecoderConfig decoder_cfg, RocalMemType mem_type, unsigned batch_size, bool keep_orig_size = false) override;
    void set_output(Tensor* output_tensor) override;
    void set_random_bbox_data_reader(std::shared_ptr<RandomBBoxCrop_MetaDataReader> randombboxcrop_meta_data_reader) override;
    size_t remaining_count() override;  // returns number of remaining items to be loaded
    void reset() override;              // Resets the loader to load from the beginning of the media
    Timing timing() override;
    void start_loading() override;
    LoaderModuleStatus set_cpu_affinity(cpu_set_t cpu_mask);
    LoaderModuleStatus set_cpu_sched_policy(struct sched_param sched_policy);
    void set_gpu_device_id(int device_id);
    std::vector<std::string> get_id() override;
    DecodedDataInfo get_decode_data_info() override;
    CropImageInfo get_crop_image_info() override;
    void set_prefetch_queue_depth(size_t prefetch_queue_depth) override;
    void shut_down() override;
    void feed_external_input(const std::vector<std::string>& input_images_names, const std::vector<unsigned char*>& input_buffer,
                             const std::vector<ROIxywh>& roi_xywh, unsigned int max_width, unsigned int max_height, unsigned int channels, ExternalSourceFileMode mode, bool eos) override;
    size_t last_batch_padded_size() override;

   private:
    bool is_out_of_data();
    void de_init();
    void stop_internal_thread();
    std::shared_ptr<ImageReadAndDecode> _image_loader;
    LoaderModuleStatus update_output_image();
    LoaderModuleStatus load_routine();

    std::shared_ptr<RandomBBoxCrop_MetaDataReader> _randombboxcrop_meta_data_reader = nullptr;
    Tensor* _output_tensor;
    std::vector<std::string> _output_names;  //!< image name/ids that are stores in the _output_image
    size_t _output_mem_size;
    std::vector<std::vector<float>> _bbox_coords;
    bool _internal_thread_running;
    size_t _batch_size;
    std::thread _load_thread;
    RocalMemType _mem_type;
    CropImageInfo _crop_image_info;
    CropImageInfo _output_cropped_img_info;
    CircularBuffer _circ_buff;
    TimingDbg _swap_handle_time;
    bool _is_initialized;
    bool _stopped = false;
    bool _loop;                     //<! If true the reader will wrap around at the end of the media (files/images/...) and wouldn't stop
    size_t _prefetch_queue_depth;   // Used for circular buffer's internal buffer
    size_t _image_counter = 0;      //!< How many images have been loaded already
    size_t _remaining_image_count;  //!< How many images are there yet to be loaded
    bool _decoder_keep_original = false;
    int _device_id;
    size_t _max_tensor_width, _max_tensor_height;
    bool _external_source_reader = false;  //!< Set to true if external source reader
    bool _external_input_eos = false;      //!< Set to true for last batch for the sequence
#if ENABLE_HIP
    hipStream_t _hip_stream = nullptr;
#endif
};
