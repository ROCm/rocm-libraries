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
#include "video_read_and_decode.h"

#ifdef ROCAL_VIDEO

//
// VideoLoader runs an internal thread for loading an decoding of sequences asynchronously
// it uses a circular buffer to store decoded sequence of frames for the user
class VideoLoader : public LoaderModule {
   public:
    explicit VideoLoader(void* dev_resources);
    ~VideoLoader() override;
    LoaderModuleStatus load_next() override;
    void initialize(ReaderConfig reader_cfg, DecoderConfig decoder_cfg, RocalMemType mem_type, unsigned batch_size, bool keep_orig_size = false) override;
    void set_output(Tensor* output_image) override;
    size_t remaining_count() override;  // returns number of remaining items to be loaded
    void reset() override;              // Resets the loader to load from the beginning
    Timing timing() override;
    void start_loading() override;
    LoaderModuleStatus set_cpu_affinity(cpu_set_t cpu_mask);
    LoaderModuleStatus set_cpu_sched_policy(struct sched_param sched_policy);
    void set_gpu_device_id(int device_id);
    std::vector<std::string> get_id() override;
    DecodedDataInfo get_decode_data_info() override;
    void set_prefetch_queue_depth(size_t prefetch_queue_depth) override;
    CropImageInfo get_crop_image_info() override { return _crop_img_info; }
    void set_random_bbox_data_reader(std::shared_ptr<RandomBBoxCrop_MetaDataReader> randombboxcrop_meta_data_reader) override{};
    std::vector<size_t> get_sequence_start_frame_number() override;
    std::vector<std::vector<float>> get_sequence_frame_timestamps() override;
    void shut_down() override;
    void feed_external_input(const std::vector<std::string>& input_images_names, const std::vector<unsigned char*>& input_buffer,
                             const std::vector<ROIxywh>& roi_xywh, unsigned int max_width, unsigned int max_height, unsigned int channels, ExternalSourceFileMode mode, bool eos) override {}

   private:
    bool is_out_of_data();
    void de_init();
    void stop_internal_thread();
    std::shared_ptr<VideoReadAndDecode> _video_loader;
    LoaderModuleStatus update_output_image();
    LoaderModuleStatus load_routine();
    Tensor* _output_tensor;
    std::vector<std::string> _output_names;  //!< frame name/ids that are stored in the _output_image
    size_t _output_mem_size;
    bool _internal_thread_running;
    size_t _batch_size;
    size_t _sequence_length;
    std::thread _load_thread;
    RocalMemType _mem_type;
    CircularBuffer _circ_buff;
    TimingDbg _swap_handle_time;
    bool _is_initialized;
    bool _stopped = false;
    bool _loop;                         //<! If true the reader will wrap around at the end of the media (files/images/...) and wouldn't stop
    size_t _prefetch_queue_depth;       // Used for circular buffer's internal buffer
    size_t _image_counter = 0;          //!< How many frames have been loaded already
    size_t _remaining_sequences_count;  //!< How many frames are there yet to be loaded
    bool _decoder_keep_original = false;
    int _device_id = 0;         //!< device_id for this loader
    std::vector<std::vector<size_t>> _sequence_start_framenum_vec;
    std::vector<std::vector<std::vector<float>>> _sequence_frame_timestamps_vec;
    CropImageInfo _crop_img_info;
    size_t _max_tensor_width, _max_tensor_height;
#if ENABLE_HIP
    hipStream_t _hip_stream = nullptr;
#endif
};
#endif
