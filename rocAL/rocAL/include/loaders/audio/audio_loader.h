/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#include "loaders/audio/audio_read_and_decode.h"
#include "loaders/circular_buffer.h"
#include "pipeline/commons.h"
#include "meta_data/meta_data_reader.h"

#ifdef ROCAL_AUDIO

// AudioLoader runs an internal thread for loading and decoding of audios asynchronously
// It uses a circular buffer to store decoded audios for the user
class AudioLoader : public LoaderModule {
   public:
    explicit AudioLoader(void* dev_resources);
    ~AudioLoader() override;
    LoaderModuleStatus load_next() override;
    void initialize(ReaderConfig reader_cfg, DecoderConfig decoder_cfg, RocalMemType mem_type, unsigned batch_size, bool keep_orig_size = false) override;
    void set_output(Tensor* output_audio) override;
    size_t remaining_count() override;  // returns number of remaining items to be loaded
    void reset() override;              // Resets the loader to load from the beginning of the media
    Timing timing() override;
    void start_loading() override;
    LoaderModuleStatus set_cpu_affinity(cpu_set_t cpu_mask);
    LoaderModuleStatus set_cpu_sched_policy(struct sched_param sched_policy);
    std::vector<std::string> get_id() override;
    DecodedDataInfo get_decode_data_info() override;
    void set_prefetch_queue_depth(size_t prefetch_queue_depth) override;
    void set_gpu_device_id(int device_id);
    void shut_down() override;
    void feed_external_input(const std::vector<std::string>& input_images_names, const std::vector<unsigned char*>& input_buffer,
                             const std::vector<ROIxywh>& roi_xywh, unsigned int max_width, unsigned int max_height, unsigned int channels,
                             ExternalSourceFileMode mode, bool eos) override { THROW("external source feed is not supported in audio loader") }
    size_t last_batch_padded_size() override;

   private:
    bool is_out_of_data();
    void de_init();
    void stop_internal_thread();
    LoaderModuleStatus update_output_audio();
    LoaderModuleStatus load_routine();
    std::shared_ptr<AudioReadAndDecode> _audio_loader;
    Tensor* _output_tensor;
    std::vector<std::string> _output_names; // audio file name/ids that are stored in the _output_audio
    bool _internal_thread_running;
    size_t _output_mem_size, _batch_size, _max_decoded_samples, _max_decoded_channels;
    std::thread _load_thread;
    RocalMemType _mem_type;
    DecodedDataInfo _decoded_audio_info;
    DecodedDataInfo _output_decoded_audio_info;
    CircularBuffer _circ_buff;
    TimingDbg _swap_handle_time;
    bool _is_initialized;
    bool _stopped = false;
    bool _loop;                         // If true the reader will wrap around at the end of the media (files/audios/...) and wouldn't stop
    size_t _prefetch_queue_depth = 0;   // Used for circular buffer's internal buffer allocation
    size_t _audio_counter = 0;          // How many audios have been loaded already
    size_t _remaining_audio_count;      // How many audios are there yet to be loaded
    int _device_id;
};
#endif
