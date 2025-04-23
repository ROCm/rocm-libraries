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

#include "loaders/audio/audio_loader_sharded.h"

#ifdef ROCAL_AUDIO

AudioLoaderSharded::AudioLoaderSharded(void* dev_resources) : _dev_resources(dev_resources) {
    _loader_idx = 0;
}

void AudioLoaderSharded::set_prefetch_queue_depth(size_t prefetch_queue_depth) {
    if (prefetch_queue_depth <= 0)
        THROW("Prefetch queue depth value cannot be zero or negative");
    _prefetch_queue_depth = prefetch_queue_depth;
}

std::vector<std::string> AudioLoaderSharded::get_id() {
    if (!_initialized)
        THROW("get_id() should be called after initialize() function");
    return _loaders[_loader_idx]->get_id();
}

DecodedDataInfo AudioLoaderSharded::get_decode_data_info() {
    return _loaders[_loader_idx]->get_decode_data_info();
}

AudioLoaderSharded::~AudioLoaderSharded() {
    _loaders.clear();
}

size_t AudioLoaderSharded::last_batch_padded_size() {
    size_t sum = 0;
    for (auto& loader : _loaders)
        sum += loader->last_batch_padded_size();
    return sum;
}

void AudioLoaderSharded::fast_forward_through_empty_loaders() {
    int loaders_count = _loaders.size();
    // reject empty loaders and get to a loader that still has audios to play
    while (_loaders[_loader_idx]->remaining_count() == 0 && loaders_count-- > 0)
        increment_loader_idx();
}

LoaderModuleStatus AudioLoaderSharded::load_next() {
    if (!_initialized)
        return LoaderModuleStatus::NOT_INITIALIZED;
    increment_loader_idx();
    // Since loaders may have different number of audios loaded, some run out earlier than other.
    // Fast forward through loaders that are empty to get to a loader that is not empty.
    fast_forward_through_empty_loaders();
    auto ret = _loaders[_loader_idx]->load_next();
    return ret;
}

void AudioLoaderSharded::initialize(ReaderConfig reader_cfg, DecoderConfig decoder_cfg, RocalMemType mem_type,
                                    unsigned batch_size, bool keep_orig_size) {
    if (_initialized)
        return;
    _shard_count = reader_cfg.get_shard_count();
    // Create loader modules
    for (size_t i = 0; i < _shard_count; i++) {
        std::shared_ptr loader = std::make_shared<AudioLoader>(_dev_resources);
        loader->set_prefetch_queue_depth(_prefetch_queue_depth);
        _loaders.push_back(loader);
    }
    // Initialize loader modules
    for (size_t idx = 0; idx < _shard_count; idx++) {
        _loaders[idx]->set_output(_output_tensor);
        _loaders[idx]->set_gpu_device_id(idx);
        reader_cfg.set_shard_count(_shard_count);
        reader_cfg.set_shard_id(idx);
        _loaders[idx]->initialize(reader_cfg, decoder_cfg, mem_type, batch_size, keep_orig_size);
    }
    _initialized = true;
}

void AudioLoaderSharded::start_loading() {
    for (unsigned i = 0; i < _loaders.size(); i++) {
        _loaders[i]->start_loading();
        //  Changing thread scheduling policy and it's priority does not help on latest Ubuntu builds
        //  and needs tweaking the Linux security settings , can be turned on for experimentation
#if 0
        // Set thread scheduling policy
        struct sched_param params;
        params.sched_priority = sched_get_priority_max(SCHED_FIFO);
        _loaders[i]->set_cpu_sched_policy(params);
#endif
        // Setting cpu affinity for threads works and can be activated below for experimentation
#if 0
        // Set thread affinity thread 0 to core 0 , 1 toc core 1 , ...
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(i, &cpuset);
        _loaders[i]->set_cpu_affinity(cpuset);
#endif
    }
}

void AudioLoaderSharded::shut_down() {
    for (unsigned i = 0; i < _loaders.size(); i++)
        _loaders[i]->shut_down();
}

void AudioLoaderSharded::set_output(Tensor* output_tensor) {
    _output_tensor = output_tensor;
}

size_t AudioLoaderSharded::remaining_count() {
    int sum = 0;
    for (auto& loader : _loaders)
        sum += loader->remaining_count();
    return sum;
}

void AudioLoaderSharded::reset() {
    for (auto& loader : _loaders) {
        loader->reset();
    }
}

void AudioLoaderSharded::increment_loader_idx() {
    _loader_idx = (_loader_idx + 1) % _shard_count;
}

Timing AudioLoaderSharded::timing() {
    Timing t;
    long long unsigned max_decode_time = 0;
    long long unsigned max_read_time = 0;
    long long unsigned accumulated_process_time = 0;
    // audio read and decode runs in parallel using multiple loaders, and the observable latency that the AudioLoaderSharded user
    // is experiences on the load_next() call due to read and decode time is the maximum of all
    for (auto& loader : _loaders) {
        auto info = loader->timing();
        max_read_time = (info.read_time > max_read_time) ? info.read_time : max_read_time;
        max_decode_time = (info.decode_time > max_decode_time) ? info.decode_time : max_decode_time;
        accumulated_process_time += info.process_time;
    }
    t.decode_time = max_decode_time;
    t.read_time = max_read_time;
    t.process_time = accumulated_process_time;
    return t;
}
#endif
