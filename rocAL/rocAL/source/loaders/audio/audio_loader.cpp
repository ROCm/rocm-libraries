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

#include "loaders/audio/audio_loader.h"
#include "loaders/audio/audio_read_and_decode.h"

#include <chrono>
#include <thread>

#ifdef ROCAL_AUDIO

AudioLoader::AudioLoader(void* dev_resources) : _circ_buff(dev_resources),
                                                _swap_handle_time("Swap_handle_time", DBG_TIMING) {
    _output_tensor = nullptr;
    _mem_type = RocalMemType::HOST;
    _internal_thread_running = false;
    _output_mem_size = 0;
    _batch_size = 1;
    _is_initialized = false;
    _remaining_audio_count = 0;
    _device_id = 0;
}

AudioLoader::~AudioLoader() {
    de_init();
}

void AudioLoader::shut_down() {
    if (_internal_thread_running)
        stop_internal_thread();
    _circ_buff.release();
}

void AudioLoader::set_prefetch_queue_depth(size_t prefetch_queue_depth) {
    if (prefetch_queue_depth <= 0)
        THROW("Prefetch queue depth value cannot be zero or negative");
    _prefetch_queue_depth = prefetch_queue_depth;
}

void AudioLoader::set_gpu_device_id(int device_id) {
    if (device_id < 0)
        THROW("invalid device_id passed to loader");
    _device_id = device_id;
}

size_t
AudioLoader::remaining_count() {
    return _remaining_audio_count;
}

void AudioLoader::reset() {
    // stop the writer thread and empty the internal circular buffer
    _internal_thread_running = false;
    _circ_buff.unblock_writer();
    if (_load_thread.joinable())
        _load_thread.join();
    // Emptying the internal circular buffer
    _circ_buff.reset();
    // resetting the reader thread to the start of the media
    _audio_counter = 0;
    _audio_loader->Reset();
    // Start loading (writer thread) again
    start_loading();
}

void AudioLoader::de_init() {
    // Set running to 0 and wait for the internal thread to join
    stop_internal_thread();
    _output_mem_size = 0;
    _batch_size = 1;
    _is_initialized = false;
}

LoaderModuleStatus
AudioLoader::load_next() {
    return update_output_audio();
}

void AudioLoader::set_output(Tensor* output_tensor) {
    _output_tensor = output_tensor;
    _output_mem_size = _output_tensor->info().data_size();
}

void AudioLoader::stop_internal_thread() {
    _internal_thread_running = false;
    _stopped = true;
    _circ_buff.unblock_reader();
    _circ_buff.unblock_writer();
    _circ_buff.reset();
    if (_load_thread.joinable())
        _load_thread.join();
}

void AudioLoader::initialize(ReaderConfig reader_cfg, DecoderConfig decoder_cfg, RocalMemType mem_type, unsigned batch_size, bool decoder_keep_original) {
    if (_is_initialized)
        WRN("initialize() function is already called and loader module is initialized")
    if (_output_mem_size == 0)
        THROW("output audio size is 0, set_output() should be called before initialize for loader modules")
    _mem_type = mem_type;
    _batch_size = batch_size;
    _loop = reader_cfg.loop();
    _audio_loader = std::make_shared<AudioReadAndDecode>();
    size_t shard_count = reader_cfg.get_shard_count();
    int device_id = reader_cfg.get_shard_id();
    try {
        // set the device_id for decoder same as shard_id for number of shards > 1
        if (shard_count > 1) {
            _audio_loader->Create(reader_cfg, decoder_cfg, _batch_size, device_id);
        } else {
            _audio_loader->Create(reader_cfg, decoder_cfg, _batch_size);
        }
    } catch (const std::exception& e) {
        de_init();
        throw;
    }
    _max_decoded_samples = _output_tensor->info().max_shape().at(0);
    _max_decoded_channels = _output_tensor->info().max_shape().at(1);
    _decoded_audio_info._data_names.resize(_batch_size);
    _decoded_audio_info._audio_samples.resize(_batch_size);
    _decoded_audio_info._audio_channels.resize(_batch_size);
    _decoded_audio_info._audio_sample_rates.resize(_batch_size);
    _circ_buff.init(_mem_type, _output_mem_size, _prefetch_queue_depth);
    _is_initialized = true;
    LOG("Loader module initialized");
}

void AudioLoader::start_loading() {
    if (!_is_initialized)
        THROW("start_loading() should be called after initialize() function is called")

    _remaining_audio_count = _audio_loader->Count();
    _internal_thread_running = true;
    _load_thread = std::thread(&AudioLoader::load_routine, this);
}

LoaderModuleStatus
AudioLoader::load_routine() {
    LOG("Started the internal loader thread");
    LoaderModuleStatus last_load_status = LoaderModuleStatus::OK;
    // Initially record number of all the audios that are going to be loaded, this is used to know how many still there

    while (_internal_thread_running) {
        auto data = reinterpret_cast<float*>(_circ_buff.get_write_buffer());
        if (!_internal_thread_running)
            break;

        auto load_status = LoaderModuleStatus::NO_MORE_DATA_TO_READ;
        {
            load_status = _audio_loader->Load(data,
                                              _decoded_audio_info,
                                              _max_decoded_samples,
                                              _max_decoded_channels);

            if (load_status == LoaderModuleStatus::OK) {
                _circ_buff.set_decoded_data_info(_decoded_audio_info);
                _circ_buff.push();
                _audio_counter += _output_tensor->info().batch_size();
            }
        }
        if (load_status != LoaderModuleStatus::OK) {
            if (last_load_status != load_status) {
                if (load_status == LoaderModuleStatus::NO_MORE_DATA_TO_READ ||
                    load_status == LoaderModuleStatus::NO_FILES_TO_READ) {
                    LOG("Cycled through all audios, count " + TOSTR(_audio_counter));
                } else {
                    ERR("ERROR: Detected error in reading the audios");
                }
                last_load_status = load_status;
            }
            // Here it sets the out-of-data flag and signal the circular buffer's internal
            // read semaphore using release() call
            // , and calls the release() allows the reader thread to wake up and handle
            // the out-of-data case properly
            // It also slows down the reader thread since there is no more data to read,
            // till program ends or till reset is called
            _circ_buff.unblock_reader();
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
    return LoaderModuleStatus::OK;
}

// Uses the same condition in audio_read_and_decode.cpp to check if enough samples are present
bool AudioLoader::is_out_of_data() {
    return (remaining_count() < _batch_size);
}

size_t AudioLoader::last_batch_padded_size() {
    return _audio_loader->last_batch_padded_size();
}

LoaderModuleStatus
AudioLoader::update_output_audio() {
    LoaderModuleStatus status = LoaderModuleStatus::OK;

    if (is_out_of_data())
        return LoaderModuleStatus::NO_MORE_DATA_TO_READ;
    if (_stopped)
        return LoaderModuleStatus::OK;
    // _circ_buff.get_read_buffer_x() is blocking and puts the caller on sleep until new audios are written to the _circ_buff
    if ((_mem_type == RocalMemType::OCL) || (_mem_type == RocalMemType::HIP)) {
        auto data_buffer = _circ_buff.get_read_buffer_dev();
        _swap_handle_time.start();
        if (_output_tensor->swap_handle(data_buffer) != 0)
            return LoaderModuleStatus ::DEVICE_BUFFER_SWAP_FAILED;
        _swap_handle_time.end();
    } else {
        auto data_buffer = _circ_buff.get_read_buffer_host();
        _swap_handle_time.start();
        if (_output_tensor->swap_handle(data_buffer) != 0)
            return LoaderModuleStatus::HOST_BUFFER_SWAP_FAILED;
        _swap_handle_time.end();
    }
    if (_stopped)
        return LoaderModuleStatus::OK;
    _output_decoded_audio_info = _circ_buff.get_decoded_data_info();
    _output_names = _output_decoded_audio_info._data_names;
    _output_tensor->update_tensor_roi(_output_decoded_audio_info._audio_samples, _output_decoded_audio_info._audio_channels);
    _output_tensor->update_audio_tensor_sample_rate(_output_decoded_audio_info._audio_sample_rates);
    _circ_buff.pop();
    if (!_loop)
        _remaining_audio_count -= _batch_size;
    return status;
}

Timing AudioLoader::timing() {
    auto t = _audio_loader->GetTiming();
    t.process_time = _swap_handle_time.get_timing();
    return t;
}

LoaderModuleStatus AudioLoader::set_cpu_affinity(cpu_set_t cpu_mask) {
    if (!_internal_thread_running)
        THROW("set_cpu_affinity() should be called after start_loading function is called")
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#else
    int ret = pthread_setaffinity_np(_load_thread.native_handle(),
                                     sizeof(cpu_set_t), &cpu_mask);
    if (ret != 0)
        WRN("Error calling pthread_setaffinity_np: " + TOSTR(ret));
#endif
    return LoaderModuleStatus::OK;
}

LoaderModuleStatus AudioLoader::set_cpu_sched_policy(struct sched_param sched_policy) {
    if (!_internal_thread_running)
        THROW("set_cpu_sched_policy() should be called after start_loading function is called")
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#else
    auto ret = pthread_setschedparam(_load_thread.native_handle(), SCHED_FIFO, &sched_policy);
    if (ret != 0)
        WRN("Unsuccessful in setting thread realtime priority for loader thread err = " + TOSTR(ret))
#endif
    return LoaderModuleStatus::OK;
}

std::vector<std::string> AudioLoader::get_id() {
    return _output_names;
}

DecodedDataInfo AudioLoader::get_decode_data_info() {
    return _output_decoded_audio_info;
}
#endif
