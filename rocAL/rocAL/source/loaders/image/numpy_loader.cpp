/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

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

#include "loaders/image/numpy_loader.h"

#include <chrono>
#include <thread>

#include "vx_ext_amd.h"

NumpyLoader::NumpyLoader(void *dev_resources) : _circ_buff(dev_resources),
                                                _file_load_time("file load time", DBG_TIMING),
                                                _swap_handle_time("Swap_handle_time", DBG_TIMING) {
    _output_tensor = nullptr;
    _mem_type = RocalMemType::HOST;
    _internal_thread_running = false;
    _output_mem_size = 0;
    _batch_size = 1;
    _is_initialized = false;
    _remaining_file_count = 0;
    _device_id = 0;
}

NumpyLoader::~NumpyLoader() {
    de_init();
}

void NumpyLoader::shut_down() {
    if (_internal_thread_running)
        stop_internal_thread();
    _circ_buff.release();
}

void NumpyLoader::set_prefetch_queue_depth(size_t prefetch_queue_depth) {
    if (prefetch_queue_depth <= 0)
        THROW("Prefetch quque depth value cannot be zero or negative");
    _prefetch_queue_depth = prefetch_queue_depth;
}

void NumpyLoader::set_gpu_device_id(int device_id) {
    if (device_id < 0)
        THROW("invalid device_id passed to loader");
    _device_id = device_id;
}

size_t
NumpyLoader::remaining_count() {
    return _remaining_file_count;
}

void NumpyLoader::reset() {
    // stop the writer thread and empty the internal circular buffer
    _internal_thread_running = false;
    _circ_buff.unblock_writer();

    if (_load_thread.joinable())
        _load_thread.join();

    // Emptying the internal circular buffer
    _circ_buff.reset();

    // resetting the reader thread to the start of the media
    _file_counter = 0;
    _reader->reset();

    // Start loading (writer thread) again
    start_loading();
}

void NumpyLoader::de_init() {
    // Set running to 0 and wait for the internal thread to join
    stop_internal_thread();
    _output_mem_size = 0;
    _batch_size = 1;
    _is_initialized = false;
    _remaining_file_count = 0;
}

LoaderModuleStatus
NumpyLoader::load_next() {
    return update_output_tensor();
}

void NumpyLoader::set_output(Tensor *output_tensor) {
    _output_tensor = output_tensor;
    _output_mem_size = ((_output_tensor->info().data_size() + 8) & ~7);
}

void NumpyLoader::stop_internal_thread() {
    _internal_thread_running = false;
    _stopped = true;
    _circ_buff.unblock_reader();
    _circ_buff.unblock_writer();
    _circ_buff.reset();
    if (_load_thread.joinable())
        _load_thread.join();
}

void NumpyLoader::initialize(ReaderConfig reader_cfg, DecoderConfig decoder_cfg, RocalMemType mem_type, unsigned batch_size, bool decoder_keep_original) {
    if (_is_initialized)
        WRN("initialize() function is already called and loader module is initialized")

    if (_output_mem_size == 0)
        THROW("output tensor size is 0, set_output() should be called before initialize for loader modules")

    _mem_type = mem_type;
    _batch_size = batch_size;
    _loop = reader_cfg.loop();
    _tensor_size = _output_tensor->info().data_size() / batch_size;
    _output_names.resize(batch_size);
    try {
        _reader = create_reader(reader_cfg);
    } catch (const std::exception &e) {
        de_init();
        throw;
    }
    _decoded_data_info._data_names.resize(_batch_size);
    _tensor_roi.resize(_batch_size);
    _circ_buff.init(_mem_type, _output_mem_size, _prefetch_queue_depth);
    _is_initialized = true;
    LOG("Loader module initialized");
}

void NumpyLoader::start_loading() {
    if (!_is_initialized)
        THROW("start_loading() should be called after initialize() function is called")

    _remaining_file_count = _reader->count_items();
    _internal_thread_running = true;
    _load_thread = std::thread(&NumpyLoader::load_routine, this);
}

LoaderModuleStatus
NumpyLoader::load_routine() {
    LOG("Started the internal loader thread");
    LoaderModuleStatus last_load_status = LoaderModuleStatus::OK;
    // Initially record number of all the numpy arrays that are going to be loaded, this is used to know how many still there
    const std::vector<size_t> tensor_dims = _output_tensor->info().dims();
    auto num_dims = tensor_dims.size() - 1;
    auto data_layout = _output_tensor->info().layout();
    std::vector<size_t> max_shape(tensor_dims.begin() + 1, tensor_dims.end());
    std::vector<unsigned> strides_in_dims(num_dims + 1);
    strides_in_dims[num_dims] = 1;
    for (int i = num_dims - 1; i >= 0; i--) {
        strides_in_dims[i] = strides_in_dims[i + 1] * max_shape[i];
    }

    while (_internal_thread_running) {
        auto data = _circ_buff.get_write_buffer();
        if (!_internal_thread_running)
            break;

        auto load_status = LoaderModuleStatus::NO_MORE_DATA_TO_READ;
        {
            unsigned file_counter = 0;
            _file_load_time.start();  // Debug timing

            while ((file_counter != _batch_size) && _reader->count_items() > 0) {
                auto read_ptr = data + _tensor_size * file_counter;
                size_t read_size = _reader->open();
                if (read_size == 0) {
                    ERR("Opened file " + _reader->id() + " of size 0");
                    _reader->close();
                    continue;
                }
                auto fsize = _reader->read_numpy_data(read_ptr, read_size, strides_in_dims);
                if (fsize == 0) {
                    ERR("Cannot read numpy data from " + _reader->id());
                    _reader->close();
                    continue;
                }
                _decoded_data_info._data_names[file_counter] = _reader->id();
                auto original_roi = _reader->get_numpy_header_data().shape();
                // The numpy header data contains the full array shape. We require only width and height for ROI updation
                if (data_layout == RocalTensorlayout::NHWC) {
                    _tensor_roi[file_counter] = {original_roi[1], original_roi[0]};
                } else if (data_layout == RocalTensorlayout::NCHW) {
                    _tensor_roi[file_counter] = {original_roi[2], original_roi[1]};
                } else {
                    _tensor_roi[file_counter] = original_roi;
                }
                _reader->close();
                file_counter++;
            }
            _file_load_time.end();  // Debug timing
            _circ_buff.set_decoded_data_info(_decoded_data_info);
            _circ_buff.push();
            _file_counter += _output_tensor->info().batch_size();
            load_status = LoaderModuleStatus::OK;
        }
        if (load_status != LoaderModuleStatus::OK) {
            if (last_load_status != load_status) {
                if (load_status == LoaderModuleStatus::NO_MORE_DATA_TO_READ ||
                    load_status == LoaderModuleStatus::NO_FILES_TO_READ) {
                    LOG("Cycled through all numpy files, count " + TOSTR(_file_counter));
                } else {
                    ERR("ERROR: Detected error in reading the numpy files");
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

bool NumpyLoader::is_out_of_data() {
    return (remaining_count() < _batch_size);
}

size_t NumpyLoader::last_batch_padded_size() {
    return _reader->last_batch_padded_size();
}

LoaderModuleStatus
NumpyLoader::update_output_tensor() {
    LoaderModuleStatus status = LoaderModuleStatus::OK;

    if (is_out_of_data())
        return LoaderModuleStatus::NO_MORE_DATA_TO_READ;
    if (_stopped)
        return LoaderModuleStatus::OK;

    // _circ_buff.get_read_buffer_x() is blocking and puts the caller on sleep until new output is written to the _circ_buff
    if ((_mem_type == RocalMemType::OCL) || (_mem_type == RocalMemType::HIP)) {
        auto data_buffer = _circ_buff.get_read_buffer_dev();
        _swap_handle_time.start();
        if (_output_tensor->swap_handle(data_buffer) != 0)
            return LoaderModuleStatus::DEVICE_BUFFER_SWAP_FAILED;
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

    _output_decoded_data_info = _circ_buff.get_decoded_data_info();
    _output_names = _output_decoded_data_info._data_names;
    _output_tensor->update_tensor_roi(_tensor_roi);
    _circ_buff.pop();
    if (!_loop)
        _remaining_file_count -= _batch_size;

    return status;
}

Timing NumpyLoader::timing() {
    Timing t;
    t.read_time = _file_load_time.get_timing();
    t.process_time = _swap_handle_time.get_timing();
    return t;
}

LoaderModuleStatus NumpyLoader::set_cpu_affinity(cpu_set_t cpu_mask) {
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

LoaderModuleStatus NumpyLoader::set_cpu_sched_policy(struct sched_param sched_policy) {
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

std::vector<std::string> NumpyLoader::get_id() {
    return _output_names;
}

DecodedDataInfo NumpyLoader::get_decode_data_info() {
    return _output_decoded_data_info;
}
