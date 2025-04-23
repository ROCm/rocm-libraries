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

#include "loaders/image/image_loader.h"

#include <chrono>
#include <thread>

#include "loaders/image/image_read_and_decode.h"
#include "vx_ext_amd.h"

ImageLoader::ImageLoader(void *dev_resources) : _circ_buff(dev_resources),
                                                _swap_handle_time("Swap_handle_time", DBG_TIMING) {
    _output_tensor = nullptr;
    _mem_type = RocalMemType::HOST;
    _internal_thread_running = false;
    _output_mem_size = 0;
    _batch_size = 1;
    _is_initialized = false;
    _remaining_image_count = 0;
    _device_id = 0;
#if ENABLE_HIP
    DeviceResourcesHip *hipres = static_cast<DeviceResourcesHip *>(dev_resources);
    _hip_stream = hipres->hip_stream;
#endif
}

ImageLoader::~ImageLoader() {
    de_init();
}

void ImageLoader::shut_down() {
    if (_internal_thread_running)
        stop_internal_thread();
    _circ_buff.release();
}

void ImageLoader::set_prefetch_queue_depth(size_t prefetch_queue_depth) {
    if (prefetch_queue_depth <= 0)
        THROW("Prefetch quque depth value cannot be zero or negative");
    _prefetch_queue_depth = prefetch_queue_depth;
}

void ImageLoader::set_gpu_device_id(int device_id) {
    if (device_id < 0)
        THROW("invalid device_id passed to loader");
    _device_id = device_id;
}

size_t
ImageLoader::remaining_count() {
    if (_external_source_reader) {
        if ((_image_loader->count() != 0) && !_external_input_eos)
            return _batch_size;
        else {
            return 0;
        }
    }
    return _remaining_image_count;
}

void ImageLoader::reset() {
    // stop the writer thread and empty the internal circular buffer
    _internal_thread_running = false;
    _circ_buff.unblock_writer();

    if (_load_thread.joinable())
        _load_thread.join();

    // Emptying the internal circular buffer
    _circ_buff.reset();

    // resetting the reader thread to the start of the media
    _image_counter = 0;
    _image_loader->reset();

    // Start loading (writer thread) again
    start_loading();
}

void ImageLoader::de_init() {
    // Set running to 0 and wait for the internal thread to join
    stop_internal_thread();
    _output_mem_size = 0;
    _batch_size = 1;
    _is_initialized = false;
}

LoaderModuleStatus
ImageLoader::load_next() {
    return update_output_image();
}

void ImageLoader::set_output(Tensor *output_tensor) {
    _output_tensor = output_tensor;
    _output_mem_size = ((_output_tensor->info().data_size() + 8) & ~7);  // Making output size as a multiple of 8 to support vectorized load and store in RPP
}

void ImageLoader::set_random_bbox_data_reader(std::shared_ptr<RandomBBoxCrop_MetaDataReader> randombboxcrop_meta_data_reader) {
    _randombboxcrop_meta_data_reader = randombboxcrop_meta_data_reader;
    _circ_buff.random_bbox_crop_flag = true;
}

void ImageLoader::stop_internal_thread() {
    _internal_thread_running = false;
    _stopped = true;
    _circ_buff.unblock_reader();
    _circ_buff.unblock_writer();
    _circ_buff.reset();
    if (_load_thread.joinable())
        _load_thread.join();
}

void ImageLoader::initialize(ReaderConfig reader_cfg, DecoderConfig decoder_cfg, RocalMemType mem_type, unsigned batch_size, bool decoder_keep_original) {
    if (_is_initialized)
        WRN("initialize() function is already called and loader module is initialized")

    if (_output_mem_size == 0)
        THROW("output image size is 0, set_output() should be called before initialize for loader modules")

    _mem_type = mem_type;
    _batch_size = batch_size;
    _loop = reader_cfg.loop();
    _decoder_keep_original = decoder_keep_original;
    _image_loader = std::make_shared<ImageReadAndDecode>();
    size_t shard_count = reader_cfg.get_shard_count();
    int device_id = reader_cfg.get_shard_id();
#if ENABLE_HIP
    // Set stream in decoder config, to be used by rocJpeg decoder for scaling
    if (decoder_cfg._type == DecoderType::ROCJPEG_DEC) {
        decoder_cfg.set_hip_stream(_hip_stream);
    }
#endif
    try {
        // set the device_id for decoder same as shard_id for number of shards > 1
        if (shard_count > 1)
            _image_loader->create(reader_cfg, decoder_cfg, _batch_size, device_id);
        else
            _image_loader->create(reader_cfg, decoder_cfg, _batch_size);
    } catch (const std::exception &e) {
        de_init();
        throw;
    }
    _max_tensor_width = _output_tensor->info().max_shape().at(0);
    _max_tensor_height = _output_tensor->info().max_shape().at(1);
    _decoded_data_info._data_names.resize(_batch_size);
    _decoded_data_info._roi_height.resize(_batch_size);
    _decoded_data_info._roi_width.resize(_batch_size);
    _decoded_data_info._original_height.resize(_batch_size);
    _decoded_data_info._original_width.resize(_batch_size);
    _crop_image_info._crop_image_coords.resize(_batch_size);
    if (decoder_cfg._type == DecoderType::ROCJPEG_DEC) {
        // Initialize circular buffer with HIP memory for rocJPEG hardware decoder
        _circ_buff.init(_mem_type, _output_mem_size, _prefetch_queue_depth, true);
    } else {
        _circ_buff.init(_mem_type, _output_mem_size, _prefetch_queue_depth);
    }
    _is_initialized = true;
    _image_loader->set_random_bbox_data_reader(_randombboxcrop_meta_data_reader);
    LOG("Loader module initialized");
}

void ImageLoader::start_loading() {
    if (!_is_initialized)
        THROW("start_loading() should be called after initialize() function is called")

    _remaining_image_count = _image_loader->count();
    _internal_thread_running = true;
    _load_thread = std::thread(&ImageLoader::load_routine, this);
}

LoaderModuleStatus
ImageLoader::load_routine() {
    LOG("Started the internal loader thread");
    LoaderModuleStatus last_load_status = LoaderModuleStatus::OK;
    // Initially record number of all the images that are going to be loaded, this is used to know how many still there

    while (_internal_thread_running) {
        auto data = _circ_buff.get_write_buffer();
        if (!_internal_thread_running)
            break;

        auto load_status = LoaderModuleStatus::NO_MORE_DATA_TO_READ;
        {
            load_status = _image_loader->load(data,
                                              _decoded_data_info._data_names,
                                              _max_tensor_width,
                                              _max_tensor_height,
                                              _decoded_data_info._roi_width,
                                              _decoded_data_info._roi_height,
                                              _decoded_data_info._original_width,
                                              _decoded_data_info._original_height,
                                              _output_tensor->info().color_format(), _decoder_keep_original);

            if (load_status == LoaderModuleStatus::OK) {
                if (_randombboxcrop_meta_data_reader) {
                    _crop_image_info._crop_image_coords = _image_loader->get_batch_random_bbox_crop_coords();
                    _circ_buff.set_crop_image_info(_crop_image_info);
                }
                _circ_buff.set_decoded_data_info(_decoded_data_info);
                _circ_buff.push();
                _image_counter += _output_tensor->info().batch_size();
            }
        }
        if (load_status != LoaderModuleStatus::OK) {
            if (last_load_status != load_status) {
                if (load_status == LoaderModuleStatus::NO_MORE_DATA_TO_READ ||
                    load_status == LoaderModuleStatus::NO_FILES_TO_READ) {
                    LOG("Cycled through all images, count " + TOSTR(_image_counter));
                } else {
                    ERR("ERROR: Detected error in reading the images");
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

bool ImageLoader::is_out_of_data() {
    return (remaining_count() < _batch_size);
}

size_t ImageLoader::last_batch_padded_size() {
    return _image_loader->last_batch_padded_size();
}

LoaderModuleStatus
ImageLoader::update_output_image() {
    LoaderModuleStatus status = LoaderModuleStatus::OK;

    if (is_out_of_data())
        return LoaderModuleStatus::NO_MORE_DATA_TO_READ;
    if (_stopped)
        return LoaderModuleStatus::OK;

    // _circ_buff.get_read_buffer_x() is blocking and puts the caller on sleep until new images are written to the _circ_buff
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

    _output_decoded_data_info = _circ_buff.get_decoded_data_info();
    if (_randombboxcrop_meta_data_reader) {
        _output_cropped_img_info = _circ_buff.get_cropped_image_info();
    }
    _output_names = _output_decoded_data_info._data_names;
    _output_tensor->update_tensor_roi(_output_decoded_data_info._roi_width, _output_decoded_data_info._roi_height);
    _circ_buff.pop();
    if (!_loop)
        _remaining_image_count -= _batch_size;

    return status;
}

Timing ImageLoader::timing() {
    auto t = _image_loader->timing();
    t.process_time = _swap_handle_time.get_timing();
    return t;
}

LoaderModuleStatus ImageLoader::set_cpu_affinity(cpu_set_t cpu_mask) {
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

LoaderModuleStatus ImageLoader::set_cpu_sched_policy(struct sched_param sched_policy) {
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

std::vector<std::string> ImageLoader::get_id() {
    return _output_names;
}

DecodedDataInfo ImageLoader::get_decode_data_info() {
    return _output_decoded_data_info;
}

CropImageInfo ImageLoader::get_crop_image_info() {
    return _output_cropped_img_info;
}

void ImageLoader::feed_external_input(const std::vector<std::string>& input_images_names, const std::vector<unsigned char *>& input_buffer, const std::vector<ROIxywh>& roi_xywh, unsigned int max_width, unsigned int max_height, unsigned int channels, ExternalSourceFileMode mode, bool eos) {
    _external_source_reader = true;
    _external_input_eos = eos;
    _image_loader->feed_external_input(input_images_names, input_buffer, roi_xywh, max_width, max_height, channels, mode, eos);
}
