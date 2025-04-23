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

#include "pipeline/ring_buffer.h"
#include "device/device_manager.h"

RingBuffer::RingBuffer(unsigned buffer_depth) : BUFF_DEPTH(buffer_depth),
                                                _dev_sub_buffer(buffer_depth),
                                                _host_sub_buffers(buffer_depth),
                                                _dev_roi_buffers(buffer_depth),
                                                _host_roi_buffers(buffer_depth),
                                                _dev_bbox_buffer(buffer_depth),
                                                _dev_labels_buffer(buffer_depth) {
    reset();
}

void RingBuffer::block_if_empty() {
    std::unique_lock<std::mutex> lock(_lock);
    if (empty()) {  // if the current read buffer is being written wait on it
        if (_dont_block)
            return;
        _wait_for_load.wait(lock);
    }
}

void RingBuffer::block_if_full() {
    std::unique_lock<std::mutex> lock(_lock);
    // Write the whole buffer except for the last spot which is being read by the reader thread
    if (full()) {
        if (_dont_block)
            return;
        _wait_for_unload.wait(lock);
    }
}

std::pair<std::vector<void *>, std::vector<unsigned *>> RingBuffer::get_read_buffers() {
    block_if_empty();
    if ((_mem_type == RocalMemType::OCL) || (_mem_type == RocalMemType::HIP))
        return std::make_pair(_dev_sub_buffer[_read_ptr], _dev_roi_buffers[_read_ptr]);
    return std::make_pair(_host_sub_buffers[_read_ptr], _host_roi_buffers[_read_ptr]);
}

std::pair<void *, void *> RingBuffer::get_box_encode_read_buffers() {
    block_if_empty();
    if ((_mem_type == RocalMemType::OCL) || (_mem_type == RocalMemType::HIP))
        return std::make_pair(_dev_bbox_buffer[_read_ptr], _dev_labels_buffer[_read_ptr]);
    return std::make_pair(_host_meta_data_buffers[_read_ptr][1], _host_meta_data_buffers[_read_ptr][0]);
}

std::pair<std::vector<void *>, std::vector<unsigned *>> RingBuffer::get_write_buffers() {
    block_if_full();
    if ((_mem_type == RocalMemType::OCL) || (_mem_type == RocalMemType::HIP))
        return std::make_pair(_dev_sub_buffer[_write_ptr], _dev_roi_buffers[_write_ptr]);
    return std::make_pair(_host_sub_buffers[_write_ptr], _host_roi_buffers[_write_ptr]);
}

std::pair<void *, void *> RingBuffer::get_box_encode_write_buffers() {
    block_if_full();
    if ((_mem_type == RocalMemType::OCL) || (_mem_type == RocalMemType::HIP))
        return std::make_pair(_dev_bbox_buffer[_write_ptr], _dev_labels_buffer[_write_ptr]);
    return std::make_pair(_host_meta_data_buffers[_write_ptr][1], _host_meta_data_buffers[_write_ptr][0]);
}

std::vector<void *> RingBuffer::get_meta_read_buffers() {
    block_if_empty();
    return _host_meta_data_buffers[_read_ptr];
}

std::vector<void *> RingBuffer::get_meta_write_buffers() {
    block_if_full();
    return _host_meta_data_buffers[_write_ptr];
}

void RingBuffer::unblock_reader() {
    // Wake up the reader thread in case it's waiting for a load
    _wait_for_load.notify_all();
}

void RingBuffer::release_all_blocked_calls() {
    _dont_block = true;
    unblock_reader();
    unblock_writer();
}

void RingBuffer::release_if_empty() {
    if (empty()) unblock_reader();
}

void RingBuffer::unblock_writer() {
    // Wake up the writer thread in case it's waiting for an unload
    _wait_for_unload.notify_all();
}

void RingBuffer::init(RocalMemType mem_type, void *devres, std::vector<size_t> &sub_buffer_size, std::vector<size_t> &roi_buffer_size) {
    _mem_type = mem_type;
    _dev = devres;
    _sub_buffer_size = sub_buffer_size;
    auto sub_buffer_count = sub_buffer_size.size();
    if (BUFF_DEPTH < 2)
        THROW("Error internal buffer size for the ring buffer should be greater than one")

#if ENABLE_OPENCL
    DeviceResources *dev_ocl = static_cast<DeviceResources *>(_dev);
    // Allocating buffers
    if (mem_type == RocalMemType::OCL) {
        if (dev_ocl->cmd_queue == nullptr || dev_ocl->device_id == nullptr || dev_ocl->context == nullptr)
            THROW("Error ocl structure needed since memory type is OCL");

        cl_int err = CL_SUCCESS;

        for (size_t buffIdx = 0; buffIdx < BUFF_DEPTH; buffIdx++) {
            cl_mem_flags flags = CL_MEM_READ_ONLY;

            _dev_sub_buffer[buffIdx].resize(sub_buffer_count);
            for (unsigned sub_idx = 0; sub_idx < sub_buffer_count; sub_idx++) {
                _dev_sub_buffer[buffIdx][sub_idx] = clCreateBuffer(dev_ocl->context, flags, _sub_buffer_size[sub_idx], NULL, &err);

                if (err) {
                    _dev_sub_buffer.clear();
                    THROW("clCreateBuffer of size " + TOSTR(_sub_buffer_size[sub_idx]) + " index " + TOSTR(sub_idx) +
                          " failed " + TOSTR(err));
                }

                clRetainMemObject((cl_mem)_dev_sub_buffer[buffIdx][sub_idx]);
            }
        }
    } else {
#elif ENABLE_HIP
    DeviceResourcesHip *dev_hip = static_cast<DeviceResourcesHip *>(_dev);
    // Allocating buffers
    if (_mem_type == RocalMemType::HIP) {
        if (dev_hip->device_id == -1)
            THROW("Error Hip Device is not initialzed");

        for (size_t buffIdx = 0; buffIdx < BUFF_DEPTH; buffIdx++) {
            _dev_sub_buffer[buffIdx].resize(sub_buffer_count);
            _dev_roi_buffers[buffIdx].resize(sub_buffer_count);
            for (unsigned sub_idx = 0; sub_idx < sub_buffer_count; sub_idx++) {
                hipError_t err = hipMalloc(&_dev_sub_buffer[buffIdx][sub_idx], _sub_buffer_size[sub_idx]);
                // printf("allocated HIP device buffer <%d, %d, %d, %p>\n", buffIdx, sub_idx, _sub_buffer_size[sub_idx], _dev_sub_buffer[buffIdx][sub_idx]);
                if (err != hipSuccess) {
                    _dev_sub_buffer.clear();
                    THROW("hipMalloc of size " + TOSTR(_sub_buffer_size[sub_idx]) + " index " + TOSTR(sub_idx) +
                          " failed " + TOSTR(err));
                }
                err = hipHostMalloc((void **)&_dev_roi_buffers[buffIdx][sub_idx], roi_buffer_size[sub_idx], hipHostMallocDefault);  // Allocate HIP page locked ROI buffers
                if (err != hipSuccess || !_dev_roi_buffers[buffIdx][sub_idx]) {
                    _dev_roi_buffers.clear();
                    THROW("hipHostMalloc of size " + TOSTR(roi_buffer_size[sub_idx]) + " failed " + TOSTR(err))
                }
            }
        }
    } else {
#endif
        for (size_t buffIdx = 0; buffIdx < BUFF_DEPTH; buffIdx++) {
            // a minimum of extra MEM_ALIGNMENT is allocated
            _host_sub_buffers[buffIdx].resize(sub_buffer_count);
            _host_roi_buffers[buffIdx].resize(sub_buffer_count);
            for (size_t sub_buff_idx = 0; sub_buff_idx < sub_buffer_count; sub_buff_idx++) {
                _host_sub_buffers[buffIdx][sub_buff_idx] = aligned_alloc(MEM_ALIGNMENT, MEM_ALIGNMENT * (_sub_buffer_size[sub_buff_idx] / MEM_ALIGNMENT + 1));
                _host_roi_buffers[buffIdx][sub_buff_idx] = static_cast<unsigned *>(malloc(roi_buffer_size[sub_buff_idx]));  // Allocate HOST ROI buffers
            }
        }
#if ENABLE_OPENCL || ENABLE_HIP
    }
#endif
}

void RingBuffer::initBoxEncoderMetaData(RocalMemType mem_type, size_t encoded_bbox_size, size_t encoded_labels_size) {
    _box_encoder = true;
#if ENABLE_HIP
    DeviceResourcesHip *dev_hip = static_cast<DeviceResourcesHip *>(_dev);
    if (_mem_type == RocalMemType::HIP) {
        if (dev_hip->hip_stream == nullptr || dev_hip->device_id == -1)
            THROW("initBoxEncoderMetaData::Error Hip Device is not initialzed");
        hipError_t err;
        for (size_t buffIdx = 0; buffIdx < BUFF_DEPTH; buffIdx++) {
            err = hipMalloc(&_dev_bbox_buffer[buffIdx], encoded_bbox_size);
            if (err != hipSuccess) {
                _dev_bbox_buffer.clear();
                THROW("hipMalloc of size " + TOSTR(encoded_bbox_size) + " failed " + TOSTR(err));
            }
            err = hipMalloc(&_dev_labels_buffer[buffIdx], encoded_labels_size);
            if (err != hipSuccess) {
                _dev_labels_buffer.clear();
                THROW("hipMalloc of size " + TOSTR(encoded_bbox_size) + " failed " + TOSTR(err));
            }
        }
    }
#elif ENABLE_OPENCL
        DeviceResources *dev_ocl = static_cast<DeviceResources *>(_dev);
        if (mem_type == RocalMemType::OCL) {
            if (dev_ocl->cmd_queue == nullptr || dev_ocl->device_id == nullptr || dev_ocl->context == nullptr)
                THROW("Error ocl structure needed since memory type is OCL");

            cl_int err = CL_SUCCESS;
            for (size_t buffIdx = 0; buffIdx < BUFF_DEPTH; buffIdx++) {
                _dev_bbox_buffer[buffIdx] = clCreateBuffer(dev_ocl->context, CL_MEM_READ_WRITE, encoded_bbox_size, NULL, &err);
                if (err) {
                    _dev_bbox_buffer.clear();
                    THROW("clCreateBuffer of size " + TOSTR(encoded_bbox_size) + " failed " + TOSTR(err));
                }
                _dev_labels_buffer[buffIdx] = clCreateBuffer(dev_ocl->context, CL_MEM_READ_WRITE, encoded_labels_size, NULL, &err);
                if (err) {
                    _dev_labels_buffer.clear();
                    THROW("clCreateBuffer of size " + TOSTR(encoded_labels_size) + " failed " + TOSTR(err));
                }
            }
        }
#else
    {
        if (_meta_data_sub_buffer_count < 2)
            THROW("Insufficient HOST metadata buffers for Box Encoder");
        // Check if sufficient data has been allocated for the labels and bbox host buffers if not reallocate
        for (size_t buffIdx = 0; buffIdx < BUFF_DEPTH; buffIdx++) {
            if (_meta_data_sub_buffer_size[BUFF_DEPTH][0] < encoded_labels_size)
                rellocate_meta_data_buffer(_host_meta_data_buffers[BUFF_DEPTH][0], _meta_data_sub_buffer_size[BUFF_DEPTH][0], 0);
            if (_meta_data_sub_buffer_size[BUFF_DEPTH][1] < encoded_bbox_size)
                rellocate_meta_data_buffer(_host_meta_data_buffers[BUFF_DEPTH][1], _meta_data_sub_buffer_size[BUFF_DEPTH][1], 1);
        }
    }
#endif
}

void RingBuffer::init_metadata(RocalMemType mem_type, std::vector<size_t> &sub_buffer_size) {
    if (BUFF_DEPTH < 2)
        THROW("Error internal buffer size for the ring buffer should be greater than one")

    // Allocating buffers
    _meta_data_sub_buffer_count = sub_buffer_size.size();
    if (mem_type == RocalMemType::OCL || mem_type == RocalMemType::HIP) {
        THROW("Metadata is not supported with GPU backends")
    } else {
        _host_meta_data_buffers.resize(BUFF_DEPTH);
        _meta_data_sub_buffer_size.resize(BUFF_DEPTH);
        for (size_t buffIdx = 0; buffIdx < BUFF_DEPTH; buffIdx++) {
            _host_meta_data_buffers[buffIdx].resize(_meta_data_sub_buffer_count);
            for (size_t sub_buff_idx = 0; sub_buff_idx < _meta_data_sub_buffer_count; sub_buff_idx++) {
                _meta_data_sub_buffer_size[buffIdx].emplace_back(sub_buffer_size[sub_buff_idx]);
                _host_meta_data_buffers[buffIdx][sub_buff_idx] = malloc(sub_buffer_size[sub_buff_idx]);
            }
        }
    }
}

void RingBuffer::push() {
    // pushing and popping to and from image and metadata buffer should be atomic so that their level stays the same at all times
    std::unique_lock<std::mutex> lock(_names_buff_lock);
    _meta_ring_buffer.push(_last_image_meta_data);
    increment_write_ptr();
}

void RingBuffer::pop() {
    if (empty())
        return;
    // pushing and popping to and from image and metadata buffer should be atomic so that their level stays the same at all times
    std::unique_lock<std::mutex> lock(_names_buff_lock);
    increment_read_ptr();
    _meta_ring_buffer.pop();
}

void RingBuffer::reset() {
    _write_ptr = 0;
    _read_ptr = 0;
    _level = 0;
    _dont_block = false;
    while (!_meta_ring_buffer.empty())
        _meta_ring_buffer.pop();
}

void RingBuffer::release_gpu_res() {
#if ENABLE_HIP
    if (_mem_type == RocalMemType::HIP) {
        for (size_t buffIdx = 0; buffIdx < _dev_sub_buffer.size(); buffIdx++) {
            for (unsigned sub_buf_idx = 0; sub_buf_idx < _dev_sub_buffer[buffIdx].size(); sub_buf_idx++) {
                if (_dev_sub_buffer[buffIdx][sub_buf_idx])
                    if (hipFree((void *)_dev_sub_buffer[buffIdx][sub_buf_idx]) != hipSuccess) {
                        // printf("Error Freeing device buffer <%d, %d, %p>\n", buffIdx, sub_buf_idx, _dev_sub_buffer[buffIdx][sub_buf_idx]);
                        ERR("Could not release hip memory in the ring buffer")
                    }
                if (_dev_roi_buffers[buffIdx][sub_buf_idx]) {
                    if (hipHostFree((void *)_dev_roi_buffers[buffIdx][sub_buf_idx]) != hipSuccess) {
                        ERR("Could not release hip memory for ROI in the ring buffer")
                    }
                }
            }
            if (_host_meta_data_buffers.size() != 0) {
                for (unsigned sub_buf_idx = 0; sub_buf_idx < _host_meta_data_buffers[buffIdx].size(); sub_buf_idx++) {
                    if (_host_meta_data_buffers[buffIdx][sub_buf_idx])
                        free(_host_meta_data_buffers[buffIdx][sub_buf_idx]);
                }
            }
        }
        _dev_sub_buffer.clear();
        _host_meta_data_buffers.clear();
        _dev_roi_buffers.clear();
    }
#elif ENABLE_OPENCL
        if (_mem_type == RocalMemType::OCL) {
            for (size_t buffIdx = 0; buffIdx < _dev_sub_buffer.size(); buffIdx++) {
                for (unsigned sub_buf_idx = 0; sub_buf_idx < _dev_sub_buffer[buffIdx].size(); sub_buf_idx++) {
                    if (_dev_sub_buffer[buffIdx][sub_buf_idx])
                        if (clReleaseMemObject((cl_mem)_dev_sub_buffer[buffIdx][sub_buf_idx]) != CL_SUCCESS)
                            ERR("Could not release ocl memory in the ring buffer")
                }
                if (_host_meta_data_buffers.size() != 0) {
                    for (unsigned sub_buf_idx = 0; sub_buf_idx < _host_meta_data_buffers[buffIdx].size(); sub_buf_idx++) {
                        if (_host_meta_data_buffers[buffIdx][sub_buf_idx])
                            free(_host_meta_data_buffers[buffIdx][sub_buf_idx]);
                    }
                }
            }
            _dev_sub_buffer.clear();
            _host_meta_data_buffers.clear();
        }
#endif
}

RingBuffer::~RingBuffer() {
    if (_mem_type == RocalMemType::HOST) {
        for (unsigned buffIdx = 0; buffIdx < _host_sub_buffers.size(); buffIdx++) {
            for (unsigned sub_buf_idx = 0; sub_buf_idx < _host_sub_buffers[buffIdx].size(); sub_buf_idx++) {
                if (_host_sub_buffers[buffIdx][sub_buf_idx])
                    free(_host_sub_buffers[buffIdx][sub_buf_idx]);
                if (_host_roi_buffers[buffIdx][sub_buf_idx])
                    free(_host_roi_buffers[buffIdx][sub_buf_idx]);
            }
            if (_host_meta_data_buffers.size() != 0) {
                for (unsigned sub_buf_idx = 0; sub_buf_idx < _host_meta_data_buffers[buffIdx].size(); sub_buf_idx++) {
                    if (_host_meta_data_buffers[buffIdx][sub_buf_idx])
                        free(_host_meta_data_buffers[buffIdx][sub_buf_idx]);
                }
            }
        }
        _host_sub_buffers.clear();
        _host_meta_data_buffers.clear();
        _host_roi_buffers.clear();
    }
}

bool RingBuffer::empty() {
    return (_level <= 0);
}

bool RingBuffer::full() {
    return (_level >= BUFF_DEPTH - 1);
}

size_t RingBuffer::level() {
    return _level;
}
void RingBuffer::increment_read_ptr() {
    std::unique_lock<std::mutex> lock(_lock);
    _read_ptr = (_read_ptr + 1) % BUFF_DEPTH;
    _level--;
    lock.unlock();
    // Wake up the writer thread (in case waiting) since there is an empty spot to write to,
    _wait_for_unload.notify_all();
}

void RingBuffer::increment_write_ptr() {
    std::unique_lock<std::mutex> lock(_lock);
    _write_ptr = (_write_ptr + 1) % BUFF_DEPTH;
    _level++;
    lock.unlock();
    // Wake up the reader thread (in case waiting) since there is a new load to be read
    _wait_for_load.notify_all();
}

void RingBuffer::set_meta_data(ImageNameBatch names, pMetaDataBatch meta_data) {
    if (meta_data == nullptr)
        _last_image_meta_data = std::make_pair(std::move(names), pMetaDataBatch());
    else {
        _last_image_meta_data = std::make_pair(std::move(names), meta_data);
        if (!_box_encoder) {
            auto actual_buffer_size = meta_data->get_buffer_size();
            for (unsigned i = 0; i < actual_buffer_size.size(); i++) {
                if (actual_buffer_size[i] > _meta_data_sub_buffer_size[_write_ptr][i])
                    rellocate_meta_data_buffer(_host_meta_data_buffers[_write_ptr][i], actual_buffer_size[i], i);
            }
            meta_data->copy_data(_host_meta_data_buffers[_write_ptr]);
        }
    }
}

void RingBuffer::rellocate_meta_data_buffer(void *buffer, size_t buffer_size, unsigned buff_idx) {
    void *new_ptr = realloc(buffer, buffer_size);
    if (buffer == nullptr)
        THROW("Metadata ring buffer reallocation failed")
    _host_meta_data_buffers[_write_ptr][buff_idx] = new_ptr;
    _meta_data_sub_buffer_size[_write_ptr][buff_idx] = buffer_size;
}

MetaDataNamePair &RingBuffer::get_meta_data() {
    block_if_empty();
    std::unique_lock<std::mutex> lock(_names_buff_lock);
    if (_level != _meta_ring_buffer.size())
        THROW("ring buffer internals error, image and metadata sizes not the same " + TOSTR(_level) + " != " + TOSTR(_meta_ring_buffer.size()))
    return _meta_ring_buffer.front();
}
