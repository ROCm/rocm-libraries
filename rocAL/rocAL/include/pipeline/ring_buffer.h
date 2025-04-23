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
#include <condition_variable>
#include <vector>

#if ENABLE_OPENCL
#include <CL/cl.h>
#endif
#include <queue>

#include "pipeline/commons.h"
#include "device/device_manager.h"
#include "device/device_manager_hip.h"
#include "meta_data/meta_data.h"

using MetaDataNamePair = std::pair<ImageNameBatch, pMetaDataBatch>;
class RingBuffer {
   public:
    explicit RingBuffer(unsigned buffer_depth);
    ~RingBuffer();
    size_t level();
    bool empty();
    ///\param mem_type
    ///\param dev
    ///\param sub_buffer_size
    ///\param sub_buffer_count
    void init(RocalMemType mem_type, void *dev, std::vector<size_t> &sub_buffer_size, std::vector<size_t> &roi_buffer_size);
    void initBoxEncoderMetaData(RocalMemType mem_type, size_t encoded_bbox_size, size_t encoded_labels_size);
    void init_metadata(RocalMemType mem_type, std::vector<size_t> &sub_buffer_size);
    void release_gpu_res();
    std::pair<std::vector<void *>, std::vector<unsigned *>> get_read_buffers();
    std::pair<std::vector<void *>, std::vector<unsigned *>> get_write_buffers();
    std::pair<void *, void *> get_box_encode_write_buffers();
    std::pair<void *, void *> get_box_encode_read_buffers();
    MetaDataNamePair &get_meta_data();
    std::vector<void *> get_meta_read_buffers();
    std::vector<void *> get_meta_write_buffers();
    void set_meta_data(ImageNameBatch names, pMetaDataBatch meta_data);
    void rellocate_meta_data_buffer(void *buffer, size_t buffer_size, unsigned buff_idx);
    void reset();
    void pop();
    void push();
    void unblock_reader();
    void unblock_writer();
    void release_all_blocked_calls();
    RocalMemType mem_type() { return _mem_type; }
    void block_if_empty();
    void block_if_full();
    void release_if_empty();

   private:
    std::queue<MetaDataNamePair> _meta_ring_buffer;
    MetaDataNamePair _last_image_meta_data;
    void increment_read_ptr();
    void increment_write_ptr();
    bool full();
    const unsigned BUFF_DEPTH;
    std::vector<size_t> _sub_buffer_size;
    std::vector<std::vector<size_t>> _meta_data_sub_buffer_size;
    unsigned _meta_data_sub_buffer_count;
    std::mutex _lock;
    std::condition_variable _wait_for_load;
    std::condition_variable _wait_for_unload;
    std::vector<std::vector<void *>> _dev_sub_buffer;
    std::vector<std::vector<void *>> _host_sub_buffers;
    std::vector<std::vector<unsigned *>> _dev_roi_buffers;
    std::vector<std::vector<unsigned *>> _host_roi_buffers;
    std::vector<std::vector<void *>> _host_meta_data_buffers;
    std::vector<void *> _dev_bbox_buffer;
    std::vector<void *> _dev_labels_buffer;
    bool _dont_block = false;
    RocalMemType _mem_type;
    void *_dev;
    size_t _write_ptr;
    size_t _read_ptr;
    size_t _level;
    std::mutex _names_buff_lock;
    const size_t MEM_ALIGNMENT = 256;
    bool _box_encoder = false;
};
