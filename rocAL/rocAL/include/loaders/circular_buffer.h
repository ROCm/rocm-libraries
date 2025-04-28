/*
Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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
struct DecodedDataInfo {
    std::vector<std::string> _data_names;
    std::vector<uint32_t> _roi_width;
    std::vector<uint32_t> _roi_height;
    std::vector<uint32_t> _original_width;
    std::vector<uint32_t> _original_height;
    std::vector<uint32_t> _audio_samples; //! Amplitude of an audio signal at a specific point in time
    std::vector<uint32_t> _audio_channels; //! Number of audio channels in an audio signal
    std::vector<float> _audio_sample_rates; //! The number of samples of audio carried per second
};

struct CropImageInfo {
    // Batch of Image Crop Coordinates in "xywh" format
    std::vector<std::vector<float>> _crop_image_coords;
};
class CircularBuffer {
   public:
    CircularBuffer(void* devres);
    ~CircularBuffer();
    void init(RocalMemType output_mem_type, size_t output_mem_size, size_t buff_depth, bool use_hip_memory = false);
    void release();         // release resources
    void sync();            // Syncs device buffers with host
    void unblock_reader();  // Unblocks the thread currently waiting on a call to get_read_buffer
    void unblock_writer();  // Unblocks the thread currently waiting on get_write_buffer
    void push();            // The latest write goes through, effectively adds one element to the buffer
    void pop();             // The oldest write will be erased and overwritten in upcoming writes
    void set_decoded_data_info(const DecodedDataInfo& info) { _last_data_info = info; }
    void set_crop_image_info(const CropImageInfo& info) { _last_crop_image_info = info; }
    DecodedDataInfo& get_decoded_data_info();
    CropImageInfo& get_cropped_image_info();
    bool random_bbox_crop_flag = false;
    void* get_read_buffer_dev();
    unsigned char* get_read_buffer_host();  // blocks the caller if the buffer is empty
    unsigned char* get_write_buffer();      // blocks the caller if the buffer is full
    size_t level();                         // Returns the number of elements stored
    void reset();                           // sets the buffer level to 0
    void block_if_empty();                  // blocks the caller if the buffer is empty
    void block_if_full();                   // blocks the caller if the buffer is full

   private:
    void increment_read_ptr();
    void increment_write_ptr();
    bool full();
    bool empty();
    size_t _buff_depth;
    DecodedDataInfo _last_data_info;
    std::queue<DecodedDataInfo> _circ_buff_data_info;    //!< Stores the loaded data names, decoded_width and decoded_height(data is stored in the _circ_buff)
    CropImageInfo _last_crop_image_info;              // for Random BBox crop coordinates
    std::queue<CropImageInfo> _circ_crop_image_info;  //!< Stores the crop coordinates of the images for random bbox crop (data is stored in the _circ_buff)
    std::mutex _names_buff_lock;
#if ENABLE_HIP
    hipStream_t _hip_stream;
    int _hip_device_id, _hip_canMapHostMemory;
#elif ENABLE_OPENCL
    cl_command_queue _cl_cmdq = nullptr;
    cl_context _cl_context = nullptr;
    cl_device_id _device_id = nullptr;
#endif
    std::vector<void*> _dev_buffer;  // Actual memory allocated on the device (in the case of GPU affinity)
    std::vector<unsigned char*> _host_buffer_ptrs;
    std::condition_variable _wait_for_load;
    std::condition_variable _wait_for_unload;
    std::mutex _lock;
    RocalMemType _output_mem_type;
    size_t _output_mem_size;
    bool _initialized = false;
    const size_t MEM_ALIGNMENT = 256;
    size_t _write_ptr;
    size_t _read_ptr;
    size_t _level;
    bool _use_pinned_memory = true;
};