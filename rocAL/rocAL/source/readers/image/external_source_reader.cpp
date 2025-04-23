/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#include "readers/image/external_source_reader.h"

#include <algorithm>
#include <cassert>

ExternalSourceReader::ExternalSourceReader() {
    _curr_file_idx = 0;
    _current_file_size = 0;
    _current_fPtr = nullptr;
    _file_id = 0;
    _shuffle = false;
    _file_count_all_shards = 0;
    _loop = false;  // loop not supported for external source
    _end_of_sequence = false;
}

// return batch_size() for count_items unless end_of_sequence has been signalled.
unsigned ExternalSourceReader::count_items() {
    if (_file_mode == ExternalSourceFileMode::FILENAME) {
        if (_end_of_sequence && _file_names_queue.empty()) {
            return 0;
        }
    } else {
        if (_end_of_sequence && _images_data_queue.empty()) {
            return 0;
        }
    }
    return _batch_count;
}

Reader::Status ExternalSourceReader::initialize(ReaderConfig desc) {
    auto ret = Reader::Status::OK;
    _folder_path = desc.path();
    _file_id = 0;
    _shard_id = desc.get_shard_id();
    _shard_count = desc.get_shard_count();
    _batch_count = desc.get_batch_size();
    _shuffle = desc.shuffle();
    _loop = desc.loop();
    _file_mode = desc.mode();
    _end_of_sequence = false;
    _file_data.reserve(_batch_count);
    return ret;
}

void ExternalSourceReader::increment_read_ptr() {
    _read_counter++;
    _curr_file_idx = (_curr_file_idx + 1) % _batch_count;
}

size_t ExternalSourceReader::open() {
    if (_file_mode == ExternalSourceFileMode::FILENAME) {
        std::string next_file_name;
        bool ret = pop_file_name(next_file_name);  // Get next file name: blocking call, will wait till next file is received from external source
        if (_end_of_sequence && !ret)
            return 0;
        _last_id = next_file_name;
        filesys::path pathObj(next_file_name);
        if (filesys::exists(pathObj) && filesys::is_regular_file(pathObj)) {
            _current_fPtr = fopen(next_file_name.c_str(), "rb");  // Open the file,
            if (!_current_fPtr)                                   // Check if it is ready for reading
                return 0;
            fseek(_current_fPtr, 0, SEEK_END);          // Take the file read pointer to the end
            _current_file_size = ftell(_current_fPtr);  // Check how many bytes are there between and the current read pointer position (end of the file)
            if (_current_file_size == 0) {              // If file is empty continue
                fclose(_current_fPtr);
                _current_fPtr = nullptr;
                return 0;
            }
            fseek(_current_fPtr, 0, SEEK_SET);  // Take the file pointer back to the start
            ExternalSourceImageInfo image_info;
            image_info.file_data = (unsigned char*)next_file_name.data();
            image_info.file_read_size = _current_file_size;
            _file_data[_curr_file_idx] = image_info;
            increment_read_ptr();
        }
    } else {
        ExternalSourceImageInfo image_info;
        bool ret = pop_file_data(image_info);
        if (_end_of_sequence && !ret) {
            WRN(" EOS || POP FAILED ")
            return 0;
        }
        _file_data[_curr_file_idx] = image_info;
        _current_file_size = image_info.file_read_size;
    }
    return _current_file_size;
}

size_t ExternalSourceReader::read_data(unsigned char* buf, size_t read_size) {
    if (_file_mode == ExternalSourceFileMode::FILENAME) {
        if (!_current_fPtr)
            return 0;

        // Requested read size bigger than the file size? just read as many bytes as the file size
        read_size = std::min(static_cast<unsigned int>(read_size), _current_file_size);
        size_t actual_read_size = fread(buf, sizeof(unsigned char), read_size, _current_fPtr);
        return actual_read_size;
    } else {
        unsigned char* file_data_ptr = _file_data[_curr_file_idx].file_data;
        size_t size = _current_file_size;
        if (size > read_size)
            THROW("Requested size doesn't match the actual size for file read")
        memcpy(static_cast<void*>(buf), static_cast<void*>(file_data_ptr), size);
        increment_read_ptr();
        return size;
    }
}

void ExternalSourceReader::get_dims(int cur_idx, int& width, int& height, int& channels, unsigned& roi_width, unsigned& roi_height) {
    if (cur_idx >= 0) {
        width = _file_data[cur_idx].width;
        height = _file_data[cur_idx].height;
        channels = _file_data[cur_idx].channels;
        roi_width = _file_data[cur_idx].roi_width;
        roi_height = _file_data[cur_idx].roi_height;
    }
}

int ExternalSourceReader::close() {
    return release();
}

ExternalSourceReader::~ExternalSourceReader() {
    release();
}

int ExternalSourceReader::release() {
    if (_file_mode != ExternalSourceFileMode::FILENAME) {
        if (!_current_fPtr)
            return 0;
        fclose(_current_fPtr);
        _current_fPtr = nullptr;
        _end_of_sequence = false;  // reset for looping
    }
    return 0;
}

void ExternalSourceReader::reset() {
    _read_counter = 0;
    _curr_file_idx = 0;
    _end_of_sequence = false;  // reset for looping
}

size_t ExternalSourceReader::get_file_shard_id() {
    if (_batch_count == 0 || _shard_count == 0)
        THROW("Shard (Batch) size cannot be set to 0")
    return _file_id % _shard_count;
}

void ExternalSourceReader::push_file_name(const std::string& file_name) {
    std::unique_lock<std::mutex> lock(_lock);
    _file_names_queue.push(file_name);
    lock.unlock();
    // notify waiting thread of new data
    _wait_for_input.notify_all();
}

bool ExternalSourceReader::pop_file_name(std::string& file_name) {
    std::unique_lock<std::mutex> lock(_lock);
    if (_file_names_queue.empty() && !_end_of_sequence)
        _wait_for_input.wait(lock);
    if (!_file_names_queue.empty()) {
        file_name = _file_names_queue.front();
        _file_names_queue.pop();
        return true;
    } else
        return false;
}

void ExternalSourceReader::push_file_data(ExternalSourceImageInfo& image_info) {
    std::unique_lock<std::mutex> lock(_lock);
    _images_data_queue.push(image_info);
    lock.unlock();
    // notify waiting thread of new data
    _wait_for_input.notify_all();
}

bool ExternalSourceReader::pop_file_data(ExternalSourceImageInfo& image_info) {
    std::unique_lock<std::mutex> lock(_lock);
    if (_images_data_queue.empty() && !_end_of_sequence)
        _wait_for_input.wait(lock);
    if (!_images_data_queue.empty()) {
        image_info = _images_data_queue.front();
        _images_data_queue.pop();
        return true;
    } else
        return false;
}

void ExternalSourceReader::feed_file_names(const std::vector<std::string>& file_names, size_t num_images, bool eos) {
    for (unsigned n = 0; n < num_images; n++) {
        push_file_name(file_names[n]);
    }
    _end_of_sequence = eos;
}

void ExternalSourceReader::feed_data(const std::vector<unsigned char*>& images, const std::vector<size_t>& image_size, ExternalSourceFileMode mode, bool eos, const std::vector<unsigned> roi_width, const std::vector<unsigned> roi_height, unsigned int width, unsigned int height, unsigned int channels) {
    if (mode == ExternalSourceFileMode::RAWDATA_COMPRESSED) {
        for (unsigned n = 0; n < images.size(); n++) {
            ExternalSourceImageInfo image_info = {images[n], image_size[n], width, height, channels, 0, 0};
            push_file_data(image_info);
        }
    } else {
        for (unsigned n = 0; n < images.size(); n++) {
            ExternalSourceImageInfo image_info = {images[n], image_size[n], width, height, channels, roi_width[n], roi_height[n]};
            push_file_data(image_info);
        }
    }
    _end_of_sequence = eos;
}
