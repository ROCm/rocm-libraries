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

#include <cassert>
#include "pipeline/commons.h"
#include <cstring>
#include <algorithm>
#include "readers/image/cifar10_data_reader.h"
#include "readers/file_source_reader.h"
#include "pipeline/filesystem.h"

CIFAR10DataReader::CIFAR10DataReader() {
    _src_dir = nullptr;
    _sub_dir = nullptr;
    _entity = nullptr;
    _curr_file_idx = 0;
    _current_file_size = 0;
    _current_fPtr = nullptr;
    _loop = false;
    _total_file_size = 0;
    _last_file_idx = 0;
    _file_count_all_shards = 0;
}

Reader::Status CIFAR10DataReader::initialize(ReaderConfig desc) {
    auto ret = Reader::Status::OK;
    _folder_path = desc.path();
    _batch_size = desc.get_batch_size();
    _loop = desc.loop();
    _file_name_prefix = desc.file_prefix();
    _sharding_info = desc.get_sharding_info();
    _pad_last_batch_repeated = _sharding_info.pad_last_batch_repeated;
    _stick_to_shard = _sharding_info.stick_to_shard;
    _shard_size = _sharding_info.shard_size;
    _shuffle = desc.shuffle();
    ret = subfolder_reading();
    _curr_file_idx = _shard_start_idx_vector[_shard_id]; // shard's start_idx would vary for every shard in the vector
    // shuffle dataset if set
    if (ret == Reader::Status::OK && _shuffle)
        std::random_shuffle(_file_names.begin() + _shard_start_idx_vector[_shard_id],
                            _file_names.begin() + _shard_end_idx_vector[_shard_id]);
    return ret;

}

void CIFAR10DataReader::incremenet_read_ptr() {
    _read_counter++;
    increment_curr_file_idx(_file_names.size());
}

size_t CIFAR10DataReader::open() {
    auto file_path = _file_names[_curr_file_idx];  // Get next file name
    auto file_offset = _file_offsets[_curr_file_idx];
    _last_file_idx = _file_idx[_curr_file_idx];
    incremenet_read_ptr();
    // update _last_id for the next record
    _last_id = file_path;
    auto last_slash_idx = _last_id.find_last_of("\\/");
    if (std::string::npos != last_slash_idx) {
        _last_id.erase(0, last_slash_idx + 1);
    }
    // add file_idx to last_id so the loader knows the index within the same master file
    _last_id.append("_");
    _last_id.append(std::to_string(_last_file_idx));
    // compare the file_name with the last one opened
    if (file_path.compare(_last_file_name) != 0) {
        if (_current_fPtr) {
            fclose(_current_fPtr);
            _current_fPtr = nullptr;
        }
        _current_fPtr = fopen(file_path.c_str(), "rb");  // Open the file,
        _last_file_name = file_path;
        fseek(_current_fPtr, 0, SEEK_END);  // Take the file read pointer to the end
        _total_file_size = ftell(_current_fPtr);
        fseek(_current_fPtr, 0, SEEK_SET);  // Take the file read pointer to the beginning
    }

    if (!_current_fPtr)  // Check if it is ready for reading
        return 0;

    fseek(_current_fPtr, file_offset, SEEK_END);  // Take the file read pointer to the end

    _current_file_size = ftell(_current_fPtr);  // Check how many bytes are there between and the current read pointer position (end of the file)

    if (_current_file_size < _raw_file_size)  // not enough data in the file to read
    {                                         // If file is empty continue
        fclose(_current_fPtr);
        _current_fPtr = nullptr;
        return 0;
    }

    fseek(_current_fPtr, file_offset + 1, SEEK_SET);  // Take the file pointer back to the fileoffset + 1 extra byte for label

    return (_raw_file_size - 1);
}

size_t CIFAR10DataReader::read_data(unsigned char* buf, size_t read_size) {
    if (!_current_fPtr)
        return 0;

    // Requested read size bigger than the raw file size? just read as many bytes as the raw file size
    read_size = (read_size > (_raw_file_size - 1)) ? _raw_file_size - 1 : read_size;

    size_t actual_read_size = fread(buf, sizeof(unsigned char), read_size, _current_fPtr);
    return actual_read_size;
}

int CIFAR10DataReader::close() {
    return release();
}

CIFAR10DataReader::~CIFAR10DataReader() {
    if (_current_fPtr) {
        fclose(_current_fPtr);
        _current_fPtr = nullptr;
    }
}

int CIFAR10DataReader::release() {
    // do not need to close file here since data is read from the same file continuously
    return 0;
}

void CIFAR10DataReader::reset() {
    if (_shuffle)
        std::random_shuffle(_file_names.begin() + _shard_start_idx_vector[_shard_id],
                            _file_names.begin() + _shard_end_idx_vector[_shard_id]);
    if (_stick_to_shard == false)  // Pick elements from the next shard - hence increment shard_id
        increment_shard_id();      // Should work for both single and multiple shards
    _read_counter = 0;
    if (_sharding_info.last_batch_policy == RocalBatchPolicy::DROP) {  // Skipping the dropped batch in next epoch
        for (uint32_t i = 0; i < _batch_size; i++)
            increment_curr_file_idx(_file_names.size());
    }
}

Reader::Status CIFAR10DataReader::subfolder_reading() {
    if ((_sub_dir = opendir(_folder_path.c_str())) == nullptr)
        THROW("CIFAR10DataReader ERROR: Failed opening the directory at " + _folder_path);

    std::vector<std::string> entry_name_list;
    std::string _full_path = _folder_path;

    while ((_entity = readdir(_sub_dir)) != nullptr) {
        std::string entry_name(_entity->d_name);
        if (strcmp(_entity->d_name, ".") == 0 || strcmp(_entity->d_name, "..") == 0) continue;
        entry_name_list.push_back(entry_name);
        LOG("CIFAR10DataReader  Got entry name " + entry_name)
    }
    std::sort(entry_name_list.begin(), entry_name_list.end());
    std::string subfolder_path = _full_path + "/" + entry_name_list[0];
    filesys::path pathObj(subfolder_path);
    auto ret = Reader::Status::OK;
    if (filesys::exists(pathObj) && filesys::is_regular_file(pathObj)) {
        ret = open_folder();
    } else if (filesys::exists(pathObj) && filesys::is_directory(pathObj)) {
        for (unsigned dir_count = 0; dir_count < entry_name_list.size(); ++dir_count) {
            std::string subfolder_path = _full_path + "/" + entry_name_list[dir_count];
            _folder_path = subfolder_path;
            if (open_folder() != Reader::Status::OK)
                WRN("CIFAR10DataReader: File reader cannot access the storage at " + _folder_path);
        }
    }
    if (!_file_names.empty())
        LOG("CIFAR10DataReader  Total of " + TOSTR(_file_names.size()) + " images loaded from " + _full_path)

    auto dataset_size = _file_count_all_shards;
    size_t padded_samples = 0;
    // Pad the _file_names with last element of the shard in the vector when _pad_last_batch_repeated is True
    padded_samples = ((_shard_size > 0) ? _shard_size : largest_shard_size_without_padding()) % _batch_size;
    _last_batch_padded_size = ((_batch_size > 1) && (padded_samples > 0)) ? (_batch_size - padded_samples) : 0;

    if (_pad_last_batch_repeated == true) {
        // pad the last sample when the dataset_size is not divisible by
        // the number of shard's (or) when the shard's size is not
        // divisible by the batch size making each shard having equal
        // number of samples
        uint32_t total_padded_samples = 0; // initialize the total_padded_samples to 0
        for (uint32_t shard_id = 0; shard_id < _shard_count; shard_id++) {
            uint32_t start_idx = (dataset_size * shard_id) / _shard_count;
            uint32_t actual_shard_size_without_padding = std::floor((shard_id + 1) * dataset_size / _shard_count) - std::floor(shard_id * dataset_size / _shard_count);
            uint32_t largest_shard_size = std::ceil(dataset_size * 1.0 / _shard_count);
            auto start = _file_names.begin() + start_idx + total_padded_samples;
            auto end = start + actual_shard_size_without_padding;
            auto start_offset = _file_offsets.begin() + start_idx + total_padded_samples;
            auto end_offset = start_offset + actual_shard_size_without_padding;
            auto start_file_idx = _file_idx.begin() + start_idx + total_padded_samples;
            auto end_file_idx = start_file_idx + actual_shard_size_without_padding;
            if (largest_shard_size % _batch_size) {
                size_t num_padded_samples = 0;
                num_padded_samples = (largest_shard_size - actual_shard_size_without_padding) + _batch_size - (largest_shard_size % _batch_size);
                _file_count_all_shards += num_padded_samples;
                _file_names.insert(end, num_padded_samples, _file_names[start_idx + actual_shard_size_without_padding + total_padded_samples - 1]);
                _file_offsets.insert(end_offset, num_padded_samples, _file_offsets[start_idx + actual_shard_size_without_padding + total_padded_samples - 1]);
                _file_idx.insert(end_file_idx, num_padded_samples, _file_idx[start_idx + actual_shard_size_without_padding + total_padded_samples - 1]);
                total_padded_samples += num_padded_samples;
            }
        }
    }
    _last_file_name = _file_names[_file_names.size() - 1];
    compute_start_and_end_idx_of_all_shards();
    closedir(_sub_dir);
    return ret;
}

Reader::Status CIFAR10DataReader::open_folder() {
    if ((_src_dir = opendir(_folder_path.c_str())) == nullptr)
        THROW("CIFAR10DataReader ERROR: Failed opening the directory at " + _folder_path);

    while ((_entity = readdir(_src_dir)) != nullptr) {
        if (_entity->d_type != DT_REG)
            continue;
        std::string file_path = _folder_path;
        // check if the filename has the _file_name_prefix
        std::string data_file_name = std::string(_entity->d_name);
        if (data_file_name.find(_file_name_prefix) != std::string::npos) {
            file_path.append("/");
            file_path.append(_entity->d_name);
            FILE* fp = fopen(file_path.c_str(), "rb");  // Open the file,
            fseek(fp, 0, SEEK_END);                     // Take the file read pointer to the end
            size_t total_file_size = ftell(fp);
            size_t num_of_raw_files = _raw_file_size ? total_file_size / _raw_file_size : 0;
            unsigned file_offset = 0;
            for (unsigned i = 0; i < num_of_raw_files; i++) {
                _file_names.push_back(file_path);
                _file_offsets.push_back(file_offset);
                _file_idx.push_back(i);
                _file_count_all_shards++;
                file_offset += _raw_file_size;
            }
            fclose(fp);
        }
    }
    if (_file_names.empty())
        WRN("CIFAR10DataReader:: Did not load any file from " + _folder_path)

    closedir(_src_dir);
    return Reader::Status::OK;
}
