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
#include <algorithm>
#include <cstring>
#include "pipeline/commons.h"
#include "readers/video/sequence_file_source_reader.h"
#include "pipeline/filesystem.h"

SequenceFileSourceReader::SequenceFileSourceReader() {
    _src_dir = nullptr;
    _sub_dir = nullptr;
    _entity = nullptr;
    _curr_file_idx = 0;
    _current_file_size = 0;
    _current_fPtr = nullptr;
    _loop = false;
    _sequence_id = 0;
    _shuffle = false;
    _sequence_count_all_shards = 0;
}

unsigned SequenceFileSourceReader::count_items() {
    if (_loop)
        return _frame_names.size();
    int ret = ((int)_frame_names.size() - _read_counter);
    return ((ret < 0) ? 0 : ret);
}

Reader::Status SequenceFileSourceReader::initialize(ReaderConfig desc) {
    auto ret = Reader::Status::OK;
    _sequence_id = 0;
    _folder_path = desc.path();
    _shard_id = desc.get_shard_id();
    _shard_count = desc.get_shard_count();
    _user_batch_count = desc.get_batch_size();
    _shuffle = desc.shuffle();
    _loop = desc.loop();
    _sequence_length = desc.get_sequence_length();
    _step = desc.get_frame_step();
    _stride = desc.get_frame_stride();
    _batch_count = _user_batch_count / _sequence_length;
    ret = subfolder_reading();
    if (ret != Reader::Status::OK)
        return ret;
    ret = get_sequences();

    // the following code is required to make every shard the same size:: required for multi-gpu training
    if (_shard_count > 1 && _batch_count > 1) {
        int _num_batches = _sequence_frame_names.size() / _batch_count;
        int max_batches_per_shard = (_sequence_count_all_shards + _shard_count - 1) / _shard_count;
        max_batches_per_shard = (max_batches_per_shard + _batch_count - 1) / _batch_count;
        if (_num_batches < max_batches_per_shard) {
            replicate_last_batch_to_pad_partial_shard();
        }
    }

    // shuffle dataset if set
    if (ret == Reader::Status::OK && _shuffle)
        std::random_shuffle(_sequence_frame_names.begin(), _sequence_frame_names.end());

    for (auto &&seq : _sequence_frame_names) {
        _frame_names.insert(_frame_names.end(), seq.begin(), seq.end());
    }
    return ret;
}

void SequenceFileSourceReader::incremenet_read_ptr() {
    _read_counter++;
    _curr_file_idx = (_curr_file_idx + 1) % _frame_names.size();
}

size_t SequenceFileSourceReader::open() {
    auto file_path = _frame_names[_curr_file_idx];  // Get next file name
    incremenet_read_ptr();
    _last_id = file_path;
    auto last_slash_idx = _last_id.find_last_of("\\/");
    if (std::string::npos != last_slash_idx) {
        _last_id.erase(0, last_slash_idx + 1);
    }
    _current_fPtr = fopen(file_path.c_str(), "rb");  // Open the file,
    if (!_current_fPtr)                              // Check if it is ready for reading
        return 0;
    fseek(_current_fPtr, 0, SEEK_END);          // Take the file read pointer to the end
    _current_file_size = ftell(_current_fPtr);  // Check how many bytes are there between and the current read pointer position (end of the file)
    if (_current_file_size == 0) {
        // If file is empty continue
        fclose(_current_fPtr);
        _current_fPtr = nullptr;
        return 0;
    }
    fseek(_current_fPtr, 0, SEEK_SET);  // Take the file pointer back to the start
    return _current_file_size;
}

size_t SequenceFileSourceReader::read_data(unsigned char *buf, size_t read_size) {
    if (!_current_fPtr)
        return 0;

    // Requested read size bigger than the file size? just read as many bytes as the file size
    read_size = (read_size > _current_file_size) ? _current_file_size : read_size;
    size_t actual_read_size = fread(buf, sizeof(unsigned char), read_size, _current_fPtr);
    return actual_read_size;
}

int SequenceFileSourceReader::close() {
    return release();
}

SequenceFileSourceReader::~SequenceFileSourceReader() {
    release();
}

int SequenceFileSourceReader::release() {
    if (!_current_fPtr)
        return 0;
    fclose(_current_fPtr);
    _current_fPtr = nullptr;
    return 0;
}

void SequenceFileSourceReader::reset() {
    if (_shuffle)
        std::random_shuffle(_sequence_frame_names.begin(), _sequence_frame_names.end());

    _read_counter = 0;
    _curr_file_idx = 0;
}

Reader::Status SequenceFileSourceReader::get_sequences() {
    for (unsigned folder_idx = 0; folder_idx < _folder_file_names.size(); folder_idx++) {
        if (_folder_file_names[folder_idx].size() == 0) {
            WRN("\nFolder #" + TOSTR(folder_idx) + "does not have any files")
            continue;
        }
        if (_sequence_length > _folder_file_names[folder_idx].size()) {
            THROW("Sequence length is not valid");
        }
        for (unsigned file_idx = 0; (file_idx + (_stride * (_sequence_length - 1))) < _folder_file_names[folder_idx].size(); file_idx += _step) {
            if (get_sequence_shard_id() != _shard_id) {
                _sequence_count_all_shards++;
                incremenet_sequence_id();
                continue;
            }
            _in_batch_read_count++;
            _in_batch_read_count = (_in_batch_read_count % _batch_count == 0) ? 0 : _in_batch_read_count;
            std::vector<std::string> temp_sequence;
            for (unsigned frame_count = 0, frame_idx = file_idx; (frame_count < _sequence_length); frame_count++, frame_idx += _stride) {
                temp_sequence.push_back(_folder_file_names[folder_idx][frame_idx]);
            }
            _last_sequence = temp_sequence;
            _sequence_frame_names.push_back(temp_sequence);
            _sequence_count_all_shards++;
            incremenet_sequence_id();
        }
    }
    if (_sequence_frame_names.empty())
        WRN("SequenceReader ShardID [" + TOSTR(_shard_id) + "] Did not load any file from " + _folder_path)
    if (_in_batch_read_count > 0 && _in_batch_read_count < _batch_count) {
        replicate_last_sequence_to_fill_last_shard();
        LOG("SequenceReader ShardID [" + TOSTR(_shard_id) + "] Replicated last sequence " + TOSTR((_batch_count - _in_batch_read_count)) + " times to fill the last batch")
    }
    return Reader::Status::OK;
}

Reader::Status SequenceFileSourceReader::subfolder_reading() {
    if ((_sub_dir = opendir(_folder_path.c_str())) == nullptr)
        THROW("SequenceReader ShardID [" + TOSTR(_shard_id) + "] ERROR: Failed opening the directory at " + _folder_path);
    std::vector<std::string> entry_name_list;
    std::string _full_path = _folder_path;
    while ((_entity = readdir(_sub_dir)) != nullptr) {
        std::string entry_name(_entity->d_name);
        if (strcmp(_entity->d_name, ".") == 0 || strcmp(_entity->d_name, "..") == 0)
            continue;
        entry_name_list.push_back(entry_name);
    }
    closedir(_sub_dir);
    std::sort(entry_name_list.begin(), entry_name_list.end());
    auto ret = Reader::Status::OK;
    for (unsigned dir_count = 0; dir_count < entry_name_list.size(); ++dir_count) {
        std::string subfolder_path = _full_path + "/" + entry_name_list[dir_count];
        filesys::path pathObj(subfolder_path);
        if (filesys::exists(pathObj) && filesys::is_regular_file(pathObj)) {
            // ignore files with extensions .tar, .zip, .7z, .mp4
            auto file_extension_idx = subfolder_path.find_last_of(".");
            if (file_extension_idx != std::string::npos) {
                std::string file_extension = subfolder_path.substr(file_extension_idx + 1);
                if ((file_extension == "tar") || (file_extension == "zip") || (file_extension == "7z") || (file_extension == "rar") || (file_extension == "mp4"))
                    continue;
            }
            _file_names.push_back(subfolder_path);
        } else if (filesys::exists(pathObj) && filesys::is_directory(pathObj)) {
            _folder_path = subfolder_path;
            if (open_folder() != Reader::Status::OK)
                WRN("SequenceReader ShardID [" + TOSTR(_shard_id) + "] File reader cannot access the storage at " + _folder_path);
            _folder_file_names.push_back(_file_names);
            _file_names.clear();
        }
    }
    if (!_file_names.empty()) {
        _folder_file_names.push_back(_file_names);
        _file_names.clear();
    }
    return ret;
}

void SequenceFileSourceReader::replicate_last_sequence_to_fill_last_shard() {
    for (size_t i = _in_batch_read_count; i < _batch_count; i++)
        _sequence_frame_names.push_back(_last_sequence);
}

void SequenceFileSourceReader::replicate_last_batch_to_pad_partial_shard() {
    if (_sequence_frame_names.size() >= _batch_count) {
        for (size_t i = 0; i < _batch_count; i++)
            _sequence_frame_names.push_back(_sequence_frame_names[i - _batch_count]);
    }
}

Reader::Status SequenceFileSourceReader::open_folder() {
    if ((_src_dir = opendir(_folder_path.c_str())) == nullptr)
        THROW("SequenceReader ShardID [" + TOSTR(_shard_id) + "] ERROR: Failed opening the directory at " + _folder_path);
    while ((_entity = readdir(_src_dir)) != nullptr) {
        if (_entity->d_type != DT_REG)
            continue;

        std::string file_path = _folder_path;
        file_path.append("/");
        file_path.append(_entity->d_name);
        _file_names.push_back(file_path);
    }
    std::sort(_file_names.begin(), _file_names.end());
    closedir(_src_dir);
    return Reader::Status::OK;
}

size_t SequenceFileSourceReader::get_sequence_shard_id() {
    if (_batch_count == 0 || _shard_count == 0)
        THROW("Shard (Batch) size cannot be set to 0")
    // return (_sequence_id / (_batch_count)) % _shard_count;
    return _sequence_id % _shard_count;
}
