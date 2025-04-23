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

#include "readers/image/image_reader.h"

void Reader::increment_curr_file_idx(size_t dataset_size) {
    // The condition satisfies for both pad_last_batch = True (or) False
    if (_stick_to_shard == false) {  // The elements of each shard rotate in a round-robin fashion once the elements in particular shard is exhausted
        _curr_file_idx = (_curr_file_idx + 1) % dataset_size;
    } else {
        if (_curr_file_idx >= _shard_start_idx_vector[_shard_id] &&
            _curr_file_idx < _shard_end_idx_vector[_shard_id]) // checking if current-element lies within the shard size [begin_idx, last_idx -1]
            _curr_file_idx = (_curr_file_idx + 1);
        else
            _curr_file_idx = _shard_start_idx_vector[_shard_id];
    }
}

unsigned Reader::count_items() {
    int size = get_max_size_of_shard(_batch_size, _loop);
    if (_loop) return size;

    int ret = (size - _read_counter);
    if (_sharding_info.last_batch_policy == RocalBatchPolicy::DROP && _last_batch_padded_size != 0)
        ret -= _batch_size;
    return ((ret < 0) ? 0 : ret);
}

void Reader::compute_start_and_end_idx_of_all_shards() {
    for (uint32_t shard_id = 0; shard_id < _shard_count; shard_id++) {
        auto start_idx_of_shard = (_file_count_all_shards * shard_id) / _shard_count;
        auto end_idx_of_shard = start_idx_of_shard + actual_shard_size_without_padding() - 1;
        _shard_start_idx_vector.push_back(start_idx_of_shard);
        _shard_end_idx_vector.push_back(end_idx_of_shard);
    }
}

void Reader::increment_shard_id() {
    _shard_id = (_shard_id + 1) % _shard_count;
}

size_t Reader::actual_shard_size_without_padding() {
    return std::floor((_shard_id + 1) * _file_count_all_shards / _shard_count) - std::floor(_shard_id * _file_count_all_shards / _shard_count);
}

size_t Reader::largest_shard_size_without_padding() {
    return std::ceil(_file_count_all_shards * 1.0 / _shard_count);
}

size_t Reader::get_max_size_of_shard(size_t batch_size, bool loop) {
    int size = 0;
    if (_shard_size == -1) { // When shard_size is set to -1, The shard_size variable is not used
        if (loop) 
            return largest_shard_size_without_padding();  // Return the size of the largest shard amongst all the shard's size
        size = std::max(largest_shard_size_without_padding(), batch_size);
    } else if (_shard_size > 0) {
        auto largest_shard_size_with_padding = _shard_size + (batch_size - (_shard_size % batch_size));  // The shard size used here is padded
        if (loop)
            return largest_shard_size_with_padding;
        size = std::max(largest_shard_size_with_padding, batch_size);
    }
    return size;
}

void Reader::update_filenames_with_padding(std::vector<std::string> &file_names, size_t batch_size) {
    // pad the last sample when the dataset_size is not divisible by
    // the number of shard's (or) when the shard's size is not
    // divisible by the batch size making each shard having equal
    // number of samples
    size_t dataset_size = _file_count_all_shards;
    uint32_t total_padded_samples = 0; // initialize the total_padded_samples to 0
    for (uint32_t shard_id = 0; shard_id < _shard_count; shard_id++) {
        uint32_t start_idx = (dataset_size * shard_id) / _shard_count;
        uint32_t actual_shard_size_without_padding = std::floor((shard_id + 1) * dataset_size / _shard_count) - std::floor(shard_id * dataset_size / _shard_count);
        uint32_t largest_shard_size = std::ceil(dataset_size * 1.0 / _shard_count);
        auto start = file_names.begin() + start_idx + total_padded_samples;
        auto end = start + actual_shard_size_without_padding;
        if (largest_shard_size % batch_size) {
            size_t num_padded_samples = 0;
            num_padded_samples = (largest_shard_size - actual_shard_size_without_padding) + batch_size - (largest_shard_size % batch_size);
            _file_count_all_shards += num_padded_samples;
            file_names.insert(end, num_padded_samples, file_names[start_idx + actual_shard_size_without_padding + total_padded_samples - 1]);
            total_padded_samples += num_padded_samples;
        }
    }
}
