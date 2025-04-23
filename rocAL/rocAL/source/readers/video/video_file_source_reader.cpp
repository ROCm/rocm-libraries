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

#include "readers/video/video_file_source_reader.h"

#ifdef ROCAL_VIDEO
VideoFileSourceReader::VideoFileSourceReader() {
    _curr_sequence_idx = 0;
    _loop = false;
    _sequence_id = 0;
    _shuffle = false;
    _sequence_count_all_shards = 0;
}

unsigned VideoFileSourceReader::count_items() {
    if (_loop)
        return _total_sequences_count;
    int ret = (int)(_total_sequences_count - _read_counter);
    return ((ret <= 0) ? 0 : ret);
}

VideoReader::Status VideoFileSourceReader::initialize(ReaderConfig desc) {
    auto ret = VideoReader::Status::OK;
    _sequence_id = 0;
    _folder_path = desc.path();
    _shard_id = desc.get_shard_id();
    _shard_count = desc.get_shard_count();
    _shuffle = desc.shuffle();
    _loop = desc.loop();
    _video_prop = desc.get_video_properties();
    _video_count = _video_prop.videos_count;
    _video_file_names.resize(_video_count);
    _start_end_frame.resize(_video_count);
    _video_file_names = _video_prop.video_file_names;
    _sequence_length = desc.get_sequence_length();
    _step = desc.get_frame_step();
    _stride = desc.get_frame_stride();
    _video_frame_count = _video_prop.frames_count;
    _start_end_frame = _video_prop.start_end_frame_num;
    _batch_count = desc.get_batch_size();
    _total_sequences_count = 0;
    ret = create_sequence_info();

    // the following code is required to make every shard the same size:: required for multi-gpu training
    if (_shard_count > 1 && _batch_count > 1) {
        int _num_batches = _sequences.size() / _batch_count;
        int max_batches_per_shard = (_sequence_count_all_shards + _shard_count - 1) / _shard_count;
        max_batches_per_shard = (max_batches_per_shard + _batch_count - 1) / _batch_count;
        if (_num_batches < max_batches_per_shard) {
            replicate_last_batch_to_pad_partial_shard();
        }
    }
    // shuffle dataset if set
    if (ret == VideoReader::Status::OK && _shuffle)
        std::random_shuffle(_sequences.begin(), _sequences.end());

    return ret;
}

void VideoFileSourceReader::incremenet_read_ptr() {
    _read_counter++;
    _curr_sequence_idx = (_curr_sequence_idx + 1) % _sequences.size();
}

SequenceInfo VideoFileSourceReader::get_sequence_info() {
    auto current_sequence = _sequences[_curr_sequence_idx];
    auto file_path = current_sequence.video_file_name;
    _last_id = file_path;
    incremenet_read_ptr();
    return current_sequence;
}

VideoFileSourceReader::~VideoFileSourceReader() {
}

void VideoFileSourceReader::reset() {
    if (_shuffle)
        std::random_shuffle(_sequences.begin(), _sequences.end());
    _read_counter = 0;
    _curr_sequence_idx = 0;
}

VideoReader::Status VideoFileSourceReader::create_sequence_info() {
    VideoReader::Status status = VideoReader::Status::OK;
    for (size_t i = 0; i < _video_count; i++) {
        unsigned start = std::get<0>(_start_end_frame[i]);
        size_t max_sequence_frames = (_sequence_length - 1) * _stride;
        for (size_t sequence_start = start; (sequence_start + max_sequence_frames) < (start + _video_frame_count[i]); sequence_start += _step) {
            if (get_sequence_shard_id() != _shard_id) {
                _sequence_count_all_shards++;
                incremenet_sequence_id();
                continue;
            }
            _in_batch_read_count++;
            _in_batch_read_count = (_in_batch_read_count % _batch_count == 0) ? 0 : _in_batch_read_count;
            _sequences.push_back({sequence_start, _video_file_names[i]});
            _last_sequence = _sequences.back();
            _total_sequences_count++;
            _sequence_count_all_shards++;
            incremenet_sequence_id();
        }
    }
    if (_in_batch_read_count > 0 && _in_batch_read_count < _batch_count) {
        replicate_last_sequence_to_fill_last_shard();
        LOG("VideoFileSourceReader ShardID [" + TOSTR(_shard_id) + "] Replicated the last sequence " + TOSTR((_batch_count - _in_batch_read_count)) + " times to fill the last batch")
    }
    if (_sequences.empty())
        WRN("VideoFileSourceReader ShardID [" + TOSTR(_shard_id) + "] Did not load any sequences from " + _folder_path)
    return status;
}

void VideoFileSourceReader::replicate_last_sequence_to_fill_last_shard() {
    for (size_t i = _in_batch_read_count; i < _batch_count; i++) {
        _sequences.push_back(_last_sequence);
        _total_sequences_count++;
    }
}

void VideoFileSourceReader::replicate_last_batch_to_pad_partial_shard() {
    if (_sequences.size() >= _batch_count) {
        for (size_t i = 0; i < _batch_count; i++) {
            _sequences.push_back(_sequences[i - _batch_count]);
            _total_sequences_count++;
        }
    }
}

size_t VideoFileSourceReader::get_sequence_shard_id() {
    if (_batch_count == 0 || _shard_count == 0)
        THROW("Shard (Batch) size cannot be set to 0")
    // return (_sequence_id / (_batch_count)) % _shard_count;
    return _sequence_id % _shard_count;
}
#endif
