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
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "pipeline/commons.h"
#include "pipeline/timing_debug.h"
#include "readers/video/video_reader.h"

#ifdef ROCAL_VIDEO
class VideoFileSourceReader : public VideoReader {
   public:
    //! Looks up the folder which contains the files, amd loads the video sequences
    /*!
     \param desc  User provided descriptor containing the files' path.
    */
    VideoReader::Status initialize(ReaderConfig desc) override;
    //! Reads the next resource item
    SequenceInfo get_sequence_info() override;

    //! Resets the object's state to read from the first file in the folder
    void reset() override;

    //! Returns the name of the latest file opened
    std::string id() override { return _last_id; };

    unsigned count_items() override;

    ~VideoFileSourceReader() override;

    VideoFileSourceReader();

   private:
    std::string _folder_path;
    std::vector<std::string> _video_file_names;
    VideoProperties _video_prop;
    size_t _video_count;
    size_t _total_sequences_count;
    std::vector<size_t> _video_frame_count;
    std::vector<SequenceInfo> _sequences;
    std::vector<std::tuple<unsigned, unsigned>> _start_end_frame;
    SequenceInfo _last_sequence;
    size_t _sequence_length;
    size_t _step;
    size_t _stride;
    unsigned _curr_sequence_idx;
    std::string _last_id;
    size_t _shard_id = 0;
    size_t _shard_count = 1;  // equivalent of batch size
    //!< _batch_count Defines the quantum count of the sequences to be read. It's usually equal to the user's batch size.
    /// The loader will repeat sequences if necessary to be able to have the sequences available in multiples of the load_batch_count,
    /// for instance if there are 10 sequences in the dataset and _batch_count is 3, the loader repeats 2 sequences as if there are 12 sequences available.
    size_t _batch_count = 1;
    size_t _sequence_id = 0;
    size_t _in_batch_read_count = 0;
    bool _loop;
    bool _shuffle;
    int _read_counter = 0;
    //!< _sequence_count_all_shards total_number of sequences to figure out the max_batch_size (usually needed for distributed training).
    size_t _sequence_count_all_shards;
    void incremenet_read_ptr();
    size_t get_sequence_shard_id();
    void incremenet_sequence_id() { _sequence_id++; }
    void replicate_last_sequence_to_fill_last_shard();
    void replicate_last_batch_to_pad_partial_shard();
    VideoReader::Status create_sequence_info();
};
#endif
