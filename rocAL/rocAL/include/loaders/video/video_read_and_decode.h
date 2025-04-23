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
#include <dirent.h>
#include <vector>
#include <memory>
#include <iterator>
#include <cstring>
#include <map>
#include <tuple>
#include "pipeline/commons.h"
#include "decoders/video/video_decoder.h"
#include "readers/video/video_reader_factory.h"
#include "pipeline/timing_debug.h"
#include "loaders/loader_module.h"
#include "readers/video/video_properties.h"
#include "readers/video/video_reader.h"
#include "pipeline/filesystem.h"

#ifdef ROCAL_VIDEO
extern "C" {
#include <libavutil/pixdesc.h>
}

class VideoReadAndDecode {
   public:
    VideoReadAndDecode();
    ~VideoReadAndDecode();
    size_t count();
    void reset();
    void create(ReaderConfig reader_config, DecoderConfig decoder_config, int batch_size, int device_id = 0);
    void set_video_process_count(size_t video_count) {
        _video_process_count = (video_count <= _max_video_count) ? video_count : _max_video_count;
    }
    float convert_framenum_to_timestamp(size_t frame_number);
    void decode_sequence(size_t sequence_index);

    //! Loads a decompressed batch of sequence of frames into the buffer indicated by buff
    /// \param buff User's buffer provided to be filled with decoded sequence samples
    /// \param names User's buffer provided to be filled with name of the frames in a decoded sequence
    /// \param max_decoded_width User's buffer maximum width per decoded sequence.
    /// \param max_decoded_height user's buffer maximum height per decoded sequence.
    /// \param roi_width is set by the load() function to the width of the region that decoded frames are located.
    /// \param roi_height  is set by the load() function to the height of the region that decoded frames are located.
    /// \param sequence_start_framenum_vec is set by the load() function. The starting frame number of the sequences will be updated.
    /// \param sequence_frame_timestamps_vec is set by the load() function. The timestamps of each of the frames in the sequences will be updated.
    /// \param output_color_format defines what color format user expects decoder to decode frames into if capable of doing so supported is
    LoaderModuleStatus load(
        unsigned char *buff,
        std::vector<std::string> &names,
        const size_t max_decoded_width,
        const size_t max_decoded_height,
        std::vector<uint32_t> &roi_width,
        std::vector<uint32_t> &roi_height,
        std::vector<uint32_t> &actual_width,
        std::vector<uint32_t> &actual_height,
        std::vector<std::vector<size_t>> &sequence_start_framenum_vec,
        std::vector<std::vector<std::vector<float>>> &sequence_frame_timestamps_vec,
        RocalColorFormat output_color_format);

    //! returns timing info or other status information
    Timing timing();

   private:
    struct video_map {
        int _video_map_idx;
        bool _is_decoder_instance;
    };
    std::vector<std::shared_ptr<VideoDecoder>> _video_decoder;
    std::shared_ptr<VideoReader> _video_reader;
    size_t _max_video_count = 50;
    size_t _video_process_count;
    VideoProperties _video_prop;
    std::vector<std::string> _video_names;
    std::map<std::string, video_map> _video_file_name_map;
    std::vector<unsigned char *> _decompressed_buff_ptrs;
    std::vector<size_t> _actual_decoded_width;
    std::vector<size_t> _actual_decoded_height;
    std::vector<size_t> _sequence_start_frame_num;
    std::vector<std::string> _sequence_video_path;
    std::vector<int> _sequence_video_idx;
    TimingDbg _file_load_time, _decode_time;
    size_t _batch_size;
    size_t _sequence_length;
    size_t _stride;
    size_t _video_count;
    float _frame_rate;
    size_t _max_decoded_width;
    size_t _max_decoded_height;
    size_t _max_decoded_stride;
    AVPixelFormat _out_pix_fmt;
    DecoderConfig _video_decoder_config;
    int _device_id = 0;
};
#endif