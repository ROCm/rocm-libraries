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

#include <fstream>
#include <iostream>
#include <sstream>
#include <tuple>
#ifdef ROCAL_VIDEO
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}
#endif
#include "pipeline/commons.h"

#ifdef ROCAL_VIDEO
typedef struct VideoProperties {
    unsigned width, height, videos_count;
    float frame_rate = 0;
    std::vector<size_t> frames_count;
    std::vector<std::string> video_file_names;
    std::vector<std::tuple<unsigned, unsigned>> start_end_frame_num;
    std::vector<std::tuple<float, float>> start_end_timestamps;
    std::vector<int> labels;
} VideoProperties;

typedef struct Properties {
    unsigned width, height, frames_count, avg_frame_rate_num, avg_frame_rate_den;
} Properties;

void substring_extraction(std::string const &str, const char delim, std::vector<std::string> &out);
void open_video_context(const char *video_file_path, Properties &props);
void get_video_properties_from_txt_file(VideoProperties &video_props, const char *file_path, bool file_list_frame_num);
void find_video_properties(VideoProperties &video_props, const char *source_path, bool file_list_frame_num);
#endif
