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

#include <map>

#include "pipeline/commons.h"
#include "meta_data/meta_data.h"
#include "meta_data/meta_data_reader.h"
#include "readers/video/video_properties.h"

#ifdef ROCAL_VIDEO
class VideoLabelReader : public MetaDataReader {
   public:
    void init(const MetaDataConfig &cfg, pMetaDataBatch meta_data_batch) override;
    void lookup(const std::vector<std::string> &frame_names) override;
    void read_all(const std::string &path) override;
    void release(std::string frame_name);
    void release() override;
    bool set_timestamp_mode() override {
        _file_list_frame_num = false;
        return _file_list_frame_num;
    }
    void print_map_contents();
    const std::map<std::string, std::shared_ptr<MetaData>> &get_map_content() override { return _map_content; }
    VideoLabelReader();

   private:
    void read_files(const std::string &_path);
    void read_text_file(const std::string &_path);
    bool exists(const std::string &frame_name) override;
    void add(std::string frame_name, int label, unsigned int video_frame_count = 0, unsigned int start_frame = 0);
    std::map<std::string, std::shared_ptr<MetaData>> _map_content;
    std::map<std::string, std::shared_ptr<MetaData>>::iterator _itr;
    std::string _path;
    pMetaDataBatch _output;
    DIR *_src_dir, *_sub_dir;
    struct dirent *_entity;
    std::vector<std::string> _file_names;
    std::vector<std::string> _subfolder_video_file_names;
    int _video_idx = 0;
    bool _file_list_frame_num = true;
    unsigned _sequence_length;
    unsigned _step;
    unsigned _stride;
};
#endif
