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

#include "readers/video/video_properties.h"

#include <cmath>
#include <cassert>
#include <algorithm>
#include "pipeline/filesystem.h"

#ifdef ROCAL_VIDEO
void substring_extraction(std::string const &str, const char delim, std::vector<std::string> &out) {
    size_t start;
    size_t end = 0;
    while ((start = str.find_first_not_of(delim, end)) != std::string::npos) {
        end = str.find(delim, start);
        out.push_back(str.substr(start, end - start));
    }
}

// Opens the context of the Video file to obtain the width, heigh and frame rate info.
void open_video_context(const char *video_file_path, Properties &props) {
    AVFormatContext *pFormatCtx = NULL;
    int videoStream = -1;
    unsigned int i = 0;

    // open video file
    int ret = avformat_open_input(&pFormatCtx, video_file_path, NULL, NULL);
    if (ret != 0) {
        WRN("Unable to open video file: " + STR(video_file_path))
        exit(0);
    }

    // Retrieve stream information
    ret = avformat_find_stream_info(pFormatCtx, NULL);
    assert(ret >= 0);
    for (i = 0; i < pFormatCtx->nb_streams; i++) {
        if (pFormatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO && videoStream < 0) {
            videoStream = i;
        }
    }
    assert(videoStream != -1);

    // Get a pointer to the codec context for the video stream
    props.width = pFormatCtx->streams[videoStream]->codecpar->width;
    props.height = pFormatCtx->streams[videoStream]->codecpar->height;
    props.frames_count = pFormatCtx->streams[videoStream]->nb_frames;
    props.avg_frame_rate_num = pFormatCtx->streams[videoStream]->avg_frame_rate.num;
    props.avg_frame_rate_den = pFormatCtx->streams[videoStream]->avg_frame_rate.den;
    avformat_close_input(&pFormatCtx);
}

void get_video_properties_from_txt_file(VideoProperties &video_props, const char *file_path, bool file_list_frame_num) {
    std::ifstream text_file(file_path);

    if (text_file.good()) {
        Properties props;
        std::string line;
        unsigned max_width = 0;
        unsigned max_height = 0;
        unsigned video_count = 0;
        while (std::getline(text_file, line)) {
            int label;
            std::string video_file_name;
            unsigned start_frame_number = 0;
            unsigned end_frame_number = 0;
            float start_time = 0.0;
            float end_time = 0.0;
            std::istringstream line_ss(line);
            if (!(line_ss >> video_file_name >> label))
                continue;
            open_video_context(video_file_name.c_str(), props);
            if (max_width == props.width || max_width == 0)
                max_width = props.width;
            else
                THROW("The given video files are of different resolution\n")
            if (max_height == props.height || max_height == 0)
                max_height = props.height;
            else
                THROW("The given video files are of different resolution\n")
            if (!file_list_frame_num) {
                line_ss >> start_time >> end_time;
                start_frame_number = static_cast<unsigned int>(std::ceil(start_time * (props.avg_frame_rate_num / (double)props.avg_frame_rate_den)));
                end_frame_number = static_cast<unsigned int>(std::floor(end_time * (props.avg_frame_rate_num / (double)props.avg_frame_rate_den)));
            } else {
                line_ss >> start_frame_number >> end_frame_number;
            }
            end_frame_number = end_frame_number != 0 ? end_frame_number : props.frames_count;
            if ((end_frame_number > props.frames_count) || (start_frame_number >= end_frame_number)) {
                INFO("Invalid start or end time/frame number passed, skipping the file" + video_file_name)
                continue;
            }
            video_file_name = std::to_string(video_count) + "#" + video_file_name;  // Video index is added to each video file name to identify repeated videos files.
            video_props.video_file_names.push_back(video_file_name);
            video_props.labels.push_back(label);
            video_props.start_end_frame_num.push_back(std::make_tuple(start_frame_number, end_frame_number));
            video_props.start_end_timestamps.push_back(std::make_tuple(start_time, end_time));
            video_props.frames_count.push_back(end_frame_number - start_frame_number);
            float video_frame_rate = std::floor(props.avg_frame_rate_num / props.avg_frame_rate_den);
            if (video_props.frame_rate != 0 && video_frame_rate != video_props.frame_rate)
                THROW("Variable frame rate videos cannot be processed")
            video_props.frame_rate = video_frame_rate;
            video_count++;
        }
        video_props.width = max_width;
        video_props.height = max_height;
        video_props.videos_count = video_count;
    } else
        THROW("Can't open the metadata file at " + std::string(file_path))
}

void find_video_properties(VideoProperties &video_props, const char *source_path, bool file_list_frame_num) {
    DIR *_sub_dir;
    struct dirent *_entity;
    std::string video_file_path;
    Properties props;
    unsigned max_width = 0;
    unsigned max_height = 0;
    std::string _full_path = source_path;
    filesys::path pathObj(_full_path);
    if (filesys::exists(pathObj) && filesys::is_regular_file(pathObj))  // Single video file / text file as input
    {
        if (pathObj.has_extension() && pathObj.extension().string() == ".txt") {
            get_video_properties_from_txt_file(video_props, source_path, file_list_frame_num);
        } else {
            // Single Video File Input
            open_video_context(source_path, props);
            video_props.width = props.width;
            video_props.height = props.height;
            video_props.videos_count = 1;
            video_props.frames_count.push_back(props.frames_count);
            float video_frame_rate = std::floor(props.avg_frame_rate_num / props.avg_frame_rate_den);
            if (video_props.frame_rate != 0 && video_frame_rate != video_props.frame_rate)
                THROW("Variable frame rate videos cannot be processed")
            video_props.frame_rate = video_frame_rate;
            video_props.start_end_frame_num.push_back(std::make_tuple(0, (int)props.frames_count));
            video_file_path = std::to_string(0) + "#" + _full_path;  // Video index is added to each video file name to identify repeated videos files.
            video_props.video_file_names.push_back(video_file_path);
        }
    } else if (filesys::exists(pathObj) && filesys::is_directory(pathObj)) {
        std::vector<std::string> video_files;
        unsigned video_count = 0;
        std::vector<std::string> entry_name_list;
        std::string _folder_path = source_path;
        if ((_sub_dir = opendir(_folder_path.c_str())) == nullptr)
            THROW("ERROR: Failed opening the directory at " + _folder_path);
        while ((_entity = readdir(_sub_dir)) != nullptr) {
            std::string entry_name(_entity->d_name);
            if (strcmp(_entity->d_name, ".") == 0 || strcmp(_entity->d_name, "..") == 0)
                continue;
            entry_name_list.push_back(entry_name);
        }
        closedir(_sub_dir);
        std::sort(entry_name_list.begin(), entry_name_list.end());

        for (unsigned dir_count = 0; dir_count < entry_name_list.size(); ++dir_count) {
            std::string subfolder_path = _folder_path + "/" + entry_name_list[dir_count];
            filesys::path pathObj(subfolder_path);
            if (filesys::exists(pathObj) && filesys::is_regular_file(pathObj)) {
                open_video_context(subfolder_path.c_str(), props);
                if (max_width == props.width || max_width == 0)
                    max_width = props.width;
                else
                    THROW("The given video files are of different resolution\n")
                if (max_height == props.height || max_height == 0)
                    max_height = props.height;
                else
                    THROW("The given video files are of different resolution\n")
                video_props.frames_count.push_back(props.frames_count);
                float video_frame_rate = std::floor(props.avg_frame_rate_num / props.avg_frame_rate_den);
                if (video_props.frame_rate != 0 && video_frame_rate != video_props.frame_rate)
                    THROW("Variable frame rate videos cannot be processed")
                video_props.frame_rate = video_frame_rate;
                video_file_path = std::to_string(video_count) + "#" + subfolder_path;  // Video index is added to each video file name to identify repeated videos files.
                video_props.video_file_names.push_back(video_file_path);
                video_props.start_end_frame_num.push_back(std::make_tuple(0, (int)props.frames_count));
                video_count++;
            } else if (filesys::exists(pathObj) && filesys::is_directory(pathObj)) {
                std::string _full_path = subfolder_path;
                if ((_sub_dir = opendir(_full_path.c_str())) == nullptr)
                    THROW("VideoReader ERROR: Failed opening the directory at " + source_path);
                while ((_entity = readdir(_sub_dir)) != nullptr) {
                    std::string entry_name(_entity->d_name);
                    if (strcmp(_entity->d_name, ".") == 0 || strcmp(_entity->d_name, "..") == 0)
                        continue;
                    video_files.push_back(entry_name);
                }
                closedir(_sub_dir);
                std::sort(video_files.begin(), video_files.end());
                for (unsigned i = 0; i < video_files.size(); i++) {
                    std::string file_path = _full_path;
                    file_path.append("/");
                    file_path.append(video_files[i]);
                    _full_path = file_path;

                    open_video_context(_full_path.c_str(), props);
                    if (max_width == props.width || max_width == 0)
                        max_width = props.width;
                    else
                        THROW("The given video files are of different resolution\n")
                    if (max_height == props.height || max_height == 0)
                        max_height = props.height;
                    else
                        THROW("The given video files are of different resolution\n")
                    video_file_path = std::to_string(video_count) + "#" + _full_path;  // Video index is added to each video file name to identify repeated videos files.
                    video_props.video_file_names.push_back(video_file_path);
                    video_props.frames_count.push_back(props.frames_count);
                    float video_frame_rate = std::floor(props.avg_frame_rate_num / props.avg_frame_rate_den);
                    if (video_props.frame_rate != 0 && video_frame_rate != video_props.frame_rate)
                        THROW("Variable frame rate videos cannot be processed")
                    video_props.frame_rate = video_frame_rate;
                    video_props.start_end_frame_num.push_back(std::make_tuple(0, (int)props.frames_count));
                    video_count++;
                    _full_path = subfolder_path;
                }
                video_files.clear();
            }
        }
        video_props.videos_count = video_count;
        video_props.width = max_width;
        video_props.height = max_height;
    }
}
#endif
