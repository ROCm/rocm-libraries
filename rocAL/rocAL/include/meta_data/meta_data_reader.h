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

#include <map>
#include <memory>
#include <string>
#include <set>
#include "meta_data/meta_data.h"

enum class MetaDataReaderType {
    FOLDER_BASED_LABEL_READER = 0,  // Used for imagenet-like dataset
    TEXT_FILE_META_DATA_READER,     // Used when metadata is stored in a text file
    COCO_META_DATA_READER,
    COCO_KEY_POINTS_META_DATA_READER,
    CIFAR10_META_DATA_READER,  // meta_data for cifar10 data which is store as part of bin file
    TF_META_DATA_READER,
    CAFFE_META_DATA_READER,
    CAFFE_DETECTION_META_DATA_READER,
    CAFFE2_META_DATA_READER,
    CAFFE2_DETECTION_META_DATA_READER,
    TF_DETECTION_META_DATA_READER,
    VIDEO_LABEL_READER,
    MXNET_META_DATA_READER,
    WEBDATASET_META_DATA_READER
};

struct MetaDataConfig {
   private:
    MetaDataType _type;
    MetaDataReaderType _reader_type;
    std::string _path;
    std::map<std::string, std::string> _feature_key_map;
    std::string _file_prefix;  // if we want to read only filenames with prefix (needed for cifar10 meta data)
    unsigned _sequence_length;
    unsigned _frame_step;
    unsigned _frame_stride;
    unsigned _out_img_width;
    unsigned _out_img_height;
    bool _avoid_class_remapping;
    bool _aspect_ratio_grouping;
    std::string _index_path;
    MissingComponentsBehaviour _missing_component_behaviour;
    std::vector<std::set<std::string>>_exts;

   public:
    MetaDataConfig(const MetaDataType& type, const MetaDataReaderType& reader_type, const std::string& path, const std::map<std::string, std::string>& feature_key_map = std::map<std::string, std::string>(), const std::string file_prefix = std::string(), const unsigned& sequence_length = 3, const unsigned& frame_step = 3, const unsigned& frame_stride = 1, const std::string index_path = std::string(), const MissingComponentsBehaviour& missing_component_behaviour = MissingComponentsBehaviour::MISSING_COMPONENT_SKIP, const std::vector<std::set<std::string>> &exts  = std::vector<std::set<std::string>>())
        : _type(type), _reader_type(reader_type), _path(path), _feature_key_map(feature_key_map), _file_prefix(file_prefix), _sequence_length(sequence_length), _frame_step(frame_step), _frame_stride(frame_stride), _index_path(index_path), _missing_component_behaviour(missing_component_behaviour), _exts(exts){}
    MetaDataConfig() = delete;
    MetaDataType type() const { return _type; }
    MetaDataReaderType reader_type() const { return _reader_type; }
    std::string path() const { return _path; }
    std::string index_path() const { return _index_path; }
    std::vector<std::set<std::string>> exts() const { return _exts; }
    MissingComponentsBehaviour get_missing_component_behaviour() const { return _missing_component_behaviour; }
    std::map<std::string, std::string> feature_key_map() const { return _feature_key_map; }
    std::string file_prefix() const { return _file_prefix; }
    bool class_remapping() const { return _avoid_class_remapping; }
    bool get_aspect_ratio_grouping() const { return _aspect_ratio_grouping; }
    unsigned sequence_length() const { return _sequence_length; }
    unsigned frame_step() const { return _frame_step; }
    unsigned frame_stride() const { return _frame_stride; }
    unsigned out_img_width() const { return _out_img_width; }
    unsigned out_img_height() const { return _out_img_height; }
    void set_out_img_width(unsigned out_img_width) { _out_img_width = out_img_width; }
    void set_out_img_height(unsigned out_img_height) { _out_img_height = out_img_height; }
    void set_avoid_class_remapping(bool avoid_class_remapping) { _avoid_class_remapping = avoid_class_remapping; }
    void set_aspect_ratio_grouping(bool aspect_ratio_grouping) { _aspect_ratio_grouping = aspect_ratio_grouping; }
};

class MetaDataReader {
   protected:
    bool _aspect_ratio_grouping;

   public:
    enum class Status {
        OK = 0
    };
    virtual ~MetaDataReader() = default;
    virtual void init(const MetaDataConfig& cfg, pMetaDataBatch meta_data_batch) = 0;
    virtual void read_all(const std::string& path) = 0;                    // Reads all the meta data information
    virtual void lookup(const std::vector<std::string>& image_names) = 0;  // finds meta_data info associated with given names and fills the output
    virtual void release() = 0;                                            // Deletes the loaded information
    virtual const std::map<std::string, std::shared_ptr<MetaData>>& get_map_content() = 0;
    virtual bool exists(const std::string& image_name) = 0;
    virtual bool set_timestamp_mode() = 0;
    virtual ImgSize lookup_image_size(const std::string& image_name) { return {}; }
    virtual void set_aspect_ratio_grouping(bool aspect_ratio_grouping) { return; }
    virtual bool get_aspect_ratio_grouping() const { return {}; }
    virtual std::vector<std::string> get_relative_file_path() { return {}; } // Returns the relative file_path's of the reader 
};
