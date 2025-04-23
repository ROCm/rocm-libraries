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

#include "meta_data/tf_meta_data_reader.h"

#include <google/protobuf/message_lite.h>
#include <stdint.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>

#include "example.pb.h"
#include "feature.pb.h"

using namespace std;

void TFMetaDataReader::init(const MetaDataConfig &cfg, pMetaDataBatch meta_data_batch) {
    _path = cfg.path();
    _feature_key_map = cfg.feature_key_map();
    _output = meta_data_batch;
    _last_rec = false;
}

bool TFMetaDataReader::exists(const std::string &_image_name) {
    return _map_content.find(_image_name) != _map_content.end();
}

void TFMetaDataReader::add(std::string image_name, int label) {
    pMetaData info = std::make_shared<Label>(label);
    if (exists(image_name)) {
        WRN("Entity with the same name exists")
        return;
    }
    _map_content.insert(std::pair<std::string, std::shared_ptr<Label>>(image_name, info));
}

void TFMetaDataReader::lookup(const std::vector<std::string> &_image_names) {
    if (_image_names.empty()) {
        WRN("No image names passed")
        return;
    }
    if (_image_names.size() != (unsigned)_output->size())
        _output->resize(_image_names.size());

    for (unsigned i = 0; i < _image_names.size(); i++) {
        auto _image_name = _image_names[i];
        auto it = _map_content.find(_image_name);
        if (_map_content.end() == it)
            THROW("ERROR: Given name not present in the map" + _image_name)
        _output->get_labels_batch()[i] = it->second->get_labels();
    }
}

void TFMetaDataReader::print_map_contents() {
    std::cerr << "\nMap contents: \n";
    for (auto &elem : _map_content) {
        std::cerr << "Name :\t " << elem.first << "\t ID:  " << elem.second->get_labels()[0] << std::endl;
    }
}

void TFMetaDataReader::read_record(std::ifstream &file_contents, uint file_size, std::vector<std::string> &_image_name, std::string user_label_key, std::string user_filename_key) {
    // std::cerr << "The user_label_key is " << user_label_key << ", and the user_filename_key is " << user_filename_key << "\n";
    uint length;
    uint64_t data_length;
    uint32_t length_crc, data_crc;

    length = file_contents.tellg();
    file_contents.read((char *)&data_length, sizeof(data_length));
    if (!file_contents)
        THROW("TFMetaDataReader: Error in reading TF records")
    file_contents.read((char *)&length_crc, sizeof(length_crc));
    if (!file_contents)
        THROW("TFMetaDataReader: Error in reading TF records")
    if (length + data_length + 16 == file_size) {
        _last_rec = true;
    }
    char *data = new char[data_length];
    file_contents.read(data, data_length);
    if (!file_contents)
        THROW("TFMetaDataReader: Error in reading TF records")
    rocal::tensorflow::Example single_example;
    single_example.ParseFromArray(data, data_length);
    rocal::tensorflow::Features features = single_example.features();
    // features.PrintDebugString();
    auto feature = features.feature();
    rocal::tensorflow::Feature single_feature;
    std::string fname;
    if (!user_filename_key.empty()) {
        single_feature = feature.at(user_filename_key);
        fname = single_feature.bytes_list().value()[0];
    } else {
        // adding for raw images
        fname = std::to_string(_file_id);
        incremenet_file_id();
    }
    _image_name.push_back(fname);
    uint label;
    single_feature = feature.at(user_label_key);
    label = single_feature.int64_list().value()[0];
    // std::cout << "TFMeta read record <name, label>" << fname << " " << label << std::endl;
    add(fname, label);
    file_contents.read((char *)&data_crc, sizeof(data_crc));
    if (!file_contents)
        THROW("TFMetaDataReader: Error in reading TF records")
    delete[] data;
}

void TFMetaDataReader::read_all(const std::string &path) {
    std::string label_key = "image/class/label";
    std::string filename_key = "image/filename";
    label_key = _feature_key_map.at(label_key);
    filename_key = _feature_key_map.at(filename_key);

    read_files(path);
    for (unsigned i = 0; i < _file_names.size(); i++) {
        std::string fname = path + "/" + _file_names[i];
        uint length;
        std::cerr << "Reading for image classification - file_name:: " << fname << std::endl;
        std::ifstream file_contents(fname.c_str(), std::ios::binary);
        file_contents.seekg(0, std::ifstream::end);
        length = file_contents.tellg();
        file_contents.seekg(0, std::ifstream::beg);
        while (!_last_rec) {
            read_record(file_contents, length, _image_name, label_key, filename_key);
        }
        _last_rec = false;
        file_contents.close();
    }
}

void TFMetaDataReader::release(std::string _image_name) {
    if (!exists(_image_name)) {
        WRN("ERROR: Given not present in the map" + _image_name);
        return;
    }
    _map_content.erase(_image_name);
}

void TFMetaDataReader::release() {
    _map_content.clear();
}

void TFMetaDataReader::read_files(const std::string &_path) {
    if ((_src_dir = opendir(_path.c_str())) == nullptr)
        THROW("ERROR: Failed opening the directory at " + _path);

    while ((_entity = readdir(_src_dir)) != nullptr) {
        if (_entity->d_type != DT_REG)
            continue;

        _file_names.push_back(_entity->d_name);
    }
    if (_file_names.empty())
        WRN("TFMetadataReader: Could not find any file in " + _path)
    closedir(_src_dir);
}

TFMetaDataReader::TFMetaDataReader() {
}
