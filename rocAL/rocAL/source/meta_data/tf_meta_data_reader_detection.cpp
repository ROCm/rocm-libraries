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

#include "meta_data/tf_meta_data_reader_detection.h"

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

void TFMetaDataReaderDetection::init(const MetaDataConfig &cfg, pMetaDataBatch meta_data_batch) {
    _path = cfg.path();
    _feature_key_map = cfg.feature_key_map();
    _output = meta_data_batch;
    _last_rec = false;
}

bool TFMetaDataReaderDetection::exists(const std::string &_image_name) {
    return _map_content.find(_image_name) != _map_content.end();
}

void TFMetaDataReaderDetection::add(std::string image_name, BoundingBoxCords bb_coords, Labels bb_labels, ImgSize image_size) {
    if (exists(image_name)) {
        auto it = _map_content.find(image_name);
        it->second->get_bb_cords().push_back(bb_coords[0]);
        it->second->get_labels().push_back(bb_labels[0]);
        return;
    }
    pMetaDataBox info = std::make_shared<BoundingBox>(bb_coords, bb_labels, image_size);
    _map_content.insert(pair<std::string, std::shared_ptr<BoundingBox>>(image_name, info));
}

void TFMetaDataReaderDetection::lookup(const std::vector<std::string> &image_names) {
    if (image_names.empty()) {
        WRN("No image names passed")
        return;
    }
    if (image_names.size() != (unsigned)_output->size())
        _output->resize(image_names.size());

    for (unsigned i = 0; i < image_names.size(); i++) {
        auto image_name = image_names[i];
        auto it = _map_content.find(image_name);

        if (_map_content.end() == it)
            THROW("ERROR: Given name not present in the map" + image_name)
        _output->get_bb_cords_batch()[i] = it->second->get_bb_cords();
        _output->get_labels_batch()[i] = it->second->get_labels();
        _output->get_img_sizes_batch()[i] = it->second->get_img_size();
    }
}

void TFMetaDataReaderDetection::print_map_contents() {
    BoundingBoxCords bb_coords;
    Labels bb_labels;

    std::cerr << "\nMap contents: \n";
    for (auto &elem : _map_content) {
        std::cerr << "Name :\t " << elem.first;
        bb_coords = elem.second->get_bb_cords();
        bb_labels = elem.second->get_labels();
        std::cerr << "\nsize of the element  : " << bb_coords.size() << std::endl;
        for (unsigned int i = 0; i < bb_coords.size(); i++) {
            std::cerr << " l : " << bb_coords[i].l << " t: :" << bb_coords[i].t << " r : " << bb_coords[i].r << " b: :" << bb_coords[i].b << std::endl;
            std::cerr << "Label Id : " << bb_labels[i] << std::endl;
        }
    }
}

void TFMetaDataReaderDetection::read_record(std::ifstream &file_contents, uint file_size, std::vector<std::string> &_image_name,
                                            std::string user_label_key, std::string user_text_key,
                                            std::string user_xmin_key, std::string user_ymin_key, std::string user_xmax_key, std::string user_ymax_key,
                                            std::string user_filename_key) {
    uint length;
    length = file_contents.tellg();

    std::string temp;
    size_t uint64_size, uint32_size;
    uint64_t data_length;
    uint32_t length_crc, data_crc;
    uint64_size = sizeof(uint64_t);
    uint32_size = sizeof(uint32_t);
    char *header_length = new char[uint64_size];
    char *header_crc = new char[uint32_size];
    char *footer_crc = new char[uint32_size];
    file_contents.read(header_length, uint64_size);
    if (!file_contents)
        THROW("TFMetaDataReaderDetection: Error in reading TF records")
    file_contents.read(header_crc, uint32_size);
    if (!file_contents)
        THROW("TFMetaDataReaderDetection: Error in reading TF records")
    memcpy(&data_length, header_length, sizeof(data_length));
    memcpy(&length_crc, header_crc, sizeof(length_crc));

    if (length + data_length + 16 == file_size) {
        _last_rec = true;
    }
    char *data = new char[data_length];
    file_contents.read(data, data_length);
    if (!file_contents)
        THROW("TFMetaDataReaderDetection: Error in reading TF records")
    rocal::tensorflow::Example single_example;
    single_example.ParseFromArray(data, data_length);
    rocal::tensorflow::Features features = single_example.features();

    auto feature = features.feature();
    rocal::tensorflow::Feature single_feature, sf_xmin, sf_ymin, sf_xmax, sf_ymax, sf_fname, sf_label, sf_height, sf_width;

    single_feature = feature.at(user_filename_key);
    std::string fname = single_feature.bytes_list().value()[0];
    float size_b_xmin;
    single_feature = feature.at(user_xmin_key);
    size_b_xmin = single_feature.float_list().value().size();

    BoundingBoxCords bb_coords;
    Labels bb_labels;
    BoundingBoxCord box;

    sf_label = feature.at(user_label_key);
    sf_xmin = feature.at(user_xmin_key);
    sf_ymin = feature.at(user_ymin_key);
    sf_xmax = feature.at(user_xmax_key);
    sf_ymax = feature.at(user_ymax_key);

    sf_height = feature.at("image/height");
    sf_width = feature.at("image/width");

    int image_height, image_width;
    image_height = sf_height.int64_list().value()[0];
    image_width = sf_width.int64_list().value()[0];

    ImgSize img_size;
    img_size.w = image_width;
    img_size.h = image_height;

    for (int i = 0; i < size_b_xmin; i++) {
        int label;
        label = sf_label.int64_list().value()[i];
        box.l = sf_xmin.float_list().value()[i] * image_width;
        box.t = sf_ymin.float_list().value()[i] * image_height;
        box.r = sf_xmax.float_list().value()[i] * image_width;
        box.b = sf_ymax.float_list().value()[i] * image_height;
        bb_coords.push_back(box);
        bb_labels.push_back(label);
        add(fname, bb_coords, bb_labels, img_size);
        bb_coords.clear();
        bb_labels.clear();
    }
    file_contents.read(footer_crc, sizeof(data_crc));
    if (!file_contents)
        THROW("TFMetaDataReaderDetection: Error in reading TF records")
    memcpy(&data_crc, footer_crc, sizeof(data_crc));
    delete[] header_length;
    delete[] header_crc;
    delete[] footer_crc;
    delete[] data;
}

void TFMetaDataReaderDetection::read_all(const std::string &path) {
    std::string label_key = "image/class/label";
    std::string text_key = "image/class/text";
    std::string xmin_key = "image/object/bbox/xmin";
    std::string ymin_key = "image/object/bbox/ymin";
    std::string xmax_key = "image/object/bbox/xmax";
    std::string ymax_key = "image/object/bbox/ymax";
    std::string filename_key = "image/filename";
    label_key = _feature_key_map.at(label_key);
    text_key = _feature_key_map.at(text_key);
    xmin_key = _feature_key_map.at(xmin_key);
    ymin_key = _feature_key_map.at(ymin_key);
    xmax_key = _feature_key_map.at(xmax_key);
    ymax_key = _feature_key_map.at(ymax_key);
    filename_key = _feature_key_map.at(filename_key);

    read_files(path);
    for (unsigned i = 0; i < _file_names.size(); i++) {
        std::string fname = path + _file_names[i];
        uint length;
        std::cerr << "Reading for object detection - file_name:: " << fname << std::endl;
        std::ifstream file_contents(fname.c_str(), std::ios::binary);
        file_contents.seekg(0, std::ifstream::end);
        length = file_contents.tellg();
        file_contents.seekg(0, std::ifstream::beg);
        while (!_last_rec) {
            read_record(file_contents, length, _image_name, label_key, text_key, xmin_key, ymin_key, xmax_key, ymax_key, filename_key);
        }
        _last_rec = false;
        file_contents.close();
    }
    // google::protobuf::ShutdownProtobufLibrary();
    // print_map_contents();
}

void TFMetaDataReaderDetection::release(std::string _image_name) {
    if (!exists(_image_name)) {
        WRN("ERROR: Given not present in the map" + _image_name);
        return;
    }
    _map_content.erase(_image_name);
}

void TFMetaDataReaderDetection::release() {
    _map_content.clear();
}

void TFMetaDataReaderDetection::read_files(const std::string &_path) {
    if ((_src_dir = opendir(_path.c_str())) == nullptr)
        THROW("ERROR: Failed opening the directory at " + _path);

    while ((_entity = readdir(_src_dir)) != nullptr) {
        if (_entity->d_type != DT_REG)
            continue;

        _file_names.push_back(_entity->d_name);
    }
    if (_file_names.empty())
        WRN("LabelReader: Could not find any file in " + _path)
    closedir(_src_dir);
}

TFMetaDataReaderDetection::TFMetaDataReaderDetection() {
}